use std::{
    fs::File,
    io::{self, Seek, Write},
    path::Path,
    thread::panicking,
};

use angsd_saf::version::Version;

use super::{to_u64, to_usize, Header};

use crate::{
    em::StreamEmSite,
    io::{Intersect, ReadSite},
    saf::Site,
};

/// A pseudo-shuffled SAF file writer.
///
/// Note that the writer has a fallible drop check.
/// See [`Writer::create`] and [`Writer::try_finish`] for more, as well as
/// the [module docs](index.html#write) for general usage..
pub struct Writer<W> {
    writers: Vec<W>,
    header: Header,
    current: usize,
    finish_flag: bool, // Flag used for drop check
}

impl<W> Writer<W> {
    /// Check if reader is finished.
    fn is_finished(&self) -> bool {
        self.current >= to_usize(self.header.sites())
    }

    /// Creates a new writer.
    fn new(writers: Vec<W>, header: Header) -> Self {
        let finish_flag = header.sites() == 0;

        Self {
            writers,
            header,
            current: 0,
            finish_flag,
        }
    }

    /// Fallible drop check, used in both the actual Drop impl and try_finish.
    fn try_drop(&mut self) -> io::Result<()> {
        if self.is_finished() | self.finish_flag {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "closing pseudo-shuffled SAF file writer before it was filled",
            ))
        }
    }

    /// Consumes the writer fallibly.
    ///
    /// This can be use to drop the writer and handle an error in the drop. See [`Writer::create`]
    /// for more information on when the writer can be dropped.
    pub fn try_finish(mut self) -> io::Result<()> {
        let result = self.try_drop();
        // Set the flag here so that drop check doesn't panic now
        self.finish_flag = true;
        result
    }
}

impl Writer<io::BufWriter<File>> {
    /// Creates a new pseudo-shuffled SAF file writer.
    ///
    /// Note that this will pre-allocate the full disk space needed to fit the data described in
    /// the header. If the path already exists, it will be overwritten. The header information will
    /// be written to the file.
    ///
    /// Since the full file space is pre-allocated, and since data is not written sequentially,
    /// it is considered an error if less sites are written than specified in the `header`.
    /// This condition is checked when dropping the reader, and the drop check will panic if the
    /// check is failed. See [`Writer::try_finish`] to handle the result of this check.
    pub fn create<P>(path: P, header: Header) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        let file_size = header.file_size();

        let mut f = File::create(&path)?;
        f.set_len(to_u64(file_size))?;
        header.write(&mut f)?;

        let writers = header
            .block_offsets()
            .map(|offset| open_writer_at_offset(&path, to_u64(offset)))
            .collect::<io::Result<Vec<_>>>()?;

        Ok(Self::new(writers, header))
    }

    /// Writes an entire reader to the writer.
    ///
    /// Assumes that the reader contains the appropriate number of sites.
    pub fn write_intersect<const D: usize, R, V>(
        mut self,
        mut intersect: Intersect<D, R, V>,
    ) -> io::Result<()>
    where
        Intersect<D, R, V>: ReadSite<Site = Site<D>>,
        R: io::BufRead + io::Seek,
        V: Version,
    {
        let shape = intersect
            .get()
            .get_readers()
            .iter()
            .map(|reader| reader.index().alleles() + 1)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let mut site = Site::from_shape(shape);

        while intersect.read_site_unnormalised(&mut site)?.is_not_done() {
            self.write_site(site.as_slice())?
        }

        self.try_finish()
    }

    /// Writes a single site to the writer.
    ///
    /// No more sites can be written than specified in the header specified to [`Writer::create`].
    /// Also, the number of values in `site` must match the sum of the shape provided in the header.
    /// If either of those conditions are not met, an error will be returned.
    pub fn write_site(&mut self, values: &[f32]) -> io::Result<()> {
        if self.is_finished() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "attempted to write more sites to writer than allocated",
            ));
        } else if values.len() != self.header.width() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "number of values provided to writer does not match provided shape",
            ));
        }

        let next_idx = self.current % self.writers.len();
        let writer = &mut self.writers[next_idx];
        for v in values {
            writer.write_all(&v.to_le_bytes())?;
        }

        self.current += 1;

        Ok(())
    }

    /// Writes a single site split across multiple slices to the writer.
    ///
    /// The different slices here may for instance correspond to different populations. As for
    /// [`Writer::write_site`], no more sites can be than specified in the header specified to
    /// [`Writer::create`]. The provided sites must match the shape provided in the header.
    /// If either of those conditions are not met, an error will be returned.
    pub fn write_disjoint_site<I>(&mut self, values_iter: I) -> io::Result<()>
    where
        I: IntoIterator,
        I::Item: AsRef<[f32]>,
        I::IntoIter: ExactSizeIterator,
    {
        let values_iter = values_iter.into_iter();
        let shape = self.header.shape();

        if self.is_finished() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "attempted to write more sites to writer than allocated",
            ));
        } else if values_iter.len() != shape.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "more value slices provided for writing than shapes provided in header",
            ));
        }

        let next_idx = self.current % self.writers.len();
        let writer = &mut self.writers[next_idx];

        for (values, &shape) in values_iter.zip(shape) {
            if values.as_ref().len() != shape {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "provided values does not fit corresponding header shape",
                ));
            }

            for v in values.as_ref() {
                writer.write_all(&v.to_le_bytes())?
            }
        }

        self.current += 1;

        Ok(())
    }
}

impl<W> Drop for Writer<W> {
    fn drop(&mut self) {
        // Don't check if writer is finished if already unwinding from panic,
        // or we will likely get a double panic
        if !panicking() {
            self.try_drop().unwrap()
        }
    }
}

/// Opens path for writing without truncating and creates a writer positioned at byte offset.
fn open_writer_at_offset<P>(path: P, offset: u64) -> io::Result<io::BufWriter<File>>
where
    P: AsRef<Path>,
{
    let mut f = File::options().write(true).open(&path)?;
    f.seek(io::SeekFrom::Start(offset))?;

    Ok(io::BufWriter::new(f))
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::{
        io::{Read, SeekFrom},
        mem::size_of,
    };

    use tempfile::NamedTempFile;

    #[test]
    fn test_writer_too_few_sites_error() -> io::Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path();

        let header = Header::new(514, vec![15, 7], 20);
        let writer = Writer::create(path, header)?;

        assert_eq!(
            writer.try_finish().unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );

        file.close()
    }

    #[test]
    fn test_writer_too_many_sites_error() -> io::Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path();

        let header = Header::new(2, vec![1, 2], 2);
        let mut writer = Writer::create(path, header.clone())?;

        let values = vec![0.0; header.width()];
        writer.write_site(values.as_slice())?;
        writer.write_site(values.as_slice())?;

        let result = writer.write_site(values.as_slice());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);

        file.close()
    }

    #[test]
    fn test_create_writer() -> io::Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path();

        let header = Header::new(514, vec![15, 7], 20);
        let mut writer = Writer::create(path, header.clone())?;

        assert_eq!(
            file.as_file().metadata()?.len() as usize,
            header.file_size(),
        );

        let initial_offsets = writer
            .writers
            .iter_mut()
            .map(|writer| writer.get_mut().stream_position().map(to_usize))
            .collect::<io::Result<Vec<_>>>()?;
        let expected_offsets = header.block_offsets().collect::<Vec<_>>();
        assert_eq!(initial_offsets, expected_offsets);

        let _error = writer.try_finish();
        file.close()
    }

    // Helper for testing that writing the provided sites with the given header produced
    // the expected data when read back in
    fn test_shuffled<I, F>(
        header: Header,
        sites: I,
        expected: &[f32],
        mut write_fn: F,
    ) -> io::Result<()>
    where
        I: IntoIterator,
        F: FnMut(&mut Writer<io::BufWriter<File>>, I::Item) -> io::Result<()>,
    {
        let mut file = NamedTempFile::new()?;
        let path = file.path();

        let mut writer = Writer::create(path, header.clone())?;

        for site in sites {
            write_fn(&mut writer, site)?;
        }

        // Drop the writer to flush
        writer.try_finish().unwrap();

        let mut data = Vec::new();
        file.seek(SeekFrom::Start(header.header_size() as u64))?;
        file.read_to_end(&mut data)?;

        let written: Vec<f32> = data
            .chunks(size_of::<f32>())
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();

        assert_eq!(written, expected);

        file.close()
    }

    #[test]
    fn test_writer_shuffle() -> io::Result<()> {
        let header = Header::new(10, vec![1, 2], 4);

        let sites = vec![
            &[0., 0., 0.],
            &[1., 1., 1.],
            &[2., 2., 2.],
            &[3., 3., 3.],
            &[4., 4., 4.],
            &[5., 5., 5.],
            &[6., 6., 6.],
            &[7., 7., 7.],
            &[8., 8., 8.],
            &[9., 9., 9.],
        ];

        #[rustfmt::skip]
        let expected = vec![
            0., 0., 0.,
            4., 4., 4.,
            8., 8., 8.,
            1., 1., 1.,
            5., 5., 5.,
            9., 9., 9.,
            2., 2., 2.,
            6., 6., 6.,
            3., 3., 3.,
            7., 7., 7.,
        ];

        test_shuffled(header, sites, expected.as_slice(), |writer, site| {
            writer.write_site(site)
        })
    }

    #[test]
    fn test_writer_disjoint_shuffle() -> io::Result<()> {
        let header = Header::new(10, vec![1, 2], 4);

        let sites = vec![
            vec![&[0.][..], &[0., 0.][..]],
            vec![&[1.][..], &[1., 1.][..]],
            vec![&[2.][..], &[2., 2.][..]],
            vec![&[3.][..], &[3., 3.][..]],
            vec![&[4.][..], &[4., 4.][..]],
            vec![&[5.][..], &[5., 5.][..]],
            vec![&[6.][..], &[6., 6.][..]],
            vec![&[7.][..], &[7., 7.][..]],
            vec![&[8.][..], &[8., 8.][..]],
            vec![&[9.][..], &[9., 9.][..]],
        ];

        #[rustfmt::skip]
        let expected = vec![
            0., 0., 0.,
            4., 4., 4.,
            8., 8., 8.,
            1., 1., 1.,
            5., 5., 5.,
            9., 9., 9.,
            2., 2., 2.,
            6., 6., 6.,
            3., 3., 3.,
            7., 7., 7.,
        ];

        test_shuffled(header, sites, expected.as_slice(), |writer, site| {
            writer.write_disjoint_site(site)
        })
    }
}
