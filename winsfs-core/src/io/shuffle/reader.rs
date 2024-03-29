use std::{
    fs::File,
    io::{self, Seek},
    path::Path,
};

use byteorder::{ReadBytesExt, LE};

use crate::{
    em::Sites,
    io::{ReadSite, ReadStatus, Rewind},
    saf::Site,
};

use super::{to_u64, Header};

/// A pseudo-shuffled SAF file reader.
pub struct Reader<R> {
    inner: R,
    header: Header,
}

/// A pseudo-shuffled SAF file reader.
impl<R> Reader<R>
where
    R: io::BufRead,
{
    /// Returns the inner reader.
    pub fn get(&self) -> &R {
        &self.inner
    }

    /// Returns a mutable reference to the the inner reader.
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.inner
    }

    /// Returns the header of the reader.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Returns the inner reader, consuming `self`.
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// Creates a new reader.
    ///
    /// The stream is expected to be positioned after the header.
    pub fn new(reader: R, header: Header) -> Self {
        Self {
            inner: reader,
            header,
        }
    }

    /// Reads the header from the reader.
    ///
    /// The stream is assumed to be positioned at the beginning.
    pub fn read_header(&mut self) -> io::Result<Header> {
        Header::read(&mut self.inner)
    }
}

impl<R> io::Seek for Reader<R>
where
    R: io::BufRead + io::Seek,
{
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        self.inner.seek(pos)
    }
}

impl Reader<io::BufReader<File>> {
    /// Creates a new reader from a path, and read its header.
    ///
    /// Returns an error if the dimensionality defined in the header is not `D`.
    ///
    /// The stream will be positioned after the header.
    pub fn try_from_path<P>(path: P) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        let mut reader = File::open(path).map(io::BufReader::new)?;
        let header = Header::read(&mut reader)?;
        Ok(Self::new(reader, header))
    }
}

impl<R> Rewind for Reader<R>
where
    R: io::BufRead + io::Seek,
{
    fn is_done(&mut self) -> io::Result<bool> {
        // TODO: This can use io::BufRead::has_data_left once stable,
        // see github.com/rust-lang/rust/issues/86423
        self.inner.fill_buf().map(|b| b.is_empty())
    }

    fn rewind(&mut self) -> io::Result<()> {
        self.seek(io::SeekFrom::Start(to_u64(self.header.header_size())))
            .map(|_| ())
    }
}

impl<R> Sites for Reader<R>
where
    R: io::BufRead,
{
    fn sites(&self) -> usize {
        self.header().sites()
    }
}

impl<R> ReadSite for Reader<R>
where
    R: io::BufRead + io::Seek,
{
    fn read_site<const D: usize>(&mut self, buf: &mut Site<D>) -> io::Result<ReadStatus> {
        let status = self.read_site_unnormalised(buf)?;

        buf.iter_mut().for_each(|x| *x = x.exp());

        Ok(status)
    }

    fn read_site_unnormalised<const D: usize>(
        &mut self,
        buf: &mut Site<D>,
    ) -> io::Result<ReadStatus> {
        // TODO: There's probably a better way to handle this.
        assert_eq!(self.header.shape(), buf.shape());

        if ReadStatus::check(&mut self.inner)?.is_done() {
            return Ok(ReadStatus::Done);
        }

        self.inner.read_f32_into::<LE>(buf.as_mut_slice())?;

        Ok(ReadStatus::NotDone)
    }
}
