use std::{
    fs::File,
    io::{self, Seek},
    path::Path,
};

use angsd_io::saf::reader::ValueReader;

use crate::io::{ReadSite, ReadStatus, Rewind};

use super::{to_u64, Header};

/// A pseudo-shuffled SAF file reader.
pub struct Reader<R> {
    inner: ValueReader<R>,
    header: Header,
}

/// A pseudo-shuffled SAF file reader.
impl<R> Reader<R>
where
    R: io::BufRead,
{
    /// Returns the inner reader.
    pub fn get(&self) -> &ValueReader<R> {
        &self.inner
    }

    /// Returns a mutable reference to the the inner reader.
    pub fn get_mut(&mut self) -> &mut ValueReader<R> {
        &mut self.inner
    }

    /// Returns the header of the reader.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Returns the inner reader, consuming `self`.
    pub fn into_inner(self) -> ValueReader<R> {
        self.inner
    }

    /// Creates a new reader.
    ///
    /// The stream is expected to be positioned after the header.
    pub fn new(reader: R, header: Header) -> Self {
        let inner = ValueReader::new(reader);

        Self { inner, header }
    }

    /// Reads the header from the reader.
    ///
    /// The stream is assumed to be positioned at the beginning.
    pub fn read_header(&mut self) -> io::Result<Header> {
        Header::read(self.inner.get_mut())
    }
}

impl<R> io::Seek for Reader<R>
where
    R: io::BufRead + io::Seek,
{
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        self.inner.get_mut().seek(pos)
    }
}

impl Reader<io::BufReader<File>> {
    /// Creates a new reader from a path, and read its header.
    ///
    /// The stream will be positioned after the header.
    pub fn from_path<P>(path: P) -> io::Result<Self>
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
        self.inner.get_mut().fill_buf().map(|b| b.is_empty())
    }

    fn rewind(&mut self) -> io::Result<()> {
        self.seek(io::SeekFrom::Start(to_u64(self.header.header_size())))
            .map(|_| ())
    }
}

impl<R> ReadSite for Reader<R>
where
    R: io::BufRead + io::Seek,
{
    fn read_site(&mut self, buf: &mut [f32]) -> io::Result<ReadStatus> {
        let status = self.inner.read_values(buf);

        buf.iter_mut().for_each(|x| *x = x.exp());

        status
    }
}
