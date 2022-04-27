use std::{
    fs,
    io::{self, Seek},
    path::Path,
};

use angsd_io::{saf, ReadStatus};

mod header;
pub use header::{Header, HeaderError, MAGIC_NUMBER};

pub trait ReadSite {
    fn read_site<T>(&mut self, bufs: &mut [T]) -> io::Result<ReadStatus>
    where
        T: AsMut<[f32]>;
}

impl<R> ReadSite for Reader<R>
where
    R: io::BufRead,
{
    fn read_site<T>(&mut self, bufs: &mut [T]) -> io::Result<ReadStatus>
    where
        T: AsMut<[f32]>,
    {
        let status = self.read_values(bufs)?;

        bufs.iter_mut()
            .for_each(|buf| buf.as_mut().iter_mut().for_each(|x| *x = x.exp()));

        Ok(status)
    }
}

impl<'a, R> ReadSite for Take<'a, R>
where
    R: io::BufRead,
{
    fn read_site<T>(&mut self, bufs: &mut [T]) -> io::Result<ReadStatus>
    where
        T: AsMut<[f32]>,
    {
        if self.current < self.max {
            let status = self.inner.read_site(bufs)?;
            self.current += 1;
            Ok(status)
        } else {
            Ok(ReadStatus::Done)
        }
    }
}

pub struct Take<'a, R> {
    inner: &'a mut Reader<R>,
    current: usize,
    max: usize,
}

impl<'a, R> Take<'a, R> {
    pub fn current(&self) -> usize {
        self.current
    }
}

pub struct Reader<R> {
    inner: saf::reader::ValueReader<R>,
}

impl<R> Reader<R>
where
    R: io::BufRead,
{
    pub fn get(&self) -> &saf::reader::ValueReader<R> {
        &self.inner
    }

    pub fn get_mut(&mut self) -> &mut saf::reader::ValueReader<R> {
        &mut self.inner
    }

    pub fn into_inner(self) -> saf::reader::ValueReader<R> {
        self.inner
    }

    pub fn is_done(&mut self) -> io::Result<bool> {
        // TODO: This can use io::BufRead::has_data_left once stable,
        // see github.com/rust-lang/rust/issues/86423
        self.inner.get_mut().fill_buf().map(|b| b.is_empty())
    }

    pub fn new(inner: saf::reader::ValueReader<R>) -> Self {
        Self { inner }
    }

    pub fn read_header(&mut self) -> io::Result<Header> {
        Header::read(self.inner.get_mut())
    }

    pub fn read_values<T>(&mut self, bufs: &mut [T]) -> io::Result<ReadStatus>
    where
        T: AsMut<[f32]>,
    {
        for (i, buf) in bufs.iter_mut().enumerate() {
            if self.inner.read_values(buf.as_mut())?.is_done() {
                return if i == 0 {
                    Ok(ReadStatus::Done)
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "could not fill all value buffers before end of file",
                    ))
                };
            }
        }

        Ok(ReadStatus::NotDone)
    }

    pub fn rewind(&mut self, header: &Header) -> io::Result<()>
    where
        R: io::Seek,
    {
        let offset = header.header_size();

        self.inner
            .get_mut()
            .seek(io::SeekFrom::Start(offset as u64))?;

        Ok(())
    }

    pub fn take(&mut self, sites: usize) -> Take<R> {
        Take {
            inner: self,
            current: 0,
            max: sites,
        }
    }
}

impl Reader<io::BufReader<fs::File>> {
    pub fn from_path<P>(path: P) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        fs::File::open(path)
            .map(io::BufReader::new)
            .map(saf::reader::ValueReader::new)
            .map(Reader::new)
    }
}

pub struct Writers<W> {
    inner: Vec<saf::writer::ValueWriter<W>>,
}

impl<W> Writers<W> {
    pub fn blocks(&self) -> usize {
        self.inner.len()
    }

    pub fn new(inner: Vec<saf::writer::ValueWriter<W>>) -> Self {
        Self { inner }
    }

    pub fn get(&self) -> &[saf::writer::ValueWriter<W>] {
        &self.inner
    }

    pub fn get_mut(&mut self) -> &mut [saf::writer::ValueWriter<W>] {
        &mut self.inner
    }
}

impl Writers<io::BufWriter<fs::File>> {
    pub fn create<P>(path: P, header: &Header) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        let header_size = header.header_size();
        let data_size = header.data_size();
        let file_size = header.total_size();

        let mut f = allocate_file(&path, file_size as u64)?;
        header.write(&mut f)?;

        let blocks = header.blocks();
        let block_size = if data_size % blocks as usize == 0 {
            data_size / blocks as usize
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "size of data is not multiple of blocks",
            ));
        };

        let writers = (0..blocks as usize)
            .map(|block_index| header_size + block_index * block_size)
            .map(|offset| open_writer_at_offset(&path, offset as u64))
            .collect::<io::Result<Vec<_>>>()?;

        Ok(Self::new(writers))
    }
}

fn allocate_file<P>(path: P, bytes: u64) -> io::Result<fs::File>
where
    P: AsRef<Path>,
{
    let f = fs::File::create(&path)?;
    f.set_len(bytes)?;
    Ok(f)
}

fn open_file_at_offset<P>(path: P, offset: u64) -> io::Result<fs::File>
where
    P: AsRef<Path>,
{
    let mut f = fs::File::options().write(true).open(&path)?;
    f.seek(io::SeekFrom::Start(offset))?;
    Ok(f)
}

fn open_writer_at_offset<P>(
    path: P,
    offset: u64,
) -> io::Result<saf::writer::ValueWriter<io::BufWriter<fs::File>>>
where
    P: AsRef<Path>,
{
    open_file_at_offset(&path, offset)
        .map(io::BufWriter::new)
        .map(saf::writer::ValueWriter::new)
}
