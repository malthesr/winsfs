use std::{fmt, io, mem};

pub const MAGIC_NUMBER: &[u8; 8] = b"safvshuf";

#[derive(Clone, Debug)]
pub struct Header {
    sites: u64,
    alleles: Vec<u64>,
    blocks: u64,
}

impl Header {
    pub fn alleles(&self) -> &[u64] {
        &self.alleles
    }

    pub fn blocks(&self) -> u64 {
        self.blocks
    }

    pub fn sites(&self) -> u64 {
        self.sites
    }

    pub fn new(sites: u64, alleles: Vec<u64>, blocks: u64) -> Result<Self, HeaderError> {
        if sites % blocks == 0 {
            Ok(Self {
                sites,
                alleles,
                blocks,
            })
        } else {
            Err(HeaderError)
        }
    }

    pub fn read<R>(reader: &mut R) -> io::Result<Self>
    where
        R: io::Read,
    {
        let mut reader = Reader::new(reader);
        reader.read()
    }

    pub fn write<W>(&self, writer: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        let mut writer = Writer::new(writer);
        writer.write(self)
    }

    pub fn header_size(&self) -> usize {
        let magic_size = mem::size_of::<u8>() * MAGIC_NUMBER.len();
        let sites_size = mem::size_of::<u64>();
        let alleles_size = mem::size_of::<usize>() + self.alleles.len() * mem::size_of::<u64>();
        let blocks_size = mem::size_of::<u64>();

        magic_size + sites_size + alleles_size + blocks_size
    }

    pub fn data_size(&self) -> usize {
        let values_per_site = self.alleles.iter().map(|x| (x + 1) as usize).sum::<usize>();
        let site_size = values_per_site * mem::size_of::<f32>();

        self.sites as usize * site_size
    }

    pub fn total_size(&self) -> usize {
        self.header_size() + self.data_size()
    }
}

#[derive(Debug)]
pub struct HeaderError;

impl fmt::Display for HeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid header")
    }
}

impl std::error::Error for HeaderError {}

impl From<HeaderError> for io::Error {
    fn from(error: HeaderError) -> Self {
        io::Error::new(io::ErrorKind::InvalidData, error)
    }
}

struct Writer<'a, W> {
    inner: &'a mut W,
}

impl<'a, W> Writer<'a, W>
where
    W: io::Write,
{
    pub fn new(inner: &'a mut W) -> Self {
        Self { inner }
    }

    pub fn write(&mut self, header: &Header) -> io::Result<()> {
        self.inner.write_all(MAGIC_NUMBER)?;
        self.inner.write_all(&header.sites.to_le_bytes())?;
        self.write_alleles(header.alleles())?;
        self.inner.write_all(&header.blocks.to_le_bytes())
    }

    fn write_alleles(&mut self, alleles: &[u64]) -> io::Result<()> {
        self.inner.write_all(&alleles.len().to_le_bytes())?;
        for allele in alleles {
            self.inner.write_all(&allele.to_le_bytes())?;
        }
        Ok(())
    }
}

struct Reader<'a, R> {
    inner: &'a mut R,
}

impl<'a, R> Reader<'a, R>
where
    R: io::Read,
{
    pub fn new(inner: &'a mut R) -> Self {
        Self { inner }
    }

    fn read(&mut self) -> io::Result<Header> {
        self.read_magic()?;

        let sites = self.read_u64()?;
        let alleles = self.read_alleles()?;
        let blocks = self.read_u64()?;

        Header::new(sites, alleles, blocks).map_err(io::Error::from)
    }

    fn read_alleles(&mut self) -> io::Result<Vec<u64>> {
        let len = self.read_usize()?;
        let mut alleles = Vec::with_capacity(len);
        for _ in 0..len {
            let allele = self.read_u64()?;
            alleles.push(allele);
        }
        Ok(alleles)
    }

    fn read_magic(&mut self) -> io::Result<()> {
        let mut magic = [0; MAGIC_NUMBER.len()];
        self.inner.read_exact(&mut magic)?;

        if &magic == MAGIC_NUMBER {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid or unsupported SAF magic number \
                    (found '{magic:02x?}', expected '{MAGIC_NUMBER:02x?}')"
                ),
            ))
        }
    }

    fn read_usize(&mut self) -> io::Result<usize> {
        let mut buf = [0; mem::size_of::<usize>()];
        self.inner.read_exact(&mut buf)?;
        Ok(usize::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> io::Result<u64> {
        let mut buf = [0; mem::size_of::<u64>()];
        self.inner.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }
}
