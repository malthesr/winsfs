use std::{io, iter::once, mem::size_of};

use super::{to_u16, to_u32, to_u64, to_usize};

/// The magic number written as the first 8 bytes of a pseudo-shuffled SAF file.
pub const MAGIC_NUMBER: [u8; 8] = *b"safvshuf";

/// The header for a pseudo-shuffled SAF file.
///
/// The header is written at the top of the file, and contains information about the size and layout
/// of the file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Header {
    sites: usize,
    shape: Vec<usize>,
    blocks: usize,
}

impl Header {
    /// Returns the number of blocks used for shuffling.
    pub fn blocks(&self) -> usize {
        self.blocks
    }

    /// Returns the size (in bytes) of the data that the file is expected to contain.
    pub(super) fn data_size(&self) -> usize {
        to_usize(self.sites) * self.width() * size_of::<f32>()
    }

    /// Returns an iterator over the byte offset of the start of each block.
    pub(super) fn block_offsets(&self) -> impl Iterator<Item = usize> {
        once(self.header_size())
            .chain(self.block_sizes().take(self.blocks - 1))
            .scan(0, |acc, x| {
                *acc += x;
                Some(*acc)
            })
    }

    /// Returns an iterator over the number of sites per block.
    pub(super) fn block_sites(&self) -> impl Iterator<Item = usize> {
        let div = self.sites / self.blocks;
        let rem = self.sites % self.blocks;

        (0..self.blocks).map(move |i| if i < rem { div + 1 } else { div })
    }

    /// Returns an iterator over the number of bytes per block.
    pub(super) fn block_sizes(&self) -> impl Iterator<Item = usize> {
        let width = self.width();
        self.block_sites()
            .map(move |sites| sites * width * size_of::<f32>())
    }

    /// Returns the size (in bytes) of the entire file.
    ///
    /// This is equal to the size of the header and the size of the data.
    pub(super) fn file_size(&self) -> usize {
        self.header_size() + self.data_size()
    }

    /// Returns the size (in bytes) of the header as it will be written to a file.
    pub(super) fn header_size(&self) -> usize {
        let shape_size = size_of::<u8>() + self.shape.len() * size_of::<u32>();

        size_of::<[u8; 8]>() + size_of::<u64>() + shape_size + size_of::<u16>()
    }

    /// Creates a new header.
    pub fn new(sites: usize, shape: Vec<usize>, blocks: usize) -> Self {
        Self {
            sites,
            shape,
            blocks,
        }
    }

    /// Reads the header, including the magic number, from a reader.
    pub(super) fn read<R>(mut reader: R) -> io::Result<Self>
    where
        R: io::Read,
    {
        let mut magic = [0; MAGIC_NUMBER.len()];
        reader.read_exact(&mut magic)?;

        if magic != MAGIC_NUMBER {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid or unsupported SAF magic number \
                    (found '{magic:02x?}', expected '{MAGIC_NUMBER:02x?}')"
                ),
            ));
        }

        let mut sites_buf = [0u8; size_of::<u64>()];
        reader.read_exact(&mut sites_buf)?;
        let sites = to_usize(u64::from_le_bytes(sites_buf));

        let mut shape_len_buf = [0u8; size_of::<u8>()];
        reader.read_exact(&mut shape_len_buf)?;
        let shape_len = u8::from_le_bytes(shape_len_buf);

        let mut shape_buf = [0u8; size_of::<u32>()];
        let mut shape = Vec::with_capacity(shape_len.into());
        for _ in 0..shape_len {
            reader.read_exact(&mut shape_buf)?;
            shape.push(to_usize(u32::from_le_bytes(shape_buf)));
        }

        let mut blocks_buf = [0u8; size_of::<u16>()];
        reader.read_exact(&mut blocks_buf)?;
        let blocks = usize::from(u16::from_le_bytes(blocks_buf));

        Ok(Self::new(sites, shape, blocks))
    }

    /// Returns the shape of each site in the file.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the number of sites in the file.
    pub fn sites(&self) -> usize {
        self.sites
    }

    /// Returns the width of each site, i.e. the total number of values.
    pub(super) fn width(&self) -> usize {
        self.shape.iter().sum()
    }

    /// Writes the header, including the magic number, to a writer.
    pub(super) fn write<W>(&self, mut writer: W) -> io::Result<()>
    where
        W: io::Write,
    {
        writer.write_all(&MAGIC_NUMBER)?;

        let sites = to_u64(self.sites);
        writer.write_all(&sites.to_le_bytes())?;

        let shape_len: u8 = self.shape.len().try_into().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "number of header dimensions exceeds {u8::MAX}",
            )
        })?;
        writer.write_all(&shape_len.to_le_bytes())?;
        for &v in self.shape.iter() {
            writer.write_all(&to_u32(v).to_le_bytes())?;
        }

        writer.write_all(&to_u16(self.blocks).to_le_bytes())?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    const TEST_HEADER: &[u8] = &[
        0x73, 0x61, 0x66, 0x76, 0x73, 0x68, 0x75, 0x66, // magic number
        0x69, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 105u64 sites
        0x02,                                           // 2u8 shapes
        0x07, 0x00, 0x00, 0x00,                         // 5u32 = shape[0]
        0x05, 0x00, 0x00, 0x00,                         // 7u32 = shape[1]
        0x0A, 0x00,                                     // 10u16 blocks
    ];

    #[test]
    fn test_write_header() -> io::Result<()> {
        let header = Header::new(105, vec![7, 5], 10);
        let mut dest = Vec::new();
        header.write(&mut dest)?;

        let expected = TEST_HEADER;
        assert_eq!(dest, expected);

        Ok(())
    }

    #[test]
    fn test_read_header() -> io::Result<()> {
        let src = TEST_HEADER;
        let header = Header::read(src)?;

        let expected = Header::new(105, vec![7, 5], 10);
        assert_eq!(header, expected);

        Ok(())
    }

    #[test]
    fn test_read_header_fails_wrong_magic() {
        let mut wrong_header = TEST_HEADER.to_vec();
        wrong_header[0] = 0;

        let result = Header::read(wrong_header.as_slice());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_header_size() {
        assert_eq!(Header::new(105, vec![7], 10).header_size(), 23);
        assert_eq!(Header::new(1005, vec![7, 5], 20).header_size(), 27);
        assert_eq!(Header::new(15, vec![7, 5, 11], 5).header_size(), 31);
    }

    #[test]
    fn test_data_size() {
        assert_eq!(Header::new(105, vec![7], 10).data_size(), 2940);
        assert_eq!(Header::new(1005, vec![7, 5], 20).data_size(), 48240);
        assert_eq!(Header::new(15, vec![7, 5, 11], 5).data_size(), 1380);
    }

    #[test]
    fn test_block_sites_even() {
        let header = Header::new(100, vec![3, 9], 5);
        let expected = vec![20; 5];
        assert_eq!(header.block_sites().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn test_block_sites_not_even() {
        let header = Header::new(99, vec![3, 9], 5);
        let expected: Vec<_> = vec![20, 20, 20, 20, 19];
        assert_eq!(header.block_sites().collect::<Vec<_>>(), expected);

        let header = Header::new(101, vec![3, 9], 5);
        let expected: Vec<_> = vec![21, 20, 20, 20, 20];
        assert_eq!(header.block_sites().collect::<Vec<_>>(), expected);

        let header = Header::new(10, vec![1, 2], 4);
        let expected: Vec<_> = vec![3, 3, 2, 2];
        assert_eq!(header.block_sites().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn test_block_sizes() {
        let header = Header::new(10, vec![1, 2], 4);
        let expected: Vec<_> = vec![36, 36, 24, 24];
        assert_eq!(header.block_sizes().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn test_block_offsets() {
        let header = Header::new(10, vec![1, 2], 4);
        let x = header.header_size();
        let expected: Vec<_> = vec![x, x + 36, x + 72, x + 96];
        assert_eq!(header.block_offsets().collect::<Vec<_>>(), expected);
    }
}
