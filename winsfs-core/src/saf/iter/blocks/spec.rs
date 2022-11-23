/// A specification for how to split a SAF into blocks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Blocks {
    /// Splits the SAF into a fixed number of blocks.
    ///
    /// If the number of blocks `b` does not evenly divide the number of sites `n`,
    /// then the first `n mod b` blocks will have one more site than the remaining blocks.
    Number(usize),
    /// Splits the SAF into blocks of a fixed size.
    ///
    /// If the block size `m` does not evenly divide the number of sites `n`,
    /// then the last block will contain `n mod m` sites.
    Size(usize),
}

impl Blocks {
    pub(crate) fn to_spec(self, sites: usize) -> BlockSpec {
        match self {
            Self::Number(number) => BlockSpec::number(number, sites),
            Self::Size(size) => BlockSpec::size(size, sites),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Inner {
    Number {
        /// The block size for blocks not receiving an extra site
        block_size: usize,
        /// The number of starting blocks that need an extra site
        rem: usize,
    },
    Size {
        /// The block size for all blocks except (possibly) the last
        block_size: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BlockSpec {
    inner: Inner,
    sites: usize,
}

impl BlockSpec {
    /// Returns the number of blocks that will be created.
    pub(crate) fn blocks(&self) -> usize {
        let sites = self.sites;

        match self.inner {
            Inner::Number { block_size, rem } => (sites - rem) / block_size,
            Inner::Size { block_size } => {
                sites / block_size + if sites % block_size == 0 { 0 } else { 1 }
            }
        }
    }

    /// Returns the block offset in sites of the block with the given index.
    pub(super) fn block_offset(&self, index: usize) -> usize {
        match self.inner {
            Inner::Number {
                block_size, rem, ..
            } => block_size * index + rem.min(index),
            Inner::Size { block_size, .. } => block_size * index,
        }
    }

    /// Returns the block size in sites of the block with the given index.
    pub(super) fn block_size(&self, index: usize) -> usize {
        let sites = self.sites;

        match self.inner {
            Inner::Number { block_size, rem } => block_size + if index >= rem { 0 } else { 1 },
            Inner::Size { block_size } => {
                if (index + 1) * block_size <= sites {
                    block_size
                } else {
                    sites % block_size
                }
            }
        }
    }

    /// Returns an iterator over the sizes of blocks that will be created.
    pub(crate) fn iter_block_sizes(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.blocks()).map(|i| self.block_size(i))
    }

    /// Creates a new block specification.
    fn new(inner: Inner, sites: usize) -> Self {
        Self { inner, sites }
    }

    /// Creates a new block specification from a number of blocks.
    fn number(blocks: usize, sites: usize) -> Self {
        if blocks <= sites {
            let inner = Inner::Number {
                block_size: sites / blocks,
                rem: sites % blocks,
            };

            Self::new(inner, sites)
        } else {
            panic!("tried to split SAF more blocks {blocks} than sites {sites}")
        }
    }

    /// Creates a new block specification from a block size.
    fn size(block_size: usize, sites: usize) -> Self {
        if block_size <= sites {
            let inner = Inner::Size { block_size };

            Self::new(inner, sites)
        } else {
            panic!("tried to split SAF into block sizes {block_size} larger than sites {sites}")
        }
    }

    /// Splits the block specification at given index.
    pub(super) fn split(&self, index: usize) -> (Self, Self) {
        let (hd_inner, tl_inner) = match self.inner {
            Inner::Number { block_size, rem } => (
                Inner::Number {
                    block_size,
                    rem: rem.min(index),
                },
                Inner::Number {
                    block_size,
                    rem: rem.saturating_sub(index),
                },
            ),
            inner @ Inner::Size { .. } => (inner, inner),
        };

        let hd_sites = self.block_offset(index);
        let tl_sites = self.sites.saturating_sub(hd_sites);

        (Self::new(hd_inner, hd_sites), Self::new(tl_inner, tl_sites))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_blocks() {
        assert_eq!(BlockSpec::number(10, 10).blocks(), 10);
        assert_eq!(BlockSpec::number(10, 11).blocks(), 10);
        assert_eq!(BlockSpec::number(10, 19).blocks(), 10);
        assert_eq!(BlockSpec::number(11, 12).blocks(), 11);

        assert_eq!(BlockSpec::number(5, 10).blocks(), 5);
        assert_eq!(BlockSpec::number(3, 10).blocks(), 3);
        assert_eq!(BlockSpec::number(2, 10).blocks(), 2);
    }

    #[test]
    fn size_blocks() {
        assert_eq!(BlockSpec::size(10, 10).blocks(), 1);
        assert_eq!(BlockSpec::size(10, 11).blocks(), 2);
        assert_eq!(BlockSpec::size(10, 19).blocks(), 2);
        assert_eq!(BlockSpec::size(11, 12).blocks(), 2);

        assert_eq!(BlockSpec::size(5, 10).blocks(), 2);
        assert_eq!(BlockSpec::size(3, 10).blocks(), 4);
        assert_eq!(BlockSpec::size(2, 10).blocks(), 5);
    }

    #[test]
    fn number_block_size() {
        assert_eq!(BlockSpec::number(10, 10).block_size(0), 1);

        let spec = BlockSpec::number(10, 11);
        assert_eq!(spec.block_size(0), 2);
        assert_eq!(spec.block_size(1), 1);
        assert_eq!(spec.block_size(2), 1);

        let spec = BlockSpec::number(10, 19);
        assert_eq!(spec.block_size(0), 2);
        assert_eq!(spec.block_size(1), 2);
        assert_eq!(spec.block_size(8), 2);
        assert_eq!(spec.block_size(9), 1);
    }

    #[test]
    fn size_block_size() {
        assert_eq!(BlockSpec::size(10, 10).block_size(0), 10);

        let spec = BlockSpec::size(10, 11);
        assert_eq!(spec.block_size(0), 10);
        assert_eq!(spec.block_size(1), 1);

        let spec = BlockSpec::size(11, 28);
        assert_eq!(spec.block_size(0), 11);
        assert_eq!(spec.block_size(1), 11);
        assert_eq!(spec.block_size(2), 6);
    }

    #[test]
    fn number_block_offset() {
        assert_eq!(BlockSpec::number(10, 10).block_offset(0), 0);

        let spec = BlockSpec::number(10, 11);
        assert_eq!(spec.block_offset(0), 0);
        assert_eq!(spec.block_offset(1), 2);
        assert_eq!(spec.block_offset(2), 3);

        let spec = BlockSpec::number(10, 19);
        assert_eq!(spec.block_offset(0), 0);
        assert_eq!(spec.block_offset(1), 2);
        assert_eq!(spec.block_offset(8), 16);
        assert_eq!(spec.block_offset(9), 18);
    }

    #[test]
    fn size_block_offset() {
        assert_eq!(BlockSpec::size(10, 10).block_offset(0), 0);

        let spec = BlockSpec::size(10, 11);
        assert_eq!(spec.block_offset(0), 0);
        assert_eq!(spec.block_offset(1), 10);

        let spec = BlockSpec::size(11, 28);
        assert_eq!(spec.block_offset(0), 0);
        assert_eq!(spec.block_offset(1), 11);
        assert_eq!(spec.block_offset(2), 22);
    }

    #[test]
    fn number_iter_block_sizes() {
        let iter = BlockSpec::number(3, 14);
        let block_sizes: Vec<_> = iter.iter_block_sizes().collect();
        assert_eq!(&block_sizes, &[5, 5, 4]);
    }

    #[test]
    fn size_iter_block_sizes() {
        let iter = BlockSpec::size(6, 14);
        let block_sizes: Vec<_> = iter.iter_block_sizes().collect();
        assert_eq!(&block_sizes, &[6, 6, 2]);
    }

    #[test]
    fn number_split() {
        let spec = BlockSpec::number(10, 108);

        let (hd, tl) = spec.split(4);

        let hd_block_sizes: Vec<_> = hd.iter_block_sizes().collect();
        assert_eq!(&hd_block_sizes, &[11, 11, 11, 11]);

        let tl_block_sizes: Vec<_> = tl.iter_block_sizes().collect();
        assert_eq!(&tl_block_sizes, &[11, 11, 11, 11, 10, 10]);
    }

    #[test]
    fn size_split() {
        let spec = BlockSpec::size(10, 108);

        let (hd, tl) = spec.split(7);

        let hd_block_sizes: Vec<_> = hd.iter_block_sizes().collect();
        assert_eq!(&hd_block_sizes, &[10, 10, 10, 10, 10, 10, 10]);

        let tl_block_sizes: Vec<_> = tl.iter_block_sizes().collect();
        assert_eq!(&tl_block_sizes, &[10, 10, 10, 8]);
    }
}
