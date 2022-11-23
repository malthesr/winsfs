use std::iter::FusedIterator;

use rayon::iter::{
    plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer},
    IndexedParallelIterator, ParallelIterator,
};

use crate::saf::{AsSafView, Saf, SafView};

mod spec;
pub(crate) use spec::BlockSpec;
pub use spec::Blocks;

/// A type that can be turned into an iterator blocks of SAF sites.
pub trait IntoBlockIterator<const N: usize> {
    /// The type of each individual block.
    type Item: AsSafView<N>;
    /// The type of iterator.
    type Iter: ExactSizeIterator<Item = Self::Item>;

    /// Convert this type into an iterator over blocks of SAF sites.
    fn into_block_iter(self, blocks: Blocks) -> Self::Iter;
}

impl<'a, const N: usize> IntoBlockIterator<N> for &'a Saf<N> {
    type Item = SafView<'a, N>;
    type Iter = BlockIter<'a, N>;

    fn into_block_iter(self, blocks: Blocks) -> Self::Iter {
        BlockIter::new(self.view(), blocks.to_spec(self.sites()))
    }
}

impl<'a, const N: usize> IntoBlockIterator<N> for SafView<'a, N> {
    type Item = SafView<'a, N>;
    type Iter = BlockIter<'a, N>;

    fn into_block_iter(self, blocks: Blocks) -> Self::Iter {
        BlockIter::new(self, blocks.to_spec(self.sites()))
    }
}

impl<'a, 'b, const N: usize> IntoBlockIterator<N> for &'b SafView<'a, N> {
    type Item = SafView<'a, N>;
    type Iter = BlockIter<'a, N>;

    fn into_block_iter(self, blocks: Blocks) -> Self::Iter {
        BlockIter::new(*self, blocks.to_spec(self.sites()))
    }
}

/// A type that can be turned into a parallel iterator blocks of SAF sites.
pub trait IntoParallelBlockIterator<const N: usize> {
    /// The type of each individual block.
    type Item: AsSafView<N>;
    /// The type of iterator.
    type Iter: IndexedParallelIterator<Item = Self::Item>;

    /// Convert this type into a parallel iterator over blocks of SAF sites().
    fn into_par_block_iter(self, blocks: Blocks) -> Self::Iter;
}

impl<'a, const N: usize> IntoParallelBlockIterator<N> for &'a Saf<N> {
    type Item = SafView<'a, N>;
    type Iter = ParBlockIter<'a, N>;

    fn into_par_block_iter(self, blocks: Blocks) -> Self::Iter {
        ParBlockIter::new(self.view(), blocks.to_spec(self.sites()))
    }
}

impl<'a, const N: usize> IntoParallelBlockIterator<N> for SafView<'a, N> {
    type Item = SafView<'a, N>;
    type Iter = ParBlockIter<'a, N>;

    fn into_par_block_iter(self, blocks: Blocks) -> Self::Iter {
        ParBlockIter::new(self, blocks.to_spec(self.sites()))
    }
}

impl<'a, 'b, const N: usize> IntoParallelBlockIterator<N> for &'b SafView<'a, N> {
    type Item = SafView<'a, N>;
    type Iter = ParBlockIter<'a, N>;

    fn into_par_block_iter(self, blocks: Blocks) -> Self::Iter {
        ParBlockIter::new(*self, blocks.to_spec(self.sites()))
    }
}

/// An iterator over blocks of SAF sites.
#[derive(Debug)]
pub struct BlockIter<'a, const N: usize> {
    saf: SafView<'a, N>,
    block_spec: BlockSpec,
    current: usize,
    max: usize,
}

impl<'a, const N: usize> BlockIter<'a, N> {
    fn new(saf: SafView<'a, N>, block_spec: BlockSpec) -> Self {
        Self {
            saf,
            block_spec,
            current: 0,
            max: block_spec.blocks(),
        }
    }
}

impl<'a, const N: usize> Iterator for BlockIter<'a, N> {
    type Item = SafView<'a, N>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        (self.current < self.max).then(|| {
            let start = self.block_spec.block_offset(self.current);
            let size = self.block_spec.block_size(self.current);
            self.current += 1;

            self.saf.block(start, size)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<'a, const N: usize> ExactSizeIterator for BlockIter<'a, N> {
    fn len(&self) -> usize {
        self.max - self.current
    }
}

impl<'a, const N: usize> DoubleEndedIterator for BlockIter<'a, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        (self.max > self.current).then(|| {
            let start = self.block_spec.block_offset(self.max - 1);
            let size = self.block_spec.block_size(self.max - 1);
            self.max -= 1;

            self.saf.block(start, size)
        })
    }
}

impl<'a, const N: usize> FusedIterator for BlockIter<'a, N> {}

/// A parallel iterator over blocks of SAF sites.
#[derive(Debug)]
pub struct ParBlockIter<'a, const N: usize> {
    saf: SafView<'a, N>,
    block_spec: BlockSpec,
}

impl<'a, const N: usize> ParBlockIter<'a, N> {
    fn new(saf: SafView<'a, N>, block_spec: BlockSpec) -> Self {
        Self { saf, block_spec }
    }
}

impl<'a, const N: usize> ParallelIterator for ParBlockIter<'a, N> {
    type Item = SafView<'a, N>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.len())
    }
}

impl<'a, const N: usize> IndexedParallelIterator for ParBlockIter<'a, N> {
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.block_spec.blocks()
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(BlockProducer {
            saf: self.saf,
            block_spec: self.block_spec,
        })
    }
}

struct BlockProducer<'a, const N: usize> {
    saf: SafView<'a, N>,
    block_spec: BlockSpec,
}

impl<'a, const N: usize> Producer for BlockProducer<'a, N> {
    type Item = SafView<'a, N>;
    type IntoIter = BlockIter<'a, N>;

    fn into_iter(self) -> Self::IntoIter {
        BlockIter::new(self.saf, self.block_spec)
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let offset = self.block_spec.block_offset(index);
        let (hd, tl) = self.saf.split(offset);
        let (hd_spec, tl_spec) = self.block_spec.split(index);

        (
            Self {
                saf: hd,
                block_spec: hd_spec,
            },
            Self {
                saf: tl,
                block_spec: tl_spec,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{saf1d, saf2d};

    macro_rules! assert_iter {
        ($iter:ident.$next:ident(), $expected:expr $(, len: $len:literal )?) => {
            assert_eq!($iter.$next().unwrap().as_slice(), $expected);
            $(
                assert_eq!($iter.len(), $len);
                match $len {
                    0 => {
                        assert!($iter.next().is_none());
                        assert!($iter.next_back().is_none());
                    }
                    _ => (),
                }
            )?
        };
    }

    #[test]
    fn number_iter_1d() {
        let saf = saf1d![
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ];

        let mut iter = saf.iter_blocks(Blocks::Number(1));
        assert_eq!(iter.len(), 1);
        assert_eq!(iter.next().unwrap(), saf.view());
        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());

        let mut iter = saf.iter_blocks(Blocks::Number(4));
        assert_eq!(iter.len(), 4);
        assert_iter!(iter.next(), &[0.0, 0.0, 1.0, 1.0], len: 3);
        assert_iter!(iter.next(), &[2.0, 2.0, 3.0, 3.0], len: 2);
        assert_iter!(iter.next(), &[4.0, 4.0], len: 1);
        assert_iter!(iter.next(), &[5.0, 5.0], len: 0);
    }

    #[test]
    fn number_iter_double_ended_2d() {
        let saf = saf2d![
            [0.0, 0.0; 10.0],
            [1.0, 1.0; 11.0],
            [2.0, 2.0; 12.0],
            [3.0, 3.0; 13.0],
            [4.0, 4.0; 14.0],
        ];

        let mut iter = saf.iter_blocks(Blocks::Number(3));
        assert_eq!(iter.len(), 3);
        assert_iter!(iter.next_back(), &[4.0, 4.0, 14.0], len: 2);
        assert_iter!(iter.next(), &[0.0, 0.0, 10.0, 1.0, 1.0, 11.0], len: 1);
        assert_iter!(iter.next_back(), &[2.0, 2.0, 12.0, 3.0, 3.0, 13.0], len: 0);
    }

    #[test]
    fn size_iter_2d() {
        let saf = saf2d![
            [0.0, 0.0; 10.0],
            [1.0, 1.0; 11.0],
            [2.0, 2.0; 12.0],
            [3.0, 3.0; 13.0],
            [4.0, 4.0; 14.0],
            [5.0, 5.0; 15.0],
        ];

        let mut iter = saf.iter_blocks(Blocks::Size(6));
        assert_eq!(iter.len(), 1);
        assert_eq!(iter.next().unwrap(), saf.view());
        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());

        let mut iter = saf.iter_blocks(Blocks::Size(4));
        assert_eq!(iter.len(), 2);
        assert_iter!(
            iter.next(),
            &[0.0, 0.0, 10.0, 1.0, 1.0, 11.0, 2.0, 2.0, 12.0, 3.0, 3.0, 13.0],
            len: 1
        );
        assert_iter!(iter.next(), &[4.0, 4.0, 14.0, 5.0, 5.0, 15.0], len: 0);
    }

    #[test]
    fn size_iter_double_ended_1d() {
        let saf = saf1d![
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
        ];

        let mut iter = saf.iter_blocks(Blocks::Size(2));

        assert_eq!(iter.len(), 5);
        assert_iter!(iter.next_back(), &[8.0], len: 4);
        assert_iter!(iter.next_back(), &[6.0, 7.0], len: 3);
        assert_iter!(iter.next_back(), &[4.0, 5.0], len: 2);
        assert_iter!(iter.next_back(), &[2.0, 3.0], len: 1);
        assert_iter!(iter.next_back(), &[0.0, 1.0], len: 0);
    }

    #[test]
    fn par_iter_fold_sum() {
        let saf = saf1d![[1.0], [1.0], [1.0], [1.0], [1.0]];

        let sum = |iter: ParBlockIter<1>| iter.map(|x| x.iter().sum::<f32>()).sum::<f32>();

        for i in 1..5 {
            assert_eq!(sum(saf.par_iter_blocks(Blocks::Number(i))), 5.0);
            assert_eq!(sum(saf.par_iter_blocks(Blocks::Size(i))), 5.0);
        }
    }
}
