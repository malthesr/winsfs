use rayon::iter::{
    plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer},
    IndexedParallelIterator, ParallelIterator,
};

use crate::saf::{AsSafView, Saf, SafView};

/// A type that can be turned into an iterator blocks of SAF sites.
pub trait IntoBlockIterator<const N: usize> {
    /// The type of each individual block.
    type Item: AsSafView<N>;
    /// The type of iterator.
    type Iter: Iterator<Item = Self::Item>;

    /// Convert this type into an iterator over SAF blocks containing
    /// `block_size` sites per block.
    fn into_block_iter(self, block_size: usize) -> Self::Iter;
}

impl<'a, const N: usize> IntoBlockIterator<N> for &'a Saf<N> {
    type Item = SafView<'a, N>;
    type Iter = BlockIter<'a, N>;

    fn into_block_iter(self, block_size: usize) -> Self::Iter {
        BlockIter::new(self.view(), block_size)
    }
}

impl<'a, const N: usize> IntoBlockIterator<N> for SafView<'a, N> {
    type Item = SafView<'a, N>;
    type Iter = BlockIter<'a, N>;

    fn into_block_iter(self, block_size: usize) -> Self::Iter {
        BlockIter::new(self, block_size)
    }
}

impl<'a, 'b, const N: usize> IntoBlockIterator<N> for &'b SafView<'a, N> {
    type Item = SafView<'a, N>;
    type Iter = BlockIter<'a, N>;

    fn into_block_iter(self, block_size: usize) -> Self::Iter {
        BlockIter::new(*self, block_size)
    }
}

/// A type that can be turned into a parallel iterator blocks of SAF sites.
pub trait IntoParallelBlockIterator<const N: usize> {
    /// The type of each individual block.
    type Item: AsSafView<N>;
    /// The type of iterator.
    type Iter: IndexedParallelIterator<Item = Self::Item>;

    /// Convert this type into a parallel iterator over SAF blocks containing
    /// `block_size` sites per block.
    fn into_par_block_iter(self, block_size: usize) -> Self::Iter;
}

impl<'a, const N: usize> IntoParallelBlockIterator<N> for &'a Saf<N> {
    type Item = SafView<'a, N>;
    type Iter = ParBlockIter<'a, N>;

    fn into_par_block_iter(self, block_size: usize) -> Self::Iter {
        ParBlockIter::new(self.view(), block_size)
    }
}

impl<'a, const N: usize> IntoParallelBlockIterator<N> for SafView<'a, N> {
    type Item = SafView<'a, N>;
    type Iter = ParBlockIter<'a, N>;

    fn into_par_block_iter(self, block_size: usize) -> Self::Iter {
        ParBlockIter::new(self, block_size)
    }
}

impl<'a, 'b, const N: usize> IntoParallelBlockIterator<N> for &'b SafView<'a, N> {
    type Item = SafView<'a, N>;
    type Iter = ParBlockIter<'a, N>;

    fn into_par_block_iter(self, block_size: usize) -> Self::Iter {
        ParBlockIter::new(*self, block_size)
    }
}

/// An iterator over blocks of SAF sites.
#[derive(Debug)]
pub struct BlockIter<'a, const N: usize> {
    iter: ::std::slice::Chunks<'a, f32>,
    shape: [usize; N],
}

impl<'a, const N: usize> BlockIter<'a, N> {
    pub(in crate::saf) fn new(saf: SafView<'a, N>, block_size: usize) -> Self {
        let iter = saf.values.chunks(saf.width() * block_size);

        Self {
            iter,
            shape: saf.shape,
        }
    }
}

impl<'a, const N: usize> Iterator for BlockIter<'a, N> {
    type Item = SafView<'a, N>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(|item| SafView::new_unchecked(item, self.shape))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, const N: usize> ExactSizeIterator for BlockIter<'a, N> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, const N: usize> DoubleEndedIterator for BlockIter<'a, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter
            .next_back()
            .map(|item| SafView::new_unchecked(item, self.shape))
    }
}

/// A parallel iterator over blocks of SAF sites.
#[derive(Debug)]
pub struct ParBlockIter<'a, const N: usize> {
    values: &'a [f32],
    shape: [usize; N],
    chunk_size: usize,
}

impl<'a, const N: usize> ParBlockIter<'a, N> {
    pub(in crate::saf) fn new(saf: SafView<'a, N>, block_size: usize) -> Self {
        Self {
            values: saf.values,
            shape: saf.shape,
            chunk_size: saf.width() * block_size,
        }
    }
}

/*
    All the trait impls below are mainly boilerplate, and largely adapted from the
    implementation of rayon::slice::Chunks,
    see https://docs.rs/rayon/latest/src/rayon/slice/chunks.rs.html#8-11
*/

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
        let n = self.values.len();
        if n == 0 {
            0
        } else {
            (n - 1) / self.chunk_size + 1
        }
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(BlockProducer {
            values: self.values,
            shape: self.shape,
            chunk_size: self.chunk_size,
        })
    }
}

struct BlockProducer<'a, const N: usize> {
    values: &'a [f32],
    shape: [usize; N],
    chunk_size: usize,
}

impl<'a, const N: usize> Producer for BlockProducer<'a, N> {
    type Item = SafView<'a, N>;
    type IntoIter = BlockIter<'a, N>;

    fn into_iter(self) -> Self::IntoIter {
        BlockIter {
            iter: self.values.chunks(self.chunk_size),
            shape: self.shape,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let elem_index = self.values.len().min(index * self.chunk_size);
        let (left, right) = self.values.split_at(elem_index);

        (
            Self {
                values: left,
                shape: self.shape,
                chunk_size: self.chunk_size,
            },
            Self {
                values: right,
                shape: self.shape,
                chunk_size: self.chunk_size,
            },
        )
    }
}
