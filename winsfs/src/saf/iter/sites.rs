use rayon::iter::{
    plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer},
    IndexedParallelIterator, IntoParallelIterator, ParallelIterator,
};

use crate::saf::{AsSiteView, SafView, SiteView};

/// A type that can be turned into an iterator over SAF sites.
pub trait IntoSiteIterator<const N: usize> {
    /// The type of each individual site.
    type Item: AsSiteView<N>;
    /// The type of iterator.
    type Iter: Iterator<Item = Self::Item>;

    /// Convert this type into a SAF site iterator.
    fn into_site_iter(self) -> Self::Iter;
}

impl<'a, const N: usize> IntoSiteIterator<N> for SafView<'a, N> {
    type Item = SiteView<'a, N>;
    type Iter = SiteIter<'a, N>;

    fn into_site_iter(self) -> Self::Iter {
        SiteIter::new(self)
    }
}

impl<'a, 'b, const N: usize> IntoSiteIterator<N> for &'b SafView<'a, N> {
    type Item = SiteView<'a, N>;
    type Iter = SiteIter<'a, N>;

    fn into_site_iter(self) -> Self::Iter {
        SiteIter::new(*self)
    }
}

impl<const N: usize, T> IntoSiteIterator<N> for T
where
    T: IntoIterator,
    T::Item: AsSiteView<N>,
{
    type Item = T::Item;
    type Iter = T::IntoIter;

    fn into_site_iter(self) -> Self::Iter {
        self.into_iter()
    }
}

/// A type that can be turned into a parallel iterator over SAF sites.
pub trait IntoParallelSiteIterator<const N: usize> {
    /// The type of each individual site.
    type Item: AsSiteView<N>;
    /// The type of iterator.
    type Iter: IndexedParallelIterator<Item = Self::Item>;

    /// Convert this type into a parallel SAF site iterator.
    fn into_par_site_iter(self) -> Self::Iter;
}

impl<'a, const N: usize> IntoParallelSiteIterator<N> for SafView<'a, N> {
    type Item = SiteView<'a, N>;
    type Iter = ParSiteIter<'a, N>;

    fn into_par_site_iter(self) -> Self::Iter {
        ParSiteIter::new(self)
    }
}

impl<'a, 'b, const N: usize> IntoParallelSiteIterator<N> for &'b SafView<'a, N> {
    type Item = SiteView<'a, N>;
    type Iter = ParSiteIter<'a, N>;

    fn into_par_site_iter(self) -> Self::Iter {
        ParSiteIter::new(*self)
    }
}

impl<const N: usize, T> IntoParallelSiteIterator<N> for T
where
    T: IntoParallelIterator,
    T::Iter: IndexedParallelIterator,
    T::Item: AsSiteView<N>,
{
    type Item = T::Item;
    type Iter = T::Iter;

    fn into_par_site_iter(self) -> Self::Iter {
        self.into_par_iter()
    }
}

/// An iterator over SAF sites.
#[derive(Debug)]
pub struct SiteIter<'a, const N: usize> {
    iter: ::std::slice::Chunks<'a, f32>,
    shape: [usize; N],
}

impl<'a, const N: usize> SiteIter<'a, N> {
    pub(in crate::saf) fn new(saf: SafView<'a, N>) -> Self {
        let iter = saf.values.chunks(saf.width());

        Self {
            iter,
            shape: saf.shape,
        }
    }
}

impl<'a, const N: usize> Iterator for SiteIter<'a, N> {
    type Item = SiteView<'a, N>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(|item| SiteView::new_unchecked(item, self.shape))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, const N: usize> ExactSizeIterator for SiteIter<'a, N> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, const N: usize> DoubleEndedIterator for SiteIter<'a, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter
            .next_back()
            .map(|item| SiteView::new_unchecked(item, self.shape))
    }
}

/// A parallel iterator over SAF sites.
#[derive(Debug)]
pub struct ParSiteIter<'a, const N: usize> {
    values: &'a [f32],
    shape: [usize; N],
    chunk_size: usize,
}

impl<'a, const N: usize> ParSiteIter<'a, N> {
    pub(in crate::saf) fn new(saf: SafView<'a, N>) -> Self {
        Self {
            values: saf.values,
            shape: saf.shape,
            chunk_size: saf.width(),
        }
    }
}

/*
    All the trait impls below are mainly boilerplate, and largely adapted from the
    implementation of rayon::slice::Chunks,
    see https://docs.rs/rayon/latest/src/rayon/slice/chunks.rs.html#8-11
*/

impl<'a, const N: usize> ParallelIterator for ParSiteIter<'a, N> {
    type Item = SiteView<'a, N>;

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

impl<'a, const N: usize> IndexedParallelIterator for ParSiteIter<'a, N> {
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
        callback.callback(SiteProducer {
            values: self.values,
            shape: self.shape,
            chunk_size: self.chunk_size,
        })
    }
}

struct SiteProducer<'a, const N: usize> {
    values: &'a [f32],
    shape: [usize; N],
    chunk_size: usize,
}

impl<'a, const N: usize> Producer for SiteProducer<'a, N> {
    type Item = SiteView<'a, N>;
    type IntoIter = SiteIter<'a, N>;

    fn into_iter(self) -> Self::IntoIter {
        SiteIter {
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
