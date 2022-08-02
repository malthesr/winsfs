//! Structs for SFS iteration.
//!
//! These types must be public since they are returned by [`Sfs`] methods, but they simply exist
//! to be consumed as iterators. The corresponding method docs on [`Sfs`] are likely to be more
//! informative.

use super::compute_index_unchecked;

// Used for doc links, see github.com/rust-lang/rust/issues/79542
#[allow(unused_imports)]
use super::Sfs;

/// An iterator over the indices of an SFS.
#[derive(Clone, Debug)]
pub struct Indices<const N: usize> {
    n: usize,
    i: usize,
    rev_i: usize,
    shape: [usize; N],
}

impl<const N: usize> Indices<N> {
    /// Returns a new iterator over the indices of a given shape.
    ///
    /// See also [`Sfs::indices`] to construct directly from an SFS, and for more documentation.
    pub fn from_shape(shape: [usize; N]) -> Self {
        let n = shape.iter().product::<usize>();

        Self {
            n,
            i: 0,
            rev_i: n,
            shape,
        }
    }
}

impl<const N: usize> Iterator for Indices<N> {
    type Item = [usize; N];

    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.rev_i {
            let idx = compute_index_unchecked(self.i, self.n, self.shape);
            self.i += 1;
            Some(idx)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.rev_i - self.i;
        (len, Some(len))
    }
}

impl<const N: usize> ExactSizeIterator for Indices<N> {}

impl<const N: usize> DoubleEndedIterator for Indices<N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.i < self.rev_i {
            self.rev_i -= 1;
            let idx = compute_index_unchecked(self.rev_i, self.n, self.shape);
            Some(idx)
        } else {
            None
        }
    }
}

impl<const N: usize> std::iter::FusedIterator for Indices<N> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indices_1d() {
        let mut iter = Indices::from_shape([4]);

        assert_eq!(iter.len(), 4);

        assert_eq!(iter.next(), Some([0]));
        assert_eq!(iter.next(), Some([1]));

        assert_eq!(iter.len(), 2);

        assert_eq!(iter.next(), Some([2]));
        assert_eq!(iter.next(), Some([3]));

        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_indices_2d() {
        let mut iter = Indices::from_shape([2, 3]);

        assert_eq!(iter.len(), 6);

        assert_eq!(iter.next(), Some([0, 0]));
        assert_eq!(iter.next(), Some([0, 1]));
        assert_eq!(iter.next(), Some([0, 2]));

        assert_eq!(iter.len(), 3);

        assert_eq!(iter.next(), Some([1, 0]));
        assert_eq!(iter.next(), Some([1, 1]));
        assert_eq!(iter.next(), Some([1, 2]));

        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_indices_2d_mixed_direction() {
        let mut iter = Indices::from_shape([2, 3]);

        assert_eq!(iter.len(), 6);

        assert_eq!(iter.next(), Some([0, 0]));
        assert_eq!(iter.next_back(), Some([1, 2]));
        assert_eq!(iter.next_back(), Some([1, 1]));

        assert_eq!(iter.len(), 3);

        assert_eq!(iter.next(), Some([0, 1]));
        assert_eq!(iter.next_back(), Some([1, 0]));
        assert_eq!(iter.next(), Some([0, 2]));

        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_indices_3d() {
        let mut iter = Indices::from_shape([2, 1, 3]);

        assert_eq!(iter.len(), 6);

        assert_eq!(iter.next(), Some([0, 0, 0]));
        assert_eq!(iter.next(), Some([0, 0, 1]));
        assert_eq!(iter.next(), Some([0, 0, 2]));

        assert_eq!(iter.len(), 3);

        assert_eq!(iter.next(), Some([1, 0, 0]));
        assert_eq!(iter.next(), Some([1, 0, 1]));
        assert_eq!(iter.next(), Some([1, 0, 2]));

        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_indices_3d_rev() {
        let mut iter = Indices::from_shape([2, 1, 3]).rev();

        assert_eq!(iter.len(), 6);

        assert_eq!(iter.next(), Some([1, 0, 2]));
        assert_eq!(iter.next(), Some([1, 0, 1]));
        assert_eq!(iter.next(), Some([1, 0, 0]));

        assert_eq!(iter.len(), 3);

        assert_eq!(iter.next(), Some([0, 0, 2]));
        assert_eq!(iter.next(), Some([0, 0, 1]));
        assert_eq!(iter.next(), Some([0, 0, 0]));

        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());
    }
}
