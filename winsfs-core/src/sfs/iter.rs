//! Structs for SFS iteration.
//!
//! These types must be public since they are returned by [`SfsBase`](super::SfsBase) methods, but
//! are just exposed to be consumed as iterators. The corresponding method docs on the base struct
//! are likely to be more informative.

use super::{ConstShape, Shape};

/// An iterator over the indices of an SFS.
#[derive(Clone, Debug)]
pub struct Indices<S: Shape> {
    n: usize,
    i: usize,
    rev_i: usize,
    shape: S,
}

impl<S: Shape> Indices<S> {
    /// Returns a new iterator over the indices of a given shape.
    pub fn from_shape(shape: S) -> Self {
        let n = shape.iter().product::<usize>();

        Self {
            n,
            i: 0,
            rev_i: n,
            shape,
        }
    }
}

impl<const D: usize> Iterator for Indices<ConstShape<D>> {
    type Item = [usize; D];

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

impl<const D: usize> DoubleEndedIterator for Indices<ConstShape<D>> {
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

impl<S: Shape> ExactSizeIterator for Indices<S> where Indices<S>: Iterator {}

impl<S: Shape> std::iter::FusedIterator for Indices<S> where Indices<S>: Iterator {}

fn compute_index_unchecked<const D: usize>(
    mut flat: usize,
    mut n: usize,
    shape: [usize; D],
) -> [usize; D] {
    let mut index = [0; D];
    for i in 0..D {
        n /= shape[i];
        index[i] = flat / n;
        flat %= n;
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_index() {
        assert_eq!(compute_index_unchecked(3, 4, [4]), [3]);
        assert_eq!(compute_index_unchecked(16, 28, [4, 7]), [2, 2]);
        assert_eq!(compute_index_unchecked(3, 6, [1, 3, 2]), [0, 1, 1]);
    }

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
