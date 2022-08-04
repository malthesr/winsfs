//! Types for parameterising an SFS.
//!
//! This module contains the traits and structs used for the generics on the
//! [`SfsBase`](super::SfsBase) struct. In most situations, it should be possible to use the
//! provided type aliases in the parent module to avoid having to interact with these types
//! directly.

use std::{fmt, slice};

/// A type that can act as a shape for an SFS.
///
/// This is simply a trait alias for things like arrays and slices of usize.
/// Users should not need to implement this trait or care about its internals.
pub trait Shape: AsRef<[usize]> + Clone + fmt::Debug + Eq + PartialEq {
    /// Returns the strides of the shape.
    fn strides(&self) -> Self;

    /// Returns an iterator of the values of the shape.
    fn iter(&self) -> slice::Iter<'_, usize> {
        self.as_ref().iter()
    }

    /// Returns the length of the shape.
    fn len(&self) -> usize {
        self.as_ref().len()
    }

    /// Returns `true` if the shape is empty, `false` otherwise.
    fn is_empty(&self) -> bool {
        // Mainly to appease clippy
        self.as_ref().is_empty()
    }
}

/// An SFS shape that is known at compile-time.
pub type ConstShape<const D: usize> = [usize; D];

impl<const D: usize> Shape for ConstShape<D> {
    fn strides(&self) -> Self {
        let mut strides = [1; D];
        compute_strides(&self[..], &mut strides);
        strides
    }

    fn len(&self) -> usize {
        D
    }
}

/// An SFS shape that is known at run-time.
pub type DynShape = Box<[usize]>;

impl Shape for DynShape {
    fn strides(&self) -> Self {
        let mut strides = vec![1; self.len()];
        compute_strides(self, &mut strides);
        strides.into_boxed_slice()
    }
}

/// A marker trait for SFS normalisation.
///
/// An SFS can either be in a normalised or unnormalised state, i.e. the values of the SFS can
/// either be in probability space or not. When not in probability space, the values are
/// typically in count space, in which case they sum to the number of sites in the input from
/// which the SFS was constructed.
pub trait Normalisation {
    /// A boolean indicating whether the spectrum is normalised.
    const NORM: bool;
}

/// A marker for a normalised SFS.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Norm {}
impl Normalisation for Norm {
    const NORM: bool = true;
}

/// A marker for a (potentially) unnormalised SFS.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Unnorm {}
impl Normalisation for Unnorm {
    const NORM: bool = false;
}

/// Compute strides of `shape` into 1-initialised `strides`
fn compute_strides(shape: &[usize], strides: &mut [usize]) {
    debug_assert_eq!(shape.len(), strides.len());

    for (i, v) in shape.iter().enumerate().skip(1).rev() {
        strides.iter_mut().take(i).for_each(|stride| *stride *= v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_strides() {
        assert_eq!([7].strides(), [1]);
        assert_eq!([9, 3].strides(), [3, 1]);
        assert_eq!([3, 7, 5].strides(), [35, 5, 1]);
        assert_eq!([9, 3, 5, 7].strides(), [105, 35, 7, 1]);
    }
}
