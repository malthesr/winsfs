//! Methods for inferring the site frequency spectrum from low-quality data
//! using various forms of the expectation-maximisation algorithm.

#![allow(unstable_name_collisions)]
#![warn(missing_docs)]
use std::mem::MaybeUninit;

pub mod em;
pub mod io;
pub mod saf;
pub mod sfs;

pub mod prelude {
    //! The types required for common usage.
    pub use crate::em::{
        stopping::{LogLikelihoodTolerance, Steps},
        Em, ParallelStandardEm, StandardEm, StreamingEm, WindowEm,
    };
    pub use crate::saf::Saf;
    pub use crate::sfs::{Sfs, UnnormalisedSfs};
}

/// Sets the number of threads to use for parallelization.
///
/// This is a thin wrapper around [`rayon::ThreadPoolBuilder`] to save users from having to
/// import `rayon` to control parallelism. The meaning of the `threads` parameter here derives
/// from [`rayon::ThreadPoolBuilder::num_threads`], see it's documentation for details.
pub fn set_threads(threads: usize) -> Result<(), rayon::ThreadPoolBuildError> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
}

/// This is an internal implementation detail.
#[doc(hidden)]
#[macro_export]
macro_rules! matrix {
    ($([$($x:literal),+ $(,)?]),+ $(,)?) => {{
        let cols = vec![$($crate::matrix!(count: $($x),+)),+];
        assert!(cols.windows(2).all(|w| w[0] == w[1]));
        let vec = vec![$($($x),+),+];
        (cols, vec)
    }};
    (count: $($x:expr),+) => {
        <[()]>::len(&[$($crate::matrix!(replace: $x)),*])
    };
    (replace: $x:expr) => {()};
}

pub(crate) trait ArrayExt<const N: usize, T> {
    // TODO: Use each_ref when stable,
    // see github.com/rust-lang/rust/issues/76118
    fn each_ref(&self) -> [&T; N];

    // TODO: Use each_mut when stable,
    // see github.com/rust-lang/rust/issues/76118
    fn each_mut(&mut self) -> [&mut T; N];

    // TODO: Use zip when stable,
    // see github.com/rust-lang/rust/issues/80094
    fn zip<U>(self, rhs: [U; N]) -> [(T, U); N];
}

impl<const N: usize, T> ArrayExt<N, T> for [T; N] {
    fn each_ref(&self) -> [&T; N] {
        // Adapted from code in tracking issue, see above.
        let mut out: MaybeUninit<[&T; N]> = MaybeUninit::uninit();

        let buf = out.as_mut_ptr() as *mut &T;
        let mut refs = self.iter();

        for i in 0..N {
            unsafe { buf.add(i).write(refs.next().unwrap()) }
        }

        unsafe { out.assume_init() }
    }

    fn each_mut(&mut self) -> [&mut T; N] {
        // Adapted from code in tracking issue, see above.
        let mut out: MaybeUninit<[&mut T; N]> = MaybeUninit::uninit();

        let buf = out.as_mut_ptr() as *mut &mut T;
        let mut refs = self.iter_mut();

        for i in 0..N {
            unsafe { buf.add(i).write(refs.next().unwrap()) }
        }

        unsafe { out.assume_init() }
    }

    fn zip<U>(self, rhs: [U; N]) -> [(T, U); N] {
        // Adapted from code in implementation PR, see github.com/rust-lang/rust/pull/79451
        let mut dst = MaybeUninit::<[(T, U); N]>::uninit();

        let ptr = dst.as_mut_ptr() as *mut (T, U);

        for (idx, (lhs, rhs)) in self.into_iter().zip(rhs.into_iter()).enumerate() {
            unsafe { ptr.add(idx).write((lhs, rhs)) }
        }

        unsafe { dst.assume_init() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_each_ref() {
        assert_eq!([1, 2, 3].each_ref(), [&1, &2, &3]);
    }

    #[test]
    fn test_each_mut() {
        assert_eq!([1, 2, 3].each_ref(), [&mut 1, &mut 2, &mut 3]);
    }

    #[test]
    fn test_zip() {
        assert_eq!(
            [1, 2, 3].zip([0.1, 0.2, 0.3]),
            [(1, 0.1), (2, 0.2), (3, 0.3)],
        )
    }
}
