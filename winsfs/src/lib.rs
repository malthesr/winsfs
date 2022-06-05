pub mod saf;
pub use saf::{JointSaf, JointSafView, Saf, SafView};

pub mod sfs;
pub use sfs::{Em, Sfs};

pub mod stream;

pub mod window;
pub use window::{StoppingRule, Window};

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
