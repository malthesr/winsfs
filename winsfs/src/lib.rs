pub mod saf;
pub use saf::{JointSaf, JointSafView, Saf, SafView};

pub mod sfs;
pub use sfs::{Em, Sfs, Sfs1d, Sfs2d};

pub mod stream;

pub mod window;
pub use window::{StoppingRule, Window};

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
