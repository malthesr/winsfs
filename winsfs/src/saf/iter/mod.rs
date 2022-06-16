//! Structs and traits for iterating over SAF likelihood matrices and views.

mod blocks;
pub use blocks::{BlockIter, IntoBlockIterator, IntoParallelBlockIterator, ParBlockIter};

mod sites;
pub use sites::{IntoParallelSiteIterator, IntoSiteIterator, ParSiteIter, SiteIter};
