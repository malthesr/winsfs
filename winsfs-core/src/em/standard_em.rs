use std::io;

use crate::{
    io::Rewind,
    saf::iter::{IntoParallelSiteIterator, IntoSiteIterator},
    sfs::{Sfs, USfs},
};

use super::{
    likelihood::{LogLikelihood, SumOf},
    Em, EmStep, StreamEmSite, StreamingEm,
};

/// A parallel runner of the standard EM algorithm.
pub type ParallelStandardEm = StandardEm<true>;

/// A runner of the standard EM algorithm.
///
/// Whether to parallelise over the input in the E-step is controlled by the `PAR` parameter.
#[derive(Clone, Debug, Eq, PartialEq)]
// TODO: Use an enum here when stable, see github.com/rust-lang/rust/issues/95174
pub struct StandardEm<const PAR: bool = false> {
    // Ensure unit struct cannot be constructed without constructor
    _private: (),
}

impl<const PAR: bool> StandardEm<PAR> {
    /// Returns a new instance of the runner.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl<const PAR: bool> Default for StandardEm<PAR> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const PAR: bool> EmStep for StandardEm<PAR> {
    type Status = SumOf<LogLikelihood>;
}

impl<const D: usize, I> Em<D, I> for StandardEm<false>
where
    for<'a> &'a I: IntoSiteIterator<D>,
{
    fn e_step(&mut self, sfs: Sfs<D>, input: &I) -> (Self::Status, USfs<D>) {
        sfs.e_step(input)
    }
}

impl<const D: usize, R> StreamingEm<D, R> for StandardEm<false>
where
    R: Rewind,
    R::Site: StreamEmSite<D>,
{
    fn stream_e_step(
        &mut self,
        sfs: Sfs<D>,
        reader: &mut R,
    ) -> io::Result<(Self::Status, USfs<D>)> {
        sfs.stream_e_step(reader)
    }
}

impl<const D: usize, I> Em<D, I> for StandardEm<true>
where
    for<'a> &'a I: IntoParallelSiteIterator<D>,
{
    fn e_step(&mut self, sfs: Sfs<D>, input: &I) -> (Self::Status, USfs<D>) {
        sfs.par_e_step(input)
    }
}
