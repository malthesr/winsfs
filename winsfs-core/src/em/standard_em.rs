use std::io;

use crate::{
    io::Rewind,
    saf::iter::{IntoParallelSiteIterator, IntoSiteIterator},
    sfs::{Sfs, USfs},
};

use super::{
    likelihood::{LogLikelihood, SumOf},
    Em, EmStep, StreamingEm,
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

impl<const N: usize, I> Em<N, I> for StandardEm<false>
where
    for<'a> &'a I: IntoSiteIterator<N>,
{
    fn e_step(&mut self, sfs: &Sfs<N>, input: &I) -> (Self::Status, USfs<N>) {
        sfs.e_step(input)
    }
}

impl<const N: usize, R> StreamingEm<N, R> for StandardEm<false>
where
    R: Rewind,
{
    fn stream_e_step(
        &mut self,
        sfs: &Sfs<N>,
        reader: &mut R,
    ) -> io::Result<(Self::Status, USfs<N>)> {
        sfs.stream_e_step(reader)
    }
}

impl<const N: usize, I> Em<N, I> for StandardEm<true>
where
    for<'a> &'a I: IntoParallelSiteIterator<N>,
{
    fn e_step(&mut self, sfs: &Sfs<N>, input: &I) -> (Self::Status, USfs<N>) {
        sfs.par_e_step(input)
    }
}
