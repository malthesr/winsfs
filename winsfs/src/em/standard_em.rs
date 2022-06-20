use std::io;

use crate::{
    io::ReadSite,
    saf::iter::{IntoParallelSiteIterator, IntoSiteIterator},
    sfs::{Sfs, UnnormalisedSfs},
};

use super::{
    likelihood::{LogLikelihood, SumOf},
    Em, EmStep, StreamingEm,
};

/// A runner of the standard EM algorithm.
///
/// The standard EM algorithm makes a single update of the estimated SFS for each EM-step.
#[derive(Clone, Debug, PartialEq)]
pub struct StandardEm {
    // Ensure unit struct cannot be constructed without constructor
    _private: (),
}

impl StandardEm {
    /// Returns a new instance of the runner.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for StandardEm {
    fn default() -> Self {
        Self::new()
    }
}

impl EmStep for StandardEm {
    type Status = SumOf<LogLikelihood>;
}

impl<const N: usize, I> Em<N, I> for StandardEm
where
    for<'a> &'a I: IntoSiteIterator<N>,
{
    fn e_step(&mut self, sfs: &Sfs<N>, input: &I) -> (Self::Status, UnnormalisedSfs<N>) {
        sfs.e_step(input)
    }
}

impl<const N: usize, R> StreamingEm<N, R> for StandardEm
where
    R: ReadSite,
{
    fn stream_e_step(
        &mut self,
        sfs: &Sfs<N>,
        reader: &mut R,
    ) -> io::Result<(Self::Status, UnnormalisedSfs<N>)> {
        sfs.stream_e_step(reader)
    }
}

/// A parallel runner of the standard EM algorithm.
///
/// This behaves exactly like [`StandardEm`], except parallelising over the input data in each
/// EM-step.
#[derive(Clone, Debug, PartialEq)]
pub struct ParallelStandardEm {
    // Ensure unit struct cannot be constructed without constructor
    _private: (),
}

impl ParallelStandardEm {
    /// Returns a new instance of the runner.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for ParallelStandardEm {
    fn default() -> Self {
        Self::new()
    }
}

impl EmStep for ParallelStandardEm {
    type Status = SumOf<LogLikelihood>;
}

impl<const N: usize, I> Em<N, I> for ParallelStandardEm
where
    for<'a> &'a I: IntoParallelSiteIterator<N>,
{
    fn e_step(&mut self, sfs: &Sfs<N>, input: &I) -> (Self::Status, UnnormalisedSfs<N>) {
        sfs.par_e_step(input)
    }
}
