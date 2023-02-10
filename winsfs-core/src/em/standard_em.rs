use std::{convert::Infallible, io};

use crate::{
    io::ReadSite,
    saf::SafView,
    sfs::{Sfs, USfs},
};

use super::{
    likelihood::{LogLikelihood, SumOf},
    EmStep, WithStatus,
};

/// A parallel runner of the standard EM algorithm.
pub type ParallelEm = StandardEm<true, false>;

/// A streaming runner of the standard EM algorithm.
pub type StreamingEm = StandardEm<false, true>;

/// A runner of the standard EM algorithm.
///
/// Whether to parallelise over the input in the E-step is controlled by the `PAR` parameter,
/// whether to stream through data on disk is controlled by the `STREAM` parameter.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
// TODO: Use const enum here when stable, see github.com/rust-lang/rust/issues/95174
pub struct StandardEm<const PAR: bool = false, const STREAM: bool = false>;

impl<const PAR: bool, const STREAM: bool> StandardEm<PAR, STREAM> {
    /// Returns a new instance of the runner.
    pub fn new() -> Self {
        Self {}
    }
}

impl<const PAR: bool, const STREAM: bool> WithStatus for StandardEm<PAR, STREAM> {
    type Status = SumOf<LogLikelihood>;
}

impl<'a, const D: usize, const PAR: bool> EmStep<D, SafView<'a, D>> for StandardEm<PAR, false> {
    type Error = Infallible;

    fn log_likelihood(
        &mut self,
        sfs: Sfs<D>,
        saf: SafView<'a, D>,
    ) -> Result<SumOf<LogLikelihood>, Self::Error> {
        if PAR {
            Ok(sfs.par_log_likelihood(saf))
        } else {
            Ok(sfs.log_likelihood(saf))
        }
    }

    fn e_step(
        &mut self,
        sfs: Sfs<D>,
        saf: SafView<D>,
    ) -> Result<(Self::Status, USfs<D>), Self::Error> {
        if PAR {
            Ok(sfs.par_e_step(saf))
        } else {
            Ok(sfs.e_step(saf))
        }
    }
}

impl<'a, const D: usize, R> EmStep<D, &'a mut R> for StandardEm<false, true>
where
    R: ReadSite,
{
    type Error = io::Error;

    fn log_likelihood(
        &mut self,
        sfs: Sfs<D>,
        reader: &'a mut R,
    ) -> Result<SumOf<LogLikelihood>, Self::Error> {
        sfs.stream_log_likelihood(reader)
    }

    fn e_step(
        &mut self,
        sfs: Sfs<D>,
        reader: &'a mut R,
    ) -> Result<(Self::Status, USfs<D>), Self::Error> {
        sfs.stream_e_step(reader)
    }
}
