//! Expectation-maximisation ("EM") algorithms for SFS inference.

mod adaptors;
pub use adaptors::Inspect;

pub mod likelihood;
use std::io;

use likelihood::{LogLikelihood, SumOf};

mod site;
pub use site::{EmSite, StreamEmSite};

mod standard_em;
pub use standard_em::{ParallelEm, StandardEm, StreamingEm};

pub mod stopping;
use stopping::Stop;

mod window_em;
pub use window_em::{StreamingWindowEm, WindowEm};

use crate::{
    io::Rewind,
    saf::SafView,
    sfs::{Sfs, USfs},
};

/// A type that knows how many sites it contains.
pub trait Sites {
    /// Returns the number of contained sites.
    fn sites(&self) -> usize;
}

/// An EM-like type that runs in steps.
///
/// This serves as a supertrait bound for both [`Em`] and [`StreamingEm`] and gathers
/// behaviour shared around running consecutive EM-steps.
pub trait WithStatus: Sized {
    /// The status returned after each step.
    ///
    /// This may be used, for example, to determine convergence by the stopping rule,
    /// or can be logged using [`EmStep::inspect`]. An example of a status might
    /// be the log-likelihood of the data given the SFS after the E-step.
    type Status;

    /// Inspect the status after each E-step.
    fn inspect<const N: usize, F>(self, f: F) -> Inspect<Self, F>
    where
        F: FnMut(&Self, &Self::Status, &USfs<N>),
    {
        Inspect::new(self, f)
    }
}

/// A type capable of running a single step of an EM-like algorithm.
pub trait EmStep<const N: usize, I>: WithStatus {
    /// The error of the result type returned after each step.
    type Error: std::error::Error;

    /// Evaluate the log-likelihood of the current SFS.
    fn log_likelihood(
        &mut self,
        sfs: Sfs<N>,
        input: I,
    ) -> Result<SumOf<LogLikelihood>, Self::Error>;

    /// The E-step of the algorithm.
    ///
    /// This should correspond to a full pass over the `input`.
    ///
    /// # Panics
    ///
    /// Panics if the shapes of the SFS and the input do not match.
    fn e_step(&mut self, sfs: Sfs<N>, input: I) -> Result<(Self::Status, USfs<N>), Self::Error>;

    /// A full EM-step of the algorithm.
    ///
    /// Like the [`Em::e_step`], this should correspond to a full pass over the `input`.
    ///
    /// # Panics
    ///
    /// Panics if the shapes of the SFS and the input do not match.
    fn em_step(&mut self, sfs: Sfs<N>, input: I) -> Result<(Self::Status, Sfs<N>), Self::Error> {
        let (status, posterior) = self.e_step(sfs, input)?;

        Ok((status, posterior.normalise()))
    }
}

/// A type capable of running an EM-like algorithm for SFS inference using data in-memory.
pub trait Em<const N: usize, I>: EmStep<N, I> {
    /// Runs the EM algorithm until convergence.
    ///
    /// This consists of running EM-steps until convergence, which is decided by the provided
    /// `stopping_rule`. The converged SFS, and the status of the last EM-step, are returned.
    ///
    /// # Panics
    ///
    /// Panics if the shapes of the SFS and the input do not match.
    fn em<S>(
        &mut self,
        sfs: Sfs<N>,
        input: I,
        stopping_rule: S,
    ) -> Result<(Self::Status, Sfs<N>), Self::Error>
    where
        S: Stop<Self>;
}

impl<'a, const N: usize, T> Em<N, SafView<'a, N>> for T
where
    T: EmStep<N, SafView<'a, N>>,
{
    fn em<S>(
        &mut self,
        mut sfs: Sfs<N>,
        saf: SafView<'a, N>,
        mut stopping_rule: S,
    ) -> Result<(Self::Status, Sfs<N>), Self::Error>
    where
        S: Stop<Self>,
    {
        loop {
            let (status, new_sfs) = self.em_step(sfs, saf)?;
            sfs = new_sfs;

            if stopping_rule.stop(self, &status, &sfs) {
                break Ok((status, sfs));
            }
        }
    }
}

impl<'a, const N: usize, R, T> Em<N, &'a mut R> for T
where
    for<'b> T: EmStep<N, &'b mut R, Error = io::Error>,
    R: Rewind + Sites,
{
    fn em<S>(
        &mut self,
        mut sfs: Sfs<N>,
        reader: &'a mut R,
        mut stopping_rule: S,
    ) -> Result<(Self::Status, Sfs<N>), Self::Error>
    where
        S: Stop<Self>,
    {
        loop {
            let (status, new_sfs) = self.em_step(sfs, reader)?;
            sfs = new_sfs;

            if stopping_rule.stop(self, &status, &sfs) {
                break Ok((status, sfs));
            } else {
                reader.rewind()?;
            }
        }
    }
}

pub(self) fn to_f64(x: usize) -> f64 {
    let result = x as f64;
    if result as usize != x {
        panic!("cannot convert {x} (usize) into f64");
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        saf::{Blocks, SafView},
        saf1d, sfs1d,
    };

    fn impl_test_em_zero_not_nan<E>(mut runner: E)
    where
        for<'a> E: Em<1, SafView<'a, 1>>,
    {
        let saf = saf1d![[0., 0., 1.]];
        let init_sfs = sfs1d![1., 0., 0.].into_normalised().unwrap();

        let (_, sfs) = runner
            .em(init_sfs, saf.view(), stopping::Steps::new(1))
            .unwrap();

        let has_nan = sfs.iter().any(|x| x.is_nan());
        assert!(!has_nan);
    }

    #[test]
    fn test_em_zero_sfs_not_nan() {
        impl_test_em_zero_not_nan(StandardEm::<false>::new())
    }

    #[test]
    fn test_parallel_em_zero_sfs_not_nan() {
        impl_test_em_zero_not_nan(ParallelEm::new())
    }

    #[test]
    fn test_window_em_zero_sfs_not_nan() {
        impl_test_em_zero_not_nan(WindowEm::new(
            StandardEm::<false>::new(),
            1,
            Blocks::Size(1),
        ))
    }
}
