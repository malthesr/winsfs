//! Expectation-maximisation ("EM") algorithms for SFS inference.

use std::io;

pub mod likelihood;

mod adaptors;
pub use adaptors::Inspect;

mod site;
pub use site::{EmSite, StreamEmSite};

mod standard_em;
pub use standard_em::{ParallelStandardEm, StandardEm};

pub mod stopping;
use stopping::Stop;

mod window_em;
pub use window_em::{Window, WindowEm};

use crate::{
    io::Rewind,
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
pub trait EmStep: Sized {
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

/// A type capable of running an EM-like algorithm for SFS inference using data in-memory.
pub trait Em<const N: usize, I>: EmStep {
    /// The E-step of the algorithm.
    ///
    /// This should correspond to a full pass over the `input`.
    ///
    /// # Panics
    ///
    /// Panics if the shapes of the SFS and the input do not match.
    fn e_step(&mut self, sfs: Sfs<N>, input: &I) -> (Self::Status, USfs<N>);

    /// A full EM-step of the algorithm.
    ///
    /// Like the [`Em::e_step`], this should correspond to a full pass over the `input`.
    ///
    /// # Panics
    ///
    /// Panics if the shapes of the SFS and the input do not match.
    fn em_step(&mut self, sfs: Sfs<N>, input: &I) -> (Self::Status, Sfs<N>) {
        let (status, posterior) = self.e_step(sfs, input);

        (status, posterior.normalise())
    }

    /// Runs the EM algorithm until convergence.
    ///
    /// This consists of running EM-steps until convergence, which is decided by the provided
    /// `stopping_rule`. The converged SFS, and the status of the last EM-step, are returned.
    ///
    /// # Panics
    ///
    /// Panics if the shapes of the SFS and the input do not match.
    fn em<S>(&mut self, mut sfs: Sfs<N>, input: &I, mut stopping_rule: S) -> (Self::Status, Sfs<N>)
    where
        S: Stop<Self, Status = Self::Status>,
    {
        loop {
            let (status, new_sfs) = self.em_step(sfs, input);
            sfs = new_sfs;

            if stopping_rule.stop(self, &status, &sfs) {
                break (status, sfs);
            }
        }
    }
}

/// A type capable of running an EM-like algorithm for SFS inference by streaming through data.
pub trait StreamingEm<const D: usize, R>: EmStep
where
    R: Rewind,
    R::Site: EmSite<D>,
{
    /// The E-step of the algorithm.
    ///
    /// This should correspond to a full pass through the `reader`.
    ///
    /// # Panics
    ///
    /// Panics if the shapes of the SFS and the input do not match.
    fn stream_e_step(&mut self, sfs: Sfs<D>, reader: &mut R)
        -> io::Result<(Self::Status, USfs<D>)>;

    /// A full EM-step of the algorithm.
    ///
    /// Like the [`Em::e_step`], this should correspond to a full pass through the `reader`.
    ///
    /// # Panics
    ///
    /// Panics if the shapes of the SFS and the input do not match.
    fn stream_em_step(
        &mut self,
        sfs: Sfs<D>,
        reader: &mut R,
    ) -> io::Result<(Self::Status, Sfs<D>)> {
        let (status, posterior) = self.stream_e_step(sfs, reader)?;

        Ok((status, posterior.normalise()))
    }

    /// Runs the EM algorithm until convergence.
    ///
    /// This consists of running EM-steps until convergence, which is decided by the provided
    /// `stopping_rule`. The converged SFS, and the status of the last EM-step, are returned.
    ///
    /// # Panics
    ///
    /// Panics if the shapes of the SFS and the input do not match.
    fn stream_em<S>(
        &mut self,
        mut sfs: Sfs<D>,
        reader: &mut R,
        mut stopping_rule: S,
    ) -> io::Result<(Self::Status, Sfs<D>)>
    where
        S: Stop<Self, Status = Self::Status>,
    {
        loop {
            let (status, new_sfs) = self.stream_em_step(sfs, reader)?;
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
        saf::{Blocks, Saf},
        saf1d, sfs1d,
    };

    fn impl_test_em_zero_not_nan<E>(mut runner: E)
    where
        E: Em<1, Saf<1>>,
    {
        let saf = saf1d![[0., 0., 1.]];
        let init_sfs = sfs1d![1., 0., 0.].into_normalised().unwrap();

        let (_, sfs) = runner.em(init_sfs, &saf, stopping::Steps::new(1));

        let has_nan = sfs.iter().any(|x| x.is_nan());
        assert!(!has_nan);
    }

    #[test]
    fn test_em_zero_sfs_not_nan() {
        impl_test_em_zero_not_nan(StandardEm::<false>::new())
    }

    #[test]
    fn test_parallel_em_zero_sfs_not_nan() {
        impl_test_em_zero_not_nan(ParallelStandardEm::new())
    }

    #[test]
    fn test_window_em_zero_sfs_not_nan() {
        impl_test_em_zero_not_nan(WindowEm::new(
            StandardEm::<false>::new(),
            Window::from_zeros([3], 1),
            Blocks::Size(1),
        ))
    }
}
