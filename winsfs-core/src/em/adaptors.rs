use crate::sfs::{Sfs, USfs};

use super::{
    likelihood::{LogLikelihood, SumOf},
    stopping::{Stop, StoppingRule},
    Em, EmStep, WithStatus,
};

/// A combinator for types that allows inspection after each E-step.
///
/// To inspect the EM process itself, this can be constructed using [`EmStep::inspect`],
/// which is available on anything implementing [`Em`] or [`StreamingEm`]. To inspect a
/// stopping rule, this can be constructed using [`StoppingRule::inspect`].
#[derive(Debug)]
pub struct Inspect<T, F> {
    inner: T,
    f: F,
}

impl<T, F> Inspect<T, F> {
    pub(super) fn new(inner: T, f: F) -> Self {
        Self { inner, f }
    }
}

impl<T, F> WithStatus for Inspect<T, F>
where
    T: WithStatus,
{
    type Status = T::Status;
}

impl<const D: usize, T, F, I> EmStep<D, I> for Inspect<T, F>
where
    T: Em<D, I>,
    F: FnMut(&T, &T::Status, &USfs<D>),
{
    type Error = T::Error;

    fn log_likelihood(
        &mut self,
        sfs: Sfs<D>,
        input: I,
    ) -> Result<SumOf<LogLikelihood>, Self::Error> {
        self.inner.log_likelihood(sfs, input)
    }

    fn e_step(&mut self, sfs: Sfs<D>, input: I) -> Result<(Self::Status, USfs<D>), Self::Error> {
        let (status, sfs) = self.inner.e_step(sfs, input)?;

        (self.f)(&self.inner, &status, &sfs);

        Ok((status, sfs))
    }
}

impl<S, F> StoppingRule for Inspect<S, F> where S: StoppingRule {}

impl<T, S, F> Stop<T> for Inspect<S, F>
where
    T: WithStatus,
    S: Stop<T>,
    F: FnMut(&S),
{
    fn stop<const D: usize>(&mut self, em: &T, status: &T::Status, sfs: &Sfs<D>) -> bool {
        (self.f)(&self.inner);

        self.inner.stop(em, status, sfs)
    }
}
