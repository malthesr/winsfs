use crate::{
    io::Rewind,
    sfs::{Sfs, UnnormalisedSfs},
};

use super::{
    stopping::{Stop, StoppingRule},
    Em, EmStep, StreamingEm,
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

impl<T, F> EmStep for Inspect<T, F>
where
    T: EmStep,
{
    type Status = T::Status;
}

impl<const N: usize, T, F, I> Em<N, I> for Inspect<T, F>
where
    T: Em<N, I>,
    F: FnMut(&T, &T::Status, &UnnormalisedSfs<N>),
{
    fn e_step(&mut self, sfs: &Sfs<N>, input: &I) -> (Self::Status, UnnormalisedSfs<N>) {
        let (status, sfs) = self.inner.e_step(sfs, input);

        (self.f)(&self.inner, &status, &sfs);

        (status, sfs)
    }
}

impl<const N: usize, T, F, R> StreamingEm<N, R> for Inspect<T, F>
where
    R: Rewind,
    T: StreamingEm<N, R>,
    F: FnMut(&T, &T::Status, &UnnormalisedSfs<N>),
{
    fn stream_e_step(
        &mut self,
        sfs: &Sfs<N>,
        reader: &mut R,
    ) -> std::io::Result<(Self::Status, UnnormalisedSfs<N>)> {
        let (status, sfs) = self.inner.stream_e_step(sfs, reader)?;

        (self.f)(&self.inner, &status, &sfs);

        Ok((status, sfs))
    }
}

impl<S, F> StoppingRule for Inspect<S, F> where S: StoppingRule {}

impl<T, S, F> Stop<T> for Inspect<S, F>
where
    T: EmStep,
    S: Stop<T, Status = T::Status>,
    F: FnMut(&S),
{
    type Status = T::Status;

    fn stop<const N: usize>(&mut self, em: &T, status: &Self::Status, sfs: &Sfs<N>) -> bool {
        (self.f)(&self.inner);

        self.inner.stop(em, status, sfs)
    }
}
