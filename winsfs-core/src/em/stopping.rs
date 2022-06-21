//! Stopping rules used for deciding convergence for EM algorithms.

use crate::sfs::Sfs;

use super::{
    likelihood::{LogLikelihood, SumOf},
    EmStep, Inspect,
};

/// A type that can be combined in various ways to create a stopping rule for EM types.
///
/// This primary serves as a supertrait bound for [`Stop`]. [`Stop`] cannot provide these
/// methods directly without breaking type inference when calling them.
pub trait StoppingRule {
    /// Returns a new stopping rule that requires that *both* this *and* another stopping
    /// rule must indicicate convergence before stopping.
    fn and<S>(self, other: S) -> Both<Self, S>
    where
        Self: Sized,
    {
        Both::new(self, other)
    }

    /// Inspect the stopping rule after each E-step.
    ///
    /// This can only be used for inspecting the state of the stopping rule itself. To inspect
    /// other aspects of the algorithm, see [`EmStep::inspect`].
    fn inspect<F>(self, f: F) -> Inspect<Self, F>
    where
        Self: Sized,
        F: FnMut(&Self),
    {
        Inspect::new(self, f)
    }

    /// Returns a new stopping rule that requires that *either* this *or* another stopping
    /// rule must indicicate convergence before stopping.
    fn or<S>(self, other: S) -> Either<Self, S>
    where
        Self: Sized,
    {
        Either::new(self, other)
    }
}

/// A type capable of deciding whether an EM algorithm should stop.
pub trait Stop<T>: StoppingRule {
    /// A status from the E-step that may be used for checking convergence.
    type Status;

    /// Returns `true` if the algorithm should stop, `false` otherwise.
    fn stop<const N: usize>(&mut self, em: &T, status: &Self::Status, sfs: &Sfs<N>) -> bool;
}

/// A stopping rule that lets the EM algorithm run for a specific number of EM-steps.
pub struct Steps {
    current_step: usize,
    max_steps: usize,
}

impl Steps {
    /// Returns the current step.
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Returns the total number of steps before stopping.
    pub fn steps(&self) -> usize {
        self.max_steps
    }

    /// Creates a new stopping rule that allows `steps` EM steps.
    pub fn new(steps: usize) -> Self {
        Self {
            current_step: 0,
            max_steps: steps,
        }
    }
}

impl StoppingRule for Steps {}

impl<T> Stop<T> for Steps
where
    T: EmStep,
{
    type Status = T::Status;

    fn stop<const N: usize>(&mut self, _em: &T, _status: &Self::Status, _sfs: &Sfs<N>) -> bool {
        self.current_step += 1;
        self.current_step >= self.max_steps
    }
}

/// A stopping rule that lets the EM algorithm run until the absolute difference in successive,
/// normalised log-likelihood values falls below some tolerance.
///
/// The log-likelihood will be normalised by the number of sites, so that it becomes a per-site
/// measure. This makes it easier to find a reasonable tolerance for a range of input sizes.
pub struct LogLikelihoodTolerance {
    abs_diff: f64,
    log_likelihood: f64,
    tolerance: f64,
}

impl LogLikelihoodTolerance {
    /// Returns the absolute difference between the two most recent normalised log-likelihood values.
    pub fn absolute_difference(&self) -> f64 {
        self.abs_diff
    }

    /// Returns the current, normalised log-likelihood value.
    pub fn log_likelihood(&self) -> LogLikelihood {
        self.log_likelihood.into()
    }

    /// Creates a new stopping rule that allows EM steps until the absolute difference in successive,
    /// normalised log-likelihood values falls below `tolerance`.
    pub fn new(tolerance: f64) -> Self {
        Self {
            abs_diff: f64::INFINITY,
            log_likelihood: f64::NEG_INFINITY,
            tolerance,
        }
    }

    /// Returns the tolerance defining convergence.
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Provides the implementation of `stop`, shared between `LogLikelihoodTolerance`
    /// and `WindowLogLikelihoodTolerance`.
    fn stop_inner(&mut self, new_log_likelihood: f64) -> bool {
        self.abs_diff = (new_log_likelihood - self.log_likelihood).abs();
        self.log_likelihood = new_log_likelihood;

        self.abs_diff <= self.tolerance
    }
}

impl StoppingRule for LogLikelihoodTolerance {}

impl<T> Stop<T> for LogLikelihoodTolerance
where
    T: EmStep<Status = SumOf<LogLikelihood>>,
{
    type Status = T::Status;

    fn stop<const N: usize>(&mut self, _em: &T, status: &Self::Status, _sfs: &Sfs<N>) -> bool {
        let new_log_likelihood = status.normalise();

        self.stop_inner(new_log_likelihood)
    }
}

/// A stopping rule for window EM that lets the algorithm run until the successive sum of
/// normalised block log-likelihood values falls below a certain tolerance.
///
/// This is analogous to [`LogLikelihoodTolerance`], but instead of considering the full
/// (normalised) data log-likelihood, we consider the sum of these values over blocks.
pub struct WindowLogLikelihoodTolerance {
    inner: LogLikelihoodTolerance,
}

impl WindowLogLikelihoodTolerance {
    /// Returns the absolute difference between the two most recent window log-likelihood values.
    pub fn absolute_difference(&self) -> f64 {
        self.inner.absolute_difference()
    }

    /// Returns the current window log-likelihood value.
    pub fn log_likelihood(&self) -> LogLikelihood {
        self.inner.log_likelihood()
    }

    /// Creates a new stopping rule that allows EM steps until the absolute difference in successive,
    /// window log-likelihood values falls below `tolerance`.
    pub fn new(tolerance: f64) -> Self {
        Self {
            inner: LogLikelihoodTolerance::new(tolerance),
        }
    }

    /// Returns the tolerance defining convergence.
    pub fn tolerance(&self) -> f64 {
        self.inner.tolerance()
    }
}

impl StoppingRule for WindowLogLikelihoodTolerance {}

impl<T> Stop<T> for WindowLogLikelihoodTolerance
where
    T: EmStep<Status = Vec<SumOf<LogLikelihood>>>,
{
    type Status = T::Status;

    fn stop<const N: usize>(&mut self, _em: &T, status: &Self::Status, _sfs: &Sfs<N>) -> bool {
        let new_log_likelihood = status
            .iter()
            .map(|block_log_likelihood| block_log_likelihood.normalise())
            .sum();

        self.inner.stop_inner(new_log_likelihood)
    }
}

/// A stopping rule that lets the EM algorithm run until *both* the contained stopping rules
/// indicate convergence.
///
/// Typically constructed using [`StoppingRule::and`].
pub struct Both<A, B> {
    left: A,
    right: B,
}

impl<A, B> Both<A, B> {
    /// Returns a new stopping rule.
    fn new(left: A, right: B) -> Self {
        Self { left, right }
    }

    /// Returns the "left" stopping rule.
    pub fn left(&self) -> &A {
        &self.left
    }

    /// Returns the "right" stopping rule.
    pub fn right(&self) -> &B {
        &self.right
    }
}

impl<A, B> StoppingRule for Both<A, B>
where
    A: StoppingRule,
    B: StoppingRule,
{
}

impl<T, A, B> Stop<T> for Both<A, B>
where
    T: EmStep,
    A: Stop<T, Status = T::Status>,
    B: Stop<T, Status = T::Status>,
{
    type Status = T::Status;

    fn stop<const N: usize>(&mut self, em: &T, status: &Self::Status, sfs: &Sfs<N>) -> bool {
        self.left.stop(em, status, sfs) && self.right.stop(em, status, sfs)
    }
}

/// A stopping rule that lets the EM algorithm run until *either* of the contained stopping rules
/// indicate convergence.
///
/// Typically constructed using [`StoppingRule::or`].
pub struct Either<A, B> {
    left: A,
    right: B,
}

impl<A, B> Either<A, B> {
    /// Returns a new stopping rule.
    fn new(left: A, right: B) -> Self {
        Self { left, right }
    }

    /// Returns the "left" stopping rule.
    pub fn left(&self) -> &A {
        &self.left
    }

    /// Returns the "right" stopping rule.
    pub fn right(&self) -> &B {
        &self.right
    }
}

impl<A, B> StoppingRule for Either<A, B>
where
    A: StoppingRule,
    B: StoppingRule,
{
}

impl<T, A, B> Stop<T> for Either<A, B>
where
    T: EmStep,
    A: Stop<T, Status = T::Status>,
    B: Stop<T, Status = T::Status>,
{
    type Status = T::Status;

    fn stop<const N: usize>(&mut self, em: &T, status: &Self::Status, sfs: &Sfs<N>) -> bool {
        self.left.stop(em, status, sfs) || self.right.stop(em, status, sfs)
    }
}
