//! Types describing likelihood of the data given an SFS.
//!
//! The types here exist largely as newtype-wrappers around `f64`.
//! Getting likelihoods and log-likelihoods mixed up is an easy error to make; typing these
//! helps avoid such bugs, and make method signatures clearer.

use std::{
    iter::Sum,
    ops::{Add, AddAssign},
};

use super::to_f64;

/// The likelihood of the data given an SFS.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Likelihood(f64);

impl Likelihood {
    /// Returns the log-likelihood.
    pub fn ln(self) -> LogLikelihood {
        LogLikelihood(self.0.ln())
    }
}

impl From<f64> for Likelihood {
    #[inline]
    fn from(v: f64) -> Self {
        Self(v)
    }
}

impl From<Likelihood> for f64 {
    #[inline]
    fn from(v: Likelihood) -> Self {
        v.0
    }
}

/// The log-likelihood of the data given an SFS.
///
/// This is always the natural logarithm.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct LogLikelihood(f64);

impl From<f64> for LogLikelihood {
    #[inline]
    fn from(v: f64) -> Self {
        Self(v)
    }
}

impl From<LogLikelihood> for f64 {
    #[inline]
    fn from(v: LogLikelihood) -> Self {
        v.0
    }
}

impl Sum for LogLikelihood {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(LogLikelihood::from(0.0), |acc, x| acc + x)
    }
}

impl Add for LogLikelihood {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for LogLikelihood {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0
    }
}

/// A sum of items, and the number of items summed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SumOf<T> {
    sum: T,
    n: usize,
}

impl<T> SumOf<T> {
    /// Returns the sum of items, consuming `self`.
    pub fn into_sum(self) -> T {
        self.sum
    }

    /// Returns the number of items summed.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Creates a new sum.
    pub fn new(sum: T, n: usize) -> Self {
        Self { sum, n }
    }

    /// Returns the sum of items.
    pub fn sum(&self) -> &T {
        &self.sum
    }
}

impl SumOf<LogLikelihood> {
    /// Returns the log-likelihood normalised by the input size.
    pub(super) fn normalise(&self) -> f64 {
        f64::from(self.sum) / to_f64(self.n)
    }
}

impl<T> From<SumOf<T>> for (T, usize) {
    fn from(sum: SumOf<T>) -> Self {
        (sum.sum, sum.n)
    }
}
