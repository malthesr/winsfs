use winsfs_core::{
    em::{
        likelihood::{LogLikelihood, SumOf},
        stopping::{Either, Steps, Stop, StoppingRule, WindowLogLikelihoodTolerance},
        EmStep,
    },
    sfs::Sfs,
};

use crate::Cli;

use super::DEFAULT_TOLERANCE;

/// A stopping rule comprising the possible convergence criteria exposed through the cli.
pub enum Rule {
    Steps(Steps),
    LogLikelihood(WindowLogLikelihoodTolerance),
    Either(Either<Steps, WindowLogLikelihoodTolerance>),
}

impl StoppingRule for Rule {}

impl<T> Stop<T> for Rule
where
    T: EmStep<Status = Vec<SumOf<LogLikelihood>>>,
{
    fn stop<const N: usize>(&mut self, em: &T, status: &T::Status, sfs: &Sfs<N>) -> bool {
        match self {
            Self::Steps(rule) => {
                let stop = rule.stop(em, status, sfs);
                log_steps(rule);
                stop
            }
            Self::LogLikelihood(rule) => {
                let stop = rule.stop(em, status, sfs);
                log_log_likelihood(rule);
                stop
            }
            Self::Either(rule) => {
                let stop = rule.stop(em, status, sfs);
                log_steps(rule.left());
                log_log_likelihood(rule.right());
                stop
            }
        }
    }
}

impl From<&Cli> for Rule {
    fn from(args: &Cli) -> Self {
        match (args.max_epochs, args.tolerance) {
            (Some(n), Some(v)) => {
                log::debug!(
                    target: "stop",
                    "Stopping rule set to either {n} epochs or log-likelihood tolerance {v:.4e}"
                );

                Self::Either(Steps::new(n).or(WindowLogLikelihoodTolerance::new(v)))
            }
            (Some(n), None) => {
                log::debug!(
                    target: "stop",
                    "Stopping rule set to {n} epochs"
                );

                Self::Steps(Steps::new(n))
            }
            (None, Some(v)) => {
                log::debug!(
                    target: "stop",
                    "Stopping rule set to log-likelihood tolerance {v:.4e}"
                );

                Self::LogLikelihood(WindowLogLikelihoodTolerance::new(v))
            }
            (None, None) => {
                log::debug!(
                    target: "stop",
                    "Stopping rule set to log-likelihood tolerance {DEFAULT_TOLERANCE} (default)"
                );

                Self::LogLikelihood(WindowLogLikelihoodTolerance::new(DEFAULT_TOLERANCE))
            }
        }
    }
}

fn log_steps(rule: &Steps) {
    log::debug!(
        target: "stop",
        "Current epoch {i}/{max}",
        i = rule.current_step(),
        max = rule.steps(),
    )
}

fn log_log_likelihood(rule: &WindowLogLikelihoodTolerance) {
    log::debug!(
        target: "stop",
        "Current log-likelihood {lik:.4e}, Δ={diff:.4e} {sym} {tole:.4e}",
        lik = f64::from(rule.log_likelihood()),
        diff = rule.absolute_difference(),
        sym = if rule.absolute_difference() > rule.tolerance() { '>' } else { '≤' },
        tole = rule.tolerance(),
    )
}
