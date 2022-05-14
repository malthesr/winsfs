use std::collections::VecDeque;

pub const DEFAULT_TOLERANCE: f64 = 1e-4;

#[derive(Clone, Debug)]
pub struct StoppingRule(StoppingRuleInner);

impl StoppingRule {
    pub fn either(max_epochs: usize, log_likelihood_tolerance: f64) -> Self {
        Self(StoppingRuleInner::Either(
            EpochRule::new(max_epochs),
            LogLikelihoodRule::new(log_likelihood_tolerance),
        ))
    }

    pub fn epochs(max: usize) -> Self {
        Self(StoppingRuleInner::Epochs(EpochRule::new(max)))
    }

    pub fn log_likelihood(tolerance: f64) -> Self {
        Self(StoppingRuleInner::LogLikelihood(LogLikelihoodRule::new(
            tolerance,
        )))
    }

    pub(crate) fn stop(&self) -> bool {
        self.0.stop()
    }

    pub(crate) fn block_update(&mut self, log_likelihood: f64) {
        self.0.block_update(log_likelihood)
    }

    pub(crate) fn epoch_update(&mut self) {
        self.0.epoch_update()
    }
}

#[derive(Clone, Debug)]
enum StoppingRuleInner {
    Epochs(EpochRule),
    LogLikelihood(LogLikelihoodRule),
    Either(EpochRule, LogLikelihoodRule),
}

impl StoppingRuleInner {
    pub fn stop(&self) -> bool {
        match self {
            StoppingRuleInner::Epochs(epoch_rule) => epoch_rule.stop(),
            StoppingRuleInner::LogLikelihood(log_likelihood_rule) => log_likelihood_rule.stop(),
            StoppingRuleInner::Either(epoch_rule, log_likelihood_rule) => {
                epoch_rule.stop() || log_likelihood_rule.stop()
            }
        }
    }

    pub fn block_update(&mut self, log_likelihood: f64) {
        match self {
            StoppingRuleInner::Epochs(_) => (),
            StoppingRuleInner::LogLikelihood(log_likelihood_rule) => {
                log_likelihood_rule.block_update(log_likelihood)
            }
            StoppingRuleInner::Either(_, log_likelihood_rule) => {
                log_likelihood_rule.block_update(log_likelihood);
            }
        }
    }

    pub fn epoch_update(&mut self) {
        match self {
            StoppingRuleInner::Epochs(epoch_rule) => epoch_rule.epoch_update(),
            StoppingRuleInner::LogLikelihood(log_likelihood_rule) => {
                log_likelihood_rule.epoch_update()
            }
            StoppingRuleInner::Either(epoch_rule, log_likelihood_rule) => {
                epoch_rule.epoch_update();
                log_likelihood_rule.epoch_update();
            }
        }
    }
}

#[derive(Clone, Debug)]
struct EpochRule {
    current: usize,
    max: usize,
}

impl EpochRule {
    pub fn new(max: usize) -> Self {
        Self { current: 0, max }
    }

    pub fn stop(&self) -> bool {
        log::info!(
            target: "stopping",
            "Epoch {epoch}, epoch stopping rule, current epoch: {epoch}/{max}",
            epoch=self.current, max=self.max
        );

        self.current >= self.max
    }

    pub fn epoch_update(&mut self) {
        self.current += 1;
    }
}

#[derive(Clone, Debug)]
struct LogLikelihoodRule {
    epoch: usize,
    current: LogLikelihoods,
    last: LogLikelihoods,
    tolerance: f64,
}

impl LogLikelihoodRule {
    pub fn new(tolerance: f64) -> Self {
        Self {
            epoch: 0,
            current: LogLikelihoods::default(),
            last: LogLikelihoods::default(),
            tolerance,
        }
    }

    pub fn stop(&self) -> bool {
        let current = self.current.sum;
        let last = self.last.sum;
        let improv = current - last;

        if self.epoch >= 2 {
            log::info!(
                target: "stopping",
                "Epoch {}, log-likelihood stopping rule, current log-likelihood: {current:.8e} (Δ{improv:.8e})",
                self.epoch
            );
        }

        improv < self.tolerance
    }

    pub fn block_update(&mut self, item: f64) {
        match self.epoch {
            0 => {
                self.last.add(item);
            }
            1 => {
                self.current.add(item);
            }
            _ => {
                self.current.add(item);
                let old = self.current.remove().unwrap();
                let _last_old = self.last.update(old);
            }
        }
    }

    pub fn epoch_update(&mut self) {
        self.epoch += 1;
    }
}

#[derive(Clone, Debug, Default)]
struct LogLikelihoods {
    log_likelihoods: VecDeque<f64>,
    sum: f64,
}

impl LogLikelihoods {
    pub fn add(&mut self, item: f64) {
        self.log_likelihoods.push_back(item);
        self.sum += item;
    }

    pub fn remove(&mut self) -> Option<f64> {
        let old = self.log_likelihoods.pop_front()?;
        self.sum -= old;
        Some(old)
    }

    pub fn update(&mut self, item: f64) -> Option<f64> {
        self.add(item);
        self.remove()
    }
}
