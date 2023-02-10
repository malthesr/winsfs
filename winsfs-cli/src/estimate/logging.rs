use std::process;

use winsfs_core::{
    em::{
        likelihood::{LogLikelihood, SumOf},
        EmStep, WithStatus,
    },
    sfs::{Sfs, USfs},
};

#[derive(Clone)]
pub struct Checker<T> {
    inner: T,
}

impl<T> Checker<T> {
    pub fn new(inner: T) -> Self {
        Self { inner }
    }
}

impl<T> WithStatus for Checker<T>
where
    T: WithStatus,
{
    type Status = T::Status;
}

impl<const N: usize, I, T> EmStep<N, I> for Checker<T>
where
    T: EmStep<N, I>,
{
    type Error = T::Error;

    fn log_likelihood(
        &mut self,
        sfs: Sfs<N>,
        input: I,
    ) -> Result<SumOf<LogLikelihood>, Self::Error> {
        self.inner.log_likelihood(sfs, input)
    }

    fn e_step(&mut self, sfs: Sfs<N>, input: I) -> Result<(Self::Status, USfs<N>), Self::Error> {
        let (status, sfs) = self.inner.e_step(sfs, input)?;

        if sfs.iter().any(|x| x.is_nan()) {
            log::error!(
                target: "windowem",
                "Found NaN: this is a bug, and the process will abort, please file an issue"
            );

            process::exit(1);
        };

        Ok((status, sfs))
    }
}

type LogFn = fn(&str, usize, &'static str, log::Level, log::Level);

#[derive(Clone)]
pub struct LoggerBuilder<const READY: bool> {
    log_fn: Option<LogFn>,
    log_target: &'static str,
    log_counter_level: log::Level,
    log_sfs_level: log::Level,
}

impl<const READY: bool> LoggerBuilder<READY> {
    pub fn log_counter_level(mut self, level: log::Level) -> Self {
        self.log_counter_level = level;
        self
    }

    pub fn log_sfs_level(mut self, level: log::Level) -> Self {
        self.log_sfs_level = level;
        self
    }

    pub fn log_target(mut self, target: &'static str) -> Self {
        self.log_target = target;
        self
    }
}

impl LoggerBuilder<false> {
    pub fn with_block_logging(self) -> LoggerBuilder<true> {
        LoggerBuilder {
            log_fn: Some(block_log_fn),
            log_target: self.log_target,
            log_counter_level: self.log_counter_level,
            log_sfs_level: self.log_sfs_level,
        }
    }

    pub fn with_epoch_logging(self) -> LoggerBuilder<true> {
        LoggerBuilder {
            log_fn: Some(epoch_log_fn),
            log_target: self.log_target,
            log_counter_level: self.log_counter_level,
            log_sfs_level: self.log_sfs_level,
        }
    }
}

impl LoggerBuilder<true> {
    pub fn build<T>(self, em: T) -> Logger<T> {
        Logger::new(
            em,
            self.log_fn.unwrap(),
            self.log_target,
            self.log_counter_level,
            self.log_sfs_level,
        )
    }
}

impl Default for LoggerBuilder<false> {
    fn default() -> Self {
        Self {
            log_fn: None,
            log_target: "winsfs",
            log_counter_level: log::Level::Info,
            log_sfs_level: log::Level::Debug,
        }
    }
}

pub struct Logger<T> {
    inner: T,
    counter: usize,
    log_fn: LogFn,
    log_target: &'static str,
    log_counter_level: log::Level,
    log_sfs_level: log::Level,
}

impl Logger<()> {
    pub fn builder() -> LoggerBuilder<false> {
        LoggerBuilder::default()
    }
}

impl<T> Logger<T> {
    fn new(
        em: T,
        log_fn: LogFn,
        log_target: &'static str,
        log_counter_level: log::Level,
        log_sfs_level: log::Level,
    ) -> Self {
        Self {
            inner: em,
            counter: 0,
            log_fn,
            log_target,
            log_counter_level,
            log_sfs_level,
        }
    }
}

impl<T> WithStatus for Logger<T>
where
    T: WithStatus,
{
    type Status = T::Status;
}

impl<const N: usize, I, T> EmStep<N, I> for Logger<T>
where
    T: EmStep<N, I>,
{
    type Error = T::Error;

    fn log_likelihood(
        &mut self,
        sfs: Sfs<N>,
        input: I,
    ) -> Result<SumOf<LogLikelihood>, Self::Error> {
        self.inner.log_likelihood(sfs, input)
    }

    fn e_step(&mut self, sfs: Sfs<N>, input: I) -> Result<(Self::Status, USfs<N>), Self::Error> {
        let (status, sfs) = self.inner.e_step(sfs, input)?;

        self.counter += 1;
        (self.log_fn)(
            &sfs.format_flat(" ", 6),
            self.counter,
            self.log_target,
            self.log_counter_level,
            self.log_sfs_level,
        );

        Ok((status, sfs))
    }
}

fn block_log_fn(
    fmt_sfs: &str,
    block: usize,
    target: &'static str,
    block_level: log::Level,
    sfs_level: log::Level,
) {
    log::log!(target: target, block_level, "Finished block {block}");
    log::log!(target: target, sfs_level, "Current block SFS: {fmt_sfs}");
}

fn epoch_log_fn(
    fmt_sfs: &str,
    epoch: usize,
    target: &'static str,
    epoch_level: log::Level,
    sfs_level: log::Level,
) {
    log::log!(target: target, epoch_level, "Finished epoch {epoch}");
    log::log!(target: target, sfs_level, "Current SFS: {fmt_sfs}");
}
