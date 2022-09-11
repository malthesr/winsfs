use clap::CommandFactory;

use rand::{rngs::StdRng, SeedableRng};

use winsfs_core::saf::Saf;

use super::Cli;

pub fn init_logger(verbosity_arg: usize) -> clap::Result<()> {
    let level = match verbosity_arg {
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        2 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };

    simple_logger::SimpleLogger::new()
        .with_level(level)
        .init()
        .map_err(|_| Cli::command().error(clap::ErrorKind::Io, "Failed to initialise logger"))
}

pub fn set_threads(threads: usize) -> clap::Result<()> {
    winsfs_core::set_threads(threads)
        .map_err(|_| Cli::command().error(clap::ErrorKind::Io, "Failed to initialise threadpool"))
}

pub fn join<I, T>(iter: I, sep: &str) -> String
where
    I: IntoIterator<Item = T>,
    T: ToString,
{
    iter.into_iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(sep)
}

pub fn shuffle_saf<const N: usize>(saf: &mut Saf<N>, seed: Option<u64>) {
    let mut rng = match seed {
        Some(v) => StdRng::seed_from_u64(v),
        None => StdRng::from_entropy(),
    };

    log::debug!(target: "init", "Shuffling SAF sites");

    saf.shuffle(&mut rng);
}
