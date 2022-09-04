use std::{fs::File, io, num::NonZeroUsize, path::Path, thread};

use angsd_saf as saf;
use saf::version::Version;

use clap::CommandFactory;

use rand::{rngs::StdRng, SeedableRng};

use winsfs_core::{io::Intersect, saf::Saf};

use super::Cli;

pub fn init_logger(verbosity_arg: usize) -> clap::Result<()> {
    let level = match verbosity_arg {
        0 => return Ok(()),
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

/// Creates a new intersecting SAF reader with the provided number of threads.
pub fn setup_intersect<P, V>(
    paths: &[P],
    threads: usize,
) -> io::Result<Intersect<io::BufReader<File>, V>>
where
    P: AsRef<Path>,
    V: Version,
{
    let threads = NonZeroUsize::new(threads).unwrap_or(thread::available_parallelism()?);

    paths
        .iter()
        .map(|p| {
            saf::reader::Builder::<V>::default()
                .set_threads(threads)
                .build_from_member_path(p)
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Intersect::new)
}

pub fn shuffle_saf<const N: usize>(saf: &mut Saf<N>, seed: Option<u64>) {
    let mut rng = match seed {
        Some(v) => StdRng::seed_from_u64(v),
        None => StdRng::from_entropy(),
    };

    log::debug!(target: "init", "Shuffling SAF sites");

    saf.shuffle(&mut rng);
}
