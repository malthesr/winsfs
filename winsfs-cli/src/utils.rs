use std::num::NonZeroUsize;

use clap::{
    error::{ErrorKind, Result as ClapResult},
    CommandFactory,
};

use rand::{rngs::StdRng, SeedableRng};

use winsfs_core::saf::{Blocks, Saf};

use super::Cli;

pub fn init_logger(verbosity_arg: u8) -> ClapResult<()> {
    let level = match verbosity_arg {
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        2 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };

    simple_logger::SimpleLogger::new()
        .with_level(level)
        .init()
        .map_err(|_| Cli::command().error(ErrorKind::Io, "Failed to initialise logger"))
}

pub fn set_threads(threads: usize) -> ClapResult<()> {
    winsfs_core::set_threads(threads)
        .map_err(|_| Cli::command().error(ErrorKind::Io, "Failed to initialise threadpool"))
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

pub fn get_block_spec(
    blocks: Option<NonZeroUsize>,
    block_size: Option<NonZeroUsize>,
    sites: usize,
    default: Blocks,
    name: &'static str,
) -> Blocks {
    let spec = match (blocks, block_size) {
        (Some(number), None) => Blocks::Number(number.get()),
        (None, Some(block_size)) => Blocks::Size(block_size.get()),
        (None, None) => default,
        (Some(_), Some(_)) => unreachable!("checked by clap"),
    };

    // We log the block spec with some precision: it's useful information to output, and also
    // helpful for debugging.
    match spec {
        Blocks::Number(number) => {
            let block_size = sites / number;
            let rem = sites % number;
            if rem == 0 {
                log::debug!(
                    target: "init",
                    "Using {number} {name}s, all containing {block_size} sites"
                );
            } else {
                log::debug!(
                    target: "init",
                    "Using {number} {name}s, the first {rem} containing {} sites \
                    and the remaining {name}s containing {block_size} sites",
                    block_size + 1
                );
            }
        }
        Blocks::Size(size) => {
            let rem = sites % size;
            let blocks = sites / size;
            if rem == 0 {
                log::debug!(
                    target: "init",
                    "Using {blocks} {name}s, all containing {size} sites",
                );
            } else {
                log::debug!(
                    target: "init",
                    "Using {} {name}s, the first {blocks} containing {size} sites \
                    and the last {name} containing {rem} sites",
                    blocks + 1
                );
            }
        }
    }

    spec
}
