use std::{num::NonZeroUsize, thread};

use clap::CommandFactory;

use rand::{rngs::StdRng, SeedableRng};

use super::Cli;
const DEFAULT_NUMBER_OF_BLOCKS: usize = 500;
const DEFAULT_BLOCKS_TO_WINDOWS: usize = 5;

pub fn get_blocks(block_size: usize, sites: usize) -> usize {
    if sites.rem_euclid(block_size) == 0 {
        sites / block_size
    } else {
        sites / block_size + 1
    }
}

pub fn get_block_size(block_size_arg: Option<NonZeroUsize>, sites: usize) -> usize {
    match block_size_arg {
        Some(v) => v.get(),
        None => (sites.rem_euclid(DEFAULT_NUMBER_OF_BLOCKS) + sites) / DEFAULT_NUMBER_OF_BLOCKS,
    }
}

pub fn get_window_size(window_size_arg: Option<NonZeroUsize>, blocks: usize) -> usize {
    match window_size_arg {
        Some(v) => v.get(),
        None => match blocks / DEFAULT_BLOCKS_TO_WINDOWS {
            0 => 1,
            v => v,
        },
    }
}

pub fn get_rng(seed_arg: Option<u64>) -> StdRng {
    match seed_arg {
        Some(v) => StdRng::seed_from_u64(v),
        None => StdRng::from_entropy(),
    }
}
pub fn init_logger(verbosity_arg: usize) -> clap::Result<()> {
    let level = match verbosity_arg {
        0 => return Ok(()),
        1 => log::LevelFilter::Info,
        2 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };

    simple_logger::SimpleLogger::new()
        .with_level(level)
        .with_colors(true)
        .init()
        .map_err(|_| Cli::command().error(clap::ErrorKind::Io, "Failed to initialise logger"))
}

pub fn set_threads(thread_arg: i32) -> clap::Result<()> {
    // If value is non-positive, set to available minus value.
    let threads = match usize::try_from(thread_arg) {
        Ok(0) => thread::available_parallelism()?.get(),
        Ok(v) => v,
        Err(_) => {
            let available = thread::available_parallelism()?.get();

            let subtract = usize::try_from(thread_arg.abs()).map_err(|_| {
                Cli::command().error(
                    clap::ErrorKind::ValueValidation,
                    "Cannot convert number of threads to usize",
                )
            })?;

            available.checked_sub(subtract).unwrap_or(1)
        }
    };

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .map_err(|_| Cli::command().error(clap::ErrorKind::Io, "Failed to initialise threadpool"))
}
