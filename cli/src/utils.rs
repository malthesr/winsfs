use std::thread;

use clap::CommandFactory;

use rand::{rngs::StdRng, SeedableRng};

use super::Cli;

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

pub fn validate_shape<const N: usize>(shape: [usize; N], expected: [usize; N]) -> clap::Result<()> {
    match shape == expected {
        true => Ok(()),
        false => {
            let msg = format!(
                "Shape of provided SFS ({}) does not match SAFs ({})",
                format_shape(shape),
                format_shape(expected)
            );
            Err(Cli::command().error(clap::ErrorKind::ValueValidation, msg))
        }
    }
}

fn format_shape<const N: usize>(shape: [usize; N]) -> String {
    shape.map(|x| x.to_string()).join("/")
}
