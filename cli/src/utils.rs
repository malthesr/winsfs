use std::thread;

use clap::CommandFactory;

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

    winsfs::set_threads(threads)
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
