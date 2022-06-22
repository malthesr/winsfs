use std::{io, path::Path, thread};

use clap::CommandFactory;

use rand::{rngs::StdRng, SeedableRng};

use winsfs_core::{saf::Saf, sfs::Sfs};

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

pub fn read_saf<const N: usize, P>(paths: [P; N]) -> io::Result<Saf<N>>
where
    P: AsRef<Path>,
{
    log::info!(
        target: "init",
        "Reading (intersecting) sites in input SAF files:\n\t{}",
        join(paths.iter().map(|p| p.as_ref().display()), "\n\t"),
    );

    let saf = Saf::read_from_paths(paths)?;

    log::debug!(
        target: "init",
        "Found {sites} (intersecting) sites in SAF files with shape {shape}",
        sites = saf.sites(),
        shape = join(saf.shape(), "/"),
    );

    Ok(saf)
}

pub fn read_sfs<const N: usize, P>(path: P) -> io::Result<Sfs<N>>
where
    P: AsRef<Path>,
{
    log::debug!(
        target: "init",
        "Reading SFS from path:\n\t{}",
        path.as_ref().display()
    );

    Sfs::read_from_angsd(path).map(Sfs::normalise)
}

pub fn shuffle_saf<const N: usize>(saf: &mut Saf<N>, seed: Option<u64>) {
    let mut rng = match seed {
        Some(v) => StdRng::seed_from_u64(v),
        None => StdRng::from_entropy(),
    };

    log::debug!(target: "init", "Shuffling SAF sites");

    saf.shuffle(&mut rng);
}
