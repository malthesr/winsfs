use std::{io, num::NonZeroUsize, thread};

use clap::CommandFactory;

use rand::{rngs::StdRng, SeedableRng};

use super::{Cli, Format};

const DEFAULT_NUMBER_OF_BLOCKS: usize = 500;
const DEFAULT_BLOCKS_TO_WINDOWS: usize = 5;

pub fn infer_format<R>(reader: &mut R) -> io::Result<Option<Format>>
where
    R: io::Read + io::Seek,
{
    const MAGIC_NUMBER_LEN: usize = angsd_io::saf::MAGIC_NUMBER.len();

    let mut buf = [0; MAGIC_NUMBER_LEN];
    reader.read_exact(&mut buf)?;
    reader.seek(io::SeekFrom::Current(-(MAGIC_NUMBER_LEN as i64)))?;

    Ok(match &buf {
        angsd_io::saf::MAGIC_NUMBER => Some(Format::Standard),
        crate::io::MAGIC_NUMBER => Some(Format::Shuffled),
        _ => None,
    })
}

pub fn get_block_size_and_blocks(
    block_size_arg: Option<NonZeroUsize>,
    blocks_arg: Option<NonZeroUsize>,
    sites: usize,
) -> (usize, usize) {
    match (block_size_arg.map(|x| x.get()), blocks_arg.map(|x| x.get())) {
        (Some(_), Some(_)) => {
            unreachable!("clap checks '--blocks' and '--block-size' conflict")
        }
        (Some(size), None) => (size, get_blocks(size, sites)),
        (None, Some(blocks)) => (get_block_size(blocks, sites), blocks),
        (None, None) => {
            let block_size = default_block_size(sites);
            let blocks = get_blocks(block_size, sites);
            (block_size, blocks)
        }
    }
}

fn div_ceil(lhs: usize, rhs: usize) -> usize {
    if lhs % rhs == 0 {
        lhs / rhs
    } else {
        lhs / rhs + 1
    }
}

fn get_blocks(block_size: usize, sites: usize) -> usize {
    div_ceil(sites, block_size)
}

fn get_block_size(blocks: usize, sites: usize) -> usize {
    div_ceil(sites, blocks)
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

pub fn default_block_size(sites: usize) -> usize {
    (sites.rem_euclid(DEFAULT_NUMBER_OF_BLOCKS) + sites) / DEFAULT_NUMBER_OF_BLOCKS
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
