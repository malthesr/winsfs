use std::{io, num::NonZeroUsize, path::Path, process};

use clap::error::Result as ClapResult;

use winsfs_core::{
    em::{Em, EmStep, ParallelStandardEm, StandardEm, StreamingEm, Window, WindowEm},
    io::shuffle::Reader,
    sfs::{self, Sfs},
};

use crate::{
    input,
    utils::{set_threads, shuffle_saf},
};

use super::Cli;

mod blocks;
pub use blocks::BlockSpecification;

mod format;
pub use format::Format;

mod stopping;
pub use stopping::Rule;

pub const DEFAULT_NUMBER_OF_BLOCKS: usize = 500;
pub const DEFAULT_TOLERANCE: f64 = 1e-4;
pub const DEFAULT_WINDOW_SIZE: usize = 100;

// This could be a function, except the return type cannot be named;
// using an opaque type does not help, since a different trait impl is required for
// EM and streaming EM, and this depend on the block EM type param - hence the macro
macro_rules! setup {
    ($args:expr, $sites:expr, $shape:expr, $block_em:ty) => {{
        let block_size = BlockSpecification::from($args).block_size($sites)?.get();
        let window_size = get_window_size($args.window_size).get();

        let (initial_sfs, window) =
            get_initial_sfs_and_window($args.initial.as_ref(), $shape, block_size, window_size)?;

        let mut block = 1;
        let block_runner = <$block_em>::new().inspect(move |_, _, sfs| {
            log::trace!(target: "windowem", "Finished block {block}");
            log::trace!(target: "windowem", "Current block SFS: {}", sfs.format_flat(" ", 6));
            block += 1
        });

        let mut epoch = 1;
        let runner = WindowEm::new(block_runner, window, block_size).inspect(move |_, _, sfs| {
            if sfs.iter().any(|x| x.is_nan()) {
                log::error!(
                    target: "windowem",
                    "Found NaN: this is a bug, and the process will abort, please file an issue"
                );

                process::exit(1);
            }

            log::info!(target: "windowem", "Finished epoch {epoch}");
            log::debug!(target: "windowem", "Current SFS: {}", sfs.format_flat(" ", 6));
            epoch += 1;
        });

        (initial_sfs, runner)
    }}
}

impl Cli {
    pub fn run(self) -> ClapResult<()> {
        match Format::try_from(&self)? {
            Format::Standard | Format::Banded => self.run_in_memory(),
            Format::Shuffled => self.run_streaming(),
        }
    }

    fn run_in_memory(self) -> ClapResult<()> {
        set_threads(self.threads)?;

        match &self.paths[..] {
            [p] => self.run_in_memory_n([p]),
            [p1, p2] => self.run_in_memory_n([p1, p2]),
            [p1, p2, p3] => self.run_in_memory_n([p1, p2, p3]),
            _ => unreachable!(), // Checked by clap
        }
    }

    fn run_in_memory_n<const N: usize, P>(&self, paths: [P; N]) -> ClapResult<()>
    where
        P: AsRef<Path>,
    {
        let mut saf = input::saf::Readers::from_member_paths(&paths, self.threads)?.read_saf()?;
        shuffle_saf(&mut saf, self.seed);
        let sites = saf.sites();
        let shape = saf.shape();

        let (initial_sfs, mut runner) = setup!(self, sites, shape, ParallelStandardEm);
        let stopping_rule = Rule::from(self);

        let (_status, sfs) = runner.em(initial_sfs, &saf, stopping_rule);

        let stdout = io::stdout();
        let mut writer = stdout.lock();
        sfs::io::plain_text::write_sfs(&mut writer, &sfs.scale(sites as f64))?;

        Ok(())
    }

    fn run_streaming(&self) -> ClapResult<()> {
        if let [path] = &self.paths[..] {
            log::info!(
                target: "init",
                "Streaming through shuffled SAF file from path:\n\t{}",
                path.display()
            );

            if let Ok(reader) = Reader::<1, _>::try_from_path(path) {
                self.run_streaming_n::<1, _>(reader)
            } else if let Ok(reader) = Reader::<2, _>::try_from_path(path) {
                self.run_streaming_n::<2, _>(reader)
            } else if let Ok(reader) = Reader::<3, _>::try_from_path(path) {
                self.run_streaming_n::<3, _>(reader)
            } else {
                unimplemented!("only dimensions up to three currently supported")
            }
        } else {
            // Checked and handled properly in format inference
            unreachable!("cannot run streaming with multiple input files")
        }
    }

    fn run_streaming_n<const D: usize, R>(&self, mut reader: Reader<D, R>) -> ClapResult<()>
    where
        R: io::BufRead + io::Seek,
    {
        let header = reader.header();
        let sites = header.sites();
        let shape: [usize; D] = header.shape().to_vec().try_into().unwrap();

        let (initial_sfs, mut runner) = setup!(self, sites, shape, StandardEm);
        let stopping_rule = Rule::from(self);

        let (_status, sfs) = runner.stream_em(initial_sfs, &mut reader, stopping_rule)?;

        let stdout = io::stdout();
        let mut writer = stdout.lock();
        sfs::io::plain_text::write_sfs(&mut writer, &sfs.scale(sites as f64))?;

        Ok(())
    }
}

fn get_window_size(window_size: Option<NonZeroUsize>) -> NonZeroUsize {
    let window_size = match window_size {
        Some(v) => v,
        None => NonZeroUsize::new(DEFAULT_WINDOW_SIZE).unwrap(),
    };

    log::debug!(
        target: "init",
        "Using window size of {window_size} blocks per window"
    );

    window_size
}

fn get_initial_sfs_and_window<const N: usize, P>(
    initial_sfs_path: Option<P>,
    shape: [usize; N],
    block_size: usize,
    window_size: usize,
) -> io::Result<(Sfs<N>, Window<N>)>
where
    P: AsRef<Path>,
{
    Ok(match initial_sfs_path {
        Some(path) => {
            let initial_sfs = input::sfs::Reader::from_path(path)?.read()?.normalise();

            // The initial block SFSs should be scaled by block size for weighting
            let initial_block_sfs = initial_sfs.clone().scale(block_size as f64);
            let window = Window::from_initial(initial_block_sfs, window_size);

            (initial_sfs, window)
        }
        None => {
            log::debug!(target: "init", "Creating uniform initial SFS");

            let initial_sfs = Sfs::uniform(shape);
            let window = Window::from_zeros(shape, window_size);

            (initial_sfs, window)
        }
    })
}
