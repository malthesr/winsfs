use std::{io, num::NonZeroUsize, path::Path, process};

use clap::error::Result as ClapResult;

use winsfs_core::{
    em::{Em, EmStep, ParallelStandardEm, StandardEm, StreamingEm, WindowEm},
    io::shuffle::Reader,
    saf::Blocks,
    sfs::{self, Sfs},
};

use crate::{
    input,
    utils::{set_threads, shuffle_saf},
};

use super::Cli;

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
        let block_spec = get_block_spec($args, $sites);
        let approx_block_size = match block_spec {
            Blocks::Number(number) => $sites / number,
            Blocks::Size(size) => size,
        };
        let window_size = get_window_size($args.window_size).get();

        let mut block = 1;
        let block_runner = <$block_em>::new().inspect(move |_, _, sfs| {
            log::trace!(target: "windowem", "Finished block {block}");
            log::trace!(target: "windowem", "Current block SFS: {}", sfs.format_flat(" ", 6));
            block += 1
        });

        let (initial_sfs, runner) = match &$args.initial {
            Some(path) => {
                let initial_sfs = input::sfs::Reader::from_path(path)?.read()?.normalise();

                let initial_block_sfs = initial_sfs.clone().scale(approx_block_size as f64);
                let window_em = WindowEm::with_initial_sfs(
                    block_runner,
                    &initial_block_sfs,
                    window_size,
                    block_spec
                );

                (initial_sfs, window_em)
            },
            None => {
                log::debug!(target: "init", "Creating uniform initial SFS");

                let initial_sfs = Sfs::uniform($shape);
                let window_em = WindowEm::new(block_runner, window_size, block_spec);

                (initial_sfs, window_em)
            }
        };


        let mut epoch = 1;
        let runner = runner.inspect(move |_, _, sfs| {
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

        let (_status, sfs) = runner.em(initial_sfs, &saf.view(), stopping_rule);

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

pub fn get_block_spec(args: &Cli, sites: usize) -> Blocks {
    let spec = match (args.blocks, args.block_size) {
        (Some(number), None) => Blocks::Number(number.get()),
        (None, Some(block_size)) => Blocks::Size(block_size.get()),
        (None, None) => Blocks::Number(DEFAULT_NUMBER_OF_BLOCKS),
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
                    "Using {number} blocks, all containing {block_size} sites"
                );
            } else {
                log::debug!(
                    target: "init",
                    "Using {number} blocks, the first {rem} containing {} sites \
                    and the remaining blocks containing {block_size} sites",
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
                    "Using {blocks} blocks, all containing {size} sites",
                );
            } else {
                log::debug!(
                    target: "init",
                    "Using {} blocks, the first {blocks} containing {size} sites \
                    and the last block containing {rem} sites",
                    blocks + 1
                );
            }
        }
    }

    spec
}
