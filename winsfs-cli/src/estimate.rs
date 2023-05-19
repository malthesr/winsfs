use std::{io, num::NonZeroUsize, path::Path};

use clap::error::Result as ClapResult;

use winsfs_core::{
    em::{stopping::Stop, Em, Sites, StandardEm, WindowEm},
    io::shuffle::Reader,
    saf::Blocks,
    sfs::{io::plain_text::write_sfs, Sfs},
};

use crate::{
    input,
    utils::{set_threads, shuffle_saf},
};

use super::Cli;

mod format;
pub use format::Format;

mod logging;
pub use logging::{Checker, Logger, LoggerBuilder};

mod stopping;
pub use stopping::Rule;

pub const DEFAULT_NUMBER_OF_BLOCKS: usize = 500;
pub const DEFAULT_TOLERANCE: f64 = 1e-4;
pub const DEFAULT_WINDOW_SIZE: usize = 100;

type Runner<const PAR: bool, const STREAM: bool> =
    Checker<Logger<WindowEm<Logger<StandardEm<PAR, STREAM>>, STREAM>>>;

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
            #[cfg(feature = "hd")]
            [p1, p2, p3, p4] => self.run_in_memory_n([p1, p2, p3, p4]),
            #[cfg(feature = "hd")]
            [p1, p2, p3, p4, p5] => self.run_in_memory_n([p1, p2, p3, p4, p5]),
            #[cfg(feature = "hd")]
            [p1, p2, p3, p4, p5, p6] => self.run_in_memory_n([p1, p2, p3, p4, p5, p6]),
            _ => unreachable!(), // Checked by clap
        }
    }

    fn run_n<I, const N: usize, const PAR: bool, const STREAM: bool>(
        &self,
        input: I,
        shape: [usize; N],
    ) -> ClapResult<()>
    where
        I: Sites,
        Runner<PAR, STREAM>: Em<N, I>,
        Rule: Stop<Runner<PAR, STREAM>>,
    {
        let sites = input.sites();
        let block_spec = get_block_spec(
            self.blocks,
            self.block_size,
            sites,
            DEFAULT_NUMBER_OF_BLOCKS,
        );
        let window_size = get_window_size(self.window_size).get();

        let (initial_sfs, mut runner) = setup::<_, N, PAR, STREAM>(
            self.initial.as_ref(),
            shape,
            sites,
            window_size,
            block_spec,
        )?;
        let stopping_rule = Rule::from(self);

        let (_status, sfs) = runner.em(initial_sfs, input, stopping_rule).unwrap();

        let stdout = io::stdout();
        let mut writer = stdout.lock();
        write_sfs(&mut writer, &sfs.scale(sites as f64))?;

        Ok(())
    }

    fn run_in_memory_n<const N: usize, P>(&self, paths: [P; N]) -> ClapResult<()>
    where
        P: AsRef<Path>,
    {
        let mut saf = input::saf::Readers::from_member_paths(&paths, self.threads)?.read_saf()?;
        shuffle_saf(&mut saf, self.seed);

        self.run_n::<_, N, true, false>(saf.view(), saf.shape())
    }

    fn run_streaming(&self) -> ClapResult<()> {
        if let [path] = &self.paths[..] {
            log::info!(
                target: "init",
                "Streaming through shuffled SAF file from path:\n\t{}",
                path.display()
            );

            let reader = Reader::try_from_path(path)?;
            let dim = reader.header().shape().len();

            match dim {
                1 => self.run_streaming_n::<1, _>(reader),
                2 => self.run_streaming_n::<2, _>(reader),
                3 => self.run_streaming_n::<3, _>(reader),
                #[cfg(feature = "hd")]
                4 => self.run_streaming_n::<4, _>(reader),
                #[cfg(feature = "hd")]
                5 => self.run_streaming_n::<5, _>(reader),
                #[cfg(feature = "hd")]
                6 => self.run_streaming_n::<6, _>(reader),
                #[cfg(feature = "hd")]
                _ => unimplemented!("only dimensions up to six currently supported"),
                #[cfg(not(feature = "hd"))]
                _ => unimplemented!(
                    "only dimensions up to three currently supported - \
                    recompile with the '--features hd' flag for dimensions up to six"
                ),
            }
        } else {
            // Checked and handled properly in format inference
            unreachable!("cannot run streaming with multiple input files")
        }
    }

    fn run_streaming_n<const N: usize, R>(&self, mut reader: Reader<R>) -> ClapResult<()>
    where
        R: io::BufRead + io::Seek,
    {
        let shape = reader.header().shape().to_vec().try_into().unwrap();
        self.run_n::<_, N, false, true>(&mut reader, shape)
    }
}

fn setup<P, const D: usize, const PAR: bool, const STREAM: bool>(
    sfs_path: Option<P>,
    shape: [usize; D],
    sites: usize,
    window_size: usize,
    block_spec: Blocks,
) -> ClapResult<(Sfs<D>, Runner<PAR, STREAM>)>
where
    P: AsRef<Path>,
{
    let block_runner = Logger::builder()
        .log_counter_level(log::Level::Trace)
        .log_sfs_level(log::Level::Trace)
        .log_target("windowem")
        .with_block_logging()
        .build(StandardEm::<PAR, STREAM>::new());

    let (sfs, runner) = if let Some(path) = sfs_path {
        let sfs = input::sfs::Reader::from_path(path)?.read()?;

        let approx_block_size = match block_spec {
            Blocks::Number(number) => sites / number,
            Blocks::Size(size) => size,
        };
        let block_sfs = sfs.clone().normalise().scale(approx_block_size as f64);

        let runner = WindowEm::<_, STREAM>::with_initial_sfs(
            block_runner,
            &block_sfs,
            window_size,
            block_spec,
        );
        (sfs.normalise(), runner)
    } else {
        log::debug!(target: "init", "Creating uniform initial SFS");

        let sfs = Sfs::uniform(shape);
        let runner = WindowEm::<_, STREAM>::new(block_runner, window_size, block_spec);
        (sfs, runner)
    };

    let runner = Logger::builder()
        .log_counter_level(log::Level::Info)
        .log_sfs_level(log::Level::Debug)
        .log_target("windowem")
        .with_epoch_logging()
        .build(runner);

    Ok((sfs, Checker::new(runner)))
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

pub fn get_block_spec(
    blocks: Option<NonZeroUsize>,
    block_size: Option<NonZeroUsize>,
    sites: usize,
    default_number_of_blocks: usize,
) -> Blocks {
    let spec = match (blocks, block_size) {
        (Some(number), None) => Blocks::Number(number.get()),
        (None, Some(block_size)) => Blocks::Size(block_size.get()),
        (None, None) => Blocks::Number(default_number_of_blocks),
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
