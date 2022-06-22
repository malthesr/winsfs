use std::{io, num::NonZeroUsize, path::Path};

use clap::CommandFactory;

use winsfs_core::{
    em::{Em, EmStep, ParallelStandardEm, StandardEm, StreamingEm, Window, WindowEm},
    io::shuffle::Reader,
    sfs::Sfs,
};

use crate::utils::{read_saf, read_sfs, set_threads, shuffle_saf};

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
        let sites = NonZeroUsize::new($sites).ok_or_else(|| {
            Cli::command().error(clap::ErrorKind::Io, "input contains 0 (intersecting) sites")
        })?;
        let block_size = BlockSpecification::from($args).block_size(sites).get();
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
            log::info!(target: "windowem", "Finished epoch {epoch}");
            log::debug!(target: "windowem", "Current SFS: {}", sfs.format_flat(" ", 6));
            epoch += 1;
        });

        (initial_sfs, runner)
    }}
}

impl Cli {
    pub fn run(self) -> clap::Result<()> {
        match Format::try_from(&self)? {
            Format::Standard => self.run_in_memory(),
            Format::Shuffled => self.run_streaming(),
        }
    }

    fn run_in_memory(self) -> clap::Result<()> {
        set_threads(self.threads)?;

        match &self.paths[..] {
            [p] => self.run_in_memory_n([p]),
            [p1, p2] => self.run_in_memory_n([p1, p2]),
            [p1, p2, p3] => self.run_in_memory_n([p1, p2, p3]),
            _ => unreachable!(), // Checked by clap
        }
    }

    fn run_in_memory_n<const N: usize, P>(&self, paths: [P; N]) -> clap::Result<()>
    where
        P: AsRef<Path>,
    {
        let mut saf = read_saf(paths)?;
        shuffle_saf(&mut saf, self.seed);
        let sites = saf.sites();
        let shape = saf.shape();

        let (initial_sfs, mut runner) = setup!(self, sites, shape, ParallelStandardEm);
        let stopping_rule = Rule::from(self);

        let (_status, sfs) = runner.em(&initial_sfs, &saf, stopping_rule);

        println!("{}", sfs.scale(sites as f64).format_angsd(None));

        Ok(())
    }

    fn run_streaming(&self) -> clap::Result<()> {
        let reader = match &self.paths[..] {
            [p] => {
                log::info!(
                    target: "init",
                    "Streaming through shuffled SAF file from path:\n\t{}",
                    p.display()
                );

                Reader::from_path(p)?
            }
            // Checked and handled properly in format inference
            _ => unreachable!("cannot run streaming with multiple input files"),
        };

        let n = reader.header().shape().len();
        match n {
            1 => self.run_streaming_n::<1, _>(reader),
            2 => self.run_streaming_n::<2, _>(reader),
            3 => self.run_streaming_n::<3, _>(reader),
            _ => unimplemented!("only dimensions up to three currently supported"),
        }
    }

    fn run_streaming_n<const N: usize, R>(&self, mut reader: Reader<R>) -> clap::Result<()>
    where
        R: io::BufRead + io::Seek,
    {
        let header = reader.header();
        let sites = header.sites();
        let shape: [usize; N] = header.shape().to_vec().try_into().unwrap();

        let (initial_sfs, mut runner) = setup!(self, sites, shape, StandardEm);
        let stopping_rule = Rule::from(self);

        let (_status, sfs) = runner.stream_em(&initial_sfs, &mut reader, stopping_rule)?;

        println!("{}", sfs.scale(sites as f64).format_angsd(None));

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
            let initial_sfs = read_sfs(path)?;

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
