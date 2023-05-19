use std::{
    io,
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

use rayon::iter::{IndexedParallelIterator, ParallelIterator};

use clap::{error::Result as ClapResult, ArgGroup, Args};
use winsfs_core::{
    em::{stopping::LogLikelihoodTolerance, Em, StandardEm},
    sfs::io::plain_text,
};

use crate::{
    cli::MAX_PATHS,
    estimate::{get_block_spec, Checker, Logger},
    input,
    utils::set_threads,
};

const DEFAULT_NUMBER_OF_SPLITS: usize = 50;

/// Calculate SFS separately in smaller splits across input.
#[derive(Args, Debug)]
#[clap(group(ArgGroup::new("split")))]
pub struct Split {
    /// Input SAF file paths.
    ///
    /// For each set of SAF files (conventially named 'prefix'.{saf.idx,saf.pos.gz,saf.gz}),
    /// specify either the shared prefix or the full path to any one member file.
    /// Up to three SAF files currently supported (six with the experimental '--features hd' compile
    /// flag).
    #[clap(
        value_parser,
        num_args = 1..=MAX_PATHS,
        required = true,
        value_name = "PATHS"
    )]
    pub paths: Vec<PathBuf>,

    /// Number of splits.
    ///
    /// Note that due to the way splits are handled, the specified number of splits
    /// cannot always be created exactly, and the true number of splits may be very slightly
    /// different.
    #[clap(short = 'S', long, group = "split", value_name = "INT")]
    pub splits: Option<NonZeroUsize>,

    /// Number of sites per split.
    ///
    /// If both this and `--splits` are unset, the split size will be chosen so that approximately
    /// 50 splits are created.
    #[clap(short = 's', long, group = "split", value_name = "INT")]
    pub split_size: Option<NonZeroUsize>,

    /// Input global SFS to use for starting estimates.
    ///
    /// This can be calculated using the main `winsfs` command.
    #[clap(short = 'i', long, value_name = "PATH")]
    pub sfs: PathBuf,

    /// Number of threads to use.
    #[clap(short = 't', long, default_value_t = 4, value_name = "INT")]
    pub threads: usize,

    /// Log-likelihood difference tolerated between epochs before stopping.
    ///
    /// Each split SFS will run until the difference in successive log-likelihood values falls
    /// below the specified tolerance. Log-likelihood values are normalised by the number of sites
    /// in the block.
    #[clap(short = 'l', long, default_value_t = 1e-8, value_name = "FLOAT")]
    pub tolerance: f64,
}

impl Split {
    pub fn run(self) -> ClapResult<()> {
        // TODO: Implement a streaming version of this from shuffled input
        self.run_in_memory()
    }

    pub fn run_in_memory(&self) -> ClapResult<()> {
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

    pub fn run_in_memory_n<const D: usize, P>(&self, paths: [P; D]) -> ClapResult<()>
    where
        P: AsRef<Path>,
    {
        let initial_sfs = input::sfs::Reader::from_path(&self.sfs)?
            .read::<D>()?
            .normalise();
        let saf = input::saf::Readers::from_member_paths(&paths, self.threads)?.read_saf()?;

        let sites = saf.sites();
        let block_spec = get_block_spec(
            self.splits,
            self.split_size,
            sites,
            DEFAULT_NUMBER_OF_SPLITS,
        );

        let block_sfs = saf
            .view()
            .par_iter_blocks(block_spec)
            .enumerate()
            .map(|(i, block)| {
                let mut runner = Checker::new(
                    Logger::builder()
                        .log_counter_level(log::Level::Info)
                        .log_sfs_level(log::Level::Trace)
                        .log_target(format!("split {i}"))
                        .with_epoch_logging()
                        .build(StandardEm::<false, false>::new()),
                );

                let stopping_rule = LogLikelihoodTolerance::new(self.tolerance);

                let (_status, block_sfs) = runner
                    .em(initial_sfs.clone(), block, stopping_rule)
                    .unwrap();

                block_sfs.scale(block.sites() as f64)
            })
            .collect::<Vec<_>>();

        let mut stdout = io::stdout().lock();
        for sfs in block_sfs {
            plain_text::write_sfs(&mut stdout, &sfs)?;
        }

        Ok(())
    }
}
