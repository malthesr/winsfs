use std::{
    io::{self, Write},
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

use clap::{ArgGroup, Args};

use rayon::iter::{IndexedParallelIterator, ParallelIterator};

use winsfs_core::{
    em::{
        stopping::{LogLikelihoodTolerance, StoppingRule},
        Em, EmStep, StandardEm,
    },
    saf::SafView,
    sfs::{Sfs, UnnormalisedSfs},
};

use crate::{
    estimate::BlockSpecification,
    utils::{read_saf, read_sfs, set_threads},
};

pub const DEFAULT_BLOCK_TOLERANCE: f64 = 1e-4;
pub const DEFAULT_NUMBER_OF_SFS_BLOCKS: usize = 50;

/// Calculate SFS in smaller blocks across input.
#[derive(Args, Debug)]
#[clap(group(ArgGroup::new("block")))]
pub struct Blocks {
    /// Input SAF file paths.
    ///
    /// For each set of SAF files (conventially named 'prefix'.{saf.idx,saf.pos.gz,saf.gz}),
    /// specify either the shared prefix or the full path to any one member file.
    /// Up to three SAF files currently supported.
    #[clap(
        parse(from_os_str),
        max_values = 3,
        required = true,
        value_name = "PATHS"
    )]
    pub paths: Vec<PathBuf>,

    /// Number of blocks.
    ///
    /// Unlike the main `winsfs` command, this is not a hyperparameter but a number of blocks
    /// to split the data into for independent estimation.
    /// Note that due to the way blocks are handled, the specified number of blocks
    /// cannot always be created exactly, and the true number of blocks may be very slightly
    /// different.
    #[clap(short = 'B', long, group = "block", value_name = "INT")]
    pub blocks: Option<NonZeroUsize>,

    /// Number of sites per block.
    ///
    /// Unlike the main `winsfs` command, this is not a hyperparameter but a number of blocks
    /// to split the data into for independent estimation.
    /// If both this and `--blocks` are unset,
    /// the block size will be chosen so that approximately 50 blocks are created.
    #[clap(short = 'b', long, group = "block", value_name = "INT")]
    pub block_size: Option<NonZeroUsize>,

    /// Input global SFS to use for starting block estimates.
    ///
    /// This can be calculated using the main `winsfs` command.
    #[clap(short = 'i', long, value_name = "PATH")]
    pub sfs: PathBuf,

    /// Number of threads.
    ///
    /// If the provided value is less than or equal to zero, the number of threads used will be
    /// equal to the available threads minus the provided value.
    #[clap(short = 't', long, default_value_t = 4, value_name = "INT")]
    pub threads: i32,

    /// Log-likelihood difference tolerated between epochs before stopping.
    ///
    /// Each block SFS will run until the difference in successive log-likelihood values falls
    /// below the specified tolerance. Log-likelihood values are normalised by the number of sites
    /// in the block.
    #[clap(short = 'l', long, default_value_t = DEFAULT_BLOCK_TOLERANCE, value_name = "FLOAT")]
    pub tolerance: f64,
}

impl Blocks {
    pub fn run(self) -> clap::Result<()> {
        // TODO: Implement a streaming version of this from shuffled input
        self.run_in_memory()
    }

    pub fn run_in_memory(&self) -> clap::Result<()> {
        set_threads(self.threads)?;

        match &self.paths[..] {
            [p] => self.run_in_memory_n([p]),
            [p1, p2] => self.run_in_memory_n([p1, p2]),
            [p1, p2, p3] => self.run_in_memory_n([p1, p2, p3]),
            _ => unreachable!(), // Checked by clap
        }
    }

    pub fn run_in_memory_n<const N: usize, P>(&self, paths: [P; N]) -> clap::Result<()>
    where
        P: AsRef<Path>,
    {
        let sfs = read_sfs::<N, _>(&self.sfs)?;

        let saf = read_saf(paths)?;

        let block_size = BlockSpecification::from(self)
            .block_size(saf.sites())?
            .get();

        let block_sfs: Vec<UnnormalisedSfs<N>> = saf
            .par_iter_blocks(block_size)
            .enumerate()
            .map(|(i, block)| run_block(&sfs, block, i + 1, self.tolerance))
            .collect();

        let stdout = io::stdout();
        let mut writer = stdout.lock();

        for sfs in block_sfs {
            writeln!(writer, "{}", sfs.format_angsd(None))?;
        }

        Ok(())
    }
}

fn run_block<const N: usize>(
    initial_sfs: &Sfs<N>,
    block: SafView<N>,
    i: usize,
    tolerance: f64,
) -> UnnormalisedSfs<N> {
    let mut epoch = 1;
    let mut runner = StandardEm::<false>::new().inspect(move |_, _, sfs: &UnnormalisedSfs<N>| {
        log::debug!(target: "blocks", "Block {i}, finished epoch {epoch}");
        log::trace!(target: "blocks", "Block {i}, current SFS: {}", sfs.format_flat(" ", 6));
        epoch += 1;
    });

    let stop = LogLikelihoodTolerance::new(tolerance).inspect(|rule| {
        log::trace!(
            target: "stop",
            "Block {i}, current log-likelihood {lik:.4e}, Δ={diff:.4e} {sym} {tole:.4e}",
            lik = f64::from(rule.log_likelihood()),
            diff = rule.absolute_difference(),
            sym = if rule.absolute_difference() > rule.tolerance() { '>' } else { '≤' },
            tole = rule.tolerance(),
        );
    });

    let (_status, block_sfs) = runner.em(initial_sfs, &block, stop);

    log::info!(target: "blocks", "Block {i} finished");

    block_sfs.scale(block.sites() as f64)
}

impl From<&Blocks> for BlockSpecification {
    fn from(args: &Blocks) -> Self {
        match (args.blocks, args.block_size) {
            (Some(_), Some(_)) => unreachable!("checked by clap"),
            (Some(number_of_blocks), None) => Self::NumberOfBlocks(number_of_blocks),
            (None, Some(block_size)) => Self::BlockSize(block_size),
            (None, None) => {
                Self::NumberOfBlocks(NonZeroUsize::new(DEFAULT_NUMBER_OF_SFS_BLOCKS).unwrap())
            }
        }
    }
}
