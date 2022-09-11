use std::{
    io,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    process,
};

use clap::{error::Result as ClapResult, ArgGroup, Args};

use rayon::iter::{IndexedParallelIterator, ParallelIterator};

use winsfs_core::{
    em::{
        stopping::{LogLikelihoodTolerance, StoppingRule},
        Em, EmStep, StandardEm,
    },
    saf::{Blocks, SafView},
    sfs::{self, Multi, Sfs, USfs},
};

use crate::{input, utils::set_threads};

pub const DEFAULT_NUMBER_OF_SPLITS: usize = 50;
pub const DEFAULT_TOLERANCE: f64 = 1e-4;

/// Calculate SFS separately in smaller splits across input.
#[derive(Args, Debug)]
#[clap(group(ArgGroup::new("split")))]
pub struct Split {
    /// Input SAF file paths.
    ///
    /// For each set of SAF files (conventially named 'prefix'.{saf.idx,saf.pos.gz,saf.gz}),
    /// specify either the shared prefix or the full path to any one member file.
    /// Up to three SAF files currently supported.
    #[clap(
        value_parser,
        num_args = 1..=3,
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
    ///
    /// If set to 0, all available cores will be used.
    #[clap(short = 't', long, default_value_t = 4, value_name = "INT")]
    pub threads: usize,

    /// Log-likelihood difference tolerated between epochs before stopping.
    ///
    /// Each split SFS will run until the difference in successive log-likelihood values falls
    /// below the specified tolerance. Log-likelihood values are normalised by the number of sites
    /// in the block.
    #[clap(short = 'l', long, default_value_t = DEFAULT_TOLERANCE, value_name = "FLOAT")]
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
            _ => unreachable!(), // Checked by clap
        }
    }

    pub fn run_in_memory_n<const D: usize, P>(&self, paths: [P; D]) -> ClapResult<()>
    where
        P: AsRef<Path>,
    {
        let sfs = input::sfs::Reader::from_path(&self.sfs)?
            .read::<D>()?
            .normalise();

        let saf = input::saf::Readers::from_member_paths(&paths, self.threads)?.read_saf()?;

        let split_spec = get_split_spec(self, saf.sites());

        let sfs_split: Multi<USfs<D>> = saf
            .par_iter_blocks(split_spec)
            .enumerate()
            .map(|(i, block)| run_split(sfs.clone(), block, i + 1, self.tolerance))
            .collect::<Vec<_>>()
            .try_into()
            .expect("split SFS should all have the same shape");

        let stdout = io::stdout();
        let mut writer = stdout.lock();

        sfs::io::plain_text::write_multi_sfs(&mut writer, &sfs_split)?;

        Ok(())
    }
}

fn run_split<const D: usize>(
    initial_sfs: Sfs<D>,
    split: SafView<D>,
    i: usize,
    tolerance: f64,
) -> USfs<D> {
    let mut epoch = 1;
    let mut runner = StandardEm::<false>::new().inspect(move |_, _, sfs: &USfs<D>| {
        if sfs.iter().any(|x| x.is_nan()) {
            log::error!(
                target: "windowem",
                "Found NaN: this is a bug, and the process will abort, please file an issue"
            );

            process::exit(1);
        }

        log::debug!(target: "split", "Split {i}, finished epoch {epoch}");
        log::trace!(target: "split", "Split {i}, current SFS: {}", sfs.format_flat(" ", 6));
        epoch += 1;
    });

    let stop = LogLikelihoodTolerance::new(tolerance).inspect(|rule| {
        log::trace!(
            target: "stop",
            "Split {i}, current log-likelihood {lik:.4e}, Δ={diff:.4e} {sym} {tole:.4e}",
            lik = f64::from(rule.log_likelihood()),
            diff = rule.absolute_difference(),
            sym = if rule.absolute_difference() > rule.tolerance() { '>' } else { '≤' },
            tole = rule.tolerance(),
        );
    });

    let (_status, sfs) = runner.em(initial_sfs, &split, stop);

    log::info!(target: "split", "Split {i} finished");

    sfs.scale(split.sites() as f64)
}

fn get_split_spec(args: &Split, sites: usize) -> Blocks {
    let default = Blocks::Number(DEFAULT_NUMBER_OF_SPLITS);

    crate::utils::get_block_spec(args.splits, args.split_size, sites, default, "split")
}
