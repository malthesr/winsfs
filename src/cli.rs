use std::{num::NonZeroUsize, path::PathBuf};

use clap::Parser;

pub mod utils;

mod run;
pub use run::{run_1d, run_2d};

const NAME: &str = env!("CARGO_BIN_NAME");
const VERSION: &str = env!("CARGO_PKG_VERSION");
const AUTHOR: &str = env!("CARGO_PKG_AUTHORS");

/// Estimate site frequency spectrum using a window expectation-maximisation algorithm.
#[derive(Parser)]
#[clap(name = NAME, author = AUTHOR, version = VERSION, about)]
pub struct Cli {
    #[clap(parse(from_os_str), min_values = 1, max_values = 2)]
    pub paths: Vec<PathBuf>,

    /// Number of sites per block.
    ///
    /// If unset, the block size will be chosen so that approximately 500 blocks are created.
    #[clap(short = 'b', long)]
    pub block_size: Option<NonZeroUsize>,

    /// Number of epochs to run.
    #[clap(short = 'n', long, default_value_t = 1)]
    pub epochs: usize,

    /// Random seed.
    ///
    /// If unset, a seed will be chosen at random.
    #[clap(short = 's', long)]
    pub seed: Option<u64>,

    /// Number of threads.
    ///
    /// If the provided value is less than or equal to zero, the number of threads used will be
    /// equal to the available threads minus the provided value.
    #[clap(short = 't', long, default_value_t = 4)]
    pub threads: i32,

    /// Verbosity.
    ///
    /// Flag can be set multiply times to increase verbosity, or left unset for quiet mode.
    #[clap(short = 'v', long, parse(from_occurrences))]
    pub verbose: usize,

    /// Number of blocks per window.
    ///
    /// If unset, the window size will be chosen as approximately 1/5 of the number of blocks.
    #[clap(short = 'w', long)]
    pub window_size: Option<NonZeroUsize>,
}
