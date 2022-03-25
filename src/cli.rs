use std::{num::NonZeroUsize, path::PathBuf};

use clap::{ArgGroup, Parser};

pub mod utils;

mod run;
pub use run::{run_1d, run_2d};

const NAME: &str = env!("CARGO_BIN_NAME");
const VERSION: &str = env!("CARGO_PKG_VERSION");
const AUTHOR: &str = env!("CARGO_PKG_AUTHORS");

/// Estimate site frequency spectrum using a window expectation-maximisation algorithm.
#[derive(Debug, Parser)]
#[clap(name = NAME, author = AUTHOR, version = VERSION, about)]
#[clap(group(ArgGroup::new("block")))]
pub struct Cli {
    #[clap(parse(from_os_str), max_values = 2, required = true)]
    pub paths: Vec<PathBuf>,

    /// Number of blocks.
    ///
    /// If both this and `--block-size` are unset,
    /// the block size will be chosen so that approximately 500 blocks are created.
    #[clap(short = 'B', long, group = "block")]
    pub blocks: Option<NonZeroUsize>,

    /// Number of sites per block.
    ///
    /// If both this and `--blocks` are unset,
    /// the block size will be chosen so that approximately 500 blocks are created.
    #[clap(short = 'b', long, group = "block")]
    pub block_size: Option<NonZeroUsize>,

    #[clap(long, hide = true)]
    pub debug: bool,

    /// Number of epochs to run.
    #[clap(short = 'n', long, default_value_t = 1)]
    pub epochs: usize,

    /// Initial SFS.
    ///
    /// If unset, a uniform SFS will be used to initialise optimisation.
    #[clap(short = 'i', long)]
    pub initial: Option<PathBuf>,

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

    /// Run vanilla EM.
    #[clap(long, hide = true, conflicts_with_all = &["blocks", "block-size", "window-size"])]
    pub vanilla: bool,

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

#[cfg(test)]
mod tests {
    use super::*;

    fn try_parse_args(cmd: &str) -> clap::Result<Cli> {
        Parser::try_parse_from(cmd.split_whitespace())
    }

    fn parse_args(cmd: &str) -> Cli {
        try_parse_args(cmd).expect("failed to parse command")
    }

    #[test]
    fn test_no_path_errors() {
        let result = try_parse_args("winsfs");

        assert_eq!(
            result.unwrap_err().kind(),
            clap::ErrorKind::MissingRequiredArgument
        );
    }

    #[test]
    fn test_three_paths_errors() {
        let result = try_parse_args("winsfs a b c");

        assert_eq!(result.unwrap_err().kind(), clap::ErrorKind::TooManyValues);
    }

    #[test]
    fn test_paths() {
        let args = parse_args("winsfs /path/to/saf");
        assert_eq!(args.paths, &[PathBuf::from("/path/to/saf")]);

        let args = parse_args("winsfs first second");
        assert_eq!(
            args.paths,
            &[PathBuf::from("first"), PathBuf::from("second")]
        );
    }

    #[test]
    fn test_block_group() {
        let args = parse_args("winsfs --blocks 10 /path/to/saf");
        assert_eq!(args.blocks.unwrap().get(), 10);

        let args = parse_args("winsfs --block-size 5 /path/to/saf");
        assert_eq!(args.block_size.unwrap().get(), 5);

        let args = parse_args("winsfs /path/to/saf");
        assert_eq!(args.blocks, None);
        assert_eq!(args.block_size, None);

        let result = try_parse_args("winsfs -b 5 -B 10 /path/to/saf");
        assert_eq!(
            result.unwrap_err().kind(),
            clap::ErrorKind::ArgumentConflict
        );
    }
}
