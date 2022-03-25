use std::{num::NonZeroUsize, path::PathBuf};

use clap::{ArgGroup, Parser, Subcommand};

mod utils;
use utils::{init_logger, set_threads};

mod log_likelihood;
use log_likelihood::LogLikelihood;

mod run;
use run::{run_1d, run_2d};

const NAME: &str = env!("CARGO_BIN_NAME");
const VERSION: &str = env!("CARGO_PKG_VERSION");
const AUTHOR: &str = env!("CARGO_PKG_AUTHORS");

/// Estimate site frequency spectrum using a window expectation-maximisation algorithm.
#[derive(Debug, Parser)]
#[clap(name = NAME, author = AUTHOR, version = VERSION, about)]
#[clap(group(ArgGroup::new("block")))]
#[clap(args_conflicts_with_subcommands = true, subcommand_negates_reqs = true)]
pub struct Cli {
    /// Input SAF file paths.
    ///
    /// For each set of SAF files (conventially named [prefix].{saf.idx,saf.pos.gz,saf.gz}),
    /// specify either the shared prefix or the full path to any one member file.
    /// Up to two SAF files currently supported.
    #[clap(
        parse(from_os_str),
        max_values = 2,
        required = true,
        value_name = "PATHS"
    )]
    pub paths: Vec<PathBuf>,

    /// Number of blocks.
    ///
    /// If both this and `--block-size` are unset,
    /// the block size will be chosen so that approximately 500 blocks are created.
    #[clap(short = 'B', long, group = "block", value_name = "INT")]
    pub blocks: Option<NonZeroUsize>,

    /// Number of sites per block.
    ///
    /// If both this and `--blocks` are unset,
    /// the block size will be chosen so that approximately 500 blocks are created.
    #[clap(short = 'b', long, group = "block", value_name = "INT")]
    pub block_size: Option<NonZeroUsize>,

    #[clap(long, hide = true, global = true)]
    pub debug: bool,

    /// Number of epochs to run.
    #[clap(short = 'n', long, default_value_t = 1, value_name = "INT")]
    pub epochs: usize,

    /// Initial SFS.
    ///
    /// If unset, a uniform SFS will be used to initialise optimisation.
    #[clap(short = 'i', long, value_name = "PATH")]
    pub initial: Option<PathBuf>,

    /// Random seed.
    ///
    /// If unset, a seed will be chosen at random.
    #[clap(short = 's', long, value_name = "INT")]
    pub seed: Option<u64>,

    /// Number of threads.
    ///
    /// If the provided value is less than or equal to zero, the number of threads used will be
    /// equal to the available threads minus the provided value.
    #[clap(short = 't', long, default_value_t = 4, value_name = "INT")]
    pub threads: i32,

    /// Run vanilla EM.
    #[clap(long, hide = true, conflicts_with_all = &["blocks", "block-size", "window-size"])]
    pub vanilla: bool,

    /// Verbosity.
    ///
    /// Flag can be set multiply times to increase verbosity, or left unset for quiet mode.
    #[clap(short = 'v', long, parse(from_occurrences), global = true)]
    pub verbose: usize,

    /// Number of blocks per window.
    ///
    /// If unset, the window size will be chosen as approximately 1/5 of the number of blocks.
    #[clap(short = 'w', long, value_name = "INT")]
    pub window_size: Option<NonZeroUsize>,

    #[clap(subcommand)]
    pub subcommand: Option<Command>,
}

impl Cli {
    pub fn run(self) -> clap::Result<()> {
        init_logger(self.verbose)?;
        set_threads(self.threads)?;

        if let Some(subcommand) = self.subcommand {
            subcommand.run()
        } else {
            match self.paths.as_slice() {
                [path] => run_1d(path, &self),
                [first_path, second_path] => run_2d(first_path, second_path, &self),
                _ => unreachable!(), // Checked by clap
            }
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum Command {
    LogLikelihood(LogLikelihood),
}

impl Command {
    fn run(self) -> Result<(), clap::Error> {
        match self {
            Command::LogLikelihood(log_likelihood) => log_likelihood.run(),
        }
    }
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

    #[test]
    fn test_subcommand_conflicts_with_args() {
        let result = try_parse_args("winsfs -b 5 log-likelihood --sfs /path/to/sfs /path/to/saf");
        assert_eq!(result.unwrap_err().kind(), clap::ErrorKind::UnknownArgument,);
    }

    #[test]
    fn test_subcommand_verbosity() {
        let args = parse_args("winsfs log-likelihood -vv --sfs /path/to/sfs /path/to/saf");
        assert_eq!(args.verbose, 2);
    }
}
