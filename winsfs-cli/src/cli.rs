use std::{num::NonZeroUsize, path::PathBuf};

use clap::{ArgGroup, Parser, Subcommand};

use crate::{estimate::Format, LogLikelihood, Shuffle};

const NAME: &str = env!("CARGO_BIN_NAME");
const VERSION: &str = env!("CARGO_PKG_VERSION");
const AUTHOR: &str = env!("CARGO_PKG_AUTHORS");

/// Estimate site frequency spectrum using a window expectation-maximisation algorithm.
#[derive(Debug, Parser)]
#[clap(name = NAME, author = AUTHOR, version = VERSION, about)]
#[clap(group(ArgGroup::new("block")))]
#[clap(args_conflicts_with_subcommands = true, subcommand_negates_reqs = true)]
#[clap(next_help_heading = "GENERAL")]
pub struct Cli {
    /// Input SAF file paths.
    ///
    /// For each set of SAF files (conventially named 'prefix'.{saf.idx,saf.pos.gz,saf.gz}),
    /// specify either the shared prefix or the full path to any one member file.
    /// Up to three SAF files currently supported.
    #[clap(
        parse(from_os_str),
        max_values = 3,
        required = true,
        help_heading = "INPUT",
        value_name = "PATHS"
    )]
    pub paths: Vec<PathBuf>,

    /// Number of blocks.
    ///
    /// If both this and `--block-size` are unset,
    /// the block size will be chosen so that approximately 500 blocks are created.
    /// Note that due to the way blocks are handled, the specified number of blocks
    /// cannot always be created exactly, and the true number of blocks may be very slightly
    /// different.
    #[clap(
        short = 'B',
        long,
        group = "block",
        help_heading = "HYPERPARAMETERS",
        value_name = "INT"
    )]
    pub blocks: Option<NonZeroUsize>,

    /// Number of sites per block.
    ///
    /// If both this and `--blocks` are unset,
    /// the block size will be chosen so that approximately 500 blocks are created.
    #[clap(
        short = 'b',
        long,
        group = "block",
        help_heading = "HYPERPARAMETERS",
        value_name = "INT"
    )]
    pub block_size: Option<NonZeroUsize>,

    #[clap(long, hide = true, global = true)]
    pub debug: bool,

    /// Maximum number of epochs to run.
    ///
    /// If both this and `--tolerance` are unset, the default stopping rule is a log-likelihood
    /// tolerance of 1e-4. If both are set, the first stopping rule to be triggered will stop the
    /// algorithm.
    #[clap(long, help_heading = "STOPPING", value_name = "INT")]
    pub max_epochs: Option<usize>,

    /// Initial SFS.
    ///
    /// If unset, a non-informative SFS will be used to initialise optimisation. This is fine
    /// for most purposes.
    #[clap(short = 'i', long, help_heading = "INPUT", value_name = "PATH")]
    pub initial: Option<PathBuf>,

    /// Input format file type.
    ///
    /// By default, the input file format is inferred from the file magic bytes, but this can be
    /// specified if it is preferred to be explicit.
    #[clap(
        short = 'I',
        long,
        arg_enum,
        help_heading = "INPUT",
        value_name = "STRING"
    )]
    pub input_format: Option<Format>,

    /// Random seed.
    ///
    /// If unset, a seed will be chosen at random.
    #[clap(short = 's', long, value_name = "INT")]
    pub seed: Option<u64>,

    /// Log-likelihood difference tolerated between epochs before stopping.
    ///
    /// The log-likelihood of blocks are summed over each epoch, and the difference between these
    /// sums are compared after each epoch. When the difference falls below the provided tolerance,
    /// the algorithm will stop. Due to overfitting, a lower tolerance will not necessarily yield a
    /// better estimate of the SFS. Also, note that a minumum of two epochs will be required;
    /// set `--max-epochs 1` if you wish to run one epoch only.
    ///
    /// If both this and `--max-epochs` are unset, the default stopping rule is a log-likelihood
    /// tolerance of 1e-4. If both are set, the first stopping rule to be triggered will stop the
    /// algorithm.
    #[clap(short = 'l', long, help_heading = "STOPPING", value_name = "FLOAT")]
    pub tolerance: Option<f64>,

    /// Number of threads.
    ///
    /// If the provided value is less than or equal to zero, the number of threads used will be
    /// equal to the available threads minus the provided value.
    #[clap(short = 't', long, default_value_t = 4, value_name = "INT")]
    pub threads: i32,

    /// Verbosity.
    ///
    /// Flag can be set multiply times to increase verbosity, or left unset for quiet mode.
    #[clap(short = 'v', long, parse(from_occurrences), global = true)]
    pub verbose: usize,

    /// Number of blocks per window.
    ///
    /// If unset, the window size will be chosen as approximately 1/5 of the number of blocks.
    #[clap(
        short = 'w',
        long,
        help_heading = "HYPERPARAMETERS",
        value_name = "INT"
    )]
    pub window_size: Option<NonZeroUsize>,

    #[clap(subcommand)]
    pub subcommand: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Shuffle(Shuffle),
    LogLikelihood(LogLikelihood),
}

impl Command {
    pub fn run(self) -> Result<(), clap::Error> {
        match self {
            Command::Shuffle(shuffle) => shuffle.run(),
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
    fn test_four_paths_errors() {
        let result = try_parse_args("winsfs a b c d");

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
