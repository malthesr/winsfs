use std::path::{Path, PathBuf};

use clap::{error::Result as ClapResult, Args};

use crate::input;

/// Calculate log-likelihood of site frequency spectrum.
///
/// The SAF files will be streamed, and therefore the calculation requires only constant memory
/// usage.
#[derive(Args, Debug)]
pub struct LogLikelihood {
    /// Input SAF file paths.
    ///
    /// For each set of SAF files (conventially named 'prefix'.{saf.idx,saf.pos.gz,saf.gz}),
    /// specify either the shared prefix or the full path to any one member file.
    /// Up to three SAF files currently supported.
    #[clap(value_parser, num_args = 1..=3, required = true, value_name = "PATHS")]
    pub paths: Vec<PathBuf>,

    /// Input SFS to calculate log-likelihood from.
    #[clap(short = 'i', long)]
    pub sfs: PathBuf,

    /// Number of threads to use for reading.
    ///
    /// If set to 0, all available cores will be used.
    #[clap(short = 't', long, default_value_t = 4, value_name = "INT")]
    pub threads: usize,
}

impl LogLikelihood {
    pub fn run(self) -> ClapResult<()> {
        match &self.paths[..] {
            [p] => self.run_n([p]),
            [p1, p2] => self.run_n([p1, p2]),
            [p1, p2, p3] => self.run_n([p1, p2, p3]),
            _ => unreachable!(), // Checked by clap
        }
    }

    pub fn run_n<const D: usize, P>(&self, paths: [P; D]) -> ClapResult<()>
    where
        P: AsRef<Path>,
    {
        let sfs = input::sfs::Reader::from_path(&self.sfs)?
            .read::<D>()?
            .normalise();

        let readers = input::saf::Readers::from_member_paths(&paths, self.threads)?;

        log::info!(
            target: "init",
            "Streaming (intersecting) sites in input SAF files",
        );

        let (log_likelihood, sites) = readers.log_likelihood(sfs)?;

        log::info!(target: "log-likelihood", "Processed {sites} sites");

        println!("{}", f64::from(log_likelihood));

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use clap::{error::ErrorKind, Parser};

    use crate::cli::{Cli, Command};

    fn try_parse_args(cmd: &str) -> ClapResult<LogLikelihood> {
        Cli::try_parse_from(cmd.split_whitespace()).map(|cli| match cli.subcommand {
            Some(Command::LogLikelihood(log_likelihood)) => log_likelihood,
            _ => panic!(),
        })
    }

    fn parse_args(cmd: &str) -> LogLikelihood {
        try_parse_args(cmd).expect("failed to parse subcommand")
    }

    #[test]
    fn test_basic() {
        assert_eq!(
            parse_args("winsfs log-likelihood --sfs /path/to/sfs saf").paths,
            vec![PathBuf::from("saf")]
        );
        assert_eq!(
            parse_args("winsfs log-likelihood --sfs /path/to/sfs saf1 saf2").paths,
            vec![PathBuf::from("saf1"), PathBuf::from("saf2")]
        );
        assert_eq!(
            parse_args("winsfs log-likelihood --sfs /path/to/sfs saf1 saf2 saf3").paths,
            vec![
                PathBuf::from("saf1"),
                PathBuf::from("saf2"),
                PathBuf::from("saf3")
            ]
        );
    }

    #[test]
    fn test_missing_sfs() {
        let result = try_parse_args("winsfs log-likelihood --sfs /path/to/saf");
        assert_eq!(
            result.unwrap_err().kind(),
            ErrorKind::MissingRequiredArgument,
        );
    }
}
