use std::path::{Path, PathBuf};

use clap::Args;

use winsfs::{io::Intersect, sfs::UnnormalisedSfs};

use crate::utils::join;

/// Calculate log-likelihood of site frequency spectrum.
///
/// The SAF files will be streamed, and therefore the calculation requires only constant memory
/// usage.
#[derive(Args, Debug)]
pub struct LogLikelihood {
    /// Input SAF file paths.
    ///
    /// For each set of SAF files (conventially named [prefix].{saf.idx,saf.pos.gz,saf.gz}),
    /// specify either the shared prefix or the full path to any one member file.
    /// Up to three SAF files currently supported.
    #[clap(
        parse(from_os_str),
        max_values = 3,
        required = true,
        value_name = "PATHS"
    )]
    pub paths: Vec<PathBuf>,

    /// Input SFS to calculate log-likelihood from.
    #[clap(short = 'i', long)]
    pub sfs: PathBuf,
}

impl LogLikelihood {
    pub fn run(self) -> clap::Result<()> {
        match &self.paths[..] {
            [p] => self.run_n([p]),
            [p1, p2] => self.run_n([p1, p2]),
            [p1, p2, p3] => self.run_n([p1, p2, p3]),
            _ => unreachable!(), // Checked by clap
        }
    }

    pub fn run_n<const N: usize, P>(&self, paths: [P; N]) -> clap::Result<()>
    where
        P: AsRef<Path>,
    {
        log::debug!(
            target: "init",
            "Reading SFS from path:\n\t{}",
            self.sfs.display()
        );

        let sfs = UnnormalisedSfs::<N>::read_from_angsd(&self.sfs)?.normalise();

        log::info!(
            target: "init",
            "Streaming (intersecting) sites in input SAF files:\n\t{}",
            join(paths.iter().map(|p| p.as_ref().display()), "\n\t")
        );

        let mut reader = Intersect::from_paths(paths.as_slice())?;

        let (log_likelihood, sites) = sfs.stream_log_likelihood(&mut reader)?.into();

        log::info!(target: "log-likelihood", "Processed {sites} sites");

        println!("{}", f64::from(log_likelihood));

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use clap::Parser;

    use crate::cli::{Cli, Command};

    fn try_parse_args(cmd: &str) -> clap::Result<LogLikelihood> {
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
            clap::ErrorKind::MissingRequiredArgument,
        );
    }
}
