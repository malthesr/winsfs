use std::{
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
};

use angsd_io::saf;

use clap::{ArgGroup, Args};

use crate::Sfs;

use super::utils::validate_shape;

/// Calculate log-likelihood of site frequency spectrum.
///
/// The SAF files will be streamed, and therefore the calculation requires only constant memory
/// usage.
#[derive(Args, Debug)]
#[clap(group(ArgGroup::new("input").required(true)))]
pub struct LogLikelihood {
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

    /// Input SFS to calculate log-likelihood from.
    ///
    /// Optionally, multiple SFS may be provided as a comma-separated list,
    /// and the log-likelihood will be calculated for each SFS in the list.
    #[clap(
        short = 'i',
        long,
        multiple_values = true,
        use_value_delimiter = true,
        require_value_delimiter = true,
        min_values = 1,
        group = "input"
    )]
    pub sfs: Option<Vec<PathBuf>>,

    /// File containing list of input SFS to calculate log-likelihood from.
    ///
    /// Each line in file should contain a path to an SFS to use.
    #[clap(short = 'I', long, parse(from_os_str), group = "input")]
    pub sfs_list: Option<PathBuf>,
}

impl LogLikelihood {
    pub(crate) fn run(self) -> Result<(), clap::Error> {
        match self.paths.as_slice() {
            [path] => run_1d(path, &self),
            [first_path, second_path] => run_2d(first_path, second_path, &self),
            _ => unreachable!(), // Checked by clap
        }
    }
}

fn run_1d<P>(saf_path: P, args: &LogLikelihood) -> clap::Result<()>
where
    P: AsRef<Path>,
{
    let all_sfs = read_sfs(args)?;

    let mut reader = saf::Reader::from_bgzf_member_path(saf_path)?;
    let sites = reader.index().total_sites();
    let cols = reader.index().alleles() + 1;

    for sfs in all_sfs.iter() {
        validate_shape(sfs.shape(), [cols])?;
    }

    log::info!(
        target: "log-likelihood",
        "Streamining {sites} sites in SAF file with {cols} cols."
    );

    let mut log_likelihoods = vec![0.0; all_sfs.len()];

    let value_reader = reader.value_reader_mut();
    let mut buf = vec![0.0; cols];
    let site = buf.as_mut_slice();

    while value_reader.read_values(site)?.is_not_done() {
        exp(site);

        log_likelihoods
            .iter_mut()
            .zip(all_sfs.iter())
            .for_each(|(log_likelihood, sfs)| *log_likelihood += sfs.site_log_likelihood(site));
    }

    for log_likelihood in log_likelihoods {
        println!("{log_likelihood}");
    }

    Ok(())
}

pub fn run_2d<P>(first_path: P, second_path: P, args: &LogLikelihood) -> clap::Result<()>
where
    P: AsRef<Path>,
{
    let all_sfs = read_sfs(args)?;

    let first_reader = saf::Reader::from_bgzf_member_path(first_path)?;
    let first_sites = first_reader.index().total_sites();
    let first_cols = first_reader.index().alleles() + 1;

    let second_reader = saf::Reader::from_bgzf_member_path(second_path)?;
    let second_sites = second_reader.index().total_sites();
    let second_cols = second_reader.index().alleles() + 1;

    for sfs in all_sfs.iter() {
        validate_shape(sfs.shape(), [first_cols, second_cols])?;
    }

    log::info!(
        target: "log-likelihood",
        "Streamining intersecting sites among {first_sites}/{second_sites} total sites \
         in SAF files with {first_cols}/{second_cols} cols."
    );

    let mut reader = saf::reader::Intersect::new(first_reader, second_reader);

    let mut log_likelihoods = vec![0.0; all_sfs.len()];

    let (mut first_record, mut second_record) = reader.create_record_buf();

    while reader
        .read_record_pair(&mut first_record, &mut second_record)?
        .is_not_done()
    {
        exp(first_record.values_mut());
        exp(second_record.values_mut());

        log_likelihoods
            .iter_mut()
            .zip(all_sfs.iter())
            .for_each(|(log_likelihood, sfs)| {
                *log_likelihood +=
                    sfs.site_log_likelihood(first_record.values(), second_record.values())
            });
    }

    for log_likelihood in log_likelihoods {
        println!("{log_likelihood}");
    }

    Ok(())
}

fn exp(values: &mut [f32]) {
    values.iter_mut().for_each(|x| *x = x.exp());
}

fn read_sfs<const N: usize>(args: &LogLikelihood) -> clap::Result<Vec<Sfs<N>>> {
    let paths = if let Some(list) = &args.sfs_list {
        let mut reader = fs::File::open(list)?;
        let mut buf = String::new();
        reader.read_to_string(&mut buf)?;

        buf.lines().map(PathBuf::from).collect()
    } else if let Some(sfs) = &args.sfs {
        sfs.clone()
    } else {
        unreachable!()
    };

    paths
        .iter()
        .map(|p| {
            Sfs::read_from_angsd(p).map(|mut sfs| {
                sfs.normalise();
                sfs
            })
        })
        .collect::<io::Result<Vec<_>>>()
        .map_err(|e| e.into())
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
    fn test_input_group() {
        let result =
            try_parse_args("winsfs log-likelihood --sfs-list list --sfs first,second /path/to/saf");
        assert_eq!(
            result.unwrap_err().kind(),
            clap::ErrorKind::ArgumentConflict,
        );

        let result = try_parse_args("winsfs log-likelihood /path/to/saf");
        assert_eq!(
            result.unwrap_err().kind(),
            clap::ErrorKind::MissingRequiredArgument,
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

    #[test]
    fn test_comma_separated_sfs() {
        let args = parse_args("winsfs log-likelihood --sfs first,second /path/to/saf");
        assert_eq!(
            args.sfs,
            Some(vec![PathBuf::from("first"), PathBuf::from("second")])
        );
    }
}
