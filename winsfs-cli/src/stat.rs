use std::{error::Error, fmt, io, path::PathBuf};

use clap::{ArgEnum, Args, CommandFactory};

use winsfs_core::sfs::{DynUSfs, USfs};

use crate::{input, utils::join, Cli};

/// Calculate statistics from SFS.
#[derive(Args, Debug)]
pub struct Stat {
    /// Input SFS.
    ///
    /// The input SFS can be provided here or read from stdin. The SFS will be normalised as
    /// required for particular statistics, so the input SFS does not need to be normalised.
    #[clap(parse(from_os_str), value_name = "PATH")]
    pub path: Option<PathBuf>,

    /// Delimiter between statistics.
    #[clap(short = 'd', long, default_value_t = ',', value_name = "CHAR")]
    pub delimiter: char,

    /// Output a header with the names of statistics.
    #[clap(short = 'H', long)]
    pub header: bool,

    /// Precision to use when printing statistics.
    ///
    /// If a single value is provided, this will be used for all statistics. If more than one
    /// statistic is calculated, the same number of precision specifiers may be provided, and they
    /// will be applied in the same order. Use comma to separate precision specifiers.
    #[clap(
        short = 'p',
        long,
        default_value = "6",
        use_value_delimiter = true,
        value_name = "INT(S)"
    )]
    pub precision: Vec<usize>,

    /// Statistics to calculate.
    ///
    /// More than one statistic can be output. Use comma to separate statistics.
    /// An error will be thrown if the shape or dimensionality of the SFS is incompatible with the
    /// required statistics.
    #[clap(
        short = 's',
        long,
        arg_enum,
        required = true,
        use_value_delimiter = true,
        value_name = "STAT(S)"
    )]
    pub statistics: Vec<Statistic>,
}

/// Statistics that can be calculated.
#[derive(ArgEnum, Clone, Copy, Debug, Eq, PartialEq)]
pub enum Statistic {
    F2,
    King,
    R0,
    R1,
    Sum,
}

impl Statistic {
    /// Calculate statistic from provided SFS.
    ///
    /// Different statistics have various requirements on shape or dimensionality of the SFS.
    /// An error is returned if the statistic cannot be calculated from the provided SFS.
    pub fn calculate(&self, sfs: DynUSfs) -> Result<f64, StatisticError> {
        let dim = sfs.shape().len();

        match self {
            Statistic::F2 => match USfs::<2>::try_from(sfs) {
                Ok(sfs_2d) => Ok(sfs_2d.normalise().f2()),
                Err(_) => Err(StatisticError::DimensionError {
                    name: "F2",
                    expected: 2,
                    found: dim,
                }),
            },
            Statistic::King => calculate_kinship_stat(sfs, "King", |sfs| sfs.king()),
            Statistic::R0 => calculate_kinship_stat(sfs, "R0", |sfs| sfs.r0()),
            Statistic::R1 => calculate_kinship_stat(sfs, "R1", |sfs| sfs.r1()),
            Statistic::Sum => Ok(sfs.iter().sum::<f64>()),
        }
    }

    /// Returns the name of the statistic as it should be used in the output header.
    pub fn header_name(&self) -> String {
        match self {
            Statistic::F2 => "f2",
            Statistic::King => "king",
            Statistic::R0 => "r0",
            Statistic::R1 => "r1",
            Statistic::Sum => "sum",
        }
        .to_string()
    }
}

impl Stat {
    pub fn run(self) -> clap::Result<()> {
        let sfs = input::sfs::Reader::from_path_or_stdin(self.path.as_ref())?.read_dyn()?;

        // Calculate all values early to check whether shape/dimensionality fits before
        // e.g. writing header
        let values = self.calculate(&sfs)?;

        let stdout = io::stdout();
        let mut writer = stdout.lock();

        if self.header {
            self.print_header(&mut writer)?;
        }

        let precisions = self.get_precisions()?;

        self.print_values(&mut writer, &values, &precisions)
    }

    /// Calculate the required statistic for a single SFS.
    fn calculate(&self, sfs: &DynUSfs) -> clap::Result<Vec<f64>> {
        self.statistics
            .iter()
            .map(|stat| stat.calculate(sfs.clone()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| Cli::command().error(clap::ErrorKind::ValueValidation, e))
    }

    /// Gets the precisions to be used for printing the calculated statistics.
    ///
    /// The output vector will always be the same length as the number of statistics asked for.
    /// If the provided number of statistics is one, it will be repeated to match; otherwise the
    /// number must match the number of statistics, or an error will be returned.
    fn get_precisions(&self) -> clap::Result<Vec<usize>> {
        let n = self.statistics.len();

        match &self.precision[..] {
            [v] => Ok(vec![*v; n]),
            vs if vs.len() == n => Ok(self.precision.clone()),
            _ => Err(Cli::command().error(
                clap::ErrorKind::ValueValidation,
                "number of precision values must be one or match number of statistics calculated",
            )),
        }
    }

    /// Prints the header corresponding to the arguments provided.
    pub fn print_header<W>(&self, writer: &mut W) -> clap::Result<()>
    where
        W: io::Write,
    {
        let (first, rest) = self.statistics.split_first().expect("checked by clap");

        write!(writer, "{}", first.header_name())?;
        for stat in rest {
            write!(writer, "{}{}", self.delimiter, stat.header_name())?;
        }
        writeln!(writer)?;

        Ok(())
    }

    /// Prints a set of values to a single line with the provided precisions and other arguments
    /// provided.
    ///
    /// `values` and `precisions` here must be of the same length.
    pub fn print_values<W>(
        &self,
        writer: &mut W,
        values: &[f64],
        precisions: &[usize],
    ) -> clap::Result<()>
    where
        W: io::Write,
    {
        debug_assert_eq!(values.len(), precisions.len());

        for (i, (value, precision)) in values.iter().zip(precisions).enumerate() {
            if value.is_nan() {
                log::warn!(
                    target: "stat",
                    "Output has NaN in statistics"
                );
            }

            if i > 0 {
                write!(writer, "{}", self.delimiter)?;
            }

            write!(writer, "{value:.precision$}")?;
        }
        writeln!(writer)?;

        Ok(())
    }
}

/// An error associated with calculation of the various statistics from the input SFS.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StatisticError {
    DimensionError {
        name: &'static str,
        expected: usize,
        found: usize,
    },
    ShapeError {
        name: &'static str,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
}

impl fmt::Display for StatisticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StatisticError::DimensionError {
                name,
                expected,
                found,
            } => {
                write!(
                    f,
                    "calculating {name} requires SFS with dimension {expected}, \
                    found SFS with dimension {found}"
                )
            }
            StatisticError::ShapeError {
                name,
                expected,
                found,
            } => {
                write!(
                    f,
                    "calculating {name} requires SFS with shape {}, found SFS with shape {}",
                    join(expected, "/"),
                    join(found, "/"),
                )
            }
        }
    }
}

impl Error for StatisticError {}

/// Helper to calculate R0, R1, or King statistic.
///
/// This factors out the error checking and handling.
fn calculate_kinship_stat<F>(sfs: DynUSfs, name: &'static str, f: F) -> Result<f64, StatisticError>
where
    F: Fn(&USfs<2>) -> Option<f64>,
{
    let shape = sfs.shape().clone();
    let dim = shape.len();

    match USfs::<2>::try_from(sfs) {
        Ok(sfs_2d) => match f(&sfs_2d) {
            Some(stat) => Ok(stat),
            None => Err(StatisticError::ShapeError {
                name,
                expected: vec![3, 3],
                found: shape.to_vec(),
            }),
        },
        Err(_) => Err(StatisticError::DimensionError {
            name,
            expected: 2,
            found: dim,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use clap::Parser;

    use crate::cli::Command;

    fn try_parse_args(cmd: &str) -> clap::Result<Stat> {
        Cli::try_parse_from(cmd.split_whitespace()).map(|cli| match cli.subcommand {
            Some(Command::Stat(stat)) => stat,
            _ => panic!(),
        })
    }

    fn parse_args(cmd: &str) -> Stat {
        try_parse_args(cmd).expect("failed to parse command")
    }

    #[test]
    fn test_missing_statistic() {
        let result = try_parse_args("winsfs stat /path/to/sfs");
        assert_eq!(
            result.unwrap_err().kind(),
            clap::ErrorKind::MissingRequiredArgument,
        );
    }

    #[test]
    fn test_multiple_statistics() {
        let args = parse_args("winsfs stat -s sum,f2 /path/to/sfs");
        assert_eq!(args.statistics, &[Statistic::Sum, Statistic::F2]);
    }

    #[test]
    fn test_repeated_statistics() {
        let args = parse_args("winsfs stat -s f2,f2 /path/to/sfs");
        assert_eq!(args.statistics, &[Statistic::F2, Statistic::F2]);
    }

    #[test]
    fn test_default_precision() {
        let args = parse_args("winsfs stat -s sum /path/to/sfs");
        assert_eq!(args.precision, &[6]);
    }
}
