use std::{error::Error, fmt, str::FromStr};

use super::{Sfs, ShapeError, UnnormalisedSfs};

const ANGSD_SHAPE_SEP: &str = "/";
const ANGSD_DEFAULT_PRECISION: usize = 6;
const ANGSD_SEP: &str = " ";

pub fn format<const N: usize, const NORM: bool>(
    sfs: &Sfs<N, NORM>,
    precision: Option<usize>,
) -> String {
    format!(
        "{}\n{}",
        format_header(sfs.shape()),
        sfs.format_flat(ANGSD_SEP, precision.unwrap_or(ANGSD_DEFAULT_PRECISION))
    )
}

fn format_header<const N: usize>(shape: [usize; N]) -> String {
    let shape_fmt = shape.map(|x| x.to_string()).join(ANGSD_SHAPE_SEP);

    format!("#SHAPE=<{shape_fmt}>")
}

pub fn parse<const N: usize>(s: &str) -> Result<UnnormalisedSfs<N>, ParseAngsdError<N>> {
    if let Some((header, flat)) = s.split_once('\n') {
        let shape = parse_header(header)?;

        let values = parse_values(flat.trim_end_matches(|x: char| x.is_ascii_whitespace()))?;

        Sfs::from_vec_shape(values, shape).map_err(ParseAngsdError::MismatchedShape)
    } else {
        Err(ParseAngsdError::Other(s.to_string()))
    }
}

fn parse_header<const N: usize>(s: &str) -> Result<[usize; N], ParseAngsdError<N>> {
    let v = s
        .trim_start_matches(|c: char| !c.is_numeric())
        .trim_end_matches(|c: char| !c.is_numeric())
        .split(ANGSD_SHAPE_SEP)
        .map(usize::from_str)
        .collect::<Result<Vec<_>, _>>()
        .map_err(ParseAngsdError::InvalidShape)?;

    let dims = v.len();
    v.try_into()
        .map_err(|_| ParseAngsdError::MismatchedDimensionality(dims))
}

fn parse_values<const N: usize>(s: &str) -> Result<Vec<f64>, ParseAngsdError<N>> {
    s.split(ANGSD_SEP)
        .map(f64::from_str)
        .collect::<Result<Vec<_>, _>>()
        .map_err(ParseAngsdError::InvalidValue)
}

/// An error type associated with parsing an invalid ANGSD format SFS.
#[derive(Clone, Debug)]
pub enum ParseAngsdError<const N: usize> {
    /// Failed to parse shape values in header.
    InvalidShape(std::num::ParseIntError),
    /// Failed to parse values in SFS.
    InvalidValue(std::num::ParseFloatError),
    /// Header dimensionality did not match requested.
    MismatchedDimensionality(usize),
    /// Header shape did not match values.
    MismatchedShape(ShapeError<N>),
    /// Other error.
    Other(String),
}

impl<const N: usize> fmt::Display for ParseAngsdError<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseAngsdError::InvalidShape(e) => write!(f, "{e}"),
            ParseAngsdError::InvalidValue(e) => write!(f, "{e}"),
            ParseAngsdError::MismatchedDimensionality(dims) => {
                write!(f, "found {dims}-dimensional SFS, expected {N}-dimensions")
            }
            ParseAngsdError::MismatchedShape(e) => write!(f, "{e}"),
            ParseAngsdError::Other(s) => write!(
                f,
                "failed to parse SFS from ANGSD format from input:\n'{s}'"
            ),
        }
    }
}

impl<const N: usize> Error for ParseAngsdError<N> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_angsd_header() {
        assert_eq!(parse_header::<1>("#SHAPE=<3>").unwrap(), [3]);
        assert_eq!(parse_header::<2>("#SHAPE=<11/13>").unwrap(), [11, 13]);
    }

    #[test]
    fn test_format_angsd_header() {
        assert_eq!(format_header::<1>([25]), "#SHAPE=<25>");
        assert_eq!(format_header::<2>([7, 9]), "#SHAPE=<7/9>");
    }
}
