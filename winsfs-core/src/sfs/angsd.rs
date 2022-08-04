use std::{error::Error, fmt, str::FromStr};

use super::{DynShape, DynUSfs, Normalisation, SfsBase, Shape, ShapeError};

const ANGSD_SHAPE_SEP: &str = "/";
const ANGSD_DEFAULT_PRECISION: usize = 6;
const ANGSD_SEP: &str = " ";

pub fn format<S: Shape, N: Normalisation>(sfs: &SfsBase<S, N>, precision: Option<usize>) -> String {
    format!(
        "{}\n{}",
        format_header(&sfs.shape),
        sfs.format_flat(ANGSD_SEP, precision.unwrap_or(ANGSD_DEFAULT_PRECISION))
    )
}

fn format_header<S: Shape>(shape: &S) -> String {
    let shape_fmt = shape
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(ANGSD_SHAPE_SEP);

    format!("#SHAPE=<{shape_fmt}>")
}

pub fn parse(s: &str) -> Result<DynUSfs, ParseAngsdError> {
    if let Some((header, flat)) = s.split_once('\n') {
        let shape = parse_header(header)?;

        let values = parse_values(flat.trim_end_matches(|x: char| x.is_ascii_whitespace()))?;

        SfsBase::from_vec_shape(values, shape).map_err(ParseAngsdError::MismatchedShape)
    } else {
        Err(ParseAngsdError::Other(s.to_string()))
    }
}

fn parse_header(s: &str) -> Result<DynShape, ParseAngsdError> {
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

fn parse_values(s: &str) -> Result<Vec<f64>, ParseAngsdError> {
    s.split(ANGSD_SEP)
        .map(f64::from_str)
        .collect::<Result<Vec<_>, _>>()
        .map_err(ParseAngsdError::InvalidValue)
}

/// An error type associated with parsing an invalid ANGSD format SFS.
#[derive(Clone, Debug)]
pub enum ParseAngsdError {
    /// Failed to parse shape values in header.
    InvalidShape(std::num::ParseIntError),
    /// Failed to parse values in SFS.
    InvalidValue(std::num::ParseFloatError),
    /// Header dimensionality did not match requested.
    MismatchedDimensionality(usize),
    /// Header shape did not match values.
    MismatchedShape(ShapeError<DynShape>),
    /// Other error.
    Other(String),
}

impl fmt::Display for ParseAngsdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseAngsdError::InvalidShape(e) => write!(f, "{e}"),
            ParseAngsdError::InvalidValue(e) => write!(f, "{e}"),
            ParseAngsdError::MismatchedDimensionality(e) => write!(f, "{e}"),
            ParseAngsdError::MismatchedShape(e) => write!(f, "{e}"),
            ParseAngsdError::Other(s) => write!(
                f,
                "failed to parse SFS from ANGSD format from input:\n'{s}'"
            ),
        }
    }
}

impl Error for ParseAngsdError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_angsd_header() {
        assert_eq!(parse_header("#SHAPE=<3>").unwrap().as_ref(), &[3]);
        assert_eq!(parse_header("#SHAPE=<11/13>").unwrap().as_ref(), &[11, 13]);
    }

    #[test]
    fn test_format_angsd_header() {
        assert_eq!(format_header(&[25]), "#SHAPE=<25>");
        assert_eq!(format_header(&[7, 9]), "#SHAPE=<7/9>");
    }
}
