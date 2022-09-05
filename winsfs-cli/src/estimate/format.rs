use std::{fmt, fs::File, io};

use angsd_saf as saf;
use saf::version::Version;

use clap::{ArgEnum, CommandFactory};

use super::Cli;

/// The possible input formats for SFS estimation.
#[derive(ArgEnum, Clone, Copy, Debug, Eq, PartialEq)]
pub enum Format {
    /// One or more standard SAF ANGSD files
    Standard,
    /// One or more standard SAF ANGSD files
    Banded,
    /// A single pseudo-shuffled SAF file, which may contain one or more populations
    Shuffled,
}

impl Format {
    /// Infer format from magic number in reader, and rewind reader to start.
    ///
    /// Note that this is sensitive to whether the input is bgzipped or not.
    pub fn infer_from_magic<R>(reader: &mut R) -> io::Result<Self>
    where
        R: io::Read + io::Seek,
    {
        const MAGIC_LEN: usize = 8;

        let mut buf = [0; MAGIC_LEN];
        reader.read_exact(&mut buf)?;
        reader.seek(io::SeekFrom::Current(-(MAGIC_LEN as i64)))?;

        match buf {
            saf::version::V3::MAGIC_NUMBER => Ok(Self::Standard),
            saf::version::V4::MAGIC_NUMBER => Ok(Self::Banded),
            winsfs_core::io::shuffle::MAGIC_NUMBER => Ok(Self::Shuffled),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to detect SAF file version from magic number {buf:02x?}",),
            )),
        }
    }

    /// Returns the format as a string representation of the corresponding SAF file format.
    pub fn version_string(&self) -> String {
        match self {
            Self::Standard => "v3",
            Self::Banded => "v4",
            Self::Shuffled => "vshuf",
        }
        .to_string()
    }
}

impl TryFrom<&Cli> for Format {
    type Error = clap::Error;

    fn try_from(args: &Cli) -> Result<Self, Self::Error> {
        match args.paths.as_slice() {
            [] => unreachable!(), // Checked by clap
            [path] => {
                // Single input file, could be either standard, banded, or shuffled;
                // if user provided format, trust that and defer check to file reader,
                if let Some(expected_format) = args.input_format {
                    Ok(expected_format)
                } else {
                    Format::infer_from_magic(&mut File::open(path)?).map_err(|e| e.into())
                }
            }
            [..] => {
                // Multiple input files, must be standard format
                if let Some(Format::Shuffled) = args.input_format {
                    Err(Cli::command().error(
                        clap::ErrorKind::ValueValidation,
                        "only standard input file format valid for more than a single input path",
                    ))
                } else {
                    Ok(Format::Standard)
                }
            }
        }
    }
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard => f.write_str("full"),
            Self::Banded => f.write_str("banded"),
            Self::Shuffled => f.write_str("shuffled"),
        }
    }
}
