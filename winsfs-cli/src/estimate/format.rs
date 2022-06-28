use std::{fs::File, io, path::Path};

use angsd_io::saf::MAGIC_NUMBER as STANDARD_MAGIC_NUMBER;

use clap::{ArgEnum, CommandFactory};

use winsfs_core::io::shuffle::MAGIC_NUMBER as SHUFFLED_MAGIC_NUMBER;

use super::Cli;

/// The possible input formats for SFS estimation.
#[derive(ArgEnum, Clone, Copy, Debug, Eq, PartialEq)]
pub enum Format {
    /// One or more standard SAF ANGSD files
    Standard,
    /// A single pseudo-shuffled SAF file, which may contain one or more populations
    Shuffled,
}

impl Format {
    /// Infer format from path name.
    fn infer_from_path<P>(path: P) -> Option<Self>
    where
        P: AsRef<Path>,
    {
        // Cannot use Path::extension, since it splits on last '.'
        let (_stem, ext) = path.as_ref().to_str()?.split_once('.')?;

        match ext {
            "saf.idx" | "saf.gz" | "saf.pos.gz" => Some(Self::Standard),
            "saf.shuf" => Some(Self::Shuffled),
            _ => None,
        }
    }

    /// Infer format from magic number in reader, and rewind reader to start.
    ///
    /// Note that this is sensitive to whether the input is bgzipped or not.
    fn infer_from_magic<R>(reader: &mut R) -> io::Result<Option<Self>>
    where
        R: io::Read + io::Seek,
    {
        const MAGIC_NUMBER_LEN: usize = magic_number_len();

        let mut buf = [0; MAGIC_NUMBER_LEN];
        reader.read_exact(&mut buf)?;
        reader.seek(io::SeekFrom::Current(-(MAGIC_NUMBER_LEN as i64)))?;

        Ok(match &buf {
            STANDARD_MAGIC_NUMBER => Some(Self::Standard),
            &SHUFFLED_MAGIC_NUMBER => Some(Self::Shuffled),
            _ => None,
        })
    }
}

impl TryFrom<&Cli> for Format {
    type Error = clap::Error;

    fn try_from(args: &Cli) -> Result<Self, Self::Error> {
        match args.paths.as_slice() {
            [] => unreachable!(), // Checked by clap
            [path] => {
                // Single input file, could be either standard or shuffled;
                // if user provided format, trust that and defer check to file reader,
                if let Some(expected_format) = args.input_format {
                    return Ok(expected_format);
                }

                // Otherwise, infer from path name; if that fails, try to infer from magic number.
                // This procedure fails if user provides standard non-index file with unconventional
                // name, since both path check and magic check will fail (the latter due to bgzip).
                if let Some(format) = Self::infer_from_path(path) {
                    Ok(format)
                } else if let Some(format) = Format::infer_from_magic(&mut File::open(path)?)? {
                    Ok(format)
                } else {
                    Err(Cli::command().error(
                        clap::ErrorKind::ValueValidation,
                        "unrecognised input file type",
                    ))
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

const fn magic_number_len() -> usize {
    if STANDARD_MAGIC_NUMBER.len() != SHUFFLED_MAGIC_NUMBER.len() {
        panic!("length of magic numbers do not match")
    }
    STANDARD_MAGIC_NUMBER.len()
}
