use std::{
    io::{self, Read},
    path::Path,
};

use clap::ValueEnum;

use winsfs_core::sfs::{
    io::{npy, plain_text},
    DynUSfs, Multi, USfs,
};

use super::StdinOrFile;

/// The npy magic number.
const NPY_MAGIC: [u8; 6] = *b"\x93NUMPY";

/// The npz magic number.
///
/// The npz format is just a ZIP archive, so this is really just the zip archive magic.
const NPZ_MAGIC: [u8; 4] = [b'P', b'K', 0x03, 0x04];

/// The beginning of a plain text format file.
const PLAIN_TEXT_START: [u8; 6] = *b"#SHAPE";

/// A reader for an input SFS.
pub struct Reader {
    inner: StdinOrFile,
}

impl Reader {
    /// Creates a new reader from a file path.
    pub fn from_path<P>(path: P) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        log::debug!(
            target: "init",
            "Reading SFS from path:\n\t{}",
            path.as_ref().display()
        );

        StdinOrFile::from_path(path).map(Self::new)
    }

    /// Creates a new reader from a file path if `Some`, otherwise fall back to stdin.
    ///
    /// This will fail if no path is provided and stdin is not readable.
    pub fn from_path_or_stdin<P>(path: Option<P>) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        match path {
            Some(p) => Self::from_path(p),
            None => Self::from_stdin(),
        }
    }

    /// Creates a new reader from stdin.
    ///
    /// This will fail if stdin is not readable.
    pub fn from_stdin() -> io::Result<Self> {
        log::debug!(
            target: "init",
            "Reading SFS from stdin",
        );
        StdinOrFile::from_stdin().map(Self::new)
    }

    /// Creates a new reader.
    fn new(inner: StdinOrFile) -> Self {
        Self { inner }
    }

    /// Reads an SFS with dynamic dimensions.
    ///
    /// Assumes the stream is positioned at the beginning. This will automatically attempt to infer
    /// the format of the SFS among the supported formats.
    pub fn read_dyn(&mut self) -> io::Result<DynUSfs> {
        let (format, bytes) = self.read_format()?;

        let reader = &mut &bytes[..];
        match format {
            Format::PlainText => plain_text::read_sfs(reader),
            Format::Npy => npy::read_sfs(reader),
            Format::Npz => {
                let multi = npy::read_multi_sfs(&mut io::Cursor::new(reader))?;

                if multi.len() == 1 {
                    Ok(Vec::from(multi).pop().unwrap())
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "found SFS in npz format with more than one SFS \
                        when trying to read single SFS",
                    ))
                }
            }
        }
    }

    /// Reads a multi-SFS with dynamic dimensions.
    ///
    /// Assumes the stream is positioned at the beginning. This will automatically attempt to infer
    /// the format of the SFS among the supported formats.
    pub fn read_dyn_multi(&mut self) -> io::Result<Multi<DynUSfs>> {
        let (format, bytes) = self.read_format()?;

        let mut reader = io::Cursor::new(bytes);
        match format {
            Format::PlainText => plain_text::read_multi_sfs(&mut reader),
            Format::Npy => {
                let sfs = npy::read_sfs(&mut reader)?;

                Ok(Multi::from(sfs))
            }
            Format::Npz => npy::read_multi_sfs(&mut reader),
        }
    }

    /// Reads all bytes in the underlying format and infers the format.
    fn read_format(&mut self) -> io::Result<(Format, Vec<u8>)> {
        let mut bytes = Vec::new();
        self.inner.read_to_end(&mut bytes)?;

        let format = Format::detect(&bytes).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot infer SFS input file format",
            )
        })?;

        Ok((format, bytes))
    }

    /// Reads an SFS with static dimensions.
    ///
    /// The resulting SFS will not be normalised.
    ///
    /// Assumes the stream is positioned at the beginning. This will automatically attempt to infer
    /// the format of the SFS among the supported formats.
    pub fn read<const D: usize>(&mut self) -> io::Result<USfs<D>> {
        self.read_dyn().and_then(|sfs| {
            USfs::try_from(sfs).map_err(|err_sfs| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "found SFS with {} dimensions (expected {D} dimensions)",
                        err_sfs.shape().len()
                    ),
                )
            })
        })
    }
}

/// An SFS input format.
#[derive(ValueEnum, Clone, Debug, Eq, PartialEq)]
pub enum Format {
    /// Plain text format.
    #[clap(name = "txt")]
    PlainText,
    /// Numpy npy format.
    Npy,
    /// Numpy npz format.
    Npz,
}

impl Format {
    /// Returns the format detected from a byte stream.
    pub fn detect(bytes: &[u8]) -> Option<Self> {
        Self::detect_npz(bytes)
            .xor(Self::detect_npy(bytes))
            .xor(Self::detect_plain_text(bytes))
    }

    /// Returns the npy format if detected in byte stream.
    pub fn detect_npy(bytes: &[u8]) -> Option<Self> {
        (bytes[..NPY_MAGIC.len()] == NPY_MAGIC).then_some(Self::Npy)
    }

    /// Returns the npz format if detected in byte stream.
    pub fn detect_npz(bytes: &[u8]) -> Option<Self> {
        (bytes[..NPZ_MAGIC.len()] == NPZ_MAGIC).then_some(Self::Npz)
    }

    /// Returns the plain text format if detected in byte stream.
    pub fn detect_plain_text(bytes: &[u8]) -> Option<Self> {
        (bytes[..PLAIN_TEXT_START.len()] == PLAIN_TEXT_START).then_some(Self::PlainText)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_npy() {
        assert_eq!(Format::detect_npy(&NPY_MAGIC), Some(Format::Npy));

        let mut bytes = NPY_MAGIC.to_vec();
        bytes.extend(b"arbitrary bytes");
        assert_eq!(Format::detect(&bytes), Some(Format::Npy));
    }

    #[test]
    fn test_detect_plain_text() {
        assert_eq!(
            Format::detect_plain_text(&PLAIN_TEXT_START),
            Some(Format::PlainText)
        );

        let mut bytes = PLAIN_TEXT_START.to_vec();
        bytes.extend(b"=<17/19>\n1 2 3");
        assert_eq!(Format::detect(&bytes), Some(Format::PlainText));
    }
}
