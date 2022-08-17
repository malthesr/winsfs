use std::{
    io::{self, Read},
    path::Path,
};

use winsfs_core::sfs::{io::plain_text, DynUSfs, USfs};

use super::StdinOrFile;

/// A reader for an input SFS.
pub struct SfsReader {
    inner: StdinOrFile,
}

impl SfsReader {
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
    /// The resulting SFS will not be normalised.
    ///
    /// Assumes the stream is positioned at the beginning. This will automatically attempt to infer
    /// the format of the SFS among the supported formats.
    pub fn read_dyn(&mut self) -> io::Result<DynUSfs> {
        let mut bytes = Vec::new();
        self.inner.read_to_end(&mut bytes)?;

        let format = Format::detect(&bytes).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot infer SFS input file format",
            )
        })?;

        let reader = &mut &bytes[..];
        match format {
            Format::PlainText => plain_text::read_sfs(reader),
        }
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
#[derive(Clone, Debug)]
enum Format {
    /// Plain text format.
    PlainText,
}

impl Format {
    /// Returns the format detected from a byte stream.
    pub fn detect(bytes: &[u8]) -> Option<Self> {
        Some(Self::PlainText)
    }
}
