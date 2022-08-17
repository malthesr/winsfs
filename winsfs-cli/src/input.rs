use std::{fs, io, path::Path};

pub mod sfs;

/// A reader that can either read from stdin or from a file.
pub enum StdinOrFile {
    Stdin(io::Stdin),
    File(fs::File),
}

impl StdinOrFile {
    /// Creates a new reader from a file path if `Some`, otherwise fall back to stdin.
    ///
    /// This will fail if no path is provided and stdin is not readable.
    pub fn from_path_or_stdin<P>(path: Option<P>) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        path.map(Self::from_path).unwrap_or_else(Self::from_stdin)
    }

    /// Creates a new reader from a file path.
    pub fn from_path<P>(path: P) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        fs::File::open(path).map(Self::File)
    }

    /// Creates a new reader from stdin.
    ///
    /// This will fail if stdin is not readable.
    pub fn from_stdin() -> io::Result<Self> {
        if atty::isnt(atty::Stream::Stdin) {
            Ok(Self::Stdin(io::stdin()))
        } else {
            Err(io::Error::new(
                io::ErrorKind::Other,
                "cannot read from stdin",
            ))
        }
    }
}

impl io::Read for StdinOrFile {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            StdinOrFile::Stdin(stdin) => stdin.read(buf),
            StdinOrFile::File(file) => file.read(buf),
        }
    }
}
