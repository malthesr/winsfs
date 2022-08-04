//! Reading and writing for the plain text SFS format.
//!
//! The plain text format is a simple format consisting of two lines.
//! The first line contains a header line `#SHAPE=<[shape]>`, where `[shape]`
//! is a `/`-separated representation of the shape of the SFS. The next line
//! gives the SFS in flat, row-major order separated by a single space.
//!
//! In other words, the plain text format is like the format output by realSFS,
//! except with the addition of a header line so that the SFS can be read without
//! passing the shape separately.

use std::{error::Error, fmt, fs::File, io, path::Path, str::FromStr};

use crate::sfs::{
    generics::{DynShape, Normalisation, Shape},
    DynUSfs, SfsBase,
};

/// Reads an SFS in plain text format from a reader.
///
/// The stream is assumed to be positioned at the start.
pub fn read_sfs<R>(reader: &mut R) -> io::Result<DynUSfs>
where
    R: io::BufRead,
{
    let header = Header::read(reader)?;

    let mut buf = String::new();
    let _bytes_read = reader.read_to_string(&mut buf)?;

    buf.split_ascii_whitespace()
        .map(f64::from_str)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
        .and_then(|vec| {
            DynUSfs::from_vec_shape(vec, header.shape)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
        })
}

/// Reads an SFS in plain text format from a file path.
pub fn read_sfs_from_path<P>(path: P) -> io::Result<DynUSfs>
where
    P: AsRef<Path>,
{
    let mut reader = File::open(path).map(io::BufReader::new)?;
    read_sfs(&mut reader)
}

/// Writes an SFS in plain text format to a writer.
pub fn write_sfs<W, S, N>(writer: &mut W, sfs: &SfsBase<S, N>) -> io::Result<()>
where
    W: io::Write,
    S: Shape,
    N: Normalisation,
{
    let header = Header::new(sfs.shape().as_ref().to_vec().into_boxed_slice());
    header.write(writer)?;

    writeln!(writer, "{}", sfs.format_flat(" ", 6))
}

/// Writes an SFS in plain text format to a file path.
///
/// If the file already exists, it will be overwritten.
pub fn write_sfs_to_path<P, S, N>(path: P, sfs: &SfsBase<S, N>) -> io::Result<()>
where
    P: AsRef<Path>,
    S: Shape,
    N: Normalisation,
{
    let mut writer = File::create(path)?;
    write_sfs(&mut writer, sfs)
}

/// A plain text SFS header.
#[derive(Clone, Debug)]
struct Header {
    shape: DynShape,
}

impl Header {
    /// Creates a new header.
    pub fn new(shape: DynShape) -> Self {
        Self { shape }
    }

    /// Reads a header from a reader.
    ///
    /// Assumes the stream is positioned immediately in front of the header.
    pub fn read<R>(reader: &mut R) -> io::Result<Self>
    where
        R: io::BufRead,
    {
        let mut buf = String::new();

        reader.read_line(&mut buf)?;

        Self::from_str(&buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Writes a header to a stream.
    pub fn write<W>(&self, writer: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        writeln!(writer, "{self}")
    }
}

impl fmt::Display for Header {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape_fmt = self
            .shape
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join("/");

        write!(f, "#SHAPE=<{shape_fmt}>")
    }
}

impl FromStr for Header {
    type Err = ParseHeaderError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.trim_start_matches(|c: char| !c.is_numeric())
            .trim_end_matches(|c: char| !c.is_numeric())
            .split('/')
            .map(usize::from_str)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| ParseHeaderError(String::from(s)))
            .map(Vec::into_boxed_slice)
            .map(Header::new)
    }
}

/// An error associated with parsing the plain text format header.
#[derive(Debug)]
pub struct ParseHeaderError(String);

impl fmt::Display for ParseHeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "failed to parse '{}' as plain SFS format header", self.0)
    }
}

impl Error for ParseHeaderError {}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{sfs1d, sfs2d};

    #[test]
    fn test_parse_header() {
        assert_eq!(Header::from_str("#SHAPE=<3>").unwrap().shape.as_ref(), [3]);
        assert_eq!(
            Header::from_str("#SHAPE=<11/13>").unwrap().shape.as_ref(),
            &[11, 13]
        );
    }

    #[test]
    fn test_display_header() {
        assert_eq!(Header::new(Box::new([25])).to_string(), "#SHAPE=<25>");
        assert_eq!(Header::new(Box::new([7, 9])).to_string(), "#SHAPE=<7/9>");
    }

    #[test]
    fn test_read_1d() -> io::Result<()> {
        let src = b"#SHAPE=<3>\n0.0 1.0 2.0\n";

        assert_eq!(read_sfs(&mut &src[..])?, DynUSfs::from(sfs1d![0., 1., 2.]));

        Ok(())
    }

    #[test]
    fn test_read_2d() -> io::Result<()> {
        let src = b"#SHAPE=<2/3>\n0.0 1.0 2.0 3.0 4.0 5.0\n";

        assert_eq!(
            read_sfs(&mut &src[..])?,
            DynUSfs::from(sfs2d![[0., 1., 2.], [3., 4., 5.]])
        );

        Ok(())
    }

    #[test]
    fn test_write_1d() -> io::Result<()> {
        let mut dest = Vec::new();
        write_sfs(&mut dest, &sfs1d![0., 1., 2.])?;

        eprintln!("{}", String::from_utf8(dest.clone()).unwrap());

        assert_eq!(dest, b"#SHAPE=<3>\n0.000000 1.000000 2.000000\n");

        Ok(())
    }

    #[test]
    fn test_write_2d() -> io::Result<()> {
        let mut dest = Vec::new();
        write_sfs(&mut dest, &sfs2d![[0., 1., 2.], [3., 4., 5.]])?;

        eprintln!("{}", String::from_utf8(dest.clone()).unwrap());

        assert_eq!(
            dest,
            b"#SHAPE=<2/3>\n0.000000 1.000000 2.000000 3.000000 4.000000 5.000000\n",
        );

        Ok(())
    }
}
