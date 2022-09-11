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

use std::{error::Error, fmt, io, str::FromStr};

use crate::sfs::{
    generics::{DynShape, Normalisation, Shape},
    DynUSfs, Multi, SfsBase,
};

/// Parses an SFS in plain text format from the raw, flat text representation.
///
/// `s` is assumed to not contain the header.
fn parse_sfs(s: &str, shape: DynShape) -> io::Result<DynUSfs> {
    s.split_ascii_whitespace()
        .map(f64::from_str)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
        .and_then(|vec| {
            DynUSfs::from_vec_shape(vec, shape)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
        })
}

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

    parse_sfs(&buf, header.shape)
}

/// Reads a multi-SFS in plain text format from a reader.
///
/// The stream is assumed to be positioned at the start.
pub fn read_multi_sfs<R>(reader: &mut R) -> io::Result<Multi<DynUSfs>>
where
    R: io::BufRead,
{
    let header = Header::read(reader)?;

    let mut buf = String::new();
    let mut vec = Vec::new();

    while reader.read_line(&mut buf)? != 0 {
        let sfs = parse_sfs(&buf, header.shape.clone())?;
        vec.push(sfs);

        buf.clear()
    }

    Multi::try_from(vec).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
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

/// Writes a multi-SFS in plain text format to a writer.
pub fn write_multi_sfs<W, S, N>(writer: &mut W, multi: &Multi<SfsBase<S, N>>) -> io::Result<()>
where
    W: io::Write,
    S: Shape,
    N: Normalisation,
{
    let header = Header::new(multi[0].shape().as_ref().to_vec().into_boxed_slice());
    header.write(writer)?;

    for sfs in multi.iter() {
        writeln!(writer, "{}", sfs.format_flat(" ", 6))?;
    }

    Ok(())
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
    fn test_read_multi_1d() -> io::Result<()> {
        let src = b"#SHAPE=<3>\n0.0 1.0 2.0\n3.0 4.0 5.0\n";

        assert_eq!(
            read_multi_sfs(&mut &src[..])?,
            Multi::try_from(vec![
                DynUSfs::from(sfs1d![0., 1., 2.]),
                DynUSfs::from(sfs1d![3., 4., 5.])
            ])
            .unwrap()
        );

        Ok(())
    }

    #[test]
    fn test_read_multi_2d() -> io::Result<()> {
        let src = b"#SHAPE=<3/2>\n0.0 1.0 2.0 3.0 4.0 5.0\n10.0 11.0 12.0 13.0 14.0 15.0\n";

        assert_eq!(
            read_multi_sfs(&mut &src[..])?,
            Multi::try_from(vec![
                DynUSfs::from(sfs2d![[0., 1.], [2., 3.], [4., 5.]]),
                DynUSfs::from(sfs2d![[10., 11.], [12., 13.], [14., 15.]]),
            ])
            .unwrap()
        );

        Ok(())
    }

    #[test]
    fn test_write_1d() -> io::Result<()> {
        let mut dest = Vec::new();
        write_sfs(&mut dest, &sfs1d![0., 1., 2.])?;

        assert_eq!(dest, b"#SHAPE=<3>\n0.000000 1.000000 2.000000\n");

        Ok(())
    }

    #[test]
    fn test_write_2d() -> io::Result<()> {
        let mut dest = Vec::new();
        write_sfs(&mut dest, &sfs2d![[0., 1., 2.], [3., 4., 5.]])?;

        assert_eq!(
            dest,
            b"#SHAPE=<2/3>\n0.000000 1.000000 2.000000 3.000000 4.000000 5.000000\n",
        );

        Ok(())
    }

    #[test]
    fn test_write_multi_1d() -> io::Result<()> {
        let mut dest = Vec::new();
        write_multi_sfs(
            &mut dest,
            &Multi::try_from(vec![
                DynUSfs::from(sfs1d![0., 1., 2.]),
                DynUSfs::from(sfs1d![3., 4., 5.]),
            ])
            .unwrap(),
        )?;

        assert_eq!(
            dest,
            b"#SHAPE=<3>\n0.000000 1.000000 2.000000\n3.000000 4.000000 5.000000\n"
        );

        Ok(())
    }

    #[test]
    fn test_write_multi_2d() -> io::Result<()> {
        let mut dest = Vec::new();
        write_multi_sfs(
            &mut dest,
            &Multi::try_from(vec![
                DynUSfs::from(sfs2d![[0., 1.], [2., 3.]]),
                DynUSfs::from(sfs2d![[10., 11.], [12., 13.]]),
            ])
            .unwrap(),
        )?;

        assert_eq!(
            dest,
            b"#SHAPE=<2/2>\n0.000000 1.000000 2.000000 3.000000\n\
            10.000000 11.000000 12.000000 13.000000\n"
        );

        Ok(())
    }
}
