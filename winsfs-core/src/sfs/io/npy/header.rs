use std::{
    fmt, io,
    str::{self, FromStr},
};

mod parse;

/// The npy magic number.
const MAGIC: [u8; 6] = *b"\x93NUMPY";

/// The alignment used for padding the npy header.
///
/// From the spec:
///     "[the header] is terminated by a newline (\n) and padded with spaces (\x20)  to make the
///     total of len(magic string) + 2 + len(length) + HEADER_LEN be evenly divisible by 64 for
///     alignment purposes."
const ALIGN: usize = 64;

/// A npy header.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Header {
    pub version: Version,
    pub dict: HeaderDict,
}

impl Header {
    /// Creates a new npy header.
    pub fn new(version: Version, dict: HeaderDict) -> Self {
        Self { version, dict }
    }

    /// Reads a npy header from a reader.
    ///
    /// The stream is assumed to be positioned at the start.
    pub fn read<R>(reader: &mut R) -> io::Result<Self>
    where
        R: io::BufRead,
    {
        let mut magic_buf = [0; MAGIC.len()];
        reader.read_exact(&mut magic_buf)?;
        if magic_buf != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unexpected npy magic number",
            ));
        }

        let mut version_buf = [0; 2];
        reader.read_exact(&mut version_buf)?;
        let version = Version::from_header_bytes(version_buf).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid npy version specification",
            )
        })?;

        let header_len = version.read_header_len(reader)?;
        let mut dict_buf = vec![0; header_len];
        reader.read_exact(&mut dict_buf)?;

        let dict_str =
            str::from_utf8(&dict_buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let dict = HeaderDict::from_str(dict_str)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        Ok(Self::new(version, dict))
    }

    /// Writes a npy header to a writer.
    pub fn write<W>(&self, writer: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        writer.write_all(&MAGIC)?;
        let version_bytes = self.version.to_header_bytes();
        writer.write_all(&version_bytes)?;

        let fmt_dict = self.dict.to_string();

        let len = MAGIC.len()
            + version_bytes.len()
            + self.version.header_len_bytes_len()
            + fmt_dict.len();
        let rem = len % ALIGN;
        let pad_len = if rem == 0 { 0 } else { ALIGN - rem };
        assert_eq!((len + pad_len) % ALIGN, 0);

        let header_len = fmt_dict.len() + pad_len;
        self.version.write_header_len(header_len, writer)?;

        writer.write_all(&fmt_dict.into_bytes())?;

        let mut pad = vec![b' '; pad_len];
        pad[pad_len - 1] = b'\n';
        writer.write_all(&pad[..])
    }
}

/// A npy header literal dict.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct HeaderDict {
    pub type_descriptor: TypeDescriptor,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
}

impl HeaderDict {
    /// Creates a new npy literal header dict.
    pub fn new(type_descriptor: TypeDescriptor, fortran_order: bool, shape: Vec<usize>) -> Self {
        Self {
            type_descriptor,
            fortran_order,
            shape,
        }
    }
}

impl fmt::Display for HeaderDict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let descr = self.type_descriptor.to_string();
        let fortran_order = if self.fortran_order { "True" } else { "False" };
        let shape_fmt = self
            .shape
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let shape = format!("({shape_fmt},)");

        write!(
            f,
            "{{'descr': '{descr}', 'fortran_order': {fortran_order}, 'shape': {shape}, }}"
        )
    }
}

impl FromStr for HeaderDict {
    type Err = ParseHeaderError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut type_descriptor: Option<TypeDescriptor> = None;
        let mut fortran_order: Option<bool> = None;
        let mut shape: Option<Vec<usize>> = None;

        for entry in parse::parse_header_dict(s)? {
            match entry {
                parse::Entry::Descr(val) => {
                    type_descriptor = Some(val);
                }
                parse::Entry::FortranOrder(val) => {
                    fortran_order = Some(val);
                }
                parse::Entry::Shape(val) => {
                    shape = Some(val);
                }
            }
        }

        match (type_descriptor, fortran_order, shape) {
            (Some(type_descriptor), Some(fortran_order), Some(shape)) => {
                Ok(Self::new(type_descriptor, fortran_order, shape))
            }
            _ => Err(ParseHeaderError(s.to_string())),
        }
    }
}

/// A npy version.
///
/// Later versions are defined relative to the first:
/// Version 2.0 changes the HEADER_LEN field to be four bytes instead of two, and
/// version 3.0 adds support for UTF8 in the header dict.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum Version {
    V1,
    V2,
    V3,
}

impl Version {
    /// Creates a new version from the header bytes specifying the version.
    fn from_header_bytes(bytes: [u8; 2]) -> Result<Self, [u8; 2]> {
        match bytes {
            [1, _] => Ok(Self::V1),
            [2, _] => Ok(Self::V2),
            [3, _] => Ok(Self::V3),
            _ => Err(bytes),
        }
    }

    /// Reads the header_len from a reader.
    fn read_header_len<R>(&self, reader: &mut R) -> io::Result<usize>
    where
        R: io::BufRead,
    {
        match self {
            Version::V1 => {
                let mut header_len_buf = [0; 2];
                reader.read_exact(&mut header_len_buf)?;
                Ok(u16::from_le_bytes(header_len_buf).into())
            }
            Version::V2 | Version::V3 => {
                let mut header_len_buf = [0; 4];
                reader.read_exact(&mut header_len_buf)?;
                Ok(usize::try_from(u32::from_le_bytes(header_len_buf))
                    .expect("cannot convert npy u32 header_len to usize"))
            }
        }
    }

    /// Creates a the header bytes corresponding to the version.
    fn to_header_bytes(&self) -> [u8; 2] {
        match self {
            Version::V1 => [1, 0],
            Version::V2 => [2, 0],
            Version::V3 => [3, 0],
        }
    }

    /// Writes the header_len to a writer.
    fn write_header_len<W>(&self, header_len: usize, writer: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        match self {
            Version::V1 => writer.write_all(
                &u16::try_from(header_len)
                    .expect("cannot convert npy header_len to u16")
                    .to_le_bytes(),
            ),
            Version::V2 | Version::V3 => writer.write_all(
                &u32::try_from(header_len)
                    .expect("cannot convert npy header_len to u16")
                    .to_le_bytes(),
            ),
        }
    }

    /// Returns the number of bytes taken up by the header_len in the given version.
    fn header_len_bytes_len(&self) -> usize {
        match self {
            Version::V1 => 2,
            Version::V2 | Version::V3 => 4,
        }
    }
}

/// A npy type descriptor.
///
/// The type descriptor contains the endianness, the size, and the kind of type. For example '<f8',
/// indicates a little-endian 8-byte float, while '>i4' is a big-endian 4-byte
/// signed integer, and '<u2' is a little-endian two-byte unsigned integer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct TypeDescriptor {
    endian: Endian,
    ty: Type,
}

macro_rules! impl_get_read_fn {
    ($ty:ty, $fn:ident) => {{
        |reader: &mut R| {
            let mut buf = [0; std::mem::size_of::<$ty>()];
            reader.read_exact(&mut buf)?;
            Ok(<$ty>::$fn(buf) as f64)
        }
    }};
}

impl TypeDescriptor {
    /// Returns a new type descriptor.
    pub fn new(endian: Endian, ty: Type) -> Self {
        Self { endian, ty }
    }

    /// Returns a function that can read the described type from a reader and cast it to a `f64`.
    fn get_read_fn<R>(&self) -> impl Fn(&mut R) -> io::Result<f64>
    where
        R: io::BufRead,
    {
        match (&self.endian, &self.ty) {
            (Endian::Little, Type::F4) => impl_get_read_fn!(f32, from_le_bytes),
            (Endian::Little, Type::F8) => impl_get_read_fn!(f64, from_le_bytes),
            (Endian::Little, Type::I1) => impl_get_read_fn!(i8, from_le_bytes),
            (Endian::Little, Type::I2) => impl_get_read_fn!(i16, from_le_bytes),
            (Endian::Little, Type::I4) => impl_get_read_fn!(i32, from_le_bytes),
            (Endian::Little, Type::I8) => impl_get_read_fn!(i64, from_le_bytes),
            (Endian::Little, Type::U1) => impl_get_read_fn!(u8, from_le_bytes),
            (Endian::Little, Type::U2) => impl_get_read_fn!(u16, from_le_bytes),
            (Endian::Little, Type::U4) => impl_get_read_fn!(u32, from_le_bytes),
            (Endian::Little, Type::U8) => impl_get_read_fn!(u64, from_le_bytes),
            (Endian::Big, Type::F4) => impl_get_read_fn!(f32, from_be_bytes),
            (Endian::Big, Type::F8) => impl_get_read_fn!(f64, from_be_bytes),
            (Endian::Big, Type::I1) => impl_get_read_fn!(i8, from_be_bytes),
            (Endian::Big, Type::I2) => impl_get_read_fn!(i16, from_be_bytes),
            (Endian::Big, Type::I4) => impl_get_read_fn!(i32, from_be_bytes),
            (Endian::Big, Type::I8) => impl_get_read_fn!(i64, from_be_bytes),
            (Endian::Big, Type::U1) => impl_get_read_fn!(u8, from_be_bytes),
            (Endian::Big, Type::U2) => impl_get_read_fn!(u16, from_be_bytes),
            (Endian::Big, Type::U4) => impl_get_read_fn!(u32, from_be_bytes),
            (Endian::Big, Type::U8) => impl_get_read_fn!(u64, from_be_bytes),
        }
    }

    /// Reads the described type (cast to `f64`) from a reader into a provided buffer.
    pub(super) fn read<R>(&self, reader: &mut R) -> io::Result<Vec<f64>>
    where
        R: io::BufRead,
    {
        let read_fn = self.get_read_fn();

        let mut values = Vec::new();

        // TODO: This can use BufRead::has_data_left if/once stabilised,
        // see github.com/rust-lang/rust/issues/86423
        while !reader.fill_buf()?.is_empty() {
            values.push(read_fn(reader)?)
        }

        Ok(values)
    }
}

impl fmt::Display for TypeDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let endian_str = match self.endian {
            Endian::Little => "<",
            Endian::Big => ">",
        };

        let type_str = match self.ty {
            Type::F4 => "f4",
            Type::F8 => "f8",
            Type::I1 => "i1",
            Type::I2 => "i2",
            Type::I4 => "i4",
            Type::I8 => "i8",
            Type::U1 => "u1",
            Type::U2 => "u2",
            Type::U4 => "u4",
            Type::U8 => "u8",
        };

        write!(f, "{endian_str}{type_str}")
    }
}

impl FromStr for TypeDescriptor {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let create_err = || Err(format!("invalid type descriptor '{s}'"));

        if s.len() != 3 {
            return create_err();
        }

        let (endian_str, type_str) = s.split_at(1);

        let endian = match endian_str {
            "<" | "|" => Endian::Little,
            ">" => Endian::Big,
            _ => return create_err(),
        };

        let ty = match type_str {
            "f4" => Type::F4,
            "f8" => Type::F8,
            "i1" => Type::I1,
            "i2" => Type::I2,
            "i4" => Type::I4,
            "i8" => Type::I8,
            "u1" => Type::U1,
            "u2" => Type::U2,
            "u4" => Type::U4,
            "u8" => Type::U8,
            _ => return create_err(),
        };

        Ok(Self::new(endian, ty))
    }
}

/// A byte encoding endianness.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum Endian {
    Little,
    Big,
}

/// A type and size.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum Type {
    F4,
    F8,
    I1,
    I2,
    I4,
    I8,
    U1,
    U2,
    U4,
    U8,
}

/// An error associated with parsing the npy format header.
#[derive(Debug, Eq, PartialEq)]
pub struct ParseHeaderError(String);

impl fmt::Display for ParseHeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "failed to parse '{}' as npy format header", self.0)
    }
}

impl std::error::Error for ParseHeaderError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_descriptor_read() -> io::Result<()> {
        let src: Vec<u8> = (0i16..10).flat_map(|x| x.to_be_bytes()).collect();
        let expected: Vec<f64> = (0..10).map(|x| x as f64).collect();
        assert_eq!(
            TypeDescriptor::new(Endian::Big, Type::I2).read(&mut &src[..])?,
            expected
        );

        Ok(())
    }

    #[test]
    fn test_parse_header_dict() {
        assert_eq!(
            "{ 'descr': '<f8', 'shape': (15, 3), 'fortran_order': False }".parse(),
            Ok(HeaderDict::new(
                TypeDescriptor::new(Endian::Little, Type::F8),
                false,
                vec![15, 3]
            ))
        )
    }

    #[test]
    fn test_display_header_dict() {
        assert_eq!(
            HeaderDict::new(
                TypeDescriptor::new(Endian::Big, Type::U4),
                true,
                vec![3, 1, 2]
            )
            .to_string(),
            String::from("{'descr': '>u4', 'fortran_order': True, 'shape': (3, 1, 2,), }"),
        )
    }

    #[test]
    fn test_read_header() -> io::Result<()> {
        let header_dict = HeaderDict::new(
            TypeDescriptor::new(Endian::Little, Type::F8),
            false,
            vec![2, 3],
        );

        let mut src = vec![
            147, 78, 85, 77, 80, 89, // magic
            1, 0, // version 1.0
            118, 0, // header_len (2 bytes in version 1.0)
        ];
        src.extend(header_dict.to_string().as_bytes());
        src.extend([32; 58]); // whitespace padding for alignment
        src.extend([10]); // newline

        assert_eq!(
            Header::read(&mut &src[..])?,
            Header::new(Version::V1, header_dict)
        );

        Ok(())
    }

    #[test]
    fn test_write_header() -> io::Result<()> {
        let header_dict =
            HeaderDict::new(TypeDescriptor::new(Endian::Big, Type::F4), false, vec![2]);
        let fmt_dict = header_dict.to_string();

        let header = Header::new(Version::V2, header_dict);
        let mut dest = Vec::new();
        header.write(&mut dest)?;

        let mut expected = vec![
            147, 78, 85, 77, 80, 89, // magic
            2, 0, // version 2.0
            116, 0, 0, 0, // header_len (4 bytes in version 2.0)
        ];
        expected.extend(fmt_dict.as_bytes());
        expected.extend([32; 58]); // whitespace padding for alignment
        expected.extend([10]); // newline

        assert_eq!(dest, expected);

        Ok(())
    }
}
