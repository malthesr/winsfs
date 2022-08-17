//! Reading and writing for SFS in the numpy npy format.
//!
//! The npy format is described [here][spec]. Only a subset required to read/write an SFS
//! is supported. Only simple type descriptors for the basic integer and float types are
//! supported. In addition, only reading/writing C-order is supported; trying to read a
//! Fortran-order npy file will result in a run-time error.
//!
//! [spec]: https://numpy.org/neps/nep-0001-npy-format.html

use std::{fs::File, io, path::Path};

use crate::sfs::{
    generics::{Normalisation, Shape},
    DynUSfs, SfsBase,
};

mod header;
use header::{Endian, Header, HeaderDict, Type, TypeDescriptor, Version};

/// Reads an SFS in npy format from a reader.
///
/// The stream is assumed to be positioned at the start.
pub fn read_sfs<R>(reader: &mut R) -> io::Result<DynUSfs>
where
    R: io::BufRead,
{
    let header = Header::read(reader)?;
    let dict = header.dict;

    match (dict.type_descriptor, dict.fortran_order) {
        (_, true) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Fortran order not supported when reading npy",
        )),
        (descr, false) => {
            let values = descr.read(reader)?;

            DynUSfs::from_vec_shape(values, dict.shape.into_boxed_slice()).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "npy shape does not fit values")
            })
        }
    }
}

/// Reads an SFS in npy format from a file path.
pub fn read_sfs_from_path<P>(path: P) -> io::Result<DynUSfs>
where
    P: AsRef<Path>,
{
    let mut reader = File::open(path).map(io::BufReader::new)?;
    read_sfs(&mut reader)
}

/// Writes an SFS in npy format to a writer.
pub fn write_sfs<W, S, N>(writer: &mut W, sfs: &SfsBase<S, N>) -> io::Result<()>
where
    W: io::Write,
    S: Shape,
    N: Normalisation,
{
    let header = Header::new(
        Version::V1,
        HeaderDict::new(
            TypeDescriptor::new(Endian::Little, Type::F8),
            false,
            sfs.shape().as_ref().to_vec(),
        ),
    );

    header.write(writer)?;

    for v in sfs.iter() {
        writer.write_all(&v.to_le_bytes())?;
    }

    Ok(())
}

/// Writes an SFS in npy format to a file path.
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
