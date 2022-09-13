//! Reading and writing for SFS in the numpy npy format.
//!
//! The npy format is described [here][spec]. Only a subset required to read/write an SFS
//! is supported. Only simple type descriptors for the basic integer and float types are
//! supported. In addition, only reading/writing C-order is supported; trying to read a
//! Fortran-order npy file will result in a run-time error.
//!
//! [spec]: https://numpy.org/neps/nep-0001-npy-format.html

use std::io;

use zip::{write::FileOptions, ZipArchive, ZipWriter};

use crate::sfs::{
    generics::{Normalisation, Shape},
    DynUSfs, Multi, SfsBase,
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

/// Reads a multi-SFS in npz format from a reader.
///
/// The stream is assumed to be positioned at the start.
pub fn read_multi_sfs<R>(reader: &mut R) -> io::Result<Multi<DynUSfs>>
where
    R: io::BufRead + io::Seek,
{
    let mut zip = ZipArchive::new(reader)?;

    let vec = (0..zip.len())
        .map(|i| {
            let mut reader = zip.by_index(i).map(io::BufReader::new)?;

            read_sfs(&mut reader)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Multi::try_from(vec).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
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

/// Writes a multi-SFS in npz format to a writer.
pub fn write_multi_sfs<W, S, N>(writer: &mut W, multi: &Multi<SfsBase<S, N>>) -> io::Result<()>
where
    W: io::Seek + io::Write,
    S: Shape,
    N: Normalisation,
{
    let mut zip = ZipWriter::new(writer);
    let options = FileOptions::default();

    for (i, sfs) in multi.iter().enumerate() {
        let name = i.to_string();

        zip.start_file(name, options)?;
        write_sfs(&mut io::BufWriter::new(&mut zip), sfs)?;
    }

    let writer = zip.finish()?;
    writer.flush()
}
