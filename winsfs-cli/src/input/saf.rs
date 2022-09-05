use std::{fs::File, io, num::NonZeroUsize, path::Path, thread};

use angsd_saf as saf;
use saf::version::{Version as _, V3, V4};

use winsfs_core::saf::Saf;

use crate::utils::join;

/// A collection of SAF file readers from one of the supported SAF file formats.
pub enum Readers<const N: usize, R> {
    /// A collection of full SAF V3 readers.
    Full([saf::ReaderV3<R>; N]),
    /// A collection of banded SAF V4 readers.
    Banded([saf::ReaderV4<R>; N]),
}

impl<const N: usize, R> Readers<N, R>
where
    R: io::BufRead + io::Seek,
{
    /// Reads a SAF from the readers.
    ///
    /// Note that this will read a full SAF even if the version is V4. In other words, even if the
    /// input is banded, a full SAF is read.
    pub fn read_saf(self) -> io::Result<Saf<N>> {
        log::info!(
            target: "init",
            "Reading (intersecting) sites in input SAF files into memory.",
        );

        let saf = match self {
            Readers::Full(readers) => Saf::read(readers),
            Readers::Banded(readers) => Saf::read_from_banded(readers),
        }?;

        log::debug!(
            target: "init",
            "Found {sites} (intersecting) sites in SAF files with shape {shape}",
            sites = saf.sites(),
            shape = join(saf.shape(), "/"),
        );

        Ok(saf)
    }
}

impl<const N: usize> Readers<N, io::BufReader<File>> {
    /// Returns a new collection of SAF file readers from member file paths.
    ///
    /// This will automatically attempt to infer the SAF file version based on the magic number of
    /// the first provided path. An error is thrown if the format cannot be inferred based on the
    /// magic number.
    pub fn from_member_paths<P>(paths: [P; N], threads: usize) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        let mut file = File::open(&paths[0])?;
        let format = Format::detect(&mut file)?;

        log::info!(
            target: "init",
            "Opening input SAF {} files:\n\t{}",
            if format == Format::Full { "v3" } else { "v4" },
            join(paths.iter().map(|p| p.as_ref().display()), "\n\t"),
        );

        match format {
            Format::Full => create_readers(paths, threads).map(Self::Full),
            Format::Banded => create_readers(paths, threads).map(Self::Banded),
        }
    }
}

/// Helper function to set up a collection of readers with the provided number of threads.
fn create_readers<const N: usize, P, V>(
    paths: [P; N],
    threads: usize,
) -> io::Result<[saf::Reader<io::BufReader<File>, V>; N]>
where
    P: AsRef<Path>,
    V: saf::version::Version,
{
    let threads = NonZeroUsize::new(threads).unwrap_or(thread::available_parallelism()?);

    // TODO: Use array::try_map when stable here
    paths
        .iter()
        .map(|p| {
            saf::reader::Builder::<V>::default()
                .set_threads(threads)
                .build_from_member_path(p)
        })
        .collect::<io::Result<Vec<_>>>()
        .map(|vec| {
            vec.try_into()
                .map_err(|_| ()) // Reader is not debug, so this is necessary to unwrap
                .unwrap()
        })
}

/// The supported SAF file version.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Format {
    Full,
    Banded,
}

impl Format {
    /// Returns the SAF file version by reading the magic number from a reader.
    ///
    /// The stream will be positioned immediately after the magic number.
    pub fn detect<R>(reader: &mut R) -> io::Result<Self>
    where
        R: io::Read,
    {
        let mut buf = [0; 8];
        reader.read_exact(&mut buf)?;

        match buf {
            V3::MAGIC_NUMBER => Ok(Self::Full),
            V4::MAGIC_NUMBER => Ok(Self::Banded),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to detect SAF file version from magic number {buf:02x?}",),
            )),
        }
    }
}
