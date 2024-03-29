use std::{fs::File, io, num::NonZeroUsize, path::Path, thread};

use angsd_saf as saf;
use saf::version::Version;

use winsfs_core::{
    em::likelihood::LogLikelihood,
    io::{shuffle, Intersect, ReadSite},
    saf::Saf,
    sfs::Sfs,
};

use crate::{estimate::Format, utils::join};

/// A collection of SAF file readers from one of the supported SAF file formats.
pub enum Readers<const D: usize, R> {
    /// A collection of full SAF V3 readers.
    Standard([saf::ReaderV3<R>; D]),
    /// A collection of banded SAF V4 readers.
    Banded([saf::ReaderV4<R>; D]),
}

impl<const D: usize, R> Readers<D, R>
where
    R: io::BufRead + io::Seek,
{
    /// Returns the number of intersecting sites in the readers.
    ///
    /// Note that this requires taking a full pass through the readers to count, as the number of
    /// intersections cannot be known ahead of time. The exception is if there is only a single
    /// reader, in which case the number of sites can be taken directly from the index.
    pub fn count_sites(self) -> io::Result<usize> {
        match self {
            Self::Standard(readers) => readers.count_sites(),
            Self::Banded(readers) => readers.count_sites(),
        }
    }

    /// Returns the log-likelihood of an SFS given the data in readers, as well as the number of
    /// (intersecting) sites in the readers.
    pub fn log_likelihood(self, sfs: Sfs<D>) -> io::Result<(LogLikelihood, usize)> {
        match self {
            Self::Standard(readers) => readers.log_likelihood(sfs),
            Self::Banded(readers) => readers.log_likelihood(sfs),
        }
    }

    /// Returns the shape of the SAF to be read.
    pub fn shape(&self) -> [usize; D] {
        match self {
            Self::Standard(readers) => readers.shape(),
            Self::Banded(readers) => readers.shape(),
        }
    }

    /// Pseudo-shuffles the sites in the readers into the provided shuffle writer.
    pub fn shuffle(self, writer: shuffle::Writer<io::BufWriter<File>>) -> io::Result<()> {
        match self {
            Self::Standard(readers) => {
                let intersect = Intersect::new(readers);
                writer.write_intersect(intersect)
            }
            Self::Banded(readers) => {
                let intersect = Intersect::new(readers);
                writer.write_intersect(intersect)
            }
        }
    }

    /// Reads a SAF from the readers.
    ///
    /// Note that this will read a full SAF even if the version is V4. In other words, even if the
    /// input is banded, a full SAF is read.
    pub fn read_saf(self) -> io::Result<Saf<D>> {
        log::info!(
            target: "init",
            "Reading (intersecting) sites in input SAF files into memory",
        );

        let saf = match self {
            Readers::Standard(readers) => Saf::read(readers),
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

impl<const D: usize> Readers<D, io::BufReader<File>> {
    /// Returns a new collection of SAF file readers from member file paths.
    ///
    /// This will automatically attempt to infer the SAF file version based on the magic number of
    /// the first provided path. An error is thrown if the format cannot be inferred based on the
    /// magic number.
    pub fn from_member_paths<P>(paths: &[P; D], threads: usize) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        let mut file = File::open(&paths[0])?;
        let format = Format::infer_from_magic(&mut file)?;

        log::info!(
            target: "init",
            "Opening input {format} ({}) SAF files:\n\t{}",
            format.version_string(),
            join(paths.iter().map(|p| p.as_ref().display()), "\n\t"),
        );

        match format {
            Format::Standard => create_readers(paths, threads).map(Self::Standard),
            Format::Banded => create_readers(paths, threads).map(Self::Banded),
            Format::Shuffled => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot construct joint reader from shuffled format file",
            )),
        }
    }
}

/// A helper extension trait for SAF readers with different file formats.
///
/// This helps reduce code duplication in the [`Readers`] enum. See the methods on the enum for
/// documentation.
trait ReadersExt<const D: usize, R, V>
where
    V: Version,
{
    fn count_sites(self) -> io::Result<usize>;

    fn log_likelihood(self, sfs: Sfs<D>) -> io::Result<(LogLikelihood, usize)>
    where
        winsfs_core::io::Intersect<D, R, V>: ReadSite;

    fn shape(&self) -> [usize; D];
}

impl<const D: usize, R, V> ReadersExt<D, R, V> for [saf::Reader<R, V>; D]
where
    R: io::BufRead + io::Seek,
    V: Version,
{
    fn count_sites(self) -> io::Result<usize> {
        if let [single_reader] = &self[..] {
            return Ok(single_reader.index().total_sites());
        }

        let mut intersect = saf::reader::Intersect::new(self.into());

        let mut bufs = intersect.create_record_bufs();

        let mut sites = 0;
        while intersect.read_records(&mut bufs)?.is_not_done() {
            sites += 1;
        }
        Ok(sites)
    }

    fn log_likelihood(self, sfs: Sfs<D>) -> io::Result<(LogLikelihood, usize)>
    where
        winsfs_core::io::Intersect<D, R, V>: ReadSite,
    {
        let mut intersect = winsfs_core::io::Intersect::new(self);
        sfs.stream_log_likelihood(&mut intersect)
            .map(|sum_of| sum_of.into())
    }

    fn shape(&self) -> [usize; D] {
        self.iter()
            .map(|reader| reader.index().alleles() + 1)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}

/// Helper function to set up a collection of readers with the provided number of threads.
fn create_readers<const D: usize, P, V>(
    paths: &[P; D],
    threads: usize,
) -> io::Result<[saf::Reader<io::BufReader<File>, V>; D]>
where
    P: AsRef<Path>,
    V: saf::version::Version,
{
    let threads = NonZeroUsize::new(threads).unwrap_or(thread::available_parallelism()?);

    log::debug!(target: "init", "Using {threads} threads for reading");

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
