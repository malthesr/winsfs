//! Utilities for working with SAF files stored on disk.
//!
//! To read and write standard SAF files, see the [`angsd_saf`] crate. This module contains
//! utilities based on that for doing SFS estimation from files kept on disk.

use std::{fs::File, io, path::Path};

use angsd_saf as saf;
pub use saf::{
    record::{Id, Likelihoods},
    version::{Version, V3, V4},
    ReadStatus,
};

mod adaptors;
pub use adaptors::{Enumerate, Take};

pub mod shuffle;

/// A type that can read SAF sites from a source.
pub trait ReadSite {
    /// Reads a single site into the provided buffer.
    ///
    /// In the multi-dimensional case, values should be read from the first population into the
    /// start of the buffer, then the next population, and so on. That is, providing
    /// [`Site::as_mut_slice`](crate::saf::Site::as_mut_slice) as a buffer will be correct, given
    /// a site of correct shape for the underlying reader.
    fn read_site(&mut self, buf: &mut [f32]) -> io::Result<ReadStatus>;

    /// Returns a reader adaptor which counts the number of sites read.
    fn enumerate(self) -> Enumerate<Self>
    where
        Self: Sized,
    {
        Enumerate::new(self)
    }

    /// Returns a reader adaptor which limits the number of sites read.
    fn take(self, max_sites: usize) -> Take<Enumerate<Self>>
    where
        Self: Sized,
    {
        Take::new(Enumerate::new(self), max_sites)
    }
}

/// A reader type that can return to the beginning of the data.
pub trait Rewind: ReadSite {
    /// Returns `true` if reader has reached the end of the data, `false` otherwise.
    #[allow(clippy::wrong_self_convention)]
    fn is_done(&mut self) -> io::Result<bool>;

    /// Positions reader at the beginning of the data.
    ///
    /// The stream should be positioned so as to be ready to call [`ReadSite::read_site`].
    /// In particular, the stream should be positioned past any magic number, headers, etc.
    fn rewind(&mut self) -> io::Result<()>;
}

impl<'a, T> Rewind for &'a mut T
where
    T: Rewind,
{
    fn is_done(&mut self) -> io::Result<bool> {
        <T as Rewind>::is_done(*self)
    }

    fn rewind(&mut self) -> io::Result<()> {
        <T as Rewind>::rewind(*self)
    }
}

impl<'a, T> ReadSite for &'a mut T
where
    T: ReadSite,
{
    fn read_site(&mut self, buf: &mut [f32]) -> io::Result<ReadStatus> {
        <T as ReadSite>::read_site(*self, buf)
    }
}

/// An intersecting SAF reader.
///
/// This a wrapper around the [`Intersect`](saf::Intersect) type that can
/// implement [`ReadSite`]. This can be used to stream through the intersecting sites of multiple
/// SAF files when shuffling is not required.
pub struct Intersect<R, V>
where
    V: Version,
{
    inner: saf::Intersect<R, V>,
    bufs: Vec<saf::Record<Id, V::Item>>,
}

impl<R, V> Intersect<R, V>
where
    R: io::BufRead + io::Seek,
    V: Version,
{
    /// Returns the inner reader.
    pub fn get(&self) -> &saf::Intersect<R, V> {
        &self.inner
    }

    /// Returns a mutable reference to the the inner reader.
    pub fn get_mut(&mut self) -> &mut saf::Intersect<R, V> {
        &mut self.inner
    }

    /// Returns the inner reader, consuming `self`.
    pub fn into_inner(self) -> saf::Intersect<R, V> {
        self.inner
    }

    /// Creates a new reader.
    pub fn new(readers: Vec<saf::Reader<R, V>>) -> Self {
        let inner = saf::Intersect::new(readers);
        let bufs = inner.create_record_bufs();

        Self { inner, bufs }
    }
}

impl<V> Intersect<io::BufReader<File>, V>
where
    V: Version,
{
    /// Creates a new reader from a collection of member file paths.
    ///
    /// The stream will be positioned immediately after the magic number.
    pub fn from_paths<P>(paths: &[P]) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        paths
            .iter()
            .map(|p| saf::reader::Builder::<V>::default().build_from_member_path(p))
            .collect::<io::Result<Vec<_>>>()
            .map(Self::new)
    }
}

impl<R, V> From<saf::Intersect<R, V>> for Intersect<R, V>
where
    R: io::BufRead + io::Seek,
    V: Version,
{
    fn from(inner: saf::Intersect<R, V>) -> Self {
        let bufs = inner.create_record_bufs();

        Self { inner, bufs }
    }
}

impl<R> ReadSite for Intersect<R, V3>
where
    R: io::BufRead + io::Seek,
{
    fn read_site(&mut self, buf: &mut [f32]) -> io::Result<ReadStatus> {
        let status = self.inner.read_records(&mut self.bufs)?;

        let src = self.bufs.iter().map(|record| record.item());
        copy_from_slices(src, buf);

        buf.iter_mut().for_each(|x| *x = x.exp());

        Ok(status)
    }
}

impl<R> ReadSite for Intersect<R, V4>
where
    R: io::BufRead + io::Seek,
{
    fn read_site(&mut self, buf: &mut [f32]) -> io::Result<ReadStatus> {
        let status = self.inner.read_records(&mut self.bufs)?;

        let alleles_iter = self
            .inner
            .get_readers()
            .iter()
            .map(|reader| reader.index().alleles());
        let src =
            self.bufs.iter().zip(alleles_iter).map(|(record, alleles)| {
                record.item().clone().into_full(alleles, f32::NEG_INFINITY)
            });
        copy_from_slices(src, buf);

        buf.iter_mut().for_each(|x| *x = x.exp());

        Ok(status)
    }
}

/// Copy multiple slices into successive subslices of a new slice.
///
/// `dest` is assumed to have length equal to the sum of the lengths of slice sin `src`.
fn copy_from_slices<I, T>(src: I, dest: &mut [T])
where
    I: IntoIterator,
    I::Item: AsRef<[T]>,
    T: Copy,
{
    let mut offset = 0;
    for s in src {
        let n = s.as_ref().len();
        dest[offset..][..n].copy_from_slice(s.as_ref());
        offset += n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copy_from_slices() {
        let src = vec![&[0, 1][..], &[2, 3, 4, 5]];
        let mut dest = vec![0; 6];
        copy_from_slices(src, dest.as_mut_slice());
        assert_eq!(dest, (0..6).collect::<Vec<_>>());
    }
}
