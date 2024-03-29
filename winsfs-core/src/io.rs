//! Utilities for working with SAF files stored on disk.
//!
//! To read and write standard SAF files, see the [`angsd_saf`] crate. This module contains
//! utilities based on that for doing SFS estimation from files kept on disk.

use std::{fs::File, io, path::Path};

pub use angsd_saf::{
    record::Id,
    version::{Version, V3, V4},
    ReadStatus,
};

use crate::{em::Sites, saf::Site};

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
    fn read_site<const D: usize>(&mut self, buf: &mut Site<D>) -> io::Result<ReadStatus>;

    /// Reads a single site into the provided buffer without normalising out of log-space.
    ///
    /// See also documentation for [`Self::read_site`].
    fn read_site_unnormalised<const D: usize>(
        &mut self,
        buf: &mut Site<D>,
    ) -> io::Result<ReadStatus>;

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
    fn read_site<const D: usize>(&mut self, buf: &mut Site<D>) -> io::Result<ReadStatus> {
        <T as ReadSite>::read_site(*self, buf)
    }

    fn read_site_unnormalised<const D: usize>(
        &mut self,
        buf: &mut Site<D>,
    ) -> io::Result<ReadStatus> {
        <T as ReadSite>::read_site_unnormalised(*self, buf)
    }
}

impl<'a, T> Sites for &'a mut T
where
    T: Sites,
{
    fn sites(&self) -> usize {
        <T as Sites>::sites(*self)
    }
}

/// An intersecting SAF reader.
///
/// This a wrapper around the [`Intersect`](angsd_saf::Intersect) with a static number of readers
/// and holds its read buffers internally.
pub struct Intersect<const D: usize, R, V>
where
    V: Version,
{
    // D readers in inner intersect is maintained as invariant
    inner: angsd_saf::Intersect<R, V>,
    bufs: [angsd_saf::Record<Id, V::Item>; D],
}

impl<const D: usize, R, V> Intersect<D, R, V>
where
    R: io::BufRead + io::Seek,
    V: Version,
{
    /// Returns the inner reader.
    pub fn get(&self) -> &angsd_saf::Intersect<R, V> {
        &self.inner
    }

    /// Returns a mutable reference to the the inner reader.
    pub fn get_mut(&mut self) -> &mut angsd_saf::Intersect<R, V> {
        &mut self.inner
    }

    /// Returns the inner reader, consuming `self`.
    pub fn into_inner(self) -> angsd_saf::Intersect<R, V> {
        self.inner
    }

    /// Creates a new reader.
    pub fn new(readers: [angsd_saf::Reader<R, V>; D]) -> Self {
        let inner = angsd_saf::Intersect::new(readers.into());
        let bufs = inner
            .create_record_bufs()
            .try_into()
            .map_err(|_| ())
            .unwrap();

        Self { inner, bufs }
    }
}

impl<const D: usize, V> Intersect<D, io::BufReader<File>, V>
where
    V: Version,
{
    /// Creates a new reader from a collection of member file paths.
    ///
    /// The stream will be positioned immediately after the magic number.
    pub fn from_paths<P>(paths: &[P; D]) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        paths
            .iter()
            .map(|p| angsd_saf::reader::Builder::<V>::default().build_from_member_path(p))
            .collect::<io::Result<Vec<_>>>()
            .map(|vec| Self::new(vec.try_into().map_err(|_| ()).unwrap()))
    }
}

impl<const N: usize, R> ReadSite for Intersect<N, R, V3>
where
    R: io::BufRead + io::Seek,
{
    fn read_site<const D: usize>(&mut self, buf: &mut Site<D>) -> io::Result<ReadStatus> {
        let status = self.read_site_unnormalised(buf)?;

        buf.iter_mut().for_each(|x| *x = x.exp());

        Ok(status)
    }

    fn read_site_unnormalised<const D: usize>(
        &mut self,
        buf: &mut Site<D>,
    ) -> io::Result<ReadStatus> {
        // TODO: This should really be type-enforced somehow, but requires a bit more work.
        assert_eq!(N, D);

        let status = self.inner.read_records(&mut self.bufs)?;

        let src = self.bufs.iter().map(|record| record.item());
        copy_from_slices(src, buf.as_mut_slice());

        Ok(status)
    }
}

impl<const N: usize, R> ReadSite for Intersect<N, R, V4>
where
    R: io::BufRead + io::Seek,
{
    fn read_site<const D: usize>(&mut self, buf: &mut Site<D>) -> io::Result<ReadStatus> {
        let status = self.read_site_unnormalised(buf)?;

        buf.iter_mut().for_each(|x| *x = x.exp());

        Ok(status)
    }

    fn read_site_unnormalised<const D: usize>(
        &mut self,
        buf: &mut Site<D>,
    ) -> io::Result<ReadStatus> {
        // TODO: This should really be type-enforced somehow, but requires a bit more work.
        assert_eq!(N, D);

        let status = self.inner.read_records(&mut self.bufs)?;

        let alleles_iter = self
            .inner
            .get_readers()
            .iter()
            .map(|reader| reader.index().alleles());
        let src = self
            .bufs
            .iter_mut()
            .zip(alleles_iter)
            .map(|(record, alleles)| record.item().clone().into_full(alleles, f32::NEG_INFINITY));
        copy_from_slices(src, buf.as_mut_slice());

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
