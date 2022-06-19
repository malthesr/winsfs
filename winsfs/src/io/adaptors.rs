use std::io;

use super::{ReadSite, ReadStatus};

/// A type that keeps track of how many sites have been read from the underlying source.
///
/// Constructed using [`ReadSite::enumerate`].
pub struct Enumerate<R> {
    inner: R,
    sites_read: usize,
}

impl<R> Enumerate<R>
where
    R: ReadSite,
{
    pub(super) fn new(inner: R) -> Self {
        Self {
            inner,
            sites_read: 0,
        }
    }

    /// Returns the number of sites read from the underlying source.
    pub fn sites_read(&self) -> usize {
        self.sites_read
    }

    /// Returns a reader adaptor which limits the number of sites read.
    ///
    /// See also [`ReadSite::take`].
    pub fn take(self, max_sites: usize) -> Take<Self> {
        Take::new(self, max_sites)
    }
}

impl<R> ReadSite for Enumerate<R>
where
    R: ReadSite,
{
    fn read_site(&mut self, buf: &mut [f32]) -> io::Result<ReadStatus> {
        self.sites_read += 1;
        self.inner.read_site(buf)
    }
}

/// A type that limits the number of sites that can be read from the underlying source.
///
/// Constructed using [`ReadSite::take`] or [`Enumerate::take`].
pub struct Take<R> {
    inner: R,
    max_sites: usize,
}

impl<R> Take<Enumerate<R>>
where
    R: ReadSite,
{
    pub(super) fn new(inner: Enumerate<R>, max_sites: usize) -> Self {
        Self { inner, max_sites }
    }

    /// Returns the maximum number of sites that can be read from the underlying source.
    pub fn max_sites(&self) -> usize {
        self.max_sites
    }

    /// Returns the number of sites read from the underlying source.
    pub fn sites_read(&self) -> usize {
        self.inner.sites_read()
    }
}

impl<R> ReadSite for Take<Enumerate<R>>
where
    R: ReadSite,
{
    fn read_site(&mut self, buf: &mut [f32]) -> io::Result<ReadStatus> {
        if self.inner.sites_read() < self.max_sites {
            self.inner.read_site(buf)
        } else {
            Ok(ReadStatus::Done)
        }
    }
}
