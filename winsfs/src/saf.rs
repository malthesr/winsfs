//! Site allelele frequency ("SAF") likelihoods.
//!
//! SAF likelihoods represent a generalisation of genotype likelihoods from individuals to
//! populations. To estimate the N-dimensional SFS, we need the SAF likelihoods from intersecting
//! sites for those N populations. We represent those by the owned type [`Saf`] or the borrowed type
//! [`SafView`], which may represent the full data or only a smaller block of sites.
//!
//! # Examples
//!
//! ## Reading from SAF files
//!
//! It is expected that SAF likelihoods will be read from [ANGSD][angsd] SAF files.
//! A convenience constructor is provided to read all intersecting sites from `N` SAF files:
//!
//! ```no_run
//! # fn main() -> ::std::io::Result<()> {
//! // Any of the SAF member files (with extensions ".saf.gz", ".saf.pos.gz", or ".saf.idx")
//! // can be used, and the remaining will be found
//! use winsfs::Saf;
//! let paths = ["/path/to/A.saf.gz", "path/to/B.saf.idx"];
//!
//! // Read joint SAF containing the intersecting sites in populations A and B.
//! let saf = Saf::read_from_paths(paths)?;
//! # Ok(()) }
//! ```
//!
//! [angsd]: http://www.popgen.dk/angsd/index.php/ANGSD

use std::{cmp::Ordering, error::Error, fmt, io, path::Path};

use angsd_io::saf;

use rand::Rng;

use crate::ArrayExt;

pub mod iter;
use iter::{BlockIter, ParBlockIter, ParSiteIter, SiteIter};

mod site;
pub use site::{AsSiteView, Site, SiteView};

// This exists to ensure that the lifetime bound on `Lifetime` cannot be changed by an external
// implementor.
// See: sabrinajewson.org/blog/the-better-alternative-to-lifetime-gats#the-better-gats
mod sealed {
    pub trait Sealed: Sized {}
    pub struct Bounds<T>(T);
    impl<T> Sealed for Bounds<T> {}
}
use sealed::{Bounds, Sealed};

/// Stable workaround for lifetime GATs.
///
/// See the [GAT tracking issue][gat_tracking_issue] and [stabilisation PR][gat_stabilisation],
/// and in particular [this blog post][sabrina_jewson] for details on this workaround.
///
/// [gat_tracking_issue]: https://github.com/rust-lang/rust/issues/44265
/// [gat_stabilisation]: https://github.com/rust-lang/rust/pull/96709
/// [sabrina_jewson]: https://sabrinajewson.org/blog/the-better-alternative-to-lifetime-gats
pub trait Lifetime<'a, SELF: Sealed = Bounds<&'a Self>> {
    // TODO: Replace with GAT when stable.

    /// The inner item, the lifetime of which should be tied to `Self`.
    type Item;
}

/// A type that can be cheaply converted to a SAF view.
///
/// This is akin to GATified [`AsRef`] for SAF views.
pub trait AsSafView<const N: usize>: for<'a> Lifetime<'a, Item = SafView<'a, N>> {
    /// Returns a SAF view of `self`.
    fn as_saf_view(&self) -> <Self as Lifetime<'_>>::Item;
}

/// Creates a SAF matrix of a single population.
///
/// This is mainly intended for readability in doc-tests.
///
/// # Examples
///
/// ```
/// use winsfs::saf1d;
/// let saf = saf1d![
///     [0.0,  0.1,  0.2],
///     [0.3,  0.4,  0.5],
///     [0.6,  0.7,  0.8],
///     [0.9,  0.10, 0.11],
///     [0.12, 0.13, 0.14],
/// ];
/// assert_eq!(saf.sites(), 5);
/// assert_eq!(saf.shape(), [3]);
/// assert_eq!(saf.get_site(0).as_slice(), &[0.0, 0.1, 0.2]);
/// assert_eq!(saf.get_site(2).as_slice(), &[0.6, 0.7, 0.8]);
/// ```
#[macro_export]
macro_rules! saf1d {
    ($([$($x:literal),+ $(,)?]),+ $(,)?) => {{
        let (shape, vec) = $crate::matrix!($([$($x),+]),+);
        $crate::saf::Saf::new(vec, [shape[0]]).unwrap()
    }};
}

/// Creates a joint SAF matrix of two populations.
///
/// This is mainly intended for readability in doc-tests.
///
/// # Examples
///
/// ```
/// use winsfs::saf2d;
/// let saf = saf2d![
///     [0.0,  0.1,  0.2  ; 1.0, 1.1],
///     [0.3,  0.4,  0.5  ; 1.2, 1.3],
///     [0.6,  0.7,  0.8  ; 1.4, 1.5],
///     [0.9,  0.10, 0.11 ; 1.6, 1.7],
///     [0.12, 0.13, 0.14 ; 1.8, 1.9],
/// ];
/// assert_eq!(saf.sites(), 5);
/// assert_eq!(saf.shape(), [3, 2]);
/// assert_eq!(saf.get_site(0).as_slice(), &[0.0, 0.1, 0.2, 1.0, 1.1]);
/// assert_eq!(saf.get_site(2).as_slice(), &[0.6, 0.7, 0.8, 1.4, 1.5]);
/// ```
#[macro_export]
macro_rules! saf2d {
    ($([$($x:literal),+ $(,)?; $($y:literal),+ $(,)?]),+ $(,)?) => {{
        let x_cols = vec![$($crate::matrix!(count: $($x),+)),+];
        let y_cols = vec![$($crate::matrix!(count: $($y),+)),+];
        for cols in [&x_cols, &y_cols] {
            assert!(cols.windows(2).all(|w| w[0] == w[1]));
        }
        let vec = vec![$($($x),+, $($y),+),+];
        $crate::saf::Saf::new(vec, [x_cols[0], y_cols[0]]).unwrap()
    }};
}

macro_rules! impl_shared_saf_methods {
    () => {
        /// Returns the values of the SAF as a flat slice.
        ///
        /// See the [`Saf`] documentation for details on the storage order.
        pub fn as_slice(&self) -> &[f32] {
            &self.values
        }

        /// Returns an iterator over all values in the SAF.
        ///
        /// See the [`Saf`] documentation for details on the storage order.
        pub fn iter(&self) -> ::std::slice::Iter<f32> {
            self.values.iter()
        }

        /// Returns a single site in the SAF.
        pub fn get_site(&self, index: usize) -> SiteView<N> {
            let width = self.width();

            SiteView::new_unchecked(&self.values[index * width..][..width], self.shape)
        }

        /// Returns the number of sites in the SAF.
        #[inline]
        pub fn sites(&self) -> usize {
            self.values.len() / self.width()
        }

        /// Returns the shape of the SAF.
        #[inline]
        pub fn shape(&self) -> [usize; N] {
            self.shape
        }

        /// Returns the width of the SAF, i.e. the number of elements per site.
        #[inline]
        pub(self) fn width(&self) -> usize {
            self.shape.iter().sum()
        }
    };
}

/// Joint SAF likelihood matrix for `N` populations.
///
/// Internally, the matrix is represented with each site in continuous memory.
/// That is, the first values are those from the first site of all populations,
/// then comes the next site, and so on. [`Saf::shape`] gives the number of values
/// per site per population. This should only be important when operating directly
/// on the underlying storage, e.g. using [`Saf::as_slice`] or [`Saf::as_mut_slice`].
#[derive(Clone, Debug, PartialEq)]
pub struct Saf<const N: usize> {
    values: Vec<f32>,
    shape: [usize; N],
}

impl<const N: usize> Saf<N> {
    /// Returns a mutable reference to the values of the SAF as a flat slice.
    ///
    /// See the [`Saf`] documentation for details on the storage order.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.values
    }

    /// Returns an iterator over mutable references to all values in the SAF.
    ///
    /// See the [`Saf`] documentation for details on the storage order.
    pub fn iter_mut(&mut self) -> ::std::slice::IterMut<f32> {
        self.values.iter_mut()
    }

    /// Returns an iterator over blocks of sites in the SAF.
    ///
    /// If the number of sites in the SAF is not evenly divided by `block_size`,
    /// the last block will be smaller than the others.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::saf1d;
    /// let saf = saf1d![
    ///     [0.0,  0.1,  0.2],
    ///     [0.3,  0.4,  0.5],
    ///     [0.6,  0.7,  0.8],
    ///     [0.9,  0.10, 0.11],
    ///     [0.12, 0.13, 0.14],
    /// ];
    /// let mut iter = saf.iter_blocks(2);
    /// assert_eq!(
    ///     iter.next().unwrap(),
    ///     saf1d![[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]].view()
    /// );
    /// assert_eq!(
    ///     iter.next().unwrap(),
    ///     saf1d![[0.6, 0.7, 0.8], [0.9, 0.10, 0.11]].view()
    /// );
    /// assert_eq!(iter.next().unwrap(), saf1d![[0.12, 0.13, 0.14]].view());
    /// assert!(iter.next().is_none());
    /// ```
    pub fn iter_blocks(&self, block_size: usize) -> BlockIter<N> {
        BlockIter::new(self.view(), block_size)
    }

    /// Returns an iterator over the sites in the SAF.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::saf1d;
    /// let saf = saf1d![
    ///     [0.0,  0.1,  0.2],
    ///     [0.3,  0.4,  0.5],
    ///     [0.6,  0.7,  0.8],
    /// ];
    /// let mut iter = saf.iter_sites();
    /// assert_eq!(iter.next().unwrap().as_slice(), [0.0,  0.1,  0.2]);
    /// assert_eq!(iter.next().unwrap().as_slice(), [0.3,  0.4,  0.5]);
    /// assert_eq!(iter.next().unwrap().as_slice(), [0.6,  0.7,  0.8]);
    /// assert!(iter.next().is_none());
    /// ```
    pub fn iter_sites(&self) -> SiteIter<N> {
        SiteIter::new(self.view())
    }

    /// Returns a new SAF.
    ///
    /// The number of provided values must be a multiple of the sum of shapes.
    /// See the [`Saf`] documentation for details on the storage order.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::{Saf, saf2d};
    /// let vec = vec![0.0, 0.1, 0.2, 1.0, 1.1, 0.3, 0.4, 0.5, 1.2, 1.3];
    /// let shape = [3, 2];
    /// assert_eq!(
    ///     Saf::new(vec, shape).unwrap(),
    ///     saf2d![
    ///         [0.0,  0.1,  0.2 ; 1.0, 1.1],
    ///         [0.3,  0.4,  0.5 ; 1.2, 1.3],
    ///     ],
    /// );
    /// ```
    /// A [`ShapeError`] is thrown if the shape does not fit the number of values:
    ///
    /// ```
    /// use winsfs::Saf;
    /// let vec = vec![0.0, 0.1, 0.2, 1.0, 1.1, 0.3, 0.4, 0.5, 1.2, 1.3];
    /// let wrong_shape = [4, 2];
    /// assert!(Saf::new(vec, wrong_shape).is_err());
    /// ```
    pub fn new(values: Vec<f32>, shape: [usize; N]) -> Result<Self, ShapeError<N>> {
        let len = values.len();
        let width: usize = shape.iter().sum();

        if len % width == 0 {
            Ok(Self::new_unchecked(values, shape))
        } else {
            Err(ShapeError { len, shape })
        }
    }

    /// Returns a new SAF without checking that the shape fits the number of values.
    pub(crate) fn new_unchecked(values: Vec<f32>, shape: [usize; N]) -> Self {
        Self { values, shape }
    }

    /// Returns a parallel iterator over the blocks in the SAF.
    ///
    /// This is the parallel version of [`Saf::iter_blocks`].
    /// If the number of sites in the SAF is not evenly divided by `block_size`,
    /// the last block will be smaller than the others.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::{saf1d, SafView};
    /// use rayon::iter::ParallelIterator;
    /// let saf = saf1d![
    ///     [0.0,  0.1,  0.2],
    ///     [0.3,  0.4,  0.5],
    ///     [0.6,  0.7,  0.8],
    ///     [0.9,  0.10, 0.11],
    ///     [0.12, 0.13, 0.14],
    /// ];
    /// let blocks: Vec<SafView<1>> = saf.par_iter_blocks(2).collect();
    /// assert_eq!(blocks.len(), 3);
    /// assert_eq!(
    ///     blocks[0],
    ///     saf1d![[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]].view()
    /// );
    /// assert_eq!(
    ///     blocks[1],
    ///     saf1d![[0.6, 0.7, 0.8], [0.9, 0.10, 0.11]].view()
    /// );
    /// assert_eq!(blocks[2], saf1d![[0.12,  0.13,  0.14]].view());
    pub fn par_iter_blocks(&self, block_size: usize) -> ParBlockIter<N> {
        ParBlockIter::new(self.view(), block_size)
    }

    /// Returns a parallel iterator over the sites in the SAF.
    ///
    /// This is the parallel version of [`Saf::iter_sites`].
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::saf1d;
    /// use rayon::iter::ParallelIterator;
    /// let saf = saf1d![
    ///     [1.,  1.,  1.],
    ///     [1.,  1.,  1.],
    ///     [1.,  1.,  1.],
    /// ];
    /// saf.par_iter_sites().all(|site| site.as_slice() == &[1., 1., 1.]);
    pub fn par_iter_sites(&self) -> ParSiteIter<N> {
        ParSiteIter::new(self.view())
    }

    /// Creates a new SAF by reading intersecting sites among SAF readers.
    ///
    /// SAF files contain values in log-space. The returned values will be exponentiated
    /// to get out of log-space.
    ///
    /// # Panics
    ///
    /// Panics if `N == 0`.
    pub fn read<R>(readers: [saf::BgzfReader<R>; N]) -> io::Result<Self>
    where
        R: io::BufRead + io::Seek,
    {
        assert!(N > 0);

        let max_sites = readers
            .iter()
            .map(|reader| reader.index().total_sites())
            .min()
            .unwrap();

        let shape = readers
            .each_ref()
            .map(|reader| reader.index().alleles() + 1);

        // The number of intersecting sites is as most the smallest number of sites,
        // so we preallocate this number and free excess capacity at the end.
        let capacity = shape.iter().map(|shape| shape * max_sites).sum();
        let mut values = Vec::with_capacity(capacity);

        let mut intersect = saf::reader::Intersect::new(Vec::from(readers));
        let mut bufs = intersect.create_record_bufs();

        while intersect.read_records(&mut bufs)?.is_not_done() {
            for buf in bufs.iter() {
                values.extend_from_slice(buf.values());
            }
        }
        // The allocated capacity is an overestimate unless all sites in smallest file intersected.
        values.shrink_to_fit();

        // Representation in SAF file is in log-space.
        values.iter_mut().for_each(|x| *x = x.exp());

        Ok(Self::new_unchecked(values, shape))
    }

    /// Creates a new SAF by reading intersecting sites among SAF member file paths.
    ///
    /// This is a convenience wrapper around [`Saf::read`], see also its documentation.
    ///
    /// # Panics
    ///
    /// Panics if `N == 0`.
    ///
    /// # Examples
    ///
    /// See [module docs](index.html#reading-from-saf-files).
    pub fn read_from_paths<P>(paths: [P; N]) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        // TODO: Use array::try_map when stable here
        let readers: [_; N] = paths
            .iter()
            .map(saf::Reader::from_bgzf_member_path)
            .collect::<io::Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| ()) // Reader is not debug, so this is necessary to unwrap
            .unwrap();

        Self::read(readers)
    }

    /// Shuffles the SAF sitewise according to a random permutation.
    pub fn shuffle<R>(&mut self, rng: &mut R)
    where
        R: Rng,
    {
        let width = self.width();

        // Modified from rand::seq::SliceRandom::shuffle
        for i in (1..self.sites()).rev() {
            let j = rng.gen_range(0..i + 1);

            self.swap_sites(i, j, width);
        }
    }

    /// Swap sites `i` and `j` in SAF.
    ///
    /// `width` is passed in to avoid recalculating for each swap.
    ///
    /// # Panics
    ///
    /// If `i` or `j` are greater than the number of sites according to `width`.
    fn swap_sites(&mut self, mut i: usize, mut j: usize, width: usize) {
        debug_assert_eq!(width, self.width());

        match i.cmp(&j) {
            Ordering::Less => (i, j) = (j, i),
            Ordering::Equal => {
                if i >= self.sites() || j >= self.sites() {
                    panic!("index out of bounds for swapping sites")
                } else {
                    return;
                }
            }
            Ordering::Greater => (),
        }

        let (hd, tl) = self.as_mut_slice().split_at_mut(i * width);

        let left = &mut hd[j * width..][..width];
        let right = &mut tl[..width];

        left.swap_with_slice(right)
    }

    /// Returns a view of the entire SAF.
    pub fn view(&self) -> SafView<N> {
        SafView {
            values: self.values.as_slice(),
            shape: self.shape,
        }
    }

    impl_shared_saf_methods! {}
}

impl<'a, const N: usize> Lifetime<'a> for Saf<N> {
    type Item = SafView<'a, N>;
}

impl<const N: usize> AsSafView<N> for Saf<N> {
    #[inline]
    fn as_saf_view(&self) -> <Self as Lifetime<'_>>::Item {
        self.view()
    }
}

impl<'a, 'b, const N: usize> Lifetime<'a> for &'b Saf<N> {
    type Item = SafView<'a, N>;
}

impl<'a, const N: usize> AsSafView<N> for &'a Saf<N> {
    #[inline]
    fn as_saf_view(&self) -> <Self as Lifetime<'_>>::Item {
        self.view()
    }
}

/// A view of a joint SAF likelihood matrix for `N` populations.
///
/// This may or may not be the entire matrix, but it always represents a contiguous block of sites.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SafView<'a, const N: usize> {
    values: &'a [f32],
    shape: [usize; N],
}

impl<'a, const N: usize> SafView<'a, N> {
    /// Returns an iterator over blocks of sites in the SAF.
    ///
    /// If the number of sites in the SAF is not evenly divided by `block_size`,
    /// the last block will be smaller than the others.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::saf1d;
    /// let saf = saf1d![
    ///     [0.0,  0.1,  0.2],
    ///     [0.3,  0.4,  0.5],
    ///     [0.6,  0.7,  0.8],
    ///     [0.9,  0.10, 0.11],
    ///     [0.12, 0.13, 0.14],
    /// ];
    /// let mut iter = saf.view().iter_blocks(2);
    /// assert_eq!(
    ///     iter.next().unwrap(),
    ///     saf1d![[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]].view()
    /// );
    /// assert_eq!(
    ///     iter.next().unwrap(),
    ///     saf1d![[0.6, 0.7, 0.8], [0.9, 0.10, 0.11]].view()
    /// );
    /// assert_eq!(iter.next().unwrap(), saf1d![[0.12, 0.13, 0.14]].view());
    /// assert!(iter.next().is_none());
    /// ```
    pub fn iter_blocks(&self, block_size: usize) -> BlockIter<'a, N> {
        BlockIter::new(*self, block_size)
    }

    /// Returns an iterator over the sites in the SAF.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::saf1d;
    /// let saf = saf1d![
    ///     [0.0,  0.1,  0.2],
    ///     [0.3,  0.4,  0.5],
    ///     [0.6,  0.7,  0.8],
    /// ];
    /// let mut iter = saf.view().iter_sites();
    /// assert_eq!(iter.next().unwrap().as_slice(), [0.0,  0.1,  0.2]);
    /// assert_eq!(iter.next().unwrap().as_slice(), [0.3,  0.4,  0.5]);
    /// assert_eq!(iter.next().unwrap().as_slice(), [0.6,  0.7,  0.8]);
    /// assert!(iter.next().is_none());
    /// ```
    pub fn iter_sites(&self) -> SiteIter<'a, N> {
        SiteIter::new(*self)
    }

    /// Returns a new SAF view.
    ///
    /// The number of provided values must be a multiple of the sum of shapes.
    /// See the [`Saf`] documentation for details on the storage order.
    ///
    /// To create an owned SAF matrix, see [`Saf::new`] for the equivalent method.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::{SafView, saf2d};
    /// let slice = &[0.0, 0.1, 0.2, 1.0, 1.1, 0.3, 0.4, 0.5, 1.2, 1.3];
    /// let shape = [3, 2];
    /// assert_eq!(
    ///     SafView::new(slice, shape).unwrap(),
    ///     saf2d![
    ///         [0.0,  0.1,  0.2 ; 1.0, 1.1],
    ///         [0.3,  0.4,  0.5 ; 1.2, 1.3],
    ///     ].view(),
    /// );
    /// ```
    /// A [`ShapeError`] is thrown if the shape does not fit the number of values:
    ///
    /// ```
    /// use winsfs::SafView;
    /// let slice = &[0.0, 0.1, 0.2, 1.0, 1.1, 0.3, 0.4, 0.5, 1.2, 1.3];
    /// let wrong_shape = [4, 2];
    /// assert!(SafView::new(slice, wrong_shape).is_err());
    /// ```
    pub fn new(values: &'a [f32], shape: [usize; N]) -> Result<Self, ShapeError<N>> {
        let len = values.len();
        let width: usize = shape.iter().sum();

        if len % width == 0 {
            Ok(Self::new_unchecked(values, shape))
        } else {
            Err(ShapeError { len, shape })
        }
    }

    /// Returns a new SAF view without checking that the shape fits the number of values.
    pub(crate) fn new_unchecked(values: &'a [f32], shape: [usize; N]) -> Self {
        Self { values, shape }
    }

    /// Returns a parallel iterator over the blocks in the SAF.
    ///
    /// This is the parallel version of [`SafView::iter_blocks`].
    /// If the number of sites in the SAF is not evenly divided by `block_size`,
    /// the last block will be smaller than the others.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::{saf1d, SafView};
    /// use rayon::iter::ParallelIterator;
    /// let saf = saf1d![
    ///     [0.0,  0.1,  0.2],
    ///     [0.3,  0.4,  0.5],
    ///     [0.6,  0.7,  0.8],
    ///     [0.9,  0.10, 0.11],
    ///     [0.12, 0.13, 0.14],
    /// ];
    /// let view = saf.view();
    /// let blocks: Vec<SafView<1>> = view.par_iter_blocks(2).collect();
    /// assert_eq!(blocks.len(), 3);
    /// assert_eq!(
    ///     blocks[0],
    ///     saf1d![[0.0, 0.1, 0.2], [0.3,  0.4,  0.5]].view()
    /// );
    /// assert_eq!(
    ///     blocks[1],
    ///     saf1d![[0.6, 0.7, 0.8], [0.9,  0.10,  0.11]].view()
    /// );
    /// assert_eq!(blocks[2], saf1d![[0.12, 0.13, 0.14]].view());
    pub fn par_iter_blocks(&self, block_size: usize) -> ParBlockIter<N> {
        ParBlockIter::new(*self, block_size)
    }

    /// Returns a parallel iterator over the sites in the SAF.
    ///
    /// This is the parallel version of [`SafView::iter_sites`].
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::saf1d;
    /// use rayon::iter::ParallelIterator;
    /// let saf = saf1d![
    ///     [1.,  1.,  1.],
    ///     [1.,  1.,  1.],
    ///     [1.,  1.,  1.],
    /// ];
    /// saf.view().par_iter_sites().all(|site| site.as_slice() == &[1., 1., 1.]);
    pub fn par_iter_sites(&self) -> ParSiteIter<N> {
        ParSiteIter::new(*self)
    }

    impl_shared_saf_methods! {}
}

impl<'a, 'b, const N: usize> Lifetime<'a> for SafView<'b, N> {
    type Item = SafView<'a, N>;
}

impl<'a, const N: usize> AsSafView<N> for SafView<'a, N> {
    #[inline]
    fn as_saf_view(&self) -> <Self as Lifetime<'_>>::Item {
        *self
    }
}

/// An error associated with SAF or SAF site construction using invalid shape.
#[derive(Clone, Debug)]
pub struct ShapeError<const N: usize> {
    shape: [usize; N],
    len: usize,
}

impl<const N: usize> fmt::Display for ShapeError<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cannot construct shape {} from {} values",
            self.shape.map(|x| x.to_string()).join("/"),
            self.len,
        )
    }
}

impl<const N: usize> Error for ShapeError<N> {}

#[cfg(test)]
mod tests {
    #[test]
    fn test_swap_1d() {
        let mut saf = saf1d![
            [0., 0., 0.],
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
            [4., 4., 4.],
            [5., 5., 5.],
        ];

        let width = saf.width();
        assert_eq!(width, 3);

        saf.swap_sites(3, 3, width);
        assert_eq!(saf.get_site(3).as_slice(), &[3., 3., 3.]);

        saf.swap_sites(0, 1, width);
        assert_eq!(saf.get_site(0).as_slice(), &[1., 1., 1.]);
        assert_eq!(saf.get_site(1).as_slice(), &[0., 0., 0.]);

        saf.swap_sites(5, 0, width);
        assert_eq!(saf.get_site(0).as_slice(), &[5., 5., 5.]);
        assert_eq!(saf.get_site(5).as_slice(), &[1., 1., 1.]);
    }

    #[test]
    fn test_swap_2d() {
        #[rustfmt::skip]
        let mut saf = saf2d![
            [0., 0., 0.; 10., 10.],
            [1., 1., 1.; 11., 11.],
            [2., 2., 2.; 12., 12.],
            [3., 3., 3.; 13., 13.],
            [4., 4., 4.; 14., 14.],
            [5., 5., 5.; 15., 15.],
        ];

        let width = saf.width();
        assert_eq!(width, 5);

        saf.swap_sites(0, 5, width);
        assert_eq!(saf.get_site(0).as_slice(), &[5., 5., 5., 15., 15.,]);
        saf.swap_sites(5, 0, width);
        assert_eq!(saf.get_site(0).as_slice(), &[0., 0., 0., 10., 10.,]);
    }

    #[test]
    #[should_panic]
    fn test_swap_panics_out_of_bounds() {
        let mut saf = saf1d![
            [0., 0., 0.],
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
            [4., 4., 4.],
            [5., 5., 5.],
        ];

        saf.swap_sites(6, 5, saf.width());
    }
}
