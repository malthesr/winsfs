//! Site allelele frequency ("SAF") likelihoods.

use std::{cmp::Ordering, error::Error, fmt, io, path::Path};

use angsd_io::saf;

use rand::Rng;

use rayon::slice::ParallelSlice;

mod joint_saf;
pub use joint_saf::{JointSaf, JointSafView, JointShapeError};

mod blocks;
pub use blocks::Blocks;

mod traits;
pub use traits::{BlockIterator, IntoArray, ParSiteIterator};

/// Creates a SAF containing the arguments.
///
/// This is mainly intended for readability in doc-tests, but may also be useful elsewhere.
///
/// # Examples
///
/// ```
/// use winsfs::saf;
/// let saf = saf![
///     [0.1,  0.2,  0.3],
///     [0.4,  0.5,  0.6],
///     [0.7,  0.8,  0.9],
///     [0.10, 0.11, 0.12],
///     [0.13, 0.14, 0.15],
/// ];
/// assert_eq!(saf.sites(), 5);
/// assert_eq!(saf.shape(), 3);
/// ```
#[macro_export]
macro_rules! saf {
    ($([$($x:literal),+ $(,)?]),+ $(,)?) => {{
        let (cols, vec) = $crate::matrix!($([$($x),+]),+);
        $crate::saf::Saf::new(vec, cols[0]).unwrap()
    }};
}

macro_rules! impl_shared_saf_methods {
    () => {
        /// Returns the values of the SAF as a flat, row-major slice.
        pub fn as_slice(&self) -> &[f32] {
            &self.values
        }

        /// Returns the values for a single site in the SAF.
        pub fn site(&self, index: usize) -> &[f32] {
            &self.values[index * self.shape..][..self.shape]
        }

        /// Returns the number of sites in the SAF.
        ///
        /// This corresponds to the number of rows in the matrix.
        pub fn sites(&self) -> usize {
            self.values.len() / self.shape
        }

        /// Returns the shape of the SAF.
        ///
        /// This corresponds to the number of columns in the matrix.
        pub fn shape(&self) -> usize {
            self.shape
        }
    };
}

/// A matrix of site allele frequency ("SAF") likelihoods.
#[derive(Clone, Debug, PartialEq)]
pub struct Saf {
    values: Vec<f32>,
    shape: usize,
}

impl Saf {
    /// Returns a mutable reference to the values of the SAF as a flat, row-major slice.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.values
    }

    pub(super) fn from_log(mut values: Vec<f32>, shape: usize) -> Result<Self, ShapeError> {
        values.iter_mut().for_each(|x| *x = x.exp());

        Self::new(values, shape)
    }

    /// Returns an iterator over the sites in the the SAF.
    pub fn iter_sites(&self) -> std::slice::Chunks<f32> {
        self.values.chunks(self.shape)
    }

    /// Creates a new SAF.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::Saf;
    /// let values: Vec<f32> = (0..12).map(|x| x as f32).collect();
    /// let saf = Saf::new(values, 3).expect("shape does not evenly divide number of values");
    /// assert_eq!(saf.shape(), 3);
    /// assert_eq!(saf.sites(), 4);
    /// ```
    pub fn new(values: Vec<f32>, shape: usize) -> Result<Self, ShapeError> {
        let len = values.len();

        if len % shape == 0 {
            Ok(Self::new_unchecked(values, shape))
        } else {
            Err(ShapeError { len, shape })
        }
    }

    fn new_unchecked(values: Vec<f32>, shape: usize) -> Self {
        Self { values, shape }
    }

    /// Returns a parallel iterator over the sites in the SAF.
    pub fn par_iter_sites(&self) -> rayon::slice::Chunks<f32> {
        self.values.par_chunks(self.shape)
    }

    /// Creates a SAF by reading from a reader.
    ///
    /// It is assumed that the values in the reader are in log-space,
    /// so all values will be exponentiated.
    pub fn read<R>(mut reader: saf::BgzfReader<R>) -> io::Result<Self>
    where
        R: io::BufRead,
    {
        let total_sites: usize = reader.index().total_sites();
        let shape = reader.index().alleles() + 1;

        let capacity = shape * total_sites;
        let mut values = vec![0.0; capacity];

        reader
            .value_reader_mut()
            .read_values(values.as_mut_slice())?;

        Self::from_log(values, shape).map_err(io::Error::from)
    }

    /// Creates a SAF by reading from a path.
    ///
    /// The path can be any SAF file member path. This is simply a convenience wrapper for
    /// opening a [`angsd_io::saf::BgzfReader`] and using the [`Saf::read`] constructor.
    /// See its documentation for details.
    pub fn read_from_path<P>(path: P) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        let reader = saf::reader::BgzfReader::from_bgzf_member_path(path)?;

        Self::read(reader)
    }

    /// Returns a mutable reference to the values for a single site in the SAF.
    pub fn site_mut(&mut self, index: usize) -> &mut [f32] {
        &mut self.values[index * self.shape..][..self.shape]
    }

    /// Shuffles the SAF sitewise according to a random permutation.
    pub fn shuffle<R>(&mut self, rng: &mut R)
    where
        R: Rng,
    {
        // Modified from rand::seq::SliceRandom::shuffle
        for i in (1..self.sites()).rev() {
            let j = rng.gen_range(0..i + 1);

            self.swap_sites(i, j);
        }
    }

    /// Swap two sites in SAF.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::saf;
    /// let mut saf = saf![
    ///    [0., 0., 0.],
    ///    [1., 1., 1.],
    ///    [2., 2., 2.],
    ///    [3., 3., 3.],
    /// ];
    /// assert_eq!(saf.site(0), &[0., 0., 0.]);
    /// assert_eq!(saf.site(2), &[2., 2., 2.]);
    /// saf.swap_sites(0, 2);
    /// assert_eq!(saf.site(0), &[2., 2., 2.]);
    /// assert_eq!(saf.site(2), &[0., 0., 0.]);
    /// ```
    pub fn swap_sites(&mut self, mut i: usize, mut j: usize) {
        match i.cmp(&j) {
            Ordering::Less => (i, j) = (j, i),
            Ordering::Equal => return,
            Ordering::Greater => (),
        }

        let shape = self.shape;
        let (hd, tl) = self.as_mut_slice().split_at_mut(i * shape);

        let left = &mut hd[j * shape..][..shape];
        let right = &mut tl[..shape];

        left.swap_with_slice(right)
    }

    /// Returns a SAF view of the entire matrix.
    pub fn view(&self) -> SafView<'_> {
        SafView::new_unchecked(&self.values, self.shape)
    }

    impl_shared_saf_methods!();
}

/// A view of a matrix of site allele frequency ("SAF") likelihoods.
///
/// The view may correspond to some or all of the sites in the original matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SafView<'a> {
    values: &'a [f32],
    shape: usize,
}

impl<'a> SafView<'a> {
    /// Returns an iterator over the sites in the the SAF view.
    pub fn iter_sites(&self) -> std::slice::Chunks<'a, f32> {
        self.values.chunks(self.shape)
    }

    /// Creates a new SAF view from a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::SafView;
    /// let values: Vec<f32> = (0..12).map(|x| x as f32).collect();
    /// let saf = SafView::new(&values[..], 3)
    ///     .expect("shape does not evenly divide number of values");
    /// assert_eq!(saf.shape(), 3);
    /// assert_eq!(saf.sites(), 4);
    /// ```
    pub fn new(values: &'a [f32], shape: usize) -> Result<Self, ShapeError> {
        let len = values.len();

        if len % shape == 0 {
            Ok(Self::new_unchecked(values, shape))
        } else {
            Err(ShapeError { len, shape })
        }
    }

    fn new_unchecked(values: &'a [f32], shape: usize) -> Self {
        Self { values, shape }
    }

    /// Returns a parallel iterator over the sites in the SAF.
    pub fn par_iter_sites(&self) -> rayon::slice::Chunks<'a, f32> {
        self.values.par_chunks(self.shape)
    }

    /// Returns two SAF views containing sites before and after a specified site.
    ///
    /// The first view will include the site corresponding to the specified index.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::saf;
    /// let saf = saf![
    ///     [0., 0., 0.],
    ///     [1., 1., 1.],
    ///     [2., 2., 2.],
    ///     [3., 3., 3.],
    ///     [4., 4., 4.],
    ///     [5., 5., 5.],
    /// ];
    /// let (hd, tl) = saf.view().split_at_site(4);
    /// assert_eq!(hd.as_slice(), &[0., 0., 0., 1., 1., 1., 2., 2., 2., 3., 3., 3.]);
    /// assert_eq!(tl.as_slice(), &[4., 4., 4., 5., 5., 5.]);
    /// ```
    pub fn split_at_site(&self, site: usize) -> (Self, Self) {
        let (hd, tl) = self.values.split_at(site * self.shape);

        (
            Self::new_unchecked(hd, self.shape),
            Self::new_unchecked(tl, self.shape),
        )
    }

    impl_shared_saf_methods!();
}

/// An error associated with SAF construction using invalid shape.
#[derive(Clone, Debug)]
pub struct ShapeError {
    shape: usize,
    len: usize,
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cannot construct a SAF of shape {} from {} values",
            self.shape, self.len,
        )
    }
}

impl Error for ShapeError {}

impl From<ShapeError> for io::Error {
    fn from(e: ShapeError) -> Self {
        io::Error::new(io::ErrorKind::InvalidData, e)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_swap_chunks() {
        let mut saf = saf![
            [0., 0., 0.],
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
            [4., 4., 4.],
            [5., 5., 5.],
        ];

        saf.swap_sites(3, 3);
        assert_eq!(saf.site(3), &[3., 3., 3.]);

        saf.swap_sites(0, 1);
        assert_eq!(saf.site(0), &[1., 1., 1.]);
        assert_eq!(saf.site(1), &[0., 0., 0.]);

        saf.swap_sites(5, 0);
        assert_eq!(saf.site(0), &[5., 5., 5.]);
        assert_eq!(saf.site(5), &[1., 1., 1.]);
    }
}
