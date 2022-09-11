//! Multi-dimensional site frequency spectra ("SFS").
//!
//! The central type is the [`SfsBase`] struct, which represents an SFS with a dimensionality
//! that may or may not be known at compile time, and which may or may not be normalised to
//! probability scale. Type aliases [`Sfs`], [`USfs`], [`DynSfs`], and [`DynUSfs`] are exposed
//! for convenience.

use std::{
    cmp::Ordering,
    error::Error,
    fmt::{self, Write as _},
    marker::PhantomData,
    ops::{Add, AddAssign, Index, IndexMut, Sub, SubAssign},
    slice,
};

use crate::ArrayExt;

pub mod generics;
use generics::{ConstShape, DynShape, Norm, Normalisation, Shape, Unnorm};

pub mod io;

pub mod iter;
use iter::Indices;

pub mod multi;
pub use multi::Multi;

mod em;

const NORMALISATION_TOLERANCE: f64 = 10. * f64::EPSILON;

/// Creates an unnormalised 1D SFS.
///
/// This is mainly intended for readability in doc-tests, but may also be useful elsewhere.
///
/// # Examples
///
/// Create SFS by repeating an element:
///
/// ```
/// use winsfs_core::sfs1d;
/// let sfs = sfs1d![0.1; 10];
/// assert!(sfs.iter().all(|&x| x == 0.1));
/// ```
///
/// Create SFS from a list of elements:
///
/// ```
/// use winsfs_core::sfs1d;
/// let sfs = sfs1d![0.1, 0.2, 0.3];
/// assert_eq!(sfs[[0]], 0.1);
/// assert_eq!(sfs[[1]], 0.2);
/// assert_eq!(sfs[[2]], 0.3);
/// ```
#[macro_export]
macro_rules! sfs1d {
    ($elem:expr; $n:expr) => {
        $crate::sfs::USfs::from_elem($elem, [$n])
    };
    ($($x:expr),+ $(,)?) => {
        $crate::sfs::USfs::from_vec(vec![$($x),+])
    };
}

/// Creates an unnormalised 2D SFS.
///
/// This is mainly intended for readability in doc-tests, but may also be useful elsewhere.
///
/// # Examples
///
/// ```
/// use winsfs_core::sfs2d;
/// let sfs = sfs2d![
///     [0.1, 0.2, 0.3],
///     [0.4, 0.5, 0.6],
///     [0.7, 0.8, 0.9],
/// ];
/// assert_eq!(sfs[[0, 0]], 0.1);
/// assert_eq!(sfs[[1, 0]], 0.4);
/// assert_eq!(sfs[[2, 0]], 0.7);
/// ```
#[macro_export]
macro_rules! sfs2d {
    ($([$($x:literal),+ $(,)?]),+ $(,)?) => {{
        let (cols, vec) = $crate::matrix!($([$($x),+]),+);
        let shape = [cols.len(), cols[0]];
        $crate::sfs::SfsBase::from_vec_shape(vec, shape).unwrap()
    }};
}

/// An multi-dimensional site frequency spectrum ("SFS").
///
/// Elements are stored in row-major order: the last index varies the fastest.
///
/// The number of dimensions of the SFS may either be known at compile-time or run-time,
/// and this is governed by the [`Shape`] trait. Moreover, the SFS may or may not be normalised
/// to probability scale, and this is controlled by the [`Normalisation`] trait.
/// See also the [`Sfs`], [`USfs`], [`DynSfs`], and [`DynUSfs`] type aliases.
#[derive(Clone, Debug, PartialEq)]
// TODO: Replace normalisation with const enum once these are permitted in const generics,
// see github.com/rust-lang/rust/issues/95174
pub struct SfsBase<S: Shape, N: Normalisation> {
    values: Vec<f64>,
    pub(crate) shape: S,
    pub(crate) strides: S,
    norm: PhantomData<N>,
}

/// A normalised SFS with shape known at compile-time.
pub type Sfs<const D: usize> = SfsBase<ConstShape<D>, Norm>;

/// An unnormalised SFS with shape known at compile-time.
pub type USfs<const D: usize> = SfsBase<ConstShape<D>, Unnorm>;

/// A normalised SFS with shape known at run-time.
pub type DynSfs = SfsBase<DynShape, Norm>;

/// An unnormalised SFS with shape known at run-time.
pub type DynUSfs = SfsBase<DynShape, Unnorm>;

impl<S: Shape, N: Normalisation> SfsBase<S, N> {
    /// Returns the values of the SFS as a flat, row-major slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs2d;
    /// let sfs = sfs2d![
    ///     [0., 1., 2.],
    ///     [3., 4., 5.],
    /// ];
    /// assert_eq!(sfs.as_slice(), [0., 1., 2., 3., 4., 5.]);
    /// ```
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Returns a folded version of the SFS.
    ///
    /// Folding is useful when the spectrum has not been properly polarised, so that there is
    /// no meaningful distinction between having 0 and 2N (in the diploid case) variants at a site.
    /// The folding operation collapses these indistinguishable bins by adding the value from the
    /// lower part of the spectrum onto the upper, and setting the lower value to zero.
    ///
    /// Note that we adopt the convention that on the "diagonal" of the SFS, where there is less of
    /// a convention on what is the correct way of folding, the arithmetic mean of the candidates is
    /// used. The examples below illustrate this.
    ///
    /// # Examples
    ///
    /// Folding in 1D:
    ///
    /// ```
    /// use winsfs_core::sfs1d;
    /// let sfs = sfs1d![5., 2., 3., 10., 1.];
    /// assert_eq!(sfs.fold(), sfs1d![6., 12., 3., 0., 0.]);
    /// ```
    ///
    /// Folding in 2D (square input):
    ///
    /// ```
    /// use winsfs_core::sfs2d;
    /// let sfs = sfs2d![
    ///     [4., 2., 10.],
    ///     [0., 3., 4.],
    ///     [7., 2., 1.],
    /// ];
    /// let expected = sfs2d![
    ///     [5., 4., 8.5],
    ///     [4., 3., 0.],
    ///     [8.5, 0., 0.],
    /// ];
    /// assert_eq!(sfs.fold(), expected);
    /// ```
    ///
    /// Folding in 2D (non-square input):
    ///
    /// ```
    /// use winsfs_core::sfs2d;
    /// let sfs = sfs2d![
    ///     [4., 2., 10.],
    ///     [0., 3., 4.],
    /// ];
    /// let expected = sfs2d![
    ///     [8., 5., 0.],
    ///     [10., 0., 0.],
    /// ];
    /// assert_eq!(sfs.fold(), expected);
    /// ```    
    pub fn fold(&self) -> Self {
        let n = self.values.len();
        let total_count = self.shape.iter().sum::<usize>() - self.shape.len();

        // In general, this point divides the folding line. Since we are folding onto the "upper"
        // part of the array, we want to fold anything "below" it onto something "above" it.
        let mid_count = total_count / 2;

        // The spectrum may or may not have a "diagonal", i.e. a hyperplane that falls exactly on
        // the midpoint. If such a diagonal exists, we need to handle it as a special case when
        // folding below.
        //
        // For example, in 1D a spectrum with five elements has a "diagonal", marked X:
        // [-, -, X, -, -]
        // Whereas on with four elements would not.
        //
        // In two dimensions, e.g. three-by-three elements has a diagonal:
        // [-, -, X]
        // [-, X, -]
        // [X, -, -]
        // whereas two-by-three would not. On the other hand, two-by-four has a diagonal:
        // [-, -, X, -]
        // [-, X, -, -]
        //
        // Note that even-ploidy data should always have a diagonal, whereas odd-ploidy data
        // may or may not.
        let has_diagonal = total_count % 2 == 0;

        // Note that we cannot use the algorithm below in-place, since the reverse iterator
        // may reach elements that have already been folded, which causes bugs. Hence we fold
        // into a zero-initialised copy.
        let mut folded = Self::new_unchecked(vec![0.0; n], self.shape.clone());

        // We iterate over indices rather than values since we have to mutate on the array
        // while looking at it from both directions.
        (0..n).zip((0..n).rev()).for_each(|(i, rev_i)| {
            let count = compute_index_sum_unchecked(i, n, self.shape.as_ref());

            match (count.cmp(&mid_count), has_diagonal) {
                (Ordering::Less, _) | (Ordering::Equal, false) => {
                    // We are in the upper part of the spectrum that should be folded onto.
                    folded.values[i] = self.values[i] + self.values[rev_i];
                }
                (Ordering::Equal, true) => {
                    // We are on a diagonal, which must be handled as a special case:
                    // there are apparently different opinions on what the most correct
                    // thing to do is. This adopts the same strategy as e.g. in dadi.
                    folded.values[i] = 0.5 * self.values[i] + 0.5 * self.values[rev_i];
                }
                (Ordering::Greater, _) => (),
            }
        });

        folded
    }

    /// Returns a string containing a flat, row-major represention of the SFS.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs1d;
    /// let sfs = sfs1d![0.0, 0.1, 0.2];
    /// assert_eq!(sfs.format_flat(" ", 1), "0.0 0.1 0.2");
    /// ```
    ///
    /// ```
    /// use winsfs_core::sfs2d;
    /// let  sfs = sfs2d![[0.01, 0.12], [0.23, 0.34]];
    /// assert_eq!(sfs.format_flat(",", 2), "0.01,0.12,0.23,0.34");
    /// ```
    pub fn format_flat(&self, sep: &str, precision: usize) -> String {
        if let Some(first) = self.values.first() {
            let cap = self.values.len() * (precision + 3);
            let mut init = String::with_capacity(cap);
            write!(init, "{first:.precision$}").unwrap();
            // init.push_str(&format!("{:.precision$}", first));

            self.iter().skip(1).fold(init, |mut s, x| {
                s.push_str(sep);
                write!(s, "{x:.precision$}").unwrap();
                s
            })
        } else {
            String::new()
        }
    }

    /// Returns a value at an index in the SFS.
    ///
    /// If the index is out of bounds, `None` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs1d;
    /// let sfs = sfs1d![0.0, 0.1, 0.2];
    /// assert_eq!(sfs.get(&[0]), Some(&0.0));
    /// assert_eq!(sfs.get(&[1]), Some(&0.1));
    /// assert_eq!(sfs.get(&[2]), Some(&0.2));
    /// assert_eq!(sfs.get(&[3]), None);
    /// ```
    ///
    /// ```
    /// use winsfs_core::sfs2d;
    /// let sfs = sfs2d![[0.0, 0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]];
    /// assert_eq!(sfs.get(&[0, 0]), Some(&0.0));
    /// assert_eq!(sfs.get(&[1, 2]), Some(&0.5));
    /// assert_eq!(sfs.get(&[3, 0]), None);
    /// ```
    #[inline]
    pub fn get(&self, index: &S) -> Option<&f64> {
        self.values.get(compute_flat(index, &self.shape)?)
    }

    /// Returns a normalised SFS, consuming `self`.
    ///
    /// This works purely on the type level, and does not modify the actual values in the SFS.
    /// If the SFS is not already normalised, an error is returned. To modify the SFS to become
    /// normalised, see [`Sfs::normalise`].
    ///
    /// # Examples
    ///
    /// An unnormalised SFS with values summing to one can be turned into a normalised SFS:
    ///
    /// ```
    /// use winsfs_core::{sfs1d, sfs::{Sfs, USfs}};
    /// let sfs: USfs<1> = sfs1d![0.2; 5];
    /// let sfs: Sfs<1> = sfs.into_normalised().unwrap();
    /// ```
    ///
    /// Otherwise, an unnormalised SFS cannot be normalised SFS using this method:
    ///
    /// ```
    /// use winsfs_core::{sfs1d, sfs::USfs};
    /// let sfs: USfs<1> = sfs1d![2.; 5];
    /// assert!(sfs.into_normalised().is_err());
    /// ```
    ///
    /// Use [`Sfs::normalise`] instead.
    #[inline]
    pub fn into_normalised(self) -> Result<SfsBase<S, Norm>, NormError> {
        let sum = self.sum();

        if (sum - 1.).abs() <= NORMALISATION_TOLERANCE {
            Ok(self.into_normalised_unchecked())
        } else {
            Err(NormError { sum })
        }
    }

    #[inline]
    fn into_normalised_unchecked(self) -> SfsBase<S, Norm> {
        SfsBase {
            values: self.values,
            shape: self.shape,
            strides: self.strides,
            norm: PhantomData,
        }
    }

    /// Returns an unnormalised SFS, consuming `self`.
    ///
    /// This works purely on the type level, and does not modify the actual values in the SFS.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs::{Sfs, USfs};
    /// let sfs: Sfs<1> = Sfs::uniform([7]);
    /// let sfs: USfs<1> = sfs.into_unnormalised();
    /// ```
    #[inline]
    pub fn into_unnormalised(self) -> SfsBase<S, Unnorm> {
        SfsBase {
            values: self.values,
            shape: self.shape,
            strides: self.strides,
            norm: PhantomData,
        }
    }

    /// Returns an iterator over the elements in the SFS in row-major order.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs2d;
    /// let sfs = sfs2d![
    ///     [0., 1., 2.],
    ///     [3., 4., 5.],
    ///     [6., 7., 8.],
    /// ];
    /// let expected = (0..9).map(|x| x as f64);
    /// assert!(sfs.iter().zip(expected).all(|(&x, y)| x == y));
    /// ```
    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, f64> {
        self.values.iter()
    }

    /// Creates a new SFS.
    #[inline]
    fn new_unchecked(values: Vec<f64>, shape: S) -> Self {
        let strides = shape.strides();

        Self {
            values,
            shape,
            strides,
            norm: PhantomData,
        }
    }

    /// Returns an unnormalised SFS scaled by some constant, consuming `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs1d;
    /// assert_eq!(
    ///     sfs1d![0., 1.,  2.,  3.,  4.].scale(10.),
    ///     sfs1d![0., 10., 20., 30., 40.],
    /// );
    /// ```
    #[inline]
    #[must_use = "returns scaled SFS, doesn't modify in-place"]
    pub fn scale(mut self, scale: f64) -> SfsBase<S, Unnorm> {
        self.values.iter_mut().for_each(|x| *x *= scale);

        self.into_unnormalised()
    }

    /// Returns the SFS shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs2d;
    /// let sfs = sfs2d![
    ///     [0., 1., 2.],
    ///     [3., 4., 5.],
    /// ];
    /// assert_eq!(sfs.shape(), &[2, 3]);
    /// ```
    pub fn shape(&self) -> &S {
        &self.shape
    }

    /// Returns the sum of values in the SFS.
    #[inline]
    fn sum(&self) -> f64 {
        self.iter().sum()
    }
}

impl<const D: usize, N: Normalisation> SfsBase<ConstShape<D>, N> {
    /// Returns an iterator over the sample frequencies of the SFS in row-major order.
    ///
    /// Note that this is *not* the contents of SFS, but the frequencies corresponding
    /// to the indices. See [`Sfs::iter`] for an iterator over the SFS values themselves.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs::Sfs;
    /// let sfs = Sfs::uniform([2, 3]);
    /// let mut iter = sfs.frequencies();
    /// assert_eq!(iter.next(), Some([0., 0.]));
    /// assert_eq!(iter.next(), Some([0., 0.5]));
    /// assert_eq!(iter.next(), Some([0., 1.]));
    /// assert_eq!(iter.next(), Some([1., 0.]));
    /// assert_eq!(iter.next(), Some([1., 0.5]));
    /// assert_eq!(iter.next(), Some([1., 1.]));
    /// assert!(iter.next().is_none());
    /// ```
    pub fn frequencies(&self) -> impl Iterator<Item = [f64; D]> {
        let n_arr = self.shape.map(|n| n - 1);
        self.indices()
            .map(move |idx_arr| idx_arr.array_zip(n_arr).map(|(i, n)| i as f64 / n as f64))
    }

    /// Returns an iterator over the indices in the SFS in row-major order.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs::Sfs;
    /// let sfs = Sfs::uniform([2, 3]);
    /// let mut iter = sfs.indices();
    /// assert_eq!(iter.next(), Some([0, 0]));
    /// assert_eq!(iter.next(), Some([0, 1]));
    /// assert_eq!(iter.next(), Some([0, 2]));
    /// assert_eq!(iter.next(), Some([1, 0]));
    /// assert_eq!(iter.next(), Some([1, 1]));
    /// assert_eq!(iter.next(), Some([1, 2]));
    /// assert!(iter.next().is_none());
    /// ```
    pub fn indices(&self) -> Indices<ConstShape<D>> {
        Indices::from_shape(self.shape)
    }
}

impl<S: Shape> SfsBase<S, Norm> {
    /// Creates a new, normalised, and uniform SFS.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs::Sfs;
    /// let sfs = Sfs::uniform([2, 5]);
    /// assert!(sfs.iter().all(|&x| x == 0.1));
    /// ```
    pub fn uniform(shape: S) -> SfsBase<S, Norm> {
        let n: usize = shape.iter().product();

        let elem = 1.0 / n as f64;

        SfsBase::new_unchecked(vec![elem; n], shape)
    }
}

impl<S: Shape> SfsBase<S, Unnorm> {
    /// Returns the a mutable reference values of the SFS as a flat, row-major slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs2d;
    /// let mut sfs = sfs2d![
    ///     [0., 1., 2.],
    ///     [3., 4., 5.],
    /// ];
    /// assert_eq!(sfs.as_slice(), [0., 1., 2., 3., 4., 5.]);
    /// sfs.as_mut_slice()[0] = 100.;
    /// assert_eq!(sfs.as_slice(), [100., 1., 2., 3., 4., 5.]);
    /// ```
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// Creates a new, unnormalised SFS by repeating a single value.
    ///
    /// See also [`Sfs::uniform`] to create a normalised SFS with uniform values.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs::USfs;
    /// let sfs = USfs::from_elem(0.1, [7, 5]);
    /// assert_eq!(sfs.shape(), &[7, 5]);
    /// assert!(sfs.iter().all(|&x| x == 0.1));
    /// ```
    pub fn from_elem(elem: f64, shape: S) -> Self {
        let n = shape.iter().product();

        Self::new_unchecked(vec![elem; n], shape)
    }

    /// Creates a new, unnormalised SFS from an iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs::USfs;
    /// let iter = (0..9).map(|x| x as f64);
    /// let sfs = USfs::from_iter_shape(iter, [3, 3]).expect("shape didn't fit iterator!");
    /// assert_eq!(sfs[[1, 2]], 5.0);
    /// ```
    pub fn from_iter_shape<I>(iter: I, shape: S) -> Result<Self, ShapeError<S>>
    where
        I: IntoIterator<Item = f64>,
    {
        Self::from_vec_shape(iter.into_iter().collect(), shape)
    }

    /// Creates a new, unnormalised SFS from a vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs::USfs;
    /// let vec: Vec<f64> = (0..9).map(|x| x as f64).collect();
    /// let sfs = USfs::from_vec_shape(vec, [3, 3]).expect("shape didn't fit vector!");
    /// assert_eq!(sfs[[2, 0]], 6.0);
    /// ```
    pub fn from_vec_shape(vec: Vec<f64>, shape: S) -> Result<Self, ShapeError<S>> {
        let n: usize = shape.iter().product();

        match vec.len() == n {
            true => Ok(Self::new_unchecked(vec, shape)),
            false => Err(ShapeError::new(n, shape)),
        }
    }

    /// Returns a mutable reference to a value at an index in the SFS.
    ///
    /// If the index is out of bounds, `None` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs1d;
    /// let mut sfs = sfs1d![0.0, 0.1, 0.2];
    /// assert_eq!(sfs[[0]], 0.0);
    /// if let Some(v) = sfs.get_mut(&[0]) {
    ///     *v = 0.5;
    /// }
    /// assert_eq!(sfs[[0]], 0.5);
    /// ```
    ///
    /// ```
    /// use winsfs_core::sfs2d;
    /// let mut sfs = sfs2d![[0.0, 0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]];
    /// assert_eq!(sfs[[0, 0]], 0.0);
    /// if let Some(v) = sfs.get_mut(&[0, 0]) {
    ///     *v = 0.5;
    /// }
    /// assert_eq!(sfs[[0, 0]], 0.5);
    /// ```
    #[inline]
    pub fn get_mut(&mut self, index: &S) -> Option<&mut f64> {
        self.values.get_mut(compute_flat(index, &self.shape)?)
    }

    /// Returns an iterator over mutable references to the elements in the SFS in row-major order.
    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, f64> {
        self.values.iter_mut()
    }

    /// Returns a normalised SFS, consuming `self`.
    ///
    /// The values in the SFS are modified to sum to one.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::{sfs1d, sfs::{Sfs, USfs}};
    /// let sfs: USfs<1> = sfs1d![0., 1., 2., 3., 4.];
    /// let sfs: Sfs<1> = sfs.normalise();
    /// assert_eq!(sfs[[1]], 0.1);
    /// ```
    #[inline]
    #[must_use = "returns normalised SFS, doesn't modify in-place"]
    pub fn normalise(mut self) -> SfsBase<S, Norm> {
        let sum = self.sum();

        self.iter_mut().for_each(|x| *x /= sum);

        self.into_normalised_unchecked()
    }

    /// Creates a new, unnnormalised SFS with all entries set to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs::USfs;
    /// let sfs = USfs::zeros([2, 5]);
    /// assert!(sfs.iter().all(|&x| x == 0.0));
    /// ```
    pub fn zeros(shape: S) -> Self {
        Self::from_elem(0.0, shape)
    }
}

impl SfsBase<ConstShape<1>, Unnorm> {
    /// Creates a new SFS from a vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs::USfs;
    /// let sfs = USfs::from_vec(vec![0., 1., 2.]);
    /// assert_eq!(sfs.shape(), &[3]);
    /// assert_eq!(sfs[[1]], 1.);
    /// ```
    pub fn from_vec(values: Vec<f64>) -> Self {
        let shape = [values.len()];

        Self::new_unchecked(values, shape)
    }
}

impl SfsBase<ConstShape<2>, Norm> {
    /// Returns the f2-statistic.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs2d;
    /// let sfs = sfs2d![
    ///     [1., 0., 0.],
    ///     [0., 1., 0.],
    ///     [0., 0., 1.],
    /// ].normalise();
    /// assert_eq!(sfs.f2(), 0.);
    /// ```
    pub fn f2(&self) -> f64 {
        self.iter()
            .zip(self.frequencies())
            .map(|(v, [f_i, f_j])| v * (f_i - f_j).powi(2))
            .sum()
    }

    /// Returns the Fst statistic.
    ///
    /// The Fst calculation implemented here corresponds to the recommendation in
    /// [Bhatia et al. (2013)][bhatia], i.e. what they term Hudson's estimator using a ratio
    /// of averages.
    ///
    /// [bhatia]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3759727/
    pub fn fst(&self) -> f64 {
        let [n_i_sub, n_j_sub] = self.shape().map(|x| (x - 2) as f64);

        // We only want the polymorphic parts of the spectrum and corresponding frequencies,
        // so we drop the first and last values
        let polymorphic_iter = self
            .values
            .iter()
            .zip(self.frequencies())
            .take(self.values.len() - 1)
            .skip(1);

        let (num, denom) = polymorphic_iter
            .map(|(v, f)| (v, f, f.map(|f| 1. - f)))
            .map(|(v, [f_i, f_j], [g_i, g_j])| {
                let num = (f_i - f_j).powi(2) - f_i * g_i / n_i_sub - f_j * g_j / n_j_sub;
                let denom = f_i * g_j + f_j * g_i;
                (v * num, v * denom)
            })
            .fold((0., 0.), |(n_sum, d_sum), (n, d)| (n_sum + n, d_sum + d));

        num / denom
    }
}

impl<N: Normalisation> SfsBase<ConstShape<2>, N> {
    /// Returns the King kinship statistic.
    ///
    /// If the SFS does not have shape 3x3, `None` is returned. If all heterozygote bins are zero,
    /// `NaN` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs2d;
    /// let sfs = sfs2d![
    ///     [0.,  2., 1.],
    ///     [2., 12., 2.],
    ///     [1.,  2., 0.],
    /// ].normalise();
    /// assert!((sfs.king().unwrap() - 0.25).abs() < f64::EPSILON);
    /// ```
    pub fn king(&self) -> Option<f64> {
        match &self.shape[..] {
            [3, 3] => {
                let numer = self[[1, 1]] - 2. * (self[[0, 2]] + self[[2, 0]]);
                let denom =
                    self[[0, 1]] + self[[1, 0]] + 2. * self[[1, 1]] + self[[1, 2]] + self[[2, 1]];

                let king = numer / denom;

                Some(king)
            }
            _ => None,
        }
    }

    /// Returns the R0 kinship statistic.
    ///
    /// If the SFS does not have shape 3x3, `None` is returned. If the `[1, 1]` bin is zero, `NaN`
    /// is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs2d;
    /// let sfs = sfs2d![
    ///     [0., 0., 1.],
    ///     [0., 4., 0.],
    ///     [1., 0., 0.],
    /// ].normalise();
    /// assert!((sfs.r0().unwrap() - 0.5).abs() < f64::EPSILON);
    /// ```
    pub fn r0(&self) -> Option<f64> {
        match &self.shape[..] {
            [3, 3] => {
                let r0 = (self[[0, 2]] + self[[2, 0]]) / self[[1, 1]];

                Some(r0)
            }
            _ => None,
        }
    }

    /// Returns the R1 kinship statistic.
    ///
    /// If the SFS does not have shape 3x3, `None` is returned. If all off-diagonal bins are zero,
    /// `NaN` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs2d;
    /// let sfs = sfs2d![
    ///     [0., 1., 1.],
    ///     [1., 3., 1.],
    ///     [1., 1., 0.],
    /// ].normalise();
    /// assert!((sfs.r1().unwrap() - 0.5).abs() < f64::EPSILON);
    /// ```
    pub fn r1(&self) -> Option<f64> {
        match &self.shape[..] {
            [3, 3] => {
                let denom = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]
                    .iter()
                    .map(|&i| self[i])
                    .sum::<f64>();
                let r1 = self[[1, 1]] / denom;

                Some(r1)
            }
            _ => None,
        }
    }
}

macro_rules! impl_op {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident) => {
        impl<S: Shape, N: Normalisation> $assign_trait<&SfsBase<S, N>> for SfsBase<S, Unnorm> {
            #[inline]
            fn $assign_method(&mut self, rhs: &SfsBase<S, N>) {
                assert_eq!(self.shape, rhs.shape);

                self.iter_mut()
                    .zip(rhs.iter())
                    .for_each(|(x, rhs)| x.$assign_method(rhs));
            }
        }

        impl<S: Shape, N: Normalisation> $assign_trait<SfsBase<S, N>> for SfsBase<S, Unnorm> {
            #[inline]
            fn $assign_method(&mut self, rhs: SfsBase<S, N>) {
                self.$assign_method(&rhs);
            }
        }

        impl<S: Shape, N: Normalisation, M: Normalisation> $trait<SfsBase<S, M>> for SfsBase<S, N> {
            type Output = SfsBase<S, Unnorm>;

            #[inline]
            fn $method(self, rhs: SfsBase<S, M>) -> Self::Output {
                let mut sfs = self.into_unnormalised();
                sfs.$assign_method(&rhs);
                sfs
            }
        }

        impl<S: Shape, N: Normalisation, M: Normalisation> $trait<&SfsBase<S, M>>
            for SfsBase<S, N>
        {
            type Output = SfsBase<S, Unnorm>;

            #[inline]
            fn $method(self, rhs: &SfsBase<S, M>) -> Self::Output {
                let mut sfs = self.into_unnormalised();
                sfs.$assign_method(rhs);
                sfs
            }
        }
    };
}
impl_op!(Add, add, AddAssign, add_assign);
impl_op!(Sub, sub, SubAssign, sub_assign);

impl<S: Shape, N: Normalisation> Index<S> for SfsBase<S, N> {
    type Output = f64;

    #[inline]
    fn index(&self, index: S) -> &Self::Output {
        self.get(&index).unwrap()
    }
}

impl<S: Shape> IndexMut<S> for SfsBase<S, Unnorm> {
    #[inline]
    fn index_mut(&mut self, index: S) -> &mut Self::Output {
        self.get_mut(&index).unwrap()
    }
}

impl<const D: usize, N: Normalisation> From<SfsBase<ConstShape<D>, N>> for SfsBase<DynShape, N> {
    fn from(sfs: SfsBase<ConstShape<D>, N>) -> Self {
        SfsBase {
            values: sfs.values,
            shape: sfs.shape.into(),
            strides: sfs.strides.into(),
            norm: PhantomData,
        }
    }
}

impl<const D: usize, N: Normalisation> TryFrom<SfsBase<DynShape, N>> for SfsBase<ConstShape<D>, N> {
    type Error = SfsBase<DynShape, N>;

    fn try_from(sfs: SfsBase<DynShape, N>) -> Result<Self, Self::Error> {
        match (
            <[usize; D]>::try_from(&sfs.shape[..]),
            <[usize; D]>::try_from(&sfs.strides[..]),
        ) {
            (Ok(shape), Ok(strides)) => Ok(SfsBase {
                values: sfs.values,
                shape,
                strides,
                norm: PhantomData,
            }),
            (Err(_), Err(_)) => Err(sfs),
            (Ok(_), Err(_)) | (Err(_), Ok(_)) => {
                unreachable!("conversion of dyn shape and strides succeeds or fails together")
            }
        }
    }
}

/// An error associated with SFS construction using invalid shape.
#[derive(Clone, Copy, Debug)]
pub struct ShapeError<S: Shape> {
    n: usize,
    shape: S,
}

impl<S: Shape> ShapeError<S> {
    fn new(n: usize, shape: S) -> Self {
        Self { n, shape }
    }
}

impl<S: Shape> fmt::Display for ShapeError<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape_fmt = self
            .shape
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join("/");
        let n = self.n;
        let d = self.shape.as_ref().len();

        write!(
            f,
            "cannot create {d}D SFS with shape {shape_fmt} from {n} elements"
        )
    }
}

impl<S: Shape> Error for ShapeError<S> {}

/// An error associated with normalised SFS construction using unnormalised input.
#[derive(Clone, Copy, Debug)]
pub struct NormError {
    sum: f64,
}

impl fmt::Display for NormError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cannot create normalised SFS using values summing to {}",
            self.sum
        )
    }
}

impl Error for NormError {}

fn compute_flat<S: Shape>(index: &S, shape: &S) -> Option<usize> {
    assert_eq!(index.len(), shape.len());

    for i in 1..index.len() {
        if index.as_ref()[i] >= shape.as_ref()[i] {
            return None;
        }
    }
    Some(compute_flat_unchecked(index, shape))
}

fn compute_flat_unchecked<S: Shape>(index: &S, shape: &S) -> usize {
    let mut flat = index.as_ref()[0];
    for i in 1..index.len() {
        flat *= shape.as_ref()[i];
        flat += index.as_ref()[i];
    }
    flat
}

fn compute_index_sum_unchecked(mut flat: usize, mut n: usize, shape: &[usize]) -> usize {
    let mut sum = 0;
    for v in shape {
        n /= v;
        sum += flat / n;
        flat %= n;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_1d() {
        let sfs = sfs1d![0., 1., 2., 3., 4., 5.];
        assert_eq!(sfs.get(&[0]), Some(&0.));
        assert_eq!(sfs.get(&[2]), Some(&2.));
        assert_eq!(sfs.get(&[5]), Some(&5.));
        assert_eq!(sfs.get(&[6]), None);
    }

    #[test]
    fn test_index_2d() {
        let sfs = sfs2d![[0., 1., 2.], [3., 4., 5.]];
        assert_eq!(sfs.get(&[0, 0]), Some(&0.));
        assert_eq!(sfs.get(&[1, 0]), Some(&3.));
        assert_eq!(sfs.get(&[1, 1]), Some(&4.));
        assert_eq!(sfs.get(&[1, 2]), Some(&5.));
        assert_eq!(sfs.get(&[2, 0]), None);
        assert_eq!(sfs.get(&[0, 3]), None);
    }

    #[test]
    fn test_f2() {
        #[rustfmt::skip]
        let sfs = sfs2d![
            [0., 1., 2.],
            [3., 4., 5.]
        ].normalise();
        assert!((sfs.f2() - 0.4166667).abs() < 1e-6);
    }

    #[test]
    fn test_sfs_addition() {
        let mut lhs = sfs1d![0., 1., 2.];
        let rhs = sfs1d![5., 6., 7.];
        let sum = sfs1d![5., 7., 9.];

        assert_eq!(lhs.clone() + rhs.clone(), sum);
        assert_eq!(lhs.clone() + &rhs, sum);

        lhs += rhs.clone();
        assert_eq!(lhs, sum);
        lhs += &rhs;
        assert_eq!(lhs, sum + rhs);
    }

    #[test]
    fn test_sfs_subtraction() {
        let mut lhs = sfs1d![5., 6., 7.];
        let rhs = sfs1d![0., 1., 2.];
        let sub = sfs1d![5., 5., 5.];

        assert_eq!(lhs.clone() - rhs.clone(), sub);
        assert_eq!(lhs.clone() - &rhs, sub);

        lhs -= rhs.clone();
        assert_eq!(lhs, sub);
        lhs -= &rhs;
        assert_eq!(lhs, sub - rhs);
    }

    #[test]
    fn test_fold_4() {
        let sfs = sfs1d![0., 1., 2., 3.];

        assert_eq!(sfs.fold(), sfs1d![3., 3., 0., 0.],);
    }

    #[test]
    fn test_fold_5() {
        let sfs = sfs1d![0., 1., 2., 3., 4.];

        assert_eq!(sfs.fold(), sfs1d![4., 4., 2., 0., 0.],);
    }

    #[test]
    fn test_fold_3x3() {
        #[rustfmt::skip]
        let sfs = sfs2d![
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.],
        ];

        #[rustfmt::skip]
        let expected = sfs2d![
            [8., 8., 4.],
            [8., 4., 0.],
            [4., 0., 0.],
        ];

        assert_eq!(sfs.fold(), expected);
    }

    #[test]
    fn test_fold_2x4() {
        #[rustfmt::skip]
        let sfs = sfs2d![
            [0., 1., 2., 3.],
            [4., 5., 6., 7.],
        ];

        #[rustfmt::skip]
        let expected = sfs2d![
            [7., 7.,  3.5, 0.],
            [7., 3.5, 0.,  0.],
        ];

        assert_eq!(sfs.fold(), expected);
    }

    #[test]
    fn test_fold_3x4() {
        #[rustfmt::skip]
        let sfs = sfs2d![
            [0., 1.,  2.,  3.],
            [4., 5.,  6.,  7.],
            [8., 9., 10., 11.],
        ];

        #[rustfmt::skip]
        let expected = sfs2d![
            [11., 11., 11., 0.],
            [11., 11.,  0., 0.],
            [11.,  0.,  0., 0.],
        ];

        assert_eq!(sfs.fold(), expected);
    }

    #[test]
    fn test_fold_3x7() {
        #[rustfmt::skip]
        let sfs = sfs2d![
            [ 0.,  1.,  2.,  3.,  4.,  5.,  6.],
            [ 7.,  8.,  9., 10., 11., 12., 13.],
            [14., 15., 16., 17., 18., 19., 20.],
        ];

        #[rustfmt::skip]
        let expected = sfs2d![
            [20., 20., 20., 20., 10., 0., 0.],
            [20., 20., 20., 10.,  0., 0., 0.],
            [20., 20., 10.,  0.,  0., 0., 0.],
        ];

        assert_eq!(sfs.fold(), expected);
    }

    #[test]
    fn test_fold_2x2x2() {
        let sfs = USfs::from_iter_shape((0..8).map(|x| x as f64), [2, 2, 2]).unwrap();

        #[rustfmt::skip]
        let expected = USfs::from_vec_shape(
            vec![
                7., 7.,
                7., 0.,
                
                7., 0.,
                0., 0.,
            ],
            [2, 2, 2]
        ).unwrap();

        assert_eq!(sfs.fold(), expected);
    }

    #[test]
    fn test_fold_2x3x2() {
        let sfs = USfs::from_iter_shape((0..12).map(|x| x as f64), [2, 3, 2]).unwrap();

        #[rustfmt::skip]
        let expected = USfs::from_vec_shape(
            vec![
                11., 11.,  
                11.,  5.5,
                5.5,  0.,
                
                11.,  5.5,
                 5.5, 0.,
                 0.,  0.,
            ],
            [2, 3, 2]
        ).unwrap();

        assert_eq!(sfs.fold(), expected);
    }

    #[test]
    fn test_fold_3x3x3() {
        let sfs = USfs::from_iter_shape((0..27).map(|x| x as f64), [3, 3, 3]).unwrap();

        #[rustfmt::skip]
        let expected = USfs::from_vec_shape(
            vec![
                26., 26., 26.,
                26., 26., 13.,
                26., 13.,  0.,
                
                26., 26., 13.,
                26., 13.,  0.,
                13.,  0.,  0.,

                26., 13.,  0.,
                13.,  0.,  0.,
                 0.,  0.,  0.,
            ],
            [3, 3, 3]
        ).unwrap();

        assert_eq!(sfs.fold(), expected);
    }

    #[test]
    #[rustfmt::skip]
    fn test_king_bins_used() {
        let fst = sfs2d![
            [0., 1., 1.], 
            [1., 1., 1.], 
            [1., 1., 0.],
        ];
        let snd = sfs2d![
            [2., 1., 1.], 
            [1., 1., 1.], 
            [1., 1., 2.],
        ];
        assert!(!fst.king().unwrap().is_nan());
        assert_eq!(fst.king(), snd.king());
    }

    #[test]
    #[rustfmt::skip]
    fn test_r0_bins_used() {
        let fst = sfs2d![
            [0., 0., 1.], 
            [0., 1., 0.], 
            [1., 0., 0.],
        ];
        let snd = sfs2d![
            [2., 2., 1.], 
            [2., 1., 2.], 
            [1., 2., 2.],
        ];
        assert!(!fst.r0().unwrap().is_nan());
        assert_eq!(fst.r0(), snd.r0());
    }

    #[test]
    #[rustfmt::skip]
    fn test_r1_bins_used() {
        let fst = sfs2d![
            [0., 1., 1.], 
            [1., 1., 1.], 
            [1., 1., 0.],
        ];
        let snd = sfs2d![
            [2., 1., 1.], 
            [1., 1., 1.], 
            [1., 1., 2.],
        ];
        assert!(!fst.r1().unwrap().is_nan());
        assert_eq!(fst.r1(), snd.r1());
    }

    #[test]
    fn test_fst() {
        #[rustfmt::skip]
        let sfs = sfs2d![
            [29880., 13., 19., 4., 14., 0., 1., 0., 0.],
            [    7.,  0.,  0., 0.,  0., 0., 0., 0., 0.],
            [   24.,  0.,  0., 0.,  0., 0., 1., 0., 0.],
            [    0.,  0.,  0., 0.,  0., 0., 0., 0., 0.],
            [    9.,  0.,  0., 0.,  4., 0., 2., 0., 5.],
            [    0.,  0.,  0., 0.,  0., 0., 0., 0., 0.],
            [    1.,  0.,  0., 0.,  7., 0., 1., 0., 8.],
        ].normalise();
        let expected = 0.307106;
        let fst = sfs.fst();
        dbg!(fst);
        assert!((fst - expected).abs() < 1e-6);
    }
}
