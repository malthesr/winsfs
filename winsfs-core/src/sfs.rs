//! N-dimensional site frequency spectra ("SFS").

use std::{
    error::Error,
    fmt::{self, Write as _},
    fs,
    io::{self, Read},
    ops::{Add, AddAssign, Index, IndexMut, Sub, SubAssign},
    path::Path,
    slice,
};

mod angsd;
pub use angsd::ParseAngsdError;

mod em;

use crate::ArrayExt;

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
        $crate::sfs::Sfs::from_elem($elem, [$n])
    };
    ($($x:expr),+ $(,)?) => {
        $crate::sfs::Sfs::from_vec(vec![$($x),+])
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
        $crate::sfs::Sfs::from_vec_shape(vec, shape).unwrap()
    }};
}

/// An unnormalised, N-dimensional site frequency spectrum ("SFS").
pub type UnnormalisedSfs<const N: usize> = Sfs<N, false>;

/// An N-dimensional site frequency spectrum ("SFS").
///
/// Elements are stored in row-major order: the last index varies the fastest.
/// The SFS may or may not be normalised to probability scale, and this is controlled
/// at the type-level by the `NORM` parameter, which by default is `true`.
#[derive(Clone, Debug, PartialEq)]
// TODO: Replace bool with enum once these are permitted in const generics,
// see github.com/rust-lang/rust/issues/95174
pub struct Sfs<const N: usize, const NORM: bool = true> {
    values: Vec<f64>,
    shape: [usize; N],
    strides: [usize; N],
}

impl<const N: usize, const NORM: bool> Sfs<N, NORM> {
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

    /// Returns a string containing the SFS formatted in ANGSD format.
    ///
    /// The resulting string contains a header giving the shape of the SFS,
    /// and a flat representation of the SFS.
    pub fn format_angsd(&self, precision: Option<usize>) -> String {
        angsd::format(self, precision)
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
    pub fn frequencies(&self) -> impl Iterator<Item = [f64; N]> {
        let n_arr = self.shape.map(|n| n - 1);
        self.indices()
            .map(move |idx_arr| idx_arr.zip(n_arr).map(|(i, n)| i as f64 / n as f64))
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
    /// assert_eq!(sfs.get([0]), Some(&0.0));
    /// assert_eq!(sfs.get([1]), Some(&0.1));
    /// assert_eq!(sfs.get([2]), Some(&0.2));
    /// assert_eq!(sfs.get([3]), None);
    /// ```
    ///
    /// ```
    /// use winsfs_core::sfs2d;
    /// let sfs = sfs2d![[0.0, 0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]];
    /// assert_eq!(sfs.get([0, 0]), Some(&0.0));
    /// assert_eq!(sfs.get([1, 2]), Some(&0.5));
    /// assert_eq!(sfs.get([3, 0]), None);
    /// ```
    #[inline]
    pub fn get(&self, index: [usize; N]) -> Option<&f64> {
        self.values.get(compute_flat(index, self.shape)?)
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
    pub fn indices(&self) -> impl Iterator<Item = [usize; N]> {
        let n = self.as_slice().len();
        let shape = self.shape;
        (0..n).map(move |flat| compute_index_unchecked(flat, n, shape))
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
    /// use winsfs_core::sfs1d;
    /// let sfs = sfs1d![0.2; 5];
    /// assert!(!sfs.is_normalised());
    /// let sfs = sfs.into_normalised().unwrap();
    /// assert!(sfs.is_normalised());
    /// ```
    ///
    /// Otherwise, an unnormalised SFS cannot be normalised SFS using this method:
    ///
    /// ```
    /// use winsfs_core::sfs1d;
    /// let sfs = sfs1d![2.; 5];
    /// assert!(sfs.into_normalised().is_err());
    /// ```
    ///
    /// Use [`Sfs::normalise`] instead.
    #[inline]
    pub fn into_normalised(self) -> Result<Sfs<N>, NormalisationError> {
        let sum = self.sum();

        if (sum - 1.).abs() <= NORMALISATION_TOLERANCE {
            Ok(self.into_normalised_unchecked())
        } else {
            Err(NormalisationError { sum })
        }
    }

    #[inline]
    fn into_normalised_unchecked(self) -> Sfs<N> {
        Sfs {
            values: self.values,
            shape: self.shape,
            strides: self.strides,
        }
    }

    /// Returns an unnormalised SFS, consuming `self`.
    ///
    /// This works purely on the type level, and does not modify the actual values in the SFS.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs::Sfs;
    /// let sfs = Sfs::uniform([7]);
    /// assert!(sfs.is_normalised());
    /// let sfs = sfs.into_unnormalised();
    /// assert!(!sfs.is_normalised());
    /// ```
    #[inline]
    pub fn into_unnormalised(self) -> UnnormalisedSfs<N> {
        Sfs {
            values: self.values,
            shape: self.shape,
            strides: self.strides,
        }
    }

    /// Returns `true` if the SFS is normalised, `false` otherwise.
    ///
    /// This works purely on the type level.
    pub const fn is_normalised(&self) -> bool {
        NORM
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
    fn new_unchecked(values: Vec<f64>, shape: [usize; N]) -> Self {
        let strides = compute_strides(shape);

        Self {
            values,
            shape,
            strides,
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
    pub fn scale(mut self, scale: f64) -> UnnormalisedSfs<N> {
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
    /// assert_eq!(sfs.shape(), [2, 3]);
    /// ```
    pub fn shape(&self) -> [usize; N] {
        self.shape
    }

    /// Returns the sum of values in the SFS.
    #[inline]
    fn sum(&self) -> f64 {
        self.iter().sum()
    }
}

impl<const N: usize> Sfs<N> {
    /// Creates a new, normalised, and uniform SFS.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs::Sfs;
    /// let sfs = Sfs::uniform([2, 5]);
    /// assert!(sfs.iter().all(|&x| x == 0.1));
    /// ```
    pub fn uniform(shape: [usize; N]) -> Self {
        let n: usize = shape.iter().product();

        let elem = 1.0 / n as f64;

        Sfs::new_unchecked(vec![elem; n], shape)
    }
}

impl<const N: usize> UnnormalisedSfs<N> {
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
    /// use winsfs_core::sfs::Sfs;
    /// let sfs = Sfs::from_elem(0.1, [7, 5]);
    /// assert_eq!(sfs.shape(), [7, 5]);
    /// assert!(sfs.iter().all(|&x| x == 0.1));
    /// ```
    pub fn from_elem(elem: f64, shape: [usize; N]) -> Self {
        let n = shape.iter().product();

        Self::new_unchecked(vec![elem; n], shape)
    }

    /// Creates a new, unnormalised SFS from an iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs::Sfs;
    /// let iter = (0..9).map(|x| x as f64);
    /// let sfs = Sfs::from_iter_shape(iter, [3, 3]).expect("shape didn't fit iterator!");
    /// assert_eq!(sfs[[1, 2]], 5.0);
    /// ```
    pub fn from_iter_shape<I>(iter: I, shape: [usize; N]) -> Result<Self, ShapeError<N>>
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
    /// use winsfs_core::sfs::Sfs;
    /// let vec: Vec<f64> = (0..9).map(|x| x as f64).collect();
    /// let sfs = Sfs::from_vec_shape(vec, [3, 3]).expect("shape didn't fit vector!");
    /// assert_eq!(sfs[[2, 0]], 6.0);
    /// ```
    pub fn from_vec_shape(vec: Vec<f64>, shape: [usize; N]) -> Result<Self, ShapeError<N>> {
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
    /// if let Some(v) = sfs.get_mut([0]) {
    ///     *v = 0.5;
    /// }
    /// assert_eq!(sfs[[0]], 0.5);
    /// ```
    ///
    /// ```
    /// use winsfs_core::sfs2d;
    /// let mut sfs = sfs2d![[0.0, 0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]];
    /// assert_eq!(sfs[[0, 0]], 0.0);
    /// if let Some(v) = sfs.get_mut([0, 0]) {
    ///     *v = 0.5;
    /// }
    /// assert_eq!(sfs[[0, 0]], 0.5);
    /// ```
    #[inline]
    pub fn get_mut(&mut self, index: [usize; N]) -> Option<&mut f64> {
        self.values.get_mut(compute_flat(index, self.shape)?)
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
    /// use winsfs_core::sfs1d;
    /// let sfs = sfs1d![0., 1., 2., 3., 4.];
    /// assert!(!sfs.is_normalised());
    /// let sfs = sfs.normalise();
    /// assert!(sfs.is_normalised());
    /// assert_eq!(sfs[[1]], 0.1);
    /// ```
    #[inline]
    #[must_use = "returns normalised SFS, doesn't modify in-place"]
    pub fn normalise(mut self) -> Sfs<N> {
        let sum = self.sum();

        self.iter_mut().for_each(|x| *x /= sum);

        self.into_normalised_unchecked()
    }

    /// Creates a new, unnormalised SFS from a string containing an SFS formatted in ANGSD format.
    pub fn parse_from_angsd(s: &str) -> Result<Self, ParseAngsdError<N>> {
        angsd::parse(s)
    }

    /// Creates a new, unnormalised SFS from a path containing an SFS formatted in ANGSD format.
    pub fn read_from_angsd<P>(path: P) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        let mut file = fs::File::open(path)?;
        let mut buf = String::new();
        file.read_to_string(&mut buf)?;
        Self::parse_from_angsd(&buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Creates a new, unnnormalised SFS with all entries set to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs::Sfs;
    /// let sfs = Sfs::zeros([2, 5]);
    /// assert!(sfs.iter().all(|&x| x == 0.0));
    /// ```
    pub fn zeros(shape: [usize; N]) -> Self {
        Self::from_elem(0.0, shape)
    }
}

macro_rules! impl_op {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident) => {
        impl<const N: usize, const NORM: bool> $assign_trait<&Sfs<N, NORM>> for UnnormalisedSfs<N> {
            #[inline]
            fn $assign_method(&mut self, rhs: &Sfs<N, NORM>) {
                assert_eq!(self.shape, rhs.shape);

                self.iter_mut()
                    .zip(rhs.iter())
                    .for_each(|(x, rhs)| x.$assign_method(rhs));
            }
        }

        impl<const N: usize, const NORM: bool> $assign_trait<Sfs<N, NORM>> for UnnormalisedSfs<N> {
            #[inline]
            fn $assign_method(&mut self, rhs: Sfs<N, NORM>) {
                self.$assign_method(&rhs);
            }
        }

        impl<const N: usize, const NORM: bool, const NORM_RHS: bool> $trait<Sfs<N, NORM_RHS>>
            for Sfs<N, NORM>
        {
            type Output = UnnormalisedSfs<N>;

            #[inline]
            fn $method(self, rhs: Sfs<N, NORM_RHS>) -> Self::Output {
                let mut sfs = self.into_unnormalised();
                sfs.$assign_method(&rhs);
                sfs
            }
        }

        impl<const N: usize, const NORM: bool, const NORM_RHS: bool> $trait<&Sfs<N, NORM_RHS>>
            for Sfs<N, NORM>
        {
            type Output = UnnormalisedSfs<N>;

            #[inline]
            fn $method(self, rhs: &Sfs<N, NORM_RHS>) -> Self::Output {
                let mut sfs = self.into_unnormalised();
                sfs.$assign_method(rhs);
                sfs
            }
        }
    };
}
impl_op!(Add, add, AddAssign, add_assign);
impl_op!(Sub, sub, SubAssign, sub_assign);

impl<const N: usize, const NORM: bool> Index<[usize; N]> for Sfs<N, NORM> {
    type Output = f64;

    #[inline]
    fn index(&self, index: [usize; N]) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl<const N: usize> IndexMut<[usize; N]> for UnnormalisedSfs<N> {
    #[inline]
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        self.get_mut(index).unwrap()
    }
}

impl UnnormalisedSfs<1> {
    /// Creates a new SFS from a vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::sfs::Sfs;
    /// let sfs = Sfs::from_vec(vec![0., 1., 2.]);
    /// assert_eq!(sfs.shape(), [3]);
    /// assert_eq!(sfs[[1]], 1.);
    /// ```
    pub fn from_vec(values: Vec<f64>) -> Self {
        let shape = [values.len()];

        Self::new_unchecked(values, shape)
    }
}

impl Sfs<2> {
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
}

/// An error associated with SFS construction using invalid shape.
#[derive(Clone, Copy, Debug)]
pub struct ShapeError<const N: usize> {
    n: usize,
    shape: [usize; N],
}

impl<const N: usize> ShapeError<N> {
    fn new(n: usize, shape: [usize; N]) -> Self {
        Self { n, shape }
    }
}

impl<const N: usize> fmt::Display for ShapeError<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape_fmt = self.shape.map(|x| x.to_string()).join("/");
        let n = self.n;

        write!(
            f,
            "cannot create {N}D SFS with shape {shape_fmt} from {n} elements"
        )
    }
}

impl<const N: usize> Error for ShapeError<N> {}

/// An error associated with normalised SFS construction using unnormalised input.
#[derive(Clone, Copy, Debug)]
pub struct NormalisationError {
    sum: f64,
}

impl fmt::Display for NormalisationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cannot create normalised SFS using values summing to {}",
            self.sum
        )
    }
}

impl Error for NormalisationError {}

fn compute_flat<const N: usize>(index: [usize; N], shape: [usize; N]) -> Option<usize> {
    for i in 1..N {
        if index[i] >= shape[i] {
            return None;
        }
    }
    Some(compute_flat_unchecked(index, shape))
}

fn compute_flat_unchecked<const N: usize>(index: [usize; N], shape: [usize; N]) -> usize {
    let mut flat = index[0];
    for i in 1..N {
        flat *= shape[i];
        flat += index[i];
    }
    flat
}

fn compute_index_unchecked<const N: usize>(
    mut flat: usize,
    mut n: usize,
    shape: [usize; N],
) -> [usize; N] {
    let mut index = [0; N];
    for i in 0..N {
        n /= shape[i];
        index[i] = flat / n;
        flat %= n;
    }
    index
}

fn compute_strides<const N: usize>(shape: [usize; N]) -> [usize; N] {
    let mut strides = [1; N];
    for (i, v) in shape.iter().enumerate().skip(1).rev() {
        strides.iter_mut().take(i).for_each(|stride| *stride *= v)
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_abs_diff_eq;

    #[test]
    fn test_index_1d() {
        let sfs = sfs1d![0., 1., 2., 3., 4., 5.];
        assert_eq!(sfs.get([0]), Some(&0.));
        assert_eq!(sfs.get([2]), Some(&2.));
        assert_eq!(sfs.get([5]), Some(&5.));
        assert_eq!(sfs.get([6]), None);
    }

    #[test]
    fn test_index_2d() {
        let sfs = sfs2d![[0., 1., 2.], [3., 4., 5.]];
        assert_eq!(sfs.get([0, 0]), Some(&0.));
        assert_eq!(sfs.get([1, 0]), Some(&3.));
        assert_eq!(sfs.get([1, 1]), Some(&4.));
        assert_eq!(sfs.get([1, 2]), Some(&5.));
        assert_eq!(sfs.get([2, 0]), None);
        assert_eq!(sfs.get([0, 3]), None);
    }

    #[test]
    fn test_compute_index() {
        assert_eq!(compute_index_unchecked(3, 4, [4]), [3]);
        assert_eq!(compute_index_unchecked(16, 28, [4, 7]), [2, 2]);
        assert_eq!(compute_index_unchecked(3, 6, [1, 3, 2]), [0, 1, 1]);
    }

    #[test]
    fn test_compute_strides() {
        assert_eq!(compute_strides([7]), [1]);
        assert_eq!(compute_strides([9, 3]), [3, 1]);
        assert_eq!(compute_strides([3, 7, 5]), [35, 5, 1]);
        assert_eq!(compute_strides([9, 3, 5, 7]), [105, 35, 7, 1]);
    }

    #[test]
    fn test_f2() {
        #[rustfmt::skip]
        let sfs = sfs2d![
            [0., 1., 2.],
            [3., 4., 5.]
        ].normalise();
        assert_abs_diff_eq!(sfs.f2(), 0.4166667, epsilon = 1e-6);
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
}
