//! N-dimensional site frequency spectra ("SFS").

use std::{
    error::Error,
    fmt, fs,
    io::{self, Read},
    ops::{Add, AddAssign, Index, IndexMut},
    path::Path,
    slice,
};

pub type Sfs1d = Sfs<1>;
pub type Sfs2d = Sfs<2>;

mod angsd;
pub use angsd::ParseAngsdError;

mod em;
pub use em::Em;

/// Creates a 1D SFS containing the arguments.
///
/// This is mainly intended for readability in doc-tests, but may also be useful elsewhere.
///
/// # Examples
///
/// Create SFS by repeating an element:
///
/// ```
/// use winsfs::sfs1d;
/// let sfs = sfs1d![0.1; 10];
/// assert!(sfs.iter().all(|&x| x == 0.1));
/// ```
///
/// Create SFS from a list of elements:
///
/// ```
/// use winsfs::sfs1d;
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

/// Creates a 2D SFS containing the arguments.
///
/// This is mainly intended for readability in doc-tests, but may also be useful elsewhere.
///
/// # Examples
///
/// ```
/// use winsfs::sfs2d;
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
        $crate::sfs::Sfs2d::from_vec_shape(vec, shape).unwrap()
    }};
}

macro_rules! log_sfs {
    (target: $target:expr, $level:expr, $fmt_str:literal, $sfs:expr, $sites:expr) => {
        if log::log_enabled!(target: $target, $level) {
            let fmt_sfs = $sfs
                .iter()
                .map(|v| format!("{:.6}", v * $sites as f64))
                .collect::<Vec<_>>()
                .join(" ");

            log::log!(target: $target, $level, $fmt_str, fmt_sfs);
        }
    };
}
pub(crate) use log_sfs;

/// An N-dimensional site frequency spectrum ("SFS").
///
/// Elements are stored in row-major order: the last index varies the fastest.
/// The SFS may or may not be normalised.
#[derive(Clone, Debug, PartialEq)]
pub struct Sfs<const N: usize> {
    values: Vec<f64>,
    shape: [usize; N],
}

impl<const N: usize> Sfs<N> {
    /// Creates a new SFS by repeating a single value.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::Sfs;
    /// let sfs = Sfs::from_elem(0.1, [7, 5]);
    /// assert_eq!(sfs.shape(), [7, 5]);
    /// assert!(sfs.iter().all(|&x| x == 0.1));
    /// ```
    pub fn from_elem(elem: f64, shape: [usize; N]) -> Self {
        let n = shape.iter().product();

        Self::new_unchecked(vec![elem; n], shape)
    }

    /// Creates a new SFS from an iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::Sfs;
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

    /// Creates a new SFS from a vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::Sfs;
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

    /// Returns a value at an index in the SFS.
    ///
    /// If the index is out of bounds, `None` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::sfs1d;
    /// let sfs = sfs1d![0.0, 0.1, 0.2];
    /// assert_eq!(sfs.get([0]), Some(&0.0));
    /// assert_eq!(sfs.get([1]), Some(&0.1));
    /// assert_eq!(sfs.get([2]), Some(&0.2));
    /// assert_eq!(sfs.get([3]), None);
    /// ```
    ///
    /// ```
    /// use winsfs::sfs2d;
    /// let sfs = sfs2d![[0.0, 0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]];
    /// assert_eq!(sfs.get([0, 0]), Some(&0.0));
    /// assert_eq!(sfs.get([1, 2]), Some(&0.5));
    /// assert_eq!(sfs.get([3, 0]), None);
    /// ```
    #[inline]
    pub fn get(&self, index: [usize; N]) -> Option<&f64> {
        self.values.get(compute_flat(index, self.shape))
    }

    /// Returns a mutable reference to a value at an index in the SFS.
    ///
    /// If the index is out of bounds, `None` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::sfs1d;
    /// let mut sfs = sfs1d![0.0, 0.1, 0.2];
    /// assert_eq!(sfs[[0]], 0.0);
    /// if let Some(v) = sfs.get_mut([0]) {
    ///     *v = 0.5;
    /// }
    /// assert_eq!(sfs[[0]], 0.5);
    /// ```
    ///
    /// ```
    /// use winsfs::sfs2d;
    /// let mut sfs = sfs2d![[0.0, 0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]];
    /// assert_eq!(sfs[[0, 0]], 0.0);
    /// if let Some(v) = sfs.get_mut([0, 0]) {
    ///     *v = 0.5;
    /// }
    /// assert_eq!(sfs[[0, 0]], 0.5);
    /// ```
    #[inline]
    pub fn get_mut(&mut self, index: [usize; N]) -> Option<&mut f64> {
        self.values.get_mut(compute_flat(index, self.shape))
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
    /// use winsfs::sfs1d;
    /// let sfs = sfs1d![0.0, 0.1, 0.2];
    /// assert_eq!(sfs.format_flat(" ", 1), "0.0 0.1 0.2");
    /// ```
    ///
    /// ```
    /// use winsfs::sfs2d;
    /// let  sfs = sfs2d![[0.01, 0.12], [0.23, 0.34]];
    /// assert_eq!(sfs.format_flat(",", 2), "0.01,0.12,0.23,0.34");
    /// ```
    pub fn format_flat(&self, sep: &str, precision: usize) -> String {
        if let Some(first) = self.values.first() {
            let cap = self.values.len() * (precision + 3);
            let mut init = String::with_capacity(cap);
            init.push_str(&format!("{:.precision$}", first));

            self.iter().skip(1).fold(init, |mut s, x| {
                s.push_str(sep);
                s.push_str(&format!("{x:.precision$}"));
                s
            })
        } else {
            String::new()
        }
    }

    /// Creates a new SFS.
    fn new_unchecked(values: Vec<f64>, shape: [usize; N]) -> Self {
        Self { values, shape }
    }

    /// Creates a new SFS from a string containing an SFS formatted in ANGSD format.
    pub fn parse_from_angsd(s: &str) -> Result<Self, ParseAngsdError<N>> {
        angsd::parse(s)
    }

    /// Creates a new SFS from a path containing an SFS formatted in ANGSD format.
    pub fn read_from_angsd<P>(path: P) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        let mut file = fs::File::open(path)?;
        let mut buf = String::new();
        file.read_to_string(&mut buf)?;
        Self::parse_from_angsd(&buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Creates a uniform SFS in probability space.
    pub fn uniform(shape: [usize; N]) -> Self {
        let n: usize = shape.iter().product();

        let elem = 1.0 / n as f64;

        Self::from_elem(elem, shape)
    }

    /// Creates an SFS with all entries set to zero.
    pub fn zeros(shape: [usize; N]) -> Self {
        Self::from_elem(0.0, shape)
    }

    /// Returns an iterator over the elements in the SFS in row-major order.
    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, f64> {
        self.values.iter()
    }

    /// Returns an iterator over mutable references to the elements in the SFS in row-major order.
    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, f64> {
        self.values.iter_mut()
    }

    /// Normalises the SFS to probability scale.
    #[inline]
    pub fn normalise(&mut self) {
        let sum = self.sum();

        self.iter_mut().for_each(|x| *x /= sum);
    }

    /// Re-scales the SFS by some constant.
    #[inline]
    pub fn scale(&mut self, scale: f64) {
        self.iter_mut().for_each(|x| *x *= scale)
    }

    /// Returns the SFS shape.
    pub fn shape(&self) -> [usize; N] {
        self.shape
    }

    /// Returns the sum of values in the SFS.
    #[inline]
    fn sum(&self) -> f64 {
        self.iter().sum()
    }

    /// Returns the values of the SFS as a flat, row-major slice.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Returns the a mutable reference values of the SFS as a flat, row-major slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.values
    }
}

impl<const N: usize> Add for Sfs<N> {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: Self) -> Self::Output {
        self += &rhs;
        self
    }
}

impl<const N: usize> Add<&Sfs<N>> for Sfs<N> {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: &Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<const N: usize> AddAssign for Sfs<N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

impl<const N: usize> AddAssign<&Sfs<N>> for Sfs<N> {
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        assert_eq!(self.shape, rhs.shape);

        self.iter_mut()
            .zip(rhs.iter())
            .for_each(|(x, rhs)| *x += rhs);
    }
}

impl<const N: usize> Index<[usize; N]> for Sfs<N> {
    type Output = f64;

    #[inline]
    fn index(&self, index: [usize; N]) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl<const N: usize> IndexMut<[usize; N]> for Sfs<N> {
    #[inline]
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        self.get_mut(index).unwrap()
    }
}

impl Sfs1d {
    /// Creates a new SFS from a vector.
    pub fn from_vec(values: Vec<f64>) -> Self {
        let shape = [values.len()];

        Self::new_unchecked(values, shape)
    }
}

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

fn compute_flat<const N: usize>(index: [usize; N], shape: [usize; N]) -> usize {
    let mut flat = index[0];

    for i in 1..N {
        flat *= shape[i];
        flat += index[i];
    }

    flat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_1d() {
        let sfs = Sfs1d::from_vec(vec![0., 1., 2., 3., 4., 5.]);
        let n = sfs.shape;

        for i in 0..n[0] {
            assert_eq!(sfs[[i]], i as f64);
        }

        assert_eq!(sfs.get(n), None);
    }

    #[test]
    fn test_index_2d() {
        let sfs = Sfs2d::from_vec_shape(vec![0.0, 0.1, 0.2, 1.0, 1.1, 1.2], [2, 3]).unwrap();

        assert_eq!(sfs[[0, 0]], 0.0);
        assert_eq!(sfs[[1, 1]], 1.1);
        assert_eq!(sfs[[1, 2]], 1.2);
    }
}
