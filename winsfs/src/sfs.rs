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

/// An N-dimensional SFS.
///
/// The SFS may or may not be normalised: that is, it may be in probability space or count space.
///
/// Elements in the SFS are stored in row-major order.
#[derive(Clone, Debug, PartialEq)]
pub struct Sfs<const N: usize> {
    values: Vec<f64>,
    shape: [usize; N],
}

impl<const N: usize> Sfs<N> {
    /// Creates a new SFS from a single element.
    pub fn from_elem(elem: f64, shape: [usize; N]) -> Self {
        let n = shape.iter().product();

        Self::new_unchecked(vec![elem; n], shape)
    }

    /// Creates a new SFS from an iterator of values and a given shape.
    pub fn from_iter_shape<I>(iter: I, shape: [usize; N]) -> Result<Self, ShapeError<N>>
    where
        I: IntoIterator<Item = f64>,
    {
        Self::from_vec_shape(iter.into_iter().collect(), shape)
    }

    /// Creates a new SFS from a vector of values and a given shape.
    pub fn from_vec_shape(vec: Vec<f64>, shape: [usize; N]) -> Result<Self, ShapeError<N>> {
        let n: usize = shape.iter().product();

        match vec.len() == n {
            true => Ok(Self::new_unchecked(vec, shape)),
            false => Err(ShapeError::new(n, shape)),
        }
    }

    /// Returns the value at the specified index in the SFS, if it exists.
    #[inline]
    pub fn get(&self, index: [usize; N]) -> Option<&f64> {
        self.values.get(compute_flat(index, self.shape))
    }

    /// Returns a mutable reference to the value at the specified index in the SFS, if it exists.
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

    /// Returns a string containing a flat represention of the SFS.
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
