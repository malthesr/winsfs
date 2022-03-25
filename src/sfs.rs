use std::{
    error::Error,
    fmt, fs,
    io::{self, Read},
    ops::{Add, AddAssign, Index, IndexMut},
    path::Path,
    slice,
};

use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSlice,
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
    /// Calculates the likelihood and posterior count probabilities given `self` of the
    /// SAF site `site`, adds the posterior probability to `posterior`, and returns the likelihood.
    ///
    /// The posterior probability is proportional to the element-wise product of the SFS
    /// and the SAF site. In order to normalise, an intermediate buffer `buf` is required to
    /// avoid having to allocate. The buffer will be overwritten.
    ///
    /// `self`, `site`, `posterior`, and `buf` should all be of the same shape.
    fn posterior_into(&self, site: &[f32], posterior: &mut Self, buf: &mut Self) -> f64 {
        debug_assert_eq!(self.shape[0], site.len());

        let mut sum = 0.0;

        self.iter()
            .zip(site.iter())
            .zip(buf.iter_mut())
            .for_each(|((&sfs, &saf), buf)| {
                let v = sfs * saf as f64;
                *buf = v;
                sum += v;
            });

        buf.iter_mut().for_each(|x| *x /= sum);

        *posterior += &*buf;

        sum
    }

    /// Calculates the sum posterior count probabilities given `self` of the SAF sites in `sites`.
    ///
    /// The `sites` will be chunked according to the shape of self, i.e. `sites.len()`
    /// should be some multiple of `self.shape()[0]`.
    pub(crate) fn e_step(&self, sites: &[f32]) -> Self {
        self.fold_with_sites(
            sites,
            || (Self::zeros(self.shape), Self::zeros(self.shape)),
            |(mut post, mut buf), site| {
                self.posterior_into(site, &mut post, &mut buf);

                (post, buf)
            },
        )
        .map(|(post, _buf)| post)
        .reduce(|| Self::zeros(self.shape), |a, b| a + b)
    }

    /// Calculates the sum posterior count probabilities given `self` of the SAF sites in `sites`.
    ///
    /// The `sites` will be chunked according to the shape of self, i.e. `sites.len()`
    /// should be some multiple of `self.shape()[0]`.
    pub(crate) fn e_step_with_log_likelihood(&self, sites: &[f32]) -> (f64, Self) {
        self.fold_with_sites(
            sites,
            || (0.0, Self::zeros(self.shape), Self::zeros(self.shape)),
            |(mut ll, mut post, mut buf), site| {
                ll += self.posterior_into(site, &mut post, &mut buf).ln();

                (ll, post, buf)
            },
        )
        .map(|(ll, post, _buf)| (ll, post))
        .reduce(
            || (0.0, Self::zeros(self.shape)),
            |a, b| (a.0 + b.0, a.1 + b.1),
        )
    }

    /// Calculates the log-likelihood given `self` of the SAF sites in `sites`.
    ///
    /// The `sites` will be chunked according to the shape of self, i.e. `sites.len()`
    /// should be some multiple of `self.shape()[0]`.
    pub(crate) fn log_likelihood(&self, sites: &[f32]) -> f64 {
        self.fold_with_sites(
            sites,
            || 0.0,
            |ll, site| ll + self.site_log_likelihood(site),
        )
        .sum()
    }

    /// Calculates the log-likelihood given `self` of the SAF site `site`.
    ///
    /// `self` and`site` should be of the same shape.
    fn site_log_likelihood(&self, site: &[f32]) -> f64 {
        debug_assert_eq!(self.shape[0], site.len());

        self.iter()
            .zip(site.iter())
            .map(|(&sfs, &saf)| sfs * saf as f64)
            .sum::<f64>()
            .ln()
    }

    /// Helper to set up a fold over sites chunked appropriately.
    fn fold_with_sites<'a, T, F: 'a, G: 'a>(
        &self,
        sites: &'a [f32],
        init: G,
        fold: F,
    ) -> impl ParallelIterator<Item = T> + 'a
    where
        T: Send,
        F: Fn(T, &'a [f32]) -> T + Sync + Send,
        G: Fn() -> T + Sync + Send,
    {
        let n = self.shape[0];
        debug_assert_eq!(sites.len() % n, 0);

        sites.par_chunks(n).fold(init, fold)
    }
}

impl Sfs2d {
    /// Calculates the likelihood and posterior count probabilities given `self` of the
    /// SAF sites `row_site` and `col_site`, adds the posterior probability to `posterior`,
    /// and returns the likelihood.
    ///
    /// The posterior probability is proportional to the element-wise product of the SFS
    /// with the matrix product of `row_site` and the transpose of `col_site`. In order to
    /// normalise, an intermediate buffer `buf` is required to avoid having to allocate.
    /// The buffer will be overwritten.
    ///
    /// `self`, `posterior`, and `buf` should all be of the same shape, while
    /// `row_site.len()` should match the number of rows of the SFS, and `col_site.len()`
    /// should match the number of columns of the SFS.
    fn posterior_into(
        &self,
        row_site: &[f32],
        col_site: &[f32],
        posterior: &mut Self,
        buf: &mut Self,
    ) -> f64 {
        debug_assert_eq!(self.shape[0], row_site.len());
        debug_assert_eq!(self.shape[1], col_site.len());

        let sum = matmul_into(&mut buf.values, &self.values, row_site, col_site);

        buf.iter_mut().for_each(|x| *x /= sum);

        *posterior += &*buf;

        sum
    }

    /// Calculates the sum posterior count probabilities given `self` of the SAF sites in `sites`.
    ///
    /// The `row_sites` and `col_sites` will be chunked according to the shape of self,
    /// i.e. `row_sites.len()` should be some multiple of `self.shape()[0]` and `col_sites.len()`
    /// should be some multiple of `self.shape()[1]`.
    pub(crate) fn e_step(&self, row_sites: &[f32], col_sites: &[f32]) -> Self {
        self.fold_with_sites(
            row_sites,
            col_sites,
            || (Self::zeros(self.shape), Self::zeros(self.shape)),
            |(mut posterior, mut buf), (row_site, col_site)| {
                self.posterior_into(row_site, col_site, &mut posterior, &mut buf);

                (posterior, buf)
            },
        )
        .map(|(posterior, _buf)| posterior)
        .reduce(|| Self::zeros(self.shape), |a, b| a + b)
    }

    /// Calculates the sum posterior count probabilities given `self` of the SAF sites in `sites`.
    ///
    /// The `row_sites` and `col_sites` will be chunked according to the shape of self,
    /// i.e. `row_sites.len()` should be some multiple of `self.shape()[0]` and `col_sites.len()`
    /// should be some multiple of `self.shape()[1]`.
    pub(crate) fn e_step_with_log_likelihood(
        &self,
        row_sites: &[f32],
        col_sites: &[f32],
    ) -> (f64, Self) {
        self.fold_with_sites(
            row_sites,
            col_sites,
            || (0.0, Self::zeros(self.shape), Self::zeros(self.shape)),
            |(mut ll, mut post, mut buf), (row_site, col_site)| {
                ll += self
                    .posterior_into(row_site, col_site, &mut post, &mut buf)
                    .ln();

                (ll, post, buf)
            },
        )
        .map(|(ll, post, _buf)| (ll, post))
        .reduce(
            || (0.0, Self::zeros(self.shape)),
            |a, b| (a.0 + b.0, a.1 + b.1),
        )
    }

    /// Calculates the log-likelihood given `self` of the SAF sites in `sites`.
    ///
    /// The `row_sites` and `col_sites` will be chunked according to the shape of self,
    /// i.e. `row_sites.len()` should be some multiple of `self.shape()[0]` and `col_sites.len()`
    /// should be some multiple of `self.shape()[1]`.
    pub(crate) fn log_likelihood(&self, row_sites: &[f32], col_sites: &[f32]) -> f64 {
        self.fold_with_sites(
            row_sites,
            col_sites,
            || 0.0,
            |log_likelihood, (row_site, col_site)| {
                log_likelihood + self.site_log_likelihood(row_site, col_site)
            },
        )
        .sum()
    }

    /// Calculates the log-likelihood given `self` of the  SAF sites `row_site` and `col_site`
    ///
    /// `row_site.len()` should match the number of rows of the SFS, and `col_site.len()`
    /// should match the number of columns of the SFS.
    fn site_log_likelihood(&self, row_site: &[f32], col_site: &[f32]) -> f64 {
        debug_assert_eq!(self.shape[0], row_site.len());
        debug_assert_eq!(self.shape[1], col_site.len());

        matmul_sum(&self.values, row_site, col_site).ln()
    }

    /// Helper to set up a fold over sites chunked appropriately.
    fn fold_with_sites<'a, T, F: 'a, G: 'a>(
        &self,
        row_sites: &'a [f32],
        col_sites: &'a [f32],
        init: G,
        fold: F,
    ) -> impl ParallelIterator<Item = T> + 'a
    where
        T: Send,
        F: Fn(T, (&'a [f32], &'a [f32])) -> T + Sync + Send,
        G: Fn() -> T + Sync + Send,
    {
        let [rows, cols] = self.shape;
        debug_assert_eq!(row_sites.len() % rows, 0);
        debug_assert_eq!(col_sites.len() % cols, 0);

        row_sites
            .par_chunks(rows)
            .zip(col_sites.par_chunks(cols))
            .fold(init, fold)
    }
}

// Computes the sum of the matrix product `with * (a * b^t)`.
#[inline]
fn matmul_sum(with: &[f64], a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0;

    for (i, x) in a.iter().enumerate() {
        // Get the slice starting with the appropriate row.
        // These are zipped onto the `b` below,
        // so it is fine that they run past the row.
        let with_row = &with[i * b.len()..];

        with_row.iter().zip(b.iter()).for_each(|(w, y)| {
            sum += w * (*x as f64) * (*y as f64);
        });
    }

    sum
}

// Computes matrix product `into = with * (a * b^T)` and returns the sum of `into`
#[inline]
fn matmul_into(into: &mut [f64], with: &[f64], a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0;

    for (i, x) in a.iter().enumerate() {
        // Get the slice starting with the appropriate row.
        // These are zipped onto the `b` below,
        // so it is fine that they run past the row.
        let into_row = &mut into[i * b.len()..];
        let with_row = &with[i * b.len()..];

        into_row
            .iter_mut()
            .zip(with_row.iter())
            .zip(b.iter())
            .for_each(|((i, w), y)| {
                let v = w * (*x as f64) * (*y as f64);
                *i = v;
                sum += v;
            });
    }

    sum
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

    use approx::assert_abs_diff_eq;

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
    #[test]
    fn test_sfs_1d_posterior() {
        let sfs = Sfs {
            values: vec![1., 2., 3.],
            shape: [3],
        };

        let site = &[2., 2., 2.];
        let mut posterior = Sfs {
            values: vec![10., 20., 30.],
            shape: sfs.shape(),
        };
        let mut buf = Sfs::zeros(sfs.shape());

        sfs.posterior_into(site, &mut posterior, &mut buf);

        let expected = vec![10. + 1. / 6., 20. + 1. / 3., 30. + 1. / 2.];
        assert_abs_diff_eq!(posterior.values.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_sfs_2d_posterior() {
        let sfs = Sfs {
            values: (1..16).map(|x| x as f64).collect(),
            shape: [3, 5],
        };

        let row_site = &[2., 2., 2.];
        let col_site = &[2., 4., 6., 8., 10.];
        let mut posterior = Sfs::from_elem(1., sfs.shape());
        let mut buf = Sfs::zeros(sfs.shape());

        sfs.posterior_into(row_site, col_site, &mut posterior, &mut buf);

        #[rustfmt::skip]
        let expected = vec![
            1.002564, 1.010256, 1.023077, 1.041026, 1.064103,
            1.015385, 1.035897, 1.061538, 1.092308, 1.128205,
            1.028205, 1.061538, 1.100000, 1.143590, 1.192308,
        ];
        assert_abs_diff_eq!(
            posterior.values.as_slice(),
            expected.as_slice(),
            epsilon = 1e-6
        );
    }
}
