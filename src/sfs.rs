use std::{
    fmt,
    ops::{Add, AddAssign},
    slice,
};

use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};

pub type Sfs1d = Sfs<1>;
pub type Sfs2d = Sfs<2>;

/// An N-dimensional SFS.
///
/// The SFS may or may not be normalised: that is, it may be in probability space or count space.
///
/// Elements in the SFS are stored in row-major order.
#[derive(Clone, Debug, PartialEq)]
pub struct Sfs<const N: usize> {
    values: Vec<f64>,
    dim: [usize; N],
}

impl<const N: usize> Sfs<N> {
    /// Returns the dimensions of the SFS.
    pub fn dim(&self) -> [usize; N] {
        self.dim
    }

    /// Creates a new SFS from a single element.
    pub fn from_elem(elem: f64, dim: [usize; N]) -> Self {
        let n = dim.iter().product();

        Self {
            values: vec![elem; n],
            dim,
        }
    }

    /// Creates a uniform SFS in probability space.
    pub fn uniform(dim: [usize; N]) -> Self {
        let n: usize = dim.iter().product();

        let elem = 1.0 / n as f64;

        Self::from_elem(elem, dim)
    }

    /// Creates an SFS with all entries set to zero.
    pub fn zeros(dim: [usize; N]) -> Self {
        Self::from_elem(0.0, dim)
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
        assert_eq!(self.dim, rhs.dim);

        self.iter_mut()
            .zip(rhs.iter())
            .for_each(|(x, rhs)| *x += rhs);
    }
}

impl<const N: usize> fmt::Display for Sfs<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fmt_dim = self.dim.map(|x| x.to_string()).join("/");
        writeln!(f, "# Dimensions: {fmt_dim}")?;

        let precision = f.precision().unwrap_or(6);
        let fmt_sfs = self
            .iter()
            .map(|v| format!("{v:.precision$}", precision = precision))
            .collect::<Vec<_>>()
            .join(" ");
        write!(f, "{}", fmt_sfs)?;

        Ok(())
    }
}

impl Sfs1d {
    /// Calculates the posterior count probabilities given `self` of the SAF site `site`,
    /// and adds the posterior probability to `posterior`.
    ///
    /// The posterior probability is proportional to the element-wise product of the SFS
    /// and the SAF site. In order to normalise, an intermediate buffer `buf` is required to
    /// avoid having to allocate. The buffer will be overwritten.
    ///
    /// `self`, `site`, `posterior`, and `buf` should all be of the same dimensionality.
    fn posterior_into(&self, site: &[f32], posterior: &mut Self, buf: &mut Self) {
        debug_assert_eq!(self.dim[0], site.len());

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
    }

    /// Calculates the sum posterior count probabilities given `self` of the SAF sites in `sites`.
    ///
    /// The `sites` will be chunked according to the dimensionality of self, i.e. `sites.len()`
    /// should be some multiple of `self.dim()[0]`.
    pub(crate) fn e_step(&self, sites: &[f32]) -> Self {
        let dim = self.dim;
        let n = dim[0];

        debug_assert_eq!(sites.len() % n, 0);

        sites
            .par_chunks(n)
            .fold(
                || (Self::zeros(dim), Self::zeros(dim)),
                |(mut posterior, mut buf), site| {
                    self.posterior_into(site, &mut posterior, &mut buf);

                    (posterior, buf)
                },
            )
            .map(|(posterior, _buf)| posterior)
            .reduce(|| Self::zeros(dim), |a, b| a + b)
    }
}

impl Sfs2d {
    /// Calculates the posterior count probabilities given `self` of the SAF sites
    /// `row_site` and `col_site`
    ///
    /// The posterior probability is proportional to the element-wise product of the SFS
    /// with the matrix product of `row_site` and the transpose of `col_site`. In order to
    /// normalise, an intermediate buffer `buf` is required to avoid having to allocate.
    /// The buffer will be overwritten.
    ///
    /// `self`, `posterior`, and `buf` should all be of the same dimensionality, while
    /// `row_site.len()` should match the number of rows of the SFS, and `col_site.len()`
    /// should match the number of columns of the SFS.
    fn posterior_into(
        &self,
        row_site: &[f32],
        col_site: &[f32],
        posterior: &mut Self,
        buf: &mut Self,
    ) {
        debug_assert_eq!(self.dim[0], row_site.len());
        debug_assert_eq!(self.dim[1], col_site.len());

        let sum = matmul(&mut buf.values, &self.values, row_site, col_site);

        buf.iter_mut().for_each(|x| *x /= sum);

        *posterior += &*buf;
    }

    /// Calculates the sum posterior count probabilities given `self` of the SAF sites in `sites`.
    ///
    /// The `row_sites` and `col_sites` will be chunked according to the dimensionality of self,
    /// i.e. `row_sites.len()` should be some multiple of `self.dim()[0]` and `col_sites.len()`
    /// should be some multiple of `self.dim()[1]`.
    pub(crate) fn e_step(&self, row_sites: &[f32], col_sites: &[f32]) -> Self {
        let dim = self.dim;
        let [rows, cols] = dim;

        row_sites
            .par_chunks(rows)
            .zip(col_sites.par_chunks(cols))
            .fold(
                || (Self::zeros(dim), Self::zeros(dim)),
                |(mut posterior, mut buf), (row_site, col_site)| {
                    self.posterior_into(row_site, col_site, &mut posterior, &mut buf);

                    (posterior, buf)
                },
            )
            .map(|(posterior, _buf)| posterior)
            .reduce(|| Self::zeros(dim), |a, b| a + b)
    }
}

// Computes matrix product `into = with * (a * b^T)` and returns the sum of `into`
#[inline]
fn matmul(into: &mut [f64], with: &[f64], a: &[f32], b: &[f32]) -> f64 {
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

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sfs_1d_posterior() {
        let sfs = Sfs {
            values: vec![1., 2., 3.],
            dim: [3],
        };

        let site = &[2., 2., 2.];
        let mut posterior = Sfs {
            values: vec![10., 20., 30.],
            dim: sfs.dim(),
        };
        let mut buf = Sfs::zeros(sfs.dim());

        sfs.posterior_into(site, &mut posterior, &mut buf);

        let expected = vec![10. + 1. / 6., 20. + 1. / 3., 30. + 1. / 2.];
        assert_abs_diff_eq!(posterior.values.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_sfs_2d_posterior() {
        let sfs = Sfs {
            values: (1..16).map(|x| x as f64).collect(),
            dim: [3, 5],
        };

        let row_site = &[2., 2., 2.];
        let col_site = &[2., 4., 6., 8., 10.];
        let mut posterior = Sfs::from_elem(1., sfs.dim());
        let mut buf = Sfs::zeros(sfs.dim());

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
