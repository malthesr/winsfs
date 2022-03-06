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

#[derive(Clone, Debug, PartialEq)]
pub struct Sfs<const N: usize> {
    values: Vec<f64>,
    dim: [usize; N],
}

impl<const N: usize> Sfs<N> {
    pub fn dim(&self) -> [usize; N] {
        self.dim
    }

    pub fn from_elem(elem: f64, dim: [usize; N]) -> Self {
        let n = dim.iter().product();

        Self {
            values: vec![elem; n],
            dim,
        }
    }

    pub fn uniform(dim: [usize; N]) -> Self {
        let n: usize = dim.iter().product();

        let elem = 1.0 / n as f64;

        Self::from_elem(elem, dim)
    }

    pub fn zeros(dim: [usize; N]) -> Self {
        Self::from_elem(0.0, dim)
    }

    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, f64> {
        self.values.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, f64> {
        self.values.iter_mut()
    }

    #[inline]
    pub fn normalise(&mut self) {
        let sum = self.sum();

        self.iter_mut().for_each(|x| *x /= sum);
    }

    #[inline]
    pub fn scale(&mut self, scale: f64) {
        self.iter_mut().for_each(|x| *x *= scale)
    }

    #[inline]
    pub fn sum(&self) -> f64 {
        self.iter().sum()
    }

    #[inline]
    pub fn values_to_string(&self, precision: usize) -> String {
        self.iter()
            .map(|v| format!("{v:.precision$}"))
            .collect::<Vec<_>>()
            .join(" ")
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
        write!(f, "{}", self.values_to_string(precision))?;

        Ok(())
    }
}

impl Sfs1d {
    fn posterior_into(&self, site: &[f32], posterior: &mut Self, buf: &mut Self) {
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

    pub(crate) fn e_step(&self, sites: &[f32]) -> Self {
        let dim = self.dim;
        let n = dim[0];

        assert_eq!(sites.len() % n, 0);

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
    fn posterior_into(
        &self,
        row_site: &[f32],
        col_site: &[f32],
        posterior: &mut Self,
        buf: &mut Self,
    ) {
        let cols = col_site.len();

        let mut sum = 0.0;

        for (i, x) in row_site.iter().enumerate() {
            let sfs_row = &self.values[i * cols..];
            let buf_row = &mut buf.values[i * cols..];

            sfs_row
                .iter()
                .zip(col_site.iter())
                .zip(buf_row.iter_mut())
                .for_each(|((sfs, y), buf)| {
                    let v = sfs * (*x as f64) * (*y as f64);
                    *buf = v;
                    sum += v;
                });
        }

        buf.iter_mut().for_each(|x| *x /= sum);

        *posterior += &*buf;
    }

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
