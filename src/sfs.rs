use std::{
    fmt,
    ops::{Add, AddAssign},
};

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sfs1d<const N: usize>([f64; N]);

impl<const N: usize> Sfs1d<N> {
    pub fn posterior_into(&self, site: &[f32; N], posterior: &mut Self, buf: &mut Self) {
        self.iter()
            .zip(site.iter())
            .zip(buf.iter_mut())
            .for_each(|((&sfs, &saf), buf)| *buf = sfs * saf as f64);

        buf.normalise();

        *posterior += *buf;
    }

    pub fn e_step(&self, sites: &[[f32; N]]) -> Self {
        sites
            .into_par_iter()
            .fold(
                || (Self::zero(), Self::zero()),
                |(mut posterior, mut buf), site| {
                    self.posterior_into(site, &mut posterior, &mut buf);

                    (posterior, buf)
                },
            )
            .map(|(posterior, _buf)| posterior)
            .reduce(Self::zero, |a, b| a + b)
    }
}

impl<const N: usize> AsRef<[f64]> for Sfs1d<N> {
    #[inline]
    fn as_ref(&self) -> &[f64] {
        self.0.as_slice()
    }
}

impl<const N: usize> AsMut<[f64]> for Sfs1d<N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [f64] {
        self.0.as_mut_slice()
    }
}

impl<const N: usize> Add for Sfs1d<N> {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<const N: usize> AddAssign for Sfs1d<N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.iter_mut()
            .zip(rhs.iter())
            .for_each(|(post, &buf)| *post += buf);
    }
}

impl<const N: usize> fmt::Display for Sfs1d<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "# Dimensions: {N}")?;

        let precision = f.precision().unwrap_or(6);
        write!(f, "{}", self.values_to_string(precision))?;

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sfs2d<const R: usize, const C: usize>([[f64; C]; R]);

impl<const R: usize, const C: usize> Sfs2d<R, C> {
    pub fn posterior_into(
        &self,
        row_site: &[f32; R],
        col_site: &[f32; C],
        posterior: &mut Self,
        buf: &mut Self,
    ) {
        #[allow(clippy::needless_range_loop)]
        for r in 0..R {
            for c in 0..C {
                buf.0[r][c] = self.0[r][c] * row_site[r] as f64 * col_site[c] as f64;
            }
        }

        buf.normalise();

        *posterior += *buf;
    }

    pub fn e_step(&self, row_sites: &[[f32; R]], col_sites: &[[f32; C]]) -> Self {
        row_sites
            .into_par_iter()
            .zip(col_sites.into_par_iter())
            .fold(
                || (Self::zero(), Self::zero()),
                |(mut posterior, mut buf), (row_site, col_site)| {
                    self.posterior_into(row_site, col_site, &mut posterior, &mut buf);

                    (posterior, buf)
                },
            )
            .map(|(posterior, _buf)| posterior)
            .reduce(Self::zero, |a, b| a + b)
    }
}

impl<const R: usize, const C: usize> AsRef<[f64]> for Sfs2d<R, C> {
    #[inline]
    fn as_ref(&self) -> &[f64] {
        bytemuck::cast_slice(&self.0)
    }
}

impl<const R: usize, const C: usize> AsMut<[f64]> for Sfs2d<R, C> {
    #[inline]
    fn as_mut(&mut self) -> &mut [f64] {
        bytemuck::cast_slice_mut(&mut self.0)
    }
}

impl<const R: usize, const C: usize> Add for Sfs2d<R, C> {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<const R: usize, const C: usize> AddAssign for Sfs2d<R, C> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.iter_mut()
            .zip(rhs.iter())
            .for_each(|(post, &buf)| *post += buf);
    }
}

impl<const R: usize, const C: usize> fmt::Display for Sfs2d<R, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "# Dimensions: {R}/{C}")?;

        let precision = f.precision().unwrap_or(6);
        write!(f, "{}", self.values_to_string(precision))?;

        Ok(())
    }
}

pub trait Sfs:
    Add<Output = Self>
    + AddAssign
    + AsRef<[f64]>
    + AsMut<[f64]>
    + Copy
    + fmt::Debug
    + fmt::Display
    + Sized
{
    fn uniform() -> Self;

    fn zero() -> Self;

    #[inline]
    fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.as_ref().iter()
    }

    #[inline]
    fn iter_mut(&mut self) -> std::slice::IterMut<'_, f64> {
        self.as_mut().iter_mut()
    }

    #[inline]
    fn normalise(&mut self) {
        let sum = self.sum();

        self.iter_mut().for_each(|x| *x /= sum);
    }

    #[inline]
    fn scale(&mut self, scale: f64) {
        self.iter_mut().for_each(|x| *x *= scale)
    }

    #[inline]
    fn sum(&self) -> f64 {
        self.iter().sum()
    }

    #[inline]
    fn values_to_string(&self, precision: usize) -> String {
        self.iter()
            .map(|v| format!("{v:.precision$}"))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl<const N: usize> Sfs for Sfs1d<N> {
    fn uniform() -> Self {
        let value = 1.0 / N as f64;

        Self([value; N])
    }

    fn zero() -> Self {
        Self([0.0; N])
    }
}

impl<const R: usize, const C: usize> Sfs for Sfs2d<R, C> {
    fn uniform() -> Self {
        let value = 1.0 / (C * R) as f64;

        Self([[value; C]; R])
    }

    fn zero() -> Self {
        Self([[0.0; C]; R])
    }
}
