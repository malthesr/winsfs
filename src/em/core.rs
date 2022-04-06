use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};

use crate::{Sfs, Sfs1d, Sfs2d};

impl<const N: usize> Sfs<N>
where
    for<'a> Self: Em<'a, N>,
{
    pub fn e_step(&self, input: &<Self as Em<N>>::Input) -> Self {
        input
            .iter_sites(self.shape())
            .fold(
                || (Self::zeros(self.shape()), Self::zeros(self.shape())),
                |(mut post, mut buf), site| {
                    self.posterior_into(site, &mut post, &mut buf);

                    (post, buf)
                },
            )
            .map(|(post, _buf)| post)
            .reduce(|| Self::zeros(self.shape()), |a, b| a + b)
    }

    pub fn e_step_with_log_likelihood(&self, input: &<Self as Em<N>>::Input) -> (f64, Self) {
        input
            .iter_sites(self.shape())
            .fold(
                || (0.0, Self::zeros(self.shape()), Self::zeros(self.shape())),
                |(mut ll, mut post, mut buf), site| {
                    ll += self.posterior_into(site, &mut post, &mut buf).ln();

                    (ll, post, buf)
                },
            )
            .map(|(ll, post, _buf)| (ll, post))
            .reduce(
                || (0.0, Self::zeros(self.shape())),
                |a, b| (a.0 + b.0, a.1 + b.1),
            )
    }

    pub fn log_likelihood(&self, input: &<Self as Em<N>>::Input) -> f64 {
        input
            .iter_sites(self.shape())
            .fold(|| 0.0, |ll, site| ll + self.site_log_likelihood(site))
            .sum()
    }

    pub fn em_step(&self, input: &<Self as Em<N>>::Input) -> Self {
        let mut posterior = self.e_step(input);
        posterior.normalise();

        posterior
    }

    pub fn em_step_with_log_likelihood(&self, input: &<Self as Em<N>>::Input) -> (f64, Self) {
        let (log_likelihood, mut posterior) = self.e_step_with_log_likelihood(input);
        posterior.normalise();

        (log_likelihood, posterior)
    }

    pub fn em(&self, input: &<Self as Em<N>>::Input, epochs: usize) -> Self {
        let sites = input.sites(self.shape());
        let mut sfs = self.clone();

        for i in 0..epochs {
            log_sfs!(
                target: "em",
                log::Level::Debug,
                "Epoch {i}, current SFS: {}",
                sfs, sites
            );

            sfs = sfs.em_step(input);
        }

        sfs
    }
}

pub trait Em<'a, const N: usize> {
    // TODO: The trait lifetime should really be a GAT on this type once stable,
    // see github.com/rust-lang/rust/issues/44265
    type Input: Input<N>;

    fn posterior_into(&self, site: Self::Input, posterior: &mut Self, buf: &mut Self) -> f64;

    fn site_log_likelihood(&self, site: Self::Input) -> f64;
}

impl<'a> Em<'a, 1> for Sfs1d {
    type Input = &'a [f32];

    fn posterior_into(&self, site: Self::Input, posterior: &mut Self, buf: &mut Self) -> f64 {
        let mut sum = 0.0;

        self.iter()
            .zip(site.iter())
            .zip(buf.iter_mut())
            .for_each(|((&sfs, &site), buf)| {
                let v = sfs * site as f64;
                *buf = v;
                sum += v;
            });

        buf.iter_mut().for_each(|x| *x /= sum);

        *posterior += &*buf;

        sum
    }

    fn site_log_likelihood(&self, site: Self::Input) -> f64 {
        self.iter()
            .zip(site.iter())
            .map(|(&sfs, &site)| sfs * site as f64)
            .sum::<f64>()
            .ln()
    }
}

impl<'a> Em<'a, 2> for Sfs2d {
    type Input = (&'a [f32], &'a [f32]);

    fn posterior_into(&self, site: Self::Input, posterior: &mut Self, buf: &mut Self) -> f64 {
        let (row_site, col_site) = site;

        let cols = col_site.len();

        let mut sum = 0.0;

        for (i, x) in row_site.iter().enumerate() {
            // Get the slice starting with the appropriate row.
            // These are zipped onto the `col_site` below,
            // so it is fine that they run past the row.
            let sfs_row = &self.as_slice()[i * cols..];
            let buf_row = &mut buf.as_mut_slice()[i * cols..];

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

        sum
    }

    fn site_log_likelihood(&self, site: Self::Input) -> f64 {
        let (row_site, col_site) = site;

        let mut sum = 0.0;

        for (i, x) in row_site.iter().enumerate() {
            // Get the slice starting with the appropriate row.
            // These are zipped onto the `col_site` below,
            // so it is fine that they run past the row.
            let sfs_row = &self.as_slice()[i * col_site.len()..];

            sfs_row.iter().zip(col_site.iter()).for_each(|(w, y)| {
                sum += w * (*x as f64) * (*y as f64);
            });
        }

        sum.ln()
    }
}

pub trait Input<const N: usize> {
    // TODO: These types are long, and should be replaced by TAIT once stable,
    // see github.com/rust-lang/rust/issues/63063
    type SiteIter: IndexedParallelIterator<Item = Self>;
    type BlockIter: ExactSizeIterator<Item = Self>;

    fn sites(&self, shape: [usize; N]) -> usize;

    fn iter_sites(&self, shape: [usize; N]) -> Self::SiteIter;

    fn iter_blocks(&self, shape: [usize; N], block_size: usize) -> Self::BlockIter;
}

impl<'a> Input<1> for &'a [f32] {
    type SiteIter = rayon::slice::Chunks<'a, f32>;
    type BlockIter = std::slice::Chunks<'a, f32>;

    fn sites(&self, shape: [usize; 1]) -> usize {
        assert_eq!(self.len() % shape[0], 0);
        self.len() / shape[0]
    }

    fn iter_sites(&self, shape: [usize; 1]) -> Self::SiteIter {
        assert_eq!(self.len() % shape[0], 0);
        self.par_chunks(shape[0])
    }

    fn iter_blocks(&self, shape: [usize; 1], block_size: usize) -> Self::BlockIter {
        assert_eq!(self.len() % shape[0], 0);
        self.chunks(shape[0] * block_size)
    }
}

impl<'a> Input<2> for (&'a [f32], &'a [f32]) {
    type SiteIter = rayon::iter::Zip<rayon::slice::Chunks<'a, f32>, rayon::slice::Chunks<'a, f32>>;
    type BlockIter = std::iter::Zip<std::slice::Chunks<'a, f32>, std::slice::Chunks<'a, f32>>;

    fn sites(&self, shape: [usize; 2]) -> usize {
        let [n, m] = shape;
        assert_eq!(self.0.len() % n, 0);
        assert_eq!(self.1.len() % m, 0);
        assert_eq!(self.0.len() / n, self.1.len() / m);
        self.0.len() / n
    }

    fn iter_sites(&self, shape: [usize; 2]) -> Self::SiteIter {
        let [n, m] = shape;
        assert_eq!(self.0.len() % n, 0);
        assert_eq!(self.1.len() % m, 0);
        self.0.par_chunks(n).zip(self.1.par_chunks(m))
    }

    fn iter_blocks(&self, shape: [usize; 2], block_size: usize) -> Self::BlockIter {
        let [n, m] = shape;
        assert_eq!(self.0.len() % n, 0);
        assert_eq!(self.1.len() % m, 0);
        self.0
            .chunks(n * block_size)
            .zip(self.1.chunks(m * block_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sfs_1d_posterior() {
        let sfs = Sfs::from_vec_shape(vec![1., 2., 3.], [3]).unwrap();

        let site = &[2., 2., 2.];
        let mut posterior = Sfs::from_vec_shape(vec![10., 20., 30.], sfs.shape()).unwrap();
        let mut buf = Sfs::zeros(sfs.shape());

        sfs.posterior_into(site, &mut posterior, &mut buf);

        let expected = vec![10. + 1. / 6., 20. + 1. / 3., 30. + 1. / 2.];
        assert_abs_diff_eq!(posterior.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_sfs_2d_posterior() {
        let sfs = Sfs::from_vec_shape((1..16).map(|x| x as f64).collect(), [3, 5]).unwrap();

        let row_site = &[2., 2., 2.];
        let col_site = &[2., 4., 6., 8., 10.];
        let mut posterior = Sfs::from_elem(1., sfs.shape());
        let mut buf = Sfs::zeros(sfs.shape());

        sfs.posterior_into((row_site, col_site), &mut posterior, &mut buf);

        #[rustfmt::skip]
        let expected = vec![
            1.002564, 1.010256, 1.023077, 1.041026, 1.064103,
            1.015385, 1.035897, 1.061538, 1.092308, 1.128205,
            1.028205, 1.061538, 1.100000, 1.143590, 1.192308,
        ];
        assert_abs_diff_eq!(posterior.as_slice(), expected.as_slice(), epsilon = 1e-6);
    }
}
