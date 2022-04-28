use std::io;

use rayon::iter::ParallelIterator;

use crate::{
    io::ReadSite,
    saf::{IntoArray, JointSafView, ParSiteIterator},
    Sfs,
};

impl<const N: usize> Sfs<N>
where
    Self: Em<N>,
{
    pub fn e_step<'a, I: 'a>(&self, input: &I) -> Self
    where
        I: ParSiteIterator<'a, N>,
    {
        input
            .par_iter_sites()
            .fold(
                || (Self::zeros(self.shape()), Self::zeros(self.shape())),
                |(mut post, mut buf), site| {
                    self.posterior_into(&site.into_array(), &mut post, &mut buf);

                    (post, buf)
                },
            )
            .map(|(post, _buf)| post)
            .reduce(|| Self::zeros(self.shape()), |a, b| a + b)
    }

    pub fn e_step_io<R>(&self, reader: &mut R) -> io::Result<Self>
    where
        R: ReadSite,
    {
        let mut post = Self::zeros(self.shape());
        let mut buf = Self::zeros(self.shape());

        let mut site: [Box<[f32]>; N] = self.shape().map(|d| vec![0.0; d].into_boxed_slice());
        while reader.read_site(&mut site)?.is_not_done() {
            self.posterior_into(&site, &mut post, &mut buf);
        }

        Ok(post)
    }

    pub fn e_step_with_log_likelihood<'a, I: 'a>(&self, input: &I) -> (f64, Self)
    where
        I: ParSiteIterator<'a, N>,
    {
        input
            .par_iter_sites()
            .fold(
                || (0.0, Self::zeros(self.shape()), Self::zeros(self.shape())),
                |(mut ll, mut post, mut buf), site| {
                    ll += self
                        .posterior_into(&site.into_array(), &mut post, &mut buf)
                        .ln();

                    (ll, post, buf)
                },
            )
            .map(|(ll, post, _buf)| (ll, post))
            .reduce(
                || (0.0, Self::zeros(self.shape())),
                |a, b| (a.0 + b.0, a.1 + b.1),
            )
    }

    pub fn e_step_with_log_likelihood_io<R>(&self, reader: &mut R) -> io::Result<(f64, Self)>
    where
        R: ReadSite,
    {
        let mut post = Self::zeros(self.shape());
        let mut buf = Self::zeros(self.shape());

        let mut site: [Box<[f32]>; N] = self.shape().map(|d| vec![0.0; d].into_boxed_slice());
        let mut ll = 0.0;
        while reader.read_site(&mut site)?.is_not_done() {
            ll += self.posterior_into(&site, &mut post, &mut buf).ln();
        }

        Ok((ll, post))
    }

    pub fn log_likelihood<'a, I: 'a>(&self, input: &I) -> f64
    where
        I: ParSiteIterator<'a, N>,
    {
        input
            .par_iter_sites()
            .fold(
                || 0.0,
                |ll, site| ll + self.site_log_likelihood(&site.into_array()),
            )
            .sum()
    }

    pub fn em_step<'a, I: 'a>(&self, input: &I) -> Self
    where
        I: ParSiteIterator<'a, N>,
    {
        let mut posterior = self.e_step(input);
        posterior.normalise();

        posterior
    }

    pub fn em_step_io<R>(&self, reader: &mut R) -> io::Result<Self>
    where
        R: ReadSite,
    {
        let mut posterior = self.e_step_io(reader)?;
        posterior.normalise();

        Ok(posterior)
    }

    pub fn em_step_with_log_likelihood<'a, I: 'a>(&self, input: &I) -> (f64, Self)
    where
        I: ParSiteIterator<'a, N>,
    {
        let (log_likelihood, mut posterior) = self.e_step_with_log_likelihood(input);
        posterior.normalise();

        (log_likelihood, posterior)
    }

    pub fn em_step_with_log_likelihood_io<R>(&self, reader: &mut R) -> io::Result<(f64, Self)>
    where
        R: ReadSite,
    {
        let (log_likelihood, mut posterior) = self.e_step_with_log_likelihood_io(reader)?;
        posterior.normalise();

        Ok((log_likelihood, posterior))
    }

    pub fn em<'a>(&self, input: &JointSafView<'a, N>, epochs: usize) -> Self
    where
        JointSafView<'a, N>: ParSiteIterator<'a, N>,
    {
        let sites = input.sites();
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

pub trait Em<const N: usize> {
    fn posterior_into<T>(&self, site: &[T; N], posterior: &mut Self, buf: &mut Self) -> f64
    where
        T: AsRef<[f32]>;

    fn site_log_likelihood<T>(&self, site: &[T; N]) -> f64
    where
        T: AsRef<[f32]>;
}

impl Em<1> for Sfs<1> {
    fn posterior_into<T>(&self, site: &[T; 1], posterior: &mut Self, buf: &mut Self) -> f64
    where
        T: AsRef<[f32]>,
    {
        let mut sum = 0.0;

        self.iter()
            .zip(site[0].as_ref().iter())
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

    fn site_log_likelihood<T>(&self, site: &[T; 1]) -> f64
    where
        T: AsRef<[f32]>,
    {
        self.iter()
            .zip(site[0].as_ref().iter())
            .map(|(&sfs, &site)| sfs * site as f64)
            .sum::<f64>()
            .ln()
    }
}

impl Em<2> for Sfs<2> {
    fn posterior_into<T>(&self, site: &[T; 2], posterior: &mut Self, buf: &mut Self) -> f64
    where
        T: AsRef<[f32]>,
    {
        let row_site = site[0].as_ref();
        let col_site = site[1].as_ref();

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

    fn site_log_likelihood<T>(&self, site: &[T; 2]) -> f64
    where
        T: AsRef<[f32]>,
    {
        let row_site = site[0].as_ref();
        let col_site = site[1].as_ref();

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

impl Em<3> for Sfs<3> {
    fn posterior_into<T>(&self, site: &[T; 3], posterior: &mut Self, buf: &mut Self) -> f64
    where
        T: AsRef<[f32]>,
    {
        let fst_site = site[0].as_ref();
        let snd_site = site[1].as_ref();
        let trd_site = site[2].as_ref();

        let [n, m, o] = self.shape();

        let mut sum = 0.0;

        for i in 0..n {
            for j in 0..m {
                for k in 0..o {
                    let v = self[[i, j, k]]
                        * fst_site[i] as f64
                        * snd_site[j] as f64
                        * trd_site[k] as f64;
                    sum += v;
                    buf[[i, j, k]] = v;
                }
            }
        }

        buf.iter_mut().for_each(|x| *x /= sum);

        *posterior += &*buf;

        sum
    }

    fn site_log_likelihood<T>(&self, site: &[T; 3]) -> f64
    where
        T: AsRef<[f32]>,
    {
        let fst_site = site[0].as_ref();
        let snd_site = site[1].as_ref();
        let trd_site = site[2].as_ref();

        let [n, m, o] = self.shape();

        let mut sum = 0.0;

        for i in 0..n {
            for j in 0..m {
                for k in 0..o {
                    sum += self[[i, j, k]]
                        * fst_site[i] as f64
                        * snd_site[j] as f64
                        * trd_site[k] as f64;
                }
            }
        }

        sum
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

        sfs.posterior_into(&[site], &mut posterior, &mut buf);

        let expected = vec![10. + 1. / 6., 20. + 1. / 3., 30. + 1. / 2.];
        assert_abs_diff_eq!(posterior.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_sfs_2d_posterior() {
        let sfs = Sfs::from_vec_shape((1..16).map(|x| x as f64).collect(), [3, 5]).unwrap();

        let row_site = &[2., 2., 2.][..];
        let col_site = &[2., 4., 6., 8., 10.][..];
        let mut posterior = Sfs::from_elem(1., sfs.shape());
        let mut buf = Sfs::zeros(sfs.shape());

        sfs.posterior_into(&[row_site, col_site], &mut posterior, &mut buf);

        #[rustfmt::skip]
        let expected = vec![
            1.002564, 1.010256, 1.023077, 1.041026, 1.064103,
            1.015385, 1.035897, 1.061538, 1.092308, 1.128205,
            1.028205, 1.061538, 1.100000, 1.143590, 1.192308,
        ];
        assert_abs_diff_eq!(posterior.as_slice(), expected.as_slice(), epsilon = 1e-6);
    }

    #[test]
    fn test_sfs_3d_posterior() {
        let sfs = Sfs::from_vec_shape((1..28).map(|x| x as f64).collect(), [3, 3, 3]).unwrap();

        let fst_site = &[1., 2., 3.][..];
        let snd_site = &[4., 5., 6.][..];
        let trd_site = &[7., 8., 9.][..];
        let mut posterior = Sfs::from_elem(1., sfs.shape());
        let mut buf = Sfs::zeros(sfs.shape());

        sfs.posterior_into(&[fst_site, snd_site, trd_site], &mut posterior, &mut buf);

        #[rustfmt::skip]
        let expected = vec![
            1.000741, 1.001694, 1.002859,
            1.003707, 1.005296, 1.007149,
            1.007785, 1.010168, 1.012869,

            1.014828, 1.018642, 1.022878,
            1.024097, 1.029657, 1.035748,
            1.035589, 1.043215, 1.051477,

            1.042262, 1.050842, 1.0600571,
            1.061169, 1.073085, 1.0857959,
            1.083412, 1.099142, 1.1158245,
        ];
        assert_abs_diff_eq!(posterior.as_slice(), expected.as_slice(), epsilon = 1e-6);
    }
}
