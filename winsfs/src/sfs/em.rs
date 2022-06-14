use std::io;

use rayon::iter::ParallelIterator;

use crate::{
    saf::{IntoArray, ParSiteIterator},
    stream::ReadSite,
    ArrayExt, Sfs, UnnormalisedSfs,
};

impl<const N: usize> Sfs<N> {
    pub fn e_step<'a, I: 'a>(&self, input: &I) -> (f64, UnnormalisedSfs<N>)
    where
        I: ParSiteIterator<'a, N>,
    {
        input
            .par_iter_sites()
            .fold(
                || (0.0, Sfs::zeros(self.shape()), Sfs::zeros(self.shape())),
                |(mut ll, mut post, mut buf), site| {
                    ll += self
                        .posterior_into(&site.into_array(), &mut post, &mut buf)
                        .ln();

                    (ll, post, buf)
                },
            )
            .map(|(ll, post, _buf)| (ll, post))
            .reduce(
                || (0.0, Sfs::zeros(self.shape())),
                |a, b| (a.0 + b.0, a.1 + b.1),
            )
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

    pub fn em_step<'a, I: 'a>(&self, input: &I) -> (f64, Self)
    where
        I: ParSiteIterator<'a, N>,
    {
        let (log_likelihood, posterior) = self.e_step(input);

        (log_likelihood, posterior.normalise())
    }

    pub fn streaming_e_step<R>(&self, reader: &mut R) -> io::Result<(f64, UnnormalisedSfs<N>)>
    where
        R: ReadSite,
    {
        let mut post = Sfs::zeros(self.shape());
        let mut buf = Sfs::zeros(self.shape());

        let mut site: [Box<[f32]>; N] = self.shape().map(|d| vec![0.0; d].into_boxed_slice());
        let mut ll = 0.0;
        while reader.read_site(&mut site)?.is_not_done() {
            ll += self.posterior_into(&site, &mut post, &mut buf).ln();
        }

        Ok((ll, post))
    }

    pub fn streaming_em_step<R>(&self, reader: &mut R) -> io::Result<(f64, Self)>
    where
        R: ReadSite,
    {
        let (log_likelihood, posterior) = self.streaming_e_step(reader)?;

        Ok((log_likelihood, posterior.normalise()))
    }

    pub fn posterior_into<T>(
        &self,
        site: &[T; N],
        posterior: &mut UnnormalisedSfs<N>,
        buf: &mut UnnormalisedSfs<N>,
    ) -> f64
    where
        T: AsRef<[f32]>,
    {
        let mut sum = 0.;

        posterior_inner(
            self.as_slice(),
            self.strides.as_slice(),
            site.each_ref().map(|x| x.as_ref()).as_slice(),
            buf.as_mut_slice(),
            &mut sum,
            1.,
        );

        buf.iter_mut()
            .zip(posterior.iter_mut())
            .for_each(|(buf, posterior)| {
                *buf /= sum;
                *posterior += *buf;
            });

        sum
    }

    pub fn site_log_likelihood<T>(&self, site: &[T; N]) -> f64
    where
        T: AsRef<[f32]>,
    {
        self.site_likelihood(site).ln()
    }

    fn site_likelihood<T>(&self, site: &[T; N]) -> f64
    where
        T: AsRef<[f32]>,
    {
        let mut sum = 0.;

        site_likelihood_inner(
            self.as_slice(),
            self.strides.as_slice(),
            site.each_ref().map(|x| x.as_ref()).as_slice(),
            &mut sum,
            1.,
        );

        sum
    }
}

fn posterior_inner(
    sfs: &[f64],
    strides: &[usize],
    site: &[&[f32]],
    buf: &mut [f64],
    sum: &mut f64,
    acc: f64,
) {
    match site {
        &[hd] => {
            debug_assert_eq!(sfs.len(), hd.len());

            buf.iter_mut()
                .zip(sfs)
                .zip(hd)
                .for_each(|((buf, sfs), &saf)| {
                    let v = sfs * saf as f64 * acc;
                    *sum += v;
                    *buf = v
                })
        }
        [hd, cons @ ..] => {
            let (stride, strides) = strides.split_first().expect("invalid strides");

            for (i, &saf) in hd.iter().enumerate() {
                let offset = i * stride;

                posterior_inner(
                    &sfs[offset..][..*stride],
                    strides,
                    cons,
                    &mut buf[offset..][..*stride],
                    sum,
                    saf as f64 * acc,
                );
            }
        }
        [] => (),
    }
}

fn site_likelihood_inner(sfs: &[f64], strides: &[usize], site: &[&[f32]], sum: &mut f64, acc: f64) {
    match site {
        &[hd] => sfs.iter().zip(hd).for_each(|(sfs, &saf)| {
            *sum += sfs * saf as f64 * acc;
        }),
        [hd, cons @ ..] => {
            let (stride, strides) = strides.split_first().expect("invalid strides");

            for (i, &saf) in hd.iter().enumerate() {
                let offset = i * stride;

                site_likelihood_inner(&sfs[offset..], strides, cons, sum, saf as f64 * acc);
            }
        }
        [] => (),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_abs_diff_eq;

    use crate::{sfs1d, sfs2d};

    #[test]
    fn test_1d() {
        let sfs = sfs1d![1., 2., 3.].normalise();

        let site = &[&[2., 2., 2.]];
        let mut posterior = sfs1d![10., 20., 30.];
        let mut buf = Sfs::zeros(sfs.shape());

        let posterior_likelihood = sfs.posterior_into(site, &mut posterior, &mut buf);

        let expected = vec![10. + 1. / 6., 20. + 1. / 3., 30. + 1. / 2.];
        assert_abs_diff_eq!(posterior.as_slice(), expected.as_slice());

        let likelihood = sfs.site_likelihood(site);
        assert_abs_diff_eq!(likelihood, 2.);
        assert_abs_diff_eq!(likelihood, posterior_likelihood);
    }

    #[test]
    fn test_2d() {
        #[rustfmt::skip]
        let sfs = sfs2d![
            [1.,  2.,  3.,  4.,  5.],
            [6.,  7.,  8.,  9.,  10.],
            [11., 12., 13., 14., 15.],
        ].normalise();

        let site = &[&[2., 2., 2.][..], &[2., 4., 6., 8., 10.][..]];
        let mut posterior = Sfs::from_elem(1., sfs.shape());
        let mut buf = Sfs::zeros(sfs.shape());

        let posterior_likelihood = sfs.posterior_into(site, &mut posterior, &mut buf);

        #[rustfmt::skip]
        let expected = vec![
            1.002564, 1.010256, 1.023077, 1.041026, 1.064103,
            1.015385, 1.035897, 1.061538, 1.092308, 1.128205,
            1.028205, 1.061538, 1.100000, 1.143590, 1.192308,
        ];
        assert_abs_diff_eq!(posterior.as_slice(), expected.as_slice(), epsilon = 1e-6);

        let likelihood = sfs.site_likelihood(site);
        assert_abs_diff_eq!(likelihood, 13.);
        assert_abs_diff_eq!(likelihood, posterior_likelihood);
    }

    #[test]
    fn test_3d() {
        let sfs = Sfs::from_vec_shape((0..60).map(|x| x as f64).collect(), [3, 4, 5])
            .unwrap()
            .normalise();

        let site = &[
            &[1., 2., 3.][..],
            &[4., 5., 6., 7.][..],
            &[8., 9., 10., 11., 12.][..],
        ];
        let mut posterior = Sfs::from_elem(1., sfs.shape());
        let mut buf = Sfs::zeros(sfs.shape());

        let posterior_likelihood = sfs.posterior_into(site, &mut posterior, &mut buf);

        let expected = vec![
            1.00000, 1.00015, 1.00032, 1.00053, 1.00078, 1.00081, 1.00109, 1.00141, 1.00178,
            1.00218, 1.00194, 1.00240, 1.00291, 1.00347, 1.00407, 1.00339, 1.00407, 1.00481,
            1.00560, 1.00645, 1.00517, 1.00611, 1.00711, 1.00818, 1.00931, 1.00808, 1.00945,
            1.01091, 1.01244, 1.01406, 1.01164, 1.01353, 1.01551, 1.01760, 1.01978, 1.01584,
            1.01833, 1.02093, 1.02364, 1.02647, 1.01551, 1.01789, 1.02036, 1.02293, 1.02560,
            1.02182, 1.02509, 1.02848, 1.03200, 1.03563, 1.02909, 1.03338, 1.03782, 1.04240,
            1.04712, 1.03733, 1.04276, 1.04836, 1.05413, 1.06007,
        ];
        assert_abs_diff_eq!(posterior.as_slice(), expected.as_slice(), epsilon = 1e-5);

        let likelihood = sfs.site_likelihood(site);
        assert_abs_diff_eq!(likelihood, 139.8418, epsilon = 1e-4);
        assert_abs_diff_eq!(likelihood, posterior_likelihood);
    }
}
