use std::io;

use rayon::iter::{IndexedParallelIterator, ParallelIterator};

use crate::{
    em::likelihood::{Likelihood, LogLikelihood, SumOf},
    io::ReadSite,
    saf::{
        iter::{IntoParallelSiteIterator, IntoSiteIterator},
        AsSiteView, Site,
    },
};

use super::{Sfs, UnnormalisedSfs};

impl<const N: usize> Sfs<N> {
    /// Returns the log-likelihood of the data given the SFS, and the expected number of sites
    /// in each frequency bin given the SFS and the input.
    ///
    /// This corresponds to an E-step for the EM algorithm. The returned SFS corresponds to the
    /// expected number of sites in each bin given `self` and the `input`.
    /// The sum of the returned SFS will be equal to the number of sites in the input.
    ///
    /// # Panics
    ///
    /// Panics if any of the sites in the input does not fit the shape of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::{sfs::Sfs, saf1d, sfs1d};
    /// let sfs = Sfs::uniform([5]);
    /// let saf = saf1d![
    ///     [1., 0., 0., 0., 0.],
    ///     [0., 1., 0., 0., 0.],
    ///     [1., 0., 0., 0., 0.],
    ///     [0., 0., 0., 1., 0.],
    /// ];
    /// let (log_likelihood, posterior) = sfs.e_step(&saf);
    /// assert_eq!(posterior, sfs1d![2., 1., 0., 1., 0.]);
    /// assert_eq!(log_likelihood, sfs.log_likelihood(&saf));
    /// ```
    pub fn e_step<I>(&self, input: I) -> (SumOf<LogLikelihood>, UnnormalisedSfs<N>)
    where
        I: IntoSiteIterator<N>,
    {
        let iter = input.into_site_iter();
        let sites = iter.len();

        let (log_likelihood, posterior, _) = iter.fold(
            (
                LogLikelihood::from(0.0),
                Sfs::zeros(self.shape()),
                Sfs::zeros(self.shape()),
            ),
            |(mut log_likelihood, mut posterior, mut buf), site| {
                log_likelihood += self.posterior_into(site, &mut posterior, &mut buf).ln();

                (log_likelihood, posterior, buf)
            },
        );

        (SumOf::new(log_likelihood, sites), posterior)
    }

    /// Returns the log-likelihood of the data given the SFS, and the expected number of sites
    /// in each frequency bin given the SFS and the input.
    ///
    /// This is the parallel version of [`Sfs::e_step`], see also its documentation for more.
    ///
    /// # Panics
    ///
    /// Panics if any of the sites in the input does not fit the shape of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::{sfs::Sfs, saf1d, sfs1d};
    /// let sfs = Sfs::uniform([5]);
    /// let saf = saf1d![
    ///     [1., 0., 0., 0., 0.],
    ///     [0., 1., 0., 0., 0.],
    ///     [1., 0., 0., 0., 0.],
    ///     [0., 0., 0., 1., 0.],
    /// ];
    /// let (log_likelihood, posterior) = sfs.par_e_step(&saf);
    /// assert_eq!(posterior, sfs1d![2., 1., 0., 1., 0.]);
    /// assert_eq!(log_likelihood, sfs.log_likelihood(&saf));
    /// ```
    pub fn par_e_step<I>(&self, input: I) -> (SumOf<LogLikelihood>, UnnormalisedSfs<N>)
    where
        I: IntoParallelSiteIterator<N>,
    {
        let iter = input.into_par_site_iter();
        let sites = iter.len();

        let (log_likelihood, posterior) = iter
            .fold(
                || {
                    (
                        LogLikelihood::from(0.0),
                        Sfs::zeros(self.shape()),
                        Sfs::zeros(self.shape()),
                    )
                },
                |(mut log_likelihood, mut posterior, mut buf), site| {
                    log_likelihood += self.posterior_into(site, &mut posterior, &mut buf).ln();

                    (log_likelihood, posterior, buf)
                },
            )
            .map(|(log_likelihood, posterior, _buf)| (log_likelihood, posterior))
            .reduce(
                || (LogLikelihood::from(0.0), Sfs::zeros(self.shape())),
                |a, b| (a.0 + b.0, a.1 + b.1),
            );

        (SumOf::new(log_likelihood, sites), posterior)
    }

    /// Returns the log-likelihood of the data given the SFS.
    ///
    /// # Panics
    ///
    /// Panics if any of the sites in the input does not fit the shape of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::{em::likelihood::{Likelihood, SumOf}, sfs::Sfs, saf1d, sfs1d};
    /// let sfs = Sfs::uniform([5]);
    /// let saf = saf1d![
    ///     [1., 0., 0., 0., 0.],
    ///     [0., 1., 0., 0., 0.],
    ///     [1., 0., 0., 0., 0.],
    ///     [0., 0., 0., 1., 0.],
    /// ];
    /// let expected = SumOf::new(Likelihood::from(0.2f64.powi(4)).ln(), saf.sites());
    /// assert_eq!(sfs.log_likelihood(&saf), expected);
    /// ```
    pub fn log_likelihood<I>(&self, input: I) -> SumOf<LogLikelihood>
    where
        I: IntoSiteIterator<N>,
    {
        let iter = input.into_site_iter();
        let sites = iter.len();

        let log_likelihood = iter.fold(LogLikelihood::from(0.0), |log_likelihood, site| {
            log_likelihood + self.site_log_likelihood(site)
        });

        SumOf::new(log_likelihood, sites)
    }

    /// Returns the log-likelihood of the data given the SFS.
    ///
    /// This is the parallel version of [`Sfs::log_likelihood`].
    ///
    /// # Panics
    ///
    /// Panics if any of the sites in the input does not fit the shape of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::{em::likelihood::{Likelihood, SumOf}, sfs::Sfs, saf1d, sfs1d};
    /// let sfs = Sfs::uniform([5]);
    /// let saf = saf1d![
    ///     [1., 0., 0., 0., 0.],
    ///     [0., 1., 0., 0., 0.],
    ///     [1., 0., 0., 0., 0.],
    ///     [0., 0., 0., 1., 0.],
    /// ];
    /// let expected = SumOf::new(Likelihood::from(0.2f64.powi(4)).ln(), saf.sites());
    /// assert_eq!(sfs.par_log_likelihood(&saf), expected);
    /// ```
    pub fn par_log_likelihood<I>(&self, input: I) -> SumOf<LogLikelihood>
    where
        I: IntoParallelSiteIterator<N>,
    {
        let iter = input.into_par_site_iter();
        let sites = iter.len();

        let log_likelihood = iter
            .fold(
                || LogLikelihood::from(0.0),
                |log_likelihood, site| log_likelihood + self.site_log_likelihood(site),
            )
            .sum();

        SumOf::new(log_likelihood, sites)
    }

    /// Adds the posterior counts for `site` into the provided `posterior buffer`, using the
    /// extra `buf` to avoid extraneous allocations.
    ///
    /// The `buf` will be overwritten, and so it's state is unimportant. The shape of the `site`
    /// will be matched against the shape of `self`, and a panic will be thrown if they do not
    /// match. The shapes of `posterior` and `buf` are unchecked, but must match the shape of self.
    ///
    /// The likelihood of the site given the SFS is returned.
    pub(crate) fn posterior_into<T>(
        &self,
        site: T,
        posterior: &mut UnnormalisedSfs<N>,
        buf: &mut UnnormalisedSfs<N>,
    ) -> Likelihood
    where
        T: AsSiteView<N>,
    {
        let site = site.as_site_view();
        assert_eq!(self.shape, site.shape());

        let mut sum = 0.;

        posterior_inner(
            self.as_slice(),
            self.strides.as_slice(),
            site.split().as_slice(),
            buf.as_mut_slice(),
            &mut sum,
            1.,
        );

        // Normalising and adding to the posterior in a single iterator has slightly better perf
        // than normalising and then adding to posterior.
        buf.iter_mut()
            .zip(posterior.iter_mut())
            .for_each(|(buf, posterior)| {
                *buf /= sum;
                *posterior += *buf;
            });

        sum.into()
    }

    /// Returns the log-likelihood of a single site given the SFS.
    ///
    /// # Panics
    ///
    /// Panics if the shape of the site does not fit the shape of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::{em::likelihood::LogLikelihood, saf::Site, sfs::Sfs};
    /// let sfs = Sfs::uniform([5]);
    /// let site = Site::new(vec![1.0, 0.0, 0.0, 0.0, 0.0], [5]).unwrap();
    /// assert_eq!(sfs.site_log_likelihood(site), LogLikelihood::from(0.2f64.ln()));
    /// ```
    pub fn site_log_likelihood<T>(&self, site: T) -> LogLikelihood
    where
        T: AsSiteView<N>,
    {
        self.site_likelihood(site).ln()
    }

    /// Returns the likelihood of a single site given the SFS.
    ///
    /// # Panics
    ///
    /// Panics if the shape of the site does not fit the shape of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::{em::likelihood::Likelihood, saf::Site, sfs::Sfs};
    /// let sfs = Sfs::uniform([5]);
    /// let site = Site::new(vec![1.0, 0.0, 0.0, 0.0, 0.0], [5]).unwrap();
    /// assert_eq!(sfs.site_likelihood(site), Likelihood::from(0.2));
    /// ```
    pub fn site_likelihood<T>(&self, site: T) -> Likelihood
    where
        T: AsSiteView<N>,
    {
        let site = site.as_site_view();
        assert_eq!(self.shape, site.shape());

        let mut sum = 0.;

        site_likelihood_inner(
            self.as_slice(),
            self.strides.as_slice(),
            site.split().as_slice(),
            &mut sum,
            1.,
        );

        sum.into()
    }

    /// Returns the log-likelihood of the data given the SFS, and the expected number of sites
    /// in each frequency bin given the SFS and the input.
    ///
    /// This is the streaming version of [`Sfs::e_step`], see also its documentation for more.
    ///
    /// # Panics
    ///
    /// Panics if any of the sites in the input does not fit the shape of `self`.
    pub fn stream_e_step<R>(
        &self,
        mut reader: R,
    ) -> io::Result<(SumOf<LogLikelihood>, UnnormalisedSfs<N>)>
    where
        R: ReadSite,
    {
        let mut post = Sfs::zeros(self.shape());
        let mut buf = Sfs::zeros(self.shape());

        let vec = vec![0.0; self.shape().iter().sum()];
        let mut site = Site::new(vec, self.shape()).unwrap();

        let mut sites = 0;
        let mut log_likelihood = LogLikelihood::from(0.0);
        while reader.read_site(site.as_mut_slice())?.is_not_done() {
            log_likelihood += self.posterior_into(&site, &mut post, &mut buf).ln();

            sites += 1;
        }

        Ok((SumOf::new(log_likelihood, sites), post))
    }

    /// Returns the log-likelihood of the data given the SFS.
    ///
    /// This is the streaming version of [`Sfs::log_likelihood`].
    ///
    /// # Panics
    ///
    /// Panics if any of the sites in the input does not fit the shape of `self`.
    pub fn stream_log_likelihood<R>(&self, mut reader: R) -> io::Result<SumOf<LogLikelihood>>
    where
        R: ReadSite,
    {
        let vec = vec![0.0; self.shape().iter().sum()];
        let mut site = Site::new(vec, self.shape()).unwrap();

        let mut sites = 0;
        let mut log_likelihood = LogLikelihood::from(0.0);
        while reader.read_site(site.as_mut_slice())?.is_not_done() {
            log_likelihood += self.site_log_likelihood(&site);

            sites += 1;
        }

        Ok(SumOf::new(log_likelihood, sites))
    }
}

/// Calculate the posterior for a site any dimension recursively.
///
/// The posterior is written into the `buf`, which is not normalised. The `sum` will contain
/// the likelihood, which can be used to normalise. The passed-in `sum` should typically be zero,
/// whereas the passed-in `acc` should typically be one.
///
/// It is  assumed that `sfs` and `buf` have the same length, which should correspond to the product
/// of the length of the sites in `site`.
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
            // Base case: we have a single site, which signifies that the SFS slice
            // now corresponds to a single slice along its last dimension, e.g. a row in 2D.
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
            // Recursive case: we have multiple sites. For each value in the first site,
            // we add the value to the accumulant, "peel" the corresponding slice of the SFS,
            // and recurse to a lower dimension.
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

/// Calculate the likelihood for a site any dimension recursively.
///
/// The logic here is a simplified version of `posterior_inner`: see the comments there for more.
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

    use crate::{saf::Site, sfs1d, sfs2d};

    #[test]
    fn test_1d() {
        let sfs = sfs1d![1., 2., 3.].normalise();

        let site = Site::new(vec![2., 2., 2.], [3]).unwrap();
        let mut posterior = sfs1d![10., 20., 30.];
        let mut buf = Sfs::zeros(sfs.shape());

        let posterior_likelihood = sfs.posterior_into(&site, &mut posterior, &mut buf);

        let expected = vec![10. + 1. / 6., 20. + 1. / 3., 30. + 1. / 2.];
        assert_abs_diff_eq!(posterior.as_slice(), expected.as_slice());

        let likelihood = sfs.site_likelihood(site);
        assert_abs_diff_eq!(likelihood, Likelihood::from(2.));
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

        let site = Site::new(vec![2., 2., 2., 2., 4., 6., 8., 10.], [3, 5]).unwrap();
        let mut posterior = Sfs::from_elem(1., sfs.shape());
        let mut buf = Sfs::zeros(sfs.shape());

        let posterior_likelihood = sfs.posterior_into(&site, &mut posterior, &mut buf);

        #[rustfmt::skip]
        let expected = vec![
            1.002564, 1.010256, 1.023077, 1.041026, 1.064103,
            1.015385, 1.035897, 1.061538, 1.092308, 1.128205,
            1.028205, 1.061538, 1.100000, 1.143590, 1.192308,
        ];
        assert_abs_diff_eq!(posterior.as_slice(), expected.as_slice(), epsilon = 1e-6);

        let likelihood = sfs.site_likelihood(site);
        assert_abs_diff_eq!(likelihood, Likelihood::from(13.));
        assert_abs_diff_eq!(likelihood, posterior_likelihood);
    }

    #[test]
    fn test_3d() {
        let sfs = Sfs::from_vec_shape((0..60).map(|x| x as f64).collect(), [3, 4, 5])
            .unwrap()
            .normalise();

        let site = Site::new((1..=12).map(|x| x as f32).collect(), [3, 4, 5]).unwrap();
        let mut posterior = Sfs::from_elem(1., sfs.shape());
        let mut buf = Sfs::zeros(sfs.shape());

        let posterior_likelihood = sfs.posterior_into(&site, &mut posterior, &mut buf);

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
        assert_abs_diff_eq!(likelihood, Likelihood::from(139.8418), epsilon = 1e-4);
        assert_abs_diff_eq!(likelihood, posterior_likelihood);
    }
}
