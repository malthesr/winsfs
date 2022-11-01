use crate::{
    saf::{AsSiteView, Site},
    sfs::{Sfs, USfs},
};

use super::likelihood::{Likelihood, LogLikelihood};

/// A type of SAF site that can be used as input for EM.
///
/// This trait should not typically be used in user code, except as a trait bound where code has to
/// be written that is generic over different EM input types.
pub trait EmSite<const D: usize> {
    /// Returns the likelihood of a single site given the SFS.
    ///
    /// # Panics
    ///
    /// Panics if the shape of the SFS does not fit the shape of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::{em::{likelihood::LogLikelihood, EmSite}, saf::Site, sfs::Sfs};
    /// let sfs = Sfs::uniform([5]);
    /// let site = Site::new(vec![1.0, 0.0, 0.0, 0.0, 0.0], [5]).unwrap();
    /// assert_eq!(site.log_likelihood(&sfs), LogLikelihood::from(0.2f64.ln()));
    /// ```
    fn likelihood(&self, sfs: &Sfs<D>) -> Likelihood;

    /// Returns the log-likelihood of a single site given the SFS.
    ///
    /// # Panics
    ///
    /// Panics if the shape of the SFS does not fit the shape of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs_core::{em::{likelihood::LogLikelihood, EmSite}, saf::Site, sfs::Sfs};
    /// let sfs = Sfs::uniform([5]);
    /// let site = Site::new(vec![1.0, 0.0, 0.0, 0.0, 0.0], [5]).unwrap();
    /// assert_eq!(site.log_likelihood(&sfs), LogLikelihood::from(0.2f64.ln()));
    /// ```
    fn log_likelihood(&self, sfs: &Sfs<D>) -> LogLikelihood {
        self.likelihood(sfs).ln()
    }

    /// Adds the posterior counts for the site into the provided `posterior` buffer, using the
    /// extra `buf` to avoid extraneous allocations.
    ///
    /// The `buf` will be overwritten, and so it's state is unimportant. The shape of the site
    /// will be matched against the shape of the SFS, and a panic will be thrown if they do not
    /// match. The shapes of `posterior` and `buf` are unchecked, but must match the shape of self.
    ///
    /// The likelihood of the site given the SFS is returned.
    ///
    /// # Panics
    ///
    /// Panics if the shape of the SFS does not fit the shape of `self`.
    fn posterior_into(
        &self,
        sfs: &Sfs<D>,
        posterior: &mut USfs<D>,
        buf: &mut USfs<D>,
    ) -> Likelihood;
}

impl<const D: usize, T> EmSite<D> for T
where
    T: AsSiteView<D>,
{
    fn likelihood(&self, sfs: &Sfs<D>) -> Likelihood {
        let site = self.as_site_view();
        assert_eq!(sfs.shape, site.shape());

        let mut sum = 0.;

        likelihood_inner(
            sfs.as_slice(),
            sfs.strides.as_slice(),
            site.split().as_slice(),
            &mut sum,
            1.,
        );

        sum.into()
    }

    fn posterior_into(
        &self,
        sfs: &Sfs<D>,
        posterior: &mut USfs<D>,
        buf: &mut USfs<D>,
    ) -> Likelihood {
        let site = self.as_site_view();
        assert_eq!(sfs.shape, site.shape());

        let mut sum = 0.;

        posterior_inner(
            sfs.as_slice(),
            sfs.strides.as_slice(),
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
}

/// A type of SAF site that can be used as input for streaming EM.
///
/// Like [`EmSite`], this trait should not typically be used in user code, except as a trait bound
/// where code has to be written that is generic over different EM input types.
pub trait StreamEmSite<const D: usize>: EmSite<D> {
    /// Creates a new site from its shape.
    ///
    /// The returned site should be suitable for use as a read buffer.
    fn from_shape(shape: [usize; D]) -> Self;
}

impl<const D: usize> StreamEmSite<D> for Site<D> {
    fn from_shape(shape: [usize; D]) -> Self {
        let vec = vec![0.0; shape.iter().sum()];
        Site::new(vec, shape).unwrap()
    }
}

/// Calculate the likelihood for a site any dimension recursively.
///
/// The logic here is a simplified version of `posterior_inner`: see the comments there for more.
fn likelihood_inner(sfs: &[f64], strides: &[usize], site: &[&[f32]], sum: &mut f64, acc: f64) {
    match site {
        &[hd] => sfs.iter().zip(hd).for_each(|(sfs, &saf)| {
            *sum += sfs * saf as f64 * acc;
        }),
        [hd, cons @ ..] => {
            let (stride, strides) = strides.split_first().expect("invalid strides");

            for (i, &saf) in hd.iter().enumerate() {
                let offset = i * stride;

                likelihood_inner(&sfs[offset..], strides, cons, sum, saf as f64 * acc);
            }
        }
        [] => (),
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{saf::Site, sfs1d, sfs2d};

    fn test_f64_equal(x: f64, y: f64, epsilon: f64) {
        assert!((x - y).abs() < epsilon)
    }

    fn test_f64_slice_equal(xs: &[f64], ys: &[f64], epsilon: f64) {
        assert_eq!(xs.len(), ys.len());

        for (&x, &y) in xs.iter().zip(ys) {
            test_f64_equal(x, y, epsilon)
        }
    }

    #[test]
    fn test_1d() {
        let sfs = sfs1d![1., 2., 3.].normalise();

        let site = Site::new(vec![2., 2., 2.], [3]).unwrap();
        let mut posterior = sfs1d![10., 20., 30.];
        let mut buf = USfs::zeros(sfs.shape);

        let posterior_likelihood = site.posterior_into(&sfs, &mut posterior, &mut buf);

        let expected = vec![10. + 1. / 6., 20. + 1. / 3., 30. + 1. / 2.];
        test_f64_slice_equal(posterior.as_slice(), expected.as_slice(), f64::EPSILON);

        let likelihood = site.likelihood(&sfs);
        test_f64_equal(likelihood.into(), 2., f64::EPSILON);
        test_f64_equal(likelihood.into(), posterior_likelihood.into(), f64::EPSILON);
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
        let mut posterior = USfs::from_elem(1., sfs.shape);
        let mut buf = USfs::zeros(sfs.shape);

        let posterior_likelihood = site.posterior_into(&sfs, &mut posterior, &mut buf);

        #[rustfmt::skip]
        let expected = vec![
            1.002564, 1.010256, 1.023077, 1.041026, 1.064103,
            1.015385, 1.035897, 1.061538, 1.092308, 1.128205,
            1.028205, 1.061538, 1.100000, 1.143590, 1.192308,
        ];
        test_f64_slice_equal(posterior.as_slice(), expected.as_slice(), 1e-6);

        let likelihood = site.likelihood(&sfs);
        test_f64_equal(likelihood.into(), 13., f64::EPSILON);
        test_f64_equal(likelihood.into(), posterior_likelihood.into(), f64::EPSILON);
    }

    #[test]
    fn test_3d() {
        let sfs = USfs::from_vec_shape((0..60).map(|x| x as f64).collect(), [3, 4, 5])
            .unwrap()
            .normalise();

        let site = Site::new((1..=12).map(|x| x as f32).collect(), [3, 4, 5]).unwrap();
        let mut posterior = USfs::from_elem(1., sfs.shape);
        let mut buf = USfs::zeros(sfs.shape);

        let posterior_likelihood = site.posterior_into(&sfs, &mut posterior, &mut buf);

        let expected = vec![
            1.00000, 1.00015, 1.00032, 1.00053, 1.00078, 1.00081, 1.00109, 1.00141, 1.00178,
            1.00218, 1.00194, 1.00240, 1.00291, 1.00347, 1.00407, 1.00339, 1.00407, 1.00481,
            1.00560, 1.00645, 1.00517, 1.00611, 1.00711, 1.00818, 1.00931, 1.00808, 1.00945,
            1.01091, 1.01244, 1.01406, 1.01164, 1.01353, 1.01551, 1.01760, 1.01978, 1.01584,
            1.01833, 1.02093, 1.02364, 1.02647, 1.01551, 1.01789, 1.02036, 1.02293, 1.02560,
            1.02182, 1.02509, 1.02848, 1.03200, 1.03563, 1.02909, 1.03338, 1.03782, 1.04240,
            1.04712, 1.03733, 1.04276, 1.04836, 1.05413, 1.06007,
        ];
        test_f64_slice_equal(posterior.as_slice(), expected.as_slice(), 1e-5);

        let likelihood = site.likelihood(&sfs);
        test_f64_equal(likelihood.into(), 139.8418, 1e-4);
        test_f64_equal(likelihood.into(), posterior_likelihood.into(), f64::EPSILON);
    }
}
