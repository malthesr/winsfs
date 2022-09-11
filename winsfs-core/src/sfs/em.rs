use std::io;

use rayon::iter::{IndexedParallelIterator, ParallelIterator};

use crate::{
    em::{
        likelihood::{LogLikelihood, SumOf},
        EmSite, StreamEmSite,
    },
    io::ReadSite,
    saf::iter::{IntoParallelSiteIterator, IntoSiteIterator},
};

use super::{Sfs, USfs};

/// The minimum allowable SFS value during EM.
const RESTRICT_MIN: f64 = f64::EPSILON;

impl<const D: usize> Sfs<D> {
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
    /// let (log_likelihood, posterior) = sfs.clone().e_step(&saf);
    /// assert_eq!(posterior, sfs1d![2., 1., 0., 1., 0.]);
    /// assert_eq!(log_likelihood, sfs.log_likelihood(&saf));
    /// ```
    pub fn e_step<I>(mut self, input: I) -> (SumOf<LogLikelihood>, USfs<D>)
    where
        I: IntoSiteIterator<D>,
        I::Item: EmSite<D>,
    {
        self = restrict(self, RESTRICT_MIN);
        let iter = input.into_site_iter();
        let sites = iter.len();

        let (log_likelihood, posterior, _) = iter.fold(
            (
                LogLikelihood::from(0.0),
                USfs::zeros(self.shape),
                USfs::zeros(self.shape),
            ),
            |(mut log_likelihood, mut posterior, mut buf), site| {
                log_likelihood += site.posterior_into(&self, &mut posterior, &mut buf).ln();

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
    /// let (log_likelihood, posterior) = sfs.clone().par_e_step(&saf);
    /// assert_eq!(posterior, sfs1d![2., 1., 0., 1., 0.]);
    /// assert_eq!(log_likelihood, sfs.log_likelihood(&saf));
    /// ```
    pub fn par_e_step<I>(mut self, input: I) -> (SumOf<LogLikelihood>, USfs<D>)
    where
        I: IntoParallelSiteIterator<D>,
        I::Item: EmSite<D>,
    {
        self = restrict(self, RESTRICT_MIN);
        let iter = input.into_par_site_iter();
        let sites = iter.len();

        let (log_likelihood, posterior) = iter
            .fold(
                || {
                    (
                        LogLikelihood::from(0.0),
                        USfs::zeros(self.shape),
                        USfs::zeros(self.shape),
                    )
                },
                |(mut log_likelihood, mut posterior, mut buf), site| {
                    log_likelihood += site.posterior_into(&self, &mut posterior, &mut buf).ln();

                    (log_likelihood, posterior, buf)
                },
            )
            .map(|(log_likelihood, posterior, _buf)| (log_likelihood, posterior))
            .reduce(
                || (LogLikelihood::from(0.0), USfs::zeros(self.shape)),
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
    pub fn log_likelihood<I>(mut self, input: I) -> SumOf<LogLikelihood>
    where
        I: IntoSiteIterator<D>,
    {
        self = restrict(self, RESTRICT_MIN);
        let iter = input.into_site_iter();
        let sites = iter.len();

        let log_likelihood = iter.fold(LogLikelihood::from(0.0), |log_likelihood, site| {
            log_likelihood + site.log_likelihood(&self)
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
    pub fn par_log_likelihood<I>(mut self, input: I) -> SumOf<LogLikelihood>
    where
        I: IntoParallelSiteIterator<D>,
    {
        self = restrict(self, RESTRICT_MIN);
        let iter = input.into_par_site_iter();
        let sites = iter.len();

        let log_likelihood = iter
            .fold(
                || LogLikelihood::from(0.0),
                |log_likelihood, site| log_likelihood + site.log_likelihood(&self),
            )
            .sum();

        SumOf::new(log_likelihood, sites)
    }

    /// Returns the log-likelihood of the data given the SFS, and the expected number of sites
    /// in each frequency bin given the SFS and the input.
    ///
    /// This is the streaming version of [`Sfs::e_step`], see also its documentation for more.
    ///
    /// # Panics
    ///
    /// Panics if any of the sites in the input does not fit the shape of `self`.
    pub fn stream_e_step<R>(mut self, mut reader: R) -> io::Result<(SumOf<LogLikelihood>, USfs<D>)>
    where
        R: ReadSite,
        R::Site: StreamEmSite<D>,
    {
        self = restrict(self, RESTRICT_MIN);
        let mut post = USfs::zeros(self.shape);
        let mut buf = USfs::zeros(self.shape);

        let mut site = <R::Site>::from_shape(self.shape);

        let mut sites = 0;
        let mut log_likelihood = LogLikelihood::from(0.0);
        while reader.read_site(&mut site)?.is_not_done() {
            log_likelihood += site.posterior_into(&self, &mut post, &mut buf).ln();

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
    pub fn stream_log_likelihood<R>(mut self, mut reader: R) -> io::Result<SumOf<LogLikelihood>>
    where
        R: ReadSite,
        R::Site: StreamEmSite<D>,
    {
        self = restrict(self, RESTRICT_MIN);
        let mut site = <R::Site>::from_shape(self.shape);

        let mut sites = 0;
        let mut log_likelihood = LogLikelihood::from(0.0);
        while reader.read_site(&mut site)?.is_not_done() {
            log_likelihood += site.log_likelihood(&self);

            sites += 1;
        }

        Ok(SumOf::new(log_likelihood, sites))
    }
}

/// Restricts the SFS so that all values in the spectrum are above `min`.
///
/// We have to ensure that that no value in the SFS is zero. If that happens, a situation can
/// arise in which a site arrives with information only in the part of the SFS that is zero: this
/// will lead to a zero posterior that cannot be normalised.
fn restrict<const D: usize>(mut sfs: Sfs<D>, min: f64) -> Sfs<D> {
    sfs.values.iter_mut().for_each(|v| {
        if *v < min {
            *v = min;
        }
    });

    sfs
}
