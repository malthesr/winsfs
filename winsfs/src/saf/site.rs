use super::{Lifetime, ShapeError};

/// A type that can be cheaply converted to a SAF site.
///
/// This is akin to GATified [`AsRef`] for SAF sites.
pub trait AsSiteView<const N: usize>: for<'a> Lifetime<'a, Item = SiteView<'a, N>> {
    /// Returns a SAF site view of `self`.
    fn as_site_view(&self) -> <Self as Lifetime<'_>>::Item;
}

macro_rules! impl_shared_site_methods {
    () => {
        /// Returns the values of the site as a flat slice.
        ///
        /// See the [`Site`] documentation for details on the storage order.
        pub fn as_slice(&self) -> &[f32] {
            &self.values
        }

        /// Returns an iterator over all values in the site.
        ///
        /// See the [`Site`] documentation for details on the storage order.
        pub fn iter(&self) -> ::std::slice::Iter<f32> {
            self.values.iter()
        }

        /// Returns the shape of the site.
        #[inline]
        pub fn shape(&self) -> [usize; N] {
            self.shape
        }

        /// Returns an array of slices corresponding to the sites in each population.
        #[inline]
        pub fn split(&self) -> [&[f32]; N] {
            let mut buf = &self.values[..];

            self.shape.map(|i| {
                let (hd, tl) = buf.split_at(i);
                buf = tl;
                hd
            })
        }
    };
}

/// A single site of SAF likelihoods for `N` populations.
///
/// Internally, the site is stored as a contiguous block of memory, with all values from the first
/// population first, then the second, and so on. [`Site::shape`] gives the number of values for
/// each population.
#[derive(Clone, Debug, PartialEq)]
pub struct Site<const N: usize> {
    values: Vec<f32>,
    shape: [usize; N],
}

impl<const N: usize> Site<N> {
    /// Returns a mutable reference to the values of the SAF site as a flat slice.
    ///
    /// See the [`Site`] documentation for details on the storage order.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.values
    }

    /// Returns an iterator over all values in the site.
    ///
    /// See the [`Site`] documentation for details on the storage order.
    pub fn iter_mut(&mut self) -> ::std::slice::IterMut<f32> {
        self.values.iter_mut()
    }

    /// Returns a new SAF site.
    ///
    /// The number of provided values must be equal to the sum of shapes.
    /// See the [`Site`] documentation for details on the storage order.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::saf::Site;
    /// let vec = vec![0.0, 0.1, 0.2, 1.0, 1.2];
    /// let shape = [3, 2];
    /// let site = Site::new(vec, shape).unwrap();
    /// assert_eq!(site.split(), [&[0.0, 0.1, 0.2][..], &[1.0, 1.2][..]]);
    /// ```
    ///
    /// A [`ShapeError`] is thrown if the shape does not fit the number of values:
    ///
    /// ```
    /// use winsfs::saf::Site;
    /// let vec = vec![0.0, 0.1, 0.2, 1.0, 1.2];
    /// let wrong_shape = [6, 2];
    /// assert!(Site::new(vec, wrong_shape).is_err());
    /// ```
    pub fn new(values: Vec<f32>, shape: [usize; N]) -> Result<Self, ShapeError<N>> {
        let len = values.len();
        let width: usize = shape.iter().sum();

        if len == width {
            Ok(Self::new_unchecked(values, shape))
        } else {
            Err(ShapeError { len, shape })
        }
    }

    /// Returns a new SAF site without checking that the shape fits the number of values.
    pub(crate) fn new_unchecked(values: Vec<f32>, shape: [usize; N]) -> Self {
        Self { values, shape }
    }

    /// Returns a view of the entire site.
    #[inline]
    pub fn view(&self) -> SiteView<N> {
        SiteView::new_unchecked(self.values.as_slice(), self.shape)
    }

    impl_shared_site_methods! {}
}

impl<'a, const N: usize> Lifetime<'a> for Site<N> {
    type Item = SiteView<'a, N>;
}

impl<const N: usize> AsSiteView<N> for Site<N> {
    #[inline]
    fn as_site_view(&self) -> <Self as Lifetime<'_>>::Item {
        self.view()
    }
}

impl<'a, 'b, const N: usize> Lifetime<'a> for &'b Site<N> {
    type Item = SiteView<'a, N>;
}

impl<'a, const N: usize> AsSiteView<N> for &'a Site<N> {
    #[inline]
    fn as_site_view(&self) -> <Self as Lifetime<'_>>::Item {
        self.view()
    }
}

/// A view of a single site of SAF likelihoods for `N` populations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SiteView<'a, const N: usize> {
    values: &'a [f32],
    shape: [usize; N],
}

impl<'a, const N: usize> SiteView<'a, N> {
    /// Returns a new SAF site view;
    ///
    /// The number of provided values must be equal to the sum of shapes.
    /// See the [`Site`] documentation for details on the storage order.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::saf::SiteView;
    /// let slice = &[0.0, 0.1, 0.2, 1.0, 1.2];
    /// let shape = [3, 2];
    /// let site = SiteView::new(slice, shape).unwrap();
    /// assert_eq!(site.split(), [&[0.0, 0.1, 0.2][..], &[1.0, 1.2][..]]);
    /// ```
    ///
    /// A [`ShapeError`] is thrown if the shape does not fit the number of values:
    ///
    /// ```
    /// use winsfs::saf::SiteView;
    /// let slice = &[0.0, 0.1, 0.2, 1.0, 1.2];
    /// let wrong_shape = [6, 2];
    /// assert!(SiteView::new(slice, wrong_shape).is_err());
    /// ```
    pub fn new(values: &'a [f32], shape: [usize; N]) -> Result<Self, ShapeError<N>> {
        let len = values.len();
        let width: usize = shape.iter().sum();

        if len == width {
            Ok(Self::new_unchecked(values, shape))
        } else {
            Err(ShapeError { len, shape })
        }
    }

    /// Returns a new SAF site view without checking that the shape fits the number of values.
    pub(crate) fn new_unchecked(values: &'a [f32], shape: [usize; N]) -> Self {
        Self { values, shape }
    }

    impl_shared_site_methods! {}
}

impl<'a, 'b, const N: usize> Lifetime<'a> for SiteView<'b, N> {
    type Item = SiteView<'a, N>;
}

impl<'a, const N: usize> AsSiteView<N> for SiteView<'a, N> {
    #[inline]
    fn as_site_view(&self) -> <Self as Lifetime<'_>>::Item {
        *self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_1d() {
        let vec = vec![0., 1., 2.];
        let site = Site::new(vec, [3]).unwrap();
        assert_eq!(site.split(), [&[0., 1., 2.][..]]);
        assert_eq!(site.split(), site.view().split());
    }

    #[test]
    fn test_split_2d() {
        let vec = vec![0., 1., 10., 11., 12., 13.];
        let site = Site::new(vec, [2, 4]).unwrap();
        assert_eq!(site.split(), [&[0., 1.][..], &[10., 11., 12., 13.][..]]);
        assert_eq!(site.split(), site.view().split());
    }

    #[test]
    fn test_split_3d() {
        let vec = vec![0., 10., 11., 12., 20., 21.];
        let site = Site::new(vec, [1, 3, 2]).unwrap();
        assert_eq!(
            site.split(),
            [&[0.,][..], &[10., 11., 12.][..], &[20., 21.][..]]
        );
        assert_eq!(site.split(), site.view().split());
    }
}
