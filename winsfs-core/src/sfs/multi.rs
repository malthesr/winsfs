//! Multiple SFS with the same shape.

use std::{
    error::Error,
    fmt,
    ops::{Deref, DerefMut},
};

use super::{Normalisation, SfsBase, Shape};

/// A non-empty collection of multiple SFS with the same shape.
///
/// This is simply a newtype around a slice of SFSs, and can be used directly as such via
/// [`Deref`]/[`DerefMut`]. It exists primarily to avoid orphan rules.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Multi<T>(Vec<T>);

impl<S: Shape, N: Normalisation> Multi<SfsBase<S, N>> {
    /// Returns the shape of the spectra in the collection.
    pub fn shape(&self) -> &S {
        self.0[0].shape()
    }
}

impl<S: Shape, N: Normalisation> From<SfsBase<S, N>> for Multi<SfsBase<S, N>> {
    fn from(sfs: SfsBase<S, N>) -> Self {
        Self(vec![sfs])
    }
}

impl<S: Shape, N: Normalisation> TryFrom<Vec<SfsBase<S, N>>> for Multi<SfsBase<S, N>> {
    type Error = MultiError;

    fn try_from(vec: Vec<SfsBase<S, N>>) -> Result<Self, Self::Error> {
        let all_equal = vec.split_first().map(|(first, rest)| {
            rest.iter()
                .map(|sfs| sfs.shape() == first.shape())
                .all(|b| b)
        });

        match all_equal {
            Some(true) => Ok(Self(vec)),
            Some(false) => Err(MultiError::DifferentShapes),
            None => Err(MultiError::EmptyInput),
        }
    }
}

impl<S: Shape, N: Normalisation> From<Multi<SfsBase<S, N>>> for Vec<SfsBase<S, N>> {
    fn from(multi: Multi<SfsBase<S, N>>) -> Self {
        multi.0
    }
}

impl<S: Shape, N: Normalisation> Deref for Multi<SfsBase<S, N>> {
    type Target = [SfsBase<S, N>];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S: Shape, N: Normalisation> DerefMut for Multi<SfsBase<S, N>> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// An error associated with multi-SFS construction.
#[derive(Clone, Debug)]
pub enum MultiError {
    /// Spectra of different shapes provided.
    DifferentShapes,
    /// No spectra provided
    EmptyInput,
}

impl fmt::Display for MultiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DifferentShapes => {
                f.write_str("cannot construct multi-SFS with SFS from different shapes")
            }
            Self::EmptyInput => f.write_str("cannot construct multi-SFS from empty input"),
        }
    }
}

impl Error for MultiError {}
