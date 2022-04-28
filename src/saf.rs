#![allow(unstable_name_collisions)]

use std::{error::Error, fmt, io};

use angsd_io::saf;

use rand::Rng;

use rayon::slice::ParallelSlice;

mod joint_saf;
pub use joint_saf::{JointSaf, JointSafView, JointShapeError};

mod blocks;
pub use blocks::Blocks;

mod traits;
pub use traits::{ArrayExt, BlockIterator, IntoArray, ParSiteIterator};

macro_rules! impl_shared_saf_methods {
    () => {
        pub fn as_slice(&self) -> &[f32] {
            &self.values
        }

        pub fn sites(&self) -> usize {
            self.values.len() / self.shape
        }

        pub fn shape(&self) -> usize {
            self.shape
        }
    };
}

#[derive(Clone, Debug, PartialEq)]
pub struct Saf {
    values: Vec<f32>,
    shape: usize,
}

impl Saf {
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.values
    }

    pub(super) fn from_log(mut values: Vec<f32>, shape: usize) -> Result<Self, ShapeError> {
        values.iter_mut().for_each(|x| *x = x.exp());

        Self::new(values, shape)
    }

    pub fn new(values: Vec<f32>, shape: usize) -> Result<Self, ShapeError> {
        let len = values.len();

        if len % shape == 0 {
            Ok(Self::new_unchecked(values, shape))
        } else {
            Err(ShapeError { len, shape })
        }
    }

    fn new_unchecked(values: Vec<f32>, shape: usize) -> Self {
        Self { values, shape }
    }

    pub fn read<R>(mut reader: saf::BgzfReader<R>) -> io::Result<Self>
    where
        R: io::BufRead,
    {
        let total_sites: usize = reader.index().total_sites();
        let shape = reader.index().alleles() + 1;

        let capacity = shape * total_sites;
        let mut values = vec![0.0; capacity];

        reader
            .value_reader_mut()
            .read_values(values.as_mut_slice())?;

        Self::from_log(values, shape).map_err(io::Error::from)
    }

    pub fn shuffle<R>(&mut self, rng: &mut R)
    where
        R: Rng,
    {
        // Modified from rand::seq::SliceRandom::shuffle
        for i in (1..self.sites()).rev() {
            let j = rng.gen_range(0..i + 1);

            self.swap_sites(i, j);
        }
    }

    pub(super) fn swap_sites(&mut self, i: usize, j: usize) {
        swap_chunks(self.values.as_mut_slice(), i, j, self.shape);
    }

    pub fn view(&self) -> SafView<'_> {
        SafView::new_unchecked(&self.values, self.shape)
    }

    impl_shared_saf_methods!();
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SafView<'a> {
    values: &'a [f32],
    shape: usize,
}

impl<'a> SafView<'a> {
    fn new_unchecked(values: &'a [f32], shape: usize) -> Self {
        Self { values, shape }
    }

    pub fn par_iter_sites(&self) -> rayon::slice::Chunks<'a, f32> {
        self.values.par_chunks(self.shape)
    }

    pub fn split_at_site(&self, site: usize) -> (Self, Self) {
        let (hd, tl) = self.values.split_at(site * self.shape);

        (
            Self::new_unchecked(hd, self.shape),
            Self::new_unchecked(tl, self.shape),
        )
    }

    impl_shared_saf_methods!();
}

#[derive(Clone, Debug)]
pub struct ShapeError {
    shape: usize,
    len: usize,
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cannot construct a SAF of shape {} from {} values",
            self.shape, self.len,
        )
    }
}

impl Error for ShapeError {}

impl From<ShapeError> for io::Error {
    fn from(e: ShapeError) -> Self {
        io::Error::new(io::ErrorKind::InvalidData, e)
    }
}

/// Split `s` into chunks of `chunk_size` and swap chunks `i` and `j`, where `i` > `j`
fn swap_chunks<T>(s: &mut [T], i: usize, j: usize, chunk_size: usize)
where
    T: std::fmt::Debug,
{
    if i == j {
        return;
    }

    let (hd, tl) = s.split_at_mut(i * chunk_size);

    let left = &mut hd[j * chunk_size..][..chunk_size];
    let right = &mut tl[..chunk_size];

    left.swap_with_slice(right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_at_site() {
        let values = vec![1., 1., 1., 2., 2., 2., 3., 3., 3.];
        let saf = Saf::new_unchecked(values, 3);

        let (hd, tl) = saf.view().split_at_site(1);
        assert_eq!(hd.values, &[1., 1., 1.]);
        assert_eq!(tl.values, &[2., 2., 2., 3., 3., 3.]);
    }

    #[test]
    fn test_swap_chunks() {
        let mut v = vec![0, 0, 1, 1, 2, 2, 3, 3];

        swap_chunks(v.as_mut_slice(), 1, 1, 2);
        assert_eq!(v, v);

        swap_chunks(v.as_mut_slice(), 2, 0, 2);
        let expected = vec![2, 2, 1, 1, 0, 0, 3, 3];
        assert_eq!(v, expected);

        swap_chunks(v.as_mut_slice(), 1, 0, 4);
        let expected = vec![0, 0, 3, 3, 2, 2, 1, 1];
        assert_eq!(v, expected);
    }
}
