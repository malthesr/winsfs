use std::io;

use angsd_io::saf;

use rand::Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct Saf1d {
    values: Vec<f32>,
    sites: usize,
    cols: usize,
}

impl Saf1d {
    pub fn cols(&self) -> [usize; 1] {
        [self.cols]
    }

    pub fn read<R>(mut reader: saf::BgzfReader<R>) -> io::Result<Self>
    where
        R: io::BufRead,
    {
        let total_sites: usize = reader.index().total_sites();

        let capacity = (reader.index().alleles() + 1) * total_sites;
        let mut values = vec![0.0; capacity];

        reader
            .value_reader_mut()
            .read_values(values.as_mut_slice())?;

        Ok(Self::from_log(values, total_sites))
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

    pub fn sites(&self) -> usize {
        self.sites
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    fn from_log(mut values: Vec<f32>, sites: usize) -> Self {
        values.iter_mut().for_each(|x| *x = x.exp());

        Self::new(values, sites)
    }

    fn new(values: Vec<f32>, sites: usize) -> Self {
        assert_eq!(values.len() % sites, 0);

        let cols = values.len() / sites;

        Self {
            values,
            sites,
            cols,
        }
    }

    #[inline]
    fn swap_sites(&mut self, i: usize, j: usize) {
        swap_chunks(self.values.as_mut_slice(), i, j, self.cols);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Saf2d(Saf1d, Saf1d);

impl Saf2d {
    pub fn cols(&self) -> [usize; 2] {
        [self.0.cols, self.1.cols]
    }

    pub fn shuffle<R>(&mut self, rng: &mut R)
    where
        R: Rng,
    {
        // Modified from rand::seq::SliceRandom::shuffle
        for i in (1..self.sites()).rev() {
            let j = rng.gen_range(0..i + 1);

            self.0.swap_sites(i, j);
            self.1.swap_sites(i, j);
        }
    }

    pub fn read<R>(
        first_reader: saf::BgzfReader<R>,
        second_reader: saf::BgzfReader<R>,
    ) -> io::Result<Self>
    where
        R: io::BufRead + io::Seek,
    {
        let max_sites = usize::min(
            first_reader.index().total_sites(),
            second_reader.index().total_sites(),
        );

        let left_cols = first_reader.index().alleles() + 1;
        let left_capacity = left_cols * max_sites;
        let mut left_values = Vec::with_capacity(left_capacity);

        let right_cols = second_reader.index().alleles() + 1;
        let right_capacity = right_cols * max_sites;
        let mut right_values = Vec::with_capacity(right_capacity);

        let mut reader = saf::reader::Intersect::new(first_reader, second_reader);

        let (mut left, mut right) = reader.create_record_buf();
        while reader
            .read_record_pair(&mut left, &mut right)?
            .is_not_done()
        {
            left_values.extend_from_slice(left.values());
            right_values.extend_from_slice(right.values());
        }

        left_values.shrink_to_fit();
        right_values.shrink_to_fit();

        let left_sites = left_values.len() / left_cols;
        let right_sites = right_values.len() / right_cols;

        Ok(Self::new(
            Saf1d::from_log(left_values, left_sites),
            Saf1d::from_log(right_values, right_sites),
        ))
    }

    pub fn sites(&self) -> usize {
        // Equal length maintained as invariant, so either is fine
        self.0.sites()
    }

    pub fn values(&self) -> (&[f32], &[f32]) {
        (self.0.values(), self.1.values())
    }

    fn new(left: Saf1d, right: Saf1d) -> Self {
        assert_eq!(left.sites(), right.sites());

        Self(left, right)
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
