use std::io;

use angsd_io::saf;

use rand::Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct Saf {
    values: Vec<f32>,
    sites: usize,
    cols: usize,
}

impl Saf {
    pub fn cols(&self) -> usize {
        self.cols
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
pub struct JointSaf<const N: usize>([Saf; N]);

impl<const N: usize> JointSaf<N> {
    pub fn cols(&self) -> [usize; N] {
        self.0
            .iter()
            .map(|saf| saf.cols())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    pub fn shuffle<R>(&mut self, rng: &mut R)
    where
        R: Rng,
    {
        // Modified from rand::seq::SliceRandom::shuffle
        for i in (1..self.sites()).rev() {
            let j = rng.gen_range(0..i + 1);

            for n in 0..N {
                self.0[n].swap_sites(i, j);
            }
        }
    }

    pub fn read<R>(readers: [saf::BgzfReader<R>; N]) -> io::Result<Self>
    where
        R: io::BufRead + io::Seek,
    {
        let max_sites = readers
            .iter()
            .map(|reader| reader.index().total_sites())
            .min()
            .expect("no readers provided");

        let cols: Vec<usize> = readers
            .iter()
            .map(|reader| reader.index().alleles() + 1)
            .collect();

        let mut vecs: Vec<Vec<f32>> = cols
            .iter()
            .map(|cols| Vec::with_capacity(cols * max_sites))
            .collect();

        let readers = Vec::from(readers);
        let mut intersect = saf::reader::Intersect::new(readers).unwrap();

        let mut bufs = intersect.create_record_bufs();
        while intersect.read_records(&mut bufs)?.is_not_done() {
            for (buf, vec) in bufs.iter().zip(vecs.iter_mut()) {
                vec.extend_from_slice(buf.values());
            }
        }

        let safs = vecs
            .into_iter()
            .zip(cols.iter())
            .map(|(mut vec, cols)| {
                vec.shrink_to_fit();
                let n = vec.len();

                assert_eq!(n % cols, 0);
                let sites = n / cols;

                Saf::from_log(vec, sites)
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Ok(Self::new(safs))
    }

    pub fn sites(&self) -> usize {
        // Equal length maintained as invariant, so either is fine
        self.0[0].sites()
    }

    pub fn values(&self) -> [&[f32]; N] {
        self.0
            .iter()
            .map(|saf| saf.values())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn new(safs: [Saf; N]) -> Self {
        safs.windows(2)
            .for_each(|x| assert_eq!(x[0].sites(), x[1].sites()));

        Self(safs)
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
