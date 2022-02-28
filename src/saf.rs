use std::io;

use angsd_io::saf;

use rand::{seq::SliceRandom, Rng};

#[derive(Clone, Debug, PartialEq)]
pub struct Saf1d<const N: usize>(Vec<[f32; N]>);

impl<const N: usize> Saf1d<N> {
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn new(values: Vec<[f32; N]>) -> Self {
        Self(values)
    }

    pub fn read<R>(mut reader: saf::BgzfReader<R>) -> io::Result<Self>
    where
        R: io::BufRead,
    {
        let total_sites: usize = reader.index().records().iter().map(|x| x.sites()).sum();
        let mut values = Vec::with_capacity(total_sites);

        let mut buf = [0.0; N];
        while reader
            .value_reader_mut()
            .read_values(buf.as_mut_slice())?
            .is_not_done()
        {
            values.push(buf);
        }

        let mut new = Self::new(values);

        new.exp();

        Ok(new)
    }

    pub fn shuffle<R>(&mut self, rng: &mut R)
    where
        R: Rng,
    {
        self.0.shuffle(rng)
    }

    pub fn sites(&self) -> &[[f32; N]] {
        &self.0
    }

    #[inline]
    fn exp(&mut self) {
        bytemuck::cast_slice_mut::<_, f32>(self.0.as_mut_slice())
            .iter_mut()
            .for_each(|x| *x = x.exp());
    }

    #[inline]
    fn swap(&mut self, i: usize, j: usize) {
        self.0.swap(i, j)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Saf2d<const N: usize, const M: usize>(Saf1d<N>, Saf1d<M>);

impl<const N: usize, const M: usize> Saf2d<N, M> {
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        // Equal length maintained as invariant, so either is fine
        self.0.len()
    }

    pub fn new(left: Vec<[f32; N]>, right: Vec<[f32; M]>) -> Self {
        assert_eq!(left.len(), right.len());

        Self(Saf1d::new(left), Saf1d::new(right))
    }

    pub fn shuffle<R>(&mut self, rng: &mut R)
    where
        R: Rng,
    {
        // Modified from rand::seq::SliceRandom::shuffle
        for i in (1..self.len()).rev() {
            let j = rng.gen_range(0..i + 1);

            self.0.swap(i, j);
            self.1.swap(i, j);
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
        let mut left_values = Vec::with_capacity(max_sites);
        let mut right_values = Vec::with_capacity(max_sites);

        let mut reader = saf::reader::Intersect::new(first_reader, second_reader);

        let (mut left, mut right) = reader.create_record_buf();
        while reader
            .read_record_pair(&mut left, &mut right)?
            .is_not_done()
        {
            let left_array = left
                .values()
                .try_into()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            left_values.push(left_array);

            let right_array = right
                .values()
                .try_into()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            right_values.push(right_array);
        }

        let mut new = Self::new(left_values, right_values);

        new.exp();

        Ok(new)
    }

    pub fn sites(&self) -> (&[[f32; N]], &[[f32; M]]) {
        (self.0.sites(), self.1.sites())
    }

    #[inline]
    fn exp(&mut self) {
        self.0.exp();
        self.1.exp();
    }
}
