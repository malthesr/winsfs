use std::{collections::VecDeque, iter};

use crate::{Saf1d, Saf2d, Sfs, Sfs1d, Sfs2d};

impl<const N: usize> Sfs<N>
where
    Sfs<N>: EmSfs<N>,
{
    pub fn em(&self, input: &<Self as EmSfs<N>>::Input, epochs: usize) -> Self {
        let mut sfs = self.clone();

        for i in 0..epochs {
            log::info!(target: "em", "Epoch {i}, current SFS: {}", sfs.values_to_string(6));
            sfs = sfs.em_step(input);
        }

        sfs
    }

    pub fn window_em(
        &self,
        input: &<Self as EmSfs<N>>::Input,
        window_size: usize,
        block_size: usize,
        epochs: usize,
    ) -> Self {
        let mut window = Window::zeros(self.dim(), window_size);

        let mut sfs = self.clone();

        for i in 0..epochs {
            log::info!(
                target: "windowem",
                "Epoch {i}, current SFS: {}",
                sfs.values_to_string(6)
            );
            sfs = sfs.window_em_step(input, &mut window, block_size);
        }

        sfs
    }
}

pub trait EmSfs<const N: usize> {
    type Input;

    fn em_step(&self, input: &Self::Input) -> Self;

    fn window_em_step(&self, saf: &Self::Input, window: &mut Window<N>, block_size: usize) -> Self;
}

impl EmSfs<1> for Sfs1d {
    type Input = Saf1d;

    fn em_step(&self, saf: &Self::Input) -> Self {
        let mut posterior = self.e_step(saf.values());
        posterior.normalise();

        posterior
    }

    fn window_em_step(&self, saf: &Self::Input, window: &mut Window<1>, block_size: usize) -> Self {
        let mut sfs = self.clone();

        for (i, block) in saf.values().chunks(saf.cols() * block_size).enumerate() {
            let block_posterior = sfs.e_step(block);
            window.update(block_posterior);

            sfs = window.sum();
            sfs.normalise();

            log::trace!(
                target: "windowem",
                "Block {i}, current SFS: {}",
                sfs.values_to_string(6)
            );
        }

        sfs
    }
}

impl EmSfs<2> for Sfs2d {
    type Input = Saf2d;

    fn em_step(&self, safs: &Self::Input) -> Self {
        let (row_sites, col_sites) = safs.values();

        let mut posterior = self.e_step(row_sites, col_sites);
        posterior.normalise();

        posterior
    }

    fn window_em_step(
        &self,
        safs: &Self::Input,
        window: &mut Window<2>,
        block_size: usize,
    ) -> Self {
        let (row_sites, col_sites) = safs.values();
        let [row_cols, col_cols] = safs.cols();

        let mut sfs = self.clone();

        for (i, (row_block, col_block)) in row_sites
            .chunks(block_size * row_cols)
            .zip(col_sites.chunks(block_size * col_cols))
            .enumerate()
        {
            let block_posterior = sfs.e_step(row_block, col_block);
            window.update(block_posterior);

            sfs = window.sum();
            sfs.normalise();

            log::trace!(
                target: "windowem",
                "Window EM block {i}, current SFS: {}",
                sfs.values_to_string(6)
            );
        }

        sfs
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Window<const N: usize>(VecDeque<Sfs<N>>);

impl<const N: usize> Window<N> {
    pub fn update(&mut self, item: Sfs<N>) {
        let _old = self.0.pop_front();
        self.0.push_back(item);
    }

    pub fn sum(&self) -> Sfs<N> {
        self.0
            .iter()
            .fold(Sfs::zeros(self.0[0].dim()), |sum, item| sum + item)
    }

    pub fn zeros(dim: [usize; N], window_size: usize) -> Self {
        Self(iter::repeat(Sfs::zeros(dim)).take(window_size).collect())
    }
}
