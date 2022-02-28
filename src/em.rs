use std::{collections::VecDeque, iter};

use crate::{Saf1d, Saf2d, Sfs, Sfs1d, Sfs2d};

#[derive(Clone, Debug, PartialEq)]
pub struct Window<T>(VecDeque<T>);

impl<T> Window<T>
where
    T: Sfs,
{
    pub fn update(&mut self, item: T) {
        let _old = self.0.pop_front();
        self.0.push_back(item);
    }

    pub fn sum(&self) -> T {
        self.0.iter().fold(T::zero(), |sum, item| sum + *item)
    }

    pub fn zero(window_size: usize) -> Self {
        Self(iter::repeat(T::zero()).take(window_size).collect())
    }
}

pub trait Em: Sfs {
    type Input;

    fn em_step(&self, input: &Self::Input) -> Self;

    fn window_em_step(
        &self,
        saf: &Self::Input,
        window: &mut Window<Self>,
        block_size: usize,
    ) -> Self;

    fn em(&self, input: &Self::Input, epochs: usize) -> Self {
        let mut sfs = *self;

        for i in 0..epochs {
            log::info!(target: "em", "Epoch {i}, current SFS: {}", sfs.values_to_string(6));
            sfs = sfs.em_step(input);
        }

        sfs
    }

    fn from_em(input: &Self::Input, epochs: usize) -> Self {
        let sfs = Self::uniform();

        sfs.em(input, epochs)
    }

    fn window_em(
        &self,
        input: &Self::Input,
        window_size: usize,
        block_size: usize,
        epochs: usize,
    ) -> Self {
        let mut window = Window::zero(window_size);

        let mut sfs = *self;

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

    fn from_window_em(
        input: &Self::Input,
        window_size: usize,
        block_size: usize,
        epochs: usize,
    ) -> Self {
        let sfs = Self::uniform();

        sfs.window_em(input, window_size, block_size, epochs)
    }
}

impl<const N: usize> Em for Sfs1d<N> {
    type Input = Saf1d<N>;

    fn em_step(&self, saf: &Self::Input) -> Self {
        let mut posterior = self.e_step(saf.sites());
        posterior.normalise();

        posterior
    }

    fn window_em_step(
        &self,
        saf: &Self::Input,
        window: &mut Window<Self>,
        block_size: usize,
    ) -> Self {
        let mut sfs = *self;

        for (i, block) in saf.sites().chunks(block_size).enumerate() {
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

impl<const R: usize, const C: usize> Em for Sfs2d<R, C> {
    type Input = Saf2d<R, C>;

    fn em_step(&self, safs: &Self::Input) -> Self {
        let (row_sites, col_sites) = safs.sites();

        let mut posterior = self.e_step(row_sites, col_sites);
        posterior.normalise();

        posterior
    }

    fn window_em_step(
        &self,
        safs: &Self::Input,
        window: &mut Window<Self>,
        block_size: usize,
    ) -> Self {
        let (row_sites, col_sites) = safs.sites();

        let mut sfs = *self;

        for (i, (row_block, col_block)) in row_sites
            .chunks(block_size)
            .zip(col_sites.chunks(block_size))
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
