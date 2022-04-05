use std::collections::VecDeque;

use crate::Sfs;

use super::{Em, Input};

#[derive(Clone, Debug)]
pub struct Window<const N: usize> {
    sfs: Sfs<N>,
    window: Blocks<N>,
    block_size: usize,
    epochs: usize,
}

impl<const N: usize> Window<N>
where
    for<'a> Sfs<N>: Em<'a, N>,
{
    pub fn em(&mut self, input: &<Sfs<N> as Em<N>>::Input) {
        let sites = input.sites(self.sfs.shape());

        for i in 0..self.epochs {
            info_sfs!(target: "em", "Epoch {i}, current SFS: {}", self.sfs, sites);

            self.em_step(input);
        }
    }

    pub fn em_step(&mut self, input: &<Sfs<N> as Em<N>>::Input) {
        let sites = input.sites(self.sfs.shape());

        for (i, block) in input
            .iter_blocks(self.sfs.shape(), self.block_size)
            .enumerate()
        {
            let block_posterior = self.sfs.e_step(&block);
            self.window.update(block_posterior);

            self.sfs = self.window.sum();
            self.sfs.normalise();

            trace_sfs!(target: "windowem", "Block {i}, current SFS: {}", self.sfs, sites);
        }
    }

    pub fn into_sfs(self) -> Sfs<N> {
        self.sfs
    }

    pub fn new(initial: Sfs<N>, window_size: usize, block_size: usize, epochs: usize) -> Self {
        let window = Blocks::zeros(initial.shape(), window_size);

        Self {
            sfs: initial,
            window,
            block_size,
            epochs,
        }
    }

    pub fn sfs(&self) -> &Sfs<N> {
        &self.sfs
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Blocks<const N: usize>(VecDeque<Sfs<N>>);

impl<const N: usize> Blocks<N> {
    pub fn update(&mut self, item: Sfs<N>) {
        let _old = self.0.pop_front();
        self.0.push_back(item);
    }

    pub fn sum(&self) -> Sfs<N> {
        self.0
            .iter()
            .fold(Sfs::zeros(self.0[0].shape()), |sum, item| sum + item)
    }

    pub fn zeros(shape: [usize; N], blocks: usize) -> Self {
        Self(std::iter::repeat(Sfs::zeros(shape)).take(blocks).collect())
    }
}
