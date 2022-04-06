use std::collections::VecDeque;

use crate::Sfs;

use super::{Em, Input, StoppingRule};

#[derive(Clone, Debug)]
pub struct Window<const N: usize> {
    sfs: Sfs<N>,
    window: Blocks<N>,
    block_size: usize,
    stopping_rule: StoppingRule,
}

impl<const N: usize> Window<N>
where
    for<'a> Sfs<N>: Em<'a, N>,
{
    pub fn em(&mut self, input: &<Sfs<N> as Em<N>>::Input) {
        let sites = input.sites(self.sfs.shape());

        let mut epoch = 0;
        while !self.stopping_rule.stop() {
            info_sfs!(target: "windowem", "Epoch {epoch}, current SFS: {}", self.sfs, sites);

            self.em_step(input);

            self.stopping_rule.epoch_update();
            epoch += 1;
        }
    }

    fn em_step(&mut self, input: &<Sfs<N> as Em<N>>::Input) {
        let sites = input.sites(self.sfs.shape());

        for (i, block) in input
            .iter_blocks(self.sfs.shape(), self.block_size)
            .enumerate()
        {
            let (block_log_likelihood, block_posterior) =
                self.sfs.e_step_with_log_likelihood(&block);
            self.window.update(block_posterior);

            self.sfs = self.window.sum();
            self.sfs.normalise();

            trace_sfs!(target: "windowem", "Block {i}, current SFS: {}", self.sfs, sites);

            let block_sites = block.sites(self.sfs.shape());
            let norm_block_log_likelihood = block_log_likelihood / block_sites as f64;
            self.stopping_rule.block_update(norm_block_log_likelihood);

            log::trace!(
                target: "windowem",
                "Block {i}, block log-likelihood: {block_log_likelihood:.8e} ({block_sites} sites)"
            );
        }
    }

    pub fn into_sfs(self) -> Sfs<N> {
        self.sfs
    }

    pub fn new(
        initial: Sfs<N>,
        window_size: usize,
        block_size: usize,
        stopping_rule: StoppingRule,
    ) -> Self {
        let window = Blocks::zeros(initial.shape(), window_size);

        Self {
            sfs: initial,
            window,
            block_size,
            stopping_rule,
        }
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
