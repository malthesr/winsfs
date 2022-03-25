use std::{collections::VecDeque, iter};

use crate::{Saf1d, Saf2d, Sfs, Sfs1d, Sfs2d};

macro_rules! info_sfs {
    (target: $target:expr, $fmt_str:literal, $sfs:ident * $sites:expr) => {
        log_sfs!(target: $target, log::Level::Info, $fmt_str, $sfs * $sites)
    };
}

macro_rules! trace_sfs {
    (target: $target:expr, $fmt_str:literal, $sfs:ident * $sites:expr) => {
        log_sfs!(target: $target, log::Level::Trace, $fmt_str, $sfs * $sites)
    };
}

macro_rules! log_sfs {
    (target: $target:expr, $level:expr, $fmt_str:literal, $sfs:ident * $sites:expr) => {
        if log::log_enabled!(target: $target, $level) {
            let fmt_sfs = $sfs
                .iter()
                .map(|v| format!("{:.6}", v * $sites as f64))
                .collect::<Vec<_>>()
                .join(" ");

            log::log!(target: $target, $level, $fmt_str, fmt_sfs);
        }
    };
}

impl<const N: usize> Sfs<N>
where
    Sfs<N>: EmSfs<N>,
{
    pub fn em(&self, input: &<Self as EmSfs<N>>::Input, epochs: usize) -> Self {
        let sites = input.sites();
        let mut sfs = self.clone();

        for i in 0..epochs {
            info_sfs!(target: "em", "Epoch {i}, current SFS: {}", sfs * sites);
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
        let mut window = Window::zeros(self.shape(), window_size);
        let sites = input.sites();
        let mut sfs = self.clone();

        for i in 0..epochs {
            info_sfs!(target: "em", "Epoch {i}, current SFS: {}", sfs * sites);
            sfs = sfs.window_em_step(input, &mut window, block_size);
        }

        sfs
    }
}

pub trait EmInput {
    fn sites(&self) -> usize;
}

impl EmInput for Saf1d {
    fn sites(&self) -> usize {
        Saf1d::sites(self)
    }
}

impl EmInput for Saf2d {
    fn sites(&self) -> usize {
        Saf2d::sites(self)
    }
}

pub trait EmSfs<const N: usize> {
    type Input: EmInput;

    fn em_step(&self, input: &Self::Input) -> Self;

    fn em_step_with_log_likelihood(&self, saf: &Self::Input) -> (f64, Self);

    fn log_likelihood(&self, input: &Self::Input) -> f64;

    fn window_em_step(
        &self,
        input: &Self::Input,
        window: &mut Window<N>,
        block_size: usize,
    ) -> Self;
}

impl EmSfs<1> for Sfs1d {
    type Input = Saf1d;

    fn em_step(&self, saf: &Self::Input) -> Self {
        let mut posterior = self.e_step(saf.values());
        posterior.normalise();

        posterior
    }

    fn em_step_with_log_likelihood(&self, saf: &Self::Input) -> (f64, Self) {
        let (log_likelihood, mut posterior) = self.e_step_with_log_likelihood(saf.values());
        posterior.normalise();

        (log_likelihood, posterior)
    }

    fn log_likelihood(&self, saf: &Self::Input) -> f64 {
        Sfs1d::log_likelihood(self, saf.values())
    }

    fn window_em_step(&self, saf: &Self::Input, window: &mut Window<1>, block_size: usize) -> Self {
        let sites = saf.sites();
        let mut sfs = self.clone();

        for (i, block) in saf.values().chunks(saf.cols()[0] * block_size).enumerate() {
            let block_posterior = sfs.e_step(block);
            window.update(block_posterior);

            sfs = window.sum();
            sfs.normalise();

            trace_sfs!(target: "windowem", "Block {i}, current SFS: {}", sfs * sites);
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

    fn em_step_with_log_likelihood(&self, safs: &Self::Input) -> (f64, Self) {
        let (row_sites, col_sites) = safs.values();

        let (log_likelihood, mut posterior) = self.e_step_with_log_likelihood(row_sites, col_sites);
        posterior.normalise();

        (log_likelihood, posterior)
    }

    fn log_likelihood(&self, safs: &Self::Input) -> f64 {
        let (row_sites, col_sites) = safs.values();

        Sfs2d::log_likelihood(self, row_sites, col_sites)
    }

    fn window_em_step(
        &self,
        safs: &Self::Input,
        window: &mut Window<2>,
        block_size: usize,
    ) -> Self {
        let (row_sites, col_sites) = safs.values();
        let [row_cols, col_cols] = safs.cols();
        let sites = safs.sites();
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

            trace_sfs!(target: "windowem", "Block {i}, current SFS: {}", sfs * sites);
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
            .fold(Sfs::zeros(self.0[0].shape()), |sum, item| sum + item)
    }

    pub fn zeros(shape: [usize; N], window_size: usize) -> Self {
        Self(iter::repeat(Sfs::zeros(shape)).take(window_size).collect())
    }
}
