use std::{collections::VecDeque, io};

use crate::{
    saf::{BlockIterator, JointSafView},
    stream::{Header, Reader},
    Sfs,
};

use super::{Em, StoppingRule};

#[derive(Clone, Debug)]
pub struct Window<const N: usize> {
    sfs: Sfs<N>,
    window: Blocks<N>,
    block_size: usize,
    stopping_rule: StoppingRule,
}

impl<const N: usize> Window<N>
where
    Sfs<N>: Em<N>,
{
    pub fn em<'a>(&mut self, input: &JointSafView<'a, N>)
    where
        JointSafView<'a, N>: BlockIterator<'a, N, Block = JointSafView<'a, N>>,
    {
        let sites: usize = input.sites();

        let mut epoch = 0;
        while !self.stopping_rule.stop() {
            self.em_step(input);

            epoch += 1;

            self.epoch_update(epoch, sites);
        }
    }

    pub fn streaming_em<R>(&mut self, reader: &mut Reader<R>, header: &Header) -> io::Result<()>
    where
        R: io::BufRead + io::Seek,
    {
        let sites = header.sites() as usize;

        let mut epoch = 0;
        while !self.stopping_rule.stop() {
            if epoch > 0 {
                reader.rewind(header)?;
            }

            self.streaming_em_step(reader, header)?;

            self.epoch_update(epoch, sites);

            epoch += 1;
        }

        Ok(())
    }

    fn em_step<'a>(&mut self, input: &JointSafView<'a, N>)
    where
        JointSafView<'a, N>: BlockIterator<'a, N, Block = JointSafView<'a, N>>,
    {
        for (i, block) in input.iter_blocks(self.block_size).enumerate() {
            let (log_likelihood, posterior) = self.sfs.e_step(&block);

            self.block_update(i, log_likelihood, posterior, input.sites(), block.sites());
        }
    }

    pub fn streaming_em_step<R>(
        &mut self,
        reader: &mut Reader<R>,
        header: &Header,
    ) -> io::Result<()>
    where
        R: io::BufRead,
    {
        let mut i = 0;

        loop {
            let mut block_reader = reader.take(self.block_size);

            let (log_likelihood, posterior) = self.sfs.streaming_e_step(&mut block_reader)?;

            self.block_update(
                i,
                log_likelihood,
                posterior,
                header.sites() as usize,
                block_reader.current(),
            );

            i += 0;

            if reader.is_done()? {
                break Ok(());
            }
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

    fn epoch_update(&mut self, epoch: usize, sites: usize) {
        log_sfs!(
            target:
            "windowem", log::Level::Debug,
            "Epoch {epoch}, current SFS: {}",
            self.sfs, sites
        );

        self.stopping_rule.epoch_update();
    }

    fn block_update(
        &mut self,
        i: usize,
        log_likelihood: f64,
        posterior: Sfs<N>,
        total_sites: usize,
        block_sites: usize,
    ) {
        self.window.update(posterior);

        self.sfs = self.window.sum();
        self.sfs.normalise();

        log_sfs!(
            target: "windowem",
            log::Level::Trace, "Block {i}, current SFS: {}",
            self.sfs, total_sites
        );

        let norm_log_likelihood = log_likelihood / block_sites as f64;
        self.stopping_rule.block_update(norm_log_likelihood);

        log::trace!(
            target: "windowem",
            "Block {i}, block log-likelihood: {log_likelihood:.8e} ({block_sites} sites)"
        );
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
