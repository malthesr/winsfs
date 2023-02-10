use std::{collections::VecDeque, io, iter::repeat};

use crate::{
    io::{Enumerate, ReadSite, Take},
    saf::{AsSafView, Blocks, SafView},
    sfs::{
        generics::{Shape, Unnorm},
        Sfs, SfsBase, USfs,
    },
};

use super::{
    likelihood::{LogLikelihood, SumOf},
    to_f64, EmStep, Sites, WithStatus,
};

/// A streaming runner of the window EM algorithm.
pub type StreamingWindowEm<T> = WindowEm<T, true>;

/// A runner of the window EM algorithm.
///
/// The window EM algorithm updates the SFS estimate in smaller blocks of data, leading to multiple
/// updates to the estimate per full EM-step. These block estimates are averaged over a sliding
/// window to smooth the global estimate. The algorithm can be configured to use different EM-like
/// algorithms (corresponding to the parameter `T`) for each inner block update step.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowEm<T, const STREAM: bool = false> {
    em: T,
    // The window gets filled if an initial SFS is provided, or during the first E-step when an
    // SFS is provided.
    window: Option<Window>,
    // The window size has to be stored here for the initial phase when `window` is `None`;
    // afterwards, it is redundant with the length of the individual ring buffers in the windows.
    window_size: usize,
    blocks: Blocks,
}

impl<T, const STREAM: bool> WindowEm<T, STREAM> {
    /// Returns a new instance of the runner.
    ///
    /// The `em` is the inner kind of EM to handle the blocks. The way the input should be split
    /// into blocks can be controlled using [`Blocks`]. The window size is in units of blocks.
    pub fn new(em: T, window_size: usize, blocks: Blocks) -> Self {
        Self {
            em,
            window: None,
            window_size,
            blocks,
        }
    }

    /// Returns a new instance of the runner with an initial SFS.
    ///
    /// See also documentation for [`WindowEm::new`].
    pub fn with_initial_sfs<S: Shape>(
        em: T,
        initial: &SfsBase<S, Unnorm>,
        window_size: usize,
        blocks: Blocks,
    ) -> Self {
        Self {
            em,
            window: Some(Window::from_initial(initial, window_size)),
            window_size,
            blocks,
        }
    }
}

impl<T, const STREAM: bool> WithStatus for WindowEm<T, STREAM>
where
    T: WithStatus,
{
    type Status = Vec<T::Status>;
}

impl<'a, const D: usize, T> EmStep<D, SafView<'a, D>> for WindowEm<T, false>
where
    T: EmStep<D, SafView<'a, D>>,
{
    type Error = T::Error;

    fn log_likelihood(
        &mut self,
        sfs: Sfs<D>,
        saf: SafView<'a, D>,
    ) -> Result<SumOf<LogLikelihood>, Self::Error> {
        self.em.log_likelihood(sfs, saf)
    }

    fn e_step(
        &mut self,
        mut sfs: Sfs<D>,
        saf: SafView<'a, D>,
    ) -> Result<(Self::Status, USfs<D>), Self::Error> {
        let window = self
            .window
            .get_or_insert_with(|| Window::from_zeros(*sfs.shape(), self.window_size));

        let blocks_inner = self.blocks.to_spec(saf.sites());
        let mut log_likelihoods = Vec::with_capacity(blocks_inner.blocks());

        let mut sites = 0;

        for block in saf.iter_blocks(self.blocks) {
            sites += block.as_saf_view().sites();

            let (log_likelihood, posterior) = self.em.e_step(sfs, block)?;

            window.update(posterior);

            sfs = window.sum().normalise();
            log_likelihoods.push(log_likelihood);
        }

        Ok((log_likelihoods, sfs.scale(to_f64(sites))))
    }
}

impl<'a, const D: usize, T, R> EmStep<D, &'a mut R> for WindowEm<T, true>
where
    for<'b> T: EmStep<D, &'b mut R, Error = io::Error>,
    for<'b, 'c> T: EmStep<D, &'b mut Take<Enumerate<&'c mut R>>, Error = io::Error>,
    R: ReadSite + Sites,
{
    type Error = io::Error;

    fn log_likelihood(
        &mut self,
        sfs: Sfs<D>,
        reader: &'a mut R,
    ) -> Result<SumOf<LogLikelihood>, Self::Error> {
        self.em.log_likelihood(sfs, reader)
    }

    fn e_step(
        &mut self,
        mut sfs: Sfs<D>,
        reader: &'a mut R,
    ) -> Result<(Self::Status, USfs<D>), Self::Error> {
        let window = self
            .window
            .get_or_insert_with(|| Window::from_zeros(*sfs.shape(), self.window_size));

        let block_spec = self.blocks.to_spec(reader.sites());
        let mut log_likelihoods = Vec::with_capacity(block_spec.blocks());

        let mut sites = 0;

        for block_size in block_spec.iter_block_sizes() {
            let mut block_reader = reader.take(block_size);

            let (log_likelihood, posterior) = self.em.e_step(sfs, &mut block_reader)?;
            window.update(posterior);

            sfs = window.sum().normalise();
            log_likelihoods.push(log_likelihood);

            sites += block_reader.sites_read();
        }

        Ok((log_likelihoods, sfs.scale(to_f64(sites))))
    }
}

/// A window of block SFS estimates, used in window EM.
///
/// As part of the window EM algorithm, "windows" of block estimates are averaged out to give
/// a running estimate of the SFS. The "window size" governs the number of past block estimates
/// that are remembered and averaged over.
///
/// We go through a bit of effort to not keep `USfs<D>` in the window to avoid the const bound
/// propagating to the `WindowEm` struct itself.
#[derive(Clone, Debug, PartialEq)]
struct Window {
    // In theory, it would be nicer to have a ringbuffer structure with a moving sum,
    // so that on each update the popped value is subtracted, and the pushed value is added;
    // in practice, this leads to weird numerical stuff, like -0.000... starting to show up in
    // results. Since this is not a bottleneck, therefore, we stick to just summing out deques
    // each time the sum is needed.
    buffers: Vec<VecDeque<f64>>,
    shape: Vec<usize>,
}

impl Window {
    /// Creates a new window of with size `window_size` by repeating a provided SFS.
    pub fn from_initial<S: Shape>(initial: &SfsBase<S, Unnorm>, window_size: usize) -> Self {
        Self {
            buffers: initial
                .iter()
                .map(|&v| repeat(v).take(window_size).collect())
                .collect(),
            shape: initial.shape().as_ref().to_vec(),
        }
    }

    /// Creates a new window of zero-initialised SFS with size `window_size`.
    pub fn from_zeros<S: Shape>(shape: S, window_size: usize) -> Self {
        Self::from_initial(&SfsBase::zeros(shape), window_size)
    }

    /// Returns the sum of SFS in the window.
    fn sum<const D: usize>(&self) -> USfs<D> {
        let sums = self
            .buffers
            .iter()
            .map(|buf| buf.iter().sum::<f64>())
            .collect();

        let shape = self
            .shape
            .clone()
            .try_into()
            .expect("window dimension does not match SFS dimension");

        USfs::from_vec_shape(sums, shape).expect("window shape does not match sums")
    }

    /// Updates the window after a new iteration of window EM.
    ///
    /// This corresponds to removing the oldest SFS from the window, and adding the new `sfs`.
    fn update<S: Shape>(&mut self, sfs: SfsBase<S, Unnorm>) {
        assert_eq!(
            sfs.shape().as_ref(),
            self.shape.as_slice(),
            "shape of provided SFS does not match shape of window"
        );

        sfs.iter()
            .zip(self.buffers.iter_mut())
            .for_each(|(&v, buf)| {
                buf.pop_front().unwrap();
                buf.push_back(v);
            });
    }
}
