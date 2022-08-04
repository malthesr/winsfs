use std::{collections::VecDeque, io, iter::repeat};

use crate::{
    io::{Enumerate, ReadSite, Rewind, Take},
    saf::{iter::IntoBlockIterator, AsSafView},
    sfs::{Sfs, USfs},
};

use super::{to_f64, Em, EmStep, StreamingEm};

/// A runner of the window EM algorithm.
///
/// The window EM algorithm updates the SFS estimate in smaller blocks of data, leading to multiple
/// updates to the estimate per full EM-step. These block estimates are averaged over a sliding
/// window to smooth the global estimate. The algorithm can be configured to use different EM-like
/// algorithms (corresponding to the parameter `T`) for each inner block update step.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowEm<const N: usize, T> {
    em: T,
    window: Window<N>,
    block_size: usize,
}

impl<const N: usize, T> WindowEm<N, T> {
    /// Returns a new instance of the runner.
    ///
    /// The `em` is the inner kind of EM to handle the blocks, and the `block_size` is the number
    /// of sites per blocks. The provided `window` should match the shape of the input and SFS
    /// that is provided for inference later. Where no good prior guess for the SFS exists,
    /// using [`Window::from_zeros`] is recommended.
    pub fn new(em: T, window: Window<N>, block_size: usize) -> Self {
        Self {
            em,
            window,
            block_size,
        }
    }
}

impl<const N: usize, T> EmStep for WindowEm<N, T>
where
    T: EmStep,
{
    type Status = Vec<T::Status>;
}

impl<const N: usize, I, T> Em<N, I> for WindowEm<N, T>
where
    for<'a> &'a I: IntoBlockIterator<N>,
    for<'a> T: Em<N, <&'a I as IntoBlockIterator<N>>::Item>,
{
    fn e_step(&mut self, sfs: &Sfs<N>, input: &I) -> (Self::Status, USfs<N>) {
        let mut sfs = sfs.clone();
        let mut log_likelihoods = Vec::with_capacity(self.block_size);

        let mut sites = 0;

        for block in input.into_block_iter(self.block_size) {
            sites += block.as_saf_view().sites();

            let (log_likelihood, posterior) = self.em.e_step(&sfs, &block);

            self.window.update(posterior);

            sfs = self.window.sum().normalise();
            log_likelihoods.push(log_likelihood);
        }

        (log_likelihoods, sfs.scale(to_f64(sites)))
    }
}

impl<const N: usize, R, T> StreamingEm<N, R> for WindowEm<N, T>
where
    R: Rewind,
    for<'a> T: StreamingEm<N, Take<Enumerate<&'a mut R>>>,
{
    fn stream_e_step(
        &mut self,
        sfs: &Sfs<N>,
        reader: &mut R,
    ) -> io::Result<(Self::Status, USfs<N>)> {
        let mut sfs = sfs.clone();
        let mut log_likelihoods = Vec::with_capacity(self.block_size);

        let mut sites = 0;

        loop {
            let mut block_reader = reader.take(self.block_size);

            let (log_likelihood, posterior) = self.em.stream_e_step(&sfs, &mut block_reader)?;
            self.window.update(posterior);

            sfs = self.window.sum().normalise();
            log_likelihoods.push(log_likelihood);

            sites += block_reader.sites_read();

            if reader.is_done()? {
                break;
            }
        }

        Ok((log_likelihoods, sfs.scale(to_f64(sites))))
    }
}

/// A window of block SFS estimates, used in window EM.
///
/// As part of the window EM algorithm, "windows" of block estimates are averaged out to give
/// a running estimate of the SFS. The "window size" governs the number of past block estimates
/// that are remembered and averaged over.
#[derive(Clone, Debug, PartialEq)]
pub struct Window<const N: usize> {
    // Items are ordered old to new: oldest iterations are at the front, newest at the back
    deque: VecDeque<USfs<N>>,
}

impl<const N: usize> Window<N> {
    /// Creates a new window of with size `window_size` by repeating a provided SFS.
    pub fn from_initial(initial: USfs<N>, window_size: usize) -> Self {
        let deque = repeat(initial).take(window_size).collect();

        Self { deque }
    }

    /// Creates a new window of zero-initialised SFS with size `window_size`.
    pub fn from_zeros(shape: [usize; N], window_size: usize) -> Self {
        Self::from_initial(USfs::zeros(shape), window_size)
    }

    /// Returns the shape of the window.
    pub fn shape(&self) -> [usize; N] {
        // We maintain as invariant that all items in deque have same shape,
        // in order to make this okay
        *(self.deque[0].shape())
    }

    /// Returns the sum of SFS in the window.
    fn sum(&self) -> USfs<N> {
        let first = USfs::zeros(self.shape());

        self.deque.iter().fold(first, |sum, item| sum + item)
    }

    /// Updates the window after a new iteration of window EM.
    ///
    /// This corresponds to removing the oldest SFS from the window, and adding the new `sfs`.
    fn update(&mut self, sfs: USfs<N>) {
        if *sfs.shape() != self.shape() {
            panic!("shape of provided SFS does not match shape of window")
        }

        let _old = self.deque.pop_front();
        self.deque.push_back(sfs);
    }

    /// Returns the window size, corresponding to the number of past block estimates in the window.
    pub fn window_size(&self) -> usize {
        self.deque.len()
    }
}
