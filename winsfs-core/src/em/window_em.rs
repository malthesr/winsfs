use std::{collections::VecDeque, io, iter::repeat};

use crate::{
    io::{Enumerate, ReadSite, Rewind, Take},
    saf::{iter::IntoBlockIterator, AsSafView},
    sfs::{Sfs, USfs},
};

use super::{to_f64, Em, EmSite, EmStep, StreamingEm};

/// A runner of the window EM algorithm.
///
/// The window EM algorithm updates the SFS estimate in smaller blocks of data, leading to multiple
/// updates to the estimate per full EM-step. These block estimates are averaged over a sliding
/// window to smooth the global estimate. The algorithm can be configured to use different EM-like
/// algorithms (corresponding to the parameter `T`) for each inner block update step.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowEm<const D: usize, T> {
    em: T,
    window: Window<D>,
    block_size: usize,
}

impl<const D: usize, T> WindowEm<D, T> {
    /// Returns a new instance of the runner.
    ///
    /// The `em` is the inner kind of EM to handle the blocks, and the `block_size` is the number
    /// of sites per blocks. The provided `window` should match the shape of the input and SFS
    /// that is provided for inference later. Where no good prior guess for the SFS exists,
    /// using [`Window::from_zeros`] is recommended.
    pub fn new(em: T, window: Window<D>, block_size: usize) -> Self {
        Self {
            em,
            window,
            block_size,
        }
    }
}

impl<const D: usize, T> EmStep for WindowEm<D, T>
where
    T: EmStep,
{
    type Status = Vec<T::Status>;
}

impl<const D: usize, I, T> Em<D, I> for WindowEm<D, T>
where
    for<'a> &'a I: IntoBlockIterator<D>,
    for<'a> T: Em<D, <&'a I as IntoBlockIterator<D>>::Item>,
{
    fn e_step(&mut self, mut sfs: Sfs<D>, input: &I) -> (Self::Status, USfs<D>) {
        let mut log_likelihoods = Vec::with_capacity(self.block_size);

        let mut sites = 0;

        for block in input.into_block_iter(self.block_size) {
            sites += block.as_saf_view().sites();

            let (log_likelihood, posterior) = self.em.e_step(sfs, &block);

            self.window.update(posterior);

            sfs = self.window.sum().normalise();
            log_likelihoods.push(log_likelihood);
        }

        (log_likelihoods, sfs.scale(to_f64(sites)))
    }
}

impl<const D: usize, R, T> StreamingEm<D, R> for WindowEm<D, T>
where
    R: Rewind,
    R::Site: EmSite<D>,
    for<'a> T: StreamingEm<D, Take<Enumerate<&'a mut R>>>,
{
    fn stream_e_step(
        &mut self,
        mut sfs: Sfs<D>,
        reader: &mut R,
    ) -> io::Result<(Self::Status, USfs<D>)> {
        let mut log_likelihoods = Vec::with_capacity(self.block_size);

        let mut sites = 0;

        loop {
            let mut block_reader = reader.take(self.block_size);

            let (log_likelihood, posterior) = self.em.stream_e_step(sfs, &mut block_reader)?;
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
pub struct Window<const D: usize> {
    // Items are ordered old to new: oldest iterations are at the front, newest at the back
    deque: VecDeque<USfs<D>>,
}

impl<const D: usize> Window<D> {
    /// Creates a new window of with size `window_size` by repeating a provided SFS.
    pub fn from_initial(initial: USfs<D>, window_size: usize) -> Self {
        let deque = repeat(initial).take(window_size).collect();

        Self { deque }
    }

    /// Creates a new window of zero-initialised SFS with size `window_size`.
    pub fn from_zeros(shape: [usize; D], window_size: usize) -> Self {
        Self::from_initial(USfs::zeros(shape), window_size)
    }

    /// Returns the shape of the window.
    pub fn shape(&self) -> [usize; D] {
        // We maintain as invariant that all items in deque have same shape,
        // in order to make this okay
        *(self.deque[0].shape())
    }

    /// Returns the sum of SFS in the window.
    fn sum(&self) -> USfs<D> {
        let first = USfs::zeros(self.shape());

        self.deque.iter().fold(first, |sum, item| sum + item)
    }

    /// Updates the window after a new iteration of window EM.
    ///
    /// This corresponds to removing the oldest SFS from the window, and adding the new `sfs`.
    fn update(&mut self, sfs: USfs<D>) {
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
