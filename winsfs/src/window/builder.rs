use std::{error::Error, fmt};

use crate::{saf::JointSafView, sfs::Sfs};

use super::{defaults::*, StoppingRule, Window};

/// Builder type for a window EM runner.
///
/// See [`Window::builder`] for constructor.
pub struct WindowBuilder<const N: usize> {
    sfs: Option<Sfs<N>>,
    window_size: Option<usize>,
    block_specification: Option<BlockSpecification>,
    stopping_rule: Option<StoppingRule>,
}

impl<const N: usize> WindowBuilder<N> {
    /// Sets the window EM block size.
    ///
    /// The block size must be non-zero and smaller than the number of sites.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::Window;
    /// let runner = Window::builder().block_size(100).build(10_000, [5]).unwrap();
    /// assert_eq!(runner.block_size(), 100);
    /// ```
    pub fn block_size(mut self, block_size: usize) -> Self {
        self.block_specification = Some(BlockSpecification::BlockSize(block_size));
        self
    }

    /// Returns a window EM runner, consuming `self`.
    ///
    /// Building requires knowing the number of sites and the shape of the input.
    /// See also [`WindowBuilder::build_from_input`] to extract these from a [`JointSafView`]
    /// directly.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::{window::defaults::*, StoppingRule, Window};
    ///
    /// let sites = 10_000;
    /// let shape = [5, 7];
    /// let runner = Window::builder().build(sites, shape).unwrap();
    ///
    /// assert!(runner.sfs().iter().all(|&x| x == runner.sfs()[[0, 0]])); // Uniform SFS
    /// assert_eq!(sites / runner.block_size(), DEFAULT_BLOCKS);
    /// assert_eq!(runner.window_size(), DEFAULT_WINDOW_SIZE);
    /// assert_eq!(runner.stopping_rule(), &StoppingRule::log_likelihood(DEFAULT_TOLERANCE));
    /// ```
    pub fn build(
        self,
        sites: usize,
        shape: [usize; N],
    ) -> Result<Window<N>, WindowBuilderError<N>> {
        let sfs = match self.sfs {
            Some(sfs) if sfs.shape() == shape => sfs,
            Some(sfs) => {
                return Err(WindowBuilderError::ShapeMismatch {
                    sfs_shape: sfs.shape(),
                    input_shape: shape,
                })
            }
            None => Sfs::uniform(shape),
        };

        let window_size = match self.window_size {
            Some(window_size) if window_size > 0 => window_size,
            Some(window_size) => return Err(WindowBuilderError::InvalidWindowSize { window_size }),
            None => DEFAULT_WINDOW_SIZE,
        };

        let block_size = self
            .block_specification
            .unwrap_or(BlockSpecification::Blocks(DEFAULT_BLOCKS))
            .block_size(sites)?;

        let stopping_rule = self
            .stopping_rule
            .unwrap_or_else(|| StoppingRule::log_likelihood(DEFAULT_TOLERANCE));

        Ok(Window::new(sfs, window_size, block_size, stopping_rule))
    }

    /// Returns a window EM runner, consuming `self`.
    ///
    /// See also [`WindowBuilder::build`].
    pub fn build_from_input(
        self,
        input: &JointSafView<N>,
    ) -> Result<Window<N>, WindowBuilderError<N>> {
        self.build(input.sites(), input.shape())
    }

    /// Sets the initial SFS for window EM.
    ///
    /// The initial SFS must be normalised.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::{sfs1d, Window};
    /// let runner = Window::builder()
    ///     .initial_sfs(sfs1d![5., 3., 2.].normalise())
    ///     .build(10_000, [3])
    ///     .unwrap();
    /// assert_eq!(runner.sfs()[[0]], 0.5);
    /// ```
    pub fn initial_sfs(mut self, sfs: Sfs<N>) -> Self {
        self.sfs = Some(sfs);
        self
    }

    /// Sets the number of blocks for window EM.
    ///
    /// The number of blocks must be non-zero and smaller than the number of sites.
    /// Note also that where the number of blocks does not evenly divide the number of sites,
    /// an extra, partial block will be appended at the end of extra epoch. The partial
    /// block will be weighted according to its relative size.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::Window;
    /// let runner = Window::builder().blocks(10).build(10_000, [5]).unwrap();
    /// assert_eq!(runner.block_size(), 1_000);
    /// ```
    pub fn blocks(mut self, blocks: usize) -> Self {
        self.block_specification = Some(BlockSpecification::Blocks(blocks));
        self
    }

    /// Sets the stopping rule for window EM.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::{StoppingRule, Window};
    /// let runner = Window::builder()
    ///     .stopping_rule(StoppingRule::epochs(10))
    ///     .build(10_000, [5])
    ///     .unwrap();
    /// assert_eq!(runner.stopping_rule(), &StoppingRule::epochs(10));
    /// ```
    pub fn stopping_rule(mut self, stopping_rule: StoppingRule) -> Self {
        self.stopping_rule = Some(stopping_rule);
        self
    }

    /// Sets the window size for window EM.
    ///
    /// The window size is given in number of blocks and must be non-zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use winsfs::Window;
    /// let runner = Window::builder().window_size(3).build(10_000, [5]).unwrap();
    /// assert_eq!(runner.window_size(), 3);
    /// ```
    pub fn window_size(mut self, window_size: usize) -> Self {
        self.window_size = Some(window_size);
        self
    }
}

impl<const N: usize> Default for WindowBuilder<N> {
    fn default() -> Self {
        Self {
            sfs: None,
            block_specification: None,
            window_size: None,
            stopping_rule: None,
        }
    }
}

/// Error type associated with the window EM builder.
#[derive(Debug, PartialEq)]
pub enum WindowBuilderError<const N: usize> {
    /// Invalid number of blocks, either zero or larger than number of sites.
    InvalidBlocks { blocks: usize, sites: usize },
    /// Invalid block size, either zero or larger than number of sites.
    InvalidBlockSize { block_size: usize, sites: usize },
    /// Mismatch between shape of SFS and shape of input.
    ShapeMismatch {
        sfs_shape: [usize; N],
        input_shape: [usize; N],
    },
    /// Invalid window size of zero.
    InvalidWindowSize { window_size: usize },
}

impl<const N: usize> fmt::Display for WindowBuilderError<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WindowBuilderError::InvalidBlocks { blocks, sites } => {
                write!(f, "can't split {sites} sites into {blocks} blocks")
            }
            WindowBuilderError::InvalidBlockSize { block_size, sites } => {
                write!(
                    f,
                    "can't split {sites} sites into blocks of size {block_size}"
                )
            }
            WindowBuilderError::InvalidWindowSize { window_size } => {
                write!(f, "invalid window size {window_size}")
            }
            WindowBuilderError::ShapeMismatch {
                sfs_shape,
                input_shape,
            } => {
                write!(
                    f,
                    "mismatch between SFS shape {sfs_shape:?} and input shape {input_shape:?}"
                )
            }
        }
    }
}

impl<const N: usize> Error for WindowBuilderError<N> {}

enum BlockSpecification {
    Blocks(usize),
    BlockSize(usize),
}

impl BlockSpecification {
    fn block_size<const N: usize>(self, sites: usize) -> Result<usize, WindowBuilderError<N>> {
        let block_size = match self {
            BlockSpecification::Blocks(blocks) => {
                if blocks == 0 || blocks > sites {
                    return Err(WindowBuilderError::InvalidBlocks { blocks, sites });
                }

                sites / blocks
            }
            BlockSpecification::BlockSize(block_size) => block_size,
        };

        if block_size == 0 || block_size > sites {
            Err(WindowBuilderError::InvalidBlockSize { block_size, sites })
        } else {
            Ok(block_size)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blocks_to_block_size() {
        assert_eq!(BlockSpecification::Blocks(10).block_size::<1>(100), Ok(10));
        assert_eq!(BlockSpecification::Blocks(10).block_size::<1>(101), Ok(10));
        assert_eq!(BlockSpecification::Blocks(10).block_size::<1>(109), Ok(10));
        assert_eq!(BlockSpecification::Blocks(10).block_size::<1>(110), Ok(11));
    }

    #[test]
    fn test_builder_error_on_shape_mismatch() {
        let result = Window::builder()
            .initial_sfs(Sfs::uniform([3]))
            .build(10, [4]);
        assert!(matches!(
            result,
            Err(WindowBuilderError::ShapeMismatch { .. })
        ));
    }
}
