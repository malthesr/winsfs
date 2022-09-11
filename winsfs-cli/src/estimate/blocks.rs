use std::num::NonZeroUsize;

use clap::{
    error::{ErrorKind, Result as ClapResult},
    CommandFactory,
};

use super::{Cli, DEFAULT_NUMBER_OF_BLOCKS};

/// A type representing the various ways of specifying block sizes.
pub enum BlockSpecification {
    /// A fixed number of blocks, can be converted to block size given the input size
    Number(NonZeroUsize),
    /// A fixed block size
    Size(NonZeroUsize),
}

impl BlockSpecification {
    /// Convert into block size and log block size and number of blocks.
    pub fn get_size(&self, sites: usize) -> ClapResult<NonZeroUsize> {
        let sites = NonZeroUsize::new(sites).ok_or_else(|| {
            Cli::command().error(ErrorKind::Io, "input contains 0 (intersecting) sites")
        })?;

        let block_size = self.block_size_inner(sites);

        if log::log_enabled!(log::Level::Debug) {
            let number_of_full_blocks = sites.get() / block_size.get();
            let last_block_size = sites.get() % block_size.get();

            log::debug!(
                target: "init",
                "Using {number_of_full_blocks} full blocks of size {block_size}"
            );
            if last_block_size != 0 {
                log::debug!(
                    target: "init",
                    "Last block has size {last_block_size}"
                );
            }
        }

        Ok(block_size)
    }

    /// Convert into block size.
    fn block_size_inner(&self, sites: NonZeroUsize) -> NonZeroUsize {
        match self {
            Self::Number(number_of_blocks) => {
                match NonZeroUsize::new(sites.get() / number_of_blocks.get()) {
                    Some(block_size) => block_size,
                    None => {
                        // Number of sites is lower than number of blocks,
                        // recursively try half and warn
                        let next = NonZeroUsize::new(number_of_blocks.get() / 2)
                            .unwrap_or(NonZeroUsize::new(1).unwrap());
                        let block_size = Self::Number(next).block_size_inner(sites);

                        log::warn!(
                            target: "init",
                            "Fewer sites than blocks, defaulting to block size {block_size};
                            consider checking input and/or setting hyperparameters manually"
                        );

                        block_size
                    }
                }
            }
            Self::Size(block_size) => *block_size,
        }
    }
}

impl From<&Cli> for BlockSpecification {
    fn from(args: &Cli) -> Self {
        match (args.blocks, args.block_size) {
            (Some(_), Some(_)) => unreachable!("checked by clap"),
            (Some(number_of_blocks), None) => Self::Number(number_of_blocks),
            (None, Some(block_size)) => Self::Size(block_size),
            (None, None) => Self::Number(NonZeroUsize::new(DEFAULT_NUMBER_OF_BLOCKS).unwrap()),
        }
    }
}
