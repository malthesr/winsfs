//! Read and write pseudo-shuffled SAF format.
//!
//! The aim of the format is not to give a true, random shuffle of the sites in the input SAF
//! file(s), but to move them around enough to break blocks of linkage disequilibrium. However,
//! this must be done in constant memory, i.e. streaming through the input.
//!
//! In overview, the strategy is to pre-allocate a file of the correct size, which must be known up
//! front, and then "split" this file into `B` blocks. Then consecutive writes go into consecutive
//! blocks to be spread across the file.
//!
//! # Examples
//!
//! ## Write
//!
//! ```no_run
//! # fn main() -> ::std::io::Result<()> {
//! use winsfs_core::io::shuffle::{Header, Writer};
//!
//! // This must be known up front; for a single SAF, this can be gotten from the index.
//! let sites = 100_000; // Number of sites in the input.
//! let shape = vec![9]; // Number of values per site (per population, just one here).
//! let blocks = 20;     // Number of blocks to use for pseudo-shuffle.
//! let header = Header::new(sites, shape, blocks);
//!
//! let mut writer = Writer::create("/path/to/saf.shuf", header)?;
//!
//! // Get sites from somewhere and write. The number of values must match the shape.
//! let site = vec![0.; 9];
//! for _ in 0..sites {
//!     writer.write_site(site.as_slice())?;
//! }
//!
//! // Writer expects exactly as many sites as provided in the header:
//! // any more will throw error in [`write_site`], any less will panic
//! // in the drop check; use try_finish to check for this error.
//! writer.try_finish()?;
//!
//! # Ok(()) }
//! ```

mod header;
pub use header::{Header, MAGIC_NUMBER};

mod reader;
pub use reader::Reader;

mod writer;
pub use writer::Writer;

/// Create checked conversion function.
macro_rules! impl_convert_to_fn {
    ($to:ty, $name:ident) => {
        /// Converts a value into a usize with checking.
        ///
        /// Panics if the conversion cannot be made exactly.
        pub(self) fn $name<T>(x: T) -> $to
        where
            $to: TryFrom<T>,
            T: Copy + std::fmt::Display,
        {
            match x.try_into() {
                Ok(v) => v,
                Err(_) => {
                    panic!(
                        "cannot convert {x} ({ty}) into {to}",
                        ty = std::any::type_name::<T>(),
                        to = stringify!($to),
                    )
                }
            }
        }
    };
}

impl_convert_to_fn!(u16, to_u16);
impl_convert_to_fn!(u32, to_u32);
impl_convert_to_fn!(u64, to_u64);
impl_convert_to_fn!(usize, to_usize);

#[cfg(test)]
mod tests {
    use super::*;

    use std::io;

    use tempfile::NamedTempFile;

    use crate::{io::ReadSite, saf::Site};

    #[test]
    fn test_write_then_read() -> io::Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path();

        let header = Header::new(9, vec![1, 4], 4);
        let mut writer = Writer::create(path, header.clone())?;

        let sites = vec![
            &[0., 0., 0., 0., 0.],
            &[1., 1., 1., 1., 1.],
            &[2., 2., 2., 2., 2.],
            &[3., 3., 3., 3., 3.],
            &[4., 4., 4., 4., 4.],
            &[5., 5., 5., 5., 5.],
            &[6., 6., 6., 6., 6.],
            &[7., 7., 7., 7., 7.],
            &[8., 8., 8., 8., 8.],
        ];

        for &site in sites.iter() {
            writer.write_site(site)?;
        }

        // Drop the writer to flush
        writer.try_finish().unwrap();

        let mut reader = Reader::from_path(path)?;

        let expected_order = &[0, 4, 8, 1, 5, 2, 6, 3, 7];

        let mut site = Site::new(vec![0.; 5], [5]).unwrap();
        for &i in expected_order {
            reader.read_site(site.as_mut_slice())?;

            // Should be normalised, e.g. exp'd
            let expected_site = sites[i].iter().map(|x| x.exp()).collect::<Vec<_>>();
            assert_eq!(site.as_slice(), expected_site.as_slice());
        }

        assert!(reader.read_site(site.as_mut_slice())?.is_done());

        file.close()
    }
}
