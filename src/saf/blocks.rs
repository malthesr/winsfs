use std::cmp;

use super::JointSafView;

#[derive(Debug)]
pub struct Blocks<'a, const N: usize> {
    safs: JointSafView<'a, N>,
    sites_per_block: usize,
}

impl<'a, const N: usize> Blocks<'a, N> {
    pub(super) fn new(safs: JointSafView<'a, N>, sites_per_block: usize) -> Self {
        Self {
            safs,
            sites_per_block,
        }
    }
}

impl<'a, const N: usize> Iterator for Blocks<'a, N> {
    type Item = JointSafView<'a, N>;

    fn next(&mut self) -> Option<Self::Item> {
        let sites = self.safs.sites();

        if sites == 0 {
            None
        } else {
            let sites_per_block = cmp::min(sites, self.sites_per_block);
            let (hd, tl) = self.safs.split_at_site(sites_per_block);

            self.safs = tl;

            Some(hd)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let sites = self.safs.sites();

        if sites == 0 {
            (0, Some(0))
        } else {
            let n = sites / self.sites_per_block;
            let rem = sites % self.sites_per_block;
            let n = if rem > 0 { n + 1 } else { n };

            (n, Some(n))
        }
    }
}

impl<'a, const N: usize> ExactSizeIterator for Blocks<'a, N> {}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::saf::{JointSaf, Saf};

    #[test]
    fn test_block_iter() {
        let joint_saf = JointSaf::new_unchecked([
            Saf::new_unchecked(vec![1., 1., 1., 2., 2., 2., 3., 3., 3.], 3),
            Saf::new_unchecked(
                vec![1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3.],
                5,
            ),
        ]);

        let sites_per_block = 2;
        let mut blocks = Blocks::new(joint_saf.view(), sites_per_block);

        assert_eq!(blocks.len(), 2);

        let first = blocks.next().expect("no first block");
        assert_eq!(first.as_array()[0].as_slice(), &[1., 1., 1., 2., 2., 2.]);
        assert_eq!(
            first.as_array()[1].as_slice(),
            &[1., 1., 1., 1., 1., 2., 2., 2., 2., 2.]
        );

        let second = blocks.next().expect("no second block");
        assert_eq!(second.as_array()[0].as_slice(), &[3., 3., 3.]);
        assert_eq!(second.as_array()[1].as_slice(), &[3., 3., 3., 3., 3.]);

        assert!(blocks.next().is_none())
    }
}
