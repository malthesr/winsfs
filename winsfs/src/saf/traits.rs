use std::mem::MaybeUninit;

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator};

use super::{Blocks, JointSaf, JointSafView};

pub trait ParSiteIterator<'a, const N: usize> {
    // TODO: These types should be replaced by TAIT once stable,
    // see github.com/rust-lang/rust/issues/63063
    // TODO: The trait lifetime should be a GAT once stable,
    // see github.com/rust-lang/rust/issues/44265
    type Site: IntoArray<N, &'a [f32]>;
    type SiteIter: IndexedParallelIterator<Item = Self::Site>;

    fn par_iter_sites(&self) -> Self::SiteIter;
}

impl<'a> ParSiteIterator<'a, 1> for JointSafView<'a, 1> {
    type Site = &'a [f32];
    type SiteIter = rayon::slice::Chunks<'a, f32>;

    fn par_iter_sites(&self) -> Self::SiteIter {
        self.safs()[0].par_iter_sites()
    }
}

impl<'a> ParSiteIterator<'a, 2> for JointSafView<'a, 2> {
    type Site = (&'a [f32], &'a [f32]);
    type SiteIter = rayon::iter::Zip<rayon::slice::Chunks<'a, f32>, rayon::slice::Chunks<'a, f32>>;

    fn par_iter_sites(&self) -> Self::SiteIter {
        let [fst, snd] = self.safs();

        fst.par_iter_sites().zip(snd.par_iter_sites())
    }
}

impl<'a> ParSiteIterator<'a, 3> for JointSafView<'a, 3> {
    type Site = (&'a [f32], &'a [f32], &'a [f32]);
    type SiteIter = rayon::iter::MultiZip<(
        rayon::slice::Chunks<'a, f32>,
        rayon::slice::Chunks<'a, f32>,
        rayon::slice::Chunks<'a, f32>,
    )>;

    fn par_iter_sites(&self) -> Self::SiteIter {
        let [fst, snd, trd] = self.safs();

        (
            fst.par_iter_sites(),
            snd.par_iter_sites(),
            trd.par_iter_sites(),
        )
            .into_par_iter()
    }
}

impl<'a, const N: usize> ParSiteIterator<'a, N> for &'a JointSaf<N>
where
    JointSafView<'a, N>: ParSiteIterator<'a, N>,
{
    type Site = <JointSafView<'a, N> as ParSiteIterator<'a, N>>::Site;
    type SiteIter = <JointSafView<'a, N> as ParSiteIterator<'a, N>>::SiteIter;

    fn par_iter_sites(&self) -> Self::SiteIter {
        self.view().par_iter_sites()
    }
}

pub trait BlockIterator<'a, const N: usize>: ParSiteIterator<'a, N> {
    // TODO: These types should be replaced by TAIT once stable,
    // see github.com/rust-lang/rust/issues/63063
    // TODO: The trait lifetime should be a GAT once stable,
    // see github.com/rust-lang/rust/issues/44265
    type Block: BlockIterator<'a, N>;
    type BlockIter: ExactSizeIterator<Item = Self::Block>;

    fn iter_blocks(&self, block_size: usize) -> Self::BlockIter;
}

impl<'a, const N: usize> BlockIterator<'a, N> for JointSafView<'a, N>
where
    JointSafView<'a, N>: ParSiteIterator<'a, N>,
{
    type Block = JointSafView<'a, N>;
    type BlockIter = Blocks<'a, N>;

    fn iter_blocks(&self, block_size: usize) -> Self::BlockIter {
        Blocks::new(*self, block_size)
    }
}

impl<'a, const N: usize> BlockIterator<'a, N> for &'a JointSaf<N>
where
    JointSafView<'a, N>: BlockIterator<'a, N>,
{
    type Block = <JointSafView<'a, N> as BlockIterator<'a, N>>::Block;
    type BlockIter = <JointSafView<'a, N> as BlockIterator<'a, N>>::BlockIter;

    fn iter_blocks(&self, block_size: usize) -> Self::BlockIter {
        self.view().iter_blocks(block_size)
    }
}

pub trait IntoArray<const N: usize, T> {
    fn into_array(self) -> [T; N];
}

impl<T> IntoArray<1, T> for T {
    fn into_array(self) -> [T; 1] {
        [self]
    }
}

impl<T> IntoArray<2, T> for (T, T) {
    fn into_array(self) -> [T; 2] {
        [self.0, self.1]
    }
}

impl<T> IntoArray<3, T> for (T, T, T) {
    fn into_array(self) -> [T; 3] {
        [self.0, self.1, self.2]
    }
}

impl<const N: usize, T> IntoArray<N, T> for [T; N] {
    fn into_array(self) -> [T; N] {
        self
    }
}

pub trait ArrayExt<const N: usize, T> {
    fn each_ref(&self) -> [&T; N];

    fn each_mut(&mut self) -> [&mut T; N];
}

impl<const N: usize, T> ArrayExt<N, T> for [T; N] {
    // TODO: Use each_ref when stable,
    // see github.com/rust-lang/rust/issues/76118
    fn each_ref(&self) -> [&T; N] {
        // Adapted from code in tracking issue, see above.
        let mut out: MaybeUninit<[&T; N]> = MaybeUninit::uninit();

        let buf = out.as_mut_ptr() as *mut &T;
        let mut refs = self.iter();

        for i in 0..N {
            unsafe { buf.add(i).write(refs.next().unwrap()) }
        }

        unsafe { out.assume_init() }
    }

    // TODO: Use each_mut when stable,
    // see github.com/rust-lang/rust/issues/76118
    fn each_mut(&mut self) -> [&mut T; N] {
        // Adapted from code in tracking issue, see above.
        let mut out: MaybeUninit<[&mut T; N]> = MaybeUninit::uninit();

        let buf = out.as_mut_ptr() as *mut &mut T;
        let mut refs = self.iter_mut();

        for i in 0..N {
            unsafe { buf.add(i).write(refs.next().unwrap()) }
        }

        unsafe { out.assume_init() }
    }
}
