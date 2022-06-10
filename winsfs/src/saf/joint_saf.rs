use std::{error::Error, fmt, io, path::Path};

use angsd_io::saf;

use rand::Rng;

use crate::ArrayExt;

use super::{Saf, SafView};

macro_rules! impl_shared_joint_saf_methods {
    () => {
        pub fn shape(&self) -> [usize; N] {
            self.inner.each_ref().map(|x| x.shape())
        }

        pub fn sites(&self) -> usize {
            self.inner[0].sites()
        }
    };
}

#[derive(Clone, Debug, PartialEq)]
pub struct JointSaf<const N: usize> {
    inner: [Saf; N],
}

impl<const N: usize> JointSaf<N> {
    pub fn as_array(&self) -> &[Saf; N] {
        &self.inner
    }

    pub fn new(safs: [Saf; N]) -> Result<Self, JointShapeError<N>> {
        let sites = safs.each_ref().map(|saf| saf.sites());
        let all_sites_equal = sites.windows(2).map(|x| x[0] == x[1]).all(|x| x);

        if all_sites_equal && N > 0 {
            Ok(Self::new_unchecked(safs))
        } else {
            Err(JointShapeError { sites })
        }
    }

    pub(super) fn new_unchecked(safs: [Saf; N]) -> Self {
        assert!(N > 0, "no SAFs provided");

        Self { inner: safs }
    }

    pub fn shuffle<R>(&mut self, rng: &mut R)
    where
        R: Rng,
    {
        // Modified from rand::seq::SliceRandom::shuffle
        for i in (1..self.sites()).rev() {
            let j = rng.gen_range(0..i + 1);

            for n in 0..N {
                self.inner[n].swap_sites(i, j);
            }
        }
    }

    pub fn read<R>(readers: [saf::BgzfReader<R>; N]) -> io::Result<Self>
    where
        R: io::BufRead + io::Seek,
    {
        let max_sites = readers
            .iter()
            .map(|reader| reader.index().total_sites())
            .min()
            .expect("no readers provided");

        let shape = readers
            .each_ref()
            .map(|reader| reader.index().alleles() + 1);

        let mut vecs = shape.map(|s| Vec::with_capacity(s * max_sites));

        let mut intersect =
            saf::reader::Intersect::new(Vec::from(readers)).expect("no readers provided");

        let mut bufs = intersect.create_record_bufs();
        while intersect.read_records(&mut bufs)?.is_not_done() {
            for (buf, vec) in bufs.iter().zip(vecs.iter_mut()) {
                vec.extend_from_slice(buf.values());
            }
        }

        for vec in vecs.iter_mut() {
            vec.shrink_to_fit();
        }

        let safs = vecs
            .into_iter()
            .zip(shape.iter())
            .map(|(vec, &shape)| Saf::from_log(vec, shape))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .unwrap();

        Self::new(safs).map_err(io::Error::from)
    }

    pub fn read_from_paths<P>(paths: [P; N]) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        // TODO: Use array::try_map when stable here
        let readers: [_; N] = paths
            .iter()
            .map(saf::Reader::from_bgzf_member_path)
            .collect::<io::Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| ()) // Reader is not debug, so this is necessary to unwrap
            .unwrap();

        Self::read(readers)
    }

    pub fn view(&self) -> JointSafView<N> {
        JointSafView::new_unchecked(self.inner.each_ref().map(|saf| saf.view()))
    }

    impl_shared_joint_saf_methods!();
}

impl From<Saf> for JointSaf<1> {
    fn from(saf: Saf) -> Self {
        Self::new_unchecked([saf])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointSafView<'a, const N: usize> {
    inner: [SafView<'a>; N],
}

impl<'a, const N: usize> JointSafView<'a, N> {
    pub fn as_array(&self) -> &[SafView<'a>; N] {
        &self.inner
    }

    pub fn new(safs: [SafView<'a>; N]) -> Result<Self, JointShapeError<N>> {
        let sites = safs.each_ref().map(|saf| saf.sites());
        let all_sites_equal = sites.windows(2).map(|x| x[0] == x[1]).all(|x| x);

        if all_sites_equal && N > 0 {
            Ok(Self::new_unchecked(safs))
        } else {
            Err(JointShapeError { sites })
        }
    }

    fn new_unchecked(safs: [SafView<'a>; N]) -> Self {
        Self { inner: safs }
    }

    pub fn safs(&self) -> [SafView<'a>; N] {
        self.inner
    }

    pub fn split_at_site(&self, site: usize) -> (Self, Self) {
        let split = self.inner.each_ref().map(|saf| saf.split_at_site(site));

        (
            Self::new_unchecked(split.map(|(hd, _)| hd)),
            Self::new_unchecked(split.map(|(_, tl)| tl)),
        )
    }

    impl_shared_joint_saf_methods!();
}

#[derive(Clone, Debug)]
pub struct JointShapeError<const N: usize> {
    sites: [usize; N],
}

impl<const N: usize> fmt::Display for JointShapeError<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if N > 0 {
            write!(
                f,
                "cannot construct joint SAF from SAFs with different number of sites {}",
                self.sites.map(|x| x.to_string()).join("/")
            )
        } else {
            write!(f, "cannot construct empty joint SAF",)
        }
    }
}

impl<const N: usize> Error for JointShapeError<N> {}

impl<const N: usize> From<JointShapeError<N>> for io::Error {
    fn from(e: JointShapeError<N>) -> Self {
        io::Error::new(io::ErrorKind::InvalidData, e)
    }
}
