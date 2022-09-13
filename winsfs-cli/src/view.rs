use std::{io, path::PathBuf};

use clap::{error::Result as ClapResult, Args, ValueEnum};

use winsfs_core::sfs::{
    io::{npy, plain_text},
    DynUSfs, Multi,
};

use crate::input;

/// View and modify site frequency spectrum.
#[derive(Args, Debug)]
pub struct View {
    /// Input SFS.
    ///
    /// The input SFS can be provided here or read from stdin.
    #[clap(value_parser, value_name = "PATH")]
    pub path: Option<PathBuf>,

    /// Fold site frequency spectrum.
    ///
    /// When the data cannot be properly polarised, it does not make sense to distinguish between
    /// 0 variants and 2N variants (in the diploid case) at a particular site. To accommodate this,
    /// the spectrum can be folded by collapsing such indistinguishable combinations. The folding
    /// here is onto the "upper" part of the spectrum, and the lower part will be set to zero. On the
    /// diagonal (if any), the arithmetic mean is used.
    #[clap(short = 'f', long)]
    pub fold: bool,

    /// Normalise site frequency spectrum.
    ///
    /// Ensures that the values in the spectrum adds up to one.
    #[clap(short = 'n', long)]
    pub normalise: bool,

    /// Output format of the SFS.
    ///
    /// By default, the output SFS is written in a plain text format, where the first line is a
    /// header giving the shape of the SFS, and the second line gives the values of the SFS in flat
    /// row-major order. Alternatively, the SFS can be written in the npy/npz formats (depending
    /// on whether one or more SFS are provided).
    #[clap(short = 'o', long, value_enum, default_value_t = Format::Txt)]
    pub output_format: Format,
}

/// An SFS input format.
#[derive(ValueEnum, Clone, Debug, Eq, PartialEq)]
pub enum Format {
    /// Plain text format.
    Txt,
    /// Numpy npy/npz format.
    Np,
}

impl Format {
    /// Write provided SFS to writer.
    ///
    /// If format is np, the written format will be npy if only a single SFS is present, otherwise
    /// npz.
    fn write<W>(&self, writer: &mut W, multi: &Multi<DynUSfs>) -> io::Result<()>
    where
        W: io::Write,
    {
        match self {
            Self::Txt => plain_text::write_multi_sfs(writer, multi),
            Self::Np => {
                if let [sfs] = &multi[..] {
                    npy::write_sfs(writer, sfs)
                } else {
                    // Writing npz requires io::Seek, so we have to write the full output to a
                    // buffer before writing to stdout
                    let mut buf = io::Cursor::new(Vec::new());
                    npy::write_multi_sfs(&mut buf, multi)?;

                    writer.write_all(&buf.into_inner())
                }
            }
        }
    }
}

impl View {
    /// Process single SFS with the arguments provided.
    fn process(&self, mut sfs: DynUSfs) -> DynUSfs {
        if self.normalise {
            sfs = sfs.normalise().into_unnormalised();
        }

        if self.fold {
            sfs = sfs.fold();
        }

        sfs
    }

    /// Process all the provided SFS in with the arguments provided.
    fn process_all(&self, multi: Multi<DynUSfs>) -> Multi<DynUSfs> {
        Vec::from(multi)
            .into_iter()
            .map(|sfs| self.process(sfs))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    pub fn run(self) -> ClapResult<()> {
        let multi_sfs =
            input::sfs::Reader::from_path_or_stdin(self.path.as_ref())?.read_dyn_multi()?;

        let new_multi_sfs = self.process_all(multi_sfs);

        let stdout = io::stdout();
        let mut writer = stdout.lock();

        self.output_format
            .write(&mut writer, &new_multi_sfs)
            .map_err(clap::Error::from)
    }
}
