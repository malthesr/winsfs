use std::{io, path::PathBuf};

use clap::Args;

use winsfs_core::sfs::io::{npy, plain_text};

use crate::input;

/// View and modify site frequency spectrum.
#[derive(Args, Debug)]
pub struct View {
    /// Input SFS.
    ///
    /// The input SFS can be provided here or read from stdin.
    #[clap(parse(from_os_str), value_name = "PATH")]
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
    /// row-major order. Alternatively, the SFS can be written in the npy binary format.
    #[clap(short = 'o', long, arg_enum, default_value_t = input::sfs::Format::PlainText)]
    pub output_format: input::sfs::Format,
}

impl View {
    pub fn run(self) -> clap::Result<()> {
        let mut sfs = input::sfs::Reader::from_path_or_stdin(self.path)?.read_dyn()?;

        if self.normalise {
            sfs = sfs.normalise().into_unnormalised();
        }

        if self.fold {
            sfs = sfs.fold();
        }

        let stdout = io::stdout();
        let mut writer = stdout.lock();

        match self.output_format {
            input::sfs::Format::PlainText => plain_text::write_sfs(&mut writer, &sfs),
            input::sfs::Format::Npy => npy::write_sfs(&mut writer, &sfs),
        }
        .map_err(clap::Error::from)
    }
}
