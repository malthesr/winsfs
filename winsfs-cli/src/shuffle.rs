use std::path::{Path, PathBuf};

use clap::{error::Result as ClapResult, Args};

use winsfs_core::io::shuffle::{Header, Writer};

use crate::{input, utils::join};

/// Jointly pseudo-shuffle SAF files blockwise on disk.
///
/// This command prepares for running SFS estimation using constant memory by interleaving sites
/// from a number of blocks across the genome into a "pseudo-shuffled" file. Note that when running
/// joint SFS estimation, the files must also be shuffled jointly to ensure only intersecting sites
/// are considered.
#[derive(Args, Debug)]
pub struct Shuffle {
    /// Input SAF file paths.
    ///
    /// For each set of SAF files (conventially named 'prefix'.{saf.idx,saf.pos.gz,saf.gz}),
    /// specify either the shared prefix or the full path to any one member file.
    /// Up to three SAF files currently supported.
    #[clap(value_parser, num_args = 1..=3, required = true, value_name = "PATHS")]
    pub paths: Vec<PathBuf>,

    /// Output file path.
    ///
    /// The output destination must be known up front, since block pseudo-shuffling requires
    /// seeking within a known file.
    #[clap(short = 'o', long, value_parser, value_name = "PATH")]
    pub output: PathBuf,

    /// Number of blocks to use.
    #[clap(short = 'B', long, value_name = "INT", default_value_t = 100)]
    pub blocks: u16,

    /// Number of threads to use for reading.
    ///
    /// If set to 0, all available cores will be used.
    #[clap(short = 't', long, default_value_t = 4, value_name = "INT")]
    pub threads: usize,
}

impl Shuffle {
    pub fn run(self) -> ClapResult<()> {
        log::info!(
            target: "init",
            "Shuffling (intersecting) sites in input SAF files:\n\t{}",
            join(self.paths.iter().map(|p| p.display()), "\n\t"),
        );

        log::info!(
            target: "init",
            "Using {blocks} blocks and outputting to file:\n\t{path}",
            blocks = self.blocks,
            path = self.output.display(),
        );

        // Even though there are no const generics here, we split the implementation in two,
        // since 1d requires only one pass through the data, while higher dimensions must do
        // several to find the number of intersecting sites to pre-allocate for
        match &self.paths[..] {
            [] => unreachable!("checked by clap"),
            [p] => self.run_n([p]),
            [p1, p2] => self.run_n([p1, p2]),
            [p1, p2, p3] => self.run_n([p1, p2, p3]),
            _ => unreachable!(), // Checked by clap
        }
    }

    fn run_n<const N: usize, P>(&self, paths: [P; N]) -> ClapResult<()>
    where
        P: AsRef<Path>,
    {
        let readers = input::saf::Readers::from_member_paths(&paths, self.threads)?;
        let shape = readers.shape();

        // In 2D we cannot know the number of intersecting sites ahead of time,
        // so we must do a full pass through the data to count; in 1D this is not necessary
        // this is handled in `conut_sites` by checking the number of readers
        let sites = readers.count_sites()?;

        let header = Header::new(sites, shape.into(), usize::from(self.blocks));

        log::info!(
            target: "init",
            "Pre-allocating {bytes} bytes on disk for {sites} for populations with shape {shape}",
            bytes = header.file_size(),
            sites = header.sites(),
            shape = join(header.shape(), "/")
        );

        // Readers were consumed by counting sites above, so recreate.
        let readers = input::saf::Readers::from_member_paths(&paths, self.threads)?;

        let writer = Writer::create(&self.output, header)?;

        readers.shuffle(writer).map_err(|e| e.into())
    }
}
