use std::path::{Path, PathBuf};

use angsd_saf as saf;

use clap::Args;

use winsfs_core::io::{
    shuffle::{Header, Writer},
    Intersect,
};

use crate::utils::join;

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
    #[clap(
        parse(from_os_str),
        max_values = 3,
        required = true,
        value_name = "PATHS"
    )]
    pub paths: Vec<PathBuf>,

    /// Output file path.
    ///
    /// The output destination must be known up front, since block pseudo-shuffling requires
    /// seeking within a known file.
    #[clap(short = 'o', long, parse(from_os_str), value_name = "PATH")]
    pub output: PathBuf,

    /// Number of blocks to use.
    #[clap(short = 'B', long, value_name = "INT", default_value_t = 100)]
    pub blocks: u16,
}

impl Shuffle {
    pub fn run(self) -> clap::Result<()> {
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
            [path] => self.run_1(path),
            paths => self.run_n(paths),
        }
    }

    fn run_1<P>(&self, path: P) -> clap::Result<()>
    where
        P: AsRef<Path>,
    {
        let mut reader = saf::ReaderV3::from_member_path(&path)?;

        // In 1D we can get the expected number of sites directly from the SAF file index
        let index = reader.index();
        let sites = index.total_sites();
        let width = index.alleles() + 1;
        let header = Header::new(sites, vec![width], usize::from(self.blocks));

        log_header(&header);

        let mut writer = Writer::create(&self.output, header)?;

        let mut buf = vec![0.0; index.alleles() + 1].into();
        while reader.read_item(&mut buf)?.is_not_done() {
            writer.write_site(&buf)?;
        }

        writer.try_finish().map_err(clap::Error::from)
    }

    fn run_n<P>(&self, paths: &[P]) -> clap::Result<()>
    where
        P: AsRef<Path>,
    {
        let mut intersect = Intersect::from_paths(paths)?.into_inner();
        let shape = intersect
            .get_readers()
            .iter()
            .map(|reader| reader.index().alleles() + 1)
            .collect::<Vec<usize>>();

        // In 2D wee cannot know the number of intersecting sites ahead of time,
        // so we must do a full pass through the data to count
        let mut sites = 0;
        let mut bufs = intersect.create_record_bufs();
        while intersect.read_records(&mut bufs)?.is_not_done() {
            sites += 1;
        }

        let header = Header::new(sites, shape, usize::from(self.blocks));

        log_header(&header);

        let mut writer = Writer::create(&self.output, header)?;

        // Rewind readers for reuse
        for reader in intersect.get_readers_mut() {
            reader.seek(0)?;
        }

        while intersect.read_records(&mut bufs)?.is_not_done() {
            writer.write_disjoint_site(bufs.iter().map(|rec| rec.item()))?;
        }

        writer.try_finish().map_err(clap::Error::from)
    }
}

fn log_header(header: &Header) {
    log::info!(
        target: "init",
        "Pre-allocating {bytes} bytes on disk for {sites} for populations with shape {shape}",
        bytes = header.file_size(),
        sites = header.sites(),
        shape = join(header.shape(), "/")
    );
}
