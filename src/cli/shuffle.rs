use std::{
    fs, io,
    path::{Path, PathBuf},
};

use angsd_io::saf;

use clap::{Args, CommandFactory};

use crate::stream::{Header, Writers};

use super::Cli;

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
    /// For each set of SAF files (conventially named [prefix].{saf.idx,saf.pos.gz,saf.gz}),
    /// specify either the shared prefix or the full path to any one member file.
    /// Up to two SAF files currently supported.
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
    #[clap(short = 'B', long, value_name = "INT", default_value_t = 20)]
    pub blocks: u64,
}

impl Shuffle {
    pub(crate) fn run(self) -> Result<(), clap::Error> {
        match self.paths.as_slice() {
            [] => unreachable!(), // Checked by clap
            [path] => run_1d(path, &self),
            paths => run_nd(paths, &self),
        }
    }
}

fn run_1d<P>(saf_path: P, args: &Shuffle) -> clap::Result<()>
where
    P: AsRef<Path>,
{
    let mut reader = saf::Reader::from_bgzf_member_path(saf_path)?;

    let all_sites = reader.index().total_sites() as u64;
    let sites = all_sites - (all_sites % args.blocks);
    let alleles = vec![reader.index().alleles() as u64];
    let header = Header::new(sites, alleles, args.blocks)
        .map_err(|e| Cli::command().error(clap::ErrorKind::Io, e))?;

    let mut writers = Writers::create(&args.output, &header)?;

    let mut buf = vec![0.0; reader.index().alleles() + 1];

    let mut i = 0;
    while i < sites / args.blocks {
        for writer in writers.get_mut().iter_mut() {
            if reader.value_reader_mut().read_values(&mut buf)?.is_done() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "number of sites in SAF file does not match index",
                )
                .into());
            }

            writer.write_values(&buf)?;
        }

        i += 1;
    }

    Ok(())
}

fn run_nd<P>(paths: &[P], args: &Shuffle) -> clap::Result<()>
where
    P: AsRef<Path>,
{
    let header = create_intersect_header(paths, args.blocks)?;
    let mut reader = create_intersect(paths)?;
    let mut writers = Writers::create(&args.output, &header)?;

    let mut bufs = reader.create_record_bufs();

    let mut i = 0;
    while i < header.sites() / header.blocks() {
        for writer in writers.get_mut().iter_mut() {
            if reader.read_records(&mut bufs)?.is_done() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "number of sites in SAF file intersection does not match previous estimate",
                )
                .into());
            }

            for buf in bufs.iter() {
                writer.write_values(buf.values())?;
            }
        }

        i += 1;
    }

    Ok(())
}

fn create_intersect_header<P>(paths: &[P], blocks: u64) -> clap::Result<Header>
where
    P: AsRef<Path>,
{
    let mut reader = create_intersect(paths)?;

    let mut all_sites = 0;
    let mut bufs = reader.create_record_bufs();
    while reader.read_records(&mut bufs)?.is_not_done() {
        all_sites += 1;
    }

    let alleles = reader
        .get_readers()
        .iter()
        .map(|r| r.index().alleles() as u64)
        .collect();

    let sites = all_sites - (all_sites % blocks);

    Header::new(sites, alleles, blocks).map_err(|e| Cli::command().error(clap::ErrorKind::Io, e))
}

fn create_intersect<P>(paths: &[P]) -> clap::Result<saf::reader::Intersect<io::BufReader<fs::File>>>
where
    P: AsRef<Path>,
{
    let readers = paths
        .iter()
        .map(saf::BgzfReader::from_bgzf_member_path)
        .collect::<io::Result<Vec<_>>>()?;

    Ok(saf::reader::Intersect::new(readers).expect("less than two readers for intersect"))
}
