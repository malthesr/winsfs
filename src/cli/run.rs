use std::{fs, io, path::Path};

use angsd_io::saf;

use clap::CommandFactory;

// Include the run functions created by build.rs
include!(concat!(env!("OUT_DIR"), "/run.rs"));

use super::{
    utils::{get_block_size, get_blocks, get_rng, get_window_size},
    Cli,
};

use crate::{Em, Saf1d, Saf2d, Sfs, Sfs1d, Sfs2d};

fn run_1d_inner<const N: usize, R>(reader: saf::BgzfReader<R>, args: &Cli) -> clap::Result<()>
where
    R: io::BufRead + io::Seek,
{
    let mut saf = Saf1d::<N>::read(reader)?;
    let sites = saf.len();
    log::info!(target: "init", "Read {sites} sites in SAF files with dimensions {N}.");

    let mut rng = get_rng(args.seed);
    saf.shuffle(&mut rng);

    let (block_size, window_size) = setup(sites, args);

    let init = Sfs1d::<N>::uniform();
    let mut est = init.window_em(&saf, window_size, block_size, args.epochs);
    est.scale(sites as f64);

    println!("{est}");

    Ok(())
}

fn run_2d_inner<const N: usize, const M: usize>(
    first_reader: saf::BgzfReader<io::BufReader<fs::File>>,
    second_reader: saf::BgzfReader<io::BufReader<fs::File>>,
    args: &Cli,
) -> clap::Result<()> {
    let mut safs = Saf2d::<N, M>::read(first_reader, second_reader)?;
    let sites = safs.len();
    log::info!(target: "init", "Read {sites} shared sites in SAF files with dimensions {N}/{M}.");

    let mut rng = get_rng(args.seed);
    safs.shuffle(&mut rng);

    let (block_size, window_size) = setup(sites, args);

    let init = Sfs2d::<N, M>::uniform();
    let mut est = init.window_em(&safs, window_size, block_size, args.epochs);
    est.scale(sites as f64);

    println!("{est}");

    Ok(())
}

fn setup(sites: usize, args: &Cli) -> (usize, usize) {
    let block_size = get_block_size(args.block_size, sites);
    let blocks = get_blocks(block_size, sites);
    let window_size = get_window_size(args.window_size, blocks);

    log::info!(
        target: "init",
        "Using window size {window_size}/{blocks} blocks ({block_size} sites per block)."
    );

    (block_size, window_size)
}
