use std::path::Path;

use angsd_io::saf;

use super::{
    utils::{get_block_size, get_blocks, get_rng, get_window_size},
    Cli,
};

use crate::{Saf1d, Saf2d, Sfs1d, Sfs2d};

pub fn run_1d<P>(path: P, args: &Cli) -> clap::Result<()>
where
    P: AsRef<Path>,
{
    let reader = saf::Reader::from_bgzf_member_path(path)?;

    let mut saf = Saf1d::read(reader)?;
    let sites = saf.sites();
    let cols = saf.cols();
    log::info!(target: "init", "Read {sites} sites in SAF files with dimensions {cols}.");

    let mut rng = get_rng(args.seed);
    saf.shuffle(&mut rng);

    let (block_size, window_size) = setup(sites, args);

    let init = Sfs1d::uniform([cols]);
    let mut est = init.window_em(&saf, window_size, block_size, args.epochs);
    est.scale(sites as f64);

    println!("{est}");

    Ok(())
}

pub fn run_2d<P>(first_path: P, second_path: P, args: &Cli) -> clap::Result<()>
where
    P: AsRef<Path>,
{
    let first_reader = saf::Reader::from_bgzf_member_path(first_path)?;
    let second_reader = saf::Reader::from_bgzf_member_path(second_path)?;

    let mut safs = Saf2d::read(first_reader, second_reader)?;
    let sites = safs.sites();
    let [first_cols, second_cols] = safs.cols();
    log::info!(target: "init", "Read {sites} shared sites in SAF files with dimensions {first_cols}/{second_cols}.");

    let mut rng = get_rng(args.seed);
    safs.shuffle(&mut rng);

    let (block_size, window_size) = setup(sites, args);

    let init = Sfs2d::uniform([first_cols, second_cols]);
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
