use std::path::Path;

use angsd_io::saf;

use super::{
    utils::{get_block_size_and_blocks, get_rng, get_window_size},
    Cli,
};

use crate::{Saf1d, Saf2d, Sfs1d, Sfs2d};

macro_rules! run {
    ($sfs:ident, $saf:ident, $args:ident, $sites:ident) => {{
        let mut estimate = if $args.vanilla {
            $sfs.em(&$saf, $args.epochs)
        } else {
            let mut rng = get_rng($args.seed);
            $saf.shuffle(&mut rng);

            let (block_size, blocks) =
                get_block_size_and_blocks($args.block_size, $args.blocks, $sites);
            let window_size = get_window_size($args.window_size, blocks);

            log::info!(
                target: "init",
                "Using window size {window_size}/{blocks} blocks ({block_size} sites per block)."
            );

            $sfs.window_em(&$saf, window_size, block_size, $args.epochs)
        };

        estimate.scale($sites as f64);

        println!("{estimate}");
    }}
}

pub fn run_1d<P>(path: P, args: &Cli) -> clap::Result<()>
where
    P: AsRef<Path>,
{
    let reader = saf::Reader::from_bgzf_member_path(path)?;

    let mut saf = Saf1d::read(reader)?;
    let sites = saf.sites();
    let cols = saf.cols();
    log::info!(target: "init", "Read {sites} sites in SAF files with dimensions {cols}.");

    let initial_sfs = Sfs1d::uniform([cols]);

    run!(initial_sfs, saf, args, sites);

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

    let initial_sfs = Sfs2d::uniform([first_cols, second_cols]);

    run!(initial_sfs, safs, args, sites);

    Ok(())
}
