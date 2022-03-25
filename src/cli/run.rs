use std::path::Path;

use angsd_io::saf;

use super::{
    utils::{get_block_size_and_blocks, get_rng, get_window_size, validate_shape},
    Cli,
};

use crate::{Saf1d, Saf2d, Sfs};

macro_rules! run {
    ($saf:ident, $args:ident, $sites:ident) => {{
        let initial_sfs = if let Some(path) = &$args.initial {
            let mut sfs = Sfs::read_from_angsd(path)?;
            validate_shape(sfs.shape(), $saf.cols())?;
            sfs.normalise();
            sfs
        } else {
            Sfs::uniform($saf.cols())
        };

        let mut estimate = if $args.vanilla {
            initial_sfs.em(&$saf, $args.epochs)
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

            initial_sfs.window_em(&$saf, window_size, block_size, $args.epochs)
        };

        estimate.scale($sites as f64);

        println!("{}", estimate.format_angsd(None));
    }}
}

pub fn run_1d<P>(path: P, args: &Cli) -> clap::Result<()>
where
    P: AsRef<Path>,
{
    let reader = saf::Reader::from_bgzf_member_path(path)?;

    let mut saf = Saf1d::read(reader)?;
    let sites = saf.sites();
    log::info!(target: "init", "Read {sites} sites in SAF files with dimensions {}.", saf.cols()[0]);

    run!(saf, args, sites);

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
    log::info!(target: "init", "Read {sites} shared sites in SAF files with dimensions {}/{}.", safs.cols()[0], safs.cols()[1]);

    run!(safs, args, sites);

    Ok(())
}
