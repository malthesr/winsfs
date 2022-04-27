use std::path::Path;

use angsd_io::saf;

use clap::CommandFactory;

use super::{
    utils::{get_block_size_and_blocks, get_rng, get_window_size, set_threads, validate_shape},
    Cli,
};

use crate::{
    em::{Em, StoppingRule, Window, DEFAULT_TOLERANCE},
    Saf1d, Saf2d, Sfs,
};

fn create_runner<const N: usize>(
    shape: [usize; N],
    sites: usize,
    args: &Cli,
) -> clap::Result<Window<N>>
where
    Sfs<N>: Em<N>,
{
    let initial_sfs = if let Some(path) = &args.initial {
        let mut sfs = Sfs::read_from_angsd(path)?;
        validate_shape(sfs.shape(), shape)?;
        sfs.normalise();
        sfs
    } else {
        Sfs::uniform(shape)
    };

    let (block_size, blocks) = get_block_size_and_blocks(args.block_size, args.blocks, sites);
    let window_size = get_window_size(args.window_size, blocks);

    log::info!(
        target: "init",
        "Using window size {window_size}/{blocks} blocks ({block_size} sites per block)."
    );

    let stopping_rule = match (args.max_epochs, args.tolerance) {
        (Some(n), Some(v)) => StoppingRule::either(n, v),
        (Some(n), None) => StoppingRule::epochs(n),
        (None, Some(v)) => StoppingRule::log_likelihood(v),
        (None, None) => StoppingRule::log_likelihood(DEFAULT_TOLERANCE),
    };

    Ok(Window::new(
        initial_sfs,
        window_size,
        block_size,
        stopping_rule,
    ))
}

macro_rules! run {
    ($saf:ident, $args:ident, $sites:ident) => {{
        set_threads($args.threads)?;

        let mut rng = get_rng($args.seed);
        $saf.shuffle(&mut rng);

        let mut window = create_runner($saf.cols(), $saf.sites(), $args)?;
        window.em(&$saf.values());
        let mut estimate = window.into_sfs();

        estimate.scale($sites as f64);

        println!("{}", estimate.format_angsd(None));
    }};
}

pub fn run_1d<P>(path: P, args: &Cli) -> clap::Result<()>
where
    P: AsRef<Path>,
{
    log::info!(target: "init", "Reading SAF file into memory:\n\t{}", path.as_ref().display());

    let reader = saf::Reader::from_bgzf_member_path(path)?;

    let mut saf = Saf1d::read(reader)?;
    let sites = saf.sites();
    log::info!(target: "init", "Read {sites} sites in SAF file with {} cols.", saf.cols()[0]);

    run!(saf, args, sites);

    Ok(())
}

pub fn run_2d<P>(first_path: P, second_path: P, args: &Cli) -> clap::Result<()>
where
    P: AsRef<Path>,
{
    log::info!(
        target: "init",
        "Reading and intersecting SAF files into memory:\n\t{}\n\t{}",
        first_path.as_ref().display(), second_path.as_ref().display()
    );

    let first_reader = saf::Reader::from_bgzf_member_path(&first_path)?;
    let second_reader = saf::Reader::from_bgzf_member_path(&second_path)?;

    let mut safs = Saf2d::read(first_reader, second_reader)?;
    let sites = safs.sites();
    log::info!(target: "init", "Read {sites} shared sites in SAF files with {}/{} cols.", safs.cols()[0], safs.cols()[1]);

    run!(safs, args, sites);

    Ok(())
}

macro_rules! run_io {
    ($reader:ident, $header:ident, $args:ident, $shape:expr, $sites:ident) => {{
        let mut window = create_runner($shape, $sites, $args)?;

        window.em_io(&mut $reader, &$header)?;
        let mut estimate = window.into_sfs();

        estimate.scale($sites as f64);

        println!("{}", estimate.format_angsd(None));
    }};
}

pub fn run_io<P>(path: P, args: &Cli) -> clap::Result<()>
where
    P: AsRef<Path>,
{
    log::info!(
        target: "init",
        "Streaming pseudo-shuffled SAF file:\n\t{}",
        path.as_ref().display()
    );

    let mut reader = crate::io::Reader::from_path(path)?;
    let header = reader.read_header()?;

    let alleles = header
        .alleles()
        .iter()
        .map(|x| (x + 1) as usize)
        .collect::<Vec<_>>();
    let sites = header.sites() as usize;

    log::info!(
        target: "init",
        "Streaming {sites} sites in pseudo-shuffled SAF file with {} cols and {} blocks.",
        alleles.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("/"), header.blocks()
    );

    match alleles[..] {
        [n] => run_io!(reader, header, args, [n], sites),
        [n, m] => run_io!(reader, header, args, [n, m], sites),
        _ => {
            return Err(Cli::command().error(
                clap::ErrorKind::InvalidValue,
                "max two dimensions currently supported",
            ))
        }
    }

    Ok(())
}
