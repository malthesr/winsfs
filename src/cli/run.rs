#![allow(unstable_name_collisions)]

use std::{io, path::Path};

use clap::CommandFactory;

use super::{
    utils::{get_block_size_and_blocks, get_rng, get_window_size, set_threads, validate_shape},
    Cli,
};

use crate::{
    em::{Em, StoppingRule, Window, DEFAULT_TOLERANCE},
    saf::{ArrayExt, BlockIterator, JointSaf, JointSafView, Saf},
    stream::{Header, Reader},
    Sfs,
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

pub fn read_saf<P>(path: P) -> clap::Result<JointSaf<1>>
where
    P: AsRef<Path>,
{
    log::info!(
        target: "init",
        "Reading SAF file into memory:\n\t{}",
        path.as_ref().display()
    );

    Saf::read_from_path(path)
        .map(|saf| JointSaf::new([saf]).expect("joint SAF constructor cannot fail with single SAF"))
        .map_err(clap::Error::from)
}

pub fn read_safs<const N: usize, P>(paths: [P; N]) -> clap::Result<JointSaf<N>>
where
    P: AsRef<Path>,
{
    log::info!(
        target: "init",
        "Reading and intersecting SAF files into memory:\n\t{}",
        paths.each_ref().map(|p| p.as_ref().display().to_string()).join("\n\t")
    );

    JointSaf::read_from_paths(paths).map_err(clap::Error::from)
}

pub fn run<const N: usize>(mut safs: JointSaf<N>, args: &Cli) -> clap::Result<()>
where
    Sfs<N>: Em<N>,
    for<'a> JointSafView<'a, N>: BlockIterator<'a, N, Block = JointSafView<'a, N>>,
{
    set_threads(args.threads)?;

    let shape = safs.shape();
    let sites = safs.sites();

    log::info!(
        target: "init",
        "Read {sites} sites with shape {}.",
        shape.each_ref().map(|p| p.to_string()).join("/")
    );

    let mut rng = get_rng(args.seed);
    log::info!(
        target: "init",
        "Shuffling sites with {} seed.",
        args.seed.map_or_else(|| String::from("random"), |s| s.to_string())
    );
    safs.shuffle(&mut rng);

    let mut window = create_runner(shape, sites, args)?;
    window.em(&safs.view());
    let mut estimate = window.into_sfs();

    estimate.scale(sites as f64);

    println!("{}", estimate.format_angsd(None));

    Ok(())
}

pub fn streaming_run<P>(path: P, args: &Cli) -> clap::Result<()>
where
    P: AsRef<Path>,
{
    log::info!(
        target: "init",
        "Streaming pseudo-shuffled SAF file:\n\t{}",
        path.as_ref().display()
    );

    let mut reader = Reader::from_path(path)?;
    let header = reader.read_header()?;

    match header.alleles().len() {
        1 => streaming_run_inner::<1, _>(reader, header, args),
        2 => streaming_run_inner::<2, _>(reader, header, args),
        3 => streaming_run_inner::<3, _>(reader, header, args),
        _ => {
            return Err(Cli::command().error(
                clap::ErrorKind::InvalidValue,
                "max three dimensions currently supported; \
                if you are affected by this, please open an issue",
            ))
        }
    }
}

fn streaming_run_inner<const N: usize, R>(
    mut reader: Reader<R>,
    header: Header,
    args: &Cli,
) -> clap::Result<()>
where
    R: io::BufRead + io::Seek,
    Sfs<N>: Em<N>,
{
    let shape: [usize; N] = header
        .alleles()
        .iter()
        .map(|x| (x + 1) as usize)
        .collect::<Vec<_>>()
        .try_into()
        .expect("unexpected number of alleles in header");
    let sites = header.sites() as usize;
    let blocks = header.blocks();

    log::info!(
        target: "init",
        "Streaming {sites} sites in pseudo-shuffled SAF file with shape {} and {blocks} blocks.",
        shape.map(|x| x.to_string()).join("/")
    );

    let mut window = create_runner(shape, sites, args)?;
    window.streaming_em(&mut reader, &header)?;
    let mut estimate = window.into_sfs();

    estimate.scale(sites as f64);

    println!("{}", estimate.format_angsd(None));

    Ok(())
}
