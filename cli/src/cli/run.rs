#![allow(unstable_name_collisions)]

use std::{io, path::Path};

use clap::CommandFactory;

use winsfs::{
    saf::{BlockIterator, JointSaf, JointSafView, Saf},
    stream::{Header, Reader},
    Sfs, StoppingRule, Window,
};

use crate::utils::{get_rng, set_threads};

use super::Cli;

fn create_runner<const N: usize>(
    shape: [usize; N],
    sites: usize,
    args: &Cli,
) -> clap::Result<Window<N>> {
    let mut builder = Window::builder();

    if let Some(path) = &args.initial {
        log::info!(
            target: "init",
            "Reading initial SFS from path:\n\t{}",
            path.display()
        );

        let initial_sfs = Sfs::read_from_angsd(path)?.normalise();
        builder = builder.initial_sfs(initial_sfs);
    }

    match (args.block_size, args.blocks) {
        (None, None) => (),
        (None, Some(blocks)) => builder = builder.blocks(blocks.get()),
        (Some(block_size), None) => builder = builder.block_size(block_size.get()),
        (Some(_), Some(_)) => {
            unreachable!("clap checks '--blocks' and '--block-size' conflict")
        }
    }

    if let Some(window_size) = args.window_size {
        builder = builder.window_size(window_size.get())
    }

    match (args.max_epochs, args.tolerance) {
        (Some(n), Some(v)) => builder = builder.stopping_rule(StoppingRule::either(n, v)),
        (Some(n), None) => builder = builder.stopping_rule(StoppingRule::epochs(n)),
        (None, Some(v)) => builder = builder.stopping_rule(StoppingRule::log_likelihood(v)),
        (None, None) => (),
    }

    let runner = builder
        .build(sites, shape)
        .map_err(|e| Cli::command().error(clap::ErrorKind::ValueValidation, e))?;

    log::info!(
        target: "init",
        "Using {blocks} blocks, {block_size} sites per block, {window_size} blocks per window.",
        block_size = runner.block_size(),
        blocks = sites / runner.block_size(),
        window_size = runner.window_size(),
    );

    Ok(runner)
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
    let fmt_paths: Vec<String> = paths
        .iter()
        .map(|p| p.as_ref().display().to_string())
        .collect();
    log::info!(
        target: "init",
        "Reading and intersecting SAF files into memory:\n\t{}",
        fmt_paths.join("\n\t")
    );

    JointSaf::read_from_paths(paths).map_err(clap::Error::from)
}

pub fn run<const N: usize>(mut safs: JointSaf<N>, args: &Cli) -> clap::Result<()>
where
    for<'a> JointSafView<'a, N>: BlockIterator<'a, N, Block = JointSafView<'a, N>>,
{
    set_threads(args.threads)?;

    let shape = safs.shape();
    let sites = safs.sites();

    let fmt_shape: Vec<String> = shape.iter().map(|p| p.to_string()).collect();
    log::info!(
        target: "init",
        "Read {sites} sites with shape {}.",
        fmt_shape.join("/")
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

    let estimate = window.into_sfs().scale(sites as f64);

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
    let estimate = window.into_sfs().scale(sites as f64);

    println!("{}", estimate.format_angsd(None));

    Ok(())
}
