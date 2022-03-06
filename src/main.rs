#![deny(unsafe_code)]

use std::process;

use clap::Parser;

mod cli;
use cli::{
    run_1d, run_2d,
    utils::{init_logger, set_threads},
    Cli,
};

mod saf;
pub use saf::{Saf1d, Saf2d};

mod sfs;
pub use sfs::{Sfs, Sfs1d, Sfs2d};

mod em;
pub use em::Window;

fn try_main(args: &Cli) -> clap::Result<()> {
    init_logger(args.verbose)?;
    set_threads(args.threads)?;

    match args.paths.as_slice() {
        [path] => run_1d(path, args),
        [first_path, second_path] => run_2d(first_path, second_path, args),
        _ => unreachable!(), // Checked by clap
    }
}

fn main() {
    let args = Cli::parse();

    if args.debug {
        eprintln!("{args:#?}");
    }

    match try_main(&args) {
        Ok(()) => (),
        Err(e) => {
            eprintln!("{e}");
            process::exit(1);
        }
    }
}
