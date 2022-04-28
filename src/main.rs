use std::process;

use clap::Parser;

mod cli;
use cli::Cli;

mod em;

pub mod saf;

mod sfs;
pub use sfs::{Sfs, Sfs1d, Sfs2d};

pub mod stream;

fn main() {
    let args = Cli::parse();

    if args.debug {
        eprintln!("{args:#?}");
    }

    match args.run() {
        Ok(()) => (),
        Err(e) => {
            eprintln!("{e}");
            process::exit(1);
        }
    }
}
