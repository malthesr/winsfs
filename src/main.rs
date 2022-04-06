#![deny(unsafe_code)]

use std::process;

use clap::Parser;

mod cli;
use cli::Cli;

mod saf;
pub use saf::{Saf1d, Saf2d};

mod sfs;
pub use sfs::{Sfs, Sfs1d, Sfs2d};

mod em;

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
