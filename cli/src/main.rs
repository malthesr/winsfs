use std::process;

use clap::Parser;

mod cli;
use cli::Cli;

pub mod utils;

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
