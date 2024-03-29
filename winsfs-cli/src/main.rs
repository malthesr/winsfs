use std::process;

use clap::Parser;

mod cli;
use cli::Cli;

mod estimate;

mod input;

mod log_likelihood;
pub use log_likelihood::LogLikelihood;

mod shuffle;
pub use shuffle::Shuffle;

mod split;
pub use split::Split;

mod stat;
pub use stat::Stat;

pub mod utils;

mod view;
pub use view::View;

fn main() {
    let args = Cli::parse();

    if args.debug {
        eprintln!("{args:#?}");
    }

    match utils::init_logger(args.verbose) {
        Ok(()) => (),
        Err(e) => eprintln!("{e}"),
    }

    let result = match args.subcommand {
        Some(command) => command.run(),
        None => args.run(),
    };

    match result {
        Ok(()) => (),
        Err(e) => {
            eprintln!("{e}");
            process::exit(1);
        }
    }
}
