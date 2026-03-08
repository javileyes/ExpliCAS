//! CAS CLI - Computer Algebra System Command Line Interface.
//!
//! Supports both interactive REPL mode (default) and subcommands for scripting.

mod cli_args;
mod commands;
mod completer;
pub mod repl;

pub use cli_args::*;

use clap::Parser;

fn main() -> rustyline::Result<()> {
    let cli = Cli::parse();
    commands::app::run(cli)
}
