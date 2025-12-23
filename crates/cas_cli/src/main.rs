//! CAS CLI - Computer Algebra System Command Line Interface
//!
//! Supports both interactive REPL mode (default) and JSON subcommands for scripting.

// Clippy allows for patterns that are difficult to refactor without breaking REPL
#![allow(clippy::manual_strip)] // strip_prefix pattern in command handling
#![allow(clippy::map_identity)] // .map(|x| x) for Option clone semantics

mod commands;
mod completer;
mod config;
mod format;
mod health_suite;
pub mod json_types;
pub mod repl;

use clap::{Parser, Subcommand};
use repl::Repl;
use std::env;

/// Rust CAS - Computer Algebra System
#[derive(Parser, Debug)]
#[command(name = "cas_cli")]
#[command(about = "A symbolic mathematics engine with step-by-step explanations")]
#[command(version)]
struct Cli {
    /// Use ASCII output (*, ^) instead of Unicode (·, ²)
    #[arg(long)]
    no_pretty: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Evaluate an expression and return JSON output
    EvalJson(commands::eval_json::EvalJsonArgs),
}

fn main() -> rustyline::Result<()> {
    let args: Vec<String> = env::args().collect();

    // Detect if first arg is a known subcommand → use clap
    // Otherwise → use manual parsing for backwards compatibility
    let is_subcommand = args.get(1).is_some_and(|arg| {
        matches!(
            arg.as_str(),
            "eval-json" | "script-json" | "mm-gcd-modp-json" | "help" | "--help" | "-h"
        )
    });

    if is_subcommand {
        // Use clap for JSON subcommands
        let cli = Cli::parse();

        // Configure pretty output
        if cli.no_pretty {
            cas_ast::display::disable_pretty_output();
        } else {
            cas_ast::display::enable_pretty_output();
        }

        // Handle subcommand
        if let Some(cmd) = cli.command {
            match cmd {
                Command::EvalJson(args) => {
                    commands::eval_json::run(args);
                }
            }
        }
        Ok(())
    } else {
        // Legacy manual parsing for REPL mode
        run_repl_mode()
    }
}

/// Run REPL mode with legacy argument parsing
fn run_repl_mode() -> rustyline::Result<()> {
    let args: Vec<String> = env::args().collect();

    let mut pretty_mode = true;

    for arg in &args[1..] {
        match arg.as_str() {
            "--no-pretty" => pretty_mode = false,
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            arg if arg.starts_with('-') => {
                eprintln!("Unknown option: {}", arg);
                eprintln!("Use --help for usage information");
                std::process::exit(1);
            }
            _ => {} // Ignore positional arguments for now
        }
    }

    // Configure pretty output based on CLI flag
    if pretty_mode {
        cas_ast::display::enable_pretty_output();
    } else {
        cas_ast::display::disable_pretty_output();
    }

    // Initialize tracing subscriber with WARN as default level
    // Use RUST_LOG=info or RUST_LOG=debug for more verbose output
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(tracing::Level::WARN.into())
                .from_env_lossy(),
        )
        .init();

    let mut repl = Repl::new();
    repl.run()
}

fn print_help() {
    println!("Rust CAS - Computer Algebra System");
    println!();
    println!("USAGE:");
    println!("    cas_cli [OPTIONS]                         # Start interactive REPL");
    println!("    cas_cli eval-json <EXPR> [OPTIONS]        # Evaluate and return JSON");
    println!();
    println!("REPL OPTIONS:");
    println!("    --no-pretty    Use ASCII output (*, ^) instead of Unicode (·, ²)");
    println!("    --help, -h     Print this help message");
    println!();
    println!("JSON SUBCOMMANDS:");
    println!("    eval-json      Evaluate an expression and return JSON");
    println!("    script-json    Execute a script from stdin and return JSON");
    println!("    mm-gcd-modp-json   Run mm_gcd benchmark and return JSON");
    println!();
    println!("EXAMPLES:");
    println!("    cas_cli                                   # Start REPL");
    println!("    cas_cli eval-json \"x^2 + 1\"               # Evaluate expression");
    println!("    cas_cli eval-json \"expand((x+1)^5)\" --max-chars 500");
    println!();
    println!("For detailed help on subcommands:");
    println!("    cas_cli eval-json --help");
}
