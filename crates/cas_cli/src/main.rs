mod completer;
mod config;
mod health_suite;
pub mod repl;

use repl::Repl;
use std::env;

fn print_help() {
    println!("Rust CAS - Computer Algebra System");
    println!();
    println!("USAGE:");
    println!("    cas_cli [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    --no-pretty    Use ASCII output (*, ^) instead of Unicode (·, ²)");
    println!("    --help, -h     Print this help message");
    println!();
    println!("EXAMPLES:");
    println!("    cas_cli              # Start REPL with pretty Unicode output");
    println!("    cas_cli --no-pretty  # Start REPL with ASCII output");
}

fn main() -> rustyline::Result<()> {
    let args: Vec<String> = env::args().collect();

    // Parse CLI arguments
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
