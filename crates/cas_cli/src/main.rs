mod completer;
mod config;
mod health_suite;
pub mod repl;

use repl::Repl;

fn main() -> rustyline::Result<()> {
    // Initialize tracing subscriber to handle logs based on RUST_LOG env var
    tracing_subscriber::fmt::init();

    let mut repl = Repl::new();
    repl.run()
}
