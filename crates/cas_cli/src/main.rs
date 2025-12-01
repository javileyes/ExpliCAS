pub mod repl;
mod completer;
mod config;

use repl::Repl;

fn main() -> rustyline::Result<()> {
    // Initialize tracing subscriber to handle logs based on RUST_LOG env var
    tracing_subscriber::fmt::init();

    let mut repl = Repl::new();
    repl.run()
}
