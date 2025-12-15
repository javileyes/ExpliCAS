mod completer;
mod config;
mod health_suite;
pub mod repl;

use repl::Repl;

fn main() -> rustyline::Result<()> {
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
