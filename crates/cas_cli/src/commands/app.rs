//! CLI app orchestration (startup + command dispatch).

use super::output::CommandOutput;
use crate::{repl::Repl, Cli, Command};

/// Run the CLI application from parsed arguments.
pub fn run(cli: Cli) -> rustyline::Result<()> {
    configure_pretty_output(cli.no_pretty);
    init_tracing();
    dispatch_command(cli.command)
}

fn configure_pretty_output(no_pretty: bool) {
    if no_pretty {
        cas_formatter::display::disable_pretty_output();
    } else {
        cas_formatter::display::enable_pretty_output();
    }
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(tracing::Level::WARN.into())
                .from_env_lossy(),
        )
        .init();
}

fn dispatch_command(command: Option<Command>) -> rustyline::Result<()> {
    match command {
        Some(command) => match crate::commands::dispatch::render_command(command) {
            Ok(Some(output)) => {
                print_rendered(output);
                Ok(())
            }
            Ok(None) => {
                let mut repl = Repl::new();
                repl.run()
            }
            Err(message) => exit_with_error(message),
        },
        None => {
            let mut repl = Repl::new();
            repl.run()
        }
    }
}

fn print_rendered(output: CommandOutput) {
    println!("{}", output.stdout);
    for line in output.stderr_lines {
        eprintln!("{}", line);
    }
}

fn exit_with_error(message: String) -> ! {
    eprintln!("{}", message);
    std::process::exit(1);
}
