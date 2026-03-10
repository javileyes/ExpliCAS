//! CLI app orchestration (startup + command dispatch).

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
        Some(Command::Eval(args)) => {
            crate::commands::eval::run(args);
            Ok(())
        }
        Some(Command::EvalJson(args)) => {
            let eval_args = crate::commands::eval_json::from_legacy_eval_wire_args(args);
            crate::commands::eval_json::run(eval_args);
            Ok(())
        }
        Some(Command::EnvelopeJson(args)) => {
            crate::commands::envelope_json::run(args);
            Ok(())
        }
        Some(Command::Limit(args)) => {
            crate::commands::limit::run(args);
            Ok(())
        }
        Some(Command::Substitute(args)) => {
            crate::commands::substitute::run(args);
            Ok(())
        }
        Some(Command::Repl) | None => {
            let mut repl = Repl::new();
            repl.run()
        }
    }
}
