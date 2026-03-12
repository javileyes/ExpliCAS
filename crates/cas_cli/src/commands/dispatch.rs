//! Pure command-to-render dispatch for non-REPL CLI commands.

use super::output::CommandOutput;
use crate::Command;

pub(crate) fn render_command(command: Command) -> Result<Option<CommandOutput>, String> {
    match command {
        Command::Eval(args) => crate::commands::eval::render(args).map(Some),
        Command::Envelope(args) => Ok(Some(crate::commands::envelope::render(args))),
        Command::Limit(args) => crate::commands::limit::render(args).map(Some),
        Command::Substitute(args) => crate::commands::substitute::render(args).map(Some),
        Command::Repl => Ok(None),
    }
}
