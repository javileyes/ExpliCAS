mod algebra;
mod analysis;
mod session;

use crate::repl_command_types::ReplCommandInput;

pub(crate) fn parse_repl_command_routing(line: &str) -> ReplCommandInput<'_> {
    if let Some(command) = session::try_parse_session_command(line) {
        return command;
    }

    if let Some(command) = analysis::try_parse_analysis_command(line) {
        return command;
    }

    if let Some(command) = algebra::try_parse_algebra_command(line) {
        return command;
    }

    ReplCommandInput::Eval(line)
}
