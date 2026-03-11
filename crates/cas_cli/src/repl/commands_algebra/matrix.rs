use super::*;

impl Repl {
    pub(crate) fn handle_det_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        match cas_session::solver_exports::evaluate_det_command_message_on_repl_core(
            &mut self.core,
            line,
            Self::set_display_mode_from_verbosity(verbosity),
        ) {
            Ok(message) => reply_output(message),
            Err(message) => reply_output(message),
        }
    }

    pub(crate) fn handle_transpose_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        match cas_session::solver_exports::evaluate_transpose_command_message_on_repl_core(
            &mut self.core,
            line,
            Self::set_display_mode_from_verbosity(verbosity),
        ) {
            Ok(message) => reply_output(message),
            Err(message) => reply_output(message),
        }
    }

    pub(crate) fn handle_trace_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        match cas_session::solver_exports::evaluate_trace_command_message_on_repl_core(
            &mut self.core,
            line,
            Self::set_display_mode_from_verbosity(verbosity),
        ) {
            Ok(message) => reply_output(message),
            Err(message) => reply_output(message),
        }
    }
}
