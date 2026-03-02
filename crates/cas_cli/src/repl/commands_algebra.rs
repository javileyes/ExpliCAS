use super::*;

impl Repl {
    pub(crate) fn handle_det_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        match cas_solver::evaluate_det_command_lines_with_engine(
            &mut self.core.engine,
            line,
            Self::set_display_mode_from_verbosity(verbosity),
        ) {
            Ok(lines) => reply_output(lines.join("\n")),
            Err(message) => reply_output(message),
        }
    }

    pub(crate) fn handle_transpose_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        match cas_solver::evaluate_transpose_command_lines_with_engine(
            &mut self.core.engine,
            line,
            Self::set_display_mode_from_verbosity(verbosity),
        ) {
            Ok(lines) => reply_output(lines.join("\n")),
            Err(message) => reply_output(message),
        }
    }

    pub(crate) fn handle_trace_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        match cas_solver::evaluate_trace_command_lines_with_engine(
            &mut self.core.engine,
            line,
            Self::set_display_mode_from_verbosity(verbosity),
        ) {
            Ok(lines) => reply_output(lines.join("\n")),
            Err(message) => reply_output(message),
        }
    }

    /// Handle the 'telescope' command for proving telescoping identities like Dirichlet kernel
    pub(crate) fn handle_telescope_core(&mut self, line: &str) -> ReplReply {
        match cas_solver::evaluate_telescope_command_lines_with_engine(&mut self.core.engine, line)
        {
            Ok(lines) => reply_output(lines.join("\n")),
            Err(message) => reply_output(message),
        }
    }

    /// Handle the 'expand' command for aggressive polynomial expansion
    /// Uses the engine `expand()` path which distributes without educational guards
    pub(crate) fn handle_expand_core(&mut self, line: &str) -> ReplReply {
        // Delegate to normal line processing with expand() function wrapper.
        // This ensures steps are shown, consistent with using expand() as a function.
        let wrapped = match cas_solver::evaluate_expand_command_wrapped_line(line) {
            Ok(wrapped) => wrapped,
            Err(message) => return reply_output(message),
        };
        self.handle_eval_core(&wrapped)
    }

    /// Handle the 'expand_log' command for explicit logarithm expansion
    /// Expands ln(xy) → ln(x) + ln(y), ln(x/y) → ln(x) - ln(y), ln(x^n) → n*ln(x)
    pub(crate) fn handle_expand_log_core(&mut self, line: &str) -> ReplReply {
        match cas_solver::evaluate_expand_log_command_lines_with_engine(&mut self.core.engine, line)
        {
            Ok(lines) => reply_output(lines.join("\n")),
            Err(message) => reply_output(message),
        }
    }
}
