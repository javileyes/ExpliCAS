use super::*;

impl Repl {
    /// Handle "semantics" command - unified control for semantic axes
    pub(crate) fn handle_semantics_core(&mut self, line: &str) -> ReplReply {
        let out = cas_solver::evaluate_semantics_command_line(
            line,
            &mut self.core.simplify_options,
            self.core.state.options_mut(),
        );
        if out.sync_simplifier {
            self.sync_config_to_simplifier();
        }
        reply_output(out.lines.join("\n"))
    }
}
