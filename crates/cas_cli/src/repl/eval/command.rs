use super::super::*;

impl Repl {
    /// Core: handle eval command, returns ReplReply (no I/O)
    pub(crate) fn handle_eval_core(&mut self, line: &str) -> ReplReply {
        match cas_session::solver_exports::evaluate_eval_command_render_plan_on_repl_core(
            &mut self.core,
            line,
            self.verbosity == Verbosity::None,
        ) {
            Ok(plan) => self.render_eval_plan_to_reply(plan),
            Err(message) => vec![ReplMsg::error(message)],
        }
    }
}
