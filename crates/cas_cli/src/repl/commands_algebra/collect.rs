use super::*;

impl Repl {
    /// Handle the `collect` command by wrapping it into the explicit
    /// `collect(expr, var)` engine path.
    pub(crate) fn handle_collect_core(&mut self, line: &str) -> ReplReply {
        match cas_solver::session_api::eval::evaluate_collect_command_render_plan_on_repl_core(
            &mut self.core,
            line,
            self.verbosity == Verbosity::None,
        ) {
            Ok(plan) => self.render_eval_plan_to_reply(plan),
            Err(message) => reply_output(message),
        }
    }
}
