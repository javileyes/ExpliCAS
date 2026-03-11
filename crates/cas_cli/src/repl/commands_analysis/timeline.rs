use super::*;

impl Repl {
    pub(crate) fn handle_timeline_core(&mut self, line: &str) -> ReplReply {
        let eval_options = cas_session::solver_exports::eval_options_from_repl_core(&self.core);
        let actions = self.core.with_engine_and_state(|engine, state| {
            cas_didactic::evaluate_timeline_invocation_cli_actions_with_session(
                engine,
                state,
                line,
                &eval_options,
                cas_didactic::VerbosityLevel::Normal,
            )
        });
        let actions = match actions {
            Ok(out) => out,
            Err(error) => {
                return reply_output(
                    cas_session::solver_exports::format_timeline_command_error_message(&error),
                );
            }
        };

        output::timeline_cli_actions_to_reply(actions)
    }
}
