use super::*;

impl Repl {
    pub(crate) fn timeline_cli_actions_to_reply(
        &self,
        actions: Vec<cas_didactic::TimelineCliAction>,
    ) -> ReplReply {
        use std::path::PathBuf;

        let mut reply = ReplReply::new();
        for action in actions {
            match action {
                cas_didactic::TimelineCliAction::Output(line) => reply.push(ReplMsg::output(line)),
                cas_didactic::TimelineCliAction::WriteFile { path, contents } => {
                    reply.push(ReplMsg::WriteFile {
                        path: PathBuf::from(path),
                        contents,
                    });
                }
                cas_didactic::TimelineCliAction::OpenFile { path } => {
                    reply.push(ReplMsg::OpenFile {
                        path: PathBuf::from(path),
                    });
                }
            }
        }
        reply
    }

    pub(crate) fn handle_equiv_core(&mut self, line: &str) -> ReplReply {
        match cas_session::evaluate_equiv_invocation_message_on_repl_core(&mut self.core, line) {
            Ok(message) => reply_output(message),
            Err(message) => reply_output(message),
        }
    }

    pub(crate) fn handle_subst_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        let message = match cas_session::evaluate_substitute_invocation_user_message_on_repl_core(
            &mut self.core,
            line,
            Self::set_display_mode_from_verbosity(verbosity),
        ) {
            Ok(message) => message,
            Err(message) => return reply_output(message),
        };
        reply_output(message)
    }

    pub(crate) fn handle_timeline_core(&mut self, line: &str) -> ReplReply {
        let eval_options = cas_session::eval_options_from_repl_core(&self.core);
        let actions = match cas_didactic::evaluate_timeline_invocation_cli_actions_with_session(
            &mut self.core.engine,
            &mut self.core.state,
            line,
            &eval_options,
            cas_didactic::VerbosityLevel::Normal,
        ) {
            Ok(out) => out,
            Err(error) => {
                return reply_output(cas_session::format_timeline_command_error_message(&error))
            }
        };

        self.timeline_cli_actions_to_reply(actions)
    }

    pub(crate) fn handle_visualize_core(&mut self, line: &str) -> ReplReply {
        use std::path::PathBuf;

        let output = match cas_session::evaluate_visualize_invocation_output_on_repl_core(
            &mut self.core,
            line,
        ) {
            Ok(output) => output,
            Err(message) => return reply_output(message),
        };
        let mut reply = vec![ReplMsg::WriteFile {
            path: PathBuf::from(output.file_name),
            contents: output.dot_source,
        }];
        for line in output.hint_lines {
            reply.push(ReplMsg::output(line));
        }
        reply
    }

    pub(crate) fn handle_explain_core(&mut self, line: &str) -> ReplReply {
        let message = match cas_session::evaluate_explain_invocation_message_on_repl_core(
            &mut self.core,
            line,
        ) {
            Ok(message) => message,
            Err(message) => return reply_output(message),
        };
        reply_output(message)
    }
}
