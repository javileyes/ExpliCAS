use super::*;

impl Repl {
    pub(crate) fn timeline_cli_render_to_reply(
        &self,
        render: cas_didactic::TimelineCliRender,
    ) -> ReplReply {
        use std::path::PathBuf;

        match render {
            cas_didactic::TimelineCliRender::NoSteps { lines } => reply_output(lines.join("\n")),
            cas_didactic::TimelineCliRender::Html {
                file_name,
                html,
                lines,
            } => {
                let mut reply = ReplReply::new();
                reply.push(ReplMsg::WriteFile {
                    path: PathBuf::from(file_name),
                    contents: html,
                });
                reply.push(ReplMsg::OpenFile {
                    path: PathBuf::from(file_name),
                });
                for line in lines {
                    reply.push(ReplMsg::output(line));
                }
                reply
            }
        }
    }

    pub(crate) fn handle_equiv_core(&mut self, line: &str) -> ReplReply {
        match cas_solver::evaluate_equiv_command_lines_with_engine(&mut self.core.engine, line) {
            Ok(lines) => reply_output(lines.join("\n")),
            Err(message) => reply_output(message),
        }
    }

    pub(crate) fn handle_subst_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        // Format: subst <expr>, <target>, <replacement>
        // Examples:
        //   subst x^4 + x^2 + 1, x^2, y   → y² + y + 1 (power-aware)
        //   subst x^2 + x, x, 3          → 12 (variable substitution)
        match cas_solver::evaluate_substitute_command_lines_for_display_mode_with_engine(
            &mut self.core.engine,
            line,
            Self::set_display_mode_from_verbosity(verbosity),
        ) {
            Ok(lines) => reply_output(lines.join("\n")),
            Err(message) => reply_output(message),
        }
    }

    pub(crate) fn handle_timeline_core(&mut self, line: &str) -> ReplReply {
        let eval_output = match cas_solver::evaluate_timeline_command_line_with_session_options(
            &mut self.core.engine,
            &mut self.core.state,
            line,
        ) {
            Ok(out) => out,
            Err(message) => return reply_output(message),
        };

        let render = cas_didactic::render_timeline_command_cli_output_with_engine(
            &mut self.core.engine,
            &eval_output,
            cas_didactic::VerbosityLevel::Normal,
        );

        self.timeline_cli_render_to_reply(render)
    }

    pub(crate) fn handle_visualize_core(&mut self, line: &str) -> ReplReply {
        use std::path::PathBuf;

        match cas_solver::evaluate_visualize_command_output_with_engine(&mut self.core.engine, line)
        {
            Ok(render) => {
                let mut reply = vec![ReplMsg::WriteFile {
                    path: PathBuf::from(render.file_name),
                    contents: render.contents,
                }];
                for line in render.hint_lines {
                    reply.push(ReplMsg::output(line));
                }
                reply
            }
            Err(message) => reply_output(message),
        }
    }

    pub(crate) fn handle_explain_core(&mut self, line: &str) -> ReplReply {
        match cas_solver::evaluate_explain_command_lines_with_engine(&mut self.core.engine, line) {
            Ok(lines) => reply_output(lines.join("\n")),
            Err(message) => reply_output(message),
        }
    }
}
