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
        match cas_solver::evaluate_equiv_command_lines(&mut self.core.engine.simplifier, line) {
            Ok(lines) => reply_output(lines.join("\n")),
            Err(message) => reply_output(message),
        }
    }

    pub(crate) fn handle_subst_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        // Format: subst <expr>, <target>, <replacement>
        // Examples:
        //   subst x^4 + x^2 + 1, x^2, y   → y² + y + 1 (power-aware)
        //   subst x^2 + x, x, 3          → 12 (variable substitution)
        let render_mode = cas_solver::substitute_render_mode_from_display_mode(
            Self::set_display_mode_from_verbosity(verbosity),
        );
        match cas_solver::evaluate_substitute_command_lines(
            &mut self.core.engine.simplifier,
            line,
            render_mode,
        ) {
            Ok(lines) => reply_output(lines.join("\n")),
            Err(message) => reply_output(message),
        }
    }

    pub(crate) fn handle_timeline_core(&mut self, line: &str) -> ReplReply {
        let eval_options = self.core.state.options().clone();
        let eval_output = match cas_solver::evaluate_timeline_command_line(
            &mut self.core.engine,
            &mut self.core.state,
            line,
            &eval_options,
        ) {
            Ok(out) => out,
            Err(message) => return reply_output(message),
        };

        let simplify_out = match eval_output {
            cas_solver::TimelineCommandEvalOutput::Solve(out) => {
                return self.render_timeline_solve_eval_output(out)
            }
            cas_solver::TimelineCommandEvalOutput::Simplify(out) => out,
        };

        let render = cas_didactic::render_simplify_timeline_cli_output(
            &mut self.core.engine.simplifier.context,
            &simplify_out,
            cas_didactic::VerbosityLevel::Normal,
        );

        self.timeline_cli_render_to_reply(render)
    }

    pub(crate) fn handle_visualize_core(&mut self, line: &str) -> ReplReply {
        use std::path::PathBuf;

        match cas_solver::evaluate_visualize_command_output(&mut self.core.engine.simplifier, line)
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
        match cas_solver::evaluate_explain_command_lines(&mut self.core.engine.simplifier, line) {
            Ok(lines) => reply_output(lines.join("\n")),
            Err(message) => reply_output(message),
        }
    }
}
