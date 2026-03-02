use super::*;

impl Repl {
    pub(crate) fn render_timeline_solve_eval_output(
        &mut self,
        out: cas_solver::TimelineSolveEvalOutput,
    ) -> ReplReply {
        use std::path::PathBuf;

        if out.display_steps.0.is_empty() {
            return reply_output(cas_solver::format_timeline_solve_no_steps_message(
                &self.core.engine.simplifier.context,
                &out.solution_set,
            ));
        }

        let mut timeline = cas_didactic::SolveTimelineHtml::new(
            &mut self.core.engine.simplifier.context,
            &out.display_steps.0,
            &out.equation,
            &out.solution_set,
            &out.var,
        );
        let html = timeline.to_html();

        let result_line = cas_solver::format_timeline_solve_result_line(
            &self.core.engine.simplifier.context,
            &out.solution_set,
        );
        vec![
            ReplMsg::WriteFile {
                path: PathBuf::from("timeline.html"),
                contents: html,
            },
            ReplMsg::output(result_line),
            ReplMsg::output(cas_solver::timeline_open_hint_message()),
        ]
    }

    /// Handle the 'weierstrass' command for applying Weierstrass substitution
    /// Transforms sin(x), cos(x), tan(x) into rational expressions in t = tan(x/2)
    pub(crate) fn handle_weierstrass(&mut self, line: &str) {
        let reply = self.handle_weierstrass_core(line);
        self.print_reply(reply);
    }

    fn handle_weierstrass_core(&mut self, line: &str) -> ReplReply {
        let rest = match cas_solver::parse_weierstrass_command_input(line) {
            cas_solver::WeierstrassCommandInput::MissingInput => {
                return reply_output(cas_solver::weierstrass_usage_message());
            }
            cas_solver::WeierstrassCommandInput::Expr(rest) => rest,
        };

        match cas_solver::evaluate_weierstrass_input(&mut self.core.engine.simplifier, rest) {
            Ok(out) => {
                let mut lines = cas_solver::format_weierstrass_eval_lines(
                    &self.core.engine.simplifier.context,
                    rest,
                    &out,
                );
                clean_result_line(&mut lines);
                reply_output(lines.join("\n"))
            }
            Err(e) => reply_output(cas_solver::format_transform_eval_error_message(&e)),
        }
    }

    pub(crate) fn handle_solve(&mut self, line: &str) {
        let reply = self.handle_solve_core(line, self.verbosity);
        self.print_reply(reply);
    }

    fn handle_solve_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        let mut lines: Vec<String> = Vec::new();

        // solve [--check] <equation>, <var>
        let rest = cas_solver::extract_solve_command_tail(line);
        let default_check_enabled = self.core.state.options().check_solutions;
        let step_verbosity = cas_solver::solve_step_verbosity_from_display_mode(
            Self::set_display_mode_from_verbosity(verbosity),
        );

        match cas_solver::evaluate_solve_invocation_input(
            &mut self.core.engine,
            &mut self.core.state,
            rest,
            default_check_enabled,
            true,
        ) {
            Ok(invocation_out) => {
                lines.extend(cas_solver::format_solve_command_eval_lines(
                    &mut self.core.engine,
                    &invocation_out.eval_output,
                    cas_solver::SolveCommandRenderConfig {
                        show_steps: step_verbosity.show_steps,
                        show_verbose_substeps: step_verbosity.show_verbose_substeps,
                        requires_display: self.core.state.options().requires_display,
                        debug_mode: self.core.debug_mode,
                        hints_enabled: self.core.state.options().hints_enabled,
                        domain_mode: self.core.state.options().shared.semantics.domain_mode,
                        check_solutions: invocation_out.check_enabled,
                    },
                ));
            }
            Err(e) => lines.push(cas_solver::format_solve_command_error_message(&e)),
        }

        reply_output(lines.join("\n"))
    }
}
