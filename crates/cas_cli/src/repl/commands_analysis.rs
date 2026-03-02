use super::*;

impl Repl {
    pub(crate) fn handle_equiv(&mut self, line: &str) {
        let reply = self.handle_equiv_core(line);
        self.print_reply(reply);
    }

    fn handle_equiv_core(&mut self, line: &str) -> ReplReply {
        let rest = cas_solver::extract_equiv_command_tail(line);
        match cas_solver::evaluate_equiv_input(&mut self.core.engine.simplifier, rest) {
            Ok(result) => {
                reply_output(cas_solver::format_equivalence_result_lines(&result).join("\n"))
            }
            Err(e) => reply_output(cas_solver::format_expr_pair_parse_error_message(
                &e, "equiv",
            )),
        }
    }

    pub(crate) fn handle_subst(&mut self, line: &str) {
        let reply = self.handle_subst_core(line, self.verbosity);
        self.print_reply(reply);
    }

    fn handle_subst_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        // Format: subst <expr>, <target>, <replacement>
        // Examples:
        //   subst x^4 + x^2 + 1, x^2, y   → y² + y + 1 (power-aware)
        //   subst x^2 + x, x, 3          → 12 (variable substitution)
        let rest = cas_solver::extract_substitute_command_tail(line);

        let output = match cas_solver::evaluate_substitute_and_simplify_input(
            &mut self.core.engine.simplifier,
            rest,
            cas_solver::SubstituteOptions::default(),
        ) {
            Ok(out) => out,
            Err(e) => return reply_output(cas_solver::format_substitute_parse_error_message(&e)),
        };

        let render_mode = cas_solver::substitute_render_mode_from_display_mode(
            Self::set_display_mode_from_verbosity(verbosity),
        );
        let mut lines = cas_solver::format_substitute_eval_lines(
            &self.core.engine.simplifier.context,
            rest,
            &output,
            render_mode,
        );
        clean_result_line(&mut lines);

        reply_output(lines.join("\n"))
    }

    pub(crate) fn handle_timeline(&mut self, line: &str) {
        let reply = self.handle_timeline_core(line);

        // Post-processing: auto-open timeline.html on macOS after WriteFile is printed
        let has_timeline_write = reply.iter().any(|msg| {
            matches!(msg, ReplMsg::WriteFile { path, .. } if path.to_string_lossy().ends_with("timeline.html"))
        });

        self.print_reply(reply);

        // Try to auto-open on macOS (I/O stays in shell)
        if has_timeline_write {
            #[cfg(target_os = "macos")]
            {
                let _ = std::process::Command::new("open")
                    .arg("timeline.html")
                    .spawn();
            }
        }
    }

    fn handle_timeline_core(&mut self, line: &str) -> ReplReply {
        use std::path::PathBuf;

        let rest = cas_solver::extract_timeline_command_tail(line);
        let eval_options = self.core.state.options().clone();
        let eval_output = match cas_solver::evaluate_timeline_command_input(
            &mut self.core.engine,
            &mut self.core.state,
            rest,
            &eval_options,
        ) {
            Ok(out) => out,
            Err(e) => return reply_output(cas_solver::format_timeline_command_error_message(&e)),
        };

        let simplify_out = match eval_output {
            cas_solver::TimelineCommandEvalOutput::Solve(out) => {
                return self.render_timeline_solve_eval_output(out)
            }
            cas_solver::TimelineCommandEvalOutput::Simplify(out) => out,
        };
        let steps = simplify_out.steps;
        let expr_id = simplify_out.parsed_expr;
        let simplified = simplify_out.simplified_expr;

        if steps.is_empty() {
            return reply_output(cas_solver::timeline_no_steps_message());
        }

        // NOTE: filter_non_productive_steps removed here as timeline already handles filtering
        // and the result was previously unused (prefixed with _)

        // Convert CLI verbosity to timeline verbosity
        // Use Normal level - shows important steps without low-level canonicalization
        let timeline_verbosity = cas_didactic::VerbosityLevel::Normal;

        // Generate HTML timeline with ALL steps and the known simplified result
        // V2.14.40: Pass input string for style preference sniffing (exponential vs radical)
        let mut timeline = cas_didactic::TimelineHtml::new_with_result_and_style(
            &mut self.core.engine.simplifier.context,
            &steps,
            expr_id,
            Some(simplified),
            timeline_verbosity,
            Some(simplify_out.expr_input.as_str()),
        );
        let html = timeline.to_html();

        // Return WriteFile action + info messages
        let mut reply = ReplReply::new();
        reply.push(ReplMsg::WriteFile {
            path: PathBuf::from("timeline.html"),
            contents: html,
        });
        for line in cas_solver::format_timeline_simplify_info_lines(simplify_out.use_aggressive) {
            reply.push(ReplMsg::output(line));
        }
        reply
    }

    pub(crate) fn handle_visualize(&mut self, line: &str) {
        let reply = self.handle_visualize_core(line);
        self.print_reply(reply);
    }

    fn handle_visualize_core(&mut self, line: &str) -> ReplReply {
        use std::path::PathBuf;

        let rest = cas_solver::extract_visualize_command_tail(line);

        match cas_solver::evaluate_visualize_input(&mut self.core.engine.simplifier, rest) {
            Ok(out) => {
                let mut reply = vec![ReplMsg::WriteFile {
                    path: PathBuf::from("ast.dot"),
                    contents: out.dot,
                }];
                for line in cas_solver::visualize_output_hint_lines() {
                    reply.push(ReplMsg::output(line));
                }
                reply
            }
            Err(e) => reply_output(cas_solver::format_transform_eval_error_message(&e)),
        }
    }

    pub(crate) fn handle_explain(&mut self, line: &str) {
        let reply = self.handle_explain_core(line);
        self.print_reply(reply);
    }

    fn handle_explain_core(&mut self, line: &str) -> ReplReply {
        let rest = cas_solver::extract_explain_command_tail(line);
        match cas_solver::evaluate_explain_gcd_input(&mut self.core.engine.simplifier, rest) {
            Ok(out) => {
                let mut lines = cas_solver::format_explain_gcd_eval_lines(
                    &self.core.engine.simplifier.context,
                    rest,
                    &out,
                );
                clean_result_line(&mut lines);
                reply_output(lines.join("\n"))
            }
            Err(e) => reply_output(cas_solver::format_explain_error_message(&e)),
        }
    }
}
