use super::*;

impl Repl {
    pub(crate) fn handle_equiv(&mut self, line: &str) {
        let reply = self.handle_equiv_core(line);
        self.print_reply(reply);
    }

    fn handle_equiv_core(&mut self, line: &str) -> ReplReply {
        let rest = line[6..].trim();
        match cas_solver::evaluate_equiv_input(&mut self.core.engine.simplifier, rest) {
            Ok(result) => {
                let mut lines = Vec::new();
                match result {
                    cas_solver::EquivalenceResult::True => {
                        lines.push("True".to_string());
                    }
                    cas_solver::EquivalenceResult::ConditionalTrue { requires } => {
                        lines.push("True (conditional)".to_string());
                        for line in cas_solver::format_text_requires_lines(&requires) {
                            lines.push(line);
                        }
                    }
                    cas_solver::EquivalenceResult::False => {
                        lines.push("False".to_string());
                    }
                    cas_solver::EquivalenceResult::Unknown => {
                        lines.push("Unknown (cannot prove equivalence)".to_string());
                    }
                }
                reply_output(lines.join("\n"))
            }
            Err(cas_solver::ParseExprPairError::MissingDelimiter) => {
                reply_output("Usage: equiv <expr1>, <expr2>")
            }
            Err(cas_solver::ParseExprPairError::FirstArg(e)) => {
                reply_output(format!("Error parsing first arg: {}", e))
            }
            Err(cas_solver::ParseExprPairError::SecondArg(e)) => {
                reply_output(format!("Error parsing second arg: {}", e))
            }
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
        let rest = line[6..].trim();

        let usage = "Usage: subst <expr>, <target>, <replacement>\n\n\
                     Examples:\n\
                       subst x^2 + x, x, 3              → 12\n\
                       subst x^4 + x^2 + 1, x^2, y      → y² + y + 1\n\
                       subst x^3, x^2, y                → y·x";

        let output = match cas_solver::evaluate_substitute_and_simplify_input(
            &mut self.core.engine.simplifier,
            rest,
            cas_solver::SubstituteOptions::default(),
        ) {
            Ok(out) => out,
            Err(cas_solver::ParseSubstituteArgsError::InvalidArity) => {
                return reply_output(usage);
            }
            Err(cas_solver::ParseSubstituteArgsError::Expression(e)) => {
                return reply_output(format!("Error parsing expression: {}", e));
            }
            Err(cas_solver::ParseSubstituteArgsError::Target(e)) => {
                return reply_output(format!("Error parsing target: {}", e));
            }
            Err(cas_solver::ParseSubstituteArgsError::Replacement(e)) => {
                return reply_output(format!("Error parsing replacement: {}", e));
            }
        };
        let strategy = output.strategy;

        let display_parts = cas_solver::split_by_comma_ignoring_parens(rest);
        let expr_str = display_parts.first().map(|s| s.trim()).unwrap_or_default();
        let target_str = display_parts.get(1).map(|s| s.trim()).unwrap_or_default();
        let replacement_str = display_parts.get(2).map(|s| s.trim()).unwrap_or_default();

        let mut lines = Vec::new();
        if verbosity != Verbosity::None {
            let label = match strategy {
                cas_solver::SubstituteStrategy::Variable => "Variable substitution",
                cas_solver::SubstituteStrategy::PowerAware => "Expression substitution",
            };
            lines.push(format!(
                "{label}: {} → {} in {}",
                target_str, replacement_str, expr_str
            ));
        }

        let result = output.simplified_expr;
        let steps = output.steps;
        if verbosity != Verbosity::None && !steps.is_empty() {
            if verbosity != Verbosity::Succinct {
                lines.push("Steps:".to_string());
            }
            for step in steps.iter() {
                if should_show_step(step, verbosity) {
                    if verbosity == Verbosity::Succinct {
                        lines.push(format!(
                            "-> {}",
                            DisplayExpr {
                                context: &self.core.engine.simplifier.context,
                                id: step.global_after.unwrap_or(step.after)
                            }
                        ));
                    } else {
                        lines.push(format!("  {}  [{}]", step.description, step.rule_name));
                    }
                }
            }
        }
        lines.push(format!(
            "Result: {}",
            clean_display_string(&format!(
                "{}",
                DisplayExpr {
                    context: &self.core.engine.simplifier.context,
                    id: result
                }
            ))
        ));

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

        let rest = line[9..].trim();

        let timeline_input = cas_solver::parse_timeline_command_input(rest);
        let (expr_str, use_aggressive) = match timeline_input {
            cas_solver::TimelineCommandInput::Solve(solve_rest) => {
                self.handle_timeline_solve(&solve_rest);
                return ReplReply::new();
            }
            cas_solver::TimelineCommandInput::Simplify { expr, aggressive } => (expr, aggressive),
        };

        let eval_output = if use_aggressive {
            cas_solver::evaluate_timeline_simplify_aggressive_input(
                &mut self.core.engine.simplifier,
                &expr_str,
            )
        } else {
            cas_solver::evaluate_timeline_simplify_input(
                &mut self.core.engine,
                &mut self.core.state,
                &expr_str,
            )
        };
        let eval_output = match eval_output {
            Ok(out) => out,
            Err(cas_solver::TimelineEvalError::Parse(e)) => {
                return reply_output(format!("Parse error: {}", e));
            }
            Err(cas_solver::TimelineEvalError::Eval(e)) => {
                return reply_output(format!("Simplification error: {}", e));
            }
        };
        let steps = eval_output.steps;
        let expr_id = eval_output.parsed_expr;
        let simplified = eval_output.simplified_expr;

        if steps.is_empty() {
            return reply_output("No simplification steps to visualize.");
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
            Some(expr_str.as_str()),
        );
        let html = timeline.to_html();

        // Return WriteFile action + info messages
        let mut reply = ReplReply::new();
        reply.push(ReplMsg::WriteFile {
            path: PathBuf::from("timeline.html"),
            contents: html,
        });
        if use_aggressive {
            reply.push(ReplMsg::output("(Aggressive simplification mode)"));
        }
        reply.push(ReplMsg::output(
            "Open in browser to view interactive visualization.",
        ));
        reply
    }

    pub(crate) fn handle_visualize(&mut self, line: &str) {
        let reply = self.handle_visualize_core(line);
        self.print_reply(reply);
    }

    fn handle_visualize_core(&mut self, line: &str) -> ReplReply {
        use std::path::PathBuf;

        let rest = line
            .strip_prefix("visualize ")
            .or_else(|| line.strip_prefix("viz "))
            .unwrap_or(line)
            .trim();

        match cas_solver::evaluate_visualize_input(&mut self.core.engine.simplifier, rest) {
            Ok(out) => vec![
                ReplMsg::WriteFile {
                    path: PathBuf::from("ast.dot"),
                    contents: out.dot,
                },
                ReplMsg::output("Render with: dot -Tsvg ast.dot -o ast.svg"),
                ReplMsg::output("Or: dot -Tpng ast.dot -o ast.png"),
            ],
            Err(cas_solver::TransformEvalError::Parse(e)) => {
                reply_output(format!("Parse error: {}", e))
            }
        }
    }

    pub(crate) fn handle_explain(&mut self, line: &str) {
        let reply = self.handle_explain_core(line);
        self.print_reply(reply);
    }

    fn handle_explain_core(&mut self, line: &str) -> ReplReply {
        let rest = line[8..].trim(); // Remove "explain "
        match cas_solver::evaluate_explain_gcd_input(&mut self.core.engine.simplifier, rest) {
            Ok(out) => {
                let mut lines = Vec::new();
                lines.push(format!("Parsed: {}", rest));
                lines.push(String::new());
                lines.push("Educational Steps:".to_string());
                lines.push("─".repeat(60));

                for step in &out.steps {
                    lines.push(step.clone());
                }

                lines.push("─".repeat(60));
                lines.push(String::new());

                if let Some(result_expr) = out.value {
                    lines.push(format!(
                        "Result: {}",
                        clean_display_string(&format!(
                            "{}",
                            DisplayExpr {
                                context: &self.core.engine.simplifier.context,
                                id: result_expr
                            }
                        ))
                    ));
                } else {
                    lines.push("Could not compute GCD".to_string());
                }
                reply_output(lines.join("\n"))
            }
            Err(cas_solver::ExplainEvalError::Parse(e)) => {
                reply_output(format!("Parse error: {}", e))
            }
            Err(cas_solver::ExplainEvalError::NotFunctionCall) => reply_output(
                "Explain mode currently only supports function calls\n\
                 Try: explain gcd(48, 18)",
            ),
            Err(cas_solver::ExplainEvalError::UnsupportedFunction(name)) => reply_output(format!(
                "Explain mode not yet implemented for function '{}'\n\
                 Currently supported: gcd",
                name
            )),
            Err(cas_solver::ExplainEvalError::InvalidArity {
                function: _,
                expected: _,
                actual: _,
            }) => reply_output("Usage: explain gcd(a, b)"),
        }
    }
}
