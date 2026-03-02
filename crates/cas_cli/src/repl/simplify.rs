use super::*;

impl Repl {
    pub(crate) fn handle_full_simplify(&mut self, line: &str) {
        let reply = self.handle_full_simplify_core(line, self.verbosity);
        self.print_reply(reply);
    }

    fn handle_full_simplify_core(&mut self, line: &str, verbosity: Verbosity) -> ReplReply {
        // simplify <expr>
        // Uses a temporary simplifier with ALL default rules (including aggressive distribution)
        let expr_str = line[9..].trim();
        let mut lines: Vec<String> = Vec::new();
        let output = match cas_solver::evaluate_full_simplify_input(
            &mut self.core.engine,
            &self.core.state,
            expr_str,
            verbosity != Verbosity::None,
        ) {
            Ok(out) => out,
            Err(cas_solver::FullSimplifyEvalError::Parse(e)) => {
                return reply_output(format!("Error: {}", e));
            }
            Err(cas_solver::FullSimplifyEvalError::Resolve(e)) => {
                return reply_output(format!("Error resolving variables: {}", e));
            }
        };

        let resolved_expr = output.resolved_expr;
        let simplified = output.simplified_expr;
        let steps = output.steps;

        let style_signals = ParseStyleSignals::from_input_string(expr_str);
        let style_prefs = StylePreferences::from_expression_with_signals(
            &self.core.engine.simplifier.context,
            resolved_expr,
            Some(&style_signals),
        );

        lines.push(format!(
            "Parsed: {}",
            DisplayExpr {
                context: &self.core.engine.simplifier.context,
                id: resolved_expr
            }
        ));

        if verbosity != Verbosity::None {
            if steps.is_empty() {
                if verbosity != Verbosity::Succinct {
                    lines.push("No simplification steps needed.".to_string());
                }
            } else {
                if verbosity != Verbosity::Succinct {
                    lines.push("Steps (Aggressive Mode):".to_string());
                }
                let mut current_root = resolved_expr;
                let mut step_count = 0;
                for step in steps.iter() {
                    if should_show_step(step, verbosity) {
                        step_count += 1;

                        if verbosity == Verbosity::Succinct {
                            current_root = reconstruct_global_expr(
                                &mut self.core.engine.simplifier.context,
                                current_root,
                                step.path(),
                                step.after,
                            );
                            lines.push(format!(
                                "-> {}",
                                DisplayExpr {
                                    context: &self.core.engine.simplifier.context,
                                    id: current_root
                                }
                            ));
                        } else {
                            lines.push(format!(
                                "{}. {}  [{}]",
                                step_count, step.description, step.rule_name
                            ));

                            if verbosity == Verbosity::Verbose || verbosity == Verbosity::Normal {
                                if let Some(global_before) = step.global_before {
                                    lines.push(format!(
                                        "   Before: {}",
                                        clean_display_string(&format!(
                                            "{}",
                                            DisplayExprStyled::new(
                                                &self.core.engine.simplifier.context,
                                                global_before,
                                                &style_prefs
                                            )
                                        ))
                                    ));
                                } else {
                                    lines.push(format!(
                                        "   Before: {}",
                                        clean_display_string(&format!(
                                            "{}",
                                            DisplayExprStyled::new(
                                                &self.core.engine.simplifier.context,
                                                current_root,
                                                &style_prefs
                                            )
                                        ))
                                    ));
                                }

                                let (rule_before_id, rule_after_id) =
                                    match (step.before_local(), step.after_local()) {
                                        (Some(bl), Some(al)) => (bl, al),
                                        _ => (step.before, step.after),
                                    };

                                let before_disp = clean_display_string(&format!(
                                    "{}",
                                    DisplayExprStyled::new(
                                        &self.core.engine.simplifier.context,
                                        rule_before_id,
                                        &style_prefs
                                    )
                                ));
                                let after_disp = clean_display_string(&render_with_rule_scope(
                                    &self.core.engine.simplifier.context,
                                    rule_after_id,
                                    &step.rule_name,
                                    &style_prefs,
                                ));

                                lines.push(format!("   Rule: {} -> {}", before_disp, after_disp));
                            }

                            if let Some(global_after) = step.global_after {
                                current_root = global_after;
                            } else {
                                current_root = reconstruct_global_expr(
                                    &mut self.core.engine.simplifier.context,
                                    current_root,
                                    step.path(),
                                    step.after,
                                );
                            }

                            lines.push(format!(
                                "   After: {}",
                                clean_display_string(&format!(
                                    "{}",
                                    DisplayExprStyled::new(
                                        &self.core.engine.simplifier.context,
                                        current_root,
                                        &style_prefs
                                    )
                                ))
                            ));

                            for assumption_line in cas_solver::format_displayable_assumption_lines(
                                step.assumption_events(),
                            ) {
                                lines.push(format!("   {}", assumption_line));
                            }
                        }
                    } else if let Some(global_after) = step.global_after {
                        current_root = global_after;
                    } else {
                        current_root = reconstruct_global_expr(
                            &mut self.core.engine.simplifier.context,
                            current_root,
                            step.path(),
                            step.after,
                        );
                    }
                }
            }
        }

        lines.push(format!(
            "Result: {}",
            clean_display_string(&format!(
                "{}",
                DisplayExprStyled::new(
                    &self.core.engine.simplifier.context,
                    simplified,
                    &style_prefs
                )
            ))
        ));

        // Store health report for the `health` command (if health tracking is enabled)
        if self.core.health_enabled {
            self.core.last_health_report =
                Some(self.core.engine.simplifier.profiler.health_report());
        }

        reply_output(lines.join("\n"))
    }
}
