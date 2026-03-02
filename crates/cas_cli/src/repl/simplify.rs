use super::*;

fn extract_simplify_command_tail(line: &str) -> &str {
    line.strip_prefix("simplify").unwrap_or(line).trim()
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FullSimplifyEvalError {
    Parse(String),
    Resolve(String),
}

#[derive(Debug, Clone)]
struct FullSimplifyEvalOutput {
    resolved_expr: cas_ast::ExprId,
    simplified_expr: cas_ast::ExprId,
    steps: Vec<cas_solver::Step>,
}

fn evaluate_full_simplify_input(
    simplifier: &mut cas_solver::Simplifier,
    session: &cas_session::SessionState,
    input: &str,
    collect_steps: bool,
) -> Result<FullSimplifyEvalOutput, FullSimplifyEvalError> {
    let mut temp_simplifier = cas_solver::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, &mut temp_simplifier.context);
    std::mem::swap(&mut simplifier.profiler, &mut temp_simplifier.profiler);

    let result = (|| {
        let parsed_expr = cas_parser::parse(input, &mut temp_simplifier.context)
            .map_err(|e| FullSimplifyEvalError::Parse(e.to_string()))?;
        let (resolved_expr, _diag, _cache_hits) = session
            .resolve_state_refs_with_diagnostics(&mut temp_simplifier.context, parsed_expr)
            .map_err(|e| FullSimplifyEvalError::Resolve(e.to_string()))?;

        let mut opts = session.options().to_simplify_options();
        opts.collect_steps = collect_steps;
        let (simplified_expr, steps, stats) =
            temp_simplifier.simplify_with_stats(resolved_expr, opts);
        let _ = stats;
        Ok(FullSimplifyEvalOutput {
            resolved_expr,
            simplified_expr,
            steps,
        })
    })();

    std::mem::swap(&mut simplifier.context, &mut temp_simplifier.context);
    std::mem::swap(&mut simplifier.profiler, &mut temp_simplifier.profiler);
    result
}

fn should_show_simplify_step(step: &cas_solver::Step, mode: SetDisplayMode) -> bool {
    match mode {
        SetDisplayMode::None => false,
        SetDisplayMode::Verbose => true,
        SetDisplayMode::Succinct | SetDisplayMode::Normal => {
            if step.get_importance() < cas_solver::ImportanceLevel::Medium {
                return false;
            }
            if let (Some(before), Some(after)) = (step.global_before, step.global_after) {
                if before == after {
                    return false;
                }
            }
            true
        }
    }
}

fn format_full_simplify_eval_lines(
    ctx: &mut cas_ast::Context,
    expr_input: &str,
    output: &FullSimplifyEvalOutput,
    mode: SetDisplayMode,
) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let resolved_expr = output.resolved_expr;
    let simplified = output.simplified_expr;
    let steps = &output.steps;

    let style_signals = cas_formatter::ParseStyleSignals::from_input_string(expr_input);
    let style_prefs = cas_formatter::StylePreferences::from_expression_with_signals(
        ctx,
        resolved_expr,
        Some(&style_signals),
    );

    lines.push(format!(
        "Parsed: {}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: resolved_expr
        }
    ));

    if mode != SetDisplayMode::None {
        if steps.is_empty() {
            if mode != SetDisplayMode::Succinct {
                lines.push("No simplification steps needed.".to_string());
            }
        } else {
            if mode != SetDisplayMode::Succinct {
                lines.push("Steps (Aggressive Mode):".to_string());
            }

            let mut current_root = resolved_expr;
            let mut step_count = 0;

            for step in steps {
                if should_show_simplify_step(step, mode) {
                    step_count += 1;

                    if mode == SetDisplayMode::Succinct {
                        current_root = cas_solver::reconstruct_global_expr(
                            ctx,
                            current_root,
                            step.path(),
                            step.after,
                        );
                        lines.push(format!(
                            "-> {}",
                            cas_formatter::DisplayExpr {
                                context: ctx,
                                id: current_root
                            }
                        ));
                    } else {
                        lines.push(format!(
                            "{}. {}  [{}]",
                            step_count, step.description, step.rule_name
                        ));

                        if let Some(global_before) = step.global_before {
                            lines.push(format!(
                                "   Before: {}",
                                cas_formatter::clean_display_string(&format!(
                                    "{}",
                                    cas_formatter::DisplayExprStyled::new(
                                        ctx,
                                        global_before,
                                        &style_prefs
                                    )
                                ))
                            ));
                        } else {
                            lines.push(format!(
                                "   Before: {}",
                                cas_formatter::clean_display_string(&format!(
                                    "{}",
                                    cas_formatter::DisplayExprStyled::new(
                                        ctx,
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

                        let before_disp = cas_formatter::clean_display_string(&format!(
                            "{}",
                            cas_formatter::DisplayExprStyled::new(
                                ctx,
                                rule_before_id,
                                &style_prefs
                            )
                        ));
                        let after_disp = cas_formatter::clean_display_string(
                            &cas_formatter::render_with_rule_scope(
                                ctx,
                                rule_after_id,
                                &step.rule_name,
                                &style_prefs,
                            ),
                        );

                        lines.push(format!("   Rule: {} -> {}", before_disp, after_disp));

                        if let Some(global_after) = step.global_after {
                            current_root = global_after;
                        } else {
                            current_root = cas_solver::reconstruct_global_expr(
                                ctx,
                                current_root,
                                step.path(),
                                step.after,
                            );
                        }

                        lines.push(format!(
                            "   After: {}",
                            cas_formatter::clean_display_string(&format!(
                                "{}",
                                cas_formatter::DisplayExprStyled::new(
                                    ctx,
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
                    current_root = cas_solver::reconstruct_global_expr(
                        ctx,
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
        cas_formatter::clean_display_string(&format!(
            "{}",
            cas_formatter::DisplayExprStyled::new(ctx, simplified, &style_prefs)
        ))
    ));

    lines
}

impl Repl {
    pub(crate) fn handle_full_simplify_core(
        &mut self,
        line: &str,
        verbosity: Verbosity,
    ) -> ReplReply {
        let display_mode = Self::set_display_mode_from_verbosity(verbosity);
        let expr_input = extract_simplify_command_tail(line);
        let output = match evaluate_full_simplify_input(
            &mut self.core.engine.simplifier,
            &self.core.state,
            expr_input,
            !matches!(display_mode, SetDisplayMode::None),
        ) {
            Ok(output) => output,
            Err(error) => {
                let message = match error {
                    FullSimplifyEvalError::Parse(message) => {
                        format!("Error: {message}")
                    }
                    FullSimplifyEvalError::Resolve(message) => {
                        format!("Error resolving variables: {message}")
                    }
                };
                return reply_output(message);
            }
        };
        let lines = format_full_simplify_eval_lines(
            &mut self.core.engine.simplifier.context,
            expr_input,
            &output,
            display_mode,
        );

        // Store health report for the `health` command (if health tracking is enabled)
        self.core.last_health_report = cas_session::capture_health_report_if_enabled(
            &self.core.engine.simplifier,
            self.core.health_enabled,
        );

        reply_output(lines.join("\n"))
    }
}
