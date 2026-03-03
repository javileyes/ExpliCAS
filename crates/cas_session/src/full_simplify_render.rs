//! Rendering helpers for `full_simplify` command output.

/// Extract expression tail from a `simplify` command line.
pub fn extract_simplify_command_tail(line: &str) -> &str {
    line.strip_prefix("simplify").unwrap_or(line).trim()
}

fn should_show_simplify_step(step: &cas_solver::Step, mode: crate::SetDisplayMode) -> bool {
    match mode {
        crate::SetDisplayMode::None => false,
        crate::SetDisplayMode::Verbose => true,
        crate::SetDisplayMode::Succinct | crate::SetDisplayMode::Normal => {
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

/// Format full simplify output lines according to display mode.
pub fn format_full_simplify_eval_lines(
    ctx: &mut cas_ast::Context,
    expr_input: &str,
    output: &crate::FullSimplifyEvalOutput,
    mode: crate::SetDisplayMode,
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

    if mode != crate::SetDisplayMode::None {
        if steps.is_empty() {
            if mode != crate::SetDisplayMode::Succinct {
                lines.push("No simplification steps needed.".to_string());
            }
        } else {
            if mode != crate::SetDisplayMode::Succinct {
                lines.push("Steps (Aggressive Mode):".to_string());
            }

            let mut current_root = resolved_expr;
            let mut step_count = 0;

            for step in steps {
                if should_show_simplify_step(step, mode) {
                    step_count += 1;

                    if mode == crate::SetDisplayMode::Succinct {
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

                        let assumption_events = cas_solver::assumption_events_from_step(step);
                        for assumption_line in
                            crate::format_displayable_assumption_lines(&assumption_events)
                        {
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

/// Evaluate a full `simplify ...` invocation and return final display lines.
pub fn evaluate_full_simplify_command_lines(
    simplifier: &mut cas_solver::Simplifier,
    session: &crate::SessionState,
    line: &str,
    display_mode: crate::SetDisplayMode,
) -> Result<Vec<String>, String> {
    let expr_input = extract_simplify_command_tail(line);
    let output = crate::evaluate_full_simplify_input(
        simplifier,
        session,
        expr_input,
        !matches!(display_mode, crate::SetDisplayMode::None),
    )
    .map_err(|error| crate::format_full_simplify_eval_error_message(&error))?;

    Ok(format_full_simplify_eval_lines(
        &mut simplifier.context,
        expr_input,
        &output,
        display_mode,
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn extract_simplify_command_tail_trims_prefix() {
        assert_eq!(super::extract_simplify_command_tail("simplify x+1"), "x+1");
    }

    #[test]
    fn evaluate_full_simplify_command_lines_runs() {
        let mut simplifier = cas_solver::Simplifier::with_default_rules();
        let session = crate::SessionState::new();
        let lines = super::evaluate_full_simplify_command_lines(
            &mut simplifier,
            &session,
            "simplify x + 0",
            crate::SetDisplayMode::Normal,
        )
        .expect("simplify");
        assert!(lines.iter().any(|line| line.starts_with("Result:")));
    }
}
