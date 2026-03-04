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

fn next_step_root(
    ctx: &mut cas_ast::Context,
    current_root: cas_ast::ExprId,
    step: &cas_solver::Step,
) -> cas_ast::ExprId {
    if let Some(global_after) = step.global_after {
        global_after
    } else {
        cas_solver::reconstruct_global_expr(ctx, current_root, step.path(), step.after)
    }
}

fn push_succinct_step_line(
    lines: &mut Vec<String>,
    ctx: &mut cas_ast::Context,
    root: cas_ast::ExprId,
) {
    lines.push(format!(
        "-> {}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: root
        }
    ));
}

fn push_detailed_step_lines(
    lines: &mut Vec<String>,
    ctx: &mut cas_ast::Context,
    step: &cas_solver::Step,
    style_prefs: &cas_formatter::StylePreferences,
    step_count: usize,
    current_root: cas_ast::ExprId,
) -> cas_ast::ExprId {
    lines.push(format!(
        "{}. {}  [{}]",
        step_count, step.description, step.rule_name
    ));

    let before_root = step.global_before.unwrap_or(current_root);
    lines.push(format!(
        "   Before: {}",
        cas_formatter::clean_display_string(&format!(
            "{}",
            cas_formatter::DisplayExprStyled::new(ctx, before_root, style_prefs)
        ))
    ));

    let (rule_before_id, rule_after_id) = match (step.before_local(), step.after_local()) {
        (Some(bl), Some(al)) => (bl, al),
        _ => (step.before, step.after),
    };

    let before_disp = cas_formatter::clean_display_string(&format!(
        "{}",
        cas_formatter::DisplayExprStyled::new(ctx, rule_before_id, style_prefs)
    ));
    let after_disp = cas_formatter::clean_display_string(&cas_formatter::render_with_rule_scope(
        ctx,
        rule_after_id,
        &step.rule_name,
        style_prefs,
    ));
    lines.push(format!("   Rule: {} -> {}", before_disp, after_disp));

    let next_root = next_step_root(ctx, current_root, step);
    lines.push(format!(
        "   After: {}",
        cas_formatter::clean_display_string(&format!(
            "{}",
            cas_formatter::DisplayExprStyled::new(ctx, next_root, style_prefs)
        ))
    ));

    for assumption_line in cas_solver::format_displayable_assumption_lines_for_step(step) {
        lines.push(format!("   {}", assumption_line));
    }

    next_root
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
                        current_root = next_step_root(ctx, current_root, step);
                        push_succinct_step_line(&mut lines, ctx, current_root);
                    } else {
                        current_root = push_detailed_step_lines(
                            &mut lines,
                            ctx,
                            step,
                            &style_prefs,
                            step_count,
                            current_root,
                        );
                    }
                } else {
                    current_root = next_step_root(ctx, current_root, step);
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
