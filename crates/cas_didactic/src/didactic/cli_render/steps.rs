use super::super::display_policy::{
    render_cli_enriched_substeps_lines, CliSubstepsRenderState, StepDisplayMode,
};
use super::super::enrich_steps;
use super::super::step_visibility::should_show_simplify_step;
use cas_ast::{Context, ExprId};
use cas_solver::Step;

pub(super) fn render_simplification_step_lines(
    ctx: &mut Context,
    expr: ExprId,
    steps: &[Step],
    style_prefs: &cas_formatter::root_style::StylePreferences,
    display_mode: StepDisplayMode,
) -> Vec<String> {
    use cas_formatter::DisplayExprStyled;

    let mut lines = Vec::new();
    if display_mode != StepDisplayMode::Succinct {
        lines.push("Steps:".to_string());
    }

    let enriched_steps = enrich_steps(ctx, expr, steps.to_vec());
    let mut current_root = expr;
    let mut step_count = 0;
    let mut cli_substeps_state = CliSubstepsRenderState::default();

    for (step_idx, step) in steps.iter().enumerate() {
        if should_show_simplify_step(step, display_mode) {
            let before_disp = cas_formatter::clean_display_string(&format!(
                "{}",
                DisplayExprStyled::new(ctx, step.before, style_prefs)
            ));
            let after_disp = cas_formatter::clean_display_string(&format!(
                "{}",
                DisplayExprStyled::new(ctx, step.after, style_prefs)
            ));

            if before_disp == after_disp {
                if let Some(global_after) = step.global_after {
                    current_root = global_after;
                }
                continue;
            }

            step_count += 1;

            if display_mode == StepDisplayMode::Succinct {
                current_root =
                    cas_solver::reconstruct_global_expr(ctx, current_root, step.path(), step.after);
                lines.push(format!(
                    "-> {}",
                    DisplayExprStyled::new(ctx, current_root, style_prefs)
                ));
                continue;
            }

            lines.push(format!(
                "{}. {}  [{}]",
                step_count, step.description, step.rule_name
            ));

            if let Some(global_before) = step.global_before {
                lines.push(format!(
                    "   Before: {}",
                    cas_formatter::clean_display_string(&format!(
                        "{}",
                        DisplayExprStyled::new(ctx, global_before, style_prefs)
                    ))
                ));
            } else {
                lines.push(format!(
                    "   Before: {}",
                    cas_formatter::clean_display_string(&format!(
                        "{}",
                        DisplayExprStyled::new(ctx, current_root, style_prefs)
                    ))
                ));
            }

            if let Some(enriched_step) = enriched_steps.get(step_idx) {
                lines.extend(render_cli_enriched_substeps_lines(
                    enriched_step,
                    &mut cli_substeps_state,
                ));
            }

            let (rule_before_id, rule_after_id) = match (step.before_local(), step.after_local()) {
                (Some(bl), Some(al)) => (bl, al),
                _ => (step.before, step.after),
            };

            let before_disp = cas_formatter::clean_display_string(&format!(
                "{}",
                DisplayExprStyled::new(ctx, rule_before_id, style_prefs)
            ));
            let after_disp =
                cas_formatter::clean_display_string(&cas_formatter::render_with_rule_scope(
                    ctx,
                    rule_after_id,
                    &step.rule_name,
                    style_prefs,
                ));

            if before_disp == after_disp {
                if let Some(global_after) = step.global_after {
                    current_root = global_after;
                }
                continue;
            }

            lines.push(format!("   Rule: {} -> {}", before_disp, after_disp));

            if !step.substeps().is_empty() {
                for substep in step.substeps() {
                    lines.push(format!("   [{}]", substep.title));
                    for line in &substep.lines {
                        lines.push(format!("      • {}", line));
                    }
                }
            }

            if let Some(global_after) = step.global_after {
                current_root = global_after;
            } else {
                current_root =
                    cas_solver::reconstruct_global_expr(ctx, current_root, step.path(), step.after);
            }

            lines.push(format!(
                "   After: {}",
                cas_formatter::clean_display_string(&format!(
                    "{}",
                    DisplayExprStyled::new(ctx, current_root, style_prefs)
                ))
            ));

            for assumption_line in cas_solver::format_displayable_assumption_lines_for_step(step) {
                lines.push(format!("   {}", assumption_line));
            }
        } else if let Some(global_after) = step.global_after {
            current_root = global_after;
        } else {
            current_root =
                cas_solver::reconstruct_global_expr(ctx, current_root, step.path(), step.after);
        }
    }

    lines
}
