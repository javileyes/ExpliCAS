use cas_api_models::StepJson;
use cas_ast::Context;
use cas_solver::{pathsteps_to_expr_path, Step};

/// Convert engine steps to eval-json step payloads.
///
/// Keeps JSON step formatting behavior consistent with timeline rendering.
pub fn collect_eval_json_steps(steps: &[Step], ctx: &Context, steps_mode: &str) -> Vec<StepJson> {
    if steps_mode != "on" {
        return vec![];
    }

    let filtered = crate::didactic::clone_steps_matching_visibility(
        steps,
        crate::didactic::StepVisibility::MediumOrHigher,
    );

    if filtered.is_empty() {
        return vec![];
    }

    let original_expr = match crate::didactic::infer_original_expr_for_steps(&filtered) {
        Some(expr) => expr,
        None => return vec![],
    };
    let enriched_steps = crate::didactic::enrich_steps(ctx, original_expr, filtered.clone());

    enriched_steps
        .iter()
        .enumerate()
        .map(|(i, enriched)| {
            let step = &enriched.base_step;

            let before_expr = step.global_before.unwrap_or(step.before);
            let after_expr = step.global_after.unwrap_or(step.after);

            let focus_before = step.before_local().unwrap_or(step.before);
            let focus_after = step.after_local().unwrap_or(step.after);

            let before_str = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: before_expr
                }
            );
            let after_str = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: after_expr
                }
            );

            let expr_path = pathsteps_to_expr_path(step.path());
            let before_latex = crate::eval_json_render::render_step_path_latex(
                ctx,
                before_expr,
                expr_path.clone(),
                cas_formatter::HighlightColor::Red,
            );
            let after_latex = crate::eval_json_render::render_step_path_latex(
                ctx,
                after_expr,
                expr_path,
                cas_formatter::HighlightColor::Green,
            );
            let rule_latex =
                crate::eval_json_render::render_local_rule_latex(ctx, focus_before, focus_after);
            let substeps = crate::eval_json_render::collect_step_json_substeps(step, enriched);

            StepJson {
                index: i + 1,
                rule: step.rule_name.clone(),
                rule_latex,
                before: before_str,
                after: after_str,
                before_latex,
                after_latex,
                substeps,
            }
        })
        .collect()
}
