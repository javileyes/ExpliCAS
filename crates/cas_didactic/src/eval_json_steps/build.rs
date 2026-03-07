use cas_api_models::StepJson;
use cas_ast::Context;

pub(super) fn build_step_json(
    context: &Context,
    index: usize,
    enriched: &crate::didactic::EnrichedStep,
) -> StepJson {
    let step = &enriched.base_step;
    let before_expr = step.global_before.unwrap_or(step.before);
    let after_expr = step.global_after.unwrap_or(step.after);

    let focus_before = step.before_local().unwrap_or(step.before);
    let focus_after = step.after_local().unwrap_or(step.after);

    let before = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context,
            id: before_expr
        }
    );
    let after = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context,
            id: after_expr
        }
    );

    let expr_path = cas_solver::pathsteps_to_expr_path(step.path());
    let before_latex = crate::eval_json_render::render_step_path_latex(
        context,
        before_expr,
        expr_path.clone(),
        cas_formatter::HighlightColor::Red,
    );
    let after_latex = crate::eval_json_render::render_step_path_latex(
        context,
        after_expr,
        expr_path,
        cas_formatter::HighlightColor::Green,
    );
    let rule_latex =
        crate::eval_json_render::render_local_rule_latex(context, focus_before, focus_after);
    let substeps = crate::eval_json_render::collect_step_json_substeps(step, enriched);

    StepJson {
        index,
        rule: step.rule_name.clone(),
        rule_latex,
        before,
        after,
        before_latex,
        after_latex,
        substeps,
    }
}
