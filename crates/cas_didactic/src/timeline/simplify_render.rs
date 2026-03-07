use super::simplify_highlights::{
    render_timeline_step_math, resolve_timeline_step_global_snapshots,
};
use super::simplify_step_html::render_timeline_step_html;
use super::simplify_substeps::{
    render_timeline_domain_assumptions_html, render_timeline_enriched_substeps_html,
    render_timeline_rule_substeps_html, TimelineSubstepsRenderState,
};
use super::simplify_summary::{
    render_timeline_final_result_html, render_timeline_global_requires_html,
};
use cas_ast::{Context, ExprId};
use cas_solver::{ImplicitCondition, Step};
use std::collections::HashSet;

pub(super) fn render_timeline_filtered_enriched(
    context: &mut Context,
    steps: &[Step],
    original_expr: ExprId,
    simplified_result: Option<ExprId>,
    global_requires: &[ImplicitCondition],
    style_prefs: &cas_formatter::root_style::StylePreferences,
    filtered_steps: &[&Step],
    enriched_steps: &[crate::didactic::EnrichedStep],
) -> String {
    let mut html = String::from("        <div class=\"timeline\">\n");

    let display_hints = cas_formatter::build_display_context_with_result(
        context,
        original_expr,
        steps,
        simplified_result,
    );

    let mut step_number = 0;
    let mut last_global_after = original_expr;
    let mut substeps_state = TimelineSubstepsRenderState::default();
    let filtered_indices: HashSet<_> = filtered_steps.iter().map(|s| *s as *const Step).collect();

    for (step_idx, step) in steps.iter().enumerate() {
        let snapshots =
            resolve_timeline_step_global_snapshots(context, steps, original_expr, step_idx, step);
        last_global_after = snapshots.global_after_expr;

        let step_ptr = step as *const Step;
        if !filtered_indices.contains(&step_ptr) {
            continue;
        }
        step_number += 1;

        let rendered_step_math =
            render_timeline_step_math(context, step, snapshots, &display_hints, style_prefs);

        let sub_steps_html = enriched_steps
            .get(step_idx)
            .map(|enriched| render_timeline_enriched_substeps_html(enriched, &mut substeps_state))
            .unwrap_or_default();

        let rule_substeps_html = render_timeline_rule_substeps_html(step);
        let domain_html = render_timeline_domain_assumptions_html(step);
        html.push_str(&render_timeline_step_html(
            step_number,
            step,
            &rendered_step_math,
            &sub_steps_html,
            &rule_substeps_html,
            &domain_html,
        ));
    }

    let final_result_expr = simplified_result.unwrap_or(last_global_after);
    html.push_str(&render_timeline_final_result_html(
        context,
        final_result_expr,
        &display_hints,
        style_prefs,
    ));
    html.push_str(&render_timeline_global_requires_html(
        context,
        global_requires,
    ));

    html.push_str(
        r#"    </div>
"#,
    );
    html
}
