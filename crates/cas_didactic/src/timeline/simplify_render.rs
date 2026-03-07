mod footer;
mod steps;

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

    let filtered_indices: HashSet<_> = filtered_steps
        .iter()
        .map(|step| *step as *const Step)
        .collect();
    let last_global_after = steps::render_timeline_filtered_steps(
        context,
        &mut html,
        steps,
        original_expr,
        style_prefs,
        enriched_steps,
        &display_hints,
        &filtered_indices,
    );

    footer::render_timeline_footer(
        context,
        &mut html,
        simplified_result.unwrap_or(last_global_after),
        global_requires,
        &display_hints,
        style_prefs,
    );

    html.push_str(
        r#"    </div>
"#,
    );
    html
}
