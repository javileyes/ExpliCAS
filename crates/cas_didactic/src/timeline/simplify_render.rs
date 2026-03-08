mod footer;
mod prepare;
mod shell;
mod steps;

use cas_ast::{Context, ExprId};
use cas_solver::ImplicitCondition;

#[allow(clippy::too_many_arguments)]
pub(super) fn render_timeline_filtered_enriched(
    context: &mut Context,
    steps: &[cas_solver::Step],
    original_expr: ExprId,
    simplified_result: Option<ExprId>,
    global_requires: &[ImplicitCondition],
    style_prefs: &cas_formatter::root_style::StylePreferences,
    filtered_steps: &[&cas_solver::Step],
    enriched_steps: &[crate::didactic::EnrichedStep],
) -> String {
    let mut html = shell::open_timeline_html();
    let prepared = prepare::prepare_timeline_render(
        context,
        &mut html,
        original_expr,
        steps,
        simplified_result,
        filtered_steps,
    );
    let last_global_after = steps::render_timeline_filtered_steps(
        context,
        &mut html,
        steps,
        original_expr,
        style_prefs,
        enriched_steps,
        &prepared.display_hints,
        &prepared.filtered_indices,
    );

    footer::render_timeline_footer(
        context,
        &mut html,
        simplified_result.unwrap_or(last_global_after),
        global_requires,
        &prepared.display_hints,
        style_prefs,
    );

    html.push_str(shell::TIMELINE_CLOSING_HTML);
    html
}
