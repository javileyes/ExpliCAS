use super::VerbosityLevel;
use crate::runtime::Step;
use cas_ast::{Context, ExprId};

pub(super) struct TimelineRenderData<'a> {
    pub filtered_steps: Vec<&'a Step>,
    pub enriched_steps: Vec<crate::didactic::EnrichedStep>,
}

pub(super) fn prepare_timeline_render_data<'a>(
    context: &mut Context,
    steps: &'a [Step],
    original_expr: ExprId,
    verbosity: VerbosityLevel,
) -> TimelineRenderData<'a> {
    let filtered_steps = steps
        .iter()
        .filter(|step| crate::didactic::step_matches_visibility(step, verbosity.step_visibility()))
        .collect();
    let enriched_steps = crate::didactic::enrich_steps(context, original_expr, steps.to_vec());

    TimelineRenderData {
        filtered_steps,
        enriched_steps,
    }
}
