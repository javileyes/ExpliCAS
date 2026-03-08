mod build;
mod prepare;

use cas_api_models::StepJson;
use cas_ast::Context;
use cas_solver::Step;

/// Convert engine steps to eval-json step payloads.
///
/// Keeps JSON step formatting behavior consistent with timeline rendering.
pub fn collect_eval_json_steps(steps: &[Step], ctx: &Context, steps_mode: &str) -> Vec<StepJson> {
    prepare::prepare_eval_json_steps(steps, ctx, steps_mode)
        .iter()
        .enumerate()
        .map(|(index, enriched)| build::build_step_json(ctx, index + 1, enriched))
        .collect()
}
