mod result;
mod steps;

use crate::{DisplayEvalSteps, EvalResult};
use cas_api_models::EngineWireStep;

pub fn render_eval_result(ctx: &mut cas_ast::Context, result: &EvalResult) -> String {
    result::render_eval_result(ctx, result)
}

pub fn build_engine_wire_steps(
    ctx: &mut cas_ast::Context,
    steps: &DisplayEvalSteps,
    steps_enabled: bool,
) -> Vec<EngineWireStep> {
    steps::build_engine_wire_steps(ctx, steps, steps_enabled)
}
