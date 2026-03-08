use cas_api_models::{EvalJsonOutput, EvalJsonSessionRunConfig, StepJson};
use cas_solver_core::engine_events::EngineEvent;

use super::PreparedEvalJsonRun;

mod collect;
mod finalize;

pub(super) fn finalize_eval_json_run<F>(
    engine: &mut crate::Engine,
    config: EvalJsonSessionRunConfig<'_>,
    collect_steps: F,
    prepared: PreparedEvalJsonRun,
    total_us: u64,
) -> Result<EvalJsonOutput, String>
where
    F: Fn(&[crate::Step], &[EngineEvent], &cas_ast::Context, &str) -> Vec<StepJson>,
{
    let collected = collect::collect_eval_json_artifacts(
        &engine.simplifier.context,
        config.steps_mode,
        &prepared,
        total_us,
        collect_steps,
    );
    finalize::finalize_eval_json_collected(engine, config, prepared, collected)
}
