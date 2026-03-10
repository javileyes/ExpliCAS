use cas_api_models::StepJson;
use cas_solver_core::engine_events::EngineEvent;

use super::{EvalCommandOutput, EvalCommandRunConfig, PreparedEvalRun};

mod collect;
mod finalize;

pub(super) fn finalize_eval_run<F>(
    engine: &mut crate::Engine,
    config: EvalCommandRunConfig<'_>,
    collect_steps: F,
    prepared: PreparedEvalRun,
    total_us: u64,
) -> Result<EvalCommandOutput, String>
where
    F: Fn(&[crate::Step], &[EngineEvent], &cas_ast::Context, &str) -> Vec<StepJson>,
{
    let collected = collect::collect_eval_artifacts(
        &engine.simplifier.context,
        config.steps_mode,
        &prepared,
        total_us,
        collect_steps,
    );
    finalize::finalize_eval_collected(engine, config, prepared, collected)
}
