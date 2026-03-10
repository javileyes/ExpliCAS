mod prepare;
mod present;

use std::time::Instant;

use cas_api_models::{EvalJsonOutput, EvalJsonSessionRunConfig, StepJson};
use cas_solver_core::engine_events::EngineEvent;

struct PreparedEvalRun {
    parsed_input: cas_ast::ExprId,
    parse_us: u64,
    simplify_us: u64,
    output_view: crate::EvalOutputView,
    events: Vec<EngineEvent>,
}

pub fn evaluate_eval_with_session<S, F>(
    engine: &mut crate::Engine,
    session: &mut S,
    config: EvalJsonSessionRunConfig<'_>,
    collect_steps: F,
) -> Result<EvalJsonOutput, String>
where
    S: crate::SolverEvalSession,
    F: Fn(&[crate::Step], &[EngineEvent], &cas_ast::Context, &str) -> Vec<StepJson>,
{
    let total_start = Instant::now();
    let prepared = prepare::prepare_eval_run(engine, session, &config)?;
    present::finalize_eval_run(
        engine,
        config,
        collect_steps,
        prepared,
        total_start.elapsed().as_micros() as u64,
    )
}
