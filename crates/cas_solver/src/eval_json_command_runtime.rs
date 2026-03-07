mod prepare;
mod present;

use std::time::Instant;

use cas_api_models::{EvalJsonOutput, EvalJsonSessionRunConfig, StepJson};

struct PreparedEvalJsonRun {
    parsed_input: cas_ast::ExprId,
    parse_us: u64,
    simplify_us: u64,
    output_view: crate::EvalOutputView,
}

pub fn evaluate_eval_json_with_session<S, F>(
    engine: &mut crate::Engine,
    session: &mut S,
    config: EvalJsonSessionRunConfig<'_>,
    collect_steps: F,
) -> Result<EvalJsonOutput, String>
where
    S: crate::SolverEvalSession,
    F: Fn(&[crate::Step], &cas_ast::Context, &str) -> Vec<StepJson>,
{
    let total_start = Instant::now();
    let prepared = prepare::prepare_eval_json_run(engine, session, &config)?;
    present::finalize_eval_json_run(
        engine,
        config,
        collect_steps,
        prepared,
        total_start.elapsed().as_micros() as u64,
    )
}
