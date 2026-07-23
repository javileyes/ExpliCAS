mod prepare;
mod present;

use web_time::Instant;

use cas_api_models::{EvalSessionRunConfig, StepWire};
use cas_solver_core::engine_events::EngineEvent;

use crate::eval_output_finalize::EvalOutputWire;

pub(super) type EvalCommandRunConfig<'a> = EvalSessionRunConfig<'a>;
pub(super) type EvalCommandOutput = EvalOutputWire;

struct PreparedEvalRun {
    parsed_input: cas_ast::ExprId,
    derive_target: Option<cas_ast::ExprId>,
    equiv_target: Option<cas_ast::ExprId>,
    style_signals: cas_formatter::ParseStyleSignals,
    parse_us: u64,
    simplify_us: u64,
    equivalence_diagnostics: Option<cas_api_models::EquivalenceDiagnosticsWire>,
    output_view: crate::EvalOutputView,
    events: Vec<EngineEvent>,
}

pub fn evaluate_eval_with_session<S, F>(
    engine: &mut crate::Engine,
    session: &mut S,
    config: EvalCommandRunConfig<'_>,
    language: cas_solver_core::eval_option_axes::Language,
    collect_steps: F,
) -> Result<EvalCommandOutput, String>
where
    S: crate::SolverEvalSession,
    F: Fn(&[crate::Step], &[EngineEvent], &cas_ast::Context, &str) -> Vec<StepWire>,
{
    let total_start = Instant::now();
    let prepared = prepare::prepare_eval_run(engine, session, &config)?;
    present::finalize_eval_run(
        engine,
        config,
        language,
        collect_steps,
        prepared,
        total_start.elapsed().as_micros() as u64,
    )
}
