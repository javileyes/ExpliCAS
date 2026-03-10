use std::time::Instant;

use cas_api_models::EvalJsonSessionRunConfig;
use cas_solver_core::engine_event_collector::EngineEventCollector;

use super::PreparedEvalRun;

pub(super) fn prepare_eval_run<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    config: &EvalJsonSessionRunConfig<'_>,
) -> Result<PreparedEvalRun, String>
where
    S: crate::SolverEvalSession,
{
    crate::eval_option_axes::apply_eval_option_axes(
        session.options_mut(),
        crate::eval_option_axes::EvalOptionAxes {
            context: config.context_mode,
            branch: config.branch_mode,
            complex: config.complex_mode,
            autoexpand: config.expand_policy,
            steps: config.steps_mode,
            domain: config.domain,
            value_domain: config.value_domain,
            inv_trig: config.inv_trig,
            complex_branch: config.complex_branch,
            assume_scope: config.assume_scope,
        },
    );

    let parse_start = Instant::now();
    let req = crate::eval_input::build_prepared_eval_request_for_input(
        config.expr,
        &mut engine.simplifier.context,
        config.auto_store,
    )?;
    let parsed_input = req.parsed();
    let parse_us = parse_start.elapsed().as_micros() as u64;

    let collector = EngineEventCollector::new();
    let previous_listener = engine
        .simplifier
        .replace_step_listener(Some(Box::new(collector.clone())));

    let simplify_start = Instant::now();
    let output_view_result =
        crate::eval_request_runtime::evaluate_prepared_request_with_session(engine, session, req);
    let _ = engine.simplifier.replace_step_listener(previous_listener);
    let output_view = output_view_result?;
    let simplify_us = simplify_start.elapsed().as_micros() as u64;

    Ok(PreparedEvalRun {
        parsed_input,
        parse_us,
        simplify_us,
        output_view,
        events: collector.into_events(),
    })
}
