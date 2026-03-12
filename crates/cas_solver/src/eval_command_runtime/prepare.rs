use std::time::Instant;

use cas_api_models::EvalStepsMode;
use cas_solver_core::engine_event_collector::EngineEventCollector;

use super::{EvalCommandRunConfig, PreparedEvalRun};

fn try_direct_cached_eval_run<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    req: &crate::eval_input::PreparedEvalRequest,
    steps_mode: EvalStepsMode,
) -> Result<Option<crate::EvalOutputView>, String>
where
    S: crate::SolverEvalSession,
{
    if !matches!(steps_mode, EvalStepsMode::Off) {
        return Ok(None);
    }

    let crate::eval_input::PreparedEvalRequest::Eval {
        parsed,
        action: crate::eval_input::EvalNonSolveAction::Simplify,
        auto_store,
        ..
    } = req
    else {
        return Ok(None);
    };

    let Some(hit) = session
        .try_direct_cached_eval(&mut engine.simplifier.context, *parsed, *auto_store)
        .map_err(|e| format!("Resolution error: {}", e))?
    else {
        return Ok(None);
    };

    let mut diagnostics = crate::Diagnostics::new();
    let input_domain = crate::infer_implicit_domain(
        &engine.simplifier.context,
        hit.resolved,
        crate::ValueDomain::RealOnly,
    );
    diagnostics.extend_required(
        input_domain.conditions().iter().cloned(),
        crate::RequireOrigin::InputImplicit,
    );
    let output_domain = crate::infer_implicit_domain(
        &engine.simplifier.context,
        hit.resolved,
        crate::ValueDomain::RealOnly,
    );
    diagnostics.extend_required(
        output_domain.conditions().iter().cloned(),
        crate::RequireOrigin::OutputImplicit,
    );
    diagnostics.inherit_requires_from(&hit.inherited_diagnostics);
    diagnostics.dedup_and_sort(&engine.simplifier.context);

    let required_conditions = diagnostics.required_conditions();
    Ok(Some(crate::EvalOutputView {
        stored_id: None,
        parsed: *parsed,
        resolved: hit.resolved,
        result: crate::EvalResult::Expr(hit.resolved),
        steps: crate::DisplayEvalSteps::default(),
        solve_steps: Vec::new(),
        output_scopes: Vec::new(),
        diagnostics,
        required_conditions,
        domain_warnings: Vec::new(),
        blocked_hints: Vec::new(),
        solver_assumptions: Vec::new(),
    }))
}

pub(super) fn prepare_eval_run<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    config: &EvalCommandRunConfig<'_>,
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

    let simplify_start = Instant::now();
    if let Some(output_view) = try_direct_cached_eval_run(engine, session, &req, config.steps_mode)?
    {
        return Ok(PreparedEvalRun {
            parsed_input,
            parse_us,
            simplify_us: simplify_start.elapsed().as_micros() as u64,
            output_view,
            events: Vec::new(),
        });
    }

    let (collector, previous_listener) = if matches!(config.steps_mode, EvalStepsMode::Off) {
        (None, None)
    } else {
        let collector = EngineEventCollector::new();
        let previous_listener = engine
            .simplifier
            .replace_step_listener(Some(Box::new(collector.clone())));
        (Some(collector), Some(previous_listener))
    };

    let output_view_result =
        crate::eval_request_runtime::evaluate_prepared_request_with_session(engine, session, req);
    if let Some(previous_listener) = previous_listener {
        let _ = engine.simplifier.replace_step_listener(previous_listener);
    }
    let output_view = output_view_result?;
    let simplify_us = simplify_start.elapsed().as_micros() as u64;

    Ok(PreparedEvalRun {
        parsed_input,
        parse_us,
        simplify_us,
        output_view,
        events: collector
            .map(EngineEventCollector::into_events)
            .unwrap_or_default(),
    })
}
