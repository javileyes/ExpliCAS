use std::time::Instant;

use cas_api_models::EvalStepsMode;
use cas_formatter::{DisplayExpr, LaTeXExpr, ParseStyleSignals};
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

    if let Some(name) = cas_session_core::eval::first_unknown_function_name(
        session,
        &engine.simplifier.context,
        hit.resolved,
    ) {
        return Err(format!("Error: {}", crate::CasError::UnknownFunction(name)));
    }

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
    let mut output_view = crate::EvalOutputView {
        stored_id: None,
        parsed: *parsed,
        resolved: hit.resolved,
        result: crate::EvalResult::Expr(hit.resolved),
        strategy: None,
        steps: crate::DisplayEvalSteps::default(),
        solve_steps: Vec::new(),
        output_scopes: Vec::new(),
        diagnostics,
        required_conditions,
        domain_warnings: Vec::new(),
        blocked_hints: Vec::new(),
        solver_assumptions: Vec::new(),
    };
    crate::eval_request_runtime::augment_output_view_required_conditions(
        &mut engine.simplifier.context,
        &mut output_view,
    );
    Ok(Some(output_view))
}

fn collect_equivalence_diagnostics(
    engine: &mut crate::Engine,
    output_view: &crate::EvalOutputView,
    equiv_target: Option<cas_ast::ExprId>,
) -> Option<cas_api_models::EquivalenceDiagnosticsWire> {
    if !matches!(output_view.result, crate::EvalResult::Bool(false)) {
        return None;
    }

    let rhs = equiv_target?;
    let residual = crate::equiv_command::simplified_equivalence_residual_expr(
        &mut engine.simplifier,
        output_view.resolved,
        rhs,
    );
    let ctx = &engine.simplifier.context;

    Some(cas_api_models::EquivalenceDiagnosticsWire {
        residual: DisplayExpr {
            context: ctx,
            id: residual,
        }
        .to_string(),
        residual_latex: Some(
            LaTeXExpr {
                context: ctx,
                id: residual,
            }
            .to_latex(),
        ),
    })
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
            const_fold: config.const_fold,
            autoexpand: config.expand_policy,
            steps: config.steps_mode,
            domain: config.domain,
            value_domain: config.value_domain,
            inv_trig: config.inv_trig,
            complex_branch: config.complex_branch,
            assume_scope: config.assume_scope,
        },
    );
    session.options_mut().time_budget_ms = config.time_budget_ms;

    let parse_start = Instant::now();
    let style_signals = ParseStyleSignals::from_input_string(config.expr);
    let req = crate::eval_input::build_prepared_eval_request_for_input(
        config.expr,
        &mut engine.simplifier.context,
        config.auto_store,
    )?;
    let parsed_input = req.parsed();
    let derive_target = match &req {
        crate::eval_input::PreparedEvalRequest::Derive { target, .. } => Some(*target),
        _ => None,
    };
    let equiv_target = match &req {
        crate::eval_input::PreparedEvalRequest::Eval {
            action: crate::eval_input::EvalNonSolveAction::Equiv { other },
            ..
        } => Some(*other),
        _ => None,
    };
    let parse_us = parse_start.elapsed().as_micros() as u64;

    let simplify_start = Instant::now();
    if let Some(output_view) = try_direct_cached_eval_run(engine, session, &req, config.steps_mode)?
    {
        return Ok(PreparedEvalRun {
            parsed_input,
            derive_target,
            equiv_target,
            style_signals,
            parse_us,
            simplify_us: simplify_start.elapsed().as_micros() as u64,
            equivalence_diagnostics: collect_equivalence_diagnostics(
                engine,
                &output_view,
                equiv_target,
            ),
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
    let equivalence_diagnostics =
        collect_equivalence_diagnostics(engine, &output_view, equiv_target);

    Ok(PreparedEvalRun {
        parsed_input,
        derive_target,
        equiv_target,
        style_signals,
        parse_us,
        simplify_us,
        equivalence_diagnostics,
        output_view,
        events: collector
            .map(EngineEventCollector::into_events)
            .unwrap_or_default(),
    })
}
