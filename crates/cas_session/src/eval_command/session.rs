use cas_engine::Step;
use cas_formatter::{DisplayExpr, LaTeXExpr};
use cas_solver::runtime::Engine;
use cas_solver::session_api::bindings::{
    evaluate_let_assignment_command, parse_let_assignment_input,
};
use cas_solver_core::engine_events::EngineEvent;
use std::path::Path;
use std::time::Instant;

/// Session-backed config for eval command orchestration.
pub type EvalCommandConfig<'a> = cas_api_models::EvalSessionRunConfig<'a>;
type EvalOutputWire = cas_api_models::EvalWireOutput;
type EvalCommandResult = Result<EvalOutputWire, String>;

fn parse_lazy_assignment(
    expr: &str,
) -> Option<cas_solver::session_api::bindings::ParsedLetAssignment<'_>> {
    if !expr.contains(":=") {
        return None;
    }
    parse_let_assignment_input(expr)
        .ok()
        .filter(|parsed| parsed.lazy)
}

fn build_assignment_wire_output(
    engine: &Engine,
    input: &str,
    config: EvalCommandConfig<'_>,
    output: &cas_api_models::AssignmentCommandOutput,
    total_us: u64,
) -> EvalOutputWire {
    let ctx = &engine.simplifier.context;
    let result = DisplayExpr {
        context: ctx,
        id: output.expr,
    }
    .to_string();
    let result_latex = Some(
        LaTeXExpr {
            context: ctx,
            id: output.expr,
        }
        .to_latex(),
    );
    let (node_count, depth) = cas_ast::traversal::count_nodes_and_max_depth(ctx, output.expr);

    cas_api_models::EvalWireOutput::from_build(cas_api_models::EvalOutputBuild {
        input,
        input_latex: None,
        stored_id: None,
        result_chars: result.chars().count(),
        result,
        result_truncated: false,
        result_latex,
        steps_mode: config.steps_mode.as_str(),
        steps_count: 0,
        steps: Vec::new(),
        solve_steps: Vec::new(),
        warnings: Vec::new(),
        required_conditions: Vec::new(),
        required_display: Vec::new(),
        budget_preset: config.budget_preset.as_str(),
        strict: config.strict,
        domain: config.domain.as_str(),
        stats: cas_api_models::ExprStatsWire {
            node_count,
            depth,
            term_count: None,
        },
        hash: None,
        timings_us: cas_api_models::TimingsWire {
            parse_us: 0,
            simplify_us: total_us,
            total_us,
        },
        context_mode: config.context_mode.as_str(),
        branch_mode: config.branch_mode.as_str(),
        expand_policy: config.expand_policy.as_str(),
        complex_mode: config.complex_mode.as_str(),
        const_fold: config.const_fold.as_str(),
        value_domain: config.value_domain.as_str(),
        complex_branch: config.complex_branch.as_str(),
        inv_trig: config.inv_trig.as_str(),
        assume_scope: config.assume_scope.as_str(),
        wire: None,
    })
}

fn evaluate_assignment_command_in_memory(config: EvalCommandConfig<'_>) -> EvalCommandResult {
    let mut engine = Engine::new();
    let mut state = crate::state_core::SessionState::new();
    let started = Instant::now();
    let output = evaluate_let_assignment_command(&mut state, &mut engine.simplifier, config.expr)?;
    let total_us = started.elapsed().as_micros() as u64;
    Ok(build_assignment_wire_output(
        &engine,
        config.expr,
        config,
        &output,
        total_us,
    ))
}

fn evaluate_assignment_command_with_persisted_session(
    session_path: Option<&Path>,
    config: EvalCommandConfig<'_>,
) -> (EvalCommandResult, Option<String>, Option<String>) {
    crate::session_io::run_with_domain_session(
        session_path,
        config.domain.as_str(),
        |engine, state| {
            let started = Instant::now();
            let output =
                evaluate_let_assignment_command(state, &mut engine.simplifier, config.expr)?;
            let total_us = started.elapsed().as_micros() as u64;
            Ok(build_assignment_wire_output(
                engine,
                config.expr,
                config,
                &output,
                total_us,
            ))
        },
    )
}

fn can_skip_persisted_session_state(expr: &str, auto_store: bool) -> bool {
    !auto_store && !expr.contains('#')
}

fn should_use_read_only_persisted_session(expr: &str, auto_store: bool) -> bool {
    !auto_store && expr.contains('#')
}

/// Evaluate `eval` using optional persisted session state.
///
/// Keeps CLI/frontends thin by centralizing session load/run/save orchestration.
pub fn evaluate_eval_command_with_session<F>(
    session_path: Option<&Path>,
    config: EvalCommandConfig<'_>,
    collect_steps: F,
) -> (EvalCommandResult, Option<String>, Option<String>)
where
    F: Fn(&[Step], &[EngineEvent], &cas_ast::Context, &str) -> Vec<cas_api_models::StepWire>,
{
    if parse_lazy_assignment(config.expr).is_some() {
        if session_path.is_none() {
            return (evaluate_assignment_command_in_memory(config), None, None);
        }
        return evaluate_assignment_command_with_persisted_session(session_path, config);
    }

    if session_path.is_none() || can_skip_persisted_session_state(config.expr, config.auto_store) {
        let mut engine = Engine::new();
        let mut state = crate::state_core::SessionState::new();
        let output = cas_solver::session_api::eval::evaluate_eval_with_session(
            &mut engine,
            &mut state,
            config,
            |steps, events, ctx, mode| collect_steps(steps, events, ctx, mode),
        );
        return (output, None, None);
    }

    if should_use_read_only_persisted_session(config.expr, config.auto_store) {
        return crate::session_io::run_read_only_with_domain_session(
            session_path,
            config.domain.as_str(),
            |engine, state| {
                cas_solver::session_api::eval::evaluate_eval_with_session(
                    engine,
                    state,
                    config,
                    |steps, events, ctx, mode| collect_steps(steps, events, ctx, mode),
                )
            },
        );
    }

    crate::session_io::run_with_domain_session(
        session_path,
        config.domain.as_str(),
        |engine, state| {
            cas_solver::session_api::eval::evaluate_eval_with_session(
                engine,
                state,
                config,
                |steps, events, ctx, mode| collect_steps(steps, events, ctx, mode),
            )
        },
    )
}
