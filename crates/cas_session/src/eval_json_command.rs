//! Session-backed eval-json command orchestration.

use std::path::Path;
use std::time::Instant;

use cas_api_models::{EvalJsonOutput, EvalJsonSessionRunConfig, StepJson, TimingsJson};

use crate::eval_json_finalize::{finalize_eval_json_output, EvalJsonFinalizeInput};
use crate::eval_json_presentation::{
    collect_required_conditions_eval_json, collect_required_display_eval_json,
    collect_solve_steps_eval_json, collect_warnings_eval_json, format_eval_input_latex,
};

/// Session-backed config for `eval-json` command orchestration.
pub type EvalJsonCommandConfig<'a> = cas_api_models::EvalJsonSessionRunConfig<'a>;

fn evaluate_eval_json_with_session<S, F>(
    engine: &mut cas_solver::Engine,
    session: &mut S,
    config: EvalJsonSessionRunConfig<'_>,
    collect_steps: F,
) -> Result<EvalJsonOutput, String>
where
    S: cas_solver::EvalSession<
        Options = cas_solver::EvalOptions,
        Diagnostics = cas_solver::Diagnostics,
    >,
    S::Store: cas_solver::EvalStore<
        DomainMode = cas_solver::DomainMode,
        RequiredItem = cas_solver::RequiredItem,
        Step = cas_solver::Step,
        Diagnostics = cas_solver::Diagnostics,
    >,
    F: Fn(&[cas_solver::Step], &cas_ast::Context, &str) -> Vec<StepJson>,
{
    let total_start = Instant::now();

    crate::eval_json_options::apply_eval_json_options(
        session.options_mut(),
        crate::eval_json_options::EvalJsonOptionAxes {
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
    let req = crate::eval_json_input::build_eval_request_for_input(
        config.expr,
        &mut engine.simplifier.context,
        config.auto_store,
    )?;
    let parsed_input = req.parsed;
    let parse_us = parse_start.elapsed().as_micros() as u64;

    let simplify_start = Instant::now();
    let output = engine.eval(session, req).map_err(|e| e.to_string())?;
    let simplify_us = simplify_start.elapsed().as_micros() as u64;
    let output_view = cas_solver::eval_output_view(&output);

    let input_latex = Some(format_eval_input_latex(
        &engine.simplifier.context,
        parsed_input,
    ));
    let steps_raw = output_view.steps.as_slice();
    let solve_steps_raw = output_view.solve_steps.as_slice();
    let steps = collect_steps(steps_raw, &engine.simplifier.context, config.steps_mode);
    let solve_steps = collect_solve_steps_eval_json(
        solve_steps_raw,
        &engine.simplifier.context,
        config.steps_mode,
    );
    let warnings = collect_warnings_eval_json(&output_view.domain_warnings);
    let required_conditions_raw = output_view.required_conditions.as_slice();
    let required_conditions =
        collect_required_conditions_eval_json(required_conditions_raw, &engine.simplifier.context);
    let required_display =
        collect_required_display_eval_json(required_conditions_raw, &engine.simplifier.context);
    let timings_us = TimingsJson {
        parse_us,
        simplify_us,
        total_us: total_start.elapsed().as_micros() as u64,
    };

    finalize_eval_json_output(EvalJsonFinalizeInput {
        result: &output_view.result,
        ctx: &engine.simplifier.context,
        max_chars: config.max_chars,
        input: config.expr,
        input_latex,
        steps_mode: config.steps_mode,
        steps,
        solve_steps,
        warnings,
        required_conditions,
        required_display,
        raw_steps_count: steps_raw.len(),
        raw_solve_steps_count: solve_steps_raw.len(),
        budget_preset: config.budget_preset,
        strict: config.strict,
        domain: config.domain,
        timings_us,
        context_mode: config.context_mode,
        branch_mode: config.branch_mode,
        expand_policy: config.expand_policy,
        complex_mode: config.complex_mode,
        const_fold: config.const_fold,
        value_domain: config.value_domain,
        complex_branch: config.complex_branch,
        inv_trig: config.inv_trig,
        assume_scope: config.assume_scope,
    })
}

/// Evaluate `eval-json` using optional persisted session state.
///
/// Keeps CLI/frontends thin by centralizing session load/run/save orchestration.
pub fn evaluate_eval_json_command_with_session<F>(
    session_path: Option<&Path>,
    config: EvalJsonCommandConfig<'_>,
    collect_steps: F,
) -> (
    Result<cas_api_models::EvalJsonOutput, String>,
    Option<String>,
    Option<String>,
)
where
    F: Fn(&[cas_solver::Step], &cas_ast::Context, &str) -> Vec<cas_api_models::StepJson>,
{
    crate::run_with_domain_session(session_path, config.domain, |engine, state| {
        evaluate_eval_json_with_session(engine, state, config, |steps, ctx, mode| {
            collect_steps(steps, ctx, mode)
        })
    })
}

/// Evaluate `eval-json` and always return a pretty JSON string.
///
/// Successful runs return canonical JSON payload. Errors are normalized into
/// canonical JSON error output.
pub fn evaluate_eval_json_command_pretty_with_session<F>(
    session_path: Option<&Path>,
    config: EvalJsonCommandConfig<'_>,
    collect_steps: F,
) -> String
where
    F: Fn(&[cas_solver::Step], &cas_ast::Context, &str) -> Vec<cas_api_models::StepJson>,
{
    let input = config.expr.to_string();
    let (output, _, _) =
        evaluate_eval_json_command_with_session(session_path, config, collect_steps);
    match output {
        Ok(payload) => payload.to_json_pretty(),
        Err(error) => cas_api_models::ErrorJsonOutput::from_eval_error_message(&error, &input)
            .to_json_pretty(),
    }
}

#[cfg(test)]
mod tests {
    use super::{evaluate_eval_json_with_session, EvalJsonSessionRunConfig};

    #[test]
    fn evaluate_eval_json_with_session_runs() {
        let mut engine = cas_solver::Engine::new();
        let mut session = crate::SessionState::new();
        let out = evaluate_eval_json_with_session(
            &mut engine,
            &mut session,
            EvalJsonSessionRunConfig {
                expr: "x + x",
                auto_store: false,
                max_chars: 2000,
                steps_mode: "off",
                budget_preset: "standard",
                strict: false,
                domain: "generic",
                context_mode: "auto",
                branch_mode: "strict",
                expand_policy: "off",
                complex_mode: "auto",
                const_fold: "off",
                value_domain: "real",
                complex_branch: "principal",
                inv_trig: "strict",
                assume_scope: "real",
            },
            |_steps, _context, _steps_mode| Vec::new(),
        )
        .expect("eval-json");

        assert!(out.ok);
        assert!(out.result.contains("2 * x"));
    }
}
