use std::time::Instant;

use cas_api_models::{EvalJsonOutput, StepJson, TimingsJson};

/// Configuration for session-backed eval-json execution.
#[derive(Debug, Clone, Copy)]
pub struct EvalJsonSessionRunConfig<'a> {
    pub expr: &'a str,
    pub auto_store: bool,
    pub max_chars: usize,
    pub steps_mode: &'a str,
    pub budget_preset: &'a str,
    pub strict: bool,
    pub domain: &'a str,
    pub context_mode: &'a str,
    pub branch_mode: &'a str,
    pub expand_policy: &'a str,
    pub complex_mode: &'a str,
    pub const_fold: &'a str,
    pub value_domain: &'a str,
    pub complex_branch: &'a str,
    pub inv_trig: &'a str,
    pub assume_scope: &'a str,
}

/// Evaluate eval-json request end-to-end with a mutable session.
///
/// This helper centralizes JSON orchestration:
/// parse/build request, engine eval, timings, warnings/requires extraction, and
/// final payload assembly.
pub fn evaluate_eval_json_with_session<S, F>(
    engine: &mut crate::Engine,
    session: &mut S,
    config: EvalJsonSessionRunConfig<'_>,
    collect_steps: F,
) -> Result<EvalJsonOutput, String>
where
    S: cas_engine::EvalSession<
        Options = cas_engine::EvalOptions,
        Diagnostics = cas_engine::Diagnostics,
    >,
    S::Store: cas_engine::EvalStore<
        DomainMode = cas_engine::DomainMode,
        RequiredItem = cas_engine::RequiredItem,
        Step = cas_engine::Step,
        Diagnostics = cas_engine::Diagnostics,
    >,
    F: Fn(&crate::EvalOutput, &cas_ast::Context, &str) -> Vec<StepJson>,
{
    let total_start = Instant::now();

    crate::json::apply_eval_json_options(
        session.options_mut(),
        config.context_mode,
        config.branch_mode,
        config.complex_mode,
        config.expand_policy,
        config.steps_mode,
        config.domain,
        config.value_domain,
        config.inv_trig,
        config.complex_branch,
        config.assume_scope,
    );

    let parse_start = Instant::now();
    let req = crate::json::build_eval_request_for_input(
        config.expr,
        &mut engine.simplifier.context,
        config.auto_store,
    )?;
    let parsed_input = req.parsed;
    let parse_us = parse_start.elapsed().as_micros() as u64;

    let simplify_start = Instant::now();
    let output = engine.eval(session, req).map_err(|e| e.to_string())?;
    let simplify_us = simplify_start.elapsed().as_micros() as u64;

    let input_latex = Some(crate::json::format_eval_input_latex(
        &engine.simplifier.context,
        parsed_input,
    ));
    let steps = collect_steps(&output, &engine.simplifier.context, config.steps_mode);
    let solve_steps = crate::json::collect_solve_steps_eval_json(
        &output,
        &engine.simplifier.context,
        config.steps_mode,
    );
    let warnings = crate::json::collect_warnings_eval_json(&output);
    let required_conditions =
        crate::json::collect_required_conditions_eval_json(&output, &engine.simplifier.context);
    let required_display =
        crate::json::collect_required_display_eval_json(&output, &engine.simplifier.context);
    let timings_us = TimingsJson {
        parse_us,
        simplify_us,
        total_us: total_start.elapsed().as_micros() as u64,
    };

    crate::json::finalize_eval_json_output(crate::json::EvalJsonFinalizeInput {
        result: &output.result,
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
        raw_steps_count: output.steps.len(),
        raw_solve_steps_count: output.solve_steps.len(),
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

#[cfg(test)]
mod tests {
    use super::{evaluate_eval_json_with_session, EvalJsonSessionRunConfig};

    #[test]
    fn evaluate_eval_json_with_session_runs() {
        let mut engine = crate::Engine::new();
        let mut session = cas_session::SessionState::new();
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
            |_output, _context, _steps_mode| Vec::new(),
        )
        .expect("eval-json");

        assert!(out.ok);
        assert!(out.result.contains("2 * x"));
    }
}
