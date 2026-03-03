//! Runtime adapters for REPL command evaluation on `ReplCore`.

use crate::{ReplCore, SetDisplayMode, VisualizeCommandOutput};

/// Evaluate `equiv ...` against the active REPL simplifier.
pub fn evaluate_equiv_invocation_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> Result<String, String> {
    crate::evaluate_equiv_invocation_message(core.simplifier_mut(), line)
}

/// Evaluate `subst ...` against the active REPL simplifier.
pub fn evaluate_substitute_invocation_user_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
    display_mode: SetDisplayMode,
) -> Result<String, String> {
    crate::evaluate_substitute_invocation_user_message(core.simplifier_mut(), line, display_mode)
}

/// Evaluate unary command invocation (`det`, `transpose`, `trace`) against REPL simplifier.
pub(crate) fn evaluate_unary_command_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
    function_name: &str,
    display_mode: SetDisplayMode,
    show_parsed: bool,
    clean_result: bool,
) -> Result<String, String> {
    crate::evaluate_unary_command_message(
        core.simplifier_mut(),
        line,
        function_name,
        display_mode,
        show_parsed,
        clean_result,
    )
}

/// Evaluate `det ...` invocation against REPL simplifier.
pub fn evaluate_det_command_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
    display_mode: SetDisplayMode,
) -> Result<String, String> {
    evaluate_unary_command_message_on_repl_core(core, line, "det", display_mode, true, true)
}

/// Evaluate `transpose ...` invocation against REPL simplifier.
pub fn evaluate_transpose_command_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
    display_mode: SetDisplayMode,
) -> Result<String, String> {
    evaluate_unary_command_message_on_repl_core(core, line, "transpose", display_mode, false, false)
}

/// Evaluate `trace ...` invocation against REPL simplifier.
pub fn evaluate_trace_command_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
    display_mode: SetDisplayMode,
) -> Result<String, String> {
    evaluate_unary_command_message_on_repl_core(core, line, "trace", display_mode, false, true)
}

/// Evaluate `weierstrass ...` invocation against REPL simplifier.
pub fn evaluate_weierstrass_invocation_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> Result<String, String> {
    crate::evaluate_weierstrass_invocation_message(core.simplifier_mut(), line)
}

/// Evaluate `telescope ...` using REPL context.
pub fn evaluate_telescope_invocation_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> Result<String, String> {
    crate::evaluate_telescope_invocation_message(&mut core.simplifier_mut().context, line)
}

/// Evaluate `expand_log ...` using REPL context.
pub fn evaluate_expand_log_invocation_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> Result<String, String> {
    crate::evaluate_expand_log_invocation_message(&mut core.simplifier_mut().context, line)
}

/// Evaluate `solve_system ...` using REPL context.
pub fn evaluate_linear_system_command_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> String {
    crate::evaluate_linear_system_command_message(&mut core.simplifier_mut().context, line)
}

/// Evaluate `visualize ...` using REPL context.
pub fn evaluate_visualize_invocation_output_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> Result<VisualizeCommandOutput, String> {
    crate::evaluate_visualize_invocation_output(&mut core.simplifier_mut().context, line)
}

/// Evaluate `solve ...` invocation against REPL core engine/session state.
pub fn evaluate_solve_command_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
    display_mode: SetDisplayMode,
) -> Result<String, String> {
    let debug_mode = core.debug_mode();
    core.with_engine_and_state(|engine, state| {
        crate::evaluate_solve_command_message(engine, state, line, display_mode, debug_mode)
    })
}

/// Evaluate `simplify ...` invocation against REPL core simplifier/session state.
pub fn evaluate_full_simplify_command_lines_on_repl_core(
    core: &mut ReplCore,
    line: &str,
    display_mode: SetDisplayMode,
) -> Result<Vec<String>, String> {
    core.with_state_and_simplifier_mut(|state, simplifier| {
        crate::evaluate_full_simplify_command_lines(simplifier, state, line, display_mode)
    })
}

/// Evaluate `rationalize ...` invocation against REPL core simplifier.
pub fn evaluate_rationalize_command_lines_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> Result<Vec<String>, String> {
    crate::evaluate_rationalize_command_lines(core.simplifier_mut(), line)
}

/// Refresh last health report using current REPL simplifier and health flag.
pub fn update_health_report_on_repl_core(core: &mut ReplCore) {
    core.set_last_health_report(crate::capture_health_report_if_enabled(
        core.simplifier(),
        core.health_enabled(),
    ));
}

/// Evaluate `explain ...` using REPL context.
pub fn evaluate_explain_invocation_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> Result<String, String> {
    crate::evaluate_explain_invocation_message(&mut core.simplifier_mut().context, line)
}

/// Render `vars` command output using REPL core state/context.
pub fn evaluate_vars_command_message_on_repl_core(core: &ReplCore) -> String {
    crate::evaluate_vars_command_lines_with_context(core.state(), &core.simplifier().context)
        .join("\n")
}

/// Render `history` command output using REPL core state/context.
pub fn evaluate_history_command_message_on_repl_core(core: &ReplCore) -> String {
    crate::evaluate_history_command_lines_with_context(core.state(), &core.simplifier().context)
        .join("\n")
}

/// Evaluate `show` command lines against REPL core state/engine.
pub fn evaluate_show_command_lines_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> Result<Vec<String>, String> {
    core.with_engine_and_state(|engine, state| {
        crate::evaluate_show_command_lines(state, engine, line)
    })
}

/// Evaluate `clear` command lines against REPL core state.
pub fn evaluate_clear_command_lines_on_repl_core(core: &mut ReplCore, line: &str) -> Vec<String> {
    crate::evaluate_clear_command_lines(core.state_mut(), line)
}

/// Evaluate `del` command message against REPL core state.
pub fn evaluate_delete_history_command_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> String {
    crate::evaluate_delete_history_command_message(core.state_mut(), line)
}

/// Evaluate profile `cache` command lines against REPL core engine.
pub fn evaluate_profile_cache_command_lines_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> Vec<String> {
    crate::evaluate_profile_cache_command_lines(core.engine_mut(), line)
}

/// Evaluate `budget ...` command message against REPL core session state.
pub fn evaluate_solve_budget_command_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> String {
    crate::evaluate_solve_budget_command_message(core.state_mut(), line)
}

/// Evaluate `let ...` command against REPL core and return user-facing message.
pub fn evaluate_let_assignment_command_message_on_repl_core(
    core: &mut ReplCore,
    input: &str,
) -> Result<String, String> {
    core.with_state_and_simplifier_mut(|state, simplifier| {
        crate::evaluate_let_assignment_command_message_with_simplifier(state, simplifier, input)
    })
}

/// Evaluate assignment command against REPL core and return user-facing message.
pub fn evaluate_assignment_command_message_on_repl_core(
    core: &mut ReplCore,
    name: &str,
    expr_str: &str,
    lazy: bool,
) -> Result<String, String> {
    core.with_state_and_simplifier_mut(|state, simplifier| {
        crate::evaluate_assignment_command_message_with_simplifier(
            state, simplifier, name, expr_str, lazy,
        )
    })
}

/// Evaluate `health ...` command against REPL core and apply returned side-effects.
pub fn evaluate_health_command_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
) -> Result<String, String> {
    let last_stats = core.last_stats().cloned();
    let last_health_report = core.last_health_report().map(str::to_string);
    let out = crate::evaluate_health_command(
        core.simplifier_mut(),
        line,
        last_stats.as_ref(),
        last_health_report.as_deref(),
    )?;

    if let Some(enabled) = out.set_enabled {
        core.set_health_enabled(enabled);
    }
    if out.clear_last_report {
        core.clear_last_health_report();
    }

    Ok(out.lines.join("\n"))
}

/// Evaluate REPL expression and return a frontend-agnostic render plan.
pub fn evaluate_eval_command_render_plan_on_repl_core(
    core: &mut ReplCore,
    line: &str,
    verbosity_is_none: bool,
) -> Result<crate::EvalCommandRenderPlan, String> {
    let debug_mode = core.debug_mode();
    let out = core
        .with_engine_and_state(|engine, state| {
            crate::evaluate_eval_command_output(engine, state, line, debug_mode)
        })
        .map_err(|error| match error {
            crate::EvalCommandError::Parse(parse_error) => {
                crate::render_parse_error(line, &parse_error)
            }
            crate::EvalCommandError::Eval(message) => message,
        })?;
    Ok(crate::build_eval_command_render_plan(
        out,
        verbosity_is_none,
    ))
}

/// Evaluate `expand ...` and return a frontend-agnostic render plan.
pub fn evaluate_expand_command_render_plan_on_repl_core(
    core: &mut ReplCore,
    line: &str,
    verbosity_is_none: bool,
) -> Result<crate::EvalCommandRenderPlan, String> {
    let wrapped = crate::evaluate_expand_wrapped_expression(line)?;
    evaluate_eval_command_render_plan_on_repl_core(core, &wrapped, verbosity_is_none)
}

/// Return profile cache size for the current REPL core engine.
pub fn profile_cache_len_on_repl_core(core: &ReplCore) -> usize {
    core.profile_cache_len()
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate_assignment_command_message_on_repl_core,
        evaluate_clear_command_lines_on_repl_core,
        evaluate_delete_history_command_message_on_repl_core,
        evaluate_det_command_message_on_repl_core, evaluate_equiv_invocation_message_on_repl_core,
        evaluate_eval_command_render_plan_on_repl_core,
        evaluate_expand_command_render_plan_on_repl_core,
        evaluate_explain_invocation_message_on_repl_core,
        evaluate_health_command_message_on_repl_core,
        evaluate_history_command_message_on_repl_core,
        evaluate_let_assignment_command_message_on_repl_core,
        evaluate_linear_system_command_message_on_repl_core,
        evaluate_profile_cache_command_lines_on_repl_core,
        evaluate_rationalize_command_lines_on_repl_core, evaluate_show_command_lines_on_repl_core,
        evaluate_solve_budget_command_message_on_repl_core,
        evaluate_solve_command_message_on_repl_core,
        evaluate_telescope_invocation_message_on_repl_core,
        evaluate_trace_command_message_on_repl_core,
        evaluate_transpose_command_message_on_repl_core,
        evaluate_unary_command_message_on_repl_core, evaluate_vars_command_message_on_repl_core,
        evaluate_weierstrass_invocation_message_on_repl_core, profile_cache_len_on_repl_core,
    };

    #[test]
    fn evaluate_equiv_invocation_message_on_repl_core_true() {
        let mut core = crate::ReplCore::new();
        let out = evaluate_equiv_invocation_message_on_repl_core(&mut core, "equiv x+1,1+x")
            .expect("equiv should evaluate");
        assert!(out.contains("True"));
    }

    #[test]
    fn evaluate_telescope_invocation_message_on_repl_core_requires_input() {
        let mut core = crate::ReplCore::new();
        let err = evaluate_telescope_invocation_message_on_repl_core(&mut core, "telescope")
            .expect_err("usage expected");
        assert!(err.contains("Usage: telescope"));
    }

    #[test]
    fn evaluate_linear_system_command_message_on_repl_core_solves_2x2() {
        let mut core = crate::ReplCore::new();
        let shown = evaluate_linear_system_command_message_on_repl_core(
            &mut core,
            "solve_system(x+y=3; x-y=1; x; y)",
        );
        assert_eq!(shown, "{ x = 2, y = 1 }");
    }

    #[test]
    fn evaluate_explain_invocation_message_on_repl_core_contains_result() {
        let mut core = crate::ReplCore::new();
        let out = evaluate_explain_invocation_message_on_repl_core(&mut core, "explain gcd(8,6)")
            .expect("explain should evaluate");
        assert!(out.contains("Result:"));
    }

    #[test]
    fn evaluate_vars_and_history_command_messages_on_repl_core_render() {
        let core = crate::ReplCore::new();
        let vars = evaluate_vars_command_message_on_repl_core(&core);
        let history = evaluate_history_command_message_on_repl_core(&core);
        assert!(!vars.trim().is_empty());
        assert!(!history.trim().is_empty());
    }

    #[test]
    fn evaluate_show_command_lines_on_repl_core_reports_invalid_id() {
        let mut core = crate::ReplCore::new();
        let out = evaluate_show_command_lines_on_repl_core(&mut core, "show nope");
        assert!(out.is_err());
    }

    #[test]
    fn evaluate_clear_and_delete_on_repl_core_return_messages() {
        let mut core = crate::ReplCore::new();
        let lines = evaluate_clear_command_lines_on_repl_core(&mut core, "clear");
        assert!(!lines.is_empty());

        let message = evaluate_delete_history_command_message_on_repl_core(&mut core, "del #1");
        assert!(!message.trim().is_empty());
    }

    #[test]
    fn evaluate_profile_cache_command_lines_on_repl_core_reports_status() {
        let mut core = crate::ReplCore::new();
        let lines = evaluate_profile_cache_command_lines_on_repl_core(&mut core, "cache status");
        assert!(lines.iter().any(|line| line.contains("Profile Cache:")));
    }

    #[test]
    fn evaluate_profile_cache_command_lines_on_repl_core_reports_populated_cache() {
        let mut core = crate::ReplCore::new();
        let _ = evaluate_eval_command_render_plan_on_repl_core(&mut core, "x+x", false)
            .expect("eval should populate cache");

        let lines = evaluate_profile_cache_command_lines_on_repl_core(&mut core, "cache status");
        assert!(lines
            .iter()
            .any(|line| line.contains("Profile Cache: 1 profiles cached")));
    }

    #[test]
    fn evaluate_profile_cache_command_lines_on_repl_core_clear_empties_cache() {
        let mut core = crate::ReplCore::new();
        let _ = evaluate_eval_command_render_plan_on_repl_core(&mut core, "x+x", false)
            .expect("eval should populate cache");
        assert_eq!(profile_cache_len_on_repl_core(&core), 1);

        let lines = evaluate_profile_cache_command_lines_on_repl_core(&mut core, "cache clear");
        assert!(lines
            .iter()
            .any(|line| line.contains("Profile cache cleared")));
        assert_eq!(profile_cache_len_on_repl_core(&core), 0);
    }

    #[test]
    fn evaluate_unary_command_message_on_repl_core_runs_det() {
        let mut core = crate::ReplCore::new();
        let out = evaluate_unary_command_message_on_repl_core(
            &mut core,
            "det([[1,2],[3,4]])",
            "det",
            crate::SetDisplayMode::Normal,
            true,
            true,
        )
        .expect("det should evaluate");
        assert!(out.contains("Result:"));
    }

    #[test]
    fn evaluate_det_transpose_trace_wrappers_on_repl_core_run() {
        let mut core = crate::ReplCore::new();

        let det = evaluate_det_command_message_on_repl_core(
            &mut core,
            "det([[1,2],[3,4]])",
            crate::SetDisplayMode::Normal,
        )
        .expect("det should evaluate");
        assert!(det.contains("Result:"));

        let transpose = evaluate_transpose_command_message_on_repl_core(
            &mut core,
            "transpose([[1,2],[3,4]])",
            crate::SetDisplayMode::Normal,
        )
        .expect("transpose should evaluate");
        assert!(transpose.contains("Result:"));

        let trace = evaluate_trace_command_message_on_repl_core(
            &mut core,
            "trace([[1,2],[3,4]])",
            crate::SetDisplayMode::Normal,
        )
        .expect("trace should evaluate");
        assert!(trace.contains("Result:"));
    }

    #[test]
    fn evaluate_weierstrass_and_rationalize_on_repl_core_run() {
        let mut core = crate::ReplCore::new();
        let wei =
            evaluate_weierstrass_invocation_message_on_repl_core(&mut core, "weierstrass sin(x)")
                .expect("weierstrass");
        assert!(wei.contains("Result:"));

        let rat =
            evaluate_rationalize_command_lines_on_repl_core(&mut core, "rationalize 1/(1+sqrt(2))")
                .expect("rationalize");
        assert!(!rat.is_empty());
    }

    #[test]
    fn evaluate_solve_command_message_on_repl_core_runs() {
        let mut core = crate::ReplCore::new();
        let out = evaluate_solve_command_message_on_repl_core(
            &mut core,
            "solve x+2=5, x",
            crate::SetDisplayMode::Normal,
        )
        .expect("solve");
        assert!(!out.trim().is_empty());
        assert!(out.contains('x'));
    }

    #[test]
    fn evaluate_budget_let_assignment_and_health_on_repl_core_run() {
        let mut core = crate::ReplCore::new();
        let budget = evaluate_solve_budget_command_message_on_repl_core(&mut core, "budget");
        assert!(!budget.trim().is_empty());

        let let_msg =
            evaluate_let_assignment_command_message_on_repl_core(&mut core, "x = 2").expect("let");
        assert!(!let_msg.trim().is_empty());
        assert!(let_msg.contains('x'));

        let assign_msg =
            evaluate_assignment_command_message_on_repl_core(&mut core, "y", "x+1", true)
                .expect("assign");
        assert!(!assign_msg.trim().is_empty());
        assert!(assign_msg.contains('y'));

        let health = evaluate_health_command_message_on_repl_core(&mut core, "health");
        assert!(health.is_ok());
    }

    #[test]
    fn evaluate_eval_command_render_plan_on_repl_core_returns_plan() {
        let mut core = crate::ReplCore::new();
        let plan = evaluate_eval_command_render_plan_on_repl_core(&mut core, "x+1", false)
            .expect("eval plan");
        assert!(plan.result_message.is_some());
        assert!(profile_cache_len_on_repl_core(&core) >= 1);
    }

    #[test]
    fn evaluate_expand_command_render_plan_on_repl_core_returns_plan() {
        let mut core = crate::ReplCore::new();
        let plan =
            evaluate_expand_command_render_plan_on_repl_core(&mut core, "expand (x+1)^2", false)
                .expect("expand plan");
        assert!(plan.result_message.is_some());
    }
}
