#[allow(unused_imports)]
use cas_solver::session_api::{
    formatting::*, options::*, runtime::*, session_support::*, symbolic_commands::*, types::*,
};

#[test]
fn evaluate_equiv_invocation_message_on_repl_core_true() {
    let mut core = crate::repl_core::ReplCore::new();
    let out = evaluate_equiv_invocation_message_on_repl_core(&mut core, "equiv x+1,1+x")
        .expect("equiv should evaluate");
    assert!(out.contains("True"));
}

#[test]
fn evaluate_telescope_invocation_message_on_repl_core_requires_input() {
    let mut core = crate::repl_core::ReplCore::new();
    let err = evaluate_telescope_invocation_message_on_repl_core(&mut core, "telescope")
        .expect_err("usage expected");
    assert!(err.contains("Usage: telescope"));
}

#[test]
fn evaluate_linear_system_command_message_on_repl_core_solves_2x2() {
    let mut core = crate::repl_core::ReplCore::new();
    let shown = evaluate_linear_system_command_message_on_repl_core(
        &mut core,
        "solve_system(x+y=3; x-y=1; x; y)",
    );
    assert_eq!(shown, "{ x = 2, y = 1 }");
}

#[test]
fn evaluate_explain_invocation_message_on_repl_core_contains_result() {
    let mut core = crate::repl_core::ReplCore::new();
    let out = evaluate_explain_invocation_message_on_repl_core(&mut core, "explain gcd(8,6)")
        .expect("explain should evaluate");
    assert!(out.contains("Result:"));
}

#[test]
fn evaluate_vars_and_history_command_messages_on_repl_core_render() {
    let core = crate::repl_core::ReplCore::new();
    let vars = evaluate_vars_command_message_on_repl_core(&core);
    let history = evaluate_history_command_message_on_repl_core(&core);
    assert!(!vars.trim().is_empty());
    assert!(!history.trim().is_empty());
}

#[test]
fn evaluate_show_command_lines_on_repl_core_reports_invalid_id() {
    let mut core = crate::repl_core::ReplCore::new();
    let out = evaluate_show_command_lines_on_repl_core(&mut core, "show nope");
    assert!(out.is_err());
}

#[test]
fn evaluate_clear_and_delete_on_repl_core_return_messages() {
    let mut core = crate::repl_core::ReplCore::new();
    let lines = evaluate_clear_command_lines_on_repl_core(&mut core, "clear");
    assert!(!lines.is_empty());

    let message = evaluate_delete_history_command_message_on_repl_core(&mut core, "del #1");
    assert!(!message.trim().is_empty());
}

#[test]
fn evaluate_profile_cache_command_lines_on_repl_core_reports_status() {
    let mut core = crate::repl_core::ReplCore::new();
    let lines = evaluate_profile_cache_command_lines_on_repl_core(&mut core, "cache status");
    assert!(lines.iter().any(|line| line.contains("Profile Cache:")));
}

#[test]
fn evaluate_profile_cache_command_lines_on_repl_core_reports_populated_cache() {
    let mut core = crate::repl_core::ReplCore::new();
    let _ = evaluate_eval_command_render_plan_on_repl_core(&mut core, "x+x", false)
        .expect("eval should populate cache");

    let lines = evaluate_profile_cache_command_lines_on_repl_core(&mut core, "cache status");
    assert!(lines
        .iter()
        .any(|line| line.contains("Profile Cache: 1 profiles cached")));
}

#[test]
fn evaluate_profile_cache_command_lines_on_repl_core_clear_empties_cache() {
    let mut core = crate::repl_core::ReplCore::new();
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
    let mut core = crate::repl_core::ReplCore::new();
    let out = cas_solver::session_api::runtime::evaluate_unary_command_message_on_runtime(
        &mut core,
        "det([[1,2],[3,4]])",
        "det",
        SetDisplayMode::Normal,
        true,
        true,
    )
    .expect("det should evaluate");
    assert!(out.contains("Result:"));
}

#[test]
fn evaluate_det_transpose_trace_wrappers_on_repl_core_run() {
    let mut core = crate::repl_core::ReplCore::new();

    let det = evaluate_det_command_message_on_repl_core(
        &mut core,
        "det([[1,2],[3,4]])",
        SetDisplayMode::Normal,
    )
    .expect("det should evaluate");
    assert!(det.contains("Result:"));

    let transpose = evaluate_transpose_command_message_on_repl_core(
        &mut core,
        "transpose([[1,2],[3,4]])",
        SetDisplayMode::Normal,
    )
    .expect("transpose should evaluate");
    assert!(transpose.contains("Result:"));

    let trace = evaluate_trace_command_message_on_repl_core(
        &mut core,
        "trace([[1,2],[3,4]])",
        SetDisplayMode::Normal,
    )
    .expect("trace should evaluate");
    assert!(trace.contains("Result:"));
}

#[test]
fn evaluate_weierstrass_and_rationalize_on_repl_core_run() {
    let mut core = crate::repl_core::ReplCore::new();
    let wei = evaluate_weierstrass_invocation_message_on_repl_core(&mut core, "weierstrass sin(x)")
        .expect("weierstrass");
    assert!(wei.contains("Result:"));

    let rat =
        evaluate_rationalize_command_lines_on_repl_core(&mut core, "rationalize 1/(1+sqrt(2))")
            .expect("rationalize");
    assert!(!rat.is_empty());
}

#[test]
fn evaluate_solve_command_message_on_repl_core_runs() {
    let mut core = crate::repl_core::ReplCore::new();
    let out = evaluate_solve_command_message_on_repl_core(
        &mut core,
        "solve x+2=5, x",
        SetDisplayMode::Normal,
    )
    .expect("solve");
    assert!(!out.trim().is_empty());
    assert!(out.contains('x'));
}

#[test]
fn evaluate_solve_command_message_on_repl_core_reports_ambiguous_variables() {
    let mut core = crate::repl_core::ReplCore::new();
    let err = evaluate_solve_command_message_on_repl_core(
        &mut core,
        "solve x+y=0",
        SetDisplayMode::Normal,
    )
    .expect_err("expected ambiguous-variable error");
    assert!(err.contains("ambiguous variables"));
}

#[test]
fn evaluate_full_simplify_command_lines_on_repl_core_runs() {
    let mut core = crate::repl_core::ReplCore::new();
    let lines = evaluate_full_simplify_command_lines_on_repl_core(
        &mut core,
        "simplify x + 0",
        SetDisplayMode::Normal,
    )
    .expect("simplify");
    assert!(lines.iter().any(|line| line.starts_with("Result:")));
}

#[test]
fn evaluate_budget_let_assignment_and_health_on_repl_core_run() {
    let mut core = crate::repl_core::ReplCore::new();
    let budget = evaluate_solve_budget_command_message_on_repl_core(&mut core, "budget");
    assert!(!budget.trim().is_empty());

    let let_msg =
        evaluate_let_assignment_command_message_on_repl_core(&mut core, "x = 2").expect("let");
    assert!(!let_msg.trim().is_empty());
    assert!(let_msg.contains('x'));

    let assign_msg = evaluate_assignment_command_message_on_repl_core(&mut core, "y", "x+1", true)
        .expect("assign");
    assert!(!assign_msg.trim().is_empty());
    assert!(assign_msg.contains('y'));

    let health = evaluate_health_command_message_on_repl_core(&mut core, "health");
    assert!(health.is_ok());
}

#[test]
fn evaluate_eval_command_render_plan_on_repl_core_returns_plan() {
    let mut core = crate::repl_core::ReplCore::new();
    let plan =
        evaluate_eval_command_render_plan_on_repl_core(&mut core, "x+1", false).expect("eval plan");
    assert!(plan.result_message.is_some());
    assert!(profile_cache_len_on_repl_core(&core) >= 1);
}

#[test]
fn evaluate_expand_command_render_plan_on_repl_core_returns_plan() {
    let mut core = crate::repl_core::ReplCore::new();
    let plan = evaluate_expand_command_render_plan_on_repl_core(&mut core, "expand (x+1)^2", false)
        .expect("expand plan");
    assert!(plan.result_message.is_some());
}
