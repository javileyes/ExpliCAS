//! Relative session references (`#-1` = newest cell) through the WIRE
//! funnel — the exact entrypoint the wasm browser session and the persisted
//! server sessions share (`evaluate_eval_command_in_memory_with_state`).
//!
//! `#-k` normalizes to an absolute `#N` BEFORE parsing (session_core
//! `rewrite_relative_session_refs`), so the stored raw_text replays
//! identically later; these tests pin the end-to-end behavior: arithmetic
//! through a relative ref, chained relatives, and the honest decline when
//! the session has fewer cells than requested.

use cas_api_models::{
    EvalAssumeScope, EvalBranchMode, EvalBudgetPreset, EvalComplexMode, EvalConstFoldMode,
    EvalContextMode, EvalDomainMode, EvalExpandPolicy, EvalInvTrigPolicy, EvalNumericDisplay,
    EvalStepsMode, EvalValueDomain,
};
use cas_didactic::Language;
use cas_session::eval::{evaluate_eval_command_in_memory_with_state, EvalCommandConfig};

fn config(expr: &str) -> EvalCommandConfig<'_> {
    EvalCommandConfig {
        expr,
        // auto_store ON: cells must accumulate for #-k to point at (this is
        // how the wasm session runs).
        auto_store: true,
        max_chars: 2000,
        time_budget_ms: None,
        steps_mode: EvalStepsMode::Off,
        budget_preset: EvalBudgetPreset::Standard,
        strict: false,
        domain: EvalDomainMode::Generic,
        context_mode: EvalContextMode::Auto,
        branch_mode: EvalBranchMode::Strict,
        expand_policy: EvalExpandPolicy::Off,
        complex_mode: EvalComplexMode::Auto,
        const_fold: EvalConstFoldMode::Off,
        value_domain: EvalValueDomain::Real,
        complex_branch: EvalBranchMode::Principal,
        inv_trig: EvalInvTrigPolicy::Strict,
        assume_scope: EvalAssumeScope::Real,
        numeric_display: EvalNumericDisplay::Exact,
        approx_hint: false,
    }
}

fn eval_on(
    engine: &mut cas_solver::runtime::Engine,
    state: &mut cas_session::SessionState,
    expr: &str,
) -> Result<cas_api_models::EvalWireOutput, String> {
    evaluate_eval_command_in_memory_with_state(
        engine,
        state,
        config(expr),
        Language::Es,
        |steps, events, ctx, mode| {
            cas_didactic::collect_step_payloads_with_events_localized(
                steps,
                events,
                ctx,
                mode,
                Language::Es,
            )
        },
    )
}

#[test]
fn relative_ref_resolves_to_newest_cell_on_the_wire_funnel() {
    let mut engine = cas_solver::runtime::Engine::new();
    let mut state = cas_session::SessionState::new();

    let first = eval_on(&mut engine, &mut state, "2 + 2").expect("first eval");
    assert_eq!(first.result, "4");

    let second = eval_on(&mut engine, &mut state, "#-1 * 3").expect("relative eval");
    assert_eq!(second.result, "12", "#-1 must resolve to the 2+2 cell");

    let third = eval_on(&mut engine, &mut state, "#-2 + #-1").expect("chained relatives");
    assert_eq!(third.result, "16", "#-2 is the 4, #-1 the 12");
}

#[test]
fn relative_ref_out_of_range_declines_honestly() {
    let mut engine = cas_solver::runtime::Engine::new();
    let mut state = cas_session::SessionState::new();

    let err = eval_on(&mut engine, &mut state, "#-1 + 1").expect_err("empty session must err");
    assert!(
        err.contains("no hay celdas"),
        "error must say the session is empty: {err}"
    );

    eval_on(&mut engine, &mut state, "5").expect("store one cell");
    let err = eval_on(&mut engine, &mut state, "#-2").expect_err("only one cell");
    assert!(
        err.contains("solo hay 1"),
        "error must carry the honest count: {err}"
    );
}
