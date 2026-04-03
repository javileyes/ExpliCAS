use cas_api_models::{
    EvalAssumeScope, EvalBranchMode, EvalBudgetPreset, EvalComplexMode, EvalConstFoldMode,
    EvalContextMode, EvalDomainMode, EvalExpandPolicy, EvalInvTrigPolicy, EvalStepsMode,
    EvalValueDomain,
};
use cas_session::eval::{evaluate_eval_command_pretty_with_session, EvalCommandConfig};

#[test]
fn fractional_symbolic_solve_prep_dedupes_scaled_nonzero_requires() {
    let json = evaluate_eval_command_pretty_with_session(
        None,
        EvalCommandConfig {
            expr: "derive((a/2)*x^2 + b*x + c, (a/2)*(x + b/a)^2 + c - b^2/(2*a))",
            auto_store: false,
            max_chars: 4000,
            steps_mode: EvalStepsMode::On,
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
        },
        |_steps, _events, _context, _steps_mode| Vec::new(),
    );
    let payload: serde_json::Value = serde_json::from_str(&json).expect("json");
    let required_display = payload["required_display"]
        .as_array()
        .expect("required_display array");
    let required_conditions = payload["required_conditions"]
        .as_array()
        .expect("required_conditions array");

    assert_eq!(required_display.len(), 1, "expected one displayed require");
    assert_eq!(
        required_display.first().and_then(|v| v.as_str()),
        Some("a ≠ 0")
    );
    assert_eq!(
        required_conditions.len(),
        1,
        "expected one normalized required condition"
    );
    assert_eq!(
        required_conditions.first().and_then(|v| v["kind"].as_str()),
        Some("NonZero")
    );
    assert_eq!(
        required_conditions
            .first()
            .and_then(|v| v["expr_display"].as_str()),
        Some("a")
    );
}
