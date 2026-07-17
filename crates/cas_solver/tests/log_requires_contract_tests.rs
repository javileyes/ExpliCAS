use cas_api_models::{
    EvalAssumeScope, EvalBranchMode, EvalBudgetPreset, EvalComplexMode, EvalConstFoldMode,
    EvalContextMode, EvalDomainMode, EvalExpandPolicy, EvalInvTrigPolicy, EvalStepsMode,
    EvalValueDomain,
};
use cas_session::eval::{evaluate_eval_command_pretty_with_session, EvalCommandConfig};

#[test]
fn log_power_requires_drop_redundant_power_guards() {
    let json = evaluate_eval_command_pretty_with_session(
        None,
        EvalCommandConfig {
            expr: "ln(x^n) + ln(y^n)",
            auto_store: false,
            max_chars: 4000,
            time_budget_ms: None,
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
            numeric_display: cas_api_models::EvalNumericDisplay::Exact,
        },
        cas_solver_core::eval_option_axes::Language::Es,
        |_steps, _events, _context, _steps_mode| Vec::new(),
    );
    let payload: serde_json::Value = serde_json::from_str(&json).expect("json");
    let required_display = payload["required_display"]
        .as_array()
        .expect("required_display array");
    let required_conditions = payload["required_conditions"]
        .as_array()
        .expect("required_conditions array");

    assert_eq!(
        required_display
            .iter()
            .filter_map(|v| v.as_str())
            .collect::<Vec<_>>(),
        vec!["x > 0", "y > 0"]
    );
    assert_eq!(
        required_conditions.len(),
        2,
        "expected only the base positivity guards to survive normalization"
    );
    assert_eq!(
        required_conditions
            .iter()
            .filter_map(|v| v["expr_display"].as_str())
            .collect::<Vec<_>>(),
        vec!["x", "y"]
    );
}
