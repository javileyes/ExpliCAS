use cas_api_models::{
    EvalAssumeScope, EvalBranchMode, EvalBudgetPreset, EvalComplexMode, EvalConstFoldMode,
    EvalContextMode, EvalDomainMode, EvalExpandPolicy, EvalInvTrigPolicy, EvalStepsMode,
    EvalValueDomain,
};
use cas_ast::ordering::compare_expr;
use cas_formatter::DisplayExpr;
use cas_parser::parse;
use cas_session::eval::{evaluate_eval_command_pretty_with_session, EvalCommandConfig};
use cas_session::SessionState;
use cas_solver::api::ImplicitCondition;
use cas_solver::runtime::{Engine, EvalAction, EvalRequest, EvalResult};

#[test]
fn consecutive_factorial_ratio_simplifies_and_includes_factorial_domain_requirement() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let input = "(n + 1)! / n!";
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result = match output.result {
        EvalResult::Expr(expr) => DisplayExpr {
            context: &engine.simplifier.context,
            id: expr,
        }
        .to_string(),
        other => panic!("expected expression result, got {other:?}"),
    };

    assert_eq!(result, "n + 1");

    let expected_arg = parse("n", &mut engine.simplifier.context).expect("arg");
    assert!(output.required_conditions.iter().any(|cond| {
        matches!(
            cond,
            ImplicitCondition::NonNegative(arg)
                if compare_expr(&engine.simplifier.context, *arg, expected_arg)
                    == std::cmp::Ordering::Equal
        )
    }));
}

#[test]
fn consecutive_factorial_ratio_wire_uses_factorial_latex_and_nonnegative_require() {
    let json = evaluate_eval_command_pretty_with_session(
        None,
        EvalCommandConfig {
            expr: "(n + 1)! / n!",
            auto_store: false,
            max_chars: 2000,
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
        payload["input_latex"].as_str(),
        Some("\\frac{(n + 1)!}{n!}")
    );
    assert_eq!(
        required_display.len(),
        1,
        "expected exactly one displayed require"
    );
    assert_eq!(
        required_display.first().and_then(|v| v.as_str()),
        Some("n ≥ 0")
    );
    assert_eq!(
        required_conditions.len(),
        1,
        "expected exactly one normalized required condition"
    );
    assert_eq!(
        required_conditions.first().and_then(|v| v["kind"].as_str()),
        Some("NonNegative")
    );
}
