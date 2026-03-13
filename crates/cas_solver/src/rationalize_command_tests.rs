use crate::rationalize_command::{RationalizeCommandEvalError, RationalizeCommandOutcome};
use crate::rationalize_command_eval::evaluate_rationalize_command_input;

#[test]
fn evaluate_rationalize_command_lines_empty_input_returns_usage() {
    let mut simplifier = crate::Simplifier::new();
    let err = crate::evaluate_rationalize_command_lines(&mut simplifier, "rationalize")
        .expect_err("expected usage");
    assert!(err.contains("Usage: rationalize"));
}

#[test]
fn evaluate_rationalize_command_input_parse_error_is_typed() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let err =
        evaluate_rationalize_command_input(&mut simplifier, "1/(1+").expect_err("parse error");
    assert!(matches!(err, RationalizeCommandEvalError::Parse(_)));
}

#[test]
fn evaluate_rationalize_command_input_runs() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let out = evaluate_rationalize_command_input(&mut simplifier, "1/(1+sqrt(2))").expect("eval");
    match out.outcome {
        RationalizeCommandOutcome::Success(_)
        | RationalizeCommandOutcome::NotApplicable
        | RationalizeCommandOutcome::BudgetExceeded => {}
    }
}
