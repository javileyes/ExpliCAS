//! End-to-end contract tests for fast modular exponentiation, `mod(base^exp, m)`
//! (2026-07-14, F6 of docs/AUDITORIA_FRONTERA_2026-07-14.md). The simplifier is
//! strictly bottom-up, so the child `base^exp` fully materializes before the
//! `mod` handler runs — for competition-scale exponents that intermediate has
//! tens of millions of digits and construction hangs for minutes. A raw-tree
//! guard in `transform_function` intercepts the exact `mod(int^nonneg, nonzero)`
//! shape and evaluates by square-and-multiply, yielding the identical residue
//! without ever building the power. These tests exercise the full engine path
//! (not just the arithmetic helper) so a regression that stops the guard from
//! firing — and lets the power materialize — would time out here.

use cas_parser::parse;
use cas_solver::runtime::{Engine, EvalAction, EvalOptions, EvalRequest, EvalResult};

fn eval_number(input: &str) -> String {
    let mut engine = Engine::new();
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");
    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };
    let output = engine
        .eval_stateless(EvalOptions::default(), req)
        .expect("eval");
    let id = match output.result {
        EvalResult::Expr(id) => id,
        other => panic!("expected Expr, got {other:?}"),
    };
    format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &engine.simplifier.context,
            id
        }
    )
}

#[test]
fn fast_modpow_evaluates_competition_scale_inputs_without_hanging() {
    // Each of these materializes a power with tens of millions of digits under
    // the naive path; the guard must return the residue essentially instantly.
    assert_eq!(
        eval_number("mod(123456789^987654321, 1000000007)"),
        "652541198"
    );
    assert_eq!(eval_number("mod(6^50000000, 100)"), "76");
    assert_eq!(eval_number("mod(10^100000000, 7)"), "4");
}

#[test]
fn fast_modpow_matches_ordinary_reduction_on_small_inputs() {
    assert_eq!(eval_number("mod(7^13, 100)"), "7");
    assert_eq!(eval_number("mod(2^10, 1000)"), "24");
    assert_eq!(eval_number("mod(3^4, 5)"), "1");
    // Negative base normalizes to the non-negative residue: (-3)^3 = -27 ≡ 1 (mod 7).
    assert_eq!(eval_number("mod((-3)^3, 7)"), "1");
    // exp = 0 -> 1; modulus 1 -> 0.
    assert_eq!(eval_number("mod(5^0, 7)"), "1");
    assert_eq!(eval_number("mod(10^2, 1)"), "0");
}

#[test]
fn non_power_and_fractional_mod_are_unchanged() {
    // Plain integer mod: unaffected by the power guard.
    assert_eq!(eval_number("mod(17, 5)"), "2");
    assert_eq!(eval_number("mod(-7, 5)"), "3");
    // Negative exponent is a fraction, not a residue: the guard declines and the
    // ordinary path keeps the symbolic `mod(1/4, 5)`.
    assert_eq!(eval_number("mod(2^(-2), 5)"), "mod(1/4, 5)");
}
