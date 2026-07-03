//! Contracts for universality cycle U4: PERIODIC trig functions no longer
//! invert through their principal branch when the argument is NOT
//! rational-affine in the target variable.
//!
//! The unary-inverse route (`sin → arcsin`) asserted a FINITE set while the
//! true preimage is an infinite family the SolutionSet cannot represent:
//! `cos(π·x²) = 1` returned `{ 0 }` (losing `±√(2k)`), `sin(π·sin(x)) = 1`
//! returned `{ π/6 }` (losing `5π/6 + 2kπ`) — adjacents found by the
//! post-PIU adversarial audit. Non-affine arguments now decline to the
//! honest operator-preserving residual; the rational-affine principal
//! convention for a SYMBOLIC RHS (`sin(x) = c → { arcsin(c) }`) and the
//! globally-invertible inverses (ln/sqrt/exp) are untouched.

use assert_cmd::cargo;
use assert_cmd::Command;
use serde_json::Value;

fn solve(input: &str) -> String {
    let out = Command::new(cargo::cargo_bin!("cas_cli"))
        .args(["eval", &format!("solve({input}, x)"), "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
    wire["result"].as_str().unwrap_or("").to_string()
}

#[test]
fn nonaffine_arguments_decline_instead_of_asserting_finite_sets() {
    assert_eq!(solve("cos(pi*x^2)=1"), "solve(cos(pi·x^2) = 1, x)");
    assert_eq!(solve("sin(pi*sin(x))=1"), "solve(sin(pi·sin(x)) = 1, x)");
    assert_eq!(solve("sin(x^3)=1"), "solve(sin(x^3) = 1, x)");
    assert_eq!(solve("sin(ln(x))=1/2"), "solve(sin(ln(x)) = 1 / 2, x)");
    assert_eq!(solve("cos(x^2)=1/2"), "solve(cos(x^2) = 1 / 2, x)");
}

#[test]
fn rational_affine_principal_convention_is_preserved() {
    // Symbolic RHS keeps the textbook principal answer.
    assert_eq!(solve("sin(x)=c"), "{ arcsin(c) }");
    assert_eq!(solve("sin(2*x)=c"), "{ 1/2·arcsin(c) }");
    assert_eq!(solve("cos(x-1)=c"), "{ arccos(c) + 1 }");
    // Concrete RHS keeps the full periodic family (periodic handler).
    assert_eq!(
        solve("sin(x)=1/2"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(solve("sin(pi*x)=1"), "{ 1/2 + k·2 : k ∈ ℤ }");
}

#[test]
fn globally_invertible_inverses_still_handle_nonaffine_args() {
    assert_eq!(solve("ln(x^2)=4"), "{ e^2, -(e^2) }");
    assert_eq!(solve("e^(x^2)=2"), "{ -(ln(2)^(1/2)), ln(2)^(1/2) }");
}
