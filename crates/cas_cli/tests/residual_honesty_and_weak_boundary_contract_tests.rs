//! Contracts for scout 2026-07-03 cycle 3: the residual honesty contract and
//! the weak-boundary trig inequality handler.
//!
//! Residual honesty: a declined relation used to be echoed with a MUTATED
//! operator and a dangling `= 0` (`sin(x) > 1/2` → "Solve: solve(sin(x) =
//! 1/2, x) = 0"). Residual construction now threads the original `RelOp`
//! through `mk_residual_solve`, and both renderers (text and LaTeX) emit the
//! self-describing `solve(rel, var)` echo — the same convention as
//! `integrate(...)` residuals — with no `= 0` suffix.
//!
//! Weak boundary: `A·sin/cos(g) ⋚ c` touching the range edge (|c/A| ≥ 1) used
//! to decline or garbage. The handler now resolves boundary contact via the
//! boundary EQUATION through the full solve pipeline (periodic sets), and
//! exterior/vacuous thresholds via Empty/AllReals. Interior thresholds
//! (|c/A| < 1) still decline honestly until `PeriodicIntervalUnion` exists.

use assert_cmd::cargo;
use assert_cmd::Command;
use serde_json::Value;

fn eval_wire(input: &str) -> Value {
    let out = Command::new(cargo::cargo_bin!("cas_cli"))
        .args(["eval", input, "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    serde_json::from_slice(&out.stdout).expect("Invalid wire output")
}

fn solve(input: &str) -> String {
    eval_wire(input)["result"]
        .as_str()
        .unwrap_or("")
        .to_string()
}

// ────────────────── Residual honesty: operator preserved ──────────────────

#[test]
fn residual_preserves_strict_greater() {
    // P2 recontract: sin/cos interior thresholds now SOLVE (PeriodicIntervalUnion);
    // tan still declines until its P3 sibling handler and carries the pin.
    assert_eq!(solve("solve(tan(x)>1/2, x)"), "solve(tan(x) > 1 / 2, x)");
}

#[test]
fn residual_preserves_strict_less() {
    assert_eq!(solve("solve(tan(x)<1/2, x)"), "solve(tan(x) < 1 / 2, x)");
}

#[test]
fn residual_preserves_nonstrict_operators() {
    assert_eq!(solve("solve(tan(x)>=1/2, x)"), "solve(tan(x) >= 1 / 2, x)");
    assert_eq!(solve("solve(tan(x)<=2, x)"), "solve(tan(x) <= 2, x)");
}

#[test]
fn residual_latex_is_self_describing_without_dangling_zero() {
    let wire = eval_wire("solve(tan(x)>1/2, x)");
    let latex = wire["result_latex"].as_str().unwrap_or("");
    assert!(
        latex.contains(">") && !latex.contains("= 0") && !latex.contains("Solve:"),
        "expected self-describing LaTeX echo, got: {latex}"
    );
}

#[test]
fn equation_residual_keeps_equation_form() {
    // Fixpoint isolation declines: the echo is a solve() over an EQUATION,
    // which is what the input was — no operator mutation either way.
    assert_eq!(solve("solve(x*e^x=1, x)"), "solve(x - 1 / e^x = 0, x)");
}

#[test]
fn former_garbage_declines_are_now_honest_echoes() {
    // Scout cycle-3 backlog #1: these emitted mutated/malformed residuals.
    assert_eq!(solve("solve(1/sqrt(x)>2, x)"), "solve(1 / sqrt(x) > 2, x)");
    assert_eq!(
        solve("solve(e^(2*x)-e^x-1<0, x)"),
        "solve(e^(2·x) - e^x - 1 < 0, x)"
    );
}

// ────────────────── Weak boundary: range-edge trig inequalities ──────────────────

#[test]
fn sine_weak_boundary_yields_periodic_set() {
    assert_eq!(solve("solve(sin(x)>=1, x)"), "{ 1/2·pi + k·2·pi : k ∈ ℤ }");
    assert_eq!(
        solve("solve(sin(x)<=-1, x)"),
        "{ -1/2·pi + k·2·pi : k ∈ ℤ }"
    );
}

#[test]
fn cosine_weak_boundary_yields_periodic_set() {
    assert_eq!(solve("solve(cos(x)>=1, x)"), "{ k·2·pi : k ∈ ℤ }");
}

#[test]
fn weak_boundary_handles_coefficient_wrappers() {
    // 2·cos(3x) ≤ −2 ⇔ cos(3x) = −1; −3·sin(2x) ≥ 3 ⇔ sin(2x) = −1.
    assert_eq!(
        solve("solve(2*cos(3*x)<=-2, x)"),
        "{ 1/3·pi + k·2/3·pi : k ∈ ℤ }"
    );
    assert_eq!(
        solve("solve(-3*sin(2*x)>=3, x)"),
        "{ -1/4·pi + k·pi : k ∈ ℤ }"
    );
}

#[test]
fn weak_boundary_handles_affine_shifts() {
    assert_eq!(
        solve("solve(sin(x+pi/3)>=1, x)"),
        "{ 1/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        solve("solve(cos(x-pi/4)<=-1, x)"),
        "{ 5/4·pi + k·2·pi : k ∈ ℤ }"
    );
}

#[test]
fn exterior_thresholds_resolve_to_empty_or_all_reals() {
    assert_eq!(solve("solve(sin(x)>1, x)"), "No solution");
    assert_eq!(solve("solve(sin(x)>=2, x)"), "No solution");
    assert_eq!(solve("solve(cos(x)>1, x)"), "No solution");
    assert_eq!(solve("solve(sin(x)<=2, x)"), "All real numbers");
    assert_eq!(solve("solve(sin(x)>=-1, x)"), "All real numbers");
}

#[test]
fn interior_thresholds_now_solve_or_decline_honestly() {
    // P2 recontract: the sin/cos interior threshold this test was DESIGNED
    // to hand over now solves via PeriodicIntervalUnion; tan stays an honest
    // residual until the P3 sibling handler.
    assert_eq!(
        solve("solve(cos(2*x)>0, x)"),
        "{ (-1/4·pi + k·pi, 1/4·pi + k·pi) : k ∈ ℤ }"
    );
    assert_eq!(solve("solve(tan(x)>=1, x)"), "solve(tan(x) >= 1, x)");
}
