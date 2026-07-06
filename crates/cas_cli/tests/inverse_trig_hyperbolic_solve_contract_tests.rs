//! Contracts for solving inverse-trig / hyperbolic equations (auto-improvement
//! cycle): `arcsin/arccos/arctan/sinh/cosh/tanh(g) = c`.
//!
//! The isolation dispatch has no inverse for these and used to ERROR
//! ("función no definida"). A dedicated handler applies the (single, monotone)
//! forward inverse — `arcsin(x)=c → x=sin(c)`, `sinh(x)=c → x=asinh(c)`,
//! `cosh(x)=c → x=±acosh(c)` — GATED by the exact const-decision layer: a
//! threshold provably outside the range is `No solution`, provably inside
//! reduces and recurses. (solve's root verification does NOT catch these
//! transcendental range violations, so the gate is mandatory.)

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
fn in_range_thresholds_apply_the_forward_inverse() {
    assert_eq!(solve("arcsin(x) = 1/2"), "{ sin(1/2) }");
    assert_eq!(solve("arccos(x) = 1"), "{ cos(1) }");
    assert_eq!(solve("arctan(x) = 1"), "{ tan(1) }");
    assert_eq!(solve("sinh(x) = 1"), "{ asinh(1) }");
    assert_eq!(solve("tanh(x) = 1/2"), "{ atanh(1/2) }");
    // arctan(x) = π/3 folds to the exact tan(π/3) = sqrt(3).
    assert_eq!(solve("arctan(x) = pi/3"), "{ sqrt(3) }");
}

#[test]
fn cosh_yields_both_branches_and_dedups_at_one() {
    assert_eq!(solve("cosh(x) = 2"), "{ acosh(2), -acosh(2) }");
    // acosh(1) = 0, so the two branches coincide — no duplicate root.
    assert_eq!(solve("cosh(x) = 1"), "{ 0 }");
    assert_eq!(solve("cosh(2*x) = 2"), "{ 1/2·acosh(2), -1/2·acosh(2) }");
}

#[test]
fn out_of_range_thresholds_have_no_real_solution() {
    // The const-decision gate is exact even at fine borderlines vs the
    // transcendental bounds π/2 ≈ 1.5708, π ≈ 3.1416.
    assert_eq!(solve("arcsin(x) = 5"), "No solution");
    assert_eq!(solve("arcsin(x) = 8/5"), "No solution"); // 1.6 > π/2
    assert_eq!(solve("arctan(x) = 158/100"), "No solution"); // 1.58 > π/2
    assert_eq!(solve("arccos(x) = 16/5"), "No solution"); // 3.2 > π
    assert_eq!(solve("arccos(x) = -1/10"), "No solution"); // < 0
    assert_eq!(solve("tanh(x) = 2"), "No solution");
    assert_eq!(solve("tanh(x) = 1"), "No solution"); // open range excludes ±1
    assert_eq!(solve("cosh(x) = 1/2"), "No solution"); // cosh ≥ 1
    assert_eq!(solve("arctan(x) = pi/2"), "No solution"); // tan undefined there
}

#[test]
fn just_inside_the_range_still_solves() {
    assert_eq!(solve("arcsin(x) = 3/2"), "{ sin(3/2) }"); // 1.5 < π/2
    assert_eq!(solve("arctan(x) = 157/100"), "{ tan(157/100) }"); // 1.57 < π/2
    assert_eq!(solve("arccos(x) = 3"), "{ cos(3) }"); // 3 < π
    assert_eq!(solve("tanh(x) = 99/100"), "{ atanh(99/100) }");
}

#[test]
fn range_boundaries_and_affine_arguments() {
    assert_eq!(solve("arccos(x) = 0"), "{ 1 }");
    assert_eq!(solve("arccos(x) = pi"), "{ -1 }");
    assert_eq!(solve("arcsin(x) = pi/2"), "{ 1 }");
    assert_eq!(solve("arcsin(2*x) = 1/2"), "{ 1/2·sin(1/2) }");
    assert_eq!(solve("sinh(x-1) = 1"), "{ asinh(1) + 1 }");
}

#[test]
fn sinh_is_unconditional_and_forward_trig_is_untouched() {
    assert_eq!(solve("sinh(x) = -3"), "{ asinh(-3) }");
    // The forward trig / log solvers are unchanged.
    assert_eq!(
        solve("sin(x) = 1/2"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(solve("cos(x) = 1"), "{ k·2·pi : k ∈ ℤ }");
    assert_eq!(solve("ln(x) = 2"), "{ e^2 }");
}
