//! Contracts for scout 2026-07-03 families B and C.
//!
//! Family B: mixed-base exponential INEQUALITIES used to collapse to the
//! boundary equation (`2^x > 3^x` → `{0}`, where `>` is false). The two-base
//! handler now divides by the positive `N^x` (operator preserved) and hands
//! `A·(M/N)^x + B ⋚ 0` to the single-exponential path.
//!
//! Family C: `A/|g| ⋚ c` lost the `g = 0` pole (`1/|x| > 2` → `(−1/2, 1/2)`)
//! and the c = 0 branch emitted degenerate `(−∞,−∞)` endpoints. The dedicated
//! handler now solves the two rational relations on `A/g` directly (their
//! path punctures poles) for c > 0 and uses the abs-threshold sign shortcut
//! for c ≤ 0.

use assert_cmd::cargo;
use assert_cmd::Command;
use serde_json::Value;

fn solve(input: &str) -> String {
    let out = Command::new(cargo::cargo_bin!("cas_cli"))
        .args(["eval", input, "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
    wire["result"].as_str().unwrap_or("").to_string()
}

// ───────────────────────── Family B ─────────────────────────

#[test]
fn mixed_base_strict_inequalities_return_rays() {
    assert_eq!(solve("solve(2^x>3^x, x)"), "(-infinity, 0)");
    assert_eq!(solve("solve(2^x<3^x, x)"), "(0, infinity)");
    assert_eq!(solve("solve(5^x>2^x, x)"), "(0, infinity)");
}

#[test]
fn mixed_base_nonstrict_inequalities_close_the_boundary() {
    assert_eq!(solve("solve(2^x>=3^x, x)"), "(-infinity, 0]");
    assert_eq!(solve("solve(3^x<=2^x, x)"), "(-infinity, 0]");
}

#[test]
fn affine_wrapped_mixed_base_keeps_the_open_ray() {
    // 2^(x+1) > 3^x ⇔ x < ln2/ln(3/2); the engine emits the equivalent
    // ln(1/2)/ln(2/3) form. The old bug returned ONLY the boundary point.
    let out = solve("solve(2^(x+1)>3^x, x)");
    assert!(
        out.starts_with("(-infinity,") && out.ends_with(")") && out.contains("ln"),
        "expected an open ray with a log endpoint, got: {out}"
    );
}

#[test]
fn mixed_base_equation_still_solves_discretely() {
    assert_eq!(solve("solve(2^x=3^x, x)"), "{ 0 }");
    assert_eq!(solve("solve(4^x-9^x=0, x)"), "{ 0 }");
}

// ───────────────────────── Family C ─────────────────────────

#[test]
fn reciprocal_abs_punctures_the_pole() {
    assert_eq!(solve("solve(1/abs(x)>2, x)"), "(-1/2, 0) U (0, 1/2)");
    assert_eq!(solve("solve(1/abs(x)>=2, x)"), "[-1/2, 0) U (0, 1/2]");
    assert_eq!(solve("solve(2/abs(x)>4, x)"), "(-1/2, 0) U (0, 1/2)");
    // Shifted argument: the pole moves with g.
    assert_eq!(solve("solve(3/abs(x+1)>6, x)"), "(-3/2, -1) U (-1, -1/2)");
    assert_eq!(solve("solve(1/abs(x-1)>2, x)"), "(1/2, 1) U (1, 3/2)");
}

#[test]
fn reciprocal_abs_c_zero_no_degenerate_intervals() {
    // Used to emit (−∞,−∞) ∪ (∞,∞).
    assert_eq!(
        solve("solve(1/abs(x)>0, x)"),
        "(-infinity, 0) U (0, infinity)"
    );
}

#[test]
fn reciprocal_abs_less_than_and_negative_coefficient() {
    assert_eq!(
        solve("solve(1/abs(x)<2, x)"),
        "(-infinity, -1/2) U (1/2, infinity)"
    );
    assert_eq!(
        solve("solve(1/abs(x)<=2, x)"),
        "(-infinity, -1/2] U [1/2, infinity)"
    );
    // −1/|x| > −2 ⇔ 1/|x| < 2
    assert_eq!(
        solve("solve(-1/abs(x)>-2, x)"),
        "(-infinity, -1/2) U (1/2, infinity)"
    );
    // A positive LHS can never be ≤ 0.
    assert_eq!(solve("solve(1/abs(x)<=0, x)"), "No solution");
}

#[test]
fn reciprocal_abs_reversed_sides() {
    assert_eq!(solve("solve(2<1/abs(x), x)"), "(-1/2, 0) U (0, 1/2)");
}

#[test]
fn twin_paths_stay_consistent() {
    // The twin forms that always punctured must keep doing so.
    assert_eq!(solve("solve(abs(1/x)>2, x)"), "(-1/2, 0) U (0, 1/2)");
    assert_eq!(solve("solve(1/x^2>4, x)"), "(-1/2, 0) U (0, 1/2)");
}
