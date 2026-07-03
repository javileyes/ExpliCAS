//! Contracts for the trig-inequality ORIENTATION soundness fix (PIU design
//! review P0, 2026-07-03).
//!
//! `try_decline_periodic_trig_inequality` and
//! `try_solve_boundary_trig_inequality` used to inspect only `eq.lhs`; with
//! the trig on the RHS the relation fell through to the generic monotonic
//! inversion, which asserted WRONG rays: `solve(1/2<sin(x))` → `(π/6, ∞)`
//! (false at x=π), `solve(2<tan(x))` → `(arctan(2), ∞)`, `solve(1>sin(x))` →
//! `(−∞, π/2)`. Both handlers now normalize to trig-on-LHS (swapping sides
//! flips the operator), so each orientation pair below must produce the SAME
//! result: exact sets where the engine knows them, honest normalized
//! residual echoes where it does not (interior thresholds pending
//! PeriodicIntervalUnion).

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

fn assert_pair(reversed: &str, direct: &str) {
    let r = solve(reversed);
    let d = solve(direct);
    assert_eq!(
        r, d,
        "orientation must not change the answer: solve({reversed}) = {r:?} but solve({direct}) = {d:?}"
    );
}

#[test]
fn reversed_interior_thresholds_solve_or_decline_instead_of_wrong_rays() {
    // The three live wrong answers found by the design review panel. Since
    // cycle P2 the sin/cos cases SOLVE (PeriodicIntervalUnion); tan declines
    // honestly until its P3 handler. Neither may regress to a ray.
    assert_eq!(
        solve("1/2<sin(x)"),
        "{ (1/6·pi + k·2·pi, 5/6·pi + k·2·pi) : k ∈ ℤ }"
    );
    assert_eq!(solve("2<tan(x)"), "solve(tan(x) > 2, x)");
    assert_eq!(
        solve("1/3<cos(x)"),
        "{ (-arccos(1/3) + k·2·pi, arccos(1/3) + k·2·pi) : k ∈ ℤ }"
    );
}

#[test]
fn reversed_boundary_complement_declines_instead_of_wrong_ray() {
    // `1 > sin(x)` fell through the LHS-only boundary handler to `(−∞, π/2)`.
    assert_eq!(solve("1>sin(x)"), "solve(sin(x) < 1, x)");
    assert_eq!(solve("-1<sin(x)"), "solve(sin(x) > -1, x)");
}

#[test]
fn orientation_pairs_agree_on_interior_declines() {
    assert_pair("1/2<sin(x)", "sin(x)>1/2");
    assert_pair("1/2>=cos(2*x)", "cos(2*x)<=1/2");
    assert_pair("3<tan(x)", "tan(x)>3");
    assert_pair("0<sin(2*x)", "sin(2*x)>0");
}

#[test]
fn orientation_pairs_agree_on_boundary_touch_sets() {
    assert_pair("1<=sin(x)", "sin(x)>=1");
    assert_pair("-1>=cos(x)", "cos(x)<=-1");
    assert_eq!(solve("1<=sin(x)"), "{ 1/2·pi + k·2·pi : k ∈ ℤ }");
    assert_eq!(solve("-1>=cos(x)"), "{ pi + k·2·pi : k ∈ ℤ }");
}

#[test]
fn orientation_pairs_agree_on_range_guards() {
    assert_pair("2<=sin(x)", "sin(x)>=2");
    assert_pair("2>sin(x)", "sin(x)<2");
    assert_eq!(solve("2<=sin(x)"), "No solution");
    assert_eq!(solve("2>sin(x)"), "All real numbers");
}

#[test]
fn non_trig_inequalities_are_untouched() {
    assert_eq!(solve("2*x+1<7"), "(-infinity, 3)");
    assert_eq!(solve("x^2<4"), "(-2, 2)");
    // Reversed orientation of a plain monotonic relation still inverts fine.
    assert_eq!(solve("7>2*x+1"), "(-infinity, 3)");
}
