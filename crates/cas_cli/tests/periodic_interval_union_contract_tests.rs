//! Contracts for the PeriodicIntervalUnion sin/cos producer (cycle P2,
//! design docs/DESIGN_PERIODIC_INTERVAL_UNION.md §5).
//!
//! Interior trig inequalities (`|c/A| < 1`) now emit the exact periodic
//! union of intervals instead of declining: the analytic u-space window
//! table mapped through the inverse affine transform, with per-op
//! closedness, wrap-capable windows (span ≤ period), symbolic
//! `arcsin/arccos` endpoints, and a numeric emission airbag that can only
//! degrade to a decline. Every set below was verified by independent
//! multi-k numeric membership sampling (~4000 points each) at review time.

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
fn bare_sine_all_four_operators() {
    assert_eq!(
        solve("sin(x)>1/2"),
        "{ (1/6·pi + k·2·pi, 5/6·pi + k·2·pi) : k ∈ ℤ }"
    );
    assert_eq!(
        solve("sin(x)>=1/2"),
        "{ [1/6·pi + k·2·pi, 5/6·pi + k·2·pi] : k ∈ ℤ }"
    );
    assert_eq!(
        solve("sin(x)<1/2"),
        "{ (5/6·pi + k·2·pi, 13/6·pi + k·2·pi) : k ∈ ℤ }"
    );
    assert_eq!(
        solve("sin(x)<=-1/2"),
        "{ [7/6·pi + k·2·pi, 11/6·pi + k·2·pi] : k ∈ ℤ }"
    );
}

#[test]
fn cosine_wrap_window_crosses_the_fundamental_domain() {
    assert_eq!(
        solve("cos(x)>1/2"),
        "{ (-1/3·pi + k·2·pi, 1/3·pi + k·2·pi) : k ∈ ℤ }"
    );
    assert_eq!(
        solve("cos(x)<=1/2"),
        "{ [1/3·pi + k·2·pi, 5/3·pi + k·2·pi] : k ∈ ℤ }"
    );
}

#[test]
fn zero_threshold_c_equals_zero() {
    // Repo lesson: sweeps that branch on a constant MUST include c = 0.
    assert_eq!(solve("sin(x)>0"), "{ (k·2·pi, pi + k·2·pi) : k ∈ ℤ }");
    assert_eq!(
        solve("cos(x)<0"),
        "{ (1/2·pi + k·2·pi, 3/2·pi + k·2·pi) : k ∈ ℤ }"
    );
}

#[test]
fn multiple_angle_halves_the_period() {
    assert_eq!(
        solve("sin(2*x)>1/2"),
        "{ (1/12·pi + k·pi, 5/12·pi + k·pi) : k ∈ ℤ }"
    );
}

#[test]
fn affine_wrappers_scale_shift_and_full_stack() {
    assert_eq!(
        solve("sin(x+pi/3)>1/2"),
        "{ (-1/6·pi + k·2·pi, 1/2·pi + k·2·pi) : k ∈ ℤ }"
    );
    // Full affine stack + nonstrict: u = 2x − π/4 ∈ [π/3, 5π/3] mod 2π.
    assert_eq!(
        solve("cos(2*x-pi/4)<=1/2"),
        "{ [7/24·pi + k·pi, 23/24·pi + k·pi] : k ∈ ℤ }"
    );
}

#[test]
fn negative_argument_slope_swaps_endpoint_pairs() {
    // a = −1: endpoints swap as (value, BoundType) PAIRS; both stay open.
    // sin(π/3−x) > 1/2 ⇔ x ∈ (π/3 − 5π/6, π/3 − π/6) = (−π/2, π/6).
    assert_eq!(
        solve("sin(pi/3-x)>1/2"),
        "{ (-1/2·pi + k·2·pi, 1/6·pi + k·2·pi) : k ∈ ℤ }"
    );
}

#[test]
fn coefficient_wrappers_divide_and_flip() {
    assert_eq!(
        solve("2*sin(x)>1"),
        "{ (1/6·pi + k·2·pi, 5/6·pi + k·2·pi) : k ∈ ℤ }"
    );
    // Negative coefficient flips the relation: −2·sin(x) > 1 ⇔ sin(x) < −1/2.
    assert_eq!(
        solve("-2*sin(x)>1"),
        "{ (7/6·pi + k·2·pi, 11/6·pi + k·2·pi) : k ∈ ℤ }"
    );
}

#[test]
fn additive_constant_peels_into_the_threshold() {
    // 3·cos(x) + 1 > 2 ⇔ cos(x) > 1/3: symbolic arccos endpoint.
    assert_eq!(
        solve("3*cos(x)+1>2"),
        "{ (-arccos(1/3) + k·2·pi, arccos(1/3) + k·2·pi) : k ∈ ℤ }"
    );
    // 1 − 2·cos(x) ≥ 0 ⇔ cos(x) ≤ 1/2 (sign folded by the simplifier).
    assert_eq!(
        solve("1-2*cos(x)>=0"),
        "{ [1/3·pi + k·2·pi, 5/3·pi + k·2·pi] : k ∈ ℤ }"
    );
}

#[test]
fn non_exact_threshold_keeps_symbolic_arcsin_endpoints() {
    assert_eq!(
        solve("sin(x)>1/3"),
        "{ (arcsin(1/3) + k·2·pi, pi - arcsin(1/3) + k·2·pi) : k ∈ ℤ }"
    );
}

#[test]
fn window_longer_than_half_period_wraps() {
    assert_eq!(
        solve("sin(x)>-1/2"),
        "{ (-1/6·pi + k·2·pi, 7/6·pi + k·2·pi) : k ∈ ℤ }"
    );
}

#[test]
fn reversed_orientation_produces_the_same_window() {
    assert_eq!(solve("1/2<sin(x)"), solve("sin(x)>1/2"));
    assert_eq!(solve("1/3<cos(x)"), solve("cos(x)>1/3"));
}

#[test]
fn out_of_scope_shapes_still_decline_or_solve_exactly() {
    // Non-affine argument: the window table must NOT fire.
    assert_eq!(solve("sin(x^2)>1/2"), "solve(sin(x^2) > 1 / 2, x)");
    // tan solves via its P3 sibling branch (asymptote end always open).
    assert_eq!(
        solve("tan(x)>1"),
        "{ (1/4·pi + k·pi, 1/2·pi + k·pi) : k ∈ ℤ }"
    );
    // Boundary and equation paths are untouched.
    assert_eq!(solve("sin(x)>=1"), "{ 1/2·pi + k·2·pi : k ∈ ℤ }");
    assert_eq!(
        solve("sin(x)=1/2"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(solve("sin(x)<2"), "All real numbers");
    assert_eq!(solve("sin(x)>2"), "No solution");
}

#[test]
fn latex_renders_the_set_builder_frame() {
    let out = Command::new(cargo::cargo_bin!("cas_cli"))
        .args(["eval", "solve(sin(x)>1/2, x)", "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
    let latex = wire["result_latex"].as_str().unwrap_or("");
    assert!(
        latex.contains(r"k \in \mathbb{Z}") && latex.contains(r"\left("),
        "unexpected LaTeX: {latex}"
    );
}
