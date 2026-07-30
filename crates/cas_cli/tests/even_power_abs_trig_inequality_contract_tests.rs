//! Contracts for the even-power / absolute-value trig inequality reduction
//! (auto-improvement cycle): `A·trig(g)² ⋚ c` and `A·|trig(g)| ⋚ c` now
//! reduce the square/abs to a sign case split on `trig(g)` and combine the
//! windows with the PeriodicIntervalUnion circular algebra.
//!
//! `sin(x)² < 1/4` ⟺ `|sin(x)| < 1/2` ⟺ `sin(x) > −1/2 ∩ sin(x) < 1/2`.
//! Detection runs on the RAW tree because `simplify` rewrites `tan(x)²` into
//! `sin(x)²/cos(x)²`. Perfect-square rational thresholds and every abs
//! threshold reduce to a rational sub-threshold the producer accepts;
//! non-perfect-square sin/cos squares (`sin(x)² < 1/3`, √ irrational) fall
//! back to the DOUBLE-ANGLE reduction (`sin² ⋚ t ⟺ cos(2g) ⋛ 1−2t`), whose
//! threshold is rational again (F4 cycle). Reciprocal squares/abs (`sec²`,
//! `csc²`, `A/trig²`, `|csc|`) invert through the same reducer with the
//! pole puncture supplied by the `T > 0` conjunct. All sets were verified by
//! multi-k numeric membership sampling.

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
fn square_and_abs_reduce_to_the_same_window_union() {
    let expected =
        "{ (-1/6·pi + k·2·pi, 1/6·pi + k·2·pi), (5/6·pi + k·2·pi, 7/6·pi + k·2·pi) : k ∈ ℤ }";
    assert_eq!(solve("sin(x)^2 < 1/4"), expected);
    assert_eq!(solve("abs(sin(x)) < 1/2"), expected);
    // Positive rational coefficient normalizes into the threshold.
    assert_eq!(solve("4*sin(x)^2 < 1"), expected);
    // Orientation does not change the answer.
    assert_eq!(solve("1/4 > sin(x)^2"), expected);
}

#[test]
fn greater_takes_the_union_branch() {
    assert_eq!(
        solve("sin(x)^2 > 1/4"),
        "{ (1/6·pi + k·2·pi, 5/6·pi + k·2·pi), (7/6·pi + k·2·pi, 11/6·pi + k·2·pi) : k ∈ ℤ }"
    );
    assert_eq!(
        solve("abs(tan(x)) > 1"),
        "{ (1/4·pi + k·pi, 1/2·pi + k·pi), (1/2·pi + k·pi, 3/4·pi + k·pi) : k ∈ ℤ }"
    );
}

#[test]
fn tangent_square_detected_on_the_raw_tree() {
    // simplify rewrites tan(x)^2 -> sin^2/cos^2; raw detection catches it.
    assert_eq!(
        solve("tan(x)^2 < 1"),
        "{ (-1/4·pi + k·pi, 1/4·pi + k·pi) : k ∈ ℤ }"
    );
    assert_eq!(
        solve("abs(tan(x)) < 1"),
        "{ (-1/4·pi + k·pi, 1/4·pi + k·pi) : k ∈ ℤ }"
    );
}

#[test]
fn nonstrict_closes_the_endpoints() {
    assert_eq!(
        solve("sin(x)^2 <= 1/4"),
        "{ [-1/6·pi + k·2·pi, 1/6·pi + k·2·pi], [5/6·pi + k·2·pi, 7/6·pi + k·2·pi] : k ∈ ℤ }"
    );
}

#[test]
fn affine_arguments_map_through() {
    assert_eq!(
        solve("abs(sin(2*x)) < 1/2"),
        "{ (-1/12·pi + k·pi, 1/12·pi + k·pi), (5/12·pi + k·pi, 7/12·pi + k·pi) : k ∈ ℤ }"
    );
    assert_eq!(
        solve("sin(x+pi/3)^2 < 1/4"),
        "{ (-1/2·pi + k·2·pi, -1/6·pi + k·2·pi), (1/2·pi + k·2·pi, 5/6·pi + k·2·pi) : k ∈ ℤ }"
    );
}

#[test]
fn nonpositive_and_boundary_thresholds_settle_or_decline() {
    assert_eq!(solve("sin(x)^2 < 0"), "No solution");
    assert_eq!(solve("sin(x)^2 >= 0"), "All real numbers");
    assert_eq!(solve("cos(x)^2 <= 1"), "All real numbers");
    // trig^2 > 0 is the punctured line (trig != 0), NOT All real numbers.
    assert_eq!(
        solve("sin(x)^2 > 0"),
        "{ (k·2·pi, pi + k·2·pi), (pi + k·2·pi, 2·pi + k·2·pi) : k ∈ ℤ }"
    );
    // trig^2 <= 0 -> trig = 0, a point set the window combiner declines.
    assert_eq!(solve("cos(x)^2 <= 0"), "solve(cos(x)^2 <= 0, x)");
}

#[test]
fn irrational_sqrt_threshold_takes_the_double_angle_route() {
    // √(1/3) is irrational, so the |sin| ⋚ √t route declines its sub-solves;
    // the double-angle fallback rewrites sin² < 1/3 ⟺ cos(2x) > 1/3 and the
    // symbolic-arccos producer answers exactly (F4 cycle).
    assert_eq!(
        solve("sin(x)^2 < 1/3"),
        "{ (-1/2·arccos(1/3) + k·pi, 1/2·arccos(1/3) + k·pi) : k ∈ ℤ }"
    );
    assert_eq!(
        solve("cos(x)^2 > 1/2"),
        "{ (-1/4·pi + k·pi, 1/4·pi + k·pi) : k ∈ ℤ }"
    );
}

#[test]
fn reciprocal_square_punctures_the_poles() {
    // sec(x)² > 2 ⟺ 0 < cos² < 1/2: the pole π/2 + kπ sits at each window
    // midpoint and must stay EXCLUDED (F4 P0: this returned «No solution»).
    let sec_gt = "{ (-1/2·pi + k·2·pi, -1/4·pi + k·2·pi), (1/4·pi + k·2·pi, 1/2·pi + k·2·pi), \
                 (1/2·pi + k·2·pi, 3/4·pi + k·2·pi), (5/4·pi + k·2·pi, 3/2·pi + k·2·pi) : k ∈ ℤ }";
    assert_eq!(solve("sec(x)^2 > 2"), sec_gt);
    // The explicit reciprocal spellings reduce identically.
    assert_eq!(solve("2/cos(x)^2 > 4"), sec_gt);
    assert_eq!(
        solve("csc(x)^2 > 2"),
        "{ (k·2·pi, 1/4·pi + k·2·pi), (3/4·pi + k·2·pi, pi + k·2·pi), \
         (pi + k·2·pi, 5/4·pi + k·2·pi), (7/4·pi + k·2·pi, 2·pi + k·2·pi) : k ∈ ℤ }"
    );
    // Nonstrict closes the attained bound but keeps the poles open.
    assert_eq!(
        solve("sec(x)^2 >= 2"),
        "{ (-1/2·pi + k·2·pi, -1/4·pi + k·2·pi], [1/4·pi + k·2·pi, 1/2·pi + k·2·pi), \
         (1/2·pi + k·2·pi, 3/4·pi + k·2·pi], [5/4·pi + k·2·pi, 3/2·pi + k·2·pi) : k ∈ ℤ }"
    );
}

#[test]
fn reciprocal_square_less_than_needs_no_puncture() {
    // sec² < 2 ⟺ cos² > 1/2 (T > 1/r > 0 already excludes the pole).
    assert_eq!(
        solve("sec(x)^2 < 2"),
        "{ (-1/4·pi + k·pi, 1/4·pi + k·pi) : k ∈ ℤ }"
    );
    assert_eq!(
        solve("csc(x)^2 < 2"),
        "{ (1/4·pi + k·pi, 3/4·pi + k·pi) : k ∈ ℤ }"
    );
    assert_eq!(
        solve("csc(x)^2 <= 2"),
        "{ [1/4·pi + k·pi, 3/4·pi + k·pi] : k ∈ ℤ }"
    );
    // Negative-power spelling of the same reciprocal square.
    assert_eq!(
        solve("sin(x)^(-2) < 4"),
        "{ (1/6·pi + k·2·pi, 5/6·pi + k·2·pi), (7/6·pi + k·2·pi, 11/6·pi + k·2·pi) : k ∈ ℤ }"
    );
    // Reciprocal ABS: |csc| < 2 ⟺ |sin| > 1/2 (pole exclusion automatic).
    assert_eq!(
        solve("abs(csc(x)) < 2"),
        "{ (1/6·pi + k·2·pi, 5/6·pi + k·2·pi), (7/6·pi + k·2·pi, 11/6·pi + k·2·pi) : k ∈ ℤ }"
    );
}

#[test]
fn declined_branches_never_fabricate_empty() {
    // The hyperbolic members of F4: the abs split's branches (`tanh < 1`,
    // `tanh > −1`) both decline, and `Residual ∩ X → Empty` in the core
    // used to fabricate «No solution» (true set: ℝ). The combiner now
    // propagates the unsolved fragment — an HONEST residual echo. Full
    // range-based hyperbolic capability is a named follow-up.
    let tanh_abs = solve("abs(tanh(x)) < 1");
    assert_ne!(tanh_abs, "No solution");
    assert!(
        tanh_abs.starts_with("solve("),
        "expected echo, got {tanh_abs}"
    );
    let tanh_sq = solve("tanh(x)^2 < 1");
    assert_ne!(tanh_sq, "No solution");
    assert!(
        tanh_sq.starts_with("solve("),
        "expected echo, got {tanh_sq}"
    );
    let sinh_sq = solve("sinh(x)^2 < 1");
    assert_ne!(sinh_sq, "No solution");
    assert!(
        sinh_sq.starts_with("solve("),
        "expected echo, got {sinh_sq}"
    );
}

#[test]
fn unmatched_reciprocal_trig_declines_honestly() {
    // sec/csc/cot forms outside the handled shapes must DECLINE (the
    // periodic-decline gate now recognizes the reciprocal names) rather
    // than fall to the monotonic inversion (F4: wrong finite sets).
    assert_eq!(solve("sec(x)^3 > 2"), "solve(sec(x)^3 > 2, x)");
    assert_eq!(solve("cot(x)^2 < 3"), "solve(cot(x)^2 < 3, x)");
}

#[test]
fn non_trig_even_powers_and_abs_are_untouched() {
    assert_eq!(solve("abs(x) < 2"), "(-2, 2)");
    assert_eq!(solve("x^2 > 1"), "(-infinity, -1) U (1, infinity)");
    assert_eq!(solve("(x-1)^2 < 4"), "(-1, 3)");
}
