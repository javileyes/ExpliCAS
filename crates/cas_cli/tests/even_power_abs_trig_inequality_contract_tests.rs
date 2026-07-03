//! Contracts for the even-power / absolute-value trig inequality reduction
//! (auto-improvement cycle): `A·trig(g)² ⋚ c` and `A·|trig(g)| ⋚ c` now
//! reduce the square/abs to a sign case split on `trig(g)` and combine the
//! windows with the PeriodicIntervalUnion circular algebra.
//!
//! `sin(x)² < 1/4` ⟺ `|sin(x)| < 1/2` ⟺ `sin(x) > −1/2 ∩ sin(x) < 1/2`.
//! Detection runs on the RAW tree because `simplify` rewrites `tan(x)²` into
//! `sin(x)²/cos(x)²`. Perfect-square rational thresholds and every abs
//! threshold reduce to a rational sub-threshold the producer accepts;
//! non-perfect-square squares (`sin(x)² < 1/3`, √ irrational) decline
//! honestly. All sets were verified by multi-k numeric membership sampling.

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
    // Irrational sqrt threshold (producer needs rational): honest decline.
    assert_eq!(solve("sin(x)^2 < 1/3"), "solve(sin(x)^2 < 1 / 3, x)");
    assert_eq!(solve("cos(x)^2 > 1/2"), "solve(cos(x)^2 > 1 / 2, x)");
}

#[test]
fn non_trig_even_powers_and_abs_are_untouched() {
    assert_eq!(solve("abs(x) < 2"), "(-2, 2)");
    assert_eq!(solve("x^2 > 1"), "(-infinity, -1) U (1, infinity)");
    assert_eq!(solve("(x-1)^2 < 4"), "(-1, 3)");
}
