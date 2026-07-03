//! Contracts for cycle P3b of PeriodicIntervalUnion: reciprocal-trig
//! inequalities `A / trig(g) ⋚ c` (including the simplifier's unit-numerator
//! refolds `csc`/`sec`/`cot`).
//!
//! The handler normalizes to `1/s ⋚ r`, splits by the sign of `r` into
//! window relations on `s = trig(g)`, sub-solves each through the P2/P3a
//! producers, and combines with the P1 circular same-period algebra —
//! `1/sin(x) > 2` ⟺ `0 < sin(x) < 1/2` is the design's witness #8: TWO
//! disjoint windows per period, which a flat (non-circular) intersection
//! would silently truncate. All sets verified by independent multi-k
//! numeric sampling (~5000 points each).

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
fn witness_8_two_windows_per_period() {
    // 1/sin(x) > 2 ⟺ 0 < sin(x) < 1/2: the poles at kπ stay excluded.
    assert_eq!(
        solve("1/sin(x)>2"),
        "{ (k·2·pi, 1/6·pi + k·2·pi), (5/6·pi + k·2·pi, pi + k·2·pi) : k ∈ ℤ }"
    );
    // The coefficient shape reduces identically (2/sin(x) > 4 ⟺ same set).
    assert_eq!(solve("2/sin(x)>4"), solve("1/sin(x)>2"));
}

#[test]
fn nonstrict_closes_only_the_attained_bound() {
    // 1/sin(x) ≥ 2 ⟺ 0 < sin(x) ≤ 1/2: closed at sin = 1/2, open at poles.
    assert_eq!(
        solve("1/sin(x)>=2"),
        "{ (k·2·pi, 1/6·pi + k·2·pi], [5/6·pi + k·2·pi, pi + k·2·pi) : k ∈ ℤ }"
    );
    // Mixed closedness lands on the right ends per piece for sec.
    assert_eq!(
        solve("1/cos(x)<=-2"),
        "{ [-2/3·pi + k·2·pi, -1/2·pi + k·2·pi), (1/2·pi + k·2·pi, 2/3·pi + k·2·pi] : k ∈ ℤ }"
    );
}

#[test]
fn negative_threshold_takes_the_union_branch() {
    // 1/sin(x) > −2 ⟺ sin(x) > 0 OR sin(x) < −1/2: a genuine union.
    assert_eq!(
        solve("1/sin(x)>-2"),
        "{ (k·2·pi, pi + k·2·pi), (7/6·pi + k·2·pi, 11/6·pi + k·2·pi) : k ∈ ℤ }"
    );
    // 1/sin(x) < −1 ⟺ −1 < sin(x) < 0: punctures sin = −1 at 3π/2.
    assert_eq!(
        solve("1/sin(x)<-1"),
        "{ (-1/2·pi + k·2·pi, k·2·pi), (pi + k·2·pi, 3/2·pi + k·2·pi) : k ∈ ℤ }"
    );
}

#[test]
fn zero_threshold_reduces_to_the_sign_of_the_trig() {
    assert_eq!(solve("1/sin(x)>0"), "{ (k·2·pi, pi + k·2·pi) : k ∈ ℤ }");
    // 1/s ≤ 0 ⟺ s < 0 (1/s never equals 0).
    assert_eq!(
        solve("1/sin(x)<=0"),
        "{ (pi + k·2·pi, 2·pi + k·2·pi) : k ∈ ℤ }"
    );
}

#[test]
fn cot_shape_declines_and_affine_argument_solves() {
    // `1/tan(x)` refolds to cot(x), which is DEFINED at tan's poles
    // (cot(π/2) = 0) — the naive 1/tan reduction silently loses those points
    // from any set that should contain cot = 0 (final-audit finding), so cot
    // shapes DECLINE honestly until they get their own window table.
    assert_eq!(solve("1/tan(x)>1"), "solve(1 / tan(x) > 1, x)");
    assert_eq!(solve("cos(x)/sin(x)<=1"), "solve(cos(x) / sin(x) <= 1, x)");
    // Affine argument halves the period through the sub-solves.
    assert_eq!(
        solve("1/sin(2*x)>2"),
        "{ (k·pi, 1/12·pi + k·pi), (5/12·pi + k·pi, 1/2·pi + k·pi) : k ∈ ℤ }"
    );
}

#[test]
fn negative_coefficient_flips_before_the_case_split() {
    // −1/sin(x) > 2 ⟺ 1/sin(x) < −2 ⟺ −1/2 < sin(x) < 0.
    assert_eq!(
        solve("-1/sin(x)>2"),
        "{ (-1/6·pi + k·2·pi, k·2·pi), (pi + k·2·pi, 7/6·pi + k·2·pi) : k ∈ ℤ }"
    );
}

#[test]
fn irrational_argument_coefficients_solve_with_full_periodicity() {
    // Final-audit finding, then U1: the equation path used to drop
    // periodicity for irrational argument coefficients (sin(pi*x)=1 ->
    // { 1/2 }); the hotfix declined the inequality inheritance, and cycle
    // U1 now solves the family outright with rational x-periods.
    assert_eq!(solve("sin(pi*x)>=1"), "{ 1/2 + k·2 : k ∈ ℤ }");
    assert_eq!(solve("cos(pi*x)<=-1"), "{ 1 + k·2 : k ∈ ℤ }");
    assert_eq!(solve("cos(2*pi*x)>=1"), "{ k·1 : k ∈ ℤ }");
}

#[test]
fn non_trig_reciprocals_are_untouched() {
    assert_eq!(solve("1/abs(x)>2"), "(-1/2, 0) U (0, 1/2)");
    assert_eq!(solve("1/x>2"), "(0, 1/2)");
}
