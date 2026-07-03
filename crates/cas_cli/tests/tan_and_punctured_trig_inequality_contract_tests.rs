//! Contracts for cycle P3a of PeriodicIntervalUnion: the tan sibling branch,
//! the punctured-line complements, and the Neg-signed additive peel.
//!
//! tan dispatches BEFORE the sin/cos |r| range ladder (its range is ℝ — the
//! panel trap `tan(x) >= 2` must never hit the `r > 1 → Empty` arm), its
//! windows always keep the asymptote endpoint OPEN, and mixed closedness in
//! one window (`tan(x) >= 0` → `[kπ, π/2+kπ)`) exercises the per-endpoint
//! BoundTypes. Boundary complements (`sin(x) < 1`) emit the punctured line —
//! the single all-open window of length == period — with the emission
//! airbag's outside-samples skipped (the complement is measure-zero).
//! All sets verified by independent multi-k numeric sampling.

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
fn tan_all_four_operators_with_asymptote_open() {
    assert_eq!(
        solve("tan(x)>1"),
        "{ (1/4·pi + k·pi, 1/2·pi + k·pi) : k ∈ ℤ }"
    );
    // MIXED closedness in one window: closed at the zero, open at the asymptote.
    assert_eq!(solve("tan(x)>=0"), "{ [k·pi, 1/2·pi + k·pi) : k ∈ ℤ }");
    assert_eq!(solve("tan(x)<=0"), "{ (-1/2·pi + k·pi, k·pi] : k ∈ ℤ }");
}

#[test]
fn tan_thresholds_beyond_one_do_not_collapse_to_empty() {
    // Panel trap: the sin/cos range ladder would answer ∅/ℝ for |r| ≥ 1.
    assert_eq!(
        solve("tan(x)>=2"),
        "{ [arctan(2) + k·pi, 1/2·pi + k·pi) : k ∈ ℤ }"
    );
    assert_eq!(
        solve("tan(x)<-3"),
        "{ (-1/2·pi + k·pi, -arctan(3) + k·pi) : k ∈ ℤ }"
    );
    assert_eq!(
        solve("tan(x)>1000"),
        "{ (arctan(1000) + k·pi, 1/2·pi + k·pi) : k ∈ ℤ }"
    );
}

#[test]
fn tan_affine_wrappers_and_reflection() {
    // Period π/2; the LOWER endpoint is the asymptote here.
    assert_eq!(
        solve("tan(2*x)<1"),
        "{ (-1/4·pi + k·1/2·pi, 1/8·pi + k·1/2·pi) : k ∈ ℤ }"
    );
    // a = −1: the pair-swap moves the Closed bound to the zero endpoint.
    assert_eq!(solve("tan(-x)>=0"), "{ (-1/2·pi + k·pi, k·pi] : k ∈ ℤ }");
    // Reversed orientation agrees.
    assert_eq!(solve("2<tan(x)"), solve("tan(x)>2"));
}

#[test]
fn boundary_complements_emit_the_punctured_line() {
    // ℝ ∖ {touch points}: single all-open window of length == period.
    assert_eq!(
        solve("sin(x)<1"),
        "{ (1/2·pi + k·2·pi, 5/2·pi + k·2·pi) : k ∈ ℤ }"
    );
    assert_eq!(
        solve("sin(x)>-1"),
        "{ (-1/2·pi + k·2·pi, 3/2·pi + k·2·pi) : k ∈ ℤ }"
    );
    assert_eq!(solve("cos(x)<1"), "{ (k·2·pi, 2·pi + k·2·pi) : k ∈ ℤ }");
    assert_eq!(
        solve("cos(x)>-1"),
        "{ (-pi + k·2·pi, pi + k·2·pi) : k ∈ ℤ }"
    );
    // Reversed orientation of the historical wrong ray `1 > sin(x)`.
    assert_eq!(solve("1>sin(x)"), solve("sin(x)<1"));
}

#[test]
fn neg_signed_additive_core_folds_into_the_coefficient() {
    // −cos(3x) − 2 ≥ −3 ⇔ cos(3x) ≤ 1: a tautology.
    assert_eq!(solve("-cos(3*x)-2>=-3"), "All real numbers");
    // 2 − sin(x) > 1 ⇔ sin(x) < 1: the punctured line.
    assert_eq!(
        solve("2-sin(x)>1"),
        "{ (1/2·pi + k·2·pi, 5/2·pi + k·2·pi) : k ∈ ℤ }"
    );
}

#[test]
fn tan_equation_and_exact_boundaries_untouched() {
    assert_eq!(solve("tan(x)=1"), "{ 1/4·pi + k·pi : k ∈ ℤ }");
    assert_eq!(solve("sin(x)>=1"), "{ 1/2·pi + k·2·pi : k ∈ ℤ }");
    assert_eq!(solve("sin(x)<=1"), "All real numbers");
}
