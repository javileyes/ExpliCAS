//! Contracts for universality cycle U1: trig equations whose argument slope
//! is IRRATIONAL (π·x, √2·x, e·x) now emit the full periodic family.
//!
//! The periodic handler's affine gate accepted only rational slopes
//! (`Polynomial::from_expr`), so these declined into the principal-root
//! isolation, which asserted a SINGLETON as the complete answer —
//! `solve(sin(pi*x)=1)` → `{ 1/2 }`, losing `{ 1/2 + 2k }` (10 confirmed
//! wrong answers in the post-PIU adversarial audit; `sin(pi*x)=1/2` even
//! dropped the whole second branch 5/6). The gate now also accepts var-free
//! symbolic slopes with PROVABLY positive sign (exact second-difference
//! affinity + `provable_const_sign`; no f64), and the symbolic division
//! makes π-slopes yield fully RATIONAL x-periods (2π/π = 2). The
//! weak-boundary inequality reduction inherits the fix (`sin(pi*x) >= 1`
//! solves instead of declining). Every family verified numerically at 7-14
//! members at review time.

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
fn pi_slope_yields_rational_periods() {
    assert_eq!(solve("sin(pi*x)=1"), "{ 1/2 + k·2 : k ∈ ℤ }");
    assert_eq!(solve("sin(pi*x)=0"), "{ k·1 : k ∈ ℤ }");
    assert_eq!(solve("cos(pi*x)=1"), "{ k·2 : k ∈ ℤ }");
    assert_eq!(solve("cos(pi*x)=0"), "{ 1/2 + k·1 : k ∈ ℤ }");
    assert_eq!(solve("tan(pi*x)=0"), "{ k·1 : k ∈ ℤ }");
    assert_eq!(solve("sin(2*pi*x)=0"), "{ k·1/2 : k ∈ ℤ }");
}

#[test]
fn both_in_period_branches_survive() {
    // The audit's worst case: the singleton { 1/6 } lost BOTH the +2k
    // periodicity AND the entire second branch 5/6.
    assert_eq!(solve("sin(pi*x)=1/2"), "{ 1/6 + k·2, 5/6 + k·2 : k ∈ ℤ }");
}

#[test]
fn fractional_pi_slopes_fold_to_canonical_rationals() {
    assert_eq!(solve("sin(pi*x/2)=1"), "{ 1 + k·4 : k ∈ ℤ }");
    assert_eq!(solve("cos(2*pi*x/3)=0"), "{ 3/4 + k·3/2 : k ∈ ℤ }");
}

#[test]
fn symbolic_offset_maps_through() {
    assert_eq!(
        solve("sin(pi*x+1)=1"),
        "{ (pi - 2) / (2·pi) + k·2 : k ∈ ℤ }"
    );
}

#[test]
fn generic_irrational_slopes_keep_symbolic_periods() {
    assert_eq!(
        solve("sin(sqrt(2)*x)=1"),
        "{ 1/2·pi·2^(-1/2) + k·2·pi·2^(-1/2) : k ∈ ℤ }"
    );
    assert_eq!(solve("cos(e*x)=1"), "{ k·(pi·2)/e : k ∈ ℤ }");
}

#[test]
fn boundary_inequalities_inherit_the_fix() {
    // These asserted degenerate intervals like [1/2, 1/2] before U1.
    assert_eq!(solve("sin(pi*x)>=1"), "{ 1/2 + k·2 : k ∈ ℤ }");
    assert_eq!(solve("cos(pi*x)<=-1"), "{ 1 + k·2 : k ∈ ℤ }");
    assert_eq!(solve("cos(2*pi*x)>=1"), "{ k·1 : k ∈ ℤ }");
}

#[test]
fn rational_slope_families_are_untouched() {
    assert_eq!(
        solve("sin(3*x)=1/2"),
        "{ 1/18·pi + k·2/3·pi, 5/18·pi + k·2/3·pi : k ∈ ℤ }"
    );
    assert_eq!(solve("tan(3*x)=1"), "{ 1/12·pi + k·1/3·pi : k ∈ ℤ }");
    assert_eq!(solve("cos(2*x)=1"), "{ k·pi : k ∈ ℤ }");
}
