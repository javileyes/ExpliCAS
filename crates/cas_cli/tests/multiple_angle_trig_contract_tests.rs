//! Contract: pure multiple-angle trig equations `trig(n·x) = c` return the
//! COMPLETE periodic family (scout 2026-07-03, family A).
//!
//! Historical bug: the multiple-angle EXPANSION rewrites (`sin(3x) → 3s−4s³`)
//! fired inside the periodic handler's own simplifies, so the polynomial
//! fallback asserted a finite arcsin set as the complete answer — dropping
//! the `2kπ/n` periodicity (e.g. `13π/18` solves `sin(3x)=1/2` and was
//! missing) with `ok:true` and no residual. The handler now gates those
//! expansions for its own scope (contractions stay live) and reduces
//! `A·sin(u)·cos(u)=c` to `sin(2u)=2c/A`.

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

/// The family must be periodic (contains the parameter `k`) and include the
/// expected fragments.
fn assert_periodic_family(input: &str, must_contain: &[&str]) {
    let out = solve(input);
    assert!(
        out.contains('k'),
        "{input}: expected a periodic family, got finite/other: {out}"
    );
    for frag in must_contain {
        assert!(
            out.contains(frag),
            "{input}: missing fragment {frag:?} in: {out}"
        );
    }
}

#[test]
fn sin_3x_half_returns_full_periodic_family() {
    // {π/18 + 2kπ/3, 5π/18 + 2kπ/3}
    assert_periodic_family("solve(sin(3*x)=1/2, x)", &["1/18·pi", "5/18·pi", "2/3·pi"]);
}

#[test]
fn coefficient_wrapped_forms_normalize_to_the_same_family() {
    for eq in [
        "solve(2*sin(3*x)-1=0, x)",
        "solve(2*sin(3*x)=1, x)",
        "solve(sin(3*x)-1/2=0, x)",
    ] {
        assert_periodic_family(eq, &["1/18·pi", "5/18·pi", "2/3·pi"]);
    }
}

#[test]
fn cos_3x_half_returns_full_periodic_family() {
    // {π/9 + 2kπ/3, 5π/9 + 2kπ/3} (≡ ±π/9 + 2kπ/3)
    assert_periodic_family("solve(cos(3*x)=1/2, x)", &["1/9·pi", "5/9·pi", "2/3·pi"]);
}

#[test]
fn sin_5x_half_keeps_all_ten_residue_classes() {
    // {π/30 + 2kπ/5, π/6 + 2kπ/5} — the quintic path used to silently drop
    // the quartic factor and return only the sin(x)=1/2 family.
    assert_periodic_family("solve(sin(5*x)=1/2, x)", &["1/30·pi", "1/6·pi", "2/5·pi"]);
}

#[test]
fn sin_6x_half_does_not_inherit_the_triple_angle_path() {
    assert_periodic_family("solve(sin(6*x)=1/2, x)", &["1/36·pi", "5/36·pi", "1/3·pi"]);
}

#[test]
fn tan_3x_solves_instantly_with_full_family() {
    // Used to blow a 10s budget through the expanded rational form and echo a
    // pseudo-result; now {π/12 + kπ/3}.
    assert_periodic_family("solve(tan(3*x)=1, x)", &["1/12·pi", "1/3·pi"]);
}

#[test]
fn sin_times_cos_reduces_to_double_angle() {
    // sin(x)·cos(x) = 1/4 ⇔ sin(2x) = 1/2 → {π/12 + kπ, 5π/12 + kπ}
    assert_periodic_family("solve(sin(x)*cos(x)=1/4, x)", &["1/12·pi", "5/12·pi"]);
}

#[test]
fn out_of_range_rhs_still_declines() {
    let out = solve("solve(sin(3*x)=2, x)");
    assert!(out.contains("No solution"), "got: {out}");
}

#[test]
fn previously_working_shifted_and_power_forms_stay_intact() {
    // The gate must not regress the affine-shifted or power reductions.
    assert_periodic_family("solve(sin(3*x+1)=1/2, x)", &["2/3·pi"]);
    assert_periodic_family("solve(sin(x)^2=1/4, x)", &["1/6·pi", "5/6·pi"]);
    assert_periodic_family("solve(2*sin(3*x)^2=1, x)", &["1/12·pi", "1/6·pi"]);
}
