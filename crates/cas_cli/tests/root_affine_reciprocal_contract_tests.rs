//! Contracts for universality cycle U2 (scout backlog #4): reciprocals of
//! ROOTS of rational-affine arguments — `c / (a·x+b)^(1/q) ⋚ k`.
//!
//! Previously honest declines; now solved by the two-stage monotone
//! substitution `w = (a·x+b)^(1/q)`: the w-space relation `c/w ⋚ k` has
//! rational breakpoints and an exact owner, the w-set maps through the
//! increasing power `u = w^q` (clamped to `w > 0` for even q — the root's
//! domain), then through the inverse affine to x. All sets verified by
//! independent numeric sampling (~6000 points each) at review time.

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
fn sqrt_reciprocals_respect_the_domain() {
    // The scout's three witnesses.
    assert_eq!(solve("1/sqrt(x)>2"), "(0, 1/4)");
    assert_eq!(solve("1/sqrt(x-1)>2"), "(1, 5/4)");
    assert_eq!(solve("1/x^(1/3)>2"), "(0, 1/8)");
    // The complement stays inside the root's domain (x > 0 required).
    assert_eq!(solve("1/sqrt(x)<2"), "(1/4, infinity)");
    assert_eq!(solve("1/sqrt(x)>=2"), "(0, 1/4]");
}

#[test]
fn odd_roots_cover_the_negative_branch() {
    // w = x^(1/3) < 0 for x < 0: the negative branch is real and monotone.
    assert_eq!(solve("1/x^(1/3)<-1"), "(-1, 0)");
    // Negative threshold takes the union branch: w > 0 OR w < -1/2.
    assert_eq!(solve("1/x^(1/3)>-2"), "(-infinity, -1/8) U (0, infinity)");
    assert_eq!(solve("1/x^(1/5)>1"), "(0, 1)");
}

#[test]
fn coefficients_shifts_and_negations_normalize() {
    assert_eq!(solve("2/sqrt(x+3)<=1"), "[1, infinity)");
    assert_eq!(solve("-1/sqrt(x)<-2"), "(0, 1/4)");
    assert_eq!(solve("1/(x+1)^(1/3)>2"), "(-1, -7/8)");
    assert_eq!(solve("1/x^(1/2)>2"), "(0, 1/4)");
}

#[test]
fn adjacent_owners_are_untouched() {
    assert_eq!(solve("1/sqrt(x)=2"), "{ 1/4 }");
    assert_eq!(solve("sqrt(x)>2"), "(4, infinity)");
    assert_eq!(solve("1/x>2"), "(0, 1/2)");
    assert_eq!(solve("1/x^2>4"), "(-1/2, 0) U (0, 1/2)");
    // Even-numerator valleys keep their own reduction.
    assert_eq!(solve("(x-1)^(2/3)>4"), "(-infinity, -7) U (9, infinity)");
}
