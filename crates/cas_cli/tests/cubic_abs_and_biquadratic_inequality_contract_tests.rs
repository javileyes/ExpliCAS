//! Contracts for the cubic-abs cycle (2026-07-31): THREE stacked defects made
//! `solve(abs(x³−x)<1)` drop whole interior regions.
//!
//! 1. `const_value_bounds` declined `Pow(negative, 1/3)` — the Cardano
//!    endpoints of single-real-root cubics spell `cbrt(negative)` that way,
//!    so the interval algebra fell to the value-blind structural order
//!    (fixed: odd-root negative-base bounds by odd monotonicity).
//! 2. The sign-split and abs-threshold assemblies COMMITTED to undecidable
//!    endpoint orders (casus-irreducibilis trig endpoints) — now an order
//!    guard declines to an honest echo of the ORIGINAL relation.
//! 3. The generic isolation's unconditional 4th root asserted «No solution»
//!    for the tautology `x⁴−x²+1>0` (possibly-negative radicand, branch with
//!    the variable on both sides) — now the biquadratic-inequality recovery
//!    re-derives through the exact `z = x²` sign analysis.
//!
//! Every pinned set verified by dense f64 membership sampling (204 cases).

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
fn single_root_cubic_abs_is_one_interval() {
    // |x³−x| < 1: max on [−1,1] is 2/(3√3) < 1, so the set is the SINGLE
    // band between the real roots of x³−x = ±1 (r ≈ ±1.3247). The old answer
    // `[−1,0] ∪ [1, r)` dropped (0,1) and (−r,−1) entirely.
    assert_eq!(
        solve("abs(x^3-x) < 1"),
        "(cbrt(1/6·(-sqrt(23/3) - 3)) + cbrt(1/6·(sqrt(23/3) - 3)), \
         cbrt(1/6·(sqrt(23/3) + 3)) + cbrt(1/6·(3 - sqrt(23/3))))"
    );
    // Complement and closures stay consistent.
    assert_eq!(
        solve("abs(x^3-x) >= 1"),
        "(-infinity, cbrt(1/6·(-sqrt(23/3) - 3)) + cbrt(1/6·(sqrt(23/3) - 3))] U \
         [cbrt(1/6·(sqrt(23/3) + 3)) + cbrt(1/6·(3 - sqrt(23/3))), infinity)"
    );
    // Monotone-argument sibling: exact rational endpoints.
    assert_eq!(solve("abs(x^3+x) < 2"), "(-1, 1)");
}

#[test]
fn three_root_cubic_abs_declines_honestly() {
    // x³−4x has THREE real roots; the branch endpoints are the
    // casus-irreducibilis trig forms the bounds oracle cannot yet separate.
    // The order guard must decline with the ORIGINAL relation echoed —
    // never the old bridged sets, never a mangled internal residual.
    let res = solve("abs(x^3-4*x) < 2");
    assert_eq!(res, "solve(|x^3 - 4·x| < 2, x)");
}

#[test]
fn biquadratic_tautology_settles_all_reals() {
    // z-quadratic disc < 0 with positive lead: constant sign over ℝ. The
    // unconditional-4th-root isolation asserted «No solution».
    assert_eq!(solve("x^4-x^2+1 > 0"), "All real numbers");
    assert_eq!(solve("x^4-x^2+1 < 0"), "No solution");
    assert_eq!(solve("x^4-x^2 > -1"), "All real numbers");
    assert_eq!(solve("x^4-x^2+1/2 > 0"), "All real numbers");
}

#[test]
fn quartic_abs_assembles_the_full_band() {
    // |x⁴−x²| < 1: all of [−1,1] qualifies (min is −1/4), and the band runs
    // to x² = φ. The broken neg-branch used to erase (−1,0)∪(0,1).
    assert_eq!(solve("abs(x^4-x^2) < 1"), "(-sqrt(phi), sqrt(phi))");
    assert_eq!(solve("x^4-x^2 < 1"), "(-sqrt(phi), sqrt(phi))");
}

#[test]
fn biquadratic_edge_cases_stay_exact() {
    // Double z-root at z0 = 1 (perfect square quartic).
    assert_eq!(
        solve("x^4-2*x^2+1 > 0"),
        "(-infinity, -1) U (-1, 1) U (1, infinity)"
    );
    // Roots as plain surds (the pre-existing owner keeps its presentation).
    assert_eq!(
        solve("x^4-8*x^2+15 < 0"),
        "(-sqrt(5), -sqrt(3)) U (sqrt(3), sqrt(5))"
    );
    // z0 < 0: the x² = z0 slice is empty.
    assert_eq!(solve("x^4+2*x^2+1 <= 0"), "No solution");
}

#[test]
fn plain_abs_owners_are_untouched() {
    assert_eq!(solve("abs(x) < 2"), "(-2, 2)");
    assert_eq!(solve("abs(x^2-4) < 3"), "(-sqrt(7), -1) U (1, sqrt(7))");
    assert_eq!(solve("x^2-3*abs(x)+2 < 0"), "(-2, -1) U (1, 2)");
    assert_eq!(solve("abs(ln(x)) < 1"), "(1 / e, e)");
    assert_eq!(solve("abs(sinh(x)) <= 2"), "[-asinh(2), asinh(2)]");
}
