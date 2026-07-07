//! Contracts for solving inverse-trig / hyperbolic equations (auto-improvement
//! cycle): `arcsin/arccos/arctan/sinh/cosh/tanh(g) = c`.
//!
//! The isolation dispatch has no inverse for these and used to ERROR
//! ("función no definida"). A dedicated handler applies the (single, monotone)
//! forward inverse — `arcsin(x)=c → x=sin(c)`, `sinh(x)=c → x=asinh(c)`,
//! `cosh(x)=c → x=±acosh(c)` — GATED by the exact const-decision layer: a
//! threshold provably outside the range is `No solution`, provably inside
//! reduces and recurses. (solve's root verification does NOT catch these
//! transcendental range violations, so the gate is mandatory.)

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
fn in_range_thresholds_apply_the_forward_inverse() {
    assert_eq!(solve("arcsin(x) = 1/2"), "{ sin(1/2) }");
    assert_eq!(solve("arccos(x) = 1"), "{ cos(1) }");
    assert_eq!(solve("arctan(x) = 1"), "{ tan(1) }");
    assert_eq!(solve("sinh(x) = 1"), "{ asinh(1) }");
    assert_eq!(solve("tanh(x) = 1/2"), "{ atanh(1/2) }");
    // arctan(x) = π/3 folds to the exact tan(π/3) = sqrt(3).
    assert_eq!(solve("arctan(x) = pi/3"), "{ sqrt(3) }");
}

#[test]
fn cosh_yields_both_branches_and_dedups_at_one() {
    assert_eq!(solve("cosh(x) = 2"), "{ acosh(2), -acosh(2) }");
    // acosh(1) = 0, so the two branches coincide — no duplicate root.
    assert_eq!(solve("cosh(x) = 1"), "{ 0 }");
    assert_eq!(solve("cosh(2*x) = 2"), "{ 1/2·acosh(2), -1/2·acosh(2) }");
}

#[test]
fn out_of_range_thresholds_have_no_real_solution() {
    // The const-decision gate is exact even at fine borderlines vs the
    // transcendental bounds π/2 ≈ 1.5708, π ≈ 3.1416.
    assert_eq!(solve("arcsin(x) = 5"), "No solution");
    assert_eq!(solve("arcsin(x) = 8/5"), "No solution"); // 1.6 > π/2
    assert_eq!(solve("arctan(x) = 158/100"), "No solution"); // 1.58 > π/2
    assert_eq!(solve("arccos(x) = 16/5"), "No solution"); // 3.2 > π
    assert_eq!(solve("arccos(x) = -1/10"), "No solution"); // < 0
    assert_eq!(solve("tanh(x) = 2"), "No solution");
    assert_eq!(solve("tanh(x) = 1"), "No solution"); // open range excludes ±1
    assert_eq!(solve("cosh(x) = 1/2"), "No solution"); // cosh ≥ 1
    assert_eq!(solve("arctan(x) = pi/2"), "No solution"); // tan undefined there
}

#[test]
fn just_inside_the_range_still_solves() {
    assert_eq!(solve("arcsin(x) = 3/2"), "{ sin(3/2) }"); // 1.5 < π/2
    assert_eq!(solve("arctan(x) = 157/100"), "{ tan(157/100) }"); // 1.57 < π/2
    assert_eq!(solve("arccos(x) = 3"), "{ cos(3) }"); // 3 < π
    assert_eq!(solve("tanh(x) = 99/100"), "{ atanh(99/100) }");
}

#[test]
fn range_boundaries_and_affine_arguments() {
    assert_eq!(solve("arccos(x) = 0"), "{ 1 }");
    assert_eq!(solve("arccos(x) = pi"), "{ -1 }");
    assert_eq!(solve("arcsin(x) = pi/2"), "{ 1 }");
    assert_eq!(solve("arcsin(2*x) = 1/2"), "{ 1/2·sin(1/2) }");
    assert_eq!(solve("sinh(x-1) = 1"), "{ asinh(1) + 1 }");
}

#[test]
fn sinh_is_unconditional_and_forward_trig_is_untouched() {
    assert_eq!(solve("sinh(x) = -3"), "{ asinh(-3) }");
    // The forward trig / log solvers are unchanged.
    assert_eq!(
        solve("sin(x) = 1/2"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(solve("cos(x) = 1"), "{ k·2·pi : k ∈ ℤ }");
    assert_eq!(solve("ln(x) = 2"), "{ e^2 }");
}

/// Runs `solve(input, x)` and returns `(ok, result_or_error)`.
fn solve_raw(input: &str) -> (bool, String) {
    let out = Command::new(cargo::cargo_bin!("cas_cli"))
        .args(["eval", &format!("solve({input}, x)"), "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
    let ok = wire["ok"].as_bool().unwrap_or(false);
    let payload = if ok {
        wire["result"].as_str().unwrap_or("").to_string()
    } else {
        wire["error"].as_str().unwrap_or("").to_string()
    };
    (ok, payload)
}

#[test]
fn non_invertible_builtin_declines_to_a_residual_not_an_error() {
    // A DEFINED builtin the isolation cannot invert used to ERROR
    // `función [f] no definida` in the bare `f(x) = c` form — misleading,
    // because the function IS defined; only the solver cannot invert it.
    // It now declines to the honest operator-preserving residual, matching
    // the compound form `x + arcsin(x) = 1` which already residualized.
    assert_eq!(
        solve_raw("arcsin(x) = a"),
        (true, "solve(arcsin(x) = a, x)".into())
    );
    assert_eq!(
        solve_raw("arccos(x) = a"),
        (true, "solve(arccos(x) = a, x)".into())
    );
    assert_eq!(
        solve_raw("arctan(x) = a"),
        (true, "solve(arctan(x) = a, x)".into())
    );
    // The inverse-reciprocal trig (`acot`/`asec`, which rewrite through
    // `arctan(1/x)`/`arccos(1/x)`) are defined builtins the handler does not
    // reduce, so they residualize rather than error.
    assert_eq!(
        solve_raw("acot(x) = 1"),
        (true, "solve(arctan(1 / x) = 1, x)".into())
    );
}

#[test]
fn inverse_hyperbolics_as_outer_function_solve() {
    // Mirror of the inverse-trig arms: asinh: ℝ→ℝ and atanh: (−1,1)→ℝ are
    // bijections, so the forward hyperbolic applies unconditionally.
    assert_eq!(solve("asinh(x) = 1"), "{ sinh(1) }");
    assert_eq!(solve("atanh(x) = 1/2"), "{ tanh(1/2) }");
    // atanh has no threshold — every real c reduces (tanh(c) ∈ (−1,1)).
    assert_eq!(solve("atanh(x) = 5"), "{ tanh(5) }");
    // Bijection onto ℝ ⇒ even a symbolic RHS solves (was a residual before).
    assert_eq!(solve("asinh(x) = a"), "{ sinh(a) }");
    // The `arcsinh` input spelling resolves to the same handler.
    assert_eq!(solve("arcsinh(x) = 1"), "{ sinh(1) }");
    // Affine arguments compose through the reduce-to-canonical recursion.
    assert_eq!(solve("asinh(2*x) = 1"), "{ 1/2·sinh(1) }");
}

#[test]
fn acosh_is_gated_on_the_nonnegative_range() {
    // acosh's range is [0, ∞): `acosh(x) = c` needs `c ≥ 0`, and the preimage
    // `x = cosh(c) ≥ 1` is single (acosh is the non-negative branch).
    assert_eq!(solve("acosh(x) = 2"), "{ cosh(2) }");
    assert_eq!(solve("acosh(x) = 0"), "{ 1 }"); // cosh(0) = 1
    assert_eq!(solve("acosh(x) = -1"), "No solution"); // c < 0, no real preimage
}

#[test]
fn genuinely_unknown_functions_keep_the_no_definida_error() {
    // `gamma`/`erf`/`foo` are NOT builtins — `no definida` is the honest
    // answer (the function truly is not defined), so it must NOT residualize.
    for f in ["gamma", "erf", "foo"] {
        let (ok, msg) = solve_raw(&format!("{f}(x) = 1"));
        assert!(!ok, "{f} should stay an error, got ok with {msg}");
        assert!(
            msg.contains("no definida"),
            "{f} should keep the `no definida` error, got: {msg}"
        );
    }
}
