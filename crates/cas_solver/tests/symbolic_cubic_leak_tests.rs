//! Contract tests for degree>=3 polynomial equations with SYMBOLIC coefficients
//! (2026-07-13, family F8 of docs/AUDITORIA_FRONTERA_2026-07-13.md). The engine
//! has no symbolic-coefficient solver for degree >= 3 (`Polynomial` stores
//! rational coeffs; Cardano is rational-only), and base-side power isolation took
//! the n-th root of both sides with no "rhs still has the variable" progress
//! guard, leaking a self-referential `solve(x - (-p*x - q)^(1/3) = 0, x)`. The
//! recovery echoes the ORIGINAL equation honestly instead (var on one side, a
//! collected polynomial) -- neither the (unimplemented) symbolic Cardano roots
//! nor the mangled radical.

use cas_ast::{Equation, RelOp};
use cas_parser::parse;
use cas_solver::api::solve;
use cas_solver::command_api::solve::display_solution_set;
use cas_solver::runtime::Simplifier;

fn solve_display(lhs: &str, rhs: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let lhs = parse(lhs, &mut simplifier.context).expect("parse lhs");
    let rhs = parse(rhs, &mut simplifier.context).expect("parse rhs");
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    match solve(&eq, "x", &mut simplifier) {
        Ok((set, _)) => display_solution_set(&simplifier.context, &set),
        Err(e) => format!("ERR: {e:?}"),
    }
}

#[test]
fn symbolic_degree3_plus_polynomial_echoes_instead_of_leaking_a_radical() {
    // Every member: an honest `solve(<polynomial> = 0, x)` echo. The defining
    // properties (render-robust): the variable is NOT under a fractional root
    // (no `^(1/3)`/`^(1/4)`/`^(1/5)`), and the leading power survives.
    for (lhs, wants) in [
        ("x^3 + p*x + q", "x^3"),
        ("x^3 + a*x^2 + b*x + c", "x^3"),
        ("x^3 + p*x + 1", "x^3"),
        ("x^3 + p*x", "x^3"),
        ("x^4 + p*x + q", "x^4"),
        ("x^5 + p*x + q", "x^5"),
    ] {
        let out = solve_display(lhs, "0");
        assert!(
            out.contains(wants) && !out.contains("^(1/"),
            "{lhs}=0 -> {out}"
        );
    }
    // `x^3 = p*x^2` echoes the collected polynomial rather than `(p*x^2)^(1/3)`.
    let out = solve_display("x^3", "p*x^2");
    assert!(
        out.contains("x^3") && !out.contains("^(1/"),
        "x^3=p*x^2 -> {out}"
    );
}

#[test]
fn productive_and_lower_degree_paths_are_unchanged() {
    // rhs free of the variable: genuine n-th-root solution, NOT echoed.
    assert_eq!(solve_display("x^3", "a"), "{ a^(1/3) }");
    // Fractional-exponent radical isolation: still solved.
    assert_eq!(solve_display("sqrt(x)", "x - 2"), "{ 4 }");
    assert_eq!(solve_display("x^2", "sqrt(x)"), "{ 0, 1 }");
    // Degree-2 symbolic: the quadratic formula path is untouched.
    let quad = solve_display("x^2 + p*x + q", "0");
    assert!(
        quad.contains("p^2 - 4") && !quad.contains("solve("),
        "{quad}"
    );
    // Numeric cubics: Cardano / rational roots, not an echo.
    assert_eq!(solve_display("x^3 - 6*x^2 + 11*x - 6", "0"), "{ 1, 2, 3 }");
    // Biquadratic surds: the z = x^2 substitution still fires (degree-4 but solved).
    assert!(
        !solve_display("x^4 - 8*x^2 + 15", "0").contains("solve("),
        "biquadratic must still resolve"
    );
}
