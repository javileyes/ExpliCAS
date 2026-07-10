//! Contract tests for `trig(u) = trig(v)` via sum-to-product (2026-07-10).
//! Degree-≥3 multiple-angle expansions are not quadratic-in-one-atom, so the
//! generic isolation leaked a self-referential arcsin echo
//! (`solve(x − arcsin(2·sin(x)³) = 0)` for `sin(3x) = sin(x)`). Family F21 of
//! docs/AUDITORIA_FRONTERA_2026-07-09.md. The identities reduce the difference to
//! a product of two trig factors whose zero set the periodic solver unions
//! exactly over a common period.

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
    let (set, _) = solve(&eq, "x", &mut simplifier).expect("solve");
    display_solution_set(&simplifier.context, &set)
}

#[test]
fn multiple_angle_sin_and_cos_equalities_resolve_periodically() {
    // sin(3x) = sin(x): {kπ} ∪ {π/4 + kπ/2}, presented as bases over 2π.
    assert_eq!(
        solve_display("sin(3*x)", "sin(x)"),
        "{ k·2 * pi, pi + k·2 * pi, -1/4 * pi + k·2 * pi, 5/4 * pi + k·2 * pi, 1/4 * pi + k·2 * pi, 3/4 * pi + k·2 * pi : k ∈ ℤ }"
    );
    // sin(5x) = sin(x): {kπ/2} ∪ {π/6 + kπ/3} — eight bases over 2π.
    assert_eq!(
        solve_display("sin(5*x)", "sin(x)"),
        "{ -1/2 * pi + k·2 * pi, -1/6 * pi + k·2 * pi, 7/6 * pi + k·2 * pi, k·2 * pi, pi + k·2 * pi, 1/6 * pi + k·2 * pi, 5/6 * pi + k·2 * pi, 1/2 * pi + k·2 * pi : k ∈ ℤ }"
    );
    // sin(3x) = sin(2x): {2kπ} ∪ {π/5 + 2kπ/5}.
    assert_eq!(
        solve_display("sin(3*x)", "sin(2*x)"),
        "{ k·2 * pi, 1/5 * pi + k·2 * pi, 3/5 * pi + k·2 * pi, pi + k·2 * pi, 7/5 * pi + k·2 * pi, 9/5 * pi + k·2 * pi : k ∈ ℤ }"
    );
    // cos(2x) = cos(x): {2kπ/3} (previously a conditional arccos leak).
    assert_eq!(
        solve_display("cos(2*x)", "cos(x)"),
        "{ k·2 * pi, 2/3 * pi + k·2 * pi, 4/3 * pi + k·2 * pi : k ∈ ℤ }"
    );
}

#[test]
fn previously_working_trig_shapes_keep_their_owners() {
    assert_eq!(
        solve_display("sin(2*x)", "sin(x)"),
        "{ k·2 * pi, 1/3 * pi + k·2 * pi, pi + k·2 * pi, 5/3 * pi + k·2 * pi : k ∈ ℤ }"
    );
    assert_eq!(
        solve_display("sin(3*x) + sin(x)", "0"),
        "{ -1/2 * pi + k·2 * pi, k·2 * pi, pi + k·2 * pi, 1/2 * pi + k·2 * pi : k ∈ ℤ }"
    );
    // Mixed sin/cos stays with its (correct) owners — not this handler's family.
    assert_eq!(
        solve_display("sin(x)", "cos(x)"),
        "{ 1/4 * pi + k·pi : k ∈ ℤ }"
    );
    // The identity case is the var-eliminated pipeline's.
    assert_eq!(solve_display("sin(x)", "sin(x)"), "All real numbers");
    // Bare periodic solves untouched.
    assert_eq!(
        solve_display("sin(x)", "1/2"),
        "{ 1/6 * pi + k·2 * pi, 5/6 * pi + k·2 * pi : k ∈ ℤ }"
    );
}
