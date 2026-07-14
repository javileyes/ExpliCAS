//! Contract tests for `|k·x + b| {<,<=,>,>=} c` with a SYMBOLIC constant center
//! `b` and a numeric threshold `c > 0` (2026-07-14, F7 of
//! docs/AUDITORIA_FRONTERA_2026-07-14.md). The two-sided reduction
//! `|g| {op} c ⟶ g {op} ±c` produces symbolic endpoints `(±c − b)/k`, which the
//! value-blind set-algebra `intersect`/`union` cannot order — collapsing `<`/`<=`
//! to a degenerate two-point union or Empty and `>`/`>=` to AllReals. Since the
//! endpoint order is KNOWN from sign(k), `try_solve_abs_threshold_inequality` now
//! builds the band/rays DIRECTLY. Verified against the numeric solver by
//! substituting a concrete value for the parameter.

use cas_ast::{Equation, RelOp};
use cas_parser::parse;
use cas_solver::api::solve;
use cas_solver::command_api::solve::display_solution_set;
use cas_solver::runtime::Simplifier;

fn solve_display(lhs: &str, op: RelOp, rhs: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let lhs = parse(lhs, &mut simplifier.context).expect("parse lhs");
    let rhs = parse(rhs, &mut simplifier.context).expect("parse rhs");
    let eq = Equation { lhs, rhs, op };
    let (set, _) = solve(&eq, "x", &mut simplifier).expect("solve");
    display_solution_set(&simplifier.context, &set)
        .replace(" * ", "·")
        .replace('*', "·")
}

#[test]
fn abs_symbolic_center_builds_the_band_directly() {
    // |x - a| <= 3  ->  [a-3, a+3]  (was the degenerate {a-3} U {a+3}).
    assert_eq!(solve_display("abs(x-a)", RelOp::Leq, "3"), "[a - 3, a + 3]");
    // Strict interior.
    assert_eq!(solve_display("abs(x-a)", RelOp::Lt, "3"), "(a - 3, a + 3)");
    // Outside union (was AllReals).
    assert_eq!(
        solve_display("abs(x-a)", RelOp::Gt, "3"),
        "(-infinity, a - 3) U (a + 3, infinity)"
    );
    assert_eq!(
        solve_display("abs(x-a)", RelOp::Geq, "3"),
        "(-infinity, a - 3] U [a + 3, infinity)"
    );
}

#[test]
fn abs_symbolic_center_handles_sub_form_and_nonunit_slope() {
    // Center on the LEFT: |a - x| has k = -1; endpoints stay ordered a-3 < a+3.
    assert_eq!(solve_display("abs(a-x)", RelOp::Leq, "3"), "[a - 3, a + 3]");
    // |-x - a| : k = -1, b = -a  ->  [-a-3, 3-a].
    assert_eq!(
        solve_display("abs(-x-a)", RelOp::Leq, "3"),
        "[-a - 3, 3 - a]"
    );
    // Non-unit slope scales the endpoints by 1/k.
    assert_eq!(
        solve_display("abs(2*x-a)", RelOp::Lt, "6"),
        "(1/2·(a - 6), 1/2·(a + 6))"
    );
    assert_eq!(
        solve_display("abs(3*x-a)", RelOp::Geq, "6"),
        "(-infinity, 1/3·(a - 6)] U [1/3·(a + 6), infinity)"
    );
    // Negative non-unit slope: |a - 2x| orders the same band.
    assert_eq!(
        solve_display("abs(a-2*x)", RelOp::Lt, "6"),
        "(1/2·(a - 6), 1/2·(a + 6))"
    );
    // Add-form center (the b cancels naturally): |x + b| < 2 -> (-b-2, 2-b).
    assert_eq!(solve_display("abs(x+b)", RelOp::Lt, "2"), "(-b - 2, 2 - b)");
}

#[test]
fn abs_numeric_center_and_symbolic_threshold_owners_unchanged() {
    // Numeric center keeps its existing (pinned) reduction path.
    assert_eq!(solve_display("abs(x-2)", RelOp::Leq, "3"), "[-1, 5]");
    assert_eq!(solve_display("abs(x)", RelOp::Lt, "3"), "(-3, 3)");
    assert_eq!(
        solve_display("abs(x-2)", RelOp::Gt, "3"),
        "(-infinity, -1) U (5, infinity)"
    );
    assert_eq!(solve_display("abs(2*x-1)", RelOp::Lt, "5"), "(-2, 3)");
    // Symbolic THRESHOLD (not center) still owned by its own handler.
    assert_eq!(
        solve_display("abs(x-1)", RelOp::Lt, "a"),
        "if a > 0: (1 - a, a + 1)"
    );
    // Substituting the parameter must reproduce the numeric band: a = 5.
    assert_eq!(solve_display("abs(x-5)", RelOp::Leq, "3"), "[2, 8]");
    assert_eq!(
        solve_display("abs(x-5)", RelOp::Gt, "3"),
        "(-infinity, 2) U (8, infinity)"
    );
}
