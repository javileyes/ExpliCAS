//! Contract tests for `sin/cos(affine) = parameter` (2026-07-11). The unit-RHS
//! classifier cannot place a free symbol against {-1, 0, 1}, and declining fell
//! through to the principal-only inversion: solve(sin(x) = a) returned the bare
//! {arcsin(a)} - no supplementary branch, no +2k*pi family, no range gate.
//! Family F3 of docs/AUDITORIA_FRONTERA_2026-07-09.md. The two-family InOpen set
//! is correct for EVERY -1 <= a <= 1 (the families coincide or interlace at the
//! endpoints), so it is emitted under the closed-range guard.

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
fn parametric_sin_cos_emit_the_guarded_two_family_set() {
    let sin_form = solve_display("sin(x)", "a");
    assert!(
        sin_form.contains("arcsin(a) + k·2")
            && sin_form.contains("pi - arcsin(a)")
            && sin_form.contains("a + 1 >= 0")
            && sin_form.contains("1 - a >= 0"),
        "sin(x)=a must carry both families and the closed-range guard, got {sin_form}"
    );
    let cos_form = solve_display("cos(x)", "a");
    assert!(
        cos_form.contains("arccos(a) + k·2") && cos_form.contains("pi - arccos(a)"),
        "cos(x)=a must carry both families, got {cos_form}"
    );
    // Scaled argument: the family maps back through the slope (period pi).
    let scaled = solve_display("sin(2*x)", "a");
    assert!(
        scaled.contains("arcsin(a)") && scaled.contains("+ k·pi"),
        "sin(2x)=a must halve the family, got {scaled}"
    );
    // An outer coefficient folds into the threshold AND the guard.
    let coef = solve_display("2*sin(x)", "a");
    assert!(
        coef.contains("arcsin(a / 2)"),
        "2 sin(x)=a must fold the coefficient, got {coef}"
    );
}

#[test]
fn decidable_thresholds_keep_their_owners() {
    // tan accepts every constant: the parametric family was already correct.
    assert_eq!(solve_display("tan(x)", "a"), "{ arctan(a) + k·pi : k ∈ ℤ }");
    // Rational, unit, zero, and provably-out-of-range constants: unchanged.
    let half = solve_display("sin(x)", "1/2");
    assert!(
        half.contains("1/6 * pi") || half.contains("1/6·pi"),
        "got {half}"
    );
    assert_eq!(solve_display("sin(x)", "e"), "No solution");
    let unit = solve_display("sin(x)", "1");
    assert!(unit.contains("1/2"), "got {unit}");
}
