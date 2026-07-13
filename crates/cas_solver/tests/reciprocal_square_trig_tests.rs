//! Contract tests for reciprocal-square trig equations `A/trig(x)^2 = c`
//! (2026-07-13b, family F5 of docs/AUDITORIA_FRONTERA_2026-07-13b.md). `sec(x)^2`,
//! `csc(x)^2`, `1/cos(x)^2`, `1/sin(x)^2` all canonicalize to
//! `Div(A, Pow(cos|sin(g), 2))`, which the bare-squared reducer never matched, so
//! the generic isolation returned only the finite principal-value roots and
//! dropped the periodic family. The reciprocal is now inverted to the equivalent
//! `trig(g)^2 = -A/k` and fed to the existing double-angle reducer. Answers match
//! sympy's `solveset`.

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
    // Normalize the multiplication separator (the harness mixes ` * ` and `·`) so the
    // assertions are render-agnostic.
    display_solution_set(&simplifier.context, &set)
        .replace(" * ", "·")
        .replace('*', "·")
}

#[test]
fn reciprocal_square_trig_emits_the_full_periodic_family() {
    // `1/cos^2 = 2` <=> `cos^2 = 1/2` -> {pi/4 + k*pi/2}.
    assert_eq!(
        solve_display("1/cos(x)^2", "2"),
        "{ 1/4·pi + k·1/2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        solve_display("1/sin(x)^2", "2"),
        "{ 1/4·pi + k·1/2·pi : k ∈ ℤ }"
    );
    // sec^2 / csc^2 canonicalize to the same Div form.
    assert_eq!(
        solve_display("sec(x)^2", "4"),
        "{ 1/3·pi + k·pi, 2/3·pi + k·pi : k ∈ ℤ }"
    );
    assert_eq!(
        solve_display("csc(x)^2", "4"),
        "{ 1/6·pi + k·pi, 5/6·pi + k·pi : k ∈ ℤ }"
    );
    // Constant-shifted form `sec^2 - 2 = 0` handled on the difference.
    assert_eq!(
        solve_display("sec(x)^2 - 2", "0"),
        "{ 1/4·pi + k·1/2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        solve_display("csc(x)^2 - 2", "0"),
        "{ 1/4·pi + k·1/2·pi : k ∈ ℤ }"
    );
}

#[test]
fn direct_and_power_one_trig_forms_are_unchanged() {
    // The direct squared forms this reduces TO are unchanged.
    assert_eq!(
        solve_display("cos(x)^2", "1/2"),
        "{ 1/4·pi + k·1/2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        solve_display("cos(x)^2", "1/4"),
        "{ 1/3·pi + k·pi, 2/3·pi + k·pi : k ∈ ℤ }"
    );
    // Power-1 reciprocal (sec = 2) keeps its own 2*pi-period owner.
    assert_eq!(
        solve_display("sec(x)", "2"),
        "{ 1/3·pi + k·2·pi, 5/3·pi + k·2·pi : k ∈ ℤ }"
    );
    // Boundary sin^2 = 1 single-family case unchanged.
    assert_eq!(solve_display("sin(x)^2", "1"), "{ 1/2·pi + k·pi : k ∈ ℤ }");
}
