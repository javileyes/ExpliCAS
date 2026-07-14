//! Contract tests for `cot(g)² = c` (2026-07-14, re-cycle D — F8 Layer-2 of
//! docs/AUDITORIA_FRONTERA_2026-07-13b.md). The sin/cos fold of cot² produced a
//! `cos = |sin|·√c` shape whose principal-branch inversion fabricated a
//! self-referential arccos tree; Layer-1 declined it honestly, and this reducer
//! returns the CORRECT periodic family via the raw-tree reduction
//! `cot²(g) = c ⟺ sin²(g) = 1/(1+c)` — unconditionally sound for c ≥ 0
//! (solutions have sin ≠ 0 automatically, so the cot-pole exclusion is free);
//! c < 0 is exactly Empty. Families verified against sympy.

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
        .replace(" * ", "·")
        .replace('*', "·")
}

#[test]
fn cot_square_emits_the_full_periodic_family() {
    // cot² = 1 ⟺ sin² = 1/2 → {π/4 + kπ/2} (sympy-identical).
    assert_eq!(
        solve_display("cot(x)^2", "1"),
        "{ 1/4·pi + k·1/2·pi : k ∈ ℤ }"
    );
    // Shifted diff form.
    assert_eq!(
        solve_display("cot(x)^2 - 1", "0"),
        "{ 1/4·pi + k·1/2·pi : k ∈ ℤ }"
    );
    // cot² = 3 → cot = ±√3 → {π/6 + kπ, 5π/6 + kπ}.
    assert_eq!(
        solve_display("cot(x)^2", "3"),
        "{ 1/6·pi + k·pi, 5/6·pi + k·pi : k ∈ ℤ }"
    );
    // Boundary c = 0: cos = 0 with sin ≠ 0.
    assert_eq!(solve_display("cot(x)^2", "0"), "{ 1/2·pi + k·pi : k ∈ ℤ }");
    // Negative RHS: a real square cannot be negative.
    assert_eq!(solve_display("cot(x)^2", "-1"), "No solution");
    // Affine argument scales the family.
    assert_eq!(
        solve_display("cot(2*x)^2", "1"),
        "{ 1/8·pi + k·1/4·pi : k ∈ ℤ }"
    );
}

#[test]
fn cot_and_trig_square_owners_are_unchanged() {
    assert_eq!(solve_display("cot(x)", "1"), "{ 1/4·pi + k·pi : k ∈ ℤ }");
    assert_eq!(
        solve_display("sin(x)^2", "1/2"),
        "{ 1/4·pi + k·1/2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        solve_display("sec(x)^2", "4"),
        "{ 1/3·pi + k·pi, 2/3·pi + k·pi : k ∈ ℤ }"
    );
    // Non-constant RHS keeps the honest residual.
    let out = solve_display("cot(x)", "x");
    assert!(out.contains("solve("), "{out}");
}
