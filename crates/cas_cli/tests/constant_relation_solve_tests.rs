//! A relation whose variable fully cancels reduces to a CONSTANT comparison
//! `k (op) c`. The solver must evaluate its actual truth value (false => Empty,
//! true => AllReals) instead of the equation-only "identity => AllReals" default,
//! and must subtract any canceled pole (`x != denom-root`) from the result.

use cas_ast::{Equation, RelOp, SolutionSet};
use cas_parser::parse;
use cas_solver::api::solve;
use cas_solver::runtime::Simplifier;

fn solve_set(input: &str, op: RelOp, rhs_src: &str) -> SolutionSet {
    let mut simplifier = Simplifier::with_default_rules();
    let lhs = parse(input, &mut simplifier.context).unwrap();
    let rhs = parse(rhs_src, &mut simplifier.context).unwrap();
    let (result, _) =
        solve(&Equation { lhs, rhs, op }, "x", &mut simplifier).expect("solve failed");
    result
}

#[test]
fn false_constant_relation_is_empty() {
    // x - x = 0; 0 > 0 is FALSE -> Empty (previously wrongly "All real numbers").
    assert_eq!(solve_set("x - x", RelOp::Gt, "0"), SolutionSet::Empty);
    assert_eq!(solve_set("x - x", RelOp::Lt, "0"), SolutionSet::Empty);
    // 0 > 1 is FALSE.
    assert_eq!(solve_set("x - x", RelOp::Gt, "1"), SolutionSet::Empty);
    // Neq inversion: 0 != 0 is FALSE.
    assert_eq!(solve_set("x - x", RelOp::Neq, "0"), SolutionSet::Empty);
}

#[test]
fn true_constant_relation_is_all_reals() {
    // 0 >= 0 and 0 <= 0 are TRUE; the equality identity is unchanged.
    assert_eq!(solve_set("x - x", RelOp::Geq, "0"), SolutionSet::AllReals);
    assert_eq!(solve_set("x - x", RelOp::Leq, "0"), SolutionSet::AllReals);
    assert_eq!(solve_set("x - x", RelOp::Eq, "0"), SolutionSet::AllReals);
    // 0 < 1 is TRUE.
    assert_eq!(solve_set("x - x", RelOp::Lt, "1"), SolutionSet::AllReals);
}

#[test]
fn canceled_fraction_false_relation_is_empty() {
    // (2x-4)/(x-2) = 2 off the pole; 2 > 2 is FALSE -> Empty.
    assert_eq!(
        solve_set("(2*x - 4)/(x - 2)", RelOp::Gt, "2"),
        SolutionSet::Empty
    );
    // 2 = 1 is FALSE.
    assert_eq!(
        solve_set("(2*x - 4)/(x - 2)", RelOp::Eq, "1"),
        SolutionSet::Empty
    );
    // 1 > 1 is FALSE.
    assert_eq!(
        solve_set("(x - 1)/(x - 1)", RelOp::Gt, "1"),
        SolutionSet::Empty
    );
}

#[test]
fn canceled_fraction_true_relation_excludes_pole() {
    // (2x-4)/(x-2) = 2 off the pole; 2 >= 2 TRUE -> R\{2}, rendered as a guarded
    // Conditional (previously wrongly plain "All real numbers", leaking the pole).
    assert!(matches!(
        solve_set("(2*x - 4)/(x - 2)", RelOp::Geq, "2"),
        SolutionSet::Conditional(_)
    ));
    // The EQUATION path was broken the same way: 2 = 2 off pole -> R\{2}.
    assert!(matches!(
        solve_set("(2*x - 4)/(x - 2)", RelOp::Eq, "2"),
        SolutionSet::Conditional(_)
    ));
    // (x-1)/(x-1) = 1 off the pole; 1 >= 1 TRUE -> R\{1}.
    assert!(matches!(
        solve_set("(x - 1)/(x - 1)", RelOp::Geq, "1"),
        SolutionSet::Conditional(_)
    ));
}

#[test]
fn genuine_solves_unchanged() {
    // Regression guards: ordinary solves (variable NOT eliminated) must be untouched.
    // 1/(x-2) = 1 -> {3}
    assert!(matches!(
        solve_set("1/(x - 2)", RelOp::Eq, "1"),
        SolutionSet::Discrete(roots) if roots.len() == 1
    ));
    // 1/(x-2) > 1 -> (2, 3)
    assert!(matches!(
        solve_set("1/(x - 2)", RelOp::Gt, "1"),
        SolutionSet::Continuous(_)
    ));
    // x^2 - 5x + 6 = 0 -> {2, 3}
    assert!(matches!(
        solve_set("x^2 - 5*x + 6", RelOp::Eq, "0"),
        SolutionSet::Discrete(roots) if roots.len() == 2
    ));
}

#[test]
fn irrational_constant_relations_use_the_exact_sign_oracle() {
    // The variable cancels to an IRRATIONAL constant whose sign is provable by
    // exact rational bounds (pi/e/phi/sqrt + bare-log sign), so the relation is
    // truth-evaluated instead of defaulting to "All real numbers".
    assert_eq!(solve_set("x - x + pi", RelOp::Gt, "4"), SolutionSet::Empty); // pi < 4
    assert_eq!(
        solve_set("x - x + pi", RelOp::Gt, "3"),
        SolutionSet::AllReals
    ); // pi > 3
    assert_eq!(
        solve_set("x - x + 2*pi", RelOp::Lt, "6"),
        SolutionSet::Empty
    ); // 2pi > 6
    assert_eq!(solve_set("x - x + e", RelOp::Lt, "2"), SolutionSet::Empty); // e > 2
    assert_eq!(
        solve_set("x - x + sqrt(2)", RelOp::Gt, "2"),
        SolutionSet::Empty
    ); // sqrt2 < 2
    assert_eq!(
        solve_set("x - x + sqrt(5)", RelOp::Gt, "2"),
        SolutionSet::AllReals
    ); // sqrt5 > 2
    assert_eq!(
        solve_set("x - x + ln(2)", RelOp::Gt, "0"),
        SolutionSet::AllReals
    ); // ln2 > 0
       // The equation arm overrides only for a provably-NONZERO constant.
    assert_eq!(solve_set("x - x + pi", RelOp::Eq, "4"), SolutionSet::Empty); // pi != 4
    assert_eq!(
        solve_set("x - x + pi", RelOp::Neq, "4"),
        SolutionSet::AllReals
    );
}

#[test]
fn undecidable_constant_inequality_is_honest_conditional_not_a_wrong_verdict() {
    // A variable-free constant whose sign the oracle CANNOT prove must NOT default to
    // a definite verdict -- it returns an honest `AllReals if <relation>, else Empty`.
    // After the value-bounds cycles, the remaining undecidable trig cases are a LARGE
    // rational argument (range reduction not yet implemented -> bail) and an IRRATIONAL
    // (interval) argument.
    assert!(matches!(
        solve_set("x - x + sin(200)", RelOp::Gt, "0"),
        SolutionSet::Conditional(_)
    ));
    // sqrt(8) = 2.83 lies in (pi/2, pi), OUTSIDE sin's principal monotone piece, so
    // the interval bound bails (an interval argument inside (-pi/2, pi/2) IS decided --
    // see the sin(sqrt(2)) assertion below).
    assert!(matches!(
        solve_set("x - x + sin(sqrt(8))", RelOp::Gt, "0"),
        SolutionSet::Conditional(_)
    ));
    // Oracle-DECIDABLE constants stay definite verdicts (no over-hedging).
    assert_eq!(solve_set("x - x + pi", RelOp::Gt, "4"), SolutionSet::Empty);
    assert_eq!(
        solve_set("x - x + ln(2)", RelOp::Gt, "0"),
        SolutionSet::AllReals
    );
    // ln/log VALUE comparisons are decided by the exact atanh-series bounds.
    assert_eq!(
        solve_set("x - x + ln(2)", RelOp::Lt, "1"),
        SolutionSet::AllReals
    );
    assert_eq!(
        solve_set("x - x + ln(2)", RelOp::Gt, "1"),
        SolutionSet::Empty
    );
    assert_eq!(
        solve_set("x - x + ln(5)", RelOp::Gt, "1"),
        SolutionSet::AllReals
    );
    assert_eq!(
        solve_set("x - x + log2(3)", RelOp::Lt, "2"),
        SolutionSet::AllReals
    );
    // sin/cos/tan VALUE comparisons (rational argument) are decided by the exact
    // Taylor bounds with Lagrange remainder: sin(1)~0.841, cos(2)~-0.416, tan(1)~1.557.
    assert_eq!(
        solve_set("x - x + sin(1)", RelOp::Gt, "1/2"),
        SolutionSet::AllReals
    );
    assert_eq!(
        solve_set("x - x + sin(1)", RelOp::Gt, "2"),
        SolutionSet::Empty
    );
    assert_eq!(
        solve_set("x - x + cos(2)", RelOp::Lt, "0"),
        SolutionSet::AllReals
    );
    assert_eq!(
        solve_set("x - x + tan(1)", RelOp::Gt, "1"),
        SolutionSet::AllReals
    );
    // An IRRATIONAL argument inside the principal monotone piece is also decided
    // (interval bound via the monotone endpoints): sin(sqrt(2))~0.988, sin(pi/7)~0.434.
    assert_eq!(
        solve_set("x - x + sin(sqrt(2))", RelOp::Gt, "0"),
        SolutionSet::AllReals
    );
    assert_eq!(
        solve_set("x - x + sin(pi/7)", RelOp::Gt, "1/2"),
        SolutionSet::Empty
    );
}

#[test]
fn undecidable_constant_equation_separates_identity_from_false_equality() {
    // A GENUINE but un-foldable identity (`log2(8) = 3`, i.e. `log2(8) - 3 = 0`) must
    // stay a definite AllReals via the exact EqZero prover (`8 = 2^3`).
    assert_eq!(
        solve_set("x - x + log2(8) - 3", RelOp::Eq, "0"),
        SolutionSet::AllReals
    );
    assert_eq!(
        solve_set("x - x + log10(100) - 2", RelOp::Eq, "0"),
        SolutionSet::AllReals
    );
    // A FALSE equality the engine cannot refute (a LARGE-argument trig value, where
    // range reduction is not yet implemented) stays an honest conditional.
    assert!(matches!(
        solve_set("x - x + sin(200)", RelOp::Eq, "1/2"),
        SolutionSet::Conditional(_)
    ));
    // A small-argument trig equality IS now refuted: sin(1) ~ 0.841 != 1/2 is provably
    // nonzero via the exact Taylor bounds, so the equation is a definite "No solution".
    assert_eq!(
        solve_set("x - x + sin(1)", RelOp::Eq, "1/2"),
        SolutionSet::Empty
    );
    // The provably-irrational residual `ln(2) - 1` stays a definite "No solution"
    // (a regression guard: the EqZero work must not hedge a provably-nonzero diff).
    assert_eq!(
        solve_set("x - x + ln(2) - 1", RelOp::Eq, "0"),
        SolutionSet::Empty
    );
    // Provably-nonzero algebraic constant stays Empty.
    assert_eq!(solve_set("x - x + pi", RelOp::Eq, "4"), SolutionSet::Empty);
}
