//! Tests del backend local de solve, extraídos del módulo (P4).

use super::{
    intersect_inequality_with_function_domain, required_conditions_are_contradictory,
    root_violates_required_condition, solution_contains_nonfinite,
};
use cas_ast::{Context, Expr};
use cas_solver_core::domain_condition::ImplicitCondition;

#[test]
fn biquadratic_rescue_gates_are_exact() {
    // SOUNDNESS (auditoría 2026-07-30, ficha Q3c-005): the f64 chain
    // (disc sign in f64 → z_f sign → f64 back-substitution → value
    // dedup) FABRICATED `Empty` when cancellation flipped the disc sign:
    // `x⁴ − 2⁵⁰-ish·x² + (N²+3)/4` has disc = −3 exactly, f64 says 0.0,
    // and the old path published «No solution» with 4 existing roots.
    use super::try_solve_biquadratic;
    use crate::Simplifier;
    use cas_ast::{RelOp, SolutionSet};
    use cas_parser::parse;

    let eq_from = |s: &mut Simplifier, lhs: &str| cas_ast::Equation {
        lhs: parse(lhs, &mut s.context).expect("lhs"),
        rhs: parse("0", &mut s.context).expect("rhs"),
        op: RelOp::Eq,
    };

    // The audit reproduction, complex mode: 4 exact roots, never Empty.
    let mut s = Simplifier::with_default_rules();
    let eq = eq_from(
        &mut s,
        "x^4 - 1125899906842625*x^2 + 316912650057057913324129222657",
    );
    match try_solve_biquadratic(&mut s, &eq, "x", false) {
        Some(SolutionSet::Discrete(roots)) => assert_eq!(roots.len(), 4),
        other => panic!("expected 4 exact complex roots, got {other:?}"),
    }
    // Same equation, real-only: disc = −3 < 0 EXACTLY ⇒ Empty is sound
    // (all four roots are complex).
    let eq = eq_from(
        &mut s,
        "x^4 - 1125899906842625*x^2 + 316912650057057913324129222657",
    );
    assert!(matches!(
        try_solve_biquadratic(&mut s, &eq, "x", true),
        Some(SolutionSet::Empty)
    ));

    // Surd roots keep working (the rescue's raison d'être).
    let eq = eq_from(&mut s, "x^4 - 8*x^2 + 15");
    match try_solve_biquadratic(&mut s, &eq, "x", true) {
        Some(SolutionSet::Discrete(roots)) => assert_eq!(roots.len(), 4),
        other => panic!("expected ±√3, ±√5, got {other:?}"),
    }

    // disc = 0 double root: structural dedup keeps exactly {−1, 1}.
    let eq = eq_from(&mut s, "x^4 - 2*x^2 + 1");
    match try_solve_biquadratic(&mut s, &eq, "x", true) {
        Some(SolutionSet::Discrete(roots)) => assert_eq!(roots.len(), 2),
        other => panic!("expected deduped double roots, got {other:?}"),
    }
}

#[test]
fn check_root_is_exact_not_f64() {
    // SOUNDNESS (auditoría 2026-07-30, fichas S2b-001/S3-002/Q3c-001):
    // the old f64 back-substitution (a) DROPPED exact rational roots
    // because its tolerance scaled with the result while the rounding
    // error scales with the coefficients, and (b) CONFIRMED spurious
    // roots whose residual sat under the 1e-9 tolerance.
    use super::{check_root, RootCheck};
    use cas_ast::Equation;
    use cas_parser::parse;
    let mut ctx = Context::new();
    let eq_of = |ctx: &mut Context, l: &str, r: &str| Equation {
        lhs: parse(l, ctx).expect("lhs"),
        rhs: parse(r, ctx).expect("rhs"),
        op: cas_ast::RelOp::Eq,
    };

    // (a) 9-digit-coefficient linear: the exact root must VERIFY (the old
    // gate refuted it — f64 residual ≈ 1e-8 > tol ≈ 1e-9).
    let eq = eq_of(&mut ctx, "(585738843*x - 85131377)*(x-2)", "0");
    let root = parse("85131377/585738843", &mut ctx).expect("root");
    assert_eq!(check_root(&mut ctx, &eq, "x", root), RootCheck::Verified);

    // (a') the 54-bit flip family: (x+2^40)² − 2^80 − 2^41·x = 1/4 at
    // x = 1/2 — unrepresentable in f64, exact in BigRational.
    let eq = eq_of(&mut ctx, "(x+2^40)^2 - 2^80 - 2^41*x", "1/4");
    let half = parse("1/2", &mut ctx).expect("half");
    assert_eq!(check_root(&mut ctx, &eq, "x", half), RootCheck::Verified);

    // (b) spurious root under the old tolerance: sqrt(10^-20) + 10^-10 =
    // 2·10^-10 > 0, PROVABLY nonzero — must be Extraneous.
    let eq = eq_of(&mut ctx, "sqrt(x) + 10^(-10)", "0");
    let tiny = parse("1/100000000000000000000", &mut ctx).expect("tiny");
    assert_eq!(check_root(&mut ctx, &eq, "x", tiny), RootCheck::Extraneous);

    // Motivating case preserved: |1/2| ≠ 1/2 − 1 (needs the new exact
    // Abs interval arm in const_sign).
    let eq = eq_of(&mut ctx, "abs(x)", "x - 1");
    let half = parse("1/2", &mut ctx).expect("half");
    assert_eq!(check_root(&mut ctx, &eq, "x", half), RootCheck::Extraneous);

    // Irrational candidates keep the historical contract: Unknown (kept).
    let eq = eq_of(&mut ctx, "x^2", "2");
    let surd = parse("sqrt(2)", &mut ctx).expect("surd");
    assert_eq!(check_root(&mut ctx, &eq, "x", surd), RootCheck::Unknown);
}

#[test]
fn symbolic_poly_degree_in_var_measures_and_rejects_nonpoly() {
    use super::symbolic_poly_degree_in_var;
    use crate::Simplifier;
    use cas_parser::parse;
    // Measure on the SIMPLIFIED tree, exactly as the leak-recovery hook does (so
    // subtraction is already the canonical `Add(Neg(...))` the degree walker expects).
    let mut s = Simplifier::with_default_rules();
    let deg = |s: &mut Simplifier, src: &str| {
        let e = parse(src, &mut s.context).expect("parse");
        let (e, _) = s.simplify(e);
        symbolic_poly_degree_in_var(&s.context, e, "x")
    };
    // Symbolic-coefficient polynomials: degree is the max power of x.
    assert_eq!(deg(&mut s, "x^3 + p*x + q"), Some(3));
    assert_eq!(deg(&mut s, "x^3 + a*x^2 + b*x + c"), Some(3));
    assert_eq!(deg(&mut s, "x^5 + p*x + q"), Some(5));
    assert_eq!(deg(&mut s, "x^3 - p*x^2"), Some(3));
    assert_eq!(deg(&mut s, "x^2 + p*x + q"), Some(2));
    assert_eq!(deg(&mut s, "p*x + q"), Some(1));
    // Non-polynomial occurrences of x -> None, so the guard never fires on these.
    assert_eq!(deg(&mut s, "x^2 - sqrt(x)"), None); // fractional power
    assert_eq!(deg(&mut s, "x^2 - 2^x"), None); // x in the exponent
    assert_eq!(deg(&mut s, "x + 1/x"), None); // x in a denominator
    assert_eq!(deg(&mut s, "x + sin(x)"), None); // x inside a function
}

#[test]
fn trig_unit_rhs_classifies_named_transcendental_constants_via_bounds() {
    use super::{classify_trig_unit_rhs, TrigUnitClass};
    use cas_ast::Constant;
    let mut ctx = Context::new();
    let phi = ctx.add(Expr::Constant(Constant::Phi));
    let e = ctx.add(Expr::Constant(Constant::E));
    let pi = ctx.add(Expr::Constant(Constant::Pi));
    let one = ctx.num(1);
    let two = ctx.num(2);
    let four = ctx.num(4);
    let neg_phi = ctx.add(Expr::Neg(phi));
    let inv_e = ctx.add(Expr::Div(one, e));
    let pi_4 = ctx.add(Expr::Div(pi, four));
    let e_minus_2 = ctx.add(Expr::Sub(e, two));
    // Strictly outside [-1, 1]: phi (the (1+sqrt(5))/2 fold), e, -phi.
    for c in [phi, e, neg_phi] {
        assert!(
            matches!(
                classify_trig_unit_rhs(&ctx, c),
                Some(TrigUnitClass::OutOfRange)
            ),
            "expected OutOfRange"
        );
    }
    // Strictly inside (-1, 1) with proven sign: 1/e, pi/4, e - 2.
    for c in [inv_e, pi_4, e_minus_2] {
        assert!(
            matches!(classify_trig_unit_rhs(&ctx, c), Some(TrigUnitClass::InOpen)),
            "expected InOpen"
        );
    }
    // Symbolic RHS stays undecided (owned by the honest decline path).
    let y = ctx.var("y");
    assert!(classify_trig_unit_rhs(&ctx, y).is_none());
}

#[test]
fn trig_threshold_classification_and_bare_detection() {
    use super::{bare_sin_or_cos_of_var, classify_trig_threshold, TrigThresholdRegion};
    use num_rational::BigRational;
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let sin_x = ctx.call("sin", vec![x]);
    let cos_x = ctx.call("cos", vec![x]);
    assert!(bare_sin_or_cos_of_var(&ctx, sin_x, "x"));
    assert!(bare_sin_or_cos_of_var(&ctx, cos_x, "x"));
    // Compound argument and non-sin/cos are rejected (owned by the periodic residual path).
    let two = ctx.num(2);
    let two_x = ctx.add(Expr::Mul(two, x));
    let sin_2x = ctx.call("sin", vec![two_x]);
    assert!(!bare_sin_or_cos_of_var(&ctx, sin_2x, "x"));
    let tan_x = ctx.call("tan", vec![x]);
    assert!(!bare_sin_or_cos_of_var(&ctx, tan_x, "x"));
    // Threshold regions vs the [-1, 1] range, decided exactly.
    let c2 = ctx.num(2);
    let c1 = ctx.num(1);
    let cm1 = ctx.num(-1);
    let chalf = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
    assert!(matches!(
        classify_trig_threshold(&ctx, c2),
        Some(TrigThresholdRegion::AboveRange)
    ));
    assert!(matches!(
        classify_trig_threshold(&ctx, c1),
        Some(TrigThresholdRegion::AtUpperBound)
    ));
    assert!(matches!(
        classify_trig_threshold(&ctx, cm1),
        Some(TrigThresholdRegion::AtLowerBound)
    ));
    assert!(classify_trig_threshold(&ctx, chalf).is_none()); // in range
}

#[test]
fn contains_trig_of_var_detects_periodic_argument() {
    use super::contains_trig_of_var;
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let sin_x = ctx.call("sin", vec![x]);
    assert!(contains_trig_of_var(&ctx, sin_x, "x"));
    // Nested in a sum / product still detected.
    let one = ctx.num(1);
    let x_plus_sin = ctx.add(Expr::Add(x, sin_x));
    assert!(contains_trig_of_var(&ctx, x_plus_sin, "x"));
    let two = ctx.num(2);
    let two_x = ctx.add(Expr::Mul(two, x));
    let sin_2x = ctx.call("sin", vec![two_x]);
    assert!(contains_trig_of_var(&ctx, sin_2x, "x")); // compound argument
                                                      // Constant trig (`sin(2)·x`) does NOT contain the variable inside the trig argument.
    let sin_2 = ctx.call("sin", vec![two]);
    let sin2_times_x = ctx.add(Expr::Mul(sin_2, x));
    assert!(!contains_trig_of_var(&ctx, sin2_times_x, "x"));
    // A non-trig expression is false.
    let lin = ctx.add(Expr::Add(x, one));
    assert!(!contains_trig_of_var(&ctx, lin, "x"));
}

#[test]
fn cmp_rational_to_quadratic_surd_is_exact() {
    use num_rational::BigRational;
    use std::cmp::Ordering;
    let r = |n: i64, d: i64| BigRational::new(n.into(), d.into());
    // φ = ½ + ½·√5 ≈ 1.618 — the worst case for a float gate near the boundary.
    let (a, b, n) = (r(1, 2), r(1, 2), BigRational::from_integer(5.into()));
    assert_eq!(
        cas_math::root_forms::cmp_rational_to_quadratic_surd(&r(8, 5), &a, &b, &n),
        Ordering::Less // 1.6 < φ
    );
    assert_eq!(
        cas_math::root_forms::cmp_rational_to_quadratic_surd(&r(13, 8), &a, &b, &n),
        Ordering::Greater // 1.625 > φ
    );
    // 2√5 ≈ 4.472 (a = 0, b = 2, n = 5): exact ordering of nearby rationals.
    let (a0, b2) = (
        BigRational::from_integer(0.into()),
        BigRational::from_integer(2.into()),
    );
    assert_eq!(
        cas_math::root_forms::cmp_rational_to_quadratic_surd(&r(9, 2), &a0, &b2, &n),
        Ordering::Greater // 4.5 > 2√5
    );
    assert_eq!(
        cas_math::root_forms::cmp_rational_to_quadratic_surd(&r(4, 1), &a0, &b2, &n),
        Ordering::Less // 4 < 2√5
    );
    // Negative coefficient `1 − √2` ≈ −0.414 (a = 1, b = −1, n = 2): sign handled.
    let (a1, bn1, n2) = (
        BigRational::from_integer(1.into()),
        BigRational::from_integer((-1).into()),
        BigRational::from_integer(2.into()),
    );
    assert_eq!(
        cas_math::root_forms::cmp_rational_to_quadratic_surd(&r(-1, 2), &a1, &bn1, &n2),
        Ordering::Less // −0.5 < 1 − √2
    );
    assert_eq!(
        cas_math::root_forms::cmp_rational_to_quadratic_surd(&r(-2, 5), &a1, &bn1, &n2),
        Ordering::Greater // −0.4 > 1 − √2
    );
    // Degenerate (b = 0): a plain rational comparison.
    let z = BigRational::from_integer(0.into());
    assert_eq!(
        cas_math::root_forms::cmp_rational_to_quadratic_surd(&r(3, 1), &r(3, 1), &z, &z),
        Ordering::Equal
    );
}

#[test]
fn monotonic_inequality_intersects_argument_domain() {
    use cas_ast::{BoundType, Equation, ExprId, RelOp, SolutionSet};
    use cas_solver_core::solution_set::{neg_inf, pos_inf};
    use num_traits::Zero;

    let mut simp = crate::Simplifier::with_default_rules();
    let p = |simp: &mut crate::Simplifier, s: &str| -> ExprId {
        cas_parser::parse(s, &mut simp.context).expect("parse")
    };
    // Build the engine's naive half-line: (-inf, bound) for `<`, (bound, inf) for `>`.
    let half_line = |simp: &mut crate::Simplifier, bound: ExprId, lt: bool| -> SolutionSet {
        let (min, min_t, max, max_t) = if lt {
            (
                neg_inf(&mut simp.context),
                BoundType::Open,
                bound,
                BoundType::Open,
            )
        } else {
            (
                bound,
                BoundType::Open,
                pos_inf(&mut simp.context),
                BoundType::Open,
            )
        };
        SolutionSet::Continuous(cas_ast::Interval {
            min,
            min_type: min_t,
            max,
            max_type: max_t,
        })
    };
    let eqn = |lhs: ExprId, rhs: ExprId, op: RelOp| Equation { lhs, rhs, op };

    // sqrt(x) < 2 : naive (-inf, 2^2) intersected with [0, inf) => [0, 4).
    let (sx, two, b4) = (
        p(&mut simp, "sqrt(x)"),
        p(&mut simp, "2"),
        p(&mut simp, "2^2"),
    );
    let set = half_line(&mut simp, b4, true);
    match intersect_inequality_with_function_domain(&mut simp, &eqn(sx, two, RelOp::Lt), "x", set) {
        SolutionSet::Continuous(i) => {
            assert_eq!(i.min_type, BoundType::Closed, "lower bound closed at 0");
            assert!(
                matches!(simp.context.get(i.min), Expr::Number(n) if n.is_zero()),
                "lower bound is 0"
            );
        }
        other => panic!("expected [0, 4), got {other:?}"),
    }

    // sqrt(x) < -1 : even-root range disjoint => No solution.
    let (sx2, neg1, b1) = (
        p(&mut simp, "sqrt(x)"),
        p(&mut simp, "-1"),
        p(&mut simp, "(-1)^2"),
    );
    let set = half_line(&mut simp, b1, true);
    assert!(matches!(
        intersect_inequality_with_function_domain(&mut simp, &eqn(sx2, neg1, RelOp::Lt), "x", set),
        SolutionSet::Empty
    ));

    // sqrt(x) > -1 : always true on the domain => [0, inf).
    let (sx3, neg1b, b1b) = (
        p(&mut simp, "sqrt(x)"),
        p(&mut simp, "-1"),
        p(&mut simp, "(-1)^2"),
    );
    let set = half_line(&mut simp, b1b, false);
    match intersect_inequality_with_function_domain(
        &mut simp,
        &eqn(sx3, neg1b, RelOp::Gt),
        "x",
        set,
    ) {
        SolutionSet::Continuous(i) => {
            assert_eq!(i.min_type, BoundType::Closed);
            assert!(matches!(simp.context.get(i.min), Expr::Number(n) if n.is_zero()));
        }
        other => panic!("expected [0, inf), got {other:?}"),
    }

    // POSITIVE multiplicative coefficient is seen through: 2·sqrt(x) < 6 has
    // the same [0, ∞) range, so the naive (-inf, 9) folds to [0, 9).
    let (csx, six, b9) = (
        p(&mut simp, "2*sqrt(x)"),
        p(&mut simp, "6"),
        p(&mut simp, "9"),
    );
    let set = half_line(&mut simp, b9, true);
    match intersect_inequality_with_function_domain(&mut simp, &eqn(csx, six, RelOp::Lt), "x", set)
    {
        SolutionSet::Continuous(i) => {
            assert_eq!(
                i.min_type,
                BoundType::Closed,
                "2·sqrt(x)<6 lower bound closed at 0"
            );
            assert!(matches!(simp.context.get(i.min), Expr::Number(n) if n.is_zero()));
        }
        other => panic!("expected [0, 9), got {other:?}"),
    }

    // EQUATION path is untouched: sqrt(x) = 2 keeps its Discrete result.
    let (sx4, two4) = (p(&mut simp, "sqrt(x)"), p(&mut simp, "2"));
    let disc = SolutionSet::Discrete(vec![p(&mut simp, "4")]);
    assert!(matches!(
        intersect_inequality_with_function_domain(&mut simp, &eqn(sx4, two4, RelOp::Eq), "x", disc),
        SolutionSet::Discrete(_)
    ));
}

#[test]
fn extraneous_root_violates_recorded_domain_condition() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    // ln(x)+ln(x+5)=0 records x > 0. The extraneous root ½(-√29-5) ≈ -5.19
    // violates it; the valid root ½(√29-5) ≈ 0.19 does not.
    let positive_x = vec![ImplicitCondition::Positive(x)];
    let ext = cas_parser::parse("1/2*(-sqrt(29) - 5)", &mut ctx).expect("ext");
    let valid = cas_parser::parse("1/2*(sqrt(29) - 5)", &mut ctx).expect("valid");
    assert!(root_violates_required_condition(
        &mut ctx,
        "x",
        ext,
        &positive_x
    ));
    assert!(!root_violates_required_condition(
        &mut ctx,
        "x",
        valid,
        &positive_x
    ));
    // No recorded conditions => never a violation (byte-identical to before).
    assert!(!root_violates_required_condition(&mut ctx, "x", ext, &[]));

    // sqrt(x)=0 => x=0 sits on the NonNegative boundary and must be KEPT.
    let nonneg_x = vec![ImplicitCondition::NonNegative(x)];
    let zero = ctx.num(0);
    assert!(!root_violates_required_condition(
        &mut ctx, "x", zero, &nonneg_x
    ));

    // Reciprocal-surd negative branch -13·13^(-1/2) = -√13 violates x-2 ≥ 0.
    let xm2 = cas_parser::parse("x - 2", &mut ctx).expect("xm2");
    let nonneg_xm2 = vec![ImplicitCondition::NonNegative(xm2)];
    let neg_sqrt13 = cas_parser::parse("-13*13^(-1/2)", &mut ctx).expect("neg13");
    assert!(root_violates_required_condition(
        &mut ctx,
        "x",
        neg_sqrt13,
        &nonneg_xm2
    ));

    // The adversarial convergent: NonZero(93222358·x - 131836323) at √2 must
    // NOT fire (the value is irrational, provably nonzero) — exact arithmetic
    // keeps the valid root where a float gate would drop it.
    let denom = cas_parser::parse("93222358*x - 131836323", &mut ctx).expect("denom");
    let nonzero = vec![ImplicitCondition::NonZero(denom)];
    let root2 = cas_parser::parse("sqrt(2)", &mut ctx).expect("sqrt2");
    assert!(!root_violates_required_condition(
        &mut ctx, "x", root2, &nonzero
    ));
}

#[test]
fn contradictory_positive_conditions_empty_the_real_domain() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let neg_x = cas_parser::parse("-x", &mut ctx).expect("neg x");
    let xp5 = cas_parser::parse("x + 5", &mut ctx).expect("x+5");

    // `x > 0` AND `-x > 0` is impossible — this is the `ln(x)=ln(-x)`
    // collapse (an `AllReals` carrying both must become Empty).
    assert!(required_conditions_are_contradictory(
        &ctx,
        &[
            ImplicitCondition::Positive(x),
            ImplicitCondition::Positive(neg_x),
        ]
    ));
    // `x > 0` AND `x+5 > 0` is satisfiable.
    assert!(!required_conditions_are_contradictory(
        &ctx,
        &[
            ImplicitCondition::Positive(x),
            ImplicitCondition::Positive(xp5),
        ]
    ));
    // NonNegative pair (`x >= 0` AND `-x >= 0`) meets at 0 — NOT contradictory
    // (the check must be strict `> 0`, not `>= 0`).
    assert!(!required_conditions_are_contradictory(
        &ctx,
        &[
            ImplicitCondition::NonNegative(x),
            ImplicitCondition::NonNegative(neg_x),
        ]
    ));
    // `-8x > 0` AND `x > 0` is impossible — the generalized negative-multiple
    // case (`log(2,-8x)=log(2,x)+k`), not just exact negation.
    let neg_8x = cas_parser::parse("-8*x", &mut ctx).expect("-8x");
    assert!(required_conditions_are_contradictory(
        &ctx,
        &[
            ImplicitCondition::Positive(neg_8x),
            ImplicitCondition::Positive(x),
        ]
    ));
    // `2x > 0` AND `x > 0` is a POSITIVE multiple — satisfiable, NOT collapsed.
    let two_x = cas_parser::parse("2*x", &mut ctx).expect("2x");
    assert!(!required_conditions_are_contradictory(
        &ctx,
        &[
            ImplicitCondition::Positive(two_x),
            ImplicitCondition::Positive(x),
        ]
    ));
    // A single condition (or none) is never contradictory.
    assert!(!required_conditions_are_contradictory(
        &ctx,
        &[ImplicitCondition::Positive(x)]
    ));
    assert!(!required_conditions_are_contradictory(&ctx, &[]));
}

#[test]
fn out_of_range_inverse_trig_root_is_not_real() {
    let mut ctx = Context::new();
    let two = ctx.num(2);
    let neg_three = ctx.num(-3);
    let half = ctx.add(Expr::Number(num_rational::BigRational::new(
        1.into(),
        2.into(),
    )));
    let one = ctx.num(1);
    let neg_one = ctx.num(-1);

    // arcsin/arccos of |c| > 1 are undefined over ℝ — not real solutions.
    let arcsin_two = ctx.call("arcsin", vec![two]);
    let arccos_two = ctx.call("arccos", vec![two]);
    let arcsin_neg_three = ctx.call("arcsin", vec![neg_three]);
    assert!(solution_contains_nonfinite(&ctx, arcsin_two));
    assert!(solution_contains_nonfinite(&ctx, arccos_two));
    assert!(solution_contains_nonfinite(&ctx, arcsin_neg_three));

    // A root that merely CONTAINS an out-of-range inverse-trig term is also
    // non-real (e.g. `arcsin(2) + 1`).
    let arcsin_two2 = ctx.call("arcsin", vec![two]);
    let shifted = ctx.add(Expr::Add(arcsin_two2, one));
    assert!(solution_contains_nonfinite(&ctx, shifted));

    // In-range / boundary arguments are genuine real values — kept.
    let arcsin_half = ctx.call("arcsin", vec![half]);
    let arccos_one = ctx.call("arccos", vec![one]);
    let arcsin_neg_one = ctx.call("arcsin", vec![neg_one]);
    assert!(!solution_contains_nonfinite(&ctx, arcsin_half));
    assert!(!solution_contains_nonfinite(&ctx, arccos_one));
    assert!(!solution_contains_nonfinite(&ctx, arcsin_neg_one));

    // Round-4 Cluster C: a SURD argument outside [-1, 1] (`√2 ≈ 1.41`, `√3`,
    // `2√2`) is non-real and must be dropped — EXACTLY, via the quadratic-surd
    // sign logic (a float gate could drop a valid root). In-range surds
    // (`√2/2 ≈ 0.71`, `√3/2`) are kept.
    for src in ["sqrt(2)", "sqrt(3)", "2*sqrt(2)", "sqrt(8)", "-sqrt(2)"] {
        let arg = cas_parser::parse(src, &mut ctx).expect("surd");
        let call = ctx.call("arcsin", vec![arg]);
        assert!(
            solution_contains_nonfinite(&ctx, call),
            "arcsin({src}) is non-real (|arg| > 1)"
        );
    }
    for src in ["sqrt(2)/2", "sqrt(3)/2", "1/2", "-sqrt(2)/2"] {
        let arg = cas_parser::parse(src, &mut ctx).expect("surd");
        let call = ctx.call("arccos", vec![arg]);
        assert!(
            !solution_contains_nonfinite(&ctx, call),
            "arccos({src}) is in range (|arg| <= 1), must be KEPT"
        );
    }
}

#[test]
fn identically_zero_denominator_makes_equation_unsolvable() {
    use super::equation_has_identically_zero_denominator;
    use cas_ast::{Equation, RelOp};

    let mut simplifier = crate::Simplifier::with_default_rules();
    let build = |simp: &mut crate::Simplifier, lhs: &str, rhs: &str| -> Equation {
        let l = cas_parser::parse(lhs, &mut simp.context).expect("lhs");
        let r = cas_parser::parse(rhs, &mut simp.context).expect("rhs");
        Equation {
            lhs: l,
            rhs: r,
            op: RelOp::Eq,
        }
    };

    // Provably-zero denominators (literal 0, x-x, 0*x) => identically
    // undefined over ℝ => caught, so the solver returns "No solution".
    for (l, r) in [
        ("x/0", "5"),
        ("x", "1/0"),
        ("x/(x-x)", "0"),
        ("1/(0*x)", "2"),
    ] {
        let eq = build(&mut simplifier, l, r);
        assert!(
            equation_has_identically_zero_denominator(&mut simplifier, &eq),
            "{l} = {r} has an identically-zero denominator"
        );
    }

    // Denominators that are nonzero or merely vanish at isolated points are
    // legitimate (excluded points, not an undefined equation) => NOT caught.
    for (l, r) in [
        ("3/x", "0"),
        ("1/x", "2"),
        ("x/(x-1)", "2"),
        ("1/(x-x+1)", "1"),
    ] {
        let eq = build(&mut simplifier, l, r);
        assert!(
            !equation_has_identically_zero_denominator(&mut simplifier, &eq),
            "{l} = {r} denominator is nonzero or only vanishes at a point"
        );
    }
}
