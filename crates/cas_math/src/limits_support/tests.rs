//! Tests de `limits_support`, extraídos del módulo (P3).

use super::*;
use cas_formatter::DisplayExpr;
use cas_parser::parse;

fn parse_expr(ctx: &mut Context, s: &str) -> ExprId {
    parse(s, ctx).expect("parse failed")
}

#[test]
fn combine_difference_over_common_denominator_builds_single_fraction() {
    use std::collections::HashMap;
    // 1/x - 1/y -> (y - x)/(x*y); check the combined fraction's value at x=2, y=3 is 1/6.
    let mut ctx = Context::new();
    let lhs = parse_expr(&mut ctx, "1/x");
    let rhs = parse_expr(&mut ctx, "1/y");
    let combined = combine_difference_over_common_denominator(&mut ctx, lhs, rhs)
        .expect("two fractions combine");
    assert!(
        matches!(ctx.get(combined), Expr::Div(_, _)),
        "combined is a single fraction"
    );
    let mut map = HashMap::new();
    map.insert("x".to_string(), 2.0);
    map.insert("y".to_string(), 3.0);
    let v = crate::evaluator_f64::eval_f64(&ctx, combined, &map).expect("foldable");
    assert!((v - 1.0 / 6.0).abs() < 1e-12, "1/2 - 1/3 = 1/6, got {v}");

    // Neither side is a fraction -> declines (no new structure to gain).
    let a = parse_expr(&mut ctx, "x^2");
    let b = parse_expr(&mut ctx, "x");
    assert!(combine_difference_over_common_denominator(&mut ctx, a, b).is_none());
}

#[test]
fn as_fraction_extracts_reciprocal_trig_forms() {
    use std::collections::HashMap;
    // csc(x)^2 -> (1, sin(x)^2); cot(x) -> (cos(x), sin(x)); (cos/sin)^2 -> ((cos)^2, (sin)^2).
    // Verify num/den numerically equals the original at x = 0.5.
    let mut map = HashMap::new();
    map.insert("x".to_string(), 0.5);
    for (src, expect) in [
        ("csc(x)^2", 1.0 / (0.5_f64.sin().powi(2))),
        ("cot(x)", 0.5_f64.cos() / 0.5_f64.sin()),
        ("(cos(x)/sin(x))^2", (0.5_f64.cos() / 0.5_f64.sin()).powi(2)),
    ] {
        let mut ctx = Context::new();
        let e = parse_expr(&mut ctx, src);
        let (num, den) = as_fraction(&mut ctx, e)
            .unwrap_or_else(|| panic!("{src} should decompose as a fraction"));
        let ratio = ctx.add(Expr::Div(num, den));
        let v = crate::evaluator_f64::eval_f64(&ctx, ratio, &map).expect("foldable");
        assert!(
            (v - expect).abs() < 1e-9,
            "{src}: num/den = {v}, expected {expect}"
        );
    }
    // A non-reciprocal-trig function is NOT a fraction.
    let mut ctx = Context::new();
    let s = parse_expr(&mut ctx, "sin(x)");
    assert!(as_fraction(&mut ctx, s).is_none());
}

#[test]
fn finite_sub_result_declines_same_sign_infinity_difference() {
    // (+inf) - (+inf) is INDETERMINATE: must NOT collapse to 0 (the `lhs == rhs` shortcut on
    // equal interned infinities was the `lim 1/sin^2 x - 1/x^2 = 0` wrong-answer).
    let mut ctx = Context::new();
    let pos_inf = ctx.add(Expr::Constant(Constant::Infinity));
    assert_eq!(
        finite_sub_result(&mut ctx, pos_inf, pos_inf),
        None,
        "(+inf) - (+inf) must decline, not return 0"
    );
    // -inf - -inf also indeterminate.
    let inf_a = ctx.add(Expr::Constant(Constant::Infinity));
    let neg_inf = ctx.add(Expr::Neg(inf_a));
    let inf_b = ctx.add(Expr::Constant(Constant::Infinity));
    let neg_inf_b = ctx.add(Expr::Neg(inf_b));
    assert_eq!(
        finite_sub_result(&mut ctx, neg_inf, neg_inf_b),
        None,
        "(-inf) - (-inf) must decline"
    );
    // DETERMINATE cases must still resolve (not decline):
    //   +inf - (-inf) = +inf, +inf - finite = +inf, finite - finite = value.
    assert!(
        finite_sub_result(&mut ctx, pos_inf, neg_inf).is_some(),
        "(+inf) - (-inf) is determinate (+inf), must resolve"
    );
    let five = ctx.num(5);
    assert!(
        finite_sub_result(&mut ctx, pos_inf, five).is_some(),
        "(+inf) - 5 is determinate, must resolve"
    );
    let seven = ctx.num(7);
    let three = ctx.num(3);
    let diff = finite_sub_result(&mut ctx, seven, three).expect("7 - 3 resolves");
    assert!(
        matches!(ctx.get(diff), Expr::Number(n) if *n == num_rational::BigRational::from_integer(4.into()))
    );
}

fn display_expr(ctx: &Context, expr: ExprId) -> String {
    DisplayExpr {
        context: ctx,
        id: expr,
    }
    .to_string()
}

fn assert_rational_taylor(src: &str, order: usize, expected: &[(i64, i64)]) {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, src);
    let poly = taylor_at_zero_with_rational(&ctx, expr, "x", order)
        .unwrap_or_else(|| panic!("{src} should expand"));
    for (k, (num, den)) in expected.iter().enumerate() {
        let coeff = poly
            .coeffs
            .get(k)
            .cloned()
            .unwrap_or_else(|| BigRational::new(0.into(), 1.into()));
        assert_eq!(
            coeff,
            BigRational::new((*num).into(), (*den).into()),
            "{src}: coefficient of x^{k}"
        );
    }
}

#[test]
fn rational_taylor_matches_known_geometric_series() {
    assert_rational_taylor("1/(1-x)", 4, &[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]);
    assert_rational_taylor("1/(1+x)", 4, &[(1, 1), (-1, 1), (1, 1), (-1, 1), (1, 1)]);
    assert_rational_taylor("1/(1+x^2)", 4, &[(1, 1), (0, 1), (-1, 1), (0, 1), (1, 1)]);
    assert_rational_taylor("1/(2-x)", 3, &[(1, 2), (1, 4), (1, 8), (1, 16)]);
    assert_rational_taylor("1/(1-x)^2", 3, &[(1, 1), (2, 1), (3, 1), (4, 1)]);
    assert_rational_taylor("x/(1-x)", 4, &[(0, 1), (1, 1), (1, 1), (1, 1), (1, 1)]);
}

#[test]
fn rational_taylor_declines_pole_at_zero() {
    // 1/x and 1/(x - 1 + 1) style poles at 0 have no Maclaurin expansion.
    for src in ["1/x", "1/x^2", "(1+x)/x"] {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, src);
        assert!(
            taylor_at_zero_with_rational(&ctx, expr, "x", 4).is_none(),
            "{src} has a pole at 0 and must not expand"
        );
    }
}

fn assert_number_expr(ctx: &Context, expr: ExprId, numerator: i64, denominator: i64) {
    let Expr::Number(value) = ctx.get(expr) else {
        panic!("expected exact rational expression");
    };
    assert_eq!(
        value,
        &BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
    );
}

fn assert_ratio_over_ln_base(
    ctx: &Context,
    expr: ExprId,
    numerator: i64,
    denominator: i64,
    base_numerator: i64,
    base_denominator: i64,
) {
    let Expr::Div(num_expr, den_expr) = ctx.get(expr).clone() else {
        panic!(
            "expected quotient over ln(base), got {}",
            display_expr(ctx, expr)
        );
    };
    assert_number_expr(ctx, num_expr, numerator, denominator);

    let Expr::Function(fn_id, args) = ctx.get(den_expr).clone() else {
        panic!(
            "expected ln(base) denominator, got {}",
            display_expr(ctx, den_expr)
        );
    };
    assert!(ctx.is_builtin(fn_id, BuiltinFn::Ln));
    assert_eq!(args.len(), 1);
    assert_number_expr(ctx, args[0], base_numerator, base_denominator);
}

#[test]
fn depends_on_detects_simple_variable() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "x + 1");
    let x = parse_expr(&mut ctx, "x");
    assert!(depends_on(&ctx, expr, x));
}

#[test]
fn depends_on_rejects_constant_expression() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "5 + pi");
    let x = parse_expr(&mut ctx, "x");
    assert!(!depends_on(&ctx, expr, x));
}

#[test]
fn parse_pow_int_extracts_integer_exponent() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "x^3");
    let (_, n) = parse_pow_int(&ctx, expr).expect("power");
    assert_eq!(n, 3);
}

#[test]
fn limit_sign_handles_neg_infinity_parity() {
    assert_eq!(limit_sign(InfSign::Pos, 7), InfSign::Pos);
    assert_eq!(limit_sign(InfSign::Neg, 2), InfSign::Pos);
    assert_eq!(limit_sign(InfSign::Neg, 3), InfSign::Neg);
}

#[test]
fn mk_limit_builds_limit_call_with_signed_infinity_symbol() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "x^2");
    let var = parse_expr(&mut ctx, "x");
    let lim = mk_limit(&mut ctx, expr, var, InfSign::Neg);

    let Expr::Function(_fn_id, args) = ctx.get(lim) else {
        panic!("expected limit function call");
    };
    assert_eq!(args.len(), 3);
    assert_eq!(args[0], expr);
    assert_eq!(args[1], var);

    let approach = args[2];
    match ctx.get(approach) {
        Expr::Neg(inner) => {
            assert!(matches!(
                ctx.get(*inner),
                Expr::Constant(Constant::Infinity)
            ));
        }
        _ => panic!("expected negative infinity argument"),
    }
}

#[test]
fn one_sided_composition_saturates_inner_infinity() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    for (source, side, expected) in [
        ("e^(1/x)", FiniteLimitSide::Right, "infinity"),
        ("e^(1/x)", FiniteLimitSide::Left, "0"),
        ("atan(1/x)", FiniteLimitSide::Right, "pi / 2"),
        ("atan(1/x)", FiniteLimitSide::Left, "-pi / 2"),
        ("tanh(1/x)", FiniteLimitSide::Right, "1"),
        ("e^(-1/x)", FiniteLimitSide::Right, "0"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = try_limit_rules_at_finite_one_sided(&mut ctx, expr, x, zero, side)
            .unwrap_or_else(|| panic!("must resolve: {source} {side:?}"));
        // The one-sided rule returns f(+-inf); the eval layer folds it,
        // but fold_infinity_saturation already runs inside the helper.
        assert_eq!(display_expr(&ctx, out), expected, "{source} {side:?}");
    }
}

#[test]
fn one_sided_product_of_zero_and_finite_collapses_to_zero() {
    // e^(1/x) -> 0 and atan(1/x) -> -pi/2 from the left, so their product
    // is 0 * (-pi/2) = 0. combine_limit_product must fold the zero factor
    // instead of emitting the un-normalized product `-0 * pi/2`.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    let expr = parse_expr(&mut ctx, "e^(1/x) * atan(1/x)");
    let out = try_limit_rules_at_finite_one_sided(&mut ctx, expr, x, zero, FiniteLimitSide::Left)
        .expect("0 * finite must resolve");
    assert_eq!(display_expr(&ctx, out), "0");
}

#[test]
fn one_sided_composition_declines_oscillating_outers() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    // sin/cos at infinity do not converge; the saturation fold leaves
    // them symbolic, so the rule must decline (stay residual).
    for source in ["sin(1/x)", "cos(1/x)"] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            try_limit_rules_at_finite_one_sided(&mut ctx, expr, x, zero, FiniteLimitSide::Right)
                .is_none(),
            "must decline: {source}"
        );
    }
}

#[test]
fn bilateral_even_cosh_over_pole_saturates_to_infinity() {
    // cosh is even, so cosh(1/x) -> +inf from both sides even though the
    // inner pole 1/x diverges with opposite signs; shifted and scaled
    // poles behave the same.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (source, point) in [
        ("cosh(1/x)", "0"),
        ("cosh(3/x)", "0"),
        ("cosh(1/(x - 2))", "2"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let point_expr = parse_expr(&mut ctx, point);
        let out = try_limit_rules_at_finite(&mut ctx, expr, x, point_expr)
            .unwrap_or_else(|| panic!("cosh pole must saturate: {source} at {point}"));
        assert_eq!(display_expr(&ctx, out), "infinity", "{source} at {point}");
    }
}

#[test]
fn bilateral_even_cosh_rule_declines_non_cosh_and_non_pole() {
    // Odd outers (sinh: sides disagree), oscillating inners (cosh(sin(1/x))
    // has no inner infinity), and a convergent inner (cosh(x): inner -> 0)
    // must not be folded to +inf by this rule.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    for source in ["sinh(1/x)", "cosh(sin(1/x))", "cosh(x)", "cos(1/x)"] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            apply_finite_bilateral_even_saturating_pole_rule(&mut ctx, expr, x, zero).is_none(),
            "even-cosh pole rule must decline: {source}"
        );
    }
}

#[test]
fn oscillating_outer_over_even_pole_stays_residual() {
    // sin/cos of an even pole (inner -> +inf both sides) oscillate, so the
    // limit does not exist. The composition rule must decline rather than
    // leak an unfolded sin(infinity)/cos(infinity) atom. Finite arguments
    // and saturating outers (atan/cosh) are unaffected.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    for source in ["cos(1/x^2)", "sin(1/x^2)", "cos(1/x^4)"] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            apply_finite_total_real_unary_composition_rule(&mut ctx, expr, x, zero).is_none(),
            "oscillating outer over an even pole must decline: {source}"
        );
    }
    // A saturating sibling still resolves (raw atan(infinity), which the
    // eval layer folds to pi/2 - foldable, unlike the oscillating atoms),
    // and a finite argument folds cleanly here.
    let atan_pole = parse_expr(&mut ctx, "atan(1/x^2)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, atan_pole, x, zero).is_some(),
        "atan over an even pole must still resolve"
    );
    let cos_finite = parse_expr(&mut ctx, "cos(x^2)");
    let cos_finite_out = try_limit_rules_at_finite(&mut ctx, cos_finite, x, zero)
        .expect("cos of a finite-argument limit must resolve");
    assert_eq!(display_expr(&ctx, cos_finite_out), "1");
}

#[test]
fn finite_squeeze_bounded_product_collapses_to_zero() {
    // Squeeze theorem: an infinitesimal times a bounded oscillator
    // tends to 0 even though the oscillator itself has no limit.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (source, point) in [
        ("x * sin(1/x)", "0"),
        ("x^2 * cos(1/x)", "0"),
        ("sin(x) * sin(1/x)", "0"),
        ("x * sin(1/x^2)", "0"),
        ("-x * sin(1/x)", "0"),
        ("3 * x * sin(1/x) * cos(1/x)", "0"),
        ("(x - 2) * sin(1/(x - 2))", "2"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let point_expr = parse_expr(&mut ctx, point);
        let out = try_limit_rules_at_finite(&mut ctx, expr, x, point_expr)
            .unwrap_or_else(|| panic!("squeeze must resolve: {source} at {point}"));
        assert_eq!(display_expr(&ctx, out), "0", "{source} at {point}");
    }
}

#[test]
fn finite_squeeze_declines_unsound_shapes() {
    // Each of these must stay residual: a bare oscillator (no
    // infinitesimal), a scaled oscillator (no infinitesimal), an
    // unbounded outer (tan), a divergent cofactor (1/x), and a
    // domain-restricted argument (ln/sqrt are not two-sided).
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    for source in [
        "sin(1/x)",
        "2 * sin(1/x)",
        "x * tan(1/x)",
        "x * sin(ln(x))",
        "x * sin(1/sqrt(x))",
        // Identically-zero denominator: sin(1/(x - x)) = sin(1/0) is
        // undefined on the whole neighbourhood, so it is NOT a bounded
        // oscillator and the product has no limit.
        "x * sin(1/(x - x))",
    ] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            apply_finite_squeeze_bounded_product_rule(&mut ctx, expr, x, zero).is_none(),
            "squeeze must decline: {source}"
        );
    }
}

#[test]
fn finite_zero_times_unbounded_function_stays_residual() {
    // SOUNDNESS: 0 * infinity is indeterminate. An infinitesimal times an
    // UNBOUNDED function (sinh/cosh/exp of an argument that diverges) must
    // NOT collapse to 0 - the divergent cofactor can dominate
    // (x * sinh(1/x^2) -> +inf). These stay honest residuals.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    for source in [
        "x * sinh(1/x^2)",
        "x * cosh(1/x^2)",
        "x * exp(1/x^2)",
        "x * cosh(1/x)",
        "2 * x * sinh(1/x^2)",
        "x^2 * sinh(1/x^2)",
    ] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            try_limit_rules_at_finite(&mut ctx, expr, x, zero).is_none(),
            "0 * unbounded must stay residual: {source}"
        );
    }
}

#[test]
fn finite_unbounded_function_saturates_and_bounded_product_collapses() {
    // The saturation fold makes a growing function of a divergent argument
    // resolve (sinh(1/x^2) -> inf, tanh(1/x^2) -> 1, exp(-1/x^2) -> 0), and
    // a genuinely DECAYING or BOUNDED cofactor still collapses the product.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    for (source, expected) in [
        ("sinh(1/x^2)", "infinity"),
        ("cosh(1/x^2)", "infinity"),
        ("tanh(1/x^2)", "1"),
        ("x * exp(-1/x^2)", "0"),
        ("x * tanh(1/x^2)", "0"),
        ("x * sin(1/x^2)", "0"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = try_limit_rules_at_finite(&mut ctx, expr, x, zero)
            .unwrap_or_else(|| panic!("must resolve: {source}"));
        assert_eq!(display_expr(&ctx, out), expected, "{source}");
    }
}

#[test]
fn finite_radical_conjugate_resolves_removable_root_quotients() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (source, point_src, num, den) in [
        ("(sqrt(x) - 2)/(x - 4)", "4", 1, 4),
        ("(sqrt(x) - 3)/(x - 9)", "9", 1, 6),
        ("(sqrt(x + 1) - 2)/(x - 3)", "3", 1, 4),
        ("(2*sqrt(x) - 4)/(x - 4)", "4", 1, 2),
        ("(sqrt(2*x + 1) - 3)/(x - 4)", "4", 1, 3),
        ("(sqrt(x) - 2)/(x^2 - 16)", "4", 1, 32),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let point = parse_expr(&mut ctx, point_src);
        let out = apply_finite_radical_conjugate_rule(&mut ctx, expr, x, point)
            .unwrap_or_else(|| panic!("must resolve: {source}"));
        assert_number_expr(&ctx, out, num, den);
    }
}

#[test]
fn finite_radical_conjugate_declines_non_zero_over_zero_and_irrational() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (source, point_src) in [
        // numerator nonzero at the point (pole, not removable):
        ("(sqrt(x) - 2)/(x - 1)", "1"),
        // irrational radical value at the point:
        ("(sqrt(x) - 1)/(x - 2)", "2"),
        // not a 0/0 form (denominator nonzero):
        ("(sqrt(x) - 2)/(x - 9)", "9"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let point = parse_expr(&mut ctx, point_src);
        assert!(
            apply_finite_radical_conjugate_rule(&mut ctx, expr, x, point).is_none(),
            "must decline: {source}"
        );
    }
}

#[test]
fn finite_radical_difference_conjugate_resolves_sqrt_minus_sqrt() {
    // (s1 sqrt(L1) + s2 sqrt(L2))/den at a 0/0 point, rationalized by the
    // conjugate. Values cross-checked numerically (mpmath dps 40).
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (source, point_src, num, den) in [
        ("(sqrt(1+x) - sqrt(1-x))/x", "0", 1, 1),
        ("(sqrt(1+2*x) - sqrt(1-2*x))/x", "0", 2, 1),
        ("(sqrt(4+x) - sqrt(4-x))/x", "0", 1, 2),
        ("(sqrt(x+3) - sqrt(2*x+2))/(x-1)", "1", -1, 4),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let point = parse_expr(&mut ctx, point_src);
        let out = apply_finite_radical_difference_conjugate_rule(&mut ctx, expr, x, point)
            .unwrap_or_else(|| panic!("must resolve: {source}"));
        assert_number_expr(&ctx, out, num, den);
    }
}

#[test]
fn finite_radical_difference_conjugate_declines_pole_irrational_and_nonlinear() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (source, point_src) in [
        // numerator nonzero at the point (sqrt(2) - 1 != 0): a pole.
        ("(sqrt(2+x) - sqrt(1-x))/x", "0"),
        // a SUM of square roots (both positive): not 0/0 at the point.
        ("(sqrt(1+x) + sqrt(1-x))/x", "0"),
        // a nonlinear radicand is out of scope (1 + x^2).
        ("(sqrt(1+x) - sqrt(1+x^2))/x", "0"),
        // irrational radical value at the point (sqrt(2)).
        ("(sqrt(2+x) - sqrt(2-x))/x", "0"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let point = parse_expr(&mut ctx, point_src);
        assert!(
            apply_finite_radical_difference_conjugate_rule(&mut ctx, expr, x, point).is_none(),
            "must decline: {source}"
        );
    }
}

#[test]
fn finite_rational_polynomial_limit_resolves_exact_removable_holes_only() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let point = parse_expr(&mut ctx, "1");

    let simple_hole = parse_expr(&mut ctx, "(x^2 - 1)/(x - 1)");
    let simple_hole_out = try_limit_rules_at_finite(&mut ctx, simple_hole, x, point)
        .expect("expected exact removable rational-polynomial limit");
    let Expr::Number(value) = ctx.get(simple_hole_out) else {
        panic!("expected exact numeric removable rational-polynomial limit");
    };
    assert_eq!(value, &BigRational::from_integer(2.into()));

    let higher_numerator_multiplicity = parse_expr(&mut ctx, "(x - 1)^2/(x - 1)");
    let higher_numerator_out =
        try_limit_rules_at_finite(&mut ctx, higher_numerator_multiplicity, x, point)
            .expect("expected removable zero limit");
    let Expr::Number(value) = ctx.get(higher_numerator_out) else {
        panic!("expected exact zero removable rational-polynomial limit");
    };
    assert_eq!(value, &BigRational::zero());

    let finite_pole = parse_expr(&mut ctx, "(x - 1)/(x - 1)^2");
    assert!(
        try_limit_rules_at_finite(&mut ctx, finite_pole, x, point).is_none(),
        "odd-order finite poles must remain residual because the two-sided limit diverges differently"
    );

    let even_positive_pole = parse_expr(&mut ctx, "2/(x - 1)^2");
    let even_positive_out = try_limit_rules_at_finite(&mut ctx, even_positive_pole, x, point)
        .expect("expected bilateral even-order rational pole");
    assert_eq!(display_expr(&ctx, even_positive_out), "infinity");

    let even_negative_pole = parse_expr(&mut ctx, "-2/(x - 1)^2");
    let even_negative_out = try_limit_rules_at_finite(&mut ctx, even_negative_pole, x, point)
        .expect("expected negative bilateral even-order rational pole");
    assert_eq!(display_expr(&ctx, even_negative_out), "-infinity");
}

#[test]
fn finite_one_sided_limits_resolve_orientation_and_simple_poles() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let point_zero = parse_expr(&mut ctx, "0");

    let abs_ratio = parse_expr(&mut ctx, "abs(x)/x");
    let right_abs = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        abs_ratio,
        x,
        point_zero,
        FiniteLimitSide::Right,
    )
    .expect("expected right-sided abs orientation limit");
    assert_number_expr(&ctx, right_abs, 1, 1);

    let left_abs = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        abs_ratio,
        x,
        point_zero,
        FiniteLimitSide::Left,
    )
    .expect("expected left-sided abs orientation limit");
    assert_number_expr(&ctx, left_abs, -1, 1);

    let sign_right = parse_expr(&mut ctx, "sign(x)");
    let sign_right_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        sign_right,
        x,
        point_zero,
        FiniteLimitSide::Right,
    )
    .expect("expected right-sided sign orientation limit");
    assert_number_expr(&ctx, sign_right_out, 1, 1);

    let sign_left = parse_expr(&mut ctx, "sign(x)");
    let sign_left_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        sign_left,
        x,
        point_zero,
        FiniteLimitSide::Left,
    )
    .expect("expected left-sided sign orientation limit");
    assert_number_expr(&ctx, sign_left_out, -1, 1);

    let point_one = parse_expr(&mut ctx, "1");
    let sign_quadratic_left = parse_expr(&mut ctx, "sign(x^2 - 1)");
    let sign_quadratic_left_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        sign_quadratic_left,
        x,
        point_one,
        FiniteLimitSide::Left,
    )
    .expect("expected left-sided quadratic sign orientation limit");
    assert_number_expr(&ctx, sign_quadratic_left_out, -1, 1);

    let sign_even_left = parse_expr(&mut ctx, "sign(x^2)");
    let sign_even_left_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        sign_even_left,
        x,
        point_zero,
        FiniteLimitSide::Left,
    )
    .expect("expected even-order sign orientation limit");
    assert_number_expr(&ctx, sign_even_left_out, 1, 1);

    let reciprocal = parse_expr(&mut ctx, "1/x");
    let right_pole = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        reciprocal,
        x,
        point_zero,
        FiniteLimitSide::Right,
    )
    .expect("expected right-sided rational pole");
    assert_eq!(display_expr(&ctx, right_pole), "infinity");

    let left_pole = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        reciprocal,
        x,
        point_zero,
        FiniteLimitSide::Left,
    )
    .expect("expected left-sided rational pole");
    assert_eq!(display_expr(&ctx, left_pole), "-infinity");

    let ln_right_endpoint = parse_expr(&mut ctx, "ln(x)");
    let ln_right_endpoint_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        ln_right_endpoint,
        x,
        point_zero,
        FiniteLimitSide::Right,
    )
    .expect("expected right-sided log endpoint");
    assert_eq!(display_expr(&ctx, ln_right_endpoint_out), "-infinity");

    let ln_left_endpoint = parse_expr(&mut ctx, "ln(x)");
    assert!(
        try_limit_rules_at_finite_one_sided(
            &mut ctx,
            ln_left_endpoint,
            x,
            point_zero,
            FiniteLimitSide::Left,
        )
        .is_none(),
        "wrong-side log endpoint must remain residual"
    );

    let ln_neg_left_endpoint = parse_expr(&mut ctx, "ln(-x)");
    let ln_neg_left_endpoint_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        ln_neg_left_endpoint,
        x,
        point_zero,
        FiniteLimitSide::Left,
    )
    .expect("expected left-sided positive-tail log endpoint");
    assert_eq!(display_expr(&ctx, ln_neg_left_endpoint_out), "-infinity");

    let log2_right_endpoint = parse_expr(&mut ctx, "log2(x)");
    let log2_right_endpoint_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        log2_right_endpoint,
        x,
        point_zero,
        FiniteLimitSide::Right,
    )
    .expect("expected fixed-base log endpoint");
    assert_eq!(display_expr(&ctx, log2_right_endpoint_out), "-infinity");

    let reciprocal_base_log = parse_expr(&mut ctx, "log(1/2, x)");
    let reciprocal_base_log_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        reciprocal_base_log,
        x,
        point_zero,
        FiniteLimitSide::Right,
    )
    .expect("expected reciprocal-base log endpoint");
    assert_eq!(display_expr(&ctx, reciprocal_base_log_out), "infinity");

    let point_one = parse_expr(&mut ctx, "1");
    let unit_boundary_base_above = parse_expr(&mut ctx, "log(x, (x - 1)/(x + 3))");
    let unit_boundary_base_above_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        unit_boundary_base_above,
        x,
        point_one,
        FiniteLimitSide::Right,
    )
    .expect("expected unit-boundary base log endpoint from above");
    assert_eq!(
        display_expr(&ctx, unit_boundary_base_above_out),
        "-infinity"
    );

    let unit_boundary_base_below = parse_expr(&mut ctx, "log(2 - x, (x - 1)/(x + 3))");
    let unit_boundary_base_below_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        unit_boundary_base_below,
        x,
        point_one,
        FiniteLimitSide::Right,
    )
    .expect("expected unit-boundary base log endpoint from below");
    assert_eq!(display_expr(&ctx, unit_boundary_base_below_out), "infinity");

    let unit_boundary_wrong_side = parse_expr(&mut ctx, "log(x, (x - 1)/(x + 3))");
    assert!(
        try_limit_rules_at_finite_one_sided(
            &mut ctx,
            unit_boundary_wrong_side,
            x,
            point_one,
            FiniteLimitSide::Left,
        )
        .is_none(),
        "unit-boundary base log endpoint must still reject wrong-side arguments"
    );

    let sqrt_right_endpoint = parse_expr(&mut ctx, "sqrt(x)");
    let sqrt_right_endpoint_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        sqrt_right_endpoint,
        x,
        point_zero,
        FiniteLimitSide::Right,
    )
    .expect("expected right-sided sqrt endpoint");
    assert_eq!(display_expr(&ctx, sqrt_right_endpoint_out), "0");

    let sqrt_left_endpoint = parse_expr(&mut ctx, "sqrt(x)");
    assert!(
        try_limit_rules_at_finite_one_sided(
            &mut ctx,
            sqrt_left_endpoint,
            x,
            point_zero,
            FiniteLimitSide::Left,
        )
        .is_none(),
        "wrong-side sqrt endpoint must remain residual"
    );

    let sqrt_neg_left_endpoint = parse_expr(&mut ctx, "sqrt(-x)");
    let sqrt_neg_left_endpoint_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        sqrt_neg_left_endpoint,
        x,
        point_zero,
        FiniteLimitSide::Left,
    )
    .expect("expected left-sided sqrt endpoint");
    assert_eq!(display_expr(&ctx, sqrt_neg_left_endpoint_out), "0");

    let sqrt_even_endpoint = parse_expr(&mut ctx, "sqrt(x^2)");
    let sqrt_even_endpoint_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        sqrt_even_endpoint,
        x,
        point_zero,
        FiniteLimitSide::Left,
    )
    .expect("expected even-order sqrt endpoint");
    assert_eq!(display_expr(&ctx, sqrt_even_endpoint_out), "0");

    let sqrt_abs_endpoint = parse_expr(&mut ctx, "sqrt(abs(x))");
    let sqrt_abs_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        sqrt_abs_endpoint,
        x,
        point_zero,
        FiniteLimitSide::Right,
    )
    .expect("abs resolves by approach side: |x| = x from the right");
    assert_eq!(display_expr(&ctx, sqrt_abs_out), "0");

    let point_one = parse_expr(&mut ctx, "1");
    let acosh_right_endpoint = parse_expr(&mut ctx, "acosh(x)");
    let acosh_right_endpoint_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        acosh_right_endpoint,
        x,
        point_one,
        FiniteLimitSide::Right,
    )
    .expect("expected right-sided acosh lower-bound endpoint");
    assert_eq!(display_expr(&ctx, acosh_right_endpoint_out), "0");

    let acosh_left_endpoint = parse_expr(&mut ctx, "acosh(x)");
    assert!(
        try_limit_rules_at_finite_one_sided(
            &mut ctx,
            acosh_left_endpoint,
            x,
            point_one,
            FiniteLimitSide::Left,
        )
        .is_none(),
        "wrong-side acosh lower-bound endpoint must remain residual"
    );

    let acosh_neg_orientation_endpoint = parse_expr(&mut ctx, "acosh(2 - x)");
    let acosh_neg_orientation_endpoint_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        acosh_neg_orientation_endpoint,
        x,
        point_one,
        FiniteLimitSide::Left,
    )
    .expect("expected left-sided negative-orientation acosh lower-bound endpoint");
    assert_eq!(display_expr(&ctx, acosh_neg_orientation_endpoint_out), "0");

    let acosh_even_endpoint = parse_expr(&mut ctx, "acosh(1 + x^2)");
    let acosh_even_endpoint_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        acosh_even_endpoint,
        x,
        point_zero,
        FiniteLimitSide::Left,
    )
    .expect("expected even-order acosh lower-bound endpoint");
    assert_eq!(display_expr(&ctx, acosh_even_endpoint_out), "0");

    let acosh_sqrt_endpoint = parse_expr(&mut ctx, "acosh(sqrt(x))");
    assert!(
        try_limit_rules_at_finite_one_sided(
            &mut ctx,
            acosh_sqrt_endpoint,
            x,
            point_one,
            FiniteLimitSide::Right,
        )
        .is_none(),
        "non-polynomial acosh endpoint remains residual for a later policy"
    );

    let acos_upper_left_endpoint = parse_expr(&mut ctx, "acos(x)");
    let acos_upper_left_endpoint_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        acos_upper_left_endpoint,
        x,
        point_one,
        FiniteLimitSide::Left,
    )
    .expect("expected left-sided inverse-trig upper endpoint");
    assert_eq!(display_expr(&ctx, acos_upper_left_endpoint_out), "0");

    let asin_upper_left_endpoint = parse_expr(&mut ctx, "asin(x)");
    let asin_upper_left_endpoint_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        asin_upper_left_endpoint,
        x,
        point_one,
        FiniteLimitSide::Left,
    )
    .expect("expected left-sided arcsin upper endpoint");
    assert_eq!(display_expr(&ctx, asin_upper_left_endpoint_out), "pi / 2");

    let acos_upper_right_endpoint = parse_expr(&mut ctx, "acos(x)");
    assert!(
        try_limit_rules_at_finite_one_sided(
            &mut ctx,
            acos_upper_right_endpoint,
            x,
            point_one,
            FiniteLimitSide::Right,
        )
        .is_none(),
        "wrong-side inverse-trig upper endpoint must remain residual"
    );

    let acos_upper_neg_orientation_endpoint = parse_expr(&mut ctx, "acos(2 - x)");
    let acos_upper_neg_orientation_endpoint_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        acos_upper_neg_orientation_endpoint,
        x,
        point_one,
        FiniteLimitSide::Right,
    )
    .expect("expected right-sided negative-orientation inverse-trig upper endpoint");
    assert_eq!(
        display_expr(&ctx, acos_upper_neg_orientation_endpoint_out),
        "0"
    );

    let point_minus_one = parse_expr(&mut ctx, "-1");
    let acos_lower_right_endpoint = parse_expr(&mut ctx, "acos(x)");
    let acos_lower_right_endpoint_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        acos_lower_right_endpoint,
        x,
        point_minus_one,
        FiniteLimitSide::Right,
    )
    .expect("expected right-sided inverse-trig lower endpoint");
    assert_eq!(display_expr(&ctx, acos_lower_right_endpoint_out), "pi");

    let asin_lower_right_endpoint = parse_expr(&mut ctx, "asin(x)");
    let asin_lower_right_endpoint_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        asin_lower_right_endpoint,
        x,
        point_minus_one,
        FiniteLimitSide::Right,
    )
    .expect("expected right-sided arcsin lower endpoint");
    assert_eq!(display_expr(&ctx, asin_lower_right_endpoint_out), "-pi / 2");

    let asin_lower_left_endpoint = parse_expr(&mut ctx, "asin(x)");
    assert!(
        try_limit_rules_at_finite_one_sided(
            &mut ctx,
            asin_lower_left_endpoint,
            x,
            point_minus_one,
            FiniteLimitSide::Left,
        )
        .is_none(),
        "wrong-side inverse-trig lower endpoint must remain residual"
    );

    let acos_above_domain_endpoint = parse_expr(&mut ctx, "acos(1 + x^2)");
    assert!(
        try_limit_rules_at_finite_one_sided(
            &mut ctx,
            acos_above_domain_endpoint,
            x,
            point_zero,
            FiniteLimitSide::Right,
        )
        .is_none(),
        "empty-domain one-sided inverse-trig endpoint must remain residual"
    );

    let atanh_upper_left_endpoint = parse_expr(&mut ctx, "atanh(x)");
    let atanh_upper_left_endpoint_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        atanh_upper_left_endpoint,
        x,
        point_one,
        FiniteLimitSide::Left,
    )
    .expect("expected left-sided atanh upper endpoint");
    assert_eq!(
        display_expr(&ctx, atanh_upper_left_endpoint_out),
        "infinity"
    );

    let atanh_upper_right_endpoint = parse_expr(&mut ctx, "atanh(x)");
    assert!(
        try_limit_rules_at_finite_one_sided(
            &mut ctx,
            atanh_upper_right_endpoint,
            x,
            point_one,
            FiniteLimitSide::Right,
        )
        .is_none(),
        "wrong-side atanh upper endpoint must remain residual"
    );

    let atanh_upper_neg_orientation_endpoint = parse_expr(&mut ctx, "atanh(2 - x)");
    let atanh_upper_neg_orientation_endpoint_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        atanh_upper_neg_orientation_endpoint,
        x,
        point_one,
        FiniteLimitSide::Right,
    )
    .expect("expected right-sided negative-orientation atanh upper endpoint");
    assert_eq!(
        display_expr(&ctx, atanh_upper_neg_orientation_endpoint_out),
        "infinity"
    );

    let atanh_lower_right_endpoint = parse_expr(&mut ctx, "atanh(x)");
    let atanh_lower_right_endpoint_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        atanh_lower_right_endpoint,
        x,
        point_minus_one,
        FiniteLimitSide::Right,
    )
    .expect("expected right-sided atanh lower endpoint");
    assert_eq!(
        display_expr(&ctx, atanh_lower_right_endpoint_out),
        "-infinity"
    );

    let atanh_lower_left_endpoint = parse_expr(&mut ctx, "atanh(x)");
    assert!(
        try_limit_rules_at_finite_one_sided(
            &mut ctx,
            atanh_lower_left_endpoint,
            x,
            point_minus_one,
            FiniteLimitSide::Left,
        )
        .is_none(),
        "wrong-side atanh lower endpoint must remain residual"
    );

    let atanh_above_domain_endpoint = parse_expr(&mut ctx, "atanh(1 + x^2)");
    let atanh_above_domain_endpoint_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        atanh_above_domain_endpoint,
        x,
        point_zero,
        FiniteLimitSide::Right,
    )
    .expect("expected empty-domain one-sided atanh endpoint to be undefined");
    assert_eq!(
        display_expr(&ctx, atanh_above_domain_endpoint_out),
        "undefined"
    );

    let acos_sqrt_endpoint = parse_expr(&mut ctx, "acos(sqrt(x))");
    assert!(
        try_limit_rules_at_finite_one_sided(
            &mut ctx,
            acos_sqrt_endpoint,
            x,
            point_one,
            FiniteLimitSide::Left,
        )
        .is_none(),
        "non-polynomial inverse-trig endpoint remains residual for a later policy"
    );
}

#[test]
fn finite_bilateral_abs_polynomial_ratio_resolves_only_matching_sides() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let point_zero = parse_expr(&mut ctx, "0");

    let bilateral_abs_even_pole = parse_expr(&mut ctx, "abs(x)/x^2");
    let bilateral_abs_even_pole_out =
        try_limit_rules_at_finite(&mut ctx, bilateral_abs_even_pole, x, point_zero)
            .expect("expected matching bilateral abs polynomial-ratio pole");
    assert_eq!(display_expr(&ctx, bilateral_abs_even_pole_out), "infinity");

    let abs_orientation_jump = parse_expr(&mut ctx, "abs(x)/x");
    assert!(
        try_limit_rules_at_finite(&mut ctx, abs_orientation_jump, x, point_zero).is_none(),
        "bilateral abs orientation jump must remain residual when one-sided limits differ"
    );

    let sign_even_orientation = parse_expr(&mut ctx, "sign(x^2)");
    let sign_even_orientation_out =
        try_limit_rules_at_finite(&mut ctx, sign_even_orientation, x, point_zero)
            .expect("expected matching bilateral sign polynomial limit");
    assert_number_expr(&ctx, sign_even_orientation_out, 1, 1);

    let sign_orientation_jump = parse_expr(&mut ctx, "sign(x)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, sign_orientation_jump, x, point_zero).is_none(),
        "bilateral sign orientation jump must remain residual when one-sided limits differ"
    );

    let point_one = parse_expr(&mut ctx, "1");
    let shifted_abs_even_pole = parse_expr(&mut ctx, "abs(x - 1)/(x - 1)^2");
    let shifted_abs_even_pole_out =
        try_limit_rules_at_finite(&mut ctx, shifted_abs_even_pole, x, point_one)
            .expect("expected shifted matching bilateral abs polynomial-ratio pole");
    assert_eq!(display_expr(&ctx, shifted_abs_even_pole_out), "infinity");
}

#[test]
fn finite_bilateral_trig_power_poles_resolve_only_matching_sides() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let point_zero = parse_expr(&mut ctx, "0");

    let even_sine_pole = parse_expr(&mut ctx, "1/sin(x)^2");
    let even_sine_pole_out = try_limit_rules_at_finite(&mut ctx, even_sine_pole, x, point_zero)
        .expect("expected bilateral even-order sine pole");
    assert_eq!(display_expr(&ctx, even_sine_pole_out), "infinity");

    let negative_scaled_sine_pole = parse_expr(&mut ctx, "-1/sin(2*x)^2");
    let negative_scaled_sine_pole_out =
        try_limit_rules_at_finite(&mut ctx, negative_scaled_sine_pole, x, point_zero)
            .expect("expected negative bilateral even-order sine pole");
    assert_eq!(
        display_expr(&ctx, negative_scaled_sine_pole_out),
        "-infinity"
    );

    let first_order_sine_pole = parse_expr(&mut ctx, "1/sin(x)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, first_order_sine_pole, x, point_zero).is_none(),
        "bilateral first-order sine pole must remain residual when one-sided limits differ"
    );

    let reciprocal_sine_even_pole = parse_expr(&mut ctx, "csc(x + pi)^2");
    let reciprocal_sine_even_pole_out =
        try_limit_rules_at_finite(&mut ctx, reciprocal_sine_even_pole, x, point_zero)
            .expect("expected bilateral even-order reciprocal sine pole");
    assert_eq!(
        display_expr(&ctx, reciprocal_sine_even_pole_out),
        "infinity"
    );

    let reciprocal_sine_first_order_pole = parse_expr(&mut ctx, "csc(x + pi)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, reciprocal_sine_first_order_pole, x, point_zero)
            .is_none(),
        "bilateral first-order reciprocal sine pole must remain residual"
    );

    let right_reciprocal_sine_first_order_pole = parse_expr(&mut ctx, "csc(x + pi)");
    let right_reciprocal_sine_first_order_pole_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        right_reciprocal_sine_first_order_pole,
        x,
        point_zero,
        FiniteLimitSide::Right,
    )
    .expect("expected right-sided first-order reciprocal sine pole");
    assert_eq!(
        display_expr(&ctx, right_reciprocal_sine_first_order_pole_out),
        "-infinity"
    );

    let scaled_right_reciprocal_sine_first_order_pole = parse_expr(&mut ctx, "-2*csc(x + pi)");
    let scaled_right_reciprocal_sine_first_order_pole_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        scaled_right_reciprocal_sine_first_order_pole,
        x,
        point_zero,
        FiniteLimitSide::Right,
    )
    .expect("expected scaled right-sided first-order reciprocal sine pole");
    assert_eq!(
        display_expr(&ctx, scaled_right_reciprocal_sine_first_order_pole_out),
        "infinity"
    );

    let negative_scaled_reciprocal_sine_even_pole = parse_expr(&mut ctx, "-3*csc(x + pi)^2");
    let negative_scaled_reciprocal_sine_even_pole_out = try_limit_rules_at_finite(
        &mut ctx,
        negative_scaled_reciprocal_sine_even_pole,
        x,
        point_zero,
    )
    .expect("expected negative scaled bilateral even-order reciprocal sine pole");
    assert_eq!(
        display_expr(&ctx, negative_scaled_reciprocal_sine_even_pole_out),
        "-infinity"
    );

    let right_first_order_sine_pole = parse_expr(&mut ctx, "1/sin(x)");
    let right_first_order_sine_pole_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        right_first_order_sine_pole,
        x,
        point_zero,
        FiniteLimitSide::Right,
    )
    .expect("expected right-sided first-order sine pole");
    assert_eq!(
        display_expr(&ctx, right_first_order_sine_pole_out),
        "infinity"
    );

    let left_first_order_sine_pole = parse_expr(&mut ctx, "1/sin(x)");
    let left_first_order_sine_pole_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        left_first_order_sine_pole,
        x,
        point_zero,
        FiniteLimitSide::Left,
    )
    .expect("expected left-sided first-order sine pole");
    assert_eq!(
        display_expr(&ctx, left_first_order_sine_pole_out),
        "-infinity"
    );

    let point_one = parse_expr(&mut ctx, "1");
    let shifted_sine_pole = parse_expr(&mut ctx, "1/sin(x - 1)^2");
    let shifted_sine_pole_out =
        try_limit_rules_at_finite(&mut ctx, shifted_sine_pole, x, point_one)
            .expect("expected shifted bilateral even-order sine pole");
    assert_eq!(display_expr(&ctx, shifted_sine_pole_out), "infinity");

    let even_cosine_pole = parse_expr(&mut ctx, "1/cos(pi/2 + x)^2");
    let even_cosine_pole_out = try_limit_rules_at_finite(&mut ctx, even_cosine_pole, x, point_zero)
        .expect("expected bilateral even-order cosine pole at a tabulated zero");
    assert_eq!(display_expr(&ctx, even_cosine_pole_out), "infinity");

    let negative_scaled_cosine_pole = parse_expr(&mut ctx, "-1/cos(pi/2 + 2*x)^2");
    let negative_scaled_cosine_pole_out =
        try_limit_rules_at_finite(&mut ctx, negative_scaled_cosine_pole, x, point_zero)
            .expect("expected negative bilateral even-order cosine pole");
    assert_eq!(
        display_expr(&ctx, negative_scaled_cosine_pole_out),
        "-infinity"
    );

    let first_order_cosine_pole = parse_expr(&mut ctx, "1/cos(pi/2 + x)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, first_order_cosine_pole, x, point_zero).is_none(),
        "bilateral first-order cosine pole must remain residual when one-sided limits differ"
    );

    let reciprocal_cosine_even_pole = parse_expr(&mut ctx, "sec(pi/2 + x)^2");
    let reciprocal_cosine_even_pole_out =
        try_limit_rules_at_finite(&mut ctx, reciprocal_cosine_even_pole, x, point_zero)
            .expect("expected bilateral even-order reciprocal cosine pole");
    assert_eq!(
        display_expr(&ctx, reciprocal_cosine_even_pole_out),
        "infinity"
    );

    let scaled_right_reciprocal_cosine_first_order_pole = parse_expr(&mut ctx, "2*sec(pi/2 + x)");
    let scaled_right_reciprocal_cosine_first_order_pole_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        scaled_right_reciprocal_cosine_first_order_pole,
        x,
        point_zero,
        FiniteLimitSide::Right,
    )
    .expect("expected scaled right-sided first-order reciprocal cosine pole");
    assert_eq!(
        display_expr(&ctx, scaled_right_reciprocal_cosine_first_order_pole_out),
        "-infinity"
    );

    let right_first_order_cosine_pole = parse_expr(&mut ctx, "1/cos(pi/2 + x)");
    let right_first_order_cosine_pole_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        right_first_order_cosine_pole,
        x,
        point_zero,
        FiniteLimitSide::Right,
    )
    .expect("expected right-sided first-order cosine pole");
    assert_eq!(
        display_expr(&ctx, right_first_order_cosine_pole_out),
        "-infinity"
    );

    let left_first_order_cosine_pole = parse_expr(&mut ctx, "1/cos(pi/2 + x)");
    let left_first_order_cosine_pole_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        left_first_order_cosine_pole,
        x,
        point_zero,
        FiniteLimitSide::Left,
    )
    .expect("expected left-sided first-order cosine pole");
    assert_eq!(
        display_expr(&ctx, left_first_order_cosine_pole_out),
        "infinity"
    );

    let point_pi_over_two = parse_expr(&mut ctx, "pi/2");
    let direct_special_point_cosine_pole = parse_expr(&mut ctx, "1/cos(x)^2");
    let direct_special_point_cosine_pole_out = try_limit_rules_at_finite(
        &mut ctx,
        direct_special_point_cosine_pole,
        x,
        point_pi_over_two,
    )
    .expect("expected bilateral even-order cosine pole at direct special-angle point");
    assert_eq!(
        display_expr(&ctx, direct_special_point_cosine_pole_out),
        "infinity"
    );

    let direct_first_order_cosine_pole = parse_expr(&mut ctx, "1/cos(x)");
    assert!(
        try_limit_rules_at_finite(
            &mut ctx,
            direct_first_order_cosine_pole,
            x,
            point_pi_over_two
        )
        .is_none(),
        "direct bilateral first-order cosine pole must remain residual"
    );

    let direct_right_first_order_cosine_pole = parse_expr(&mut ctx, "1/cos(x)");
    let direct_right_first_order_cosine_pole_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        direct_right_first_order_cosine_pole,
        x,
        point_pi_over_two,
        FiniteLimitSide::Right,
    )
    .expect("expected direct right-sided first-order cosine pole at special-angle point");
    assert_eq!(
        display_expr(&ctx, direct_right_first_order_cosine_pole_out),
        "-infinity"
    );

    let point_two_pi = parse_expr(&mut ctx, "2*pi");
    let direct_rational_pi_sine_pole = parse_expr(&mut ctx, "1/sin(x)^2");
    let direct_rational_pi_sine_pole_out =
        try_limit_rules_at_finite(&mut ctx, direct_rational_pi_sine_pole, x, point_two_pi)
            .expect("expected bilateral even-order sine pole at rational-pi point");
    assert_eq!(
        display_expr(&ctx, direct_rational_pi_sine_pole_out),
        "infinity"
    );

    let direct_rational_pi_first_order_sine_pole = parse_expr(&mut ctx, "1/sin(x)");
    assert!(
        try_limit_rules_at_finite(
            &mut ctx,
            direct_rational_pi_first_order_sine_pole,
            x,
            point_two_pi
        )
        .is_none(),
        "direct bilateral first-order sine pole at rational-pi point must remain residual"
    );

    let right_tangent_first_order_pole = parse_expr(&mut ctx, "tan(pi/2 + x)");
    let right_tangent_first_order_pole_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        right_tangent_first_order_pole,
        x,
        point_zero,
        FiniteLimitSide::Right,
    )
    .expect("expected right-sided first-order tangent pole");
    assert_eq!(
        display_expr(&ctx, right_tangent_first_order_pole_out),
        "-infinity"
    );

    let left_tangent_first_order_pole = parse_expr(&mut ctx, "tan(pi/2 + x)");
    let left_tangent_first_order_pole_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        left_tangent_first_order_pole,
        x,
        point_zero,
        FiniteLimitSide::Left,
    )
    .expect("expected left-sided first-order tangent pole");
    assert_eq!(
        display_expr(&ctx, left_tangent_first_order_pole_out),
        "infinity"
    );

    let tangent_first_order_pole = parse_expr(&mut ctx, "tan(pi/2 + x)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, tangent_first_order_pole, x, point_zero).is_none(),
        "bilateral first-order tangent pole must remain residual"
    );

    let tangent_even_pole = parse_expr(&mut ctx, "tan(pi/2 + x)^2");
    let tangent_even_pole_out =
        try_limit_rules_at_finite(&mut ctx, tangent_even_pole, x, point_zero)
            .expect("expected bilateral even-order tangent pole");
    assert_eq!(display_expr(&ctx, tangent_even_pole_out), "infinity");

    let right_cotangent_first_order_pole = parse_expr(&mut ctx, "cot(x + pi)");
    let right_cotangent_first_order_pole_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        right_cotangent_first_order_pole,
        x,
        point_zero,
        FiniteLimitSide::Right,
    )
    .expect("expected right-sided first-order cotangent pole");
    assert_eq!(
        display_expr(&ctx, right_cotangent_first_order_pole_out),
        "infinity"
    );

    let explicit_right_tangent_first_order_pole =
        parse_expr(&mut ctx, "sin(pi/2 + x)/cos(pi/2 + x)");
    let explicit_right_tangent_first_order_pole_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        explicit_right_tangent_first_order_pole,
        x,
        point_zero,
        FiniteLimitSide::Right,
    )
    .expect("expected right-sided explicit sine/cosine tangent-ratio pole");
    assert_eq!(
        display_expr(&ctx, explicit_right_tangent_first_order_pole_out),
        "-infinity"
    );

    let explicit_tangent_first_order_pole = parse_expr(&mut ctx, "sin(pi/2 + x)/cos(pi/2 + x)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, explicit_tangent_first_order_pole, x, point_zero)
            .is_none(),
        "explicit bilateral first-order tangent-ratio pole must remain residual"
    );

    let explicit_tangent_even_pole = parse_expr(&mut ctx, "(sin(pi/2 + x)/cos(pi/2 + x))^2");
    let explicit_tangent_even_pole_out =
        try_limit_rules_at_finite(&mut ctx, explicit_tangent_even_pole, x, point_zero)
            .expect("expected bilateral even-order explicit tangent-ratio pole");
    assert_eq!(
        display_expr(&ctx, explicit_tangent_even_pole_out),
        "infinity"
    );

    let explicit_right_cotangent_first_order_pole = parse_expr(&mut ctx, "cos(x + pi)/sin(x + pi)");
    let explicit_right_cotangent_first_order_pole_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        explicit_right_cotangent_first_order_pole,
        x,
        point_zero,
        FiniteLimitSide::Right,
    )
    .expect("expected right-sided explicit cosine/sine cotangent-ratio pole");
    assert_eq!(
        display_expr(&ctx, explicit_right_cotangent_first_order_pole_out),
        "infinity"
    );

    let cross_argument_explicit_tangent_pole = parse_expr(&mut ctx, "sin(pi/2 + x)/cos(pi/2 - x)");
    let cross_argument_explicit_tangent_pole_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        cross_argument_explicit_tangent_pole,
        x,
        point_zero,
        FiniteLimitSide::Right,
    )
    .expect("expected cross-argument explicit tangent-ratio pole");
    assert_eq!(
        display_expr(&ctx, cross_argument_explicit_tangent_pole_out),
        "infinity"
    );

    let noisy_denominator_explicit_tangent_pole =
        parse_expr(&mut ctx, "sin(pi/2 + x)/cos(pi/2 + x + 0)");
    let noisy_denominator_explicit_tangent_pole_out = try_limit_rules_at_finite_one_sided(
        &mut ctx,
        noisy_denominator_explicit_tangent_pole,
        x,
        point_zero,
        FiniteLimitSide::Right,
    )
    .expect("expected explicit tangent-ratio pole with harmless denominator noise");
    assert_eq!(
        display_expr(&ctx, noisy_denominator_explicit_tangent_pole_out),
        "-infinity"
    );

    let zero_numerator_explicit_trig_ratio = parse_expr(&mut ctx, "sin(x)/cos(pi/2 - x)");
    assert!(
        try_limit_rules_at_finite_one_sided(
            &mut ctx,
            zero_numerator_explicit_trig_ratio,
            x,
            point_zero,
            FiniteLimitSide::Right
        )
        .is_none(),
        "explicit trig ratio with zero numerator limit must not be treated as a pole"
    );

    let point_three_pi_over_two = parse_expr(&mut ctx, "3*pi/2");
    let direct_rational_pi_cosine_pole = parse_expr(&mut ctx, "1/cos(x)^2");
    let direct_rational_pi_cosine_pole_out = try_limit_rules_at_finite(
        &mut ctx,
        direct_rational_pi_cosine_pole,
        x,
        point_three_pi_over_two,
    )
    .expect("expected bilateral even-order cosine pole at rational-pi point");
    assert_eq!(
        display_expr(&ctx, direct_rational_pi_cosine_pole_out),
        "infinity"
    );
}

#[test]
fn finite_sine_zero_quotient_limit_uses_removable_polynomial_ratio() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let point_zero = parse_expr(&mut ctx, "0");

    let cases = [
        ("sin(x)/x", 1, 1),
        ("sin(2*x)/x", 2, 1),
        ("3*sin(2*x)/(5*x)", 6, 5),
        ("sin(x^2)/x", 0, 1),
    ];

    for (input, expected_num, expected_den) in cases {
        let expr = parse_expr(&mut ctx, input);
        let out = try_limit_rules_at_finite(&mut ctx, expr, x, point_zero)
            .unwrap_or_else(|| panic!("expected finite sine quotient limit for {input}"));
        let Expr::Number(value) = ctx.get(out) else {
            panic!("expected exact rational sine quotient limit for {input}");
        };
        assert_eq!(
            value,
            &BigRational::new(BigInt::from(expected_num), BigInt::from(expected_den))
        );
    }

    let point_one = parse_expr(&mut ctx, "1");
    let shifted = parse_expr(&mut ctx, "sin(x - 1)/(x^2 - 1)");
    let shifted_out = try_limit_rules_at_finite(&mut ctx, shifted, x, point_one)
        .expect("expected shifted finite sine quotient limit");
    let Expr::Number(value) = ctx.get(shifted_out) else {
        panic!("expected exact shifted sine quotient limit");
    };
    assert_eq!(value, &BigRational::new(BigInt::from(1), BigInt::from(2)));

    let nonzero_argument = parse_expr(&mut ctx, "sin(x + 1)/x");
    assert!(
        try_limit_rules_at_finite(&mut ctx, nonzero_argument, x, point_zero).is_none(),
        "sine quotient rule must only apply when the sine argument tends to zero"
    );

    let finite_pole = parse_expr(&mut ctx, "sin(x)/x^2");
    assert!(
        try_limit_rules_at_finite(&mut ctx, finite_pole, x, point_zero).is_none(),
        "sine quotient rule must not promote finite poles"
    );
}

#[test]
fn finite_equivalent_infinitesimal_quotient_resolves_ratios() {
    // Ratio of first-order equivalent infinitesimals: inversion,
    // composition, the missing atoms (tan/asin/arctan/sinh/tanh),
    // exp, sign, and a Sub INSIDE an atom argument (which is exact).
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    for (source, expected) in [
        ("x / sin(x)", "1"),
        ("sin(3*x) / sin(5*x)", "3/5"),
        ("sin(x) / sin(2*x)", "1/2"),
        ("tan(x) / x", "1"),
        ("asin(x) / x", "1"),
        ("arctan(x) / x", "1"),
        ("sinh(x) / x", "1"),
        ("tanh(x) / x", "1"),
        ("sin(-3*x) / sin(x)", "-3"),
        ("tan(2*x) / sin(3*x)", "2/3"),
        ("(exp(x) - 1) / sin(x)", "1"),
        ("sin(x^2 - x) / x", "-1"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = try_limit_rules_at_finite(&mut ctx, expr, x, zero)
            .unwrap_or_else(|| panic!("equivalent-infinitesimal quotient must resolve: {source}"));
        assert_eq!(display_expr(&ctx, out), expected, "{source}");
    }
}

#[test]
fn finite_equivalent_infinitesimal_quotient_declines_unsound_shapes() {
    // Each must stay residual: higher-order (cos / cubic Taylor) and
    // sum-cancellation forms (first-order equivalents are invalid inside
    // a difference), cos (not a zero atom), a finite pole, and an atom
    // whose argument does NOT tend to 0 at the point.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    for source in [
        "(1 - cos(x)) / x^2",
        "(sin(x) - x) / x^3",
        "(tan(x) - x) / x^3",
        "cos(x) / sin(x)",
        "sin(x) / x^2",
        "sin(x + 1) / x",
    ] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            apply_finite_equivalent_infinitesimal_quotient_rule(&mut ctx, expr, x, zero).is_none(),
            "equivalent-infinitesimal quotient must decline: {source}"
        );
    }
}

#[test]
fn power_log_polynomial_dominance_resolves_antiderivative_endpoints() {
    // The one-sided limits of x^a ln(x)^b antiderivatives at 0+, which the
    // definite integrator needs to certify int_0^1 ln(x)^2 = 2 etc.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    for source in [
        "x * (ln(x)^2 - 2*ln(x) + 2)",
        "x^2 * (2*ln(x) - 1)",
        "2*sqrt(x)*ln(x) - 4*sqrt(x)",
        "x * (ln(x) - 1)",
        "x^(3/2) * ln(x)^3",
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out =
            try_limit_rules_at_finite_one_sided(&mut ctx, expr, x, zero, FiniteLimitSide::Right)
                .unwrap_or_else(|| panic!("power-log dominance must resolve: {source}"));
        assert_eq!(display_expr(&ctx, out), "0", "{source}");
    }
}

#[test]
fn power_log_polynomial_dominance_declines_non_vanishing() {
    // Each must NOT be folded to 0: no positive power (pure log diverges),
    // a bare constant term (tends to that constant), a negative power
    // (diverges), and a non-power/non-log factor.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    for source in ["ln(x)^2", "x * ln(x) + 5", "ln(x) / x", "sin(x) * ln(x)"] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            apply_finite_one_sided_power_log_polynomial_zero(
                &mut ctx,
                expr,
                x,
                zero,
                FiniteLimitSide::Right
            )
            .is_none(),
            "power-log dominance must decline: {source}"
        );
    }
}

#[test]
fn finite_exp_zero_quotient_limit_uses_removable_polynomial_ratio() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let point_zero = parse_expr(&mut ctx, "0");

    let cases = [
        ("(exp(x)-1)/x", 1, 1),
        ("(exp(2*x)-1)/x", 2, 1),
        ("3*(exp(2*x)-1)/(5*x)", 6, 5),
        ("(exp(x^2)-1)/x", 0, 1),
        ("(1-exp(2*x))/x", -2, 1),
    ];

    for (input, expected_num, expected_den) in cases {
        let expr = parse_expr(&mut ctx, input);
        let out = try_limit_rules_at_finite(&mut ctx, expr, x, point_zero)
            .unwrap_or_else(|| panic!("expected finite exp quotient limit for {input}"));
        let Expr::Number(value) = ctx.get(out) else {
            panic!("expected exact rational exp quotient limit for {input}");
        };
        assert_eq!(
            value,
            &BigRational::new(BigInt::from(expected_num), BigInt::from(expected_den))
        );
    }

    let point_one = parse_expr(&mut ctx, "1");
    let shifted = parse_expr(&mut ctx, "(exp(x - 1) - 1)/(x^2 - 1)");
    let shifted_out = try_limit_rules_at_finite(&mut ctx, shifted, x, point_one)
        .expect("expected shifted finite exp quotient limit");
    let Expr::Number(value) = ctx.get(shifted_out) else {
        panic!("expected exact shifted exp quotient limit");
    };
    assert_eq!(value, &BigRational::new(BigInt::from(1), BigInt::from(2)));

    let nonzero_argument = parse_expr(&mut ctx, "(exp(x + 1) - 1)/x");
    assert!(
        try_limit_rules_at_finite(&mut ctx, nonzero_argument, x, point_zero).is_none(),
        "exp quotient rule must only apply when the exponent tends to zero"
    );

    let finite_pole = parse_expr(&mut ctx, "(exp(x)-1)/x^2");
    assert!(
        try_limit_rules_at_finite(&mut ctx, finite_pole, x, point_zero).is_none(),
        "exp quotient rule must not promote finite poles"
    );
}

#[test]
fn finite_exp_combination_ratio_yields_ratio_of_derivatives() {
    // (sum exp)/(sum exp) at 0 -> N'(0)/D'(0), a ratio of log combinations.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    for (source, expected) in [
        ("(2^x-3^x)/(5^x-7^x)", "(ln(2) - ln(3)) / (ln(5) - ln(7))"),
        ("(2^x-3^x)/(2^x-5^x)", "(ln(2) - ln(3)) / (ln(2) - ln(5))"),
        ("(2^x-2^x)/(5^x-7^x)", "0"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = apply_finite_exp_combination_ratio_rule(&mut ctx, expr, x, zero)
            .unwrap_or_else(|| panic!("exp combination ratio must resolve: {source}"));
        assert_eq!(display_expr(&ctx, out), expected, "{source}");
    }
}

#[test]
fn finite_exp_combination_ratio_declines_cancelled_denominator() {
    // A denominator whose first derivative cancels to 0 - trivially
    // (5^x-5^x) or via a log identity (ln6 = ln2 + ln3) - is a higher-order
    // form and must stay residual, not divide by zero.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    for source in ["(2^x-3^x)/(5^x-5^x)", "(2^x-3^x)/(6^x-2^x-3^x+1)"] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            apply_finite_exp_combination_ratio_rule(&mut ctx, expr, x, zero).is_none(),
            "exp combination ratio must decline a cancelled denominator: {source}"
        );
    }
}

#[test]
fn finite_general_exp_ratio_yields_ratio_of_logs() {
    // (a^x - 1)/(b^x - 1) -> ln(a)/ln(b): ratio of first-order coefficients.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    for (source, expected) in [
        ("(3^x-1)/(2^x-1)", "ln(3) / ln(2)"),
        ("(2^x-1)/(3^x-1)", "ln(2) / ln(3)"),
        ("(2^(2*x)-1)/(2^x-1)", "2"),
        ("(3^x-1)/(3^x-1)", "1"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = apply_finite_general_exp_ratio_rule(&mut ctx, expr, x, zero)
            .unwrap_or_else(|| panic!("exp ratio must resolve: {source}"));
        assert_eq!(display_expr(&ctx, out), expected, "{source}");
    }
}

#[test]
fn finite_general_exp_ratio_declines_non_exp_denominator() {
    // A non-(b^x-1) denominator (sin, x) and a non-zero point decline.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    let one = parse_expr(&mut ctx, "1");
    for source in ["(2^x-1)/sin(x)", "(2^x-1)/x"] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            apply_finite_general_exp_ratio_rule(&mut ctx, expr, x, zero).is_none(),
            "exp ratio must decline: {source}"
        );
    }
    // The natural base is a PROVABLE base like any other: exp(x)
    // normalizes to e^x and the ratio resolves to ln(e)/ln(2) (folded
    // to 1/ln(2) downstream) — the rational-only decline is history.
    let expr = parse_expr(&mut ctx, "(exp(x)-1)/(2^x-1)");
    let out = apply_finite_general_exp_ratio_rule(&mut ctx, expr, x, zero)
        .expect("natural-base ratio resolves");
    assert_eq!(display_expr(&ctx, out), "ln(e) / ln(2)");
    let expr = parse_expr(&mut ctx, "(3^x-1)/(2^x-1)");
    assert!(
        apply_finite_general_exp_ratio_rule(&mut ctx, expr, x, one).is_none(),
        "exp ratio is the form at 0"
    );
}

#[test]
fn finite_general_exp_zero_quotient_yields_log_of_base() {
    // (a^g - 1)/h -> ln(a) lim(g/h): the derivative of a^x at 0.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    for (source, expected) in [
        ("(2^x - 1)/x", "ln(2)"),
        ("(3^x - 1)/x", "ln(3)"),
        ("(2^(3*x) - 1)/x", "3 * ln(2)"),
        ("(2^x - 1)/(2*x)", "1/2 * ln(2)"),
        ("(10^x - 1)/x", "ln(10)"),
        ("(1 - 5^x)/x", "-ln(5)"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = try_limit_rules_at_finite(&mut ctx, expr, x, zero)
            .unwrap_or_else(|| panic!("general exp quotient must resolve: {source}"));
        assert_eq!(display_expr(&ctx, out), expected, "{source}");
    }
}

#[test]
fn finite_general_exp_zero_quotient_declines_e_unit_base_and_poles() {
    // Base e is left to the exp rule; base 1 has no log; a finite pole and a
    // non-vanishing exponent stay residual via this rule.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    for source in ["(1^x - 1)/x", "(2^x - 1)/x^2", "(2^(x+1) - 1)/x"] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            apply_finite_general_exp_zero_quotient_rule(&mut ctx, expr, x, zero).is_none(),
            "general exp quotient must decline: {source}"
        );
    }
    // e^x - 1 is NOT matched here (base E is not a numeric rational base).
    let exp_form = parse_expr(&mut ctx, "(exp(x) - 1)/x");
    assert!(
        apply_finite_general_exp_zero_quotient_rule(&mut ctx, exp_form, x, zero).is_none(),
        "natural base is left to the exp rule"
    );
}

#[test]
fn finite_general_exp_rules_accept_provable_bases() {
    // (a^g - 1)/h with a provable constant base: the first-order
    // equivalent a^g - 1 ~ g·ln(a) needs only a provably positive
    // base != 1 (π, sqrt(2)) — never a rational-only base.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    for (source, expected) in [
        ("(pi^x - 1)/x", "ln(pi)"),
        ("(pi^(2*x) - 1)/x", "2 * ln(pi)"),
        ("(1 - pi^x)/x", "-ln(pi)"),
        ("(sqrt(2)^x - 1)/x", "ln(sqrt(2))"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = try_limit_rules_at_finite(&mut ctx, expr, x, zero)
            .unwrap_or_else(|| panic!("provable-base quotient must resolve: {source}"));
        assert_eq!(display_expr(&ctx, out), expected, "{source}");
    }
    // Ratio form: the natural base is allowed on a ratio side (ln(e)
    // folds to 1 downstream), closing (π^x-1)/(e^x-1) -> ln(π).
    let ratio = parse_expr(&mut ctx, "(pi^x - 1)/(e^x - 1)");
    let out =
        apply_finite_general_exp_ratio_rule(&mut ctx, ratio, x, zero).expect("provable-base ratio");
    assert_eq!(display_expr(&ctx, out), "ln(pi) / ln(e)");
    // Same provable base on both sides: the logs cancel exactly.
    let same = parse_expr(&mut ctx, "(pi^(2*x) - 1)/(pi^x - 1)");
    let out = apply_finite_general_exp_ratio_rule(&mut ctx, same, x, zero)
        .expect("same provable base ratio");
    assert_eq!(display_expr(&ctx, out), "2");
    // A disguised unit base declines: 2e/(e+e) = 1 unsimplified has no
    // provable position against 1, so no ln coefficient is emitted.
    let disguised = parse_expr(&mut ctx, "((2*e/(e+e))^x - 1)/x");
    assert!(
        apply_finite_general_exp_zero_quotient_rule(&mut ctx, disguised, x, zero).is_none(),
        "unprovable base must decline"
    );
}

#[test]
fn exact_log_combination_zero_decision() {
    // The exact kernel behind every zero gate on `K + Σ cᵢ·ln(bᵢ)`.
    let r = |n: i64, d: i64| BigRational::new(n.into(), d.into());
    let zero = BigRational::from_integer(0.into());
    // ln(4) - 2·ln(2) = 0 and 3·ln(8) - 9·ln(2) = 0 (the latter is the
    // float-error-prone shape once coefficients grow).
    assert_eq!(
        exact_log_combination_is_zero(&[(r(1, 1), r(4, 1)), (r(-2, 1), r(2, 1))], &zero),
        Some(true)
    );
    assert_eq!(
        exact_log_combination_is_zero(&[(r(3, 1), r(8, 1)), (r(-9, 1), r(2, 1))], &zero),
        Some(true)
    );
    // 2·ln(12) - ln(144) = 0: composite bases, decided by gcd refinement
    // alone (no primality anywhere).
    assert_eq!(
        exact_log_combination_is_zero(&[(r(2, 1), r(12, 1)), (r(-1, 1), r(144, 1))], &zero),
        Some(true)
    );
    // ln(12) - ln(18) = ln(2/3) != 0.
    assert_eq!(
        exact_log_combination_is_zero(&[(r(1, 1), r(12, 1)), (r(-1, 1), r(18, 1))], &zero),
        Some(false)
    );
    // ln(2) - 6931471805599453/10^16 ≈ -1.1e-17: NONZERO by Lindemann-
    // Weierstrass, exactly what the old 1e-12 float gate got wrong.
    let near_ln2 = BigRational::new(
        6931471805599453i64.into(),
        num_bigint::BigInt::from(10u64).pow(16),
    );
    assert_eq!(
        exact_log_combination_is_zero(&[(r(1, 1), r(2, 1))], &(-near_ln2)),
        Some(false)
    );
    // Degenerate combinations: no log terms — the constant decides.
    assert_eq!(exact_log_combination_is_zero(&[], &zero), Some(true));
    assert_eq!(exact_log_combination_is_zero(&[], &r(1, 1)), Some(false));
    // Fractional bases split into numerator/denominator contributions:
    // ln(2/3) + ln(3) - ln(2) = 0.
    assert_eq!(
        exact_log_combination_is_zero(
            &[(r(1, 1), r(2, 3)), (r(1, 1), r(3, 1)), (r(-1, 1), r(2, 1))],
            &zero
        ),
        Some(true)
    );
}

#[test]
fn finite_exp_combination_zero_gates_are_exact() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    // THE adversarial case for the old float gate: N'(0) = ln(2) − c with
    // c a 16-digit convergent of ln 2 — about −1.1e-17, but PROVABLY
    // nonzero. Must emit the expression, never fold to 0.
    let adversarial = parse_expr(
        &mut ctx,
        "(2^x - e^((6931471805599453/10000000000000000)*x))/(5^x - 7^x)",
    );
    let out = apply_finite_exp_combination_ratio_rule(&mut ctx, adversarial, x, zero)
        .expect("tiny-but-nonzero numerator must resolve");
    let shown = display_expr(&ctx, out);
    assert_ne!(shown, "0", "float fold would fabricate 0");
    assert!(shown.contains("ln(2)"), "kept the exact log part: {shown}");
    // Exactly-zero numerators DO fold — including composite bases that
    // need gcd refinement (4 = 2², 144 = 12²).
    for source in ["(4^x - 2^(2*x))/(3^x - 1)", "(12^(2*x) - 144^x)/(3^x - 1)"] {
        let expr = parse_expr(&mut ctx, source);
        let out = apply_finite_exp_combination_ratio_rule(&mut ctx, expr, x, zero)
            .unwrap_or_else(|| panic!("exact-zero numerator must fold: {source}"));
        assert_eq!(display_expr(&ctx, out), "0", "{source}");
    }
    // A zero-valued DENOMINATOR combination declines exactly (the ratio
    // is second-order): no fabricated division by zero, and no wrong 0
    // when the numerator is second-order too.
    for source in [
        "(3^x - 1)/(2^x + 2^(-x) - 2)",
        "(2^x + 2^(-x) - 2)/(3^x + 3^(-x) - 2)",
    ] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            apply_finite_exp_combination_ratio_rule(&mut ctx, expr, x, zero).is_none(),
            "zero-derivative denominator must decline: {source}"
        );
    }
    // Classic nonzero ratio unchanged.
    let classic = parse_expr(&mut ctx, "(2^x - 3^x)/(5^x - 7^x)");
    let out =
        apply_finite_exp_combination_ratio_rule(&mut ctx, classic, x, zero).expect("classic ratio");
    assert_eq!(display_expr(&ctx, out), "(ln(2) - ln(3)) / (ln(5) - ln(7))");
    // The linear-combination sibling folds exact zeros to a clean 0…
    let linear_zero = parse_expr(&mut ctx, "(12^(2*x) - 144^x)/x");
    let out =
        try_limit_rules_at_finite(&mut ctx, linear_zero, x, zero).expect("zero combination over x");
    assert_eq!(display_expr(&ctx, out), "0");
    // …and keeps tiny-but-nonzero derivatives as expressions.
    let linear_tiny = parse_expr(
        &mut ctx,
        "(2^x - e^((6931471805599453/10000000000000000)*x))/x",
    );
    let out =
        try_limit_rules_at_finite(&mut ctx, linear_tiny, x, zero).expect("tiny linear combination");
    let shown = display_expr(&ctx, out);
    assert_ne!(shown, "0");
    assert!(shown.contains("ln(2)"), "{shown}");
}

#[test]
fn finite_exp_linear_combination_yields_difference_of_logs() {
    // (a^x - b^x)/x -> ln(a) - ln(b): the derivative of a difference of
    // general-base exponentials at 0, where ln(a) is transcendental.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    for (source, expected) in [
        ("(2^x - 3^x)/x", "ln(2) - ln(3)"),
        ("(3^x - 2^x)/x", "ln(3) - ln(2)"),
        ("(2^(3*x) - 3^x)/x", "3 * ln(2) - ln(3)"),
        ("(5*2^x - 5*3^x)/x", "5 * ln(2) - 5 * ln(3)"),
        ("(exp(x) - 2^x)/x", "-ln(2) + 1"),
        ("(2*exp(x) - 2^x - 1)/x", "-ln(2) + 2"),
        ("(2^x + 3^x - 2)/x", "ln(2) + ln(3)"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = apply_finite_exp_linear_combination_quotient_rule(&mut ctx, expr, x, zero)
            .unwrap_or_else(|| panic!("exp combination must resolve: {source}"));
        assert_eq!(display_expr(&ctx, out), expected, "{source}");
    }
}

#[test]
fn finite_exp_linear_combination_declines_non_class_and_non_indeterminate() {
    // Honest declines: a non-vanishing numerator (not 0/0), a higher-order
    // denominator, a foreign-variable exponent, and a bare oscillation.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    let one = parse_expr(&mut ctx, "1");
    for source in [
        "(2^x + 3^x)/x",   // numerator -> 2, not 0/0
        "2^x/x",           // numerator -> 1, not 0/0
        "(2^x - 3^x)/x^2", // denominator vanishes to second order
        "(2^x - 3^y)/x",   // foreign-variable exponent is not a polynomial in x
        "sin(1/x)",        // not a quotient of this class (honesty list)
    ] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            apply_finite_exp_linear_combination_quotient_rule(&mut ctx, expr, x, zero).is_none(),
            "exp combination must decline: {source}"
        );
    }
    // Only defined at the origin (a^x -> 1 needs the exponent at 0).
    let expr = parse_expr(&mut ctx, "(2^x - 3^x)/x");
    assert!(
        apply_finite_exp_linear_combination_quotient_rule(&mut ctx, expr, x, one).is_none(),
        "exp combination is only defined at the origin"
    );
}

#[test]
fn finite_taylor_quotient_resolves_higher_order_zero_over_zero() {
    // Both sides vanish at 0; the limit is the ratio of leading Taylor
    // coefficients once the numerator's order matches the denominator's.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    for (source, expected) in [
        ("(1 - cos(x))/x^2", "1/2"),
        ("(sin(x) - x)/x^3", "-1/6"),
        ("(x - sin(x))/x^3", "1/6"),
        ("(tan(x) - x)/x^3", "1/3"),
        ("(exp(x) - 1 - x)/x^2", "1/2"),
        ("(cosh(x) - 1)/x^2", "1/2"),
        ("(sinh(x) - x)/x^3", "1/6"),
        ("(1 - cos(2*x))/x^2", "2"),
        ("(ln(1+x) - x)/x^2", "-1/2"),
        ("(arctan(x) - x)/x^3", "-1/3"),
        ("(arcsin(x) - x)/x^3", "1/6"),
        ("(1 - cos(x))/(x*sin(x))", "1/2"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = apply_finite_taylor_quotient_rule(&mut ctx, expr, x, zero)
            .unwrap_or_else(|| panic!("taylor quotient must resolve: {source}"));
        assert_eq!(display_expr(&ctx, out), expected, "{source}");
    }
}

#[test]
fn finite_taylor_quotient_declines_non_vanishing_and_unsupported() {
    // Honest declines: a numerator that does not out-vanish the denominator
    // (m < d), an oscillation whose argument does not tend to 0, a constant
    // numerator (den vanishes alone), and a nonzero approach point.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    let one = parse_expr(&mut ctx, "1");
    for source in [
        "(1 - cos(x))/x^3", // numerator order 2 < denominator order 3
        "sin(1/x)/x",       // argument 1/x does not tend to 0 (honesty list)
        "cos(x)/x",         // numerator does not vanish at 0
        "x/sin(1/x)",       // unsupported inner series
    ] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            apply_finite_taylor_quotient_rule(&mut ctx, expr, x, zero).is_none(),
            "taylor quotient must decline: {source}"
        );
    }
    // A nonzero approach point is out of scope for the at-zero series.
    let expr = parse_expr(&mut ctx, "(1 - cos(x))/x^2");
    assert!(
        apply_finite_taylor_quotient_rule(&mut ctx, expr, x, one).is_none(),
        "taylor quotient is only defined at the origin"
    );
}

#[test]
fn finite_lhopital_nonzero_point_resolves_shifted_zero_over_zero() {
    // 0/0 forms whose vanishing happens at a non-zero point: the at-zero
    // equivalent/Taylor rules cannot reach them, so L'Hôpital differentiates
    // and re-evaluates until the form is no longer 0/0. Values cross-checked
    // numerically (mpmath dps 40).
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (source, point_src, expected) in [
        ("sin(x)/(x-pi)", "pi", "-1"),
        ("(x-pi)/sin(x)", "pi", "-1"),
        ("tan(x)/sin(x)", "pi", "-1"),
        ("cos(x)/(x-pi/2)", "pi/2", "-1"),
        ("(1 - cos(x-1))/(x-1)^2", "1", "1/2"), // two applications
        ("(sin(x-1) - (x-1))/(x-1)^3", "1", "-1/6"), // three applications
        ("sin(x-3)/(x^2-9)", "3", "1/6"),
        ("ln(x)/(x-1)", "1", "1"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let point = parse_expr(&mut ctx, point_src);
        let out = apply_finite_lhopital_nonzero_point_quotient_rule(&mut ctx, expr, x, point)
            .unwrap_or_else(|| panic!("L'Hôpital must resolve {source} at {point_src}"));
        assert_eq!(display_expr(&ctx, out), expected, "{source} at {point_src}");
    }
}

#[test]
fn finite_lhopital_declines_poles_non_quotients_and_origin() {
    // Honest declines: a pole (numerator does not vanish while the
    // denominator does), an even-order pole that diverges bilaterally, a
    // non-0/0 quotient owned by ordinary substitution, an oscillation, and a
    // symbolic (non-rational) value which we leave residual rather than guess.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (source, point_src) in [
        ("1/(x-1)", "1"),               // simple pole, not 0/0
        ("1/sin(x)", "pi"),             // pole at a sine zero
        ("sin(x)/(x-pi)^2", "pi"),      // odd/even -> 1/(x-pi) blows up
        ("sin(1/(x-1))", "1"),          // inner oscillates, no limit
        ("cos(x)/cos(x)", "pi"),        // not 0/0 (continuous, owned elsewhere)
        ("(cos(x)-cos(2))/(x-2)", "2"), // value -sin(2) is not rational
    ] {
        let expr = parse_expr(&mut ctx, source);
        let point = parse_expr(&mut ctx, point_src);
        assert!(
            apply_finite_lhopital_nonzero_point_quotient_rule(&mut ctx, expr, x, point).is_none(),
            "L'Hôpital must decline {source} at {point_src}"
        );
    }
    // The origin is owned by the equivalent-infinitesimal / Taylor rules
    // (with their small-angle narration); L'Hôpital declines there.
    let zero = parse_expr(&mut ctx, "0");
    let expr = parse_expr(&mut ctx, "sin(x)/x");
    assert!(
        apply_finite_lhopital_nonzero_point_quotient_rule(&mut ctx, expr, x, zero).is_none(),
        "L'Hôpital declines at the origin"
    );
}

#[test]
fn finite_log_unit_quotient_limit_uses_removable_polynomial_ratio() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let point_zero = parse_expr(&mut ctx, "0");

    let cases = [
        ("ln(1+x)/x", 1, 1),
        ("ln(1+2*x)/x", 2, 1),
        ("3*ln(1+2*x)/(5*x)", 6, 5),
        ("ln(1+x^2)/x", 0, 1),
        ("-ln(1+2*x)/x", -2, 1),
    ];

    for (input, expected_num, expected_den) in cases {
        let expr = parse_expr(&mut ctx, input);
        let out = try_limit_rules_at_finite(&mut ctx, expr, x, point_zero)
            .unwrap_or_else(|| panic!("expected finite log unit quotient limit for {input}"));
        assert_number_expr(&ctx, out, expected_num, expected_den);
    }

    let fixed_base_cases = [
        ("log2(1+2*x)/x", 2, 1, 2, 1),
        ("3*log10(1+2*x)/(5*x)", 6, 5, 10, 1),
        ("log(3, 1+2*x)/x", 2, 1, 3, 1),
        ("5*log(1/2, 1+2*x)/(2*x)", 5, 1, 1, 2),
    ];

    for (input, expected_num, expected_den, expected_base_num, expected_base_den) in
        fixed_base_cases
    {
        let expr = parse_expr(&mut ctx, input);
        let out = try_limit_rules_at_finite(&mut ctx, expr, x, point_zero)
            .unwrap_or_else(|| panic!("expected fixed-base log unit quotient limit for {input}"));
        assert_ratio_over_ln_base(
            &ctx,
            out,
            expected_num,
            expected_den,
            expected_base_num,
            expected_base_den,
        );
    }

    let variable_base_cases = [
        ("log(x+2, 1+2*x)/x", 2, 1, 2, 1),
        ("log(x+1/4, 1+2*x)/x", 2, 1, 1, 4),
        ("5*log(x+3/2, 1+2*x)/(2*x)", 5, 1, 3, 2),
        ("log(exp(x)+2, 1+2*x)/x", 2, 1, 3, 1),
        ("log(sin(x)+2, 1+2*x)/x", 2, 1, 2, 1),
        ("log(sqrt(x+4)+1, 1+2*x)/x", 2, 1, 3, 1),
    ];

    for (input, expected_num, expected_den, expected_base_num, expected_base_den) in
        variable_base_cases
    {
        let expr = parse_expr(&mut ctx, input);
        let out = try_limit_rules_at_finite(&mut ctx, expr, x, point_zero).unwrap_or_else(|| {
            panic!("expected variable-base log unit quotient limit for {input}")
        });
        assert_ratio_over_ln_base(
            &ctx,
            out,
            expected_num,
            expected_den,
            expected_base_num,
            expected_base_den,
        );
    }

    let fixed_zero = parse_expr(&mut ctx, "log10(1+x^2)/x");
    let fixed_zero_out = try_limit_rules_at_finite(&mut ctx, fixed_zero, x, point_zero)
        .expect("expected zero fixed-base log unit quotient limit");
    assert_number_expr(&ctx, fixed_zero_out, 0, 1);

    let fixed_nonunit_argument = parse_expr(&mut ctx, "log2(2 + x)/x");
    assert!(
        try_limit_rules_at_finite(&mut ctx, fixed_nonunit_argument, x, point_zero).is_none(),
        "fixed-base log quotient rule must only apply when the log argument tends to one"
    );

    let fixed_finite_pole = parse_expr(&mut ctx, "log10(1+x)/x^2");
    assert!(
        try_limit_rules_at_finite(&mut ctx, fixed_finite_pole, x, point_zero).is_none(),
        "fixed-base log quotient rule must not promote finite poles"
    );

    let variable_base_one_log = parse_expr(&mut ctx, "log(x+1, 1+2*x)/x");
    assert!(
        try_limit_rules_at_finite(&mut ctx, variable_base_one_log, x, point_zero).is_none(),
        "binary log quotient rule must not promote variable-base quotients whose base tends to one"
    );

    let variable_base_zero_log = parse_expr(&mut ctx, "log(x, 1+2*x)/x");
    assert!(
        try_limit_rules_at_finite(&mut ctx, variable_base_zero_log, x, point_zero).is_none(),
        "binary log quotient rule must not promote variable-base quotients whose base tends to zero"
    );

    let variable_base_negative_log = parse_expr(&mut ctx, "log(x-1, 1+2*x)/x");
    assert!(
        try_limit_rules_at_finite(&mut ctx, variable_base_negative_log, x, point_zero)
            .is_none(),
        "binary log quotient rule must not promote variable-base quotients whose base tends negative"
    );

    let non_rational_variable_base_log = parse_expr(&mut ctx, "log(ln(x+3), 1+2*x)/x");
    assert!(
        try_limit_rules_at_finite(&mut ctx, non_rational_variable_base_log, x, point_zero)
            .is_none(),
        "binary log quotient rule must keep non-rational resolved variable bases residual"
    );

    let unit_base_log = parse_expr(&mut ctx, "log(1, 1+x)/x");
    let unit_base_log_out = try_limit_rules_at_finite(&mut ctx, unit_base_log, x, point_zero)
        .expect("constant-base log with base one has empty real domain");
    assert_eq!(
        display_expr(&ctx, unit_base_log_out),
        "undefined",
        "binary log quotient rule must reject base one as an empty real domain"
    );

    let negative_base_log = parse_expr(&mut ctx, "log(-2, 1+x)/x");
    let negative_base_log_out =
        try_limit_rules_at_finite(&mut ctx, negative_base_log, x, point_zero)
            .expect("constant-base log with negative base has empty real domain");
    assert_eq!(
        display_expr(&ctx, negative_base_log_out),
        "undefined",
        "binary log quotient rule must reject negative bases as an empty real domain"
    );

    let binary_nonunit_argument = parse_expr(&mut ctx, "log(3, 2+x)/x");
    assert!(
        try_limit_rules_at_finite(&mut ctx, binary_nonunit_argument, x, point_zero).is_none(),
        "binary log quotient rule must only apply when the log argument tends to one"
    );

    let point_one = parse_expr(&mut ctx, "1");
    let shifted = parse_expr(&mut ctx, "ln(x)/(x - 1)");
    let shifted_out = try_limit_rules_at_finite(&mut ctx, shifted, x, point_one)
        .expect("expected shifted finite log unit quotient limit");
    let Expr::Number(value) = ctx.get(shifted_out) else {
        panic!("expected exact shifted log unit quotient limit");
    };
    assert_eq!(value, &BigRational::one());

    let fixed_shifted = parse_expr(&mut ctx, "log2(x)/(x - 1)");
    let fixed_shifted_out = try_limit_rules_at_finite(&mut ctx, fixed_shifted, x, point_one)
        .expect("expected shifted fixed-base log unit quotient limit");
    assert_ratio_over_ln_base(&ctx, fixed_shifted_out, 1, 1, 2, 1);

    let quadratic_argument = parse_expr(&mut ctx, "ln(x^2)/(x - 1)");
    let quadratic_out = try_limit_rules_at_finite(&mut ctx, quadratic_argument, x, point_one)
        .expect("expected quadratic log unit quotient limit");
    let Expr::Number(value) = ctx.get(quadratic_out) else {
        panic!("expected exact quadratic log unit quotient limit");
    };
    assert_eq!(value, &BigRational::from_integer(BigInt::from(2)));

    let nonunit_argument = parse_expr(&mut ctx, "ln(2 + x)/x");
    assert!(
        try_limit_rules_at_finite(&mut ctx, nonunit_argument, x, point_zero).is_none(),
        "log quotient rule must only apply when the log argument tends to one"
    );

    let finite_pole = parse_expr(&mut ctx, "ln(1+x)/x^2");
    assert!(
        try_limit_rules_at_finite(&mut ctx, finite_pole, x, point_zero).is_none(),
        "log quotient rule must not promote finite poles"
    );
}

#[test]
fn finite_elementary_polynomial_limit_handles_total_real_functions() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let point = parse_expr(&mut ctx, "-1");

    let cases = [
        ("exp(x^2 + 1)", BuiltinFn::Exp),
        ("sin(x^2 + 1)", BuiltinFn::Sin),
        ("cos(x^2 + 1)", BuiltinFn::Cos),
        ("sinh(x^2 + 1)", BuiltinFn::Sinh),
        ("cosh(x^2 + 1)", BuiltinFn::Cosh),
        ("tanh(x^2 + 1)", BuiltinFn::Tanh),
        ("atan(x^2 + 1)", BuiltinFn::Atan),
        ("arctan(x^2 + 1)", BuiltinFn::Arctan),
        ("asinh(x^2 + 1)", BuiltinFn::Asinh),
        ("cbrt(x^2 + 1)", BuiltinFn::Cbrt),
    ];

    for (input, expected_builtin) in cases {
        let expr = parse_expr(&mut ctx, input);
        let out = apply_finite_elementary_polynomial_rule(&mut ctx, expr, x, point)
            .unwrap_or_else(|| panic!("expected finite elementary limit for {input}"));

        let Expr::Function(fn_id, args) = ctx.get(out).clone() else {
            panic!("expected function output for {input}");
        };
        assert_eq!(ctx.builtin_of(fn_id), Some(expected_builtin));
        assert_eq!(args.len(), 1);

        let Expr::Number(value) = ctx.get(args[0]) else {
            panic!("expected numeric function argument for {input}");
        };
        assert_eq!(value, &BigRational::from_integer(2.into()));
    }
}

#[test]
fn finite_elementary_polynomial_limit_evaluates_zero_special_values() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let point = parse_expr(&mut ctx, "0");

    let cases = [
        ("exp(x)", 1),
        ("sin(x)", 0),
        ("cos(x)", 1),
        ("sinh(x)", 0),
        ("cosh(x)", 1),
        ("tanh(x)", 0),
        ("atan(x)", 0),
        ("arctan(x)", 0),
        ("asinh(x)", 0),
        ("cbrt(x)", 0),
        ("abs(x)", 0),
    ];

    for (input, expected) in cases {
        let expr = parse_expr(&mut ctx, input);
        let out = apply_finite_elementary_polynomial_rule(&mut ctx, expr, x, point)
            .unwrap_or_else(|| panic!("expected finite elementary limit for {input}"));

        let Expr::Number(value) = ctx.get(out) else {
            panic!("expected numeric special value for {input}");
        };
        assert_eq!(value, &BigRational::from_integer(expected.into()));
    }
}

#[test]
fn finite_abs_polynomial_limit_evaluates_exact_rational_absolute_value() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let point = parse_expr(&mut ctx, "0");

    let cases = [("abs(x^2 - 1)", 1), ("abs(x - 2)", 2)];

    for (input, expected) in cases {
        let expr = parse_expr(&mut ctx, input);
        let out = apply_finite_elementary_polynomial_rule(&mut ctx, expr, x, point)
            .unwrap_or_else(|| panic!("expected finite abs polynomial limit for {input}"));

        let Expr::Number(value) = ctx.get(out) else {
            panic!("expected numeric absolute value for {input}");
        };
        assert_eq!(value, &BigRational::from_integer(expected.into()));
    }
}

#[test]
fn finite_real_cube_root_limit_evaluates_exact_and_symbolic_values() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");

    let point_neg_two = parse_expr(&mut ctx, "-2");
    let exact_builtin = parse_expr(&mut ctx, "cbrt(x^3)");
    let exact_builtin_out = try_limit_rules_at_finite(&mut ctx, exact_builtin, x, point_neg_two)
        .expect("expected exact finite cbrt limit");
    let Expr::Number(value) = ctx.get(exact_builtin_out) else {
        panic!("expected exact cbrt limit to collapse to a number");
    };
    assert_eq!(value, &BigRational::from_integer((-2).into()));

    let point_one = parse_expr(&mut ctx, "1");
    let exact_power = parse_expr(&mut ctx, "(x^2 - 9)^(1/3)");
    let exact_power_out = try_limit_rules_at_finite(&mut ctx, exact_power, x, point_one)
        .expect("expected exact finite one-third power limit");
    let Expr::Number(value) = ctx.get(exact_power_out) else {
        panic!("expected exact one-third power limit to collapse to a number");
    };
    assert_eq!(value, &BigRational::from_integer((-2).into()));

    let sqrt_power = parse_expr(&mut ctx, "(2*x + 3)^(1/2)");
    let sqrt_power_out = try_limit_rules_at_finite(&mut ctx, sqrt_power, x, point_one)
        .expect("expected finite square-root power limit");
    assert_eq!(display_expr(&ctx, sqrt_power_out), "sqrt(5)");

    let sqrt_power_endpoint = parse_expr(&mut ctx, "x^(1/2)");
    let point_zero = parse_expr(&mut ctx, "0");
    assert!(
        try_limit_rules_at_finite(&mut ctx, sqrt_power_endpoint, x, point_zero).is_none(),
        "finite square-root power endpoint must remain residual"
    );

    let point_neg_one = parse_expr(&mut ctx, "-1");
    let symbolic_builtin = parse_expr(&mut ctx, "cbrt(x^2 + 1)");
    let symbolic_builtin_out =
        try_limit_rules_at_finite(&mut ctx, symbolic_builtin, x, point_neg_one)
            .expect("expected symbolic finite cbrt limit");
    assert_eq!(display_expr(&ctx, symbolic_builtin_out), "cbrt(2)");
}

#[test]
fn finite_total_real_unary_composition_limit_reuses_resolved_sublimits() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");

    let point_zero = parse_expr(&mut ctx, "0");
    let nested_trig = parse_expr(&mut ctx, "cos(sin(x))");
    let nested_trig_out = try_limit_rules_at_finite(&mut ctx, nested_trig, x, point_zero)
        .expect("expected nested total-real trig finite limit");
    let Expr::Number(value) = ctx.get(nested_trig_out) else {
        panic!("expected cos(sin(x)) at 0 to collapse to a number");
    };
    assert_eq!(value, &BigRational::from_integer(1.into()));

    let point_neg_two = parse_expr(&mut ctx, "-2");
    let sin_sqrt = parse_expr(&mut ctx, "sin(sqrt(x^2 + 1))");
    let sin_sqrt_out = try_limit_rules_at_finite(&mut ctx, sin_sqrt, x, point_neg_two)
        .expect("expected total-real unary composition over safe sqrt sublimit");
    assert_eq!(display_expr(&ctx, sin_sqrt_out), "sin(sqrt(5))");

    let sin_special_angle = parse_expr(&mut ctx, "sin(x + pi/6)");
    let sin_special_angle_out =
        try_limit_rules_at_finite(&mut ctx, sin_special_angle, x, point_zero)
            .expect("expected sin over exact special-angle sublimit");
    assert_eq!(display_expr(&ctx, sin_special_angle_out), "1 / 2");

    let cos_special_angle = parse_expr(&mut ctx, "cos(x + pi/3)");
    let cos_special_angle_out =
        try_limit_rules_at_finite(&mut ctx, cos_special_angle, x, point_zero)
            .expect("expected cos over exact special-angle sublimit");
    assert_eq!(display_expr(&ctx, cos_special_angle_out), "1 / 2");

    let arctan_special_input = parse_expr(&mut ctx, "arctan(x + 1)");
    let arctan_special_input_out =
        try_limit_rules_at_finite(&mut ctx, arctan_special_input, x, point_zero)
            .expect("expected arctan over exact table input sublimit");
    assert_eq!(display_expr(&ctx, arctan_special_input_out), "pi / 4");

    let arctan_sqrt_special_input = parse_expr(&mut ctx, "arctan(sqrt(x + 3))");
    let arctan_sqrt_special_input_out =
        try_limit_rules_at_finite(&mut ctx, arctan_sqrt_special_input, x, point_zero)
            .expect("expected arctan over exact radical table input sublimit");
    assert_eq!(display_expr(&ctx, arctan_sqrt_special_input_out), "pi / 3");

    let exp_abs = parse_expr(&mut ctx, "exp(abs(x))");
    let exp_abs_out = try_limit_rules_at_finite(&mut ctx, exp_abs, x, point_neg_two)
        .expect("expected exp over resolved abs sublimit");
    let Expr::Function(fn_id, args) = ctx.get(exp_abs_out).clone() else {
        panic!("expected exp(abs(x)) finite limit to remain an exp function");
    };
    assert_eq!(ctx.builtin_of(fn_id), Some(BuiltinFn::Exp));
    assert_eq!(args.len(), 1);
    let Expr::Number(value) = ctx.get(args[0]) else {
        panic!("expected exp argument to be exact numeric absolute value");
    };
    assert_eq!(value, &BigRational::from_integer(2.into()));

    let point_eight = parse_expr(&mut ctx, "8");
    let sin_cbrt = parse_expr(&mut ctx, "sin(cbrt(x))");
    let sin_cbrt_out = try_limit_rules_at_finite(&mut ctx, sin_cbrt, x, point_eight)
        .expect("expected total-real unary composition over exact cbrt sublimit");
    assert_eq!(display_expr(&ctx, sin_cbrt_out), "sin(2)");
}

#[test]
fn finite_arithmetic_composition_folds_safe_numeric_and_structural_results() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let point_neg_two = parse_expr(&mut ctx, "-2");

    let numeric_sum = parse_expr(&mut ctx, "abs(x) + 1");
    let numeric_sum_out = try_limit_rules_at_finite(&mut ctx, numeric_sum, x, point_neg_two)
        .expect("expected safe numeric finite sum");
    let Expr::Number(value) = ctx.get(numeric_sum_out) else {
        panic!("expected exact numeric finite sum");
    };
    assert_eq!(value, &BigRational::from_integer(3.into()));

    let structural_zero = parse_expr(&mut ctx, "sqrt(x^2 + 1) - sqrt(x^2 + 1)");
    let structural_zero_out =
        try_limit_rules_at_finite(&mut ctx, structural_zero, x, point_neg_two)
            .expect("expected safe structural zero finite difference");
    let Expr::Number(value) = ctx.get(structural_zero_out) else {
        panic!("expected structural zero finite difference to fold");
    };
    assert_eq!(value, &BigRational::zero());

    let zero_quotient = parse_expr(&mut ctx, "(sqrt(x^2 + 1) - sqrt(x^2 + 1))/(abs(x) + 1)");
    let zero_quotient_out = try_limit_rules_at_finite(&mut ctx, zero_quotient, x, point_neg_two)
        .expect("expected safe zero quotient finite limit");
    let Expr::Number(value) = ctx.get(zero_quotient_out) else {
        panic!("expected safe zero quotient to fold");
    };
    assert_eq!(value, &BigRational::zero());

    let symbolic_sum = parse_expr(&mut ctx, "sqrt(x^2 + 1) + ln(x + 5)");
    let symbolic_sum_out = try_limit_rules_at_finite(&mut ctx, symbolic_sum, x, point_neg_two)
        .expect("expected safe symbolic finite sum");
    assert_eq!(display_expr(&ctx, symbolic_sum_out), "ln(3) + sqrt(5)");

    let unsafe_zero_product = parse_expr(&mut ctx, "0 * sqrt(x)");
    let point_zero = parse_expr(&mut ctx, "0");
    assert!(
        try_limit_rules_at_finite(&mut ctx, unsafe_zero_product, x, point_zero).is_none(),
        "zero product must not hide an unresolved finite sublimit"
    );
}

#[test]
fn finite_positive_domain_unary_composition_requires_positive_sublimit() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let point_neg_two = parse_expr(&mut ctx, "-2");

    let ln_sqrt = parse_expr(&mut ctx, "ln(sqrt(x^2 + 1))");
    let ln_sqrt_out = try_limit_rules_at_finite(&mut ctx, ln_sqrt, x, point_neg_two)
        .expect("expected ln over proven-positive sqrt sublimit");
    assert_eq!(display_expr(&ctx, ln_sqrt_out), "ln(sqrt(5))");

    let sqrt_abs_shift = parse_expr(&mut ctx, "sqrt(abs(x) + 1)");
    let sqrt_abs_shift_out = try_limit_rules_at_finite(&mut ctx, sqrt_abs_shift, x, point_neg_two)
        .expect("expected sqrt over positive arithmetic sublimit");
    assert_eq!(display_expr(&ctx, sqrt_abs_shift_out), "sqrt(3)");

    let ln_abs = parse_expr(&mut ctx, "ln(abs(x))");
    let ln_abs_out = try_limit_rules_at_finite(&mut ctx, ln_abs, x, point_neg_two)
        .expect("expected ln over positive abs sublimit");
    assert_eq!(display_expr(&ctx, ln_abs_out), "ln(2)");

    let log2_poly = parse_expr(&mut ctx, "log2(x^2 + 1)");
    let log2_poly_out = try_limit_rules_at_finite(&mut ctx, log2_poly, x, point_neg_two)
        .expect("expected log2 over positive polynomial argument");
    assert_eq!(display_expr(&ctx, log2_poly_out), "log2(5)");

    let log10_sqrt = parse_expr(&mut ctx, "log10(sqrt(x^2 + 1))");
    let log10_sqrt_out = try_limit_rules_at_finite(&mut ctx, log10_sqrt, x, point_neg_two)
        .expect("expected log10 over proven-positive sqrt sublimit");
    assert_eq!(display_expr(&ctx, log10_sqrt_out), "log10(sqrt(5))");

    let log2_abs = parse_expr(&mut ctx, "log2(abs(x))");
    let log2_abs_out = try_limit_rules_at_finite(&mut ctx, log2_abs, x, point_neg_two)
        .expect("expected log2 over positive abs sublimit");
    assert_eq!(display_expr(&ctx, log2_abs_out), "1");

    let point_zero = parse_expr(&mut ctx, "0");
    let sqrt_perfect_square_poly = parse_expr(&mut ctx, "sqrt(x^2 + 4*x + 4)");
    let sqrt_perfect_square_poly_out =
        try_limit_rules_at_finite(&mut ctx, sqrt_perfect_square_poly, x, point_zero)
            .expect("expected exact sqrt over positive rational square sublimit");
    assert_eq!(display_expr(&ctx, sqrt_perfect_square_poly_out), "2");

    let ln_one = parse_expr(&mut ctx, "ln(x^2 + 1)");
    let ln_one_out = try_limit_rules_at_finite(&mut ctx, ln_one, x, point_zero)
        .expect("expected exact ln(1) finite limit");
    assert_eq!(display_expr(&ctx, ln_one_out), "0");

    let point_e = parse_expr(&mut ctx, "e");
    let ln_e = parse_expr(&mut ctx, "ln(x)");
    let ln_e_out = try_limit_rules_at_finite(&mut ctx, ln_e, x, point_e)
        .expect("expected exact ln(e) finite limit");
    assert_eq!(display_expr(&ctx, ln_e_out), "1");

    let log2_one = parse_expr(&mut ctx, "log2(x^2 + 1)");
    let log2_one_out = try_limit_rules_at_finite(&mut ctx, log2_one, x, point_zero)
        .expect("expected exact log2(1) finite limit");
    assert_eq!(display_expr(&ctx, log2_one_out), "0");

    let log10_one = parse_expr(&mut ctx, "log10(x^2 + 1)");
    let log10_one_out = try_limit_rules_at_finite(&mut ctx, log10_one, x, point_zero)
        .expect("expected exact log10(1) finite limit");
    assert_eq!(display_expr(&ctx, log10_one_out), "0");

    let point_two = parse_expr(&mut ctx, "2");
    let log2_exact_power = parse_expr(&mut ctx, "log2(x^2 + 4)");
    let log2_exact_power_out = try_limit_rules_at_finite(&mut ctx, log2_exact_power, x, point_two)
        .expect("expected exact integer log2 finite limit");
    assert_eq!(display_expr(&ctx, log2_exact_power_out), "3");

    let log10_exact_power = parse_expr(&mut ctx, "log10(x^2 + 96)");
    let log10_exact_power_out =
        try_limit_rules_at_finite(&mut ctx, log10_exact_power, x, point_two)
            .expect("expected exact integer log10 finite limit");
    assert_eq!(display_expr(&ctx, log10_exact_power_out), "2");

    let exp_ln_abs = parse_expr(&mut ctx, "exp(ln(abs(x)))");
    let exp_ln_abs_out = try_limit_rules_at_finite(&mut ctx, exp_ln_abs, x, point_neg_two)
        .expect("expected exact exp(ln(g)) finite limit when g is positive");
    assert_eq!(display_expr(&ctx, exp_ln_abs_out), "2");

    let ln_exp_abs = parse_expr(&mut ctx, "ln(exp(abs(x)))");
    let ln_exp_abs_out = try_limit_rules_at_finite(&mut ctx, ln_exp_abs, x, point_neg_two)
        .expect("expected exact ln(exp(g)) finite limit");
    assert_eq!(display_expr(&ctx, ln_exp_abs_out), "2");

    let abs_sqrt = parse_expr(&mut ctx, "abs(sqrt(x^2 + 1))");
    let abs_sqrt_out = try_limit_rules_at_finite(&mut ctx, abs_sqrt, x, point_neg_two)
        .expect("expected exact abs over positive sqrt finite limit");
    assert_eq!(display_expr(&ctx, abs_sqrt_out), "sqrt(5)");

    let abs_neg_sqrt = parse_expr(&mut ctx, "abs(-sqrt(x^2 + 1))");
    let abs_neg_sqrt_out = try_limit_rules_at_finite(&mut ctx, abs_neg_sqrt, x, point_neg_two)
        .expect("expected exact abs over negative positive-sqrt finite limit");
    assert_eq!(display_expr(&ctx, abs_neg_sqrt_out), "sqrt(5)");

    let exp_ln_abs_zero = parse_expr(&mut ctx, "exp(ln(abs(x)))");
    assert!(
        try_limit_rules_at_finite(&mut ctx, exp_ln_abs_zero, x, point_zero).is_none(),
        "exp(ln(abs(x))) at zero must remain residual"
    );

    let sqrt_abs_zero = parse_expr(&mut ctx, "sqrt(abs(x))");
    assert!(
        try_limit_rules_at_finite(&mut ctx, sqrt_abs_zero, x, point_zero).is_none(),
        "sqrt over zero sublimit must remain residual"
    );

    let ln_sin_zero = parse_expr(&mut ctx, "ln(sin(x))");
    assert!(
        try_limit_rules_at_finite(&mut ctx, ln_sin_zero, x, point_zero).is_none(),
        "ln over zero sublimit must remain residual"
    );

    let log10_abs_zero = parse_expr(&mut ctx, "log10(abs(x))");
    assert!(
        try_limit_rules_at_finite(&mut ctx, log10_abs_zero, x, point_zero).is_none(),
        "log10 over zero sublimit must remain residual"
    );
}

#[test]
fn finite_partial_domain_unary_composition_requires_strict_interior_sublimit() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let point_zero = parse_expr(&mut ctx, "0");

    let arcsin_half = parse_expr(&mut ctx, "arcsin(x/2)");
    let arcsin_half_out = try_limit_rules_at_finite(&mut ctx, arcsin_half, x, point_zero)
        .expect("expected arcsin over strict interior numeric sublimit");
    assert_eq!(display_expr(&ctx, arcsin_half_out), "0");

    let atanh_half = parse_expr(&mut ctx, "atanh(x/2)");
    let atanh_half_out = try_limit_rules_at_finite(&mut ctx, atanh_half, x, point_zero)
        .expect("expected atanh over strict interior numeric sublimit");
    assert_eq!(display_expr(&ctx, atanh_half_out), "0");

    let acos_half = parse_expr(&mut ctx, "acos(x/2)");
    let acos_half_out = try_limit_rules_at_finite(&mut ctx, acos_half, x, point_zero)
        .expect("expected acos over strict interior numeric sublimit");
    assert_eq!(display_expr(&ctx, acos_half_out), "pi / 2");

    let arcsin_shifted_half = parse_expr(&mut ctx, "arcsin(x/2 + 1/2)");
    let arcsin_shifted_half_out =
        try_limit_rules_at_finite(&mut ctx, arcsin_shifted_half, x, point_zero)
            .expect("expected arcsin exact-table hit over strict interior sublimit");
    assert_eq!(display_expr(&ctx, arcsin_shifted_half_out), "pi / 6");

    let arccos_shifted_half = parse_expr(&mut ctx, "arccos(x/2 + 1/2)");
    let arccos_shifted_half_out =
        try_limit_rules_at_finite(&mut ctx, arccos_shifted_half, x, point_zero)
            .expect("expected arccos exact-table hit over strict interior sublimit");
    assert_eq!(display_expr(&ctx, arccos_shifted_half_out), "pi / 3");

    let acosh_abs_shift = parse_expr(&mut ctx, "acosh(abs(x) + 2)");
    let acosh_abs_shift_out = try_limit_rules_at_finite(&mut ctx, acosh_abs_shift, x, point_zero)
        .expect("expected acosh over strict interior numeric sublimit");
    assert_eq!(display_expr(&ctx, acosh_abs_shift_out), "acosh(2)");

    let point_one = parse_expr(&mut ctx, "1");
    let acosh_sqrt_affine = parse_expr(&mut ctx, "acosh(sqrt(2*x + 3))");
    let acosh_sqrt_affine_out =
        try_limit_rules_at_finite(&mut ctx, acosh_sqrt_affine, x, point_one)
            .expect("expected acosh over strict interior square-root sublimit");
    assert_eq!(display_expr(&ctx, acosh_sqrt_affine_out), "acosh(sqrt(5))");

    let acosh_sqrt_endpoint = parse_expr(&mut ctx, "acosh(sqrt(x))");
    assert!(
        try_limit_rules_at_finite(&mut ctx, acosh_sqrt_endpoint, x, point_one).is_none(),
        "acosh sqrt endpoint must remain residual"
    );

    let point_neg_five_four = parse_expr(&mut ctx, "-5/4");
    let atanh_sqrt_non_square = parse_expr(&mut ctx, "atanh(sqrt(2*x + 3))");
    let atanh_sqrt_non_square_out =
        try_limit_rules_at_finite(&mut ctx, atanh_sqrt_non_square, x, point_neg_five_four)
            .expect("expected atanh over strict interior square-root sublimit");
    assert_eq!(
        display_expr(&ctx, atanh_sqrt_non_square_out),
        "atanh(sqrt(1/2))"
    );

    let atanh_neg_sqrt_non_square = parse_expr(&mut ctx, "atanh(-sqrt(2*x + 3))");
    let atanh_neg_sqrt_non_square_out =
        try_limit_rules_at_finite(&mut ctx, atanh_neg_sqrt_non_square, x, point_neg_five_four)
            .expect("expected atanh over negated strict interior square-root sublimit");
    assert_eq!(
        display_expr(&ctx, atanh_neg_sqrt_non_square_out),
        "atanh(-sqrt(1/2))"
    );

    let arcsin_sqrt_non_square = parse_expr(&mut ctx, "arcsin(sqrt(2*x + 3))");
    let arcsin_sqrt_non_square_out =
        try_limit_rules_at_finite(&mut ctx, arcsin_sqrt_non_square, x, point_neg_five_four)
            .expect("expected arcsin over strict interior square-root sublimit");
    assert_eq!(display_expr(&ctx, arcsin_sqrt_non_square_out), "pi / 4");

    let acos_sqrt_non_square = parse_expr(&mut ctx, "acos(sqrt(2*x + 3))");
    let acos_sqrt_non_square_out =
        try_limit_rules_at_finite(&mut ctx, acos_sqrt_non_square, x, point_neg_five_four)
            .expect("expected acos over strict interior square-root sublimit");
    assert_eq!(display_expr(&ctx, acos_sqrt_non_square_out), "pi / 4");

    let atanh_sqrt_endpoint = parse_expr(&mut ctx, "atanh(sqrt(x))");
    assert!(
        try_limit_rules_at_finite(&mut ctx, atanh_sqrt_endpoint, x, point_one).is_none(),
        "atanh sqrt endpoint must remain residual"
    );

    let point_two = parse_expr(&mut ctx, "2");
    let arcsin_endpoint = parse_expr(&mut ctx, "arcsin(x)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, arcsin_endpoint, x, point_one).is_none(),
        "arcsin endpoint must remain residual"
    );
    let acos_endpoint = parse_expr(&mut ctx, "acos(x)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, acos_endpoint, x, point_one).is_none(),
        "acos one-sided-only endpoint must remain residual for bilateral limits"
    );
    let acos_even_gap_endpoint = parse_expr(&mut ctx, "acos(1 - x^2)");
    let acos_even_gap_endpoint_out =
        try_limit_rules_at_finite(&mut ctx, acos_even_gap_endpoint, x, point_zero)
            .expect("expected bilateral inverse-trig upper endpoint");
    assert_eq!(display_expr(&ctx, acos_even_gap_endpoint_out), "0");

    let arcsin_even_gap_endpoint = parse_expr(&mut ctx, "arcsin(1 - x^2)");
    let arcsin_even_gap_endpoint_out =
        try_limit_rules_at_finite(&mut ctx, arcsin_even_gap_endpoint, x, point_zero)
            .expect("expected bilateral arcsin upper endpoint");
    assert_eq!(display_expr(&ctx, arcsin_even_gap_endpoint_out), "pi / 2");

    let acos_shifted_even_gap_endpoint = parse_expr(&mut ctx, "acos(1 - (x - 2)^2)");
    let acos_shifted_even_gap_endpoint_out =
        try_limit_rules_at_finite(&mut ctx, acos_shifted_even_gap_endpoint, x, point_two)
            .expect("expected shifted bilateral inverse-trig upper endpoint");
    assert_eq!(display_expr(&ctx, acos_shifted_even_gap_endpoint_out), "0");

    let acos_odd_gap_endpoint = parse_expr(&mut ctx, "acos(1 - x^3)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, acos_odd_gap_endpoint, x, point_zero).is_none(),
        "one-sided-only inverse-trig upper endpoint must remain residual"
    );

    let acos_above_domain_endpoint = parse_expr(&mut ctx, "acos(1 + x^2)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, acos_above_domain_endpoint, x, point_zero).is_none(),
        "empty-punctured-domain inverse-trig upper endpoint must remain residual"
    );

    let acos_lower_even_gap_endpoint = parse_expr(&mut ctx, "acos(-1 + x^2)");
    let acos_lower_even_gap_endpoint_out =
        try_limit_rules_at_finite(&mut ctx, acos_lower_even_gap_endpoint, x, point_zero)
            .expect("expected bilateral inverse-trig lower endpoint");
    assert_eq!(display_expr(&ctx, acos_lower_even_gap_endpoint_out), "pi");

    let arcsin_lower_even_gap_endpoint = parse_expr(&mut ctx, "arcsin(-1 + x^2)");
    let arcsin_lower_even_gap_endpoint_out =
        try_limit_rules_at_finite(&mut ctx, arcsin_lower_even_gap_endpoint, x, point_zero)
            .expect("expected bilateral arcsin lower endpoint");
    assert_eq!(
        display_expr(&ctx, arcsin_lower_even_gap_endpoint_out),
        "-pi / 2"
    );

    let acos_shifted_lower_even_gap_endpoint = parse_expr(&mut ctx, "acos(-1 + (x - 2)^2)");
    let acos_shifted_lower_even_gap_endpoint_out =
        try_limit_rules_at_finite(&mut ctx, acos_shifted_lower_even_gap_endpoint, x, point_two)
            .expect("expected shifted bilateral inverse-trig lower endpoint");
    assert_eq!(
        display_expr(&ctx, acos_shifted_lower_even_gap_endpoint_out),
        "pi"
    );

    let acos_lower_odd_gap_endpoint = parse_expr(&mut ctx, "acos(-1 + x^3)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, acos_lower_odd_gap_endpoint, x, point_zero).is_none(),
        "one-sided-only inverse-trig lower endpoint must remain residual"
    );

    let acos_below_domain_endpoint = parse_expr(&mut ctx, "acos(-1 - x^2)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, acos_below_domain_endpoint, x, point_zero).is_none(),
        "empty-punctured-domain inverse-trig lower endpoint must remain residual"
    );
    let arcsin_out_of_domain = parse_expr(&mut ctx, "arcsin(x)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, arcsin_out_of_domain, x, point_two).is_none(),
        "arcsin out-of-domain point must remain residual"
    );
    let atanh_endpoint = parse_expr(&mut ctx, "atanh(x)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, atanh_endpoint, x, point_one).is_none(),
        "atanh endpoint must remain residual"
    );
    let acosh_endpoint = parse_expr(&mut ctx, "acosh(x)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, acosh_endpoint, x, point_one).is_none(),
        "acosh endpoint must remain residual"
    );

    let acosh_even_gap_endpoint = parse_expr(&mut ctx, "acosh(1 + x^2)");
    let acosh_even_gap_endpoint_out =
        try_limit_rules_at_finite(&mut ctx, acosh_even_gap_endpoint, x, point_zero)
            .expect("expected bilateral acosh lower-bound endpoint");
    assert_eq!(display_expr(&ctx, acosh_even_gap_endpoint_out), "0");

    let acosh_shifted_even_gap_endpoint = parse_expr(&mut ctx, "acosh(1 + (x - 2)^2)");
    let acosh_shifted_even_gap_endpoint_out =
        try_limit_rules_at_finite(&mut ctx, acosh_shifted_even_gap_endpoint, x, point_two)
            .expect("expected shifted bilateral acosh lower-bound endpoint");
    assert_eq!(display_expr(&ctx, acosh_shifted_even_gap_endpoint_out), "0");

    let acosh_odd_gap_endpoint = parse_expr(&mut ctx, "acosh(1 + x^3)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, acosh_odd_gap_endpoint, x, point_zero).is_none(),
        "one-sided-only acosh polynomial endpoint must remain residual"
    );

    let acosh_negative_gap_endpoint = parse_expr(&mut ctx, "acosh(1 - x^2)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, acosh_negative_gap_endpoint, x, point_zero).is_none(),
        "empty-punctured-domain acosh endpoint must remain residual"
    );
}

#[test]
fn finite_bilateral_sqrt_endpoint_requires_positive_tail_on_both_sides() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let point_zero = parse_expr(&mut ctx, "0");

    let even_gap = parse_expr(&mut ctx, "sqrt(x^2)");
    let even_gap_out = try_limit_rules_at_finite(&mut ctx, even_gap, x, point_zero)
        .expect("expected bilateral sqrt endpoint over positive even gap");
    assert_eq!(display_expr(&ctx, even_gap_out), "0");

    let point_one = parse_expr(&mut ctx, "1");
    let shifted_even_gap = parse_expr(&mut ctx, "sqrt((x - 1)^2)");
    let shifted_even_gap_out = try_limit_rules_at_finite(&mut ctx, shifted_even_gap, x, point_one)
        .expect("expected shifted bilateral sqrt endpoint over positive even gap");
    assert_eq!(display_expr(&ctx, shifted_even_gap_out), "0");

    let one_sided_only = parse_expr(&mut ctx, "sqrt(x)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, one_sided_only, x, point_zero).is_none(),
        "sqrt(x) at a finite endpoint must remain residual for bilateral limits"
    );

    let odd_gap = parse_expr(&mut ctx, "sqrt(x^3)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, odd_gap, x, point_zero).is_none(),
        "sqrt over an odd local tail must remain residual for bilateral limits"
    );

    let log_even_gap = parse_expr(&mut ctx, "ln(x^2)");
    let log_even_gap_out = try_limit_rules_at_finite(&mut ctx, log_even_gap, x, point_zero)
        .expect("expected bilateral log endpoint over positive even gap");
    assert_eq!(display_expr(&ctx, log_even_gap_out), "-infinity");

    let reciprocal_base_log_even_gap = parse_expr(&mut ctx, "log(1/2, x^2)");
    let reciprocal_base_log_even_gap_out =
        try_limit_rules_at_finite(&mut ctx, reciprocal_base_log_even_gap, x, point_zero)
            .expect("expected reciprocal-base bilateral log endpoint over positive even gap");
    assert_eq!(
        display_expr(&ctx, reciprocal_base_log_even_gap_out),
        "infinity"
    );

    let log_one_sided_only = parse_expr(&mut ctx, "ln(x)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, log_one_sided_only, x, point_zero).is_none(),
        "ln(x) at a finite endpoint must remain residual for bilateral limits"
    );

    let log_empty_punctured = parse_expr(&mut ctx, "ln(-x^2)");
    let log_empty_punctured_out =
        try_limit_rules_at_finite(&mut ctx, log_empty_punctured, x, point_zero)
            .expect("log over an empty real domain should be undefined");
    assert_eq!(
        display_expr(&ctx, log_empty_punctured_out),
        "undefined",
        "log over an empty real domain must be undefined"
    );
}

#[test]
fn finite_residual_warning_marks_empty_punctured_domains() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let point_zero = parse_expr(&mut ctx, "0");

    let empty_punctured = parse_expr(&mut ctx, "sqrt(-x^2)");
    let outcome = eval_limit_at_infinity(
        &mut ctx,
        empty_punctured,
        x,
        Approach::Finite(point_zero),
        &LimitOptions::default(),
    );
    assert_eq!(
        display_expr(&ctx, outcome.expr),
        "limit(sqrt(-(x^2)), x, 0)"
    );
    let warning = outcome.warning.expect("expected residual warning");
    assert!(warning.contains(FINITE_POINT_LIMIT_UNSUPPORTED_WARNING));
    assert!(warning.contains("no punctured real neighbourhood"));

    let empty_punctured_acosh = parse_expr(&mut ctx, "acosh(1 - x^2)");
    let acosh_outcome = eval_limit_at_infinity(
        &mut ctx,
        empty_punctured_acosh,
        x,
        Approach::Finite(point_zero),
        &LimitOptions::default(),
    );
    assert_eq!(
        display_expr(&ctx, acosh_outcome.expr),
        "limit(acosh(1 - x^2), x, 0)"
    );
    let acosh_warning = acosh_outcome.warning.expect("expected residual warning");
    assert!(acosh_warning.contains(FINITE_POINT_LIMIT_UNSUPPORTED_WARNING));
    assert!(acosh_warning.contains("no punctured real neighbourhood"));

    let empty_punctured_inverse_trig = parse_expr(&mut ctx, "acos(1 + x^2)");
    let inverse_trig_outcome = eval_limit_at_infinity(
        &mut ctx,
        empty_punctured_inverse_trig,
        x,
        Approach::Finite(point_zero),
        &LimitOptions::default(),
    );
    assert_eq!(
        display_expr(&ctx, inverse_trig_outcome.expr),
        "limit(acos(x^2 + 1), x, 0)"
    );
    let inverse_trig_warning = inverse_trig_outcome
        .warning
        .expect("expected residual warning");
    assert!(inverse_trig_warning.contains(FINITE_POINT_LIMIT_UNSUPPORTED_WARNING));
    assert!(inverse_trig_warning.contains("no punctured real neighbourhood"));

    let empty_punctured_inverse_trig_lower = parse_expr(&mut ctx, "acos(-1 - x^2)");
    let inverse_trig_lower_outcome = eval_limit_at_infinity(
        &mut ctx,
        empty_punctured_inverse_trig_lower,
        x,
        Approach::Finite(point_zero),
        &LimitOptions::default(),
    );
    let inverse_trig_lower_warning = inverse_trig_lower_outcome
        .warning
        .expect("expected residual warning");
    assert!(inverse_trig_lower_warning.contains(FINITE_POINT_LIMIT_UNSUPPORTED_WARNING));
    assert!(inverse_trig_lower_warning.contains("no punctured real neighbourhood"));

    let one_sided_only = parse_expr(&mut ctx, "sqrt(x^3)");
    let one_sided_outcome = eval_limit_at_infinity(
        &mut ctx,
        one_sided_only,
        x,
        Approach::Finite(point_zero),
        &LimitOptions::default(),
    );
    let one_sided_warning = one_sided_outcome
        .warning
        .expect("expected generic residual warning");
    assert!(one_sided_warning.contains(FINITE_POINT_LIMIT_UNSUPPORTED_WARNING));
    assert!(!one_sided_warning.contains("no punctured real neighbourhood"));

    let one_sided_only_acosh = parse_expr(&mut ctx, "acosh(1 + x^3)");
    let one_sided_acosh_outcome = eval_limit_at_infinity(
        &mut ctx,
        one_sided_only_acosh,
        x,
        Approach::Finite(point_zero),
        &LimitOptions::default(),
    );
    let one_sided_acosh_warning = one_sided_acosh_outcome
        .warning
        .expect("expected generic residual warning");
    assert!(one_sided_acosh_warning.contains(FINITE_POINT_LIMIT_UNSUPPORTED_WARNING));
    assert!(!one_sided_acosh_warning.contains("no punctured real neighbourhood"));

    let one_sided_only_inverse_trig = parse_expr(&mut ctx, "acos(1 - x^3)");
    let one_sided_inverse_trig_outcome = eval_limit_at_infinity(
        &mut ctx,
        one_sided_only_inverse_trig,
        x,
        Approach::Finite(point_zero),
        &LimitOptions::default(),
    );
    let one_sided_inverse_trig_warning = one_sided_inverse_trig_outcome
        .warning
        .expect("expected generic residual warning");
    assert!(one_sided_inverse_trig_warning.contains(FINITE_POINT_LIMIT_UNSUPPORTED_WARNING));
    assert!(!one_sided_inverse_trig_warning.contains("no punctured real neighbourhood"));

    let positive_even_gap = parse_expr(&mut ctx, "sqrt(x^2)");
    let resolved_outcome = eval_limit_at_infinity(
        &mut ctx,
        positive_even_gap,
        x,
        Approach::Finite(point_zero),
        &LimitOptions::default(),
    );
    assert_eq!(display_expr(&ctx, resolved_outcome.expr), "0");
    assert!(resolved_outcome.warning.is_none());
}

#[test]
fn finite_domain_checked_trig_unary_composition_accepts_defined_table_sublimits() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let point_zero = parse_expr(&mut ctx, "0");

    let tan_half = parse_expr(&mut ctx, "tan(x/2)");
    let tan_half_out = try_limit_rules_at_finite(&mut ctx, tan_half, x, point_zero)
        .expect("expected tan over exact zero sublimit");
    assert_eq!(display_expr(&ctx, tan_half_out), "0");

    let sec_square = parse_expr(&mut ctx, "sec(x^2)");
    let sec_square_out = try_limit_rules_at_finite(&mut ctx, sec_square, x, point_zero)
        .expect("expected sec over exact zero sublimit");
    assert_eq!(display_expr(&ctx, sec_square_out), "1");

    let tan_sin = parse_expr(&mut ctx, "tan(sin(x))");
    let tan_sin_out = try_limit_rules_at_finite(&mut ctx, tan_sin, x, point_zero)
        .expect("expected tan over resolved zero sin sublimit");
    assert_eq!(display_expr(&ctx, tan_sin_out), "0");

    let sec_abs = parse_expr(&mut ctx, "sec(abs(x))");
    let sec_abs_out = try_limit_rules_at_finite(&mut ctx, sec_abs, x, point_zero)
        .expect("expected sec over resolved zero abs sublimit");
    assert_eq!(display_expr(&ctx, sec_abs_out), "1");

    let tan_special_angle = parse_expr(&mut ctx, "tan(x + pi/4)");
    let tan_special_angle_out =
        try_limit_rules_at_finite(&mut ctx, tan_special_angle, x, point_zero)
            .expect("expected tan over defined special-angle sublimit");
    assert_eq!(display_expr(&ctx, tan_special_angle_out), "1");

    let sec_special_angle = parse_expr(&mut ctx, "sec(x + pi/3)");
    let sec_special_angle_out =
        try_limit_rules_at_finite(&mut ctx, sec_special_angle, x, point_zero)
            .expect("expected sec over defined special-angle sublimit");
    assert_eq!(display_expr(&ctx, sec_special_angle_out), "2");

    let csc_special_angle = parse_expr(&mut ctx, "csc(x + pi/6)");
    let csc_special_angle_out =
        try_limit_rules_at_finite(&mut ctx, csc_special_angle, x, point_zero)
            .expect("expected csc over defined special-angle sublimit");
    assert_eq!(display_expr(&ctx, csc_special_angle_out), "2");

    let cot_special_angle = parse_expr(&mut ctx, "cot(x + pi/4)");
    let cot_special_angle_out =
        try_limit_rules_at_finite(&mut ctx, cot_special_angle, x, point_zero)
            .expect("expected cot over defined special-angle sublimit");
    assert_eq!(display_expr(&ctx, cot_special_angle_out), "1");

    let tan_pole_angle = parse_expr(&mut ctx, "tan(x + pi/2)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, tan_pole_angle, x, point_zero).is_none(),
        "tan at a table-undefined pole must remain residual"
    );

    let sec_pole_angle = parse_expr(&mut ctx, "sec(x + pi/2)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, sec_pole_angle, x, point_zero).is_none(),
        "sec at a table-undefined pole must remain residual"
    );

    let csc_pole_angle = parse_expr(&mut ctx, "csc(x + pi)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, csc_pole_angle, x, point_zero).is_none(),
        "csc at a table-undefined pole must remain residual"
    );

    let cot_pole_angle = parse_expr(&mut ctx, "cot(x + pi)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, cot_pole_angle, x, point_zero).is_none(),
        "cot at a table-undefined pole must remain residual"
    );

    let point_one = parse_expr(&mut ctx, "1");
    let tan_nonzero = parse_expr(&mut ctx, "tan(x)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, tan_nonzero, x, point_one).is_none(),
        "tan at nonzero rational sublimit must remain residual without pole proof"
    );
    let sec_nonzero = parse_expr(&mut ctx, "sec(x)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, sec_nonzero, x, point_one).is_none(),
        "sec at nonzero rational sublimit must remain residual without pole proof"
    );
    let csc_zero = parse_expr(&mut ctx, "csc(x)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, csc_zero, x, point_zero).is_none(),
        "csc at zero must remain residual"
    );
    let cot_zero = parse_expr(&mut ctx, "cot(x)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, cot_zero, x, point_zero).is_none(),
        "cot at zero must remain residual"
    );
}

#[test]
fn finite_binary_log_composition_requires_valid_base_and_positive_sublimits() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let point_neg_two = parse_expr(&mut ctx, "-2");

    let binary_log_poly = parse_expr(&mut ctx, "log(2, x^2 + 1)");
    let binary_log_poly_out =
        try_limit_rules_at_finite(&mut ctx, binary_log_poly, x, point_neg_two)
            .expect("expected constant-base log over positive polynomial argument");
    assert_eq!(display_expr(&ctx, binary_log_poly_out), "log(2, 5)");

    let binary_log_sqrt = parse_expr(&mut ctx, "log(1/2, sqrt(x^2 + 1))");
    let binary_log_sqrt_out =
        try_limit_rules_at_finite(&mut ctx, binary_log_sqrt, x, point_neg_two)
            .expect("expected constant-base log over proven-positive sqrt sublimit");
    assert_eq!(
        display_expr(&ctx, binary_log_sqrt_out),
        "log(1 / 2, sqrt(5))"
    );

    let binary_log_abs = parse_expr(&mut ctx, "log(2, abs(x))");
    let binary_log_abs_out = try_limit_rules_at_finite(&mut ctx, binary_log_abs, x, point_neg_two)
        .expect("expected constant-base log over positive abs sublimit");
    assert_eq!(display_expr(&ctx, binary_log_abs_out), "1");

    let point_zero = parse_expr(&mut ctx, "0");
    let binary_log_arg_one = parse_expr(&mut ctx, "log(2, x^2 + 1)");
    let binary_log_arg_one_out =
        try_limit_rules_at_finite(&mut ctx, binary_log_arg_one, x, point_zero)
            .expect("expected exact binary log of one finite limit");
    assert_eq!(display_expr(&ctx, binary_log_arg_one_out), "0");

    let point_two = parse_expr(&mut ctx, "2");
    let binary_log_integer_power = parse_expr(&mut ctx, "log(2, x^2 + 4)");
    let binary_log_integer_power_out =
        try_limit_rules_at_finite(&mut ctx, binary_log_integer_power, x, point_two)
            .expect("expected exact integer binary log finite limit");
    assert_eq!(display_expr(&ctx, binary_log_integer_power_out), "3");

    let binary_log_negative_integer_power = parse_expr(&mut ctx, "log(1/2, x^2 + 4)");
    let binary_log_negative_integer_power_out =
        try_limit_rules_at_finite(&mut ctx, binary_log_negative_integer_power, x, point_two)
            .expect("expected exact negative-integer binary log finite limit");
    assert_eq!(
        display_expr(&ctx, binary_log_negative_integer_power_out),
        "-3"
    );

    let binary_log_fractional_power = parse_expr(&mut ctx, "log(4, x^2 + 4)");
    let binary_log_fractional_power_out =
        try_limit_rules_at_finite(&mut ctx, binary_log_fractional_power, x, point_two)
            .expect("expected exact rational-exponent binary log finite limit");
    assert_eq!(display_expr(&ctx, binary_log_fractional_power_out), "3/2");

    let binary_log_negative_fractional_power = parse_expr(&mut ctx, "log(1/4, x^2 + 4)");
    let binary_log_negative_fractional_power_out =
        try_limit_rules_at_finite(&mut ctx, binary_log_negative_fractional_power, x, point_two)
            .expect("expected exact negative rational-exponent binary log finite limit");
    assert_eq!(
        display_expr(&ctx, binary_log_negative_fractional_power_out),
        "-3/2"
    );

    let binary_log_two_thirds = parse_expr(&mut ctx, "log(27, x^2 + 5)");
    let binary_log_two_thirds_out =
        try_limit_rules_at_finite(&mut ctx, binary_log_two_thirds, x, point_two)
            .expect("expected exact two-thirds binary log finite limit");
    assert_eq!(display_expr(&ctx, binary_log_two_thirds_out), "2/3");

    let binary_log_not_exact = parse_expr(&mut ctx, "log(2, x^2 + 1)");
    let binary_log_not_exact_out =
        try_limit_rules_at_finite(&mut ctx, binary_log_not_exact, x, point_two)
            .expect("expected safe binary log finite limit without exact rational fold");
    assert_eq!(display_expr(&ctx, binary_log_not_exact_out), "log(2, 5)");

    let variable_base_log_poly = parse_expr(&mut ctx, "log(x^2 + 3, x^2 + 1)");
    let variable_base_log_poly_out =
        try_limit_rules_at_finite(&mut ctx, variable_base_log_poly, x, point_neg_two)
            .expect("expected log over safe finite base and argument sublimits");
    assert_eq!(display_expr(&ctx, variable_base_log_poly_out), "log(7, 5)");

    let variable_base_log_sqrt = parse_expr(&mut ctx, "log(x^2 + 3, sqrt(x^2 + 1))");
    let variable_base_log_sqrt_out =
        try_limit_rules_at_finite(&mut ctx, variable_base_log_sqrt, x, point_neg_two)
            .expect("expected log over safe finite base and positive sqrt argument sublimit");
    assert_eq!(
        display_expr(&ctx, variable_base_log_sqrt_out),
        "log(7, sqrt(5))"
    );

    let point_neg_one = parse_expr(&mut ctx, "-1");
    let variable_base_log_same = parse_expr(&mut ctx, "log(x^2 + 3, x^2 + 3)");
    let variable_base_log_same_out =
        try_limit_rules_at_finite(&mut ctx, variable_base_log_same, x, point_neg_one)
            .expect("expected exact binary log with equal finite base and argument");
    assert_eq!(display_expr(&ctx, variable_base_log_same_out), "1");

    let binary_log_abs_zero = parse_expr(&mut ctx, "log(2, abs(x))");
    assert!(
        try_limit_rules_at_finite(&mut ctx, binary_log_abs_zero, x, point_zero).is_none(),
        "constant-base log over zero sublimit must remain residual"
    );

    let log_base_one = parse_expr(&mut ctx, "log(1, x^2 + 1)");
    let log_base_one_out = try_limit_rules_at_finite(&mut ctx, log_base_one, x, point_neg_two)
        .expect("constant-base log with base one has empty real domain");
    assert_eq!(display_expr(&ctx, log_base_one_out), "undefined");

    let log_negative_base = parse_expr(&mut ctx, "log(-2, x^2 + 1)");
    let log_negative_base_out =
        try_limit_rules_at_finite(&mut ctx, log_negative_base, x, point_neg_two)
            .expect("constant-base log with negative base has empty real domain");
    assert_eq!(display_expr(&ctx, log_negative_base_out), "undefined");

    let log_variable_base_one = parse_expr(&mut ctx, "log(x^2 - 3, x^2 + 1)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, log_variable_base_one, x, point_neg_two).is_none(),
        "variable-base log with base sublimit one must remain residual"
    );

    let log_variable_base_zero = parse_expr(&mut ctx, "log(x^2 - 4, x^2 + 1)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, log_variable_base_zero, x, point_neg_two).is_none(),
        "variable-base log with zero base sublimit must remain residual"
    );
}

#[test]
fn finite_integer_power_composition_requires_safe_sublimit_and_nonzero_base_when_needed() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let point_neg_two = parse_expr(&mut ctx, "-2");

    let numeric_positive_power = parse_expr(&mut ctx, "(abs(x) + 1)^2");
    let numeric_positive_power_out =
        try_limit_rules_at_finite(&mut ctx, numeric_positive_power, x, point_neg_two)
            .expect("expected integer power over exact numeric sublimit");
    let Expr::Number(value) = ctx.get(numeric_positive_power_out) else {
        panic!("expected exact integer power to fold to a number");
    };
    assert_eq!(value, &BigRational::from_integer(9.into()));

    let symbolic_positive_power = parse_expr(&mut ctx, "(sqrt(x^2 + 1))^2");
    let symbolic_positive_power_out =
        try_limit_rules_at_finite(&mut ctx, symbolic_positive_power, x, point_neg_two)
            .expect("expected integer power over safe symbolic sublimit");
    let Expr::Number(value) = ctx.get(symbolic_positive_power_out) else {
        panic!("expected even power over exact sqrt sublimit to fold to a number");
    };
    assert_eq!(value, &BigRational::from_integer(5.into()));

    let symbolic_odd_power = parse_expr(&mut ctx, "(sqrt(x^2 + 1))^3");
    let symbolic_odd_power_out =
        try_limit_rules_at_finite(&mut ctx, symbolic_odd_power, x, point_neg_two)
            .expect("expected odd integer power over safe symbolic sublimit");
    assert_eq!(display_expr(&ctx, symbolic_odd_power_out), "sqrt(5)^3");

    let numeric_negative_power = parse_expr(&mut ctx, "(abs(x) + 1)^(-2)");
    let numeric_negative_power_out =
        try_limit_rules_at_finite(&mut ctx, numeric_negative_power, x, point_neg_two)
            .expect("expected negative integer power over nonzero numeric sublimit");
    let Expr::Number(value) = ctx.get(numeric_negative_power_out) else {
        panic!("expected exact negative integer power to fold to a number");
    };
    assert_eq!(value, &BigRational::new(BigInt::from(1), BigInt::from(9)));

    let symbolic_negative_power = parse_expr(&mut ctx, "(sqrt(x^2 + 1))^(-1)");
    let symbolic_negative_power_out =
        try_limit_rules_at_finite(&mut ctx, symbolic_negative_power, x, point_neg_two)
            .expect("expected negative integer power over proven nonzero symbolic sublimit");
    assert_eq!(
        display_expr(&ctx, symbolic_negative_power_out),
        "1 / sqrt(5)"
    );

    let symbolic_negative_square_power = parse_expr(&mut ctx, "(sqrt(x^2 + 1))^(-2)");
    let symbolic_negative_square_power_out =
        try_limit_rules_at_finite(&mut ctx, symbolic_negative_square_power, x, point_neg_two)
            .expect("expected negative even power over exact sqrt sublimit to fold");
    let Expr::Number(value) = ctx.get(symbolic_negative_square_power_out) else {
        panic!("expected negative even power over exact sqrt sublimit to fold to a number");
    };
    assert_eq!(value, &BigRational::new(BigInt::from(1), BigInt::from(5)));

    let point_zero = parse_expr(&mut ctx, "0");
    let affine_root_negative_square_power = parse_expr(&mut ctx, "(sqrt(x + 4))^(-2)");
    let affine_root_negative_square_power_out =
        try_limit_rules_at_finite(&mut ctx, affine_root_negative_square_power, x, point_zero)
            .expect("expected negative even power over positive affine sqrt sublimit to fold");
    let Expr::Number(value) = ctx.get(affine_root_negative_square_power_out) else {
        panic!("expected negative even power over affine sqrt sublimit to fold to a number");
    };
    assert_eq!(value, &BigRational::new(BigInt::from(1), BigInt::from(4)));

    let point_neg_one = parse_expr(&mut ctx, "-1");
    let cbrt_cube_power = parse_expr(&mut ctx, "(cbrt(x^2 + 1))^3");
    let cbrt_cube_power_out =
        try_limit_rules_at_finite(&mut ctx, cbrt_cube_power, x, point_neg_one)
            .expect("expected cube power over exact cbrt sublimit to fold");
    let Expr::Number(value) = ctx.get(cbrt_cube_power_out) else {
        panic!("expected cube power over exact cbrt sublimit to fold to a number");
    };
    assert_eq!(value, &BigRational::from_integer(2.into()));

    let cbrt_square_power = parse_expr(&mut ctx, "(cbrt(x^2 + 1))^2");
    let cbrt_square_power_out =
        try_limit_rules_at_finite(&mut ctx, cbrt_square_power, x, point_neg_one)
            .expect("expected non-multiple cbrt power to remain explicit");
    assert_eq!(display_expr(&ctx, cbrt_square_power_out), "cbrt(2)^2");

    let cbrt_negative_cube_power = parse_expr(&mut ctx, "(cbrt(x^2 + 1))^(-3)");
    let cbrt_negative_cube_power_out =
        try_limit_rules_at_finite(&mut ctx, cbrt_negative_cube_power, x, point_neg_one)
            .expect("expected negative cube power over exact nonzero cbrt sublimit to fold");
    let Expr::Number(value) = ctx.get(cbrt_negative_cube_power_out) else {
        panic!("expected negative cube power over exact cbrt sublimit to fold to a number");
    };
    assert_eq!(value, &BigRational::new(BigInt::from(1), BigInt::from(2)));

    let cbrt_zero_power = parse_expr(&mut ctx, "(cbrt(x^2 + 1))^0");
    let cbrt_zero_power_out =
        try_limit_rules_at_finite(&mut ctx, cbrt_zero_power, x, point_neg_one)
            .expect("expected zero power over nonzero cbrt sublimit to fold");
    let Expr::Number(value) = ctx.get(cbrt_zero_power_out) else {
        panic!("expected zero power over nonzero cbrt sublimit to fold to one");
    };
    assert_eq!(value, &BigRational::one());

    let numeric_zero_power = parse_expr(&mut ctx, "(abs(x) + 1)^0");
    let numeric_zero_power_out =
        try_limit_rules_at_finite(&mut ctx, numeric_zero_power, x, point_neg_two)
            .expect("expected zero power over nonzero sublimit");
    let Expr::Number(value) = ctx.get(numeric_zero_power_out) else {
        panic!("expected safe zero power to fold to one");
    };
    assert_eq!(value, &BigRational::one());

    let zero_base_negative_power = parse_expr(&mut ctx, "(abs(x) - 2)^(-1)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, zero_base_negative_power, x, point_neg_two).is_none(),
        "negative integer power over zero sublimit must remain residual"
    );

    let zero_base_zero_power = parse_expr(&mut ctx, "abs(x)^0");
    assert!(
        try_limit_rules_at_finite(&mut ctx, zero_base_zero_power, x, point_zero).is_none(),
        "zero power over zero sublimit must remain residual"
    );

    let zero_cbrt_base_negative_power = parse_expr(&mut ctx, "cbrt(x)^(-3)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, zero_cbrt_base_negative_power, x, point_zero).is_none(),
        "negative cube power over zero cbrt sublimit must remain residual"
    );

    let unresolved_base_power = parse_expr(&mut ctx, "sqrt(x)^2");
    assert!(
        try_limit_rules_at_finite(&mut ctx, unresolved_base_power, x, point_zero).is_none(),
        "integer power must not hide an unresolved finite base sublimit"
    );

    let unresolved_base_negative_power = parse_expr(&mut ctx, "sqrt(x)^(-2)");
    assert!(
        try_limit_rules_at_finite(&mut ctx, unresolved_base_negative_power, x, point_zero)
            .is_none(),
        "negative integer power must not hide an unresolved finite base sublimit"
    );
}

#[test]
fn finite_total_real_unary_composition_rejects_unresolved_inner_limit() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let point = parse_expr(&mut ctx, "0");
    let expr = parse_expr(&mut ctx, "sin(sign(x))");

    assert!(
        try_limit_rules_at_finite(&mut ctx, expr, x, point).is_none(),
        "outer total-real function must not hide unresolved discontinuous inner limit"
    );
}

#[test]
fn rational_poly_limit_handles_equal_and_higher_degree_cases() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");

    let equal = parse_expr(&mut ctx, "(3*x^2 + 1)/(6*x^2 - 5)");
    let higher = parse_expr(&mut ctx, "(2*x^3)/(x^2+1)");

    let equal_out = rational_poly_limit(&mut ctx, equal, x, InfSign::Pos).expect("equal");
    let higher_out = rational_poly_limit(&mut ctx, higher, x, InfSign::Neg).expect("higher");

    assert!(matches!(ctx.get(equal_out), Expr::Number(_)));
    assert!(matches!(ctx.get(higher_out), Expr::Neg(_)));
}

#[test]
fn rational_poly_limit_rejects_non_polynomial_and_symbolic_leading_coeff() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let non_poly = parse_expr(&mut ctx, "sin(x)/x");
    let symbolic_lc = parse_expr(&mut ctx, "(y*x^2)/x^2");

    let out1 = rational_poly_limit(&mut ctx, non_poly, x, InfSign::Pos);
    let out2 = rational_poly_limit(&mut ctx, symbolic_lc, x, InfSign::Pos);

    assert!(out1.is_none());
    assert!(out2.is_none());
}

#[test]
fn sqrt_quadratic_minus_linear_limit_resolves_finite_cancellations() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (source, num, den) in [
        ("sqrt(x^2 + x) - x", 1, 2),
        ("sqrt(x^2 + 1) - x", 0, 1),
        ("x - sqrt(x^2 - x)", 1, 2),
        ("sqrt(x^2 + 3*x) - x", 3, 2),
        ("sqrt(4*x^2 + x) - 2*x", 1, 4),
        ("sqrt(x^2 + x + 1) - x", 1, 2),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = sqrt_quadratic_minus_linear_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("must resolve: {source}"));
        assert_number_expr(&ctx, out, num, den);
    }
}

#[test]
fn sqrt_quadratic_minus_linear_limit_declines_divergent_and_irrational() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    // Leading terms that do not cancel diverge; irrational sqrt(a)
    // and non-quadratic radicands have no rational closed form here.
    for source in [
        "sqrt(x^2 + 1) - 2*x",
        "sqrt(2*x^2 + x) - x",
        "sqrt(x^2 + 1) + x",
        "sqrt(x^3 + 1) - x",
    ] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            sqrt_quadratic_minus_linear_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos)
                .is_none(),
            "must decline: {source}"
        );
    }
}

#[test]
fn sqrt_minus_sqrt_limit_resolves_matching_leading_radicands() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (source, num, den) in [
        ("sqrt(x + 1) - sqrt(x)", 0, 1),
        ("sqrt(x^2 + x) - sqrt(x^2 - x)", 1, 1),
        ("sqrt(x^2 + 1) - sqrt(x^2 - 1)", 0, 1),
        ("sqrt(x^2 + 3*x) - sqrt(x^2 + x)", 1, 1),
        ("sqrt(4*x^2 + x) - sqrt(4*x^2 - x)", 1, 2),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = sqrt_minus_sqrt_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("must resolve: {source}"));
        assert_number_expr(&ctx, out, num, den);
    }
}

#[test]
fn sqrt_minus_sqrt_limit_declines_mismatched_and_high_degree() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    // Different degrees, irrational leading sqrt, degree > 2
    // (divergent), and additive (non-difference) forms decline.
    for source in [
        "sqrt(x^2 + x) - sqrt(x - 1)",
        "sqrt(2*x^2 + x) - sqrt(2*x^2 - x)",
        "sqrt(x^3 + x) - sqrt(x^3 - x)",
        "sqrt(x^2 + 1) + sqrt(x^2 - 1)",
    ] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            sqrt_minus_sqrt_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos).is_none(),
            "must decline: {source}"
        );
    }
}

#[test]
fn radical_conjugate_product_resolves_zero_times_infinity_forms() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    // factor * (radical difference that decays) -> finite. The decay rate
    // times the factor's growth lands a rational constant.
    for (source, num, den) in [
        ("x*(sqrt(x^2 + 1) - x)", 1, 2),
        ("x*(sqrt(x^2 + 4) - x)", 2, 1),
        ("x*(sqrt(x^2 - 1) - x)", -1, 2),
        ("x*(sqrt(x^2 + 2*x) - x - 1)", -1, 2),
        ("(sqrt(x^2 + 1) - x)*x", 1, 2),
        ("2*x*(sqrt(x^2 + 1) - x)", 1, 1),
        ("x*(sqrt(9*x^2 + 1) - 3*x)", 1, 6),
        ("x*(2*x - sqrt(4*x^2 + 1))", -1, 4),
        ("sqrt(x)*(sqrt(x + 1) - sqrt(x))", 1, 2),
        ("(sqrt(x + 1) - sqrt(x))*sqrt(x)", 1, 2),
        ("sqrt(x)*(sqrt(x + 2) - sqrt(x))", 1, 1),
        ("x*(sqrt(x^2 + x + 1) - sqrt(x^2 + x))", 1, 2),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = radical_conjugate_product_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("must resolve: {source}"));
        assert_number_expr(&ctx, out, num, den);
    }
}

#[test]
fn radical_conjugate_product_declines_divergent_irrational_and_neg_infinity() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    // Divergent products (the difference tends to a nonzero constant or the
    // factor outruns the decay), an irrational factor sqrt, a non-cancelling
    // additive form, and a non-radical second factor all decline.
    for source in [
        "x*(sqrt(x^2 + x) - x)",             // difference -> 1/2, product -> +inf
        "x^2*(sqrt(x^2 + 1) - x)",           // factor outruns the 1/x decay
        "sqrt(2*x)*(sqrt(x + 1) - sqrt(x))", // irrational leading sqrt(2)
        "x*(sqrt(x^2 + 1) + x)",             // additive: no leading cancellation
        "x*sin(x)",                          // second factor is not a radical difference
    ] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            radical_conjugate_product_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos).is_none(),
            "must decline: {source}"
        );
    }
    // The finite +inf forms are undefined toward -inf (radicands/sqrt factors
    // need x > 0), so the rule stays honest there.
    let neg_form = parse_expr(&mut ctx, "x*(sqrt(x^2 + 1) - x)");
    assert!(
        radical_conjugate_product_limit_at_infinity(&mut ctx, neg_form, x, InfSign::Neg).is_none(),
        "must decline toward -inf"
    );
}

#[test]
fn cbrt_conjugate_resolves_bare_and_zero_times_infinity_forms() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    // Bare cube-root conjugate differences and their 0*inf products resolve
    // to rationals via the a^2+ab+b^2 rationalization (~ 3 x^2 denominator).
    for (source, num, den) in [
        ("cbrt(x^3 + x^2) - x", 1, 3),
        ("cbrt(x^3 + 2*x^2) - x", 2, 3),
        ("cbrt(x^3 + 1) - x", 0, 1),
        ("cbrt(x^3 + x) - x", 0, 1),
        ("x^2*(cbrt(x^3 + 1) - x)", 1, 3),
        ("(cbrt(x^3 + 1) - x)*x^2", 1, 3),
        ("x*(cbrt(x^3 + 1) - x)", 0, 1),
        ("(x^3 + x^2)^(1/3) - x", 1, 3),
        ("x^2*((x^3 + 1)^(1/3) - x)", 1, 3),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = cbrt_conjugate_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("must resolve: {source}"));
        assert_number_expr(&ctx, out, num, den);
    }
}

#[test]
fn cbrt_conjugate_declines_divergent_irrational_and_neg_infinity() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    // Divergent products, an irrational cube root, a non-cubic radicand, and
    // a non-cube-root factor all decline.
    for source in [
        "x*(cbrt(x^3 + 3*x^2) - x)", // difference -> 1, product -> +inf
        "x^2*(cbrt(x^3 + x) - x)",   // factor outruns the 1/x decay
        "cbrt(2*x^3 + x^2) - x",     // irrational cbrt(2) leading
        "cbrt(x^2 + x) - x",         // radicand not a cubic
        "x*sin(x)",                  // second factor is not a cube-root difference
    ] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            cbrt_conjugate_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos).is_none(),
            "must decline: {source}"
        );
    }
    // Defined only toward +inf in this rule; the -inf side stays honest.
    let neg_form = parse_expr(&mut ctx, "cbrt(x^3 + x^2) - x");
    assert!(
        cbrt_conjugate_limit_at_infinity(&mut ctx, neg_form, x, InfSign::Neg).is_none(),
        "must decline toward -inf"
    );
}

#[test]
fn nth_root_conjugate_resolves_general_root_forms() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    // (P)^(1/n) - L conjugate differences and 0*inf products for n >= 4, via
    // the n-term conjugate ~ n d^(n-1) x^(n-1). a/n for the surviving tail.
    for (source, num, den) in [
        ("(x^4 + x^3)^(1/4) - x", 1, 4),
        ("(x^4 + 2*x^3)^(1/4) - x", 1, 2),
        ("(x^5 + x^4)^(1/5) - x", 1, 5),
        ("(16*x^4 + x^3)^(1/4) - 2*x", 1, 32),
        ("(x^4 + x^3)^(1/4) - x - 1", -3, 4),
        ("x^3*((x^4 + 1)^(1/4) - x)", 1, 4),
        ("((x^4 + 1)^(1/4) - x)*x^3", 1, 4),
        ("(x^4 + x^2)^(1/4) - x", 0, 1),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = nth_root_conjugate_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("must resolve: {source}"));
        assert_number_expr(&ctx, out, num, den);
    }
}

#[test]
fn nth_root_conjugate_declines_irrational_divergent_and_neg_infinity() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for source in [
        "(2*x^4 + x^3)^(1/4) - x",   // irrational 2^(1/4) leading
        "x^4*((x^4 + 1)^(1/4) - x)", // factor outruns the 1/x^3 decay
        "(x^3 + x^2)^(1/4) - x",     // radicand degree != n
        "x*sin(x)",                  // not an nth-root difference
    ] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            nth_root_conjugate_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos).is_none(),
            "must decline: {source}"
        );
    }
    let neg_form = parse_expr(&mut ctx, "(x^4 + x^3)^(1/4) - x");
    assert!(
        nth_root_conjugate_limit_at_infinity(&mut ctx, neg_form, x, InfSign::Neg).is_none(),
        "must decline toward -inf"
    );
}

#[test]
fn sqrt_polynomial_ratio_limit_at_infinity_handles_matching_growth() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");

    let pos = parse_expr(&mut ctx, "sqrt(x^2 + 1)/x");
    let neg = parse_expr(&mut ctx, "sqrt(x^2 + 1)/x");
    let scaled = parse_expr(&mut ctx, "sqrt(4*x^2 + 1)/(2*x)");
    let even_den = parse_expr(&mut ctx, "sqrt(x^4 + 1)/x^2");
    let irrational_coeff = parse_expr(&mut ctx, "sqrt(2*x^2 + 1)/x");
    let scaled_surd_den = parse_expr(&mut ctx, "sqrt(2*x^2 + 1)/(3*x)");
    let neg_scaled_surd_den = parse_expr(&mut ctx, "sqrt(2*x^2 + 1)/(-3*x)");
    let noisy_scaled_surd_den = parse_expr(&mut ctx, "sqrt(2*x^2 + x + 1)/(3*x + 1)");
    let bounded_noise_surd_den = parse_expr(&mut ctx, "sqrt((3*x + 1)^2 + sin(x))/(2*x + 1)");
    let bounded_noise_surd_noisy_den =
        parse_expr(&mut ctx, "sqrt((3*x + 1)^2 + sin(x))/(2*x + 1 + cos(x))");
    let scaled_bounded_noise_surd_noisy_den =
        parse_expr(&mut ctx, "5*sqrt((3*x + 1)^2 + sin(x))/(2*x + 1 + cos(x))");
    let bounded_noise_surd_scaled_noisy_den = parse_expr(
        &mut ctx,
        "sqrt((3*x + 1)^2 + sin(x))/(2*(2*x + 1 + cos(x)))",
    );

    let pos_out =
        sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, pos, x, InfSign::Pos).expect("+inf");
    let neg_out =
        sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, neg, x, InfSign::Neg).expect("-inf");
    let scaled_out =
        sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, scaled, x, InfSign::Pos).expect("scaled");
    let even_den_out = sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, even_den, x, InfSign::Neg)
        .expect("even denominator degree");
    let irrational_coeff_out =
        sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, irrational_coeff, x, InfSign::Pos)
            .expect("irrational leading coefficient");
    let scaled_surd_den_out =
        sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, scaled_surd_den, x, InfSign::Pos)
            .expect("scaled surd denominator");
    let neg_scaled_surd_den_out =
        sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, neg_scaled_surd_den, x, InfSign::Neg)
            .expect("negative scaled surd denominator");
    let noisy_scaled_surd_den_pos_out =
        sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, noisy_scaled_surd_den, x, InfSign::Pos)
            .expect("noisy scaled surd denominator at +inf");
    let noisy_scaled_surd_den_neg_out =
        sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, noisy_scaled_surd_den, x, InfSign::Neg)
            .expect("noisy scaled surd denominator at -inf");
    let bounded_noise_surd_den_pos_out =
        sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, bounded_noise_surd_den, x, InfSign::Pos)
            .expect("bounded radicand noise at +inf");
    let bounded_noise_surd_den_neg_out =
        sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, bounded_noise_surd_den, x, InfSign::Neg)
            .expect("bounded radicand noise at -inf");
    let bounded_noise_surd_noisy_den_pos_out = sqrt_polynomial_ratio_limit_at_infinity(
        &mut ctx,
        bounded_noise_surd_noisy_den,
        x,
        InfSign::Pos,
    )
    .expect("bounded radicand and denominator noise at +inf");
    let bounded_noise_surd_noisy_den_neg_out = sqrt_polynomial_ratio_limit_at_infinity(
        &mut ctx,
        bounded_noise_surd_noisy_den,
        x,
        InfSign::Neg,
    )
    .expect("bounded radicand and denominator noise at -inf");
    let scaled_bounded_noise_surd_noisy_den_pos_out = sqrt_polynomial_ratio_limit_at_infinity(
        &mut ctx,
        scaled_bounded_noise_surd_noisy_den,
        x,
        InfSign::Pos,
    )
    .expect("scaled bounded radicand and denominator noise at +inf");
    let bounded_noise_surd_scaled_noisy_den_pos_out = sqrt_polynomial_ratio_limit_at_infinity(
        &mut ctx,
        bounded_noise_surd_scaled_noisy_den,
        x,
        InfSign::Pos,
    )
    .expect("bounded radicand and scaled denominator noise at +inf");

    let one = BigRational::from_integer(BigInt::from(1));
    let minus_one = -one.clone();
    let two = BigRational::from_integer(BigInt::from(2));
    let three = BigRational::from_integer(BigInt::from(3));
    let three_halves = BigRational::new(BigInt::from(3), BigInt::from(2));
    let minus_three_halves = -three_halves.clone();
    let fifteen_halves = BigRational::new(BigInt::from(15), BigInt::from(2));
    let three_quarters = BigRational::new(BigInt::from(3), BigInt::from(4));
    assert!(matches!(ctx.get(pos_out), Expr::Number(n) if n == &one));
    assert!(matches!(ctx.get(neg_out), Expr::Number(n) if n == &minus_one));
    assert!(matches!(ctx.get(scaled_out), Expr::Number(n) if n == &one));
    assert!(matches!(ctx.get(even_den_out), Expr::Number(n) if n == &one));
    assert!(matches!(
        ctx.get(irrational_coeff_out),
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt)
                && matches!(args.as_slice(), [arg] if matches!(ctx.get(*arg), Expr::Number(n) if n == &two))
    ));
    assert!(matches!(
        ctx.get(scaled_surd_den_out),
        Expr::Div(num, den)
            if matches!(ctx.get(*num), Expr::Function(fn_id, _) if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt))
                && matches!(ctx.get(*den), Expr::Number(n) if n == &three)
    ));
    assert!(matches!(
        ctx.get(neg_scaled_surd_den_out),
        Expr::Div(num, den)
            if matches!(ctx.get(*num), Expr::Function(fn_id, _) if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt))
                && matches!(ctx.get(*den), Expr::Number(n) if n == &three)
    ));
    assert!(matches!(
        ctx.get(noisy_scaled_surd_den_pos_out),
        Expr::Div(num, den)
            if matches!(ctx.get(*num), Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt)
                    && matches!(args.as_slice(), [arg] if matches!(ctx.get(*arg), Expr::Number(n) if n == &two)))
                && matches!(ctx.get(*den), Expr::Number(n) if n == &three)
    ));
    assert!(matches!(
        ctx.get(noisy_scaled_surd_den_neg_out),
        Expr::Neg(inner)
            if matches!(ctx.get(*inner), Expr::Div(num, den)
                if matches!(ctx.get(*num), Expr::Function(fn_id, args)
                    if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt)
                        && matches!(args.as_slice(), [arg] if matches!(ctx.get(*arg), Expr::Number(n) if n == &two)))
                    && matches!(ctx.get(*den), Expr::Number(n) if n == &three))
    ));
    assert!(
        matches!(ctx.get(bounded_noise_surd_den_pos_out), Expr::Number(n) if n == &three_halves)
    );
    assert!(
        matches!(ctx.get(bounded_noise_surd_den_neg_out), Expr::Number(n) if n == &minus_three_halves)
    );
    assert!(
        matches!(ctx.get(bounded_noise_surd_noisy_den_pos_out), Expr::Number(n) if n == &three_halves)
    );
    assert!(
        matches!(ctx.get(bounded_noise_surd_noisy_den_neg_out), Expr::Number(n) if n == &minus_three_halves)
    );
    assert!(
        matches!(ctx.get(scaled_bounded_noise_surd_noisy_den_pos_out), Expr::Number(n) if n == &fifteen_halves)
    );
    assert!(
        matches!(ctx.get(bounded_noise_surd_scaled_noisy_den_pos_out), Expr::Number(n) if n == &three_quarters)
    );
}

#[test]
fn sqrt_polynomial_ratio_limit_at_infinity_rejects_unsafe_shapes() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");

    let negative_leading_coeff = parse_expr(&mut ctx, "sqrt(1 - 2*x^2)/x");
    let odd_radicand_degree = parse_expr(&mut ctx, "sqrt(x^3 + 1)/x");
    let mismatched_growth = parse_expr(&mut ctx, "sqrt(x^2 + 1)/x^2");
    let unbounded_noise = parse_expr(&mut ctx, "sqrt((3*x + 1)^2 + x*sin(x))/(2*x + 1)");
    let unbounded_den_noise =
        parse_expr(&mut ctx, "sqrt((3*x + 1)^2 + sin(x))/(2*x + 1 + x*cos(x))");

    assert!(sqrt_polynomial_ratio_limit_at_infinity(
        &mut ctx,
        negative_leading_coeff,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(sqrt_polynomial_ratio_limit_at_infinity(
        &mut ctx,
        odd_radicand_degree,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(
        sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, mismatched_growth, x, InfSign::Pos)
            .is_none()
    );
    assert!(
        sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, unbounded_noise, x, InfSign::Pos)
            .is_none()
    );
    assert!(sqrt_polynomial_ratio_limit_at_infinity(
        &mut ctx,
        unbounded_den_noise,
        x,
        InfSign::Pos
    )
    .is_none());
}

#[test]
fn polynomial_sqrt_ratio_limit_at_infinity_handles_matching_growth() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");

    let pos = parse_expr(&mut ctx, "x/sqrt(2*x^2 + 1)");
    let neg = parse_expr(&mut ctx, "x/sqrt(2*x^2 + 1)");
    let even_degree = parse_expr(&mut ctx, "x^2/sqrt(2*x^4 + 1)");
    let rational_coeff = parse_expr(&mut ctx, "x/sqrt(4*x^2 + 1)");
    let noisy = parse_expr(&mut ctx, "(3*x + 1)/sqrt(2*x^2 + x + 1)");
    let bounded_noise_num = parse_expr(&mut ctx, "(2*x + 1 + cos(x))/sqrt((3*x + 1)^2 + sin(x))");

    let pos_out =
        polynomial_sqrt_ratio_limit_at_infinity(&mut ctx, pos, x, InfSign::Pos).expect("+inf");
    let neg_out =
        polynomial_sqrt_ratio_limit_at_infinity(&mut ctx, neg, x, InfSign::Neg).expect("-inf");
    let even_degree_out =
        polynomial_sqrt_ratio_limit_at_infinity(&mut ctx, even_degree, x, InfSign::Neg)
            .expect("even degree");
    let rational_coeff_out =
        polynomial_sqrt_ratio_limit_at_infinity(&mut ctx, rational_coeff, x, InfSign::Pos)
            .expect("rational sqrt coefficient");
    let noisy_out = polynomial_sqrt_ratio_limit_at_infinity(&mut ctx, noisy, x, InfSign::Pos)
        .expect("lower-order polynomial noise");
    let bounded_noise_num_pos_out =
        polynomial_sqrt_ratio_limit_at_infinity(&mut ctx, bounded_noise_num, x, InfSign::Pos)
            .expect("bounded numerator and radicand noise at +inf");
    let bounded_noise_num_neg_out =
        polynomial_sqrt_ratio_limit_at_infinity(&mut ctx, bounded_noise_num, x, InfSign::Neg)
            .expect("bounded numerator and radicand noise at -inf");

    let two = BigRational::from_integer(BigInt::from(2));
    let three = BigRational::from_integer(BigInt::from(3));
    let two_thirds = BigRational::new(BigInt::from(2), BigInt::from(3));
    let minus_two_thirds = -two_thirds.clone();
    assert!(matches!(
        ctx.get(pos_out),
        Expr::Div(num, den)
            if matches!(ctx.get(*num), Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt)
                    && matches!(args.as_slice(), [arg] if matches!(ctx.get(*arg), Expr::Number(n) if n == &two)))
                && matches!(ctx.get(*den), Expr::Number(n) if n == &two)
    ));
    assert!(matches!(ctx.get(neg_out), Expr::Neg(_)));
    assert!(matches!(
        ctx.get(even_degree_out),
        Expr::Div(num, den)
            if matches!(ctx.get(*num), Expr::Function(fn_id, _) if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt))
                && matches!(ctx.get(*den), Expr::Number(n) if n == &two)
    ));
    assert!(matches!(
        ctx.get(rational_coeff_out),
        Expr::Number(n) if n == &BigRational::new(BigInt::from(1), BigInt::from(2))
    ));
    assert!(matches!(
        ctx.get(noisy_out),
        Expr::Div(num, den)
            if matches!(ctx.get(*num), Expr::Mul(coeff, sqrt)
                if matches!(ctx.get(*coeff), Expr::Number(n) if n == &three)
                    && matches!(ctx.get(*sqrt), Expr::Function(fn_id, args)
                        if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt)
                            && matches!(args.as_slice(), [arg] if matches!(ctx.get(*arg), Expr::Number(n) if n == &two))))
                && matches!(ctx.get(*den), Expr::Number(n) if n == &two)
    ));
    assert!(matches!(ctx.get(bounded_noise_num_pos_out), Expr::Number(n) if n == &two_thirds));
    assert!(
        matches!(ctx.get(bounded_noise_num_neg_out), Expr::Number(n) if n == &minus_two_thirds)
    );
}

#[test]
fn polynomial_sqrt_ratio_limit_at_infinity_rejects_unsafe_shapes() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");

    let negative_leading_coeff = parse_expr(&mut ctx, "x/sqrt(1 - 2*x^2)");
    let odd_radicand_degree = parse_expr(&mut ctx, "x/sqrt(x^3 + 1)");
    let mismatched_growth = parse_expr(&mut ctx, "x/sqrt(x^4 + 1)");
    let unbounded_num_noise =
        parse_expr(&mut ctx, "(2*x + 1 + x*cos(x))/sqrt((3*x + 1)^2 + sin(x))");

    assert!(polynomial_sqrt_ratio_limit_at_infinity(
        &mut ctx,
        negative_leading_coeff,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(polynomial_sqrt_ratio_limit_at_infinity(
        &mut ctx,
        odd_radicand_degree,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(
        polynomial_sqrt_ratio_limit_at_infinity(&mut ctx, mismatched_growth, x, InfSign::Pos)
            .is_none()
    );
    assert!(polynomial_sqrt_ratio_limit_at_infinity(
        &mut ctx,
        unbounded_num_noise,
        x,
        InfSign::Pos
    )
    .is_none());
}

#[test]
fn polynomial_limit_at_infinity_handles_numeric_leading_coeff_and_parity() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let positive_even = parse_expr(&mut ctx, "x^2 + 1");
    let negative_odd = parse_expr(&mut ctx, "x - 2*x^3");

    let pos_even_out = polynomial_limit_at_infinity(&mut ctx, positive_even, x, InfSign::Neg)
        .expect("positive even polynomial");
    let neg_odd_pos_out = polynomial_limit_at_infinity(&mut ctx, negative_odd, x, InfSign::Pos)
        .expect("negative odd at +inf");
    let neg_odd_neg_out = polynomial_limit_at_infinity(&mut ctx, negative_odd, x, InfSign::Neg)
        .expect("negative odd at -inf");

    assert!(matches!(
        ctx.get(pos_even_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(ctx.get(neg_odd_pos_out), Expr::Neg(_)));
    assert!(matches!(
        ctx.get(neg_odd_neg_out),
        Expr::Constant(Constant::Infinity)
    ));
}

#[test]
fn polynomial_limit_at_infinity_rejects_symbolic_leading_coeff() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let symbolic_lc = parse_expr(&mut ctx, "y*x^2 + 1");

    let out = polynomial_limit_at_infinity(&mut ctx, symbolic_lc, x, InfSign::Pos);

    assert!(out.is_none());
}

#[test]
fn elementary_function_limit_at_infinity_handles_exact_growth_cases() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let sqrt_x = parse_expr(&mut ctx, "sqrt(x)");
    let cbrt_x = parse_expr(&mut ctx, "cbrt(x)");
    let cbrt_neg_linear = parse_expr(&mut ctx, "cbrt(1 - x)");
    let asinh_x = parse_expr(&mut ctx, "asinh(x)");
    let asinh_neg_linear = parse_expr(&mut ctx, "asinh(1 - x)");
    let acosh_x = parse_expr(&mut ctx, "acosh(x)");
    let acosh_neg_linear = parse_expr(&mut ctx, "acosh(1 - x)");
    let atan_x = parse_expr(&mut ctx, "atan(x)");
    let arctan_neg_linear = parse_expr(&mut ctx, "arctan(1 - x)");
    let tanh_x = parse_expr(&mut ctx, "tanh(x)");
    let tanh_neg_linear = parse_expr(&mut ctx, "tanh(1 - x)");
    let sinh_x = parse_expr(&mut ctx, "sinh(x)");
    let sinh_neg_linear = parse_expr(&mut ctx, "sinh(1 - x)");
    let cosh_x = parse_expr(&mut ctx, "cosh(x)");
    let cosh_neg_linear = parse_expr(&mut ctx, "cosh(1 - x)");
    let ln_x = parse_expr(&mut ctx, "ln(x)");
    let ln_quadratic = parse_expr(&mut ctx, "ln(x^2)");
    let shifted_ln_quadratic = parse_expr(&mut ctx, "ln(x^2 - 3)");
    let base_log_quadratic = parse_expr(&mut ctx, "log(2, x^2)");
    let reciprocal_base_log_quadratic = parse_expr(&mut ctx, "log(1/2, x^2)");
    let exp_x = parse_expr(&mut ctx, "exp(x)");
    let exp_neg_x = parse_expr(&mut ctx, "exp(-x)");
    let exp_two_x = parse_expr(&mut ctx, "exp(2*x)");
    let ln_neg_linear = parse_expr(&mut ctx, "ln(-x + 1)");
    let ln_negative_tail_quadratic = parse_expr(&mut ctx, "ln(3 - x^2)");
    let base_log_negative_tail = parse_expr(&mut ctx, "log(2, 3 - x^2)");
    let invalid_base_log_quadratic = parse_expr(&mut ctx, "log(1, x^2)");
    let sqrt_neg_linear = parse_expr(&mut ctx, "sqrt(1 - x)");
    let cbrt_exp_x = parse_expr(&mut ctx, "cbrt(exp(x))");
    let cbrt_exp_neg_x = parse_expr(&mut ctx, "cbrt(exp(-x))");
    let asinh_exp_x = parse_expr(&mut ctx, "asinh(exp(x))");
    let asinh_exp_neg_x = parse_expr(&mut ctx, "asinh(exp(-x))");
    let acosh_exp_x = parse_expr(&mut ctx, "acosh(exp(x))");
    let acosh_exp_neg_x = parse_expr(&mut ctx, "acosh(exp(-x))");
    let atan_exp_x = parse_expr(&mut ctx, "atan(exp(x))");
    let arctan_exp_neg_x = parse_expr(&mut ctx, "arctan(exp(-x))");
    let tanh_exp_x = parse_expr(&mut ctx, "tanh(exp(x))");
    let tanh_exp_neg_x = parse_expr(&mut ctx, "tanh(exp(-x))");
    let sinh_exp_x = parse_expr(&mut ctx, "sinh(exp(x))");
    let sinh_exp_neg_x = parse_expr(&mut ctx, "sinh(exp(-x))");
    let cosh_exp_x = parse_expr(&mut ctx, "cosh(exp(x))");
    let cosh_exp_neg_x = parse_expr(&mut ctx, "cosh(exp(-x))");
    let exp_quadratic = parse_expr(&mut ctx, "exp(x^2)");
    let negative_tail_exp_quartic = parse_expr(&mut ctx, "exp(2 - x^4)");
    let exp_cubic = parse_expr(&mut ctx, "exp(x^3 - 2*x)");
    let parametric_tail_exp_quadratic = parse_expr(&mut ctx, "exp(a*x^2 + 1)");
    let nested_exp_quadratic = parse_expr(&mut ctx, "exp(exp(x^2))");
    let cbrt_quadratic = parse_expr(&mut ctx, "cbrt(x^2)");
    let negative_tail_cbrt_quartic = parse_expr(&mut ctx, "cbrt(2 - x^4)");
    let parametric_tail_cbrt_quadratic = parse_expr(&mut ctx, "cbrt(a*x^2 + 1)");
    let cbrt_exp_quadratic = parse_expr(&mut ctx, "cbrt(exp(x^2))");
    let asinh_quadratic = parse_expr(&mut ctx, "asinh(x^2)");
    let negative_tail_asinh_quartic = parse_expr(&mut ctx, "asinh(2 - x^4)");
    let parametric_tail_asinh_quadratic = parse_expr(&mut ctx, "asinh(a*x^2 + 1)");
    let asinh_exp_quadratic = parse_expr(&mut ctx, "asinh(exp(x^2))");
    let acosh_quadratic = parse_expr(&mut ctx, "acosh(x^2)");
    let shifted_acosh_quadratic = parse_expr(&mut ctx, "acosh(x^2 - 3)");
    let negative_tail_acosh_quadratic = parse_expr(&mut ctx, "acosh(3 - x^2)");
    let parametric_tail_acosh_quadratic = parse_expr(&mut ctx, "acosh(a*x^2 + 1)");
    let acosh_exp_quadratic = parse_expr(&mut ctx, "acosh(exp(x^2))");
    let atan_quadratic = parse_expr(&mut ctx, "atan(x^2)");
    let negative_tail_atan_quartic = parse_expr(&mut ctx, "atan(2 - x^4)");
    let arctan_cubic = parse_expr(&mut ctx, "arctan(x^3 - 2*x)");
    let parametric_tail_atan_quadratic = parse_expr(&mut ctx, "atan(a*x^2 + 1)");
    let arctan_exp_quadratic = parse_expr(&mut ctx, "arctan(exp(x^2))");
    let tanh_quadratic = parse_expr(&mut ctx, "tanh(x^2)");
    let negative_tail_tanh_quartic = parse_expr(&mut ctx, "tanh(2 - x^4)");
    let parametric_tail_tanh_quadratic = parse_expr(&mut ctx, "tanh(a*x^2 + 1)");
    let tanh_exp_quadratic = parse_expr(&mut ctx, "tanh(exp(x^2))");
    let sinh_quadratic = parse_expr(&mut ctx, "sinh(x^2)");
    let negative_tail_sinh_quartic = parse_expr(&mut ctx, "sinh(2 - x^4)");
    let sinh_cubic = parse_expr(&mut ctx, "sinh(x^3 - 2*x)");
    let parametric_tail_sinh_quadratic = parse_expr(&mut ctx, "sinh(a*x^2 + 1)");
    let sinh_exp_quadratic = parse_expr(&mut ctx, "sinh(exp(x^2))");
    let cosh_quadratic = parse_expr(&mut ctx, "cosh(x^2)");
    let negative_tail_cosh_quartic = parse_expr(&mut ctx, "cosh(2 - x^4)");
    let parametric_tail_cosh_quadratic = parse_expr(&mut ctx, "cosh(a*x^2 + 1)");
    let cosh_exp_quadratic = parse_expr(&mut ctx, "cosh(exp(x^2))");

    let sqrt_pos = elementary_function_limit_at_infinity(&mut ctx, sqrt_x, x, InfSign::Pos)
        .expect("sqrt at +inf");
    let cbrt_pos = elementary_function_limit_at_infinity(&mut ctx, cbrt_x, x, InfSign::Pos)
        .expect("cbrt at +inf");
    let cbrt_neg = elementary_function_limit_at_infinity(&mut ctx, cbrt_x, x, InfSign::Neg)
        .expect("cbrt at -inf");
    let cbrt_neg_linear_pos =
        elementary_function_limit_at_infinity(&mut ctx, cbrt_neg_linear, x, InfSign::Pos)
            .expect("cbrt(1 - x) at +inf");
    let asinh_pos = elementary_function_limit_at_infinity(&mut ctx, asinh_x, x, InfSign::Pos)
        .expect("asinh at +inf");
    let asinh_neg = elementary_function_limit_at_infinity(&mut ctx, asinh_x, x, InfSign::Neg)
        .expect("asinh at -inf");
    let asinh_neg_linear_pos =
        elementary_function_limit_at_infinity(&mut ctx, asinh_neg_linear, x, InfSign::Pos)
            .expect("asinh(1 - x) at +inf");
    let acosh_pos = elementary_function_limit_at_infinity(&mut ctx, acosh_x, x, InfSign::Pos)
        .expect("acosh at +inf");
    let acosh_neg_linear_neg =
        elementary_function_limit_at_infinity(&mut ctx, acosh_neg_linear, x, InfSign::Neg)
            .expect("acosh(1 - x) at -inf");
    let acosh_quadratic_pos =
        elementary_function_limit_at_infinity(&mut ctx, acosh_quadratic, x, InfSign::Pos)
            .expect("acosh(x^2) at +inf");
    let acosh_quadratic_neg =
        elementary_function_limit_at_infinity(&mut ctx, acosh_quadratic, x, InfSign::Neg)
            .expect("acosh(x^2) at -inf");
    let shifted_acosh_quadratic_pos =
        elementary_function_limit_at_infinity(&mut ctx, shifted_acosh_quadratic, x, InfSign::Pos)
            .expect("acosh(x^2 - 3) at +inf");
    let atan_pos = elementary_function_limit_at_infinity(&mut ctx, atan_x, x, InfSign::Pos)
        .expect("atan at +inf");
    let atan_neg = elementary_function_limit_at_infinity(&mut ctx, atan_x, x, InfSign::Neg)
        .expect("atan at -inf");
    let arctan_neg_linear_pos =
        elementary_function_limit_at_infinity(&mut ctx, arctan_neg_linear, x, InfSign::Pos)
            .expect("arctan(1 - x) at +inf");
    let tanh_pos = elementary_function_limit_at_infinity(&mut ctx, tanh_x, x, InfSign::Pos)
        .expect("tanh at +inf");
    let tanh_neg = elementary_function_limit_at_infinity(&mut ctx, tanh_x, x, InfSign::Neg)
        .expect("tanh at -inf");
    let tanh_neg_linear_pos =
        elementary_function_limit_at_infinity(&mut ctx, tanh_neg_linear, x, InfSign::Pos)
            .expect("tanh(1 - x) at +inf");
    let sinh_pos = elementary_function_limit_at_infinity(&mut ctx, sinh_x, x, InfSign::Pos)
        .expect("sinh at +inf");
    let sinh_neg = elementary_function_limit_at_infinity(&mut ctx, sinh_x, x, InfSign::Neg)
        .expect("sinh at -inf");
    let sinh_neg_linear_pos =
        elementary_function_limit_at_infinity(&mut ctx, sinh_neg_linear, x, InfSign::Pos)
            .expect("sinh(1 - x) at +inf");
    let cosh_pos = elementary_function_limit_at_infinity(&mut ctx, cosh_x, x, InfSign::Pos)
        .expect("cosh at +inf");
    let cosh_neg = elementary_function_limit_at_infinity(&mut ctx, cosh_x, x, InfSign::Neg)
        .expect("cosh at -inf");
    let cosh_neg_linear_pos =
        elementary_function_limit_at_infinity(&mut ctx, cosh_neg_linear, x, InfSign::Pos)
            .expect("cosh(1 - x) at +inf");
    let ln_pos =
        elementary_function_limit_at_infinity(&mut ctx, ln_x, x, InfSign::Pos).expect("ln at +inf");
    let ln_quadratic_pos =
        elementary_function_limit_at_infinity(&mut ctx, ln_quadratic, x, InfSign::Pos)
            .expect("ln(x^2) at +inf");
    let ln_quadratic_neg =
        elementary_function_limit_at_infinity(&mut ctx, ln_quadratic, x, InfSign::Neg)
            .expect("ln(x^2) at -inf");
    let shifted_ln_quadratic_pos =
        elementary_function_limit_at_infinity(&mut ctx, shifted_ln_quadratic, x, InfSign::Pos)
            .expect("ln(x^2 - 3) at +inf");
    let base_log_quadratic_pos =
        elementary_function_limit_at_infinity(&mut ctx, base_log_quadratic, x, InfSign::Pos)
            .expect("log(2, x^2) at +inf");
    let reciprocal_base_log_quadratic_pos = elementary_function_limit_at_infinity(
        &mut ctx,
        reciprocal_base_log_quadratic,
        x,
        InfSign::Pos,
    )
    .expect("log(1/2, x^2) at +inf");
    let exp_pos = elementary_function_limit_at_infinity(&mut ctx, exp_x, x, InfSign::Pos)
        .expect("exp at +inf");
    let exp_neg = elementary_function_limit_at_infinity(&mut ctx, exp_x, x, InfSign::Neg)
        .expect("exp at -inf");
    let exp_neg_x_pos = elementary_function_limit_at_infinity(&mut ctx, exp_neg_x, x, InfSign::Pos)
        .expect("exp(-x) at +inf");
    let exp_two_x_neg = elementary_function_limit_at_infinity(&mut ctx, exp_two_x, x, InfSign::Neg)
        .expect("exp(2*x) at -inf");
    let exp_quadratic_pos =
        elementary_function_limit_at_infinity(&mut ctx, exp_quadratic, x, InfSign::Pos)
            .expect("exp(x^2) at +inf");
    let exp_quadratic_neg =
        elementary_function_limit_at_infinity(&mut ctx, exp_quadratic, x, InfSign::Neg)
            .expect("exp(x^2) at -inf");
    let negative_tail_exp_quartic_pos =
        elementary_function_limit_at_infinity(&mut ctx, negative_tail_exp_quartic, x, InfSign::Pos)
            .expect("exp(2 - x^4) at +inf");
    let exp_cubic_neg = elementary_function_limit_at_infinity(&mut ctx, exp_cubic, x, InfSign::Neg)
        .expect("exp(x^3 - 2*x) at -inf");
    let ln_neg_linear_neg =
        elementary_function_limit_at_infinity(&mut ctx, ln_neg_linear, x, InfSign::Neg)
            .expect("ln(-x + 1) at -inf");
    let sqrt_neg_linear_neg =
        elementary_function_limit_at_infinity(&mut ctx, sqrt_neg_linear, x, InfSign::Neg)
            .expect("sqrt(1 - x) at -inf");
    let cbrt_exp_pos = elementary_function_limit_at_infinity(&mut ctx, cbrt_exp_x, x, InfSign::Pos)
        .expect("cbrt(exp(x)) at +inf");
    let cbrt_exp_neg_pos =
        elementary_function_limit_at_infinity(&mut ctx, cbrt_exp_neg_x, x, InfSign::Pos)
            .expect("cbrt(exp(-x)) at +inf");
    let cbrt_quadratic_pos =
        elementary_function_limit_at_infinity(&mut ctx, cbrt_quadratic, x, InfSign::Pos)
            .expect("cbrt(x^2) at +inf");
    let negative_tail_cbrt_quartic_pos = elementary_function_limit_at_infinity(
        &mut ctx,
        negative_tail_cbrt_quartic,
        x,
        InfSign::Pos,
    )
    .expect("cbrt(2 - x^4) at +inf");
    let asinh_exp_pos =
        elementary_function_limit_at_infinity(&mut ctx, asinh_exp_x, x, InfSign::Pos)
            .expect("asinh(exp(x)) at +inf");
    let asinh_exp_neg_pos =
        elementary_function_limit_at_infinity(&mut ctx, asinh_exp_neg_x, x, InfSign::Pos)
            .expect("asinh(exp(-x)) at +inf");
    let asinh_quadratic_pos =
        elementary_function_limit_at_infinity(&mut ctx, asinh_quadratic, x, InfSign::Pos)
            .expect("asinh(x^2) at +inf");
    let negative_tail_asinh_quartic_pos = elementary_function_limit_at_infinity(
        &mut ctx,
        negative_tail_asinh_quartic,
        x,
        InfSign::Pos,
    )
    .expect("asinh(2 - x^4) at +inf");
    let acosh_exp_pos =
        elementary_function_limit_at_infinity(&mut ctx, acosh_exp_x, x, InfSign::Pos)
            .expect("acosh(exp(x)) at +inf");
    let acosh_exp_neg_neg =
        elementary_function_limit_at_infinity(&mut ctx, acosh_exp_neg_x, x, InfSign::Neg)
            .expect("acosh(exp(-x)) at -inf");
    let atan_exp_pos = elementary_function_limit_at_infinity(&mut ctx, atan_exp_x, x, InfSign::Pos)
        .expect("atan(exp(x)) at +inf");
    let arctan_exp_neg_pos =
        elementary_function_limit_at_infinity(&mut ctx, arctan_exp_neg_x, x, InfSign::Pos)
            .expect("arctan(exp(-x)) at +inf");
    let atan_quadratic_pos =
        elementary_function_limit_at_infinity(&mut ctx, atan_quadratic, x, InfSign::Pos)
            .expect("atan(x^2) at +inf");
    let atan_quadratic_neg =
        elementary_function_limit_at_infinity(&mut ctx, atan_quadratic, x, InfSign::Neg)
            .expect("atan(x^2) at -inf");
    let negative_tail_atan_quartic_pos = elementary_function_limit_at_infinity(
        &mut ctx,
        negative_tail_atan_quartic,
        x,
        InfSign::Pos,
    )
    .expect("atan(2 - x^4) at +inf");
    let arctan_cubic_neg =
        elementary_function_limit_at_infinity(&mut ctx, arctan_cubic, x, InfSign::Neg)
            .expect("arctan(x^3 - 2*x) at -inf");
    let tanh_exp_pos = elementary_function_limit_at_infinity(&mut ctx, tanh_exp_x, x, InfSign::Pos)
        .expect("tanh(exp(x)) at +inf");
    let tanh_exp_neg_pos =
        elementary_function_limit_at_infinity(&mut ctx, tanh_exp_neg_x, x, InfSign::Pos)
            .expect("tanh(exp(-x)) at +inf");
    let tanh_quadratic_pos =
        elementary_function_limit_at_infinity(&mut ctx, tanh_quadratic, x, InfSign::Pos)
            .expect("tanh(x^2) at +inf");
    let tanh_quadratic_neg =
        elementary_function_limit_at_infinity(&mut ctx, tanh_quadratic, x, InfSign::Neg)
            .expect("tanh(x^2) at -inf");
    let negative_tail_tanh_quartic_pos = elementary_function_limit_at_infinity(
        &mut ctx,
        negative_tail_tanh_quartic,
        x,
        InfSign::Pos,
    )
    .expect("tanh(2 - x^4) at +inf");
    let sinh_exp_pos = elementary_function_limit_at_infinity(&mut ctx, sinh_exp_x, x, InfSign::Pos)
        .expect("sinh(exp(x)) at +inf");
    let sinh_exp_neg_pos =
        elementary_function_limit_at_infinity(&mut ctx, sinh_exp_neg_x, x, InfSign::Pos)
            .expect("sinh(exp(-x)) at +inf");
    let sinh_quadratic_pos =
        elementary_function_limit_at_infinity(&mut ctx, sinh_quadratic, x, InfSign::Pos)
            .expect("sinh(x^2) at +inf");
    let sinh_quadratic_neg =
        elementary_function_limit_at_infinity(&mut ctx, sinh_quadratic, x, InfSign::Neg)
            .expect("sinh(x^2) at -inf");
    let negative_tail_sinh_quartic_pos = elementary_function_limit_at_infinity(
        &mut ctx,
        negative_tail_sinh_quartic,
        x,
        InfSign::Pos,
    )
    .expect("sinh(2 - x^4) at +inf");
    let sinh_cubic_neg =
        elementary_function_limit_at_infinity(&mut ctx, sinh_cubic, x, InfSign::Neg)
            .expect("sinh(x^3 - 2*x) at -inf");
    let cosh_exp_pos = elementary_function_limit_at_infinity(&mut ctx, cosh_exp_x, x, InfSign::Pos)
        .expect("cosh(exp(x)) at +inf");
    let cosh_exp_neg_pos =
        elementary_function_limit_at_infinity(&mut ctx, cosh_exp_neg_x, x, InfSign::Pos)
            .expect("cosh(exp(-x)) at +inf");
    let cosh_quadratic_pos =
        elementary_function_limit_at_infinity(&mut ctx, cosh_quadratic, x, InfSign::Pos)
            .expect("cosh(x^2) at +inf");
    let cosh_quadratic_neg =
        elementary_function_limit_at_infinity(&mut ctx, cosh_quadratic, x, InfSign::Neg)
            .expect("cosh(x^2) at -inf");
    let negative_tail_cosh_quartic_pos = elementary_function_limit_at_infinity(
        &mut ctx,
        negative_tail_cosh_quartic,
        x,
        InfSign::Pos,
    )
    .expect("cosh(2 - x^4) at +inf");

    assert!(matches!(
        ctx.get(sqrt_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(cbrt_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(cbrt_neg), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
    );
    assert!(
        matches!(ctx.get(cbrt_neg_linear_pos), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
    );
    assert!(matches!(
        ctx.get(asinh_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(asinh_neg), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
    );
    assert!(
        matches!(ctx.get(asinh_neg_linear_pos), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
    );
    assert!(matches!(
        ctx.get(acosh_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(acosh_neg_linear_neg),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(acosh_quadratic_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(acosh_quadratic_neg),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(shifted_acosh_quadratic_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert_eq!(display_expr(&ctx, atan_pos), "pi / 2");
    assert_eq!(display_expr(&ctx, atan_neg), "-pi / 2");
    assert_eq!(display_expr(&ctx, arctan_neg_linear_pos), "-pi / 2");
    assert_eq!(display_expr(&ctx, tanh_pos), "1");
    assert_eq!(display_expr(&ctx, tanh_neg), "-1");
    assert_eq!(display_expr(&ctx, tanh_neg_linear_pos), "-1");
    assert!(matches!(
        ctx.get(sinh_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(sinh_neg), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
    );
    assert!(
        matches!(ctx.get(sinh_neg_linear_pos), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
    );
    assert!(matches!(
        ctx.get(cosh_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(cosh_neg),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(cosh_neg_linear_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(ln_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(ln_quadratic_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(ln_quadratic_neg),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(shifted_ln_quadratic_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(base_log_quadratic_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(reciprocal_base_log_quadratic_pos), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
    );
    assert!(matches!(
        ctx.get(exp_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(exp_neg), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(
        matches!(ctx.get(exp_neg_x_pos), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(
        matches!(ctx.get(exp_two_x_neg), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(exp_quadratic_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(exp_quadratic_neg),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(negative_tail_exp_quartic_pos), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(
        matches!(ctx.get(exp_cubic_neg), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(ln_neg_linear_neg),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(sqrt_neg_linear_neg),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(cbrt_exp_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(cbrt_exp_neg_pos), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(cbrt_quadratic_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(negative_tail_cbrt_quartic_pos), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
    );
    assert!(matches!(
        ctx.get(asinh_exp_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(asinh_exp_neg_pos), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(asinh_quadratic_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(negative_tail_asinh_quartic_pos), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
    );
    assert!(matches!(
        ctx.get(acosh_exp_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(acosh_exp_neg_neg),
        Expr::Constant(Constant::Infinity)
    ));
    assert_eq!(display_expr(&ctx, atan_exp_pos), "pi / 2");
    assert!(
        matches!(ctx.get(arctan_exp_neg_pos), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert_eq!(display_expr(&ctx, atan_quadratic_pos), "pi / 2");
    assert_eq!(display_expr(&ctx, atan_quadratic_neg), "pi / 2");
    assert_eq!(
        display_expr(&ctx, negative_tail_atan_quartic_pos),
        "-pi / 2"
    );
    assert_eq!(display_expr(&ctx, arctan_cubic_neg), "-pi / 2");
    assert_eq!(display_expr(&ctx, tanh_exp_pos), "1");
    assert!(
        matches!(ctx.get(tanh_exp_neg_pos), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert_eq!(display_expr(&ctx, tanh_quadratic_pos), "1");
    assert_eq!(display_expr(&ctx, tanh_quadratic_neg), "1");
    assert_eq!(display_expr(&ctx, negative_tail_tanh_quartic_pos), "-1");
    assert!(matches!(
        ctx.get(sinh_exp_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(sinh_exp_neg_pos), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(sinh_quadratic_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(sinh_quadratic_neg),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(negative_tail_sinh_quartic_pos), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
    );
    assert!(
        matches!(ctx.get(sinh_cubic_neg), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
    );
    assert!(matches!(
        ctx.get(cosh_exp_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert_eq!(display_expr(&ctx, cosh_exp_neg_pos), "1");
    assert!(matches!(
        ctx.get(cosh_quadratic_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(cosh_quadratic_neg),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(negative_tail_cosh_quartic_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(elementary_function_limit_at_infinity(&mut ctx, sqrt_x, x, InfSign::Neg).is_none());
    assert!(elementary_function_limit_at_infinity(&mut ctx, ln_x, x, InfSign::Neg).is_none());
    assert!(elementary_function_limit_at_infinity(&mut ctx, acosh_x, x, InfSign::Neg).is_none());
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        ln_negative_tail_quadratic,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        base_log_negative_tail,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        invalid_base_log_quadratic,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(
        elementary_function_limit_at_infinity(&mut ctx, acosh_neg_linear, x, InfSign::Pos)
            .is_none()
    );
    assert!(
        elementary_function_limit_at_infinity(&mut ctx, acosh_exp_x, x, InfSign::Neg).is_none()
    );
    assert!(
        elementary_function_limit_at_infinity(&mut ctx, acosh_exp_neg_x, x, InfSign::Pos).is_none()
    );
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        parametric_tail_exp_quadratic,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(
        elementary_function_limit_at_infinity(&mut ctx, nested_exp_quadratic, x, InfSign::Pos)
            .is_none()
    );
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        parametric_tail_cbrt_quadratic,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(
        elementary_function_limit_at_infinity(&mut ctx, cbrt_exp_quadratic, x, InfSign::Pos)
            .is_none()
    );
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        parametric_tail_asinh_quadratic,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(
        elementary_function_limit_at_infinity(&mut ctx, asinh_exp_quadratic, x, InfSign::Pos)
            .is_none()
    );
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        negative_tail_acosh_quadratic,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        parametric_tail_acosh_quadratic,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(
        elementary_function_limit_at_infinity(&mut ctx, acosh_exp_quadratic, x, InfSign::Pos)
            .is_none()
    );
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        parametric_tail_atan_quadratic,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(
        elementary_function_limit_at_infinity(&mut ctx, arctan_exp_quadratic, x, InfSign::Pos)
            .is_none()
    );
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        parametric_tail_tanh_quadratic,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(
        elementary_function_limit_at_infinity(&mut ctx, tanh_exp_quadratic, x, InfSign::Pos)
            .is_none()
    );
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        parametric_tail_sinh_quadratic,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(
        elementary_function_limit_at_infinity(&mut ctx, sinh_exp_quadratic, x, InfSign::Pos)
            .is_none()
    );
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        parametric_tail_cosh_quadratic,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(
        elementary_function_limit_at_infinity(&mut ctx, cosh_exp_quadratic, x, InfSign::Pos)
            .is_none()
    );
}

#[test]
fn elementary_function_limit_at_infinity_handles_rational_positive_unbounded_arguments() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let ln_rational = parse_expr(&mut ctx, "ln((x^2 + 1)/(x + 1))");
    let base_log_rational = parse_expr(&mut ctx, "log(2, (x^2 + 1)/(x + 1))");
    let reciprocal_base_log_rational = parse_expr(&mut ctx, "log(1/2, (x^2 + 1)/(x + 1))");
    let acosh_rational = parse_expr(&mut ctx, "acosh((x^2 + 1)/(x + 1))");
    let negative_tail_log = parse_expr(&mut ctx, "ln((x^2 + 1)/(1 - x))");
    let proper_ln = parse_expr(&mut ctx, "ln((x + 1)/(x^2 + 1))");
    let proper_log2 = parse_expr(&mut ctx, "log2((x + 1)/(x^2 + 1))");
    let proper_base_log = parse_expr(&mut ctx, "log(2, (x + 1)/(x^2 + 1))");
    let proper_reciprocal_base_log = parse_expr(&mut ctx, "log(1/2, (x + 1)/(x^2 + 1))");
    let constant_num_log = parse_expr(&mut ctx, "log2(1/(x^2 + 1))");
    let proper_acosh = parse_expr(&mut ctx, "acosh((x + 1)/(x^2 + 1))");
    let parametric_log = parse_expr(&mut ctx, "ln((a*x^2 + 1)/(x + 1))");
    let negative_zero_tail_log = parse_expr(&mut ctx, "ln((1 - x)/(x^2 + 1))");
    let parametric_zero_tail_log = parse_expr(&mut ctx, "ln((a*x + 1)/(x^2 + 1))");
    let finite_ratio_ln = parse_expr(&mut ctx, "ln((2*x^2 + 1)/(x^2 + 1))");
    let finite_ratio_log2 = parse_expr(&mut ctx, "log2((2*x^2 + 1)/(x^2 + 1))");
    let finite_ratio_log10 = parse_expr(&mut ctx, "log10((100*x^2 + 1)/(x^2 + 1))");
    let finite_ratio_base_log = parse_expr(&mut ctx, "log(2, (2*x^2 + 1)/(x^2 + 1))");
    let finite_ratio_reciprocal_base_log = parse_expr(&mut ctx, "log(1/2, (2*x^2 + 1)/(x^2 + 1))");
    let unit_ratio_ln = parse_expr(&mut ctx, "ln((x^2 + 1)/(x^2 + 1))");
    let finite_ratio_sqrt = parse_expr(&mut ctx, "sqrt((2*x^2 + 1)/(x^2 + 1))");
    let finite_square_ratio_sqrt = parse_expr(&mut ctx, "sqrt((4*x^2 + 1)/(x^2 + 1))");
    let unit_ratio_sqrt = parse_expr(&mut ctx, "sqrt((x^2 + 1)/(x^2 + 1))");
    let proper_sqrt = parse_expr(&mut ctx, "sqrt((x + 1)/(x^2 + 1))");
    let negative_direction_proper_sqrt = parse_expr(&mut ctx, "sqrt((1 - x)/(x^2 + 1))");
    let negative_finite_ratio_sqrt = parse_expr(&mut ctx, "sqrt((1 - 2*x^2)/(x^2 + 1))");
    let parametric_zero_tail_sqrt = parse_expr(&mut ctx, "sqrt((a*x + 1)/(x^2 + 1))");
    let parametric_finite_ratio_sqrt = parse_expr(&mut ctx, "sqrt((a*x^2 + 1)/(x^2 + 1))");
    let unbounded_ratio_cbrt = parse_expr(&mut ctx, "cbrt((x^2 + 1)/(x + 1))");
    let negative_unbounded_ratio_cbrt = parse_expr(&mut ctx, "cbrt((x^2 + 1)/(1 - x))");
    let finite_ratio_cbrt = parse_expr(&mut ctx, "cbrt((8*x^2 + 1)/(x^2 + 1))");
    let negative_finite_ratio_cbrt = parse_expr(&mut ctx, "cbrt((1 - 8*x^2)/(x^2 + 1))");
    let nonexact_finite_ratio_cbrt = parse_expr(&mut ctx, "cbrt((2*x^2 + 1)/(x^2 + 1))");
    let proper_cbrt = parse_expr(&mut ctx, "cbrt((x + 1)/(x^2 + 1))");
    let negative_zero_tail_cbrt = parse_expr(&mut ctx, "cbrt((1 - x)/(x^2 + 1))");
    let parametric_finite_ratio_cbrt = parse_expr(&mut ctx, "cbrt((a*x^2 + 1)/(x^2 + 1))");
    let finite_ratio_acosh = parse_expr(&mut ctx, "acosh((2*x^2 + 1)/(x^2 + 1))");
    let unit_ratio_acosh = parse_expr(&mut ctx, "acosh((x^2 + 1)/(x^2 + 1))");
    let small_finite_ratio_acosh = parse_expr(&mut ctx, "acosh((x^2 + 1)/(2*x^2 + 1))");
    let negative_finite_ratio_acosh = parse_expr(&mut ctx, "acosh((1 - 2*x^2)/(x^2 + 1))");
    let parametric_finite_ratio_acosh = parse_expr(&mut ctx, "acosh((a*x^2 + 1)/(x^2 + 1))");
    let negative_finite_ratio_log = parse_expr(&mut ctx, "ln((1 - 2*x^2)/(x^2 + 1))");
    let parametric_finite_ratio_log = parse_expr(&mut ctx, "ln((a*x^2 + 1)/(x^2 + 1))");

    let ln_out = elementary_function_limit_at_infinity(&mut ctx, ln_rational, x, InfSign::Pos)
        .expect("ln rational positive unbounded");
    let base_log_out =
        elementary_function_limit_at_infinity(&mut ctx, base_log_rational, x, InfSign::Pos)
            .expect("base log rational positive unbounded");
    let reciprocal_base_log_out = elementary_function_limit_at_infinity(
        &mut ctx,
        reciprocal_base_log_rational,
        x,
        InfSign::Pos,
    )
    .expect("base < 1 log rational positive unbounded");
    let acosh_out =
        elementary_function_limit_at_infinity(&mut ctx, acosh_rational, x, InfSign::Pos)
            .expect("acosh rational positive unbounded");
    let proper_ln_out = elementary_function_limit_at_infinity(&mut ctx, proper_ln, x, InfSign::Pos)
        .expect("ln rational positive zero tail");
    let proper_log2_out =
        elementary_function_limit_at_infinity(&mut ctx, proper_log2, x, InfSign::Pos)
            .expect("log2 rational positive zero tail");
    let proper_base_log_out =
        elementary_function_limit_at_infinity(&mut ctx, proper_base_log, x, InfSign::Pos)
            .expect("base log rational positive zero tail");
    let proper_reciprocal_base_log_out = elementary_function_limit_at_infinity(
        &mut ctx,
        proper_reciprocal_base_log,
        x,
        InfSign::Pos,
    )
    .expect("base < 1 log rational positive zero tail");
    let constant_num_log_out =
        elementary_function_limit_at_infinity(&mut ctx, constant_num_log, x, InfSign::Pos)
            .expect("log2 rational positive zero tail with constant numerator");
    let finite_ratio_ln_out =
        elementary_function_limit_at_infinity(&mut ctx, finite_ratio_ln, x, InfSign::Pos)
            .expect("ln rational positive finite tail");
    let finite_ratio_log2_out =
        elementary_function_limit_at_infinity(&mut ctx, finite_ratio_log2, x, InfSign::Pos)
            .expect("log2 rational positive finite tail");
    let finite_ratio_log10_out =
        elementary_function_limit_at_infinity(&mut ctx, finite_ratio_log10, x, InfSign::Pos)
            .expect("log10 rational positive finite tail");
    let finite_ratio_base_log_out =
        elementary_function_limit_at_infinity(&mut ctx, finite_ratio_base_log, x, InfSign::Pos)
            .expect("base log rational positive finite tail");
    let finite_ratio_reciprocal_base_log_out = elementary_function_limit_at_infinity(
        &mut ctx,
        finite_ratio_reciprocal_base_log,
        x,
        InfSign::Pos,
    )
    .expect("base < 1 log rational positive finite tail");
    let unit_ratio_ln_out =
        elementary_function_limit_at_infinity(&mut ctx, unit_ratio_ln, x, InfSign::Pos)
            .expect("ln rational unit finite tail");
    let finite_ratio_sqrt_out =
        elementary_function_limit_at_infinity(&mut ctx, finite_ratio_sqrt, x, InfSign::Pos)
            .expect("sqrt rational positive finite tail");
    let finite_square_ratio_sqrt_out =
        elementary_function_limit_at_infinity(&mut ctx, finite_square_ratio_sqrt, x, InfSign::Pos)
            .expect("sqrt rational positive finite square tail");
    let unit_ratio_sqrt_out =
        elementary_function_limit_at_infinity(&mut ctx, unit_ratio_sqrt, x, InfSign::Pos)
            .expect("sqrt rational unit finite tail");
    let proper_sqrt_out =
        elementary_function_limit_at_infinity(&mut ctx, proper_sqrt, x, InfSign::Pos)
            .expect("sqrt rational positive zero tail");
    let negative_direction_proper_sqrt_out = elementary_function_limit_at_infinity(
        &mut ctx,
        negative_direction_proper_sqrt,
        x,
        InfSign::Neg,
    )
    .expect("sqrt rational positive zero tail at negative infinity");
    let unbounded_ratio_cbrt_out =
        elementary_function_limit_at_infinity(&mut ctx, unbounded_ratio_cbrt, x, InfSign::Pos)
            .expect("cbrt rational positive unbounded tail");
    let negative_unbounded_ratio_cbrt_out = elementary_function_limit_at_infinity(
        &mut ctx,
        negative_unbounded_ratio_cbrt,
        x,
        InfSign::Pos,
    )
    .expect("cbrt rational negative unbounded tail");
    let finite_ratio_cbrt_out =
        elementary_function_limit_at_infinity(&mut ctx, finite_ratio_cbrt, x, InfSign::Pos)
            .expect("cbrt rational exact finite tail");
    let negative_finite_ratio_cbrt_out = elementary_function_limit_at_infinity(
        &mut ctx,
        negative_finite_ratio_cbrt,
        x,
        InfSign::Pos,
    )
    .expect("cbrt rational exact negative finite tail");
    let nonexact_finite_ratio_cbrt_out = elementary_function_limit_at_infinity(
        &mut ctx,
        nonexact_finite_ratio_cbrt,
        x,
        InfSign::Pos,
    )
    .expect("cbrt rational non-exact finite tail");
    let proper_cbrt_out =
        elementary_function_limit_at_infinity(&mut ctx, proper_cbrt, x, InfSign::Pos)
            .expect("cbrt rational positive zero tail");
    let negative_zero_tail_cbrt_out =
        elementary_function_limit_at_infinity(&mut ctx, negative_zero_tail_cbrt, x, InfSign::Pos)
            .expect("cbrt rational negative zero tail");
    let finite_ratio_acosh_out =
        elementary_function_limit_at_infinity(&mut ctx, finite_ratio_acosh, x, InfSign::Pos)
            .expect("acosh rational finite tail strictly inside domain");

    assert!(matches!(
        ctx.get(ln_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(base_log_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(reciprocal_base_log_out), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
    );
    assert!(matches!(
        ctx.get(acosh_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(proper_ln_out), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
    );
    assert!(
        matches!(ctx.get(proper_log2_out), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
    );
    assert!(
        matches!(ctx.get(proper_base_log_out), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
    );
    assert!(matches!(
        ctx.get(proper_reciprocal_base_log_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(constant_num_log_out), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
    );
    assert_eq!(display_expr(&ctx, finite_ratio_ln_out), "ln(2)");
    assert!(matches!(
        ctx.get(finite_ratio_log2_out),
        Expr::Number(value) if value == &rational_one()
    ));
    assert!(matches!(
        ctx.get(finite_ratio_log10_out),
        Expr::Number(value) if value == &BigRational::from_integer(BigInt::from(2))
    ));
    assert!(matches!(
        ctx.get(finite_ratio_base_log_out),
        Expr::Number(value) if value == &rational_one()
    ));
    assert!(matches!(
        ctx.get(finite_ratio_reciprocal_base_log_out),
        Expr::Number(value) if value == &BigRational::from_integer(BigInt::from(-1))
    ));
    assert!(matches!(
        ctx.get(unit_ratio_ln_out),
        Expr::Number(value) if value.is_zero()
    ));
    assert_eq!(display_expr(&ctx, finite_ratio_sqrt_out), "sqrt(2)");
    assert!(matches!(
        ctx.get(finite_square_ratio_sqrt_out),
        Expr::Number(value) if value == &BigRational::from_integer(BigInt::from(2))
    ));
    assert!(matches!(
        ctx.get(unit_ratio_sqrt_out),
        Expr::Number(value) if value == &rational_one()
    ));
    assert!(matches!(ctx.get(proper_sqrt_out), Expr::Number(value) if value.is_zero()));
    assert!(
        matches!(ctx.get(negative_direction_proper_sqrt_out), Expr::Number(value) if value.is_zero())
    );
    assert!(matches!(
        ctx.get(unbounded_ratio_cbrt_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(negative_unbounded_ratio_cbrt_out), Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)))
    );
    assert!(matches!(
        ctx.get(finite_ratio_cbrt_out),
        Expr::Number(value) if value == &BigRational::from_integer(BigInt::from(2))
    ));
    assert!(matches!(
        ctx.get(negative_finite_ratio_cbrt_out),
        Expr::Number(value) if value == &BigRational::from_integer(BigInt::from(-2))
    ));
    assert_eq!(
        display_expr(&ctx, nonexact_finite_ratio_cbrt_out),
        "cbrt(2)"
    );
    assert!(matches!(ctx.get(proper_cbrt_out), Expr::Number(value) if value.is_zero()));
    assert!(matches!(ctx.get(negative_zero_tail_cbrt_out), Expr::Number(value) if value.is_zero()));
    assert_eq!(display_expr(&ctx, finite_ratio_acosh_out), "acosh(2)");
    assert!(
        elementary_function_limit_at_infinity(&mut ctx, negative_tail_log, x, InfSign::Pos)
            .is_none()
    );
    assert!(
        elementary_function_limit_at_infinity(&mut ctx, proper_acosh, x, InfSign::Pos).is_none()
    );
    assert!(
        elementary_function_limit_at_infinity(&mut ctx, parametric_log, x, InfSign::Pos).is_none()
    );
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        negative_zero_tail_log,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        parametric_zero_tail_log,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        negative_direction_proper_sqrt,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        parametric_zero_tail_sqrt,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        negative_finite_ratio_sqrt,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        parametric_finite_ratio_sqrt,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        parametric_finite_ratio_cbrt,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(
        elementary_function_limit_at_infinity(&mut ctx, unit_ratio_acosh, x, InfSign::Pos)
            .is_none()
    );
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        small_finite_ratio_acosh,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        negative_finite_ratio_acosh,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        parametric_finite_ratio_acosh,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        negative_finite_ratio_log,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(elementary_function_limit_at_infinity(
        &mut ctx,
        parametric_finite_ratio_log,
        x,
        InfSign::Pos
    )
    .is_none());
}

#[test]
fn additive_limit_at_infinity_combines_finite_and_infinite_terms_conservatively() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let sqrt_plus_one = parse_expr(&mut ctx, "sqrt(x) + 1");
    let decaying_exp_plus_one = parse_expr(&mut ctx, "exp(-x) + 1");
    let exp_minus_poly = parse_expr(&mut ctx, "exp(x) - x^2");
    let poly_cancel = parse_expr(&mut ctx, "x^2 - x^2");

    let sqrt_plus_one_out = try_limit_rules_at_infinity(&mut ctx, sqrt_plus_one, x, InfSign::Pos)
        .expect("sqrt plus one");
    let decaying_exp_plus_one_out =
        try_limit_rules_at_infinity(&mut ctx, decaying_exp_plus_one, x, InfSign::Pos)
            .expect("decaying exp plus one");
    let poly_cancel_out = try_limit_rules_at_infinity(&mut ctx, poly_cancel, x, InfSign::Pos)
        .expect("polynomial cancellation");

    assert!(matches!(
        ctx.get(sqrt_plus_one_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(decaying_exp_plus_one_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(1)))
    );
    let exp_minus_poly_out = try_limit_rules_at_infinity(&mut ctx, exp_minus_poly, x, InfSign::Pos)
        .expect("exp dominates polynomial");
    assert!(matches!(
        ctx.get(exp_minus_poly_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(poly_cancel_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
}

#[test]
fn at_infinity_composition_handles_symbolic_finite_factors_and_radical_tails() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    // Symbolic finite factors compose (pi/2 from arctan).
    for source in ["2*arctan(x)", "arctan(x)/2", "arctan(x)*arctan(x)"] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            try_limit_rules_at_infinity(&mut ctx, expr, x, InfSign::Pos).is_some(),
            "must resolve: {source}"
        );
    }
    // Radical unbounded tails reach the saturating composition table.
    let cases = [
        ("arctan(sqrt(x))", "pi / 2"),
        ("arctan(-sqrt(x))", "-pi / 2"),
        ("tanh(sqrt(x))", "1"),
        ("e^(-sqrt(x))", "0"),
        ("arctan(x^(3/2))", "pi / 2"),
    ];
    for (source, expected) in cases {
        let expr = parse_expr(&mut ctx, source);
        let out = try_limit_rules_at_infinity(&mut ctx, expr, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("must resolve: {source}"));
        assert_eq!(
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &ctx,
                    id: out
                }
            ),
            expected,
            "{source}"
        );
    }
    // x * arctan(x) stays refused here: infinite times symbolic
    // finite needs a numeric scale to fix the sign.
    let mixed = parse_expr(&mut ctx, "x*arctan(x)");
    assert!(try_limit_rules_at_infinity(&mut ctx, mixed, x, InfSign::Pos).is_none());
}

#[test]
fn multiplicative_limit_at_infinity_combines_only_determined_products_and_quotients() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let scaled_sqrt = parse_expr(&mut ctx, "2*sqrt(x)");
    let neg_sqrt = parse_expr(&mut ctx, "-sqrt(x)");
    let reciprocal_exp = parse_expr(&mut ctx, "1/exp(x)");
    let indeterminate_exp_difference = parse_expr(&mut ctx, "exp(x)-exp(x)");

    let scaled_sqrt_out =
        try_limit_rules_at_infinity(&mut ctx, scaled_sqrt, x, InfSign::Pos).expect("scaled sqrt");
    let neg_sqrt_out =
        try_limit_rules_at_infinity(&mut ctx, neg_sqrt, x, InfSign::Pos).expect("neg sqrt");
    let reciprocal_exp_out = try_limit_rules_at_infinity(&mut ctx, reciprocal_exp, x, InfSign::Pos)
        .expect("reciprocal exp");

    assert!(matches!(
        ctx.get(scaled_sqrt_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(ctx.get(neg_sqrt_out), Expr::Neg(_)));
    assert!(
        matches!(ctx.get(reciprocal_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(
        try_limit_rules_at_infinity(&mut ctx, indeterminate_exp_difference, x, InfSign::Pos)
            .is_none()
    );
}

#[test]
fn exponential_polynomial_dominance_handles_only_exact_safe_shapes() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let exp_minus_poly = parse_expr(&mut ctx, "exp(x) - x^2");
    let poly_minus_exp = parse_expr(&mut ctx, "x^2 - exp(x)");
    let poly_over_exp = parse_expr(&mut ctx, "x^2/exp(x)");
    let exp_over_poly = parse_expr(&mut ctx, "exp(x)/x^2");
    let poly_times_decaying_exp = parse_expr(&mut ctx, "x*exp(x)");
    let poly_over_linear_exp = parse_expr(&mut ctx, "x^2/exp(2*x)");
    let linear_exp_over_poly = parse_expr(&mut ctx, "exp(2*x)/x^2");
    let poly_times_decaying_linear_exp = parse_expr(&mut ctx, "x^2*exp(2*x)");
    let poly_over_decaying_linear_exp = parse_expr(&mut ctx, "x^2/exp(-2*x)");
    let constant_over_decaying_linear_exp = parse_expr(&mut ctx, "1/exp(-2*x)");
    let poly_over_polynomial_exp = parse_expr(&mut ctx, "x^2/exp(x^2)");
    let polynomial_exp_over_poly = parse_expr(&mut ctx, "exp(x^2)/x^2");
    let polynomial_times_decaying_polynomial_exp = parse_expr(&mut ctx, "x^2*exp(2 - x^4)");
    let polynomial_over_decaying_polynomial_exp = parse_expr(&mut ctx, "x/exp(-x^2)");
    let neg_scaled_polynomial_exp_over_poly = parse_expr(&mut ctx, "-2*exp(x^2)/x^2");
    let parametric_exp_over_poly = parse_expr(&mut ctx, "exp(a*x^2)/x");
    let nested_exp_over_poly = parse_expr(&mut ctx, "exp(exp(x^2))/x");
    let even_poly_over_exp_neg = parse_expr(&mut ctx, "x^2/exp(x)");
    let odd_poly_over_exp_neg = parse_expr(&mut ctx, "x/exp(x)");
    let neg_scaled_exp_den_neg = parse_expr(&mut ctx, "x^2/(-2*exp(x))");
    let zero_scaled_exp_den = parse_expr(&mut ctx, "x^2/(0*exp(x))");
    let zero_scaled_linear_exp_den = parse_expr(&mut ctx, "x^2/(0*exp(2*x))");

    let exp_minus_poly_out = try_limit_rules_at_infinity(&mut ctx, exp_minus_poly, x, InfSign::Pos)
        .expect("exp dominates polynomial difference");
    let poly_minus_exp_out = try_limit_rules_at_infinity(&mut ctx, poly_minus_exp, x, InfSign::Pos)
        .expect("negative exponential dominance");
    let poly_over_exp_out = try_limit_rules_at_infinity(&mut ctx, poly_over_exp, x, InfSign::Pos)
        .expect("polynomial over exp");
    let exp_over_poly_out = try_limit_rules_at_infinity(&mut ctx, exp_over_poly, x, InfSign::Pos)
        .expect("exp over polynomial");
    let poly_times_decaying_exp_out =
        try_limit_rules_at_infinity(&mut ctx, poly_times_decaying_exp, x, InfSign::Neg)
            .expect("polynomial times decaying exp");
    let poly_over_linear_exp_out =
        try_limit_rules_at_infinity(&mut ctx, poly_over_linear_exp, x, InfSign::Pos)
            .expect("polynomial over linear exp");
    let linear_exp_over_poly_out =
        try_limit_rules_at_infinity(&mut ctx, linear_exp_over_poly, x, InfSign::Pos)
            .expect("linear exp over polynomial");
    let poly_times_decaying_linear_exp_out =
        try_limit_rules_at_infinity(&mut ctx, poly_times_decaying_linear_exp, x, InfSign::Neg)
            .expect("polynomial times decaying linear exp");
    let poly_over_decaying_linear_exp_out =
        try_limit_rules_at_infinity(&mut ctx, poly_over_decaying_linear_exp, x, InfSign::Pos)
            .expect("polynomial over decaying linear exp");
    let constant_over_decaying_linear_exp_out =
        try_limit_rules_at_infinity(&mut ctx, constant_over_decaying_linear_exp, x, InfSign::Pos)
            .expect("constant over decaying linear exp");
    let even_poly_over_exp_neg_out =
        try_limit_rules_at_infinity(&mut ctx, even_poly_over_exp_neg, x, InfSign::Neg)
            .expect("even polynomial over decaying exp");
    let odd_poly_over_exp_neg_out =
        try_limit_rules_at_infinity(&mut ctx, odd_poly_over_exp_neg, x, InfSign::Neg)
            .expect("odd polynomial over decaying exp");
    let neg_scaled_exp_den_neg_out =
        try_limit_rules_at_infinity(&mut ctx, neg_scaled_exp_den_neg, x, InfSign::Neg)
            .expect("negative scaled exp denominator");
    let poly_over_polynomial_exp_out =
        try_limit_rules_at_infinity(&mut ctx, poly_over_polynomial_exp, x, InfSign::Pos)
            .expect("polynomial over polynomial exp");
    let polynomial_exp_over_poly_out =
        try_limit_rules_at_infinity(&mut ctx, polynomial_exp_over_poly, x, InfSign::Pos)
            .expect("polynomial exp over polynomial");
    let polynomial_times_decaying_polynomial_exp_out = try_limit_rules_at_infinity(
        &mut ctx,
        polynomial_times_decaying_polynomial_exp,
        x,
        InfSign::Pos,
    )
    .expect("polynomial times decaying polynomial exp");
    let polynomial_over_decaying_polynomial_exp_out = try_limit_rules_at_infinity(
        &mut ctx,
        polynomial_over_decaying_polynomial_exp,
        x,
        InfSign::Neg,
    )
    .expect("polynomial over decaying polynomial exp");
    let neg_scaled_polynomial_exp_over_poly_out = try_limit_rules_at_infinity(
        &mut ctx,
        neg_scaled_polynomial_exp_over_poly,
        x,
        InfSign::Pos,
    )
    .expect("negative scaled polynomial exp over polynomial");

    assert!(matches!(
        ctx.get(exp_minus_poly_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(ctx.get(poly_minus_exp_out), Expr::Neg(_)));
    assert!(
        matches!(ctx.get(poly_over_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(exp_over_poly_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(poly_times_decaying_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(
        matches!(ctx.get(poly_over_linear_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(linear_exp_over_poly_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(poly_times_decaying_linear_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(poly_over_decaying_linear_exp_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(constant_over_decaying_linear_exp_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(even_poly_over_exp_neg_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(ctx.get(odd_poly_over_exp_neg_out), Expr::Neg(_)));
    assert!(matches!(ctx.get(neg_scaled_exp_den_neg_out), Expr::Neg(_)));
    assert!(
        matches!(ctx.get(poly_over_polynomial_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(polynomial_exp_over_poly_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(polynomial_times_decaying_polynomial_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(polynomial_over_decaying_polynomial_exp_out),
        Expr::Neg(_)
    ));
    assert!(matches!(
        ctx.get(neg_scaled_polynomial_exp_over_poly_out),
        Expr::Neg(_)
    ));
    assert!(
        try_limit_rules_at_infinity(&mut ctx, parametric_exp_over_poly, x, InfSign::Pos).is_none()
    );
    assert!(try_limit_rules_at_infinity(&mut ctx, nested_exp_over_poly, x, InfSign::Pos).is_none());
    assert!(try_limit_rules_at_infinity(&mut ctx, zero_scaled_exp_den, x, InfSign::Pos).is_none());
    assert!(
        try_limit_rules_at_infinity(&mut ctx, zero_scaled_linear_exp_den, x, InfSign::Pos)
            .is_none()
    );
}

#[test]
fn subpolynomial_polynomial_dominance_handles_only_domain_safe_shapes() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let log_over_poly = parse_expr(&mut ctx, "ln(x)/x");
    let poly_over_log = parse_expr(&mut ctx, "x/ln(x)");
    let root_over_poly = parse_expr(&mut ctx, "sqrt(x)/x");
    let poly_over_root = parse_expr(&mut ctx, "x/sqrt(x)");
    let cbrt_over_poly = parse_expr(&mut ctx, "cbrt(x)/x");
    let poly_over_cbrt = parse_expr(&mut ctx, "x/cbrt(x)");
    let poly_over_neg_tail_cbrt = parse_expr(&mut ctx, "x/cbrt(1 - x)");
    let asinh_over_poly = parse_expr(&mut ctx, "asinh(x)/x");
    let poly_over_asinh = parse_expr(&mut ctx, "x/asinh(x)");
    let poly_over_neg_tail_asinh = parse_expr(&mut ctx, "x/asinh(1 - x)");
    let acosh_over_poly = parse_expr(&mut ctx, "acosh(x)/x");
    let poly_over_acosh = parse_expr(&mut ctx, "x/acosh(x)");
    let poly_arg_acosh_over_poly = parse_expr(&mut ctx, "acosh(x^2)/x");
    let shifted_poly_arg_acosh_over_poly = parse_expr(&mut ctx, "acosh(x^2 - 3)/x");
    let poly_over_poly_arg_acosh = parse_expr(&mut ctx, "x/acosh(x^2)");
    let neg_tail_acosh_over_even_poly = parse_expr(&mut ctx, "acosh(1 - x)/x^2");
    let neg_tail_poly_over_acosh = parse_expr(&mut ctx, "x/acosh(1 - x)");
    let base_log_over_poly = parse_expr(&mut ctx, "log(2, x)/x");
    let unary_log10_over_poly = parse_expr(&mut ctx, "log10(x)/x");
    let poly_arg_log_over_poly = parse_expr(&mut ctx, "ln(x^2)/x");
    let shifted_poly_arg_log_over_poly = parse_expr(&mut ctx, "ln(x^2 - 3)/x");
    let poly_over_poly_arg_log = parse_expr(&mut ctx, "x/ln(x^2)");
    let base_log_poly_arg_over_poly = parse_expr(&mut ctx, "log(2, x^2)/x");
    let poly_over_half_base_log = parse_expr(&mut ctx, "x/log(1/2, x)");
    let e_base_log_over_poly = parse_expr(&mut ctx, "log(e, x)/x");
    let pi_base_log_over_poly = parse_expr(&mut ctx, "log(pi, x)/x");
    let phi_base_log_over_poly = parse_expr(&mut ctx, "log(phi, x)/x");
    let poly_over_reciprocal_e_base_log = parse_expr(&mut ctx, "x/log(1/e, x)");
    let powered_e_base_log_over_poly = parse_expr(&mut ctx, "log(e^2, x)/x");
    let powered_phi_base_log_over_poly = parse_expr(&mut ctx, "log(phi^3, x)/x");
    let poly_over_negative_power_e_base_log = parse_expr(&mut ctx, "x/log(e^-2, x)");
    let poly_over_reciprocal_power_pi_base_log = parse_expr(&mut ctx, "x/log((1/pi)^2, x)");
    let neg_tail_log_over_poly = parse_expr(&mut ctx, "ln(1 - x)/x^2");
    let neg_tail_poly_over_log = parse_expr(&mut ctx, "x/ln(1 - x)");
    let log_minus_poly = parse_expr(&mut ctx, "ln(x) - x");
    let poly_minus_root = parse_expr(&mut ctx, "x - sqrt(x)");
    let bad_domain_log = parse_expr(&mut ctx, "ln(x)/x");
    let bad_domain_base_log = parse_expr(&mut ctx, "log(2, x)/x");
    let invalid_base_log = parse_expr(&mut ctx, "log(1, x)/x");
    let negative_named_base_log = parse_expr(&mut ctx, "log(-e, x)/x");
    let zero_power_named_base_log = parse_expr(&mut ctx, "log(e^0, x)/x");
    let negative_tail_poly_log = parse_expr(&mut ctx, "ln(3 - x^2)/x");
    let parametric_leading_tail_poly_log = parse_expr(&mut ctx, "ln(a*x^2 + 1)/x");
    let nonlinear_cbrt_over_poly = parse_expr(&mut ctx, "cbrt(x^2)/x");
    let poly_over_nonlinear_cbrt = parse_expr(&mut ctx, "x/cbrt(x^2)");
    let nonlinear_asinh_over_poly = parse_expr(&mut ctx, "asinh(x^2)/x");
    let poly_over_nonlinear_asinh = parse_expr(&mut ctx, "x/asinh(x^2)");
    let bad_domain_acosh = parse_expr(&mut ctx, "acosh(x)/x");
    let bad_domain_neg_tail_acosh = parse_expr(&mut ctx, "x/acosh(1 - x)");
    let negative_tail_poly_arg_acosh_over_poly = parse_expr(&mut ctx, "acosh(3 - x^2)/x");
    let poly_over_negative_tail_poly_arg_acosh = parse_expr(&mut ctx, "x/acosh(3 - x^2)");
    let parametric_tail_poly_arg_acosh_over_poly = parse_expr(&mut ctx, "acosh(a*x^2 + 1)/x");
    let subpoly_over_subpoly = parse_expr(&mut ctx, "ln(x)/sqrt(x)");
    let zero_scaled_log_den = parse_expr(&mut ctx, "x/(0*ln(x))");

    let log_over_poly_out = try_limit_rules_at_infinity(&mut ctx, log_over_poly, x, InfSign::Pos)
        .expect("log over polynomial");
    let poly_over_log_out = try_limit_rules_at_infinity(&mut ctx, poly_over_log, x, InfSign::Pos)
        .expect("polynomial over log");
    let root_over_poly_out = try_limit_rules_at_infinity(&mut ctx, root_over_poly, x, InfSign::Pos)
        .expect("root over polynomial");
    let poly_over_root_out = try_limit_rules_at_infinity(&mut ctx, poly_over_root, x, InfSign::Pos)
        .expect("polynomial over root");
    let cbrt_over_poly_out = try_limit_rules_at_infinity(&mut ctx, cbrt_over_poly, x, InfSign::Pos)
        .expect("cube root over polynomial");
    let poly_over_cbrt_pos_out =
        try_limit_rules_at_infinity(&mut ctx, poly_over_cbrt, x, InfSign::Pos)
            .expect("polynomial over positive-tail cube root");
    let poly_over_cbrt_neg_out =
        try_limit_rules_at_infinity(&mut ctx, poly_over_cbrt, x, InfSign::Neg)
            .expect("polynomial over negative-tail cube root");
    let poly_over_neg_tail_cbrt_out =
        try_limit_rules_at_infinity(&mut ctx, poly_over_neg_tail_cbrt, x, InfSign::Pos)
            .expect("polynomial over negative linear-tail cube root");
    let asinh_over_poly_out =
        try_limit_rules_at_infinity(&mut ctx, asinh_over_poly, x, InfSign::Pos)
            .expect("asinh over polynomial");
    let poly_over_asinh_pos_out =
        try_limit_rules_at_infinity(&mut ctx, poly_over_asinh, x, InfSign::Pos)
            .expect("polynomial over positive-tail asinh");
    let poly_over_asinh_neg_out =
        try_limit_rules_at_infinity(&mut ctx, poly_over_asinh, x, InfSign::Neg)
            .expect("polynomial over negative-tail asinh");
    let poly_over_neg_tail_asinh_out =
        try_limit_rules_at_infinity(&mut ctx, poly_over_neg_tail_asinh, x, InfSign::Pos)
            .expect("polynomial over negative linear-tail asinh");
    let acosh_over_poly_out =
        try_limit_rules_at_infinity(&mut ctx, acosh_over_poly, x, InfSign::Pos)
            .expect("acosh over polynomial");
    let poly_over_acosh_out =
        try_limit_rules_at_infinity(&mut ctx, poly_over_acosh, x, InfSign::Pos)
            .expect("polynomial over positive-tail acosh");
    let poly_arg_acosh_over_poly_out =
        try_limit_rules_at_infinity(&mut ctx, poly_arg_acosh_over_poly, x, InfSign::Pos)
            .expect("polynomial-argument acosh over polynomial");
    let shifted_poly_arg_acosh_over_poly_out =
        try_limit_rules_at_infinity(&mut ctx, shifted_poly_arg_acosh_over_poly, x, InfSign::Pos)
            .expect("shifted polynomial-argument acosh over polynomial");
    let poly_over_poly_arg_acosh_pos_out =
        try_limit_rules_at_infinity(&mut ctx, poly_over_poly_arg_acosh, x, InfSign::Pos)
            .expect("polynomial over polynomial-argument acosh");
    let poly_over_poly_arg_acosh_neg_out =
        try_limit_rules_at_infinity(&mut ctx, poly_over_poly_arg_acosh, x, InfSign::Neg)
            .expect("negative-approach polynomial over polynomial-argument acosh");
    let neg_tail_acosh_over_even_poly_out =
        try_limit_rules_at_infinity(&mut ctx, neg_tail_acosh_over_even_poly, x, InfSign::Neg)
            .expect("negative-approach acosh over even polynomial");
    let neg_tail_poly_over_acosh_out =
        try_limit_rules_at_infinity(&mut ctx, neg_tail_poly_over_acosh, x, InfSign::Neg)
            .expect("negative-approach polynomial over acosh");
    let base_log_over_poly_out =
        try_limit_rules_at_infinity(&mut ctx, base_log_over_poly, x, InfSign::Pos)
            .expect("general-base log over polynomial");
    let unary_log10_over_poly_out =
        try_limit_rules_at_infinity(&mut ctx, unary_log10_over_poly, x, InfSign::Pos)
            .expect("log10 over polynomial");
    let poly_arg_log_over_poly_out =
        try_limit_rules_at_infinity(&mut ctx, poly_arg_log_over_poly, x, InfSign::Pos)
            .expect("polynomial-argument log over polynomial");
    let shifted_poly_arg_log_over_poly_out =
        try_limit_rules_at_infinity(&mut ctx, shifted_poly_arg_log_over_poly, x, InfSign::Pos)
            .expect("shifted polynomial-argument log over polynomial");
    let poly_over_poly_arg_log_out =
        try_limit_rules_at_infinity(&mut ctx, poly_over_poly_arg_log, x, InfSign::Pos)
            .expect("polynomial over polynomial-argument log");
    let base_log_poly_arg_over_poly_out =
        try_limit_rules_at_infinity(&mut ctx, base_log_poly_arg_over_poly, x, InfSign::Pos)
            .expect("base log with polynomial argument over polynomial");
    let poly_over_half_base_log_out =
        try_limit_rules_at_infinity(&mut ctx, poly_over_half_base_log, x, InfSign::Pos)
            .expect("polynomial over base < 1 log");
    let e_base_log_over_poly_out =
        try_limit_rules_at_infinity(&mut ctx, e_base_log_over_poly, x, InfSign::Pos)
            .expect("e-base log over polynomial");
    let pi_base_log_over_poly_out =
        try_limit_rules_at_infinity(&mut ctx, pi_base_log_over_poly, x, InfSign::Pos)
            .expect("pi-base log over polynomial");
    let phi_base_log_over_poly_out =
        try_limit_rules_at_infinity(&mut ctx, phi_base_log_over_poly, x, InfSign::Pos)
            .expect("phi-base log over polynomial");
    let poly_over_reciprocal_e_base_log_out =
        try_limit_rules_at_infinity(&mut ctx, poly_over_reciprocal_e_base_log, x, InfSign::Pos)
            .expect("polynomial over reciprocal e-base log");
    let powered_e_base_log_over_poly_out =
        try_limit_rules_at_infinity(&mut ctx, powered_e_base_log_over_poly, x, InfSign::Pos)
            .expect("powered e-base log over polynomial");
    let powered_phi_base_log_over_poly_out =
        try_limit_rules_at_infinity(&mut ctx, powered_phi_base_log_over_poly, x, InfSign::Pos)
            .expect("powered phi-base log over polynomial");
    let poly_over_negative_power_e_base_log_out = try_limit_rules_at_infinity(
        &mut ctx,
        poly_over_negative_power_e_base_log,
        x,
        InfSign::Pos,
    )
    .expect("polynomial over negative powered e-base log");
    let poly_over_reciprocal_power_pi_base_log_out = try_limit_rules_at_infinity(
        &mut ctx,
        poly_over_reciprocal_power_pi_base_log,
        x,
        InfSign::Pos,
    )
    .expect("polynomial over reciprocal power pi-base log");
    let neg_tail_log_over_poly_out =
        try_limit_rules_at_infinity(&mut ctx, neg_tail_log_over_poly, x, InfSign::Neg)
            .expect("negative-tail log over polynomial");
    let neg_tail_poly_over_log_out =
        try_limit_rules_at_infinity(&mut ctx, neg_tail_poly_over_log, x, InfSign::Neg)
            .expect("negative-tail polynomial over log");
    let log_minus_poly_out = try_limit_rules_at_infinity(&mut ctx, log_minus_poly, x, InfSign::Pos)
        .expect("log minus polynomial");
    let poly_minus_root_out =
        try_limit_rules_at_infinity(&mut ctx, poly_minus_root, x, InfSign::Pos)
            .expect("polynomial minus root");

    assert!(
        matches!(ctx.get(log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(poly_over_log_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(root_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(poly_over_root_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(cbrt_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(poly_over_cbrt_pos_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(poly_over_cbrt_neg_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(ctx.get(poly_over_neg_tail_cbrt_out), Expr::Neg(_)));
    assert!(
        matches!(ctx.get(asinh_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(poly_over_asinh_pos_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(poly_over_asinh_neg_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(poly_over_neg_tail_asinh_out),
        Expr::Neg(_)
    ));
    assert!(
        matches!(ctx.get(acosh_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(poly_over_acosh_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(poly_arg_acosh_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(
        matches!(ctx.get(shifted_poly_arg_acosh_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(poly_over_poly_arg_acosh_pos_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(poly_over_poly_arg_acosh_neg_out),
        Expr::Neg(_)
    ));
    assert!(
        matches!(ctx.get(neg_tail_acosh_over_even_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(neg_tail_poly_over_acosh_out),
        Expr::Neg(_)
    ));
    assert!(
        matches!(ctx.get(base_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(
        matches!(ctx.get(unary_log10_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(
        matches!(ctx.get(poly_arg_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(
        matches!(ctx.get(shifted_poly_arg_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(poly_over_poly_arg_log_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(base_log_poly_arg_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(ctx.get(poly_over_half_base_log_out), Expr::Neg(_)));
    assert!(
        matches!(ctx.get(e_base_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(
        matches!(ctx.get(pi_base_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(
        matches!(ctx.get(phi_base_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(poly_over_reciprocal_e_base_log_out),
        Expr::Neg(_)
    ));
    assert!(
        matches!(ctx.get(powered_e_base_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(
        matches!(ctx.get(powered_phi_base_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(poly_over_negative_power_e_base_log_out),
        Expr::Neg(_)
    ));
    assert!(matches!(
        ctx.get(poly_over_reciprocal_power_pi_base_log_out),
        Expr::Neg(_)
    ));
    assert!(
        matches!(ctx.get(neg_tail_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(ctx.get(neg_tail_poly_over_log_out), Expr::Neg(_)));
    assert!(matches!(ctx.get(log_minus_poly_out), Expr::Neg(_)));
    assert!(matches!(
        ctx.get(poly_minus_root_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(try_limit_rules_at_infinity(&mut ctx, bad_domain_log, x, InfSign::Neg).is_none());
    assert!(try_limit_rules_at_infinity(&mut ctx, bad_domain_base_log, x, InfSign::Neg).is_none());
    let invalid_base_log_out =
        try_limit_rules_at_infinity(&mut ctx, invalid_base_log, x, InfSign::Pos)
            .expect("invalid-base log has empty real domain");
    assert_eq!(display_expr(&ctx, invalid_base_log_out), "undefined");
    let negative_named_base_log_out =
        try_limit_rules_at_infinity(&mut ctx, negative_named_base_log, x, InfSign::Pos)
            .expect("negative-base log has empty real domain");
    assert_eq!(display_expr(&ctx, negative_named_base_log_out), "undefined");
    assert!(
        try_limit_rules_at_infinity(&mut ctx, zero_power_named_base_log, x, InfSign::Pos).is_none()
    );
    assert!(
        try_limit_rules_at_infinity(&mut ctx, negative_tail_poly_log, x, InfSign::Pos).is_none()
    );
    assert!(try_limit_rules_at_infinity(
        &mut ctx,
        parametric_leading_tail_poly_log,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(
        try_limit_rules_at_infinity(&mut ctx, nonlinear_cbrt_over_poly, x, InfSign::Pos).is_none()
    );
    assert!(
        try_limit_rules_at_infinity(&mut ctx, poly_over_nonlinear_cbrt, x, InfSign::Pos).is_none()
    );
    assert!(
        try_limit_rules_at_infinity(&mut ctx, nonlinear_asinh_over_poly, x, InfSign::Pos).is_none()
    );
    assert!(
        try_limit_rules_at_infinity(&mut ctx, poly_over_nonlinear_asinh, x, InfSign::Pos).is_none()
    );
    assert!(try_limit_rules_at_infinity(&mut ctx, bad_domain_acosh, x, InfSign::Neg).is_none());
    assert!(
        try_limit_rules_at_infinity(&mut ctx, bad_domain_neg_tail_acosh, x, InfSign::Pos).is_none()
    );
    assert!(try_limit_rules_at_infinity(
        &mut ctx,
        negative_tail_poly_arg_acosh_over_poly,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(try_limit_rules_at_infinity(
        &mut ctx,
        poly_over_negative_tail_poly_arg_acosh,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(try_limit_rules_at_infinity(
        &mut ctx,
        parametric_tail_poly_arg_acosh_over_poly,
        x,
        InfSign::Pos
    )
    .is_none());
    // ln(x)/sqrt(x): the subpolynomial/polynomial rule declines (sqrt is
    // not an integer-degree polynomial), but the polylog/power dominance
    // rule now resolves it to 0 (a positive power dominates the logarithm).
    let subpoly_over_subpoly_out =
        try_limit_rules_at_infinity(&mut ctx, subpoly_over_subpoly, x, InfSign::Pos)
            .expect("ln(x)/sqrt(x) resolves via polylog/power dominance");
    assert_eq!(display_expr(&ctx, subpoly_over_subpoly_out), "0");
    assert!(try_limit_rules_at_infinity(&mut ctx, zero_scaled_log_den, x, InfSign::Pos).is_none());
}

#[test]
fn polylog_power_dominance_at_infinity_resolves_fractional_and_higher_log() {
    // A positive power of x dominates any power of the logarithm:
    // ln(x)^a / x^b -> 0 and x^b / ln(x)^a -> +inf, for fractional b and
    // higher log powers a that the subpolynomial/polynomial rule misses.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (source, expected) in [
        ("ln(x)/sqrt(x)", "0"),
        ("ln(x)/x^(1/3)", "0"),
        ("ln(x)^2/x", "0"),
        ("ln(x)^3/x", "0"),
        ("ln(x)^2/x^2", "0"),
        ("ln(x)^2/sqrt(x)", "0"),
        ("sqrt(x)/ln(x)", "infinity"),
        ("x/ln(x)^2", "infinity"),
        // Negated power numerator (top-level Neg) flips the sign.
        ("-x/ln(x)", "-infinity"),
        ("-sqrt(x)/ln(x)^2", "-infinity"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = try_limit_rules_at_infinity(&mut ctx, expr, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("polylog/power dominance must resolve: {source}"));
        assert_eq!(display_expr(&ctx, out), expected, "{source}");
    }
}

#[test]
fn bounded_noise_rational_quotient_at_infinity_uses_leading_ratio() {
    // A polynomial plus bounded additive noise has the polynomial's
    // growth: (x + sin x)/x -> 1, the noise is dominated.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (source, expected) in [
        ("(x + sin(x))/x", "1"),
        ("(2*x + cos(x))/x", "2"),
        ("(x^2 + sin(x))/(x^2 - 1)", "1"),
        ("(x + sin(x))/(2*x + 1)", "1/2"),
        ("x/(x + sin(x))", "1"),
        ("(x + cos(x))/x^2", "0"),
        ("(x^2 + sin(x))/x", "infinity"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = try_limit_rules_at_infinity(&mut ctx, expr, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("bounded-noise quotient must resolve: {source}"));
        assert_eq!(display_expr(&ctx, out), expected, "{source}");
    }
}

#[test]
fn general_base_exponential_and_inf_to_zero_power_at_infinity() {
    // b^x growth and the inf^0 form, which together close (2^x+3^x)^(1/x)=3.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (source, expected) in [
        ("2^x", "infinity"),
        ("(1/2)^x", "0"),
        ("2^(-x)", "0"),
        ("1^x", "1"),
        ("(2^x+3^x)^(1/x)", "3"),
        ("(2^x+3^x+5^x)^(1/x)", "5"),
        ("x^(1/x)", "1"),
        ("(x^2+1)^(1/x)", "1"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = try_limit_rules_at_infinity(&mut ctx, expr, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("must resolve at +inf: {source}"));
        assert_eq!(display_expr(&ctx, out), expected, "{source}");
    }
    // At -infinity, 2^x -> 0.
    let two_x = parse_expr(&mut ctx, "2^x");
    let out = try_limit_rules_at_infinity(&mut ctx, two_x, x, InfSign::Neg).expect("2^x at -inf");
    assert_eq!(display_expr(&ctx, out), "0");
}

#[test]
fn constant_base_exponential_ratio_at_infinity() {
    // Same-exponent exponential quotients with PROVABLE constant bases:
    // e/π < 1 by exact rational interval bounds, so e^x/π^x decays and
    // the reciprocal ratio grows. The direct Pow form (e/π)^x classifies
    // through the same generalized base rule.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (source, expected) in [
        ("e^x/pi^x", "0"),
        ("exp(x)/pi^x", "0"),
        ("pi^x/e^x", "infinity"),
        ("(e/pi)^x", "0"),
        ("(pi/e)^x", "infinity"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = try_limit_rules_at_infinity(&mut ctx, expr, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("must resolve at +inf: {source}"));
        assert_eq!(display_expr(&ctx, out), expected, "{source}");
    }
    // At -infinity the decaying ratio flips: e^x/π^x -> +inf.
    let ratio = parse_expr(&mut ctx, "e^x/pi^x");
    let out =
        try_limit_rules_at_infinity(&mut ctx, ratio, x, InfSign::Neg).expect("e^x/pi^x at -inf");
    assert_eq!(display_expr(&ctx, out), "infinity");
    // Negative bases must DECLINE: (-2)^x/(-3)^x is not real-valued
    // along the reals, so combining to (2/3)^x would fabricate a limit.
    let negative = parse_expr(&mut ctx, "(-2)^x/(-3)^x");
    assert!(
        try_limit_rules_at_infinity(&mut ctx, negative, x, InfSign::Pos).is_none(),
        "negative bases must not combine"
    );
    // A base whose bounds straddle 1 (2e/(e+e) = 1 unsimplified) is
    // undecidable vs 1 by intervals: the rule must decline, honestly.
    let unit = parse_expr(&mut ctx, "(2*e/(e+e))^x");
    assert!(
        try_limit_rules_at_infinity(&mut ctx, unit, x, InfSign::Pos).is_none(),
        "unprovable base-vs-1 must decline"
    );
}

#[test]
fn provable_constant_base_dominance_at_infinity() {
    // The e-only exp-tail classifiers accept provable constant bases:
    // π^x beats any polynomial, (e/π)^x and π^(-x) decay against any
    // polynomial factor, and the root-enclosure arm proves sqrt(2) > 1
    // (closing sqrt(2)^x growth and the sqrt(2)^x/e^x ratio).
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (source, expected) in [
        ("x^5/pi^x", "0"),
        ("pi^x/x^5", "infinity"),
        ("x^4*(e/pi)^x", "0"),
        ("x^3*pi^(-x)", "0"),
        ("sqrt(2)^x", "infinity"),
        ("sqrt(2)^x/e^x", "0"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = try_limit_rules_at_infinity(&mut ctx, expr, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("must resolve at +inf: {source}"));
        assert_eq!(display_expr(&ctx, out), expected, "{source}");
    }
    // At -infinity π^x decays, so the polynomial product still dies.
    let product = parse_expr(&mut ctx, "x^5*pi^x");
    let out =
        try_limit_rules_at_infinity(&mut ctx, product, x, InfSign::Neg).expect("x^5*pi^x at -inf");
    assert_eq!(display_expr(&ctx, out), "0");
    // A disguised unit base (2e/(e+e) = 1 unsimplified) stays residual:
    // its bounds straddle 1, so no growth class is provable.
    let disguised = parse_expr(&mut ctx, "x^5/(2*e/(e+e))^x");
    assert!(
        try_limit_rules_at_infinity(&mut ctx, disguised, x, InfSign::Pos).is_none(),
        "unprovable base must keep the quotient residual"
    );
}

#[test]
fn inf_to_zero_power_declines_non_divergent_base() {
    // The base must diverge to +inf (positive, ln real). x^x (base -> inf
    // but exp does not -> 0) and a 1^inf base are NOT inf^0.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for source in ["x^x", "(1+1/x)^x"] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            inf_to_zero_power_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos).is_none(),
            "inf^0 must decline: {source}"
        );
    }
}

#[test]
fn log_exp_sum_dominance_resolves_to_log_of_dominant_base() {
    // ln(sum of exponentials)/x -> ln(dominant effective base).
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (source, expected) in [
        ("ln(2^x+3^x)/x", "ln(3)"),
        ("ln(2^x+3^x+5^x)/x", "ln(5)"),
        ("ln(2^x+1)/x", "ln(2)"),
        ("ln(5^x-3^x)/x", "ln(5)"),
        ("ln(2^(2*x)+3^x)/x", "ln(4)"),
        ("ln(2*5^x-5^x)/x", "ln(5)"),
        ("ln(3*2^x+3^x)/(2*x)", "1/2 * ln(3)"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = log_exp_sum_dominance_at_infinity(&mut ctx, expr, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("log-exp-sum must resolve: {source}"));
        assert_eq!(display_expr(&ctx, out), expected, "{source}");
    }
}

#[test]
fn log_exp_sum_dominance_declines_unsound_and_out_of_class() {
    // Negative dominant coefficient (sum -> -inf, ln undefined), an exact
    // dominant cancellation, a higher-order denominator, an e-base sum, and
    // a non-exponential argument all stay residual.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for source in [
        "ln(3^x-5^x)/x",
        "ln(5^x-5^x+3^x)/x",
        "ln(2^x+3^x)/x^2",
        "ln(exp(x)+exp(2*x))/x",
        "ln(x^2+1)/x",
    ] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            log_exp_sum_dominance_at_infinity(&mut ctx, expr, x, InfSign::Pos).is_none(),
            "log-exp-sum must decline: {source}"
        );
    }
}

#[test]
fn log_difference_at_infinity_collapses_to_log_of_ratio() {
    // ln(P) - ln(Q) -> ln(lim P/Q): a finite ln(lc_P/lc_Q) when degrees
    // match, +inf when P outgrows Q, -inf when Q outgrows P.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (source, expected) in [
        ("ln(x+1) - ln(x)", "0"),
        ("ln(2*x) - ln(x)", "ln(2)"),
        ("ln(x^2+x) - ln(x^2-x)", "0"),
        ("ln(3*x+1) - ln(x)", "ln(3)"),
        ("ln(x) - ln(2*x)", "ln(1/2)"),
        ("ln(x^2) - ln(x)", "infinity"),
        ("ln(x) - ln(x^2)", "-infinity"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = try_limit_rules_at_infinity(&mut ctx, expr, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("log difference must resolve: {source}"));
        assert_eq!(display_expr(&ctx, out), expected, "{source}");
    }
}

#[test]
fn log_difference_at_infinity_declines_non_polynomial_or_negative() {
    // A non-polynomial log argument, a negative-leading argument (ln
    // undefined as x -> +inf), and the wrong approach must decline.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for source in ["ln(sin(x)) - ln(x)", "ln(-x) - ln(x)"] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            log_difference_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos).is_none(),
            "log difference must decline: {source}"
        );
    }
    let neg_approach = parse_expr(&mut ctx, "ln(x+1) - ln(x)");
    assert!(
        log_difference_limit_at_infinity(&mut ctx, neg_approach, x, InfSign::Neg).is_none(),
        "x -> -inf makes ln undefined, so the rule must decline"
    );
}

#[test]
fn bounded_noise_rational_quotient_declines_unbounded_noise() {
    // x*sin(x) is unbounded, so (x + x sin x)/x has no limit and must
    // stay residual; a pure poly/poly ratio is left to the exact rule.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let unbounded = parse_expr(&mut ctx, "(x + x*sin(x))/x");
    assert!(
        bounded_noise_rational_limit_at_infinity(&mut ctx, unbounded, x, InfSign::Pos).is_none(),
        "unbounded noise must decline"
    );
    let pure_poly = parse_expr(&mut ctx, "(x^2 + 1)/(2*x^2 - 3)");
    assert!(
        bounded_noise_rational_limit_at_infinity(&mut ctx, pure_poly, x, InfSign::Pos).is_none(),
        "pure poly/poly is left to the exact rational rule"
    );
}

#[test]
fn polylog_power_dominance_declines_non_dominating_shapes() {
    // Not a polylog-over-power (or vice versa): a zero scale, a log/log
    // ratio, an oscillating factor, and the left (x -> -inf) approach
    // where the logarithm is undefined.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for source in ["x/(0*ln(x))", "ln(x)/ln(x)", "sin(x)/x^(1/2)"] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            polylog_power_dominance_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos).is_none(),
            "polylog/power dominance must decline: {source}"
        );
    }
    let log_over_sqrt = parse_expr(&mut ctx, "ln(x)/sqrt(x)");
    assert!(
        polylog_power_dominance_limit_at_infinity(&mut ctx, log_over_sqrt, x, InfSign::Neg)
            .is_none(),
        "ln(x) is undefined as x -> -inf, so the rule must decline"
    );
}

#[test]
fn exponential_subpolynomial_dominance_handles_only_domain_safe_shapes() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let log_over_exp = parse_expr(&mut ctx, "ln(x)/exp(x)");
    let exp_over_log = parse_expr(&mut ctx, "exp(x)/ln(x)");
    let root_times_decaying_exp = parse_expr(&mut ctx, "sqrt(x)*exp(-x)");
    let cbrt_times_decaying_exp = parse_expr(&mut ctx, "cbrt(x)*exp(-x)");
    let asinh_times_decaying_exp = parse_expr(&mut ctx, "asinh(x)*exp(-x)");
    let acosh_times_decaying_exp = parse_expr(&mut ctx, "acosh(x)*exp(-x)");
    let poly_arg_acosh_times_decaying_exp = parse_expr(&mut ctx, "acosh(x^2)*exp(-x)");
    let log_over_decaying_exp = parse_expr(&mut ctx, "ln(x)/exp(-x)");
    let cbrt_over_decaying_exp = parse_expr(&mut ctx, "cbrt(1 - x)/exp(-x)");
    let asinh_over_decaying_exp = parse_expr(&mut ctx, "asinh(1 - x)/exp(-x)");
    let neg_tail_acosh_times_decaying_exp = parse_expr(&mut ctx, "acosh(1 - x)*exp(x)");
    let negative_log_over_decaying_exp = parse_expr(&mut ctx, "-ln(x)/exp(-x)");
    let exp_over_negative_root = parse_expr(&mut ctx, "exp(x)/(-sqrt(x))");
    let exp_over_neg_tail_cbrt = parse_expr(&mut ctx, "exp(x)/cbrt(1 - x)");
    let exp_over_neg_tail_asinh = parse_expr(&mut ctx, "exp(x)/asinh(1 - x)");
    let exp_over_acosh = parse_expr(&mut ctx, "exp(x)/acosh(x)");
    let base_log_over_exp = parse_expr(&mut ctx, "log(2, x)/exp(x)");
    let exp_over_half_base_log = parse_expr(&mut ctx, "exp(x)/log(1/2, x)");
    let exp_over_unary_log2 = parse_expr(&mut ctx, "exp(x)/log2(x)");
    let exp_over_e_base_log = parse_expr(&mut ctx, "exp(x)/log(e, x)");
    let exp_over_reciprocal_e_base_log = parse_expr(&mut ctx, "exp(x)/log(1/e, x)");
    let exp_over_powered_e_base_log = parse_expr(&mut ctx, "exp(x)/log(e^2, x)");
    let exp_over_negative_power_e_base_log = parse_expr(&mut ctx, "exp(x)/log(e^-2, x)");
    let exp_minus_log = parse_expr(&mut ctx, "exp(x) - ln(x)");
    let log_minus_exp = parse_expr(&mut ctx, "ln(x) - exp(x)");
    let neg_tail_log_times_decaying_exp = parse_expr(&mut ctx, "ln(1 - x)*exp(x)");
    let bad_domain_log_over_exp = parse_expr(&mut ctx, "ln(x)/exp(-x)");
    let invalid_base_log_over_exp = parse_expr(&mut ctx, "log(1, x)/exp(x)");
    let negative_named_base_log_over_exp = parse_expr(&mut ctx, "log(-e, x)/exp(x)");
    let polynomial_exp_over_log = parse_expr(&mut ctx, "exp(x^2)/ln(x)");
    let polynomial_exp_over_cbrt = parse_expr(&mut ctx, "exp(x^2)/cbrt(x)");
    let log_over_polynomial_exp = parse_expr(&mut ctx, "ln(x)/exp(x^2)");
    let root_times_decaying_polynomial_exp = parse_expr(&mut ctx, "sqrt(x)*exp(2 - x^4)");
    let polynomial_exp_over_negative_root = parse_expr(&mut ctx, "exp(x^2)/(-sqrt(x))");
    let parametric_polynomial_exp_over_log = parse_expr(&mut ctx, "exp(a*x^2)/ln(x)");
    let nested_polynomial_exp_over_log = parse_expr(&mut ctx, "exp(exp(x^2))/ln(x)");
    let exp_over_nonlinear_cbrt = parse_expr(&mut ctx, "exp(x)/cbrt(x^2)");
    let exp_over_nonlinear_asinh = parse_expr(&mut ctx, "exp(x)/asinh(x^2)");
    let bad_domain_exp_over_acosh = parse_expr(&mut ctx, "exp(x)/acosh(1 - x)");
    let exp_over_poly_arg_acosh = parse_expr(&mut ctx, "exp(x)/acosh(x^2)");
    let exp_over_negative_tail_poly_arg_acosh = parse_expr(&mut ctx, "exp(x)/acosh(3 - x^2)");
    let zero_exp_denominator = parse_expr(&mut ctx, "ln(x)/(0*exp(x))");

    let log_over_exp_out = try_limit_rules_at_infinity(&mut ctx, log_over_exp, x, InfSign::Pos)
        .expect("log over growing exp");
    let exp_over_log_out = try_limit_rules_at_infinity(&mut ctx, exp_over_log, x, InfSign::Pos)
        .expect("growing exp over log");
    let root_times_decaying_exp_out =
        try_limit_rules_at_infinity(&mut ctx, root_times_decaying_exp, x, InfSign::Pos)
            .expect("root times decaying exp");
    let cbrt_times_decaying_exp_out =
        try_limit_rules_at_infinity(&mut ctx, cbrt_times_decaying_exp, x, InfSign::Pos)
            .expect("cube root times decaying exp");
    let asinh_times_decaying_exp_out =
        try_limit_rules_at_infinity(&mut ctx, asinh_times_decaying_exp, x, InfSign::Pos)
            .expect("asinh times decaying exp");
    let acosh_times_decaying_exp_out =
        try_limit_rules_at_infinity(&mut ctx, acosh_times_decaying_exp, x, InfSign::Pos)
            .expect("acosh times decaying exp");
    let poly_arg_acosh_times_decaying_exp_out =
        try_limit_rules_at_infinity(&mut ctx, poly_arg_acosh_times_decaying_exp, x, InfSign::Pos)
            .expect("polynomial-argument acosh times decaying exp");
    let log_over_decaying_exp_out =
        try_limit_rules_at_infinity(&mut ctx, log_over_decaying_exp, x, InfSign::Pos)
            .expect("log over decaying exp");
    let cbrt_over_decaying_exp_out =
        try_limit_rules_at_infinity(&mut ctx, cbrt_over_decaying_exp, x, InfSign::Pos)
            .expect("negative-tail cube root over decaying exp");
    let asinh_over_decaying_exp_out =
        try_limit_rules_at_infinity(&mut ctx, asinh_over_decaying_exp, x, InfSign::Pos)
            .expect("negative-tail asinh over decaying exp");
    let neg_tail_acosh_times_decaying_exp_out =
        try_limit_rules_at_infinity(&mut ctx, neg_tail_acosh_times_decaying_exp, x, InfSign::Neg)
            .expect("negative-approach acosh times decaying exp");
    let negative_log_over_decaying_exp_out =
        try_limit_rules_at_infinity(&mut ctx, negative_log_over_decaying_exp, x, InfSign::Pos)
            .expect("negative log over decaying exp");
    let exp_over_negative_root_out =
        try_limit_rules_at_infinity(&mut ctx, exp_over_negative_root, x, InfSign::Pos)
            .expect("exp over negative root");
    let exp_over_neg_tail_cbrt_out =
        try_limit_rules_at_infinity(&mut ctx, exp_over_neg_tail_cbrt, x, InfSign::Pos)
            .expect("exp over negative-tail cube root");
    let exp_over_neg_tail_asinh_out =
        try_limit_rules_at_infinity(&mut ctx, exp_over_neg_tail_asinh, x, InfSign::Pos)
            .expect("exp over negative-tail asinh");
    let exp_over_acosh_out = try_limit_rules_at_infinity(&mut ctx, exp_over_acosh, x, InfSign::Pos)
        .expect("exp over positive-tail acosh");
    let exp_over_poly_arg_acosh_out =
        try_limit_rules_at_infinity(&mut ctx, exp_over_poly_arg_acosh, x, InfSign::Pos)
            .expect("exp over polynomial-argument acosh");
    let base_log_over_exp_out =
        try_limit_rules_at_infinity(&mut ctx, base_log_over_exp, x, InfSign::Pos)
            .expect("general-base log over growing exp");
    let exp_over_half_base_log_out =
        try_limit_rules_at_infinity(&mut ctx, exp_over_half_base_log, x, InfSign::Pos)
            .expect("growing exp over base < 1 log");
    let exp_over_unary_log2_out =
        try_limit_rules_at_infinity(&mut ctx, exp_over_unary_log2, x, InfSign::Pos)
            .expect("growing exp over log2");
    let exp_over_e_base_log_out =
        try_limit_rules_at_infinity(&mut ctx, exp_over_e_base_log, x, InfSign::Pos)
            .expect("growing exp over e-base log");
    let exp_over_reciprocal_e_base_log_out =
        try_limit_rules_at_infinity(&mut ctx, exp_over_reciprocal_e_base_log, x, InfSign::Pos)
            .expect("growing exp over reciprocal e-base log");
    let exp_over_powered_e_base_log_out =
        try_limit_rules_at_infinity(&mut ctx, exp_over_powered_e_base_log, x, InfSign::Pos)
            .expect("growing exp over powered e-base log");
    let exp_over_negative_power_e_base_log_out = try_limit_rules_at_infinity(
        &mut ctx,
        exp_over_negative_power_e_base_log,
        x,
        InfSign::Pos,
    )
    .expect("growing exp over negative powered e-base log");
    let exp_minus_log_out = try_limit_rules_at_infinity(&mut ctx, exp_minus_log, x, InfSign::Pos)
        .expect("exp minus log");
    let log_minus_exp_out = try_limit_rules_at_infinity(&mut ctx, log_minus_exp, x, InfSign::Pos)
        .expect("log minus exp");
    let neg_tail_log_times_decaying_exp_out =
        try_limit_rules_at_infinity(&mut ctx, neg_tail_log_times_decaying_exp, x, InfSign::Neg)
            .expect("negative-tail log times decaying exp");
    let polynomial_exp_over_log_out =
        try_limit_rules_at_infinity(&mut ctx, polynomial_exp_over_log, x, InfSign::Pos)
            .expect("polynomial exp over log");
    let polynomial_exp_over_cbrt_out =
        try_limit_rules_at_infinity(&mut ctx, polynomial_exp_over_cbrt, x, InfSign::Pos)
            .expect("polynomial exp over cube root");
    let log_over_polynomial_exp_out =
        try_limit_rules_at_infinity(&mut ctx, log_over_polynomial_exp, x, InfSign::Pos)
            .expect("log over polynomial exp");
    let root_times_decaying_polynomial_exp_out = try_limit_rules_at_infinity(
        &mut ctx,
        root_times_decaying_polynomial_exp,
        x,
        InfSign::Pos,
    )
    .expect("root times decaying polynomial exp");
    let polynomial_exp_over_negative_root_out =
        try_limit_rules_at_infinity(&mut ctx, polynomial_exp_over_negative_root, x, InfSign::Pos)
            .expect("polynomial exp over negative root");

    assert!(
        matches!(ctx.get(log_over_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(exp_over_log_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(root_times_decaying_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(
        matches!(ctx.get(cbrt_times_decaying_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(
        matches!(ctx.get(asinh_times_decaying_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(
        matches!(ctx.get(acosh_times_decaying_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(
        matches!(ctx.get(poly_arg_acosh_times_decaying_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(log_over_decaying_exp_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(ctx.get(cbrt_over_decaying_exp_out), Expr::Neg(_)));
    assert!(matches!(ctx.get(asinh_over_decaying_exp_out), Expr::Neg(_)));
    assert!(
        matches!(ctx.get(neg_tail_acosh_times_decaying_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(negative_log_over_decaying_exp_out),
        Expr::Neg(_)
    ));
    assert!(matches!(ctx.get(exp_over_negative_root_out), Expr::Neg(_)));
    assert!(matches!(ctx.get(exp_over_neg_tail_cbrt_out), Expr::Neg(_)));
    assert!(matches!(ctx.get(exp_over_neg_tail_asinh_out), Expr::Neg(_)));
    assert!(matches!(
        ctx.get(exp_over_acosh_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(exp_over_poly_arg_acosh_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(base_log_over_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(ctx.get(exp_over_half_base_log_out), Expr::Neg(_)));
    assert!(matches!(
        ctx.get(exp_over_unary_log2_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(exp_over_e_base_log_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(exp_over_reciprocal_e_base_log_out),
        Expr::Neg(_)
    ));
    assert!(matches!(
        ctx.get(exp_over_powered_e_base_log_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(exp_over_negative_power_e_base_log_out),
        Expr::Neg(_)
    ));
    assert!(matches!(
        ctx.get(exp_minus_log_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(ctx.get(log_minus_exp_out), Expr::Neg(_)));
    assert!(
        matches!(ctx.get(neg_tail_log_times_decaying_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(polynomial_exp_over_log_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(polynomial_exp_over_cbrt_out),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(
        matches!(ctx.get(log_over_polynomial_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(
        matches!(ctx.get(root_times_decaying_polynomial_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(polynomial_exp_over_negative_root_out),
        Expr::Neg(_)
    ));
    assert!(
        try_limit_rules_at_infinity(&mut ctx, bad_domain_log_over_exp, x, InfSign::Neg).is_none()
    );
    let invalid_base_log_over_exp_out =
        try_limit_rules_at_infinity(&mut ctx, invalid_base_log_over_exp, x, InfSign::Pos)
            .expect("invalid-base log over exp has empty real domain");
    assert_eq!(
        display_expr(&ctx, invalid_base_log_over_exp_out),
        "undefined"
    );
    let negative_named_base_log_over_exp_out =
        try_limit_rules_at_infinity(&mut ctx, negative_named_base_log_over_exp, x, InfSign::Pos)
            .expect("negative-base log over exp has empty real domain");
    assert_eq!(
        display_expr(&ctx, negative_named_base_log_over_exp_out),
        "undefined"
    );
    assert!(try_limit_rules_at_infinity(
        &mut ctx,
        parametric_polynomial_exp_over_log,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(
        try_limit_rules_at_infinity(&mut ctx, nested_polynomial_exp_over_log, x, InfSign::Pos)
            .is_none()
    );
    assert!(
        try_limit_rules_at_infinity(&mut ctx, exp_over_nonlinear_cbrt, x, InfSign::Pos).is_none()
    );
    assert!(
        try_limit_rules_at_infinity(&mut ctx, exp_over_nonlinear_asinh, x, InfSign::Pos).is_none()
    );
    assert!(
        try_limit_rules_at_infinity(&mut ctx, bad_domain_exp_over_acosh, x, InfSign::Pos).is_none()
    );
    assert!(try_limit_rules_at_infinity(
        &mut ctx,
        exp_over_negative_tail_poly_arg_acosh,
        x,
        InfSign::Pos
    )
    .is_none());
    assert!(try_limit_rules_at_infinity(&mut ctx, zero_exp_denominator, x, InfSign::Pos).is_none());
}

#[test]
fn apply_power_rule_handles_zero_and_negative_exponents() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let x0 = parse_expr(&mut ctx, "x^0");
    let xneg = parse_expr(&mut ctx, "x^-3");

    let out0 = apply_power_rule(&mut ctx, x0, x, InfSign::Pos).expect("x^0");
    let out_neg = apply_power_rule(&mut ctx, xneg, x, InfSign::Neg).expect("x^-3");

    assert!(
        matches!(ctx.get(out0), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(1)))
    );
    assert!(
        matches!(ctx.get(out_neg), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
}

#[test]
fn apply_rational_power_rule_resolves_fractional_exponents_at_positive_infinity() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let neg_half = parse_expr(&mut ctx, "x^(-1/2)");
    let pos_half = parse_expr(&mut ctx, "x^(1/2)");
    let three_halves = parse_expr(&mut ctx, "x^(3/2)");

    let out_neg =
        apply_rational_power_rule(&mut ctx, neg_half, x, InfSign::Pos).expect("x^(-1/2) -> 0");
    let out_pos =
        apply_rational_power_rule(&mut ctx, pos_half, x, InfSign::Pos).expect("x^(1/2) -> ∞");
    let out_three =
        apply_rational_power_rule(&mut ctx, three_halves, x, InfSign::Pos).expect("x^(3/2) -> ∞");

    assert!(
        matches!(ctx.get(out_neg), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(matches!(
        ctx.get(out_pos),
        Expr::Constant(Constant::Infinity)
    ));
    assert!(matches!(
        ctx.get(out_three),
        Expr::Constant(Constant::Infinity)
    ));
}

#[test]
fn apply_rational_power_rule_resolves_decidable_irrational_exponents() {
    // Lo único que la regla decide es el SIGNO del exponente, y para eso ya
    // existe una capa exacta. Con `as_rational_const` a secas, `x^pi` o
    // `x^(pi-3)` no daban una respuesta INCORRECTA: no daban ninguna.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");

    for src in ["x^pi", "x^(pi - 3)", "x^e", "x^(pi/2)"] {
        let expr = parse_expr(&mut ctx, src);
        let out = apply_rational_power_rule(&mut ctx, expr, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("{src} -> ∞"));
        assert!(
            matches!(ctx.get(out), Expr::Constant(Constant::Infinity)),
            "{src}"
        );
    }

    for src in ["x^(3 - pi)", "x^(2 - e)"] {
        let expr = parse_expr(&mut ctx, src);
        let out = apply_rational_power_rule(&mut ctx, expr, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("{src} -> 0"));
        assert!(
            matches!(ctx.get(out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0))),
            "{src}"
        );
    }

    // Un exponente cuyo signo NO es decidible declina, y `x -> -∞` sigue
    // declinando porque una potencia no entera de una magnitud negativa no es
    // real — el mismo argumento que ya protegía a los racionales.
    let symbolic = parse_expr(&mut ctx, "x^a");
    assert!(apply_rational_power_rule(&mut ctx, symbolic, x, InfSign::Pos).is_none());
    let irrational = parse_expr(&mut ctx, "x^(pi - 3)");
    assert!(apply_rational_power_rule(&mut ctx, irrational, x, InfSign::Neg).is_none());
}

#[test]
fn exp_sum_top_level_decides_by_dominant_base() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (src, positive) in [
        ("3^x - 2^x", true),
        ("2^x - 3^x", false),
        ("2^x + 3^x - 4^x", false),
        ("5 * 2^x - 3^x", false),
    ] {
        let expr = parse_expr(&mut ctx, src);
        let out = exp_sum_top_level_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("{src} should resolve"));
        if positive {
            assert!(
                matches!(ctx.get(out), Expr::Constant(Constant::Infinity)),
                "{src}"
            );
        } else {
            assert!(matches!(ctx.get(out), Expr::Neg(_)), "{src}");
        }
    }

    // Declines: dominante con suma CERO (2^x − 2^x), término único (dueño
    // propio), −∞ (ese camino ya funciona por decaimiento), y bases no
    // racionales (pi/e quedan para la generalización a pares provables).
    let zero_dominant = parse_expr(&mut ctx, "2^x - 2^x");
    assert!(
        exp_sum_top_level_limit_at_infinity(&mut ctx, zero_dominant, x, InfSign::Pos).is_none()
    );
    let single = parse_expr(&mut ctx, "3^x");
    assert!(exp_sum_top_level_limit_at_infinity(&mut ctx, single, x, InfSign::Pos).is_none());
    let diff = parse_expr(&mut ctx, "3^x - 2^x");
    assert!(exp_sum_top_level_limit_at_infinity(&mut ctx, diff, x, InfSign::Neg).is_none());
    let transcendental = parse_expr(&mut ctx, "pi^x - e^x");
    assert!(
        exp_sum_top_level_limit_at_infinity(&mut ctx, transcendental, x, InfSign::Pos).is_none()
    );
}

#[test]
fn same_base_power_quotient_reduces_and_delegates() {
    // `x^a/x^b` con exponentes constantes → `x^(a−b)` → reglas de potencia.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (src, expect_inf) in [
        ("x^(5/2) / x^(3/2)", true),
        ("x^(3/2) / x^(5/2)", false),
        ("x^pi / x^3", true),
        ("x^3 / x^pi", false),
        ("x / x^(1/2)", true),
        ("x^(1/2) / x", false),
    ] {
        let expr = parse_expr(&mut ctx, src);
        let out = apply_same_base_power_quotient_rule(&mut ctx, expr, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("{src} should resolve"));
        if expect_inf {
            assert!(
                matches!(ctx.get(out), Expr::Constant(Constant::Infinity)),
                "{src}"
            );
        } else {
            assert!(
                matches!(ctx.get(out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0))),
                "{src}"
            );
        }
    }

    // Declines honestos: diferencia de signo indecidible o cero, exponente
    // dependiente de la variable, y −∞ con exponentes NO enteros (la
    // original no es real para x<0: reducirla fabricaría un límite de algo
    // indefinido — el wrong-answer que este guard cazó antes de commitear).
    for src in ["x^pi / x^pi", "x^a / x^2", "x^x / x^2"] {
        let expr = parse_expr(&mut ctx, src);
        assert!(
            apply_same_base_power_quotient_rule(&mut ctx, expr, x, InfSign::Pos).is_none(),
            "{src}"
        );
    }
    let fractional = parse_expr(&mut ctx, "x^(5/2) / x^(3/2)");
    assert!(apply_same_base_power_quotient_rule(&mut ctx, fractional, x, InfSign::Neg).is_none());
    let integers = parse_expr(&mut ctx, "x^5 / x^2");
    assert!(apply_same_base_power_quotient_rule(&mut ctx, integers, x, InfSign::Neg).is_some());
}

#[test]
fn apply_rational_power_rule_declines_integers_and_negative_infinity() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");

    // Integer exponents stay with `apply_power_rule`.
    let int_exp = parse_expr(&mut ctx, "x^2");
    assert!(apply_rational_power_rule(&mut ctx, int_exp, x, InfSign::Pos).is_none());

    // A non-integer power of a negative magnitude is not real-valued.
    let frac = parse_expr(&mut ctx, "x^(1/2)");
    assert!(apply_rational_power_rule(&mut ctx, frac, x, InfSign::Neg).is_none());

    // Base must be exactly the limit variable.
    let other = parse_expr(&mut ctx, "y^(1/2)");
    assert!(apply_rational_power_rule(&mut ctx, other, x, InfSign::Pos).is_none());
}

#[test]
fn apply_reciprocal_power_rule_handles_one_over_xn() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let expr1 = parse_expr(&mut ctx, "1/x");
    let expr2 = parse_expr(&mut ctx, "5/x^3");

    let out1 = apply_reciprocal_power_rule(&mut ctx, expr1, x).expect("1/x");
    let out2 = apply_reciprocal_power_rule(&mut ctx, expr2, x).expect("5/x^3");

    assert!(
        matches!(ctx.get(out1), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
    assert!(
        matches!(ctx.get(out2), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
}

#[test]
fn try_limit_rules_at_infinity_resolves_constant_and_variable() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let c = parse_expr(&mut ctx, "7");

    let c_out = try_limit_rules_at_infinity(&mut ctx, c, x, InfSign::Pos).expect("constant");
    let x_out = try_limit_rules_at_infinity(&mut ctx, x, x, InfSign::Neg).expect("variable");

    assert_eq!(c_out, c);
    assert!(matches!(ctx.get(x_out), Expr::Neg(_)));
}

#[test]
fn try_limit_rules_at_infinity_uses_rational_poly_fallback() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let expr = parse_expr(&mut ctx, "x^2/x^3");

    let out = try_limit_rules_at_infinity(&mut ctx, expr, x, InfSign::Pos).expect("rational");
    assert!(
        matches!(ctx.get(out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
    );
}

#[test]
fn exp_sum_quotient_dominance_resolves_by_dominant_base() {
    // A quotient of exponential sums is decided by the dominant base.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (source, expected) in [
        ("3^x/2^x", "infinity"),
        ("2^x/3^x", "0"),
        ("(2^x+3^x)/3^x", "1"),
        ("(3^x-2^x)/(3^x+2^x)", "1"),
        ("(2*3^x+2^x)/3^x", "2"),
        ("(5^x+2^x)/(3^x+4^x)", "infinity"),
        ("(-3^x)/2^x", "-infinity"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = exp_sum_quotient_dominance_at_infinity(&mut ctx, expr, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("exp quotient must resolve: {source}"));
        assert_eq!(display_expr(&ctx, out), expected, "{source}");
    }
}

#[test]
fn general_exp_vs_polynomial_dominance_resolves() {
    // An exponential (base > 1) beats any polynomial in a quotient.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (source, expected) in [
        ("2^x/x^2", "infinity"),
        ("x^10/2^x", "0"),
        ("2^x/x", "infinity"),
        ("(2^x+3^x)/x^5", "infinity"),
        ("x^3/(2^x+3^x)", "0"),
        ("(-2^x)/x^2", "-infinity"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = general_exp_vs_polynomial_dominance_at_infinity(&mut ctx, expr, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("exp-vs-poly must resolve: {source}"));
        assert_eq!(display_expr(&ctx, out), expected, "{source}");
    }
}

#[test]
fn polynomial_times_decaying_exponential_collapses_to_zero() {
    // A polynomial times a decaying exponential -> 0 (decay beats growth).
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (source, approach) in [
        ("x*2^(-x)", InfSign::Pos),
        ("x^3*2^(-x)", InfSign::Pos),
        ("x^2*(1/2)^x", InfSign::Pos),
        ("x*2^x", InfSign::Neg),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = polynomial_times_decaying_exponential_at_infinity(&mut ctx, expr, x, approach)
            .unwrap_or_else(|| panic!("poly*decaying-exp must resolve: {source}"));
        assert_eq!(display_expr(&ctx, out), "0", "{source}");
    }
    // A GROWING exponential factor is not this rule (x*2^x at +inf -> +inf).
    let grow = parse_expr(&mut ctx, "x*2^x");
    assert!(
        polynomial_times_decaying_exponential_at_infinity(&mut ctx, grow, x, InfSign::Pos)
            .is_none(),
        "a growing exponential is not the decaying-product rule"
    );
}

#[test]
fn general_exp_vs_polynomial_dominance_declines_pure_rational_and_decaying() {
    // Both polynomial (no exponential), and a decaying base (<= 1) which is
    // not a growing exponential, stay out of this rule.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for source in ["x^2/x^3", "(1/2)^x/x^2"] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            general_exp_vs_polynomial_dominance_at_infinity(&mut ctx, expr, x, InfSign::Pos)
                .is_none(),
            "exp-vs-poly must decline: {source}"
        );
    }
}

#[test]
fn exp_sum_quotient_dominance_declines_non_exponential_and_cancelled() {
    // No growing exponential on a side (pure rational, exp-vs-poly) and a
    // cancelled dominant denominator coefficient stay out of this rule.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for source in ["(x^2+1)/(x+1)", "2^x/x^2", "3^x/(2^x+3^x-3^x)"] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            exp_sum_quotient_dominance_at_infinity(&mut ctx, expr, x, InfSign::Pos).is_none(),
            "exp quotient must decline: {source}"
        );
    }
}

#[test]
fn rational_difference_at_infinity_combines_into_one_fraction() {
    // inf - inf of rational functions: the per-term additive rule leaves it
    // residual; combining over a common denominator resolves it.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (source, expected) in [
        ("(x^2+1)/(x+1) - x", "-1"),
        ("x^2/(x-1) - x", "1"),
        ("(x^3+1)/(x^2+1) - x", "0"),
        ("x - x^2/(x+1)", "1"),
        ("x^2/(x+1) - x^2/(x+2)", "1"),
        ("(2*x^2)/(x+1) - 2*x", "-2"),
        ("x^3/(x+1) - x", "infinity"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = rational_difference_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("rational difference must resolve: {source}"));
        assert_eq!(display_expr(&ctx, out), expected, "{source}");
    }
}

#[test]
fn rational_difference_at_infinity_declines_non_rational_operands() {
    // Non-rational operands (sqrt/sin/exp) make the multipoly conversion
    // fail, so the conjugate/elementary/dominance paths keep their forms.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for source in ["sqrt(x^2+1) - x", "sin(x) - x", "exp(x) - x"] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            rational_difference_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos).is_none(),
            "rational difference must decline non-rational operands: {source}"
        );
    }
}

#[test]
fn one_to_infinity_power_resolves_the_e_family() {
    // 1^inf: base -> 1, exponent -> inf, limit = exp(lim exp*(base-1)).
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (source, expected) in [
        ("(1+1/x)^x", "e"),
        ("(1+2/x)^x", "e^2"),
        ("(1+1/x)^(2*x)", "e^2"),
        ("(1+3/x)^(2*x)", "e^6"),
        ("(1-1/x)^x", "e^(-1)"),
        ("(1+1/x^2)^x", "1"),
        ("(1+1/x)^(x^2)", "infinity"),
        ("((2*x+1)/(2*x-1))^x", "e"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = one_to_infinity_power_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("1^inf must resolve: {source}"));
        assert_eq!(display_expr(&ctx, out), expected, "{source}");
    }
}

#[test]
fn one_to_infinity_power_declines_non_indeterminate_and_oscillating() {
    // Honest declines: a constant base (not -> 1), a base -> inf (x^x), a
    // FINITE exponent (1^5 = 1 is continuous, not indeterminate), and an
    // oscillating base-1 whose product has no limit.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for source in ["2^x", "x^x", "(1+1/x)^5", "(1+sin(x)/x)^x"] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            one_to_infinity_power_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos).is_none(),
            "1^inf must decline: {source}"
        );
    }
}

#[test]
fn product_log_unit_argument_resolves_inf_times_zero() {
    // inf * 0 with ln(f), f -> 1: lim g*ln(f) = lim g*(f-1) (ln(1+h) ~ h).
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for (source, expected) in [
        ("x*ln(1+1/x)", "1"),
        ("x*ln(1+2/x)", "2"),
        ("x*ln((x+1)/x)", "1"),
        ("x*ln(1-1/x)", "-1"),
        ("x^2*ln(1+1/x^2)", "1"),
        ("ln(1+1/x)*x", "1"),
        ("x*ln(1+1/x^2)", "0"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = product_log_unit_argument_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("inf*0 ln must resolve: {source}"));
        assert_eq!(display_expr(&ctx, out), expected, "{source}");
    }
}

#[test]
fn product_log_unit_argument_declines_non_unit_arg_and_finite_cofactor() {
    // Honest declines: ln argument -> inf (x ln x), a constant argument
    // (ln(2) x, arg = 2 != 1), a non-divergent cofactor (the continuous
    // finite*0 case), and an oscillating reduction with no limit.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    for source in [
        "x*ln(x)",
        "ln(2)*x",
        "(1/x)*ln(1+1/x)",
        "x*ln(1+sin(1/x)/x)",
    ] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            product_log_unit_argument_limit_at_infinity(&mut ctx, expr, x, InfSign::Pos).is_none(),
            "inf*0 ln must decline: {source}"
        );
    }
}

#[test]
fn finite_one_to_infinity_power_resolves_the_e_definition() {
    // 1^inf at a finite point: (1+x)^(1/x) = e, the textbook definition.
    // The product limit is computed by the full finite machinery, so
    // SECOND-ORDER cases (cos(x)^(1/x^2) = e^(-1/2)) resolve correctly.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    for (source, expected) in [
        ("(1+x)^(1/x)", "e"),
        ("(1+2*x)^(1/x)", "e^2"),
        ("(1+x)^(3/x)", "e^3"),
        ("(1-x)^(1/x)", "e^(-1)"),
        ("(1+sin(x))^(1/x)", "e"),
        ("cos(x)^(1/x^2)", "e^(-1/2)"),
    ] {
        let expr = parse_expr(&mut ctx, source);
        let out = apply_finite_one_to_infinity_power_rule(&mut ctx, expr, x, zero)
            .unwrap_or_else(|| panic!("finite 1^inf must resolve: {source}"));
        assert_eq!(display_expr(&ctx, out), expected, "{source}");
    }
}

#[test]
fn finite_zero_base_power_resolves_x_to_the_x() {
    // The 0^0 form x^g -> exp(lim g ln x) at 0+: x^x = exp(lim x ln x) = 1.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    for (source, expected) in [("x^x", "1"), ("x^(2*x)", "1"), ("x^(x^2)", "1")] {
        let expr = parse_expr(&mut ctx, source);
        let out =
            apply_finite_zero_base_power_rule(&mut ctx, expr, x, zero, FiniteLimitSide::Right)
                .unwrap_or_else(|| panic!("0^0 must resolve: {source}"));
        assert_eq!(display_expr(&ctx, out), expected, "{source}");
    }
}

#[test]
fn finite_zero_base_power_declines_left_side_and_non_variable_base() {
    // Honest declines: the LEFT side (x^x is complex for x < 0), a
    // non-variable base (sign unknown), and a nonzero point.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    let one = parse_expr(&mut ctx, "1");
    let xx = parse_expr(&mut ctx, "x^x");
    assert!(
        apply_finite_zero_base_power_rule(&mut ctx, xx, x, zero, FiniteLimitSide::Left).is_none(),
        "0^0 is real only from the right of 0"
    );
    let sinx = parse_expr(&mut ctx, "sin(x)^x");
    assert!(
        apply_finite_zero_base_power_rule(&mut ctx, sinx, x, zero, FiniteLimitSide::Right)
            .is_none(),
        "a non-variable base has unknown sign"
    );
    assert!(
        apply_finite_zero_base_power_rule(&mut ctx, xx, x, one, FiniteLimitSide::Right).is_none(),
        "0^0 is the form at 0, not at 1"
    );
}

#[test]
fn finite_one_to_infinity_power_declines_continuous_and_non_unit_base() {
    // Honest declines: a base not -> 1 (x^x, (2+x)^(1/x)), a zero product
    // limit (the continuous 1^finite case (1+x)^x and (1+x^2)^(1/x)), and
    // a constant exponent.
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let zero = parse_expr(&mut ctx, "0");
    let one = parse_expr(&mut ctx, "1");
    for source in ["x^x", "(2+x)^(1/x)", "(1+x)^x", "(1+x^2)^(1/x)", "(1+x)^2"] {
        let expr = parse_expr(&mut ctx, source);
        assert!(
            apply_finite_one_to_infinity_power_rule(&mut ctx, expr, x, zero).is_none(),
            "finite 1^inf must decline: {source}"
        );
    }
    // At x = 1 the base (1+x) -> 2, not 1: not the indeterminate form.
    let expr = parse_expr(&mut ctx, "(1+x)^(1/x)");
    assert!(
        apply_finite_one_to_infinity_power_rule(&mut ctx, expr, x, one).is_none(),
        "finite 1^inf needs a unit base limit"
    );
}

#[test]
fn try_limit_rules_at_infinity_uses_polynomial_growth_before_residual() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let expr = parse_expr(&mut ctx, "2*x^3 + x");

    let out = try_limit_rules_at_infinity(&mut ctx, expr, x, InfSign::Pos).expect("polynomial");

    assert!(matches!(ctx.get(out), Expr::Constant(Constant::Infinity)));
}

#[test]
fn bounded_elementary_over_divergent_limit_at_infinity_resolves_to_zero() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let cases = [
        "sin(x)/x",
        "cos(2*x + 1)/(x^2 + 1)",
        "(2*sin(x) - cos(x))/(-x)",
        "sin(x)*cos(x)/exp(x)",
        "arctan(x)/x",
        "atan(x^2 + 1)/(x^2 + 1)",
        "(arctan(x) + sin(x))/(0 - x)",
        "tanh(x)/x",
        "tanh(x^2 + 1)/(x^2 + 1)",
        "(tanh(x) - cos(x))/exp(x)",
        "sin(sqrt(x))/x",
        "sin(ln(x))/x",
    ];

    for expr in cases {
        let parsed = parse_expr(&mut ctx, expr);
        let out = try_limit_rules_at_infinity(&mut ctx, parsed, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("expected bounded-over-divergent zero for {expr}"));
        assert!(
            matches!(ctx.get(out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0))),
            "expected zero for {expr}, got {:?}",
            ctx.get(out)
        );
    }

    for expr in ["sin(sqrt(-x))/x", "sin(ln(-x))/x"] {
        let parsed = parse_expr(&mut ctx, expr);
        let out =
            try_limit_rules_at_infinity(&mut ctx, parsed, x, InfSign::Neg).unwrap_or_else(|| {
                panic!("expected negative-infinity bounded-over-divergent zero for {expr}")
            });
        assert!(
            matches!(ctx.get(out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0))),
            "expected zero for {expr}, got {:?}",
            ctx.get(out)
        );
    }
}

#[test]
fn bounded_elementary_over_divergent_limit_at_infinity_rejects_unbounded_or_nondominant_shapes() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let unbounded_num = parse_expr(&mut ctx, "x*sin(x)/x");
    let nondominant_den = parse_expr(&mut ctx, "sin(x)/cos(x)");
    let arctan_nondominant_den = parse_expr(&mut ctx, "arctan(x)/cos(x)");
    let tanh_nondominant_den = parse_expr(&mut ctx, "tanh(x)/cos(x)");
    let sqrt_domain_conflict = parse_expr(&mut ctx, "sin(sqrt(1 - x))/x");
    let log_domain_conflict = parse_expr(&mut ctx, "sin(ln(1 - x))/x");
    let neg_sqrt_domain_conflict = parse_expr(&mut ctx, "sin(sqrt(x))/x");
    let neg_log_domain_conflict = parse_expr(&mut ctx, "sin(ln(x))/x");

    assert!(bounded_elementary_over_divergent_limit_at_infinity(
        &mut ctx,
        unbounded_num,
        x,
        InfSign::Pos,
    )
    .is_none());
    assert!(bounded_elementary_over_divergent_limit_at_infinity(
        &mut ctx,
        nondominant_den,
        x,
        InfSign::Pos,
    )
    .is_none());
    assert!(bounded_elementary_over_divergent_limit_at_infinity(
        &mut ctx,
        arctan_nondominant_den,
        x,
        InfSign::Pos,
    )
    .is_none());
    assert!(bounded_elementary_over_divergent_limit_at_infinity(
        &mut ctx,
        tanh_nondominant_den,
        x,
        InfSign::Pos,
    )
    .is_none());
    assert!(bounded_elementary_over_divergent_limit_at_infinity(
        &mut ctx,
        sqrt_domain_conflict,
        x,
        InfSign::Pos,
    )
    .is_none());
    assert!(bounded_elementary_over_divergent_limit_at_infinity(
        &mut ctx,
        log_domain_conflict,
        x,
        InfSign::Pos,
    )
    .is_none());
    assert!(bounded_elementary_over_divergent_limit_at_infinity(
        &mut ctx,
        neg_sqrt_domain_conflict,
        x,
        InfSign::Neg,
    )
    .is_none());
    assert!(bounded_elementary_over_divergent_limit_at_infinity(
        &mut ctx,
        neg_log_domain_conflict,
        x,
        InfSign::Neg,
    )
    .is_none());
}

#[test]
fn bounded_elementary_times_decaying_exp_limit_at_infinity_resolves_to_zero() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let cases = [
        "sin(x)*exp(-x)",
        "exp(-2*x)*cos(x)",
        "sin(x)*exp(2 - x^4)",
        "exp(-x^2)*cos(x)",
        "(sin(x) + cos(x))*exp(-x)",
        "(sin(x) + cos(x))*exp(1 - x^2)",
        "arctan(x)*exp(-x)",
        "arctan(x)*exp(2 - x^4)",
        "-tanh(x)*exp(-x)",
        "-tanh(x)*exp(-x^2)",
        "sin(sqrt(x))*exp(-x)",
        "sin(ln(x))*exp(-x)",
    ];

    for expr in cases {
        let parsed = parse_expr(&mut ctx, expr);
        let out = try_limit_rules_at_infinity(&mut ctx, parsed, x, InfSign::Pos)
            .unwrap_or_else(|| panic!("expected bounded-times-decaying-exp zero for {expr}"));
        assert!(
            matches!(ctx.get(out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0))),
            "expected zero for {expr}, got {:?}",
            ctx.get(out)
        );
    }

    let tan_product = parse_expr(&mut ctx, "tan(x)*exp(-x)");
    let bad_sqrt_domain = parse_expr(&mut ctx, "sin(sqrt(1 - x))*exp(-x)");
    let bad_log_domain = parse_expr(&mut ctx, "sin(ln(1 - x))*exp(-x)");
    let bad_poly_exp_sqrt_domain = parse_expr(&mut ctx, "sin(sqrt(1 - x))*exp(2 - x^4)");
    let parametric_exp_tail = parse_expr(&mut ctx, "sin(x)*exp(a*x^2)");
    let nested_exp_tail = parse_expr(&mut ctx, "sin(x)*exp(exp(0 - x))");
    assert!(bounded_elementary_times_decaying_exp_limit_at_infinity(
        &mut ctx,
        tan_product,
        x,
        InfSign::Pos,
    )
    .is_none());
    assert!(bounded_elementary_times_decaying_exp_limit_at_infinity(
        &mut ctx,
        bad_sqrt_domain,
        x,
        InfSign::Pos,
    )
    .is_none());
    assert!(bounded_elementary_times_decaying_exp_limit_at_infinity(
        &mut ctx,
        bad_log_domain,
        x,
        InfSign::Pos,
    )
    .is_none());
    assert!(bounded_elementary_times_decaying_exp_limit_at_infinity(
        &mut ctx,
        bad_poly_exp_sqrt_domain,
        x,
        InfSign::Pos,
    )
    .is_none());
    assert!(bounded_elementary_times_decaying_exp_limit_at_infinity(
        &mut ctx,
        parametric_exp_tail,
        x,
        InfSign::Pos,
    )
    .is_none());
    assert!(bounded_elementary_times_decaying_exp_limit_at_infinity(
        &mut ctx,
        nested_exp_tail,
        x,
        InfSign::Pos,
    )
    .is_none());
}

#[test]
fn presimplify_safe_for_limit_applies_allowlisted_rewrites() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let expr = parse_expr(&mut ctx, "x + 0");
    let out = presimplify_safe_for_limit(&mut ctx, expr);
    assert_eq!(out, x);
}

#[test]
fn presimplify_safe_for_limit_does_not_apply_domain_sensitive_rewrites() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "x/x");
    let out = presimplify_safe_for_limit(&mut ctx, expr);
    assert!(matches!(ctx.get(out), Expr::Div(_, _)));
}

#[test]
fn reciprocal_substitution_resolves_notable_infinity_limits() {
    // `lim_{x→∞} g(x) = lim_{u→0⁺} g(1/u)`: the notable products `x·f(c/x)` the direct ∞ rules
    // miss. The artifact reducer turns the substituted `f(1/(1/x))·(1/x)` into `f(x)/x`.
    for (src, expected) in [
        ("x*sin(1/x)", "1"),
        ("x*sin(3/x)", "3"),
        ("x*tan(1/x)", "1"),
        ("x*arctan(1/x)", "1"),
        ("x*(exp(1/x)-1)", "1"),
        ("2*x*sin(1/x)", "2"),
    ] {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, src);
        let x = ctx.var("x");
        let result =
            try_limit_at_infinity_by_reciprocal_substitution(&mut ctx, expr, x, InfSign::Pos)
                .unwrap_or_else(|| panic!("must resolve: {src}"));
        assert_eq!(display_expr(&ctx, result), expected, "{src}");
    }
    // Genuinely limitless oscillators must decline — the substitution must not fabricate a value.
    for src in ["x*sin(x)", "sin(x)"] {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, src);
        let x = ctx.var("x");
        assert!(
            try_limit_at_infinity_by_reciprocal_substitution(&mut ctx, expr, x, InfSign::Pos)
                .is_none(),
            "{src} has no limit and must decline"
        );
    }
}

#[test]
fn abs_wrapped_unbounded_arguments_resolve_at_both_infinities() {
    let mut ctx = Context::new();
    let cases = [
        ("ln(|x|)", InfSign::Pos, "infinity"),
        ("ln(|x|)", InfSign::Neg, "infinity"),
        ("|x^2+1|", InfSign::Pos, "infinity"),
    ];
    for (source, approach, expected) in cases {
        let expr = cas_parser::parse(source, &mut ctx).expect(source);
        let var = ctx.var("x");
        let result = try_limit_rules_at_infinity(&mut ctx, expr, var, approach)
            .unwrap_or_else(|| panic!("must resolve: {source}"));
        assert_eq!(
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &ctx,
                    id: result
                }
            ),
            expected,
            "{source}"
        );
    }
}

#[test]
fn one_sided_composition_resolves_endpoint_combinations() {
    let mut ctx = Context::new();
    let cases = [
        ("x*ln(x)", "0"),
        ("x*ln(x) - x", "0"),
        ("2*sqrt(x)", "0"),
        ("sqrt(x)*ln(x)", "0"),
        ("3*ln(x)", "-infinity"),
        // Power atom: (x - 0)^q -> 0 from the right for rational q > 0.
        ("x^(3/2)", "0"),
        ("x^(1/3)", "0"),
        // Product of two variable factors (no constant cofactor).
        ("sqrt(x)*x", "0"),
        // Division by a foldable rational constant.
        ("x^(1/3 + 1) / (1/3 + 1)", "0"),
        ("ln(x)/2", "-infinity"),
    ];
    for (source, expected) in cases {
        let expr = cas_parser::parse(source, &mut ctx).expect(source);
        let var = ctx.var("x");
        let zero = ctx.num(0);
        let result =
            try_limit_rules_at_finite_one_sided(&mut ctx, expr, var, zero, FiniteLimitSide::Right)
                .unwrap_or_else(|| panic!("must resolve: {source}"));
        assert_eq!(
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &ctx,
                    id: result
                }
            ),
            expected,
            "{source}"
        );
    }
}

#[test]
fn one_sided_composition_refuses_indeterminate_forms() {
    let mut ctx = Context::new();
    // -infinity + infinity and the left side of the log domain.
    for source in ["ln(x) + 1/x", "ln(x) - ln(x)"] {
        let expr = cas_parser::parse(source, &mut ctx).expect(source);
        let var = ctx.var("x");
        let zero = ctx.num(0);
        assert!(
            try_limit_rules_at_finite_one_sided(&mut ctx, expr, var, zero, FiniteLimitSide::Right,)
                .is_none(),
            "must refuse: {source}"
        );
    }
}

#[test]
fn one_sided_product_combination_guards_indeterminate_signs() {
    let mut ctx = Context::new();
    let zero = ctx.num(0);
    let two = ctx.num(2);
    let infinity = ctx.add(Expr::Constant(Constant::Infinity));
    let neg_infinity = ctx.add(Expr::Neg(infinity));
    let pi = ctx.add(Expr::Constant(Constant::Pi));

    // 0 * infinity is indeterminate.
    assert!(combine_limit_product(&mut ctx, zero, infinity).is_none());
    // Symbolic finite cofactor: sign unknown, refuse.
    assert!(combine_limit_product(&mut ctx, pi, infinity).is_none());
    // Numeric nonzero cofactor decides the sign.
    let scaled = combine_limit_product(&mut ctx, two, neg_infinity).expect("signed");
    assert_eq!(
        format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: scaled
            }
        ),
        "-infinity"
    );
    // infinity * -infinity carries the product sign.
    let crossed = combine_limit_product(&mut ctx, infinity, neg_infinity).expect("signed");
    assert_eq!(
        format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: crossed
            }
        ),
        "-infinity"
    );
    // Finite * finite folds numerically.
    let folded = combine_limit_product(&mut ctx, two, two).expect("folded");
    assert_eq!(
        format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: folded
            }
        ),
        "4"
    );
}

#[test]
fn complex_domain_kill_switch_declines_every_approach() {
    // F0 (Fase 3): under a complex value domain no rule may run — the same
    // inputs that COMPUTE in the real domain must return the residual call
    // with the complex-domain warning, for every approach shape.
    let mut ctx = Context::new();
    let opts = LimitOptions {
        complex_enabled: true,
        ..LimitOptions::default()
    };
    let var = ctx.var("z");
    let zero = ctx.num(0);
    let cases = [
        ("e^(-1/z^2)", Approach::Finite(zero)),
        ("1/z^2", Approach::Finite(zero)),
        ("z^2", Approach::PosInfinity),
        (
            "1/z",
            Approach::FiniteOneSided(zero, FiniteLimitSide::Right),
        ),
    ];
    for (source, approach) in cases {
        let expr = cas_parser::parse(source, &mut ctx).expect(source);
        let outcome = eval_limit_at_infinity(&mut ctx, expr, var, approach, &opts);
        assert_eq!(
            outcome.warning.as_deref(),
            Some(COMPLEX_DOMAIN_LIMIT_UNSUPPORTED_WARNING),
            "must decline with the complex-domain warning: {source}"
        );
        let rendered = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: outcome.expr
            }
        );
        assert!(
            rendered.starts_with("limit("),
            "must stay a residual limit call: {source} -> {rendered}"
        );
    }
}

#[test]
fn real_domain_imaginary_point_declines_to_residual() {
    // F0 (Fase 3): a real-domain approach point containing the imaginary
    // unit has no real neighbourhood — substitution would fabricate a value
    // at e.g. the tanh pole iπ/2.
    let mut ctx = Context::new();
    let opts = LimitOptions::default();
    let var = ctx.var("z");
    for (source, point_src) in [("tanh(z)", "i*pi/2"), ("1/(z^2+1)", "i*1")] {
        let expr = cas_parser::parse(source, &mut ctx).expect(source);
        let point = cas_parser::parse(point_src, &mut ctx).expect(point_src);
        let outcome = eval_limit_at_infinity(&mut ctx, expr, var, Approach::Finite(point), &opts);
        assert_eq!(
            outcome.warning.as_deref(),
            Some(IMAGINARY_POINT_LIMIT_UNSUPPORTED_WARNING),
            "must decline with the imaginary-point warning: {source}"
        );
        let rendered = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: outcome.expr
            }
        );
        assert!(
            rendered.starts_with("limit("),
            "must stay a residual limit call: {source} -> {rendered}"
        );
    }
}

#[test]
fn taylor_removable_singularity_cancels_common_power() {
    // F1 (Fase 3): num(0)=0 y den(0)=0 → cancelar la potencia común
    // re-expandiendo a order+s (la cola truncada se vuelve término bajo).
    let mut ctx = Context::new();
    let render = |ctx: &mut Context, src: &str, order: usize| -> Option<String> {
        let expr = cas_parser::parse(src, ctx).expect(src);
        taylor_series_at_zero_expr(ctx, expr, "x", order)
            .map(|id| format!("{}", cas_formatter::DisplayExpr { context: ctx, id }))
    };
    // sin(x)/x = 1 − x²/6 + x⁴/120
    let s = render(&mut ctx, "sin(x)/x", 4).expect("sin(x)/x expands");
    for coeff in ["1/120", "1/6", "1"] {
        assert!(s.contains(coeff), "sin(x)/x: falta {coeff} en {s}");
    }
    // (1−cos x)/x² = 1/2 − x²/24 + x⁴/720
    let s = render(&mut ctx, "(1-cos(x))/x^2", 4).expect("(1-cos)/x^2 expands");
    for coeff in ["1/720", "1/24", "1/2"] {
        assert!(s.contains(coeff), "(1-cos)/x^2: falta {coeff} en {s}");
    }
    // Polo genuino: valuación del num < valuación del den → decline.
    assert_eq!(render(&mut ctx, "sin(x)/x^2", 3), None);
    assert_eq!(render(&mut ctx, "cos(x)/x", 3), None);
}

#[test]
fn taylor_definitional_declines_singular_coefficients() {
    // F1 (Fase 3): la definicional emitía series con coeficientes 0/0 ó
    // ln(0) que el simplificador colapsaba a `undefined` — la respuesta
    // honesta es declinar (eco residual del comando).
    let mut ctx = Context::new();
    let zero = ctx.num(0);
    for src in ["ln(x)", "1/x", "sin(x)/x^2"] {
        let expr = cas_parser::parse(src, &mut ctx).expect(src);
        assert!(
            taylor_series_at_point_expr(&mut ctx, expr, "x", zero, 2).is_none(),
            "debe declinar en el punto singular: {src}"
        );
    }
    // Pin: coeficientes SIMBÓLICOS (paramétricos) siguen pasando.
    let expr = cas_parser::parse("e^(x+y)", &mut ctx).expect("param");
    assert!(taylor_series_at_point_expr(&mut ctx, expr, "x", zero, 2).is_some());
}

#[test]
fn taylor_multivar_total_degree_expansion() {
    // F2 (Fase 3): grado TOTAL — e^(x+y) a orden 2 NO lleva x²y².
    let mut ctx = Context::new();
    let render = |ctx: &mut Context, src: &str, order: usize| -> Option<String> {
        let expr = cas_parser::parse(src, ctx).expect(src);
        let zero = ctx.num(0);
        let points = vec![zero; 2];
        taylor_multivar_series_expr(ctx, expr, &["x".into(), "y".into()], &points, order)
            .map(|id| format!("{}", cas_formatter::DisplayExpr { context: ctx, id }))
    };
    // x²+y² es su propio desarrollo de orden 2.
    let s = render(&mut ctx, "x^2+y^2", 2).expect("expands");
    assert!(s.contains("x^2") && s.contains("y^2"), "{s}");
    assert!(
        !s.contains("x^2·y^2") && !s.contains("y^2·x^2"),
        "sin término de grado 4: {s}"
    );
    // sin(x·y) a orden 2 expande (la forma FINAL `x·y` la pinea el e2e:
    // el ensamblador crudo lleva litter `cos(0)` que pliega el simplify).
    let s = render(&mut ctx, "sin(x*y)", 2).expect("expands");
    assert!(s.contains("x") && s.contains("y"), "{s}");
    // Punto singular → decline all-or-nothing.
    assert_eq!(render(&mut ctx, "ln(x*y)", 2), None);
    // Cap de términos: C(33+2,2) > 64 → decline (orden alto en 2 vars).
    assert_eq!(render(&mut ctx, "e^(x+y)", 20), None);
}

#[test]
fn multivar_squeeze_zero_exact_family_only() {
    // F8b: P/(x²+y²)^k con min-grado(P) > 2k → 0 probado; el borde
    // min-grado == 2k DECLINA (complementario del driver de caminos).
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let y = ctx.var("y");
    let zero = ctx.num(0);
    let vars = [x, y];
    let points = [zero, zero];
    let ok = |ctx: &mut Context, src: &str| -> bool {
        let e = cas_parser::parse(src, ctx).expect(src);
        try_multivar_squeeze_zero(ctx, e, &vars, &points).is_some()
    };
    assert!(ok(&mut ctx, "x^2*y/(x^2+y^2)"));
    assert!(ok(&mut ctx, "x^3/(x^2+y^2)"));
    assert!(ok(&mut ctx, "x^3*y^2/(x^2+y^2)^2"));
    // Bordes que DECLINAN: grado igual (x·y), menor (xy² sobre ^2), den
    // que no es potencia exacta de r², punto no-origen.
    assert!(!ok(&mut ctx, "x*y/(x^2+y^2)"));
    assert!(!ok(&mut ctx, "x*y^2/(x^2+y^2)^2"));
    assert!(!ok(&mut ctx, "x^3/(x^2+2*y^2)"));
    let one = ctx.num(1);
    let pts = [one, zero];
    let e = cas_parser::parse("x^2*y/(x^2+y^2)", &mut ctx).expect("e");
    assert!(try_multivar_squeeze_zero(&mut ctx, e, &vars, &pts).is_none());
}

#[test]
fn multivar_dne_by_paths_decides_only_from_proven_facts() {
    // F8 (Fase 3): dos caminos con racionales exactos DISTINTOS → DNE con
    // ambos testigos; caminos que COINCIDEN → None (jamás existencia desde
    // finitos caminos — el pin de soundness central).
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let y = ctx.var("y");
    let zero = ctx.num(0);
    let vars = [x, y];
    let points = [zero, zero];
    // El clásico: y=0 → 0, y=x → 1/2.
    let expr = cas_parser::parse("x*y/(x^2+y^2)", &mut ctx).expect("expr");
    let verdict = try_multivar_dne_by_paths(&mut ctx, expr, &vars, &points).expect("DNE probado");
    let b = verdict.witness_b.expect("dos testigos");
    assert_eq!(verdict.witness_a.value_display, "0");
    assert_eq!(b.value_display, "1/2");
    assert_eq!(b.path_display, "y = x");
    // Todos los caminos dan 0 → None (residual honesto).
    let expr = cas_parser::parse("x^2*y/(x^2+y^2)", &mut ctx).expect("expr");
    assert!(try_multivar_dne_by_paths(&mut ctx, expr, &vars, &points).is_none());
    // Punto no racional → None (la batería exige aritmética exacta).
    let e_pt = ctx.add(Expr::Constant(Constant::E));
    let points_e = [e_pt, zero];
    let expr = cas_parser::parse("x*y/(x^2+y^2)", &mut ctx).expect("expr");
    assert!(try_multivar_dne_by_paths(&mut ctx, expr, &vars, &points_e).is_none());
}

#[test]
fn matrix_limit_is_componentwise_all_or_nothing() {
    // P0 2026-07-19: `depends_on` skipped Matrix entries, so the constant
    // rule asserted `[[1/x,0],[0,1]]` as its OWN limit (x-dependent "value").
    let mut ctx = Context::new();
    let opts = LimitOptions::default();
    let var = ctx.var("x");
    let render =
        |ctx: &Context, id: ExprId| format!("{}", cas_formatter::DisplayExpr { context: ctx, id });
    // Every entry resolves → matrix of entry limits.
    let m = cas_parser::parse("[[1/x,0],[0,1]]", &mut ctx).expect("matrix");
    let outcome = eval_limit_at_infinity(&mut ctx, m, var, Approach::PosInfinity, &opts);
    assert!(outcome.warning.is_none());
    assert_eq!(render(&ctx, outcome.expr), "[[0, 0], [0, 1]]");
    // An entry with a PROVEN-DNE limit (disagreeing laterals) decides the whole.
    let m = cas_parser::parse("[[1/x,2]]", &mut ctx).expect("matrix");
    let zero = ctx.num(0);
    let outcome = eval_limit_at_infinity(&mut ctx, m, var, Approach::Finite(zero), &opts);
    assert_eq!(render(&ctx, outcome.expr), "undefined");
    assert!(
        outcome
            .warning
            .as_deref()
            .is_some_and(|w| w.starts_with("the matrix limit does not exist")),
        "DNE entry must decide the matrix: {:?}",
        outcome.warning
    );
    // A declining entry keeps the WHOLE matrix an honest residual.
    let m = cas_parser::parse("[[e^(i*x),1]]", &mut ctx).expect("matrix");
    let outcome = eval_limit_at_infinity(&mut ctx, m, var, Approach::PosInfinity, &opts);
    assert!(
        render(&ctx, outcome.expr).starts_with("limit("),
        "must stay residual"
    );
    assert!(outcome
        .warning
        .as_deref()
        .is_some_and(|w| w.starts_with("matrix entry declines")));
}

#[test]
fn real_domain_real_point_is_untouched_by_the_domain_guard() {
    // Pin: the guard must be invisible in the real domain — the same case
    // that computes today keeps computing (no warning, no residual).
    let mut ctx = Context::new();
    let opts = LimitOptions::default();
    let var = ctx.var("x");
    let zero = ctx.num(0);
    let expr = cas_parser::parse("1/x^2", &mut ctx).expect("1/x^2");
    let outcome = eval_limit_at_infinity(&mut ctx, expr, var, Approach::Finite(zero), &opts);
    assert!(outcome.warning.is_none(), "real domain must stay resolved");
    let rendered = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: outcome.expr
        }
    );
    assert_eq!(rendered, "infinity");
}
