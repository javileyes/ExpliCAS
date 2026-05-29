use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::multipoly::{multipoly_from_expr, multipoly_to_expr, PolyBudget};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

pub(super) fn rational_const_for_calculus_presentation(
    ctx: &mut Context,
    value: BigRational,
) -> ExprId {
    if value == BigRational::one() {
        ctx.num(1)
    } else {
        ctx.add(Expr::Number(value))
    }
}

pub(super) fn scale_expr_for_calculus_presentation(
    ctx: &mut Context,
    coeff: BigRational,
    expr: ExprId,
) -> ExprId {
    if coeff.is_one() {
        return expr;
    }
    let coeff = rational_const_for_calculus_presentation(ctx, coeff);
    if let Some(value) = cas_ast::views::as_rational_const(ctx, expr, 8) {
        if value == BigRational::one() {
            return coeff;
        }
        if let Some(coeff_value) = cas_ast::views::as_rational_const(ctx, coeff, 8) {
            return rational_const_for_calculus_presentation(ctx, coeff_value * value);
        }
    }
    cas_math::expr_nary::build_balanced_mul(ctx, &[coeff, expr])
}

pub(super) fn nonzero_rational_parts(value: &BigRational) -> Option<(BigRational, BigRational)> {
    if value.is_zero() {
        return None;
    }

    let numerator = BigRational::from_integer(value.numer().clone());
    let denominator = BigRational::from_integer(value.denom().clone());
    Some((numerator, denominator))
}

pub(super) fn add_one_for_calculus_presentation(ctx: &mut Context, expr: ExprId) -> ExprId {
    add_rational_for_calculus_presentation(ctx, expr, BigRational::one())
}

pub(super) fn subtract_from_one_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> ExprId {
    let one = ctx.num(1);
    let raw = ctx.add(Expr::Sub(one, expr));
    let budget = PolyBudget {
        max_terms: 8,
        max_total_degree: 4,
        max_pow_exp: 4,
    };

    multipoly_from_expr(ctx, raw, &budget)
        .map(|poly| multipoly_to_expr(&poly, ctx))
        .unwrap_or(raw)
}

pub(super) fn subtract_from_rational_for_calculus_presentation(
    ctx: &mut Context,
    value: BigRational,
    expr: ExprId,
) -> ExprId {
    let constant = rational_const_for_calculus_presentation(ctx, value);
    let raw = ctx.add(Expr::Sub(constant, expr));
    let budget = PolyBudget {
        max_terms: 8,
        max_total_degree: 4,
        max_pow_exp: 4,
    };

    multipoly_from_expr(ctx, raw, &budget)
        .map(|poly| multipoly_to_expr(&poly, ctx))
        .unwrap_or(raw)
}

pub(super) fn subtract_expr_for_calculus_presentation(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> ExprId {
    let raw = ctx.add(Expr::Sub(left, right));
    let budget = PolyBudget {
        max_terms: 8,
        max_total_degree: 4,
        max_pow_exp: 4,
    };

    multipoly_from_expr(ctx, raw, &budget)
        .map(|poly| multipoly_to_expr(&poly, ctx))
        .unwrap_or(raw)
}

pub(super) fn add_rational_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    value: BigRational,
) -> ExprId {
    if value.is_zero() {
        return expr;
    }

    let constant = rational_const_for_calculus_presentation(ctx, value);
    let raw = ctx.add(Expr::Add(expr, constant));
    let budget = PolyBudget {
        max_terms: 8,
        max_total_degree: 4,
        max_pow_exp: 4,
    };

    multipoly_from_expr(ctx, raw, &budget)
        .map(|poly| multipoly_to_expr(&poly, ctx))
        .unwrap_or(raw)
}

pub(super) fn add_rational_combining_additive_constant_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    value: BigRational,
) -> ExprId {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
    if terms.len() < 2 {
        return add_rational_for_calculus_presentation(ctx, expr, value);
    }

    let mut constant = value;
    let mut saw_constant = false;
    let mut rebuilt_terms = Vec::new();
    for (term, sign) in terms {
        if let Some(term_value) = cas_ast::views::as_rational_const(ctx, term, 8) {
            saw_constant = true;
            if sign == cas_math::expr_nary::Sign::Neg {
                constant -= term_value;
            } else {
                constant += term_value;
            }
            continue;
        }

        if sign == cas_math::expr_nary::Sign::Neg {
            rebuilt_terms.push(ctx.add(Expr::Neg(term)));
        } else {
            rebuilt_terms.push(term);
        }
    }

    if !saw_constant {
        return add_rational_for_calculus_presentation(ctx, expr, constant);
    }
    if !constant.is_zero() {
        rebuilt_terms.push(rational_const_for_calculus_presentation(ctx, constant));
    }

    match rebuilt_terms.len() {
        0 => ctx.num(0),
        1 => rebuilt_terms[0],
        _ => cas_math::expr_nary::build_balanced_add(ctx, &rebuilt_terms),
    }
}

pub(super) fn reciprocal_integer_radicand_content_for_calculus_presentation(
    value: &BigRational,
) -> Option<BigRational> {
    if value.is_positive() && value.numer().is_one() && value.denom() > &BigInt::one() {
        Some(BigRational::from_integer(value.denom().clone()))
    } else {
        None
    }
}

pub(super) fn positive_constant_over_expr_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    match ctx.get(expr) {
        Expr::Div(num, den) => {
            let value = cas_ast::views::as_rational_const(ctx, *num, 8)?;
            value.is_positive().then_some((value, *den))
        }
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, *exp, 8)
                == Some(BigRational::new((-1).into(), 1.into())) =>
        {
            Some((BigRational::one(), *base))
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut denominator = None;
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                    scale *= value;
                    continue;
                }
                let Expr::Pow(base, exp) = ctx.get(factor) else {
                    return None;
                };
                if cas_ast::views::as_rational_const(ctx, *exp, 8)
                    != Some(BigRational::new((-1).into(), 1.into()))
                    || denominator.replace(*base).is_some()
                {
                    return None;
                }
            }
            scale.is_positive().then_some((scale, denominator?))
        }
        _ => None,
    }
}

pub(super) fn positive_rational_sqrt_denominator_factor_for_calculus_presentation(
    ctx: &mut Context,
    value: &BigRational,
) -> Option<(BigRational, Option<ExprId>)> {
    if let Some(root) = exact_positive_rational_sqrt_for_calculus_presentation(value) {
        return Some((root, None));
    }

    let radicand = rational_const_for_calculus_presentation(ctx, value.clone());
    let sqrt_factor = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    Some((value.clone(), Some(sqrt_factor)))
}

pub(super) fn signed_numerator_for_calculus_presentation(
    ctx: &mut Context,
    coeff: BigRational,
    expr: ExprId,
) -> ExprId {
    if coeff == -BigRational::one()
        && !cas_ast::views::as_rational_const(ctx, expr, 8).is_some_and(|value| value.is_one())
    {
        return ctx.add(Expr::Neg(expr));
    }
    scale_expr_for_calculus_presentation(ctx, coeff, expr)
}

pub(super) fn signed_rational_const_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<BigRational> {
    if let Some(value) = cas_ast::views::as_rational_const(ctx, expr, 8) {
        return Some(value);
    }
    if let Expr::Neg(inner) = ctx.get(expr) {
        return cas_ast::views::as_rational_const(ctx, *inner, 8).map(|value| -value);
    }
    None
}

pub(super) fn exact_positive_rational_sqrt_for_calculus_presentation(
    value: &BigRational,
) -> Option<BigRational> {
    if !value.is_positive() {
        return None;
    }

    let numer_sqrt = value.numer().sqrt();
    let denom_sqrt = value.denom().sqrt();
    if &numer_sqrt * &numer_sqrt == *value.numer() && &denom_sqrt * &denom_sqrt == *value.denom() {
        Some(BigRational::new(numer_sqrt, denom_sqrt))
    } else {
        None
    }
}

fn split_square_factor_positive_bigint_for_calculus_presentation(
    value: &BigInt,
) -> (BigInt, BigInt) {
    let mut outside = BigInt::one();
    let mut inside = value.clone();

    for factor in 2_i64..=97 {
        let factor = BigInt::from(factor);
        let square = &factor * &factor;
        while (&inside % &square).is_zero() {
            inside /= &square;
            outside *= &factor;
        }
    }

    let root = inside.sqrt();
    if root > BigInt::one() && &root * &root == inside {
        outside *= root;
        inside = BigInt::one();
    }

    (outside, inside)
}

pub(super) fn split_square_factor_positive_rational_for_calculus_presentation(
    value: &BigRational,
) -> (BigRational, BigRational) {
    debug_assert!(value.is_positive());

    let (numer_outside, numer_inside) =
        split_square_factor_positive_bigint_for_calculus_presentation(value.numer());
    let (denom_outside, denom_inside) =
        split_square_factor_positive_bigint_for_calculus_presentation(value.denom());

    (
        BigRational::new(numer_outside, denom_outside),
        BigRational::new(numer_inside, denom_inside),
    )
}

pub(super) fn sqrt_positive_rational_expr_for_calculus_presentation(
    ctx: &mut Context,
    value: BigRational,
) -> ExprId {
    if let Some(sqrt_value) = exact_positive_rational_sqrt_for_calculus_presentation(&value) {
        return rational_const_for_calculus_presentation(ctx, sqrt_value);
    }

    let (outside, inside) = split_square_factor_positive_rational_for_calculus_presentation(&value);
    if inside.is_one() {
        return rational_const_for_calculus_presentation(ctx, outside);
    }

    let radicand = rational_const_for_calculus_presentation(ctx, inside);
    let sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    scale_expr_for_calculus_presentation(ctx, outside, sqrt)
}

pub(super) fn scale_expr_by_sqrt_positive_rational_for_calculus_presentation(
    ctx: &mut Context,
    value: BigRational,
    expr: ExprId,
) -> ExprId {
    if value.is_one() {
        return expr;
    }

    let sqrt_scale = sqrt_positive_rational_expr_for_calculus_presentation(ctx, value);
    if let Some(expr_value) = cas_ast::views::as_rational_const(ctx, expr, 8) {
        if expr_value.is_one() {
            return sqrt_scale;
        }
        if expr_value == -BigRational::one() {
            return ctx.add(Expr::Neg(sqrt_scale));
        }
        return scale_expr_for_calculus_presentation(ctx, expr_value, sqrt_scale);
    }
    cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_scale, expr])
}
