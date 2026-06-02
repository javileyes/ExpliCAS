use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

use super::scalar_presentation::{
    rational_const_for_calculus_presentation, scale_expr_for_calculus_presentation,
};

pub(super) fn reciprocal_integer_radicand_content_for_calculus_presentation(
    value: &BigRational,
) -> Option<BigRational> {
    if value.is_positive() && value.numer().is_one() && value.denom() > &BigInt::one() {
        Some(BigRational::from_integer(value.denom().clone()))
    } else {
        None
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
