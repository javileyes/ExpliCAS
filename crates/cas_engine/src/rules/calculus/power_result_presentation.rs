use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation, strictly_positive_quadratic_on_reals,
};
use super::presentation_utils::{
    is_calculus_presentation_one, negative_half_power_base_for_calculus_presentation,
    unwrap_internal_hold_for_calculus,
};
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    scale_expr_for_calculus_presentation, signed_numerator_for_calculus_presentation,
};
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::multipoly::{multipoly_from_expr, multipoly_to_expr, PolyBudget};
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

fn half_power_term_for_integration_presentation(
    ctx: &Context,
    term: ExprId,
    sign: cas_math::expr_nary::Sign,
    var_name: &str,
) -> Option<(ExprId, BigRational, u32)> {
    let mut coefficient = BigRational::one();
    let mut base_and_offset = None;

    for factor in cas_math::expr_nary::MulView::from_expr(ctx, term).factors {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            coefficient *= value;
            continue;
        }

        let (base, power) = match ctx.get(factor) {
            Expr::Function(fn_id, args)
                if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) =>
            {
                (args[0], BigRational::new(1.into(), 2.into()))
            }
            Expr::Pow(base, exp) => {
                let power = cas_ast::views::as_rational_const(ctx, *exp, 8)?;
                (*base, power)
            }
            _ => return None,
        };

        polynomial_radicand_for_calculus_presentation(ctx, base, var_name)?;
        let offset = power - BigRational::new(1.into(), 2.into());
        if !offset.is_integer() || offset.is_negative() {
            return None;
        }
        let offset = offset.to_integer().to_u32()?;
        if offset > 4 || base_and_offset.replace((base, offset)).is_some() {
            return None;
        }
    }

    if sign == cas_math::expr_nary::Sign::Neg {
        coefficient = -coefficient;
    }

    let (base, offset) = base_and_offset?;
    (!coefficient.is_zero()).then_some((base, coefficient, offset))
}

pub(super) fn compact_half_power_sum_root_product_for_integration_presentation(
    ctx: &mut Context,
    result: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let terms = cas_math::expr_nary::AddView::from_expr(ctx, result).terms;
    if terms.len() < 2 || terms.len() > 4 {
        return None;
    }

    let mut common_base = None;
    let mut polynomial_terms = Vec::with_capacity(terms.len());
    for (term, sign) in terms {
        let (base, coefficient, offset) =
            half_power_term_for_integration_presentation(ctx, term, sign, var_name)?;
        if let Some(existing) = common_base {
            if compare_expr(ctx, existing, base) != std::cmp::Ordering::Equal {
                return None;
            }
        } else {
            common_base = Some(base);
        }

        let power_free = match offset {
            0 => ctx.num(1),
            1 => base,
            _ => {
                let exp = ctx.num(offset as i64);
                ctx.add(Expr::Pow(base, exp))
            }
        };
        polynomial_terms.push(scale_expr_for_calculus_presentation(
            ctx,
            coefficient,
            power_free,
        ));
    }

    let base = common_base?;
    let raw_polynomial = cas_math::expr_nary::build_balanced_add(ctx, &polynomial_terms);
    let budget = PolyBudget {
        max_terms: 8,
        max_total_degree: 4,
        max_pow_exp: 4,
    };
    let polynomial = multipoly_from_expr(ctx, raw_polynomial, &budget)
        .map(|poly| multipoly_to_expr(&poly, ctx))
        .unwrap_or(raw_polynomial);
    let sqrt_base = ctx.call_builtin(BuiltinFn::Sqrt, vec![base]);
    Some(cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[sqrt_base, polynomial],
    ))
}

pub(super) fn compact_negative_half_power_result_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Pow(base, exp) = ctx.get(expr).clone() {
        let exponent = cas_ast::views::as_rational_const(ctx, exp, 8)?;
        if exponent == BigRational::new((-1).into(), 2.into()) {
            let one = ctx.num(1);
            let sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![base]);
            return Some(ctx.add(Expr::Div(one, sqrt)));
        }
    }

    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let compact = compact_negative_half_power_result_for_integration_presentation(ctx, inner)?;
        if let Expr::Div(num, den) = ctx.get(compact).clone() {
            let numerator = ctx.add(Expr::Neg(num));
            return Some(ctx.add(Expr::Div(numerator, den)));
        }
        return Some(ctx.add(Expr::Neg(compact)));
    }

    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };
    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    let mut coefficient = BigRational::one();
    let mut base = None;

    for factor in factors {
        match ctx.get(factor) {
            Expr::Pow(pow_base, exp) => {
                let exponent = cas_ast::views::as_rational_const(ctx, *exp, 8)?;
                if exponent != BigRational::new((-1).into(), 2.into()) || base.is_some() {
                    return None;
                }
                base = Some(*pow_base);
            }
            _ => {
                coefficient *= cas_ast::views::as_rational_const(ctx, factor, 8)?;
            }
        }
    }

    let base = base?;
    if coefficient.is_zero() {
        return Some(ctx.num(0));
    }
    let numerator = ctx.add(Expr::Number(coefficient));
    let sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![base]);
    Some(ctx.add(Expr::Div(numerator, sqrt)))
}

pub(super) fn compact_negative_half_power_product_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let compact = compact_negative_half_power_product_for_calculus_presentation(ctx, inner)?;
        return Some(ctx.add(Expr::Neg(compact)));
    }

    let mut numerator_factors = Vec::new();
    let mut denominator_factors = Vec::new();
    collect_fraction_product_factors_for_calculus_presentation(
        ctx,
        expr,
        &mut numerator_factors,
        &mut denominator_factors,
    );

    let mut base = None;
    let mut retained_numerator = Vec::new();
    for factor in numerator_factors {
        if base.is_none() {
            if let Some(pow_base) = negative_half_power_base_for_calculus_presentation(ctx, factor)
            {
                base = Some(pow_base);
                continue;
            }
        }
        retained_numerator.push(factor);
    }

    let base = base?;
    denominator_factors.push(ctx.call_builtin(BuiltinFn::Sqrt, vec![base]));

    let one = ctx.num(1);
    let numerator = if retained_numerator.is_empty() {
        one
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &retained_numerator)
    };
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
    Some(ctx.add_raw(Expr::Div(numerator, denominator)))
}

fn collect_fraction_product_factors_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    numerator_factors: &mut Vec<ExprId>,
    denominator_factors: &mut Vec<ExprId>,
) {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Mul(left, right) => {
            collect_fraction_product_factors_for_calculus_presentation(
                ctx,
                left,
                numerator_factors,
                denominator_factors,
            );
            collect_fraction_product_factors_for_calculus_presentation(
                ctx,
                right,
                numerator_factors,
                denominator_factors,
            );
        }
        Expr::Div(num, den) => {
            collect_fraction_product_factors_for_calculus_presentation(
                ctx,
                num,
                numerator_factors,
                denominator_factors,
            );
            collect_fraction_denominator_factors_for_calculus_presentation(
                ctx,
                den,
                denominator_factors,
            );
        }
        _ if split_rational_factor_for_calculus_presentation(
            ctx,
            expr,
            numerator_factors,
            denominator_factors,
        ) => {}
        _ if !is_calculus_presentation_one(ctx, expr) => numerator_factors.push(expr),
        _ => {}
    }
}

fn collect_fraction_denominator_factors_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    denominator_factors: &mut Vec<ExprId>,
) {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Mul(left, right) => {
            collect_fraction_denominator_factors_for_calculus_presentation(
                ctx,
                left,
                denominator_factors,
            );
            collect_fraction_denominator_factors_for_calculus_presentation(
                ctx,
                right,
                denominator_factors,
            );
        }
        _ if split_rational_factor_for_calculus_presentation(
            ctx,
            expr,
            denominator_factors,
            &mut Vec::new(),
        ) => {}
        _ if !is_calculus_presentation_one(ctx, expr) => denominator_factors.push(expr),
        _ => {}
    }
}

fn split_rational_factor_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    numerator_factors: &mut Vec<ExprId>,
    denominator_factors: &mut Vec<ExprId>,
) -> bool {
    let Some(value) = cas_ast::views::as_rational_const(ctx, expr, 8) else {
        return false;
    };
    if value.is_zero() {
        numerator_factors.push(expr);
        return true;
    }

    let numerator = BigRational::from_integer(value.numer().clone());
    let denominator = BigRational::from_integer(value.denom().clone());
    if !numerator.is_one() {
        numerator_factors.push(ctx.add(Expr::Number(numerator)));
    }
    if !denominator.is_one() {
        denominator_factors.push(ctx.add(Expr::Number(denominator)));
    }
    true
}

pub(super) fn compact_negative_three_half_power_result_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    allow_conditional_positive_quadratic: bool,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Pow(base, exp) = ctx.get(expr).clone() {
        let exponent = cas_ast::views::as_rational_const(ctx, exp, 8)?;
        if matches_supported_negative_odd_half_power(&exponent) {
            if !quadratic_for_fractional_power_calculus_presentation(
                ctx,
                base,
                var_name,
                allow_conditional_positive_quadratic,
            ) {
                return None;
            }
            let one = ctx.num(1);
            let denominator =
                negative_odd_half_power_denominator_for_presentation(ctx, base, -exponent)?;
            return Some(ctx.add(Expr::Div(one, denominator)));
        }
    }

    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            ctx,
            inner,
            var_name,
            allow_conditional_positive_quadratic,
        )?;
        if let Expr::Div(num, den) = ctx.get(compact).clone() {
            let numerator = ctx.add(Expr::Neg(num));
            return Some(ctx.add(Expr::Div(numerator, den)));
        }
        return Some(ctx.add(Expr::Neg(compact)));
    }

    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        let numerator_value = cas_ast::views::as_rational_const(ctx, num, 8)?;
        let mut denominator_scale = BigRational::one();
        let mut base = None;
        for factor in cas_math::expr_nary::mul_leaves(ctx, den) {
            if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                denominator_scale *= value;
                continue;
            }
            let Expr::Pow(pow_base, exp) = ctx.get(factor).clone() else {
                return None;
            };
            let exponent = cas_ast::views::as_rational_const(ctx, exp, 8)?;
            if !matches_supported_positive_odd_half_power(&exponent) || base.is_some() {
                return None;
            }
            base = Some((pow_base, exponent));
        }
        if denominator_scale.is_zero() {
            return None;
        }
        let (base, denominator_power) = base?;
        if !quadratic_for_fractional_power_calculus_presentation(
            ctx,
            base,
            var_name,
            allow_conditional_positive_quadratic,
        ) {
            return None;
        }
        let coefficient = numerator_value / denominator_scale;
        let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
        let one = ctx.num(1);
        let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, one);
        let mut denominator_parts = Vec::new();
        if !denominator_coeff.is_one() {
            denominator_parts.push(rational_const_for_calculus_presentation(
                ctx,
                denominator_coeff,
            ));
        }
        denominator_parts.push(negative_odd_half_power_denominator_for_presentation(
            ctx,
            base,
            denominator_power,
        )?);
        let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_parts);
        return Some(ctx.add(Expr::Div(numerator, denominator)));
    }

    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };
    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    let mut coefficient = BigRational::one();
    let mut base = None;

    for factor in factors {
        match ctx.get(factor) {
            Expr::Pow(pow_base, exp) => {
                let exponent = cas_ast::views::as_rational_const(ctx, *exp, 8)?;
                if !matches_supported_negative_odd_half_power(&exponent) || base.is_some() {
                    return None;
                }
                base = Some((*pow_base, -exponent));
            }
            _ => {
                coefficient *= cas_ast::views::as_rational_const(ctx, factor, 8)?;
            }
        }
    }

    let (base, denominator_power) = base?;
    if !quadratic_for_fractional_power_calculus_presentation(
        ctx,
        base,
        var_name,
        allow_conditional_positive_quadratic,
    ) {
        return None;
    }
    if coefficient.is_zero() {
        return Some(ctx.num(0));
    }
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let one = ctx.num(1);
    let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, one);
    let mut denominator_parts = Vec::new();
    if !denominator_coeff.is_one() {
        denominator_parts.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_coeff,
        ));
    }
    denominator_parts.push(negative_odd_half_power_denominator_for_presentation(
        ctx,
        base,
        denominator_power,
    )?);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_parts);
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn matches_supported_negative_odd_half_power(exponent: &BigRational) -> bool {
    exponent.is_negative() && odd_half_denominator_base_power(&(-exponent.clone())).is_some()
}

fn matches_supported_positive_odd_half_power(exponent: &BigRational) -> bool {
    odd_half_denominator_base_power(exponent).is_some()
}

fn quadratic_for_fractional_power_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    allow_conditional_positive_quadratic: bool,
) -> bool {
    if strictly_positive_quadratic_for_calculus_presentation(ctx, expr, var_name) {
        return true;
    }
    if !allow_conditional_positive_quadratic {
        return false;
    }
    let Some(poly) = polynomial_radicand_for_calculus_presentation(ctx, expr, var_name) else {
        return false;
    };
    if poly.degree() != 2 || poly.coeffs.len() < 3 {
        return false;
    }
    let leading = poly
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    leading.is_positive()
}

fn negative_odd_half_power_denominator_for_presentation(
    ctx: &mut Context,
    base: ExprId,
    denominator_power: BigRational,
) -> Option<ExprId> {
    let sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![base]);
    let base_power = odd_half_denominator_base_power(&denominator_power)?;
    let base_factor = if base_power == 1 {
        base
    } else {
        let exponent = ctx.num(base_power);
        ctx.add(Expr::Pow(base, exponent))
    };
    Some(cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[base_factor, sqrt],
    ))
}

fn odd_half_denominator_base_power(denominator_power: &BigRational) -> Option<i64> {
    if denominator_power.denom() != &BigInt::from(2) {
        return None;
    }
    let numerator = denominator_power.numer();
    if !numerator.is_positive() || !numerator.is_odd() {
        return None;
    }
    let numerator = numerator.to_i64()?;
    if numerator < 3 {
        return None;
    }
    Some((numerator - 1) / 2)
}

fn strictly_positive_quadratic_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let Some(poly) = polynomial_radicand_for_calculus_presentation(ctx, expr, var_name) else {
        return false;
    };
    strictly_positive_quadratic_on_reals(&poly)
}

pub(super) fn compact_positive_quadratic_square_derivative_result(
    ctx: &mut Context,
    result: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let result = unwrap_internal_hold_for_calculus(ctx, result);
    let Expr::Div(numerator, denominator) = ctx.get(result).clone() else {
        return None;
    };
    let numerator_value = cas_ast::views::as_rational_const(ctx, numerator, 8)?;
    if numerator_value.is_zero() {
        return None;
    }

    let Expr::Pow(base, exp) = ctx.get(denominator).clone() else {
        return None;
    };
    let two = BigRational::from_integer(2.into());
    if cas_ast::views::as_rational_const(ctx, exp, 8).as_ref() != Some(&two) {
        return None;
    }

    let base_poly = polynomial_radicand_for_calculus_presentation(ctx, base, var_name)?;
    if !strictly_positive_quadratic_on_reals(&base_poly) {
        return None;
    }

    let (base_core, base_content) = split_polynomial_content_for_calculus_presentation(ctx, base);
    let compact_numerator_value = numerator_value / (&base_content * &base_content);
    let numerator = rational_const_for_calculus_presentation(ctx, compact_numerator_value);
    let two_expr = ctx.num(2);
    let denominator = ctx.add(Expr::Pow(base_core, two_expr));
    let compact = ctx.add(Expr::Div(numerator, denominator));

    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

pub(super) fn compact_positive_half_power_result_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Pow(base, exp) = ctx.get(expr).clone() {
        let exponent = cas_ast::views::as_rational_const(ctx, exp, 8)?;
        if exponent == BigRational::new(1.into(), 2.into()) {
            return Some(ctx.call_builtin(BuiltinFn::Sqrt, vec![base]));
        }
        if exponent == BigRational::new(3.into(), 2.into()) {
            return Some(product_with_sqrt_for_positive_three_half_power(ctx, base));
        }
    }

    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        if expression_contains_positive_three_half_power(ctx, inner) {
            return None;
        }
        let compact = compact_positive_half_power_result_for_integration_presentation(ctx, inner)?;
        return Some(ctx.add(Expr::Neg(compact)));
    }

    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };
    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    let mut coefficient = BigRational::one();
    let mut base = None;

    for factor in factors {
        match ctx.get(factor) {
            Expr::Pow(pow_base, exp) => {
                let exponent = cas_ast::views::as_rational_const(ctx, *exp, 8)?;
                if !(exponent == BigRational::new(1.into(), 2.into())
                    || exponent == BigRational::new(3.into(), 2.into()))
                    || base.is_some()
                {
                    return None;
                }
                base = Some((*pow_base, exponent));
            }
            _ => {
                coefficient *= cas_ast::views::as_rational_const(ctx, factor, 8)?;
            }
        }
    }

    let (base, exponent) = base?;
    if coefficient.is_zero() {
        return Some(ctx.num(0));
    }
    let core = if exponent == BigRational::new(1.into(), 2.into()) {
        ctx.call_builtin(BuiltinFn::Sqrt, vec![base])
    } else if exponent == BigRational::new(3.into(), 2.into()) {
        let product = product_with_sqrt_for_positive_three_half_power(ctx, base);
        return Some(scale_three_half_power_product_for_presentation(
            ctx,
            coefficient,
            product,
        ));
    } else {
        return None;
    };
    Some(scale_expr_for_calculus_presentation(ctx, coefficient, core))
}

fn expression_contains_positive_three_half_power(ctx: &Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Pow(_, exp) => cas_ast::views::as_rational_const(ctx, *exp, 8)
            .is_some_and(|exponent| exponent == BigRational::new(3.into(), 2.into())),
        Expr::Mul(_, _) => cas_math::expr_nary::mul_leaves(ctx, expr)
            .iter()
            .any(|factor| expression_contains_positive_three_half_power(ctx, *factor)),
        _ => false,
    }
}

fn product_with_sqrt_for_positive_three_half_power(ctx: &mut Context, base: ExprId) -> ExprId {
    let sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![base]);
    cas_math::expr_nary::build_balanced_mul(ctx, &[base, sqrt])
}

fn scale_three_half_power_product_for_presentation(
    ctx: &mut Context,
    coefficient: BigRational,
    product: ExprId,
) -> ExprId {
    if coefficient.is_negative() {
        let positive = scale_expr_for_calculus_presentation(ctx, -coefficient, product);
        return ctx.add(Expr::Neg(positive));
    }
    scale_expr_for_calculus_presentation(ctx, coefficient, product)
}
