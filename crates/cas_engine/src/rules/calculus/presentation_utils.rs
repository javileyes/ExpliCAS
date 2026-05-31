use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::multipoly::{multipoly_from_expr, PolyBudget};
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

pub(super) fn unwrap_internal_hold_for_calculus(ctx: &mut Context, target: ExprId) -> ExprId {
    cas_ast::hold::strip_all_holds(ctx, target)
}

pub(super) fn variable_named(ctx: &Context, expr: ExprId, var_name: &str) -> bool {
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var_name)
}

pub(super) fn squared_expr(ctx: &mut Context, expr: ExprId) -> ExprId {
    let two = ctx.num(2);
    ctx.add(Expr::Pow(expr, two))
}

pub(super) fn sqrt_raw_for_calculus_presentation(ctx: &mut Context, radicand: ExprId) -> ExprId {
    let fn_id = ctx.builtin_id(BuiltinFn::Sqrt);
    ctx.add_raw(Expr::Function(fn_id, vec![radicand]))
}

pub(super) fn positive_rational_scale_between_exprs(
    ctx: &mut Context,
    scaled: ExprId,
    base: ExprId,
) -> Option<BigRational> {
    let budget = PolyBudget {
        max_terms: 24,
        max_total_degree: 8,
        max_pow_exp: 4,
    };
    if let (Ok(scaled_poly), Ok(base_poly)) = (
        multipoly_from_expr(ctx, scaled, &budget),
        multipoly_from_expr(ctx, base, &budget),
    ) {
        if scaled_poly.vars == base_poly.vars
            && !scaled_poly.terms.is_empty()
            && scaled_poly.terms.len() == base_poly.terms.len()
        {
            let scale = scaled_poly.terms[0].0.clone() / base_poly.terms[0].0.clone();
            if scale.is_positive() && base_poly.mul_scalar(&scale) == scaled_poly {
                return Some(scale);
            }
        }
    }

    positive_rational_scale_between_structural_additions(ctx, scaled, base)
}

fn positive_rational_scale_between_structural_additions(
    ctx: &mut Context,
    scaled: ExprId,
    base: ExprId,
) -> Option<BigRational> {
    let (direct_scale, direct_core) =
        split_numeric_scale_product_for_calculus_presentation(ctx, scaled);
    if direct_scale.is_positive() && structurally_equivalent_for_calculus(ctx, direct_core, base) {
        return Some(direct_scale);
    }

    let scaled_terms = scaled_additive_terms_for_calculus_presentation(ctx, scaled);
    let base_terms = scaled_additive_terms_for_calculus_presentation(ctx, base);
    if scaled_terms.is_empty() || scaled_terms.len() != base_terms.len() {
        return None;
    }

    let mut matched = vec![false; base_terms.len()];
    let mut common_scale = None;
    for (scaled_coeff, scaled_core) in scaled_terms {
        let (index, base_coeff, _base_core) =
            base_terms
                .iter()
                .enumerate()
                .find_map(|(index, (base_coeff, base_core))| {
                    (!matched[index]
                        && structurally_equivalent_for_calculus(ctx, scaled_core, *base_core))
                    .then_some((index, base_coeff, base_core))
                })?;
        if base_coeff.is_zero() {
            return None;
        }
        let scale = scaled_coeff / base_coeff.clone();
        if common_scale.as_ref().is_some_and(|common| common != &scale) {
            return None;
        }
        common_scale = Some(scale);
        matched[index] = true;
    }

    common_scale.filter(|scale: &BigRational| scale.is_positive())
}

fn scaled_additive_terms_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Vec<(BigRational, ExprId)> {
    cas_math::expr_nary::add_terms_signed(ctx, expr)
        .into_iter()
        .map(|(term, sign)| {
            let (scale, core) = split_numeric_scale_product_for_calculus_presentation(ctx, term);
            let scale = if sign == cas_math::expr_nary::Sign::Neg {
                -scale
            } else {
                scale
            };
            (scale, core)
        })
        .collect()
}

fn split_numeric_scale_product_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> (BigRational, ExprId) {
    if let Some(value) = cas_ast::views::as_rational_const(ctx, expr, 8) {
        return (value, ctx.num(1));
    }

    let mut scale = BigRational::one();
    let mut non_numeric = Vec::new();
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else {
            non_numeric.push(factor);
        }
    }

    let core = match non_numeric.as_slice() {
        [] => ctx.num(1),
        [single] => *single,
        _ => cas_math::expr_nary::build_balanced_mul(ctx, &non_numeric),
    };
    (scale, core)
}

pub(super) fn structurally_equivalent_for_calculus(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    left == right || cas_math::expr_domain::exprs_equivalent(ctx, left, right)
}

pub(super) fn is_calculus_presentation_one(ctx: &Context, expr: ExprId) -> bool {
    cas_ast::views::as_rational_const(ctx, expr, 8).is_some_and(|value| value.is_one())
}

pub(super) fn is_half_power_exponent(ctx: &Context, expr: ExprId) -> bool {
    cas_ast::views::as_rational_const(ctx, expr, 8)
        .is_some_and(|value| value == BigRational::new(1.into(), 2.into()))
}

pub(super) fn calculus_sqrt_like_radicand(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    extract_square_root_base(ctx, expr).or_else(|| match ctx.get(expr) {
        Expr::Pow(base, exp) if is_half_power_exponent(ctx, *exp) => Some(*base),
        _ => None,
    })
}

pub(super) fn scaled_sqrt_argument_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Some(radicand) = calculus_sqrt_like_radicand(ctx, expr) {
        return Some((radicand, BigRational::one()));
    }

    match ctx.get(expr) {
        Expr::Neg(inner) => {
            let (radicand, scale) = scaled_sqrt_argument_for_calculus_presentation(ctx, *inner)?;
            Some((radicand, -scale))
        }
        Expr::Div(num, den) => {
            let den_scale = cas_ast::views::as_rational_const(ctx, *den, 8)?;
            if den_scale.is_zero() {
                return None;
            }
            let (radicand, num_scale) = scaled_sqrt_argument_for_calculus_presentation(ctx, *num)?;
            Some((radicand, num_scale / den_scale))
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut radicand = None;
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                    scale *= value;
                    continue;
                }

                if radicand.is_none() {
                    if let Some(base) = calculus_sqrt_like_radicand(ctx, factor) {
                        radicand = Some(base);
                        continue;
                    }
                }

                return None;
            }

            Some((radicand?, scale))
        }
        _ => None,
    }
}

pub(super) fn same_sqrt_like_argument(ctx: &mut Context, left: ExprId, right: ExprId) -> bool {
    if compare_expr(ctx, left, right) == std::cmp::Ordering::Equal {
        return true;
    }
    let Some(left_base) = calculus_sqrt_like_radicand(ctx, left) else {
        return false;
    };
    let Some(right_base) = calculus_sqrt_like_radicand(ctx, right) else {
        return false;
    };
    compare_expr(ctx, left_base, right_base) == std::cmp::Ordering::Equal
}

pub(super) fn multiply_by_sqrt_factor_for_calculus_presentation(
    ctx: &mut Context,
    factor: ExprId,
    radicand: ExprId,
) -> ExprId {
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
        if value.is_one() {
            return sqrt_radicand;
        }
        if value == -BigRational::one() {
            return ctx.add(Expr::Neg(sqrt_radicand));
        }
    }

    cas_math::expr_nary::build_balanced_mul(ctx, &[factor, sqrt_radicand])
}

pub(super) fn rational_const_for_hold(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Number(value) => Some(value.clone()),
        Expr::Div(num, den) => {
            let num = rational_const_for_hold(ctx, *num)?;
            let den = rational_const_for_hold(ctx, *den)?;
            (!den.is_zero()).then_some(num / den)
        }
        Expr::Neg(inner) => rational_const_for_hold(ctx, *inner).map(|value| -value),
        _ => None,
    }
}

pub(super) fn small_rational_const_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<BigRational> {
    if let Some(value) = cas_ast::views::as_rational_const(ctx, expr, 8) {
        return Some(value);
    }

    match ctx.get(expr) {
        Expr::Add(left, right) => Some(
            small_rational_const_for_calculus_presentation(ctx, *left)?
                + small_rational_const_for_calculus_presentation(ctx, *right)?,
        ),
        Expr::Sub(left, right) => Some(
            small_rational_const_for_calculus_presentation(ctx, *left)?
                - small_rational_const_for_calculus_presentation(ctx, *right)?,
        ),
        Expr::Neg(inner) => Some(-small_rational_const_for_calculus_presentation(
            ctx, *inner,
        )?),
        _ => None,
    }
}

pub(super) fn negative_half_power_base_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let exponent = small_rational_const_for_calculus_presentation(ctx, *exp)?;
    (exponent == BigRational::new((-1).into(), 2.into())).then_some(*base)
}
