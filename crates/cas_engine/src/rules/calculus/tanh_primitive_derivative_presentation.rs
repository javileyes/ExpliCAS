//! Post-calculus presentation for bounded `tanh` primitive derivatives.
//!
//! This module owns the narrow route that recognizes derivatives of primitives
//! shaped like `x - tanh(u) - tanh(u)^3/3 - ...` and presents the result as a
//! compact even power of `tanh(u)`. The caller keeps the same route order and
//! domain policy; this module only separates the family-specific parser and
//! presentation builder from `calculus/mod.rs`.

use super::polynomial_support::nonzero_affine_variable_derivative;
use super::presentation_utils::unwrap_internal_hold_for_calculus;
use super::scalar_presentation::scale_expr_for_calculus_presentation;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};

fn tanh_power_term_for_derivative_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, u32)> {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    if let Expr::Function(fn_id, args) = ctx.get(expr).clone() {
        if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Tanh) {
            return Some((args[0], 1));
        }
        return None;
    }

    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    let power = cas_ast::views::as_rational_const(ctx, exp, 4)?;
    if power.denom() != &1.into() {
        return None;
    }
    let power = power.numer().to_u32()?;
    if !matches!(power, 3 | 5 | 7) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(base).clone() else {
        return None;
    };
    if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Tanh) {
        Some((args[0], power))
    } else {
        None
    }
}

fn collect_scaled_tanh_even_primitive_terms_for_derivative_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    scale: BigRational,
    linear_coeff: &mut BigRational,
    terms: &mut Vec<(ExprId, u32, BigRational)>,
) -> Option<()> {
    if scale.is_zero() {
        return Some(());
    }

    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Number(_) => Some(()),
        Expr::Add(left, right) => {
            collect_scaled_tanh_even_primitive_terms_for_derivative_presentation(
                ctx,
                left,
                var_name,
                scale.clone(),
                linear_coeff,
                terms,
            )?;
            collect_scaled_tanh_even_primitive_terms_for_derivative_presentation(
                ctx,
                right,
                var_name,
                scale,
                linear_coeff,
                terms,
            )
        }
        Expr::Sub(left, right) => {
            collect_scaled_tanh_even_primitive_terms_for_derivative_presentation(
                ctx,
                left,
                var_name,
                scale.clone(),
                linear_coeff,
                terms,
            )?;
            collect_scaled_tanh_even_primitive_terms_for_derivative_presentation(
                ctx,
                right,
                var_name,
                -scale,
                linear_coeff,
                terms,
            )
        }
        Expr::Neg(inner) => collect_scaled_tanh_even_primitive_terms_for_derivative_presentation(
            ctx,
            inner,
            var_name,
            -scale,
            linear_coeff,
            terms,
        ),
        Expr::Div(num, den) => {
            let den_scale = cas_ast::views::as_rational_const(ctx, den, 8)?;
            if den_scale.is_zero() {
                return None;
            }
            collect_scaled_tanh_even_primitive_terms_for_derivative_presentation(
                ctx,
                num,
                var_name,
                scale / den_scale,
                linear_coeff,
                terms,
            )
        }
        Expr::Mul(_, _) => {
            let mut term_scale = scale;
            let mut non_numeric = None;
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                    term_scale *= value;
                    continue;
                }
                if non_numeric.replace(factor).is_some() {
                    return None;
                }
            }
            collect_scaled_tanh_even_primitive_terms_for_derivative_presentation(
                ctx,
                non_numeric?,
                var_name,
                term_scale,
                linear_coeff,
                terms,
            )
        }
        Expr::Variable(sym_id) if ctx.sym_name(sym_id) == var_name => {
            *linear_coeff += scale;
            Some(())
        }
        _ => {
            let (arg, power) = tanh_power_term_for_derivative_presentation(ctx, expr)?;
            terms.push((arg, power, scale));
            Some(())
        }
    }
}

fn tanh_even_primitive_derivative_presentation_coeff(
    terms: &[(ExprId, u32, BigRational)],
    power: u32,
) -> BigRational {
    terms
        .iter()
        .filter_map(|(_, term_power, coeff)| (*term_power == power).then_some(coeff.clone()))
        .fold(BigRational::zero(), |acc, coeff| acc + coeff)
}

pub(super) fn affine_tanh_even_primitive_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let mut linear_coeff = BigRational::zero();
    let mut terms = Vec::new();
    collect_scaled_tanh_even_primitive_terms_for_derivative_presentation(
        ctx,
        target,
        var_name,
        BigRational::one(),
        &mut linear_coeff,
        &mut terms,
    )?;
    if linear_coeff.is_zero() || terms.is_empty() {
        return None;
    }

    let arg = terms
        .iter()
        .find_map(|(arg, _, coeff)| (!coeff.is_zero()).then_some(*arg))?;
    if terms.iter().any(|(term_arg, power, coeff)| {
        !coeff.is_zero()
            && (compare_expr(ctx, *term_arg, arg) != std::cmp::Ordering::Equal
                || !matches!(*power, 1 | 3 | 5 | 7))
    }) {
        return None;
    }

    let slope = nonzero_affine_variable_derivative(ctx, arg, var_name)?;
    let linear_tanh_coeff = tanh_even_primitive_derivative_presentation_coeff(&terms, 1);
    let cubic_coeff = tanh_even_primitive_derivative_presentation_coeff(&terms, 3);
    let fifth_coeff = tanh_even_primitive_derivative_presentation_coeff(&terms, 5);
    let seventh_coeff = tanh_even_primitive_derivative_presentation_coeff(&terms, 7);
    if linear_tanh_coeff.is_zero() || cubic_coeff.is_zero() || fifth_coeff.is_zero() {
        return None;
    }

    if linear_tanh_coeff != -(linear_coeff.clone() / slope.clone()) {
        return None;
    }
    if cubic_coeff
        != -(linear_coeff.clone() / (slope.clone() * BigRational::from_integer(3.into())))
    {
        return None;
    }
    if fifth_coeff
        != -(linear_coeff.clone() / (slope.clone() * BigRational::from_integer(5.into())))
    {
        return None;
    }
    let output_power = if seventh_coeff.is_zero() {
        6
    } else {
        if seventh_coeff != -(linear_coeff.clone() / (slope * BigRational::from_integer(7.into())))
        {
            return None;
        }
        8
    };

    let tanh_arg = ctx.call_builtin(BuiltinFn::Tanh, vec![arg]);
    let output_power = ctx.num(output_power);
    let tanh_power = ctx.add(Expr::Pow(tanh_arg, output_power));
    Some(scale_expr_for_calculus_presentation(
        ctx,
        linear_coeff,
        tanh_power,
    ))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::affine_tanh_even_primitive_derivative_presentation;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn affine_tanh_even_primitive_derivative_presentation_accepts_direct_arg() {
        let mut ctx = Context::new();
        let expr = parse("x - tanh(x) - tanh(x)^3/3 - tanh(x)^5/5", &mut ctx).unwrap();
        let compact =
            affine_tanh_even_primitive_derivative_presentation(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, compact), "tanh(x)^6");
    }

    #[test]
    fn affine_tanh_even_primitive_derivative_presentation_accepts_positive_affine_arg() {
        let mut ctx = Context::new();
        let expr = parse(
            "x - 1/2*(tanh(2*x+1) + tanh(2*x+1)^3/3 + tanh(2*x+1)^5/5)",
            &mut ctx,
        )
        .unwrap();
        let compact =
            affine_tanh_even_primitive_derivative_presentation(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, compact), "tanh(2 * x + 1)^6");
    }

    #[test]
    fn affine_tanh_even_primitive_derivative_presentation_accepts_negative_affine_arg() {
        let mut ctx = Context::new();
        let expr = parse(
            "x + 1/2*(tanh(1-2*x) + tanh(1-2*x)^3/3 + tanh(1-2*x)^5/5)",
            &mut ctx,
        )
        .unwrap();
        let compact =
            affine_tanh_even_primitive_derivative_presentation(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, compact), "tanh(1 - 2 * x)^6");
    }

    #[test]
    fn affine_tanh_even_primitive_derivative_presentation_accepts_eighth_power() {
        let mut ctx = Context::new();
        let expr = parse(
            "x - 1/2*(tanh(2*x+1) + tanh(2*x+1)^3/3 + tanh(2*x+1)^5/5 + tanh(2*x+1)^7/7)",
            &mut ctx,
        )
        .unwrap();
        let compact =
            affine_tanh_even_primitive_derivative_presentation(&mut ctx, expr, "x").unwrap();
        assert_eq!(rendered(&ctx, compact), "tanh(2 * x + 1)^8");

        let expr = parse(
            "x + 1/2*(tanh(1-2*x) + tanh(1-2*x)^3/3 + tanh(1-2*x)^5/5 + tanh(1-2*x)^7/7)",
            &mut ctx,
        )
        .unwrap();
        let compact =
            affine_tanh_even_primitive_derivative_presentation(&mut ctx, expr, "x").unwrap();
        assert_eq!(rendered(&ctx, compact), "tanh(1 - 2 * x)^8");
    }
}
