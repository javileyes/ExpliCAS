//! Derivative presentation for `sqrt(a * log_b(f(x)) + c)`.

use super::polynomial_support::split_polynomial_content_for_calculus_presentation;
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    scale_expr_for_calculus_presentation,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::{One, Signed};

pub(super) fn sqrt_shifted_ln_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = extract_square_root_base(ctx, target)?;
    let (ln_arg, ln_scale, base_ln_factor, shift) =
        scaled_ln_plus_positive_rational_shift(ctx, radicand)?;
    if !ln_scale.is_positive() || !shift.is_positive() {
        return None;
    }

    let arg_poly = Polynomial::from_expr(ctx, ln_arg, var_name).ok()?;
    let derivative_poly = arg_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = ln_scale * derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let mut denominator_factors = Vec::new();
    if !denominator_coeff.is_one() {
        denominator_factors.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_coeff,
        ));
    }
    denominator_factors.push(ln_arg);
    if let Some(base_ln_factor) = base_ln_factor {
        denominator_factors.push(base_ln_factor);
    }
    denominator_factors.push(sqrt_radicand);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn scaled_ln_plus_positive_rational_shift(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational, Option<ExprId>, BigRational)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let (left, right) = match ctx.get(expr) {
        Expr::Add(left, right) => (*left, *right),
        _ => return None,
    };

    if let Some((arg, scale, base_ln_factor)) = scaled_ln_term_arg(ctx, left) {
        if let Some(shift) = cas_ast::views::as_rational_const(ctx, right, 8) {
            return Some((arg, scale, base_ln_factor, shift));
        }
    }
    if let Some((arg, scale, base_ln_factor)) = scaled_ln_term_arg(ctx, right) {
        if let Some(shift) = cas_ast::views::as_rational_const(ctx, left, 8) {
            return Some((arg, scale, base_ln_factor, shift));
        }
    }

    None
}

fn scaled_ln_term_arg(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational, Option<ExprId>)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Some((arg, base_ln_factor)) = shifted_root_log_term_arg(ctx, expr) {
        return Some((arg, BigRational::one(), base_ln_factor));
    }

    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };

    let mut scale = BigRational::one();
    let mut ln_arg = None;
    let mut base_ln_factor = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
            continue;
        }
        if let Some((arg, factor_base_ln)) = shifted_root_log_term_arg(ctx, factor) {
            if ln_arg.replace(arg).is_none() && base_ln_factor.replace(factor_base_ln).is_none() {
                continue;
            }
        }
        return None;
    }

    Some((ln_arg?, scale, base_ln_factor?))
}

fn shifted_root_log_term_arg(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, Option<ExprId>)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Ln) => Some((args[0], None)),
        Some(BuiltinFn::Log2) => {
            let two = ctx.num(2);
            Some((args[0], Some(ctx.call_builtin(BuiltinFn::Ln, vec![two]))))
        }
        Some(BuiltinFn::Log10) => {
            let ten = ctx.num(10);
            Some((args[0], Some(ctx.call_builtin(BuiltinFn::Ln, vec![ten]))))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::sqrt_shifted_ln_derivative_presentation;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn sqrt_shifted_ln_derivative_keeps_compact_denominator() {
        let mut ctx = Context::new();
        let target = parse("sqrt(ln(x^2+1)+1)", &mut ctx).unwrap();
        let derivative = sqrt_shifted_ln_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "x / ((x^2 + 1) * sqrt(ln(x^2 + 1) + 1))"
        );
    }

    #[test]
    fn sqrt_shifted_log2_derivative_keeps_base_factor() {
        let mut ctx = Context::new();
        let target = parse("sqrt(log2(x^2+1)+1)", &mut ctx).unwrap();
        let derivative = sqrt_shifted_ln_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "x / ((x^2 + 1) * ln(2) * sqrt(log2(x^2 + 1) + 1))"
        );
    }
}
