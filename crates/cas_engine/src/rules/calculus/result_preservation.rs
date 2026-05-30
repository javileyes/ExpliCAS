use super::polynomial_support::polynomial_radicand_for_calculus_presentation;
use super::presentation_utils::rational_const_for_hold;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use cas_math::root_forms::extract_square_root_base;

pub(super) fn sqrt_reciprocal_trig_antiderivative_result(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Sec | BuiltinFn::Csc)) =>
        {
            let Some(radicand) = extract_square_root_base(ctx, args[0]) else {
                return false;
            };
            polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name).is_some_and(
                |poly| {
                    let derivative = poly.derivative();
                    !derivative.is_zero() && derivative.degree() == 0
                },
            )
        }
        Expr::Neg(inner) => sqrt_reciprocal_trig_antiderivative_result(ctx, inner, var_name),
        Expr::Mul(_, _) => {
            let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
            let mut non_numeric = Vec::new();
            for factor in factors {
                if rational_const_for_hold(ctx, factor).is_none() {
                    non_numeric.push(factor);
                }
            }
            non_numeric.len() == 1
                && sqrt_reciprocal_trig_antiderivative_result(ctx, non_numeric[0], var_name)
        }
        _ => false,
    }
}

pub(super) fn inverse_sqrt_quotient_arg_result(ctx: &Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(
                    ctx.builtin_of(*fn_id),
                    Some(
                        BuiltinFn::Arcsin | BuiltinFn::Asin | BuiltinFn::Arctan | BuiltinFn::Asinh
                    )
                ) =>
        {
            matches!(ctx.get(args[0]), Expr::Div(_, den) if extract_square_root_base(ctx, *den).is_some())
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right) => {
            inverse_sqrt_quotient_arg_result(ctx, *left)
                || inverse_sqrt_quotient_arg_result(ctx, *right)
        }
        Expr::Neg(inner) => inverse_sqrt_quotient_arg_result(ctx, *inner),
        _ => false,
    }
}

pub(super) fn has_sqrt_denominator_result(ctx: &Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Div(num, den) => {
            extract_square_root_base(ctx, *den).is_some()
                || has_sqrt_denominator_result(ctx, *num)
                || has_sqrt_denominator_result(ctx, *den)
        }
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            has_sqrt_denominator_result(ctx, *left) || has_sqrt_denominator_result(ctx, *right)
        }
        Expr::Neg(inner) => has_sqrt_denominator_result(ctx, *inner),
        _ => false,
    }
}

pub(super) fn target_has_top_level_negative_orientation(ctx: &Context, target: ExprId) -> bool {
    match ctx.get(target) {
        Expr::Neg(_) => true,
        Expr::Mul(left, right) => {
            matches!(ctx.get(*left), Expr::Neg(_)) || matches!(ctx.get(*right), Expr::Neg(_))
        }
        _ => false,
    }
}

pub(super) fn expr_contains_direct_trig_with_affine_arg(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(cas_ast::hold::unwrap_internal_hold(ctx, current)) {
            Expr::Function(fn_id, args)
                if args.len() == 1
                    && matches!(
                        ctx.builtin_of(*fn_id),
                        Some(BuiltinFn::Sin | BuiltinFn::Cos)
                    )
                    && Polynomial::from_expr(ctx, args[0], var_name)
                        .is_ok_and(|poly| poly.degree() == 1) =>
            {
                return true;
            }
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::Pow(left, right) => {
                stack.push(*left);
                stack.push(*right);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Number(_)
            | Expr::Constant(_)
            | Expr::Variable(_)
            | Expr::SessionRef(_)
            | Expr::Matrix { .. } => {}
        }
    }
    false
}
