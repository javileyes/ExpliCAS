//! Source-side sqrt-chain reciprocal-trig-product integrand detection.

use super::polynomial_support::polynomial_radicand_for_calculus_presentation;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::root_forms::extract_square_root_base;

pub(super) fn sqrt_reciprocal_trig_product_integrand_target(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let mut primary_radicands = Vec::new();
    let mut companion_radicands = Vec::new();
    let mut denominator_radicands = Vec::new();
    collect_sqrt_reciprocal_trig_product_signal(
        ctx,
        target,
        false,
        &mut primary_radicands,
        &mut companion_radicands,
        &mut denominator_radicands,
    );

    primary_radicands.iter().any(|primary| {
        let same_companion = companion_radicands
            .iter()
            .any(|companion| compare_expr(ctx, *primary, *companion) == std::cmp::Ordering::Equal);
        let same_denominator = denominator_radicands.iter().any(|denominator| {
            compare_expr(ctx, *primary, *denominator) == std::cmp::Ordering::Equal
        });
        same_companion
            && same_denominator
            && polynomial_radicand_for_calculus_presentation(ctx, *primary, var_name).is_some_and(
                |poly| {
                    let derivative = poly.derivative();
                    !derivative.is_zero() && derivative.degree() == 0
                },
            )
    })
}

fn collect_sqrt_reciprocal_trig_product_signal(
    ctx: &mut Context,
    root: ExprId,
    in_denominator: bool,
    primary_radicands: &mut Vec<ExprId>,
    companion_radicands: &mut Vec<ExprId>,
    denominator_radicands: &mut Vec<ExprId>,
) {
    if in_denominator {
        if let Some(radicand) = extract_square_root_base(ctx, root) {
            denominator_radicands.push(radicand);
        }
    }

    match ctx.get(root).clone() {
        Expr::Function(fn_id, args) => {
            if args.len() == 1 {
                if let Some(radicand) = extract_square_root_base(ctx, args[0]) {
                    match ctx.builtin_of(fn_id) {
                        Some(BuiltinFn::Sec | BuiltinFn::Csc) => {
                            primary_radicands.push(radicand);
                        }
                        Some(BuiltinFn::Tan | BuiltinFn::Cot) => {
                            companion_radicands.push(radicand);
                        }
                        _ => {}
                    }
                }
            }
            for arg in args {
                collect_sqrt_reciprocal_trig_product_signal(
                    ctx,
                    arg,
                    in_denominator,
                    primary_radicands,
                    companion_radicands,
                    denominator_radicands,
                );
            }
        }
        Expr::Div(numerator, denominator) => {
            collect_sqrt_reciprocal_trig_product_signal(
                ctx,
                numerator,
                in_denominator,
                primary_radicands,
                companion_radicands,
                denominator_radicands,
            );
            collect_sqrt_reciprocal_trig_product_signal(
                ctx,
                denominator,
                true,
                primary_radicands,
                companion_radicands,
                denominator_radicands,
            );
        }
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            collect_sqrt_reciprocal_trig_product_signal(
                ctx,
                left,
                in_denominator,
                primary_radicands,
                companion_radicands,
                denominator_radicands,
            );
            collect_sqrt_reciprocal_trig_product_signal(
                ctx,
                right,
                in_denominator,
                primary_radicands,
                companion_radicands,
                denominator_radicands,
            );
        }
        Expr::Pow(base, exp) => {
            collect_sqrt_reciprocal_trig_product_signal(
                ctx,
                base,
                in_denominator,
                primary_radicands,
                companion_radicands,
                denominator_radicands,
            );
            collect_sqrt_reciprocal_trig_product_signal(
                ctx,
                exp,
                in_denominator,
                primary_radicands,
                companion_radicands,
                denominator_radicands,
            );
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            collect_sqrt_reciprocal_trig_product_signal(
                ctx,
                inner,
                in_denominator,
                primary_radicands,
                companion_radicands,
                denominator_radicands,
            );
        }
        Expr::Matrix { data, .. } => {
            for item in data {
                collect_sqrt_reciprocal_trig_product_signal(
                    ctx,
                    item,
                    in_denominator,
                    primary_radicands,
                    companion_radicands,
                    denominator_radicands,
                );
            }
        }
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
    }
}
