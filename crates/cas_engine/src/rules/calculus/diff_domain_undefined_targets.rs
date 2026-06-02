use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;
use cas_math::root_forms::extract_square_root_base;
use num_traits::Signed;

const CALCULUS_DOMAIN_PROOF_DEPTH: usize = 12;

fn nonfinite_or_undefined_calculus_target(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Constant(Constant::Infinity | Constant::Undefined) => true,
        Expr::Neg(inner) | Expr::Hold(inner) => nonfinite_or_undefined_calculus_target(ctx, *inner),
        _ => false,
    }
}

fn logarithm_known_empty_positive_domain(ctx: &mut Context, target: ExprId) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return false;
    };
    match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Ln | BuiltinFn::Log2 | BuiltinFn::Log10) if args.len() == 1 => {
            cas_math::calculus_domain_support::positive_condition_is_impossible_over_reals(
                ctx,
                args[0],
                CALCULUS_DOMAIN_PROOF_DEPTH,
            )
        }
        Some(BuiltinFn::Log) if args.len() == 2 => {
            cas_math::calculus_domain_support::log_base_is_invalid_over_reals(
                ctx,
                args[0],
                CALCULUS_DOMAIN_PROOF_DEPTH,
            ) || cas_math::calculus_domain_support::positive_condition_is_impossible_over_reals(
                ctx,
                args[1],
                CALCULUS_DOMAIN_PROOF_DEPTH,
            )
        }
        _ => false,
    }
}

fn sqrt_known_empty_positive_domain(ctx: &mut Context, target: ExprId, var_name: &str) -> bool {
    let Some(radicand) = extract_square_root_base(ctx, target) else {
        return false;
    };
    if !contains_named_var(ctx, radicand, var_name) {
        return cas_ast::views::as_rational_const(ctx, radicand, 8)
            .is_some_and(|value| value.is_negative());
    }

    cas_math::calculus_domain_support::positive_condition_is_impossible_over_reals(
        ctx,
        radicand,
        CALCULUS_DOMAIN_PROOF_DEPTH,
    )
}

pub(crate) fn diff_target_known_undefined_over_reals(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    nonfinite_or_undefined_calculus_target(ctx, target)
        || logarithm_known_empty_positive_domain(ctx, target)
        || sqrt_known_empty_positive_domain(ctx, target, var_name)
}
