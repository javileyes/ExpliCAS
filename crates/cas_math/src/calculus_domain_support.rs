//! Shared real-domain predicates for calculus-facing policies.

use crate::tri_proof::TriProof;
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use num_integer::Integer;
use num_traits::{One, Zero};

/// Returns true when `expr > 0` has no real-domain solution provable within
/// `proof_depth`, by proving sign-preserved `-expr >= 0`.
pub fn positive_condition_is_impossible_over_reals(
    ctx: &mut Context,
    expr: ExprId,
    proof_depth: usize,
) -> bool {
    let negated = ctx.add(Expr::Neg(expr));
    nonnegative_condition_is_proven_over_reals(ctx, negated, proof_depth)
}

/// Returns true when `base` cannot be a valid real logarithm base because
/// `base > 0` is impossible or because the base is the constant `1`.
pub fn log_base_is_invalid_over_reals(ctx: &mut Context, base: ExprId, proof_depth: usize) -> bool {
    positive_condition_is_impossible_over_reals(ctx, base, proof_depth)
        || crate::numeric_eval::as_rational_const(ctx, base).is_some_and(|value| value.is_one())
}

/// Returns true when a logarithm call has no real-domain value for any input
/// assignment provable within `proof_depth`.
pub fn logarithm_real_domain_is_empty_over_reals(
    ctx: &mut Context,
    builtin: Option<BuiltinFn>,
    args: &[ExprId],
    proof_depth: usize,
) -> bool {
    match builtin {
        Some(BuiltinFn::Ln | BuiltinFn::Log2 | BuiltinFn::Log10) if args.len() == 1 => {
            positive_condition_is_impossible_over_reals(ctx, args[0], proof_depth)
        }
        Some(BuiltinFn::Log) if args.len() == 2 => {
            log_base_is_invalid_over_reals(ctx, args[0], proof_depth)
                || positive_condition_is_impossible_over_reals(ctx, args[1], proof_depth)
        }
        _ => false,
    }
}

/// Returns true when a variable-independent calculus expression has a statically
/// empty real domain. Unknown symbolic domains are deliberately preserved.
pub fn real_domain_is_empty_for_static_expr(
    ctx: &mut Context,
    expr: ExprId,
    proof_depth: usize,
    scan_depth: usize,
) -> bool {
    real_domain_issue_over_reals(ctx, expr, proof_depth, scan_depth, false)
}

/// Returns true when `expr` contains a statically empty real-domain subexpression
/// or a nonfinite/undefined constant. This is the shared calculus guard for
/// commands that must reject nonfinite operands instead of treating them as
/// valid constants.
pub fn real_domain_is_empty_or_nonfinite_over_reals(
    ctx: &mut Context,
    expr: ExprId,
    proof_depth: usize,
    scan_depth: usize,
) -> bool {
    real_domain_issue_over_reals(ctx, expr, proof_depth, scan_depth, true)
}

/// Returns true for top-level nonfinite constants, allowing transparent wrappers
/// such as unary negation and hold.
pub fn nonfinite_or_undefined_constant(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Constant(Constant::Infinity | Constant::Undefined) => true,
        Expr::Neg(inner) | Expr::Hold(inner) => nonfinite_or_undefined_constant(ctx, *inner),
        _ => false,
    }
}

fn real_domain_issue_over_reals(
    ctx: &mut Context,
    expr: ExprId,
    proof_depth: usize,
    scan_depth: usize,
    include_nonfinite: bool,
) -> bool {
    if include_nonfinite && nonfinite_or_undefined_constant(ctx, expr) {
        return true;
    }
    if scan_depth == 0 {
        return false;
    }

    match ctx.get(expr).clone() {
        Expr::Constant(Constant::Undefined) => true,
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            real_domain_issue_over_reals(ctx, left, proof_depth, scan_depth - 1, include_nonfinite)
                || real_domain_issue_over_reals(
                    ctx,
                    right,
                    proof_depth,
                    scan_depth - 1,
                    include_nonfinite,
                )
        }
        Expr::Div(num, den) => {
            crate::numeric_eval::as_rational_const(ctx, den).is_some_and(|value| value.is_zero())
                || real_domain_issue_over_reals(
                    ctx,
                    num,
                    proof_depth,
                    scan_depth - 1,
                    include_nonfinite,
                )
                || real_domain_issue_over_reals(
                    ctx,
                    den,
                    proof_depth,
                    scan_depth - 1,
                    include_nonfinite,
                )
        }
        Expr::Pow(base, exp) => {
            if rational_exponent_requires_nonnegative_base(ctx, exp)
                && nonnegative_condition_is_impossible_over_reals(ctx, base, proof_depth)
            {
                return true;
            }
            real_domain_issue_over_reals(ctx, base, proof_depth, scan_depth - 1, include_nonfinite)
                || real_domain_issue_over_reals(
                    ctx,
                    exp,
                    proof_depth,
                    scan_depth - 1,
                    include_nonfinite,
                )
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            real_domain_issue_over_reals(ctx, inner, proof_depth, scan_depth - 1, include_nonfinite)
        }
        Expr::Function(fn_id, args) => {
            let builtin = ctx.builtin_of(fn_id);
            if logarithm_real_domain_is_empty_over_reals(ctx, builtin, &args, proof_depth) {
                return true;
            }
            if matches!(builtin, Some(BuiltinFn::Sqrt))
                && args.len() == 1
                && nonnegative_condition_is_impossible_over_reals(ctx, args[0], proof_depth)
            {
                return true;
            }

            args.into_iter().any(|arg| {
                real_domain_issue_over_reals(
                    ctx,
                    arg,
                    proof_depth,
                    scan_depth - 1,
                    include_nonfinite,
                )
            })
        }
        _ => false,
    }
}

fn rational_exponent_requires_nonnegative_base(ctx: &Context, exp: ExprId) -> bool {
    crate::numeric_eval::as_rational_const(ctx, exp)
        .is_some_and(|value| !value.denom().is_one() && value.denom().is_even())
}

/// Returns true when `expr >= 0` has no real-domain solution provable within
/// `proof_depth`, by proving sign-preserved `-expr > 0`.
pub fn nonnegative_condition_is_impossible_over_reals(
    ctx: &mut Context,
    expr: ExprId,
    proof_depth: usize,
) -> bool {
    let negated = ctx.add(Expr::Neg(expr));
    let normalized =
        crate::expr_normalization::normalize_condition_expr_preserve_sign(ctx, negated);
    crate::prove_sign::prove_positive_depth_with(
        ctx,
        normalized,
        proof_depth,
        true,
        |_inner_ctx, _inner_expr, _inner_depth| TriProof::Unknown,
    )
    .is_proven()
}

/// Returns true when `expr >= 0` is provable over the real domain within
/// `proof_depth` after sign-preserving condition normalization.
pub fn nonnegative_condition_is_proven_over_reals(
    ctx: &mut Context,
    expr: ExprId,
    proof_depth: usize,
) -> bool {
    let normalized = crate::expr_normalization::normalize_condition_expr_preserve_sign(ctx, expr);
    crate::prove_sign::prove_nonnegative_depth_with(
        ctx,
        normalized,
        proof_depth,
        true,
        |_inner_ctx, _inner_expr, _inner_depth| TriProof::Unknown,
    )
    .is_proven()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_impossible_positive_condition_for_nonpositive_quadratics() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let x_squared = ctx.add(Expr::Pow(x, two));
        let neg_x_squared = ctx.add(Expr::Neg(x_squared));
        let one = ctx.num(1);
        let nonpositive = ctx.add(Expr::Sub(neg_x_squared, one));

        assert!(positive_condition_is_impossible_over_reals(
            &mut ctx,
            nonpositive,
            12
        ));

        let neg_square = ctx.add(Expr::Neg(x_squared));
        assert!(positive_condition_is_impossible_over_reals(
            &mut ctx, neg_square, 12
        ));
    }

    #[test]
    fn detects_invalid_real_log_base_policy() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let one = ctx.num(1);
        let two = ctx.num(2);
        let neg_two = ctx.add(Expr::Neg(two));

        assert!(log_base_is_invalid_over_reals(&mut ctx, zero, 12));
        assert!(log_base_is_invalid_over_reals(&mut ctx, one, 12));
        assert!(log_base_is_invalid_over_reals(&mut ctx, neg_two, 12));

        let two = ctx.num(2);
        assert!(!log_base_is_invalid_over_reals(&mut ctx, two, 12));
    }

    #[test]
    fn detects_impossible_nonnegative_condition_for_strictly_negative_quadratics() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let x_squared = ctx.add(Expr::Pow(x, two));
        let neg_x_squared = ctx.add(Expr::Neg(x_squared));
        let one = ctx.num(1);
        let strictly_negative = ctx.add(Expr::Sub(neg_x_squared, one));

        assert!(nonnegative_condition_is_impossible_over_reals(
            &mut ctx,
            strictly_negative,
            12
        ));

        let x_squared = ctx.add(Expr::Pow(x, two));
        let nonpositive = ctx.add(Expr::Neg(x_squared));
        assert!(!nonnegative_condition_is_impossible_over_reals(
            &mut ctx,
            nonpositive,
            12
        ));
    }

    #[test]
    fn proves_shifted_square_nonnegative_condition() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let shifted = ctx.add(Expr::Add(x, one));
        let two = ctx.num(2);
        let shifted_square = ctx.add(Expr::Pow(shifted, two));

        assert!(nonnegative_condition_is_proven_over_reals(
            &mut ctx,
            shifted_square,
            12
        ));
    }

    #[test]
    fn preserves_unknown_or_nonempty_positive_conditions() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let x_squared = ctx.add(Expr::Pow(x, two));
        let one = ctx.num(1);
        let sign_changing = ctx.add(Expr::Sub(x_squared, one));

        assert!(!positive_condition_is_impossible_over_reals(
            &mut ctx,
            sign_changing,
            12
        ));

        let one = ctx.num(1);
        let strictly_positive = ctx.add(Expr::Add(x_squared, one));
        assert!(!positive_condition_is_impossible_over_reals(
            &mut ctx,
            strictly_positive,
            12
        ));
    }

    #[test]
    fn detects_static_empty_real_domain_expressions() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let one = ctx.num(1);
        let two = ctx.num(2);
        let neg_one = ctx.add(Expr::Neg(one));
        let sqrt_neg_one = ctx.call_builtin(BuiltinFn::Sqrt, vec![neg_one]);
        let ln_zero = ctx.call_builtin(BuiltinFn::Ln, vec![zero]);
        let base_one_log = ctx.call_builtin(BuiltinFn::Log, vec![one, two]);

        assert!(real_domain_is_empty_for_static_expr(
            &mut ctx,
            sqrt_neg_one,
            12,
            16
        ));
        assert!(real_domain_is_empty_for_static_expr(
            &mut ctx, ln_zero, 12, 16
        ));
        assert!(real_domain_is_empty_for_static_expr(
            &mut ctx,
            base_one_log,
            12,
            16
        ));
    }

    #[test]
    fn preserves_symbolic_or_nonempty_static_domains() {
        let mut ctx = Context::new();
        let y = ctx.var("y");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let y_squared = ctx.add(Expr::Pow(y, two));
        let nonempty = ctx.add(Expr::Neg(y_squared));
        let sqrt_nonempty = ctx.call_builtin(BuiltinFn::Sqrt, vec![nonempty]);
        let ln_symbolic = ctx.call_builtin(BuiltinFn::Ln, vec![y]);
        let sqrt_one = ctx.call_builtin(BuiltinFn::Sqrt, vec![one]);

        assert!(!real_domain_is_empty_for_static_expr(
            &mut ctx,
            sqrt_nonempty,
            12,
            16
        ));
        assert!(!real_domain_is_empty_for_static_expr(
            &mut ctx,
            ln_symbolic,
            12,
            16
        ));
        assert!(!real_domain_is_empty_for_static_expr(
            &mut ctx, sqrt_one, 12, 16
        ));
    }

    #[test]
    fn nonfinite_scan_is_explicit_policy_not_static_empty_domain() {
        let mut ctx = Context::new();
        let infinity = ctx.add(Expr::Constant(Constant::Infinity));
        let one = ctx.num(1);
        let expr = ctx.add(Expr::Add(infinity, one));

        assert!(!real_domain_is_empty_for_static_expr(
            &mut ctx, expr, 12, 16
        ));
        assert!(real_domain_is_empty_or_nonfinite_over_reals(
            &mut ctx, expr, 12, 16
        ));
    }
}
