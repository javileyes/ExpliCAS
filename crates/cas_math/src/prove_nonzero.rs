//! Generic non-zero proof helper extracted from runtime-specific engines.
//!
//! Runtime crates provide:
//! - positivity prover callback (for `base > 0` checks),
//! - optional ground fallback callback for variable-free subtrees.

use crate::expr_extract::extract_unary_log_argument_view;
use crate::expr_predicates::contains_variable;
use crate::pi_helpers::extract_rational_pi_multiple;
use crate::tri_proof::TriProof;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_traits::{One, Zero};

/// Prove whether an expression is non-zero using caller-provided callbacks.
///
/// `depth` bounds recursive descent to prevent stack overflow on malformed trees.
pub fn prove_nonzero_depth_with<FProvePositive, FGroundFallback>(
    ctx: &Context,
    expr: ExprId,
    depth: usize,
    mut prove_positive: FProvePositive,
    mut try_ground_nonzero: FGroundFallback,
) -> TriProof
where
    FProvePositive: FnMut(&Context, ExprId) -> TriProof,
    FGroundFallback: FnMut(&Context, ExprId) -> Option<TriProof>,
{
    if depth == 0 {
        return TriProof::Unknown;
    }

    prove_nonzero_depth_inner(
        ctx,
        expr,
        depth,
        &mut prove_positive,
        &mut try_ground_nonzero,
    )
}

fn prove_nonzero_depth_inner<FProvePositive, FGroundFallback>(
    ctx: &Context,
    expr: ExprId,
    depth: usize,
    prove_positive: &mut FProvePositive,
    try_ground_nonzero: &mut FGroundFallback,
) -> TriProof
where
    FProvePositive: FnMut(&Context, ExprId) -> TriProof,
    FGroundFallback: FnMut(&Context, ExprId) -> Option<TriProof>,
{
    if depth == 0 {
        return TriProof::Unknown;
    }

    match ctx.get(expr) {
        Expr::Number(n) => {
            if n.is_zero() {
                TriProof::Disproven
            } else {
                TriProof::Proven
            }
        }
        Expr::Constant(c) => {
            if matches!(
                c,
                cas_ast::Constant::Pi | cas_ast::Constant::E | cas_ast::Constant::I
            ) {
                TriProof::Proven
            } else {
                TriProof::Unknown
            }
        }
        Expr::Neg(a) | Expr::Hold(a) => {
            prove_nonzero_depth_inner(ctx, *a, depth - 1, prove_positive, try_ground_nonzero)
        }
        Expr::Mul(a, b) => {
            let proof_a =
                prove_nonzero_depth_inner(ctx, *a, depth - 1, prove_positive, try_ground_nonzero);
            let proof_b =
                prove_nonzero_depth_inner(ctx, *b, depth - 1, prove_positive, try_ground_nonzero);
            match (proof_a, proof_b) {
                (TriProof::Disproven, _) | (_, TriProof::Disproven) => TriProof::Disproven,
                (TriProof::Proven, TriProof::Proven) => TriProof::Proven,
                _ => TriProof::Unknown,
            }
        }
        Expr::Pow(base, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                let zero = num_rational::BigRational::zero();
                if n.is_integer() && *n > zero {
                    return prove_nonzero_depth_inner(
                        ctx,
                        *base,
                        depth - 1,
                        prove_positive,
                        try_ground_nonzero,
                    );
                }
                if *n != zero {
                    if prove_positive(ctx, *base).is_proven() {
                        return TriProof::Proven;
                    }
                    if n.is_integer() {
                        let base_nz = prove_nonzero_depth_inner(
                            ctx,
                            *base,
                            depth - 1,
                            prove_positive,
                            try_ground_nonzero,
                        );
                        if base_nz.is_proven() {
                            return TriProof::Proven;
                        }
                    }
                }
            }

            let base_pos = prove_positive(ctx, *base);
            let exp_nz =
                prove_nonzero_depth_inner(ctx, *exp, depth - 1, prove_positive, try_ground_nonzero);
            if base_pos.is_proven() && !exp_nz.is_disproven() {
                return TriProof::Proven;
            }
            TriProof::Unknown
        }
        Expr::Div(a, b) => {
            let proof_a =
                prove_nonzero_depth_inner(ctx, *a, depth - 1, prove_positive, try_ground_nonzero);
            let proof_b =
                prove_nonzero_depth_inner(ctx, *b, depth - 1, prove_positive, try_ground_nonzero);
            match (proof_a, proof_b) {
                (TriProof::Disproven, _) => TriProof::Disproven,
                (TriProof::Proven, TriProof::Proven) => TriProof::Proven,
                _ => TriProof::Unknown,
            }
        }
        Expr::Function(_, _) if extract_unary_log_argument_view(ctx, expr).is_some() => {
            let Some(arg) = extract_unary_log_argument_view(ctx, expr) else {
                return TriProof::Unknown;
            };
            match ctx.get(arg) {
                Expr::Number(n) => {
                    let one = num_rational::BigRational::one();
                    let zero = num_rational::BigRational::zero();
                    if *n > zero && *n != one {
                        TriProof::Proven
                    } else if *n == one {
                        TriProof::Disproven
                    } else {
                        TriProof::Unknown
                    }
                }
                Expr::Div(num, denom) => match (ctx.get(*num), ctx.get(*denom)) {
                    (Expr::Number(n), Expr::Number(d)) => {
                        let zero = num_rational::BigRational::zero();
                        if *n > zero && *d > zero && n != d {
                            TriProof::Proven
                        } else if n == d {
                            TriProof::Disproven
                        } else {
                            TriProof::Unknown
                        }
                    }
                    _ => TriProof::Unknown,
                },
                Expr::Constant(c) => {
                    if matches!(c, cas_ast::Constant::Pi | cas_ast::Constant::E) {
                        TriProof::Proven
                    } else {
                        TriProof::Unknown
                    }
                }
                _ => TriProof::Unknown,
            }
        }
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Sin) && args.len() == 1 =>
        {
            if let Some(k) = extract_rational_pi_multiple(ctx, args[0]) {
                if k.is_integer() {
                    TriProof::Disproven
                } else {
                    TriProof::Proven
                }
            } else {
                TriProof::Unknown
            }
        }
        Expr::Add(_, _) | Expr::Sub(_, _) => {
            if !contains_variable(ctx, expr) {
                if let Some(proof) = try_ground_nonzero(ctx, expr) {
                    return proof;
                }
            }
            TriProof::Unknown
        }
        Expr::Function(_, _) => {
            if !contains_variable(ctx, expr) {
                if let Some(proof) = try_ground_nonzero(ctx, expr) {
                    return proof;
                }
            }
            TriProof::Unknown
        }
        _ => TriProof::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::prove_nonzero_depth_with;
    use crate::tri_proof::TriProof;
    use cas_parser::parse;

    #[test]
    fn proves_numbers_nonzero_status() {
        let mut ctx = cas_ast::Context::new();
        let zero = parse("0", &mut ctx).expect("parse");
        let two = parse("2", &mut ctx).expect("parse");

        let prove_positive = |_ctx: &cas_ast::Context, _expr| TriProof::Unknown;
        let ground = |_ctx: &cas_ast::Context, _expr| None;

        assert_eq!(
            prove_nonzero_depth_with(&ctx, zero, 10, prove_positive, ground),
            TriProof::Disproven
        );
        assert_eq!(
            prove_nonzero_depth_with(&ctx, two, 10, prove_positive, ground),
            TriProof::Proven
        );
    }

    #[test]
    fn proves_sin_rational_pi_multiple() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("sin(pi/9)", &mut ctx).expect("parse");

        let proof = prove_nonzero_depth_with(
            &ctx,
            expr,
            20,
            |_ctx, _expr| TriProof::Unknown,
            |_ctx, _expr| None,
        );
        assert_eq!(proof, TriProof::Proven);
    }

    #[test]
    fn uses_ground_fallback_for_variable_free_addition() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("1+2", &mut ctx).expect("parse");

        let proof = prove_nonzero_depth_with(
            &ctx,
            expr,
            20,
            |_ctx, _expr| TriProof::Unknown,
            |_ctx, _expr| Some(TriProof::Proven),
        );
        assert_eq!(proof, TriProof::Proven);
    }

    #[test]
    fn pow_uses_positive_base_callback() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("a^(1/2)", &mut ctx).expect("parse");
        let a = parse("a", &mut ctx).expect("parse");

        let proof = prove_nonzero_depth_with(
            &ctx,
            expr,
            20,
            |_ctx, e| {
                if e == a {
                    TriProof::Proven
                } else {
                    TriProof::Unknown
                }
            },
            |_ctx, _expr| None,
        );
        assert_eq!(proof, TriProof::Proven);
    }
}
