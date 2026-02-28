//! Generic sign-proof helpers (`> 0` and `>= 0`) shared by runtime crates.
//!
//! Runtime crates provide a non-zero prover callback so this module stays
//! independent from runtime domain/predicate stacks.

use crate::expr_extract::{extract_abs_argument_view, extract_sqrt_argument_view};
use crate::expr_predicates::is_zero_expr as is_zero;
use crate::tri_proof::TriProof;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_traits::Signed;

/// Prove whether an expression is strictly positive (`> 0`).
///
/// `real_only = true` models a real-only value domain. `false` models a
/// complex-enabled domain where positivity is only provable in fewer cases.
pub fn prove_positive_depth_with<FProveNonzero>(
    ctx: &Context,
    expr: ExprId,
    depth: usize,
    real_only: bool,
    mut prove_nonzero: FProveNonzero,
) -> TriProof
where
    FProveNonzero: FnMut(&Context, ExprId, usize) -> TriProof,
{
    prove_positive_depth_inner(ctx, expr, depth, real_only, &mut prove_nonzero)
}

/// Prove whether an expression is non-negative (`>= 0`).
pub fn prove_nonnegative_depth_with<FProveNonzero>(
    ctx: &Context,
    expr: ExprId,
    depth: usize,
    real_only: bool,
    mut prove_nonzero: FProveNonzero,
) -> TriProof
where
    FProveNonzero: FnMut(&Context, ExprId, usize) -> TriProof,
{
    prove_nonnegative_depth_inner(ctx, expr, depth, real_only, &mut prove_nonzero)
}

fn prove_positive_depth_inner<FProveNonzero>(
    ctx: &Context,
    expr: ExprId,
    depth: usize,
    real_only: bool,
    prove_nonzero: &mut FProveNonzero,
) -> TriProof
where
    FProveNonzero: FnMut(&Context, ExprId, usize) -> TriProof,
{
    use num_traits::Zero;

    if depth == 0 {
        return TriProof::Unknown;
    }

    match ctx.get(expr) {
        Expr::Number(n) => {
            if *n > num_rational::BigRational::zero() {
                TriProof::Proven
            } else {
                TriProof::Disproven
            }
        }
        Expr::Constant(c) => {
            if matches!(c, cas_ast::Constant::Pi | cas_ast::Constant::E) {
                TriProof::Proven
            } else {
                TriProof::Unknown
            }
        }
        Expr::Add(a, b) => {
            let proof_a_pos =
                prove_positive_depth_inner(ctx, *a, depth - 1, real_only, prove_nonzero);
            let proof_b_pos =
                prove_positive_depth_inner(ctx, *b, depth - 1, real_only, prove_nonzero);

            if proof_a_pos == TriProof::Proven && proof_b_pos == TriProof::Proven {
                return TriProof::Proven;
            }

            if proof_a_pos == TriProof::Proven {
                let b_nonneg =
                    prove_nonnegative_depth_inner(ctx, *b, depth - 1, real_only, prove_nonzero);
                if b_nonneg == TriProof::Proven {
                    return TriProof::Proven;
                }
            }
            if proof_b_pos == TriProof::Proven {
                let a_nonneg =
                    prove_nonnegative_depth_inner(ctx, *a, depth - 1, real_only, prove_nonzero);
                if a_nonneg == TriProof::Proven {
                    return TriProof::Proven;
                }
            }

            TriProof::Unknown
        }
        Expr::Mul(a, b) => {
            if is_zero(ctx, *a) || is_zero(ctx, *b) {
                return TriProof::Disproven;
            }

            let proof_a = prove_positive_depth_inner(ctx, *a, depth - 1, real_only, prove_nonzero);
            let proof_b = prove_positive_depth_inner(ctx, *b, depth - 1, real_only, prove_nonzero);

            match (proof_a, proof_b) {
                (TriProof::Proven, TriProof::Proven) => TriProof::Proven,
                _ => TriProof::Unknown,
            }
        }
        Expr::Div(a, b) => {
            let proof_a = prove_positive_depth_inner(ctx, *a, depth - 1, real_only, prove_nonzero);
            let proof_b = prove_positive_depth_inner(ctx, *b, depth - 1, real_only, prove_nonzero);

            match (proof_a, proof_b) {
                (TriProof::Proven, TriProof::Proven) => TriProof::Proven,
                _ => TriProof::Unknown,
            }
        }
        Expr::Pow(base, exp) => {
            let base_positive =
                prove_positive_depth_inner(ctx, *base, depth - 1, real_only, prove_nonzero);

            if real_only {
                if base_positive == TriProof::Proven {
                    return TriProof::Proven;
                }
            } else {
                let exp_is_real_numeric = matches!(ctx.get(*exp), Expr::Number(_));
                if base_positive == TriProof::Proven && exp_is_real_numeric {
                    return TriProof::Proven;
                }
            }

            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() {
                    let int_val = n.to_integer();
                    let two: num_bigint::BigInt = 2.into();
                    if &int_val % &two == 0.into() {
                        let base_nonzero = prove_nonzero(ctx, *base, depth - 1);
                        if base_nonzero == TriProof::Proven {
                            return TriProof::Proven;
                        }
                    }
                }
            }
            TriProof::Unknown
        }
        Expr::Function(_, _) if extract_abs_argument_view(ctx, expr).is_some() => {
            let Some(arg) = extract_abs_argument_view(ctx, expr) else {
                return TriProof::Unknown;
            };
            let inner_nonzero = prove_nonzero(ctx, arg, depth - 1);
            if inner_nonzero == TriProof::Proven {
                TriProof::Proven
            } else if inner_nonzero == TriProof::Disproven {
                TriProof::Disproven
            } else {
                TriProof::Unknown
            }
        }
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Exp) && args.len() == 1 =>
        {
            if real_only {
                TriProof::Proven
            } else {
                match ctx.get(args[0]) {
                    Expr::Number(_)
                    | Expr::Constant(cas_ast::Constant::Pi)
                    | Expr::Constant(cas_ast::Constant::E) => TriProof::Proven,
                    _ => TriProof::Unknown,
                }
            }
        }
        Expr::Function(_, _) if extract_sqrt_argument_view(ctx, expr).is_some() => {
            let Some(arg) = extract_sqrt_argument_view(ctx, expr) else {
                return TriProof::Unknown;
            };
            prove_positive_depth_inner(ctx, arg, depth - 1, real_only, prove_nonzero)
        }
        Expr::Neg(inner) => {
            let inner_proof =
                prove_positive_depth_inner(ctx, *inner, depth - 1, real_only, prove_nonzero);
            match inner_proof {
                TriProof::Proven => TriProof::Disproven,
                TriProof::Disproven => {
                    if let Expr::Number(n) = ctx.get(*inner) {
                        if n.is_negative() {
                            return TriProof::Proven;
                        }
                    }
                    TriProof::Unknown
                }
                _ => TriProof::Unknown,
            }
        }
        Expr::Hold(inner) => {
            prove_positive_depth_inner(ctx, *inner, depth - 1, real_only, prove_nonzero)
        }
        _ => TriProof::Unknown,
    }
}

fn prove_nonnegative_depth_inner<FProveNonzero>(
    ctx: &Context,
    expr: ExprId,
    depth: usize,
    real_only: bool,
    prove_nonzero: &mut FProveNonzero,
) -> TriProof
where
    FProveNonzero: FnMut(&Context, ExprId, usize) -> TriProof,
{
    use num_traits::Zero;

    if depth == 0 {
        return TriProof::Unknown;
    }

    match ctx.get(expr) {
        Expr::Number(n) => {
            if *n >= num_rational::BigRational::zero() {
                TriProof::Proven
            } else {
                TriProof::Disproven
            }
        }
        Expr::Constant(c) => {
            if matches!(c, cas_ast::Constant::Pi | cas_ast::Constant::E) {
                TriProof::Proven
            } else {
                TriProof::Unknown
            }
        }
        Expr::Pow(base, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() {
                    let int_val = n.to_integer();
                    let two: num_bigint::BigInt = 2.into();
                    if &int_val % &two == 0.into() && int_val > 0.into() {
                        return TriProof::Proven;
                    }
                }
            }

            if real_only {
                let base_positive =
                    prove_positive_depth_inner(ctx, *base, depth - 1, real_only, prove_nonzero);
                if base_positive == TriProof::Proven {
                    return TriProof::Proven;
                }
            }

            TriProof::Unknown
        }
        Expr::Function(_, _) if extract_abs_argument_view(ctx, expr).is_some() => TriProof::Proven,
        Expr::Function(_, _) if extract_sqrt_argument_view(ctx, expr).is_some() => {
            let Some(arg) = extract_sqrt_argument_view(ctx, expr) else {
                return TriProof::Unknown;
            };
            prove_nonnegative_depth_inner(ctx, arg, depth - 1, real_only, prove_nonzero)
        }
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Exp) && args.len() == 1 =>
        {
            if real_only {
                TriProof::Proven
            } else {
                match ctx.get(args[0]) {
                    Expr::Number(_)
                    | Expr::Constant(cas_ast::Constant::Pi)
                    | Expr::Constant(cas_ast::Constant::E) => TriProof::Proven,
                    _ => TriProof::Unknown,
                }
            }
        }
        Expr::Mul(a, b) => {
            let proof_a =
                prove_nonnegative_depth_inner(ctx, *a, depth - 1, real_only, prove_nonzero);
            let proof_b =
                prove_nonnegative_depth_inner(ctx, *b, depth - 1, real_only, prove_nonzero);

            match (proof_a, proof_b) {
                (TriProof::Proven, TriProof::Proven) => TriProof::Proven,
                _ => TriProof::Unknown,
            }
        }
        Expr::Add(a, b) => {
            let proof_a =
                prove_nonnegative_depth_inner(ctx, *a, depth - 1, real_only, prove_nonzero);
            let proof_b =
                prove_nonnegative_depth_inner(ctx, *b, depth - 1, real_only, prove_nonzero);

            match (proof_a, proof_b) {
                (TriProof::Proven, TriProof::Proven) => TriProof::Proven,
                _ => TriProof::Unknown,
            }
        }
        Expr::Div(a, b) => {
            let proof_a =
                prove_nonnegative_depth_inner(ctx, *a, depth - 1, real_only, prove_nonzero);
            let proof_b = prove_positive_depth_inner(ctx, *b, depth - 1, real_only, prove_nonzero);

            match (proof_a, proof_b) {
                (TriProof::Proven, TriProof::Proven) => TriProof::Proven,
                _ => TriProof::Unknown,
            }
        }
        Expr::Neg(inner) => {
            let inner_proof =
                prove_nonnegative_depth_inner(ctx, *inner, depth - 1, real_only, prove_nonzero);
            match inner_proof {
                TriProof::Disproven => TriProof::Proven,
                _ => TriProof::Unknown,
            }
        }
        Expr::Hold(inner) => {
            prove_nonnegative_depth_inner(ctx, *inner, depth - 1, real_only, prove_nonzero)
        }
        _ => TriProof::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::{prove_nonnegative_depth_with, prove_positive_depth_with};
    use crate::tri_proof::TriProof;
    use cas_parser::parse;

    #[test]
    fn positive_proves_numeric_literal() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("2", &mut ctx).expect("parse");
        let out = prove_positive_depth_with(&ctx, expr, 20, true, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Proven);
    }

    #[test]
    fn nonnegative_proves_even_power() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("x^2", &mut ctx).expect("parse");
        let out = prove_nonnegative_depth_with(&ctx, expr, 20, true, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Proven);
    }

    #[test]
    fn positive_abs_uses_nonzero_callback() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("abs(x)", &mut ctx).expect("parse");
        let x = parse("x", &mut ctx).expect("parse");
        let out = prove_positive_depth_with(&ctx, expr, 20, true, |_ctx, candidate, _depth| {
            if candidate == x {
                TriProof::Proven
            } else {
                TriProof::Unknown
            }
        });
        assert_eq!(out, TriProof::Proven);
    }

    #[test]
    fn complex_domain_exp_symbolic_is_unknown() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("exp(x)", &mut ctx).expect("parse");
        let out = prove_positive_depth_with(&ctx, expr, 20, false, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Unknown);
    }
}
