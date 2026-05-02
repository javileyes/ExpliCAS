//! Generic sign-proof helpers (`> 0` and `>= 0`) shared by runtime crates.
//!
//! Runtime crates provide a non-zero prover callback so this module stays
//! independent from runtime domain/predicate stacks.

use crate::expr_extract::{extract_abs_argument_view, extract_sqrt_argument_view};
use crate::expr_predicates::is_zero_expr as is_zero;
use crate::polynomial::Polynomial;
use crate::tri_proof::TriProof;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

fn univariate_quadratic_shape(ctx: &Context, expr: ExprId) -> Option<(BigRational, BigRational)> {
    let vars = cas_ast::collect_variables(ctx, expr);
    if vars.len() != 1 {
        return None;
    }

    let var = vars.iter().next()?;
    let poly = Polynomial::from_expr(ctx, expr, var).ok()?;
    if poly.degree() != 2 || poly.coeffs.len() < 3 {
        return None;
    }

    let a = poly.coeffs.get(2)?.clone();
    if a.is_zero() {
        return None;
    }

    let b = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let c = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let four = BigRational::from_integer(4.into());
    let discriminant = b.clone() * b - four * a.clone() * c;

    Some((a, discriminant))
}

fn is_strictly_positive_univariate_quadratic(ctx: &Context, expr: ExprId) -> bool {
    univariate_quadratic_shape(ctx, expr)
        .is_some_and(|(a, discriminant)| a.is_positive() && discriminant.is_negative())
}

fn is_nonnegative_univariate_quadratic(ctx: &Context, expr: ExprId) -> bool {
    univariate_quadratic_shape(ctx, expr)
        .is_some_and(|(a, discriminant)| a.is_positive() && !discriminant.is_positive())
}

fn square_coeff(coeffs: &[BigRational], degree: usize) -> BigRational {
    let mut out = BigRational::zero();
    for i in 0..=degree {
        let Some(left) = coeffs.get(i) else {
            continue;
        };
        let Some(right) = coeffs.get(degree - i) else {
            continue;
        };
        out += left * right;
    }
    out
}

fn scaled_monic_square_root_and_constant_residual(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, Polynomial, BigRational)> {
    let vars = cas_ast::collect_variables(ctx, expr);
    if vars.len() != 1 {
        return None;
    }

    let var = vars.iter().next()?;
    let poly = Polynomial::from_expr(ctx, expr, var).ok()?;
    let degree = poly.degree();
    let scale = poly.leading_coeff();
    if degree < 2 || degree % 2 != 0 || !scale.is_positive() {
        return None;
    }

    let half = degree / 2;
    let mut root_coeffs = vec![BigRational::zero(); half + 1];
    root_coeffs[half] = BigRational::one();
    let two = BigRational::from_integer(2.into());

    for root_degree in (0..half).rev() {
        let target_degree = half + root_degree;
        let target_coeff = poly
            .coeffs
            .get(target_degree)
            .cloned()
            .unwrap_or_else(BigRational::zero)
            / scale.clone();
        let known = square_coeff(&root_coeffs, target_degree);
        root_coeffs[root_degree] = (target_coeff - known) / two.clone();
    }

    let root = Polynomial::new(root_coeffs, var.clone());
    let scaled_square = Polynomial::new(
        root.mul(&root)
            .coeffs
            .into_iter()
            .map(|coeff| coeff * scale.clone())
            .collect(),
        var.clone(),
    );
    let residual = poly.sub(&scaled_square);
    if residual.is_zero() {
        return Some((scale, root, BigRational::zero()));
    }
    if residual.degree() != 0 {
        return None;
    }

    residual
        .coeffs
        .first()
        .cloned()
        .map(|constant| (scale, root, constant))
}

fn positive_quadratic_minimum(poly: &Polynomial) -> Option<BigRational> {
    if poly.degree() != 2 || poly.coeffs.len() < 3 {
        return None;
    }

    let a = poly.coeffs.get(2)?.clone();
    if !a.is_positive() {
        return None;
    }

    let b = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let c = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let four = BigRational::from_integer(4.into());
    let minimum = c - (b.clone() * b) / (four * a);
    minimum.is_positive().then_some(minimum)
}

fn is_strictly_positive_monic_square_with_constant_offset(ctx: &Context, expr: ExprId) -> bool {
    let Some((scale, root, constant)) = scaled_monic_square_root_and_constant_residual(ctx, expr)
    else {
        return false;
    };

    if constant.is_positive() {
        return true;
    }

    if !constant.is_negative() {
        return false;
    }

    positive_quadratic_minimum(&root)
        .map(|minimum| scale * minimum.clone() * minimum + constant)
        .is_some_and(|lower_bound| lower_bound.is_positive())
}

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
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Hold) && args.len() == 1 =>
        {
            prove_positive_depth_inner(ctx, args[0], depth - 1, real_only, prove_nonzero)
        }
        Expr::Add(_, _) | Expr::Sub(_, _)
            if real_only && is_strictly_positive_univariate_quadratic(ctx, expr) =>
        {
            TriProof::Proven
        }
        Expr::Add(_, _) | Expr::Sub(_, _)
            if real_only && is_strictly_positive_monic_square_with_constant_offset(ctx, expr) =>
        {
            TriProof::Proven
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
        Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, BuiltinFn::Ln) && args.len() == 1 => {
            let Some(n) = crate::numeric_eval::as_rational_const(ctx, args[0]) else {
                return TriProof::Unknown;
            };
            let zero = num_rational::BigRational::from_integer(0.into());
            let one = num_rational::BigRational::from_integer(1.into());
            if n > one {
                TriProof::Proven
            } else if n > zero {
                TriProof::Disproven
            } else {
                TriProof::Unknown
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
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Hold) && args.len() == 1 =>
        {
            prove_nonnegative_depth_inner(ctx, args[0], depth - 1, real_only, prove_nonzero)
        }
        Expr::Add(_, _) | Expr::Sub(_, _)
            if real_only && is_nonnegative_univariate_quadratic(ctx, expr) =>
        {
            TriProof::Proven
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
        Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, BuiltinFn::Ln) && args.len() == 1 => {
            let Some(n) = crate::numeric_eval::as_rational_const(ctx, args[0]) else {
                return TriProof::Unknown;
            };
            let zero = num_rational::BigRational::from_integer(0.into());
            let one = num_rational::BigRational::from_integer(1.into());
            if n >= one {
                TriProof::Proven
            } else if n > zero {
                TriProof::Disproven
            } else {
                TriProof::Unknown
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
    fn internal_hold_is_transparent_to_sign_proofs() {
        let mut ctx = cas_ast::Context::new();
        let positive = parse("(2*x+1)^2 + 1", &mut ctx).expect("parse");
        let held_positive = cas_ast::hold::wrap_hold(&mut ctx, positive);
        let out =
            prove_positive_depth_with(&ctx, held_positive, 20, true, |_ctx, _expr, _depth| {
                TriProof::Unknown
            });
        assert_eq!(out, TriProof::Proven);

        let nonnegative = parse("(x+1)^2", &mut ctx).expect("parse");
        let held_nonnegative = cas_ast::hold::wrap_hold(&mut ctx, nonnegative);
        let out = prove_nonnegative_depth_with(
            &ctx,
            held_nonnegative,
            20,
            true,
            |_ctx, _expr, _depth| TriProof::Unknown,
        );
        assert_eq!(out, TriProof::Proven);
    }

    #[test]
    fn positive_proves_positive_definite_quadratic() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("2*x^2 + 2*x + 1", &mut ctx).expect("parse");
        let out = prove_positive_depth_with(&ctx, expr, 20, true, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Proven);
    }

    #[test]
    fn positive_does_not_prove_perfect_square_quadratic() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("x^2 + 2*x + 1", &mut ctx).expect("parse");
        let out = prove_positive_depth_with(&ctx, expr, 20, true, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Unknown);
    }

    #[test]
    fn positive_proves_expanded_monic_square_plus_constant() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("x^4 + 2*x^3 + 3*x^2 + 2*x + 8", &mut ctx).expect("parse");
        let out = prove_positive_depth_with(&ctx, expr, 20, true, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Proven);
    }

    #[test]
    fn positive_proves_expanded_positive_quadratic_square_minus_small_constant() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("x^4 + 2*x^3 + 7*x^2 + 6*x + 7", &mut ctx).expect("parse");
        let out = prove_positive_depth_with(&ctx, expr, 20, true, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Proven);
    }

    #[test]
    fn positive_proves_expanded_scaled_positive_quadratic_square_minus_small_constant() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("2*x^4 + 4*x^3 + 14*x^2 + 12*x + 17", &mut ctx).expect("parse");
        let out = prove_positive_depth_with(&ctx, expr, 20, true, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Proven);
    }

    #[test]
    fn positive_does_not_prove_expanded_monic_perfect_square_without_offset() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("x^4 + 2*x^3 + 3*x^2 + 2*x + 1", &mut ctx).expect("parse");
        let out = prove_positive_depth_with(&ctx, expr, 20, true, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Unknown);
    }

    #[test]
    fn positive_does_not_prove_expanded_quadratic_square_minus_large_constant() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("x^4 + 2*x^3 + 7*x^2 + 6*x - 1", &mut ctx).expect("parse");
        let out = prove_positive_depth_with(&ctx, expr, 20, true, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Unknown);
    }

    #[test]
    fn positive_does_not_prove_expanded_scaled_quadratic_square_minus_large_constant() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("2*x^4 + 4*x^3 + 14*x^2 + 12*x", &mut ctx).expect("parse");
        let out = prove_positive_depth_with(&ctx, expr, 20, true, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Unknown);
    }

    #[test]
    fn nonnegative_proves_expanded_perfect_square_quadratic() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("x^2 + 2*x + 1", &mut ctx).expect("parse");
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

    #[test]
    fn proves_ln_of_rational_constant_sign() {
        let mut ctx = cas_ast::Context::new();
        let ln_two = parse("ln(2)", &mut ctx).expect("parse");
        let ln_half = parse("ln(1/2)", &mut ctx).expect("parse");
        let ln_one = parse("ln(1)", &mut ctx).expect("parse");

        let positive_two =
            prove_positive_depth_with(&ctx, ln_two, 20, true, |_ctx, _expr, _depth| {
                TriProof::Unknown
            });
        let positive_half =
            prove_positive_depth_with(&ctx, ln_half, 20, true, |_ctx, _expr, _depth| {
                TriProof::Unknown
            });
        let nonnegative_one =
            prove_nonnegative_depth_with(&ctx, ln_one, 20, true, |_ctx, _expr, _depth| {
                TriProof::Unknown
            });

        assert_eq!(positive_two, TriProof::Proven);
        assert_eq!(positive_half, TriProof::Disproven);
        assert_eq!(nonnegative_one, TriProof::Proven);
    }
}
