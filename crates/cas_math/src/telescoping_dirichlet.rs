use crate::expr_extract::extract_i64_multiplier_and_base_factors;
use crate::expr_nary::{AddView, Sign};
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::cmp::Ordering;

/// Result of Dirichlet kernel detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirichletKernelResult {
    pub n: usize,         // The n in the sum (highest cosine multiple).
    pub base_var: ExprId, // The base variable x.
    pub base_multiplier: usize,
}

/// Try to detect Dirichlet kernel identity pattern:
/// 1 + 2*cos(x) + 2*cos(2x) + ... + 2*cos(nx) - sin((n+1/2)x)/sin(x/2) = 0
pub fn try_dirichlet_kernel_identity(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<DirichletKernelResult> {
    let view = AddView::from_expr(ctx, expr);

    let mut has_one = false;
    let mut cosine_multiples: Vec<(usize, Vec<ExprId>)> = Vec::new(); // (k, base factors)
    let mut sin_ratio: Option<(ExprId, ExprId)> = None; // (numerator arg, denominator arg)
    let mut sin_ratio_is_negative = false;

    for &(term, sign) in &view.terms {
        let is_positive = sign == Sign::Pos;
        let term_data = ctx.get(term).clone();

        // Check for constant 1.
        if let Expr::Number(n) = &term_data {
            if n.is_one() && is_positive {
                has_one = true;
                continue;
            }
        }

        // Check for 2*cos(k*x).
        if let Some((k, base)) = extract_cosine_multiple(ctx, term) {
            if is_positive {
                cosine_multiples.push((k, base));
            }
            continue;
        }

        // Check for sin(a)/sin(b) ratio.
        if let Some((num_arg, den_arg)) = extract_sin_ratio(ctx, term) {
            sin_ratio = Some((num_arg, den_arg));
            sin_ratio_is_negative = !is_positive; // Should be subtracted (negative).
        }
    }

    // Verify pattern: need 1, consecutive cosine multiples, and matching sin ratio.
    if !has_one || cosine_multiples.is_empty() {
        return None;
    }

    // Sort cosines by their multiple.
    cosine_multiples.sort_by_key(|(k, _)| *k);

    // Check if we have 1, 2, 3, ..., n.
    let n = cosine_multiples.len();
    let base_factors = cosine_multiples[0].1.clone();
    if base_factors.is_empty() {
        return None;
    }
    if cosine_multiples
        .iter()
        .any(|(_, base)| !same_factor_basis(ctx, &base_factors, base))
    {
        return None;
    }
    let base_var = crate::expr_nary::build_balanced_mul(ctx, &base_factors);

    let base_multiplier = cosine_multiples.iter().map(|(k, _)| *k).reduce(gcd_usize)?;
    if base_multiplier == 0 {
        return None;
    }

    for (i, (k, _)) in cosine_multiples.iter().enumerate() {
        if *k != (i + 1) * base_multiplier {
            return None;
        }
    }

    // Verify sin ratio matches expected form: sin((n+1/2)*x)/sin(x/2).
    if let Some((num_arg, den_arg)) = sin_ratio {
        if sin_ratio_is_negative
            && is_half_angle(ctx, den_arg, &base_factors, base_multiplier)
            && is_half_integer_multiple(ctx, num_arg, &base_factors, n, base_multiplier)
        {
            return Some(DirichletKernelResult {
                n,
                base_var,
                base_multiplier,
            });
        }
    }

    None
}

// nary-lint: allow-binary (structural pattern match for 2*cos(k*x))
fn extract_cosine_multiple(ctx: &Context, expr: ExprId) -> Option<(usize, Vec<ExprId>)> {
    // Pattern: 2 * cos(k * x) or cos(...) * 2
    if let Expr::Mul(l, r) = ctx.get(expr) {
        // Check if one side is 2.
        let (two_side, other_side) = if is_number(ctx, *l, 2) {
            (Some(*l), *r)
        } else if is_number(ctx, *r, 2) {
            (Some(*r), *l)
        } else {
            return None;
        };

        if two_side.is_some() {
            if let Expr::Function(fn_id, args) = ctx.get(other_side) {
                let name = ctx.sym_name(*fn_id);
                if name == "cos" && args.len() == 1 {
                    return extract_multiple_of_var(ctx, args[0]);
                }
            }
        }
    }
    None
}

/// Extract sin(a)/sin(b) pattern, returning (a, b).
fn extract_sin_ratio(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    if let Expr::Div(num, den) = ctx.get(expr) {
        if let Expr::Function(num_fn_id, num_args) = ctx.get(*num) {
            let num_name = ctx.sym_name(*num_fn_id);
            if num_name == "sin" && num_args.len() == 1 {
                if let Expr::Function(den_fn_id, den_args) = ctx.get(*den) {
                    let den_name = ctx.sym_name(*den_fn_id);
                    if den_name == "sin" && den_args.len() == 1 {
                        return Some((num_args[0], den_args[0]));
                    }
                }
            }
        }
    }
    None
}

// nary-lint: allow-binary (structural pattern match for half-angle)
fn is_half_angle(
    ctx: &Context,
    expr: ExprId,
    base_factors: &[ExprId],
    base_multiplier: usize,
) -> bool {
    let Some((coef, basis)) = extract_rational_multiple_of_basis(ctx, expr) else {
        return false;
    };
    same_factor_basis(ctx, &basis, base_factors)
        && coef == BigRational::new((base_multiplier as i64).into(), 2.into())
}

// nary-lint: allow-binary (structural pattern match for half-integer multiples)
fn is_half_integer_multiple(
    ctx: &Context,
    expr: ExprId,
    base_factors: &[ExprId],
    n: usize,
    base_multiplier: usize,
) -> bool {
    let expected_num = 2 * n + 1; // (n+1/2) = (2n+1)/2
    let Some((coef, basis)) = extract_rational_multiple_of_basis(ctx, expr) else {
        return false;
    };
    let expected_coef =
        BigRational::new(((expected_num * base_multiplier) as i64).into(), 2.into());
    same_factor_basis(ctx, &basis, base_factors) && coef == expected_coef
}

// nary-lint: allow-binary (structural pattern match for k*x extraction)
fn extract_multiple_of_var(ctx: &Context, expr: ExprId) -> Option<(usize, Vec<ExprId>)> {
    let (multiple, base_factors) = extract_i64_multiplier_and_base_factors(ctx, expr);
    if multiple <= 0 || base_factors.is_empty() {
        return None;
    }
    Some((multiple as usize, base_factors.into_vec()))
}

fn is_number(ctx: &Context, expr: ExprId, expected: i32) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        *n == num_rational::BigRational::from_integer(expected.into())
    } else {
        false
    }
}

fn gcd_usize(a: usize, b: usize) -> usize {
    let mut a = a;
    let mut b = b;
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn same_factor_basis(ctx: &Context, left: &[ExprId], right: &[ExprId]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(lhs, rhs)| compare_expr(ctx, *lhs, *rhs) == Ordering::Equal)
}

fn extract_rational_multiple_of_basis(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, Vec<ExprId>)> {
    match ctx.get(expr) {
        Expr::Variable(_) => Some((BigRational::from_integer(1.into()), vec![expr])),
        Expr::Mul(_, _) => extract_mul_chain_rational_multiple(ctx, expr),
        Expr::Div(num, den) => {
            let Expr::Number(den_val) = ctx.get(*den) else {
                return None;
            };
            if den_val.is_zero() {
                return None;
            }
            let (num_coef, basis) = extract_rational_multiple_of_basis(ctx, *num)?;
            Some((num_coef / den_val.clone(), basis))
        }
        Expr::Neg(inner) => {
            let (coef, basis) = extract_rational_multiple_of_basis(ctx, *inner)?;
            Some((-coef, basis))
        }
        Expr::Number(_) => None,
        _ => Some((BigRational::from_integer(1.into()), vec![expr])),
    }
}

fn extract_mul_chain_rational_multiple(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, Vec<ExprId>)> {
    let mut coef = BigRational::from_integer(1.into());
    let mut basis = Vec::new();

    for factor in crate::expr_nary::mul_leaves(ctx, expr) {
        if let Expr::Number(value) = ctx.get(factor) {
            coef *= value.clone();
        } else {
            basis.push(factor);
        }
    }

    if basis.is_empty() {
        return None;
    }
    basis.sort_by(|a, b| compare_expr(ctx, *a, *b));
    Some((coef, basis))
}

#[cfg(test)]
mod tests {
    use super::try_dirichlet_kernel_identity;
    use cas_ast::ordering::compare_expr;
    use cas_ast::Context;
    use cas_parser::parse;
    use std::cmp::Ordering;

    #[test]
    fn detects_basic_dirichlet_identity() {
        let mut ctx = Context::new();
        let expr =
            parse("1 + 2*cos(x) + 2*cos(2*x) - sin(5*x/2)/sin(x/2)", &mut ctx).expect("parse");
        let result = try_dirichlet_kernel_identity(&mut ctx, expr).expect("detect");
        assert_eq!(result.n, 2);
        assert_eq!(result.base_multiplier, 1);
    }

    #[test]
    fn detects_scaled_dirichlet_identity() {
        let mut ctx = Context::new();
        let expr = parse(
            "1 + 2*cos(3*x) + 2*cos(6*x) - sin(15*x/2)/sin(3*x/2)",
            &mut ctx,
        )
        .expect("parse");
        let result = try_dirichlet_kernel_identity(&mut ctx, expr).expect("detect");
        assert_eq!(result.n, 2);
        assert_eq!(result.base_multiplier, 3);
    }

    #[test]
    fn detects_symbolic_scale_dirichlet_identity() {
        let mut ctx = Context::new();
        let expr = parse(
            "1 + 2*cos(a*x) + 2*cos(2*a*x) - sin(5*a*x/2)/sin(a*x/2)",
            &mut ctx,
        )
        .expect("parse");
        let result = try_dirichlet_kernel_identity(&mut ctx, expr).expect("detect");
        assert_eq!(result.n, 2);
        assert_eq!(result.base_multiplier, 1);
        let expected = parse("a*x", &mut ctx).expect("parse expected");
        assert_eq!(
            compare_expr(&ctx, result.base_var, expected),
            Ordering::Equal
        );
    }
}
