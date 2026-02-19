//! Pi-specific structural helpers.

use cas_ast::{Context, Expr, ExprId};
use num_traits::{One, Zero};

/// Check if expression equals π/n for a given denominator (handles both Div and Mul forms)
pub fn is_pi_over_n(ctx: &Context, expr: ExprId, denom: i32) -> bool {
    // Handle Div form: pi/n
    if let Expr::Div(num, den) = ctx.get(expr) {
        if let Expr::Constant(c) = ctx.get(*num) {
            if matches!(c, cas_ast::Constant::Pi) {
                if let Expr::Number(n) = ctx.get(*den) {
                    return *n == num_rational::BigRational::from_integer(denom.into());
                }
            }
        }
    }

    // Handle Mul form: (1/n) * pi
    if let Expr::Mul(l, r) = ctx.get(expr) {
        let (num_part, const_part) = if let Expr::Constant(_) = ctx.get(*l) {
            (*r, *l)
        } else if let Expr::Constant(_) = ctx.get(*r) {
            (*l, *r)
        } else {
            return false;
        };

        if let Expr::Constant(c) = ctx.get(const_part) {
            if matches!(c, cas_ast::Constant::Pi) {
                if let Expr::Number(n) = ctx.get(num_part) {
                    return *n == num_rational::BigRational::new(1.into(), denom.into());
                }
            }
        }
    }

    false
}

/// Check if expression is exactly π
pub fn is_pi(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Constant(cas_ast::Constant::Pi))
}

/// Build π/n expression
pub fn build_pi_over_n(ctx: &mut Context, denom: i64) -> ExprId {
    let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
    let d = ctx.num(denom);
    ctx.add(Expr::Div(pi, d))
}

/// Extract the rational coefficient k from an expression θ = k·π.
///
/// Matches patterns:
/// - `π` → k = 1
/// - `π/n` → k = 1/n
/// - `m*π/n` or `(m/n)*π` → k = m/n
/// - `k*π` where k is numeric → k
///
/// Returns None if not a recognizable rational multiple of π.
pub fn extract_rational_pi_multiple(
    ctx: &Context,
    expr: ExprId,
) -> Option<num_rational::BigRational> {
    use num_bigint::BigInt;
    use num_rational::BigRational;

    // Pattern 1: Just π → k = 1
    if matches!(ctx.get(expr), Expr::Constant(cas_ast::Constant::Pi)) {
        return Some(BigRational::one());
    }

    // Pattern 2: Div(π, n) → k = 1/n
    if let Expr::Div(num, denom) = ctx.get(expr) {
        if matches!(ctx.get(*num), Expr::Constant(cas_ast::Constant::Pi)) {
            if let Expr::Number(d) = ctx.get(*denom) {
                if d.is_integer() && !d.is_zero() {
                    return Some(BigRational::new(BigInt::from(1), d.to_integer()));
                }
            }
        }
        // Pattern 2b: Div(m*π, n) → k = m/n
        if let Expr::Mul(l, r) = ctx.get(*num) {
            let (coef, pi_id) = if matches!(ctx.get(*l), Expr::Constant(cas_ast::Constant::Pi)) {
                (*r, *l)
            } else if matches!(ctx.get(*r), Expr::Constant(cas_ast::Constant::Pi)) {
                (*l, *r)
            } else {
                return None;
            };
            if matches!(ctx.get(pi_id), Expr::Constant(cas_ast::Constant::Pi)) {
                if let (Expr::Number(m), Expr::Number(n)) = (ctx.get(coef), ctx.get(*denom)) {
                    if m.is_integer() && n.is_integer() && !n.is_zero() {
                        return Some(BigRational::new(m.to_integer(), n.to_integer()));
                    }
                }
            }
        }
    }

    // Pattern 3: Mul(k, π) or Mul(π, k) → k
    if let Expr::Mul(l, r) = ctx.get(expr) {
        let (coef_id, pi_id) = if matches!(ctx.get(*l), Expr::Constant(cas_ast::Constant::Pi)) {
            (*r, *l)
        } else if matches!(ctx.get(*r), Expr::Constant(cas_ast::Constant::Pi)) {
            (*l, *r)
        } else {
            return None;
        };

        if !matches!(ctx.get(pi_id), Expr::Constant(cas_ast::Constant::Pi)) {
            return None;
        }

        // coef_id should be a numeric
        if let Expr::Number(k) = ctx.get(coef_id) {
            return Some(k.clone());
        }

        // Or coef_id is Div(m, n)
        if let Expr::Div(m_id, n_id) = ctx.get(coef_id) {
            if let (Expr::Number(m), Expr::Number(n)) = (ctx.get(*m_id), ctx.get(*n_id)) {
                if m.is_integer() && n.is_integer() && !n.is_zero() {
                    return Some(BigRational::new(m.to_integer(), n.to_integer()));
                }
            }
        }
    }

    None
}

/// Check if sin(θ) is provably non-zero.
///
/// Uses the identity: sin(k·π) = 0 iff k is an integer.
/// So if θ = k·π with k ∈ ℚ and k is NOT an integer, then sin(θ) ≠ 0.
///
/// Returns true only if we can prove sin(θ) ≠ 0. Returns false otherwise
/// (including cases where we cannot determine).
pub fn is_provably_sin_nonzero(ctx: &Context, theta: ExprId) -> bool {
    if let Some(k) = extract_rational_pi_multiple(ctx, theta) {
        // sin(k·π) = 0 iff k is an integer
        // So sin(k·π) ≠ 0 iff k is NOT an integer
        !k.is_integer()
    } else {
        // Cannot extract k, so cannot prove
        false
    }
}

/// Check if expression equals 1/2
pub fn is_half(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        *n.numer() == 1.into() && *n.denom() == 2.into()
    } else {
        false
    }
}
