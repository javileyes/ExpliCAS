pub mod isolation_strategy;
pub mod quadratic;
pub mod rational_roots;
pub mod substitution;

pub use isolation_strategy::{
    CollectTermsStrategy, IsolationStrategy, RationalExponentStrategy, UnwrapStrategy,
};
pub use quadratic::QuadraticStrategy;
pub use rational_roots::RationalRootsStrategy;
pub use substitution::SubstitutionStrategy;

use cas_ast::{Context, Expr, ExprId};
use num_traits::ToPrimitive;

// --- Helper Functions (Keep these as they are useful helpers) ---

/// Helper to pull perfect square numeric factors out of sqrt.
/// Converts sqrt(k * expr) → m * sqrt(expr) when k = m² is a perfect square.
/// Also handles sqrt(4a + 4b) → 2*sqrt(a + b) by factoring GCD.
/// This is semantically equivalent (no new assumptions).
fn pull_square_from_sqrt(ctx: &mut Context, sqrt_expr: ExprId) -> ExprId {
    // Check if expr is Pow(base, 1/2) - our representation of sqrt
    let expr_data = ctx.get(sqrt_expr).clone();
    let Expr::Pow(base, exp) = expr_data else {
        return sqrt_expr;
    };

    // Check if exponent is 1/2
    let is_half = matches!(ctx.get(exp), Expr::Div(n, d)
        if matches!(ctx.get(*n), Expr::Number(num) if num == &num_rational::BigRational::from_integer(1.into()))
        && matches!(ctx.get(*d), Expr::Number(den) if den == &num_rational::BigRational::from_integer(2.into()))
    );
    if !is_half {
        return sqrt_expr;
    }

    // Extract numeric factor from base: base = k * rest
    let (factor, rest) = split_numeric_factor(ctx, base);
    let Some(k) = factor else {
        return sqrt_expr;
    };

    // Check if k is a perfect square (only for positive integers)
    if k <= 0 {
        return sqrt_expr;
    }

    let sqrt_k = (k as f64).sqrt();
    let m = sqrt_k.round() as i64;
    if m * m != k {
        return sqrt_expr; // Not a perfect square
    }
    if m == 1 {
        return sqrt_expr; // No simplification needed
    }

    // For Add/Sub expressions, rest == base, so we need to divide by k
    // For Mul expressions, rest is already the divided part
    let actual_rest = if matches!(ctx.get(rest), Expr::Add(_, _) | Expr::Sub(_, _)) {
        // Divide entire expression by k: (4c² - 4b²) / 4 = c² - b²
        let k_expr = ctx.add(Expr::Number(num_rational::BigRational::from_integer(
            k.into(),
        )));
        ctx.add(Expr::Div(rest, k_expr))
    } else {
        rest
    };

    // Build m * sqrt(actual_rest)
    let one = ctx.num(1);
    let two = ctx.num(2);
    let half = ctx.add(Expr::Div(one, two));
    let sqrt_rest = ctx.add(Expr::Pow(actual_rest, half));
    let m_expr = ctx.add(Expr::Number(num_rational::BigRational::from_integer(
        m.into(),
    )));
    ctx.add(Expr::Mul(m_expr, sqrt_rest))
}

/// Split a numeric coefficient from an expression: k * rest
/// Returns (Some(k), rest) if there's a numeric factor, (None, original) otherwise
///
/// Handles:
/// - Mul(k, rest) → (k, rest)
/// - Add/Sub where all terms have common factor: 4*c² - 4*b² → (4, c² - b²)
fn split_numeric_factor(ctx: &Context, expr: ExprId) -> (Option<i64>, ExprId) {
    match ctx.get(expr) {
        // Direct multiplication: k * rest
        Expr::Mul(l, r) => {
            if let Expr::Number(n) = ctx.get(*l) {
                if n.is_integer() {
                    if let Some(k) = n.to_integer().to_i64() {
                        return (Some(k), *r);
                    }
                }
            }
            if let Expr::Number(n) = ctx.get(*r) {
                if n.is_integer() {
                    if let Some(k) = n.to_integer().to_i64() {
                        return (Some(k), *l);
                    }
                }
            }
            (None, expr)
        }
        // For Add/Sub, try to extract common numeric factor
        Expr::Add(_, _) | Expr::Sub(_, _) => {
            // Collect all additive terms with their signs
            let terms = collect_additive_terms(ctx, expr);
            if terms.is_empty() {
                return (None, expr);
            }

            // Extract coefficient from each term
            let coeffs: Vec<i64> = terms
                .iter()
                .filter_map(|(id, _)| get_term_coefficient(ctx, *id))
                .collect();

            if coeffs.len() != terms.len() || coeffs.is_empty() {
                return (None, expr); // Not all terms have numeric coefficients
            }

            // Compute GCD of all coefficients
            let gcd = coeffs.iter().fold(0i64, |acc, &c| gcd_i64(acc, c.abs()));
            if gcd <= 1 {
                return (None, expr); // No common factor > 1
            }

            // Check if gcd is a perfect square (otherwise not useful for sqrt)
            let sqrt_gcd = (gcd as f64).sqrt();
            let m = sqrt_gcd.round() as i64;
            if m * m != gcd {
                return (None, expr); // Not a perfect square, not useful
            }

            (Some(gcd), expr) // Return gcd and original expr (factoring happens in caller)
        }
        _ => (None, expr),
    }
}

/// Get numeric coefficient of a term (from k*x returns k, from x returns 1)
fn get_term_coefficient(ctx: &Context, expr: ExprId) -> Option<i64> {
    match ctx.get(expr) {
        Expr::Mul(l, r) => {
            if let Expr::Number(n) = ctx.get(*l) {
                if n.is_integer() {
                    return n.to_integer().to_i64();
                }
            }
            if let Expr::Number(n) = ctx.get(*r) {
                if n.is_integer() {
                    return n.to_integer().to_i64();
                }
            }
            Some(1) // Implicit coefficient 1
        }
        Expr::Neg(inner) => get_term_coefficient(ctx, *inner).map(|c| -c),
        Expr::Number(n) if n.is_integer() => n.to_integer().to_i64(),
        _ => Some(1), // Implicit coefficient 1
    }
}

/// Collect additive terms from Add/Sub chain: a + b - c → [(a, +), (b, +), (c, -)]
fn collect_additive_terms(ctx: &Context, expr: ExprId) -> Vec<(ExprId, bool)> {
    let mut terms = Vec::new();
    collect_additive_terms_recursive(ctx, expr, true, &mut terms);
    terms
}

fn collect_additive_terms_recursive(
    ctx: &Context,
    expr: ExprId,
    positive: bool,
    terms: &mut Vec<(ExprId, bool)>,
) {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            collect_additive_terms_recursive(ctx, *l, positive, terms);
            collect_additive_terms_recursive(ctx, *r, positive, terms);
        }
        Expr::Sub(l, r) => {
            collect_additive_terms_recursive(ctx, *l, positive, terms);
            collect_additive_terms_recursive(ctx, *r, !positive, terms);
        }
        Expr::Neg(inner) => {
            collect_additive_terms_recursive(ctx, *inner, !positive, terms);
        }
        _ => {
            terms.push((expr, positive));
        }
    }
}

/// GCD of two integers
fn gcd_i64(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs()
    } else {
        gcd_i64(b, a % b)
    }
}
