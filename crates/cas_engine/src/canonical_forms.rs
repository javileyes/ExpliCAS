use crate::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use std::cmp::Ordering;

/// Detects if an expression is in a canonical (elegant) form that should not be expanded.
/// These forms are mathematically clean and expanding them would only create unnecessary complexity.
pub fn is_canonical_form(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        // (product)^n where product is factored elegantly
        Expr::Pow(base, exp) => {
            if is_product_of_factors(ctx, *base) && is_small_positive_integer(ctx, *exp) {
                return true;
            }
            false
        }
        _ => false,
    }
}

/// Checks if base is already a product in elegant factored form
fn is_product_of_factors(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        // (a + b) * (a - b) - difference of squares form
        Expr::Mul(l, r) => {
            // Check if this is a conjugate pair (difference of squares pattern)
            if is_conjugate(ctx, *l, *r) {
                return true;
            }

            // Check if factors are linear or simple
            if is_linear_or_simple(ctx, *l) && is_linear_or_simple(ctx, *r) {
                return true;
            }

            // Recursive: check if it's a product of multiple factors
            is_product_of_factors(ctx, *l) || is_product_of_factors(ctx, *r)
        }
        _ => false,
    }
}

/// Check if expression is linear (degree 1) or simple (constant, variable)
fn is_linear_or_simple(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_) => true,
        Expr::Add(a, b) | Expr::Sub(a, b) => {
            // x + 1, x - 2, etc.
            matches!(
                (ctx.get(*a), ctx.get(*b)),
                (Expr::Variable(_), Expr::Number(_))
                    | (Expr::Number(_), Expr::Variable(_))
                    | (Expr::Neg(_), Expr::Variable(_))
                    | (Expr::Variable(_), Expr::Neg(_))
            )
        }
        Expr::Neg(inner) => is_linear_or_simple(ctx, *inner),
        _ => false,
    }
}

/// Check if exponent is a small positive integer (2, 3, etc.)
fn is_small_positive_integer(ctx: &Context, exp: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(exp) {
        if n.is_integer() && *n > num_rational::BigRational::from_integer(1.into()) {
            return true;
        }
    }
    false
}

/// Checks for conjugate pairs: (A+B) and (A-B)
fn is_conjugate(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    let a_expr = ctx.get(a);
    let b_expr = ctx.get(b);

    match (a_expr, b_expr) {
        (Expr::Add(a1, a2), Expr::Sub(b1, b2)) | (Expr::Sub(b1, b2), Expr::Add(a1, a2)) => {
            let a1 = *a1;
            let a2 = *a2;
            let b1 = *b1;
            let b2 = *b2;

            // Check: A+B vs A-B
            if compare_expr(ctx, a1, b1) == Ordering::Equal
                && compare_expr(ctx, a2, b2) == Ordering::Equal
            {
                return true;
            }
            // Check commutative: B+A vs A-B
            if compare_expr(ctx, a2, b1) == Ordering::Equal
                && compare_expr(ctx, a1, b2) == Ordering::Equal
            {
                return true;
            }
            false
        }
        (Expr::Add(a1, a2), Expr::Add(b1, b2)) => {
            // Handle (A+B) vs (A+(-B)) patterns
            let a1 = *a1;
            let a2 = *a2;
            let b1 = *b1;
            let b2 = *b2;

            // b2 is neg(a2) -> (A+B)(A-B)
            if is_negation(ctx, a2, b2) && compare_expr(ctx, a1, b1) == Ordering::Equal {
                return true;
            }
            // b1 is neg(a2) -> (A+B)(-B+A)
            if is_negation(ctx, a2, b1) && compare_expr(ctx, a1, b2) == Ordering::Equal {
                return true;
            }
            // b2 is neg(a1) -> (A+B)(B-A)
            if is_negation(ctx, a1, b2) && compare_expr(ctx, a2, b1) == Ordering::Equal {
                return true;
            }
            // b1 is neg(a1) -> (A+B)(-A+B)
            if is_negation(ctx, a1, b1) && compare_expr(ctx, a2, b2) == Ordering::Equal {
                return true;
            }
            false
        }
        _ => false,
    }
}

fn is_negation(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    check_negation_structure(ctx, b, a) || check_negation_structure(ctx, a, b)
}

fn check_negation_structure(ctx: &Context, potential_neg: ExprId, original: ExprId) -> bool {
    match ctx.get(potential_neg) {
        Expr::Neg(n) => compare_expr(ctx, original, *n) == Ordering::Equal,
        Expr::Mul(l, r) => {
            // Check for -1 * original
            if let Expr::Number(n) = ctx.get(*l) {
                if *n == -num_rational::BigRational::from_integer(1.into())
                    && compare_expr(ctx, *r, original) == Ordering::Equal
                {
                    return true;
                }
            }
            // Check for original * -1
            if let Expr::Number(n) = ctx.get(*r) {
                if *n == -num_rational::BigRational::from_integer(1.into())
                    && compare_expr(ctx, *l, original) == Ordering::Equal
                {
                    return true;
                }
            }
            false
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Context;

    #[test]
    fn test_canonical_difference_of_squares_squared() {
        let mut ctx = Context::new();
        // ((x+1)*(x-1))^2
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_plus_1 = ctx.add(Expr::Add(x, one));
        let x_minus_1 = ctx.add(Expr::Sub(x, one));
        let product = ctx.add(Expr::Mul(x_plus_1, x_minus_1));
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Pow(product, two));

        assert!(
            is_canonical_form(&ctx, expr),
            "((x+1)*(x-1))^2 should be canonical"
        );
    }

    #[test]
    fn test_simple_binomial_not_canonical() {
        let mut ctx = Context::new();
        // (x+1)^2 - should expand for educational purposes
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_plus_1 = ctx.add(Expr::Add(x, one));
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Pow(x_plus_1, two));

        assert!(
            !is_canonical_form(&ctx, expr),
            "(x+1)^2 should NOT be canonical (should expand)"
        );
    }

    #[test]
    fn test_x_squared_minus_one_squared() {
        let mut ctx = Context::new();
        // (x^2-1)^2 can be expanded
        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let x_sq = ctx.add(Expr::Pow(x, two));
        let x_sq_minus_1 = ctx.add(Expr::Sub(x_sq, one));
        let expr = ctx.add(Expr::Pow(x_sq_minus_1, two));

        // This is NOT a product, so not canonical
        assert!(!is_canonical_form(&ctx, expr));
    }
}
