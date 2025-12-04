use crate::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use tracing::debug;

/// Detects if an expression is in a canonical (elegant) form that should not be expanded.
/// These forms are mathematically clean and expanding them would only create unnecessary complexity.
pub fn is_canonical_form(ctx: &Context, expr: ExprId) -> bool {
    debug!("Checking if canonical: {:?}", ctx.get(expr));
    match ctx.get(expr) {
        // Case 1: (product)^n where product is factored elegantly
        Expr::Pow(base, exp) => {
            is_product_of_factors(ctx, *base) && is_small_positive_integer(ctx, *exp)
        }

        // Case 2: Product of conjugates without power (e.g., (x+y)*(x-y))
        // This is already in difference of squares form, expanding serves no purpose
        Expr::Mul(l, r) => is_conjugate(ctx, *l, *r),

        // Case 3: Functions containing powers or products that should be preserved
        // Examples: sqrt((x-1)^2), sqrt((x-2)(x+2)), abs((x+y)^3)
        Expr::Function(name, args) if (name == "sqrt" || name == "abs") && args.len() == 1 => {
            let inner = args[0];
            match ctx.get(inner) {
                // Protect sqrt(x^2), sqrt((x-1)^2), etc.
                Expr::Pow(base, exp) => {
                    // Any power of 2, or product raised to a power
                    if let Expr::Number(n) = ctx.get(*exp) {
                        if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into())
                        {
                            return true; // sqrt(anything^2) should use SimplifySqrtSquareRule
                        }
                    }
                    // Also protect if base is a product
                    is_product_of_factors(ctx, *base) && is_small_positive_integer(ctx, *exp)
                }
                // Protect sqrt((x-2)(x+2)), abs((x+1)(x-1)), etc.
                Expr::Mul(l, r) => is_conjugate(ctx, *l, *r) || is_product_of_factors(ctx, inner),
                _ => false,
            }
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

            // Only protect if at least one factor is a binomial (Add/Sub)
            // This prevents protecting simple monomials like (3*x)^3
            // which should expand to 27*x^3
            let l_is_binomial = matches!(ctx.get(*l), Expr::Add(_, _) | Expr::Sub(_, _));
            let r_is_binomial = matches!(ctx.get(*r), Expr::Add(_, _) | Expr::Sub(_, _));

            if (l_is_binomial || r_is_binomial)
                && is_linear_or_simple(ctx, *l)
                && is_linear_or_simple(ctx, *r)
            {
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
/// Order-invariant: handles (x+1)(x-1), (x-1)(x+1), (-1+x)(1+x), etc.
fn is_conjugate(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    debug!("Checking conjugate: {:?} vs {:?}", ctx.get(a), ctx.get(b));
    // Extract terms and base signs from both binomials
    let (a_terms, a_base_signs) = match ctx.get(a) {
        Expr::Add(x, y) => (vec![*x, *y], vec![true, true]),
        Expr::Sub(x, y) => (vec![*x, *y], vec![true, false]),
        _ => return false,
    };

    let (b_terms, b_base_signs) = match ctx.get(b) {
        Expr::Add(x, y) => (vec![*x, *y], vec![true, true]),
        Expr::Sub(x, y) => (vec![*x, *y], vec![true, false]),
        _ => return false,
    };

    // Normalize each term (extract negative signs from numbers/Neg)
    let mut a_norm = Vec::new();
    let mut a_signs = Vec::new();
    for (&term, &base_sign) in a_terms.iter().zip(a_base_signs.iter()) {
        let (norm_term, is_pos) = normalize_term(ctx, term);
        a_norm.push(norm_term);
        a_signs.push(base_sign == is_pos);
    }

    let mut b_norm = Vec::new();
    let mut b_signs = Vec::new();
    for (&term, &base_sign) in b_terms.iter().zip(b_base_signs.iter()) {
        let (norm_term, is_pos) = normalize_term(ctx, term);
        b_norm.push(norm_term);
        b_signs.push(base_sign == is_pos);
    }

    // Check if terms match in same order
    let same_order = a_norm.len() == b_norm.len()
        && a_norm
            .iter()
            .zip(b_norm.iter())
            .all(|(&x, &y)| terms_equal_normalized(ctx, x, y));

    // Check if terms match in swapped order
    let swapped_order = a_norm.len() == 2
        && b_norm.len() == 2
        && terms_equal_normalized(ctx, a_norm[0], b_norm[1])
        && terms_equal_normalized(ctx, a_norm[1], b_norm[0]);

    if !same_order && !swapped_order {
        return false;
    }

    // Check signs: exactly one should differ
    let b_signs_to_check = if same_order {
        b_signs
    } else {
        vec![b_signs[1], b_signs[0]] // Swap to match order
    };

    let diff_count = a_signs
        .iter()
        .zip(b_signs_to_check.iter())
        .filter(|(a, b)| a != b)
        .count();

    diff_count == 1
}

/// Compare two terms that have been normalized (signs extracted)
/// For numbers, compare by absolute value
fn terms_equal_normalized(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    use num_traits::Signed;
    use std::cmp::Ordering;

    // Special case: if both are numbers, compare absolute values
    if let (Expr::Number(na), Expr::Number(nb)) = (ctx.get(a), ctx.get(b)) {
        return na.abs() == nb.abs();
    }

    // Otherwise use normal comparison
    compare_expr(ctx, a, b) == Ordering::Equal
}

/// Normalize a term: extract sign, return (term_without_sign, is_positive)
fn normalize_term(ctx: &Context, expr: ExprId) -> (ExprId, bool) {
    match ctx.get(expr) {
        Expr::Neg(inner) => (*inner, false),
        Expr::Number(n) => {
            use num_traits::Signed;
            if n.is_negative() {
                // For negative numbers, we need to compare by absolute value
                // Since we can't mutate ctx, we compare numbers directly
                (expr, false)
            } else {
                (expr, true)
            }
        }
        _ => (expr, true),
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

#[test]
fn test_canonicalized_conjugate_add_add() {
    let mut ctx = Context::new();
    // (-1 + x) * (1 + x) should be detected as conjugates
    let x = ctx.var("x");
    let one = ctx.num(1);
    let neg_one = ctx.num(-1);

    // (-1 + x) * (1 + x)
    let left = ctx.add(Expr::Add(neg_one, x));
    let right = ctx.add(Expr::Add(one, x));
    let product = ctx.add(Expr::Mul(left, right));

    assert!(
        is_canonical_form(&ctx, product),
        "(-1+x)*(1+x) should be canonical"
    );
}
