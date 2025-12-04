use crate::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use std::cmp::Ordering;

/// Detects if an expression is in a canonical (elegant) form that should not be expanded.
/// These forms are mathematically clean and expanding them would only create unnecessary complexity.
pub fn is_canonical_form(ctx: &Context, expr: ExprId) -> bool {
    use cas_ast::DisplayExpr;
    eprintln!(
        "DEBUG is_canonical_form called with: {}",
        DisplayExpr {
            context: ctx,
            id: expr
        }
    );

    let result = match ctx.get(expr) {
        // Case 1: (product)^n where product is factored elegantly
        Expr::Pow(base, exp) => {
            if is_product_of_factors(ctx, *base) && is_small_positive_integer(ctx, *exp) {
                eprintln!("DEBUG: Match case 1 - Pow with product base");
                true
            } else {
                false
            }
        }

        // Case 2: Product of conjugates without power (e.g., (x+y)*(x-y))
        // This is already in difference of squares form, expanding serves no purpose
        Expr::Mul(l, r) => {
            let is_conj = is_conjugate(ctx, *l, *r);
            eprintln!("DEBUG: Match case 2 - Mul, is_conjugate: {}", is_conj);
            is_conj
        }

        _ => {
            eprintln!("DEBUG: No match, returning false");
            false
        }
    };

    eprintln!("DEBUG is_canonical_form returning: {}", result);
    result
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
    use cas_ast::DisplayExpr;

    eprintln!(
        "DEBUG is_conjugate called with: ({}) and ({})",
        DisplayExpr {
            context: ctx,
            id: a
        },
        DisplayExpr {
            context: ctx,
            id: b
        }
    );

    // Extract terms and base signs from both binomials
    let (a_terms, a_base_signs) = match ctx.get(a) {
        Expr::Add(x, y) => (vec![*x, *y], vec![true, true]),
        Expr::Sub(x, y) => (vec![*x, *y], vec![true, false]),
        _ => {
            eprintln!("DEBUG: a is not Add/Sub, returning false");
            return false;
        }
    };

    let (b_terms, b_base_signs) = match ctx.get(b) {
        Expr::Add(x, y) => (vec![*x, *y], vec![true, true]),
        Expr::Sub(x, y) => (vec![*x, *y], vec![true, false]),
        _ => {
            eprintln!("DEBUG: b is not Add/Sub, returning false");
            return false;
        }
    };

    // Normalize each term (extract negative signs from numbers/Neg)
    let mut a_norm = Vec::new();
    let mut a_signs = Vec::new();
    for (&term, &base_sign) in a_terms.iter().zip(a_base_signs.iter()) {
        let (norm_term, is_pos) = normalize_term(ctx, term);
        a_norm.push(norm_term);
        let final_sign = base_sign == is_pos;
        a_signs.push(final_sign);
        eprintln!(
            "DEBUG a term: {} → norm: {}, is_pos: {}, base_sign: {}, final_sign: {}",
            DisplayExpr {
                context: ctx,
                id: term
            },
            DisplayExpr {
                context: ctx,
                id: norm_term
            },
            is_pos,
            base_sign,
            final_sign
        );
    }

    let mut b_norm = Vec::new();
    let mut b_signs = Vec::new();
    for (&term, &base_sign) in b_terms.iter().zip(b_base_signs.iter()) {
        let (norm_term, is_pos) = normalize_term(ctx, term);
        b_norm.push(norm_term);
        let final_sign = base_sign == is_pos;
        b_signs.push(final_sign);
        eprintln!(
            "DEBUG b term: {} → norm: {}, is_pos: {}, base_sign: {}, final_sign: {}",
            DisplayExpr {
                context: ctx,
                id: term
            },
            DisplayExpr {
                context: ctx,
                id: norm_term
            },
            is_pos,
            base_sign,
            final_sign
        );
    }

    // Check if terms match in same order
    let same_order = a_norm.len() == b_norm.len()
        && a_norm
            .iter()
            .zip(b_norm.iter())
            .all(|(&x, &y)| terms_equal_normalized(ctx, x, y));

    eprintln!("DEBUG same_order: {}", same_order);

    // Check if terms match in swapped order
    let swapped_order = a_norm.len() == 2
        && b_norm.len() == 2
        && terms_equal_normalized(ctx, a_norm[0], b_norm[1])
        && terms_equal_normalized(ctx, a_norm[1], b_norm[0]);

    eprintln!("DEBUG swapped_order: {}", swapped_order);

    if !same_order && !swapped_order {
        eprintln!("DEBUG: Terms don't match, returning false");
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

    eprintln!("DEBUG diff_count: {} (need exactly 1)", diff_count);
    eprintln!(
        "DEBUG a_signs: {:?}, b_signs_to_check: {:?}",
        a_signs, b_signs_to_check
    );

    let result = diff_count == 1;
    eprintln!("DEBUG is_conjugate returning: {}", result);
    result
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
