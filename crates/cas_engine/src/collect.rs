use cas_ast::{Expr, ExprId, Context};

use num_traits::{One, Zero};
use num_rational::BigRational;

/// Collects like terms in an expression.
/// e.g. 2*x + 3*x -> 5*x
///      x + x -> 2*x
///      x^2 + 2*x^2 -> 3*x^2
pub fn collect(ctx: &mut Context, expr: ExprId) -> ExprId {
    // 1. Check if expression is an Add/Sub chain
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Add(_, _) | Expr::Sub(_, _) => {
            // Proceed to collect
        },
        _ => return expr, // Nothing to collect at top level
    }

    // 2. Flatten terms
    let terms = flatten_add_chain(ctx, expr);

    // 3. Group terms by their "non-coefficient" part
    // We need a canonical representation of the term without its numerical coefficient.
    // e.g. 2*x -> (2, x)
    //      x -> (1, x)
    //      3*x*y -> (3, x*y)
    //      5 -> (5, 1)
    
    // Map: TermSignature -> Coefficient
    // We can't use ExprId as key directly because we want structural equality, 
    // but for now let's assume canonicalization handles structural equality or we use a string key?
    // Using a string key is slow but safe for now. 
    // Better: Use a helper to extract (coeff, rest) and compare 'rest' structurally.
    
    // Let's use a Vec of groups to avoid complex hashing for now.
    // Vec<(coeff, term_part)>
    let mut groups: Vec<(BigRational, ExprId)> = Vec::new();

    for term in terms {
        let (coeff, term_part) = extract_numerical_coeff(ctx, term);
        
        // Find if we already have this term_part in groups
        let mut found = false;
        for (g_coeff, g_term) in groups.iter_mut() {
            if are_structurally_equal(ctx, *g_term, term_part) {
                *g_coeff += coeff.clone();
                found = true;
                break;
            }
        }
        
        if !found {
            groups.push((coeff, term_part));
        }
    }

    // Sort groups to ensure canonical order
    groups.sort_by(|a, b| crate::ordering::compare_expr(ctx, a.1, b.1));

    // 4. Reconstruct expression
    let mut new_terms = Vec::new();
    for (coeff, term_part) in groups {
        if coeff.is_zero() {
            continue;
        }
        
        let term = if is_one_term(ctx, term_part) {
            // Just the coefficient (constant term)
            ctx.add(Expr::Number(coeff))
        } else {
            if coeff.is_one() {
                term_part
            } else if coeff == BigRational::from_integer((-1).into()) {
                // Use Mul(-1, x) instead of Neg(x) to match CanonicalizeNegationRule
                let minus_one = ctx.num(-1);
                ctx.add(Expr::Mul(minus_one, term_part))
            } else {
                let coeff_expr = ctx.add(Expr::Number(coeff));
                ctx.add(Expr::Mul(coeff_expr, term_part))
            }
        };
        new_terms.push(term);
    }

    // Sort terms by global canonical order to match CanonicalizeAddRule
    new_terms.sort_by(|a, b| crate::ordering::compare_expr(ctx, *a, *b));

    if new_terms.is_empty() {
        return ctx.num(0);
    }

    // Construct Add chain (Right-Associative to match CanonicalizeAddRule)
    let mut result = *new_terms.last().unwrap();
    for t in new_terms.iter().rev().skip(1) {
        // Optimization: Handle negative terms?
        // For right-associative, it's harder to peek.
        // Let's just use Add for now.
        result = ctx.add(Expr::Add(*t, result));
    }

    result
}

// --- Helpers ---

fn flatten_add_chain(ctx: &mut Context, expr: ExprId) -> Vec<ExprId> {
    let mut terms = Vec::new();
    flatten_recursive(ctx, expr, &mut terms, false);
    terms
}

fn flatten_recursive(ctx: &mut Context, expr: ExprId, terms: &mut Vec<ExprId>, negate: bool) {
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Add(l, r) => {
            flatten_recursive(ctx, l, terms, negate);
            flatten_recursive(ctx, r, terms, negate);
        },
        Expr::Sub(l, r) => {
            flatten_recursive(ctx, l, terms, negate);
            flatten_recursive(ctx, r, terms, !negate);
        },
        _ => {
            if negate {
                terms.push(ctx.add(Expr::Neg(expr)));
            } else {
                terms.push(expr);
            }
        }
    }
}

fn extract_numerical_coeff(ctx: &mut Context, expr: ExprId) -> (BigRational, ExprId) {
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Number(n) => (n, ctx.num(1)), // 5 -> 5 * 1
        Expr::Neg(e) => {
            let (c, t) = extract_numerical_coeff(ctx, e);
            (-c, t)
        },
        Expr::Mul(l, r) => {
            // Check if l is number
            if let Expr::Number(n) = ctx.get(l) {
                (n.clone(), r)
            } else {
                (BigRational::one(), expr)
            }
        },
        _ => (BigRational::one(), expr),
    }
}

fn are_structurally_equal(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    crate::ordering::compare_expr(ctx, a, b) == std::cmp::Ordering::Equal
}

fn is_one_term(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        n.is_one()
    } else {
        false
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;
    use cas_ast::DisplayExpr;

    fn s(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn test_collect_integers() {
        let mut ctx = Context::new();
        let expr = parse("1 + 2", &mut ctx).unwrap();
        let res = collect(&mut ctx, expr);
        assert_eq!(s(&ctx, res), "3");
    }

    #[test]
    fn test_collect_variables() {
        let mut ctx = Context::new();
        let expr = parse("x + x", &mut ctx).unwrap();
        let res = collect(&mut ctx, expr);
        assert_eq!(s(&ctx, res), "2 * x");
    }

    #[test]
    fn test_collect_mixed() {
        let mut ctx = Context::new();
        let expr = parse("2*x + 3*y + 4*x", &mut ctx).unwrap();
        let res = collect(&mut ctx, expr);
        // Order depends on implementation, but should have 6*x and 3*y
        let res_str = s(&ctx, res);
        assert!(res_str.contains("6 * x"));
        assert!(res_str.contains("3 * y"));
    }

    #[test]
    fn test_collect_cancel() {
        let mut ctx = Context::new();
        let expr = parse("x - x", &mut ctx).unwrap();
        let res = collect(&mut ctx, expr);
        assert_eq!(s(&ctx, res), "0");
    }
    
    #[test]
    fn test_collect_powers() {
        let mut ctx = Context::new();
        let expr = parse("x^2 + 2*x^2", &mut ctx).unwrap();
        let res = collect(&mut ctx, expr);
        assert_eq!(s(&ctx, res), "3 * x^2");
    }
}
