use crate::build::mul2_raw;
use crate::helpers::flatten_add_sub_chain;
use cas_ast::{Context, Expr, ExprId};

use num_rational::BigRational;
use num_traits::{One, Zero};

/// Helper: Build a 2-factor product (no normalization).
#[inline]

/// Collects like terms in an expression.
/// e.g. 2*x + 3*x -> 5*x
///      x + x -> 2*x
///      x^2 + 2*x^2 -> 3*x^2
pub fn collect(ctx: &mut Context, expr: ExprId) -> ExprId {
    // CRITICAL: Do NOT collect non-commutative expressions (e.g., matrices)
    // Matrix addition/subtraction has dedicated rules (MatrixAddRule, MatrixSubRule)
    // Collecting M + M would incorrectly produce 2*M
    if !ctx.is_mul_commutative(expr) {
        return expr;
    }

    // 1. Check if expression is an Add/Sub chain
    // We used to bail out if not Add/Sub, but we want to handle Neg(Neg(x)) -> x
    // and other simplifications even for single terms.
    // So we proceed regardless.

    // let expr_data = ctx.get(expr).clone();
    // match expr_data {
    //     Expr::Add(_, _) | Expr::Sub(_, _) => {
    //         // Proceed to collect
    //     },
    //     _ => return expr, // Nothing to collect at top level
    // }

    // 2. Flatten terms (using shared helper from crate::helpers)
    let terms = flatten_add_sub_chain(ctx, expr);

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
        } else if coeff.is_one() {
            term_part
        } else if coeff == BigRational::from_integer((-1).into()) {
            // Use Neg(x) instead of Mul(-1, x) for conciseness
            ctx.add(Expr::Neg(term_part))
        } else {
            let coeff_expr = ctx.add(Expr::Number(coeff));
            mul2_raw(ctx, coeff_expr, term_part)
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

// flatten_add_chain is now provided by crate::helpers::flatten_add_sub_chain

fn extract_numerical_coeff(ctx: &mut Context, expr: ExprId) -> (BigRational, ExprId) {
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Number(n) => (n, ctx.num(1)), // 5 -> 5 * 1
        Expr::Neg(e) => {
            let (c, t) = extract_numerical_coeff(ctx, e);
            (-c, t)
        }
        Expr::Mul(l, r) => {
            // Check if l is number
            if let Expr::Number(n) = ctx.get(l) {
                (n.clone(), r)
            } else {
                (BigRational::one(), expr)
            }
        }
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

/// Simplify numeric sums in exponents throughout an expression tree.
/// e.g., x^(1/2 + 1/3) → x^(5/6)
/// This is applied during the collect phase for early, visible simplification.
pub fn simplify_numeric_exponents(ctx: &mut Context, expr: ExprId) -> ExprId {
    let expr_data = ctx.get(expr).clone();

    match expr_data {
        Expr::Pow(base, exp) => {
            // Recursively simplify base first
            let new_base = simplify_numeric_exponents(ctx, base);

            // Try to sum numeric fractions in exponent
            if let Some(sum) = try_sum_numeric_fractions(ctx, exp) {
                let new_exp = ctx.add(Expr::Number(sum));
                ctx.add(Expr::Pow(new_base, new_exp))
            } else {
                // Recursively simplify exponent if not purely numeric
                let new_exp = simplify_numeric_exponents(ctx, exp);
                if new_base != base || new_exp != exp {
                    ctx.add(Expr::Pow(new_base, new_exp))
                } else {
                    expr
                }
            }
        }
        Expr::Add(l, r) => {
            let nl = simplify_numeric_exponents(ctx, l);
            let nr = simplify_numeric_exponents(ctx, r);
            if nl != l || nr != r {
                ctx.add(Expr::Add(nl, nr))
            } else {
                expr
            }
        }
        Expr::Sub(l, r) => {
            let nl = simplify_numeric_exponents(ctx, l);
            let nr = simplify_numeric_exponents(ctx, r);
            if nl != l || nr != r {
                ctx.add(Expr::Sub(nl, nr))
            } else {
                expr
            }
        }
        Expr::Mul(l, r) => {
            let nl = simplify_numeric_exponents(ctx, l);
            let nr = simplify_numeric_exponents(ctx, r);
            if nl != l || nr != r {
                mul2_raw(ctx, nl, nr)
            } else {
                expr
            }
        }
        Expr::Div(l, r) => {
            let nl = simplify_numeric_exponents(ctx, l);
            let nr = simplify_numeric_exponents(ctx, r);
            if nl != l || nr != r {
                ctx.add(Expr::Div(nl, nr))
            } else {
                expr
            }
        }
        Expr::Neg(e) => {
            let ne = simplify_numeric_exponents(ctx, e);
            if ne != e {
                ctx.add(Expr::Neg(ne))
            } else {
                expr
            }
        }
        Expr::Function(name, args) => {
            let mut changed = false;
            let new_args: Vec<ExprId> = args
                .iter()
                .map(|a| {
                    let na = simplify_numeric_exponents(ctx, *a);
                    if na != *a {
                        changed = true;
                    }
                    na
                })
                .collect();
            if changed {
                ctx.add(Expr::Function(name, new_args))
            } else {
                expr
            }
        }
        _ => expr, // Number, Variable, Constant, Matrix - no processing needed
    }
}

/// Try to extract and sum all numeric fractions from an Add chain.
/// Returns Some(sum) if the entire chain is numeric fractions, None otherwise.
fn try_sum_numeric_fractions(ctx: &Context, exp: ExprId) -> Option<BigRational> {
    let mut addends: Vec<BigRational> = Vec::new();
    let mut stack = vec![exp];

    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Add(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Number(n) => {
                addends.push(n.clone());
            }
            Expr::Div(num, den) => {
                if let (Expr::Number(n), Expr::Number(d)) = (ctx.get(*num), ctx.get(*den)) {
                    if !d.is_zero() {
                        addends.push(n / d);
                    } else {
                        return None; // Division by zero
                    }
                } else {
                    return None; // Non-numeric fraction
                }
            }
            _ => return None, // Non-numeric term in Add chain
        }
    }

    // Only return sum if there are at least 2 addends (actual simplification)
    if addends.len() >= 2 {
        Some(addends.iter().sum())
    } else {
        None
    }
}

#[cfg(test)]

mod tests {
    use super::*;
    use cas_ast::DisplayExpr;
    use cas_parser::parse;

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

    #[test]
    fn test_simplify_numeric_exponents() {
        let mut ctx = Context::new();
        // x^(1/2 + 1/3) should become x^(5/6)
        let expr = parse("x^(1/2 + 1/3)", &mut ctx).unwrap();
        let res = simplify_numeric_exponents(&mut ctx, expr);
        // The result should be different (simplified)
        assert_ne!(res, expr, "Expression should be simplified");
        assert_eq!(s(&ctx, res), "x^(5/6)");
    }
}
