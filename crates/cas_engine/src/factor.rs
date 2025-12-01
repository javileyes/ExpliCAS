use cas_ast::{Expr, ExprId, Context};
use num_traits::{One, ToPrimitive, Signed};
// use num_rational::BigRational;
use crate::polynomial::Polynomial;
use crate::helpers::{get_square_root, get_trig_arg, is_trig_pow};
use std::collections::HashSet;
use std::cmp::Ordering;

/// Factors an expression.
/// This is the main entry point for factorization.
pub fn factor(ctx: &mut Context, expr: ExprId) -> ExprId {
    // 1. Try polynomial factorization
    if let Some(res) = factor_polynomial(ctx, expr) {
        return res;
    }

    // 2. Try difference of squares
    if let Some(res) = factor_difference_squares(ctx, expr) {
        return res;
    }

    // 3. Recursive factorization?
    // For now, just return original if no top-level factorization applies.
    // Ideally we should factor sub-expressions too.
    // But `factor` usually means "factor this polynomial".
    // Let's stick to top-level for now, or maybe recurse if it's a product/sum?
    
    expr
}

/// Factors a polynomial expression using rational roots.
pub fn factor_polynomial(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let vars = collect_variables(ctx, expr);
    if vars.len() != 1 {
        return None;
    }
    let var = vars.iter().next().unwrap();
    
    if let Ok(poly) = Polynomial::from_expr(ctx, expr, var) {
        if poly.is_zero() { return None; }

        // 1. Extract content (common constant factor)
        let factors = poly.factor_rational_roots();
        
        if factors.len() == 1 {
            // Irreducible (over rationals) or just trivial
            let content = poly.content();
            let min_deg = poly.min_degree();
            if content.is_one() && min_deg == 0 {
                return None; // No change
            }
        }

        // Group identical factors into powers
        let mut counts: Vec<(Polynomial, u32)> = Vec::new();
        for f in factors {
            if let Some((_, count)) = counts.iter_mut().find(|(p, _)| p == &f) {
                *count += 1;
            } else {
                counts.push((f, 1));
            }
        }
        
        // Construct expression
        let mut terms = Vec::new();
        for (p, count) in counts {
            let base = p.to_expr(ctx);
            if count == 1 {
                terms.push(base);
            } else {
                let exp = ctx.num(count as i64);
                terms.push(ctx.add(Expr::Pow(base, exp)));
            }
        }
        
        if terms.is_empty() { return None; }

        let mut res = terms[0];
        for t in terms.iter().skip(1) {
            res = ctx.add(Expr::Mul(res, *t));
        }
        
        // println!("factor_polynomial: {} -> {}", cas_ast::DisplayExpr { context: ctx, id: expr }, cas_ast::DisplayExpr { context: ctx, id: res });

        return Some(res);
    }
    None
}

/// Factors difference of squares: a^2 - b^2 -> (a-b)(a+b)
pub fn factor_difference_squares(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let expr_data = ctx.get(expr).clone();
    let (l, r) = match expr_data {
        Expr::Sub(l, r) => (l, r),
        Expr::Add(a, b) => {
            // Check if one is negative
            if is_negative_term(ctx, b) {
                (a, negate_term(ctx, b))
            } else if is_negative_term(ctx, a) {
                (b, negate_term(ctx, a))
            } else {
                return None;
            }
        },
        _ => return None,
    };

    if let (Some(root_l), Some(root_r)) = (get_square_root(ctx, l), get_square_root(ctx, r)) {
        // a^2 - b^2 = (a - b)(a + b)
        let term1 = ctx.add(Expr::Sub(root_l, root_r));
        
        // Check for Pythagorean identity in term2 (a + b)
        // sin^2 + cos^2 = 1
        let mut term2 = ctx.add(Expr::Add(root_l, root_r));
        let mut is_pythagorean = false;
        
        if is_sin_cos_pair(ctx, root_l, root_r) {
             term2 = ctx.num(1);
             is_pythagorean = true;
        }

        let new_expr = if is_pythagorean {
            term1
        } else {
            ctx.add(Expr::Mul(term1, term2))
        };
        
        return Some(new_expr);
    }
    None
}

// Helpers

pub fn collect_variables(ctx: &Context, expr: ExprId) -> HashSet<String> {
    use crate::visitors::VariableCollector;
    use cas_ast::Visitor;
    
    let mut collector = VariableCollector::new();
    collector.visit_expr(ctx, expr);
    collector.vars
}

fn is_sin_cos_pair(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    let arg_a = get_trig_arg(ctx, a);
    let arg_b = get_trig_arg(ctx, b);
    
    // Check if args match and are Some
    if arg_a.is_none() || arg_b.is_none() {
        return false;
    }
    
    let a_val = arg_a.unwrap();
    let b_val = arg_b.unwrap();
    
    if a_val != b_val && crate::ordering::compare_expr(ctx, a_val, b_val) != Ordering::Equal {
        return false;
    }

    let is_sin_a = is_trig_pow(ctx, a, "sin", 2);
    let is_cos_b = is_trig_pow(ctx, b, "cos", 2);
    let is_cos_a = is_trig_pow(ctx, a, "cos", 2);
    let is_sin_b = is_trig_pow(ctx, b, "sin", 2);
    
    (is_sin_a && is_cos_b) || (is_cos_a && is_sin_b)
}

fn is_negative_term(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Neg(_) => true,
        Expr::Mul(l, _) => {
            if let Expr::Number(n) = ctx.get(*l) {
                n.is_negative()
            } else {
                false
            }
        },
        Expr::Number(n) => n.is_negative(),
        _ => false
    }
}

fn negate_term(ctx: &mut Context, expr: ExprId) -> ExprId {
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Neg(inner) => inner,
        Expr::Mul(l, r) => {
            if let Expr::Number(n) = ctx.get(l) {
                if n.is_negative() {
                    let new_n = (-n).to_i64().unwrap();
                    if new_n == 1 {
                        return r;
                    }
                    let num_expr = ctx.num(new_n);
                    return ctx.add(Expr::Mul(num_expr, r));
                }
            }
            ctx.add(Expr::Neg(expr))
        },
        Expr::Number(n) => ctx.num((-n).to_i64().unwrap()),
        _ => ctx.add(Expr::Neg(expr))
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
    fn test_factor_poly_diff_squares() {
        let mut ctx = Context::new();
        let expr = parse("x^2 - 1", &mut ctx).unwrap();
        let res = factor(&mut ctx, expr);
        // factor_polynomial should catch this first as (x-1)(x+1)
        let str_res = s(&ctx, res);
        assert!(str_res.contains("x - 1") || str_res.contains("-1 + x") || str_res.contains("x + -1"));
        assert!(str_res.contains("x + 1") || str_res.contains("1 + x"));
    }

    #[test]
    fn test_factor_poly_perfect_square() {
        let mut ctx = Context::new();
        let expr = parse("x^2 + 2*x + 1", &mut ctx).unwrap();
        let res = factor(&mut ctx, expr);
        let str_res = s(&ctx, res);
        // (x+1)^2
        assert!(str_res.contains("x + 1"));
        assert!(str_res.contains("^ 2") || str_res.contains("^2"));
    }

    #[test]
    fn test_factor_diff_squares_structural() {
        let mut ctx = Context::new();
        // sin(x)^2 - cos(x)^2 -> (sin(x) - cos(x))(sin(x) + cos(x))
        // This is NOT a polynomial in x, so factor_polynomial fails.
        // factor_difference_squares should pick it up.
        let expr = parse("sin(x)^2 - cos(x)^2", &mut ctx).unwrap();
        let res = factor(&mut ctx, expr);
        let str_res = s(&ctx, res);
        assert!(str_res.contains("sin(x) - cos(x)"));
        assert!(str_res.contains("sin(x) + cos(x)"));
    }
}
