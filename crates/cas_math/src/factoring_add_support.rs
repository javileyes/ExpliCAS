//! Structural support for heuristic common-factor extraction on 2-term sums.

use cas_ast::{Context, Expr, ExprId};
use num_traits::Signed;

/// Parsed representation of a term in an Add/Sub expression.
/// Represents: sign * coeff * base^exp.
#[derive(Debug, Clone, Copy)]
struct ParsedTerm {
    sign: i8,     // +1 or -1
    coeff: i64,   // Integer coefficient (1 if implicit)
    base: ExprId, // The base expression (e.g., x+1)
    exp: u32,     // The exponent (>= 1)
}

/// Try to extract a common polynomial-like factor from a 2-term Add/Sub expression.
///
/// Pattern example:
/// - `(x+1)^3 + 2*(x+1)^2` -> `__hold((x+1)^2 * (x + 3))`
pub fn try_extract_common_factor_add_expr(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    // SAFE MODE: only exact 2-term Add/Sub.
    let (term1_id, term2_id, term2_positive) = match ctx.get(expr) {
        Expr::Add(l, r) => (*l, *r, true),
        Expr::Sub(l, r) => (*l, *r, false),
        _ => return None,
    };

    let term1 = parse_term(ctx, term1_id, true)?;
    let term2 = parse_term(ctx, term2_id, term2_positive)?;

    if !bases_equal(ctx, term1.base, term2.base) {
        return None;
    }

    // Base must be compound (Add) to be interesting for polynomial factorization.
    if !matches!(ctx.get(term1.base), Expr::Add(_, _)) {
        return None;
    }

    let g_exp = term1.exp.min(term2.exp);
    if g_exp == 0 {
        return None;
    }

    let q1 = build_quotient_term(ctx, term1, g_exp);
    let q2 = build_quotient_term(ctx, term2, g_exp);

    let inner_sum_raw = ctx.add(Expr::Add(q1, q2));
    let inner_sum = simplify_add_constants(ctx, inner_sum_raw);

    let factor = if g_exp == 1 {
        term1.base
    } else {
        let exp_id = ctx.num(g_exp as i64);
        ctx.add(Expr::Pow(term1.base, exp_id))
    };

    let product = ctx.add(Expr::Mul(factor, inner_sum));
    let new_expr = cas_ast::hold::wrap_hold(ctx, product);

    let old_nodes = cas_ast::count_nodes(ctx, expr);
    let new_nodes = cas_ast::count_nodes(ctx, new_expr);
    if new_nodes > old_nodes + 5 {
        return None;
    }

    Some(new_expr)
}

fn parse_term(ctx: &Context, term: ExprId, positive: bool) -> Option<ParsedTerm> {
    let sign: i8 = if positive { 1 } else { -1 };

    match ctx.get(term) {
        Expr::Pow(base, exp_id) => {
            let exp = extract_int_exp(ctx, *exp_id)?;
            if exp >= 1 {
                Some(ParsedTerm {
                    sign,
                    coeff: 1,
                    base: *base,
                    exp,
                })
            } else {
                None
            }
        }
        Expr::Mul(l, r) => {
            if let Some(c) = extract_int_coeff(ctx, *l) {
                if let Expr::Pow(base, exp_id) = ctx.get(*r) {
                    let exp = extract_int_exp(ctx, *exp_id)?;
                    if exp >= 1 {
                        return Some(ParsedTerm {
                            sign,
                            coeff: c,
                            base: *base,
                            exp,
                        });
                    }
                }
                if matches!(ctx.get(*r), Expr::Add(_, _)) {
                    return Some(ParsedTerm {
                        sign,
                        coeff: c,
                        base: *r,
                        exp: 1,
                    });
                }
            }

            if let Some(c) = extract_int_coeff(ctx, *r) {
                if let Expr::Pow(base, exp_id) = ctx.get(*l) {
                    let exp = extract_int_exp(ctx, *exp_id)?;
                    if exp >= 1 {
                        return Some(ParsedTerm {
                            sign,
                            coeff: c,
                            base: *base,
                            exp,
                        });
                    }
                }
                if matches!(ctx.get(*l), Expr::Add(_, _)) {
                    return Some(ParsedTerm {
                        sign,
                        coeff: c,
                        base: *l,
                        exp: 1,
                    });
                }
            }
            None
        }
        Expr::Neg(inner) => parse_term(ctx, *inner, !positive),
        _ => None,
    }
}

fn extract_int_exp(ctx: &Context, exp_id: ExprId) -> Option<u32> {
    if let Expr::Number(n) = ctx.get(exp_id) {
        if n.is_integer() && !n.is_negative() {
            use num_traits::ToPrimitive;
            return n.to_integer().to_u32();
        }
    }
    None
}

fn extract_int_coeff(ctx: &Context, expr: ExprId) -> Option<i64> {
    if let Expr::Number(n) = ctx.get(expr) {
        if n.is_integer() {
            use num_traits::ToPrimitive;
            return n.to_integer().to_i64();
        }
    }
    None
}

fn bases_equal(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    if a == b {
        return true;
    }
    exprs_equal_recursive(ctx, a, b)
}

fn exprs_equal_recursive(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    if a == b {
        return true;
    }
    match (ctx.get(a), ctx.get(b)) {
        (Expr::Number(n1), Expr::Number(n2)) => n1 == n2,
        (Expr::Variable(v1), Expr::Variable(v2)) => v1 == v2,
        (Expr::Constant(c1), Expr::Constant(c2)) => c1 == c2,
        (Expr::Add(l1, r1), Expr::Add(l2, r2))
        | (Expr::Sub(l1, r1), Expr::Sub(l2, r2))
        | (Expr::Mul(l1, r1), Expr::Mul(l2, r2))
        | (Expr::Div(l1, r1), Expr::Div(l2, r2))
        | (Expr::Pow(l1, r1), Expr::Pow(l2, r2)) => {
            exprs_equal_recursive(ctx, *l1, *l2) && exprs_equal_recursive(ctx, *r1, *r2)
        }
        (Expr::Neg(e1), Expr::Neg(e2)) => exprs_equal_recursive(ctx, *e1, *e2),
        (Expr::Function(n1, args1), Expr::Function(n2, args2)) => {
            n1 == n2
                && args1.len() == args2.len()
                && args1
                    .iter()
                    .zip(args2.iter())
                    .all(|(a1, a2)| exprs_equal_recursive(ctx, *a1, *a2))
        }
        _ => false,
    }
}

fn build_quotient_term(ctx: &mut Context, term: ParsedTerm, g_exp: u32) -> ExprId {
    let remaining_exp = term.exp.saturating_sub(g_exp);

    let coeff_part = if term.coeff == 1 {
        None
    } else {
        Some(ctx.num(term.coeff.abs()))
    };

    let power_part = if remaining_exp == 0 {
        None
    } else if remaining_exp == 1 {
        Some(term.base)
    } else {
        let exp_id = ctx.num(remaining_exp as i64);
        Some(ctx.add(Expr::Pow(term.base, exp_id)))
    };

    let unsigned_result = match (coeff_part, power_part) {
        (None, None) => ctx.num(1),
        (Some(c), None) => c,
        (None, Some(p)) => p,
        (Some(c), Some(p)) => ctx.add(Expr::Mul(c, p)),
    };

    if term.sign < 0 || term.coeff < 0 {
        let total_negative = (term.sign < 0) ^ (term.coeff < 0);
        if total_negative {
            ctx.add(Expr::Neg(unsigned_result))
        } else {
            unsigned_result
        }
    } else {
        unsigned_result
    }
}

fn simplify_add_constants(ctx: &mut Context, expr: ExprId) -> ExprId {
    let mut numeric_sum: i64 = 0;
    let mut non_numeric: Vec<ExprId> = Vec::new();

    collect_add_terms_for_const_fold(ctx, expr, true, &mut numeric_sum, &mut non_numeric);

    if non_numeric.is_empty() {
        ctx.num(numeric_sum)
    } else {
        let mut result = non_numeric[0];
        for term in &non_numeric[1..] {
            result = ctx.add(Expr::Add(result, *term));
        }
        if numeric_sum != 0 {
            let num_expr = ctx.num(numeric_sum);
            result = ctx.add(Expr::Add(result, num_expr));
        }
        result
    }
}

fn collect_add_terms_for_const_fold(
    ctx: &Context,
    expr: ExprId,
    positive: bool,
    numeric_sum: &mut i64,
    non_numeric: &mut Vec<ExprId>,
) {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            collect_add_terms_for_const_fold(ctx, *l, positive, numeric_sum, non_numeric);
            collect_add_terms_for_const_fold(ctx, *r, positive, numeric_sum, non_numeric);
        }
        Expr::Sub(l, r) => {
            collect_add_terms_for_const_fold(ctx, *l, positive, numeric_sum, non_numeric);
            collect_add_terms_for_const_fold(ctx, *r, !positive, numeric_sum, non_numeric);
        }
        Expr::Neg(inner) => {
            collect_add_terms_for_const_fold(ctx, *inner, !positive, numeric_sum, non_numeric);
        }
        Expr::Number(n) => {
            if n.is_integer() {
                use num_traits::ToPrimitive;
                if let Some(v) = n.to_integer().to_i64() {
                    if positive {
                        *numeric_sum += v;
                    } else {
                        *numeric_sum -= v;
                    }
                    return;
                }
            }
            non_numeric.push(expr);
        }
        _ => {
            non_numeric.push(expr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::try_extract_common_factor_add_expr;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn extracts_common_factor_for_two_term_add() {
        let mut ctx = Context::new();
        let expr = parse("(x+1)^3 + 2*(x+1)^2", &mut ctx).expect("parse");
        let out = try_extract_common_factor_add_expr(&mut ctx, expr);
        assert!(out.is_some());
    }

    #[test]
    fn rejects_non_add_sub_shape() {
        let mut ctx = Context::new();
        let expr = parse("(x+1)^3 * 2", &mut ctx).expect("parse");
        assert!(try_extract_common_factor_add_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn wraps_result_in_hold() {
        let mut ctx = Context::new();
        let expr = parse("(x+1)^3 + (x+1)^2", &mut ctx).expect("parse");
        let out = try_extract_common_factor_add_expr(&mut ctx, expr).expect("out");
        assert!(cas_ast::hold::is_hold(&ctx, out));
    }
}
