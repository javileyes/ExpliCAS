//! Helpers for opaque-atom polynomial detection and substitution.
//!
//! These utilities support rules that prove polynomial identities by
//! temporarily treating opaque nodes (function calls, `e^(k*u)` atoms) as
//! algebraic variables.

use cas_ast::ordering::compare_expr;
use cas_ast::{Constant, Context, Expr, ExprId};
use num_traits::{Signed, ToPrimitive};
use std::cmp::Ordering;

/// Quick check: expression is polynomial-like up to configured limits.
pub fn is_polynomial_candidate(
    ctx: &Context,
    expr: ExprId,
    max_depth: usize,
    max_pow_exp: u32,
) -> bool {
    is_polynomial_candidate_inner(ctx, expr, 0, max_depth, max_pow_exp)
}

fn is_polynomial_candidate_inner(
    ctx: &Context,
    expr: ExprId,
    depth: usize,
    max_depth: usize,
    max_pow_exp: u32,
) -> bool {
    if depth > max_depth {
        return false;
    }

    match ctx.get(expr) {
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) => true,
        Expr::Function(_, _) => true,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            is_polynomial_candidate_inner(ctx, *l, depth + 1, max_depth, max_pow_exp)
                && is_polynomial_candidate_inner(ctx, *r, depth + 1, max_depth, max_pow_exp)
        }
        Expr::Neg(inner) => {
            is_polynomial_candidate_inner(ctx, *inner, depth + 1, max_depth, max_pow_exp)
        }
        Expr::Pow(base, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() && !n.is_negative() {
                    if let Some(e) = n.to_integer().to_u32() {
                        if e <= max_pow_exp {
                            return is_polynomial_candidate_inner(
                                ctx,
                                *base,
                                depth + 1,
                                max_depth,
                                max_pow_exp,
                            );
                        }
                    }
                }
            }
            matches!(ctx.get(*base), Expr::Constant(Constant::E))
        }
        _ => false,
    }
}

/// Collect all function calls under `expr` up to `max_depth`.
pub fn collect_function_calls(ctx: &Context, expr: ExprId, max_depth: usize) -> Vec<ExprId> {
    let mut out = Vec::new();
    collect_function_calls_inner(ctx, expr, &mut out, 0, max_depth);
    out
}

fn collect_function_calls_inner(
    ctx: &Context,
    expr: ExprId,
    out: &mut Vec<ExprId>,
    depth: usize,
    max_depth: usize,
) {
    if depth > max_depth {
        return;
    }
    match ctx.get(expr) {
        Expr::Function(_, _) => out.push(expr),
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            collect_function_calls_inner(ctx, *l, out, depth + 1, max_depth);
            collect_function_calls_inner(ctx, *r, out, depth + 1, max_depth);
        }
        Expr::Pow(base, exp) => {
            collect_function_calls_inner(ctx, *base, out, depth + 1, max_depth);
            collect_function_calls_inner(ctx, *exp, out, depth + 1, max_depth);
        }
        Expr::Neg(inner) => collect_function_calls_inner(ctx, *inner, out, depth + 1, max_depth),
        _ => {}
    }
}

/// Deduplicate ids by structural equality in encounter order.
pub fn dedup_expr_ids(ctx: &Context, ids: &[ExprId]) -> Vec<ExprId> {
    let mut unique = Vec::new();
    for &id in ids {
        let already = unique
            .iter()
            .any(|&u| compare_expr(ctx, id, u) == Ordering::Equal);
        if !already {
            unique.push(id);
        }
    }
    unique
}

/// Collect all exponents from `e^(exp)` nodes under `expr` up to `max_depth`.
pub fn collect_exp_exponents(ctx: &Context, expr: ExprId, max_depth: usize) -> Vec<ExprId> {
    let mut out = Vec::new();
    collect_exp_exponents_inner(ctx, expr, &mut out, 0, max_depth);
    out
}

fn collect_exp_exponents_inner(
    ctx: &Context,
    expr: ExprId,
    out: &mut Vec<ExprId>,
    depth: usize,
    max_depth: usize,
) {
    if depth > max_depth {
        return;
    }
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            if matches!(ctx.get(*base), Expr::Constant(Constant::E)) {
                out.push(*exp);
            } else {
                collect_exp_exponents_inner(ctx, *base, out, depth + 1, max_depth);
            }
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            collect_exp_exponents_inner(ctx, *l, out, depth + 1, max_depth);
            collect_exp_exponents_inner(ctx, *r, out, depth + 1, max_depth);
        }
        Expr::Neg(inner) => collect_exp_exponents_inner(ctx, *inner, out, depth + 1, max_depth),
        _ => {}
    }
}

fn extract_integer_factor(ctx: &Context, expr: ExprId, max_factor: u32) -> (u32, ExprId) {
    match ctx.get(expr) {
        Expr::Mul(l, r) => {
            if let Expr::Number(n) = ctx.get(*l) {
                if n.is_integer() && n.is_positive() {
                    if let Some(k) = n.to_integer().to_u32() {
                        if k <= max_factor {
                            return (k, *r);
                        }
                    }
                }
            }
            if let Expr::Number(n) = ctx.get(*r) {
                if n.is_integer() && n.is_positive() {
                    if let Some(k) = n.to_integer().to_u32() {
                        if k <= max_factor {
                            return (k, *l);
                        }
                    }
                }
            }
            (1, expr)
        }
        _ => (1, expr),
    }
}

/// Find common base `u` such that every exponent is `k*u` with `1 <= k <= max_factor`.
pub fn find_exp_base(ctx: &Context, exponents: &[ExprId], max_factor: u32) -> Option<ExprId> {
    if exponents.is_empty() {
        return None;
    }

    let factored: Vec<(u32, ExprId)> = exponents
        .iter()
        .map(|&e| extract_integer_factor(ctx, e, max_factor))
        .collect();
    let base = factored[0].1;
    for &(_, rest) in &factored[1..] {
        if compare_expr(ctx, base, rest) != Ordering::Equal {
            return None;
        }
    }
    Some(base)
}

/// Substitute all `e^(k*exp_base)` nodes by `replacement_var^k`.
pub fn substitute_exp_atoms(
    ctx: &mut Context,
    expr: ExprId,
    exp_base: ExprId,
    replacement_var: ExprId,
    max_depth: usize,
    max_factor: u32,
) -> ExprId {
    substitute_exp_atoms_inner(
        ctx,
        expr,
        exp_base,
        replacement_var,
        0,
        max_depth,
        max_factor,
    )
}

fn substitute_exp_atoms_inner(
    ctx: &mut Context,
    expr: ExprId,
    exp_base: ExprId,
    replacement_var: ExprId,
    depth: usize,
    max_depth: usize,
    max_factor: u32,
) -> ExprId {
    if depth > max_depth {
        return expr;
    }
    match ctx.get(expr).clone() {
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => {
            let (k, rest) = extract_integer_factor(ctx, exp, max_factor);
            if compare_expr(ctx, rest, exp_base) == Ordering::Equal {
                if k == 1 {
                    return replacement_var;
                }
                let exp_k = ctx.num(k as i64);
                return ctx.add(Expr::Pow(replacement_var, exp_k));
            }
            expr
        }
        Expr::Add(l, r) => {
            let new_l = substitute_exp_atoms_inner(
                ctx,
                l,
                exp_base,
                replacement_var,
                depth + 1,
                max_depth,
                max_factor,
            );
            let new_r = substitute_exp_atoms_inner(
                ctx,
                r,
                exp_base,
                replacement_var,
                depth + 1,
                max_depth,
                max_factor,
            );
            if new_l == l && new_r == r {
                expr
            } else {
                ctx.add(Expr::Add(new_l, new_r))
            }
        }
        Expr::Sub(l, r) => {
            let new_l = substitute_exp_atoms_inner(
                ctx,
                l,
                exp_base,
                replacement_var,
                depth + 1,
                max_depth,
                max_factor,
            );
            let new_r = substitute_exp_atoms_inner(
                ctx,
                r,
                exp_base,
                replacement_var,
                depth + 1,
                max_depth,
                max_factor,
            );
            if new_l == l && new_r == r {
                expr
            } else {
                ctx.add(Expr::Sub(new_l, new_r))
            }
        }
        Expr::Mul(l, r) => {
            let new_l = substitute_exp_atoms_inner(
                ctx,
                l,
                exp_base,
                replacement_var,
                depth + 1,
                max_depth,
                max_factor,
            );
            let new_r = substitute_exp_atoms_inner(
                ctx,
                r,
                exp_base,
                replacement_var,
                depth + 1,
                max_depth,
                max_factor,
            );
            if new_l == l && new_r == r {
                expr
            } else {
                ctx.add(Expr::Mul(new_l, new_r))
            }
        }
        Expr::Neg(inner) => {
            let new_inner = substitute_exp_atoms_inner(
                ctx,
                inner,
                exp_base,
                replacement_var,
                depth + 1,
                max_depth,
                max_factor,
            );
            if new_inner == inner {
                expr
            } else {
                ctx.add(Expr::Neg(new_inner))
            }
        }
        Expr::Pow(base, exp) => {
            let new_base = substitute_exp_atoms_inner(
                ctx,
                base,
                exp_base,
                replacement_var,
                depth + 1,
                max_depth,
                max_factor,
            );
            if new_base == base {
                expr
            } else {
                ctx.add(Expr::Pow(new_base, exp))
            }
        }
        _ => expr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn polynomial_candidate_handles_opaque_exp_atoms() {
        let mut ctx = Context::new();
        let ok = parse("(e^(2*x)+1)^3 - 1", &mut ctx).expect("parse ok");
        let bad = parse("1/(x+1)", &mut ctx).expect("parse bad");
        assert!(is_polynomial_candidate(&ctx, ok, 30, 6));
        assert!(!is_polynomial_candidate(&ctx, bad, 30, 6));
    }

    #[test]
    fn function_call_collection_and_dedup() {
        let mut ctx = Context::new();
        let expr = parse("sin(x)+sin(x)+cos(x)", &mut ctx).expect("parse");
        let calls = collect_function_calls(&ctx, expr, 6);
        let unique = dedup_expr_ids(&ctx, &calls);
        assert_eq!(unique.len(), 2);
    }

    #[test]
    fn exponential_base_detection() {
        let mut ctx = Context::new();
        let expr = parse("e^(3*u) + e^u + e^(2*u)", &mut ctx).expect("parse");
        let exps = collect_exp_exponents(&ctx, expr, 30);
        let base = find_exp_base(&ctx, &exps, 6).expect("base");
        let expected = parse("u", &mut ctx).expect("parse u");
        assert_eq!(compare_expr(&ctx, base, expected), Ordering::Equal);
    }

    #[test]
    fn substitute_exp_atoms_rewrites_matching_terms() {
        let mut ctx = Context::new();
        let expr = parse("e^(2*x) + 3*e^x + 5", &mut ctx).expect("parse");
        let base = parse("x", &mut ctx).expect("parse x");
        let t = ctx.var("t");
        let out = substitute_exp_atoms(&mut ctx, expr, base, t, 30, 6);
        let expected = parse("t^2 + 3*t + 5", &mut ctx).expect("expected");
        assert_eq!(compare_expr(&ctx, out, expected), Ordering::Equal);
    }
}
