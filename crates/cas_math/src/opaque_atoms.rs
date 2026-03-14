//! Helpers for opaque-atom polynomial detection and substitution.
//!
//! These utilities support rules that prove polynomial identities by
//! temporarily treating opaque nodes (function calls, `e^(k*u)` atoms) as
//! algebraic variables.

use cas_ast::ordering::compare_expr;
use cas_ast::{Constant, Context, Expr, ExprId};
use num_traits::{Signed, ToPrimitive, Zero};
use std::cmp::Ordering;

fn extract_signed_rational_exponent(ctx: &Context, exp: ExprId) -> Option<(i32, u32)> {
    match ctx.get(exp) {
        Expr::Number(n) => {
            if n.is_zero() {
                return None;
            }
            let numer = n.numer().to_i32()?;
            let denom = n.denom().to_u32()?;
            Some((numer, denom))
        }
        Expr::Div(num, den) => {
            let (Expr::Number(num_n), Expr::Number(den_n)) = (ctx.get(*num), ctx.get(*den)) else {
                return None;
            };
            if !num_n.is_integer()
                || !den_n.is_integer()
                || num_n.is_zero()
                || den_n.is_zero()
                || den_n.is_negative()
            {
                return None;
            }
            Some((num_n.to_integer().to_i32()?, den_n.to_integer().to_u32()?))
        }
        Expr::Neg(inner) => {
            let (numer, denom) = extract_signed_rational_exponent(ctx, *inner)?;
            Some((-numer, denom))
        }
        _ => None,
    }
}

fn extract_positive_rational_exponent(ctx: &Context, exp: ExprId) -> Option<(u32, u32)> {
    let (numer, denom) = extract_signed_rational_exponent(ctx, exp)?;
    if numer <= 0 {
        return None;
    }
    Some((numer as u32, denom))
}

fn is_noninteger_rational_exponent(ctx: &Context, exp: ExprId) -> bool {
    let Some((numer, denom)) = extract_signed_rational_exponent(ctx, exp) else {
        return false;
    };
    numer.unsigned_abs() % denom != 0
}

fn is_opaque_power_atom(ctx: &Context, base: ExprId, exp: ExprId) -> bool {
    !matches!(ctx.get(base), Expr::Constant(Constant::E))
        && is_noninteger_rational_exponent(ctx, exp)
}

fn is_opaque_constant_atom(c: &Constant) -> bool {
    matches!(c, Constant::Pi | Constant::E | Constant::Phi)
}

pub fn extract_opaque_reciprocal_power_base(ctx: &Context, expr: ExprId) -> Option<(ExprId, u32)> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if matches!(ctx.get(*base), Expr::Constant(Constant::E)) {
        return None;
    }

    let (numer, denom) = extract_positive_rational_exponent(ctx, *exp)?;
    if numer == 1 && denom >= 2 {
        Some((*base, denom))
    } else {
        None
    }
}

pub fn extract_opaque_negative_reciprocal_power_base(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, u32)> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if matches!(ctx.get(*base), Expr::Constant(Constant::E)) {
        return None;
    }

    let (numer, denom) = extract_signed_rational_exponent(ctx, *exp)?;
    if numer == -1 && denom >= 2 {
        Some((*base, denom))
    } else {
        None
    }
}

pub fn extract_opaque_rational_power_atom(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, u32, u32)> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if matches!(ctx.get(*base), Expr::Constant(Constant::E)) {
        return None;
    }
    let (numer, denom) = extract_positive_rational_exponent(ctx, *exp)?;
    if numer % denom == 0 {
        return None;
    }
    Some((*base, numer, denom))
}

pub fn extract_opaque_signed_rational_power_atom(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, i32, u32)> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if matches!(ctx.get(*base), Expr::Constant(Constant::E)) {
        return None;
    }
    let (numer, denom) = extract_signed_rational_exponent(ctx, *exp)?;
    if numer == 0 || numer.unsigned_abs() % denom == 0 {
        return None;
    }
    Some((*base, numer, denom))
}

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
            if is_opaque_power_atom(ctx, *base, *exp) {
                return true;
            }
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
    collect_function_calls_inner(ctx, expr, &mut out, 0, max_depth, 18);
    out
}

pub fn collect_function_calls_with_pow_limit(
    ctx: &Context,
    expr: ExprId,
    max_depth: usize,
    max_pow_exp: u32,
) -> Vec<ExprId> {
    let mut out = Vec::new();
    collect_function_calls_inner(ctx, expr, &mut out, 0, max_depth, max_pow_exp);
    out
}

fn is_opaque_rational_atom(
    ctx: &Context,
    expr: ExprId,
    depth: usize,
    max_depth: usize,
    max_pow_exp: u32,
) -> bool {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return false;
    };
    if matches!(ctx.get(*den), Expr::Number(_)) {
        return false;
    }
    is_polynomial_candidate_inner(ctx, *num, depth + 1, max_depth, max_pow_exp)
        && is_polynomial_candidate_inner(ctx, *den, depth + 1, max_depth, max_pow_exp)
}

fn collect_function_calls_inner(
    ctx: &Context,
    expr: ExprId,
    out: &mut Vec<ExprId>,
    depth: usize,
    max_depth: usize,
    max_pow_exp: u32,
) {
    if depth > max_depth {
        return;
    }
    match ctx.get(expr) {
        Expr::Function(_, _) => out.push(expr),
        Expr::Pow(base, exp) if is_opaque_power_atom(ctx, *base, *exp) => out.push(expr),
        Expr::Div(_, _) if is_opaque_rational_atom(ctx, expr, depth, max_depth, max_pow_exp) => {
            out.push(expr)
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            collect_function_calls_inner(ctx, *l, out, depth + 1, max_depth, max_pow_exp);
            collect_function_calls_inner(ctx, *r, out, depth + 1, max_depth, max_pow_exp);
        }
        Expr::Pow(base, exp) => {
            collect_function_calls_inner(ctx, *base, out, depth + 1, max_depth, max_pow_exp);
            collect_function_calls_inner(ctx, *exp, out, depth + 1, max_depth, max_pow_exp);
        }
        Expr::Neg(inner) => {
            collect_function_calls_inner(ctx, *inner, out, depth + 1, max_depth, max_pow_exp)
        }
        _ => {}
    }
}

/// Collect supported opaque constants under `expr` up to `max_depth`.
pub fn collect_constant_atoms(ctx: &Context, expr: ExprId, max_depth: usize) -> Vec<ExprId> {
    let mut out = Vec::new();
    collect_constant_atoms_inner(ctx, expr, &mut out, 0, max_depth);
    out
}

fn collect_constant_atoms_inner(
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
        Expr::Constant(c) if is_opaque_constant_atom(c) => out.push(expr),
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            collect_constant_atoms_inner(ctx, *l, out, depth + 1, max_depth);
            collect_constant_atoms_inner(ctx, *r, out, depth + 1, max_depth);
        }
        Expr::Pow(base, exp) => {
            collect_constant_atoms_inner(ctx, *base, out, depth + 1, max_depth);
            collect_constant_atoms_inner(ctx, *exp, out, depth + 1, max_depth);
        }
        Expr::Function(_, args) => {
            for &arg in args {
                collect_constant_atoms_inner(ctx, arg, out, depth + 1, max_depth);
            }
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            collect_constant_atoms_inner(ctx, *inner, out, depth + 1, max_depth);
        }
        Expr::Matrix { data, .. } => {
            for &entry in data {
                collect_constant_atoms_inner(ctx, entry, out, depth + 1, max_depth);
            }
        }
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
    fn polynomial_candidate_handles_opaque_fractional_power_atoms() {
        let mut ctx = Context::new();
        let ok = parse(
            "((x^2 + 1)^(1/2) + 1)^2 - ((x^2 + 1) + 2*(x^2 + 1)^(1/2) + 1)",
            &mut ctx,
        )
        .expect("parse ok");
        assert!(is_polynomial_candidate(&ctx, ok, 30, 6));
    }

    #[test]
    fn function_call_collection_and_dedup() {
        let mut ctx = Context::new();
        let expr = parse("sin(x) + (x^2 + 1)^(1/2) + (x^2 + 1)^(1/2)", &mut ctx).expect("parse");
        let calls = collect_function_calls(&ctx, expr, 6);
        let unique = dedup_expr_ids(&ctx, &calls);
        assert_eq!(unique.len(), 2);
    }

    #[test]
    fn function_call_collection_includes_simple_rational_atoms() {
        let mut ctx = Context::new();
        let expr = parse("(u/(u+1)) + 1/(u-1)", &mut ctx).expect("parse");
        let calls = collect_function_calls_with_pow_limit(&ctx, expr, 8, 18);
        let unique = dedup_expr_ids(&ctx, &calls);
        assert_eq!(unique.len(), 2);
    }

    #[test]
    fn constant_atom_collection_and_dedup() {
        let mut ctx = Context::new();
        let expr = parse("x + pi + e + pi", &mut ctx).expect("parse");
        let constants = collect_constant_atoms(&ctx, expr, 6);
        let unique = dedup_expr_ids(&ctx, &constants);
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

    #[test]
    fn extracts_reciprocal_power_base_for_root_atom() {
        let mut ctx = Context::new();
        let expr = parse("(x^2 + 1)^(1/2)", &mut ctx).expect("parse");
        let (base, idx) = extract_opaque_reciprocal_power_base(&ctx, expr).expect("root atom");
        let expected = parse("x^2 + 1", &mut ctx).expect("expected");
        assert_eq!(compare_expr(&ctx, base, expected), Ordering::Equal);
        assert_eq!(idx, 2);
    }

    #[test]
    fn extracts_rational_power_atom_for_root_multiple() {
        let mut ctx = Context::new();
        let expr = parse("(x^2 + 1)^(3/2)", &mut ctx).expect("parse");
        let (base, numer, denom) = extract_opaque_rational_power_atom(&ctx, expr).expect("atom");
        let expected = parse("x^2 + 1", &mut ctx).expect("expected");
        assert_eq!(compare_expr(&ctx, base, expected), Ordering::Equal);
        assert_eq!((numer, denom), (3, 2));
    }

    #[test]
    fn extracts_negative_reciprocal_power_base_for_root_atom() {
        let mut ctx = Context::new();
        let expr = parse("u^(-1/2)", &mut ctx).expect("parse");
        let (base, idx) =
            extract_opaque_negative_reciprocal_power_base(&ctx, expr).expect("root atom");
        let expected = parse("u", &mut ctx).expect("expected");
        assert_eq!(compare_expr(&ctx, base, expected), Ordering::Equal);
        assert_eq!(idx, 2);
    }

    #[test]
    fn extracts_signed_rational_power_atom_for_negative_multiple() {
        let mut ctx = Context::new();
        let expr = parse("u^(-3/2)", &mut ctx).expect("parse");
        let (base, numer, denom) =
            extract_opaque_signed_rational_power_atom(&ctx, expr).expect("atom");
        let expected = parse("u", &mut ctx).expect("expected");
        assert_eq!(compare_expr(&ctx, base, expected), Ordering::Equal);
        assert_eq!((numer, denom), (-3, 2));
    }
}
