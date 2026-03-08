//! Structural polynomial GCD helpers.
//!
//! This module contains pure structural matching utilities used by the engine's
//! `poly_gcd` dispatcher and fraction-cancellation paths.

use cas_ast::views::MulBuilder;
use cas_ast::{Context, Expr, ExprId};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Compute structural GCD by intersecting factor lists.
///
/// Returns the GCD expression (or 1 if no common factors).
pub fn poly_gcd_structural(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    let a_factors = collect_mul_factors_with_exp(ctx, a);
    let b_factors = collect_mul_factors_with_exp(ctx, b);

    // Find common factors by AC-canonical key comparison.
    let mut gcd_factors: Vec<(ExprId, i64)> = Vec::new();
    let mut used_b: Vec<bool> = vec![false; b_factors.len()];

    for (a_base, a_exp) in &a_factors {
        for (j, (b_base, b_exp)) in b_factors.iter().enumerate() {
            if !used_b[j] && expr_equal_ac(ctx, *a_base, *b_base) {
                let min_exp = (*a_exp).min(*b_exp);
                if min_exp > 0 {
                    gcd_factors.push((*a_base, min_exp));
                }
                used_b[j] = true;
                break;
            }
        }
    }

    build_mul_from_factors(ctx, &gcd_factors)
}

/// Shallow GCD for fraction cancellation (stack-safe).
///
/// Returns (gcd, description) where gcd=1 means "no common factor found".
pub fn gcd_shallow_for_fraction(ctx: &mut Context, num: ExprId, den: ExprId) -> (ExprId, String) {
    let num = strip_hold(ctx, num);
    let den = strip_hold(ctx, den);

    if num == den {
        return (num, "gcd(a, a) = a".to_string());
    }

    let (num_base, num_exp) = extract_power_base_exp(ctx, num);
    let (den_base, den_exp) = extract_power_base_exp(ctx, den);

    if expr_equal_shallow(ctx, num_base, den_base) {
        let min_exp = num_exp.min(den_exp);
        if min_exp > 0 {
            let gcd = if min_exp == 1 {
                num_base
            } else {
                let exp_expr = ctx.num(min_exp);
                ctx.add(Expr::Pow(num_base, exp_expr))
            };
            return (
                gcd,
                format!(
                    "Common power base: min({}, {}) = {}",
                    num_exp, den_exp, min_exp
                ),
            );
        }
    }

    (ctx.num(1), "No common factor (shallow)".to_string())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ExprKey(u64);

fn expr_key_ac(ctx: &Context, expr: ExprId) -> ExprKey {
    let mut hasher = DefaultHasher::new();
    expr_key_hash(ctx, expr, &mut hasher);
    ExprKey(hasher.finish())
}

fn expr_key_hash<H: Hasher>(ctx: &Context, expr: ExprId, hasher: &mut H) {
    match ctx.get(expr) {
        Expr::Add(_, _) => {
            let mut keys: Vec<ExprKey> = add_terms_no_sign(ctx, expr)
                .into_iter()
                .map(|e| expr_key_ac(ctx, e))
                .collect();
            keys.sort();

            "Add".hash(hasher);
            keys.len().hash(hasher);
            for key in keys {
                key.hash(hasher);
            }
        }
        Expr::Mul(_, _) => {
            let mut keys: Vec<ExprKey> = mul_factors(ctx, expr)
                .into_iter()
                .map(|e| expr_key_ac(ctx, e))
                .collect();
            keys.sort();

            "Mul".hash(hasher);
            keys.len().hash(hasher);
            for key in keys {
                key.hash(hasher);
            }
        }
        Expr::Pow(base, exp) => {
            "Pow".hash(hasher);
            expr_key_ac(ctx, *base).hash(hasher);
            expr_key_ac(ctx, *exp).hash(hasher);
        }
        Expr::Neg(inner) => {
            "Neg".hash(hasher);
            expr_key_ac(ctx, *inner).hash(hasher);
        }
        Expr::Sub(a, b) => {
            "Add".hash(hasher);
            2usize.hash(hasher);
            let mut keys = vec![expr_key_ac(ctx, *a), expr_key_neg(ctx, *b)];
            keys.sort();
            for key in keys {
                key.hash(hasher);
            }
        }
        Expr::Div(a, b) => {
            "Div".hash(hasher);
            expr_key_ac(ctx, *a).hash(hasher);
            expr_key_ac(ctx, *b).hash(hasher);
        }
        Expr::Number(n) => {
            "Number".hash(hasher);
            n.numer().hash(hasher);
            n.denom().hash(hasher);
        }
        Expr::Variable(name) => {
            "Variable".hash(hasher);
            name.hash(hasher);
        }
        Expr::Constant(c) => {
            "Constant".hash(hasher);
            format!("{:?}", c).hash(hasher);
        }
        Expr::Function(name, args) => {
            "Function".hash(hasher);
            name.hash(hasher);
            args.len().hash(hasher);
            for arg in args {
                expr_key_ac(ctx, *arg).hash(hasher);
            }
        }
        Expr::Matrix { rows, cols, data } => {
            "Matrix".hash(hasher);
            rows.hash(hasher);
            cols.hash(hasher);
            for d in data {
                expr_key_ac(ctx, *d).hash(hasher);
            }
        }
        Expr::SessionRef(id) => {
            "SessionRef".hash(hasher);
            id.hash(hasher);
        }
        Expr::Hold(inner) => {
            "Hold".hash(hasher);
            expr_key_ac(ctx, *inner).hash(hasher);
        }
    }
}

fn expr_key_neg(ctx: &Context, expr: ExprId) -> ExprKey {
    let mut hasher = DefaultHasher::new();
    "Neg".hash(&mut hasher);
    expr_key_ac(ctx, expr).hash(&mut hasher);
    ExprKey(hasher.finish())
}

fn expr_equal_ac(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    expr_key_ac(ctx, a) == expr_key_ac(ctx, b)
}

fn strip_hold(ctx: &Context, mut expr: ExprId) -> ExprId {
    loop {
        let unwrapped = cas_ast::hold::unwrap_hold(ctx, expr);
        if unwrapped == expr {
            return expr;
        }
        expr = unwrapped;
    }
}

fn collect_mul_factors_with_exp(ctx: &Context, expr: ExprId) -> Vec<(ExprId, i64)> {
    let expr = strip_hold(ctx, expr);
    let mut factors = Vec::new();
    collect_mul_factors_rec(ctx, expr, 1, &mut factors);
    factors
}

fn collect_mul_factors_rec(
    ctx: &Context,
    expr: ExprId,
    mult: i64,
    factors: &mut Vec<(ExprId, i64)>,
) {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            collect_mul_factors_rec(ctx, *left, mult, factors);
            collect_mul_factors_rec(ctx, *right, mult, factors);
        }
        Expr::Pow(base, exp) => {
            if let Some(k) = get_integer_exp(ctx, *exp) {
                if k > 0 {
                    factors.push((*base, mult * k));
                } else {
                    factors.push((expr, mult));
                }
            } else {
                factors.push((expr, mult));
            }
        }
        _ => factors.push((expr, mult)),
    }
}

fn get_integer_exp(ctx: &Context, exp: ExprId) -> Option<i64> {
    match ctx.get(exp) {
        Expr::Number(n) => {
            if n.is_integer() {
                n.to_integer().try_into().ok()
            } else {
                None
            }
        }
        Expr::Neg(inner) => get_integer_exp(ctx, *inner).map(|k| -k),
        _ => None,
    }
}

fn build_mul_from_factors(ctx: &mut Context, factors: &[(ExprId, i64)]) -> ExprId {
    let mut builder = MulBuilder::new_simple();
    for &(base, exp) in factors {
        if exp > 0 {
            builder.push_pow(base, exp);
        }
    }
    builder.build(ctx)
}

fn expr_equal_shallow(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    if a == b {
        return true;
    }

    match (ctx.get(a), ctx.get(b)) {
        (Expr::Number(n1), Expr::Number(n2)) => n1 == n2,
        (Expr::Variable(v1), Expr::Variable(v2)) => v1 == v2,
        (Expr::Add(l1, r1), Expr::Add(l2, r2)) => (l1 == l2 && r1 == r2) || (l1 == r2 && r1 == l2),
        (Expr::Mul(l1, r1), Expr::Mul(l2, r2)) => (l1 == l2 && r1 == r2) || (l1 == r2 && r1 == l2),
        (Expr::Sub(l1, r1), Expr::Sub(l2, r2)) => l1 == l2 && r1 == r2,
        (Expr::Div(n1, d1), Expr::Div(n2, d2)) => n1 == n2 && d1 == d2,
        (Expr::Pow(b1, e1), Expr::Pow(b2, e2)) => b1 == b2 && e1 == e2,
        (Expr::Neg(i1), Expr::Neg(i2)) => i1 == i2,
        _ => false,
    }
}

fn extract_power_base_exp(ctx: &Context, expr: ExprId) -> (ExprId, i64) {
    let expr = strip_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            if let Some(k) = get_integer_exp(ctx, *exp) {
                if k > 0 {
                    return (*base, k);
                }
            }
            (expr, 1)
        }
        _ => (expr, 1),
    }
}

fn add_terms_no_sign(ctx: &Context, root: ExprId) -> Vec<ExprId> {
    let mut out = Vec::new();
    let mut stack = vec![root];

    while let Some(id) = stack.pop() {
        if crate::poly_result::is_poly_ref_or_result(ctx, id) {
            out.push(id);
            continue;
        }

        let id = cas_ast::hold::unwrap_hold(ctx, id);
        match ctx.get(id) {
            Expr::Add(l, r) => {
                stack.push(*r);
                stack.push(*l);
            }
            Expr::Sub(l, r) => {
                stack.push(*r);
                stack.push(*l);
            }
            Expr::Neg(inner) => stack.push(*inner),
            _ => out.push(id),
        }
    }

    out
}

fn mul_factors(ctx: &Context, root: ExprId) -> Vec<ExprId> {
    let mut out = Vec::new();
    let mut stack = vec![root];

    while let Some(id) = stack.pop() {
        if crate::poly_result::is_poly_ref_or_result(ctx, id) {
            out.push(id);
            continue;
        }

        let id = cas_ast::hold::unwrap_hold(ctx, id);
        match ctx.get(id) {
            Expr::Mul(l, r) => {
                stack.push(*r);
                stack.push(*l);
            }
            _ => out.push(id),
        }
    }

    out
}
