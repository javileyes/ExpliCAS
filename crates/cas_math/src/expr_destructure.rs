//! Zero-clone expression destructuring helpers.
//!
//! These helpers expose borrowed AST shape checks without forcing callers to
//! clone `Expr` nodes.

use cas_ast::{symbol::SymbolId, Context, Expr, ExprId};

/// Destruct `Add(l, r)`.
#[inline]
pub fn as_add(ctx: &Context, id: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(id) {
        Expr::Add(l, r) => Some((*l, *r)),
        _ => None,
    }
}

/// Destruct `Sub(l, r)`.
#[inline]
pub fn as_sub(ctx: &Context, id: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(id) {
        Expr::Sub(l, r) => Some((*l, *r)),
        _ => None,
    }
}

/// Destruct `Mul(l, r)`.
#[inline]
pub fn as_mul(ctx: &Context, id: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(id) {
        Expr::Mul(l, r) => Some((*l, *r)),
        _ => None,
    }
}

/// Destruct `Div(l, r)`.
#[inline]
pub fn as_div(ctx: &Context, id: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(id) {
        Expr::Div(l, r) => Some((*l, *r)),
        _ => None,
    }
}

/// Destruct `Pow(base, exp)`.
#[inline]
pub fn as_pow(ctx: &Context, id: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(id) {
        Expr::Pow(base, exp) => Some((*base, *exp)),
        _ => None,
    }
}

/// Destruct `Neg(inner)`.
#[inline]
pub fn as_neg(ctx: &Context, id: ExprId) -> Option<ExprId> {
    match ctx.get(id) {
        Expr::Neg(inner) => Some(*inner),
        _ => None,
    }
}

/// Match one-argument function by name and return its argument.
#[inline]
pub fn as_fn1(ctx: &Context, id: ExprId, name: &str) -> Option<ExprId> {
    match ctx.get(id) {
        Expr::Function(fn_id, args) if ctx.sym_name(*fn_id) == name && args.len() == 1 => {
            Some(args[0])
        }
        _ => None,
    }
}

/// Destruct `Function(fn_id, args)` as `(fn_id, &[args])`.
#[inline]
pub fn as_function(ctx: &Context, id: ExprId) -> Option<(SymbolId, &[ExprId])> {
    match ctx.get(id) {
        Expr::Function(fn_id, args) => Some((*fn_id, args.as_slice())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn binary_destructure_works() {
        let mut ctx = Context::new();
        let add = parse("a+b", &mut ctx).expect("parse");
        let sub = parse("a-b", &mut ctx).expect("parse");
        let mul = parse("a*b", &mut ctx).expect("parse");
        let div = parse("a/b", &mut ctx).expect("parse");
        let pow = parse("a^3", &mut ctx).expect("parse");

        assert!(as_add(&ctx, add).is_some());
        assert!(as_sub(&ctx, sub).is_some());
        assert!(as_mul(&ctx, mul).is_some());
        assert!(as_div(&ctx, div).is_some());
        assert!(as_pow(&ctx, pow).is_some());
        assert!(as_add(&ctx, pow).is_none());
    }

    #[test]
    fn unary_and_function_destructure_works() {
        let mut ctx = Context::new();
        let neg = parse("-x", &mut ctx).expect("parse");
        let sin = parse("sin(x)", &mut ctx).expect("parse");

        assert!(as_neg(&ctx, neg).is_some());
        assert!(as_fn1(&ctx, sin, "sin").is_some());
        assert!(as_fn1(&ctx, sin, "cos").is_none());
        assert!(as_function(&ctx, sin).is_some());
    }
}
