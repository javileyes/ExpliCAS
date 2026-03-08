//! Eager expansion pre-pass helpers.
//!
//! This module contains the AST traversal used by runtime crates to eagerly
//! evaluate heavy `expand(...)` calls before full simplification.

use cas_ast::{Context, Expr, ExprId};

/// Default threshold for eager mod-p expansion in pre-pass.
pub const DEFAULT_EAGER_EXPAND_MODP_THRESHOLD: u64 = 500;
/// Default max terms for eager AST materialization before returning poly refs.
pub const DEFAULT_EAGER_EXPAND_MATERIALIZE_LIMIT: usize = 1_000;

/// Eagerly evaluate `expand(arg)` calls whose estimated size is above `threshold`.
///
/// For matching calls, expansion uses the mod-p/poly-store path:
/// `expand_expr_modp_to_poly_ref_or_hold(ctx, arg, materialize_limit)`.
///
/// Traversal is top-down: once an `expand(...)` call is rewritten, recursion does
/// not continue inside that rewritten subtree.
pub fn eager_eval_expand_calls_with<Item, FBuildItem>(
    ctx: &mut Context,
    expr: ExprId,
    include_items: bool,
    threshold: u64,
    materialize_limit: usize,
    mut build_item: FBuildItem,
) -> (ExprId, Vec<Item>)
where
    FBuildItem: FnMut(&Context, u64, ExprId, ExprId) -> Item,
{
    let mut items = Vec::new();
    let result = eager_eval_expand_recursive(
        ctx,
        expr,
        include_items,
        threshold,
        materialize_limit,
        &mut items,
        &mut build_item,
    );
    (result, items)
}

/// Eagerly evaluate `expand(arg)` calls using crate default thresholds.
pub fn eager_eval_expand_calls_default_with<Item, FBuildItem>(
    ctx: &mut Context,
    expr: ExprId,
    include_items: bool,
    build_item: FBuildItem,
) -> (ExprId, Vec<Item>)
where
    FBuildItem: FnMut(&Context, u64, ExprId, ExprId) -> Item,
{
    eager_eval_expand_calls_with(
        ctx,
        expr,
        include_items,
        DEFAULT_EAGER_EXPAND_MODP_THRESHOLD,
        DEFAULT_EAGER_EXPAND_MATERIALIZE_LIMIT,
        build_item,
    )
}

fn eager_eval_expand_recursive<Item, FBuildItem>(
    ctx: &mut Context,
    expr: ExprId,
    include_items: bool,
    threshold: u64,
    materialize_limit: usize,
    items: &mut Vec<Item>,
    build_item: &mut FBuildItem,
) -> ExprId
where
    FBuildItem: FnMut(&Context, u64, ExprId, ExprId) -> Item,
{
    if let Some((estimate, rewritten)) =
        try_eval_expand_call(ctx, expr, threshold, materialize_limit)
    {
        if include_items {
            items.push(build_item(ctx, estimate, expr, rewritten));
        }
        return rewritten;
    }

    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        let fn_id = *fn_id;
        let args = args.clone();
        let new_args: Vec<ExprId> = args
            .iter()
            .map(|&arg| {
                eager_eval_expand_recursive(
                    ctx,
                    arg,
                    include_items,
                    threshold,
                    materialize_limit,
                    items,
                    build_item,
                )
            })
            .collect();
        if new_args
            .iter()
            .zip(args.iter())
            .any(|(new, old)| new != old)
        {
            return ctx.add(Expr::Function(fn_id, new_args));
        }
        return expr;
    }

    enum Recurse {
        Binary(ExprId, ExprId, u8), // 0=Add, 1=Sub, 2=Mul, 3=Div, 4=Pow
        Unary(ExprId),              // Neg
        Leaf,
    }

    let recurse = match ctx.get(expr) {
        Expr::Add(l, r) => Recurse::Binary(*l, *r, 0),
        Expr::Sub(l, r) => Recurse::Binary(*l, *r, 1),
        Expr::Mul(l, r) => Recurse::Binary(*l, *r, 2),
        Expr::Div(l, r) => Recurse::Binary(*l, *r, 3),
        Expr::Pow(b, e) => Recurse::Binary(*b, *e, 4),
        Expr::Neg(inner) => Recurse::Unary(*inner),
        // Hold is intentionally a leaf: expansion must not cross hold boundary.
        Expr::Hold(_)
        | Expr::Number(_)
        | Expr::Variable(_)
        | Expr::Constant(_)
        | Expr::Matrix { .. }
        | Expr::SessionRef(_)
        | Expr::Function(_, _) => Recurse::Leaf,
    };

    match recurse {
        Recurse::Binary(l, r, op) => {
            let nl = eager_eval_expand_recursive(
                ctx,
                l,
                include_items,
                threshold,
                materialize_limit,
                items,
                build_item,
            );
            let nr = eager_eval_expand_recursive(
                ctx,
                r,
                include_items,
                threshold,
                materialize_limit,
                items,
                build_item,
            );
            if nl != l || nr != r {
                match op {
                    0 => ctx.add(Expr::Add(nl, nr)),
                    1 => ctx.add(Expr::Sub(nl, nr)),
                    2 => ctx.add(Expr::Mul(nl, nr)),
                    3 => ctx.add(Expr::Div(nl, nr)),
                    _ => ctx.add(Expr::Pow(nl, nr)),
                }
            } else {
                expr
            }
        }
        Recurse::Unary(inner) => {
            let ni = eager_eval_expand_recursive(
                ctx,
                inner,
                include_items,
                threshold,
                materialize_limit,
                items,
                build_item,
            );
            if ni != inner {
                ctx.add(Expr::Neg(ni))
            } else {
                expr
            }
        }
        Recurse::Leaf => expr,
    }
}

fn try_eval_expand_call(
    ctx: &mut Context,
    expr: ExprId,
    threshold: u64,
    materialize_limit: usize,
) -> Option<(u64, ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };

    if ctx.sym_name(*fn_id) != "expand" || args.len() != 1 {
        return None;
    }

    let arg = args[0];
    let estimate = crate::expand_estimate::estimate_expand_terms(ctx, arg)?;
    if estimate <= threshold {
        return None;
    }

    let rewritten =
        crate::poly_store::expand_expr_modp_to_poly_ref_or_hold(ctx, arg, materialize_limit)?;
    Some((estimate, rewritten))
}

#[cfg(test)]
mod tests {
    use super::{
        eager_eval_expand_calls_default_with, eager_eval_expand_calls_with,
        DEFAULT_EAGER_EXPAND_MATERIALIZE_LIMIT, DEFAULT_EAGER_EXPAND_MODP_THRESHOLD,
    };
    use cas_ast::Expr;
    use cas_parser::parse;

    #[test]
    fn eager_eval_expand_calls_rewrites_expand_call_when_threshold_is_low() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("expand((x+1)^2)", &mut ctx).expect("parse expand call");

        let (rewritten, items) =
            eager_eval_expand_calls_with(&mut ctx, expr, true, 0, usize::MAX, |_ctx, est, b, a| {
                (est, b, a)
            });

        assert_ne!(rewritten, expr);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].1, expr);
        assert_eq!(items[0].2, rewritten);
    }

    #[test]
    fn eager_eval_expand_calls_keeps_tree_when_threshold_not_met() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("expand((x+1)^2)", &mut ctx).expect("parse expand call");

        let (rewritten, items) = eager_eval_expand_calls_with(
            &mut ctx,
            expr,
            true,
            u64::MAX,
            usize::MAX,
            |_ctx, _, _, _| (),
        );

        assert_eq!(rewritten, expr);
        assert!(items.is_empty());
    }

    #[test]
    fn eager_eval_expand_calls_rewrites_nested_expand_calls() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("1 + expand((x+1)^2)", &mut ctx).expect("parse expression");

        let (rewritten, items) =
            eager_eval_expand_calls_with(&mut ctx, expr, true, 0, usize::MAX, |_ctx, est, _, _| {
                est
            });

        assert!(matches!(ctx.get(rewritten), Expr::Add(_, _)));
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn eager_default_wrapper_uses_defaults() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("expand((x+1)^2)", &mut ctx).expect("parse expand call");

        let (default_rewritten, default_items) =
            eager_eval_expand_calls_default_with(&mut ctx, expr, true, |_ctx, est, b, a| {
                (est, b, a)
            });
        let (explicit_rewritten, explicit_items) = eager_eval_expand_calls_with(
            &mut ctx,
            expr,
            true,
            DEFAULT_EAGER_EXPAND_MODP_THRESHOLD,
            DEFAULT_EAGER_EXPAND_MATERIALIZE_LIMIT,
            |_ctx, est, b, a| (est, b, a),
        );

        assert_eq!(default_rewritten, explicit_rewritten);
        assert_eq!(default_items.len(), explicit_items.len());
    }
}
