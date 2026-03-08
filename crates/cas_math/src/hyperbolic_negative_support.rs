use crate::expr_complexity::dedup_node_count_within;
use crate::expr_sub_like::extract_sub_like_pair;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyperbolicNegativeRewrite {
    pub rewritten: ExprId,
    pub kind: HyperbolicNegativeRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HyperbolicNegativeRewriteKind {
    SinhExplicitNeg,
    CoshExplicitNeg,
    TanhExplicitNeg,
    AsinhExplicitNeg,
    AtanhExplicitNeg,
    SinhCanonicalSub,
    CoshCanonicalSub,
    TanhCanonicalSub,
    AsinhCanonicalSub,
    AtanhCanonicalSub,
}

/// Try to rewrite negative-argument hyperbolic forms.
///
/// Supported patterns:
/// - `f(-x)` for odd/even hyperbolic functions
/// - `f(a-b)` when canonical order says `a < b` (rewritten via `b-a`)
pub fn try_rewrite_hyperbolic_negative_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HyperbolicNegativeRewrite> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    let fn_id = *fn_id;
    let args = args.clone();
    if args.len() != 1 {
        return None;
    }
    let arg = args[0];

    // Case 1: explicit negation, f(-x).
    if let Expr::Neg(inner) = ctx.get(arg) {
        return rewrite_for_builtin(ctx, fn_id, *inner, RewritePattern::ExplicitNeg);
    }

    // Case 2: subtraction-like argument, f(a-b) with canonical swap to f(b-a).
    let (a, b) = extract_sub_like_pair(ctx, arg)?;
    if !dedup_node_count_within(ctx, a, 20) || !dedup_node_count_within(ctx, b, 20) {
        return None;
    }
    if compare_expr(ctx, a, b) != Ordering::Less {
        return None;
    }

    let canonical_arg = ctx.add(Expr::Sub(b, a));
    rewrite_for_builtin(ctx, fn_id, canonical_arg, RewritePattern::CanonicalSub)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RewritePattern {
    ExplicitNeg,
    CanonicalSub,
}

fn rewrite_for_builtin(
    ctx: &mut Context,
    fn_id: usize,
    inner: ExprId,
    pattern: RewritePattern,
) -> Option<HyperbolicNegativeRewrite> {
    let builtin = ctx.builtin_of(fn_id)?;
    match (builtin, pattern) {
        (BuiltinFn::Sinh, RewritePattern::ExplicitNeg) => {
            let sinh_inner = ctx.call_builtin(BuiltinFn::Sinh, vec![inner]);
            Some(HyperbolicNegativeRewrite {
                rewritten: ctx.add(Expr::Neg(sinh_inner)),
                kind: HyperbolicNegativeRewriteKind::SinhExplicitNeg,
            })
        }
        (BuiltinFn::Cosh, RewritePattern::ExplicitNeg) => Some(HyperbolicNegativeRewrite {
            rewritten: ctx.call_builtin(BuiltinFn::Cosh, vec![inner]),
            kind: HyperbolicNegativeRewriteKind::CoshExplicitNeg,
        }),
        (BuiltinFn::Tanh, RewritePattern::ExplicitNeg) => {
            let tanh_inner = ctx.call_builtin(BuiltinFn::Tanh, vec![inner]);
            Some(HyperbolicNegativeRewrite {
                rewritten: ctx.add(Expr::Neg(tanh_inner)),
                kind: HyperbolicNegativeRewriteKind::TanhExplicitNeg,
            })
        }
        (BuiltinFn::Asinh, RewritePattern::ExplicitNeg) => {
            let asinh_inner = ctx.call_builtin(BuiltinFn::Asinh, vec![inner]);
            Some(HyperbolicNegativeRewrite {
                rewritten: ctx.add(Expr::Neg(asinh_inner)),
                kind: HyperbolicNegativeRewriteKind::AsinhExplicitNeg,
            })
        }
        (BuiltinFn::Atanh, RewritePattern::ExplicitNeg) => {
            let atanh_inner = ctx.call_builtin(BuiltinFn::Atanh, vec![inner]);
            Some(HyperbolicNegativeRewrite {
                rewritten: ctx.add(Expr::Neg(atanh_inner)),
                kind: HyperbolicNegativeRewriteKind::AtanhExplicitNeg,
            })
        }
        (BuiltinFn::Sinh, RewritePattern::CanonicalSub) => {
            let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![inner]);
            Some(HyperbolicNegativeRewrite {
                rewritten: ctx.add(Expr::Neg(sinh)),
                kind: HyperbolicNegativeRewriteKind::SinhCanonicalSub,
            })
        }
        (BuiltinFn::Cosh, RewritePattern::CanonicalSub) => Some(HyperbolicNegativeRewrite {
            rewritten: ctx.call_builtin(BuiltinFn::Cosh, vec![inner]),
            kind: HyperbolicNegativeRewriteKind::CoshCanonicalSub,
        }),
        (BuiltinFn::Tanh, RewritePattern::CanonicalSub) => {
            let tanh = ctx.call_builtin(BuiltinFn::Tanh, vec![inner]);
            Some(HyperbolicNegativeRewrite {
                rewritten: ctx.add(Expr::Neg(tanh)),
                kind: HyperbolicNegativeRewriteKind::TanhCanonicalSub,
            })
        }
        (BuiltinFn::Asinh, RewritePattern::CanonicalSub) => {
            let asinh = ctx.call_builtin(BuiltinFn::Asinh, vec![inner]);
            Some(HyperbolicNegativeRewrite {
                rewritten: ctx.add(Expr::Neg(asinh)),
                kind: HyperbolicNegativeRewriteKind::AsinhCanonicalSub,
            })
        }
        (BuiltinFn::Atanh, RewritePattern::CanonicalSub) => {
            let atanh = ctx.call_builtin(BuiltinFn::Atanh, vec![inner]);
            Some(HyperbolicNegativeRewrite {
                rewritten: ctx.add(Expr::Neg(atanh)),
                kind: HyperbolicNegativeRewriteKind::AtanhCanonicalSub,
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::try_rewrite_hyperbolic_negative_expr;
    use cas_ast::{BuiltinFn, Context, Expr};

    #[test]
    fn rewrites_explicit_negative_sinh() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let neg_x = ctx.add(Expr::Neg(x));
        let expr = ctx.call_builtin(BuiltinFn::Sinh, vec![neg_x]);

        assert!(try_rewrite_hyperbolic_negative_expr(&mut ctx, expr).is_some());
    }

    #[test]
    fn rewrites_explicit_negative_cosh() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let neg_x = ctx.add(Expr::Neg(x));
        let expr = ctx.call_builtin(BuiltinFn::Cosh, vec![neg_x]);

        assert!(try_rewrite_hyperbolic_negative_expr(&mut ctx, expr).is_some());
    }

    #[test]
    fn rewrites_subtraction_like_argument_when_ordered() {
        let mut ctx = Context::new();
        let u = ctx.var("u");
        let one = ctx.num(1);
        let sub = ctx.add(Expr::Sub(one, u));
        let expr = ctx.call_builtin(BuiltinFn::Sinh, vec![sub]);

        assert!(try_rewrite_hyperbolic_negative_expr(&mut ctx, expr).is_some());
    }
}
