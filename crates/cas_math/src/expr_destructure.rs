//! Zero-clone expression destructuring helpers.
//!
//! These helpers expose borrowed AST shape checks without forcing callers to
//! clone `Expr` nodes.

use cas_ast::{BuiltinFn, Context, Expr, ExprId};

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
pub(crate) fn as_fn1(ctx: &Context, id: ExprId, name: &str) -> Option<ExprId> {
    match ctx.get(id) {
        Expr::Function(fn_id, args) if ctx.sym_name(*fn_id) == name && args.len() == 1 => {
            Some(args[0])
        }
        _ => None,
    }
}

/// Destruct a unary call to `builtin`, seeing through a `__hold` barrier.
///
/// A deliberate exception to this module's convention: the `as_*` helpers
/// above are pure shape checks, while this one strips `__hold` first, because
/// every caller wants the held tree to match the same way the unheld one does.
/// The name says so; four copies of this logic used to live scattered under
/// the bare name `unary_builtin_arg`.
#[inline]
pub fn unary_builtin_arg_through_hold(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    (args.len() == 1 && ctx.builtin_of(*fn_id) == Some(builtin)).then_some(args[0])
}

/// Destruct a unary call to `builtin` WITHOUT stripping `__hold`.
///
/// The counterpart to [`unary_builtin_arg_through_hold`]. The integration
/// policies use this one, so a held tree does NOT match here.
///
/// DECIDED (2026-08-01, closes the L13 open question): this is the CORRECT
/// default for integration policies, not an oversight. Evidence:
/// * `integrate_symbolic_expr` never unwraps holds at entry, so a held node at
///   a match point yields an UNEVALUATED integral — a visible, safe residual,
///   never a wrong answer.
/// * In realistic pipelines holds dissolve before reaching the backend:
///   `__hold` is transparent to Add/Mul views, so distribution strips it (the
///   `integrate(cos(x)*expand((sin(x)+1)^2), x)` probe integrates correctly).
/// * Where holds DO arrive, the integration code already unwraps them at the
///   observed seams (`general.rs`, `by_parts.rs`, `logs_exp.rs`) — transparency
///   is opt-in per seam, matching how holds are handled engine-wide.
/// * A blanket see-through default would be RISKIER: matching through a hold
///   and rebuilding from the inner nodes silently drops the barrier that
///   expand/factor installed to stop known manglings
///   (see `definite_integration.rs`, "known to mangle without the `__hold`
///   barrier").
///
/// So: use this variant by default in integration matchers; reach for
/// [`unary_builtin_arg_through_hold`] only at seams where a held tree has been
/// OBSERVED, and mind the barrier you are removing when you rebuild.
#[inline]
pub fn unary_builtin_arg_no_hold(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(builtin) =>
        {
            Some(args[0])
        }
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
}
