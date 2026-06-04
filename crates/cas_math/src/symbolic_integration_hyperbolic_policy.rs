//! General hyperbolic source-side policy helpers for symbolic integration.
//!
//! This module owns detectors that apply to ordinary `sinh(u)`/`cosh(u)`
//! integration routes. Reciprocal powers and reciprocal derivative products
//! stay in `symbolic_integration_hyperbolic_reciprocal_policy`.

use crate::expr_nary::Sign;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};

pub(crate) fn hyperbolic_like_factor(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    match ctx.builtin_of(*fn_id) {
        Some(builtin @ (BuiltinFn::Sinh | BuiltinFn::Cosh)) => Some((builtin, args[0])),
        _ => None,
    }
}

pub(crate) fn signed_hyperbolic_like_factor(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId, Sign, ExprId)> {
    match ctx.get(expr) {
        Expr::Neg(inner) => hyperbolic_like_factor(ctx, *inner)
            .map(|(builtin, arg)| (builtin, arg, Sign::Neg, *inner)),
        _ => {
            hyperbolic_like_factor(ctx, expr).map(|(builtin, arg)| (builtin, arg, Sign::Pos, expr))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn detects_unsigned_hyperbolic_like_factors() {
        let mut ctx = Context::new();
        let sinh = parse("sinh(2*x+1)", &mut ctx).unwrap();
        let cosh = parse("cosh(2*x+1)", &mut ctx).unwrap();
        let tanh = parse("tanh(2*x+1)", &mut ctx).unwrap();
        let signed_sinh = parse("-sinh(2*x+1)", &mut ctx).unwrap();

        let (sinh_builtin, sinh_arg) = hyperbolic_like_factor(&ctx, sinh).unwrap();
        assert_eq!(sinh_builtin, BuiltinFn::Sinh);
        assert_eq!(rendered(&ctx, sinh_arg), "2 * x + 1");

        let (cosh_builtin, cosh_arg) = hyperbolic_like_factor(&ctx, cosh).unwrap();
        assert_eq!(cosh_builtin, BuiltinFn::Cosh);
        assert_eq!(rendered(&ctx, cosh_arg), "2 * x + 1");

        assert!(hyperbolic_like_factor(&ctx, tanh).is_none());
        assert!(hyperbolic_like_factor(&ctx, signed_sinh).is_none());
    }

    #[test]
    fn detects_signed_hyperbolic_like_factors() {
        let mut ctx = Context::new();
        let sinh = parse("sinh(2*x+1)", &mut ctx).unwrap();
        let signed_sinh = parse("-sinh(2*x+1)", &mut ctx).unwrap();
        let signed_cosh = parse("-cosh(2*x+1)", &mut ctx).unwrap();
        let signed_tanh = parse("-tanh(2*x+1)", &mut ctx).unwrap();

        let (builtin, arg, sign, factor) = signed_hyperbolic_like_factor(&ctx, sinh).unwrap();
        assert_eq!(builtin, BuiltinFn::Sinh);
        assert_eq!(rendered(&ctx, arg), "2 * x + 1");
        assert_eq!(sign, Sign::Pos);
        assert_eq!(rendered(&ctx, factor), "sinh(2 * x + 1)");

        let (builtin, arg, sign, factor) =
            signed_hyperbolic_like_factor(&ctx, signed_sinh).unwrap();
        assert_eq!(builtin, BuiltinFn::Sinh);
        assert_eq!(rendered(&ctx, arg), "2 * x + 1");
        assert_eq!(sign, Sign::Neg);
        assert_eq!(rendered(&ctx, factor), "sinh(2 * x + 1)");

        let (builtin, arg, sign, factor) =
            signed_hyperbolic_like_factor(&ctx, signed_cosh).unwrap();
        assert_eq!(builtin, BuiltinFn::Cosh);
        assert_eq!(rendered(&ctx, arg), "2 * x + 1");
        assert_eq!(sign, Sign::Neg);
        assert_eq!(rendered(&ctx, factor), "cosh(2 * x + 1)");

        assert!(signed_hyperbolic_like_factor(&ctx, signed_tanh).is_none());
    }
}
