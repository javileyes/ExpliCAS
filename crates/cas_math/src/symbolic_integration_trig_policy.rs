//! General trig source-side policy helpers for symbolic integration.
//!
//! This module owns detectors that apply to ordinary `sin(u)`/`cos(u)`
//! integration routes. Reciprocal powers and reciprocal derivative products
//! stay in `symbolic_integration_reciprocal_trig_policy`.

use crate::expr_nary::Sign;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};

pub(crate) fn trig_like_factor(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    match ctx.builtin_of(*fn_id) {
        Some(builtin @ (BuiltinFn::Sin | BuiltinFn::Cos)) => Some((builtin, args[0])),
        _ => None,
    }
}

pub(crate) fn signed_trig_like_factor(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId, Sign, ExprId)> {
    match ctx.get(expr) {
        Expr::Neg(inner) => {
            trig_like_factor(ctx, *inner).map(|(builtin, arg)| (builtin, arg, Sign::Neg, *inner))
        }
        _ => trig_like_factor(ctx, expr).map(|(builtin, arg)| (builtin, arg, Sign::Pos, expr)),
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
    fn detects_unsigned_trig_like_factors() {
        let mut ctx = Context::new();
        let sin = parse("sin(2*x+1)", &mut ctx).unwrap();
        let cos = parse("cos(2*x+1)", &mut ctx).unwrap();
        let tan = parse("tan(2*x+1)", &mut ctx).unwrap();
        let signed_sin = parse("-sin(2*x+1)", &mut ctx).unwrap();

        let (sin_builtin, sin_arg) = trig_like_factor(&ctx, sin).unwrap();
        assert_eq!(sin_builtin, BuiltinFn::Sin);
        assert_eq!(rendered(&ctx, sin_arg), "2 * x + 1");

        let (cos_builtin, cos_arg) = trig_like_factor(&ctx, cos).unwrap();
        assert_eq!(cos_builtin, BuiltinFn::Cos);
        assert_eq!(rendered(&ctx, cos_arg), "2 * x + 1");

        assert!(trig_like_factor(&ctx, tan).is_none());
        assert!(trig_like_factor(&ctx, signed_sin).is_none());
    }

    #[test]
    fn detects_signed_trig_like_factors() {
        let mut ctx = Context::new();
        let sin = parse("sin(2*x+1)", &mut ctx).unwrap();
        let signed_sin = parse("-sin(2*x+1)", &mut ctx).unwrap();
        let signed_cos = parse("-cos(2*x+1)", &mut ctx).unwrap();
        let signed_tan = parse("-tan(2*x+1)", &mut ctx).unwrap();

        let (builtin, arg, sign, factor) = signed_trig_like_factor(&ctx, sin).unwrap();
        assert_eq!(builtin, BuiltinFn::Sin);
        assert_eq!(rendered(&ctx, arg), "2 * x + 1");
        assert_eq!(sign, Sign::Pos);
        assert_eq!(rendered(&ctx, factor), "sin(2 * x + 1)");

        let (builtin, arg, sign, factor) = signed_trig_like_factor(&ctx, signed_sin).unwrap();
        assert_eq!(builtin, BuiltinFn::Sin);
        assert_eq!(rendered(&ctx, arg), "2 * x + 1");
        assert_eq!(sign, Sign::Neg);
        assert_eq!(rendered(&ctx, factor), "sin(2 * x + 1)");

        let (builtin, arg, sign, factor) = signed_trig_like_factor(&ctx, signed_cos).unwrap();
        assert_eq!(builtin, BuiltinFn::Cos);
        assert_eq!(rendered(&ctx, arg), "2 * x + 1");
        assert_eq!(sign, Sign::Neg);
        assert_eq!(rendered(&ctx, factor), "cos(2 * x + 1)");

        assert!(signed_trig_like_factor(&ctx, signed_tan).is_none());
    }
}
