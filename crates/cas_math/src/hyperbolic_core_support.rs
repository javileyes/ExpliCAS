use crate::expr_predicates::{is_one_expr, is_zero_expr};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyperbolicCoreRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

/// Evaluate hyperbolic function values at special constants.
pub fn try_eval_hyperbolic_special_value(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HyperbolicCoreRewrite> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    let fn_id = *fn_id;
    let args = args.clone();
    if args.len() != 1 {
        return None;
    }
    let arg = args[0];

    match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Sinh) if is_zero_expr(ctx, arg) => Some(HyperbolicCoreRewrite {
            rewritten: ctx.num(0),
            desc: "sinh(0) = 0",
        }),
        Some(BuiltinFn::Tanh) if is_zero_expr(ctx, arg) => Some(HyperbolicCoreRewrite {
            rewritten: ctx.num(0),
            desc: "tanh(0) = 0",
        }),
        Some(BuiltinFn::Cosh) if is_zero_expr(ctx, arg) => Some(HyperbolicCoreRewrite {
            rewritten: ctx.num(1),
            desc: "cosh(0) = 1",
        }),
        Some(BuiltinFn::Asinh) if is_zero_expr(ctx, arg) => Some(HyperbolicCoreRewrite {
            rewritten: ctx.num(0),
            desc: "asinh(0) = 0",
        }),
        Some(BuiltinFn::Atanh) if is_zero_expr(ctx, arg) => Some(HyperbolicCoreRewrite {
            rewritten: ctx.num(0),
            desc: "atanh(0) = 0",
        }),
        Some(BuiltinFn::Acosh) if is_one_expr(ctx, arg) => Some(HyperbolicCoreRewrite {
            rewritten: ctx.num(0),
            desc: "acosh(1) = 0",
        }),
        _ => None,
    }
}

/// Rewrite direct hyperbolic-inverse compositions.
///
/// Matches:
/// - `sinh(asinh(x))`
/// - `cosh(acosh(x))`
/// - `tanh(atanh(x))`
/// - `asinh(sinh(x))`
/// - `acosh(cosh(x))`
/// - `atanh(tanh(x))`
pub fn try_rewrite_hyperbolic_composition(
    ctx: &Context,
    expr: ExprId,
) -> Option<HyperbolicCoreRewrite> {
    let Expr::Function(outer_fn, outer_args) = ctx.get(expr) else {
        return None;
    };
    let outer_fn = *outer_fn;
    let outer_args = outer_args.clone();
    if outer_args.len() != 1 {
        return None;
    }
    let inner_expr = outer_args[0];

    let Expr::Function(inner_fn, inner_args) = ctx.get(inner_expr) else {
        return None;
    };
    let inner_fn = *inner_fn;
    let inner_args = inner_args.clone();
    if inner_args.len() != 1 {
        return None;
    }
    let x = inner_args[0];

    let mapping = match (ctx.builtin_of(outer_fn), ctx.builtin_of(inner_fn)) {
        (Some(BuiltinFn::Sinh), Some(BuiltinFn::Asinh)) => Some("sinh(asinh(x)) = x"),
        (Some(BuiltinFn::Cosh), Some(BuiltinFn::Acosh)) => Some("cosh(acosh(x)) = x"),
        (Some(BuiltinFn::Tanh), Some(BuiltinFn::Atanh)) => Some("tanh(atanh(x)) = x"),
        (Some(BuiltinFn::Asinh), Some(BuiltinFn::Sinh)) => Some("asinh(sinh(x)) = x"),
        (Some(BuiltinFn::Acosh), Some(BuiltinFn::Cosh)) => Some("acosh(cosh(x)) = x"),
        (Some(BuiltinFn::Atanh), Some(BuiltinFn::Tanh)) => Some("atanh(tanh(x)) = x"),
        _ => None,
    }?;

    Some(HyperbolicCoreRewrite {
        rewritten: x,
        desc: mapping,
    })
}

#[cfg(test)]
mod tests {
    use super::{try_eval_hyperbolic_special_value, try_rewrite_hyperbolic_composition};
    use cas_ast::{BuiltinFn, Context};

    #[test]
    fn evaluates_special_values() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let one = ctx.num(1);

        let sinh0 = ctx.call_builtin(BuiltinFn::Sinh, vec![zero]);
        let cosh0 = ctx.call_builtin(BuiltinFn::Cosh, vec![zero]);
        let acosh1 = ctx.call_builtin(BuiltinFn::Acosh, vec![one]);

        assert_eq!(
            try_eval_hyperbolic_special_value(&mut ctx, sinh0)
                .expect("sinh0")
                .desc,
            "sinh(0) = 0"
        );
        assert_eq!(
            try_eval_hyperbolic_special_value(&mut ctx, cosh0)
                .expect("cosh0")
                .desc,
            "cosh(0) = 1"
        );
        assert_eq!(
            try_eval_hyperbolic_special_value(&mut ctx, acosh1)
                .expect("acosh1")
                .desc,
            "acosh(1) = 0"
        );
    }

    #[test]
    fn rewrites_compositions() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let asinh_x = ctx.call_builtin(BuiltinFn::Asinh, vec![x]);
        let expr = ctx.call_builtin(BuiltinFn::Sinh, vec![asinh_x]);

        let rewrite = try_rewrite_hyperbolic_composition(&ctx, expr).expect("rewrite");
        assert_eq!(rewrite.rewritten, x);
        assert_eq!(rewrite.desc, "sinh(asinh(x)) = x");
    }
}
