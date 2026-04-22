use crate::expr_nary::AddView;
use crate::expr_nary::Sign;
use crate::expr_predicates::{is_one_expr, is_zero_expr};
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyperbolicCoreRewrite {
    pub rewritten: ExprId,
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
        }),
        Some(BuiltinFn::Tanh) if is_zero_expr(ctx, arg) => Some(HyperbolicCoreRewrite {
            rewritten: ctx.num(0),
        }),
        Some(BuiltinFn::Cosh) if is_zero_expr(ctx, arg) => Some(HyperbolicCoreRewrite {
            rewritten: ctx.num(1),
        }),
        Some(BuiltinFn::Asinh) if is_zero_expr(ctx, arg) => Some(HyperbolicCoreRewrite {
            rewritten: ctx.num(0),
        }),
        Some(BuiltinFn::Atanh) if is_zero_expr(ctx, arg) => Some(HyperbolicCoreRewrite {
            rewritten: ctx.num(0),
        }),
        Some(BuiltinFn::Acosh) if is_one_expr(ctx, arg) => Some(HyperbolicCoreRewrite {
            rewritten: ctx.num(0),
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

    match (ctx.builtin_of(outer_fn), ctx.builtin_of(inner_fn)) {
        (Some(BuiltinFn::Sinh), Some(BuiltinFn::Asinh))
        | (Some(BuiltinFn::Cosh), Some(BuiltinFn::Acosh))
        | (Some(BuiltinFn::Tanh), Some(BuiltinFn::Atanh))
        | (Some(BuiltinFn::Asinh), Some(BuiltinFn::Sinh))
        | (Some(BuiltinFn::Acosh), Some(BuiltinFn::Cosh))
        | (Some(BuiltinFn::Atanh), Some(BuiltinFn::Tanh)) => {}
        _ => return None,
    }

    Some(HyperbolicCoreRewrite { rewritten: x })
}

fn extract_square_power_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) if matches!(ctx.get(*exp), Expr::Number(n) if n.is_integer() && n.numer() == &2.into()) => {
            Some(*base)
        }
        _ => None,
    }
}

fn normalize_signed_add_term_local(
    ctx: &Context,
    term_expr: ExprId,
    term_sign: Sign,
) -> (ExprId, Sign) {
    match ctx.get(term_expr) {
        Expr::Neg(inner) => (*inner, term_sign.negate()),
        _ => (term_expr, term_sign),
    }
}

fn extract_square_plus_minus_one_pattern(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, Sign, Sign)> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return None;
    }

    let mut square_term = None;
    let mut one_term = None;
    for (term_expr, term_sign) in terms {
        let (term_expr, term_sign) = normalize_signed_add_term_local(ctx, term_expr, term_sign);
        if is_one_expr(ctx, term_expr) {
            if one_term.replace(term_sign).is_some() {
                return None;
            }
            continue;
        }

        let square_base = extract_square_power_base(ctx, term_expr)?;
        if square_term.replace((square_base, term_sign)).is_some() {
            return None;
        }
    }

    let (square_base, square_sign) = square_term?;
    Some((square_base, square_sign, one_term?))
}

pub fn try_rewrite_atanh_square_ratio_to_ln(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HyperbolicCoreRewrite> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Atanh) || args.len() != 1 {
        return None;
    }

    let Expr::Div(numerator, denominator) = ctx.get(args[0]) else {
        return None;
    };
    let (numerator, denominator) = (*numerator, *denominator);

    let (den_base, den_square_sign, den_one_sign) =
        extract_square_plus_minus_one_pattern(ctx, denominator)?;
    if den_square_sign != Sign::Pos || den_one_sign != Sign::Pos {
        return None;
    }

    let (num_base, num_square_sign, num_one_sign) =
        extract_square_plus_minus_one_pattern(ctx, numerator)?;
    if compare_expr(ctx, den_base, num_base) != Ordering::Equal {
        return None;
    }

    let ln_expr = ctx.call_builtin(BuiltinFn::Ln, vec![den_base]);
    let rewritten = match (num_square_sign, num_one_sign) {
        (Sign::Pos, Sign::Neg) => ln_expr,
        (Sign::Neg, Sign::Pos) => ctx.add(Expr::Neg(ln_expr)),
        _ => return None,
    };

    Some(HyperbolicCoreRewrite { rewritten })
}

#[cfg(test)]
mod tests {
    use super::{
        try_eval_hyperbolic_special_value, try_rewrite_atanh_square_ratio_to_ln,
        try_rewrite_hyperbolic_composition,
    };
    use cas_ast::{BuiltinFn, Context};
    use cas_parser::parse;

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
                .rewritten,
            zero
        );
        assert_eq!(
            try_eval_hyperbolic_special_value(&mut ctx, cosh0)
                .expect("cosh0")
                .rewritten,
            one
        );
        assert_eq!(
            try_eval_hyperbolic_special_value(&mut ctx, acosh1)
                .expect("acosh1")
                .rewritten,
            zero
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
    }

    #[test]
    fn rewrites_atanh_square_ratio_to_ln() {
        let mut ctx = Context::new();
        let expr = parse("atanh((x^2 - 1)/(x^2 + 1))", &mut ctx).expect("expr");

        let rewrite = try_rewrite_atanh_square_ratio_to_ln(&mut ctx, expr).expect("rewrite");

        assert_eq!(rewrite.rewritten, parse("ln(x)", &mut ctx).expect("ln(x)"));
    }
}
