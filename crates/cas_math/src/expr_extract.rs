//! Shared helpers to parse primitive values from AST expressions.

use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use num_traits::ToPrimitive;

/// Extract an integer as `i64` from an expression.
///
/// Returns `None` for non-numeric expressions, non-integers, or values that do
/// not fit in `i64`.
pub fn extract_i64_integer(ctx: &Context, expr: ExprId) -> Option<i64> {
    if let Expr::Number(n) = ctx.get(expr) {
        if n.is_integer() {
            return n.to_integer().to_i64();
        }
    }
    None
}

/// Extract an integer as an exact `BigInt` from an expression.
///
/// Also accepts unary negation wrappers, e.g. `-(5)` -> `-5`.
pub fn extract_integer_exact(ctx: &Context, expr: ExprId) -> Option<num_bigint::BigInt> {
    match ctx.get(expr) {
        Expr::Number(n) => {
            if n.is_integer() {
                Some(n.to_integer())
            } else {
                None
            }
        }
        Expr::Neg(inner) => extract_integer_exact(ctx, *inner).map(|n| -n),
        _ => None,
    }
}

/// Extract a non-negative integer as `u64` from an expression.
pub fn extract_u64_integer(ctx: &Context, expr: ExprId) -> Option<u64> {
    if let Expr::Number(n) = ctx.get(expr) {
        if n.is_integer() {
            return n.to_integer().to_u64();
        }
    }
    None
}

/// Extract a non-negative integer as `usize` from an expression.
pub fn extract_usize_integer(ctx: &Context, expr: ExprId) -> Option<usize> {
    if let Expr::Number(n) = ctx.get(expr) {
        if n.is_integer() {
            return n.to_integer().to_usize();
        }
    }
    None
}

/// Extract a symbol token from an expression (represented as `Variable`).
pub fn extract_symbol_name(ctx: &Context, expr: ExprId) -> Option<&str> {
    if let Expr::Variable(sym_id) = ctx.get(expr) {
        return Some(ctx.sym_name(*sym_id));
    }
    None
}

/// Extract the exponent argument from exponential forms.
///
/// Recognizes:
/// - `exp(x)` -> `x`
/// - `e^x` -> `x`
pub fn extract_exp_argument(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Exp) && args.len() == 1 =>
        {
            Some(args[0])
        }
        Expr::Pow(base, exp) if matches!(ctx.get(*base), Expr::Constant(Constant::E)) => Some(*exp),
        _ => None,
    }
}

/// Extract `(base, arg)` from logarithmic forms.
///
/// Recognizes:
/// - `log(base, arg)` -> `(base, arg)`
/// - `ln(arg)` -> `(e, arg)` where `e` is inserted in the context
pub fn extract_log_base_argument(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let (base_opt, arg) = extract_log_base_argument_view(ctx, expr)?;
    if let Some(base) = base_opt {
        return Some((base, arg));
    }
    let e = ctx.add(Expr::Constant(Constant::E));
    Some((e, arg))
}

/// Extract `(base_opt, arg)` from logarithmic forms without mutating context.
///
/// Recognizes:
/// - `log(base, arg)` -> `(Some(base), arg)`
/// - `ln(arg)` -> `(None, arg)` (implicit base `e`)
pub fn extract_log_base_argument_view(
    ctx: &Context,
    expr: ExprId,
) -> Option<(Option<ExprId>, ExprId)> {
    let (fn_id, args) = match ctx.get(expr) {
        Expr::Function(fn_id, args) => (*fn_id, args),
        _ => return None,
    };
    if ctx.is_builtin(fn_id, BuiltinFn::Log) && args.len() == 2 {
        return Some((Some(args[0]), args[1]));
    }
    if ctx.is_builtin(fn_id, BuiltinFn::Ln) && args.len() == 1 {
        return Some((None, args[0]));
    }
    None
}

/// Extract the argument from unary logarithmic forms.
///
/// Recognizes:
/// - `ln(arg)` -> `arg`
/// - `log(arg)` -> `arg`
///
/// This intentionally excludes `log(base, arg)`.
pub fn extract_unary_log_argument_view(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(expr) {
        Expr::Function(fn_id, args) => (*fn_id, args),
        _ => return None,
    };
    if ctx.is_builtin(fn_id, BuiltinFn::Ln) && args.len() == 1 {
        return Some(args[0]);
    }
    if ctx.is_builtin(fn_id, BuiltinFn::Log) && args.len() == 1 {
        return Some(args[0]);
    }
    None
}

/// Extract the argument from unary square-root form.
///
/// Recognizes:
/// - `sqrt(arg)` -> `arg`
pub fn extract_sqrt_argument_view(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(expr) {
        Expr::Function(fn_id, args) => (*fn_id, args),
        _ => return None,
    };
    if ctx.is_builtin(fn_id, BuiltinFn::Sqrt) && args.len() == 1 {
        return Some(args[0]);
    }
    None
}

/// Extract the argument from unary absolute-value form.
///
/// Recognizes:
/// - `abs(arg)` -> `arg`
pub fn extract_abs_argument_view(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(expr) {
        Expr::Function(fn_id, args) => (*fn_id, args),
        _ => return None,
    };
    if ctx.is_builtin(fn_id, BuiltinFn::Abs) && args.len() == 1 {
        return Some(args[0]);
    }
    None
}

fn strip_unary_neg(ctx: &Context, mut expr: ExprId) -> ExprId {
    loop {
        match ctx.get(expr) {
            Expr::Neg(inner) => expr = *inner,
            _ => return expr,
        }
    }
}

/// Extract `(base_opt, arg)` from logarithmic forms with permissive log arity.
///
/// Recognizes:
/// - `log(arg)` -> `(None, arg)` (implicit base)
/// - `log(base, arg)` -> `(Some(base), arg)`
/// - `ln(arg)` -> `(None, arg)`
/// - unary negation wrappers around those forms
pub fn extract_log_base_argument_relaxed_view(
    ctx: &Context,
    expr: ExprId,
) -> Option<(Option<ExprId>, ExprId)> {
    let expr = strip_unary_neg(ctx, expr);
    let (fn_id, args) = match ctx.get(expr) {
        Expr::Function(fn_id, args) => (*fn_id, args),
        _ => return None,
    };
    if ctx.is_builtin(fn_id, BuiltinFn::Log) {
        return match args.as_slice() {
            [arg] => Some((None, *arg)),
            [base, arg] => Some((Some(*base), *arg)),
            _ => None,
        };
    }
    if ctx.is_builtin(fn_id, BuiltinFn::Ln) && args.len() == 1 {
        return Some((None, args[0]));
    }
    None
}

/// Extract `(base_opt, arg)` from a plain or scaled logarithm term.
///
/// Recognizes:
/// - plain logs accepted by `extract_log_base_argument_relaxed_view`
/// - `k * log(...)` / `k * ln(...)` where one factor is numeric
/// - unary negation wrappers around those forms
pub fn extract_scaled_log_base_argument_relaxed_view(
    ctx: &Context,
    expr: ExprId,
) -> Option<(Option<ExprId>, ExprId)> {
    let expr = strip_unary_neg(ctx, expr);
    if let Some(info) = extract_log_base_argument_relaxed_view(ctx, expr) {
        return Some(info);
    }
    let (l, r) = match ctx.get(expr) {
        Expr::Mul(l, r) => (*l, *r),
        _ => return None,
    };
    let l_is_num = matches!(ctx.get(l), Expr::Number(_));
    let r_is_num = matches!(ctx.get(r), Expr::Number(_));
    if l_is_num == r_is_num {
        return None;
    }
    let log_expr = if l_is_num { r } else { l };
    extract_log_base_argument_relaxed_view(ctx, log_expr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn extracts_i64_integer_from_number() {
        let mut ctx = Context::new();
        let five = parse("5", &mut ctx).expect("parse 5");
        let half = parse("1/2", &mut ctx).expect("parse 1/2");
        assert_eq!(extract_i64_integer(&ctx, five), Some(5));
        assert_eq!(extract_i64_integer(&ctx, half), None);
    }

    #[test]
    fn extracts_integer_exact_with_negation() {
        let mut ctx = Context::new();
        let value = parse("-(42)", &mut ctx).expect("parse -(42)");
        assert_eq!(
            extract_integer_exact(&ctx, value),
            Some(num_bigint::BigInt::from(-42))
        );
    }

    #[test]
    fn extracts_exp_argument_from_builtin_exp() {
        let mut ctx = Context::new();
        let expr = parse("exp(x)", &mut ctx).expect("parse exp");
        let arg = extract_exp_argument(&ctx, expr).expect("must extract exp arg");
        let x = parse("x", &mut ctx).expect("parse x");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, arg, x),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn extracts_exp_argument_from_e_pow() {
        let mut ctx = Context::new();
        let expr = parse("e^(2*x)", &mut ctx).expect("parse e^(2*x)");
        let arg = extract_exp_argument(&ctx, expr).expect("must extract pow arg");
        let expected = parse("2*x", &mut ctx).expect("parse 2*x");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, arg, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn non_exponential_returns_none() {
        let mut ctx = Context::new();
        let expr = parse("x^2", &mut ctx).expect("parse x^2");
        assert!(extract_exp_argument(&ctx, expr).is_none());
    }

    #[test]
    fn extracts_log_base_argument_from_log() {
        let mut ctx = Context::new();
        let expr = parse("log(2, x)", &mut ctx).expect("parse log");
        let (base, arg) = extract_log_base_argument(&mut ctx, expr).expect("must extract log");
        let two = parse("2", &mut ctx).expect("parse 2");
        let x = parse("x", &mut ctx).expect("parse x");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, base, two),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, arg, x),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn extracts_log_base_argument_from_ln_as_e() {
        let mut ctx = Context::new();
        let expr = parse("ln(x)", &mut ctx).expect("parse ln");
        let (base, arg) = extract_log_base_argument(&mut ctx, expr).expect("must extract ln");
        let x = parse("x", &mut ctx).expect("parse x");
        assert!(matches!(ctx.get(base), Expr::Constant(Constant::E)));
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, arg, x),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn extracts_log_base_argument_view_without_mutation() {
        let mut ctx = Context::new();
        let expr = parse("ln(x)", &mut ctx).expect("parse ln");
        let (base_opt, arg) =
            extract_log_base_argument_view(&ctx, expr).expect("must extract view");
        let x = parse("x", &mut ctx).expect("parse x");
        assert!(base_opt.is_none());
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, arg, x),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn extracts_relaxed_log_argument_from_single_arg_log() {
        let mut ctx = Context::new();
        let expr = parse("log(x)", &mut ctx).expect("parse log(x)");
        let (base_opt, arg) =
            extract_log_base_argument_relaxed_view(&ctx, expr).expect("must extract relaxed log");
        let x = parse("x", &mut ctx).expect("parse x");
        assert!(base_opt.is_none());
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, arg, x),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn extracts_relaxed_log_argument_through_negation() {
        let mut ctx = Context::new();
        let expr = parse("-ln(x)", &mut ctx).expect("parse -ln(x)");
        let (base_opt, arg) = extract_log_base_argument_relaxed_view(&ctx, expr)
            .expect("must extract relaxed negated ln");
        let x = parse("x", &mut ctx).expect("parse x");
        assert!(base_opt.is_none());
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, arg, x),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn extracts_scaled_log_argument_from_numeric_coefficient_term() {
        let mut ctx = Context::new();
        let expr = parse("-2*log(3, y)", &mut ctx).expect("parse -2*log(3, y)");
        let (base_opt, arg) = extract_scaled_log_base_argument_relaxed_view(&ctx, expr)
            .expect("must extract scaled log");
        let three = parse("3", &mut ctx).expect("parse 3");
        let y = parse("y", &mut ctx).expect("parse y");
        assert_eq!(
            cas_ast::ordering::compare_expr(
                &ctx,
                base_opt.expect("base expected for log(3, y)"),
                three
            ),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, arg, y),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn extracts_unary_log_argument_from_ln() {
        let mut ctx = Context::new();
        let expr = parse("ln(x)", &mut ctx).expect("parse ln(x)");
        let arg = extract_unary_log_argument_view(&ctx, expr).expect("must extract ln arg");
        let x = parse("x", &mut ctx).expect("parse x");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, arg, x),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn extracts_unary_log_argument_from_log() {
        let mut ctx = Context::new();
        let expr = parse("log(x)", &mut ctx).expect("parse log(x)");
        let arg = extract_unary_log_argument_view(&ctx, expr).expect("must extract log arg");
        let x = parse("x", &mut ctx).expect("parse x");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, arg, x),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn unary_log_argument_rejects_binary_log() {
        let mut ctx = Context::new();
        let expr = parse("log(2, x)", &mut ctx).expect("parse log(2, x)");
        assert!(extract_unary_log_argument_view(&ctx, expr).is_none());
    }

    #[test]
    fn extracts_sqrt_argument_from_unary_sqrt() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(z)", &mut ctx).expect("parse sqrt(z)");
        let arg = extract_sqrt_argument_view(&ctx, expr).expect("must extract sqrt arg");
        let z = parse("z", &mut ctx).expect("parse z");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, arg, z),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn sqrt_argument_rejects_non_sqrt_function() {
        let mut ctx = Context::new();
        let expr = parse("sin(z)", &mut ctx).expect("parse sin(z)");
        assert!(extract_sqrt_argument_view(&ctx, expr).is_none());
    }

    #[test]
    fn extracts_abs_argument_from_unary_abs() {
        let mut ctx = Context::new();
        let expr = parse("abs(t)", &mut ctx).expect("parse abs(t)");
        let arg = extract_abs_argument_view(&ctx, expr).expect("must extract abs arg");
        let t = parse("t", &mut ctx).expect("parse t");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, arg, t),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn abs_argument_rejects_non_abs_function() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(t)", &mut ctx).expect("parse sqrt(t)");
        assert!(extract_abs_argument_view(&ctx, expr).is_none());
    }
}
