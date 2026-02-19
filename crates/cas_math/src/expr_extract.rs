//! Shared helpers to parse primitive values from AST expressions.

use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use num_traits::ToPrimitive;

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

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

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
}
