use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_traits::One;

/// Simplify `base^(log(base, x))` subterms when splitting exponent sums.
pub fn simplify_exp_log(context: &mut Context, base: ExprId, exp: ExprId) -> ExprId {
    if let Expr::Function(name, args) = context.get(exp) {
        if context.is_builtin(*name, BuiltinFn::Log) && args.len() == 2 {
            let log_base = args[0];
            let log_arg = args[1];
            if log_base == base {
                return log_arg;
            }
        }
    }
    context.add(Expr::Pow(base, exp))
}

/// Detect whether expression is a log/ln call (possibly nested in a product).
pub fn is_log(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Function(name, _) = ctx.get(expr) {
        if let Some(b) = ctx.builtin_of(*name) {
            return b == BuiltinFn::Log || b == BuiltinFn::Ln;
        }
    }
    if let Expr::Mul(l, r) = ctx.get(expr) {
        return is_log(ctx, *l) || is_log(ctx, *r);
    }
    false
}

/// Normalize expression to `(core, exponent)` form:
/// - `a` => `(a, 1)`
/// - `a^m` => `(a, m)`
/// - `1/a` => `(a, -1)`
pub fn normalize_to_power(ctx: &mut Context, expr: ExprId) -> (ExprId, ExprId) {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => (*base, *exp),
        Expr::Div(num, den) => {
            let (num, den) = (*num, *den);
            if matches!(ctx.get(num), Expr::Number(n) if n.is_one()) {
                let neg_one = ctx.num(-1);
                (den, neg_one)
            } else {
                let one = ctx.num(1);
                (expr, one)
            }
        }
        _ => {
            let one = ctx.num(1);
            (expr, one)
        }
    }
}

/// Count multiplicative factors in a flattened multiplication tree.
pub fn count_mul_factors(ctx: &Context, expr: ExprId) -> u32 {
    match ctx.get(expr) {
        Expr::Mul(a, b) => count_mul_factors(ctx, *a) + count_mul_factors(ctx, *b),
        _ => 1,
    }
}

/// Collect multiplicative factors in left-to-right order from a multiplication tree.
pub fn collect_mul_factors(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    match ctx.get(expr) {
        Expr::Mul(a, b) => {
            let mut factors = collect_mul_factors(ctx, *a);
            factors.extend(collect_mul_factors(ctx, *b));
            factors
        }
        _ => vec![expr],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn log_detection_handles_nested_product() {
        let mut ctx = Context::new();
        let expr = parse("3*log(x)", &mut ctx).expect("expr");
        assert!(is_log(&ctx, expr));
    }

    #[test]
    fn simplify_exp_log_cancels_matching_base() {
        let mut ctx = Context::new();
        let base = parse("b", &mut ctx).expect("base");
        let exp = parse("log(b,x)", &mut ctx).expect("exp");
        let out = simplify_exp_log(&mut ctx, base, exp);
        let x = parse("x", &mut ctx).expect("x");
        assert_eq!(out, x);
    }

    #[test]
    fn normalize_to_power_reciprocal_form() {
        let mut ctx = Context::new();
        let expr = parse("1/x", &mut ctx).expect("expr");
        let (_core, exp) = normalize_to_power(&mut ctx, expr);
        assert!(
            matches!(ctx.get(exp), Expr::Number(n) if n.is_integer() && n.to_integer() == (-1).into())
        );
    }

    #[test]
    fn multiplicative_factor_count_and_collect_flatten() {
        let mut ctx = Context::new();
        let expr = parse("a*b*c", &mut ctx).expect("expr");
        assert_eq!(count_mul_factors(&ctx, expr), 3);
        assert_eq!(collect_mul_factors(&ctx, expr).len(), 3);
    }
}
