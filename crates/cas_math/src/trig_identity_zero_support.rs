use crate::expr_destructure::{as_add, as_mul, as_pow};
use crate::trig_sum_product_support::{args_match_as_multiset, extract_trig_arg};
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};

fn is_trig_squared_t(ctx: &Context, expr: ExprId, trig: BuiltinFn, t: ExprId) -> bool {
    let Some((base, exp)) = as_pow(ctx, expr) else {
        return false;
    };
    let Expr::Number(n) = ctx.get(exp) else {
        return false;
    };
    if *n != num_rational::BigRational::from_integer(2.into()) {
        return false;
    }

    let Some(arg) = extract_trig_arg(ctx, base, trig.name()) else {
        return false;
    };
    compare_expr(ctx, arg, t) == std::cmp::Ordering::Equal
}

pub fn is_sin_squared_t(ctx: &Context, expr: ExprId, t: ExprId) -> bool {
    is_trig_squared_t(ctx, expr, BuiltinFn::Sin, t)
}

pub fn is_cos_squared_t(ctx: &Context, expr: ExprId, t: ExprId) -> bool {
    is_trig_squared_t(ctx, expr, BuiltinFn::Cos, t)
}

/// Match `1 + tan(a)*tan(b)` (in any operand order).
pub fn match_one_plus_tan_product(ctx: &Context, expr: ExprId, a: ExprId, b: ExprId) -> bool {
    let Some((l, r)) = as_add(ctx, expr) else {
        return false;
    };
    let (one_part, product_part) = if let Expr::Number(n) = ctx.get(l) {
        if *n == num_rational::BigRational::from_integer(1.into()) {
            (l, r)
        } else {
            return false;
        }
    } else if let Expr::Number(n) = ctx.get(r) {
        if *n == num_rational::BigRational::from_integer(1.into()) {
            (r, l)
        } else {
            return false;
        }
    } else {
        return false;
    };
    let _ = one_part;

    let Some((ml, mr)) = as_mul(ctx, product_part) else {
        return false;
    };
    let Some(arg1) = extract_trig_arg(ctx, ml, BuiltinFn::Tan.name()) else {
        return false;
    };
    let Some(arg2) = extract_trig_arg(ctx, mr, BuiltinFn::Tan.name()) else {
        return false;
    };
    args_match_as_multiset(ctx, arg1, arg2, a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn detects_sin_and_cos_squared_of_target() {
        let mut ctx = Context::new();
        let t = parse("u+v", &mut ctx).expect("t");
        let sin_sq = parse("sin(u+v)^2", &mut ctx).expect("sin_sq");
        let cos_sq = parse("cos(u+v)^2", &mut ctx).expect("cos_sq");
        let wrong = parse("sin(u)^2", &mut ctx).expect("wrong");

        assert!(is_sin_squared_t(&ctx, sin_sq, t));
        assert!(is_cos_squared_t(&ctx, cos_sq, t));
        assert!(!is_sin_squared_t(&ctx, wrong, t));
    }

    #[test]
    fn matches_one_plus_tan_product_both_orders() {
        let mut ctx = Context::new();
        let a = parse("a", &mut ctx).expect("a");
        let b = parse("b", &mut ctx).expect("b");
        let expr1 = parse("1 + tan(a)*tan(b)", &mut ctx).expect("expr1");
        let expr2 = parse("tan(b)*tan(a) + 1", &mut ctx).expect("expr2");
        let wrong = parse("1 + tan(a)*tan(c)", &mut ctx).expect("wrong");

        assert!(match_one_plus_tan_product(&ctx, expr1, a, b));
        assert!(match_one_plus_tan_product(&ctx, expr2, a, b));
        assert!(!match_one_plus_tan_product(&ctx, wrong, a, b));
    }
}
