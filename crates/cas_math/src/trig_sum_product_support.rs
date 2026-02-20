use crate::expr_nary::add_leaves;
use crate::numeric::as_i64;
use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;
use std::cmp::Ordering;

/// Extract the argument from a trig function: `sin(arg)`/`cos(arg)` -> `arg`.
pub fn extract_trig_arg(ctx: &Context, id: ExprId, fn_name: &str) -> Option<ExprId> {
    if let Expr::Function(fn_id, args) = ctx.get(id) {
        if ctx.builtin_of(*fn_id).is_some_and(|b| b.name() == fn_name) && args.len() == 1 {
            return Some(args[0]);
        }
    }
    None
}

/// Extract two trig args from a 2-term sum: `sin(A)+sin(B)` / `cos(A)+cos(B)`.
pub fn extract_trig_two_term_sum(
    ctx: &Context,
    expr: ExprId,
    fn_name: &str,
) -> Option<(ExprId, ExprId)> {
    let terms = add_leaves(ctx, expr);
    if terms.len() != 2 {
        return None;
    }
    let arg1 = extract_trig_arg(ctx, terms[0], fn_name)?;
    let arg2 = extract_trig_arg(ctx, terms[1], fn_name)?;
    Some((arg1, arg2))
}

/// Extract two trig args from a 2-term difference:
/// - `Sub(sin(A), sin(B))`
/// - `Add(sin(A), Neg(sin(B)))`
pub fn extract_trig_two_term_diff(
    ctx: &Context,
    expr: ExprId,
    fn_name: &str,
) -> Option<(ExprId, ExprId)> {
    if let Expr::Sub(l, r) = ctx.get(expr) {
        let arg1 = extract_trig_arg(ctx, *l, fn_name)?;
        let arg2 = extract_trig_arg(ctx, *r, fn_name)?;
        return Some((arg1, arg2));
    }

    if let Expr::Add(l, r) = ctx.get(expr) {
        if let Expr::Neg(inner) = ctx.get(*r) {
            let arg1 = extract_trig_arg(ctx, *l, fn_name)?;
            let arg2 = extract_trig_arg(ctx, *inner, fn_name)?;
            return Some((arg1, arg2));
        }
        if let Expr::Neg(inner) = ctx.get(*l) {
            let arg1 = extract_trig_arg(ctx, *r, fn_name)?;
            let arg2 = extract_trig_arg(ctx, *inner, fn_name)?;
            return Some((arg1, arg2));
        }
    }

    None
}

/// Check if two pairs match as multisets: `{a1, a2} == {b1, b2}`.
pub fn args_match_as_multiset(
    ctx: &Context,
    a1: ExprId,
    a2: ExprId,
    b1: ExprId,
    b2: ExprId,
) -> bool {
    let direct = cas_ast::ordering::compare_expr(ctx, a1, b1) == Ordering::Equal
        && cas_ast::ordering::compare_expr(ctx, a2, b2) == Ordering::Equal;
    let crossed = cas_ast::ordering::compare_expr(ctx, a1, b2) == Ordering::Equal
        && cas_ast::ordering::compare_expr(ctx, a2, b1) == Ordering::Equal;
    direct || crossed
}

/// Normalize an expression for even functions: `f(-x) == f(x)`.
pub fn normalize_for_even_fn(ctx: &Context, expr: ExprId) -> ExprId {
    let minus_one = BigRational::from_integer(BigInt::from(-1));

    if let Expr::Neg(inner) = ctx.get(expr) {
        return *inner;
    }
    if let Expr::Mul(l, r) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*l) {
            if n == &minus_one {
                return *r;
            }
        }
        if let Expr::Number(n) = ctx.get(*r) {
            if n == &minus_one {
                return *l;
            }
        }
    }
    expr
}

/// Simplify a numeric division in coefficient-linear forms.
/// Examples: `4*x/2 -> 2*x`, `-2*x/2 -> -x`, `4/2 -> 2`.
pub fn simplify_numeric_div(ctx: &mut Context, expr: ExprId) -> ExprId {
    let (num, den) = if let Expr::Div(n, d) = ctx.get(expr) {
        (*n, *d)
    } else {
        return expr;
    };

    let Some(den_val) = as_i64(ctx, den) else {
        return expr;
    };
    if den_val == 0 {
        return expr;
    }

    if let Expr::Mul(l, r) = ctx.get(num) {
        let (l, r) = (*l, *r);
        if let Some(coeff) = as_i64(ctx, l) {
            if coeff % den_val == 0 {
                let new_coeff = coeff / den_val;
                if new_coeff == 1 {
                    return r;
                }
                if new_coeff == -1 {
                    return ctx.add(Expr::Neg(r));
                }
                let new_coeff_expr = ctx.num(new_coeff);
                return ctx.add(Expr::Mul(new_coeff_expr, r));
            }
        }
        if let Some(coeff) = as_i64(ctx, r) {
            if coeff % den_val == 0 {
                let new_coeff = coeff / den_val;
                if new_coeff == 1 {
                    return l;
                }
                if new_coeff == -1 {
                    return ctx.add(Expr::Neg(l));
                }
                let new_coeff_expr = ctx.num(new_coeff);
                return ctx.add(Expr::Mul(l, new_coeff_expr));
            }
        }
    }

    if let Some(num_val) = as_i64(ctx, num) {
        if num_val % den_val == 0 {
            return ctx.num(num_val / den_val);
        }
    }

    expr
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn extract_sum_and_diff_patterns() {
        let mut ctx = Context::new();
        let sum = parse("sin(a)+sin(b)", &mut ctx).expect("sum");
        let diff = parse("sin(a)-sin(b)", &mut ctx).expect("diff");

        let sum_args = extract_trig_two_term_sum(&ctx, sum, "sin");
        let diff_args = extract_trig_two_term_diff(&ctx, diff, "sin");

        assert!(sum_args.is_some());
        assert!(diff_args.is_some());
    }

    #[test]
    fn multiset_match_is_order_invariant() {
        let mut ctx = Context::new();
        let a = parse("x", &mut ctx).expect("a");
        let b = parse("y", &mut ctx).expect("b");
        assert!(args_match_as_multiset(&ctx, a, b, b, a));
    }

    #[test]
    fn normalize_for_even_strips_negation_shapes() {
        let mut ctx = Context::new();
        let neg = parse("-x", &mut ctx).expect("neg");
        let mul = parse("-1*x", &mut ctx).expect("mul");
        let x = parse("x", &mut ctx).expect("x");

        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, normalize_for_even_fn(&ctx, neg), x),
            Ordering::Equal
        );
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, normalize_for_even_fn(&ctx, mul), x),
            Ordering::Equal
        );
    }

    #[test]
    fn simplify_numeric_div_reduces_linear_forms() {
        let mut ctx = Context::new();
        let expr = parse("(4*x)/2", &mut ctx).expect("expr");
        let simplified = simplify_numeric_div(&mut ctx, expr);
        let expected = parse("2*x", &mut ctx).expect("expected");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, simplified, expected),
            Ordering::Equal
        );
    }
}
