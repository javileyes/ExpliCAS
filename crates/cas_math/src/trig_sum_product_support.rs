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

/// Build `(A-B)/2`, optionally canonicalizing order before subtraction.
///
/// The caller provides `simplify_expr` to pre-simplify the numerator difference
/// before dividing by 2 (engine currently uses `collect` for this step).
pub fn build_half_diff_with_simplifier(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    canonical_order: bool,
    simplify_expr: fn(&mut Context, ExprId) -> ExprId,
) -> ExprId {
    let (first, second) =
        if canonical_order && cas_ast::ordering::compare_expr(ctx, a, b) == Ordering::Greater {
            (b, a)
        } else {
            (a, b)
        };

    let diff = ctx.add(Expr::Sub(first, second));
    let diff_simplified = simplify_expr(ctx, diff);
    let two = ctx.num(2);
    let result = ctx.add(Expr::Div(diff_simplified, two));
    simplify_numeric_div(ctx, result)
}

/// Build `(A+B)/2`, pre-simplifying the sum via the caller-provided callback.
pub fn build_avg_with_simplifier(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    simplify_expr: fn(&mut Context, ExprId) -> ExprId,
) -> ExprId {
    let sum = ctx.add(Expr::Add(a, b));
    let sum_simplified = simplify_expr(ctx, sum);
    let two = ctx.num(2);
    let result = ctx.add(Expr::Div(sum_simplified, two));
    simplify_numeric_div(ctx, result)
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

    #[test]
    fn build_half_diff_with_simplifier_respects_canonical_order() {
        fn passthrough(_: &mut Context, id: ExprId) -> ExprId {
            id
        }

        let mut ctx = Context::new();
        let y = parse("y", &mut ctx).expect("y");
        let x = parse("x", &mut ctx).expect("x");

        let canonical = build_half_diff_with_simplifier(&mut ctx, y, x, true, passthrough);
        let expected_canonical = parse("(x-y)/2", &mut ctx).expect("expected_canonical");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, canonical, expected_canonical),
            Ordering::Equal
        );

        let non_canonical = build_half_diff_with_simplifier(&mut ctx, y, x, false, passthrough);
        let expected_non_canonical = parse("(y-x)/2", &mut ctx).expect("expected_non_canonical");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, non_canonical, expected_non_canonical),
            Ordering::Equal
        );
    }

    #[test]
    fn build_avg_with_simplifier_applies_callback_before_division() {
        fn collapse_duplicate_add(ctx: &mut Context, expr: ExprId) -> ExprId {
            let (l, r) = if let Expr::Add(l, r) = ctx.get(expr) {
                (*l, *r)
            } else {
                return expr;
            };
            if cas_ast::ordering::compare_expr(ctx, l, r) == Ordering::Equal {
                let two = ctx.num(2);
                return ctx.add(Expr::Mul(two, l));
            }
            expr
        }

        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("x");
        let avg = build_avg_with_simplifier(&mut ctx, x, x, collapse_duplicate_add);
        let expected = parse("x", &mut ctx).expect("expected");

        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, avg, expected),
            Ordering::Equal
        );
    }
}
