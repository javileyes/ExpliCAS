use crate::expr_predicates::is_one_expr;
use crate::numeric_eval::as_rational_const;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_traits::One;
use std::cmp::Ordering;

/// Check whether two expressions are reciprocals: `a = 1/b` or `b = 1/a`.
pub fn are_reciprocals(ctx: &Context, expr1: ExprId, expr2: ExprId) -> bool {
    if let Expr::Div(num, den) = ctx.get(expr2) {
        if is_one_expr(ctx, *num)
            && cas_ast::ordering::compare_expr(ctx, *den, expr1) == Ordering::Equal
        {
            return true;
        }
    }

    if let Expr::Div(num, den) = ctx.get(expr1) {
        if is_one_expr(ctx, *num)
            && cas_ast::ordering::compare_expr(ctx, *den, expr2) == Ordering::Equal
        {
            return true;
        }
    }

    if let (Some(n1), Some(n2)) = (as_rational_const(ctx, expr1), as_rational_const(ctx, expr2)) {
        return (n1 * n2).is_one();
    }

    false
}

/// Check if a list of additive terms contains any reciprocal `atan`/`arctan` argument pair.
pub fn has_reciprocal_atan_pair(ctx: &Context, terms: &[ExprId]) -> bool {
    let mut atan_args: Vec<ExprId> = Vec::new();
    for &term in terms {
        if let Expr::Function(fn_id, args) = ctx.get(term) {
            if let Some(b) = ctx.builtin_of(*fn_id) {
                if matches!(b, BuiltinFn::Atan | BuiltinFn::Arctan) && args.len() == 1 {
                    atan_args.push(args[0]);
                }
            }
        }
    }

    for i in 0..atan_args.len() {
        for j in (i + 1)..atan_args.len() {
            if are_reciprocals(ctx, atan_args[i], atan_args[j]) {
                return true;
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn reciprocal_detection_works_for_symbolic_and_numeric_forms() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("x");
        let inv_x = parse("1/x", &mut ctx).expect("1/x");
        let two = parse("2", &mut ctx).expect("2");
        let half = parse("1/2", &mut ctx).expect("1/2");

        assert!(are_reciprocals(&ctx, x, inv_x));
        assert!(are_reciprocals(&ctx, two, half));
    }

    #[test]
    fn reciprocal_atan_pair_detected_in_term_list() {
        let mut ctx = Context::new();
        let t1 = parse("atan(x)", &mut ctx).expect("t1");
        let t2 = parse("arctan(1/x)", &mut ctx).expect("t2");
        let t3 = parse("5", &mut ctx).expect("t3");

        assert!(has_reciprocal_atan_pair(&ctx, &[t1, t2, t3]));
    }
}
