use crate::expr_nary::mul_leaves;
use cas_ast::{Context, Expr, ExprId};
use std::cmp::Ordering;

/// Check if an expression is `π/3` (division or canonical multiplication form).
pub fn is_pi_over_3(ctx: &Context, expr: ExprId) -> bool {
    let is_one_third_factor = |id: ExprId| -> bool {
        if let Expr::Number(n) = ctx.get(id) {
            return *n == num_rational::BigRational::new(1.into(), 3.into());
        }
        if let Expr::Div(num, den) = ctx.get(id) {
            if let (Expr::Number(n), Expr::Number(d)) = (ctx.get(*num), ctx.get(*den)) {
                return n.is_integer()
                    && d.is_integer()
                    && *n.numer() == 1.into()
                    && *d.numer() == 3.into();
            }
        }
        false
    };

    if let Expr::Div(num, den) = ctx.get(expr) {
        if matches!(ctx.get(*num), Expr::Constant(cas_ast::Constant::Pi)) {
            if let Expr::Number(n) = ctx.get(*den) {
                if n.is_integer() && *n.numer() == 3.into() {
                    return true;
                }
            }
        }
    }

    if let Expr::Mul(l, r) = ctx.get(expr) {
        if is_one_third_factor(*l) && matches!(ctx.get(*r), Expr::Constant(cas_ast::Constant::Pi)) {
            return true;
        }
        if is_one_third_factor(*r) && matches!(ctx.get(*l), Expr::Constant(cas_ast::Constant::Pi)) {
            return true;
        }
    }

    false
}

/// Check if `expr` equals `u + π/3` or `π/3 + u`.
pub fn is_u_plus_pi_over_3(ctx: &Context, expr: ExprId, u: ExprId) -> bool {
    if let Expr::Add(l, r) = ctx.get(expr) {
        if cas_ast::ordering::compare_expr(ctx, *l, u) == Ordering::Equal {
            return is_pi_over_3(ctx, *r);
        }
        if cas_ast::ordering::compare_expr(ctx, *r, u) == Ordering::Equal {
            return is_pi_over_3(ctx, *l);
        }
    }
    false
}

/// Check if `expr` equals `π/3 - u` or canonicalized `-u + π/3`.
pub fn is_pi_over_3_minus_u(ctx: &Context, expr: ExprId, u: ExprId) -> bool {
    if let Expr::Sub(l, r) = ctx.get(expr) {
        if is_pi_over_3(ctx, *l) && cas_ast::ordering::compare_expr(ctx, *r, u) == Ordering::Equal {
            return true;
        }
    }

    if let Expr::Add(l, r) = ctx.get(expr) {
        if is_pi_over_3(ctx, *l) {
            if let Expr::Neg(inner) = ctx.get(*r) {
                if cas_ast::ordering::compare_expr(ctx, *inner, u) == Ordering::Equal {
                    return true;
                }
            }
        }
        if is_pi_over_3(ctx, *r) {
            if let Expr::Neg(inner) = ctx.get(*l) {
                if cas_ast::ordering::compare_expr(ctx, *inner, u) == Ordering::Equal {
                    return true;
                }
            }
        }
    }

    false
}

/// Check if `tan_expr` participates in a triple-product scaffold
/// `tan(u)·tan(π/3+u)·tan(π/3-u)` within the provided ancestor chain.
pub fn is_part_of_tan_triple_product_with_ancestors(
    ctx: &Context,
    tan_expr: ExprId,
    ancestors: &[ExprId],
) -> bool {
    if !matches!(
        ctx.get(tan_expr),
        Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Tan) && args.len() == 1
    ) {
        return false;
    }

    let mut mul_root: Option<ExprId> = None;
    for &ancestor in ancestors {
        if matches!(ctx.get(ancestor), Expr::Mul(_, _)) {
            mul_root = Some(ancestor);
            break;
        }
    }

    let Some(mul_root) = mul_root else {
        return false;
    };

    let factors = mul_leaves(ctx, mul_root);
    let mut tan_args: Vec<ExprId> = Vec::new();

    for &factor in &factors {
        if let Expr::Function(fn_id, args) = ctx.get(factor) {
            if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Tan) && args.len() == 1 {
                tan_args.push(args[0]);
            }
        }
    }

    if tan_args.len() != 3 {
        return false;
    }

    for i in 0..3 {
        let u = tan_args[i];
        let others: Vec<_> = tan_args
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, &arg)| arg)
            .collect();

        let arg_j = others[0];
        let arg_k = others[1];

        let match1 = is_u_plus_pi_over_3(ctx, arg_j, u) && is_pi_over_3_minus_u(ctx, arg_k, u);
        let match2 = is_pi_over_3_minus_u(ctx, arg_j, u) && is_u_plus_pi_over_3(ctx, arg_k, u);
        if match1 || match2 {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn pi_over_3_matcher_accepts_div_and_mul_forms() {
        let mut ctx = Context::new();
        let div = parse("pi/3", &mut ctx).expect("pi/3");
        let mul = parse("(1/3)*pi", &mut ctx).expect("(1/3)*pi");

        assert!(is_pi_over_3(&ctx, div));
        assert!(is_pi_over_3(&ctx, mul));
    }

    #[test]
    fn u_plus_pi_over_3_and_minus_u_patterns_match() {
        let mut ctx = Context::new();
        let u = parse("u", &mut ctx).expect("u");
        let plus = parse("u + pi/3", &mut ctx).expect("plus");
        let minus = parse("pi/3 - u", &mut ctx).expect("minus");
        let minus_canonical = parse("(-u) + pi/3", &mut ctx).expect("minus canonical");

        assert!(is_u_plus_pi_over_3(&ctx, plus, u));
        assert!(is_pi_over_3_minus_u(&ctx, minus, u));
        assert!(is_pi_over_3_minus_u(&ctx, minus_canonical, u));
    }

    #[test]
    fn tan_triple_product_pattern_is_detected_from_ancestors() {
        let mut ctx = Context::new();
        let expr = parse("tan(u)*tan(pi/3+u)*tan(pi/3-u)", &mut ctx).expect("expr");
        let tan_u = parse("tan(u)", &mut ctx).expect("tan_u");

        assert!(is_part_of_tan_triple_product_with_ancestors(
            &ctx,
            tan_u,
            &[expr]
        ));
    }

    #[test]
    fn non_matching_tan_product_is_rejected() {
        let mut ctx = Context::new();
        let expr = parse("tan(u)*tan(v)*tan(w)", &mut ctx).expect("expr");
        let tan_u = parse("tan(u)", &mut ctx).expect("tan_u");

        assert!(!is_part_of_tan_triple_product_with_ancestors(
            &ctx,
            tan_u,
            &[expr]
        ));
    }
}
