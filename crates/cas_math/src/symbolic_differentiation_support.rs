//! Symbolic differentiation helpers shared by differentiation-facing rule layers.

use crate::build::mul2_raw;
use crate::expr_predicates::contains_named_var;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_traits::{One, Zero};

fn is_zero(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_zero())
}

fn is_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_one())
}

fn add_pruned(ctx: &mut Context, left: ExprId, right: ExprId) -> ExprId {
    if is_zero(ctx, left) {
        right
    } else if is_zero(ctx, right) {
        left
    } else {
        ctx.add(Expr::Add(left, right))
    }
}

fn sub_pruned(ctx: &mut Context, left: ExprId, right: ExprId) -> ExprId {
    if is_zero(ctx, right) {
        left
    } else if is_zero(ctx, left) {
        ctx.add(Expr::Neg(right))
    } else {
        ctx.add(Expr::Sub(left, right))
    }
}

fn mul_pruned(ctx: &mut Context, left: ExprId, right: ExprId) -> ExprId {
    if is_zero(ctx, left) || is_zero(ctx, right) {
        ctx.num(0)
    } else if is_one(ctx, left) {
        right
    } else if is_one(ctx, right) {
        left
    } else {
        mul2_raw(ctx, left, right)
    }
}

/// Differentiate `expr` with respect to variable `var`.
pub fn differentiate_symbolic_expr(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    if !contains_named_var(ctx, expr, var) {
        return Some(ctx.num(0));
    }

    match ctx.get(expr) {
        Expr::Variable(sym_id) => {
            if ctx.sym_name(*sym_id) == var {
                Some(ctx.num(1))
            } else {
                Some(ctx.num(0))
            }
        }
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            let dl = differentiate_symbolic_expr(ctx, l, var)?;
            let dr = differentiate_symbolic_expr(ctx, r, var)?;
            Some(add_pruned(ctx, dl, dr))
        }
        Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            let dl = differentiate_symbolic_expr(ctx, l, var)?;
            let dr = differentiate_symbolic_expr(ctx, r, var)?;
            Some(sub_pruned(ctx, dl, dr))
        }
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            let dl = differentiate_symbolic_expr(ctx, l, var)?;
            let dr = differentiate_symbolic_expr(ctx, r, var)?;
            let term1 = mul_pruned(ctx, dl, r);
            let term2 = mul_pruned(ctx, l, dr);
            Some(add_pruned(ctx, term1, term2))
        }
        Expr::Div(l, r) => {
            let (l, r) = (*l, *r);
            let dl = differentiate_symbolic_expr(ctx, l, var)?;
            let dr = differentiate_symbolic_expr(ctx, r, var)?;
            let term1 = mul_pruned(ctx, dl, r);
            let term2 = mul_pruned(ctx, l, dr);
            let num = sub_pruned(ctx, term1, term2);
            let two = ctx.num(2);
            let den = ctx.add(Expr::Pow(r, two));
            Some(ctx.add(Expr::Div(num, den)))
        }
        Expr::Pow(base, exp) => {
            let (base, exp) = (*base, *exp);
            let db = differentiate_symbolic_expr(ctx, base, var)?;
            let de = differentiate_symbolic_expr(ctx, exp, var)?;

            if !contains_named_var(ctx, exp, var) {
                let one = ctx.num(1);
                let n_minus_one = ctx.add(Expr::Sub(exp, one));
                let pow_term = ctx.add(Expr::Pow(base, n_minus_one));
                let term = mul_pruned(ctx, exp, pow_term);
                Some(mul_pruned(ctx, term, db))
            } else if !contains_named_var(ctx, base, var) {
                let ln_a = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
                let term = mul_pruned(ctx, expr, ln_a);
                Some(mul_pruned(ctx, term, de))
            } else {
                let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
                let term1 = mul_pruned(ctx, de, ln_base);
                let term2_num = mul_pruned(ctx, exp, db);
                let term2 = ctx.add(Expr::Div(term2_num, base));
                let inner = add_pruned(ctx, term1, term2);
                Some(mul_pruned(ctx, expr, inner))
            }
        }
        Expr::Function(fn_id, args) => {
            let (fn_id, args) = (*fn_id, args.clone());
            if args.len() != 1 {
                return None;
            }
            let arg = args[0];
            let da = differentiate_symbolic_expr(ctx, arg, var)?;

            match ctx.builtin_of(fn_id) {
                Some(BuiltinFn::Sin) => {
                    let cos_u = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
                    Some(mul_pruned(ctx, cos_u, da))
                }
                Some(BuiltinFn::Cos) => {
                    let sin_u = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
                    let neg_sin = ctx.add(Expr::Neg(sin_u));
                    Some(mul_pruned(ctx, neg_sin, da))
                }
                Some(BuiltinFn::Tan) => {
                    let cos_u = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
                    let two = ctx.num(2);
                    let cos_sq = ctx.add(Expr::Pow(cos_u, two));
                    let one = ctx.num(1);
                    let sec_sq = ctx.add(Expr::Div(one, cos_sq));
                    Some(mul_pruned(ctx, sec_sq, da))
                }
                Some(BuiltinFn::Exp) => Some(mul_pruned(ctx, expr, da)),
                Some(BuiltinFn::Ln) => Some(ctx.add(Expr::Div(da, arg))),
                Some(BuiltinFn::Abs) => {
                    let term = ctx.add(Expr::Div(arg, expr));
                    Some(mul_pruned(ctx, term, da))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::differentiate_symbolic_expr;
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: cas_ast::ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn differentiates_product_rule() {
        let mut ctx = Context::new();
        let expr = parse("x*sin(x)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);
        assert!(text.contains("sin(x)"));
        assert!(text.contains("cos(x)"));
    }

    #[test]
    fn differentiates_chain_rule_exp() {
        let mut ctx = Context::new();
        let expr = parse("exp(x^2)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);
        assert!(text.contains("exp(x^2)") || text.contains("e^(x^2)"));
    }

    #[test]
    fn prunes_zero_product_terms_from_polynomial_derivative() {
        let mut ctx = Context::new();
        let expr = parse("x^3 + 2*x^2 - 5*x + 1", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        let text = rendered(&ctx, out);
        assert!(!text.contains("0 ·"), "unexpected zero product in {text}");
        assert!(!text.contains("· 1"), "unexpected unit factor in {text}");
        assert!(text.contains("3"));
        assert!(text.contains("- 5"));
    }

    #[test]
    fn prunes_unit_chain_factor_for_sine() {
        let mut ctx = Context::new();
        let expr = parse("sin(x)", &mut ctx).expect("parse");
        let out = differentiate_symbolic_expr(&mut ctx, expr, "x").expect("diff");
        assert_eq!(rendered(&ctx, out), "cos(x)");
    }
}
