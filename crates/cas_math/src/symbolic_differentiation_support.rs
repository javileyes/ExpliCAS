//! Symbolic differentiation helpers shared by differentiation-facing rule layers.

use crate::build::mul2_raw;
use crate::expr_predicates::contains_named_var;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};

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
            Some(ctx.add(Expr::Add(dl, dr)))
        }
        Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            let dl = differentiate_symbolic_expr(ctx, l, var)?;
            let dr = differentiate_symbolic_expr(ctx, r, var)?;
            Some(ctx.add(Expr::Sub(dl, dr)))
        }
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            let dl = differentiate_symbolic_expr(ctx, l, var)?;
            let dr = differentiate_symbolic_expr(ctx, r, var)?;
            let term1 = mul2_raw(ctx, dl, r);
            let term2 = mul2_raw(ctx, l, dr);
            Some(ctx.add(Expr::Add(term1, term2)))
        }
        Expr::Div(l, r) => {
            let (l, r) = (*l, *r);
            let dl = differentiate_symbolic_expr(ctx, l, var)?;
            let dr = differentiate_symbolic_expr(ctx, r, var)?;
            let term1 = mul2_raw(ctx, dl, r);
            let term2 = mul2_raw(ctx, l, dr);
            let num = ctx.add(Expr::Sub(term1, term2));
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
                let term = mul2_raw(ctx, exp, pow_term);
                Some(mul2_raw(ctx, term, db))
            } else if !contains_named_var(ctx, base, var) {
                let ln_a = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
                let term = mul2_raw(ctx, expr, ln_a);
                Some(mul2_raw(ctx, term, de))
            } else {
                let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
                let term1 = mul2_raw(ctx, de, ln_base);
                let term2_num = mul2_raw(ctx, exp, db);
                let term2 = ctx.add(Expr::Div(term2_num, base));
                let inner = ctx.add(Expr::Add(term1, term2));
                Some(mul2_raw(ctx, expr, inner))
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
                    Some(mul2_raw(ctx, cos_u, da))
                }
                Some(BuiltinFn::Cos) => {
                    let sin_u = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
                    let neg_sin = ctx.add(Expr::Neg(sin_u));
                    Some(mul2_raw(ctx, neg_sin, da))
                }
                Some(BuiltinFn::Tan) => {
                    let cos_u = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
                    let two = ctx.num(2);
                    let cos_sq = ctx.add(Expr::Pow(cos_u, two));
                    let one = ctx.num(1);
                    let sec_sq = ctx.add(Expr::Div(one, cos_sq));
                    Some(mul2_raw(ctx, sec_sq, da))
                }
                Some(BuiltinFn::Exp) => Some(mul2_raw(ctx, expr, da)),
                Some(BuiltinFn::Ln) => Some(ctx.add(Expr::Div(da, arg))),
                Some(BuiltinFn::Abs) => {
                    let term = ctx.add(Expr::Div(arg, expr));
                    Some(mul2_raw(ctx, term, da))
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
}
