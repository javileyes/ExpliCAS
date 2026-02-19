//! Symbolic differentiation engine.
//!
//! Contains the `differentiate()` function implementing standard
//! differentiation rules: constant, sum, product, quotient, power, chain.

use crate::build::mul2_raw;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;

pub(crate) fn differentiate(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    // 1. Constant Rule: diff(c, x) = 0
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
            let dl = differentiate(ctx, l, var)?;
            let dr = differentiate(ctx, r, var)?;
            Some(ctx.add(Expr::Add(dl, dr)))
        }
        Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            let dl = differentiate(ctx, l, var)?;
            let dr = differentiate(ctx, r, var)?;
            Some(ctx.add(Expr::Sub(dl, dr)))
        }
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            // Product Rule: (uv)' = u'v + uv'
            let dl = differentiate(ctx, l, var)?;
            let dr = differentiate(ctx, r, var)?;
            let term1 = mul2_raw(ctx, dl, r);
            let term2 = mul2_raw(ctx, l, dr);
            Some(ctx.add(Expr::Add(term1, term2)))
        }
        Expr::Div(l, r) => {
            let (l, r) = (*l, *r);
            // Quotient Rule: (u/v)' = (u'v - uv') / v^2
            let dl = differentiate(ctx, l, var)?;
            let dr = differentiate(ctx, r, var)?;
            let term1 = mul2_raw(ctx, dl, r);
            let term2 = mul2_raw(ctx, l, dr);
            let num = ctx.add(Expr::Sub(term1, term2));
            let two = ctx.num(2);
            let den = ctx.add(Expr::Pow(r, two));
            Some(ctx.add(Expr::Div(num, den)))
        }
        Expr::Pow(base, exp) => {
            let (base, exp) = (*base, *exp);
            // Generalized Power Rule: (u^v)' = u^v * (v'*ln(u) + v*u'/u)
            // Simplified for constant exponent n: (u^n)' = n*u^(n-1)*u'
            // Simplified for exponential a^u: (a^u)' = a^u * ln(a) * u'

            let db = differentiate(ctx, base, var)?;
            let de = differentiate(ctx, exp, var)?;

            // If exponent is constant (de = 0)
            if !contains_named_var(ctx, exp, var) {
                // n * u^(n-1) * u'
                let one = ctx.num(1);
                let n_minus_one = ctx.add(Expr::Sub(exp, one));
                let pow_term = ctx.add(Expr::Pow(base, n_minus_one));
                let term = mul2_raw(ctx, exp, pow_term);
                Some(mul2_raw(ctx, term, db))
            } else if !contains_named_var(ctx, base, var) {
                // a^u * ln(a) * u'
                let ln_a = ctx.call_builtin(cas_ast::BuiltinFn::Ln, vec![base]);
                let term = mul2_raw(ctx, expr, ln_a);
                Some(mul2_raw(ctx, term, de))
            } else {
                // Full rule: u^v * (v'*ln(u) + v*u'/u)
                // = u^v * (de * ln(base) + exp * db / base)
                let ln_base = ctx.call_builtin(cas_ast::BuiltinFn::Ln, vec![base]);
                let term1 = mul2_raw(ctx, de, ln_base);
                let term2_num = mul2_raw(ctx, exp, db);
                let term2 = ctx.add(Expr::Div(term2_num, base));
                let inner = ctx.add(Expr::Add(term1, term2));
                Some(mul2_raw(ctx, expr, inner))
            }
        }
        Expr::Function(fn_id, ref args) => {
            let (fn_id, args) = (*fn_id, args.clone());
            if args.len() == 1 {
                let arg = args[0];
                let da = differentiate(ctx, arg, var)?;

                match ctx.builtin_of(fn_id) {
                    Some(BuiltinFn::Sin) => {
                        // cos(u) * u'
                        let cos_u = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![arg]);
                        Some(mul2_raw(ctx, cos_u, da))
                    }
                    Some(BuiltinFn::Cos) => {
                        // -sin(u) * u'
                        let sin_u = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![arg]);
                        let neg_sin = ctx.add(Expr::Neg(sin_u));
                        Some(mul2_raw(ctx, neg_sin, da))
                    }
                    Some(BuiltinFn::Tan) => {
                        // sec^2(u) * u' = (1/cos^2(u)) * u'
                        let cos_u = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![arg]);
                        let two = ctx.num(2);
                        let cos_sq = ctx.add(Expr::Pow(cos_u, two));
                        let one = ctx.num(1);
                        let sec_sq = ctx.add(Expr::Div(one, cos_sq));
                        Some(mul2_raw(ctx, sec_sq, da))
                    }
                    Some(BuiltinFn::Exp) => {
                        // exp(u) * u'
                        Some(mul2_raw(ctx, expr, da))
                    }
                    Some(BuiltinFn::Ln) => {
                        // u'/u
                        Some(ctx.add(Expr::Div(da, arg)))
                    }
                    Some(BuiltinFn::Abs) => {
                        // abs(u)/u * u' (sign(u) * u')
                        // or u/abs(u) * u'
                        let term = ctx.add(Expr::Div(arg, expr)); // u / abs(u)
                        Some(mul2_raw(ctx, term, da))
                    }
                    _ => None, // Unknown function
                }
            } else {
                None // Multi-arg functions not supported yet (except log base?)
            }
        }
        _ => None,
    }
}
