//! Symbolic integration helpers shared by integration-facing rule layers.

use crate::build::mul2_raw;
use crate::expr_predicates::contains_named_var;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::One;

/// Integrate `expr` with respect to `var` using a small set of symbolic rules.
pub fn integrate_symbolic_expr(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    // Extract variant info in one borrow, then process with owned ExprId values.
    enum IntKind {
        Add(ExprId, ExprId),
        Sub(ExprId, ExprId),
        Mul(ExprId, ExprId),
        Pow(ExprId, ExprId),
        Variable(usize),
        Div(ExprId, ExprId),
        Function(usize, Vec<ExprId>),
        Other,
    }
    let kind = match ctx.get(expr) {
        Expr::Add(l, r) => IntKind::Add(*l, *r),
        Expr::Sub(l, r) => IntKind::Sub(*l, *r),
        Expr::Mul(l, r) => IntKind::Mul(*l, *r),
        Expr::Pow(b, e) => IntKind::Pow(*b, *e),
        Expr::Variable(s) => IntKind::Variable(*s),
        Expr::Div(n, d) => IntKind::Div(*n, *d),
        Expr::Function(f, args) => IntKind::Function(*f, args.clone()),
        _ => IntKind::Other,
    };

    if let IntKind::Add(l, r) = kind {
        let int_l = integrate_symbolic_expr(ctx, l, var)?;
        let int_r = integrate_symbolic_expr(ctx, r, var)?;
        return Some(ctx.add(Expr::Add(int_l, int_r)));
    }

    if let IntKind::Sub(l, r) = kind {
        let int_l = integrate_symbolic_expr(ctx, l, var)?;
        let int_r = integrate_symbolic_expr(ctx, r, var)?;
        return Some(ctx.add(Expr::Sub(int_l, int_r)));
    }

    if let IntKind::Mul(l, r) = kind {
        if !contains_named_var(ctx, l, var) {
            if let Some(int_r) = integrate_symbolic_expr(ctx, r, var) {
                return Some(mul2_raw(ctx, l, int_r));
            }
        }
        if !contains_named_var(ctx, r, var) {
            if let Some(int_l) = integrate_symbolic_expr(ctx, l, var) {
                return Some(mul2_raw(ctx, r, int_l));
            }
        }
    }

    if !contains_named_var(ctx, expr, var) {
        let var_expr = ctx.var(var);
        return Some(mul2_raw(ctx, expr, var_expr));
    }

    if let IntKind::Pow(base, exp) = kind {
        if let Some((a, _)) = get_linear_coeffs(ctx, base, var) {
            if !contains_named_var(ctx, exp, var) {
                if let Expr::Number(n) = ctx.get(exp) {
                    if *n == BigRational::from_integer((-1).into()) {
                        let ln_u = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
                        return Some(ctx.add(Expr::Div(ln_u, a)));
                    }
                }

                let one = ctx.num(1);
                let new_exp = ctx.add(Expr::Add(exp, one));

                let is_a_one = if let Expr::Number(n) = ctx.get(a) {
                    n.is_one()
                } else {
                    false
                };
                let new_denom = if is_a_one {
                    new_exp
                } else {
                    mul2_raw(ctx, a, new_exp)
                };

                let pow_expr = ctx.add(Expr::Pow(base, new_exp));
                return Some(ctx.add(Expr::Div(pow_expr, new_denom)));
            }
        }

        if !contains_named_var(ctx, base, var) {
            if let Some((a, _)) = get_linear_coeffs(ctx, exp, var) {
                let is_a_one = if let Expr::Number(n) = ctx.get(a) {
                    n.is_one()
                } else {
                    false
                };

                let is_e = if let Expr::Constant(c) = ctx.get(base) {
                    c == &cas_ast::Constant::E
                } else {
                    false
                };

                if is_e {
                    if is_a_one {
                        return Some(expr);
                    }
                    return Some(ctx.add(Expr::Div(expr, a)));
                }

                let ln_c = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
                let denom = if is_a_one {
                    ln_c
                } else {
                    mul2_raw(ctx, a, ln_c)
                };
                return Some(ctx.add(Expr::Div(expr, denom)));
            }
        }
    }

    if let IntKind::Variable(sym_id) = kind {
        if ctx.sym_name(sym_id) == var {
            let var_expr = ctx.var(var);
            let two = ctx.num(2);
            let pow_expr = ctx.add(Expr::Pow(var_expr, two));
            return Some(ctx.add(Expr::Div(pow_expr, two)));
        }
    }

    if let IntKind::Div(num, den) = kind {
        if let Expr::Number(n) = ctx.get(num) {
            if n.is_one() {
                if let Some((a, _)) = get_linear_coeffs(ctx, den, var) {
                    let ln_den = ctx.call_builtin(BuiltinFn::Ln, vec![den]);
                    return Some(ctx.add(Expr::Div(ln_den, a)));
                }
            }
        }
    }

    if let IntKind::Function(fn_id, args) = kind {
        if args.len() == 1 {
            let arg = args[0];
            if let Some((a, _)) = get_linear_coeffs(ctx, arg, var) {
                let is_a_one = if let Expr::Number(n) = ctx.get(a) {
                    n.is_one()
                } else {
                    false
                };

                match ctx.builtin_of(fn_id) {
                    Some(BuiltinFn::Sin) => {
                        let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
                        let integral = ctx.add(Expr::Neg(cos_arg));
                        if is_a_one {
                            return Some(integral);
                        }
                        return Some(ctx.add(Expr::Div(integral, a)));
                    }
                    Some(BuiltinFn::Cos) => {
                        let integral = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
                        if is_a_one {
                            return Some(integral);
                        }
                        return Some(ctx.add(Expr::Div(integral, a)));
                    }
                    Some(BuiltinFn::Exp) => {
                        if is_a_one {
                            return Some(expr);
                        }
                        return Some(ctx.add(Expr::Div(expr, a)));
                    }
                    _ => {}
                }
            }
        }
    }

    None
}

/// Returns `(a, b)` such that `expr = a*var + b`.
pub fn get_linear_coeffs(ctx: &mut Context, expr: ExprId, var: &str) -> Option<(ExprId, ExprId)> {
    if !contains_named_var(ctx, expr, var) {
        return Some((ctx.num(0), expr));
    }

    match ctx.get(expr) {
        Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var => Some((ctx.num(1), ctx.num(0))),
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            if !contains_named_var(ctx, l, var) && is_var(ctx, r, var) {
                return Some((l, ctx.num(0)));
            }
            if is_var(ctx, l, var) && !contains_named_var(ctx, r, var) {
                return Some((r, ctx.num(0)));
            }
            None
        }
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            let l_coeffs = get_linear_coeffs(ctx, l, var);
            let r_coeffs = get_linear_coeffs(ctx, r, var);

            if let (Some((a1, b1)), Some((a2, b2))) = (l_coeffs, r_coeffs) {
                if !contains_named_var(ctx, a1, var) && !contains_named_var(ctx, a2, var) {
                    let a = ctx.add(Expr::Add(a1, a2));
                    let b = ctx.add(Expr::Add(b1, b2));
                    return Some((a, b));
                }
            }
            None
        }
        Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            let l_coeffs = get_linear_coeffs(ctx, l, var);
            let r_coeffs = get_linear_coeffs(ctx, r, var);
            if let (Some((a1, b1)), Some((a2, b2))) = (l_coeffs, r_coeffs) {
                if !contains_named_var(ctx, a1, var) && !contains_named_var(ctx, a2, var) {
                    let a = ctx.add(Expr::Sub(a1, a2));
                    let b = ctx.add(Expr::Sub(b1, b2));
                    return Some((a, b));
                }
            }
            None
        }
        _ => None,
    }
}

fn is_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    if let Expr::Variable(sym_id) = ctx.get(expr) {
        ctx.sym_name(*sym_id) == var
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::{get_linear_coeffs, integrate_symbolic_expr};
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: cas_ast::ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn integrates_simple_power() {
        let mut ctx = Context::new();
        let expr = parse("x^2", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "x^(1 + 2) / (1 + 2)");
    }

    #[test]
    fn integrates_linear_trig_substitution() {
        let mut ctx = Context::new();
        let expr = parse("sin(2*x)", &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
        assert_eq!(rendered(&ctx, out), "-cos(2 * x) / 2");
    }

    #[test]
    fn extracts_linear_coeffs() {
        let mut ctx = Context::new();
        let expr = parse("2*x + 3", &mut ctx).expect("parse");
        let (a, b) = get_linear_coeffs(&mut ctx, expr, "x").expect("coeffs");
        let a_text = rendered(&ctx, a);
        assert!(a_text == "2" || a_text == "0 + 2");
        let b_text = rendered(&ctx, b);
        assert!(b_text == "3" || b_text == "0 + 3");
    }
}
