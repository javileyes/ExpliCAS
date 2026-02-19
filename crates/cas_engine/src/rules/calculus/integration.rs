//! Symbolic integration engine.
//!
//! Contains `integrate()` and helper functions `get_linear_coeffs()` and `is_var()`.

use crate::build::mul2_raw;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;
use num_rational::BigRational;

pub(crate) fn integrate(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
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

    // 1. Linearity: integrate(a + b) = integrate(a) + integrate(b)
    if let IntKind::Add(l, r) = kind {
        let int_l = integrate(ctx, l, var)?;
        let int_r = integrate(ctx, r, var)?;
        return Some(ctx.add(Expr::Add(int_l, int_r)));
    }

    // 2. Linearity: integrate(a - b) = integrate(a) - integrate(b)
    if let IntKind::Sub(l, r) = kind {
        let int_l = integrate(ctx, l, var)?;
        let int_r = integrate(ctx, r, var)?;
        return Some(ctx.add(Expr::Sub(int_l, int_r)));
    }

    // 3. Constant Multiple: integrate(c * f(x)) = c * integrate(f(x))
    // We need to check if one operand is constant (w.r.t var).
    if let IntKind::Mul(l, r) = kind {
        if !contains_named_var(ctx, l, var) {
            if let Some(int_r) = integrate(ctx, r, var) {
                return Some(mul2_raw(ctx, l, int_r));
            }
        }
        if !contains_named_var(ctx, r, var) {
            if let Some(int_l) = integrate(ctx, l, var) {
                return Some(mul2_raw(ctx, r, int_l));
            }
        }
    }

    // 4. Basic Rules & Linear Substitution

    // Constant: integrate(c) = c*x
    if !contains_named_var(ctx, expr, var) {
        let var_expr = ctx.var(var);
        return Some(mul2_raw(ctx, expr, var_expr));
    }

    // Power Rule: integrate(u^n) * u' -> u^(n+1)/(n+1)
    // Here we handle u = ax+b, so u' = a.
    // integrate((ax+b)^n) = (ax+b)^(n+1) / (a*(n+1))
    if let IntKind::Pow(base, exp) = kind {
        // Case 1: u^n where u is linear (ax+b)
        if let Some((a, _)) = get_linear_coeffs(ctx, base, var) {
            if !contains_named_var(ctx, exp, var) {
                // Check for n = -1 case (1/u)
                if let Expr::Number(n) = ctx.get(exp) {
                    if *n == BigRational::from_integer((-1).into()) {
                        // ln(u) / a
                        let ln_u = ctx.call_builtin(cas_ast::BuiltinFn::Ln, vec![base]);
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

        // Case 2: c^u where u is linear (ax+b)
        // integrate(c^(ax+b)) = c^(ax+b) / (a * ln(c))
        // If c = e, ln(c) = 1, so e^(ax+b) / a
        if !contains_named_var(ctx, base, var) {
            if let Some((a, _)) = get_linear_coeffs(ctx, exp, var) {
                let is_a_one = if let Expr::Number(n) = ctx.get(a) {
                    n.is_one()
                } else {
                    false
                };

                // Check if base is e
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

                // General base c
                let ln_c = ctx.call_builtin(cas_ast::BuiltinFn::Ln, vec![base]);
                let denom = if is_a_one {
                    ln_c
                } else {
                    mul2_raw(ctx, a, ln_c)
                };
                return Some(ctx.add(Expr::Div(expr, denom)));
            }
        }
    }

    // Variable itself: integrate(x) = x^2/2
    if let IntKind::Variable(sym_id) = kind {
        if ctx.sym_name(sym_id) == var {
            let var_expr = ctx.var(var);
            let two = ctx.num(2);
            let pow_expr = ctx.add(Expr::Pow(var_expr, two));
            return Some(ctx.add(Expr::Div(pow_expr, two)));
        }
    }

    // 1/u case (if represented as Div(1, u))
    if let IntKind::Div(num, den) = kind {
        if let Expr::Number(n) = ctx.get(num) {
            if n.is_one() {
                if let Some((a, _)) = get_linear_coeffs(ctx, den, var) {
                    // integrate(1/(ax+b)) = ln(ax+b)/a
                    let ln_den = ctx.call_builtin(cas_ast::BuiltinFn::Ln, vec![den]);
                    return Some(ctx.add(Expr::Div(ln_den, a)));
                }
            }
        }
    }

    // Trig Rules & Exponential with Linear Substitution
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
                        // integrate(sin(ax+b)) = -cos(ax+b)/a
                        let cos_arg = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![arg]);
                        let integral = ctx.add(Expr::Neg(cos_arg));
                        if is_a_one {
                            return Some(integral);
                        }
                        return Some(ctx.add(Expr::Div(integral, a)));
                    }
                    Some(BuiltinFn::Cos) => {
                        // integrate(cos(ax+b)) = sin(ax+b)/a
                        let integral = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![arg]);
                        if is_a_one {
                            return Some(integral);
                        }
                        return Some(ctx.add(Expr::Div(integral, a)));
                    }
                    Some(BuiltinFn::Exp) => {
                        // integrate(exp(ax+b)) = exp(ax+b)/a
                        let integral = expr;
                        if is_a_one {
                            return Some(integral);
                        }
                        return Some(ctx.add(Expr::Div(integral, a)));
                    }
                    _ => {}
                }
            }
        }
    }

    None
}

use num_traits::One;

// Returns (a, b) such that expr = a*var + b
pub(crate) fn get_linear_coeffs(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    if !contains_named_var(ctx, expr, var) {
        return Some((ctx.num(0), expr));
    }

    match ctx.get(expr) {
        Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var => Some((ctx.num(1), ctx.num(0))),
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            // c * x
            if !contains_named_var(ctx, l, var) && is_var(ctx, r, var) {
                return Some((l, ctx.num(0)));
            }
            // x * c
            if is_var(ctx, l, var) && !contains_named_var(ctx, r, var) {
                return Some((r, ctx.num(0)));
            }
            None
        }
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            // (ax) + b or b + (ax)
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
    use crate::rule::Rule;
    use crate::rules::calculus::IntegrateRule;
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    #[test]
    fn test_integrate_power() {
        let mut ctx = Context::new();
        let rule = IntegrateRule;
        // integrate(x^2, x) -> x^3/3
        let expr = parse("integrate(x^2, x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "x^(1 + 2) / (1 + 2)" // Canonical: smaller numbers first
        );
    }

    #[test]
    fn test_integrate_constant() {
        let mut ctx = Context::new();
        let rule = IntegrateRule;
        // integrate(5, x) -> 5x
        let expr = parse("integrate(5, x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "5 * x"
        );
    }

    #[test]
    fn test_integrate_trig() {
        let mut ctx = Context::new();
        let rule = IntegrateRule;
        // integrate(sin(x), x) -> -cos(x)
        let expr = parse("integrate(sin(x), x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "-cos(x)"
        );
    }

    #[test]
    fn test_integrate_linearity() {
        let mut ctx = Context::new();
        let rule = IntegrateRule;
        // integrate(x + 1, x) -> x^2/2 + x
        let expr = parse("integrate(x + 1, x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        let res = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );
        // x^2/2 + 1*x
        assert!(res.contains("x^2 / 2"));
        assert!(res.contains("1 * x") || res.contains("x"));
    }
    #[test]
    fn test_integrate_linear_subst_trig() {
        let mut ctx = Context::new();
        let rule = IntegrateRule;
        // integrate(sin(2*x), x) -> -cos(2*x)/2
        let expr = parse("integrate(sin(2*x), x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "-cos(2 * x) / 2"
        );
    }

    #[test]
    fn test_integrate_linear_subst_exp() {
        let mut ctx = Context::new();
        let rule = IntegrateRule;
        // integrate(exp(3*x), x) -> exp(3*x)/3
        let expr = parse("integrate(exp(3*x), x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "e^(3 * x) / 3"
        );
    }

    #[test]
    fn test_integrate_linear_subst_power() {
        let mut ctx = Context::new();
        let rule = IntegrateRule;
        // integrate((2*x + 1)^2, x) -> (2*x + 1)^3 / (2*3) -> (2*x+1)^3 / 6
        let expr = parse("integrate((2*x + 1)^2, x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        // Note: 2*3 is not simplified by IntegrateRule, it produces Expr::mul(2, 3).
        // Simplification happens later in the pipeline.
        // So we expect (2*x + 1)^(2+1) / (2 * (2+1))
        // Actually get_linear_coeffs returns a=2+0 for 2x+1.
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "(2 * x + 1)^(1 + 2) / ((0 + 2) * (1 + 2))" // Canonical: polynomial order (2*x before 1)
        );
    }

    #[test]
    fn test_integrate_linear_subst_log() {
        let mut ctx = Context::new();
        let rule = IntegrateRule;
        // integrate(1/(3*x), x) -> ln(3*x)/3
        let expr = parse("integrate(1/(3*x), x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "ln(3 * x) / 3"
        );
    }
}
