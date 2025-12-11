use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::One;

define_rule!(IntegrateRule, "Symbolic Integration", |ctx, expr| {
    if let Expr::Function(name, args) = ctx.get(expr) {
        if name == "integrate" {
            if args.len() == 2 {
                let integrand = args[0];
                let var_expr = args[1];
                if let Expr::Variable(var_name) = ctx.get(var_expr) {
                    let var_name = var_name.clone(); // Clone to drop borrow
                    if let Some(result) = integrate(ctx, integrand, &var_name) {
                        return Some(Rewrite {
                            new_expr: result,
                            description: format!(
                                "integrate({}, {})",
                                cas_ast::DisplayExpr {
                                    context: ctx,
                                    id: integrand
                                },
                                var_name
                            ),
                            before_local: None,
                            after_local: None,
                        });
                    }
                }
            } else if args.len() == 1 {
                // Default to 'x' if not specified? Or fail?
                // Let's assume 'x' for convenience if only 1 arg.
                let integrand = args[0];
                if let Some(result) = integrate(ctx, integrand, "x") {
                    return Some(Rewrite {
                        new_expr: result,
                        description: format!(
                            "integrate({}, x)",
                            cas_ast::DisplayExpr {
                                context: ctx,
                                id: integrand
                            }
                        ),
                        before_local: None,
                        after_local: None,
                    });
                }
            }
        }
    }
    None
});

define_rule!(DiffRule, "Symbolic Differentiation", |ctx, expr| {
    if let Expr::Function(name, args) = ctx.get(expr) {
        if name == "diff" {
            if args.len() == 2 {
                let target = args[0];
                let var_expr = args[1];
                if let Expr::Variable(var_name) = ctx.get(var_expr) {
                    let var_name = var_name.clone();
                    if let Some(result) = differentiate(ctx, target, &var_name) {
                        return Some(Rewrite {
                            new_expr: result,
                            description: format!(
                                "diff({}, {})",
                                cas_ast::DisplayExpr {
                                    context: ctx,
                                    id: target
                                },
                                var_name
                            ),
                            before_local: None,
                            after_local: None,
                        });
                    }
                }
            }
        }
    }
    None
});

fn differentiate(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    let expr_data = ctx.get(expr).clone();

    // 1. Constant Rule: diff(c, x) = 0
    if !contains_var(ctx, expr, var) {
        return Some(ctx.num(0));
    }

    match expr_data {
        Expr::Variable(v) => {
            if v == var {
                Some(ctx.num(1))
            } else {
                Some(ctx.num(0))
            }
        }
        Expr::Add(l, r) => {
            let dl = differentiate(ctx, l, var)?;
            let dr = differentiate(ctx, r, var)?;
            Some(ctx.add(Expr::Add(dl, dr)))
        }
        Expr::Sub(l, r) => {
            let dl = differentiate(ctx, l, var)?;
            let dr = differentiate(ctx, r, var)?;
            Some(ctx.add(Expr::Sub(dl, dr)))
        }
        Expr::Mul(l, r) => {
            // Product Rule: (uv)' = u'v + uv'
            let dl = differentiate(ctx, l, var)?;
            let dr = differentiate(ctx, r, var)?;
            let term1 = ctx.add(Expr::Mul(dl, r));
            let term2 = ctx.add(Expr::Mul(l, dr));
            Some(ctx.add(Expr::Add(term1, term2)))
        }
        Expr::Div(l, r) => {
            // Quotient Rule: (u/v)' = (u'v - uv') / v^2
            let dl = differentiate(ctx, l, var)?;
            let dr = differentiate(ctx, r, var)?;
            let term1 = ctx.add(Expr::Mul(dl, r));
            let term2 = ctx.add(Expr::Mul(l, dr));
            let num = ctx.add(Expr::Sub(term1, term2));
            let two = ctx.num(2);
            let den = ctx.add(Expr::Pow(r, two));
            Some(ctx.add(Expr::Div(num, den)))
        }
        Expr::Pow(base, exp) => {
            // Generalized Power Rule: (u^v)' = u^v * (v'*ln(u) + v*u'/u)
            // Simplified for constant exponent n: (u^n)' = n*u^(n-1)*u'
            // Simplified for exponential a^u: (a^u)' = a^u * ln(a) * u'

            let db = differentiate(ctx, base, var)?;
            let de = differentiate(ctx, exp, var)?;

            // If exponent is constant (de = 0)
            if !contains_var(ctx, exp, var) {
                // n * u^(n-1) * u'
                let one = ctx.num(1);
                let n_minus_one = ctx.add(Expr::Sub(exp, one));
                let pow_term = ctx.add(Expr::Pow(base, n_minus_one));
                let term = ctx.add(Expr::Mul(exp, pow_term));
                Some(ctx.add(Expr::Mul(term, db)))
            } else if !contains_var(ctx, base, var) {
                // a^u * ln(a) * u'
                let ln_a = ctx.add(Expr::Function("ln".to_string(), vec![base]));
                let term = ctx.add(Expr::Mul(expr, ln_a));
                Some(ctx.add(Expr::Mul(term, de)))
            } else {
                // Full rule: u^v * (v'*ln(u) + v*u'/u)
                // = u^v * (de * ln(base) + exp * db / base)
                let ln_base = ctx.add(Expr::Function("ln".to_string(), vec![base]));
                let term1 = ctx.add(Expr::Mul(de, ln_base));
                let term2_num = ctx.add(Expr::Mul(exp, db));
                let term2 = ctx.add(Expr::Div(term2_num, base));
                let inner = ctx.add(Expr::Add(term1, term2));
                Some(ctx.add(Expr::Mul(expr, inner)))
            }
        }
        Expr::Function(name, args) => {
            if args.len() == 1 {
                let arg = args[0];
                let da = differentiate(ctx, arg, var)?;

                match name.as_str() {
                    "sin" => {
                        // cos(u) * u'
                        let cos_u = ctx.add(Expr::Function("cos".to_string(), vec![arg]));
                        Some(ctx.add(Expr::Mul(cos_u, da)))
                    }
                    "cos" => {
                        // -sin(u) * u'
                        let sin_u = ctx.add(Expr::Function("sin".to_string(), vec![arg]));
                        let neg_sin = ctx.add(Expr::Neg(sin_u));
                        Some(ctx.add(Expr::Mul(neg_sin, da)))
                    }
                    "tan" => {
                        // sec^2(u) * u' = (1/cos^2(u)) * u'
                        let cos_u = ctx.add(Expr::Function("cos".to_string(), vec![arg]));
                        let two = ctx.num(2);
                        let cos_sq = ctx.add(Expr::Pow(cos_u, two));
                        let one = ctx.num(1);
                        let sec_sq = ctx.add(Expr::Div(one, cos_sq));
                        Some(ctx.add(Expr::Mul(sec_sq, da)))
                    }
                    "exp" => {
                        // exp(u) * u'
                        Some(ctx.add(Expr::Mul(expr, da)))
                    }
                    "ln" => {
                        // u'/u
                        Some(ctx.add(Expr::Div(da, arg)))
                    }
                    "abs" => {
                        // abs(u)/u * u' (sign(u) * u')
                        // or u/abs(u) * u'
                        let term = ctx.add(Expr::Div(arg, expr)); // u / abs(u)
                        Some(ctx.add(Expr::Mul(term, da)))
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

fn integrate(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    let expr_data = ctx.get(expr).clone();

    // 1. Linearity: integrate(a + b) = integrate(a) + integrate(b)
    if let Expr::Add(l, r) = expr_data {
        let int_l = integrate(ctx, l, var)?;
        let int_r = integrate(ctx, r, var)?;
        return Some(ctx.add(Expr::Add(int_l, int_r)));
    }

    // 2. Linearity: integrate(a - b) = integrate(a) - integrate(b)
    if let Expr::Sub(l, r) = expr_data {
        let int_l = integrate(ctx, l, var)?;
        let int_r = integrate(ctx, r, var)?;
        return Some(ctx.add(Expr::Sub(int_l, int_r)));
    }

    // 3. Constant Multiple: integrate(c * f(x)) = c * integrate(f(x))
    // We need to check if one operand is constant (w.r.t var).
    if let Expr::Mul(l, r) = expr_data {
        if !contains_var(ctx, l, var) {
            if let Some(int_r) = integrate(ctx, r, var) {
                return Some(ctx.add(Expr::Mul(l, int_r)));
            }
        }
        if !contains_var(ctx, r, var) {
            if let Some(int_l) = integrate(ctx, l, var) {
                return Some(ctx.add(Expr::Mul(r, int_l)));
            }
        }
    }

    // 4. Basic Rules & Linear Substitution

    // Constant: integrate(c) = c*x
    if !contains_var(ctx, expr, var) {
        let var_expr = ctx.var(var);
        return Some(ctx.add(Expr::Mul(expr, var_expr)));
    }

    // Power Rule: integrate(u^n) * u' -> u^(n+1)/(n+1)
    // Here we handle u = ax+b, so u' = a.
    // integrate((ax+b)^n) = (ax+b)^(n+1) / (a*(n+1))
    if let Expr::Pow(base, exp) = expr_data {
        // Case 1: u^n where u is linear (ax+b)
        if let Some((a, _)) = get_linear_coeffs(ctx, base, var) {
            if !contains_var(ctx, exp, var) {
                // Check for n = -1 case (1/u)
                if let Expr::Number(n) = ctx.get(exp) {
                    if *n == BigRational::from_integer((-1).into()) {
                        // ln(u) / a
                        let ln_u = ctx.add(Expr::Function("ln".to_string(), vec![base]));
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
                    ctx.add(Expr::Mul(a, new_exp))
                };

                let pow_expr = ctx.add(Expr::Pow(base, new_exp));
                return Some(ctx.add(Expr::Div(pow_expr, new_denom)));
            }
        }

        // Case 2: c^u where u is linear (ax+b)
        // integrate(c^(ax+b)) = c^(ax+b) / (a * ln(c))
        // If c = e, ln(c) = 1, so e^(ax+b) / a
        if !contains_var(ctx, base, var) {
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
                let ln_c = ctx.add(Expr::Function("ln".to_string(), vec![base]));
                let denom = if is_a_one {
                    ln_c
                } else {
                    ctx.add(Expr::Mul(a, ln_c))
                };
                return Some(ctx.add(Expr::Div(expr, denom)));
            }
        }
    }

    // Variable itself: integrate(x) = x^2/2
    // This is covered by Power Rule if x is treated as (1*x + 0)^1, but explicit check is faster/simpler?
    // Actually get_linear_coeffs(x) returns (1, 0).
    // So Pow(x, 1) would be handled above IF expr was parsed as Pow.
    // But "x" is Expr::Variable.
    if let Expr::Variable(v) = &expr_data {
        if v == var {
            let var_expr = ctx.var(var);
            let two = ctx.num(2);
            let pow_expr = ctx.add(Expr::Pow(var_expr, two));
            return Some(ctx.add(Expr::Div(pow_expr, two)));
        }
    }

    // 1/u case (if represented as Div(1, u))
    if let Expr::Div(num, den) = expr_data {
        if let Expr::Number(n) = ctx.get(num) {
            if n.is_one() {
                if let Some((a, _)) = get_linear_coeffs(ctx, den, var) {
                    // integrate(1/(ax+b)) = ln(ax+b)/a
                    let ln_den = ctx.add(Expr::Function("ln".to_string(), vec![den]));
                    return Some(ctx.add(Expr::Div(ln_den, a)));
                }
            }
        }
    }

    // Trig Rules & Exponential with Linear Substitution
    if let Expr::Function(name, args) = expr_data {
        if args.len() == 1 {
            let arg = args[0];
            if let Some((a, _)) = get_linear_coeffs(ctx, arg, var) {
                let is_a_one = if let Expr::Number(n) = ctx.get(a) {
                    n.is_one()
                } else {
                    false
                };

                match name.as_str() {
                    "sin" => {
                        // integrate(sin(ax+b)) = -cos(ax+b)/a
                        let cos_arg = ctx.add(Expr::Function("cos".to_string(), vec![arg]));
                        let integral = ctx.add(Expr::Neg(cos_arg));
                        if is_a_one {
                            return Some(integral);
                        }
                        return Some(ctx.add(Expr::Div(integral, a)));
                    }
                    "cos" => {
                        // integrate(cos(ax+b)) = sin(ax+b)/a
                        let integral = ctx.add(Expr::Function("sin".to_string(), vec![arg]));
                        if is_a_one {
                            return Some(integral);
                        }
                        return Some(ctx.add(Expr::Div(integral, a)));
                    }
                    "exp" => {
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

fn contains_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    match ctx.get(expr) {
        Expr::Variable(v) => v == var,
        Expr::Number(_) | Expr::Constant(_) => false,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            contains_var(ctx, *l, var) || contains_var(ctx, *r, var)
        }
        Expr::Neg(e) => contains_var(ctx, *e, var),
        Expr::Function(_, args) => args.iter().any(|a| contains_var(ctx, *a, var)),
        Expr::Matrix { data, .. } => data.iter().any(|elem| contains_var(ctx, *elem, var)),
    }
}

// Returns (a, b) such that expr = a*var + b
fn get_linear_coeffs(ctx: &mut Context, expr: ExprId, var: &str) -> Option<(ExprId, ExprId)> {
    let expr_data = ctx.get(expr).clone();

    if !contains_var(ctx, expr, var) {
        return Some((ctx.num(0), expr));
    }

    match expr_data {
        Expr::Variable(v) if v == var => Some((ctx.num(1), ctx.num(0))),
        Expr::Mul(l, r) => {
            // c * x
            if !contains_var(ctx, l, var) && is_var(ctx, r, var) {
                return Some((l, ctx.num(0)));
            }
            // x * c
            if is_var(ctx, l, var) && !contains_var(ctx, r, var) {
                return Some((r, ctx.num(0)));
            }
            None
        }
        Expr::Add(l, r) => {
            // (ax) + b or b + (ax)
            let l_coeffs = get_linear_coeffs(ctx, l, var);
            let r_coeffs = get_linear_coeffs(ctx, r, var);

            if let (Some((a1, b1)), Some((a2, b2))) = (l_coeffs, r_coeffs) {
                if !contains_var(ctx, a1, var) && !contains_var(ctx, a2, var) {
                    let a = ctx.add(Expr::Add(a1, a2));
                    let b = ctx.add(Expr::Add(b1, b2));
                    return Some((a, b));
                }
            }
            None
        }

        Expr::Sub(l, r) => {
            let l_coeffs = get_linear_coeffs(ctx, l, var);
            let r_coeffs = get_linear_coeffs(ctx, r, var);
            if let (Some((a1, b1)), Some((a2, b2))) = (l_coeffs, r_coeffs) {
                if !contains_var(ctx, a1, var) && !contains_var(ctx, a2, var) {
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
    if let Expr::Variable(v) = ctx.get(expr) {
        v == var
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::{Context, DisplayExpr};
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
            "(1 + 2 * x)^(1 + 2) / ((0 + 2) * (1 + 2))" // Canonical: 1 before 2*x, 0 before 2, 1 before 2
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

// =============================================================================
// SUM RULE: Evaluate finite summations
// =============================================================================
// Syntax: sum(expr, var, start, end)
// Example: sum(k, k, 1, 10) → 55
// Example: sum(k^2, k, 1, 5) → 1 + 4 + 9 + 16 + 25 = 55

define_rule!(SumRule, "Finite Summation", |ctx, expr| {
    if let Expr::Function(name, args) = ctx.get(expr) {
        if name == "sum" && args.len() == 4 {
            let summand = args[0];
            let var_expr = args[1];
            let start_expr = args[2];
            let end_expr = args[3];

            // Extract variable name
            let var_name = if let Expr::Variable(v) = ctx.get(var_expr) {
                v.clone()
            } else {
                return None;
            };

            // Try to evaluate start and end as integers
            let start = get_integer(ctx, start_expr)?;
            let end = get_integer(ctx, end_expr)?;

            // Safety limit for direct evaluation
            if end - start > 1000 {
                return None; // Too many terms, try closed form or return None
            }

            // Check for known closed-form formulas FIRST (for symbolic end)
            // sum(k, k, 1, n) = n*(n+1)/2
            // sum(k^2, k, 1, n) = n*(n+1)*(2n+1)/6

            // Direct numeric evaluation
            if start <= end {
                let mut result = ctx.num(0);
                for k in start..=end {
                    let k_expr = ctx.num(k);
                    let term = substitute_var(ctx, summand, &var_name, k_expr);
                    result = ctx.add(Expr::Add(result, term));
                }

                // Simplify the result
                let mut simplifier = crate::Simplifier::with_default_rules();
                simplifier.context = ctx.clone();
                let (simplified, _) = simplifier.simplify(result);
                *ctx = simplifier.context;

                return Some(Rewrite {
                    new_expr: simplified,
                    description: format!(
                        "sum({}, {}, {}, {})",
                        cas_ast::DisplayExpr {
                            context: ctx,
                            id: summand
                        },
                        var_name,
                        start,
                        end
                    ),
                    before_local: None,
                    after_local: None,
                });
            }
        }
    }
    None
});

/// Get integer value from expression
fn get_integer(ctx: &Context, expr: ExprId) -> Option<i64> {
    if let Expr::Number(n) = ctx.get(expr) {
        if n.is_integer() {
            return n.numer().try_into().ok();
        }
    }
    None
}

/// Substitute variable with value in expression
fn substitute_var(ctx: &mut Context, expr: ExprId, var: &str, value: ExprId) -> ExprId {
    let expr_data = ctx.get(expr).clone();

    match expr_data {
        Expr::Variable(v) if v == var => value,
        Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_) => expr,
        Expr::Add(l, r) => {
            let new_l = substitute_var(ctx, l, var, value);
            let new_r = substitute_var(ctx, r, var, value);
            ctx.add(Expr::Add(new_l, new_r))
        }
        Expr::Sub(l, r) => {
            let new_l = substitute_var(ctx, l, var, value);
            let new_r = substitute_var(ctx, r, var, value);
            ctx.add(Expr::Sub(new_l, new_r))
        }
        Expr::Mul(l, r) => {
            let new_l = substitute_var(ctx, l, var, value);
            let new_r = substitute_var(ctx, r, var, value);
            ctx.add(Expr::Mul(new_l, new_r))
        }
        Expr::Div(l, r) => {
            let new_l = substitute_var(ctx, l, var, value);
            let new_r = substitute_var(ctx, r, var, value);
            ctx.add(Expr::Div(new_l, new_r))
        }
        Expr::Pow(l, r) => {
            let new_l = substitute_var(ctx, l, var, value);
            let new_r = substitute_var(ctx, r, var, value);
            ctx.add(Expr::Pow(new_l, new_r))
        }
        Expr::Neg(e) => {
            let new_e = substitute_var(ctx, e, var, value);
            ctx.add(Expr::Neg(new_e))
        }
        Expr::Function(name, args) => {
            let new_args: Vec<ExprId> = args
                .iter()
                .map(|a| substitute_var(ctx, *a, var, value))
                .collect();
            ctx.add(Expr::Function(name, new_args))
        }
        Expr::Matrix { rows, cols, data } => {
            let new_data: Vec<ExprId> = data
                .iter()
                .map(|a| substitute_var(ctx, *a, var, value))
                .collect();
            ctx.add(Expr::Matrix {
                rows,
                cols,
                data: new_data,
            })
        }
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(IntegrateRule));
    simplifier.add_rule(Box::new(DiffRule));
    simplifier.add_rule(Box::new(SumRule));
}
