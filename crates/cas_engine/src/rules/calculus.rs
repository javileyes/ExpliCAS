use crate::build::mul2_raw;
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
                            domain_assumption: None,
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
                        domain_assumption: None,
                    });
                }
            }
        }
    }
    None
});

define_rule!(DiffRule, "Symbolic Differentiation", |ctx, expr| {
    if let Expr::Function(name, args) = ctx.get(expr) {
        if name == "diff" && args.len() == 2 {
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
                        domain_assumption: None,
                    });
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
            let term1 = mul2_raw(ctx, dl, r);
            let term2 = mul2_raw(ctx, l, dr);
            Some(ctx.add(Expr::Add(term1, term2)))
        }
        Expr::Div(l, r) => {
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
                let term = mul2_raw(ctx, exp, pow_term);
                Some(mul2_raw(ctx, term, db))
            } else if !contains_var(ctx, base, var) {
                // a^u * ln(a) * u'
                let ln_a = ctx.add(Expr::Function("ln".to_string(), vec![base]));
                let term = mul2_raw(ctx, expr, ln_a);
                Some(mul2_raw(ctx, term, de))
            } else {
                // Full rule: u^v * (v'*ln(u) + v*u'/u)
                // = u^v * (de * ln(base) + exp * db / base)
                let ln_base = ctx.add(Expr::Function("ln".to_string(), vec![base]));
                let term1 = mul2_raw(ctx, de, ln_base);
                let term2_num = mul2_raw(ctx, exp, db);
                let term2 = ctx.add(Expr::Div(term2_num, base));
                let inner = ctx.add(Expr::Add(term1, term2));
                Some(mul2_raw(ctx, expr, inner))
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
                        Some(mul2_raw(ctx, cos_u, da))
                    }
                    "cos" => {
                        // -sin(u) * u'
                        let sin_u = ctx.add(Expr::Function("sin".to_string(), vec![arg]));
                        let neg_sin = ctx.add(Expr::Neg(sin_u));
                        Some(mul2_raw(ctx, neg_sin, da))
                    }
                    "tan" => {
                        // sec^2(u) * u' = (1/cos^2(u)) * u'
                        let cos_u = ctx.add(Expr::Function("cos".to_string(), vec![arg]));
                        let two = ctx.num(2);
                        let cos_sq = ctx.add(Expr::Pow(cos_u, two));
                        let one = ctx.num(1);
                        let sec_sq = ctx.add(Expr::Div(one, cos_sq));
                        Some(mul2_raw(ctx, sec_sq, da))
                    }
                    "exp" => {
                        // exp(u) * u'
                        Some(mul2_raw(ctx, expr, da))
                    }
                    "ln" => {
                        // u'/u
                        Some(ctx.add(Expr::Div(da, arg)))
                    }
                    "abs" => {
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
                return Some(mul2_raw(ctx, l, int_r));
            }
        }
        if !contains_var(ctx, r, var) {
            if let Some(int_l) = integrate(ctx, l, var) {
                return Some(mul2_raw(ctx, r, int_l));
            }
        }
    }

    // 4. Basic Rules & Linear Substitution

    // Constant: integrate(c) = c*x
    if !contains_var(ctx, expr, var) {
        let var_expr = ctx.var(var);
        return Some(mul2_raw(ctx, expr, var_expr));
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
                    mul2_raw(ctx, a, new_exp)
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
                    mul2_raw(ctx, a, ln_c)
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
        // SessionRef is a leaf - doesn't contain variables
        Expr::SessionRef(_) => false,
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

            // =====================================================================
            // TELESCOPING DETECTION: Check for rational sums that telescope
            // =====================================================================
            // Pattern: 1/(k*(k+a)) = (1/a) * (1/k - 1/(k+a))
            // Telescoping sum: sum from m to n = (1/a) * (1/m - 1/(n+a))

            if let Some(result) =
                try_telescoping_rational_sum(ctx, summand, &var_name, start_expr, end_expr)
            {
                return Some(Rewrite {
                    new_expr: result,
                    description: format!(
                        "Telescoping sum: Σ({}, {}) from {} to {}",
                        cas_ast::DisplayExpr {
                            context: ctx,
                            id: summand
                        },
                        var_name,
                        cas_ast::DisplayExpr {
                            context: ctx,
                            id: start_expr
                        },
                        cas_ast::DisplayExpr {
                            context: ctx,
                            id: end_expr
                        }
                    ),
                    before_local: None,
                    after_local: None,
                    domain_assumption: None,
                });
            }

            // Try to evaluate start and end as integers for numeric evaluation
            let start = get_integer(ctx, start_expr);
            let end = get_integer(ctx, end_expr);

            if let (Some(start), Some(end)) = (start, end) {
                // Safety limit for direct evaluation
                if end - start > 1000 {
                    return None; // Too many terms
                }

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
                        domain_assumption: None,
                    });
                }
            }
        }
    }
    None
});

/// Try to detect and evaluate telescoping rational sums
/// Pattern: 1/(k*(k+a)) where a is an integer constant
/// Result: (1/a) * (1/start - 1/(end+a))
fn try_telescoping_rational_sum(
    ctx: &mut Context,
    summand: ExprId,
    var: &str,
    start: ExprId,
    end: ExprId,
) -> Option<ExprId> {
    // Check if summand is 1/(k*(k+a)) or 1/((k+b)*(k+c))
    if let Expr::Div(num, den) = ctx.get(summand).clone() {
        // Numerator should be 1
        if let Expr::Number(n) = ctx.get(num) {
            if !n.is_one() {
                return None;
            }
        } else {
            return None;
        }

        // Denominator should be k*(k+a)
        if let Expr::Mul(left, right) = ctx.get(den).clone() {
            // Try to extract (k+b) and (k+c) where c-b = integer
            let factor1_offset = extract_linear_offset(ctx, left, var)?;
            let factor2_offset = extract_linear_offset(ctx, right, var)?;

            // Compute a = factor2_offset - factor1_offset
            let a = factor2_offset - factor1_offset;

            if a == 0 {
                return None; // Same factors, not telescoping
            }

            // For 1/(k*(k+a)), the partial fraction is:
            // 1/(k*(k+a)) = (1/a) * (1/k - 1/(k+a))
            // Sum from m to n = (1/a) * (1/(m+factor1_offset) - 1/(n+1+factor2_offset-1))
            //                 = (1/a) * (1/(m+b1) - 1/(n+b2+1)) where b1, b2 are offsets

            // Build result: (1/|a|) * (1/start_shifted - 1/(end_shifted+1))
            // where start_shifted = start + factor1_offset
            // and end_shifted = end + factor2_offset

            let a_expr = ctx.num(a.abs());

            // First term: 1/(start + factor1_offset)
            let offset1 = ctx.num(factor1_offset);
            let start_shifted = if factor1_offset == 0 {
                start
            } else {
                ctx.add(Expr::Add(start, offset1))
            };
            let one = ctx.num(1);
            let first_term = ctx.add(Expr::Div(one, start_shifted));

            // Second term: 1/(end + factor2_offset + 1) = 1/(end + factor2_offset + 1)
            // Wait, for telescoping: F(m) - F(n+1) where F(k) = 1/(k+offset)
            // So second term is 1/(end + 1 + factor1_offset)
            let offset1_plus_1 = ctx.num(factor1_offset + 1);
            let end_shifted = ctx.add(Expr::Add(end, offset1_plus_1));
            let one2 = ctx.num(1);
            let second_term = ctx.add(Expr::Div(one2, end_shifted));

            // Result = (1/a) * (first - second)
            let diff = ctx.add(Expr::Sub(first_term, second_term));

            let result = if a.abs() == 1 {
                if a > 0 {
                    diff
                } else {
                    ctx.add(Expr::Neg(diff))
                }
            } else {
                let unsigned_result = ctx.add(Expr::Div(diff, a_expr));
                if a > 0 {
                    unsigned_result
                } else {
                    ctx.add(Expr::Neg(unsigned_result))
                }
            };

            // Simplify the result
            let mut simplifier = crate::Simplifier::with_default_rules();
            simplifier.context = ctx.clone();
            let (simplified, _) = simplifier.simplify(result);
            *ctx = simplifier.context;

            return Some(simplified);
        }
    }

    None
}

/// Extract the constant offset from a linear expression: k+offset or k
/// Returns Some(offset) if expr = var + offset, None otherwise
fn extract_linear_offset(ctx: &Context, expr: ExprId, var: &str) -> Option<i64> {
    match ctx.get(expr) {
        // Just the variable: k+0
        Expr::Variable(v) if v == var => Some(0),

        // k + c
        Expr::Add(l, r) => {
            if let Expr::Variable(v) = ctx.get(*l) {
                if v == var {
                    return get_integer(ctx, *r);
                }
            }
            if let Expr::Variable(v) = ctx.get(*r) {
                if v == var {
                    return get_integer(ctx, *l);
                }
            }
            None
        }

        // k - c = k + (-c)
        Expr::Sub(l, r) => {
            if let Expr::Variable(v) = ctx.get(*l) {
                if v == var {
                    return get_integer(ctx, *r).map(|c| -c);
                }
            }
            None
        }

        _ => None,
    }
}

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
            mul2_raw(ctx, new_l, new_r)
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
        // SessionRef is a leaf - no substitution needed
        Expr::SessionRef(_) => expr,
    }
}

// =============================================================================
// PRODUCT RULE: Evaluate finite products (productorio)
// =============================================================================
// Syntax: product(expr, var, start, end)
// Example: product(k, k, 1, 5) → 120  (5!)
// Example: product((k+1)/k, k, 1, n) → n+1  (telescoping)

define_rule!(ProductRule, "Finite Product", |ctx, expr| {
    if let Expr::Function(name, args) = ctx.get(expr) {
        if name == "product" && args.len() == 4 {
            let factor = args[0];
            let var_expr = args[1];
            let start_expr = args[2];
            let end_expr = args[3];

            // Extract variable name
            let var_name = if let Expr::Variable(v) = ctx.get(var_expr) {
                v.clone()
            } else {
                return None;
            };

            // =====================================================================
            // TELESCOPING DETECTION: Check for rational products that telescope
            // =====================================================================
            // Pattern: (k+a)/k → product = (end+a)! / (start-1+a)! * (start-1)! / end!
            // Simple case: (k+1)/k → product = (n+1)/1 = n+1

            if let Some(result) =
                try_telescoping_product(ctx, factor, &var_name, start_expr, end_expr)
            {
                return Some(Rewrite {
                    new_expr: result,
                    description: format!(
                        "Telescoping product: Π({}, {}) from {} to {}",
                        cas_ast::DisplayExpr {
                            context: ctx,
                            id: factor
                        },
                        var_name,
                        cas_ast::DisplayExpr {
                            context: ctx,
                            id: start_expr
                        },
                        cas_ast::DisplayExpr {
                            context: ctx,
                            id: end_expr
                        }
                    ),
                    before_local: None,
                    after_local: None,
                    domain_assumption: None,
                });
            }

            // =====================================================================
            // FACTORIZABLE PRODUCT: Handle patterns like 1 - 1/k²
            // =====================================================================
            // Pattern: 1 - 1/k² = (k²-1)/k² = (k-1)(k+1)/k² = [(k-1)/k]·[(k+1)/k]
            // Each factor telescopes separately, then combine results

            if let Some(result) =
                try_factorizable_product(ctx, factor, &var_name, start_expr, end_expr)
            {
                return Some(Rewrite {
                    new_expr: result,
                    description: format!(
                        "Factorized telescoping product: Π({}, {}) from {} to {}",
                        cas_ast::DisplayExpr {
                            context: ctx,
                            id: factor
                        },
                        var_name,
                        cas_ast::DisplayExpr {
                            context: ctx,
                            id: start_expr
                        },
                        cas_ast::DisplayExpr {
                            context: ctx,
                            id: end_expr
                        }
                    ),
                    before_local: None,
                    after_local: None,
                    domain_assumption: None,
                });
            }

            // Try to evaluate start and end as integers for numeric evaluation
            let start = get_integer(ctx, start_expr);
            let end = get_integer(ctx, end_expr);

            if let (Some(start), Some(end)) = (start, end) {
                // Safety limit for direct evaluation
                if end - start > 1000 {
                    return None; // Too many terms
                }

                // Direct numeric evaluation
                if start <= end {
                    let mut result = ctx.num(1);
                    for k in start..=end {
                        let k_expr = ctx.num(k);
                        let term = substitute_var(ctx, factor, &var_name, k_expr);
                        result = mul2_raw(ctx, result, term);
                    }

                    // Simplify the result
                    let mut simplifier = crate::Simplifier::with_default_rules();
                    simplifier.context = ctx.clone();
                    let (simplified, _) = simplifier.simplify(result);
                    *ctx = simplifier.context;

                    return Some(Rewrite {
                        new_expr: simplified,
                        description: format!(
                            "product({}, {}, {}, {})",
                            cas_ast::DisplayExpr {
                                context: ctx,
                                id: factor
                            },
                            var_name,
                            start,
                            end
                        ),
                        before_local: None,
                        after_local: None,
                        domain_assumption: None,
                    });
                }
            }
        }
    }
    None
});

/// Try to detect and evaluate telescoping products
/// Pattern: (k+a)/(k+b) where a > b → product = (end+a)!/(start-1+a)! * (start-1+b)!/(end+b)!
/// Simple case: (k+1)/k → (n+1)/start
fn try_telescoping_product(
    ctx: &mut Context,
    factor: ExprId,
    var: &str,
    start: ExprId,
    end: ExprId,
) -> Option<ExprId> {
    // Check if factor is (k+a)/(k+b) pattern
    if let Expr::Div(num, den) = ctx.get(factor).clone() {
        // Extract offsets from numerator and denominator
        let num_offset = extract_linear_offset(ctx, num, var)?;
        let den_offset = extract_linear_offset(ctx, den, var)?;

        // For telescoping, we need num_offset > den_offset
        // (k+1)/k means num_offset=1, den_offset=0
        let shift = num_offset - den_offset;

        if shift <= 0 {
            return None; // Not a telescoping pattern
        }

        // For (k+a)/(k+b) with shift = a-b:
        // Product telescopes to: (end+a) * (end+a-1) * ... * (end+b+1) / (start+b-1) * ... / (start+a-1)
        //
        // Simple case shift=1: (k+1)/k from 1 to n
        // = (2/1) * (3/2) * ... * ((n+1)/n) = (n+1)/1 = n+1
        //
        // In general for shift=1:
        // Result = (end + num_offset) / (start + den_offset - 1 + 1) = (end + num_offset) / start_shifted

        if shift == 1 {
            // Simple telescoping: result = (end + num_offset) / (start + den_offset)
            // For (k+1)/k: result = (n+1) / 1 = n+1

            let end_plus_offset = if num_offset == 0 {
                end
            } else {
                let offset = ctx.num(num_offset);
                ctx.add(Expr::Add(end, offset))
            };

            let start_plus_offset = if den_offset == 0 {
                start
            } else {
                let offset = ctx.num(den_offset);
                ctx.add(Expr::Add(start, offset))
            };

            let result = ctx.add(Expr::Div(end_plus_offset, start_plus_offset));

            // Simplify the result
            let mut simplifier = crate::Simplifier::with_default_rules();
            simplifier.context = ctx.clone();
            let (simplified, _) = simplifier.simplify(result);
            *ctx = simplifier.context;

            return Some(simplified);
        }

        // For shift > 1, the pattern is more complex
        // We can still handle it but leave for future enhancement
    }

    None
}

/// Try to factor and evaluate products of factorizable expressions
/// Pattern: 1 - 1/k² = (k²-1)/k² = (k-1)(k+1)/k² = [(k-1)/k]·[(k+1)/k]
///
/// This handles:
/// - 1 - 1/k² (difference with reciprocal square)
/// - (k² - 1)/k² (already as fraction with factorable numerator)
///
/// Result for product from 2 to n:
/// - ∏(k-1)/k = 1/n
/// - ∏(k+1)/k = (n+1)/2
/// - Total: (n+1)/(2n)
fn try_factorizable_product(
    ctx: &mut Context,
    factor: ExprId,
    var: &str,
    start: ExprId,
    end: ExprId,
) -> Option<ExprId> {
    // Pattern 1: 1 - 1/k² or 1 - k^(-2)
    // This is the most common form
    if let Some((base_var, power)) = detect_one_minus_reciprocal_power(ctx, factor, var) {
        if power == 2 && base_var == var {
            // This is 1 - 1/k²
            // Factor as (k-1)(k+1)/k² = [(k-1)/k]·[(k+1)/k]

            // Evaluate ∏(k-1)/k from start to end
            // = (start-1)/start · start/(start+1) · ... · (end-1)/end
            // = (start-1) / end (telescopes to first numerator / last denominator)
            let start_minus_1 = if let Some(n) = get_integer(ctx, start) {
                ctx.num(n - 1)
            } else {
                let one = ctx.num(1);
                ctx.add(Expr::Sub(start, one))
            };

            // Evaluate ∏(k+1)/k from start to end
            // = (start+1)/start · (start+2)/(start+1) · ... · (end+1)/end
            // = (end+1) / start (telescopes to last numerator / first denominator)
            let end_plus_1 = if let Some(n) = get_integer(ctx, end) {
                ctx.num(n + 1)
            } else {
                let one = ctx.num(1);
                ctx.add(Expr::Add(end, one))
            };

            // Combine: (start-1)/end * (end+1)/start = (start-1)*(end+1) / (start*end)
            // Build as a single fraction for better simplification
            let combined_num = mul2_raw(ctx, start_minus_1, end_plus_1);
            let combined_den = mul2_raw(ctx, start, end);
            let result = ctx.add(Expr::Div(combined_num, combined_den));

            return Some(result);
        }
    }

    None
}

/// Detect pattern: 1 - 1/var^power or 1 - var^(-power)
/// Returns (variable name, power) if matched
fn detect_one_minus_reciprocal_power(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(String, i64)> {
    // Pattern: 1 - 1/k² or k^(-2) - 1 (canonicalized)
    // Also handles: 1 - k^(-2)

    if let Expr::Sub(left, right) = ctx.get(expr) {
        // Check if left is 1
        if let Expr::Number(n) = ctx.get(*left) {
            if n.is_one() {
                // Right should be 1/k² or k^(-2)
                return detect_reciprocal_power(ctx, *right, var);
            }
        }
    }

    // Also check Add with negative: 1 + (-1/k²) (canonical form: -1/k² + 1)
    if let Expr::Add(left, right) = ctx.get(expr) {
        // Check for -1/k² + 1 pattern
        if let Expr::Number(n) = ctx.get(*right) {
            if n.is_one() {
                if let Expr::Neg(inner) = ctx.get(*left) {
                    return detect_reciprocal_power(ctx, *inner, var);
                }
            }
        }
        // Check for 1 + (-1/k²) pattern
        if let Expr::Number(n) = ctx.get(*left) {
            if n.is_one() {
                if let Expr::Neg(inner) = ctx.get(*right) {
                    return detect_reciprocal_power(ctx, *inner, var);
                }
            }
        }
    }

    None
}

/// Detect 1/var^power or var^(-power)
fn detect_reciprocal_power(ctx: &Context, expr: ExprId, var: &str) -> Option<(String, i64)> {
    // Pattern 1: 1/k^n
    if let Expr::Div(num, den) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*num) {
            if n.is_one() {
                // den should be k^power
                if let Expr::Pow(base, exp) = ctx.get(*den) {
                    if let Expr::Variable(v) = ctx.get(*base) {
                        if v == var {
                            if let Some(power) = get_integer(ctx, *exp) {
                                return Some((v.clone(), power));
                            }
                        }
                    }
                }
                // Also check if den is just k (power = 1)
                if let Expr::Variable(v) = ctx.get(*den) {
                    if v == var {
                        return Some((v.clone(), 1));
                    }
                }
            }
        }
    }

    // Pattern 2: k^(-n)
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if let Expr::Variable(v) = ctx.get(*base) {
            if v == var {
                if let Expr::Neg(inner_exp) = ctx.get(*exp) {
                    if let Some(power) = get_integer(ctx, *inner_exp) {
                        return Some((v.clone(), power));
                    }
                }
                // Check for negative number exponent
                if let Expr::Number(n) = ctx.get(*exp) {
                    if *n < num_rational::BigRational::from_integer(0.into()) {
                        if let Some(power) = get_integer(ctx, *exp) {
                            return Some((v.clone(), -power));
                        }
                    }
                }
            }
        }
    }

    None
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(IntegrateRule));
    simplifier.add_rule(Box::new(DiffRule));
    simplifier.add_rule(Box::new(SumRule));
    simplifier.add_rule(Box::new(ProductRule));
}
