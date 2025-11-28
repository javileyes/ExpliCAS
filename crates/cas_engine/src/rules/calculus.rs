use crate::rule::Rewrite;
use crate::define_rule;
use cas_ast::{Expr, ExprId, Context};
use num_traits::{One};
use num_rational::BigRational;

define_rule!(
    IntegrateRule,
    "Symbolic Integration",
    |ctx, expr| {
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
                                description: format!("integrate({:?})", integrand), // Debug format for now
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
                            description: format!("integrate({:?})", integrand),
                            });
                    }
                }
            }
        }
        None
    }
);

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
                
                let is_a_one = if let Expr::Number(n) = ctx.get(a) { n.is_one() } else { false };
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
                let is_a_one = if let Expr::Number(n) = ctx.get(a) { n.is_one() } else { false };
                
                // Check if base is e
                let is_e = if let Expr::Constant(c) = ctx.get(base) {
                    c == &cas_ast::Constant::E
                } else {
                    false
                };

                if is_e {
                    if is_a_one { return Some(expr); }
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
                let is_a_one = if let Expr::Number(n) = ctx.get(a) { n.is_one() } else { false };
                
                match name.as_str() {
                    "sin" => {
                        // integrate(sin(ax+b)) = -cos(ax+b)/a
                        let cos_arg = ctx.add(Expr::Function("cos".to_string(), vec![arg]));
                        let integral = ctx.add(Expr::Neg(cos_arg));
                        if is_a_one { return Some(integral); }
                        return Some(ctx.add(Expr::Div(integral, a)));
                    },
                    "cos" => {
                        // integrate(cos(ax+b)) = sin(ax+b)/a
                        let integral = ctx.add(Expr::Function("sin".to_string(), vec![arg]));
                        if is_a_one { return Some(integral); }
                        return Some(ctx.add(Expr::Div(integral, a)));
                    },
                    "exp" => {
                        // integrate(exp(ax+b)) = exp(ax+b)/a
                        let integral = expr;
                        if is_a_one { return Some(integral); }
                        return Some(ctx.add(Expr::Div(integral, a)));
                    },
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
        },
        Expr::Neg(e) => contains_var(ctx, *e, var),
        Expr::Function(_, args) => args.iter().any(|a| contains_var(ctx, *a, var)),
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
        },
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
        },

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
        },
        _ => None
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
    use cas_parser::parse;
    use cas_ast::{DisplayExpr, Context};

    #[test]
    fn test_integrate_power() {
        let mut ctx = Context::new();
        let rule = IntegrateRule;
        // integrate(x^2, x) -> x^3/3
        let expr = parse("integrate(x^2, x)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "x^(2 + 1) / (2 + 1)");
    }

    #[test]
    fn test_integrate_constant() {
        let mut ctx = Context::new();
        let rule = IntegrateRule;
        // integrate(5, x) -> 5x
        let expr = parse("integrate(5, x)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "5 * x");
    }

    #[test]
    fn test_integrate_trig() {
        let mut ctx = Context::new();
        let rule = IntegrateRule;
        // integrate(sin(x), x) -> -cos(x)
        let expr = parse("integrate(sin(x), x)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "-cos(x)");
    }

    #[test]
    fn test_integrate_linearity() {
        let mut ctx = Context::new();
        let rule = IntegrateRule;
        // integrate(x + 1, x) -> x^2/2 + x
        let expr = parse("integrate(x + 1, x)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        let res = format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr });
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
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "-cos(2 * x) / 2");
    }

    #[test]
    fn test_integrate_linear_subst_exp() {
        let mut ctx = Context::new();
        let rule = IntegrateRule;
        // integrate(exp(3*x), x) -> exp(3*x)/3
        let expr = parse("integrate(exp(3*x), x)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "e^(3 * x) / 3");
    }

    #[test]
    fn test_integrate_linear_subst_power() {
        let mut ctx = Context::new();
        let rule = IntegrateRule;
        // integrate((2*x + 1)^2, x) -> (2*x + 1)^3 / (2*3) -> (2*x+1)^3 / 6
        let expr = parse("integrate((2*x + 1)^2, x)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        // Note: 2*3 is not simplified by IntegrateRule, it produces Expr::mul(2, 3).
        // Simplification happens later in the pipeline.
        // So we expect (2*x + 1)^(2+1) / (2 * (2+1))
        // Actually get_linear_coeffs returns a=2+0 for 2x+1.
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "(2 * x + 1)^(2 + 1) / ((2 + 0) * (2 + 1))");
    }

    #[test]
    fn test_integrate_linear_subst_log() {
        let mut ctx = Context::new();
        let rule = IntegrateRule;
        // integrate(1/(3*x), x) -> ln(3*x)/3
        let expr = parse("integrate(1/(3*x), x)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "ln(3 * x) / 3");
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(IntegrateRule));
}
