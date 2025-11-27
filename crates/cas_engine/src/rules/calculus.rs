use crate::rule::{Rule, Rewrite};
use cas_ast::Expr;
use std::rc::Rc;
use num_traits::{One, Zero};
use num_rational::BigRational;

pub struct IntegrateRule;

impl Rule for IntegrateRule {
    fn name(&self) -> &str {
        "Symbolic Integration"
    }

    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
        if let Expr::Function(name, args) = expr.as_ref() {
            if name == "integrate" {
                if args.len() == 2 {
                    let integrand = &args[0];
                    let var_expr = &args[1];
                    if let Expr::Variable(var_name) = var_expr.as_ref() {
                        if let Some(result) = integrate(integrand, var_name) {
                            return Some(Rewrite {
                                new_expr: result,
                                description: format!("integrate({})", integrand),
                            });
                        }
                    }
                } else if args.len() == 1 {
                    // Default to 'x' if not specified? Or fail?
                    // Let's assume 'x' for convenience if only 1 arg.
                    let integrand = &args[0];
                    if let Some(result) = integrate(integrand, "x") {
                        return Some(Rewrite {
                            new_expr: result,
                            description: format!("integrate({})", integrand),
                        });
                    }
                }
            }
        }
        None
    }
}

fn integrate(expr: &Rc<Expr>, var: &str) -> Option<Rc<Expr>> {
    // 1. Linearity: integrate(a + b) = integrate(a) + integrate(b)
    if let Expr::Add(l, r) = expr.as_ref() {
        let int_l = integrate(l, var)?;
        let int_r = integrate(r, var)?;
        return Some(Expr::add(int_l, int_r));
    }
    
    // 2. Linearity: integrate(a - b) = integrate(a) - integrate(b)
    if let Expr::Sub(l, r) = expr.as_ref() {
        let int_l = integrate(l, var)?;
        let int_r = integrate(r, var)?;
        return Some(Expr::sub(int_l, int_r));
    }

    // 3. Constant Multiple: integrate(c * f(x)) = c * integrate(f(x))
    // We need to check if one operand is constant (w.r.t var).
    if let Expr::Mul(l, r) = expr.as_ref() {
        if !contains_var(l, var) {
            if let Some(int_r) = integrate(r, var) {
                return Some(Expr::mul(l.clone(), int_r));
            }
        }
        if !contains_var(r, var) {
            if let Some(int_l) = integrate(l, var) {
                return Some(Expr::mul(r.clone(), int_l));
            }
        }
    }

    // 4. Basic Rules & Linear Substitution
    
    // Constant: integrate(c) = c*x
    if !contains_var(expr, var) {
        return Some(Expr::mul(expr.clone(), Expr::var(var)));
    }

    // Power Rule: integrate(u^n) * u' -> u^(n+1)/(n+1)
    // Here we handle u = ax+b, so u' = a.
    // integrate((ax+b)^n) = (ax+b)^(n+1) / (a*(n+1))
    if let Expr::Pow(base, exp) = expr.as_ref() {
        // Case 1: u^n where u is linear (ax+b)
        if let Some((a, _)) = get_linear_coeffs(base, var) {
            if !contains_var(exp, var) {
                // Check for n = -1 case (1/u)
                if let Expr::Number(n) = exp.as_ref() {
                     if *n == BigRational::from_integer((-1).into()) {
                         // ln(u) / a
                         return Some(Expr::div(Expr::ln(base.clone()), a));
                     }
                }
                
                let new_exp = Expr::add(exp.clone(), Expr::num(1));
                
                let is_a_one = if let Expr::Number(n) = a.as_ref() { n.is_one() } else { false };
                let new_denom = if is_a_one {
                    new_exp.clone()
                } else {
                    Expr::mul(a, new_exp.clone())
                };
                
                return Some(Expr::div(Expr::pow(base.clone(), new_exp), new_denom));
            }
        }
        
        // Case 2: c^u where u is linear (ax+b)
        // integrate(c^(ax+b)) = c^(ax+b) / (a * ln(c))
        // If c = e, ln(c) = 1, so e^(ax+b) / a
        if !contains_var(base, var) {
            if let Some((a, _)) = get_linear_coeffs(exp, var) {
                let is_a_one = if let Expr::Number(n) = a.as_ref() { n.is_one() } else { false };
                
                // Check if base is e
                if let Expr::Constant(c) = base.as_ref() {
                    if c == &cas_ast::Constant::E {
                        if is_a_one { return Some(expr.clone()); }
                        return Some(Expr::div(expr.clone(), a));
                    }
                }
                
                // General base c
                let ln_c = Expr::ln(base.clone());
                let denom = if is_a_one {
                    ln_c
                } else {
                    Expr::mul(a, ln_c)
                };
                return Some(Expr::div(expr.clone(), denom));
            }
        }
    }

    // Variable itself: integrate(x) = x^2/2
    // This is covered by Power Rule if x is treated as (1*x + 0)^1, but explicit check is faster/simpler?
    // Actually get_linear_coeffs(x) returns (1, 0).
    // So Pow(x, 1) would be handled above IF expr was parsed as Pow.
    // But "x" is Expr::Variable.
    if let Expr::Variable(v) = expr.as_ref() {
        if v == var {
            return Some(Expr::div(Expr::pow(Expr::var(var), Expr::num(2)), Expr::num(2)));
        }
    }
    
    // 1/u case (if represented as Div(1, u))
    if let Expr::Div(num, den) = expr.as_ref() {
        if let Expr::Number(n) = num.as_ref() {
            if n.is_one() {
                 if let Some((a, _)) = get_linear_coeffs(den, var) {
                     // integrate(1/(ax+b)) = ln(ax+b)/a
                     return Some(Expr::div(Expr::ln(den.clone()), a));
                 }
            }
        }
    }

    // Trig Rules & Exponential with Linear Substitution
    if let Expr::Function(name, args) = expr.as_ref() {
        if args.len() == 1 {
            let arg = &args[0];
            if let Some((a, _)) = get_linear_coeffs(arg, var) {
                let is_a_one = if let Expr::Number(n) = a.as_ref() { n.is_one() } else { false };
                
                match name.as_str() {
                    "sin" => {
                        // integrate(sin(ax+b)) = -cos(ax+b)/a
                        let integral = Expr::neg(Expr::cos(arg.clone()));
                        if is_a_one { return Some(integral); }
                        return Some(Expr::div(integral, a));
                    },
                    "cos" => {
                        // integrate(cos(ax+b)) = sin(ax+b)/a
                        let integral = Expr::sin(arg.clone());
                        if is_a_one { return Some(integral); }
                        return Some(Expr::div(integral, a));
                    },
                    "exp" => {
                        // integrate(exp(ax+b)) = exp(ax+b)/a
                        let integral = expr.clone();
                        if is_a_one { return Some(integral); }
                        return Some(Expr::div(integral, a));
                    },
                    _ => {}
                }
            }
        }
    }

    None
}

fn contains_var(expr: &Expr, var: &str) -> bool {
    match expr {
        Expr::Variable(v) => v == var,
        Expr::Number(_) | Expr::Constant(_) => false,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            contains_var(l, var) || contains_var(r, var)
        },
        Expr::Neg(e) => contains_var(e, var),
        Expr::Function(_, args) => args.iter().any(|a| contains_var(a, var)),
    }
}

// Returns (a, b) such that expr = a*var + b
fn get_linear_coeffs(expr: &Rc<Expr>, var: &str) -> Option<(Rc<Expr>, Rc<Expr>)> {
    if !contains_var(expr, var) {
        return Some((Expr::num(0), expr.clone()));
    }

    match expr.as_ref() {
        Expr::Variable(v) if v == var => Some((Expr::num(1), Expr::num(0))),
        Expr::Mul(l, r) => {
            // c * x
            if !contains_var(l, var) && is_var(r, var) {
                return Some((l.clone(), Expr::num(0)));
            }
            // x * c
            if is_var(l, var) && !contains_var(r, var) {
                return Some((r.clone(), Expr::num(0)));
            }
            None
        },
        Expr::Add(l, r) => {
            // (ax) + b or b + (ax)
            let l_coeffs = get_linear_coeffs(l, var);
            let r_coeffs = get_linear_coeffs(r, var);
            
            if let (Some((a1, b1)), Some((a2, b2))) = (l_coeffs, r_coeffs) {
                // Check if a1 and a2 are constants (they should be if get_linear_coeffs returned Some)
                // Actually get_linear_coeffs returns Some only if it's linear.
                // Sum of linear is linear.
                // a = a1 + a2, b = b1 + b2
                // But we need to ensure a is constant.
                // get_linear_coeffs ensures 'a' is the coefficient of var.
                // If expr is linear, 'a' must not contain var.
                if !contains_var(&a1, var) && !contains_var(&a2, var) {
                     return Some((Expr::add(a1, a2), Expr::add(b1, b2)));
                }
            }
            None
        },

        Expr::Sub(l, r) => {
             let l_coeffs = get_linear_coeffs(l, var);
             let r_coeffs = get_linear_coeffs(r, var);
             if let (Some((a1, b1)), Some((a2, b2))) = (l_coeffs, r_coeffs) {
                 if !contains_var(&a1, var) && !contains_var(&a2, var) {
                     return Some((Expr::sub(a1, a2), Expr::sub(b1, b2)));
                }
             }
             None
        },
        _ => None
    }
}

fn is_var(expr: &Expr, var: &str) -> bool {
    if let Expr::Variable(v) = expr {
        v == var
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn test_integrate_power() {
        let rule = IntegrateRule;
        // integrate(x^2, x) -> x^3/3
        let expr = parse("integrate(x^2, x)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "x^(2 + 1) / (2 + 1)");
    }

    #[test]
    fn test_integrate_constant() {
        let rule = IntegrateRule;
        // integrate(5, x) -> 5x
        let expr = parse("integrate(5, x)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "5 * x");
    }

    #[test]
    fn test_integrate_trig() {
        let rule = IntegrateRule;
        // integrate(sin(x), x) -> -cos(x)
        let expr = parse("integrate(sin(x), x)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "-cos(x)");
    }

    #[test]
    fn test_integrate_linearity() {
        let rule = IntegrateRule;
        // integrate(x + 1, x) -> x^2/2 + x
        let expr = parse("integrate(x + 1, x)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        let res = format!("{}", rewrite.new_expr);
        // x^2/2 + 1*x
        assert!(res.contains("x^2 / 2"));
        assert!(res.contains("1 * x") || res.contains("x")); 
    }
    #[test]
    fn test_integrate_linear_subst_trig() {
        let rule = IntegrateRule;
        // integrate(sin(2*x), x) -> -cos(2*x)/2
        let expr = parse("integrate(sin(2*x), x)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "-cos(2 * x) / 2");
    }

    #[test]
    fn test_integrate_linear_subst_exp() {
        let rule = IntegrateRule;
        // integrate(exp(3*x), x) -> exp(3*x)/3
        let expr = parse("integrate(exp(3*x), x)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "e^(3 * x) / 3");
    }

    #[test]
    fn test_integrate_linear_subst_power() {
        let rule = IntegrateRule;
        // integrate((2*x + 1)^2, x) -> (2*x + 1)^3 / (2*3) -> (2*x+1)^3 / 6
        let expr = parse("integrate((2*x + 1)^2, x)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        // Note: 2*3 is not simplified by IntegrateRule, it produces Expr::mul(2, 3).
        // Simplification happens later in the pipeline.
        // So we expect (2*x + 1)^(2+1) / (2 * (2+1))
        // Actually get_linear_coeffs returns a=2+0 for 2x+1.
        assert_eq!(format!("{}", rewrite.new_expr), "(2 * x + 1)^(2 + 1) / ((2 + 0) * (2 + 1))");
    }

    #[test]
    fn test_integrate_linear_subst_log() {
        let rule = IntegrateRule;
        // integrate(1/(3*x), x) -> ln(3*x)/3
        let expr = parse("integrate(1/(3*x), x)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "ln(3 * x) / 3");
    }
}
