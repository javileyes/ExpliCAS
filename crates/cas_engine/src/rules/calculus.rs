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

    // 4. Basic Rules
    
    // Constant: integrate(c) = c*x
    if !contains_var(expr, var) {
        return Some(Expr::mul(expr.clone(), Expr::var(var)));
    }

    // Power Rule: integrate(x^n) = x^(n+1)/(n+1)
    if let Expr::Pow(base, exp) = expr.as_ref() {
        if let Expr::Variable(v) = base.as_ref() {
            if v == var {
                if !contains_var(exp, var) {
                    // Check for n = -1 case (1/x)
                    if let Expr::Number(n) = exp.as_ref() {
                         if *n == BigRational::from_integer((-1).into()) {
                             return Some(Expr::ln(base.clone()));
                         }
                    }
                    // Also check for -1 in general constant expression? Harder.
                    // Assuming number for now.
                    
                    let new_exp = Expr::add(exp.clone(), Expr::num(1));
                    let new_denom = new_exp.clone(); // n+1
                    return Some(Expr::div(Expr::pow(base.clone(), new_exp), new_denom));
                }
            }
        }
    }

    // Variable itself: integrate(x) = x^2/2
    if let Expr::Variable(v) = expr.as_ref() {
        if v == var {
            return Some(Expr::div(Expr::pow(Expr::var(var), Expr::num(2)), Expr::num(2)));
        }
    }
    
    // 1/x case (if represented as Div(1, x))
    if let Expr::Div(num, den) = expr.as_ref() {
        if let Expr::Number(n) = num.as_ref() {
            if n.is_one() {
                 if let Expr::Variable(v) = den.as_ref() {
                     if v == var {
                         return Some(Expr::ln(den.clone()));
                     }
                 }
            }
        }
    }

    // Trig Rules
    if let Expr::Function(name, args) = expr.as_ref() {
        if args.len() == 1 {
            let arg = &args[0];
            // Only handle simple sin(x), cos(x), e^x for now.
            // Chain rule (u-substitution) is harder.
            if let Expr::Variable(v) = arg.as_ref() {
                if v == var {
                    match name.as_str() {
                        "sin" => return Some(Expr::neg(Expr::cos(arg.clone()))), // -cos(x)
                        "cos" => return Some(Expr::sin(arg.clone())), // sin(x)
                        "exp" => return Some(expr.clone()), // e^x
                        _ => {}
                    }
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
}
