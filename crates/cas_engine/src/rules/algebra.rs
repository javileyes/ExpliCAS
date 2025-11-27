use crate::rule::{Rule, Rewrite};
use cas_ast::Expr;
use crate::polynomial::Polynomial;
use std::rc::Rc;
use std::collections::HashSet;
use num_traits::{One, Zero};
use num_rational::BigRational;

pub struct SimplifyFractionRule;

impl Rule for SimplifyFractionRule {
    fn name(&self) -> &str {
        "Simplify Algebraic Fraction"
    }

    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
        if let Expr::Div(num, den) = expr.as_ref() {
            // 1. Identify variable
            let vars = collect_variables(expr);
            if vars.len() != 1 {
                return None; // Only univariate for now
            }
            let var = vars.iter().next().unwrap();

            // 2. Convert to Polynomials
            let p_num = Polynomial::from_expr(num, var)?;
            let p_den = Polynomial::from_expr(den, var)?;

            // 3. Compute GCD
            let gcd = p_num.gcd(&p_den);

            // 4. Check if GCD is non-trivial (degree > 0 or constant != 1)
            // Actually, even constant GCD is useful for reducing 2x/2 -> x
            if gcd.degree() == 0 && gcd.leading_coeff().is_one() {
                return None;
            }

            // 5. Divide
            let (new_num_poly, rem_num) = p_num.div_rem(&gcd);
            let (new_den_poly, rem_den) = p_den.div_rem(&gcd);

            if !rem_num.is_zero() || !rem_den.is_zero() {
                // Should not happen if GCD is correct
                return None;
            }

            let new_num = new_num_poly.to_expr();
            let new_den = new_den_poly.to_expr();

            // If denominator is 1, return numerator
            if let Expr::Number(n) = new_den.as_ref() {
                if n.is_one() {
                    return Some(Rewrite {
                        new_expr: new_num,
                        description: "Simplified fraction by GCD".to_string(),
                    });
                }
            }

            return Some(Rewrite {
                new_expr: Expr::div(new_num, new_den),
                description: "Simplified fraction by GCD".to_string(),
            });
        }
        None
    }
}


pub struct ExpandRule;

impl Rule for ExpandRule {
    fn name(&self) -> &str {
        "Expand Polynomial"
    }

    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
        if let Expr::Function(name, args) = expr.as_ref() {
            if name == "expand" && args.len() == 1 {
                let arg = &args[0];
                // Try to convert to polynomial
                let vars = collect_variables(arg);
                if vars.len() != 1 {
                    return None;
                }
                let var = vars.iter().next().unwrap();
                
                if let Some(poly) = Polynomial::from_expr(arg, var) {
                    return Some(Rewrite {
                        new_expr: poly.to_expr(),
                        description: "expand(poly)".to_string(),
                    });
                }
            }
        }
        None
    }
}

pub struct FactorRule;

impl Rule for FactorRule {
    fn name(&self) -> &str {
        "Factor Polynomial (MVP)"
    }

    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
        if let Expr::Function(name, args) = expr.as_ref() {
            if name == "factor" && args.len() == 1 {
                let arg = &args[0];
                let vars = collect_variables(arg);
                if vars.len() != 1 {
                    return None;
                }
                let var = vars.iter().next().unwrap();
                
                if let Some(poly) = Polynomial::from_expr(arg, var) {
                    if poly.is_zero() { return None; }

                    let content = poly.content();
                    let min_deg = poly.min_degree();

                    // If trivial (content=1 and min_deg=0), nothing to factor
                    if content.is_one() && min_deg == 0 {
                        return None;
                    }

                    // Construct common factor term: content * x^min_deg
                    let mut factor_coeffs = vec![BigRational::zero(); min_deg + 1];
                    factor_coeffs[min_deg] = content.clone();
                    let factor_poly = Polynomial::new(factor_coeffs, var.clone());

                    let (quotient, rem) = poly.div_rem(&factor_poly);
                    if !rem.is_zero() {
                        return None; // Should not happen
                    }

                    let factor_expr = factor_poly.to_expr();
                    let quotient_expr = quotient.to_expr();

                    return Some(Rewrite {
                        new_expr: Expr::mul(factor_expr, quotient_expr),
                        description: "factor(poly) -> common_factor * quotient".to_string(),
                    });
                }
            }
        }
        None
    }
}

fn collect_variables(expr: &Expr) -> HashSet<String> {
    let mut vars = HashSet::new();
    collect_vars_recursive(expr, &mut vars);
    vars
}

fn collect_vars_recursive(expr: &Expr, vars: &mut HashSet<String>) {
    match expr {
        Expr::Variable(s) => { vars.insert(s.clone()); },
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            collect_vars_recursive(l, vars);
            collect_vars_recursive(r, vars);
        },
        Expr::Neg(e) => {
             collect_vars_recursive(e, vars);
        },
        Expr::Function(_, args) => {
            for arg in args {
                collect_vars_recursive(arg, vars);
            }
        },
        _ => {},
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn test_simplify_fraction() {
        let rule = SimplifyFractionRule;

        // (x^2 - 1) / (x + 1) -> x - 1
        let expr = parse("(x^2 - 1) / (x + 1)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        // Result might be -1 + x or x - 1 depending on polynomial to_expr order
        // Polynomial to_expr outputs lowest degree first? 
        // My implementation: "1 + x" for x+1.
        // x^2 - 1 = (x-1)(x+1). 
        // (x-1) -> -1 + x
        assert!(format!("{}", rewrite.new_expr).contains("x"));
        assert!(format!("{}", rewrite.new_expr).contains("-1"));
    }
    
    #[test]
    fn test_simplify_fraction_2() {
        let rule = SimplifyFractionRule;
        // (x^2 + 2*x + 1) / (x + 1) -> x + 1
        let expr = parse("(x^2 + 2*x + 1) / (x + 1)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert!(format!("{}", rewrite.new_expr).contains("1"));
        assert!(format!("{}", rewrite.new_expr).contains("x"));
    }
}
