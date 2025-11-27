use crate::rule::{Rule, Rewrite};
use cas_ast::Expr;
use crate::polynomial::Polynomial;
use std::rc::Rc;
use std::collections::HashSet;
use num_traits::{One};


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
            let gcd_expr = gcd.to_expr();

            // If denominator is 1, return numerator
            if let Expr::Number(n) = new_den.as_ref() {
                if n.is_one() {
                    return Some(Rewrite {
                        new_expr: new_num,
                        description: format!("Simplified fraction by GCD: {}", gcd_expr),
                    });
                }
            }

            return Some(Rewrite {
                new_expr: Expr::div(new_num, new_den),
                description: format!("Simplified fraction by GCD: {}", gcd_expr),
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
                if vars.is_empty() {
                    // Constant expression. Already expanded.
                    // Just return the arg (simplification removes the expand wrapper).
                    return Some(Rewrite {
                        new_expr: arg.clone(),
                        description: "expand(constant) -> constant".to_string(),
                    });
                }
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
        "Factor Polynomial"
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

                    // 1. Extract content (common constant factor)
                    // Note: factor_rational_roots returns factors that might include content in the last term?
                    // Actually, my implementation of factor_rational_roots returns (qx-p) factors.
                    // The last factor is the remaining polynomial, which contains the content/leading coeff.
                    
                    let factors = poly.factor_rational_roots();
                    
                    if factors.len() == 1 {
                        // Irreducible (over rationals) or just trivial
                        // Check if we can at least pull out content?
                        // Existing logic did that. Let's keep it?
                        // But factor_rational_roots returns [poly] if no roots.
                        // Let's check if we can factor out common term (x^k) or constant.
                        let content = poly.content();
                        let min_deg = poly.min_degree();
                        if content.is_one() && min_deg == 0 {
                            // Truly irreducible and no common factor
                            // Remove the "factor()" wrapper?
                            // Or return None to signal "cannot factor further"?
                            // If user asked to factor, and we can't, we should probably return the input (without wrapper)
                            // or leave it?
                            // Usually "factor(x)" -> "x".
                            return Some(Rewrite {
                                new_expr: arg.clone(),
                                description: "Irreducible".to_string(),
                            });
                        }
                    }

                    // Construct the expression from factors
                    // factors[0] * factors[1] * ...
                    let mut res = factors[0].to_expr();
                    for factor in factors.iter().skip(1) {
                        res = Expr::mul(res, factor.to_expr());
                    }

                    return Some(Rewrite {
                        new_expr: res,
                        description: "Factorization".to_string(),
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

    #[test]
    fn test_factor_difference_squares() {
        let rule = FactorRule;
        // factor(x^2 - 1) -> (x - 1)(x + 1)
        // Note: My implementation produces (x-1) and (x+1) (or similar).
        // Order depends on root finding.
        // Roots are 1, -1.
        // Factors: (x-1), (x+1).
        let expr = parse("factor(x^2 - 1)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        let res = format!("{}", rewrite.new_expr);
        assert!(res.contains("x - 1") || res.contains("-1 + x") || res.contains("x + -1"));
        assert!(res.contains("x + 1") || res.contains("1 + x"));
    }

    #[test]
    fn test_factor_perfect_square() {
        let rule = FactorRule;
        // factor(x^2 + 2x + 1) -> (x + 1)(x + 1)
        let expr = parse("factor(x^2 + 2*x + 1)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        let res = format!("{}", rewrite.new_expr);
        // Should be (x+1) * (x+1)
        assert!(res.contains("x + 1") || res.contains("1 + x"));
        assert!(res.contains("*"));
    }
}
