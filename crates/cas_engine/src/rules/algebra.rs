use crate::rule::{Rule, Rewrite};
use cas_ast::Expr;
use crate::polynomial::Polynomial;
use std::rc::Rc;
use std::collections::HashSet;
use num_traits::{One, Signed, ToPrimitive};


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


pub struct NestedFractionRule;

impl Rule for NestedFractionRule {
    fn name(&self) -> &str {
        "Simplify Nested Fraction"
    }

    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
        if let Expr::Div(num, den) = expr.as_ref() {
            let num_denoms = collect_denominators(num);
            let den_denoms = collect_denominators(den);
            
            if num_denoms.is_empty() && den_denoms.is_empty() {
                return None;
            }
            
            // Collect all unique denominators
            let mut all_denoms = Vec::new();
            all_denoms.extend(num_denoms);
            all_denoms.extend(den_denoms);
            
            if all_denoms.is_empty() {
                return None;
            }
            
            // Construct the common multiplier (product of all unique denominators)
            // Ideally LCM, but product is safer for now.
            // We need to deduplicate.
            let mut unique_denoms: Vec<Rc<Expr>> = Vec::new();
            for d in all_denoms {
                if !unique_denoms.contains(&d) {
                    unique_denoms.push(d);
                }
            }
            
            if unique_denoms.is_empty() {
                return None;
            }

            let mut multiplier = unique_denoms[0].clone();
            for i in 1..unique_denoms.len() {
                multiplier = Expr::mul(multiplier, unique_denoms[i].clone());
            }
            
            // Multiply num and den by multiplier
            let new_num = distribute(num, &multiplier);
            let new_den = distribute(den, &multiplier);
            
            return Some(Rewrite {
                new_expr: Expr::div(new_num, new_den),
                description: format!("Multiply by common denominator {}", multiplier),
            });
        }
        None
    }
}

fn distribute(target: &Rc<Expr>, multiplier: &Rc<Expr>) -> Rc<Expr> {
    match target.as_ref() {
        Expr::Add(l, r) => Expr::add(distribute(l, multiplier), distribute(r, multiplier)),
        Expr::Sub(l, r) => Expr::sub(distribute(l, multiplier), distribute(r, multiplier)),
        Expr::Div(l, r) => {
            // (l / r) * m. If m == r, return l.
            if r == multiplier {
                return l.clone();
            }
            // If m contains r (e.g. m = x*y, r = x), we can simplify.
            // For now, just return (l*m)/r
            Expr::mul(Expr::div(l.clone(), r.clone()), multiplier.clone())
        },
        _ => Expr::mul(target.clone(), multiplier.clone())
    }
}

fn collect_denominators(expr: &Rc<Expr>) -> Vec<Rc<Expr>> {
    let mut denoms = Vec::new();
    match expr.as_ref() {
        Expr::Div(_, den) => {
            denoms.push(den.clone());
            // Recurse? Maybe not needed for simple cases.
        },
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            denoms.extend(collect_denominators(l));
            denoms.extend(collect_denominators(r));
        },
        Expr::Pow(b, e) => {
            // Check for negative exponent?
            if let Expr::Number(n) = e.as_ref() {
                if n.is_negative() {
                    // b^-k = 1/b^k. Denominator is b^k (or b if k=-1)
                    // For simplicity, let's just handle 1/x style Divs first.
                }
            }
            denoms.extend(collect_denominators(b));
        },
        _ => {}
    }
    denoms
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


pub struct FactorDifferenceSquaresRule;

impl Rule for FactorDifferenceSquaresRule {
    fn name(&self) -> &str {
        "Factor Difference of Squares"
    }

    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
        let (l, r) = match expr.as_ref() {
            Expr::Sub(l, r) => (l.clone(), r.clone()),
            Expr::Add(a, b) => {
                // Check if one is negative
                if is_negative_term(b) {
                    (a.clone(), negate_term(b))
                } else if is_negative_term(a) {
                    (b.clone(), negate_term(a))
                } else {
                    return None;
                }
            },
            _ => return None,
        };

        if let (Some(root_l), Some(root_r)) = (get_square_root(&l), get_square_root(&r)) {
            // a^2 - b^2 = (a - b)(a + b)
            let term1 = Expr::sub(root_l.clone(), root_r.clone());
            
            // Check for Pythagorean identity in term2 (a + b)
            // sin^2 + cos^2 = 1
            let mut term2 = Expr::add(root_l.clone(), root_r.clone());
            
            if is_sin_cos_pair(&root_l, &root_r) {
                 term2 = Rc::new(Expr::Number(num_rational::BigRational::one()));
            }

            let new_expr = Expr::mul(term1, term2);
            
            return Some(Rewrite {
                new_expr,
                description: "Factor difference of squares".to_string(),
            });
        }
        None
    }
}

fn is_sin_cos_pair(a: &Rc<Expr>, b: &Rc<Expr>) -> bool {
    (is_trig_pow(a, "sin", 2) && is_trig_pow(b, "cos", 2) && get_trig_arg(a) == get_trig_arg(b)) ||
    (is_trig_pow(a, "cos", 2) && is_trig_pow(b, "sin", 2) && get_trig_arg(a) == get_trig_arg(b))
}

fn is_trig_pow(expr: &Rc<Expr>, name: &str, power: i64) -> bool {
    if let Expr::Pow(base, exp) = expr.as_ref() {
        if let Expr::Number(n) = exp.as_ref() {
            if n.is_integer() && n.to_integer() == power.into() {
                if let Expr::Function(func_name, args) = base.as_ref() {
                    return func_name == name && args.len() == 1;
                }
            }
        }
    }
    false
}

fn get_trig_arg(expr: &Rc<Expr>) -> Option<Rc<Expr>> {
    if let Expr::Pow(base, _) = expr.as_ref() {
        if let Expr::Function(_, args) = base.as_ref() {
            if args.len() == 1 {
                return Some(args[0].clone());
            }
        }
    }
    None
}

fn is_negative_term(expr: &Rc<Expr>) -> bool {
    match expr.as_ref() {
        Expr::Neg(_) => true,
        Expr::Mul(l, _) => {
            if let Expr::Number(n) = l.as_ref() {
                n.is_negative()
            } else {
                false
            }
        },
        Expr::Number(n) => n.is_negative(),
        _ => false
    }
}

fn negate_term(expr: &Rc<Expr>) -> Rc<Expr> {
    match expr.as_ref() {
        Expr::Neg(inner) => inner.clone(),
        Expr::Mul(l, r) => {
            if let Expr::Number(n) = l.as_ref() {
                if n.is_negative() {
                    return Expr::mul(Expr::num((-n).to_i64().unwrap()), r.clone());
                }
            }
            Expr::neg(expr.clone())
        },
        Expr::Number(n) => Expr::num((-n).to_i64().unwrap()),
        _ => Expr::neg(expr.clone())
    }
}

fn get_square_root(expr: &Rc<Expr>) -> Option<Rc<Expr>> {
    match expr.as_ref() {
        Expr::Pow(b, e) => {
            if let Expr::Number(n) = e.as_ref() {
                if n.is_integer() {
                    let val = n.to_integer();
                    if &val % 2 == 0.into() {
                        let two = num_bigint::BigInt::from(2);
                        let new_exp = Expr::num((val / two).to_i64().unwrap());
                        // If new_exp is 1, simplify to b
                        if let Expr::Number(ne) = new_exp.as_ref() {
                            if ne.is_one() {
                                return Some(b.clone());
                            }
                        }
                        return Some(Expr::pow(b.clone(), new_exp));
                    }
                }
            }
            None
        },
        // Handle sin(x)^4 -> sin(x)^2
        // Handle 4 -> 2
        Expr::Number(n) => {
             // Check if n is a perfect square
             // For simplicity, only handle positive integers for now
             if n.is_integer() && !n.is_negative() {
                 let val = n.to_integer();
                 // Simple integer sqrt check
                 let sqrt = val.sqrt();
                 if &sqrt * &sqrt == val {
                     return Some(Expr::num(sqrt.to_i64().unwrap()));
                 }
             }
             None
        },
        _ => None
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
