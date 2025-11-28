use crate::rule::Rewrite;
use crate::define_rule;
use cas_ast::Expr;
use std::rc::Rc;
use std::collections::HashMap;
use num_traits::{One, Zero};

define_rule!(
    CollectRule,
    "Collect Terms",
    |expr| {
        if let Expr::Function(name, args) = expr.as_ref() {
            if name == "collect" && args.len() == 2 {
                let target_expr = &args[0];
                let var_expr = &args[1];

                // Ensure second argument is a variable
                let var_name = if let Expr::Variable(v) = var_expr.as_ref() {
                    v.clone()
                } else {
                    return None;
                };

                // 1. Flatten terms
                let terms = flatten_add_chain(target_expr);

                // 2. Group by degree of var
                // Map: degree -> Vec<CoefficientExpr>
                let mut groups: HashMap<i64, Vec<Rc<Expr>>> = HashMap::new();

                for term in terms {
                    let (coeff, degree) = extract_coeff_degree(&term, &var_name);
                    if degree == 0 {
                        // Check if it really is independent or just degree 0 (constant wrt var)
                        // extract_coeff_degree returns degree 0 for terms not containing var.
                        // We treat them as independent terms to be appended at the end, 
                        // or we can group them under x^0.
                        // Let's group them under x^0 for now, effectively "collecting constants".
                        groups.entry(0).or_default().push(coeff);
                    } else {
                        groups.entry(degree).or_default().push(coeff);
                    }
                }

                // 3. Reconstruct expression
                let mut new_terms = Vec::new();

                // Sort degrees descending
                let mut degrees: Vec<i64> = groups.keys().cloned().collect();
                degrees.sort_by(|a, b| b.cmp(a));

                for deg in degrees {
                    let coeffs = groups.get(&deg).unwrap();
                    if coeffs.is_empty() { continue; }

                    // Sum coefficients: (c1 + c2 + ...)
                    let combined_coeff = if coeffs.len() == 1 {
                        coeffs[0].clone()
                    } else {
                        let mut sum = coeffs[0].clone();
                        for c in coeffs.iter().skip(1) {
                            sum = Expr::add(sum, c.clone());
                        }
                        sum
                    };

                    // Construct term: coeff * var^deg
                    let term = if deg == 0 {
                        combined_coeff
                    } else {
                        let var_part = if deg == 1 {
                            Expr::var(&var_name)
                        } else {
                            Expr::pow(Expr::var(&var_name), Expr::num(deg))
                        };

                        if is_one(&combined_coeff) {
                            var_part
                        } else if is_zero(&combined_coeff) {
                            // 0 * x^n = 0, skip
                            continue;
                        } else {
                            Expr::mul(combined_coeff, var_part)
                        }
                    };
                    new_terms.push(term);
                }

                if new_terms.is_empty() {
                    return Some(Rewrite {
                        new_expr: Expr::num(0),
                        description: format!("collect({}, {})", target_expr, var_name),
                    });
                }

                let mut result = new_terms[0].clone();
                for t in new_terms.into_iter().skip(1) {
                    result = Expr::add(result, t);
                }

                return Some(Rewrite {
                    new_expr: result,
                    description: format!("collect({}, {})", target_expr, var_name),
                });
            }
        }
        None
    }
);

// Helper to check if expr is effectively 1
fn is_one(expr: &Expr) -> bool {
    if let Expr::Number(n) = expr {
        n.is_one()
    } else {
        false
    }
}

fn is_zero(expr: &Expr) -> bool {
    if let Expr::Number(n) = expr {
        n.is_zero()
    } else {
        false
    }
}

fn flatten_add_chain(expr: &Rc<Expr>) -> Vec<Rc<Expr>> {
    let mut terms = Vec::new();
    flatten_recursive(expr, &mut terms, false);
    terms
}

fn flatten_recursive(expr: &Rc<Expr>, terms: &mut Vec<Rc<Expr>>, negate: bool) {
    match expr.as_ref() {
        Expr::Add(l, r) => {
            flatten_recursive(l, terms, negate);
            flatten_recursive(r, terms, negate);
        },
        Expr::Sub(l, r) => {
            flatten_recursive(l, terms, negate);
            flatten_recursive(r, terms, !negate); // Sub is Add(l, Neg(r))
        },
        _ => {
            if negate {
                terms.push(Expr::neg(expr.clone()));
            } else {
                terms.push(expr.clone());
            }
        }
    }
}

// Returns (coefficient, degree) for a term with respect to var
fn extract_coeff_degree(term: &Rc<Expr>, var: &str) -> (Rc<Expr>, i64) {
    // Cases:
    // x -> (1, 1)
    // x^n -> (1, n)
    // a * x -> (a, 1)
    // a * x^n -> (a, n)
    // a -> (a, 0)
    // x * y -> (y, 1) if we treat y as coeff? Yes.
    
    // We need to traverse Mul chain to find var powers.
    // Assume term is a product of factors.
    let factors = flatten_mul_chain(term);
    
    let mut degree = 0;
    let mut coeff_factors = Vec::new();

    for factor in factors {
        match factor.as_ref() {
            Expr::Variable(v) if v == var => {
                degree += 1;
            },
            Expr::Pow(base, exp) => {
                if let Expr::Variable(v) = base.as_ref() {
                    if v == var {
                        if let Expr::Number(n) = exp.as_ref() {
                            if n.is_integer() {
                                degree += n.to_integer().try_into().unwrap_or(0);
                                continue;
                            }
                        }
                    }
                }
                coeff_factors.push(factor.clone());
            },
            _ => {
                coeff_factors.push(factor.clone());
            }
        }
    }

    let coeff = if coeff_factors.is_empty() {
        Expr::num(1)
    } else {
        let mut c = coeff_factors[0].clone();
        for f in coeff_factors.into_iter().skip(1) {
            c = Expr::mul(c, f);
        }
        c
    };

    (coeff, degree)
}

fn flatten_mul_chain(expr: &Rc<Expr>) -> Vec<Rc<Expr>> {
    let mut factors = Vec::new();
    flatten_mul_recursive(expr, &mut factors);
    factors
}

fn flatten_mul_recursive(expr: &Rc<Expr>, factors: &mut Vec<Rc<Expr>>) {
    match expr.as_ref() {
        Expr::Mul(l, r) => {
            flatten_mul_recursive(l, factors);
            flatten_mul_recursive(r, factors);
        },
        Expr::Neg(e) => {
            // Treat Neg(e) as -1 * e
            factors.push(Expr::num(-1));
            flatten_mul_recursive(e, factors);
        }
        _ => {
            factors.push(expr.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_parser::parse;

    #[test]
    fn test_collect_basic() {
        let rule = CollectRule;
        // collect(a*x + b*x, x) -> (a+b)*x
        let expr = parse("collect(a*x + b*x, x)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        // Result could be (a+b)*x or (b+a)*x
        let s = format!("{}", rewrite.new_expr);
        assert!(s.contains("x"));
        assert!(s.contains("a + b") || s.contains("b + a"));
    }

    #[test]
    fn test_collect_with_constants() {
        let rule = CollectRule;
        // collect(a*x + 2*x + 5, x) -> (a+2)*x + 5
        let expr = parse("collect(a*x + 2*x + 5, x)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        let s = format!("{}", rewrite.new_expr);
        assert!(s.contains("a + 2") || s.contains("2 + a"));
        assert!(s.contains("5"));
    }

    #[test]
    fn test_collect_powers() {
        let rule = CollectRule;
        // collect(3*x^2 + y*x^2 + x, x) -> (3+y)*x^2 + x
        let expr = parse("collect(3*x^2 + y*x^2 + x, x)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        let s = format!("{}", rewrite.new_expr);
        assert!(s.contains("3 + y") || s.contains("y + 3"));
        assert!(s.contains("x^2"));
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(CollectRule));
}
