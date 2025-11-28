use crate::rule::{Rewrite, Rule};
use crate::define_rule;
use cas_ast::Expr;
use std::rc::Rc;
use num_traits::{ToPrimitive, Signed};
use num_rational::BigRational;
use num_traits::One;

define_rule!(
    DistributeRule,
    "Distributive Property",
    |expr| {
        if let Expr::Mul(l, r) = expr.as_ref() {
            // a * (b + c) -> a*b + a*c
            if let Expr::Add(b, c) = r.as_ref() {
                return Some(Rewrite {
                    new_expr: Expr::add(
                        Expr::mul(l.clone(), b.clone()),
                        Expr::mul(l.clone(), c.clone()),
                    ),
                    description: "Distribute".to_string(),
                });
            }
            // (b + c) * a -> b*a + c*a
            if let Expr::Add(b, c) = l.as_ref() {
                return Some(Rewrite {
                    new_expr: Expr::add(
                        Expr::mul(b.clone(), r.clone()),
                        Expr::mul(c.clone(), r.clone()),
                    ),
                    description: "Distribute".to_string(),
                });
            }
        }
        None
    }
);

define_rule!(
    AnnihilationRule,
    "Annihilation",
    |expr| {
        if let Expr::Sub(l, r) = expr.as_ref() {
            if l == r {
                return Some(Rewrite {
                    new_expr: Expr::num(0),
                    description: "x - x = 0".to_string(),
                });
            }
        }
        None
    }
);

define_rule!(
    CombineLikeTermsRule,
    "Combine Like Terms",
    |expr| {
        if let Expr::Add(l, r) = expr.as_ref() {
            // Helper to extract (coeff, var_part)
            // 2x -> (2, x)
            // x -> (1, x)
            let get_parts = |e: &Rc<Expr>| -> Option<(BigRational, Rc<Expr>)> {
                match e.as_ref() {
                    Expr::Mul(a, b) => {
                        if let Expr::Number(n) = a.as_ref() {
                            Some((n.clone(), b.clone()))
                        } else if let Expr::Number(n) = b.as_ref() {
                            Some((n.clone(), a.clone()))
                        } else {
                            None
                        }
                    }
                    Expr::Number(_) => None, // Handled by CombineConstantsRule
                    _ => Some((BigRational::one(), e.clone())),
                }
            };

            if let (Some((c1, v1)), Some((c2, v2))) = (get_parts(l), get_parts(r)) {
                if v1 == v2 {
                    let new_coeff = &c1 + &c2;
                    let new_term = if new_coeff.is_one() {
                        v1.clone()
                    } else {
                        Expr::mul(Rc::new(Expr::Number(new_coeff.clone())), v1.clone())
                    };
                    return Some(Rewrite {
                        new_expr: new_term,
                        description: format!("Combine like terms: {}{} + {}{}", c1, v1, c2, v2),
                    });
                }
            }
        }
        None
    }
);

define_rule!(
    BinomialExpansionRule,
    "Binomial Expansion",
    |expr| {
        // (a + b)^n
        if let Expr::Pow(base, exp) = expr.as_ref() {
            if let Expr::Add(a, b) = base.as_ref() {
                if let Expr::Number(n) = exp.as_ref() {
                    if n.is_integer() && !n.is_negative() {
                        let n_val = n.to_integer().to_u32()?;
                        // Limit expansion to reasonable size to prevent explosion
                        if n_val >= 2 && n_val <= 10 {
                            // Expand: sum(k=0 to n) (n choose k) * a^(n-k) * b^k
                            let mut terms = Vec::new();
                            for k in 0..=n_val {
                                let coeff = binomial_coeff(n_val, k);
                                let exp_a = n_val - k;
                                let exp_b = k;
                                
                                let term_a = if exp_a == 0 { Expr::num(1) } else if exp_a == 1 { a.clone() } else { Expr::pow(a.clone(), Expr::num(exp_a as i64)) };
                                let term_b = if exp_b == 0 { Expr::num(1) } else if exp_b == 1 { b.clone() } else { Expr::pow(b.clone(), Expr::num(exp_b as i64)) };
                                
                                let mut term = Expr::mul(term_a, term_b);
                                if coeff > 1 {
                                    term = Expr::mul(Expr::num(coeff as i64), term);
                                }
                                terms.push(term);
                            }
                            
                            // Sum up terms
                            let mut expanded = terms[0].clone();
                            for i in 1..terms.len() {
                                expanded = Expr::add(expanded, terms[i].clone());
                            }
                            
                            return Some(Rewrite {
                                new_expr: expanded,
                                description: format!("Expand binomial power ^{}", n_val),
                            });
                        }
                    }
                }
            }
        }
        None
    }
);

fn binomial_coeff(n: u32, k: u32) -> u32 {
    if k == 0 || k == n {
        return 1;
    }
    if k > n {
        return 0;
    }
    let mut res = 1;
    for i in 0..k {
        res = res * (n - i) / (i + 1);
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distribute() {
        let rule = DistributeRule;
        // 2 * (x + 3)
        let expr = Expr::mul(
            Expr::num(2),
            Expr::add(Expr::var("x"), Expr::num(3))
        );
        let rewrite = rule.apply(&expr).unwrap();
        // Should be (2 * x) + (2 * 3)
        // Note: Simplification of 2*3 happens in a later pass by CombineConstantsRule
        assert_eq!(format!("{}", rewrite.new_expr), "2 * x + 2 * 3");
    }

    #[test]
    fn test_annihilation() {
        let rule = AnnihilationRule;
        let expr = Expr::sub(Expr::var("x"), Expr::var("x"));
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "0");
    }

    #[test]
    fn test_combine_like_terms() {
        let rule = CombineLikeTermsRule;
        
        // 2x + 3x -> 5x
        let expr = Expr::add(
            Expr::mul(Expr::num(2), Expr::var("x")),
            Expr::mul(Expr::num(3), Expr::var("x"))
        );
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "5 * x");

        // x + 2x -> 3x
        let expr2 = Expr::add(
            Expr::var("x"),
            Expr::mul(Expr::num(2), Expr::var("x"))
        );
        let rewrite2 = rule.apply(&expr2).unwrap();
        assert_eq!(format!("{}", rewrite2.new_expr), "3 * x");

        // ln(x) + ln(x) -> 2 * ln(x)
        let expr3 = Expr::add(
            Expr::ln(Expr::var("x")),
            Expr::ln(Expr::var("x"))
        );
        let rewrite3 = rule.apply(&expr3).unwrap();
        // ln(x) is log(e, x), prints as ln(x)
        assert_eq!(format!("{}", rewrite3.new_expr), "2 * ln(x)");
    }
}
