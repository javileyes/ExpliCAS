use crate::rule::{Rewrite, Rule};
use cas_ast::Expr;
use std::rc::Rc;

pub struct DistributeRule;
impl Rule for DistributeRule {
    fn name(&self) -> &str {
        "Distributive Property"
    }
    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
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
}

pub struct AnnihilationRule;
impl Rule for AnnihilationRule {
    fn name(&self) -> &str {
        "Annihilation"
    }
    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
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
}

use num_rational::BigRational;
use num_traits::One;

pub struct CombineLikeTermsRule;
impl Rule for CombineLikeTermsRule {
    fn name(&self) -> &str {
        "Combine Like Terms"
    }
    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
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
        // ln(x) is log(e, x)
        assert_eq!(format!("{}", rewrite3.new_expr), "2 * log(e, x)");
    }
}
