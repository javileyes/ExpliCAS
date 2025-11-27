use crate::rule::{Rewrite, Rule};
use cas_ast::Expr;
use std::rc::Rc;
use num_traits::{Zero, One};

pub struct AddZeroRule;

impl Rule for AddZeroRule {
    fn name(&self) -> &str {
        "Identity Property of Addition"
    }

    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
        if let Expr::Add(lhs, rhs) = expr.as_ref() {
            if let Expr::Number(n) = rhs.as_ref() {
                if n.is_zero() {
                    return Some(Rewrite {
                        new_expr: lhs.clone(),
                        description: "x + 0 = x".to_string(),
                    });
                }
            }
            if let Expr::Number(n) = lhs.as_ref() {
                if n.is_zero() {
                    return Some(Rewrite {
                        new_expr: rhs.clone(),
                        description: "0 + x = x".to_string(),
                    });
                }
            }
        }
        None
    }
}

pub struct MulOneRule;

impl Rule for MulOneRule {
    fn name(&self) -> &str {
        "Identity Property of Multiplication"
    }

    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
        if let Expr::Mul(lhs, rhs) = expr.as_ref() {
            if let Expr::Number(n) = rhs.as_ref() {
                if n.is_one() {
                    return Some(Rewrite {
                        new_expr: lhs.clone(),
                        description: "x * 1 = x".to_string(),
                    });
                }
            }
            if let Expr::Number(n) = lhs.as_ref() {
                if n.is_one() {
                    return Some(Rewrite {
                        new_expr: rhs.clone(),
                        description: "1 * x = x".to_string(),
                    });
                }
            }
        }
        None
    }
}

pub struct MulZeroRule;

impl Rule for MulZeroRule {
    fn name(&self) -> &str {
        "Zero Property of Multiplication"
    }

    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
        if let Expr::Mul(lhs, rhs) = expr.as_ref() {
            if let Expr::Number(n) = rhs.as_ref() {
                if n.is_zero() {
                    return Some(Rewrite {
                        new_expr: Rc::new(Expr::Number(num_rational::BigRational::zero())),
                        description: "x * 0 = 0".to_string(),
                    });
                }
            }
            if let Expr::Number(n) = lhs.as_ref() {
                if n.is_zero() {
                    return Some(Rewrite {
                        new_expr: Rc::new(Expr::Number(num_rational::BigRational::zero())),
                        description: "0 * x = 0".to_string(),
                    });
                }
            }
        }
        None
    }
}

pub struct CombineConstantsRule;

impl Rule for CombineConstantsRule {
    fn name(&self) -> &str {
        "Combine Constants"
    }

    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
        match expr.as_ref() {
            Expr::Add(lhs, rhs) => {
                if let Expr::Number(n1) = lhs.as_ref() {
                    if let Expr::Number(n2) = rhs.as_ref() {
                        return Some(Rewrite {
                            new_expr: Rc::new(Expr::Number(n1 + n2)),
                            description: format!("{} + {} = {}", n1, n2, n1 + n2),
                        });
                    }
                    // Handle nested: c1 + (c2 + x) -> (c1+c2) + x
                    if let Expr::Add(rl, rr) = rhs.as_ref() {
                        if let Expr::Number(n2) = rl.as_ref() {
                            return Some(Rewrite {
                                new_expr: Expr::add(Rc::new(Expr::Number(n1 + n2)), rr.clone()),
                                description: format!("Combine nested constants: {} + {}", n1, n2),
                            });
                        }
                    }

                    // // Handle Add(Number, Mul(Number, Number)) - e.g. 1 + (-1 * 1)
                    // if let Expr::Mul(ml, mr) = rhs.as_ref() {
                    //     if let (Expr::Number(n2), Expr::Number(n3)) = (ml.as_ref(), mr.as_ref()) {
                    //         let product = n2 * n3;
                    //         return Some(Rewrite {
                    //             new_expr: Rc::new(Expr::Number(n1 + product.clone())),
                    //             description: format!("{} + ({} * {}) = {}", n1, n2, n3, n1 + product),
                    //         });
                    //     }
                    // }
                }
            }
            Expr::Mul(lhs, rhs) => {
                if let Expr::Number(n1) = lhs.as_ref() {
                    if let Expr::Number(n2) = rhs.as_ref() {
                        return Some(Rewrite {
                            new_expr: Rc::new(Expr::Number(n1 * n2)),
                            description: format!("{} * {} = {}", n1, n2, n1 * n2),
                        });
                    }
                    // Handle nested: c1 * (c2 * x) -> (c1*c2) * x
                    if let Expr::Mul(rl, rr) = rhs.as_ref() {
                        if let Expr::Number(n2) = rl.as_ref() {
                            return Some(Rewrite {
                                new_expr: Expr::mul(Rc::new(Expr::Number(n1 * n2)), rr.clone()),
                                description: format!("Combine nested constants: {} * {}", n1, n2),
                            });
                        }
                    }
                }
            }
            Expr::Sub(lhs, rhs) => {
                if let (Expr::Number(n1), Expr::Number(n2)) = (lhs.as_ref(), rhs.as_ref()) {
                    return Some(Rewrite {
                        new_expr: Rc::new(Expr::Number(n1 - n2)),
                        description: format!("{} - {} = {}", n1, n2, n1 - n2),
                    });
                }
            }
            Expr::Div(lhs, rhs) => {
                if let (Expr::Number(n1), Expr::Number(n2)) = (lhs.as_ref(), rhs.as_ref()) {
                    if !n2.is_zero() {
                        return Some(Rewrite {
                            new_expr: Rc::new(Expr::Number(n1 / n2)),
                            description: format!("{} / {} = {}", n1, n2, n1 / n2),
                        });
                    }
                }
            }
            _ => {}
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_zero() {
        let rule = AddZeroRule;
        let expr = Expr::add(Expr::var("x"), Expr::num(0));
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "x");
    }

    #[test]
    fn test_mul_one() {
        let rule = MulOneRule;
        let expr = Expr::mul(Expr::num(1), Expr::var("y"));
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "y");
    }

    #[test]
    fn test_combine_constants() {
        let rule = CombineConstantsRule;
        let expr = Expr::add(Expr::num(2), Expr::num(3));
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "5");
    }
}
