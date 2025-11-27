use crate::rule::{Rule, Rewrite};
use cas_ast::Expr;
use std::rc::Rc;
use std::cmp::Ordering;
use crate::ordering::compare_expr;

pub struct CanonicalizeNegationRule;
// ... (existing CanonicalizeNegationRule impl) ...

pub struct CanonicalizeAddRule;

impl Rule for CanonicalizeAddRule {
    fn name(&self) -> &str {
        "Canonicalize Addition"
    }

    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
        if let Expr::Add(lhs, rhs) = expr.as_ref() {
            // 1. Basic Swap: b + a -> a + b if b < a
            if compare_expr(rhs, lhs) == Ordering::Less {
                return Some(Rewrite {
                    new_expr: Expr::add(rhs.clone(), lhs.clone()),
                    description: "Reorder addition terms".to_string(),
                });
            }
            
            // 2. Rotation: a + (b + c) -> b + (a + c) if b < a
            // This allows sorting nested terms.
            if let Expr::Add(rl, rr) = rhs.as_ref() {
                if compare_expr(rl, lhs) == Ordering::Less {
                    return Some(Rewrite {
                        new_expr: Expr::add(rl.clone(), Expr::add(lhs.clone(), rr.clone())),
                        description: "Rotate addition terms".to_string(),
                    });
                }
            }
        }
        None
    }
}

pub struct CanonicalizeMulRule;

impl Rule for CanonicalizeMulRule {
    fn name(&self) -> &str {
        "Canonicalize Multiplication"
    }

    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
        if let Expr::Mul(lhs, rhs) = expr.as_ref() {
            // 1. Basic Swap: b * a -> a * b if b < a
            if compare_expr(rhs, lhs) == Ordering::Less {
                return Some(Rewrite {
                    new_expr: Expr::mul(rhs.clone(), lhs.clone()),
                    description: "Reorder multiplication factors".to_string(),
                });
            }

            // 2. Rotation: a * (b * c) -> b * (a * c) if b < a
            if let Expr::Mul(rl, rr) = rhs.as_ref() {
                if compare_expr(rl, lhs) == Ordering::Less {
                    return Some(Rewrite {
                        new_expr: Expr::mul(rl.clone(), Expr::mul(lhs.clone(), rr.clone())),
                        description: "Rotate multiplication factors".to_string(),
                    });
                }
            }
        }
        None
    }
}

impl Rule for CanonicalizeNegationRule {
    fn name(&self) -> &str {
        "Canonicalize Negation"
    }

    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
        // 1. Subtraction: a - b -> a + (-b)
        if let Expr::Sub(lhs, rhs) = expr.as_ref() {
            return Some(Rewrite {
                new_expr: Expr::add(lhs.clone(), Expr::neg(rhs.clone())),
                description: "a - b = a + (-b)".to_string(),
            });
        }

        // 2. Negation: -x -> -1 * x
        if let Expr::Neg(inner) = expr.as_ref() {
            if let Expr::Number(n) = inner.as_ref() {
                // -(-5) -> 5 (Handled by parser usually, but good to have)
                // Actually parser produces Neg(Number(5)).
                // If we have Neg(Number(5)), we want Number(-5).
                return Some(Rewrite {
                    new_expr: Rc::new(Expr::Number(-n)),
                    description: format!("-({}) = {}", n, -n),
                });
            } else {
                // -x -> -1 * x
                return Some(Rewrite {
                    new_expr: Expr::mul(Expr::num(-1), inner.clone()),
                    description: "-x = -1 * x".to_string(),
                });
            }
        }
        None
    }
}

pub struct CanonicalizeRootRule;

impl Rule for CanonicalizeRootRule {
    fn name(&self) -> &str {
        "Canonicalize Roots"
    }

    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
        if let Expr::Function(name, args) = expr.as_ref() {
            if name == "sqrt" {
                if args.len() == 1 {
                    // sqrt(x) -> x^(1/2)
                    return Some(Rewrite {
                        new_expr: Expr::pow(args[0].clone(), Expr::rational(1, 2)),
                        description: "sqrt(x) = x^(1/2)".to_string(),
                    });
                } else if args.len() == 2 {
                    // sqrt(x, n) -> x^(1/n)
                    return Some(Rewrite {
                        new_expr: Expr::pow(args[0].clone(), Expr::div(Expr::num(1), args[1].clone())),
                        description: format!("sqrt(x, n) = x^(1/n)"),
                    });
                }
            } else if name == "root" && args.len() == 2 {
                 // root(x, n) -> x^(1/n)
                 return Some(Rewrite {
                    new_expr: Expr::pow(args[0].clone(), Expr::div(Expr::num(1), args[1].clone())),
                    description: format!("root(x, n) = x^(1/n)"),
                });
            }
        }
        None
    }
}

pub struct AssociativityRule;

impl Rule for AssociativityRule {
    fn name(&self) -> &str {
        "Associativity (Flattening)"
    }

    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
        match expr.as_ref() {
            // (a + b) + c -> a + (b + c)
            Expr::Add(lhs, rhs) => {
                if let Expr::Add(ll, lr) = lhs.as_ref() {
                    return Some(Rewrite {
                        new_expr: Expr::add(ll.clone(), Expr::add(lr.clone(), rhs.clone())),
                        description: "Associativity: (a + b) + c -> a + (b + c)".to_string(),
                    });
                }
            }
            // (a * b) * c -> a * (b * c)
            Expr::Mul(lhs, rhs) => {
                if let Expr::Mul(ll, lr) = lhs.as_ref() {
                    return Some(Rewrite {
                        new_expr: Expr::mul(ll.clone(), Expr::mul(lr.clone(), rhs.clone())),
                        description: "Associativity: (a * b) * c -> a * (b * c)".to_string(),
                    });
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
    use cas_parser::parse;

    #[test]
    fn test_canonicalize_negation() {
        let rule = CanonicalizeNegationRule;
        // -5 -> -5 (Number)
        let expr = parse("-5").unwrap(); // Neg(Number(5))
        let rewrite = rule.apply(&expr).unwrap();
        // The display might look the same "-5", but the structure is different.
        // Let's check if it's a Number.
        if let Expr::Number(n) = rewrite.new_expr.as_ref() {
            assert_eq!(format!("{}", n), "-5");
        } else {
            panic!("Expected Number, got {:?}", rewrite.new_expr);
        }
    }

    #[test]
    fn test_canonicalize_sqrt() {
        let rule = CanonicalizeRootRule;
        // sqrt(x)
        let expr = Rc::new(Expr::Function("sqrt".to_string(), vec![Expr::var("x")]));
        let rewrite = rule.apply(&expr).unwrap();
        // Should be x^(1/2)
        assert_eq!(format!("{}", rewrite.new_expr), "x^(1/2)");
    }

    #[test]
    fn test_canonicalize_nth_root() {
        let rule = CanonicalizeRootRule;
        
        // sqrt(x, 3) -> x^(1/3)
        let expr = Rc::new(Expr::Function("sqrt".to_string(), vec![Expr::var("x"), Expr::num(3)]));
        let rewrite = rule.apply(&expr).unwrap();
        // 1/3 is a division of numbers, which engine simplifies to rational if possible, 
        // but here we construct Expr::div(1, 3). 
        // Wait, Expr::div(1, 3) prints as "1 / 3". 
        // If it was simplified, it would be "1/3".
        // The rule produces Expr::div(1, args[1]).
        // If args[1] is 3, it is Expr::div(1, 3).
        // This is NOT Expr::Number(1/3). It is an operation.
        // So it prints as "1 / 3".
        // And because it's in exponent, it gets parens: x^(1 / 3).
        assert_eq!(format!("{}", rewrite.new_expr), "x^(1 / 3)");

        // root(x, 4) -> x^(1/4)
        let expr2 = Rc::new(Expr::Function("root".to_string(), vec![Expr::var("x"), Expr::num(4)]));
        let rewrite2 = rule.apply(&expr2).unwrap();
        assert_eq!(format!("{}", rewrite2.new_expr), "x^(1 / 4)");
    }
}
