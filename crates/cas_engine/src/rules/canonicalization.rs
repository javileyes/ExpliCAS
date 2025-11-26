use crate::rule::{Rule, Rewrite};
use cas_ast::Expr;
use std::rc::Rc;

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

#[cfg(test)]
mod tests {
    use super::*;

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
