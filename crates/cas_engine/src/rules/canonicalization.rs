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
            if name == "sqrt" && args.len() == 1 {
                // sqrt(x) -> x^(1/2)
                return Some(Rewrite {
                    new_expr: Expr::pow(args[0].clone(), Expr::rational(1, 2)),
                    description: "sqrt(x) = x^(1/2)".to_string(),
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
}
