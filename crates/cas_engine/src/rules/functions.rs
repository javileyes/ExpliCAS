use crate::rule::{Rule, Rewrite};
use cas_ast::Expr;
use std::rc::Rc;
use num_traits::Signed;

pub struct EvaluateAbsRule;

impl Rule for EvaluateAbsRule {
    fn name(&self) -> &str {
        "Evaluate Absolute Value"
    }

    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
        if let Expr::Function(name, args) = expr.as_ref() {
            if name == "abs" && args.len() == 1 {
                let arg = &args[0];
                
                // Case 1: abs(number)
                if let Expr::Number(n) = arg.as_ref() {
                    // Always evaluate to positive number
                    return Some(Rewrite {
                        new_expr: Rc::new(Expr::Number(n.abs())),
                        description: format!("abs({}) = {}", n, n.abs()),
                    });
                }
                
                // Case 2: abs(-x) -> abs(x)
                if let Expr::Neg(inner) = arg.as_ref() {
                    // If inner is a number, we can simplify fully: abs(-5) -> 5
                    if let Expr::Number(n) = inner.as_ref() {
                        return Some(Rewrite {
                            new_expr: Rc::new(Expr::Number(n.clone())),
                            description: format!("abs(-{}) = {}", n, n),
                        });
                    }

                    return Some(Rewrite {
                        new_expr: Expr::abs(inner.clone()),
                        description: "abs(-x) = abs(x)".to_string(),
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
    use cas_parser::parse;

    #[test]
    fn test_evaluate_abs() {
        let rule = EvaluateAbsRule;

        // abs(-5) -> 5
        let expr1 = parse("abs(-5)").expect("Failed to parse abs(-5)");
        let rewrite1 = rule.apply(&expr1).expect("Rule failed to apply");
        assert_eq!(format!("{}", rewrite1.new_expr), "5");

        // abs(5) -> 5
        let expr2 = parse("abs(5)").expect("Failed to parse abs(5)");
        let rewrite2 = rule.apply(&expr2).expect("Rule failed to apply");
        assert_eq!(format!("{}", rewrite2.new_expr), "5");

        // abs(-x) -> abs(x)
        let expr3 = parse("abs(-x)").expect("Failed to parse abs(-x)");
        let rewrite3 = rule.apply(&expr3).expect("Rule failed to apply");
        assert_eq!(format!("{}", rewrite3.new_expr), "|x|");
    }
}
