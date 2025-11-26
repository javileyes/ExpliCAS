use crate::rule::{Rewrite, Rule};
use cas_ast::Expr;
use std::rc::Rc;

pub struct AddZeroRule;
impl Rule for AddZeroRule {
    fn name(&self) -> &str {
        "Identity Property of Addition"
    }
    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
        if let Expr::Add(l, r) = expr.as_ref() {
            if let Expr::Number(0) = l.as_ref() {
                return Some(Rewrite {
                    new_expr: r.clone(),
                    description: "0 + x = x".to_string(),
                });
            }
            if let Expr::Number(0) = r.as_ref() {
                return Some(Rewrite {
                    new_expr: l.clone(),
                    description: "x + 0 = x".to_string(),
                });
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
        if let Expr::Mul(l, r) = expr.as_ref() {
            if let Expr::Number(1) = l.as_ref() {
                return Some(Rewrite {
                    new_expr: r.clone(),
                    description: "1 * x = x".to_string(),
                });
            }
            if let Expr::Number(1) = r.as_ref() {
                return Some(Rewrite {
                    new_expr: l.clone(),
                    description: "x * 1 = x".to_string(),
                });
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
            Expr::Add(l, r) => {
                if let (Expr::Number(a), Expr::Number(b)) = (l.as_ref(), r.as_ref()) {
                    return Some(Rewrite {
                        new_expr: Expr::num(a + b),
                        description: format!("{} + {} = {}", a, b, a + b),
                    });
                }
            }
            Expr::Mul(l, r) => {
                if let (Expr::Number(a), Expr::Number(b)) = (l.as_ref(), r.as_ref()) {
                    return Some(Rewrite {
                        new_expr: Expr::num(a * b),
                        description: format!("{} * {} = {}", a, b, a * b),
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
