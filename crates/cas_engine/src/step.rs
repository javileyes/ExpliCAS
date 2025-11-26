use cas_ast::Expr;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct Step {
    pub description: String,
    pub rule_name: String,
    pub before: Rc<Expr>,
    pub after: Rc<Expr>,
}

impl Step {
    pub fn new(description: &str, rule_name: &str, before: Rc<Expr>, after: Rc<Expr>) -> Self {
        Self {
            description: description.to_string(),
            rule_name: rule_name.to_string(),
            before,
            after,
        }
    }
}
