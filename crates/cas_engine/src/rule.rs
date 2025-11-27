use cas_ast::Expr;
use std::rc::Rc;

pub struct Rewrite {
    pub new_expr: Rc<Expr>,
    pub description: String,
}

pub trait Rule {
    fn name(&self) -> &str;
    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite>;
    
    // Optional: Return list of Expr variant names this rule targets.
    // If None, the rule is applied to all nodes.
    // Common variants: "Add", "Sub", "Mul", "Div", "Pow", "Neg", "Function", "Variable", "Number", "Constant"
    fn target_types(&self) -> Option<Vec<&str>> {
        None
    }
}
