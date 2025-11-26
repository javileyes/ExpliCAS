use cas_ast::Expr;
use std::rc::Rc;

pub struct Rewrite {
    pub new_expr: Rc<Expr>,
    pub description: String,
}

pub trait Rule {
    fn name(&self) -> &str;
    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite>;
}
