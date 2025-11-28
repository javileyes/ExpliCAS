use cas_ast::{ExprId, Context};

pub struct Rewrite {
    pub new_expr: ExprId,
    pub description: String,
}

pub trait Rule {
    fn name(&self) -> &str;
    fn apply(&self, context: &mut Context, expr: ExprId) -> Option<Rewrite>;
    
    // Optional: Return list of Expr variant names this rule targets.
    // If None, the rule is applied to all nodes.
    // Common variants: "Add", "Sub", "Mul", "Div", "Pow", "Neg", "Function", "Variable", "Number", "Constant"
    fn target_types(&self) -> Option<Vec<&str>> {
        None
    }
}
