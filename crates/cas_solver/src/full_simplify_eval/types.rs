use cas_ast::ExprId;

#[derive(Debug, Clone)]
pub struct FullSimplifyEvalOutput {
    pub resolved_expr: ExprId,
    pub simplified_expr: ExprId,
    pub steps: Vec<crate::Step>,
}
