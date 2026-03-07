use cas_ast::ExprId;

/// Successful output payload for assignment-style commands (`let`, `:=`, direct assign).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssignmentCommandOutput {
    pub name: String,
    pub expr: ExprId,
    pub lazy: bool,
}
