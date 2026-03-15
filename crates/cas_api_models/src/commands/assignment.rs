use cas_ast::ExprId;

/// Errors returned when applying a `let` assignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssignmentError {
    EmptyName,
    InvalidNameStart,
    ReservedName(String),
    Parse(String),
}

/// Successful output payload for assignment-style commands (`let`, `:=`, direct assign).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssignmentCommandOutput {
    pub name: String,
    pub expr: ExprId,
    pub lazy: bool,
}
