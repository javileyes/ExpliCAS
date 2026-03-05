pub(crate) const RATIONALIZE_USAGE_MESSAGE: &str = "Usage: rationalize <expr>\n\
                 Example: rationalize 1/(1 + sqrt(2) + sqrt(3))";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RationalizeCommandOutcome {
    Success(cas_ast::ExprId),
    NotApplicable,
    BudgetExceeded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RationalizeCommandEvalError {
    Parse(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RationalizeCommandEvalOutput {
    pub(crate) normalized_expr: cas_ast::ExprId,
    pub(crate) outcome: RationalizeCommandOutcome,
}
