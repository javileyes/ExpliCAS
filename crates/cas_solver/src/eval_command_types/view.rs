use cas_ast::ExprId;

#[derive(Debug, Clone)]
pub(crate) struct EvalCommandEvalView {
    pub(crate) stored_id: Option<u64>,
    pub(crate) parsed: ExprId,
    pub(crate) resolved: ExprId,
    pub(crate) result: crate::EvalResult,
    pub(crate) diagnostics: crate::Diagnostics,
    pub(crate) steps: crate::DisplayEvalSteps,
    pub(crate) domain_warnings: Vec<crate::DomainWarning>,
    pub(crate) blocked_hints: Vec<crate::BlockedHint>,
}
