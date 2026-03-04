use cas_ast::ExprId;

/// Hint emitted when an Analytic condition blocks transformation in Generic mode.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BlockedHint {
    /// Required condition key that prevented this rewrite.
    pub key: crate::assumption_model::AssumptionKey,
    /// Expression associated with the blocked condition.
    pub expr_id: ExprId,
    /// Rule name that was blocked.
    pub rule: String,
    /// User-facing suggestion.
    pub suggestion: &'static str,
}
