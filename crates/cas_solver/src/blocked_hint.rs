use cas_ast::ExprId;

/// Hint emitted when a simplification is blocked by domain policy.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BlockedHint {
    /// Required condition key that prevented this rewrite.
    pub key: crate::AssumptionKey,
    /// Expression associated with the blocked condition.
    pub expr_id: ExprId,
    /// Rule name that was blocked.
    pub rule: String,
    /// User-facing suggestion.
    pub suggestion: &'static str,
}

fn blocked_hint_from_engine_owned(value: cas_engine::BlockedHint) -> BlockedHint {
    BlockedHint {
        key: crate::assumption_model::assumption_key_from_engine(value.key),
        expr_id: value.expr_id,
        rule: value.rule,
        suggestion: value.suggestion,
    }
}

fn blocked_hint_to_engine_owned(value: BlockedHint) -> cas_engine::BlockedHint {
    cas_engine::BlockedHint {
        key: crate::assumption_model::assumption_key_to_engine(value.key),
        expr_id: value.expr_id,
        rule: value.rule,
        suggestion: value.suggestion,
    }
}

/// Convert blocked hints from engine payloads into solver-owned models.
pub fn blocked_hints_from_engine(hints: &[cas_engine::BlockedHint]) -> Vec<BlockedHint> {
    hints
        .iter()
        .cloned()
        .map(blocked_hint_from_engine_owned)
        .collect()
}

/// Register a blocked hint in the engine-side thread-local collector.
pub fn register_blocked_hint(hint: BlockedHint) {
    cas_engine::register_blocked_hint(blocked_hint_to_engine_owned(hint));
}

/// Take and clear blocked hints from the thread-local collector.
pub fn take_blocked_hints() -> Vec<BlockedHint> {
    cas_engine::take_blocked_hints()
        .into_iter()
        .map(blocked_hint_from_engine_owned)
        .collect()
}

/// Clear blocked hints without returning them.
pub fn clear_blocked_hints() {
    cas_engine::clear_blocked_hints();
}
