use cas_ast::ExprId;

/// Engine-level events emitted during symbolic transformation.
///
/// These events are transport-agnostic and can be consumed by didactic/timeline
/// layers to build user-facing narratives without coupling to core rewrite code.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineEvent {
    /// A rewrite rule transformed one local expression into another.
    RuleApplied {
        /// Rule identifier (`Rule::name()`).
        rule_name: String,
        /// Local expression id before applying the rule.
        before: ExprId,
        /// Local expression id after applying the rule.
        after: ExprId,
        /// Global expression id before applying the rule at path, if available.
        global_before: Option<ExprId>,
        /// Global expression id after applying the rule at path, if available.
        global_after: Option<ExprId>,
        /// Whether the event comes from a chained rewrite.
        is_chained: bool,
    },
}

/// Observer trait for consumers that want to listen to engine events.
pub trait StepListener {
    /// Receive one event emitted by the engine.
    fn on_event(&mut self, event: &EngineEvent);
}
