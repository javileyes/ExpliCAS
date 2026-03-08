//! Generic shared context for recursive solver orchestration.
//!
//! Encapsulates:
//! - recursion depth tracking
//! - shared required-condition sink
//! - shared assumption-event sink
//! - shared output-scope sink
//!
//! Runtime crates can wrap this with domain-specific state (for example,
//! per-level semantic environments) without re-implementing shared-sink logic.

use std::collections::HashSet;
use std::hash::Hash;

use crate::shared_sink::SharedSink;

/// Shared recursive solve context state.
#[derive(Debug, Clone)]
pub struct SolveSharedContext<Required, Assumption, Scope> {
    depth: usize,
    required_sink: SharedSink<HashSet<Required>>,
    assumptions_sink: SharedSink<Vec<Assumption>>,
    output_scopes_sink: SharedSink<Vec<Scope>>,
}

impl<Required, Assumption, Scope> SolveSharedContext<Required, Assumption, Scope> {
    /// Build a child context sharing all sinks and increasing recursion depth.
    pub fn fork_next_depth(&self) -> Self {
        Self {
            depth: self.depth.saturating_add(1),
            required_sink: self.required_sink.clone(),
            assumptions_sink: self.assumptions_sink.clone(),
            output_scopes_sink: self.output_scopes_sink.clone(),
        }
    }

    /// Current recursion depth.
    pub fn depth(&self) -> usize {
        self.depth
    }
}

impl<Required, Assumption, Scope> Default for SolveSharedContext<Required, Assumption, Scope>
where
    Required: Eq + Hash,
{
    fn default() -> Self {
        Self {
            depth: 0,
            required_sink: SharedSink::new(HashSet::new()),
            assumptions_sink: SharedSink::new(Vec::new()),
            output_scopes_sink: SharedSink::new(Vec::new()),
        }
    }
}

impl<Required, Assumption, Scope> SolveSharedContext<Required, Assumption, Scope>
where
    Required: Eq + Hash + Clone,
    Assumption: Clone,
    Scope: Clone + PartialEq,
{
    /// Record one required condition in the shared accumulator.
    pub fn note_required_condition(&self, condition: Required) {
        self.required_sink.with_mut(|required| {
            required.insert(condition);
        });
    }

    /// Snapshot all required conditions accumulated by this solve tree.
    pub fn required_conditions(&self) -> Vec<Required> {
        self.required_sink
            .with(|required| required.iter().cloned().collect())
    }

    /// Record one assumption emitted during solve.
    pub fn note_assumption(&self, event: Assumption) {
        self.assumptions_sink
            .with_mut(|assumptions| assumptions.push(event));
    }

    /// Snapshot collected assumption events.
    pub fn assumptions(&self) -> Vec<Assumption> {
        self.assumptions_sink
            .with(|assumptions| assumptions.clone())
    }

    /// Emit one output scope while preserving first-seen order and deduplication.
    pub fn emit_scope(&self, scope: Scope) {
        self.output_scopes_sink.with_mut(|scopes| {
            if !scopes.contains(&scope) {
                scopes.push(scope);
            }
        });
    }

    /// Snapshot collected output scopes.
    pub fn output_scopes(&self) -> Vec<Scope> {
        self.output_scopes_sink.with(|scopes| scopes.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::SolveSharedContext;

    #[test]
    fn fork_shares_required_and_depth_progresses() {
        let parent: SolveSharedContext<i32, &'static str, &'static str> =
            SolveSharedContext::default();
        parent.note_required_condition(1);

        let child = parent.fork_next_depth();
        child.note_required_condition(2);

        let mut required = parent.required_conditions();
        required.sort_unstable();
        assert_eq!(required, vec![1, 2]);
        assert_eq!(parent.depth(), 0);
        assert_eq!(child.depth(), 1);
    }

    #[test]
    fn assumptions_and_scopes_are_shared_with_scope_dedup() {
        let parent: SolveSharedContext<i32, &'static str, &'static str> =
            SolveSharedContext::default();
        let child = parent.fork_next_depth();

        child.note_assumption("a1");
        child.note_assumption("a2");
        child.emit_scope("s1");
        child.emit_scope("s1");
        child.emit_scope("s2");

        assert_eq!(parent.assumptions(), vec!["a1", "a2"]);
        assert_eq!(parent.output_scopes(), vec!["s1", "s2"]);
    }
}
