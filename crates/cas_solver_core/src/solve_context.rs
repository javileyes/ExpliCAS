//! Generic solver context wrapper for recursive solve orchestration.
//!
//! Combines:
//! - per-level domain environment (`domain_env`)
//! - shared recursive state (`SolveSharedContext`)
//!
//! Runtime crates can alias this type with domain-specific payloads.

use std::hash::Hash;

use crate::shared_context::SolveSharedContext;
use crate::solve_types::SolveDiagnostics;

/// Snapshot of shared solve accumulators.
#[derive(Debug, Clone)]
pub struct SolveContextSnapshot<Required, Assumption, Scope> {
    pub required: Vec<Required>,
    pub assumed: Vec<Assumption>,
    pub output_scopes: Vec<Scope>,
}

impl<Required, Assumption, Scope> SolveContextSnapshot<Required, Assumption, Scope> {
    /// Build diagnostics from this snapshot using a caller-provided
    /// assumption-record aggregation function.
    pub fn into_diagnostics<AssumptionRecord, F>(
        self,
        mut build_assumed_records: F,
    ) -> SolveDiagnostics<Required, Assumption, AssumptionRecord, Scope>
    where
        Assumption: Clone,
        F: FnMut(&[Assumption]) -> Vec<AssumptionRecord>,
    {
        let assumed_records = build_assumed_records(&self.assumed);
        SolveDiagnostics {
            required: self.required,
            assumed: self.assumed,
            assumed_records,
            output_scopes: self.output_scopes,
        }
    }
}

/// Generic recursive solve context.
#[derive(Debug, Clone)]
pub struct SolveContext<DomainEnv, Required, Assumption, Scope> {
    /// Domain environment inferred for the current recursion level.
    pub domain_env: DomainEnv,
    shared: SolveSharedContext<Required, Assumption, Scope>,
}

impl<DomainEnv, Required, Assumption, Scope> SolveContext<DomainEnv, Required, Assumption, Scope>
where
    Required: Eq + Hash + Clone,
    Assumption: Clone,
    Scope: Clone + PartialEq,
{
    /// Build a child context that shares accumulators and bumps solve depth.
    pub fn fork_with_domain_env_next_depth(&self, domain_env: DomainEnv) -> Self {
        Self {
            domain_env,
            shared: self.shared.fork_next_depth(),
        }
    }

    /// Current solve recursion depth.
    pub fn depth(&self) -> usize {
        self.shared.depth()
    }

    /// Record one required domain condition in the shared accumulator.
    pub fn note_required_condition(&self, condition: Required) {
        self.shared.note_required_condition(condition);
    }

    /// Snapshot all required conditions accumulated by this solve tree.
    pub fn required_conditions(&self) -> Vec<Required> {
        self.shared.required_conditions()
    }

    /// Record one assumption emitted during solve.
    pub fn note_assumption(&self, event: Assumption) {
        self.shared.note_assumption(event);
    }

    /// Snapshot collected solver assumptions.
    pub fn assumptions(&self) -> Vec<Assumption> {
        self.shared.assumptions()
    }

    /// Emit one output scope tag.
    pub fn emit_scope(&self, scope: Scope) {
        self.shared.emit_scope(scope);
    }

    /// Snapshot collected output scopes.
    pub fn output_scopes(&self) -> Vec<Scope> {
        self.shared.output_scopes()
    }

    /// Snapshot all shared accumulators (required, assumptions, scopes).
    pub fn snapshot(&self) -> SolveContextSnapshot<Required, Assumption, Scope> {
        SolveContextSnapshot {
            required: self.required_conditions(),
            assumed: self.assumptions(),
            output_scopes: self.output_scopes(),
        }
    }

    /// Build diagnostics directly from the shared accumulators.
    pub fn diagnostics_with_records<AssumptionRecord, F>(
        &self,
        build_assumed_records: F,
    ) -> SolveDiagnostics<Required, Assumption, AssumptionRecord, Scope>
    where
        F: FnMut(&[Assumption]) -> Vec<AssumptionRecord>,
    {
        self.snapshot().into_diagnostics(build_assumed_records)
    }
}

impl<DomainEnv, Required, Assumption, Scope> Default
    for SolveContext<DomainEnv, Required, Assumption, Scope>
where
    DomainEnv: Default,
    Required: Eq + Hash,
{
    fn default() -> Self {
        Self {
            domain_env: DomainEnv::default(),
            shared: SolveSharedContext::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SolveContext;

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    struct DomainEnv {
        id: u8,
    }

    type Ctx = SolveContext<DomainEnv, i32, &'static str, &'static str>;

    #[test]
    fn fork_shares_sinks_and_increments_depth() {
        let parent = Ctx::default();
        parent.note_required_condition(1);
        parent.note_assumption("a1");
        parent.emit_scope("s1");

        let child = parent.fork_with_domain_env_next_depth(DomainEnv { id: 7 });
        child.note_required_condition(2);
        child.note_assumption("a2");
        child.emit_scope("s2");

        assert_eq!(parent.depth(), 0);
        assert_eq!(child.depth(), 1);
        assert_eq!(child.domain_env.id, 7);

        let mut required = parent.required_conditions();
        required.sort_unstable();
        assert_eq!(required, vec![1, 2]);
        assert_eq!(parent.assumptions(), vec!["a1", "a2"]);
        assert_eq!(parent.output_scopes(), vec!["s1", "s2"]);
    }

    #[test]
    fn snapshot_into_diagnostics_preserves_payload() {
        let parent = Ctx::default();
        parent.note_required_condition(10);
        parent.note_assumption("assume:positive(x)");
        parent.emit_scope("scope:quadratic");

        let diagnostics = parent
            .snapshot()
            .into_diagnostics(|assumed| vec![format!("count={}", assumed.len())]);

        assert_eq!(diagnostics.required, vec![10]);
        assert_eq!(diagnostics.assumed, vec!["assume:positive(x)"]);
        assert_eq!(diagnostics.output_scopes, vec!["scope:quadratic"]);
        assert_eq!(diagnostics.assumed_records, vec!["count=1".to_string()]);
    }

    #[test]
    fn diagnostics_with_records_preserves_payload() {
        let parent = Ctx::default();
        parent.note_required_condition(7);
        parent.note_assumption("a1");
        parent.note_assumption("a2");
        parent.emit_scope("s1");

        let diagnostics =
            parent.diagnostics_with_records(|assumed| vec![format!("records={}", assumed.len())]);

        assert_eq!(diagnostics.required, vec![7]);
        assert_eq!(diagnostics.assumed, vec!["a1", "a2"]);
        assert_eq!(diagnostics.output_scopes, vec!["s1"]);
        assert_eq!(diagnostics.assumed_records, vec!["records=2".to_string()]);
    }
}
