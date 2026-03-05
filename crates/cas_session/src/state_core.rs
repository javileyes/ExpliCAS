use cas_solver::EvalOptions;

use crate::{
    snapshot::SessionSnapshot, state_eval_store::SessionEvalStore, Environment, SessionStore,
    SimplifyCacheKey, SnapshotError,
};

/// Bundled session state for portability (CLI/Web/FFI).
///
/// This crate-local type is the migration target for Phase 3.
/// `cas_engine` remains stateless and consumes it via the shared `EvalSession` trait.
#[derive(Default, Debug)]
pub struct SessionState {
    pub(crate) store: SessionEvalStore,
    pub(crate) env: Environment,
    pub(crate) options: EvalOptions,
}

impl SessionState {
    pub fn assumptions(&self) -> &EvalOptions {
        &self.options
    }

    pub fn options(&self) -> &EvalOptions {
        &self.options
    }

    pub fn options_mut(&mut self) -> &mut EvalOptions {
        &mut self.options
    }

    pub fn new() -> Self {
        Self {
            store: SessionEvalStore::new(),
            env: Environment::new(),
            options: EvalOptions::default(),
        }
    }

    /// Create a state from an existing session store (for snapshot restoration).
    fn from_store(store: SessionStore) -> Self {
        Self {
            store: SessionEvalStore::from_store(store),
            env: Environment::new(),
            options: EvalOptions::default(),
        }
    }

    /// Restore context + state from a persisted snapshot.
    fn from_snapshot(snapshot: SessionSnapshot) -> (cas_ast::Context, Self) {
        let (context, store) = snapshot.into_parts();
        (context, Self::from_store(store))
    }

    /// Load a snapshot from disk and restore it only if compatible with `cache_key`.
    pub fn load_compatible_snapshot(
        path: &std::path::Path,
        cache_key: &SimplifyCacheKey,
    ) -> Result<Option<(cas_ast::Context, Self)>, SnapshotError> {
        let snapshot = SessionSnapshot::load(path)?;
        if !snapshot.is_compatible(cache_key) {
            return Ok(None);
        }
        Ok(Some(Self::from_snapshot(snapshot)))
    }
}

impl cas_solver::SolveBudgetContext for SessionState {
    fn solve_budget_max_branches(&self) -> usize {
        self.options().budget.max_branches
    }

    fn set_solve_budget_max_branches(&mut self, max_branches: usize) {
        self.options_mut().budget.max_branches = max_branches;
    }
}
