use crate::snapshot::SnapshotError;
use crate::{snapshot::SessionSnapshot, state_core::SessionState, SimplifyCacheKey};

impl SessionState {
    /// Build a serializable snapshot from the current state.
    fn snapshot(&self, context: &cas_ast::Context, cache_key: SimplifyCacheKey) -> SessionSnapshot {
        SessionSnapshot::new(context, &self.store, cache_key)
    }

    /// Persist the current state atomically to disk.
    pub fn save_snapshot(
        &self,
        context: &cas_ast::Context,
        path: &std::path::Path,
        cache_key: SimplifyCacheKey,
    ) -> Result<(), SnapshotError> {
        self.snapshot(context, cache_key).save_atomic(path)
    }
}
