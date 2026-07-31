use crate::snapshot::SnapshotError;
use crate::{cache::SimplifyCacheKey, snapshot::SessionSnapshot, state_core::SessionState};

impl SessionState {
    /// Build a serializable snapshot from the current state.
    fn snapshot(&self, context: &cas_ast::Context, cache_key: SimplifyCacheKey) -> SessionSnapshot {
        SessionSnapshot::new(context, &self.store, &self.env, cache_key)
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

    /// In-memory twin of [`Self::save_snapshot`]: the full session (Context
    /// arena, `#N` store, `:=` environment) as one owned byte buffer.
    /// Decoded by `decode_compatible_snapshot_bytes` with the same
    /// `domain_flag`; used by the wasm worker's stop/restore cycle, where
    /// there is no filesystem.
    pub fn encode_snapshot_bytes(
        &self,
        context: &cas_ast::Context,
        domain_flag: &str,
    ) -> Result<Vec<u8>, SnapshotError> {
        let key = crate::cache::SimplifyCacheKey::from_domain_flag(domain_flag);
        cas_session_core::snapshot_io::encode_bincode(&self.snapshot(context, key))
    }
}
