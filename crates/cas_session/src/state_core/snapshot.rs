use std::path::Path;

use super::SessionState;
use crate::snapshot::SnapshotError;
use crate::{snapshot::SessionSnapshot, SimplifyCacheKey};

impl SessionState {
    /// Restore context + state from a persisted snapshot.
    pub(crate) fn from_snapshot(snapshot: SessionSnapshot) -> (cas_ast::Context, Self) {
        let (context, store) = snapshot.into_parts();
        (context, Self::from_store(store))
    }

    /// Load a snapshot from disk and restore it only if compatible with `cache_key`.
    pub fn load_compatible_snapshot(
        path: &Path,
        cache_key: &SimplifyCacheKey,
    ) -> Result<Option<(cas_ast::Context, Self)>, SnapshotError> {
        let snapshot = SessionSnapshot::load(path)?;
        if !snapshot.is_compatible(cache_key) {
            return Ok(None);
        }
        Ok(Some(Self::from_snapshot(snapshot)))
    }
}
