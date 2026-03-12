use std::path::Path;

use cas_engine::Engine;

use crate::snapshot::SnapshotError;
use crate::{state_core::SessionState, SimplifyCacheKey};

/// Save session snapshot to disk.
pub fn save_session(
    engine: &Engine,
    state: &mut SessionState,
    path: &Path,
    key: &SimplifyCacheKey,
) -> Result<(), SnapshotError> {
    if !state.is_dirty() {
        return Ok(());
    }
    state.save_snapshot(&engine.simplifier.context, path, key.clone())?;
    state.mark_clean();
    Ok(())
}
