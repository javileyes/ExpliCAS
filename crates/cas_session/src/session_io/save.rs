use std::path::Path;

use cas_engine::Engine;

use crate::snapshot::SnapshotError;
use crate::{SessionState, SimplifyCacheKey};

/// Save session snapshot to disk.
pub fn save_session(
    engine: &Engine,
    state: &SessionState,
    path: &Path,
    key: &SimplifyCacheKey,
) -> Result<(), SnapshotError> {
    state.save_snapshot(&engine.simplifier.context, path, key.clone())
}
