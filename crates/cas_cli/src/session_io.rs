//! Shared session snapshot I/O helpers for CLI entrypoints.

use std::path::Path;

use cas_session::{SessionState, SimplifyCacheKey, SnapshotError};
use cas_solver::Engine;

/// Load a compatible session snapshot if available, otherwise create a fresh engine/state.
///
/// Returns `(engine, state, warning_message)` where `warning_message` is present when
/// a snapshot existed but could not be reused.
pub fn load_or_new_session(
    path: Option<&Path>,
    key: &SimplifyCacheKey,
) -> (Engine, SessionState, Option<String>) {
    let Some(path) = path else {
        return (Engine::new(), SessionState::new(), None);
    };

    if !path.exists() {
        return (Engine::new(), SessionState::new(), None);
    }

    match SessionState::load_compatible_snapshot(path, key) {
        Ok(Some((ctx, state))) => (Engine::with_context(ctx), state, None),
        Ok(None) => (
            Engine::new(),
            SessionState::new(),
            Some("Session snapshot incompatible, starting fresh".to_string()),
        ),
        Err(e) => (
            Engine::new(),
            SessionState::new(),
            Some(format!(
                "Warning: Failed to load session ({}), starting fresh",
                e
            )),
        ),
    }
}

/// Save session snapshot to disk.
pub fn save_session(
    engine: &Engine,
    state: &SessionState,
    path: &Path,
    key: &SimplifyCacheKey,
) -> Result<(), SnapshotError> {
    state.save_snapshot(&engine.simplifier.context, path, key.clone())
}
