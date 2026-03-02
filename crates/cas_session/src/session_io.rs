//! Shared session snapshot I/O helpers for CLI entrypoints.

use std::path::Path;

use cas_solver::Engine;

use crate::{SessionState, SimplifyCacheKey, SnapshotError};

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

/// Load session state, run an operation, and persist snapshot when a path is provided.
///
/// Returns `(result, load_warning, save_warning)`.
pub fn run_with_session<R, F>(
    path: Option<&Path>,
    key: &SimplifyCacheKey,
    run: F,
) -> (R, Option<String>, Option<String>)
where
    F: FnOnce(&mut Engine, &mut SessionState) -> R,
{
    let (mut engine, mut state, load_warning) = load_or_new_session(path, key);
    let result = run(&mut engine, &mut state);
    let save_warning = path.and_then(|snapshot_path| {
        save_session(&engine, &state, snapshot_path, key)
            .err()
            .map(|e| format!("Warning: Failed to save session: {}", e))
    });
    (result, load_warning, save_warning)
}

/// Variant of [`run_with_session`] that builds cache key from domain flag.
pub fn run_with_domain_session<R, F>(
    path: Option<&Path>,
    domain: &str,
    run: F,
) -> (R, Option<String>, Option<String>)
where
    F: FnOnce(&mut Engine, &mut SessionState) -> R,
{
    let key = SimplifyCacheKey::from_domain_flag(domain);
    run_with_session(path, &key, run)
}
