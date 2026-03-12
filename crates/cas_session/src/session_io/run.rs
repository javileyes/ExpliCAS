use std::path::Path;

use cas_engine::Engine;

use super::{load_or_new_session, save_session};
use crate::{state_core::SessionState, SimplifyCacheKey};

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
        save_session(&engine, &mut state, snapshot_path, key)
            .err()
            .map(|error| format!("Warning: Failed to save session: {}", error))
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
