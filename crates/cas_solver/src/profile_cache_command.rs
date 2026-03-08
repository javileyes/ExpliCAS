mod apply;
mod format;
mod parse;
mod store;
mod types;

pub use apply::apply_profile_cache_command;
pub use format::format_profile_cache_command_lines;
pub use store::ProfileCacheStore;
pub use types::ProfileCacheCommandResult;

/// Evaluate a `cache` command and return user-facing output lines.
pub fn evaluate_profile_cache_command_lines<E: ProfileCacheStore>(
    engine: &mut E,
    line: &str,
) -> Vec<String> {
    let result = apply_profile_cache_command(engine, line);
    format_profile_cache_command_lines(&result)
}
