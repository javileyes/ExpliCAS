use super::parse::parse_profile_cache_command_input;
use super::store::ProfileCacheStore;
use super::types::{ProfileCacheCommandInput, ProfileCacheCommandResult};

/// Apply a `cache` command line to an engine profile cache.
pub fn apply_profile_cache_command<E: ProfileCacheStore>(
    engine: &mut E,
    line: &str,
) -> ProfileCacheCommandResult {
    match parse_profile_cache_command_input(line) {
        ProfileCacheCommandInput::Status => ProfileCacheCommandResult::Status {
            cached_profiles: engine.profile_cache_len(),
        },
        ProfileCacheCommandInput::Clear => {
            engine.clear_profile_cache();
            ProfileCacheCommandResult::Cleared
        }
        ProfileCacheCommandInput::Unknown(command) => {
            ProfileCacheCommandResult::Unknown { command }
        }
    }
}
