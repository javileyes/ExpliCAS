use super::types::ProfileCacheCommandResult;

/// Render a `cache` command result into output lines for UI/frontends.
pub fn format_profile_cache_command_lines(result: &ProfileCacheCommandResult) -> Vec<String> {
    match result {
        ProfileCacheCommandResult::Status {
            cached_profiles: count,
        } => {
            let mut lines = vec![format!("Profile Cache: {} profiles cached", count)];
            if *count == 0 {
                lines.push("  (empty - profiles will be built on first eval)".to_string());
            } else {
                lines.push("  (profiles are reused across evaluations)".to_string());
            }
            vec![lines.join("\n")]
        }
        ProfileCacheCommandResult::Cleared => vec!["Profile cache cleared.".to_string()],
        ProfileCacheCommandResult::Unknown { command } => vec![
            format!("Unknown cache command: {}", command),
            "Usage: cache [status|clear]".to_string(),
        ],
    }
}
