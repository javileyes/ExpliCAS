/// Result of applying a `cache` command against engine profile cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileCacheCommandResult {
    Status { cached_profiles: usize },
    Cleared,
    Unknown { command: String },
}

enum ProfileCacheCommandInput {
    Status,
    Clear,
    Unknown(String),
}

/// Minimal profile-cache surface needed by `cache` REPL commands.
pub trait ProfileCacheStore {
    fn profile_cache_len(&self) -> usize;
    fn clear_profile_cache(&mut self);
}

impl ProfileCacheStore for crate::Engine {
    fn profile_cache_len(&self) -> usize {
        crate::Engine::profile_cache_len(self)
    }

    fn clear_profile_cache(&mut self) {
        crate::Engine::clear_profile_cache(self);
    }
}

fn parse_profile_cache_command_input(line: &str) -> ProfileCacheCommandInput {
    let args: Vec<&str> = line.split_whitespace().collect();
    match args.get(1).copied() {
        None | Some("status") => ProfileCacheCommandInput::Status,
        Some("clear") => ProfileCacheCommandInput::Clear,
        Some(other) => ProfileCacheCommandInput::Unknown(other.to_string()),
    }
}

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

/// Evaluate a `cache` command and return user-facing output lines.
pub fn evaluate_profile_cache_command_lines<E: ProfileCacheStore>(
    engine: &mut E,
    line: &str,
) -> Vec<String> {
    let result = apply_profile_cache_command(engine, line);
    format_profile_cache_command_lines(&result)
}
