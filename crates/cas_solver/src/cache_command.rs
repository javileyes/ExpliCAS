use crate::CacheCommandInput;

/// Result of applying a `cache` command against engine profile cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileCacheCommandResult {
    Status { cached_profiles: usize },
    Cleared,
    Unknown { command: String },
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

/// Apply a `cache` command line to engine profile cache.
pub fn apply_profile_cache_command(
    engine: &mut cas_engine::Engine,
    line: &str,
) -> ProfileCacheCommandResult {
    match crate::parse_cache_command_input(line) {
        CacheCommandInput::Status => ProfileCacheCommandResult::Status {
            cached_profiles: engine.profile_cache_len(),
        },
        CacheCommandInput::Clear => {
            engine.clear_profile_cache();
            ProfileCacheCommandResult::Cleared
        }
        CacheCommandInput::Unknown(command) => ProfileCacheCommandResult::Unknown { command },
    }
}

#[cfg(test)]
mod tests {
    use super::{
        apply_profile_cache_command, format_profile_cache_command_lines, ProfileCacheCommandResult,
    };

    #[test]
    fn apply_profile_cache_command_status_defaults_to_zero() {
        let mut engine = cas_engine::Engine::new();
        let result = apply_profile_cache_command(&mut engine, "cache status");
        assert_eq!(
            result,
            ProfileCacheCommandResult::Status { cached_profiles: 0 }
        );
    }

    #[test]
    fn apply_profile_cache_command_clear_empties_cache() {
        let mut engine = cas_engine::Engine::new();

        let parsed = cas_parser::parse("x + x", &mut engine.simplifier.context).expect("parse");
        let req = cas_engine::EvalRequest {
            raw_input: "x + x".to_string(),
            parsed,
            action: cas_engine::EvalAction::Simplify,
            auto_store: false,
        };
        let _ = engine
            .eval_stateless(cas_engine::EvalOptions::default(), req)
            .expect("eval");
        assert_eq!(engine.profile_cache_len(), 1);

        let result = apply_profile_cache_command(&mut engine, "cache clear");
        assert_eq!(result, ProfileCacheCommandResult::Cleared);
        assert_eq!(engine.profile_cache_len(), 0);
    }

    #[test]
    fn apply_profile_cache_command_reports_unknown_subcommand() {
        let mut engine = cas_engine::Engine::new();
        let result = apply_profile_cache_command(&mut engine, "cache nope");
        assert_eq!(
            result,
            ProfileCacheCommandResult::Unknown {
                command: "nope".to_string(),
            }
        );
    }

    #[test]
    fn format_profile_cache_command_lines_status_includes_hint() {
        let lines = format_profile_cache_command_lines(&ProfileCacheCommandResult::Status {
            cached_profiles: 0,
        });
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("Profile Cache: 0 profiles cached"));
        assert!(lines[0].contains("empty - profiles will be built on first eval"));
    }

    #[test]
    fn format_profile_cache_command_lines_unknown_returns_usage_line() {
        let lines = format_profile_cache_command_lines(&ProfileCacheCommandResult::Unknown {
            command: "nope".to_string(),
        });
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "Unknown cache command: nope");
        assert_eq!(lines[1], "Usage: cache [status|clear]");
    }
}
