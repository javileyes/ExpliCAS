/// Result of applying a `cache` command against engine profile cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileCacheCommandResult {
    Status { cached_profiles: usize },
    Cleared,
    Unknown { command: String },
}
