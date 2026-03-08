use cas_ast::ExprId;

/// Unique identifier for a session entry.
pub type EntryId = u64;

/// Type of entry stored in the session.
#[derive(Debug, Clone)]
pub enum EntryKind {
    /// A single expression.
    Expr(ExprId),
    /// An equation (lhs = rhs).
    Eq { lhs: ExprId, rhs: ExprId },
}

/// How to resolve session references.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RefMode {
    /// Use cached simplified result if available and valid (default, fast).
    #[default]
    PreferSimplified,
    /// Use original parsed expression (for debugging, "raw" command).
    Raw,
}

/// Configuration for simplified cache memory limits.
///
/// Controls how many cached simplified results are retained to
/// prevent unbounded memory growth in long sessions.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Max entries with cached simplified result (0 = unlimited).
    pub max_cached_entries: usize,
    /// Max total steps across all cached entries (0 = unlimited).
    pub max_cached_steps: usize,
    /// Drop steps for entries with > N steps (light cache mode).
    pub light_cache_threshold: Option<usize>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_cached_entries: 100,          // Reasonable default
            max_cached_steps: 5000,           // ~50 steps avg per entry
            light_cache_threshold: Some(200), // Drop steps if > 200
        }
    }
}

/// Error during session reference resolution.
#[derive(Debug, Clone, PartialEq)]
pub enum ResolveError {
    /// Reference to non-existent entry.
    NotFound(EntryId),
    /// Circular reference detected (e.g., #3 contains #3, or #3 -> #4 -> #3).
    CircularReference(EntryId),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::NotFound(id) => write!(f, "Session reference #{} not found", id),
            ResolveError::CircularReference(id) => {
                write!(f, "Circular reference detected involving #{}", id)
            }
        }
    }
}

impl std::error::Error for ResolveError {}
