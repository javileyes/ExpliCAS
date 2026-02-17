//! Session-related components extracted from `cas_engine`.

pub mod env;

pub use cas_engine::session::{
    resolve_session_refs, resolve_session_refs_with_diagnostics, resolve_session_refs_with_mode,
    CacheConfig, CacheHitTrace, Entry, EntryId, EntryKind, RefMode, ResolveError, ResolvedExpr,
    SessionStore, SimplifiedCache, SimplifyCacheKey,
};
pub use cas_engine::session_snapshot::{SessionSnapshot, SnapshotError};
pub use cas_engine::session_state::SessionState;
pub use env::{is_reserved, substitute, substitute_with_shadow, Environment};
