//! Session-related components extracted from `cas_engine`.

pub mod env;

pub use cas_engine::session::{
    resolve_session_refs, resolve_session_refs_with_diagnostics, resolve_session_refs_with_mode,
    CacheHitTrace, Entry, ResolvedExpr, SessionStore, SimplifiedCache, SimplifyCacheKey,
};
pub use cas_engine::session_snapshot::{SessionSnapshot, SnapshotError};
pub use cas_engine::session_state::SessionState;
pub use cas_session_core::types::{CacheConfig, EntryId, EntryKind, RefMode, ResolveError};
pub use env::{is_reserved, substitute, substitute_with_shadow, Environment};
