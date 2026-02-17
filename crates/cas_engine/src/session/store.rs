//! Session store engine-specific type aliases.

pub use cas_session_core::store::{Entry as CoreEntry, SessionStore as CoreStore};
pub use cas_session_core::types::{CacheConfig, EntryId, EntryKind, RefMode};

/// Engine-specialized cache key.
pub type SimplifyCacheKey = cas_session_core::cache::SimplifyCacheKey<crate::domain::DomainMode>;

/// Engine-specialized cached simplification payload.
pub type SimplifiedCache = cas_session_core::cache::SimplifiedCache<
    crate::domain::DomainMode,
    crate::diagnostics::RequiredItem,
    crate::step::Step,
>;

/// Engine-specialized cache hit trace.
pub type CacheHitTrace = cas_session_core::cache::CacheHitTrace<crate::diagnostics::RequiredItem>;

/// Engine-specialized resolved expression payload.
pub type ResolvedExpr = cas_session_core::cache::ResolvedExpr<crate::diagnostics::RequiredItem>;

/// Engine specialization of a session entry.
pub type Entry = CoreEntry<crate::diagnostics::Diagnostics, SimplifiedCache>;

/// Engine specialization of the generic session store.
pub type SessionStore = CoreStore<crate::diagnostics::Diagnostics, SimplifiedCache>;
