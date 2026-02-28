//! Session Snapshot Persistence (V2.15.36)
//!
//! Enables persistent sessions across CLI invocations via binary snapshots.
//! The snapshot contains the complete Context (arena) and SessionStore,
//! allowing `#N` references and cached results to survive process restarts.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::{EntryKind, SessionStore, SimplifiedCache, SimplifyCacheKey};
pub use cas_session_core::context_snapshot::ContextSnapshot;
pub use cas_session_core::snapshot_error::SnapshotError;
use cas_session_core::snapshot_header::SnapshotHeader;
use cas_session_core::store_snapshot::{
    EntryKindSnapshot, SessionStoreSnapshot as CoreSessionStoreSnapshot, SimplifiedCacheSnapshot,
};
use cas_session_core::types::CacheConfig;

type SessionSnapshotHeader = SnapshotHeader<SimplifyCacheKey>;
pub type SessionStoreSnapshot = CoreSessionStoreSnapshot<SimplifyCacheKey>;

/// Complete session state for persistence.
/// Contains header for compatibility checking, plus Context and SessionStore.
#[derive(Debug, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub header: SessionSnapshotHeader,
    pub context: ContextSnapshot,
    pub session: SessionStoreSnapshot,
}

// ============================================================================
// Conversion: SessionStore <-> SessionStoreSnapshot
// ============================================================================

fn session_store_snapshot_from_store(store: &SessionStore) -> SessionStoreSnapshot {
    cas_session_core::store_snapshot::snapshot_from_store_with(
        store,
        |kind| match kind {
            EntryKind::Expr(id) => EntryKindSnapshot::Expr(id.index() as u32),
            EntryKind::Eq { lhs, rhs } => EntryKindSnapshot::Eq {
                lhs: lhs.index() as u32,
                rhs: rhs.index() as u32,
            },
        },
        |cache| {
            Some(SimplifiedCacheSnapshot {
                key: cache.key.clone(),
                expr: cache.expr.index() as u32,
            })
        },
    )
}

fn session_store_snapshot_into_store(snapshot: SessionStoreSnapshot) -> SessionStore {
    use cas_ast::ExprId;

    cas_session_core::store_snapshot::restore_store_from_snapshot_with(
        snapshot,
        |kind| match kind {
            EntryKindSnapshot::Expr(id) => EntryKind::Expr(ExprId::from_raw(id)),
            EntryKindSnapshot::Eq { lhs, rhs } => EntryKind::Eq {
                lhs: ExprId::from_raw(lhs),
                rhs: ExprId::from_raw(rhs),
            },
        },
        |cache| SimplifiedCache {
            key: cache.key,
            expr: ExprId::from_raw(cache.expr),
            requires: Vec::new(), // Recalculated on use if needed
            steps: None,          // Light cache: no steps persisted
        },
        |config: CacheConfig| crate::session_store_with_cache_config(config),
    )
}

// ============================================================================
// SessionSnapshot: Main API
// ============================================================================

impl SessionSnapshot {
    pub const MAGIC: [u8; 8] = *b"EXPLICAS";
    pub const VERSION: u32 = 1;

    pub fn new(
        context: &cas_ast::Context,
        session: &SessionStore,
        cache_key: SimplifyCacheKey,
    ) -> Self {
        Self {
            header: SessionSnapshotHeader::new(Self::MAGIC, Self::VERSION, cache_key),
            context: ContextSnapshot::from_context(context),
            session: session_store_snapshot_from_store(session),
        }
    }

    pub fn is_compatible(&self, key: &SimplifyCacheKey) -> bool {
        self.header.is_valid_with(Self::MAGIC, Self::VERSION) && &self.header.cache_key == key
    }

    pub fn load(path: &Path) -> Result<Self, SnapshotError> {
        cas_session_core::snapshot_io::load_bincode(path)
    }

    /// Atomic save: write to temp file then rename.
    pub fn save_atomic(&self, path: &Path) -> Result<(), SnapshotError> {
        cas_session_core::snapshot_io::save_bincode_atomic(self, path)
    }

    /// Extract Context and SessionStore from snapshot.
    pub fn into_parts(self) -> (cas_ast::Context, SessionStore) {
        (
            self.context.into_context(),
            session_store_snapshot_into_store(self.session),
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_session_snapshot_save_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.session");

        // Create a context with some expressions
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let expr = ctx.add(cas_ast::Expr::Add(x, one));

        // Create a session store with an entry
        let mut store = SessionStore::new();
        store.push(crate::EntryKind::Expr(expr), "x + 1".to_string());

        let key = SimplifyCacheKey {
            domain: crate::CacheDomainMode::Generic,
            ruleset_rev: 1,
        };

        // Save
        let snapshot = SessionSnapshot::new(&ctx, &store, key.clone());
        snapshot.save_atomic(&path).unwrap();

        // Load
        let loaded = SessionSnapshot::load(&path).unwrap();
        assert!(loaded.is_compatible(&key));

        // Verify
        let (restored_ctx, restored_store) = loaded.into_parts();
        assert_eq!(ctx.nodes.len(), restored_ctx.nodes.len());
        assert_eq!(store.len(), restored_store.len());
    }
}
