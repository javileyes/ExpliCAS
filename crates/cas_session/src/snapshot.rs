//! Session Snapshot Persistence (V2.15.36)
//!
//! Enables persistent sessions across CLI invocations via binary snapshots.
//! The snapshot contains the complete Context (arena) and SessionStore,
//! allowing `#N` references and cached results to survive process restarts.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::snapshot_store_convert::{
    session_store_snapshot_from_store, session_store_snapshot_into_store,
    SessionStoreSnapshot as StoreSnapshotCore,
};
use crate::SimplifyCacheKey;
pub use cas_session_core::context_snapshot::ContextSnapshot;
pub use cas_session_core::snapshot_error::SnapshotError;
use cas_session_core::snapshot_header::SnapshotHeader;

type SessionSnapshotHeader = SnapshotHeader<SimplifyCacheKey>;
pub type SessionStoreSnapshot = StoreSnapshotCore;

/// Complete session state for persistence.
/// Contains header for compatibility checking, plus Context and SessionStore.
#[derive(Debug, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub header: SessionSnapshotHeader,
    pub context: ContextSnapshot,
    pub session: SessionStoreSnapshot,
}

// ============================================================================
// SessionSnapshot: Main API
// ============================================================================

impl SessionSnapshot {
    pub const MAGIC: [u8; 8] = *b"EXPLICAS";
    pub const VERSION: u32 = 1;

    pub fn new(
        context: &cas_ast::Context,
        session: &crate::SessionStore,
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
    pub fn into_parts(self) -> (cas_ast::Context, crate::SessionStore) {
        (
            self.context.into_context(),
            session_store_snapshot_into_store(self.session),
        )
    }
}
