//! Session Snapshot Persistence (V2.15.36)
//!
//! Enables persistent sessions across CLI invocations via binary snapshots.
//! The snapshot contains the complete Context (arena) and SessionStore,
//! allowing `#N` references and cached results to survive process restarts.

mod io;

use serde::{Deserialize, Serialize};

use crate::cache::SimplifyCacheKey;
use crate::snapshot_store_convert::SessionStoreSnapshot as StoreSnapshotCore;
pub use cas_session_core::context_snapshot::ContextSnapshot;
pub use cas_session_core::snapshot_error::SnapshotError;
use cas_session_core::snapshot_header::SnapshotHeader;

pub(super) type SessionSnapshotHeader = SnapshotHeader<SimplifyCacheKey>;
pub type SessionStoreSnapshot = StoreSnapshotCore;

/// Complete session state for persistence.
/// Contains header for compatibility checking, plus Context and SessionStore.
#[derive(Debug, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub header: SessionSnapshotHeader,
    pub context: ContextSnapshot,
    pub session: SessionStoreSnapshot,
}
