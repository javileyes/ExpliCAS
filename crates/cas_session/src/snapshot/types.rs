use serde::{Deserialize, Serialize};

use crate::snapshot_store_convert::SessionStoreSnapshot as StoreSnapshotCore;
use crate::SimplifyCacheKey;
use cas_session_core::snapshot_header::SnapshotHeader;

pub(super) type SessionSnapshotHeader = SnapshotHeader<SimplifyCacheKey>;
pub type SessionStoreSnapshot = StoreSnapshotCore;

/// Complete session state for persistence.
/// Contains header for compatibility checking, plus Context and SessionStore.
#[derive(Debug, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub header: SessionSnapshotHeader,
    pub context: super::ContextSnapshot,
    pub session: SessionStoreSnapshot,
}
