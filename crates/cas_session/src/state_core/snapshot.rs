use std::path::Path;

use cas_session_core::snapshot_header::SnapshotHeader;
use cas_session_core::snapshot_io::{load_bincode_from_reader, open_bincode_reader};

use super::SessionState;
use crate::cache::SimplifyCacheKey;
use crate::snapshot::{
    session_store_snapshot_into_store, ContextSnapshot, SessionSnapshot, SessionStoreSnapshot,
    SnapshotError,
};

impl SessionState {
    /// Load a snapshot from disk and restore it only if compatible with `cache_key`.
    pub fn load_compatible_snapshot(
        path: &Path,
        cache_key: &SimplifyCacheKey,
    ) -> Result<Option<(cas_ast::Context, Self)>, SnapshotError> {
        let mut reader = open_bincode_reader(path)?;
        let header: SnapshotHeader<SimplifyCacheKey> = load_bincode_from_reader(&mut reader)?;
        if !header.is_valid_with(SessionSnapshot::MAGIC, SessionSnapshot::VERSION)
            || !header.cache_key.is_compatible(cache_key)
        {
            return Ok(None);
        }

        let context = load_bincode_from_reader::<_, ContextSnapshot>(&mut reader)?.into_context();
        let session = load_bincode_from_reader::<_, SessionStoreSnapshot>(&mut reader)?;
        Ok(Some((
            context,
            Self::from_store(session_store_snapshot_into_store(session)),
        )))
    }
}
