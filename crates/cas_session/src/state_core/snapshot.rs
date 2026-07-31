use std::io::Read;
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
        Self::read_compatible_snapshot(&mut reader, cache_key)
    }

    /// In-memory twin of [`Self::load_compatible_snapshot`] for hosts without
    /// a filesystem: the wasm worker snapshots the session before each
    /// evaluation so a user STOP (which tears the whole worker down) can
    /// rebuild engine context, `#N` store and `:=` environment from bytes.
    /// `domain_flag` must match the one the snapshot was encoded with.
    pub fn decode_compatible_snapshot_bytes(
        bytes: &[u8],
        domain_flag: &str,
    ) -> Result<Option<(cas_ast::Context, Self)>, SnapshotError> {
        let key = SimplifyCacheKey::from_domain_flag(domain_flag);
        Self::read_compatible_snapshot(&mut &bytes[..], &key)
    }

    /// Shared decode body: one bincode payload per snapshot section, in
    /// declaration order (header gates the rest).
    fn read_compatible_snapshot<R: Read>(
        reader: &mut R,
        cache_key: &SimplifyCacheKey,
    ) -> Result<Option<(cas_ast::Context, Self)>, SnapshotError> {
        let header: SnapshotHeader<SimplifyCacheKey> = load_bincode_from_reader(reader)?;
        if !header.is_valid_with(SessionSnapshot::MAGIC, SessionSnapshot::VERSION)
            || !header.cache_key.is_compatible(cache_key)
        {
            return Ok(None);
        }

        let context = load_bincode_from_reader::<_, ContextSnapshot>(reader)?.into_context();
        let session = load_bincode_from_reader::<_, SessionStoreSnapshot>(reader)?;
        let env = load_bincode_from_reader::<_, crate::environment_snapshot::EnvironmentSnapshot>(
            reader,
        )?
        .into_env();
        Ok(Some((
            context,
            Self::from_store_and_env(session_store_snapshot_into_store(session), env),
        )))
    }
}
