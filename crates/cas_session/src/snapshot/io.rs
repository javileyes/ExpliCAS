use std::path::Path;

use super::{SessionSnapshot, SnapshotError};
use crate::cache::SimplifyCacheKey;

#[cfg(test)]
use super::session_store_snapshot_into_store;

impl SessionSnapshot {
    pub const MAGIC: [u8; 8] = *b"EXPLICAS";
    pub const VERSION: u32 = 2;

    pub fn new(
        context: &cas_ast::Context,
        session: &crate::SessionStore,
        env: &crate::env::Environment,
        cache_key: SimplifyCacheKey,
    ) -> Self {
        Self {
            header: super::SessionSnapshotHeader::new(Self::MAGIC, Self::VERSION, cache_key),
            context: super::ContextSnapshot::from_context(context),
            session: super::session_store_snapshot_from_store(session),
            environment: super::EnvironmentSnapshot::from_env(env),
        }
    }

    #[cfg(test)]
    pub fn is_compatible(&self, key: &SimplifyCacheKey) -> bool {
        self.header.is_valid_with(Self::MAGIC, Self::VERSION) && &self.header.cache_key == key
    }

    #[cfg(test)]
    pub fn load(path: &Path) -> Result<Self, SnapshotError> {
        cas_session_core::snapshot_io::load_bincode(path)
    }

    /// Atomic save: write to temp file then rename.
    pub fn save_atomic(&self, path: &Path) -> Result<(), SnapshotError> {
        cas_session_core::snapshot_io::save_bincode_atomic(self, path)
    }

    #[cfg(test)]
    pub fn into_parts_with_env(
        self,
    ) -> (
        cas_ast::Context,
        crate::SessionStore,
        crate::env::Environment,
    ) {
        (
            self.context.into_context(),
            session_store_snapshot_into_store(self.session),
            self.environment.into_env(),
        )
    }
}
