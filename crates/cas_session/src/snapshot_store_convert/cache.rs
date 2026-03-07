use crate::{SessionStore, SimplifiedCache, SimplifyCacheKey};
use cas_session_core::store_snapshot::{
    SessionStoreSnapshot as CoreSessionStoreSnapshot, SimplifiedCacheSnapshot,
};
use cas_session_core::types::CacheConfig;

use super::entry_kind::{restore_entry_kind, snapshot_entry_kind};

pub(crate) type SessionStoreSnapshot = CoreSessionStoreSnapshot<SimplifyCacheKey>;

pub(crate) fn session_store_snapshot_from_store(store: &SessionStore) -> SessionStoreSnapshot {
    cas_session_core::store_snapshot::snapshot_from_store_with(
        store,
        snapshot_entry_kind,
        |cache| {
            Some(SimplifiedCacheSnapshot {
                key: cache.key.clone(),
                expr: cache.expr.index() as u32,
            })
        },
    )
}

pub(crate) fn session_store_snapshot_into_store(snapshot: SessionStoreSnapshot) -> SessionStore {
    use cas_ast::ExprId;

    cas_session_core::store_snapshot::restore_store_from_snapshot_with(
        snapshot,
        restore_entry_kind,
        |cache| SimplifiedCache {
            key: cache.key,
            expr: ExprId::from_raw(cache.expr),
            requires: Vec::new(),
            steps: None,
        },
        |config: CacheConfig| crate::store_cache_policy::session_store_with_cache_config(config),
    )
}
