use crate::{EntryKind, SessionStore, SimplifiedCache, SimplifyCacheKey};
use cas_session_core::store_snapshot::{
    EntryKindSnapshot, SessionStoreSnapshot as CoreSessionStoreSnapshot, SimplifiedCacheSnapshot,
};
use cas_session_core::types::CacheConfig;

pub(crate) type SessionStoreSnapshot = CoreSessionStoreSnapshot<SimplifyCacheKey>;

pub(crate) fn session_store_snapshot_from_store(store: &SessionStore) -> SessionStoreSnapshot {
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

pub(crate) fn session_store_snapshot_into_store(snapshot: SessionStoreSnapshot) -> SessionStore {
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
        |config: CacheConfig| crate::store_cache_policy::session_store_with_cache_config(config),
    )
}
