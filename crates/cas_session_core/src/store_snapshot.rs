//! Serializable store snapshot DTOs shared by session runtimes.

use serde::{Deserialize, Serialize};

/// Serializable session-store representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStoreSnapshot<CacheKey> {
    pub next_id: u64,
    pub entries: Vec<EntrySnapshot<CacheKey>>,
    pub cache_order: Vec<u64>,
    pub cache_config: CacheConfigSnapshot,
    pub cached_steps_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntrySnapshot<CacheKey> {
    pub id: u64,
    pub raw_text: String,
    pub kind: EntryKindSnapshot,
    pub simplified: Option<SimplifiedCacheSnapshot<CacheKey>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntryKindSnapshot {
    Expr(u32), // ExprId index
    Eq { lhs: u32, rhs: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedCacheSnapshot<CacheKey> {
    pub key: CacheKey,
    pub expr: u32, // ExprId index
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfigSnapshot {
    pub max_cached_entries: usize,
    pub max_cached_steps: usize,
    pub light_cache_threshold: Option<usize>,
}

/// Build a serializable snapshot from one session store using caller-provided
/// kind/cache mappers.
pub fn snapshot_from_store_with<D, CacheValue, CacheKey, FKind, FCache>(
    store: &crate::store::SessionStore<D, CacheValue>,
    mut map_kind: FKind,
    mut map_cache: FCache,
) -> SessionStoreSnapshot<CacheKey>
where
    D: Default,
    FKind: FnMut(&crate::types::EntryKind) -> EntryKindSnapshot,
    FCache: FnMut(&CacheValue) -> Option<SimplifiedCacheSnapshot<CacheKey>>,
{
    let entries = store
        .entries()
        .map(|entry| EntrySnapshot {
            id: entry.id,
            raw_text: entry.raw_text.clone(),
            kind: map_kind(&entry.kind),
            simplified: entry.simplified.as_ref().and_then(&mut map_cache),
        })
        .collect();

    let (_cached_entries, cached_steps_count) = store.cache_stats();

    SessionStoreSnapshot {
        next_id: store.next_id(),
        entries,
        cache_order: store.cache_order().iter().copied().collect(),
        cache_config: CacheConfigSnapshot {
            max_cached_entries: store.cache_config().max_cached_entries,
            max_cached_steps: store.cache_config().max_cached_steps,
            light_cache_threshold: store.cache_config().light_cache_threshold,
        },
        cached_steps_count,
    }
}

/// Restore a session store from snapshot DTOs using caller-provided
/// kind/cache mappers and store constructor.
pub fn restore_store_from_snapshot_with<D, CacheValue, CacheKey, FKind, FCache, FStore>(
    snapshot: SessionStoreSnapshot<CacheKey>,
    mut map_kind: FKind,
    mut map_cache: FCache,
    build_store: FStore,
) -> crate::store::SessionStore<D, CacheValue>
where
    D: Default,
    FKind: FnMut(EntryKindSnapshot) -> crate::types::EntryKind,
    FCache: FnMut(SimplifiedCacheSnapshot<CacheKey>) -> CacheValue,
    FStore: FnOnce(crate::types::CacheConfig) -> crate::store::SessionStore<D, CacheValue>,
{
    let config = crate::types::CacheConfig {
        max_cached_entries: snapshot.cache_config.max_cached_entries,
        max_cached_steps: snapshot.cache_config.max_cached_steps,
        light_cache_threshold: snapshot.cache_config.light_cache_threshold,
    };
    let entry_capacity = snapshot.entries.len();
    let cache_order_capacity = snapshot.cache_order.len();

    let mut store = build_store(config);
    store.reserve_for_restore(entry_capacity, cache_order_capacity);

    for entry in snapshot.entries {
        let restored = crate::store::Entry {
            id: entry.id,
            kind: map_kind(entry.kind),
            raw_text: entry.raw_text,
            diagnostics: D::default(),
            simplified: entry.simplified.map(&mut map_cache),
        };
        store.restore_entry(restored);
    }

    store.restore_cache_order(snapshot.cache_order);
    store
}
