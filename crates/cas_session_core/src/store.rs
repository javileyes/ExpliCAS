//! Generic session store and entry types.
//!
//! This module is engine-agnostic and can be reused by multiple frontends.

use crate::types::{CacheConfig, EntryId, EntryKind};

fn default_cache_steps_len<C>(_cache: &C) -> usize {
    0
}

fn default_apply_light_cache<C>(cache: C, _light_cache_threshold: Option<usize>) -> C {
    cache
}

/// A stored entry in the session.
#[derive(Debug, Clone)]
pub struct Entry<Diagnostics, CacheValue> {
    /// Unique ID (auto-incrementing, never reused).
    pub id: EntryId,
    /// The stored expression or equation.
    pub kind: EntryKind,
    /// Original raw text input (for display).
    pub raw_text: String,
    /// Entry-level diagnostics (domain requirements, assumptions, etc.).
    pub diagnostics: Diagnostics,
    /// Cached simplified result (if available).
    pub simplified: Option<CacheValue>,
}

impl<D, C> Entry<D, C> {
    /// Check if this entry is an expression.
    pub fn is_expr(&self) -> bool {
        matches!(self.kind, EntryKind::Expr(_))
    }

    /// Check if this entry is an equation.
    pub fn is_eq(&self) -> bool {
        matches!(self.kind, EntryKind::Eq { .. })
    }

    /// Get the type as a string for display.
    pub fn type_str(&self) -> &'static str {
        match self.kind {
            EntryKind::Expr(_) => "Expr",
            EntryKind::Eq { .. } => "Eq",
        }
    }
}

/// Storage for session entries with auto-incrementing IDs.
#[derive(Debug, Clone)]
pub struct SessionStore<Diagnostics, CacheValue> {
    next_id: EntryId,
    entries: Vec<Entry<Diagnostics, CacheValue>>,
    /// LRU tracking for cached entries (most recent at back).
    cache_order: std::collections::VecDeque<EntryId>,
    /// Cache memory configuration.
    cache_config: CacheConfig,
    /// Running total of cached steps for budget enforcement.
    cached_steps_count: usize,
    /// Policy hook: count derivation steps carried by a cache payload.
    cache_steps_len: fn(&CacheValue) -> usize,
    /// Policy hook: optionally strip heavy payload details when caching.
    cache_apply_light: fn(CacheValue, Option<usize>) -> CacheValue,
}

impl<D: Default, C> Default for SessionStore<D, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: Default, C> SessionStore<D, C> {
    /// Create a new empty session store.
    pub fn new() -> Self {
        Self::with_cache_config(CacheConfig::default())
    }

    /// Create a session store with custom cache configuration.
    pub fn with_cache_config(config: CacheConfig) -> Self {
        Self::with_cache_config_and_policy(
            config,
            default_cache_steps_len::<C>,
            default_apply_light_cache::<C>,
        )
    }

    /// Create a session store with custom cache configuration and policy hooks.
    pub fn with_cache_config_and_policy(
        config: CacheConfig,
        cache_steps_len: fn(&C) -> usize,
        cache_apply_light: fn(C, Option<usize>) -> C,
    ) -> Self {
        Self {
            next_id: 1,
            entries: Vec::new(),
            cache_order: std::collections::VecDeque::new(),
            cache_config: config,
            cached_steps_count: 0,
            cache_steps_len,
            cache_apply_light,
        }
    }

    /// Get cache statistics `(cached_entries, total_steps)`.
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cache_order.len(), self.cached_steps_count)
    }

    /// Store a new entry with default diagnostics and return its ID.
    pub fn push(&mut self, kind: EntryKind, raw_text: String) -> EntryId {
        self.push_with_diagnostics(kind, raw_text, D::default())
    }

    /// Store a new entry with diagnostics and return its ID.
    pub fn push_with_diagnostics(
        &mut self,
        kind: EntryKind,
        raw_text: String,
        diagnostics: D,
    ) -> EntryId {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push(Entry {
            id,
            kind,
            raw_text,
            diagnostics,
            simplified: None,
        });
        id
    }

    /// Get an entry by ID.
    pub fn get(&self, id: EntryId) -> Option<&Entry<D, C>> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Remove entries by IDs (IDs are never reused).
    pub fn remove(&mut self, ids: &[EntryId]) {
        self.entries.retain(|e| !ids.contains(&e.id));
    }

    /// Clear all entries (IDs are still never reused).
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get all entries.
    pub fn list(&self) -> &[Entry<D, C>] {
        &self.entries
    }

    /// Check if an entry exists.
    pub fn contains(&self, id: EntryId) -> bool {
        self.entries.iter().any(|e| e.id == id)
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the next ID that will be assigned (for preview).
    pub fn next_id(&self) -> EntryId {
        self.next_id
    }

    /// Update diagnostics for an entry.
    pub fn update_diagnostics(&mut self, id: EntryId, diagnostics: D) {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) {
            entry.diagnostics = diagnostics;
        }
    }

    /// Iterate over all entries (for snapshot serialization).
    pub fn entries(&self) -> impl Iterator<Item = &Entry<D, C>> {
        self.entries.iter()
    }

    /// Get the LRU cache order (for snapshot serialization).
    pub fn cache_order(&self) -> &std::collections::VecDeque<EntryId> {
        &self.cache_order
    }

    /// Get the cache configuration.
    pub fn cache_config(&self) -> &CacheConfig {
        &self.cache_config
    }

    /// Restore an entry from snapshot (bypasses normal ID allocation).
    pub fn restore_entry(&mut self, entry: Entry<D, C>) {
        let cache_steps_len = self.cache_steps_len;
        if entry.id >= self.next_id {
            self.next_id = entry.id + 1;
        }
        if let Some(ref cache) = entry.simplified {
            self.cached_steps_count += cache_steps_len(cache);
        }
        self.entries.push(entry);
    }

    /// Restore the LRU cache order from snapshot.
    pub fn restore_cache_order(&mut self, order: Vec<EntryId>) {
        self.cache_order = order.into_iter().collect();
    }
}

impl<D: Default, C> SessionStore<D, C> {
    /// Update the simplified cache for an entry.
    ///
    /// Implements LRU eviction with configurable limits:
    /// - Applies light-cache mode for large entries.
    /// - Evicts oldest cached entries when over budget.
    pub fn update_simplified<V>(&mut self, id: EntryId, simplified: V)
    where
        V: Into<C>,
    {
        let cache_steps_len = self.cache_steps_len;
        let cache_apply_light = self.cache_apply_light;
        let mut simplified = simplified.into();
        simplified = cache_apply_light(simplified, self.cache_config.light_cache_threshold);

        if let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) {
            let old_steps = entry.simplified.as_ref().map(cache_steps_len).unwrap_or(0);
            let new_steps = cache_steps_len(&simplified);

            entry.simplified = Some(simplified);

            self.cache_order.retain(|&eid| eid != id);
            self.cache_order.push_back(id);

            self.cached_steps_count = self.cached_steps_count + new_steps - old_steps;
            self.evict_if_needed();
        }
    }

    /// Touch a cached entry to mark it as recently used (for LRU).
    pub fn touch_cached(&mut self, id: EntryId) {
        if let Some(entry) = self.entries.iter().find(|e| e.id == id) {
            if entry.simplified.is_some() {
                self.cache_order.retain(|&eid| eid != id);
                self.cache_order.push_back(id);
            }
        }
    }

    /// Evict oldest cached entries until within configured limits.
    fn evict_if_needed(&mut self) {
        loop {
            let over_entries = self.cache_config.max_cached_entries > 0
                && self.cache_order.len() > self.cache_config.max_cached_entries;
            let over_steps = self.cache_config.max_cached_steps > 0
                && self.cached_steps_count > self.cache_config.max_cached_steps;

            if !(over_entries || over_steps) {
                break;
            }

            if let Some(oldest_id) = self.cache_order.pop_front() {
                if let Some(entry) = self.entries.iter_mut().find(|e| e.id == oldest_id) {
                    if let Some(cache) = entry.simplified.take() {
                        self.cached_steps_count = self
                            .cached_steps_count
                            .saturating_sub((self.cache_steps_len)(&cache));
                    }
                }
            } else {
                break;
            }
        }
    }
}
