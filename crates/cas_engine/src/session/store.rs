//! Session store types and `SessionStore` implementation.

use cas_ast::ExprId;

/// Unique identifier for a session entry
pub type EntryId = u64;

/// Type of entry stored in the session
#[derive(Debug, Clone)]
pub enum EntryKind {
    /// A single expression
    Expr(ExprId),
    /// An equation (lhs = rhs)
    Eq { lhs: ExprId, rhs: ExprId },
}

// =============================================================================
// Session Reference Caching (V2.15.36)
// =============================================================================

/// Key for cache invalidation - must match for cache hit.
///
/// If any of these settings change between when the cache was created
/// and when it's being used, the cache is invalid.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SimplifyCacheKey {
    /// Domain mode at time of simplification
    pub domain: crate::domain::DomainMode,
    /// Build/version hash for ruleset (currently static)
    pub ruleset_rev: u64,
}

impl SimplifyCacheKey {
    /// Create a cache key from current context settings
    pub fn from_context(domain: crate::domain::DomainMode) -> Self {
        Self {
            domain,
            // For now, use a static value. In the future, could hash ruleset config.
            ruleset_rev: 1,
        }
    }

    /// Check if this key is compatible with another (for cache hit)
    pub fn is_compatible(&self, other: &Self) -> bool {
        self == other
    }
}

/// Configuration for simplified cache memory limits.
///
/// Controls how many cached simplified results are retained to
/// prevent unbounded memory growth in long sessions.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Max entries with cached simplified result (0 = unlimited)
    pub max_cached_entries: usize,
    /// Max total steps across all cached entries (0 = unlimited)
    pub max_cached_steps: usize,
    /// Drop steps for entries with > N steps (light cache mode)
    pub light_cache_threshold: Option<usize>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_cached_entries: 100,          // Reasonable default
            max_cached_steps: 5000,           // ~50 steps avg per entry
            light_cache_threshold: Some(200), // Drop steps if > 200
        }
    }
}

/// Cached simplification result for a session entry.
///
/// Stored after evaluation to enable fast resolution of `#N` references
/// without re-running the simplification pipeline.
#[derive(Debug, Clone)]
pub struct SimplifiedCache {
    /// Key for invalidation (must match current context)
    pub key: SimplifyCacheKey,
    /// Simplified expression
    pub expr: ExprId,
    /// Domain requirements from this entry (for propagation)
    pub requires: Vec<crate::diagnostics::RequiredItem>,
    /// Derivation steps (None = light cache, steps omitted for large entries)
    pub steps: Option<std::sync::Arc<Vec<crate::step::Step>>>,
}

/// How to resolve session references
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RefMode {
    /// Use cached simplified result if available and valid (default, fast)
    #[default]
    PreferSimplified,
    /// Use original parsed expression (for debugging, "raw" command)
    Raw,
}

/// Record of a single cache hit during resolution.
///
/// Used to generate synthetic timeline steps showing which
/// cached results were used and what they resolved to.
#[derive(Debug, Clone)]
pub struct CacheHitTrace {
    /// The entry ID that was resolved from cache
    pub entry_id: EntryId,
    /// The ExprId of the `#N` node in the AST before resolution
    pub before_ref_expr: ExprId,
    /// The cached simplified ExprId that replaced the reference
    pub after_expr: ExprId,
    /// Domain requirements from the cached entry
    pub requires: Vec<crate::diagnostics::RequiredItem>,
}

/// Result of resolving session references with accumulated requires.
#[derive(Debug, Clone)]
pub struct ResolvedExpr {
    /// The resolved expression
    pub expr: ExprId,
    /// Accumulated domain requirements from all referenced entries
    pub requires: Vec<crate::diagnostics::RequiredItem>,
    /// Whether cache was used (for timeline step generation)
    pub used_cache: bool,
    /// Chain of referenced entry IDs (for debugging)
    pub ref_chain: smallvec::SmallVec<[EntryId; 4]>,
    /// Cache hits recorded during resolution (for synthetic step generation)
    pub cache_hits: Vec<CacheHitTrace>,
}

/// A stored entry in the session
#[derive(Debug, Clone)]
pub struct Entry {
    /// Unique ID (auto-incrementing, never reused)
    pub id: EntryId,
    /// The stored expression or equation
    pub kind: EntryKind,
    /// Original raw text input (for display)
    pub raw_text: String,
    /// Diagnostics from evaluation (for SessionPropagated tracking)
    pub diagnostics: crate::diagnostics::Diagnostics,
    /// Cached simplified result (populated after eval)
    pub simplified: Option<SimplifiedCache>,
}

impl Entry {
    /// Check if this entry is an expression
    pub fn is_expr(&self) -> bool {
        matches!(self.kind, EntryKind::Expr(_))
    }

    /// Check if this entry is an equation
    pub fn is_eq(&self) -> bool {
        matches!(self.kind, EntryKind::Eq { .. })
    }

    /// Get the type as a string for display
    pub fn type_str(&self) -> &'static str {
        match self.kind {
            EntryKind::Expr(_) => "Expr",
            EntryKind::Eq { .. } => "Eq",
        }
    }
}

/// Storage for session entries with auto-incrementing IDs
#[derive(Debug, Clone)]
pub struct SessionStore {
    next_id: EntryId,
    entries: Vec<Entry>,
    /// V2.15.36: LRU tracking for cache eviction (most recent at back)
    cache_order: std::collections::VecDeque<EntryId>,
    /// Cache memory configuration
    cache_config: CacheConfig,
    /// Running total of cached steps for budget enforcement
    cached_steps_count: usize,
}

impl Default for SessionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionStore {
    /// Create a new empty session store
    pub fn new() -> Self {
        Self {
            next_id: 1,
            entries: Vec::new(),
            cache_order: std::collections::VecDeque::new(),
            cache_config: CacheConfig::default(),
            cached_steps_count: 0,
        }
    }

    /// Create a session store with custom cache configuration
    pub fn with_cache_config(config: CacheConfig) -> Self {
        Self {
            next_id: 1,
            entries: Vec::new(),
            cache_order: std::collections::VecDeque::new(),
            cache_config: config,
            cached_steps_count: 0,
        }
    }

    /// Get cache statistics (cached_entries, total_steps)
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cache_order.len(), self.cached_steps_count)
    }

    /// Store a new entry and return its ID (no diagnostics)
    pub fn push(&mut self, kind: EntryKind, raw_text: String) -> EntryId {
        self.push_with_diagnostics(kind, raw_text, crate::diagnostics::Diagnostics::default())
    }

    /// Store a new entry with diagnostics and return its ID
    pub fn push_with_diagnostics(
        &mut self,
        kind: EntryKind,
        raw_text: String,
        diagnostics: crate::diagnostics::Diagnostics,
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

    /// Get an entry by ID
    pub fn get(&self, id: EntryId) -> Option<&Entry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Remove entries by IDs (IDs are never reused)
    pub fn remove(&mut self, ids: &[EntryId]) {
        self.entries.retain(|e| !ids.contains(&e.id));
    }

    /// Clear all entries (IDs are still never reused)
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get all entries
    pub fn list(&self) -> &[Entry] {
        &self.entries
    }

    /// Check if an entry exists
    pub fn contains(&self, id: EntryId) -> bool {
        self.entries.iter().any(|e| e.id == id)
    }

    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the next ID that will be assigned (for preview)
    pub fn next_id(&self) -> EntryId {
        self.next_id
    }

    /// Update the diagnostics for an entry (used after eval completes)
    pub fn update_diagnostics(
        &mut self,
        id: EntryId,
        diagnostics: crate::diagnostics::Diagnostics,
    ) {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) {
            entry.diagnostics = diagnostics;
        }
    }

    /// Update the simplified cache for an entry (populated after eval).
    ///
    /// This caches the simplified result so that subsequent `#id` references
    /// can use the cached value instead of re-simplifying.
    ///
    /// V2.15.36: Implements LRU eviction with configurable limits.
    /// - Applies light-cache (drops steps) for large entries
    /// - Evicts oldest cached entries when over budget (after insert)
    pub fn update_simplified(&mut self, id: EntryId, mut simplified: SimplifiedCache) {
        // Apply light-cache mode: drop steps for large entries
        simplified = self.apply_light_cache(simplified);

        if let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) {
            // Compute step count delta
            let old_steps = entry
                .simplified
                .as_ref()
                .and_then(|c| c.steps.as_ref())
                .map(|s| s.len())
                .unwrap_or(0);
            let new_steps = simplified.steps.as_ref().map(|s| s.len()).unwrap_or(0);

            // Update cache
            entry.simplified = Some(simplified);

            // Update LRU order (remove old position, add to back)
            self.cache_order.retain(|&eid| eid != id);
            self.cache_order.push_back(id);

            // Update step count budget
            self.cached_steps_count = self.cached_steps_count + new_steps - old_steps;

            // Evict if over limits (AFTER insert)
            self.evict_if_needed();
        }
    }

    /// Touch a cached entry to mark it as recently used (for LRU).
    ///
    /// Call this when resolving `#N` from cache to keep hot entries alive.
    pub fn touch_cached(&mut self, id: EntryId) {
        if let Some(entry) = self.entries.iter().find(|e| e.id == id) {
            if entry.simplified.is_some() {
                self.cache_order.retain(|&eid| eid != id);
                self.cache_order.push_back(id);
            }
        }
    }

    /// Apply light-cache mode: drop steps for entries over threshold.
    fn apply_light_cache(&self, mut simplified: SimplifiedCache) -> SimplifiedCache {
        if let Some(threshold) = self.cache_config.light_cache_threshold {
            if let Some(ref steps) = simplified.steps {
                if steps.len() > threshold {
                    simplified.steps = None; // Drop steps to save memory
                }
            }
        }
        simplified
    }

    /// Evict oldest cached entries until within limits.
    fn evict_if_needed(&mut self) {
        loop {
            // Check if over entry limit (0 = unlimited)
            let over_entries = self.cache_config.max_cached_entries > 0
                && self.cache_order.len() > self.cache_config.max_cached_entries;

            // Check if over steps budget (0 = unlimited)
            let over_steps = self.cache_config.max_cached_steps > 0
                && self.cached_steps_count > self.cache_config.max_cached_steps;

            if !(over_entries || over_steps) {
                break;
            }

            // Evict oldest (front of queue)
            if let Some(oldest_id) = self.cache_order.pop_front() {
                if let Some(entry) = self.entries.iter_mut().find(|e| e.id == oldest_id) {
                    if let Some(cache) = entry.simplified.take() {
                        let step_count = cache.steps.as_ref().map(|s| s.len()).unwrap_or(0);
                        self.cached_steps_count =
                            self.cached_steps_count.saturating_sub(step_count);
                    }
                }
            } else {
                break; // No more to evict
            }
        }
    }

    // =========================================================================
    // Snapshot Persistence Support (V2.15.36)
    // =========================================================================

    /// Iterate over all entries (for snapshot serialization).
    pub fn entries(&self) -> impl Iterator<Item = &Entry> {
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
    pub fn restore_entry(&mut self, entry: Entry) {
        // Track next_id to ensure future entries don't collide
        if entry.id >= self.next_id {
            self.next_id = entry.id + 1;
        }
        // Update cached_steps_count if entry has cached steps
        if let Some(ref cache) = entry.simplified {
            if let Some(ref steps) = cache.steps {
                self.cached_steps_count += steps.len();
            }
        }
        self.entries.push(entry);
    }

    /// Restore the LRU cache order from snapshot.
    pub fn restore_cache_order(&mut self, order: Vec<EntryId>) {
        self.cache_order = order.into_iter().collect();
    }
}
