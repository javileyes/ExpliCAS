//! Session storage for expressions and equations with auto-incrementing IDs.
//!
//! Provides a "notebook-style" storage where each input gets a unique `#id`
//! that can be referenced in subsequent commands.

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

// =============================================================================
// Session Reference Resolution
// =============================================================================

use std::collections::{HashMap, HashSet};

/// Error during session reference resolution
#[derive(Debug, Clone, PartialEq)]
pub enum ResolveError {
    /// Reference to non-existent entry
    NotFound(EntryId),
    /// Circular reference detected (e.g., #3 contains #3, or #3 -> #4 -> #3)
    CircularReference(EntryId),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::NotFound(id) => write!(f, "Session reference #{} not found", id),
            ResolveError::CircularReference(id) => {
                write!(f, "Circular reference detected involving #{}", id)
            }
        }
    }
}

impl std::error::Error for ResolveError {}

/// Resolve all `Expr::SessionRef` in an expression tree.
///
/// - For expression entries: replaces `#id` with the stored ExprId
/// - For equation entries: replaces `#id` with `(lhs - rhs)` (residue form)
///
/// Uses memoization to avoid re-resolving the same reference.
/// Detects circular references and returns an error.
pub fn resolve_session_refs(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    store: &SessionStore,
) -> Result<ExprId, ResolveError> {
    let mut cache: HashMap<EntryId, ExprId> = HashMap::new();
    let mut visiting: HashSet<EntryId> = HashSet::new();
    let mut _inherited = crate::diagnostics::Diagnostics::new();
    resolve_recursive(ctx, expr, store, &mut cache, &mut visiting, &mut _inherited)
}

/// Resolve session refs AND accumulate inherited diagnostics.
///
/// When an expression references `#id`, the diagnostics from that entry
/// are accumulated for SessionPropagated tracking.
pub fn resolve_session_refs_with_diagnostics(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    store: &SessionStore,
) -> Result<(ExprId, crate::diagnostics::Diagnostics), ResolveError> {
    let mut cache: HashMap<EntryId, ExprId> = HashMap::new();
    let mut visiting: HashSet<EntryId> = HashSet::new();
    let mut inherited = crate::diagnostics::Diagnostics::new();
    let resolved = resolve_recursive(ctx, expr, store, &mut cache, &mut visiting, &mut inherited)?;
    Ok((resolved, inherited))
}

/// Resolve session refs with mode selection and cache checking (V2.15.36).
///
/// This is the preferred resolution method when you have a `SimplifyCacheKey`.
/// It checks the simplified cache before falling back to raw expressions.
///
/// # Arguments
/// * `ctx` - The expression context
/// * `expr` - Expression to resolve (may contain `#N` references)
/// * `store` - Session store with entries
/// * `mode` - PreferSimplified (use cache) or Raw (use parsed expr)
/// * `cache_key` - Current context key for cache validation
///
/// # Returns
/// * `ResolvedExpr` with resolved expression and accumulated requires
pub fn resolve_session_refs_with_mode(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    store: &SessionStore,
    mode: RefMode,
    cache_key: &SimplifyCacheKey,
) -> Result<ResolvedExpr, ResolveError> {
    use std::collections::HashSet;

    let mut memo: HashMap<ExprId, ExprId> = HashMap::new();
    let mut visiting: Vec<EntryId> = Vec::new();
    let mut requires: Vec<crate::diagnostics::RequiredItem> = Vec::new();
    let mut used_cache = false;
    let mut ref_chain: smallvec::SmallVec<[EntryId; 4]> = smallvec::SmallVec::new();
    // V2.15.36: Track cache hits for synthetic timeline step
    let mut seen_hits: HashSet<EntryId> = HashSet::new();
    let mut cache_hits: Vec<CacheHitTrace> = Vec::new();

    let resolved = resolve_with_mode_recursive(
        ctx,
        expr,
        store,
        mode,
        cache_key,
        &mut memo,
        &mut visiting,
        &mut requires,
        &mut used_cache,
        &mut ref_chain,
        &mut seen_hits,
        &mut cache_hits,
    )?;

    Ok(ResolvedExpr {
        expr: resolved,
        requires,
        used_cache,
        ref_chain,
        cache_hits,
    })
}

/// Internal recursive resolver with cache checking
#[allow(clippy::too_many_arguments)]
fn resolve_with_mode_recursive(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    store: &SessionStore,
    mode: RefMode,
    cache_key: &SimplifyCacheKey,
    memo: &mut HashMap<ExprId, ExprId>,
    visiting: &mut Vec<EntryId>,
    requires: &mut Vec<crate::diagnostics::RequiredItem>,
    used_cache: &mut bool,
    ref_chain: &mut smallvec::SmallVec<[EntryId; 4]>,
    seen_hits: &mut std::collections::HashSet<EntryId>,
    cache_hits: &mut Vec<CacheHitTrace>,
) -> Result<ExprId, ResolveError> {
    use cas_ast::Expr;

    // Check memo first
    if let Some(&cached) = memo.get(&expr) {
        return Ok(cached);
    }

    let node = ctx.get(expr).clone();

    let result = match node {
        Expr::SessionRef(id) => resolve_entry_with_mode(
            ctx, expr, id, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
            seen_hits, cache_hits,
        )?,

        // Handle Variable that might be a #N reference (legacy parsing)
        Expr::Variable(sym_id) => {
            let name = ctx.sym_name(sym_id);
            if name.starts_with('#') && name.len() > 1 && name[1..].chars().all(char::is_numeric) {
                if let Ok(id) = name[1..].parse::<u64>() {
                    resolve_entry_with_mode(
                        ctx, expr, id, store, mode, cache_key, memo, visiting, requires,
                        used_cache, ref_chain, seen_hits, cache_hits,
                    )?
                } else {
                    expr
                }
            } else {
                expr
            }
        }

        // Binary operators - recurse into children
        Expr::Add(l, r) => {
            let new_l = resolve_with_mode_recursive(
                ctx, l, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            let new_r = resolve_with_mode_recursive(
                ctx, r, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            if new_l == l && new_r == r {
                expr
            } else {
                ctx.add(Expr::Add(new_l, new_r))
            }
        }
        Expr::Sub(l, r) => {
            let new_l = resolve_with_mode_recursive(
                ctx, l, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            let new_r = resolve_with_mode_recursive(
                ctx, r, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            if new_l == l && new_r == r {
                expr
            } else {
                ctx.add(Expr::Sub(new_l, new_r))
            }
        }
        Expr::Mul(l, r) => {
            let new_l = resolve_with_mode_recursive(
                ctx, l, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            let new_r = resolve_with_mode_recursive(
                ctx, r, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            if new_l == l && new_r == r {
                expr
            } else {
                ctx.add(Expr::Mul(new_l, new_r))
            }
        }
        Expr::Div(l, r) => {
            let new_l = resolve_with_mode_recursive(
                ctx, l, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            let new_r = resolve_with_mode_recursive(
                ctx, r, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            if new_l == l && new_r == r {
                expr
            } else {
                ctx.add(Expr::Div(new_l, new_r))
            }
        }
        Expr::Pow(b, e) => {
            let new_b = resolve_with_mode_recursive(
                ctx, b, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            let new_e = resolve_with_mode_recursive(
                ctx, e, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            if new_b == b && new_e == e {
                expr
            } else {
                ctx.add(Expr::Pow(new_b, new_e))
            }
        }

        // Unary
        Expr::Neg(e) => {
            let new_e = resolve_with_mode_recursive(
                ctx, e, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            if new_e == e {
                expr
            } else {
                ctx.add(Expr::Neg(new_e))
            }
        }

        // Function
        Expr::Function(name, args) => {
            let mut changed = false;
            let mut new_args = Vec::with_capacity(args.len());
            for arg in &args {
                let new_arg = resolve_with_mode_recursive(
                    ctx, *arg, store, mode, cache_key, memo, visiting, requires, used_cache,
                    ref_chain, seen_hits, cache_hits,
                )?;
                if new_arg != *arg {
                    changed = true;
                }
                new_args.push(new_arg);
            }
            if changed {
                ctx.add(Expr::Function(name, new_args))
            } else {
                expr
            }
        }

        // Matrix
        Expr::Matrix { rows, cols, data } => {
            let mut changed = false;
            let mut new_data = Vec::with_capacity(data.len());
            for elem in &data {
                let new_elem = resolve_with_mode_recursive(
                    ctx, *elem, store, mode, cache_key, memo, visiting, requires, used_cache,
                    ref_chain, seen_hits, cache_hits,
                )?;
                if new_elem != *elem {
                    changed = true;
                }
                new_data.push(new_elem);
            }
            if changed {
                ctx.add(Expr::Matrix {
                    rows,
                    cols,
                    data: new_data,
                })
            } else {
                expr
            }
        }

        // Leaf nodes
        Expr::Number(_) | Expr::Constant(_) => expr,
    };

    memo.insert(expr, result);
    Ok(result)
}

/// Resolve a single entry ID using cache if available
#[allow(clippy::too_many_arguments)]
fn resolve_entry_with_mode(
    ctx: &mut cas_ast::Context,
    ref_expr_id: ExprId, // The ExprId of the #N node in AST (for cache hit trace)
    id: EntryId,
    store: &SessionStore,
    mode: RefMode,
    cache_key: &SimplifyCacheKey,
    memo: &mut HashMap<ExprId, ExprId>,
    visiting: &mut Vec<EntryId>,
    requires: &mut Vec<crate::diagnostics::RequiredItem>,
    used_cache: &mut bool,
    ref_chain: &mut smallvec::SmallVec<[EntryId; 4]>,
    seen_hits: &mut std::collections::HashSet<EntryId>,
    cache_hits: &mut Vec<CacheHitTrace>,
) -> Result<ExprId, ResolveError> {
    use cas_ast::Expr;

    // Cycle detection
    if visiting.contains(&id) {
        return Err(ResolveError::CircularReference(id));
    }

    // Get entry
    let entry = store.get(id).ok_or(ResolveError::NotFound(id))?;

    // Track reference chain
    ref_chain.push(id);

    // 1) PreferSimplified: check cache first
    if mode == RefMode::PreferSimplified {
        if let Some(cache) = &entry.simplified {
            if cache.key.is_compatible(cache_key) {
                // Cache hit! Use cached expression and accumulate requires
                *used_cache = true;
                requires.extend(cache.requires.iter().cloned());

                // V2.15.36: Record cache hit for synthetic step (dedup by entry_id)
                if seen_hits.insert(id) {
                    cache_hits.push(CacheHitTrace {
                        entry_id: id,
                        before_ref_expr: ref_expr_id,
                        after_expr: cache.expr,
                        requires: cache.requires.clone(),
                    });
                }

                return Ok(cache.expr);
            }
        }
    }

    // 2) Fallback: use raw parsed expression
    visiting.push(id);

    // Inherit requires from entry's diagnostics (for SessionPropagated tracking)
    for item in &entry.diagnostics.requires {
        if !requires.iter().any(|r| r.cond == item.cond) {
            let mut new_item = item.clone();
            new_item.merge_origin(crate::diagnostics::RequireOrigin::SessionPropagated);
            requires.push(new_item);
        }
    }

    let resolved = match &entry.kind {
        EntryKind::Expr(stored_expr) => {
            // Recursively resolve (it may contain #refs too)
            resolve_with_mode_recursive(
                ctx,
                *stored_expr,
                store,
                mode,
                cache_key,
                memo,
                visiting,
                requires,
                used_cache,
                ref_chain,
                seen_hits,
                cache_hits,
            )?
        }
        EntryKind::Eq { lhs, rhs } => {
            // Equation as expression: use residue form (lhs - rhs)
            let resolved_lhs = resolve_with_mode_recursive(
                ctx, *lhs, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            let resolved_rhs = resolve_with_mode_recursive(
                ctx, *rhs, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            ctx.add(Expr::Sub(resolved_lhs, resolved_rhs))
        }
    };

    visiting.pop();
    Ok(resolved)
}

fn resolve_recursive(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    store: &SessionStore,
    cache: &mut HashMap<EntryId, ExprId>,
    visiting: &mut HashSet<EntryId>,
    inherited: &mut crate::diagnostics::Diagnostics,
) -> Result<ExprId, ResolveError> {
    use cas_ast::Expr;

    let node = ctx.get(expr).clone();

    match node {
        Expr::SessionRef(id) => resolve_session_id(ctx, id, store, cache, visiting, inherited),
        Expr::Variable(sym_id) => {
            let name = ctx.sym_name(sym_id);
            if name.starts_with('#') && name.len() > 1 && name[1..].chars().all(char::is_numeric) {
                if let Ok(id) = name[1..].parse::<u64>() {
                    return resolve_session_id(ctx, id, store, cache, visiting, inherited);
                }
            }
            Ok(expr)
        }

        // Binary operators - recurse into children
        Expr::Add(l, r) => {
            let new_l = resolve_recursive(ctx, l, store, cache, visiting, inherited)?;
            let new_r = resolve_recursive(ctx, r, store, cache, visiting, inherited)?;
            if new_l == l && new_r == r {
                Ok(expr)
            } else {
                Ok(ctx.add(Expr::Add(new_l, new_r)))
            }
        }
        Expr::Sub(l, r) => {
            let new_l = resolve_recursive(ctx, l, store, cache, visiting, inherited)?;
            let new_r = resolve_recursive(ctx, r, store, cache, visiting, inherited)?;
            if new_l == l && new_r == r {
                Ok(expr)
            } else {
                Ok(ctx.add(Expr::Sub(new_l, new_r)))
            }
        }
        Expr::Mul(l, r) => {
            let new_l = resolve_recursive(ctx, l, store, cache, visiting, inherited)?;
            let new_r = resolve_recursive(ctx, r, store, cache, visiting, inherited)?;
            if new_l == l && new_r == r {
                Ok(expr)
            } else {
                Ok(ctx.add(Expr::Mul(new_l, new_r)))
            }
        }
        Expr::Div(l, r) => {
            let new_l = resolve_recursive(ctx, l, store, cache, visiting, inherited)?;
            let new_r = resolve_recursive(ctx, r, store, cache, visiting, inherited)?;
            if new_l == l && new_r == r {
                Ok(expr)
            } else {
                Ok(ctx.add(Expr::Div(new_l, new_r)))
            }
        }
        Expr::Pow(b, e) => {
            let new_b = resolve_recursive(ctx, b, store, cache, visiting, inherited)?;
            let new_e = resolve_recursive(ctx, e, store, cache, visiting, inherited)?;
            if new_b == b && new_e == e {
                Ok(expr)
            } else {
                Ok(ctx.add(Expr::Pow(new_b, new_e)))
            }
        }

        // Unary
        Expr::Neg(e) => {
            let new_e = resolve_recursive(ctx, e, store, cache, visiting, inherited)?;
            if new_e == e {
                Ok(expr)
            } else {
                Ok(ctx.add(Expr::Neg(new_e)))
            }
        }

        // Function - recurse into args
        Expr::Function(name, args) => {
            let mut changed = false;
            let mut new_args = Vec::with_capacity(args.len());
            for arg in &args {
                let new_arg = resolve_recursive(ctx, *arg, store, cache, visiting, inherited)?;
                if new_arg != *arg {
                    changed = true;
                }
                new_args.push(new_arg);
            }
            if changed {
                Ok(ctx.add(Expr::Function(name, new_args)))
            } else {
                Ok(expr)
            }
        }

        // Matrix - recurse into elements
        Expr::Matrix { rows, cols, data } => {
            let mut changed = false;
            let mut new_data = Vec::with_capacity(data.len());
            for elem in &data {
                let new_elem = resolve_recursive(ctx, *elem, store, cache, visiting, inherited)?;
                if new_elem != *elem {
                    changed = true;
                }
                new_data.push(new_elem);
            }
            if changed {
                Ok(ctx.add(Expr::Matrix {
                    rows,
                    cols,
                    data: new_data,
                }))
            } else {
                Ok(expr)
            }
        }

        // Leaf nodes - no change needed
        Expr::Number(_) | Expr::Constant(_) => Ok(expr),
    }
}

fn resolve_session_id(
    ctx: &mut cas_ast::Context,
    id: EntryId,
    store: &SessionStore,
    cache: &mut HashMap<EntryId, ExprId>,
    visiting: &mut HashSet<EntryId>,
    inherited: &mut crate::diagnostics::Diagnostics,
) -> Result<ExprId, ResolveError> {
    use cas_ast::Expr;

    // Check cache first
    if let Some(&resolved) = cache.get(&id) {
        return Ok(resolved);
    }

    // Cycle detection
    if visiting.contains(&id) {
        return Err(ResolveError::CircularReference(id));
    }

    // Get entry from store
    let entry = store.get(id).ok_or(ResolveError::NotFound(id))?;

    // SessionPropagated: inherit requires from this entry
    inherited.inherit_requires_from(&entry.diagnostics);

    // Mark as visiting for cycle detection
    visiting.insert(id);

    // Get the expression to substitute
    let substitution = match &entry.kind {
        EntryKind::Expr(stored_expr) => {
            // Recursively resolve the stored expression (it may contain #refs too)
            resolve_recursive(ctx, *stored_expr, store, cache, visiting, inherited)?
        }
        EntryKind::Eq { lhs, rhs } => {
            // For equations used as expressions, use residue form: (lhs - rhs)
            let resolved_lhs = resolve_recursive(ctx, *lhs, store, cache, visiting, inherited)?;
            let resolved_rhs = resolve_recursive(ctx, *rhs, store, cache, visiting, inherited)?;
            ctx.add(Expr::Sub(resolved_lhs, resolved_rhs))
        }
    };

    // Done visiting
    visiting.remove(&id);

    // Cache the result
    cache.insert(id, substitution);

    Ok(substitution)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_get() {
        let mut store = SessionStore::new();

        // Create a dummy ExprId (in real usage this comes from Context)
        let expr_id = ExprId::from_raw(0);

        let id1 = store.push(EntryKind::Expr(expr_id), "x + 1".to_string());
        let id2 = store.push(EntryKind::Expr(expr_id), "x^2".to_string());

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);

        let entry1 = store.get(1).unwrap();
        assert_eq!(entry1.raw_text, "x + 1");
        assert!(entry1.is_expr());

        let entry2 = store.get(2).unwrap();
        assert_eq!(entry2.raw_text, "x^2");
    }

    #[test]
    fn test_ids_not_reused_after_delete() {
        let mut store = SessionStore::new();
        let expr_id = ExprId::from_raw(0);

        let id1 = store.push(EntryKind::Expr(expr_id), "a".to_string());
        let id2 = store.push(EntryKind::Expr(expr_id), "b".to_string());

        // Delete id1
        store.remove(&[id1]);

        // Next ID should be 3, not 1
        let id3 = store.push(EntryKind::Expr(expr_id), "c".to_string());
        assert_eq!(id3, 3);
        assert!(!store.contains(id1));
        assert!(store.contains(id2));
        assert!(store.contains(id3));
    }

    #[test]
    fn test_remove_multiple() {
        let mut store = SessionStore::new();
        let expr_id = ExprId::from_raw(0);

        store.push(EntryKind::Expr(expr_id), "a".to_string());
        store.push(EntryKind::Expr(expr_id), "b".to_string());
        store.push(EntryKind::Expr(expr_id), "c".to_string());

        store.remove(&[1, 3]);
        assert_eq!(store.len(), 1);
        assert!(store.contains(2));
    }

    #[test]
    fn test_clear() {
        let mut store = SessionStore::new();
        let expr_id = ExprId::from_raw(0);

        store.push(EntryKind::Expr(expr_id), "a".to_string());
        store.push(EntryKind::Expr(expr_id), "b".to_string());

        store.clear();
        assert!(store.is_empty());

        // Next ID should still be 3
        let id = store.push(EntryKind::Expr(expr_id), "c".to_string());
        assert_eq!(id, 3);
    }

    #[test]
    fn test_equation_entry() {
        let mut store = SessionStore::new();
        let lhs = ExprId::from_raw(0);
        let rhs = ExprId::from_raw(1);

        let id = store.push(EntryKind::Eq { lhs, rhs }, "x + 1 = 5".to_string());

        let entry = store.get(id).unwrap();
        assert!(entry.is_eq());
        assert_eq!(entry.type_str(), "Eq");
    }

    // ========== resolve_session_refs Tests ==========

    #[test]
    fn test_resolve_simple_ref() {
        use cas_ast::{Context, DisplayExpr, Expr};

        let mut ctx = Context::new();
        let mut store = SessionStore::new();

        // Store x + 1 as #1
        let x = ctx.var("x");
        let one = ctx.num(1);
        let expr1 = ctx.add(Expr::Add(x, one));
        store.push(EntryKind::Expr(expr1), "x + 1".to_string());

        // Create #1 * 2
        let ref1 = ctx.add(Expr::SessionRef(1));
        let two = ctx.num(2);
        let input = ctx.add(Expr::Mul(ref1, two));

        // Resolve
        let resolved = resolve_session_refs(&mut ctx, input, &store).unwrap();

        // Check using DisplayExpr - should contain (x + 1) and 2
        let display = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: resolved
            }
        );
        // Resolved should not contain "#" anymore
        assert!(
            !display.contains('#'),
            "Resolved should not contain session refs: {}",
            display
        );
        // Should contain x and 2
        assert!(display.contains('x'), "Should contain x: {}", display);
        assert!(display.contains('2'), "Should contain 2: {}", display);
        // Should be a multiplication
        assert!(
            display.contains('*'),
            "Should contain multiplication: {}",
            display
        );
    }

    #[test]
    fn test_resolve_not_found() {
        use cas_ast::{Context, Expr};

        let mut ctx = Context::new();
        let store = SessionStore::new();

        // Reference to non-existent #99
        let ref99 = ctx.add(Expr::SessionRef(99));

        let result = resolve_session_refs(&mut ctx, ref99, &store);
        assert!(matches!(result, Err(ResolveError::NotFound(99))));
    }

    #[test]
    fn test_resolve_equation_as_residue() {
        use cas_ast::{Context, Expr};

        let mut ctx = Context::new();
        let mut store = SessionStore::new();

        // Store equation: x + 1 = 5 as #1
        let x = ctx.var("x");
        let one = ctx.num(1);
        let five = ctx.num(5);
        let lhs = ctx.add(Expr::Add(x, one));
        store.push(EntryKind::Eq { lhs, rhs: five }, "x + 1 = 5".to_string());

        // Create just #1
        let ref1 = ctx.add(Expr::SessionRef(1));

        // Resolve - should get (x + 1) - 5
        let resolved = resolve_session_refs(&mut ctx, ref1, &store).unwrap();

        // Should be Sub
        if let Expr::Sub(l, r) = ctx.get(resolved) {
            // Left should be (x + 1)
            assert!(matches!(ctx.get(*l), Expr::Add(_, _)));
            // Right should be 5
            if let Expr::Number(n) = ctx.get(*r) {
                assert_eq!(n.to_integer(), 5.into());
            } else {
                panic!("Expected Number(5)");
            }
        } else {
            panic!("Expected Sub for equation residue");
        }
    }

    #[test]
    fn test_resolve_no_refs() {
        use cas_ast::{Context, Expr};

        let mut ctx = Context::new();
        let store = SessionStore::new();

        // Expression without refs: x + 1
        let x = ctx.var("x");
        let one = ctx.num(1);
        let input = ctx.add(Expr::Add(x, one));

        // Should return same expression
        let resolved = resolve_session_refs(&mut ctx, input, &store).unwrap();
        assert_eq!(resolved, input);
    }

    #[test]
    fn test_resolve_chained_refs() {
        use cas_ast::{Context, DisplayExpr, Expr};

        let mut ctx = Context::new();
        let mut store = SessionStore::new();

        // #1 = x
        let x = ctx.var("x");
        store.push(EntryKind::Expr(x), "x".to_string());

        // #2 = #1 + 1 (references #1)
        let ref1 = ctx.add(Expr::SessionRef(1));
        let one = ctx.num(1);
        let expr2 = ctx.add(Expr::Add(ref1, one));
        store.push(EntryKind::Expr(expr2), "#1 + 1".to_string());

        // Input: #2 * 2
        let ref2 = ctx.add(Expr::SessionRef(2));
        let two = ctx.num(2);
        let input = ctx.add(Expr::Mul(ref2, two));

        // Resolve - should get (x + 1) * 2
        let resolved = resolve_session_refs(&mut ctx, input, &store).unwrap();

        // Check using DisplayExpr
        let display = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: resolved
            }
        );
        // Should not contain any # refs
        assert!(
            !display.contains('#'),
            "Resolved should not contain session refs: {}",
            display
        );
        // Should contain x, 1, 2 and be a multiplication
        assert!(display.contains('x'), "Should contain x: {}", display);
        assert!(display.contains('2'), "Should contain 2: {}", display);
        assert!(
            display.contains('*'),
            "Should contain multiplication: {}",
            display
        );
    }

    // ========== Phase 2: resolve_session_refs_with_mode Tests ==========

    #[test]
    fn test_resolve_with_mode_cache_hit() {
        use cas_ast::{Context, Expr};

        let mut ctx = Context::new();
        let mut store = SessionStore::new();

        // Store raw expr: 5*sqrt(x)/sqrt(x) as #1
        let x = ctx.var("x");
        let sqrt_x = ctx.add(Expr::Function("sqrt".to_string(), vec![x]));
        let five = ctx.num(5);
        let mul = ctx.add(Expr::Mul(five, sqrt_x));
        let raw_expr = ctx.add(Expr::Div(mul, sqrt_x));
        store.push(EntryKind::Expr(raw_expr), "5*sqrt(x)/sqrt(x)".to_string());

        // Inject simplified cache: simplified = 5
        let simplified_five = ctx.num(5);
        let cache_key = SimplifyCacheKey::from_context(crate::domain::DomainMode::Generic);
        let cache = SimplifiedCache {
            key: cache_key.clone(),
            expr: simplified_five,
            requires: vec![],
            steps: Some(std::sync::Arc::new(vec![])),
        };
        store.update_simplified(1, cache);

        // Resolve #1 + 3
        let ref1 = ctx.add(Expr::SessionRef(1));
        let three = ctx.num(3);
        let input = ctx.add(Expr::Add(ref1, three));

        let result = resolve_session_refs_with_mode(
            &mut ctx,
            input,
            &store,
            RefMode::PreferSimplified,
            &cache_key,
        )
        .unwrap();

        // Should use cache
        assert!(result.used_cache, "Should have used cache");

        // Result should be 5 + 3 (order may vary), not the raw fraction
        if let Expr::Add(l, r) = ctx.get(result.expr) {
            // Extract both operands as numbers
            let left_num = match ctx.get(*l) {
                Expr::Number(n) => n.to_integer(),
                _ => panic!("Left should be Number"),
            };
            let right_num = match ctx.get(*r) {
                Expr::Number(n) => n.to_integer(),
                _ => panic!("Right should be Number"),
            };
            // Should contain 5 (from cache) and 3, order doesn't matter
            let has_five = left_num == 5.into() || right_num == 5.into();
            let has_three = left_num == 3.into() || right_num == 3.into();
            assert!(
                has_five,
                "Should contain 5 from cache, got {} and {}",
                left_num, right_num
            );
            assert!(
                has_three,
                "Should contain 3, got {} and {}",
                left_num, right_num
            );
        } else {
            panic!("Expected Add, got {:?}", ctx.get(result.expr));
        }
    }

    #[test]
    fn test_resolve_with_mode_cache_miss_key_mismatch() {
        use cas_ast::{Context, Expr};

        let mut ctx = Context::new();
        let mut store = SessionStore::new();

        // Store raw expr as #1
        let x = ctx.var("x");
        store.push(EntryKind::Expr(x), "x".to_string());

        // Inject cache with different domain mode
        let simplified = ctx.num(5);
        let cache_key_strict = SimplifyCacheKey::from_context(crate::domain::DomainMode::Strict);
        let cache = SimplifiedCache {
            key: cache_key_strict,
            expr: simplified,
            requires: vec![],
            steps: Some(std::sync::Arc::new(vec![])),
        };
        store.update_simplified(1, cache);

        // Resolve with Generic mode (different from Strict cache)
        let ref1 = ctx.add(Expr::SessionRef(1));
        let cache_key_generic = SimplifyCacheKey::from_context(crate::domain::DomainMode::Generic);

        let result = resolve_session_refs_with_mode(
            &mut ctx,
            ref1,
            &store,
            RefMode::PreferSimplified,
            &cache_key_generic,
        )
        .unwrap();

        // Should NOT use cache (key mismatch)
        assert!(
            !result.used_cache,
            "Should NOT have used cache due to key mismatch"
        );

        // Result should be raw x, not 5
        assert!(matches!(ctx.get(result.expr), Expr::Variable(name) if ctx.sym_name(*name) == "x"));
    }

    #[test]
    fn test_resolve_with_mode_raw_mode() {
        use cas_ast::{Context, Expr};

        let mut ctx = Context::new();
        let mut store = SessionStore::new();

        // Store raw expr as #1
        let x = ctx.var("x");
        store.push(EntryKind::Expr(x), "x".to_string());

        // Inject cache
        let simplified = ctx.num(5);
        let cache_key = SimplifyCacheKey::from_context(crate::domain::DomainMode::Generic);
        let cache = SimplifiedCache {
            key: cache_key.clone(),
            expr: simplified,
            requires: vec![],
            steps: Some(std::sync::Arc::new(vec![])),
        };
        store.update_simplified(1, cache);

        // Resolve with Raw mode - should ignore cache
        let ref1 = ctx.add(Expr::SessionRef(1));

        let result = resolve_session_refs_with_mode(
            &mut ctx,
            ref1,
            &store,
            RefMode::Raw, // Force raw mode
            &cache_key,
        )
        .unwrap();

        // Should NOT use cache (Raw mode)
        assert!(!result.used_cache, "Should NOT have used cache in Raw mode");

        // Result should be raw x
        assert!(matches!(ctx.get(result.expr), Expr::Variable(name) if ctx.sym_name(*name) == "x"));
    }

    #[test]
    fn test_resolve_with_mode_tracks_ref_chain() {
        use cas_ast::{Context, Expr};

        let mut ctx = Context::new();
        let mut store = SessionStore::new();

        // #1 = x
        let x = ctx.var("x");
        store.push(EntryKind::Expr(x), "x".to_string());

        // #2 = #1 + 1
        let ref1 = ctx.add(Expr::SessionRef(1));
        let one = ctx.num(1);
        let expr2 = ctx.add(Expr::Add(ref1, one));
        store.push(EntryKind::Expr(expr2), "#1 + 1".to_string());

        // Resolve #2
        let ref2 = ctx.add(Expr::SessionRef(2));
        let cache_key = SimplifyCacheKey::from_context(crate::domain::DomainMode::Generic);

        let result = resolve_session_refs_with_mode(
            &mut ctx,
            ref2,
            &store,
            RefMode::PreferSimplified,
            &cache_key,
        )
        .unwrap();

        // Should track both #2 and #1 in ref chain
        assert!(result.ref_chain.contains(&1), "Should track #1 in chain");
        assert!(result.ref_chain.contains(&2), "Should track #2 in chain");
    }
}
