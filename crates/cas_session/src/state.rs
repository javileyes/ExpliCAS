use cas_ast::ExprId;
use cas_engine::diagnostics::Diagnostics;
use cas_engine::eval::{CacheHitEntryId, EvalSession, EvalStore, StoredInputKind};
use cas_engine::options::EvalOptions;
use cas_engine::profile_cache::ProfileCache;

use crate::{
    snapshot::SessionSnapshot, Environment, ResolveError, SessionStore, SimplifyCacheKey,
    SnapshotError,
};

fn map_eval_resolve_error(err: ResolveError) -> cas_engine::eval::EvalResolveError {
    match err {
        ResolveError::NotFound(id) => cas_engine::eval::EvalResolveError::NotFound(id),
        ResolveError::CircularReference(id) => {
            cas_engine::eval::EvalResolveError::CircularReference(id)
        }
    }
}

/// Local adapter that bridges `cas_session::SessionStore` to `cas_engine::eval::EvalStore`.
#[derive(Debug, Default)]
pub struct SessionEvalStore(pub SessionStore);

impl SessionEvalStore {
    fn new() -> Self {
        Self(SessionStore::new())
    }

    fn from_store(store: SessionStore) -> Self {
        Self(store)
    }
}

impl std::ops::Deref for SessionEvalStore {
    type Target = SessionStore;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for SessionEvalStore {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl EvalStore for SessionEvalStore {
    fn push_raw_input(&mut self, kind: StoredInputKind, raw_input: String) -> u64 {
        let mapped = match kind {
            StoredInputKind::Expr(expr) => crate::EntryKind::Expr(expr),
            StoredInputKind::Eq { lhs, rhs } => crate::EntryKind::Eq { lhs, rhs },
        };
        self.0.push(mapped, raw_input)
    }

    fn touch_cached(&mut self, entry_id: u64) {
        self.0.touch_cached(entry_id);
    }

    fn update_diagnostics(&mut self, id: u64, diagnostics: Diagnostics) {
        self.0.update_diagnostics(id, diagnostics);
    }

    fn update_simplified(&mut self, id: u64, cache: cas_engine::eval::SimplifiedCache) {
        self.0.update_simplified(id, cache);
    }
}

/// Bundled session state for portability (CLI/Web/FFI).
///
/// This crate-local type is the migration target for Phase 3.
/// `cas_engine` remains stateless and consumes it via the `EvalSession` trait.
#[derive(Default, Debug)]
pub struct SessionState {
    store: SessionEvalStore,
    env: Environment,
    options: EvalOptions,
    profile_cache: ProfileCache,
}

impl SessionState {
    pub fn assumptions(&self) -> &EvalOptions {
        &self.options
    }

    pub fn options(&self) -> &EvalOptions {
        &self.options
    }

    pub fn options_mut(&mut self) -> &mut EvalOptions {
        &mut self.options
    }

    pub fn new() -> Self {
        Self {
            store: SessionEvalStore::new(),
            env: Environment::new(),
            options: EvalOptions::default(),
            profile_cache: ProfileCache::new(),
        }
    }

    /// Create a state from an existing session store (for snapshot restoration).
    fn from_store(store: SessionStore) -> Self {
        Self {
            store: SessionEvalStore::from_store(store),
            env: Environment::new(),
            options: EvalOptions::default(),
            profile_cache: ProfileCache::new(),
        }
    }

    /// Restore context + state from a persisted snapshot.
    fn from_snapshot(snapshot: SessionSnapshot) -> (cas_ast::Context, Self) {
        let (context, store) = snapshot.into_parts();
        (context, Self::from_store(store))
    }

    /// Load a snapshot from disk and restore it only if compatible with `cache_key`.
    pub fn load_compatible_snapshot(
        path: &std::path::Path,
        cache_key: &SimplifyCacheKey,
    ) -> Result<Option<(cas_ast::Context, Self)>, SnapshotError> {
        let snapshot = SessionSnapshot::load(path)?;
        if !snapshot.is_compatible(cache_key) {
            return Ok(None);
        }
        Ok(Some(Self::from_snapshot(snapshot)))
    }

    pub fn history_entries(&self) -> &[crate::Entry] {
        self.store.list()
    }

    pub fn history_push<S: Into<String>>(
        &mut self,
        kind: crate::EntryKind,
        raw_text: S,
    ) -> crate::EntryId {
        self.store.push(kind, raw_text.into())
    }

    pub fn history_get(&self, id: crate::EntryId) -> Option<&crate::Entry> {
        self.store.get(id)
    }

    pub fn history_len(&self) -> usize {
        self.store.len()
    }

    pub fn history_remove(&mut self, ids: &[crate::EntryId]) {
        self.store.remove(ids);
    }

    /// Resolve session refs (`#N`) and environment bindings with current state.
    pub fn resolve_state_refs(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
    ) -> Result<ExprId, ResolveError> {
        crate::resolve_all(ctx, expr, &self.store, &self.env)
    }

    /// Resolve refs with inherited diagnostics and cache hit traces.
    pub fn resolve_state_refs_with_diagnostics(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
    ) -> Result<(ExprId, Diagnostics, Vec<CacheHitEntryId>), ResolveError> {
        crate::resolve_all_with_diagnostics(
            ctx,
            expr,
            &self.store,
            &self.env,
            self.options.shared.semantics.domain_mode,
        )
    }

    /// Build a serializable snapshot from the current state.
    fn snapshot(&self, context: &cas_ast::Context, cache_key: SimplifyCacheKey) -> SessionSnapshot {
        SessionSnapshot::new(context, &self.store, cache_key)
    }

    /// Persist the current state atomically to disk.
    pub fn save_snapshot(
        &self,
        context: &cas_ast::Context,
        path: &std::path::Path,
        cache_key: SimplifyCacheKey,
    ) -> Result<(), SnapshotError> {
        self.snapshot(context, cache_key).save_atomic(path)
    }

    pub fn get_binding(&self, name: &str) -> Option<ExprId> {
        self.env.get(name)
    }

    pub fn set_binding<S: Into<String>>(&mut self, name: S, expr: ExprId) {
        self.env.set(name.into(), expr);
    }

    pub fn unset_binding(&mut self, name: &str) -> bool {
        self.env.unset(name)
    }

    pub fn bindings(&self) -> Vec<(String, ExprId)> {
        self.env
            .list()
            .into_iter()
            .map(|(name, expr)| (name.to_string(), expr))
            .collect()
    }

    pub fn binding_count(&self) -> usize {
        self.env.len()
    }

    pub fn clear_bindings(&mut self) {
        self.env.clear_all();
    }

    pub fn profile_cache_len(&self) -> usize {
        self.profile_cache.len()
    }

    pub fn clear_profile_cache(&mut self) {
        self.profile_cache.clear();
    }

    /// Clear all session data (history + env bindings).
    /// Note: options are intentionally preserved.
    pub fn clear(&mut self) {
        self.store.clear();
        self.env.clear_all();
    }
}

impl EvalSession for SessionState {
    type Store = SessionEvalStore;

    fn store_mut(&mut self) -> &mut Self::Store {
        &mut self.store
    }

    fn options(&self) -> &EvalOptions {
        &self.options
    }

    fn profile_cache_mut(&mut self) -> &mut ProfileCache {
        &mut self.profile_cache
    }

    fn resolve_all(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
    ) -> Result<ExprId, cas_engine::eval::EvalResolveError> {
        self.resolve_state_refs(ctx, expr)
            .map_err(map_eval_resolve_error)
    }

    fn resolve_all_with_diagnostics(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
    ) -> Result<(ExprId, Diagnostics, Vec<CacheHitEntryId>), cas_engine::eval::EvalResolveError>
    {
        self.resolve_state_refs_with_diagnostics(ctx, expr)
            .map_err(map_eval_resolve_error)
    }
}
