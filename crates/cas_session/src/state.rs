use cas_ast::ExprId;
use cas_engine::eval::{EvalSession, EvalStore};
use cas_engine::options::EvalOptions;
use cas_engine::profile_cache::ProfileCache;
use cas_engine::{diagnostics::Diagnostics, poly_store::PolyStore};

use crate::{CacheHitTrace as SessionCacheHitTrace, Environment, ResolveError, SessionStore};

fn map_resolve_error(err: ResolveError) -> cas_engine::eval::EvalResolveError {
    match err {
        ResolveError::NotFound(id) => cas_engine::eval::EvalResolveError::NotFound(id),
        ResolveError::CircularReference(id) => {
            cas_engine::eval::EvalResolveError::CircularReference(id)
        }
    }
}

fn map_cache_hit_trace(hit: SessionCacheHitTrace) -> cas_engine::eval::CacheHitTrace {
    cas_engine::eval::CacheHitTrace {
        entry_id: hit.entry_id,
        before_ref_expr: hit.before_ref_expr,
        after_expr: hit.after_expr,
        requires: hit.requires,
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
    fn push_raw_input(&mut self, ctx: &cas_ast::Context, parsed: ExprId, raw_input: String) -> u64 {
        let kind = if let Some((lhs, rhs)) = cas_ast::eq::unwrap_eq(ctx, parsed) {
            crate::EntryKind::Eq { lhs, rhs }
        } else {
            crate::EntryKind::Expr(parsed)
        };
        self.0.push(kind, raw_input)
    }

    fn touch_cached(&mut self, entry_id: u64) {
        self.0.touch_cached(entry_id);
    }

    fn update_diagnostics(&mut self, id: u64, diagnostics: Diagnostics) {
        self.0.update_diagnostics(id, diagnostics);
    }

    fn update_simplified(&mut self, id: u64, cache: cas_engine::eval::SimplifiedCache) {
        let mapped_cache = crate::SimplifiedCache {
            key: crate::SimplifyCacheKey {
                domain: cache.key.domain,
                ruleset_rev: cache.key.ruleset_rev,
            },
            expr: cache.expr,
            requires: cache.requires,
            steps: cache.steps,
        };
        self.0.update_simplified(id, mapped_cache);
    }
}

/// Bundled session state for portability (CLI/Web/FFI).
///
/// This crate-local type is the migration target for Phase 3.
/// `cas_engine` remains stateless and consumes it via the `EvalSession` trait.
#[derive(Default, Debug)]
pub struct SessionState {
    pub store: SessionEvalStore,
    pub env: Environment,
    pub options: EvalOptions,
    pub profile_cache: ProfileCache,
    pub poly_store: PolyStore,
}

impl SessionState {
    pub fn assumptions(&self) -> &EvalOptions {
        &self.options
    }

    pub fn new() -> Self {
        Self {
            store: SessionEvalStore::new(),
            env: Environment::new(),
            options: EvalOptions::default(),
            profile_cache: ProfileCache::new(),
            poly_store: PolyStore::new(),
        }
    }

    /// Create a state from an existing session store (for snapshot restoration).
    pub fn from_store(store: SessionStore) -> Self {
        Self {
            store: SessionEvalStore::from_store(store),
            env: Environment::new(),
            options: EvalOptions::default(),
            profile_cache: ProfileCache::new(),
            poly_store: PolyStore::new(),
        }
    }

    pub fn store(&self) -> &SessionStore {
        &self.store
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
        crate::resolve_all(ctx, expr, &self.store, &self.env).map_err(map_resolve_error)
    }

    fn resolve_all_with_diagnostics(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
    ) -> Result<
        (ExprId, Diagnostics, Vec<cas_engine::eval::CacheHitTrace>),
        cas_engine::eval::EvalResolveError,
    > {
        crate::resolve_all_with_diagnostics(
            ctx,
            expr,
            &self.store,
            &self.env,
            self.options.shared.semantics.domain_mode,
        )
        .map(|(resolved, diagnostics, cache_hits)| {
            (
                resolved,
                diagnostics,
                cache_hits.into_iter().map(map_cache_hit_trace).collect(),
            )
        })
        .map_err(map_resolve_error)
    }
}
