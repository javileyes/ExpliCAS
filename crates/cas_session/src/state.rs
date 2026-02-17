use cas_ast::ExprId;
use cas_engine::eval::EvalSession;
use cas_engine::options::EvalOptions;
use cas_engine::profile_cache::ProfileCache;
use cas_engine::session::{CacheHitTrace, ResolveError, SessionStore};
use cas_engine::{diagnostics::Diagnostics, poly_store::PolyStore};

use crate::Environment;

/// Bundled session state for portability (CLI/Web/FFI).
///
/// This crate-local type is the migration target for Phase 3.
/// `cas_engine` remains stateless and consumes it via the `EvalSession` trait.
#[derive(Default, Debug)]
pub struct SessionState {
    pub store: SessionStore,
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
            store: SessionStore::new(),
            env: Environment::new(),
            options: EvalOptions::default(),
            profile_cache: ProfileCache::new(),
            poly_store: PolyStore::new(),
        }
    }

    /// Create a state from an existing session store (for snapshot restoration).
    pub fn from_store(store: SessionStore) -> Self {
        Self {
            store,
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
    type Store = SessionStore;

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
    ) -> Result<ExprId, ResolveError> {
        cas_engine::session::resolve_all(ctx, expr, &self.store, &self.env)
    }

    fn resolve_all_with_diagnostics(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
    ) -> Result<(ExprId, Diagnostics, Vec<CacheHitTrace>), ResolveError> {
        cas_engine::session::resolve_all_with_diagnostics(
            ctx,
            expr,
            &self.store,
            &self.env,
            self.options.shared.semantics.domain_mode,
        )
    }
}
