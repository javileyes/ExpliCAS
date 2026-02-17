use crate::env::Environment;
use crate::options::EvalOptions;
use crate::poly_store::PolyStore;
use crate::profile_cache::ProfileCache;
use crate::session::{CacheHitTrace, ResolveError, SessionStore};
use cas_ast::{Context, ExprId};

/// bundled session state for portability (GUI/Web/CLI)
/// NOTE: Cannot derive Clone due to ProfileCache
#[derive(Default, Debug)]
pub struct SessionState {
    pub store: SessionStore,
    pub env: Environment,
    /// Evaluation options (branch mode, context mode, etc.)
    pub options: EvalOptions,
    /// Cached rule profiles for performance
    pub profile_cache: ProfileCache,
    /// Opaque polynomial storage for fast mod-p operations
    pub poly_store: PolyStore,
}

// Backwards compatibility: expose assumptions as alias
impl SessionState {
    pub fn assumptions(&self) -> &EvalOptions {
        &self.options
    }
}

impl SessionState {
    pub fn new() -> Self {
        Self {
            store: SessionStore::new(),
            env: Environment::new(),
            options: EvalOptions::default(),
            profile_cache: ProfileCache::new(),
            poly_store: PolyStore::new(),
        }
    }

    /// Create a SessionState from an existing SessionStore (for session restoration).
    pub fn from_store(store: SessionStore) -> Self {
        Self {
            store,
            env: Environment::new(),
            options: EvalOptions::default(),
            profile_cache: ProfileCache::new(),
            poly_store: PolyStore::new(),
        }
    }

    /// Get a reference to the session store (for snapshot serialization).
    pub fn store(&self) -> &SessionStore {
        &self.store
    }

    /// Resolve all references in an expression:
    /// 1. Resolve session references (#id) -> ExprId
    /// 2. Substitute environment variables (x=5) -> ExprId
    pub fn resolve_all(&self, ctx: &mut Context, expr: ExprId) -> Result<ExprId, ResolveError> {
        crate::session_resolution::resolve_all(ctx, expr, &self.store, &self.env)
    }

    /// Resolve all references AND return inherited diagnostics + cache hits.
    ///
    /// When the expression contains session references (#id), the diagnostics
    /// from those entries are accumulated for SessionPropagated origin tracking.
    ///
    /// V2.15.36: Uses cache-aware resolution - if an entry has a cached
    /// simplified result with matching key, uses that instead of raw expression.
    /// Also returns cache hit traces for synthetic timeline step generation.
    pub fn resolve_all_with_diagnostics(
        &self,
        ctx: &mut Context,
        expr: ExprId,
    ) -> Result<(ExprId, crate::diagnostics::Diagnostics, Vec<CacheHitTrace>), ResolveError> {
        crate::session_resolution::resolve_all_with_diagnostics(
            ctx,
            expr,
            &self.store,
            &self.env,
            self.options.shared.semantics.domain_mode,
        )
    }

    /// Clear all session state (history and environment)
    pub fn clear(&mut self) {
        self.store.clear();
        self.env.clear_all();
        // Note: assumptions are NOT cleared - user must explicitly change mode
    }
}
