use crate::env::Environment;
use crate::options::EvalOptions;
use crate::poly_store::PolyStore;
use crate::profile_cache::ProfileCache;
use crate::session::SessionStore;

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

    /// Clear all session state (history and environment)
    pub fn clear(&mut self) {
        self.store.clear();
        self.env.clear_all();
        // Note: assumptions are NOT cleared - user must explicitly change mode
    }
}
