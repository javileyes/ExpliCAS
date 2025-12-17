use crate::env::Environment;
use crate::options::Assumptions;
use crate::session::{resolve_session_refs, ResolveError, SessionStore};
use cas_ast::{Context, ExprId};

/// bundled session state for portability (GUI/Web/CLI)
#[derive(Default, Debug, Clone)]
pub struct SessionState {
    pub store: SessionStore,
    pub env: Environment,
    /// Assumptions for evaluation (branch mode, etc.)
    pub assumptions: Assumptions,
}

impl SessionState {
    pub fn new() -> Self {
        Self {
            store: SessionStore::new(),
            env: Environment::new(),
            assumptions: Assumptions::default(),
        }
    }

    /// Resolve all references in an expression:
    /// 1. Resolve session references (#id) -> ExprId
    /// 2. Substitute environment variables (x=5) -> ExprId
    pub fn resolve_all(&self, ctx: &mut Context, expr: ExprId) -> Result<ExprId, ResolveError> {
        // 1. Resolve session refs (#1, #2...)
        let expr_with_refs = resolve_session_refs(ctx, expr, &self.store)?;

        // 2. Substitute variables from environment
        let fully_resolved = crate::env::substitute(ctx, &self.env, expr_with_refs);

        Ok(fully_resolved)
    }

    /// Clear all session state (history and environment)
    pub fn clear(&mut self) {
        self.store.clear();
        self.env.clear_all();
        // Note: assumptions are NOT cleared - user must explicitly change mode
    }
}
