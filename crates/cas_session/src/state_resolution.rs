use cas_ast::ExprId;
use cas_solver_core::diagnostics_model::Diagnostics;

use crate::resolve_refs::{
    resolve_session_refs_with_env, resolve_session_refs_with_mode_and_diagnostics,
};
use crate::{cache::SimplifyCacheKey, state_core::SessionState};
use cas_session_core::types::ResolveError;

impl SessionState {
    /// Resolve session refs (`#N`) and environment bindings with current state.
    pub fn resolve_state_refs(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
    ) -> Result<ExprId, ResolveError> {
        resolve_session_refs_with_env(ctx, expr, &self.store, &self.env)
    }

    /// Resolve refs with inherited diagnostics and cache hit traces.
    pub fn resolve_state_refs_with_diagnostics(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
    ) -> Result<(ExprId, Diagnostics, Vec<u64>), ResolveError> {
        let cache_key = SimplifyCacheKey::from_context(self.options.shared.semantics.domain_mode);
        resolve_session_refs_with_mode_and_diagnostics(
            ctx,
            expr,
            &self.store,
            cas_session_core::types::RefMode::PreferSimplified,
            &cache_key,
            &self.env,
        )
    }
}
