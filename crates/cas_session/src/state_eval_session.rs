use cas_ast::ExprId;
use cas_session_core::eval::EvalSession;
use cas_solver::EvalOptions;
use cas_solver_core::diagnostics_model::Diagnostics;

use crate::{state_core::SessionState, state_eval_store::SessionEvalStore};

impl EvalSession for SessionState {
    type Store = SessionEvalStore;
    type Options = EvalOptions;
    type Diagnostics = Diagnostics;

    fn store_mut(&mut self) -> &mut Self::Store {
        &mut self.store
    }

    fn options(&self) -> &EvalOptions {
        &self.options
    }

    fn options_mut(&mut self) -> &mut EvalOptions {
        &mut self.options
    }

    fn resolve_all_with_diagnostics(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
    ) -> anyhow::Result<(ExprId, Diagnostics, Vec<u64>)> {
        self.resolve_state_refs_with_diagnostics(ctx, expr)
            .map_err(cas_session_core::eval::map_resolve_error_to_anyhow)
    }
}
