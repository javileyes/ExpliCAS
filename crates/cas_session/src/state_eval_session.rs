use cas_ast::{Expr, ExprId};
use cas_session_core::{
    eval::{DirectCachedEval, EvalSession},
    resolve::parse_legacy_session_ref,
};
use cas_solver_core::diagnostics_model::{Diagnostics, RequireOrigin};
use cas_solver_core::eval_options::EvalOptions;

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

    fn try_direct_cached_eval(
        &mut self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        auto_store: bool,
    ) -> anyhow::Result<Option<DirectCachedEval<Diagnostics>>> {
        if auto_store {
            return Ok(None);
        }

        let entry_id = match ctx.get(expr) {
            Expr::SessionRef(id) => Some(*id),
            Expr::Variable(sym_id) => parse_legacy_session_ref(ctx.sym_name(*sym_id)),
            _ => None,
        };
        let Some(entry_id) = entry_id else {
            return Ok(None);
        };

        let current_key =
            crate::cache::SimplifyCacheKey::from_context(self.options.shared.semantics.domain_mode);
        let Some((cached_expr, cached_requires)) = ({
            let Some(entry) = self.store.get(entry_id) else {
                return Ok(None);
            };
            let Some(cache) = entry.simplified.as_ref() else {
                return Ok(None);
            };
            if !cache.key.is_compatible(&current_key) {
                return Ok(None);
            }
            Some((cache.expr, cache.requires.clone()))
        }) else {
            return Ok(None);
        };

        self.store.touch_cached(entry_id);

        let resolved = crate::env::substitute(ctx, &self.env, cached_expr);
        let mut inherited_diagnostics = Diagnostics::new();
        for item in &cached_requires {
            inherited_diagnostics
                .push_required(item.cond.clone(), RequireOrigin::SessionPropagated);
        }

        Ok(Some(DirectCachedEval {
            resolved,
            inherited_diagnostics,
        }))
    }
}
