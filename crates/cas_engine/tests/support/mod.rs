#![allow(dead_code)]

use cas_ast::ExprId;
use cas_engine::diagnostics::{Diagnostics, RequireOrigin, RequiredItem};
use cas_engine::eval::{
    CacheHitEntryId, EvalResolveError, EvalSession, EvalStore, SimplifiedCache, SimplifyCacheKey,
    StoredInputKind,
};
use cas_engine::options::EvalOptions;
use cas_engine::profile_cache::ProfileCache;

pub use cas_session_core::types::EntryKind;

fn map_eval_resolve_error(err: cas_session_core::types::ResolveError) -> EvalResolveError {
    match err {
        cas_session_core::types::ResolveError::NotFound(id) => EvalResolveError::NotFound(id),
        cas_session_core::types::ResolveError::CircularReference(id) => {
            EvalResolveError::CircularReference(id)
        }
    }
}

#[derive(Debug, Default)]
pub struct SessionEvalStore(cas_session_core::store::SessionStore<Diagnostics, SimplifiedCache>);

impl EvalStore for SessionEvalStore {
    fn push_raw_input(&mut self, kind: StoredInputKind, raw_input: String) -> u64 {
        let mapped = match kind {
            StoredInputKind::Expr(expr) => EntryKind::Expr(expr),
            StoredInputKind::Eq { lhs, rhs } => EntryKind::Eq { lhs, rhs },
        };
        self.0.push(mapped, raw_input)
    }

    fn touch_cached(&mut self, entry_id: u64) {
        self.0.touch_cached(entry_id);
    }

    fn update_diagnostics(&mut self, id: u64, diagnostics: Diagnostics) {
        self.0.update_diagnostics(id, diagnostics);
    }

    fn update_simplified(&mut self, id: u64, cache: SimplifiedCache) {
        self.0.update_simplified(id, cache);
    }
}

#[derive(Default, Debug)]
pub struct SessionState {
    store: SessionEvalStore,
    env: cas_session_core::env::Environment,
    options: EvalOptions,
    profile_cache: ProfileCache,
}

impl SessionState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn options_mut(&mut self) -> &mut EvalOptions {
        &mut self.options
    }

    pub fn history_push<S: Into<String>>(&mut self, kind: EntryKind, raw_text: S) -> u64 {
        self.store.0.push(kind, raw_text.into())
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

    fn resolve_all_with_diagnostics(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
    ) -> Result<(ExprId, Diagnostics, Vec<CacheHitEntryId>), EvalResolveError> {
        let cache_key = SimplifyCacheKey::from_context(self.options.shared.semantics.domain_mode);
        let mut lookup = |id: u64| {
            let entry = self.store.0.get(id)?;
            Some(cas_session_core::resolve::ModeEntry {
                kind: entry.kind.clone(),
                requires: entry.diagnostics.requires.clone(),
                cache: entry.simplified.as_ref().map(|cache| {
                    cas_session_core::resolve::ModeCacheEntry {
                        key: cache.key.clone(),
                        expr: cache.expr,
                        requires: cache.requires.clone(),
                    }
                }),
            })
        };
        let mut same_requirement = |lhs: &RequiredItem, rhs: &RequiredItem| lhs.cond == rhs.cond;
        let mut mark_session_propagated = |item: &mut RequiredItem| {
            item.merge_origin(RequireOrigin::SessionPropagated);
        };

        let resolved = cas_session_core::resolve::resolve_session_refs_with_mode_lookup(
            ctx,
            expr,
            cas_session_core::types::RefMode::PreferSimplified,
            &cache_key,
            &mut lookup,
            &mut same_requirement,
            &mut mark_session_propagated,
        )
        .map_err(map_eval_resolve_error)?;

        let mut inherited = Diagnostics::new();
        for item in resolved.requires {
            inherited.push_required(item.cond, RequireOrigin::SessionPropagated);
        }
        let fully_resolved = cas_session_core::env::substitute(ctx, &self.env, resolved.expr);
        let cache_hits = resolved
            .cache_hits
            .into_iter()
            .map(|h| h.entry_id)
            .collect();

        Ok((fully_resolved, inherited, cache_hits))
    }
}
