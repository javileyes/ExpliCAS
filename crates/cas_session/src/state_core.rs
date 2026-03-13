use cas_ast::ExprId;
use cas_engine::Step;
use cas_session_core::eval::EvalStore;
use cas_solver_core::diagnostics_model::{Diagnostics, RequiredItem};
use cas_solver_core::domain_mode::DomainMode;
use cas_solver_core::eval_options::EvalOptions;

use crate::cache::SimplifyCacheKey;
use crate::env::Environment;

mod bindings;
mod constructors;
mod history;
mod runtime;
mod snapshot;

/// Local adapter that bridges `cas_session::SessionStore` to `cas_session_core::EvalStore`.
#[derive(Debug, Default)]
pub struct SessionEvalStore {
    pub(crate) store: crate::SessionStore,
    dirty: bool,
}

impl SessionEvalStore {
    pub(crate) fn new() -> Self {
        Self {
            store: crate::store_cache_policy::session_store_with_cache_config(
                cas_session_core::types::CacheConfig::default(),
            ),
            dirty: false,
        }
    }

    pub(crate) fn from_store(store: crate::SessionStore) -> Self {
        Self {
            store,
            dirty: false,
        }
    }

    pub(crate) fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub(crate) fn mark_clean(&mut self) {
        self.dirty = false;
    }

    pub(crate) fn push(
        &mut self,
        kind: cas_session_core::types::EntryKind,
        raw_text: String,
    ) -> u64 {
        self.dirty = true;
        self.store.push(kind, raw_text)
    }

    pub(crate) fn clear(&mut self) {
        if !self.store.is_empty() {
            self.dirty = true;
        }
        self.store.clear();
    }

    pub(crate) fn remove(&mut self, ids: &[u64]) {
        let before = self.store.len();
        self.store.remove(ids);
        if self.store.len() != before {
            self.dirty = true;
        }
    }

    pub(crate) fn update_diagnostics(&mut self, id: u64, diagnostics: Diagnostics) {
        self.dirty = true;
        self.store.update_diagnostics(id, diagnostics);
    }

    pub(crate) fn update_simplified(&mut self, id: u64, simplified: crate::cache::SimplifiedCache) {
        self.dirty = true;
        self.store.update_simplified(id, simplified);
    }

    pub(crate) fn touch_cached(&mut self, entry_id: u64) {
        self.store.touch_cached(entry_id);
    }
}

impl std::ops::Deref for SessionEvalStore {
    type Target = crate::SessionStore;

    fn deref(&self) -> &Self::Target {
        &self.store
    }
}

impl EvalStore for SessionEvalStore {
    type DomainMode = DomainMode;
    type RequiredItem = RequiredItem;
    type Step = Step;
    type Diagnostics = Diagnostics;

    fn push_raw_expr(&mut self, expr: ExprId, raw_input: String) -> u64 {
        self.push(cas_session_core::types::EntryKind::Expr(expr), raw_input)
    }

    fn push_raw_equation(&mut self, lhs: ExprId, rhs: ExprId, raw_input: String) -> u64 {
        self.push(
            cas_session_core::types::EntryKind::Eq { lhs, rhs },
            raw_input,
        )
    }

    fn touch_cached(&mut self, entry_id: u64) {
        self.touch_cached(entry_id);
    }

    fn update_diagnostics(&mut self, id: u64, diagnostics: Diagnostics) {
        self.update_diagnostics(id, diagnostics);
    }

    fn update_simplified(
        &mut self,
        id: u64,
        domain: DomainMode,
        expr: ExprId,
        requires: Vec<RequiredItem>,
        steps: Option<std::sync::Arc<Vec<Step>>>,
    ) {
        self.update_simplified(
            id,
            crate::cache::SimplifiedCache {
                key: SimplifyCacheKey::from_context(domain),
                expr,
                requires,
                steps,
            },
        );
    }
}

/// Bundled session state for portability (CLI/Web/FFI).
///
/// This crate-local type is the migration target for Phase 3.
/// `cas_engine` remains stateless and consumes it via the shared `EvalSession` trait.
#[derive(Default, Debug)]
pub struct SessionState {
    pub(crate) store: SessionEvalStore,
    pub(crate) env: Environment,
    pub(crate) options: EvalOptions,
    pub(crate) dirty: bool,
}
