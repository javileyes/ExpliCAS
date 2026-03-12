use cas_ast::ExprId;
use cas_engine::Step;
use cas_session_core::eval::EvalStore;
use cas_solver_core::diagnostics_model::{Diagnostics, RequiredItem};
use cas_solver_core::domain_mode::DomainMode;

use super::SessionEvalStore;
use crate::SimplifyCacheKey;

impl EvalStore for SessionEvalStore {
    type DomainMode = DomainMode;
    type RequiredItem = RequiredItem;
    type Step = Step;
    type Diagnostics = Diagnostics;

    fn push_raw_expr(&mut self, expr: ExprId, raw_input: String) -> u64 {
        self.push(crate::EntryKind::Expr(expr), raw_input)
    }

    fn push_raw_equation(&mut self, lhs: ExprId, rhs: ExprId, raw_input: String) -> u64 {
        self.push(crate::EntryKind::Eq { lhs, rhs }, raw_input)
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
            crate::SimplifiedCache {
                key: SimplifyCacheKey::from_context(domain),
                expr,
                requires,
                steps,
            },
        );
    }
}
