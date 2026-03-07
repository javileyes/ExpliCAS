use cas_ast::ExprId;
use cas_session_core::eval::EvalStore;
use cas_solver::Diagnostics;

use super::SessionEvalStore;
use crate::SimplifyCacheKey;

impl EvalStore for SessionEvalStore {
    type DomainMode = cas_solver::DomainMode;
    type RequiredItem = cas_solver::RequiredItem;
    type Step = cas_solver::Step;
    type Diagnostics = Diagnostics;

    fn push_raw_expr(&mut self, expr: ExprId, raw_input: String) -> u64 {
        self.0.push(crate::EntryKind::Expr(expr), raw_input)
    }

    fn push_raw_equation(&mut self, lhs: ExprId, rhs: ExprId, raw_input: String) -> u64 {
        self.0.push(crate::EntryKind::Eq { lhs, rhs }, raw_input)
    }

    fn touch_cached(&mut self, entry_id: u64) {
        self.0.touch_cached(entry_id);
    }

    fn update_diagnostics(&mut self, id: u64, diagnostics: Diagnostics) {
        self.0.update_diagnostics(id, diagnostics);
    }

    fn update_simplified(
        &mut self,
        id: u64,
        domain: cas_solver::DomainMode,
        expr: ExprId,
        requires: Vec<cas_solver::RequiredItem>,
        steps: Option<std::sync::Arc<Vec<cas_solver::Step>>>,
    ) {
        self.0.update_simplified(
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
