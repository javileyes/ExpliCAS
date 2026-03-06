use crate::Simplifier;
use cas_ast::ExprId;

impl cas_solver_core::solve_runtime_adapter_state_runtime::RuntimeSolveAdapterState for Simplifier {
    fn runtime_context(&self) -> &cas_ast::Context {
        &self.context
    }

    fn runtime_context_mut(&mut self) -> &mut cas_ast::Context {
        &mut self.context
    }

    fn runtime_simplify_expr(&mut self, expr: ExprId) -> ExprId {
        self.simplify(expr).0
    }

    fn runtime_expand_expr(&mut self, expr: ExprId) -> ExprId {
        crate::expand(&mut self.context, expr)
    }

    fn runtime_collect_steps(&self) -> bool {
        self.collect_steps()
    }

    fn runtime_prove_nonzero(&self, expr: ExprId) -> crate::Proof {
        crate::proof_runtime::prove_nonzero(&self.context, expr)
    }

    fn runtime_simplify_with_steps(
        &mut self,
        expr: ExprId,
    ) -> (ExprId, Vec<cas_solver_core::step_model::Step>) {
        self.simplify(expr)
    }
}
