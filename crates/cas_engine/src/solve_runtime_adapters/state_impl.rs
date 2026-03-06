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

    fn runtime_expand_full_expr(&mut self, expr: ExprId) -> ExprId {
        self.expand(expr).0
    }

    fn runtime_collect_steps(&self) -> bool {
        self.collect_steps()
    }

    fn runtime_set_collect_steps(&mut self, collect: bool) {
        self.set_collect_steps(collect);
    }

    fn runtime_simplify_for_solve(&mut self, expr: ExprId) -> ExprId {
        self.simplify_for_solve(expr)
    }

    fn runtime_simplify_with_options_expr(
        &mut self,
        expr: ExprId,
        options: cas_solver_core::simplify_options::SimplifyOptions,
    ) -> ExprId {
        self.simplify_with_options(expr, options).0
    }

    fn runtime_are_equivalent(&mut self, lhs: ExprId, rhs: ExprId) -> bool {
        self.are_equivalent(lhs, rhs)
    }

    fn runtime_clear_blocked_hints(&mut self) {
        crate::clear_blocked_hints();
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
