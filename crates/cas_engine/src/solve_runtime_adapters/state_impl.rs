use crate::Simplifier;
use cas_ast::{Context, ExprId};

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
        cas_solver_core::proof_runtime_bound_runtime::prove_nonzero_with_runtime_proof_simplifier::<
            crate::Simplifier,
        >(&self.context, expr)
    }

    fn runtime_prove_positive(
        &self,
        expr: ExprId,
        value_domain: cas_solver_core::value_domain::ValueDomain,
    ) -> crate::Proof {
        cas_solver_core::proof_runtime_bound_runtime::prove_positive_with_runtime_proof_simplifier::<
            crate::Simplifier,
        >(&self.context, expr, value_domain)
    }

    fn runtime_infer_implicit_domain(
        &self,
        expr: ExprId,
        value_domain: cas_solver_core::value_domain::ValueDomain,
    ) -> cas_solver_core::solve_runtime_types::RuntimeImplicitDomain {
        crate::infer_implicit_domain(&self.context, expr, value_domain)
    }

    fn runtime_derive_requires_from_equation(
        &self,
        lhs: ExprId,
        rhs: ExprId,
        existing: &cas_solver_core::solve_runtime_types::RuntimeImplicitDomain,
        value_domain: cas_solver_core::value_domain::ValueDomain,
    ) -> Vec<cas_solver_core::solve_runtime_types::RuntimeImplicitCondition> {
        crate::derive_requires_from_equation(&self.context, lhs, rhs, existing, value_domain)
    }

    fn runtime_cancel_additive_terms_semantic(
        &mut self,
        lhs: ExprId,
        rhs: ExprId,
    ) -> Option<(ExprId, ExprId)> {
        crate::cancel_runtime::cancel_additive_terms_semantic(self, lhs, rhs)
            .map(|rewrite| (rewrite.new_lhs, rewrite.new_rhs))
    }

    fn runtime_simplify_with_steps(
        &mut self,
        expr: ExprId,
    ) -> (ExprId, Vec<cas_solver_core::step_model::Step>) {
        self.simplify(expr)
    }
}

impl cas_solver_core::proof_runtime_bound_runtime::RuntimeProofSimplifierFactory for Simplifier {
    fn runtime_proof_with_context(ctx: Context) -> Self {
        Self::with_context(ctx)
    }

    fn runtime_proof_set_collect_steps(&mut self, collect: bool) {
        self.set_collect_steps(collect);
    }

    fn runtime_proof_simplify_with_options_expr(
        &mut self,
        expr: ExprId,
        opts: cas_solver_core::simplify_options::SimplifyOptions,
    ) -> ExprId {
        self.simplify_with_stats(expr, opts).0
    }

    fn runtime_proof_into_context(self) -> Context {
        self.context
    }
}
