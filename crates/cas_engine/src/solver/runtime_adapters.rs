use crate::engine::Simplifier;
use cas_ast::ExprId;
use cas_solver_core::linear_collect::LinearCollectRuntime;
use cas_solver_core::rational_roots::RationalRootsStrategyRuntime;

pub(crate) struct EngineLinearCollectRuntime<'a> {
    pub(crate) simplifier: &'a mut Simplifier,
}

impl LinearCollectRuntime for EngineLinearCollectRuntime<'_> {
    fn context(&mut self) -> &mut cas_ast::Context {
        &mut self.simplifier.context
    }

    fn simplify_expr(&mut self, expr: ExprId) -> ExprId {
        let (simplified, _) = self.simplifier.simplify(expr);
        simplified
    }

    fn prove_nonzero_status(
        &mut self,
        expr: ExprId,
    ) -> cas_solver_core::linear_solution::NonZeroStatus {
        crate::solver::proof_bridge::proof_to_nonzero_status(crate::helpers::prove_nonzero(
            &self.simplifier.context,
            expr,
        ))
    }
}

pub(crate) struct EngineRationalRootsRuntime<'a> {
    pub(crate) simplifier: &'a mut Simplifier,
}

impl RationalRootsStrategyRuntime for EngineRationalRootsRuntime<'_> {
    fn context(&mut self) -> &mut cas_ast::Context {
        &mut self.simplifier.context
    }

    fn simplify_expr(&mut self, expr: ExprId) -> ExprId {
        let (simplified, _) = self.simplifier.simplify(expr);
        simplified
    }

    fn expand_expr(&mut self, expr: ExprId) -> ExprId {
        crate::expand::expand(&mut self.simplifier.context, expr)
    }
}
