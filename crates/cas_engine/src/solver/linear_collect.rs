//! Linear Collect Strategy for solving equations with additive terms.
//!
//! This module handles equations where the target variable appears in multiple
//! additive terms, like A = P + P*r*t. It factors out the variable and solves
//! by division, returning a Conditional solution when the coefficient might be zero.
//!
//! Example: A = P + P*r*t
//! 1. Move all to LHS: P + P*r*t - A = 0
//! 2. Factor: P*(1 + r*t) - A = 0
//! 3. Solve: P = A / (1 + r*t)  [guard: 1 + r*t ≠ 0]

use cas_ast::{ExprId, SolutionSet};
use cas_solver_core::linear_collect::{
    solve_linear_collect_additive_pipeline_with_runtime_and_items_runtime,
    solve_linear_collect_factored_pipeline_with_runtime_and_items_runtime,
    LinearCollectDidacticRuntime, LinearCollectExecutionItem, LinearCollectRuntime,
};

use crate::engine::Simplifier;
use crate::solver::{medium_step, render_expr, SolveStep};

impl LinearCollectRuntime for Simplifier {
    fn context(&mut self) -> &mut cas_ast::Context {
        &mut self.context
    }

    fn simplify_expr(&mut self, expr: ExprId) -> ExprId {
        self.simplify(expr).0
    }

    fn prove_nonzero_status(
        &mut self,
        expr: ExprId,
    ) -> cas_solver_core::linear_solution::NonZeroStatus {
        crate::solver::prove_nonzero_status(&self.context, expr)
    }
}

struct LinearCollectDidacticMapper;

impl LinearCollectDidacticRuntime<SolveStep> for LinearCollectDidacticMapper {
    fn render_expr(&mut self, ctx: &cas_ast::Context, expr: ExprId) -> String {
        render_expr(ctx, expr)
    }

    fn map_item_to_step(&mut self, item: LinearCollectExecutionItem) -> SolveStep {
        medium_step(item.description().to_string(), item.equation)
    }
}

/// Try to solve a linear equation where variable appears in multiple additive terms.
///
/// Returns Some((SolutionSet, steps)) if successful, None if not applicable.
///
/// Example: P + P*r*t - A = 0 → P = A / (1 + r*t) with guard 1+r*t ≠ 0
pub(crate) fn try_linear_collect(
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<(SolutionSet, Vec<SolveStep>)> {
    let include_items = simplifier.collect_steps();
    let mut didactic_runtime = LinearCollectDidacticMapper;
    solve_linear_collect_factored_pipeline_with_runtime_and_items_runtime(
        simplifier,
        lhs,
        rhs,
        var,
        include_items,
        &mut didactic_runtime,
    )
}

/// Try to solve using the structural linear form extractor.
///
/// This is an alternative to the term-based approach that works
/// better for expressions like `y*(1+x)` where the coefficient
/// is itself an expression.
pub(crate) fn try_linear_collect_v2(
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<(SolutionSet, Vec<SolveStep>)> {
    let include_items = simplifier.collect_steps();
    let mut didactic_runtime = LinearCollectDidacticMapper;
    solve_linear_collect_additive_pipeline_with_runtime_and_items_runtime(
        simplifier,
        lhs,
        rhs,
        var,
        include_items,
        &mut didactic_runtime,
    )
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, Expr};
    use cas_solver_core::linear_terms::decompose_linear_collect_terms;
    use cas_solver_core::linear_terms::{split_linear_term, TermClass};

    #[test]
    fn test_add_terms_signed() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");

        // a + b + c
        let ab = ctx.add(Expr::Add(a, b));
        let abc = ctx.add(Expr::Add(ab, c));

        let decomp = decompose_linear_collect_terms(&mut ctx, abc, "x");
        assert!(decomp.is_none(), "no linear terms for variable x");
    }

    #[test]
    fn test_split_linear_term_const() {
        let mut ctx = Context::new();
        let a = ctx.var("A");

        match split_linear_term(&mut ctx, a, "P") {
            TermClass::Const(_) => {}
            _ => panic!("A should be Const with respect to P"),
        }
    }

    #[test]
    fn test_split_linear_term_var() {
        let mut ctx = Context::new();
        let p = ctx.var("P");

        match split_linear_term(&mut ctx, p, "P") {
            TermClass::Linear(_) => {}
            _ => panic!("P should be Linear(1) with respect to P"),
        }
    }
}
