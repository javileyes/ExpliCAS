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
    build_linear_collect_additive_execution_with, build_linear_collect_factored_execution_with,
    solve_linear_collect_additive_with_runtime, solve_linear_collect_execution_with_items,
    solve_linear_collect_factored_with_runtime, LinearCollectRuntime,
};
use cas_solver_core::linear_solution::build_linear_solution_set;

use crate::engine::Simplifier;
use crate::solver::SolveStep;

struct EngineLinearCollectRuntime<'a> {
    simplifier: &'a mut Simplifier,
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
    let solved = {
        let mut runtime = EngineLinearCollectRuntime { simplifier };
        solve_linear_collect_factored_with_runtime(&mut runtime, lhs, rhs, var)?
    };

    let mut steps = Vec::new();
    let solution_set = if simplifier.collect_steps() {
        let execution = build_linear_collect_factored_execution_with(
            &mut simplifier.context,
            var,
            solved.coeff,
            solved.rhs_term,
            solved.solution,
            solved.coeff_status,
            solved.rhs_status,
            |ctx, id| format!("{}", cas_formatter::DisplayExpr { context: ctx, id }),
        );
        let solved_execution = solve_linear_collect_execution_with_items(execution, |items, set| {
            for item in items {
                steps.push(SolveStep {
                    description: item.description().to_string(),
                    equation_after: item.equation,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            set
        });
        solved_execution.solved
    } else {
        build_linear_solution_set(
            solved.coeff,
            solved.rhs_term,
            solved.solution,
            solved.coeff_status,
            solved.rhs_status,
        )
    };

    Some((solution_set, steps))
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
    let solved = {
        let mut runtime = EngineLinearCollectRuntime { simplifier };
        solve_linear_collect_additive_with_runtime(&mut runtime, lhs, rhs, var)?
    };

    let mut steps = Vec::new();
    let solution_set = if simplifier.collect_steps() {
        let execution = build_linear_collect_additive_execution_with(
            &mut simplifier.context,
            var,
            solved.coeff,
            solved.constant,
            solved.solution,
            solved.coeff_status,
            solved.constant_status,
            |ctx, id| format!("{}", cas_formatter::DisplayExpr { context: ctx, id }),
        );
        let solved_execution = solve_linear_collect_execution_with_items(execution, |items, set| {
            for item in items {
                steps.push(SolveStep {
                    description: item.description().to_string(),
                    equation_after: item.equation,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            set
        });
        solved_execution.solved
    } else {
        build_linear_solution_set(
            solved.coeff,
            solved.constant,
            solved.solution,
            solved.coeff_status,
            solved.constant_status,
        )
    };

    Some((solution_set, steps))
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
