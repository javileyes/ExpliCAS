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

use cas_ast::{Expr, ExprId, SolutionSet};
use cas_solver_core::isolation_utils::contains_var;
use cas_solver_core::linear_didactic::{
    build_linear_collect_additive_steps_with, build_linear_collect_factored_steps_with,
};
use cas_solver_core::linear_solution::{build_linear_solution_set, derive_linear_nonzero_statuses};
use cas_solver_core::linear_terms::{build_sum, decompose_linear_collect_terms};

use crate::engine::Simplifier;
use crate::helpers::prove_nonzero;
use crate::solver::proof_bridge::proof_to_nonzero_status;
use crate::solver::SolveStep;

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
    let ctx = &mut simplifier.context;

    // 1. Build expr = lhs - rhs (move everything to LHS, so expr = 0)
    let expr = ctx.add(Expr::Sub(lhs, rhs));
    let (expr, _) = simplifier.simplify(expr);

    // 2. Decompose signed terms into linear and constant parts.
    let decomposition = decompose_linear_collect_terms(&mut simplifier.context, expr, var)?;
    let coeff_parts = decomposition.coeff_parts;
    let const_parts = decomposition.const_parts;

    // 4. Build coeff = sum of linear coefficients
    let coeff = build_sum(&mut simplifier.context, &coeff_parts);
    let (coeff, _) = simplifier.simplify(coeff);

    // 5. Build const = sum of constant parts (with sign flipped for solution)
    // coeff*var + const = 0 → var = -const / coeff
    let const_sum = build_sum(&mut simplifier.context, &const_parts);
    let neg_const = simplifier.context.add(Expr::Neg(const_sum));
    let (neg_const, _) = simplifier.simplify(neg_const);

    // 6. Build solution: var = -const / coeff
    let solution = simplifier.context.add(Expr::Div(neg_const, coeff));
    let (solution, _) = simplifier.simplify(solution);

    // 7. Build step description
    let mut steps = Vec::new();
    if simplifier.collect_steps() {
        let didactic = build_linear_collect_factored_steps_with(
            &mut simplifier.context,
            var,
            coeff,
            neg_const,
            solution,
            |ctx, id| format!("{}", cas_formatter::DisplayExpr { context: ctx, id }),
        );
        steps.push(SolveStep {
            description: didactic.collect.description,
            equation_after: didactic.collect.equation_after,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
        steps.push(SolveStep {
            description: didactic.divide.description,
            equation_after: didactic.divide.equation_after,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }

    // 8. Derive proof statuses for coefficient/constant degeneracy checks.
    // Keep previous behavior: only attempt proof when coefficient is var-free.
    let (coef_status, constant_status) = derive_linear_nonzero_statuses(
        contains_var(&simplifier.context, coeff, var),
        coeff,
        neg_const,
        |expr| proof_to_nonzero_status(prove_nonzero(&simplifier.context, expr)),
    );

    let solution_set =
        build_linear_solution_set(coeff, neg_const, solution, coef_status, constant_status);

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
    let kernel = cas_solver_core::linear_kernel::derive_linear_solve_kernel(
        &mut simplifier.context,
        lhs,
        rhs,
        var,
    )?;

    // Simplify for cleaner display
    let (coef, _) = simplifier.simplify(kernel.coef);
    let (constant, _) = simplifier.simplify(kernel.constant);

    // Solution: var = -constant / coef  (from coef*var + constant = 0)
    let neg_constant = simplifier.context.add(Expr::Neg(constant));
    let solution = simplifier.context.add(Expr::Div(neg_constant, coef));
    let (solution, _) = simplifier.simplify(solution);

    // Build steps
    let mut steps = Vec::new();

    if simplifier.collect_steps() {
        let didactic = build_linear_collect_additive_steps_with(
            &mut simplifier.context,
            var,
            coef,
            constant,
            solution,
            |ctx, id| format!("{}", cas_formatter::DisplayExpr { context: ctx, id }),
        );
        steps.push(SolveStep {
            description: didactic.collect.description,
            equation_after: didactic.collect.equation_after,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
        steps.push(SolveStep {
            description: didactic.divide.description,
            equation_after: didactic.divide.equation_after,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }

    // Derive proof statuses for coefficient/constant degeneracy checks.
    // Keep previous behavior: only attempt proof when coefficient is var-free.
    let (coef_status, constant_status) = derive_linear_nonzero_statuses(
        contains_var(&simplifier.context, coef, var),
        coef,
        constant,
        |expr| proof_to_nonzero_status(prove_nonzero(&simplifier.context, expr)),
    );

    let solution_set =
        build_linear_solution_set(coef, constant, solution, coef_status, constant_status);

    Some((solution_set, steps))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Context;
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
