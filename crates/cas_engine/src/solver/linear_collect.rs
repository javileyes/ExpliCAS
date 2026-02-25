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
use cas_solver_core::linear_collect::{
    build_linear_collect_additive_execution_with, build_linear_collect_factored_execution_with,
    build_linear_collect_solution_expr, derive_linear_collect_additive_kernel,
    derive_linear_collect_factored_kernel, execute_linear_collect_additive_pipeline_with_and_items,
    execute_linear_collect_factored_pipeline_with_and_items, LinearCollectExecutionItem,
};

use crate::engine::Simplifier;
use crate::solver::{medium_step, render_expr, SolveStep};

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
    let runtime_cell = std::cell::RefCell::new(simplifier);
    execute_linear_collect_factored_pipeline_with_and_items(
        lhs,
        rhs,
        var,
        include_items,
        |lhs_expr, rhs_expr| {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            simplifier_ref.context.add(Expr::Sub(lhs_expr, rhs_expr))
        },
        |expr| {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            simplifier_ref.simplify(expr).0
        },
        |expr, inner_name| {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            derive_linear_collect_factored_kernel(&mut simplifier_ref.context, expr, inner_name)
        },
        |numerator, coeff| {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            build_linear_collect_solution_expr(&mut simplifier_ref.context, numerator, coeff)
        },
        |expr, inner_name| {
            let simplifier_ref = runtime_cell.borrow();
            contains_var(&simplifier_ref.context, expr, inner_name)
        },
        |expr| {
            let simplifier_ref = runtime_cell.borrow();
            crate::solver::prove_nonzero_status(&simplifier_ref.context, expr)
        },
        |name, solved| {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            build_linear_collect_factored_execution_with(
                &mut simplifier_ref.context,
                name,
                solved.coeff,
                solved.rhs_term,
                solved.solution,
                solved.coeff_status,
                solved.rhs_status,
                render_expr,
            )
        },
        |item: LinearCollectExecutionItem| {
            medium_step(item.description().to_string(), item.equation)
        },
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
    let runtime_cell = std::cell::RefCell::new(simplifier);
    execute_linear_collect_additive_pipeline_with_and_items(
        lhs,
        rhs,
        var,
        include_items,
        |lhs_expr, rhs_expr, inner_name| {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            derive_linear_collect_additive_kernel(
                &mut simplifier_ref.context,
                lhs_expr,
                rhs_expr,
                inner_name,
            )
        },
        |expr| {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            simplifier_ref.simplify(expr).0
        },
        |constant, coeff| {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            let neg_constant = simplifier_ref.context.add(Expr::Neg(constant));
            build_linear_collect_solution_expr(&mut simplifier_ref.context, neg_constant, coeff)
        },
        |expr, inner_name| {
            let simplifier_ref = runtime_cell.borrow();
            contains_var(&simplifier_ref.context, expr, inner_name)
        },
        |expr| {
            let simplifier_ref = runtime_cell.borrow();
            crate::solver::prove_nonzero_status(&simplifier_ref.context, expr)
        },
        |name, solved| {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            build_linear_collect_additive_execution_with(
                &mut simplifier_ref.context,
                name,
                solved.coeff,
                solved.constant,
                solved.solution,
                solved.coeff_status,
                solved.constant_status,
                render_expr,
            )
        },
        |item: LinearCollectExecutionItem| {
            medium_step(item.description().to_string(), item.equation)
        },
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
