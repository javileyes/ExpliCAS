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
    derive_linear_collect_factored_kernel, solve_linear_collect_execution_pipeline_with_items,
};
use cas_solver_core::linear_solution::{build_linear_solution_set, NonZeroStatus};

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
    let equation_diff = simplifier.context.add(Expr::Sub(lhs, rhs));
    let simplified_diff = simplifier.simplify(equation_diff).0;
    let factored_kernel =
        derive_linear_collect_factored_kernel(&mut simplifier.context, simplified_diff, var)?;

    let coeff = simplifier.simplify(factored_kernel.coeff).0;
    let rhs_term = simplifier.simplify(factored_kernel.rhs_term).0;
    let solution_expr =
        build_linear_collect_solution_expr(&mut simplifier.context, rhs_term, coeff);
    let solution = simplifier.simplify(solution_expr).0;

    let coeff_contains_var = contains_var(&simplifier.context, coeff, var);
    let (coeff_status, rhs_status) = if coeff_contains_var {
        (NonZeroStatus::Unknown, NonZeroStatus::Unknown)
    } else {
        let coeff_status = crate::solver::prove_nonzero_status(&simplifier.context, coeff);
        let rhs_status = if coeff_status == NonZeroStatus::Zero {
            crate::solver::prove_nonzero_status(&simplifier.context, rhs_term)
        } else {
            NonZeroStatus::Unknown
        };
        (coeff_status, rhs_status)
    };

    if include_items {
        let execution = build_linear_collect_factored_execution_with(
            &mut simplifier.context,
            var,
            coeff,
            rhs_term,
            solution,
            coeff_status,
            rhs_status,
            render_expr,
        );
        let solved = solve_linear_collect_execution_pipeline_with_items(execution, true, |item| {
            medium_step(item.description().to_string(), item.equation)
        });
        return Some(solved.solved);
    }

    let solution_set =
        build_linear_solution_set(coeff, rhs_term, solution, coeff_status, rhs_status);
    Some((solution_set, Vec::new()))
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
    let additive_kernel =
        derive_linear_collect_additive_kernel(&mut simplifier.context, lhs, rhs, var)?;

    let coeff = simplifier.simplify(additive_kernel.coeff).0;
    let constant = simplifier.simplify(additive_kernel.constant).0;
    let neg_constant = simplifier.context.add(Expr::Neg(constant));
    let solution_expr =
        build_linear_collect_solution_expr(&mut simplifier.context, neg_constant, coeff);
    let solution = simplifier.simplify(solution_expr).0;

    let coeff_contains_var = contains_var(&simplifier.context, coeff, var);
    let (coeff_status, constant_status) = if coeff_contains_var {
        (NonZeroStatus::Unknown, NonZeroStatus::Unknown)
    } else {
        let coeff_status = crate::solver::prove_nonzero_status(&simplifier.context, coeff);
        let constant_status = if coeff_status == NonZeroStatus::Zero {
            crate::solver::prove_nonzero_status(&simplifier.context, constant)
        } else {
            NonZeroStatus::Unknown
        };
        (coeff_status, constant_status)
    };

    if include_items {
        let execution = build_linear_collect_additive_execution_with(
            &mut simplifier.context,
            var,
            coeff,
            constant,
            solution,
            coeff_status,
            constant_status,
            render_expr,
        );
        let solved = solve_linear_collect_execution_pipeline_with_items(execution, true, |item| {
            medium_step(item.description().to_string(), item.equation)
        });
        return Some(solved.solved);
    }

    let solution_set =
        build_linear_solution_set(coeff, constant, solution, coeff_status, constant_status);
    Some((solution_set, Vec::new()))
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
