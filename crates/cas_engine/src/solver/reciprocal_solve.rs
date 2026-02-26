//! Reciprocal Solve Strategy
//!
//! Handles equations of the form `1/var = sum_of_fractions` with pedagogical steps:
//! 1. "Combine fractions on RHS (common denominator)"
//! 2. "Take reciprocal"
//!
//! Example: `1/R = 1/R1 + 1/R2` → `R = R1·R2/(R1+R2)`

use cas_ast::{ExprId, SolutionSet};
use cas_solver_core::reciprocal::{
    build_reciprocal_execution_from_kernel_prepared, build_reciprocal_solve_plan,
    derive_reciprocal_solve_kernel, execute_reciprocal_kernel_execution_pipeline_with,
    execute_reciprocal_solve_pipeline_with_items,
};

use crate::engine::Simplifier;
use crate::solver::{medium_step, SolveStep};

/// Try to solve `1/var = expr` using pedagogical steps.
///
/// Returns `Some((SolutionSet, steps))` if pattern matches,
/// `None` to fall through to standard isolation.
pub(crate) fn try_reciprocal_solve(
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<(SolutionSet, Vec<SolveStep>)> {
    let include_items = simplifier.collect_steps();
    let runtime_cell = std::cell::RefCell::new(&mut *simplifier);
    execute_reciprocal_solve_pipeline_with_items(
        lhs,
        rhs,
        var,
        include_items,
        |inner_lhs, inner_rhs, inner_var| {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            derive_reciprocal_solve_kernel(
                &mut simplifier_ref.context,
                inner_lhs,
                inner_rhs,
                inner_var,
            )
        },
        |inner_var, kernel| {
            execute_reciprocal_kernel_execution_pipeline_with(
                inner_var,
                kernel,
                |plan_var, plan_kernel| {
                    let mut simplifier_ref = runtime_cell.borrow_mut();
                    build_reciprocal_solve_plan(
                        &mut simplifier_ref.context,
                        plan_var,
                        plan_kernel.numerator,
                        plan_kernel.denominator,
                    )
                },
                |expr| {
                    let mut simplifier_ref = runtime_cell.borrow_mut();
                    simplifier_ref.simplify(expr).0
                },
                |expr| {
                    let simplifier_ref = runtime_cell.borrow();
                    crate::solver::prove_nonzero_status(&simplifier_ref.context, expr)
                },
                |build_var, build_kernel, prepared| {
                    let mut simplifier_ref = runtime_cell.borrow_mut();
                    build_reciprocal_execution_from_kernel_prepared(
                        &mut simplifier_ref.context,
                        build_var,
                        build_kernel,
                        prepared,
                    )
                },
            )
        },
        |item| medium_step(item.description().to_string(), item.equation),
    )
}

#[cfg(test)]
mod tests {
    use cas_ast::Context;
    use cas_ast::Expr;
    use cas_solver_core::isolation_utils::contains_var;

    #[test]
    fn test_is_simple_reciprocal() {
        let mut ctx = Context::new();
        let r = ctx.var("R");
        let one = ctx.num(1);
        let reciprocal = ctx.add(Expr::Div(one, r));

        assert!(cas_solver_core::isolation_utils::is_simple_reciprocal(
            &ctx, reciprocal, "R"
        ));
        assert!(!cas_solver_core::isolation_utils::is_simple_reciprocal(
            &ctx, reciprocal, "X"
        ));
        assert!(!cas_solver_core::isolation_utils::is_simple_reciprocal(
            &ctx, r, "R"
        ));
    }

    #[test]
    fn test_combine_fractions_simple() {
        let mut ctx = Context::new();
        let r1 = ctx.var("R1");
        let r2 = ctx.var("R2");
        let one = ctx.num(1);

        // 1/R1 + 1/R2
        let frac1 = ctx.add(Expr::Div(one, r1));
        let one2 = ctx.num(1);
        let frac2 = ctx.add(Expr::Div(one2, r2));
        let sum = ctx.add(Expr::Add(frac1, frac2));

        let result = cas_solver_core::reciprocal::combine_fractions_deterministic(&mut ctx, sum);
        assert!(result.is_some());

        let (num, denom) = result.unwrap();
        // Numerator should contain R1 and R2
        // Denominator should be R1*R2
        assert!(contains_var(&ctx, num, "R1") || contains_var(&ctx, num, "R2"));
        assert!(contains_var(&ctx, denom, "R1"));
        assert!(contains_var(&ctx, denom, "R2"));
    }
}
