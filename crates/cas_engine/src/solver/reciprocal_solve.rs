//! Reciprocal Solve Strategy
//!
//! Handles equations of the form `1/var = sum_of_fractions` with pedagogical steps:
//! 1. "Combine fractions on RHS (common denominator)"
//! 2. "Take reciprocal"
//!
//! Example: `1/R = 1/R1 + 1/R2` → `R = R1·R2/(R1+R2)`

use cas_ast::{ExprId, SolutionSet};

use crate::engine::Simplifier;
use crate::solver::proof_bridge::proof_to_nonzero_status;
use crate::solver::SolveStep;

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
    let kernel = cas_solver_core::reciprocal::derive_reciprocal_solve_kernel(
        &mut simplifier.context,
        lhs,
        rhs,
        var,
    )?;
    let numerator = kernel.numerator;
    let denominator = kernel.denominator;
    let plan = cas_solver_core::reciprocal::build_reciprocal_solve_plan(
        &mut simplifier.context,
        var,
        numerator,
        denominator,
    );

    let mut steps = Vec::new();

    // Step 1: Combine fractions on RHS -> 1/var = numerator/denominator.
    // Light simplification for display (order terms).
    let (display_rhs, _) = simplifier.simplify(plan.combined_rhs);

    if simplifier.collect_steps() {
        let mut equation_after = plan.combine_equation.clone();
        equation_after.rhs = display_rhs;
        let combine_step =
            cas_solver_core::reciprocal::build_reciprocal_combine_step(equation_after);

        steps.push(SolveStep {
            description: combine_step.description,
            equation_after: combine_step.equation_after,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }

    // Step 2: Take reciprocal -> var = denominator/numerator.
    let (simplified_solution, _) = simplifier.simplify(plan.solution_rhs);

    if simplifier.collect_steps() {
        let mut equation_after = plan.solve_equation.clone();
        equation_after.rhs = simplified_solution;
        let invert_step = cas_solver_core::reciprocal::build_reciprocal_invert_step(equation_after);

        steps.push(SolveStep {
            description: invert_step.description,
            equation_after: invert_step.equation_after,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }

    // Build solution set with domain guard for numerator != 0 when needed.
    use crate::helpers::prove_nonzero;

    // Simplify the numerator for cleaner display and proof checking
    let (simplified_numerator, _) = simplifier.simplify(numerator);

    let numerator_status =
        proof_to_nonzero_status(prove_nonzero(&simplifier.context, simplified_numerator));
    let solution_set = cas_solver_core::reciprocal::build_reciprocal_solution_set(
        simplified_numerator,
        simplified_solution,
        numerator_status,
    );

    Some((solution_set, steps))
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
