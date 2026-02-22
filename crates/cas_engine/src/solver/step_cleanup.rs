//! Solver step cleanup for didactic display.
//!
//! This module provides display-layer cleanup for solve steps:
//! 1. Sign normalization: `0 - (-(t))` → `t`, `a - -b` → `a + b`
//! 2. Redundant step removal: detect undo/redo patterns
//!
//! These transformations are purely display-level and don't affect
//! the solver's internal operation or correctness.

use crate::solver::SolveStep;
use cas_ast::Context;
use cas_solver_core::step_cleanup::CleanupStep;

/// Clean up solve steps for better didactic display.
///
/// Returns a filtered and normalized list of steps suitable for showing
/// to users. This function:
/// 1. Removes redundant step pairs (undo/redo patterns) - using ORIGINAL descriptions
/// 2. Rewrites log-linear steps for better didactic flow (Phase 2)
/// 3. Normalizes signs in each equation (display only)
/// 4. Removes consecutive steps with identical equations
///
/// # Arguments
/// * `ctx` - Expression context
/// * `steps` - Original solve steps  
/// * `detailed` - If true, decompose into atomic sub-steps (Normal/Verbose verbosity)
///   If false, use compact representation (Succinct verbosity)
pub(crate) fn cleanup_solve_steps(
    ctx: &mut Context,
    steps: Vec<SolveStep>,
    detailed: bool,
) -> Vec<SolveStep> {
    cas_solver_core::step_cleanup::cleanup_steps_by(
        ctx,
        steps,
        detailed,
        "x",
        |s| CleanupStep {
            description: s.description.clone(),
            equation_after: s.equation_after.clone(),
        },
        |template, payload| SolveStep {
            description: payload.description,
            equation_after: payload.equation_after,
            importance: template.importance,
            substeps: template.substeps,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Expr;
    use cas_solver_core::sign_normalize::normalize_expr_signs;

    #[test]
    fn test_normalize_zero_minus_neg() {
        let mut ctx = Context::new();
        let t = ctx.var("t");
        let neg_t = ctx.add(Expr::Neg(t));
        let zero = ctx.num(0);
        let expr = ctx.add(Expr::Sub(zero, neg_t)); // 0 - (-(t))

        let result = normalize_expr_signs(&mut ctx, expr);

        // Should become t
        assert!(matches!(ctx.get(result), Expr::Variable(v) if ctx.sym_name(*v) == "t"));
    }

    #[test]
    fn test_normalize_sub_neg() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let neg_b = ctx.add(Expr::Neg(b));
        let expr = ctx.add(Expr::Sub(a, neg_b)); // a - -b

        let result = normalize_expr_signs(&mut ctx, expr);

        // Should become a + b
        assert!(matches!(ctx.get(result), Expr::Add(_, _)));
    }

    #[test]
    fn test_normalize_double_neg() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let neg_x = ctx.add(Expr::Neg(x));
        let double_neg = ctx.add(Expr::Neg(neg_x)); // -(-(x))

        let result = normalize_expr_signs(&mut ctx, double_neg);

        // Should become x
        assert!(matches!(ctx.get(result), Expr::Variable(v) if ctx.sym_name(*v) == "x"));
    }
}
