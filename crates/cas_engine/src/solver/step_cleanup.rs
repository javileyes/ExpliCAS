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
use cas_solver_core::sign_normalize::{cleanup_step_description, normalize_expr_signs};

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
    if steps.is_empty() {
        return steps;
    }

    // Phase 1: Remove redundant steps (using original descriptions for detection)
    let filtered = cas_solver_core::step_cleanup::remove_redundant_steps_by(
        ctx,
        steps,
        |s| s.description.as_str(),
        |s| &s.equation_after,
    );

    // Phase 2: Rewrite log-linear steps for didactic clarity
    // detailed=true → atomic sub-steps (Expand, Move, Factor)
    // detailed=false → compact step (Collect and factor)
    use crate::solver::log_linear_narrator;
    let narrated = log_linear_narrator::rewrite_log_linear_steps(ctx, filtered, detailed);

    // Phase 3: Normalize signs in remaining steps
    let normalized: Vec<SolveStep> = narrated
        .into_iter()
        .map(|step| normalize_step_signs(ctx, step))
        .collect();

    // Phase 4: Remove consecutive steps with identical equations
    // Now safe for both modes since detailed generates distinct equations
    cas_solver_core::step_cleanup::remove_duplicate_equations_by(normalized, |s| &s.equation_after)
}

/// Normalize signs in a single step's equation for cleaner display.
///
/// Patterns handled:
/// - `0 - (-(t))` → `t`
/// - `0 - t` → `-t`
/// - `a - -b` → `a + b`
/// - `-(-(x))` → `x`
/// - Description: "Subtract -(..." → "Move terms to one side"
fn normalize_step_signs(ctx: &mut Context, mut step: SolveStep) -> SolveStep {
    step.equation_after.lhs = normalize_expr_signs(ctx, step.equation_after.lhs);
    step.equation_after.rhs = normalize_expr_signs(ctx, step.equation_after.rhs);

    // Clean up ugly descriptions
    step.description = cleanup_step_description(&step.description);

    step
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Expr;

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
