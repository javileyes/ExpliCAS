//! Solver step cleanup for didactic display.
//!
//! This module provides display-layer cleanup for solve steps:
//! 1. Sign normalization: `0 - (-(t))` → `t`, `a - -b` → `a + b`
//! 2. Redundant step removal: detect undo/redo patterns
//!
//! These transformations are purely display-level and don't affect
//! the solver's internal operation or correctness.

use crate::helpers::is_zero;
use crate::solver::SolveStep;
use cas_ast::{Context, Expr, ExprId};

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
pub fn cleanup_solve_steps(
    ctx: &mut Context,
    steps: Vec<SolveStep>,
    detailed: bool,
) -> Vec<SolveStep> {
    if steps.is_empty() {
        return steps;
    }

    // Phase 1: Remove redundant steps (using original descriptions for detection)
    let filtered = remove_redundant_steps(ctx, steps);

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
    remove_duplicate_equations(normalized)
}

/// Remove consecutive steps where the equation_after is identical.
fn remove_duplicate_equations(steps: Vec<SolveStep>) -> Vec<SolveStep> {
    if steps.len() < 2 {
        return steps;
    }

    let mut result = Vec::with_capacity(steps.len());
    result.push(steps[0].clone());

    for i in 1..steps.len() {
        let prev = &steps[i - 1];
        let curr = &steps[i];

        // Check if equations are identical (by ExprId)
        if prev.equation_after.lhs == curr.equation_after.lhs
            && prev.equation_after.rhs == curr.equation_after.rhs
        {
            continue; // Skip - shows same equation
        }

        result.push(curr.clone());
    }

    result
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

/// Clean up step descriptions for better readability.
fn cleanup_step_description(desc: &str) -> String {
    // Pattern: "Subtract -(...)" → "Move terms to one side"
    if desc.starts_with("Subtract -(") || desc.starts_with("Subtract -") {
        return "Move terms to one side".to_string();
    }

    // Pattern: "Add -(...)" → "Move terms to one side"
    if desc.starts_with("Add -(") || desc.starts_with("Add -") {
        return "Move terms to one side".to_string();
    }

    desc.to_string()
}

/// Normalize signs in an expression for cleaner display.
fn normalize_expr_signs(ctx: &mut Context, expr: ExprId) -> ExprId {
    normalize_signs_recursive(ctx, expr)
}

fn normalize_signs_recursive(ctx: &mut Context, expr: ExprId) -> ExprId {
    let expr_data = ctx.get(expr).clone();

    match expr_data {
        // Pattern: 0 - (-(t)) → t
        // Pattern: 0 - t → -t (cleaner)
        Expr::Sub(lhs, rhs) => {
            // First normalize children
            let norm_lhs = normalize_signs_recursive(ctx, lhs);
            let norm_rhs = normalize_signs_recursive(ctx, rhs);

            // Check if LHS is 0
            if is_zero(ctx, norm_lhs) {
                // 0 - (-(t)) → t
                if let Expr::Neg(inner) = ctx.get(norm_rhs).clone() {
                    return normalize_signs_recursive(ctx, inner);
                }
                // 0 - t → -t (more concise)
                return ctx.add(Expr::Neg(norm_rhs));
            }

            // Pattern: a - -b → a + b
            if let Expr::Neg(inner) = ctx.get(norm_rhs).clone() {
                return ctx.add(Expr::Add(norm_lhs, inner));
            }

            // No change needed, but children may have changed
            if norm_lhs != lhs || norm_rhs != rhs {
                ctx.add(Expr::Sub(norm_lhs, norm_rhs))
            } else {
                expr
            }
        }

        // Pattern: -(-(x)) → x
        Expr::Neg(inner) => {
            let norm_inner = normalize_signs_recursive(ctx, inner);
            if let Expr::Neg(inner_inner) = ctx.get(norm_inner).clone() {
                return normalize_signs_recursive(ctx, inner_inner);
            }
            if norm_inner != inner {
                ctx.add(Expr::Neg(norm_inner))
            } else {
                expr
            }
        }

        // Recursively normalize other expressions
        Expr::Add(l, r) => {
            let nl = normalize_signs_recursive(ctx, l);
            let nr = normalize_signs_recursive(ctx, r);
            if nl != l || nr != r {
                ctx.add(Expr::Add(nl, nr))
            } else {
                expr
            }
        }

        Expr::Mul(l, r) => {
            let nl = normalize_signs_recursive(ctx, l);
            let nr = normalize_signs_recursive(ctx, r);
            if nl != l || nr != r {
                ctx.add(Expr::Mul(nl, nr))
            } else {
                expr
            }
        }

        Expr::Div(l, r) => {
            let nl = normalize_signs_recursive(ctx, l);
            let nr = normalize_signs_recursive(ctx, r);
            if nl != l || nr != r {
                ctx.add(Expr::Div(nl, nr))
            } else {
                expr
            }
        }

        Expr::Pow(base, exp) => {
            let nb = normalize_signs_recursive(ctx, base);
            let ne = normalize_signs_recursive(ctx, exp);
            if nb != base || ne != exp {
                ctx.add(Expr::Pow(nb, ne))
            } else {
                expr
            }
        }

        Expr::Function(name, args) => {
            let new_args: Vec<ExprId> = args
                .iter()
                .map(|&a| normalize_signs_recursive(ctx, a))
                .collect();
            if new_args != args {
                ctx.add(Expr::Function(name, new_args))
            } else {
                expr
            }
        }

        // Leaf nodes: no change
        _ => expr,
    }
}

/// Remove redundant step pairs from the step list.
///
/// Detects patterns like:
/// - Step i: `E = 0`
/// - Step i+1: `E = something` (undoes step i)
///
/// In these cases, we remove the earlier "worse" step.
fn remove_redundant_steps(ctx: &Context, steps: Vec<SolveStep>) -> Vec<SolveStep> {
    if steps.len() < 2 {
        return steps;
    }

    let mut result = Vec::with_capacity(steps.len());
    let mut i = 0;

    while i < steps.len() {
        let current = &steps[i];

        // Check if next step effectively undoes this one
        if i + 1 < steps.len() {
            let next = &steps[i + 1];

            // Pattern: current has RHS = 0, next reintroduces RHS
            // This is the "normalize to 0 then undo" pattern
            if is_step_normalize_to_zero(ctx, current)
                && is_step_undo_normalization(ctx, current, next)
            {
                // Skip current step (the =0 normalization), keep next
                i += 1;
                continue;
            }
        }

        result.push(current.clone());
        i += 1;
    }

    result
}

/// Check if a step normalizes to = 0 form.
fn is_step_normalize_to_zero(ctx: &Context, step: &SolveStep) -> bool {
    is_zero(ctx, step.equation_after.rhs)
}

/// Check if next step effectively undoes a =0 normalization.
///
/// This happens when:
/// - Previous step: `LHS - something = 0`
/// - Current step: `LHS = something` (or similar restructuring)
fn is_step_undo_normalization(ctx: &Context, prev: &SolveStep, curr: &SolveStep) -> bool {
    let prev_rhs_zero = is_zero(ctx, prev.equation_after.rhs);
    let curr_rhs_zero = is_zero(ctx, curr.equation_after.rhs);

    if prev_rhs_zero && !curr_rhs_zero {
        // Pattern 1: "Subtract X" followed by step that undoes it
        if prev.description.contains("Subtract")
            && curr.description.contains("Subtract")
            && (curr.description.contains("-(") || curr.description.contains("- -("))
        {
            return true;
        }

        // Pattern 2: Description got cleaned to "Move terms to one side"
        // This means we detected an undo pattern
        if curr.description == "Move terms to one side" {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_zero_minus_neg() {
        let mut ctx = Context::new();
        let t = ctx.var("t");
        let neg_t = ctx.add(Expr::Neg(t));
        let zero = ctx.num(0);
        let expr = ctx.add(Expr::Sub(zero, neg_t)); // 0 - (-(t))

        let result = normalize_expr_signs(&mut ctx, expr);

        // Should become t
        assert!(matches!(ctx.get(result), Expr::Variable(v) if v == "t"));
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
        assert!(matches!(ctx.get(result), Expr::Variable(v) if v == "x"));
    }
}
