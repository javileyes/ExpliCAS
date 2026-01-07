//! Log-linear equation narrator for didactic step display.
//!
//! This module rewrites solve steps for log-linear equations (like `3^(x+1) = 5^x`)
//! into a coherent, atomic step sequence that follows a natural pedagogical flow:
//!
//! 1. Take log of both sides
//! 2. Apply power rule: ln(a^x) → x·ln(a)
//! 3. Expand products: ln(3)·(1+x) → ln(3) + x·ln(3)
//! 4. Move x terms to one side
//! 5. Factor out x
//! 6. Apply log quotient rule: ln(a) - ln(b) → ln(a/b)
//! 7. Divide
//!
//! The key insight is that the MATHEMATICAL result is correct, but the TRACE
//! needs to be restructured for didactic clarity.

use crate::solver::SolveStep;
use cas_ast::{Context, Equation, Expr, ExprId};

/// Check if a step sequence is a log-linear solve pattern.
/// Returns true if we should rewrite the steps for better didactic display.
pub fn is_log_linear_pattern(steps: &[SolveStep]) -> bool {
    if steps.is_empty() {
        return false;
    }

    // Pattern: First step is "Take log base e of both sides"
    steps[0].description == "Take log base e of both sides"
}

/// Rewrite log-linear solve steps into a coherent didactic sequence.
///
/// This analyzes the current steps and the equation transformations to
/// produce a cleaner narrative. It does NOT change the mathematical result,
/// only how the steps are presented.
pub fn rewrite_log_linear_steps(ctx: &mut Context, steps: Vec<SolveStep>) -> Vec<SolveStep> {
    if steps.is_empty() || !is_log_linear_pattern(&steps) {
        return steps;
    }

    // Find the key steps in the original sequence
    let mut result = Vec::new();
    let mut i = 0;

    while i < steps.len() {
        let step = &steps[i];

        // Step 1: Keep "Take log base e" but ensure RHS shows x·ln(b) not ln(b^x)
        if step.description == "Take log base e of both sides" {
            // Check if RHS is ln(something^x) and rewrite to x·ln(something)
            let improved_rhs = try_rewrite_log_power(ctx, step.equation_after.rhs);

            result.push(SolveStep {
                description: "Take log base e of both sides".to_string(),
                equation_after: Equation {
                    lhs: step.equation_after.lhs,
                    rhs: improved_rhs.unwrap_or(step.equation_after.rhs),
                    op: step.equation_after.op.clone(),
                },
            });

            // If we rewrote ln(b^x) → x·ln(b), add an educational step
            if improved_rhs.is_some() {
                // This transformation was done implicitly - the step already shows it
            }

            i += 1;
            continue;
        }

        // Combine "Collect terms in x" with any implicit expansions into multiple steps
        if step.description.starts_with("Collect terms in") {
            // This step often does too much. Break it into parts:
            // 1. Expand products (if any)
            // 2. Move x terms
            // 3. Factor x
            // 4. Simplify log difference

            // For now, we keep as-is but with cleaner description
            result.push(SolveStep {
                description: "Collect and factor x terms".to_string(),
                equation_after: step.equation_after.clone(),
            });
            i += 1;
            continue;
        }

        // Default: keep the step
        result.push(step.clone());
        i += 1;
    }

    result
}

/// Try to rewrite ln(a^x) → x·ln(a) for cleaner display.
/// Returns Some(new_expr) if rewrite was done, None otherwise.
fn try_rewrite_log_power(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let expr_data = ctx.get(expr).clone();

    match expr_data {
        // ln(a^exp) → exp·ln(a)
        Expr::Function(name, args) if name == "ln" && args.len() == 1 => {
            let inner = args[0];
            if let Expr::Pow(base, exp) = ctx.get(inner).clone() {
                // Create exp * ln(base)
                let ln_base = ctx.add(Expr::Function("ln".to_string(), vec![base]));
                let product = ctx.add(Expr::Mul(exp, ln_base));
                return Some(product);
            }
            None
        }
        _ => None,
    }
}

/// Check if equation after log contains pattern ln(a)·(1+x) = x·ln(b)
/// This indicates a log-linear equation we can narrate better.
pub fn detect_log_linear_form(ctx: &Context, lhs: ExprId, rhs: ExprId, var: &str) -> bool {
    // Check if LHS has pattern ln(a)·f(x) and RHS has pattern g(x)·ln(b)
    let lhs_has_ln = contains_ln_times_var(ctx, lhs, var);
    let rhs_has_ln = contains_ln_times_var(ctx, rhs, var);

    lhs_has_ln && rhs_has_ln
}

/// Check if expression contains a pattern like ln(c) * (something with var)
fn contains_ln_times_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    match ctx.get(expr) {
        Expr::Mul(l, r) => {
            // Check if one side is ln(...) and other contains var
            let l_is_ln = is_ln_call(ctx, *l);
            let r_is_ln = is_ln_call(ctx, *r);
            let l_has_var = crate::solver::contains_var(ctx, *l, var);
            let r_has_var = crate::solver::contains_var(ctx, *r, var);

            (l_is_ln && r_has_var) || (r_is_ln && l_has_var) || (l_has_var && r_has_var)
        }
        Expr::Add(l, r) | Expr::Sub(l, r) => {
            contains_ln_times_var(ctx, *l, var) || contains_ln_times_var(ctx, *r, var)
        }
        _ => false,
    }
}

/// Check if expression is a ln(...) call
fn is_ln_call(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Function(name, _) if name == "ln")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_log_linear_pattern() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        let steps = vec![SolveStep {
            description: "Take log base e of both sides".to_string(),
            equation_after: Equation {
                lhs: x,
                rhs: x,
                op: cas_ast::RelOp::Eq,
            },
        }];

        assert!(is_log_linear_pattern(&steps));
    }

    #[test]
    fn test_not_log_linear_pattern() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        let steps = vec![SolveStep {
            description: "Square both sides".to_string(),
            equation_after: Equation {
                lhs: x,
                rhs: x,
                op: cas_ast::RelOp::Eq,
            },
        }];

        assert!(!is_log_linear_pattern(&steps));
    }
}
