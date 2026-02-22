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
use cas_ast::{Context, Equation};
use cas_solver_core::log_linear_narration::{build_detailed_collect_steps, try_rewrite_ln_power};

/// Check if a step sequence is a log-linear solve pattern.
/// Returns true if we should rewrite the steps for better didactic display.
pub(crate) fn is_log_linear_pattern(steps: &[SolveStep]) -> bool {
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
///
/// # Arguments
/// * `ctx` - Expression context
/// * `steps` - Original solve steps  
/// * `detailed` - If true, decompose "Collect and factor" into atomic sub-steps
pub(crate) fn rewrite_log_linear_steps(
    ctx: &mut Context,
    steps: Vec<SolveStep>,
    detailed: bool,
) -> Vec<SolveStep> {
    if steps.is_empty() || !is_log_linear_pattern(&steps) {
        return steps;
    }

    // Find the key steps in the original sequence
    let mut result = Vec::new();
    let mut i = 0;

    // Track the equation after "Take log" for building intermediates
    let mut log_eq: Option<Equation> = None;

    while i < steps.len() {
        let step = &steps[i];

        // Step 1: Keep "Take log base e" but ensure RHS shows x·ln(b) not ln(b^x)
        if step.description == "Take log base e of both sides" {
            // Check if RHS is ln(something^x) and rewrite to x·ln(something)
            let improved_rhs = try_rewrite_ln_power(ctx, step.equation_after.rhs);

            let eq = Equation {
                lhs: step.equation_after.lhs,
                rhs: improved_rhs.unwrap_or(step.equation_after.rhs),
                op: step.equation_after.op.clone(),
            };

            // Save for building intermediates
            log_eq = Some(eq.clone());

            result.push(SolveStep {
                description: "Take log base e of both sides".to_string(),
                equation_after: eq,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });

            i += 1;
            continue;
        }

        // Handle "Collect terms in x" - decompose if detailed mode
        if step.description.starts_with("Collect terms in") {
            if detailed {
                // Decompose into atomic sub-steps with REAL intermediate equations
                let sub_steps =
                    build_detailed_collect_steps(ctx, log_eq.as_ref(), &step.equation_after, "x")
                        .into_iter()
                        .map(|s| SolveStep {
                            description: s.description,
                            equation_after: s.equation_after,
                            importance: crate::step::ImportanceLevel::Medium,
                            substeps: vec![],
                        })
                        .collect::<Vec<_>>();
                result.extend(sub_steps);
            } else {
                // Compact mode: single step with cleaner description
                result.push(SolveStep {
                    description: "Collect and factor x terms".to_string(),
                    equation_after: step.equation_after.clone(),
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            i += 1;
            continue;
        }

        // Default: keep the step
        result.push(step.clone());
        i += 1;
    }

    result
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
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
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
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        }];

        assert!(!is_log_linear_pattern(&steps));
    }
}
