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
use cas_ast::Context;
use cas_solver_core::log_linear_narration::rewrite_log_linear_steps_by;

#[cfg(test)]
use cas_solver_core::log_linear_narration::is_log_linear_pattern_by;

/// Check if a step sequence is a log-linear solve pattern.
/// Returns true if we should rewrite the steps for better didactic display.
#[cfg(test)]
pub(crate) fn is_log_linear_pattern(steps: &[SolveStep]) -> bool {
    is_log_linear_pattern_by(steps, |s| s.description.as_str())
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
    rewrite_log_linear_steps_by(
        ctx,
        steps,
        detailed,
        "x",
        |s| s.description.as_str(),
        |s| &s.equation_after,
        |_template, payload| SolveStep {
            description: payload.description,
            equation_after: payload.equation_after,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_solver_core::log_linear_narration::TAKE_LOG_BOTH_SIDES_STEP;

    #[test]
    fn test_is_log_linear_pattern() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        let steps = vec![SolveStep {
            description: TAKE_LOG_BOTH_SIDES_STEP.to_string(),
            equation_after: cas_ast::Equation {
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
            equation_after: cas_ast::Equation {
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
