//! Runtime step optimization pipeline shared across integration crates.

use crate::soundness_label::SoundnessLabel;
use crate::step_absorption::{absorb_indices, find_absorption_indices_before_markers_with};
use crate::step_model::{Step, StepMeta};
use crate::step_optimize::optimize_steps_with_rules;
use crate::step_rules::{
    is_canonicalization_rule_name, is_expansion_rule_name, is_mechanical_rule_name,
};
use crate::step_semantic::is_semantic_noop_without_didactic_steps;
use crate::step_types::{ImportanceLevel, StepCategory};
use cas_ast::{Context, ExprId};

/// Result of step optimization with semantic analysis.
#[derive(Debug)]
pub enum StepOptimizationResult {
    /// Steps were optimized normally.
    Steps(Vec<Step>),
    /// No real simplification occurred (result semantically equals input).
    NoSimplificationNeeded,
}

/// Optimize steps with semantic cycle detection.
///
/// Returns `NoSimplificationNeeded` if final result is semantically equal to
/// original input and there are no didactically important steps to preserve.
pub fn optimize_steps_semantic(
    steps: Vec<Step>,
    ctx: &Context,
    original_expr: ExprId,
    final_expr: ExprId,
) -> StepOptimizationResult {
    // Skip output only when simplification is a semantic no-op and there are no
    // didactically important steps to preserve.
    let is_semantic_noop =
        is_semantic_noop_without_didactic_steps(ctx, original_expr, final_expr, &steps, |step| {
            step.rule_name == "Sum Exponents" || step.rule_name == "Evaluate Numeric Power"
        });
    if is_semantic_noop {
        return StepOptimizationResult::NoSimplificationNeeded;
    }

    // Check if there are polynomial identity steps - use absorption to hide mechanical steps.
    let has_poly_identity = steps.iter().any(|s| s.poly_proof().is_some());

    // Otherwise, apply normal optimization (with absorption if PolyZero present).
    if has_poly_identity {
        StepOptimizationResult::Steps(optimize_steps_with_absorption(steps))
    } else {
        StepOptimizationResult::Steps(optimize_steps(steps))
    }
}

/// Collapse low-signal step chains while preserving important transitions.
pub fn optimize_steps(steps: Vec<Step>) -> Vec<Step> {
    optimize_steps_with_rules(
        steps,
        5,
        |step| step.rule_name.as_str(),
        is_expansion_rule_name,
        is_canonicalization_rule_name,
        |step| step.importance >= ImportanceLevel::Medium,
        |a, b| a.path() == b.path(),
        |step| {
            step.rule_name == "Evaluate Numeric Power"
                && step.description.contains("1^")
                && step.description.contains("-> 1")
        },
        |first, last| Step {
            description: "Canonicalization".to_string(),
            rule_name: "Canonicalize".to_string(),
            before: first.before,
            after: last.after,
            global_before: first.global_before,
            global_after: last.global_after,
            importance: ImportanceLevel::Low,
            category: StepCategory::Canonicalize,
            soundness: SoundnessLabel::Equivalence,
            meta: Some(Box::new(StepMeta {
                path: first.path().to_vec(),
                after_str: last.after_str().map(|s| s.to_string()),
                ..Default::default()
            })),
        },
        |step| step.before,
        |step| step.after,
        |step| step.global_before,
        |step| step.global_after,
    )
}

/// Absorb mechanical steps preceding a polynomial-identity step.
///
/// Uses a bounded look-back window and never absorbs medium/high-importance
/// steps.
pub fn find_steps_to_absorb_for_polyzero(steps: &[Step]) -> Vec<usize> {
    find_absorption_indices_before_markers_with(
        steps,
        8, // Max steps to look back.
        |s| s.poly_proof().is_some(),
        |s| is_mechanical_rule_name(&s.rule_name),
        |s| s.importance >= ImportanceLevel::Medium,
    )
}

/// Enhanced step optimization with polynomial identity absorption.
pub fn optimize_steps_with_absorption(steps: Vec<Step>) -> Vec<Step> {
    let indices = find_steps_to_absorb_for_polyzero(&steps);
    let filtered = absorb_indices(steps, &indices);
    optimize_steps(filtered)
}
