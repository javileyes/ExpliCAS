use crate::step::Step;
use cas_ast::{Context, ExprId};
use cas_math::semantic_equality::SemanticEqualityChecker;

/// Result of step optimization with semantic analysis
#[derive(Debug)]
pub enum StepOptimizationResult {
    /// Steps were optimized normally
    Steps(Vec<Step>),
    /// No real simplification occurred (result semantically equals input)
    NoSimplificationNeeded,
}

/// Optimize steps with semantic cycle detection
/// Returns NoSimplificationNeeded if final result is semantically equal to original input
pub fn optimize_steps_semantic(
    steps: Vec<Step>,
    ctx: &Context,
    original_expr: ExprId,
    final_expr: ExprId,
) -> StepOptimizationResult {
    // Check if there are didactically important steps that should always be shown
    // - Sum Exponents is didactically important even if x^(1/2+1/3) == x^(5/6) semantically
    // - Evaluate Numeric Power shows root simplification like sqrt(12) → 2*√3
    let has_didactic_steps = steps
        .iter()
        .any(|s| s.rule_name == "Sum Exponents" || s.rule_name == "Evaluate Numeric Power");

    // First check if the entire simplification was a no-op
    // Use the lax cycle check that considers Sub(a,b) equal to Add(-b,a)
    // But only skip if there are no didactic steps to show
    let checker = SemanticEqualityChecker::new(ctx);
    if !has_didactic_steps && checker.are_equal_for_cycle_check(original_expr, final_expr) {
        return StepOptimizationResult::NoSimplificationNeeded;
    }

    // Check if there are polynomial identity steps - use absorption to hide mechanical steps
    let has_poly_identity = steps.iter().any(|s| s.poly_proof().is_some());

    // Otherwise, apply normal optimization (with absorption if PolyZero present)
    if has_poly_identity {
        StepOptimizationResult::Steps(optimize_steps_with_absorption(steps))
    } else {
        StepOptimizationResult::Steps(optimize_steps(steps))
    }
}

pub fn optimize_steps(steps: Vec<Step>) -> Vec<Step> {
    let mut optimized = Vec::new();
    let mut i = 0;

    while i < steps.len() {
        let current = &steps[i];

        // === Cycle Detection: Expand followed by Factor returning to same ===
        // Pattern: Binomial Expansion → ... → Factor where the final result equals the input
        if is_expansion_rule(&current.rule_name) {
            // Look ahead for a Factor step that might close the cycle
            if let Some(end_idx) = find_expand_factor_cycle(&steps, i) {
                let last = &steps[end_idx];

                // Check if it's a true no-op: global_before of expansion == global_after of factor
                let is_noop = match (current.global_before, last.global_after) {
                    (Some(before), Some(after)) => before == after,
                    _ => false,
                };

                if is_noop {
                    // Skip the entire cycle - it's a no-op
                    i = end_idx + 1;
                    continue;
                }
            }
        }

        // === Canonicalization Coalescing ===
        // Only coalesce Low-importance canonicalization steps; Medium+ are didactic barriers
        if is_canonicalization_rule(&current.rule_name)
            && current.importance < crate::step::ImportanceLevel::Medium
        {
            let mut j = i + 1;
            let mut last_same_path_idx = i;

            while j < steps.len() {
                let next = &steps[j];
                // Stop coalescing if we hit a Medium+ importance step
                if next.importance >= crate::step::ImportanceLevel::Medium {
                    break;
                }
                if is_canonicalization_rule(&next.rule_name) && next.path() == current.path() {
                    last_same_path_idx = j;
                    j += 1;
                } else {
                    break;
                }
            }

            if last_same_path_idx > i {
                let last = &steps[last_same_path_idx];
                let coalesced = Step {
                    description: "Canonicalization".to_string(),
                    rule_name: "Canonicalize".to_string(),
                    before: current.before,
                    after: last.after,
                    global_before: current.global_before,
                    global_after: last.global_after,
                    importance: crate::step::ImportanceLevel::Low,
                    category: crate::step::StepCategory::Canonicalize,
                    soundness: crate::rule::SoundnessLabel::Equivalence,
                    meta: Some(Box::new(crate::step::StepMeta {
                        path: current.path().to_vec(),
                        after_str: last.after_str().map(|s| s.to_string()),
                        ..Default::default()
                    })),
                };
                optimized.push(coalesced);
                i = last_same_path_idx + 1;
                continue;
            }
        }

        // === Filter trivial power evaluations like 1^2 → 1 ===
        if current.rule_name == "Evaluate Numeric Power"
            && current.description.contains("1^")
            && current.description.contains("-> 1")
        {
            i += 1;
            continue;
        }

        optimized.push(current.clone());
        i += 1;
    }

    optimized
}

/// Find expand→factor cycle: returns index of closing Factor step if cycle exists
fn find_expand_factor_cycle(steps: &[Step], start: usize) -> Option<usize> {
    let start_before = steps[start].before;

    // Look for a Factor step within reasonable window (e.g., 5 steps)
    for (j, step) in steps.iter().enumerate().skip(start + 1).take(5) {
        if step.rule_name == "Factor" && step.after == start_before {
            return Some(j);
        }
    }
    None
}

fn is_expansion_rule(name: &str) -> bool {
    name == "Binomial Expansion" || name == "Expand"
}

fn is_canonicalization_rule(name: &str) -> bool {
    name.starts_with("Canonicalize") || name == "Collect" || name.starts_with("Sort")
}

/// Mechanical rules that can be absorbed when followed by Polynomial Identity
/// These rules represent intermediate "grind" computation that PolyZero makes redundant
fn is_mechanical_rule(name: &str) -> bool {
    // Explicit match for common mechanical rules
    if matches!(
        name,
        "Distributive Property"
            | "Distribute"
            | "Combine Like Terms"
            | "Expand"
            | "Binomial Expansion"
            | "Add Inverse"
            | "Identity Property of Addition"
    ) {
        return true;
    }

    // Also absorb all canonicalization rules (they're just reordering/cleanup)
    if name.starts_with("Canonicalize") || name == "Collect" || name.starts_with("Sort") {
        return true;
    }

    false
}

/// Absorb mechanical steps preceding a PolynomialIdentity step
/// Uses window-based approach: absorb last K mechanical steps before PolyZero
/// without requiring exact before/after chain matching (handles reorderings)
///
/// IMPORTANT: Steps with importance >= Medium are NEVER absorbed.
/// Medium/High importance means "didactically important - user should see this".
pub fn find_steps_to_absorb_for_polyzero(steps: &[Step]) -> Vec<usize> {
    use crate::step::ImportanceLevel;

    const ABSORPTION_WINDOW: usize = 8; // Max steps to look back
    let mut to_absorb = Vec::new();

    for (j, step) in steps.iter().enumerate() {
        // Detect PolynomialIdentity step by poly_proof data
        if step.poly_proof().is_some() {
            // Walk backwards within window and absorb mechanical steps
            let window_start = j.saturating_sub(ABSORPTION_WINDOW);

            for i in (window_start..j).rev() {
                let s = &steps[i];

                // GUARD: Never absorb steps with Medium+ importance
                // These are marked as "didactically important" and should always be visible
                if s.importance >= ImportanceLevel::Medium {
                    break; // Stop absorption chain - this step is a barrier
                }

                // Only absorb mechanical rules
                if !is_mechanical_rule(&s.rule_name) {
                    // Non-mechanical step breaks the absorption chain
                    break;
                }

                to_absorb.push(i);
            }
        }
    }

    to_absorb
}

/// Enhanced step optimization with polynomial identity absorption
pub fn optimize_steps_with_absorption(steps: Vec<Step>) -> Vec<Step> {
    let steps_to_absorb: std::collections::HashSet<usize> =
        find_steps_to_absorb_for_polyzero(&steps)
            .into_iter()
            .collect();

    let filtered: Vec<_> = steps
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !steps_to_absorb.contains(i))
        .map(|(_, s)| s)
        .collect();

    // Apply normal optimization on the filtered steps
    optimize_steps(filtered)
}
