//! Witness survival checks and assumption classification.

use super::normalization::conditions_equivalent;
use super::ImplicitCondition;
use cas_ast::{Context, ExprId};
use cas_math::expr_domain::{
    exprs_equivalent, is_odd_power_of, is_positive_multiple_of, is_power_of_base,
};
use cas_math::expr_witness::{self, WitnessKind as MathWitnessKind};

// =============================================================================
// Witness Survival
// =============================================================================

/// Kind of witness to look for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WitnessKind {
    /// sqrt(t) or t^(1/2) for NonNegative(t)
    Sqrt,
    /// ln(t) or log(t) for Positive(t)
    Log,
    /// 1/t or Div(_, t) for NonZero(t)
    Division,
}

impl From<WitnessKind> for MathWitnessKind {
    fn from(value: WitnessKind) -> Self {
        match value {
            WitnessKind::Sqrt => MathWitnessKind::Sqrt,
            WitnessKind::Log => MathWitnessKind::Log,
            WitnessKind::Division => MathWitnessKind::Division,
        }
    }
}

/// Check if a witness for a condition survives in the output expression.
///
/// This is the critical safety check: a condition from implicit domain
/// is only valid if the witness that enforces it still exists in the output.
///
/// # Arguments
/// * `ctx` - AST context
/// * `target` - The target expression (e.g., `x` in `x ≥ 0`)
/// * `output` - The expression to search for witnesses
/// * `kind` - What kind of witness to look for
///
/// # Returns
/// `true` if a witness survives in output, `false` otherwise.
pub fn witness_survives(ctx: &Context, target: ExprId, output: ExprId, kind: WitnessKind) -> bool {
    expr_witness::witness_survives(ctx, target, output, kind.into())
}

/// Check if a witness survives in the context of the full expression tree,
/// considering that a specific node is being replaced with a new value.
///
/// This is the key function for implicit domain safety: it ensures that when
/// we simplify `sqrt(x)² → x`, a witness (`sqrt(x)`) still exists elsewhere
/// in the expression tree.
///
/// # Arguments
/// * `ctx` - AST context
/// * `target` - The target expression (e.g., `x` in `x ≥ 0`)
/// * `root` - The root expression of the full tree
/// * `replaced_node` - The node being replaced (will be skipped in search)
/// * `replacement` - Optional replacement value to search in instead
/// * `kind` - What kind of witness to look for
///
/// # Returns
/// `true` if a witness survives in the tree after replacement
pub fn witness_survives_in_context(
    ctx: &Context,
    target: ExprId,
    root: ExprId,
    replaced_node: ExprId,
    replacement: Option<ExprId>,
    kind: WitnessKind,
) -> bool {
    expr_witness::witness_survives_in_context(
        ctx,
        target,
        root,
        replaced_node,
        replacement,
        kind.into(),
    )
}

// =============================================================================
// Assumption Classification (V2.12.13)
// =============================================================================

/// Context for domain condition tracking during step processing.
///
/// Used by the central classifier to determine whether conditions are
/// derived from input requires or newly introduced.
#[derive(Debug, Clone, Default)]
pub struct DomainContext {
    /// Conditions inferred from the original input expression
    pub global_requires: Vec<ImplicitCondition>,
    /// Conditions introduced by previous steps (accumulated)
    pub introduced_requires: Vec<ImplicitCondition>,
}

impl DomainContext {
    /// Create a new DomainContext with global requires from the input expression.
    pub fn new(global_requires: Vec<ImplicitCondition>) -> Self {
        Self {
            global_requires,
            introduced_requires: Vec::new(),
        }
    }

    /// Check if a condition is implied by the known requires (global ∪ introduced).
    ///
    /// Implication rules:
    /// - Exact polynomial equivalence
    /// - x > 0 is implied by x^(odd positive) > 0  
    /// - x ≠ 0 is implied by x > 0
    pub fn is_condition_implied(&self, ctx: &Context, cond: &ImplicitCondition) -> bool {
        let all_known: Vec<_> = self
            .global_requires
            .iter()
            .chain(self.introduced_requires.iter())
            .collect();

        for known in all_known {
            // Direct equivalence check
            if conditions_equivalent(ctx, cond, known) {
                return true;
            }

            // Implication rules
            match (cond, known) {
                // x ≠ 0 is implied by x > 0 or x ≥ 0 (for our purposes, x > 0)
                (ImplicitCondition::NonZero(target), ImplicitCondition::Positive(source)) => {
                    if exprs_equivalent(ctx, *target, *source) {
                        return true;
                    }
                }
                // x > 0 is implied by x^(positive odd) > 0 (e.g., b > 0 implied by b^3 > 0)
                // x^n > 0 is implied by x > 0 (e.g., a^2 > 0 implied by a > 0)
                (ImplicitCondition::Positive(target), ImplicitCondition::Positive(source)) => {
                    // Check if source is target^(odd positive) -> target is implied
                    if is_odd_power_of(ctx, *source, *target) {
                        return true;
                    }
                    // NEW: Check if target is source^n -> target is implied by source
                    // e.g., a^2 > 0 is implied by a > 0
                    if is_power_of_base(ctx, *target, *source) {
                        return true;
                    }
                }
                // V2.15.8: x ≥ 0 is implied by x > 0 (strict positivity implies non-negativity)
                (ImplicitCondition::NonNegative(target), ImplicitCondition::Positive(source)) => {
                    if exprs_equivalent(ctx, *target, *source) {
                        return true;
                    }
                }
                // V2.15.8: x ≥ 0 is implied by k*x ≥ 0 when k > 0
                // e.g., x ≥ 0 implied by 4*x ≥ 0 (since we know 4 > 0)
                (
                    ImplicitCondition::NonNegative(target),
                    ImplicitCondition::NonNegative(source),
                ) => {
                    // Check direct equivalence first (handled above in conditions_equivalent)
                    // Check if source is k*target where k > 0
                    if is_positive_multiple_of(ctx, *source, *target) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }

    /// Add a newly introduced condition (from a step that introduces constraints).
    pub fn add_introduced(&mut self, cond: ImplicitCondition) {
        self.introduced_requires.push(cond);
    }
}

/// Classify an AssumptionEvent based on whether its condition is implied by known requires.
///
/// # Reclassification Logic (V2.12.13)
///
/// 1. If the event has kind `BranchChoice`, `HeuristicAssumption`, or `DomainExtension`:
///    **Keep as-is** (never promote to requires)
///
/// 2. If the event's condition IS implied by `global ∪ introduced`:
///    Reclassify to `DerivedFromRequires` (will not be displayed)
///
/// 3. If NOT implied AND kind was `DerivedFromRequires` or `RequiresIntroduced`:
///    Promote to `RequiresIntroduced` (will be displayed, add to introduced)
///
/// # Returns
/// The new kind for the event, and whether to add to introduced_requires.
pub fn classify_assumption(
    ctx: &Context,
    dc: &DomainContext,
    event: &crate::assumptions::AssumptionEvent,
) -> (
    crate::assumptions::AssumptionKind,
    Option<ImplicitCondition>,
) {
    use crate::assumptions::AssumptionKind;

    // Rule 1: Branch/Domain never get reclassified (they are structural, not algebraic)
    match event.kind {
        AssumptionKind::BranchChoice | AssumptionKind::DomainExtension => {
            return (event.kind, None);
        }
        _ => {}
    }

    // Try to convert the event to an ImplicitCondition
    let implicit_cond = assumption_to_condition(event);

    match implicit_cond {
        Some(cond) => {
            // Check if this condition is already implied by global/introduced requires
            if dc.is_condition_implied(ctx, &cond) {
                // If it was HeuristicAssumption but is implied, downgrade to DerivedFromRequires
                // This prevents showing ⚠ b > 0 when b > 0 is already in Requires
                (AssumptionKind::DerivedFromRequires, None)
            } else {
                // Not implied - behavior depends on original kind
                match event.kind {
                    AssumptionKind::HeuristicAssumption => {
                        // Keep as HeuristicAssumption (shows ⚠) since it's a new assumption
                        (AssumptionKind::HeuristicAssumption, None)
                    }
                    _ => {
                        // Promote to RequiresIntroduced
                        (AssumptionKind::RequiresIntroduced, Some(cond))
                    }
                }
            }
        }
        None => {
            // Cannot convert to condition (e.g., InvTrigPrincipalRange)
            // Keep original kind
            (event.kind, None)
        }
    }
}

/// Convert an AssumptionEvent to an ImplicitCondition if possible.
///
/// Uses the `expr_id` field (V2.12.13) for proper condition comparison.
fn assumption_to_condition(
    event: &crate::assumptions::AssumptionEvent,
) -> Option<ImplicitCondition> {
    use crate::assumptions::AssumptionKey;

    // V2.12.13: Use expr_id if available for proper condition creation
    let expr_id = event.expr_id?;

    match &event.key {
        AssumptionKey::NonZero { .. } => Some(ImplicitCondition::NonZero(expr_id)),
        AssumptionKey::Positive { .. } => Some(ImplicitCondition::Positive(expr_id)),
        AssumptionKey::NonNegative { .. } => Some(ImplicitCondition::NonNegative(expr_id)),
        // Defined has no direct ImplicitCondition counterpart
        AssumptionKey::Defined { .. } => None,
        // Branch choices are not conditions
        AssumptionKey::InvTrigPrincipalRange { .. } => None,
        AssumptionKey::ComplexPrincipalBranch { .. } => None,
    }
}

/// Filter and reclassify a list of AssumptionEvents in place.
///
/// After calling, events have updated `kind` fields.
/// Use `event.kind.should_display()` to determine which to show.
pub fn classify_assumptions_in_place(
    ctx: &Context,
    dc: &mut DomainContext,
    events: &mut [crate::assumptions::AssumptionEvent],
) {
    for event in events.iter_mut() {
        let (new_kind, new_cond) = classify_assumption(ctx, dc, event);
        event.kind = new_kind;
        if let Some(cond) = new_cond {
            dc.add_introduced(cond);
        }
    }
}

#[cfg(test)]
mod tests;
