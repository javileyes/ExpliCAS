//! Domain condition implication context shared by engine/solver layers.

use crate::domain_condition::ImplicitCondition;
use crate::domain_normalization::conditions_equivalent;
use cas_ast::Context;
use cas_math::expr_domain::{
    exprs_equivalent, is_odd_power_of, is_positive_multiple_of, is_power_of_base,
};

/// Context for domain condition tracking during step processing.
#[derive(Debug, Clone, Default)]
pub struct DomainContext {
    /// Conditions inferred from the original input expression.
    pub global_requires: Vec<ImplicitCondition>,
    /// Conditions introduced by previous steps (accumulated).
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
            if conditions_equivalent(ctx, cond, known) {
                return true;
            }

            match (cond, known) {
                (ImplicitCondition::NonZero(target), ImplicitCondition::Positive(source)) => {
                    if exprs_equivalent(ctx, *target, *source) {
                        return true;
                    }
                }
                (ImplicitCondition::Positive(target), ImplicitCondition::Positive(source)) => {
                    if is_odd_power_of(ctx, *source, *target) {
                        return true;
                    }
                    if is_power_of_base(ctx, *target, *source) {
                        return true;
                    }
                }
                (ImplicitCondition::NonNegative(target), ImplicitCondition::Positive(source)) => {
                    if exprs_equivalent(ctx, *target, *source) {
                        return true;
                    }
                }
                (
                    ImplicitCondition::NonNegative(target),
                    ImplicitCondition::NonNegative(source),
                ) => {
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
