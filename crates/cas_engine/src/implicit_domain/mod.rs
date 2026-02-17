//! Implicit Domain Inference.
//!
//! This module infers domain constraints that are implicitly required by
//! expression structure. For example, `sqrt(x)` in RealOnly mode implies `x ≥ 0`.
//!
//! # Key Concepts
//!
//! - **Implicit Domain**: Constraints derived from expression structure, not assumptions.
//! - **Witness**: The subexpression that enforces a constraint (e.g., `sqrt(x)` for `x ≥ 0`).
//! - **Witness Survival**: A constraint is only valid if its witness survives in the output.
//!
//! # Usage
//!
//! ```ignore
//! let implicit = infer_implicit_domain(ctx, root, ValueDomain::RealOnly);
//! // Later, when checking if x ≥ 0 is valid:
//! if implicit.contains_nonnegative(x) && witness_survives(ctx, x, output, WitnessKind::Sqrt) {
//!     // Can use ProvenImplicit
//! }
//! ```

mod inference;
mod normalization;
mod witness;

// Re-export all public items from submodules
pub(crate) use inference::contains_variable;
pub use inference::{
    check_analytic_expansion, derive_requires_from_equation, domain_delta_check,
    expands_analytic_domain, expands_analytic_in_context, infer_implicit_domain,
    AnalyticExpansionResult, DomainDelta,
};
pub use normalization::{
    normalize_and_dedupe_conditions, normalize_condition, normalize_condition_expr,
    render_conditions_normalized,
};
pub use witness::{
    classify_assumption, classify_assumptions_in_place, witness_survives,
    witness_survives_in_context, DomainContext, WitnessKind,
};

use cas_ast::{Context, ExprId};
use std::collections::HashSet;

// =============================================================================
// Domain Inference Call Counter (for regression testing)
// =============================================================================
// Tracks how many times infer_implicit_domain is called per simplify operation.
// This helps detect regressions where rules accidentally recompute the domain.

use std::cell::Cell;

thread_local! {
    static INFER_DOMAIN_CALLS: Cell<usize> = const { Cell::new(0) };
}

/// Reset the domain inference call counter. Call at the start of each simplify.
#[inline]
pub fn infer_domain_calls_reset() {
    INFER_DOMAIN_CALLS.with(|c| c.set(0));
}

/// Get the current domain inference call count.
#[inline]
pub fn infer_domain_calls_get() -> usize {
    INFER_DOMAIN_CALLS.with(|c| c.get())
}

/// Increment the domain inference call counter.
#[inline]
fn infer_domain_calls_inc() {
    INFER_DOMAIN_CALLS.with(|c| c.set(c.get() + 1));
}

// =============================================================================
// Implicit Condition Types
// =============================================================================

/// An implicit condition inferred from expression structure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ImplicitCondition {
    /// x ≥ 0 (from sqrt(x) or x^(1/2))
    NonNegative(ExprId),
    /// x > 0 (from ln(x) or log(x))  
    Positive(ExprId),
    /// x ≠ 0 (from 1/x or Div(_, x))
    NonZero(ExprId),
}

impl ImplicitCondition {
    /// Human-readable display for REPL/UI.
    pub fn display(&self, ctx: &Context) -> String {
        use cas_formatter::DisplayExpr;
        match self {
            ImplicitCondition::NonNegative(e) => {
                format!(
                    "{} ≥ 0",
                    DisplayExpr {
                        context: ctx,
                        id: *e
                    }
                )
            }
            ImplicitCondition::Positive(e) => {
                format!(
                    "{} > 0",
                    DisplayExpr {
                        context: ctx,
                        id: *e
                    }
                )
            }
            ImplicitCondition::NonZero(e) => {
                format!(
                    "{} ≠ 0",
                    DisplayExpr {
                        context: ctx,
                        id: *e
                    }
                )
            }
        }
    }

    /// Check if this condition is trivial (always true or on a constant expression).
    /// Trivial conditions like "2 > 0" or "x² ≥ 0" should be filtered from display.
    pub fn is_trivial(&self, ctx: &Context) -> bool {
        let expr = match self {
            ImplicitCondition::NonNegative(e) => *e,
            ImplicitCondition::Positive(e) => *e,
            ImplicitCondition::NonZero(e) => *e,
        };

        // Case 1: No variables = fully numeric constant (always trivial)
        if !inference::contains_variable(ctx, expr) {
            return true;
        }

        // Case 2: For NonNegative, check if expression is always ≥ 0 (like x²)
        if let ImplicitCondition::NonNegative(e) = self {
            // Check for patterns that are always non-negative:
            // - x² (even power of variable)
            // - |x| (absolute value)
            // - x² + y² (sum of squares)
            if inference::is_always_nonnegative(ctx, *e) {
                return true;
            }
        }

        false
    }

    /// Check if this condition's witness survives in the output expression.
    ///
    /// This is used for the "witness survival" display policy:
    /// - If `sqrt(x)` survives in output → `x ≥ 0` is implicitly shown (no need to display)
    /// - If `sqrt(x)` was consumed (e.g., `sqrt(x)^2 → x`) → `x ≥ 0` must be displayed
    pub fn witness_survives_in(&self, ctx: &Context, output: ExprId) -> bool {
        match self {
            ImplicitCondition::NonNegative(e) => {
                witness_survives(ctx, *e, output, WitnessKind::Sqrt)
            }
            ImplicitCondition::Positive(e) => witness_survives(ctx, *e, output, WitnessKind::Log),
            ImplicitCondition::NonZero(e) => {
                witness_survives(ctx, *e, output, WitnessKind::Division)
            }
        }
    }
}

// =============================================================================
// Display Level for Requires
// =============================================================================

/// Display level for required conditions.
///
/// Controls how many requires are shown to the user:
/// - `Essential`: Only show requires whose witness was consumed (pedagogically important)
/// - `All`: Show all requires including those with surviving witnesses
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RequiresDisplayLevel {
    /// Show only essential requires (witness consumed or from equation derivation)
    #[default]
    Essential,
    /// Show all requires including implicit ones (witness survives)
    All,
}

/// Filter required conditions based on display level and witness survival.
///
/// In `Essential` mode, only shows requires whose witness was consumed:
/// - `sqrt(x) + 1` → x≥0 witness survives → HIDE
/// - `sqrt(x)^2 → x` → x≥0 witness consumed → SHOW
///
/// In `All` mode, shows everything.
///
/// # Arguments
/// * `requires` - All required conditions
/// * `ctx` - AST context  
/// * `result` - The result expression (to check witness survival)
/// * `level` - Display level (Essential or All)
///
/// # Returns
/// Filtered list of conditions to display
pub fn filter_requires_for_display<'a>(
    requires: &'a [ImplicitCondition],
    ctx: &Context,
    result: ExprId,
    level: RequiresDisplayLevel,
) -> Vec<&'a ImplicitCondition> {
    requires
        .iter()
        .filter(|cond| {
            // Always show if level is All
            if level == RequiresDisplayLevel::All {
                return true;
            }

            // Essential: show only if witness does NOT survive
            !cond.witness_survives_in(ctx, result)
        })
        .collect()
}

/// Set of implicit conditions inferred from an expression.
#[derive(Debug, Clone, Default)]
pub struct ImplicitDomain {
    conditions: HashSet<ImplicitCondition>,
}

impl ImplicitDomain {
    /// Create an empty implicit domain.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Check if domain is empty.
    pub fn is_empty(&self) -> bool {
        self.conditions.is_empty()
    }

    /// Check if an expression has an implicit NonNegative constraint.
    pub fn contains_nonnegative(&self, expr: ExprId) -> bool {
        self.conditions
            .contains(&ImplicitCondition::NonNegative(expr))
    }

    /// Check if an expression has an implicit Positive constraint.
    pub fn contains_positive(&self, expr: ExprId) -> bool {
        self.conditions.contains(&ImplicitCondition::Positive(expr))
    }

    /// Check if an expression has an implicit NonZero constraint.
    pub fn contains_nonzero(&self, expr: ExprId) -> bool {
        self.conditions.contains(&ImplicitCondition::NonZero(expr))
    }

    /// Add a NonNegative condition.
    pub(crate) fn add_nonnegative(&mut self, expr: ExprId) {
        self.conditions.insert(ImplicitCondition::NonNegative(expr));
    }

    /// Add a Positive condition.
    pub(crate) fn add_positive(&mut self, expr: ExprId) {
        self.conditions.insert(ImplicitCondition::Positive(expr));
    }

    /// Add a NonZero condition.
    pub(crate) fn add_nonzero(&mut self, expr: ExprId) {
        self.conditions.insert(ImplicitCondition::NonZero(expr));
    }

    /// Get all conditions (for iteration/comparison)
    pub fn conditions(&self) -> &HashSet<ImplicitCondition> {
        &self.conditions
    }

    /// Get mutable access to conditions
    pub fn conditions_mut(&mut self) -> &mut HashSet<ImplicitCondition> {
        &mut self.conditions
    }

    /// Check if this domain is a superset of another (contains all its conditions)
    pub fn contains_all(&self, other: &ImplicitDomain) -> bool {
        other.conditions.is_subset(&self.conditions)
    }

    /// Get conditions that are in self but not in other (dropped conditions)
    pub fn dropped_from<'a>(&'a self, other: &'a ImplicitDomain) -> Vec<&'a ImplicitCondition> {
        self.conditions.difference(&other.conditions).collect()
    }

    /// Merge conditions from another domain into this one
    pub fn extend(&mut self, other: &ImplicitDomain) {
        for cond in &other.conditions {
            self.conditions.insert(cond.clone());
        }
    }

    /// Convert to a ConditionSet for solver use
    pub fn to_condition_set(&self) -> cas_ast::ConditionSet {
        let predicates: Vec<cas_ast::ConditionPredicate> =
            self.conditions.iter().map(|c| c.into()).collect();
        cas_ast::ConditionSet::from_predicates(predicates)
    }
}

// =============================================================================
// Adapters: ImplicitCondition ↔ ConditionPredicate
// =============================================================================

impl From<&ImplicitCondition> for cas_ast::ConditionPredicate {
    fn from(cond: &ImplicitCondition) -> Self {
        match cond {
            ImplicitCondition::NonNegative(e) => cas_ast::ConditionPredicate::NonNegative(*e),
            ImplicitCondition::Positive(e) => cas_ast::ConditionPredicate::Positive(*e),
            ImplicitCondition::NonZero(e) => cas_ast::ConditionPredicate::NonZero(*e),
        }
    }
}

impl From<ImplicitCondition> for cas_ast::ConditionPredicate {
    fn from(cond: ImplicitCondition) -> Self {
        (&cond).into()
    }
}

impl TryFrom<&cas_ast::ConditionPredicate> for ImplicitCondition {
    type Error = ();

    fn try_from(pred: &cas_ast::ConditionPredicate) -> Result<Self, Self::Error> {
        match pred {
            cas_ast::ConditionPredicate::NonNegative(e) => Ok(ImplicitCondition::NonNegative(*e)),
            cas_ast::ConditionPredicate::Positive(e) => Ok(ImplicitCondition::Positive(*e)),
            cas_ast::ConditionPredicate::NonZero(e) => Ok(ImplicitCondition::NonZero(*e)),
            _ => Err(()), // Defined, InvTrigPrincipalRange, EqZero, EqOne not mapped
        }
    }
}
