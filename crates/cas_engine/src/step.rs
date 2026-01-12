//! # Step Module - Didactic Simplification Steps
//!
//! This module defines the `Step` struct for recording transformation steps
//! during expression simplification.
//!
//! ## Pre-order Didactic Step Pattern (V2.12+)
//!
//! For pre-order rules (rules that intercept before children are simplified),
//! follow this pattern to create coherent, didactically correct step sequences:
//!
//! ### 1. Create Intermediate Expressions for Each State
//!
//! ```rust,ignore
//! // Example: Difference of Squares in a fraction
//! let factored_num = ctx.add(Expr::Mul(a_minus_b, a_plus_b));
//! let intermediate_orig_den = ctx.add(Expr::Div(factored_num, den));      // State after Factor
//! let intermediate_simplified = ctx.add(Expr::Div(factored_num, den_simplified)); // State after Combine
//! ```
//!
//! This creates snapshots for each transformation step.
//!
//! ### 2. Use `before_local` / `after_local` to Focus the Rule Line
//!
//! ```rust,ignore
//! factor_step.before_local = Some(numerator);       // Only show: numerator -> factored
//! factor_step.after_local = Some(factored_num);
//! ```
//!
//! - `before`/`after`: Full expression state (for Before/After display)
//! - `before_local`/`after_local`: Focused sub-expression (for Rule: X -> Y line)
//!
//! ### 3. Chain Steps Correctly
//!
//! Each step's `after` must equal the next step's `before`:
//!
//! - Step 1: `before=expr_id`, `after=intermediate_orig_den`
//! - Step 2: `before=intermediate_orig_den`, `after=intermediate_simplified`
//! - Step 3: `before=intermediate_simplified`, `after=final_result`
//!
//! ### 4. Add Conditional Steps Only When Needed
//!
//! ```rust,ignore
//! if den != den_simplified {
//!     // Add "Combine like terms" step only if denominator actually changed
//!     let mut step = Step::new(...);
//!     step.before_local = Some(den);
//!     step.after_local = Some(den_simplified);
//!     steps.push(step);
//! }
//! ```
//!
//! See `rules/algebra/mod.rs::try_difference_of_squares_preorder` for a complete
//! implementation example.

use cas_ast::ExprId;

#[derive(Debug, Clone, PartialEq)]
pub enum PathStep {
    Left,       // Binary op left / Div numerator
    Right,      // Binary op right / Div denominator
    Arg(usize), // Function argument index
    Base,       // Power base
    Exponent,   // Power exponent
    Inner,      // Negation inner / other unary
}

impl PathStep {
    /// Convert to child index for ExprPath (V2.9.16)
    pub fn to_child_index(&self) -> u8 {
        match self {
            PathStep::Left => 0,
            PathStep::Right => 1,
            PathStep::Base => 0,
            PathStep::Exponent => 1,
            PathStep::Inner => 0,
            PathStep::Arg(i) => *i as u8,
        }
    }
}

/// Convert a Vec<PathStep> to ExprPath (V2.9.16)
///
/// ExprPath uses numeric indices for children:
/// - 0 = left/base/numerator/inner
/// - 1 = right/exponent/denominator
/// - i = function argument index
pub fn pathsteps_to_expr_path(steps: &[PathStep]) -> cas_ast::ExprPath {
    steps.iter().map(|s| s.to_child_index()).collect()
}

/// Importance level for step filtering
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ImportanceLevel {
    Trivial = 0, // x + 0 → x, x * 1 → x
    Low = 1,     // Combine constants, simple evaluations
    Medium = 2,  // Most algebraic transformations
    High = 3,    // Factor, expand, integrate, complex transforms
}

/// Category of step for grouping and filtering by type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StepCategory {
    #[default]
    General, // Default for all rules
    Canonicalize, // Ordering, normalization
    Simplify,     // Algebraic simplification
    Expand,       // Distribution, expansion
    Factor,       // Factorization
    Rationalize,  // Rationalization
    ConstEval,    // Constant evaluation
    ConstFold,    // Constant folding pass
    Domain,       // Domain-related transforms
    Solve,        // Solver steps
    Substitute,   // Substitution
    Limits,       // Limit calculations
}

#[derive(Debug, Clone)]
pub struct Step {
    pub description: String,
    pub rule_name: String,
    /// The local expression before transformation (the subexpression that was rewritten)
    pub before: ExprId,
    /// The local expression after transformation
    pub after: ExprId,
    /// Path from root to the transformed node (kept for debugging/reference)
    pub path: Vec<PathStep>,
    /// String representation of after (for display)
    pub after_str: Option<String>,
    /// Complete root expression BEFORE this step's transformation
    pub global_before: Option<ExprId>,
    /// Complete root expression AFTER this step's transformation
    pub global_after: Option<ExprId>,
    /// Optional: The specific pattern matched (for n-ary rules like atan(x)+atan(1/x))
    /// Use this for "Rule: X -> Y" display if set, otherwise use before/after
    pub before_local: Option<ExprId>,
    /// Optional: The specific result of the pattern (for n-ary rules)
    pub after_local: Option<ExprId>,
    /// Optional domain assumption used by the rule (e.g., "x > 0" assumed)
    /// LEGACY: use assumption_events for structured emission, this is fallback.
    /// Structured assumption events (preferred over domain_assumption string).
    /// Propagated from Rewrite.assumption_events during step creation.
    pub assumption_events: smallvec::SmallVec<[crate::assumptions::AssumptionEvent; 1]>,
    /// Required conditions for validity (implicit domain preservation) - NOT assumptions!
    /// These are conditions that were already implicitly required by the input expression.
    /// Propagated from Rewrite.required_conditions during step creation.
    pub required_conditions: Vec<crate::implicit_domain::ImplicitCondition>,
    /// Importance level for step filtering (from Rule::importance())
    pub importance: ImportanceLevel,
    /// Category of step for grouping (from Rule::category() or pipeline origin)
    pub category: StepCategory,
    /// Optional: Polynomial proof data for identity cancellation (PolyZero airbag)
    /// Propagated from Rewrite.poly_proof during step creation for didactic display
    pub poly_proof: Option<crate::multipoly_display::PolynomialProofData>,
}

impl Step {
    pub fn new(
        description: &str,
        rule_name: &str,
        before: ExprId,
        after: ExprId,
        path: Vec<PathStep>,
        context: Option<&cas_ast::Context>,
    ) -> Self {
        let after_str = context.map(|ctx| {
            // Unwrap __hold for display purposes
            let display_id = match ctx.get(after) {
                cas_ast::Expr::Function(name, args) if name == "__hold" && args.len() == 1 => {
                    args[0]
                }
                _ => after,
            };
            format!(
                "{}",
                cas_ast::DisplayExpr {
                    context: ctx,
                    id: display_id
                }
            )
        });
        Self {
            description: description.to_string(),
            rule_name: rule_name.to_string(),
            before,
            after,
            path,
            after_str,
            global_before: None,
            global_after: None,
            before_local: None,
            after_local: None,
            assumption_events: Default::default(),
            required_conditions: vec![],
            poly_proof: None,
            importance: ImportanceLevel::Low, // Default, will be overwritten by caller
            category: StepCategory::General,  // Default, can be set by caller
        }
    }

    /// Create a compact step without display formatting (for StepsMode::Compact).
    /// Skips the expensive format! call and sets after_str to None.
    pub fn new_compact(description: &str, rule_name: &str, before: ExprId, after: ExprId) -> Self {
        Self {
            description: description.to_string(),
            rule_name: rule_name.to_string(),
            before,
            after,
            path: Vec::new(),
            after_str: None,
            global_before: None,
            global_after: None,
            before_local: None,
            after_local: None,
            assumption_events: Default::default(),
            required_conditions: vec![],
            poly_proof: None,
            importance: ImportanceLevel::Low, // Default, will be overwritten by caller
            category: StepCategory::General,  // Default, can be set by caller
        }
    }

    /// Create a step with complete global snapshots before and after transformation
    #[allow(clippy::too_many_arguments)] // All parameters are semantically distinct
    pub fn with_snapshots(
        description: &str,
        rule_name: &str,
        before: ExprId,
        after: ExprId,
        path: Vec<PathStep>,
        context: Option<&cas_ast::Context>,
        global_before: ExprId,
        global_after: ExprId,
    ) -> Self {
        let mut step = Self::new(description, rule_name, before, after, path, context);
        step.global_before = Some(global_before);
        step.global_after = Some(global_after);
        step
    }

    /// Get the importance/significance of this step
    /// Uses the declarative importance set when the Step was created (from Rule::importance())
    /// With overrides for special cases (no-op, domain assumptions)
    pub fn get_importance(&self) -> ImportanceLevel {
        // No-op steps are always trivial (before == after means no visible change)
        if self.before == self.after {
            return ImportanceLevel::Trivial;
        }

        // Steps with assumptions are always shown - important for user awareness
        if !self.assumption_events.is_empty() {
            return ImportanceLevel::Medium;
        }

        // Use the declarative importance from the Rule
        self.importance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_importance_classification() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");

        // Test 1: Default importance is Low (from Step::new)
        let step = Step::new("Some rule", "Some Rule", x, y, vec![], Some(&ctx));
        assert_eq!(step.get_importance(), ImportanceLevel::Low);

        // Test 2: No-op steps (same before/after) are always Trivial
        let step = Step::new("No change", "Any Rule", x, x, vec![], Some(&ctx));
        assert_eq!(step.get_importance(), ImportanceLevel::Trivial);

        // Test 3: Steps with assumption_events are bumped to Medium
        let mut step = Step::new("Rule with assumption", "Rule", x, y, vec![], Some(&ctx));
        step.assumption_events
            .push(crate::assumptions::AssumptionEvent {
                key: crate::assumptions::AssumptionKey::NonZero {
                    expr_fingerprint: 12345,
                },
                expr_display: "x".to_string(),
                message: "Assuming x ≠ 0".to_string(),
            });
        assert_eq!(step.get_importance(), ImportanceLevel::Medium);

        // Test 4: Declaratively set importance is respected
        let mut step = Step::new("Medium rule", "Important Rule", x, y, vec![], Some(&ctx));
        step.importance = ImportanceLevel::Medium;
        assert_eq!(step.get_importance(), ImportanceLevel::Medium);

        let mut step = Step::new("High rule", "Major Transform", x, y, vec![], Some(&ctx));
        step.importance = ImportanceLevel::High;
        assert_eq!(step.get_importance(), ImportanceLevel::High);
    }
}

// =============================================================================
// V2.9.9: Type-Safe Eval Step Wrappers
// =============================================================================

/// V2.9.9: Internal steps before cleanup/enrichment.
/// NOT exported from crate. Use for debugging only.
#[allow(dead_code)]
pub(crate) struct RawEvalSteps(pub Vec<Step>);

/// V2.9.9: Display-ready steps after cleanup/enrichment.
/// The ONLY type external consumers should receive.
///
/// This wrapper enforces that step cleanup has been applied via
/// `eval_step_pipeline::to_display_steps()`. No raw steps can
/// escape to display layers (Text, HTML, JSON).
#[derive(Debug, Clone, Default)]
pub struct DisplayEvalSteps(pub Vec<Step>);

impl DisplayEvalSteps {
    /// Create an empty DisplayEvalSteps.
    pub fn empty() -> Self {
        Self(vec![])
    }

    /// Check if there are no steps.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get the number of steps.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Iterate over steps.
    pub fn iter(&self) -> std::slice::Iter<'_, Step> {
        self.0.iter()
    }

    /// Get inner Vec reference.
    pub fn as_slice(&self) -> &[Step] {
        &self.0
    }

    /// Consume and return inner Vec.
    pub fn into_inner(self) -> Vec<Step> {
        self.0
    }
}

// Deref for ergonomic slice access (enables &display_steps[..], .len(), etc.)
impl std::ops::Deref for DisplayEvalSteps {
    type Target = [Step];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// IntoIterator for `for step in &output.steps` pattern
impl<'a> IntoIterator for &'a DisplayEvalSteps {
    type Item = &'a Step;
    type IntoIter = std::slice::Iter<'a, Step>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

// Consuming iterator for `for step in output.steps`
impl IntoIterator for DisplayEvalSteps {
    type Item = Step;
    type IntoIter = std::vec::IntoIter<Step>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
