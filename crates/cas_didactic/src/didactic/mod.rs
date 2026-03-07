//! Didactic Step Enhancement Layer
//!
//! This module provides visualization-layer enrichment of engine steps
//! without modifying the core engine. It post-processes steps to add
//! instructive detail for human learners.
//!
//! # Architecture
//! - Pure post-processing: never modifies engine behavior
//! - Optional: can be enabled/disabled via verbosity
//! - Extensible: easy to add new enrichers
//!
//! # Contract (V2.12.13)
//!
//! **SubSteps explain techniques within a Step. They MUST NOT duplicate
//! decompositions that already exist as chained Steps via ChainedRewrite.**
//!
//! When `step.is_chained == true`, the Step was created from a ChainedRewrite
//! and already has proper before/after expressions. Skip substep generation
//! that would duplicate this information (e.g., GCD factorization substeps).
//!
//! # When to use which:
//!
//! - **ChainedRewrite**: Multi-step algebraic decomposition with real ExprIds
//!   (e.g., Factor → Cancel as separate visible Steps)
//! - **SubSteps**: Educational annotation explaining technique (e.g., "Find conjugate")
//!
//! # Example
//! ```ignore
//! let enriched = didactic::enrich_steps(&ctx, original_expr, steps);
//! for step in enriched {
//!     println!("{}", step.base_step.description);
//!     for sub in &step.sub_steps {
//!         println!("    → {}", sub.description);
//!     }
//! }
//! ```

mod cli_render;
mod display_policy;
mod enrichment_pipeline;
mod fraction_steps;
mod fraction_sum_analysis;
mod gcd_factorization;
mod latex_plain_text;
mod nested_fraction_analysis;
mod nested_fractions;
mod polynomial_identity;
mod rationalization;
mod root_denesting;
mod shared_numeric;
mod step_visibility;
mod sum_three_cubes;

use cas_solver::Step;

pub use cli_render::{
    format_cli_simplification_steps, format_cli_simplification_steps_with_simplifier,
};
pub use display_policy::{
    build_cli_substeps_render_plan, build_timeline_substeps_render_plan, classify_sub_steps,
    CliSubstepsRenderPlan, StepDisplayMode, SubStepClassification, TimelineSubstepsRenderPlan,
};
pub use enrichment_pipeline::{enrich_steps, get_standalone_substeps};
pub use latex_plain_text::latex_to_plain_text;
pub(crate) use shared_numeric::{
    collect_add_terms, format_fraction, lcm_bigint, try_as_fraction, IsOne,
};
pub(crate) use step_visibility::{
    clone_steps_matching_visibility, infer_original_expr_for_steps, step_matches_visibility,
    StepVisibility,
};
pub use step_visibility::{is_high_or_higher_step, is_medium_or_higher_step};

/// An enriched step with optional sub-steps for didactic explanation
#[derive(Debug, Clone)]
pub struct EnrichedStep {
    /// The original step from the engine
    pub base_step: Step,
    /// Synthetic sub-steps that explain hidden operations
    pub sub_steps: Vec<SubStep>,
}

/// A synthetic sub-step that explains a hidden operation
#[derive(Debug, Clone)]
pub struct SubStep {
    /// Human-readable description of the operation
    pub description: String,
    /// Expression before the operation (plain text for CLI display)
    pub before_expr: String,
    /// Expression after the operation (plain text for CLI display)
    pub after_expr: String,
    /// Optional LaTeX for `before_expr` (for web/MathJax rendering).
    /// When set, the JSON layer uses this instead of `before_expr`.
    pub before_latex: Option<String>,
    /// Optional LaTeX for `after_expr` (for web/MathJax rendering).
    /// When set, the JSON layer uses this instead of `after_expr`.
    pub after_latex: Option<String>,
}

impl SubStep {
    /// Create a plain-text sub-step (no LaTeX).
    /// Text will be wrapped in `\text{}` by the JSON layer.
    pub fn new(
        description: impl Into<String>,
        before_expr: impl Into<String>,
        after_expr: impl Into<String>,
    ) -> Self {
        Self {
            description: description.into(),
            before_expr: before_expr.into(),
            after_expr: after_expr.into(),
            before_latex: None,
            after_latex: None,
        }
    }

    /// Set the LaTeX for `before_expr`.
    pub fn with_before_latex(mut self, latex: impl Into<String>) -> Self {
        self.before_latex = Some(latex.into());
        self
    }

    /// Set the LaTeX for `after_expr`.
    pub fn with_after_latex(mut self, latex: impl Into<String>) -> Self {
        self.after_latex = Some(latex.into());
        self
    }
}

#[cfg(test)]
mod tests;
