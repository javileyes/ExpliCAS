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
#[path = "types/enriched_step.rs"]
mod enriched_step;
mod enrichment_pipeline;
mod focused_rule_substeps;
mod fraction_steps;
mod fraction_sum_analysis;
mod gcd_factorization;
mod generic_rule_substeps;
mod latex_plain_text;
pub(crate) mod locale;
mod nested_fraction_analysis;
mod nested_fractions;
mod polynomial_identity;
mod rationalization;
mod root_denesting;
mod shared_numeric;
mod step_visibility;
#[path = "types/substep.rs"]
mod substep;
mod sum_three_cubes;
mod visible_rule_names;

pub use cli_render::format_cli_simplification_steps_with_simplifier;
pub(crate) use display_policy::{build_timeline_substeps_render_plan, classify_sub_steps};
pub use display_policy::{
    CliSubstepsRenderPlan, StepDisplayMode, SubStepClassification, TimelineSubstepsRenderPlan,
};
pub use enriched_step::EnrichedStep;
pub use enrichment_pipeline::enrich_steps;
pub(crate) use enrichment_pipeline::get_standalone_substeps;
pub(crate) use latex_plain_text::latex_to_plain_text;
pub(crate) use shared_numeric::{
    collect_add_terms, format_fraction, lcm_bigint, try_as_fraction, IsOne,
};
pub(crate) use step_visibility::{
    clone_steps_matching_visibility, infer_original_expr_for_steps,
    should_absorb_preparatory_step_at, step_matches_visibility, StepVisibility,
};
pub use substep::SubStep;
pub(crate) use visible_rule_names::{
    rule_name_es_to_en, visible_rule_name, visible_rule_name_for_step, visible_step_description,
};

#[cfg(test)]
mod tests;
