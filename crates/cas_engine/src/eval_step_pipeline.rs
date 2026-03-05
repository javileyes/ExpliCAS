//! V2.9.9: Unified Eval Step Pipeline
//!
//! This module is the **single point of truth** for converting raw simplification
//! steps into display-ready steps. All cleanup, enrichment, and optimization
//! happens here — not scattered across engine.rs, repl.rs, or timeline.rs.
//!
//! The pipeline enforces the "impossible by construction" principle: raw steps
//! cannot escape to display layers because only `to_display_steps()` produces
//! the required `DisplayEvalSteps` type.

use crate::step::{DisplayEvalSteps, Step};

/// Convert raw steps to display-ready steps.
///
/// This is the **ONLY** function that should produce `DisplayEvalSteps`.
/// All step cleanup/enrichment MUST happen here to ensure consistency
/// across all renderers (Text, HTML, JSON).
///
/// # V2.9.9 Pipeline Stages
///
/// 1. **Remove no-ops**: Steps where `before == after` (no visible change)
/// 2. **Collapse duplicates**: Consecutive steps with identical descriptions (future)
/// 3. **Normalize descriptions**: Consistent formatting (future)
/// 4. **Enrich**: Add narrator text for didactic display (future)
///
/// # Arguments
///
/// * `raw_steps` - The raw steps from simplification, in order of application
///
/// # Returns
///
/// A `DisplayEvalSteps` wrapper guaranteeing cleanup has been applied.
#[must_use = "the result of pipeline processing should be used"]
pub fn to_display_steps(raw_steps: Vec<Step>) -> DisplayEvalSteps {
    cas_solver_core::eval_step_pipeline::to_display_eval_steps(raw_steps)
}
