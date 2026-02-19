//! Substitute API with optional step rendering for upper layers (CLI/JSON).
//!
//! Core substitution logic remains in `cas_math::substitute`.

use cas_ast::{Context, ExprId};
use cas_formatter::DisplayExpr;

pub use cas_math::substitute::SubstituteOptions;

/// A single substitution step for traceability.
#[derive(Clone, Debug)]
pub struct SubstituteStep {
    /// Rule name: "SubstituteExact", "SubstitutePowerMultiple", "SubstitutePowOfTarget"
    pub rule: String,
    /// Expression before substitution (formatted)
    pub before: String,
    /// Expression after substitution (formatted)
    pub after: String,
    /// Optional note (e.g., "n=4, k=2, m=2")
    pub note: Option<String>,
}

/// Result of substitution including optional steps.
#[derive(Clone, Debug)]
pub struct SubstituteResult {
    pub expr: ExprId,
    pub steps: Vec<SubstituteStep>,
}

/// Perform power-aware substitution.
pub fn substitute_power_aware(
    ctx: &mut Context,
    root: ExprId,
    target: ExprId,
    replacement: ExprId,
    opts: SubstituteOptions,
) -> ExprId {
    cas_math::substitute::substitute_power_aware(ctx, root, target, replacement, opts)
}

/// Perform power-aware substitution with step collection.
pub fn substitute_with_steps(
    ctx: &mut Context,
    root: ExprId,
    target: ExprId,
    replacement: ExprId,
    opts: SubstituteOptions,
) -> SubstituteResult {
    let trace = cas_math::substitute::substitute_with_trace(ctx, root, target, replacement, opts);
    let steps = trace
        .steps
        .into_iter()
        .map(|step| SubstituteStep {
            rule: step.rule,
            before: format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: step.before,
                }
            ),
            after: format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: step.after,
                }
            ),
            note: step.note,
        })
        .collect();

    SubstituteResult {
        expr: trace.expr,
        steps,
    }
}
