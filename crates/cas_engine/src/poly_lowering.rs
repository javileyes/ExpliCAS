//! Poly lowering adapter for `cas_engine`.
//!
//! Core polynomial lowering lives in `cas_math::poly_lowering`.
//! This module only maps core lowering events into engine `Step`s.

use crate::Step;
use cas_ast::{Context, ExprId};
use cas_math::poly_lowering::{
    self, PolyLowerResult as MathPolyLowerResult, PolyLowerStep, PolyLowerStepKind,
};
use cas_math::poly_lowering_ops::PolyBinaryOp;

/// Result of poly lowering pass.
pub struct PolyLowerResult {
    /// Transformed expression.
    pub expr: ExprId,
    /// Steps generated during lowering.
    pub steps: Vec<Step>,
    #[allow(dead_code)] // Constructed but not yet consumed; kept for future pipeline diagnostics
    pub combined_any: bool,
}

/// Run the poly lowering pass on an expression.
///
/// This should be called AFTER eager_eval_expand_calls and BEFORE the simplifier.
pub fn poly_lower_pass(ctx: &mut Context, expr: ExprId, collect_steps: bool) -> PolyLowerResult {
    let MathPolyLowerResult {
        expr,
        steps: math_steps,
        combined_any,
    } = poly_lowering::poly_lower_pass(ctx, expr, collect_steps);

    let steps = if collect_steps {
        math_steps
            .into_iter()
            .map(|step| to_engine_step(ctx, step))
            .collect()
    } else {
        Vec::new()
    };

    PolyLowerResult {
        expr,
        steps,
        combined_any,
    }
}

fn to_engine_step(ctx: &Context, step: PolyLowerStep) -> Step {
    Step::new(
        poly_step_message(step.kind),
        "Polynomial Combination",
        step.before,
        step.after,
        Vec::new(),
        Some(ctx),
    )
}

fn poly_step_message(kind: PolyLowerStepKind) -> &'static str {
    match kind {
        PolyLowerStepKind::Direct { op } => match op {
            PolyBinaryOp::Add => "Poly lowering: combined poly_result + poly_result",
            PolyBinaryOp::Sub => "Poly lowering: combined poly_result - poly_result",
            PolyBinaryOp::Mul => "Poly lowering: combined poly_result * poly_result",
        },
        PolyLowerStepKind::Promoted => "Poly lowering: promoted and combined expressions",
    }
}
