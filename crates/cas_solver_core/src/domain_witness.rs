//! Witness survival helpers for implicit-domain guards.

use cas_ast::{Context, ExprId};
use cas_math::expr_witness::{self, WitnessKind as MathWitnessKind};

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
pub fn witness_survives(ctx: &Context, target: ExprId, output: ExprId, kind: WitnessKind) -> bool {
    expr_witness::witness_survives(ctx, target, output, kind.into())
}

/// Check if a witness survives in full-tree context, with one node replacement.
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
