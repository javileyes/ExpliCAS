//! Data-driven trigonometric evaluation rule.
//!
//! This replaces the verbose ~360-line `EvaluateTrigRule` with a compact
//! table-lookup approach.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::trig_eval_table_support::{
    try_rewrite_trig_eval_table_expr, TrigEvalRewriteKind, TrigNegativeParityKind,
};

fn format_trig_negative_parity_desc(kind: TrigNegativeParityKind) -> &'static str {
    match kind {
        TrigNegativeParityKind::Sin => "sin(-x) = -sin(x)",
        TrigNegativeParityKind::Cos => "cos(-x) = cos(x)",
        TrigNegativeParityKind::Tan => "tan(-x) = -tan(x)",
    }
}

define_rule!(
    EvaluateTrigTableRule,
    "Evaluate Trigonometric Functions (Table)",
    |ctx, expr| {
        let rewrite = try_rewrite_trig_eval_table_expr(ctx, expr)?;
        let desc = match rewrite.kind {
            TrigEvalRewriteKind::Table(desc) => desc,
            TrigEvalRewriteKind::NegativeParity(kind) => {
                format_trig_negative_parity_desc(kind).to_string()
            }
        };
        Some(Rewrite::new(rewrite.rewritten).desc(desc))
    }
);
