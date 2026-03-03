//! Data-driven trigonometric evaluation rule.
//!
//! This replaces the verbose ~360-line `EvaluateTrigRule` with a compact
//! table-lookup approach.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::trig_eval_table_support::try_rewrite_trig_eval_table_expr;

define_rule!(
    EvaluateTrigTableRule,
    "Evaluate Trigonometric Functions (Table)",
    |ctx, expr| {
        let rewrite = try_rewrite_trig_eval_table_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);
