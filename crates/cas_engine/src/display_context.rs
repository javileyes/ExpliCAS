//! Compatibility wrapper around display hint builders in `cas_formatter`.

use crate::step::Step;
use cas_ast::Context;
use cas_formatter::{DisplayContext, DisplayStepLike};

impl DisplayStepLike for Step {
    fn rule_name(&self) -> &str {
        &self.rule_name
    }

    fn before(&self) -> cas_ast::ExprId {
        self.before
    }

    fn after(&self) -> cas_ast::ExprId {
        self.after
    }

    fn global_before(&self) -> Option<cas_ast::ExprId> {
        self.global_before
    }

    fn global_after(&self) -> Option<cas_ast::ExprId> {
        self.global_after
    }
}

/// Build DisplayContext by analyzing the original expression and simplification steps.
#[allow(dead_code)] // Convenience API used by module tests; `_with_result` variant is the hot path
pub fn build_display_context(
    ctx: &Context,
    original_expr: cas_ast::ExprId,
    steps: &[Step],
) -> DisplayContext {
    cas_formatter::build_display_context(ctx, original_expr, steps)
}

/// Build DisplayContext, optionally including the final simplified result.
pub fn build_display_context_with_result(
    ctx: &Context,
    original_expr: cas_ast::ExprId,
    steps: &[Step],
    simplified_result: Option<cas_ast::ExprId>,
) -> DisplayContext {
    cas_formatter::build_display_context_with_result(ctx, original_expr, steps, simplified_result)
}
