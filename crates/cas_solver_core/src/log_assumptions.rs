//! Generic adapters for logarithmic-assumption events and blocked hints.
//!
//! Keeps log-assumption target selection in `cas_solver_core`, while runtime
//! crates provide concrete event/hint types through callbacks.

use cas_ast::{Context, ExprId};

use crate::log_domain::LogAssumption;
use crate::solve_outcome::LogBlockedHintRecord;

/// Generic mapped blocked-hint payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MappedLogBlockedHint<E> {
    pub event: E,
    pub expr_id: ExprId,
    pub rule: &'static str,
    pub suggestion: &'static str,
}

/// Map one logarithmic assumption target into a caller-defined event type.
pub fn map_log_assumption_target_with<E, FMapEvent>(
    ctx: &Context,
    assumption: LogAssumption,
    base: ExprId,
    rhs: ExprId,
    mut map_event: FMapEvent,
) -> E
where
    FMapEvent: FnMut(&Context, ExprId) -> E,
{
    let target = crate::log_domain::assumption_target_expr(assumption, base, rhs);
    map_event(ctx, target)
}

/// Map one blocked log hint into a caller-defined event type.
pub fn map_log_blocked_hint_event_with<E, FMapEvent>(
    ctx: &Context,
    hint: LogBlockedHintRecord,
    mut map_event: FMapEvent,
) -> E
where
    FMapEvent: FnMut(&Context, ExprId) -> E,
{
    map_event(ctx, hint.expr_id)
}

/// Map one blocked log hint into a caller-defined event plus transferable payload.
pub fn map_log_blocked_hint_with<E, FMapEvent>(
    ctx: &Context,
    hint: LogBlockedHintRecord,
    mut map_event: FMapEvent,
) -> MappedLogBlockedHint<E>
where
    FMapEvent: FnMut(&Context, ExprId) -> E,
{
    let event = map_event(ctx, hint.expr_id);
    MappedLogBlockedHint {
        event,
        expr_id: hint.expr_id,
        rule: hint.rule,
        suggestion: hint.suggestion,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        map_log_assumption_target_with, map_log_blocked_hint_event_with, map_log_blocked_hint_with,
    };
    use crate::log_domain::LogAssumption;
    use crate::solve_outcome::LogBlockedHintRecord;

    #[test]
    fn map_log_assumption_target_with_selects_base_or_rhs() {
        let mut ctx = cas_ast::Context::new();
        let base = ctx.var("b");
        let rhs = ctx.var("r");

        let base_target = map_log_assumption_target_with(
            &ctx,
            LogAssumption::PositiveBase,
            base,
            rhs,
            |_ctx, expr| expr,
        );
        let rhs_target = map_log_assumption_target_with(
            &ctx,
            LogAssumption::PositiveRhs,
            base,
            rhs,
            |_ctx, expr| expr,
        );

        assert_eq!(base_target, base);
        assert_eq!(rhs_target, rhs);
    }

    #[test]
    fn map_log_blocked_hint_event_with_uses_hint_expr() {
        let mut ctx = cas_ast::Context::new();
        let expr = ctx.var("x");
        let hint = LogBlockedHintRecord {
            assumption: LogAssumption::PositiveBase,
            expr_id: expr,
            rule: "rule",
            suggestion: "suggestion",
        };

        let event = map_log_blocked_hint_event_with(&ctx, hint, |_ctx, id| id);
        assert_eq!(event, expr);
    }

    #[test]
    fn map_log_blocked_hint_with_forwards_payload_fields() {
        let mut ctx = cas_ast::Context::new();
        let expr = ctx.var("x");
        let hint = LogBlockedHintRecord {
            assumption: LogAssumption::PositiveRhs,
            expr_id: expr,
            rule: "Take log of both sides",
            suggestion: "use `semantics set domain assume`",
        };

        let mapped = map_log_blocked_hint_with(&ctx, hint, |_ctx, id| id);
        assert_eq!(mapped.event, expr);
        assert_eq!(mapped.expr_id, expr);
        assert_eq!(mapped.rule, "Take log of both sides");
        assert_eq!(mapped.suggestion, "use `semantics set domain assume`");
    }
}
