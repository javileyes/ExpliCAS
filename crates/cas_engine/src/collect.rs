use crate::collect_rule_support::CollectRulePlan;
use crate::parent_context::ParentContext;
use cas_ast::{Context, ExprId};
use cas_math::collect_semantics_support::{
    collect_semantics_mode_from_flags, collect_semantics_needs_undefined_risk_scan,
    collect_with_semantics_mode, CollectSemanticsMode, CollectSemanticsResult,
};

/// Result of a semantics-aware collection operation.
/// Contains the new expression and tracking of what was cancelled/combined.
pub(crate) type CollectResult = CollectSemanticsResult;

/// Check if an expression contains any Div with a denominator that is not proven non-zero.
/// This indicates "undefined risk" - the expression could be undefined at some points.
pub(crate) fn has_undefined_risk(ctx: &Context, expr: ExprId) -> bool {
    use crate::helpers::prove_nonzero;
    use crate::Proof;

    cas_math::undefined_risk_support::has_undefined_risk_with(ctx, expr, |core_ctx, den| {
        prove_nonzero(core_ctx, den) == Proof::Proven
    })
}

/// Collects like terms with domain_mode awareness.
/// In Strict mode, refuses to cancel terms that might be undefined.
/// In Assume mode, cancels with a warning.
/// In Generic mode, cancels unconditionally.
///
/// Returns None if the result would be identical to input or if blocked by Strict mode.
pub(crate) fn collect_with_semantics(
    ctx: &mut Context,
    expr: ExprId,
    parent_ctx: &ParentContext,
) -> Option<CollectResult> {
    let (mode, risk) = resolve_mode_and_risk(ctx, expr, parent_ctx);

    collect_with_semantics_mode(ctx, expr, mode, risk)
}

/// Build a didactic rewrite plan for CombineLikeTerms rule.
pub(crate) fn plan_collect_rule_rewrite(
    ctx: &mut Context,
    expr: ExprId,
    parent_ctx: &ParentContext,
) -> Option<CollectRulePlan> {
    let (mode, risk) = resolve_mode_and_risk(ctx, expr, parent_ctx);
    crate::collect_rule_support::try_plan_collect_rule_expr(ctx, expr, mode, risk)
}

/// Collects like terms in an expression using Generic mode semantics.
/// e.g. 2*x + 3*x -> 5*x
///      x + x -> 2*x
///      x^2 + 2*x^2 -> 3*x^2
pub(crate) fn collect(ctx: &mut Context, expr: ExprId) -> ExprId {
    // Generic mode keeps legacy behavior (no blocking, no warnings).
    match collect_with_semantics_mode(ctx, expr, CollectSemanticsMode::Generic, false) {
        Some(result) => result.new_expr,
        None => expr, // No change or blocked
    }
}

fn resolve_mode_and_risk(
    ctx: &Context,
    expr: ExprId,
    parent_ctx: &ParentContext,
) -> (CollectSemanticsMode, bool) {
    let domain_mode = parent_ctx.domain_mode();
    let mode = collect_semantics_mode_from_flags(
        matches!(domain_mode, crate::DomainMode::Assume),
        matches!(domain_mode, crate::DomainMode::Strict),
    );
    let risk = if collect_semantics_needs_undefined_risk_scan(mode) {
        has_undefined_risk(ctx, expr)
    } else {
        false
    };
    (mode, risk)
}

/// Simplify numeric sums in exponents throughout an expression tree.
/// e.g., x^(1/2 + 1/3) → x^(5/6)
/// This is applied during the collect phase for early, visible simplification.
#[allow(dead_code)] // Infrastructure: tested, reserved for collect pipeline
pub fn simplify_numeric_exponents(ctx: &mut Context, expr: ExprId) -> ExprId {
    cas_math::collect_terms::simplify_numeric_exponents(ctx, expr)
}
