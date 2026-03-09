//! GCD-based cancellation rules and didactic factorization helpers.
//!
//! This module contains the heavy cancellation rules that use structural
//! comparison, polynomial GCD, and factorization to simplify fractions.

use super::didactic_factor_support::try_plan_fraction_didactic_cancel;
use crate::define_rule;
use crate::rule::{ChainedRewrite, Rewrite};
use cas_ast::{Context, Expr, ExprId};
use cas_math::fraction_gcd_plan_support::{try_plan_fraction_gcd_rewrite, FractionGcdRoute};
use cas_math::fraction_mul_div_support::{try_rewrite_simplify_mul_div_expr, MulDivRewriteKind};
use cas_math::fraction_power_cancel_support::{
    try_rewrite_cancel_identical_fraction_expr, try_rewrite_cancel_power_fraction_expr,
    try_rewrite_cancel_same_base_powers_div_expr, CancelIdenticalFractionRewriteKind,
    CancelPowerFractionRewriteKind, CancelSameBasePowersRewriteKind,
};
use cas_math::nested_fraction_support::try_rewrite_simplify_nested_fraction_expr;

fn format_factor_by_gcd_desc(ctx: &Context, gcd_expr: cas_ast::ExprId) -> String {
    format!(
        "Factor by GCD: {}",
        cas_formatter::render_expr(ctx, gcd_expr)
    )
}

fn format_mul_div_desc(kind: MulDivRewriteKind) -> &'static str {
    match kind {
        MulDivRewriteKind::CancelLeftFractionTimesDenominator => "Cancel division: (a/b)*b -> a",
        MulDivRewriteKind::CancelRightFractionTimesDenominator => "Cancel division: a*(b/a) -> b",
        MulDivRewriteKind::CombineFractionsInMultiplication => {
            "Combine fractions in multiplication"
        }
    }
}

fn format_cancel_same_base_powers_desc(kind: CancelSameBasePowersRewriteKind) -> String {
    match kind {
        CancelSameBasePowersRewriteKind::EqualPowers => "Cancel: P^n/P^n -> 1".to_string(),
        CancelSameBasePowersRewriteKind::CollapseToBase => "Cancel: P^n/P^m -> P".to_string(),
        CancelSameBasePowersRewriteKind::CollapseToReciprocalBase => {
            "Cancel: P^n/P^m -> 1/P".to_string()
        }
        CancelSameBasePowersRewriteKind::CollapseToPositivePower(diff) => {
            format!("Cancel: P^n/P^m -> P^{diff}")
        }
        CancelSameBasePowersRewriteKind::CollapseToReciprocalPower(diff) => {
            format!("Cancel: P^n/P^m -> 1/P^{diff}")
        }
    }
}

fn format_cancel_identical_fraction_desc(
    _kind: CancelIdenticalFractionRewriteKind,
) -> &'static str {
    "Cancel: P/P -> 1"
}

fn format_cancel_power_fraction_desc(kind: CancelPowerFractionRewriteKind) -> &'static str {
    match kind {
        CancelPowerFractionRewriteKind::SameSign => "Cancel: P^n/P -> P^(n-1)",
        CancelPowerFractionRewriteKind::NegatedDenominator => "Cancel: P^n/(-P) -> -P^(n-1)",
    }
}

fn fast_variable_nonzero_decision(
    ctx: &Context,
    mode: crate::DomainMode,
    expr: ExprId,
) -> Option<crate::CancelDecision> {
    if !mode.allows_unproven(crate::ConditionClass::Definability) {
        return None;
    }

    if !matches!(ctx.get(expr), Expr::Variable(_)) {
        return None;
    }

    Some(crate::CancelDecision::allow_with_keys(
        "cancelled factor assumed nonzero",
        smallvec::smallvec![crate::AssumptionKey::nonzero_key(ctx, expr)],
    ))
}

// ========== Micro-API for safe Mul construction ==========
// Use this instead of ctx.add(Expr::Mul(...)) in this file.

// =============================================================================
// STEP 1.5: Cancel same-base power fractions P^m/P^n → P^(m-n) (shallow, PRE-ORDER)
// =============================================================================

// V2.14.35: Ultra-light rule for Pow(base,m)/Pow(base,n) → base^(m-n)
// Uses shallow ExprId comparison to avoid recursion/stack depth issues.
// This handles cases like ((x+y)^10)/((x+y)^9) that would otherwise overflow stack.
define_rule!(
    CancelPowersDivisionRule,
    "Cancel Same-Base Powers",
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::Predicate;
        use crate::ImplicitCondition;

        let plan = try_rewrite_cancel_same_base_powers_div_expr(ctx, expr)?;

        // DOMAIN GATE: need base ≠ 0 (derived from original denominator P^n ≠ 0)
        let domain_mode = parent_ctx.domain_mode();
        let decision = fast_variable_nonzero_decision(ctx, domain_mode, plan.nonzero_target)
            .unwrap_or_else(|| {
                crate::oracle_allows_with_hint(
                    ctx,
                    domain_mode,
                    parent_ctx.value_domain(),
                    &Predicate::NonZero(plan.nonzero_target),
                    "Cancel Same-Base Powers",
                )
            });

        if !decision.allow {
            return None;
        }

        Some(
            Rewrite::new(plan.rewritten)
                .desc(format_cancel_same_base_powers_desc(plan.kind))
                .local(expr, plan.rewritten)
                .requires(ImplicitCondition::NonZero(plan.nonzero_target))
                .assume_all(decision.assumption_events(ctx, plan.nonzero_target)),
        )
    }
);

// =============================================================================
// STEP 2: Cancel identical numerator/denominator (P/P → 1)
// =============================================================================

// Cancels P/P → 1 when numerator equals denominator structurally.
// This is Step 2 after didactic expansion rules (e.g., (a+b)² → a² + 2ab + b²).
define_rule!(
    CancelIdenticalFractionRule,
    "Cancel Identical Numerator/Denominator",
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::Predicate;
        use crate::ImplicitCondition;

        let plan = try_rewrite_cancel_identical_fraction_expr(ctx, expr)?;

        // DOMAIN GATE: In Strict mode, only cancel if den is provably non-zero
        let domain_mode = parent_ctx.domain_mode();
        let decision = fast_variable_nonzero_decision(ctx, domain_mode, plan.nonzero_target)
            .unwrap_or_else(|| {
                crate::oracle_allows_with_hint(
                    ctx,
                    domain_mode,
                    parent_ctx.value_domain(),
                    &Predicate::NonZero(plan.nonzero_target),
                    "Cancel Identical Numerator/Denominator",
                )
            });

        if !decision.allow {
            // Strict mode + Unknown proof: don't simplify (e.g., x/x stays)
            return None;
        }

        Some(
            Rewrite::new(plan.rewritten)
                .desc(format_cancel_identical_fraction_desc(plan.kind))
                .local(expr, plan.rewritten)
                .requires(ImplicitCondition::NonZero(plan.nonzero_target))
                .assume_all(decision.assumption_events(ctx, plan.nonzero_target)),
        )
    }
);

// Rule to cancel P^n / P → P^(n-1) (didactic step 2 for perfect squares and similar)
// Handles patterns like (x-y)²/(x-y) → x-y
define_rule!(
    CancelPowerFractionRule,
    "Cancel Power Fraction",
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::Predicate;
        use crate::ImplicitCondition;

        let plan = try_rewrite_cancel_power_fraction_expr(ctx, expr)?;

        // DOMAIN GATE
        let domain_mode = parent_ctx.domain_mode();
        let decision = fast_variable_nonzero_decision(ctx, domain_mode, plan.nonzero_target)
            .unwrap_or_else(|| {
                crate::oracle_allows_with_hint(
                    ctx,
                    domain_mode,
                    parent_ctx.value_domain(),
                    &Predicate::NonZero(plan.nonzero_target),
                    "Cancel Power Fraction",
                )
            });

        if !decision.allow {
            return None;
        }

        Some(
            Rewrite::new(plan.rewritten)
                .desc(format_cancel_power_fraction_desc(plan.kind))
                .local(expr, plan.rewritten)
                .requires(ImplicitCondition::NonZero(plan.nonzero_target))
                .assume_all(decision.assumption_events(ctx, plan.nonzero_target)),
        )
    }
);

define_rule!(
    SimplifyFractionRule,
    "Simplify Nested Fraction",
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::Predicate;
        use cas_ast::views::RationalFnView;

        // Capture domain mode for cancellation decisions
        let domain_mode = parent_ctx.domain_mode();

        // Use RationalFnView to detect any fraction form while preserving structure
        let view = RationalFnView::from(ctx, expr)?;
        let (num, den) = (view.num, view.den);
        let trace_payloads_enabled = crate::rule::trace_payloads_enabled();

        // EARLY RETURN: didactic factorization/cancellation plans (ordered in cas_math).
        if let Some(plan) = try_plan_fraction_didactic_cancel(ctx, num, den) {
            use crate::ImplicitCondition;
            let rewrite = Rewrite::new(plan.rewritten)
                .desc(plan.desc)
                .requires(ImplicitCondition::NonZero(den));
            return Some(if trace_payloads_enabled {
                rewrite.local(num, plan.local_after)
            } else {
                rewrite
            });
        }

        // NOTE: PR-2 shallow GCD integration deferred.
        // The gcd_shallow_for_fraction function exists in poly_gcd.rs but calling it
        // here adds stack depth that causes overflow on complex expressions.
        // Future work: investigate stack-safe approach for power cancellation.

        // GCD rewrite planning (multivariate or univariate structural path)
        let plan = try_plan_fraction_gcd_rewrite(ctx, expr, num, den, trace_payloads_enabled)?;

        // DOMAIN GATE: Check if we can cancel by this GCD
        // In Strict mode, only allow if GCD is provably non-zero
        let decision = match plan.route {
            // Structural scalar-multiple plans prove denominator = c*gcd with c != 0,
            // so the existing implicit condition den != 0 already implies gcd != 0.
            FractionGcdRoute::StructuralScalarMultiple => crate::CancelDecision::allow(),
            _ => crate::oracle_allows_with_hint(
                ctx,
                domain_mode,
                parent_ctx.value_domain(),
                &Predicate::NonZero(plan.gcd_expr),
                "Simplify Nested Fraction",
            ),
        };
        if !decision.allow {
            // STRICT PARTIAL CANCEL: Try to cancel only numeric content
            // The numeric_gcd is always provably nonzero (it's a rational ≠ 0)
            if let (Some(result), Some(numeric_gcd)) =
                (plan.strict_partial_result, plan.strict_partial_numeric_gcd)
            {
                return Some(
                    Rewrite::new(result)
                        .desc_lazy(|| {
                            format!("Reduced numeric content by gcd {} (strict-safe)", numeric_gcd)
                        })
                        .local(expr, result),
                );
            }
            // No numeric content to cancel, don't simplify
            return None;
        }

        if plan.forms.numerator_is_zero {
            let zero = ctx.num(0);
            use crate::ImplicitCondition;
            return Some(
                Rewrite::new(zero)
                    .desc("Numerator simplifies to 0")
                    .local(num, zero)
                    .requires(ImplicitCondition::NonZero(den)),
            );
        }

        // === ChainedRewrite Pattern: Factor -> Cancel ===
        // Step 1 (main): Factor - show the factored form
        // Use requires (not assume) to avoid duplicate Requires/Assumed display
        use crate::ImplicitCondition;
        if let Some(factored_form_norm) = plan.forms.factored_form_norm {
            let factor_desc = format_factor_by_gcd_desc(ctx, plan.gcd_expr);
            let factor_rw = Rewrite::new(factored_form_norm)
                .desc(factor_desc)
                .local(expr, factored_form_norm)
                .requires(ImplicitCondition::NonZero(den));

            // Step 2 (chained): Cancel - reduce to final result
            let cancel = ChainedRewrite::new(plan.forms.result_norm)
                .desc("Cancel common factor")
                .local(factored_form_norm, plan.forms.result_norm);

            return Some(factor_rw.chain(cancel));
        }

        return Some(
            Rewrite::new(plan.forms.result_norm)
                .desc("Cancel common factor")
                .local(expr, plan.forms.result_norm)
                .requires(ImplicitCondition::NonZero(den)),
        );
    }
);

define_rule!(
    NestedFractionRule,
    "Simplify Complex Fraction",
    |ctx, expr| {
        let rewrite = try_rewrite_simplify_nested_fraction_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten)
                .desc("Simplify nested fraction")
                .local(expr, rewrite.rewritten),
        )
    }
);

define_rule!(
    SimplifyMulDivRule,
    "Simplify Multiplication with Division",
    |ctx, expr| {
        let rewrite = try_rewrite_simplify_mul_div_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_mul_div_desc(rewrite.kind)))
    }
);
