use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};

define_rule!(
    IdentityPowerRule,
    "Identity Power",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        if let Some(rewrite) =
            cas_math::power_identity_support::try_rewrite_pow_one_or_one_pow_expr(ctx, expr)
        {
            return Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc));
        }

        if let Some(pattern) =
            cas_math::power_identity_support::classify_power_identity_policy_pattern(ctx, expr)
        {
            match pattern {
                cas_math::power_identity_support::PowerIdentityPolicyPattern::PowZero {
                    base,
                    base_is_literal_zero,
                } => {
                    // x^0 -> 1 REQUIRES x ≠ 0 (because 0^0 is undefined)
                    let domain_mode = parent_ctx.domain_mode();
                    let action =
                        cas_math::power_identity_support::plan_pow_zero_policy_action_with_mode_flags(
                        matches!(domain_mode, crate::domain::DomainMode::Assume),
                        matches!(domain_mode, crate::domain::DomainMode::Strict),
                        base_is_literal_zero,
                        matches!(ctx.get(base), Expr::Number(_)),
                        crate::helpers::prove_nonzero_core(ctx, base),
                    );

                    match action {
                        cas_math::power_identity_support::PowZeroPolicyAction::RewriteToUndefined { desc } => {
                            return Some(
                                Rewrite::new(ctx.add(Expr::Constant(cas_ast::Constant::Undefined)))
                                    .desc(desc),
                            );
                        }
                        cas_math::power_identity_support::PowZeroPolicyAction::RewriteToOne {
                            desc,
                            assume_nonzero,
                        } => {
                            let mut rewrite = Rewrite::new(ctx.num(1)).desc(desc);
                            if assume_nonzero {
                                rewrite = rewrite
                                    .assume(crate::assumptions::AssumptionEvent::nonzero(
                                        ctx, base,
                                    ));
                            }
                            return Some(rewrite);
                        }
                        cas_math::power_identity_support::PowZeroPolicyAction::NoRewrite => {
                            return None;
                        }
                    }
                }
                cas_math::power_identity_support::PowerIdentityPolicyPattern::ZeroPow {
                    exp,
                    exp_is_numeric_positive,
                    exp_is_numeric_non_positive,
                } => {
                    // 0^x -> 0 REQUIRES x > 0
                    match cas_math::power_identity_support::plan_zero_pow_policy_action(
                        exp_is_numeric_positive,
                        exp_is_numeric_non_positive,
                    ) {
                        cas_math::power_identity_support::ZeroPowPolicyAction::RewriteToZero {
                            desc,
                        } => {
                            return Some(Rewrite::new(ctx.num(0)).desc(desc));
                        }
                        cas_math::power_identity_support::ZeroPowPolicyAction::NoRewrite => {
                            return None;
                        }
                        cas_math::power_identity_support::ZeroPowPolicyAction::NeedsPositiveCondition => {}
                    }

                    // Use unified oracle for Positive condition (Analytic class)
                    let decision = crate::domain_oracle::oracle_allows_with_hint(
                        ctx,
                        parent_ctx.domain_mode(),
                        parent_ctx.value_domain(),
                        &crate::domain_facts::Predicate::Positive(exp),
                        "Evaluate Power",
                    );

                    if decision.allow {
                        let mut rewrite = Rewrite::new(ctx.num(0)).desc("0^x → 0");
                        for event in decision.assumption_events(ctx, exp) {
                            rewrite = rewrite.assume(event);
                        }
                        return Some(rewrite);
                    }
                    return None;
                }
            }
        }
        None
    }
);

define_rule!(
    PowerProductRule,
    "Power of a Product",
    None,
    PhaseMask::CORE | PhaseMask::TRANSFORM | PhaseMask::RATIONALIZE,
    |ctx, expr| {
        let rewrite = cas_math::power_product_support::try_rewrite_power_product_distribution_expr(
            ctx, expr,
        )?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(PowerQuotientRule, "Power of a Quotient", |ctx, expr| {
    let rewrite = cas_math::power_product_support::try_rewrite_power_quotient_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
});

// ============================================================================
// ExpQuotientRule: e^a / e^b → e^(a-b)
// ============================================================================
define_rule!(ExpQuotientRule, "Exp Quotient", |ctx, expr| {
    let rewrite = cas_math::power_product_support::try_rewrite_exp_quotient_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
});

// ============================================================================
// MulNaryCombinePowersRule
// ============================================================================
pub struct MulNaryCombinePowersRule;

impl crate::rule::Rule for MulNaryCombinePowersRule {
    fn name(&self) -> &str {
        "N-ary Mul Combine Powers"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        let rewrite =
            cas_math::power_product_support::try_rewrite_mul_nary_combine_powers_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::MUL)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Low
    }
}

define_rule!(NegativeBasePowerRule, "Negative Base Power", |ctx, expr| {
    let rewrite = cas_math::power_eval_support::try_rewrite_negative_base_power_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
});

// Canonicalize bases in even powers: (b-a)^even → (a-b)^even when a < b
define_rule!(
    EvenPowSubSwapRule,
    "Canonicalize Even Power Base",
    None,
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    importance: crate::step::ImportanceLevel::Medium,
    |ctx, expr| {
        let rewrite = cas_math::power_eval_support::try_rewrite_even_pow_sub_swap_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten)
                .desc(rewrite.desc)
                .local(rewrite.old_base, rewrite.new_base),
        )
    }
);
