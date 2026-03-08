use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use num_traits::One;

fn format_power_eval_static_desc(
    kind: cas_math::power_eval_support::PowerEvalStaticRewriteKind,
) -> &'static str {
    match kind {
        cas_math::power_eval_support::PowerEvalStaticRewriteKind::NegativeExponentNormalization => {
            "x^(-n) -> 1/x^n"
        }
        cas_math::power_eval_support::PowerEvalStaticRewriteKind::NegativeBaseEven => {
            "(-x)^even -> x^even"
        }
        cas_math::power_eval_support::PowerEvalStaticRewriteKind::NegativeBaseOdd => {
            "(-x)^odd -> -(x^odd)"
        }
    }
}

fn format_power_product_desc(
    kind: cas_math::power_product_support::PowerProductRewriteKind,
) -> &'static str {
    match kind {
        cas_math::power_product_support::PowerProductRewriteKind::DistributePowerOverProduct => {
            "Distribute power over product"
        }
        cas_math::power_product_support::PowerProductRewriteKind::DistributePowerOverQuotient => {
            "Distribute power over quotient"
        }
        cas_math::power_product_support::PowerProductRewriteKind::ExpQuotient => {
            "e^a / e^b = e^(a-b)"
        }
        cas_math::power_product_support::PowerProductRewriteKind::ExpOverExpPower => {
            "e / e^b = e^(1-b)"
        }
        cas_math::power_product_support::PowerProductRewriteKind::ExpPowerOverExp => {
            "e^a / e = e^(a-1)"
        }
        cas_math::power_product_support::PowerProductRewriteKind::AllFactorsCancelled => {
            "All factors cancelled"
        }
        cas_math::power_product_support::PowerProductRewriteKind::SameBaseNary => {
            "Combine powers with same base (n-ary)"
        }
        _ => "Power product rewrite",
    }
}

fn format_pow_zero_desc(
    mode: cas_math::power_identity_support::PowerIdentityDomainMode,
) -> &'static str {
    match mode {
        cas_math::power_identity_support::PowerIdentityDomainMode::Strict => {
            "x^0 -> 1 (x ≠ 0 proven)"
        }
        cas_math::power_identity_support::PowerIdentityDomainMode::Generic => "x^0 -> 1",
        cas_math::power_identity_support::PowerIdentityDomainMode::Assume => {
            "x^0 -> 1 (assuming x ≠ 0)"
        }
    }
}

fn format_pow_one_desc(ctx: &Context, expr: ExprId) -> &'static str {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            if matches!(ctx.get(*exp), Expr::Number(n) if n.is_one()) {
                "x^1 -> x"
            } else if matches!(ctx.get(*base), Expr::Number(n) if n.is_one()) {
                "1^x -> 1"
            } else {
                "Power identity"
            }
        }
        _ => "Power identity",
    }
}

define_rule!(
    IdentityPowerRule,
    "Identity Power",
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        if let Some(rewritten) =
            cas_math::power_identity_support::try_rewrite_pow_one_or_one_pow_expr(ctx, expr)
        {
            return Some(Rewrite::new(rewritten).desc(format_pow_one_desc(ctx, expr)));
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
                    let power_mode =
                        cas_math::power_identity_support::power_identity_mode_from_flags(
                            matches!(domain_mode, crate::DomainMode::Assume),
                            matches!(domain_mode, crate::DomainMode::Strict),
                        );
                    let action =
                        cas_math::power_identity_support::plan_pow_zero_policy_action_with_mode_flags(
                        matches!(domain_mode, crate::DomainMode::Assume),
                        matches!(domain_mode, crate::DomainMode::Strict),
                        base_is_literal_zero,
                        matches!(ctx.get(base), Expr::Number(_)),
                        cas_solver_core::predicate_proofs::prove_nonzero_core_with(
                            ctx,
                            base,
                            crate::helpers::prove_nonzero,
                        ),
                    );

                    match action {
                        cas_math::power_identity_support::PowZeroPolicyAction::RewriteToUndefined => {
                            return Some(
                                Rewrite::new(ctx.add(Expr::Constant(cas_ast::Constant::Undefined)))
                                    .desc("0^0 -> undefined"),
                            );
                        }
                        cas_math::power_identity_support::PowZeroPolicyAction::RewriteToOne {
                            assume_nonzero,
                        } => {
                            let mut rewrite =
                                Rewrite::new(ctx.num(1)).desc(format_pow_zero_desc(power_mode));
                            if assume_nonzero {
                                rewrite = rewrite
                                    .assume(crate::AssumptionEvent::nonzero(
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
                        cas_math::power_identity_support::ZeroPowPolicyAction::RewriteToZero => {
                            return Some(Rewrite::new(ctx.num(0)).desc("0^n -> 0 (n > 0)"));
                        }
                        cas_math::power_identity_support::ZeroPowPolicyAction::NoRewrite => {
                            return None;
                        }
                        cas_math::power_identity_support::ZeroPowPolicyAction::NeedsPositiveCondition => {}
                    }

                    // Use unified oracle for Positive condition (Analytic class)
                    let decision = crate::oracle_allows_with_hint(
                        ctx,
                        parent_ctx.domain_mode(),
                        parent_ctx.value_domain(),
                        &crate::Predicate::Positive(exp),
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
        Some(Rewrite::new(rewrite.rewritten).desc(format_power_product_desc(rewrite.kind)))
    }
);

define_rule!(PowerQuotientRule, "Power of a Quotient", |ctx, expr| {
    let rewrite = cas_math::power_product_support::try_rewrite_power_quotient_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(format_power_product_desc(rewrite.kind)))
});

// ============================================================================
// ExpQuotientRule: e^a / e^b → e^(a-b)
// ============================================================================
define_rule!(ExpQuotientRule, "Exp Quotient", |ctx, expr| {
    let rewrite = cas_math::power_product_support::try_rewrite_exp_quotient_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(format_power_product_desc(rewrite.kind)))
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
        Some(Rewrite::new(rewrite.rewritten).desc(format_power_product_desc(rewrite.kind)))
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
    Some(Rewrite::new(rewrite.rewritten).desc(format_power_eval_static_desc(rewrite.kind)))
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
                .desc("For even exponent: (a-b)² = (b-a)², normalize for cancellation")
                .local(rewrite.old_base, rewrite.new_base),
        )
    }
);
