use crate::define_rule;
use crate::ordering::compare_expr;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_extract::extract_log_base_argument_relaxed_view;
use cas_math::expr_predicates::is_e_constant_expr;
use cas_math::logarithm_inverse_support::{
    estimate_log_terms, log_auto_expand_emits_blocked_hint, log_auto_expand_mode_from_flags,
    log_auto_expand_needs_implicit_domain, log_exp_inverse_policy_mode_from_flags,
    plan_exponential_log_inverse_policy, plan_log_auto_expand_positivity,
    plan_log_exp_inverse_symbolic_policy, plan_log_power_base_numeric_policy,
    try_match_log_exp_inverse_expr, try_rewrite_exponential_log_inverse_expr,
    try_rewrite_log_inverse_power_expr, try_rewrite_log_power_base_numeric_expr,
    try_rewrite_split_log_exponents_expr,
};
use std::cmp::Ordering;

/// Domain-aware rule for b^log(b, x) → x.
/// Requires x > 0 (domain of log). Respects domain_mode.
pub struct ExponentialLogRule;

impl crate::rule::Rule for ExponentialLogRule {
    fn name(&self) -> &str {
        "Exponential-Log Inverse"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::helpers::prove_positive;
        use crate::ImplicitCondition;
        use crate::Proof;

        let planned = try_rewrite_exponential_log_inverse_expr(ctx, expr)?;
        let vd = parent_ctx.value_domain();
        let positive = prove_positive(ctx, planned.positive_subject, vd);
        let domain_mode = parent_ctx.domain_mode();
        let mode = log_exp_inverse_policy_mode_from_flags(
            matches!(domain_mode, crate::DomainMode::Assume),
            matches!(domain_mode, crate::DomainMode::Strict),
        );
        let policy = plan_exponential_log_inverse_policy(mode, positive == Proof::Proven);

        match policy {
            cas_math::logarithm_inverse_support::ExponentialLogInversePolicyPlan::Block => None,
            cas_math::logarithm_inverse_support::ExponentialLogInversePolicyPlan::Rewrite {
                require_positive_subject,
            } => {
                let mut rewrite = crate::rule::Rewrite::new(planned.rewritten).desc(planned.desc);
                if require_positive_subject {
                    rewrite =
                        rewrite.requires(ImplicitCondition::Positive(planned.positive_subject));
                }
                Some(rewrite)
            }
        }
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::POW)
    }

    fn solve_safety(&self) -> crate::SolveSafety {
        // Intrinsic: the condition x > 0 is already guaranteed by ln(x)/log(b,x)
        // being present in the input expression. This is inherited, not introduced.
        crate::SolveSafety::IntrinsicCondition(crate::ConditionClass::Analytic)
    }
}

define_rule!(
    SplitLogExponentsRule,
    "Split Log Exponents",
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Analytic
    ),
    |ctx, expr, _parent_ctx| {
    let rewrite = try_rewrite_split_log_exponents_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
});

define_rule!(
    LogInversePowerRule,
    "Log Inverse Power",
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Analytic
    ),
    |ctx, expr, _parent_ctx| {
    let rewrite = try_rewrite_log_inverse_power_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
});

/// Domain-aware rule for log(b, b^x) → x.
/// Variable exponents only simplify when domain_mode is NOT strict.
/// Numeric exponents (like log(x, x^2) → 2) always apply.
/// This is controlled by domain_mode because it's a domain assumption (x is real),
/// not an inverse trig composition.
pub struct LogExpInverseRule;

impl crate::rule::Rule for LogExpInverseRule {
    fn name(&self) -> &str {
        "Log-Exp Inverse"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::helpers::prove_positive;
        use crate::semantics::ValueDomain;
        use crate::Proof;

        let matched = try_match_log_exp_inverse_expr(ctx, expr)?;
        match matched {
            cas_math::logarithm_inverse_support::LogExpInverseMatch::Numeric {
                rewritten,
                desc,
            } => Some(crate::rule::Rewrite::new(rewritten).desc(desc)),
            cas_math::logarithm_inverse_support::LogExpInverseMatch::Symbolic {
                base,
                exponent,
            } => {
                let vd = parent_ctx.value_domain();
                let domain_mode = parent_ctx.domain_mode();
                let mode = log_exp_inverse_policy_mode_from_flags(
                    matches!(domain_mode, crate::DomainMode::Assume),
                    matches!(domain_mode, crate::DomainMode::Strict),
                );
                let plan = plan_log_exp_inverse_symbolic_policy(
                    mode,
                    vd == ValueDomain::ComplexEnabled,
                    is_e_constant_expr(ctx, base),
                    prove_positive(ctx, base, vd) == Proof::Proven,
                    matches!(ctx.get(base), Expr::Number(n) if *n == num_rational::BigRational::from_integer(1.into())),
                );

                match plan {
                    cas_math::logarithm_inverse_support::LogExpInversePolicyPlan::Block => None,
                    cas_math::logarithm_inverse_support::LogExpInversePolicyPlan::Rewrite {
                        assume_positive_base,
                    } => {
                        let one = ctx.num(1);
                        let base_minus_1 = ctx.add(Expr::Sub(base, one));
                        let mut rewrite = crate::rule::Rewrite::new(exponent)
                            .desc("log(b, b^x) → x")
                            .requires(crate::ImplicitCondition::NonZero(base_minus_1));
                        if assume_positive_base {
                            rewrite =
                                rewrite.assume(crate::AssumptionEvent::positive_assumed(ctx, base));
                        }
                        Some(rewrite)
                    }
                }
            }
        }
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }
}

/// Rule for log(a^m, a^n) → n/m
///
/// Handles cases like:
/// - log(x^2, x^6) → 6/2 = 3
/// - log(1/x, x) → log(x^(-1), x^1) → 1/(-1) = -1
///
/// Normalizes bases and arguments to power form:
/// - a → (a, 1)
/// - a^m → (a, m)
/// - 1/a → (a, -1)
pub struct LogPowerBaseRule;

impl crate::rule::Rule for LogPowerBaseRule {
    fn name(&self) -> &str {
        "Log Power Base"
    }

    fn soundness(&self) -> crate::rule::SoundnessLabel {
        crate::rule::SoundnessLabel::EquivalenceUnderIntroducedRequires
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::helpers::prove_positive;
        use crate::semantics::ValueDomain;
        use crate::ImplicitCondition;
        use crate::Proof;

        let planned = try_rewrite_log_power_base_numeric_expr(ctx, expr)?;

        let vd = parent_ctx.value_domain();
        let domain_mode = parent_ctx.domain_mode();
        let mode = log_exp_inverse_policy_mode_from_flags(
            matches!(domain_mode, crate::DomainMode::Assume),
            matches!(domain_mode, crate::DomainMode::Strict),
        );
        let one = ctx.num(1);
        let policy = plan_log_power_base_numeric_policy(
            mode,
            vd == ValueDomain::ComplexEnabled,
            prove_positive(ctx, planned.base_core, vd) == Proof::Proven,
            compare_expr(ctx, planned.base_core, one) == Ordering::Equal,
        );

        match policy {
            cas_math::logarithm_inverse_support::LogPowerBasePolicyPlan::Block => None,
            cas_math::logarithm_inverse_support::LogPowerBasePolicyPlan::Rewrite {
                require_positive_base,
                require_nonzero_base_minus_one,
            } => {
                let mut rewrite = crate::rule::Rewrite::new(planned.rewritten).desc(planned.desc);
                if require_positive_base {
                    rewrite = rewrite.requires(ImplicitCondition::Positive(planned.base_core));
                }
                if require_nonzero_base_minus_one {
                    let base_minus_1 = ctx.add(Expr::Sub(planned.base_expr, one));
                    rewrite = rewrite.requires(ImplicitCondition::NonZero(base_minus_1));
                }
                Some(rewrite)
            }
        }
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Medium
    }

    fn priority(&self) -> i32 {
        // Higher than LogEvenPowerWithChainedAbsRule (10) to match log(x^2, x^6) first
        // Otherwise LogEvenPower would expand to 6·log(x², |x|) before we can simplify to 3
        15
    }
}

// ============================================================================
// Auto Expand Log Rule with ExpandBudget Integration
// ============================================================================

// NOTE: Local is_provably_positive was removed in V2.15.9.
// Use crate::helpers::prove_positive instead, which handles:
// - base > 0 → base^(p/q) > 0 (RealOnly)
// - sqrt(x) > 0 when x > 0
// - exp(x) > 0 in RealOnly
// - etc.

/// AutoExpandLogRule: Automatically expand log(a*b) -> log(a) + log(b) during simplify
/// when log_expand_policy = Auto and the expansion passes budget checks.
///
/// This rule uses domain gating:
/// - Assume mode: expands with HeuristicAssumption (⚠️) for a>0, b>0
/// - Generic mode: blocks and registers hint if positivity not proven
/// - Strict mode: blocks without hint
pub struct AutoExpandLogRule;

impl crate::rule::Rule for AutoExpandLogRule {
    fn name(&self) -> &'static str {
        "AutoExpandLogRule"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // GATE: Expand if global auto-expand mode OR inside a marked cancellation context
        // This mirrors AutoExpandPowSumRule behavior exactly
        let in_expand_context = parent_ctx.in_auto_expand_context();
        if !(parent_ctx.is_auto_expand() || in_expand_context) {
            return None;
        }

        // Match log(arg), log(base, arg), or ln(arg)
        let (_, arg) = extract_log_base_argument_relaxed_view(ctx, expr)?;

        // Check if expandable and get term estimates
        let (base_terms, gen_terms, pow_exp) = estimate_log_terms(ctx, arg)?;

        // Get budget - use default if in context but no explicit budget set
        let default_budget = crate::phase::ExpandBudget::default();
        let budget = parent_ctx.auto_expand_budget().unwrap_or(&default_budget);

        // Budget check
        if !budget.allows_log_expansion(base_terms, gen_terms, pow_exp) {
            return None;
        }

        // Don't expand if it wouldn't help (gen_terms <= 1)
        if gen_terms <= 1 {
            return None;
        }

        // Get domain mode from parent context
        let domain_mode = parent_ctx.domain_mode();
        let mode = log_auto_expand_mode_from_flags(
            matches!(domain_mode, crate::DomainMode::Assume),
            matches!(domain_mode, crate::DomainMode::Strict),
        );

        // V2.15: Use cached implicit_domain if available, fallback to computation
        let vd = parent_ctx.value_domain();
        let implicit_domain: Option<crate::ImplicitDomain> =
            if log_auto_expand_needs_implicit_domain(mode) {
                parent_ctx.implicit_domain().cloned().or_else(|| {
                    parent_ctx
                        .root_expr()
                        .map(|root| crate::infer_implicit_domain(ctx, root, vd))
                })
            } else {
                None
            };

        let plan = plan_log_auto_expand_positivity(
            ctx,
            arg,
            mode,
            |factor| crate::helpers::prove_positive(ctx, factor, vd).is_proven(),
            |factor| {
                let cond = crate::ImplicitCondition::Positive(factor);
                implicit_domain.as_ref().is_some_and(|id| {
                    let dc = crate::DomainContext::new(id.conditions().iter().cloned().collect());
                    dc.is_condition_implied(ctx, &cond)
                })
            },
        );

        match plan {
            cas_math::logarithm_inverse_support::LogAutoExpandPositivityPlan::AllowNoAssumptions => {
                expand_log_for_rule(ctx, expr, arg, &[])
            }
            cas_math::logarithm_inverse_support::LogAutoExpandPositivityPlan::AllowWithAssumptions(factors) => {
                let events: Vec<crate::AssumptionEvent> = factors
                    .into_iter()
                    .map(|factor| crate::AssumptionEvent::positive_assumed(ctx, factor))
                    .collect();
                expand_log_for_rule(ctx, expr, arg, &events)
            }
            cas_math::logarithm_inverse_support::LogAutoExpandPositivityPlan::Blocked { factor } => {
                if log_auto_expand_emits_blocked_hint(mode) {
                    let hint = crate::BlockedHint {
                        key: crate::AssumptionKey::Positive {
                            expr_fingerprint: crate::expr_fingerprint(ctx, factor),
                        },
                        expr_id: factor,
                        rule: "AutoExpandLogRule".to_string(),
                        suggestion: "Use 'semantics set domain assume' to enable log expansion.",
                    };
                    crate::register_blocked_hint(hint);
                }
                None
            }
        }
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        // Same as AutoExpandPowSumRule: CORE, TRANSFORM, RATIONALIZE
        crate::phase::PhaseMask::CORE
            | crate::phase::PhaseMask::TRANSFORM
            | crate::phase::PhaseMask::RATIONALIZE
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        // Didactically important: users should see log expansions
        crate::step::ImportanceLevel::Medium
    }
}

/// Perform the log expansion for AutoExpandLogRule.
fn expand_log_for_rule(
    ctx: &mut Context,
    original: ExprId,
    arg: ExprId,
    events: &[crate::AssumptionEvent],
) -> Option<Rewrite> {
    let planned =
        cas_math::logarithm_inverse_support::try_expand_log_auto_rule_expr(ctx, original, arg)?;
    let mut rewrite = Rewrite::new(planned.rewritten).desc(planned.desc);
    for event in events {
        rewrite = rewrite.assume(event.clone());
    }
    Some(rewrite)
}
