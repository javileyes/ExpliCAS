use crate::rule::Rewrite;
use cas_ast::ExprId;
use cas_math::logarithm_inverse_support::{
    expand_logs_collect_positive_assumptions, log_even_power_needs_requires_lookup,
    log_exp_inverse_policy_mode_from_flags, plan_log_abs_simplify_policy,
    plan_log_even_power_policy, try_plan_log_even_power_abs_expr, try_rewrite_log_abs_power_expr,
    try_rewrite_log_abs_simplify_expr, try_rewrite_log_chain_product_expr,
    try_rewrite_log_mul_div_expansion_expr, LogChainProductRewriteKind,
};

/// Domain-aware expansion rule for log products/quotients.
///
/// log(b, x*y) → log(b, x) + log(b, y) and log(b, x/y) → log(b, x) - log(b, y)
///
/// These expansions require:
/// - RealOnly value_domain (complex domain with principal branch: NEVER expand)
/// - Strict: only if prove_positive(x) && prove_positive(y)
/// - Assume: expand with warning if not Disproven
/// - Generic: same as Strict (conservative - no silent assumptions)
pub struct LogExpansionRule;

impl crate::rule::Rule for LogExpansionRule {
    fn name(&self) -> &str {
        "Log Expansion"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::Predicate;

        // GATE 1: Never expand in complex domain (principal branch causes 2πi jumps)
        if parent_ctx.value_domain() == crate::semantics::ValueDomain::ComplexEnabled {
            return None;
        }

        let planned = try_rewrite_log_mul_div_expansion_expr(ctx, expr)?;
        let mode = parent_ctx.domain_mode();
        let vd = parent_ctx.value_domain();

        let lhs_decision = crate::oracle_allows_with_hint(
            ctx,
            mode,
            vd,
            &Predicate::Positive(planned.positive_lhs),
            "Log Expansion",
        );
        let rhs_decision = crate::oracle_allows_with_hint(
            ctx,
            mode,
            vd,
            &Predicate::Positive(planned.positive_rhs),
            "Log Expansion",
        );

        if !lhs_decision.allow || !rhs_decision.allow {
            return None;
        }

        let mut events: smallvec::SmallVec<[crate::AssumptionEvent; 2]> = smallvec::SmallVec::new();
        if lhs_decision.assumption.is_some() {
            events.push(crate::AssumptionEvent::positive(ctx, planned.positive_lhs));
        }
        if rhs_decision.assumption.is_some() {
            events.push(crate::AssumptionEvent::positive(ctx, planned.positive_rhs));
        }

        Some(
            crate::rule::Rewrite::new(planned.rewritten)
                .desc(planned.desc)
                .assume_all(events),
        )
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }

    fn solve_safety(&self) -> crate::SolveSafety {
        crate::SolveSafety::NeedsCondition(crate::ConditionClass::Analytic)
    }
}

/// Recursively expand logarithms throughout an expression tree.
///
/// This is a specialized expansion function that applies log expansion rules:
/// - log(x*y) → log(x) + log(y)
/// - log(x/y) → log(x) - log(y)
///
/// Returns the expanded expression and any assumption events generated.
/// Used by the `expand_log()` meta-function.
pub(crate) fn expand_logs_with_assumptions(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> (cas_ast::ExprId, Vec<crate::AssumptionEvent>) {
    let plan = expand_logs_collect_positive_assumptions(ctx, expr);
    let events = plan
        .assumed_positive
        .into_iter()
        .map(|subject| crate::AssumptionEvent::positive(ctx, subject))
        .collect();
    (plan.rewritten, events)
}

/// Domain-aware:
/// - Strict: only if prove_positive(expr) == Proven
/// - Generic: allow (like x/x → 1 in Generic)
/// - Assume: allow with domain_assumption for traceability
///
/// ValueDomain-aware:
/// - ComplexEnabled: only if prove_positive == Proven (no assume for ℂ)
/// - RealOnly: use DomainMode policy
///
/// NOTE: This rule should be registered BEFORE LogContractionRule to catch
/// `ln(|x|) - ln(x)` before it becomes `ln(|x|/x)`.
///
/// V2.14.20: LogEvenPowerWithChainedAbsRule
/// Handles ln(x^even) → even·ln(|x|) with optional ChainedRewrite for |x|→x
/// when x > 0 is provable or in requires.
///
/// This produces TWO contiguous steps:
/// 1. ln(x^even) → even·ln(|x|)
/// 2. |x| → x (if x > 0 provable)
///
/// Priority: higher than EvaluateLogRule to match first.
pub struct LogEvenPowerWithChainedAbsRule;

impl crate::rule::Rule for LogEvenPowerWithChainedAbsRule {
    fn name(&self) -> &str {
        "Log Even Power" // Distinct from EvaluateLogRule for engine registration
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::helpers::prove_positive;
        use crate::rule::ChainedRewrite;
        use crate::Proof;

        let planned = try_plan_log_even_power_abs_expr(ctx, expr)?;

        // Check if we can simplify |x| → x
        let vd = parent_ctx.value_domain();
        let dm = parent_ctx.domain_mode();
        let pos = prove_positive(ctx, planned.inner_base, vd);
        let mode = log_exp_inverse_policy_mode_from_flags(
            matches!(dm, crate::DomainMode::Assume),
            matches!(dm, crate::DomainMode::Strict),
        );

        let in_requires = if log_even_power_needs_requires_lookup(
            mode,
            pos == Proof::Proven,
            pos == Proof::Disproven,
        ) {
            // V2.14.21: Check if x > 0 is in global requires using implicit_domain
            // V2.15: Use cached implicit_domain if available, fallback to computation with root_expr
            let implicit_domain: Option<crate::ImplicitDomain> =
                parent_ctx.implicit_domain().cloned().or_else(|| {
                    parent_ctx
                        .root_expr()
                        .map(|root| crate::infer_implicit_domain(ctx, root, vd))
                });

            implicit_domain.as_ref().is_some_and(|id| {
                let dc = crate::DomainContext::new(id.conditions().iter().cloned().collect());
                let cond = crate::ImplicitCondition::Positive(planned.inner_base);
                dc.is_condition_implied(ctx, &cond)
            })
        } else {
            false
        };

        let policy = plan_log_even_power_policy(
            mode,
            pos == Proof::Proven,
            pos == Proof::Disproven,
            in_requires,
        );

        match policy {
            cas_math::logarithm_inverse_support::LogEvenPowerPolicyPlan::Block => None,
            cas_math::logarithm_inverse_support::LogEvenPowerPolicyPlan::RewriteWithChain {
                chain_desc,
                assume_positive_subject,
            } => {
                let mut rw = crate::rule::Rewrite::new(planned.with_abs_rewrite)
                    .desc("log(b, x^(even)) = even·log(b, |x|)");
                let mut chain = ChainedRewrite::new(planned.without_abs_rewrite)
                    .desc(chain_desc)
                    .local(planned.abs_inner_base, planned.inner_base);
                if assume_positive_subject {
                    chain = chain.assume(crate::AssumptionEvent::positive_assumed(
                        ctx,
                        planned.inner_base,
                    ));
                }
                rw = rw.chain(chain);
                Some(rw)
            }
            cas_math::logarithm_inverse_support::LogEvenPowerPolicyPlan::RewriteWithAbsAssume {
                assume_positive_subject,
            } => {
                let mut rw = crate::rule::Rewrite::new(planned.with_abs_rewrite)
                    .desc("log(b, x^(even)) = even·log(b, |x|)");
                if assume_positive_subject {
                    rw = rw.assume(crate::AssumptionEvent::positive_assumed(
                        ctx,
                        planned.inner_base,
                    ));
                }
                Some(rw)
            }
        }
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }

    fn priority(&self) -> i32 {
        10 // Higher priority than EvaluateLogRule (default 0)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Medium
    }
}

/// LogAbsPowerRule: ln(|u|^n) → n·ln(|u|) for positive integer n
///
/// This rule handles the case where the argument of the log is already wrapped
/// in abs(), so we don't "introduce" a new abs - it's already there.
///
/// Priority: Very high (15) - must apply BEFORE:
/// - AbsSquareRule (|x|^2 → x^2) which would lose the abs
/// - LogEvenPowerWithChainedAbsRule which handles ln(x^n) without abs
///
/// Requires: u ≠ 0 (so ln(|u|) is defined)
pub struct LogAbsPowerRule;

impl crate::rule::Rule for LogAbsPowerRule {
    fn name(&self) -> &str {
        "Log Abs Power"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        let planned = try_rewrite_log_abs_power_expr(ctx, expr)?;

        // Register hint that u ≠ 0 is required (for ln(|u|) to be defined)
        let key = crate::AssumptionKey::nonzero_key(ctx, planned.inner_subject);
        let hint = crate::BlockedHint {
            key,
            expr_id: planned.inner_subject,
            rule: "Log Abs Power".to_string(),
            suggestion: "requires u ≠ 0 for ln(|u|) to be defined",
        };
        crate::register_blocked_hint(hint);

        Some(crate::rule::Rewrite::new(planned.rewritten).desc(planned.desc))
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }

    fn priority(&self) -> i32 {
        15 // Higher than LogEvenPowerWithChainedAbsRule (10) and AbsSquareRule
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Medium
    }
}

pub struct LogAbsSimplifyRule;

impl crate::rule::Rule for LogAbsSimplifyRule {
    fn name(&self) -> &str {
        "Log Abs Simplify"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::helpers::prove_positive;
        let planned = try_rewrite_log_abs_simplify_expr(ctx, expr)?;
        let vd = parent_ctx.value_domain();
        let policy = plan_log_abs_simplify_policy(
            log_exp_inverse_policy_mode_from_flags(
                matches!(parent_ctx.domain_mode(), crate::DomainMode::Assume),
                matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict),
            ),
            vd == crate::semantics::ValueDomain::ComplexEnabled,
            prove_positive(ctx, planned.inner_subject, vd).is_proven(),
        );

        match policy {
            cas_math::logarithm_inverse_support::LogAbsSimplifyPolicyPlan::Block => None,
            cas_math::logarithm_inverse_support::LogAbsSimplifyPolicyPlan::Rewrite {
                assume_positive_subject,
                desc,
            } => {
                let mut rewrite = Rewrite::new(planned.rewritten).desc(desc);
                if assume_positive_subject {
                    rewrite = rewrite.assume(crate::AssumptionEvent::positive_assumed(
                        ctx,
                        planned.inner_subject,
                    ));
                }
                Some(rewrite)
            }
        }
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }

    // V2.14.20: Run in POST phase only so ln(|a|) created by LogPowerRule exists first
    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        crate::phase::PhaseMask::POST
    }

    // Ensure step is visible - domain simplification is didactically important
    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Low
    }
}

// =============================================================================
// LOG CHAIN PRODUCT RULE (LOG TELESCOPING)
// log(base, a) * log(a, c) → log(base, c)
// =============================================================================
// This implements the "change of base" telescoping identity:
//   log_b(a) * log_a(c) = log_b(c)
//
// Using the definition log_b(x) = ln(x)/ln(b):
//   (ln(a)/ln(b)) * (ln(c)/ln(a)) = ln(c)/ln(b) = log_b(c)
//
// The rule scans Mul chains for pairs of logs where:
// - Value of log_i == Base of log_j (or vice versa, since Mul is commutative)
//
// REDUCES log count: 2 logs → 1 log (naturally terminante)
//
// Soundness: EquivalenceUnderIntroducedRequires
// - Requires: both log arguments > 0, bases ≠ 1
// - These are already implied by the logs being defined
pub struct LogChainProductRule;

fn format_log_chain_product_desc(kind: LogChainProductRewriteKind) -> &'static str {
    match kind {
        LogChainProductRewriteKind::Telescoping => "log(b, a) * log(a, c) = log(b, c)",
    }
}

impl crate::rule::Rule for LogChainProductRule {
    fn name(&self) -> &str {
        "Log Chain (Telescoping)"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        let rewrite = try_rewrite_log_chain_product_expr(ctx, expr)?;
        Some(
            crate::rule::Rewrite::new(rewrite.rewritten)
                .desc(format_log_chain_product_desc(rewrite.kind)),
        )
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::MUL)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}
