use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::Context;
use cas_ast::ExprId;

const ROOT_CANCEL_ASSUME_SUGGESTION: &str =
    "Use 'semantics set domain assume' to simplify (x^n)^(1/n) → x.";

fn register_symbolic_root_cancel_hints(
    ctx: &cas_ast::Context,
    rule: &str,
    inner_base: ExprId,
    inner_exp: ExprId,
) {
    let hint1 = crate::BlockedHint {
        key: crate::AssumptionKey::positive_key(ctx, inner_base),
        expr_id: inner_base,
        rule: rule.to_string(),
        suggestion: ROOT_CANCEL_ASSUME_SUGGESTION,
    };
    let hint2 = crate::BlockedHint {
        key: crate::AssumptionKey::nonzero_key(ctx, inner_exp),
        expr_id: inner_exp,
        rule: rule.to_string(),
        suggestion: ROOT_CANCEL_ASSUME_SUGGESTION,
    };
    crate::register_blocked_hint(hint1);
    crate::register_blocked_hint(hint2);
}

fn assumed_symbolic_root_cancel_rewrite(
    ctx: &Context,
    rewritten: ExprId,
    inner_base: ExprId,
    inner_exp: ExprId,
) -> Rewrite {
    use crate::ImplicitCondition;
    Rewrite::new(rewritten)
        .desc("(x^n)^(1/n) = x (assuming x > 0, n ≠ 0)")
        .requires(ImplicitCondition::Positive(inner_base))
        .requires(ImplicitCondition::NonZero(inner_exp))
        .assume(crate::AssumptionEvent::positive_assumed(ctx, inner_base))
}

fn apply_symbolic_root_cancel_action(
    ctx: &Context,
    rule_name: &str,
    action: cas_math::root_power_canonical_support::SymbolicRootCancelAction,
    unconditional_desc: &'static str,
) -> Option<Rewrite> {
    match action {
        cas_math::root_power_canonical_support::SymbolicRootCancelAction::BlockedNeedsAssumeMode {
            inner_base,
            inner_exp,
        } => {
            register_symbolic_root_cancel_hints(ctx, rule_name, inner_base, inner_exp);
            None
        }
        cas_math::root_power_canonical_support::SymbolicRootCancelAction::ApplyWithAssumptions {
            rewritten,
            inner_base,
            inner_exp,
        } => Some(assumed_symbolic_root_cancel_rewrite(
            ctx, rewritten, inner_base, inner_exp,
        )),
        cas_math::root_power_canonical_support::SymbolicRootCancelAction::ApplyUnconditionally {
            rewritten,
        } => Some(Rewrite::new(rewritten).desc(unconditional_desc)),
    }
}

fn symbolic_root_cancel_action_for_parent(
    parent_ctx: &crate::parent_context::ParentContext,
    rewritten: ExprId,
    inner_base: ExprId,
    inner_exp: ExprId,
) -> cas_math::root_power_canonical_support::SymbolicRootCancelAction {
    use crate::semantics::ValueDomain;

    let domain_mode = parent_ctx.domain_mode();
    cas_math::root_power_canonical_support::plan_symbolic_root_cancel_action_with_mode_flags(
        parent_ctx.value_domain() == ValueDomain::RealOnly,
        matches!(domain_mode, crate::DomainMode::Assume),
        matches!(domain_mode, crate::DomainMode::Strict),
        rewritten,
        inner_base,
        inner_exp,
    )
}

fn power_power_nonnegative_proof_with_witness(
    core_ctx: &Context,
    base: ExprId,
    full_expr: ExprId,
    parent_ctx: &crate::parent_context::ParentContext,
    value_domain: crate::semantics::ValueDomain,
) -> cas_math::tri_proof::TriProof {
    use crate::{witness_survives_in_context, WitnessKind};

    let explicit = cas_solver_core::predicate_proofs::prove_nonnegative_core_with(
        core_ctx,
        base,
        value_domain,
        crate::helpers::prove_nonnegative,
    );
    let (implicit_contains_nonnegative, witness_survives) = if let (Some(implicit), Some(root)) =
        (parent_ctx.implicit_domain(), parent_ctx.root_expr())
    {
        let contains = implicit.contains_nonnegative(base);
        let survives = contains
            && witness_survives_in_context(
                core_ctx,
                base,
                root,
                full_expr,
                Some(base),
                WitnessKind::Sqrt,
            );
        (contains, survives)
    } else {
        (false, false)
    };

    cas_math::root_power_canonical_support::merge_nonnegative_proof_with_witness(
        explicit,
        implicit_contains_nonnegative,
        witness_survives,
    )
}

fn apply_power_power_even_root_action(
    ctx: &Context,
    parent_ctx: &crate::parent_context::ParentContext,
    action: cas_math::root_power_canonical_support::PowerPowerEvenRootAction,
) -> Option<Rewrite> {
    match action {
        cas_math::root_power_canonical_support::PowerPowerEvenRootAction::Apply { rewritten } => {
            Some(Rewrite::new(rewritten).desc("Multiply exponents"))
        }
        cas_math::root_power_canonical_support::PowerPowerEvenRootAction::NeedsNonNegativeCondition {
            rewritten,
            inner_base,
        } => {
            let mode = parent_ctx.domain_mode();
            let vd = parent_ctx.value_domain();
            let decision = crate::oracle_allows_with_hint(
                ctx,
                mode,
                vd,
                &crate::Predicate::NonNegative(inner_base),
                "Power of a Power",
            );
            if !decision.allow {
                return None;
            }

            let mut rewrite = Rewrite::new(rewritten).desc("Multiply exponents");
            if decision.assumption.is_some() {
                rewrite =
                    rewrite.assume(crate::AssumptionEvent::nonnegative(ctx, inner_base));
            }
            Some(rewrite)
        }
    }
}

define_rule!(ProductPowerRule, "Product of Powers", |ctx, expr| {
    let rewrite = cas_math::power_product_support::try_rewrite_product_power_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
});

// a^n * b^n = (ab)^n - combines products of powers with same exponent
// Guard: at least one base must contain a numeric factor to avoid infinite loop with PowerProductRule
define_rule!(
    ProductSameExponentRule,
    "Product Same Exponent",
    None,
    PhaseMask::CORE | PhaseMask::TRANSFORM | PhaseMask::RATIONALIZE,
    |ctx, expr| {
        let rewrite =
            cas_math::power_product_support::try_rewrite_product_same_exponent_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// a^n / b^n = (a/b)^n
define_rule!(
    QuotientSameExponentRule,
    "Quotient Same Exponent",
    None,
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    |ctx, expr| {
        let rewrite =
            cas_math::power_product_support::try_rewrite_quotient_same_exponent_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// ============================================================================
// RootPowCancelRule: (x^n)^(1/n) → x (odd n) or |x| (even n)
// ============================================================================
pub struct RootPowCancelRule;

impl crate::rule::Rule for RootPowCancelRule {
    fn name(&self) -> &str {
        "Root Power Cancel"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::semantics::ValueDomain;

        let pattern =
            cas_math::root_power_canonical_support::classify_root_pow_cancel_pattern(ctx, expr)?;

        let vd = parent_ctx.value_domain();
        if vd == ValueDomain::ComplexEnabled {
            return None;
        }

        match pattern {
            cas_math::root_power_canonical_support::RootPowCancelPattern::NumericEven {
                rewritten,
            } => Some(crate::rule::Rewrite::new(rewritten).desc("(x^n)^(1/n) = |x| for even n")),
            cas_math::root_power_canonical_support::RootPowCancelPattern::NumericOdd {
                rewritten,
            } => Some(crate::rule::Rewrite::new(rewritten).desc("(x^n)^(1/n) = x for odd n")),
            cas_math::root_power_canonical_support::RootPowCancelPattern::SymbolicCandidate {
                rewritten,
                inner_base,
                inner_exp,
            } => {
                let action = symbolic_root_cancel_action_for_parent(
                    parent_ctx, rewritten, inner_base, inner_exp,
                );
                apply_symbolic_root_cancel_action(
                    ctx,
                    "Root Power Cancel",
                    action,
                    "(x^n)^(1/n) = x",
                )
            }
        }
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::POW)
    }

    fn priority(&self) -> i32 {
        15
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

define_rule!(
    PowerPowerRule,
    "Power of a Power",
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Analytic
    ),
    |ctx, expr, parent_ctx| {
    // (x^a)^b -> x^(a*b)
    if let Some(pattern) = cas_math::root_power_canonical_support::classify_power_power_pattern(ctx, expr) {
        match pattern {
            cas_math::root_power_canonical_support::PowerPowerPattern::EvenRootAbs { rewritten } => {
                return Some(Rewrite::new(rewritten).desc(
                    "Power of power with even root: (x^2k)^(1/2) -> |x|^k",
                ));
            }
            cas_math::root_power_canonical_support::PowerPowerPattern::EvenRootNeedsNonNegative { .. } => {
                let vd = parent_ctx.value_domain();

                let action = cas_math::root_power_canonical_support::plan_power_power_even_root_action_with(
                    ctx,
                    expr,
                    |core_ctx, base| {
                        power_power_nonnegative_proof_with_witness(
                            core_ctx, base, expr, parent_ctx, vd,
                        )
                    },
                )?;
                return apply_power_power_even_root_action(ctx, parent_ctx, action);
            }
            cas_math::root_power_canonical_support::PowerPowerPattern::SymbolicRootCancelCandidate {
                rewritten,
                inner_base,
                inner_exp,
            } => {
                let action =
                    symbolic_root_cancel_action_for_parent(parent_ctx, rewritten, inner_base, inner_exp);
                if let Some(rewrite) = apply_symbolic_root_cancel_action(
                    ctx,
                    "Power of a Power",
                    action,
                    "Multiply exponents",
                ) {
                    return Some(rewrite);
                }
                return None;
            }
            cas_math::root_power_canonical_support::PowerPowerPattern::MultiplyExponents {
                rewritten,
            } => {
                return Some(Rewrite::new(rewritten).desc("Multiply exponents"));
            }
        }
    }
    None
});

// ============================================================================
// NegativeExponentNormalizationRule: x^(-n) → 1/x^n
// ============================================================================
define_rule!(
    NegativeExponentNormalizationRule,
    "Normalize Negative Exponent",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite =
            cas_math::power_eval_support::try_rewrite_negative_exponent_normalization_expr(
                ctx, expr,
            )?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(EvaluatePowerRule, "Evaluate Numeric Power", importance: crate::step::ImportanceLevel::Low, |ctx, expr| {
    let rewrite = cas_math::power_eval_support::try_rewrite_evaluate_power_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
});
