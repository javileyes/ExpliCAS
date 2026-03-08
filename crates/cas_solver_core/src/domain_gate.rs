/// Canonical helper: Decide whether to cancel a factor based on domain mode.
///
/// This is the single point of truth for cancellation-style conditions
/// (`NonZero` / definability class).
pub fn can_cancel_factor(
    mode: crate::domain_mode::DomainMode,
    proof: crate::domain_proof::Proof,
) -> crate::domain_cancel_decision::CancelDecision {
    decide_by_class(
        mode,
        crate::solve_safety_policy::ConditionClass::Definability,
        crate::domain_facts_model::proof_to_strength(proof),
        "cancelled factor assumed nonzero",
    )
}

/// Rich version of `can_cancel_factor` that emits pedagogical hints when
/// Strict mode blocks an unknown nonzero condition.
pub fn can_cancel_factor_with_hint(
    mode: crate::domain_mode::DomainMode,
    proof: crate::domain_proof::Proof,
    key: crate::assumption_model::AssumptionKey,
    expr_id: cas_ast::ExprId,
    rule: &'static str,
) -> crate::domain_cancel_decision::CancelDecision {
    use crate::domain_mode::DomainMode;
    use crate::domain_proof::Proof;
    use crate::solve_safety_policy::ConditionClass;

    match proof {
        Proof::Proven | Proof::ProvenImplicit => {
            crate::domain_cancel_decision::CancelDecision::allow()
        }
        Proof::Disproven => crate::domain_cancel_decision::CancelDecision::deny(),
        Proof::Unknown => {
            if mode.allows_unproven(ConditionClass::Definability) {
                let keys = smallvec::smallvec![key.clone()];
                crate::domain_cancel_decision::CancelDecision::allow_with_keys(
                    "cancelled factor assumed nonzero",
                    keys,
                )
            } else if mode == DomainMode::Strict {
                let hint = crate::blocked_hint::BlockedHint {
                    key: key.clone(),
                    expr_id,
                    rule: rule.to_string(),
                    suggestion: "use `domain generic` to allow definability assumptions",
                };
                crate::blocked_hint_store::register_blocked_hint(hint);
                crate::domain_cancel_decision::CancelDecision::deny_with_hint(key, expr_id, rule)
            } else {
                crate::domain_cancel_decision::CancelDecision::deny()
            }
        }
    }
}

/// Canonical helper: Decide whether to apply an Analytic condition
/// (`Positive` / `NonNegative`) based on domain mode.
pub fn can_apply_analytic(
    mode: crate::domain_mode::DomainMode,
    proof: crate::domain_proof::Proof,
) -> crate::domain_cancel_decision::CancelDecision {
    decide_by_class(
        mode,
        crate::solve_safety_policy::ConditionClass::Analytic,
        crate::domain_facts_model::proof_to_strength(proof),
        "assumed positive",
    )
}

/// Rich version of `can_apply_analytic` that emits pedagogical hints when
/// Generic mode blocks an unknown analytic condition.
pub fn can_apply_analytic_with_hint(
    mode: crate::domain_mode::DomainMode,
    proof: crate::domain_proof::Proof,
    key: crate::assumption_model::AssumptionKey,
    expr_id: cas_ast::ExprId,
    rule: &'static str,
) -> crate::domain_cancel_decision::CancelDecision {
    use crate::domain_mode::DomainMode;
    use crate::domain_proof::Proof;
    use crate::solve_safety_policy::ConditionClass;

    match proof {
        Proof::Proven | Proof::ProvenImplicit => {
            crate::domain_cancel_decision::CancelDecision::allow()
        }
        Proof::Disproven => crate::domain_cancel_decision::CancelDecision::deny(),
        Proof::Unknown => {
            if mode.allows_unproven(ConditionClass::Analytic) {
                let keys = smallvec::smallvec![key.clone()];
                crate::domain_cancel_decision::CancelDecision::allow_with_keys(
                    "assumed positive",
                    keys,
                )
            } else if mode == DomainMode::Generic {
                let hint = crate::blocked_hint::BlockedHint {
                    key: key.clone(),
                    expr_id,
                    rule: rule.to_string(),
                    suggestion: "use `semantics set domain assume` to allow analytic assumptions",
                };
                crate::blocked_hint_store::register_blocked_hint(hint);
                crate::domain_cancel_decision::CancelDecision::deny_with_hint(key, expr_id, rule)
            } else {
                crate::domain_cancel_decision::CancelDecision::deny()
            }
        }
    }
}

/// Unified decision helper for a full predicate.
pub fn decide(
    mode: crate::domain_mode::DomainMode,
    pred: &crate::domain_facts_model::Predicate,
    strength: crate::domain_facts_model::FactStrength,
) -> crate::domain_cancel_decision::CancelDecision {
    match crate::domain_policy::decide_policy(mode, pred, strength) {
        crate::domain_policy::PolicyDecision::AllowProven => {
            crate::domain_cancel_decision::CancelDecision::allow()
        }
        crate::domain_policy::PolicyDecision::AllowAssumed => {
            crate::domain_cancel_decision::CancelDecision::allow_with_assumption(pred.describe())
        }
        crate::domain_policy::PolicyDecision::Deny => {
            crate::domain_cancel_decision::CancelDecision::deny()
        }
    }
}

/// Unified decision helper when condition class is already known.
pub fn decide_by_class(
    mode: crate::domain_mode::DomainMode,
    class: crate::solve_safety_policy::ConditionClass,
    strength: crate::domain_facts_model::FactStrength,
    assumption_label: &'static str,
) -> crate::domain_cancel_decision::CancelDecision {
    match crate::domain_policy::decide_policy_by_class(mode, class, strength) {
        crate::domain_policy::PolicyDecision::AllowProven => {
            crate::domain_cancel_decision::CancelDecision::allow()
        }
        crate::domain_policy::PolicyDecision::AllowAssumed => {
            crate::domain_cancel_decision::CancelDecision::allow_with_assumption(assumption_label)
        }
        crate::domain_policy::PolicyDecision::Deny => {
            crate::domain_cancel_decision::CancelDecision::deny()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        can_apply_analytic, can_apply_analytic_with_hint, can_cancel_factor,
        can_cancel_factor_with_hint, decide, decide_by_class,
    };
    use crate::assumption_model::AssumptionKey;
    use crate::domain_facts_model::{proof_to_strength, FactStrength, Predicate};
    use crate::domain_mode::DomainMode;
    use crate::domain_proof::Proof;
    use crate::solve_safety_policy::ConditionClass;

    #[test]
    fn can_cancel_factor_matrix() {
        let proven = can_cancel_factor(DomainMode::Strict, Proof::Proven);
        assert!(proven.allow);
        assert!(proven.assumption.is_none());

        let strict_unknown = can_cancel_factor(DomainMode::Strict, Proof::Unknown);
        assert!(!strict_unknown.allow);

        let generic_unknown = can_cancel_factor(DomainMode::Generic, Proof::Unknown);
        assert!(generic_unknown.allow);
        assert_eq!(
            generic_unknown.assumption,
            Some("cancelled factor assumed nonzero")
        );
    }

    #[test]
    fn can_apply_analytic_matrix() {
        let proven = can_apply_analytic(DomainMode::Strict, Proof::Proven);
        assert!(proven.allow);

        let generic_unknown = can_apply_analytic(DomainMode::Generic, Proof::Unknown);
        assert!(!generic_unknown.allow);

        let assume_unknown = can_apply_analytic(DomainMode::Assume, Proof::Unknown);
        assert!(assume_unknown.allow);
        assert_eq!(assume_unknown.assumption, Some("assumed positive"));
    }

    #[test]
    fn hint_helpers_attach_or_block_as_expected() {
        let mut ctx = cas_ast::Context::default();
        let x = ctx.var("x");
        let key = AssumptionKey::nonzero_key(&ctx, x);

        let strict_block =
            can_cancel_factor_with_hint(DomainMode::Strict, Proof::Unknown, key.clone(), x, "Rule");
        assert!(!strict_block.allow);
        assert!(strict_block.blocked_hint.is_some());

        let generic_allow = can_cancel_factor_with_hint(
            DomainMode::Generic,
            Proof::Unknown,
            key.clone(),
            x,
            "Rule",
        );
        assert!(generic_allow.allow);
        assert_eq!(generic_allow.assumed_keys.len(), 1);

        let pos_key = AssumptionKey::positive_key(&ctx, x);
        let generic_block =
            can_apply_analytic_with_hint(DomainMode::Generic, Proof::Unknown, pos_key, x, "Rule");
        assert!(!generic_block.allow);
        assert!(generic_block.blocked_hint.is_some());
    }

    #[test]
    fn decide_helpers_map_policy_to_cancel_decision() {
        let mut ctx = cas_ast::Context::default();
        let x = ctx.var("x");
        let pred = Predicate::NonZero(x);

        let strict_unknown = decide(DomainMode::Strict, &pred, FactStrength::Unknown);
        assert!(!strict_unknown.allow);

        let generic_unknown = decide(DomainMode::Generic, &pred, FactStrength::Unknown);
        assert!(generic_unknown.allow);
        assert!(generic_unknown.assumption.is_some());

        let analytic_unknown = decide_by_class(
            DomainMode::Generic,
            ConditionClass::Analytic,
            FactStrength::Unknown,
            "assumed positive",
        );
        assert!(!analytic_unknown.allow);
    }

    #[test]
    fn decide_nonzero_matches_can_cancel_factor_for_all_mode_proof_pairs() {
        let mut ctx = cas_ast::Context::default();
        let x = ctx.var("x");
        let pred = Predicate::NonZero(x);

        for mode in [DomainMode::Strict, DomainMode::Generic, DomainMode::Assume] {
            for proof in [
                Proof::Proven,
                Proof::ProvenImplicit,
                Proof::Unknown,
                Proof::Disproven,
            ] {
                let via_predicate = decide(mode, &pred, proof_to_strength(proof));
                let via_class = can_cancel_factor(mode, proof);
                assert_eq!(
                    via_predicate.allow, via_class.allow,
                    "mode={mode:?} proof={proof:?}"
                );
            }
        }
    }
}
