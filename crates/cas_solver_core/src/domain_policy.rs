/// Core domain-policy outcome independent from any outer diagnostic type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PolicyDecision {
    /// Allowed because the predicate is proven.
    AllowProven,
    /// Allowed because predicate is unknown but mode permits assumption.
    AllowAssumed,
    /// Not allowed.
    Deny,
}

/// Check whether a mode allows an unproven predicate type.
#[inline]
pub fn mode_allows_predicate(
    mode: crate::domain_mode::DomainMode,
    pred: &crate::domain_facts_model::Predicate,
) -> bool {
    mode.allows_unproven(pred.condition_class())
}

/// Decide policy for a full predicate.
#[inline]
pub fn decide_policy(
    mode: crate::domain_mode::DomainMode,
    pred: &crate::domain_facts_model::Predicate,
    strength: crate::domain_facts_model::FactStrength,
) -> PolicyDecision {
    decide_policy_by_class(mode, pred.condition_class(), strength)
}

/// Decide policy when condition class is already known.
#[inline]
pub fn decide_policy_by_class(
    mode: crate::domain_mode::DomainMode,
    class: crate::solve_safety_policy::ConditionClass,
    strength: crate::domain_facts_model::FactStrength,
) -> PolicyDecision {
    use crate::domain_facts_model::FactStrength;

    match strength {
        FactStrength::Proven => PolicyDecision::AllowProven,
        FactStrength::Disproven => PolicyDecision::Deny,
        FactStrength::Unknown => {
            if mode.allows_unproven(class) {
                PolicyDecision::AllowAssumed
            } else {
                PolicyDecision::Deny
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{decide_policy, decide_policy_by_class, mode_allows_predicate, PolicyDecision};
    use crate::domain_facts_model::{FactStrength, Predicate};
    use crate::domain_mode::DomainMode;
    use crate::solve_safety_policy::ConditionClass;

    #[test]
    fn mode_allows_predicate_matches_contract() {
        let mut ctx = cas_ast::Context::default();
        let x = ctx.var("x");
        let nonzero = Predicate::NonZero(x);
        let positive = Predicate::Positive(x);

        assert!(!mode_allows_predicate(DomainMode::Strict, &nonzero));
        assert!(!mode_allows_predicate(DomainMode::Strict, &positive));

        assert!(mode_allows_predicate(DomainMode::Generic, &nonzero));
        assert!(!mode_allows_predicate(DomainMode::Generic, &positive));

        assert!(mode_allows_predicate(DomainMode::Assume, &nonzero));
        assert!(mode_allows_predicate(DomainMode::Assume, &positive));
    }

    #[test]
    fn decide_policy_handles_strength_matrix() {
        let mut ctx = cas_ast::Context::default();
        let x = ctx.var("x");
        let pred = Predicate::NonZero(x);

        assert_eq!(
            decide_policy(DomainMode::Strict, &pred, FactStrength::Proven),
            PolicyDecision::AllowProven
        );
        assert_eq!(
            decide_policy(DomainMode::Assume, &pred, FactStrength::Disproven),
            PolicyDecision::Deny
        );
        assert_eq!(
            decide_policy(DomainMode::Strict, &pred, FactStrength::Unknown),
            PolicyDecision::Deny
        );
        assert_eq!(
            decide_policy(DomainMode::Generic, &pred, FactStrength::Unknown),
            PolicyDecision::AllowAssumed
        );
        assert_eq!(
            decide_policy(DomainMode::Assume, &pred, FactStrength::Unknown),
            PolicyDecision::AllowAssumed
        );
    }

    #[test]
    fn decide_policy_by_class_matches_expected() {
        assert_eq!(
            decide_policy_by_class(
                DomainMode::Strict,
                ConditionClass::Definability,
                FactStrength::Unknown,
            ),
            PolicyDecision::Deny
        );
        assert_eq!(
            decide_policy_by_class(
                DomainMode::Generic,
                ConditionClass::Definability,
                FactStrength::Unknown,
            ),
            PolicyDecision::AllowAssumed
        );
        assert_eq!(
            decide_policy_by_class(
                DomainMode::Generic,
                ConditionClass::Analytic,
                FactStrength::Unknown,
            ),
            PolicyDecision::Deny
        );
    }
}
