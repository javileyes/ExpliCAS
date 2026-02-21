use cas_ast::{ConditionPredicate, ConditionSet, ExprId};

/// Domain mode used by the pure logarithmic decision table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainModeKind {
    Strict,
    Generic,
    Assume,
}

/// Trivalent proof status used by the pure logarithmic decision table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofStatus {
    Proven,
    Unknown,
    Disproven,
}

/// Missing assumptions required to justify a logarithmic transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogAssumption {
    PositiveRhs,
    PositiveBase,
}

/// Pure decision for whether a logarithmic solve step is justified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogSolveDecision {
    Ok,
    OkWithAssumptions(Vec<LogAssumption>),
    EmptySet(&'static str),
    NeedsComplex(&'static str),
    Unsupported(&'static str, Vec<LogAssumption>),
}

/// Convert one logical assumption into a `ConditionPredicate`.
pub fn assumption_to_condition_predicate(
    assumption: LogAssumption,
    base: ExprId,
    rhs: ExprId,
) -> ConditionPredicate {
    match assumption {
        LogAssumption::PositiveRhs => ConditionPredicate::Positive(rhs),
        LogAssumption::PositiveBase => ConditionPredicate::Positive(base),
    }
}

/// Convert a list of assumptions into a `ConditionSet`.
pub fn assumptions_to_condition_set(
    assumptions: &[LogAssumption],
    base: ExprId,
    rhs: ExprId,
) -> ConditionSet {
    let predicates: Vec<ConditionPredicate> = assumptions
        .iter()
        .map(|a| assumption_to_condition_predicate(*a, base, rhs))
        .collect();
    ConditionSet::from_predicates(predicates)
}

/// Classify a logarithmic solve step using only proof states and domain mode.
///
/// This function is intentionally context-free and does not inspect AST nodes.
pub fn classify_log_solve_by_proofs(
    mode: DomainModeKind,
    base_proof: ProofStatus,
    rhs_proof: ProofStatus,
) -> LogSolveDecision {
    // Case: base>0 proven and rhs<0 proven => EmptySet
    // (a^x > 0 for all real x when a > 0, so no solution exists)
    if base_proof == ProofStatus::Proven && rhs_proof == ProofStatus::Disproven {
        return LogSolveDecision::EmptySet(
            "No real solutions: base^x > 0 for all real x, but RHS <= 0",
        );
    }

    // If base<=0 proven => needs complex (can't take real log of non-positive base)
    if base_proof == ProofStatus::Disproven {
        return LogSolveDecision::NeedsComplex("Cannot take real logarithm: base is not positive");
    }

    // If rhs<=0 proven => needs complex
    if rhs_proof == ProofStatus::Disproven {
        return LogSolveDecision::NeedsComplex("Cannot take real logarithm: RHS is not positive");
    }

    let base_ok = base_proof == ProofStatus::Proven;
    let rhs_ok = rhs_proof == ProofStatus::Proven;

    match (base_ok, rhs_ok, mode) {
        (true, true, _) => LogSolveDecision::Ok,
        (true, false, DomainModeKind::Assume) => {
            LogSolveDecision::OkWithAssumptions(vec![LogAssumption::PositiveRhs])
        }
        (true, false, DomainModeKind::Strict | DomainModeKind::Generic) => {
            LogSolveDecision::Unsupported(
                "Cannot prove RHS > 0 for logarithm",
                vec![LogAssumption::PositiveRhs],
            )
        }
        (false, true, DomainModeKind::Assume) => {
            LogSolveDecision::OkWithAssumptions(vec![LogAssumption::PositiveBase])
        }
        (false, true, DomainModeKind::Strict | DomainModeKind::Generic) => {
            LogSolveDecision::Unsupported(
                "Cannot prove base > 0 for logarithm",
                vec![LogAssumption::PositiveBase],
            )
        }
        (false, false, DomainModeKind::Assume) => LogSolveDecision::OkWithAssumptions(vec![
            LogAssumption::PositiveBase,
            LogAssumption::PositiveRhs,
        ]),
        (false, false, DomainModeKind::Strict | DomainModeKind::Generic) => {
            LogSolveDecision::Unsupported(
                "Cannot prove base > 0 and RHS > 0 for logarithm",
                vec![LogAssumption::PositiveBase, LogAssumption::PositiveRhs],
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_proven_is_ok() {
        let d = classify_log_solve_by_proofs(
            DomainModeKind::Generic,
            ProofStatus::Proven,
            ProofStatus::Proven,
        );
        assert_eq!(d, LogSolveDecision::Ok);
    }

    #[test]
    fn proven_base_and_disproven_rhs_is_empty_set() {
        let d = classify_log_solve_by_proofs(
            DomainModeKind::Generic,
            ProofStatus::Proven,
            ProofStatus::Disproven,
        );
        assert!(matches!(d, LogSolveDecision::EmptySet(_)));
    }

    #[test]
    fn unknown_rhs_in_assume_mode_requires_positive_rhs_assumption() {
        let d = classify_log_solve_by_proofs(
            DomainModeKind::Assume,
            ProofStatus::Proven,
            ProofStatus::Unknown,
        );
        assert_eq!(
            d,
            LogSolveDecision::OkWithAssumptions(vec![LogAssumption::PositiveRhs])
        );
    }

    #[test]
    fn unknown_rhs_in_generic_mode_is_unsupported() {
        let d = classify_log_solve_by_proofs(
            DomainModeKind::Generic,
            ProofStatus::Proven,
            ProofStatus::Unknown,
        );
        assert!(matches!(d, LogSolveDecision::Unsupported(_, _)));
    }
}
