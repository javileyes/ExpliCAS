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

/// Terminal handling action for a log-solve decision in solver pipelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogTerminalAction {
    Continue,
    ReturnEmptySet,
    ReturnResidualInWildcard,
}

/// Policy for whether an `A^x = B` log-linear rewrite can proceed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLinearRewritePolicy<'a> {
    /// Rewrite is allowed. Any returned assumptions must be recorded by caller.
    Proceed { assumptions: &'a [LogAssumption] },
    /// Rewrite is blocked under current decision.
    Blocked,
}

/// Map a decision to terminal solver action.
pub fn classify_terminal_action(
    decision: &LogSolveDecision,
    mode: DomainModeKind,
    wildcard_scope: bool,
) -> LogTerminalAction {
    match decision {
        LogSolveDecision::EmptySet(_) => LogTerminalAction::ReturnEmptySet,
        LogSolveDecision::NeedsComplex(_) if mode == DomainModeKind::Assume && wildcard_scope => {
            LogTerminalAction::ReturnResidualInWildcard
        }
        _ => LogTerminalAction::Continue,
    }
}

/// Classify whether the solver may rewrite `A^x = B` into a log-linear form.
pub fn classify_log_linear_rewrite_policy<'a>(
    decision: &'a LogSolveDecision,
) -> LogLinearRewritePolicy<'a> {
    match decision {
        LogSolveDecision::Ok => LogLinearRewritePolicy::Proceed { assumptions: &[] },
        LogSolveDecision::OkWithAssumptions(assumptions) => {
            LogLinearRewritePolicy::Proceed { assumptions }
        }
        LogSolveDecision::EmptySet(_)
        | LogSolveDecision::NeedsComplex(_)
        | LogSolveDecision::Unsupported(_, _) => LogLinearRewritePolicy::Blocked,
    }
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

/// Select the expression targeted by a logarithmic assumption.
pub fn assumption_target_expr(assumption: LogAssumption, base: ExprId, rhs: ExprId) -> ExprId {
    match assumption {
        LogAssumption::PositiveBase => base,
        LogAssumption::PositiveRhs => rhs,
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

    #[test]
    fn terminal_action_empty_set() {
        let d = LogSolveDecision::EmptySet("x");
        let action = classify_terminal_action(&d, DomainModeKind::Generic, false);
        assert_eq!(action, LogTerminalAction::ReturnEmptySet);
    }

    #[test]
    fn terminal_action_needs_complex_wildcard_assume() {
        let d = LogSolveDecision::NeedsComplex("x");
        let action = classify_terminal_action(&d, DomainModeKind::Assume, true);
        assert_eq!(action, LogTerminalAction::ReturnResidualInWildcard);
    }

    #[test]
    fn terminal_action_needs_complex_non_wildcard() {
        let d = LogSolveDecision::NeedsComplex("x");
        let action = classify_terminal_action(&d, DomainModeKind::Assume, false);
        assert_eq!(action, LogTerminalAction::Continue);
    }

    #[test]
    fn assumption_target_expr_maps_base_and_rhs() {
        let mut ctx = cas_ast::Context::new();
        let base = ctx.var("b");
        let rhs = ctx.var("r");
        assert_eq!(
            assumption_target_expr(LogAssumption::PositiveBase, base, rhs),
            base
        );
        assert_eq!(
            assumption_target_expr(LogAssumption::PositiveRhs, base, rhs),
            rhs
        );
    }

    #[test]
    fn log_linear_policy_allows_ok_without_assumptions() {
        let decision = LogSolveDecision::Ok;
        assert_eq!(
            classify_log_linear_rewrite_policy(&decision),
            LogLinearRewritePolicy::Proceed { assumptions: &[] }
        );
    }

    #[test]
    fn log_linear_policy_allows_assume_with_assumptions() {
        let decision = LogSolveDecision::OkWithAssumptions(vec![
            LogAssumption::PositiveBase,
            LogAssumption::PositiveRhs,
        ]);
        match classify_log_linear_rewrite_policy(&decision) {
            LogLinearRewritePolicy::Proceed { assumptions } => {
                assert_eq!(
                    assumptions,
                    &[LogAssumption::PositiveBase, LogAssumption::PositiveRhs]
                );
            }
            LogLinearRewritePolicy::Blocked => panic!("expected proceed policy"),
        }
    }

    #[test]
    fn log_linear_policy_blocks_unsupported_and_needs_complex() {
        let unsupported = LogSolveDecision::Unsupported("u", vec![LogAssumption::PositiveBase]);
        let needs_complex = LogSolveDecision::NeedsComplex("c");
        assert_eq!(
            classify_log_linear_rewrite_policy(&unsupported),
            LogLinearRewritePolicy::Blocked
        );
        assert_eq!(
            classify_log_linear_rewrite_policy(&needs_complex),
            LogLinearRewritePolicy::Blocked
        );
    }
}
