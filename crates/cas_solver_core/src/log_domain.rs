use cas_ast::{ConditionPredicate, ConditionSet, Context, ExprId};

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

/// Full route for log-linear rewrites, including explicit base-one shortcut.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLinearRewriteRoute<'a> {
    /// `1^x = B` is handled by a higher-level shortcut.
    BaseOneShortcut,
    /// Rewrite can proceed under the listed assumptions.
    Proceed { assumptions: &'a [LogAssumption] },
    /// Rewrite should be skipped in current mode/decision.
    Blocked,
}

/// Handling route for `LogSolveDecision::Unsupported`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogUnsupportedRoute<'a> {
    NotUnsupported,
    ResidualBudgetExhausted {
        message: &'a str,
    },
    Guarded {
        message: &'a str,
        missing_conditions: &'a [LogAssumption],
    },
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

/// Classify log-linear rewrite route, folding in the base-one pre-check.
pub fn classify_log_linear_rewrite_route<'a>(
    base_is_one: bool,
    decision: &'a LogSolveDecision,
) -> LogLinearRewriteRoute<'a> {
    if base_is_one {
        return LogLinearRewriteRoute::BaseOneShortcut;
    }
    match classify_log_linear_rewrite_policy(decision) {
        LogLinearRewritePolicy::Proceed { assumptions } => {
            LogLinearRewriteRoute::Proceed { assumptions }
        }
        LogLinearRewritePolicy::Blocked => LogLinearRewriteRoute::Blocked,
    }
}

/// Extract assumptions implied by a decision (empty for non-assume decisions).
pub fn decision_assumptions(decision: &LogSolveDecision) -> &[LogAssumption] {
    match decision {
        LogSolveDecision::OkWithAssumptions(assumptions) => assumptions,
        _ => &[],
    }
}

/// Decide how to handle unsupported logarithmic rewrites.
pub fn classify_log_unsupported_route<'a>(
    decision: &'a LogSolveDecision,
    can_branch: bool,
) -> LogUnsupportedRoute<'a> {
    match decision {
        LogSolveDecision::Unsupported(message, missing_conditions) if can_branch => {
            LogUnsupportedRoute::Guarded {
                message,
                missing_conditions,
            }
        }
        LogSolveDecision::Unsupported(message, _) => {
            LogUnsupportedRoute::ResidualBudgetExhausted { message }
        }
        _ => LogUnsupportedRoute::NotUnsupported,
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

/// Classify with environment-proven positivity overrides.
///
/// When an expression is already known positive from the solver environment,
/// we treat it as `Proven` regardless of local proof status.
pub fn classify_log_solve_with_env(
    mode: DomainModeKind,
    base_in_env: bool,
    rhs_in_env: bool,
    base_proof: ProofStatus,
    rhs_proof: ProofStatus,
) -> LogSolveDecision {
    let effective_base = if base_in_env {
        ProofStatus::Proven
    } else {
        base_proof
    };
    let effective_rhs = if rhs_in_env {
        ProofStatus::Proven
    } else {
        rhs_proof
    };
    classify_log_solve_by_proofs(mode, effective_base, effective_rhs)
}

/// Classify a log-solve step taking value-domain gating into account.
///
/// When not operating in real-only mode, logarithmic rewrite checks are skipped
/// and the caller may proceed (multi-valued complex handling is outside this policy).
pub fn classify_log_solve_for_value_domain(
    value_domain_is_real_only: bool,
    mode: DomainModeKind,
    base_in_env: bool,
    rhs_in_env: bool,
    base_proof: ProofStatus,
    rhs_proof: ProofStatus,
) -> LogSolveDecision {
    if !value_domain_is_real_only {
        return LogSolveDecision::Ok;
    }
    classify_log_solve_with_env(mode, base_in_env, rhs_in_env, base_proof, rhs_proof)
}

/// Classify a log-solve step by proving positivity only when env facts are missing.
pub fn classify_log_solve_with_prover<F>(
    ctx: &Context,
    base: ExprId,
    rhs: ExprId,
    value_domain_is_real_only: bool,
    mode: DomainModeKind,
    base_in_env: bool,
    rhs_in_env: bool,
    mut prove_positive_status: F,
) -> LogSolveDecision
where
    F: FnMut(&Context, ExprId) -> ProofStatus,
{
    let base_proof = if base_in_env {
        ProofStatus::Proven
    } else {
        prove_positive_status(ctx, base)
    };
    let rhs_proof = if rhs_in_env {
        ProofStatus::Proven
    } else {
        prove_positive_status(ctx, rhs)
    };

    classify_log_solve_for_value_domain(
        value_domain_is_real_only,
        mode,
        base_in_env,
        rhs_in_env,
        base_proof,
        rhs_proof,
    )
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

    #[test]
    fn decision_assumptions_returns_assumptions_only_for_ok_with_assume() {
        let ok = LogSolveDecision::OkWithAssumptions(vec![LogAssumption::PositiveRhs]);
        let none = LogSolveDecision::Ok;
        assert_eq!(decision_assumptions(&ok), &[LogAssumption::PositiveRhs]);
        assert!(decision_assumptions(&none).is_empty());
    }

    #[test]
    fn classify_log_unsupported_route_budget_exhausted() {
        let decision = LogSolveDecision::Unsupported("u", vec![LogAssumption::PositiveBase]);
        assert_eq!(
            classify_log_unsupported_route(&decision, false),
            LogUnsupportedRoute::ResidualBudgetExhausted { message: "u" }
        );
    }

    #[test]
    fn classify_log_unsupported_route_guarded_when_branch_available() {
        let decision = LogSolveDecision::Unsupported(
            "u",
            vec![LogAssumption::PositiveBase, LogAssumption::PositiveRhs],
        );
        match classify_log_unsupported_route(&decision, true) {
            LogUnsupportedRoute::Guarded {
                message,
                missing_conditions,
            } => {
                assert_eq!(message, "u");
                assert_eq!(
                    missing_conditions,
                    &[LogAssumption::PositiveBase, LogAssumption::PositiveRhs]
                );
            }
            _ => panic!("expected guarded route"),
        }
    }

    #[test]
    fn log_linear_route_shortcuts_base_one() {
        let decision = LogSolveDecision::Ok;
        assert_eq!(
            classify_log_linear_rewrite_route(true, &decision),
            LogLinearRewriteRoute::BaseOneShortcut
        );
    }

    #[test]
    fn log_linear_route_proceeds_for_ok_with_assumptions() {
        let decision = LogSolveDecision::OkWithAssumptions(vec![LogAssumption::PositiveBase]);
        match classify_log_linear_rewrite_route(false, &decision) {
            LogLinearRewriteRoute::Proceed { assumptions } => {
                assert_eq!(assumptions, &[LogAssumption::PositiveBase]);
            }
            _ => panic!("expected proceed route"),
        }
    }

    #[test]
    fn log_linear_route_blocks_for_unsupported() {
        let decision = LogSolveDecision::Unsupported("u", vec![LogAssumption::PositiveBase]);
        assert_eq!(
            classify_log_linear_rewrite_route(false, &decision),
            LogLinearRewriteRoute::Blocked
        );
    }

    #[test]
    fn classify_log_solve_with_env_overrides_unknown_rhs_to_proven() {
        let out = classify_log_solve_with_env(
            DomainModeKind::Generic,
            true,
            true,
            ProofStatus::Unknown,
            ProofStatus::Unknown,
        );
        assert_eq!(out, LogSolveDecision::Ok);
    }

    #[test]
    fn classify_log_solve_with_env_respects_non_env_disproof() {
        let out = classify_log_solve_with_env(
            DomainModeKind::Generic,
            false,
            false,
            ProofStatus::Disproven,
            ProofStatus::Proven,
        );
        assert!(matches!(out, LogSolveDecision::NeedsComplex(_)));
    }

    #[test]
    fn classify_log_solve_for_value_domain_skips_checks_when_not_real_only() {
        let out = classify_log_solve_for_value_domain(
            false,
            DomainModeKind::Generic,
            false,
            false,
            ProofStatus::Disproven,
            ProofStatus::Disproven,
        );
        assert_eq!(out, LogSolveDecision::Ok);
    }

    #[test]
    fn classify_log_solve_for_value_domain_uses_real_policy_when_enabled() {
        let out = classify_log_solve_for_value_domain(
            true,
            DomainModeKind::Generic,
            false,
            false,
            ProofStatus::Proven,
            ProofStatus::Disproven,
        );
        assert!(matches!(out, LogSolveDecision::EmptySet(_)));
    }

    #[test]
    fn classify_log_solve_with_prover_queries_only_missing_env_proofs() {
        let mut ctx = cas_ast::Context::new();
        let base = ctx.var("b");
        let rhs = ctx.var("r");
        let mut calls = 0usize;

        let out = classify_log_solve_with_prover(
            &ctx,
            base,
            rhs,
            true,
            DomainModeKind::Generic,
            true,
            false,
            |_ctx, _expr| {
                calls += 1;
                ProofStatus::Unknown
            },
        );

        assert_eq!(calls, 1);
        assert!(matches!(out, LogSolveDecision::Unsupported(_, _)));
    }

    #[test]
    fn classify_log_solve_with_prover_skips_callback_when_env_proves_both() {
        let mut ctx = cas_ast::Context::new();
        let base = ctx.var("b");
        let rhs = ctx.var("r");
        let mut calls = 0usize;

        let out = classify_log_solve_with_prover(
            &ctx,
            base,
            rhs,
            true,
            DomainModeKind::Generic,
            true,
            true,
            |_ctx, _expr| {
                calls += 1;
                ProofStatus::Unknown
            },
        );

        assert_eq!(calls, 0);
        assert_eq!(out, LogSolveDecision::Ok);
    }
}
