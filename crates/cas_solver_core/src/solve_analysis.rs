use cas_ast::{
    Case, ConditionPredicate, ConditionSet, Context, Equation, Expr, ExprId, SolutionSet,
};
use std::collections::HashSet;
use std::hash::Hash;

use crate::solve_outcome::{
    solve_var_eliminated_outcome_pipeline_with, VarEliminatedOutcomePipelineSolved,
};

/// Check if an expression is symbolic (contains variables/functions/constants).
pub fn is_symbolic_expr(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(_) => false,
        Expr::Constant(_) => true,
        Expr::Variable(_) => true,
        Expr::Function(_, _) => true,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            is_symbolic_expr(ctx, *l) || is_symbolic_expr(ctx, *r)
        }
        Expr::Neg(e) | Expr::Hold(e) => is_symbolic_expr(ctx, *e),
        Expr::Matrix { data, .. } => data.iter().any(|d| is_symbolic_expr(ctx, *d)),
        Expr::SessionRef(_) => true,
    }
}

/// Split discrete solutions into `(symbolic, non_symbolic)` buckets.
pub fn partition_discrete_symbolic(ctx: &Context, sols: &[ExprId]) -> (Vec<ExprId>, Vec<ExprId>) {
    let mut symbolic = Vec::new();
    let mut non_symbolic = Vec::new();
    for &sol in sols {
        if is_symbolic_expr(ctx, sol) {
            symbolic.push(sol);
        } else {
            non_symbolic.push(sol);
        }
    }
    (symbolic, non_symbolic)
}

/// Keep only solutions accepted by a verifier callback.
pub fn retain_verified_discrete<F>(sols: Vec<ExprId>, mut verify: F) -> Vec<ExprId>
where
    F: FnMut(ExprId) -> bool,
{
    let mut out = Vec::new();
    for sol in sols {
        if verify(sol) {
            out.push(sol);
        }
    }
    out
}

/// Merge symbolic roots with verified numeric roots.
///
/// Returns `symbolic ++ verified_numeric` preserving solver behavior.
pub fn merge_symbolic_with_verified_numeric<F>(
    mut symbolic_solutions: Vec<ExprId>,
    numeric_solutions: Vec<ExprId>,
    mut verify_numeric: F,
) -> Vec<ExprId>
where
    F: FnMut(ExprId) -> bool,
{
    let verified_numeric = retain_verified_discrete(numeric_solutions, &mut verify_numeric);
    symbolic_solutions.extend(verified_numeric);
    symbolic_solutions
}

/// Resolve discrete strategy candidates by preserving symbolic roots and
/// verifying only numeric roots with caller-provided callbacks.
pub fn resolve_discrete_strategy_solutions_with<FIsSymbolic, FVerify>(
    solutions: Vec<ExprId>,
    mut is_symbolic: FIsSymbolic,
    mut verify_numeric: FVerify,
) -> Vec<ExprId>
where
    FIsSymbolic: FnMut(ExprId) -> bool,
    FVerify: FnMut(ExprId) -> bool,
{
    let mut symbolic_solutions = Vec::new();
    let mut numeric_solutions = Vec::new();
    for solution in solutions {
        if is_symbolic(solution) {
            symbolic_solutions.push(solution);
        } else {
            numeric_solutions.push(solution);
        }
    }

    merge_symbolic_with_verified_numeric(symbolic_solutions, numeric_solutions, &mut verify_numeric)
}

/// Resolve discrete strategy candidates using one mutable caller state.
///
/// This variant is useful when symbolic classification and numeric verification
/// both need mutable access to the same runtime state object.
pub fn resolve_discrete_strategy_solutions_with_state<S, FIsSymbolic, FVerify>(
    state: &mut S,
    solutions: Vec<ExprId>,
    mut is_symbolic: FIsSymbolic,
    mut verify_numeric: FVerify,
) -> Vec<ExprId>
where
    FIsSymbolic: FnMut(&mut S, ExprId) -> bool,
    FVerify: FnMut(&mut S, ExprId) -> bool,
{
    let mut symbolic_solutions = Vec::new();
    let mut verified_numeric = Vec::new();

    for solution in solutions {
        if is_symbolic(state, solution) {
            symbolic_solutions.push(solution);
        } else if verify_numeric(state, solution) {
            verified_numeric.push(solution);
        }
    }

    symbolic_solutions.extend(verified_numeric);
    symbolic_solutions
}

/// Resolve discrete strategy candidates against one equation:
/// - keep symbolic candidates untouched,
/// - verify only numeric candidates against `(equation, var)`.
pub fn resolve_discrete_strategy_solutions_against_equation_with_state<
    S,
    FIsSymbolic,
    FVerifyAgainstEquation,
>(
    state: &mut S,
    equation: &Equation,
    var: &str,
    solutions: Vec<ExprId>,
    mut is_symbolic: FIsSymbolic,
    mut verify_against_equation: FVerifyAgainstEquation,
) -> Vec<ExprId>
where
    FIsSymbolic: FnMut(&mut S, ExprId) -> bool,
    FVerifyAgainstEquation: FnMut(&mut S, &Equation, &str, ExprId) -> bool,
{
    resolve_discrete_strategy_solutions_with_state(
        state,
        solutions,
        |state, solution| is_symbolic(state, solution),
        |state, solution| verify_against_equation(state, equation, var, solution),
    )
}

/// Resolve discrete candidates against an equation and return a `SolutionSet`
/// paired with the caller-provided strategy steps.
pub fn resolve_discrete_strategy_result_against_equation_with_state<
    S,
    Step,
    FIsSymbolic,
    FVerifyAgainstEquation,
>(
    state: &mut S,
    equation: &Equation,
    var: &str,
    solutions: Vec<ExprId>,
    steps: Vec<Step>,
    is_symbolic: FIsSymbolic,
    verify_against_equation: FVerifyAgainstEquation,
) -> (SolutionSet, Vec<Step>)
where
    FIsSymbolic: FnMut(&mut S, ExprId) -> bool,
    FVerifyAgainstEquation: FnMut(&mut S, &Equation, &str, ExprId) -> bool,
{
    let valid_solutions = resolve_discrete_strategy_solutions_against_equation_with_state(
        state,
        equation,
        var,
        solutions,
        is_symbolic,
        verify_against_equation,
    );
    (SolutionSet::Discrete(valid_solutions), steps)
}

/// Classification of one strategy attempt result from the solve loop.
#[derive(Debug, Clone, PartialEq)]
pub enum StrategyAttemptResolution<S, E> {
    /// Strategy did not apply.
    Skip,
    /// Strategy solved without further discrete verification work.
    Solved {
        solution_set: SolutionSet,
        steps: Vec<S>,
    },
    /// Strategy returned discrete solutions that need caller-side verification.
    NeedsDiscreteVerification {
        solutions: Vec<ExprId>,
        steps: Vec<S>,
    },
    /// Recoverable strategy error; solve loop may continue with next strategy.
    SoftError(E),
    /// Non-recoverable strategy error.
    HardError(E),
}

/// Resolution for a full strategy-attempt sequence.
#[derive(Debug, Clone, PartialEq)]
pub enum StrategyAttemptSequenceResolution<S, E> {
    /// Sequence produced a solved result.
    Solved {
        solution_set: SolutionSet,
        steps: Vec<S>,
    },
    /// Sequence produced discrete candidates that require caller-side verification.
    NeedsDiscreteVerification {
        solutions: Vec<ExprId>,
        steps: Vec<S>,
    },
    /// Sequence produced a hard error and should abort solve.
    HardError(E),
    /// Sequence exhausted without a solved result.
    Exhausted { last_soft_error: Option<E> },
}

/// Classify one strategy attempt into skip/solved/discrete-verify/soft/hard.
///
/// This keeps strategy-loop control flow in `cas_solver_core` while leaving
/// expensive numeric verification to the caller.
pub fn classify_strategy_attempt_result<S, E, FSoft>(
    strategy_attempt: Option<Result<(SolutionSet, Vec<S>), E>>,
    should_verify_discrete: bool,
    mut is_soft_error: FSoft,
) -> StrategyAttemptResolution<S, E>
where
    FSoft: FnMut(&E) -> bool,
{
    match strategy_attempt {
        None => StrategyAttemptResolution::Skip,
        Some(Ok((SolutionSet::Discrete(solutions), steps))) if should_verify_discrete => {
            StrategyAttemptResolution::NeedsDiscreteVerification { solutions, steps }
        }
        Some(Ok((solution_set, steps))) => StrategyAttemptResolution::Solved {
            solution_set,
            steps,
        },
        Some(Err(error)) => {
            if is_soft_error(&error) {
                StrategyAttemptResolution::SoftError(error)
            } else {
                StrategyAttemptResolution::HardError(error)
            }
        }
    }
}

/// Execute a full ordered sequence of strategy attempts.
///
/// Each item is `(attempt, should_verify_discrete)` where `attempt` is the raw
/// strategy result (`None` when the strategy did not apply).
pub fn run_strategy_attempt_sequence<S, E, I, FSoft>(
    attempts: I,
    mut is_soft_error: FSoft,
) -> StrategyAttemptSequenceResolution<S, E>
where
    I: IntoIterator<Item = (Option<Result<(SolutionSet, Vec<S>), E>>, bool)>,
    FSoft: FnMut(&E) -> bool,
{
    let mut last_soft_error: Option<E> = None;

    for (attempt, should_verify_discrete) in attempts {
        match classify_strategy_attempt_result(attempt, should_verify_discrete, |err| {
            is_soft_error(err)
        }) {
            StrategyAttemptResolution::Skip => continue,
            StrategyAttemptResolution::Solved {
                solution_set,
                steps,
            } => {
                return StrategyAttemptSequenceResolution::Solved {
                    solution_set,
                    steps,
                };
            }
            StrategyAttemptResolution::NeedsDiscreteVerification { solutions, steps } => {
                return StrategyAttemptSequenceResolution::NeedsDiscreteVerification {
                    solutions,
                    steps,
                };
            }
            StrategyAttemptResolution::SoftError(error) => {
                last_soft_error = Some(error);
            }
            StrategyAttemptResolution::HardError(error) => {
                return StrategyAttemptSequenceResolution::HardError(error);
            }
        }
    }

    StrategyAttemptSequenceResolution::Exhausted { last_soft_error }
}

/// Finalize a strategy-attempt sequence into a plain solve result.
///
/// Caller provides:
/// - `resolve_discrete`: how to post-process discrete candidates that require
///   verification/filtering.
/// - `no_solution_error`: fallback error when sequence is exhausted without a
///   soft error.
pub fn finalize_strategy_attempt_sequence_with<S, E, FResolveDiscrete>(
    resolution: StrategyAttemptSequenceResolution<S, E>,
    mut resolve_discrete: FResolveDiscrete,
    no_solution_error: E,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FResolveDiscrete: FnMut(Vec<ExprId>, Vec<S>) -> (SolutionSet, Vec<S>),
{
    match resolution {
        StrategyAttemptSequenceResolution::Solved {
            solution_set,
            steps,
        } => Ok((solution_set, steps)),
        StrategyAttemptSequenceResolution::NeedsDiscreteVerification { solutions, steps } => {
            Ok(resolve_discrete(solutions, steps))
        }
        StrategyAttemptSequenceResolution::HardError(error) => Err(error),
        StrategyAttemptSequenceResolution::Exhausted { last_soft_error } => {
            Err(last_soft_error.unwrap_or(no_solution_error))
        }
    }
}

/// Execute strategy attempts and finalize into a plain solve result in one call.
pub fn execute_strategy_attempt_sequence_with_resolution<S, E, I, FSoft, FResolveDiscrete>(
    attempts: I,
    is_soft_error: FSoft,
    resolve_discrete: FResolveDiscrete,
    no_solution_error: E,
) -> Result<(SolutionSet, Vec<S>), E>
where
    I: IntoIterator<Item = (Option<Result<(SolutionSet, Vec<S>), E>>, bool)>,
    FSoft: FnMut(&E) -> bool,
    FResolveDiscrete: FnMut(Vec<ExprId>, Vec<S>) -> (SolutionSet, Vec<S>),
{
    finalize_strategy_attempt_sequence_with(
        run_strategy_attempt_sequence(attempts, is_soft_error),
        resolve_discrete,
        no_solution_error,
    )
}

/// Execute strategy sequence by evaluating one strategy at a time against
/// caller-provided mutable state.
///
/// This avoids pre-collecting attempt vectors at call sites and keeps
/// strategy-loop orchestration inside `cas_solver_core`.
pub fn run_strategies_with_state<SState, Strategy, S, E, I, FEvaluate, FSoft>(
    state: &mut SState,
    strategies: I,
    mut evaluate_strategy: FEvaluate,
    mut is_soft_error: FSoft,
) -> StrategyAttemptSequenceResolution<S, E>
where
    I: IntoIterator<Item = Strategy>,
    FEvaluate: FnMut(&mut SState, Strategy) -> (Option<Result<(SolutionSet, Vec<S>), E>>, bool),
    FSoft: FnMut(&E) -> bool,
{
    let mut last_soft_error: Option<E> = None;

    for strategy in strategies {
        let (attempt, should_verify_discrete) = evaluate_strategy(state, strategy);
        match classify_strategy_attempt_result(attempt, should_verify_discrete, |err| {
            is_soft_error(err)
        }) {
            StrategyAttemptResolution::Skip => continue,
            StrategyAttemptResolution::Solved {
                solution_set,
                steps,
            } => {
                return StrategyAttemptSequenceResolution::Solved {
                    solution_set,
                    steps,
                };
            }
            StrategyAttemptResolution::NeedsDiscreteVerification { solutions, steps } => {
                return StrategyAttemptSequenceResolution::NeedsDiscreteVerification {
                    solutions,
                    steps,
                };
            }
            StrategyAttemptResolution::SoftError(error) => {
                last_soft_error = Some(error);
            }
            StrategyAttemptResolution::HardError(error) => {
                return StrategyAttemptSequenceResolution::HardError(error);
            }
        }
    }

    StrategyAttemptSequenceResolution::Exhausted { last_soft_error }
}

/// Execute + finalize a stateful strategy sequence in one call.
pub fn execute_strategies_with_state_and_resolution<
    SState,
    Strategy,
    S,
    E,
    I,
    FEvaluate,
    FSoft,
    FResolveDiscrete,
>(
    state: &mut SState,
    strategies: I,
    evaluate_strategy: FEvaluate,
    is_soft_error: FSoft,
    mut resolve_discrete: FResolveDiscrete,
    no_solution_error: E,
) -> Result<(SolutionSet, Vec<S>), E>
where
    I: IntoIterator<Item = Strategy>,
    FEvaluate: FnMut(&mut SState, Strategy) -> (Option<Result<(SolutionSet, Vec<S>), E>>, bool),
    FSoft: FnMut(&E) -> bool,
    FResolveDiscrete: FnMut(&mut SState, Vec<ExprId>, Vec<S>) -> (SolutionSet, Vec<S>),
{
    match run_strategies_with_state(state, strategies, evaluate_strategy, is_soft_error) {
        StrategyAttemptSequenceResolution::Solved {
            solution_set,
            steps,
        } => Ok((solution_set, steps)),
        StrategyAttemptSequenceResolution::NeedsDiscreteVerification { solutions, steps } => {
            Ok(resolve_discrete(state, solutions, steps))
        }
        StrategyAttemptSequenceResolution::HardError(error) => Err(error),
        StrategyAttemptSequenceResolution::Exhausted { last_soft_error } => {
            Err(last_soft_error.unwrap_or(no_solution_error))
        }
    }
}

/// Execute the prepared-equation pipeline after pre-normalization:
/// 1) if residual no longer contains `var`, resolve directly,
/// 2) otherwise enter cycle guard,
/// 3) execute strategy sequence and finalize.
#[allow(clippy::too_many_arguments)]
pub fn execute_prepared_equation_strategy_pipeline_with_state<
    SState,
    Strategy,
    S,
    E,
    Guard,
    I,
    FContainsVar,
    FResolveVarEliminated,
    FEnterCycle,
    FEvaluateStrategy,
    FSoftError,
    FResolveDiscrete,
>(
    state: &mut SState,
    equation: &Equation,
    residual: ExprId,
    var: &str,
    strategies: I,
    mut contains_var: FContainsVar,
    mut resolve_var_eliminated: FResolveVarEliminated,
    mut enter_cycle: FEnterCycle,
    evaluate_strategy: FEvaluateStrategy,
    is_soft_error: FSoftError,
    resolve_discrete: FResolveDiscrete,
    no_solution_error: E,
) -> Result<(SolutionSet, Vec<S>), E>
where
    I: IntoIterator<Item = Strategy>,
    FContainsVar: FnMut(&mut SState, ExprId, &str) -> bool,
    FResolveVarEliminated: FnMut(&mut SState, ExprId, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FEnterCycle: FnMut(&mut SState, &Equation, &str) -> Result<Guard, E>,
    FEvaluateStrategy:
        FnMut(&mut SState, Strategy) -> (Option<Result<(SolutionSet, Vec<S>), E>>, bool),
    FSoftError: FnMut(&E) -> bool,
    FResolveDiscrete: FnMut(&mut SState, Vec<ExprId>, Vec<S>) -> (SolutionSet, Vec<S>),
{
    if !contains_var(state, residual, var) {
        return resolve_var_eliminated(state, residual, var);
    }

    let _cycle_guard = enter_cycle(state, equation, var)?;
    execute_strategies_with_state_and_resolution(
        state,
        strategies,
        evaluate_strategy,
        is_soft_error,
        resolve_discrete,
        no_solution_error,
    )
}

/// Decide whether an error should be treated as "soft" (strategy-local dead end)
/// based on optional detail-message channels.
///
/// Current policy marks as soft when:
/// - isolation details mention "variable appears on both sides", or
/// - solver details mention "Cycle detected".
pub fn is_soft_strategy_error_from_message_parts(
    isolation_detail: Option<&str>,
    solver_detail: Option<&str>,
) -> bool {
    isolation_detail.is_some_and(|detail| detail.contains("variable appears on both sides"))
        || solver_detail.is_some_and(|detail| detail.contains("Cycle detected"))
}

/// Generic adapter for soft-error policy over arbitrary error types.
///
/// Callers provide channel extractors for:
/// - isolation detail text
/// - solver detail text
///
/// so core policy can remain centralized without depending on runtime error enums.
pub fn is_soft_strategy_error_with<E, FIsolation, FSolver>(
    error: &E,
    mut isolation_detail: FIsolation,
    mut solver_detail: FSolver,
) -> bool
where
    FIsolation: for<'a> FnMut(&'a E) -> Option<&'a str>,
    FSolver: for<'a> FnMut(&'a E) -> Option<&'a str>,
{
    is_soft_strategy_error_from_message_parts(isolation_detail(error), solver_detail(error))
}

/// Decide whether a rewritten residual should replace the current one.
///
/// Accept when:
/// - The target variable was eliminated, or
/// - Tree size was reduced by more than 25% (avoids cosmetic rewrites).
pub fn should_accept_rewritten_residual(
    var_eliminated: bool,
    old_nodes: usize,
    new_nodes: usize,
) -> bool {
    var_eliminated || (old_nodes > 4 && new_nodes * 4 < old_nodes * 3)
}

/// Variable presence classification across equation sides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EquationVarPresence {
    None,
    LhsOnly,
    RhsOnly,
    BothSides,
}

/// Ensure equation variable-presence check passed, otherwise return caller error.
pub fn ensure_equation_has_variable_or_error<E, FError>(
    presence: EquationVarPresence,
    missing_var_error: FError,
) -> Result<(), E>
where
    FError: FnOnce() -> E,
{
    if matches!(presence, EquationVarPresence::None) {
        Err(missing_var_error())
    } else {
        Ok(())
    }
}

/// Ensure recursion depth stays within a caller-provided maximum.
pub fn ensure_recursion_depth_within_limit_or_error<E, FError>(
    current_depth: usize,
    max_depth: usize,
    depth_error: FError,
) -> Result<(), E>
where
    FError: FnOnce() -> E,
{
    if current_depth > max_depth {
        Err(depth_error())
    } else {
        Ok(())
    }
}

/// Validate solve-entry guards for one equation:
/// - recursion depth bound
/// - target variable appears in at least one side
pub fn ensure_solve_entry_for_equation_or_error<E, FDepthError, FMissingVarError>(
    ctx: &Context,
    equation: &Equation,
    var: &str,
    current_depth: usize,
    max_depth: usize,
    depth_error: FDepthError,
    missing_var_error: FMissingVarError,
) -> Result<(), E>
where
    FDepthError: FnOnce() -> E,
    FMissingVarError: FnOnce() -> E,
{
    ensure_recursion_depth_within_limit_or_error(current_depth, max_depth, depth_error)?;
    ensure_equation_has_variable_or_error(
        classify_equation_var_presence(ctx, equation, var),
        missing_var_error,
    )?;
    Ok(())
}

/// Enter equation-fingerprint cycle guard or return caller-provided cycle error.
pub fn try_enter_equation_cycle_guard_with_error<E, FError>(
    ctx: &Context,
    equation: &Equation,
    var: &str,
    cycle_error: FError,
) -> Result<crate::cycle_guard::CycleGuard, E>
where
    FError: FnOnce() -> E,
{
    crate::cycle_guard::try_enter_equation_fingerprint(ctx, equation.lhs, equation.rhs, var)
        .ok_or_else(cycle_error)
}

/// Collect required conditions for an equation from side-inference and
/// equation-derived propagation hooks, preserving insertion order while
/// deduplicating by value.
pub fn collect_required_conditions_for_equation_with<C, V, FInferSide, FDeriveEq>(
    lhs: ExprId,
    rhs: ExprId,
    value_domain: V,
    mut infer_side_conditions: FInferSide,
    mut derive_equation_conditions: FDeriveEq,
) -> Vec<C>
where
    C: Eq + Hash + Clone,
    V: Copy,
    FInferSide: FnMut(ExprId, V) -> Vec<C>,
    FDeriveEq: FnMut(ExprId, ExprId, &[C], V) -> Vec<C>,
{
    let mut seen = HashSet::new();
    let mut ordered = Vec::new();

    fn push_unique<C: Eq + Hash + Clone>(cond: C, seen: &mut HashSet<C>, ordered: &mut Vec<C>) {
        if seen.insert(cond.clone()) {
            ordered.push(cond);
        }
    }

    for cond in infer_side_conditions(lhs, value_domain) {
        push_unique(cond, &mut seen, &mut ordered);
    }
    for cond in infer_side_conditions(rhs, value_domain) {
        push_unique(cond, &mut seen, &mut ordered);
    }
    for cond in derive_equation_conditions(lhs, rhs, &ordered, value_domain) {
        push_unique(cond, &mut seen, &mut ordered);
    }

    ordered
}

/// Build a domain view from existing requirements and derive additional
/// equation-level requirements from it.
pub fn derive_equation_conditions_from_existing_with<C, Domain, V, FNewDomain, FInsert, FDerive>(
    lhs: ExprId,
    rhs: ExprId,
    existing: &[C],
    value_domain: V,
    mut new_domain: FNewDomain,
    mut insert_condition: FInsert,
    mut derive_from_domain: FDerive,
) -> Vec<C>
where
    C: Clone,
    V: Copy,
    FNewDomain: FnMut() -> Domain,
    FInsert: FnMut(&mut Domain, C),
    FDerive: FnMut(ExprId, ExprId, &Domain, V) -> Vec<C>,
{
    let mut domain = new_domain();
    for cond in existing.iter().cloned() {
        insert_condition(&mut domain, cond);
    }
    derive_from_domain(lhs, rhs, &domain, value_domain)
}

/// Preflight outputs needed before strategy dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EquationPreflight<C> {
    pub domain_exclusions: Vec<ExprId>,
    pub required_conditions: Vec<C>,
}

/// Analyze one equation before strategy dispatch:
/// - collect denominator exclusions containing `var`,
/// - collect required conditions inferred/derived from equation semantics.
pub fn analyze_equation_preflight_with<C, V, FInferSide, FDeriveEq>(
    ctx: &Context,
    equation: &Equation,
    var: &str,
    value_domain: V,
    infer_side_conditions: FInferSide,
    derive_equation_conditions: FDeriveEq,
) -> EquationPreflight<C>
where
    C: Eq + Hash + Clone,
    V: Copy,
    FInferSide: FnMut(ExprId, V) -> Vec<C>,
    FDeriveEq: FnMut(ExprId, ExprId, &[C], V) -> Vec<C>,
{
    let domain_exclusions =
        collect_unique_denominators_with_var(ctx, equation.lhs, equation.rhs, var);
    let required_conditions = collect_required_conditions_for_equation_with(
        equation.lhs,
        equation.rhs,
        value_domain,
        infer_side_conditions,
        derive_equation_conditions,
    );

    EquationPreflight {
        domain_exclusions,
        required_conditions,
    }
}

/// Apply required conditions to two sinks:
/// - one sink receives borrowed conditions (e.g. domain set insertion),
/// - one sink receives owned conditions (e.g. shared accumulator event).
pub fn apply_required_conditions_with<C, I, FInsert, FNote>(
    conditions: I,
    mut insert_condition: FInsert,
    mut note_condition: FNote,
) where
    C: Clone,
    I: IntoIterator<Item = C>,
    FInsert: FnMut(&C),
    FNote: FnMut(C),
{
    for cond in conditions {
        insert_condition(&cond);
        note_condition(cond);
    }
}

/// Preflight output with a child solve context already forked and hydrated
/// with required conditions.
#[derive(Debug, Clone)]
pub struct PreflightContext<Ctx> {
    pub domain_exclusions: Vec<ExprId>,
    pub ctx: Ctx,
}

/// Analyze equation preflight data and fork a child context with required
/// conditions applied into both the domain env and the shared required sink.
#[allow(clippy::too_many_arguments)]
pub fn analyze_equation_preflight_and_fork_context_with<
    C,
    V,
    DomainEnv,
    Assumption,
    Scope,
    FInferSide,
    FDeriveEq,
    FInsertRequired,
>(
    ctx: &Context,
    equation: &Equation,
    var: &str,
    value_domain: V,
    parent_ctx: &crate::solve_context::SolveContext<DomainEnv, C, Assumption, Scope>,
    infer_side_conditions: FInferSide,
    derive_equation_conditions: FDeriveEq,
    mut domain_env: DomainEnv,
    mut insert_required_into_domain_env: FInsertRequired,
) -> PreflightContext<crate::solve_context::SolveContext<DomainEnv, C, Assumption, Scope>>
where
    C: Eq + Hash + Clone,
    V: Copy,
    Assumption: Clone,
    Scope: Clone + PartialEq,
    FInferSide: FnMut(ExprId, V) -> Vec<C>,
    FDeriveEq: FnMut(ExprId, ExprId, &[C], V) -> Vec<C>,
    FInsertRequired: FnMut(&mut DomainEnv, &C),
{
    let preflight = analyze_equation_preflight_with(
        ctx,
        equation,
        var,
        value_domain,
        infer_side_conditions,
        derive_equation_conditions,
    );

    apply_required_conditions_with(
        preflight.required_conditions,
        |cond| insert_required_into_domain_env(&mut domain_env, cond),
        |cond| parent_ctx.note_required_condition(cond),
    );

    PreflightContext {
        domain_exclusions: preflight.domain_exclusions,
        ctx: parent_ctx.fork_with_domain_env_next_depth(domain_env),
    }
}

/// Classify where `var` appears in an equation.
pub fn classify_equation_var_presence(
    ctx: &Context,
    equation: &Equation,
    var: &str,
) -> EquationVarPresence {
    let lhs_has = super::isolation_utils::contains_var(ctx, equation.lhs, var);
    let rhs_has = super::isolation_utils::contains_var(ctx, equation.rhs, var);
    match (lhs_has, rhs_has) {
        (false, false) => EquationVarPresence::None,
        (true, false) => EquationVarPresence::LhsOnly,
        (false, true) => EquationVarPresence::RhsOnly,
        (true, true) => EquationVarPresence::BothSides,
    }
}

/// Simplify only equation sides that contain `var` and recompose `a^x / b^x` when possible
/// using caller-provided hooks.
pub fn simplify_equation_sides_for_presence_with<FSimplify, FRecompose>(
    eq: &Equation,
    lhs_has_var: bool,
    rhs_has_var: bool,
    mut simplify_for_solve: FSimplify,
    mut try_recompose_pow_quotient: FRecompose,
) -> Equation
where
    FSimplify: FnMut(ExprId) -> ExprId,
    FRecompose: FnMut(ExprId) -> Option<ExprId>,
{
    let mut simplified_eq = eq.clone();

    if lhs_has_var {
        let sim_lhs = simplify_for_solve(eq.lhs);
        simplified_eq.lhs = sim_lhs;
        if let Some(recomposed) = try_recompose_pow_quotient(sim_lhs) {
            simplified_eq.lhs = recomposed;
        }
    }

    if rhs_has_var {
        let sim_rhs = simplify_for_solve(eq.rhs);
        simplified_eq.rhs = sim_rhs;
        if let Some(recomposed) = try_recompose_pow_quotient(sim_rhs) {
            simplified_eq.rhs = recomposed;
        }
    }

    simplified_eq
}

/// Stateful variant of [`simplify_equation_sides_for_presence_with`].
///
/// This form lets callers avoid interior mutability when both simplify and
/// recompose hooks need shared mutable state.
pub fn simplify_equation_sides_for_presence_with_state<S, FSimplify, FRecompose>(
    state: &mut S,
    eq: &Equation,
    lhs_has_var: bool,
    rhs_has_var: bool,
    mut simplify_for_solve: FSimplify,
    mut try_recompose_pow_quotient: FRecompose,
) -> Equation
where
    FSimplify: FnMut(&mut S, ExprId) -> ExprId,
    FRecompose: FnMut(&mut S, ExprId) -> Option<ExprId>,
{
    let mut simplified_eq = eq.clone();

    if lhs_has_var {
        let sim_lhs = simplify_for_solve(state, eq.lhs);
        simplified_eq.lhs = sim_lhs;
        if let Some(recomposed) = try_recompose_pow_quotient(state, sim_lhs) {
            simplified_eq.lhs = recomposed;
        }
    }

    if rhs_has_var {
        let sim_rhs = simplify_for_solve(state, eq.rhs);
        simplified_eq.rhs = sim_rhs;
        if let Some(recomposed) = try_recompose_pow_quotient(state, sim_rhs) {
            simplified_eq.rhs = recomposed;
        }
    }

    simplified_eq
}

/// Simplify only equation sides that contain `var` and recompose `a^x / b^x` when possible
/// using caller-provided hooks.
pub fn simplify_equation_sides_for_var_with<FContains, FSimplify, FRecompose>(
    eq: &Equation,
    var: &str,
    mut contains_var: FContains,
    simplify_for_solve: FSimplify,
    try_recompose_pow_quotient: FRecompose,
) -> Equation
where
    FContains: FnMut(ExprId, &str) -> bool,
    FSimplify: FnMut(ExprId) -> ExprId,
    FRecompose: FnMut(ExprId) -> Option<ExprId>,
{
    let lhs_has_var = contains_var(eq.lhs, var);
    let rhs_has_var = contains_var(eq.rhs, var);
    simplify_equation_sides_for_presence_with(
        eq,
        lhs_has_var,
        rhs_has_var,
        simplify_for_solve,
        try_recompose_pow_quotient,
    )
}

/// Apply ordered equation-side rewrites and re-simplify each changed side.
///
/// The sequence is:
/// 1) structural rewrite attempt
/// 2) semantic rewrite attempt on the latest side pair
///
/// Each successful rewrite updates both sides and runs `simplify_for_solve`
/// on each changed side before the next phase.
pub fn apply_equation_pair_rewrite_sequence_with<FStructural, FSemantic, FSimplify>(
    lhs: ExprId,
    rhs: ExprId,
    mut structural_rewrite: FStructural,
    mut semantic_rewrite: FSemantic,
    mut simplify_for_solve: FSimplify,
) -> (ExprId, ExprId)
where
    FStructural: FnMut(ExprId, ExprId) -> Option<(ExprId, ExprId)>,
    FSemantic: FnMut(ExprId, ExprId) -> Option<(ExprId, ExprId)>,
    FSimplify: FnMut(ExprId) -> ExprId,
{
    let mut current_lhs = lhs;
    let mut current_rhs = rhs;

    if let Some((new_lhs, new_rhs)) = structural_rewrite(current_lhs, current_rhs) {
        current_lhs = simplify_for_solve(new_lhs);
        current_rhs = simplify_for_solve(new_rhs);
    }

    if let Some((new_lhs, new_rhs)) = semantic_rewrite(current_lhs, current_rhs) {
        current_lhs = simplify_for_solve(new_lhs);
        current_rhs = simplify_for_solve(new_rhs);
    }

    (current_lhs, current_rhs)
}

/// Stateful variant of [`apply_equation_pair_rewrite_sequence_with`].
pub fn apply_equation_pair_rewrite_sequence_with_state<S, FStructural, FSemantic, FSimplify>(
    state: &mut S,
    lhs: ExprId,
    rhs: ExprId,
    mut structural_rewrite: FStructural,
    mut semantic_rewrite: FSemantic,
    mut simplify_for_solve: FSimplify,
) -> (ExprId, ExprId)
where
    FStructural: FnMut(&mut S, ExprId, ExprId) -> Option<(ExprId, ExprId)>,
    FSemantic: FnMut(&mut S, ExprId, ExprId) -> Option<(ExprId, ExprId)>,
    FSimplify: FnMut(&mut S, ExprId) -> ExprId,
{
    let mut current_lhs = lhs;
    let mut current_rhs = rhs;

    if let Some((new_lhs, new_rhs)) = structural_rewrite(state, current_lhs, current_rhs) {
        current_lhs = simplify_for_solve(state, new_lhs);
        current_rhs = simplify_for_solve(state, new_rhs);
    }

    if let Some((new_lhs, new_rhs)) = semantic_rewrite(state, current_lhs, current_rhs) {
        current_lhs = simplify_for_solve(state, new_lhs);
        current_rhs = simplify_for_solve(state, new_rhs);
    }

    (current_lhs, current_rhs)
}

/// Return the candidate residual when the rewrite is meaningfully better.
pub fn accept_residual_rewrite_candidate(
    ctx: &Context,
    current: ExprId,
    candidate: ExprId,
    var: &str,
) -> Option<ExprId> {
    let old_nodes = cas_ast::traversal::count_all_nodes(ctx, current);
    let new_nodes = cas_ast::traversal::count_all_nodes(ctx, candidate);
    let var_eliminated = !super::isolation_utils::contains_var(ctx, candidate, var);
    if should_accept_rewritten_residual(var_eliminated, old_nodes, new_nodes) {
        Some(candidate)
    } else {
        None
    }
}

/// Normalize a residual expression by applying the two engine-level fallback rewrites:
/// 1) algebraic expand + simplify
/// 2) trig expand mode
///
/// A rewrite is accepted only if it eliminates `var` or reduces tree size significantly.
pub fn normalize_variable_residual_with<
    FContains,
    FExpandAlgebraic,
    FSimplifyForSolve,
    FExpandTrig,
    FAcceptCandidate,
>(
    residual: ExprId,
    var: &str,
    mut contains_var: FContains,
    mut expand_algebraic: FExpandAlgebraic,
    mut simplify_for_solve: FSimplifyForSolve,
    mut expand_trig: FExpandTrig,
    mut accept_candidate: FAcceptCandidate,
) -> ExprId
where
    FContains: FnMut(ExprId, &str) -> bool,
    FExpandAlgebraic: FnMut(ExprId) -> ExprId,
    FSimplifyForSolve: FnMut(ExprId) -> ExprId,
    FExpandTrig: FnMut(ExprId) -> ExprId,
    FAcceptCandidate: FnMut(ExprId, ExprId, &str) -> Option<ExprId>,
{
    let mut current = residual;

    if contains_var(current, var) {
        let expanded = expand_algebraic(current);
        let re_simplified = simplify_for_solve(expanded);
        if let Some(accepted) = accept_candidate(current, re_simplified, var) {
            current = accepted;
        }
    }

    if contains_var(current, var) {
        let trig_expanded = expand_trig(current);
        if let Some(accepted) = accept_candidate(current, trig_expanded, var) {
            current = accepted;
        }
    }

    current
}

/// Stateful variant of [`normalize_variable_residual_with`].
#[allow(clippy::too_many_arguments)]
pub fn normalize_variable_residual_with_state<
    S,
    FContains,
    FExpandAlgebraic,
    FSimplifyForSolve,
    FExpandTrig,
    FAcceptCandidate,
>(
    state: &mut S,
    residual: ExprId,
    var: &str,
    mut contains_var: FContains,
    mut expand_algebraic: FExpandAlgebraic,
    mut simplify_for_solve: FSimplifyForSolve,
    mut expand_trig: FExpandTrig,
    mut accept_candidate: FAcceptCandidate,
) -> ExprId
where
    FContains: FnMut(&mut S, ExprId, &str) -> bool,
    FExpandAlgebraic: FnMut(&mut S, ExprId) -> ExprId,
    FSimplifyForSolve: FnMut(&mut S, ExprId) -> ExprId,
    FExpandTrig: FnMut(&mut S, ExprId) -> ExprId,
    FAcceptCandidate: FnMut(&mut S, ExprId, ExprId, &str) -> Option<ExprId>,
{
    let mut current = residual;

    if contains_var(state, current, var) {
        let expanded = expand_algebraic(state, current);
        let re_simplified = simplify_for_solve(state, expanded);
        if let Some(accepted) = accept_candidate(state, current, re_simplified, var) {
            current = accepted;
        }
    }

    if contains_var(state, current, var) {
        let trig_expanded = expand_trig(state, current);
        if let Some(accepted) = accept_candidate(state, current, trig_expanded, var) {
            current = accepted;
        }
    }

    current
}

/// Prepared equation and residual after the default pre-strategy normalization
/// pipeline used by solve loops.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedEquationResidual {
    pub equation: Equation,
    pub residual: ExprId,
}

/// Execute the default pre-strategy equation normalization pipeline:
/// 1) simplify sides that contain the target variable (+ optional recomposition),
/// 2) apply structural + semantic pair rewrites,
/// 3) build/simplify residual `lhs - rhs`,
/// 4) normalize residual with algebraic/trig fallback rewrites,
/// 5) if residual changed, rewrite equation as `residual = 0`.
#[allow(clippy::too_many_arguments)]
pub fn prepare_equation_for_strategy_with_state<
    S,
    FContainsVar,
    FSimplifyForSolve,
    FRecomposePowQuotient,
    FStructuralRewrite,
    FSemanticRewrite,
    FBuildDifference,
    FExpandAlgebraic,
    FExpandTrig,
    FAcceptResidualCandidate,
    FZeroExpr,
>(
    state: &mut S,
    equation: &Equation,
    var: &str,
    mut contains_var: FContainsVar,
    simplify_for_solve: FSimplifyForSolve,
    mut try_recompose_pow_quotient: FRecomposePowQuotient,
    mut structural_rewrite: FStructuralRewrite,
    mut semantic_rewrite: FSemanticRewrite,
    mut build_difference: FBuildDifference,
    mut expand_algebraic: FExpandAlgebraic,
    mut expand_trig: FExpandTrig,
    mut accept_residual_candidate: FAcceptResidualCandidate,
    mut zero_expr: FZeroExpr,
) -> PreparedEquationResidual
where
    FContainsVar: FnMut(&mut S, ExprId, &str) -> bool,
    FSimplifyForSolve: FnMut(&mut S, ExprId) -> ExprId,
    FRecomposePowQuotient: FnMut(&mut S, ExprId) -> Option<ExprId>,
    FStructuralRewrite: FnMut(&mut S, ExprId, ExprId) -> Option<(ExprId, ExprId)>,
    FSemanticRewrite: FnMut(&mut S, ExprId, ExprId) -> Option<(ExprId, ExprId)>,
    FBuildDifference: FnMut(&mut S, ExprId, ExprId) -> ExprId,
    FExpandAlgebraic: FnMut(&mut S, ExprId) -> ExprId,
    FExpandTrig: FnMut(&mut S, ExprId) -> ExprId,
    FAcceptResidualCandidate: FnMut(&mut S, ExprId, ExprId, &str) -> Option<ExprId>,
    FZeroExpr: FnMut(&mut S) -> ExprId,
{
    let lhs_has_var = contains_var(state, equation.lhs, var);
    let rhs_has_var = contains_var(state, equation.rhs, var);
    let simplify_for_solve = std::cell::RefCell::new(simplify_for_solve);

    let mut simplified_eq = simplify_equation_sides_for_presence_with_state(
        state,
        equation,
        lhs_has_var,
        rhs_has_var,
        |state, expr| (simplify_for_solve.borrow_mut())(state, expr),
        |state, expr| try_recompose_pow_quotient(state, expr),
    );

    let (lhs, rhs) = apply_equation_pair_rewrite_sequence_with_state(
        state,
        simplified_eq.lhs,
        simplified_eq.rhs,
        |state, lhs, rhs| structural_rewrite(state, lhs, rhs),
        |state, lhs, rhs| semantic_rewrite(state, lhs, rhs),
        |state, expr| (simplify_for_solve.borrow_mut())(state, expr),
    );
    simplified_eq.lhs = lhs;
    simplified_eq.rhs = rhs;

    let difference = build_difference(state, simplified_eq.lhs, simplified_eq.rhs);
    let mut residual = (simplify_for_solve.borrow_mut())(state, difference);

    let normalized_residual = normalize_variable_residual_with_state(
        state,
        residual,
        var,
        |state, expr, var_name| contains_var(state, expr, var_name),
        |state, expr| expand_algebraic(state, expr),
        |state, expr| (simplify_for_solve.borrow_mut())(state, expr),
        |state, expr| expand_trig(state, expr),
        |state, current, candidate, var_name| {
            accept_residual_candidate(state, current, candidate, var_name)
        },
    );

    if normalized_residual != residual {
        residual = normalized_residual;
        simplified_eq.lhs = residual;
        simplified_eq.rhs = zero_expr(state);
    }

    PreparedEquationResidual {
        equation: simplified_eq,
        residual,
    }
}

/// Extract all denominators that contain the target variable.
pub fn extract_denominators_with_var(ctx: &Context, expr: ExprId, var: &str) -> Vec<ExprId> {
    let mut denoms_set: HashSet<ExprId> = HashSet::new();
    collect_denominators_into_set(ctx, expr, var, &mut denoms_set);
    denoms_set.into_iter().collect()
}

/// Collect unique denominator expressions containing `var` across equation sides.
pub fn collect_unique_denominators_with_var(
    ctx: &Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
) -> Vec<ExprId> {
    let mut denoms_set: HashSet<ExprId> = HashSet::new();
    denoms_set.extend(extract_denominators_with_var(ctx, lhs, var));
    denoms_set.extend(extract_denominators_with_var(ctx, rhs, var));
    denoms_set.into_iter().collect()
}

fn collect_denominators_into_set(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    denoms: &mut HashSet<ExprId>,
) {
    match ctx.get(expr) {
        Expr::Div(num, denom) => {
            if super::isolation_utils::contains_var(ctx, *denom, var) {
                denoms.insert(*denom);
            }
            collect_denominators_into_set(ctx, *num, var, denoms);
            collect_denominators_into_set(ctx, *denom, var, denoms);
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Pow(l, r) => {
            collect_denominators_into_set(ctx, *l, var, denoms);
            collect_denominators_into_set(ctx, *r, var, denoms);
        }
        Expr::Neg(e) | Expr::Hold(e) => {
            collect_denominators_into_set(ctx, *e, var, denoms);
        }
        Expr::Function(_, args) => {
            for arg in args {
                collect_denominators_into_set(ctx, *arg, var, denoms);
            }
        }
        Expr::Matrix { data, .. } => {
            for elem in data {
                collect_denominators_into_set(ctx, *elem, var, denoms);
            }
        }
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
    }
}

/// Apply non-zero exclusion guards to a solution set.
pub fn apply_nonzero_exclusion_guards(
    solution_set: SolutionSet,
    exclusions: &[ExprId],
) -> SolutionSet {
    if exclusions.is_empty() {
        return solution_set;
    }

    let mut guard = ConditionSet::empty();
    for &denom in exclusions {
        guard.push(ConditionPredicate::NonZero(denom));
    }

    let cases = vec![
        Case::new(guard, solution_set),
        Case::new(ConditionSet::empty(), SolutionSet::Empty),
    ];
    SolutionSet::Conditional(cases).simplify()
}

/// Apply non-zero exclusion guards only when exclusions exist.
pub fn apply_nonzero_exclusion_guards_if_any(
    solution_set: SolutionSet,
    exclusions: &[ExprId],
) -> SolutionSet {
    if exclusions.is_empty() {
        solution_set
    } else {
        apply_nonzero_exclusion_guards(solution_set, exclusions)
    }
}

/// Resolve a variable-eliminated residual (`diff == 0`) to a final solution set,
/// applying denominator non-zero guards when needed.
pub fn resolve_var_eliminated_residual_with_exclusions<S, FRender, FMapStep>(
    ctx: &mut Context,
    diff_simplified: ExprId,
    var: &str,
    include_item: bool,
    domain_exclusions: &[ExprId],
    render_expr: FRender,
    map_step: FMapStep,
) -> (SolutionSet, Vec<S>)
where
    FRender: Fn(&Context, ExprId) -> String,
    FMapStep: FnMut(String, Equation) -> S,
{
    let reduced_outcome = solve_var_eliminated_outcome_pipeline_with(
        ctx,
        diff_simplified,
        var,
        include_item,
        render_expr,
        map_step,
    );

    match reduced_outcome {
        VarEliminatedOutcomePipelineSolved::IdentityAllReals => (SolutionSet::AllReals, vec![]),
        VarEliminatedOutcomePipelineSolved::ContradictionEmpty => (SolutionSet::Empty, vec![]),
        VarEliminatedOutcomePipelineSolved::ConstraintAllReals { steps } => (
            apply_nonzero_exclusion_guards_if_any(SolutionSet::AllReals, domain_exclusions),
            steps,
        ),
    }
}

/// Lift guard application over solved `(SolutionSet, payload)` results.
pub fn guard_solved_result_with_exclusions<T, E>(
    result: Result<(SolutionSet, T), E>,
    exclusions: &[ExprId],
) -> Result<(SolutionSet, T), E> {
    result.map(|(solution_set, payload)| {
        (
            apply_nonzero_exclusion_guards_if_any(solution_set, exclusions),
            payload,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::RelOp;
    use std::cell::{Cell, RefCell};

    #[test]
    fn symbolic_number_vs_variable() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let x = ctx.var("x");
        assert!(!is_symbolic_expr(&ctx, two));
        assert!(is_symbolic_expr(&ctx, x));
    }

    #[test]
    fn extract_denominators_basic() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let div = ctx.add(Expr::Div(y, x));
        let denoms = extract_denominators_with_var(&ctx, div, "x");
        assert_eq!(denoms.len(), 1);
        assert_eq!(denoms[0], x);
    }

    #[test]
    fn apply_guards_builds_conditional() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sol = SolutionSet::Discrete(vec![ctx.num(1)]);
        let guarded = apply_nonzero_exclusion_guards(sol, &[x]);
        assert!(matches!(guarded, SolutionSet::Conditional(_)));
    }

    #[test]
    fn apply_guards_if_any_keeps_solution_unchanged_for_empty_exclusions() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let sol = SolutionSet::Discrete(vec![one]);
        let guarded = apply_nonzero_exclusion_guards_if_any(sol.clone(), &[]);
        assert_eq!(guarded, sol);
    }

    #[test]
    fn guard_solved_result_with_exclusions_wraps_solution_set() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let result: Result<(SolutionSet, usize), ()> =
            Ok((SolutionSet::Discrete(vec![ctx.num(2)]), 7));
        let guarded = guard_solved_result_with_exclusions(result, &[x])
            .expect("guarding should preserve successful result");
        assert!(matches!(guarded.0, SolutionSet::Conditional(_)));
        assert_eq!(guarded.1, 7);
    }

    #[test]
    fn resolve_var_eliminated_residual_with_exclusions_returns_identity() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let solved = resolve_var_eliminated_residual_with_exclusions(
            &mut ctx,
            zero,
            "x",
            false,
            &[],
            |_ctx, _expr| String::new(),
            |_description, _equation| String::new(),
        );

        assert_eq!(solved.0, SolutionSet::AllReals);
        assert!(solved.1.is_empty());
    }

    #[test]
    fn resolve_var_eliminated_residual_with_exclusions_applies_nonzero_guards_for_constraints() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let zero = ctx.num(0);
        let diff = ctx.add(Expr::Sub(y, zero));
        let solved = resolve_var_eliminated_residual_with_exclusions(
            &mut ctx,
            diff,
            "x",
            true,
            &[x],
            |_ctx, _expr| String::new(),
            |_description, _equation| String::new(),
        );

        assert!(matches!(solved.0, SolutionSet::Conditional(_)));
    }

    #[test]
    fn collect_unique_denominators_deduplicates_between_sides() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let lhs = ctx.add(Expr::Div(one, x));
        let one2 = ctx.num(1);
        let rhs = ctx.add(Expr::Div(one2, x));
        let denoms = collect_unique_denominators_with_var(&ctx, lhs, rhs, "x");
        assert_eq!(denoms.len(), 1);
    }

    #[test]
    fn analyze_equation_preflight_with_collects_domain_exclusions_and_required_conditions() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let one = ctx.num(1);
        let lhs = ctx.add(Expr::Div(one, x));
        let equation = Equation {
            lhs,
            rhs: y,
            op: RelOp::Eq,
        };

        let preflight = analyze_equation_preflight_with(
            &ctx,
            &equation,
            "x",
            0u8,
            |side, _vd| {
                if side == lhs {
                    vec![10, 20]
                } else {
                    vec![20, 30]
                }
            },
            |_lhs, _rhs, existing, _vd| {
                assert_eq!(existing, &[10, 20, 30]);
                vec![30, 40]
            },
        );

        assert_eq!(preflight.required_conditions, vec![10, 20, 30, 40]);
        assert_eq!(preflight.domain_exclusions.len(), 1);
        assert!(preflight.domain_exclusions.contains(&x));
    }

    #[test]
    fn partition_discrete_symbolic_splits_expected() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let x = ctx.var("x");
        let (symbolic, non_symbolic) = partition_discrete_symbolic(&ctx, &[two, x]);
        assert_eq!(symbolic, vec![x]);
        assert_eq!(non_symbolic, vec![two]);
    }

    #[test]
    fn retain_verified_discrete_keeps_only_verified() {
        let sols = vec![
            cas_ast::ExprId::from_raw(1),
            cas_ast::ExprId::from_raw(2),
            cas_ast::ExprId::from_raw(3),
        ];
        let kept = retain_verified_discrete(sols, |id| id.index() % 2 == 1);
        assert_eq!(
            kept,
            vec![cas_ast::ExprId::from_raw(1), cas_ast::ExprId::from_raw(3)]
        );
    }

    #[test]
    fn retain_verified_discrete_invokes_verifier_for_each_solution() {
        let sols = vec![
            cas_ast::ExprId::from_raw(1),
            cas_ast::ExprId::from_raw(2),
            cas_ast::ExprId::from_raw(3),
        ];
        let calls = Cell::new(0usize);
        let kept = retain_verified_discrete(sols, |solution| {
            calls.set(calls.get() + 1);
            solution.index() % 2 == 1
        });
        assert_eq!(
            kept,
            vec![cas_ast::ExprId::from_raw(1), cas_ast::ExprId::from_raw(3)]
        );
        assert_eq!(calls.get(), 3);
    }

    #[test]
    fn accept_rewritten_residual_when_variable_eliminated() {
        assert!(should_accept_rewritten_residual(true, 100, 99));
    }

    #[test]
    fn accept_rewritten_residual_on_significant_reduction() {
        assert!(should_accept_rewritten_residual(false, 20, 14));
    }

    #[test]
    fn reject_rewritten_residual_on_cosmetic_change() {
        assert!(!should_accept_rewritten_residual(false, 20, 19));
    }

    #[test]
    fn merge_symbolic_with_verified_numeric_preserves_order_and_filters_numeric() {
        let x = cas_ast::ExprId::from_raw(11);
        let y = cas_ast::ExprId::from_raw(12);
        let two = cas_ast::ExprId::from_raw(2);
        let three = cas_ast::ExprId::from_raw(3);

        let out =
            merge_symbolic_with_verified_numeric(vec![x, y], vec![two, three], |id| id == three);
        assert_eq!(out, vec![x, y, three]);
    }

    #[test]
    fn merge_symbolic_with_verified_numeric_invokes_numeric_verifier_for_each_candidate() {
        let x = cas_ast::ExprId::from_raw(11);
        let y = cas_ast::ExprId::from_raw(12);
        let two = cas_ast::ExprId::from_raw(2);
        let three = cas_ast::ExprId::from_raw(3);
        let calls = Cell::new(0usize);
        let out = merge_symbolic_with_verified_numeric(vec![x, y], vec![two, three], |solution| {
            calls.set(calls.get() + 1);
            solution == three
        });
        assert_eq!(out, vec![x, y, three]);
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn resolve_discrete_strategy_solutions_preserves_symbolic_and_filters_numeric() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let two = ctx.num(2);
        let three = ctx.num(3);

        let out = resolve_discrete_strategy_solutions_with(
            vec![x, two, y, three],
            |solution| is_symbolic_expr(&ctx, solution),
            |sol| sol == three,
        );
        assert_eq!(out, vec![x, y, three]);
    }

    #[test]
    fn resolve_discrete_strategy_solutions_verifies_only_numeric_candidates() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let calls = Cell::new(0usize);

        let out = resolve_discrete_strategy_solutions_with(
            vec![x, two, three],
            |solution| is_symbolic_expr(&ctx, solution),
            |solution| {
                calls.set(calls.get() + 1);
                solution == three
            },
        );
        assert_eq!(out, vec![x, three]);
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn resolve_discrete_strategy_solutions_with_state_preserves_symbolic_and_filters_numeric() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let two = ctx.num(2);
        let three = ctx.num(3);

        struct State {
            ctx: Context,
            verify_calls: usize,
        }

        let mut state = State {
            ctx,
            verify_calls: 0,
        };

        let out = resolve_discrete_strategy_solutions_with_state(
            &mut state,
            vec![x, two, y, three],
            |state, solution| is_symbolic_expr(&state.ctx, solution),
            |state, solution| {
                state.verify_calls += 1;
                solution == three
            },
        );
        assert_eq!(out, vec![x, y, three]);
        assert_eq!(state.verify_calls, 2);
    }

    #[test]
    fn resolve_discrete_strategy_solutions_with_state_supports_custom_symbolic_policy() {
        let one = ExprId::from_raw(1);
        let two = ExprId::from_raw(2);
        let three = ExprId::from_raw(3);

        let mut state = ();
        let out = resolve_discrete_strategy_solutions_with_state(
            &mut state,
            vec![one, two, three],
            |_state, solution| solution.index() == 2,
            |_state, solution| solution.index() == 3,
        );

        assert_eq!(out, vec![two, three]);
    }

    #[test]
    fn resolve_discrete_strategy_solutions_against_equation_with_state_threads_equation_and_var() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let eq = Equation {
            lhs: x,
            rhs: two,
            op: RelOp::Eq,
        };

        struct State {
            ctx: Context,
            verify_calls: usize,
            last_var: Option<String>,
        }

        let mut state = State {
            ctx,
            verify_calls: 0,
            last_var: None,
        };

        let out = resolve_discrete_strategy_solutions_against_equation_with_state(
            &mut state,
            &eq,
            "x",
            vec![y, two, three],
            |state, solution| is_symbolic_expr(&state.ctx, solution),
            |state, _equation, solve_var, solution| {
                state.verify_calls += 1;
                state.last_var = Some(solve_var.to_string());
                solution == three
            },
        );

        assert_eq!(out, vec![y, three]);
        assert_eq!(state.verify_calls, 2);
        assert_eq!(state.last_var.as_deref(), Some("x"));
    }

    #[test]
    fn resolve_discrete_strategy_result_against_equation_with_state_wraps_solution_set() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let eq = Equation {
            lhs: x,
            rhs: two,
            op: RelOp::Eq,
        };

        struct State {
            ctx: Context,
            verify_calls: usize,
        }

        let mut state = State {
            ctx,
            verify_calls: 0,
        };

        let (solutions, steps) = resolve_discrete_strategy_result_against_equation_with_state(
            &mut state,
            &eq,
            "x",
            vec![y, two, three],
            vec!["s1".to_string()],
            |state, solution| is_symbolic_expr(&state.ctx, solution),
            |state, _equation, _var, solution| {
                state.verify_calls += 1;
                solution == three
            },
        );

        assert_eq!(steps, vec!["s1".to_string()]);
        assert_eq!(state.verify_calls, 2);
        assert_eq!(solutions, SolutionSet::Discrete(vec![y, three]));
    }

    #[test]
    fn classify_equation_var_presence_none() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let eq = Equation {
            lhs: one,
            rhs: two,
            op: RelOp::Eq,
        };
        assert_eq!(
            classify_equation_var_presence(&ctx, &eq, "x"),
            EquationVarPresence::None
        );
    }

    #[test]
    fn classify_equation_var_presence_lhs_only() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let eq = Equation {
            lhs: x,
            rhs: one,
            op: RelOp::Eq,
        };
        assert_eq!(
            classify_equation_var_presence(&ctx, &eq, "x"),
            EquationVarPresence::LhsOnly
        );
    }

    #[test]
    fn classify_equation_var_presence_rhs_only() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let eq = Equation {
            lhs: one,
            rhs: x,
            op: RelOp::Eq,
        };
        assert_eq!(
            classify_equation_var_presence(&ctx, &eq, "x"),
            EquationVarPresence::RhsOnly
        );
    }

    #[test]
    fn classify_equation_var_presence_both_sides() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let lhs = ctx.add(Expr::Add(x, one));
        let rhs = ctx.add(Expr::Sub(x, one));
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };
        assert_eq!(
            classify_equation_var_presence(&ctx, &eq, "x"),
            EquationVarPresence::BothSides
        );
    }

    #[test]
    fn ensure_equation_has_variable_or_error_rejects_none_presence() {
        let result: Result<(), &str> =
            ensure_equation_has_variable_or_error(EquationVarPresence::None, || "missing");
        assert_eq!(result, Err("missing"));
    }

    #[test]
    fn ensure_equation_has_variable_or_error_accepts_non_none_presence() {
        let result: Result<(), &str> =
            ensure_equation_has_variable_or_error(EquationVarPresence::LhsOnly, || "missing");
        assert!(result.is_ok());
    }

    #[test]
    fn ensure_recursion_depth_within_limit_or_error_accepts_in_bounds_depth() {
        let result: Result<(), &str> =
            ensure_recursion_depth_within_limit_or_error(3, 3, || "too deep");
        assert!(result.is_ok());
    }

    #[test]
    fn ensure_recursion_depth_within_limit_or_error_rejects_out_of_bounds_depth() {
        let result: Result<(), &str> =
            ensure_recursion_depth_within_limit_or_error(4, 3, || "too deep");
        assert_eq!(result, Err("too deep"));
    }

    #[test]
    fn ensure_solve_entry_for_equation_or_error_accepts_valid_entry() {
        let mut ctx = Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.num(1);
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };

        let result: Result<(), &str> = ensure_solve_entry_for_equation_or_error(
            &ctx,
            &eq,
            "x",
            1,
            3,
            || "too deep",
            || "missing",
        );

        assert!(result.is_ok());
    }

    #[test]
    fn ensure_solve_entry_for_equation_or_error_rejects_depth_overflow() {
        let mut ctx = Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.num(1);
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };

        let result: Result<(), &str> = ensure_solve_entry_for_equation_or_error(
            &ctx,
            &eq,
            "x",
            4,
            3,
            || "too deep",
            || "missing",
        );

        assert_eq!(result, Err("too deep"));
    }

    #[test]
    fn ensure_solve_entry_for_equation_or_error_rejects_missing_variable() {
        let mut ctx = Context::new();
        let lhs = ctx.var("y");
        let rhs = ctx.num(1);
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };

        let result: Result<(), &str> = ensure_solve_entry_for_equation_or_error(
            &ctx,
            &eq,
            "x",
            1,
            3,
            || "too deep",
            || "missing",
        );

        assert_eq!(result, Err("missing"));
    }

    #[test]
    fn collect_required_conditions_for_equation_with_dedupes_and_preserves_order() {
        let out = collect_required_conditions_for_equation_with(
            ExprId::from_raw(10),
            ExprId::from_raw(20),
            0u8,
            |side, _vd| {
                if side.index() == 10 {
                    vec![1, 2, 1]
                } else {
                    vec![2, 3]
                }
            },
            |_lhs, _rhs, existing, _vd| {
                // Derive from existing set shape and include one duplicate.
                assert_eq!(existing, &[1, 2, 3]);
                vec![3, 4]
            },
        );
        assert_eq!(out, vec![1, 2, 3, 4]);
    }

    #[test]
    fn derive_equation_conditions_from_existing_with_builds_domain_from_existing() {
        #[derive(Default)]
        struct TestDomain {
            values: Vec<i32>,
        }

        let out = derive_equation_conditions_from_existing_with(
            ExprId::from_raw(11),
            ExprId::from_raw(12),
            &[1, 2, 3],
            0u8,
            TestDomain::default,
            |domain, cond| domain.values.push(cond),
            |lhs, rhs, domain, _| {
                assert_eq!(lhs.index(), 11);
                assert_eq!(rhs.index(), 12);
                assert_eq!(domain.values, vec![1, 2, 3]);
                vec![4, 5]
            },
        );
        assert_eq!(out, vec![4, 5]);
    }

    #[test]
    fn apply_required_conditions_with_writes_into_both_sinks() {
        let conditions = vec![10, 20, 30];
        let mut inserted = Vec::new();
        let mut noted = Vec::new();
        apply_required_conditions_with(
            conditions,
            |cond| inserted.push(*cond),
            |cond| noted.push(cond),
        );
        assert_eq!(inserted, vec![10, 20, 30]);
        assert_eq!(noted, vec![10, 20, 30]);
    }

    #[derive(Debug, Clone, Default)]
    struct TestDomainEnv {
        inserted: Vec<i32>,
    }

    type TestSolveCtx = crate::solve_context::SolveContext<TestDomainEnv, i32, &'static str, ()>;

    #[test]
    fn analyze_equation_preflight_and_fork_context_with_hydrates_child_context() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let lhs = ctx.add(Expr::Div(one, x));
        let rhs = ctx.num(0);
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };

        let parent = TestSolveCtx::default();
        let out = analyze_equation_preflight_and_fork_context_with(
            &ctx,
            &eq,
            "x",
            (),
            &parent,
            |side, _| {
                if side == lhs {
                    vec![10]
                } else {
                    vec![20]
                }
            },
            |_lhs, _rhs, existing, _| {
                assert_eq!(existing, &[10, 20]);
                vec![20, 30]
            },
            TestDomainEnv::default(),
            |env, cond| env.inserted.push(*cond),
        );

        assert_eq!(out.ctx.depth(), 1);
        assert_eq!(out.ctx.domain_env.inserted, vec![10, 20, 30]);
        assert_eq!(out.domain_exclusions, vec![x]);

        let mut required = parent.required_conditions();
        required.sort_unstable();
        assert_eq!(required, vec![10, 20, 30]);
    }

    #[test]
    fn classify_strategy_attempt_result_skip_when_not_applicable() {
        let out = classify_strategy_attempt_result::<(), (), _>(None, true, |_| false);
        assert_eq!(out, StrategyAttemptResolution::Skip);
    }

    #[test]
    fn classify_strategy_attempt_result_solved_for_non_discrete() {
        let out = classify_strategy_attempt_result::<String, (), _>(
            Some(Ok((SolutionSet::AllReals, vec!["step".to_string()]))),
            true,
            |_: &()| false,
        );
        assert_eq!(
            out,
            StrategyAttemptResolution::Solved {
                solution_set: SolutionSet::AllReals,
                steps: vec!["step".to_string()]
            }
        );
    }

    #[test]
    fn classify_strategy_attempt_result_discrete_verification_requested() {
        let out = classify_strategy_attempt_result::<String, (), _>(
            Some(Ok((
                SolutionSet::Discrete(vec![ExprId::from_raw(3)]),
                vec!["step".to_string()],
            ))),
            true,
            |_: &()| false,
        );
        assert_eq!(
            out,
            StrategyAttemptResolution::NeedsDiscreteVerification {
                solutions: vec![ExprId::from_raw(3)],
                steps: vec!["step".to_string()]
            }
        );
    }

    #[test]
    fn classify_strategy_attempt_result_discrete_without_verification() {
        let out = classify_strategy_attempt_result::<String, (), _>(
            Some(Ok((
                SolutionSet::Discrete(vec![ExprId::from_raw(3)]),
                vec!["step".to_string()],
            ))),
            false,
            |_: &()| false,
        );
        assert_eq!(
            out,
            StrategyAttemptResolution::Solved {
                solution_set: SolutionSet::Discrete(vec![ExprId::from_raw(3)]),
                steps: vec!["step".to_string()]
            }
        );
    }

    #[test]
    fn classify_strategy_attempt_result_soft_vs_hard_error() {
        let soft = classify_strategy_attempt_result::<(), _, _>(Some(Err("soft")), true, |error| {
            *error == "soft"
        });
        assert_eq!(soft, StrategyAttemptResolution::SoftError("soft"));

        let hard = classify_strategy_attempt_result::<(), _, _>(Some(Err("hard")), true, |error| {
            *error == "soft"
        });
        assert_eq!(hard, StrategyAttemptResolution::HardError("hard"));
    }

    #[test]
    fn run_strategy_attempt_sequence_returns_first_solved() {
        let attempts = vec![
            (None, true),
            (Some(Ok((SolutionSet::AllReals, vec!["done"]))), true),
            (Some(Err("later")), true),
        ];

        let out =
            run_strategy_attempt_sequence::<&str, &str, _, _>(attempts, |error| *error == "soft");
        assert_eq!(
            out,
            StrategyAttemptSequenceResolution::Solved {
                solution_set: SolutionSet::AllReals,
                steps: vec!["done"],
            }
        );
    }

    #[test]
    fn run_strategy_attempt_sequence_returns_discrete_verification_when_needed() {
        let attempts = vec![(
            Some(Ok((
                SolutionSet::Discrete(vec![ExprId::from_raw(9)]),
                vec!["verify"],
            ))),
            true,
        )];

        let out =
            run_strategy_attempt_sequence::<&str, &str, _, _>(attempts, |error| *error == "soft");
        assert_eq!(
            out,
            StrategyAttemptSequenceResolution::NeedsDiscreteVerification {
                solutions: vec![ExprId::from_raw(9)],
                steps: vec!["verify"],
            }
        );
    }

    #[test]
    fn run_strategy_attempt_sequence_returns_hard_error_immediately() {
        let attempts = vec![
            (Some(Err("hard")), true),
            (Some(Ok((SolutionSet::AllReals, vec!["never"]))), true),
        ];
        let out = run_strategy_attempt_sequence(attempts, |error| *error == "soft");
        assert_eq!(out, StrategyAttemptSequenceResolution::HardError("hard"));
    }

    #[test]
    fn run_strategy_attempt_sequence_exhausted_keeps_last_soft_error() {
        let attempts = vec![
            (Some(Err("soft-1")), true),
            (None, true),
            (Some(Err("soft-2")), true),
        ];
        let out = run_strategy_attempt_sequence::<(), &str, _, _>(attempts, |error| {
            error.starts_with("soft")
        });
        assert_eq!(
            out,
            StrategyAttemptSequenceResolution::Exhausted {
                last_soft_error: Some("soft-2")
            }
        );
    }

    #[test]
    fn run_strategy_attempt_sequence_exhausted_without_soft_error() {
        let attempts = vec![(None, true), (None, false)];
        let out = run_strategy_attempt_sequence::<(), &str, _, _>(attempts, |_| false);
        assert_eq!(
            out,
            StrategyAttemptSequenceResolution::Exhausted {
                last_soft_error: None
            }
        );
    }

    #[test]
    fn run_strategies_with_state_tracks_state_and_stops_on_solved() {
        #[derive(Default)]
        struct State {
            visited: Vec<u8>,
        }

        let mut state = State::default();
        let strategies = vec![1u8, 2, 3, 4];
        let out = run_strategies_with_state(
            &mut state,
            strategies,
            |state, strategy_id| {
                state.visited.push(strategy_id);
                match strategy_id {
                    1 => (None, true),
                    2 => (Some(Err("soft")), true),
                    3 => (Some(Ok((SolutionSet::AllReals, vec!["done"]))), true),
                    _ => (Some(Err("hard")), true),
                }
            },
            |err| *err == "soft",
        );

        assert_eq!(state.visited, vec![1, 2, 3]);
        assert_eq!(
            out,
            StrategyAttemptSequenceResolution::Solved {
                solution_set: SolutionSet::AllReals,
                steps: vec!["done"],
            }
        );
    }

    #[test]
    fn execute_strategies_with_state_and_resolution_handles_discrete_resolution() {
        let mut state = ();
        let strategies = vec![7u8];
        let resolved = execute_strategies_with_state_and_resolution(
            &mut state,
            strategies,
            |_state, _strategy| {
                (
                    Some(Ok((
                        SolutionSet::Discrete(vec![ExprId::from_raw(5)]),
                        vec!["verify"],
                    ))),
                    true,
                )
            },
            |_: &&str| false,
            |_state, _solutions, steps| (SolutionSet::Discrete(vec![ExprId::from_raw(8)]), steps),
            "fallback",
        )
        .expect("discrete verification path should resolve");

        assert_eq!(
            resolved,
            (
                SolutionSet::Discrete(vec![ExprId::from_raw(8)]),
                vec!["verify"],
            )
        );
    }

    #[test]
    fn execute_prepared_equation_strategy_pipeline_with_state_resolves_var_eliminated_without_cycle(
    ) {
        #[derive(Default)]
        struct State {
            cycle_attempts: usize,
            strategy_attempts: usize,
        }

        let mut ctx = Context::new();
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let equation = Equation {
            lhs: x,
            rhs: zero,
            op: RelOp::Eq,
        };
        let mut state = State::default();

        let solved = execute_prepared_equation_strategy_pipeline_with_state(
            &mut state,
            &equation,
            zero,
            "x",
            vec![1u8, 2u8],
            |_state, _expr, _var| false,
            |_state, _residual, _var| Ok((SolutionSet::AllReals, vec!["resolved"])),
            |state, _equation, _var| -> Result<(), &str> {
                state.cycle_attempts += 1;
                Ok(())
            },
            |state, _strategy| {
                state.strategy_attempts += 1;
                (None, true)
            },
            |_err| false,
            |_state, _solutions, steps| (SolutionSet::Discrete(vec![]), steps),
            "fallback",
        )
        .expect("var-eliminated path should resolve immediately");

        assert_eq!(solved, (SolutionSet::AllReals, vec!["resolved"]));
        assert_eq!(state.cycle_attempts, 0);
        assert_eq!(state.strategy_attempts, 0);
    }

    #[test]
    fn execute_prepared_equation_strategy_pipeline_with_state_runs_cycle_and_strategies() {
        #[derive(Default)]
        struct State {
            cycle_attempts: usize,
            strategy_attempts: usize,
        }

        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let equation = Equation {
            lhs: x,
            rhs: one,
            op: RelOp::Eq,
        };
        let mut state = State::default();

        let solved = execute_prepared_equation_strategy_pipeline_with_state(
            &mut state,
            &equation,
            x,
            "x",
            vec![1u8, 2u8],
            |_state, _expr, _var| true,
            |_state, _residual, _var| Ok((SolutionSet::Empty, vec!["never"])),
            |state, _equation, _var| -> Result<(), &str> {
                state.cycle_attempts += 1;
                Ok(())
            },
            |state, strategy| {
                state.strategy_attempts += 1;
                match strategy {
                    1 => (None, true),
                    _ => (Some(Ok((SolutionSet::AllReals, vec!["done"]))), true),
                }
            },
            |_err| false,
            |_state, _solutions, steps| (SolutionSet::Discrete(vec![]), steps),
            "fallback",
        )
        .expect("strategy path should solve");

        assert_eq!(solved, (SolutionSet::AllReals, vec!["done"]));
        assert_eq!(state.cycle_attempts, 1);
        assert_eq!(state.strategy_attempts, 2);
    }

    #[test]
    fn execute_prepared_equation_strategy_pipeline_with_state_propagates_cycle_error() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let equation = Equation {
            lhs: x,
            rhs: one,
            op: RelOp::Eq,
        };

        let mut state = ();
        let err = execute_prepared_equation_strategy_pipeline_with_state(
            &mut state,
            &equation,
            x,
            "x",
            vec![1u8],
            |_state, _expr, _var| true,
            |_state, _residual, _var| Ok((SolutionSet::Empty, vec!["never"])),
            |_state, _equation, _var| -> Result<(), &str> { Err("cycle") },
            |_state, _strategy| (None, true),
            |_err| false,
            |_state, _solutions, steps| (SolutionSet::Discrete(vec![]), steps),
            "fallback",
        )
        .expect_err("cycle error should propagate");

        assert_eq!(err, "cycle");
    }

    #[test]
    fn is_soft_strategy_error_from_message_parts_matches_expected_patterns() {
        assert!(is_soft_strategy_error_from_message_parts(
            Some("variable appears on both sides after isolate"),
            None,
        ));
        assert!(is_soft_strategy_error_from_message_parts(
            None,
            Some("Cycle detected: equivalent form loop"),
        ));
        assert!(!is_soft_strategy_error_from_message_parts(
            Some("other isolation error"),
            Some("hard failure"),
        ));
    }

    #[derive(Debug)]
    struct SoftErrorFixture {
        isolation_detail: Option<&'static str>,
        solver_detail: Option<&'static str>,
    }

    #[test]
    fn is_soft_strategy_error_with_uses_extractors() {
        let soft_isolation = SoftErrorFixture {
            isolation_detail: Some("variable appears on both sides during rewrite"),
            solver_detail: None,
        };
        assert!(is_soft_strategy_error_with(
            &soft_isolation,
            |err| err.isolation_detail,
            |err| err.solver_detail,
        ));

        let soft_cycle = SoftErrorFixture {
            isolation_detail: None,
            solver_detail: Some("Cycle detected: equivalent form loop"),
        };
        assert!(is_soft_strategy_error_with(
            &soft_cycle,
            |err| err.isolation_detail,
            |err| err.solver_detail,
        ));

        let hard = SoftErrorFixture {
            isolation_detail: Some("incompatible branch"),
            solver_detail: Some("hard failure"),
        };
        assert!(!is_soft_strategy_error_with(
            &hard,
            |err| err.isolation_detail,
            |err| err.solver_detail,
        ));
    }

    #[test]
    fn try_enter_equation_cycle_guard_with_error_reports_reentry() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let eq = Equation {
            lhs: x,
            rhs: one,
            op: RelOp::Eq,
        };

        let guard = try_enter_equation_cycle_guard_with_error(&ctx, &eq, "x", || "cycle")
            .expect("first entry should succeed");
        let reentry = match try_enter_equation_cycle_guard_with_error(&ctx, &eq, "x", || "cycle") {
            Ok(_) => panic!("reentry should fail"),
            Err(err) => err,
        };
        assert_eq!(reentry, "cycle");
        drop(guard);
        assert!(
            try_enter_equation_cycle_guard_with_error(&ctx, &eq, "x", || "cycle").is_ok(),
            "entry should succeed again after guard drop"
        );
    }

    #[test]
    fn finalize_strategy_attempt_sequence_with_returns_solved_directly() {
        let resolved = finalize_strategy_attempt_sequence_with(
            StrategyAttemptSequenceResolution::Solved {
                solution_set: SolutionSet::AllReals,
                steps: vec!["done"],
            },
            |_solutions, _steps| (SolutionSet::Empty, Vec::<&str>::new()),
            "fallback",
        )
        .expect("solved result should pass through");
        assert_eq!(resolved, (SolutionSet::AllReals, vec!["done"]));
    }

    #[test]
    fn finalize_strategy_attempt_sequence_with_resolves_discrete_candidates() {
        let resolved = finalize_strategy_attempt_sequence_with(
            StrategyAttemptSequenceResolution::NeedsDiscreteVerification {
                solutions: vec![ExprId::from_raw(3)],
                steps: vec!["verify"],
            },
            |_solutions, steps| (SolutionSet::Discrete(vec![ExprId::from_raw(7)]), steps),
            "fallback",
        )
        .expect("discrete resolution should be delegated");
        assert_eq!(
            resolved,
            (
                SolutionSet::Discrete(vec![ExprId::from_raw(7)]),
                vec!["verify"]
            )
        );
    }

    #[test]
    fn finalize_strategy_attempt_sequence_with_prefers_soft_error_over_fallback() {
        let err = finalize_strategy_attempt_sequence_with::<(), &str, _>(
            StrategyAttemptSequenceResolution::Exhausted {
                last_soft_error: Some("soft"),
            },
            |_solutions, _steps| (SolutionSet::Empty, Vec::<()>::new()),
            "fallback",
        )
        .expect_err("soft error should surface");
        assert_eq!(err, "soft");
    }

    #[test]
    fn finalize_strategy_attempt_sequence_with_uses_fallback_when_exhausted_without_soft_error() {
        let err = finalize_strategy_attempt_sequence_with::<(), &str, _>(
            StrategyAttemptSequenceResolution::Exhausted {
                last_soft_error: None,
            },
            |_solutions, _steps| (SolutionSet::Empty, Vec::<()>::new()),
            "fallback",
        )
        .expect_err("fallback error should surface");
        assert_eq!(err, "fallback");
    }

    #[test]
    fn simplify_equation_sides_for_var_only_simplifies_sides_with_variable() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let lhs = ctx.add(Expr::Add(x, one));
        let eq = Equation {
            lhs,
            rhs: two,
            op: RelOp::Eq,
        };
        let simplified_calls = RefCell::new(Vec::new());

        let simplified = simplify_equation_sides_for_var_with(
            &eq,
            "x",
            |expr, _| expr == lhs,
            |expr| {
                simplified_calls.borrow_mut().push(expr);
                expr
            },
            |_expr| None,
        );
        assert_eq!(simplified.lhs, lhs);
        assert_eq!(simplified.rhs, two);
        assert_eq!(*simplified_calls.borrow(), vec![lhs]);
    }

    #[test]
    fn simplify_equation_sides_for_presence_with_uses_precomputed_presence() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let lhs = ctx.add(Expr::Add(x, one));
        let eq = Equation {
            lhs,
            rhs: two,
            op: RelOp::Eq,
        };
        let simplified_calls = RefCell::new(Vec::new());

        let simplified = simplify_equation_sides_for_presence_with(
            &eq,
            true,
            false,
            |expr| {
                simplified_calls.borrow_mut().push(expr);
                expr
            },
            |_expr| None,
        );
        assert_eq!(simplified.lhs, lhs);
        assert_eq!(simplified.rhs, two);
        assert_eq!(*simplified_calls.borrow(), vec![lhs]);
    }

    #[test]
    fn simplify_equation_sides_for_presence_with_state_uses_precomputed_presence() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let lhs = ctx.add(Expr::Add(x, one));
        let eq = Equation {
            lhs,
            rhs: two,
            op: RelOp::Eq,
        };
        let mut simplified_calls = Vec::new();

        let simplified = simplify_equation_sides_for_presence_with_state(
            &mut simplified_calls,
            &eq,
            true,
            false,
            |calls, expr| {
                calls.push(expr);
                expr
            },
            |_calls, _expr| None,
        );
        assert_eq!(simplified.lhs, lhs);
        assert_eq!(simplified.rhs, two);
        assert_eq!(simplified_calls, vec![lhs]);
    }

    #[test]
    fn apply_equation_pair_rewrite_sequence_runs_structural_then_semantic() {
        let mut ctx = Context::new();
        let lhs0 = ctx.num(1);
        let rhs0 = ctx.num(2);
        let lhs1 = ctx.num(3);
        let rhs1 = ctx.num(4);
        let lhs1_sim = ctx.num(5);
        let rhs1_sim = ctx.num(6);
        let lhs2 = ctx.num(7);
        let rhs2 = ctx.num(8);

        let seen_semantic = RefCell::new(None::<(ExprId, ExprId)>);
        let simplify_calls = RefCell::new(Vec::new());

        let (lhs_out, rhs_out) = apply_equation_pair_rewrite_sequence_with(
            lhs0,
            rhs0,
            |lhs, rhs| {
                if lhs == lhs0 && rhs == rhs0 {
                    Some((lhs1, rhs1))
                } else {
                    None
                }
            },
            |lhs, rhs| {
                *seen_semantic.borrow_mut() = Some((lhs, rhs));
                if lhs == lhs1_sim && rhs == rhs1_sim {
                    Some((lhs2, rhs2))
                } else {
                    None
                }
            },
            |expr| {
                simplify_calls.borrow_mut().push(expr);
                if expr == lhs1 {
                    lhs1_sim
                } else if expr == rhs1 {
                    rhs1_sim
                } else {
                    expr
                }
            },
        );

        assert_eq!(seen_semantic.borrow().as_ref(), Some(&(lhs1_sim, rhs1_sim)));
        assert_eq!(lhs_out, lhs2);
        assert_eq!(rhs_out, rhs2);
        assert_eq!(
            *simplify_calls.borrow(),
            vec![lhs1, rhs1, lhs2, rhs2],
            "each accepted rewrite should simplify both sides"
        );
    }

    #[test]
    fn apply_equation_pair_rewrite_sequence_keeps_original_when_no_rewrites() {
        let mut ctx = Context::new();
        let lhs = ctx.num(11);
        let rhs = ctx.num(12);
        let simplify_calls = Cell::new(0usize);

        let (lhs_out, rhs_out) = apply_equation_pair_rewrite_sequence_with(
            lhs,
            rhs,
            |_lhs, _rhs| None,
            |_lhs, _rhs| None,
            |expr| {
                simplify_calls.set(simplify_calls.get() + 1);
                expr
            },
        );

        assert_eq!(lhs_out, lhs);
        assert_eq!(rhs_out, rhs);
        assert_eq!(simplify_calls.get(), 0);
    }

    #[test]
    fn apply_equation_pair_rewrite_sequence_with_state_runs_structural_then_semantic() {
        let mut ctx = Context::new();
        let lhs0 = ctx.num(1);
        let rhs0 = ctx.num(2);
        let lhs1 = ctx.num(3);
        let rhs1 = ctx.num(4);
        let lhs1_sim = ctx.num(5);
        let rhs1_sim = ctx.num(6);
        let lhs2 = ctx.num(7);
        let rhs2 = ctx.num(8);

        #[derive(Default)]
        struct RewriteState {
            seen_semantic: Option<(ExprId, ExprId)>,
            simplify_calls: Vec<ExprId>,
        }

        let mut state = RewriteState::default();
        let (lhs_out, rhs_out) = apply_equation_pair_rewrite_sequence_with_state(
            &mut state,
            lhs0,
            rhs0,
            |_state, lhs, rhs| {
                if lhs == lhs0 && rhs == rhs0 {
                    Some((lhs1, rhs1))
                } else {
                    None
                }
            },
            |state, lhs, rhs| {
                state.seen_semantic = Some((lhs, rhs));
                if lhs == lhs1_sim && rhs == rhs1_sim {
                    Some((lhs2, rhs2))
                } else {
                    None
                }
            },
            |state, expr| {
                state.simplify_calls.push(expr);
                if expr == lhs1 {
                    lhs1_sim
                } else if expr == rhs1 {
                    rhs1_sim
                } else {
                    expr
                }
            },
        );

        assert_eq!(state.seen_semantic.as_ref(), Some(&(lhs1_sim, rhs1_sim)));
        assert_eq!(lhs_out, lhs2);
        assert_eq!(rhs_out, rhs2);
        assert_eq!(
            state.simplify_calls,
            vec![lhs1, rhs1, lhs2, rhs2],
            "each accepted rewrite should simplify both sides"
        );
    }

    #[test]
    fn accept_residual_candidate_accepts_variable_elimination() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let current = ctx.add(Expr::Add(x, one));
        let candidate = one;

        let accepted = accept_residual_rewrite_candidate(&ctx, current, candidate, "x");
        assert_eq!(accepted, Some(candidate));
    }

    #[test]
    fn accept_residual_candidate_rejects_cosmetic_rewrite() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let current = ctx.add(Expr::Add(x, one));
        let candidate = current;

        let accepted = accept_residual_rewrite_candidate(&ctx, current, candidate, "x");
        assert_eq!(accepted, None);
    }

    #[test]
    fn normalize_variable_residual_stops_after_algebraic_eliminates_variable() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let residual = ctx.add(Expr::Add(x, one));
        let algebraic_calls = Cell::new(0usize);
        let simplify_calls = Cell::new(0usize);
        let trig_calls = Cell::new(0usize);

        let normalized = normalize_variable_residual_with(
            residual,
            "x",
            |expr, _| expr == residual,
            |_expr| {
                algebraic_calls.set(algebraic_calls.get() + 1);
                one
            },
            |expr| {
                simplify_calls.set(simplify_calls.get() + 1);
                expr
            },
            |_expr| {
                trig_calls.set(trig_calls.get() + 1);
                residual
            },
            |current, candidate, var| {
                accept_residual_rewrite_candidate(&ctx, current, candidate, var)
            },
        );
        assert_eq!(normalized, one);
        assert_eq!(algebraic_calls.get(), 1);
        assert_eq!(simplify_calls.get(), 1);
        assert_eq!(trig_calls.get(), 0);
    }

    #[test]
    fn normalize_variable_residual_uses_trig_fallback_when_algebraic_is_not_better() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let residual = ctx.add(Expr::Add(x, one));
        let algebraic_cosmetic = residual;
        let trig_eliminated = one;
        let algebraic_calls = Cell::new(0usize);
        let simplify_calls = Cell::new(0usize);
        let trig_calls = Cell::new(0usize);

        let normalized = normalize_variable_residual_with(
            residual,
            "x",
            |expr, _| expr == residual,
            |_expr| {
                algebraic_calls.set(algebraic_calls.get() + 1);
                algebraic_cosmetic
            },
            |expr| {
                simplify_calls.set(simplify_calls.get() + 1);
                expr
            },
            |_expr| {
                trig_calls.set(trig_calls.get() + 1);
                trig_eliminated
            },
            |current, candidate, var| {
                accept_residual_rewrite_candidate(&ctx, current, candidate, var)
            },
        );
        assert_eq!(normalized, trig_eliminated);
        assert_eq!(algebraic_calls.get(), 1);
        assert_eq!(simplify_calls.get(), 1);
        assert_eq!(trig_calls.get(), 1);
    }

    #[test]
    fn normalize_variable_residual_with_state_uses_trig_fallback_when_algebraic_is_not_better() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let residual = ctx.add(Expr::Add(x, one));
        let algebraic_cosmetic = residual;
        let trig_eliminated = one;

        #[derive(Default)]
        struct ResidualState {
            algebraic_calls: usize,
            simplify_calls: usize,
            trig_calls: usize,
        }

        let mut state = ResidualState::default();
        let normalized = normalize_variable_residual_with_state(
            &mut state,
            residual,
            "x",
            |_state, expr, _| expr == residual,
            |state, _expr| {
                state.algebraic_calls += 1;
                algebraic_cosmetic
            },
            |state, expr| {
                state.simplify_calls += 1;
                expr
            },
            |state, _expr| {
                state.trig_calls += 1;
                trig_eliminated
            },
            |_state, current, candidate, var| {
                accept_residual_rewrite_candidate(&ctx, current, candidate, var)
            },
        );
        assert_eq!(normalized, trig_eliminated);
        assert_eq!(state.algebraic_calls, 1);
        assert_eq!(state.simplify_calls, 1);
        assert_eq!(state.trig_calls, 1);
    }

    #[test]
    fn prepare_equation_for_strategy_with_state_rewrites_to_zero_equation_when_residual_changes() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let zero = ctx.num(0);
        let equation = Equation {
            lhs: x,
            rhs: one,
            op: RelOp::Eq,
        };

        let prepared = prepare_equation_for_strategy_with_state(
            &mut ctx,
            &equation,
            "x",
            |ctx, expr, var| crate::isolation_utils::contains_var(ctx, expr, var),
            |_ctx, expr| expr,
            |_ctx, _expr| None,
            |_ctx, _lhs, _rhs| None,
            |_ctx, _lhs, _rhs| None,
            |ctx, lhs, rhs| ctx.add(Expr::Sub(lhs, rhs)),
            |_ctx, _expr| zero,
            |_ctx, expr| expr,
            |ctx, current, candidate, var| {
                accept_residual_rewrite_candidate(ctx, current, candidate, var)
            },
            |_ctx| zero,
        );

        assert_eq!(prepared.residual, zero);
        assert_eq!(prepared.equation.lhs, zero);
        assert_eq!(prepared.equation.rhs, zero);
    }

    #[test]
    fn prepare_equation_for_strategy_with_state_preserves_equation_when_residual_not_improved() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let equation = Equation {
            lhs: x,
            rhs: one,
            op: RelOp::Eq,
        };

        let prepared = prepare_equation_for_strategy_with_state(
            &mut ctx,
            &equation,
            "x",
            |ctx, expr, var| crate::isolation_utils::contains_var(ctx, expr, var),
            |_ctx, expr| expr,
            |_ctx, _expr| None,
            |_ctx, _lhs, _rhs| None,
            |_ctx, _lhs, _rhs| None,
            |ctx, lhs, rhs| ctx.add(Expr::Sub(lhs, rhs)),
            |_ctx, expr| expr,
            |_ctx, expr| expr,
            |_ctx, _current, _candidate, _var| None,
            |ctx| ctx.num(0),
        );

        assert_eq!(prepared.equation, equation);
        assert!(matches!(ctx.get(prepared.residual), Expr::Sub(_, _)));
    }
}
