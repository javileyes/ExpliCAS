use cas_ast::{
    Case, ConditionPredicate, ConditionSet, Context, Equation, Expr, ExprId, SolutionSet,
};
use std::collections::HashSet;

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

/// Resolve discrete strategy candidates using core symbolic classification.
pub fn resolve_discrete_strategy_solutions_for_context_with<F>(
    ctx: &Context,
    solutions: Vec<ExprId>,
    verify_numeric: F,
) -> Vec<ExprId>
where
    F: FnMut(ExprId) -> bool,
{
    resolve_discrete_strategy_solutions_with(
        solutions,
        |solution| is_symbolic_expr(ctx, solution),
        verify_numeric,
    )
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

        let out = resolve_discrete_strategy_solutions_for_context_with(
            &ctx,
            vec![x, two, y, three],
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

        let out = resolve_discrete_strategy_solutions_for_context_with(
            &ctx,
            vec![x, two, three],
            |solution| {
                calls.set(calls.get() + 1);
                solution == three
            },
        );
        assert_eq!(out, vec![x, three]);
        assert_eq!(calls.get(), 2);
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
}
