use crate::isolation_utils::{contains_var, mk_residual_solve, NumericSign};
use crate::log_domain::{
    assumptions_to_condition_set, classify_log_unsupported_route, classify_terminal_action,
    decision_assumptions, DomainModeKind, LogAssumption, LogSolveDecision, LogTerminalAction,
    LogUnsupportedRoute,
};
use crate::solution_set::{isolated_var_solution, open_positive_domain};
use cas_ast::{
    Case, ConditionPredicate, ConditionSet, Context, Equation, Expr, ExprId, RelOp, SolutionSet,
    SolveResult,
};

/// Classification of a variable-free equation residual `diff = lhs - rhs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarFreeDiffKind {
    /// `diff == 0`: identity (`0 = 0`)
    IdentityZero,
    /// `diff != 0` as a concrete number: contradiction (`c = 0`)
    ContradictionNonZero,
    /// Non-numeric residual over other symbols: constraint on parameters
    Constraint,
}

/// Normalized solve outcome when the target variable disappears from `lhs - rhs`.
#[derive(Debug, Clone, PartialEq)]
pub enum VarEliminatedSolveOutcome {
    IdentityAllReals,
    ContradictionEmpty,
    ConstraintAllReals {
        description: String,
        equation_after: Equation,
    },
}

/// Generic terminal solve outcome (message + solution set).
#[derive(Debug, Clone, PartialEq)]
pub struct TerminalSolveOutcome {
    pub message: &'static str,
    pub solutions: SolutionSet,
}

/// Outcome for a direct isolated-variable equation `x op rhs`.
#[derive(Debug, Clone, PartialEq)]
pub enum IsolatedVariableOutcome {
    Solved(SolutionSet),
    ContainsTargetVariable,
}

/// Fallback outcome when an isolated-variable RHS still contains the target.
#[derive(Debug, Clone, PartialEq)]
pub enum CircularIsolatedOutcome<S> {
    Solved {
        solution_set: SolutionSet,
        steps: Vec<S>,
    },
    Residual(SolutionSet),
}

/// Resolve the final outcome for `x op rhs` once the variable is syntactically
/// isolated on the left-hand side.
pub fn resolve_isolated_variable_outcome(
    ctx: &mut Context,
    rhs: ExprId,
    op: RelOp,
    var: &str,
) -> IsolatedVariableOutcome {
    if contains_var(ctx, rhs, var) {
        IsolatedVariableOutcome::ContainsTargetVariable
    } else {
        IsolatedVariableOutcome::Solved(isolated_var_solution(ctx, rhs, op))
    }
}

/// Execute circular-isolation fallback in canonical order:
/// `linear_collect` -> `linear_collect_v2` -> residual.
pub fn resolve_circular_isolated_outcome_with<S, FTry1, FTry2, FResidual>(
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    mut try_linear_collect: FTry1,
    mut try_linear_collect_v2: FTry2,
    mut residual_solution: FResidual,
) -> CircularIsolatedOutcome<S>
where
    FTry1: FnMut(ExprId, ExprId, &str) -> Option<(SolutionSet, Vec<S>)>,
    FTry2: FnMut(ExprId, ExprId, &str) -> Option<(SolutionSet, Vec<S>)>,
    FResidual: FnMut(ExprId, ExprId, &str) -> SolutionSet,
{
    if let Some((solution_set, steps)) = try_linear_collect(lhs, rhs, var) {
        return CircularIsolatedOutcome::Solved {
            solution_set,
            steps,
        };
    }

    if let Some((solution_set, steps)) = try_linear_collect_v2(lhs, rhs, var) {
        return CircularIsolatedOutcome::Solved {
            solution_set,
            steps,
        };
    }

    CircularIsolatedOutcome::Residual(residual_solution(lhs, rhs, var))
}

/// Solve a branch where `lhs` is already the target variable (`x op rhs`).
///
/// Returned steps are non-empty only when circular fallback resolves via
/// linear-collect routes.
#[allow(clippy::too_many_arguments)]
pub fn solve_isolated_variable_lhs_with_resolver<S, FResolve, FSimplify, FTry1, FTry2, FResidual>(
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    mut resolve_isolated_outcome: FResolve,
    mut simplify_rhs: FSimplify,
    try_linear_collect: FTry1,
    try_linear_collect_v2: FTry2,
    residual_solution: FResidual,
) -> (SolutionSet, Vec<S>)
where
    FResolve: FnMut(ExprId, RelOp, &str) -> IsolatedVariableOutcome,
    FSimplify: FnMut(ExprId) -> ExprId,
    FTry1: FnMut(ExprId, ExprId, &str) -> Option<(SolutionSet, Vec<S>)>,
    FTry2: FnMut(ExprId, ExprId, &str) -> Option<(SolutionSet, Vec<S>)>,
    FResidual: FnMut(ExprId, ExprId, &str) -> SolutionSet,
{
    let sim_rhs = simplify_rhs(rhs);
    match resolve_isolated_outcome(sim_rhs, op, var) {
        IsolatedVariableOutcome::Solved(set) => (set, Vec::new()),
        IsolatedVariableOutcome::ContainsTargetVariable => {
            match resolve_circular_isolated_outcome_with(
                lhs,
                rhs,
                var,
                try_linear_collect,
                try_linear_collect_v2,
                residual_solution,
            ) {
                CircularIsolatedOutcome::Solved {
                    solution_set,
                    steps,
                } => (solution_set, steps),
                CircularIsolatedOutcome::Residual(set) => (set, Vec::new()),
            }
        }
    }
}

/// Solve a branch where `lhs` is already the target variable (`x op rhs`).
///
/// Returned steps are non-empty only when circular fallback resolves via
/// linear-collect routes.
#[allow(clippy::too_many_arguments)]
pub fn solve_isolated_variable_lhs_with<S, FSimplify, FTry1, FTry2, FResidual>(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplify_rhs: FSimplify,
    try_linear_collect: FTry1,
    try_linear_collect_v2: FTry2,
    residual_solution: FResidual,
) -> (SolutionSet, Vec<S>)
where
    FSimplify: FnMut(ExprId) -> ExprId,
    FTry1: FnMut(ExprId, ExprId, &str) -> Option<(SolutionSet, Vec<S>)>,
    FTry2: FnMut(ExprId, ExprId, &str) -> Option<(SolutionSet, Vec<S>)>,
    FResidual: FnMut(ExprId, ExprId, &str) -> SolutionSet,
{
    solve_isolated_variable_lhs_with_resolver(
        lhs,
        rhs,
        op,
        var,
        |sim_rhs, rel_op, solve_var| {
            resolve_isolated_variable_outcome(ctx, sim_rhs, rel_op, solve_var)
        },
        simplify_rhs,
        try_linear_collect,
        try_linear_collect_v2,
        residual_solution,
    )
}

/// Solve an already-isolated variable branch and prepend solved steps before
/// caller-owned existing steps.
#[allow(clippy::too_many_arguments)]
pub fn solve_isolated_variable_lhs_with_resolver_and_merge_with_existing_steps<
    S,
    FResolve,
    FSimplify,
    FTry1,
    FTry2,
    FResidual,
>(
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    existing_steps: Vec<S>,
    resolve_isolated_outcome: FResolve,
    simplify_rhs: FSimplify,
    try_linear_collect: FTry1,
    try_linear_collect_v2: FTry2,
    residual_solution: FResidual,
) -> (SolutionSet, Vec<S>)
where
    FResolve: FnMut(ExprId, RelOp, &str) -> IsolatedVariableOutcome,
    FSimplify: FnMut(ExprId) -> ExprId,
    FTry1: FnMut(ExprId, ExprId, &str) -> Option<(SolutionSet, Vec<S>)>,
    FTry2: FnMut(ExprId, ExprId, &str) -> Option<(SolutionSet, Vec<S>)>,
    FResidual: FnMut(ExprId, ExprId, &str) -> SolutionSet,
{
    let solved = solve_isolated_variable_lhs_with_resolver(
        lhs,
        rhs,
        op,
        var,
        resolve_isolated_outcome,
        simplify_rhs,
        try_linear_collect,
        try_linear_collect_v2,
        residual_solution,
    );
    merge_solved_with_existing_steps_prepend(solved, existing_steps)
}

/// Route for handling `base^x = base` shortcuts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerEqualsBaseRoute {
    /// `0^x = 0` -> `x > 0`.
    ExponentGreaterThanZero,
    /// Numeric base (non-zero): `x = 1`.
    ExponentEqualsOneNumericBase,
    /// Symbolic base with no branch budget: fallback `x = 1`.
    ExponentEqualsOneNoBranchBudget,
    /// Symbolic base with branching budget: produce conditional case split.
    SymbolicCaseSplit,
}

/// Initial dispatch for isolating power equations `b^e = rhs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowIsolationRoute {
    VariableInBase,
    VariableInExponent,
}

/// Initial dispatch for isolating additive equations `(l + r) = rhs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddIsolationRoute {
    LeftOperand,
    RightOperand,
    BothOperands,
}

/// Selected operands for additive isolation `(l + r) = rhs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddIsolationOperands {
    pub route: AddIsolationRoute,
    pub isolated_addend: ExprId,
    pub moved_addend: ExprId,
}

/// Derive which addends of `(l + r)` contain the solve variable.
pub fn derive_add_isolation_route(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
    var: &str,
) -> AddIsolationRoute {
    let left_has = contains_var(ctx, left, var);
    let right_has = contains_var(ctx, right, var);
    match (left_has, right_has) {
        (true, true) => AddIsolationRoute::BothOperands,
        (true, false) => AddIsolationRoute::LeftOperand,
        (false, true) | (false, false) => AddIsolationRoute::RightOperand,
    }
}

/// Derive the isolated and moved addends for `(l + r) = rhs`.
pub fn derive_add_isolation_operands(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
    var: &str,
) -> AddIsolationOperands {
    let route = derive_add_isolation_route(ctx, left, right, var);
    match route {
        AddIsolationRoute::LeftOperand | AddIsolationRoute::BothOperands => AddIsolationOperands {
            route,
            isolated_addend: left,
            moved_addend: right,
        },
        AddIsolationRoute::RightOperand => AddIsolationOperands {
            route,
            isolated_addend: right,
            moved_addend: left,
        },
    }
}

/// Initial dispatch for isolating subtractive equations `(l - r) = rhs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubIsolationRoute {
    Minuend,
    Subtrahend,
}

/// Selected terms for subtractive isolation `(l - r) = rhs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubIsolationOperands {
    pub route: SubIsolationRoute,
    pub isolated_term: ExprId,
    pub moved_term: ExprId,
}

/// Derive which term of `(l - r)` contains the solve variable.
pub fn derive_sub_isolation_route(ctx: &Context, minuend: ExprId, var: &str) -> SubIsolationRoute {
    if contains_var(ctx, minuend, var) {
        SubIsolationRoute::Minuend
    } else {
        SubIsolationRoute::Subtrahend
    }
}

/// Derive the isolated and moved terms for `(l - r) = rhs`.
pub fn derive_sub_isolation_operands(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
    var: &str,
) -> SubIsolationOperands {
    let route = derive_sub_isolation_route(ctx, left, var);
    match route {
        SubIsolationRoute::Minuend => SubIsolationOperands {
            route,
            isolated_term: left,
            moved_term: right,
        },
        SubIsolationRoute::Subtrahend => SubIsolationOperands {
            route,
            isolated_term: right,
            moved_term: left,
        },
    }
}

/// Initial dispatch for isolating multiplicative equations `(l * r) = rhs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MulIsolationRoute {
    LeftFactor,
    RightFactor,
}

/// Selected operands for multiplicative isolation `(l * r) = rhs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MulIsolationOperands {
    pub route: MulIsolationRoute,
    pub isolated_factor: ExprId,
    pub moved_factor: ExprId,
}

/// Derive which factor of `(l * r)` is isolated first.
pub fn derive_mul_isolation_route(ctx: &Context, left: ExprId, var: &str) -> MulIsolationRoute {
    if contains_var(ctx, left, var) {
        MulIsolationRoute::LeftFactor
    } else {
        MulIsolationRoute::RightFactor
    }
}

/// Derive the isolated factor and moved factor for `(l * r) = rhs`.
pub fn derive_mul_isolation_operands(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
    var: &str,
) -> MulIsolationOperands {
    let route = derive_mul_isolation_route(ctx, left, var);
    match route {
        MulIsolationRoute::LeftFactor => MulIsolationOperands {
            route,
            isolated_factor: left,
            moved_factor: right,
        },
        MulIsolationRoute::RightFactor => MulIsolationOperands {
            route,
            isolated_factor: right,
            moved_factor: left,
        },
    }
}

/// Safety gate for multiplicative isolation: linear-collect fallback is only
/// attempted when RHS also contains the solve variable.
pub fn mul_rhs_contains_variable(ctx: &Context, rhs: ExprId, var: &str) -> bool {
    contains_var(ctx, rhs, var)
}

/// Initial dispatch for isolating division equations `(l / r) = rhs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivIsolationRoute {
    VariableInNumerator,
    VariableInDenominator,
}

/// Derive where the solve variable appears in `(numerator / denominator)`.
pub fn derive_div_isolation_route(
    ctx: &Context,
    numerator: ExprId,
    var: &str,
) -> DivIsolationRoute {
    if contains_var(ctx, numerator, var) {
        DivIsolationRoute::VariableInNumerator
    } else {
        DivIsolationRoute::VariableInDenominator
    }
}

/// Derive which side of `Pow(base, exponent)` contains the solve variable.
pub fn derive_pow_isolation_route(ctx: &Context, base: ExprId, var: &str) -> PowIsolationRoute {
    if contains_var(ctx, base, var) {
        PowIsolationRoute::VariableInBase
    } else {
        PowIsolationRoute::VariableInExponent
    }
}

/// Safety gate for exponent-isolation: logarithmic inversion requires
/// the solve variable to be absent from RHS.
pub fn pow_exponent_rhs_contains_variable(ctx: &Context, rhs: ExprId, var: &str) -> bool {
    contains_var(ctx, rhs, var)
}

/// Shortcut routing for equations with variable in exponent (`base^x = rhs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowExponentShortcut {
    None,
    /// `base^x = base`
    PowerEqualsBase(PowerEqualsBaseRoute),
    /// `base^x = base^n`
    EqualPowBases {
        rhs_exp: ExprId,
    },
}

/// Operational resolution for exponent shortcut routes.
#[derive(Debug, Clone, PartialEq)]
pub enum PowExponentShortcutResolution {
    Continue,
    IsolateExponent { rhs: ExprId, op: RelOp },
    ReturnSolutionSet(SolutionSet),
}

/// Executable action for exponent shortcuts, preserving the originating route.
#[derive(Debug, Clone, PartialEq)]
pub enum PowExponentShortcutAction {
    Continue,
    IsolateExponent {
        shortcut: PowExponentShortcut,
        rhs: ExprId,
        op: RelOp,
    },
    ReturnSolutionSet {
        shortcut: PowExponentShortcut,
        solutions: SolutionSet,
    },
}

/// Narrative category for exponent shortcut didactic steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowExponentShortcutNarrative {
    ZeroBaseExponentPositive,
    NumericBaseExponentOne,
    SymbolicBaseExponentOneNoBudget,
    SymbolicBaseCaseSplit,
    EqualPowBases,
}

/// Normalized execution plan for exponent shortcuts.
#[derive(Debug, Clone, PartialEq)]
pub enum PowExponentShortcutExecutionPlan {
    Continue,
    IsolateExponent {
        rhs: ExprId,
        op: RelOp,
        narrative: PowExponentShortcutNarrative,
        rhs_exponent: Option<ExprId>,
    },
    ReturnSolutionSet {
        solutions: SolutionSet,
        narrative: PowExponentShortcutNarrative,
    },
}

/// Didactic payload for a planned exponent-shortcut step.
#[derive(Debug, Clone, PartialEq)]
pub struct PowExponentShortcutDidacticStep {
    pub description: String,
    pub equation_after: Equation,
}

/// One executable exponent-shortcut item aligned with didactic payload.
#[derive(Debug, Clone, PartialEq)]
pub struct PowExponentShortcutExecutionItem {
    pub equation: Equation,
    pub description: String,
}

impl PowExponentShortcutExecutionItem {
    /// User-facing narration for this execution item.
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Engine-facing action for a normalized exponent shortcut, including optional
/// didactic step payload built in solver-core.
#[derive(Debug, Clone, PartialEq)]
pub enum PowExponentShortcutEngineAction {
    Continue,
    IsolateExponent {
        rhs: ExprId,
        op: RelOp,
        items: Vec<PowExponentShortcutExecutionItem>,
    },
    ReturnSolutionSet {
        solutions: SolutionSet,
        items: Vec<PowExponentShortcutExecutionItem>,
    },
}

/// Collect exponent-shortcut didactic steps in display order.
pub fn collect_pow_exponent_shortcut_didactic_steps(
    action: &PowExponentShortcutEngineAction,
) -> Vec<PowExponentShortcutDidacticStep> {
    match action {
        PowExponentShortcutEngineAction::Continue => vec![],
        PowExponentShortcutEngineAction::IsolateExponent { items, .. }
        | PowExponentShortcutEngineAction::ReturnSolutionSet { items, .. } => items
            .iter()
            .cloned()
            .map(|item| PowExponentShortcutDidacticStep {
                description: item.description,
                equation_after: item.equation,
            })
            .collect(),
    }
}

/// Collect exponent-shortcut execution items in display order.
pub fn collect_pow_exponent_shortcut_execution_items(
    action: &PowExponentShortcutEngineAction,
) -> Vec<PowExponentShortcutExecutionItem> {
    match action {
        PowExponentShortcutEngineAction::Continue => vec![],
        PowExponentShortcutEngineAction::IsolateExponent { items, .. }
        | PowExponentShortcutEngineAction::ReturnSolutionSet { items, .. } => items.clone(),
    }
}

/// Return the first exponent-shortcut execution item, if any.
pub fn first_pow_exponent_shortcut_execution_item(
    action: &PowExponentShortcutEngineAction,
) -> Option<PowExponentShortcutExecutionItem> {
    collect_pow_exponent_shortcut_execution_items(action)
        .into_iter()
        .next()
}

/// Solved outcome for exponent-shortcut action execution.
#[derive(Debug, Clone, PartialEq)]
pub enum PowExponentShortcutSolved<T> {
    Continue,
    Isolated(T),
    ReturnedSolutionSet(SolutionSet),
}

/// Execute exponent-shortcut action with caller-provided isolate callback.
pub fn solve_pow_exponent_shortcut_action_with<E, T, FSolve>(
    action: PowExponentShortcutEngineAction,
    mut solve_isolate: FSolve,
) -> Result<PowExponentShortcutSolved<T>, E>
where
    FSolve: FnMut(ExprId, RelOp) -> Result<T, E>,
{
    match action {
        PowExponentShortcutEngineAction::Continue => Ok(PowExponentShortcutSolved::Continue),
        PowExponentShortcutEngineAction::IsolateExponent { rhs, op, .. } => {
            Ok(PowExponentShortcutSolved::Isolated(solve_isolate(rhs, op)?))
        }
        PowExponentShortcutEngineAction::ReturnSolutionSet { solutions, .. } => {
            Ok(PowExponentShortcutSolved::ReturnedSolutionSet(solutions))
        }
    }
}

/// Solved result for an exponent-shortcut execution pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum PowExponentShortcutPipelineSolved<S> {
    Continue,
    Isolated {
        solution_set: SolutionSet,
        steps: Vec<S>,
    },
    ReturnedSolutionSet {
        solution_set: SolutionSet,
        steps: Vec<S>,
    },
}

/// Execute exponent-shortcut solve + optional first-item dispatch.
pub fn solve_pow_exponent_shortcut_pipeline_with_item<E, S, FSolve, FStep>(
    action: PowExponentShortcutEngineAction,
    include_item: bool,
    mut solve_isolate: FSolve,
    mut map_item_to_step: FStep,
) -> Result<PowExponentShortcutPipelineSolved<S>, E>
where
    FSolve: FnMut(ExprId, RelOp) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(PowExponentShortcutExecutionItem) -> S,
{
    match action {
        PowExponentShortcutEngineAction::Continue => {
            Ok(PowExponentShortcutPipelineSolved::Continue)
        }
        PowExponentShortcutEngineAction::IsolateExponent { rhs, op, items } => {
            let mut steps = Vec::new();
            if include_item {
                if let Some(item) = items.into_iter().next() {
                    steps.push(map_item_to_step(item));
                }
            }
            let (solution_set, mut sub_steps) = solve_isolate(rhs, op)?;
            steps.append(&mut sub_steps);
            Ok(PowExponentShortcutPipelineSolved::Isolated {
                solution_set,
                steps,
            })
        }
        PowExponentShortcutEngineAction::ReturnSolutionSet { solutions, items } => {
            let mut steps = Vec::new();
            if include_item {
                if let Some(item) = items.into_iter().next() {
                    steps.push(map_item_to_step(item));
                }
            }
            Ok(PowExponentShortcutPipelineSolved::ReturnedSolutionSet {
                solution_set: solutions,
                steps,
            })
        }
    }
}

/// Execute exponent-shortcut planning + solve pipeline with optional first-item dispatch.
///
/// This combines `execute_pow_exponent_shortcut_with` and
/// `solve_pow_exponent_shortcut_pipeline_with_item` into a single helper.
#[allow(clippy::too_many_arguments)]
pub fn execute_pow_exponent_shortcut_pipeline_with_item_with<
    E,
    S,
    FReadExpr,
    FPlanAction,
    FBasesEquivalent,
    FRenderExpr,
    FSolve,
    FStep,
>(
    exponent_lhs: ExprId,
    base: ExprId,
    rhs: ExprId,
    original_op: RelOp,
    var: &str,
    base_is_zero: bool,
    base_is_numeric: bool,
    can_branch: bool,
    include_item: bool,
    read_expr: FReadExpr,
    plan_action_from_inputs: FPlanAction,
    bases_equivalent: FBasesEquivalent,
    render_expr: FRenderExpr,
    solve_isolate: FSolve,
    map_item_to_step: FStep,
) -> Result<PowExponentShortcutPipelineSolved<S>, E>
where
    FReadExpr: FnMut(ExprId) -> Expr,
    FPlanAction:
        FnMut(ExprId, RelOp, bool, Option<ExprId>, bool, bool, bool) -> PowExponentShortcutAction,
    FBasesEquivalent: FnMut(ExprId, ExprId) -> bool,
    FRenderExpr: FnMut(ExprId) -> String,
    FSolve: FnMut(ExprId, RelOp) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(PowExponentShortcutExecutionItem) -> S,
{
    let action = execute_pow_exponent_shortcut_with(
        exponent_lhs,
        base,
        rhs,
        original_op,
        var,
        base_is_zero,
        base_is_numeric,
        can_branch,
        read_expr,
        plan_action_from_inputs,
        bases_equivalent,
        render_expr,
    );
    solve_pow_exponent_shortcut_pipeline_with_item(
        action,
        include_item,
        solve_isolate,
        map_item_to_step,
    )
}

/// Execute exponent-shortcut planning + solve pipeline and finalize against
/// caller-owned mutable step buffer.
///
/// Returns `Ok(None)` when shortcut resolution is `Continue`.
#[allow(clippy::too_many_arguments)]
pub fn execute_pow_exponent_shortcut_pipeline_with_item_and_finalize_with_existing_steps_with<
    E,
    S,
    FReadExpr,
    FPlanAction,
    FBasesEquivalent,
    FRenderExpr,
    FSolve,
    FStep,
>(
    exponent_lhs: ExprId,
    base: ExprId,
    rhs: ExprId,
    original_op: RelOp,
    var: &str,
    base_is_zero: bool,
    base_is_numeric: bool,
    can_branch: bool,
    include_item: bool,
    existing_steps: &mut Vec<S>,
    read_expr: FReadExpr,
    plan_action_from_inputs: FPlanAction,
    bases_equivalent: FBasesEquivalent,
    render_expr: FRenderExpr,
    solve_isolate: FSolve,
    map_item_to_step: FStep,
) -> Result<Option<(SolutionSet, Vec<S>)>, E>
where
    FReadExpr: FnMut(ExprId) -> Expr,
    FPlanAction:
        FnMut(ExprId, RelOp, bool, Option<ExprId>, bool, bool, bool) -> PowExponentShortcutAction,
    FBasesEquivalent: FnMut(ExprId, ExprId) -> bool,
    FRenderExpr: FnMut(ExprId) -> String,
    FSolve: FnMut(ExprId, RelOp) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(PowExponentShortcutExecutionItem) -> S,
{
    let solved = execute_pow_exponent_shortcut_pipeline_with_item_with(
        exponent_lhs,
        base,
        rhs,
        original_op,
        var,
        base_is_zero,
        base_is_numeric,
        can_branch,
        include_item,
        read_expr,
        plan_action_from_inputs,
        bases_equivalent,
        render_expr,
        solve_isolate,
        map_item_to_step,
    )?;
    Ok(finalize_pow_exponent_shortcut_pipeline_with_existing_steps(
        solved,
        existing_steps,
    ))
}

/// Merge a shortcut pipeline outcome with caller-owned pre-existing steps.
///
/// Returns `None` when shortcut pipeline indicates `Continue`.
pub fn merge_pow_exponent_shortcut_pipeline_with_existing_steps<S>(
    solved: PowExponentShortcutPipelineSolved<S>,
    existing_steps: Vec<S>,
) -> Option<(SolutionSet, Vec<S>)> {
    match solved {
        PowExponentShortcutPipelineSolved::Continue => None,
        PowExponentShortcutPipelineSolved::Isolated {
            solution_set,
            steps,
        } => {
            let mut merged = steps;
            merged.extend(existing_steps);
            Some((solution_set, merged))
        }
        PowExponentShortcutPipelineSolved::ReturnedSolutionSet {
            solution_set,
            steps,
        } => {
            let mut merged = existing_steps;
            merged.extend(steps);
            Some((solution_set, merged))
        }
    }
}

/// Finalize shortcut pipeline against caller-owned mutable step buffer.
///
/// Returns `None` for `Continue` without consuming `existing_steps`.
pub fn finalize_pow_exponent_shortcut_pipeline_with_existing_steps<S>(
    solved: PowExponentShortcutPipelineSolved<S>,
    existing_steps: &mut Vec<S>,
) -> Option<(SolutionSet, Vec<S>)> {
    if matches!(solved, PowExponentShortcutPipelineSolved::Continue) {
        return None;
    }
    let existing = std::mem::take(existing_steps);
    merge_pow_exponent_shortcut_pipeline_with_existing_steps(solved, existing)
}

/// Solved base-one shortcut (`1^x = rhs`) with didactic payload.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerBaseOneShortcutOutcome {
    pub solutions: SolutionSet,
    pub items: Vec<PowerBaseOneShortcutExecutionItem>,
}

/// One executable base-one shortcut item aligned with didactic payload.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerBaseOneShortcutExecutionItem {
    pub equation: Equation,
    pub description: String,
}

impl PowerBaseOneShortcutExecutionItem {
    /// User-facing narration for this execution item.
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Collect base-one shortcut didactic steps in display order.
pub fn collect_power_base_one_shortcut_didactic_steps(
    outcome: &PowerBaseOneShortcutOutcome,
) -> Vec<PowExponentShortcutDidacticStep> {
    outcome
        .items
        .iter()
        .cloned()
        .map(|item| PowExponentShortcutDidacticStep {
            description: item.description,
            equation_after: item.equation,
        })
        .collect()
}

/// Collect base-one shortcut execution items in display order.
pub fn collect_power_base_one_shortcut_execution_items(
    outcome: &PowerBaseOneShortcutOutcome,
) -> Vec<PowerBaseOneShortcutExecutionItem> {
    outcome.items.clone()
}

/// Return the first base-one shortcut execution item, if any.
pub fn first_power_base_one_shortcut_execution_item(
    outcome: &PowerBaseOneShortcutOutcome,
) -> Option<PowerBaseOneShortcutExecutionItem> {
    collect_power_base_one_shortcut_execution_items(outcome)
        .into_iter()
        .next()
}

/// Solved result for a base-one shortcut pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerBaseOneShortcutPipelineSolved<S> {
    pub solution_set: SolutionSet,
    pub steps: Vec<S>,
}

/// Execute base-one shortcut item dispatch, optionally mapping first didactic
/// item to caller step payload.
pub fn solve_power_base_one_shortcut_pipeline_with_item<S, FStep>(
    outcome: PowerBaseOneShortcutOutcome,
    include_item: bool,
    mut map_item_to_step: FStep,
) -> PowerBaseOneShortcutPipelineSolved<S>
where
    FStep: FnMut(PowerBaseOneShortcutExecutionItem) -> S,
{
    let PowerBaseOneShortcutOutcome { solutions, items } = outcome;
    let mut steps = Vec::new();
    if include_item {
        if let Some(item) = items.into_iter().next() {
            steps.push(map_item_to_step(item));
        }
    }
    PowerBaseOneShortcutPipelineSolved {
        solution_set: solutions,
        steps,
    }
}

/// Resolve and execute base-one shortcut pipeline for equations `base^x op rhs`.
#[allow(clippy::too_many_arguments)]
pub fn execute_power_base_one_shortcut_pipeline_with_item_for_pow_with<S, FRender, FStep>(
    ctx: &Context,
    base: ExprId,
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    include_item: bool,
    render_expr: FRender,
    map_item_to_step: FStep,
) -> Option<PowerBaseOneShortcutPipelineSolved<S>>
where
    FRender: FnMut(&Context, ExprId) -> String,
    FStep: FnMut(PowerBaseOneShortcutExecutionItem) -> S,
{
    let outcome =
        resolve_power_base_one_shortcut_for_pow_with(ctx, base, lhs, rhs, op, render_expr)?;
    Some(solve_power_base_one_shortcut_pipeline_with_item(
        outcome,
        include_item,
        map_item_to_step,
    ))
}

/// Resolve and execute base-one shortcut pipeline, then append solved steps
/// after caller-owned existing steps.
#[allow(clippy::too_many_arguments)]
pub fn execute_power_base_one_shortcut_pipeline_with_item_for_pow_and_merge_with_existing_steps_with<
    S,
    FRender,
    FStep,
>(
    ctx: &Context,
    base: ExprId,
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    include_item: bool,
    existing_steps: Vec<S>,
    render_expr: FRender,
    map_item_to_step: FStep,
) -> Option<(SolutionSet, Vec<S>)>
where
    FRender: FnMut(&Context, ExprId) -> String,
    FStep: FnMut(PowerBaseOneShortcutExecutionItem) -> S,
{
    let solved = execute_power_base_one_shortcut_pipeline_with_item_for_pow_with(
        ctx,
        base,
        lhs,
        rhs,
        op,
        include_item,
        render_expr,
        map_item_to_step,
    )?;
    Some(merge_solved_with_existing_steps_append(
        (solved.solution_set, solved.steps),
        existing_steps,
    ))
}

/// Resolve and execute base-one shortcut pipeline, then finalize against
/// caller-owned mutable step buffer.
///
/// Returns `None` when the base-one shortcut does not apply.
#[allow(clippy::too_many_arguments)]
pub fn execute_power_base_one_shortcut_pipeline_with_item_for_pow_and_finalize_with_existing_steps_with<
    S,
    FRender,
    FStep,
>(
    ctx: &Context,
    base: ExprId,
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    include_item: bool,
    existing_steps: &mut Vec<S>,
    render_expr: FRender,
    map_item_to_step: FStep,
) -> Option<(SolutionSet, Vec<S>)>
where
    FRender: FnMut(&Context, ExprId) -> String,
    FStep: FnMut(PowerBaseOneShortcutExecutionItem) -> S,
{
    let solved = execute_power_base_one_shortcut_pipeline_with_item_for_pow_with(
        ctx,
        base,
        lhs,
        rhs,
        op,
        include_item,
        render_expr,
        map_item_to_step,
    )?;
    Some(merge_solved_with_existing_steps_append(
        (solved.solution_set, solved.steps),
        std::mem::take(existing_steps),
    ))
}

/// Didactic payload for logarithmic isolation of exponent equations.
#[derive(Debug, Clone, PartialEq)]
pub struct PowExponentLogIsolationStep {
    pub description: String,
    pub equation_after: Equation,
}

/// Combined rewrite + didactic step for logarithmic exponent isolation.
#[derive(Debug, Clone, PartialEq)]
pub struct PowExponentLogIsolationRewritePlan {
    pub equation: Equation,
    pub items: Vec<PowExponentLogIsolationExecutionItem>,
}

/// One executable logarithmic-isolation item aligned with didactic payload.
#[derive(Debug, Clone, PartialEq)]
pub struct PowExponentLogIsolationExecutionItem {
    pub equation: Equation,
    pub description: String,
}

impl PowExponentLogIsolationExecutionItem {
    /// User-facing narration for this execution item.
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Collect logarithmic-isolation didactic steps in display order.
pub fn collect_pow_exponent_log_isolation_didactic_steps(
    plan: &PowExponentLogIsolationRewritePlan,
) -> Vec<PowExponentLogIsolationStep> {
    plan.items
        .iter()
        .cloned()
        .map(|item| PowExponentLogIsolationStep {
            description: item.description,
            equation_after: item.equation,
        })
        .collect()
}

/// Collect logarithmic-isolation execution items in display order.
pub fn collect_pow_exponent_log_isolation_execution_items(
    plan: &PowExponentLogIsolationRewritePlan,
) -> Vec<PowExponentLogIsolationExecutionItem> {
    plan.items.clone()
}

/// Return the first logarithmic-isolation execution item, if any.
pub fn first_pow_exponent_log_isolation_execution_item(
    plan: &PowExponentLogIsolationRewritePlan,
) -> Option<PowExponentLogIsolationExecutionItem> {
    collect_pow_exponent_log_isolation_execution_items(plan)
        .into_iter()
        .next()
}

/// Execution plan for guarded logarithmic isolation in exponent equations.
#[derive(Debug, Clone, PartialEq)]
pub struct GuardedPowExponentLogExecutionPlan {
    pub rewrite: PowExponentLogIsolationRewritePlan,
    pub followup_success: TermIsolationExecutionItem,
    pub followup_residual: TermIsolationExecutionItem,
}

impl GuardedPowExponentLogExecutionPlan {
    /// Follow-up item after guarded solve attempt.
    pub fn followup_item(&self, guarded_solve_succeeded: bool) -> TermIsolationExecutionItem {
        if guarded_solve_succeeded {
            self.followup_success.clone()
        } else {
            self.followup_residual.clone()
        }
    }
}

/// Executed guarded logarithmic-isolation branch with follow-up and optional
/// guarded-solve solution set.
#[derive(Debug, Clone, PartialEq)]
pub struct GuardedPowExponentLogExecution {
    pub rewrite: PowExponentLogIsolationRewritePlan,
    pub followup: TermIsolationExecutionItem,
    pub guarded_solutions: Option<SolutionSet>,
}

/// Solved payload for one logarithmic exponent-isolation rewrite.
#[derive(Debug, Clone, PartialEq)]
pub struct PowExponentLogIsolationSolved<T> {
    pub rewrite: PowExponentLogIsolationRewritePlan,
    pub solved: T,
}

/// Structured outcome for unsupported logarithmic rewrites.
#[derive(Debug, Clone, PartialEq)]
pub enum LogUnsupportedOutcome<'a> {
    ResidualBudgetExhausted {
        message: &'a str,
        solutions: SolutionSet,
    },
    Guarded {
        message: &'a str,
        missing_conditions: &'a [LogAssumption],
        guard: ConditionSet,
        residual: ExprId,
    },
}

/// Blocked-hint payload for guarded log isolation requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogBlockedHintRecord {
    pub assumption: LogAssumption,
    pub expr_id: ExprId,
    pub rule: &'static str,
    pub suggestion: &'static str,
}

/// Build blocked-hint records for missing log assumptions in guarded mode.
pub fn collect_guarded_log_blocked_hints(
    missing_conditions: &[LogAssumption],
    base: ExprId,
    rhs: ExprId,
) -> Vec<LogBlockedHintRecord> {
    missing_conditions
        .iter()
        .copied()
        .map(|assumption| LogBlockedHintRecord {
            assumption,
            expr_id: crate::log_domain::assumption_target_expr(assumption, base, rhs),
            rule: "Take log of both sides",
            suggestion: "use `semantics set domain assume`",
        })
        .collect()
}

/// Planned execution for unsupported logarithmic isolation outcomes.
#[derive(Debug, Clone, PartialEq)]
pub enum PowExponentLogUnsupportedExecution {
    Residual {
        item: TermIsolationExecutionItem,
        solutions: SolutionSet,
    },
    Guarded {
        blocked_hints: Vec<LogBlockedHintRecord>,
        plan: GuardedPowExponentLogExecutionPlan,
        guard: ConditionSet,
        residual: ExprId,
    },
}

/// Executed unsupported logarithmic isolation route.
///
/// - `Residual` forwards a prebuilt residual item + solution set.
/// - `Guarded` executes the guarded rewrite plan and materializes final
///   conditional-or-residual solution set, plus didactic payload.
#[derive(Debug, Clone, PartialEq)]
pub enum PowExponentLogUnsupportedSolvedExecution {
    Residual {
        item: TermIsolationExecutionItem,
        solutions: SolutionSet,
    },
    Guarded {
        blocked_hints: Vec<LogBlockedHintRecord>,
        rewrite_item: Option<PowExponentLogIsolationExecutionItem>,
        followup_item: TermIsolationExecutionItem,
        solutions: SolutionSet,
    },
}

/// Convert one logarithmic-isolation rewrite execution item to term-isolation shape.
pub fn pow_exponent_log_execution_item_as_term_item(
    item: PowExponentLogIsolationExecutionItem,
) -> TermIsolationExecutionItem {
    TermIsolationExecutionItem {
        description: item.description,
        equation: item.equation,
    }
}

/// Plan unsupported logarithmic isolation execution for exponent equations.
///
/// This maps the low-level unsupported outcome to engine-facing execution data:
/// either a residual-budget terminal item or a guarded-log branch plan with
/// blocked-hint payload.
#[allow(clippy::too_many_arguments)]
pub fn plan_pow_exponent_log_unsupported_execution_with<'a, F>(
    ctx: &mut Context,
    outcome: LogUnsupportedOutcome<'a>,
    exponent: ExprId,
    base: ExprId,
    rhs: ExprId,
    op: RelOp,
    source_equation: Equation,
    render_expr: F,
) -> PowExponentLogUnsupportedExecution
where
    F: FnMut(&Context, ExprId) -> String,
{
    match outcome {
        LogUnsupportedOutcome::ResidualBudgetExhausted { message, solutions } => {
            PowExponentLogUnsupportedExecution::Residual {
                item: build_residual_budget_exhausted_item(message, source_equation),
                solutions,
            }
        }
        LogUnsupportedOutcome::Guarded {
            message,
            missing_conditions,
            guard,
            residual,
        } => PowExponentLogUnsupportedExecution::Guarded {
            blocked_hints: collect_guarded_log_blocked_hints(missing_conditions, base, rhs),
            plan: plan_guarded_pow_exponent_log_execution_with(
                ctx,
                exponent,
                base,
                rhs,
                op,
                message,
                source_equation,
                render_expr,
            ),
            guard,
            residual,
        },
    }
}

/// Resolve unsupported log decision and materialize engine execution payload
/// for exponent isolation in one step.
#[allow(clippy::too_many_arguments)]
pub fn plan_pow_exponent_log_unsupported_execution_from_decision_with<F>(
    ctx: &mut Context,
    decision: &LogSolveDecision,
    can_branch: bool,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    exponent: ExprId,
    base: ExprId,
    rhs_expr: ExprId,
    op: RelOp,
    source_equation: Equation,
    render_expr: F,
) -> Option<PowExponentLogUnsupportedExecution>
where
    F: FnMut(&Context, ExprId) -> String,
{
    let outcome =
        resolve_log_unsupported_outcome(ctx, decision, can_branch, lhs, rhs, var, base, rhs_expr)?;
    Some(plan_pow_exponent_log_unsupported_execution_with(
        ctx,
        outcome,
        exponent,
        base,
        rhs_expr,
        op,
        source_equation,
        render_expr,
    ))
}

/// Routing for isolation when the variable is in the power base (`B^E = RHS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowBaseIsolationRoute {
    /// `E` is even and `RHS` is known negative.
    EvenExponentNegativeRhsImpossible,
    /// `E` is even and root isolation must use absolute value.
    EvenExponentUseAbsRoot,
    /// General root isolation, optionally flipping inequality for negative exponent.
    GeneralRoot {
        flip_inequality_for_negative_exponent: bool,
    },
}

/// Execution plan for isolating equations of the form `B^E op RHS`.
#[derive(Debug, Clone, PartialEq)]
pub enum PowBaseIsolationPlan {
    ReturnSolutionSet {
        route: PowBaseIsolationRoute,
        equation: Equation,
        solutions: SolutionSet,
    },
    IsolateBase {
        route: PowBaseIsolationRoute,
        equation: Equation,
        op: RelOp,
        use_abs_root: bool,
    },
}

/// Didactic payload for a planned base-isolation step.
#[derive(Debug, Clone, PartialEq)]
pub struct PowBaseIsolationDidacticStep {
    pub description: String,
    pub equation_after: Equation,
}

/// One executable base-isolation item aligned with didactic payload.
#[derive(Debug, Clone, PartialEq)]
pub struct PowBaseIsolationExecutionItem {
    pub equation: Equation,
    pub description: String,
}

impl PowBaseIsolationExecutionItem {
    /// User-facing narration for this execution item.
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Engine-facing action for base-isolation planning with didactic payload.
#[derive(Debug, Clone, PartialEq)]
pub enum PowBaseIsolationEngineAction {
    ReturnSolutionSet {
        solutions: SolutionSet,
        items: Vec<PowBaseIsolationExecutionItem>,
    },
    IsolateBase {
        lhs: ExprId,
        rhs: ExprId,
        op: RelOp,
        items: Vec<PowBaseIsolationExecutionItem>,
    },
}

/// Collect base-isolation didactic steps in display order.
pub fn collect_pow_base_isolation_didactic_steps(
    action: &PowBaseIsolationEngineAction,
) -> Vec<PowBaseIsolationDidacticStep> {
    match action {
        PowBaseIsolationEngineAction::ReturnSolutionSet { items, .. }
        | PowBaseIsolationEngineAction::IsolateBase { items, .. } => items
            .iter()
            .cloned()
            .map(|item| PowBaseIsolationDidacticStep {
                description: item.description,
                equation_after: item.equation,
            })
            .collect(),
    }
}

/// Collect base-isolation execution items in display order.
pub fn collect_pow_base_isolation_execution_items(
    action: &PowBaseIsolationEngineAction,
) -> Vec<PowBaseIsolationExecutionItem> {
    match action {
        PowBaseIsolationEngineAction::ReturnSolutionSet { items, .. }
        | PowBaseIsolationEngineAction::IsolateBase { items, .. } => items.clone(),
    }
}

/// Return the first base-isolation execution item, if any.
pub fn first_pow_base_isolation_execution_item(
    action: &PowBaseIsolationEngineAction,
) -> Option<PowBaseIsolationExecutionItem> {
    collect_pow_base_isolation_execution_items(action)
        .into_iter()
        .next()
}

/// Solved outcome for base-isolation action execution.
#[derive(Debug, Clone, PartialEq)]
pub enum PowBaseIsolationSolved<T> {
    ReturnedSolutionSet(SolutionSet),
    Isolated(T),
}

/// Execute base-isolation action with caller-provided isolate callback.
pub fn solve_pow_base_isolation_action_with<E, T, FSolve>(
    action: PowBaseIsolationEngineAction,
    mut solve_isolate: FSolve,
) -> Result<PowBaseIsolationSolved<T>, E>
where
    FSolve: FnMut(ExprId, ExprId, RelOp) -> Result<T, E>,
{
    match action {
        PowBaseIsolationEngineAction::ReturnSolutionSet { solutions, .. } => {
            Ok(PowBaseIsolationSolved::ReturnedSolutionSet(solutions))
        }
        PowBaseIsolationEngineAction::IsolateBase { lhs, rhs, op, .. } => Ok(
            PowBaseIsolationSolved::Isolated(solve_isolate(lhs, rhs, op)?),
        ),
    }
}

/// Solved result for a base-isolation execution pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum PowBaseIsolationPipelineSolved<S> {
    ReturnedSolutionSet {
        solution_set: SolutionSet,
        steps: Vec<S>,
    },
    Isolated {
        solution_set: SolutionSet,
        steps: Vec<S>,
    },
}

/// Execute base-isolation solve + optional first-item dispatch.
pub fn solve_pow_base_isolation_pipeline_with_item<E, S, FSolve, FStep>(
    action: PowBaseIsolationEngineAction,
    include_item: bool,
    mut solve_isolate: FSolve,
    mut map_item_to_step: FStep,
) -> Result<PowBaseIsolationPipelineSolved<S>, E>
where
    FSolve: FnMut(ExprId, ExprId, RelOp) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(PowBaseIsolationExecutionItem) -> S,
{
    match action {
        PowBaseIsolationEngineAction::ReturnSolutionSet { solutions, items } => {
            let mut steps = Vec::new();
            if include_item {
                if let Some(item) = items.into_iter().next() {
                    steps.push(map_item_to_step(item));
                }
            }
            Ok(PowBaseIsolationPipelineSolved::ReturnedSolutionSet {
                solution_set: solutions,
                steps,
            })
        }
        PowBaseIsolationEngineAction::IsolateBase {
            lhs,
            rhs,
            op,
            items,
        } => {
            let mut steps = Vec::new();
            if include_item {
                if let Some(item) = items.into_iter().next() {
                    steps.push(map_item_to_step(item));
                }
            }
            let (solution_set, mut sub_steps) = solve_isolate(lhs, rhs, op)?;
            steps.append(&mut sub_steps);
            Ok(PowBaseIsolationPipelineSolved::Isolated {
                solution_set,
                steps,
            })
        }
    }
}

/// Execute base-isolation pipeline from a deferred action builder.
pub fn execute_pow_base_isolation_pipeline_with_item_with<E, S, FPlan, FSolve, FStep>(
    include_item: bool,
    plan_action: FPlan,
    solve_isolate: FSolve,
    map_item_to_step: FStep,
) -> Result<PowBaseIsolationPipelineSolved<S>, E>
where
    FPlan: FnOnce() -> PowBaseIsolationEngineAction,
    FSolve: FnMut(ExprId, ExprId, RelOp) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(PowBaseIsolationExecutionItem) -> S,
{
    let action = plan_action();
    solve_pow_base_isolation_pipeline_with_item(
        action,
        include_item,
        solve_isolate,
        map_item_to_step,
    )
}

/// Merge a base-isolation pipeline outcome with caller-owned pre-existing steps.
pub fn merge_pow_base_isolation_pipeline_with_existing_steps<S>(
    solved: PowBaseIsolationPipelineSolved<S>,
    existing_steps: Vec<S>,
) -> (SolutionSet, Vec<S>) {
    match solved {
        PowBaseIsolationPipelineSolved::ReturnedSolutionSet {
            solution_set,
            steps,
        } => {
            let mut merged = existing_steps;
            merged.extend(steps);
            (solution_set, merged)
        }
        PowBaseIsolationPipelineSolved::Isolated {
            solution_set,
            steps,
        } => {
            let mut merged = steps;
            merged.extend(existing_steps);
            (solution_set, merged)
        }
    }
}

/// Execute and merge base-isolation pipeline with caller-owned pre-existing steps.
pub fn execute_pow_base_isolation_pipeline_with_item_and_merge_with_existing_steps_with<
    E,
    S,
    FPlan,
    FSolve,
    FStep,
>(
    include_item: bool,
    existing_steps: Vec<S>,
    plan_action: FPlan,
    solve_isolate: FSolve,
    map_item_to_step: FStep,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FPlan: FnOnce() -> PowBaseIsolationEngineAction,
    FSolve: FnMut(ExprId, ExprId, RelOp) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(PowBaseIsolationExecutionItem) -> S,
{
    let solved = execute_pow_base_isolation_pipeline_with_item_with(
        include_item,
        plan_action,
        solve_isolate,
        map_item_to_step,
    )?;
    Ok(merge_pow_base_isolation_pipeline_with_existing_steps(
        solved,
        existing_steps,
    ))
}

/// Didactic payload for one branch produced by absolute-value splitting.
#[derive(Debug, Clone, PartialEq)]
pub struct AbsSplitDidacticStep {
    pub description: String,
    pub equation_after: Equation,
}

/// Shortcut outcomes for `1^x = rhs`-style equations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerBaseOneShortcut {
    NotApplicable,
    AllReals,
    Empty,
}

/// Classify the simplified variable-free residual.
pub fn classify_var_free_difference(ctx: &Context, diff: ExprId) -> VarFreeDiffKind {
    match ctx.get(diff) {
        Expr::Number(n) if *n == num_rational::BigRational::from_integer(0.into()) => {
            VarFreeDiffKind::IdentityZero
        }
        Expr::Number(_) => VarFreeDiffKind::ContradictionNonZero,
        _ => VarFreeDiffKind::Constraint,
    }
}

/// Resolve variable-eliminated residuals into terminal solve outcomes.
///
/// This consolidates identity/contradiction handling and, for symbolic
/// constraints, produces the didactic step payload (`diff = 0`).
pub fn resolve_var_eliminated_outcome_with<F>(
    ctx: &mut Context,
    diff: ExprId,
    var: &str,
    mut render_expr: F,
) -> VarEliminatedSolveOutcome
where
    F: FnMut(&Context, ExprId) -> String,
{
    match classify_var_free_difference(ctx, diff) {
        VarFreeDiffKind::IdentityZero => VarEliminatedSolveOutcome::IdentityAllReals,
        VarFreeDiffKind::ContradictionNonZero => VarEliminatedSolveOutcome::ContradictionEmpty,
        VarFreeDiffKind::Constraint => {
            let diff_display = render_expr(ctx, diff);
            VarEliminatedSolveOutcome::ConstraintAllReals {
                description: variable_canceled_constraint_message(var, &diff_display),
                equation_after: build_zero_constraint_equation(ctx, diff),
            }
        }
    }
}

/// Map variable-eliminated constraint payload into optional caller-owned steps.
pub fn solve_var_eliminated_constraint_pipeline_with_item<S, FStep>(
    description: String,
    equation_after: Equation,
    include_item: bool,
    mut map_item_to_step: FStep,
) -> Vec<S>
where
    FStep: FnMut(String, Equation) -> S,
{
    if include_item {
        vec![map_item_to_step(description, equation_after)]
    } else {
        vec![]
    }
}

/// Solved payload for variable-eliminated outcome pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum VarEliminatedOutcomePipelineSolved<S> {
    IdentityAllReals,
    ContradictionEmpty,
    ConstraintAllReals { steps: Vec<S> },
}

/// Resolve variable-eliminated residuals and optionally materialize one
/// constraint step.
pub fn solve_var_eliminated_outcome_pipeline_with<S, FRender, FStep>(
    ctx: &mut Context,
    diff: ExprId,
    var: &str,
    include_item: bool,
    mut render_expr: FRender,
    map_item_to_step: FStep,
) -> VarEliminatedOutcomePipelineSolved<S>
where
    FRender: FnMut(&Context, ExprId) -> String,
    FStep: FnMut(String, Equation) -> S,
{
    match resolve_var_eliminated_outcome_with(ctx, diff, var, &mut render_expr) {
        VarEliminatedSolveOutcome::IdentityAllReals => {
            VarEliminatedOutcomePipelineSolved::IdentityAllReals
        }
        VarEliminatedSolveOutcome::ContradictionEmpty => {
            VarEliminatedOutcomePipelineSolved::ContradictionEmpty
        }
        VarEliminatedSolveOutcome::ConstraintAllReals {
            description,
            equation_after,
        } => {
            let steps = solve_var_eliminated_constraint_pipeline_with_item(
                description,
                equation_after,
                include_item,
                map_item_to_step,
            );
            VarEliminatedOutcomePipelineSolved::ConstraintAllReals { steps }
        }
    }
}

/// Solve outcome for `B^E op RHS` when `E` is even and `RHS` is proven negative.
pub fn even_power_negative_rhs_outcome(op: RelOp) -> SolutionSet {
    match op {
        RelOp::Eq => SolutionSet::Empty,
        RelOp::Gt | RelOp::Geq | RelOp::Neq => SolutionSet::AllReals,
        RelOp::Lt | RelOp::Leq => SolutionSet::Empty,
    }
}

/// Outcome for `1^x op rhs` in real arithmetic.
pub fn power_base_one_outcome(rhs_is_one: bool) -> SolutionSet {
    if rhs_is_one {
        SolutionSet::AllReals
    } else {
        SolutionSet::Empty
    }
}

/// Classify whether the base-one shortcut applies and which outcome it yields.
pub fn classify_power_base_one_shortcut(
    base_is_one: bool,
    rhs_is_one: bool,
) -> PowerBaseOneShortcut {
    if !base_is_one {
        return PowerBaseOneShortcut::NotApplicable;
    }
    if rhs_is_one {
        PowerBaseOneShortcut::AllReals
    } else {
        PowerBaseOneShortcut::Empty
    }
}

/// Convert shortcut classification to solution set, if applicable.
pub fn power_base_one_shortcut_solutions(shortcut: PowerBaseOneShortcut) -> Option<SolutionSet> {
    match shortcut {
        PowerBaseOneShortcut::NotApplicable => None,
        PowerBaseOneShortcut::AllReals => Some(SolutionSet::AllReals),
        PowerBaseOneShortcut::Empty => Some(SolutionSet::Empty),
    }
}

/// Build didactic narration for base-one exponential shortcut (`1^x = rhs`).
pub fn power_base_one_shortcut_message(
    shortcut: PowerBaseOneShortcut,
    rhs_display: &str,
) -> Option<String> {
    match shortcut {
        PowerBaseOneShortcut::NotApplicable => None,
        PowerBaseOneShortcut::AllReals => {
            Some("1^x = 1 for all x -> any real number is a solution".to_string())
        }
        PowerBaseOneShortcut::Empty => Some(format!(
            "1^x = 1 for all x, but RHS = {} != 1 -> no solution",
            rhs_display
        )),
    }
}

/// Resolve `1^x = rhs` shortcut and build an optional didactic step payload.
pub fn resolve_power_base_one_shortcut_with<F>(
    base_is_one: bool,
    rhs_is_one: bool,
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    mut render_expr: F,
) -> Option<PowerBaseOneShortcutOutcome>
where
    F: FnMut(ExprId) -> String,
{
    let shortcut = classify_power_base_one_shortcut(base_is_one, rhs_is_one);
    let solutions = power_base_one_shortcut_solutions(shortcut)?;
    let rhs_desc = render_expr(rhs);
    let description =
        power_base_one_shortcut_message(shortcut, &rhs_desc).expect("shortcut was applicable");
    let items = vec![PowerBaseOneShortcutExecutionItem {
        equation: Equation { lhs, rhs, op },
        description,
    }];
    Some(PowerBaseOneShortcutOutcome { solutions, items })
}

/// Resolve `base^x = rhs` base-one shortcut directly from equation inputs.
///
/// This helper computes `base == 1` and `rhs == 1` internally, then delegates
/// to [`resolve_power_base_one_shortcut_with`].
pub fn resolve_power_base_one_shortcut_for_pow_with<F>(
    ctx: &Context,
    base: ExprId,
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    mut render_expr: F,
) -> Option<PowerBaseOneShortcutOutcome>
where
    F: FnMut(&Context, ExprId) -> String,
{
    resolve_power_base_one_shortcut_with(
        crate::isolation_utils::is_numeric_one(ctx, base),
        crate::isolation_utils::is_numeric_one(ctx, rhs),
        lhs,
        rhs,
        op,
        |id| render_expr(ctx, id),
    )
}

/// Decide how to handle `base^x = base`.
pub fn classify_power_equals_base_route(
    base_is_zero: bool,
    base_is_numeric: bool,
    can_branch: bool,
) -> PowerEqualsBaseRoute {
    if base_is_zero {
        PowerEqualsBaseRoute::ExponentGreaterThanZero
    } else if base_is_numeric {
        PowerEqualsBaseRoute::ExponentEqualsOneNumericBase
    } else if can_branch {
        PowerEqualsBaseRoute::SymbolicCaseSplit
    } else {
        PowerEqualsBaseRoute::ExponentEqualsOneNoBranchBudget
    }
}

/// Compute shortcut inputs for exponent isolation (`base^x = rhs`) from `rhs` shape.
///
/// The caller provides `same_base`, which decides whether the original base and
/// a candidate expression are equivalent under its local notion of equivalence.
pub fn detect_pow_exponent_shortcut_inputs<F>(
    rhs: ExprId,
    rhs_expr: &Expr,
    mut same_base: F,
) -> (bool, Option<ExprId>)
where
    F: FnMut(ExprId) -> bool,
{
    let bases_equal = same_base(rhs);
    let rhs_pow_base_equal = match rhs_expr {
        Expr::Pow(rhs_base, rhs_exp) if same_base(*rhs_base) => Some(*rhs_exp),
        _ => None,
    };
    (bases_equal, rhs_pow_base_equal)
}

pub fn shortcut_bases_equivalent_with<F>(
    base: ExprId,
    candidate: ExprId,
    mut equivalent_nontrivial: F,
) -> bool
where
    F: FnMut(ExprId, ExprId) -> bool,
{
    base == candidate || equivalent_nontrivial(base, candidate)
}

/// Compare shortcut bases by first short-circuiting identical ids and then
/// using simplified-difference equivalence for non-identical candidates.
pub fn shortcut_bases_equivalent_by_difference_with<FBuildSub, FSimplify, FIsZero>(
    base: ExprId,
    candidate: ExprId,
    mut build_sub: FBuildSub,
    mut simplify: FSimplify,
    mut is_zero: FIsZero,
) -> bool
where
    FBuildSub: FnMut(ExprId, ExprId) -> ExprId,
    FSimplify: FnMut(ExprId) -> ExprId,
    FIsZero: FnMut(ExprId) -> bool,
{
    shortcut_bases_equivalent_with(base, candidate, |left, right| {
        crate::isolation_utils::are_equivalent_by_difference_with(
            left,
            right,
            &mut build_sub,
            &mut simplify,
            &mut is_zero,
        )
    })
}

/// Classify base flags used by exponent-shortcut planning.
pub fn classify_pow_exponent_base_flags(ctx: &Context, base: ExprId) -> (bool, bool) {
    (
        crate::isolation_utils::is_numeric_zero(ctx, base),
        matches!(ctx.get(base), Expr::Number(_)),
    )
}

/// Classify exponent-isolation shortcuts before generic logarithmic isolation.
pub fn classify_pow_exponent_shortcut(
    op: RelOp,
    bases_equal: bool,
    rhs_pow_base_equal: Option<ExprId>,
    base_is_zero: bool,
    base_is_numeric: bool,
    can_branch: bool,
) -> PowExponentShortcut {
    if bases_equal && op == RelOp::Eq {
        return PowExponentShortcut::PowerEqualsBase(classify_power_equals_base_route(
            base_is_zero,
            base_is_numeric,
            can_branch,
        ));
    }
    if let Some(rhs_exp) = rhs_pow_base_equal {
        return PowExponentShortcut::EqualPowBases { rhs_exp };
    }
    PowExponentShortcut::None
}

/// Resolve a classified exponent shortcut into either:
/// - an equation target for exponent isolation, or
/// - a terminal solution set.
pub fn resolve_pow_exponent_shortcut(
    ctx: &mut Context,
    shortcut: PowExponentShortcut,
    base: ExprId,
    op: RelOp,
) -> PowExponentShortcutResolution {
    match shortcut {
        PowExponentShortcut::PowerEqualsBase(route) => match route {
            PowerEqualsBaseRoute::ExponentGreaterThanZero => {
                PowExponentShortcutResolution::IsolateExponent {
                    rhs: ctx.num(0),
                    op: RelOp::Gt,
                }
            }
            PowerEqualsBaseRoute::ExponentEqualsOneNumericBase
            | PowerEqualsBaseRoute::ExponentEqualsOneNoBranchBudget => {
                PowExponentShortcutResolution::IsolateExponent {
                    rhs: ctx.num(1),
                    op,
                }
            }
            PowerEqualsBaseRoute::SymbolicCaseSplit => {
                PowExponentShortcutResolution::ReturnSolutionSet(
                    power_equals_base_symbolic_outcome(ctx, base),
                )
            }
        },
        PowExponentShortcut::EqualPowBases { rhs_exp } => {
            PowExponentShortcutResolution::IsolateExponent { rhs: rhs_exp, op }
        }
        PowExponentShortcut::None => PowExponentShortcutResolution::Continue,
    }
}

/// Build an executable shortcut action by pairing classification and resolution.
pub fn plan_pow_exponent_shortcut_action(
    ctx: &mut Context,
    shortcut: PowExponentShortcut,
    base: ExprId,
    op: RelOp,
) -> PowExponentShortcutAction {
    match resolve_pow_exponent_shortcut(ctx, shortcut, base, op) {
        PowExponentShortcutResolution::Continue => PowExponentShortcutAction::Continue,
        PowExponentShortcutResolution::IsolateExponent { rhs, op } => {
            PowExponentShortcutAction::IsolateExponent { shortcut, rhs, op }
        }
        PowExponentShortcutResolution::ReturnSolutionSet(solutions) => {
            PowExponentShortcutAction::ReturnSolutionSet {
                shortcut,
                solutions,
            }
        }
    }
}

/// Convert shortcut action into a normalized execution plan for solver pipelines.
pub fn build_pow_exponent_shortcut_execution_plan(
    action: PowExponentShortcutAction,
) -> PowExponentShortcutExecutionPlan {
    match action {
        PowExponentShortcutAction::Continue => PowExponentShortcutExecutionPlan::Continue,
        PowExponentShortcutAction::IsolateExponent { shortcut, rhs, op } => match shortcut {
            PowExponentShortcut::PowerEqualsBase(PowerEqualsBaseRoute::ExponentGreaterThanZero) => {
                PowExponentShortcutExecutionPlan::IsolateExponent {
                    rhs,
                    op,
                    narrative: PowExponentShortcutNarrative::ZeroBaseExponentPositive,
                    rhs_exponent: None,
                }
            }
            PowExponentShortcut::PowerEqualsBase(
                PowerEqualsBaseRoute::ExponentEqualsOneNumericBase,
            ) => PowExponentShortcutExecutionPlan::IsolateExponent {
                rhs,
                op,
                narrative: PowExponentShortcutNarrative::NumericBaseExponentOne,
                rhs_exponent: None,
            },
            PowExponentShortcut::PowerEqualsBase(
                PowerEqualsBaseRoute::ExponentEqualsOneNoBranchBudget,
            ) => PowExponentShortcutExecutionPlan::IsolateExponent {
                rhs,
                op,
                narrative: PowExponentShortcutNarrative::SymbolicBaseExponentOneNoBudget,
                rhs_exponent: None,
            },
            PowExponentShortcut::EqualPowBases { rhs_exp } => {
                PowExponentShortcutExecutionPlan::IsolateExponent {
                    rhs,
                    op,
                    narrative: PowExponentShortcutNarrative::EqualPowBases,
                    rhs_exponent: Some(rhs_exp),
                }
            }
            PowExponentShortcut::PowerEqualsBase(PowerEqualsBaseRoute::SymbolicCaseSplit)
            | PowExponentShortcut::None => PowExponentShortcutExecutionPlan::Continue,
        },
        PowExponentShortcutAction::ReturnSolutionSet {
            shortcut,
            solutions,
        } => match shortcut {
            PowExponentShortcut::PowerEqualsBase(PowerEqualsBaseRoute::SymbolicCaseSplit) => {
                PowExponentShortcutExecutionPlan::ReturnSolutionSet {
                    solutions,
                    narrative: PowExponentShortcutNarrative::SymbolicBaseCaseSplit,
                }
            }
            _ => PowExponentShortcutExecutionPlan::Continue,
        },
    }
}

/// Map a normalized shortcut plan to an engine action with didactic payload.
///
/// This keeps the `base^x` shortcut narration/equation wiring centralized in
/// solver-core while allowing caller-controlled expression rendering.
pub fn map_pow_exponent_shortcut_with<F>(
    plan: PowExponentShortcutExecutionPlan,
    exponent_lhs: ExprId,
    base: ExprId,
    rhs: ExprId,
    original_op: RelOp,
    var: &str,
    mut render_expr: F,
) -> PowExponentShortcutEngineAction
where
    F: FnMut(ExprId) -> String,
{
    match plan {
        PowExponentShortcutExecutionPlan::Continue => PowExponentShortcutEngineAction::Continue,
        PowExponentShortcutExecutionPlan::IsolateExponent {
            rhs: target_rhs,
            op: target_op,
            narrative,
            rhs_exponent,
        } => {
            let base_desc = render_expr(base);
            let rhs_desc = render_expr(rhs);
            let rhs_exp_desc = rhs_exponent.map(render_expr);
            let description = pow_exponent_shortcut_message(
                narrative,
                var,
                &base_desc,
                &rhs_desc,
                rhs_exp_desc.as_deref(),
            );
            let equation = Equation {
                lhs: exponent_lhs,
                rhs: target_rhs,
                op: target_op.clone(),
            };
            let items = vec![PowExponentShortcutExecutionItem {
                equation,
                description,
            }];
            PowExponentShortcutEngineAction::IsolateExponent {
                rhs: target_rhs,
                op: target_op,
                items,
            }
        }
        PowExponentShortcutExecutionPlan::ReturnSolutionSet {
            solutions,
            narrative,
        } => {
            let base_desc = render_expr(base);
            let rhs_desc = render_expr(rhs);
            let description =
                pow_exponent_shortcut_message(narrative, var, &base_desc, &rhs_desc, None);
            let items = vec![PowExponentShortcutExecutionItem {
                equation: Equation {
                    lhs: exponent_lhs,
                    rhs: base,
                    op: original_op,
                },
                description,
            }];
            PowExponentShortcutEngineAction::ReturnSolutionSet { solutions, items }
        }
    }
}

/// Build didactic narration for normalized exponent shortcut plans.
pub fn pow_exponent_shortcut_message(
    narrative: PowExponentShortcutNarrative,
    var: &str,
    base_display: &str,
    rhs_display: &str,
    rhs_exponent_display: Option<&str>,
) -> String {
    match narrative {
        PowExponentShortcutNarrative::ZeroBaseExponentPositive => format!(
            "Power Equals Base Shortcut: 0^{} = 0 -> {} > 0 (0^0 undefined, 0^t for t<0 undefined)",
            var, var
        ),
        PowExponentShortcutNarrative::NumericBaseExponentOne => format!(
            "Power Equals Base Shortcut: {}^{} = {} -> {} = 1 (B^1 = B always holds)",
            base_display, var, rhs_display, var
        ),
        PowExponentShortcutNarrative::SymbolicBaseExponentOneNoBudget => format!(
            "Power Equals Base: {}^{} = {} -> {} = 1 (assuming base != 0, 1)",
            base_display, var, rhs_display, var
        ),
        PowExponentShortcutNarrative::SymbolicBaseCaseSplit => format!(
            "Power Equals Base with symbolic base '{}': case split -> a=1: AllReals, a=0: x>0, otherwise: x=1",
            base_display
        ),
        PowExponentShortcutNarrative::EqualPowBases => {
            let rhs_exp_display =
                rhs_exponent_display.expect("equal-pow-bases narrative requires rhs exponent");
            format!(
                "Pattern: {}^{} = {}^{} -> {} = {} (equal bases imply equal exponents when base != 0, 1)",
                base_display, var, base_display, rhs_exp_display, var, rhs_exp_display
            )
        }
    }
}

/// Plan exponent shortcut action from already-detected RHS shortcut inputs.
#[allow(clippy::too_many_arguments)]
pub fn plan_pow_exponent_shortcut_action_from_inputs(
    ctx: &mut Context,
    base: ExprId,
    op: RelOp,
    bases_equal: bool,
    rhs_pow_base_equal: Option<ExprId>,
    base_is_zero: bool,
    base_is_numeric: bool,
    can_branch: bool,
) -> PowExponentShortcutAction {
    let shortcut = classify_pow_exponent_shortcut(
        op.clone(),
        bases_equal,
        rhs_pow_base_equal,
        base_is_zero,
        base_is_numeric,
        can_branch,
    );
    plan_pow_exponent_shortcut_action(ctx, shortcut, base, op)
}

/// Plan exponent shortcut action from a pre-read RHS expression and
/// caller-provided base-equivalence check.
///
/// This helper centralizes the common engine flow:
/// detect shortcut inputs from RHS shape, classify route, and build action.
#[allow(clippy::too_many_arguments)]
pub fn plan_pow_exponent_shortcut_action_detecting_with<F>(
    ctx: &mut Context,
    rhs: ExprId,
    rhs_expr: &Expr,
    base: ExprId,
    op: RelOp,
    base_is_zero: bool,
    base_is_numeric: bool,
    can_branch: bool,
    same_base: F,
) -> PowExponentShortcutAction
where
    F: FnMut(ExprId) -> bool,
{
    let (bases_equal, rhs_pow_base_equal) =
        detect_pow_exponent_shortcut_inputs(rhs, rhs_expr, same_base);
    plan_pow_exponent_shortcut_action_from_inputs(
        ctx,
        base,
        op,
        bases_equal,
        rhs_pow_base_equal,
        base_is_zero,
        base_is_numeric,
        can_branch,
    )
}

/// Plan exponent shortcut action directly from equation shape `base^x op rhs`.
///
/// This combines RHS pattern detection, shortcut classification, and operational
/// action resolution into a single core helper.
#[allow(clippy::too_many_arguments)]
pub fn plan_pow_exponent_shortcut_action_for_rhs<F>(
    ctx: &mut Context,
    rhs: ExprId,
    base: ExprId,
    op: RelOp,
    base_is_zero: bool,
    base_is_numeric: bool,
    can_branch: bool,
    mut same_base: F,
) -> PowExponentShortcutAction
where
    F: FnMut(ExprId) -> bool,
{
    let rhs_expr = ctx.get(rhs).clone();
    let (bases_equal, rhs_pow_base_equal) =
        detect_pow_exponent_shortcut_inputs(rhs, &rhs_expr, &mut same_base);
    let shortcut = classify_pow_exponent_shortcut(
        op.clone(),
        bases_equal,
        rhs_pow_base_equal,
        base_is_zero,
        base_is_numeric,
        can_branch,
    );
    plan_pow_exponent_shortcut_action(ctx, shortcut, base, op)
}

/// Execute the full exponent-shortcut pipeline for `base^x op rhs`
/// using closure hooks.
///
/// This performs:
/// 1. RHS shortcut-shape detection
/// 2. Shortcut action planning
/// 3. Didactic/executable action mapping
#[allow(clippy::too_many_arguments)]
pub fn execute_pow_exponent_shortcut_with<FReadExpr, FPlanAction, FBasesEquivalent, FRenderExpr>(
    exponent_lhs: ExprId,
    base: ExprId,
    rhs: ExprId,
    original_op: RelOp,
    var: &str,
    base_is_zero: bool,
    base_is_numeric: bool,
    can_branch: bool,
    mut read_expr: FReadExpr,
    mut plan_action_from_inputs: FPlanAction,
    mut bases_equivalent: FBasesEquivalent,
    mut render_expr: FRenderExpr,
) -> PowExponentShortcutEngineAction
where
    FReadExpr: FnMut(ExprId) -> Expr,
    FPlanAction:
        FnMut(ExprId, RelOp, bool, Option<ExprId>, bool, bool, bool) -> PowExponentShortcutAction,
    FBasesEquivalent: FnMut(ExprId, ExprId) -> bool,
    FRenderExpr: FnMut(ExprId) -> String,
{
    let rhs_expr = read_expr(rhs);
    let bases_equal = bases_equivalent(base, rhs);
    let rhs_pow_base_equal = match rhs_expr {
        Expr::Pow(rhs_base, rhs_exp) if bases_equivalent(base, rhs_base) => Some(rhs_exp),
        _ => None,
    };

    let action = plan_action_from_inputs(
        base,
        original_op.clone(),
        bases_equal,
        rhs_pow_base_equal,
        base_is_zero,
        base_is_numeric,
        can_branch,
    );
    let plan = build_pow_exponent_shortcut_execution_plan(action);

    map_pow_exponent_shortcut_with(plan, exponent_lhs, base, rhs, original_op, var, |id| {
        render_expr(id)
    })
}

/// Classify route for base-isolation in a power equation.
pub fn classify_pow_base_isolation_route(
    exponent_is_even: bool,
    rhs_is_known_negative: bool,
    exponent_is_known_negative: bool,
) -> PowBaseIsolationRoute {
    if exponent_is_even && rhs_is_known_negative {
        PowBaseIsolationRoute::EvenExponentNegativeRhsImpossible
    } else if exponent_is_even {
        PowBaseIsolationRoute::EvenExponentUseAbsRoot
    } else {
        PowBaseIsolationRoute::GeneralRoot {
            flip_inequality_for_negative_exponent: exponent_is_known_negative,
        }
    }
}

/// Build executable plan for base isolation in `B^E op RHS`.
#[allow(clippy::too_many_arguments)]
pub fn plan_pow_base_isolation(
    ctx: &mut Context,
    base: ExprId,
    exponent: ExprId,
    rhs: ExprId,
    op: RelOp,
    exponent_is_even: bool,
    rhs_is_known_negative: bool,
    exponent_is_known_negative: bool,
) -> PowBaseIsolationPlan {
    let source_equation = Equation {
        lhs: ctx.add(Expr::Pow(base, exponent)),
        rhs,
        op: op.clone(),
    };
    let route = classify_pow_base_isolation_route(
        exponent_is_even,
        rhs_is_known_negative,
        exponent_is_known_negative,
    );

    match route {
        PowBaseIsolationRoute::EvenExponentNegativeRhsImpossible => {
            PowBaseIsolationPlan::ReturnSolutionSet {
                route,
                equation: source_equation,
                solutions: even_power_negative_rhs_outcome(op),
            }
        }
        PowBaseIsolationRoute::EvenExponentUseAbsRoot
        | PowBaseIsolationRoute::GeneralRoot { .. } => {
            let use_abs_root = matches!(route, PowBaseIsolationRoute::EvenExponentUseAbsRoot);
            let equation = crate::rational_power::build_root_isolation_equation(
                ctx,
                base,
                exponent,
                rhs,
                op.clone(),
                use_abs_root,
            );
            let normalized_op = match route {
                PowBaseIsolationRoute::GeneralRoot {
                    flip_inequality_for_negative_exponent,
                } => crate::isolation_utils::apply_sign_flip(
                    op,
                    flip_inequality_for_negative_exponent,
                ),
                _ => op,
            };
            PowBaseIsolationPlan::IsolateBase {
                route,
                equation,
                op: normalized_op,
                use_abs_root,
            }
        }
    }
}

/// Map base-isolation plan to an engine action with didactic payload.
pub fn map_pow_base_isolation_plan_with<F>(
    plan: PowBaseIsolationPlan,
    base: ExprId,
    exponent: ExprId,
    rhs: ExprId,
    original_op: RelOp,
    mut render_expr: F,
) -> PowBaseIsolationEngineAction
where
    F: FnMut(ExprId) -> String,
{
    match plan {
        PowBaseIsolationPlan::ReturnSolutionSet {
            route,
            equation,
            solutions,
        } => {
            let base_display = render_expr(base);
            let rhs_display = render_expr(rhs);
            let description = pow_base_isolation_terminal_message(
                route,
                &base_display,
                &original_op.to_string(),
                &rhs_display,
            );
            let items = vec![PowBaseIsolationExecutionItem {
                equation,
                description,
            }];
            PowBaseIsolationEngineAction::ReturnSolutionSet { solutions, items }
        }
        PowBaseIsolationPlan::IsolateBase {
            equation,
            op,
            use_abs_root,
            ..
        } => {
            let exponent_display = render_expr(exponent);
            let description = pow_base_root_isolation_message(&exponent_display, use_abs_root);
            let lhs = equation.lhs;
            let rhs = equation.rhs;
            let items = vec![PowBaseIsolationExecutionItem {
                equation,
                description,
            }];
            PowBaseIsolationEngineAction::IsolateBase {
                lhs,
                rhs,
                op,
                items,
            }
        }
    }
}

/// Build and map base-isolation action for `base^exponent op rhs` in one call.
///
/// This helper centralizes:
/// - route classification predicates (`even exponent`, `negative rhs`, `negative exponent`)
/// - plan construction
/// - didactic execution mapping
#[allow(clippy::too_many_arguments)]
pub fn build_pow_base_isolation_action_with<F>(
    ctx: &mut Context,
    base: ExprId,
    exponent: ExprId,
    rhs: ExprId,
    op: RelOp,
    mut render_expr: F,
) -> PowBaseIsolationEngineAction
where
    F: FnMut(&Context, ExprId) -> String,
{
    let plan = plan_pow_base_isolation(
        ctx,
        base,
        exponent,
        rhs,
        op.clone(),
        crate::isolation_utils::is_even_integer_expr(ctx, exponent),
        crate::isolation_utils::is_known_negative(ctx, rhs),
        crate::isolation_utils::is_known_negative(ctx, exponent),
    );

    map_pow_base_isolation_plan_with(plan, base, exponent, rhs, op, |id| render_expr(ctx, id))
}

/// Build didactic narration for terminal base-isolation outcomes.
pub fn pow_base_isolation_terminal_message(
    route: PowBaseIsolationRoute,
    base_display: &str,
    op_display: &str,
    rhs_display: &str,
) -> String {
    match route {
        PowBaseIsolationRoute::EvenExponentNegativeRhsImpossible => format!(
            "Even power cannot be negative ({} {} {})",
            base_display, op_display, rhs_display
        ),
        PowBaseIsolationRoute::EvenExponentUseAbsRoot
        | PowBaseIsolationRoute::GeneralRoot { .. } => "Power isolation terminated".to_string(),
    }
}

/// Build didactic narration for root-isolation steps in `B^E op RHS`.
pub fn pow_base_root_isolation_message(exponent_display: &str, use_abs_root: bool) -> String {
    if use_abs_root {
        format!(
            "Take {}-th root of both sides (even root implies absolute value)",
            exponent_display
        )
    } else {
        format!("Take {}-th root of both sides", exponent_display)
    }
}

/// Build didactic narration for additive isolation: `lhs ± term = rhs`.
pub fn subtract_both_sides_message(term_display: &str) -> String {
    format!("Subtract {} from both sides", term_display)
}

/// Build didactic narration for subtraction isolation: `lhs - term = rhs`.
pub fn add_both_sides_message(term_display: &str) -> String {
    format!("Add {} to both sides", term_display)
}

/// Build narration when moving a term requires multiplying by `-1`.
pub fn move_and_flip_message(term_display: &str) -> String {
    format!(
        "Move {} and multiply by -1 (flips inequality)",
        term_display
    )
}

/// Standard narration for isolating a negated left-hand side.
pub const NEGATED_LHS_ISOLATION_MESSAGE: &str = "Multiply both sides by -1 (flips inequality)";

/// Standard narration for swapping equation sides to expose the solve variable.
pub const SWAP_SIDES_TO_LHS_MESSAGE: &str = "Swap sides to put variable on LHS";

/// Build execution payload for isolating a negated left-hand side.
pub fn build_negated_lhs_isolation_item(equation_after: Equation) -> TermIsolationExecutionItem {
    TermIsolationExecutionItem {
        description: NEGATED_LHS_ISOLATION_MESSAGE.to_string(),
        equation: equation_after,
    }
}

/// Build didactic payload for isolating a negated left-hand side.
pub fn build_negated_lhs_isolation_step(equation_after: Equation) -> TermIsolationDidacticStep {
    term_isolation_didactic_step_from_execution_item(build_negated_lhs_isolation_item(
        equation_after,
    ))
}

/// Build execution payload for side swap (`rhs op lhs`) to place variable on LHS.
pub fn build_swap_sides_item(equation_after: Equation) -> TermIsolationExecutionItem {
    TermIsolationExecutionItem {
        description: SWAP_SIDES_TO_LHS_MESSAGE.to_string(),
        equation: equation_after,
    }
}

/// Build didactic payload for side swap (`rhs op lhs`) to place variable on LHS.
pub fn build_swap_sides_step(equation_after: Equation) -> TermIsolationDidacticStep {
    term_isolation_didactic_step_from_execution_item(build_swap_sides_item(equation_after))
}

/// Plan side swap rewrite and corresponding didactic step.
pub fn plan_swap_sides_step(equation: &Equation) -> TermIsolationRewritePlan {
    let swapped = crate::equation_rewrite::swap_sides_with_inequality_flip(equation);
    let item = build_swap_sides_item(swapped.clone());
    build_term_isolation_rewrite_plan_from_item(swapped, item)
}

/// Standard narration when solve-tactic normalization rewrites `base^x = rhs`
/// before logarithm isolation in Assume mode.
pub const SOLVE_TACTIC_NORMALIZATION_MESSAGE: &str =
    "Applied SolveTactic normalization (Assume mode) to enable logarithm isolation";

/// Build execution payload for solve-tactic normalization in exponent isolation.
pub fn build_solve_tactic_normalization_item(
    equation_after: Equation,
) -> TermIsolationExecutionItem {
    TermIsolationExecutionItem {
        description: SOLVE_TACTIC_NORMALIZATION_MESSAGE.to_string(),
        equation: equation_after,
    }
}

/// Build didactic payload for solve-tactic normalization in exponent isolation.
pub fn build_solve_tactic_normalization_step(
    equation_after: Equation,
) -> TermIsolationDidacticStep {
    term_isolation_didactic_step_from_execution_item(build_solve_tactic_normalization_item(
        equation_after,
    ))
}

/// Plan solve-tactic normalization rewrite (`base^exponent = rhs`) and step.
pub fn plan_solve_tactic_normalization_step(
    ctx: &mut Context,
    base: ExprId,
    exponent: ExprId,
    rhs: ExprId,
    op: RelOp,
) -> TermIsolationRewritePlan {
    let equation = Equation {
        lhs: ctx.add(Expr::Pow(base, exponent)),
        rhs,
        op,
    };
    let item = build_solve_tactic_normalization_item(equation.clone());
    build_term_isolation_rewrite_plan_from_item(equation, item)
}

/// Build and optionally map the first solve-tactic normalization step.
pub fn solve_solve_tactic_normalization_pipeline_with_item<S, FStep>(
    ctx: &mut Context,
    base: ExprId,
    exponent: ExprId,
    rhs: ExprId,
    op: RelOp,
    include_item: bool,
    map_item_to_step: FStep,
) -> Vec<S>
where
    FStep: FnMut(TermIsolationRewriteExecutionItem) -> S,
{
    let rewrite = plan_solve_tactic_normalization_step(ctx, base, exponent, rhs, op);
    collect_term_isolation_rewrite_first_step_with_item(&rewrite, include_item, map_item_to_step)
}

/// Execute solve-tactic normalization for exponent isolation (`base^x = rhs`) with
/// caller-provided side effects:
/// - clear blocked-hint sink before/after tactic simplification,
/// - simplify both sides under solve-tactic options,
/// - build didactic steps only when a rewrite happened.
#[allow(clippy::too_many_arguments)]
pub fn execute_pow_exponent_solve_tactic_normalization_with<S, FClear, FSimplify, FBuildSteps>(
    base: ExprId,
    exponent: ExprId,
    rhs: ExprId,
    op: RelOp,
    enabled: bool,
    mut clear_blocked_hints: FClear,
    mut simplify_with_tactic: FSimplify,
    mut build_steps: FBuildSteps,
) -> (ExprId, ExprId, Vec<S>)
where
    FClear: FnMut(),
    FSimplify: FnMut(ExprId) -> ExprId,
    FBuildSteps: FnMut(ExprId, ExprId, ExprId, RelOp) -> Vec<S>,
{
    if !enabled {
        return (base, rhs, Vec::new());
    }

    clear_blocked_hints();
    let sim_base = simplify_with_tactic(base);
    let sim_rhs = simplify_with_tactic(rhs);
    clear_blocked_hints();

    let steps = if sim_base != base || sim_rhs != rhs {
        build_steps(sim_base, exponent, sim_rhs, op)
    } else {
        Vec::new()
    };

    (sim_base, sim_rhs, steps)
}

/// Build didactic narration for multiplicative isolation.
pub fn divide_both_sides_message(term_display: &str) -> String {
    format!("Divide both sides by {}", term_display)
}

/// Build didactic narration for division isolation.
pub fn multiply_both_sides_message(term_display: &str) -> String {
    format!("Multiply both sides by {}", term_display)
}

/// Build separator message for branch-based didactic traces.
pub fn end_case_message(case_index: usize) -> String {
    format!("--- End of Case {} ---", case_index)
}

/// Build narration for denominator-sign case split (`den > 0`).
pub fn denominator_positive_case_message(den_display: &str) -> String {
    format!(
        "Case 1: Assume {} > 0. Multiply by positive denominator.",
        den_display
    )
}

/// Build narration for denominator-sign case split (`den < 0`).
pub fn denominator_negative_case_message(den_display: &str) -> String {
    format!(
        "Case 2: Assume {} < 0. Multiply by negative denominator (flips inequality).",
        den_display
    )
}

/// Build narration for isolated-denominator split (`den > 0`).
pub fn isolated_denominator_positive_case_message(den_display: &str) -> String {
    format!(
        "Case 1: Assume {} > 0. Multiply by {} (positive). Inequality direction preserved (flipped from isolation logic).",
        den_display, den_display
    )
}

/// Build narration for isolated-denominator split (`den < 0`).
pub fn isolated_denominator_negative_case_message(den_display: &str) -> String {
    format!(
        "Case 2: Assume {} < 0. Multiply by {} (negative). Inequality flips.",
        den_display, den_display
    )
}

/// Didactic payload for one term-isolation rewrite step.
#[derive(Debug, Clone, PartialEq)]
pub struct TermIsolationDidacticStep {
    pub description: String,
    pub equation_after: Equation,
}

/// Planned isolation rewrite plus its didactic step payload.
#[derive(Debug, Clone, PartialEq)]
pub struct TermIsolationRewritePlan {
    pub equation: Equation,
    pub items: Vec<TermIsolationRewriteExecutionItem>,
}

/// Solved payload for one term-isolation rewrite.
#[derive(Debug, Clone, PartialEq)]
pub struct TermIsolationRewriteSolved<T> {
    pub rewrite: TermIsolationRewritePlan,
    pub solved: T,
}

/// One executable isolation rewrite item aligned with a didactic payload.
#[derive(Debug, Clone, PartialEq)]
pub struct TermIsolationRewriteExecutionItem {
    pub equation: Equation,
    pub description: String,
}

impl TermIsolationRewriteExecutionItem {
    /// User-facing narration for this execution item.
    pub fn description(&self) -> &str {
        &self.description
    }
}

fn build_term_isolation_rewrite_plan_from_item(
    equation: Equation,
    item: TermIsolationExecutionItem,
) -> TermIsolationRewritePlan {
    let items = vec![TermIsolationRewriteExecutionItem {
        equation: item.equation,
        description: item.description,
    }];
    TermIsolationRewritePlan { equation, items }
}

/// Collect term-isolation rewrite didactic steps in display order.
pub fn collect_term_isolation_rewrite_didactic_steps(
    plan: &TermIsolationRewritePlan,
) -> Vec<TermIsolationDidacticStep> {
    plan.items
        .iter()
        .cloned()
        .map(|item| TermIsolationDidacticStep {
            description: item.description,
            equation_after: item.equation,
        })
        .collect()
}

/// Collect isolation rewrite execution items in display order.
pub fn collect_term_isolation_rewrite_execution_items(
    plan: &TermIsolationRewritePlan,
) -> Vec<TermIsolationRewriteExecutionItem> {
    plan.items.clone()
}

/// Return the first isolation-rewrite execution item, if any.
pub fn first_term_isolation_rewrite_execution_item(
    plan: &TermIsolationRewritePlan,
) -> Option<TermIsolationRewriteExecutionItem> {
    collect_term_isolation_rewrite_execution_items(plan)
        .into_iter()
        .next()
}

/// Collect at most one mapped rewrite step from term-isolation plan.
pub fn collect_term_isolation_rewrite_first_step_with_item<S, FStep>(
    rewrite: &TermIsolationRewritePlan,
    include_item: bool,
    mut map_item_to_step: FStep,
) -> Vec<S>
where
    FStep: FnMut(TermIsolationRewriteExecutionItem) -> S,
{
    let mut steps = Vec::new();
    if include_item {
        if let Some(item) = first_term_isolation_rewrite_execution_item(rewrite) {
            steps.push(map_item_to_step(item));
        }
    }
    steps
}

/// Execute one term-isolation rewrite with caller-provided solve callback.
pub fn solve_term_isolation_rewrite_with<E, T, FSolve>(
    rewrite: TermIsolationRewritePlan,
    mut solve: FSolve,
) -> Result<TermIsolationRewriteSolved<T>, E>
where
    FSolve: FnMut(Equation) -> Result<T, E>,
{
    let solved = solve(rewrite.equation.clone())?;
    Ok(TermIsolationRewriteSolved { rewrite, solved })
}

/// Solved result for a term-isolation rewrite pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct TermIsolationRewritePipelineSolved<S> {
    pub solution_set: SolutionSet,
    pub steps: Vec<S>,
}

/// Execute term-isolation rewrite solving + optional item dispatch.
pub fn solve_term_isolation_rewrite_pipeline_with_item<E, S, FSolve, FStep>(
    rewrite: TermIsolationRewritePlan,
    include_item: bool,
    solve_rewritten: FSolve,
    mut map_item_to_step: FStep,
) -> Result<TermIsolationRewritePipelineSolved<S>, E>
where
    FSolve: FnMut(Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(TermIsolationRewriteExecutionItem) -> S,
{
    let solved_rewrite = solve_term_isolation_rewrite_with(rewrite, solve_rewritten)?;
    let mut steps = Vec::new();
    if include_item {
        if let Some(item) = first_term_isolation_rewrite_execution_item(&solved_rewrite.rewrite) {
            steps.push(map_item_to_step(item));
        }
    }
    let (solution_set, mut sub_steps) = solved_rewrite.solved;
    steps.append(&mut sub_steps);
    Ok(TermIsolationRewritePipelineSolved {
        solution_set,
        steps,
    })
}

/// Execute term-isolation plan with optional RHS pre-simplification before
/// recursive isolation.
pub fn solve_term_isolation_plan_with<E, S, FSimplify, FSolve, FStep>(
    plan: TermIsolationRewritePlan,
    include_item: bool,
    simplify_rhs_before_solve: bool,
    mut simplify_rhs: FSimplify,
    mut solve_rewritten: FSolve,
    map_item_to_step: FStep,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FSimplify: FnMut(ExprId) -> ExprId,
    FSolve: FnMut(Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(TermIsolationRewriteExecutionItem) -> S,
{
    let solved = solve_term_isolation_rewrite_pipeline_with_item(
        plan,
        include_item,
        |equation| {
            let mut rewritten = equation;
            if simplify_rhs_before_solve {
                rewritten.rhs = simplify_rhs(rewritten.rhs);
            }
            solve_rewritten(rewritten)
        },
        map_item_to_step,
    )?;
    Ok((solved.solution_set, solved.steps))
}

/// Build and execute one term-isolation plan with optional RHS pre-simplification
/// before recursive isolation.
pub fn execute_term_isolation_plan_with<E, S, FPlan, FSimplify, FSolve, FStep>(
    mut plan_rewrite: FPlan,
    include_item: bool,
    simplify_rhs_before_solve: bool,
    simplify_rhs: FSimplify,
    solve_rewritten: FSolve,
    map_item_to_step: FStep,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FPlan: FnMut() -> TermIsolationRewritePlan,
    FSimplify: FnMut(ExprId) -> ExprId,
    FSolve: FnMut(Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(TermIsolationRewriteExecutionItem) -> S,
{
    let plan = plan_rewrite();
    solve_term_isolation_plan_with(
        plan,
        include_item,
        simplify_rhs_before_solve,
        simplify_rhs,
        solve_rewritten,
        map_item_to_step,
    )
}

/// Execute term-isolation plan and merge solved steps before caller-owned
/// existing steps.
pub fn execute_term_isolation_plan_with_and_merge_with_existing_steps_with<
    E,
    S,
    FPlan,
    FSimplify,
    FSolve,
    FStep,
>(
    plan_rewrite: FPlan,
    include_item: bool,
    simplify_rhs_before_solve: bool,
    existing_steps: Vec<S>,
    simplify_rhs: FSimplify,
    solve_rewritten: FSolve,
    map_item_to_step: FStep,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FPlan: FnMut() -> TermIsolationRewritePlan,
    FSimplify: FnMut(ExprId) -> ExprId,
    FSolve: FnMut(Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(TermIsolationRewriteExecutionItem) -> S,
{
    let solved = execute_term_isolation_plan_with(
        plan_rewrite,
        include_item,
        simplify_rhs_before_solve,
        simplify_rhs,
        solve_rewritten,
        map_item_to_step,
    )?;
    Ok(merge_solved_with_existing_steps_prepend(
        solved,
        existing_steps,
    ))
}

/// Merge a solved `(SolutionSet, steps)` payload by prepending solved steps
/// before caller-owned existing steps.
pub fn merge_solved_with_existing_steps_prepend<S>(
    solved: (SolutionSet, Vec<S>),
    existing_steps: Vec<S>,
) -> (SolutionSet, Vec<S>) {
    let (solution_set, mut solved_steps) = solved;
    solved_steps.extend(existing_steps);
    (solution_set, solved_steps)
}

/// Merge a solved `(SolutionSet, steps)` payload by appending solved steps
/// after caller-owned existing steps.
pub fn merge_solved_with_existing_steps_append<S>(
    solved: (SolutionSet, Vec<S>),
    existing_steps: Vec<S>,
) -> (SolutionSet, Vec<S>) {
    let (solution_set, solved_steps) = solved;
    let mut merged = existing_steps;
    merged.extend(solved_steps);
    (solution_set, merged)
}

/// Merge an optional solved payload by appending solved steps after
/// caller-owned existing steps.
pub fn merge_optional_solved_with_existing_steps_append<S>(
    solved: Option<(SolutionSet, Vec<S>)>,
    existing_steps: Vec<S>,
) -> Option<(SolutionSet, Vec<S>)> {
    solved.map(|solved| merge_solved_with_existing_steps_append(solved, existing_steps))
}

/// Merge an optional solved payload by appending solved steps after
/// caller-owned existing steps stored in-place.
///
/// Returns `None` without modifying `existing_steps` when `solved` is `None`.
pub fn merge_optional_solved_with_existing_steps_append_mut<S>(
    solved: Option<(SolutionSet, Vec<S>)>,
    existing_steps: &mut Vec<S>,
) -> Option<(SolutionSet, Vec<S>)> {
    solved.map(|(solution_set, solved_steps)| {
        let mut merged = std::mem::take(existing_steps);
        merged.extend(solved_steps);
        (solution_set, merged)
    })
}

/// Execute log-isolation pipeline and prepend solved steps before caller-owned
/// existing steps.
#[allow(clippy::type_complexity)]
pub fn execute_log_isolation_result_pipeline_or_else_with_and_merge_with_existing_steps_with<
    E,
    S,
    FPlan,
    FSolve,
    FMap,
    FError,
>(
    include_item: bool,
    existing_steps: Vec<S>,
    plan_rewrite: FPlan,
    solve_rewritten: FSolve,
    map_item_to_step: FMap,
    not_plannable_error: FError,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FPlan: FnMut() -> Option<crate::log_isolation::LogIsolationRewritePlan>,
    FSolve: FnMut(&Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FMap: FnMut(crate::log_isolation::LogIsolationExecutionItem) -> S,
    FError: FnOnce() -> E,
{
    let solved = crate::log_isolation::execute_log_isolation_result_pipeline_or_else_with(
        include_item,
        plan_rewrite,
        solve_rewritten,
        map_item_to_step,
        not_plannable_error,
    )?;
    Ok(merge_solved_with_existing_steps_prepend(
        solved,
        existing_steps,
    ))
}

/// Execute unary-inverse pipeline and prepend solved steps before caller-owned
/// existing steps.
#[allow(clippy::too_many_arguments)]
pub fn execute_unary_inverse_result_pipeline_or_else_with_and_merge_with_existing_steps_with<
    E,
    S,
    FPlan,
    FSimplifyRhs,
    FSolve,
    FStep,
    FError,
>(
    fn_name: &str,
    arg: ExprId,
    other: ExprId,
    op: RelOp,
    is_lhs: bool,
    include_items: bool,
    existing_steps: Vec<S>,
    plan_unary_inverse_step: FPlan,
    simplify_rhs_with_entries: FSimplifyRhs,
    solve: FSolve,
    map_item_to_step: FStep,
    unsupported_error: FError,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FPlan: FnMut(
        &str,
        ExprId,
        ExprId,
        RelOp,
        bool,
    ) -> Option<crate::function_inverse::UnaryInverseIsolationStepPlan>,
    FSimplifyRhs: FnMut(ExprId) -> (ExprId, Vec<(String, ExprId)>),
    FSolve: FnMut(ExprId, ExprId, RelOp) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(crate::function_inverse::UnaryInverseSolveExecutionItem) -> S,
    FError: FnOnce() -> E,
{
    let solved = crate::function_inverse::execute_unary_inverse_result_pipeline_or_else_with(
        fn_name,
        arg,
        other,
        op,
        is_lhs,
        include_items,
        plan_unary_inverse_step,
        simplify_rhs_with_entries,
        solve,
        map_item_to_step,
        unsupported_error,
    )?;
    Ok(merge_solved_with_existing_steps_prepend(
        solved,
        existing_steps,
    ))
}

/// Execute negated-LHS isolation rewrite and optional first-item didactic
/// projection via caller-provided callbacks.
pub fn solve_negated_lhs_isolation_with<E, S, FPlan, FSolve, FStep>(
    mut plan_rewrite: FPlan,
    var: &str,
    include_item: bool,
    mut solve_rewritten: FSolve,
    mut map_item_to_step: FStep,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FPlan: FnMut() -> TermIsolationRewritePlan,
    FSolve: FnMut(Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(TermIsolationRewriteExecutionItem) -> S,
{
    let rewrite = plan_rewrite();
    let first_item = if include_item {
        first_term_isolation_rewrite_execution_item(&rewrite)
    } else {
        None
    };
    let (solution_set, mut sub_steps) = solve_rewritten(rewrite.equation.clone(), var)?;

    let mut steps = Vec::new();
    if let Some(item) = first_item {
        steps.push(map_item_to_step(item));
    }
    steps.append(&mut sub_steps);
    Ok((solution_set, steps))
}

/// Solve negated-LHS isolation rewrite and prepend solved steps before caller
/// existing steps.
pub fn solve_negated_lhs_isolation_with_and_merge_with_existing_steps<E, S, FPlan, FSolve, FStep>(
    plan_rewrite: FPlan,
    var: &str,
    include_item: bool,
    existing_steps: Vec<S>,
    solve_rewritten: FSolve,
    map_item_to_step: FStep,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FPlan: FnMut() -> TermIsolationRewritePlan,
    FSolve: FnMut(Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(TermIsolationRewriteExecutionItem) -> S,
{
    let solved = solve_negated_lhs_isolation_with(
        plan_rewrite,
        var,
        include_item,
        solve_rewritten,
        map_item_to_step,
    )?;
    Ok(merge_solved_with_existing_steps_prepend(
        solved,
        existing_steps,
    ))
}

/// Route chosen for denominator isolation with zero-RHS guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivDenominatorIsolationRoute {
    RhsZeroToInfinity,
    DivisionRewrite,
}

/// Planned denominator isolation rewrite with route metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct DivDenominatorIsolationRewritePlan {
    pub equation: Equation,
    pub route: DivDenominatorIsolationRoute,
}

fn build_term_isolation_item_with<F>(
    equation_after: Equation,
    moved_term: ExprId,
    mut render_expr: F,
    message_builder: fn(&str) -> String,
) -> TermIsolationExecutionItem
where
    F: FnMut(ExprId) -> String,
{
    let moved_desc = render_expr(moved_term);
    TermIsolationExecutionItem {
        description: message_builder(&moved_desc),
        equation: equation_after,
    }
}

/// Build execution payload for `A + B = RHS -> A = RHS - B` (or symmetric case).
pub fn build_add_operand_isolation_item_with<F>(
    equation_after: Equation,
    moved_term: ExprId,
    render_expr: F,
) -> TermIsolationExecutionItem
where
    F: FnMut(ExprId) -> String,
{
    build_term_isolation_item_with(
        equation_after,
        moved_term,
        render_expr,
        subtract_both_sides_message,
    )
}

/// Build didactic payload for `A + B = RHS -> A = RHS - B` (or symmetric case).
pub fn build_add_operand_isolation_step_with<F>(
    equation_after: Equation,
    moved_term: ExprId,
    render_expr: F,
) -> TermIsolationDidacticStep
where
    F: FnMut(ExprId) -> String,
{
    term_isolation_didactic_step_from_execution_item(build_add_operand_isolation_item_with(
        equation_after,
        moved_term,
        render_expr,
    ))
}

/// Build execution payload for `A - B = RHS -> A = RHS + B`.
pub fn build_sub_minuend_isolation_item_with<F>(
    equation_after: Equation,
    moved_term: ExprId,
    render_expr: F,
) -> TermIsolationExecutionItem
where
    F: FnMut(ExprId) -> String,
{
    build_term_isolation_item_with(
        equation_after,
        moved_term,
        render_expr,
        add_both_sides_message,
    )
}

/// Build didactic payload for `A - B = RHS -> A = RHS + B`.
pub fn build_sub_minuend_isolation_step_with<F>(
    equation_after: Equation,
    moved_term: ExprId,
    render_expr: F,
) -> TermIsolationDidacticStep
where
    F: FnMut(ExprId) -> String,
{
    term_isolation_didactic_step_from_execution_item(build_sub_minuend_isolation_item_with(
        equation_after,
        moved_term,
        render_expr,
    ))
}

/// Build execution payload for `A - B = RHS -> B = A - RHS`.
pub fn build_sub_subtrahend_isolation_item_with<F>(
    equation_after: Equation,
    moved_term: ExprId,
    render_expr: F,
) -> TermIsolationExecutionItem
where
    F: FnMut(ExprId) -> String,
{
    build_term_isolation_item_with(
        equation_after,
        moved_term,
        render_expr,
        move_and_flip_message,
    )
}

/// Build didactic payload for `A - B = RHS -> B = A - RHS`.
pub fn build_sub_subtrahend_isolation_step_with<F>(
    equation_after: Equation,
    moved_term: ExprId,
    render_expr: F,
) -> TermIsolationDidacticStep
where
    F: FnMut(ExprId) -> String,
{
    term_isolation_didactic_step_from_execution_item(build_sub_subtrahend_isolation_item_with(
        equation_after,
        moved_term,
        render_expr,
    ))
}

/// Build execution payload for `A * B = RHS -> A = RHS / B` (or symmetric case).
pub fn build_mul_factor_isolation_item_with<F>(
    equation_after: Equation,
    moved_term: ExprId,
    render_expr: F,
) -> TermIsolationExecutionItem
where
    F: FnMut(ExprId) -> String,
{
    build_term_isolation_item_with(
        equation_after,
        moved_term,
        render_expr,
        divide_both_sides_message,
    )
}

/// Build didactic payload for `A * B = RHS -> A = RHS / B` (or symmetric case).
pub fn build_mul_factor_isolation_step_with<F>(
    equation_after: Equation,
    moved_term: ExprId,
    render_expr: F,
) -> TermIsolationDidacticStep
where
    F: FnMut(ExprId) -> String,
{
    term_isolation_didactic_step_from_execution_item(build_mul_factor_isolation_item_with(
        equation_after,
        moved_term,
        render_expr,
    ))
}

/// Build execution payload for `A / B = RHS -> A = RHS * B`.
pub fn build_div_numerator_isolation_item_with<F>(
    equation_after: Equation,
    moved_term: ExprId,
    render_expr: F,
) -> TermIsolationExecutionItem
where
    F: FnMut(ExprId) -> String,
{
    build_term_isolation_item_with(
        equation_after,
        moved_term,
        render_expr,
        multiply_both_sides_message,
    )
}

/// Build didactic payload for `A / B = RHS -> A = RHS * B`.
pub fn build_div_numerator_isolation_step_with<F>(
    equation_after: Equation,
    moved_term: ExprId,
    render_expr: F,
) -> TermIsolationDidacticStep
where
    F: FnMut(ExprId) -> String,
{
    term_isolation_didactic_step_from_execution_item(build_div_numerator_isolation_item_with(
        equation_after,
        moved_term,
        render_expr,
    ))
}

/// Plan negated-LHS isolation and corresponding didactic step.
pub fn plan_negated_lhs_isolation_step(
    ctx: &mut Context,
    inner: ExprId,
    rhs: ExprId,
    op: RelOp,
) -> TermIsolationRewritePlan {
    let equation = crate::equation_rewrite::isolate_negated_lhs(ctx, inner, rhs, op);
    let item = build_negated_lhs_isolation_item(equation.clone());
    build_term_isolation_rewrite_plan_from_item(equation, item)
}

/// Plan add-operand isolation and corresponding didactic step.
pub fn plan_add_operand_isolation_step_with<F>(
    ctx: &mut Context,
    kept: ExprId,
    moved: ExprId,
    rhs: ExprId,
    op: RelOp,
    render_expr: F,
) -> TermIsolationRewritePlan
where
    F: FnMut(ExprId) -> String,
{
    let equation = crate::equation_rewrite::isolate_add_operand(ctx, kept, moved, rhs, op);
    let item = build_add_operand_isolation_item_with(equation.clone(), moved, render_expr);
    build_term_isolation_rewrite_plan_from_item(equation, item)
}

/// Plan sub-minuend isolation and corresponding didactic step.
pub fn plan_sub_minuend_isolation_step_with<F>(
    ctx: &mut Context,
    minuend: ExprId,
    subtrahend: ExprId,
    rhs: ExprId,
    op: RelOp,
    render_expr: F,
) -> TermIsolationRewritePlan
where
    F: FnMut(ExprId) -> String,
{
    let equation = crate::equation_rewrite::isolate_sub_minuend(ctx, minuend, subtrahend, rhs, op);
    let item = build_sub_minuend_isolation_item_with(equation.clone(), subtrahend, render_expr);
    build_term_isolation_rewrite_plan_from_item(equation, item)
}

/// Plan sub-subtrahend isolation and corresponding didactic step.
pub fn plan_sub_subtrahend_isolation_step_with<F>(
    ctx: &mut Context,
    minuend: ExprId,
    subtrahend: ExprId,
    rhs: ExprId,
    op: RelOp,
    render_expr: F,
) -> TermIsolationRewritePlan
where
    F: FnMut(ExprId) -> String,
{
    let equation =
        crate::equation_rewrite::isolate_sub_subtrahend(ctx, minuend, subtrahend, rhs, op);
    let item = build_sub_subtrahend_isolation_item_with(equation.clone(), minuend, render_expr);
    build_term_isolation_rewrite_plan_from_item(equation, item)
}

/// Plan subtraction isolation `(l - r) = rhs` using variable-position routing.
pub fn plan_sub_isolation_step_with<F>(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    render_expr: F,
) -> TermIsolationRewritePlan
where
    F: FnMut(ExprId) -> String,
{
    match derive_sub_isolation_route(ctx, left, var) {
        SubIsolationRoute::Minuend => {
            plan_sub_minuend_isolation_step_with(ctx, left, right, rhs, op, render_expr)
        }
        SubIsolationRoute::Subtrahend => {
            plan_sub_subtrahend_isolation_step_with(ctx, left, right, rhs, op, render_expr)
        }
    }
}

/// Plan multiplicative-factor isolation and corresponding didactic step.
pub fn plan_mul_factor_isolation_step_with<F>(
    ctx: &mut Context,
    kept: ExprId,
    moved: ExprId,
    rhs: ExprId,
    op: RelOp,
    moved_is_negative: bool,
    render_expr: F,
) -> TermIsolationRewritePlan
where
    F: FnMut(ExprId) -> String,
{
    let equation =
        crate::equation_rewrite::isolate_mul_factor(ctx, kept, moved, rhs, op, moved_is_negative);
    let item = build_mul_factor_isolation_item_with(equation.clone(), moved, render_expr);
    build_term_isolation_rewrite_plan_from_item(equation, item)
}

/// Plan division-numerator isolation and corresponding didactic step.
pub fn plan_div_numerator_isolation_step_with<F>(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
    rhs: ExprId,
    op: RelOp,
    denominator_is_negative: bool,
    render_expr: F,
) -> TermIsolationRewritePlan
where
    F: FnMut(ExprId) -> String,
{
    let equation = crate::equation_rewrite::isolate_div_numerator(
        ctx,
        numerator,
        denominator,
        rhs,
        op,
        denominator_is_negative,
    );
    let item = build_div_numerator_isolation_item_with(equation.clone(), denominator, render_expr);
    build_term_isolation_rewrite_plan_from_item(equation, item)
}

/// Plan denominator isolation with `rhs == 0` safety guard.
pub fn plan_div_denominator_isolation_with_zero_rhs_guard(
    ctx: &mut Context,
    denominator: ExprId,
    numerator: ExprId,
    rhs: ExprId,
    op: RelOp,
) -> DivDenominatorIsolationRewritePlan {
    let (equation, kind) = crate::equation_rewrite::isolate_div_denominator_with_zero_rhs_guard(
        ctx,
        denominator,
        numerator,
        rhs,
        op,
    );
    let route = match kind {
        crate::equation_rewrite::DivDenominatorIsolationKind::RhsZeroToInfinity => {
            DivDenominatorIsolationRoute::RhsZeroToInfinity
        }
        crate::equation_rewrite::DivDenominatorIsolationKind::DivisionRewrite => {
            DivDenominatorIsolationRoute::DivisionRewrite
        }
    };
    DivDenominatorIsolationRewritePlan { equation, route }
}

/// Didactic payload for one equation step.
#[derive(Debug, Clone, PartialEq)]
pub struct DivisionCaseDidacticStep {
    pub description: String,
    pub equation_after: Equation,
}

/// One executable division didactic item aligned with an equation payload.
#[derive(Debug, Clone, PartialEq)]
pub struct DivisionDidacticExecutionItem {
    pub equation: Equation,
    pub description: String,
}

impl DivisionDidacticExecutionItem {
    /// User-facing narration for this execution item.
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Didactic payload for denominator-sign split traces.
#[derive(Debug, Clone, PartialEq)]
pub struct DivisionDenominatorSignSplitDidactic {
    pub positive_case: DivisionCaseDidacticStep,
    pub negative_case: DivisionCaseDidacticStep,
    pub case_boundary: DivisionCaseDidacticStep,
}

/// Collect denominator sign-split didactic steps in display order:
/// positive case, negative case, and boundary marker.
pub fn collect_division_denominator_sign_split_didactic_steps(
    didactic: &DivisionDenominatorSignSplitDidactic,
) -> Vec<DivisionCaseDidacticStep> {
    vec![
        didactic.positive_case.clone(),
        didactic.negative_case.clone(),
        didactic.case_boundary.clone(),
    ]
}

/// Collect denominator sign-split execution items in display order:
/// positive case, negative case, and boundary marker.
pub fn collect_division_denominator_sign_split_execution_items(
    execution: &DivisionDenominatorSignSplitExecutionPlan,
) -> Vec<DivisionDidacticExecutionItem> {
    execution.items.clone()
}

/// Collect isolated-denominator sign-split execution items in display order:
/// positive case, negative case, and boundary marker.
pub fn collect_isolated_denominator_sign_split_execution_items(
    execution: &IsolatedDenominatorSignSplitExecutionPlan,
) -> Vec<DivisionDidacticExecutionItem> {
    execution.items.clone()
}

/// Build didactic payload for division denominator sign-split:
/// - Case 1 (`den > 0`)
/// - Case 2 (`den < 0`)
/// - Case separator marker between both branches.
pub fn build_division_denominator_sign_split_steps_with<F>(
    positive_equation: Equation,
    negative_equation: Equation,
    denominator: ExprId,
    case_boundary_lhs: ExprId,
    case_boundary_op: RelOp,
    mut render_expr: F,
) -> DivisionDenominatorSignSplitDidactic
where
    F: FnMut(ExprId) -> String,
{
    let den_display = render_expr(denominator);
    let boundary_rhs = negative_equation.rhs;
    DivisionDenominatorSignSplitDidactic {
        positive_case: DivisionCaseDidacticStep {
            description: denominator_positive_case_message(&den_display),
            equation_after: positive_equation,
        },
        negative_case: DivisionCaseDidacticStep {
            description: denominator_negative_case_message(&den_display),
            equation_after: negative_equation,
        },
        case_boundary: DivisionCaseDidacticStep {
            description: end_case_message(1),
            equation_after: build_case_boundary_equation(
                case_boundary_lhs,
                boundary_rhs,
                case_boundary_op,
            ),
        },
    }
}

/// Build didactic payload for already-isolated denominator sign-split:
/// - Case 1 (`den > 0`)
/// - Case 2 (`den < 0`)
/// - Case separator marker between both branches.
pub fn build_isolated_denominator_sign_split_steps_with<F>(
    positive_equation: Equation,
    negative_equation: Equation,
    denominator: ExprId,
    case_boundary_lhs: ExprId,
    case_boundary_op: RelOp,
    mut render_expr: F,
) -> DivisionDenominatorSignSplitDidactic
where
    F: FnMut(ExprId) -> String,
{
    let den_display = render_expr(denominator);
    let boundary_rhs = negative_equation.rhs;
    DivisionDenominatorSignSplitDidactic {
        positive_case: DivisionCaseDidacticStep {
            description: isolated_denominator_positive_case_message(&den_display),
            equation_after: positive_equation,
        },
        negative_case: DivisionCaseDidacticStep {
            description: isolated_denominator_negative_case_message(&den_display),
            equation_after: negative_equation,
        },
        case_boundary: DivisionCaseDidacticStep {
            description: end_case_message(1),
            equation_after: build_case_boundary_equation(
                case_boundary_lhs,
                boundary_rhs,
                case_boundary_op,
            ),
        },
    }
}

/// Build narration for taking logarithm on both sides.
pub fn take_log_base_message(base_display: &str) -> String {
    format!("Take log base {} of both sides", base_display)
}

/// Build narration for guarded logarithm isolation.
pub fn take_log_base_under_guard_message(base_display: &str, guard_message: &str) -> String {
    format!(
        "Take log base {} of both sides (under guard: {})",
        base_display, guard_message
    )
}

/// Build narration for conditional solutions produced under assumptions.
pub fn conditional_solution_message(message: &str) -> String {
    format!("Conditional solution: {}", message)
}

/// Build narration for residual fallback.
pub fn residual_message(message: &str) -> String {
    format!("{} (residual)", message)
}

/// Build narration for residual fallback when branch budget is exhausted.
pub fn residual_budget_exhausted_message(message: &str) -> String {
    format!("{} (residual, budget exhausted)", message)
}

/// Build didactic payload for a terminal log outcome (empty/residual).
pub fn build_terminal_outcome_item(
    outcome: &TerminalSolveOutcome,
    equation_after: Equation,
    residual_suffix: &str,
) -> TermIsolationExecutionItem {
    TermIsolationExecutionItem {
        description: terminal_outcome_message(outcome, residual_suffix),
        equation: equation_after,
    }
}

/// Build didactic payload for a terminal log outcome (empty/residual).
pub fn build_terminal_outcome_step(
    outcome: &TerminalSolveOutcome,
    equation_after: Equation,
    residual_suffix: &str,
) -> TermIsolationDidacticStep {
    term_isolation_didactic_step_from_execution_item(build_terminal_outcome_item(
        outcome,
        equation_after,
        residual_suffix,
    ))
}

/// Solved result for terminal-outcome pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct TerminalOutcomePipelineSolved<S> {
    pub solution_set: SolutionSet,
    pub steps: Vec<S>,
}

/// Execute terminal-outcome item dispatch, optionally mapping the terminal
/// didactic item to caller step payload.
pub fn solve_terminal_outcome_pipeline_with_item<S, FStep>(
    outcome: TerminalSolveOutcome,
    equation_after: Equation,
    residual_suffix: &str,
    include_item: bool,
    mut map_item_to_step: FStep,
) -> TerminalOutcomePipelineSolved<S>
where
    FStep: FnMut(TermIsolationExecutionItem) -> S,
{
    let solution_set = outcome.solutions.clone();
    let mut steps = Vec::new();
    if include_item {
        let item = build_terminal_outcome_item(&outcome, equation_after, residual_suffix);
        steps.push(map_item_to_step(item));
    }
    TerminalOutcomePipelineSolved {
        solution_set,
        steps,
    }
}

/// Resolve log terminal outcome from decision and execute terminal pipeline.
#[allow(clippy::too_many_arguments)]
pub fn execute_log_terminal_outcome_pipeline_with_item<S, FStep>(
    ctx: &mut Context,
    decision: &LogSolveDecision,
    mode: DomainModeKind,
    wildcard_scope: bool,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    equation_after: Equation,
    residual_suffix: &str,
    include_item: bool,
    map_item_to_step: FStep,
) -> Option<TerminalOutcomePipelineSolved<S>>
where
    FStep: FnMut(TermIsolationExecutionItem) -> S,
{
    let outcome = resolve_log_terminal_outcome(ctx, decision, mode, wildcard_scope, lhs, rhs, var)?;
    Some(solve_terminal_outcome_pipeline_with_item(
        outcome,
        equation_after,
        residual_suffix,
        include_item,
        map_item_to_step,
    ))
}

/// Resolve log terminal outcome, execute terminal pipeline, and append solved
/// steps after caller-owned existing steps.
#[allow(clippy::too_many_arguments)]
pub fn execute_log_terminal_outcome_pipeline_with_item_and_merge_with_existing_steps_with<
    S,
    FStep,
>(
    ctx: &mut Context,
    decision: &LogSolveDecision,
    mode: DomainModeKind,
    wildcard_scope: bool,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    equation_after: Equation,
    residual_suffix: &str,
    include_item: bool,
    existing_steps: Vec<S>,
    map_item_to_step: FStep,
) -> Option<(SolutionSet, Vec<S>)>
where
    FStep: FnMut(TermIsolationExecutionItem) -> S,
{
    let solved = execute_log_terminal_outcome_pipeline_with_item(
        ctx,
        decision,
        mode,
        wildcard_scope,
        lhs,
        rhs,
        var,
        equation_after,
        residual_suffix,
        include_item,
        map_item_to_step,
    )?;
    Some(merge_solved_with_existing_steps_append(
        (solved.solution_set, solved.steps),
        existing_steps,
    ))
}

/// Gated outcome after evaluating terminal log cases and post-terminal decision state.
#[derive(Debug, Clone, PartialEq)]
pub enum LogDecisionTerminalGate<S> {
    Terminal {
        solution_set: SolutionSet,
        steps: Vec<S>,
    },
    NeedsComplex {
        message: &'static str,
        assumptions: Vec<LogAssumption>,
    },
    Continue {
        assumptions: Vec<LogAssumption>,
    },
}

/// Simplified terminal gate outcome after optional assumption side-effects have
/// been emitted by the caller-provided visitor.
#[derive(Debug, Clone, PartialEq)]
pub enum LogDecisionTerminalResult<S> {
    Terminal {
        solution_set: SolutionSet,
        steps: Vec<S>,
    },
    NeedsComplex {
        message: &'static str,
    },
    Continue,
}

/// Unified decision outcome for exponent-log isolation routing:
/// - terminal solve handled,
/// - explicit real-domain rejection,
/// - unsupported route solved via guarded/residual pipeline,
/// - or continue with regular log isolation.
#[derive(Debug, Clone, PartialEq)]
pub enum PowExponentLogDecisionPipelineResult<S> {
    Terminal {
        solution_set: SolutionSet,
        steps: Vec<S>,
    },
    NeedsComplex {
        message: &'static str,
    },
    UnsupportedSolved {
        solution_set: SolutionSet,
        steps: Vec<S>,
    },
    Continue,
}

/// Resolve the exponent-log decision pipeline into:
/// - `Ok(Some((solutions, steps)))` when solved,
/// - `Err(message)` when complex domain is required,
/// - `Ok(None)` when regular log-isolation should continue.
pub fn resolve_pow_exponent_log_decision_pipeline_result<S>(
    pipeline: PowExponentLogDecisionPipelineResult<S>,
) -> Result<Option<(SolutionSet, Vec<S>)>, &'static str> {
    match pipeline {
        PowExponentLogDecisionPipelineResult::Terminal {
            solution_set,
            steps,
        }
        | PowExponentLogDecisionPipelineResult::UnsupportedSolved {
            solution_set,
            steps,
        } => Ok(Some((solution_set, steps))),
        PowExponentLogDecisionPipelineResult::NeedsComplex { message } => Err(message),
        PowExponentLogDecisionPipelineResult::Continue => Ok(None),
    }
}

/// Validate the exponent-log precondition: `rhs` must not contain target
/// variable for `base^x = rhs` logarithmic isolation.
pub fn ensure_pow_exponent_rhs_without_variable(
    ctx: &Context,
    rhs: ExprId,
    var: &str,
) -> Result<(), &'static str> {
    if pow_exponent_rhs_contains_variable(ctx, rhs, var) {
        Err("Cannot isolate exponential: variable appears on both sides")
    } else {
        Ok(())
    }
}

/// Resolve terminal log outcomes first; if not terminal, emit implied assumptions
/// and classify whether solving can continue in the current value domain.
#[allow(clippy::too_many_arguments)]
pub fn execute_log_terminal_outcome_and_assumptions_gate_with<S, FStep>(
    ctx: &mut Context,
    decision: &LogSolveDecision,
    mode: DomainModeKind,
    wildcard_scope: bool,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    equation_after: Equation,
    residual_suffix: &str,
    include_item: bool,
    existing_steps: Vec<S>,
    map_item_to_step: FStep,
) -> LogDecisionTerminalGate<S>
where
    FStep: FnMut(TermIsolationExecutionItem) -> S,
{
    if let Some((solution_set, steps)) =
        execute_log_terminal_outcome_pipeline_with_item_and_merge_with_existing_steps_with(
            ctx,
            decision,
            mode,
            wildcard_scope,
            lhs,
            rhs,
            var,
            equation_after,
            residual_suffix,
            include_item,
            existing_steps,
            map_item_to_step,
        )
    {
        return LogDecisionTerminalGate::Terminal {
            solution_set,
            steps,
        };
    }

    let assumptions = decision_assumptions(decision).to_vec();

    if let Some(msg) = log_decision_needs_complex_message(decision) {
        return LogDecisionTerminalGate::NeedsComplex {
            message: msg,
            assumptions,
        };
    }
    debug_assert!(
        !log_decision_is_empty_set(decision),
        "empty-set decision should be terminal and handled above"
    );

    LogDecisionTerminalGate::Continue { assumptions }
}

/// Mutable-step variant of terminal/assumption gate: terminal paths consume and
/// merge caller-owned steps, while non-terminal paths preserve them untouched.
#[allow(clippy::too_many_arguments)]
pub fn execute_log_terminal_outcome_and_assumptions_gate_with_existing_steps_mut<S, FStep>(
    ctx: &mut Context,
    decision: &LogSolveDecision,
    mode: DomainModeKind,
    wildcard_scope: bool,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    equation_after: Equation,
    residual_suffix: &str,
    include_item: bool,
    existing_steps: &mut Vec<S>,
    mut map_item_to_step: FStep,
) -> LogDecisionTerminalGate<S>
where
    FStep: FnMut(TermIsolationExecutionItem) -> S,
{
    if let Some(solved) = execute_log_terminal_outcome_pipeline_with_item(
        ctx,
        decision,
        mode,
        wildcard_scope,
        lhs,
        rhs,
        var,
        equation_after,
        residual_suffix,
        include_item,
        &mut map_item_to_step,
    ) {
        let (solution_set, steps) = merge_solved_with_existing_steps_append(
            (solved.solution_set, solved.steps),
            std::mem::take(existing_steps),
        );
        return LogDecisionTerminalGate::Terminal {
            solution_set,
            steps,
        };
    }

    let assumptions = decision_assumptions(decision).to_vec();
    if let Some(msg) = log_decision_needs_complex_message(decision) {
        return LogDecisionTerminalGate::NeedsComplex {
            message: msg,
            assumptions,
        };
    }
    debug_assert!(
        !log_decision_is_empty_set(decision),
        "empty-set decision should be terminal and handled above"
    );
    LogDecisionTerminalGate::Continue { assumptions }
}

/// Mutable-step terminal/assumption gate that emits each inferred assumption
/// through `visit_assumption` and returns a simplified terminal result enum.
#[allow(clippy::too_many_arguments)]
pub fn execute_log_terminal_outcome_and_assumptions_gate_with_existing_steps_mut_and_each_assumption<
    S,
    FStep,
    FAssumption,
>(
    ctx: &mut Context,
    decision: &LogSolveDecision,
    mode: DomainModeKind,
    wildcard_scope: bool,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    equation_after: Equation,
    residual_suffix: &str,
    include_item: bool,
    existing_steps: &mut Vec<S>,
    map_item_to_step: FStep,
    mut visit_assumption: FAssumption,
) -> LogDecisionTerminalResult<S>
where
    FStep: FnMut(TermIsolationExecutionItem) -> S,
    FAssumption: FnMut(&Context, LogAssumption),
{
    match execute_log_terminal_outcome_and_assumptions_gate_with_existing_steps_mut(
        ctx,
        decision,
        mode,
        wildcard_scope,
        lhs,
        rhs,
        var,
        equation_after,
        residual_suffix,
        include_item,
        existing_steps,
        map_item_to_step,
    ) {
        LogDecisionTerminalGate::Terminal {
            solution_set,
            steps,
        } => LogDecisionTerminalResult::Terminal {
            solution_set,
            steps,
        },
        LogDecisionTerminalGate::NeedsComplex {
            message,
            assumptions,
        } => {
            for assumption in assumptions {
                visit_assumption(ctx, assumption);
            }
            LogDecisionTerminalResult::NeedsComplex { message }
        }
        LogDecisionTerminalGate::Continue { assumptions } => {
            for assumption in assumptions {
                visit_assumption(ctx, assumption);
            }
            LogDecisionTerminalResult::Continue
        }
    }
}

/// Execute exponent-log decision routing in one helper:
/// 1) terminal/assumption gate,
/// 2) optional unsupported guarded/residual pipeline,
/// 3) continue flag for regular logarithmic isolation.
#[allow(clippy::too_many_arguments)]
pub fn execute_pow_exponent_log_decision_pipeline_with_existing_steps_mut<
    S,
    FStep,
    FAssumption,
    FPlan,
    FGuarded,
    FHint,
>(
    ctx: &mut Context,
    decision: &LogSolveDecision,
    mode: DomainModeKind,
    wildcard_scope: bool,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    equation_after: Equation,
    residual_suffix: &str,
    include_terminal_item: bool,
    existing_steps: &mut Vec<S>,
    mut map_item_to_step: FStep,
    mut visit_assumption: FAssumption,
    include_unsupported_items: bool,
    plan_unsupported_execution: FPlan,
    try_guarded_solve: FGuarded,
    register_blocked_hint: FHint,
) -> PowExponentLogDecisionPipelineResult<S>
where
    FStep: FnMut(TermIsolationExecutionItem) -> S,
    FAssumption: FnMut(&Context, LogAssumption),
    FPlan: FnOnce() -> Option<PowExponentLogUnsupportedExecution>,
    FGuarded: FnMut(&Equation) -> Option<SolutionSet>,
    FHint: FnMut(LogBlockedHintRecord),
{
    match execute_log_terminal_outcome_and_assumptions_gate_with_existing_steps_mut_and_each_assumption(
        ctx,
        decision,
        mode,
        wildcard_scope,
        lhs,
        rhs,
        var,
        equation_after,
        residual_suffix,
        include_terminal_item,
        existing_steps,
        &mut map_item_to_step,
        &mut visit_assumption,
    ) {
        LogDecisionTerminalResult::Terminal {
            solution_set,
            steps,
        } => {
            return PowExponentLogDecisionPipelineResult::Terminal {
                solution_set,
                steps,
            };
        }
        LogDecisionTerminalResult::NeedsComplex { message } => {
            return PowExponentLogDecisionPipelineResult::NeedsComplex { message };
        }
        LogDecisionTerminalResult::Continue => {}
    }

    if let Some((solution_set, steps)) =
        execute_pow_exponent_log_unsupported_pipeline_from_decision_and_finalize_with_existing_steps_with(
            include_unsupported_items,
            existing_steps,
            plan_unsupported_execution,
            try_guarded_solve,
            &mut map_item_to_step,
            register_blocked_hint,
        )
    {
        return PowExponentLogDecisionPipelineResult::UnsupportedSolved {
            solution_set,
            steps,
        };
    }

    PowExponentLogDecisionPipelineResult::Continue
}

/// Execute post-terminal exponent-log routing:
/// - map terminal gate results,
/// - optionally run unsupported guarded/residual pipeline when continuing.
pub fn execute_pow_exponent_log_post_terminal_pipeline_with_existing_steps_mut<
    S,
    FPlan,
    FGuarded,
    FStep,
    FHint,
>(
    terminal_result: LogDecisionTerminalResult<S>,
    include_unsupported_items: bool,
    existing_steps: &mut Vec<S>,
    plan_unsupported_execution: FPlan,
    try_guarded_solve: FGuarded,
    map_item_to_step: FStep,
    register_blocked_hint: FHint,
) -> PowExponentLogDecisionPipelineResult<S>
where
    FPlan: FnOnce() -> Option<PowExponentLogUnsupportedExecution>,
    FGuarded: FnMut(&Equation) -> Option<SolutionSet>,
    FStep: FnMut(TermIsolationExecutionItem) -> S,
    FHint: FnMut(LogBlockedHintRecord),
{
    match terminal_result {
        LogDecisionTerminalResult::Terminal {
            solution_set,
            steps,
        } => {
            return PowExponentLogDecisionPipelineResult::Terminal {
                solution_set,
                steps,
            };
        }
        LogDecisionTerminalResult::NeedsComplex { message } => {
            return PowExponentLogDecisionPipelineResult::NeedsComplex { message };
        }
        LogDecisionTerminalResult::Continue => {}
    }

    if let Some((solution_set, steps)) =
        execute_pow_exponent_log_unsupported_pipeline_from_decision_and_finalize_with_existing_steps_with(
            include_unsupported_items,
            existing_steps,
            plan_unsupported_execution,
            try_guarded_solve,
            map_item_to_step,
            register_blocked_hint,
        )
    {
        return PowExponentLogDecisionPipelineResult::UnsupportedSolved {
            solution_set,
            steps,
        };
    }

    PowExponentLogDecisionPipelineResult::Continue
}

/// Execute post-terminal exponent-log routing and immediately resolve it into:
/// - `Ok(Some((solutions, steps)))` when solved,
/// - `Err(message)` when complex-domain escalation is required,
/// - `Ok(None)` when regular logarithmic isolation should continue.
pub fn execute_and_resolve_pow_exponent_log_post_terminal_pipeline_with_existing_steps_mut<
    S,
    FPlan,
    FGuarded,
    FStep,
    FHint,
>(
    terminal_result: LogDecisionTerminalResult<S>,
    include_unsupported_items: bool,
    existing_steps: &mut Vec<S>,
    plan_unsupported_execution: FPlan,
    try_guarded_solve: FGuarded,
    map_item_to_step: FStep,
    register_blocked_hint: FHint,
) -> Result<Option<(SolutionSet, Vec<S>)>, &'static str>
where
    FPlan: FnOnce() -> Option<PowExponentLogUnsupportedExecution>,
    FGuarded: FnMut(&Equation) -> Option<SolutionSet>,
    FStep: FnMut(TermIsolationExecutionItem) -> S,
    FHint: FnMut(LogBlockedHintRecord),
{
    resolve_pow_exponent_log_decision_pipeline_result(
        execute_pow_exponent_log_post_terminal_pipeline_with_existing_steps_mut(
            terminal_result,
            include_unsupported_items,
            existing_steps,
            plan_unsupported_execution,
            try_guarded_solve,
            map_item_to_step,
            register_blocked_hint,
        ),
    )
}

/// Execute the full exponent-log decision pipeline and resolve it into:
/// - `Ok(Some((solutions, steps)))` when solved,
/// - `Err(message)` when complex-domain escalation is required,
/// - `Ok(None)` when regular logarithmic isolation should continue.
#[allow(clippy::too_many_arguments)]
pub fn execute_and_resolve_pow_exponent_log_decision_pipeline_with_existing_steps_mut<
    S,
    FStep,
    FAssumption,
    FPlan,
    FGuarded,
    FHint,
>(
    ctx: &mut Context,
    decision: &LogSolveDecision,
    mode: DomainModeKind,
    wildcard_scope: bool,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    equation_after: Equation,
    residual_suffix: &str,
    include_terminal_item: bool,
    existing_steps: &mut Vec<S>,
    map_item_to_step: FStep,
    visit_assumption: FAssumption,
    include_unsupported_items: bool,
    plan_unsupported_execution: FPlan,
    try_guarded_solve: FGuarded,
    register_blocked_hint: FHint,
) -> Result<Option<(SolutionSet, Vec<S>)>, &'static str>
where
    FStep: FnMut(TermIsolationExecutionItem) -> S,
    FAssumption: FnMut(&Context, LogAssumption),
    FPlan: FnOnce() -> Option<PowExponentLogUnsupportedExecution>,
    FGuarded: FnMut(&Equation) -> Option<SolutionSet>,
    FHint: FnMut(LogBlockedHintRecord),
{
    resolve_pow_exponent_log_decision_pipeline_result(
        execute_pow_exponent_log_decision_pipeline_with_existing_steps_mut(
            ctx,
            decision,
            mode,
            wildcard_scope,
            lhs,
            rhs,
            var,
            equation_after,
            residual_suffix,
            include_terminal_item,
            existing_steps,
            map_item_to_step,
            visit_assumption,
            include_unsupported_items,
            plan_unsupported_execution,
            try_guarded_solve,
            register_blocked_hint,
        ),
    )
}

/// Build didactic payload for conditional-solution messaging.
pub fn build_conditional_solution_item(
    message: &str,
    equation_after: Equation,
) -> TermIsolationExecutionItem {
    TermIsolationExecutionItem {
        description: conditional_solution_message(message),
        equation: equation_after,
    }
}

/// Build didactic payload for conditional-solution messaging.
pub fn build_conditional_solution_step(
    message: &str,
    equation_after: Equation,
) -> TermIsolationDidacticStep {
    term_isolation_didactic_step_from_execution_item(build_conditional_solution_item(
        message,
        equation_after,
    ))
}

/// Build a follow-up didactic step after guarded logarithmic solving:
/// `conditional` when guarded solve succeeds, `residual` otherwise.
pub fn build_guarded_log_followup_item(
    guarded_solve_succeeded: bool,
    message: &str,
    equation_after: Equation,
) -> TermIsolationExecutionItem {
    if guarded_solve_succeeded {
        build_conditional_solution_item(message, equation_after)
    } else {
        build_residual_item(message, equation_after)
    }
}

/// Build a follow-up didactic step after guarded logarithmic solving:
/// `conditional` when guarded solve succeeds, `residual` otherwise.
pub fn build_guarded_log_followup_step(
    guarded_solve_succeeded: bool,
    message: &str,
    equation_after: Equation,
) -> TermIsolationDidacticStep {
    term_isolation_didactic_step_from_execution_item(build_guarded_log_followup_item(
        guarded_solve_succeeded,
        message,
        equation_after,
    ))
}

/// Plan guarded logarithmic isolation (`x = log_b(rhs)`) including:
/// 1) the guarded rewrite equation/item and
/// 2) follow-up narration for success vs residual fallback.
#[allow(clippy::too_many_arguments)]
pub fn plan_guarded_pow_exponent_log_execution(
    ctx: &mut Context,
    exponent: ExprId,
    base: ExprId,
    rhs: ExprId,
    op: RelOp,
    message: &str,
    base_display: &str,
    original_equation: Equation,
) -> GuardedPowExponentLogExecutionPlan {
    let rewrite = plan_pow_exponent_log_isolation_step(
        ctx,
        exponent,
        base,
        rhs,
        op,
        Some(message),
        base_display,
    );
    let followup_success =
        build_guarded_log_followup_item(true, message, original_equation.clone());
    let followup_residual = build_guarded_log_followup_item(false, message, original_equation);
    GuardedPowExponentLogExecutionPlan {
        rewrite,
        followup_success,
        followup_residual,
    }
}

/// Build didactic payload for residual fallback messaging.
pub fn build_residual_item(message: &str, equation_after: Equation) -> TermIsolationExecutionItem {
    TermIsolationExecutionItem {
        description: residual_message(message),
        equation: equation_after,
    }
}

/// Build didactic payload for residual fallback messaging.
pub fn build_residual_step(message: &str, equation_after: Equation) -> TermIsolationDidacticStep {
    term_isolation_didactic_step_from_execution_item(build_residual_item(message, equation_after))
}

/// Build didactic payload for residual fallback when branch budget is exhausted.
pub fn build_residual_budget_exhausted_item(
    message: &str,
    equation_after: Equation,
) -> TermIsolationExecutionItem {
    TermIsolationExecutionItem {
        description: residual_budget_exhausted_message(message),
        equation: equation_after,
    }
}

/// Build didactic payload for residual fallback when branch budget is exhausted.
pub fn build_residual_budget_exhausted_step(
    message: &str,
    equation_after: Equation,
) -> TermIsolationDidacticStep {
    term_isolation_didactic_step_from_execution_item(build_residual_budget_exhausted_item(
        message,
        equation_after,
    ))
}

/// Collect generic term-isolation didactic steps in display order.
pub fn collect_term_isolation_didactic_steps(
    step: &TermIsolationDidacticStep,
) -> Vec<TermIsolationDidacticStep> {
    vec![step.clone()]
}

/// One executable term-isolation item aligned with didactic payload.
#[derive(Debug, Clone, PartialEq)]
pub struct TermIsolationExecutionItem {
    pub equation: Equation,
    pub description: String,
}

impl TermIsolationExecutionItem {
    /// User-facing narration for this execution item.
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Convert one didactic term-isolation step into its execution representation.
pub fn term_isolation_execution_item_from_didactic_step(
    step: TermIsolationDidacticStep,
) -> TermIsolationExecutionItem {
    TermIsolationExecutionItem {
        equation: step.equation_after,
        description: step.description,
    }
}

/// Convert one executable term-isolation item into its didactic representation.
pub fn term_isolation_didactic_step_from_execution_item(
    item: TermIsolationExecutionItem,
) -> TermIsolationDidacticStep {
    TermIsolationDidacticStep {
        description: item.description,
        equation_after: item.equation,
    }
}

/// Collect generic term-isolation execution items in display order.
pub fn collect_term_isolation_execution_items(
    step: &TermIsolationDidacticStep,
) -> Vec<TermIsolationExecutionItem> {
    collect_term_isolation_didactic_steps(step)
        .into_iter()
        .map(term_isolation_execution_item_from_didactic_step)
        .collect()
}

/// Build `exponent = log(base, rhs)` step payload with optional guard narration.
pub fn build_pow_exponent_log_isolation_step_with<F>(
    ctx: &mut Context,
    exponent: ExprId,
    base: ExprId,
    rhs: ExprId,
    op: RelOp,
    guard_message: Option<&str>,
    mut render_expr: F,
) -> PowExponentLogIsolationStep
where
    F: FnMut(&Context, ExprId) -> String,
{
    let equation_after =
        crate::rational_power::build_exponent_log_isolation_equation(ctx, exponent, base, rhs, op);
    let base_desc = render_expr(ctx, base);
    let description = match guard_message {
        Some(msg) => take_log_base_under_guard_message(&base_desc, msg),
        None => take_log_base_message(&base_desc),
    };
    PowExponentLogIsolationStep {
        description,
        equation_after,
    }
}

/// Plan `exponent = log(base, rhs)` and build its didactic payload.
pub fn plan_pow_exponent_log_isolation_step(
    ctx: &mut Context,
    exponent: ExprId,
    base: ExprId,
    rhs: ExprId,
    op: RelOp,
    guard_message: Option<&str>,
    base_display: &str,
) -> PowExponentLogIsolationRewritePlan {
    let step = build_pow_exponent_log_isolation_step_with(
        ctx,
        exponent,
        base,
        rhs,
        op,
        guard_message,
        |_, _| base_display.to_string(),
    );
    let items = vec![PowExponentLogIsolationExecutionItem {
        equation: step.equation_after.clone(),
        description: step.description.clone(),
    }];
    PowExponentLogIsolationRewritePlan {
        equation: step.equation_after.clone(),
        items,
    }
}

/// Plan `exponent = log(base, rhs)` and build its didactic payload by rendering
/// the base expression with a caller-provided formatter.
pub fn plan_pow_exponent_log_isolation_step_with<F>(
    ctx: &mut Context,
    exponent: ExprId,
    base: ExprId,
    rhs: ExprId,
    op: RelOp,
    guard_message: Option<&str>,
    render_expr: F,
) -> PowExponentLogIsolationRewritePlan
where
    F: FnMut(&Context, ExprId) -> String,
{
    let step = build_pow_exponent_log_isolation_step_with(
        ctx,
        exponent,
        base,
        rhs,
        op,
        guard_message,
        render_expr,
    );
    let items = vec![PowExponentLogIsolationExecutionItem {
        equation: step.equation_after.clone(),
        description: step.description.clone(),
    }];
    PowExponentLogIsolationRewritePlan {
        equation: step.equation_after.clone(),
        items,
    }
}

/// Plan guarded logarithmic isolation (`x = log_b(rhs)`) and render base display
/// with a caller-provided formatter.
#[allow(clippy::too_many_arguments)]
pub fn plan_guarded_pow_exponent_log_execution_with<F>(
    ctx: &mut Context,
    exponent: ExprId,
    base: ExprId,
    rhs: ExprId,
    op: RelOp,
    message: &str,
    original_equation: Equation,
    render_expr: F,
) -> GuardedPowExponentLogExecutionPlan
where
    F: FnMut(&Context, ExprId) -> String,
{
    let rewrite = plan_pow_exponent_log_isolation_step_with(
        ctx,
        exponent,
        base,
        rhs,
        op,
        Some(message),
        render_expr,
    );
    let followup_success =
        build_guarded_log_followup_item(true, message, original_equation.clone());
    let followup_residual = build_guarded_log_followup_item(false, message, original_equation);
    GuardedPowExponentLogExecutionPlan {
        rewrite,
        followup_success,
        followup_residual,
    }
}

/// Execute guarded logarithmic isolation with caller-provided guarded-solve callback.
///
/// The callback receives the rewritten equation `x = log_base(rhs)` and may
/// return solved solutions under guard or `None` to indicate residual fallback.
#[allow(clippy::too_many_arguments)]
pub fn execute_guarded_pow_exponent_log_with<F, FG>(
    ctx: &mut Context,
    exponent: ExprId,
    base: ExprId,
    rhs: ExprId,
    op: RelOp,
    message: &str,
    original_equation: Equation,
    render_expr: F,
    mut try_guarded_solve: FG,
) -> GuardedPowExponentLogExecution
where
    F: FnMut(&Context, ExprId) -> String,
    FG: FnMut(&Equation) -> Option<SolutionSet>,
{
    let plan = plan_guarded_pow_exponent_log_execution_with(
        ctx,
        exponent,
        base,
        rhs,
        op,
        message,
        original_equation,
        render_expr,
    );
    let guarded_solutions = try_guarded_solve(&plan.rewrite.equation);
    let followup = plan.followup_item(guarded_solutions.is_some());
    GuardedPowExponentLogExecution {
        rewrite: plan.rewrite,
        followup,
        guarded_solutions,
    }
}

/// Execute a preplanned guarded logarithmic isolation branch.
pub fn execute_guarded_pow_exponent_log_plan_with<FG>(
    plan: GuardedPowExponentLogExecutionPlan,
    mut try_guarded_solve: FG,
) -> GuardedPowExponentLogExecution
where
    FG: FnMut(&Equation) -> Option<SolutionSet>,
{
    let guarded_solutions = try_guarded_solve(&plan.rewrite.equation);
    let followup = plan.followup_item(guarded_solutions.is_some());
    GuardedPowExponentLogExecution {
        rewrite: plan.rewrite,
        followup,
        guarded_solutions,
    }
}

/// Solve one planned exponent-log rewrite equation with caller-provided solver.
pub fn solve_pow_exponent_log_isolation_rewrite_with<E, T, FSolve>(
    rewrite: PowExponentLogIsolationRewritePlan,
    mut solve_rewrite: FSolve,
) -> Result<PowExponentLogIsolationSolved<T>, E>
where
    FSolve: FnMut(&Equation) -> Result<T, E>,
{
    let solved = solve_rewrite(&rewrite.equation)?;
    Ok(PowExponentLogIsolationSolved { rewrite, solved })
}

/// Solved result for one exponent-log rewrite pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct PowExponentLogIsolationRewritePipelineSolved<S> {
    pub solution_set: SolutionSet,
    pub steps: Vec<S>,
}

/// Execute exponent-log rewrite solve + optional first-item dispatch.
pub fn solve_pow_exponent_log_isolation_rewrite_pipeline_with_item<E, S, FSolve, FStep>(
    rewrite: PowExponentLogIsolationRewritePlan,
    include_item: bool,
    solve_rewrite: FSolve,
    mut map_item_to_step: FStep,
) -> Result<PowExponentLogIsolationRewritePipelineSolved<S>, E>
where
    FSolve: FnMut(&Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(PowExponentLogIsolationExecutionItem) -> S,
{
    let solved_rewrite = solve_pow_exponent_log_isolation_rewrite_with(rewrite, solve_rewrite)?;
    let mut steps = Vec::new();
    if include_item {
        if let Some(item) = first_pow_exponent_log_isolation_execution_item(&solved_rewrite.rewrite)
        {
            steps.push(map_item_to_step(item));
        }
    }
    let (solution_set, mut sub_steps) = solved_rewrite.solved;
    steps.append(&mut sub_steps);
    Ok(PowExponentLogIsolationRewritePipelineSolved {
        solution_set,
        steps,
    })
}

/// Execute exponent-log rewrite pipeline from a deferred rewrite-plan builder.
///
/// This composes:
/// 1. rewrite planning
/// 2. pipeline solve with optional didactic item mapping
pub fn execute_pow_exponent_log_isolation_pipeline_with_item_with<E, S, FPlan, FSolve, FStep>(
    include_item: bool,
    plan_rewrite: FPlan,
    solve_rewrite: FSolve,
    map_item_to_step: FStep,
) -> Result<PowExponentLogIsolationRewritePipelineSolved<S>, E>
where
    FPlan: FnOnce() -> PowExponentLogIsolationRewritePlan,
    FSolve: FnMut(&Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(PowExponentLogIsolationExecutionItem) -> S,
{
    let rewrite = plan_rewrite();
    solve_pow_exponent_log_isolation_rewrite_pipeline_with_item(
        rewrite,
        include_item,
        solve_rewrite,
        map_item_to_step,
    )
}

/// Execute exponent-log rewrite pipeline and prepend solved steps before
/// caller-owned existing steps.
pub fn execute_pow_exponent_log_isolation_pipeline_with_item_and_merge_with_existing_steps_with<
    E,
    S,
    FPlan,
    FSolve,
    FStep,
>(
    include_item: bool,
    existing_steps: Vec<S>,
    plan_rewrite: FPlan,
    solve_rewrite: FSolve,
    map_item_to_step: FStep,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FPlan: FnOnce() -> PowExponentLogIsolationRewritePlan,
    FSolve: FnMut(&Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(PowExponentLogIsolationExecutionItem) -> S,
{
    let solved = execute_pow_exponent_log_isolation_pipeline_with_item_with(
        include_item,
        plan_rewrite,
        solve_rewrite,
        map_item_to_step,
    )?;
    Ok(merge_solved_with_existing_steps_prepend(
        (solved.solution_set, solved.steps),
        existing_steps,
    ))
}

/// Execute a planned unsupported logarithmic route (residual or guarded).
///
/// For guarded routes, this runs the guarded rewrite solve callback and
/// materializes final solution-set semantics (`guarded` or residual fallback).
pub fn execute_pow_exponent_log_unsupported_with<FG>(
    execution: PowExponentLogUnsupportedExecution,
    try_guarded_solve: FG,
) -> PowExponentLogUnsupportedSolvedExecution
where
    FG: FnMut(&Equation) -> Option<SolutionSet>,
{
    match execution {
        PowExponentLogUnsupportedExecution::Residual { item, solutions } => {
            PowExponentLogUnsupportedSolvedExecution::Residual { item, solutions }
        }
        PowExponentLogUnsupportedExecution::Guarded {
            blocked_hints,
            plan,
            guard,
            residual,
        } => {
            let guarded_execution =
                execute_guarded_pow_exponent_log_plan_with(plan, try_guarded_solve);
            let solutions =
                guarded_or_residual(Some(guard), guarded_execution.guarded_solutions, residual);
            PowExponentLogUnsupportedSolvedExecution::Guarded {
                blocked_hints,
                rewrite_item: first_pow_exponent_log_isolation_execution_item(
                    &guarded_execution.rewrite,
                ),
                followup_item: guarded_execution.followup,
                solutions,
            }
        }
    }
}

/// Solved result for unsupported-log execution pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct PowExponentLogUnsupportedPipelineSolved<S> {
    pub blocked_hints: Vec<LogBlockedHintRecord>,
    pub solution_set: SolutionSet,
    pub steps: Vec<S>,
}

/// Execute unsupported-log route, optionally map didactic items to steps, and
/// return blocked-hint payload for caller-side registration.
pub fn solve_pow_exponent_log_unsupported_pipeline_with_items<FG, FStep, S>(
    execution: PowExponentLogUnsupportedExecution,
    include_items: bool,
    try_guarded_solve: FG,
    mut map_item_to_step: FStep,
) -> PowExponentLogUnsupportedPipelineSolved<S>
where
    FG: FnMut(&Equation) -> Option<SolutionSet>,
    FStep: FnMut(TermIsolationExecutionItem) -> S,
{
    let solved = execute_pow_exponent_log_unsupported_with(execution, try_guarded_solve);
    match solved {
        PowExponentLogUnsupportedSolvedExecution::Residual { item, solutions } => {
            let mut steps = Vec::new();
            if include_items {
                steps.push(map_item_to_step(item));
            }
            PowExponentLogUnsupportedPipelineSolved {
                blocked_hints: vec![],
                solution_set: solutions,
                steps,
            }
        }
        PowExponentLogUnsupportedSolvedExecution::Guarded {
            blocked_hints,
            rewrite_item,
            followup_item,
            solutions,
        } => {
            let mut steps = Vec::new();
            if include_items {
                if let Some(item) = rewrite_item {
                    steps.push(map_item_to_step(
                        pow_exponent_log_execution_item_as_term_item(item),
                    ));
                }
                steps.push(map_item_to_step(followup_item));
            }
            PowExponentLogUnsupportedPipelineSolved {
                blocked_hints,
                solution_set: solutions,
                steps,
            }
        }
    }
}

/// Plan and solve unsupported exponent-log route in one helper.
///
/// Returns `None` when plan construction decides the route is supported and no
/// unsupported execution should run.
pub fn execute_pow_exponent_log_unsupported_pipeline_from_decision_with<FG, FStep, FPlan, S>(
    include_items: bool,
    plan_execution: FPlan,
    try_guarded_solve: FG,
    map_item_to_step: FStep,
) -> Option<PowExponentLogUnsupportedPipelineSolved<S>>
where
    FG: FnMut(&Equation) -> Option<SolutionSet>,
    FStep: FnMut(TermIsolationExecutionItem) -> S,
    FPlan: FnOnce() -> Option<PowExponentLogUnsupportedExecution>,
{
    let execution = plan_execution()?;
    Some(solve_pow_exponent_log_unsupported_pipeline_with_items(
        execution,
        include_items,
        try_guarded_solve,
        map_item_to_step,
    ))
}

/// Iterate assumptions implied by the logarithmic decision.
pub fn visit_log_decision_assumptions_with<F>(decision: &LogSolveDecision, mut visit: F)
where
    F: FnMut(LogAssumption),
{
    for assumption in decision_assumptions(decision).iter().copied() {
        visit(assumption);
    }
}

/// Return the "needs complex" message when the decision requires complex domain.
pub fn log_decision_needs_complex_message(decision: &LogSolveDecision) -> Option<&'static str> {
    match decision {
        LogSolveDecision::NeedsComplex(message) => Some(*message),
        _ => None,
    }
}

/// Whether the log decision is terminal-empty-set.
pub fn log_decision_is_empty_set(decision: &LogSolveDecision) -> bool {
    matches!(decision, LogSolveDecision::EmptySet(_))
}

/// Merge unsupported exponent-log pipeline outcome with caller-owned steps and
/// allow caller-side blocked-hint registration through callback.
pub fn merge_pow_exponent_log_unsupported_pipeline_with_existing_steps<S, FHint>(
    solved: Option<PowExponentLogUnsupportedPipelineSolved<S>>,
    existing_steps: Vec<S>,
    mut register_blocked_hint: FHint,
) -> Option<(SolutionSet, Vec<S>)>
where
    FHint: FnMut(LogBlockedHintRecord),
{
    let PowExponentLogUnsupportedPipelineSolved {
        blocked_hints,
        solution_set,
        steps,
    } = solved?;

    for hint in blocked_hints {
        register_blocked_hint(hint);
    }

    let mut merged = existing_steps;
    merged.extend(steps);
    Some((solution_set, merged))
}

/// Finalize unsupported-log pipeline against caller-owned mutable step buffer.
///
/// Returns `None` when there is no unsupported pipeline result and preserves
/// `existing_steps` in that case.
pub fn finalize_pow_exponent_log_unsupported_pipeline_with_existing_steps<S, FHint>(
    solved: Option<PowExponentLogUnsupportedPipelineSolved<S>>,
    existing_steps: &mut Vec<S>,
    register_blocked_hint: FHint,
) -> Option<(SolutionSet, Vec<S>)>
where
    FHint: FnMut(LogBlockedHintRecord),
{
    let solved = solved?;
    let existing = std::mem::take(existing_steps);
    merge_pow_exponent_log_unsupported_pipeline_with_existing_steps(
        Some(solved),
        existing,
        register_blocked_hint,
    )
}

/// Plan, execute, and finalize unsupported exponent-log pipeline in one helper.
///
/// Returns `None` when planning reports the route as supported and no
/// unsupported execution should run.
pub fn execute_pow_exponent_log_unsupported_pipeline_from_decision_and_finalize_with_existing_steps_with<
    FG,
    FStep,
    FPlan,
    S,
    FHint,
>(
    include_items: bool,
    existing_steps: &mut Vec<S>,
    plan_execution: FPlan,
    try_guarded_solve: FG,
    map_item_to_step: FStep,
    register_blocked_hint: FHint,
) -> Option<(SolutionSet, Vec<S>)>
where
    FG: FnMut(&Equation) -> Option<SolutionSet>,
    FStep: FnMut(TermIsolationExecutionItem) -> S,
    FPlan: FnOnce() -> Option<PowExponentLogUnsupportedExecution>,
    FHint: FnMut(LogBlockedHintRecord),
{
    let solved = execute_pow_exponent_log_unsupported_pipeline_from_decision_with(
        include_items,
        plan_execution,
        try_guarded_solve,
        map_item_to_step,
    );
    finalize_pow_exponent_log_unsupported_pipeline_with_existing_steps(
        solved,
        existing_steps,
        register_blocked_hint,
    )
}

/// Build narration for eliminating rational exponents by powering both sides.
pub fn eliminate_fractional_exponent_message(q_display: &str) -> String {
    format!(
        "Raise both sides to power {} to eliminate fractional exponent",
        q_display
    )
}

/// Build narration for eliminating rational exponents by powering both sides.
pub fn eliminate_rational_exponent_message(q_display: &str) -> String {
    format!(
        "Raise both sides to power {} to eliminate rational exponent",
        q_display
    )
}

/// Build narration when the solve variable cancels and leaves a parameter constraint.
pub fn variable_canceled_constraint_message(var: &str, diff_display: &str) -> String {
    format!(
        "Variable '{}' canceled during simplification. Solution depends on constraint: {} = 0",
        var, diff_display
    )
}

/// Build equation payload for `diff = 0` parameter constraints.
pub fn build_zero_constraint_equation(ctx: &mut Context, diff: ExprId) -> Equation {
    Equation {
        lhs: diff,
        rhs: ctx.num(0),
        op: RelOp::Eq,
    }
}

/// Build a residual solution set `solve(__eq__(lhs, rhs), var)`.
pub fn residual_solution_set(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
) -> SolutionSet {
    SolutionSet::Residual(residual_expression(ctx, lhs, rhs, var))
}

/// Build residual expression `solve(__eq__(lhs, rhs), var)`.
pub fn residual_expression(ctx: &mut Context, lhs: ExprId, rhs: ExprId, var: &str) -> ExprId {
    mk_residual_solve(ctx, lhs, rhs, var)
}

/// Resolve terminal log-solve actions into concrete solution sets.
pub fn resolve_log_terminal_outcome(
    ctx: &mut Context,
    decision: &LogSolveDecision,
    mode: DomainModeKind,
    wildcard_scope: bool,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
) -> Option<TerminalSolveOutcome> {
    match classify_terminal_action(decision, mode, wildcard_scope) {
        LogTerminalAction::ReturnEmptySet => {
            let LogSolveDecision::EmptySet(message) = decision else {
                return None;
            };
            Some(TerminalSolveOutcome {
                message,
                solutions: SolutionSet::Empty,
            })
        }
        LogTerminalAction::ReturnResidualInWildcard => {
            let LogSolveDecision::NeedsComplex(message) = decision else {
                return None;
            };
            Some(TerminalSolveOutcome {
                message,
                solutions: residual_solution_set(ctx, lhs, rhs, var),
            })
        }
        LogTerminalAction::Continue => None,
    }
}

/// Classify log-domain terminal decision for equations that have exactly one
/// exponential side with the solve variable in exponent position.
pub fn classify_single_side_exponential_log_decision<F>(
    ctx: &Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    lhs_has_var: bool,
    rhs_has_var: bool,
    mut classify_log_solve: F,
) -> Option<LogSolveDecision>
where
    F: FnMut(&Context, ExprId, ExprId) -> LogSolveDecision,
{
    let candidate = crate::isolation_utils::find_single_side_exponential_var_in_exponent(
        ctx,
        lhs,
        rhs,
        var,
        lhs_has_var,
        rhs_has_var,
    )?;
    Some(classify_log_solve(
        ctx,
        candidate.base,
        candidate.other_side,
    ))
}

/// Resolve terminal outcome for equations that have exactly one exponential side
/// with the solve variable in exponent position.
#[allow(clippy::too_many_arguments)]
pub fn resolve_single_side_exponential_terminal_outcome<F>(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    lhs_has_var: bool,
    rhs_has_var: bool,
    mode: DomainModeKind,
    wildcard_scope: bool,
    mut classify_log_solve: F,
) -> Option<TerminalSolveOutcome>
where
    F: FnMut(&Context, ExprId, ExprId) -> LogSolveDecision,
{
    let decision = classify_single_side_exponential_log_decision(
        ctx,
        lhs,
        rhs,
        var,
        lhs_has_var,
        rhs_has_var,
        &mut classify_log_solve,
    )?;
    resolve_log_terminal_outcome(ctx, &decision, mode, wildcard_scope, lhs, rhs, var)
}

/// Resolve a single-side exponential terminal outcome and return the
/// user-facing didactic message together with the terminal solution set.
#[allow(clippy::too_many_arguments)]
pub fn resolve_single_side_exponential_terminal_with_message<F>(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    lhs_has_var: bool,
    rhs_has_var: bool,
    mode: DomainModeKind,
    wildcard_scope: bool,
    residual_suffix: &str,
    classify_log_solve: F,
) -> Option<(SolutionSet, String)>
where
    F: FnMut(&Context, ExprId, ExprId) -> LogSolveDecision,
{
    let outcome = resolve_single_side_exponential_terminal_outcome(
        ctx,
        lhs,
        rhs,
        var,
        lhs_has_var,
        rhs_has_var,
        mode,
        wildcard_scope,
        classify_log_solve,
    )?;
    let message = terminal_outcome_message(&outcome, residual_suffix);
    Some((outcome.solutions, message))
}

/// Resolve single-side exponential terminal outcome and return a didactic step.
#[allow(clippy::too_many_arguments)]
pub fn resolve_single_side_exponential_terminal_with_item<F>(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    lhs_has_var: bool,
    rhs_has_var: bool,
    mode: DomainModeKind,
    wildcard_scope: bool,
    residual_suffix: &str,
    equation_after: Equation,
    classify_log_solve: F,
) -> Option<(SolutionSet, TermIsolationExecutionItem)>
where
    F: FnMut(&Context, ExprId, ExprId) -> LogSolveDecision,
{
    let (solutions, message) = resolve_single_side_exponential_terminal_with_message(
        ctx,
        lhs,
        rhs,
        var,
        lhs_has_var,
        rhs_has_var,
        mode,
        wildcard_scope,
        residual_suffix,
        classify_log_solve,
    )?;
    Some((
        solutions,
        TermIsolationExecutionItem {
            description: message,
            equation: equation_after,
        },
    ))
}

/// Solved payload for single-side exponential terminal pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct SingleSideExponentialTerminalSolved<S> {
    pub solution_set: SolutionSet,
    pub steps: Vec<S>,
}

/// Resolve single-side exponential terminal outcome with optional item dispatch.
#[allow(clippy::too_many_arguments)]
pub fn resolve_single_side_exponential_terminal_pipeline_with_item<FClassify, FStep, S>(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    lhs_has_var: bool,
    rhs_has_var: bool,
    mode: DomainModeKind,
    wildcard_scope: bool,
    residual_suffix: &str,
    equation_after: Equation,
    include_item: bool,
    classify_log_solve: FClassify,
    mut map_item_to_step: FStep,
) -> Option<SingleSideExponentialTerminalSolved<S>>
where
    FClassify: FnMut(&Context, ExprId, ExprId) -> LogSolveDecision,
    FStep: FnMut(TermIsolationExecutionItem) -> S,
{
    let (solution_set, item) = resolve_single_side_exponential_terminal_with_item(
        ctx,
        lhs,
        rhs,
        var,
        lhs_has_var,
        rhs_has_var,
        mode,
        wildcard_scope,
        residual_suffix,
        equation_after,
        classify_log_solve,
    )?;
    let mut steps = Vec::new();
    if include_item {
        steps.push(map_item_to_step(item));
    }
    Some(SingleSideExponentialTerminalSolved {
        solution_set,
        steps,
    })
}

/// Resolve single-side exponential terminal outcome and return a didactic step.
#[allow(clippy::too_many_arguments)]
pub fn resolve_single_side_exponential_terminal_with_step<F>(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    lhs_has_var: bool,
    rhs_has_var: bool,
    mode: DomainModeKind,
    wildcard_scope: bool,
    residual_suffix: &str,
    equation_after: Equation,
    classify_log_solve: F,
) -> Option<(SolutionSet, TermIsolationDidacticStep)>
where
    F: FnMut(&Context, ExprId, ExprId) -> LogSolveDecision,
{
    let (solutions, item) = resolve_single_side_exponential_terminal_with_item(
        ctx,
        lhs,
        rhs,
        var,
        lhs_has_var,
        rhs_has_var,
        mode,
        wildcard_scope,
        residual_suffix,
        equation_after,
        classify_log_solve,
    )?;
    Some((
        solutions,
        term_isolation_didactic_step_from_execution_item(item),
    ))
}

/// Build a user-facing message for terminal outcomes, appending
/// `residual_suffix` only when the outcome is residual.
pub fn terminal_outcome_message(outcome: &TerminalSolveOutcome, residual_suffix: &str) -> String {
    if matches!(outcome.solutions, SolutionSet::Residual(_)) {
        format!("{}{}", outcome.message, residual_suffix)
    } else {
        outcome.message.to_string()
    }
}

/// Resolve unsupported-log rewrite decisions into concrete residual/guarded outcomes.
#[allow(clippy::too_many_arguments)]
pub fn resolve_log_unsupported_outcome<'a>(
    ctx: &mut Context,
    decision: &'a LogSolveDecision,
    can_branch: bool,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    base: ExprId,
    rhs_expr: ExprId,
) -> Option<LogUnsupportedOutcome<'a>> {
    match classify_log_unsupported_route(decision, can_branch) {
        LogUnsupportedRoute::NotUnsupported => None,
        LogUnsupportedRoute::ResidualBudgetExhausted { message } => {
            Some(LogUnsupportedOutcome::ResidualBudgetExhausted {
                message,
                solutions: residual_solution_set(ctx, lhs, rhs, var),
            })
        }
        LogUnsupportedRoute::Guarded {
            message,
            missing_conditions,
        } => Some(LogUnsupportedOutcome::Guarded {
            message,
            missing_conditions,
            guard: assumptions_to_condition_set(missing_conditions, base, rhs_expr),
            residual: residual_expression(ctx, lhs, rhs, var),
        }),
    }
}

/// Pre-check decision for absolute-value equalities `|A| = RHS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbsEqualityPrecheck {
    /// `RHS < 0` -> impossible.
    ReturnEmptySet,
    /// `RHS = 0` -> reduce to `A = 0`.
    CollapseToZero,
    /// `RHS > 0` -> keep normal branch split.
    Continue,
}

/// Fast-path routing for absolute-value isolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbsIsolationFastPath {
    ReturnEmptySet,
    CollapseToZero,
    Continue,
}

/// Branch labels for absolute-value split rewrites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbsSplitCase {
    Positive,
    Negative,
}

/// High-level routing plan for isolating `|A| op RHS`.
#[derive(Debug, Clone, PartialEq)]
pub enum AbsIsolationPlan {
    ReturnEmptySet,
    IsolateSingleEquation {
        equation: Equation,
    },
    SplitBranches {
        positive: Equation,
        negative: Equation,
    },
}

/// Solved outcome for absolute-value isolation planning.
#[derive(Debug, Clone, PartialEq)]
pub enum AbsIsolationSolved<TSingle, TSplit> {
    ReturnedEmptySet,
    IsolatedSingle(TSingle),
    Split(TSplit),
}

/// Execute absolute-value isolation plan with caller-provided callbacks.
pub fn solve_abs_isolation_plan_with<E, TSingle, TSplit, FSingle, FSplit>(
    plan: AbsIsolationPlan,
    mut solve_single: FSingle,
    mut solve_split: FSplit,
) -> Result<AbsIsolationSolved<TSingle, TSplit>, E>
where
    FSingle: FnMut(Equation) -> Result<TSingle, E>,
    FSplit: FnMut(Equation, Equation) -> Result<TSplit, E>,
{
    match plan {
        AbsIsolationPlan::ReturnEmptySet => Ok(AbsIsolationSolved::ReturnedEmptySet),
        AbsIsolationPlan::IsolateSingleEquation { equation } => {
            Ok(AbsIsolationSolved::IsolatedSingle(solve_single(equation)?))
        }
        AbsIsolationPlan::SplitBranches { positive, negative } => {
            Ok(AbsIsolationSolved::Split(solve_split(positive, negative)?))
        }
    }
}

/// Execute absolute-value isolation plan and route each outcome through the
/// corresponding solve pipeline while preserving caller-provided existing steps.
#[allow(clippy::too_many_arguments)]
pub fn execute_abs_isolation_plan_pipeline_with_optional_items<
    E,
    S,
    FSolveSingle,
    FRenderExpr,
    FSolveBranch,
    FMapStep,
    FFinalize,
>(
    plan: AbsIsolationPlan,
    lhs_expr: ExprId,
    include_items: bool,
    existing_steps: Vec<S>,
    render_expr: FRenderExpr,
    mut solve_single_equation: FSolveSingle,
    solve_branch: FSolveBranch,
    map_item_to_step: FMapStep,
    finalize_solved_sets: FFinalize,
) -> Result<(SolutionSet, Vec<S>), E>
where
    S: Clone,
    FSolveSingle: FnMut(Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FRenderExpr: FnMut(ExprId) -> String,
    FSolveBranch: FnMut(&Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(AbsSplitExecutionItem) -> S,
    FFinalize: FnMut(SolutionSet, SolutionSet) -> SolutionSet,
{
    match plan {
        AbsIsolationPlan::ReturnEmptySet => Ok((SolutionSet::Empty, existing_steps)),
        AbsIsolationPlan::IsolateSingleEquation { equation } => {
            Ok(merge_solved_with_existing_steps_prepend(
                solve_single_equation(equation)?,
                existing_steps,
            ))
        }
        AbsIsolationPlan::SplitBranches { positive, negative } => {
            execute_abs_split_pipeline_with_optional_items(
                positive,
                negative,
                lhs_expr,
                include_items,
                &existing_steps,
                render_expr,
                solve_branch,
                map_item_to_step,
                finalize_solved_sets,
            )
        }
    }
}

/// Convenience variant of `execute_abs_isolation_plan_pipeline_with_optional_items`
/// when single-equation and split-branch paths share the same equation solver.
#[allow(clippy::too_many_arguments)]
pub fn execute_abs_isolation_plan_pipeline_with_optional_items_and_solver<
    E,
    S,
    FSolveEquation,
    FRenderExpr,
    FMapStep,
    FFinalize,
>(
    plan: AbsIsolationPlan,
    lhs_expr: ExprId,
    include_items: bool,
    existing_steps: Vec<S>,
    render_expr: FRenderExpr,
    mut solve_equation: FSolveEquation,
    map_item_to_step: FMapStep,
    finalize_solved_sets: FFinalize,
) -> Result<(SolutionSet, Vec<S>), E>
where
    S: Clone,
    FSolveEquation: FnMut(&Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FRenderExpr: FnMut(ExprId) -> String,
    FMapStep: FnMut(AbsSplitExecutionItem) -> S,
    FFinalize: FnMut(SolutionSet, SolutionSet) -> SolutionSet,
{
    match plan {
        AbsIsolationPlan::ReturnEmptySet => Ok((SolutionSet::Empty, existing_steps)),
        AbsIsolationPlan::IsolateSingleEquation { equation } => Ok(
            merge_solved_with_existing_steps_prepend(solve_equation(&equation)?, existing_steps),
        ),
        AbsIsolationPlan::SplitBranches { positive, negative } => {
            execute_abs_split_pipeline_with_optional_items(
                positive,
                negative,
                lhs_expr,
                include_items,
                &existing_steps,
                render_expr,
                solve_equation,
                map_item_to_step,
                finalize_solved_sets,
            )
        }
    }
}

/// Pre-built sign-split equations for inequalities of the form `A*B op 0`.
#[derive(Debug, Clone, PartialEq)]
pub struct ProductZeroInequalityPlan {
    pub case1_left: Equation,
    pub case1_right: Equation,
    pub case2_left: Equation,
    pub case2_right: Equation,
}

/// Solved solution-set payload for product-zero inequality branch equations.
#[derive(Debug, Clone, PartialEq)]
pub struct ProductZeroInequalitySolvedSets {
    pub case1_left: SolutionSet,
    pub case1_right: SolutionSet,
    pub case2_left: SolutionSet,
    pub case2_right: SolutionSet,
}

/// Solved payload for denominator-sign split:
/// `(num/den) op rhs` under `den > 0` and `den < 0`.
#[derive(Debug, Clone, PartialEq)]
pub struct DivisionDenominatorSignSplitSolvedCases<TBranch, TDomain> {
    pub positive_branch: TBranch,
    pub negative_branch: TBranch,
    pub positive_domain: TDomain,
    pub negative_domain: TDomain,
}

/// Solved payload for already-isolated denominator sign split:
/// `den op rhs` under `den > 0` and `den < 0`.
#[derive(Debug, Clone, PartialEq)]
pub struct IsolatedDenominatorSignSplitSolvedCases<TBranch> {
    pub positive_branch: TBranch,
    pub negative_branch: TBranch,
}

/// Pre-built split plan for inequalities with variable denominator:
/// `(num / den) op rhs` under `den > 0` and `den < 0`.
#[derive(Debug, Clone, PartialEq)]
pub struct DivisionDenominatorSignSplitPlan {
    pub positive_equation: Equation,
    pub negative_equation: Equation,
    pub positive_domain: Equation,
    pub negative_domain: Equation,
}

/// Pre-built split plan for isolated denominator inequalities:
/// `den op rhs` under `den > 0` and `den < 0`.
#[derive(Debug, Clone, PartialEq)]
pub struct IsolatedDenominatorSignSplitPlan {
    pub positive_equation: Equation,
    pub negative_equation: Equation,
}

/// Executable split plan + didactic payload for division-denominator sign cases.
#[derive(Debug, Clone, PartialEq)]
pub struct DivisionDenominatorSignSplitExecutionPlan {
    pub positive_equation: Equation,
    pub negative_equation: Equation,
    pub positive_domain: Equation,
    pub negative_domain: Equation,
    pub items: Vec<DivisionDidacticExecutionItem>,
}

/// Executable split plan + didactic payload for isolated-denominator sign cases.
#[derive(Debug, Clone, PartialEq)]
pub struct IsolatedDenominatorSignSplitExecutionPlan {
    pub positive_equation: Equation,
    pub negative_equation: Equation,
    pub items: Vec<DivisionDidacticExecutionItem>,
}

/// Executable split plan + didactic payload for absolute-value branch splits.
#[derive(Debug, Clone, PartialEq)]
pub struct AbsSplitExecutionPlan {
    pub positive_equation: Equation,
    pub negative_equation: Equation,
    pub items: Vec<AbsSplitExecutionItem>,
}

/// Pre-built didactic equations for denominator-isolation rewrite:
/// `num / den op rhs` -> `num op rhs*den` -> `den op num/rhs`.
#[derive(Debug, Clone, PartialEq)]
pub struct DivisionDenominatorDidacticPlan {
    pub multiply_equation: Equation,
    pub divide_equation: Equation,
    pub multiply_by: ExprId,
    pub divide_by: ExprId,
}

/// Didactic payload for the two explicit denominator-isolation steps.
#[derive(Debug, Clone, PartialEq)]
pub struct DivisionDenominatorDidacticSteps {
    pub items: Vec<DivisionDidacticExecutionItem>,
}

/// Collect denominator-isolation didactic steps in display order:
/// multiply step first, divide step second.
pub fn collect_division_denominator_didactic_steps(
    didactic: &DivisionDenominatorDidacticSteps,
) -> Vec<DivisionCaseDidacticStep> {
    didactic
        .items
        .iter()
        .cloned()
        .map(|item| DivisionCaseDidacticStep {
            description: item.description,
            equation_after: item.equation,
        })
        .collect()
}

/// Collect denominator-isolation execution items in display order:
/// multiply step first, divide step second.
pub fn collect_division_denominator_execution_items(
    execution: &DivisionDenominatorDidacticExecutionPlan,
) -> Vec<DivisionDidacticExecutionItem> {
    execution.items.clone()
}

/// Backward-compatible alias for denominator-isolation execution item collection.
pub fn collect_division_denominator_didactic_execution_items(
    execution: &DivisionDenominatorDidacticExecutionPlan,
) -> Vec<DivisionDidacticExecutionItem> {
    collect_division_denominator_execution_items(execution)
}

/// Execute denominator-isolation didactic plan while passing aligned execution
/// items and the rewritten equation (`divide_equation`) to the solve callback.
pub fn solve_division_denominator_execution_with_items<E, T, FSolve>(
    execution: DivisionDenominatorDidacticExecutionPlan,
    mut solve_rewritten: FSolve,
) -> Result<DivisionDenominatorDidacticSolved<T>, E>
where
    FSolve: FnMut(Vec<DivisionDidacticExecutionItem>, &Equation) -> Result<T, E>,
{
    let items = collect_division_denominator_execution_items(&execution);
    let solved = solve_rewritten(items, &execution.divide_equation)?;
    Ok(DivisionDenominatorDidacticSolved { execution, solved })
}

/// Executable denominator-isolation didactic plan, with optional simplification
/// already applied to the multiply-step RHS.
#[derive(Debug, Clone, PartialEq)]
pub struct DivisionDenominatorDidacticExecutionPlan {
    pub multiply_equation: Equation,
    pub divide_equation: Equation,
    pub items: Vec<DivisionDidacticExecutionItem>,
}

/// Solved payload for denominator-isolation didactic execution.
#[derive(Debug, Clone, PartialEq)]
pub struct DivisionDenominatorDidacticSolved<T> {
    pub execution: DivisionDenominatorDidacticExecutionPlan,
    pub solved: T,
}

/// Solved payload for denominator-isolation execution pipeline:
/// final solution set plus concatenated didactic and recursive steps.
#[derive(Debug, Clone, PartialEq)]
pub struct DivisionDenominatorExecutionPipelineSolved<TStep> {
    pub solution_set: SolutionSet,
    pub steps: Vec<TStep>,
}

/// Execute denominator-isolation didactic pipeline while mapping didactic items
/// to caller step payloads and concatenating them in front of recursive sub-steps.
pub fn solve_division_denominator_execution_pipeline_with_items<
    E,
    TStep,
    FSolveRewritten,
    FMapStep,
>(
    execution: DivisionDenominatorDidacticExecutionPlan,
    mut solve_rewritten: FSolveRewritten,
    mut map_step: FMapStep,
) -> Result<DivisionDenominatorExecutionPipelineSolved<TStep>, E>
where
    FSolveRewritten: FnMut(&Equation) -> Result<(SolutionSet, Vec<TStep>), E>,
    FMapStep: FnMut(DivisionDidacticExecutionItem) -> TStep,
{
    let items = collect_division_denominator_execution_items(&execution);
    let (solution_set, sub_steps) = solve_rewritten(&execution.divide_equation)?;
    let mut steps = items.into_iter().map(&mut map_step).collect::<Vec<_>>();
    steps.extend(sub_steps);
    Ok(DivisionDenominatorExecutionPipelineSolved {
        solution_set,
        steps,
    })
}

/// Solve denominator-isolation plan with optional didactic items.
///
/// When `include_items` is enabled:
/// 1) simplify multiply-step RHS,
/// 2) build didactic execution payload,
/// 3) prepend mapped didactic steps before recursive substeps.
///
/// When disabled:
/// - solve only the rewritten `divide_equation` directly.
pub fn solve_division_denominator_pipeline_with_optional_items<
    E,
    TStep,
    FRenderExpr,
    FSolveRewritten,
    FMapStep,
>(
    didactic_plan: DivisionDenominatorDidacticPlan,
    include_items: bool,
    simplified_multiply_rhs: ExprId,
    render_expr: FRenderExpr,
    mut solve_rewritten: FSolveRewritten,
    map_step: FMapStep,
) -> Result<DivisionDenominatorExecutionPipelineSolved<TStep>, E>
where
    FRenderExpr: FnMut(ExprId) -> String,
    FSolveRewritten: FnMut(&Equation) -> Result<(SolutionSet, Vec<TStep>), E>,
    FMapStep: FnMut(DivisionDidacticExecutionItem) -> TStep,
{
    if include_items {
        let execution = build_division_denominator_execution_with(
            didactic_plan,
            simplified_multiply_rhs,
            render_expr,
        );
        return solve_division_denominator_execution_pipeline_with_items(
            execution,
            solve_rewritten,
            map_step,
        );
    }

    let (solution_set, steps) = solve_rewritten(&didactic_plan.divide_equation)?;
    Ok(DivisionDenominatorExecutionPipelineSolved {
        solution_set,
        steps,
    })
}

/// Execute denominator-isolation didactic plan with optional items.
///
/// This combines:
/// 1. Optional simplification of multiply-step RHS (only when didactic items are enabled).
/// 2. Didactic pipeline execution.
/// 3. Conversion into `(SolutionSet, steps)` tuple for engine call sites.
pub fn execute_division_denominator_plan_with_optional_items<
    E,
    TStep,
    FSimplifyExpr,
    FRenderExpr,
    FSolveRewritten,
    FMapStep,
>(
    didactic_plan: DivisionDenominatorDidacticPlan,
    include_items: bool,
    mut simplify_expr: FSimplifyExpr,
    render_expr: FRenderExpr,
    solve_rewritten: FSolveRewritten,
    map_step: FMapStep,
) -> Result<(SolutionSet, Vec<TStep>), E>
where
    FSimplifyExpr: FnMut(ExprId) -> ExprId,
    FRenderExpr: FnMut(ExprId) -> String,
    FSolveRewritten: FnMut(&Equation) -> Result<(SolutionSet, Vec<TStep>), E>,
    FMapStep: FnMut(DivisionDidacticExecutionItem) -> TStep,
{
    let simplified_multiply_rhs = if include_items {
        simplify_expr(didactic_plan.multiply_equation.rhs)
    } else {
        didactic_plan.multiply_equation.rhs
    };

    let solved = solve_division_denominator_pipeline_with_optional_items(
        didactic_plan,
        include_items,
        simplified_multiply_rhs,
        render_expr,
        solve_rewritten,
        map_step,
    )?;
    Ok((solved.solution_set, solved.steps))
}

/// Execute denominator-isolation didactic plan with optional items and prepend
/// solved steps before caller-owned existing steps.
pub fn execute_division_denominator_plan_with_optional_items_and_merge_with_existing_steps_with<
    E,
    TStep,
    FSimplifyExpr,
    FRenderExpr,
    FSolveRewritten,
    FMapStep,
>(
    didactic_plan: DivisionDenominatorDidacticPlan,
    include_items: bool,
    existing_steps: Vec<TStep>,
    simplify_expr: FSimplifyExpr,
    render_expr: FRenderExpr,
    solve_rewritten: FSolveRewritten,
    map_step: FMapStep,
) -> Result<(SolutionSet, Vec<TStep>), E>
where
    FSimplifyExpr: FnMut(ExprId) -> ExprId,
    FRenderExpr: FnMut(ExprId) -> String,
    FSolveRewritten: FnMut(&Equation) -> Result<(SolutionSet, Vec<TStep>), E>,
    FMapStep: FnMut(DivisionDidacticExecutionItem) -> TStep,
{
    let solved = execute_division_denominator_plan_with_optional_items(
        didactic_plan,
        include_items,
        simplify_expr,
        render_expr,
        solve_rewritten,
        map_step,
    )?;
    Ok(merge_solved_with_existing_steps_prepend(
        solved,
        existing_steps,
    ))
}

/// Classify numeric RHS sign for `|A| = RHS`.
pub fn abs_equality_precheck(sign: NumericSign) -> AbsEqualityPrecheck {
    match sign {
        NumericSign::Negative => AbsEqualityPrecheck::ReturnEmptySet,
        NumericSign::Zero => AbsEqualityPrecheck::CollapseToZero,
        NumericSign::Positive => AbsEqualityPrecheck::Continue,
    }
}

/// Decide fast-path behavior for absolute-value isolation from operator and RHS sign.
pub fn classify_abs_isolation_fast_path(
    op: RelOp,
    rhs_sign: Option<NumericSign>,
) -> AbsIsolationFastPath {
    if op != RelOp::Eq {
        return AbsIsolationFastPath::Continue;
    }
    match rhs_sign.map(abs_equality_precheck) {
        Some(AbsEqualityPrecheck::ReturnEmptySet) => AbsIsolationFastPath::ReturnEmptySet,
        Some(AbsEqualityPrecheck::CollapseToZero) => AbsIsolationFastPath::CollapseToZero,
        _ => AbsIsolationFastPath::Continue,
    }
}

/// Build executable split plan for product inequalities `A*B op 0`.
///
/// Returns `None` for non-inequality operators (`=` / `!=`).
pub fn plan_product_zero_inequality_split(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    op: RelOp,
) -> Option<ProductZeroInequalityPlan> {
    let (case1_ops, case2_ops) = crate::isolation_utils::product_zero_inequality_cases(op)?;
    let (case1_left, case1_right) =
        crate::equation_rewrite::build_product_zero_sign_case(ctx, left, right, &case1_ops);
    let (case2_left, case2_right) =
        crate::equation_rewrite::build_product_zero_sign_case(ctx, left, right, &case2_ops);
    Some(ProductZeroInequalityPlan {
        case1_left,
        case1_right,
        case2_left,
        case2_right,
    })
}

/// Build product-zero inequality split plan only when split preconditions are
/// met (`left`/`right` contain target var, `rhs` is zero and `op` is inequality).
pub fn plan_product_zero_inequality_split_if_applicable(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
) -> Option<ProductZeroInequalityPlan> {
    if crate::isolation_utils::should_split_product_zero_inequality(ctx, left, right, rhs, &op, var)
    {
        plan_product_zero_inequality_split(ctx, left, right, op)
    } else {
        None
    }
}

/// Try product-zero inequality split planning and execute the full split
/// pipeline when applicable.
pub fn try_execute_product_zero_inequality_split_pipeline_with_existing_steps<
    E,
    S,
    FSolveCase,
    FFinalize,
>(
    plan: Option<ProductZeroInequalityPlan>,
    existing_steps: &[S],
    solve_case: FSolveCase,
    finalize_solved_sets: FFinalize,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    S: Clone,
    FSolveCase: FnMut(&Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FFinalize: FnMut(ProductZeroInequalitySolvedSets) -> SolutionSet,
{
    let plan = plan?;
    Some(
        execute_product_zero_inequality_split_pipeline_with_existing_steps(
            &plan,
            existing_steps.to_vec(),
            solve_case,
            finalize_solved_sets,
        ),
    )
}

/// Solve each equation of a product-zero inequality split with caller-provided
/// equation solver callback.
pub fn solve_product_zero_inequality_cases_with<E, FSolve>(
    plan: &ProductZeroInequalityPlan,
    mut solve_equation: FSolve,
) -> Result<ProductZeroInequalitySolvedSets, E>
where
    FSolve: FnMut(&Equation) -> Result<SolutionSet, E>,
{
    Ok(ProductZeroInequalitySolvedSets {
        case1_left: solve_equation(&plan.case1_left)?,
        case1_right: solve_equation(&plan.case1_right)?,
        case2_left: solve_equation(&plan.case2_left)?,
        case2_right: solve_equation(&plan.case2_right)?,
    })
}

/// Finalize solved product-zero branch sets into one solution set.
pub fn finalize_product_zero_inequality_solved_sets(
    ctx: &Context,
    solved_sets: ProductZeroInequalitySolvedSets,
) -> SolutionSet {
    crate::solution_set::finalize_product_zero_inequality_solution_set(
        ctx,
        solved_sets.case1_left,
        solved_sets.case1_right,
        solved_sets.case2_left,
        solved_sets.case2_right,
    )
}

/// Solved payload for product-zero inequality execution:
/// solved branch sets plus concatenated branch steps.
#[derive(Debug, Clone, PartialEq)]
pub struct ProductZeroInequalityExecutionSolved<TStep> {
    pub solved_sets: ProductZeroInequalitySolvedSets,
    pub steps: Vec<TStep>,
}

/// Execute product-zero inequality split by solving the four branch equations
/// and concatenating their steps in branch-evaluation order.
pub fn solve_product_zero_inequality_split_execution_with<E, TStep, FSolveCase>(
    plan: &ProductZeroInequalityPlan,
    mut solve_case: FSolveCase,
) -> Result<ProductZeroInequalityExecutionSolved<TStep>, E>
where
    FSolveCase: FnMut(&Equation) -> Result<(SolutionSet, Vec<TStep>), E>,
{
    let mut steps = Vec::new();
    let (case1_left, mut case1_left_steps) = solve_case(&plan.case1_left)?;
    steps.append(&mut case1_left_steps);
    let (case1_right, mut case1_right_steps) = solve_case(&plan.case1_right)?;
    steps.append(&mut case1_right_steps);
    let (case2_left, mut case2_left_steps) = solve_case(&plan.case2_left)?;
    steps.append(&mut case2_left_steps);
    let (case2_right, mut case2_right_steps) = solve_case(&plan.case2_right)?;
    steps.append(&mut case2_right_steps);

    let solved_sets = ProductZeroInequalitySolvedSets {
        case1_left,
        case1_right,
        case2_left,
        case2_right,
    };
    Ok(ProductZeroInequalityExecutionSolved { solved_sets, steps })
}

/// Execute product-zero inequality split pipeline and return finalized solution
/// set merged with caller-provided existing steps.
pub fn execute_product_zero_inequality_split_pipeline_with_existing_steps<
    E,
    S,
    FSolveCase,
    FFinalize,
>(
    plan: &ProductZeroInequalityPlan,
    existing_steps: Vec<S>,
    solve_case: FSolveCase,
    mut finalize_solved_sets: FFinalize,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FSolveCase: FnMut(&Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FFinalize: FnMut(ProductZeroInequalitySolvedSets) -> SolutionSet,
{
    let solved = solve_product_zero_inequality_split_execution_with(plan, solve_case)?;
    let final_set = finalize_solved_sets(solved.solved_sets);
    Ok(merge_solved_with_existing_steps_prepend(
        (final_set, solved.steps),
        existing_steps,
    ))
}

/// Solve branch equations + sign-domain constraints for division denominator
/// sign split execution.
pub fn solve_division_denominator_sign_split_cases_with<
    E,
    TBranch,
    TDomain,
    FSolveBranch,
    FSolveDomain,
>(
    execution: &DivisionDenominatorSignSplitExecutionPlan,
    mut solve_branch: FSolveBranch,
    mut solve_domain: FSolveDomain,
) -> Result<DivisionDenominatorSignSplitSolvedCases<TBranch, TDomain>, E>
where
    FSolveBranch: FnMut(&Equation) -> Result<TBranch, E>,
    FSolveDomain: FnMut(&Equation) -> Result<TDomain, E>,
{
    Ok(DivisionDenominatorSignSplitSolvedCases {
        positive_branch: solve_branch(&execution.positive_equation)?,
        positive_domain: solve_domain(&execution.positive_domain)?,
        negative_branch: solve_branch(&execution.negative_equation)?,
        negative_domain: solve_domain(&execution.negative_domain)?,
    })
}

/// Solve denominator-sign split branch equations while passing aligned optional
/// didactic execution items for each branch callback.
pub fn solve_division_denominator_sign_split_cases_with_items<
    E,
    TBranch,
    TDomain,
    FSolveBranch,
    FSolveDomain,
>(
    execution: &DivisionDenominatorSignSplitExecutionPlan,
    mut solve_branch: FSolveBranch,
    mut solve_domain: FSolveDomain,
) -> Result<DivisionDenominatorSignSplitSolvedCases<TBranch, TDomain>, E>
where
    FSolveBranch: FnMut(Option<DivisionDidacticExecutionItem>, &Equation) -> Result<TBranch, E>,
    FSolveDomain: FnMut(&Equation) -> Result<TDomain, E>,
{
    let mut items = collect_division_denominator_sign_split_execution_items(execution).into_iter();
    Ok(DivisionDenominatorSignSplitSolvedCases {
        positive_branch: solve_branch(items.next(), &execution.positive_equation)?,
        positive_domain: solve_domain(&execution.positive_domain)?,
        negative_branch: solve_branch(items.next(), &execution.negative_equation)?,
        negative_domain: solve_domain(&execution.negative_domain)?,
    })
}

/// Solved payload for denominator-sign split execution:
/// solved branch/domain sets plus concatenated steps from both branches.
#[derive(Debug, Clone, PartialEq)]
pub struct DivisionDenominatorSignSplitExecutionSolved<TStep> {
    pub positive_set: SolutionSet,
    pub negative_set: SolutionSet,
    pub positive_domain_set: SolutionSet,
    pub negative_domain_set: SolutionSet,
    pub steps: Vec<TStep>,
}

/// Pop one didactic item from an optional execution-item iterator.
fn take_optional_item<I>(items: &mut Option<I>) -> Option<I::Item>
where
    I: Iterator,
{
    items.as_mut().and_then(|iter| iter.next())
}

/// Solve denominator-sign split execution with aligned optional didactic items,
/// keeping branch/domain solution sets separate and concatenating branch steps
/// with the case boundary marker between them when present.
pub fn solve_division_denominator_sign_split_execution_with_items<
    E,
    TStep,
    FSolveBranch,
    FSolveDomain,
    FMapBoundary,
>(
    execution: &DivisionDenominatorSignSplitExecutionPlan,
    solve_branch: FSolveBranch,
    solve_domain: FSolveDomain,
    mut map_boundary_item: FMapBoundary,
) -> Result<DivisionDenominatorSignSplitExecutionSolved<TStep>, E>
where
    FSolveBranch: FnMut(
        Option<DivisionDidacticExecutionItem>,
        &Equation,
    ) -> Result<(SolutionSet, Vec<TStep>), E>,
    FSolveDomain: FnMut(&Equation) -> Result<SolutionSet, E>,
    FMapBoundary: FnMut(DivisionDidacticExecutionItem) -> TStep,
{
    let solved = solve_division_denominator_sign_split_cases_with_items(
        execution,
        solve_branch,
        solve_domain,
    )?;
    let DivisionDenominatorSignSplitSolvedCases {
        positive_branch: (positive_set, mut positive_steps),
        negative_branch: (negative_set, negative_steps),
        positive_domain: positive_domain_set,
        negative_domain: negative_domain_set,
    } = solved;

    if let Some(item) = division_denominator_sign_split_boundary_item(execution) {
        positive_steps.push(map_boundary_item(item));
    }
    positive_steps.extend(negative_steps);

    Ok(DivisionDenominatorSignSplitExecutionSolved {
        positive_set,
        negative_set,
        positive_domain_set,
        negative_domain_set,
        steps: positive_steps,
    })
}

/// Solve denominator-sign split branches while prepending shared branch steps
/// and optionally mapping per-branch didactic items.
pub fn solve_division_denominator_sign_split_execution_pipeline_with_items<
    E,
    S,
    FSolveBranch,
    FSolveDomain,
    FStep,
>(
    execution: &DivisionDenominatorSignSplitExecutionPlan,
    include_items: bool,
    branch_prefix_steps: &[S],
    mut solve_branch: FSolveBranch,
    mut solve_domain: FSolveDomain,
    mut map_item_to_step: FStep,
) -> Result<DivisionDenominatorSignSplitExecutionSolved<S>, E>
where
    S: Clone,
    FSolveBranch: FnMut(&Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FSolveDomain: FnMut(&Equation) -> Result<SolutionSet, E>,
    FStep: FnMut(DivisionDidacticExecutionItem) -> S,
{
    let mut items = if include_items {
        Some(collect_division_denominator_sign_split_execution_items(execution).into_iter())
    } else {
        None
    };

    let mut positive_steps = branch_prefix_steps.to_vec();
    if let Some(item) = take_optional_item(&mut items) {
        positive_steps.push(map_item_to_step(item));
    }
    let (positive_set, positive_sub_steps) = solve_branch(&execution.positive_equation)?;
    positive_steps.extend(positive_sub_steps);
    let positive_domain_set = solve_domain(&execution.positive_domain)?;

    let mut negative_steps = branch_prefix_steps.to_vec();
    if let Some(item) = take_optional_item(&mut items) {
        negative_steps.push(map_item_to_step(item));
    }
    let (negative_set, negative_sub_steps) = solve_branch(&execution.negative_equation)?;
    negative_steps.extend(negative_sub_steps);
    let negative_domain_set = solve_domain(&execution.negative_domain)?;

    if let Some(item) = take_optional_item(&mut items) {
        positive_steps.push(map_item_to_step(item));
    }
    positive_steps.extend(negative_steps);

    Ok(DivisionDenominatorSignSplitExecutionSolved {
        positive_set,
        negative_set,
        positive_domain_set,
        negative_domain_set,
        steps: positive_steps,
    })
}

/// Solve denominator-sign split with optional didactic items.
///
/// When `include_items` is enabled, this builds didactic execution items and
/// prepends them to per-branch solve steps. When disabled, branch equations and
/// domain guards are solved without rendering didactic payload.
#[allow(clippy::too_many_arguments)]
pub fn solve_division_denominator_sign_split_pipeline_with_optional_items<
    E,
    S,
    FRenderExpr,
    FSolveBranch,
    FSolveDomain,
    FMapStep,
>(
    split_plan: DivisionDenominatorSignSplitPlan,
    denominator: ExprId,
    case_boundary_lhs: ExprId,
    case_boundary_op: RelOp,
    simplified_rhs: ExprId,
    include_items: bool,
    branch_prefix_steps: &[S],
    render_expr: FRenderExpr,
    solve_branch: FSolveBranch,
    solve_domain: FSolveDomain,
    map_item_to_step: FMapStep,
) -> Result<DivisionDenominatorSignSplitExecutionSolved<S>, E>
where
    S: Clone,
    FRenderExpr: FnMut(ExprId) -> String,
    FSolveBranch: FnMut(&Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FSolveDomain: FnMut(&Equation) -> Result<SolutionSet, E>,
    FMapStep: FnMut(DivisionDidacticExecutionItem) -> S,
{
    let execution = if include_items {
        build_division_denominator_sign_split_execution_with(
            split_plan,
            denominator,
            case_boundary_lhs,
            case_boundary_op,
            simplified_rhs,
            render_expr,
        )
    } else {
        materialize_division_denominator_sign_split_execution(split_plan, simplified_rhs)
    };
    solve_division_denominator_sign_split_execution_pipeline_with_items(
        &execution,
        include_items,
        branch_prefix_steps,
        solve_branch,
        solve_domain,
        map_item_to_step,
    )
}

/// Return the boundary didactic item (`--- End of Case 1 ---`) when available.
pub fn division_denominator_sign_split_boundary_item(
    execution: &DivisionDenominatorSignSplitExecutionPlan,
) -> Option<DivisionDidacticExecutionItem> {
    collect_division_denominator_sign_split_execution_items(execution)
        .into_iter()
        .nth(2)
}

/// Finalize solved denominator-sign split branch/domain sets into one set.
pub fn finalize_division_denominator_sign_split_solved_sets(
    ctx: &Context,
    solved: DivisionDenominatorSignSplitSolvedCases<SolutionSet, SolutionSet>,
) -> SolutionSet {
    crate::solution_set::finalize_sign_split_solution_set(
        ctx,
        solved.positive_branch,
        solved.positive_domain,
        solved.negative_branch,
        solved.negative_domain,
    )
}

/// Execute denominator-sign split pipeline and finalize branch/domain sets.
///
/// This combines:
/// 1. Pipeline execution with optional didactic items.
/// 2. Conversion of solved branch/domain payload into finalized `SolutionSet`.
#[allow(clippy::too_many_arguments)]
pub fn execute_division_denominator_sign_split_pipeline_with_optional_items<
    E,
    S,
    FRenderExpr,
    FSolveBranch,
    FSolveDomain,
    FMapStep,
    FFinalize,
>(
    split_plan: DivisionDenominatorSignSplitPlan,
    denominator: ExprId,
    case_boundary_lhs: ExprId,
    case_boundary_op: RelOp,
    simplified_rhs: ExprId,
    include_items: bool,
    branch_prefix_steps: &[S],
    render_expr: FRenderExpr,
    solve_branch: FSolveBranch,
    solve_domain: FSolveDomain,
    map_item_to_step: FMapStep,
    mut finalize_solved_sets: FFinalize,
) -> Result<(SolutionSet, Vec<S>), E>
where
    S: Clone,
    FRenderExpr: FnMut(ExprId) -> String,
    FSolveBranch: FnMut(&Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FSolveDomain: FnMut(&Equation) -> Result<SolutionSet, E>,
    FMapStep: FnMut(DivisionDidacticExecutionItem) -> S,
    FFinalize:
        FnMut(DivisionDenominatorSignSplitSolvedCases<SolutionSet, SolutionSet>) -> SolutionSet,
{
    let solved = solve_division_denominator_sign_split_pipeline_with_optional_items(
        split_plan,
        denominator,
        case_boundary_lhs,
        case_boundary_op,
        simplified_rhs,
        include_items,
        branch_prefix_steps,
        render_expr,
        solve_branch,
        solve_domain,
        map_item_to_step,
    )?;

    let final_set = finalize_solved_sets(DivisionDenominatorSignSplitSolvedCases {
        positive_branch: solved.positive_set,
        negative_branch: solved.negative_set,
        positive_domain: solved.positive_domain_set,
        negative_domain: solved.negative_domain_set,
    });

    Ok((final_set, solved.steps))
}

/// Solve both branch equations for already-isolated denominator sign split.
pub fn solve_isolated_denominator_sign_split_cases_with<E, TBranch, FSolveBranch>(
    execution: &IsolatedDenominatorSignSplitExecutionPlan,
    mut solve_branch: FSolveBranch,
) -> Result<IsolatedDenominatorSignSplitSolvedCases<TBranch>, E>
where
    FSolveBranch: FnMut(&Equation) -> Result<TBranch, E>,
{
    Ok(IsolatedDenominatorSignSplitSolvedCases {
        positive_branch: solve_branch(&execution.positive_equation)?,
        negative_branch: solve_branch(&execution.negative_equation)?,
    })
}

/// Solve isolated-denominator sign-split branch equations while passing aligned
/// optional didactic execution items for each branch callback.
pub fn solve_isolated_denominator_sign_split_cases_with_items<E, TBranch, FSolveBranch>(
    execution: &IsolatedDenominatorSignSplitExecutionPlan,
    mut solve_branch: FSolveBranch,
) -> Result<IsolatedDenominatorSignSplitSolvedCases<TBranch>, E>
where
    FSolveBranch: FnMut(Option<DivisionDidacticExecutionItem>, &Equation) -> Result<TBranch, E>,
{
    let mut items = collect_isolated_denominator_sign_split_execution_items(execution).into_iter();
    Ok(IsolatedDenominatorSignSplitSolvedCases {
        positive_branch: solve_branch(items.next(), &execution.positive_equation)?,
        negative_branch: solve_branch(items.next(), &execution.negative_equation)?,
    })
}

/// Solved payload for isolated-denominator sign split execution:
/// solved branch sets plus concatenated branch steps.
#[derive(Debug, Clone, PartialEq)]
pub struct IsolatedDenominatorSignSplitExecutionSolved<TStep> {
    pub positive_set: SolutionSet,
    pub negative_set: SolutionSet,
    pub steps: Vec<TStep>,
}

/// Solve isolated-denominator sign split execution with aligned optional items,
/// keeping branch solution sets separate and concatenating branch steps with
/// the case boundary marker between them when present.
pub fn solve_isolated_denominator_sign_split_execution_with_items<
    E,
    TStep,
    FSolveBranch,
    FMapBoundary,
>(
    execution: &IsolatedDenominatorSignSplitExecutionPlan,
    solve_branch: FSolveBranch,
    mut map_boundary_item: FMapBoundary,
) -> Result<IsolatedDenominatorSignSplitExecutionSolved<TStep>, E>
where
    FSolveBranch: FnMut(
        Option<DivisionDidacticExecutionItem>,
        &Equation,
    ) -> Result<(SolutionSet, Vec<TStep>), E>,
    FMapBoundary: FnMut(DivisionDidacticExecutionItem) -> TStep,
{
    let solved = solve_isolated_denominator_sign_split_cases_with_items(execution, solve_branch)?;
    let IsolatedDenominatorSignSplitSolvedCases {
        positive_branch: (positive_set, mut positive_steps),
        negative_branch: (negative_set, negative_steps),
    } = solved;

    if let Some(item) = isolated_denominator_sign_split_boundary_item(execution) {
        positive_steps.push(map_boundary_item(item));
    }
    positive_steps.extend(negative_steps);

    Ok(IsolatedDenominatorSignSplitExecutionSolved {
        positive_set,
        negative_set,
        steps: positive_steps,
    })
}

/// Solve isolated-denominator sign split branches while prepending shared
/// branch steps and optionally mapping per-branch didactic items.
pub fn solve_isolated_denominator_sign_split_execution_pipeline_with_items<
    E,
    S,
    FSolveBranch,
    FStep,
>(
    execution: &IsolatedDenominatorSignSplitExecutionPlan,
    include_items: bool,
    branch_prefix_steps: &[S],
    mut solve_branch: FSolveBranch,
    mut map_item_to_step: FStep,
) -> Result<IsolatedDenominatorSignSplitExecutionSolved<S>, E>
where
    S: Clone,
    FSolveBranch: FnMut(&Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(DivisionDidacticExecutionItem) -> S,
{
    let mut items = if include_items {
        Some(collect_isolated_denominator_sign_split_execution_items(execution).into_iter())
    } else {
        None
    };

    let mut positive_steps = branch_prefix_steps.to_vec();
    if let Some(item) = take_optional_item(&mut items) {
        positive_steps.push(map_item_to_step(item));
    }
    let (positive_set, positive_sub_steps) = solve_branch(&execution.positive_equation)?;
    positive_steps.extend(positive_sub_steps);

    let mut negative_steps = branch_prefix_steps.to_vec();
    if let Some(item) = take_optional_item(&mut items) {
        negative_steps.push(map_item_to_step(item));
    }
    let (negative_set, negative_sub_steps) = solve_branch(&execution.negative_equation)?;
    negative_steps.extend(negative_sub_steps);

    if let Some(item) = take_optional_item(&mut items) {
        positive_steps.push(map_item_to_step(item));
    }
    positive_steps.extend(negative_steps);

    Ok(IsolatedDenominatorSignSplitExecutionSolved {
        positive_set,
        negative_set,
        steps: positive_steps,
    })
}

/// Solve isolated-denominator sign split with optional didactic items.
///
/// When `include_items` is enabled, this builds didactic execution items and
/// prepends them to per-branch solve steps. When disabled, branch equations are
/// solved directly without rendering didactic payload.
#[allow(clippy::too_many_arguments)]
pub fn solve_isolated_denominator_sign_split_pipeline_with_optional_items<
    E,
    S,
    FRenderExpr,
    FSolveBranch,
    FMapStep,
>(
    split_plan: IsolatedDenominatorSignSplitPlan,
    denominator: ExprId,
    case_boundary_op: RelOp,
    include_items: bool,
    branch_prefix_steps: &[S],
    render_expr: FRenderExpr,
    solve_branch: FSolveBranch,
    map_item_to_step: FMapStep,
) -> Result<IsolatedDenominatorSignSplitExecutionSolved<S>, E>
where
    S: Clone,
    FRenderExpr: FnMut(ExprId) -> String,
    FSolveBranch: FnMut(&Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(DivisionDidacticExecutionItem) -> S,
{
    let execution = if include_items {
        build_isolated_denominator_sign_split_execution_with(
            split_plan,
            denominator,
            case_boundary_op,
            render_expr,
        )
    } else {
        materialize_isolated_denominator_sign_split_execution(split_plan)
    };
    solve_isolated_denominator_sign_split_execution_pipeline_with_items(
        &execution,
        include_items,
        branch_prefix_steps,
        solve_branch,
        map_item_to_step,
    )
}

/// Return the boundary didactic item (`--- End of Case 1 ---`) when available.
pub fn isolated_denominator_sign_split_boundary_item(
    execution: &IsolatedDenominatorSignSplitExecutionPlan,
) -> Option<DivisionDidacticExecutionItem> {
    collect_isolated_denominator_sign_split_execution_items(execution)
        .into_iter()
        .nth(2)
}

/// Finalize solved already-isolated denominator sign split branch sets.
pub fn finalize_isolated_denominator_sign_split_solved_sets(
    ctx: &mut Context,
    solved: IsolatedDenominatorSignSplitSolvedCases<SolutionSet>,
) -> SolutionSet {
    crate::solution_set::finalize_isolated_denominator_sign_split_solution_set(
        ctx,
        solved.positive_branch,
        solved.negative_branch,
    )
}

/// Execute isolated-denominator sign-split pipeline and finalize branch sets.
///
/// This combines:
/// 1. Pipeline execution with optional didactic items.
/// 2. Conversion of solved branch payload into finalized `SolutionSet`.
#[allow(clippy::too_many_arguments)]
pub fn execute_isolated_denominator_sign_split_pipeline_with_optional_items<
    E,
    S,
    FRenderExpr,
    FSolveBranch,
    FMapStep,
    FFinalize,
>(
    split_plan: IsolatedDenominatorSignSplitPlan,
    denominator: ExprId,
    case_boundary_op: RelOp,
    include_items: bool,
    branch_prefix_steps: &[S],
    render_expr: FRenderExpr,
    solve_branch: FSolveBranch,
    map_item_to_step: FMapStep,
    mut finalize_solved_sets: FFinalize,
) -> Result<(SolutionSet, Vec<S>), E>
where
    S: Clone,
    FRenderExpr: FnMut(ExprId) -> String,
    FSolveBranch: FnMut(&Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(DivisionDidacticExecutionItem) -> S,
    FFinalize: FnMut(IsolatedDenominatorSignSplitSolvedCases<SolutionSet>) -> SolutionSet,
{
    let solved = solve_isolated_denominator_sign_split_pipeline_with_optional_items(
        split_plan,
        denominator,
        case_boundary_op,
        include_items,
        branch_prefix_steps,
        render_expr,
        solve_branch,
        map_item_to_step,
    )?;

    let final_set = finalize_solved_sets(IsolatedDenominatorSignSplitSolvedCases {
        positive_branch: solved.positive_set,
        negative_branch: solved.negative_set,
    });

    Ok((final_set, solved.steps))
}

/// Build executable split plan for division inequalities where denominator sign
/// determines whether inequality direction flips.
pub fn plan_division_denominator_sign_split(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
    rhs: ExprId,
    op: RelOp,
) -> Option<DivisionDenominatorSignSplitPlan> {
    let (branches, domain_pos, domain_neg) =
        crate::equation_rewrite::build_division_denominator_sign_split(
            ctx,
            numerator,
            denominator,
            rhs,
            op,
        )?;
    Some(DivisionDenominatorSignSplitPlan {
        positive_equation: branches.positive,
        negative_equation: branches.negative,
        positive_domain: domain_pos,
        negative_domain: domain_neg,
    })
}

/// Build division denominator-sign split plan only when split preconditions
/// hold (`numerator` and `denominator` contain target var and `op` is inequality).
pub fn plan_division_denominator_sign_split_if_applicable(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
) -> Option<DivisionDenominatorSignSplitPlan> {
    if crate::isolation_utils::should_split_division_denominator_sign_cases(
        ctx,
        numerator,
        denominator,
        &op,
        var,
    ) {
        plan_division_denominator_sign_split(ctx, numerator, denominator, rhs, op)
    } else {
        None
    }
}

/// Try denominator-sign split planning for division inequalities and execute
/// the full split pipeline with optional didactic items when applicable.
#[allow(clippy::too_many_arguments)]
pub fn try_execute_division_denominator_sign_split_pipeline_with_optional_items<
    E,
    S,
    FRenderExpr,
    FSolveBranch,
    FSolveDomain,
    FMapStep,
    FFinalize,
>(
    split_plan: Option<DivisionDenominatorSignSplitPlan>,
    denominator: ExprId,
    op: RelOp,
    case_boundary_lhs: ExprId,
    simplified_rhs: ExprId,
    include_items: bool,
    branch_prefix_steps: &[S],
    render_expr: FRenderExpr,
    solve_branch: FSolveBranch,
    solve_domain: FSolveDomain,
    map_item_to_step: FMapStep,
    finalize_solved_sets: FFinalize,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    S: Clone,
    FRenderExpr: FnMut(ExprId) -> String,
    FSolveBranch: FnMut(&Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FSolveDomain: FnMut(&Equation) -> Result<SolutionSet, E>,
    FMapStep: FnMut(DivisionDidacticExecutionItem) -> S,
    FFinalize:
        FnMut(DivisionDenominatorSignSplitSolvedCases<SolutionSet, SolutionSet>) -> SolutionSet,
{
    let split_plan = split_plan?;
    Some(
        execute_division_denominator_sign_split_pipeline_with_optional_items(
            split_plan,
            denominator,
            case_boundary_lhs,
            op,
            simplified_rhs,
            include_items,
            branch_prefix_steps,
            render_expr,
            solve_branch,
            solve_domain,
            map_item_to_step,
            finalize_solved_sets,
        ),
    )
}

/// Build executable split plan for already-isolated denominator inequalities.
pub fn plan_isolated_denominator_sign_split(
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
) -> Option<IsolatedDenominatorSignSplitPlan> {
    let branches = crate::equation_rewrite::build_isolated_denominator_sign_split(lhs, rhs, op)?;
    Some(IsolatedDenominatorSignSplitPlan {
        positive_equation: branches.positive,
        negative_equation: branches.negative,
    })
}

/// Build isolated-denominator sign split plan only when the denominator is the
/// target variable and operator is an inequality.
pub fn plan_isolated_denominator_sign_split_if_applicable(
    ctx: &Context,
    denominator: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
) -> Option<IsolatedDenominatorSignSplitPlan> {
    if crate::isolation_utils::should_split_isolated_denominator_variable(
        ctx,
        denominator,
        &op,
        var,
    ) {
        plan_isolated_denominator_sign_split(denominator, rhs, op)
    } else {
        None
    }
}

/// Try already-isolated denominator sign split planning and execute the full
/// split pipeline with optional didactic items when applicable.
#[allow(clippy::too_many_arguments)]
pub fn try_execute_isolated_denominator_sign_split_pipeline_with_optional_items<
    E,
    S,
    FRenderExpr,
    FSolveBranch,
    FMapStep,
    FFinalize,
>(
    split_plan: Option<IsolatedDenominatorSignSplitPlan>,
    denominator: ExprId,
    op: RelOp,
    include_items: bool,
    branch_prefix_steps: &[S],
    render_expr: FRenderExpr,
    solve_branch: FSolveBranch,
    map_item_to_step: FMapStep,
    finalize_solved_sets: FFinalize,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    S: Clone,
    FRenderExpr: FnMut(ExprId) -> String,
    FSolveBranch: FnMut(&Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(DivisionDidacticExecutionItem) -> S,
    FFinalize: FnMut(IsolatedDenominatorSignSplitSolvedCases<SolutionSet>) -> SolutionSet,
{
    let split_plan = split_plan?;
    Some(
        execute_isolated_denominator_sign_split_pipeline_with_optional_items(
            split_plan,
            denominator,
            op,
            include_items,
            branch_prefix_steps,
            render_expr,
            solve_branch,
            map_item_to_step,
            finalize_solved_sets,
        ),
    )
}

/// Build runtime execution plan for denominator-sign split using a precomputed
/// split plan and a shared simplified RHS for both branches.
pub fn build_division_denominator_sign_split_execution_with<F>(
    split_plan: DivisionDenominatorSignSplitPlan,
    denominator: ExprId,
    case_boundary_lhs: ExprId,
    case_boundary_op: RelOp,
    simplified_rhs: ExprId,
    mut render_expr: F,
) -> DivisionDenominatorSignSplitExecutionPlan
where
    F: FnMut(ExprId) -> String,
{
    let positive_equation = Equation {
        lhs: split_plan.positive_equation.lhs,
        rhs: simplified_rhs,
        op: split_plan.positive_equation.op.clone(),
    };
    let negative_equation = Equation {
        lhs: split_plan.negative_equation.lhs,
        rhs: simplified_rhs,
        op: split_plan.negative_equation.op.clone(),
    };
    let den_display = render_expr(denominator);
    let case_boundary_equation =
        build_case_boundary_equation(case_boundary_lhs, negative_equation.rhs, case_boundary_op);
    let items = vec![
        DivisionDidacticExecutionItem {
            equation: positive_equation.clone(),
            description: denominator_positive_case_message(&den_display),
        },
        DivisionDidacticExecutionItem {
            equation: negative_equation.clone(),
            description: denominator_negative_case_message(&den_display),
        },
        DivisionDidacticExecutionItem {
            equation: case_boundary_equation,
            description: end_case_message(1),
        },
    ];
    DivisionDenominatorSignSplitExecutionPlan {
        positive_equation,
        negative_equation,
        positive_domain: split_plan.positive_domain,
        negative_domain: split_plan.negative_domain,
        items,
    }
}

/// Materialize denominator-sign split equations without didactic payload.
///
/// This is useful when the caller needs executable branch equations/domains
/// but is not collecting user-facing steps.
pub fn materialize_division_denominator_sign_split_execution(
    split_plan: DivisionDenominatorSignSplitPlan,
    simplified_rhs: ExprId,
) -> DivisionDenominatorSignSplitExecutionPlan {
    let positive_equation = Equation {
        lhs: split_plan.positive_equation.lhs,
        rhs: simplified_rhs,
        op: split_plan.positive_equation.op,
    };
    let negative_equation = Equation {
        lhs: split_plan.negative_equation.lhs,
        rhs: simplified_rhs,
        op: split_plan.negative_equation.op,
    };
    DivisionDenominatorSignSplitExecutionPlan {
        positive_equation,
        negative_equation,
        positive_domain: split_plan.positive_domain,
        negative_domain: split_plan.negative_domain,
        items: vec![],
    }
}

/// Build runtime execution plan for isolated-denominator sign split.
pub fn build_isolated_denominator_sign_split_execution_with<F>(
    split_plan: IsolatedDenominatorSignSplitPlan,
    denominator: ExprId,
    case_boundary_op: RelOp,
    mut render_expr: F,
) -> IsolatedDenominatorSignSplitExecutionPlan
where
    F: FnMut(ExprId) -> String,
{
    let den_display = render_expr(denominator);
    let case_boundary_equation = build_case_boundary_equation(
        denominator,
        split_plan.negative_equation.rhs,
        case_boundary_op,
    );
    let items = vec![
        DivisionDidacticExecutionItem {
            equation: split_plan.positive_equation.clone(),
            description: isolated_denominator_positive_case_message(&den_display),
        },
        DivisionDidacticExecutionItem {
            equation: split_plan.negative_equation.clone(),
            description: isolated_denominator_negative_case_message(&den_display),
        },
        DivisionDidacticExecutionItem {
            equation: case_boundary_equation,
            description: end_case_message(1),
        },
    ];
    IsolatedDenominatorSignSplitExecutionPlan {
        positive_equation: split_plan.positive_equation,
        negative_equation: split_plan.negative_equation,
        items,
    }
}

/// Materialize isolated-denominator sign split equations without didactic payload.
///
/// This is useful when the caller needs executable branch equations
/// but is not collecting user-facing steps.
pub fn materialize_isolated_denominator_sign_split_execution(
    split_plan: IsolatedDenominatorSignSplitPlan,
) -> IsolatedDenominatorSignSplitExecutionPlan {
    IsolatedDenominatorSignSplitExecutionPlan {
        positive_equation: split_plan.positive_equation,
        negative_equation: split_plan.negative_equation,
        items: vec![],
    }
}

/// Build didactic two-step rewrite for denominator isolation:
/// 1. Multiply both sides by denominator.
/// 2. Divide both sides by previous rhs.
pub fn plan_division_denominator_didactic(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
    rhs: ExprId,
    isolated_rhs: ExprId,
    op: RelOp,
) -> DivisionDenominatorDidacticPlan {
    let multiplied_rhs = ctx.add(Expr::Mul(rhs, denominator));
    DivisionDenominatorDidacticPlan {
        multiply_equation: Equation {
            lhs: numerator,
            rhs: multiplied_rhs,
            op: op.clone(),
        },
        divide_equation: Equation {
            lhs: denominator,
            rhs: isolated_rhs,
            op,
        },
        multiply_by: denominator,
        divide_by: rhs,
    }
}

/// Plan the two denominator-isolation equations:
/// `num / den op rhs` -> `num op rhs*den` -> `den op num/rhs`.
pub fn plan_division_denominator(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
    rhs: ExprId,
    isolated_rhs: ExprId,
    op: RelOp,
) -> DivisionDenominatorDidacticPlan {
    plan_division_denominator_didactic(ctx, numerator, denominator, rhs, isolated_rhs, op)
}

/// Build didactic payload for denominator-isolation as two explicit steps:
/// 1. Multiply by denominator.
/// 2. Divide by previous RHS.
pub fn build_division_denominator_didactic_steps_with<F>(
    multiply_equation: Equation,
    divide_equation: Equation,
    multiply_by: ExprId,
    divide_by: ExprId,
    mut render_expr: F,
) -> DivisionDenominatorDidacticSteps
where
    F: FnMut(ExprId) -> String,
{
    let multiply_by_desc = render_expr(multiply_by);
    let divide_by_desc = render_expr(divide_by);
    let items = vec![
        DivisionDidacticExecutionItem {
            equation: multiply_equation,
            description: multiply_both_sides_message(&multiply_by_desc),
        },
        DivisionDidacticExecutionItem {
            equation: divide_equation,
            description: divide_both_sides_message(&divide_by_desc),
        },
    ];
    DivisionDenominatorDidacticSteps { items }
}

/// Build runtime execution plan for denominator-isolation didactic steps,
/// replacing the multiply equation RHS with a caller-provided simplified value.
pub fn build_division_denominator_didactic_execution_with<F>(
    didactic_plan: DivisionDenominatorDidacticPlan,
    simplified_multiply_rhs: ExprId,
    mut render_expr: F,
) -> DivisionDenominatorDidacticExecutionPlan
where
    F: FnMut(ExprId) -> String,
{
    let multiply_equation = Equation {
        lhs: didactic_plan.multiply_equation.lhs,
        rhs: simplified_multiply_rhs,
        op: didactic_plan.multiply_equation.op.clone(),
    };
    let divide_equation = didactic_plan.divide_equation;
    let multiply_by_desc = render_expr(didactic_plan.multiply_by);
    let divide_by_desc = render_expr(didactic_plan.divide_by);
    let items = vec![
        DivisionDidacticExecutionItem {
            equation: multiply_equation.clone(),
            description: multiply_both_sides_message(&multiply_by_desc),
        },
        DivisionDidacticExecutionItem {
            equation: divide_equation.clone(),
            description: divide_both_sides_message(&divide_by_desc),
        },
    ];
    DivisionDenominatorDidacticExecutionPlan {
        multiply_equation,
        divide_equation,
        items,
    }
}

/// Build executable denominator-isolation steps, preserving didactic payload.
pub fn build_division_denominator_execution_with<F>(
    didactic_plan: DivisionDenominatorDidacticPlan,
    simplified_multiply_rhs: ExprId,
    render_expr: F,
) -> DivisionDenominatorDidacticExecutionPlan
where
    F: FnMut(ExprId) -> String,
{
    build_division_denominator_didactic_execution_with(
        didactic_plan,
        simplified_multiply_rhs,
        render_expr,
    )
}

/// Build equation payload used when stitching branch traces with a case marker.
pub fn build_case_boundary_equation(lhs: ExprId, rhs: ExprId, op: RelOp) -> Equation {
    Equation { lhs, rhs, op }
}

/// Build an executable isolation plan for absolute-value equations.
pub fn plan_abs_isolation(
    ctx: &mut Context,
    arg: ExprId,
    rhs: ExprId,
    op: RelOp,
    rhs_sign: Option<NumericSign>,
) -> AbsIsolationPlan {
    match classify_abs_isolation_fast_path(op.clone(), rhs_sign) {
        AbsIsolationFastPath::ReturnEmptySet => AbsIsolationPlan::ReturnEmptySet,
        AbsIsolationFastPath::CollapseToZero => AbsIsolationPlan::IsolateSingleEquation {
            equation: Equation { lhs: arg, rhs, op },
        },
        AbsIsolationFastPath::Continue => {
            let (positive, negative) =
                crate::equation_rewrite::isolate_abs_branches(ctx, arg, rhs, op);
            AbsIsolationPlan::SplitBranches { positive, negative }
        }
    }
}

/// Build absolute-value isolation plan using numeric-sign classification derived
/// directly from the RHS expression.
pub fn plan_abs_isolation_with_rhs_sign(
    ctx: &mut Context,
    arg: ExprId,
    rhs: ExprId,
    op: RelOp,
) -> AbsIsolationPlan {
    let rhs_sign = crate::isolation_utils::numeric_sign(ctx, rhs);
    plan_abs_isolation(ctx, arg, rhs, op, rhs_sign)
}

/// Build didactic narration for each absolute-value split branch.
pub fn abs_split_case_message(
    case: AbsSplitCase,
    lhs_display: &str,
    op_display: &str,
    rhs_display: &str,
) -> String {
    let case_label = match case {
        AbsSplitCase::Positive => "Case 1",
        AbsSplitCase::Negative => "Case 2",
    };
    format!(
        "Split absolute value ({}): {} {} {}",
        case_label, lhs_display, op_display, rhs_display
    )
}

/// Build didactic payload for one absolute-value split branch.
pub fn build_abs_split_step_with<F>(
    case: AbsSplitCase,
    equation_after: Equation,
    lhs_expr: ExprId,
    rhs_expr: ExprId,
    op: RelOp,
    mut render_expr: F,
) -> AbsSplitDidacticStep
where
    F: FnMut(ExprId) -> String,
{
    let lhs_display = render_expr(lhs_expr);
    let rhs_display = render_expr(rhs_expr);
    let description = abs_split_case_message(case, &lhs_display, &op.to_string(), &rhs_display);
    AbsSplitDidacticStep {
        description,
        equation_after,
    }
}

/// Didactic payload for both branches produced by absolute-value split.
#[derive(Debug, Clone, PartialEq)]
pub struct AbsSplitDidacticPair {
    pub positive: AbsSplitDidacticStep,
    pub negative: AbsSplitDidacticStep,
}

/// Collect absolute-value split didactic steps in execution order:
/// positive branch first, negative branch second.
pub fn collect_abs_split_didactic_steps(
    didactic: &AbsSplitDidacticPair,
) -> Vec<AbsSplitDidacticStep> {
    vec![didactic.positive.clone(), didactic.negative.clone()]
}

/// One executable absolute-split item aligned with didactic payload.
#[derive(Debug, Clone, PartialEq)]
pub struct AbsSplitExecutionItem {
    pub equation: Equation,
    pub description: String,
}

impl AbsSplitExecutionItem {
    /// User-facing narration for this execution item.
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Collect absolute-value split execution items in execution order:
/// positive branch first, negative branch second.
pub fn collect_abs_split_execution_items(
    execution: &AbsSplitExecutionPlan,
) -> Vec<AbsSplitExecutionItem> {
    execution.items.clone()
}

/// Build didactic payload for both absolute-value split branches.
pub fn build_abs_split_steps_with<F>(
    positive_equation: Equation,
    negative_equation: Equation,
    lhs_expr: ExprId,
    mut render_expr: F,
) -> AbsSplitDidacticPair
where
    F: FnMut(ExprId) -> String,
{
    let lhs_display = render_expr(lhs_expr);
    let positive_rhs_display = render_expr(positive_equation.rhs);
    let negative_rhs_display = render_expr(negative_equation.rhs);
    AbsSplitDidacticPair {
        positive: AbsSplitDidacticStep {
            description: abs_split_case_message(
                AbsSplitCase::Positive,
                &lhs_display,
                &positive_equation.op.to_string(),
                &positive_rhs_display,
            ),
            equation_after: positive_equation,
        },
        negative: AbsSplitDidacticStep {
            description: abs_split_case_message(
                AbsSplitCase::Negative,
                &lhs_display,
                &negative_equation.op.to_string(),
                &negative_rhs_display,
            ),
            equation_after: negative_equation,
        },
    }
}

/// Build runtime execution plan for absolute-value branch splitting.
pub fn build_abs_split_execution_with<F>(
    positive_equation: Equation,
    negative_equation: Equation,
    lhs_expr: ExprId,
    mut render_expr: F,
) -> AbsSplitExecutionPlan
where
    F: FnMut(ExprId) -> String,
{
    let lhs_display = render_expr(lhs_expr);
    let positive_rhs_display = render_expr(positive_equation.rhs);
    let negative_rhs_display = render_expr(negative_equation.rhs);
    let items = vec![
        AbsSplitExecutionItem {
            equation: positive_equation.clone(),
            description: abs_split_case_message(
                AbsSplitCase::Positive,
                &lhs_display,
                &positive_equation.op.to_string(),
                &positive_rhs_display,
            ),
        },
        AbsSplitExecutionItem {
            equation: negative_equation.clone(),
            description: abs_split_case_message(
                AbsSplitCase::Negative,
                &lhs_display,
                &negative_equation.op.to_string(),
                &negative_rhs_display,
            ),
        },
    ];
    AbsSplitExecutionPlan {
        positive_equation,
        negative_equation,
        items,
    }
}

/// Materialize absolute-value split equations without didactic payload.
///
/// This is useful when the caller needs executable branch equations
/// but is not collecting user-facing steps.
pub fn materialize_abs_split_execution(
    positive_equation: Equation,
    negative_equation: Equation,
) -> AbsSplitExecutionPlan {
    AbsSplitExecutionPlan {
        positive_equation,
        negative_equation,
        items: vec![],
    }
}

/// Solved payload for absolute-value branch split:
/// `|A| op rhs` under positive and negative branch equations.
#[derive(Debug, Clone, PartialEq)]
pub struct AbsSplitSolvedCases<TBranch> {
    pub positive_branch: TBranch,
    pub negative_branch: TBranch,
}

/// Solve both branch equations of an absolute-value split execution plan with
/// a caller-provided branch solver callback.
pub fn solve_abs_split_cases_with<E, TBranch, FSolveBranch>(
    execution: &AbsSplitExecutionPlan,
    mut solve_branch: FSolveBranch,
) -> Result<AbsSplitSolvedCases<TBranch>, E>
where
    FSolveBranch: FnMut(&Equation) -> Result<TBranch, E>,
{
    Ok(AbsSplitSolvedCases {
        positive_branch: solve_branch(&execution.positive_equation)?,
        negative_branch: solve_branch(&execution.negative_equation)?,
    })
}

/// Solve both branch equations of an absolute-value split execution plan, while
/// providing aligned optional didactic items for each branch callback.
pub fn solve_abs_split_cases_with_items<E, TBranch, FSolveBranch>(
    execution: &AbsSplitExecutionPlan,
    mut solve_branch: FSolveBranch,
) -> Result<AbsSplitSolvedCases<TBranch>, E>
where
    FSolveBranch: FnMut(Option<AbsSplitExecutionItem>, &Equation) -> Result<TBranch, E>,
{
    let mut items = collect_abs_split_execution_items(execution).into_iter();
    Ok(AbsSplitSolvedCases {
        positive_branch: solve_branch(items.next(), &execution.positive_equation)?,
        negative_branch: solve_branch(items.next(), &execution.negative_equation)?,
    })
}

/// Solved payload for full absolute-value split execution:
/// branch solution sets plus concatenated branch steps.
#[derive(Debug, Clone, PartialEq)]
pub struct AbsSplitExecutionSolved<TStep> {
    pub positive_set: SolutionSet,
    pub negative_set: SolutionSet,
    pub steps: Vec<TStep>,
}

/// Solve an absolute-value split execution with branch callbacks and return
/// both branch solution sets plus concatenated branch steps
/// (positive branch first, then negative branch).
pub fn solve_abs_split_execution_with_items<E, TStep, FSolveBranch>(
    execution: &AbsSplitExecutionPlan,
    solve_branch: FSolveBranch,
) -> Result<AbsSplitExecutionSolved<TStep>, E>
where
    FSolveBranch:
        FnMut(Option<AbsSplitExecutionItem>, &Equation) -> Result<(SolutionSet, Vec<TStep>), E>,
{
    let solved = solve_abs_split_cases_with_items(execution, solve_branch)?;
    let AbsSplitSolvedCases {
        positive_branch: (positive_set, mut positive_steps),
        negative_branch: (negative_set, negative_steps),
    } = solved;

    positive_steps.extend(negative_steps);
    Ok(AbsSplitExecutionSolved {
        positive_set,
        negative_set,
        steps: positive_steps,
    })
}

/// Solve absolute-value split branches while prepending shared branch steps and
/// optionally mapping per-branch didactic items.
pub fn solve_abs_split_execution_pipeline_with_items<E, S, FSolveBranch, FStep>(
    execution: &AbsSplitExecutionPlan,
    include_items: bool,
    branch_prefix_steps: &[S],
    mut solve_branch: FSolveBranch,
    mut map_item_to_step: FStep,
) -> Result<AbsSplitExecutionSolved<S>, E>
where
    S: Clone,
    FSolveBranch: FnMut(&Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(AbsSplitExecutionItem) -> S,
{
    let mut items = if include_items {
        Some(collect_abs_split_execution_items(execution).into_iter())
    } else {
        None
    };

    let mut positive_steps = branch_prefix_steps.to_vec();
    if let Some(item) = take_optional_item(&mut items) {
        positive_steps.push(map_item_to_step(item));
    }
    let (positive_set, positive_sub_steps) = solve_branch(&execution.positive_equation)?;
    positive_steps.extend(positive_sub_steps);

    let mut negative_steps = branch_prefix_steps.to_vec();
    if let Some(item) = take_optional_item(&mut items) {
        negative_steps.push(map_item_to_step(item));
    }
    let (negative_set, negative_sub_steps) = solve_branch(&execution.negative_equation)?;
    negative_steps.extend(negative_sub_steps);

    positive_steps.extend(negative_steps);
    Ok(AbsSplitExecutionSolved {
        positive_set,
        negative_set,
        steps: positive_steps,
    })
}

/// Solve absolute-value split with optional didactic items.
///
/// When `include_items` is enabled, this builds didactic execution items from
/// `lhs_expr` and prepends them to per-branch solve steps.
/// When disabled, branch equations are solved directly without rendering items.
#[allow(clippy::too_many_arguments)]
pub fn solve_abs_split_pipeline_with_optional_items<E, S, FRenderExpr, FSolveBranch, FMapStep>(
    positive_equation: Equation,
    negative_equation: Equation,
    lhs_expr: ExprId,
    include_items: bool,
    branch_prefix_steps: &[S],
    render_expr: FRenderExpr,
    solve_branch: FSolveBranch,
    map_item_to_step: FMapStep,
) -> Result<AbsSplitExecutionSolved<S>, E>
where
    S: Clone,
    FRenderExpr: FnMut(ExprId) -> String,
    FSolveBranch: FnMut(&Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(AbsSplitExecutionItem) -> S,
{
    let execution = if include_items {
        build_abs_split_execution_with(positive_equation, negative_equation, lhs_expr, render_expr)
    } else {
        materialize_abs_split_execution(positive_equation, negative_equation)
    };
    solve_abs_split_execution_pipeline_with_items(
        &execution,
        include_items,
        branch_prefix_steps,
        solve_branch,
        map_item_to_step,
    )
}

/// Execute absolute-value split pipeline and finalize branch solution sets.
///
/// This combines:
/// 1. Pipeline execution with optional didactic items.
/// 2. Conversion of solved branch payload into finalized `SolutionSet`.
#[allow(clippy::too_many_arguments)]
pub fn execute_abs_split_pipeline_with_optional_items<
    E,
    S,
    FRenderExpr,
    FSolveBranch,
    FMapStep,
    FFinalize,
>(
    positive_equation: Equation,
    negative_equation: Equation,
    lhs_expr: ExprId,
    include_items: bool,
    branch_prefix_steps: &[S],
    render_expr: FRenderExpr,
    solve_branch: FSolveBranch,
    map_item_to_step: FMapStep,
    mut finalize_solved_sets: FFinalize,
) -> Result<(SolutionSet, Vec<S>), E>
where
    S: Clone,
    FRenderExpr: FnMut(ExprId) -> String,
    FSolveBranch: FnMut(&Equation) -> Result<(SolutionSet, Vec<S>), E>,
    FMapStep: FnMut(AbsSplitExecutionItem) -> S,
    FFinalize: FnMut(SolutionSet, SolutionSet) -> SolutionSet,
{
    let solved = solve_abs_split_pipeline_with_optional_items(
        positive_equation,
        negative_equation,
        lhs_expr,
        include_items,
        branch_prefix_steps,
        render_expr,
        solve_branch,
        map_item_to_step,
    )?;
    let final_set = finalize_solved_sets(solved.positive_set, solved.negative_set);
    Ok((final_set, solved.steps))
}

/// For `|A| = rhs`, attach the soundness guard `rhs >= 0` when
/// `rhs` depends on the solve variable.
pub fn guard_abs_solution_with_nonnegative_rhs(
    rhs_contains_var: bool,
    rhs: ExprId,
    combined: SolutionSet,
) -> SolutionSet {
    if rhs_contains_var {
        let guard = ConditionSet::single(ConditionPredicate::NonNegative(rhs));
        SolutionSet::Conditional(vec![Case::new(guard, combined)])
    } else {
        combined
    }
}

/// Finalize absolute-value split branches:
/// 1) combine branch solution sets according to operator semantics
/// 2) attach `rhs >= 0` guard when rhs depends on solve variable
pub fn finalize_abs_split_solution_set(
    ctx: &Context,
    op: RelOp,
    rhs_contains_var: bool,
    rhs: ExprId,
    positive_branch: SolutionSet,
    negative_branch: SolutionSet,
) -> SolutionSet {
    let combined =
        crate::isolation_utils::combine_abs_branch_sets(ctx, op, positive_branch, negative_branch);
    guard_abs_solution_with_nonnegative_rhs(rhs_contains_var, rhs, combined)
}

/// Finalize absolute-value split branches and infer `rhs_contains_var` from the
/// RHS expression and solve variable.
pub fn finalize_abs_split_solution_set_for_rhs(
    ctx: &Context,
    op: RelOp,
    rhs: ExprId,
    var: &str,
    positive_branch: SolutionSet,
    negative_branch: SolutionSet,
) -> SolutionSet {
    let rhs_contains_var = crate::isolation_utils::contains_var(ctx, rhs, var);
    finalize_abs_split_solution_set(
        ctx,
        op,
        rhs_contains_var,
        rhs,
        positive_branch,
        negative_branch,
    )
}

/// Build `Conditional([guard -> guarded_solutions, else -> Residual(original_eq)])`.
pub fn guarded_solutions_with_residual_fallback(
    guard: ConditionSet,
    guarded_solutions: SolutionSet,
    residual_expr: ExprId,
) -> SolutionSet {
    SolutionSet::Conditional(vec![
        Case::new(guard, guarded_solutions),
        Case::new(ConditionSet::empty(), SolutionSet::Residual(residual_expr)),
    ])
}

/// Build guarded conditional solutions when both pieces are available;
/// otherwise fall back to residual.
pub fn guarded_or_residual(
    guard: Option<ConditionSet>,
    guarded_solutions: Option<SolutionSet>,
    residual_expr: ExprId,
) -> SolutionSet {
    match (guard, guarded_solutions) {
        (Some(guard), Some(guarded_solutions)) => {
            guarded_solutions_with_residual_fallback(guard, guarded_solutions, residual_expr)
        }
        _ => SolutionSet::Residual(residual_expr),
    }
}

/// Outcome for symbolic `a^x = a` (with `a` symbolic).
///
/// Returns:
/// - `a = 1`  -> `AllReals`
/// - `a = 0`  -> `x in (0, +inf)`
/// - otherwise -> `x = 1`
pub fn power_equals_base_symbolic_outcome(ctx: &mut Context, base: ExprId) -> SolutionSet {
    let one = ctx.num(1);

    let case_one_guard = ConditionSet::single(ConditionPredicate::EqOne(base));
    let case_one = Case::with_result(case_one_guard, SolveResult::solved(SolutionSet::AllReals));

    let case_zero_guard = ConditionSet::single(ConditionPredicate::EqZero(base));
    let case_zero = Case::with_result(
        case_zero_guard,
        SolveResult::solved(open_positive_domain(ctx)),
    );

    let case_default_guard = ConditionSet::empty();
    let case_default = Case::with_result(
        case_default_guard,
        SolveResult::solved(SolutionSet::Discrete(vec![one])),
    );

    SolutionSet::Conditional(vec![case_one, case_zero, case_default])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_zero_number() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        assert_eq!(
            classify_var_free_difference(&ctx, zero),
            VarFreeDiffKind::IdentityZero
        );
    }

    #[test]
    fn classify_nonzero_number() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        assert_eq!(
            classify_var_free_difference(&ctx, two),
            VarFreeDiffKind::ContradictionNonZero
        );
    }

    #[test]
    fn classify_symbolic_constraint() {
        let mut ctx = Context::new();
        let y = ctx.var("y");
        assert_eq!(
            classify_var_free_difference(&ctx, y),
            VarFreeDiffKind::Constraint
        );
    }

    #[test]
    fn resolve_var_eliminated_identity_returns_all_reals() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let out =
            resolve_var_eliminated_outcome_with(&mut ctx, zero, "x", |_, _| "unused".to_string());
        assert_eq!(out, VarEliminatedSolveOutcome::IdentityAllReals);
    }

    #[test]
    fn resolve_var_eliminated_contradiction_returns_empty() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let out =
            resolve_var_eliminated_outcome_with(&mut ctx, one, "x", |_, _| "unused".to_string());
        assert_eq!(out, VarEliminatedSolveOutcome::ContradictionEmpty);
    }

    #[test]
    fn resolve_var_eliminated_constraint_builds_step_payload() {
        let mut ctx = Context::new();
        let y = ctx.var("y");
        let out = resolve_var_eliminated_outcome_with(&mut ctx, y, "x", |_, _| "y".to_string());
        match out {
            VarEliminatedSolveOutcome::ConstraintAllReals {
                description,
                equation_after,
            } => {
                assert!(description.contains("depends on constraint"));
                assert_eq!(equation_after.lhs, y);
                assert_eq!(equation_after.op, RelOp::Eq);
            }
            other => panic!("expected constraint outcome, got {:?}", other),
        }
    }

    #[test]
    fn solve_var_eliminated_constraint_pipeline_with_item_maps_step_when_enabled() {
        let mut ctx = Context::new();
        let y = ctx.var("y");
        let eq = Equation {
            lhs: y,
            rhs: ctx.num(0),
            op: RelOp::Eq,
        };

        let steps = solve_var_eliminated_constraint_pipeline_with_item(
            "constraint".to_string(),
            eq.clone(),
            true,
            |description, equation_after| (description, equation_after),
        );
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].0, "constraint");
        assert_eq!(steps[0].1, eq);
    }

    #[test]
    fn solve_var_eliminated_constraint_pipeline_with_item_omits_step_when_disabled() {
        let mut ctx = Context::new();
        let y = ctx.var("y");
        let eq = Equation {
            lhs: y,
            rhs: ctx.num(0),
            op: RelOp::Eq,
        };

        let steps = solve_var_eliminated_constraint_pipeline_with_item(
            "constraint".to_string(),
            eq,
            false,
            |_, _| 1u8,
        );
        assert!(steps.is_empty());
    }

    #[test]
    fn solve_var_eliminated_outcome_pipeline_with_returns_identity_for_zero_diff() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let out = solve_var_eliminated_outcome_pipeline_with(
            &mut ctx,
            zero,
            "x",
            true,
            |_, expr| expr.to_string(),
            |description, _| description,
        );
        assert_eq!(out, VarEliminatedOutcomePipelineSolved::IdentityAllReals);
    }

    #[test]
    fn solve_var_eliminated_outcome_pipeline_with_returns_contradiction_for_nonzero_diff() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let out = solve_var_eliminated_outcome_pipeline_with(
            &mut ctx,
            one,
            "x",
            true,
            |_, expr| expr.to_string(),
            |description, _| description,
        );
        assert_eq!(out, VarEliminatedOutcomePipelineSolved::ContradictionEmpty);
    }

    #[test]
    fn solve_var_eliminated_outcome_pipeline_with_maps_constraint_step_when_enabled() {
        let mut ctx = Context::new();
        let y = ctx.var("y");
        let out = solve_var_eliminated_outcome_pipeline_with(
            &mut ctx,
            y,
            "x",
            true,
            |_, expr| expr.to_string(),
            |description, _| description,
        );
        match out {
            VarEliminatedOutcomePipelineSolved::ConstraintAllReals { steps } => {
                assert_eq!(steps.len(), 1);
                assert!(steps[0].contains("depends on constraint"));
            }
            other => panic!("expected constraint outcome, got {:?}", other),
        }
    }

    #[test]
    fn even_power_negative_rhs_outcome_for_eq_is_empty() {
        assert!(matches!(
            even_power_negative_rhs_outcome(RelOp::Eq),
            SolutionSet::Empty
        ));
    }

    #[test]
    fn even_power_negative_rhs_outcome_for_neq_is_all_reals() {
        assert!(matches!(
            even_power_negative_rhs_outcome(RelOp::Neq),
            SolutionSet::AllReals
        ));
    }

    #[test]
    fn power_equals_base_symbolic_outcome_has_three_cases() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let out = power_equals_base_symbolic_outcome(&mut ctx, a);
        match out {
            SolutionSet::Conditional(cases) => assert_eq!(cases.len(), 3),
            other => panic!("Expected conditional set, got {:?}", other),
        }
    }

    #[test]
    fn power_base_one_outcome_all_reals_when_rhs_is_one() {
        assert!(matches!(
            power_base_one_outcome(true),
            SolutionSet::AllReals
        ));
    }

    #[test]
    fn power_base_one_outcome_empty_when_rhs_not_one() {
        assert!(matches!(power_base_one_outcome(false), SolutionSet::Empty));
    }

    #[test]
    fn classify_power_base_one_shortcut_not_applicable_when_base_not_one() {
        assert_eq!(
            classify_power_base_one_shortcut(false, true),
            PowerBaseOneShortcut::NotApplicable
        );
    }

    #[test]
    fn classify_power_base_one_shortcut_all_reals_when_rhs_one() {
        assert_eq!(
            classify_power_base_one_shortcut(true, true),
            PowerBaseOneShortcut::AllReals
        );
    }

    #[test]
    fn classify_power_base_one_shortcut_empty_when_rhs_not_one() {
        assert_eq!(
            classify_power_base_one_shortcut(true, false),
            PowerBaseOneShortcut::Empty
        );
    }

    #[test]
    fn power_base_one_shortcut_solutions_maps_correctly() {
        assert!(matches!(
            power_base_one_shortcut_solutions(PowerBaseOneShortcut::AllReals),
            Some(SolutionSet::AllReals)
        ));
        assert!(matches!(
            power_base_one_shortcut_solutions(PowerBaseOneShortcut::Empty),
            Some(SolutionSet::Empty)
        ));
        assert!(power_base_one_shortcut_solutions(PowerBaseOneShortcut::NotApplicable).is_none());
    }

    #[test]
    fn power_base_one_shortcut_message_maps_correctly() {
        assert_eq!(
            power_base_one_shortcut_message(PowerBaseOneShortcut::AllReals, "rhs"),
            Some("1^x = 1 for all x -> any real number is a solution".to_string())
        );
        assert_eq!(
            power_base_one_shortcut_message(PowerBaseOneShortcut::Empty, "2"),
            Some("1^x = 1 for all x, but RHS = 2 != 1 -> no solution".to_string())
        );
        assert!(
            power_base_one_shortcut_message(PowerBaseOneShortcut::NotApplicable, "rhs").is_none()
        );
    }

    #[test]
    fn resolve_power_base_one_shortcut_with_returns_none_when_not_applicable() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let rhs = ctx.var("rhs");
        assert!(
            resolve_power_base_one_shortcut_with(false, true, x, rhs, RelOp::Eq, |_| {
                "rhs".to_string()
            })
            .is_none()
        );
    }

    #[test]
    fn resolve_power_base_one_shortcut_with_builds_step_and_solution() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let rhs = ctx.num(2);
        let outcome = resolve_power_base_one_shortcut_with(true, false, x, rhs, RelOp::Eq, |_| {
            "2".to_string()
        })
        .expect("shortcut should apply");
        assert!(matches!(outcome.solutions, SolutionSet::Empty));
        assert_eq!(outcome.items[0].equation.lhs, x);
        assert_eq!(outcome.items[0].equation.rhs, rhs);
        assert!(outcome.items[0].description.contains("no solution"));
    }

    #[test]
    fn resolve_power_base_one_shortcut_for_pow_with_detects_base_and_rhs() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let x = ctx.var("x");
        let lhs = ctx.add(Expr::Pow(one, x));
        let rhs = ctx.num(2);

        let outcome =
            resolve_power_base_one_shortcut_for_pow_with(&ctx, one, lhs, rhs, RelOp::Eq, |_, _| {
                "2".to_string()
            })
            .expect("shortcut should apply");
        assert!(matches!(outcome.solutions, SolutionSet::Empty));
        assert_eq!(outcome.items.len(), 1);
        assert_eq!(outcome.items[0].equation.lhs, lhs);
        assert_eq!(outcome.items[0].equation.rhs, rhs);
    }

    #[test]
    fn collect_power_base_one_shortcut_didactic_steps_returns_single_step() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let rhs = ctx.num(2);
        let outcome = resolve_power_base_one_shortcut_with(true, false, x, rhs, RelOp::Eq, |_| {
            "2".to_string()
        })
        .expect("shortcut should apply");

        let didactic = collect_power_base_one_shortcut_didactic_steps(&outcome);
        assert_eq!(didactic.len(), 1);
        assert_eq!(didactic[0].description, outcome.items[0].description);
        assert_eq!(didactic[0].equation_after, outcome.items[0].equation);
    }

    #[test]
    fn collect_power_base_one_shortcut_execution_items_returns_single_item() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let rhs = ctx.num(2);
        let outcome = resolve_power_base_one_shortcut_with(true, false, x, rhs, RelOp::Eq, |_| {
            "2".to_string()
        })
        .expect("shortcut should apply");

        let items = collect_power_base_one_shortcut_execution_items(&outcome);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].equation, outcome.items[0].equation);
        assert_eq!(items[0].description, outcome.items[0].description);
    }

    #[test]
    fn first_power_base_one_shortcut_execution_item_returns_single_item() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let rhs = ctx.num(2);
        let outcome = resolve_power_base_one_shortcut_with(true, false, x, rhs, RelOp::Eq, |_| {
            "2".to_string()
        })
        .expect("shortcut should apply");

        let item = first_power_base_one_shortcut_execution_item(&outcome)
            .expect("expected one shortcut item");
        assert_eq!(item.equation, outcome.items[0].equation);
        assert_eq!(item.description, outcome.items[0].description);
    }

    #[test]
    fn solve_power_base_one_shortcut_pipeline_with_item_emits_step_when_enabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let rhs = ctx.num(2);
        let outcome = resolve_power_base_one_shortcut_with(true, false, x, rhs, RelOp::Eq, |_| {
            "2".to_string()
        })
        .expect("shortcut should apply");

        let solved = solve_power_base_one_shortcut_pipeline_with_item(outcome, true, |item| {
            item.description
        });

        assert!(matches!(solved.solution_set, SolutionSet::Empty));
        assert_eq!(solved.steps.len(), 1);
        assert!(solved.steps[0].contains("no solution"));
    }

    #[test]
    fn solve_power_base_one_shortcut_pipeline_with_item_omits_step_when_disabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let rhs = ctx.num(1);
        let outcome = resolve_power_base_one_shortcut_with(true, true, x, rhs, RelOp::Eq, |_| {
            "1".to_string()
        })
        .expect("shortcut should apply");

        let solved = solve_power_base_one_shortcut_pipeline_with_item(outcome, false, |item| {
            item.description
        });

        assert!(matches!(solved.solution_set, SolutionSet::AllReals));
        assert!(solved.steps.is_empty());
    }

    #[test]
    fn execute_power_base_one_shortcut_pipeline_with_item_for_pow_with_runs_end_to_end() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let x = ctx.var("x");
        let lhs = ctx.add(Expr::Pow(one, x));
        let rhs = ctx.num(2);

        let solved = execute_power_base_one_shortcut_pipeline_with_item_for_pow_with(
            &ctx,
            one,
            lhs,
            rhs,
            RelOp::Eq,
            true,
            |_, _| "2".to_string(),
            |item| item.description,
        )
        .expect("shortcut should apply");

        assert!(matches!(solved.solution_set, SolutionSet::Empty));
        assert_eq!(solved.steps.len(), 1);
        assert!(solved.steps[0].contains("no solution"));
    }

    #[test]
    fn execute_power_base_one_shortcut_pipeline_with_item_for_pow_and_merge_with_existing_steps_with_appends_existing(
    ) {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let x = ctx.var("x");
        let lhs = ctx.add(Expr::Pow(one, x));
        let rhs = ctx.num(2);

        let solved =
            execute_power_base_one_shortcut_pipeline_with_item_for_pow_and_merge_with_existing_steps_with(
                &ctx,
                one,
                lhs,
                rhs,
                RelOp::Eq,
                true,
                vec!["existing".to_string()],
                |_, _| "2".to_string(),
                |item| item.description,
            )
            .expect("shortcut should apply");

        assert!(matches!(solved.0, SolutionSet::Empty));
        assert_eq!(solved.1.len(), 2);
        assert_eq!(solved.1[0], "existing".to_string());
        assert!(solved.1[1].contains("no solution"));
    }

    #[test]
    fn execute_power_base_one_shortcut_pipeline_with_item_for_pow_and_finalize_with_existing_steps_with_appends_existing(
    ) {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let x = ctx.var("x");
        let lhs = ctx.add(Expr::Pow(one, x));
        let rhs = ctx.num(2);
        let mut existing = vec!["existing".to_string()];

        let solved =
            execute_power_base_one_shortcut_pipeline_with_item_for_pow_and_finalize_with_existing_steps_with(
                &ctx,
                one,
                lhs,
                rhs,
                RelOp::Eq,
                true,
                &mut existing,
                |_, _| "2".to_string(),
                |item| item.description,
            )
            .expect("shortcut should apply");

        assert!(matches!(solved.0, SolutionSet::Empty));
        assert_eq!(solved.1.len(), 2);
        assert_eq!(solved.1[0], "existing".to_string());
        assert!(solved.1[1].contains("no solution"));
        assert!(existing.is_empty());
    }

    #[test]
    fn execute_power_base_one_shortcut_pipeline_with_item_for_pow_and_finalize_with_existing_steps_with_preserves_existing_when_not_applicable(
    ) {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let x = ctx.var("x");
        let lhs = ctx.add(Expr::Pow(two, x));
        let rhs = ctx.num(2);
        let mut existing = vec!["existing".to_string()];

        let solved =
            execute_power_base_one_shortcut_pipeline_with_item_for_pow_and_finalize_with_existing_steps_with(
                &ctx,
                two,
                lhs,
                rhs,
                RelOp::Eq,
                true,
                &mut existing,
                |_, _| "2".to_string(),
                |item| item.description,
            );

        assert!(solved.is_none());
        assert_eq!(existing, vec!["existing".to_string()]);
    }

    #[test]
    fn abs_equality_precheck_negative_is_empty() {
        assert_eq!(
            abs_equality_precheck(NumericSign::Negative),
            AbsEqualityPrecheck::ReturnEmptySet
        );
    }

    #[test]
    fn abs_equality_precheck_zero_collapses_to_zero() {
        assert_eq!(
            abs_equality_precheck(NumericSign::Zero),
            AbsEqualityPrecheck::CollapseToZero
        );
    }

    #[test]
    fn abs_equality_precheck_positive_continues() {
        assert_eq!(
            abs_equality_precheck(NumericSign::Positive),
            AbsEqualityPrecheck::Continue
        );
    }

    #[test]
    fn abs_fast_path_non_equality_continues() {
        assert_eq!(
            classify_abs_isolation_fast_path(RelOp::Geq, Some(NumericSign::Negative)),
            AbsIsolationFastPath::Continue
        );
    }

    #[test]
    fn abs_fast_path_negative_rhs_returns_empty() {
        assert_eq!(
            classify_abs_isolation_fast_path(RelOp::Eq, Some(NumericSign::Negative)),
            AbsIsolationFastPath::ReturnEmptySet
        );
    }

    #[test]
    fn abs_fast_path_zero_rhs_collapses_to_zero() {
        assert_eq!(
            classify_abs_isolation_fast_path(RelOp::Eq, Some(NumericSign::Zero)),
            AbsIsolationFastPath::CollapseToZero
        );
    }

    #[test]
    fn abs_fast_path_missing_sign_continues() {
        assert_eq!(
            classify_abs_isolation_fast_path(RelOp::Eq, None),
            AbsIsolationFastPath::Continue
        );
    }

    #[test]
    fn abs_split_case_message_formats_positive_case() {
        let msg = abs_split_case_message(AbsSplitCase::Positive, "x", "=", "2");
        assert_eq!(msg, "Split absolute value (Case 1): x = 2");
    }

    #[test]
    fn abs_split_case_message_formats_negative_case() {
        let msg = abs_split_case_message(AbsSplitCase::Negative, "x", "=", "-2");
        assert_eq!(msg, "Split absolute value (Case 2): x = -2");
    }

    #[test]
    fn build_abs_split_step_with_builds_payload() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let equation = Equation {
            lhs: x,
            rhs: two,
            op: RelOp::Eq,
        };
        let payload = build_abs_split_step_with(
            AbsSplitCase::Positive,
            equation.clone(),
            x,
            two,
            RelOp::Eq,
            |_| "expr".to_string(),
        );
        assert_eq!(payload.equation_after, equation);
        assert_eq!(
            payload.description,
            "Split absolute value (Case 1): expr = expr"
        );
    }

    #[test]
    fn build_abs_split_steps_with_builds_both_payloads() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let neg_two = ctx.num(-2);
        let eq_pos = Equation {
            lhs: x,
            rhs: two,
            op: RelOp::Eq,
        };
        let eq_neg = Equation {
            lhs: x,
            rhs: neg_two,
            op: RelOp::Eq,
        };

        let payload = build_abs_split_steps_with(eq_pos.clone(), eq_neg.clone(), x, |id| {
            if id == two {
                "2".to_string()
            } else if id == neg_two {
                "-2".to_string()
            } else {
                "x".to_string()
            }
        });

        assert_eq!(
            payload.positive.description,
            "Split absolute value (Case 1): x = 2"
        );
        assert_eq!(payload.positive.equation_after, eq_pos);
        assert_eq!(
            payload.negative.description,
            "Split absolute value (Case 2): x = -2"
        );
        assert_eq!(payload.negative.equation_after, eq_neg);
    }

    #[test]
    fn build_abs_split_execution_with_preserves_equations_and_didactic() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let neg_two = ctx.num(-2);
        let eq_pos = Equation {
            lhs: x,
            rhs: two,
            op: RelOp::Eq,
        };
        let eq_neg = Equation {
            lhs: x,
            rhs: neg_two,
            op: RelOp::Eq,
        };

        let execution = build_abs_split_execution_with(eq_pos.clone(), eq_neg.clone(), x, |id| {
            if id == two {
                "2".to_string()
            } else if id == neg_two {
                "-2".to_string()
            } else {
                "x".to_string()
            }
        });

        assert_eq!(execution.positive_equation, eq_pos);
        assert_eq!(execution.negative_equation, eq_neg);
        assert_eq!(execution.items.len(), 2);
        assert_eq!(
            execution.items[0].description,
            "Split absolute value (Case 1): x = 2"
        );
        assert_eq!(
            execution.items[1].description,
            "Split absolute value (Case 2): x = -2"
        );
    }

    #[test]
    fn materialize_abs_split_execution_omits_items() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let neg_two = ctx.num(-2);
        let eq_pos = Equation {
            lhs: x,
            rhs: two,
            op: RelOp::Eq,
        };
        let eq_neg = Equation {
            lhs: x,
            rhs: neg_two,
            op: RelOp::Eq,
        };

        let execution = materialize_abs_split_execution(eq_pos.clone(), eq_neg.clone());
        assert_eq!(execution.positive_equation, eq_pos);
        assert_eq!(execution.negative_equation, eq_neg);
        assert!(execution.items.is_empty());
    }

    #[test]
    fn collect_abs_split_execution_items_preserves_positive_then_negative_order() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let neg_two = ctx.num(-2);
        let execution = build_abs_split_execution_with(
            Equation {
                lhs: x,
                rhs: two,
                op: RelOp::Eq,
            },
            Equation {
                lhs: x,
                rhs: neg_two,
                op: RelOp::Eq,
            },
            x,
            |id| {
                if id == two {
                    "2".to_string()
                } else if id == neg_two {
                    "-2".to_string()
                } else {
                    "x".to_string()
                }
            },
        );

        let items = collect_abs_split_execution_items(&execution);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].description, "Split absolute value (Case 1): x = 2");
        assert_eq!(
            items[1].description,
            "Split absolute value (Case 2): x = -2"
        );
    }

    #[test]
    fn collect_abs_split_execution_items_preserves_equation_alignment() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let neg_two = ctx.num(-2);
        let execution = build_abs_split_execution_with(
            Equation {
                lhs: x,
                rhs: two,
                op: RelOp::Eq,
            },
            Equation {
                lhs: x,
                rhs: neg_two,
                op: RelOp::Eq,
            },
            x,
            |id| {
                if id == two {
                    "2".to_string()
                } else if id == neg_two {
                    "-2".to_string()
                } else {
                    "x".to_string()
                }
            },
        );

        let items = collect_abs_split_execution_items(&execution);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].equation, execution.positive_equation);
        assert_eq!(items[0].description, "Split absolute value (Case 1): x = 2");
        assert_eq!(items[1].equation, execution.negative_equation);
        assert_eq!(
            items[1].description,
            "Split absolute value (Case 2): x = -2"
        );
    }

    #[test]
    fn solve_abs_split_cases_with_solves_positive_then_negative() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let neg_two = ctx.num(-2);
        let residual = ctx.var("residual");
        let execution = materialize_abs_split_execution(
            Equation {
                lhs: x,
                rhs: two,
                op: RelOp::Eq,
            },
            Equation {
                lhs: x,
                rhs: neg_two,
                op: RelOp::Eq,
            },
        );
        let mut call_idx = 0usize;
        let solved = solve_abs_split_cases_with(&execution, |_eq| {
            call_idx += 1;
            Ok::<_, ()>(match call_idx {
                1 => SolutionSet::AllReals,
                2 => SolutionSet::Residual(residual),
                _ => unreachable!("only two abs split branches"),
            })
        })
        .expect("callback should succeed");

        assert_eq!(call_idx, 2);
        assert!(matches!(solved.positive_branch, SolutionSet::AllReals));
        assert!(matches!(
            solved.negative_branch,
            SolutionSet::Residual(id) if id == residual
        ));
    }

    #[test]
    fn solve_abs_split_cases_with_items_passes_aligned_items_in_order() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let neg_two = ctx.num(-2);
        let execution = build_abs_split_execution_with(
            Equation {
                lhs: x,
                rhs: two,
                op: RelOp::Eq,
            },
            Equation {
                lhs: x,
                rhs: neg_two,
                op: RelOp::Eq,
            },
            x,
            |_| "x".to_string(),
        );
        let mut seen = Vec::new();
        let solved = solve_abs_split_cases_with_items(&execution, |item, equation| {
            if let Some(item) = item {
                seen.push((item.description, equation.rhs));
            } else {
                seen.push(("missing".to_string(), equation.rhs));
            }
            Ok::<_, ()>(equation.rhs)
        })
        .expect("callback should succeed");

        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0].0, "Split absolute value (Case 1): x = x");
        assert_eq!(seen[0].1, two);
        assert_eq!(seen[1].0, "Split absolute value (Case 2): x = x");
        assert_eq!(seen[1].1, neg_two);
        assert_eq!(solved.positive_branch, two);
        assert_eq!(solved.negative_branch, neg_two);
    }

    #[test]
    fn solve_abs_split_execution_with_items_returns_branch_sets_and_concatenates_steps() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let neg_two = ctx.num(-2);
        let execution = build_abs_split_execution_with(
            Equation {
                lhs: x,
                rhs: two,
                op: RelOp::Eq,
            },
            Equation {
                lhs: x,
                rhs: neg_two,
                op: RelOp::Eq,
            },
            x,
            |_| "x".to_string(),
        );

        let mut branch_calls = 0usize;
        let solved = solve_abs_split_execution_with_items(&execution, |item, equation| {
            branch_calls += 1;
            let mut steps = vec![format!("branch-{branch_calls}")];
            if let Some(item) = item {
                steps.push(item.description);
            }
            let set = SolutionSet::Discrete(vec![equation.rhs]);
            Ok::<_, ()>((set, steps))
        })
        .expect("split execution should solve");

        assert_eq!(branch_calls, 2);
        assert_eq!(solved.steps[0], "branch-1");
        assert_eq!(solved.steps[2], "branch-2");
        match solved.positive_set {
            SolutionSet::Discrete(ref solutions) => assert_eq!(solutions.as_slice(), &[two]),
            ref other => panic!("expected positive branch discrete set, got {:?}", other),
        }
        match solved.negative_set {
            SolutionSet::Discrete(ref solutions) => assert_eq!(solutions.as_slice(), &[neg_two]),
            ref other => panic!("expected negative branch discrete set, got {:?}", other),
        }
    }

    #[test]
    fn solve_abs_split_execution_with_items_handles_execution_without_items() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let execution = materialize_abs_split_execution(
            Equation {
                lhs: x,
                rhs: two,
                op: RelOp::Eq,
            },
            Equation {
                lhs: x,
                rhs: two,
                op: RelOp::Eq,
            },
        );
        let solved = solve_abs_split_execution_with_items(&execution, |_item, equation| {
            Ok::<_, ()>((SolutionSet::Discrete(vec![equation.rhs]), vec![1u8]))
        })
        .expect("split execution should solve");
        assert_eq!(solved.steps, vec![1u8, 1u8]);
        assert!(matches!(solved.positive_set, SolutionSet::Discrete(_)));
        assert!(matches!(solved.negative_set, SolutionSet::Discrete(_)));
    }

    #[test]
    fn solve_abs_split_execution_pipeline_with_items_prepends_prefix_and_item_steps() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let neg_two = ctx.num(-2);
        let execution = build_abs_split_execution_with(
            Equation {
                lhs: x,
                rhs: two,
                op: RelOp::Eq,
            },
            Equation {
                lhs: x,
                rhs: neg_two,
                op: RelOp::Eq,
            },
            x,
            |_| "x".to_string(),
        );

        let solved = solve_abs_split_execution_pipeline_with_items(
            &execution,
            true,
            &["prefix".to_string()],
            |equation| {
                Ok::<_, ()>((
                    SolutionSet::Discrete(vec![equation.rhs]),
                    vec!["branch".to_string()],
                ))
            },
            |item| item.description,
        )
        .expect("pipeline should solve");

        assert!(matches!(solved.positive_set, SolutionSet::Discrete(_)));
        assert!(matches!(solved.negative_set, SolutionSet::Discrete(_)));
        assert_eq!(solved.steps[0], "prefix");
        assert!(solved.steps[1].starts_with("Split absolute value (Case 1)"));
        assert_eq!(solved.steps[2], "branch");
        assert_eq!(solved.steps[3], "prefix");
        assert!(solved.steps[4].starts_with("Split absolute value (Case 2)"));
        assert_eq!(solved.steps[5], "branch");
    }

    #[test]
    fn solve_abs_split_execution_pipeline_with_items_omits_item_steps_when_disabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let execution = build_abs_split_execution_with(
            Equation {
                lhs: x,
                rhs: two,
                op: RelOp::Eq,
            },
            Equation {
                lhs: x,
                rhs: two,
                op: RelOp::Eq,
            },
            x,
            |_| "x".to_string(),
        );

        let solved = solve_abs_split_execution_pipeline_with_items(
            &execution,
            false,
            &[0u8],
            |_equation| Ok::<_, ()>((SolutionSet::Discrete(vec![two]), vec![1u8])),
            |_item| 9u8,
        )
        .expect("pipeline should solve");

        assert_eq!(solved.steps, vec![0u8, 1u8, 0u8, 1u8]);
    }

    #[test]
    fn solve_abs_split_pipeline_with_optional_items_includes_didactic_when_enabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let neg_two = ctx.num(-2);
        let solved = solve_abs_split_pipeline_with_optional_items(
            Equation {
                lhs: x,
                rhs: two,
                op: RelOp::Eq,
            },
            Equation {
                lhs: x,
                rhs: neg_two,
                op: RelOp::Eq,
            },
            x,
            true,
            &["prefix".to_string()],
            |_| "x".to_string(),
            |equation| {
                Ok::<_, ()>((
                    SolutionSet::Discrete(vec![equation.rhs]),
                    vec!["branch".to_string()],
                ))
            },
            |item| item.description,
        )
        .expect("pipeline should solve");

        assert_eq!(solved.steps[0], "prefix");
        assert!(solved.steps[1].starts_with("Split absolute value (Case 1)"));
        assert_eq!(solved.steps[2], "branch");
        assert_eq!(solved.steps[3], "prefix");
        assert!(solved.steps[4].starts_with("Split absolute value (Case 2)"));
        assert_eq!(solved.steps[5], "branch");
    }

    #[test]
    fn solve_abs_split_pipeline_with_optional_items_skips_render_and_items_when_disabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let solved = solve_abs_split_pipeline_with_optional_items(
            Equation {
                lhs: x,
                rhs: two,
                op: RelOp::Eq,
            },
            Equation {
                lhs: x,
                rhs: two,
                op: RelOp::Eq,
            },
            x,
            false,
            &[0u8],
            |_id| -> String { panic!("renderer must not run when items are disabled") },
            |_equation| Ok::<_, ()>((SolutionSet::Discrete(vec![two]), vec![1u8])),
            |_item| -> u8 { panic!("mapper must not run when items are disabled") },
        )
        .expect("pipeline should solve");

        assert_eq!(solved.steps, vec![0u8, 1u8, 0u8, 1u8]);
    }

    #[test]
    fn execute_abs_split_pipeline_with_optional_items_finalizes_sets() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let neg_two = ctx.num(-2);

        let mut finalize_called = false;
        let (final_set, steps) = execute_abs_split_pipeline_with_optional_items(
            Equation {
                lhs: x,
                rhs: two,
                op: RelOp::Eq,
            },
            Equation {
                lhs: x,
                rhs: neg_two,
                op: RelOp::Eq,
            },
            x,
            false,
            &[0u8],
            |_id| -> String { panic!("renderer must not run when items are disabled") },
            |_equation| Ok::<_, ()>((SolutionSet::Discrete(vec![two]), vec![1u8])),
            |_item| -> u8 { panic!("mapper must not run when items are disabled") },
            |positive_set, negative_set| {
                finalize_called = true;
                assert!(matches!(positive_set, SolutionSet::Discrete(_)));
                assert!(matches!(negative_set, SolutionSet::Discrete(_)));
                SolutionSet::Empty
            },
        )
        .expect("execute helper should solve and finalize");

        assert!(finalize_called);
        assert!(matches!(final_set, SolutionSet::Empty));
        assert_eq!(steps, vec![0u8, 1u8, 0u8, 1u8]);
    }

    #[test]
    fn plan_abs_isolation_returns_empty_for_negative_eq_rhs() {
        let mut ctx = Context::new();
        let arg = ctx.var("x");
        let rhs = ctx.num(-2);
        let plan = plan_abs_isolation(&mut ctx, arg, rhs, RelOp::Eq, Some(NumericSign::Negative));
        assert_eq!(plan, AbsIsolationPlan::ReturnEmptySet);
    }

    #[test]
    fn plan_abs_isolation_collapses_to_single_equation_for_zero_rhs() {
        let mut ctx = Context::new();
        let arg = ctx.var("x");
        let rhs = ctx.num(0);
        let plan = plan_abs_isolation(&mut ctx, arg, rhs, RelOp::Eq, Some(NumericSign::Zero));
        match plan {
            AbsIsolationPlan::IsolateSingleEquation { equation } => {
                assert_eq!(equation.lhs, arg);
                assert_eq!(equation.rhs, rhs);
                assert_eq!(equation.op, RelOp::Eq);
            }
            other => panic!("expected single-equation plan, got {:?}", other),
        }
    }

    #[test]
    fn plan_abs_isolation_splits_branches_for_regular_case() {
        let mut ctx = Context::new();
        let arg = ctx.var("x");
        let rhs = ctx.num(3);
        let plan = plan_abs_isolation(&mut ctx, arg, rhs, RelOp::Eq, Some(NumericSign::Positive));
        assert!(matches!(plan, AbsIsolationPlan::SplitBranches { .. }));
    }

    #[test]
    fn plan_abs_isolation_with_rhs_sign_detects_negative_rhs() {
        let mut ctx = Context::new();
        let arg = ctx.var("x");
        let rhs = ctx.num(-1);
        let plan = plan_abs_isolation_with_rhs_sign(&mut ctx, arg, rhs, RelOp::Eq);
        assert_eq!(plan, AbsIsolationPlan::ReturnEmptySet);
    }

    #[test]
    fn solve_abs_isolation_plan_with_handles_return_empty_set() {
        let solved = solve_abs_isolation_plan_with(
            AbsIsolationPlan::ReturnEmptySet,
            |_eq| Ok::<_, ()>(()),
            |_pos, _neg| Ok::<_, ()>(()),
        )
        .expect("solve should succeed");

        assert!(matches!(solved, AbsIsolationSolved::ReturnedEmptySet));
    }

    #[test]
    fn solve_abs_isolation_plan_with_executes_single_equation_callback() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let rhs = ctx.num(0);
        let equation = Equation {
            lhs: x,
            rhs,
            op: RelOp::Eq,
        };

        let mut single_calls = 0usize;
        let solved = solve_abs_isolation_plan_with(
            AbsIsolationPlan::IsolateSingleEquation {
                equation: equation.clone(),
            },
            |eq| {
                single_calls += 1;
                Ok::<_, ()>(eq)
            },
            |_pos, _neg| Ok::<_, ()>(()),
        )
        .expect("solve should succeed");

        assert_eq!(single_calls, 1);
        match solved {
            AbsIsolationSolved::IsolatedSingle(eq) => assert_eq!(eq, equation),
            other => panic!("expected single solved variant, got {:?}", other),
        }
    }

    #[test]
    fn solve_abs_isolation_plan_with_executes_split_callback() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let neg_two = ctx.num(-2);
        let positive = Equation {
            lhs: x,
            rhs: two,
            op: RelOp::Eq,
        };
        let negative = Equation {
            lhs: x,
            rhs: neg_two,
            op: RelOp::Eq,
        };

        let mut split_calls = 0usize;
        let solved = solve_abs_isolation_plan_with(
            AbsIsolationPlan::SplitBranches {
                positive: positive.clone(),
                negative: negative.clone(),
            },
            |_eq| Ok::<_, ()>(()),
            |pos, neg| {
                split_calls += 1;
                Ok::<_, ()>((pos, neg))
            },
        )
        .expect("solve should succeed");

        assert_eq!(split_calls, 1);
        match solved {
            AbsIsolationSolved::Split((pos, neg)) => {
                assert_eq!(pos, positive);
                assert_eq!(neg, negative);
            }
            other => panic!("expected split solved variant, got {:?}", other),
        }
    }

    #[test]
    fn execute_abs_isolation_plan_pipeline_with_optional_items_returns_empty_with_existing_steps() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let solved = execute_abs_isolation_plan_pipeline_with_optional_items(
            AbsIsolationPlan::ReturnEmptySet,
            x,
            true,
            vec![1u8, 2u8],
            |_id| String::new(),
            |_equation| Ok::<_, ()>((SolutionSet::AllReals, vec![9u8])),
            |_equation| Ok::<_, ()>((SolutionSet::AllReals, vec![9u8])),
            |_item| 0u8,
            |_positive, _negative| SolutionSet::AllReals,
        )
        .expect("pipeline must return empty set");

        assert!(matches!(solved.0, SolutionSet::Empty));
        assert_eq!(solved.1, vec![1u8, 2u8]);
    }

    #[test]
    fn execute_abs_isolation_plan_pipeline_with_optional_items_merges_single_equation_steps() {
        let mut ctx = Context::new();
        let lhs = ctx.var("lhs");
        let rhs = ctx.var("rhs");
        let x = ctx.var("x");
        let equation = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };
        let solved = execute_abs_isolation_plan_pipeline_with_optional_items(
            AbsIsolationPlan::IsolateSingleEquation { equation },
            x,
            true,
            vec![3u8, 4u8],
            |_id| String::new(),
            |_equation| Ok::<_, ()>((SolutionSet::AllReals, vec![1u8, 2u8])),
            |_equation| Ok::<_, ()>((SolutionSet::AllReals, vec![9u8])),
            |_item| 0u8,
            |_positive, _negative| SolutionSet::AllReals,
        )
        .expect("single-equation pipeline must succeed");

        assert!(matches!(solved.0, SolutionSet::AllReals));
        assert_eq!(solved.1, vec![1u8, 2u8, 3u8, 4u8]);
    }

    #[test]
    fn execute_abs_isolation_plan_pipeline_with_optional_items_and_solver_reuses_solver_for_single()
    {
        let mut ctx = Context::new();
        let lhs = ctx.var("lhs");
        let rhs = ctx.var("rhs");
        let x = ctx.var("x");
        let equation = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };
        let mut calls = 0usize;
        let solved = execute_abs_isolation_plan_pipeline_with_optional_items_and_solver(
            AbsIsolationPlan::IsolateSingleEquation { equation },
            x,
            true,
            vec![9u8],
            |_id| String::new(),
            |_equation| {
                calls += 1;
                Ok::<_, ()>((SolutionSet::AllReals, vec![1u8]))
            },
            |_item| 0u8,
            |_positive, _negative| SolutionSet::AllReals,
        )
        .expect("single-equation pipeline must succeed");

        assert_eq!(calls, 1);
        assert!(matches!(solved.0, SolutionSet::AllReals));
        assert_eq!(solved.1, vec![1u8, 9u8]);
    }

    #[test]
    fn abs_guard_wraps_solution_when_rhs_contains_var() {
        let mut ctx = Context::new();
        let rhs = ctx.var("x");
        let out = guard_abs_solution_with_nonnegative_rhs(true, rhs, SolutionSet::AllReals);
        match out {
            SolutionSet::Conditional(cases) => {
                assert_eq!(cases.len(), 1);
                assert_eq!(
                    cases[0].when,
                    ConditionSet::single(ConditionPredicate::NonNegative(rhs))
                );
                assert!(matches!(cases[0].then.solutions, SolutionSet::AllReals));
            }
            other => panic!("expected conditional guard, got {:?}", other),
        }
    }

    #[test]
    fn abs_guard_leaves_solution_unchanged_for_var_free_rhs() {
        let mut ctx = Context::new();
        let rhs = ctx.num(2);
        let out = guard_abs_solution_with_nonnegative_rhs(false, rhs, SolutionSet::Empty);
        assert!(matches!(out, SolutionSet::Empty));
    }

    #[test]
    fn finalize_abs_split_solution_set_combines_and_guards() {
        let mut ctx = Context::new();
        let rhs = ctx.var("x");
        let pos = ctx.num(1);
        let neg = ctx.num(-1);
        let out = finalize_abs_split_solution_set(
            &ctx,
            RelOp::Eq,
            true,
            rhs,
            SolutionSet::Discrete(vec![pos]),
            SolutionSet::Discrete(vec![neg]),
        );

        match out {
            SolutionSet::Conditional(cases) => {
                assert_eq!(cases.len(), 1);
                assert_eq!(
                    cases[0].when,
                    ConditionSet::single(ConditionPredicate::NonNegative(rhs))
                );
                assert!(matches!(cases[0].then.solutions, SolutionSet::Discrete(_)));
            }
            other => panic!("expected guarded conditional, got {:?}", other),
        }
    }

    #[test]
    fn finalize_abs_split_solution_set_for_rhs_adds_guard_when_rhs_has_target_var() {
        let mut ctx = Context::new();
        let rhs = ctx.var("x");
        let pos = ctx.num(1);
        let neg = ctx.num(-1);
        let out = finalize_abs_split_solution_set_for_rhs(
            &ctx,
            RelOp::Eq,
            rhs,
            "x",
            SolutionSet::Discrete(vec![pos]),
            SolutionSet::Discrete(vec![neg]),
        );

        assert!(matches!(out, SolutionSet::Conditional(_)));
    }

    #[test]
    fn finalize_abs_split_solution_set_for_rhs_skips_guard_when_rhs_is_other_var() {
        let mut ctx = Context::new();
        let rhs = ctx.var("y");
        let pos = ctx.num(1);
        let neg = ctx.num(-1);
        let out = finalize_abs_split_solution_set_for_rhs(
            &ctx,
            RelOp::Eq,
            rhs,
            "x",
            SolutionSet::Discrete(vec![pos]),
            SolutionSet::Discrete(vec![neg]),
        );

        assert!(matches!(out, SolutionSet::Discrete(_)));
    }

    #[test]
    fn resolve_log_terminal_empty_set() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let decision = LogSolveDecision::EmptySet("no real solutions");
        let outcome = resolve_log_terminal_outcome(
            &mut ctx,
            &decision,
            DomainModeKind::Generic,
            false,
            x,
            y,
            "x",
        )
        .expect("empty-set terminal outcome");
        assert_eq!(outcome.message, "no real solutions");
        assert!(matches!(outcome.solutions, SolutionSet::Empty));
    }

    #[test]
    fn resolve_log_terminal_residual_in_wildcard() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let decision = LogSolveDecision::NeedsComplex("needs complex log");
        let outcome = resolve_log_terminal_outcome(
            &mut ctx,
            &decision,
            DomainModeKind::Assume,
            true,
            x,
            y,
            "x",
        )
        .expect("residual terminal outcome");
        assert_eq!(outcome.message, "needs complex log");
        assert!(matches!(outcome.solutions, SolutionSet::Residual(_)));
    }

    #[test]
    fn resolve_log_terminal_continue_returns_none() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let decision = LogSolveDecision::Ok;
        let out = resolve_log_terminal_outcome(
            &mut ctx,
            &decision,
            DomainModeKind::Generic,
            false,
            x,
            y,
            "x",
        );
        assert!(out.is_none());
    }

    #[test]
    fn resolve_single_side_exponential_terminal_outcome_returns_empty_when_classifier_says_empty() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let pow = ctx.add(Expr::Pow(two, x));
        let neg_five = ctx.num(-5);

        let out = resolve_single_side_exponential_terminal_outcome(
            &mut ctx,
            pow,
            neg_five,
            "x",
            true,
            false,
            DomainModeKind::Generic,
            false,
            |_ctx, _base, _rhs| LogSolveDecision::EmptySet("no real solutions"),
        )
        .expect("must produce terminal outcome");

        assert_eq!(out.message, "no real solutions");
        assert!(matches!(out.solutions, SolutionSet::Empty));
    }

    #[test]
    fn resolve_single_side_exponential_terminal_with_message_builds_user_text() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let pow = ctx.add(Expr::Pow(two, x));
        let neg_five = ctx.num(-5);

        let (solutions, message) = resolve_single_side_exponential_terminal_with_message(
            &mut ctx,
            pow,
            neg_five,
            "x",
            true,
            false,
            DomainModeKind::Generic,
            false,
            " (residual)",
            |_ctx, _base, _rhs| LogSolveDecision::EmptySet("no real solutions"),
        )
        .expect("must produce terminal message payload");

        assert_eq!(message, "no real solutions");
        assert!(matches!(solutions, SolutionSet::Empty));
    }

    #[test]
    fn resolve_single_side_exponential_terminal_with_step_builds_didactic_payload() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let pow = ctx.add(Expr::Pow(two, x));
        let neg_five = ctx.num(-5);
        let equation_after = Equation {
            lhs: pow,
            rhs: neg_five,
            op: RelOp::Eq,
        };

        let (solutions, step) = resolve_single_side_exponential_terminal_with_step(
            &mut ctx,
            pow,
            neg_five,
            "x",
            true,
            false,
            DomainModeKind::Generic,
            false,
            " (residual)",
            equation_after.clone(),
            |_ctx, _base, _rhs| LogSolveDecision::EmptySet("no real solutions"),
        )
        .expect("must produce terminal didactic payload");

        assert!(matches!(solutions, SolutionSet::Empty));
        assert_eq!(step.description, "no real solutions");
        assert_eq!(step.equation_after, equation_after);
    }

    #[test]
    fn resolve_single_side_exponential_terminal_pipeline_with_item_forwards_step_when_enabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let pow = ctx.add(Expr::Pow(two, x));
        let neg_five = ctx.num(-5);
        let equation_after = Equation {
            lhs: pow,
            rhs: neg_five,
            op: RelOp::Eq,
        };

        let solved = resolve_single_side_exponential_terminal_pipeline_with_item(
            &mut ctx,
            pow,
            neg_five,
            "x",
            true,
            false,
            DomainModeKind::Generic,
            false,
            " (residual)",
            equation_after.clone(),
            true,
            |_ctx, _base, _rhs| LogSolveDecision::EmptySet("no real solutions"),
            |item| item.description,
        )
        .expect("must produce terminal pipeline payload");

        assert!(matches!(solved.solution_set, SolutionSet::Empty));
        assert_eq!(solved.steps, vec!["no real solutions".to_string()]);
    }

    #[test]
    fn resolve_single_side_exponential_terminal_pipeline_with_item_omits_step_when_disabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let pow = ctx.add(Expr::Pow(two, x));
        let neg_five = ctx.num(-5);
        let equation_after = Equation {
            lhs: pow,
            rhs: neg_five,
            op: RelOp::Eq,
        };

        let solved = resolve_single_side_exponential_terminal_pipeline_with_item(
            &mut ctx,
            pow,
            neg_five,
            "x",
            true,
            false,
            DomainModeKind::Generic,
            false,
            " (residual)",
            equation_after,
            false,
            |_ctx, _base, _rhs| LogSolveDecision::EmptySet("no real solutions"),
            |item| item.description,
        )
        .expect("must produce terminal pipeline payload");

        assert!(matches!(solved.solution_set, SolutionSet::Empty));
        assert!(solved.steps.is_empty());
    }

    #[test]
    fn classify_single_side_exponential_log_decision_returns_none_without_single_side_candidate() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let out = classify_single_side_exponential_log_decision(
            &ctx,
            x,
            y,
            "x",
            true,
            true,
            |_ctx, _base, _rhs| LogSolveDecision::Ok,
        );
        assert!(out.is_none());
    }

    #[test]
    fn terminal_outcome_message_appends_suffix_for_residual() {
        let mut ctx = Context::new();
        let residual = ctx.var("residual");
        let outcome = TerminalSolveOutcome {
            message: "needs complex log",
            solutions: SolutionSet::Residual(residual),
        };
        assert_eq!(
            terminal_outcome_message(&outcome, " (residual)"),
            "needs complex log (residual)"
        );
    }

    #[test]
    fn terminal_outcome_message_keeps_message_for_non_residual() {
        let outcome = TerminalSolveOutcome {
            message: "no real solutions",
            solutions: SolutionSet::Empty,
        };
        assert_eq!(
            terminal_outcome_message(&outcome, " (residual)"),
            "no real solutions"
        );
    }

    #[test]
    fn solve_terminal_outcome_pipeline_with_item_emits_step_when_enabled() {
        let mut ctx = Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.var("y");
        let outcome = TerminalSolveOutcome {
            message: "no real solutions",
            solutions: SolutionSet::Empty,
        };
        let solved = solve_terminal_outcome_pipeline_with_item(
            outcome,
            Equation {
                lhs,
                rhs,
                op: RelOp::Eq,
            },
            " (residual)",
            true,
            |item| item.description,
        );

        assert!(matches!(solved.solution_set, SolutionSet::Empty));
        assert_eq!(solved.steps, vec!["no real solutions".to_string()]);
    }

    #[test]
    fn solve_terminal_outcome_pipeline_with_item_omits_step_when_disabled() {
        let mut ctx = Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.var("y");
        let residual = ctx.var("residual");
        let outcome = TerminalSolveOutcome {
            message: "needs complex log",
            solutions: SolutionSet::Residual(residual),
        };
        let solved = solve_terminal_outcome_pipeline_with_item(
            outcome,
            Equation {
                lhs,
                rhs,
                op: RelOp::Eq,
            },
            " (residual)",
            false,
            |item| item.description,
        );

        assert!(matches!(solved.solution_set, SolutionSet::Residual(id) if id == residual));
        assert!(solved.steps.is_empty());
    }

    #[test]
    fn execute_log_terminal_outcome_pipeline_with_item_returns_none_for_non_terminal_decision() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let out = execute_log_terminal_outcome_pipeline_with_item(
            &mut ctx,
            &LogSolveDecision::Ok,
            DomainModeKind::Generic,
            false,
            x,
            y,
            "x",
            Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            " (residual)",
            true,
            |item| item.description,
        );
        assert!(out.is_none());
    }

    #[test]
    fn execute_log_terminal_outcome_pipeline_with_item_builds_step_for_terminal_decision() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let solved = execute_log_terminal_outcome_pipeline_with_item(
            &mut ctx,
            &LogSolveDecision::EmptySet("no real solutions"),
            DomainModeKind::Generic,
            false,
            x,
            y,
            "x",
            Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            " (residual)",
            true,
            |item| item.description,
        )
        .expect("must produce terminal pipeline payload");

        assert!(matches!(solved.solution_set, SolutionSet::Empty));
        assert_eq!(solved.steps, vec!["no real solutions".to_string()]);
    }

    #[test]
    fn execute_log_terminal_outcome_pipeline_with_item_and_merge_with_existing_steps_with_appends_existing(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let solved =
            execute_log_terminal_outcome_pipeline_with_item_and_merge_with_existing_steps_with(
                &mut ctx,
                &LogSolveDecision::EmptySet("no real solutions"),
                DomainModeKind::Generic,
                false,
                x,
                y,
                "x",
                Equation {
                    lhs: x,
                    rhs: y,
                    op: RelOp::Eq,
                },
                " (residual)",
                true,
                vec!["existing".to_string()],
                |item| item.description,
            )
            .expect("must produce merged terminal payload");

        assert!(matches!(solved.0, SolutionSet::Empty));
        assert_eq!(
            solved.1,
            vec!["existing".to_string(), "no real solutions".to_string()]
        );
    }

    #[test]
    fn execute_log_terminal_outcome_and_assumptions_gate_with_returns_terminal_when_applicable() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");

        let gated = execute_log_terminal_outcome_and_assumptions_gate_with(
            &mut ctx,
            &LogSolveDecision::EmptySet("no real solutions"),
            DomainModeKind::Generic,
            false,
            x,
            y,
            "x",
            Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            " (residual)",
            true,
            vec!["existing".to_string()],
            |item| item.description,
        );

        match gated {
            LogDecisionTerminalGate::Terminal {
                solution_set,
                steps,
            } => {
                assert!(matches!(solution_set, SolutionSet::Empty));
                assert_eq!(
                    steps,
                    vec!["existing".to_string(), "no real solutions".to_string()]
                );
            }
            other => panic!("expected terminal gate, got {:?}", other),
        }
    }

    #[test]
    fn execute_log_terminal_outcome_and_assumptions_gate_with_continues_and_emits_assumptions() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let decision = LogSolveDecision::OkWithAssumptions(vec![
            LogAssumption::PositiveBase,
            LogAssumption::PositiveRhs,
        ]);

        let gated = execute_log_terminal_outcome_and_assumptions_gate_with(
            &mut ctx,
            &decision,
            DomainModeKind::Generic,
            false,
            x,
            y,
            "x",
            Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            " (residual)",
            true,
            vec!["existing".to_string()],
            |item| item.description,
        );

        match gated {
            LogDecisionTerminalGate::Continue { assumptions } => assert_eq!(
                assumptions,
                vec![LogAssumption::PositiveBase, LogAssumption::PositiveRhs]
            ),
            other => panic!("expected continue gate, got {:?}", other),
        }
    }

    #[test]
    fn execute_log_terminal_outcome_and_assumptions_gate_with_reports_needs_complex() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");

        let gated = execute_log_terminal_outcome_and_assumptions_gate_with(
            &mut ctx,
            &LogSolveDecision::NeedsComplex("need complex"),
            DomainModeKind::Generic,
            false,
            x,
            y,
            "x",
            Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            " (residual)",
            true,
            vec!["existing".to_string()],
            |item| item.description,
        );

        match gated {
            LogDecisionTerminalGate::NeedsComplex {
                message,
                assumptions,
            } => {
                assert_eq!(message, "need complex");
                assert!(assumptions.is_empty());
            }
            other => panic!("expected needs-complex gate, got {:?}", other),
        }
    }

    #[test]
    fn execute_log_terminal_outcome_and_assumptions_gate_with_existing_steps_mut_merges_terminal_and_clears_existing(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let mut existing = vec!["existing".to_string()];

        let gated = execute_log_terminal_outcome_and_assumptions_gate_with_existing_steps_mut(
            &mut ctx,
            &LogSolveDecision::EmptySet("no real solutions"),
            DomainModeKind::Generic,
            false,
            x,
            y,
            "x",
            Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            " (residual)",
            true,
            &mut existing,
            |item| item.description,
        );

        match gated {
            LogDecisionTerminalGate::Terminal {
                solution_set,
                steps,
            } => {
                assert!(matches!(solution_set, SolutionSet::Empty));
                assert_eq!(
                    steps,
                    vec!["existing".to_string(), "no real solutions".to_string()]
                );
            }
            other => panic!("expected terminal gate, got {:?}", other),
        }
        assert!(existing.is_empty());
    }

    #[test]
    fn execute_log_terminal_outcome_and_assumptions_gate_with_existing_steps_mut_preserves_existing_on_continue(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let decision = LogSolveDecision::OkWithAssumptions(vec![LogAssumption::PositiveRhs]);
        let mut existing = vec!["existing".to_string()];

        let gated = execute_log_terminal_outcome_and_assumptions_gate_with_existing_steps_mut(
            &mut ctx,
            &decision,
            DomainModeKind::Generic,
            false,
            x,
            y,
            "x",
            Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            " (residual)",
            true,
            &mut existing,
            |item| item.description,
        );

        match gated {
            LogDecisionTerminalGate::Continue { assumptions } => {
                assert_eq!(assumptions, vec![LogAssumption::PositiveRhs]);
            }
            other => panic!("expected continue gate, got {:?}", other),
        }
        assert_eq!(existing, vec!["existing".to_string()]);
    }

    #[test]
    fn execute_log_terminal_outcome_and_assumptions_gate_with_existing_steps_mut_and_each_assumption_emits_continue_assumptions(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let mut existing = vec!["existing".to_string()];
        let mut emitted = Vec::new();

        let outcome =
            execute_log_terminal_outcome_and_assumptions_gate_with_existing_steps_mut_and_each_assumption(
                &mut ctx,
                &LogSolveDecision::OkWithAssumptions(vec![LogAssumption::PositiveRhs]),
                DomainModeKind::Generic,
                false,
                x,
                y,
                "x",
                Equation {
                    lhs: x,
                    rhs: y,
                    op: RelOp::Eq,
                },
                " (residual)",
                true,
                &mut existing,
                |item| item.description,
                |_ctx, assumption| emitted.push(assumption),
            );

        assert!(matches!(outcome, LogDecisionTerminalResult::Continue));
        assert_eq!(emitted, vec![LogAssumption::PositiveRhs]);
        assert_eq!(existing, vec!["existing".to_string()]);
    }

    #[test]
    fn execute_log_terminal_outcome_and_assumptions_gate_with_existing_steps_mut_and_each_assumption_for_terminal_skips_emission(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let mut existing = vec!["existing".to_string()];
        let mut emitted = Vec::new();

        let outcome =
            execute_log_terminal_outcome_and_assumptions_gate_with_existing_steps_mut_and_each_assumption(
                &mut ctx,
                &LogSolveDecision::EmptySet("no real solutions"),
                DomainModeKind::Generic,
                false,
                x,
                y,
                "x",
                Equation {
                    lhs: x,
                    rhs: y,
                    op: RelOp::Eq,
                },
                " (residual)",
                true,
                &mut existing,
                |item| item.description,
                |_ctx, assumption| emitted.push(assumption),
            );

        match outcome {
            LogDecisionTerminalResult::Terminal {
                solution_set,
                steps,
            } => {
                assert!(matches!(solution_set, SolutionSet::Empty));
                assert_eq!(
                    steps,
                    vec!["existing".to_string(), "no real solutions".to_string()]
                );
            }
            other => panic!("expected terminal outcome, got {:?}", other),
        }
        assert!(emitted.is_empty());
        assert!(existing.is_empty());
    }

    #[test]
    fn execute_pow_exponent_log_decision_pipeline_with_existing_steps_mut_returns_terminal() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let mut existing = vec!["existing".to_string()];

        let out = execute_pow_exponent_log_decision_pipeline_with_existing_steps_mut(
            &mut ctx,
            &LogSolveDecision::EmptySet("no real solutions"),
            DomainModeKind::Generic,
            false,
            x,
            y,
            "x",
            Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            " (residual)",
            true,
            &mut existing,
            |item| item.description,
            |_ctx, _assumption| panic!("assumptions should not emit on terminal empty-set"),
            true,
            || panic!("unsupported planning must not run on terminal route"),
            |_equation| panic!("guarded solve must not run on terminal route"),
            |_hint| panic!("blocked-hint registration must not run on terminal route"),
        );

        match out {
            PowExponentLogDecisionPipelineResult::Terminal {
                solution_set,
                steps,
            } => {
                assert!(matches!(solution_set, SolutionSet::Empty));
                assert_eq!(
                    steps,
                    vec!["existing".to_string(), "no real solutions".to_string()]
                );
            }
            other => panic!("expected terminal result, got {:?}", other),
        }
        assert!(existing.is_empty());
    }

    #[test]
    fn execute_pow_exponent_log_decision_pipeline_with_existing_steps_mut_reports_needs_complex() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let mut existing = vec!["existing".to_string()];
        let mut seen = Vec::new();

        let out = execute_pow_exponent_log_decision_pipeline_with_existing_steps_mut(
            &mut ctx,
            &LogSolveDecision::NeedsComplex("need complex"),
            DomainModeKind::Generic,
            false,
            x,
            y,
            "x",
            Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            " (residual)",
            true,
            &mut existing,
            |item| item.description,
            |_ctx, assumption| seen.push(assumption),
            true,
            || panic!("unsupported planning must not run for needs-complex"),
            |_equation| panic!("guarded solve must not run for needs-complex"),
            |_hint| panic!("blocked-hint registration must not run for needs-complex"),
        );

        match out {
            PowExponentLogDecisionPipelineResult::NeedsComplex { message } => {
                assert_eq!(message, "need complex");
            }
            other => panic!("expected needs-complex result, got {:?}", other),
        }
        assert!(seen.is_empty());
        assert_eq!(existing, vec!["existing".to_string()]);
    }

    #[test]
    fn execute_pow_exponent_log_decision_pipeline_with_existing_steps_mut_solves_unsupported() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let residual = ctx.add(Expr::Sub(x, y));
        let mut existing = vec!["existing".to_string()];

        let out = execute_pow_exponent_log_decision_pipeline_with_existing_steps_mut(
            &mut ctx,
            &LogSolveDecision::Unsupported(
                "Cannot prove RHS > 0 for logarithm",
                vec![LogAssumption::PositiveRhs],
            ),
            DomainModeKind::Generic,
            false,
            x,
            y,
            "x",
            Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            " (residual)",
            true,
            &mut existing,
            |item| item.description,
            |_ctx, _assumption| {},
            true,
            || {
                Some(PowExponentLogUnsupportedExecution::Residual {
                    item: TermIsolationExecutionItem {
                        description: "unsupported residual".to_string(),
                        equation: Equation {
                            lhs: x,
                            rhs: y,
                            op: RelOp::Eq,
                        },
                    },
                    solutions: SolutionSet::Residual(residual),
                })
            },
            |_equation| panic!("guarded solve must not run for residual unsupported plan"),
            |_hint| panic!("blocked hints should be empty for residual unsupported plan"),
        );

        match out {
            PowExponentLogDecisionPipelineResult::UnsupportedSolved {
                solution_set,
                steps,
            } => {
                assert!(matches!(solution_set, SolutionSet::Residual(_)));
                assert_eq!(
                    steps,
                    vec!["existing".to_string(), "unsupported residual".to_string()]
                );
            }
            other => panic!("expected unsupported-solved result, got {:?}", other),
        }
        assert!(existing.is_empty());
    }

    #[test]
    fn execute_pow_exponent_log_post_terminal_pipeline_with_existing_steps_mut_forwards_terminal() {
        let out = execute_pow_exponent_log_post_terminal_pipeline_with_existing_steps_mut::<
            String,
            _,
            _,
            _,
            _,
        >(
            LogDecisionTerminalResult::Terminal {
                solution_set: SolutionSet::Empty,
                steps: vec!["existing".to_string(), "terminal".to_string()],
            },
            true,
            &mut Vec::new(),
            || panic!("unsupported planning must not run for terminal result"),
            |_equation| panic!("guarded solve must not run for terminal result"),
            |_item| panic!("step mapper must not run for terminal result"),
            |_hint| panic!("hint registration must not run for terminal result"),
        );

        match out {
            PowExponentLogDecisionPipelineResult::Terminal {
                solution_set,
                steps,
            } => {
                assert!(matches!(solution_set, SolutionSet::Empty));
                assert_eq!(steps, vec!["existing".to_string(), "terminal".to_string()]);
            }
            other => panic!("expected terminal result, got {:?}", other),
        }
    }

    #[test]
    fn execute_pow_exponent_log_post_terminal_pipeline_with_existing_steps_mut_solves_unsupported()
    {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let residual = ctx.add(Expr::Sub(x, y));
        let mut existing = vec!["existing".to_string()];

        let out = execute_pow_exponent_log_post_terminal_pipeline_with_existing_steps_mut(
            LogDecisionTerminalResult::Continue,
            true,
            &mut existing,
            || {
                Some(PowExponentLogUnsupportedExecution::Residual {
                    item: TermIsolationExecutionItem {
                        description: "unsupported residual".to_string(),
                        equation: Equation {
                            lhs: x,
                            rhs: y,
                            op: RelOp::Eq,
                        },
                    },
                    solutions: SolutionSet::Residual(residual),
                })
            },
            |_equation| panic!("guarded solve must not run for residual unsupported plan"),
            |item| item.description,
            |_hint| panic!("blocked hints should be empty for residual unsupported plan"),
        );

        match out {
            PowExponentLogDecisionPipelineResult::UnsupportedSolved {
                solution_set,
                steps,
            } => {
                assert!(matches!(solution_set, SolutionSet::Residual(_)));
                assert_eq!(
                    steps,
                    vec!["existing".to_string(), "unsupported residual".to_string()]
                );
            }
            other => panic!("expected unsupported-solved result, got {:?}", other),
        }
        assert!(existing.is_empty());
    }

    #[test]
    fn execute_and_resolve_pow_exponent_log_post_terminal_pipeline_with_existing_steps_mut_maps_terminal(
    ) {
        let out =
            execute_and_resolve_pow_exponent_log_post_terminal_pipeline_with_existing_steps_mut::<
                String,
                _,
                _,
                _,
                _,
            >(
                LogDecisionTerminalResult::Terminal {
                    solution_set: SolutionSet::Empty,
                    steps: vec!["terminal".to_string()],
                },
                true,
                &mut Vec::new(),
                || panic!("unsupported planning must not run for terminal result"),
                |_equation| panic!("guarded solve must not run for terminal result"),
                |_item| panic!("step mapper must not run for terminal result"),
                |_hint| panic!("hint registration must not run for terminal result"),
            )
            .expect("terminal should not error");

        assert_eq!(
            out,
            Some((SolutionSet::Empty, vec!["terminal".to_string()]))
        );
    }

    #[test]
    fn execute_and_resolve_pow_exponent_log_post_terminal_pipeline_with_existing_steps_mut_maps_needs_complex(
    ) {
        let out =
            execute_and_resolve_pow_exponent_log_post_terminal_pipeline_with_existing_steps_mut::<
                String,
                _,
                _,
                _,
                _,
            >(
                LogDecisionTerminalResult::NeedsComplex {
                    message: "complex required",
                },
                true,
                &mut Vec::new(),
                || panic!("unsupported planning must not run for needs-complex"),
                |_equation| panic!("guarded solve must not run for needs-complex"),
                |_item| panic!("step mapper must not run for needs-complex"),
                |_hint| panic!("hint registration must not run for needs-complex"),
            )
            .expect_err("needs-complex should become error");

        assert_eq!(out, "complex required");
    }

    #[test]
    fn execute_and_resolve_pow_exponent_log_decision_pipeline_with_existing_steps_mut_maps_terminal(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let mut existing = vec!["existing".to_string()];
        let mut assumption_calls = 0usize;

        let out = execute_and_resolve_pow_exponent_log_decision_pipeline_with_existing_steps_mut(
            &mut ctx,
            &LogSolveDecision::EmptySet("no real solutions"),
            DomainModeKind::Generic,
            false,
            x,
            y,
            "x",
            Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            " (residual)",
            false,
            &mut existing,
            |item| item.description,
            |_core_ctx, _assumption| assumption_calls += 1,
            true,
            || panic!("unsupported planning must not run for terminal decision"),
            |_equation| panic!("guarded solve must not run for terminal decision"),
            |_hint| panic!("hint registration must not run for terminal decision"),
        )
        .expect("terminal should not error");

        assert_eq!(
            out,
            Some((SolutionSet::Empty, vec!["existing".to_string()]))
        );
        assert_eq!(assumption_calls, 0);
    }

    #[test]
    fn execute_and_resolve_pow_exponent_log_decision_pipeline_with_existing_steps_mut_maps_continue(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let mut existing = vec!["existing".to_string()];
        let mut assumption_calls = 0usize;

        let out = execute_and_resolve_pow_exponent_log_decision_pipeline_with_existing_steps_mut(
            &mut ctx,
            &LogSolveDecision::Ok,
            DomainModeKind::Generic,
            false,
            x,
            y,
            "x",
            Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            " (residual)",
            false,
            &mut existing,
            |item| item.description,
            |_core_ctx, _assumption| assumption_calls += 1,
            true,
            || None,
            |_equation| panic!("guarded solve must not run when unsupported plan is none"),
            |_hint| panic!("hint registration must not run when unsupported plan is none"),
        )
        .expect("continue should not error");

        assert_eq!(out, None);
        assert_eq!(existing, vec!["existing".to_string()]);
        assert_eq!(assumption_calls, 0);
    }

    #[test]
    fn resolve_log_unsupported_outcome_returns_none_for_supported_decision() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let out =
            resolve_log_unsupported_outcome(&mut ctx, &LogSolveDecision::Ok, true, x, x, "x", x, x);
        assert!(out.is_none());
    }

    #[test]
    fn resolve_log_unsupported_outcome_returns_residual_when_budget_exhausted() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let decision = LogSolveDecision::Unsupported(
            "Cannot prove RHS > 0 for logarithm",
            vec![crate::log_domain::LogAssumption::PositiveRhs],
        );
        let out = resolve_log_unsupported_outcome(&mut ctx, &decision, false, x, y, "x", x, y)
            .expect("must produce residual outcome");
        match out {
            LogUnsupportedOutcome::ResidualBudgetExhausted { message, solutions } => {
                assert_eq!(message, "Cannot prove RHS > 0 for logarithm");
                assert!(matches!(solutions, SolutionSet::Residual(_)));
            }
            other => panic!("expected residual outcome, got {:?}", other),
        }
    }

    #[test]
    fn resolve_log_unsupported_outcome_returns_guarded_when_branching_allowed() {
        let mut ctx = Context::new();
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let decision = LogSolveDecision::Unsupported(
            "Cannot prove base > 0 and RHS > 0 for logarithm",
            vec![
                crate::log_domain::LogAssumption::PositiveBase,
                crate::log_domain::LogAssumption::PositiveRhs,
            ],
        );
        let out =
            resolve_log_unsupported_outcome(&mut ctx, &decision, true, base, rhs, "x", base, rhs)
                .expect("must produce guarded outcome");
        match out {
            LogUnsupportedOutcome::Guarded {
                message,
                missing_conditions,
                guard,
                residual,
            } => {
                assert_eq!(message, "Cannot prove base > 0 and RHS > 0 for logarithm");
                assert_eq!(missing_conditions.len(), 2);
                assert_eq!(guard.predicates().len(), 2);
                assert!(matches!(ctx.get(residual), Expr::Function(_, _)));
            }
            other => panic!("expected guarded outcome, got {:?}", other),
        }
    }

    #[test]
    fn collect_guarded_log_blocked_hints_maps_assumptions_to_targets() {
        let mut ctx = Context::new();
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let hints = collect_guarded_log_blocked_hints(
            &[
                crate::log_domain::LogAssumption::PositiveBase,
                crate::log_domain::LogAssumption::PositiveRhs,
            ],
            base,
            rhs,
        );

        assert_eq!(hints.len(), 2);
        assert_eq!(
            hints[0].assumption,
            crate::log_domain::LogAssumption::PositiveBase
        );
        assert_eq!(hints[0].expr_id, base);
        assert_eq!(hints[0].rule, "Take log of both sides");
        assert_eq!(hints[0].suggestion, "use `semantics set domain assume`");
        assert_eq!(
            hints[1].assumption,
            crate::log_domain::LogAssumption::PositiveRhs
        );
        assert_eq!(hints[1].expr_id, rhs);
    }

    #[test]
    fn plan_pow_exponent_log_unsupported_execution_with_builds_residual_variant() {
        let mut ctx = Context::new();
        let exponent = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let decision = LogSolveDecision::Unsupported(
            "Cannot prove RHS > 0 for logarithm",
            vec![crate::log_domain::LogAssumption::PositiveRhs],
        );
        let outcome = resolve_log_unsupported_outcome(
            &mut ctx, &decision, false, exponent, rhs, "x", base, rhs,
        )
        .expect("must produce unsupported outcome");
        let execution = plan_pow_exponent_log_unsupported_execution_with(
            &mut ctx,
            outcome,
            exponent,
            base,
            rhs,
            RelOp::Eq,
            Equation {
                lhs: exponent,
                rhs,
                op: RelOp::Eq,
            },
            |_, _| "a".to_string(),
        );
        match execution {
            PowExponentLogUnsupportedExecution::Residual { item, solutions } => {
                assert_eq!(
                    item.description(),
                    "Cannot prove RHS > 0 for logarithm (residual, budget exhausted)"
                );
                assert!(matches!(solutions, SolutionSet::Residual(_)));
            }
            other => panic!("expected residual variant, got {:?}", other),
        }
    }

    #[test]
    fn plan_pow_exponent_log_unsupported_execution_with_builds_guarded_variant() {
        let mut ctx = Context::new();
        let exponent = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let decision = LogSolveDecision::Unsupported(
            "Cannot prove base > 0 and RHS > 0 for logarithm",
            vec![
                crate::log_domain::LogAssumption::PositiveBase,
                crate::log_domain::LogAssumption::PositiveRhs,
            ],
        );
        let outcome = resolve_log_unsupported_outcome(
            &mut ctx, &decision, true, exponent, rhs, "x", base, rhs,
        )
        .expect("must produce unsupported outcome");
        let execution = plan_pow_exponent_log_unsupported_execution_with(
            &mut ctx,
            outcome,
            exponent,
            base,
            rhs,
            RelOp::Eq,
            Equation {
                lhs: exponent,
                rhs,
                op: RelOp::Eq,
            },
            |_, _| "a".to_string(),
        );
        match execution {
            PowExponentLogUnsupportedExecution::Guarded {
                blocked_hints,
                plan,
                guard,
                residual,
            } => {
                assert_eq!(blocked_hints.len(), 2);
                assert_eq!(plan.rewrite.items.len(), 1);
                assert_eq!(guard.predicates().len(), 2);
                assert!(matches!(ctx.get(residual), Expr::Function(_, _)));
            }
            other => panic!("expected guarded variant, got {:?}", other),
        }
    }

    #[test]
    fn execute_pow_exponent_log_unsupported_with_passes_residual_through() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let item = build_residual_budget_exhausted_item(
            "budget exhausted",
            Equation {
                lhs: x,
                rhs: x,
                op: RelOp::Eq,
            },
        );
        let out = execute_pow_exponent_log_unsupported_with(
            PowExponentLogUnsupportedExecution::Residual {
                item: item.clone(),
                solutions: SolutionSet::Residual(x),
            },
            |_eq| Some(SolutionSet::AllReals),
        );

        match out {
            PowExponentLogUnsupportedSolvedExecution::Residual {
                item: out_item,
                solutions,
            } => {
                assert_eq!(out_item, item);
                assert!(matches!(solutions, SolutionSet::Residual(id) if id == x));
            }
            other => panic!("expected residual solved execution, got {:?}", other),
        }
    }

    #[test]
    fn execute_pow_exponent_log_unsupported_with_executes_guarded_and_materializes_solution() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let a = ctx.var("a");
        let b = ctx.var("b");
        let residual = ctx.var("residual");
        let source = Equation {
            lhs: x,
            rhs: b,
            op: RelOp::Eq,
        };
        let plan = plan_guarded_pow_exponent_log_execution(
            &mut ctx,
            x,
            a,
            b,
            RelOp::Eq,
            "a > 0",
            "a",
            source,
        );
        let guard = ConditionSet::single(ConditionPredicate::Positive(a));
        let out = execute_pow_exponent_log_unsupported_with(
            PowExponentLogUnsupportedExecution::Guarded {
                blocked_hints: vec![],
                plan,
                guard,
                residual,
            },
            |_eq| Some(SolutionSet::Discrete(vec![x])),
        );

        match out {
            PowExponentLogUnsupportedSolvedExecution::Guarded {
                blocked_hints,
                rewrite_item,
                followup_item,
                solutions,
            } => {
                assert!(blocked_hints.is_empty());
                assert!(rewrite_item.is_some());
                assert_eq!(followup_item.description(), "Conditional solution: a > 0");
                assert!(matches!(solutions, SolutionSet::Conditional(_)));
            }
            other => panic!("expected guarded solved execution, got {:?}", other),
        }
    }

    #[test]
    fn pow_exponent_log_execution_item_as_term_item_copies_fields() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let item = PowExponentLogIsolationExecutionItem {
            equation: Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            description: "rewrite".to_string(),
        };

        let term_item = pow_exponent_log_execution_item_as_term_item(item.clone());
        assert_eq!(term_item.equation, item.equation);
        assert_eq!(term_item.description, item.description);
    }

    #[test]
    fn solve_pow_exponent_log_unsupported_pipeline_with_items_maps_residual_item() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let item = build_residual_budget_exhausted_item(
            "budget exhausted",
            Equation {
                lhs: x,
                rhs: x,
                op: RelOp::Eq,
            },
        );

        let solved = solve_pow_exponent_log_unsupported_pipeline_with_items(
            PowExponentLogUnsupportedExecution::Residual {
                item: item.clone(),
                solutions: SolutionSet::Residual(x),
            },
            true,
            |_eq| Some(SolutionSet::AllReals),
            |mapped| mapped.description,
        );

        assert!(solved.blocked_hints.is_empty());
        assert!(matches!(solved.solution_set, SolutionSet::Residual(id) if id == x));
        assert_eq!(solved.steps, vec![item.description]);
    }

    #[test]
    fn solve_pow_exponent_log_unsupported_pipeline_with_items_preserves_guarded_hints() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let a = ctx.var("a");
        let b = ctx.var("b");
        let residual = ctx.var("residual");
        let source = Equation {
            lhs: x,
            rhs: b,
            op: RelOp::Eq,
        };
        let plan = plan_guarded_pow_exponent_log_execution(
            &mut ctx,
            x,
            a,
            b,
            RelOp::Eq,
            "a > 0",
            "a",
            source,
        );
        let guard = ConditionSet::single(ConditionPredicate::Positive(a));
        let solved = solve_pow_exponent_log_unsupported_pipeline_with_items(
            PowExponentLogUnsupportedExecution::Guarded {
                blocked_hints: vec![LogBlockedHintRecord {
                    assumption: crate::log_domain::LogAssumption::PositiveBase,
                    expr_id: a,
                    rule: "Take log of both sides",
                    suggestion: "use `semantics set domain assume`",
                }],
                plan,
                guard,
                residual,
            },
            false,
            |_eq| Some(SolutionSet::Discrete(vec![x])),
            |_mapped| "unused".to_string(),
        );

        assert_eq!(solved.blocked_hints.len(), 1);
        assert_eq!(
            solved.blocked_hints[0].assumption,
            crate::log_domain::LogAssumption::PositiveBase
        );
        assert!(solved.steps.is_empty());
        assert!(matches!(solved.solution_set, SolutionSet::Conditional(_)));
    }

    #[test]
    fn plan_pow_exponent_log_unsupported_execution_from_decision_with_returns_none_for_supported() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let out = plan_pow_exponent_log_unsupported_execution_from_decision_with(
            &mut ctx,
            &LogSolveDecision::Ok,
            true,
            x,
            x,
            "x",
            x,
            x,
            x,
            RelOp::Eq,
            Equation {
                lhs: x,
                rhs: x,
                op: RelOp::Eq,
            },
            |_, _| "x".to_string(),
        );
        assert!(out.is_none());
    }

    #[test]
    fn plan_pow_exponent_log_unsupported_execution_from_decision_with_maps_guarded_variant() {
        let mut ctx = Context::new();
        let exponent = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let decision = LogSolveDecision::Unsupported(
            "Cannot prove base > 0 and RHS > 0 for logarithm",
            vec![
                crate::log_domain::LogAssumption::PositiveBase,
                crate::log_domain::LogAssumption::PositiveRhs,
            ],
        );
        let out = plan_pow_exponent_log_unsupported_execution_from_decision_with(
            &mut ctx,
            &decision,
            true,
            exponent,
            rhs,
            "x",
            exponent,
            base,
            rhs,
            RelOp::Eq,
            Equation {
                lhs: exponent,
                rhs,
                op: RelOp::Eq,
            },
            |_, _| "a".to_string(),
        )
        .expect("must map unsupported decision");
        assert!(matches!(
            out,
            PowExponentLogUnsupportedExecution::Guarded { .. }
        ));
    }

    #[test]
    fn plan_pow_exponent_log_unsupported_execution_from_decision_with_uses_renderer() {
        let mut ctx = Context::new();
        let exponent = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let decision = LogSolveDecision::Unsupported(
            "Cannot prove base > 0 and RHS > 0 for logarithm",
            vec![
                crate::log_domain::LogAssumption::PositiveBase,
                crate::log_domain::LogAssumption::PositiveRhs,
            ],
        );
        let out = plan_pow_exponent_log_unsupported_execution_from_decision_with(
            &mut ctx,
            &decision,
            true,
            exponent,
            rhs,
            "x",
            exponent,
            base,
            rhs,
            RelOp::Eq,
            Equation {
                lhs: exponent,
                rhs,
                op: RelOp::Eq,
            },
            |_, id| format!("runtime({id})"),
        )
        .expect("must map unsupported decision");
        match out {
            PowExponentLogUnsupportedExecution::Guarded { plan, .. } => {
                assert!(plan.rewrite.items[0]
                    .description
                    .contains("log base runtime("));
            }
            other => panic!("expected guarded execution, got {:?}", other),
        }
    }

    #[test]
    fn execute_pow_exponent_log_unsupported_pipeline_from_decision_with_returns_none_for_supported()
    {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let out = execute_pow_exponent_log_unsupported_pipeline_from_decision_with(
            true,
            || {
                plan_pow_exponent_log_unsupported_execution_from_decision_with(
                    &mut ctx,
                    &LogSolveDecision::Ok,
                    true,
                    x,
                    x,
                    "x",
                    x,
                    x,
                    x,
                    RelOp::Eq,
                    Equation {
                        lhs: x,
                        rhs: x,
                        op: RelOp::Eq,
                    },
                    |_, _| "x".to_string(),
                )
            },
            |_eq| Some(SolutionSet::AllReals),
            |_item| "unused".to_string(),
        );
        assert!(out.is_none());
    }

    #[test]
    fn execute_pow_exponent_log_unsupported_pipeline_from_decision_with_maps_guarded_steps() {
        let mut ctx = Context::new();
        let exponent = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let decision = LogSolveDecision::Unsupported(
            "Cannot prove base > 0 and RHS > 0 for logarithm",
            vec![
                crate::log_domain::LogAssumption::PositiveBase,
                crate::log_domain::LogAssumption::PositiveRhs,
            ],
        );
        let out = execute_pow_exponent_log_unsupported_pipeline_from_decision_with(
            true,
            || {
                plan_pow_exponent_log_unsupported_execution_from_decision_with(
                    &mut ctx,
                    &decision,
                    true,
                    exponent,
                    rhs,
                    "x",
                    exponent,
                    base,
                    rhs,
                    RelOp::Eq,
                    Equation {
                        lhs: exponent,
                        rhs,
                        op: RelOp::Eq,
                    },
                    |_, id| format!("runtime({id})"),
                )
            },
            |_eq| Some(SolutionSet::Discrete(vec![exponent])),
            |item| item.description,
        )
        .expect("unsupported decision should map and solve");

        assert_eq!(out.blocked_hints.len(), 2);
        assert_eq!(out.steps.len(), 2);
        assert!(out.steps[0].contains("Take log base runtime("));
        assert!(out.steps[1].starts_with("Conditional solution:"));
        assert!(out.steps[1].contains("base > 0"));
        assert!(out.steps[1].contains("RHS > 0"));
        assert!(matches!(out.solution_set, SolutionSet::Conditional(_)));
    }

    #[test]
    fn visit_log_decision_assumptions_with_iterates_assumptions_in_order() {
        let decision = LogSolveDecision::OkWithAssumptions(vec![
            LogAssumption::PositiveBase,
            LogAssumption::PositiveRhs,
        ]);
        let mut seen = Vec::new();
        visit_log_decision_assumptions_with(&decision, |assumption| seen.push(assumption));
        assert_eq!(
            seen,
            vec![LogAssumption::PositiveBase, LogAssumption::PositiveRhs]
        );
    }

    #[test]
    fn resolve_pow_exponent_log_decision_pipeline_result_maps_terminal_and_unsupported_to_some() {
        let terminal = resolve_pow_exponent_log_decision_pipeline_result::<u8>(
            PowExponentLogDecisionPipelineResult::Terminal {
                solution_set: SolutionSet::AllReals,
                steps: vec![1u8, 2u8],
            },
        )
        .expect("terminal should not error");
        assert_eq!(terminal, Some((SolutionSet::AllReals, vec![1u8, 2u8])));

        let unsupported = resolve_pow_exponent_log_decision_pipeline_result::<u8>(
            PowExponentLogDecisionPipelineResult::UnsupportedSolved {
                solution_set: SolutionSet::Empty,
                steps: vec![9u8],
            },
        )
        .expect("unsupported solved should not error");
        assert_eq!(unsupported, Some((SolutionSet::Empty, vec![9u8])));
    }

    #[test]
    fn resolve_pow_exponent_log_decision_pipeline_result_maps_needs_complex_and_continue() {
        let complex = resolve_pow_exponent_log_decision_pipeline_result::<u8>(
            PowExponentLogDecisionPipelineResult::NeedsComplex {
                message: "complex required",
            },
        )
        .expect_err("needs-complex should return error");
        assert_eq!(complex, "complex required");

        let cont = resolve_pow_exponent_log_decision_pipeline_result::<u8>(
            PowExponentLogDecisionPipelineResult::Continue,
        )
        .expect("continue should not error");
        assert_eq!(cont, None);
    }

    #[test]
    fn ensure_pow_exponent_rhs_without_variable_accepts_rhs_without_target_var() {
        let mut ctx = Context::new();
        let y = ctx.var("y");
        let one = ctx.num(1);
        let rhs = ctx.add(Expr::Add(y, one));
        assert!(ensure_pow_exponent_rhs_without_variable(&ctx, rhs, "x").is_ok());
    }

    #[test]
    fn ensure_pow_exponent_rhs_without_variable_rejects_rhs_with_target_var() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let rhs = ctx.add(Expr::Add(x, one));
        let err = ensure_pow_exponent_rhs_without_variable(&ctx, rhs, "x")
            .expect_err("rhs containing target variable must fail");
        assert_eq!(
            err,
            "Cannot isolate exponential: variable appears on both sides"
        );
    }

    #[test]
    fn log_decision_needs_complex_message_reports_only_needs_complex() {
        assert_eq!(
            log_decision_needs_complex_message(&LogSolveDecision::NeedsComplex("need complex")),
            Some("need complex")
        );
        assert_eq!(
            log_decision_needs_complex_message(&LogSolveDecision::Ok),
            None
        );
    }

    #[test]
    fn log_decision_is_empty_set_detects_only_empty_set_variant() {
        assert!(log_decision_is_empty_set(&LogSolveDecision::EmptySet("no")));
        assert!(!log_decision_is_empty_set(&LogSolveDecision::Ok));
        assert!(!log_decision_is_empty_set(&LogSolveDecision::NeedsComplex(
            "need complex"
        )));
    }

    #[test]
    fn merge_pow_exponent_log_unsupported_pipeline_with_existing_steps_registers_hints() {
        let solved = Some(PowExponentLogUnsupportedPipelineSolved {
            blocked_hints: vec![LogBlockedHintRecord {
                assumption: LogAssumption::PositiveBase,
                expr_id: ExprId::from_raw(7),
                rule: "Take log of both sides",
                suggestion: "use `semantics set domain assume`",
            }],
            solution_set: SolutionSet::AllReals,
            steps: vec!["unsupported".to_string()],
        });

        let mut registered = Vec::new();
        let merged = merge_pow_exponent_log_unsupported_pipeline_with_existing_steps(
            solved,
            vec!["existing".to_string()],
            |hint| registered.push(hint),
        )
        .expect("must merge unsupported outcome");

        assert_eq!(registered.len(), 1);
        assert_eq!(registered[0].expr_id, ExprId::from_raw(7));
        assert_eq!(
            merged.1,
            vec!["existing".to_string(), "unsupported".to_string()]
        );
    }

    #[test]
    fn finalize_pow_exponent_log_unsupported_pipeline_with_existing_steps_preserves_steps_on_none()
    {
        let mut existing = vec!["existing".to_string()];
        let out = finalize_pow_exponent_log_unsupported_pipeline_with_existing_steps(
            None::<PowExponentLogUnsupportedPipelineSolved<String>>,
            &mut existing,
            |_hint| {},
        );
        assert!(out.is_none());
        assert_eq!(existing, vec!["existing".to_string()]);
    }

    #[test]
    fn finalize_pow_exponent_log_unsupported_pipeline_with_existing_steps_merges_and_registers_hints(
    ) {
        let mut existing = vec!["existing".to_string()];
        let mut registered = Vec::new();
        let out = finalize_pow_exponent_log_unsupported_pipeline_with_existing_steps(
            Some(PowExponentLogUnsupportedPipelineSolved {
                blocked_hints: vec![LogBlockedHintRecord {
                    assumption: LogAssumption::PositiveBase,
                    expr_id: ExprId::from_raw(17),
                    rule: "Take log of both sides",
                    suggestion: "assume positive base",
                }],
                solution_set: SolutionSet::AllReals,
                steps: vec!["unsupported".to_string()],
            }),
            &mut existing,
            |hint| registered.push(hint),
        )
        .expect("must merge unsupported outcome");
        assert_eq!(registered.len(), 1);
        assert_eq!(registered[0].expr_id, ExprId::from_raw(17));
        assert_eq!(
            out.1,
            vec!["existing".to_string(), "unsupported".to_string()]
        );
        assert!(existing.is_empty());
    }

    #[test]
    fn execute_pow_exponent_log_unsupported_pipeline_from_decision_and_finalize_with_existing_steps_with_merges(
    ) {
        let mut ctx = Context::new();
        let exponent = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let decision = LogSolveDecision::Unsupported(
            "Cannot prove base > 0 and RHS > 0 for logarithm",
            vec![LogAssumption::PositiveBase, LogAssumption::PositiveRhs],
        );
        let mut existing = vec!["existing".to_string()];
        let mut registered = Vec::new();
        let out =
            execute_pow_exponent_log_unsupported_pipeline_from_decision_and_finalize_with_existing_steps_with(
                true,
                &mut existing,
                || {
                    plan_pow_exponent_log_unsupported_execution_from_decision_with(
                        &mut ctx,
                        &decision,
                        true,
                        exponent,
                        rhs,
                        "x",
                        exponent,
                        base,
                        rhs,
                        RelOp::Eq,
                        Equation {
                            lhs: exponent,
                            rhs,
                            op: RelOp::Eq,
                        },
                        |_, id| format!("runtime({id})"),
                    )
                },
                |_eq| Some(SolutionSet::Discrete(vec![exponent])),
                |item| item.description,
                |hint| registered.push(hint),
            )
            .expect("unsupported decision should map, execute and merge");

        assert_eq!(registered.len(), 2);
        assert!(matches!(out.0, SolutionSet::Conditional(_)));
        assert!(out.1[0].contains("existing"));
        assert!(existing.is_empty());
    }

    #[test]
    fn guarded_solutions_with_residual_fallback_builds_two_cases() {
        let mut ctx = Context::new();
        let b = ctx.var("b");
        let residual = ctx.var("residual");
        let guard = ConditionSet::single(ConditionPredicate::Positive(b));
        let out = guarded_solutions_with_residual_fallback(guard, SolutionSet::AllReals, residual);
        match out {
            SolutionSet::Conditional(cases) => {
                assert_eq!(cases.len(), 2);
                assert_eq!(
                    cases[0].when,
                    ConditionSet::single(ConditionPredicate::Positive(b))
                );
                assert!(matches!(cases[0].then.solutions, SolutionSet::AllReals));
                assert_eq!(cases[1].when, ConditionSet::empty());
                assert!(
                    matches!(cases[1].then.solutions, SolutionSet::Residual(id) if id == residual)
                );
            }
            other => panic!("expected conditional, got {:?}", other),
        }
    }

    #[test]
    fn guarded_or_residual_returns_conditional_when_guard_and_solutions_exist() {
        let mut ctx = Context::new();
        let b = ctx.var("b");
        let residual = ctx.var("residual");
        let guard = ConditionSet::single(ConditionPredicate::Positive(b));
        let out = guarded_or_residual(Some(guard), Some(SolutionSet::AllReals), residual);
        assert!(matches!(out, SolutionSet::Conditional(_)));
    }

    #[test]
    fn guarded_or_residual_returns_residual_without_guard() {
        let mut ctx = Context::new();
        let residual = ctx.var("residual");
        let out = guarded_or_residual(None, Some(SolutionSet::AllReals), residual);
        assert!(matches!(out, SolutionSet::Residual(id) if id == residual));
    }

    #[test]
    fn guarded_or_residual_returns_residual_without_guarded_solutions() {
        let mut ctx = Context::new();
        let b = ctx.var("b");
        let residual = ctx.var("residual");
        let guard = ConditionSet::single(ConditionPredicate::Positive(b));
        let out = guarded_or_residual(Some(guard), None, residual);
        assert!(matches!(out, SolutionSet::Residual(id) if id == residual));
    }

    #[test]
    fn residual_solution_set_wraps_residual_expression() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let out = residual_solution_set(&mut ctx, x, y, "x");
        assert!(matches!(out, SolutionSet::Residual(_)));
    }

    #[test]
    fn residual_expression_builds_solve_call() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let expr = residual_expression(&mut ctx, x, y, "x");
        assert!(matches!(ctx.get(expr), Expr::Function(_, _)));
    }

    #[test]
    fn classify_power_equals_base_route_zero_base() {
        assert_eq!(
            classify_power_equals_base_route(true, false, true),
            PowerEqualsBaseRoute::ExponentGreaterThanZero
        );
    }

    #[test]
    fn classify_power_equals_base_route_numeric_base() {
        assert_eq!(
            classify_power_equals_base_route(false, true, false),
            PowerEqualsBaseRoute::ExponentEqualsOneNumericBase
        );
    }

    #[test]
    fn classify_power_equals_base_route_symbolic_no_budget() {
        assert_eq!(
            classify_power_equals_base_route(false, false, false),
            PowerEqualsBaseRoute::ExponentEqualsOneNoBranchBudget
        );
    }

    #[test]
    fn classify_power_equals_base_route_symbolic_with_budget() {
        assert_eq!(
            classify_power_equals_base_route(false, false, true),
            PowerEqualsBaseRoute::SymbolicCaseSplit
        );
    }

    #[test]
    fn classify_pow_exponent_shortcut_prefers_power_equals_base_in_eq() {
        let mut ctx = Context::new();
        let rhs_exp = ctx.var("n");
        let out =
            classify_pow_exponent_shortcut(RelOp::Eq, true, Some(rhs_exp), false, true, false);
        assert!(matches!(
            out,
            PowExponentShortcut::PowerEqualsBase(
                PowerEqualsBaseRoute::ExponentEqualsOneNumericBase
            )
        ));
    }

    #[test]
    fn detect_pow_exponent_shortcut_inputs_detects_equal_pow_bases() {
        let mut ctx = Context::new();
        let b = ctx.var("b");
        let n = ctx.var("n");
        let rhs = ctx.add(Expr::Pow(b, n));
        let rhs_expr = ctx.get(rhs).clone();
        let (bases_equal, rhs_pow_base_equal) =
            detect_pow_exponent_shortcut_inputs(rhs, &rhs_expr, |candidate| candidate == b);
        assert!(!bases_equal);
        assert_eq!(rhs_pow_base_equal, Some(n));
    }

    #[test]
    fn shortcut_bases_equivalent_with_short_circuits_on_identical_ids() {
        let mut ctx = Context::new();
        let b = ctx.var("b");
        let mut called = false;
        let out = shortcut_bases_equivalent_with(b, b, |_left, _right| {
            called = true;
            false
        });
        assert!(out);
        assert!(
            !called,
            "nontrivial comparator must not run when ids are equal"
        );
    }

    #[test]
    fn shortcut_bases_equivalent_with_delegates_for_distinct_ids() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let out = shortcut_bases_equivalent_with(a, b, |left, right| left == a && right == b);
        assert!(out);
    }

    #[test]
    fn shortcut_bases_equivalent_by_difference_with_short_circuits_equal_ids() {
        let mut ctx = Context::new();
        let b = ctx.var("b");
        let mut called = false;
        let out = shortcut_bases_equivalent_by_difference_with(
            b,
            b,
            |_left, _right| {
                called = true;
                b
            },
            |expr| expr,
            |_expr| false,
        );
        assert!(out);
        assert!(!called, "difference comparator must not run for equal ids");
    }

    #[test]
    fn shortcut_bases_equivalent_by_difference_with_uses_difference_path() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let zero = ctx.num(0);
        let one = ctx.num(1);
        let mut called = false;
        let out = shortcut_bases_equivalent_by_difference_with(
            a,
            b,
            |left, right| {
                called = true;
                if left == a && right == b {
                    zero
                } else {
                    one
                }
            },
            |expr| expr,
            |expr| expr == zero,
        );
        assert!(out);
        assert!(called);
    }

    #[test]
    fn classify_pow_exponent_base_flags_detects_zero_numeric_and_symbolic() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let x = ctx.var("x");

        assert_eq!(classify_pow_exponent_base_flags(&ctx, zero), (true, true));
        assert_eq!(classify_pow_exponent_base_flags(&ctx, x), (false, false));
    }

    #[test]
    fn detect_pow_exponent_shortcut_inputs_handles_non_pow_rhs() {
        let mut ctx = Context::new();
        let b = ctx.var("b");
        let rhs = ctx.var("y");
        let rhs_expr = ctx.get(rhs).clone();
        let (bases_equal, rhs_pow_base_equal) =
            detect_pow_exponent_shortcut_inputs(rhs, &rhs_expr, |candidate| candidate == b);
        assert!(!bases_equal);
        assert_eq!(rhs_pow_base_equal, None);
    }

    #[test]
    fn classify_pow_exponent_shortcut_allows_equal_pow_bases_for_inequality() {
        let mut ctx = Context::new();
        let rhs_exp = ctx.var("n");
        let out =
            classify_pow_exponent_shortcut(RelOp::Geq, false, Some(rhs_exp), false, false, false);
        assert!(matches!(
            out,
            PowExponentShortcut::EqualPowBases { rhs_exp: exp } if exp == rhs_exp
        ));
    }

    #[test]
    fn classify_pow_exponent_shortcut_returns_none_when_no_shortcut_applies() {
        let out = classify_pow_exponent_shortcut(RelOp::Eq, false, None, false, false, false);
        assert_eq!(out, PowExponentShortcut::None);
    }

    #[test]
    fn resolve_pow_exponent_shortcut_maps_zero_base_route_to_gt_zero() {
        let mut ctx = Context::new();
        let base = ctx.num(0);
        let out = resolve_pow_exponent_shortcut(
            &mut ctx,
            PowExponentShortcut::PowerEqualsBase(PowerEqualsBaseRoute::ExponentGreaterThanZero),
            base,
            RelOp::Eq,
        );
        match out {
            PowExponentShortcutResolution::IsolateExponent { rhs, op } => {
                assert_eq!(op, RelOp::Gt);
                assert!(matches!(
                    ctx.get(rhs),
                    Expr::Number(n) if *n == num_rational::BigRational::from_integer(0.into())
                ));
            }
            other => panic!("expected isolate route, got {:?}", other),
        }
    }

    #[test]
    fn resolve_pow_exponent_shortcut_maps_equal_pow_bases_to_rhs_exponent() {
        let mut ctx = Context::new();
        let base = ctx.var("b");
        let rhs_exp = ctx.var("n");
        let out = resolve_pow_exponent_shortcut(
            &mut ctx,
            PowExponentShortcut::EqualPowBases { rhs_exp },
            base,
            RelOp::Leq,
        );
        match out {
            PowExponentShortcutResolution::IsolateExponent { rhs, op } => {
                assert_eq!(rhs, rhs_exp);
                assert_eq!(op, RelOp::Leq);
            }
            other => panic!("expected isolate route, got {:?}", other),
        }
    }

    #[test]
    fn resolve_pow_exponent_shortcut_maps_symbolic_case_split_to_terminal_set() {
        let mut ctx = Context::new();
        let base = ctx.var("a");
        let out = resolve_pow_exponent_shortcut(
            &mut ctx,
            PowExponentShortcut::PowerEqualsBase(PowerEqualsBaseRoute::SymbolicCaseSplit),
            base,
            RelOp::Eq,
        );
        assert!(matches!(
            out,
            PowExponentShortcutResolution::ReturnSolutionSet(SolutionSet::Conditional(_))
        ));
    }

    #[test]
    fn plan_pow_exponent_shortcut_action_keeps_shortcut_for_isolation_route() {
        let mut ctx = Context::new();
        let base = ctx.num(0);
        let out = plan_pow_exponent_shortcut_action(
            &mut ctx,
            PowExponentShortcut::PowerEqualsBase(PowerEqualsBaseRoute::ExponentGreaterThanZero),
            base,
            RelOp::Eq,
        );

        assert!(matches!(
            out,
            PowExponentShortcutAction::IsolateExponent {
                shortcut: PowExponentShortcut::PowerEqualsBase(
                    PowerEqualsBaseRoute::ExponentGreaterThanZero
                ),
                ..
            }
        ));
    }

    #[test]
    fn plan_pow_exponent_shortcut_action_keeps_shortcut_for_solution_set_route() {
        let mut ctx = Context::new();
        let base = ctx.var("a");
        let out = plan_pow_exponent_shortcut_action(
            &mut ctx,
            PowExponentShortcut::PowerEqualsBase(PowerEqualsBaseRoute::SymbolicCaseSplit),
            base,
            RelOp::Eq,
        );

        assert!(matches!(
            out,
            PowExponentShortcutAction::ReturnSolutionSet {
                shortcut: PowExponentShortcut::PowerEqualsBase(
                    PowerEqualsBaseRoute::SymbolicCaseSplit
                ),
                solutions: SolutionSet::Conditional(_)
            }
        ));
    }

    #[test]
    fn plan_pow_exponent_shortcut_action_for_rhs_detects_equal_pow_bases() {
        let mut ctx = Context::new();
        let base = ctx.var("b");
        let rhs_exp = ctx.var("n");
        let rhs = ctx.add(Expr::Pow(base, rhs_exp));

        let out = plan_pow_exponent_shortcut_action_for_rhs(
            &mut ctx,
            rhs,
            base,
            RelOp::Eq,
            false,
            false,
            false,
            |candidate| candidate == base,
        );

        assert!(matches!(
            out,
            PowExponentShortcutAction::IsolateExponent {
                shortcut: PowExponentShortcut::EqualPowBases { .. },
                ..
            }
        ));
    }

    #[test]
    fn plan_pow_exponent_shortcut_action_for_rhs_handles_non_matching_rhs() {
        let mut ctx = Context::new();
        let base = ctx.var("b");
        let rhs = ctx.var("y");

        let out = plan_pow_exponent_shortcut_action_for_rhs(
            &mut ctx,
            rhs,
            base,
            RelOp::Eq,
            false,
            false,
            false,
            |candidate| candidate == base,
        );

        assert_eq!(out, PowExponentShortcutAction::Continue);
    }

    #[test]
    fn plan_pow_exponent_shortcut_action_from_inputs_maps_to_symbolic_split() {
        let mut ctx = Context::new();
        let base = ctx.var("a");

        let out = plan_pow_exponent_shortcut_action_from_inputs(
            &mut ctx,
            base,
            RelOp::Eq,
            true,
            None,
            false,
            false,
            true,
        );

        assert!(matches!(
            out,
            PowExponentShortcutAction::ReturnSolutionSet {
                shortcut: PowExponentShortcut::PowerEqualsBase(
                    PowerEqualsBaseRoute::SymbolicCaseSplit
                ),
                solutions: SolutionSet::Conditional(_)
            }
        ));
    }

    #[test]
    fn plan_pow_exponent_shortcut_action_detecting_with_maps_equal_pow_bases() {
        let mut ctx = Context::new();
        let base = ctx.var("b");
        let rhs_exp = ctx.var("n");
        let rhs = ctx.add(Expr::Pow(base, rhs_exp));
        let rhs_expr = ctx.get(rhs).clone();

        let out = plan_pow_exponent_shortcut_action_detecting_with(
            &mut ctx,
            rhs,
            &rhs_expr,
            base,
            RelOp::Eq,
            false,
            false,
            false,
            |candidate| candidate == base,
        );

        assert!(matches!(
            out,
            PowExponentShortcutAction::IsolateExponent {
                shortcut: PowExponentShortcut::EqualPowBases { .. },
                rhs,
                op: RelOp::Eq
            } if rhs == rhs_exp
        ));
    }

    #[test]
    fn execute_pow_exponent_shortcut_with_maps_equal_pow_bases() {
        let mut context = Context::new();
        let base = context.var("b");
        let exponent = context.var("x");
        let rhs_exp = context.var("n");
        let rhs = context.add(Expr::Pow(base, rhs_exp));
        let context_cell = std::cell::RefCell::new(context);

        let action = execute_pow_exponent_shortcut_with(
            exponent,
            base,
            rhs,
            RelOp::Eq,
            "x",
            false,
            false,
            false,
            |id| {
                let context_ref = context_cell.borrow();
                context_ref.get(id).clone()
            },
            |inner_base,
             inner_op,
             bases_equal,
             rhs_pow_base_equal,
             inner_base_is_zero,
             inner_base_is_numeric,
             inner_can_branch| {
                let mut context_ref = context_cell.borrow_mut();
                plan_pow_exponent_shortcut_action_from_inputs(
                    &mut context_ref,
                    inner_base,
                    inner_op,
                    bases_equal,
                    rhs_pow_base_equal,
                    inner_base_is_zero,
                    inner_base_is_numeric,
                    inner_can_branch,
                )
            },
            |left, right| left == right,
            |id| format!("{}", id),
        );

        match action {
            PowExponentShortcutEngineAction::IsolateExponent { rhs, op, items } => {
                assert_eq!(rhs, rhs_exp);
                assert_eq!(op, RelOp::Eq);
                assert_eq!(items.len(), 1);
                assert!(items[0].description.contains("equal exponents"));
            }
            other => panic!("expected isolate-exponent shortcut, got {:?}", other),
        }
    }

    #[test]
    fn build_pow_exponent_shortcut_execution_plan_maps_equal_pow_bases() {
        let mut ctx = Context::new();
        let rhs_exp = ctx.var("n");
        let action = PowExponentShortcutAction::IsolateExponent {
            shortcut: PowExponentShortcut::EqualPowBases { rhs_exp },
            rhs: rhs_exp,
            op: RelOp::Eq,
        };

        let plan = build_pow_exponent_shortcut_execution_plan(action);
        assert!(matches!(
            plan,
            PowExponentShortcutExecutionPlan::IsolateExponent {
                narrative: PowExponentShortcutNarrative::EqualPowBases,
                rhs_exponent: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn build_pow_exponent_shortcut_execution_plan_maps_symbolic_case_split() {
        let action = PowExponentShortcutAction::ReturnSolutionSet {
            shortcut: PowExponentShortcut::PowerEqualsBase(PowerEqualsBaseRoute::SymbolicCaseSplit),
            solutions: SolutionSet::AllReals,
        };

        let plan = build_pow_exponent_shortcut_execution_plan(action);
        assert!(matches!(
            plan,
            PowExponentShortcutExecutionPlan::ReturnSolutionSet {
                narrative: PowExponentShortcutNarrative::SymbolicBaseCaseSplit,
                ..
            }
        ));
    }

    #[test]
    fn map_pow_exponent_shortcut_with_builds_isolate_step_payload() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let n = ctx.var("n");
        let out = map_pow_exponent_shortcut_with(
            PowExponentShortcutExecutionPlan::IsolateExponent {
                rhs: n,
                op: RelOp::Eq,
                narrative: PowExponentShortcutNarrative::EqualPowBases,
                rhs_exponent: Some(n),
            },
            x,
            base,
            rhs,
            RelOp::Eq,
            "x",
            |_| "expr".to_string(),
        );

        match out {
            PowExponentShortcutEngineAction::IsolateExponent { rhs, op, items } => {
                assert_eq!(rhs, n);
                assert_eq!(op, RelOp::Eq);
                assert_eq!(items.len(), 1);
                assert_eq!(items[0].equation.lhs, x);
                assert_eq!(items[0].equation.rhs, n);
                assert!(items[0].description.contains("expr"));
            }
            other => panic!("expected isolate action, got {:?}", other),
        }
    }

    #[test]
    fn map_pow_exponent_shortcut_with_builds_terminal_solution_step() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let out = map_pow_exponent_shortcut_with(
            PowExponentShortcutExecutionPlan::ReturnSolutionSet {
                solutions: SolutionSet::Discrete(vec![x]),
                narrative: PowExponentShortcutNarrative::NumericBaseExponentOne,
            },
            x,
            base,
            rhs,
            RelOp::Eq,
            "x",
            |_| "expr".to_string(),
        );
        match out {
            PowExponentShortcutEngineAction::ReturnSolutionSet { items, .. } => {
                assert_eq!(items.len(), 1);
                assert_eq!(items[0].equation.lhs, x);
                assert_eq!(items[0].equation.rhs, base);
                assert_eq!(items[0].equation.op, RelOp::Eq);
                assert!(items[0].description.contains("expr"));
            }
            other => panic!("expected terminal action, got {:?}", other),
        }
    }

    #[test]
    fn pow_exponent_shortcut_message_formats_equal_pow_bases() {
        let msg = pow_exponent_shortcut_message(
            PowExponentShortcutNarrative::EqualPowBases,
            "x",
            "b",
            "rhs",
            Some("n"),
        );
        assert!(msg.contains("b^x = b^n"));
        assert!(msg.contains("x = n"));
    }

    #[test]
    fn classify_pow_base_isolation_route_impossible_even_negative_rhs() {
        assert_eq!(
            classify_pow_base_isolation_route(true, true, false),
            PowBaseIsolationRoute::EvenExponentNegativeRhsImpossible
        );
    }

    #[test]
    fn classify_pow_base_isolation_route_even_uses_abs_root() {
        assert_eq!(
            classify_pow_base_isolation_route(true, false, true),
            PowBaseIsolationRoute::EvenExponentUseAbsRoot
        );
    }

    #[test]
    fn classify_pow_base_isolation_route_general_tracks_negative_exponent() {
        assert_eq!(
            classify_pow_base_isolation_route(false, false, true),
            PowBaseIsolationRoute::GeneralRoot {
                flip_inequality_for_negative_exponent: true
            }
        );
    }

    #[test]
    fn plan_pow_base_isolation_returns_terminal_for_impossible_even_negative_rhs() {
        let mut ctx = Context::new();
        let b = ctx.var("b");
        let e = ctx.num(2);
        let rhs = ctx.num(-1);
        let plan = plan_pow_base_isolation(&mut ctx, b, e, rhs, RelOp::Eq, true, true, false);
        assert!(matches!(
            plan,
            PowBaseIsolationPlan::ReturnSolutionSet {
                route: PowBaseIsolationRoute::EvenExponentNegativeRhsImpossible,
                solutions: SolutionSet::Empty,
                ..
            }
        ));
    }

    #[test]
    fn plan_pow_base_isolation_builds_abs_root_for_even_exponent() {
        let mut ctx = Context::new();
        let b = ctx.var("b");
        let e = ctx.num(2);
        let rhs = ctx.var("r");
        let plan = plan_pow_base_isolation(&mut ctx, b, e, rhs, RelOp::Eq, true, false, false);
        match plan {
            PowBaseIsolationPlan::IsolateBase {
                route,
                equation,
                use_abs_root,
                ..
            } => {
                assert_eq!(route, PowBaseIsolationRoute::EvenExponentUseAbsRoot);
                assert!(use_abs_root);
                assert!(matches!(ctx.get(equation.lhs), Expr::Function(_, _)));
            }
            other => panic!("expected isolate-base plan, got {:?}", other),
        }
    }

    #[test]
    fn plan_pow_base_isolation_flips_inequality_for_negative_exponent() {
        let mut ctx = Context::new();
        let b = ctx.var("b");
        let e = ctx.num(-3);
        let rhs = ctx.var("r");
        let plan = plan_pow_base_isolation(&mut ctx, b, e, rhs, RelOp::Lt, false, false, true);
        match plan {
            PowBaseIsolationPlan::IsolateBase { op, .. } => assert_eq!(op, RelOp::Gt),
            other => panic!("expected isolate-base plan, got {:?}", other),
        }
    }

    #[test]
    fn map_pow_base_isolation_plan_with_builds_terminal_step_payload() {
        let mut ctx = Context::new();
        let base = ctx.var("x");
        let exponent = ctx.num(2);
        let rhs = ctx.num(-1);
        let plan =
            plan_pow_base_isolation(&mut ctx, base, exponent, rhs, RelOp::Eq, true, true, false);
        let action = map_pow_base_isolation_plan_with(plan, base, exponent, rhs, RelOp::Eq, |_| {
            "expr".to_string()
        });
        match action {
            PowBaseIsolationEngineAction::ReturnSolutionSet { items, .. } => {
                assert_eq!(items.len(), 1);
                assert!(items[0]
                    .description
                    .contains("Even power cannot be negative"));
            }
            other => panic!("expected terminal action, got {:?}", other),
        }
    }

    #[test]
    fn map_pow_base_isolation_plan_with_builds_isolate_step_payload() {
        let mut ctx = Context::new();
        let base = ctx.var("x");
        let exponent = ctx.num(2);
        let rhs = ctx.num(9);
        let plan =
            plan_pow_base_isolation(&mut ctx, base, exponent, rhs, RelOp::Eq, true, false, false);
        let action = map_pow_base_isolation_plan_with(plan, base, exponent, rhs, RelOp::Eq, |_| {
            "2".to_string()
        });
        match action {
            PowBaseIsolationEngineAction::IsolateBase { items, .. } => {
                assert_eq!(items.len(), 1);
                assert_eq!(
                    items[0].description,
                    "Take 2-th root of both sides (even root implies absolute value)"
                );
            }
            other => panic!("expected isolate action, got {:?}", other),
        }
    }

    #[test]
    fn build_pow_base_isolation_action_with_computes_predicates_and_maps_payload() {
        let mut ctx = Context::new();
        let base = ctx.var("x");
        let exponent = ctx.num(2);
        let rhs = ctx.num(-1);
        let action = build_pow_base_isolation_action_with(
            &mut ctx,
            base,
            exponent,
            rhs,
            RelOp::Eq,
            |_, _| "expr".to_string(),
        );

        match action {
            PowBaseIsolationEngineAction::ReturnSolutionSet { items, solutions } => {
                assert_eq!(items.len(), 1);
                assert!(items[0]
                    .description
                    .contains("Even power cannot be negative"));
                assert!(matches!(solutions, SolutionSet::Empty));
            }
            other => panic!("expected terminal action, got {:?}", other),
        }
    }

    #[test]
    fn build_pow_base_isolation_action_with_uses_renderer_for_payload() {
        let mut ctx = Context::new();
        let base = ctx.var("x");
        let exponent = ctx.num(2);
        let rhs = ctx.num(-1);
        let action = build_pow_base_isolation_action_with(
            &mut ctx,
            base,
            exponent,
            rhs,
            RelOp::Eq,
            |_, id| format!("runtime({id})"),
        );

        match action {
            PowBaseIsolationEngineAction::ReturnSolutionSet { items, solutions } => {
                assert_eq!(items.len(), 1);
                assert!(items[0]
                    .description
                    .contains("Even power cannot be negative"));
                assert!(items[0].description.contains("runtime("));
                assert!(matches!(solutions, SolutionSet::Empty));
            }
            other => panic!("expected terminal action, got {:?}", other),
        }
    }

    #[test]
    fn solve_pow_exponent_shortcut_action_with_invokes_isolate_once() {
        let mut ctx = Context::new();
        let rhs = ctx.var("n");
        let action = PowExponentShortcutEngineAction::IsolateExponent {
            rhs,
            op: RelOp::Eq,
            items: vec![],
        };

        let mut calls = 0usize;
        let solved = solve_pow_exponent_shortcut_action_with(action, |target_rhs, target_op| {
            calls += 1;
            Ok::<_, ()>((target_rhs, target_op))
        })
        .expect("solve should succeed");

        assert_eq!(calls, 1);
        match solved {
            PowExponentShortcutSolved::Isolated((target_rhs, target_op)) => {
                assert_eq!(target_rhs, rhs);
                assert_eq!(target_op, RelOp::Eq);
            }
            other => panic!("expected isolated shortcut solve, got {:?}", other),
        }
    }

    #[test]
    fn solve_pow_exponent_shortcut_action_with_returns_solution_set_without_isolate() {
        let action = PowExponentShortcutEngineAction::ReturnSolutionSet {
            solutions: SolutionSet::AllReals,
            items: vec![],
        };

        let mut calls = 0usize;
        let solved = solve_pow_exponent_shortcut_action_with(action, |_rhs, _op| {
            calls += 1;
            Ok::<_, ()>(())
        })
        .expect("solve should succeed");

        assert_eq!(calls, 0);
        assert!(matches!(
            solved,
            PowExponentShortcutSolved::ReturnedSolutionSet(SolutionSet::AllReals)
        ));
    }

    #[test]
    fn solve_pow_exponent_shortcut_pipeline_with_item_prepends_item_for_isolated_branch() {
        let mut ctx = Context::new();
        let rhs = ctx.var("n");
        let x = ctx.var("x");
        let action = PowExponentShortcutEngineAction::IsolateExponent {
            rhs,
            op: RelOp::Eq,
            items: vec![PowExponentShortcutExecutionItem {
                equation: Equation {
                    lhs: x,
                    rhs,
                    op: RelOp::Eq,
                },
                description: "shortcut-step".to_string(),
            }],
        };

        let mut calls = 0usize;
        let solved = solve_pow_exponent_shortcut_pipeline_with_item(
            action,
            true,
            |target_rhs, _op| {
                calls += 1;
                Ok::<_, ()>((
                    SolutionSet::Discrete(vec![target_rhs]),
                    vec!["substep".to_string()],
                ))
            },
            |item| item.description,
        )
        .expect("pipeline should solve");

        assert_eq!(calls, 1);
        match solved {
            PowExponentShortcutPipelineSolved::Isolated {
                solution_set,
                steps,
            } => {
                assert!(matches!(solution_set, SolutionSet::Discrete(_)));
                assert_eq!(
                    steps,
                    vec!["shortcut-step".to_string(), "substep".to_string()]
                );
            }
            other => panic!(
                "expected isolated shortcut pipeline result, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn solve_pow_exponent_shortcut_pipeline_with_item_omits_item_for_terminal_branch_when_disabled()
    {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let action = PowExponentShortcutEngineAction::ReturnSolutionSet {
            solutions: SolutionSet::Empty,
            items: vec![PowExponentShortcutExecutionItem {
                equation: Equation {
                    lhs: x,
                    rhs: x,
                    op: RelOp::Eq,
                },
                description: "terminal".to_string(),
            }],
        };

        let mut calls = 0usize;
        let solved = solve_pow_exponent_shortcut_pipeline_with_item(
            action,
            false,
            |_rhs, _op| {
                calls += 1;
                Ok::<_, ()>((SolutionSet::AllReals, vec!["unexpected".to_string()]))
            },
            |item| item.description,
        )
        .expect("pipeline should solve");

        assert_eq!(calls, 0);
        match solved {
            PowExponentShortcutPipelineSolved::ReturnedSolutionSet {
                solution_set,
                steps,
            } => {
                assert!(matches!(solution_set, SolutionSet::Empty));
                assert!(steps.is_empty());
            }
            other => panic!(
                "expected terminal shortcut pipeline result, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn merge_pow_exponent_shortcut_pipeline_with_existing_steps_merges_orders() {
        let isolated = PowExponentShortcutPipelineSolved::Isolated {
            solution_set: SolutionSet::AllReals,
            steps: vec!["pipeline".to_string()],
        };
        let merged_isolated = merge_pow_exponent_shortcut_pipeline_with_existing_steps(
            isolated,
            vec!["existing".to_string()],
        )
        .expect("isolated must produce merged output");
        assert_eq!(
            merged_isolated.1,
            vec!["pipeline".to_string(), "existing".to_string()]
        );

        let returned = PowExponentShortcutPipelineSolved::ReturnedSolutionSet {
            solution_set: SolutionSet::Empty,
            steps: vec!["pipeline".to_string()],
        };
        let merged_returned = merge_pow_exponent_shortcut_pipeline_with_existing_steps(
            returned,
            vec!["existing".to_string()],
        )
        .expect("terminal must produce merged output");
        assert_eq!(
            merged_returned.1,
            vec!["existing".to_string(), "pipeline".to_string()]
        );
    }

    #[test]
    fn finalize_pow_exponent_shortcut_pipeline_with_existing_steps_preserves_steps_on_continue() {
        let mut existing = vec!["existing".to_string()];
        let out = finalize_pow_exponent_shortcut_pipeline_with_existing_steps(
            PowExponentShortcutPipelineSolved::Continue,
            &mut existing,
        );
        assert!(out.is_none());
        assert_eq!(existing, vec!["existing".to_string()]);
    }

    #[test]
    fn finalize_pow_exponent_shortcut_pipeline_with_existing_steps_merges_isolated() {
        let mut existing = vec!["existing".to_string()];
        let out = finalize_pow_exponent_shortcut_pipeline_with_existing_steps(
            PowExponentShortcutPipelineSolved::Isolated {
                solution_set: SolutionSet::Empty,
                steps: vec!["pipeline".to_string()],
            },
            &mut existing,
        )
        .expect("isolated shortcut must merge");
        assert!(matches!(out.0, SolutionSet::Empty));
        assert_eq!(out.1, vec!["pipeline".to_string(), "existing".to_string()]);
        assert!(existing.is_empty());
    }

    #[test]
    fn execute_pow_exponent_shortcut_pipeline_with_item_with_routes_isolated_branch() {
        let mut ctx = Context::new();
        let base = ctx.var("a");
        let exponent_lhs = ctx.var("x");
        let rhs = ctx.var("n");
        let rhs_exp = ctx.var("k");

        let mut solve_calls = 0usize;
        let solved = execute_pow_exponent_shortcut_pipeline_with_item_with(
            exponent_lhs,
            base,
            rhs,
            RelOp::Eq,
            "x",
            false,
            false,
            true,
            true,
            |expr| ctx.get(expr).clone(),
            |_base, _op, _bases_equal, _rhs_pow_base_equal, _is_zero, _is_numeric, _can_branch| {
                PowExponentShortcutAction::IsolateExponent {
                    shortcut: PowExponentShortcut::EqualPowBases { rhs_exp },
                    rhs,
                    op: RelOp::Eq,
                }
            },
            |_lhs, _rhs| true,
            |_expr| "render".to_string(),
            |target_rhs, _target_op| {
                solve_calls += 1;
                Ok::<_, ()>((
                    SolutionSet::Discrete(vec![target_rhs]),
                    vec!["substep".to_string()],
                ))
            },
            |item| item.description,
        )
        .expect("pipeline should solve");

        assert_eq!(solve_calls, 1);
        match solved {
            PowExponentShortcutPipelineSolved::Isolated {
                solution_set,
                steps,
            } => {
                assert!(matches!(solution_set, SolutionSet::Discrete(_)));
                assert_eq!(steps.len(), 2);
                assert_eq!(steps[1], "substep");
            }
            other => panic!(
                "expected isolated shortcut pipeline result, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn execute_pow_exponent_shortcut_pipeline_with_item_with_skips_solver_for_terminal_branch() {
        let mut ctx = Context::new();
        let base = ctx.var("a");
        let exponent_lhs = ctx.var("x");
        let rhs = ctx.var("n");

        let mut solve_calls = 0usize;
        let solved = execute_pow_exponent_shortcut_pipeline_with_item_with(
            exponent_lhs,
            base,
            rhs,
            RelOp::Eq,
            "x",
            false,
            true,
            true,
            false,
            |expr| ctx.get(expr).clone(),
            |_base, _op, _bases_equal, _rhs_pow_base_equal, _is_zero, _is_numeric, _can_branch| {
                PowExponentShortcutAction::ReturnSolutionSet {
                    shortcut: PowExponentShortcut::PowerEqualsBase(
                        PowerEqualsBaseRoute::SymbolicCaseSplit,
                    ),
                    solutions: SolutionSet::AllReals,
                }
            },
            |_lhs, _rhs| false,
            |_expr| "render".to_string(),
            |_target_rhs, _target_op| {
                solve_calls += 1;
                Ok::<_, ()>((SolutionSet::Empty, vec!["unexpected".to_string()]))
            },
            |item| item.description,
        )
        .expect("pipeline should solve");

        assert_eq!(solve_calls, 0);
        match solved {
            PowExponentShortcutPipelineSolved::ReturnedSolutionSet {
                solution_set,
                steps,
            } => {
                assert!(matches!(solution_set, SolutionSet::AllReals));
                assert!(steps.is_empty());
            }
            other => panic!(
                "expected terminal shortcut pipeline result, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn execute_pow_exponent_shortcut_pipeline_with_item_and_finalize_with_existing_steps_with_merges_isolated(
    ) {
        let mut ctx = Context::new();
        let base = ctx.var("a");
        let exponent_lhs = ctx.var("x");
        let rhs = ctx.var("n");
        let rhs_exp = ctx.var("k");
        let mut existing = vec!["existing".to_string()];

        let solved =
            execute_pow_exponent_shortcut_pipeline_with_item_and_finalize_with_existing_steps_with(
                exponent_lhs,
                base,
                rhs,
                RelOp::Eq,
                "x",
                false,
                false,
                true,
                true,
                &mut existing,
                |expr| ctx.get(expr).clone(),
                |_base,
                 _op,
                 _bases_equal,
                 _rhs_pow_base_equal,
                 _is_zero,
                 _is_numeric,
                 _can_branch| {
                    PowExponentShortcutAction::IsolateExponent {
                        shortcut: PowExponentShortcut::EqualPowBases { rhs_exp },
                        rhs,
                        op: RelOp::Eq,
                    }
                },
                |_lhs, _rhs| true,
                |_expr| "render".to_string(),
                |target_rhs, _target_op| {
                    Ok::<_, ()>((
                        SolutionSet::Discrete(vec![target_rhs]),
                        vec!["substep".to_string()],
                    ))
                },
                |item| item.description,
            )
            .expect("pipeline should solve")
            .expect("isolated shortcut should merge");

        assert!(matches!(solved.0, SolutionSet::Discrete(_)));
        assert_eq!(solved.1.len(), 3);
        assert!(solved.1[0].contains("equal bases imply equal exponents"));
        assert_eq!(solved.1[1], "substep");
        assert_eq!(solved.1[2], "existing");
        assert!(existing.is_empty());
    }

    #[test]
    fn execute_pow_exponent_shortcut_pipeline_with_item_and_finalize_with_existing_steps_with_preserves_steps_on_continue(
    ) {
        let mut ctx = Context::new();
        let base = ctx.var("a");
        let exponent_lhs = ctx.var("x");
        let rhs = ctx.var("n");
        let mut existing = vec!["existing".to_string()];

        let solved =
            execute_pow_exponent_shortcut_pipeline_with_item_and_finalize_with_existing_steps_with(
                exponent_lhs,
                base,
                rhs,
                RelOp::Eq,
                "x",
                false,
                false,
                true,
                true,
                &mut existing,
                |expr| ctx.get(expr).clone(),
                |_base,
                 _op,
                 _bases_equal,
                 _rhs_pow_base_equal,
                 _is_zero,
                 _is_numeric,
                 _can_branch| { PowExponentShortcutAction::Continue },
                |_lhs, _rhs| false,
                |_expr| "render".to_string(),
                |_target_rhs, _target_op| {
                    Ok::<_, ()>((SolutionSet::Empty, vec!["unexpected".to_string()]))
                },
                |item| item.description,
            )
            .expect("pipeline should solve");

        assert!(solved.is_none());
        assert_eq!(existing, vec!["existing".to_string()]);
    }

    #[test]
    fn solve_pow_base_isolation_action_with_invokes_isolate_once() {
        let mut ctx = Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.var("r");
        let action = PowBaseIsolationEngineAction::IsolateBase {
            lhs,
            rhs,
            op: RelOp::Lt,
            items: vec![],
        };

        let mut calls = 0usize;
        let solved = solve_pow_base_isolation_action_with(action, |target_lhs, target_rhs, op| {
            calls += 1;
            Ok::<_, ()>((target_lhs, target_rhs, op))
        })
        .expect("solve should succeed");

        assert_eq!(calls, 1);
        match solved {
            PowBaseIsolationSolved::Isolated((target_lhs, target_rhs, target_op)) => {
                assert_eq!(target_lhs, lhs);
                assert_eq!(target_rhs, rhs);
                assert_eq!(target_op, RelOp::Lt);
            }
            other => panic!("expected isolated base solve, got {:?}", other),
        }
    }

    #[test]
    fn solve_pow_base_isolation_action_with_returns_solution_set_without_isolate() {
        let action = PowBaseIsolationEngineAction::ReturnSolutionSet {
            solutions: SolutionSet::Empty,
            items: vec![],
        };

        let mut calls = 0usize;
        let solved = solve_pow_base_isolation_action_with(action, |_lhs, _rhs, _op| {
            calls += 1;
            Ok::<_, ()>(())
        })
        .expect("solve should succeed");

        assert_eq!(calls, 0);
        assert!(matches!(
            solved,
            PowBaseIsolationSolved::ReturnedSolutionSet(SolutionSet::Empty)
        ));
    }

    #[test]
    fn solve_pow_base_isolation_pipeline_with_item_prepends_item_for_isolate_branch() {
        let mut ctx = Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.var("r");
        let item_equation = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };
        let action = PowBaseIsolationEngineAction::IsolateBase {
            lhs,
            rhs,
            op: RelOp::Eq,
            items: vec![PowBaseIsolationExecutionItem {
                equation: item_equation,
                description: "Take root".to_string(),
            }],
        };

        let mut calls = 0usize;
        let solved = solve_pow_base_isolation_pipeline_with_item(
            action,
            true,
            |_lhs, rhs_after, _op| {
                calls += 1;
                Ok::<_, ()>((
                    SolutionSet::Discrete(vec![rhs_after]),
                    vec!["substep".to_string()],
                ))
            },
            |item| item.description,
        )
        .expect("pipeline should solve");

        assert_eq!(calls, 1);
        match solved {
            PowBaseIsolationPipelineSolved::Isolated {
                solution_set,
                steps,
            } => {
                assert!(matches!(solution_set, SolutionSet::Discrete(_)));
                assert_eq!(steps, vec!["Take root".to_string(), "substep".to_string()]);
            }
            other => panic!("expected isolated pipeline result, got {:?}", other),
        }
    }

    #[test]
    fn solve_pow_base_isolation_pipeline_with_item_omits_item_for_terminal_branch_when_disabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let action = PowBaseIsolationEngineAction::ReturnSolutionSet {
            solutions: SolutionSet::Empty,
            items: vec![PowBaseIsolationExecutionItem {
                equation: Equation {
                    lhs: x,
                    rhs: x,
                    op: RelOp::Eq,
                },
                description: "terminal".to_string(),
            }],
        };

        let mut calls = 0usize;
        let solved = solve_pow_base_isolation_pipeline_with_item(
            action,
            false,
            |_lhs, _rhs, _op| {
                calls += 1;
                Ok::<_, ()>((SolutionSet::AllReals, vec!["unexpected".to_string()]))
            },
            |item| item.description,
        )
        .expect("pipeline should solve");

        assert_eq!(calls, 0);
        match solved {
            PowBaseIsolationPipelineSolved::ReturnedSolutionSet {
                solution_set,
                steps,
            } => {
                assert!(matches!(solution_set, SolutionSet::Empty));
                assert!(steps.is_empty());
            }
            other => panic!("expected terminal pipeline result, got {:?}", other),
        }
    }

    #[test]
    fn execute_pow_base_isolation_pipeline_with_item_with_plans_then_solves() {
        let mut ctx = Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.var("r");
        let item_equation = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };
        let mut plan_calls = 0usize;
        let mut solve_calls = 0usize;
        let solved = execute_pow_base_isolation_pipeline_with_item_with(
            true,
            || {
                plan_calls += 1;
                PowBaseIsolationEngineAction::IsolateBase {
                    lhs,
                    rhs,
                    op: RelOp::Eq,
                    items: vec![PowBaseIsolationExecutionItem {
                        equation: item_equation,
                        description: "Take root".to_string(),
                    }],
                }
            },
            |_lhs, _rhs, _op| {
                solve_calls += 1;
                Ok::<_, ()>((SolutionSet::AllReals, vec!["sub".to_string()]))
            },
            |item| item.description,
        )
        .expect("execute wrapper should solve");

        assert_eq!(plan_calls, 1);
        assert_eq!(solve_calls, 1);
        match solved {
            PowBaseIsolationPipelineSolved::Isolated { steps, .. } => {
                assert_eq!(steps, vec!["Take root".to_string(), "sub".to_string()]);
            }
            other => panic!("expected isolated pipeline result, got {:?}", other),
        }
    }

    #[test]
    fn execute_pow_base_isolation_pipeline_with_item_and_merge_with_existing_steps_with_merges() {
        let mut ctx = Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.var("r");
        let merged =
            execute_pow_base_isolation_pipeline_with_item_and_merge_with_existing_steps_with(
                true,
                vec!["existing".to_string()],
                || PowBaseIsolationEngineAction::IsolateBase {
                    lhs,
                    rhs,
                    op: RelOp::Eq,
                    items: vec![PowBaseIsolationExecutionItem {
                        equation: Equation {
                            lhs,
                            rhs,
                            op: RelOp::Eq,
                        },
                        description: "Take root".to_string(),
                    }],
                },
                |_lhs, _rhs, _op| Ok::<_, ()>((SolutionSet::Empty, vec!["sub".to_string()])),
                |item| item.description,
            )
            .expect("execute+merge wrapper should solve");

        assert!(matches!(merged.0, SolutionSet::Empty));
        assert_eq!(
            merged.1,
            vec![
                "Take root".to_string(),
                "sub".to_string(),
                "existing".to_string()
            ]
        );
    }

    #[test]
    fn merge_pow_base_isolation_pipeline_with_existing_steps_merges_orders() {
        let isolated = PowBaseIsolationPipelineSolved::Isolated {
            solution_set: SolutionSet::AllReals,
            steps: vec!["pipeline".to_string()],
        };
        let merged_isolated = merge_pow_base_isolation_pipeline_with_existing_steps(
            isolated,
            vec!["existing".to_string()],
        );
        assert_eq!(
            merged_isolated.1,
            vec!["pipeline".to_string(), "existing".to_string()]
        );

        let returned = PowBaseIsolationPipelineSolved::ReturnedSolutionSet {
            solution_set: SolutionSet::Empty,
            steps: vec!["pipeline".to_string()],
        };
        let merged_returned = merge_pow_base_isolation_pipeline_with_existing_steps(
            returned,
            vec!["existing".to_string()],
        );
        assert_eq!(
            merged_returned.1,
            vec!["existing".to_string(), "pipeline".to_string()]
        );
    }

    #[test]
    fn collect_pow_base_isolation_didactic_steps_returns_single_step() {
        let mut ctx = Context::new();
        let base = ctx.var("x");
        let exponent = ctx.num(2);
        let rhs = ctx.num(-1);
        let plan =
            plan_pow_base_isolation(&mut ctx, base, exponent, rhs, RelOp::Eq, true, true, false);
        let action = map_pow_base_isolation_plan_with(plan, base, exponent, rhs, RelOp::Eq, |_| {
            "expr".to_string()
        });

        let didactic = collect_pow_base_isolation_didactic_steps(&action);
        assert_eq!(didactic.len(), 1);
        assert!(didactic[0]
            .description
            .contains("Even power cannot be negative"));
    }

    #[test]
    fn collect_pow_base_isolation_execution_items_returns_single_item() {
        let mut ctx = Context::new();
        let base = ctx.var("x");
        let exponent = ctx.num(2);
        let rhs = ctx.num(-1);
        let plan =
            plan_pow_base_isolation(&mut ctx, base, exponent, rhs, RelOp::Eq, true, true, false);
        let action = map_pow_base_isolation_plan_with(plan, base, exponent, rhs, RelOp::Eq, |_| {
            "expr".to_string()
        });

        let items = collect_pow_base_isolation_execution_items(&action);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].equation, items[0].equation);
        assert!(items[0]
            .description
            .contains("Even power cannot be negative"));
    }

    #[test]
    fn first_pow_base_isolation_execution_item_returns_single_item() {
        let mut ctx = Context::new();
        let base = ctx.var("x");
        let exponent = ctx.num(2);
        let rhs = ctx.num(-1);
        let plan =
            plan_pow_base_isolation(&mut ctx, base, exponent, rhs, RelOp::Eq, true, true, false);
        let action = map_pow_base_isolation_plan_with(plan, base, exponent, rhs, RelOp::Eq, |_| {
            "expr".to_string()
        });

        let item = first_pow_base_isolation_execution_item(&action)
            .expect("expected one base isolation item");
        assert!(item.description.contains("Even power cannot be negative"));
    }

    #[test]
    fn collect_pow_exponent_shortcut_didactic_steps_handles_continue_and_isolate() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let n = ctx.var("n");

        let continue_steps = collect_pow_exponent_shortcut_didactic_steps(
            &PowExponentShortcutEngineAction::Continue,
        );
        assert!(continue_steps.is_empty());

        let action = map_pow_exponent_shortcut_with(
            PowExponentShortcutExecutionPlan::IsolateExponent {
                rhs: n,
                op: RelOp::Eq,
                narrative: PowExponentShortcutNarrative::EqualPowBases,
                rhs_exponent: Some(n),
            },
            x,
            base,
            rhs,
            RelOp::Eq,
            "x",
            |_| "expr".to_string(),
        );
        let didactic = collect_pow_exponent_shortcut_didactic_steps(&action);
        assert_eq!(didactic.len(), 1);
        assert!(didactic[0].description.contains("expr"));
    }

    #[test]
    fn collect_pow_exponent_shortcut_execution_items_handles_continue_and_isolate() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let n = ctx.var("n");

        let continue_items = collect_pow_exponent_shortcut_execution_items(
            &PowExponentShortcutEngineAction::Continue,
        );
        assert!(continue_items.is_empty());

        let action = map_pow_exponent_shortcut_with(
            PowExponentShortcutExecutionPlan::IsolateExponent {
                rhs: n,
                op: RelOp::Eq,
                narrative: PowExponentShortcutNarrative::EqualPowBases,
                rhs_exponent: Some(n),
            },
            x,
            base,
            rhs,
            RelOp::Eq,
            "x",
            |_| "expr".to_string(),
        );
        let items = collect_pow_exponent_shortcut_execution_items(&action);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].equation, items[0].equation);
        assert!(items[0].description.contains("expr"));
    }

    #[test]
    fn first_pow_exponent_shortcut_execution_item_handles_continue_and_isolate() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let n = ctx.var("n");

        assert!(first_pow_exponent_shortcut_execution_item(
            &PowExponentShortcutEngineAction::Continue
        )
        .is_none());

        let action = map_pow_exponent_shortcut_with(
            PowExponentShortcutExecutionPlan::IsolateExponent {
                rhs: n,
                op: RelOp::Eq,
                narrative: PowExponentShortcutNarrative::EqualPowBases,
                rhs_exponent: Some(n),
            },
            x,
            base,
            rhs,
            RelOp::Eq,
            "x",
            |_| "expr".to_string(),
        );
        let item = first_pow_exponent_shortcut_execution_item(&action)
            .expect("expected one exponent shortcut item");
        assert!(item.description.contains("expr"));
    }

    #[test]
    fn pow_base_isolation_terminal_message_formats_impossible_even_case() {
        let msg = pow_base_isolation_terminal_message(
            PowBaseIsolationRoute::EvenExponentNegativeRhsImpossible,
            "x",
            "=",
            "-1",
        );
        assert_eq!(msg, "Even power cannot be negative (x = -1)");
    }

    #[test]
    fn pow_base_root_isolation_message_formats_abs_root_variant() {
        let msg = pow_base_root_isolation_message("2", true);
        assert_eq!(
            msg,
            "Take 2-th root of both sides (even root implies absolute value)"
        );
    }

    #[test]
    fn arithmetic_isolation_messages_format_expected_text() {
        assert_eq!(
            subtract_both_sides_message("y"),
            "Subtract y from both sides"
        );
        assert_eq!(add_both_sides_message("y"), "Add y to both sides");
        assert_eq!(divide_both_sides_message("k"), "Divide both sides by k");
        assert_eq!(multiply_both_sides_message("k"), "Multiply both sides by k");
        assert_eq!(
            move_and_flip_message("x"),
            "Move x and multiply by -1 (flips inequality)"
        );
        assert_eq!(
            NEGATED_LHS_ISOLATION_MESSAGE,
            "Multiply both sides by -1 (flips inequality)"
        );
        assert_eq!(
            SWAP_SIDES_TO_LHS_MESSAGE,
            "Swap sides to put variable on LHS"
        );
        assert_eq!(end_case_message(1), "--- End of Case 1 ---");
    }

    #[test]
    fn build_negated_lhs_and_swap_side_steps_use_standard_messages() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let eq = Equation {
            lhs: x,
            rhs: y,
            op: RelOp::Eq,
        };

        let neg_step = build_negated_lhs_isolation_step(eq.clone());
        assert_eq!(
            neg_step.description,
            "Multiply both sides by -1 (flips inequality)"
        );
        assert_eq!(neg_step.equation_after, eq);

        let swap_step = build_swap_sides_step(eq.clone());
        assert_eq!(swap_step.description, "Swap sides to put variable on LHS");
        assert_eq!(swap_step.equation_after, eq);
    }

    #[test]
    fn plan_swap_sides_step_flips_inequality_and_attaches_step() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let eq = Equation {
            lhs: x,
            rhs: y,
            op: RelOp::Lt,
        };

        let plan = plan_swap_sides_step(&eq);
        assert_eq!(plan.equation.lhs, y);
        assert_eq!(plan.equation.rhs, x);
        assert_eq!(plan.equation.op, RelOp::Gt);
        assert_eq!(
            plan.items[0].description,
            "Swap sides to put variable on LHS"
        );
        assert_eq!(plan.items[0].equation, plan.equation);
    }

    #[test]
    fn build_solve_tactic_normalization_step_uses_standard_message() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let eq = Equation {
            lhs: x,
            rhs: y,
            op: RelOp::Eq,
        };
        let step = build_solve_tactic_normalization_step(eq.clone());
        assert_eq!(
            step.description,
            "Applied SolveTactic normalization (Assume mode) to enable logarithm isolation"
        );
        assert_eq!(step.equation_after, eq);
    }

    #[test]
    fn plan_solve_tactic_normalization_step_builds_pow_equation_and_step() {
        let mut ctx = Context::new();
        let base = ctx.var("a");
        let exponent = ctx.var("x");
        let rhs = ctx.var("b");

        let plan = plan_solve_tactic_normalization_step(&mut ctx, base, exponent, rhs, RelOp::Eq);

        assert_eq!(plan.items[0].equation, plan.equation);
        assert_eq!(
            plan.items[0].description,
            "Applied SolveTactic normalization (Assume mode) to enable logarithm isolation"
        );
        assert_eq!(plan.equation.rhs, rhs);
        assert_eq!(plan.equation.op, RelOp::Eq);
    }

    #[test]
    fn solve_solve_tactic_normalization_pipeline_with_item_maps_step_when_enabled() {
        let mut ctx = Context::new();
        let base = ctx.var("a");
        let exponent = ctx.var("x");
        let rhs = ctx.var("b");

        let steps = solve_solve_tactic_normalization_pipeline_with_item(
            &mut ctx,
            base,
            exponent,
            rhs,
            RelOp::Eq,
            true,
            |item| item.description,
        );

        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0], SOLVE_TACTIC_NORMALIZATION_MESSAGE.to_string());
    }

    #[test]
    fn solve_solve_tactic_normalization_pipeline_with_item_omits_step_when_disabled() {
        let mut ctx = Context::new();
        let base = ctx.var("a");
        let exponent = ctx.var("x");
        let rhs = ctx.var("b");

        let steps = solve_solve_tactic_normalization_pipeline_with_item(
            &mut ctx,
            base,
            exponent,
            rhs,
            RelOp::Eq,
            false,
            |_item| -> String { panic!("mapper must not run when items disabled") },
        );

        assert!(steps.is_empty());
    }

    #[test]
    fn execute_pow_exponent_solve_tactic_normalization_with_returns_inputs_when_disabled() {
        let mut ctx = Context::new();
        let base = ctx.var("a");
        let exponent = ctx.var("x");
        let rhs = ctx.var("b");

        let mut cleared = 0usize;
        let (out_base, out_rhs, steps) = execute_pow_exponent_solve_tactic_normalization_with(
            base,
            exponent,
            rhs,
            RelOp::Eq,
            false,
            || {
                cleared += 1;
            },
            |_expr| -> ExprId { panic!("simplify must not run when disabled") },
            |_base, _exp, _rhs, _op| -> Vec<String> {
                panic!("step builder must not run when disabled")
            },
        );

        assert_eq!(cleared, 0);
        assert_eq!(out_base, base);
        assert_eq!(out_rhs, rhs);
        assert!(steps.is_empty());
    }

    #[test]
    fn execute_pow_exponent_solve_tactic_normalization_with_clears_hints_and_skips_steps_when_no_rewrite(
    ) {
        let mut ctx = Context::new();
        let base = ctx.var("a");
        let exponent = ctx.var("x");
        let rhs = ctx.var("b");

        let mut cleared = 0usize;
        let mut simplified = Vec::new();
        let mut build_calls = 0usize;
        let (out_base, out_rhs, steps) = execute_pow_exponent_solve_tactic_normalization_with(
            base,
            exponent,
            rhs,
            RelOp::Eq,
            true,
            || {
                cleared += 1;
            },
            |expr| {
                simplified.push(expr);
                expr
            },
            |_base, _exp, _rhs, _op| -> Vec<String> {
                build_calls += 1;
                vec!["unexpected".to_string()]
            },
        );

        assert_eq!(cleared, 2);
        assert_eq!(simplified, vec![base, rhs]);
        assert_eq!(build_calls, 0);
        assert_eq!(out_base, base);
        assert_eq!(out_rhs, rhs);
        assert!(steps.is_empty());
    }

    #[test]
    fn execute_pow_exponent_solve_tactic_normalization_with_builds_steps_when_rewrite_happens() {
        let mut ctx = Context::new();
        let base = ctx.var("a");
        let exponent = ctx.var("x");
        let rhs = ctx.var("b");
        let rewritten_base = ctx.var("ra");
        let rewritten_rhs = ctx.var("rb");

        let mut cleared = 0usize;
        let mut simplify_calls = 0usize;
        let mut observed_args = None;
        let (out_base, out_rhs, steps) = execute_pow_exponent_solve_tactic_normalization_with(
            base,
            exponent,
            rhs,
            RelOp::Eq,
            true,
            || {
                cleared += 1;
            },
            |expr| {
                simplify_calls += 1;
                if expr == base {
                    rewritten_base
                } else if expr == rhs {
                    rewritten_rhs
                } else {
                    expr
                }
            },
            |sim_base, sim_exp, sim_rhs, rel_op| {
                observed_args = Some((sim_base, sim_exp, sim_rhs, rel_op.clone()));
                vec!["tactic-step".to_string()]
            },
        );

        assert_eq!(cleared, 2);
        assert_eq!(simplify_calls, 2);
        assert_eq!(out_base, rewritten_base);
        assert_eq!(out_rhs, rewritten_rhs);
        assert_eq!(steps, vec!["tactic-step".to_string()]);
        assert_eq!(
            observed_args,
            Some((rewritten_base, exponent, rewritten_rhs, RelOp::Eq))
        );
    }

    #[test]
    fn denominator_case_messages_include_sign_context() {
        assert_eq!(
            denominator_positive_case_message("d"),
            "Case 1: Assume d > 0. Multiply by positive denominator."
        );
        assert_eq!(
            denominator_negative_case_message("d"),
            "Case 2: Assume d < 0. Multiply by negative denominator (flips inequality)."
        );
        assert_eq!(
            isolated_denominator_positive_case_message("x"),
            "Case 1: Assume x > 0. Multiply by x (positive). Inequality direction preserved (flipped from isolation logic)."
        );
        assert_eq!(
            isolated_denominator_negative_case_message("x"),
            "Case 2: Assume x < 0. Multiply by x (negative). Inequality flips."
        );
    }

    #[test]
    fn term_isolation_step_builders_use_expected_messages() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let t = ctx.var("t");
        let eq = Equation {
            lhs: x,
            rhs: t,
            op: RelOp::Eq,
        };

        let add_step = build_add_operand_isolation_step_with(eq.clone(), t, |_| "t".to_string());
        assert_eq!(add_step.description, "Subtract t from both sides");

        let sub_step = build_sub_minuend_isolation_step_with(eq.clone(), t, |_| "t".to_string());
        assert_eq!(sub_step.description, "Add t to both sides");

        let subtrahend_step =
            build_sub_subtrahend_isolation_step_with(eq.clone(), t, |_| "t".to_string());
        assert_eq!(
            subtrahend_step.description,
            "Move t and multiply by -1 (flips inequality)"
        );

        let mul_step = build_mul_factor_isolation_step_with(eq.clone(), t, |_| "t".to_string());
        assert_eq!(mul_step.description, "Divide both sides by t");

        let div_step = build_div_numerator_isolation_step_with(eq.clone(), t, |_| "t".to_string());
        assert_eq!(div_step.description, "Multiply both sides by t");
    }

    #[test]
    fn term_isolation_rewrite_plan_helpers_build_equation_and_step() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");

        let add_plan =
            plan_add_operand_isolation_step_with(&mut ctx, x, y, z, RelOp::Eq, |_| "y".to_string());
        assert_eq!(add_plan.equation.lhs, x);
        assert_eq!(add_plan.items[0].description, "Subtract y from both sides");

        let sub_plan =
            plan_sub_minuend_isolation_step_with(&mut ctx, x, y, z, RelOp::Eq, |_| "y".to_string());
        assert_eq!(sub_plan.equation.lhs, x);
        assert_eq!(sub_plan.items[0].description, "Add y to both sides");

        let subtrahend_plan =
            plan_sub_subtrahend_isolation_step_with(&mut ctx, x, y, z, RelOp::Lt, |_| {
                "x".to_string()
            });
        assert_eq!(subtrahend_plan.equation.lhs, y);
        assert_eq!(subtrahend_plan.equation.op, RelOp::Gt);
        assert_eq!(
            subtrahend_plan.items[0].description,
            "Move x and multiply by -1 (flips inequality)"
        );

        let routed_sub_plan =
            plan_sub_isolation_step_with(&mut ctx, x, y, z, RelOp::Eq, "x", |_| "y".to_string());
        assert_eq!(routed_sub_plan.equation.lhs, x);
        assert_eq!(routed_sub_plan.items[0].description, "Add y to both sides");

        let routed_subtrahend_plan =
            plan_sub_isolation_step_with(&mut ctx, x, y, z, RelOp::Lt, "y", |_| "x".to_string());
        assert_eq!(routed_subtrahend_plan.equation.lhs, y);
        assert_eq!(routed_subtrahend_plan.equation.op, RelOp::Gt);
        assert_eq!(
            routed_subtrahend_plan.items[0].description,
            "Move x and multiply by -1 (flips inequality)"
        );

        let mul_plan =
            plan_mul_factor_isolation_step_with(&mut ctx, x, y, z, RelOp::Eq, false, |_| {
                "y".to_string()
            });
        assert_eq!(mul_plan.equation.lhs, x);
        assert_eq!(mul_plan.items[0].description, "Divide both sides by y");

        let div_plan =
            plan_div_numerator_isolation_step_with(&mut ctx, x, y, z, RelOp::Eq, false, |_| {
                "y".to_string()
            });
        assert_eq!(div_plan.equation.lhs, x);
        assert_eq!(div_plan.items[0].description, "Multiply both sides by y");

        let neg_plan = plan_negated_lhs_isolation_step(&mut ctx, x, z, RelOp::Lt);
        assert_eq!(neg_plan.equation.lhs, x);
        assert_eq!(neg_plan.equation.op, RelOp::Gt);
        assert_eq!(
            neg_plan.items[0].description,
            "Multiply both sides by -1 (flips inequality)"
        );
    }

    #[test]
    fn term_isolation_plan_helpers_build_equation_and_step_with_custom_renderer() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");

        let add_plan =
            plan_add_operand_isolation_step_with(&mut ctx, x, y, z, RelOp::Eq, |_| format!("e{y}"));
        assert_eq!(add_plan.equation.lhs, x);
        assert_eq!(
            add_plan.items[0].description,
            format!("Subtract e{} from both sides", y)
        );

        let sub_plan =
            plan_sub_isolation_step_with(&mut ctx, x, y, z, RelOp::Eq, "x", |_| format!("e{y}"));
        assert_eq!(sub_plan.equation.lhs, x);
        assert_eq!(
            sub_plan.items[0].description,
            format!("Add e{} to both sides", y)
        );

        let mul_plan =
            plan_mul_factor_isolation_step_with(&mut ctx, x, y, z, RelOp::Eq, false, |_| {
                format!("e{y}")
            });
        assert_eq!(mul_plan.equation.lhs, x);
        assert_eq!(
            mul_plan.items[0].description,
            format!("Divide both sides by e{}", y)
        );

        let div_plan =
            plan_div_numerator_isolation_step_with(&mut ctx, x, y, z, RelOp::Eq, false, |_| {
                format!("e{y}")
            });
        assert_eq!(div_plan.equation.lhs, x);
        assert_eq!(
            div_plan.items[0].description,
            format!("Multiply both sides by e{}", y)
        );
    }

    #[test]
    fn collect_term_isolation_rewrite_didactic_steps_returns_single_step() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let plan =
            plan_add_operand_isolation_step_with(&mut ctx, x, y, z, RelOp::Eq, |_| "y".to_string());

        let didactic = collect_term_isolation_rewrite_didactic_steps(&plan);
        assert_eq!(didactic.len(), 1);
        assert_eq!(didactic[0].description, plan.items[0].description);
        assert_eq!(didactic[0].equation_after, plan.items[0].equation);
    }

    #[test]
    fn collect_term_isolation_rewrite_execution_items_returns_single_item() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let plan =
            plan_add_operand_isolation_step_with(&mut ctx, x, y, z, RelOp::Eq, |_| "y".to_string());

        let items = collect_term_isolation_rewrite_execution_items(&plan);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].equation, plan.equation);
        assert_eq!(items[0].description, plan.items[0].description);
    }

    #[test]
    fn first_term_isolation_rewrite_execution_item_returns_single_item() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let plan =
            plan_add_operand_isolation_step_with(&mut ctx, x, y, z, RelOp::Eq, |_| "y".to_string());

        let item = first_term_isolation_rewrite_execution_item(&plan)
            .expect("expected one rewrite execution item");
        assert_eq!(item.equation, plan.equation);
        assert_eq!(item.description, plan.items[0].description);
    }

    #[test]
    fn collect_term_isolation_rewrite_first_step_with_item_returns_single_mapped_step() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let plan =
            plan_add_operand_isolation_step_with(&mut ctx, x, y, z, RelOp::Eq, |_| "y".to_string());

        let steps = collect_term_isolation_rewrite_first_step_with_item(&plan, true, |item| {
            item.description
        });
        assert_eq!(steps, vec![plan.items[0].description.clone()]);
    }

    #[test]
    fn collect_term_isolation_rewrite_first_step_with_item_omits_step_when_disabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let plan =
            plan_add_operand_isolation_step_with(&mut ctx, x, y, z, RelOp::Eq, |_| "y".to_string());

        let steps = collect_term_isolation_rewrite_first_step_with_item(&plan, false, |item| {
            item.description
        });
        assert!(steps.is_empty());
    }

    #[test]
    fn solve_term_isolation_rewrite_with_runs_solver_once_and_preserves_rewrite() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let plan =
            plan_add_operand_isolation_step_with(&mut ctx, x, y, z, RelOp::Eq, |_| "y".to_string());
        let expected = plan.clone();

        let mut calls = 0usize;
        let solved = solve_term_isolation_rewrite_with(plan, |equation| {
            calls += 1;
            Ok::<_, ()>(equation)
        })
        .expect("rewrite solve should succeed");

        assert_eq!(calls, 1);
        assert_eq!(solved.rewrite, expected);
        assert_eq!(solved.solved, expected.equation);
    }

    #[test]
    fn solve_term_isolation_rewrite_pipeline_with_item_forwards_item_and_substeps() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let plan =
            plan_add_operand_isolation_step_with(&mut ctx, x, y, z, RelOp::Eq, |_| "y".to_string());
        let expected_item = plan.items[0].clone();

        let solved = solve_term_isolation_rewrite_pipeline_with_item(
            plan,
            true,
            |equation| {
                Ok::<_, ()>((
                    SolutionSet::Discrete(vec![equation.lhs]),
                    vec!["substep".to_string()],
                ))
            },
            |item| item.description,
        )
        .expect("pipeline solve should succeed");

        assert_eq!(solved.solution_set, SolutionSet::Discrete(vec![x]));
        assert_eq!(
            solved.steps,
            vec![expected_item.description, "substep".to_string()]
        );
    }

    #[test]
    fn solve_term_isolation_rewrite_pipeline_with_item_omits_item_when_disabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let plan =
            plan_add_operand_isolation_step_with(&mut ctx, x, y, z, RelOp::Eq, |_| "y".to_string());

        let solved = solve_term_isolation_rewrite_pipeline_with_item(
            plan,
            false,
            |equation| {
                Ok::<_, ()>((
                    SolutionSet::Discrete(vec![equation.rhs]),
                    vec!["only-substep".to_string()],
                ))
            },
            |item| item.description,
        )
        .expect("pipeline solve should succeed");

        let expected_rhs = ctx.add(Expr::Sub(z, y));
        assert_eq!(
            solved.solution_set,
            SolutionSet::Discrete(vec![expected_rhs])
        );
        assert_eq!(solved.steps, vec!["only-substep".to_string()]);
    }

    #[test]
    fn solve_term_isolation_plan_with_simplifies_rhs_when_enabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let plan =
            plan_add_operand_isolation_step_with(&mut ctx, x, y, z, RelOp::Eq, |_| "y".to_string());
        let expected_item = plan.items[0].description.clone();
        let simplified_rhs = ctx.num(42);
        let raw_rhs = plan.equation.rhs;

        let (solution_set, steps) = solve_term_isolation_plan_with(
            plan,
            true,
            true,
            |_| simplified_rhs,
            |equation| Ok::<_, ()>((SolutionSet::Discrete(vec![equation.rhs]), vec![])),
            |item| item.description,
        )
        .expect("plan solve should succeed");

        assert_eq!(solution_set, SolutionSet::Discrete(vec![simplified_rhs]));
        assert_eq!(steps, vec![expected_item]);
        assert_ne!(raw_rhs, simplified_rhs);
    }

    #[test]
    fn solve_term_isolation_plan_with_keeps_rhs_when_disabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let plan =
            plan_add_operand_isolation_step_with(&mut ctx, x, y, z, RelOp::Eq, |_| "y".to_string());
        let raw_rhs = plan.equation.rhs;
        let unexpected_rhs = ctx.num(7);

        let (solution_set, steps) = solve_term_isolation_plan_with(
            plan,
            false,
            false,
            |_| unexpected_rhs,
            |equation| Ok::<_, ()>((SolutionSet::Discrete(vec![equation.rhs]), vec![])),
            |item| item.description,
        )
        .expect("plan solve should succeed");

        assert_eq!(solution_set, SolutionSet::Discrete(vec![raw_rhs]));
        assert!(steps.is_empty());
    }

    #[test]
    fn execute_term_isolation_plan_with_builds_and_solves_plan() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let plan =
            plan_add_operand_isolation_step_with(&mut ctx, x, y, z, RelOp::Eq, |_| "y".to_string());
        let expected_item = plan.items[0].description.clone();
        let raw_rhs = plan.equation.rhs;
        let simplified_rhs = ctx.num(99);

        let mut plan_calls = 0usize;
        let mut simplify_calls = 0usize;
        let (solution_set, steps) = execute_term_isolation_plan_with(
            || {
                plan_calls += 1;
                plan.clone()
            },
            true,
            true,
            |rhs| {
                assert_eq!(rhs, raw_rhs);
                simplify_calls += 1;
                simplified_rhs
            },
            |equation| Ok::<_, ()>((SolutionSet::Discrete(vec![equation.rhs]), vec![])),
            |item| item.description,
        )
        .expect("execute plan solve should succeed");

        assert_eq!(plan_calls, 1);
        assert_eq!(simplify_calls, 1);
        assert_eq!(solution_set, SolutionSet::Discrete(vec![simplified_rhs]));
        assert_eq!(steps, vec![expected_item]);
    }

    #[test]
    fn execute_term_isolation_plan_with_skips_simplify_when_disabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let plan =
            plan_add_operand_isolation_step_with(&mut ctx, x, y, z, RelOp::Eq, |_| "y".to_string());
        let raw_rhs = plan.equation.rhs;

        let mut simplify_calls = 0usize;
        let (solution_set, steps) = execute_term_isolation_plan_with(
            || plan.clone(),
            false,
            false,
            |_| {
                simplify_calls += 1;
                raw_rhs
            },
            |equation| Ok::<_, ()>((SolutionSet::Discrete(vec![equation.rhs]), vec![])),
            |item| item.description,
        )
        .expect("execute plan solve should succeed");

        assert_eq!(simplify_calls, 0);
        assert_eq!(solution_set, SolutionSet::Discrete(vec![raw_rhs]));
        assert!(steps.is_empty());
    }

    #[test]
    fn execute_term_isolation_plan_with_and_merge_with_existing_steps_with_prepends_pipeline_steps()
    {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let plan =
            plan_add_operand_isolation_step_with(&mut ctx, x, y, z, RelOp::Eq, |_| "y".to_string());
        let expected_item = plan.items[0].description.clone();

        let merged = execute_term_isolation_plan_with_and_merge_with_existing_steps_with(
            || plan.clone(),
            true,
            false,
            vec!["existing".to_string()],
            |rhs| rhs,
            |_equation| Ok::<_, ()>((SolutionSet::AllReals, vec!["substep".to_string()])),
            |item| item.description,
        )
        .expect("execute+merge wrapper should succeed");

        assert!(matches!(merged.0, SolutionSet::AllReals));
        assert_eq!(
            merged.1,
            vec![expected_item, "substep".to_string(), "existing".to_string()]
        );
    }

    #[test]
    fn merge_solved_with_existing_steps_prepend_and_append_preserve_order() {
        let solved = (
            SolutionSet::AllReals,
            vec!["new-1".to_string(), "new-2".to_string()],
        );
        let existing = vec!["old".to_string()];

        let prepended = merge_solved_with_existing_steps_prepend(
            (solved.0.clone(), solved.1.clone()),
            existing.clone(),
        );
        assert_eq!(
            prepended.1,
            vec!["new-1".to_string(), "new-2".to_string(), "old".to_string()]
        );

        let appended = merge_solved_with_existing_steps_append(solved, existing);
        assert_eq!(
            appended.1,
            vec!["old".to_string(), "new-1".to_string(), "new-2".to_string()]
        );
    }

    #[test]
    fn merge_optional_solved_with_existing_steps_append_handles_some_and_none() {
        let some_merged = merge_optional_solved_with_existing_steps_append(
            Some((SolutionSet::AllReals, vec!["new".to_string()])),
            vec!["old".to_string()],
        )
        .expect("some branch should merge");
        assert_eq!(some_merged.0, SolutionSet::AllReals);
        assert_eq!(some_merged.1, vec!["old".to_string(), "new".to_string()]);

        let none_merged = merge_optional_solved_with_existing_steps_append(
            None::<(SolutionSet, Vec<String>)>,
            vec!["old".to_string()],
        );
        assert!(none_merged.is_none());
    }

    #[test]
    fn merge_optional_solved_with_existing_steps_append_mut_handles_some_and_none() {
        let mut some_existing = vec!["old".to_string()];
        let some_merged = merge_optional_solved_with_existing_steps_append_mut(
            Some((SolutionSet::AllReals, vec!["new".to_string()])),
            &mut some_existing,
        )
        .expect("some branch should merge");
        assert_eq!(some_merged.0, SolutionSet::AllReals);
        assert_eq!(some_merged.1, vec!["old".to_string(), "new".to_string()]);
        assert!(some_existing.is_empty());

        let mut none_existing = vec!["old".to_string()];
        let none_merged = merge_optional_solved_with_existing_steps_append_mut(
            None::<(SolutionSet, Vec<String>)>,
            &mut none_existing,
        );
        assert!(none_merged.is_none());
        assert_eq!(none_existing, vec!["old".to_string()]);
    }

    #[test]
    fn solve_negated_lhs_isolation_with_forwards_item_and_substeps() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let one = ctx.num(1);
        let mut last_solved_equation = None;

        let (solution_set, steps) = solve_negated_lhs_isolation_with(
            || plan_negated_lhs_isolation_step(&mut ctx, x, y, RelOp::Lt),
            "x",
            true,
            |equation, _| {
                last_solved_equation = Some(equation.clone());
                Ok::<_, ()>((
                    SolutionSet::Discrete(vec![one]),
                    vec!["substep".to_string()],
                ))
            },
            |item| item.description,
        )
        .expect("negated-lhs solve should succeed");

        let rewritten = last_solved_equation
            .clone()
            .expect("expected rewritten equation to be solved");
        assert_eq!(rewritten.lhs, x);
        assert_eq!(rewritten.op, RelOp::Gt);
        assert!(matches!(ctx.get(rewritten.rhs), Expr::Neg(id) if *id == y));
        assert_eq!(solution_set, SolutionSet::Discrete(vec![one]));
        assert_eq!(
            steps,
            vec![
                "Multiply both sides by -1 (flips inequality)".to_string(),
                "substep".to_string(),
            ]
        );
    }

    #[test]
    fn solve_negated_lhs_isolation_with_omits_item_when_disabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let one = ctx.num(1);
        let (solution_set, steps) = solve_negated_lhs_isolation_with(
            || plan_negated_lhs_isolation_step(&mut ctx, x, y, RelOp::Eq),
            "x",
            false,
            |_equation, _| {
                Ok::<_, ()>((
                    SolutionSet::Discrete(vec![one]),
                    vec!["substep-only".to_string()],
                ))
            },
            |item| item.description,
        )
        .expect("negated-lhs solve should succeed");

        assert_eq!(solution_set, SolutionSet::Discrete(vec![one]));
        assert_eq!(steps, vec!["substep-only".to_string()]);
    }

    #[test]
    fn solve_negated_lhs_isolation_with_and_merge_with_existing_steps_prepends_steps() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let one = ctx.num(1);
        let merged = solve_negated_lhs_isolation_with_and_merge_with_existing_steps(
            || plan_negated_lhs_isolation_step(&mut ctx, x, y, RelOp::Eq),
            "x",
            true,
            vec!["existing".to_string()],
            |_equation, _| {
                Ok::<_, ()>((
                    SolutionSet::Discrete(vec![one]),
                    vec!["substep".to_string()],
                ))
            },
            |item| item.description,
        )
        .expect("negated-lhs execute+merge wrapper should succeed");

        assert!(matches!(merged.0, SolutionSet::Discrete(_)));
        assert_eq!(
            merged.1,
            vec![
                "Multiply both sides by -1 (flips inequality)".to_string(),
                "substep".to_string(),
                "existing".to_string()
            ]
        );
    }

    #[test]
    fn build_division_denominator_sign_split_steps_with_builds_payload() {
        let mut ctx = Context::new();
        let n = ctx.var("n");
        let d = ctx.var("d");
        let r = ctx.var("r");
        let eq_pos = Equation {
            lhs: n,
            rhs: r,
            op: RelOp::Lt,
        };
        let eq_neg = Equation {
            lhs: n,
            rhs: r,
            op: RelOp::Gt,
        };

        let payload = build_division_denominator_sign_split_steps_with(
            eq_pos.clone(),
            eq_neg.clone(),
            d,
            n,
            RelOp::Lt,
            |_| "d".to_string(),
        );

        assert_eq!(
            payload.positive_case.description,
            "Case 1: Assume d > 0. Multiply by positive denominator."
        );
        assert_eq!(payload.positive_case.equation_after, eq_pos);
        assert_eq!(
            payload.negative_case.description,
            "Case 2: Assume d < 0. Multiply by negative denominator (flips inequality)."
        );
        assert_eq!(payload.negative_case.equation_after, eq_neg);
        assert_eq!(payload.case_boundary.description, "--- End of Case 1 ---");
        assert_eq!(
            payload.case_boundary.equation_after,
            Equation {
                lhs: n,
                rhs: r,
                op: RelOp::Lt,
            }
        );
    }

    #[test]
    fn build_isolated_denominator_sign_split_steps_with_builds_payload() {
        let mut ctx = Context::new();
        let d = ctx.var("d");
        let r = ctx.var("r");
        let eq_pos = Equation {
            lhs: d,
            rhs: r,
            op: RelOp::Lt,
        };
        let eq_neg = Equation {
            lhs: d,
            rhs: r,
            op: RelOp::Gt,
        };

        let payload = build_isolated_denominator_sign_split_steps_with(
            eq_pos.clone(),
            eq_neg.clone(),
            d,
            d,
            RelOp::Lt,
            |_| "d".to_string(),
        );

        assert_eq!(
            payload.positive_case.description,
            "Case 1: Assume d > 0. Multiply by d (positive). Inequality direction preserved (flipped from isolation logic)."
        );
        assert_eq!(payload.positive_case.equation_after, eq_pos);
        assert_eq!(
            payload.negative_case.description,
            "Case 2: Assume d < 0. Multiply by d (negative). Inequality flips."
        );
        assert_eq!(payload.negative_case.equation_after, eq_neg);
        assert_eq!(payload.case_boundary.description, "--- End of Case 1 ---");
        assert_eq!(
            payload.case_boundary.equation_after,
            Equation {
                lhs: d,
                rhs: r,
                op: RelOp::Lt,
            }
        );
    }

    #[test]
    fn collect_division_denominator_sign_split_didactic_steps_preserves_expected_order() {
        let mut ctx = Context::new();
        let n = ctx.var("n");
        let d = ctx.var("d");
        let r = ctx.var("r");
        let payload = build_division_denominator_sign_split_steps_with(
            Equation {
                lhs: n,
                rhs: r,
                op: RelOp::Lt,
            },
            Equation {
                lhs: n,
                rhs: r,
                op: RelOp::Gt,
            },
            d,
            n,
            RelOp::Lt,
            |_| "d".to_string(),
        );

        let didactic = collect_division_denominator_sign_split_didactic_steps(&payload);
        assert_eq!(didactic.len(), 3);
        assert_eq!(
            didactic[0].description,
            "Case 1: Assume d > 0. Multiply by positive denominator."
        );
        assert_eq!(
            didactic[1].description,
            "Case 2: Assume d < 0. Multiply by negative denominator (flips inequality)."
        );
        assert_eq!(didactic[2].description, "--- End of Case 1 ---");
    }

    #[test]
    fn build_division_denominator_didactic_steps_with_builds_payload() {
        let mut ctx = Context::new();
        let n = ctx.var("n");
        let d = ctx.var("d");
        let r = ctx.var("r");
        let eq_mul = Equation {
            lhs: n,
            rhs: r,
            op: RelOp::Eq,
        };
        let eq_div = Equation {
            lhs: d,
            rhs: n,
            op: RelOp::Eq,
        };

        let payload = build_division_denominator_didactic_steps_with(
            eq_mul.clone(),
            eq_div.clone(),
            d,
            r,
            |id| {
                if id == d {
                    "d".to_string()
                } else {
                    "r".to_string()
                }
            },
        );

        assert_eq!(payload.items.len(), 2);
        assert_eq!(payload.items[0].description, "Multiply both sides by d");
        assert_eq!(payload.items[0].equation, eq_mul);
        assert_eq!(payload.items[1].description, "Divide both sides by r");
        assert_eq!(payload.items[1].equation, eq_div);
    }

    #[test]
    fn build_division_denominator_didactic_execution_with_rewrites_multiply_rhs() {
        let mut ctx = Context::new();
        let n = ctx.var("n");
        let d = ctx.var("d");
        let r = ctx.var("r");
        let isolated_rhs = ctx.var("isolated");
        let simplified_mul_rhs = ctx.var("simplified");
        let plan = plan_division_denominator_didactic(&mut ctx, n, d, r, isolated_rhs, RelOp::Eq);

        let execution =
            build_division_denominator_didactic_execution_with(plan, simplified_mul_rhs, |id| {
                if id == d {
                    "d".to_string()
                } else {
                    "r".to_string()
                }
            });

        assert_eq!(execution.multiply_equation.lhs, n);
        assert_eq!(execution.multiply_equation.rhs, simplified_mul_rhs);
        assert_eq!(execution.multiply_equation.op, RelOp::Eq);
        assert_eq!(execution.divide_equation.lhs, d);
        assert_eq!(execution.divide_equation.rhs, isolated_rhs);
        assert_eq!(execution.items.len(), 2);
        assert_eq!(execution.items[0].equation, execution.multiply_equation);
        assert_eq!(execution.items[1].equation, execution.divide_equation);
    }

    #[test]
    fn collect_division_denominator_execution_items_preserves_multiply_then_divide_order() {
        let mut ctx = Context::new();
        let n = ctx.var("n");
        let d = ctx.var("d");
        let r = ctx.var("r");
        let isolated_rhs = ctx.var("isolated");
        let simplified_mul_rhs = ctx.var("simplified");
        let plan = plan_division_denominator_didactic(&mut ctx, n, d, r, isolated_rhs, RelOp::Eq);
        let execution =
            build_division_denominator_didactic_execution_with(plan, simplified_mul_rhs, |id| {
                if id == d {
                    "d".to_string()
                } else {
                    "r".to_string()
                }
            });

        let items = collect_division_denominator_execution_items(&execution);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].description, "Multiply both sides by d");
        assert_eq!(items[1].description, "Divide both sides by r");
    }

    #[test]
    fn collect_division_denominator_didactic_execution_items_preserves_order() {
        let mut ctx = Context::new();
        let n = ctx.var("n");
        let d = ctx.var("d");
        let r = ctx.var("r");
        let isolated_rhs = ctx.var("isolated");
        let simplified_mul_rhs = ctx.var("simplified");
        let plan = plan_division_denominator_didactic(&mut ctx, n, d, r, isolated_rhs, RelOp::Eq);
        let execution =
            build_division_denominator_didactic_execution_with(plan, simplified_mul_rhs, |id| {
                if id == d {
                    "d".to_string()
                } else {
                    "r".to_string()
                }
            });

        let items = collect_division_denominator_didactic_execution_items(&execution);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].description, "Multiply both sides by d");
        assert_eq!(items[0].equation, execution.multiply_equation);
        assert_eq!(items[1].description, "Divide both sides by r");
        assert_eq!(items[1].equation, execution.divide_equation);
    }

    #[test]
    fn solve_division_denominator_execution_with_items_passes_items_and_divide_equation() {
        let mut ctx = Context::new();
        let n = ctx.var("n");
        let d = ctx.var("d");
        let r = ctx.var("r");
        let isolated_rhs = ctx.var("isolated");
        let simplified_mul_rhs = ctx.var("simplified");
        let plan = plan_division_denominator_didactic(&mut ctx, n, d, r, isolated_rhs, RelOp::Eq);
        let execution =
            build_division_denominator_didactic_execution_with(plan, simplified_mul_rhs, |id| {
                if id == d {
                    "d".to_string()
                } else {
                    "r".to_string()
                }
            });

        let solved =
            solve_division_denominator_execution_with_items(execution, |items, equation| {
                assert_eq!(items.len(), 2);
                assert_eq!(items[0].description, "Multiply both sides by d");
                assert_eq!(items[1].description, "Divide both sides by r");
                assert_eq!(equation.lhs, d);
                assert_eq!(equation.rhs, isolated_rhs);
                assert_eq!(equation.op, RelOp::Eq);
                Ok::<_, ()>("ok")
            })
            .expect("solve should succeed");
        assert_eq!(solved.solved, "ok");
    }

    #[test]
    fn solve_division_denominator_execution_pipeline_with_items_prepends_didactic_steps() {
        let mut ctx = Context::new();
        let n = ctx.var("n");
        let d = ctx.var("d");
        let r = ctx.var("r");
        let isolated_rhs = ctx.var("isolated");
        let simplified_mul_rhs = ctx.var("simplified");
        let plan = plan_division_denominator_didactic(&mut ctx, n, d, r, isolated_rhs, RelOp::Eq);
        let execution =
            build_division_denominator_didactic_execution_with(plan, simplified_mul_rhs, |id| {
                if id == d {
                    "d".to_string()
                } else {
                    "r".to_string()
                }
            });

        let solved = solve_division_denominator_execution_pipeline_with_items(
            execution,
            |_equation| Ok::<_, ()>((SolutionSet::AllReals, vec!["solve".to_string()])),
            |item| item.description,
        )
        .expect("pipeline should solve");

        assert!(matches!(solved.solution_set, SolutionSet::AllReals));
        assert_eq!(
            solved.steps,
            vec![
                "Multiply both sides by d".to_string(),
                "Divide both sides by r".to_string(),
                "solve".to_string(),
            ]
        );
    }

    #[test]
    fn solve_division_denominator_pipeline_with_optional_items_includes_didactic_when_enabled() {
        let mut ctx = Context::new();
        let n = ctx.var("n");
        let d = ctx.var("d");
        let r = ctx.var("r");
        let isolated_rhs = ctx.var("isolated");
        let plan = plan_division_denominator_didactic(&mut ctx, n, d, r, isolated_rhs, RelOp::Eq);
        let expected_divide = plan.divide_equation.clone();
        let simplified_mul_rhs = plan.multiply_equation.rhs;

        let solved = solve_division_denominator_pipeline_with_optional_items(
            plan,
            true,
            simplified_mul_rhs,
            |id| {
                if id == d {
                    "d".to_string()
                } else {
                    "r".to_string()
                }
            },
            |equation| {
                assert_eq!(equation, &expected_divide);
                Ok::<_, ()>((SolutionSet::AllReals, vec!["solve".to_string()]))
            },
            |item| item.description,
        )
        .expect("pipeline should solve");

        assert!(matches!(solved.solution_set, SolutionSet::AllReals));
        assert_eq!(
            solved.steps,
            vec![
                "Multiply both sides by d".to_string(),
                "Divide both sides by r".to_string(),
                "solve".to_string(),
            ]
        );
    }

    #[test]
    fn solve_division_denominator_pipeline_with_optional_items_skips_didactic_when_disabled() {
        let mut ctx = Context::new();
        let n = ctx.var("n");
        let d = ctx.var("d");
        let r = ctx.var("r");
        let isolated_rhs = ctx.var("isolated");
        let plan = plan_division_denominator_didactic(&mut ctx, n, d, r, isolated_rhs, RelOp::Eq);
        let expected_divide = plan.divide_equation.clone();
        let simplified_mul_rhs = plan.multiply_equation.rhs;

        let solved = solve_division_denominator_pipeline_with_optional_items(
            plan,
            false,
            simplified_mul_rhs,
            |_expr| -> String { panic!("renderer must not run when items are disabled") },
            |equation| {
                assert_eq!(equation, &expected_divide);
                Ok::<_, ()>((SolutionSet::Empty, vec!["direct".to_string()]))
            },
            |_item| -> String { panic!("mapper must not run when items are disabled") },
        )
        .expect("pipeline should solve");

        assert!(matches!(solved.solution_set, SolutionSet::Empty));
        assert_eq!(solved.steps, vec!["direct".to_string()]);
    }

    #[test]
    fn execute_division_denominator_plan_with_optional_items_simplifies_when_items_enabled() {
        let mut ctx = Context::new();
        let n = ctx.var("n");
        let d = ctx.var("d");
        let r = ctx.var("r");
        let isolated_rhs = ctx.var("isolated");
        let simplified_mul_rhs = ctx.var("simplified");
        let solve_tail = ctx.var("tail");
        let plan = plan_division_denominator_didactic(&mut ctx, n, d, r, isolated_rhs, RelOp::Eq);
        let expected_divide = plan.divide_equation.clone();
        let original_multiply_rhs = plan.multiply_equation.rhs;

        let mut simplify_calls = 0usize;
        let solved = execute_division_denominator_plan_with_optional_items(
            plan,
            true,
            |expr| {
                simplify_calls += 1;
                assert_eq!(expr, original_multiply_rhs);
                simplified_mul_rhs
            },
            |id| {
                if id == d {
                    "d".to_string()
                } else {
                    "r".to_string()
                }
            },
            |equation| {
                assert_eq!(equation, &expected_divide);
                Ok::<_, ()>((SolutionSet::AllReals, vec![solve_tail]))
            },
            |item| item.equation.rhs,
        )
        .expect("execute helper should solve");

        assert_eq!(simplify_calls, 1);
        assert!(matches!(solved.0, SolutionSet::AllReals));
        assert_eq!(solved.1, vec![simplified_mul_rhs, isolated_rhs, solve_tail]);
    }

    #[test]
    fn execute_division_denominator_plan_with_optional_items_skips_simplify_when_disabled() {
        let mut ctx = Context::new();
        let n = ctx.var("n");
        let d = ctx.var("d");
        let r = ctx.var("r");
        let isolated_rhs = ctx.var("isolated");
        let plan = plan_division_denominator_didactic(&mut ctx, n, d, r, isolated_rhs, RelOp::Eq);
        let expected_divide = plan.divide_equation.clone();

        let mut simplify_calls = 0usize;
        let solved = execute_division_denominator_plan_with_optional_items(
            plan,
            false,
            |expr| {
                simplify_calls += 1;
                expr
            },
            |_expr| -> String { panic!("renderer must not run when items are disabled") },
            |equation| {
                assert_eq!(equation, &expected_divide);
                Ok::<_, ()>((SolutionSet::Empty, vec!["direct".to_string()]))
            },
            |_item| -> String { panic!("mapper must not run when items are disabled") },
        )
        .expect("execute helper should solve directly");

        assert_eq!(simplify_calls, 0);
        assert!(matches!(solved.0, SolutionSet::Empty));
        assert_eq!(solved.1, vec!["direct".to_string()]);
    }

    #[test]
    fn execute_division_denominator_plan_with_optional_items_and_merge_with_existing_steps_with_prepends(
    ) {
        let mut ctx = Context::new();
        let n = ctx.var("n");
        let d = ctx.var("d");
        let r = ctx.var("r");
        let isolated_rhs = ctx.var("isolated");
        let simplified_mul_rhs = ctx.var("simplified");
        let solve_tail = ctx.var("tail");
        let existing = ctx.var("existing");
        let plan = plan_division_denominator_didactic(&mut ctx, n, d, r, isolated_rhs, RelOp::Eq);
        let expected_divide = plan.divide_equation.clone();
        let original_multiply_rhs = plan.multiply_equation.rhs;

        let solved =
            execute_division_denominator_plan_with_optional_items_and_merge_with_existing_steps_with(
                plan,
                true,
                vec![existing],
                |expr| {
                    assert_eq!(expr, original_multiply_rhs);
                    simplified_mul_rhs
                },
                |id| {
                    if id == d {
                        "d".to_string()
                    } else {
                        "r".to_string()
                    }
                },
                |equation| {
                    assert_eq!(equation, &expected_divide);
                    Ok::<_, ()>((SolutionSet::AllReals, vec![solve_tail]))
                },
                |item| item.equation.rhs,
            )
            .expect("merge execute helper should solve");

        assert!(matches!(solved.0, SolutionSet::AllReals));
        assert_eq!(
            solved.1,
            vec![simplified_mul_rhs, isolated_rhs, solve_tail, existing]
        );
    }

    #[test]
    fn log_isolation_messages_format_expected_text() {
        assert_eq!(
            take_log_base_message("10"),
            "Take log base 10 of both sides"
        );
        assert_eq!(
            take_log_base_under_guard_message("a", "a > 0 and rhs > 0"),
            "Take log base a of both sides (under guard: a > 0 and rhs > 0)"
        );
        assert_eq!(
            SOLVE_TACTIC_NORMALIZATION_MESSAGE,
            "Applied SolveTactic normalization (Assume mode) to enable logarithm isolation"
        );
        assert_eq!(
            conditional_solution_message("base != 1"),
            "Conditional solution: base != 1"
        );
        assert_eq!(residual_message("unsupported"), "unsupported (residual)");
        assert_eq!(
            residual_budget_exhausted_message("unsupported"),
            "unsupported (residual, budget exhausted)"
        );
        assert_eq!(
            eliminate_fractional_exponent_message("3"),
            "Raise both sides to power 3 to eliminate fractional exponent"
        );
        assert_eq!(
            eliminate_rational_exponent_message("3"),
            "Raise both sides to power 3 to eliminate rational exponent"
        );
        assert_eq!(
            variable_canceled_constraint_message("x", "y - 2"),
            "Variable 'x' canceled during simplification. Solution depends on constraint: y - 2 = 0"
        );
    }

    #[test]
    fn build_pow_exponent_log_isolation_step_with_plain_message() {
        let mut ctx = Context::new();
        let exponent = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let step = build_pow_exponent_log_isolation_step_with(
            &mut ctx,
            exponent,
            base,
            rhs,
            RelOp::Eq,
            None,
            |_, _| "a".to_string(),
        );
        assert_eq!(step.description, "Take log base a of both sides");
        assert_eq!(step.equation_after.lhs, exponent);
        assert_eq!(step.equation_after.op, RelOp::Eq);
    }

    #[test]
    fn build_pow_exponent_log_isolation_step_with_guarded_message() {
        let mut ctx = Context::new();
        let exponent = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let step = build_pow_exponent_log_isolation_step_with(
            &mut ctx,
            exponent,
            base,
            rhs,
            RelOp::Eq,
            Some("a > 0"),
            |_, _| "a".to_string(),
        );
        assert_eq!(
            step.description,
            "Take log base a of both sides (under guard: a > 0)"
        );
        assert_eq!(step.equation_after.lhs, exponent);
    }

    #[test]
    fn build_terminal_outcome_step_appends_suffix_only_for_residual() {
        let mut ctx = Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.var("y");
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };

        let residual_outcome = TerminalSolveOutcome {
            message: "needs complex",
            solutions: residual_solution_set(&mut ctx, lhs, rhs, "x"),
        };
        let residual_step =
            build_terminal_outcome_step(&residual_outcome, eq.clone(), " (residual)");
        assert_eq!(residual_step.description, "needs complex (residual)");

        let empty_outcome = TerminalSolveOutcome {
            message: "empty",
            solutions: SolutionSet::Empty,
        };
        let empty_step = build_terminal_outcome_step(&empty_outcome, eq, " (residual)");
        assert_eq!(empty_step.description, "empty");
    }

    #[test]
    fn build_log_status_steps_use_expected_messages() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let eq = Equation {
            lhs: x,
            rhs: y,
            op: RelOp::Eq,
        };

        let conditional = build_conditional_solution_step("base > 0", eq.clone());
        assert_eq!(conditional.description, "Conditional solution: base > 0");

        let residual = build_residual_step("unsupported", eq.clone());
        assert_eq!(residual.description, "unsupported (residual)");

        let exhausted = build_residual_budget_exhausted_step("unsupported", eq);
        assert_eq!(
            exhausted.description,
            "unsupported (residual, budget exhausted)"
        );
    }

    #[test]
    fn build_guarded_log_followup_step_selects_conditional_or_residual() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let eq = Equation {
            lhs: x,
            rhs: y,
            op: RelOp::Eq,
        };

        let conditional = build_guarded_log_followup_step(true, "base > 0", eq.clone());
        assert_eq!(conditional.description, "Conditional solution: base > 0");

        let residual = build_guarded_log_followup_step(false, "unsupported", eq);
        assert_eq!(residual.description, "unsupported (residual)");
    }

    #[test]
    fn plan_guarded_pow_exponent_log_execution_builds_rewrite_and_followups() {
        let mut ctx = Context::new();
        let exponent = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let source = Equation {
            lhs: ctx.add(Expr::Pow(base, exponent)),
            rhs,
            op: RelOp::Eq,
        };

        let plan = plan_guarded_pow_exponent_log_execution(
            &mut ctx,
            exponent,
            base,
            rhs,
            RelOp::Eq,
            "a > 0 and b > 0",
            "a",
            source,
        );

        assert_eq!(
            plan.rewrite.items[0].description,
            "Take log base a of both sides (under guard: a > 0 and b > 0)"
        );
        assert_eq!(
            plan.followup_success.description,
            "Conditional solution: a > 0 and b > 0"
        );
        assert_eq!(
            plan.followup_residual.description,
            "a > 0 and b > 0 (residual)"
        );
    }

    #[test]
    fn plan_guarded_pow_exponent_log_execution_with_uses_renderer() {
        let mut ctx = Context::new();
        let exponent = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let source = Equation {
            lhs: ctx.add(Expr::Pow(base, exponent)),
            rhs,
            op: RelOp::Eq,
        };

        let plan = plan_guarded_pow_exponent_log_execution_with(
            &mut ctx,
            exponent,
            base,
            rhs,
            RelOp::Eq,
            "a > 0",
            source,
            |_, _| "rendered(a)".to_string(),
        );

        assert_eq!(
            plan.rewrite.items[0].description,
            "Take log base rendered(a) of both sides (under guard: a > 0)"
        );
        assert_eq!(
            plan.followup_success.description,
            "Conditional solution: a > 0"
        );
        assert_eq!(plan.followup_residual.description, "a > 0 (residual)");
    }

    #[test]
    fn guarded_pow_exponent_log_plan_followup_item_selects_success_or_residual() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let a = ctx.var("a");
        let b = ctx.var("b");
        let source = Equation {
            lhs: x,
            rhs: b,
            op: RelOp::Eq,
        };
        let plan = plan_guarded_pow_exponent_log_execution(
            &mut ctx,
            x,
            a,
            b,
            RelOp::Eq,
            "a > 0",
            "a",
            source,
        );

        assert_eq!(
            plan.followup_item(true).description,
            "Conditional solution: a > 0"
        );
        assert_eq!(plan.followup_item(false).description, "a > 0 (residual)");
    }

    #[test]
    fn execute_guarded_pow_exponent_log_with_returns_followup_for_success() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let a = ctx.var("a");
        let b = ctx.var("b");
        let source = Equation {
            lhs: x,
            rhs: b,
            op: RelOp::Eq,
        };
        let solved = SolutionSet::Discrete(vec![x]);

        let execution = execute_guarded_pow_exponent_log_with(
            &mut ctx,
            x,
            a,
            b,
            RelOp::Eq,
            "a > 0",
            source,
            |_, _| "a".to_string(),
            |_eq| Some(solved.clone()),
        );

        assert_eq!(execution.rewrite.items.len(), 1);
        assert_eq!(
            execution.followup.description,
            "Conditional solution: a > 0"
        );
        assert_eq!(execution.guarded_solutions, Some(solved));
    }

    #[test]
    fn execute_guarded_pow_exponent_log_with_returns_followup_for_residual() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let a = ctx.var("a");
        let b = ctx.var("b");
        let source = Equation {
            lhs: x,
            rhs: b,
            op: RelOp::Eq,
        };

        let execution = execute_guarded_pow_exponent_log_with(
            &mut ctx,
            x,
            a,
            b,
            RelOp::Eq,
            "a > 0",
            source,
            |_, _| "a".to_string(),
            |_eq| None,
        );

        assert_eq!(execution.rewrite.items.len(), 1);
        assert_eq!(execution.followup.description, "a > 0 (residual)");
        assert_eq!(execution.guarded_solutions, None);
    }

    #[test]
    fn execute_guarded_pow_exponent_log_plan_with_selects_followup_from_callback() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let a = ctx.var("a");
        let b = ctx.var("b");
        let source = Equation {
            lhs: x,
            rhs: b,
            op: RelOp::Eq,
        };
        let plan = plan_guarded_pow_exponent_log_execution(
            &mut ctx,
            x,
            a,
            b,
            RelOp::Eq,
            "a > 0",
            "a",
            source,
        );
        let solved = SolutionSet::Discrete(vec![x]);

        let execution =
            execute_guarded_pow_exponent_log_plan_with(plan, |_eq| Some(solved.clone()));
        assert_eq!(
            execution.followup.description,
            "Conditional solution: a > 0"
        );
        assert_eq!(execution.guarded_solutions, Some(solved));
    }

    #[test]
    fn solve_pow_exponent_log_isolation_rewrite_with_runs_solver_once_and_preserves_rewrite() {
        let mut ctx = Context::new();
        let exponent = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let rewrite = plan_pow_exponent_log_isolation_step_with(
            &mut ctx,
            exponent,
            base,
            rhs,
            RelOp::Eq,
            None,
            |_, _| "a".to_string(),
        );
        let expected_equation = rewrite.equation.clone();
        let mut calls = 0usize;
        let solved = solve_pow_exponent_log_isolation_rewrite_with(rewrite, |_eq| {
            calls += 1;
            Ok::<_, ()>(SolutionSet::AllReals)
        })
        .expect("solve callback should succeed");

        assert_eq!(calls, 1);
        assert_eq!(solved.rewrite.equation, expected_equation);
        assert!(matches!(solved.solved, SolutionSet::AllReals));
    }

    #[test]
    fn solve_pow_exponent_log_isolation_rewrite_pipeline_with_item_prepends_item() {
        let mut ctx = Context::new();
        let exponent = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let rewrite = plan_pow_exponent_log_isolation_step_with(
            &mut ctx,
            exponent,
            base,
            rhs,
            RelOp::Eq,
            None,
            |_, _| "a".to_string(),
        );
        let expected_item = first_pow_exponent_log_isolation_execution_item(&rewrite)
            .expect("expected one execution item")
            .description
            .clone();

        let mut calls = 0usize;
        let solved = solve_pow_exponent_log_isolation_rewrite_pipeline_with_item(
            rewrite,
            true,
            |_equation| {
                calls += 1;
                Ok::<_, ()>((SolutionSet::AllReals, vec!["recursive-step".to_string()]))
            },
            |item| item.description,
        )
        .expect("pipeline solve should succeed");

        assert_eq!(calls, 1);
        assert!(matches!(solved.solution_set, SolutionSet::AllReals));
        assert_eq!(
            solved.steps,
            vec![expected_item, "recursive-step".to_string()]
        );
    }

    #[test]
    fn solve_pow_exponent_log_isolation_rewrite_pipeline_with_item_omits_item_when_disabled() {
        let mut ctx = Context::new();
        let exponent = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let rewrite = plan_pow_exponent_log_isolation_step_with(
            &mut ctx,
            exponent,
            base,
            rhs,
            RelOp::Eq,
            None,
            |_, _| "a".to_string(),
        );

        let solved = solve_pow_exponent_log_isolation_rewrite_pipeline_with_item(
            rewrite,
            false,
            |_equation| Ok::<_, ()>((SolutionSet::Empty, vec!["only-substep".to_string()])),
            |item| item.description,
        )
        .expect("pipeline solve should succeed");

        assert!(matches!(solved.solution_set, SolutionSet::Empty));
        assert_eq!(solved.steps, vec!["only-substep".to_string()]);
    }

    #[test]
    fn execute_pow_exponent_log_isolation_pipeline_with_item_with_builds_and_solves() {
        let mut ctx = Context::new();
        let exponent = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let mut plan_calls = 0usize;
        let mut solve_calls = 0usize;

        let solved = execute_pow_exponent_log_isolation_pipeline_with_item_with(
            true,
            || {
                plan_calls += 1;
                plan_pow_exponent_log_isolation_step_with(
                    &mut ctx,
                    exponent,
                    base,
                    rhs,
                    RelOp::Eq,
                    None,
                    |_, _| "a".to_string(),
                )
            },
            |_equation| {
                solve_calls += 1;
                Ok::<_, ()>((SolutionSet::AllReals, vec!["sub".to_string()]))
            },
            |item| item.description,
        )
        .expect("execute wrapper should solve");

        assert_eq!(plan_calls, 1);
        assert_eq!(solve_calls, 1);
        assert!(matches!(solved.solution_set, SolutionSet::AllReals));
        assert_eq!(solved.steps.len(), 2);
        assert!(solved.steps[0].contains("Take log base a of both sides"));
        assert_eq!(solved.steps[1], "sub".to_string());
    }

    #[test]
    fn execute_pow_exponent_log_isolation_pipeline_with_item_and_merge_with_existing_steps_with_merges(
    ) {
        let mut ctx = Context::new();
        let exponent = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");

        let merged =
            execute_pow_exponent_log_isolation_pipeline_with_item_and_merge_with_existing_steps_with(
                true,
                vec!["existing".to_string()],
                || {
                    plan_pow_exponent_log_isolation_step_with(
                        &mut ctx,
                        exponent,
                        base,
                        rhs,
                        RelOp::Eq,
                        None,
                        |_, _| "a".to_string(),
                    )
                },
                |_equation| Ok::<_, ()>((SolutionSet::Empty, vec!["sub".to_string()])),
                |item| item.description,
            )
            .expect("execute+merge wrapper should solve");

        assert!(matches!(merged.0, SolutionSet::Empty));
        assert_eq!(merged.1.len(), 3);
        assert!(merged.1[0].contains("Take log base a of both sides"));
        assert_eq!(merged.1[1], "sub".to_string());
        assert_eq!(merged.1[2], "existing".to_string());
    }

    #[test]
    fn collect_term_isolation_didactic_steps_returns_single_step() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let step = build_conditional_solution_step(
            "base > 0",
            Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
        );

        let didactic = collect_term_isolation_didactic_steps(&step);
        assert_eq!(didactic.len(), 1);
        assert_eq!(didactic[0], step);
    }

    #[test]
    fn collect_term_isolation_execution_items_returns_single_item() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let step = build_conditional_solution_step(
            "base > 0",
            Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
        );

        let items = collect_term_isolation_execution_items(&step);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].equation, step.equation_after);
        assert_eq!(items[0].description, step.description);
    }

    #[test]
    fn plan_pow_exponent_log_isolation_step_builds_rewrite_and_step() {
        let mut ctx = Context::new();
        let exponent = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");

        let plan = plan_pow_exponent_log_isolation_step(
            &mut ctx,
            exponent,
            base,
            rhs,
            RelOp::Eq,
            Some("a > 0"),
            "a",
        );

        assert_eq!(plan.items[0].equation, plan.equation);
        assert_eq!(
            plan.items[0].description,
            "Take log base a of both sides (under guard: a > 0)"
        );
        assert_eq!(plan.equation.lhs, exponent);
        assert_eq!(plan.equation.op, RelOp::Eq);
    }

    #[test]
    fn plan_pow_exponent_log_isolation_step_with_uses_renderer() {
        let mut ctx = Context::new();
        let exponent = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");

        let plan = plan_pow_exponent_log_isolation_step_with(
            &mut ctx,
            exponent,
            base,
            rhs,
            RelOp::Eq,
            None,
            |_, _| "rendered(a)".to_string(),
        );

        assert_eq!(plan.items[0].equation, plan.equation);
        assert_eq!(
            plan.items[0].description,
            "Take log base rendered(a) of both sides"
        );
        assert_eq!(plan.equation.lhs, exponent);
        assert_eq!(plan.equation.op, RelOp::Eq);
    }

    #[test]
    fn plan_pow_exponent_log_isolation_step_with_uses_runtime_like_renderer() {
        let mut ctx = Context::new();
        let exponent = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");

        let plan = plan_pow_exponent_log_isolation_step_with(
            &mut ctx,
            exponent,
            base,
            rhs,
            RelOp::Eq,
            None,
            |_, id| format!("runtime({id})"),
        );

        assert_eq!(plan.items[0].equation, plan.equation);
        assert!(plan.items[0].description.contains("Take log base runtime("));
        assert_eq!(plan.equation.lhs, exponent);
        assert_eq!(plan.equation.op, RelOp::Eq);
    }

    #[test]
    fn collect_pow_exponent_log_isolation_didactic_steps_returns_single_step() {
        let mut ctx = Context::new();
        let exponent = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let plan = plan_pow_exponent_log_isolation_step_with(
            &mut ctx,
            exponent,
            base,
            rhs,
            RelOp::Eq,
            None,
            |_, _| "rendered(a)".to_string(),
        );

        let didactic = collect_pow_exponent_log_isolation_didactic_steps(&plan);
        assert_eq!(didactic.len(), 1);
        assert_eq!(didactic[0].description, plan.items[0].description);
        assert_eq!(didactic[0].equation_after, plan.items[0].equation);
    }

    #[test]
    fn collect_pow_exponent_log_isolation_execution_items_returns_single_item() {
        let mut ctx = Context::new();
        let exponent = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let plan = plan_pow_exponent_log_isolation_step_with(
            &mut ctx,
            exponent,
            base,
            rhs,
            RelOp::Eq,
            None,
            |_, _| "rendered(a)".to_string(),
        );

        let items = collect_pow_exponent_log_isolation_execution_items(&plan);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].equation, plan.equation);
        assert_eq!(items[0].description, plan.items[0].description);
    }

    #[test]
    fn first_pow_exponent_log_isolation_execution_item_returns_single_item() {
        let mut ctx = Context::new();
        let exponent = ctx.var("x");
        let base = ctx.var("a");
        let rhs = ctx.var("b");
        let plan = plan_pow_exponent_log_isolation_step_with(
            &mut ctx,
            exponent,
            base,
            rhs,
            RelOp::Eq,
            None,
            |_, _| "rendered(a)".to_string(),
        );

        let item = first_pow_exponent_log_isolation_execution_item(&plan)
            .expect("expected one log isolation item");
        assert_eq!(item.equation, plan.equation);
        assert_eq!(item.description, plan.items[0].description);
    }

    #[test]
    fn plan_division_denominator_sign_split_builds_both_branches_and_domains() {
        let mut ctx = Context::new();
        let num = ctx.var("n");
        let den = ctx.var("d");
        let rhs = ctx.var("r");
        let plan = plan_division_denominator_sign_split(&mut ctx, num, den, rhs, RelOp::Lt)
            .expect("division sign split");

        assert_eq!(plan.positive_equation.lhs, num);
        assert_eq!(plan.negative_equation.lhs, num);
        assert_eq!(plan.positive_equation.op, RelOp::Lt);
        assert_eq!(plan.negative_equation.op, RelOp::Gt);
        assert_eq!(plan.positive_domain.lhs, den);
        assert_eq!(plan.negative_domain.lhs, den);
        assert_eq!(plan.positive_domain.op, RelOp::Gt);
        assert_eq!(plan.negative_domain.op, RelOp::Lt);
    }

    #[test]
    fn build_division_denominator_sign_split_execution_with_uses_simplified_rhs() {
        let mut ctx = Context::new();
        let num = ctx.var("n");
        let den = ctx.var("d");
        let rhs = ctx.var("r");
        let simplified_rhs = ctx.var("s");
        let split =
            plan_division_denominator_sign_split(&mut ctx, num, den, rhs, RelOp::Lt).unwrap();

        let exec = build_division_denominator_sign_split_execution_with(
            split,
            den,
            num,
            RelOp::Lt,
            simplified_rhs,
            |_| "d".to_string(),
        );
        assert_eq!(exec.positive_equation.rhs, simplified_rhs);
        assert_eq!(exec.negative_equation.rhs, simplified_rhs);
        assert_eq!(exec.items.len(), 3);
        assert_eq!(exec.items[0].equation, exec.positive_equation);
        assert_eq!(exec.items[1].equation, exec.negative_equation);
    }

    #[test]
    fn materialize_division_denominator_sign_split_execution_omits_items() {
        let mut ctx = Context::new();
        let num = ctx.var("n");
        let den = ctx.var("d");
        let rhs = ctx.var("r");
        let simplified_rhs = ctx.var("s");
        let split =
            plan_division_denominator_sign_split(&mut ctx, num, den, rhs, RelOp::Lt).unwrap();

        let exec = materialize_division_denominator_sign_split_execution(split, simplified_rhs);
        assert_eq!(exec.positive_equation.rhs, simplified_rhs);
        assert_eq!(exec.negative_equation.rhs, simplified_rhs);
        assert_eq!(exec.positive_equation.lhs, num);
        assert_eq!(exec.negative_equation.lhs, num);
        assert!(exec.items.is_empty());
    }

    #[test]
    fn collect_division_denominator_sign_split_execution_items_preserves_step_order() {
        let mut ctx = Context::new();
        let num = ctx.var("n");
        let den = ctx.var("d");
        let rhs = ctx.var("r");
        let simplified_rhs = ctx.var("s");
        let split =
            plan_division_denominator_sign_split(&mut ctx, num, den, rhs, RelOp::Lt).unwrap();
        let exec = build_division_denominator_sign_split_execution_with(
            split,
            den,
            num,
            RelOp::Lt,
            simplified_rhs,
            |_| "d".to_string(),
        );

        let items = collect_division_denominator_sign_split_execution_items(&exec);
        assert_eq!(items.len(), 3);
        assert_eq!(
            items[0].description,
            "Case 1: Assume d > 0. Multiply by positive denominator."
        );
        assert_eq!(items[0].equation, exec.positive_equation);
        assert_eq!(
            items[1].description,
            "Case 2: Assume d < 0. Multiply by negative denominator (flips inequality)."
        );
        assert_eq!(items[1].equation, exec.negative_equation);
        assert_eq!(items[2].description, "--- End of Case 1 ---");
    }

    #[test]
    fn solve_division_denominator_sign_split_cases_with_solves_branches_and_domains() {
        let mut ctx = Context::new();
        let num = ctx.var("n");
        let den = ctx.var("d");
        let rhs = ctx.var("r");
        let simplified_rhs = ctx.var("s");
        let residual_branch = ctx.var("residual_branch");
        let residual_domain = ctx.var("residual_domain");
        let split =
            plan_division_denominator_sign_split(&mut ctx, num, den, rhs, RelOp::Lt).unwrap();
        let execution =
            materialize_division_denominator_sign_split_execution(split, simplified_rhs);
        let mut branch_calls = 0usize;
        let mut domain_calls = 0usize;
        let solved = solve_division_denominator_sign_split_cases_with(
            &execution,
            |_eq| {
                branch_calls += 1;
                Ok::<_, ()>(match branch_calls {
                    1 => SolutionSet::AllReals,
                    2 => SolutionSet::Residual(residual_branch),
                    _ => unreachable!("only two branch equations"),
                })
            },
            |_eq| {
                domain_calls += 1;
                Ok::<_, ()>(match domain_calls {
                    1 => SolutionSet::Residual(residual_domain),
                    2 => SolutionSet::Empty,
                    _ => unreachable!("only two domain equations"),
                })
            },
        )
        .expect("callbacks should succeed");

        assert_eq!(branch_calls, 2);
        assert_eq!(domain_calls, 2);
        assert!(matches!(solved.positive_branch, SolutionSet::AllReals));
        assert!(matches!(
            solved.negative_branch,
            SolutionSet::Residual(id) if id == residual_branch
        ));
        assert!(matches!(
            solved.positive_domain,
            SolutionSet::Residual(id) if id == residual_domain
        ));
        assert!(matches!(solved.negative_domain, SolutionSet::Empty));
    }

    #[test]
    fn solve_division_denominator_sign_split_cases_with_items_aligns_items_in_order() {
        let mut ctx = Context::new();
        let num = ctx.var("n");
        let den = ctx.var("d");
        let rhs = ctx.var("r");
        let simplified_rhs = ctx.var("s");
        let split =
            plan_division_denominator_sign_split(&mut ctx, num, den, rhs, RelOp::Lt).unwrap();
        let execution = build_division_denominator_sign_split_execution_with(
            split,
            den,
            num,
            RelOp::Lt,
            simplified_rhs,
            |_| "d".to_string(),
        );

        let mut seen = Vec::new();
        let solved = solve_division_denominator_sign_split_cases_with_items(
            &execution,
            |item, equation| {
                seen.push(item.map(|entry| entry.description).unwrap_or_default());
                Ok::<_, ()>(equation.rhs)
            },
            |_domain| Ok::<_, ()>(SolutionSet::AllReals),
        )
        .expect("callbacks should succeed");

        assert_eq!(seen.len(), 2);
        assert_eq!(
            seen[0],
            "Case 1: Assume d > 0. Multiply by positive denominator."
        );
        assert_eq!(
            seen[1],
            "Case 2: Assume d < 0. Multiply by negative denominator (flips inequality)."
        );
        assert_eq!(solved.positive_branch, simplified_rhs);
        assert_eq!(solved.negative_branch, simplified_rhs);
    }

    #[test]
    fn solve_division_denominator_sign_split_execution_with_items_merges_steps_and_keeps_sets() {
        let mut ctx = Context::new();
        let num = ctx.var("n");
        let den = ctx.var("d");
        let rhs = ctx.var("r");
        let simplified_rhs = ctx.var("s");
        let split =
            plan_division_denominator_sign_split(&mut ctx, num, den, rhs, RelOp::Lt).unwrap();
        let execution = build_division_denominator_sign_split_execution_with(
            split,
            den,
            num,
            RelOp::Lt,
            simplified_rhs,
            |_| "d".to_string(),
        );

        let mut branch_calls = 0usize;
        let solved = solve_division_denominator_sign_split_execution_with_items(
            &execution,
            |item, equation| {
                branch_calls += 1;
                let mut steps = vec![format!("branch-{branch_calls}")];
                if let Some(item) = item {
                    steps.push(item.description);
                }
                Ok::<_, ()>((SolutionSet::Discrete(vec![equation.rhs]), steps))
            },
            |_domain| Ok::<_, ()>(SolutionSet::AllReals),
            |item| item.description,
        )
        .expect("execution helper should solve");

        assert_eq!(branch_calls, 2);
        assert_eq!(solved.steps[0], "branch-1");
        assert_eq!(solved.steps[2], "--- End of Case 1 ---");
        assert_eq!(solved.steps[3], "branch-2");
        assert!(matches!(solved.positive_set, SolutionSet::Discrete(_)));
        assert!(matches!(solved.negative_set, SolutionSet::Discrete(_)));
        assert!(matches!(solved.positive_domain_set, SolutionSet::AllReals));
        assert!(matches!(solved.negative_domain_set, SolutionSet::AllReals));
    }

    #[test]
    fn solve_division_denominator_sign_split_execution_pipeline_with_items_prepends_prefix_and_items(
    ) {
        let mut ctx = Context::new();
        let num = ctx.var("n");
        let den = ctx.var("d");
        let rhs = ctx.var("r");
        let simplified_rhs = ctx.var("s");
        let split =
            plan_division_denominator_sign_split(&mut ctx, num, den, rhs, RelOp::Lt).unwrap();
        let execution = build_division_denominator_sign_split_execution_with(
            split,
            den,
            num,
            RelOp::Lt,
            simplified_rhs,
            |_| "d".to_string(),
        );

        let solved = solve_division_denominator_sign_split_execution_pipeline_with_items(
            &execution,
            true,
            &["prefix".to_string()],
            |equation| {
                Ok::<_, ()>((
                    SolutionSet::Discrete(vec![equation.rhs]),
                    vec!["branch".to_string()],
                ))
            },
            |_domain_equation| Ok::<_, ()>(SolutionSet::AllReals),
            |item| item.description,
        )
        .expect("pipeline should solve");

        assert!(matches!(solved.positive_set, SolutionSet::Discrete(_)));
        assert!(matches!(solved.negative_set, SolutionSet::Discrete(_)));
        assert!(matches!(solved.positive_domain_set, SolutionSet::AllReals));
        assert!(matches!(solved.negative_domain_set, SolutionSet::AllReals));
        assert_eq!(solved.steps[0], "prefix");
        assert!(solved.steps[1].starts_with("Case 1: Assume d > 0"));
        assert_eq!(solved.steps[2], "branch");
        assert_eq!(solved.steps[3], "--- End of Case 1 ---");
        assert_eq!(solved.steps[4], "prefix");
        assert!(solved.steps[5].starts_with("Case 2: Assume d < 0"));
        assert_eq!(solved.steps[6], "branch");
    }

    #[test]
    fn solve_division_denominator_sign_split_execution_pipeline_with_items_omits_items_when_disabled(
    ) {
        let mut ctx = Context::new();
        let num = ctx.var("n");
        let den = ctx.var("d");
        let rhs = ctx.var("r");
        let simplified_rhs = ctx.var("s");
        let split =
            plan_division_denominator_sign_split(&mut ctx, num, den, rhs, RelOp::Lt).unwrap();
        let execution = build_division_denominator_sign_split_execution_with(
            split,
            den,
            num,
            RelOp::Lt,
            simplified_rhs,
            |_| "d".to_string(),
        );

        let solved = solve_division_denominator_sign_split_execution_pipeline_with_items(
            &execution,
            false,
            &[0u8],
            |_equation| Ok::<_, ()>((SolutionSet::Discrete(vec![rhs]), vec![1u8])),
            |_domain_equation| Ok::<_, ()>(SolutionSet::AllReals),
            |_item| 9u8,
        )
        .expect("pipeline should solve");

        assert_eq!(solved.steps, vec![0u8, 1u8, 0u8, 1u8]);
    }

    #[test]
    fn solve_division_denominator_sign_split_pipeline_with_optional_items_includes_didactic_when_enabled(
    ) {
        let mut ctx = Context::new();
        let num = ctx.var("n");
        let den = ctx.var("d");
        let rhs = ctx.var("r");
        let simplified_rhs = ctx.var("s");
        let split =
            plan_division_denominator_sign_split(&mut ctx, num, den, rhs, RelOp::Lt).unwrap();

        let solved = solve_division_denominator_sign_split_pipeline_with_optional_items(
            split,
            den,
            num,
            RelOp::Lt,
            simplified_rhs,
            true,
            &["prefix".to_string()],
            |_| "d".to_string(),
            |equation| {
                Ok::<_, ()>((
                    SolutionSet::Discrete(vec![equation.rhs]),
                    vec!["branch".to_string()],
                ))
            },
            |_domain_equation| Ok::<_, ()>(SolutionSet::AllReals),
            |item| item.description,
        )
        .expect("pipeline should solve");

        assert_eq!(solved.steps[0], "prefix");
        assert!(solved.steps[1].starts_with("Case 1: Assume d > 0"));
        assert_eq!(solved.steps[2], "branch");
        assert_eq!(solved.steps[3], "--- End of Case 1 ---");
        assert_eq!(solved.steps[4], "prefix");
        assert!(solved.steps[5].starts_with("Case 2: Assume d < 0"));
        assert_eq!(solved.steps[6], "branch");
    }

    #[test]
    fn solve_division_denominator_sign_split_pipeline_with_optional_items_skips_render_when_disabled(
    ) {
        let mut ctx = Context::new();
        let num = ctx.var("n");
        let den = ctx.var("d");
        let rhs = ctx.var("r");
        let simplified_rhs = ctx.var("s");
        let split =
            plan_division_denominator_sign_split(&mut ctx, num, den, rhs, RelOp::Lt).unwrap();

        let solved = solve_division_denominator_sign_split_pipeline_with_optional_items(
            split,
            den,
            num,
            RelOp::Lt,
            simplified_rhs,
            false,
            &[0u8],
            |_id| -> String { panic!("renderer must not run when items are disabled") },
            |_equation| Ok::<_, ()>((SolutionSet::Discrete(vec![rhs]), vec![1u8])),
            |_domain_equation| Ok::<_, ()>(SolutionSet::AllReals),
            |_item| -> u8 { panic!("mapper must not run when items are disabled") },
        )
        .expect("pipeline should solve");

        assert_eq!(solved.steps, vec![0u8, 1u8, 0u8, 1u8]);
    }

    #[test]
    fn execute_division_denominator_sign_split_pipeline_with_optional_items_finalizes_sets() {
        let mut ctx = Context::new();
        let num = ctx.var("n");
        let den = ctx.var("d");
        let rhs = ctx.var("r");
        let simplified_rhs = ctx.var("s");
        let split =
            plan_division_denominator_sign_split(&mut ctx, num, den, rhs, RelOp::Lt).unwrap();

        let mut finalize_called = false;
        let (final_set, steps) =
            execute_division_denominator_sign_split_pipeline_with_optional_items(
                split,
                den,
                num,
                RelOp::Lt,
                simplified_rhs,
                false,
                &[0u8],
                |_id| -> String { panic!("renderer must not run when items are disabled") },
                |_equation| Ok::<_, ()>((SolutionSet::Discrete(vec![rhs]), vec![1u8])),
                |_domain_equation| Ok::<_, ()>(SolutionSet::AllReals),
                |_item| -> u8 { panic!("mapper must not run when items are disabled") },
                |solved_cases| {
                    finalize_called = true;
                    assert!(matches!(
                        solved_cases.positive_domain,
                        SolutionSet::AllReals
                    ));
                    assert!(matches!(
                        solved_cases.negative_domain,
                        SolutionSet::AllReals
                    ));
                    SolutionSet::AllReals
                },
            )
            .expect("execute helper should solve and finalize");

        assert!(finalize_called);
        assert!(matches!(final_set, SolutionSet::AllReals));
        assert_eq!(steps, vec![0u8, 1u8, 0u8, 1u8]);
    }

    #[test]
    fn try_execute_division_denominator_sign_split_pipeline_with_optional_items_returns_none_without_plan(
    ) {
        let out = try_execute_division_denominator_sign_split_pipeline_with_optional_items::<
            (),
            u8,
            _,
            _,
            _,
            _,
            _,
        >(
            None,
            ExprId::from_raw(1),
            RelOp::Lt,
            ExprId::from_raw(2),
            ExprId::from_raw(3),
            false,
            &[],
            |_id| "x".to_string(),
            |_equation| Ok((SolutionSet::AllReals, vec![])),
            |_equation| Ok(SolutionSet::AllReals),
            |_item| 0u8,
            |_solved_cases| SolutionSet::AllReals,
        );
        assert!(out.is_none());
    }

    #[test]
    fn try_execute_division_denominator_sign_split_pipeline_with_optional_items_executes_with_plan()
    {
        let mut ctx = Context::new();
        let num = ctx.var("n");
        let den = ctx.var("d");
        let rhs = ctx.var("r");
        let simplified_rhs = ctx.var("s");
        let split =
            plan_division_denominator_sign_split(&mut ctx, num, den, rhs, RelOp::Lt).unwrap();

        let out = try_execute_division_denominator_sign_split_pipeline_with_optional_items(
            Some(split),
            den,
            RelOp::Lt,
            num,
            simplified_rhs,
            false,
            &[0u8],
            |_id| "d".to_string(),
            |_equation| Ok::<_, ()>((SolutionSet::Discrete(vec![rhs]), vec![1u8])),
            |_domain_equation| Ok::<_, ()>(SolutionSet::AllReals),
            |_item| 9u8,
            |_solved_cases| SolutionSet::AllReals,
        )
        .expect("plan is present");

        let (final_set, steps) = out.expect("execution should succeed");
        assert!(matches!(final_set, SolutionSet::AllReals));
        assert_eq!(steps, vec![0u8, 1u8, 0u8, 1u8]);
    }

    #[test]
    fn division_denominator_sign_split_boundary_item_returns_case_separator() {
        let mut ctx = Context::new();
        let num = ctx.var("n");
        let den = ctx.var("d");
        let rhs = ctx.var("r");
        let simplified_rhs = ctx.var("s");
        let split =
            plan_division_denominator_sign_split(&mut ctx, num, den, rhs, RelOp::Lt).unwrap();
        let execution = build_division_denominator_sign_split_execution_with(
            split,
            den,
            num,
            RelOp::Lt,
            simplified_rhs,
            |_| "d".to_string(),
        );

        let boundary =
            division_denominator_sign_split_boundary_item(&execution).expect("boundary item");
        assert_eq!(boundary.description, "--- End of Case 1 ---");
    }

    #[test]
    fn finalize_division_denominator_sign_split_solved_sets_uses_solution_set_combiner() {
        let ctx = Context::new();
        let out = finalize_division_denominator_sign_split_solved_sets(
            &ctx,
            DivisionDenominatorSignSplitSolvedCases {
                positive_branch: SolutionSet::AllReals,
                negative_branch: SolutionSet::Empty,
                positive_domain: SolutionSet::AllReals,
                negative_domain: SolutionSet::AllReals,
            },
        );
        assert!(matches!(out, SolutionSet::AllReals));
    }

    #[test]
    fn plan_div_denominator_isolation_with_zero_rhs_guard_marks_infinity_route() {
        let mut ctx = Context::new();
        let den = ctx.var("d");
        let num = ctx.var("n");
        let zero = ctx.num(0);

        let plan =
            plan_div_denominator_isolation_with_zero_rhs_guard(&mut ctx, den, num, zero, RelOp::Eq);

        assert_eq!(plan.equation.lhs, den);
        assert_eq!(plan.route, DivDenominatorIsolationRoute::RhsZeroToInfinity);
    }

    #[test]
    fn plan_div_denominator_isolation_with_zero_rhs_guard_marks_division_route() {
        let mut ctx = Context::new();
        let den = ctx.var("d");
        let num = ctx.var("n");
        let rhs = ctx.var("r");

        let plan =
            plan_div_denominator_isolation_with_zero_rhs_guard(&mut ctx, den, num, rhs, RelOp::Eq);

        assert_eq!(plan.equation.lhs, den);
        assert_eq!(plan.route, DivDenominatorIsolationRoute::DivisionRewrite);
    }

    #[test]
    fn plan_division_denominator_sign_split_rejects_eq_and_neq() {
        let mut ctx = Context::new();
        let num = ctx.var("n");
        let den = ctx.var("d");
        let rhs = ctx.var("r");

        assert!(plan_division_denominator_sign_split(&mut ctx, num, den, rhs, RelOp::Eq).is_none());
        assert!(
            plan_division_denominator_sign_split(&mut ctx, num, den, rhs, RelOp::Neq).is_none()
        );
    }

    #[test]
    fn plan_division_denominator_sign_split_if_applicable_returns_some_for_valid_case() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let num = ctx.add(Expr::Add(x, one));
        let den = ctx.add(Expr::Sub(x, two));
        let rhs = ctx.var("r");

        let plan = plan_division_denominator_sign_split_if_applicable(
            &mut ctx,
            num,
            den,
            rhs,
            RelOp::Lt,
            "x",
        );
        assert!(plan.is_some());
    }

    #[test]
    fn plan_division_denominator_sign_split_if_applicable_rejects_non_inequality() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let num = ctx.add(Expr::Add(x, one));
        let den = ctx.add(Expr::Sub(x, two));
        let rhs = ctx.var("r");

        let plan = plan_division_denominator_sign_split_if_applicable(
            &mut ctx,
            num,
            den,
            rhs,
            RelOp::Eq,
            "x",
        );
        assert!(plan.is_none());
    }

    #[test]
    fn plan_isolated_denominator_sign_split_builds_flipped_negative_branch() {
        let mut ctx = Context::new();
        let den = ctx.var("x");
        let rhs = ctx.var("r");
        let plan = plan_isolated_denominator_sign_split(den, rhs, RelOp::Leq)
            .expect("isolated denominator split");

        assert_eq!(plan.positive_equation.lhs, den);
        assert_eq!(plan.negative_equation.lhs, den);
        assert_eq!(plan.positive_equation.rhs, rhs);
        assert_eq!(plan.negative_equation.rhs, rhs);
        assert_eq!(plan.positive_equation.op, RelOp::Geq);
        assert_eq!(plan.negative_equation.op, RelOp::Leq);
    }

    #[test]
    fn plan_isolated_denominator_sign_split_if_applicable_returns_some_for_target_var() {
        let mut ctx = Context::new();
        let den = ctx.var("x");
        let rhs = ctx.var("r");
        let plan =
            plan_isolated_denominator_sign_split_if_applicable(&ctx, den, rhs, RelOp::Leq, "x");
        assert!(plan.is_some());
    }

    #[test]
    fn plan_isolated_denominator_sign_split_if_applicable_rejects_non_target_var() {
        let mut ctx = Context::new();
        let den = ctx.var("y");
        let rhs = ctx.var("r");
        let plan =
            plan_isolated_denominator_sign_split_if_applicable(&ctx, den, rhs, RelOp::Leq, "x");
        assert!(plan.is_none());
    }

    #[test]
    fn build_isolated_denominator_sign_split_execution_with_builds_didactic_payload() {
        let mut ctx = Context::new();
        let den = ctx.var("x");
        let rhs = ctx.var("r");
        let split = plan_isolated_denominator_sign_split(den, rhs, RelOp::Leq).unwrap();
        let exec =
            build_isolated_denominator_sign_split_execution_with(split, den, RelOp::Leq, |_| {
                "x".to_string()
            });
        assert_eq!(exec.positive_equation.rhs, rhs);
        assert_eq!(exec.negative_equation.rhs, rhs);
        assert_eq!(exec.items.len(), 3);
        assert_eq!(exec.items[0].equation, exec.positive_equation);
        assert_eq!(exec.items[1].equation, exec.negative_equation);
    }

    #[test]
    fn materialize_isolated_denominator_sign_split_execution_omits_items() {
        let mut ctx = Context::new();
        let den = ctx.var("x");
        let rhs = ctx.var("r");
        let split = plan_isolated_denominator_sign_split(den, rhs, RelOp::Leq).unwrap();

        let exec = materialize_isolated_denominator_sign_split_execution(split);
        assert_eq!(exec.positive_equation.lhs, den);
        assert_eq!(exec.negative_equation.lhs, den);
        assert_eq!(exec.positive_equation.rhs, rhs);
        assert_eq!(exec.negative_equation.rhs, rhs);
        assert!(exec.items.is_empty());
    }

    #[test]
    fn collect_isolated_denominator_sign_split_execution_items_preserves_step_order() {
        let mut ctx = Context::new();
        let den = ctx.var("x");
        let rhs = ctx.var("r");
        let split = plan_isolated_denominator_sign_split(den, rhs, RelOp::Leq).unwrap();
        let exec =
            build_isolated_denominator_sign_split_execution_with(split, den, RelOp::Leq, |_| {
                "x".to_string()
            });

        let items = collect_isolated_denominator_sign_split_execution_items(&exec);
        assert_eq!(items.len(), 3);
        assert_eq!(
            items[0].description,
            "Case 1: Assume x > 0. Multiply by x (positive). Inequality direction preserved (flipped from isolation logic)."
        );
        assert_eq!(items[0].equation, exec.positive_equation);
        assert_eq!(
            items[1].description,
            "Case 2: Assume x < 0. Multiply by x (negative). Inequality flips."
        );
        assert_eq!(items[1].equation, exec.negative_equation);
        assert_eq!(items[2].description, "--- End of Case 1 ---");
    }

    #[test]
    fn solve_isolated_denominator_sign_split_cases_with_solves_two_branches() {
        let mut ctx = Context::new();
        let den = ctx.var("x");
        let rhs = ctx.var("r");
        let residual = ctx.var("residual");
        let split = plan_isolated_denominator_sign_split(den, rhs, RelOp::Leq).unwrap();
        let execution = materialize_isolated_denominator_sign_split_execution(split);
        let mut branch_calls = 0usize;
        let solved = solve_isolated_denominator_sign_split_cases_with(&execution, |_eq| {
            branch_calls += 1;
            Ok::<_, ()>(match branch_calls {
                1 => SolutionSet::AllReals,
                2 => SolutionSet::Residual(residual),
                _ => unreachable!("only two branch equations"),
            })
        })
        .expect("callback should succeed");

        assert_eq!(branch_calls, 2);
        assert!(matches!(solved.positive_branch, SolutionSet::AllReals));
        assert!(matches!(
            solved.negative_branch,
            SolutionSet::Residual(id) if id == residual
        ));
    }

    #[test]
    fn solve_isolated_denominator_sign_split_cases_with_items_aligns_items_in_order() {
        let mut ctx = Context::new();
        let den = ctx.var("x");
        let rhs = ctx.var("r");
        let split = plan_isolated_denominator_sign_split(den, rhs, RelOp::Leq).unwrap();
        let execution =
            build_isolated_denominator_sign_split_execution_with(split, den, RelOp::Leq, |_| {
                "x".to_string()
            });

        let mut seen = Vec::new();
        let solved =
            solve_isolated_denominator_sign_split_cases_with_items(&execution, |item, equation| {
                seen.push(item.map(|entry| entry.description).unwrap_or_default());
                Ok::<_, ()>(equation.rhs)
            })
            .expect("callback should succeed");

        assert_eq!(seen.len(), 2);
        assert_eq!(
            seen[0],
            "Case 1: Assume x > 0. Multiply by x (positive). Inequality direction preserved (flipped from isolation logic)."
        );
        assert_eq!(
            seen[1],
            "Case 2: Assume x < 0. Multiply by x (negative). Inequality flips."
        );
        assert_eq!(solved.positive_branch, rhs);
        assert_eq!(solved.negative_branch, rhs);
    }

    #[test]
    fn solve_isolated_denominator_sign_split_execution_with_items_merges_steps_and_keeps_sets() {
        let mut ctx = Context::new();
        let den = ctx.var("x");
        let rhs = ctx.var("r");
        let split = plan_isolated_denominator_sign_split(den, rhs, RelOp::Leq).unwrap();
        let execution =
            build_isolated_denominator_sign_split_execution_with(split, den, RelOp::Leq, |_| {
                "x".to_string()
            });

        let mut branch_calls = 0usize;
        let solved = solve_isolated_denominator_sign_split_execution_with_items(
            &execution,
            |item, equation| {
                branch_calls += 1;
                let mut steps = vec![format!("branch-{branch_calls}")];
                if let Some(item) = item {
                    steps.push(item.description);
                }
                Ok::<_, ()>((SolutionSet::Discrete(vec![equation.rhs]), steps))
            },
            |item| item.description,
        )
        .expect("execution helper should solve");

        assert_eq!(branch_calls, 2);
        assert_eq!(solved.steps[0], "branch-1");
        assert_eq!(solved.steps[2], "--- End of Case 1 ---");
        assert_eq!(solved.steps[3], "branch-2");
        assert!(matches!(solved.positive_set, SolutionSet::Discrete(_)));
        assert!(matches!(solved.negative_set, SolutionSet::Discrete(_)));
    }

    #[test]
    fn solve_isolated_denominator_sign_split_execution_pipeline_with_items_prepends_prefix_and_items(
    ) {
        let mut ctx = Context::new();
        let den = ctx.var("x");
        let rhs = ctx.var("r");
        let split = plan_isolated_denominator_sign_split(den, rhs, RelOp::Leq).unwrap();
        let execution =
            build_isolated_denominator_sign_split_execution_with(split, den, RelOp::Leq, |_| {
                "x".to_string()
            });

        let solved = solve_isolated_denominator_sign_split_execution_pipeline_with_items(
            &execution,
            true,
            &["prefix".to_string()],
            |equation| {
                Ok::<_, ()>((
                    SolutionSet::Discrete(vec![equation.rhs]),
                    vec!["branch".to_string()],
                ))
            },
            |item| item.description,
        )
        .expect("pipeline should solve");

        assert!(matches!(solved.positive_set, SolutionSet::Discrete(_)));
        assert!(matches!(solved.negative_set, SolutionSet::Discrete(_)));
        assert_eq!(solved.steps[0], "prefix");
        assert!(solved.steps[1].starts_with("Case 1: Assume x > 0"));
        assert_eq!(solved.steps[2], "branch");
        assert_eq!(solved.steps[3], "--- End of Case 1 ---");
        assert_eq!(solved.steps[4], "prefix");
        assert!(solved.steps[5].starts_with("Case 2: Assume x < 0"));
        assert_eq!(solved.steps[6], "branch");
    }

    #[test]
    fn solve_isolated_denominator_sign_split_execution_pipeline_with_items_omits_items_when_disabled(
    ) {
        let mut ctx = Context::new();
        let den = ctx.var("x");
        let rhs = ctx.var("r");
        let split = plan_isolated_denominator_sign_split(den, rhs, RelOp::Leq).unwrap();
        let execution =
            build_isolated_denominator_sign_split_execution_with(split, den, RelOp::Leq, |_| {
                "x".to_string()
            });

        let solved = solve_isolated_denominator_sign_split_execution_pipeline_with_items(
            &execution,
            false,
            &[0u8],
            |_equation| Ok::<_, ()>((SolutionSet::Discrete(vec![rhs]), vec![1u8])),
            |_item| 9u8,
        )
        .expect("pipeline should solve");

        assert_eq!(solved.steps, vec![0u8, 1u8, 0u8, 1u8]);
    }

    #[test]
    fn solve_isolated_denominator_sign_split_pipeline_with_optional_items_includes_didactic_when_enabled(
    ) {
        let mut ctx = Context::new();
        let den = ctx.var("x");
        let rhs = ctx.var("r");
        let split = plan_isolated_denominator_sign_split(den, rhs, RelOp::Leq).unwrap();

        let solved = solve_isolated_denominator_sign_split_pipeline_with_optional_items(
            split,
            den,
            RelOp::Leq,
            true,
            &["prefix".to_string()],
            |_| "x".to_string(),
            |equation| {
                Ok::<_, ()>((
                    SolutionSet::Discrete(vec![equation.rhs]),
                    vec!["branch".to_string()],
                ))
            },
            |item| item.description,
        )
        .expect("pipeline should solve");

        assert_eq!(solved.steps[0], "prefix");
        assert!(solved.steps[1].starts_with("Case 1: Assume x > 0"));
        assert_eq!(solved.steps[2], "branch");
        assert_eq!(solved.steps[3], "--- End of Case 1 ---");
        assert_eq!(solved.steps[4], "prefix");
        assert!(solved.steps[5].starts_with("Case 2: Assume x < 0"));
        assert_eq!(solved.steps[6], "branch");
    }

    #[test]
    fn solve_isolated_denominator_sign_split_pipeline_with_optional_items_skips_render_when_disabled(
    ) {
        let mut ctx = Context::new();
        let den = ctx.var("x");
        let rhs = ctx.var("r");
        let split = plan_isolated_denominator_sign_split(den, rhs, RelOp::Leq).unwrap();

        let solved = solve_isolated_denominator_sign_split_pipeline_with_optional_items(
            split,
            den,
            RelOp::Leq,
            false,
            &[0u8],
            |_id| -> String { panic!("renderer must not run when items are disabled") },
            |_equation| Ok::<_, ()>((SolutionSet::Discrete(vec![rhs]), vec![1u8])),
            |_item| -> u8 { panic!("mapper must not run when items are disabled") },
        )
        .expect("pipeline should solve");

        assert_eq!(solved.steps, vec![0u8, 1u8, 0u8, 1u8]);
    }

    #[test]
    fn execute_isolated_denominator_sign_split_pipeline_with_optional_items_finalizes_sets() {
        let mut ctx = Context::new();
        let den = ctx.var("x");
        let rhs = ctx.var("r");
        let split = plan_isolated_denominator_sign_split(den, rhs, RelOp::Leq).unwrap();

        let mut finalize_called = false;
        let (final_set, steps) =
            execute_isolated_denominator_sign_split_pipeline_with_optional_items(
                split,
                den,
                RelOp::Leq,
                false,
                &[0u8],
                |_id| -> String { panic!("renderer must not run when items are disabled") },
                |_equation| Ok::<_, ()>((SolutionSet::Discrete(vec![rhs]), vec![1u8])),
                |_item| -> u8 { panic!("mapper must not run when items are disabled") },
                |solved_cases| {
                    finalize_called = true;
                    assert!(matches!(
                        solved_cases.positive_branch,
                        SolutionSet::Discrete(_)
                    ));
                    assert!(matches!(
                        solved_cases.negative_branch,
                        SolutionSet::Discrete(_)
                    ));
                    SolutionSet::Empty
                },
            )
            .expect("execute helper should solve and finalize");

        assert!(finalize_called);
        assert!(matches!(final_set, SolutionSet::Empty));
        assert_eq!(steps, vec![0u8, 1u8, 0u8, 1u8]);
    }

    #[test]
    fn try_execute_isolated_denominator_sign_split_pipeline_with_optional_items_returns_none_without_plan(
    ) {
        let out = try_execute_isolated_denominator_sign_split_pipeline_with_optional_items::<
            (),
            u8,
            _,
            _,
            _,
            _,
        >(
            None,
            ExprId::from_raw(1),
            RelOp::Leq,
            false,
            &[],
            |_id| "x".to_string(),
            |_equation| Ok((SolutionSet::AllReals, vec![])),
            |_item| 0u8,
            |_solved_cases| SolutionSet::AllReals,
        );
        assert!(out.is_none());
    }

    #[test]
    fn try_execute_isolated_denominator_sign_split_pipeline_with_optional_items_executes_with_plan()
    {
        let mut ctx = Context::new();
        let den = ctx.var("x");
        let rhs = ctx.var("r");
        let split = plan_isolated_denominator_sign_split(den, rhs, RelOp::Leq).unwrap();

        let out = try_execute_isolated_denominator_sign_split_pipeline_with_optional_items(
            Some(split),
            den,
            RelOp::Leq,
            false,
            &[0u8],
            |_id| "x".to_string(),
            |_equation| Ok::<_, ()>((SolutionSet::Discrete(vec![rhs]), vec![1u8])),
            |_item| 9u8,
            |_solved_cases| SolutionSet::Empty,
        )
        .expect("plan is present");

        let (final_set, steps) = out.expect("execution should succeed");
        assert!(matches!(final_set, SolutionSet::Empty));
        assert_eq!(steps, vec![0u8, 1u8, 0u8, 1u8]);
    }

    #[test]
    fn isolated_denominator_sign_split_boundary_item_returns_case_separator() {
        let mut ctx = Context::new();
        let den = ctx.var("x");
        let rhs = ctx.var("r");
        let split = plan_isolated_denominator_sign_split(den, rhs, RelOp::Leq).unwrap();
        let execution =
            build_isolated_denominator_sign_split_execution_with(split, den, RelOp::Leq, |_| {
                "x".to_string()
            });

        let boundary =
            isolated_denominator_sign_split_boundary_item(&execution).expect("boundary item");
        assert_eq!(boundary.description, "--- End of Case 1 ---");
    }

    #[test]
    fn finalize_isolated_denominator_sign_split_solved_sets_returns_empty_for_empty_branches() {
        let mut ctx = Context::new();
        let out = finalize_isolated_denominator_sign_split_solved_sets(
            &mut ctx,
            IsolatedDenominatorSignSplitSolvedCases {
                positive_branch: SolutionSet::Empty,
                negative_branch: SolutionSet::Empty,
            },
        );
        assert!(matches!(out, SolutionSet::Empty));
    }

    #[test]
    fn plan_division_denominator_didactic_builds_two_step_equations() {
        let mut ctx = Context::new();
        let num = ctx.var("a");
        let den = ctx.var("x");
        let rhs = ctx.var("b");
        let isolated_rhs = ctx.var("c");
        let plan =
            plan_division_denominator_didactic(&mut ctx, num, den, rhs, isolated_rhs, RelOp::Eq);

        assert_eq!(plan.multiply_equation.lhs, num);
        assert_eq!(plan.multiply_equation.op, RelOp::Eq);
        assert!(matches!(
            ctx.get(plan.multiply_equation.rhs),
            Expr::Mul(_, _)
        ));
        assert_eq!(plan.divide_equation.lhs, den);
        assert_eq!(plan.divide_equation.rhs, isolated_rhs);
        assert_eq!(plan.divide_equation.op, RelOp::Eq);
        assert_eq!(plan.multiply_by, den);
        assert_eq!(plan.divide_by, rhs);
    }

    #[test]
    fn build_case_boundary_equation_keeps_fields() {
        let mut ctx = Context::new();
        let lhs = ctx.var("lhs");
        let rhs = ctx.var("rhs");
        let eq = build_case_boundary_equation(lhs, rhs, RelOp::Gt);
        assert_eq!(eq.lhs, lhs);
        assert_eq!(eq.rhs, rhs);
        assert_eq!(eq.op, RelOp::Gt);
    }

    #[test]
    fn build_zero_constraint_equation_sets_eq_zero_rhs() {
        let mut ctx = Context::new();
        let diff = ctx.var("d");
        let eq = build_zero_constraint_equation(&mut ctx, diff);
        assert_eq!(eq.lhs, diff);
        assert_eq!(eq.op, RelOp::Eq);
        assert!(matches!(
            ctx.get(eq.rhs),
            Expr::Number(n) if *n == num_rational::BigRational::from_integer(0.into())
        ));
    }

    #[test]
    fn plan_product_zero_inequality_split_builds_two_sign_cases() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let plan =
            plan_product_zero_inequality_split(&mut ctx, a, b, RelOp::Gt).expect("split plan");
        assert_eq!(plan.case1_left.op, RelOp::Gt);
        assert_eq!(plan.case1_right.op, RelOp::Gt);
        assert_eq!(plan.case2_left.op, RelOp::Lt);
        assert_eq!(plan.case2_right.op, RelOp::Lt);
    }

    #[test]
    fn plan_product_zero_inequality_split_rejects_equality_operators() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        assert!(plan_product_zero_inequality_split(&mut ctx, a, b, RelOp::Eq).is_none());
        assert!(plan_product_zero_inequality_split(&mut ctx, a, b, RelOp::Neq).is_none());
    }

    #[test]
    fn plan_product_zero_inequality_split_if_applicable_returns_some_for_valid_case() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let left = ctx.add(Expr::Add(x, one));
        let right = ctx.add(Expr::Sub(x, one));
        let zero = ctx.num(0);

        let plan = plan_product_zero_inequality_split_if_applicable(
            &mut ctx,
            left,
            right,
            zero,
            RelOp::Gt,
            "x",
        );
        assert!(plan.is_some());
    }

    #[test]
    fn plan_product_zero_inequality_split_if_applicable_rejects_non_zero_rhs() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let left = ctx.add(Expr::Add(x, one));
        let right = ctx.add(Expr::Sub(x, one));

        let plan = plan_product_zero_inequality_split_if_applicable(
            &mut ctx,
            left,
            right,
            one,
            RelOp::Gt,
            "x",
        );
        assert!(plan.is_none());
    }

    #[test]
    fn solve_product_zero_inequality_cases_with_solves_all_four_equations_in_order() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let residual_1 = ctx.var("residual_1");
        let residual_2 = ctx.var("residual_2");
        let plan =
            plan_product_zero_inequality_split(&mut ctx, a, b, RelOp::Gt).expect("split plan");
        let mut call_index = 0usize;
        let solved = solve_product_zero_inequality_cases_with(&plan, |_eq| {
            call_index += 1;
            Ok::<_, ()>(match call_index {
                1 => SolutionSet::AllReals,
                2 => SolutionSet::Empty,
                3 => SolutionSet::Residual(residual_1),
                4 => SolutionSet::Residual(residual_2),
                _ => unreachable!("only four branch equations"),
            })
        })
        .expect("solver callback should succeed");

        assert_eq!(call_index, 4);
        assert!(matches!(solved.case1_left, SolutionSet::AllReals));
        assert!(matches!(solved.case1_right, SolutionSet::Empty));
        assert!(matches!(
            solved.case2_left,
            SolutionSet::Residual(id) if id == residual_1
        ));
        assert!(matches!(
            solved.case2_right,
            SolutionSet::Residual(id) if id == residual_2
        ));
    }

    #[test]
    fn finalize_product_zero_inequality_solved_sets_uses_solution_set_combiner() {
        let ctx = Context::new();
        let solved = ProductZeroInequalitySolvedSets {
            case1_left: SolutionSet::AllReals,
            case1_right: SolutionSet::Empty,
            case2_left: SolutionSet::AllReals,
            case2_right: SolutionSet::AllReals,
        };
        let out = finalize_product_zero_inequality_solved_sets(&ctx, solved);
        assert!(matches!(out, SolutionSet::AllReals));
    }

    #[test]
    fn solve_product_zero_inequality_split_execution_with_merges_steps_and_finalizes_set() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let plan =
            plan_product_zero_inequality_split(&mut ctx, a, b, RelOp::Gt).expect("split plan");
        let mut call_index = 0usize;
        let solved = solve_product_zero_inequality_split_execution_with(&plan, |_eq| {
            call_index += 1;
            Ok::<_, ()>((
                if call_index == 2 {
                    SolutionSet::Empty
                } else {
                    SolutionSet::AllReals
                },
                vec![format!("case-{call_index}")],
            ))
        })
        .expect("execution should solve");

        assert_eq!(call_index, 4);
        assert_eq!(
            solved.steps,
            vec![
                "case-1".to_string(),
                "case-2".to_string(),
                "case-3".to_string(),
                "case-4".to_string(),
            ]
        );
        let final_set = finalize_product_zero_inequality_solved_sets(&ctx, solved.solved_sets);
        assert!(matches!(final_set, SolutionSet::AllReals));
    }

    #[test]
    fn execute_product_zero_inequality_split_pipeline_with_existing_steps_finalizes_and_merges() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let plan =
            plan_product_zero_inequality_split(&mut ctx, a, b, RelOp::Gt).expect("split plan");
        let mut call_index = 0usize;
        let solved = execute_product_zero_inequality_split_pipeline_with_existing_steps(
            &plan,
            vec!["existing".to_string()],
            |_eq| {
                call_index += 1;
                Ok::<_, ()>((
                    if call_index == 2 {
                        SolutionSet::Empty
                    } else {
                        SolutionSet::AllReals
                    },
                    vec![format!("case-{call_index}")],
                ))
            },
            |solved_sets| finalize_product_zero_inequality_solved_sets(&ctx, solved_sets),
        )
        .expect("execution should solve");

        assert_eq!(call_index, 4);
        assert!(matches!(solved.0, SolutionSet::AllReals));
        assert_eq!(
            solved.1,
            vec![
                "case-1".to_string(),
                "case-2".to_string(),
                "case-3".to_string(),
                "case-4".to_string(),
                "existing".to_string(),
            ]
        );
    }

    #[test]
    fn try_execute_product_zero_inequality_split_pipeline_with_existing_steps_returns_none_without_plan(
    ) {
        let out =
            try_execute_product_zero_inequality_split_pipeline_with_existing_steps::<(), u8, _, _>(
                None,
                &[],
                |_eq| Ok((SolutionSet::AllReals, vec![])),
                |_solved| SolutionSet::AllReals,
            );
        assert!(out.is_none());
    }

    #[test]
    fn try_execute_product_zero_inequality_split_pipeline_with_existing_steps_executes_with_plan() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let plan = plan_product_zero_inequality_split(&mut ctx, a, b, RelOp::Gt).unwrap();

        let out = try_execute_product_zero_inequality_split_pipeline_with_existing_steps(
            Some(plan),
            &[9u8],
            |_eq| Ok::<_, ()>((SolutionSet::AllReals, vec![1u8])),
            |_solved| SolutionSet::AllReals,
        )
        .expect("plan is present");

        let (final_set, steps) = out.expect("execution should succeed");
        assert!(matches!(final_set, SolutionSet::AllReals));
        assert_eq!(steps, vec![1u8, 1u8, 1u8, 1u8, 9u8]);
    }

    #[test]
    fn derive_pow_isolation_route_detects_variable_in_base() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        assert_eq!(
            derive_pow_isolation_route(&ctx, x, "x"),
            PowIsolationRoute::VariableInBase
        );
    }

    #[test]
    fn derive_pow_isolation_route_detects_variable_in_exponent() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        assert_eq!(
            derive_pow_isolation_route(&ctx, two, "x"),
            PowIsolationRoute::VariableInExponent
        );
    }

    #[test]
    fn pow_exponent_rhs_contains_variable_reports_presence() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let rhs = ctx.add(Expr::Add(x, one));
        assert!(pow_exponent_rhs_contains_variable(&ctx, rhs, "x"));
        assert!(!pow_exponent_rhs_contains_variable(&ctx, one, "x"));
    }

    #[test]
    fn derive_add_isolation_route_detects_both_operands() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let left = ctx.add(Expr::Add(x, one));
        let right = ctx.add(Expr::Sub(x, one));
        assert_eq!(
            derive_add_isolation_route(&ctx, left, right, "x"),
            AddIsolationRoute::BothOperands
        );
    }

    #[test]
    fn derive_add_isolation_route_detects_left_operand() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        assert_eq!(
            derive_add_isolation_route(&ctx, x, one, "x"),
            AddIsolationRoute::LeftOperand
        );
    }

    #[test]
    fn derive_add_isolation_route_defaults_to_right_operand() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        assert_eq!(
            derive_add_isolation_route(&ctx, one, x, "x"),
            AddIsolationRoute::RightOperand
        );
        assert_eq!(
            derive_add_isolation_route(&ctx, one, one, "x"),
            AddIsolationRoute::RightOperand
        );
    }

    #[test]
    fn derive_add_isolation_operands_maps_left_route() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let operands = derive_add_isolation_operands(&ctx, x, one, "x");
        assert_eq!(operands.route, AddIsolationRoute::LeftOperand);
        assert_eq!(operands.isolated_addend, x);
        assert_eq!(operands.moved_addend, one);
    }

    #[test]
    fn derive_add_isolation_operands_maps_right_route() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let operands = derive_add_isolation_operands(&ctx, one, x, "x");
        assert_eq!(operands.route, AddIsolationRoute::RightOperand);
        assert_eq!(operands.isolated_addend, x);
        assert_eq!(operands.moved_addend, one);
    }

    #[test]
    fn derive_add_isolation_operands_maps_both_route_to_left_as_isolated() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let left = ctx.add(Expr::Add(x, one));
        let right = ctx.add(Expr::Sub(x, one));
        let operands = derive_add_isolation_operands(&ctx, left, right, "x");
        assert_eq!(operands.route, AddIsolationRoute::BothOperands);
        assert_eq!(operands.isolated_addend, left);
        assert_eq!(operands.moved_addend, right);
    }

    #[test]
    fn resolve_isolated_variable_outcome_reports_circular_rhs() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let rhs = ctx.add(Expr::Add(x, one));
        assert!(matches!(
            resolve_isolated_variable_outcome(&mut ctx, rhs, RelOp::Eq, "x"),
            IsolatedVariableOutcome::ContainsTargetVariable
        ));
    }

    #[test]
    fn resolve_isolated_variable_outcome_builds_solution_set_when_rhs_is_var_free() {
        let mut ctx = Context::new();
        let three = ctx.num(3);
        let out = resolve_isolated_variable_outcome(&mut ctx, three, RelOp::Eq, "x");
        match out {
            IsolatedVariableOutcome::Solved(SolutionSet::Discrete(values)) => {
                assert_eq!(values, vec![three]);
            }
            other => panic!("unexpected isolated outcome: {:?}", other),
        }
    }

    #[test]
    fn resolve_circular_isolated_outcome_prefers_first_strategy() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let out = resolve_circular_isolated_outcome_with(
            one,
            two,
            "x",
            |_, _, _| Some((SolutionSet::Discrete(vec![one]), vec!["s1"])),
            |_, _, _| Some((SolutionSet::Discrete(vec![two]), vec!["s2"])),
            |_, _, _| SolutionSet::Empty,
        );
        match out {
            CircularIsolatedOutcome::Solved {
                solution_set: SolutionSet::Discrete(values),
                steps,
            } => {
                assert_eq!(values, vec![one]);
                assert_eq!(steps, vec!["s1"]);
            }
            other => panic!("unexpected circular isolated outcome: {:?}", other),
        }
    }

    #[test]
    fn resolve_circular_isolated_outcome_uses_second_strategy_when_first_misses() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let mut first_calls = 0usize;
        let mut second_calls = 0usize;
        let out = resolve_circular_isolated_outcome_with(
            one,
            two,
            "x",
            |_, _, _| {
                first_calls += 1;
                None
            },
            |_, _, _| {
                second_calls += 1;
                Some((SolutionSet::Discrete(vec![two]), vec!["s2"]))
            },
            |_, _, _| SolutionSet::Empty,
        );
        assert_eq!(first_calls, 1);
        assert_eq!(second_calls, 1);
        match out {
            CircularIsolatedOutcome::Solved {
                solution_set: SolutionSet::Discrete(values),
                steps,
            } => {
                assert_eq!(values, vec![two]);
                assert_eq!(steps, vec!["s2"]);
            }
            other => panic!("unexpected circular isolated outcome: {:?}", other),
        }
    }

    #[test]
    fn resolve_circular_isolated_outcome_falls_back_to_residual() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let out = resolve_circular_isolated_outcome_with(
            one,
            two,
            "x",
            |_, _, _| None::<(SolutionSet, Vec<()>)>,
            |_, _, _| None::<(SolutionSet, Vec<()>)>,
            |_, _, _| SolutionSet::Residual(two),
        );
        assert!(matches!(
            out,
            CircularIsolatedOutcome::Residual(SolutionSet::Residual(id)) if id == two
        ));
    }

    #[test]
    fn solve_isolated_variable_lhs_with_returns_direct_solution_for_var_free_rhs() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let three = ctx.num(3);
        let mut simplify_calls = 0usize;
        let (set, steps) = solve_isolated_variable_lhs_with(
            &mut ctx,
            x,
            three,
            RelOp::Eq,
            "x",
            |rhs| {
                simplify_calls += 1;
                rhs
            },
            |_, _, _| None::<(SolutionSet, Vec<&'static str>)>,
            |_, _, _| None::<(SolutionSet, Vec<&'static str>)>,
            |_, _, _| SolutionSet::Empty,
        );

        assert_eq!(simplify_calls, 1);
        assert!(steps.is_empty());
        assert_eq!(set, SolutionSet::Discrete(vec![three]));
    }

    #[test]
    fn solve_isolated_variable_lhs_with_resolver_and_merge_with_existing_steps_appends_existing() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let rhs = ctx.num(3);

        let merged = solve_isolated_variable_lhs_with_resolver_and_merge_with_existing_steps(
            x,
            rhs,
            RelOp::Eq,
            "x",
            vec!["existing".to_string()],
            |sim_rhs, rel_op, solve_var| {
                resolve_isolated_variable_outcome(&mut ctx, sim_rhs, rel_op, solve_var)
            },
            |value| value,
            |_, _, _| None::<(SolutionSet, Vec<String>)>,
            |_, _, _| None::<(SolutionSet, Vec<String>)>,
            |_, _, _| SolutionSet::Empty,
        );

        assert_eq!(merged.0, SolutionSet::Discrete(vec![rhs]));
        assert_eq!(merged.1, vec!["existing".to_string()]);
    }

    #[test]
    fn execute_log_isolation_result_pipeline_or_else_with_and_merge_with_existing_steps_with_prepends(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let rhs = ctx.var("rhs");
        let rewrite_eq = Equation {
            lhs: x,
            rhs,
            op: RelOp::Eq,
        };
        let rewrite = crate::log_isolation::LogIsolationRewritePlan {
            equation: rewrite_eq.clone(),
            items: vec![crate::log_isolation::LogIsolationExecutionItem {
                equation: rewrite_eq.clone(),
                description: "Take log".to_string(),
            }],
        };

        let merged =
            execute_log_isolation_result_pipeline_or_else_with_and_merge_with_existing_steps_with(
                true,
                vec!["existing".to_string()],
                || Some(rewrite.clone()),
                |equation| {
                    assert_eq!(equation, &rewrite_eq);
                    Ok::<_, String>((SolutionSet::Discrete(vec![x]), vec!["sub".to_string()]))
                },
                |item| item.description().to_string(),
                || "not-plannable".to_string(),
            )
            .expect("log merge wrapper should solve");

        assert_eq!(merged.0, SolutionSet::Discrete(vec![x]));
        assert_eq!(
            merged.1,
            vec![
                "Take log".to_string(),
                "sub".to_string(),
                "existing".to_string()
            ]
        );
    }

    #[test]
    fn execute_unary_inverse_result_pipeline_or_else_with_and_merge_with_existing_steps_with_prepends(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let rhs = ctx.var("rhs");

        let merged =
            execute_unary_inverse_result_pipeline_or_else_with_and_merge_with_existing_steps_with(
                "sqrt",
                x,
                rhs,
                RelOp::Eq,
                true,
                true,
                vec!["existing".to_string()],
                |fn_name, arg, other, op, is_lhs| {
                    crate::function_inverse::plan_unary_inverse_isolation_step(
                        &mut ctx, fn_name, arg, other, op, is_lhs,
                    )
                },
                |rhs_expr| (rhs_expr, Vec::<(String, ExprId)>::new()),
                |solve_lhs, solve_rhs, solve_op| {
                    assert_eq!(solve_lhs, x);
                    assert_eq!(solve_op, RelOp::Eq);
                    Ok::<_, String>((
                        SolutionSet::Discrete(vec![solve_rhs]),
                        vec!["sub".to_string()],
                    ))
                },
                |item| item.description().to_string(),
                || "unsupported".to_string(),
            )
            .expect("unary merge wrapper should solve");

        assert_eq!(
            merged.1,
            vec![
                "Square both sides".to_string(),
                "sub".to_string(),
                "existing".to_string()
            ]
        );
    }

    #[test]
    fn solve_isolated_variable_lhs_with_prefers_circular_linear_collect() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let rhs = ctx.add(Expr::Add(x, one));
        let mut simplify_calls = 0usize;
        let (set, steps) = solve_isolated_variable_lhs_with(
            &mut ctx,
            x,
            rhs,
            RelOp::Eq,
            "x",
            |rhs| {
                simplify_calls += 1;
                rhs
            },
            |_, _, _| Some((SolutionSet::Discrete(vec![two]), vec!["lc1"])),
            |_, _, _| Some((SolutionSet::Discrete(vec![one]), vec!["lc2"])),
            |_, _, _| SolutionSet::Residual(one),
        );

        assert_eq!(simplify_calls, 1);
        assert_eq!(set, SolutionSet::Discrete(vec![two]));
        assert_eq!(steps, vec!["lc1"]);
    }

    #[test]
    fn solve_isolated_variable_lhs_with_falls_back_to_residual_without_steps() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let rhs = ctx.add(Expr::Add(x, one));
        let mut simplify_calls = 0usize;
        let (set, steps) = solve_isolated_variable_lhs_with(
            &mut ctx,
            x,
            rhs,
            RelOp::Eq,
            "x",
            |rhs| {
                simplify_calls += 1;
                rhs
            },
            |_, _, _| None::<(SolutionSet, Vec<&'static str>)>,
            |_, _, _| None::<(SolutionSet, Vec<&'static str>)>,
            |_, _, _| SolutionSet::Residual(one),
        );

        assert_eq!(simplify_calls, 1);
        assert_eq!(set, SolutionSet::Residual(one));
        assert!(steps.is_empty());
    }

    #[test]
    fn derive_sub_isolation_route_detects_minuend_variable() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        assert_eq!(
            derive_sub_isolation_route(&ctx, x, "x"),
            SubIsolationRoute::Minuend
        );
    }

    #[test]
    fn derive_sub_isolation_route_defaults_to_subtrahend() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        assert_eq!(
            derive_sub_isolation_route(&ctx, one, "x"),
            SubIsolationRoute::Subtrahend
        );
    }

    #[test]
    fn derive_sub_isolation_operands_maps_minuend_route() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let operands = derive_sub_isolation_operands(&ctx, x, one, "x");
        assert_eq!(operands.route, SubIsolationRoute::Minuend);
        assert_eq!(operands.isolated_term, x);
        assert_eq!(operands.moved_term, one);
    }

    #[test]
    fn derive_sub_isolation_operands_maps_subtrahend_route() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let operands = derive_sub_isolation_operands(&ctx, one, x, "x");
        assert_eq!(operands.route, SubIsolationRoute::Subtrahend);
        assert_eq!(operands.isolated_term, x);
        assert_eq!(operands.moved_term, one);
    }

    #[test]
    fn derive_mul_isolation_route_detects_left_factor_variable() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        assert_eq!(
            derive_mul_isolation_route(&ctx, x, "x"),
            MulIsolationRoute::LeftFactor
        );
    }

    #[test]
    fn derive_mul_isolation_route_defaults_to_right_factor() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        assert_eq!(
            derive_mul_isolation_route(&ctx, two, "x"),
            MulIsolationRoute::RightFactor
        );
    }

    #[test]
    fn derive_mul_isolation_operands_maps_left_factor_route() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let operands = derive_mul_isolation_operands(&ctx, x, two, "x");
        assert_eq!(operands.route, MulIsolationRoute::LeftFactor);
        assert_eq!(operands.isolated_factor, x);
        assert_eq!(operands.moved_factor, two);
    }

    #[test]
    fn derive_mul_isolation_operands_maps_right_factor_route() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let operands = derive_mul_isolation_operands(&ctx, two, x, "x");
        assert_eq!(operands.route, MulIsolationRoute::RightFactor);
        assert_eq!(operands.isolated_factor, x);
        assert_eq!(operands.moved_factor, two);
    }

    #[test]
    fn mul_rhs_contains_variable_reports_presence() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let rhs = ctx.add(Expr::Add(x, one));
        assert!(mul_rhs_contains_variable(&ctx, rhs, "x"));
        assert!(!mul_rhs_contains_variable(&ctx, one, "x"));
    }

    #[test]
    fn derive_div_isolation_route_detects_variable_in_numerator() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        assert_eq!(
            derive_div_isolation_route(&ctx, x, "x"),
            DivIsolationRoute::VariableInNumerator
        );
    }

    #[test]
    fn derive_div_isolation_route_defaults_to_variable_in_denominator() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        assert_eq!(
            derive_div_isolation_route(&ctx, one, "x"),
            DivIsolationRoute::VariableInDenominator
        );
    }
}
