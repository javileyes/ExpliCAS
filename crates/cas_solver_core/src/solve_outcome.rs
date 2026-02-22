use crate::isolation_utils::{mk_residual_solve, NumericSign};
use crate::log_domain::{
    assumptions_to_condition_set, classify_log_unsupported_route, classify_terminal_action,
    DomainModeKind, LogAssumption, LogSolveDecision, LogTerminalAction, LogUnsupportedRoute,
};
use crate::solution_set::open_positive_domain;
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

/// Generic terminal solve outcome (message + solution set).
#[derive(Debug, Clone, PartialEq)]
pub struct TerminalSolveOutcome {
    pub message: &'static str,
    pub solutions: SolutionSet,
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

/// Check whether two bases are equivalent for power-exponent shortcuts.
///
/// `equivalent_nontrivial` is only evaluated when the bases are not identical.
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

/// Standard narration when solve-tactic normalization rewrites `base^x = rhs`
/// before logarithm isolation in Assume mode.
pub const SOLVE_TACTIC_NORMALIZATION_MESSAGE: &str =
    "Applied SolveTactic normalization (Assume mode) to enable logarithm isolation";

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

/// Build narration for eliminating rational exponents by powering both sides.
pub fn eliminate_fractional_exponent_message(q_display: &str) -> String {
    format!(
        "Raise both sides to power {} to eliminate fractional exponent",
        q_display
    )
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

/// Pre-built sign-split equations for inequalities of the form `A*B op 0`.
#[derive(Debug, Clone, PartialEq)]
pub struct ProductZeroInequalityPlan {
    pub case1_left: Equation,
    pub case1_right: Equation,
    pub case2_left: Equation,
    pub case2_right: Equation,
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

/// Pre-built didactic equations for denominator-isolation rewrite:
/// `num / den op rhs` -> `num op rhs*den` -> `den op num/rhs`.
#[derive(Debug, Clone, PartialEq)]
pub struct DivisionDenominatorDidacticPlan {
    pub multiply_equation: Equation,
    pub divide_equation: Equation,
    pub multiply_by: ExprId,
    pub divide_by: ExprId,
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
}
