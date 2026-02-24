//! Unified unwrap rewrite planning for solver strategies.

use crate::isolation_utils::contains_var;
use crate::log_domain::{DomainModeKind, LogAssumption, LogSolveDecision};
use crate::rational_power::PowUnwrapPlan;
use crate::solve_outcome::{
    resolve_single_side_exponential_terminal_with_item, take_log_base_message,
    SingleSideExponentialTerminalSolved, TermIsolationExecutionItem,
};
use cas_ast::{Context, Equation, Expr, ExprId, RelOp, SolutionSet};

/// Planned unwrap rewrite from a target expression.
#[derive(Debug, Clone, PartialEq)]
pub enum UnwrapRewritePlan {
    Unary {
        equation: Equation,
        description: String,
    },
    PowVariableBase {
        equation: Equation,
        exponent: ExprId,
    },
    PowLogLinear {
        equation: Equation,
        base: ExprId,
        assumptions: Vec<LogAssumption>,
    },
}

/// Input bundle for planning/executing one unwrap rewrite.
#[derive(Debug, Clone, PartialEq)]
pub struct UnwrapExecutionRequest<'a> {
    pub target: ExprId,
    pub other: ExprId,
    pub var: &'a str,
    pub op: RelOp,
    pub is_lhs: bool,
}

/// Solver-ready unwrap step derived from a rewrite plan.
#[derive(Debug, Clone, PartialEq)]
pub struct UnwrapExecutionPlan {
    pub equation: Equation,
    pub description: String,
    pub assumptions: Vec<LogAssumption>,
    pub log_linear_base: Option<ExprId>,
    pub items: Vec<UnwrapExecutionItem>,
}

/// One concrete log-domain assumption emitted by an unwrap execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogLinearAssumptionRecord {
    pub assumption: LogAssumption,
    pub base: ExprId,
    pub other_side: ExprId,
}

/// One executable unwrap item aligned with a didactic payload.
#[derive(Debug, Clone, PartialEq)]
pub struct UnwrapExecutionItem {
    pub equation: Equation,
    pub description: String,
}

impl UnwrapExecutionItem {
    /// User-facing narration for this execution item.
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Didactic payload for a planned unwrap rewrite step.
#[derive(Debug, Clone, PartialEq)]
pub struct UnwrapDidacticStep {
    pub description: String,
    pub equation_after: Equation,
}

/// Build narration for variable-base power unwrap rewrites.
pub fn pow_variable_base_unwrap_message(exponent_display: &str) -> String {
    format!("Raise both sides to 1/{}", exponent_display)
}

/// Convert a rewrite plan into an executable step payload.
///
/// `render_expr` is used only for variable-base power rewrites so callers
/// can control display format (ASCII, LaTeX, etc.).
pub fn build_unwrap_execution_plan_with<F>(
    plan: UnwrapRewritePlan,
    mut render_expr: F,
) -> UnwrapExecutionPlan
where
    F: FnMut(ExprId) -> String,
{
    match plan {
        UnwrapRewritePlan::Unary {
            equation,
            description,
        } => {
            let items = vec![UnwrapExecutionItem {
                equation: equation.clone(),
                description: description.clone(),
            }];
            UnwrapExecutionPlan {
                equation,
                description,
                assumptions: vec![],
                log_linear_base: None,
                items,
            }
        }
        UnwrapRewritePlan::PowVariableBase { equation, exponent } => {
            let description = pow_variable_base_unwrap_message(&render_expr(exponent));
            let items = vec![UnwrapExecutionItem {
                equation: equation.clone(),
                description: description.clone(),
            }];
            UnwrapExecutionPlan {
                equation,
                description,
                assumptions: vec![],
                log_linear_base: None,
                items,
            }
        }
        UnwrapRewritePlan::PowLogLinear {
            equation,
            base,
            assumptions,
        } => {
            let description = take_log_base_message("e");
            let items = vec![UnwrapExecutionItem {
                equation: equation.clone(),
                description: description.clone(),
            }];
            UnwrapExecutionPlan {
                equation,
                description,
                assumptions,
                log_linear_base: Some(base),
                items,
            }
        }
    }
}

/// Plan and build an unwrap execution in one step.
///
/// Returns `None` when no unwrap rewrite is applicable for `request.target`.
pub fn plan_unwrap_execution_with<FClassify, FRender>(
    ctx: &mut Context,
    request: UnwrapExecutionRequest<'_>,
    mut classify_log_solve: FClassify,
    mut render_expr: FRender,
) -> Option<UnwrapExecutionPlan>
where
    FClassify: FnMut(&Context, ExprId, ExprId) -> LogSolveDecision,
    FRender: FnMut(&Context, ExprId) -> String,
{
    let rewrite = plan_unwrap_rewrite(
        ctx,
        request.target,
        request.other,
        request.var,
        request.op,
        request.is_lhs,
        |core_ctx, base, other_side| classify_log_solve(core_ctx, base, other_side),
    )?;
    let view_ctx: &Context = ctx;
    Some(build_unwrap_execution_plan_with(rewrite, |id| {
        render_expr(view_ctx, id)
    }))
}

/// Runtime contract for unwrap planning (classification + rendering).
pub trait UnwrapPlanRuntime {
    /// Classify whether log-based isolation can proceed for `(base, other_side)`.
    fn classify_log_solve(
        &mut self,
        ctx: &Context,
        base: ExprId,
        other_side: ExprId,
    ) -> LogSolveDecision;
    /// Render expression IDs for didactic narration.
    fn render_expr(&mut self, ctx: &Context, expr: ExprId) -> String;
}

/// Runtime contract for unwrap entry routing (planning + terminal step mapping).
pub trait UnwrapEntryRuntime<S>: UnwrapPlanRuntime {
    /// Convert one terminal execution item into caller-owned step payload.
    fn map_terminal_item_to_step(&mut self, item: TermIsolationExecutionItem) -> S;
}

/// Runtime-based unwrap planning helper.
///
/// Returns `None` when no unwrap rewrite is applicable for `request.target`.
pub fn plan_unwrap_execution_with_runtime<R>(
    ctx: &mut Context,
    request: UnwrapExecutionRequest<'_>,
    runtime: &mut R,
) -> Option<UnwrapExecutionPlan>
where
    R: UnwrapPlanRuntime,
{
    let rewrite = plan_unwrap_rewrite(
        ctx,
        request.target,
        request.other,
        request.var,
        request.op,
        request.is_lhs,
        |core_ctx, base, other_side| runtime.classify_log_solve(core_ctx, base, other_side),
    )?;
    let view_ctx: &Context = ctx;
    Some(build_unwrap_execution_plan_with(rewrite, |id| {
        runtime.render_expr(view_ctx, id)
    }))
}

/// One selected unwrap execution candidate for an equation side.
#[derive(Debug, Clone, PartialEq)]
pub struct UnwrapEquationExecution {
    pub execution: UnwrapExecutionPlan,
    pub other_side: ExprId,
}

/// Entry routing for `UnwrapStrategy`.
#[derive(Debug, Clone, PartialEq)]
pub enum UnwrapEntryRouting<S> {
    Terminal(SingleSideExponentialTerminalSolved<S>),
    Execution(UnwrapEquationExecution),
}

/// Plan the first applicable unwrap execution for an equation.
///
/// Selection order is deterministic: LHS is attempted first, then RHS.
pub fn plan_first_unwrap_equation_execution_with<FClassify, FRender>(
    ctx: &mut Context,
    equation: &Equation,
    var: &str,
    lhs_has: bool,
    rhs_has: bool,
    mut classify_log_solve: FClassify,
    mut render_expr: FRender,
) -> Option<UnwrapEquationExecution>
where
    FClassify: FnMut(&Context, ExprId, ExprId) -> LogSolveDecision,
    FRender: FnMut(&Context, ExprId) -> String,
{
    if lhs_has {
        if let Some(execution) = plan_unwrap_execution_with(
            ctx,
            UnwrapExecutionRequest {
                target: equation.lhs,
                other: equation.rhs,
                var,
                op: equation.op.clone(),
                is_lhs: true,
            },
            |core_ctx, base, other_side| classify_log_solve(core_ctx, base, other_side),
            |core_ctx, expr| render_expr(core_ctx, expr),
        ) {
            return Some(UnwrapEquationExecution {
                execution,
                other_side: equation.rhs,
            });
        }
    }
    if rhs_has {
        if let Some(execution) = plan_unwrap_execution_with(
            ctx,
            UnwrapExecutionRequest {
                target: equation.rhs,
                other: equation.lhs,
                var,
                op: equation.op.clone(),
                is_lhs: false,
            },
            |core_ctx, base, other_side| classify_log_solve(core_ctx, base, other_side),
            |core_ctx, expr| render_expr(core_ctx, expr),
        ) {
            return Some(UnwrapEquationExecution {
                execution,
                other_side: equation.lhs,
            });
        }
    }
    None
}

/// Runtime-based variant of unwrap candidate selection.
///
/// Selection order is deterministic: LHS is attempted first, then RHS.
pub fn plan_first_unwrap_equation_execution_with_runtime<R>(
    ctx: &mut Context,
    equation: &Equation,
    var: &str,
    lhs_has: bool,
    rhs_has: bool,
    runtime: &mut R,
) -> Option<UnwrapEquationExecution>
where
    R: UnwrapPlanRuntime,
{
    if lhs_has {
        if let Some(execution) = plan_unwrap_execution_with_runtime(
            ctx,
            UnwrapExecutionRequest {
                target: equation.lhs,
                other: equation.rhs,
                var,
                op: equation.op.clone(),
                is_lhs: true,
            },
            runtime,
        ) {
            return Some(UnwrapEquationExecution {
                execution,
                other_side: equation.rhs,
            });
        }
    }
    if rhs_has {
        if let Some(execution) = plan_unwrap_execution_with_runtime(
            ctx,
            UnwrapExecutionRequest {
                target: equation.rhs,
                other: equation.lhs,
                var,
                op: equation.op.clone(),
                is_lhs: false,
            },
            runtime,
        ) {
            return Some(UnwrapEquationExecution {
                execution,
                other_side: equation.lhs,
            });
        }
    }
    None
}

/// Route unwrap strategy entry:
/// 1) variable presence gate,
/// 2) single-side exponential terminal fast-path,
/// 3) LHS->RHS unwrap execution planning.
#[allow(clippy::too_many_arguments)]
pub fn route_unwrap_entry_with_item<FClassify, FRender, FStep, S>(
    ctx: &mut Context,
    equation: &Equation,
    var: &str,
    mode: DomainModeKind,
    wildcard_scope: bool,
    residual_suffix: &str,
    include_item: bool,
    mut classify_log_solve: FClassify,
    mut render_expr: FRender,
    mut map_terminal_item_to_step: FStep,
) -> Option<UnwrapEntryRouting<S>>
where
    FClassify: FnMut(&Context, ExprId, ExprId) -> LogSolveDecision,
    FRender: FnMut(&Context, ExprId) -> String,
    FStep: FnMut(TermIsolationExecutionItem) -> S,
{
    let lhs_has = contains_var(ctx, equation.lhs, var);
    let rhs_has = contains_var(ctx, equation.rhs, var);
    if !lhs_has && !rhs_has {
        return None;
    }

    if let Some((solution_set, item)) = resolve_single_side_exponential_terminal_with_item(
        ctx,
        equation.lhs,
        equation.rhs,
        var,
        lhs_has,
        rhs_has,
        mode,
        wildcard_scope,
        residual_suffix,
        equation.clone(),
        |core_ctx, base, other_side| classify_log_solve(core_ctx, base, other_side),
    ) {
        let mut steps = Vec::new();
        if include_item {
            steps.push(map_terminal_item_to_step(item));
        }
        return Some(UnwrapEntryRouting::Terminal(
            SingleSideExponentialTerminalSolved {
                solution_set,
                steps,
            },
        ));
    }

    let selected = plan_first_unwrap_equation_execution_with(
        ctx,
        equation,
        var,
        lhs_has,
        rhs_has,
        |core_ctx, base, other_side| classify_log_solve(core_ctx, base, other_side),
        |core_ctx, expr| render_expr(core_ctx, expr),
    )?;
    Some(UnwrapEntryRouting::Execution(selected))
}

/// Runtime-based entry routing for `UnwrapStrategy`.
///
/// This variant avoids closure wiring at call sites and delegates
/// classification/rendering/mapping to a runtime adapter.
#[allow(clippy::too_many_arguments)]
pub fn route_unwrap_entry_with_item_runtime<R, S>(
    ctx: &mut Context,
    equation: &Equation,
    var: &str,
    mode: DomainModeKind,
    wildcard_scope: bool,
    residual_suffix: &str,
    include_item: bool,
    runtime: &mut R,
) -> Option<UnwrapEntryRouting<S>>
where
    R: UnwrapEntryRuntime<S>,
{
    let lhs_has = contains_var(ctx, equation.lhs, var);
    let rhs_has = contains_var(ctx, equation.rhs, var);
    if !lhs_has && !rhs_has {
        return None;
    }

    if let Some((solution_set, item)) = resolve_single_side_exponential_terminal_with_item(
        ctx,
        equation.lhs,
        equation.rhs,
        var,
        lhs_has,
        rhs_has,
        mode,
        wildcard_scope,
        residual_suffix,
        equation.clone(),
        |core_ctx, base, other_side| runtime.classify_log_solve(core_ctx, base, other_side),
    ) {
        let mut steps = Vec::new();
        if include_item {
            steps.push(runtime.map_terminal_item_to_step(item));
        }
        return Some(UnwrapEntryRouting::Terminal(
            SingleSideExponentialTerminalSolved {
                solution_set,
                steps,
            },
        ));
    }

    let selected = plan_first_unwrap_equation_execution_with_runtime(
        ctx, equation, var, lhs_has, rhs_has, runtime,
    )?;
    Some(UnwrapEntryRouting::Execution(selected))
}

/// Collect didactic steps emitted by an unwrap execution in display order.
pub fn collect_unwrap_execution_didactic_steps(
    execution: &UnwrapExecutionPlan,
) -> Vec<UnwrapDidacticStep> {
    execution
        .items
        .iter()
        .cloned()
        .map(|item| UnwrapDidacticStep {
            description: item.description,
            equation_after: item.equation,
        })
        .collect()
}

/// Collect unwrap execution items in display order.
pub fn collect_unwrap_execution_items(execution: &UnwrapExecutionPlan) -> Vec<UnwrapExecutionItem> {
    execution.items.clone()
}

/// Return the first unwrap execution item, if any.
pub fn first_unwrap_execution_item(execution: &UnwrapExecutionPlan) -> Option<UnwrapExecutionItem> {
    collect_unwrap_execution_items(execution).into_iter().next()
}

/// Collect log-linear assumption records for engine-side reporting.
pub fn collect_log_linear_assumption_records(
    execution: &UnwrapExecutionPlan,
    other_side: ExprId,
) -> Vec<LogLinearAssumptionRecord> {
    let Some(base) = execution.log_linear_base else {
        return vec![];
    };
    execution
        .assumptions
        .iter()
        .copied()
        .map(|assumption| LogLinearAssumptionRecord {
            assumption,
            base,
            other_side,
        })
        .collect()
}

/// Solved payload for one unwrap execution.
#[derive(Debug, Clone, PartialEq)]
pub struct UnwrapSolvedExecution<T> {
    pub execution: UnwrapExecutionPlan,
    pub assumption_records: Vec<LogLinearAssumptionRecord>,
    pub rewritten_equation: Equation,
    pub solved: T,
}

/// Resolve the equation to continue solving after unwrap execution.
///
/// Prefer the last emitted execution item equation (most concrete rewrite),
/// and fall back to `execution.equation` when item payload is absent.
pub fn unwrap_rewritten_equation(execution: &UnwrapExecutionPlan) -> Equation {
    execution
        .items
        .last()
        .map(|item| item.equation.clone())
        .unwrap_or_else(|| execution.equation.clone())
}

/// Execute a planned unwrap rewrite with caller-provided assumption handling
/// and recursive solve callback.
pub fn solve_unwrap_execution_with<E, T, FAssumption, FSolve>(
    execution: UnwrapExecutionPlan,
    other_side: ExprId,
    mut on_assumption: FAssumption,
    mut solve_equation: FSolve,
) -> Result<UnwrapSolvedExecution<T>, E>
where
    FAssumption: FnMut(LogLinearAssumptionRecord),
    FSolve: FnMut(&Equation) -> Result<T, E>,
{
    let assumption_records = collect_log_linear_assumption_records(&execution, other_side);
    for record in assumption_records.iter().copied() {
        on_assumption(record);
    }
    let rewritten_equation = unwrap_rewritten_equation(&execution);
    let solved = solve_equation(&rewritten_equation)?;
    Ok(UnwrapSolvedExecution {
        execution,
        assumption_records,
        rewritten_equation,
        solved,
    })
}

/// Execute a planned unwrap rewrite while passing the aligned optional
/// execution item to the solve callback.
pub fn solve_unwrap_execution_with_item<E, T, FAssumption, FSolve>(
    execution: UnwrapExecutionPlan,
    other_side: ExprId,
    mut on_assumption: FAssumption,
    mut solve_equation: FSolve,
) -> Result<UnwrapSolvedExecution<T>, E>
where
    FAssumption: FnMut(LogLinearAssumptionRecord),
    FSolve: FnMut(Option<UnwrapExecutionItem>, &Equation) -> Result<T, E>,
{
    let assumption_records = collect_log_linear_assumption_records(&execution, other_side);
    for record in assumption_records.iter().copied() {
        on_assumption(record);
    }
    let rewritten_equation = unwrap_rewritten_equation(&execution);
    let item = first_unwrap_execution_item(&execution);
    let solved = solve_equation(item, &rewritten_equation)?;
    Ok(UnwrapSolvedExecution {
        execution,
        assumption_records,
        rewritten_equation,
        solved,
    })
}

/// Runtime contract for unwrap execution orchestration used by engine adapters.
pub trait UnwrapExecutionRuntime<E, S> {
    /// Handle one assumption record emitted by log-linear unwrap planning.
    fn note_assumption(&mut self, record: LogLinearAssumptionRecord);
    /// Solve one rewritten equation for the requested variable.
    fn solve_equation(
        &mut self,
        equation: &Equation,
        var: &str,
    ) -> Result<(SolutionSet, Vec<S>), E>;
    /// Materialize one caller step payload from unwrap execution item.
    fn map_item_to_step(&mut self, item: UnwrapExecutionItem) -> S;
}

/// Solved output for unwrap execution pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct UnwrapExecutionPipelineSolved<S> {
    pub solution_set: SolutionSet,
    pub steps: Vec<S>,
}

/// Execute unwrap pipeline with optional first-item step dispatch and recursive solve.
pub fn solve_unwrap_execution_pipeline_with_item<E, S, R>(
    execution: UnwrapExecutionPlan,
    other_side: ExprId,
    var: &str,
    include_item: bool,
    runtime: &mut R,
) -> Result<UnwrapExecutionPipelineSolved<S>, E>
where
    R: UnwrapExecutionRuntime<E, S>,
{
    let assumption_records = collect_log_linear_assumption_records(&execution, other_side);
    for record in assumption_records.iter().copied() {
        runtime.note_assumption(record);
    }
    let rewritten_equation = unwrap_rewritten_equation(&execution);
    let mut steps = Vec::new();
    if include_item {
        if let Some(item) = first_unwrap_execution_item(&execution) {
            steps.push(runtime.map_item_to_step(item));
        }
    }
    let (solution_set, mut sub_steps) = runtime.solve_equation(&rewritten_equation, var)?;
    steps.append(&mut sub_steps);
    Ok(UnwrapExecutionPipelineSolved {
        solution_set,
        steps,
    })
}

/// Plan unwrap rewrite for a target expression (`Function`/`Pow`).
pub fn plan_unwrap_rewrite<F>(
    ctx: &mut Context,
    target: ExprId,
    other: ExprId,
    var: &str,
    op: RelOp,
    is_lhs: bool,
    mut classify_log_solve: F,
) -> Option<UnwrapRewritePlan>
where
    F: FnMut(&Context, ExprId, ExprId) -> LogSolveDecision,
{
    let target_data = ctx.get(target).clone();
    match target_data {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let arg = args[0];
            let name = ctx.sym_name(fn_id).to_string();
            let (equation, description) =
                crate::function_inverse::plan_unary_inverse_rewrite_for_unwrap(
                    ctx, &name, arg, other, op, is_lhs,
                )?;
            Some(UnwrapRewritePlan::Unary {
                equation,
                description: description.to_string(),
            })
        }
        Expr::Pow(_, _) => match crate::rational_power::plan_pow_unwrap_rewrite(
            ctx,
            target,
            other,
            var,
            op,
            is_lhs,
            |core_ctx, base, other_side| classify_log_solve(core_ctx, base, other_side),
        ) {
            Some(PowUnwrapPlan::VariableBase { equation, exponent }) => {
                Some(UnwrapRewritePlan::PowVariableBase { equation, exponent })
            }
            Some(PowUnwrapPlan::LogLinear {
                equation,
                base,
                assumptions,
            }) => Some(UnwrapRewritePlan::PowLogLinear {
                equation,
                base,
                assumptions,
            }),
            None => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Expr;

    #[test]
    fn plans_unary_unwrap_rewrite() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let target = ctx.call("ln", vec![x]);

        let plan = plan_unwrap_rewrite(&mut ctx, target, y, "x", RelOp::Eq, true, |_, _, _| {
            LogSolveDecision::Ok
        })
        .expect("expected unary plan");

        match plan {
            UnwrapRewritePlan::Unary { .. } => {}
            other => panic!("expected unary plan, got {:?}", other),
        }
    }

    #[test]
    fn plans_pow_variable_base_unwrap_rewrite() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let exponent = ctx.add(Expr::Div(one, two));
        let target = ctx.add(Expr::Pow(x, exponent));

        let plan = plan_unwrap_rewrite(&mut ctx, target, y, "x", RelOp::Eq, true, |_, _, _| {
            LogSolveDecision::Ok
        })
        .expect("expected pow variable-base plan");

        match plan {
            UnwrapRewritePlan::PowVariableBase { .. } => {}
            other => panic!("expected variable-base plan, got {:?}", other),
        }
    }

    #[test]
    fn pow_variable_base_unwrap_message_formats_expected_text() {
        assert_eq!(
            pow_variable_base_unwrap_message("2"),
            "Raise both sides to 1/2"
        );
    }

    #[test]
    fn build_execution_plan_for_variable_base_uses_rendered_exponent() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let exponent = ctx.add(Expr::Div(one, two));
        let equation = Equation {
            lhs: x,
            rhs: y,
            op: RelOp::Eq,
        };

        let plan = build_unwrap_execution_plan_with(
            UnwrapRewritePlan::PowVariableBase { equation, exponent },
            |id| format!("{:?}", ctx.get(id)),
        );
        assert!(plan.description.contains("1/Div"));
        assert_eq!(plan.log_linear_base, None);
        assert!(plan.assumptions.is_empty());
    }

    #[test]
    fn build_execution_plan_for_log_linear_keeps_assumptions() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let equation = Equation {
            lhs: x,
            rhs: y,
            op: RelOp::Eq,
        };
        let plan = build_unwrap_execution_plan_with(
            UnwrapRewritePlan::PowLogLinear {
                equation,
                base: x,
                assumptions: vec![LogAssumption::PositiveBase],
            },
            |_| "unused".to_string(),
        );
        assert_eq!(plan.log_linear_base, Some(x));
        assert_eq!(plan.assumptions, vec![LogAssumption::PositiveBase]);
        assert_eq!(plan.description, "Take log base e of both sides");
    }

    #[test]
    fn plan_unwrap_execution_with_builds_variable_base_execution() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let exponent = ctx.add(Expr::Div(one, two));
        let target = ctx.add(Expr::Pow(x, exponent));

        let execution = plan_unwrap_execution_with(
            &mut ctx,
            UnwrapExecutionRequest {
                target,
                other: y,
                var: "x",
                op: RelOp::Eq,
                is_lhs: true,
            },
            |_, _, _| LogSolveDecision::Ok,
            |_, _| "half".to_string(),
        )
        .expect("expected unwrap execution");

        assert_eq!(execution.description, "Raise both sides to 1/half");
        assert_eq!(execution.items.len(), 1);
        assert_eq!(execution.items[0].equation, execution.equation);
    }

    #[test]
    fn plan_unwrap_execution_with_returns_none_for_non_wrappable_target() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");

        let execution = plan_unwrap_execution_with(
            &mut ctx,
            UnwrapExecutionRequest {
                target: x,
                other: y,
                var: "x",
                op: RelOp::Eq,
                is_lhs: true,
            },
            |_, _, _| LogSolveDecision::Ok,
            |_, _| "unused".to_string(),
        );

        assert!(execution.is_none());
    }

    #[test]
    fn plan_first_unwrap_equation_execution_with_prefers_lhs_candidate() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let lhs_target = ctx.call("ln", vec![x]);
        let rhs_target = ctx.call("sqrt", vec![x]);
        let equation = Equation {
            lhs: lhs_target,
            rhs: rhs_target,
            op: RelOp::Eq,
        };

        let lhs_execution = plan_unwrap_execution_with(
            &mut ctx,
            UnwrapExecutionRequest {
                target: lhs_target,
                other: rhs_target,
                var: "x",
                op: RelOp::Eq,
                is_lhs: true,
            },
            |_, _, _| LogSolveDecision::Ok,
            |_, _| "render".to_string(),
        )
        .expect("expected lhs unwrap execution");

        let selected = plan_first_unwrap_equation_execution_with(
            &mut ctx,
            &equation,
            "x",
            true,
            true,
            |_, _, _| LogSolveDecision::Ok,
            |_, _| "render".to_string(),
        )
        .expect("expected selected unwrap execution");

        assert_eq!(selected.other_side, rhs_target);
        assert_eq!(selected.execution, lhs_execution);
    }

    #[test]
    fn plan_first_unwrap_equation_execution_with_falls_back_to_rhs_candidate() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let rhs_target = ctx.call("ln", vec![x]);
        let equation = Equation {
            lhs: y,
            rhs: rhs_target,
            op: RelOp::Eq,
        };

        let rhs_execution = plan_unwrap_execution_with(
            &mut ctx,
            UnwrapExecutionRequest {
                target: rhs_target,
                other: y,
                var: "x",
                op: RelOp::Eq,
                is_lhs: false,
            },
            |_, _, _| LogSolveDecision::Ok,
            |_, _| "render".to_string(),
        )
        .expect("expected rhs unwrap execution");

        let selected = plan_first_unwrap_equation_execution_with(
            &mut ctx,
            &equation,
            "x",
            true,
            true,
            |_, _, _| LogSolveDecision::Ok,
            |_, _| "render".to_string(),
        )
        .expect("expected selected unwrap execution");

        assert_eq!(selected.other_side, y);
        assert_eq!(selected.execution, rhs_execution);
    }

    #[test]
    fn plan_first_unwrap_equation_execution_with_returns_none_without_candidates() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let equation = Equation {
            lhs: x,
            rhs: y,
            op: RelOp::Eq,
        };

        let selected = plan_first_unwrap_equation_execution_with(
            &mut ctx,
            &equation,
            "x",
            true,
            false,
            |_, _, _| LogSolveDecision::Ok,
            |_, _| "render".to_string(),
        );

        assert!(selected.is_none());
    }

    struct MockUnwrapEntryRuntime;

    impl UnwrapPlanRuntime for MockUnwrapEntryRuntime {
        fn classify_log_solve(
            &mut self,
            _ctx: &Context,
            _base: ExprId,
            _other_side: ExprId,
        ) -> LogSolveDecision {
            LogSolveDecision::Ok
        }

        fn render_expr(&mut self, _ctx: &Context, _expr: ExprId) -> String {
            "render".to_string()
        }
    }

    impl UnwrapEntryRuntime<String> for MockUnwrapEntryRuntime {
        fn map_terminal_item_to_step(&mut self, item: TermIsolationExecutionItem) -> String {
            item.description
        }
    }

    #[test]
    fn route_unwrap_entry_with_item_returns_none_without_variable_presence() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let equation = Equation {
            lhs: x,
            rhs: y,
            op: RelOp::Eq,
        };

        let routed = route_unwrap_entry_with_item(
            &mut ctx,
            &equation,
            "z",
            DomainModeKind::Generic,
            false,
            " (residual)",
            true,
            |_, _, _| LogSolveDecision::Ok,
            |_, _| "render".to_string(),
            |item| item.description,
        );
        assert!(routed.is_none());
    }

    #[test]
    fn route_unwrap_entry_with_item_returns_terminal_when_exponential_is_blocked() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let neg_five = ctx.num(-5);
        let lhs = ctx.add(Expr::Pow(two, x));
        let equation = Equation {
            lhs,
            rhs: neg_five,
            op: RelOp::Eq,
        };

        let routed = route_unwrap_entry_with_item(
            &mut ctx,
            &equation,
            "x",
            DomainModeKind::Generic,
            false,
            " (residual)",
            true,
            |_ctx, _base, _other| LogSolveDecision::EmptySet("no real solutions"),
            |_, _| "render".to_string(),
            |item| item.description,
        )
        .expect("expected unwrap route");

        match routed {
            UnwrapEntryRouting::Terminal(solved) => {
                assert!(matches!(solved.solution_set, SolutionSet::Empty));
                assert_eq!(solved.steps, vec!["no real solutions".to_string()]);
            }
            other => panic!("expected terminal route, got {:?}", other),
        }
    }

    #[test]
    fn route_unwrap_entry_with_item_returns_execution_when_terminal_not_applicable() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let lhs = ctx.call("ln", vec![x]);
        let equation = Equation {
            lhs,
            rhs: y,
            op: RelOp::Eq,
        };

        let expected_execution = plan_unwrap_execution_with(
            &mut ctx,
            UnwrapExecutionRequest {
                target: lhs,
                other: y,
                var: "x",
                op: RelOp::Eq,
                is_lhs: true,
            },
            |_, _, _| LogSolveDecision::Ok,
            |_, _| "render".to_string(),
        )
        .expect("expected unwrap execution");

        let routed = route_unwrap_entry_with_item(
            &mut ctx,
            &equation,
            "x",
            DomainModeKind::Generic,
            false,
            " (residual)",
            true,
            |_, _, _| LogSolveDecision::Ok,
            |_, _| "render".to_string(),
            |item| item.description,
        )
        .expect("expected unwrap route");

        match routed {
            UnwrapEntryRouting::Execution(selected) => {
                assert_eq!(selected.other_side, y);
                assert_eq!(selected.execution, expected_execution);
            }
            other => panic!("expected execution route, got {:?}", other),
        }
    }

    #[test]
    fn route_unwrap_entry_with_item_runtime_returns_execution_when_terminal_not_applicable() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let lhs = ctx.call("ln", vec![x]);
        let equation = Equation {
            lhs,
            rhs: y,
            op: RelOp::Eq,
        };

        let mut runtime = MockUnwrapEntryRuntime;
        let routed = route_unwrap_entry_with_item_runtime(
            &mut ctx,
            &equation,
            "x",
            DomainModeKind::Generic,
            false,
            " (residual)",
            true,
            &mut runtime,
        )
        .expect("expected unwrap route");

        match routed {
            UnwrapEntryRouting::Execution(selected) => {
                assert_eq!(selected.other_side, y);
                assert_eq!(selected.execution.items.len(), 1);
            }
            other => panic!("expected execution route, got {:?}", other),
        }
    }

    #[test]
    fn collect_unwrap_execution_didactic_steps_returns_single_step() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let execution = UnwrapExecutionPlan {
            equation: Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            description: "unwrap".to_string(),
            assumptions: vec![],
            log_linear_base: None,
            items: vec![UnwrapExecutionItem {
                equation: Equation {
                    lhs: x,
                    rhs: y,
                    op: RelOp::Eq,
                },
                description: "unwrap".to_string(),
            }],
        };

        let didactic = collect_unwrap_execution_didactic_steps(&execution);
        assert_eq!(didactic.len(), 1);
        assert_eq!(didactic[0].description, "unwrap");
        assert_eq!(didactic[0].equation_after, execution.equation);
    }

    #[test]
    fn collect_unwrap_execution_items_returns_single_item() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let execution = UnwrapExecutionPlan {
            equation: Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            description: "unwrap".to_string(),
            assumptions: vec![],
            log_linear_base: None,
            items: vec![UnwrapExecutionItem {
                equation: Equation {
                    lhs: x,
                    rhs: y,
                    op: RelOp::Eq,
                },
                description: "unwrap".to_string(),
            }],
        };

        let items = collect_unwrap_execution_items(&execution);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].equation, execution.equation);
        assert_eq!(items[0].description, "unwrap");
    }

    #[test]
    fn first_unwrap_execution_item_returns_single_item() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let execution = UnwrapExecutionPlan {
            equation: Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            description: "unwrap".to_string(),
            assumptions: vec![],
            log_linear_base: None,
            items: vec![UnwrapExecutionItem {
                equation: Equation {
                    lhs: x,
                    rhs: y,
                    op: RelOp::Eq,
                },
                description: "unwrap".to_string(),
            }],
        };

        let item = first_unwrap_execution_item(&execution).expect("expected one unwrap item");
        assert_eq!(item.equation, execution.equation);
        assert_eq!(item.description, "unwrap");
    }

    #[test]
    fn collect_log_linear_assumption_records_returns_records_when_present() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let execution = UnwrapExecutionPlan {
            equation: Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            description: "unwrap".to_string(),
            assumptions: vec![LogAssumption::PositiveBase, LogAssumption::PositiveRhs],
            log_linear_base: Some(x),
            items: vec![],
        };

        let records = collect_log_linear_assumption_records(&execution, y);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].assumption, LogAssumption::PositiveBase);
        assert_eq!(records[0].base, x);
        assert_eq!(records[0].other_side, y);
        assert_eq!(records[1].assumption, LogAssumption::PositiveRhs);
    }

    #[test]
    fn collect_log_linear_assumption_records_is_empty_without_log_linear_base() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let execution = UnwrapExecutionPlan {
            equation: Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            description: "unwrap".to_string(),
            assumptions: vec![LogAssumption::PositiveBase],
            log_linear_base: None,
            items: vec![],
        };

        let records = collect_log_linear_assumption_records(&execution, y);
        assert!(records.is_empty());
    }

    #[test]
    fn unwrap_rewritten_equation_prefers_last_execution_item() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let execution = UnwrapExecutionPlan {
            equation: Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            description: "unwrap".to_string(),
            assumptions: vec![],
            log_linear_base: None,
            items: vec![
                UnwrapExecutionItem {
                    equation: Equation {
                        lhs: y,
                        rhs: z,
                        op: RelOp::Eq,
                    },
                    description: "first".to_string(),
                },
                UnwrapExecutionItem {
                    equation: Equation {
                        lhs: z,
                        rhs: x,
                        op: RelOp::Eq,
                    },
                    description: "last".to_string(),
                },
            ],
        };

        let rewritten = unwrap_rewritten_equation(&execution);
        assert_eq!(rewritten.lhs, z);
        assert_eq!(rewritten.rhs, x);
    }

    #[test]
    fn unwrap_rewritten_equation_falls_back_to_execution_equation() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let execution = UnwrapExecutionPlan {
            equation: Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            description: "unwrap".to_string(),
            assumptions: vec![],
            log_linear_base: None,
            items: vec![],
        };

        let rewritten = unwrap_rewritten_equation(&execution);
        assert_eq!(rewritten, execution.equation);
    }

    #[test]
    fn solve_unwrap_execution_with_emits_assumptions_and_solves_last_equation() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let execution = UnwrapExecutionPlan {
            equation: Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            description: "unwrap".to_string(),
            assumptions: vec![LogAssumption::PositiveBase],
            log_linear_base: Some(x),
            items: vec![
                UnwrapExecutionItem {
                    equation: Equation {
                        lhs: y,
                        rhs: z,
                        op: RelOp::Eq,
                    },
                    description: "first".to_string(),
                },
                UnwrapExecutionItem {
                    equation: Equation {
                        lhs: z,
                        rhs: x,
                        op: RelOp::Eq,
                    },
                    description: "last".to_string(),
                },
            ],
        };

        let mut seen = Vec::new();
        let solved = solve_unwrap_execution_with(
            execution,
            y,
            |record| seen.push(record),
            |equation| {
                assert_eq!(equation.lhs, z);
                assert_eq!(equation.rhs, x);
                Ok::<_, ()>("ok")
            },
        )
        .expect("solve should succeed");

        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].assumption, LogAssumption::PositiveBase);
        assert_eq!(seen[0].base, x);
        assert_eq!(seen[0].other_side, y);
        assert_eq!(solved.solved, "ok");
        assert_eq!(solved.assumption_records, seen);
    }

    #[test]
    fn solve_unwrap_execution_with_item_passes_first_item_and_solves_last_equation() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let execution = UnwrapExecutionPlan {
            equation: Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            description: "unwrap".to_string(),
            assumptions: vec![LogAssumption::PositiveBase],
            log_linear_base: Some(x),
            items: vec![
                UnwrapExecutionItem {
                    equation: Equation {
                        lhs: y,
                        rhs: z,
                        op: RelOp::Eq,
                    },
                    description: "first".to_string(),
                },
                UnwrapExecutionItem {
                    equation: Equation {
                        lhs: z,
                        rhs: x,
                        op: RelOp::Eq,
                    },
                    description: "last".to_string(),
                },
            ],
        };

        let mut seen_item = None;
        let solved = solve_unwrap_execution_with_item(
            execution,
            y,
            |_record| {},
            |item, equation| {
                seen_item = item.map(|entry| entry.description);
                assert_eq!(equation.lhs, z);
                assert_eq!(equation.rhs, x);
                Ok::<_, ()>("ok")
            },
        )
        .expect("solve should succeed");

        assert_eq!(seen_item, Some("first".to_string()));
        assert_eq!(solved.solved, "ok");
    }

    struct MockUnwrapPipelineRuntime {
        assumptions: Vec<LogLinearAssumptionRecord>,
        solve_calls: Vec<String>,
        solve_set: SolutionSet,
        solve_steps: Vec<String>,
    }

    impl UnwrapExecutionRuntime<(), String> for MockUnwrapPipelineRuntime {
        fn note_assumption(&mut self, record: LogLinearAssumptionRecord) {
            self.assumptions.push(record);
        }

        fn solve_equation(
            &mut self,
            _equation: &Equation,
            var: &str,
        ) -> Result<(SolutionSet, Vec<String>), ()> {
            self.solve_calls.push(var.to_string());
            Ok((self.solve_set.clone(), self.solve_steps.clone()))
        }

        fn map_item_to_step(&mut self, item: UnwrapExecutionItem) -> String {
            item.description
        }
    }

    #[test]
    fn solve_unwrap_execution_pipeline_with_item_forwards_assumptions_item_and_substeps() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let execution = UnwrapExecutionPlan {
            equation: Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            description: "unwrap".to_string(),
            assumptions: vec![LogAssumption::PositiveBase],
            log_linear_base: Some(x),
            items: vec![
                UnwrapExecutionItem {
                    equation: Equation {
                        lhs: y,
                        rhs: z,
                        op: RelOp::Eq,
                    },
                    description: "first".to_string(),
                },
                UnwrapExecutionItem {
                    equation: Equation {
                        lhs: z,
                        rhs: x,
                        op: RelOp::Eq,
                    },
                    description: "last".to_string(),
                },
            ],
        };

        let mut runtime = MockUnwrapPipelineRuntime {
            assumptions: vec![],
            solve_calls: vec![],
            solve_set: SolutionSet::Discrete(vec![z]),
            solve_steps: vec!["sub-step".to_string()],
        };

        let solved =
            solve_unwrap_execution_pipeline_with_item(execution, y, "x", true, &mut runtime)
                .expect("pipeline should succeed");

        assert_eq!(runtime.assumptions.len(), 1);
        assert_eq!(
            runtime.assumptions[0].assumption,
            LogAssumption::PositiveBase
        );
        assert_eq!(runtime.assumptions[0].base, x);
        assert_eq!(runtime.assumptions[0].other_side, y);
        assert_eq!(runtime.solve_calls, vec!["x"]);
        assert_eq!(solved.solution_set, SolutionSet::Discrete(vec![z]));
        assert_eq!(
            solved.steps,
            vec!["first".to_string(), "sub-step".to_string()]
        );
    }

    #[test]
    fn solve_unwrap_execution_pipeline_with_item_omits_item_when_disabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let execution = UnwrapExecutionPlan {
            equation: Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            description: "unwrap".to_string(),
            assumptions: vec![],
            log_linear_base: None,
            items: vec![UnwrapExecutionItem {
                equation: Equation {
                    lhs: x,
                    rhs: y,
                    op: RelOp::Eq,
                },
                description: "first".to_string(),
            }],
        };

        let mut runtime = MockUnwrapPipelineRuntime {
            assumptions: vec![],
            solve_calls: vec![],
            solve_set: SolutionSet::Discrete(vec![x]),
            solve_steps: vec!["sub-step".to_string()],
        };

        let solved =
            solve_unwrap_execution_pipeline_with_item(execution, y, "x", false, &mut runtime)
                .expect("pipeline should succeed");

        assert_eq!(runtime.assumptions.len(), 0);
        assert_eq!(runtime.solve_calls, vec!["x"]);
        assert_eq!(solved.steps, vec!["sub-step".to_string()]);
    }

    #[test]
    fn solve_unwrap_execution_with_propagates_solver_error() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let execution = UnwrapExecutionPlan {
            equation: Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq,
            },
            description: "unwrap".to_string(),
            assumptions: vec![],
            log_linear_base: None,
            items: vec![],
        };

        let err = solve_unwrap_execution_with(
            execution,
            y,
            |_| {},
            |_equation| Err::<(), _>("solver error"),
        )
        .expect_err("expected solver error");
        assert_eq!(err, "solver error");
    }
}
