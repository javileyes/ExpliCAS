//! Shared strategy kernels used by engine-side solver orchestration.
//!
//! These kernels keep equation applicability checks and core rewrites in
//! `cas_solver_core`, while `cas_engine` remains responsible for recursive
//! orchestration and context-aware simplification.

use crate::isolation_utils::contains_var;
use crate::solve_analysis::{classify_equation_var_presence, EquationVarPresence};
use crate::solve_outcome::{
    eliminate_fractional_exponent_message, first_term_isolation_rewrite_execution_item,
    plan_swap_sides_step, subtract_both_sides_message, TermIsolationRewriteExecutionItem,
    TermIsolationRewritePlan,
};
use cas_ast::{Context, Equation, ExprId, RelOp, SolutionSet};

/// Didactic payload for strategy-level rewrite steps.
#[derive(Debug, Clone, PartialEq)]
pub struct StrategyDidacticStep {
    pub description: String,
    pub equation_after: Equation,
}

/// One strategy execution item with aligned equation and didactic step.
#[derive(Debug, Clone, PartialEq)]
pub struct StrategyExecutionItem {
    pub equation: Equation,
    pub description: String,
}

impl StrategyExecutionItem {
    /// User-facing narration for this execution item.
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Runtime contract for executing strategy rewrite kernels.
pub trait StrategyKernelRuntime {
    /// Mutable access to expression context.
    fn context(&mut self) -> &mut Context;
    /// Simplify an expression according to caller policy.
    fn simplify_expr(&mut self, expr: ExprId) -> ExprId;
    /// Render an expression for didactic narration.
    fn render_expr(&mut self, expr: ExprId) -> String;
}

/// Side-routing plan for isolation strategy applicability.
#[derive(Debug, Clone, PartialEq)]
pub enum IsolationStrategyRouting {
    VariableNotFound,
    VariableOnBothSides,
    SwapSides { rewrite: TermIsolationRewritePlan },
    DirectIsolation,
}

/// Runtime contract for isolation-strategy routing orchestration.
pub trait IsolationStrategyRuntime<E, S> {
    /// Solve `equation` recursively for `var`.
    fn solve_equation(
        &mut self,
        equation: &Equation,
        var: &str,
    ) -> Result<(SolutionSet, Vec<S>), E>;
    /// Convert swap-side item into caller-owned step payload.
    fn map_swap_item_to_step(&mut self, item: TermIsolationRewriteExecutionItem) -> S;
    /// Build caller-specific error for missing variable.
    fn variable_not_found_error(&mut self, var: &str) -> E;
}

/// Derive side-routing for isolation strategy from variable presence.
pub fn derive_isolation_strategy_routing(
    ctx: &Context,
    eq: &Equation,
    var: &str,
) -> IsolationStrategyRouting {
    match classify_equation_var_presence(ctx, eq, var) {
        EquationVarPresence::None => IsolationStrategyRouting::VariableNotFound,
        EquationVarPresence::BothSides => IsolationStrategyRouting::VariableOnBothSides,
        EquationVarPresence::RhsOnly => IsolationStrategyRouting::SwapSides {
            rewrite: plan_swap_sides_step(eq),
        },
        EquationVarPresence::LhsOnly => IsolationStrategyRouting::DirectIsolation,
    }
}

/// Execute isolation strategy for a pre-derived routing decision.
///
/// Returns `None` only when variable appears on both sides (let other
/// strategies rewrite first). Otherwise returns either solved result or error.
pub fn solve_isolation_strategy_routing_with_runtime<R, E, S>(
    routing: IsolationStrategyRouting,
    eq: &Equation,
    var: &str,
    include_item: bool,
    runtime: &mut R,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    R: IsolationStrategyRuntime<E, S>,
{
    match routing {
        IsolationStrategyRouting::VariableNotFound => {
            Some(Err(runtime.variable_not_found_error(var)))
        }
        IsolationStrategyRouting::VariableOnBothSides => None,
        IsolationStrategyRouting::SwapSides { rewrite } => {
            let first_item = if include_item {
                first_term_isolation_rewrite_execution_item(&rewrite)
            } else {
                None
            };
            let solved_swap = runtime.solve_equation(&rewrite.equation, var);
            Some(match solved_swap {
                Ok((solution_set, mut substeps)) => {
                    let mut steps = Vec::new();
                    if let Some(item) = first_item {
                        steps.push(runtime.map_swap_item_to_step(item));
                    }
                    steps.append(&mut substeps);
                    Ok((solution_set, steps))
                }
                Err(e) => Err(e),
            })
        }
        IsolationStrategyRouting::DirectIsolation => Some(runtime.solve_equation(eq, var)),
    }
}

/// Execute isolation strategy for a pre-derived routing decision using
/// caller-provided callbacks.
///
/// Returns `None` only when variable appears on both sides (let other
/// strategies rewrite first). Otherwise returns either solved result or error.
pub fn solve_isolation_strategy_routing_with<E, S, FSolve, FMap, FError>(
    routing: IsolationStrategyRouting,
    eq: &Equation,
    var: &str,
    include_item: bool,
    mut solve_equation: FSolve,
    mut map_swap_item_to_step: FMap,
    mut variable_not_found_error: FError,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    FSolve: FnMut(&Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FMap: FnMut(TermIsolationRewriteExecutionItem) -> S,
    FError: FnMut(&str) -> E,
{
    match routing {
        IsolationStrategyRouting::VariableNotFound => Some(Err(variable_not_found_error(var))),
        IsolationStrategyRouting::VariableOnBothSides => None,
        IsolationStrategyRouting::SwapSides { rewrite } => {
            let first_item = if include_item {
                first_term_isolation_rewrite_execution_item(&rewrite)
            } else {
                None
            };
            let solved_swap = solve_equation(&rewrite.equation, var);
            Some(match solved_swap {
                Ok((solution_set, mut substeps)) => {
                    let mut steps = Vec::new();
                    if let Some(item) = first_item {
                        steps.push(map_swap_item_to_step(item));
                    }
                    steps.append(&mut substeps);
                    Ok((solution_set, steps))
                }
                Err(e) => Err(e),
            })
        }
        IsolationStrategyRouting::DirectIsolation => Some(solve_equation(eq, var)),
    }
}

/// Fully materialized collect-terms rewrite ready for recursive solve.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectTermsSolvedRewrite {
    pub equation: Equation,
    pub items: Vec<StrategyExecutionItem>,
}

/// Solved payload for one collect-terms rewrite execution.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectTermsSolved<T> {
    pub rewrite: CollectTermsSolvedRewrite,
    pub solved: T,
}

/// Fully materialized rational-exponent rewrite ready for recursive solve.
#[derive(Debug, Clone, PartialEq)]
pub struct RationalExponentSolvedRewrite {
    pub equation: Equation,
    pub items: Vec<StrategyExecutionItem>,
}

/// Solved payload for one rational-exponent rewrite execution.
#[derive(Debug, Clone, PartialEq)]
pub struct RationalExponentSolved<T> {
    pub rewrite: RationalExponentSolvedRewrite,
    pub solved: T,
}

/// Execution payload for collect-terms strategy rewrite.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectTermsExecutionPlan {
    pub equation: Equation,
    pub description: String,
}

/// Collect collect-terms didactic steps in display order.
pub fn collect_collect_terms_didactic_steps(
    execution: &CollectTermsExecutionPlan,
) -> Vec<StrategyDidacticStep> {
    vec![StrategyDidacticStep {
        description: execution.description.clone(),
        equation_after: execution.equation.clone(),
    }]
}

/// Collect collect-terms execution items in execution order.
pub fn collect_collect_terms_execution_items(
    execution: &CollectTermsExecutionPlan,
) -> Vec<StrategyExecutionItem> {
    vec![StrategyExecutionItem {
        equation: execution.equation.clone(),
        description: execution.description.clone(),
    }]
}

/// Rewrite payload for `CollectTermsStrategy`.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectTermsKernel {
    /// Equation after subtracting RHS from both sides.
    pub rewritten: Equation,
}

/// Build collect-terms rewrite only when the solve variable appears on both sides.
pub fn derive_collect_terms_kernel(
    ctx: &mut Context,
    eq: &Equation,
    var: &str,
) -> Option<CollectTermsKernel> {
    let lhs_has = contains_var(ctx, eq.lhs, var);
    let rhs_has = contains_var(ctx, eq.rhs, var);
    if !lhs_has || !rhs_has {
        return None;
    }
    Some(CollectTermsKernel {
        rewritten: crate::equation_rewrite::subtract_rhs_from_both_sides(ctx, eq),
    })
}

/// Build didactic narration for collect-terms subtraction.
pub fn collect_terms_message(rhs_display: &str) -> String {
    subtract_both_sides_message(rhs_display)
}

/// Build didactic payload for collect-terms rewrite.
pub fn build_collect_terms_step_with<F>(
    equation_after: Equation,
    original_rhs: ExprId,
    mut render_expr: F,
) -> StrategyDidacticStep
where
    F: FnMut(ExprId) -> String,
{
    let rhs_desc = render_expr(original_rhs);
    StrategyDidacticStep {
        description: collect_terms_message(&rhs_desc),
        equation_after,
    }
}

/// Build executable collect-terms payload from simplified equation sides.
pub fn build_collect_terms_execution_with<F>(
    lhs_after: ExprId,
    rhs_after: ExprId,
    op: RelOp,
    original_rhs: ExprId,
    render_expr: F,
) -> CollectTermsExecutionPlan
where
    F: FnMut(ExprId) -> String,
{
    let equation = Equation {
        lhs: lhs_after,
        rhs: rhs_after,
        op,
    };
    let step = build_collect_terms_step_with(equation.clone(), original_rhs, render_expr);
    CollectTermsExecutionPlan {
        equation,
        description: step.description,
    }
}

/// Derive, simplify, and materialize collect-terms rewrite in one pipeline.
///
/// This centralizes strategy-kernel orchestration while caller controls
/// simplification and rendering.
pub fn execute_collect_terms_rewrite_with<FSimplify, FRender>(
    ctx: &mut Context,
    eq: &Equation,
    var: &str,
    mut simplify_expr: FSimplify,
    mut render_expr: FRender,
) -> Option<CollectTermsSolvedRewrite>
where
    FSimplify: FnMut(ExprId) -> ExprId,
    FRender: FnMut(ExprId) -> String,
{
    let kernel = derive_collect_terms_kernel(ctx, eq, var)?;
    let simp_lhs = simplify_expr(kernel.rewritten.lhs);
    let simp_rhs = simplify_expr(kernel.rewritten.rhs);
    let execution =
        build_collect_terms_execution_with(simp_lhs, simp_rhs, eq.op.clone(), eq.rhs, |id| {
            render_expr(id)
        });
    let items = collect_collect_terms_execution_items(&execution);
    Some(CollectTermsSolvedRewrite {
        equation: execution.equation,
        items,
    })
}

/// Derive, simplify, and materialize collect-terms rewrite via runtime.
pub fn execute_collect_terms_rewrite_with_runtime<R>(
    runtime: &mut R,
    eq: &Equation,
    var: &str,
) -> Option<CollectTermsSolvedRewrite>
where
    R: StrategyKernelRuntime,
{
    let kernel = {
        let ctx = runtime.context();
        derive_collect_terms_kernel(ctx, eq, var)?
    };
    let simp_lhs = runtime.simplify_expr(kernel.rewritten.lhs);
    let simp_rhs = runtime.simplify_expr(kernel.rewritten.rhs);
    let execution =
        build_collect_terms_execution_with(simp_lhs, simp_rhs, eq.op.clone(), eq.rhs, |id| {
            runtime.render_expr(id)
        });
    let items = collect_collect_terms_execution_items(&execution);
    Some(CollectTermsSolvedRewrite {
        equation: execution.equation,
        items,
    })
}

/// Execute recursive solve for a materialized collect-terms rewrite.
pub fn solve_collect_terms_rewrite_with<E, T, FSolve>(
    rewrite: CollectTermsSolvedRewrite,
    mut solve_rewritten: FSolve,
) -> Result<CollectTermsSolved<T>, E>
where
    FSolve: FnMut(&Equation) -> Result<T, E>,
{
    let solved = solve_rewritten(&rewrite.equation)?;
    Ok(CollectTermsSolved { rewrite, solved })
}

/// Execute recursive solve for a materialized collect-terms rewrite while
/// passing the aligned optional didactic item to the solve callback.
pub fn solve_collect_terms_rewrite_with_item<E, T, FSolve>(
    rewrite: CollectTermsSolvedRewrite,
    mut solve_rewritten: FSolve,
) -> Result<CollectTermsSolved<T>, E>
where
    FSolve: FnMut(Option<StrategyExecutionItem>, &Equation) -> Result<T, E>,
{
    let item = rewrite.items.first().cloned();
    let solved = solve_rewritten(item, &rewrite.equation)?;
    Ok(CollectTermsSolved { rewrite, solved })
}

/// Solved result for a collect-terms rewrite pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectTermsRewriteSolved<S> {
    pub solution_set: SolutionSet,
    pub steps: Vec<S>,
}

/// Execute collect-terms rewrite solving + optional item dispatch.
pub fn solve_collect_terms_rewrite_pipeline_with_item<E, S, FSolve, FStep>(
    rewrite: CollectTermsSolvedRewrite,
    var: &str,
    include_item: bool,
    mut solve_rewritten: FSolve,
    mut map_item_to_step: FStep,
) -> Result<CollectTermsRewriteSolved<S>, E>
where
    FSolve: FnMut(&Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(StrategyExecutionItem) -> S,
{
    let mut steps = Vec::new();
    if include_item {
        if let Some(item) = rewrite.items.first().cloned() {
            steps.push(map_item_to_step(item));
        }
    }
    let (solution_set, mut sub_steps) = solve_rewritten(&rewrite.equation, var)?;
    steps.append(&mut sub_steps);
    Ok(CollectTermsRewriteSolved {
        solution_set,
        steps,
    })
}

/// Runtime contract for collect-terms rewrite solve orchestration.
pub trait CollectTermsRewriteRuntime<E, S> {
    /// Solve rewritten equation recursively for `var`.
    fn solve_rewritten(
        &mut self,
        equation: &Equation,
        var: &str,
    ) -> Result<(SolutionSet, Vec<S>), E>;
    /// Convert one optional rewrite item into a caller step payload.
    fn map_item_to_step(&mut self, item: StrategyExecutionItem) -> S;
}

/// Execute collect-terms rewrite solving + optional item dispatch via runtime.
pub fn solve_collect_terms_rewrite_pipeline_with_item_runtime<E, S, R>(
    rewrite: CollectTermsSolvedRewrite,
    var: &str,
    include_item: bool,
    runtime: &mut R,
) -> Result<CollectTermsRewriteSolved<S>, E>
where
    R: CollectTermsRewriteRuntime<E, S>,
{
    let mut steps = Vec::new();
    if include_item {
        if let Some(item) = rewrite.items.first().cloned() {
            steps.push(runtime.map_item_to_step(item));
        }
    }
    let (solution_set, mut sub_steps) = runtime.solve_rewritten(&rewrite.equation, var)?;
    steps.append(&mut sub_steps);
    Ok(CollectTermsRewriteSolved {
        solution_set,
        steps,
    })
}

/// Rewrite payload for `RationalExponentStrategy`.
#[derive(Debug, Clone, PartialEq)]
pub struct RationalExponentKernel {
    /// Rewritten equation `base^p = rhs^q`.
    pub rewritten: Equation,
    /// Denominator from the original rational exponent `p/q`.
    pub q: i64,
}

/// Rewrite an isolated rational exponent equation (`x^(p/q) = rhs`) into `x^p = rhs^q`.
pub fn derive_rational_exponent_kernel(
    ctx: &mut Context,
    eq: &Equation,
    var: &str,
    lhs_has_var: bool,
    rhs_has_var: bool,
) -> Option<RationalExponentKernel> {
    if eq.op != RelOp::Eq {
        return None;
    }
    let (rewritten, _p, q) = crate::rational_power::rewrite_isolated_rational_power_equation(
        ctx,
        eq.lhs,
        eq.rhs,
        var,
        eq.op.clone(),
        lhs_has_var,
        rhs_has_var,
    )?;
    Some(RationalExponentKernel { rewritten, q })
}

/// Rewrite an isolated rational exponent equation while deriving side presence
/// from `eq` for `var`.
pub fn derive_rational_exponent_kernel_for_var(
    ctx: &mut Context,
    eq: &Equation,
    var: &str,
) -> Option<RationalExponentKernel> {
    let lhs_has_var = contains_var(ctx, eq.lhs, var);
    let rhs_has_var = contains_var(ctx, eq.rhs, var);
    derive_rational_exponent_kernel(ctx, eq, var, lhs_has_var, rhs_has_var)
}

/// Build didactic narration for rational exponent elimination.
pub fn rational_exponent_message(q: i64) -> String {
    eliminate_fractional_exponent_message(&q.to_string())
}

/// Build didactic payload for rational-exponent elimination rewrite.
pub fn build_rational_exponent_step(q: i64, equation_after: Equation) -> StrategyDidacticStep {
    StrategyDidacticStep {
        description: rational_exponent_message(q),
        equation_after,
    }
}

/// Execution payload for rational-exponent elimination rewrite.
#[derive(Debug, Clone, PartialEq)]
pub struct RationalExponentExecutionPlan {
    pub equation: Equation,
    pub description: String,
}

/// Collect rational-exponent didactic steps in display order.
pub fn collect_rational_exponent_didactic_steps(
    execution: &RationalExponentExecutionPlan,
) -> Vec<StrategyDidacticStep> {
    vec![StrategyDidacticStep {
        description: execution.description.clone(),
        equation_after: execution.equation.clone(),
    }]
}

/// Collect rational-exponent execution items in execution order.
pub fn collect_rational_exponent_execution_items(
    execution: &RationalExponentExecutionPlan,
) -> Vec<StrategyExecutionItem> {
    vec![StrategyExecutionItem {
        equation: execution.equation.clone(),
        description: execution.description.clone(),
    }]
}

/// Build executable rational-exponent payload from simplified equation sides.
pub fn build_rational_exponent_execution(
    q: i64,
    lhs_after: ExprId,
    rhs_after: ExprId,
) -> RationalExponentExecutionPlan {
    let equation = Equation {
        lhs: lhs_after,
        rhs: rhs_after,
        op: RelOp::Eq,
    };
    let step = build_rational_exponent_step(q, equation.clone());
    RationalExponentExecutionPlan {
        equation,
        description: step.description,
    }
}

/// Derive, simplify, and materialize rational-exponent rewrite in one pipeline.
///
/// This centralizes strategy-kernel orchestration while caller controls
/// simplification.
pub fn execute_rational_exponent_rewrite_with<FSimplify>(
    ctx: &mut Context,
    eq: &Equation,
    var: &str,
    lhs_has_var: bool,
    rhs_has_var: bool,
    mut simplify_expr: FSimplify,
) -> Option<RationalExponentSolvedRewrite>
where
    FSimplify: FnMut(ExprId) -> ExprId,
{
    let kernel = derive_rational_exponent_kernel(ctx, eq, var, lhs_has_var, rhs_has_var)?;
    let sim_lhs = simplify_expr(kernel.rewritten.lhs);
    let sim_rhs = simplify_expr(kernel.rewritten.rhs);
    let execution = build_rational_exponent_execution(kernel.q, sim_lhs, sim_rhs);
    let items = collect_rational_exponent_execution_items(&execution);
    Some(RationalExponentSolvedRewrite {
        equation: execution.equation,
        items,
    })
}

/// Derive, simplify, and materialize rational-exponent rewrite in one pipeline
/// deriving side presence from `eq` for `var`.
pub fn execute_rational_exponent_rewrite_with_for_var<FSimplify>(
    ctx: &mut Context,
    eq: &Equation,
    var: &str,
    simplify_expr: FSimplify,
) -> Option<RationalExponentSolvedRewrite>
where
    FSimplify: FnMut(ExprId) -> ExprId,
{
    let lhs_has_var = contains_var(ctx, eq.lhs, var);
    let rhs_has_var = contains_var(ctx, eq.rhs, var);
    execute_rational_exponent_rewrite_with(ctx, eq, var, lhs_has_var, rhs_has_var, simplify_expr)
}

/// Derive, simplify, and materialize rational-exponent rewrite via runtime.
pub fn execute_rational_exponent_rewrite_with_runtime<R>(
    runtime: &mut R,
    eq: &Equation,
    var: &str,
    lhs_has_var: bool,
    rhs_has_var: bool,
) -> Option<RationalExponentSolvedRewrite>
where
    R: StrategyKernelRuntime,
{
    let kernel = {
        let ctx = runtime.context();
        derive_rational_exponent_kernel(ctx, eq, var, lhs_has_var, rhs_has_var)?
    };
    let sim_lhs = runtime.simplify_expr(kernel.rewritten.lhs);
    let sim_rhs = runtime.simplify_expr(kernel.rewritten.rhs);
    let execution = build_rational_exponent_execution(kernel.q, sim_lhs, sim_rhs);
    let items = collect_rational_exponent_execution_items(&execution);
    Some(RationalExponentSolvedRewrite {
        equation: execution.equation,
        items,
    })
}

/// Derive, simplify, and materialize rational-exponent rewrite via runtime,
/// deriving side presence from `eq` for `var`.
pub fn execute_rational_exponent_rewrite_with_runtime_for_var<R>(
    runtime: &mut R,
    eq: &Equation,
    var: &str,
) -> Option<RationalExponentSolvedRewrite>
where
    R: StrategyKernelRuntime,
{
    let (lhs_has_var, rhs_has_var) = {
        let ctx = runtime.context();
        (
            contains_var(ctx, eq.lhs, var),
            contains_var(ctx, eq.rhs, var),
        )
    };
    execute_rational_exponent_rewrite_with_runtime(runtime, eq, var, lhs_has_var, rhs_has_var)
}

/// Execute recursive solve for a materialized rational-exponent rewrite.
pub fn solve_rational_exponent_rewrite_with<E, T, FSolve>(
    rewrite: RationalExponentSolvedRewrite,
    mut solve_rewritten: FSolve,
) -> Result<RationalExponentSolved<T>, E>
where
    FSolve: FnMut(&Equation) -> Result<T, E>,
{
    let solved = solve_rewritten(&rewrite.equation)?;
    Ok(RationalExponentSolved { rewrite, solved })
}

/// Execute recursive solve for a materialized rational-exponent rewrite while
/// passing the aligned optional didactic item to the solve callback.
pub fn solve_rational_exponent_rewrite_with_item<E, T, FSolve>(
    rewrite: RationalExponentSolvedRewrite,
    mut solve_rewritten: FSolve,
) -> Result<RationalExponentSolved<T>, E>
where
    FSolve: FnMut(Option<StrategyExecutionItem>, &Equation) -> Result<T, E>,
{
    let item = rewrite.items.first().cloned();
    let solved = solve_rewritten(item, &rewrite.equation)?;
    Ok(RationalExponentSolved { rewrite, solved })
}

/// Runtime contract for rational-exponent rewrite solve orchestration.
pub trait RationalExponentRewriteRuntime<E, S> {
    /// Solve rewritten equation recursively for `var`.
    fn solve_rewritten(
        &mut self,
        equation: &Equation,
        var: &str,
    ) -> Result<(SolutionSet, Vec<S>), E>;
    /// Convert one optional rewrite item into a caller step payload.
    fn map_item_to_step(&mut self, item: StrategyExecutionItem) -> S;
    /// Verify one discrete candidate against caller policy.
    fn verify_discrete_solution(&mut self, solution: ExprId) -> bool;
}

/// Solved result for a rational-exponent rewrite pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct RationalExponentRewriteSolved<S> {
    pub solution_set: SolutionSet,
    pub steps: Vec<S>,
}

/// Execute rational-exponent rewrite solving + optional item dispatch + discrete verification.
pub fn solve_rational_exponent_rewrite_pipeline_with_item_with<E, S, FSolve, FStep, FVerify>(
    rewrite: RationalExponentSolvedRewrite,
    var: &str,
    include_item: bool,
    mut solve_rewritten: FSolve,
    mut map_item_to_step: FStep,
    verify_discrete_solution: FVerify,
) -> Result<RationalExponentRewriteSolved<S>, E>
where
    FSolve: FnMut(&Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FStep: FnMut(StrategyExecutionItem) -> S,
    FVerify: FnMut(ExprId) -> bool,
{
    let mut steps = Vec::new();
    if include_item {
        if let Some(item) = rewrite.items.first().cloned() {
            steps.push(map_item_to_step(item));
        }
    }
    let (set, mut sub_steps) = solve_rewritten(&rewrite.equation, var)?;
    steps.append(&mut sub_steps);
    let solution_set = if let SolutionSet::Discrete(sols) = set {
        SolutionSet::Discrete(crate::solve_analysis::retain_verified_discrete(
            sols,
            verify_discrete_solution,
        ))
    } else {
        set
    };

    Ok(RationalExponentRewriteSolved {
        solution_set,
        steps,
    })
}

/// Execute rational-exponent rewrite solving + optional item dispatch + discrete verification.
pub fn solve_rational_exponent_rewrite_pipeline_with_item<E, S, R>(
    rewrite: RationalExponentSolvedRewrite,
    var: &str,
    include_item: bool,
    runtime: &mut R,
) -> Result<RationalExponentRewriteSolved<S>, E>
where
    R: RationalExponentRewriteRuntime<E, S>,
{
    let mut steps = Vec::new();
    if include_item {
        if let Some(item) = rewrite.items.first().cloned() {
            steps.push(runtime.map_item_to_step(item));
        }
    }
    let (set, mut sub_steps) = runtime.solve_rewritten(&rewrite.equation, var)?;
    steps.append(&mut sub_steps);
    let solution_set = if let SolutionSet::Discrete(sols) = set {
        SolutionSet::Discrete(crate::solve_analysis::retain_verified_discrete(
            sols,
            |sol| runtime.verify_discrete_solution(sol),
        ))
    } else {
        set
    };

    Ok(RationalExponentRewriteSolved {
        solution_set,
        steps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::{Context, Expr};

    struct MockIsolationRuntime {
        solved_with: Vec<Equation>,
    }

    impl MockIsolationRuntime {
        fn new() -> Self {
            Self {
                solved_with: Vec::new(),
            }
        }
    }

    impl IsolationStrategyRuntime<String, String> for MockIsolationRuntime {
        fn solve_equation(
            &mut self,
            equation: &Equation,
            _var: &str,
        ) -> Result<(SolutionSet, Vec<String>), String> {
            self.solved_with.push(equation.clone());
            Ok((
                SolutionSet::Discrete(vec![equation.lhs]),
                vec![format!("solved {}", equation.lhs)],
            ))
        }

        fn map_swap_item_to_step(&mut self, item: TermIsolationRewriteExecutionItem) -> String {
            item.description
        }

        fn variable_not_found_error(&mut self, var: &str) -> String {
            format!("var-not-found:{var}")
        }
    }

    #[test]
    fn collect_terms_kernel_requires_var_on_both_sides() {
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
        assert!(derive_collect_terms_kernel(&mut ctx, &eq, "x").is_none());
    }

    #[test]
    fn collect_terms_kernel_rewrites_when_var_on_both_sides() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let lhs = ctx.add(Expr::Add(x, one));
        let rhs = ctx.add(Expr::Add(x, two));
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };
        let kernel = derive_collect_terms_kernel(&mut ctx, &eq, "x");
        assert!(kernel.is_some());
    }

    #[test]
    fn derive_isolation_strategy_routing_reports_variable_not_found() {
        let mut ctx = Context::new();
        let y = ctx.var("y");
        let z = ctx.var("z");
        let eq = Equation {
            lhs: y,
            rhs: z,
            op: RelOp::Eq,
        };

        let routing = derive_isolation_strategy_routing(&ctx, &eq, "x");
        assert_eq!(routing, IsolationStrategyRouting::VariableNotFound);
    }

    #[test]
    fn derive_isolation_strategy_routing_reports_variable_on_both_sides() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let lhs = ctx.add(Expr::Add(x, one));
        let rhs = ctx.add(Expr::Add(x, two));
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };

        let routing = derive_isolation_strategy_routing(&ctx, &eq, "x");
        assert_eq!(routing, IsolationStrategyRouting::VariableOnBothSides);
    }

    #[test]
    fn derive_isolation_strategy_routing_reports_swap_sides_when_variable_is_rhs_only() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let eq = Equation {
            lhs: y,
            rhs: x,
            op: RelOp::Lt,
        };

        let routing = derive_isolation_strategy_routing(&ctx, &eq, "x");
        match routing {
            IsolationStrategyRouting::SwapSides { rewrite } => {
                assert_eq!(rewrite.equation.lhs, eq.rhs);
                assert_eq!(rewrite.equation.rhs, eq.lhs);
                assert_eq!(rewrite.equation.op, RelOp::Gt);
            }
            other => panic!("expected swap routing, got {:?}", other),
        }
    }

    #[test]
    fn derive_isolation_strategy_routing_reports_direct_when_variable_is_lhs_only() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let eq = Equation {
            lhs: x,
            rhs: y,
            op: RelOp::Eq,
        };

        let routing = derive_isolation_strategy_routing(&ctx, &eq, "x");
        assert_eq!(routing, IsolationStrategyRouting::DirectIsolation);
    }

    #[test]
    fn solve_isolation_strategy_routing_with_runtime_direct_calls_runtime_once() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let eq = Equation {
            lhs: x,
            rhs: y,
            op: RelOp::Eq,
        };
        let mut runtime = MockIsolationRuntime::new();
        let solved = solve_isolation_strategy_routing_with_runtime(
            IsolationStrategyRouting::DirectIsolation,
            &eq,
            "x",
            true,
            &mut runtime,
        )
        .expect("direct isolation should be applicable")
        .expect("runtime should solve");

        assert_eq!(runtime.solved_with.len(), 1);
        assert_eq!(runtime.solved_with[0], eq);
        assert!(matches!(solved.0, SolutionSet::Discrete(_)));
        assert_eq!(solved.1, vec![format!("solved {}", eq.lhs)]);
    }

    #[test]
    fn solve_isolation_strategy_routing_with_runtime_swap_prepends_item_when_enabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let eq = Equation {
            lhs: y,
            rhs: x,
            op: RelOp::Lt,
        };
        let routing = derive_isolation_strategy_routing(&ctx, &eq, "x");
        let mut runtime = MockIsolationRuntime::new();
        let solved =
            solve_isolation_strategy_routing_with_runtime(routing, &eq, "x", true, &mut runtime)
                .expect("swap route should be applicable")
                .expect("runtime should solve");

        assert_eq!(runtime.solved_with.len(), 1);
        // swap-side rewrite should call runtime with swapped equation
        assert_eq!(runtime.solved_with[0].lhs, eq.rhs);
        assert_eq!(runtime.solved_with[0].rhs, eq.lhs);
        assert_eq!(runtime.solved_with[0].op, RelOp::Gt);
        assert_eq!(solved.1.len(), 2);
        assert_eq!(solved.1[0], "Swap sides to put variable on LHS");
    }

    #[test]
    fn solve_isolation_strategy_routing_with_runtime_swap_omits_item_when_disabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let eq = Equation {
            lhs: y,
            rhs: x,
            op: RelOp::Eq,
        };
        let routing = derive_isolation_strategy_routing(&ctx, &eq, "x");
        let mut runtime = MockIsolationRuntime::new();
        let solved =
            solve_isolation_strategy_routing_with_runtime(routing, &eq, "x", false, &mut runtime)
                .expect("swap route should be applicable")
                .expect("runtime should solve");

        assert_eq!(runtime.solved_with.len(), 1);
        assert_eq!(solved.1.len(), 1);
        assert_eq!(
            solved.1[0],
            format!("solved {}", runtime.solved_with[0].lhs)
        );
    }

    #[test]
    fn solve_isolation_strategy_routing_with_runtime_variable_not_found_returns_error() {
        let mut ctx = Context::new();
        let y = ctx.var("y");
        let z = ctx.var("z");
        let eq = Equation {
            lhs: y,
            rhs: z,
            op: RelOp::Eq,
        };
        let mut runtime = MockIsolationRuntime::new();
        let solved = solve_isolation_strategy_routing_with_runtime(
            IsolationStrategyRouting::VariableNotFound,
            &eq,
            "x",
            true,
            &mut runtime,
        )
        .expect("variable-not-found should be terminal");

        assert_eq!(
            solved.expect_err("expected variable-not-found error"),
            "var-not-found:x"
        );
        assert!(runtime.solved_with.is_empty());
    }

    #[test]
    fn solve_isolation_strategy_routing_with_runtime_both_sides_is_not_applicable() {
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
        let mut runtime = MockIsolationRuntime::new();
        let solved = solve_isolation_strategy_routing_with_runtime(
            IsolationStrategyRouting::VariableOnBothSides,
            &eq,
            "x",
            true,
            &mut runtime,
        );
        assert!(solved.is_none());
        assert!(runtime.solved_with.is_empty());
    }

    #[test]
    fn rational_exponent_kernel_rewrites_isolated_power() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let exponent = ctx.add(Expr::Div(three, two));
        let lhs = ctx.add(Expr::Pow(x, exponent));
        let rhs = ctx.num(8);
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };
        let kernel = derive_rational_exponent_kernel(&mut ctx, &eq, "x", true, false)
            .expect("kernel should exist");
        assert_eq!(kernel.q, 2);

        // lhs should become x^3
        match ctx.get(kernel.rewritten.lhs) {
            Expr::Pow(base, exp) => {
                assert_eq!(*base, x);
                assert_eq!(*exp, three);
            }
            other => panic!("expected rewritten lhs pow, got {:?}", other),
        }
    }

    #[test]
    fn derive_rational_exponent_kernel_for_var_derives_side_presence() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let exponent = ctx.add(Expr::Div(three, two));
        let lhs = ctx.add(Expr::Pow(x, exponent));
        let rhs = ctx.num(8);
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };

        let kernel = derive_rational_exponent_kernel_for_var(&mut ctx, &eq, "x")
            .expect("kernel should exist");
        assert_eq!(kernel.q, 2);
    }

    #[test]
    fn strategy_step_builders_use_expected_messages() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let eq = Equation {
            lhs: x,
            rhs: y,
            op: RelOp::Eq,
        };

        let collect = build_collect_terms_step_with(eq.clone(), y, |_| "rhs".to_string());
        assert_eq!(collect.description, "Subtract rhs from both sides");
        assert_eq!(collect.equation_after, eq);

        let rational = build_rational_exponent_step(3, eq.clone());
        assert_eq!(
            rational.description,
            "Raise both sides to power 3 to eliminate fractional exponent"
        );
        assert_eq!(rational.equation_after, eq);
    }

    #[test]
    fn build_collect_terms_execution_with_builds_equation_and_didactic() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let rhs_orig = ctx.var("rhs");

        let execution =
            build_collect_terms_execution_with(x, y, RelOp::Eq, rhs_orig, |_| "rhs".to_string());
        assert_eq!(
            execution.equation,
            Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq
            }
        );
        assert_eq!(execution.description, "Subtract rhs from both sides");
    }

    #[test]
    fn collect_collect_terms_didactic_steps_returns_single_step() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let rhs_orig = ctx.var("rhs");
        let execution =
            build_collect_terms_execution_with(x, y, RelOp::Eq, rhs_orig, |_| "rhs".to_string());

        let didactic = collect_collect_terms_didactic_steps(&execution);
        assert_eq!(didactic.len(), 1);
        assert_eq!(didactic[0].description, execution.description);
        assert_eq!(didactic[0].equation_after, execution.equation);
    }

    #[test]
    fn collect_collect_terms_execution_items_returns_single_item() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let rhs_orig = ctx.var("rhs");
        let execution =
            build_collect_terms_execution_with(x, y, RelOp::Eq, rhs_orig, |_| "rhs".to_string());

        let items = collect_collect_terms_execution_items(&execution);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].equation, execution.equation);
        assert_eq!(items[0].description, execution.description);
    }

    #[test]
    fn build_rational_exponent_execution_builds_equation_and_didactic() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");

        let execution = build_rational_exponent_execution(5, x, y);
        assert_eq!(
            execution.equation,
            Equation {
                lhs: x,
                rhs: y,
                op: RelOp::Eq
            }
        );
        assert_eq!(
            execution.description,
            "Raise both sides to power 5 to eliminate fractional exponent"
        );
    }

    #[test]
    fn collect_rational_exponent_didactic_steps_returns_single_step() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let execution = build_rational_exponent_execution(5, x, y);

        let didactic = collect_rational_exponent_didactic_steps(&execution);
        assert_eq!(didactic.len(), 1);
        assert_eq!(didactic[0].description, execution.description);
        assert_eq!(didactic[0].equation_after, execution.equation);
    }

    #[test]
    fn collect_rational_exponent_execution_items_returns_single_item() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let execution = build_rational_exponent_execution(5, x, y);

        let items = collect_rational_exponent_execution_items(&execution);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].equation, execution.equation);
        assert_eq!(items[0].description, execution.description);
    }

    #[test]
    fn execute_collect_terms_rewrite_with_materializes_items_and_equation() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let lhs = ctx.add(Expr::Add(x, one));
        let rhs = ctx.add(Expr::Add(x, two));
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };

        let solved =
            execute_collect_terms_rewrite_with(&mut ctx, &eq, "x", |id| id, |_| "rhs".to_string())
                .expect("collect-terms rewrite should apply");

        assert_eq!(solved.items.len(), 1);
        assert_eq!(solved.items[0].equation, solved.equation);
        assert_eq!(solved.items[0].description, "Subtract rhs from both sides");
    }

    #[test]
    fn execute_rational_exponent_rewrite_with_materializes_items_and_equation() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let exponent = ctx.add(Expr::Div(three, two));
        let lhs = ctx.add(Expr::Pow(x, exponent));
        let rhs = ctx.num(8);
        let eq = Equation {
            lhs,
            rhs,
            op: RelOp::Eq,
        };

        let solved =
            execute_rational_exponent_rewrite_with(&mut ctx, &eq, "x", true, false, |id| id)
                .expect("rational-exponent rewrite should apply");

        assert_eq!(solved.items.len(), 1);
        assert_eq!(solved.items[0].equation, solved.equation);
        assert_eq!(
            solved.items[0].description,
            "Raise both sides to power 2 to eliminate fractional exponent"
        );
    }

    #[test]
    fn solve_collect_terms_rewrite_with_runs_solver_on_rewritten_equation() {
        let mut ctx = Context::new();
        let lhs = ctx.var("lhs");
        let rhs = ctx.var("rhs");
        let rewrite = CollectTermsSolvedRewrite {
            equation: Equation {
                lhs,
                rhs,
                op: RelOp::Eq,
            },
            items: vec![],
        };

        let solved = solve_collect_terms_rewrite_with(rewrite, |equation| {
            assert_eq!(equation.lhs, lhs);
            assert_eq!(equation.rhs, rhs);
            Ok::<_, ()>("ok")
        })
        .expect("must solve rewrite");

        assert_eq!(solved.solved, "ok");
    }

    #[test]
    fn solve_collect_terms_rewrite_with_item_passes_item_to_solver() {
        let mut ctx = Context::new();
        let lhs = ctx.var("lhs");
        let rhs = ctx.var("rhs");
        let rewrite = CollectTermsSolvedRewrite {
            equation: Equation {
                lhs,
                rhs,
                op: RelOp::Eq,
            },
            items: vec![StrategyExecutionItem {
                equation: Equation {
                    lhs,
                    rhs,
                    op: RelOp::Eq,
                },
                description: "Subtract rhs from both sides".to_string(),
            }],
        };

        let mut seen_desc = None;
        let solved = solve_collect_terms_rewrite_with_item(rewrite, |item, equation| {
            seen_desc = item.map(|entry| entry.description);
            assert_eq!(equation.lhs, lhs);
            assert_eq!(equation.rhs, rhs);
            Ok::<_, ()>("ok")
        })
        .expect("must solve rewrite");

        assert_eq!(solved.solved, "ok");
        assert_eq!(seen_desc, Some("Subtract rhs from both sides".to_string()));
    }

    #[test]
    fn solve_collect_terms_rewrite_pipeline_with_item_forwards_item_and_substeps() {
        let mut ctx = Context::new();
        let lhs = ctx.var("lhs");
        let rhs = ctx.var("rhs");
        let rewrite = CollectTermsSolvedRewrite {
            equation: Equation {
                lhs,
                rhs,
                op: RelOp::Eq,
            },
            items: vec![StrategyExecutionItem {
                equation: Equation {
                    lhs,
                    rhs,
                    op: RelOp::Eq,
                },
                description: "Subtract rhs from both sides".to_string(),
            }],
        };

        let solved = solve_collect_terms_rewrite_pipeline_with_item(
            rewrite,
            "x",
            true,
            |equation, var| {
                assert_eq!(equation.lhs, lhs);
                assert_eq!(equation.rhs, rhs);
                assert_eq!(var, "x");
                Ok::<_, ()>((
                    SolutionSet::Discrete(vec![lhs]),
                    vec!["sub-step".to_string()],
                ))
            },
            |item| item.description,
        )
        .expect("collect pipeline should succeed");

        assert_eq!(solved.solution_set, SolutionSet::Discrete(vec![lhs]));
        assert_eq!(
            solved.steps,
            vec![
                "Subtract rhs from both sides".to_string(),
                "sub-step".to_string()
            ]
        );
    }

    #[test]
    fn solve_collect_terms_rewrite_pipeline_with_item_omits_item_when_disabled() {
        let mut ctx = Context::new();
        let lhs = ctx.var("lhs");
        let rhs = ctx.var("rhs");
        let rewrite = CollectTermsSolvedRewrite {
            equation: Equation {
                lhs,
                rhs,
                op: RelOp::Eq,
            },
            items: vec![StrategyExecutionItem {
                equation: Equation {
                    lhs,
                    rhs,
                    op: RelOp::Eq,
                },
                description: "Subtract rhs from both sides".to_string(),
            }],
        };

        let solved = solve_collect_terms_rewrite_pipeline_with_item(
            rewrite,
            "x",
            false,
            |_equation, _var| {
                Ok::<_, ()>((
                    SolutionSet::Discrete(vec![rhs]),
                    vec!["sub-step".to_string()],
                ))
            },
            |item| item.description,
        )
        .expect("collect pipeline should succeed");

        assert_eq!(solved.solution_set, SolutionSet::Discrete(vec![rhs]));
        assert_eq!(solved.steps, vec!["sub-step".to_string()]);
    }

    struct TestCollectTermsRuntime {
        solve_result: (SolutionSet, Vec<String>),
        last_var: Option<String>,
    }

    impl CollectTermsRewriteRuntime<&'static str, String> for TestCollectTermsRuntime {
        fn solve_rewritten(
            &mut self,
            _: &Equation,
            var: &str,
        ) -> Result<(SolutionSet, Vec<String>), &'static str> {
            self.last_var = Some(var.to_string());
            Ok(self.solve_result.clone())
        }

        fn map_item_to_step(&mut self, item: StrategyExecutionItem) -> String {
            item.description
        }
    }

    #[test]
    fn solve_collect_terms_rewrite_pipeline_with_item_runtime_forwards_item_and_substeps() {
        let mut ctx = Context::new();
        let lhs = ctx.var("lhs");
        let rhs = ctx.var("rhs");
        let rewrite = CollectTermsSolvedRewrite {
            equation: Equation {
                lhs,
                rhs,
                op: RelOp::Eq,
            },
            items: vec![StrategyExecutionItem {
                equation: Equation {
                    lhs,
                    rhs,
                    op: RelOp::Eq,
                },
                description: "Subtract rhs from both sides".to_string(),
            }],
        };
        let mut runtime = TestCollectTermsRuntime {
            solve_result: (
                SolutionSet::Discrete(vec![lhs]),
                vec!["runtime-sub-step".to_string()],
            ),
            last_var: None,
        };

        let solved = solve_collect_terms_rewrite_pipeline_with_item_runtime(
            rewrite,
            "x",
            true,
            &mut runtime,
        )
        .expect("runtime collect pipeline should succeed");

        assert_eq!(runtime.last_var.as_deref(), Some("x"));
        assert_eq!(solved.solution_set, SolutionSet::Discrete(vec![lhs]));
        assert_eq!(
            solved.steps,
            vec![
                "Subtract rhs from both sides".to_string(),
                "runtime-sub-step".to_string()
            ]
        );
    }

    #[test]
    fn solve_collect_terms_rewrite_pipeline_with_item_runtime_omits_item_when_disabled() {
        let mut ctx = Context::new();
        let lhs = ctx.var("lhs");
        let rhs = ctx.var("rhs");
        let rewrite = CollectTermsSolvedRewrite {
            equation: Equation {
                lhs,
                rhs,
                op: RelOp::Eq,
            },
            items: vec![StrategyExecutionItem {
                equation: Equation {
                    lhs,
                    rhs,
                    op: RelOp::Eq,
                },
                description: "Subtract rhs from both sides".to_string(),
            }],
        };
        let mut runtime = TestCollectTermsRuntime {
            solve_result: (
                SolutionSet::Discrete(vec![rhs]),
                vec!["runtime-only-sub-step".to_string()],
            ),
            last_var: None,
        };

        let solved = solve_collect_terms_rewrite_pipeline_with_item_runtime(
            rewrite,
            "x",
            false,
            &mut runtime,
        )
        .expect("runtime collect pipeline should succeed");

        assert_eq!(runtime.last_var.as_deref(), Some("x"));
        assert_eq!(solved.solution_set, SolutionSet::Discrete(vec![rhs]));
        assert_eq!(solved.steps, vec!["runtime-only-sub-step".to_string()]);
    }

    #[test]
    fn solve_rational_exponent_rewrite_with_runs_solver_on_rewritten_equation() {
        let mut ctx = Context::new();
        let lhs = ctx.var("lhs");
        let rhs = ctx.var("rhs");
        let rewrite = RationalExponentSolvedRewrite {
            equation: Equation {
                lhs,
                rhs,
                op: RelOp::Eq,
            },
            items: vec![],
        };

        let solved = solve_rational_exponent_rewrite_with(rewrite, |equation| {
            assert_eq!(equation.lhs, lhs);
            assert_eq!(equation.rhs, rhs);
            Ok::<_, ()>("ok")
        })
        .expect("must solve rewrite");

        assert_eq!(solved.solved, "ok");
    }

    #[test]
    fn solve_rational_exponent_rewrite_with_item_passes_item_to_solver() {
        let mut ctx = Context::new();
        let lhs = ctx.var("lhs");
        let rhs = ctx.var("rhs");
        let rewrite = RationalExponentSolvedRewrite {
            equation: Equation {
                lhs,
                rhs,
                op: RelOp::Eq,
            },
            items: vec![StrategyExecutionItem {
                equation: Equation {
                    lhs,
                    rhs,
                    op: RelOp::Eq,
                },
                description: "Raise both sides to power 2 to eliminate fractional exponent"
                    .to_string(),
            }],
        };

        let mut seen_desc = None;
        let solved = solve_rational_exponent_rewrite_with_item(rewrite, |item, equation| {
            seen_desc = item.map(|entry| entry.description);
            assert_eq!(equation.lhs, lhs);
            assert_eq!(equation.rhs, rhs);
            Ok::<_, ()>("ok")
        })
        .expect("must solve rewrite");

        assert_eq!(solved.solved, "ok");
        assert_eq!(
            seen_desc,
            Some("Raise both sides to power 2 to eliminate fractional exponent".to_string())
        );
    }
}
