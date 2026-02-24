//! Quadratic formula didactic substep construction.
//!
//! This module builds the symbolic derivation steps (completing the square)
//! independent of engine-specific step types/renderers.

use crate::isolation_utils::contains_var;
use crate::solution_set::sort_and_dedup_exprs;
use cas_ast::{Context, Equation, Expr, ExprId, RelOp, SolutionSet};

/// Main didactic narration for quadratic-strategy activation.
pub const QUADRATIC_FORMULA_MAIN_STEP_DESCRIPTION: &str =
    "Detected quadratic equation. Applying quadratic formula.";

/// Narration for factored zero-product entrypoint.
pub fn factorized_equation_message(expr_display: &str) -> String {
    format!("Factorized equation: {} = 0", expr_display)
}

/// Narration for solving one zero-product factor.
pub fn solve_factor_message(factor_display: &str) -> String {
    format!("Solve factor: {} = 0", factor_display)
}

/// Didactic step payload for top-level quadratic strategy narration.
#[derive(Debug, Clone, PartialEq)]
pub struct QuadraticDidacticStep {
    pub description: String,
    pub equation_after: Equation,
}

/// One executable quadratic item aligned with a didactic payload.
#[derive(Debug, Clone, PartialEq)]
pub struct QuadraticExecutionItem {
    pub equation: Equation,
    pub description: String,
}

impl QuadraticExecutionItem {
    /// User-facing narration for this execution item.
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Executable batch for zero-product factor solving:
/// `factor_i = 0` equations plus matching didactic steps.
#[derive(Debug, Clone, PartialEq)]
pub struct ZeroProductFactorExecutionPlan {
    pub equations: Vec<Equation>,
    pub items: Vec<ZeroProductFactorExecutionItem>,
}

/// One factor-solving execution item.
#[derive(Debug, Clone, PartialEq)]
pub struct ZeroProductFactorExecutionItem {
    pub equation: Equation,
    pub description: String,
}

/// Collect didactic steps for one zero-product factor execution item.
pub fn collect_zero_product_factor_item_didactic_steps(
    item: &ZeroProductFactorExecutionItem,
) -> Vec<QuadraticDidacticStep> {
    vec![QuadraticDidacticStep {
        description: item.description.clone(),
        equation_after: item.equation.clone(),
    }]
}

/// Collect execution items for one zero-product factor item.
pub fn collect_zero_product_factor_item_execution_items(
    item: &ZeroProductFactorExecutionItem,
) -> Vec<QuadraticExecutionItem> {
    collect_zero_product_factor_item_didactic_steps(item)
        .into_iter()
        .map(|didactic| QuadraticExecutionItem {
            equation: item.equation.clone(),
            description: didactic.description,
        })
        .collect()
}

/// Executable payload for factorized zero-product solving:
/// top factorized entry step + per-factor equations and didactic steps.
#[derive(Debug, Clone, PartialEq)]
pub struct FactorizedZeroProductExecutionPlan {
    pub entry: QuadraticDidacticStep,
    pub factors: ZeroProductFactorExecutionPlan,
}

/// Build didactic step for a factorized equation entrypoint.
pub fn build_factorized_equation_step_with<F>(
    equation_after: Equation,
    factorized_expr: ExprId,
    mut render_expr: F,
) -> QuadraticDidacticStep
where
    F: FnMut(ExprId) -> String,
{
    QuadraticDidacticStep {
        description: factorized_equation_message(&render_expr(factorized_expr)),
        equation_after,
    }
}

/// Build didactic step for solving a specific zero-product factor.
pub fn build_solve_factor_step_with<F>(
    equation_after: Equation,
    factor_expr: ExprId,
    mut render_expr: F,
) -> QuadraticDidacticStep
where
    F: FnMut(ExprId) -> String,
{
    QuadraticDidacticStep {
        description: solve_factor_message(&render_expr(factor_expr)),
        equation_after,
    }
}

/// Build didactic steps for a batch of zero-product factor equations.
pub fn build_solve_factor_steps_with<F>(
    factor_equations: &[Equation],
    mut render_expr: F,
) -> Vec<QuadraticDidacticStep>
where
    F: FnMut(ExprId) -> String,
{
    factor_equations
        .iter()
        .cloned()
        .map(|eq| build_solve_factor_step_with(eq.clone(), eq.lhs, &mut render_expr))
        .collect()
}

/// Build execution payload for zero-product factor solving by:
/// 1) collecting relevant `factor = 0` equations
/// 2) generating matching didactic steps
pub fn build_zero_product_factor_execution_with<F>(
    ctx: &Context,
    factors: &[ExprId],
    var: &str,
    zero: ExprId,
    render_expr: F,
) -> ZeroProductFactorExecutionPlan
where
    F: FnMut(ExprId) -> String,
{
    let equations = collect_zero_product_factor_equations(ctx, factors, var, zero);
    let didactic = build_solve_factor_steps_with(&equations, render_expr);
    let items = didactic
        .into_iter()
        .map(|step| ZeroProductFactorExecutionItem {
            equation: step.equation_after,
            description: step.description,
        })
        .collect();
    ZeroProductFactorExecutionPlan { equations, items }
}

/// Materialize zero-product factor equations without didactic payload.
///
/// This is useful when caller does not collect user-facing steps.
pub fn materialize_zero_product_factor_execution(
    ctx: &Context,
    factors: &[ExprId],
    var: &str,
    zero: ExprId,
) -> ZeroProductFactorExecutionPlan {
    let equations = collect_zero_product_factor_equations(ctx, factors, var, zero);
    ZeroProductFactorExecutionPlan {
        equations,
        items: vec![],
    }
}

/// Collect zero-product factor execution items (`factor = 0`) in execution order.
pub fn collect_zero_product_factor_execution_items(
    execution: &ZeroProductFactorExecutionPlan,
) -> Vec<ZeroProductFactorExecutionItem> {
    execution.items.clone()
}

/// Solve each zero-product factor equation while passing optional aligned
/// didactic execution item to the caller-provided callback.
pub fn solve_zero_product_factor_execution_with_items<E, T, FSolve>(
    execution: &ZeroProductFactorExecutionPlan,
    mut solve_factor: FSolve,
) -> Result<Vec<T>, E>
where
    FSolve: FnMut(Option<ZeroProductFactorExecutionItem>, &Equation) -> Result<T, E>,
{
    let mut solved = Vec::with_capacity(execution.equations.len());
    for (idx, equation) in execution.equations.iter().enumerate() {
        let item = execution.items.get(idx).cloned();
        solved.push(solve_factor(item, equation)?);
    }
    Ok(solved)
}

/// Aggregate solution sets obtained from solving each `factor = 0` branch.
#[derive(Debug, Clone, PartialEq)]
pub enum ZeroProductFactorSolutionAggregate {
    /// Every factor branch yielded discrete roots (or empty), merged and deduplicated.
    Discrete(Vec<ExprId>),
    /// One factor branch is identically zero (`0 = 0`) so whole equation is true.
    AllReals,
    /// At least one branch returned non-discrete data (residual/conditional/interval).
    NonDiscrete,
}

/// Merge per-factor solution sets from zero-product solving.
pub fn aggregate_zero_product_factor_solution_sets(
    ctx: &Context,
    factor_solution_sets: impl IntoIterator<Item = SolutionSet>,
) -> ZeroProductFactorSolutionAggregate {
    let mut all_solutions = Vec::new();
    for set in factor_solution_sets {
        match set {
            SolutionSet::Discrete(sols) => all_solutions.extend(sols),
            SolutionSet::Empty => {}
            SolutionSet::AllReals => return ZeroProductFactorSolutionAggregate::AllReals,
            _ => return ZeroProductFactorSolutionAggregate::NonDiscrete,
        }
    }
    sort_and_dedup_exprs(ctx, &mut all_solutions);
    ZeroProductFactorSolutionAggregate::Discrete(all_solutions)
}

/// Convert zero-product aggregate outcome into a `SolutionSet`, using
/// `residual_expr` when at least one branch returned non-discrete data.
pub fn finalize_zero_product_factor_solution_set(
    aggregate: ZeroProductFactorSolutionAggregate,
    residual_expr: ExprId,
) -> SolutionSet {
    match aggregate {
        ZeroProductFactorSolutionAggregate::Discrete(solutions) => SolutionSet::Discrete(solutions),
        ZeroProductFactorSolutionAggregate::AllReals => SolutionSet::AllReals,
        ZeroProductFactorSolutionAggregate::NonDiscrete => SolutionSet::Residual(residual_expr),
    }
}

/// Build full execution payload for factorized zero-product solving:
/// 1) "factorized equation" entry step
/// 2) per-factor `factor = 0` equations + didactic steps
pub fn build_factorized_zero_product_execution_with<F>(
    ctx: &Context,
    factorized_expr: ExprId,
    factors: &[ExprId],
    var: &str,
    zero: ExprId,
    mut render_expr: F,
) -> FactorizedZeroProductExecutionPlan
where
    F: FnMut(ExprId) -> String,
{
    let entry = build_factorized_equation_step_with(
        Equation {
            lhs: factorized_expr,
            rhs: zero,
            op: RelOp::Eq,
        },
        factorized_expr,
        &mut render_expr,
    );
    let factors = build_zero_product_factor_execution_with(ctx, factors, var, zero, render_expr);
    FactorizedZeroProductExecutionPlan { entry, factors }
}

/// Materialize factorized zero-product execution without didactic payload.
///
/// Entry description is intentionally empty because it is only consumed when
/// didactic item collection is enabled.
pub fn materialize_factorized_zero_product_execution(
    ctx: &Context,
    factorized_expr: ExprId,
    factors: &[ExprId],
    var: &str,
    zero: ExprId,
) -> FactorizedZeroProductExecutionPlan {
    FactorizedZeroProductExecutionPlan {
        entry: QuadraticDidacticStep {
            description: String::new(),
            equation_after: Equation {
                lhs: factorized_expr,
                rhs: zero,
                op: RelOp::Eq,
            },
        },
        factors: materialize_zero_product_factor_execution(ctx, factors, var, zero),
    }
}

/// Build factorized zero-product execution, optionally including didactic items.
pub fn build_factorized_zero_product_execution_with_optional_items<F>(
    ctx: &Context,
    factorized_expr: ExprId,
    factors: &[ExprId],
    var: &str,
    zero: ExprId,
    include_items: bool,
    render_expr: F,
) -> FactorizedZeroProductExecutionPlan
where
    F: FnMut(ExprId) -> String,
{
    if include_items {
        build_factorized_zero_product_execution_with(
            ctx,
            factorized_expr,
            factors,
            var,
            zero,
            render_expr,
        )
    } else {
        materialize_factorized_zero_product_execution(ctx, factorized_expr, factors, var, zero)
    }
}

/// Runtime adapter for building factorized zero-product execution payloads.
pub trait FactorizedZeroProductPlanRuntime {
    fn render_expr(&mut self, expr: ExprId) -> String;
}

impl<F> FactorizedZeroProductPlanRuntime for F
where
    F: FnMut(ExprId) -> String,
{
    fn render_expr(&mut self, expr: ExprId) -> String {
        self(expr)
    }
}

/// Runtime-based variant of `build_factorized_zero_product_execution_with_optional_items`.
pub fn build_factorized_zero_product_execution_with_optional_items_runtime<R>(
    ctx: &Context,
    factorized_expr: ExprId,
    factors: &[ExprId],
    var: &str,
    zero: ExprId,
    include_items: bool,
    runtime: &mut R,
) -> FactorizedZeroProductExecutionPlan
where
    R: FactorizedZeroProductPlanRuntime,
{
    if include_items {
        build_factorized_zero_product_execution_with(
            ctx,
            factorized_expr,
            factors,
            var,
            zero,
            |id| runtime.render_expr(id),
        )
    } else {
        materialize_factorized_zero_product_execution(ctx, factorized_expr, factors, var, zero)
    }
}

/// Collect factorized entry didactic step in display order.
pub fn collect_factorized_zero_product_entry_didactic_steps(
    execution: &FactorizedZeroProductExecutionPlan,
) -> Vec<QuadraticDidacticStep> {
    vec![execution.entry.clone()]
}

/// Collect factorized-entry execution items in display order.
pub fn collect_factorized_zero_product_entry_execution_items(
    execution: &FactorizedZeroProductExecutionPlan,
) -> Vec<QuadraticExecutionItem> {
    collect_factorized_zero_product_entry_didactic_steps(execution)
        .into_iter()
        .map(|didactic| QuadraticExecutionItem {
            equation: didactic.equation_after.clone(),
            description: didactic.description,
        })
        .collect()
}

/// Return the first factorized-entry execution item, if present.
pub fn first_factorized_zero_product_entry_execution_item(
    execution: &FactorizedZeroProductExecutionPlan,
) -> Option<QuadraticExecutionItem> {
    collect_factorized_zero_product_entry_execution_items(execution)
        .into_iter()
        .next()
}

/// Solved payload for factorized zero-product execution:
/// per-factor callback outputs plus concatenated didactic steps.
#[derive(Debug, Clone, PartialEq)]
pub struct FactorizedZeroProductExecutionSolved<TFactor, TStep> {
    pub solved_factors: Vec<TFactor>,
    pub steps: Vec<TStep>,
}

/// Solve factorized zero-product execution while optionally mapping:
/// 1) the factorized-entry item
/// 2) each per-factor execution item.
pub fn solve_factorized_zero_product_execution_pipeline_with_items<
    E,
    TFactor,
    TStep,
    FSolveFactor,
    FEntryStep,
    FFactorStep,
>(
    execution: &FactorizedZeroProductExecutionPlan,
    include_items: bool,
    mut solve_factor: FSolveFactor,
    mut map_entry_item_to_step: FEntryStep,
    mut map_factor_item_to_step: FFactorStep,
) -> Result<FactorizedZeroProductExecutionSolved<TFactor, TStep>, E>
where
    FSolveFactor: FnMut(&Equation) -> Result<TFactor, E>,
    FEntryStep: FnMut(QuadraticExecutionItem) -> TStep,
    FFactorStep: FnMut(ZeroProductFactorExecutionItem) -> TStep,
{
    let mut steps = Vec::new();
    if include_items {
        if let Some(item) = first_factorized_zero_product_entry_execution_item(execution) {
            steps.push(map_entry_item_to_step(item));
        }
    }
    let solved_factors =
        solve_zero_product_factor_execution_with_items(&execution.factors, |item, equation| {
            if include_items {
                if let Some(item) = item {
                    steps.push(map_factor_item_to_step(item));
                }
            }
            solve_factor(equation)
        })?;
    Ok(FactorizedZeroProductExecutionSolved {
        solved_factors,
        steps,
    })
}

/// Runtime adapter for solving and mapping factorized zero-product execution.
pub trait FactorizedZeroProductExecutionRuntime<E, TFactor, TStep> {
    fn solve_factor(&mut self, equation: &Equation) -> Result<TFactor, E>;
    fn map_entry_item_to_step(&mut self, item: QuadraticExecutionItem) -> TStep;
    fn map_factor_item_to_step(&mut self, item: ZeroProductFactorExecutionItem) -> TStep;
}

/// Runtime-based variant of
/// `solve_factorized_zero_product_execution_pipeline_with_items`.
pub fn solve_factorized_zero_product_execution_pipeline_with_items_runtime<E, TFactor, TStep, R>(
    execution: &FactorizedZeroProductExecutionPlan,
    include_items: bool,
    runtime: &mut R,
) -> Result<FactorizedZeroProductExecutionSolved<TFactor, TStep>, E>
where
    R: FactorizedZeroProductExecutionRuntime<E, TFactor, TStep>,
{
    let mut steps = Vec::new();
    if include_items {
        if let Some(item) = first_factorized_zero_product_entry_execution_item(execution) {
            steps.push(runtime.map_entry_item_to_step(item));
        }
    }

    let solved_factors =
        solve_zero_product_factor_execution_with_items(&execution.factors, |item, equation| {
            if include_items {
                if let Some(item) = item {
                    steps.push(runtime.map_factor_item_to_step(item));
                }
            }
            runtime.solve_factor(equation)
        })?;

    Ok(FactorizedZeroProductExecutionSolved {
        solved_factors,
        steps,
    })
}

/// Build and solve factorized zero-product execution with optional didactic
/// payload enabled by `include_items`.
#[allow(clippy::too_many_arguments)]
pub fn solve_factorized_zero_product_pipeline_with_optional_items<
    E,
    TFactor,
    TStep,
    FRenderExpr,
    FSolveFactor,
    FEntryStep,
    FFactorStep,
>(
    ctx: &Context,
    factorized_expr: ExprId,
    factors: &[ExprId],
    var: &str,
    zero: ExprId,
    include_items: bool,
    render_expr: FRenderExpr,
    solve_factor: FSolveFactor,
    map_entry_item_to_step: FEntryStep,
    map_factor_item_to_step: FFactorStep,
) -> Result<FactorizedZeroProductExecutionSolved<TFactor, TStep>, E>
where
    FRenderExpr: FnMut(ExprId) -> String,
    FSolveFactor: FnMut(&Equation) -> Result<TFactor, E>,
    FEntryStep: FnMut(QuadraticExecutionItem) -> TStep,
    FFactorStep: FnMut(ZeroProductFactorExecutionItem) -> TStep,
{
    let execution = if include_items {
        build_factorized_zero_product_execution_with(
            ctx,
            factorized_expr,
            factors,
            var,
            zero,
            render_expr,
        )
    } else {
        materialize_factorized_zero_product_execution(ctx, factorized_expr, factors, var, zero)
    };
    solve_factorized_zero_product_execution_pipeline_with_items(
        &execution,
        include_items,
        solve_factor,
        map_entry_item_to_step,
        map_factor_item_to_step,
    )
}

/// Solved payload for factorized zero-product strategy:
/// final `SolutionSet` plus flattened didactic steps.
#[derive(Debug, Clone, PartialEq)]
pub struct FactorizedZeroProductStrategySolved<TStep> {
    pub solution_set: SolutionSet,
    pub steps: Vec<TStep>,
}

/// Finalize a solved factorized zero-product execution into a single
/// `SolutionSet` plus flattened step list.
pub fn finalize_factorized_zero_product_strategy_solved<TStep>(
    ctx: &Context,
    solved: FactorizedZeroProductExecutionSolved<(SolutionSet, Vec<TStep>), TStep>,
    residual_expr_for_non_discrete: ExprId,
    zero: ExprId,
) -> FactorizedZeroProductStrategySolved<TStep> {
    let mut steps = solved.steps;
    let mut factor_solution_sets = Vec::new();
    for (solution_set, mut factor_steps) in solved.solved_factors {
        steps.append(&mut factor_steps);
        factor_solution_sets.push(solution_set);
    }

    let aggregate = aggregate_zero_product_factor_solution_sets(ctx, factor_solution_sets);
    let needs_residual = matches!(aggregate, ZeroProductFactorSolutionAggregate::NonDiscrete);
    let residual = if needs_residual {
        residual_expr_for_non_discrete
    } else {
        zero
    };
    let solution_set = finalize_zero_product_factor_solution_set(aggregate, residual);

    FactorizedZeroProductStrategySolved {
        solution_set,
        steps,
    }
}

/// Solve factorized zero-product strategy end-to-end:
/// 1) Build/solve factor equations with optional didactic items
/// 2) Aggregate per-factor solution sets
/// 3) Finalize residual fallback for non-discrete outcomes.
#[allow(clippy::too_many_arguments)]
pub fn solve_factorized_zero_product_strategy_pipeline_with_optional_items<
    E,
    TStep,
    FRenderExpr,
    FSolveFactor,
    FEntryStep,
    FFactorStep,
>(
    ctx: &Context,
    factorized_expr: ExprId,
    factors: &[ExprId],
    var: &str,
    zero: ExprId,
    include_items: bool,
    residual_expr_for_non_discrete: ExprId,
    render_expr: FRenderExpr,
    solve_factor: FSolveFactor,
    map_entry_item_to_step: FEntryStep,
    map_factor_item_to_step: FFactorStep,
) -> Result<FactorizedZeroProductStrategySolved<TStep>, E>
where
    FRenderExpr: FnMut(ExprId) -> String,
    FSolveFactor: FnMut(&Equation) -> Result<(SolutionSet, Vec<TStep>), E>,
    FEntryStep: FnMut(QuadraticExecutionItem) -> TStep,
    FFactorStep: FnMut(ZeroProductFactorExecutionItem) -> TStep,
{
    let solved = solve_factorized_zero_product_pipeline_with_optional_items(
        ctx,
        factorized_expr,
        factors,
        var,
        zero,
        include_items,
        render_expr,
        solve_factor,
        map_entry_item_to_step,
        map_factor_item_to_step,
    )?;
    Ok(finalize_factorized_zero_product_strategy_solved(
        ctx,
        solved,
        residual_expr_for_non_discrete,
        zero,
    ))
}

/// Build the top-level "quadratic formula" strategy step payload.
pub fn build_quadratic_main_step(equation_after: Equation) -> QuadraticDidacticStep {
    QuadraticDidacticStep {
        description: QUADRATIC_FORMULA_MAIN_STEP_DESCRIPTION.to_string(),
        equation_after,
    }
}

/// Collect quadratic-main didactic step in display order.
pub fn collect_quadratic_main_didactic_steps(
    step: &QuadraticDidacticStep,
) -> Vec<QuadraticDidacticStep> {
    vec![step.clone()]
}

/// Collect top-level quadratic execution items in display order.
pub fn collect_quadratic_main_execution_items(
    step: &QuadraticDidacticStep,
) -> Vec<QuadraticExecutionItem> {
    collect_quadratic_main_didactic_steps(step)
        .into_iter()
        .map(|didactic| QuadraticExecutionItem {
            equation: didactic.equation_after.clone(),
            description: didactic.description,
        })
        .collect()
}

/// Build top-level quadratic execution items directly from the resulting equation.
pub fn build_quadratic_main_execution_items(
    equation_after: Equation,
) -> Vec<QuadraticExecutionItem> {
    collect_quadratic_main_execution_items(&build_quadratic_main_step(equation_after))
}

/// Collect zero-product factor equations `factor = 0` that are relevant for
/// the target variable.
pub fn collect_zero_product_factor_equations(
    ctx: &Context,
    factors: &[ExprId],
    var: &str,
    zero: ExprId,
) -> Vec<Equation> {
    factors
        .iter()
        .copied()
        .filter(|factor| contains_var(ctx, *factor, var))
        .map(|factor| Equation {
            lhs: factor,
            rhs: zero,
            op: RelOp::Eq,
        })
        .collect()
}

/// Core didactic step payload for quadratic derivations.
#[derive(Debug, Clone, PartialEq)]
pub struct DidacticSubstep {
    pub description: String,
    pub equation_after: Equation,
}

/// One executable quadratic-substep item aligned with didactic payload.
#[derive(Debug, Clone, PartialEq)]
pub struct QuadraticSubstepExecutionItem {
    pub equation: Equation,
    pub description: String,
}

impl QuadraticSubstepExecutionItem {
    /// User-facing narration for this execution item.
    pub fn description(&self) -> &str {
        &self.description
    }
}

fn quadratic_substep_execution_item_from_didactic_step(
    step: DidacticSubstep,
) -> QuadraticSubstepExecutionItem {
    QuadraticSubstepExecutionItem {
        equation: step.equation_after,
        description: step.description,
    }
}

/// Check if an expression is a literal zero (Number(0)).
fn is_zero_literal(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if *n == num_rational::BigRational::from_integer(0.into()))
}

/// Check if an expression is a numeric constant (Number).
fn is_numeric(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(_))
}

/// Build didactic substeps for solving a quadratic equation.
///
/// `render_expr` is injected by caller (engine/CLI) so this module remains
/// independent of formatting crates.
pub fn build_quadratic_substeps_with<F>(
    ctx: &mut Context,
    var: &str,
    a: ExprId,
    b: ExprId,
    c: ExprId,
    is_real_only: bool,
    mut render_expr: F,
) -> Vec<DidacticSubstep>
where
    F: FnMut(&Context, ExprId) -> String,
{
    let mut steps = Vec::new();
    let zero = ctx.num(0);
    let one = ctx.num(1);
    let two = ctx.num(2);
    let four = ctx.num(4);

    // Check if all coefficients are numeric - if so, don't show algebraic derivation.
    let all_numeric = is_numeric(ctx, a) && is_numeric(ctx, b) && is_numeric(ctx, c);

    // Peephole optimization: check if c is literally zero.
    let c_is_zero = is_zero_literal(ctx, c);

    // Build x variable.
    let x = ctx.var(var);

    // Build x².
    let x2 = ctx.add(Expr::Pow(x, two));

    // =========================================================================
    // NUMERIC CASE: when a, b, c are all constants, show natural algebraic steps.
    // =========================================================================
    if all_numeric {
        // Build the polynomial expression for display.
        let ax2 = ctx.add(Expr::Mul(a, x2));
        let bx = ctx.add(Expr::Mul(b, x));
        let ax2_plus_bx = ctx.add(Expr::Add(ax2, bx));
        let _poly = ctx.add(Expr::Add(ax2_plus_bx, c));

        // CASE 1: c = 0 -> factor out x.
        if c_is_zero {
            // x(ax + b) = 0.
            let ax = ctx.add(Expr::Mul(a, x));
            let inner = ctx.add(Expr::Add(ax, b)); // ax + b
            let factored_form = ctx.add(Expr::Mul(x, inner)); // x(ax + b)

            steps.push(DidacticSubstep {
                description: "Factorizar x común".to_string(),
                equation_after: Equation {
                    lhs: factored_form,
                    rhs: zero,
                    op: RelOp::Eq,
                },
            });

            steps.push(DidacticSubstep {
                description: "Producto igual a cero: algún factor es cero".to_string(),
                equation_after: Equation {
                    lhs: x,
                    rhs: zero,
                    op: RelOp::Eq,
                },
            });

            let neg_b = ctx.add(Expr::Neg(b));
            let sol2 = ctx.add(Expr::Div(neg_b, a));

            steps.push(DidacticSubstep {
                description: format!("Resolver {} = 0", render_expr(ctx, inner)),
                equation_after: Equation {
                    lhs: x,
                    rhs: sol2,
                    op: RelOp::Eq,
                },
            });

            return steps;
        }
        // General numeric case (c != 0): fall through to full derivation.
    }

    // =========================================================================
    // Step 1: Identify the equation form.
    // =========================================================================
    let ax2 = ctx.add(Expr::Mul(a, x2));
    let bx = ctx.add(Expr::Mul(b, x));
    let ax2_plus_bx = ctx.add(Expr::Add(ax2, bx));
    let poly = ctx.add(Expr::Add(ax2_plus_bx, c));

    steps.push(DidacticSubstep {
        description: format!(
            "Identificar forma cuadrática: a·x² + b·x + c = 0 con a = {}, b = {}, c = {}",
            render_expr(ctx, a),
            render_expr(ctx, b),
            render_expr(ctx, c)
        ),
        equation_after: Equation {
            lhs: poly,
            rhs: zero,
            op: RelOp::Eq,
        },
    });

    // =========================================================================
    // Step 2: Divide both sides by a.
    // =========================================================================
    let b_over_a = ctx.add(Expr::Div(b, a));
    let c_over_a = if c_is_zero {
        zero
    } else {
        ctx.add(Expr::Div(c, a))
    };
    let bax = ctx.add(Expr::Mul(b_over_a, x));
    let x2_plus_bax = ctx.add(Expr::Add(x2, bax));
    let step2_lhs = if c_is_zero {
        x2_plus_bax
    } else {
        ctx.add(Expr::Add(x2_plus_bax, c_over_a))
    };

    steps.push(DidacticSubstep {
        description: "Dividir ambos lados por a (requiere a ≠ 0)".to_string(),
        equation_after: Equation {
            lhs: step2_lhs,
            rhs: zero,
            op: RelOp::Eq,
        },
    });

    // =========================================================================
    // Step 3: Move constant to RHS.
    // =========================================================================
    let step3_rhs = if c_is_zero {
        zero
    } else {
        ctx.add(Expr::Neg(c_over_a))
    };
    let step3_lhs = x2_plus_bax;

    steps.push(DidacticSubstep {
        description: "Mover término constante al lado derecho".to_string(),
        equation_after: Equation {
            lhs: step3_lhs,
            rhs: step3_rhs,
            op: RelOp::Eq,
        },
    });

    // =========================================================================
    // Step 4: Complete the square.
    // =========================================================================
    let two_a = ctx.add(Expr::Mul(two, a));
    let b_over_2a = ctx.add(Expr::Div(b, two_a));
    let b_over_2a_sq = ctx.add(Expr::Pow(b_over_2a, two));
    let step4_lhs = ctx.add(Expr::Add(step3_lhs, b_over_2a_sq));

    let step4_rhs = if c_is_zero {
        b_over_2a_sq
    } else {
        let neg_c_a = ctx.add(Expr::Neg(c_over_a));
        ctx.add(Expr::Add(b_over_2a_sq, neg_c_a))
    };

    steps.push(DidacticSubstep {
        description: "Completar el cuadrado: sumar (b/2a)² a ambos lados".to_string(),
        equation_after: Equation {
            lhs: step4_lhs,
            rhs: step4_rhs,
            op: RelOp::Eq,
        },
    });

    // =========================================================================
    // Step 5: Factor LHS as perfect square.
    // =========================================================================
    let x_plus_b_over_2a = ctx.add(Expr::Add(x, b_over_2a));
    let step5_lhs = ctx.add(Expr::Pow(x_plus_b_over_2a, two));

    let b2 = ctx.add(Expr::Pow(b, two));
    let discriminant = if c_is_zero {
        b2
    } else {
        let four_a = ctx.add(Expr::Mul(four, a));
        let four_ac = ctx.add(Expr::Mul(four_a, c));
        ctx.add(Expr::Sub(b2, four_ac))
    };
    let a2 = ctx.add(Expr::Pow(a, two));
    let four_a2 = ctx.add(Expr::Mul(four, a2));
    let step5_rhs = ctx.add(Expr::Div(discriminant, four_a2));

    steps.push(DidacticSubstep {
        description: "Escribir lado izquierdo como cuadrado perfecto".to_string(),
        equation_after: Equation {
            lhs: step5_lhs,
            rhs: step5_rhs,
            op: RelOp::Eq,
        },
    });

    // =========================================================================
    // Step 6: Take square root.
    // =========================================================================
    let half = ctx.add(Expr::Div(one, two));
    let sqrt_rhs = ctx.add(Expr::Pow(step5_rhs, half));
    let abs_lhs = ctx.call_builtin(cas_ast::BuiltinFn::Abs, vec![x_plus_b_over_2a]);

    steps.push(DidacticSubstep {
        description: "Tomar raíz cuadrada en ambos lados".to_string(),
        equation_after: Equation {
            lhs: abs_lhs,
            rhs: sqrt_rhs,
            op: RelOp::Eq,
        },
    });

    // =========================================================================
    // Step 7: Split absolute-value equation and isolate x.
    // =========================================================================
    let neg_b = ctx.add(Expr::Neg(b));
    let neg_b_over_2a = ctx.add(Expr::Div(neg_b, two_a));
    let plus_minus = ctx.call("PlusMinus", vec![neg_b_over_2a, sqrt_rhs]);

    let description = if is_real_only {
        "|u| = a se descompone en u = a y u = -a. Despejando x (requiere Δ ≥ 0)".to_string()
    } else {
        "|u| = a se descompone en u = a y u = -a. Despejando x".to_string()
    };

    steps.push(DidacticSubstep {
        description,
        equation_after: Equation {
            lhs: x,
            rhs: plus_minus,
            op: RelOp::Eq,
        },
    });

    steps
}

/// Build quadratic derivation execution items directly.
///
/// This mirrors `build_quadratic_substeps_with` but returns execution payloads
/// for engine consumers that should not depend on didactic step structs.
pub fn build_quadratic_substep_execution_items_with<F>(
    ctx: &mut Context,
    var: &str,
    a: ExprId,
    b: ExprId,
    c: ExprId,
    is_real_only: bool,
    render_expr: F,
) -> Vec<QuadraticSubstepExecutionItem>
where
    F: FnMut(&Context, ExprId) -> String,
{
    build_quadratic_substeps_with(ctx, var, a, b, c, is_real_only, render_expr)
        .into_iter()
        .map(quadratic_substep_execution_item_from_didactic_step)
        .collect()
}

/// Simplify every equation side in quadratic execution items with a caller-provided callback.
pub fn simplify_quadratic_substep_execution_items_with<F>(
    items: &mut [QuadraticSubstepExecutionItem],
    mut simplify_expr: F,
) where
    F: FnMut(ExprId) -> ExprId,
{
    for item in items.iter_mut() {
        item.equation.lhs = simplify_expr(item.equation.lhs);
        item.equation.rhs = simplify_expr(item.equation.rhs);
    }
}

/// Combined execution payload for quadratic strategy didactic output:
/// one main step plus simplified substeps.
#[derive(Debug, Clone, PartialEq)]
pub struct QuadraticMainWithSubstepsExecution {
    pub main_items: Vec<QuadraticExecutionItem>,
    pub substep_items: Vec<QuadraticSubstepExecutionItem>,
}

/// Build quadratic main-step execution and its substeps in one call.
///
/// Callers provide:
/// - a renderer for textual descriptions,
/// - a simplifier callback for equation sides in substeps.
#[allow(clippy::too_many_arguments)]
pub fn build_quadratic_main_with_substeps_execution_with<FR, FS>(
    ctx: &mut Context,
    var: &str,
    a: ExprId,
    b: ExprId,
    c: ExprId,
    is_real_only: bool,
    main_equation_after: Equation,
    render_expr: FR,
    simplify_expr: FS,
) -> QuadraticMainWithSubstepsExecution
where
    FR: FnMut(&Context, ExprId) -> String,
    FS: FnMut(ExprId) -> ExprId,
{
    let mut substep_items =
        build_quadratic_substep_execution_items_with(ctx, var, a, b, c, is_real_only, render_expr);
    simplify_quadratic_substep_execution_items_with(&mut substep_items, simplify_expr);
    let main_items = build_quadratic_main_execution_items(main_equation_after);
    QuadraticMainWithSubstepsExecution {
        main_items,
        substep_items,
    }
}

/// Materialize empty quadratic didactic execution.
///
/// This is used when caller is not collecting user-facing steps.
pub fn materialize_quadratic_main_with_substeps_execution() -> QuadraticMainWithSubstepsExecution {
    QuadraticMainWithSubstepsExecution {
        main_items: vec![],
        substep_items: vec![],
    }
}

/// Build quadratic main/substep execution with optional didactic payload.
///
/// When `include_items` is false, this does not invoke `render_expr`.
#[allow(clippy::too_many_arguments)]
pub fn build_quadratic_main_with_substeps_execution_with_optional_items<FR>(
    ctx: &mut Context,
    var: &str,
    a: ExprId,
    b: ExprId,
    c: ExprId,
    is_real_only: bool,
    main_equation_after: Equation,
    include_items: bool,
    render_expr: FR,
) -> QuadraticMainWithSubstepsExecution
where
    FR: FnMut(&Context, ExprId) -> String,
{
    if include_items {
        build_quadratic_main_with_substeps_execution_with(
            ctx,
            var,
            a,
            b,
            c,
            is_real_only,
            main_equation_after,
            render_expr,
            |id| id,
        )
    } else {
        materialize_quadratic_main_with_substeps_execution()
    }
}

/// Runtime adapter for building quadratic-main and substep execution payloads.
pub trait QuadraticMainWithSubstepsPlanRuntime {
    fn render_expr(&mut self, ctx: &Context, expr: ExprId) -> String;
}

impl<F> QuadraticMainWithSubstepsPlanRuntime for F
where
    F: FnMut(&Context, ExprId) -> String,
{
    fn render_expr(&mut self, ctx: &Context, expr: ExprId) -> String {
        self(ctx, expr)
    }
}

/// Runtime-based variant of
/// `build_quadratic_main_with_substeps_execution_with_optional_items`.
#[allow(clippy::too_many_arguments)]
pub fn build_quadratic_main_with_substeps_execution_with_optional_items_runtime<R>(
    ctx: &mut Context,
    var: &str,
    a: ExprId,
    b: ExprId,
    c: ExprId,
    is_real_only: bool,
    main_equation_after: Equation,
    include_items: bool,
    runtime: &mut R,
) -> QuadraticMainWithSubstepsExecution
where
    R: QuadraticMainWithSubstepsPlanRuntime,
{
    if include_items {
        build_quadratic_main_with_substeps_execution_with(
            ctx,
            var,
            a,
            b,
            c,
            is_real_only,
            main_equation_after,
            |core_ctx, id| runtime.render_expr(core_ctx, id),
            |id| id,
        )
    } else {
        materialize_quadratic_main_with_substeps_execution()
    }
}

/// Map quadratic main/substep execution items into caller-owned step payloads.
///
/// When `include_items` is `false`, no mapper callbacks are invoked and this
/// returns an empty vector.
pub fn solve_quadratic_main_with_substeps_execution_pipeline_with_items<S, SS, FMain, FSub>(
    execution: &QuadraticMainWithSubstepsExecution,
    include_items: bool,
    mut map_main_item_to_step: FMain,
    mut map_substep_item_to_step: FSub,
) -> Vec<S>
where
    SS: Clone,
    FMain: FnMut(QuadraticExecutionItem, Vec<SS>) -> S,
    FSub: FnMut(QuadraticSubstepExecutionItem) -> SS,
{
    if !include_items {
        return Vec::new();
    }

    let mut mapped_substeps = Vec::with_capacity(execution.substep_items.len());
    for substep in execution.substep_items.iter().cloned() {
        mapped_substeps.push(map_substep_item_to_step(substep));
    }

    let mut mapped_steps = Vec::with_capacity(execution.main_items.len());
    for main in execution.main_items.iter().cloned() {
        mapped_steps.push(map_main_item_to_step(main, mapped_substeps.clone()));
    }

    mapped_steps
}

/// Solve quadratic main/substep execution with optional item collection and
/// optional substep simplification callback.
///
/// When `include_items` is false, this does not invoke callbacks.
pub fn solve_quadratic_main_with_substeps_execution_pipeline_with_optional_items_and_simplification<
    S,
    SS,
    FSimplify,
    FMain,
    FSub,
>(
    execution: &mut QuadraticMainWithSubstepsExecution,
    include_items: bool,
    simplify_expr: FSimplify,
    map_main_item_to_step: FMain,
    map_substep_item_to_step: FSub,
) -> Vec<S>
where
    SS: Clone,
    FSimplify: FnMut(ExprId) -> ExprId,
    FMain: FnMut(QuadraticExecutionItem, Vec<SS>) -> S,
    FSub: FnMut(QuadraticSubstepExecutionItem) -> SS,
{
    if !include_items {
        return Vec::new();
    }
    simplify_quadratic_substep_execution_items_with(&mut execution.substep_items, simplify_expr);
    solve_quadratic_main_with_substeps_execution_pipeline_with_items(
        execution,
        true,
        map_main_item_to_step,
        map_substep_item_to_step,
    )
}

/// Runtime adapter for quadratic-main didactic execution pipelines.
pub trait QuadraticMainWithSubstepsRuntime<S, SS> {
    fn simplify_expr(&mut self, expr: ExprId) -> ExprId;
    fn map_main_item_to_step(&mut self, item: QuadraticExecutionItem, substeps: Vec<SS>) -> S;
    fn map_substep_item_to_step(&mut self, item: QuadraticSubstepExecutionItem) -> SS;
}

/// Runtime-based variant of
/// `solve_quadratic_main_with_substeps_execution_pipeline_with_optional_items_and_simplification`.
///
/// When `include_items` is false, this does not invoke runtime callbacks.
pub fn solve_quadratic_main_with_substeps_execution_pipeline_with_optional_items_and_simplification_runtime<
    S,
    SS,
    R,
>(
    execution: &mut QuadraticMainWithSubstepsExecution,
    include_items: bool,
    runtime: &mut R,
) -> Vec<S>
where
    SS: Clone,
    R: QuadraticMainWithSubstepsRuntime<S, SS>,
{
    if !include_items {
        return Vec::new();
    }

    simplify_quadratic_substep_execution_items_with(&mut execution.substep_items, |id| {
        runtime.simplify_expr(id)
    });

    let mut mapped_substeps = Vec::with_capacity(execution.substep_items.len());
    for substep in execution.substep_items.iter().cloned() {
        mapped_substeps.push(runtime.map_substep_item_to_step(substep));
    }

    let mut mapped_steps = Vec::with_capacity(execution.main_items.len());
    for main in execution.main_items.iter().cloned() {
        mapped_steps.push(runtime.map_main_item_to_step(main, mapped_substeps.clone()));
    }

    mapped_steps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_quadratic_substeps_numeric_general_case() {
        let mut ctx = Context::new();
        let a = ctx.num(1);
        let b = ctx.num(2);
        let c = ctx.num(1);

        let steps = build_quadratic_substeps_with(&mut ctx, "x", a, b, c, true, |_ctx, id| {
            format!("{:?}", id)
        });

        assert_eq!(steps.len(), 7);
        assert!(steps[0].description.contains("Identificar"));
        assert!(steps[1].description.contains("Dividir"));
        assert!(steps[2].description.contains("Mover"));
        assert!(steps[3].description.contains("Completar"));
        assert!(steps[4].description.contains("cuadrado perfecto"));
        assert!(steps[5].description.contains("raíz cuadrada"));
        assert!(steps[6].description.contains("descompone"));
    }

    #[test]
    fn build_quadratic_substep_execution_items_align_with_didactic_steps() {
        let mut ctx = Context::new();
        let a = ctx.num(1);
        let b = ctx.num(2);
        let c = ctx.num(1);

        let didactic = build_quadratic_substeps_with(&mut ctx, "x", a, b, c, true, |_ctx, id| {
            format!("{:?}", id)
        });
        let items = build_quadratic_substep_execution_items_with(
            &mut ctx,
            "x",
            a,
            b,
            c,
            true,
            |_ctx, id| format!("{:?}", id),
        );

        assert_eq!(items.len(), didactic.len());
        assert_eq!(items[0].equation, didactic[0].equation_after);
        assert_eq!(items[0].description, didactic[0].description);
    }

    #[test]
    fn simplify_quadratic_substep_execution_items_with_rewrites_both_sides() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let lhs = ctx.add(Expr::Add(x, one));
        let rhs = ctx.add(Expr::Mul(one, x));
        let zero = ctx.num(0);

        let mut items = vec![QuadraticSubstepExecutionItem {
            equation: Equation {
                lhs,
                rhs,
                op: RelOp::Eq,
            },
            description: "step".to_string(),
        }];

        simplify_quadratic_substep_execution_items_with(&mut items, |id| match ctx.get(id) {
            Expr::Add(_, _) => zero,
            Expr::Mul(_, _) => x,
            _ => id,
        });

        assert_eq!(items[0].equation.lhs, zero);
        assert_eq!(items[0].equation.rhs, x);
    }

    #[test]
    fn build_quadratic_main_with_substeps_execution_with_combines_main_and_substeps() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let a = one;
        let b = one;
        let c = one;
        let eq = Equation {
            lhs: x,
            rhs: ctx.num(0),
            op: RelOp::Eq,
        };

        let execution = build_quadratic_main_with_substeps_execution_with(
            &mut ctx,
            "x",
            a,
            b,
            c,
            true,
            eq.clone(),
            |_, id| format!("{:?}", id),
            |id| id,
        );

        assert_eq!(execution.main_items.len(), 1);
        assert_eq!(execution.main_items[0].equation, eq);
        assert!(!execution.substep_items.is_empty());
    }

    #[test]
    fn build_quadratic_main_with_substeps_execution_with_optional_items_includes_items_when_enabled(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let eq = Equation {
            lhs: x,
            rhs: ctx.num(0),
            op: RelOp::Eq,
        };

        let execution = build_quadratic_main_with_substeps_execution_with_optional_items(
            &mut ctx,
            "x",
            one,
            one,
            one,
            true,
            eq,
            true,
            |_, id| format!("{:?}", id),
        );

        assert_eq!(execution.main_items.len(), 1);
        assert!(!execution.substep_items.is_empty());
    }

    #[test]
    fn build_quadratic_main_with_substeps_execution_with_optional_items_skips_render_when_disabled()
    {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let eq = Equation {
            lhs: x,
            rhs: ctx.num(0),
            op: RelOp::Eq,
        };

        let execution = build_quadratic_main_with_substeps_execution_with_optional_items(
            &mut ctx,
            "x",
            one,
            one,
            one,
            true,
            eq,
            false,
            |_ctx, _id| -> String { panic!("renderer must not run when disabled") },
        );

        assert!(execution.main_items.is_empty());
        assert!(execution.substep_items.is_empty());
    }

    #[derive(Default)]
    struct TestQuadraticMainWithSubstepsPlanRuntime {
        render_calls: usize,
    }

    impl QuadraticMainWithSubstepsPlanRuntime for TestQuadraticMainWithSubstepsPlanRuntime {
        fn render_expr(&mut self, _ctx: &Context, expr: ExprId) -> String {
            self.render_calls += 1;
            format!("{:?}", expr)
        }
    }

    #[test]
    fn build_quadratic_main_with_substeps_execution_with_optional_items_runtime_includes_items_when_enabled(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let eq = Equation {
            lhs: x,
            rhs: ctx.num(0),
            op: RelOp::Eq,
        };
        let mut runtime = TestQuadraticMainWithSubstepsPlanRuntime::default();

        let execution = build_quadratic_main_with_substeps_execution_with_optional_items_runtime(
            &mut ctx,
            "x",
            one,
            one,
            one,
            true,
            eq,
            true,
            &mut runtime,
        );

        assert_eq!(execution.main_items.len(), 1);
        assert!(!execution.substep_items.is_empty());
        assert!(runtime.render_calls > 0);
    }

    #[test]
    fn build_quadratic_main_with_substeps_execution_with_optional_items_runtime_skips_render_when_disabled(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let eq = Equation {
            lhs: x,
            rhs: ctx.num(0),
            op: RelOp::Eq,
        };
        let mut runtime = TestQuadraticMainWithSubstepsPlanRuntime::default();

        let execution = build_quadratic_main_with_substeps_execution_with_optional_items_runtime(
            &mut ctx,
            "x",
            one,
            one,
            one,
            true,
            eq,
            false,
            &mut runtime,
        );

        assert!(execution.main_items.is_empty());
        assert!(execution.substep_items.is_empty());
        assert_eq!(runtime.render_calls, 0);
    }

    #[test]
    fn build_quadratic_substeps_numeric_c_zero_factoring_path() {
        let mut ctx = Context::new();
        let a = ctx.num(1);
        let b = ctx.num(2);
        let c = ctx.num(0);

        let steps = build_quadratic_substeps_with(&mut ctx, "x", a, b, c, true, |_ctx, id| {
            format!("{:?}", id)
        });

        assert_eq!(steps.len(), 3);
        assert!(steps[0].description.contains("Factorizar"));
        assert!(steps[1].description.contains("Producto"));
        assert!(steps[2].description.contains("Resolver"));
    }

    #[test]
    fn solve_quadratic_main_with_substeps_execution_pipeline_with_items_maps_when_enabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let eq = Equation {
            lhs: x,
            rhs: ctx.num(0),
            op: RelOp::Eq,
        };
        let execution = build_quadratic_main_with_substeps_execution_with(
            &mut ctx,
            "x",
            one,
            one,
            one,
            true,
            eq,
            |_, id| format!("{:?}", id),
            |id| id,
        );

        let mapped = solve_quadratic_main_with_substeps_execution_pipeline_with_items(
            &execution,
            true,
            |main, substeps: Vec<String>| (main.description, substeps.len()),
            |substep| substep.description,
        );

        assert_eq!(mapped.len(), execution.main_items.len());
        assert_eq!(mapped[0].0, QUADRATIC_FORMULA_MAIN_STEP_DESCRIPTION);
        assert!(!execution.substep_items.is_empty());
        assert_eq!(mapped[0].1, execution.substep_items.len());
    }

    #[test]
    fn solve_quadratic_main_with_substeps_execution_pipeline_with_items_omits_when_disabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let eq = Equation {
            lhs: x,
            rhs: ctx.num(0),
            op: RelOp::Eq,
        };
        let execution = build_quadratic_main_with_substeps_execution_with(
            &mut ctx,
            "x",
            one,
            one,
            one,
            true,
            eq,
            |_, id| format!("{:?}", id),
            |id| id,
        );

        let mapped = solve_quadratic_main_with_substeps_execution_pipeline_with_items(
            &execution,
            false,
            |_main, _substeps: Vec<()>| -> () { panic!("must not map when disabled") },
            |_substep| -> () { panic!("must not map when disabled") },
        );

        assert!(mapped.is_empty());
    }

    #[test]
    fn solve_quadratic_main_with_substeps_execution_pipeline_with_optional_items_and_simplification_maps_when_enabled(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let eq = Equation {
            lhs: x,
            rhs: ctx.num(0),
            op: RelOp::Eq,
        };
        let mut execution = build_quadratic_main_with_substeps_execution_with_optional_items(
            &mut ctx,
            "x",
            one,
            one,
            one,
            true,
            eq,
            true,
            |_, id| format!("{:?}", id),
        );

        let mapped =
            solve_quadratic_main_with_substeps_execution_pipeline_with_optional_items_and_simplification(
                &mut execution,
                true,
                |id| id,
                |main, substeps: Vec<String>| (main.description, substeps.len()),
                |substep| substep.description,
            );

        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].0, QUADRATIC_FORMULA_MAIN_STEP_DESCRIPTION);
        assert!(mapped[0].1 > 0);
    }

    #[test]
    fn solve_quadratic_main_with_substeps_execution_pipeline_with_optional_items_and_simplification_omits_when_disabled(
    ) {
        let mut execution = materialize_quadratic_main_with_substeps_execution();
        let mapped =
            solve_quadratic_main_with_substeps_execution_pipeline_with_optional_items_and_simplification(
                &mut execution,
                false,
                |_id| -> ExprId { panic!("simplifier must not run when disabled") },
                |_main, _substeps: Vec<()>| -> () { panic!("main mapper must not run when disabled") },
                |_substep| -> () { panic!("sub mapper must not run when disabled") },
            );

        assert!(mapped.is_empty());
    }

    #[derive(Default)]
    struct TestQuadraticMainWithSubstepsRuntime {
        simplify_calls: usize,
        main_map_calls: usize,
        sub_map_calls: usize,
    }

    impl QuadraticMainWithSubstepsRuntime<(String, usize), String>
        for TestQuadraticMainWithSubstepsRuntime
    {
        fn simplify_expr(&mut self, expr: ExprId) -> ExprId {
            self.simplify_calls += 1;
            expr
        }

        fn map_main_item_to_step(
            &mut self,
            item: QuadraticExecutionItem,
            substeps: Vec<String>,
        ) -> (String, usize) {
            self.main_map_calls += 1;
            (item.description, substeps.len())
        }

        fn map_substep_item_to_step(&mut self, item: QuadraticSubstepExecutionItem) -> String {
            self.sub_map_calls += 1;
            item.description
        }
    }

    #[test]
    fn solve_quadratic_main_with_substeps_execution_pipeline_with_optional_items_and_simplification_runtime_maps_when_enabled(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let eq = Equation {
            lhs: x,
            rhs: ctx.num(0),
            op: RelOp::Eq,
        };
        let mut execution = build_quadratic_main_with_substeps_execution_with_optional_items(
            &mut ctx,
            "x",
            one,
            one,
            one,
            true,
            eq,
            true,
            |_, id| format!("{:?}", id),
        );
        let expected_substeps = execution.substep_items.len();
        let expected_main = execution.main_items.len();
        let mut runtime = TestQuadraticMainWithSubstepsRuntime::default();

        let mapped =
            solve_quadratic_main_with_substeps_execution_pipeline_with_optional_items_and_simplification_runtime(
                &mut execution,
                true,
                &mut runtime,
            );

        assert_eq!(mapped.len(), expected_main);
        assert_eq!(mapped[0].0, QUADRATIC_FORMULA_MAIN_STEP_DESCRIPTION);
        assert_eq!(mapped[0].1, expected_substeps);
        assert_eq!(runtime.simplify_calls, expected_substeps * 2);
        assert_eq!(runtime.main_map_calls, expected_main);
        assert_eq!(runtime.sub_map_calls, expected_substeps);
    }

    #[test]
    fn solve_quadratic_main_with_substeps_execution_pipeline_with_optional_items_and_simplification_runtime_omits_when_disabled(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let eq = Equation {
            lhs: x,
            rhs: ctx.num(0),
            op: RelOp::Eq,
        };
        let mut execution = build_quadratic_main_with_substeps_execution_with_optional_items(
            &mut ctx,
            "x",
            one,
            one,
            one,
            true,
            eq,
            true,
            |_, id| format!("{:?}", id),
        );
        let mut runtime = TestQuadraticMainWithSubstepsRuntime::default();

        let mapped =
            solve_quadratic_main_with_substeps_execution_pipeline_with_optional_items_and_simplification_runtime(
                &mut execution,
                false,
                &mut runtime,
            );

        assert!(mapped.is_empty());
        assert_eq!(runtime.simplify_calls, 0);
        assert_eq!(runtime.main_map_calls, 0);
        assert_eq!(runtime.sub_map_calls, 0);
    }

    #[test]
    fn quadratic_main_step_description_stays_stable() {
        assert_eq!(
            QUADRATIC_FORMULA_MAIN_STEP_DESCRIPTION,
            "Detected quadratic equation. Applying quadratic formula."
        );
    }

    #[test]
    fn zero_product_messages_format_expected_text() {
        assert_eq!(
            factorized_equation_message("x*(x-1)"),
            "Factorized equation: x*(x-1) = 0"
        );
        assert_eq!(solve_factor_message("x-1"), "Solve factor: x-1 = 0");
    }

    #[test]
    fn step_builders_use_expected_messages_and_equations() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let eq = Equation {
            lhs: x,
            rhs: zero,
            op: RelOp::Eq,
        };

        let factored = build_factorized_equation_step_with(eq.clone(), x, |_id| "x".to_string());
        assert_eq!(factored.description, "Factorized equation: x = 0");
        assert_eq!(factored.equation_after, eq);

        let solve = build_solve_factor_step_with(eq.clone(), x, |_id| "x".to_string());
        assert_eq!(solve.description, "Solve factor: x = 0");
        assert_eq!(solve.equation_after, eq);

        let main = build_quadratic_main_step(eq.clone());
        assert_eq!(
            main.description,
            "Detected quadratic equation. Applying quadratic formula."
        );
        assert_eq!(main.equation_after, eq);
    }

    #[test]
    fn collect_zero_product_factor_equations_filters_non_variable_factors() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let k = ctx.var("k");
        let one = ctx.num(1);
        let x_minus_one = ctx.add(Expr::Sub(x, one));
        let factors = vec![k, x_minus_one];
        let zero = ctx.num(0);

        let eqs = collect_zero_product_factor_equations(&ctx, &factors, "x", zero);
        assert_eq!(eqs.len(), 1);
        assert_eq!(eqs[0].lhs, x_minus_one);
        assert_eq!(eqs[0].rhs, zero);
        assert_eq!(eqs[0].op, RelOp::Eq);
    }

    #[test]
    fn build_solve_factor_steps_with_builds_one_step_per_equation() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let zero = ctx.num(0);
        let equations = vec![
            Equation {
                lhs: x,
                rhs: zero,
                op: RelOp::Eq,
            },
            Equation {
                lhs: y,
                rhs: zero,
                op: RelOp::Eq,
            },
        ];

        let steps = build_solve_factor_steps_with(&equations, |_| "f".to_string());
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].description, "Solve factor: f = 0");
        assert_eq!(steps[1].description, "Solve factor: f = 0");
        assert_eq!(steps[0].equation_after, equations[0]);
        assert_eq!(steps[1].equation_after, equations[1]);
    }

    #[test]
    fn build_zero_product_factor_execution_with_builds_equations_and_didactic() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let one = ctx.num(1);
        let x_minus_one = ctx.add(Expr::Sub(x, one));
        let factors = vec![x_minus_one, y];
        let zero = ctx.num(0);

        let execution = build_zero_product_factor_execution_with(&ctx, &factors, "x", zero, |_| {
            "factor".to_string()
        });

        assert_eq!(execution.equations.len(), 1);
        assert_eq!(execution.equations[0].lhs, x_minus_one);
        assert_eq!(execution.equations[0].rhs, zero);
        assert_eq!(execution.equations[0].op, RelOp::Eq);
        assert_eq!(execution.items.len(), 1);
        assert_eq!(execution.items[0].description, "Solve factor: factor = 0");
        assert_eq!(execution.items[0].equation, execution.equations[0]);
    }

    #[test]
    fn materialize_zero_product_factor_execution_omits_items() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let one = ctx.num(1);
        let x_minus_one = ctx.add(Expr::Sub(x, one));
        let factors = vec![x_minus_one, y];
        let zero = ctx.num(0);

        let execution = materialize_zero_product_factor_execution(&ctx, &factors, "x", zero);

        assert_eq!(execution.equations.len(), 1);
        assert_eq!(execution.equations[0].lhs, x_minus_one);
        assert!(execution.items.is_empty());
    }

    #[test]
    fn build_factorized_zero_product_execution_with_builds_entry_and_factors() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_minus_one = ctx.add(Expr::Sub(x, one));
        let factored_expr = ctx.add(Expr::Mul(x, x_minus_one));
        let zero = ctx.num(0);
        let factors = vec![x, x_minus_one];

        let execution = build_factorized_zero_product_execution_with(
            &ctx,
            factored_expr,
            &factors,
            "x",
            zero,
            |_| "f".to_string(),
        );

        assert_eq!(execution.entry.description, "Factorized equation: f = 0");
        assert_eq!(
            execution.entry.equation_after,
            Equation {
                lhs: factored_expr,
                rhs: zero,
                op: RelOp::Eq
            }
        );
        assert_eq!(execution.factors.equations.len(), 2);
        assert_eq!(execution.factors.items.len(), 2);
        assert_eq!(
            execution.factors.items[0].description,
            "Solve factor: f = 0"
        );
    }

    #[test]
    fn materialize_factorized_zero_product_execution_omits_items() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_minus_one = ctx.add(Expr::Sub(x, one));
        let factored_expr = ctx.add(Expr::Mul(x, x_minus_one));
        let zero = ctx.num(0);
        let factors = vec![x, x_minus_one];

        let execution =
            materialize_factorized_zero_product_execution(&ctx, factored_expr, &factors, "x", zero);

        assert_eq!(execution.entry.equation_after.lhs, factored_expr);
        assert_eq!(execution.entry.equation_after.rhs, zero);
        assert_eq!(execution.entry.equation_after.op, RelOp::Eq);
        assert!(execution.entry.description.is_empty());
        assert_eq!(execution.factors.equations.len(), 2);
        assert!(execution.factors.items.is_empty());
    }

    #[test]
    fn build_factorized_zero_product_execution_with_optional_items_includes_didactic_when_enabled()
    {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_minus_one = ctx.add(Expr::Sub(x, one));
        let factored_expr = ctx.add(Expr::Mul(x, x_minus_one));
        let zero = ctx.num(0);
        let factors = vec![x, x_minus_one];

        let execution = build_factorized_zero_product_execution_with_optional_items(
            &ctx,
            factored_expr,
            &factors,
            "x",
            zero,
            true,
            |_| "f".to_string(),
        );

        assert_eq!(execution.entry.description, "Factorized equation: f = 0");
        assert_eq!(execution.factors.items.len(), 2);
    }

    #[test]
    fn build_factorized_zero_product_execution_with_optional_items_skips_render_when_disabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_minus_one = ctx.add(Expr::Sub(x, one));
        let factored_expr = ctx.add(Expr::Mul(x, x_minus_one));
        let zero = ctx.num(0);
        let factors = vec![x, x_minus_one];

        let execution = build_factorized_zero_product_execution_with_optional_items(
            &ctx,
            factored_expr,
            &factors,
            "x",
            zero,
            false,
            |_id| -> String { panic!("renderer must not run when items are disabled") },
        );

        assert!(execution.entry.description.is_empty());
        assert!(execution.factors.items.is_empty());
        assert_eq!(execution.factors.equations.len(), 2);
    }

    #[derive(Default)]
    struct TestFactorizedZeroProductPlanRuntime {
        render_calls: usize,
    }

    impl FactorizedZeroProductPlanRuntime for TestFactorizedZeroProductPlanRuntime {
        fn render_expr(&mut self, _expr: ExprId) -> String {
            self.render_calls += 1;
            "f".to_string()
        }
    }

    #[test]
    fn build_factorized_zero_product_execution_with_optional_items_runtime_includes_didactic_when_enabled(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_minus_one = ctx.add(Expr::Sub(x, one));
        let factored_expr = ctx.add(Expr::Mul(x, x_minus_one));
        let zero = ctx.num(0);
        let factors = vec![x, x_minus_one];
        let mut runtime = TestFactorizedZeroProductPlanRuntime::default();

        let execution = build_factorized_zero_product_execution_with_optional_items_runtime(
            &ctx,
            factored_expr,
            &factors,
            "x",
            zero,
            true,
            &mut runtime,
        );

        assert_eq!(execution.entry.description, "Factorized equation: f = 0");
        assert_eq!(execution.factors.items.len(), 2);
        assert!(runtime.render_calls > 0);
    }

    #[test]
    fn build_factorized_zero_product_execution_with_optional_items_runtime_skips_render_when_disabled(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_minus_one = ctx.add(Expr::Sub(x, one));
        let factored_expr = ctx.add(Expr::Mul(x, x_minus_one));
        let zero = ctx.num(0);
        let factors = vec![x, x_minus_one];
        let mut runtime = TestFactorizedZeroProductPlanRuntime::default();

        let execution = build_factorized_zero_product_execution_with_optional_items_runtime(
            &ctx,
            factored_expr,
            &factors,
            "x",
            zero,
            false,
            &mut runtime,
        );

        assert!(execution.entry.description.is_empty());
        assert!(execution.factors.items.is_empty());
        assert_eq!(execution.factors.equations.len(), 2);
        assert_eq!(runtime.render_calls, 0);
    }

    #[test]
    fn collect_zero_product_factor_execution_items_aligns_steps_with_equations() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_minus_one = ctx.add(Expr::Sub(x, one));
        let factors = vec![x, x_minus_one];
        let zero = ctx.num(0);

        let execution = build_zero_product_factor_execution_with(&ctx, &factors, "x", zero, |_| {
            "f".to_string()
        });
        let items = collect_zero_product_factor_execution_items(&execution);

        assert_eq!(items.len(), 2);
        assert_eq!(items[0].equation, execution.equations[0]);
        assert_eq!(items[1].equation, execution.equations[1]);
        assert_eq!(items[0].description, "Solve factor: f = 0");
        assert_eq!(items[1].description, "Solve factor: f = 0");
    }

    #[test]
    fn solve_zero_product_factor_execution_with_items_aligns_items_in_order() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_minus_one = ctx.add(Expr::Sub(x, one));
        let factors = vec![x, x_minus_one];
        let zero = ctx.num(0);

        let execution = build_zero_product_factor_execution_with(&ctx, &factors, "x", zero, |_| {
            "f".to_string()
        });
        let mut seen = Vec::new();
        let solved =
            solve_zero_product_factor_execution_with_items(&execution, |item, equation| {
                seen.push(item.expect("expected aligned item").description);
                Ok::<_, ()>(equation.clone())
            })
            .expect("solve should succeed");

        assert_eq!(
            seen,
            vec![
                "Solve factor: f = 0".to_string(),
                "Solve factor: f = 0".to_string()
            ]
        );
        assert_eq!(solved.len(), 2);
        assert_eq!(solved[0], execution.equations[0]);
        assert_eq!(solved[1], execution.equations[1]);
    }

    #[test]
    fn solve_zero_product_factor_execution_with_items_passes_none_when_items_missing() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let execution = ZeroProductFactorExecutionPlan {
            equations: vec![Equation {
                lhs: x,
                rhs: zero,
                op: RelOp::Eq,
            }],
            items: vec![],
        };

        let solved =
            solve_zero_product_factor_execution_with_items(&execution, |item, equation| {
                assert!(item.is_none());
                Ok::<_, ()>(equation.clone())
            })
            .expect("solve should succeed");

        assert_eq!(solved.len(), 1);
        assert_eq!(solved[0], execution.equations[0]);
    }

    #[test]
    fn collect_zero_product_factor_item_didactic_steps_returns_present_step_only() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let item = ZeroProductFactorExecutionItem {
            equation: Equation {
                lhs: x,
                rhs: zero,
                op: RelOp::Eq,
            },
            description: "Solve factor: f = 0".to_string(),
        };
        let didactic = collect_zero_product_factor_item_didactic_steps(&item);
        assert_eq!(didactic.len(), 1);
        assert_eq!(didactic[0].description, "Solve factor: f = 0");
    }

    #[test]
    fn collect_zero_product_factor_item_execution_items_returns_present_step_only() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let item = ZeroProductFactorExecutionItem {
            equation: Equation {
                lhs: x,
                rhs: zero,
                op: RelOp::Eq,
            },
            description: "Solve factor: f = 0".to_string(),
        };
        let items = collect_zero_product_factor_item_execution_items(&item);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].equation, item.equation);
        assert_eq!(items[0].description, "Solve factor: f = 0".to_string());
    }

    #[test]
    fn collect_factorized_zero_product_entry_didactic_steps_returns_single_entry() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_minus_one = ctx.add(Expr::Sub(x, one));
        let factored_expr = ctx.add(Expr::Mul(x, x_minus_one));
        let zero = ctx.num(0);
        let execution = build_factorized_zero_product_execution_with(
            &ctx,
            factored_expr,
            &[x, x_minus_one],
            "x",
            zero,
            |_| "f".to_string(),
        );
        let didactic = collect_factorized_zero_product_entry_didactic_steps(&execution);
        assert_eq!(didactic.len(), 1);
        assert_eq!(didactic[0], execution.entry);
    }

    #[test]
    fn collect_factorized_zero_product_entry_execution_items_returns_single_item() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_minus_one = ctx.add(Expr::Sub(x, one));
        let factored_expr = ctx.add(Expr::Mul(x, x_minus_one));
        let zero = ctx.num(0);
        let execution = build_factorized_zero_product_execution_with(
            &ctx,
            factored_expr,
            &[x, x_minus_one],
            "x",
            zero,
            |_| "f".to_string(),
        );

        let items = collect_factorized_zero_product_entry_execution_items(&execution);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].equation, execution.entry.equation_after);
        assert_eq!(items[0].description, execution.entry.description);
    }

    #[test]
    fn first_factorized_zero_product_entry_execution_item_returns_single_item() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_minus_one = ctx.add(Expr::Sub(x, one));
        let factored_expr = ctx.add(Expr::Mul(x, x_minus_one));
        let zero = ctx.num(0);
        let execution = build_factorized_zero_product_execution_with(
            &ctx,
            factored_expr,
            &[x, x_minus_one],
            "x",
            zero,
            |_| "f".to_string(),
        );

        let item = first_factorized_zero_product_entry_execution_item(&execution)
            .expect("expected one factorized-entry item");
        assert_eq!(item.equation, execution.entry.equation_after);
        assert_eq!(item.description, execution.entry.description);
    }

    #[test]
    fn solve_factorized_zero_product_execution_pipeline_with_items_prepends_entry_and_factor_items()
    {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_minus_one = ctx.add(Expr::Sub(x, one));
        let factored_expr = ctx.add(Expr::Mul(x, x_minus_one));
        let zero = ctx.num(0);
        let execution = build_factorized_zero_product_execution_with(
            &ctx,
            factored_expr,
            &[x, x_minus_one],
            "x",
            zero,
            |_| "f".to_string(),
        );

        let solved = solve_factorized_zero_product_execution_pipeline_with_items(
            &execution,
            true,
            |factor_equation| Ok::<_, ()>(factor_equation.lhs),
            |item| item.description,
            |item| item.description,
        )
        .expect("pipeline should solve");

        assert_eq!(solved.solved_factors.len(), 2);
        assert_eq!(solved.steps.len(), 3);
        assert_eq!(solved.steps[0], "Factorized equation: f = 0");
        assert_eq!(solved.steps[1], "Solve factor: f = 0");
        assert_eq!(solved.steps[2], "Solve factor: f = 0");
    }

    #[test]
    fn solve_factorized_zero_product_execution_pipeline_with_items_omits_items_when_disabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_minus_one = ctx.add(Expr::Sub(x, one));
        let factored_expr = ctx.add(Expr::Mul(x, x_minus_one));
        let zero = ctx.num(0);
        let execution = build_factorized_zero_product_execution_with(
            &ctx,
            factored_expr,
            &[x, x_minus_one],
            "x",
            zero,
            |_| "f".to_string(),
        );

        let solved = solve_factorized_zero_product_execution_pipeline_with_items(
            &execution,
            false,
            |factor_equation| Ok::<_, ()>(factor_equation.lhs),
            |_item| 7u8,
            |_item| 9u8,
        )
        .expect("pipeline should solve");

        assert_eq!(solved.solved_factors.len(), 2);
        assert!(solved.steps.is_empty());
    }

    #[derive(Default)]
    struct TestFactorizedZeroProductExecutionRuntime {
        solved_lhs: Vec<ExprId>,
        entry_map_calls: usize,
        factor_map_calls: usize,
    }

    impl FactorizedZeroProductExecutionRuntime<(), ExprId, String>
        for TestFactorizedZeroProductExecutionRuntime
    {
        fn solve_factor(&mut self, equation: &Equation) -> Result<ExprId, ()> {
            self.solved_lhs.push(equation.lhs);
            Ok(equation.lhs)
        }

        fn map_entry_item_to_step(&mut self, item: QuadraticExecutionItem) -> String {
            self.entry_map_calls += 1;
            item.description
        }

        fn map_factor_item_to_step(&mut self, item: ZeroProductFactorExecutionItem) -> String {
            self.factor_map_calls += 1;
            item.description
        }
    }

    #[test]
    fn solve_factorized_zero_product_execution_pipeline_with_items_runtime_forwards_calls_and_steps(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_minus_one = ctx.add(Expr::Sub(x, one));
        let factored_expr = ctx.add(Expr::Mul(x, x_minus_one));
        let zero = ctx.num(0);
        let execution = build_factorized_zero_product_execution_with(
            &ctx,
            factored_expr,
            &[x, x_minus_one],
            "x",
            zero,
            |_| "f".to_string(),
        );
        let mut runtime = TestFactorizedZeroProductExecutionRuntime::default();

        let solved = solve_factorized_zero_product_execution_pipeline_with_items_runtime(
            &execution,
            true,
            &mut runtime,
        )
        .expect("runtime pipeline should solve");

        assert_eq!(solved.solved_factors, vec![x, x_minus_one]);
        assert_eq!(runtime.solved_lhs, vec![x, x_minus_one]);
        assert_eq!(runtime.entry_map_calls, 1);
        assert_eq!(runtime.factor_map_calls, 2);
        assert_eq!(
            solved.steps,
            vec![
                "Factorized equation: f = 0".to_string(),
                "Solve factor: f = 0".to_string(),
                "Solve factor: f = 0".to_string(),
            ]
        );
    }

    #[test]
    fn solve_factorized_zero_product_execution_pipeline_with_items_runtime_omits_item_mapping_when_disabled(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_minus_one = ctx.add(Expr::Sub(x, one));
        let factored_expr = ctx.add(Expr::Mul(x, x_minus_one));
        let zero = ctx.num(0);
        let execution = build_factorized_zero_product_execution_with(
            &ctx,
            factored_expr,
            &[x, x_minus_one],
            "x",
            zero,
            |_| "f".to_string(),
        );
        let mut runtime = TestFactorizedZeroProductExecutionRuntime::default();

        let solved = solve_factorized_zero_product_execution_pipeline_with_items_runtime(
            &execution,
            false,
            &mut runtime,
        )
        .expect("runtime pipeline should solve");

        assert_eq!(solved.solved_factors, vec![x, x_minus_one]);
        assert_eq!(runtime.solved_lhs, vec![x, x_minus_one]);
        assert_eq!(runtime.entry_map_calls, 0);
        assert_eq!(runtime.factor_map_calls, 0);
        assert!(solved.steps.is_empty());
    }

    #[test]
    fn solve_factorized_zero_product_pipeline_with_optional_items_includes_items_when_enabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_minus_one = ctx.add(Expr::Sub(x, one));
        let factored_expr = ctx.add(Expr::Mul(x, x_minus_one));
        let zero = ctx.num(0);

        let solved = solve_factorized_zero_product_pipeline_with_optional_items(
            &ctx,
            factored_expr,
            &[x, x_minus_one],
            "x",
            zero,
            true,
            |_| "f".to_string(),
            |factor_equation| Ok::<_, ()>(factor_equation.lhs),
            |item| item.description,
            |item| item.description,
        )
        .expect("pipeline should solve");

        assert_eq!(solved.solved_factors.len(), 2);
        assert_eq!(solved.steps.len(), 3);
        assert_eq!(solved.steps[0], "Factorized equation: f = 0");
        assert_eq!(solved.steps[1], "Solve factor: f = 0");
        assert_eq!(solved.steps[2], "Solve factor: f = 0");
    }

    #[test]
    fn solve_factorized_zero_product_pipeline_with_optional_items_skips_render_when_disabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_minus_one = ctx.add(Expr::Sub(x, one));
        let factored_expr = ctx.add(Expr::Mul(x, x_minus_one));
        let zero = ctx.num(0);

        let solved = solve_factorized_zero_product_pipeline_with_optional_items(
            &ctx,
            factored_expr,
            &[x, x_minus_one],
            "x",
            zero,
            false,
            |_id| -> String { panic!("renderer must not run when items are disabled") },
            |factor_equation| Ok::<_, ()>(factor_equation.lhs),
            |_item| -> u8 { panic!("entry mapper must not run when items are disabled") },
            |_item| -> u8 { panic!("factor mapper must not run when items are disabled") },
        )
        .expect("pipeline should solve");

        assert_eq!(solved.solved_factors.len(), 2);
        assert!(solved.steps.is_empty());
    }

    #[test]
    fn solve_factorized_zero_product_strategy_pipeline_with_optional_items_flattens_steps() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_minus_one = ctx.add(Expr::Sub(x, one));
        let factored_expr = ctx.add(Expr::Mul(x, x_minus_one));
        let zero = ctx.num(0);

        let solved = solve_factorized_zero_product_strategy_pipeline_with_optional_items(
            &ctx,
            factored_expr,
            &[x, x_minus_one],
            "x",
            zero,
            true,
            factored_expr,
            |_| "f".to_string(),
            |factor_equation| {
                Ok::<_, ()>((
                    SolutionSet::Discrete(vec![factor_equation.lhs]),
                    vec![format!("substep {}", factor_equation.lhs)],
                ))
            },
            |item| item.description,
            |item| item.description,
        )
        .expect("strategy pipeline should solve");

        assert!(matches!(solved.solution_set, SolutionSet::Discrete(_)));
        assert_eq!(solved.steps.len(), 5);
        assert_eq!(solved.steps[0], "Factorized equation: f = 0");
        assert_eq!(solved.steps[1], "Solve factor: f = 0");
        assert_eq!(solved.steps[2], "Solve factor: f = 0");
        assert!(solved.steps[3].starts_with("substep "));
        assert!(solved.steps[4].starts_with("substep "));
    }

    #[test]
    fn solve_factorized_zero_product_strategy_pipeline_with_optional_items_uses_residual_for_non_discrete(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let factored_expr = x;
        let simplified_residual = ctx.var("simplified_residual");
        let zero = ctx.num(0);

        let solved = solve_factorized_zero_product_strategy_pipeline_with_optional_items(
            &ctx,
            factored_expr,
            &[x],
            "x",
            zero,
            false,
            simplified_residual,
            |_id| -> String { panic!("renderer must not run when items are disabled") },
            |_factor_equation| Ok::<_, ()>((SolutionSet::Conditional(vec![]), vec![])),
            |_item| -> u8 { panic!("entry mapper must not run when items are disabled") },
            |_item| -> u8 { panic!("factor mapper must not run when items are disabled") },
        )
        .expect("strategy pipeline should solve");

        assert!(
            matches!(solved.solution_set, SolutionSet::Residual(id) if id == simplified_residual)
        );
        assert!(solved.steps.is_empty());
    }

    #[test]
    fn solve_factorized_zero_product_strategy_pipeline_with_optional_items_skips_residual_for_all_reals(
    ) {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let factored_expr = x;
        let residual = ctx.var("residual");
        let zero = ctx.num(0);

        let solved = solve_factorized_zero_product_strategy_pipeline_with_optional_items(
            &ctx,
            factored_expr,
            &[x],
            "x",
            zero,
            false,
            residual,
            |_id| -> String { panic!("renderer must not run when items are disabled") },
            |_factor_equation| Ok::<_, ()>((SolutionSet::AllReals, vec![])),
            |_item| -> u8 { panic!("entry mapper must not run when items are disabled") },
            |_item| -> u8 { panic!("factor mapper must not run when items are disabled") },
        )
        .expect("strategy pipeline should solve");

        assert!(matches!(solved.solution_set, SolutionSet::AllReals));
        assert!(solved.steps.is_empty());
    }

    #[test]
    fn collect_quadratic_main_didactic_steps_returns_single_step() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let step = build_quadratic_main_step(Equation {
            lhs: x,
            rhs: ctx.num(0),
            op: RelOp::Eq,
        });
        let didactic = collect_quadratic_main_didactic_steps(&step);
        assert_eq!(didactic.len(), 1);
        assert_eq!(didactic[0], step);
    }

    #[test]
    fn collect_quadratic_main_execution_items_returns_single_item() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let step = build_quadratic_main_step(Equation {
            lhs: x,
            rhs: ctx.num(0),
            op: RelOp::Eq,
        });

        let items = collect_quadratic_main_execution_items(&step);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].equation, step.equation_after);
        assert_eq!(items[0].description, step.description);
    }

    #[test]
    fn build_quadratic_main_execution_items_returns_single_item() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let equation = Equation {
            lhs: x,
            rhs: ctx.num(0),
            op: RelOp::Eq,
        };

        let items = build_quadratic_main_execution_items(equation.clone());
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].equation, equation);
        assert_eq!(
            items[0].description,
            QUADRATIC_FORMULA_MAIN_STEP_DESCRIPTION
        );
    }

    #[test]
    fn aggregate_zero_product_factor_solution_sets_merges_discrete_and_empty() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let sets = vec![SolutionSet::Empty, SolutionSet::Discrete(vec![x, one, x])];

        let out = aggregate_zero_product_factor_solution_sets(&ctx, sets);
        match out {
            ZeroProductFactorSolutionAggregate::Discrete(solutions) => {
                assert_eq!(solutions.len(), 2);
            }
            other => panic!("expected discrete aggregate, got {:?}", other),
        }
    }

    #[test]
    fn aggregate_zero_product_factor_solution_sets_short_circuits_to_all_reals() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sets = vec![SolutionSet::Discrete(vec![x]), SolutionSet::AllReals];

        let out = aggregate_zero_product_factor_solution_sets(&ctx, sets);
        assert!(matches!(out, ZeroProductFactorSolutionAggregate::AllReals));
    }

    #[test]
    fn aggregate_zero_product_factor_solution_sets_flags_non_discrete_inputs() {
        let mut ctx = Context::new();
        let residual = ctx.var("r");
        let sets = vec![SolutionSet::Residual(residual)];

        let out = aggregate_zero_product_factor_solution_sets(&ctx, sets);
        assert!(matches!(
            out,
            ZeroProductFactorSolutionAggregate::NonDiscrete
        ));
    }

    #[test]
    fn finalize_zero_product_factor_solution_set_maps_discrete() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let out = finalize_zero_product_factor_solution_set(
            ZeroProductFactorSolutionAggregate::Discrete(vec![x]),
            ctx.num(0),
        );
        assert!(matches!(out, SolutionSet::Discrete(solutions) if solutions == vec![x]));
    }

    #[test]
    fn finalize_zero_product_factor_solution_set_maps_all_reals() {
        let mut ctx = Context::new();
        let out = finalize_zero_product_factor_solution_set(
            ZeroProductFactorSolutionAggregate::AllReals,
            ctx.num(0),
        );
        assert!(matches!(out, SolutionSet::AllReals));
    }

    #[test]
    fn finalize_zero_product_factor_solution_set_maps_non_discrete_to_residual() {
        let mut ctx = Context::new();
        let residual = ctx.var("r");
        let out = finalize_zero_product_factor_solution_set(
            ZeroProductFactorSolutionAggregate::NonDiscrete,
            residual,
        );
        assert!(matches!(out, SolutionSet::Residual(id) if id == residual));
    }
}
