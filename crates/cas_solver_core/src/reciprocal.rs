use crate::isolation_utils::{contains_var, is_simple_reciprocal};
use crate::linear_solution::NonZeroStatus;
use cas_ast::{
    Case, ConditionPredicate, ConditionSet, Context, Equation, Expr, ExprId, RelOp, SolutionSet,
};
use cas_math::expr_nary::add_terms_no_sign;
use cas_math::expr_predicates::is_one_expr as is_one;

/// Represents a fraction as (numerator, denominator).
#[derive(Debug, Clone)]
struct Fraction {
    num: ExprId,
    den: ExprId,
}

/// Convert an expression to fraction form.
/// - `a/b` -> (a, b)
/// - `a` -> (a, 1)
fn expr_to_fraction(ctx: &mut Context, expr: ExprId) -> Fraction {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) => Fraction { num, den },
        _ => {
            let one = ctx.num(1);
            Fraction {
                num: expr,
                den: one,
            }
        }
    }
}

/// Build scale factor for a fraction: product of all OTHER denominators.
fn build_scale_factor(ctx: &mut Context, fractions: &[Fraction], my_den: ExprId) -> ExprId {
    let other_dens: Vec<ExprId> = fractions
        .iter()
        .filter(|f| f.den != my_den)
        .map(|f| f.den)
        .collect();

    if other_dens.is_empty() {
        ctx.num(1)
    } else if other_dens.len() == 1 {
        other_dens[0]
    } else {
        let mut product = other_dens[0];
        for &d in &other_dens[1..] {
            product = ctx.add(Expr::Mul(product, d));
        }
        product
    }
}

/// Combine multiple fractions into a single fraction (numerator, denominator).
///
/// Uses common denominator `D = ∏ den_i`.
/// Numerator `N = Σ (num_i × (D/den_i))`.
///
/// Returns `(N, D)` after light structural normalization.
pub(crate) fn combine_fractions_deterministic(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let terms = add_terms_no_sign(ctx, expr);
    if terms.is_empty() {
        return None;
    }

    let fractions: Vec<Fraction> = terms.iter().map(|&t| expr_to_fraction(ctx, t)).collect();

    let common_denom = if fractions.len() == 1 {
        fractions[0].den
    } else {
        let mut denom = fractions[0].den;
        for frac in &fractions[1..] {
            denom = ctx.add(Expr::Mul(denom, frac.den));
        }
        denom
    };

    let mut scaled_nums: Vec<ExprId> = Vec::new();
    for frac in &fractions {
        let scale_factor = build_scale_factor(ctx, &fractions, frac.den);
        let scaled_num = if is_one(ctx, scale_factor) {
            frac.num
        } else {
            ctx.add(Expr::Mul(frac.num, scale_factor))
        };
        scaled_nums.push(scaled_num);
    }

    let numerator = if scaled_nums.len() == 1 {
        scaled_nums[0]
    } else {
        let mut sum = scaled_nums[0];
        for &term in &scaled_nums[1..] {
            sum = ctx.add(Expr::Add(sum, term));
        }
        sum
    };

    Some((numerator, common_denom))
}

/// Normalized data for reciprocal solve `1/var = rhs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReciprocalSolveKernel {
    pub numerator: ExprId,
    pub denominator: ExprId,
}

/// Human-readable step description for combining RHS fractions.
pub const RECIPROCAL_COMBINE_STEP_DESCRIPTION: &str =
    "Combine fractions on RHS (common denominator)";

/// Human-readable step description for reciprocal inversion.
pub const RECIPROCAL_INVERT_STEP_DESCRIPTION: &str = "Take reciprocal";

/// Planned equation targets for reciprocal solve `1/var = rhs`.
#[derive(Debug, Clone, PartialEq)]
pub struct ReciprocalSolvePlan {
    pub combine_equation: Equation,
    pub solve_equation: Equation,
    pub combined_rhs: ExprId,
    pub solution_rhs: ExprId,
}

/// Engine-facing reciprocal solve execution payload.
#[derive(Debug, Clone, PartialEq)]
pub struct ReciprocalSolveExecution {
    pub items: Vec<ReciprocalExecutionItem>,
    pub solutions: SolutionSet,
}

/// Solved payload for reciprocal execution dispatch.
#[derive(Debug, Clone, PartialEq)]
pub struct ReciprocalSolvedExecution<T> {
    pub execution: ReciprocalSolveExecution,
    pub solved: T,
}

/// Prepared data for building reciprocal execution from a normalized kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReciprocalPreparedExecution {
    pub combined_rhs_display: ExprId,
    pub solution_rhs_display: ExprId,
    pub guard_numerator: ExprId,
    pub numerator_status: NonZeroStatus,
}

/// One reciprocal execution item with aligned equation and didactic step.
#[derive(Debug, Clone, PartialEq)]
pub struct ReciprocalExecutionItem {
    pub equation: Equation,
    pub description: String,
}

impl ReciprocalExecutionItem {
    /// User-facing narration for this execution item.
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Collect reciprocal execution items in execution order.
pub(crate) fn collect_reciprocal_execution_items(
    execution: &ReciprocalSolveExecution,
) -> Vec<ReciprocalExecutionItem> {
    execution.items.clone()
}

/// Dispatch reciprocal execution items plus solved solution set to a caller
/// callback, preserving execution payload in the returned solved wrapper.
pub(crate) fn solve_reciprocal_execution_with_items<T, FSolve>(
    execution: ReciprocalSolveExecution,
    mut solve: FSolve,
) -> ReciprocalSolvedExecution<T>
where
    FSolve: FnMut(Vec<ReciprocalExecutionItem>, SolutionSet) -> T,
{
    let items = collect_reciprocal_execution_items(&execution);
    let solutions = execution.solutions.clone();
    let solved = solve(items, solutions);
    ReciprocalSolvedExecution { execution, solved }
}

/// Solve reciprocal execution while optionally mapping execution items
/// into caller-owned step payloads.
pub(crate) fn solve_reciprocal_execution_pipeline_with_items<S, FStep>(
    execution: ReciprocalSolveExecution,
    include_items: bool,
    mut map_item_to_step: FStep,
) -> ReciprocalSolvedExecution<(SolutionSet, Vec<S>)>
where
    FStep: FnMut(ReciprocalExecutionItem) -> S,
{
    solve_reciprocal_execution_with_items(execution, |items, solutions| {
        let mut steps = Vec::new();
        if include_items {
            for item in items {
                steps.push(map_item_to_step(item));
            }
        }
        (solutions, steps)
    })
}

/// Derive reciprocal-solve kernel if equation matches `1/var = rhs` and
/// the RHS is independent of `var`.
pub(crate) fn derive_reciprocal_solve_kernel(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
) -> Option<ReciprocalSolveKernel> {
    if !is_simple_reciprocal(ctx, lhs, var) {
        return None;
    }
    if contains_var(ctx, rhs, var) {
        return None;
    }
    let (numerator, denominator) = combine_fractions_deterministic(ctx, rhs)?;
    Some(ReciprocalSolveKernel {
        numerator,
        denominator,
    })
}

/// Build the didactic execution plan for reciprocal solve.
///
/// Given normalized `rhs = numerator / denominator`, this returns:
/// 1. `1/var = numerator/denominator`
/// 2. `var = denominator/numerator`
pub(crate) fn build_reciprocal_solve_plan(
    ctx: &mut Context,
    var: &str,
    numerator: ExprId,
    denominator: ExprId,
) -> ReciprocalSolvePlan {
    let var_id = ctx.var(var);
    let one = ctx.num(1);
    let reciprocal_lhs = ctx.add(Expr::Div(one, var_id));
    let combined_rhs = ctx.add(Expr::Div(numerator, denominator));
    let solution_rhs = ctx.add(Expr::Div(denominator, numerator));

    let combine_equation = Equation {
        lhs: reciprocal_lhs,
        rhs: combined_rhs,
        op: RelOp::Eq,
    };
    let solve_equation = Equation {
        lhs: var_id,
        rhs: solution_rhs,
        op: RelOp::Eq,
    };

    ReciprocalSolvePlan {
        combine_equation,
        solve_equation,
        combined_rhs,
        solution_rhs,
    }
}

/// Build solution set for reciprocal equations `1/x = N/D` where
/// candidate solution is `x = D/N` and the domain requires `N != 0`.
pub(crate) fn build_reciprocal_solution_set(
    numerator: ExprId,
    solution: ExprId,
    numerator_status: NonZeroStatus,
) -> SolutionSet {
    if numerator_status == NonZeroStatus::NonZero {
        return SolutionSet::Discrete(vec![solution]);
    }

    let guard = ConditionSet::single(ConditionPredicate::NonZero(numerator));
    let case = Case::new(guard, SolutionSet::Discrete(vec![solution]));
    SolutionSet::Conditional(vec![case])
}

/// Build full reciprocal solve execution payload from prepared display/proof data.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_reciprocal_execution(
    ctx: &mut Context,
    var: &str,
    numerator: ExprId,
    denominator: ExprId,
    combined_rhs_display: ExprId,
    solution_rhs_display: ExprId,
    guard_numerator: ExprId,
    numerator_status: NonZeroStatus,
) -> ReciprocalSolveExecution {
    let plan = build_reciprocal_solve_plan(ctx, var, numerator, denominator);

    let mut combine_equation = plan.combine_equation;
    combine_equation.rhs = combined_rhs_display;

    let mut solve_equation = plan.solve_equation;
    solve_equation.rhs = solution_rhs_display;
    let items = vec![
        ReciprocalExecutionItem {
            equation: combine_equation,
            description: RECIPROCAL_COMBINE_STEP_DESCRIPTION.to_string(),
        },
        ReciprocalExecutionItem {
            equation: solve_equation,
            description: RECIPROCAL_INVERT_STEP_DESCRIPTION.to_string(),
        },
    ];

    let solutions =
        build_reciprocal_solution_set(guard_numerator, solution_rhs_display, numerator_status);

    ReciprocalSolveExecution { items, solutions }
}

/// Build reciprocal execution from a normalized kernel and prepared display/proof inputs.
pub(crate) fn build_reciprocal_execution_from_kernel_prepared(
    ctx: &mut Context,
    var: &str,
    kernel: ReciprocalSolveKernel,
    prepared: ReciprocalPreparedExecution,
) -> ReciprocalSolveExecution {
    build_reciprocal_execution(
        ctx,
        var,
        kernel.numerator,
        kernel.denominator,
        prepared.combined_rhs_display,
        prepared.solution_rhs_display,
        prepared.guard_numerator,
        prepared.numerator_status,
    )
}

/// Stateful variant of
/// [`execute_reciprocal_solve_pipeline_with_items_via_kernel_execution_pipeline`].
///
/// This form lets callers thread one mutable state object across all hooks
/// without interior mutability wrappers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_reciprocal_solve_pipeline_with_items_via_kernel_execution_pipeline_with_state<
    T,
    S,
    FDeriveKernel,
    FPlan,
    FSimplify,
    FProof,
    FBuildPrepared,
    FStep,
>(
    state: &mut T,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    include_items: bool,
    mut derive_kernel: FDeriveKernel,
    mut plan_from_kernel: FPlan,
    mut simplify_expr: FSimplify,
    mut prove_nonzero_status: FProof,
    mut build_execution_from_prepared: FBuildPrepared,
    map_item_to_step: FStep,
) -> Option<(SolutionSet, Vec<S>)>
where
    FDeriveKernel: FnMut(&mut T, ExprId, ExprId, &str) -> Option<ReciprocalSolveKernel>,
    FPlan: FnMut(&mut T, &str, ReciprocalSolveKernel) -> ReciprocalSolvePlan,
    FSimplify: FnMut(&mut T, ExprId) -> ExprId,
    FProof: FnMut(&mut T, ExprId) -> NonZeroStatus,
    FBuildPrepared: FnMut(
        &mut T,
        &str,
        ReciprocalSolveKernel,
        ReciprocalPreparedExecution,
    ) -> ReciprocalSolveExecution,
    FStep: FnMut(ReciprocalExecutionItem) -> S,
{
    let kernel = derive_kernel(state, lhs, rhs, var)?;
    let raw_plan = plan_from_kernel(state, var, kernel);
    let combined_rhs_display = simplify_expr(state, raw_plan.combined_rhs);
    let solution_rhs_display = simplify_expr(state, raw_plan.solution_rhs);
    let guard_numerator = simplify_expr(state, kernel.numerator);
    let numerator_status = prove_nonzero_status(state, guard_numerator);
    let prepared = ReciprocalPreparedExecution {
        combined_rhs_display,
        solution_rhs_display,
        guard_numerator,
        numerator_status,
    };
    let execution = build_execution_from_prepared(state, var, kernel, prepared);
    let solved_execution =
        solve_reciprocal_execution_pipeline_with_items(execution, include_items, map_item_to_step);
    Some(solved_execution.solved)
}

/// Stateful reciprocal solve pipeline with default kernel derivation, plan
/// construction, and execution construction.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_reciprocal_solve_pipeline_with_default_kernel_with_state<
    T,
    S,
    FContextMut,
    FSimplify,
    FProof,
    FStep,
>(
    state: &mut T,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    include_items: bool,
    context_mut: FContextMut,
    simplify_expr: FSimplify,
    prove_nonzero_status: FProof,
    map_item_to_step: FStep,
) -> Option<(SolutionSet, Vec<S>)>
where
    FContextMut: Fn(&mut T) -> &mut Context,
    FSimplify: FnMut(&mut T, ExprId) -> ExprId,
    FProof: FnMut(&mut T, ExprId) -> NonZeroStatus,
    FStep: FnMut(ReciprocalExecutionItem) -> S,
{
    execute_reciprocal_solve_pipeline_with_items_via_kernel_execution_pipeline_with_state(
        state,
        lhs,
        rhs,
        var,
        include_items,
        |state, left, right, var_name| {
            derive_reciprocal_solve_kernel(context_mut(state), left, right, var_name)
        },
        |state, var_name, reciprocal_kernel| {
            build_reciprocal_solve_plan(
                context_mut(state),
                var_name,
                reciprocal_kernel.numerator,
                reciprocal_kernel.denominator,
            )
        },
        simplify_expr,
        prove_nonzero_status,
        |state, var_name, reciprocal_kernel, prepared| {
            build_reciprocal_execution_from_kernel_prepared(
                context_mut(state),
                var_name,
                reciprocal_kernel,
                prepared,
            )
        },
        map_item_to_step,
    )
}

/// Stateful reciprocal solve pipeline with default kernel derivation
/// and a unified step-mapper callback.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_reciprocal_solve_pipeline_with_default_kernel_and_unified_step_mapper_with_state<
    T,
    S,
    FContextMut,
    FSimplify,
    FProof,
    FMapStep,
>(
    state: &mut T,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    include_items: bool,
    context_mut: FContextMut,
    simplify_expr: FSimplify,
    prove_nonzero_status: FProof,
    map_step: FMapStep,
) -> Option<(SolutionSet, Vec<S>)>
where
    FContextMut: Fn(&mut T) -> &mut Context,
    FSimplify: FnMut(&mut T, ExprId) -> ExprId,
    FProof: FnMut(&mut T, ExprId) -> NonZeroStatus,
    FMapStep: FnMut(String, Equation) -> S,
{
    let map_step = std::cell::RefCell::new(map_step);
    execute_reciprocal_solve_pipeline_with_default_kernel_with_state(
        state,
        lhs,
        rhs,
        var,
        include_items,
        context_mut,
        simplify_expr,
        prove_nonzero_status,
        |item| (map_step.borrow_mut())(item.description().to_string(), item.equation),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isolation_utils::{contains_var, is_simple_reciprocal};

    #[test]
    fn test_combine_fractions_simple() {
        let mut ctx = Context::new();
        let r1 = ctx.var("R1");
        let r2 = ctx.var("R2");
        let one = ctx.num(1);

        let frac1 = ctx.add(Expr::Div(one, r1));
        let one2 = ctx.num(1);
        let frac2 = ctx.add(Expr::Div(one2, r2));
        let sum = ctx.add(Expr::Add(frac1, frac2));

        let result = combine_fractions_deterministic(&mut ctx, sum);
        assert!(result.is_some());

        let (num, denom) = result.expect("must combine into a single fraction");
        assert!(contains_var(&ctx, num, "R1") || contains_var(&ctx, num, "R2"));
        assert!(contains_var(&ctx, denom, "R1"));
        assert!(contains_var(&ctx, denom, "R2"));
    }

    #[test]
    fn reciprocal_solution_nonzero_is_discrete() {
        let mut ctx = Context::new();
        let num = ctx.var("n");
        let sol = ctx.var("x0");
        let set = build_reciprocal_solution_set(num, sol, NonZeroStatus::NonZero);
        assert_eq!(set, SolutionSet::Discrete(vec![sol]));
    }

    #[test]
    fn reciprocal_solution_unknown_is_conditional() {
        let mut ctx = Context::new();
        let num = ctx.var("n");
        let sol = ctx.var("x0");
        let set = build_reciprocal_solution_set(num, sol, NonZeroStatus::Unknown);
        assert!(matches!(set, SolutionSet::Conditional(_)));
    }

    #[test]
    fn derive_reciprocal_solve_kernel_matches_simple_case() {
        let mut ctx = Context::new();
        let r = ctx.var("R");
        let r1 = ctx.var("R1");
        let r2 = ctx.var("R2");
        let one = ctx.num(1);
        let lhs = ctx.add(Expr::Div(one, r));
        let one2 = ctx.num(1);
        let frac1 = ctx.add(Expr::Div(one2, r1));
        let one3 = ctx.num(1);
        let frac2 = ctx.add(Expr::Div(one3, r2));
        let rhs = ctx.add(Expr::Add(frac1, frac2));

        let kernel = derive_reciprocal_solve_kernel(&mut ctx, lhs, rhs, "R")
            .expect("must derive reciprocal kernel");
        assert!(
            contains_var(&ctx, kernel.numerator, "R1")
                || contains_var(&ctx, kernel.numerator, "R2")
        );
        assert!(contains_var(&ctx, kernel.denominator, "R1"));
        assert!(contains_var(&ctx, kernel.denominator, "R2"));
    }

    #[test]
    fn derive_reciprocal_solve_kernel_rejects_rhs_with_var() {
        let mut ctx = Context::new();
        let r = ctx.var("R");
        let one = ctx.num(1);
        let lhs = ctx.add(Expr::Div(one, r));
        let rhs = r;
        assert!(is_simple_reciprocal(&ctx, lhs, "R"));
        assert!(derive_reciprocal_solve_kernel(&mut ctx, lhs, rhs, "R").is_none());
    }

    #[test]
    fn build_reciprocal_solve_plan_constructs_expected_shapes() {
        let mut ctx = Context::new();
        let numerator = ctx.var("n");
        let denominator = ctx.var("d");
        let plan = build_reciprocal_solve_plan(&mut ctx, "x", numerator, denominator);

        assert_eq!(plan.combine_equation.op, RelOp::Eq);
        assert_eq!(plan.solve_equation.op, RelOp::Eq);
        assert!(matches!(
            ctx.get(plan.combine_equation.lhs),
            Expr::Div(_, _)
        ));
        assert!(matches!(
            ctx.get(plan.combine_equation.rhs),
            Expr::Div(_, _)
        ));
        assert!(matches!(ctx.get(plan.solve_equation.rhs), Expr::Div(_, _)));
    }

    #[test]
    fn build_reciprocal_execution_assembles_steps_and_solution() {
        let mut ctx = Context::new();
        let num = ctx.var("n");
        let den = ctx.var("d");
        let combined_rhs = ctx.var("c");
        let solution_rhs = ctx.var("s");

        let execution = build_reciprocal_execution(
            &mut ctx,
            "x",
            num,
            den,
            combined_rhs,
            solution_rhs,
            num,
            NonZeroStatus::Unknown,
        );

        assert_eq!(execution.items.len(), 2);
        assert_eq!(
            execution.items[0].description,
            "Combine fractions on RHS (common denominator)"
        );
        assert_eq!(execution.items[0].equation.rhs, combined_rhs);
        assert_eq!(execution.items[1].description, "Take reciprocal");
        assert_eq!(execution.items[1].equation.rhs, solution_rhs);
        assert!(matches!(execution.solutions, SolutionSet::Conditional(_)));
    }

    #[test]
    fn execute_reciprocal_solve_pipeline_with_items_via_kernel_execution_pipeline_with_state_maps_steps_when_enabled(
    ) {
        let mut context = Context::new();
        let x = context.var("x");
        let r = context.var("r");
        let one = context.num(1);
        let lhs = context.add(Expr::Div(one, x));
        let rhs = context.add(Expr::Div(one, r));
        let context_cell = std::cell::RefCell::new(context);

        #[derive(Default)]
        struct HooksState {
            derive_calls: usize,
            simplify_calls: usize,
            proof_calls: usize,
            build_calls: usize,
            map_calls: usize,
        }

        let mut state = HooksState::default();
        let solved =
            execute_reciprocal_solve_pipeline_with_items_via_kernel_execution_pipeline_with_state(
                &mut state,
                lhs,
                rhs,
                "x",
                true,
                |hooks, inner_lhs, inner_rhs, inner_var| {
                    hooks.derive_calls += 1;
                    let mut context_ref = context_cell.borrow_mut();
                    derive_reciprocal_solve_kernel(
                        &mut context_ref,
                        inner_lhs,
                        inner_rhs,
                        inner_var,
                    )
                },
                |hooks, inner_var, kernel| {
                    hooks.simplify_calls += 100;
                    let mut context_ref = context_cell.borrow_mut();
                    build_reciprocal_solve_plan(
                        &mut context_ref,
                        inner_var,
                        kernel.numerator,
                        kernel.denominator,
                    )
                },
                |hooks, expr| {
                    hooks.simplify_calls += 1;
                    expr
                },
                |hooks, _expr| {
                    hooks.proof_calls += 1;
                    NonZeroStatus::Unknown
                },
                |hooks, inner_var, kernel, prepared| {
                    hooks.build_calls += 1;
                    let mut context_ref = context_cell.borrow_mut();
                    build_reciprocal_execution_from_kernel_prepared(
                        &mut context_ref,
                        inner_var,
                        kernel,
                        prepared,
                    )
                },
                |item| item.description,
            )
            .expect("reciprocal pipeline should execute");

        assert!(matches!(solved.0, SolutionSet::Conditional(_)));
        assert_eq!(solved.1.len(), 2);
        state.map_calls += solved.1.len();
        assert_eq!(state.derive_calls, 1);
        assert_eq!(state.simplify_calls, 103);
        assert_eq!(state.proof_calls, 1);
        assert_eq!(state.build_calls, 1);
        assert_eq!(state.map_calls, 2);
    }

    #[test]
    fn build_reciprocal_execution_from_kernel_prepared_uses_kernel_fields() {
        let mut ctx = Context::new();
        let numerator = ctx.var("n");
        let denominator = ctx.var("d");
        let display = ctx.var("display");
        let solution = ctx.var("solution");
        let execution = build_reciprocal_execution_from_kernel_prepared(
            &mut ctx,
            "x",
            ReciprocalSolveKernel {
                numerator,
                denominator,
            },
            ReciprocalPreparedExecution {
                combined_rhs_display: display,
                solution_rhs_display: solution,
                guard_numerator: numerator,
                numerator_status: NonZeroStatus::Unknown,
            },
        );

        assert_eq!(execution.items.len(), 2);
        assert_eq!(execution.items[0].equation.rhs, display);
        assert_eq!(execution.items[1].equation.rhs, solution);
        assert!(matches!(execution.solutions, SolutionSet::Conditional(_)));
    }

    #[test]
    fn collect_reciprocal_execution_items_aligns_equations_with_steps() {
        let mut ctx = Context::new();
        let n = ctx.var("n");
        let d = ctx.var("d");
        let c = ctx.var("c");
        let s = ctx.var("s");
        let execution =
            build_reciprocal_execution(&mut ctx, "x", n, d, c, s, n, NonZeroStatus::Unknown);

        let items = collect_reciprocal_execution_items(&execution);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], execution.items[0]);
        assert_eq!(items[1], execution.items[1]);
    }

    #[test]
    fn solve_reciprocal_execution_with_items_passes_items_and_solution_set() {
        let mut ctx = Context::new();
        let n = ctx.var("n");
        let d = ctx.var("d");
        let c = ctx.var("c");
        let s = ctx.var("s");
        let execution =
            build_reciprocal_execution(&mut ctx, "x", n, d, c, s, n, NonZeroStatus::Unknown);
        let expected = execution.clone();

        let solved = solve_reciprocal_execution_with_items(execution, |items, solutions| {
            assert_eq!(items.len(), 2);
            assert_eq!(items[0].equation, expected.items[0].equation);
            assert_eq!(items[1].equation, expected.items[1].equation);
            solutions
        });

        assert_eq!(solved.execution, expected);
        assert!(matches!(solved.solved, SolutionSet::Conditional(_)));
    }

    #[test]
    fn solve_reciprocal_execution_pipeline_with_items_maps_steps_when_enabled() {
        let mut ctx = Context::new();
        let n = ctx.var("n");
        let d = ctx.var("d");
        let c = ctx.var("c");
        let s = ctx.var("s");
        let execution =
            build_reciprocal_execution(&mut ctx, "x", n, d, c, s, n, NonZeroStatus::Unknown);
        let expected = execution.clone();
        let map_calls = std::cell::Cell::new(0usize);

        let solved = solve_reciprocal_execution_pipeline_with_items(execution, true, |item| {
            map_calls.set(map_calls.get() + 1);
            item.description
        });

        assert_eq!(solved.execution, expected);
        assert!(matches!(solved.solved.0, SolutionSet::Conditional(_)));
        assert_eq!(solved.solved.1.len(), expected.items.len());
        assert_eq!(map_calls.get(), expected.items.len());
    }

    #[test]
    fn solve_reciprocal_execution_pipeline_with_items_omits_steps_when_disabled() {
        let mut ctx = Context::new();
        let n = ctx.var("n");
        let d = ctx.var("d");
        let c = ctx.var("c");
        let s = ctx.var("s");
        let execution =
            build_reciprocal_execution(&mut ctx, "x", n, d, c, s, n, NonZeroStatus::Unknown);

        let solved = solve_reciprocal_execution_pipeline_with_items(execution, false, |_item| 1u8);
        assert!(solved.solved.1.is_empty());
    }
}
