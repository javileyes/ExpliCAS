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
pub fn combine_fractions_deterministic(
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

/// Didactic payload for one reciprocal-solve step.
#[derive(Debug, Clone, PartialEq)]
pub struct ReciprocalDidacticStep {
    pub description: String,
    pub equation_after: Equation,
}

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

/// Runtime contract for reciprocal solve orchestration.
///
/// This lets `cas_solver_core` host the reciprocal algorithm while callers
/// inject their own simplification/proof engines.
pub trait ReciprocalSolveRuntime {
    /// Mutable access to the expression context.
    fn context(&mut self) -> &mut Context;
    /// Simplify one expression and return the rewritten root.
    fn simplify_expr(&mut self, expr: ExprId) -> ExprId;
    /// Prove whether an expression is non-zero.
    fn prove_nonzero_status(&mut self, expr: ExprId) -> NonZeroStatus;
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

/// Collect reciprocal didactic steps in execution order.
pub fn collect_reciprocal_didactic_steps(
    execution: &ReciprocalSolveExecution,
) -> Vec<ReciprocalDidacticStep> {
    execution
        .items
        .iter()
        .cloned()
        .map(|item| ReciprocalDidacticStep {
            description: item.description,
            equation_after: item.equation,
        })
        .collect()
}

/// Collect reciprocal execution items in execution order.
pub fn collect_reciprocal_execution_items(
    execution: &ReciprocalSolveExecution,
) -> Vec<ReciprocalExecutionItem> {
    execution.items.clone()
}

/// Dispatch reciprocal execution items plus solved solution set to a caller
/// callback, preserving execution payload in the returned solved wrapper.
pub fn solve_reciprocal_execution_with_items<T, FSolve>(
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
pub fn solve_reciprocal_execution_pipeline_with_items<S, FStep>(
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

/// Runtime adapter for reciprocal execution didactic mapping.
pub trait ReciprocalExecutionRuntime<S> {
    fn map_item_to_step(&mut self, item: ReciprocalExecutionItem) -> S;
}

/// Runtime-based reciprocal execution pipeline with optional item mapping.
pub fn solve_reciprocal_execution_pipeline_with_items_runtime<S, R>(
    execution: ReciprocalSolveExecution,
    include_items: bool,
    runtime: &mut R,
) -> ReciprocalSolvedExecution<(SolutionSet, Vec<S>)>
where
    R: ReciprocalExecutionRuntime<S>,
{
    solve_reciprocal_execution_with_items(execution, |items, solutions| {
        let mut steps = Vec::new();
        if include_items {
            for item in items {
                steps.push(runtime.map_item_to_step(item));
            }
        }
        (solutions, steps)
    })
}

/// Derive reciprocal-solve kernel if equation matches `1/var = rhs` and
/// the RHS is independent of `var`.
pub fn derive_reciprocal_solve_kernel(
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
pub fn build_reciprocal_solve_plan(
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

/// Build didactic payload for the reciprocal combine step.
pub fn build_reciprocal_combine_step(equation_after: Equation) -> ReciprocalDidacticStep {
    ReciprocalDidacticStep {
        description: RECIPROCAL_COMBINE_STEP_DESCRIPTION.to_string(),
        equation_after,
    }
}

/// Build didactic payload for the reciprocal inversion step.
pub fn build_reciprocal_invert_step(equation_after: Equation) -> ReciprocalDidacticStep {
    ReciprocalDidacticStep {
        description: RECIPROCAL_INVERT_STEP_DESCRIPTION.to_string(),
        equation_after,
    }
}

/// Build solution set for reciprocal equations `1/x = N/D` where
/// candidate solution is `x = D/N` and the domain requires `N != 0`.
pub fn build_reciprocal_solution_set(
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
pub fn build_reciprocal_execution(
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
pub fn build_reciprocal_execution_from_kernel_prepared(
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

/// Build reciprocal execution directly from a normalized kernel, while callers
/// inject simplification and proof strategies.
pub fn build_reciprocal_execution_from_kernel_with<FS, FP>(
    ctx: &mut Context,
    var: &str,
    kernel: ReciprocalSolveKernel,
    mut simplify_expr: FS,
    mut prove_nonzero_status: FP,
) -> ReciprocalSolveExecution
where
    FS: FnMut(ExprId) -> ExprId,
    FP: FnMut(&Context, ExprId) -> NonZeroStatus,
{
    let raw_plan = build_reciprocal_solve_plan(ctx, var, kernel.numerator, kernel.denominator);
    let combined_rhs_display = simplify_expr(raw_plan.combined_rhs);
    let solution_rhs_display = simplify_expr(raw_plan.solution_rhs);
    let guard_numerator = simplify_expr(kernel.numerator);
    let numerator_status = prove_nonzero_status(ctx, guard_numerator);

    build_reciprocal_execution(
        ctx,
        var,
        kernel.numerator,
        kernel.denominator,
        combined_rhs_display,
        solution_rhs_display,
        guard_numerator,
        numerator_status,
    )
}

/// High-level reciprocal solve execution using an injected runtime.
///
/// Returns `None` when equation shape is not reciprocal-isolable.
pub fn execute_reciprocal_solve_with_runtime<R>(
    runtime: &mut R,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
) -> Option<ReciprocalSolveExecution>
where
    R: ReciprocalSolveRuntime,
{
    let kernel = {
        let ctx = runtime.context();
        derive_reciprocal_solve_kernel(ctx, lhs, rhs, var)?
    };

    let raw_plan = {
        let ctx = runtime.context();
        build_reciprocal_solve_plan(ctx, var, kernel.numerator, kernel.denominator)
    };

    let combined_rhs_display = runtime.simplify_expr(raw_plan.combined_rhs);
    let solution_rhs_display = runtime.simplify_expr(raw_plan.solution_rhs);
    let guard_numerator = runtime.simplify_expr(kernel.numerator);
    let numerator_status = runtime.prove_nonzero_status(guard_numerator);

    Some({
        let ctx = runtime.context();
        build_reciprocal_execution_from_kernel_prepared(
            ctx,
            var,
            kernel,
            ReciprocalPreparedExecution {
                combined_rhs_display,
                solution_rhs_display,
                guard_numerator,
                numerator_status,
            },
        )
    })
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
    fn reciprocal_step_builders_use_standard_messages() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let eq = Equation {
            lhs: x,
            rhs: y,
            op: RelOp::Eq,
        };
        let combine = build_reciprocal_combine_step(eq.clone());
        assert_eq!(
            combine.description,
            "Combine fractions on RHS (common denominator)"
        );
        assert_eq!(combine.equation_after, eq);

        let invert = build_reciprocal_invert_step(eq.clone());
        assert_eq!(invert.description, "Take reciprocal");
        assert_eq!(invert.equation_after, eq);
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
    fn build_reciprocal_execution_from_kernel_with_uses_callbacks() {
        let mut ctx = Context::new();
        let numerator = ctx.var("n");
        let denominator = ctx.var("d");
        let kernel = ReciprocalSolveKernel {
            numerator,
            denominator,
        };
        let one = ctx.num(1);
        let mut simplify_calls = 0usize;
        let mut prove_calls = 0usize;

        let execution = build_reciprocal_execution_from_kernel_with(
            &mut ctx,
            "x",
            kernel,
            |_| {
                simplify_calls += 1;
                one
            },
            |_, _| {
                prove_calls += 1;
                NonZeroStatus::Unknown
            },
        );

        assert_eq!(simplify_calls, 3);
        assert_eq!(prove_calls, 1);
        assert_eq!(execution.items.len(), 2);
        assert_eq!(execution.items[0].equation.rhs, one);
        assert_eq!(execution.items[1].equation.rhs, one);
        assert!(matches!(execution.solutions, SolutionSet::Conditional(_)));
    }

    struct MockReciprocalRuntime {
        context: Context,
        simplify_value: ExprId,
        simplify_calls: usize,
        prove_calls: usize,
        prove_result: NonZeroStatus,
    }

    impl ReciprocalSolveRuntime for MockReciprocalRuntime {
        fn context(&mut self) -> &mut Context {
            &mut self.context
        }

        fn simplify_expr(&mut self, _expr: ExprId) -> ExprId {
            self.simplify_calls += 1;
            self.simplify_value
        }

        fn prove_nonzero_status(&mut self, _expr: ExprId) -> NonZeroStatus {
            self.prove_calls += 1;
            self.prove_result
        }
    }

    #[test]
    fn execute_reciprocal_solve_with_runtime_uses_runtime_hooks() {
        let mut context = Context::new();
        let x = context.var("x");
        let r1 = context.var("r1");
        let one = context.num(1);
        let lhs = context.add(Expr::Div(one, x));
        let rhs = context.add(Expr::Div(one, r1));
        let simplified = context.var("s");

        let mut runtime = MockReciprocalRuntime {
            context,
            simplify_value: simplified,
            simplify_calls: 0,
            prove_calls: 0,
            prove_result: NonZeroStatus::Unknown,
        };

        let execution = execute_reciprocal_solve_with_runtime(&mut runtime, lhs, rhs, "x")
            .expect("reciprocal equation should be solvable");
        assert_eq!(runtime.simplify_calls, 3);
        assert_eq!(runtime.prove_calls, 1);
        assert_eq!(execution.items.len(), 2);
        assert_eq!(execution.items[0].equation.rhs, simplified);
        assert_eq!(execution.items[1].equation.rhs, simplified);
        assert!(matches!(execution.solutions, SolutionSet::Conditional(_)));
    }

    #[test]
    fn execute_reciprocal_solve_with_runtime_rejects_non_reciprocal_shape() {
        let mut context = Context::new();
        let x = context.var("x");
        let rhs = context.var("r");
        let simplified = context.var("s");
        let mut runtime = MockReciprocalRuntime {
            context,
            simplify_value: simplified,
            simplify_calls: 0,
            prove_calls: 0,
            prove_result: NonZeroStatus::Unknown,
        };

        let execution = execute_reciprocal_solve_with_runtime(&mut runtime, x, rhs, "x");
        assert!(execution.is_none());
        assert_eq!(runtime.simplify_calls, 0);
        assert_eq!(runtime.prove_calls, 0);
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
    fn collect_reciprocal_didactic_steps_preserves_step_order() {
        let mut ctx = Context::new();
        let n = ctx.var("n");
        let d = ctx.var("d");
        let c = ctx.var("c");
        let s = ctx.var("s");
        let execution =
            build_reciprocal_execution(&mut ctx, "x", n, d, c, s, n, NonZeroStatus::Unknown);

        let didactic = collect_reciprocal_didactic_steps(&execution);
        assert_eq!(didactic.len(), 2);
        assert_eq!(
            didactic[0].description,
            "Combine fractions on RHS (common denominator)"
        );
        assert_eq!(didactic[1].description, "Take reciprocal");
        assert_eq!(didactic[0].equation_after, execution.items[0].equation);
        assert_eq!(didactic[1].equation_after, execution.items[1].equation);
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

        let solved = solve_reciprocal_execution_pipeline_with_items(execution, true, |item| {
            item.description
        });

        assert_eq!(solved.execution, expected);
        assert!(matches!(solved.solved.0, SolutionSet::Conditional(_)));
        assert_eq!(solved.solved.1.len(), expected.items.len());
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

    #[derive(Default)]
    struct TestReciprocalExecutionRuntime {
        map_calls: usize,
    }

    impl ReciprocalExecutionRuntime<String> for TestReciprocalExecutionRuntime {
        fn map_item_to_step(&mut self, item: ReciprocalExecutionItem) -> String {
            self.map_calls += 1;
            item.description
        }
    }

    #[test]
    fn solve_reciprocal_execution_pipeline_with_items_runtime_maps_steps_when_enabled() {
        let mut ctx = Context::new();
        let n = ctx.var("n");
        let d = ctx.var("d");
        let c = ctx.var("c");
        let s = ctx.var("s");
        let execution =
            build_reciprocal_execution(&mut ctx, "x", n, d, c, s, n, NonZeroStatus::Unknown);
        let expected = execution.clone();
        let mut runtime = TestReciprocalExecutionRuntime::default();

        let solved =
            solve_reciprocal_execution_pipeline_with_items_runtime(execution, true, &mut runtime);

        assert_eq!(solved.execution, expected);
        assert!(matches!(solved.solved.0, SolutionSet::Conditional(_)));
        assert_eq!(solved.solved.1.len(), expected.items.len());
        assert_eq!(runtime.map_calls, expected.items.len());
    }

    #[test]
    fn solve_reciprocal_execution_pipeline_with_items_runtime_omits_steps_when_disabled() {
        let mut ctx = Context::new();
        let n = ctx.var("n");
        let d = ctx.var("d");
        let c = ctx.var("c");
        let s = ctx.var("s");
        let execution =
            build_reciprocal_execution(&mut ctx, "x", n, d, c, s, n, NonZeroStatus::Unknown);
        let mut runtime = TestReciprocalExecutionRuntime::default();

        let solved =
            solve_reciprocal_execution_pipeline_with_items_runtime(execution, false, &mut runtime);
        assert!(solved.solved.1.is_empty());
        assert_eq!(runtime.map_calls, 0);
    }
}
