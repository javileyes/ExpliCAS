use crate::isolation_utils::contains_var;
pub use crate::linear_didactic::LinearCollectExecutionItem;
use crate::linear_didactic::{
    build_linear_collect_additive_execution_items_with,
    build_linear_collect_factored_execution_items_with,
};
use crate::linear_kernel::derive_linear_solve_kernel;
use crate::linear_solution::{
    build_linear_solution_set, derive_linear_nonzero_statuses, NonZeroStatus,
};
use crate::linear_terms::{build_sum, decompose_linear_collect_terms};
use cas_ast::{Context, Expr, ExprId, SolutionSet};

/// Normalized linear-collect kernel for factored equations:
/// `coeff * var = rhs_term`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearCollectFactoredKernel {
    pub coeff: ExprId,
    pub rhs_term: ExprId,
}

/// Normalized linear-collect kernel for additive equations:
/// `coeff * var + constant = 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearCollectAdditiveKernel {
    pub coeff: ExprId,
    pub constant: ExprId,
}

/// Engine-facing payload for linear-collect execution.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearCollectSolveExecution {
    pub items: Vec<LinearCollectExecutionItem>,
    pub solutions: SolutionSet,
}

/// Solved payload for linear-collect execution dispatch.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearCollectSolvedExecution<T> {
    pub execution: LinearCollectSolveExecution,
    pub solved: T,
}

/// Solved factored linear-collect kernel ready for didactic materialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearCollectFactoredSolve {
    pub coeff: ExprId,
    pub rhs_term: ExprId,
    pub solution: ExprId,
    pub coeff_status: NonZeroStatus,
    pub rhs_status: NonZeroStatus,
}

/// Solved additive linear-collect kernel ready for didactic materialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearCollectAdditiveSolve {
    pub coeff: ExprId,
    pub constant: ExprId,
    pub solution: ExprId,
    pub coeff_status: NonZeroStatus,
    pub constant_status: NonZeroStatus,
}

/// Derive linear-collect kernel from a normalized `expr = 0` form.
///
/// Expected input is usually `expr = simplify(lhs - rhs)`.
pub fn derive_linear_collect_factored_kernel(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<LinearCollectFactoredKernel> {
    let decomposition = decompose_linear_collect_terms(ctx, expr, var)?;
    let coeff = build_sum(ctx, &decomposition.coeff_parts);
    let const_sum = build_sum(ctx, &decomposition.const_parts);
    let rhs_term = ctx.add(Expr::Neg(const_sum));

    Some(LinearCollectFactoredKernel { coeff, rhs_term })
}

/// Derive linear-collect additive kernel from equation sides.
pub fn derive_linear_collect_additive_kernel(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
) -> Option<LinearCollectAdditiveKernel> {
    let kernel = derive_linear_solve_kernel(ctx, lhs, rhs, var)?;
    Some(LinearCollectAdditiveKernel {
        coeff: kernel.coef,
        constant: kernel.constant,
    })
}

/// Build linear candidate solution expression: `numerator / coeff`.
pub fn build_linear_collect_solution_expr(
    ctx: &mut Context,
    numerator: ExprId,
    coeff: ExprId,
) -> ExprId {
    ctx.add(Expr::Div(numerator, coeff))
}

/// Solve factored linear-collect form from equation sides using closure hooks.
#[allow(clippy::too_many_arguments)]
pub fn solve_linear_collect_factored_with<
    FBuildDifferenceExpr,
    FSimplifyExpr,
    FDeriveKernel,
    FBuildSolutionExpr,
    FContainsVar,
    FProveNonzeroStatus,
>(
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    mut build_difference_expr: FBuildDifferenceExpr,
    mut simplify_expr: FSimplifyExpr,
    mut derive_kernel: FDeriveKernel,
    mut build_solution_expr: FBuildSolutionExpr,
    mut contains_var_hook: FContainsVar,
    mut prove_nonzero_status: FProveNonzeroStatus,
) -> Option<LinearCollectFactoredSolve>
where
    FBuildDifferenceExpr: FnMut(ExprId, ExprId) -> ExprId,
    FSimplifyExpr: FnMut(ExprId) -> ExprId,
    FDeriveKernel: FnMut(ExprId, &str) -> Option<LinearCollectFactoredKernel>,
    FBuildSolutionExpr: FnMut(ExprId, ExprId) -> ExprId,
    FContainsVar: FnMut(ExprId, &str) -> bool,
    FProveNonzeroStatus: FnMut(ExprId) -> NonZeroStatus,
{
    let expr = build_difference_expr(lhs, rhs);
    let expr = simplify_expr(expr);
    let kernel = derive_kernel(expr, var)?;
    let coeff = simplify_expr(kernel.coeff);
    let rhs_term = simplify_expr(kernel.rhs_term);
    let solution = simplify_expr(build_solution_expr(rhs_term, coeff));

    let coeff_contains_var = contains_var_hook(coeff, var);
    let (coeff_status, rhs_status) = if coeff_contains_var {
        (NonZeroStatus::Unknown, NonZeroStatus::Unknown)
    } else {
        let coeff_status = prove_nonzero_status(coeff);
        let rhs_status = if coeff_status == NonZeroStatus::Zero {
            prove_nonzero_status(rhs_term)
        } else {
            NonZeroStatus::Unknown
        };
        (coeff_status, rhs_status)
    };

    Some(LinearCollectFactoredSolve {
        coeff,
        rhs_term,
        solution,
        coeff_status,
        rhs_status,
    })
}

/// Solve additive linear-collect form from equation sides using closure hooks.
#[allow(clippy::too_many_arguments)]
pub fn solve_linear_collect_additive_with<
    FDeriveKernel,
    FSimplifyExpr,
    FBuildSolutionExpr,
    FContainsVar,
    FProveNonzeroStatus,
>(
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    mut derive_kernel: FDeriveKernel,
    mut simplify_expr: FSimplifyExpr,
    mut build_solution_expr: FBuildSolutionExpr,
    mut contains_var_hook: FContainsVar,
    mut prove_nonzero_status: FProveNonzeroStatus,
) -> Option<LinearCollectAdditiveSolve>
where
    FDeriveKernel: FnMut(ExprId, ExprId, &str) -> Option<LinearCollectAdditiveKernel>,
    FSimplifyExpr: FnMut(ExprId) -> ExprId,
    FBuildSolutionExpr: FnMut(ExprId, ExprId) -> ExprId,
    FContainsVar: FnMut(ExprId, &str) -> bool,
    FProveNonzeroStatus: FnMut(ExprId) -> NonZeroStatus,
{
    let kernel = derive_kernel(lhs, rhs, var)?;
    let coeff = simplify_expr(kernel.coeff);
    let constant = simplify_expr(kernel.constant);
    let solution = simplify_expr(build_solution_expr(constant, coeff));

    let coeff_contains_var = contains_var_hook(coeff, var);
    let (coeff_status, constant_status) = if coeff_contains_var {
        (NonZeroStatus::Unknown, NonZeroStatus::Unknown)
    } else {
        let coeff_status = prove_nonzero_status(coeff);
        let constant_status = if coeff_status == NonZeroStatus::Zero {
            prove_nonzero_status(constant)
        } else {
            NonZeroStatus::Unknown
        };
        (coeff_status, constant_status)
    };

    Some(LinearCollectAdditiveSolve {
        coeff,
        constant,
        solution,
        coeff_status,
        constant_status,
    })
}

/// Derive proof statuses for a linear-collect kernel.
pub fn derive_linear_collect_statuses_with<F>(
    ctx: &Context,
    var: &str,
    coeff: ExprId,
    constant: ExprId,
    prove_nonzero_status: F,
) -> (NonZeroStatus, NonZeroStatus)
where
    F: FnMut(ExprId) -> NonZeroStatus,
{
    derive_linear_nonzero_statuses(
        contains_var(ctx, coeff, var),
        coeff,
        constant,
        prove_nonzero_status,
    )
}

/// Build full factored linear-collect execution payload.
#[allow(clippy::too_many_arguments)]
pub fn build_linear_collect_factored_execution_with<F>(
    ctx: &mut Context,
    var: &str,
    coeff: ExprId,
    rhs_term: ExprId,
    solution: ExprId,
    coeff_status: NonZeroStatus,
    constant_status: NonZeroStatus,
    render_expr: F,
) -> LinearCollectSolveExecution
where
    F: FnMut(&Context, ExprId) -> String,
{
    let items = build_linear_collect_factored_execution_items_with(
        ctx,
        var,
        coeff,
        rhs_term,
        solution,
        render_expr,
    );
    let solutions =
        build_linear_solution_set(coeff, rhs_term, solution, coeff_status, constant_status);
    LinearCollectSolveExecution { items, solutions }
}

/// Build full additive linear-collect execution payload.
#[allow(clippy::too_many_arguments)]
pub fn build_linear_collect_additive_execution_with<F>(
    ctx: &mut Context,
    var: &str,
    coeff: ExprId,
    constant: ExprId,
    solution: ExprId,
    coeff_status: NonZeroStatus,
    constant_status: NonZeroStatus,
    render_expr: F,
) -> LinearCollectSolveExecution
where
    F: FnMut(&Context, ExprId) -> String,
{
    let items = build_linear_collect_additive_execution_items_with(
        ctx,
        var,
        coeff,
        constant,
        solution,
        render_expr,
    );
    let solutions =
        build_linear_solution_set(coeff, constant, solution, coeff_status, constant_status);
    LinearCollectSolveExecution { items, solutions }
}

/// Dispatch linear-collect execution items plus solved solution set to a caller
/// callback, preserving execution payload in the returned solved wrapper.
pub fn solve_linear_collect_execution_with_items<T, FSolve>(
    execution: LinearCollectSolveExecution,
    mut solve: FSolve,
) -> LinearCollectSolvedExecution<T>
where
    FSolve: FnMut(Vec<LinearCollectExecutionItem>, SolutionSet) -> T,
{
    let items = execution.items.clone();
    let solutions = execution.solutions.clone();
    let solved = solve(items, solutions);
    LinearCollectSolvedExecution { execution, solved }
}

/// Solve linear-collect execution while optionally mapping execution items
/// into caller-owned step payloads.
pub fn solve_linear_collect_execution_pipeline_with_items<S, FStep>(
    execution: LinearCollectSolveExecution,
    include_items: bool,
    mut map_item_to_step: FStep,
) -> LinearCollectSolvedExecution<(SolutionSet, Vec<S>)>
where
    FStep: FnMut(LinearCollectExecutionItem) -> S,
{
    solve_linear_collect_execution_with_items(execution, |items, solutions| {
        let mut steps = Vec::new();
        if include_items {
            for item in items {
                steps.push(map_item_to_step(item));
            }
        }
        (solutions, steps)
    })
}

/// Solve factored linear-collect with closure hooks and optional didactic mapping.
pub fn solve_linear_collect_factored_pipeline_with_and_items<S, FSolve, FBuildExecution, FStep>(
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    include_items: bool,
    mut solve_factored: FSolve,
    mut build_execution: FBuildExecution,
    map_item_to_step: FStep,
) -> Option<(SolutionSet, Vec<S>)>
where
    FSolve: FnMut(ExprId, ExprId, &str) -> Option<LinearCollectFactoredSolve>,
    FBuildExecution: FnMut(&str, LinearCollectFactoredSolve) -> LinearCollectSolveExecution,
    FStep: FnMut(LinearCollectExecutionItem) -> S,
{
    let solved = solve_factored(lhs, rhs, var)?;
    if include_items {
        let execution = build_execution(var, solved);
        let solved_execution =
            solve_linear_collect_execution_pipeline_with_items(execution, true, map_item_to_step);
        return Some(solved_execution.solved);
    }

    let solution_set = build_linear_solution_set(
        solved.coeff,
        solved.rhs_term,
        solved.solution,
        solved.coeff_status,
        solved.rhs_status,
    );
    Some((solution_set, Vec::new()))
}

/// Solve additive linear-collect with closure hooks and optional didactic mapping.
pub fn solve_linear_collect_additive_pipeline_with_and_items<S, FSolve, FBuildExecution, FStep>(
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    include_items: bool,
    mut solve_additive: FSolve,
    mut build_execution: FBuildExecution,
    map_item_to_step: FStep,
) -> Option<(SolutionSet, Vec<S>)>
where
    FSolve: FnMut(ExprId, ExprId, &str) -> Option<LinearCollectAdditiveSolve>,
    FBuildExecution: FnMut(&str, LinearCollectAdditiveSolve) -> LinearCollectSolveExecution,
    FStep: FnMut(LinearCollectExecutionItem) -> S,
{
    let solved = solve_additive(lhs, rhs, var)?;
    if include_items {
        let execution = build_execution(var, solved);
        let solved_execution =
            solve_linear_collect_execution_pipeline_with_items(execution, true, map_item_to_step);
        return Some(solved_execution.solved);
    }

    let solution_set = build_linear_solution_set(
        solved.coeff,
        solved.constant,
        solved.solution,
        solved.coeff_status,
        solved.constant_status,
    );
    Some((solution_set, Vec::new()))
}

/// High-level factored linear-collect pipeline using injected hooks for
/// equation rewriting, simplification, proof checks, and optional didactic mapping.
#[allow(clippy::too_many_arguments)]
pub fn execute_linear_collect_factored_pipeline_with_and_items<
    S,
    FBuildDifferenceExpr,
    FSimplifyExpr,
    FDeriveKernel,
    FBuildSolutionExpr,
    FContainsVar,
    FProveNonzeroStatus,
    FBuildExecution,
    FStep,
>(
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    include_items: bool,
    mut build_difference_expr: FBuildDifferenceExpr,
    mut simplify_expr: FSimplifyExpr,
    mut derive_kernel: FDeriveKernel,
    mut build_solution_expr: FBuildSolutionExpr,
    mut contains_var_hook: FContainsVar,
    mut prove_nonzero_status: FProveNonzeroStatus,
    build_execution: FBuildExecution,
    map_item_to_step: FStep,
) -> Option<(SolutionSet, Vec<S>)>
where
    FBuildDifferenceExpr: FnMut(ExprId, ExprId) -> ExprId,
    FSimplifyExpr: FnMut(ExprId) -> ExprId,
    FDeriveKernel: FnMut(ExprId, &str) -> Option<LinearCollectFactoredKernel>,
    FBuildSolutionExpr: FnMut(ExprId, ExprId) -> ExprId,
    FContainsVar: FnMut(ExprId, &str) -> bool,
    FProveNonzeroStatus: FnMut(ExprId) -> NonZeroStatus,
    FBuildExecution: FnMut(&str, LinearCollectFactoredSolve) -> LinearCollectSolveExecution,
    FStep: FnMut(LinearCollectExecutionItem) -> S,
{
    solve_linear_collect_factored_pipeline_with_and_items(
        lhs,
        rhs,
        var,
        include_items,
        |left, right, name| {
            solve_linear_collect_factored_with(
                left,
                right,
                name,
                &mut build_difference_expr,
                &mut simplify_expr,
                &mut derive_kernel,
                &mut build_solution_expr,
                &mut contains_var_hook,
                &mut prove_nonzero_status,
            )
        },
        build_execution,
        map_item_to_step,
    )
}

/// High-level additive linear-collect pipeline using injected hooks for
/// kernel derivation, simplification, proof checks, and optional didactic mapping.
#[allow(clippy::too_many_arguments)]
pub fn execute_linear_collect_additive_pipeline_with_and_items<
    S,
    FDeriveKernel,
    FSimplifyExpr,
    FBuildSolutionExpr,
    FContainsVar,
    FProveNonzeroStatus,
    FBuildExecution,
    FStep,
>(
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    include_items: bool,
    mut derive_kernel: FDeriveKernel,
    mut simplify_expr: FSimplifyExpr,
    mut build_solution_expr: FBuildSolutionExpr,
    mut contains_var_hook: FContainsVar,
    mut prove_nonzero_status: FProveNonzeroStatus,
    build_execution: FBuildExecution,
    map_item_to_step: FStep,
) -> Option<(SolutionSet, Vec<S>)>
where
    FDeriveKernel: FnMut(ExprId, ExprId, &str) -> Option<LinearCollectAdditiveKernel>,
    FSimplifyExpr: FnMut(ExprId) -> ExprId,
    FBuildSolutionExpr: FnMut(ExprId, ExprId) -> ExprId,
    FContainsVar: FnMut(ExprId, &str) -> bool,
    FProveNonzeroStatus: FnMut(ExprId) -> NonZeroStatus,
    FBuildExecution: FnMut(&str, LinearCollectAdditiveSolve) -> LinearCollectSolveExecution,
    FStep: FnMut(LinearCollectExecutionItem) -> S,
{
    solve_linear_collect_additive_pipeline_with_and_items(
        lhs,
        rhs,
        var,
        include_items,
        |left, right, name| {
            solve_linear_collect_additive_with(
                left,
                right,
                name,
                &mut derive_kernel,
                &mut simplify_expr,
                &mut build_solution_expr,
                &mut contains_var_hook,
                &mut prove_nonzero_status,
            )
        },
        build_execution,
        map_item_to_step,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_solution::NonZeroStatus;
    use cas_ast::Equation;

    #[test]
    fn derive_linear_collect_factored_kernel_extracts_coeff_and_rhs() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let a = ctx.var("a");
        let b = ctx.var("b");
        let ax = ctx.add(Expr::Mul(a, x));
        let expr = ctx.add(Expr::Sub(ax, b));

        let kernel = derive_linear_collect_factored_kernel(&mut ctx, expr, "x")
            .expect("must derive factored kernel");
        assert!(!contains_var(&ctx, kernel.rhs_term, "x"));
        assert!(!contains_var(&ctx, kernel.coeff, "y"));
    }

    #[test]
    fn solve_linear_collect_factored_with_uses_injected_hooks() {
        let mut context = Context::new();
        let x = context.var("x");
        let a = context.var("a");
        let b = context.var("b");
        let lhs = context.add(Expr::Mul(a, x));
        let rhs = b;
        let context_cell = std::cell::RefCell::new(context);
        let simplify_calls = std::cell::Cell::new(0usize);
        let prove_calls = std::cell::Cell::new(0usize);

        let solved = solve_linear_collect_factored_with(
            lhs,
            rhs,
            "x",
            |left, right| {
                let mut ctx = context_cell.borrow_mut();
                ctx.add(Expr::Sub(left, right))
            },
            |expr| {
                simplify_calls.set(simplify_calls.get() + 1);
                expr
            },
            |expr, name| {
                let mut ctx = context_cell.borrow_mut();
                derive_linear_collect_factored_kernel(&mut ctx, expr, name)
            },
            |numerator, coeff| {
                let mut ctx = context_cell.borrow_mut();
                build_linear_collect_solution_expr(&mut ctx, numerator, coeff)
            },
            |expr, name| {
                let ctx = context_cell.borrow();
                contains_var(&ctx, expr, name)
            },
            |_expr| {
                prove_calls.set(prove_calls.get() + 1);
                NonZeroStatus::NonZero
            },
        )
        .expect("should solve factored linear collect");

        assert_eq!(simplify_calls.get(), 4);
        assert_eq!(prove_calls.get(), 1);
        assert_eq!(solved.coeff_status, NonZeroStatus::NonZero);
        assert_eq!(solved.rhs_status, NonZeroStatus::Unknown);
    }

    #[test]
    fn derive_linear_collect_additive_kernel_maps_linear_kernel() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let a = ctx.var("a");
        let b = ctx.var("b");
        let ax = ctx.add(Expr::Mul(a, x));
        let lhs = ctx.add(Expr::Add(ax, b));
        let rhs = ctx.num(0);

        let kernel = derive_linear_collect_additive_kernel(&mut ctx, lhs, rhs, "x")
            .expect("must derive additive kernel");
        assert!(!contains_var(&ctx, kernel.coeff, "y"));
    }

    #[test]
    fn solve_linear_collect_additive_with_uses_injected_hooks() {
        let mut context = Context::new();
        let x = context.var("x");
        let a = context.var("a");
        let b = context.var("b");
        let ax = context.add(Expr::Mul(a, x));
        let lhs = context.add(Expr::Add(ax, b));
        let rhs = context.num(0);
        let context_cell = std::cell::RefCell::new(context);
        let simplify_calls = std::cell::Cell::new(0usize);
        let prove_calls = std::cell::Cell::new(0usize);

        let solved = solve_linear_collect_additive_with(
            lhs,
            rhs,
            "x",
            |left, right, name| {
                let mut ctx = context_cell.borrow_mut();
                derive_linear_collect_additive_kernel(&mut ctx, left, right, name)
            },
            |expr| {
                simplify_calls.set(simplify_calls.get() + 1);
                expr
            },
            |constant, coeff| {
                let mut ctx = context_cell.borrow_mut();
                let neg_constant = ctx.add(Expr::Neg(constant));
                build_linear_collect_solution_expr(&mut ctx, neg_constant, coeff)
            },
            |expr, name| {
                let ctx = context_cell.borrow();
                contains_var(&ctx, expr, name)
            },
            |_expr| {
                prove_calls.set(prove_calls.get() + 1);
                NonZeroStatus::NonZero
            },
        )
        .expect("should solve additive linear collect");

        assert_eq!(simplify_calls.get(), 3);
        assert_eq!(prove_calls.get(), 1);
        assert_eq!(solved.coeff_status, NonZeroStatus::NonZero);
        assert_eq!(solved.constant_status, NonZeroStatus::Unknown);
    }

    #[test]
    fn derive_linear_collect_statuses_with_delegates_to_proof_callback() {
        let mut ctx = Context::new();
        let coeff = ctx.var("a");
        let constant = ctx.var("b");
        let mut seen = Vec::new();

        let (coef_status, constant_status) =
            derive_linear_collect_statuses_with(&ctx, "x", coeff, constant, |expr| {
                seen.push(expr);
                if expr == coeff {
                    NonZeroStatus::Zero
                } else {
                    NonZeroStatus::NonZero
                }
            });

        assert_eq!(coef_status, NonZeroStatus::Zero);
        assert_eq!(constant_status, NonZeroStatus::NonZero);
        assert_eq!(seen, vec![coeff, constant]);
    }

    #[test]
    fn build_linear_collect_factored_execution_with_produces_steps_and_solution() {
        let mut ctx = Context::new();
        let coeff = ctx.var("k");
        let rhs_term = ctx.var("rhs");
        let solution = ctx.var("s");

        let execution = build_linear_collect_factored_execution_with(
            &mut ctx,
            "x",
            coeff,
            rhs_term,
            solution,
            NonZeroStatus::NonZero,
            NonZeroStatus::Unknown,
            |_, _| "k".into(),
        );

        assert_eq!(execution.items.len(), 2);
        assert!(matches!(execution.solutions, SolutionSet::Discrete(_)));
    }

    #[test]
    fn build_linear_collect_additive_execution_with_produces_conditional_solution() {
        let mut ctx = Context::new();
        let coeff = ctx.var("k");
        let constant = ctx.var("c");
        let solution = ctx.var("s");

        let execution = build_linear_collect_additive_execution_with(
            &mut ctx,
            "x",
            coeff,
            constant,
            solution,
            NonZeroStatus::Unknown,
            NonZeroStatus::Unknown,
            |_, _| "k".into(),
        );

        assert_eq!(execution.items.len(), 2);
        assert!(matches!(execution.solutions, SolutionSet::Conditional(_)));
    }

    #[test]
    fn solve_linear_collect_execution_with_items_passes_items_and_solution_set() {
        let mut ctx = Context::new();
        let coeff = ctx.var("k");
        let rhs_term = ctx.var("rhs");
        let solution = ctx.var("s");

        let execution = build_linear_collect_factored_execution_with(
            &mut ctx,
            "x",
            coeff,
            rhs_term,
            solution,
            NonZeroStatus::NonZero,
            NonZeroStatus::Unknown,
            |_, _| "k".into(),
        );
        let expected = execution.clone();

        let solved_exec =
            solve_linear_collect_execution_with_items(execution, |items, solutions| {
                assert_eq!(items, expected.items);
                solutions
            });

        assert_eq!(solved_exec.execution, expected);
        assert!(matches!(solved_exec.solved, SolutionSet::Discrete(_)));
    }

    #[test]
    fn solve_linear_collect_execution_pipeline_with_items_maps_steps_when_enabled() {
        let mut ctx = Context::new();
        let coeff = ctx.var("k");
        let rhs_term = ctx.var("rhs");
        let solution = ctx.var("s");

        let execution = build_linear_collect_factored_execution_with(
            &mut ctx,
            "x",
            coeff,
            rhs_term,
            solution,
            NonZeroStatus::NonZero,
            NonZeroStatus::Unknown,
            |_, _| "k".into(),
        );
        let expected = execution.clone();
        let map_calls = std::cell::Cell::new(0usize);

        let solved = solve_linear_collect_execution_pipeline_with_items(execution, true, |item| {
            map_calls.set(map_calls.get() + 1);
            item.description
        });

        assert_eq!(solved.execution, expected);
        assert!(matches!(solved.solved.0, SolutionSet::Discrete(_)));
        assert_eq!(solved.solved.1.len(), expected.items.len());
        assert_eq!(map_calls.get(), expected.items.len());
    }

    #[test]
    fn solve_linear_collect_execution_pipeline_with_items_omits_steps_when_disabled() {
        let mut ctx = Context::new();
        let coeff = ctx.var("k");
        let rhs_term = ctx.var("rhs");
        let solution = ctx.var("s");

        let execution = build_linear_collect_factored_execution_with(
            &mut ctx,
            "x",
            coeff,
            rhs_term,
            solution,
            NonZeroStatus::NonZero,
            NonZeroStatus::Unknown,
            |_, _| "k".into(),
        );

        let solved =
            solve_linear_collect_execution_pipeline_with_items(execution, false, |_item| 1u8);
        assert!(solved.solved.1.is_empty());
    }

    #[test]
    fn solve_linear_collect_factored_pipeline_with_and_items_maps_steps_when_enabled() {
        let mut context = Context::new();
        let x = context.var("x");
        let a = context.var("a");
        let b = context.var("b");
        let lhs = context.add(Expr::Mul(a, x));
        let rhs = b;
        let context_cell = std::cell::RefCell::new(context);
        let simplify_calls = std::cell::Cell::new(0usize);
        let prove_calls = std::cell::Cell::new(0usize);
        let map_calls = std::cell::Cell::new(0usize);

        let (solution_set, steps) = solve_linear_collect_factored_pipeline_with_and_items(
            lhs,
            rhs,
            "x",
            true,
            |left, right, name| {
                solve_linear_collect_factored_with(
                    left,
                    right,
                    name,
                    |lhs_expr, rhs_expr| {
                        let mut ctx = context_cell.borrow_mut();
                        ctx.add(Expr::Sub(lhs_expr, rhs_expr))
                    },
                    |expr| {
                        simplify_calls.set(simplify_calls.get() + 1);
                        expr
                    },
                    |expr, inner_name| {
                        let mut ctx = context_cell.borrow_mut();
                        derive_linear_collect_factored_kernel(&mut ctx, expr, inner_name)
                    },
                    |numerator, coeff| {
                        let mut ctx = context_cell.borrow_mut();
                        build_linear_collect_solution_expr(&mut ctx, numerator, coeff)
                    },
                    |expr, inner_name| {
                        let ctx = context_cell.borrow();
                        contains_var(&ctx, expr, inner_name)
                    },
                    |_expr| {
                        prove_calls.set(prove_calls.get() + 1);
                        NonZeroStatus::NonZero
                    },
                )
            },
            |name, solved| {
                let mut ctx = context_cell.borrow_mut();
                build_linear_collect_factored_execution_with(
                    &mut ctx,
                    name,
                    solved.coeff,
                    solved.rhs_term,
                    solved.solution,
                    solved.coeff_status,
                    solved.rhs_status,
                    |_, _| "k".to_string(),
                )
            },
            |item| {
                map_calls.set(map_calls.get() + 1);
                item.description
            },
        )
        .expect("pipeline should solve");

        assert!(matches!(solution_set, SolutionSet::Discrete(_)));
        assert_eq!(steps.len(), 2);
        assert_eq!(simplify_calls.get(), 4);
        assert_eq!(prove_calls.get(), 1);
        assert_eq!(map_calls.get(), 2);
    }

    #[test]
    fn solve_linear_collect_additive_pipeline_with_and_items_omits_steps_when_disabled() {
        let mut context = Context::new();
        let x = context.var("x");
        let a = context.var("a");
        let b = context.var("b");
        let ax = context.add(Expr::Mul(a, x));
        let lhs = context.add(Expr::Add(ax, b));
        let rhs = context.num(0);
        let context_cell = std::cell::RefCell::new(context);
        let simplify_calls = std::cell::Cell::new(0usize);
        let prove_calls = std::cell::Cell::new(0usize);

        let (solution_set, steps) = solve_linear_collect_additive_pipeline_with_and_items(
            lhs,
            rhs,
            "x",
            false,
            |left, right, name| {
                solve_linear_collect_additive_with(
                    left,
                    right,
                    name,
                    |lhs_expr, rhs_expr, inner_name| {
                        let mut ctx = context_cell.borrow_mut();
                        derive_linear_collect_additive_kernel(
                            &mut ctx, lhs_expr, rhs_expr, inner_name,
                        )
                    },
                    |expr| {
                        simplify_calls.set(simplify_calls.get() + 1);
                        expr
                    },
                    |constant, coeff| {
                        let mut ctx = context_cell.borrow_mut();
                        let neg_constant = ctx.add(Expr::Neg(constant));
                        build_linear_collect_solution_expr(&mut ctx, neg_constant, coeff)
                    },
                    |expr, inner_name| {
                        let ctx = context_cell.borrow();
                        contains_var(&ctx, expr, inner_name)
                    },
                    |_expr| {
                        prove_calls.set(prove_calls.get() + 1);
                        NonZeroStatus::NonZero
                    },
                )
            },
            |_name, _solved| panic!("execution builder must not run when items are disabled"),
            |_item| -> () { panic!("step mapper must not run when items are disabled") },
        )
        .expect("pipeline should solve");

        assert!(matches!(solution_set, SolutionSet::Discrete(_)));
        assert!(steps.is_empty());
        assert_eq!(simplify_calls.get(), 3);
        assert_eq!(prove_calls.get(), 1);
    }

    #[test]
    fn execute_linear_collect_factored_pipeline_with_and_items_maps_steps_when_enabled() {
        let mut context = Context::new();
        let x = context.var("x");
        let a = context.var("a");
        let b = context.var("b");
        let lhs = context.add(Expr::Mul(a, x));
        let rhs = b;
        let context_cell = std::cell::RefCell::new(context);
        let simplify_calls = std::cell::Cell::new(0usize);
        let prove_calls = std::cell::Cell::new(0usize);
        let map_calls = std::cell::Cell::new(0usize);

        let (solution_set, steps) = execute_linear_collect_factored_pipeline_with_and_items(
            lhs,
            rhs,
            "x",
            true,
            |left, right| {
                let mut ctx = context_cell.borrow_mut();
                ctx.add(Expr::Sub(left, right))
            },
            |expr| {
                simplify_calls.set(simplify_calls.get() + 1);
                expr
            },
            |expr, inner_name| {
                let mut ctx = context_cell.borrow_mut();
                derive_linear_collect_factored_kernel(&mut ctx, expr, inner_name)
            },
            |numerator, coeff| {
                let mut ctx = context_cell.borrow_mut();
                build_linear_collect_solution_expr(&mut ctx, numerator, coeff)
            },
            |expr, inner_name| {
                let ctx = context_cell.borrow();
                contains_var(&ctx, expr, inner_name)
            },
            |_expr| {
                prove_calls.set(prove_calls.get() + 1);
                NonZeroStatus::NonZero
            },
            |name, solved| {
                let mut ctx = context_cell.borrow_mut();
                build_linear_collect_factored_execution_with(
                    &mut ctx,
                    name,
                    solved.coeff,
                    solved.rhs_term,
                    solved.solution,
                    solved.coeff_status,
                    solved.rhs_status,
                    |_, _| "k".to_string(),
                )
            },
            |item| {
                map_calls.set(map_calls.get() + 1);
                item.description
            },
        )
        .expect("factored execute pipeline should solve");

        assert!(matches!(solution_set, SolutionSet::Discrete(_)));
        assert_eq!(steps.len(), 2);
        assert_eq!(simplify_calls.get(), 4);
        assert_eq!(prove_calls.get(), 1);
        assert_eq!(map_calls.get(), 2);
    }

    #[test]
    fn execute_linear_collect_additive_pipeline_with_and_items_omits_steps_when_disabled() {
        let mut context = Context::new();
        let x = context.var("x");
        let a = context.var("a");
        let b = context.var("b");
        let ax = context.add(Expr::Mul(a, x));
        let lhs = context.add(Expr::Add(ax, b));
        let rhs = context.num(0);
        let context_cell = std::cell::RefCell::new(context);
        let simplify_calls = std::cell::Cell::new(0usize);
        let prove_calls = std::cell::Cell::new(0usize);

        let (solution_set, steps) = execute_linear_collect_additive_pipeline_with_and_items(
            lhs,
            rhs,
            "x",
            false,
            |left, right, name| {
                let mut ctx = context_cell.borrow_mut();
                derive_linear_collect_additive_kernel(&mut ctx, left, right, name)
            },
            |expr| {
                simplify_calls.set(simplify_calls.get() + 1);
                expr
            },
            |constant, coeff| {
                let mut ctx = context_cell.borrow_mut();
                let neg_constant = ctx.add(Expr::Neg(constant));
                build_linear_collect_solution_expr(&mut ctx, neg_constant, coeff)
            },
            |expr, inner_name| {
                let ctx = context_cell.borrow();
                contains_var(&ctx, expr, inner_name)
            },
            |_expr| {
                prove_calls.set(prove_calls.get() + 1);
                NonZeroStatus::NonZero
            },
            |_name, _solved| panic!("execution builder must not run when items are disabled"),
            |_item| -> () { panic!("step mapper must not run when items are disabled") },
        )
        .expect("additive execute pipeline should solve");

        assert!(matches!(solution_set, SolutionSet::Discrete(_)));
        assert!(steps.is_empty());
        assert_eq!(simplify_calls.get(), 3);
        assert_eq!(prove_calls.get(), 1);
    }

    #[test]
    fn build_linear_collect_solution_expr_builds_division() {
        let mut ctx = Context::new();
        let n = ctx.var("n");
        let d = ctx.var("d");
        let expr = build_linear_collect_solution_expr(&mut ctx, n, d);
        assert!(matches!(ctx.get(expr), Expr::Div(_, _)));
    }

    #[test]
    fn execution_items_preserve_equation_shapes() {
        let mut ctx = Context::new();
        let coeff = ctx.var("k");
        let rhs_term = ctx.var("rhs");
        let solution = ctx.var("s");

        let execution = build_linear_collect_factored_execution_with(
            &mut ctx,
            "x",
            coeff,
            rhs_term,
            solution,
            NonZeroStatus::NonZero,
            NonZeroStatus::Unknown,
            |_, _| "expr".into(),
        );

        let first: &Equation = &execution.items[0].equation;
        let second: &Equation = &execution.items[1].equation;
        assert_eq!(first.op, cas_ast::RelOp::Eq);
        assert_eq!(second.op, cas_ast::RelOp::Eq);
    }
}
