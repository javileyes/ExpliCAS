use crate::isolation_utils::contains_var;
use crate::linear_didactic::{
    build_linear_collect_additive_execution_items_with,
    build_linear_collect_factored_execution_items_with, LinearCollectExecutionItem,
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

/// Runtime contract for linear-collect orchestration.
///
/// This allows solver-core to own linear collection logic while callers inject
/// simplification/proof behavior.
pub trait LinearCollectRuntime {
    /// Mutable access to context for expression construction.
    fn context(&mut self) -> &mut Context;
    /// Simplify one expression and return rewritten root.
    fn simplify_expr(&mut self, expr: ExprId) -> ExprId;
    /// Prove whether expression is non-zero.
    fn prove_nonzero_status(&mut self, expr: ExprId) -> NonZeroStatus;
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

/// Solve factored linear-collect form from equation sides using injected runtime.
///
/// Normalizes `lhs-rhs=0`, extracts `coeff*var = rhs_term`, simplifies terms,
/// builds candidate solution, and derives non-zero proof statuses.
pub fn solve_linear_collect_factored_with_runtime<R>(
    runtime: &mut R,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
) -> Option<LinearCollectFactoredSolve>
where
    R: LinearCollectRuntime,
{
    let expr = {
        let ctx = runtime.context();
        ctx.add(Expr::Sub(lhs, rhs))
    };
    let expr = runtime.simplify_expr(expr);

    let kernel = {
        let ctx = runtime.context();
        derive_linear_collect_factored_kernel(ctx, expr, var)?
    };
    let coeff = runtime.simplify_expr(kernel.coeff);
    let rhs_term = runtime.simplify_expr(kernel.rhs_term);
    let solution = {
        let ctx = runtime.context();
        build_linear_collect_solution_expr(ctx, rhs_term, coeff)
    };
    let solution = runtime.simplify_expr(solution);

    let coeff_contains_var = {
        let ctx = runtime.context();
        contains_var(ctx, coeff, var)
    };

    let (coeff_status, rhs_status) = if coeff_contains_var {
        (NonZeroStatus::Unknown, NonZeroStatus::Unknown)
    } else {
        let coeff_status = runtime.prove_nonzero_status(coeff);
        let rhs_status = if coeff_status == NonZeroStatus::Zero {
            runtime.prove_nonzero_status(rhs_term)
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

/// Solve additive linear-collect form from equation sides using injected runtime.
///
/// Extracts `coeff*var + constant = 0`, simplifies terms, builds candidate
/// solution `-constant/coeff`, and derives non-zero proof statuses.
pub fn solve_linear_collect_additive_with_runtime<R>(
    runtime: &mut R,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
) -> Option<LinearCollectAdditiveSolve>
where
    R: LinearCollectRuntime,
{
    let kernel = {
        let ctx = runtime.context();
        derive_linear_collect_additive_kernel(ctx, lhs, rhs, var)?
    };
    let coeff = runtime.simplify_expr(kernel.coeff);
    let constant = runtime.simplify_expr(kernel.constant);

    let solution = {
        let ctx = runtime.context();
        let neg_constant = ctx.add(Expr::Neg(constant));
        build_linear_collect_solution_expr(ctx, neg_constant, coeff)
    };
    let solution = runtime.simplify_expr(solution);

    let coeff_contains_var = {
        let ctx = runtime.context();
        contains_var(ctx, coeff, var)
    };

    let (coeff_status, constant_status) = if coeff_contains_var {
        (NonZeroStatus::Unknown, NonZeroStatus::Unknown)
    } else {
        let coeff_status = runtime.prove_nonzero_status(coeff);
        let constant_status = if coeff_status == NonZeroStatus::Zero {
            runtime.prove_nonzero_status(constant)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_solution::NonZeroStatus;
    use cas_ast::Equation;

    struct MockLinearCollectRuntime {
        context: Context,
        simplify_calls: usize,
        prove_calls: usize,
        prove_result: NonZeroStatus,
    }

    impl LinearCollectRuntime for MockLinearCollectRuntime {
        fn context(&mut self) -> &mut Context {
            &mut self.context
        }

        fn simplify_expr(&mut self, expr: ExprId) -> ExprId {
            self.simplify_calls += 1;
            expr
        }

        fn prove_nonzero_status(&mut self, _expr: ExprId) -> NonZeroStatus {
            self.prove_calls += 1;
            self.prove_result
        }
    }

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
    fn solve_linear_collect_factored_with_runtime_uses_runtime_hooks() {
        let mut context = Context::new();
        let x = context.var("x");
        let a = context.var("a");
        let b = context.var("b");
        let lhs = context.add(Expr::Mul(a, x));
        let rhs = b;
        let mut runtime = MockLinearCollectRuntime {
            context,
            simplify_calls: 0,
            prove_calls: 0,
            prove_result: NonZeroStatus::NonZero,
        };

        let solved = solve_linear_collect_factored_with_runtime(&mut runtime, lhs, rhs, "x")
            .expect("should solve factored linear collect");
        assert_eq!(runtime.simplify_calls, 4);
        assert_eq!(runtime.prove_calls, 1);
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
    fn solve_linear_collect_additive_with_runtime_uses_runtime_hooks() {
        let mut context = Context::new();
        let x = context.var("x");
        let a = context.var("a");
        let b = context.var("b");
        let ax = context.add(Expr::Mul(a, x));
        let lhs = context.add(Expr::Add(ax, b));
        let rhs = context.num(0);
        let mut runtime = MockLinearCollectRuntime {
            context,
            simplify_calls: 0,
            prove_calls: 0,
            prove_result: NonZeroStatus::NonZero,
        };

        let solved = solve_linear_collect_additive_with_runtime(&mut runtime, lhs, rhs, "x")
            .expect("should solve additive linear collect");
        assert_eq!(runtime.simplify_calls, 3);
        assert_eq!(runtime.prove_calls, 1);
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
