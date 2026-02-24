use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::isolation::isolate;
use crate::solver::solve_core::solve_with_ctx;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{
    medium_step, render_expr as solver_render_expr, SolveCtx, SolveStep, SolverOptions,
};
use cas_ast::{Equation, ExprId, SolutionSet};
use cas_solver_core::strategy_kernels::{
    build_collect_terms_execution_with, build_rational_exponent_execution,
    collect_collect_terms_execution_items, collect_rational_exponent_execution_items,
    derive_collect_terms_kernel, derive_isolation_strategy_routing,
    derive_rational_exponent_kernel_for_var, solve_collect_terms_rewrite_pipeline_with_item,
    solve_isolation_strategy_routing_with,
    solve_rational_exponent_rewrite_pipeline_with_item_with, CollectTermsSolvedRewrite,
    RationalExponentSolvedRewrite,
};
use cas_solver_core::unwrap_plan::{
    route_unwrap_entry_with_item, solve_unwrap_execution_pipeline_with_item_with,
    UnwrapEntryRouting, UnwrapExecutionPlan,
};
use std::cell::RefCell;

pub struct IsolationStrategy;

impl SolverStrategy for IsolationStrategy {
    fn name(&self) -> &str {
        "Isolation"
    }

    fn apply(
        &self,
        eq: &Equation,
        var: &str,
        simplifier: &mut Simplifier,
        opts: &SolverOptions,
        ctx: &SolveCtx,
    ) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        let routing = derive_isolation_strategy_routing(&simplifier.context, eq, var);
        let include_item = simplifier.collect_steps();
        solve_isolation_strategy_routing_with(
            routing,
            eq,
            var,
            include_item,
            |equation, var| {
                isolate(
                    equation.lhs,
                    equation.rhs,
                    equation.op.clone(),
                    var,
                    simplifier,
                    *opts,
                    ctx,
                )
            },
            |item| medium_step(item.description, item.equation),
            |var| CasError::VariableNotFound(var.to_string()),
        )
    }

    // Note: We use the default should_verify() = true here.
    // Selective verification in solve() handles symbolic solutions.
}

pub struct UnwrapStrategy;

fn run_unwrap_execution(
    execution: UnwrapExecutionPlan,
    other_side: ExprId,
    var: &str,
    simplifier: &mut Simplifier,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let include_item = simplifier.collect_steps();
    let simplifier_cell = RefCell::new(simplifier);
    let solved = solve_unwrap_execution_pipeline_with_item_with(
        execution,
        other_side,
        var,
        include_item,
        |record| {
            let s_ref = simplifier_cell.borrow();
            let event = crate::assumptions::AssumptionEvent::from_log_assumption(
                record.assumption,
                &s_ref.context,
                record.base,
                record.other_side,
            );
            crate::solver::note_assumption(event);
        },
        |equation, var| {
            let mut s_ref = simplifier_cell.borrow_mut();
            solve_with_ctx(equation, var, &mut s_ref, ctx)
        },
        |item| medium_step(item.description, item.equation),
    )?;
    Ok((solved.solution_set, solved.steps))
}

impl SolverStrategy for UnwrapStrategy {
    fn name(&self) -> &str {
        "Unwrap"
    }

    fn apply(
        &self,
        eq: &Equation,
        var: &str,
        simplifier: &mut Simplifier,
        opts: &SolverOptions,
        ctx: &SolveCtx,
    ) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        let mode = crate::solver::domain_guards::to_core_domain_mode(opts.domain_mode);
        let wildcard_scope = opts.assume_scope == crate::semantics::AssumeScope::Wildcard;

        let include_item = simplifier.collect_steps();
        let routed = route_unwrap_entry_with_item(
            &mut simplifier.context,
            eq,
            var,
            mode,
            wildcard_scope,
            " - use 'semantics preset complex'",
            include_item,
            |core_ctx, base, other_side| {
                crate::solver::domain_guards::classify_log_solve(
                    core_ctx,
                    base,
                    other_side,
                    opts,
                    &ctx.domain_env,
                )
            },
            solver_render_expr,
            |item| medium_step(item.description, item.equation),
        );
        let routed = routed?;
        Some(match routed {
            UnwrapEntryRouting::Terminal(solved_terminal) => {
                Ok((solved_terminal.solution_set, solved_terminal.steps))
            }
            UnwrapEntryRouting::Execution(selected) => run_unwrap_execution(
                selected.execution,
                selected.other_side,
                var,
                simplifier,
                ctx,
            ),
        })
    }

    // Note: We use the default should_verify() = true here.
    // Selective verification in solve() handles symbolic solutions.
}

// --- Helper for CollectTermsStrategy (currently unused) ---

// fn is_zero(ctx: &Context, expr: ExprId) -> bool {
//     matches!(ctx.get(expr), Expr::Number(n) if n.is_zero())
// }

// --- CollectTermsStrategy: Handles linear equations with variables on both sides ---

pub struct CollectTermsStrategy;

fn build_collect_terms_rewrite(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<CollectTermsSolvedRewrite> {
    let kernel = derive_collect_terms_kernel(&mut simplifier.context, eq, var)?;
    let (lhs_after, _) = simplifier.simplify(kernel.rewritten.lhs);
    let (rhs_after, _) = simplifier.simplify(kernel.rewritten.rhs);
    let execution =
        build_collect_terms_execution_with(lhs_after, rhs_after, eq.op.clone(), eq.rhs, |id| {
            solver_render_expr(&simplifier.context, id)
        });
    let items = collect_collect_terms_execution_items(&execution);
    Some(CollectTermsSolvedRewrite {
        equation: execution.equation,
        items,
    })
}

fn build_rational_exponent_rewrite(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<RationalExponentSolvedRewrite> {
    let kernel = derive_rational_exponent_kernel_for_var(&mut simplifier.context, eq, var)?;
    let (lhs_after, _) = simplifier.simplify(kernel.rewritten.lhs);
    let (rhs_after, _) = simplifier.simplify(kernel.rewritten.rhs);
    let execution = build_rational_exponent_execution(kernel.q, lhs_after, rhs_after);
    let items = collect_rational_exponent_execution_items(&execution);
    Some(RationalExponentSolvedRewrite {
        equation: execution.equation,
        items,
    })
}

impl SolverStrategy for CollectTermsStrategy {
    fn name(&self) -> &str {
        "Collect Terms"
    }

    fn apply(
        &self,
        eq: &Equation,
        var: &str,
        simplifier: &mut Simplifier,
        _opts: &SolverOptions,
        ctx: &SolveCtx,
    ) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        let rewrite = build_collect_terms_rewrite(eq, var, simplifier)?;
        let include_item = simplifier.collect_steps();
        let solved = solve_collect_terms_rewrite_pipeline_with_item(
            rewrite,
            var,
            include_item,
            |equation, var| solve_with_ctx(equation, var, simplifier, ctx),
            |item| medium_step(item.description, item.equation),
        );

        match solved {
            Ok(solved) => Some(Ok((solved.solution_set, solved.steps))),
            Err(e) => Some(Err(e)),
        }
    }
}

// --- RationalExponentStrategy: Handles equations like x^(p/q) = rhs ---
// Converts x^(p/q) = rhs to x^p = rhs^q to avoid infinite loops with fractional exponents

pub struct RationalExponentStrategy;

impl SolverStrategy for RationalExponentStrategy {
    fn name(&self) -> &str {
        "Rational Exponent"
    }

    fn apply(
        &self,
        eq: &Equation,
        var: &str,
        simplifier: &mut Simplifier,
        _opts: &SolverOptions,
        ctx: &SolveCtx,
    ) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        let rewrite = build_rational_exponent_rewrite(eq, var, simplifier)?;
        let include_item = simplifier.collect_steps();
        let solved = solve_rational_exponent_rewrite_pipeline_with_item_with(
            rewrite,
            var,
            include_item,
            |equation, var| solve_with_ctx(equation, var, simplifier, ctx),
            |item| medium_step(item.description, item.equation),
            |_| true,
        );

        Some(match solved {
            Ok(solved) => Ok((solved.solution_set, solved.steps)),
            Err(e) => Err(e),
        })
    }
}
