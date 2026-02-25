use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::isolation::isolate;
use crate::solver::solve_core::solve_with_ctx_and_options;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{
    medium_step, render_expr as solver_render_expr, SolveCtx, SolveStep, SolverOptions,
};
use cas_ast::{Equation, ExprId, SolutionSet};
use cas_solver_core::isolation_utils::contains_var;
use cas_solver_core::solve_outcome::{
    TermIsolationExecutionItem, TermIsolationRewriteExecutionItem,
};
use cas_solver_core::strategy_kernels::{
    build_collect_terms_execution_with, build_rational_exponent_execution,
    collect_collect_terms_execution_items, collect_rational_exponent_execution_items,
    derive_collect_terms_kernel, derive_isolation_strategy_routing,
    derive_rational_exponent_kernel, solve_collect_terms_rewrite_pipeline_with_item,
    solve_isolation_strategy_routing_with, solve_rational_exponent_rewrite_pipeline_with_item_with,
    CollectTermsSolvedRewrite, RationalExponentSolvedRewrite,
};
use cas_solver_core::unwrap_plan::{
    route_unwrap_entry_with_item, solve_unwrap_execution_pipeline_with_item,
    LogLinearAssumptionRecord, UnwrapEntryRouting, UnwrapExecutionItem, UnwrapExecutionPlan,
};

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
            |equation, solve_var| {
                isolate(
                    equation.lhs,
                    equation.rhs,
                    equation.op.clone(),
                    solve_var,
                    simplifier,
                    *opts,
                    ctx,
                )
            },
            |item: TermIsolationRewriteExecutionItem| medium_step(item.description, item.equation),
            |missing_var| CasError::VariableNotFound(missing_var.to_string()),
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
    opts: SolverOptions,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let include_item = simplifier.collect_steps();
    let runtime_cell = std::cell::RefCell::new(simplifier);
    let solved = solve_unwrap_execution_pipeline_with_item(
        execution,
        other_side,
        var,
        include_item,
        |record: LogLinearAssumptionRecord| {
            let simplifier_ref = runtime_cell.borrow();
            let event = crate::assumptions::AssumptionEvent::from_log_assumption(
                record.assumption,
                &simplifier_ref.context,
                record.base,
                record.other_side,
            );
            ctx.note_assumption(event);
        },
        |equation: &Equation, solve_var: &str| {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            solve_with_ctx_and_options(equation, solve_var, *simplifier_ref, opts, ctx)
        },
        |item: UnwrapExecutionItem| medium_step(item.description, item.equation),
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
        let mode = opts.core_domain_mode();
        let wildcard_scope = opts.wildcard_scope();

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
                crate::solver::classify_log_solve(core_ctx, base, other_side, opts, &ctx.domain_env)
            },
            solver_render_expr,
            |item: TermIsolationExecutionItem| medium_step(item.description, item.equation),
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
                *opts,
                ctx,
            ),
        })
    }

    // Note: We use the default should_verify() = true here.
    // Selective verification in solve() handles symbolic solutions.
}

// --- Helper for CollectTermsStrategy (currently unused) ---
// --- CollectTermsStrategy: Handles linear equations with variables on both sides ---

pub struct CollectTermsStrategy;

impl SolverStrategy for CollectTermsStrategy {
    fn name(&self) -> &str {
        "Collect Terms"
    }

    fn apply(
        &self,
        eq: &Equation,
        var: &str,
        simplifier: &mut Simplifier,
        opts: &SolverOptions,
        ctx: &SolveCtx,
    ) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        let include_item = simplifier.collect_steps();
        let kernel = derive_collect_terms_kernel(&mut simplifier.context, eq, var)?;
        let simp_lhs = simplifier.simplify(kernel.rewritten.lhs).0;
        let simp_rhs = simplifier.simplify(kernel.rewritten.rhs).0;
        let execution =
            build_collect_terms_execution_with(simp_lhs, simp_rhs, eq.op.clone(), eq.rhs, |expr| {
                solver_render_expr(&simplifier.context, expr)
            });
        let items = collect_collect_terms_execution_items(&execution);
        let rewrite = CollectTermsSolvedRewrite {
            equation: execution.equation,
            items,
        };
        let solved = solve_collect_terms_rewrite_pipeline_with_item(
            rewrite,
            var,
            include_item,
            |equation, solve_var| {
                solve_with_ctx_and_options(equation, solve_var, simplifier, *opts, ctx)
            },
            |item| medium_step(item.description, item.equation),
        );

        Some(match solved {
            Ok(solved) => Ok((solved.solution_set, solved.steps)),
            Err(e) => Err(e),
        })
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
        opts: &SolverOptions,
        ctx: &SolveCtx,
    ) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        let include_item = simplifier.collect_steps();
        let lhs_has_var = contains_var(&simplifier.context, eq.lhs, var);
        let rhs_has_var = contains_var(&simplifier.context, eq.rhs, var);
        let kernel = derive_rational_exponent_kernel(
            &mut simplifier.context,
            eq,
            var,
            lhs_has_var,
            rhs_has_var,
        )?;
        let sim_lhs = simplifier.simplify(kernel.rewritten.lhs).0;
        let sim_rhs = simplifier.simplify(kernel.rewritten.rhs).0;
        let execution = build_rational_exponent_execution(kernel.q, sim_lhs, sim_rhs);
        let items = collect_rational_exponent_execution_items(&execution);
        let rewrite = RationalExponentSolvedRewrite {
            equation: execution.equation,
            items,
        };
        let solved = solve_rational_exponent_rewrite_pipeline_with_item_with(
            rewrite,
            var,
            include_item,
            |equation, solve_var| {
                solve_with_ctx_and_options(equation, solve_var, simplifier, *opts, ctx)
            },
            |item| medium_step(item.description, item.equation),
            |_solution| true,
        );

        Some(match solved {
            Ok(solved) => Ok((solved.solution_set, solved.steps)),
            Err(e) => Err(e),
        })
    }
}
