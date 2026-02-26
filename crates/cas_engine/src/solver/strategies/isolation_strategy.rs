use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::isolation::isolate;
use crate::solver::solve_core::solve_with_ctx_and_options;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{
    medium_step, render_expr as solver_render_expr, SolveCtx, SolveStep, SolverOptions,
};
use cas_ast::{Equation, SolutionSet};
use cas_solver_core::solve_outcome::{
    TermIsolationExecutionItem, TermIsolationRewriteExecutionItem,
};
use cas_solver_core::strategy_kernels::{
    derive_isolation_strategy_routing, materialize_collect_terms_rewrite_from_kernel_with,
    materialize_rational_exponent_rewrite_from_kernel_with,
    solve_collect_terms_rewrite_pipeline_with_item, solve_isolation_strategy_routing_with,
    solve_rational_exponent_rewrite_pipeline_with_item_with,
};
use cas_solver_core::unwrap_plan::{
    collect_log_linear_assumption_records, first_unwrap_execution_item,
    route_unwrap_entry_with_item, unwrap_rewritten_equation, UnwrapEntryRouting,
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
        let selected = routed?;
        match selected {
            UnwrapEntryRouting::Terminal(terminal) => {
                Some(Ok((terminal.solution_set, terminal.steps)))
            }
            UnwrapEntryRouting::Execution(selected_execution) => {
                let assumption_records = collect_log_linear_assumption_records(
                    &selected_execution.execution,
                    selected_execution.other_side,
                );
                for record in assumption_records {
                    let event = crate::solver::assumption_event_from_log_assumption_targets(
                        &simplifier.context,
                        record.assumption,
                        record.base,
                        record.other_side,
                    );
                    ctx.note_assumption(event);
                }

                let rewritten_equation = unwrap_rewritten_equation(&selected_execution.execution);
                let mut steps = Vec::new();
                if include_item {
                    if let Some(item) = first_unwrap_execution_item(&selected_execution.execution) {
                        steps.push(medium_step(item.description, item.equation));
                    }
                }

                Some(
                    solve_with_ctx_and_options(&rewritten_equation, var, simplifier, *opts, ctx)
                        .map(|(solution_set, mut sub_steps)| {
                            steps.append(&mut sub_steps);
                            (solution_set, steps)
                        }),
                )
            }
        }
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
        let rhs_desc = solver_render_expr(&simplifier.context, eq.rhs);
        let collect_kernel = cas_solver_core::strategy_kernels::derive_collect_terms_kernel(
            &mut simplifier.context,
            eq,
            var,
        );
        let kernel = collect_kernel?;
        let rewrite = materialize_collect_terms_rewrite_from_kernel_with(
            kernel,
            eq.op.clone(),
            eq.rhs,
            |expr| simplifier.simplify(expr).0,
            move |_| rhs_desc.clone(),
        );
        Some(
            solve_collect_terms_rewrite_pipeline_with_item(
                rewrite,
                var,
                include_item,
                |equation, solve_var| {
                    solve_with_ctx_and_options(equation, solve_var, simplifier, *opts, ctx)
                },
                |item| medium_step(item.description, item.equation),
            )
            .map(|payload| (payload.solution_set, payload.steps)),
        )
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
        let rational_kernel =
            cas_solver_core::strategy_kernels::derive_rational_exponent_kernel_for_var(
                &mut simplifier.context,
                eq,
                var,
            );
        let kernel = rational_kernel?;
        let rewrite = materialize_rational_exponent_rewrite_from_kernel_with(kernel, |expr| {
            simplifier.simplify(expr).0
        });
        Some(
            solve_rational_exponent_rewrite_pipeline_with_item_with(
                rewrite,
                var,
                include_item,
                |equation, solve_var| {
                    solve_with_ctx_and_options(equation, solve_var, simplifier, *opts, ctx)
                },
                |item| medium_step(item.description, item.equation),
                |_solution| true,
            )
            .map(|payload| (payload.solution_set, payload.steps)),
        )
    }
}
