use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::isolation::isolate;
use crate::solver::solve_core::solve_with_ctx_and_options;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{
    medium_step, render_expr as solver_render_expr, SolveCtx, SolveStep, SolverOptions,
};
use cas_ast::{Equation, SolutionSet};
use cas_solver_core::isolation_strategy::{
    execute_collect_terms_strategy_with_default_kernel_with_state,
    execute_isolation_strategy_with_default_routing_with_state,
    execute_rational_exponent_strategy_with_default_kernel_with_state,
    execute_unwrap_strategy_with_default_route_with_state,
};
use cas_solver_core::solve_outcome::{
    TermIsolationExecutionItem, TermIsolationRewriteExecutionItem,
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
        let include_item = simplifier.collect_steps();
        execute_isolation_strategy_with_default_routing_with_state(
            simplifier,
            eq,
            var,
            include_item,
            |simplifier| &simplifier.context,
            |simplifier, equation, solve_var| {
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
        execute_unwrap_strategy_with_default_route_with_state(
            simplifier,
            eq,
            var,
            include_item,
            |simplifier| &mut simplifier.context,
            mode,
            wildcard_scope,
            " - use 'semantics preset complex'",
            |core_ctx, base, other_side| {
                crate::solver::classify_log_solve(core_ctx, base, other_side, opts, &ctx.domain_env)
            },
            solver_render_expr,
            |item: TermIsolationExecutionItem| medium_step(item.description, item.equation),
            |simplifier, record| {
                let event = crate::solver::assumption_event_from_log_assumption_targets(
                    &simplifier.context,
                    record.assumption,
                    record.base,
                    record.other_side,
                );
                ctx.note_assumption(event);
            },
            |simplifier, equation, solve_var| {
                solve_with_ctx_and_options(equation, solve_var, simplifier, *opts, ctx)
            },
            |item| medium_step(item.description, item.equation),
        )
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
        execute_collect_terms_strategy_with_default_kernel_with_state(
            simplifier,
            eq,
            var,
            include_item,
            |simplifier| &mut simplifier.context,
            |simplifier, expr| simplifier.simplify(expr).0,
            |simplifier, rhs| solver_render_expr(&simplifier.context, rhs),
            |simplifier, equation, solve_var| {
                solve_with_ctx_and_options(equation, solve_var, simplifier, *opts, ctx)
            },
            |item| medium_step(item.description, item.equation),
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
        execute_rational_exponent_strategy_with_default_kernel_with_state(
            simplifier,
            eq,
            var,
            include_item,
            |simplifier| &mut simplifier.context,
            |simplifier, expr| simplifier.simplify(expr).0,
            |simplifier, equation, solve_var| {
                solve_with_ctx_and_options(equation, solve_var, simplifier, *opts, ctx)
            },
            |item| medium_step(item.description, item.equation),
            |_simplifier, _solution| true,
        )
    }
}
