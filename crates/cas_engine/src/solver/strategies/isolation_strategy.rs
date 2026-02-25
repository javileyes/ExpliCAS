use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::isolation::isolate;
use crate::solver::solve_core::solve_with_ctx_and_options;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{
    medium_step, render_expr as solver_render_expr, SolveCtx, SolveStep, SolverOptions,
};
use cas_ast::{Equation, ExprId, SolutionSet};
use cas_solver_core::solve_outcome::{
    TermIsolationExecutionItem, TermIsolationRewriteExecutionItem,
};
use cas_solver_core::strategy_kernels::{
    derive_isolation_strategy_routing,
    solve_collect_terms_rewrite_pipeline_with_item_runtime_for_var,
    solve_isolation_strategy_routing_with_runtime,
    solve_rational_exponent_rewrite_pipeline_with_item_runtime_for_var, CollectTermsRewriteRuntime,
    IsolationStrategyRuntime, RationalExponentRewriteRuntime, StrategyExecutionItem,
    StrategyKernelRuntime,
};
use cas_solver_core::unwrap_plan::{
    route_unwrap_entry_with_item_runtime, solve_unwrap_execution_pipeline_with_item,
    LogLinearAssumptionRecord, UnwrapEntryRouting, UnwrapEntryRuntime, UnwrapExecutionItem,
    UnwrapExecutionPlan, UnwrapExecutionRuntime, UnwrapPlanRuntime,
};

pub struct IsolationStrategy;

struct IsolationRoutingRuntime<'a, 'ctx> {
    simplifier: &'a mut Simplifier,
    opts: SolverOptions,
    solve_ctx: &'ctx SolveCtx,
}

impl IsolationStrategyRuntime<CasError, SolveStep> for IsolationRoutingRuntime<'_, '_> {
    fn solve_equation(
        &mut self,
        equation: &Equation,
        var: &str,
    ) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
        isolate(
            equation.lhs,
            equation.rhs,
            equation.op.clone(),
            var,
            self.simplifier,
            self.opts,
            self.solve_ctx,
        )
    }

    fn map_swap_item_to_step(&mut self, item: TermIsolationRewriteExecutionItem) -> SolveStep {
        medium_step(item.description, item.equation)
    }

    fn variable_not_found_error(&mut self, var: &str) -> CasError {
        CasError::VariableNotFound(var.to_string())
    }
}

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
        let mut runtime = IsolationRoutingRuntime {
            simplifier,
            opts: *opts,
            solve_ctx: ctx,
        };
        solve_isolation_strategy_routing_with_runtime(routing, eq, var, include_item, &mut runtime)
    }

    // Note: We use the default should_verify() = true here.
    // Selective verification in solve() handles symbolic solutions.
}

pub struct UnwrapStrategy;

struct UnwrapEntryRuntimeAdapter<'a, 'ctx> {
    opts: &'a SolverOptions,
    solve_ctx: &'ctx SolveCtx,
}

impl UnwrapPlanRuntime for UnwrapEntryRuntimeAdapter<'_, '_> {
    fn classify_log_solve(
        &mut self,
        ctx: &cas_ast::Context,
        base: ExprId,
        other_side: ExprId,
    ) -> cas_solver_core::log_domain::LogSolveDecision {
        crate::solver::domain_guards::classify_log_solve(
            ctx,
            base,
            other_side,
            self.opts.value_domain,
            self.opts.domain_mode,
            self.solve_ctx.domain_env.has_positive(base),
            self.solve_ctx.domain_env.has_positive(other_side),
        )
    }

    fn render_expr(&mut self, ctx: &cas_ast::Context, expr: ExprId) -> String {
        solver_render_expr(ctx, expr)
    }
}

impl UnwrapEntryRuntime<SolveStep> for UnwrapEntryRuntimeAdapter<'_, '_> {
    fn map_terminal_item_to_step(&mut self, item: TermIsolationExecutionItem) -> SolveStep {
        medium_step(item.description, item.equation)
    }
}

struct UnwrapExecutionRuntimeAdapter<'a, 'ctx> {
    simplifier: &'a mut Simplifier,
    opts: SolverOptions,
    solve_ctx: &'ctx SolveCtx,
}

impl UnwrapExecutionRuntime<CasError, SolveStep> for UnwrapExecutionRuntimeAdapter<'_, '_> {
    fn note_assumption(&mut self, record: LogLinearAssumptionRecord) {
        let event = crate::assumptions::AssumptionEvent::from_log_assumption(
            record.assumption,
            &self.simplifier.context,
            record.base,
            record.other_side,
        );
        self.solve_ctx.note_assumption(event);
    }

    fn solve_equation(
        &mut self,
        equation: &Equation,
        var: &str,
    ) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
        solve_with_ctx_and_options(equation, var, self.simplifier, self.opts, self.solve_ctx)
    }

    fn map_item_to_step(&mut self, item: UnwrapExecutionItem) -> SolveStep {
        medium_step(item.description, item.equation)
    }
}

fn run_unwrap_execution(
    execution: UnwrapExecutionPlan,
    other_side: ExprId,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let include_item = simplifier.collect_steps();
    let mut runtime = UnwrapExecutionRuntimeAdapter {
        simplifier,
        opts,
        solve_ctx: ctx,
    };
    let solved = solve_unwrap_execution_pipeline_with_item(
        execution,
        other_side,
        var,
        include_item,
        &mut runtime,
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
        let wildcard_scope = opts.assume_scope == crate::semantics::AssumeScope::Wildcard;

        let include_item = simplifier.collect_steps();
        let mut runtime = UnwrapEntryRuntimeAdapter {
            opts,
            solve_ctx: ctx,
        };
        let routed = route_unwrap_entry_with_item_runtime(
            &mut simplifier.context,
            eq,
            var,
            mode,
            wildcard_scope,
            " - use 'semantics preset complex'",
            include_item,
            &mut runtime,
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

struct StrategyRewriteRuntime<'a, 'ctx> {
    simplifier: &'a mut Simplifier,
    opts: SolverOptions,
    solve_ctx: &'ctx SolveCtx,
}

impl StrategyKernelRuntime for StrategyRewriteRuntime<'_, '_> {
    fn context(&mut self) -> &mut cas_ast::Context {
        &mut self.simplifier.context
    }

    fn simplify_expr(&mut self, expr: ExprId) -> ExprId {
        self.simplifier.simplify(expr).0
    }

    fn render_expr(&mut self, expr: ExprId) -> String {
        solver_render_expr(&self.simplifier.context, expr)
    }
}

impl CollectTermsRewriteRuntime<CasError, SolveStep> for StrategyRewriteRuntime<'_, '_> {
    fn solve_rewritten(
        &mut self,
        equation: &Equation,
        var: &str,
    ) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
        solve_with_ctx_and_options(equation, var, self.simplifier, self.opts, self.solve_ctx)
    }

    fn map_item_to_step(&mut self, item: StrategyExecutionItem) -> SolveStep {
        medium_step(item.description, item.equation)
    }
}

impl RationalExponentRewriteRuntime<CasError, SolveStep> for StrategyRewriteRuntime<'_, '_> {
    fn solve_rewritten(
        &mut self,
        equation: &Equation,
        var: &str,
    ) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
        solve_with_ctx_and_options(equation, var, self.simplifier, self.opts, self.solve_ctx)
    }

    fn map_item_to_step(&mut self, item: StrategyExecutionItem) -> SolveStep {
        medium_step(item.description, item.equation)
    }

    fn verify_discrete_solution(&mut self, _solution: ExprId) -> bool {
        true
    }
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
        opts: &SolverOptions,
        ctx: &SolveCtx,
    ) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        let include_item = simplifier.collect_steps();
        let mut runtime = StrategyRewriteRuntime {
            simplifier,
            opts: *opts,
            solve_ctx: ctx,
        };
        let solved = solve_collect_terms_rewrite_pipeline_with_item_runtime_for_var(
            &mut runtime,
            eq,
            var,
            include_item,
        )?;

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
        opts: &SolverOptions,
        ctx: &SolveCtx,
    ) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        let include_item = simplifier.collect_steps();
        let mut runtime = StrategyRewriteRuntime {
            simplifier,
            opts: *opts,
            solve_ctx: ctx,
        };
        let solved = solve_rational_exponent_rewrite_pipeline_with_item_runtime_for_var(
            &mut runtime,
            eq,
            var,
            include_item,
        )?;

        Some(match solved {
            Ok(solved) => Ok((solved.solution_set, solved.steps)),
            Err(e) => Err(e),
        })
    }
}
