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
    derive_isolation_strategy_routing, execute_collect_terms_rewrite_with_runtime,
    execute_rational_exponent_rewrite_with_runtime_for_var,
    solve_collect_terms_rewrite_pipeline_with_item_runtime,
    solve_isolation_strategy_routing_with_runtime,
    solve_rational_exponent_rewrite_pipeline_with_item_with, CollectTermsRewriteRuntime,
    IsolationStrategyRuntime, StrategyExecutionItem, StrategyKernelRuntime,
};
use cas_solver_core::unwrap_plan::{
    route_unwrap_entry_with_item, solve_unwrap_execution_pipeline_with_item, UnwrapEntryRouting,
    UnwrapExecutionPlan, UnwrapExecutionRuntime,
};

pub struct IsolationStrategy;

struct EngineStrategyKernelRuntime<'a> {
    simplifier: &'a mut Simplifier,
}

impl StrategyKernelRuntime for EngineStrategyKernelRuntime<'_> {
    fn context(&mut self) -> &mut cas_ast::Context {
        &mut self.simplifier.context
    }

    fn simplify_expr(&mut self, expr: ExprId) -> ExprId {
        let (simplified, _) = self.simplifier.simplify(expr);
        simplified
    }

    fn render_expr(&mut self, expr: ExprId) -> String {
        solver_render_expr(&self.simplifier.context, expr)
    }
}

struct EngineIsolationRuntime<'a> {
    simplifier: &'a mut Simplifier,
    opts: SolverOptions,
    ctx: &'a SolveCtx,
}

impl IsolationStrategyRuntime<CasError, SolveStep> for EngineIsolationRuntime<'_> {
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
            self.ctx,
        )
    }

    fn map_swap_item_to_step(
        &mut self,
        item: cas_solver_core::solve_outcome::TermIsolationRewriteExecutionItem,
    ) -> SolveStep {
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
        let mut runtime = EngineIsolationRuntime {
            simplifier,
            opts: *opts,
            ctx,
        };
        solve_isolation_strategy_routing_with_runtime(routing, eq, var, include_item, &mut runtime)
    }

    // Note: We use the default should_verify() = true here.
    // Selective verification in solve() handles symbolic solutions.
}

pub struct UnwrapStrategy;

struct EngineUnwrapRuntime<'a> {
    simplifier: &'a mut Simplifier,
    ctx: &'a SolveCtx,
}

impl UnwrapExecutionRuntime<CasError, SolveStep> for EngineUnwrapRuntime<'_> {
    fn note_assumption(&mut self, record: cas_solver_core::unwrap_plan::LogLinearAssumptionRecord) {
        let event = crate::assumptions::AssumptionEvent::from_log_assumption(
            record.assumption,
            &self.simplifier.context,
            record.base,
            record.other_side,
        );
        crate::solver::note_assumption(event);
    }

    fn solve_equation(
        &mut self,
        equation: &Equation,
        var: &str,
    ) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
        solve_with_ctx(equation, var, self.simplifier, self.ctx)
    }

    fn map_item_to_step(
        &mut self,
        item: cas_solver_core::unwrap_plan::UnwrapExecutionItem,
    ) -> SolveStep {
        medium_step(item.description, item.equation)
    }
}

fn run_unwrap_execution(
    execution: UnwrapExecutionPlan,
    other_side: ExprId,
    var: &str,
    simplifier: &mut Simplifier,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let include_item = simplifier.collect_steps();
    let mut runtime = EngineUnwrapRuntime { simplifier, ctx };
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

struct EngineCollectTermsStrategyRuntime<'a> {
    simplifier: &'a mut Simplifier,
    ctx: &'a SolveCtx,
}

impl CollectTermsRewriteRuntime<CasError, SolveStep> for EngineCollectTermsStrategyRuntime<'_> {
    fn solve_rewritten(
        &mut self,
        equation: &Equation,
        var: &str,
    ) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
        solve_with_ctx(equation, var, self.simplifier, self.ctx)
    }

    fn map_item_to_step(&mut self, item: StrategyExecutionItem) -> SolveStep {
        medium_step(item.description, item.equation)
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
        _opts: &SolverOptions,
        ctx: &SolveCtx,
    ) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        let rewrite = {
            let mut runtime = EngineStrategyKernelRuntime { simplifier };
            execute_collect_terms_rewrite_with_runtime(&mut runtime, eq, var)?
        };
        let include_item = simplifier.collect_steps();
        let mut runtime = EngineCollectTermsStrategyRuntime { simplifier, ctx };
        let solved = solve_collect_terms_rewrite_pipeline_with_item_runtime(
            rewrite,
            var,
            include_item,
            &mut runtime,
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
        let rewrite = {
            let mut runtime = EngineStrategyKernelRuntime { simplifier };
            execute_rational_exponent_rewrite_with_runtime_for_var(&mut runtime, eq, var)?
        };
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
