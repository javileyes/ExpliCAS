use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::isolation::isolate;
use crate::solver::solve_core::solve_with_ctx;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{SolveCtx, SolveStep, SolverOptions};
use cas_ast::{Equation, ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_utils::contains_var;
use cas_solver_core::solve_outcome::{
    first_term_isolation_rewrite_execution_item, plan_swap_sides_step,
    resolve_single_side_exponential_terminal_with_item, solve_term_isolation_rewrite_with,
};
use cas_solver_core::strategy_kernels::{
    execute_collect_terms_rewrite_with_runtime, execute_rational_exponent_rewrite_with_runtime,
    solve_collect_terms_rewrite_with_item, solve_rational_exponent_rewrite_with_item,
    StrategyKernelRuntime,
};
use cas_solver_core::unwrap_plan::{
    plan_unwrap_execution_with, solve_unwrap_execution_with_item, UnwrapExecutionPlan,
    UnwrapExecutionRequest,
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
        format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &self.simplifier.context,
                id: expr
            }
        )
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
        // Isolation strategy expects variable on LHS.
        // The main solve loop handles swapping, but we should check here or just assume?
        // Let's check and swap if needed, or just rely on isolate to handle it?
        // isolate() assumes we are isolating FROM lhs.

        // If var is on RHS and not LHS, we should swap.
        // If var is on both, isolation might fail or we need to collect first.

        let lhs_has = contains_var(&simplifier.context, eq.lhs, var);
        let rhs_has = contains_var(&simplifier.context, eq.rhs, var);

        if !lhs_has && !rhs_has {
            return Some(Err(CasError::VariableNotFound(var.to_string())));
        }

        if lhs_has && rhs_has {
            // Isolation cannot handle var on both sides directly without collection
            return None; // Or error? Strategy doesn't apply if not isolated.
        }

        if !lhs_has && rhs_has {
            // Swap
            let plan = plan_swap_sides_step(eq);
            let solved_swap = solve_term_isolation_rewrite_with(plan, |equation| {
                isolate(
                    equation.lhs,
                    equation.rhs,
                    equation.op.clone(),
                    var,
                    simplifier,
                    *opts,
                    ctx,
                )
            });
            match solved_swap {
                Ok(solved_swap) => {
                    let mut steps = Vec::new();
                    if simplifier.collect_steps() {
                        if let Some(item) =
                            first_term_isolation_rewrite_execution_item(&solved_swap.rewrite)
                        {
                            steps.push(SolveStep {
                                description: item.description().to_string(),
                                equation_after: item.equation,
                                importance: crate::step::ImportanceLevel::Medium,
                                substeps: vec![],
                            });
                        }
                    }
                    let (set, mut iso_steps) = solved_swap.solved;
                    steps.append(&mut iso_steps);
                    return Some(Ok((set, steps)));
                }
                Err(e) => return Some(Err(e)),
            }
        }

        // LHS has var
        // V2.0: Pass opts through to propagate budget
        match isolate(eq.lhs, eq.rhs, eq.op.clone(), var, simplifier, *opts, ctx) {
            Ok((set, steps)) => Some(Ok((set, steps))),
            Err(e) => Some(Err(e)),
        }
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
    let mut assumption_records = Vec::new();
    let mut step_item = None;
    let solved = solve_unwrap_execution_with_item(
        execution,
        other_side,
        |record| {
            assumption_records.push(record);
        },
        |item, rewritten_eq| {
            step_item = item;
            solve_with_ctx(rewritten_eq, var, simplifier, ctx)
        },
    )?;

    for record in assumption_records {
        let event = crate::assumptions::AssumptionEvent::from_log_assumption(
            record.assumption,
            &simplifier.context,
            record.base,
            record.other_side,
        );
        crate::solver::note_assumption(event);
    }

    let mut steps = Vec::new();
    if simplifier.collect_steps() {
        if let Some(item) = step_item {
            steps.push(SolveStep {
                description: item.description().to_string(),
                equation_after: item.equation,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
    }

    let (set, mut sub_steps) = solved.solved;
    steps.append(&mut sub_steps);
    Ok((set, steps))
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
        // Try to unwrap functions on LHS or RHS to expose the variable or transform the equation.
        // This is useful when var is on both sides, e.g. sqrt(2x+3) = x.

        let lhs_has = contains_var(&simplifier.context, eq.lhs, var);
        let rhs_has = contains_var(&simplifier.context, eq.rhs, var);

        // Only apply if var is on both sides?
        // If var is only on one side, IsolationStrategy handles it.
        // But IsolationStrategy might be later in the list.
        // Let's apply if top-level is a function/pow that we can invert.

        if !lhs_has && !rhs_has {
            return None;
        }

        // EARLY CHECK: Handle exponential NeedsComplex + Wildcard -> Residual
        // This must be before the closure to be able to return SolutionSet::Residual
        use crate::solver::domain_guards::classify_log_solve;
        let mode = crate::solver::domain_guards::to_core_domain_mode(opts.domain_mode);
        let wildcard_scope = opts.assume_scope == crate::semantics::AssumeScope::Wildcard;

        if let Some((solutions, terminal_item)) = resolve_single_side_exponential_terminal_with_item(
            &mut simplifier.context,
            eq.lhs,
            eq.rhs,
            var,
            lhs_has,
            rhs_has,
            mode,
            wildcard_scope,
            " - use 'semantics preset complex'",
            eq.clone(),
            |core_ctx, base, other_side| {
                classify_log_solve(core_ctx, base, other_side, opts, &ctx.domain_env)
            },
        ) {
            let mut steps = Vec::new();
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: terminal_item.description().to_string(),
                    equation_after: terminal_item.equation,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            return Some(Ok((solutions, steps)));
        }

        // Helper to invert
        let mut invert = |target: ExprId, other: ExprId, op: RelOp, is_lhs: bool| {
            plan_unwrap_execution_with(
                &mut simplifier.context,
                UnwrapExecutionRequest {
                    target,
                    other,
                    var,
                    op,
                    is_lhs,
                },
                |core_ctx, base, other_side| {
                    classify_log_solve(core_ctx, base, other_side, opts, &ctx.domain_env)
                },
                |core_ctx, id| {
                    format!(
                        "{}",
                        cas_formatter::DisplayExpr {
                            context: core_ctx,
                            id
                        }
                    )
                },
            )
        };

        // Try LHS
        if lhs_has {
            if let Some(execution) = invert(eq.lhs, eq.rhs, eq.op.clone(), true) {
                return Some(run_unwrap_execution(
                    execution, eq.rhs, var, simplifier, ctx,
                ));
            }
        }

        // Try RHS
        if rhs_has {
            if let Some(execution) = invert(eq.rhs, eq.lhs, eq.op.clone(), false) {
                return Some(run_unwrap_execution(
                    execution, eq.lhs, var, simplifier, ctx,
                ));
            }
        }

        None
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
        let mut rewrite_item = None;
        let solved = {
            let rewrite = {
                let mut runtime = EngineStrategyKernelRuntime { simplifier };
                execute_collect_terms_rewrite_with_runtime(&mut runtime, eq, var)?
            };
            solve_collect_terms_rewrite_with_item(rewrite, |item, new_eq| {
                rewrite_item = item;
                solve_with_ctx(new_eq, var, simplifier, ctx)
            })
        };

        match solved {
            Ok(solved) => {
                let mut steps = Vec::new();
                if simplifier.collect_steps() {
                    if let Some(item) = rewrite_item {
                        steps.push(SolveStep {
                            description: item.description().to_string(),
                            equation_after: item.equation,
                            importance: crate::step::ImportanceLevel::Medium,
                            substeps: vec![],
                        });
                    }
                }

                let (set, mut solve_steps) = solved.solved;
                steps.append(&mut solve_steps);
                Some(Ok((set, steps)))
            }
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
        let lhs_has = contains_var(&simplifier.context, eq.lhs, var);
        let rhs_has = contains_var(&simplifier.context, eq.rhs, var);
        let mut rewrite_item = None;
        let solved = {
            let rewrite = {
                let mut runtime = EngineStrategyKernelRuntime { simplifier };
                execute_rational_exponent_rewrite_with_runtime(
                    &mut runtime,
                    eq,
                    var,
                    lhs_has,
                    rhs_has,
                )?
            };
            solve_rational_exponent_rewrite_with_item(rewrite, |item, new_eq| {
                rewrite_item = item;
                solve_with_ctx(new_eq, var, simplifier, ctx)
            })
        };

        match solved {
            Ok(solved) => {
                let mut steps = Vec::new();
                if simplifier.collect_steps() {
                    if let Some(item) = rewrite_item {
                        steps.push(SolveStep {
                            description: item.description().to_string(),
                            equation_after: item.equation,
                            importance: crate::step::ImportanceLevel::Medium,
                            substeps: vec![],
                        });
                    }
                }

                let (set, mut sub_steps) = solved.solved;
                steps.append(&mut sub_steps);

                // For even q, we need to verify solutions (could introduce extraneous)
                // The main solve() already verifies against original equation
                Some(Ok((set, steps)))
            }
            Err(e) => Some(Err(e)),
        }
    }
}
