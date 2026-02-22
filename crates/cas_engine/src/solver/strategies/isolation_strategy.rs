use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::isolation::isolate;
use crate::solver::solve_core::solve_with_ctx;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{SolveCtx, SolveStep, SolverOptions};
use cas_ast::{Equation, ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_utils::contains_var;
use cas_solver_core::solve_outcome::{
    plan_swap_sides_step, resolve_single_side_exponential_terminal_with_step,
};
use cas_solver_core::strategy_kernels::{
    build_collect_terms_step_with, build_rational_exponent_step, derive_collect_terms_kernel,
    derive_rational_exponent_kernel,
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
            let swapped = plan.equation;
            let mut steps = Vec::new();
            if simplifier.collect_steps() {
                let didactic_step = plan.step;
                steps.push(SolveStep {
                    description: didactic_step.description,
                    equation_after: didactic_step.equation_after,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            // V2.0: Pass opts through to propagate budget
            match isolate(
                swapped.lhs,
                swapped.rhs,
                swapped.op,
                var,
                simplifier,
                *opts,
                ctx,
            ) {
                Ok((set, mut iso_steps)) => {
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

        if let Some((solutions, didactic_step)) = resolve_single_side_exponential_terminal_with_step(
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
                    description: didactic_step.description,
                    equation_after: didactic_step.equation_after,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            return Some(Ok((solutions, steps)));
        }

        // Helper to invert
        let mut invert =
            |target: ExprId,
             other: ExprId,
             op: RelOp,
             is_lhs: bool|
             -> Option<(cas_solver_core::unwrap_plan::UnwrapExecutionPlan, ExprId)> {
                let rewrite_plan = cas_solver_core::unwrap_plan::plan_unwrap_rewrite(
                    &mut simplifier.context,
                    target,
                    other,
                    var,
                    op,
                    is_lhs,
                    |core_ctx, base, other_side| {
                        classify_log_solve(core_ctx, base, other_side, opts, &ctx.domain_env)
                    },
                )?;
                let execution = cas_solver_core::unwrap_plan::build_unwrap_execution_plan_with(
                    rewrite_plan,
                    |id| {
                        format!(
                            "{}",
                            cas_formatter::DisplayExpr {
                                context: &simplifier.context,
                                id
                            }
                        )
                    },
                );
                Some((execution, other))
            };

        // Try LHS
        if lhs_has {
            if let Some((execution, other_side)) = invert(eq.lhs, eq.rhs, eq.op.clone(), true) {
                if let Some(base) = execution.log_linear_base {
                    for assumption in execution.assumptions.iter().copied() {
                        let event = crate::assumptions::AssumptionEvent::from_log_assumption(
                            assumption,
                            &simplifier.context,
                            base,
                            other_side,
                        );
                        crate::solver::note_assumption(event);
                    }
                }
                let mut steps = Vec::new();
                if simplifier.collect_steps() {
                    steps.push(SolveStep {
                        description: execution.description.clone(),
                        equation_after: execution.equation.clone(),
                        importance: crate::step::ImportanceLevel::Medium,
                        substeps: vec![],
                    });
                }
                match solve_with_ctx(&execution.equation, var, simplifier, ctx) {
                    Ok((set, mut sub_steps)) => {
                        steps.append(&mut sub_steps);
                        return Some(Ok((set, steps)));
                    }
                    Err(e) => return Some(Err(e)),
                }
            }
        }

        // Try RHS
        if rhs_has {
            if let Some((execution, other_side)) = invert(eq.rhs, eq.lhs, eq.op.clone(), false) {
                if let Some(base) = execution.log_linear_base {
                    for assumption in execution.assumptions.iter().copied() {
                        let event = crate::assumptions::AssumptionEvent::from_log_assumption(
                            assumption,
                            &simplifier.context,
                            base,
                            other_side,
                        );
                        crate::solver::note_assumption(event);
                    }
                }
                let mut steps = Vec::new();
                if simplifier.collect_steps() {
                    steps.push(SolveStep {
                        description: execution.description.clone(),
                        equation_after: execution.equation.clone(),
                        importance: crate::step::ImportanceLevel::Medium,
                        substeps: vec![],
                    });
                }
                match solve_with_ctx(&execution.equation, var, simplifier, ctx) {
                    Ok((set, mut sub_steps)) => {
                        steps.append(&mut sub_steps);
                        return Some(Ok((set, steps)));
                    }
                    Err(e) => return Some(Err(e)),
                }
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
        let kernel = derive_collect_terms_kernel(&mut simplifier.context, eq, var)?;
        let mut steps = Vec::new();

        // Strategy: Subtract RHS from both sides to move everything to LHS
        // ax + b = cx + d  ->  ax + b - (cx + d) = cx + d - (cx + d)
        //                  ->  ax - cx + b - d = 0

        let rewritten = kernel.rewritten;

        // Simplify both sides
        let (simp_lhs, _) = simplifier.simplify(rewritten.lhs);
        let (simp_rhs, _) = simplifier.simplify(rewritten.rhs);

        if simplifier.collect_steps() {
            let collect_step = build_collect_terms_step_with(
                Equation {
                    lhs: simp_lhs,
                    rhs: simp_rhs,
                    op: eq.op.clone(),
                },
                eq.rhs,
                |id| {
                    format!(
                        "{}",
                        cas_formatter::DisplayExpr {
                            context: &simplifier.context,
                            id
                        }
                    )
                },
            );
            steps.push(SolveStep {
                description: collect_step.description,
                equation_after: collect_step.equation_after,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }

        // Now recursively solve the simplified equation
        // This should now have variable only on one side
        let new_eq = Equation {
            lhs: simp_lhs,
            rhs: simp_rhs,
            op: eq.op.clone(),
        };
        match solve_with_ctx(&new_eq, var, simplifier, ctx) {
            Ok((set, mut solve_steps)) => {
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

        let kernel =
            derive_rational_exponent_kernel(&mut simplifier.context, eq, var, lhs_has, rhs_has)?;

        let mut steps = Vec::new();

        // Simplify both sides
        let (sim_lhs, _) = simplifier.simplify(kernel.rewritten.lhs);
        let (sim_rhs, _) = simplifier.simplify(kernel.rewritten.rhs);

        let new_eq = Equation {
            lhs: sim_lhs,
            rhs: sim_rhs,
            op: RelOp::Eq,
        };

        if simplifier.collect_steps() {
            let rational_step = build_rational_exponent_step(kernel.q, new_eq.clone());
            steps.push(SolveStep {
                description: rational_step.description,
                equation_after: rational_step.equation_after,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }

        // Recursively solve the new equation
        match solve_with_ctx(&new_eq, var, simplifier, ctx) {
            Ok((set, mut sub_steps)) => {
                steps.append(&mut sub_steps);

                // For even q, we need to verify solutions (could introduce extraneous)
                // The main solve() already verifies against original equation
                Some(Ok((set, steps)))
            }
            Err(e) => Some(Err(e)),
        }
    }
}
