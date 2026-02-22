use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::isolation::isolate;
use crate::solver::solve_core::solve_with_ctx;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{SolveCtx, SolveDomainEnv, SolveStep, SolverOptions};
use cas_ast::{Equation, ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_utils::contains_var;
use cas_solver_core::solve_outcome::{
    eliminate_fractional_exponent_message, resolve_single_side_exponential_terminal_outcome,
    subtract_both_sides_message, take_log_base_message, terminal_outcome_message,
    SWAP_SIDES_TO_LHS_MESSAGE,
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
            let swapped = cas_solver_core::equation_rewrite::swap_sides_with_inequality_flip(eq);
            let mut steps = Vec::new();
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: SWAP_SIDES_TO_LHS_MESSAGE.to_string(),
                    equation_after: swapped.clone(),
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

/// Check if an exponential equation needs complex logarithm in Wildcard mode,
/// or if it has no real solutions (EmptySet).
/// Returns Some(Ok(Residual)) if Wildcard mode should return a residual.
/// Returns Some(Ok(Empty)) if no real solutions exist.
/// Returns Some(Err) if an error should be returned.
/// Returns None if this case doesn't apply (normal processing should continue).
fn check_exponential_needs_complex(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: &SolverOptions,
    env: &SolveDomainEnv,
    lhs_has: bool,
    rhs_has: bool,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    use crate::solver::domain_guards::classify_log_solve;
    let mode = crate::solver::domain_guards::to_core_domain_mode(opts.domain_mode);
    let wildcard_scope = opts.assume_scope == crate::semantics::AssumeScope::Wildcard;

    if let Some(outcome) = resolve_single_side_exponential_terminal_outcome(
        &mut simplifier.context,
        eq.lhs,
        eq.rhs,
        var,
        lhs_has,
        rhs_has,
        mode,
        wildcard_scope,
        |core_ctx, base, other_side| classify_log_solve(core_ctx, base, other_side, opts, env),
    ) {
        let mut steps = Vec::new();
        if simplifier.collect_steps() {
            steps.push(SolveStep {
                description: terminal_outcome_message(
                    &outcome,
                    " - use 'semantics preset complex'",
                ),
                equation_after: eq.clone(),
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
        return Some(Ok((outcome.solutions, steps)));
    }

    None
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
        if let Some(result) = check_exponential_needs_complex(
            eq,
            var,
            simplifier,
            opts,
            &ctx.domain_env,
            lhs_has,
            rhs_has,
        ) {
            return Some(result);
        }

        use crate::solver::domain_guards::classify_log_solve;

        // Helper to invert
        let mut invert = |target: ExprId,
                          other: ExprId,
                          op: RelOp,
                          is_lhs: bool|
         -> Option<(Equation, String)> {
            match cas_solver_core::unwrap_plan::plan_unwrap_rewrite(
                &mut simplifier.context,
                target,
                other,
                var,
                op,
                is_lhs,
                |core_ctx, base, other_side| {
                    classify_log_solve(core_ctx, base, other_side, opts, &ctx.domain_env)
                },
            )? {
                cas_solver_core::unwrap_plan::UnwrapRewritePlan::Unary {
                    equation,
                    description,
                } => Some((equation, description)),
                cas_solver_core::unwrap_plan::UnwrapRewritePlan::PowVariableBase {
                    equation,
                    exponent,
                } => {
                    let exponent_desc = format!(
                        "{}",
                        cas_formatter::DisplayExpr {
                            context: &simplifier.context,
                            id: exponent
                        }
                    );
                    Some((
                        equation,
                        cas_solver_core::unwrap_plan::pow_variable_base_unwrap_message(
                            &exponent_desc,
                        ),
                    ))
                }
                cas_solver_core::unwrap_plan::UnwrapRewritePlan::PowLogLinear {
                    equation,
                    base,
                    assumptions,
                } => {
                    for assumption in assumptions {
                        let event = crate::assumptions::AssumptionEvent::from_log_assumption(
                            assumption,
                            &simplifier.context,
                            base,
                            other,
                        );
                        crate::solver::note_assumption(event);
                    }
                    Some((equation, take_log_base_message("e")))
                }
            }
        };

        // Try LHS
        if lhs_has {
            if let Some((new_eq, desc)) = invert(eq.lhs, eq.rhs, eq.op.clone(), true) {
                let mut steps = Vec::new();
                if simplifier.collect_steps() {
                    steps.push(SolveStep {
                        description: desc,
                        equation_after: new_eq.clone(),
                        importance: crate::step::ImportanceLevel::Medium,
                        substeps: vec![],
                    });
                }
                match solve_with_ctx(&new_eq, var, simplifier, ctx) {
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
            if let Some((new_eq, desc)) = invert(eq.rhs, eq.lhs, eq.op.clone(), false) {
                let mut steps = Vec::new();
                if simplifier.collect_steps() {
                    steps.push(SolveStep {
                        description: desc,
                        equation_after: new_eq.clone(),
                        importance: crate::step::ImportanceLevel::Medium,
                        substeps: vec![],
                    });
                }
                match solve_with_ctx(&new_eq, var, simplifier, ctx) {
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
        let lhs_has = contains_var(&simplifier.context, eq.lhs, var);
        let rhs_has = contains_var(&simplifier.context, eq.rhs, var);

        // Only apply if variable is on BOTH sides
        if !lhs_has || !rhs_has {
            return None;
        }

        let mut steps = Vec::new();

        // Strategy: Subtract RHS from both sides to move everything to LHS
        // ax + b = cx + d  ->  ax + b - (cx + d) = cx + d - (cx + d)
        //                  ->  ax - cx + b - d = 0

        let rewritten = cas_solver_core::equation_rewrite::subtract_rhs_from_both_sides(
            &mut simplifier.context,
            eq,
        );

        // Simplify both sides
        let (simp_lhs, _) = simplifier.simplify(rewritten.lhs);
        let (simp_rhs, _) = simplifier.simplify(rewritten.rhs);

        if simplifier.collect_steps() {
            let rhs_desc = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id: eq.rhs
                }
            );
            steps.push(SolveStep {
                description: subtract_both_sides_message(&rhs_desc),
                equation_after: Equation {
                    lhs: simp_lhs,
                    rhs: simp_rhs,
                    op: eq.op.clone(),
                },
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

        // Rewrite isolated lhs^(p/q)=rhs into lhs^p=rhs^q via solver core helper.
        let (raw_eq, _p, q) =
            cas_solver_core::rational_power::rewrite_isolated_rational_power_equation(
                &mut simplifier.context,
                eq.lhs,
                eq.rhs,
                var,
                eq.op.clone(),
                lhs_has,
                rhs_has,
            )?;

        let mut steps = Vec::new();

        // Simplify both sides
        let (sim_lhs, _) = simplifier.simplify(raw_eq.lhs);
        let (sim_rhs, _) = simplifier.simplify(raw_eq.rhs);

        let new_eq = Equation {
            lhs: sim_lhs,
            rhs: sim_rhs,
            op: RelOp::Eq,
        };

        if simplifier.collect_steps() {
            let q_desc = q.to_string();
            steps.push(SolveStep {
                description: eliminate_fractional_exponent_message(&q_desc),
                equation_after: new_eq.clone(),
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
