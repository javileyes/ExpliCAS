use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::isolation::isolate;
use crate::solver::solve_core::solve_with_ctx;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{SolveCtx, SolveDomainEnv, SolveStep, SolverOptions};
use cas_ast::{Equation, Expr, ExprId, RelOp, SolutionSet};
use cas_solver_core::function_inverse::UnaryInverseKind;
use cas_solver_core::isolation_utils::{
    contains_var, flip_inequality, is_numeric_one, is_positive_integer_expr,
    match_exponential_var_in_base, match_exponential_var_in_exponent,
};
use cas_solver_core::log_domain::LogSolveDecision;

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
            let new_op = flip_inequality(eq.op.clone());
            let mut steps = Vec::new();
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: "Swap sides to put variable on LHS".to_string(),
                    equation_after: Equation {
                        lhs: eq.rhs,
                        rhs: eq.lhs,
                        op: new_op.clone(),
                    },
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            // V2.0: Pass opts through to propagate budget
            match isolate(eq.rhs, eq.lhs, new_op, var, simplifier, *opts, ctx) {
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

fn terminal_exponential_decision_result(
    decision: &LogSolveDecision,
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: &SolverOptions,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    use crate::domain::DomainMode;
    use crate::semantics::AssumeScope;

    match decision {
        LogSolveDecision::EmptySet(msg) => {
            let mut steps = Vec::new();
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: msg.to_string(),
                    equation_after: eq.clone(),
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            Some(Ok((SolutionSet::Empty, steps)))
        }
        LogSolveDecision::NeedsComplex(msg)
            if opts.domain_mode == DomainMode::Assume
                && opts.assume_scope == AssumeScope::Wildcard =>
        {
            let residual = cas_solver_core::isolation_utils::mk_residual_solve(
                &mut simplifier.context,
                eq.lhs,
                eq.rhs,
                var,
            );
            let mut steps = Vec::new();
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: format!("{} - use 'semantics preset complex'", msg),
                    equation_after: eq.clone(),
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            Some(Ok((SolutionSet::Residual(residual), steps)))
        }
        _ => None,
    }
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

    // Check LHS for exponential a^x pattern
    if lhs_has && !rhs_has {
        if let Some(pattern) = match_exponential_var_in_exponent(&simplifier.context, eq.lhs, var) {
            let decision = classify_log_solve(&simplifier.context, pattern.base, eq.rhs, opts, env);
            if let Some(result) =
                terminal_exponential_decision_result(&decision, eq, var, simplifier, opts)
            {
                return Some(result);
            }
        }
    }

    // Check RHS for exponential pattern (symmetric case)
    if rhs_has && !lhs_has {
        if let Some(pattern) = match_exponential_var_in_exponent(&simplifier.context, eq.rhs, var) {
            let decision = classify_log_solve(&simplifier.context, pattern.base, eq.lhs, opts, env);
            if let Some(result) =
                terminal_exponential_decision_result(&decision, eq, var, simplifier, opts)
            {
                return Some(result);
            }
        }
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

        // Helper to invert
        let mut invert = |target: ExprId,
                          other: ExprId,
                          op: RelOp,
                          is_lhs: bool|
         -> Option<(Equation, String)> {
            let target_data = simplifier.context.get(target).clone();
            match target_data {
                Expr::Function(fn_id, args) if args.len() == 1 => {
                    let arg = args[0];
                    let name = simplifier.context.sym_name(fn_id).to_string();
                    let inverse_kind = UnaryInverseKind::from_name(&name)?;
                    // Unwrap strategy intentionally keeps scope narrow; inverse trig
                    // is handled by direct isolation path.
                    if !matches!(
                        inverse_kind,
                        UnaryInverseKind::Sqrt | UnaryInverseKind::Ln | UnaryInverseKind::Exp
                    ) {
                        None
                    } else {
                        let new_other = inverse_kind.build_rhs(&mut simplifier.context, other);
                        let new_eq = if is_lhs {
                            Equation {
                                lhs: arg,
                                rhs: new_other,
                                op,
                            }
                        } else {
                            Equation {
                                lhs: new_other,
                                rhs: arg,
                                op,
                            }
                        };
                        let description = match inverse_kind {
                            UnaryInverseKind::Sqrt => "Square both sides",
                            UnaryInverseKind::Ln => "Exponentiate (base e)",
                            UnaryInverseKind::Exp => "Take natural log",
                            _ => unreachable!("filtered above"),
                        };
                        Some((new_eq, description.to_string()))
                    }
                }
                Expr::Pow(_, _) => {
                    // A^n = B -> A = B^(1/n) (if n is const)
                    // If A contains var and n does not.
                    if let Some(pattern) =
                        match_exponential_var_in_base(&simplifier.context, target, var)
                    {
                        let b = pattern.base;
                        let e = pattern.exponent;
                        // Prevent unwrapping positive integer powers (handled by Polynomial/Quadratic)
                        // e.g. x^2 = ... don't turn into x = sqrt(...)
                        if is_positive_integer_expr(&simplifier.context, e) {
                            // Don't unwrap x^2, x^4 etc.
                            return None;
                        }

                        // A^n = B -> A = B^(1/n)
                        let one = simplifier.context.num(1);
                        let inv_exp = simplifier.context.add(Expr::Div(one, e));
                        let new_other = simplifier.context.add(Expr::Pow(other, inv_exp));
                        let new_eq = if is_lhs {
                            Equation {
                                lhs: b,
                                rhs: new_other,
                                op,
                            }
                        } else {
                            Equation {
                                lhs: new_other,
                                rhs: b,
                                op,
                            }
                        };
                        Some((
                            new_eq,
                            format!("Raise both sides to 1/{:?}", simplifier.context.get(e)),
                        ))
                    } else if let Some(pattern) =
                        match_exponential_var_in_exponent(&simplifier.context, target, var)
                    {
                        let b = pattern.base;
                        let e = pattern.exponent;
                        // A^x = B -> x * ln(A) = ln(B)
                        // Use domain classifier for semantic-aware solving

                        use crate::solver::domain_guards::classify_log_solve;

                        // PRE-CHECK: Handle base = 1 before classifier.
                        // This branch is solved by higher-level strategy handling.
                        if is_numeric_one(&simplifier.context, b) {
                            return None;
                        }

                        // Use the domain classifier
                        let decision = classify_log_solve(
                            &simplifier.context,
                            b,
                            other,
                            opts,
                            &ctx.domain_env,
                        );

                        match decision {
                            LogSolveDecision::Ok => {
                                // Safe to take ln - no assumptions needed
                            }
                            LogSolveDecision::OkWithAssumptions(assumptions) => {
                                // Record each assumption via the thread-local collector
                                for assumption in assumptions {
                                    let event =
                                        crate::assumptions::AssumptionEvent::from_log_assumption(
                                            assumption,
                                            &simplifier.context,
                                            b,
                                            other,
                                        );
                                    crate::solver::note_assumption(event);
                                }
                            }
                            LogSolveDecision::EmptySet(_) => {
                                // Should have been caught by check_exponential_needs_complex
                                // but if we get here somehow, skip (let outer handler deal with it)
                                return None;
                            }
                            LogSolveDecision::NeedsComplex(msg) => {
                                // In RealOnly, can't proceed
                                // In wildcard scope: should return residual (not implemented here)
                                // For now, skip and let IsolationStrategy handle
                                let _ = msg; // suppress warning
                                return None;
                            }
                            LogSolveDecision::Unsupported(_, _) => {
                                // Cannot justify in current mode - skip
                                return None;
                            }
                        }

                        // Safe to take ln of both sides
                        // ln(A^x) -> x * ln(A)
                        // We construct x * ln(A) directly
                        let ln_b = simplifier.context.call("ln", vec![b]);
                        let new_lhs_part = simplifier.context.add(Expr::Mul(e, ln_b));

                        // ln(B)
                        let ln_other = simplifier.context.call("ln", vec![other]);

                        let new_eq = if is_lhs {
                            Equation {
                                lhs: new_lhs_part,
                                rhs: ln_other,
                                op,
                            }
                        } else {
                            Equation {
                                lhs: ln_other,
                                rhs: new_lhs_part,
                                op,
                            }
                        };
                        Some((new_eq, "Take log base e of both sides".to_string()))
                    } else {
                        None
                    }
                }
                _ => None,
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

        let neg_rhs = simplifier.context.add(Expr::Neg(eq.rhs));
        let new_lhs = simplifier.context.add(Expr::Add(eq.lhs, neg_rhs));
        let new_rhs = simplifier.context.add(Expr::Add(eq.rhs, neg_rhs));

        // Simplify both sides
        let (simp_lhs, _) = simplifier.simplify(new_lhs);
        let (simp_rhs, _) = simplifier.simplify(new_rhs);

        if simplifier.collect_steps() {
            steps.push(SolveStep {
                description: format!(
                    "Subtract {} from both sides",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: eq.rhs
                    }
                ),
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
        // Only handle equality for now
        if eq.op != RelOp::Eq {
            return None;
        }

        // Check if LHS is Pow(base, exp) where base contains var and exp is rational p/q
        let lhs_has = contains_var(&simplifier.context, eq.lhs, var);
        let rhs_has = contains_var(&simplifier.context, eq.rhs, var);

        // We need var only on one side in a power
        if rhs_has {
            return None;
        }
        if !lhs_has {
            return None;
        }

        // Try to match Pow(base, p/q) on LHS
        let (base, p, q) = cas_solver_core::rational_power::match_rational_power(
            &simplifier.context,
            eq.lhs,
            var,
        )?;

        let mut steps = Vec::new();

        // Raise both sides to power q: (base^(p/q))^q = rhs^q → base^p = rhs^q
        let q_expr = simplifier.context.num(q);

        // New LHS: base^p
        let p_expr = simplifier.context.num(p);
        let new_lhs = simplifier.context.add(Expr::Pow(base, p_expr));

        // New RHS: rhs^q
        let new_rhs = simplifier.context.add(Expr::Pow(eq.rhs, q_expr));

        // Simplify both sides
        let (sim_lhs, _) = simplifier.simplify(new_lhs);
        let (sim_rhs, _) = simplifier.simplify(new_rhs);

        let new_eq = Equation {
            lhs: sim_lhs,
            rhs: sim_rhs,
            op: RelOp::Eq,
        };

        if simplifier.collect_steps() {
            steps.push(SolveStep {
                description: format!(
                    "Raise both sides to power {} to eliminate fractional exponent",
                    q
                ),
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
