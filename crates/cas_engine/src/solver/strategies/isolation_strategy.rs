use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::isolation::isolate;
use crate::solver::solve_core::solve_with_ctx;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{SolveCtx, SolveDomainEnv, SolveStep, SolverOptions};
use cas_ast::{Context, Equation, Expr, ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_utils::contains_var;
use cas_solver_core::log_domain::LogSolveDecision;
use num_traits::Zero;

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
            let new_op = match eq.op {
                RelOp::Eq => RelOp::Eq,
                RelOp::Neq => RelOp::Neq,
                RelOp::Lt => RelOp::Gt,
                RelOp::Gt => RelOp::Lt,
                RelOp::Leq => RelOp::Geq,
                RelOp::Geq => RelOp::Leq,
            };
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
    use crate::domain::DomainMode;
    use crate::semantics::AssumeScope;
    use crate::solver::domain_guards::classify_log_solve;

    // Check LHS for exponential a^x pattern
    if lhs_has && !rhs_has {
        if let Expr::Pow(base, exp) = simplifier.context.get(eq.lhs).clone() {
            // Check if exponent contains var and base doesn't
            if contains_var(&simplifier.context, exp, var)
                && !contains_var(&simplifier.context, base, var)
            {
                let decision = classify_log_solve(&simplifier.context, base, eq.rhs, opts, env);

                match &decision {
                    LogSolveDecision::EmptySet(msg) => {
                        // base > 0 but RHS <= 0 proven: no real solutions exist
                        let mut steps = Vec::new();
                        if simplifier.collect_steps() {
                            steps.push(SolveStep {
                                description: msg.to_string(),
                                equation_after: eq.clone(),
                                importance: crate::step::ImportanceLevel::Medium,
                                substeps: vec![],
                            });
                        }
                        return Some(Ok((SolutionSet::Empty, steps)));
                    }
                    LogSolveDecision::NeedsComplex(msg) => {
                        // Check if we're in Wildcard mode
                        if opts.domain_mode == DomainMode::Assume
                            && opts.assume_scope == AssumeScope::Wildcard
                        {
                            let residual = cas_solver_core::isolation_utils::mk_residual_solve(
                                &mut simplifier.context,
                                eq.lhs,
                                eq.rhs,
                                var,
                            );

                            // Create step with warning
                            let mut steps = Vec::new();
                            if simplifier.collect_steps() {
                                steps.push(SolveStep {
                                    description: format!(
                                        "{} - use 'semantics preset complex'",
                                        msg
                                    ),
                                    equation_after: eq.clone(),
                                    importance: crate::step::ImportanceLevel::Medium,
                                    substeps: vec![],
                                });
                            }

                            return Some(Ok((SolutionSet::Residual(residual), steps)));
                        }
                        // If not Wildcard, let other handlers deal with it
                    }
                    _ => {}
                }
            }
        }
    }

    // Check RHS for exponential pattern (symmetric case)
    if rhs_has && !lhs_has {
        if let Expr::Pow(base, exp) = simplifier.context.get(eq.rhs).clone() {
            if contains_var(&simplifier.context, exp, var)
                && !contains_var(&simplifier.context, base, var)
            {
                let decision = classify_log_solve(&simplifier.context, base, eq.lhs, opts, env);

                match &decision {
                    LogSolveDecision::EmptySet(msg) => {
                        // base > 0 but LHS <= 0 proven: no real solutions exist
                        let mut steps = Vec::new();
                        if simplifier.collect_steps() {
                            steps.push(SolveStep {
                                description: msg.to_string(),
                                equation_after: eq.clone(),
                                importance: crate::step::ImportanceLevel::Medium,
                                substeps: vec![],
                            });
                        }
                        return Some(Ok((SolutionSet::Empty, steps)));
                    }
                    LogSolveDecision::NeedsComplex(msg) => {
                        if opts.domain_mode == DomainMode::Assume
                            && opts.assume_scope == AssumeScope::Wildcard
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
                                    description: format!(
                                        "{} - use 'semantics preset complex'",
                                        msg
                                    ),
                                    equation_after: eq.clone(),
                                    importance: crate::step::ImportanceLevel::Medium,
                                    substeps: vec![],
                                });
                            }

                            return Some(Ok((SolutionSet::Residual(residual), steps)));
                        }
                    }
                    _ => {}
                }
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
                    let name = simplifier.context.sym_name(fn_id);
                    match name {
                        "sqrt" => {
                            // sqrt(A) = B -> A = B^2
                            // Check domain? sqrt(A) >= 0. So B must be >= 0.
                            // We should add a constraint or verify later.
                            // For now, just transform. Verification step in solve() handles extraneous roots.
                            let two = simplifier.context.num(2);
                            let new_other = simplifier.context.add(Expr::Pow(other, two));
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
                            Some((new_eq, "Square both sides".to_string()))
                        }
                        "ln" => {
                            // ln(A) = B -> A = e^B
                            let e = simplifier.context.add(Expr::Constant(cas_ast::Constant::E));
                            let new_other = simplifier.context.add(Expr::Pow(e, other));
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
                            Some((new_eq, "Exponentiate (base e)".to_string()))
                        }
                        "exp" => {
                            // exp(A) = B -> A = ln(B)
                            let new_other = simplifier.context.call("ln", vec![other]);
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
                            Some((new_eq, "Take natural log".to_string()))
                        }
                        _ => None,
                    }
                }
                Expr::Pow(b, e) => {
                    // A^n = B -> A = B^(1/n) (if n is const)
                    // If A contains var and n does not.
                    if contains_var(&simplifier.context, b, var)
                        && !contains_var(&simplifier.context, e, var)
                    {
                        // Prevent unwrapping positive integer powers (handled by Polynomial/Quadratic)
                        // e.g. x^2 = ... don't turn into x = sqrt(...)
                        let is_pos_int = |ctx: &Context, e_id: ExprId| -> bool {
                            match ctx.get(e_id) {
                                Expr::Number(n) => {
                                    n.is_integer()
                                        && *n > num_rational::BigRational::from_integer(0.into())
                                }
                                Expr::Div(n_id, d_id) => {
                                    if let (Expr::Number(n), Expr::Number(d)) =
                                        (ctx.get(*n_id), ctx.get(*d_id))
                                    {
                                        if !d.is_zero() {
                                            let val = n / d;
                                            return val.is_integer()
                                                && val
                                                    > num_rational::BigRational::from_integer(
                                                        0.into(),
                                                    );
                                        }
                                    }
                                    false
                                }
                                _ => false,
                            }
                        };

                        if is_pos_int(&simplifier.context, e) {
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
                    } else if !contains_var(&simplifier.context, b, var)
                        && contains_var(&simplifier.context, e, var)
                    {
                        // A^x = B -> x * ln(A) = ln(B)
                        // Use domain classifier for semantic-aware solving

                        use crate::solver::domain_guards::classify_log_solve;

                        // PRE-CHECK: Handle base = 1 before classifier
                        // 1^x = 1 -> AllReals, 1^x = b (b≠1) -> Empty
                        if let Expr::Number(n) = simplifier.context.get(b) {
                            if *n == num_rational::BigRational::from_integer(1.into()) {
                                // Base is 1
                                if let Expr::Number(rhs_n) = simplifier.context.get(other) {
                                    if *rhs_n == num_rational::BigRational::from_integer(1.into()) {
                                        // 1^x = 1 -> AllReals (handled specially)
                                        // We can't return AllReals directly from invert closure,
                                        // so skip and let IsolationStrategy handle it
                                        return None;
                                    } else {
                                        // 1^x = b (b≠1) -> Empty (also skip)
                                        return None;
                                    }
                                }
                                // 1^x = symbolic -> skip (can be 1 or not)
                                return None;
                            }
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
                                    let event = crate::solver::domain_guards::assumption_to_assumption_event(
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
