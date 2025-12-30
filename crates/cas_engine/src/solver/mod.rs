use crate::build::mul2_raw;
pub mod domain_guards;
pub mod isolation;
pub mod solution_set;
pub mod strategies;
pub mod strategy;

use crate::engine::Simplifier;
use cas_ast::{Context, Equation, ExprId, SolutionSet};

pub use self::isolation::contains_var;

/// Options for solver operations, containing semantic context.
///
/// This struct passes value domain and domain mode information to the solver,
/// enabling domain-aware decisions like rejecting log operations on negative bases.
#[derive(Debug, Clone, Copy)]
pub struct SolverOptions {
    /// The value domain (RealOnly or ComplexEnabled)
    pub value_domain: crate::semantics::ValueDomain,
    /// The domain mode (Strict, Assume, Generic)
    pub domain_mode: crate::domain::DomainMode,
    /// Scope for assumptions (only active if domain_mode=Assume)
    pub assume_scope: crate::semantics::AssumeScope,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            value_domain: crate::semantics::ValueDomain::RealOnly,
            domain_mode: crate::domain::DomainMode::Generic,
            assume_scope: crate::semantics::AssumeScope::Real,
        }
    }
}

/// Helper: Build a 2-factor product (no normalization).

#[derive(Debug, Clone)]
pub struct SolveStep {
    pub description: String,
    pub equation_after: Equation,
}

use crate::error::CasError;

use crate::solver::strategies::{
    CollectTermsStrategy, IsolationStrategy, QuadraticStrategy, RationalExponentStrategy,
    SubstitutionStrategy, UnwrapStrategy,
};
use crate::solver::strategy::SolverStrategy;

/// Maximum recursion depth for solver to prevent stack overflow
pub(crate) const MAX_SOLVE_DEPTH: usize = 50;

thread_local! {
    pub(crate) static SOLVE_DEPTH: std::cell::RefCell<usize> = const { std::cell::RefCell::new(0) };
    /// Thread-local collector for solver assumptions.
    /// Used to pass assumptions from strategies back to caller without changing return type.
    static SOLVE_ASSUMPTIONS: std::cell::RefCell<Option<crate::assumptions::AssumptionCollector>> =
        const { std::cell::RefCell::new(None) };
}

/// RAII guard for solver assumption collection.
///
/// Creates a fresh collector on creation, and on drop:
/// - Returns the collected assumptions via `finish()`
/// - Restores any previous collector (for nested solves)
///
/// # Safety against leaks/reentrancy
/// - Drop always clears or restores state
/// - Nested solves get their own collectors (previous is saved)
pub struct SolveAssumptionsGuard {
    /// The previous collector (if any) that was active before this guard
    previous: Option<crate::assumptions::AssumptionCollector>,
    /// Whether collection is enabled for this guard
    enabled: bool,
}

impl SolveAssumptionsGuard {
    /// Create a new guard that starts assumption collection.
    /// If `enabled` is false, no collection happens (passthrough).
    pub fn new(enabled: bool) -> Self {
        let previous = if enabled {
            // Take any existing collector (for nested solve case)
            let prev = SOLVE_ASSUMPTIONS.with(|c| c.borrow_mut().take());
            // Install fresh collector
            SOLVE_ASSUMPTIONS.with(|c| {
                *c.borrow_mut() = Some(crate::assumptions::AssumptionCollector::new());
            });
            prev
        } else {
            None
        };

        Self { previous, enabled }
    }

    /// Finish collection and return the collected records.
    /// This consumes the guard.
    pub fn finish(self) -> Vec<crate::assumptions::AssumptionRecord> {
        // The Drop impl will restore previous, we just need to take current
        if self.enabled {
            SOLVE_ASSUMPTIONS.with(|c| {
                c.borrow_mut()
                    .take()
                    .map(|collector| collector.finish())
                    .unwrap_or_default()
            })
        } else {
            vec![]
        }
    }
}

impl Drop for SolveAssumptionsGuard {
    fn drop(&mut self) {
        if self.enabled {
            // Restore previous collector (or None if there wasn't one)
            SOLVE_ASSUMPTIONS.with(|c| {
                *c.borrow_mut() = self.previous.take();
            });
        }
    }
}

/// Start assumption collection for solver.
/// DEPRECATED: Use SolveAssumptionsGuard for RAII safety.
/// Returns true if collection was started (false if already active).
pub fn start_assumption_collection() -> bool {
    SOLVE_ASSUMPTIONS.with(|c| {
        let mut collector = c.borrow_mut();
        if collector.is_none() {
            *collector = Some(crate::assumptions::AssumptionCollector::new());
            true
        } else {
            false
        }
    })
}

/// Finish assumption collection and return the collector.
/// DEPRECATED: Use SolveAssumptionsGuard for RAII safety.
/// Returns None if collection wasn't started.
pub fn finish_assumption_collection() -> Option<crate::assumptions::AssumptionCollector> {
    SOLVE_ASSUMPTIONS.with(|c| c.borrow_mut().take())
}

/// Note an assumption during solver operation (internal use).
pub(crate) fn note_assumption(event: crate::assumptions::AssumptionEvent) {
    SOLVE_ASSUMPTIONS.with(|c| {
        if let Some(ref mut collector) = *c.borrow_mut() {
            collector.note(event);
        }
    });
}

/// Guard that decrements depth on drop
pub(crate) struct DepthGuard;

impl Drop for DepthGuard {
    fn drop(&mut self) {
        SOLVE_DEPTH.with(|d| {
            *d.borrow_mut() -= 1;
        });
    }
}

/// Solve with default options (for backward compatibility with tests).
/// Uses RealOnly domain and Generic mode.
pub fn solve(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    solve_with_options(eq, var, simplifier, SolverOptions::default())
}

/// Solve an equation with explicit semantic options.
///
/// This is the primary entry point for domain-aware solving.
/// `opts` contains ValueDomain and DomainMode which control:
/// - Whether log operations are valid (RealOnly requires positive arguments)
/// - Whether to emit assumptions or reject operations
pub fn solve_with_options(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    // Check and increment recursion depth
    let current_depth = SOLVE_DEPTH.with(|d| {
        let mut depth = d.borrow_mut();
        *depth += 1;
        *depth
    });

    // Create guard to decrement on exit
    let _guard = DepthGuard;

    if current_depth > MAX_SOLVE_DEPTH {
        return Err(CasError::SolverError(
            "Maximum solver recursion depth exceeded. The equation may be too complex.".to_string(),
        ));
    }

    // 1. Check if variable exists in equation
    if !contains_var(&simplifier.context, eq.lhs, var)
        && !contains_var(&simplifier.context, eq.rhs, var)
    {
        return Err(CasError::VariableNotFound(var.to_string()));
    }

    // EARLY CHECK: Handle rational exponent equations BEFORE simplification
    // This prevents x^(3/2) from being simplified to |x|*sqrt(x) which causes loops
    if eq.op == cas_ast::RelOp::Eq {
        if let Some(result) = try_solve_rational_exponent(eq, var, simplifier) {
            return result;
        }
    }

    // 2. Simplify both sides BEFORE applying strategies
    // This is crucial for equations like "1/3*x + 1/2*x = 5"
    // which need to be simplified to "5/6*x = 5" before isolation
    let mut simplified_eq = eq.clone();

    // Simplify LHS if it contains the variable
    if contains_var(&simplifier.context, eq.lhs, var) {
        let (sim_lhs, _) = simplifier.simplify(eq.lhs);
        simplified_eq.lhs = sim_lhs;

        // After simplification, try to recompose a^x/b^x -> (a/b)^x
        // This fixes cases where (a/b)^x was expanded to a^x/b^x during simplify,
        // which would leave 'x' on both sides after isolation attempts.
        if let Some(recomposed) =
            isolation::try_recompose_pow_quotient(&mut simplifier.context, sim_lhs)
        {
            simplified_eq.lhs = recomposed;
        }
    }

    // Simplify RHS if it contains the variable
    if contains_var(&simplifier.context, eq.rhs, var) {
        let (sim_rhs, _) = simplifier.simplify(eq.rhs);
        simplified_eq.rhs = sim_rhs;

        // Also try recomposition on RHS
        if let Some(recomposed) =
            isolation::try_recompose_pow_quotient(&mut simplifier.context, sim_rhs)
        {
            simplified_eq.rhs = recomposed;
        }
    }

    // CRITICAL: After simplification, check for identities and contradictions
    // Do this by moving everything to one side: LHS - RHS
    let difference = simplifier
        .context
        .add(cas_ast::Expr::Sub(simplified_eq.lhs, simplified_eq.rhs));
    let (diff_simplified, _) = simplifier.simplify(difference);

    // Check if the difference has NO variable
    if !contains_var(&simplifier.context, diff_simplified, var) {
        // Variable disappeared - this is either an identity or contradiction
        // Simplify the difference and check if it's zero
        use cas_ast::Expr;
        match simplifier.context.get(diff_simplified) {
            Expr::Number(n) => {
                use num_traits::Zero;
                if n.is_zero() {
                    // 0 = 0: Identity, all real numbers
                    return Ok((SolutionSet::AllReals, vec![]));
                } else {
                    // c = 0 where c ≠ 0: Contradiction, no solution
                    return Ok((SolutionSet::Empty, vec![]));
                }
            }
            _ => {
                // Difference couldn't simplify to a number
                // This might be a complex case, proceed with normal solving
            }
        }
    }

    // 3. Define strategies
    // In a real app, these might be configured in Simplifier or passed in.
    let strategies: Vec<Box<dyn SolverStrategy>> = vec![
        Box::new(RationalExponentStrategy), // Must run BEFORE UnwrapStrategy to avoid loops
        Box::new(SubstitutionStrategy),
        Box::new(UnwrapStrategy),
        Box::new(QuadraticStrategy),
        Box::new(CollectTermsStrategy), // Must run before IsolationStrategy
        Box::new(IsolationStrategy),
    ];

    // 4. Try strategies on the simplified equation
    for strategy in strategies {
        if let Some(res) = strategy.apply(&simplified_eq, var, simplifier, &opts) {
            match res {
                Ok((result, steps)) => {
                    // Verify solutions if Discrete
                    if let SolutionSet::Discrete(sols) = result {
                        if !strategy.should_verify() {
                            return Ok((SolutionSet::Discrete(sols), steps));
                        }
                        let mut valid_sols = Vec::new();
                        for sol in sols {
                            // Skip verification for symbolic solutions (containing functions/variables)
                            // These can't be verified by substitution. Examples: ln(c/d)/ln(a/b)
                            // Only verify pure numeric solutions to catch extraneous roots.
                            if is_symbolic_expr(&simplifier.context, sol) {
                                // Trust symbolic solutions - can't verify
                                valid_sols.push(sol);
                            } else {
                                // CRITICAL: Verify against ORIGINAL equation, not simplified
                                // This ensures we reject solutions that cause division by zero
                                // in the original equation, even if they work in the simplified form
                                if verify_solution(eq, var, sol, simplifier) {
                                    valid_sols.push(sol);
                                }
                            }
                        }
                        return Ok((SolutionSet::Discrete(valid_sols), steps));
                    }
                    return Ok((result, steps));
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }
    }

    Err(CasError::SolverError(
        "No strategy could solve this equation.".to_string(),
    ))
}

/// Try to solve equations with rational exponents like x^(3/2) = 8
/// by converting to x^3 = 64 (raising both sides to power q)
fn try_solve_rational_exponent(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    use cas_ast::Expr;
    use strategies::match_rational_power;

    // Check if RHS contains the variable (we only handle simple cases)
    if contains_var(&simplifier.context, eq.rhs, var) {
        return None;
    }

    // Try to match x^(p/q) on LHS
    let (base, p, q) = match_rational_power(&simplifier.context, eq.lhs, var)?;

    let mut steps = Vec::new();

    // Build new equation: base^p = rhs^q
    let p_expr = simplifier.context.num(p);
    let q_expr = simplifier.context.num(q);

    let new_lhs = simplifier.context.add(Expr::Pow(base, p_expr));
    let new_rhs = simplifier.context.add(Expr::Pow(eq.rhs, q_expr));

    // Simplify RHS (no variable, safe to simplify)
    let (sim_rhs, _) = simplifier.simplify(new_rhs);

    // DON'T simplify LHS yet - it might contain x^p which we want to solve

    let new_eq = Equation {
        lhs: new_lhs,
        rhs: sim_rhs,
        op: cas_ast::RelOp::Eq,
    };

    if simplifier.collect_steps() {
        steps.push(SolveStep {
            description: format!(
                "Raise both sides to power {} to eliminate rational exponent",
                q
            ),
            equation_after: new_eq.clone(),
        });
    }

    // Recursively solve (this will go through the full solve pipeline)
    match solve(&new_eq, var, simplifier) {
        Ok((set, mut sub_steps)) => {
            steps.append(&mut sub_steps);

            // Verify solutions against original equation (handles extraneous roots)
            if let SolutionSet::Discrete(sols) = set {
                let mut valid_sols = Vec::new();
                for sol in sols {
                    if verify_solution(eq, var, sol, simplifier) {
                        valid_sols.push(sol);
                    }
                }
                Some(Ok((SolutionSet::Discrete(valid_sols), steps)))
            } else {
                Some(Ok((set, steps)))
            }
        }
        Err(e) => Some(Err(e)),
    }
}

fn verify_solution(eq: &Equation, var: &str, sol: ExprId, simplifier: &mut Simplifier) -> bool {
    // 1. Substitute
    let lhs_sub = substitute(&mut simplifier.context, eq.lhs, var, sol);
    let rhs_sub = substitute(&mut simplifier.context, eq.rhs, var, sol);

    // 2. Simplify
    let (lhs_sim, _) = simplifier.simplify(lhs_sub);
    let (rhs_sim, _) = simplifier.simplify(rhs_sub);

    // 3. Check equality
    simplifier.are_equivalent(lhs_sim, rhs_sim)
}

/// Check if an expression is "symbolic" (contains functions or variables).
/// Symbolic expressions cannot be verified by substitution because they don't
/// simplify to pure numbers. Examples: ln(c/d)/ln(a/b), x + a, sqrt(y)
fn is_symbolic_expr(ctx: &Context, expr: ExprId) -> bool {
    use cas_ast::Expr;
    match ctx.get(expr) {
        Expr::Number(_) => false,
        Expr::Constant(_) => true, // Pi, E, etc are symbolic
        Expr::Variable(_) => true,
        Expr::Function(_, _) => true,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            is_symbolic_expr(ctx, *l) || is_symbolic_expr(ctx, *r)
        }
        Expr::Neg(e) => is_symbolic_expr(ctx, *e),
        _ => true, // Default: treat as symbolic
    }
}

fn substitute(ctx: &mut Context, expr: ExprId, var: &str, val: ExprId) -> ExprId {
    use cas_ast::Expr;
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Variable(v) if v == var => val,
        Expr::Add(l, r) => {
            let nl = substitute(ctx, l, var, val);
            let nr = substitute(ctx, r, var, val);
            if nl != l || nr != r {
                ctx.add(Expr::Add(nl, nr))
            } else {
                expr
            }
        }
        Expr::Sub(l, r) => {
            let nl = substitute(ctx, l, var, val);
            let nr = substitute(ctx, r, var, val);
            if nl != l || nr != r {
                ctx.add(Expr::Sub(nl, nr))
            } else {
                expr
            }
        }
        Expr::Mul(l, r) => {
            let nl = substitute(ctx, l, var, val);
            let nr = substitute(ctx, r, var, val);
            if nl != l || nr != r {
                mul2_raw(ctx, nl, nr)
            } else {
                expr
            }
        }
        Expr::Div(l, r) => {
            let nl = substitute(ctx, l, var, val);
            let nr = substitute(ctx, r, var, val);
            if nl != l || nr != r {
                ctx.add(Expr::Div(nl, nr))
            } else {
                expr
            }
        }
        Expr::Pow(b, e) => {
            let nb = substitute(ctx, b, var, val);
            let ne = substitute(ctx, e, var, val);
            if nb != b || ne != e {
                ctx.add(Expr::Pow(nb, ne))
            } else {
                expr
            }
        }
        Expr::Neg(e) => {
            let ne = substitute(ctx, e, var, val);
            if ne != e {
                ctx.add(Expr::Neg(ne))
            } else {
                expr
            }
        }
        Expr::Function(name, args) => {
            let mut new_args = Vec::new();
            let mut changed = false;
            for arg in args {
                let new_arg = substitute(ctx, arg, var, val);
                if new_arg != arg {
                    changed = true;
                }
                new_args.push(new_arg);
            }
            if changed {
                ctx.add(Expr::Function(name, new_args))
            } else {
                expr
            }
        }
        _ => expr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::{BoundType, Context, DisplayExpr, RelOp};
    use cas_parser::parse;

    // Helper to make equation from strings
    fn make_eq(ctx: &mut Context, lhs: &str, rhs: &str) -> Equation {
        Equation {
            lhs: parse(lhs, ctx).unwrap(),
            rhs: parse(rhs, ctx).unwrap(),
            op: RelOp::Eq,
        }
    }

    #[test]
    fn test_solve_linear() {
        // x + 2 = 5 -> x = 3
        let mut simplifier = Simplifier::new();
        let eq = make_eq(&mut simplifier.context, "x + 2", "5");
        simplifier.set_collect_steps(true);
        let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();

        if let SolutionSet::Discrete(solutions) = result {
            assert_eq!(solutions.len(), 1);
            let s = format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: solutions[0]
                }
            );
            assert_eq!(s, "3");
        } else {
            panic!("Expected Discrete solution");
        }
    }

    #[test]
    fn test_solve_mul() {
        // 2 * x = 6 -> x = 6 / 2
        let mut simplifier = Simplifier::with_default_rules();
        let eq = make_eq(&mut simplifier.context, "2 * x", "6");
        simplifier.set_collect_steps(true);
        let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();

        if let SolutionSet::Discrete(solutions) = result {
            assert_eq!(solutions.len(), 1);
            let s = format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: solutions[0]
                }
            );
            assert_eq!(s, "3");
        } else {
            panic!("Expected Discrete solution");
        }
    }

    #[test]
    fn test_solve_pow() {
        // x^2 = 4 -> x = 4^(1/2)
        let mut simplifier = Simplifier::new();
        let eq = make_eq(&mut simplifier.context, "x^2", "4");
        simplifier.add_rule(Box::new(crate::rules::exponents::EvaluatePowerRule));
        simplifier.add_rule(Box::new(
            crate::rules::canonicalization::CanonicalizeNegationRule,
        ));
        simplifier.add_rule(Box::new(crate::rules::arithmetic::CombineConstantsRule));
        simplifier.set_collect_steps(true);
        let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();

        if let SolutionSet::Discrete(mut solutions) = result {
            assert_eq!(solutions.len(), 2);
            // Sort to ensure order
            solutions.sort_by(|a, b| {
                let sa = format!(
                    "{}",
                    DisplayExpr {
                        context: &simplifier.context,
                        id: *a
                    }
                );
                let sb = format!(
                    "{}",
                    DisplayExpr {
                        context: &simplifier.context,
                        id: *b
                    }
                );
                sa.cmp(&sb)
            });

            let s1 = format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: solutions[0]
                }
            );
            let s2 = format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: solutions[1]
                }
            );

            // We want to eventually see "-2" and "2".
            assert_eq!(s1, "-2");
            assert_eq!(s2, "2");
        } else {
            panic!("Expected Discrete solution");
        }
    }

    #[test]
    fn test_solve_abs() {
        // |x| = 5 -> x=5, x=-5
        let mut simplifier = Simplifier::new();
        let eq = make_eq(&mut simplifier.context, "|x|", "5");
        let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();

        if let SolutionSet::Discrete(solutions) = result {
            assert_eq!(solutions.len(), 2);
            // Order might vary
            let s: Vec<String> = solutions
                .iter()
                .map(|e| {
                    format!(
                        "{}",
                        DisplayExpr {
                            context: &simplifier.context,
                            id: *e
                        }
                    )
                })
                .collect();
            assert!(s.contains(&"5".to_string()));
            assert!(s.contains(&"-5".to_string()));
        } else {
            panic!("Expected Discrete solution");
        }
    }

    #[test]
    fn test_solve_inequality_flip() {
        // -2x < 10 -> x > -5
        let mut simplifier = Simplifier::with_default_rules();
        let eq = Equation {
            lhs: parse("-2*x", &mut simplifier.context).unwrap(),
            rhs: parse("10", &mut simplifier.context).unwrap(),
            op: RelOp::Lt,
        };
        let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();

        if let SolutionSet::Continuous(interval) = result {
            // (-5, inf)
            let s_min = format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: interval.min
                }
            );
            let s_max = format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: interval.max
                }
            );
            assert!(
                s_min == "-5" || s_min == "10 / -2",
                "Expected -5 or canonical form 10 / -2, got: {}",
                s_min
            );
            assert_eq!(interval.min_type, BoundType::Open);
            assert_eq!(s_max, "infinity");
        } else {
            panic!("Expected Continuous solution, got {:?}", result);
        }
    }

    #[test]
    fn test_solve_abs_inequality() {
        // |x| < 5 -> (-5, 5)
        let mut simplifier = Simplifier::new();
        let eq = Equation {
            lhs: parse("|x|", &mut simplifier.context).unwrap(),
            rhs: parse("5", &mut simplifier.context).unwrap(),
            op: RelOp::Lt,
        };
        let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();

        if let SolutionSet::Continuous(interval) = result {
            let s_min = format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: interval.min
                }
            );
            let s_max = format!(
                "{}",
                DisplayExpr {
                    context: &simplifier.context,
                    id: interval.max
                }
            );
            assert_eq!(s_min, "-5");
            assert_eq!(s_max, "5");
        } else {
            panic!("Expected Continuous solution, got {:?}", result);
        }
    }
}
