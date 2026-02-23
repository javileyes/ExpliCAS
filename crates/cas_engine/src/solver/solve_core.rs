//! Core solve dispatch pipeline.
//!
//! Contains the main `solve`, `solve_with_options`, and `solve_with_display_steps`
//! entry points, plus the rational-exponent pre-check.

use super::SolveDiagnostics;
use cas_ast::{ExprId, SolutionSet};
use cas_solver_core::isolation_utils::contains_var;
use cas_solver_core::solve_analysis::{
    apply_nonzero_exclusion_guards_if_any, classify_equation_var_presence,
    guard_solved_result_with_exclusions, normalize_variable_residual_with_runtime,
    simplify_equation_sides_for_var_with_runtime, EquationVarPresence, ResidualRewriteRuntime,
    SolvePreprocessRuntime,
};
use cas_solver_core::solve_outcome::{
    resolve_var_eliminated_outcome_with, solve_var_eliminated_constraint_pipeline_with_item,
    VarEliminatedSolveOutcome,
};
use cas_solver_core::strategy_kernels::{
    execute_rational_exponent_rewrite_with_runtime_for_var,
    solve_rational_exponent_rewrite_pipeline_with_item, RationalExponentRewriteRuntime,
    StrategyKernelRuntime,
};
use cas_solver_core::verify_substitution::{verify_solution_with_runtime, VerifySolutionRuntime};

use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::strategies::isolation_strategy::{
    CollectTermsStrategy, IsolationStrategy, RationalExponentStrategy, UnwrapStrategy,
};
use crate::solver::strategies::quadratic::QuadraticStrategy;
use crate::solver::strategies::rational_roots::RationalRootsStrategy;
use crate::solver::strategies::substitution::SubstitutionStrategy;
use crate::solver::strategy::SolverStrategy;

use super::{
    DepthGuard, DisplaySolveSteps, SolveDomainEnv, SolveStep, SolverOptions, MAX_SOLVE_DEPTH,
    SOLVE_DEPTH,
};

// NOTE: Pre-solve exponent normalization (Div(p,q) → Number(p/q)) and
// common additive term cancellation (Sub(Add(A,B), B) → A) were moved
// from solver-only code to global simplifier rules:
//   - rules/rational_canonicalization.rs (CanonicalizeRationalDivRule, CanonicalizeNestedPowRule)
//   - rules/cancel_common_terms.rs (CancelCommonAdditiveTermsRule)
// These are now applied automatically by simplify_for_solve().

fn verify_solution(
    eq: &cas_ast::Equation,
    var: &str,
    sol: ExprId,
    simplifier: &mut Simplifier,
) -> bool {
    let mut runtime = EngineVerifySolutionRuntime { simplifier };
    verify_solution_with_runtime(&mut runtime, eq, var, sol)
}

struct EngineRationalExponentRuntime<'a> {
    simplifier: &'a mut Simplifier,
}

struct EngineSolvePreprocessRuntime<'a> {
    simplifier: &'a mut Simplifier,
}

struct EngineResidualRewriteRuntime<'a> {
    simplifier: &'a mut Simplifier,
}

struct EngineRationalExponentRewriteRuntime<'a> {
    simplifier: &'a mut Simplifier,
    original_equation: &'a cas_ast::Equation,
    original_var: &'a str,
    solve_ctx: &'a super::SolveCtx,
}

struct EngineVerifySolutionRuntime<'a> {
    simplifier: &'a mut Simplifier,
}

impl VerifySolutionRuntime for EngineVerifySolutionRuntime<'_> {
    fn context(&mut self) -> &mut cas_ast::Context {
        &mut self.simplifier.context
    }

    fn simplify_expr(&mut self, expr: ExprId) -> ExprId {
        let (simplified, _) = self.simplifier.simplify(expr);
        simplified
    }

    fn are_equivalent(&mut self, lhs: ExprId, rhs: ExprId) -> bool {
        self.simplifier.are_equivalent(lhs, rhs)
    }
}

impl SolvePreprocessRuntime for EngineSolvePreprocessRuntime<'_> {
    fn context(&mut self) -> &mut cas_ast::Context {
        &mut self.simplifier.context
    }

    fn simplify_for_solve(&mut self, expr: ExprId) -> ExprId {
        self.simplifier.simplify_for_solve(expr)
    }
}

impl StrategyKernelRuntime for EngineRationalExponentRuntime<'_> {
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

impl ResidualRewriteRuntime for EngineResidualRewriteRuntime<'_> {
    fn context(&mut self) -> &mut cas_ast::Context {
        &mut self.simplifier.context
    }

    fn expand_algebraic(&mut self, expr: ExprId) -> ExprId {
        crate::expand::expand(&mut self.simplifier.context, expr)
    }

    fn simplify_for_solve(&mut self, expr: ExprId) -> ExprId {
        self.simplifier.simplify_for_solve(expr)
    }

    fn expand_trig(&mut self, expr: ExprId) -> ExprId {
        let (expanded, _) = self.simplifier.expand(expr);
        expanded
    }
}

impl RationalExponentRewriteRuntime<CasError, SolveStep>
    for EngineRationalExponentRewriteRuntime<'_>
{
    fn solve_rewritten(
        &mut self,
        equation: &cas_ast::Equation,
        var: &str,
    ) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
        solve_with_ctx(equation, var, self.simplifier, self.solve_ctx)
    }

    fn map_item_to_step(
        &mut self,
        item: cas_solver_core::strategy_kernels::StrategyExecutionItem,
    ) -> SolveStep {
        SolveStep {
            description: item.description,
            equation_after: item.equation,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        }
    }

    fn verify_discrete_solution(&mut self, solution: ExprId) -> bool {
        verify_solution(
            self.original_equation,
            self.original_var,
            solution,
            self.simplifier,
        )
    }
}

/// Solve with default options (for backward compatibility with tests).
/// Uses RealOnly domain and Generic mode.
///
/// This creates a fresh `SolveCtx`; conditions are NOT propagated
/// to any parent context. For recursive calls from strategies that
/// need to accumulate conditions, use [`solve_with_ctx`] instead.
pub fn solve(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    solve_with_options(eq, var, simplifier, SolverOptions::default())
}

/// Solve with a shared `SolveCtx` so conditions accumulate across recursive calls.
///
/// Called by strategies that have a `&SolveCtx` and want sub-solve conditions
/// to feed into the top-level diagnostics.
pub(crate) fn solve_with_ctx(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
    ctx: &super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    solve_inner(eq, var, simplifier, SolverOptions::default(), ctx)
}

/// V2.9.8: Solve with type-enforced display-ready steps.
///
/// This is the PREFERRED entry point for display-facing code (REPL, timeline, JSON API).
/// Returns `DisplaySolveSteps` which enforces that all renderers consume post-processed
/// steps, eliminating bifurcation between text/timeline outputs at compile time.
///
/// The cleanup is applied automatically based on `opts.detailed_steps`:
/// - `true` → 5 atomic sub-steps for Normal/Verbose verbosity
/// - `false` → 3 compact steps for Succinct verbosity
pub fn solve_with_display_steps(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
) -> Result<(SolutionSet, DisplaySolveSteps, SolveDiagnostics), CasError> {
    // Create a SolveCtx with a fresh accumulator — all recursive calls
    // through solve_with_ctx will push conditions into this shared set.
    let ctx = super::SolveCtx::default();
    let result = solve_inner(eq, var, simplifier, opts, &ctx);

    // Read accumulated conditions from the shared sink (zero TLS)
    let required: Vec<_> = ctx.required_sink.borrow().iter().cloned().collect();

    let diagnostics = SolveDiagnostics {
        required,
        assumed: vec![],
    };

    let (solution_set, raw_steps) = result?;

    // Apply didactic cleanup using opts.detailed_steps
    let cleaned = cas_solver_core::step_cleanup::cleanup_steps_by(
        &mut simplifier.context,
        raw_steps,
        opts.detailed_steps,
        "x",
        |s| cas_solver_core::step_cleanup::CleanupStep {
            description: s.description.clone(),
            equation_after: s.equation_after.clone(),
        },
        |template, payload| SolveStep {
            description: payload.description,
            equation_after: payload.equation_after,
            importance: template.importance,
            substeps: template.substeps,
        },
    );

    Ok((solution_set, DisplaySolveSteps(cleaned), diagnostics))
}

/// Solve with options but no shared context.
///
/// Creates a fresh, isolated `SolveCtx`. Conditions derived here do NOT
/// propagate to any parent context. Prefer [`solve_with_ctx`] inside
/// strategies that already hold a `&SolveCtx`.
pub(crate) fn solve_with_options(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let ctx = super::SolveCtx::default();
    solve_inner(eq, var, simplifier, opts, &ctx)
}

/// Core solver implementation.
///
/// All public entry points delegate here. `parent_ctx` carries the shared
/// accumulator so that conditions from recursive calls are visible to the
/// top-level caller.
fn solve_inner(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    parent_ctx: &super::SolveCtx,
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
    if matches!(
        classify_equation_var_presence(&simplifier.context, eq, var),
        EquationVarPresence::None
    ) {
        return Err(CasError::VariableNotFound(var.to_string()));
    }

    // V2.1 Issue #10: Extract denominators containing the variable BEFORE simplification
    // These will be used to create NonZero guards in the final result
    let domain_exclusions = cas_solver_core::solve_analysis::collect_unique_denominators_with_var(
        &simplifier.context,
        eq.lhs,
        eq.rhs,
        var,
    );

    // V2.2+: Build SolveDomainEnv with global required conditions
    // This is the "semantic ground" for all solver decisions
    let mut domain_env = SolveDomainEnv::new();
    {
        use crate::implicit_domain::{derive_requires_from_equation, infer_implicit_domain};

        // 1. Infer implicit domain from both sides
        let lhs_domain = infer_implicit_domain(&simplifier.context, eq.lhs, opts.value_domain);
        let rhs_domain = infer_implicit_domain(&simplifier.context, eq.rhs, opts.value_domain);
        domain_env.required.extend(&lhs_domain);
        domain_env.required.extend(&rhs_domain);

        // 2. Derive additional requires from equation equality
        // e.g., 2^x = sqrt(y) → since 2^x > 0, we have sqrt(y) > 0, so y > 0
        let derived = derive_requires_from_equation(
            &simplifier.context,
            eq.lhs,
            eq.rhs,
            &domain_env.required,
            opts.value_domain,
        );
        for cond in derived {
            domain_env.required.conditions_mut().insert(cond);
        }
    }

    // Push this level's conditions into the shared accumulator
    for cond in domain_env.required.conditions().iter().cloned() {
        parent_ctx.required_sink.borrow_mut().insert(cond);
    }

    // Build a level-specific SolveCtx: fresh domain_env, same shared sink
    let ctx = super::SolveCtx {
        domain_env,
        required_sink: parent_ctx.required_sink.clone(),
    };

    // EARLY CHECK: Handle rational exponent equations BEFORE simplification
    // This prevents x^(3/2) from being simplified to |x|*sqrt(x) which causes loops
    if eq.op == cas_ast::RelOp::Eq {
        if let Some(result) = try_solve_rational_exponent(eq, var, simplifier, &ctx) {
            return guard_solved_result_with_exclusions(result, &domain_exclusions);
        }
    }

    // 2. Simplify both sides BEFORE applying strategies
    // This is crucial for equations like "1/3*x + 1/2*x = 5"
    // which need to be simplified to "5/6*x = 5" before isolation
    let mut simplified_eq = {
        let mut runtime = EngineSolvePreprocessRuntime { simplifier };
        simplify_equation_sides_for_var_with_runtime(&mut runtime, eq, var)
    };

    // NOTE: Pre-solve exponent normalization (Div(p,q) → Number(p/q)) and
    // nested-pow folding are handled by global simplifier rules:
    //   - CanonicalizeRationalDivRule: Div(p,q) → Number(p/q)
    //   - CanonicalizeNestedPowRule: Pow(Pow(b,k),r) → Pow(b,k*r) [domain-safe]
    // These fire within simplify_for_solve() on both equation sides above.

    // CANCEL COMMON ADDITIVE TERMS: equation-level cancellation that compares
    // terms from LHS and RHS as a *pair*. This is a relational operation
    // between two expression trees — cannot be a local simplifier rule because
    // CanonicalizeNegationRule converts Sub→Add(Neg) before any Sub-targeting
    // rule fires. See rules/cancel_common_terms.rs for full rationale.
    //
    // PRECONDITION: both sides must already be simplified (canonical form).
    // The simplify_for_solve() calls above guarantee this.
    debug_assert!(
        !matches!(
            simplifier.context.get(simplified_eq.lhs),
            cas_ast::Expr::Sub(_, _)
        ),
        "cancel_common_terms precondition: LHS top-level is Sub (not canonical)"
    );
    debug_assert!(
        !matches!(
            simplifier.context.get(simplified_eq.rhs),
            cas_ast::Expr::Sub(_, _)
        ),
        "cancel_common_terms precondition: RHS top-level is Sub (not canonical)"
    );
    if let Some(cr) = crate::rules::cancel_common_terms::cancel_common_additive_terms(
        &mut simplifier.context,
        simplified_eq.lhs,
        simplified_eq.rhs,
    ) {
        // Re-simplify after cancellation (cheap — expression is smaller now)
        simplified_eq.lhs = simplifier.simplify_for_solve(cr.new_lhs);
        simplified_eq.rhs = simplifier.simplify_for_solve(cr.new_rhs);
    }

    // SEMANTIC CANCEL FALLBACK: for remaining unmatched terms, try
    // simplify(term_L - term_R) == 0 to catch semantically equivalent
    // terms that don't converge to the same AST under structural comparison.
    // Examples: sin(arccos(x)) vs sqrt(1-x²), abs(x^n) vs abs(x)^n.
    // Runs AFTER structural cancel to avoid redundant simplifications.
    if let Some(cr) = crate::rules::cancel_common_terms::cancel_additive_terms_semantic(
        simplifier,
        simplified_eq.lhs,
        simplified_eq.rhs,
    ) {
        simplified_eq.lhs = simplifier.simplify_for_solve(cr.new_lhs);
        simplified_eq.rhs = simplifier.simplify_for_solve(cr.new_rhs);
    }

    // CRITICAL: After simplification, check for identities and contradictions
    // Do this by moving everything to one side: LHS - RHS
    let difference = simplifier
        .context
        .add(cas_ast::Expr::Sub(simplified_eq.lhs, simplified_eq.rhs));
    // SolveSafety: use prepass for identity/contradiction check
    let mut diff_simplified = simplifier.simplify_for_solve(difference);

    // PRE-SOLVE CANCELLATION: if diff still contains the variable, try
    // expand + simplify and a trig expand fallback (Ticket 6c). Accept
    // rewrites only when they eliminate the variable or significantly
    // reduce expression size.
    let normalized_diff = {
        let mut runtime = EngineResidualRewriteRuntime { simplifier };
        normalize_variable_residual_with_runtime(&mut runtime, diff_simplified, var)
    };
    if normalized_diff != diff_simplified {
        diff_simplified = normalized_diff;
        let zero = simplifier.context.num(0);
        simplified_eq.lhs = diff_simplified;
        simplified_eq.rhs = zero;
    }

    // Check if the difference has NO variable
    if !contains_var(&simplifier.context, diff_simplified, var) {
        // Variable disappeared - this is either an identity, contradiction, or parameter-dependent.
        let reduced_outcome = resolve_var_eliminated_outcome_with(
            &mut simplifier.context,
            diff_simplified,
            var,
            |core_ctx, id| {
                format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: core_ctx,
                        id
                    }
                )
            },
        );
        match reduced_outcome {
            VarEliminatedSolveOutcome::IdentityAllReals => {
                return Ok((SolutionSet::AllReals, vec![]));
            }
            VarEliminatedSolveOutcome::ContradictionEmpty => {
                return Ok((SolutionSet::Empty, vec![]));
            }
            VarEliminatedSolveOutcome::ConstraintAllReals {
                description,
                equation_after,
            } => {
                // Variable was eliminated during simplification (e.g., x/x = 1)
                // The equation is now a constraint on OTHER variables.
                // Example: (x*y)/x = 0 simplifies to y = 0
                // Solution: x can be any value (AllReals) when the constraint holds,
                // EXCEPT values that make denominators zero.
                let steps = solve_var_eliminated_constraint_pipeline_with_item(
                    description,
                    equation_after,
                    simplifier.collect_steps(),
                    |description, equation_after| SolveStep {
                        description,
                        equation_after,
                        importance: crate::step::ImportanceLevel::Medium,
                        substeps: vec![],
                    },
                );

                // V2.1 Issue #10: Apply domain guards if denominators contained the variable
                let guarded = apply_nonzero_exclusion_guards_if_any(
                    SolutionSet::AllReals,
                    &domain_exclusions,
                );
                return Ok((guarded, steps));
            }
        }
    }

    // CYCLE DETECTION: compute fingerprint from the simplified equation and check for repetition.
    // This catches loops where strategies rewrite equations into equivalent forms.
    // We fingerprint (var, simplified_lhs, simplified_rhs) — not the diff — to avoid
    // false positives when CollectTermsStrategy moves terms between sides.
    let fp = cas_solver_core::fingerprint::equation_fingerprint(
        &simplifier.context,
        simplified_eq.lhs,
        simplified_eq.rhs,
        var,
    );
    let _cycle_guard = cas_solver_core::cycle_guard::try_enter(fp).ok_or_else(|| {
        CasError::SolverError(
            "Cycle detected: equation revisited after rewriting (equivalent form loop)".to_string(),
        )
    })?;

    // 3. Define strategies
    // In a real app, these might be configured in Simplifier or passed in.
    let strategies: Vec<Box<dyn SolverStrategy>> = vec![
        Box::new(RationalExponentStrategy), // Must run BEFORE UnwrapStrategy to avoid loops
        Box::new(SubstitutionStrategy),
        Box::new(UnwrapStrategy),
        Box::new(QuadraticStrategy),
        Box::new(RationalRootsStrategy), // Degree ≥ 3 with numeric coefficients
        Box::new(CollectTermsStrategy),  // Must run before IsolationStrategy
        Box::new(IsolationStrategy),
    ];

    // 4. Try strategies on the simplified equation
    for strategy in strategies {
        let _strategy_name = strategy.name();
        if let Some(res) = strategy.apply(&simplified_eq, var, simplifier, &opts, &ctx) {
            match res {
                Ok((result, steps)) => {
                    // Verify solutions if Discrete
                    if let SolutionSet::Discrete(sols) = result {
                        if !strategy.should_verify() {
                            return Ok((SolutionSet::Discrete(sols), steps));
                        }
                        let (symbolic_solutions, numeric_solutions) =
                            cas_solver_core::solve_analysis::partition_discrete_symbolic(
                                &simplifier.context,
                                &sols,
                            );
                        let valid_sols =
                            cas_solver_core::solve_analysis::merge_symbolic_with_verified_numeric(
                                symbolic_solutions,
                                numeric_solutions,
                                |sol| {
                                    // CRITICAL: Verify against ORIGINAL equation, not simplified
                                    // This ensures we reject solutions that cause division by zero
                                    // in the original equation, even if they work in the simplified form
                                    verify_solution(eq, var, sol, simplifier)
                                },
                            );
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
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
    ctx: &super::SolveCtx,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    let rewrite = {
        let mut runtime = EngineRationalExponentRuntime { simplifier };
        execute_rational_exponent_rewrite_with_runtime_for_var(&mut runtime, eq, var)?
    };

    let include_item = simplifier.collect_steps();
    let mut runtime = EngineRationalExponentRewriteRuntime {
        simplifier,
        original_equation: eq,
        original_var: var,
        solve_ctx: ctx,
    };
    let solved = match solve_rational_exponent_rewrite_pipeline_with_item(
        rewrite,
        var,
        include_item,
        &mut runtime,
    ) {
        Ok(solved) => solved,
        Err(e) => return Some(Err(e)),
    };
    Some(Ok((solved.solution_set, solved.steps)))
}
