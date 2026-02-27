//! Core solve dispatch pipeline.
//!
//! Contains the main `solve`, `solve_with_options`, and `solve_with_display_steps`
//! entry points, plus the rational-exponent pre-check.

use super::SolveDiagnostics;
use crate::engine::Simplifier;
use crate::error::CasError;
use cas_ast::SolutionSet;
use cas_solver_core::isolation_utils::contains_var;
use cas_solver_core::solve_analysis::{
    apply_equation_pair_rewrite_sequence_with_state, apply_nonzero_exclusion_guards_if_any,
    classify_equation_var_presence, finalize_strategy_attempt_sequence_with,
    guard_solved_result_with_exclusions, normalize_variable_residual_with_state,
    resolve_discrete_strategy_solutions_for_context_with, run_strategy_attempt_sequence,
    simplify_equation_sides_for_presence_with_state, EquationVarPresence,
};
use cas_solver_core::solve_outcome::{
    solve_var_eliminated_outcome_pipeline_with, VarEliminatedOutcomePipelineSolved,
};
use cas_solver_core::step_cleanup::{cleanup_steps_by, CleanupStep};

use super::{
    medium_step, render_expr, DisplaySolveSteps, SolveDomainEnv, SolveStep, SolverOptions,
    MAX_SOLVE_DEPTH,
};

// NOTE: Pre-solve exponent normalization (Div(p,q) → Number(p/q)) and
// common additive term cancellation (Sub(Add(A,B), B) → A) were moved
// from solver-only code to global simplifier rules:
//   - rules/rational_canonicalization.rs (CanonicalizeRationalDivRule, CanonicalizeNestedPowRule)
//   - rules/cancel_common_terms.rs (CancelCommonAdditiveTermsRule)
// These are now applied automatically by simplify_for_solve().

/// Solve with default options (for backward compatibility with tests).
/// Uses RealOnly domain and Generic mode.
///
/// This creates a fresh `SolveCtx`; conditions are NOT propagated
/// to any parent context. For recursive calls from strategies that
/// need to accumulate conditions, use [`solve_with_ctx_and_options`] instead.
pub fn solve(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    solve_with_options(eq, var, simplifier, SolverOptions::default())
}

/// Solve with a shared `SolveCtx` and explicit options.
///
/// This should be used by recursive strategy/isolation paths so nested solves
/// preserve semantic/domain options from the top-level invocation.
pub(crate) fn solve_with_ctx_and_options(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    ctx: &super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    solve_inner(eq, var, simplifier, opts, ctx)
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
    // through solve_with_ctx_and_options will push conditions into this shared set.
    let ctx = super::SolveCtx::default();
    let result = solve_inner(eq, var, simplifier, opts, &ctx);

    // Read accumulated conditions from the shared sink (zero TLS)
    let required = ctx.required_conditions();
    let assumed = ctx.assumptions();
    let assumed_records = {
        let mut collector = crate::assumptions::AssumptionCollector::new();
        for event in assumed.iter().cloned() {
            collector.note(event);
        }
        collector.finish()
    };

    let diagnostics = SolveDiagnostics {
        required,
        assumed,
        assumed_records,
        output_scopes: ctx.output_scopes(),
    };

    let (solution_set, raw_steps) = result?;

    // Apply didactic cleanup using opts.detailed_steps
    let cleaned = cleanup_steps_by(
        &mut simplifier.context,
        raw_steps,
        opts.detailed_steps,
        var,
        |step: &SolveStep| CleanupStep {
            description: step.description.clone(),
            equation_after: step.equation_after.clone(),
        },
        |template: SolveStep, payload: CleanupStep| SolveStep {
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
/// propagate to any parent context. Prefer [`solve_with_ctx_and_options`] inside
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

/// Errors that represent a dead-end for one strategy but should not abort
/// the whole strategy pipeline. This lets later strategies attempt recovery.
fn is_soft_strategy_error(err: &CasError) -> bool {
    match err {
        CasError::IsolationError(_, detail) => detail.contains("variable appears on both sides"),
        CasError::SolverError(detail) => detail.contains("Cycle detected"),
        _ => false,
    }
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
    let current_depth = parent_ctx.depth().saturating_add(1);

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
        parent_ctx.note_required_condition(cond);
    }

    // Build a level-specific SolveCtx: fresh domain_env, same shared sinks.
    let ctx = parent_ctx.fork_with_domain_env_next_depth(domain_env);

    // EARLY CHECK: Handle rational exponent equations BEFORE simplification
    // This prevents x^(3/2) from being simplified to |x|*sqrt(x) which causes loops
    if let Some(result) =
        super::strategies::try_rational_exponent_precheck(eq, var, simplifier, &opts, &ctx)
    {
        return guard_solved_result_with_exclusions(result, &domain_exclusions);
    }

    // 2. Simplify both sides BEFORE applying strategies
    // This is crucial for equations like "1/3*x + 1/2*x = 5"
    // which need to be simplified to "5/6*x = 5" before isolation
    let lhs_has_var = contains_var(&simplifier.context, eq.lhs, var);
    let rhs_has_var = contains_var(&simplifier.context, eq.rhs, var);
    let mut simplified_eq = simplify_equation_sides_for_presence_with_state(
        simplifier,
        eq,
        lhs_has_var,
        rhs_has_var,
        |simplifier, expr| simplifier.simplify_for_solve(expr),
        |simplifier, expr| {
            cas_solver_core::isolation_utils::try_recompose_pow_quotient(
                &mut simplifier.context,
                expr,
            )
        },
    );

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
    // SEMANTIC CANCEL FALLBACK: for remaining unmatched terms, try
    // simplify(term_L - term_R) == 0 to catch semantically equivalent
    // terms that don't converge to the same AST under structural comparison.
    // Examples: sin(arccos(x)) vs sqrt(1-x²), abs(x^n) vs abs(x)^n.
    // Runs AFTER structural cancel to avoid redundant simplifications.
    let (lhs, rhs) = apply_equation_pair_rewrite_sequence_with_state(
        simplifier,
        simplified_eq.lhs,
        simplified_eq.rhs,
        |simplifier, lhs, rhs| {
            crate::rules::cancel_common_terms::cancel_common_additive_terms(
                &mut simplifier.context,
                lhs,
                rhs,
            )
            .map(|rewrite| (rewrite.new_lhs, rewrite.new_rhs))
        },
        |simplifier, lhs, rhs| {
            crate::rules::cancel_common_terms::cancel_additive_terms_semantic(simplifier, lhs, rhs)
                .map(|rewrite| (rewrite.new_lhs, rewrite.new_rhs))
        },
        |simplifier, expr| simplifier.simplify_for_solve(expr),
    );
    simplified_eq.lhs = lhs;
    simplified_eq.rhs = rhs;

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
    let normalized_diff = normalize_variable_residual_with_state(
        simplifier,
        diff_simplified,
        var,
        |simplifier, expr, var_name| contains_var(&simplifier.context, expr, var_name),
        |simplifier, expr| crate::expand::expand(&mut simplifier.context, expr),
        |simplifier, expr| simplifier.simplify_for_solve(expr),
        |simplifier, expr| simplifier.expand(expr).0,
        |simplifier, current, candidate, var_name| {
            cas_solver_core::solve_analysis::accept_residual_rewrite_candidate(
                &simplifier.context,
                current,
                candidate,
                var_name,
            )
        },
    );
    if normalized_diff != diff_simplified {
        diff_simplified = normalized_diff;
        let zero = simplifier.context.num(0);
        simplified_eq.lhs = diff_simplified;
        simplified_eq.rhs = zero;
    }

    // Check if the difference has NO variable
    if !contains_var(&simplifier.context, diff_simplified, var) {
        let include_item = simplifier.collect_steps();
        let reduced_outcome = solve_var_eliminated_outcome_pipeline_with(
            &mut simplifier.context,
            diff_simplified,
            var,
            include_item,
            render_expr,
            medium_step,
        );
        match reduced_outcome {
            VarEliminatedOutcomePipelineSolved::IdentityAllReals => {
                return Ok((SolutionSet::AllReals, vec![]));
            }
            VarEliminatedOutcomePipelineSolved::ContradictionEmpty => {
                return Ok((SolutionSet::Empty, vec![]));
            }
            VarEliminatedOutcomePipelineSolved::ConstraintAllReals { steps } => {
                // Variable was eliminated during simplification (e.g., x/x = 1)
                // The equation is now a constraint on OTHER variables.
                // Example: (x*y)/x = 0 simplifies to y = 0
                // Solution: x can be any value (AllReals) when the constraint holds,
                // EXCEPT values that make denominators zero.
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
    let strategies = super::strategies::default_strategies();

    // 4. Try strategies on the simplified equation
    let strategy_attempts = strategies.into_iter().map(|strategy| {
        let _strategy_name = strategy.name();
        (
            strategy.apply(&simplified_eq, var, simplifier, &opts, &ctx),
            strategy.should_verify(),
        )
    });

    finalize_strategy_attempt_sequence_with(
        run_strategy_attempt_sequence(strategy_attempts, is_soft_strategy_error),
        |solutions, steps| {
            let classify_ctx = simplifier.context.clone();
            let valid_sols = resolve_discrete_strategy_solutions_for_context_with(
                &classify_ctx,
                solutions,
                |solution| {
                    // Verify against ORIGINAL equation, not simplified form, so
                    // domain-invalid roots (e.g. division by zero) are rejected.
                    super::check::verify_solution_by_equivalence(simplifier, eq, var, solution)
                },
            );
            (SolutionSet::Discrete(valid_sols), steps)
        },
        CasError::SolverError("No strategy could solve this equation.".to_string()),
    )
}
