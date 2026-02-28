//! Core solve dispatch pipeline.
//!
//! Contains the internal solve pipeline (`solve_inner`) and recursive
//! context-aware entrypoint used by strategy/isolation recursion.

use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::isolation::isolate;
use cas_ast::{Equation, Expr, ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_strategy::{
    execute_collect_terms_strategy_with_default_kernel_and_unified_step_mapper_with_state,
    execute_isolation_strategy_with_default_routing_and_unified_step_mapper_with_state,
    execute_rational_exponent_strategy_with_default_kernel_and_accept_all_solutions_and_unified_step_mapper_with_state,
    execute_unwrap_strategy_with_default_route_and_residual_hint_and_unified_step_mapper_with_state,
};
use cas_solver_core::isolation_utils::contains_var;
use cas_solver_core::quadratic_strategy::execute_quadratic_strategy_with_default_factorized_and_candidate_pipelines_and_unified_step_mappers_with_state;
use cas_solver_core::rational_roots::execute_rational_roots_strategy_with_default_limits_and_default_root_sorting_and_unified_step_mapper_with_state;
use cas_solver_core::solve_analysis::{
    analyze_equation_preflight_and_fork_context_with,
    derive_equation_conditions_from_existing_with, ensure_solve_entry_for_equation_or_error,
    execute_prepared_equation_strategy_pipeline_with_state, guard_solved_result_with_exclusions,
    is_soft_strategy_error_with, is_symbolic_expr, prepare_equation_for_strategy_with_state,
    resolve_discrete_strategy_result_against_equation_with_state,
    resolve_var_eliminated_residual_with_exclusions, try_enter_equation_cycle_guard_with_error,
    PreflightContext, PreparedEquationResidual,
};
use cas_solver_core::solve_budget::MAX_SOLVE_RECURSION_DEPTH;
use cas_solver_core::solve_types::cleanup_display_solve_steps;
use cas_solver_core::strategy_order::{
    default_solve_strategy_order, dispatch_solve_strategy_kind_with_state, strategy_should_verify,
    SolveStrategyKind,
};
use cas_solver_core::substitution::execute_exponential_substitution_strategy_result_pipeline_with_default_substitution_var_and_plan_with_state;

use super::{
    medium_step, DisplaySolveSteps, SolveCtx, SolveDiagnostics, SolveDomainEnv, SolveStep,
    SolveSubStep, SolverOptions,
};

type SolvePreflightState = PreflightContext<SolveCtx>;

fn simplifier_context(simplifier: &mut Simplifier) -> &cas_ast::Context {
    &simplifier.context
}

fn simplifier_context_mut(simplifier: &mut Simplifier) -> &mut cas_ast::Context {
    &mut simplifier.context
}

fn simplifier_contains_var(simplifier: &mut Simplifier, expr: ExprId, var: &str) -> bool {
    contains_var(&simplifier.context, expr, var)
}

fn simplifier_simplify_expr(simplifier: &mut Simplifier, expr: ExprId) -> ExprId {
    simplifier.simplify(expr).0
}

fn simplifier_expand_expr(simplifier: &mut Simplifier, expr: ExprId) -> ExprId {
    crate::expand::expand(&mut simplifier.context, expr)
}

fn simplifier_render_expr(simplifier: &mut Simplifier, expr: ExprId) -> String {
    cas_formatter::render_expr(&simplifier.context, expr)
}

fn simplifier_zero_expr(simplifier: &mut Simplifier) -> ExprId {
    simplifier.context.num(0)
}

fn isolation_error_detail(err: &CasError) -> Option<&str> {
    match err {
        CasError::IsolationError(_, detail) => Some(detail.as_str()),
        _ => None,
    }
}

fn solver_error_detail(err: &CasError) -> Option<&str> {
    match err {
        CasError::SolverError(detail) => Some(detail.as_str()),
        _ => None,
    }
}

fn is_soft_strategy_error(err: &CasError) -> bool {
    is_soft_strategy_error_with(err, isolation_error_detail, solver_error_detail)
}

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

/// V2.9.8: Solve with type-enforced display-ready steps.
///
/// This is the PREFERRED entry point for display-facing code (REPL, timeline, JSON API).
/// Returns `DisplaySolveSteps` which enforces that all renderers consume post-processed
/// steps, eliminating bifurcation between text/timeline outputs at compile time.
///
/// The cleanup is applied automatically based on `opts.detailed_steps`:
/// - `true` -> 5 atomic sub-steps for Normal/Verbose verbosity
/// - `false` -> 3 compact steps for Succinct verbosity
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

    let diagnostics = ctx.diagnostics_with_records(crate::assumptions::collect_assumption_records);

    let (solution_set, raw_steps) = result?;

    // Apply didactic cleanup using opts.detailed_steps and enforce DisplaySteps output.
    let cleaned =
        cleanup_display_solve_steps(&mut simplifier.context, raw_steps, opts.detailed_steps, var);

    Ok((solution_set, cleaned, diagnostics))
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

// NOTE: Pre-solve exponent normalization (Div(p,q) → Number(p/q)) and
// common additive term cancellation (Sub(Add(A,B), B) → A) were moved
// from solver-only code to global simplifier rules:
//   - rules/rational_canonicalization.rs (CanonicalizeRationalDivRule, CanonicalizeNestedPowRule)
//   - rules/cancel_common_terms.rs (CancelCommonAdditiveTermsRule)
// These are now applied automatically by simplify_for_solve().

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

/// Core solver implementation.
///
/// All public entry points delegate here. `parent_ctx` carries the shared
/// accumulator so that conditions from recursive calls are visible to the
/// top-level caller.
pub(super) fn solve_inner(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    parent_ctx: &super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let current_depth = parent_ctx.depth().saturating_add(1);
    ensure_solve_entry_for_equation_or_error(
        &simplifier.context,
        eq,
        var,
        current_depth,
        MAX_SOLVE_RECURSION_DEPTH,
        || {
            CasError::SolverError(
                "Maximum solver recursion depth exceeded. The equation may be too complex."
                    .to_string(),
            )
        },
        || CasError::VariableNotFound(var.to_string()),
    )?;

    let preflight = build_solve_preflight_state(simplifier, eq, var, opts.value_domain, parent_ctx);
    let domain_exclusions = preflight.domain_exclusions;
    let ctx = preflight.ctx;

    // EARLY CHECK: Handle rational exponent equations BEFORE simplification
    // This prevents x^(3/2) from being simplified to |x|*sqrt(x) which causes loops
    if let Some(result) = try_rational_exponent_precheck(eq, var, simplifier, &opts, &ctx) {
        return guard_solved_result_with_exclusions(result, &domain_exclusions);
    }

    // 2) Pre-strategy equation normalization:
    // simplify var-containing sides, cancel common additive terms, and
    // normalize residual fallbacks before running strategy dispatch.
    let prepared = prepare_equation_for_strategy(simplifier, eq, var);
    let simplified_eq = prepared.equation;
    let diff_simplified = prepared.residual;

    // NOTE: Pre-solve exponent normalization (Div(p,q) → Number(p/q)),
    // nested-pow folding, and additive cancellation are applied above.
    debug_assert_no_top_level_sub(simplifier, &simplified_eq);

    // 3) Resolve var-eliminated residuals early, otherwise guard cycle + run strategies.
    execute_strategy_pipeline(
        simplifier,
        eq,
        &simplified_eq,
        diff_simplified,
        var,
        opts,
        &ctx,
        &domain_exclusions,
    )
}

/// Build per-level preflight state:
/// - domain exclusions from equation structure
/// - required conditions applied to domain env and shared sink
/// - child solve context with incremented depth
fn build_solve_preflight_state(
    simplifier: &Simplifier,
    eq: &Equation,
    var: &str,
    value_domain: crate::semantics::ValueDomain,
    parent_ctx: &SolveCtx,
) -> SolvePreflightState {
    analyze_equation_preflight_and_fork_context_with(
        &simplifier.context,
        eq,
        var,
        value_domain,
        parent_ctx,
        |expr, eval_domain| {
            crate::implicit_domain::infer_implicit_domain(&simplifier.context, expr, eval_domain)
                .conditions()
                .iter()
                .cloned()
                .collect::<Vec<_>>()
        },
        |lhs, rhs, existing, eval_domain| {
            derive_equation_conditions_from_existing_with(
                lhs,
                rhs,
                existing,
                eval_domain,
                crate::implicit_domain::ImplicitDomain::empty,
                |domain, cond| {
                    domain.conditions_mut().insert(cond);
                },
                |lhs, rhs, domain, eval_domain| {
                    crate::implicit_domain::derive_requires_from_equation(
                        &simplifier.context,
                        lhs,
                        rhs,
                        domain,
                        eval_domain,
                    )
                },
            )
        },
        SolveDomainEnv::new(),
        |domain_env, cond| {
            domain_env.required.conditions_mut().insert(cond.clone());
        },
    )
}

fn prepare_equation_for_strategy(
    simplifier: &mut Simplifier,
    equation: &Equation,
    var: &str,
) -> PreparedEquationResidual {
    prepare_equation_for_strategy_with_state(
        simplifier,
        equation,
        var,
        simplifier_contains_var,
        |state, expr| state.simplify_for_solve(expr),
        |state, expr| {
            cas_solver_core::isolation_utils::try_recompose_pow_quotient(&mut state.context, expr)
        },
        |state, lhs, rhs| {
            crate::rules::cancel_common_terms::cancel_common_additive_terms(
                &mut state.context,
                lhs,
                rhs,
            )
            .map(|rewrite| (rewrite.new_lhs, rewrite.new_rhs))
        },
        |state, lhs, rhs| {
            crate::rules::cancel_common_terms::cancel_additive_terms_semantic(state, lhs, rhs)
                .map(|rewrite| (rewrite.new_lhs, rewrite.new_rhs))
        },
        |state, lhs, rhs| state.context.add(Expr::Sub(lhs, rhs)),
        simplifier_expand_expr,
        |state, expr| state.expand(expr).0,
        |state, current, candidate, var_name| {
            cas_solver_core::solve_analysis::accept_residual_rewrite_candidate(
                &state.context,
                current,
                candidate,
                var_name,
            )
        },
        simplifier_zero_expr,
    )
}

fn debug_assert_no_top_level_sub(simplifier: &Simplifier, equation: &Equation) {
    debug_assert!(
        !matches!(
            simplifier.context.get(equation.lhs),
            cas_ast::Expr::Sub(_, _)
        ),
        "cancel_common_terms precondition: LHS top-level is Sub (not canonical)"
    );
    debug_assert!(
        !matches!(
            simplifier.context.get(equation.rhs),
            cas_ast::Expr::Sub(_, _)
        ),
        "cancel_common_terms precondition: RHS top-level is Sub (not canonical)"
    );
}

#[allow(clippy::too_many_arguments)]
fn execute_strategy_pipeline(
    simplifier: &mut Simplifier,
    original_eq: &Equation,
    simplified_eq: &Equation,
    diff_simplified: ExprId,
    var: &str,
    opts: SolverOptions,
    ctx: &SolveCtx,
    domain_exclusions: &[ExprId],
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let strategies = default_solve_strategy_order();
    execute_prepared_equation_strategy_pipeline_with_state(
        simplifier,
        simplified_eq,
        diff_simplified,
        var,
        strategies,
        simplifier_contains_var,
        |simplifier, residual, var_name| {
            let include_item = simplifier.collect_steps();
            Ok(resolve_var_eliminated_residual_with_exclusions(
                &mut simplifier.context,
                residual,
                var_name,
                include_item,
                domain_exclusions,
                cas_formatter::render_expr,
                super::medium_step,
            ))
        },
        |simplifier, equation, var_name| {
            try_enter_equation_cycle_guard_with_error(
                &simplifier.context,
                equation,
                var_name,
                || {
                    CasError::SolverError(
                        "Cycle detected: equation revisited after rewriting (equivalent form loop)"
                            .to_string(),
                    )
                },
            )
        },
        |simplifier, strategy_kind| {
            let should_verify = strategy_should_verify(*strategy_kind);
            let attempt =
                apply_strategy(*strategy_kind, simplified_eq, var, simplifier, &opts, ctx);
            (attempt, should_verify)
        },
        is_soft_strategy_error,
        |simplifier, solutions, steps| {
            resolve_discrete_strategy_result_against_equation_with_state(
                simplifier,
                original_eq,
                var,
                solutions,
                steps,
                |state, solution| is_symbolic_expr(&state.context, solution),
                |state, equation, var_name, solution| {
                    // Verify against ORIGINAL equation, not simplified form, so
                    // domain-invalid roots (e.g. division by zero) are rejected.
                    cas_solver_core::verify_substitution::verify_solution_with_state(
                        state,
                        equation,
                        var_name,
                        solution,
                        |state, equation, var_name, candidate| {
                            cas_solver_core::verify_substitution::substitute_equation_sides(
                                &mut state.context,
                                equation,
                                var_name,
                                candidate,
                            )
                        },
                        |state, expr| state.simplify(expr).0,
                        |state, lhs, rhs| state.are_equivalent(lhs, rhs),
                    )
                },
            )
        },
        CasError::SolverError("No strategy could solve this equation.".to_string()),
    )
}

/// Apply one strategy selected by kind.
fn apply_strategy(
    kind: SolveStrategyKind,
    equation: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: &SolverOptions,
    ctx: &SolveCtx,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    dispatch_solve_strategy_kind_with_state(
        simplifier,
        kind,
        |simplifier| apply_rational_exponent_strategy(equation, var, simplifier, opts, ctx),
        |simplifier| apply_substitution_strategy(equation, var, simplifier, opts, ctx),
        |simplifier| apply_unwrap_strategy(equation, var, simplifier, opts, ctx),
        |simplifier| apply_quadratic_strategy(equation, var, simplifier, opts, ctx),
        |simplifier| apply_rational_roots_strategy(equation, var, simplifier),
        |simplifier| apply_collect_terms_strategy(equation, var, simplifier, opts, ctx),
        |simplifier| apply_isolation_strategy(equation, var, simplifier, opts, ctx),
    )
}

fn apply_isolation_strategy(
    equation: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: &SolverOptions,
    ctx: &SolveCtx,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    let include_item = simplifier.collect_steps();
    execute_isolation_strategy_with_default_routing_and_unified_step_mapper_with_state(
        simplifier,
        equation,
        var,
        include_item,
        simplifier_context,
        |simplifier, next_eq, solve_var| {
            isolate(
                next_eq.lhs,
                next_eq.rhs,
                next_eq.op.clone(),
                solve_var,
                simplifier,
                *opts,
                ctx,
            )
        },
        medium_step,
        |missing_var| CasError::VariableNotFound(missing_var.to_string()),
    )
}

fn apply_collect_terms_strategy(
    equation: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: &SolverOptions,
    ctx: &SolveCtx,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    let include_item = simplifier.collect_steps();
    execute_collect_terms_strategy_with_default_kernel_and_unified_step_mapper_with_state(
        simplifier,
        equation,
        var,
        include_item,
        simplifier_context_mut,
        simplifier_simplify_expr,
        simplifier_render_expr,
        |simplifier, next_eq, solve_var| {
            solve_with_ctx_and_options(next_eq, solve_var, simplifier, *opts, ctx)
        },
        medium_step,
    )
}

fn apply_unwrap_strategy(
    equation: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: &SolverOptions,
    ctx: &SolveCtx,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    let mode = opts.core_domain_mode();
    let wildcard_scope = opts.wildcard_scope();

    let include_item = simplifier.collect_steps();
    execute_unwrap_strategy_with_default_route_and_residual_hint_and_unified_step_mapper_with_state(
        simplifier,
        equation,
        var,
        include_item,
        simplifier_context_mut,
        mode,
        wildcard_scope,
        |core_ctx, base, other_side| {
            crate::solver::classify_log_solve(core_ctx, base, other_side, opts, &ctx.domain_env)
        },
        cas_formatter::render_expr,
        |simplifier, record| {
            ctx.note_assumption(crate::assumptions::assumption_event_from_log_assumption(
                &simplifier.context,
                record.assumption,
                record.base,
                record.other_side,
            ));
        },
        |simplifier, next_eq, solve_var| {
            solve_with_ctx_and_options(next_eq, solve_var, simplifier, *opts, ctx)
        },
        medium_step,
    )
}

fn apply_rational_exponent_strategy(
    equation: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: &SolverOptions,
    ctx: &SolveCtx,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    let include_item = simplifier.collect_steps();
    execute_rational_exponent_strategy_with_default_kernel_and_accept_all_solutions_and_unified_step_mapper_with_state(
        simplifier,
        equation,
        var,
        include_item,
        simplifier_context_mut,
        simplifier_simplify_expr,
        |simplifier, equation, solve_var| {
            solve_with_ctx_and_options(equation, solve_var, simplifier, *opts, ctx)
        },
        medium_step,
    )
}

/// Run the early rational-exponent precheck (`x^(p/q) = rhs`) before generic
/// simplification to avoid fractional-power rewrite loops.
fn try_rational_exponent_precheck(
    equation: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: &SolverOptions,
    ctx: &SolveCtx,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    if equation.op != RelOp::Eq {
        return None;
    }

    apply_rational_exponent_strategy(equation, var, simplifier, opts, ctx)
}

fn apply_substitution_strategy(
    equation: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: &SolverOptions,
    ctx: &SolveCtx,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    let include_didactic_items = simplifier.collect_steps();
    execute_exponential_substitution_strategy_result_pipeline_with_default_substitution_var_and_plan_with_state(
        simplifier,
        equation,
        var,
        include_didactic_items,
        simplifier_context_mut,
        simplifier_render_expr,
        |simplifier, next_eq, solve_var| {
            solve_with_ctx_and_options(next_eq, solve_var, simplifier, *opts, ctx)
        },
        medium_step,
    )
}

fn apply_quadratic_strategy(
    equation: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: &SolverOptions,
    ctx: &SolveCtx,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    let include_items = simplifier.collect_steps();
    let is_real_only = matches!(opts.value_domain, crate::semantics::ValueDomain::RealOnly);
    execute_quadratic_strategy_with_default_factorized_and_candidate_pipelines_and_unified_step_mappers_with_state(
        simplifier,
        equation,
        var,
        include_items,
        is_real_only,
        simplifier_context,
        simplifier_context_mut,
        |simplifier, collecting| simplifier.set_collect_steps(collecting),
        simplifier_simplify_expr,
        simplifier_expand_expr,
        cas_formatter::render_expr,
        |simplifier, next_eq| solve_with_ctx_and_options(next_eq, var, simplifier, *opts, ctx),
        |description, next_eq, substeps: Option<Vec<SolveSubStep>>| {
            let step = medium_step(description, next_eq);
            if let Some(substeps) = substeps {
                step.with_substeps(substeps)
            } else {
                step
            }
        },
        |description, next_eq| SolveSubStep {
            description,
            equation_after: next_eq,
            importance: crate::step::ImportanceLevel::Low,
        },
        |_| {
            CasError::SolverError(
                "Inequalities with symbolic coefficients not yet supported".to_string(),
            )
        },
        |_| {
            ctx.emit_scope(cas_formatter::display_transforms::ScopeTag::Rule(
                "QuadraticFormula",
            ));
        },
    )
}

fn apply_rational_roots_strategy(
    equation: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    let include_item = simplifier.collect_steps();
    let solved = execute_rational_roots_strategy_with_default_limits_and_default_root_sorting_and_unified_step_mapper_with_state(
        simplifier,
        equation,
        var,
        include_item,
        simplifier_context_mut,
        simplifier_simplify_expr,
        simplifier_expand_expr,
        medium_step,
    )?;
    Some(Ok(solved))
}

#[cfg(test)]
mod tests {
    use super::build_solve_preflight_state;
    use crate::engine::Simplifier;
    use crate::solver::SolveCtx;
    use cas_ast::{Equation, RelOp};
    use cas_parser::parse;

    #[test]
    fn build_solve_preflight_state_forks_ctx_with_next_depth() {
        let mut simplifier = Simplifier::new();
        let eq = Equation {
            lhs: parse("x", &mut simplifier.context).unwrap(),
            rhs: parse("0", &mut simplifier.context).unwrap(),
            op: RelOp::Eq,
        };

        let parent = SolveCtx::default();
        let out = build_solve_preflight_state(
            &simplifier,
            &eq,
            "x",
            crate::semantics::ValueDomain::RealOnly,
            &parent,
        );
        assert_eq!(out.ctx.depth(), 1);
    }

    #[test]
    fn build_solve_preflight_state_notes_required_conditions_to_parent_sink() {
        let mut simplifier = Simplifier::new();
        let eq = Equation {
            lhs: parse("sqrt(x)", &mut simplifier.context).unwrap(),
            rhs: parse("0", &mut simplifier.context).unwrap(),
            op: RelOp::Eq,
        };

        let parent = SolveCtx::default();
        let _out = build_solve_preflight_state(
            &simplifier,
            &eq,
            "x",
            crate::semantics::ValueDomain::RealOnly,
            &parent,
        );

        // sqrt(x) imposes x >= 0 in real domain; ensure at least one
        // required condition is propagated into the shared sink.
        let snapshot = parent.snapshot();
        assert!(
            !snapshot.required.is_empty(),
            "expected required conditions from sqrt(x) preflight",
        );
    }
}
