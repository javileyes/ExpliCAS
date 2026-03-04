//! Solver facade crate.
//!
//! During migration this crate hosts the solver entry points while still
//! re-exporting selected `cas_engine` APIs for compatibility.

mod assumption_format;
mod blocked_hint_format;
mod const_fold_local;
mod eval_output_adapters;
mod json;
mod linear_system;
#[cfg(test)]
mod linear_system_tests;
mod path_rewrite;
#[cfg(test)]
mod path_rewrite_tests;
mod pipeline_display;
#[cfg(test)]
mod pipeline_display_tests;
mod solution_display;
mod solve_backend;
mod solve_safety;
pub mod substitute;
#[cfg(test)]
mod substitute_tests;
mod symbolic_transforms;
mod telescoping;
mod types;
mod unary_display;

/// Backward-compatible facade for former `cas_engine::strategies::substitute_expr` imports.
pub mod strategies {
    pub use cas_ast::substitute_expr_by_id as substitute_expr;
}

/// Backward-compatible facade for former `cas_engine::api::*` imports.
pub mod api {
    pub use cas_ast::{
        BoundType, Case, ConditionPredicate, ConditionSet, Interval, SolutionSet, SolveResult,
    };
    pub use cas_formatter::{DisplayExpr, LaTeXExpr};
    pub use cas_solver_core::solve_budget::SolveBudget;

    pub use crate::{
        contains_var, infer_solve_variable, solve, solve_with_display_steps, verify_solution,
        verify_solution_set, verify_stats, DisplaySolveSteps, SolveDiagnostics, SolveStep,
        SolveSubStep, SolverOptions, VerifyResult, VerifyStatus, VerifySummary,
    };
}

pub use assumption_format::{
    collect_assumed_conditions_from_steps, format_assumed_conditions_report_lines,
    format_assumption_records_summary, format_blocked_hint_lines,
    format_diagnostics_requires_lines, format_displayable_assumption_lines,
    format_displayable_assumption_lines_for_step, format_displayable_assumption_lines_grouped,
    format_displayable_assumption_lines_grouped_for_step, format_domain_warning_lines,
    format_normalized_condition_lines, format_required_condition_lines,
    group_assumed_conditions_by_rule,
};
pub use blocked_hint_format::{
    filter_blocked_hints_for_eval, format_eval_blocked_hints_lines,
    format_solve_assumption_and_blocked_sections, SolveAssumptionSectionConfig,
};
pub use cas_ast::ordering::compare_expr;
pub use cas_ast::target_kind;
pub use cas_engine::error;
pub use cas_engine::expand;
pub use cas_engine::rules;
pub use cas_engine::rules::logarithms::LogExpansionRule;
pub use cas_engine::ImportanceLevel;
pub use cas_engine::Orchestrator;
pub use cas_engine::ParentContext;
pub use cas_engine::Rewrite;
pub use cas_engine::Rule;
pub use cas_engine::SharedSemanticConfig;
pub use cas_engine::SimpleRule;
pub use cas_engine::{
    AutoExpandBinomials, BranchMode, Budget, CasError, ComplexMode, ContextMode, DisplayEvalSteps,
    Engine, EvalAction, EvalOptions, EvalOutput, EvalRequest, EvalResult, HeuristicPoly, Metric,
    Operation, PassStats, PathStep, PipelineStats, RuleProfiler, Simplifier, SimplifyOptions, Step,
    StepCategory, StepsMode,
};
pub use cas_engine::{BudgetExceeded, StandardOracle};
pub use cas_engine::{ExpandBudget, PhaseBudgets, PhaseMask, PhaseStats};
pub use cas_engine::{ExpandPolicy, SimplifyPhase};
pub use cas_formatter::visualizer;
pub use cas_math::canonical_forms;
pub use cas_math::evaluator_f64::{
    eval_f64, eval_f64_checked, EvalCheckedError, EvalCheckedOptions,
};
pub use cas_math::expr_nary::{add_terms_no_sign, add_terms_signed, Sign};
pub use cas_math::expr_predicates::is_zero_expr as is_zero;
pub use cas_math::factor::factor;
pub use cas_math::limit_types::{Approach, LimitOptions, PreSimplifyMode};
pub use cas_math::number_theory_support::GcdResult;
pub use cas_math::pattern_marks;
pub use cas_math::poly_store::{try_get_poly_result_term_count, try_render_poly_result};
pub use cas_math::rationalize::{rationalize_denominator, RationalizeConfig, RationalizeResult};
pub use cas_math::rationalize_policy::{AutoRationalizeLevel, RationalizeOutcome};
pub use cas_math::telescoping_dirichlet::{
    try_dirichlet_kernel_identity as try_dirichlet_kernel_identity_pub, DirichletKernelResult,
};
pub use cas_session_core::eval::{EvalSession, EvalStore};
pub use cas_solver_core::assume_scope::AssumeScope;
pub use cas_solver_core::assumption_model::AssumptionRecord;
pub use cas_solver_core::assumption_model::{
    assumption_condition_text, assumption_key_dedupe_fingerprint, blocked_hint_suggestion,
    collect_assumption_records, collect_assumption_records_from_iter, collect_blocked_hint_items,
    format_assumption_records_conditions, format_assumption_records_section_lines,
    format_blocked_hint_condition, format_blocked_simplifications_section_lines,
    group_blocked_hint_conditions_by_rule, AssumptionCollector, AssumptionEvent, AssumptionKey,
    AssumptionKind,
};
pub use cas_solver_core::assumption_reporting::AssumptionReporting;
pub use cas_solver_core::blocked_hint::BlockedHint;
pub use cas_solver_core::blocked_hint_store::{
    clear_blocked_hints, register_blocked_hint, take_blocked_hints,
};
pub use cas_solver_core::const_fold_types::{ConstFoldMode, ConstFoldResult};
pub use cas_solver_core::diagnostics_model::{Diagnostics, RequireOrigin, RequiredItem};
pub use cas_solver_core::domain_assumption_classification::classify_assumption;
pub use cas_solver_core::domain_condition::{
    filter_requires_for_display, ImplicitCondition, ImplicitDomain, RequiresDisplayLevel,
};
pub use cas_solver_core::domain_context::DomainContext;
pub use cas_solver_core::domain_facts_model::{DomainFact, FactStrength, Predicate};
pub use cas_solver_core::domain_inference::{AnalyticExpansionResult, DomainDelta};
pub use cas_solver_core::domain_inference_counter::{
    get as infer_domain_calls_get, reset as infer_domain_calls_reset,
};
pub use cas_solver_core::domain_mode::DomainMode;
pub use cas_solver_core::domain_normalization::{
    normalize_and_dedupe_conditions, normalize_condition, normalize_condition_expr,
    render_conditions_normalized,
};
pub use cas_solver_core::domain_oracle_model::DomainOracle;
pub use cas_solver_core::domain_proof::Proof;
pub use cas_solver_core::domain_warning::DomainWarning;
pub use cas_solver_core::engine_events::{EngineEvent, StepListener};
pub use cas_solver_core::equivalence::EquivalenceResult;
pub use cas_solver_core::eval_config::EvalConfig;
pub use cas_solver_core::inverse_trig_policy::InverseTrigPolicy;
pub use cas_solver_core::isolation_utils::contains_var;
pub use cas_solver_core::solve_budget::SolveBudget;
pub use cas_solver_core::solve_infer::infer_solve_variable;
pub use cas_solver_core::solve_safety_policy::ConditionClass;
pub use cas_solver_core::solve_safety_policy::ProvenanceKind as Provenance;
pub use cas_solver_core::solve_safety_policy::SimplifyPurpose;
pub use cas_solver_core::verification::{VerifyResult, VerifyStatus, VerifySummary};
pub use cas_solver_core::verify_stats;
pub use cas_solver_core::{branch_policy::BranchPolicy, value_domain::ValueDomain};
pub use eval_output_adapters::{
    assumption_records_from_eval_output, blocked_hints_from_eval_output,
    diagnostics_from_eval_output, domain_warnings_from_eval_output, eval_output_view,
    output_scopes_from_eval_output, parsed_expr_from_eval_output,
    required_conditions_from_eval_output, resolved_expr_from_eval_output, result_from_eval_output,
    solve_steps_from_eval_output, steps_from_eval_output, stored_id_from_eval_output,
    EvalOutputView,
};
pub use json::{
    eval_str_to_json, eval_str_to_output_envelope, evaluate_envelope_json_command,
    map_domain_warnings_to_engine_warnings, map_solver_assumptions_to_api_records,
    substitute_str_to_json,
};
pub use linear_system::{
    solve_2x2_linear_system, solve_3x3_linear_system, solve_nxn_linear_system, LinSolveResult,
    LinearSystemError,
};
pub use path_rewrite::reconstruct_global_expr;
pub use pipeline_display::{display_expr_or_poly, format_pipeline_stats};
pub use solution_display::{display_interval, display_solution_set, is_pure_residual_otherwise};
pub use solve_safety::{RequirementDescriptor, RuleSolveSafetyExt, SolveSafety};
pub use substitute::{
    detect_substitute_strategy, substitute_auto, substitute_auto_with_strategy,
    substitute_power_aware, substitute_with_steps, SubstituteOptions, SubstituteStrategy,
};
pub use symbolic_transforms::{apply_weierstrass_recursive, expand_log_recursive};
pub use telescoping::{telescope, TelescopingResult, TelescopingStep};
pub use types::{
    DisplaySolveSteps, SolveCtx, SolveDiagnostics, SolveDomainEnv, SolveStep, SolveSubStep,
    SolverOptions,
};
pub use unary_display::format_unary_function_eval_lines;

/// Result shape for equation-level additive cancellation.
pub type CancelResult = cas_solver_core::cancel_common_terms::CancelResult;

/// Result of symbolic limit evaluation from solver facade.
#[derive(Debug, Clone)]
pub struct LimitResult {
    /// The computed limit expression (or residual `limit(...)` when unresolved).
    pub expr: cas_ast::ExprId,
    /// Steps emitted by limit evaluation (when requested).
    pub steps: Vec<Step>,
    /// Warning emitted when limit cannot be determined safely.
    pub warning: Option<String>,
}

/// Solve an equation for a variable.
pub fn solve(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Result<(cas_ast::SolutionSet, Vec<SolveStep>), CasError> {
    let ctx = SolveCtx::default();
    solve_backend::solve_with_engine_backend(
        eq,
        var,
        simplifier,
        SolverOptions::default().to_core(),
        &ctx,
    )
}

/// Solve with display-ready steps and diagnostics.
pub fn solve_with_display_steps(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
) -> Result<(cas_ast::SolutionSet, DisplaySolveSteps, SolveDiagnostics), CasError> {
    let ctx = SolveCtx::default();
    let result =
        solve_backend::solve_with_engine_backend(eq, var, simplifier, opts.to_core(), &ctx);
    cas_solver_core::solve_types::finalize_display_solve_with_ctx(
        &ctx,
        result,
        crate::collect_assumption_records,
        |raw_steps| {
            cas_solver_core::solve_types::cleanup_display_solve_steps(
                &mut simplifier.context,
                raw_steps,
                opts.detailed_steps,
                var,
            )
        },
    )
}

/// Convert raw eval steps to display-ready, cleaned steps.
pub fn to_display_steps(raw_steps: Vec<Step>) -> DisplayEvalSteps {
    let cleaned = cas_solver_core::eval_step_pipeline::clean_eval_steps(
        raw_steps,
        |s: &Step| s.before,
        |s: &Step| s.after,
        |s: &Step| s.before_local(),
        |s: &Step| s.after_local(),
        |s: &Step| s.global_after,
        |s: &mut Step, gb| s.global_before = Some(gb),
    );
    DisplayEvalSteps(cleaned)
}

/// Expand with budget tracking, returning pass stats for charging.
pub fn expand_with_stats(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> (cas_ast::ExprId, PassStats) {
    let nodes_snap = ctx.stats().nodes_created;
    let estimated_terms = cas_math::expand_estimate::estimate_expand_terms(ctx, expr).unwrap_or(0);
    let result = expand(ctx, expr);
    let nodes_delta = ctx.stats().nodes_created.saturating_sub(nodes_snap);

    let stats = PassStats {
        op: Operation::Expand,
        rewrite_count: 0,
        nodes_delta,
        terms_materialized: estimated_terms,
        poly_ops: 0,
        stop_reason: None,
    };

    (result, stats)
}

/// Fold constants under the given semantic config and mode.
pub fn fold_constants(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    cfg: &EvalConfig,
    mode: ConstFoldMode,
    budget: &mut Budget,
) -> Result<ConstFoldResult, CasError> {
    const_fold_local::fold_constants_local(ctx, expr, cfg, mode, budget)
}

/// Cancel common additive terms between two equation sides.
pub fn cancel_common_additive_terms(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> Option<CancelResult> {
    cas_solver_core::cancel_common_terms::cancel_common_additive_terms(ctx, lhs, rhs)
}

/// Semantic fallback for equation-level additive cancellation.
pub fn cancel_additive_terms_semantic(
    simplifier: &mut Simplifier,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> Option<CancelResult> {
    use num_traits::Zero;

    // Phase 1: candidate generation (Generic allows x/x -> 1, etc.)
    let candidate_opts = SimplifyOptions {
        collect_steps: false,
        ..Default::default()
    };
    // Phase 2: strict proof for each candidate pair
    let strict_proof_opts = SimplifyOptions {
        shared: SharedSemanticConfig {
            semantics: EvalConfig::strict(),
            ..Default::default()
        },
        collect_steps: false,
        ..Default::default()
    };

    cas_solver_core::cancel_common_terms::cancel_additive_terms_semantic_with_state(
        simplifier,
        lhs,
        rhs,
        |state| &state.context,
        |state| &mut state.context,
        |state, term| state.simplify_with_stats(term, candidate_opts.clone()).0,
        expand,
        |state, lt, rt| {
            let diff = state.context.add(cas_ast::Expr::Sub(lt, rt));
            let (simplified_diff, _, _) =
                state.simplify_with_stats(diff, strict_proof_opts.clone());
            matches!(state.context.get(simplified_diff), cas_ast::Expr::Number(n) if n.is_zero())
        },
    )
}

/// Evaluate a symbolic limit with the engine's current limit evaluator.
pub fn limit(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    var: cas_ast::ExprId,
    approach: Approach,
    opts: &LimitOptions,
    _budget: &mut Budget,
) -> Result<LimitResult, CasError> {
    let outcome = cas_math::limits_support::eval_limit_at_infinity(ctx, expr, var, approach, opts);
    Ok(LimitResult {
        expr: outcome.expr,
        steps: Vec::new(),
        warning: outcome.warning,
    })
}

/// Attempt to prove that an expression is non-zero.
pub fn prove_nonzero(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> Proof {
    prove_nonzero_depth(ctx, expr, 50)
}

/// Attempt to prove that an expression is strictly positive.
pub fn prove_positive(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
    value_domain: ValueDomain,
) -> Proof {
    prove_positive_depth(ctx, expr, value_domain, 50)
}

fn prove_nonzero_depth(ctx: &cas_ast::Context, expr: cas_ast::ExprId, depth: usize) -> Proof {
    cas_solver_core::predicate_proofs::prove_nonzero_depth_with(
        ctx,
        expr,
        depth,
        |core_ctx, inner| prove_positive(core_ctx, inner, ValueDomain::RealOnly),
        try_ground_nonzero_for_proofs,
    )
}

fn prove_positive_depth(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
    value_domain: ValueDomain,
    depth: usize,
) -> Proof {
    cas_solver_core::predicate_proofs::prove_positive_depth_with(
        ctx,
        expr,
        value_domain,
        depth,
        prove_nonzero_depth,
    )
}

fn try_ground_nonzero_for_proofs(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> Option<Proof> {
    cas_math::ground_nonzero::try_ground_nonzero_with(
        ctx,
        expr,
        |source_ctx, source_expr| {
            let mut simplifier = Simplifier::with_context(source_ctx.clone());
            simplifier.set_collect_steps(false);

            let opts = SimplifyOptions {
                collect_steps: false,
                expand_mode: false,
                shared: SharedSemanticConfig {
                    semantics: EvalConfig {
                        domain_mode: DomainMode::Generic,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                budgets: PhaseBudgets {
                    core_iters: 4,
                    transform_iters: 2,
                    rationalize_iters: 0,
                    post_iters: 2,
                    max_total_rewrites: 50,
                },
                ..Default::default()
            };

            let (result, _, _) = simplifier.simplify_with_stats(source_expr, opts);
            Some((simplifier.context, result))
        },
        |evaluated_ctx, evaluated_expr| match evaluated_ctx.get(evaluated_expr) {
            cas_ast::Expr::Number(n) => {
                if num_traits::Zero::is_zero(n) {
                    Some(Proof::Disproven)
                } else {
                    Some(Proof::Proven)
                }
            }
            _ => None,
        },
        |evaluated_ctx, evaluated_expr| {
            let proof = prove_nonzero_depth(
                evaluated_ctx,
                evaluated_expr,
                8, // shallow depth budget for structural fallback
            );
            if proof == Proof::Proven || proof == Proof::Disproven {
                Some(proof)
            } else {
                None
            }
        },
    )
}

/// Verify a single solution by substituting into the equation.
pub fn verify_solution(
    simplifier: &mut Simplifier,
    equation: &cas_ast::Equation,
    var: &str,
    solution: cas_ast::ExprId,
) -> VerifyStatus {
    cas_solver_core::verification_flow::verify_solution_with_domain_modes_with_state(
        simplifier,
        equation,
        var,
        solution,
        |state, eq, solve_var, candidate| {
            cas_solver_core::verify_substitution::substitute_equation_diff(
                &mut state.context,
                eq,
                solve_var,
                candidate,
            )
        },
        |state, expr, domain_mode| {
            let opts = verify_simplify_options_for_domain(domain_mode);
            state.simplify_with_stats(expr, opts).0
        },
        |state, expr| cas_math::expr_predicates::contains_variable(&state.context, expr),
        |state, expr| fold_numeric_islands_for_verify(&mut state.context, expr),
        |state, expr| cas_solver_core::isolation_utils::is_numeric_zero(&state.context, expr),
        |state, expr| cas_formatter::render_expr(&state.context, expr),
    )
}

/// Verify an entire solution set against the source equation.
pub fn verify_solution_set(
    simplifier: &mut Simplifier,
    equation: &cas_ast::Equation,
    var: &str,
    solutions: &cas_ast::SolutionSet,
) -> VerifyResult {
    cas_solver_core::verification_flow::verify_solution_set_for_equation_with_state(
        simplifier,
        equation,
        var,
        solutions,
        verify_solution,
    )
}

fn verify_simplify_options_for_domain(domain_mode: DomainMode) -> SimplifyOptions {
    SimplifyOptions {
        shared: SharedSemanticConfig {
            semantics: EvalConfig {
                domain_mode,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    }
}

fn fold_numeric_islands_for_verify(
    ctx: &mut cas_ast::Context,
    root: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let fold_opts = SimplifyOptions {
        collect_steps: false,
        expand_mode: false,
        shared: SharedSemanticConfig {
            semantics: EvalConfig {
                domain_mode: DomainMode::Generic,
                value_domain: ValueDomain::RealOnly,
                ..Default::default()
            },
            ..Default::default()
        },
        budgets: PhaseBudgets {
            core_iters: 4,
            transform_iters: 2,
            rationalize_iters: 0,
            post_iters: 2,
            max_total_rewrites: 50,
        },
        ..Default::default()
    };

    cas_solver_core::verification_numeric_islands::fold_numeric_islands_guarded_with_default_limits_and_candidate_evaluator(
        ctx,
        root,
        cas_math::ground_eval_guard::GroundEvalGuard::enter,
        |src_ctx, id| {
            let mut tmp = Simplifier::with_context(src_ctx.clone());
            tmp.set_collect_steps(false);
            let (result, _, _) = tmp.simplify_with_stats(id, fold_opts.clone());
            Some((tmp.context, result))
        },
    )
}

/// Infer implicit domain constraints from expression structure.
pub fn infer_implicit_domain(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
    vd: ValueDomain,
) -> ImplicitDomain {
    cas_solver_core::domain_inference_counter::inc();
    cas_solver_core::domain_inference::infer_implicit_domain(ctx, root, vd == ValueDomain::RealOnly)
}

/// Derive additional required conditions from equation equality.
pub fn derive_requires_from_equation(
    ctx: &cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
    existing: &ImplicitDomain,
    vd: ValueDomain,
) -> Vec<ImplicitCondition> {
    cas_solver_core::domain_inference::derive_requires_from_equation(
        ctx,
        lhs,
        rhs,
        existing,
        vd == ValueDomain::RealOnly,
        |ctx, expr| prove_positive(ctx, expr, vd),
    )
}

/// Check if a rewrite would expand the domain by removing implicit constraints.
pub fn domain_delta_check(
    ctx: &cas_ast::Context,
    input: cas_ast::ExprId,
    output: cas_ast::ExprId,
    vd: ValueDomain,
) -> DomainDelta {
    cas_solver_core::domain_inference::domain_delta_check(ctx, input, output, |ctx, expr| {
        infer_implicit_domain(ctx, expr, vd)
    })
}

/// Convert solver path steps to a compact AST expression path.
pub fn pathsteps_to_expr_path(steps: &[PathStep]) -> cas_ast::ExprPath {
    steps.iter().map(PathStep::to_child_index).collect()
}

/// Number-theory helpers exposed by the solver facade without pulling engine rule modules.
pub mod number_theory {
    pub use cas_math::number_theory_support::{compute_gcd, explain_gcd, GcdResult};
}
