use crate::derive::{
    classify_target_profile, contains_phase_shift_term, extract_factored_division_target,
    generate_hyperbolic_additive_term_bridge_rewrites, generate_hyperbolic_bridge_rewrites,
    generate_trig_additive_term_bridge_rewrites, generate_trig_bridge_rewrites,
    looks_like_factored_target, looks_rationalizable_source, ordered_strategies_for_target,
    phase_shift_target_match, presentational_target_match, run_combine_like_terms_rewrite,
    should_try_hyperbolic_planner_before_simplify, should_try_trig_planner_before_simplify,
    strong_target_match, try_build_combined_fraction_from_fold_add,
    try_plan_log_exp_power_inverse_target_aware, try_rewrite_collect_monomial_target_aware,
    try_rewrite_consecutive_factorial_ratio_target_aware,
    try_rewrite_exact_fraction_cancel_target_aware, try_rewrite_expanded_target_aware,
    try_rewrite_exponential_sum_diff_target_aware, try_rewrite_factored_target_aware,
    try_rewrite_fraction_combination_target_aware, try_rewrite_fraction_expansion_target_aware,
    try_rewrite_hyperbolic_expansion_target_aware,
    try_rewrite_hyperbolic_exponential_bridge_target_aware,
    try_rewrite_hyperbolic_simplify_target_aware, try_rewrite_integrate_prep_target_aware,
    try_rewrite_log_argument_factorization_target_aware,
    try_rewrite_log_change_of_base_target_aware, try_rewrite_log_contraction_target_aware,
    try_rewrite_log_contraction_to_target_aware, try_rewrite_log_expansion_target_aware,
    try_rewrite_log_simplify_target_aware, try_rewrite_nested_fraction_target_aware,
    try_rewrite_odd_half_power_to_target_aware, try_rewrite_power_merge_target_aware,
    try_rewrite_pythagorean_factor_form_target_aware,
    try_rewrite_quadruple_sin_angle_contraction_target_aware, try_rewrite_radical_target_aware,
    try_rewrite_rationalized_target_aware, try_rewrite_shifted_double_angle_target_aware,
    try_rewrite_shifted_reciprocal_pythagorean_target_aware, try_rewrite_solve_prep_target_aware,
    try_rewrite_trig_contraction_target_aware, try_rewrite_trig_expansion,
    try_rewrite_trig_identity_to_one_target_aware, try_rewrite_trig_special_value_target_aware,
    DeriveHyperbolicRewriteKind, DeriveLogChangeOfBaseRewriteKind, DeriveStrategy,
    DeriveTargetForm, ExpandRewriteKind, RationalizeRewriteKind,
};
use cas_solver_core::rule_names::RULE_CANCEL_EXACT_ADDITIVE_PAIRS;

use cas_ast::{BuiltinFn, Expr, ExprId};
use cas_engine::NormalFormGoal;
use cas_math::inverse_trig_composition_support::{
    try_plan_inverse_atan_reciprocal_add_expr, try_plan_inverse_trig_composition_expr,
    try_plan_inverse_trig_sum_add_expr, InverseTrigCompositionKind, PairWithNegationPlan,
};
use cas_math::number_theory_support::{
    dispatch_number_theory_call, try_rewrite_choose_symmetry_expr,
    try_rewrite_pascal_choose_identity_expr, NumberTheoryDispatch, NumberTheorySimpleRewrite,
};
use cas_math::summation_support::{
    try_plan_finite_product_evaluation, try_plan_finite_sum_evaluation, FiniteAggregateCall,
    ProductEvaluationKind, SumEvaluationKind,
};
use cas_math::{
    expr_nary::{AddView, Sign},
    expr_predicates::contains_named_var,
    trig_roots_flatten::flatten_mul_chain,
};
use cas_solver_core::engine_event_collector::EngineEventCollector;
use cas_solver_core::engine_events::EngineEvent;
use cas_solver_core::path_rewrite::reconstruct_global_expr;
use cas_solver_core::quadratic_coeffs::{
    extract_quadratic_coefficients, extract_simplified_nonzero_quadratic_coefficients_with_state,
};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeriveEvalError {
    Parse(crate::ParseExprPairError),
    Resolve(String),
}

#[derive(Debug, Clone)]
enum DeriveStatus {
    Derived {
        strategy: DeriveStrategy,
    },
    AlreadyAtTarget,
    EquivalentButUnsupported {
        equivalence: crate::EquivalenceResult,
    },
    NotEquivalent {
        equivalence: crate::EquivalenceResult,
    },
}

#[derive(Debug, Clone)]
struct DeriveStageOutput {
    expr: ExprId,
    steps: Vec<crate::Step>,
}

fn derive_stage_applied(stage: &DeriveStageOutput, source_expr: ExprId) -> bool {
    stage.expr != source_expr || !stage.steps.is_empty()
}

fn derive_stage_contains_rationalize_step(stage: &DeriveStageOutput) -> bool {
    stage.steps.iter().any(|step| {
        step.rule_name.to_ascii_lowercase().contains("rationaliz")
            || step.description.to_ascii_lowercase().contains("rationaliz")
    })
}

fn derive_stage_contains_radical_rewrite_step(stage: &DeriveStageOutput) -> bool {
    stage.steps.iter().any(|step| {
        step.rule_name == "Sqrt Perfect Square"
            || step.rule_name == "Merge Sqrt Product"
            || step.rule_name == "Merge Sqrt Quotient"
            || step.rule_name == "Root Denesting"
            || step.rule_name.to_ascii_lowercase().contains("radical")
            || step.description.to_ascii_lowercase().contains("radical")
    })
}

#[derive(Debug, Clone)]
struct DeriveEvalOutput {
    resolved_expr: ExprId,
    target_expr: ExprId,
    derived_expr: ExprId,
    steps: Vec<crate::Step>,
    status: DeriveStatus,
}

/// Format `derive` parse/resolve errors for user-facing output.
pub(crate) fn format_derive_eval_error_message(error: &DeriveEvalError) -> String {
    match error {
        DeriveEvalError::Parse(parse_error) => {
            crate::format_expr_pair_parse_error_message(parse_error, "derive")
        }
        DeriveEvalError::Resolve(error) => error.clone(),
    }
}

/// Evaluate a full `derive ...` invocation and return final display lines.
pub fn evaluate_derive_command_lines_with_resolver<F>(
    simplifier: &mut crate::Simplifier,
    line: &str,
    display_mode: crate::FullSimplifyDisplayMode,
    simplify_options: crate::SimplifyOptions,
    resolve_expr: F,
) -> Result<Vec<String>, String>
where
    F: FnMut(&mut cas_ast::Context, ExprId) -> Result<ExprId, String>,
{
    let input = crate::extract_derive_command_tail(line);
    let output = evaluate_derive_input_with_resolver(
        simplifier,
        input,
        !matches!(display_mode, crate::FullSimplifyDisplayMode::None),
        simplify_options,
        resolve_expr,
    )
    .map_err(|error| format_derive_eval_error_message(&error))?;

    Ok(format_derive_eval_lines(
        &mut simplifier.context,
        input,
        output,
        display_mode,
    ))
}

fn build_cache_hit_step(
    ctx: &cas_ast::Context,
    description: String,
    before: ExprId,
    after: ExprId,
) -> crate::Step {
    let mut step = crate::Step::new(
        &description,
        "Use cached result",
        before,
        after,
        Vec::new(),
        Some(ctx),
    );
    step.importance = crate::ImportanceLevel::Medium;
    step.category = crate::StepCategory::Substitute;
    step
}

fn build_strategy_fallback_step(
    ctx: &cas_ast::Context,
    strategy: DeriveStrategy,
    before: ExprId,
    after: ExprId,
) -> crate::Step {
    let (description, rule_name, category) = match strategy {
        DeriveStrategy::Planner => (
            "Find a multi-step derivation path",
            "Planner",
            crate::StepCategory::Simplify,
        ),
        DeriveStrategy::Simplify | DeriveStrategy::SimplifyThenExpand => (
            "Simplify the expression",
            "Simplify",
            crate::StepCategory::Simplify,
        ),
        DeriveStrategy::CalculusDiff => (
            "Evaluate the derivative",
            "Symbolic Differentiation",
            crate::StepCategory::Simplify,
        ),
        DeriveStrategy::IntegratePrep => (
            "Prepare the expression for integration",
            "Cos Product Telescoping",
            crate::StepCategory::Simplify,
        ),
        DeriveStrategy::SolvePrep => (
            "Complete the square to rewrite the quadratic",
            "Complete the Square",
            crate::StepCategory::Simplify,
        ),
        DeriveStrategy::FiniteAggregate => (
            "Evaluate a finite sum or product",
            "Finite Aggregate",
            crate::StepCategory::Simplify,
        ),
        DeriveStrategy::NumberTheory => (
            "Evaluate an exact number-theory call",
            "Number Theory Operations",
            crate::StepCategory::Simplify,
        ),
        DeriveStrategy::CombineLikeTerms => (
            "Combine like terms directly",
            "Combine Like Terms",
            crate::StepCategory::Simplify,
        ),
        DeriveStrategy::FactorialRewrite => (
            "Cancel consecutive factorials",
            "Consecutive Factorial Ratio",
            crate::StepCategory::Simplify,
        ),
        DeriveStrategy::InverseTrigRewrite => (
            "Rewrite a direct inverse-trigonometric identity",
            "Inverse Tan Relations",
            crate::StepCategory::Simplify,
        ),
        DeriveStrategy::FractionCancel => (
            "Cancel a structured fraction directly",
            "Fraction Cancel",
            crate::StepCategory::Simplify,
        ),
        DeriveStrategy::NestedFraction => (
            "Simplify a nested fraction directly",
            "Simplify Nested Fraction",
            crate::StepCategory::Simplify,
        ),
        DeriveStrategy::RadicalRewrite => (
            "Rewrite a direct radical identity",
            "Sqrt Perfect Square",
            crate::StepCategory::Simplify,
        ),
        DeriveStrategy::ExponentialRewrite => (
            "Rewrite an exponential sum or product using exp identities",
            "Exponential Sum/Difference Identity",
            crate::StepCategory::Simplify,
        ),
        DeriveStrategy::LogInversePower => (
            "Collapse a power using inverse logarithms",
            "Log Inverse Power",
            crate::StepCategory::Simplify,
        ),
        DeriveStrategy::HyperbolicRewrite => (
            "Rewrite a hyperbolic identity directly",
            "Hyperbolic Identity",
            crate::StepCategory::Simplify,
        ),
        DeriveStrategy::TrigRewrite => (
            "Rewrite a trigonometric identity directly",
            "Trig Identity",
            crate::StepCategory::Simplify,
        ),
        DeriveStrategy::Expand => (
            "Expand the expression distributively",
            "Expand",
            crate::StepCategory::Expand,
        ),
        DeriveStrategy::Collect | DeriveStrategy::SimplifyThenCollect => (
            "Collect terms",
            "Collect Terms",
            crate::StepCategory::Simplify,
        ),
        DeriveStrategy::Factor | DeriveStrategy::SimplifyThenFactor => (
            "Factorization",
            "Factorization",
            crate::StepCategory::Factor,
        ),
        DeriveStrategy::LogExpand | DeriveStrategy::SimplifyThenLogExpand => (
            "Expand a logarithm into a sum of logarithms",
            "expand_log",
            crate::StepCategory::Expand,
        ),
        DeriveStrategy::LogContract | DeriveStrategy::SimplifyThenLogContract => (
            "Combine logarithms into a single logarithm",
            "Log Contraction",
            crate::StepCategory::Simplify,
        ),
        DeriveStrategy::TrigExpand | DeriveStrategy::SimplifyThenTrigExpand => (
            "Expand a trigonometric identity",
            "Trig Expansion",
            crate::StepCategory::Expand,
        ),
        DeriveStrategy::TrigContract | DeriveStrategy::SimplifyThenTrigContract => (
            "Contract a trigonometric expression",
            "Trig Contraction",
            crate::StepCategory::Simplify,
        ),
        DeriveStrategy::Rationalize => (
            "Rationalize the denominator",
            "Rationalize",
            crate::StepCategory::Simplify,
        ),
        DeriveStrategy::FractionExpand => (
            "Distribute a sum over the common denominator",
            "Distribute Division",
            crate::StepCategory::Expand,
        ),
        DeriveStrategy::FractionDecompose => (
            "Split a fraction into a whole part plus remainder",
            "Mixed Fraction Split",
            crate::StepCategory::Expand,
        ),
        DeriveStrategy::FractionCombine => (
            "Combine fractions into a single denominator",
            "Combine Fractions",
            crate::StepCategory::Expand,
        ),
        DeriveStrategy::FactorWithDivision | DeriveStrategy::SimplifyThenFactorWithDivision => (
            "Factor out a common term from the whole expression",
            "Factor Out With Division",
            crate::StepCategory::Factor,
        ),
        DeriveStrategy::PowerMerge => (
            "Combine powers with the same base",
            "Combine powers with same base (n-ary)",
            crate::StepCategory::Simplify,
        ),
        DeriveStrategy::OddHalfPowerExpand | DeriveStrategy::SimplifyThenOddHalfPowerExpand => (
            "Rewrite an odd half-integer power using a square root",
            "Expand Odd Half Power",
            crate::StepCategory::Expand,
        ),
    };

    let mut step = crate::Step::with_snapshots(
        description,
        rule_name,
        before,
        after,
        Vec::new(),
        Some(ctx),
        before,
        after,
    );
    step.importance = crate::ImportanceLevel::Medium;
    step.category = category;
    step
}

pub(crate) fn evaluate_derive_request_with_session<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    raw_input: String,
    parsed_expr: ExprId,
    parsed_target: ExprId,
    auto_store: bool,
) -> Result<crate::EvalOutputView, String>
where
    S: crate::SolverEvalSession,
{
    let prepared = cas_session_core::eval::resolve_and_prepare_dispatch(
        session,
        &mut engine.simplifier.context,
        cas_session_core::eval::ResolvePrepareConfig {
            parsed: parsed_expr,
            raw_input,
            auto_store,
            equiv_other: Some(parsed_target),
            cache_step_max_shown: 6,
        },
        build_cache_hit_step,
    )
    .map_err(|e| e.to_string())?;

    let cas_session_core::eval::PreparedEvalDispatch {
        stored_id,
        resolved,
        inherited_diagnostics,
        resolved_equiv_other,
        cache_hit_step,
    } = prepared;

    let target_expr =
        resolved_equiv_other.ok_or_else(|| "Internal error: missing derive target".to_string())?;

    if let Some(name) = cas_session_core::eval::first_unknown_function_name(
        session,
        &engine.simplifier.context,
        resolved,
    ) {
        return Err(format!("Error: {}", crate::CasError::UnknownFunction(name)));
    }
    if let Some(name) = cas_session_core::eval::first_unknown_function_name(
        session,
        &engine.simplifier.context,
        target_expr,
    ) {
        return Err(format!("Error: {}", crate::CasError::UnknownFunction(name)));
    }

    let simplify_options = session.options().to_simplify_options();
    let derive = evaluate_derive_resolved_input(
        &mut engine.simplifier,
        resolved,
        target_expr,
        !matches!(session.options().steps_mode, crate::StepsMode::Off),
        simplify_options,
    );

    let derived_expr = match &derive.status {
        DeriveStatus::Derived { .. } | DeriveStatus::AlreadyAtTarget => derive.derived_expr,
        DeriveStatus::EquivalentButUnsupported { .. } => {
            return Err(
                "Equivalent, but the second expression is not a supported simplification target yet."
                    .to_string(),
            );
        }
        DeriveStatus::NotEquivalent { equivalence } => {
            let detail = match equivalence {
                crate::EquivalenceResult::False => {
                    "Derive unavailable: the two expressions are not equivalent."
                }
                crate::EquivalenceResult::Unknown => {
                    "Derive unavailable: cannot prove that the two expressions are equivalent, so it cannot provide step-by-step derivation."
                }
                crate::EquivalenceResult::True
                | crate::EquivalenceResult::ConditionalTrue { .. } => "Derive unavailable.",
            };
            return Err(detail.to_string());
        }
    };
    let strategy = match &derive.status {
        DeriveStatus::Derived { strategy } => Some(strategy.label().to_string()),
        _ => None,
    };

    let mut steps = derive.steps;
    if let Some(cache_hit_step) = cache_hit_step {
        steps.insert(0, cache_hit_step);
    }

    let mut diagnostics = inherited_diagnostics;
    let value_domain = session.options().shared.semantics.value_domain;
    let domain_mode = session.options().shared.semantics.domain_mode;
    diagnostics.extend_required(
        crate::infer_implicit_domain(&engine.simplifier.context, resolved, value_domain)
            .conditions()
            .iter()
            .cloned(),
        crate::RequireOrigin::InputImplicit,
    );
    diagnostics.extend_required(
        crate::infer_implicit_domain(&engine.simplifier.context, derived_expr, value_domain)
            .conditions()
            .iter()
            .cloned(),
        crate::RequireOrigin::OutputImplicit,
    );
    diagnostics.dedup_and_sort(&engine.simplifier.context);

    let required_conditions = diagnostics.required_conditions();

    cas_session_core::eval::apply_post_dispatch_store_updates(
        session.store_mut(),
        stored_id,
        diagnostics.clone(),
        Some(cas_session_core::eval::SimplifiedUpdate {
            domain: domain_mode,
            expr: derived_expr,
            requires: diagnostics.requires.clone(),
            steps: None,
        }),
    );

    Ok(crate::EvalOutputView {
        stored_id,
        parsed: parsed_expr,
        resolved,
        result: crate::EvalResult::Expr(derived_expr),
        strategy,
        steps: crate::display_eval_steps::build_display_eval_steps(steps),
        solve_steps: Vec::new(),
        output_scopes: Vec::new(),
        diagnostics,
        required_conditions,
        domain_warnings: Vec::new(),
        blocked_hints: Vec::new(),
        solver_assumptions: Vec::new(),
    })
}

fn evaluate_derive_input_with_resolver<F>(
    simplifier: &mut crate::Simplifier,
    input: &str,
    collect_steps: bool,
    simplify_options: crate::SimplifyOptions,
    mut resolve_expr: F,
) -> Result<DeriveEvalOutput, DeriveEvalError>
where
    F: FnMut(&mut cas_ast::Context, ExprId) -> Result<ExprId, String>,
{
    let mut temp_simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, &mut temp_simplifier.context);
    std::mem::swap(&mut simplifier.profiler, &mut temp_simplifier.profiler);

    let result = (|| {
        let (parsed_expr, parsed_target) =
            crate::parse_expr_pair(&mut temp_simplifier.context, input)
                .map_err(DeriveEvalError::Parse)?;
        let resolved_expr = resolve_expr(&mut temp_simplifier.context, parsed_expr)
            .map_err(DeriveEvalError::Resolve)?;
        let target_expr = resolve_expr(&mut temp_simplifier.context, parsed_target)
            .map_err(DeriveEvalError::Resolve)?;
        Ok(evaluate_derive_resolved_input(
            &mut temp_simplifier,
            resolved_expr,
            target_expr,
            collect_steps,
            simplify_options.clone(),
        ))
    })();

    std::mem::swap(&mut simplifier.context, &mut temp_simplifier.context);
    std::mem::swap(&mut simplifier.profiler, &mut temp_simplifier.profiler);
    result
}

fn evaluate_derive_resolved_input(
    simplifier: &mut crate::Simplifier,
    resolved_expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
    mut simplify_options: crate::SimplifyOptions,
) -> DeriveEvalOutput {
    simplify_options.suppress_depth_overflow_warnings = true;

    cas_engine::with_suppressed_depth_overflow_warnings(|| {
        if presentational_target_match(&mut simplifier.context, resolved_expr, target_expr) {
            return DeriveEvalOutput {
                resolved_expr,
                target_expr,
                derived_expr: target_expr,
                steps: Vec::new(),
                status: DeriveStatus::AlreadyAtTarget,
            };
        }

        if let Some((derived_expr, steps, strategy)) = try_supported_derive_strategies_inner(
            simplifier,
            resolved_expr,
            target_expr,
            collect_steps,
            &simplify_options,
            true,
            true,
        ) {
            let steps = if collect_steps
                && steps.is_empty()
                && !presentational_target_match(
                    &mut simplifier.context,
                    resolved_expr,
                    derived_expr,
                ) {
                vec![build_strategy_fallback_step(
                    &simplifier.context,
                    strategy,
                    resolved_expr,
                    derived_expr,
                )]
            } else {
                steps
            };
            return DeriveEvalOutput {
                resolved_expr,
                target_expr,
                derived_expr,
                steps,
                status: DeriveStatus::Derived { strategy },
            };
        }

        let equivalence = simplifier.are_equivalent_extended(resolved_expr, target_expr);
        let status = match equivalence {
            crate::EquivalenceResult::True | crate::EquivalenceResult::ConditionalTrue { .. } => {
                DeriveStatus::EquivalentButUnsupported { equivalence }
            }
            crate::EquivalenceResult::False | crate::EquivalenceResult::Unknown => {
                DeriveStatus::NotEquivalent { equivalence }
            }
        };

        DeriveEvalOutput {
            resolved_expr,
            target_expr,
            derived_expr: resolved_expr,
            steps: Vec::new(),
            status,
        }
    })
}

fn try_supported_derive_strategies_inner(
    simplifier: &mut crate::Simplifier,
    resolved_expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
    simplify_options: &crate::SimplifyOptions,
    allow_factor_with_division: bool,
    allow_planner: bool,
) -> Option<(ExprId, Vec<crate::Step>, DeriveStrategy)> {
    if let Some(direct) =
        try_fast_direct_integrate_prep_derive(simplifier, resolved_expr, target_expr, collect_steps)
    {
        return Some(direct);
    }
    if let Some(direct) = try_fast_direct_calculus_diff_derive(
        simplifier,
        resolved_expr,
        target_expr,
        collect_steps,
        simplify_options,
    ) {
        return Some(direct);
    }
    if let Some(direct) =
        try_fast_direct_inverse_trig_derive(simplifier, resolved_expr, target_expr, collect_steps)
    {
        return Some(direct);
    }
    if let Some(direct) =
        try_fast_direct_trig_derive(simplifier, resolved_expr, target_expr, collect_steps)
    {
        return Some(direct);
    }
    if let Some(direct) =
        try_fast_direct_hyperbolic_derive(simplifier, resolved_expr, target_expr, collect_steps)
    {
        return Some(direct);
    }

    if let Some(direct) =
        try_fast_direct_factor_derive(simplifier, resolved_expr, target_expr, collect_steps)
    {
        return Some(direct);
    }
    if let Some(direct) = try_fast_direct_log_inverse_power_derive(
        simplifier,
        resolved_expr,
        target_expr,
        collect_steps,
        simplify_options,
    ) {
        return Some(direct);
    }
    if let Some(direct) = try_fast_direct_log_exp_power_inverse_derive(
        simplifier,
        resolved_expr,
        target_expr,
        collect_steps,
    ) {
        return Some(direct);
    }
    if let Some(direct) = try_fast_direct_exponential_rewrite_derive(
        simplifier,
        resolved_expr,
        target_expr,
        collect_steps,
    ) {
        return Some(direct);
    }
    if let Some(direct) = try_fast_direct_log_expand_cleanup_derive(
        simplifier,
        resolved_expr,
        target_expr,
        collect_steps,
        simplify_options,
    ) {
        return Some(direct);
    }
    if let Some(direct) =
        try_fast_direct_number_theory_derive(simplifier, resolved_expr, target_expr, collect_steps)
    {
        return Some(direct);
    }

    let profile = classify_target_profile(&mut simplifier.context, resolved_expr, target_expr);

    if let Some(direct) = try_fast_direct_solve_prep_derive(
        simplifier,
        resolved_expr,
        target_expr,
        &profile.shared_vars,
        collect_steps,
    ) {
        return Some(direct);
    }

    let planner_context_snapshot = allow_planner.then(|| simplifier.context.clone());

    let mut simplify_stage: Option<DeriveStageOutput> = None;
    let mut integrate_prep_stage: Option<DeriveStageOutput> = None;
    let mut solve_prep_stage: Option<DeriveStageOutput> = None;
    let mut finite_aggregate_stage: Option<DeriveStageOutput> = None;
    let mut combine_like_terms_stage: Option<DeriveStageOutput> = None;
    let mut factorial_rewrite_stage: Option<DeriveStageOutput> = None;
    let mut inverse_trig_rewrite_stage: Option<DeriveStageOutput> = None;
    let mut fraction_cancel_stage: Option<DeriveStageOutput> = None;
    let mut nested_fraction_stage: Option<DeriveStageOutput> = None;
    let mut radical_rewrite_stage: Option<DeriveStageOutput> = None;
    let mut exponential_rewrite_stage: Option<DeriveStageOutput> = None;
    let mut hyperbolic_rewrite_stage: Option<DeriveStageOutput> = None;
    let mut trig_rewrite_stage: Option<DeriveStageOutput> = None;
    let mut rationalize_stage: Option<DeriveStageOutput> = None;
    let mut log_expand_stage: Option<DeriveStageOutput> = None;
    let mut log_contract_stage: Option<DeriveStageOutput> = None;
    let mut trig_expand_stage: Option<DeriveStageOutput> = None;
    let mut fraction_expand_stage: Option<DeriveStageOutput> = None;
    let mut fraction_decompose_stage: Option<DeriveStageOutput> = None;
    let mut fraction_combine_stage: Option<DeriveStageOutput> = None;
    let mut factor_with_division_stage: Option<Option<DeriveStageOutput>> = None;
    let mut power_merge_stage: Option<DeriveStageOutput> = None;
    let mut odd_half_power_stage: Option<DeriveStageOutput> = None;
    let mut expand_stage: Option<DeriveStageOutput> = None;
    let mut collect_stage: Option<Option<DeriveStageOutput>> = None;
    let mut factor_stage: Option<DeriveStageOutput> = None;
    let mut simplify_then_log_expand_stage: Option<DeriveStageOutput> = None;
    let mut simplify_then_log_contract_stage: Option<DeriveStageOutput> = None;
    let mut simplify_then_trig_expand_stage: Option<DeriveStageOutput> = None;
    let mut simplify_then_trig_contract_stage: Option<DeriveStageOutput> = None;
    let mut simplify_then_factor_with_division_stage: Option<Option<DeriveStageOutput>> = None;
    let mut simplify_then_odd_half_power_stage: Option<DeriveStageOutput> = None;
    let mut simplify_then_expand_stage: Option<DeriveStageOutput> = None;
    let mut simplify_then_collect_stage: Option<Option<DeriveStageOutput>> = None;
    let mut simplify_then_factor_stage: Option<DeriveStageOutput> = None;
    let prefer_simplify_first = is_finite_aggregate_source(&simplifier.context, resolved_expr)
        && !matches!(profile.form, DeriveTargetForm::FiniteAggregateEvaluated);
    let prefer_planner_for_hyperbolic_exponential_target = allow_planner
        && contains_hyperbolic_fn_expr(&simplifier.context, resolved_expr)
        && contains_exponential_like_expr(&mut simplifier.context, target_expr)
        && !contains_hyperbolic_fn_expr(&simplifier.context, target_expr);
    let allow_planner_preference = allow_planner
        && (should_try_trig_planner_before_simplify(
            &mut simplifier.context,
            resolved_expr,
            target_expr,
        ) || should_try_hyperbolic_planner_before_simplify(
            &mut simplifier.context,
            resolved_expr,
            target_expr,
        ) || prefer_planner_for_hyperbolic_exponential_target);

    if !prefer_simplify_first
        && !matches!(
            profile.form,
            DeriveTargetForm::FactorialRewritten
                | DeriveTargetForm::FactoredWithDivision { .. }
                | DeriveTargetForm::FractionExpanded
                | DeriveTargetForm::FiniteAggregateEvaluated
        )
        && looks_like_factored_target(&mut simplifier.context, target_expr)
    {
        let stage = run_factored_stage(
            simplifier,
            resolved_expr,
            target_expr,
            collect_steps,
            simplify_options.clone(),
        );
        if derive_target_match(simplifier, stage.expr, target_expr)
            || derive_semantic_match(simplifier, stage.expr, target_expr)
        {
            let mut stage = stage;
            retarget_stage_output(&mut stage, target_expr);
            let steps = finalize_steps(
                resolved_expr,
                target_expr,
                vec![stage],
                &mut simplifier.context,
            );
            return Some((target_expr, steps, DeriveStrategy::Factor));
        }
    }

    if allow_planner_preference {
        if let Some(snapshot) = planner_context_snapshot.as_ref() {
            simplifier.context = snapshot.clone();
        }
        if let Some(planner_result) = try_bounded_multistage_derive(
            simplifier,
            resolved_expr,
            target_expr,
            collect_steps,
            simplify_options,
        ) {
            return Some(planner_result);
        }
    }

    if should_try_radical_combine_like_terms_derive(&simplifier.context, resolved_expr, target_expr)
    {
        let mut stage = run_simplify_stage(
            simplifier,
            resolved_expr,
            target_expr,
            true,
            simplify_options.clone(),
        );
        if derive_target_match(simplifier, stage.expr, target_expr)
            && stage
                .steps
                .iter()
                .any(|step| step.rule_name == "Combine Like Terms")
        {
            if !collect_steps {
                stage.steps.clear();
            }
            retarget_stage_output(&mut stage, target_expr);
            let steps = finalize_steps(
                resolved_expr,
                target_expr,
                vec![stage],
                &mut simplifier.context,
            );
            return Some((target_expr, steps, DeriveStrategy::CombineLikeTerms));
        }
        if collect_steps {
            simplify_stage = Some(stage);
        }
    }

    if matches!(profile.form, DeriveTargetForm::LikeTermsCombined) {
        let stage = combine_like_terms_stage.get_or_insert_with(|| {
            run_combine_like_terms_stage(simplifier, resolved_expr, target_expr, collect_steps)
        });
        if derive_target_match(simplifier, stage.expr, target_expr) {
            let mut stage = stage.clone();
            retarget_stage_output(&mut stage, target_expr);
            let steps = finalize_steps(
                resolved_expr,
                target_expr,
                vec![stage],
                &mut simplifier.context,
            );
            return Some((target_expr, steps, DeriveStrategy::CombineLikeTerms));
        }
    }

    if prefer_simplify_first {
        let stage = simplify_stage.get_or_insert_with(|| {
            run_simplify_stage(
                simplifier,
                resolved_expr,
                target_expr,
                collect_steps,
                simplify_options.clone(),
            )
        });
        if derive_target_match(simplifier, stage.expr, target_expr) {
            let mut stage = stage.clone();
            retarget_stage_output(&mut stage, target_expr);
            let steps = finalize_steps(
                resolved_expr,
                target_expr,
                vec![stage],
                &mut simplifier.context,
            );
            if let Some(planner_result) = maybe_prefer_planner_over_direct_result(
                simplifier,
                PlannerPreferenceInput {
                    planner_context_snapshot: planner_context_snapshot.as_ref(),
                    resolved_expr,
                    target_expr,
                    collect_steps,
                    simplify_options,
                    direct_steps: &steps,
                    direct_strategy: DeriveStrategy::Simplify,
                    allow_planner_preference,
                },
            ) {
                return Some(planner_result);
            }
            return Some((target_expr, steps, DeriveStrategy::Simplify));
        }
    }

    if !matches!(profile.form, DeriveTargetForm::Rationalized)
        && looks_rationalizable_source(&simplifier.context, resolved_expr)
    {
        let stage = if collect_steps {
            if simplify_stage.is_none() {
                simplify_stage = Some(run_simplify_stage(
                    simplifier,
                    resolved_expr,
                    target_expr,
                    true,
                    simplify_options.clone(),
                ));
            }
            simplify_stage
                .as_ref()
                .expect("simplify stage should be initialized")
                .clone()
        } else {
            run_simplify_stage(
                simplifier,
                resolved_expr,
                target_expr,
                true,
                simplify_options.clone(),
            )
        };

        if derive_target_match(simplifier, stage.expr, target_expr)
            && derive_stage_contains_rationalize_step(&stage)
        {
            let mut stage = stage;
            if !collect_steps {
                stage.steps.clear();
            }
            retarget_stage_output(&mut stage, target_expr);
            let steps = finalize_steps(
                resolved_expr,
                target_expr,
                vec![stage],
                &mut simplifier.context,
            );
            return Some((target_expr, steps, DeriveStrategy::Rationalize));
        }
    }

    if !matches!(profile.form, DeriveTargetForm::RadicalRewritten)
        && (contains_root_like_expr(&simplifier.context, resolved_expr)
            || contains_root_like_expr(&simplifier.context, target_expr))
    {
        let stage = if collect_steps {
            if simplify_stage.is_none() {
                simplify_stage = Some(run_simplify_stage(
                    simplifier,
                    resolved_expr,
                    target_expr,
                    true,
                    simplify_options.clone(),
                ));
            }
            simplify_stage
                .as_ref()
                .expect("simplify stage should be initialized")
                .clone()
        } else {
            run_simplify_stage(
                simplifier,
                resolved_expr,
                target_expr,
                true,
                simplify_options.clone(),
            )
        };

        if derive_target_match(simplifier, stage.expr, target_expr)
            && derive_stage_contains_radical_rewrite_step(&stage)
        {
            let mut stage = stage;
            if !collect_steps {
                stage.steps.clear();
            }
            retarget_stage_output(&mut stage, target_expr);
            let steps = finalize_steps(
                resolved_expr,
                target_expr,
                vec![stage],
                &mut simplifier.context,
            );
            return Some((target_expr, steps, DeriveStrategy::RadicalRewrite));
        }
    }

    for strategy in ordered_strategies_for_target(&profile) {
        match strategy {
            DeriveStrategy::Planner => continue,
            DeriveStrategy::CalculusDiff => continue,
            DeriveStrategy::LogInversePower => continue,
            DeriveStrategy::NumberTheory => continue,
            DeriveStrategy::Simplify => {
                let stage = simplify_stage.get_or_insert_with(|| {
                    run_simplify_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        collect_steps,
                        simplify_options.clone(),
                    )
                });
                if derive_target_match(simplifier, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    if let Some(planner_result) = maybe_prefer_planner_over_direct_result(
                        simplifier,
                        PlannerPreferenceInput {
                            planner_context_snapshot: planner_context_snapshot.as_ref(),
                            resolved_expr,
                            target_expr,
                            collect_steps,
                            simplify_options,
                            direct_steps: &steps,
                            direct_strategy: DeriveStrategy::Simplify,
                            allow_planner_preference,
                        },
                    ) {
                        return Some(planner_result);
                    }
                    return Some((target_expr, steps, DeriveStrategy::Simplify));
                }
            }
            DeriveStrategy::IntegratePrep => {
                let stage = integrate_prep_stage.get_or_insert_with(|| {
                    run_integrate_prep_stage(simplifier, resolved_expr, target_expr, collect_steps)
                });
                if strong_target_match(&mut simplifier.context, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::IntegratePrep));
                }
            }
            DeriveStrategy::SolvePrep => {
                let stage = solve_prep_stage.get_or_insert_with(|| {
                    run_solve_prep_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        &profile.shared_vars,
                        collect_steps,
                    )
                });
                if strong_target_match(&mut simplifier.context, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::SolvePrep));
                }
            }
            DeriveStrategy::FiniteAggregate => {
                let stage = finite_aggregate_stage.get_or_insert_with(|| {
                    run_finite_aggregate_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        collect_steps,
                    )
                });
                if derive_target_match(simplifier, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::FiniteAggregate));
                }
            }
            DeriveStrategy::CombineLikeTerms => {
                let stage = combine_like_terms_stage.get_or_insert_with(|| {
                    run_combine_like_terms_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        collect_steps,
                    )
                });
                if derive_target_match(simplifier, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::CombineLikeTerms));
                }
            }
            DeriveStrategy::FactorialRewrite => {
                let stage = factorial_rewrite_stage.get_or_insert_with(|| {
                    run_factorial_rewrite_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        collect_steps,
                    )
                });
                if derive_target_match(simplifier, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::FactorialRewrite));
                }
            }
            DeriveStrategy::InverseTrigRewrite => {
                let stage = inverse_trig_rewrite_stage.get_or_insert_with(|| {
                    run_inverse_trig_rewrite_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        collect_steps,
                    )
                });
                if derive_target_match(simplifier, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::InverseTrigRewrite));
                }
            }
            DeriveStrategy::FractionCancel => {
                let stage = fraction_cancel_stage.get_or_insert_with(|| {
                    run_fraction_cancel_stage(simplifier, resolved_expr, target_expr, collect_steps)
                });
                if derive_target_match(simplifier, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::FractionCancel));
                }
            }
            DeriveStrategy::NestedFraction => {
                let stage = nested_fraction_stage.get_or_insert_with(|| {
                    run_nested_fraction_stage(simplifier, resolved_expr, target_expr, collect_steps)
                });
                if derive_target_match(simplifier, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::NestedFraction));
                }
            }
            DeriveStrategy::RadicalRewrite => {
                let stage = radical_rewrite_stage.get_or_insert_with(|| {
                    run_radical_rewrite_stage(simplifier, resolved_expr, target_expr, collect_steps)
                });
                if derive_target_match(simplifier, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::RadicalRewrite));
                }
            }
            DeriveStrategy::ExponentialRewrite => {
                let stage = exponential_rewrite_stage.get_or_insert_with(|| {
                    run_exponential_sum_diff_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        collect_steps,
                    )
                });
                if derive_target_match(simplifier, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::ExponentialRewrite));
                }
            }
            DeriveStrategy::HyperbolicRewrite => {
                let stage = hyperbolic_rewrite_stage.get_or_insert_with(|| {
                    run_hyperbolic_rewrite_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        collect_steps,
                    )
                });
                if derive_target_match(simplifier, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::HyperbolicRewrite));
                }
            }
            DeriveStrategy::TrigRewrite => {
                let stage = trig_rewrite_stage.get_or_insert_with(|| {
                    run_trig_rewrite_stage(simplifier, resolved_expr, target_expr, collect_steps)
                });
                if derive_target_match(simplifier, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::TrigRewrite));
                }
            }
            DeriveStrategy::LogExpand => {
                if let Some(first_stage) =
                    run_log_expand_prep_stage(simplifier, resolved_expr, target_expr, collect_steps)
                {
                    let stage = run_log_expand_stage(
                        simplifier,
                        first_stage.expr,
                        target_expr,
                        collect_steps,
                        simplify_options.clone(),
                    );
                    if stage_makes_progress(&mut simplifier.context, first_stage.expr, &stage)
                        && derive_target_match(simplifier, stage.expr, target_expr)
                    {
                        let mut stage = stage;
                        retarget_stage_output(&mut stage, target_expr);
                        let steps = finalize_steps(
                            resolved_expr,
                            target_expr,
                            vec![first_stage, stage],
                            &mut simplifier.context,
                        );
                        return Some((target_expr, steps, DeriveStrategy::LogExpand));
                    }
                }

                let stage = log_expand_stage.get_or_insert_with(|| {
                    run_log_expand_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        collect_steps,
                        simplify_options.clone(),
                    )
                });
                if stage_makes_progress(&mut simplifier.context, resolved_expr, stage)
                    && derive_target_match(simplifier, stage.expr, target_expr)
                {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::LogExpand));
                }
                if stage_makes_progress(&mut simplifier.context, resolved_expr, stage) {
                    let first_stage = stage.clone();
                    let cleanup_stage = run_simplify_stage(
                        simplifier,
                        first_stage.expr,
                        target_expr,
                        collect_steps,
                        simplify_options.clone(),
                    );
                    if stage_makes_progress(
                        &mut simplifier.context,
                        first_stage.expr,
                        &cleanup_stage,
                    ) && derive_target_match(simplifier, cleanup_stage.expr, target_expr)
                    {
                        let mut cleanup_stage = cleanup_stage;
                        retarget_stage_output(&mut cleanup_stage, target_expr);
                        let steps = finalize_steps(
                            resolved_expr,
                            target_expr,
                            vec![first_stage, cleanup_stage],
                            &mut simplifier.context,
                        );
                        return Some((target_expr, steps, DeriveStrategy::LogExpand));
                    }
                }
            }
            DeriveStrategy::LogContract => {
                let stage = log_contract_stage.get_or_insert_with(|| {
                    run_log_contract_stage(simplifier, resolved_expr, target_expr, collect_steps)
                });
                if derive_target_match(simplifier, stage.expr, target_expr)
                    || derive_semantic_match(simplifier, stage.expr, target_expr)
                {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::LogContract));
                }
            }
            DeriveStrategy::SimplifyThenLogContract => {
                let first_stage = simplify_stage.get_or_insert_with(|| {
                    run_simplify_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        collect_steps,
                        simplify_options.clone(),
                    )
                });
                let stage = simplify_then_log_contract_stage.get_or_insert_with(|| {
                    run_log_contract_stage(simplifier, first_stage.expr, target_expr, collect_steps)
                });
                if derive_target_match(simplifier, stage.expr, target_expr)
                    || derive_semantic_match(simplifier, stage.expr, target_expr)
                {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![first_stage.clone(), stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::SimplifyThenLogContract));
                }
            }
            DeriveStrategy::TrigExpand => {
                let stage = trig_expand_stage.get_or_insert_with(|| {
                    run_trig_expand_stage(simplifier, resolved_expr, target_expr, collect_steps)
                });
                if strong_target_match(&mut simplifier.context, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::TrigExpand));
                }
            }
            DeriveStrategy::TrigContract => {
                let direct_stage =
                    run_trig_contract_stage(simplifier, resolved_expr, target_expr, collect_steps);
                if derive_target_match(simplifier, direct_stage.expr, target_expr) {
                    let mut stage = direct_stage;
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::TrigContract));
                }
            }
            DeriveStrategy::SimplifyThenTrigContract => {
                let first_stage = simplify_stage.get_or_insert_with(|| {
                    run_simplify_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        collect_steps,
                        simplify_options.clone(),
                    )
                });
                let stage = simplify_then_trig_contract_stage.get_or_insert_with(|| {
                    run_trig_contract_stage(
                        simplifier,
                        first_stage.expr,
                        target_expr,
                        collect_steps,
                    )
                });
                if derive_target_match(simplifier, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![first_stage.clone(), stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::SimplifyThenTrigContract));
                }
            }
            DeriveStrategy::Rationalize => {
                let stage = rationalize_stage.get_or_insert_with(|| {
                    run_rationalize_stage(simplifier, resolved_expr, target_expr, collect_steps)
                });
                if derive_stage_applied(stage, resolved_expr)
                    && derive_target_match(simplifier, stage.expr, target_expr)
                {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::Rationalize));
                }

                if matches!(profile.form, DeriveTargetForm::Rationalized) {
                    let stage = simplify_stage.get_or_insert_with(|| {
                        run_simplify_stage(
                            simplifier,
                            resolved_expr,
                            target_expr,
                            collect_steps,
                            simplify_options.clone(),
                        )
                    });
                    if strong_target_match(&mut simplifier.context, stage.expr, target_expr) {
                        let mut stage = stage.clone();
                        retarget_stage_output(&mut stage, target_expr);
                        let steps = finalize_steps(
                            resolved_expr,
                            target_expr,
                            vec![stage],
                            &mut simplifier.context,
                        );
                        return Some((target_expr, steps, DeriveStrategy::Rationalize));
                    }
                }
            }
            DeriveStrategy::FractionExpand => {
                let stage = fraction_expand_stage.get_or_insert_with(|| {
                    run_fraction_expand_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        collect_steps,
                        simplify_options.clone(),
                    )
                });
                if strong_target_match(&mut simplifier.context, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::FractionExpand));
                }
            }
            DeriveStrategy::FractionDecompose => {
                let stage = fraction_decompose_stage.get_or_insert_with(|| {
                    run_fraction_decompose_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        collect_steps,
                    )
                });
                if strong_target_match(&mut simplifier.context, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::FractionDecompose));
                }
            }
            DeriveStrategy::FractionCombine => {
                let stage = fraction_combine_stage.get_or_insert_with(|| {
                    run_fraction_combine_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        collect_steps,
                    )
                });
                if derive_target_match(simplifier, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::FractionCombine));
                }
            }
            DeriveStrategy::FactorWithDivision => {
                if !allow_factor_with_division {
                    continue;
                }
                let stage = factor_with_division_stage.get_or_insert_with(|| {
                    run_factor_with_division_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        profile.shared_vars.as_slice(),
                        collect_steps,
                    )
                });
                if let Some(mut stage) = stage.clone() {
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::FactorWithDivision));
                }
            }
            DeriveStrategy::PowerMerge => {
                let stage = power_merge_stage.get_or_insert_with(|| {
                    run_power_merge_stage(simplifier, resolved_expr, target_expr, collect_steps)
                });
                if strong_target_match(&mut simplifier.context, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::PowerMerge));
                }
            }
            DeriveStrategy::OddHalfPowerExpand => {
                let stage = odd_half_power_stage.get_or_insert_with(|| {
                    run_odd_half_power_stage(simplifier, resolved_expr, target_expr, collect_steps)
                });
                if strong_target_match(&mut simplifier.context, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::OddHalfPowerExpand));
                }
            }
            DeriveStrategy::Collect => {
                let stage = collect_stage.get_or_insert_with(|| {
                    run_collect_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        profile.shared_vars.as_slice(),
                    )
                });
                if let Some(mut stage) = stage.clone() {
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::Collect));
                }
            }
            DeriveStrategy::Expand => {
                if !matches!(profile.form, DeriveTargetForm::Expanded)
                    && try_rewrite_expanded_target_aware(
                        &mut simplifier.context,
                        resolved_expr,
                        target_expr,
                    )
                    .is_none()
                {
                    continue;
                }
                let stage = expand_stage.get_or_insert_with(|| {
                    run_expand_stage(simplifier, resolved_expr, target_expr, collect_steps)
                });
                let matched = derive_target_match(simplifier, stage.expr, target_expr);
                if matched {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::Expand));
                }
            }
            DeriveStrategy::Factor => {
                let stage = factor_stage.get_or_insert_with(|| {
                    run_factored_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        collect_steps,
                        simplify_options.clone(),
                    )
                });
                if strong_target_match(&mut simplifier.context, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::Factor));
                }
            }
            DeriveStrategy::SimplifyThenLogExpand => {
                if let Some(first_stage) =
                    run_log_expand_prep_stage(simplifier, resolved_expr, target_expr, collect_steps)
                {
                    let stage = run_log_expand_stage(
                        simplifier,
                        first_stage.expr,
                        target_expr,
                        collect_steps,
                        simplify_options.clone(),
                    );
                    if stage_makes_progress(&mut simplifier.context, first_stage.expr, &stage)
                        && strong_target_match(&mut simplifier.context, stage.expr, target_expr)
                    {
                        let mut stage = stage;
                        retarget_stage_output(&mut stage, target_expr);
                        let steps = finalize_steps(
                            resolved_expr,
                            target_expr,
                            vec![first_stage, stage],
                            &mut simplifier.context,
                        );
                        return Some((target_expr, steps, DeriveStrategy::SimplifyThenLogExpand));
                    }
                }

                let first_stage = simplify_stage.get_or_insert_with(|| {
                    run_simplify_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        collect_steps,
                        simplify_options.clone(),
                    )
                });
                let stage = simplify_then_log_expand_stage.get_or_insert_with(|| {
                    run_log_expand_stage(
                        simplifier,
                        first_stage.expr,
                        target_expr,
                        collect_steps,
                        simplify_options.clone(),
                    )
                });
                if stage_makes_progress(&mut simplifier.context, first_stage.expr, stage)
                    && strong_target_match(&mut simplifier.context, stage.expr, target_expr)
                {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![first_stage.clone(), stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::SimplifyThenLogExpand));
                }
            }
            DeriveStrategy::SimplifyThenTrigExpand => {
                let first_stage = simplify_stage.get_or_insert_with(|| {
                    run_simplify_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        collect_steps,
                        simplify_options.clone(),
                    )
                });
                let stage = simplify_then_trig_expand_stage.get_or_insert_with(|| {
                    run_trig_expand_stage(simplifier, first_stage.expr, target_expr, collect_steps)
                });
                if strong_target_match(&mut simplifier.context, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![first_stage.clone(), stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::SimplifyThenTrigExpand));
                }
            }
            DeriveStrategy::SimplifyThenOddHalfPowerExpand => {
                let first_stage = simplify_stage.get_or_insert_with(|| {
                    run_simplify_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        collect_steps,
                        simplify_options.clone(),
                    )
                });
                let stage = simplify_then_odd_half_power_stage.get_or_insert_with(|| {
                    run_odd_half_power_stage(
                        simplifier,
                        first_stage.expr,
                        target_expr,
                        collect_steps,
                    )
                });
                if strong_target_match(&mut simplifier.context, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![first_stage.clone(), stage],
                        &mut simplifier.context,
                    );
                    return Some((
                        target_expr,
                        steps,
                        DeriveStrategy::SimplifyThenOddHalfPowerExpand,
                    ));
                }
            }
            DeriveStrategy::SimplifyThenFactorWithDivision => {
                if !allow_factor_with_division {
                    continue;
                }
                let first_stage = simplify_stage.get_or_insert_with(|| {
                    run_simplify_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        collect_steps,
                        simplify_options.clone(),
                    )
                });
                let stage = simplify_then_factor_with_division_stage.get_or_insert_with(|| {
                    run_factor_with_division_stage(
                        simplifier,
                        first_stage.expr,
                        target_expr,
                        profile.shared_vars.as_slice(),
                        collect_steps,
                    )
                });
                if let Some(mut stage) = stage.clone() {
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![first_stage.clone(), stage],
                        &mut simplifier.context,
                    );
                    return Some((
                        target_expr,
                        steps,
                        DeriveStrategy::SimplifyThenFactorWithDivision,
                    ));
                }
            }
            DeriveStrategy::SimplifyThenCollect => {
                let first_stage = simplify_stage.get_or_insert_with(|| {
                    run_simplify_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        collect_steps,
                        simplify_options.clone(),
                    )
                });
                let stage = simplify_then_collect_stage.get_or_insert_with(|| {
                    run_collect_stage(
                        simplifier,
                        first_stage.expr,
                        target_expr,
                        profile.shared_vars.as_slice(),
                    )
                });
                if let Some(mut stage) = stage.clone() {
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![first_stage.clone(), stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::SimplifyThenCollect));
                }
            }
            DeriveStrategy::SimplifyThenExpand => {
                if !matches!(profile.form, DeriveTargetForm::Expanded) {
                    continue;
                }
                let first_stage = simplify_stage.get_or_insert_with(|| {
                    run_simplify_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        collect_steps,
                        simplify_options.clone(),
                    )
                });
                let stage = simplify_then_expand_stage.get_or_insert_with(|| {
                    run_expand_stage(simplifier, first_stage.expr, target_expr, collect_steps)
                });
                let matched = derive_target_match(simplifier, stage.expr, target_expr);
                if matched {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![first_stage.clone(), stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::SimplifyThenExpand));
                }
            }
            DeriveStrategy::SimplifyThenFactor => {
                let first_stage = simplify_stage.get_or_insert_with(|| {
                    run_simplify_stage(
                        simplifier,
                        resolved_expr,
                        target_expr,
                        collect_steps,
                        simplify_options.clone(),
                    )
                });
                let stage = simplify_then_factor_stage.get_or_insert_with(|| {
                    run_factored_stage(
                        simplifier,
                        first_stage.expr,
                        target_expr,
                        collect_steps,
                        simplify_options.clone(),
                    )
                });
                if strong_target_match(&mut simplifier.context, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![first_stage.clone(), stage],
                        &mut simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::SimplifyThenFactor));
                }
            }
        }
    }

    let stage = expand_stage.get_or_insert_with(|| {
        run_expand_stage(simplifier, resolved_expr, target_expr, collect_steps)
    });
    if derive_target_match(simplifier, stage.expr, target_expr) {
        let mut stage = stage.clone();
        retarget_stage_output(&mut stage, target_expr);
        let steps = finalize_steps(
            resolved_expr,
            target_expr,
            vec![stage],
            &mut simplifier.context,
        );
        return Some((target_expr, steps, DeriveStrategy::Expand));
    }

    if allow_planner {
        if let Some(snapshot) = planner_context_snapshot {
            simplifier.context = snapshot;
        }
        return try_bounded_multistage_derive(
            simplifier,
            resolved_expr,
            target_expr,
            collect_steps,
            simplify_options,
        );
    }

    None
}

struct PlannerPreferenceInput<'a> {
    planner_context_snapshot: Option<&'a cas_ast::Context>,
    resolved_expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
    simplify_options: &'a crate::SimplifyOptions,
    direct_steps: &'a [crate::Step],
    direct_strategy: DeriveStrategy,
    allow_planner_preference: bool,
}

fn maybe_prefer_planner_over_direct_result(
    simplifier: &mut crate::Simplifier,
    input: PlannerPreferenceInput<'_>,
) -> Option<(ExprId, Vec<crate::Step>, DeriveStrategy)> {
    if input.direct_strategy != DeriveStrategy::Simplify || !input.allow_planner_preference {
        return None;
    }

    let planner_context_snapshot = input.planner_context_snapshot?.clone();
    let direct_context_snapshot = simplifier.context.clone();

    simplifier.context = planner_context_snapshot;

    let planner_result = try_bounded_multistage_derive(
        simplifier,
        input.resolved_expr,
        input.target_expr,
        input.collect_steps,
        input.simplify_options,
    );

    let should_prefer_planner =
        planner_result
            .as_ref()
            .is_some_and(|(_expr, planner_steps, strategy)| {
                *strategy == DeriveStrategy::Planner
                    && !planner_steps.is_empty()
                    && (input.direct_steps.is_empty()
                        || planner_steps.len() < input.direct_steps.len())
            });

    if should_prefer_planner {
        return planner_result;
    }

    simplifier.context = direct_context_snapshot;
    None
}

fn try_bounded_multistage_derive(
    simplifier: &mut crate::Simplifier,
    resolved_expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
    simplify_options: &crate::SimplifyOptions,
) -> Option<(ExprId, Vec<crate::Step>, DeriveStrategy)> {
    const MAX_BRIDGE_DEPTH: usize = 3;

    let mut visited = BTreeSet::new();
    visited.insert(derive_expr_signature(&simplifier.context, resolved_expr));

    let stages = try_bounded_multistage_derive_inner(
        simplifier,
        resolved_expr,
        target_expr,
        collect_steps,
        simplify_options,
        MAX_BRIDGE_DEPTH,
        &mut visited,
    )?;

    let steps = finalize_planner_steps(resolved_expr, target_expr, stages, &mut simplifier.context);
    Some((target_expr, steps, DeriveStrategy::Planner))
}

fn try_bounded_multistage_derive_inner(
    simplifier: &mut crate::Simplifier,
    current_expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
    simplify_options: &crate::SimplifyOptions,
    depth_left: usize,
    visited: &mut BTreeSet<String>,
) -> Option<Vec<DeriveStageOutput>> {
    if depth_left == 0 {
        return None;
    }

    let mut semantic_fallback = None;
    let mut exploratory_stages = Vec::new();

    for mut stage in generate_planner_candidate_stages(
        simplifier,
        current_expr,
        target_expr,
        collect_steps,
        simplify_options,
    ) {
        if presentational_target_match(&mut simplifier.context, stage.expr, current_expr) {
            continue;
        }

        if presentational_target_match(&mut simplifier.context, stage.expr, target_expr) {
            retarget_stage_output(&mut stage, target_expr);
            return Some(vec![stage]);
        }

        if planner_stage_target_match(simplifier, stage.expr, target_expr) {
            let mut fallback_stage = stage.clone();
            retarget_stage_output(&mut fallback_stage, target_expr);
            semantic_fallback.get_or_insert_with(|| vec![fallback_stage]);
            continue;
        }

        exploratory_stages.push(stage);
    }

    if semantic_fallback
        .as_ref()
        .is_some_and(|stages| stages.len() == 1 && stages[0].steps.len() <= 1)
    {
        return semantic_fallback;
    }

    for mut stage in exploratory_stages {
        let signature = derive_expr_signature(&simplifier.context, stage.expr);
        if !visited.insert(signature) {
            continue;
        }

        if let Some(mut tail) = try_bounded_multistage_derive_inner(
            simplifier,
            stage.expr,
            target_expr,
            collect_steps,
            simplify_options,
            depth_left - 1,
            visited,
        ) {
            let mut stages = vec![stage];
            stages.append(&mut tail);
            return Some(stages);
        }

        if let Some((_derived_expr, direct_steps, _strategy)) =
            try_supported_derive_strategies_inner(
                simplifier,
                stage.expr,
                target_expr,
                collect_steps,
                simplify_options,
                true,
                true,
            )
        {
            let mut stages = vec![stage];
            stages.push(DeriveStageOutput {
                expr: target_expr,
                steps: direct_steps,
            });
            return Some(stages);
        }

        if derive_semantic_match(simplifier, stage.expr, target_expr) {
            retarget_stage_output(&mut stage, target_expr);
            return Some(vec![stage]);
        }
    }

    semantic_fallback
}

fn planner_stage_target_match(
    simplifier: &mut crate::Simplifier,
    actual_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    if strong_target_match(&mut simplifier.context, actual_expr, target_expr) {
        return true;
    }

    for rewrite in generate_hyperbolic_bridge_rewrites(&mut simplifier.context, actual_expr) {
        if strong_target_match(&mut simplifier.context, rewrite.rewritten, target_expr) {
            return true;
        }
    }

    for rewrite in
        generate_hyperbolic_additive_term_bridge_rewrites(&mut simplifier.context, actual_expr)
    {
        if strong_target_match(&mut simplifier.context, rewrite.rewritten, target_expr) {
            return true;
        }
    }

    for rewrite in generate_trig_bridge_rewrites(&mut simplifier.context, actual_expr) {
        if strong_target_match(&mut simplifier.context, rewrite.rewritten, target_expr) {
            return true;
        }
    }

    for rewrite in generate_trig_additive_term_bridge_rewrites(&mut simplifier.context, actual_expr)
    {
        if strong_target_match(&mut simplifier.context, rewrite.rewritten, target_expr) {
            return true;
        }
    }

    false
}

fn stage_makes_progress(
    ctx: &mut cas_ast::Context,
    before: ExprId,
    stage: &DeriveStageOutput,
) -> bool {
    !stage.steps.is_empty() || !presentational_target_match(ctx, stage.expr, before)
}

fn try_fast_direct_integrate_prep_derive(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<crate::Step>, DeriveStrategy)> {
    let rewrite =
        try_rewrite_integrate_prep_target_aware(&mut simplifier.context, expr, target_expr)?;
    let steps = if collect_steps {
        let mut step = crate::Step::with_snapshots(
            rewrite.kind.description(),
            rewrite.kind.rule_name(),
            expr,
            rewrite.rewritten,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            rewrite.rewritten,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Simplify;
        step.meta_mut().assumption_events.push(
            cas_solver_core::assumption_model::AssumptionEvent::nonzero(
                &simplifier.context,
                rewrite.assume_nonzero_expr,
            ),
        );
        vec![step]
    } else {
        Vec::new()
    };

    Some((rewrite.rewritten, steps, DeriveStrategy::IntegratePrep))
}

fn try_fast_direct_calculus_diff_derive(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
    simplify_options: &crate::SimplifyOptions,
) -> Option<(ExprId, Vec<crate::Step>, DeriveStrategy)> {
    if !contains_named_function_expr(&simplifier.context, expr, "diff") {
        return None;
    }

    let mut stage = run_simplify_stage(
        simplifier,
        expr,
        target_expr,
        collect_steps,
        simplify_options.clone(),
    );
    if !derive_target_match(simplifier, stage.expr, target_expr) {
        return None;
    }
    if collect_steps && !stage.steps.iter().any(is_calculus_diff_step) {
        return None;
    }
    if !collect_steps {
        stage.steps.clear();
    }
    retarget_stage_output(&mut stage, target_expr);
    let steps = finalize_steps(expr, target_expr, vec![stage], &mut simplifier.context);
    Some((target_expr, steps, DeriveStrategy::CalculusDiff))
}

fn is_calculus_diff_step(step: &crate::Step) -> bool {
    matches!(
        step.rule_name.as_str(),
        "Symbolic Differentiation" | "Calcular la derivada"
    )
}

fn contains_named_function_expr(ctx: &cas_ast::Context, expr: ExprId, name: &str) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) => {
            ctx.sym_name(*fn_id) == name
                || args
                    .iter()
                    .any(|arg| contains_named_function_expr(ctx, *arg, name))
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            contains_named_function_expr(ctx, *left, name)
                || contains_named_function_expr(ctx, *right, name)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => contains_named_function_expr(ctx, *inner, name),
        Expr::Matrix { data, .. } => data
            .iter()
            .any(|entry| contains_named_function_expr(ctx, *entry, name)),
        _ => false,
    }
}

fn try_fast_direct_inverse_trig_derive(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<crate::Step>, DeriveStrategy)> {
    let stage = run_inverse_trig_rewrite_stage(simplifier, expr, target_expr, collect_steps);
    if !stage_makes_progress(&mut simplifier.context, expr, &stage)
        || !derive_target_match(simplifier, stage.expr, target_expr)
    {
        return None;
    }

    let mut stage = stage;
    retarget_stage_output(&mut stage, target_expr);
    let steps = finalize_steps(expr, target_expr, vec![stage], &mut simplifier.context);
    Some((target_expr, steps, DeriveStrategy::InverseTrigRewrite))
}

fn try_fast_direct_solve_prep_derive(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    shared_vars: &[String],
    collect_steps: bool,
) -> Option<(ExprId, Vec<crate::Step>, DeriveStrategy)> {
    let rewrite = try_rewrite_solve_prep_target_aware(
        &mut simplifier.context,
        expr,
        target_expr,
        shared_vars,
    )?;
    let steps = if collect_steps {
        let substeps = complete_square_didactic_substeps(&mut simplifier.context, expr);
        let mut step = crate::Step::with_snapshots(
            rewrite.kind.description(),
            rewrite.kind.rule_name(),
            expr,
            rewrite.rewritten,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            rewrite.rewritten,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Simplify;
        step.meta_mut().assumption_events.push(
            cas_solver_core::assumption_model::AssumptionEvent::nonzero(
                &simplifier.context,
                rewrite.assume_nonzero_expr,
            ),
        );
        if !substeps.is_empty() {
            step.meta_mut().substeps = substeps;
        }
        vec![step]
    } else {
        Vec::new()
    };

    Some((rewrite.rewritten, steps, DeriveStrategy::SolvePrep))
}

fn try_fast_direct_log_expand_cleanup_derive(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
    simplify_options: &crate::SimplifyOptions,
) -> Option<(ExprId, Vec<crate::Step>, DeriveStrategy)> {
    if !should_try_log_expand_cleanup_derive(&simplifier.context, expr, target_expr) {
        return None;
    }

    let mut probe_context = simplifier.context.clone();
    if try_rewrite_log_argument_factorization_target_aware(&mut probe_context, expr, target_expr)
        .is_some()
    {
        return None;
    }

    let mut first_stage = run_log_expand_without_cleanup_stage(simplifier, expr, collect_steps);
    if !stage_makes_progress(&mut simplifier.context, expr, &first_stage) {
        return None;
    }

    if presentational_target_match(&mut simplifier.context, first_stage.expr, target_expr) {
        retarget_stage_output(&mut first_stage, target_expr);
        let steps = finalize_steps(
            expr,
            target_expr,
            vec![first_stage],
            &mut simplifier.context,
        );
        return Some((target_expr, steps, DeriveStrategy::LogExpand));
    }

    let mut cleanup_stage = run_simplify_stage(
        simplifier,
        first_stage.expr,
        target_expr,
        collect_steps,
        simplify_options.clone(),
    );
    if !stage_makes_progress(&mut simplifier.context, first_stage.expr, &cleanup_stage)
        || !derive_target_match(simplifier, cleanup_stage.expr, target_expr)
    {
        return None;
    }

    ensure_log_expand_cleanup_step(
        &simplifier.context,
        first_stage.expr,
        target_expr,
        &mut cleanup_stage,
    );
    retarget_stage_output(&mut cleanup_stage, target_expr);
    let steps = finalize_steps(
        expr,
        target_expr,
        vec![first_stage, cleanup_stage],
        &mut simplifier.context,
    );
    Some((target_expr, steps, DeriveStrategy::LogExpand))
}

fn try_fast_direct_number_theory_derive(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<crate::Step>, DeriveStrategy)> {
    if let Some(rewrite) = try_rewrite_pascal_choose_identity_expr(&mut simplifier.context, expr) {
        if strong_target_match(&mut simplifier.context, rewrite.rewritten, target_expr) {
            let steps = if collect_steps {
                let mut step = crate::Step::with_snapshots(
                    "Apply Pascal's identity for binomial coefficients",
                    "Pascal's Identity",
                    expr,
                    rewrite.rewritten,
                    Vec::new(),
                    Some(&simplifier.context),
                    expr,
                    rewrite.rewritten,
                );
                step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
                step.category = cas_solver_core::step_types::StepCategory::Simplify;
                vec![step]
            } else {
                Vec::new()
            };

            let mut stage = DeriveStageOutput {
                expr: rewrite.rewritten,
                steps,
            };
            retarget_stage_output(&mut stage, target_expr);
            let steps = finalize_steps(expr, target_expr, vec![stage], &mut simplifier.context);
            return Some((target_expr, steps, DeriveStrategy::NumberTheory));
        }
    }

    if let Some(rewrite) = try_rewrite_choose_symmetry_expr(&mut simplifier.context, expr) {
        if strong_target_match(&mut simplifier.context, rewrite.rewritten, target_expr) {
            let steps = if collect_steps {
                let mut step = crate::Step::with_snapshots(
                    "Apply binomial coefficient symmetry",
                    "Binomial Coefficient Symmetry",
                    expr,
                    rewrite.rewritten,
                    Vec::new(),
                    Some(&simplifier.context),
                    expr,
                    rewrite.rewritten,
                );
                step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
                step.category = cas_solver_core::step_types::StepCategory::Simplify;
                vec![step]
            } else {
                Vec::new()
            };

            let mut stage = DeriveStageOutput {
                expr: rewrite.rewritten,
                steps,
            };
            retarget_stage_output(&mut stage, target_expr);
            let steps = finalize_steps(expr, target_expr, vec![stage], &mut simplifier.context);
            return Some((target_expr, steps, DeriveStrategy::NumberTheory));
        }
    }

    let rewrite = match dispatch_number_theory_call(&mut simplifier.context, expr)? {
        NumberTheoryDispatch::Simple(rewrite) => rewrite,
        NumberTheoryDispatch::PolyGcd { .. } => return None,
    };

    let rewritten = rewrite.result();
    if !strong_target_match(&mut simplifier.context, rewritten, target_expr) {
        return None;
    }

    let steps = if collect_steps {
        let description = render_number_theory_simple_desc(&simplifier.context, rewrite);
        let mut step = crate::Step::with_snapshots(
            &description,
            "Number Theory Operations",
            expr,
            rewritten,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            rewritten,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Simplify;
        vec![step]
    } else {
        Vec::new()
    };

    let mut stage = DeriveStageOutput {
        expr: rewritten,
        steps,
    };
    retarget_stage_output(&mut stage, target_expr);
    let steps = finalize_steps(expr, target_expr, vec![stage], &mut simplifier.context);
    Some((target_expr, steps, DeriveStrategy::NumberTheory))
}

fn render_number_theory_simple_desc(
    ctx: &cas_ast::Context,
    rewrite: NumberTheorySimpleRewrite,
) -> String {
    match rewrite {
        NumberTheorySimpleRewrite::Unary { name, arg, .. } => {
            format!("{}({})", name, cas_formatter::render_expr(ctx, arg))
        }
        NumberTheorySimpleRewrite::Binary { name, lhs, rhs, .. } => {
            format!(
                "{}({}, {})",
                name,
                cas_formatter::render_expr(ctx, lhs),
                cas_formatter::render_expr(ctx, rhs)
            )
        }
    }
}

fn should_try_radical_combine_like_terms_derive(
    ctx: &cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> bool {
    matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
        && (contains_root_like_expr(ctx, expr) || contains_root_like_expr(ctx, target_expr))
}

fn try_fast_direct_log_inverse_power_derive(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
    simplify_options: &crate::SimplifyOptions,
) -> Option<(ExprId, Vec<crate::Step>, DeriveStrategy)> {
    if !contains_log_call_in_power_exponent(&simplifier.context, expr) {
        return None;
    }

    let mut stage = run_simplify_stage(
        simplifier,
        expr,
        target_expr,
        true,
        simplify_options.clone(),
    );
    if !stage_makes_progress(&mut simplifier.context, expr, &stage)
        || !derive_target_match(simplifier, stage.expr, target_expr)
        || !stage
            .steps
            .iter()
            .any(|step| step.rule_name == "Log Inverse Power")
    {
        return None;
    }

    if !collect_steps {
        stage.steps.clear();
    }
    retarget_stage_output(&mut stage, target_expr);
    let steps = finalize_steps(expr, target_expr, vec![stage], &mut simplifier.context);
    Some((target_expr, steps, DeriveStrategy::LogInversePower))
}

fn try_fast_direct_exponential_rewrite_derive(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<crate::Step>, DeriveStrategy)> {
    let mut stage = run_exponential_sum_diff_stage(simplifier, expr, target_expr, collect_steps);
    if !stage_makes_progress(&mut simplifier.context, expr, &stage)
        || !derive_target_match(simplifier, stage.expr, target_expr)
    {
        return None;
    }

    retarget_stage_output(&mut stage, target_expr);
    let steps = finalize_steps(expr, target_expr, vec![stage], &mut simplifier.context);
    Some((target_expr, steps, DeriveStrategy::ExponentialRewrite))
}

fn try_fast_direct_log_exp_power_inverse_derive(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<crate::Step>, DeriveStrategy)> {
    let plan =
        try_plan_log_exp_power_inverse_target_aware(&mut simplifier.context, expr, target_expr)?;

    let stages = if collect_steps {
        let mut power_step = crate::Step::with_snapshots(
            "Multiply exponents",
            "Power of a Power",
            plan.power_expr,
            plan.contracted_exp,
            vec![cas_solver_core::step_types::PathStep::Arg(0)],
            Some(&simplifier.context),
            expr,
            plan.normalized,
        );
        power_step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        power_step.category = cas_solver_core::step_types::StepCategory::Simplify;

        let mut inverse_step = crate::Step::with_snapshots(
            "Cancel ln(exp(u)) to u",
            "Log-Exp Inverse",
            plan.normalized,
            plan.result,
            Vec::new(),
            Some(&simplifier.context),
            plan.normalized,
            plan.result,
        );
        inverse_step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        inverse_step.category = cas_solver_core::step_types::StepCategory::Simplify;

        vec![
            DeriveStageOutput {
                expr: plan.normalized,
                steps: vec![power_step],
            },
            DeriveStageOutput {
                expr: plan.result,
                steps: vec![inverse_step],
            },
        ]
    } else {
        vec![
            DeriveStageOutput {
                expr: plan.normalized,
                steps: Vec::new(),
            },
            DeriveStageOutput {
                expr: plan.result,
                steps: Vec::new(),
            },
        ]
    };

    let steps = finalize_steps(expr, target_expr, stages, &mut simplifier.context);
    Some((target_expr, steps, DeriveStrategy::ExponentialRewrite))
}

fn contains_log_call_in_power_exponent(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            Expr::Pow(base, exponent) => {
                if contains_log_call_expr(ctx, *exponent) {
                    return true;
                }
                stack.push(*base);
                stack.push(*exponent);
            }
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right) => {
                stack.push(*left);
                stack.push(*right);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

fn contains_root_like_expr(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            Expr::Pow(base, exponent) => {
                if matches!(ctx.get(*exponent), Expr::Number(n) if !n.is_integer()) {
                    return true;
                }
                stack.push(*base);
                stack.push(*exponent);
            }
            Expr::Function(fn_id, args)
                if matches!(
                    ctx.builtin_of(*fn_id),
                    Some(BuiltinFn::Sqrt | BuiltinFn::Root)
                ) && !args.is_empty() =>
            {
                return true;
            }
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right) => {
                stack.push(*left);
                stack.push(*right);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

fn contains_log_call_expr(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            Expr::Function(fn_id, args) => {
                if matches!(
                    ctx.builtin_of(*fn_id),
                    Some(BuiltinFn::Ln | BuiltinFn::Log | BuiltinFn::Log10)
                ) {
                    return true;
                }
                stack.extend(args.iter().copied());
            }
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::Pow(left, right) => {
                stack.push(*left);
                stack.push(*right);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

fn ensure_log_expand_cleanup_step(
    ctx: &cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
    cleanup_stage: &mut DeriveStageOutput,
) {
    cleanup_stage.steps.clear();
    let mut step = crate::Step::with_snapshots(
        "log(b, x^y) = y * log(b, x)",
        "Evaluate Logarithms",
        source_expr,
        target_expr,
        Vec::new(),
        Some(ctx),
        source_expr,
        target_expr,
    );
    step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
    step.category = cas_solver_core::step_types::StepCategory::Simplify;
    cleanup_stage.steps.push(step);
}

fn run_log_expand_without_cleanup_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    let plan = cas_math::logarithm_inverse_support::expand_logs_collect_positive_assumptions(
        &mut simplifier.context,
        expr,
    );

    if plan.rewritten == expr {
        return DeriveStageOutput {
            expr,
            steps: Vec::new(),
        };
    }

    let assumed_positive = plan
        .assumed_positive
        .into_iter()
        .map(|subject| {
            cas_solver_core::assumption_model::AssumptionEvent::positive(
                &simplifier.context,
                subject,
            )
        })
        .collect::<Vec<_>>();

    let steps = if collect_steps {
        let mut step = crate::Step::with_snapshots(
            "Log expansion",
            "expand_log",
            expr,
            plan.rewritten,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            plan.rewritten,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Expand;
        step.meta_mut().assumption_events.extend(assumed_positive);
        vec![step]
    } else {
        Vec::new()
    };

    DeriveStageOutput {
        expr: plan.rewritten,
        steps,
    }
}

fn should_try_log_expand_cleanup_derive(
    ctx: &cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> bool {
    contains_expandable_log_call(ctx, expr)
        && count_log_calls_expr(ctx, target_expr) > count_log_calls_expr(ctx, expr)
}

fn contains_expandable_log_call(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            Expr::Function(fn_id, args) => {
                if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Ln | BuiltinFn::Log)) {
                    let candidate_arg = match args.as_slice() {
                        [arg] => Some(*arg),
                        [_, arg] => Some(*arg),
                        _ => None,
                    };
                    if candidate_arg.is_some_and(|arg| {
                        matches!(
                            ctx.get(arg),
                            Expr::Mul(_, _) | Expr::Div(_, _) | Expr::Pow(_, _)
                        )
                    }) {
                        return true;
                    }
                }
                stack.extend(args.iter().copied());
            }
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::Pow(left, right) => {
                stack.push(*left);
                stack.push(*right);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

fn count_log_calls_expr(ctx: &cas_ast::Context, expr: ExprId) -> usize {
    let mut count = 0;
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            Expr::Function(fn_id, args) => {
                if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Ln | BuiltinFn::Log)) {
                    count += 1;
                }
                stack.extend(args.iter().copied());
            }
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::Pow(left, right) => {
                stack.push(*left);
                stack.push(*right);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }
    count
}

fn try_fast_direct_trig_derive(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<crate::Step>, DeriveStrategy)> {
    if !contains_circular_trig_call(&simplifier.context, expr)
        && !contains_circular_trig_call(&simplifier.context, target_expr)
    {
        return None;
    }

    if let Some(direct) =
        try_fast_repeated_phase_shift_pair_derive(simplifier, expr, target_expr, collect_steps)
    {
        return Some(direct);
    }

    if let Some(rewrite) =
        try_rewrite_trig_special_value_target_aware(&mut simplifier.context, expr, target_expr)
    {
        let steps = if collect_steps {
            let mut step = crate::Step::with_snapshots(
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
                expr,
                target_expr,
                Vec::new(),
                Some(&simplifier.context),
                expr,
                target_expr,
            );
            step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
            step.category = cas_solver_core::step_types::StepCategory::Simplify;
            vec![step]
        } else {
            Vec::new()
        };

        return Some((target_expr, steps, DeriveStrategy::TrigRewrite));
    }

    if let Some(rewrite) =
        try_rewrite_shifted_double_angle_target_aware(&mut simplifier.context, expr, target_expr)
    {
        let final_expr =
            if cas_ast::ordering::compare_expr(&simplifier.context, rewrite.rewritten, target_expr)
                == std::cmp::Ordering::Equal
            {
                target_expr
            } else {
                rewrite.rewritten
            };
        let steps = if collect_steps {
            let mut step = crate::Step::with_snapshots(
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
                expr,
                final_expr,
                Vec::new(),
                Some(&simplifier.context),
                expr,
                final_expr,
            );
            step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
            step.category = cas_solver_core::step_types::StepCategory::Simplify;
            vec![step]
        } else {
            Vec::new()
        };

        return Some((final_expr, steps, DeriveStrategy::TrigExpand));
    }

    if let Some(rewrite) = try_rewrite_quadruple_sin_angle_contraction_target_aware(
        &mut simplifier.context,
        expr,
        target_expr,
    ) {
        let final_expr =
            if cas_ast::ordering::compare_expr(&simplifier.context, rewrite.rewritten, target_expr)
                == std::cmp::Ordering::Equal
            {
                target_expr
            } else {
                rewrite.rewritten
            };
        let steps = if collect_steps {
            let mut step = crate::Step::with_snapshots(
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
                expr,
                final_expr,
                Vec::new(),
                Some(&simplifier.context),
                expr,
                final_expr,
            );
            step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
            step.category = cas_solver_core::step_types::StepCategory::Simplify;
            vec![step]
        } else {
            Vec::new()
        };

        return Some((final_expr, steps, DeriveStrategy::TrigContract));
    }

    if let Some(rewrite) = try_rewrite_trig_expansion(&mut simplifier.context, expr, target_expr) {
        let steps = if collect_steps {
            vec![build_trig_expand_step(
                &simplifier.context,
                expr,
                target_expr,
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
            )]
        } else {
            Vec::new()
        };

        return Some((target_expr, steps, DeriveStrategy::TrigExpand));
    }

    if let Some(chain) = try_fast_product_to_sum_then_triple_angle_derive(
        simplifier,
        expr,
        target_expr,
        collect_steps,
    ) {
        return Some(chain);
    }

    if let Some(rewrite) =
        try_rewrite_trig_contraction_target_aware(&mut simplifier.context, expr, target_expr)
    {
        let final_expr =
            if cas_ast::ordering::compare_expr(&simplifier.context, rewrite.rewritten, target_expr)
                == std::cmp::Ordering::Equal
            {
                target_expr
            } else {
                rewrite.rewritten
            };
        let steps = if collect_steps {
            let mut step = crate::Step::with_snapshots(
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
                expr,
                final_expr,
                Vec::new(),
                Some(&simplifier.context),
                expr,
                final_expr,
            );
            step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
            step.category = cas_solver_core::step_types::StepCategory::Simplify;
            vec![step]
        } else {
            Vec::new()
        };

        return Some((final_expr, steps, DeriveStrategy::TrigContract));
    }

    if let Some(rewrite) =
        try_rewrite_trig_identity_to_one_target_aware(&mut simplifier.context, expr, target_expr)
    {
        let steps = if collect_steps {
            let mut step = crate::Step::with_snapshots(
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
                expr,
                target_expr,
                Vec::new(),
                Some(&simplifier.context),
                expr,
                target_expr,
            );
            step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
            step.category = cas_solver_core::step_types::StepCategory::Simplify;
            vec![step]
        } else {
            Vec::new()
        };

        return Some((target_expr, steps, DeriveStrategy::TrigRewrite));
    }

    if let Some(rewrite) = try_rewrite_shifted_reciprocal_pythagorean_target_aware(
        &mut simplifier.context,
        expr,
        target_expr,
    ) {
        let steps = if collect_steps {
            let mut step = crate::Step::with_snapshots(
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
                expr,
                target_expr,
                Vec::new(),
                Some(&simplifier.context),
                expr,
                target_expr,
            );
            step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
            step.category = cas_solver_core::step_types::StepCategory::Simplify;
            vec![step]
        } else {
            Vec::new()
        };

        return Some((target_expr, steps, DeriveStrategy::TrigRewrite));
    }

    if let Some(rewrite) =
        try_rewrite_pythagorean_factor_form_target_aware(&mut simplifier.context, expr, target_expr)
    {
        let steps = if collect_steps {
            let mut step = crate::Step::with_snapshots(
                &rewrite.description,
                "Pythagorean Factor Form",
                expr,
                target_expr,
                Vec::new(),
                Some(&simplifier.context),
                expr,
                target_expr,
            );
            step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
            step.category = cas_solver_core::step_types::StepCategory::Simplify;
            vec![step]
        } else {
            Vec::new()
        };

        return Some((target_expr, steps, DeriveStrategy::TrigRewrite));
    }

    None
}

fn try_fast_direct_factor_derive(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<crate::Step>, DeriveStrategy)> {
    if !looks_like_factored_target(&mut simplifier.context, target_expr) {
        return None;
    }

    let rewritten = try_rewrite_sixth_power_minus_one_factorization_target_aware(
        &mut simplifier.context,
        expr,
        target_expr,
    )
    .or_else(|| cas_math::factor::factor_binomial_cube_identity(&mut simplifier.context, expr))?;
    if !strong_target_match(&mut simplifier.context, rewritten, target_expr) {
        return None;
    }

    let final_expr = if cas_ast::ordering::compare_expr(&simplifier.context, rewritten, target_expr)
        == std::cmp::Ordering::Equal
    {
        target_expr
    } else {
        rewritten
    };

    let steps = if collect_steps {
        let mut step = crate::Step::with_snapshots(
            "Factorization",
            "Factorization",
            expr,
            final_expr,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            final_expr,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Factor;
        vec![step]
    } else {
        Vec::new()
    };

    Some((final_expr, steps, DeriveStrategy::Factor))
}

fn try_rewrite_sixth_power_minus_one_factorization_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<ExprId> {
    let base = extract_sixth_power_minus_one_base(ctx, expr)?;
    target_matches_full_sixth_power_minus_one_factorization(ctx, target_expr, base)
        .then_some(target_expr)
}

fn extract_sixth_power_minus_one_base(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Sub(left, right) = ctx.get(expr) else {
        return None;
    };
    if !is_integer_literal(ctx, *right, 1) {
        return None;
    }

    let Expr::Pow(base, exponent) = ctx.get(*left) else {
        return None;
    };
    is_integer_literal(ctx, *exponent, 6).then_some(*base)
}

fn target_matches_full_sixth_power_minus_one_factorization(
    ctx: &mut cas_ast::Context,
    target_expr: ExprId,
    base: ExprId,
) -> bool {
    let factors = flatten_mul_chain(ctx, target_expr);
    if factors.len() != 4 {
        return false;
    }

    let mut saw_plus_one = false;
    let mut saw_minus_one = false;
    let mut saw_positive_quadratic = false;
    let mut saw_negative_quadratic = false;

    for factor in factors {
        if !saw_plus_one && matches_linear_unit_factor(ctx, factor, base, Sign::Pos) {
            saw_plus_one = true;
            continue;
        }
        if !saw_minus_one && matches_linear_unit_factor(ctx, factor, base, Sign::Neg) {
            saw_minus_one = true;
            continue;
        }
        if !saw_positive_quadratic
            && matches_geometric_quadratic_factor(ctx, factor, base, Sign::Pos)
        {
            saw_positive_quadratic = true;
            continue;
        }
        if !saw_negative_quadratic
            && matches_geometric_quadratic_factor(ctx, factor, base, Sign::Neg)
        {
            saw_negative_quadratic = true;
            continue;
        }

        return false;
    }

    saw_plus_one && saw_minus_one && saw_positive_quadratic && saw_negative_quadratic
}

fn matches_linear_unit_factor(
    ctx: &mut cas_ast::Context,
    factor: ExprId,
    base: ExprId,
    constant_sign: Sign,
) -> bool {
    let terms = AddView::from_expr(ctx, factor).terms;
    if terms.len() != 2 {
        return false;
    }

    let mut saw_base = false;
    let mut saw_unit = false;
    for (term, sign) in terms {
        if sign == Sign::Pos
            && !saw_base
            && cas_ast::ordering::compare_expr(ctx, term, base) == std::cmp::Ordering::Equal
        {
            saw_base = true;
            continue;
        }
        if sign == constant_sign && !saw_unit && is_integer_literal(ctx, term, 1) {
            saw_unit = true;
            continue;
        }
    }

    saw_base && saw_unit
}

fn matches_geometric_quadratic_factor(
    ctx: &mut cas_ast::Context,
    factor: ExprId,
    base: ExprId,
    linear_sign: Sign,
) -> bool {
    let terms = AddView::from_expr(ctx, factor).terms;
    if terms.len() != 3 {
        return false;
    }

    let two = ctx.num(2);
    let base_squared = ctx.add(Expr::Pow(base, two));
    let mut saw_base_squared = false;
    let mut saw_linear = false;
    let mut saw_unit = false;

    for (term, sign) in terms {
        if sign == Sign::Pos
            && !saw_base_squared
            && cas_ast::ordering::compare_expr(ctx, term, base_squared) == std::cmp::Ordering::Equal
        {
            saw_base_squared = true;
            continue;
        }
        if sign == linear_sign
            && !saw_linear
            && cas_ast::ordering::compare_expr(ctx, term, base) == std::cmp::Ordering::Equal
        {
            saw_linear = true;
            continue;
        }
        if sign == Sign::Pos && !saw_unit && is_integer_literal(ctx, term, 1) {
            saw_unit = true;
            continue;
        }
    }

    saw_base_squared && saw_linear && saw_unit
}

fn is_integer_literal(ctx: &cas_ast::Context, expr: ExprId, value: i64) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(number) if number.is_integer() && number.to_integer() == value.into()
    )
}

fn try_fast_repeated_phase_shift_pair_derive(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<crate::Step>, DeriveStrategy)> {
    if !is_repeated_phase_shift_pair_candidate(&mut simplifier.context, expr, target_expr) {
        return None;
    }

    let expr_has_phase_shift = contains_phase_shift_term(&mut simplifier.context, expr);
    let target_has_phase_shift = contains_phase_shift_term(&mut simplifier.context, target_expr);
    let stage = try_run_two_step_additive_phase_shift_chain(
        simplifier,
        expr,
        target_expr,
        collect_steps,
        presentational_target_match,
    )?;

    let strategy = match (expr_has_phase_shift, target_has_phase_shift) {
        (false, true) => DeriveStrategy::TrigContract,
        (true, false) => DeriveStrategy::TrigExpand,
        _ => DeriveStrategy::Simplify,
    };

    Some((stage.expr, stage.steps, strategy))
}

fn is_repeated_phase_shift_pair_candidate(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> bool {
    let expr_terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
    let target_terms = cas_math::expr_nary::add_terms_signed(ctx, target_expr);

    let expr_is_shifted_pair = expr_terms.len() == 2
        && expr_terms
            .iter()
            .all(|(term, _)| contains_phase_shift_term(ctx, *term));
    let target_is_shifted_pair = target_terms.len() == 2
        && target_terms
            .iter()
            .all(|(term, _)| contains_phase_shift_term(ctx, *term));

    (expr_terms.len() == 4 && target_is_shifted_pair)
        || (target_terms.len() == 4 && expr_is_shifted_pair)
}

fn contains_circular_trig_call(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            Expr::Function(fn_id, args) => {
                if matches!(
                    ctx.builtin_of(*fn_id),
                    Some(
                        BuiltinFn::Sin
                            | BuiltinFn::Cos
                            | BuiltinFn::Tan
                            | BuiltinFn::Sec
                            | BuiltinFn::Csc
                            | BuiltinFn::Cot
                            | BuiltinFn::Asin
                            | BuiltinFn::Acos
                            | BuiltinFn::Atan
                            | BuiltinFn::Arcsin
                            | BuiltinFn::Arccos
                            | BuiltinFn::Arctan
                    )
                ) {
                    return true;
                }
                stack.extend(args.iter().copied());
            }
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::Pow(left, right) => {
                stack.push(*left);
                stack.push(*right);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

fn try_fast_direct_hyperbolic_derive(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<crate::Step>, DeriveStrategy)> {
    let expr_has_hyperbolic = contains_hyperbolic_fn_expr(&simplifier.context, expr);
    let target_has_hyperbolic = contains_hyperbolic_fn_expr(&simplifier.context, target_expr);

    if !expr_has_hyperbolic && !target_has_hyperbolic {
        return None;
    }

    if let Some(derived) = try_fast_direct_recursive_hyperbolic_angle_sum_derive(
        simplifier,
        expr,
        target_expr,
        collect_steps,
    ) {
        return Some(derived);
    }

    if let Some(kind) =
        try_match_exact_hyperbolic_sum_to_product_pair(&mut simplifier.context, expr, target_expr)
    {
        let steps = if collect_steps {
            vec![build_hyperbolic_bridge_step(
                &simplifier.context,
                expr,
                target_expr,
                kind.description(),
                kind.rule_name(),
            )]
        } else {
            Vec::new()
        };

        return Some((target_expr, steps, DeriveStrategy::Expand));
    }

    let target_is_pure_exponential =
        contains_exponential_like_expr(&mut simplifier.context, target_expr)
            && !contains_hyperbolic_fn_expr(&simplifier.context, target_expr);

    let direct_hyperbolic_rewrite = if target_is_pure_exponential {
        try_rewrite_hyperbolic_exponential_bridge_target_aware(
            &mut simplifier.context,
            expr,
            target_expr,
        )
    } else {
        try_rewrite_hyperbolic_simplify_target_aware(&mut simplifier.context, expr, target_expr)
    };

    if let Some(rewrite) = direct_hyperbolic_rewrite {
        let steps = if collect_steps {
            vec![build_hyperbolic_bridge_step(
                &simplifier.context,
                expr,
                target_expr,
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
            )]
        } else {
            Vec::new()
        };

        if matches!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ProductToSumSinhCosh
                | DeriveHyperbolicRewriteKind::ProductToSumCoshCosh
                | DeriveHyperbolicRewriteKind::ProductToSumSinhSinh
                | DeriveHyperbolicRewriteKind::SumToProductSinhCosh
                | DeriveHyperbolicRewriteKind::SumToProductCoshCosh
                | DeriveHyperbolicRewriteKind::SumToProductSinhSinh
                | DeriveHyperbolicRewriteKind::ContractSinhAngleSumDiff
                | DeriveHyperbolicRewriteKind::ContractCoshAngleSumDiff
        ) {
            if is_hyperbolic_product_sum_kind(rewrite.kind)
                && !fast_hyperbolic_product_sum_target_match(
                    &mut simplifier.context,
                    rewrite.rewritten,
                    target_expr,
                    rewrite.kind,
                )
            {
                // Preserve product-to-sum -> triple-angle derivations as two visible steps.
            } else {
                return Some((
                    target_expr,
                    steps,
                    fast_hyperbolic_simplify_strategy(rewrite.kind),
                ));
            }
        } else {
            return Some((
                target_expr,
                steps,
                fast_hyperbolic_simplify_strategy(rewrite.kind),
            ));
        }
    }

    for rewrite in generate_hyperbolic_bridge_rewrites(&mut simplifier.context, expr) {
        if !is_hyperbolic_product_sum_kind(rewrite.kind) {
            continue;
        }

        if !fast_hyperbolic_product_sum_target_match(
            &mut simplifier.context,
            rewrite.rewritten,
            target_expr,
            rewrite.kind,
        ) {
            continue;
        }

        let steps = if collect_steps {
            vec![build_hyperbolic_bridge_step(
                &simplifier.context,
                expr,
                target_expr,
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
            )]
        } else {
            Vec::new()
        };

        return Some((target_expr, steps, DeriveStrategy::Expand));
    }

    if let Some(kind) =
        try_rewrite_hyperbolic_expansion_target_aware(&mut simplifier.context, expr, target_expr)
    {
        if is_hyperbolic_product_sum_kind(kind) {
            // Keep product-to-sum polynomial targets on the didactic multi-step path.
        } else {
            let steps = if collect_steps {
                vec![build_hyperbolic_bridge_step(
                    &simplifier.context,
                    expr,
                    target_expr,
                    kind.description(),
                    kind.rule_name(),
                )]
            } else {
                Vec::new()
            };

            return Some((target_expr, steps, DeriveStrategy::Expand));
        }
    }

    for rewrite in generate_hyperbolic_bridge_rewrites(&mut simplifier.context, expr) {
        if !matches!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ProductToSumSinhCosh
                | DeriveHyperbolicRewriteKind::ProductToSumCoshCosh
                | DeriveHyperbolicRewriteKind::ProductToSumSinhSinh
                | DeriveHyperbolicRewriteKind::SumToProductSinhCosh
                | DeriveHyperbolicRewriteKind::SumToProductCoshCosh
                | DeriveHyperbolicRewriteKind::SumToProductSinhSinh
                | DeriveHyperbolicRewriteKind::ContractSinhAngleSumDiff
                | DeriveHyperbolicRewriteKind::ContractCoshAngleSumDiff
        ) {
            continue;
        }

        if is_hyperbolic_product_sum_kind(rewrite.kind)
            && !fast_hyperbolic_product_sum_target_match(
                &mut simplifier.context,
                rewrite.rewritten,
                target_expr,
                rewrite.kind,
            )
        {
            continue;
        }

        if !strong_target_match(&mut simplifier.context, rewrite.rewritten, target_expr) {
            continue;
        }

        let steps = if collect_steps {
            vec![build_hyperbolic_bridge_step(
                &simplifier.context,
                expr,
                target_expr,
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
            )]
        } else {
            Vec::new()
        };

        return Some((
            target_expr,
            steps,
            fast_hyperbolic_simplify_strategy(rewrite.kind),
        ));
    }

    for rewrite in generate_hyperbolic_bridge_rewrites(&mut simplifier.context, target_expr) {
        if !matches!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ProductToSumSinhCosh
                | DeriveHyperbolicRewriteKind::ProductToSumCoshCosh
                | DeriveHyperbolicRewriteKind::ProductToSumSinhSinh
                | DeriveHyperbolicRewriteKind::ContractSinhAngleSumDiff
                | DeriveHyperbolicRewriteKind::ContractCoshAngleSumDiff
        ) {
            continue;
        }

        if is_hyperbolic_product_sum_kind(rewrite.kind)
            && !fast_hyperbolic_product_sum_target_match(
                &mut simplifier.context,
                rewrite.rewritten,
                expr,
                rewrite.kind,
            )
        {
            continue;
        }

        if !strong_target_match(&mut simplifier.context, rewrite.rewritten, expr) {
            continue;
        }

        let steps = if collect_steps {
            vec![build_hyperbolic_bridge_step(
                &simplifier.context,
                expr,
                target_expr,
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
            )]
        } else {
            Vec::new()
        };

        return Some((
            target_expr,
            steps,
            fast_hyperbolic_simplify_strategy(rewrite.kind),
        ));
    }

    None
}

fn contains_exponential_like_expr(ctx: &mut cas_ast::Context, expr: ExprId) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        if cas_math::expr_extract::extract_exp_argument(ctx, current).is_some() {
            return true;
        }

        match ctx.get(current) {
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::Pow(left, right) => {
                stack.push(*left);
                stack.push(*right);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }

    false
}

fn contains_hyperbolic_fn_expr(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            Expr::Function(fn_id, args) => {
                if matches!(
                    ctx.builtin_of(*fn_id),
                    Some(
                        BuiltinFn::Sinh
                            | BuiltinFn::Cosh
                            | BuiltinFn::Tanh
                            | BuiltinFn::Asinh
                            | BuiltinFn::Acosh
                            | BuiltinFn::Atanh
                    )
                ) {
                    return true;
                }
                stack.extend(args.iter().copied());
            }
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::Pow(left, right) => {
                stack.push(*left);
                stack.push(*right);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }

    false
}

fn try_fast_direct_recursive_hyperbolic_angle_sum_derive(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<crate::Step>, DeriveStrategy)> {
    let (source_fn, source_arg) = derive_hyperbolic_direct_call_arg(&simplifier.context, expr)?;
    let expected_kind = match source_fn {
        BuiltinFn::Sinh => DeriveHyperbolicRewriteKind::ContractSinhAngleSumDiff,
        BuiltinFn::Cosh => DeriveHyperbolicRewriteKind::ContractCoshAngleSumDiff,
        _ => return None,
    };

    for rewrite in generate_hyperbolic_bridge_rewrites(&mut simplifier.context, target_expr) {
        if rewrite.kind != expected_kind {
            continue;
        }

        let Some((candidate_fn, candidate_arg)) =
            derive_hyperbolic_direct_call_arg(&simplifier.context, rewrite.rewritten)
        else {
            continue;
        };
        if candidate_fn != source_fn {
            continue;
        }

        if !recursive_hyperbolic_angle_arg_match(simplifier, source_arg, candidate_arg) {
            continue;
        }

        let steps = if collect_steps {
            vec![build_hyperbolic_bridge_step(
                &simplifier.context,
                expr,
                target_expr,
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
            )]
        } else {
            Vec::new()
        };

        return Some((target_expr, steps, DeriveStrategy::Expand));
    }

    None
}

fn derive_hyperbolic_direct_call_arg(
    ctx: &cas_ast::Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let builtin = ctx.builtin_of(*fn_id)?;
    match builtin {
        BuiltinFn::Sinh | BuiltinFn::Cosh => Some((builtin, args[0])),
        _ => None,
    }
}

fn recursive_hyperbolic_angle_arg_match(
    simplifier: &mut crate::Simplifier,
    source_arg: ExprId,
    candidate_arg: ExprId,
) -> bool {
    if strong_target_match(&mut simplifier.context, source_arg, candidate_arg) {
        return true;
    }

    let simplified_candidate = simplify_expr_with_default_rules(simplifier, candidate_arg);
    if strong_target_match(&mut simplifier.context, source_arg, simplified_candidate) {
        return true;
    }

    let simplified_source = simplify_expr_with_default_rules(simplifier, source_arg);
    strong_target_match(&mut simplifier.context, simplified_source, candidate_arg)
        || strong_target_match(
            &mut simplifier.context,
            simplified_source,
            simplified_candidate,
        )
}

fn simplify_expr_with_default_rules(simplifier: &mut crate::Simplifier, expr: ExprId) -> ExprId {
    let mut temp = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut temp.context, &mut simplifier.context);
    let (simplified, _steps, _stats) = temp.simplify_with_stats(
        expr,
        crate::SimplifyOptions {
            suppress_depth_overflow_warnings: true,
            ..crate::SimplifyOptions::default()
        },
    );
    std::mem::swap(&mut temp.context, &mut simplifier.context);
    simplified
}

fn is_hyperbolic_product_sum_kind(kind: DeriveHyperbolicRewriteKind) -> bool {
    matches!(
        kind,
        DeriveHyperbolicRewriteKind::ProductToSumSinhCosh
            | DeriveHyperbolicRewriteKind::ProductToSumCoshCosh
            | DeriveHyperbolicRewriteKind::ProductToSumSinhSinh
            | DeriveHyperbolicRewriteKind::SumToProductSinhCosh
            | DeriveHyperbolicRewriteKind::SumToProductCoshCosh
            | DeriveHyperbolicRewriteKind::SumToProductSinhSinh
    )
}

fn fast_hyperbolic_simplify_strategy(kind: DeriveHyperbolicRewriteKind) -> DeriveStrategy {
    if is_hyperbolic_product_sum_kind(kind) {
        DeriveStrategy::Expand
    } else {
        DeriveStrategy::HyperbolicRewrite
    }
}

fn fast_hyperbolic_product_sum_target_match(
    ctx: &mut cas_ast::Context,
    actual: ExprId,
    target: ExprId,
    kind: DeriveHyperbolicRewriteKind,
) -> bool {
    if presentational_target_match(ctx, actual, target) {
        return true;
    }

    matches!(
        kind,
        DeriveHyperbolicRewriteKind::SumToProductSinhCosh
            | DeriveHyperbolicRewriteKind::SumToProductCoshCosh
    ) && strong_target_match(ctx, actual, target)
}

fn try_match_exact_hyperbolic_sum_to_product_pair(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewriteKind> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
    if terms.len() != 2 {
        return None;
    }

    let (first_fn, first_arg) = hyperbolic_direct_call_arg(ctx, terms[0].0)?;
    let (second_fn, second_arg) = hyperbolic_direct_call_arg(ctx, terms[1].0)?;
    if first_fn != second_fn {
        return None;
    }

    let target_factors = cas_math::trig_roots_flatten::flatten_mul_chain(ctx, target_expr);
    if target_factors.len() != 3 {
        return None;
    }

    let two = ctx.num(2);
    let mut non_numeric_factors = Vec::with_capacity(2);
    let mut saw_two = false;
    for factor in target_factors {
        if cas_ast::ordering::compare_expr(ctx, factor, two) == std::cmp::Ordering::Equal {
            if saw_two {
                return None;
            }
            saw_two = true;
        } else {
            non_numeric_factors.push(factor);
        }
    }
    if !saw_two || non_numeric_factors.len() != 2 {
        return None;
    }

    let factors_match = |ctx: &mut cas_ast::Context, left: ExprId, right: ExprId| {
        (strong_target_match(ctx, non_numeric_factors[0], left)
            && strong_target_match(ctx, non_numeric_factors[1], right))
            || (strong_target_match(ctx, non_numeric_factors[0], right)
                && strong_target_match(ctx, non_numeric_factors[1], left))
    };

    if terms[0].1 == terms[1].1 {
        let avg = build_half_sum_expr(ctx, first_arg, second_arg);
        let diff = build_half_diff_expr(ctx, first_arg, second_arg);
        let flipped_diff = build_half_diff_expr(ctx, second_arg, first_arg);

        return match first_fn {
            BuiltinFn::Sinh => {
                let avg_term = ctx.call_builtin(BuiltinFn::Sinh, vec![avg]);
                let diff_term = ctx.call_builtin(BuiltinFn::Cosh, vec![diff]);
                let flipped_diff_term = ctx.call_builtin(BuiltinFn::Cosh, vec![flipped_diff]);
                if factors_match(ctx, avg_term, diff_term)
                    || factors_match(ctx, avg_term, flipped_diff_term)
                {
                    Some(DeriveHyperbolicRewriteKind::SumToProductSinhCosh)
                } else {
                    None
                }
            }
            BuiltinFn::Cosh => {
                let avg_term = ctx.call_builtin(BuiltinFn::Cosh, vec![avg]);
                let diff_term = ctx.call_builtin(BuiltinFn::Cosh, vec![diff]);
                let flipped_diff_term = ctx.call_builtin(BuiltinFn::Cosh, vec![flipped_diff]);
                if factors_match(ctx, avg_term, diff_term)
                    || factors_match(ctx, avg_term, flipped_diff_term)
                {
                    Some(DeriveHyperbolicRewriteKind::SumToProductCoshCosh)
                } else {
                    None
                }
            }
            _ => None,
        };
    }

    let (positive_arg, negative_arg) = if terms[0].1 == cas_math::expr_nary::Sign::Pos {
        (first_arg, second_arg)
    } else {
        (second_arg, first_arg)
    };
    let avg = build_half_sum_expr(ctx, positive_arg, negative_arg);
    let diff = build_half_diff_expr(ctx, positive_arg, negative_arg);

    match first_fn {
        BuiltinFn::Sinh => {
            let avg_term = ctx.call_builtin(BuiltinFn::Cosh, vec![avg]);
            let diff_term = ctx.call_builtin(BuiltinFn::Sinh, vec![diff]);
            factors_match(ctx, avg_term, diff_term)
                .then_some(DeriveHyperbolicRewriteKind::SumToProductSinhCosh)
        }
        BuiltinFn::Cosh => {
            let avg_term = ctx.call_builtin(BuiltinFn::Sinh, vec![avg]);
            let diff_term = ctx.call_builtin(BuiltinFn::Sinh, vec![diff]);
            factors_match(ctx, avg_term, diff_term)
                .then_some(DeriveHyperbolicRewriteKind::SumToProductSinhSinh)
        }
        _ => None,
    }
}

fn hyperbolic_direct_call_arg(ctx: &cas_ast::Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let builtin = ctx.builtin_of(*fn_id)?;
    match builtin {
        BuiltinFn::Sinh | BuiltinFn::Cosh => Some((builtin, args[0])),
        _ => None,
    }
}

fn build_half_sum_expr(ctx: &mut cas_ast::Context, left: ExprId, right: ExprId) -> ExprId {
    let two = ctx.num(2);
    let sum = ctx.add(Expr::Add(left, right));
    ctx.add(Expr::Div(sum, two))
}

fn build_half_diff_expr(ctx: &mut cas_ast::Context, left: ExprId, right: ExprId) -> ExprId {
    let two = ctx.num(2);
    let diff = ctx.add(Expr::Sub(left, right));
    ctx.add(Expr::Div(diff, two))
}

fn generate_planner_candidate_stages(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
    simplify_options: &crate::SimplifyOptions,
) -> Vec<DeriveStageOutput> {
    let mut stages = Vec::new();

    stages.extend(
        generate_hyperbolic_bridge_rewrites(&mut simplifier.context, expr)
            .into_iter()
            .map(|rewrite| {
                let steps = if collect_steps {
                    vec![build_hyperbolic_bridge_step(
                        &simplifier.context,
                        expr,
                        rewrite.rewritten,
                        rewrite.kind.description(),
                        rewrite.kind.rule_name(),
                    )]
                } else {
                    Vec::new()
                };

                DeriveStageOutput {
                    expr: rewrite.rewritten,
                    steps,
                }
            }),
    );

    for rewrite in generate_hyperbolic_additive_term_bridge_rewrites(&mut simplifier.context, expr)
    {
        let steps = if collect_steps {
            vec![build_hyperbolic_bridge_step(
                &simplifier.context,
                expr,
                rewrite.rewritten,
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
            )]
        } else {
            Vec::new()
        };

        stages.push(DeriveStageOutput {
            expr: rewrite.rewritten,
            steps,
        });
    }

    stages.extend(
        generate_trig_bridge_rewrites(&mut simplifier.context, expr)
            .into_iter()
            .map(|rewrite| {
                let steps = if collect_steps {
                    vec![build_trig_bridge_step(
                        &simplifier.context,
                        expr,
                        rewrite.rewritten,
                        rewrite.kind.description(),
                        rewrite.kind.rule_name(),
                    )]
                } else {
                    Vec::new()
                };

                DeriveStageOutput {
                    expr: rewrite.rewritten,
                    steps,
                }
            }),
    );

    for rewrite in generate_trig_additive_term_bridge_rewrites(&mut simplifier.context, expr) {
        let steps = if collect_steps {
            vec![build_trig_bridge_step(
                &simplifier.context,
                expr,
                rewrite.rewritten,
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
            )]
        } else {
            Vec::new()
        };

        stages.push(DeriveStageOutput {
            expr: rewrite.rewritten,
            steps,
        });
    }

    if let Some(rewrite) = try_rewrite_trig_expansion(&mut simplifier.context, expr, target_expr) {
        let steps = if collect_steps {
            vec![build_trig_bridge_step(
                &simplifier.context,
                expr,
                rewrite.rewritten,
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
            )]
        } else {
            Vec::new()
        };

        stages.push(DeriveStageOutput {
            expr: rewrite.rewritten,
            steps,
        });
    }

    if let Some(rewrite) =
        try_rewrite_trig_contraction_target_aware(&mut simplifier.context, expr, target_expr)
    {
        let steps = if collect_steps {
            vec![build_trig_bridge_step(
                &simplifier.context,
                expr,
                rewrite.rewritten,
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
            )]
        } else {
            Vec::new()
        };

        stages.push(DeriveStageOutput {
            expr: rewrite.rewritten,
            steps,
        });
    }

    let log_contract_stage = run_log_contract_stage(simplifier, expr, target_expr, collect_steps);
    if !presentational_target_match(&mut simplifier.context, log_contract_stage.expr, expr) {
        stages.push(log_contract_stage);
    }

    if !should_try_trig_planner_before_simplify(&mut simplifier.context, expr, target_expr) {
        let simplify_stage = run_simplify_stage(
            simplifier,
            expr,
            target_expr,
            collect_steps,
            simplify_options.clone(),
        );
        if !presentational_target_match(&mut simplifier.context, simplify_stage.expr, expr) {
            stages.push(simplify_stage);
        }
    }

    stages
}

fn build_hyperbolic_bridge_step(
    ctx: &cas_ast::Context,
    before: ExprId,
    after: ExprId,
    description: &str,
    rule_name: &str,
) -> crate::Step {
    let mut step = crate::Step::with_snapshots(
        description,
        rule_name,
        before,
        after,
        Vec::new(),
        Some(ctx),
        before,
        after,
    );
    step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
    step.category = cas_solver_core::step_types::StepCategory::Simplify;
    step
}

fn build_trig_bridge_step(
    ctx: &cas_ast::Context,
    before: ExprId,
    after: ExprId,
    description: &str,
    rule_name: &str,
) -> crate::Step {
    let mut step = crate::Step::with_snapshots(
        description,
        rule_name,
        before,
        after,
        Vec::new(),
        Some(ctx),
        before,
        after,
    );
    step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
    step.category = cas_solver_core::step_types::StepCategory::Simplify;
    step
}

fn build_trig_expand_step(
    ctx: &cas_ast::Context,
    before: ExprId,
    after: ExprId,
    description: &str,
    rule_name: &str,
) -> crate::Step {
    let mut step = crate::Step::with_snapshots(
        description,
        rule_name,
        before,
        after,
        Vec::new(),
        Some(ctx),
        before,
        after,
    );
    step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
    step.category = cas_solver_core::step_types::StepCategory::Expand;
    step
}

fn derive_expr_signature(ctx: &cas_ast::Context, expr: ExprId) -> String {
    cas_formatter::clean_display_string(&cas_formatter::render_expr(ctx, expr))
}

fn run_expand_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    if let Some(stage) = try_run_hyperbolic_product_sum_then_triple_angle_expand_chain(
        simplifier,
        expr,
        target_expr,
        collect_steps,
    ) {
        return stage;
    }

    if let Some(rewrite) =
        try_rewrite_expanded_target_aware(&mut simplifier.context, expr, target_expr)
    {
        if !matches!(
            rewrite.kind,
            ExpandRewriteKind::SophieGermainProduct
                | ExpandRewriteKind::HyperbolicAngleSumDiff
                | ExpandRewriteKind::HyperbolicProductSum
        ) {
            let (expanded_expr, steps) = simplifier.expand(expr);
            if derive_target_match(simplifier, expanded_expr, target_expr) {
                return DeriveStageOutput {
                    expr: expanded_expr,
                    steps: if collect_steps { steps } else { Vec::new() },
                };
            }
        }

        let steps = if collect_steps {
            vec![build_expand_step(
                &simplifier.context,
                expr,
                rewrite.rewritten,
                rewrite.kind,
            )]
        } else {
            Vec::new()
        };
        return DeriveStageOutput {
            expr: rewrite.rewritten,
            steps,
        };
    }

    let (expr, steps) = simplifier.expand(expr);
    DeriveStageOutput { expr, steps }
}

fn try_run_hyperbolic_product_sum_then_triple_angle_expand_chain(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> Option<DeriveStageOutput> {
    let mut first_stage_candidates =
        generate_hyperbolic_bridge_rewrites(&mut simplifier.context, expr);
    first_stage_candidates.extend(generate_hyperbolic_additive_term_bridge_rewrites(
        &mut simplifier.context,
        expr,
    ));

    for bridge in first_stage_candidates {
        if !matches!(
            bridge.kind,
            DeriveHyperbolicRewriteKind::ProductToSumSinhCosh
                | DeriveHyperbolicRewriteKind::ProductToSumCoshCosh
                | DeriveHyperbolicRewriteKind::ProductToSumSinhSinh
                | DeriveHyperbolicRewriteKind::SumToProductSinhCosh
                | DeriveHyperbolicRewriteKind::SumToProductCoshCosh
                | DeriveHyperbolicRewriteKind::SumToProductSinhSinh
        ) {
            continue;
        }

        for rewrite in
            generate_hyperbolic_bridge_rewrites(&mut simplifier.context, bridge.rewritten)
        {
            if !matches!(
                rewrite.kind,
                DeriveHyperbolicRewriteKind::ExpandCombinedSinhTripleAngle
                    | DeriveHyperbolicRewriteKind::ExpandCombinedCoshTripleAngle
            ) {
                continue;
            }

            if !strong_target_match(&mut simplifier.context, rewrite.rewritten, target_expr) {
                continue;
            }

            let steps = if collect_steps {
                vec![
                    build_expand_step(
                        &simplifier.context,
                        expr,
                        bridge.rewritten,
                        ExpandRewriteKind::HyperbolicProductSum,
                    ),
                    build_hyperbolic_bridge_step(
                        &simplifier.context,
                        bridge.rewritten,
                        rewrite.rewritten,
                        rewrite.kind.description(),
                        rewrite.kind.rule_name(),
                    ),
                ]
            } else {
                Vec::new()
            };

            return Some(DeriveStageOutput {
                expr: rewrite.rewritten,
                steps,
            });
        }
    }

    None
}

fn build_expand_step(
    ctx: &cas_ast::Context,
    before: ExprId,
    after: ExprId,
    kind: ExpandRewriteKind,
) -> crate::Step {
    let (description, rule_name) = match kind {
        ExpandRewriteKind::BinomialPower => ("Expand the binomial power", "Binomial Expansion"),
        ExpandRewriteKind::SophieGermainProduct => (
            "Expand the Sophie Germain identity",
            "Sophie Germain Identity",
        ),
        ExpandRewriteKind::HyperbolicAngleSumDiff => (
            "Expand a hyperbolic angle sum/difference identity",
            "Hyperbolic Angle Sum/Difference Identity",
        ),
        ExpandRewriteKind::HyperbolicProductSum => (
            "Apply a hyperbolic product-to-sum or sum-to-product identity",
            "Hyperbolic Product-to-Sum Identity",
        ),
        ExpandRewriteKind::General => ("Expand the expression distributively", "Expand"),
    };

    let mut step = crate::Step::with_snapshots(
        description,
        rule_name,
        before,
        after,
        Vec::new(),
        Some(ctx),
        before,
        after,
    );
    step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
    step.category = cas_solver_core::step_types::StepCategory::Expand;
    step
}

fn run_fraction_expand_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
    simplify_options: crate::SimplifyOptions,
) -> DeriveStageOutput {
    let Some(rewrite) = try_rewrite_fraction_expansion_target_aware(
        simplifier,
        expr,
        target_expr,
        simplify_options,
    ) else {
        return DeriveStageOutput {
            expr,
            steps: Vec::new(),
        };
    };

    let steps = if collect_steps {
        let rule_name = rewrite.kind.rule_name();
        let substeps = telescoping_fraction_didactic_substeps(
            &mut simplifier.context,
            expr,
            rewrite.rewritten,
            rule_name,
        );
        let mut step = crate::Step::with_snapshots(
            rewrite.kind.description(),
            rule_name,
            expr,
            rewrite.rewritten,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            rewrite.rewritten,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Expand;
        if let Some(focus_before) = rewrite.focus_before {
            step.meta_mut().before_local = Some(focus_before);
        }
        if let Some(focus_after) = rewrite.focus_after {
            step.meta_mut().after_local = Some(focus_after);
        } else if rewrite.intermediate != rewrite.rewritten {
            step.meta_mut().after_local = Some(rewrite.intermediate);
        }
        if !substeps.is_empty() {
            step.meta_mut().substeps = substeps;
        }
        vec![step]
    } else {
        Vec::new()
    };

    DeriveStageOutput {
        expr: rewrite.rewritten,
        steps,
    }
}

fn run_rationalize_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    let Some(rewrite) =
        try_rewrite_rationalized_target_aware(&mut simplifier.context, expr, target_expr)
    else {
        return DeriveStageOutput {
            expr,
            steps: Vec::new(),
        };
    };

    let steps = if collect_steps {
        match rewrite.kind {
            RationalizeRewriteKind::RadicalNotableQuotient => {
                let mut step = crate::Step::with_snapshots(
                    rewrite.kind.description(),
                    rewrite.kind.rule_name(),
                    expr,
                    rewrite.rewritten,
                    Vec::new(),
                    Some(&simplifier.context),
                    expr,
                    rewrite.rewritten,
                );
                step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
                step.category = cas_solver_core::step_types::StepCategory::Simplify;
                step.meta_mut().before_local = Some(expr);
                step.meta_mut().after_local = Some(rewrite.intermediate);
                step.meta_mut().substeps = vec![
                    cas_solver_core::step_types::SubStep::new(
                        "Llamar t = sqrt(x) para reconocer la forma",
                        Vec::new(),
                    ),
                    cas_solver_core::step_types::SubStep::new(
                        "Ese cociente notable se convierte en t^2 + t + 1",
                        Vec::new(),
                    ),
                    cas_solver_core::step_types::SubStep::new(
                        "Volver a poner t = sqrt(x)",
                        Vec::new(),
                    ),
                    cas_solver_core::step_types::SubStep::new(
                        "Deshacer sqrt(x)^2 como x dentro del resultado",
                        Vec::new(),
                    ),
                ];
                vec![step]
            }
            RationalizeRewriteKind::CancelToZeroAfterRationalize => {
                let mut rationalize_step = crate::Step::with_snapshots(
                    rewrite.kind.description(),
                    rewrite.kind.rule_name(),
                    expr,
                    rewrite.intermediate,
                    Vec::new(),
                    Some(&simplifier.context),
                    expr,
                    rewrite.intermediate,
                );
                rationalize_step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
                rationalize_step.category = cas_solver_core::step_types::StepCategory::Simplify;

                let mut cancel_step = crate::Step::with_snapshots(
                    "Subtract two identical expressions",
                    "Subtraction Self-Cancel",
                    rewrite.intermediate,
                    rewrite.rewritten,
                    Vec::new(),
                    Some(&simplifier.context),
                    rewrite.intermediate,
                    rewrite.rewritten,
                );
                cancel_step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
                cancel_step.category = cas_solver_core::step_types::StepCategory::Simplify;
                vec![rationalize_step, cancel_step]
            }
        }
    } else {
        Vec::new()
    };

    DeriveStageOutput {
        expr: rewrite.rewritten,
        steps,
    }
}

fn run_power_merge_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    let Some(rewrite) =
        try_rewrite_power_merge_target_aware(&mut simplifier.context, expr, target_expr)
    else {
        return DeriveStageOutput {
            expr,
            steps: Vec::new(),
        };
    };

    let mut steps = Vec::new();
    if collect_steps {
        let mut merge_step = crate::Step::with_snapshots(
            "Combine powers with same base (n-ary)",
            "Combine powers with same base (n-ary)",
            expr,
            rewrite.rewritten,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            rewrite.rewritten,
        );
        merge_step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        merge_step.category = cas_solver_core::step_types::StepCategory::Simplify;
        steps.push(merge_step);
    }

    DeriveStageOutput {
        expr: rewrite.rewritten,
        steps,
    }
}

fn run_fraction_decompose_stage(
    simplifier: &mut crate::Simplifier,
    source_expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    if !matches!(simplifier.context.get(source_expr), Expr::Div(_, _)) {
        return DeriveStageOutput {
            expr: source_expr,
            steps: Vec::new(),
        };
    }

    let Some(recombined) =
        try_build_combined_fraction_from_fold_add(&mut simplifier.context, target_expr)
    else {
        return DeriveStageOutput {
            expr: source_expr,
            steps: Vec::new(),
        };
    };

    if !derive_target_match(simplifier, recombined, source_expr) {
        return DeriveStageOutput {
            expr: source_expr,
            steps: Vec::new(),
        };
    }

    let steps = if collect_steps {
        let mut step = crate::Step::with_snapshots(
            "Split a fraction into a whole part plus remainder",
            "Mixed Fraction Split",
            source_expr,
            target_expr,
            Vec::new(),
            Some(&simplifier.context),
            source_expr,
            target_expr,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Expand;
        vec![step]
    } else {
        Vec::new()
    };

    DeriveStageOutput {
        expr: target_expr,
        steps,
    }
}

fn run_fraction_combine_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    let Some(rewrite) =
        try_rewrite_fraction_combination_target_aware(&mut simplifier.context, expr, target_expr)
    else {
        return DeriveStageOutput {
            expr,
            steps: Vec::new(),
        };
    };

    let steps = if collect_steps {
        let rule_name = rewrite.kind.rule_name();
        let substeps = telescoping_fraction_didactic_substeps(
            &mut simplifier.context,
            expr,
            rewrite.rewritten,
            rule_name,
        );
        let mut step = crate::Step::with_snapshots(
            rewrite.kind.description(),
            rule_name,
            expr,
            rewrite.rewritten,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            rewrite.rewritten,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Expand;
        if let Some(focus_before) = rewrite.focus_before {
            step.meta_mut().before_local = Some(focus_before);
        }
        if let Some(focus_after) = rewrite.focus_after {
            step.meta_mut().after_local = Some(focus_after);
        }
        if !substeps.is_empty() {
            step.meta_mut().substeps = substeps;
        }
        vec![step]
    } else {
        Vec::new()
    };

    DeriveStageOutput {
        expr: rewrite.rewritten,
        steps,
    }
}

fn run_odd_half_power_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    let rewritten_source = if let Some(rewritten) =
        try_rewrite_odd_half_power_to_target_aware(&mut simplifier.context, expr, target_expr)
    {
        rewritten
    } else {
        let (silently_simplified, _steps, _stats) = simplifier.simplify_with_stats(
            expr,
            crate::SimplifyOptions {
                collect_steps: false,
                suppress_depth_overflow_warnings: true,
                ..crate::SimplifyOptions::default()
            },
        );
        let Some(rewritten) = try_rewrite_odd_half_power_to_target_aware(
            &mut simplifier.context,
            silently_simplified,
            target_expr,
        ) else {
            return DeriveStageOutput {
                expr,
                steps: Vec::new(),
            };
        };
        rewritten
    };

    let steps = if collect_steps {
        let mut step = crate::Step::with_snapshots(
            "Rewrite an odd half-integer power using a square root",
            "Expand Odd Half Power",
            expr,
            rewritten_source,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            rewritten_source,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Expand;
        vec![step]
    } else {
        Vec::new()
    };

    DeriveStageOutput {
        expr: rewritten_source,
        steps,
    }
}

fn run_factor_with_division_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    candidate_variables: &[String],
    collect_steps: bool,
) -> Option<DeriveStageOutput> {
    for var_name in candidate_variables {
        let Some((factored_expr, inner_target)) =
            extract_factored_division_target(&mut simplifier.context, target_expr, var_name)
        else {
            continue;
        };

        let quotient_expr = simplifier
            .context
            .add(cas_ast::Expr::Div(expr, factored_expr));

        let Some((_inner_expr, _inner_steps, _strategy)) = try_supported_derive_strategies_inner(
            simplifier,
            quotient_expr,
            inner_target,
            false,
            &crate::SimplifyOptions::default(),
            false,
            false,
        ) else {
            continue;
        };

        let steps = if collect_steps {
            let factor_display = cas_formatter::render_expr(&simplifier.context, factored_expr);
            let mut step = crate::Step::with_snapshots(
                &format!("Factor out {factor_display} from the whole expression"),
                "Factor Out With Division",
                expr,
                target_expr,
                Vec::new(),
                Some(&simplifier.context),
                expr,
                target_expr,
            );
            step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
            step.category = cas_solver_core::step_types::StepCategory::Factor;
            vec![step]
        } else {
            Vec::new()
        };

        return Some(DeriveStageOutput {
            expr: target_expr,
            steps,
        });
    }

    None
}

fn try_fast_product_to_sum_then_triple_angle_derive(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<crate::Step>, DeriveStrategy)> {
    let mut first_stage_candidates = generate_trig_bridge_rewrites(&mut simplifier.context, expr);
    first_stage_candidates.extend(
        generate_trig_additive_term_bridge_rewrites(&mut simplifier.context, expr)
            .into_iter()
            .filter(|rewrite| rewrite.kind.rule_name() == "Product-to-Sum Identity"),
    );

    for first_stage in first_stage_candidates {
        if first_stage.kind.rule_name() != "Product-to-Sum Identity" {
            continue;
        }

        for second_stage in generate_trig_additive_term_bridge_rewrites(
            &mut simplifier.context,
            first_stage.rewritten,
        ) {
            if second_stage.kind.rule_name() != "Triple Angle Expansion" {
                continue;
            }

            if !presentational_target_match(
                &mut simplifier.context,
                second_stage.rewritten,
                target_expr,
            ) {
                continue;
            }

            let steps = if collect_steps {
                vec![
                    build_trig_bridge_step(
                        &simplifier.context,
                        expr,
                        first_stage.rewritten,
                        first_stage.kind.description(),
                        first_stage.kind.rule_name(),
                    ),
                    build_trig_expand_step(
                        &simplifier.context,
                        first_stage.rewritten,
                        target_expr,
                        second_stage.kind.description(),
                        second_stage.kind.rule_name(),
                    ),
                ]
            } else {
                Vec::new()
            };

            return Some((target_expr, steps, DeriveStrategy::TrigExpand));
        }
    }

    None
}

fn run_trig_expand_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    if let Some(rewrite) = try_rewrite_trig_expansion(&mut simplifier.context, expr, target_expr) {
        let steps = if collect_steps {
            vec![build_trig_expand_step(
                &simplifier.context,
                expr,
                rewrite.rewritten,
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
            )]
        } else {
            Vec::new()
        };

        return DeriveStageOutput {
            expr: rewrite.rewritten,
            steps,
        };
    }

    for bridge in generate_trig_additive_term_bridge_rewrites(&mut simplifier.context, expr) {
        if !strong_target_match(&mut simplifier.context, bridge.rewritten, target_expr) {
            continue;
        }

        let steps = if collect_steps {
            vec![build_trig_bridge_step(
                &simplifier.context,
                expr,
                target_expr,
                bridge.kind.description(),
                bridge.kind.rule_name(),
            )]
        } else {
            Vec::new()
        };

        return DeriveStageOutput {
            expr: target_expr,
            steps,
        };
    }

    if let Some(stage) = try_run_two_step_additive_phase_shift_chain(
        simplifier,
        expr,
        target_expr,
        collect_steps,
        strong_target_match,
    ) {
        return stage;
    }

    for bridge in generate_trig_bridge_rewrites(&mut simplifier.context, expr) {
        let Some(rewrite) =
            try_rewrite_trig_expansion(&mut simplifier.context, bridge.rewritten, target_expr)
        else {
            continue;
        };

        if !strong_target_match(&mut simplifier.context, rewrite.rewritten, target_expr) {
            continue;
        }

        let steps = if collect_steps {
            vec![
                build_trig_bridge_step(
                    &simplifier.context,
                    expr,
                    bridge.rewritten,
                    bridge.kind.description(),
                    bridge.kind.rule_name(),
                ),
                build_trig_expand_step(
                    &simplifier.context,
                    bridge.rewritten,
                    rewrite.rewritten,
                    rewrite.kind.description(),
                    rewrite.kind.rule_name(),
                ),
            ]
        } else {
            Vec::new()
        };

        return DeriveStageOutput {
            expr: rewrite.rewritten,
            steps,
        };
    }

    for bridge in generate_trig_additive_term_bridge_rewrites(&mut simplifier.context, expr) {
        let Some(rewrite) =
            try_rewrite_trig_expansion(&mut simplifier.context, bridge.rewritten, target_expr)
        else {
            continue;
        };

        if !strong_target_match(&mut simplifier.context, rewrite.rewritten, target_expr) {
            continue;
        }

        let steps = if collect_steps {
            vec![
                build_trig_bridge_step(
                    &simplifier.context,
                    expr,
                    bridge.rewritten,
                    bridge.kind.description(),
                    bridge.kind.rule_name(),
                ),
                build_trig_expand_step(
                    &simplifier.context,
                    bridge.rewritten,
                    rewrite.rewritten,
                    rewrite.kind.description(),
                    rewrite.kind.rule_name(),
                ),
            ]
        } else {
            Vec::new()
        };

        return DeriveStageOutput {
            expr: rewrite.rewritten,
            steps,
        };
    }

    DeriveStageOutput {
        expr,
        steps: Vec::new(),
    }
}

fn run_trig_contract_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    if let Some(rewrite) =
        try_rewrite_trig_contraction_target_aware(&mut simplifier.context, expr, target_expr)
    {
        let rewritten_expr = if rewrite.kind.rule_name() == "Phase Shift Identity"
            && phase_shift_bridge_target_match(
                &mut simplifier.context,
                rewrite.rewritten,
                target_expr,
            ) {
            target_expr
        } else {
            rewrite.rewritten
        };
        let steps = if collect_steps {
            let mut step = crate::Step::with_snapshots(
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
                expr,
                rewritten_expr,
                Vec::new(),
                Some(&simplifier.context),
                expr,
                rewritten_expr,
            );
            step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
            step.category = cas_solver_core::step_types::StepCategory::Simplify;
            vec![step]
        } else {
            Vec::new()
        };

        return DeriveStageOutput {
            expr: rewritten_expr,
            steps,
        };
    };

    for bridge in generate_trig_bridge_rewrites(&mut simplifier.context, expr) {
        if bridge.kind.rule_name() != "Phase Shift Identity"
            || !phase_shift_bridge_target_match(
                &mut simplifier.context,
                bridge.rewritten,
                target_expr,
            )
        {
            continue;
        }

        let steps = if collect_steps {
            vec![build_trig_bridge_step(
                &simplifier.context,
                expr,
                target_expr,
                bridge.kind.description(),
                bridge.kind.rule_name(),
            )]
        } else {
            Vec::new()
        };

        return DeriveStageOutput {
            expr: target_expr,
            steps,
        };
    }

    for bridge in generate_trig_additive_term_bridge_rewrites(&mut simplifier.context, expr) {
        if bridge.kind.rule_name() != "Phase Shift Identity"
            || !phase_shift_bridge_target_match(
                &mut simplifier.context,
                bridge.rewritten,
                target_expr,
            )
        {
            continue;
        }

        let steps = if collect_steps {
            vec![build_trig_bridge_step(
                &simplifier.context,
                expr,
                bridge.rewritten,
                bridge.kind.description(),
                bridge.kind.rule_name(),
            )]
        } else {
            Vec::new()
        };

        return DeriveStageOutput {
            expr: bridge.rewritten,
            steps,
        };
    }

    if let Some(stage) = try_run_two_step_additive_phase_shift_chain(
        simplifier,
        expr,
        target_expr,
        collect_steps,
        phase_shift_bridge_target_match,
    ) {
        return stage;
    }

    DeriveStageOutput {
        expr,
        steps: Vec::new(),
    }
}

fn phase_shift_bridge_target_match(
    ctx: &mut cas_ast::Context,
    actual_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    phase_shift_target_match(ctx, actual_expr, target_expr)
}

fn try_run_two_step_additive_phase_shift_chain(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
    target_match: fn(&mut cas_ast::Context, ExprId, ExprId) -> bool,
) -> Option<DeriveStageOutput> {
    for first in generate_trig_additive_term_bridge_rewrites(&mut simplifier.context, expr) {
        if first.kind.rule_name() != "Phase Shift Identity" {
            continue;
        }

        for second in
            generate_trig_additive_term_bridge_rewrites(&mut simplifier.context, first.rewritten)
        {
            if second.kind.rule_name() != "Phase Shift Identity"
                || !target_match(&mut simplifier.context, second.rewritten, target_expr)
            {
                continue;
            }

            let steps = if collect_steps {
                vec![
                    build_trig_bridge_step(
                        &simplifier.context,
                        expr,
                        first.rewritten,
                        first.kind.description(),
                        first.kind.rule_name(),
                    ),
                    build_trig_bridge_step(
                        &simplifier.context,
                        first.rewritten,
                        target_expr,
                        second.kind.description(),
                        second.kind.rule_name(),
                    ),
                ]
            } else {
                Vec::new()
            };

            return Some(DeriveStageOutput {
                expr: target_expr,
                steps,
            });
        }
    }

    None
}

fn run_integrate_prep_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    let Some(rewrite) =
        try_rewrite_integrate_prep_target_aware(&mut simplifier.context, expr, target_expr)
    else {
        return DeriveStageOutput {
            expr,
            steps: Vec::new(),
        };
    };

    let steps = if collect_steps {
        let substeps = complete_square_didactic_substeps(&mut simplifier.context, expr);
        let mut step = crate::Step::with_snapshots(
            rewrite.kind.description(),
            rewrite.kind.rule_name(),
            expr,
            rewrite.rewritten,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            rewrite.rewritten,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Simplify;
        step.meta_mut().assumption_events.push(
            cas_solver_core::assumption_model::AssumptionEvent::nonzero(
                &simplifier.context,
                rewrite.assume_nonzero_expr,
            ),
        );
        if !substeps.is_empty() {
            step.meta_mut().substeps = substeps;
        }
        vec![step]
    } else {
        Vec::new()
    };

    DeriveStageOutput {
        expr: rewrite.rewritten,
        steps,
    }
}

fn run_solve_prep_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    shared_vars: &[String],
    collect_steps: bool,
) -> DeriveStageOutput {
    let Some(rewrite) = try_rewrite_solve_prep_target_aware(
        &mut simplifier.context,
        expr,
        target_expr,
        shared_vars,
    ) else {
        return DeriveStageOutput {
            expr,
            steps: Vec::new(),
        };
    };

    let steps = if collect_steps {
        let mut step = crate::Step::with_snapshots(
            rewrite.kind.description(),
            rewrite.kind.rule_name(),
            expr,
            rewrite.rewritten,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            rewrite.rewritten,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Simplify;
        step.meta_mut().assumption_events.push(
            cas_solver_core::assumption_model::AssumptionEvent::nonzero(
                &simplifier.context,
                rewrite.assume_nonzero_expr,
            ),
        );
        vec![step]
    } else {
        Vec::new()
    };

    DeriveStageOutput {
        expr: rewrite.rewritten,
        steps,
    }
}

fn run_log_expand_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
    mut simplify_options: crate::SimplifyOptions,
) -> DeriveStageOutput {
    if let Some(rewrite) =
        try_rewrite_log_change_of_base_target_aware(&mut simplifier.context, expr, target_expr)
    {
        if matches!(
            rewrite.kind,
            DeriveLogChangeOfBaseRewriteKind::BaseLogToQuotient
                | DeriveLogChangeOfBaseRewriteKind::BaseLogToChain
        ) {
            let steps = if collect_steps {
                let substeps = log_change_of_base_didactic_substeps(
                    &mut simplifier.context,
                    expr,
                    rewrite.rewritten,
                    rewrite.kind,
                );
                let mut step = crate::Step::with_snapshots(
                    rewrite.kind.description(),
                    rewrite.kind.rule_name(),
                    expr,
                    rewrite.rewritten,
                    Vec::new(),
                    Some(&simplifier.context),
                    expr,
                    rewrite.rewritten,
                );
                step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
                step.category = cas_solver_core::step_types::StepCategory::Expand;
                if !substeps.is_empty() {
                    step.meta_mut().substeps = substeps;
                }
                vec![step]
            } else {
                Vec::new()
            };

            return DeriveStageOutput {
                expr: rewrite.rewritten,
                steps,
            };
        }
    }

    if let Some(rewrite) =
        try_rewrite_log_simplify_target_aware(&mut simplifier.context, expr, target_expr)
    {
        let steps = if collect_steps {
            let mut step = crate::Step::with_snapshots(
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
                expr,
                rewrite.rewritten,
                Vec::new(),
                Some(&simplifier.context),
                expr,
                rewrite.rewritten,
            );
            step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
            step.category = cas_solver_core::step_types::StepCategory::Expand;
            vec![step]
        } else {
            Vec::new()
        };

        return DeriveStageOutput {
            expr: rewrite.rewritten,
            steps,
        };
    }

    if let Some(rewritten) =
        try_rewrite_log_expansion_target_aware(&mut simplifier.context, expr, target_expr)
    {
        let steps = if collect_steps {
            let mut step = crate::Step::with_snapshots(
                "Expand a logarithm into a sum of logarithms",
                "expand_log",
                expr,
                rewritten,
                Vec::new(),
                Some(&simplifier.context),
                expr,
                rewritten,
            );
            step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
            step.category = cas_solver_core::step_types::StepCategory::Expand;
            vec![step]
        } else {
            Vec::new()
        };

        return DeriveStageOutput {
            expr: rewritten,
            steps,
        };
    }

    let plan = cas_math::logarithm_inverse_support::expand_logs_collect_positive_assumptions(
        &mut simplifier.context,
        expr,
    );

    if plan.rewritten == expr {
        return DeriveStageOutput {
            expr,
            steps: Vec::new(),
        };
    }

    let assumed_positive = plan
        .assumed_positive
        .into_iter()
        .map(|subject| {
            cas_solver_core::assumption_model::AssumptionEvent::positive(
                &simplifier.context,
                subject,
            )
        })
        .collect::<Vec<_>>();

    simplify_options.collect_steps = collect_steps;
    simplify_options.goal = NormalFormGoal::ExpandedLog;
    let mut cleanup = run_stage(simplifier, plan.rewritten, collect_steps, simplify_options);

    let mut steps = Vec::new();
    if collect_steps {
        let mut step = crate::Step::with_snapshots(
            "Log expansion",
            "expand_log",
            expr,
            cleanup.expr,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            cleanup.expr,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Expand;
        step.meta_mut().assumption_events.extend(assumed_positive);
        if cleanup.expr != plan.rewritten {
            step.meta_mut().after_local = Some(plan.rewritten);
        }
        steps.push(step);
    }
    steps.append(&mut cleanup.steps);

    DeriveStageOutput {
        expr: cleanup.expr,
        steps,
    }
}

fn run_log_expand_prep_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> Option<DeriveStageOutput> {
    let (rewritten, focus_before, focus_after) =
        try_rewrite_log_argument_factorization_target_aware(
            &mut simplifier.context,
            expr,
            target_expr,
        )?;

    let steps = if collect_steps {
        let mut step = crate::Step::with_snapshots(
            "Factor the logarithm argument",
            "Factorization",
            expr,
            rewritten,
            Vec::new(),
            Some(&simplifier.context),
            focus_before,
            focus_after,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Factor;
        vec![step]
    } else {
        Vec::new()
    };

    Some(DeriveStageOutput {
        expr: rewritten,
        steps,
    })
}

fn run_exponential_sum_diff_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    let Some(rewrite) =
        try_rewrite_exponential_sum_diff_target_aware(&mut simplifier.context, expr, target_expr)
    else {
        return DeriveStageOutput {
            expr,
            steps: Vec::new(),
        };
    };

    let steps = if collect_steps {
        let mut step = crate::Step::with_snapshots(
            rewrite.kind.description(),
            rewrite.kind.rule_name(),
            expr,
            rewrite.rewritten,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            rewrite.rewritten,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Simplify;
        if let Some(assumed_positive) = rewrite.assumed_positive {
            step.meta_mut().assumption_events.push(
                cas_solver_core::assumption_model::AssumptionEvent::positive(
                    &simplifier.context,
                    assumed_positive,
                ),
            );
        }
        vec![step]
    } else {
        Vec::new()
    };

    DeriveStageOutput {
        expr: rewrite.rewritten,
        steps,
    }
}

fn run_hyperbolic_rewrite_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    let Some(rewrite) =
        try_rewrite_hyperbolic_simplify_target_aware(&mut simplifier.context, expr, target_expr)
    else {
        return DeriveStageOutput {
            expr,
            steps: Vec::new(),
        };
    };

    let steps = if collect_steps {
        let mut step = crate::Step::with_snapshots(
            rewrite.kind.description(),
            rewrite.kind.rule_name(),
            expr,
            rewrite.rewritten,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            rewrite.rewritten,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Simplify;
        vec![step]
    } else {
        Vec::new()
    };

    DeriveStageOutput {
        expr: rewrite.rewritten,
        steps,
    }
}

fn run_factorial_rewrite_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    let Some(rewrite) = try_rewrite_consecutive_factorial_ratio_target_aware(
        &mut simplifier.context,
        expr,
        target_expr,
    ) else {
        return DeriveStageOutput {
            expr,
            steps: Vec::new(),
        };
    };

    let steps = if collect_steps {
        let mut step = crate::Step::with_snapshots(
            "Cancel consecutive factorials",
            "Consecutive Factorial Ratio",
            expr,
            rewrite.rewritten,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            rewrite.rewritten,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Simplify;
        vec![step]
    } else {
        Vec::new()
    };

    DeriveStageOutput {
        expr: rewrite.rewritten,
        steps,
    }
}

fn run_combine_like_terms_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    let Some((rewritten, steps)) =
        run_combine_like_terms_rewrite(&mut simplifier.context, expr, target_expr, collect_steps)
    else {
        return DeriveStageOutput {
            expr,
            steps: Vec::new(),
        };
    };

    if !strong_target_match(&mut simplifier.context, rewritten, target_expr) {
        return DeriveStageOutput {
            expr,
            steps: Vec::new(),
        };
    }

    DeriveStageOutput {
        expr: rewritten,
        steps,
    }
}

fn run_inverse_trig_rewrite_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    if let Some(output) =
        run_inverse_trig_composition_rewrite_stage(simplifier, expr, target_expr, collect_steps)
    {
        return output;
    }

    if let Some(plan) = try_plan_inverse_trig_sum_add_expr(&mut simplifier.context, expr) {
        if let Some(output) = inverse_trig_pair_plan_stage(
            simplifier,
            expr,
            target_expr,
            collect_steps,
            plan,
            "Inverse Trig Sum Identity",
        ) {
            return output;
        }
    }

    if let Some(plan) =
        try_plan_inverse_atan_reciprocal_add_expr(&mut simplifier.context, expr, false)
    {
        if let Some(output) = inverse_trig_pair_plan_stage(
            simplifier,
            expr,
            target_expr,
            collect_steps,
            plan,
            "Inverse Tan Relations",
        ) {
            return output;
        }
    }

    DeriveStageOutput {
        expr,
        steps: Vec::new(),
    }
}

fn inverse_trig_pair_plan_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
    plan: PairWithNegationPlan,
    rule_name: &'static str,
) -> Option<DeriveStageOutput> {
    let rewritten = if strong_target_match(&mut simplifier.context, plan.final_result, target_expr)
    {
        plan.final_result
    } else {
        let (simplified, _, _) = simplifier.simplify_with_stats(
            plan.final_result,
            crate::SimplifyOptions {
                suppress_depth_overflow_warnings: true,
                ..crate::SimplifyOptions::default()
            },
        );
        if !strong_target_match(&mut simplifier.context, simplified, target_expr) {
            return None;
        }
        simplified
    };

    let steps = if collect_steps {
        let mut step = crate::Step::with_snapshots(
            &plan.desc,
            rule_name,
            expr,
            rewritten,
            Vec::new(),
            Some(&simplifier.context),
            plan.local_before,
            plan.local_after,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Simplify;
        vec![step]
    } else {
        Vec::new()
    };

    Some(DeriveStageOutput {
        expr: rewritten,
        steps,
    })
}

fn inverse_trig_composition_derive_desc(kind: InverseTrigCompositionKind) -> &'static str {
    match kind {
        InverseTrigCompositionKind::SinArcsin => "sin(arcsin(x)) = x",
        InverseTrigCompositionKind::CosArccos => "cos(arccos(x)) = x",
        InverseTrigCompositionKind::CosArcsin => "cos(arcsin(x)) = sqrt(1-x^2)",
        InverseTrigCompositionKind::SinArccos => "sin(arccos(x)) = sqrt(1-x^2)",
        InverseTrigCompositionKind::TanArcsin => "tan(arcsin(x)) = x/sqrt(1-x^2)",
        InverseTrigCompositionKind::SinArctan => "sin(arctan(x)) = x/sqrt(1+x^2)",
        InverseTrigCompositionKind::CosArctan => "cos(arctan(x)) = 1/sqrt(1+x^2)",
        InverseTrigCompositionKind::TanArctan => "tan(arctan(x)) = x",
        InverseTrigCompositionKind::ArctanTanArctan => "arctan(tan(arctan(u))) = arctan(u)",
        InverseTrigCompositionKind::ArcsinSinArctan => "arcsin(x/sqrt(1+x^2)) = arctan(x)",
    }
}

fn render_derive_substep_expr(ctx: &cas_ast::Context, expr: ExprId) -> String {
    cas_formatter::clean_display_string(&cas_formatter::render_expr(ctx, expr))
}

struct MonicCompleteSquareSubstepPlan {
    balanced_expr: ExprId,
    grouped_expr: ExprId,
}

fn complete_square_didactic_substeps(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Vec<cas_solver_core::step_types::SubStep> {
    let Some(plan) = monic_complete_square_substep_plan(ctx, expr) else {
        return Vec::new();
    };

    let before_text = render_derive_substep_expr(ctx, expr);
    let balanced_text = render_derive_substep_expr(ctx, plan.balanced_expr);
    let grouped_text = render_derive_substep_expr(ctx, plan.grouped_expr);

    vec![
        cas_solver_core::step_types::SubStep::new(
            "Añadir y restar el cuadrado del semicoeficiente",
            vec![format!("{before_text} -> {balanced_text}")],
        ),
        cas_solver_core::step_types::SubStep::new(
            "Agrupar el trinomio como cuadrado perfecto",
            vec![format!("{balanced_text} -> {grouped_text}")],
        ),
    ]
}

fn monic_complete_square_substep_plan(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<MonicCompleteSquareSubstepPlan> {
    let mut vars: Vec<_> = cas_ast::collect_variables(ctx, expr).into_iter().collect();
    vars.sort();

    for var_name in vars {
        if !complete_square_source_has_explicit_square_in_var(ctx, expr, &var_name) {
            continue;
        }

        let Some((leading_coeff, linear_coeff, constant_term)) =
            extract_simplified_nonzero_quadratic_coefficients_with_state(
                ctx,
                expr,
                &var_name,
                extract_quadratic_coefficients,
                simplify_expr_for_derive_substep,
                expr_is_zero_for_derive_substep,
            )
        else {
            continue;
        };

        if !is_one_number(ctx, leading_coeff) || expr_is_zero_for_derive_substep(ctx, linear_coeff)
        {
            continue;
        }

        return Some(build_monic_complete_square_substep_plan(
            ctx,
            &var_name,
            linear_coeff,
            constant_term,
        ));
    }

    None
}

fn complete_square_source_has_explicit_square_in_var(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            is_two_number(ctx, *exp) && contains_named_var(ctx, *base, var_name)
        }
        Expr::Add(_, _) | Expr::Sub(_, _) => {
            AddView::from_expr(ctx, expr)
                .terms
                .into_iter()
                .any(|(term, _)| {
                    complete_square_source_has_explicit_square_in_var(ctx, term, var_name)
                })
        }
        Expr::Mul(_, _) if ctx.is_mul_commutative(expr) => flatten_mul_chain(ctx, expr)
            .into_iter()
            .any(|factor| complete_square_source_has_explicit_square_in_var(ctx, factor, var_name)),
        Expr::Neg(inner) | Expr::Hold(inner) => {
            complete_square_source_has_explicit_square_in_var(ctx, *inner, var_name)
        }
        _ => false,
    }
}

fn is_two_number(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(value) if value.numer() == &2.into() && value.denom() == &1.into()
    )
}

fn build_monic_complete_square_substep_plan(
    ctx: &mut cas_ast::Context,
    var_name: &str,
    linear_coeff: ExprId,
    constant_term: ExprId,
) -> MonicCompleteSquareSubstepPlan {
    let two = ctx.num(2);
    let var_expr = ctx.var(var_name);
    let var_squared = ctx.add(Expr::Pow(var_expr, two));
    let linear_term = ctx.add(Expr::Mul(linear_coeff, var_expr));
    let half_linear_raw = ctx.add(Expr::Div(linear_coeff, two));
    let half_linear = simplify_expr_for_derive_substep(ctx, half_linear_raw);
    let half_square = ctx.add(Expr::Pow(half_linear, two));

    let quadratic_with_linear = ctx.add(Expr::Add(var_squared, linear_term));
    let with_half_square = ctx.add(Expr::Add(quadratic_with_linear, half_square));
    let with_constant = ctx.add(Expr::Add(with_half_square, constant_term));
    let balanced_expr = ctx.add(Expr::Sub(with_constant, half_square));

    let completed_binomial = ctx.add(Expr::Add(var_expr, half_linear));
    let completed_square = ctx.add(Expr::Pow(completed_binomial, two));
    let tail_raw = ctx.add(Expr::Sub(constant_term, half_square));
    let tail = simplify_expr_for_derive_substep(ctx, tail_raw);
    let grouped_expr = ctx.add(Expr::Add(completed_square, tail));

    MonicCompleteSquareSubstepPlan {
        balanced_expr,
        grouped_expr,
    }
}

fn simplify_expr_for_derive_substep(ctx: &mut cas_ast::Context, expr: ExprId) -> ExprId {
    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let (rewritten, _steps, _stats) = simplifier.simplify_with_stats(
        expr,
        crate::SimplifyOptions {
            suppress_depth_overflow_warnings: true,
            ..crate::SimplifyOptions::default()
        },
    );
    std::mem::swap(&mut simplifier.context, ctx);
    rewritten
}

fn expr_is_zero_for_derive_substep(ctx: &mut cas_ast::Context, expr: ExprId) -> bool {
    let simplified = simplify_expr_for_derive_substep(ctx, expr);
    matches!(ctx.get(simplified), Expr::Number(value) if value.numer() == &0.into())
}

fn log_base_argument(ctx: &cas_ast::Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let Expr::Function(name, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() == 2 && ctx.is_builtin(*name, BuiltinFn::Log) {
        Some((args[0], args[1]))
    } else {
        None
    }
}

fn natural_log_argument(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(name, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() == 1 && ctx.is_builtin(*name, BuiltinFn::Ln) {
        Some(args[0])
    } else {
        None
    }
}

fn change_of_base_quotient_arguments(
    ctx: &cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, ExprId, ExprId)> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let numerator = *numerator;
    let denominator = *denominator;
    let argument = natural_log_argument(ctx, numerator)?;
    let base = natural_log_argument(ctx, denominator)?;
    Some((argument, base, numerator, denominator))
}

fn log_change_of_base_didactic_substeps(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    rewritten: ExprId,
    kind: DeriveLogChangeOfBaseRewriteKind,
) -> Vec<cas_solver_core::step_types::SubStep> {
    match kind {
        DeriveLogChangeOfBaseRewriteKind::BaseLogToQuotient => {
            let Some((base, argument)) = log_base_argument(ctx, expr) else {
                return Vec::new();
            };
            let ln_argument = ctx.call_builtin(BuiltinFn::Ln, vec![argument]);
            let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
            let argument_text = render_derive_substep_expr(ctx, argument);
            let base_text = render_derive_substep_expr(ctx, base);
            let ln_argument_text = render_derive_substep_expr(ctx, ln_argument);
            let ln_base_text = render_derive_substep_expr(ctx, ln_base);
            let expr_text = render_derive_substep_expr(ctx, expr);
            let rewritten_text = render_derive_substep_expr(ctx, rewritten);

            vec![
                cas_solver_core::step_types::SubStep::new(
                    "Poner el argumento en el numerador",
                    vec![format!("{argument_text} -> {ln_argument_text}")],
                ),
                cas_solver_core::step_types::SubStep::new(
                    "Poner la base en el denominador",
                    vec![format!("{base_text} -> {ln_base_text}")],
                ),
                cas_solver_core::step_types::SubStep::new(
                    "Formar el cociente de cambio de base",
                    vec![format!("{expr_text} -> {rewritten_text}")],
                ),
            ]
        }
        DeriveLogChangeOfBaseRewriteKind::QuotientToBaseLog => {
            let Some((argument, base, numerator, denominator)) =
                change_of_base_quotient_arguments(ctx, expr)
            else {
                return Vec::new();
            };
            let base_log = ctx.call_builtin(BuiltinFn::Log, vec![base, argument]);
            let numerator_text = render_derive_substep_expr(ctx, numerator);
            let denominator_text = render_derive_substep_expr(ctx, denominator);
            let argument_text = render_derive_substep_expr(ctx, argument);
            let base_text = render_derive_substep_expr(ctx, base);
            let expr_text = render_derive_substep_expr(ctx, expr);
            let base_log_text = render_derive_substep_expr(ctx, base_log);

            vec![
                cas_solver_core::step_types::SubStep::new(
                    "Leer el argumento desde el numerador",
                    vec![format!("{numerator_text} -> argumento {argument_text}")],
                ),
                cas_solver_core::step_types::SubStep::new(
                    "Leer la base desde el denominador",
                    vec![format!("{denominator_text} -> base {base_text}")],
                ),
                cas_solver_core::step_types::SubStep::new(
                    "Reconstruir el logaritmo de base indicada",
                    vec![format!("{expr_text} -> {base_log_text}")],
                ),
            ]
        }
        DeriveLogChangeOfBaseRewriteKind::BaseLogToChain => Vec::new(),
    }
}

fn is_one_number(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value.numer() == value.denom())
}

fn unit_fraction_denominator(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    is_one_number(ctx, *numerator).then_some(*denominator)
}

fn split_telescoping_unit_gap_denominators(
    ctx: &mut cas_ast::Context,
    split_expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let terms = cas_math::expr_nary::AddView::from_expr(ctx, split_expr).terms;
    if terms.len() != 2 {
        return None;
    }

    let mut base = None;
    let mut shifted_base = None;
    for (term, sign) in terms {
        match sign {
            cas_math::expr_nary::Sign::Pos => {
                base = Some(unit_fraction_denominator(ctx, term)?);
            }
            cas_math::expr_nary::Sign::Neg => {
                shifted_base = Some(unit_fraction_denominator(ctx, term)?);
            }
        }
    }

    let base = base?;
    let shifted_base = shifted_base?;
    let one = ctx.num(1);
    let expected_shifted_base = ctx.add(Expr::Add(base, one));
    (cas_ast::ordering::compare_expr(ctx, expected_shifted_base, shifted_base)
        == std::cmp::Ordering::Equal
        || strong_target_match(ctx, expected_shifted_base, shifted_base))
    .then_some((base, shifted_base))
}

fn denominator_matches_telescoping_pair(
    ctx: &cas_ast::Context,
    denominator: ExprId,
    base: ExprId,
    shifted_base: ExprId,
) -> bool {
    let factors = cas_math::expr_nary::mul_leaves(ctx, denominator);
    if factors.len() != 2 {
        return false;
    }

    let same_order = cas_ast::ordering::compare_expr(ctx, factors[0], base)
        == std::cmp::Ordering::Equal
        && cas_ast::ordering::compare_expr(ctx, factors[1], shifted_base)
            == std::cmp::Ordering::Equal;
    let swapped_order = cas_ast::ordering::compare_expr(ctx, factors[1], base)
        == std::cmp::Ordering::Equal
        && cas_ast::ordering::compare_expr(ctx, factors[0], shifted_base)
            == std::cmp::Ordering::Equal;
    same_order || swapped_order
}

fn consecutive_telescoping_common_fraction(
    ctx: &mut cas_ast::Context,
    compact_expr: ExprId,
    split_expr: ExprId,
) -> Option<ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(compact_expr) else {
        return None;
    };
    let numerator = *numerator;
    let denominator = *denominator;
    if !is_one_number(ctx, numerator) {
        return None;
    }

    let (base, shifted_base) = split_telescoping_unit_gap_denominators(ctx, split_expr)?;
    if !denominator_matches_telescoping_pair(ctx, denominator, base, shifted_base) {
        return None;
    }

    let numerator_difference = ctx.add(Expr::Sub(shifted_base, base));
    Some(ctx.add(Expr::Div(numerator_difference, denominator)))
}

fn telescoping_fraction_didactic_substeps(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    rewritten: ExprId,
    rule_name: &str,
) -> Vec<cas_solver_core::step_types::SubStep> {
    let Some((compact_expr, split_expr)) = (match rule_name {
        "Telescoping Fraction Split" => Some((expr, rewritten)),
        "Telescoping Fraction Combine" => Some((rewritten, expr)),
        _ => None,
    }) else {
        return Vec::new();
    };

    let Some(common_fraction) =
        consecutive_telescoping_common_fraction(ctx, compact_expr, split_expr)
    else {
        return Vec::new();
    };

    let compact_text = render_derive_substep_expr(ctx, compact_expr);
    let split_text = render_derive_substep_expr(ctx, split_expr);
    let common_text = render_derive_substep_expr(ctx, common_fraction);

    match rule_name {
        "Telescoping Fraction Split" => vec![
            cas_solver_core::step_types::SubStep::new(
                "Introducir el numerador telescópico",
                vec![format!("{compact_text} -> {common_text}")],
            ),
            cas_solver_core::step_types::SubStep::new(
                "Separar sobre el denominador común",
                vec![format!("{common_text} -> {split_text}")],
            ),
        ],
        "Telescoping Fraction Combine" => vec![
            cas_solver_core::step_types::SubStep::new(
                "Llevar las fracciones al denominador común",
                vec![format!("{split_text} -> {common_text}")],
            ),
            cas_solver_core::step_types::SubStep::new(
                "Simplificar el numerador telescópico",
                vec![format!("{common_text} -> {compact_text}")],
            ),
        ],
        _ => Vec::new(),
    }
}

fn unary_inverse_trig_composition_arg(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(name, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() == 1
        && (ctx.is_builtin(*name, BuiltinFn::Arcsin) || ctx.is_builtin(*name, BuiltinFn::Asin))
    {
        Some(args[0])
    } else {
        None
    }
}

fn inverse_trig_composition_didactic_substeps(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    rewritten: ExprId,
    kind: InverseTrigCompositionKind,
) -> Vec<cas_solver_core::step_types::SubStep> {
    if kind != InverseTrigCompositionKind::ArcsinSinArctan {
        return Vec::new();
    }

    let Some(arcsin_arg) = unary_inverse_trig_composition_arg(ctx, expr) else {
        return Vec::new();
    };

    let sin_rewritten = ctx.call_builtin(BuiltinFn::Sin, vec![rewritten]);
    let arcsin_sin_rewritten = ctx.call_builtin(BuiltinFn::Arcsin, vec![sin_rewritten]);
    let arcsin_arg_text = render_derive_substep_expr(ctx, arcsin_arg);
    let sin_rewritten_text = render_derive_substep_expr(ctx, sin_rewritten);
    let expr_text = render_derive_substep_expr(ctx, expr);
    let arcsin_sin_text = render_derive_substep_expr(ctx, arcsin_sin_rewritten);
    let rewritten_text = render_derive_substep_expr(ctx, rewritten);

    vec![
        cas_solver_core::step_types::SubStep::new(
            "Reconocer el argumento como seno de una arctangente",
            vec![format!("{arcsin_arg_text} = {sin_rewritten_text}")],
        ),
        cas_solver_core::step_types::SubStep::new(
            "Sustituir ese seno dentro de arcsin",
            vec![format!("{expr_text} -> {arcsin_sin_text}")],
        ),
        cas_solver_core::step_types::SubStep::new(
            "Cancelar arcsin(sin(u)) en el rango principal",
            vec![format!("{arcsin_sin_text} -> {rewritten_text}")],
        ),
    ]
}

fn run_inverse_trig_composition_rewrite_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> Option<DeriveStageOutput> {
    let plan = try_plan_inverse_trig_composition_expr(&mut simplifier.context, expr, false, true)
        .or_else(|| {
        try_plan_inverse_trig_composition_expr(&mut simplifier.context, expr, false, false)
    })?;

    let rewritten = if strong_target_match(&mut simplifier.context, plan.rewritten, target_expr) {
        plan.rewritten
    } else {
        let (simplified, _, _) = simplifier.simplify_with_stats(
            plan.rewritten,
            crate::SimplifyOptions {
                suppress_depth_overflow_warnings: true,
                ..crate::SimplifyOptions::default()
            },
        );
        if !strong_target_match(&mut simplifier.context, simplified, target_expr) {
            return None;
        }
        simplified
    };

    let steps = if collect_steps {
        let substeps = inverse_trig_composition_didactic_substeps(
            &mut simplifier.context,
            expr,
            plan.rewritten,
            plan.kind,
        );
        let mut step = crate::Step::with_snapshots(
            inverse_trig_composition_derive_desc(plan.kind),
            "Inverse Trig Composition",
            expr,
            rewritten,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            plan.rewritten,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Simplify;
        if !substeps.is_empty() {
            step.meta_mut().substeps = substeps;
        }
        vec![step]
    } else {
        Vec::new()
    };

    Some(DeriveStageOutput {
        expr: rewritten,
        steps,
    })
}

fn run_fraction_cancel_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    let Some(rewrite) =
        try_rewrite_exact_fraction_cancel_target_aware(&mut simplifier.context, expr, target_expr)
    else {
        return DeriveStageOutput {
            expr,
            steps: Vec::new(),
        };
    };

    let steps = if collect_steps {
        let (before_local, after_local) = if let (Some(before_local), Some(after_local)) =
            (rewrite.focus_before, rewrite.focus_after)
        {
            (before_local, after_local)
        } else {
            rewrite
                .kind
                .local_snapshots(expr, rewrite.intermediate, rewrite.rewritten)
        };

        let mut step = crate::Step::with_snapshots(
            rewrite.kind.description(),
            rewrite.kind.rule_name(),
            expr,
            rewrite.rewritten,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            rewrite.rewritten,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Simplify;
        step.meta_mut().before_local = Some(before_local);
        step.meta_mut().after_local = Some(after_local);
        step.meta_mut().required_conditions = rewrite.required_conditions.clone();
        vec![step]
    } else {
        Vec::new()
    };

    DeriveStageOutput {
        expr: rewrite.rewritten,
        steps,
    }
}

fn run_nested_fraction_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    let Some(rewritten) =
        try_rewrite_nested_fraction_target_aware(&mut simplifier.context, expr, target_expr)
    else {
        return DeriveStageOutput {
            expr,
            steps: Vec::new(),
        };
    };

    let steps = if collect_steps {
        let mut step = crate::Step::with_snapshots(
            "Simplify nested fraction",
            "Simplify Nested Fraction",
            expr,
            rewritten,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            rewritten,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Simplify;
        vec![step]
    } else {
        Vec::new()
    };

    DeriveStageOutput {
        expr: rewritten,
        steps,
    }
}

fn run_radical_rewrite_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    let Some(rewrite) =
        try_rewrite_radical_target_aware(&mut simplifier.context, expr, target_expr)
    else {
        return DeriveStageOutput {
            expr,
            steps: Vec::new(),
        };
    };

    let steps = if collect_steps {
        let mut step = crate::Step::with_snapshots(
            rewrite.kind.description(),
            rewrite.kind.rule_name(),
            expr,
            rewrite.rewritten,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            rewrite.rewritten,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Simplify;
        step.meta_mut().required_conditions = rewrite.required_conditions.clone();
        vec![step]
    } else {
        Vec::new()
    };

    DeriveStageOutput {
        expr: rewrite.rewritten,
        steps,
    }
}

fn run_trig_rewrite_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    if let Some(rewrite) =
        try_rewrite_trig_identity_to_one_target_aware(&mut simplifier.context, expr, target_expr)
    {
        let steps = if collect_steps {
            let mut step = crate::Step::with_snapshots(
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
                expr,
                rewrite.rewritten,
                Vec::new(),
                Some(&simplifier.context),
                expr,
                rewrite.rewritten,
            );
            step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
            step.category = cas_solver_core::step_types::StepCategory::Simplify;
            vec![step]
        } else {
            Vec::new()
        };

        return DeriveStageOutput {
            expr: rewrite.rewritten,
            steps,
        };
    }

    if let Some(rewrite) = try_rewrite_shifted_reciprocal_pythagorean_target_aware(
        &mut simplifier.context,
        expr,
        target_expr,
    ) {
        let steps = if collect_steps {
            let mut step = crate::Step::with_snapshots(
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
                expr,
                rewrite.rewritten,
                Vec::new(),
                Some(&simplifier.context),
                expr,
                rewrite.rewritten,
            );
            step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
            step.category = cas_solver_core::step_types::StepCategory::Simplify;
            vec![step]
        } else {
            Vec::new()
        };

        return DeriveStageOutput {
            expr: rewrite.rewritten,
            steps,
        };
    }

    if let Some(rewrite) =
        try_rewrite_pythagorean_factor_form_target_aware(&mut simplifier.context, expr, target_expr)
    {
        let steps = if collect_steps {
            let mut step = crate::Step::with_snapshots(
                &rewrite.description,
                "Pythagorean Factor Form",
                expr,
                rewrite.rewritten,
                Vec::new(),
                Some(&simplifier.context),
                expr,
                rewrite.rewritten,
            );
            step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
            step.category = cas_solver_core::step_types::StepCategory::Simplify;
            vec![step]
        } else {
            Vec::new()
        };

        return DeriveStageOutput {
            expr: rewrite.rewritten,
            steps,
        };
    }

    DeriveStageOutput {
        expr,
        steps: Vec::new(),
    }
}

fn run_log_contract_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    if let Some(rewrite) =
        try_rewrite_log_change_of_base_target_aware(&mut simplifier.context, expr, target_expr)
    {
        if matches!(
            rewrite.kind,
            DeriveLogChangeOfBaseRewriteKind::QuotientToBaseLog
        ) {
            let steps = if collect_steps {
                let substeps = log_change_of_base_didactic_substeps(
                    &mut simplifier.context,
                    expr,
                    rewrite.rewritten,
                    rewrite.kind,
                );
                let mut step = crate::Step::with_snapshots(
                    rewrite.kind.description(),
                    rewrite.kind.rule_name(),
                    expr,
                    rewrite.rewritten,
                    Vec::new(),
                    Some(&simplifier.context),
                    expr,
                    rewrite.rewritten,
                );
                step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
                step.category = cas_solver_core::step_types::StepCategory::Simplify;
                if !substeps.is_empty() {
                    step.meta_mut().substeps = substeps;
                }
                vec![step]
            } else {
                Vec::new()
            };

            return DeriveStageOutput {
                expr: rewrite.rewritten,
                steps,
            };
        }
    }

    let Some(rewritten) =
        try_rewrite_log_contraction_to_target_aware(&mut simplifier.context, expr, target_expr)
            .or_else(|| try_rewrite_log_contraction_target_aware(&mut simplifier.context, expr))
    else {
        return DeriveStageOutput {
            expr,
            steps: Vec::new(),
        };
    };

    let steps = if collect_steps {
        let mut step = crate::Step::with_snapshots(
            "Combine logarithms into a single logarithm",
            "Log Contraction",
            expr,
            rewritten,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            rewritten,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Simplify;
        vec![step]
    } else {
        Vec::new()
    };

    DeriveStageOutput {
        expr: rewritten,
        steps,
    }
}

fn run_simplify_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
    mut simplify_options: crate::SimplifyOptions,
) -> DeriveStageOutput {
    if let Some(stage) =
        try_run_finite_aggregate_target_aware_stage(simplifier, expr, target_expr, collect_steps)
    {
        return stage;
    }

    if let Some(rewrite) =
        try_rewrite_trig_identity_to_one_target_aware(&mut simplifier.context, expr, target_expr)
    {
        let steps = if collect_steps {
            let mut step = crate::Step::with_snapshots(
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
                expr,
                rewrite.rewritten,
                Vec::new(),
                Some(&simplifier.context),
                expr,
                rewrite.rewritten,
            );
            step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
            step.category = cas_solver_core::step_types::StepCategory::Simplify;
            vec![step]
        } else {
            Vec::new()
        };

        return DeriveStageOutput {
            expr: rewrite.rewritten,
            steps,
        };
    }

    if let Some(rewrite) = try_rewrite_shifted_reciprocal_pythagorean_target_aware(
        &mut simplifier.context,
        expr,
        target_expr,
    ) {
        let steps = if collect_steps {
            let mut step = crate::Step::with_snapshots(
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
                expr,
                rewrite.rewritten,
                Vec::new(),
                Some(&simplifier.context),
                expr,
                rewrite.rewritten,
            );
            step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
            step.category = cas_solver_core::step_types::StepCategory::Simplify;
            vec![step]
        } else {
            Vec::new()
        };

        return DeriveStageOutput {
            expr: rewrite.rewritten,
            steps,
        };
    }

    if let Some(rewrite) =
        try_rewrite_shifted_double_angle_target_aware(&mut simplifier.context, expr, target_expr)
    {
        let steps = if collect_steps {
            let mut step = crate::Step::with_snapshots(
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
                expr,
                rewrite.rewritten,
                Vec::new(),
                Some(&simplifier.context),
                expr,
                rewrite.rewritten,
            );
            step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
            step.category = cas_solver_core::step_types::StepCategory::Simplify;
            vec![step]
        } else {
            Vec::new()
        };

        return DeriveStageOutput {
            expr: rewrite.rewritten,
            steps,
        };
    }

    if let Some(rewrite) =
        try_rewrite_pythagorean_factor_form_target_aware(&mut simplifier.context, expr, target_expr)
    {
        let steps = if collect_steps {
            let mut step = crate::Step::with_snapshots(
                &rewrite.description,
                "Pythagorean Factor Form",
                expr,
                rewrite.rewritten,
                Vec::new(),
                Some(&simplifier.context),
                expr,
                rewrite.rewritten,
            );
            step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
            step.category = cas_solver_core::step_types::StepCategory::Simplify;
            vec![step]
        } else {
            Vec::new()
        };

        return DeriveStageOutput {
            expr: rewrite.rewritten,
            steps,
        };
    }

    if let Some(rewrite) =
        try_rewrite_log_simplify_target_aware(&mut simplifier.context, expr, target_expr)
    {
        let steps = if collect_steps {
            let mut step = crate::Step::with_snapshots(
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
                expr,
                rewrite.rewritten,
                Vec::new(),
                Some(&simplifier.context),
                expr,
                rewrite.rewritten,
            );
            step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
            step.category = cas_solver_core::step_types::StepCategory::Simplify;
            vec![step]
        } else {
            Vec::new()
        };

        return DeriveStageOutput {
            expr: rewrite.rewritten,
            steps,
        };
    }

    if let Some(rewrite) =
        try_rewrite_exponential_sum_diff_target_aware(&mut simplifier.context, expr, target_expr)
    {
        let steps = if collect_steps {
            let mut step = crate::Step::with_snapshots(
                rewrite.kind.description(),
                rewrite.kind.rule_name(),
                expr,
                rewrite.rewritten,
                Vec::new(),
                Some(&simplifier.context),
                expr,
                rewrite.rewritten,
            );
            step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
            step.category = cas_solver_core::step_types::StepCategory::Simplify;
            vec![step]
        } else {
            Vec::new()
        };

        return DeriveStageOutput {
            expr: rewrite.rewritten,
            steps,
        };
    }

    let fraction_cancel_stage =
        run_fraction_cancel_stage(simplifier, expr, target_expr, collect_steps);
    if fraction_cancel_stage.expr != expr {
        return fraction_cancel_stage;
    }

    simplify_options.collect_steps = collect_steps;
    run_stage(simplifier, expr, collect_steps, simplify_options)
}

fn try_run_finite_aggregate_target_aware_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> Option<DeriveStageOutput> {
    try_run_finite_sum_target_aware_stage(simplifier, expr, target_expr, collect_steps).or_else(
        || try_run_finite_product_target_aware_stage(simplifier, expr, target_expr, collect_steps),
    )
}

fn try_run_finite_sum_target_aware_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> Option<DeriveStageOutput> {
    let plan = try_plan_finite_sum_evaluation(&mut simplifier.context, expr, 1000)?;
    let mut temp = crate::Simplifier::with_default_rules();
    temp.context = simplifier.context.clone();
    let (simplified, _steps, _stats) = temp.simplify_with_stats(
        plan.candidate,
        crate::SimplifyOptions {
            suppress_depth_overflow_warnings: true,
            ..crate::SimplifyOptions::default()
        },
    );
    simplifier.context = temp.context;

    if !derive_target_match(simplifier, simplified, target_expr) {
        return None;
    }

    let steps = if collect_steps {
        let description = render_sum_evaluation_desc(&plan.kind, &plan.call, |id| {
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id
                }
            )
        });
        let mut step = crate::Step::with_snapshots(
            &description,
            "Finite Summation",
            expr,
            simplified,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            simplified,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Simplify;
        vec![step]
    } else {
        Vec::new()
    };

    Some(DeriveStageOutput {
        expr: simplified,
        steps,
    })
}

fn try_run_finite_product_target_aware_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> Option<DeriveStageOutput> {
    let plan = try_plan_finite_product_evaluation(&mut simplifier.context, expr, 1000)?;
    let mut temp = crate::Simplifier::with_default_rules();
    temp.context = simplifier.context.clone();
    let (simplified, _steps, _stats) = temp.simplify_with_stats(
        plan.candidate,
        crate::SimplifyOptions {
            suppress_depth_overflow_warnings: true,
            ..crate::SimplifyOptions::default()
        },
    );
    simplifier.context = temp.context;

    if !derive_target_match(simplifier, simplified, target_expr) {
        return None;
    }

    let steps = if collect_steps {
        let description = render_product_evaluation_desc(&plan.kind, &plan.call, |id| {
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id
                }
            )
        });
        let mut step = crate::Step::with_snapshots(
            &description,
            "Finite Product",
            expr,
            simplified,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            simplified,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Simplify;
        vec![step]
    } else {
        Vec::new()
    };

    Some(DeriveStageOutput {
        expr: simplified,
        steps,
    })
}

fn run_finite_aggregate_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    try_run_finite_aggregate_target_aware_stage(simplifier, expr, target_expr, collect_steps)
        .unwrap_or(DeriveStageOutput {
            expr,
            steps: Vec::new(),
        })
}

fn run_collect_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    candidate_variables: &[String],
) -> Option<DeriveStageOutput> {
    for var_name in candidate_variables {
        let Some(rewrite) = cas_engine::try_collect_by_var(&mut simplifier.context, expr, var_name)
        else {
            continue;
        };

        if !strong_target_match(&mut simplifier.context, rewrite.rewritten, target_expr) {
            continue;
        }

        let description = format!("Collect terms by {var_name}");
        let mut step = crate::Step::with_snapshots(
            &description,
            "Collect Terms",
            expr,
            rewrite.rewritten,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            rewrite.rewritten,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Simplify;

        return Some(DeriveStageOutput {
            expr: rewrite.rewritten,
            steps: vec![step],
        });
    }

    if let Some(rewrite) =
        try_rewrite_collect_monomial_target_aware(&mut simplifier.context, expr, target_expr)
    {
        let description = format!("Collect terms by {}", rewrite.focus_label);
        let mut step = crate::Step::with_snapshots(
            &description,
            "Collect Terms",
            expr,
            rewrite.rewritten,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            rewrite.rewritten,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Simplify;

        return Some(DeriveStageOutput {
            expr: rewrite.rewritten,
            steps: vec![step],
        });
    }

    None
}

fn run_factored_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    _collect_steps: bool,
    _simplify_options: crate::SimplifyOptions,
) -> DeriveStageOutput {
    let target_rewrite =
        try_rewrite_factored_target_aware(&mut simplifier.context, expr, target_expr);
    let factored_expr = target_rewrite
        .map(|rewrite| rewrite.rewritten)
        .unwrap_or_else(|| cas_math::factor::factor(&mut simplifier.context, expr));
    let steps = if factored_expr == expr {
        Vec::new()
    } else {
        let mut step = crate::Step::with_snapshots(
            "Factorization",
            "Factorization",
            expr,
            factored_expr,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            factored_expr,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Factor;
        if let Some(rewrite) = target_rewrite {
            step.meta_mut().before_local = rewrite.focus_before;
            step.meta_mut().after_local = rewrite.focus_after;
        }
        vec![step]
    };

    DeriveStageOutput {
        expr: factored_expr,
        steps,
    }
}

fn run_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    collect_steps: bool,
    simplify_options: crate::SimplifyOptions,
) -> DeriveStageOutput {
    let collector = collect_steps.then(EngineEventCollector::new);
    let previous_listener = collector
        .as_ref()
        .map(|collector| simplifier.replace_step_listener(Some(Box::new(collector.clone()))));

    let (expr, mut steps, _stats) = simplifier.simplify_with_stats(expr, simplify_options);

    if let Some(previous_listener) = previous_listener {
        let _ = simplifier.replace_step_listener(previous_listener);
    }

    if steps.is_empty() {
        if let Some(collector) = collector.as_ref() {
            steps = fallback_steps_from_events(&collector.events(), &simplifier.context);
        }
    }

    DeriveStageOutput { expr, steps }
}

fn fallback_steps_from_events(events: &[EngineEvent], ctx: &cas_ast::Context) -> Vec<crate::Step> {
    crate::engine_event_display_steps::build_display_eval_steps_from_events(events, ctx)
        .into_inner()
}

fn finalize_steps(
    original_expr: ExprId,
    final_expr: ExprId,
    stages: Vec<DeriveStageOutput>,
    ctx: &mut cas_ast::Context,
) -> Vec<crate::Step> {
    let stage_count = stages.len();
    let cleaned = prune_semantically_noop_derive_steps(
        prune_adjacent_inverse_derive_steps(clean_derive_stage_steps(stages), ctx),
        ctx,
    );

    let mut steps = if stage_count > 1 || cleaned.is_empty() {
        cleaned
    } else {
        // In derive mode the source and target are expected to be semantically
        // equal, so the generic semantic no-op probe just replays an expensive
        // equivalence check over the whole pair. Optimize the emitted steps
        // structurally instead of re-proving equivalence here.
        if cleaned.iter().any(|step| step.poly_proof().is_some()) {
            cas_solver_core::step_optimization_runtime::optimize_steps_with_absorption(cleaned)
        } else {
            cas_solver_core::step_optimization_runtime::optimize_steps(cleaned)
        }
    };
    steps = prune_adjacent_inverse_derive_steps(steps, ctx);
    steps = prune_semantically_noop_derive_steps(steps, ctx);

    if stage_count > 1 {
        truncate_steps_after_first_presentational_target_match(&mut steps, final_expr, ctx);
    } else {
        truncate_steps_after_first_target_match(&mut steps, final_expr, ctx);
    }

    populate_derive_global_snapshots(&mut steps, original_expr, ctx);
    if let Some(last_step) = steps.last_mut() {
        last_step.global_after = Some(final_expr);
    }
    retarget_last_step_to_final_expr_when_equivalent(&mut steps, final_expr, ctx);

    steps
}

fn finalize_planner_steps(
    original_expr: ExprId,
    final_expr: ExprId,
    stages: Vec<DeriveStageOutput>,
    ctx: &mut cas_ast::Context,
) -> Vec<crate::Step> {
    let mut steps = prune_semantically_noop_derive_steps(
        prune_adjacent_inverse_derive_steps(clean_derive_stage_steps(stages), ctx),
        ctx,
    );
    truncate_steps_after_first_presentational_target_match(&mut steps, final_expr, ctx);

    populate_derive_global_snapshots(&mut steps, original_expr, ctx);
    if let Some(last_step) = steps.last_mut() {
        last_step.global_after = Some(final_expr);
    }
    retarget_last_step_to_final_expr_when_equivalent(&mut steps, final_expr, ctx);

    steps
}

fn populate_derive_global_snapshots(
    steps: &mut [crate::Step],
    original_expr: ExprId,
    ctx: &mut cas_ast::Context,
) {
    let mut current_root = original_expr;

    for step in steps {
        let preserve_previous_snapshot_for_hidden_like_term_setup = step.path().is_empty()
            && step.rule_name == "Combine Like Terms"
            && current_root != step.before;
        let global_before =
            if step.path().is_empty() && !preserve_previous_snapshot_for_hidden_like_term_setup {
                step.before
            } else {
                current_root
            };
        step.global_before = Some(global_before);
        let reconstructed_after = if step.path().is_empty() {
            step.after
        } else {
            reconstruct_global_expr(ctx, global_before, step.path(), step.after)
        };
        step.global_after = Some(reconstructed_after);
        current_root = reconstructed_after;
    }
}

fn retarget_last_step_to_final_expr_when_equivalent(
    steps: &mut [crate::Step],
    final_expr: ExprId,
    ctx: &cas_ast::Context,
) {
    let Some(last_step) = steps.last_mut() else {
        return;
    };

    let after_candidate = last_step.after_local().unwrap_or(last_step.after);
    let mut temp_ctx = ctx.clone();
    let semantically_equal = cas_math::semantic_equality::SemanticEqualityChecker::new(&temp_ctx)
        .are_equal(after_candidate, final_expr);
    if !(strong_target_match(&mut temp_ctx, after_candidate, final_expr)
        || presentational_target_match(&mut temp_ctx, after_candidate, final_expr)
        || semantically_equal)
    {
        return;
    }

    last_step.after = final_expr;
    last_step.meta_mut().after_local = Some(final_expr);
}

fn clean_derive_stage_steps(stages: Vec<DeriveStageOutput>) -> Vec<crate::Step> {
    let raw_steps: Vec<_> = stages.into_iter().flat_map(|stage| stage.steps).collect();
    repair_derive_step_chain(raw_steps)
}

fn repair_derive_step_chain(steps: Vec<crate::Step>) -> Vec<crate::Step> {
    cas_solver_core::eval_step_pipeline::clean_eval_steps(
        steps,
        |s: &crate::Step| s.before,
        |s: &crate::Step| s.after,
        |s: &crate::Step| s.before_local(),
        |s: &crate::Step| s.after_local(),
        |s: &crate::Step| s.global_after,
        |s: &mut crate::Step, gb| s.global_before = Some(gb),
    )
}

fn prune_adjacent_inverse_derive_steps(
    steps: Vec<crate::Step>,
    ctx: &cas_ast::Context,
) -> Vec<crate::Step> {
    let mut pruned = Vec::with_capacity(steps.len());

    for step in steps {
        let cancels_previous = pruned
            .last()
            .is_some_and(|previous| should_cancel_adjacent_inverse_pair(previous, &step, ctx));
        if cancels_previous {
            pruned.pop();
        } else {
            pruned.push(step);
        }
    }

    repair_derive_step_chain(pruned)
}

fn prune_semantically_noop_derive_steps(
    steps: Vec<crate::Step>,
    ctx: &cas_ast::Context,
) -> Vec<crate::Step> {
    let mut pruned = Vec::with_capacity(steps.len());
    for step in steps {
        if !should_drop_semantically_noop_step(&step, ctx) {
            pruned.push(step);
        }
    }
    repair_derive_step_chain(pruned)
}

fn should_drop_semantically_noop_step(step: &crate::Step, ctx: &cas_ast::Context) -> bool {
    if cas_solver_core::step_rules::is_always_keep_step_rule_name(step.rule_name.as_str()) {
        return false;
    }

    if !step.assumption_events().is_empty()
        || !step.required_conditions().is_empty()
        || step.poly_proof().is_some()
        || !step.substeps().is_empty()
    {
        return false;
    }

    if !same_display_expr(ctx, step.before, step.after) {
        return false;
    }

    if let (Some(local_before), Some(local_after)) = (step.before_local(), step.after_local()) {
        if !same_display_expr(ctx, local_before, local_after) {
            return false;
        }
    }

    true
}

fn same_display_expr(ctx: &cas_ast::Context, left: ExprId, right: ExprId) -> bool {
    cas_formatter::clean_display_string(&cas_formatter::render_expr(ctx, left))
        == cas_formatter::clean_display_string(&cas_formatter::render_expr(ctx, right))
}

fn should_cancel_adjacent_inverse_pair(
    previous: &crate::Step,
    next: &crate::Step,
    ctx: &cas_ast::Context,
) -> bool {
    let mut temp_ctx = ctx.clone();
    if !strong_target_match(&mut temp_ctx, previous.before, next.after)
        || !strong_target_match(&mut temp_ctx, previous.after, next.before)
    {
        return false;
    }

    if previous.path() != next.path() {
        return false;
    }

    let allow_medium_oscillation = is_low_signal_trig_oscillation_rule(&previous.rule_name)
        && is_low_signal_trig_oscillation_rule(&next.rule_name);
    if !allow_medium_oscillation
        && (previous.get_importance() >= cas_solver_core::step_types::ImportanceLevel::Medium
            || next.get_importance() >= cas_solver_core::step_types::ImportanceLevel::Medium)
    {
        return false;
    }

    if cas_solver_core::step_rules::is_always_keep_step_rule_name(previous.rule_name.as_str())
        || cas_solver_core::step_rules::is_always_keep_step_rule_name(next.rule_name.as_str())
    {
        return false;
    }

    if !previous.assumption_events().is_empty()
        || !next.assumption_events().is_empty()
        || !previous.required_conditions().is_empty()
        || !next.required_conditions().is_empty()
        || previous.poly_proof().is_some()
        || next.poly_proof().is_some()
        || !previous.substeps().is_empty()
        || !next.substeps().is_empty()
    {
        return false;
    }

    match (
        previous.global_before,
        previous.global_after,
        next.global_before,
        next.global_after,
    ) {
        (Some(prev_before), Some(prev_after), Some(next_before), Some(next_after)) => {
            strong_target_match(&mut temp_ctx, prev_after, next_before)
                && strong_target_match(&mut temp_ctx, prev_before, next_after)
        }
        _ => true,
    }
}

fn is_low_signal_trig_oscillation_rule(rule_name: &str) -> bool {
    matches!(
        rule_name,
        "Double Angle Identity" | "Angle Consistency (Half-Angle)"
    )
}

fn truncate_steps_after_first_target_match(
    steps: &mut Vec<crate::Step>,
    final_expr: ExprId,
    ctx: &cas_ast::Context,
) {
    let mut temp_ctx = ctx.clone();
    let cutoff = steps.iter().position(|step| {
        let local_after = step.after_local().unwrap_or(step.after);
        strong_target_match(&mut temp_ctx, local_after, final_expr)
            || strong_target_match(&mut temp_ctx, step.after, final_expr)
            || step.global_after.is_some_and(|global_after| {
                strong_target_match(&mut temp_ctx, global_after, final_expr)
            })
    });

    if let Some(index) = cutoff {
        steps.truncate(index + 1);
    }
}

fn truncate_steps_after_first_presentational_target_match(
    steps: &mut Vec<crate::Step>,
    final_expr: ExprId,
    ctx: &cas_ast::Context,
) {
    let mut temp_ctx = ctx.clone();
    let cutoff = steps.iter().position(|step| {
        let local_after = step.after_local().unwrap_or(step.after);
        presentational_target_match(&mut temp_ctx, local_after, final_expr)
            || presentational_target_match(&mut temp_ctx, step.after, final_expr)
            || step.global_after.is_some_and(|global_after| {
                presentational_target_match(&mut temp_ctx, global_after, final_expr)
            })
    });

    if let Some(index) = cutoff {
        steps.truncate(index + 1);
    }
}

fn retarget_stage_output(stage: &mut DeriveStageOutput, target_expr: ExprId) {
    let previous_expr = stage.expr;
    stage.expr = target_expr;

    if let Some(last_step) = stage.steps.last_mut() {
        if last_step.after == previous_expr {
            last_step.after = target_expr;
        }
        last_step.global_after = Some(target_expr);
        if last_step.after_local() == Some(previous_expr) {
            last_step.meta_mut().after_local = Some(target_expr);
        }
    }
}

fn format_derive_eval_lines(
    ctx: &mut cas_ast::Context,
    expr_input: &str,
    output: DeriveEvalOutput,
    mode: crate::FullSimplifyDisplayMode,
) -> Vec<String> {
    let target =
        cas_formatter::clean_display_string(&cas_formatter::render_expr(ctx, output.target_expr));
    let result_target = parenthesize_plain_numeric_exponents_for_derive_result(
        &cas_formatter::clean_display_string(&format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &*ctx,
                id: output.target_expr
            }
        )),
    );
    match output.status {
        DeriveStatus::Derived { strategy } => {
            let mut lines = crate::format_full_simplify_eval_lines(
                ctx,
                expr_input,
                output.resolved_expr,
                output.derived_expr,
                &output.steps,
                mode,
            );
            insert_derive_substep_lines(
                &mut lines,
                &output.steps,
                &[
                    "Inverse Trig Composition",
                    "Change of Base",
                    "Complete the Square",
                    "Telescoping Fraction Split",
                    "Telescoping Fraction Combine",
                ],
            );
            retarget_final_after_line(&mut lines, &target);
            if matches!(strategy, DeriveStrategy::FractionDecompose) {
                retarget_result_line(&mut lines, &result_target);
            }
            humanize_derive_cli_step_rule_suffixes(&mut lines);
            lines.insert(1, format!("Target: {target}"));
            lines.insert(2, format!("Strategy: {}", strategy.label()));
            append_derive_requires_lines(
                &mut lines,
                ctx,
                output.resolved_expr,
                output.derived_expr,
            );
            lines
        }
        DeriveStatus::AlreadyAtTarget => vec![
            format!(
                "Parsed: {}",
                cas_formatter::DisplayExpr {
                    context: &*ctx,
                    id: output.resolved_expr
                }
            ),
            format!("Target: {target}"),
            "Already at target.".to_string(),
            format!("Result: {target}"),
        ],
        DeriveStatus::EquivalentButUnsupported { equivalence } => {
            let mut lines = vec![
                format!(
                    "Parsed: {}",
                    cas_formatter::DisplayExpr {
                        context: &*ctx,
                        id: output.resolved_expr
                    }
                ),
                format!("Target: {target}"),
            ];
            append_equivalence_summary(&mut lines, &equivalence);
            lines.push(
                "Equivalent, but the second expression is not a supported simplification target yet."
                    .to_string(),
            );
            lines
        }
        DeriveStatus::NotEquivalent { equivalence } => {
            let mut lines = vec![
                format!(
                    "Parsed: {}",
                    cas_formatter::DisplayExpr {
                        context: &*ctx,
                        id: output.resolved_expr
                    }
                ),
                format!("Target: {target}"),
            ];
            append_equivalence_summary(&mut lines, &equivalence);
            let detail = match equivalence {
                crate::EquivalenceResult::False => {
                    "Derive unavailable: the two expressions are not equivalent."
                }
                crate::EquivalenceResult::Unknown => {
                    "Derive unavailable: cannot prove that the two expressions are equivalent, so it cannot provide step-by-step derivation."
                }
                crate::EquivalenceResult::True
                | crate::EquivalenceResult::ConditionalTrue { .. } => "Derive unavailable.",
            };
            lines.push(detail.to_string());
            lines
        }
    }
}

fn humanize_derive_cli_step_rule_suffixes(lines: &mut [String]) {
    for line in lines {
        let Some((prefix, description, raw_rule)) = split_derive_cli_step_rule_suffix(line) else {
            continue;
        };
        let Some(visible_rule) = visible_derive_cli_rule_suffix(raw_rule, description) else {
            continue;
        };
        *line = format!("{prefix}  [{visible_rule}]");
    }
}

fn split_derive_cli_step_rule_suffix(line: &str) -> Option<(&str, &str, &str)> {
    let (prefix, suffix) = line.rsplit_once("  [")?;
    let raw_rule = suffix.strip_suffix(']')?;
    let (step_number, description) = prefix.split_once(". ")?;
    if step_number.is_empty() || !step_number.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    Some((prefix, description, raw_rule))
}

fn visible_derive_cli_rule_suffix(rule_name: &str, description: &str) -> Option<&'static str> {
    match rule_name {
        "Angle Sum/Diff Identity" => Some("Aplicar suma/diferencia de ángulos"),
        "Binomial Coefficient Symmetry" => Some("Aplicar simetría del coeficiente binomial"),
        "Binomial Expansion" | "Small Multinomial Expansion" => Some("Expandir binomio"),
        "Cancel Sum/Difference of Cubes Fraction" => Some("Factorizar cubos y cancelar"),
        "Canonicalize Even Power Base" => Some("Invertir una resta dentro de una potencia par"),
        RULE_CANCEL_EXACT_ADDITIVE_PAIRS => Some("Cancelar términos opuestos"),
        "Cancel Reciprocal Exponents" | "Square of Square Root" => Some("Deshacer raíz y potencia"),
        "Change of Base" => Some("Aplicar cambio de base"),
        "Add Fractions" => Some("Sumar fracciones en un solo denominador"),
        "Combine Same Denominator Fractions" => {
            Some("Combinar fracciones con el mismo denominador")
        }
        "Combine Same Denominator Sub" => {
            Some("Combinar resta de fracciones con el mismo denominador")
        }
        "Combine Like Terms" => Some("Agrupar términos semejantes"),
        "Combine powers with same base (n-ary)" => Some("Sumar exponentes de la misma base"),
        "Collect Terms" if description.contains(" * ") => Some("Agrupar términos por factor común"),
        "Collect Terms" => Some("Agrupar términos por variable"),
        "Cofunction Identity" => Some("Aplicar identidad de cofunción"),
        "Consecutive Factorial Ratio" => Some("Cancelar factoriales consecutivos"),
        "Complete the Square" => Some("Completar el cuadrado"),
        "Cos Product Telescoping" => Some("Aplicar telescopado de cosenos"),
        "Difference of Squares" | "Difference of Squares (Product to Difference)" => {
            Some("Expandir la expresión")
        }
        "Dirichlet Kernel Identity" => Some("Aplicar identidad del núcleo de Dirichlet"),
        "Distribute Division" => Some("Repartir el denominador común"),
        "Distributive Property" | "Expand" => Some("Expandir la expresión"),
        "Double Angle Expansion" => Some("Expandir ángulo doble"),
        "Evaluate Hyperbolic Functions" => Some("Evaluar valor hiperbólico especial"),
        "Evaluate Logarithms" => Some("Evaluar logaritmos"),
        "Evaluate Trigonometric Functions" => Some("Evaluar valor trigonométrico especial"),
        "Exponential Sum/Difference Identity" if description.starts_with("Expand ") => {
            Some("Expandir exponencial de suma o diferencia")
        }
        "Exponential Sum/Difference Identity" if description.starts_with("Contract ") => {
            Some("Contraer productos exponenciales")
        }
        "Exponential Sum/Difference Identity" => {
            Some("Aplicar identidad exponencial de suma o diferencia")
        }
        "Exponential Power Identity" if description.starts_with("Expand ") => {
            Some("Expandir potencia exponencial")
        }
        "Exponential Power Identity" => Some("Aplicar potencia de una exponencial"),
        "Exponential Reciprocal Identity" if description.starts_with("Expand ") => {
            Some("Expandir como recíproco exponencial")
        }
        "Exponential Reciprocal Identity" => Some("Reescribir recíproco exponencial"),
        "Expand Cosecant Squared" => Some("Expandir cosecante cuadrada"),
        "Expand Odd Half Power" => Some("Reescribir potencia semientera impar"),
        "Expand Secant Squared" => Some("Expandir secante cuadrada"),
        "expand_log" => Some("Expandir logaritmos"),
        "Factor Perfect Square in Logarithm" => Some("Sacar un exponente fuera del logaritmo"),
        "Hyperbolic Exponential Identity" if description.starts_with("Expand ") => {
            Some("Expandir identidad exponencial hiperbólica")
        }
        "Hyperbolic Exponential Identity" if description.starts_with("Recognize ") => {
            Some("Reconocer forma exponencial hiperbólica")
        }
        "Hyperbolic Exponential Identity" => Some("Aplicar identidad exponencial hiperbólica"),
        "Hyperbolic Angle Sum/Difference Identity" => {
            Some("Aplicar identidad hiperbólica de suma/diferencia de ángulos")
        }
        "Hyperbolic Composition" => Some("Cancelar funciones hiperbólicas inversas"),
        "Hyperbolic Double-Angle Identity" => Some("Aplicar identidad hiperbólica de ángulo doble"),
        "Hyperbolic Half-Angle Squares" => Some("Aplicar identidad hiperbólica de ángulo mitad"),
        "Hyperbolic Parity (Odd/Even)" => Some("Aplicar paridad hiperbólica"),
        "Hyperbolic Product-to-Sum Identity" => {
            Some("Aplicar identidad hiperbólica de producto a suma")
        }
        "Hyperbolic Pythagorean Identity" => Some("Aplicar identidad pitagórica hiperbólica"),
        "Hyperbolic Quotient Identity" => Some("Aplicar identidad hiperbólica de cociente"),
        "Hyperbolic Triple-Angle Identity" => {
            Some("Aplicar identidad hiperbólica de ángulo triple")
        }
        "Inverse Hyperbolic Log Identity" => {
            Some("Convertir tangente hiperbólica inversa en logaritmo")
        }
        "Inverse Trig Sum Identity" => Some("Aplicar identidad complementaria arcsin/arccos"),
        "Log-Exp Inverse" => Some("Cancelar logaritmo natural y exponencial inversos"),
        "Exponential-Log Inverse" => Some("Cancelar exponencial y logaritmo inversos"),
        "Exponential-Log Power Inverse" => {
            Some("Cancelar exponencial con logaritmo y conservar exponente")
        }
        "Factorization" => Some("Factorizar"),
        "Factor Out With Division" => Some("Sacar factor usando división"),
        "Finite Product"
            if description.starts_with("Telescoping product:")
                || description.starts_with("Factorized telescoping product:") =>
        {
            Some("Evaluar producto telescópico finito")
        }
        "Finite Product" if description.starts_with("Product of first integers:") => {
            Some("Aplicar producto factorial")
        }
        "Finite Product" if description.starts_with("Product of powers:") => {
            Some("Aplicar producto de potencias")
        }
        "Finite Product" if description.starts_with("Product of constant factor:") => {
            Some("Aplicar producto de constante")
        }
        "Finite Product" => Some("Evaluar producto finito"),
        "Finite Summation" if description.starts_with("Telescoping sum:") => {
            Some("Evaluar suma telescópica finita")
        }
        "Finite Summation" if description.starts_with("Sum of first integers:") => {
            Some("Aplicar fórmula de suma de enteros")
        }
        "Finite Summation" if description.starts_with("Sum of squares:") => {
            Some("Aplicar fórmula de suma de cuadrados")
        }
        "Finite Summation" if description.starts_with("Sum of cubes:") => {
            Some("Aplicar fórmula de suma de cubos")
        }
        "Finite Summation" if description.starts_with("Sum of constant term:") => {
            Some("Aplicar suma de constante")
        }
        "Finite Summation" if description.starts_with("Geometric sum:") => {
            Some("Aplicar fórmula de suma geométrica")
        }
        "Finite Summation" => Some("Evaluar suma finita"),
        "Inverse Trig Composition" => Some("Aplicar composición trigonométrica inversa"),
        "Log Inverse Power" => Some("Convertir potencia logarítmica inversa"),
        "Log Contraction" => Some("Contraer logaritmos"),
        "Merge Sqrt Product" => Some("Combinar raíces en un producto"),
        "Merge Sqrt Quotient" => Some("Combinar raíces en un cociente"),
        "Mixed Fraction Combine" => Some("Combinar parte entera y fracción"),
        "Mixed Fraction Split" => Some("Separar fracción en parte entera y resto"),
        "Negative Base Power" => Some("Simplificar potencia con base negativa"),
        "Polynomial Product Normalize" => Some("Expandir y reagrupar un producto polinómico"),
        "Polynomial division with opaque substitution" => Some("Reconocer un cociente notable"),
        "Pythagorean Chain Identity" | "Pythagorean Identity" => {
            Some("Aplicar la identidad pitagórica")
        }
        "Pythagorean Factor Form" => Some("Aplicar identidad pitagórica"),
        "Rationalize"
        | "Rationalize Cube Root Denominator"
        | "Rationalize Denominator"
        | "Rationalize Linear Sqrt Denominator" => Some("Racionalizar el denominador"),
        "Recognize Cosecant Squared" => Some("Reconocer cosecante cuadrada"),
        "Recognize Secant Squared" => Some("Reconocer secante cuadrada"),
        "Reciprocal Product Identity" => Some("Cancelar funciones trigonométricas recíprocas"),
        "Reciprocal Trig Identity" => Some("Aplicar identidad trigonométrica recíproca"),
        "Reciprocal Pythagorean Identity" => Some("Aplicar identidad pitagórica recíproca"),
        "Trig Quotient" | "Cos-Diff / Sin-Diff Quotient" => {
            Some("Convertir un cociente trigonométrico en tangente")
        }
        "Triple Angle Expansion" | "Triple Angle Identity" => Some("Reescribir ángulo triple"),
        "Quadruple Angle Expansion" => Some("Reescribir ángulo cuádruple"),
        "Quintuple Angle Identity" => Some("Reescribir ángulo quíntuple"),
        "Half-Angle Square Identity" => Some("Aplicar identidad de ángulo mitad"),
        "Half-Angle Tangent Identity" => Some("Aplicar identidad de tangente de ángulo mitad"),
        "Product-to-Sum Identity" => Some("Aplicar producto a suma"),
        "Sum-to-Product Identity" | "Sum-to-Product Identity Cancellation Bridge" => {
            Some("Aplicar suma a producto")
        }
        "Number Theory Operations" if description.starts_with("choose(") => {
            Some("Calcular coeficiente binomial")
        }
        "Number Theory Operations" => Some("Evaluar operación de teoría de números"),
        "Pascal's Identity" => Some("Aplicar identidad de Pascal"),
        "Phase Shift Identity" => Some("Aplicar identidad de desfase"),
        "Power Reduction Identity" => Some("Aplicar reducción de potencias"),
        "Abs Of Sum Of Squares" => Some("Quitar valor absoluto de una expresión no negativa"),
        "Tangent Angle Sum/Diff Identity" => {
            Some("Aplicar identidad de tangente de suma/diferencia de ángulos")
        }
        "Pre-order Common Factor Cancel" => Some("Cancelar un factor común"),
        "Pre-order Difference of Squares Cancel" => {
            Some("Factorizar una diferencia de cuadrados y cancelar")
        }
        "Pre-order Perfect Square Minus Cancel" => {
            Some("Cancelar un cuadrado perfecto con el mismo binomio")
        }
        "Power of a Power" => Some("Multiplicar exponentes"),
        "Sophie Germain Identity" => Some("Expandir la expresión"),
        "Sqrt Perfect Square" => Some("Reconocer un cuadrado perfecto bajo la raíz"),
        "Square Double Angle Contraction" => Some("Contraer cuadrado de ángulo doble"),
        "Simplify Nested Fraction" => Some("Simplificar fracción anidada"),
        "Subtract Fractions" => Some("Restar fracciones en un solo denominador"),
        "Subtraction Self-Cancel" => Some("Restar dos expresiones iguales"),
        "Sum/Difference of Cubes Contraction" => Some("Expandir la expresión"),
        "Telescoping Fraction Combine" => Some("Recomponer fracciones parciales telescópicas"),
        "Telescoping Fraction Split" => Some("Descomponer en fracciones parciales telescópicas"),
        "Tangent Double-Angle Identity" => Some("Aplicar identidad de tangente de ángulo doble"),
        "Trig Expansion" => Some("Expandir una identidad trigonométrica"),
        "Trig Parity (Odd/Even)" => Some("Aplicar paridad trigonométrica"),
        "Trig Square Identity" => Some("Aplicar identidad del cuadrado trigonométrico"),
        _ => None,
    }
}

fn insert_derive_substep_lines(
    lines: &mut Vec<String>,
    steps: &[crate::Step],
    rendered_rule_names: &[&str],
) {
    let mut search_from = 0usize;
    for (step_index, step) in steps.iter().enumerate() {
        if !rendered_rule_names
            .iter()
            .any(|rule_name| step.rule_name.as_str() == *rule_name)
        {
            continue;
        }
        let substeps = step.substeps();
        if substeps.is_empty() {
            continue;
        }

        let step_number = step_index + 1;
        let step_prefix = format!("{step_number}. ");
        let Some(relative_header_index) = lines[search_from..]
            .iter()
            .position(|line| line.starts_with(&step_prefix))
        else {
            continue;
        };
        let header_index = search_from + relative_header_index;
        let insert_at = if lines
            .get(header_index + 1)
            .is_some_and(|line| line.trim_start().starts_with("Before:"))
        {
            header_index + 2
        } else {
            header_index + 1
        };
        let rendered = render_derive_substep_lines(step_number, substeps);
        let rendered_len = rendered.len();
        lines.splice(insert_at..insert_at, rendered);
        search_from = insert_at + rendered_len;
    }
}

fn render_derive_substep_lines(
    step_number: usize,
    substeps: &[cas_solver_core::step_types::SubStep],
) -> Vec<String> {
    let mut lines = vec!["   Subpasos:".to_string()];
    for (substep_index, substep) in substeps.iter().enumerate() {
        lines.push(format!(
            "     {step_number}.{} {}",
            substep_index + 1,
            substep.title
        ));
        for detail in &substep.lines {
            lines.push(format!("         {detail}"));
        }
    }
    lines
}

fn retarget_final_after_line(lines: &mut [String], target: &str) {
    if let Some(index) = lines
        .iter()
        .rposition(|line| line.trim_start().starts_with("After:"))
    {
        lines[index] = format!("   After: {target}");
    }
}

fn retarget_result_line(lines: &mut [String], target: &str) {
    if let Some(index) = lines.iter().rposition(|line| line.starts_with("Result:")) {
        lines[index] = format!("Result: {target}");
    }
}

fn parenthesize_plain_numeric_exponents_for_derive_result(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        output.push(ch);
        if ch != '^' || chars.peek().is_some_and(|next| *next == '(') {
            continue;
        }

        let mut exponent = String::new();
        if chars.peek().is_some_and(|next| *next == '-') {
            exponent.push(chars.next().expect("peeked sign"));
        }

        while chars.peek().is_some_and(|next| next.is_ascii_digit()) {
            exponent.push(chars.next().expect("peeked digit"));
        }

        if exponent.is_empty() || exponent == "-" {
            output.push_str(&exponent);
        } else {
            output.push('(');
            output.push_str(&exponent);
            output.push(')');
        }
    }

    output
}

fn derive_target_match(
    simplifier: &mut crate::Simplifier,
    actual_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    if strong_target_match(&mut simplifier.context, actual_expr, target_expr) {
        return true;
    }

    let difference = simplifier.context.add(Expr::Sub(actual_expr, target_expr));
    let mut temp = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut temp.context, &mut simplifier.context);
    let simplify_options = crate::SimplifyOptions {
        suppress_depth_overflow_warnings: true,
        ..crate::SimplifyOptions::default()
    };
    let (simplified, _steps, _stats) = temp.simplify_with_stats(difference, simplify_options);
    std::mem::swap(&mut temp.context, &mut simplifier.context);

    let zero = simplifier.context.num(0);
    strong_target_match(&mut simplifier.context, simplified, zero)
}

fn derive_semantic_match(
    simplifier: &mut crate::Simplifier,
    actual_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    matches!(
        simplifier.are_equivalent_extended(actual_expr, target_expr),
        crate::EquivalenceResult::True | crate::EquivalenceResult::ConditionalTrue { .. }
    )
}

fn is_finite_aggregate_source(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return false;
    };
    args.len() == 4 && matches!(ctx.sym_name(*fn_id), "sum" | "product")
}

fn render_sum_evaluation_desc<F>(
    kind: &SumEvaluationKind,
    call: &FiniteAggregateCall,
    mut render_expr: F,
) -> String
where
    F: FnMut(ExprId) -> String,
{
    match kind {
        SumEvaluationKind::Telescoping => format!(
            "Telescoping sum: Σ({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        SumEvaluationKind::SumOfFirstIntegers => format!(
            "Sum of first integers: Σ({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        SumEvaluationKind::SumOfSquares => format!(
            "Sum of squares: Σ({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        SumEvaluationKind::SumOfCubes => format!(
            "Sum of cubes: Σ({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        SumEvaluationKind::SumOfConstant => format!(
            "Sum of constant term: Σ({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        SumEvaluationKind::GeometricPower => format!(
            "Geometric sum: Σ({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        SumEvaluationKind::PolynomialLinearity => format!(
            "Polynomial sum by linearity: Σ({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        SumEvaluationKind::FiniteDirect { start, end } => format!(
            "sum({}, {}, {}, {})",
            render_expr(call.term),
            call.var_name,
            start,
            end
        ),
        SumEvaluationKind::DivergentInfinite => format!(
            "Divergent infinite series: Σ({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        SumEvaluationKind::ConvergentInfinite => format!(
            "Convergent geometric series: Σ({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        SumEvaluationKind::UndefinedPole => format!(
            "Undefined: a term of Σ({}, {}) from {} to {} divides by zero",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
    }
}

fn render_product_evaluation_desc<F>(
    kind: &ProductEvaluationKind,
    call: &FiniteAggregateCall,
    mut render_expr: F,
) -> String
where
    F: FnMut(ExprId) -> String,
{
    match kind {
        ProductEvaluationKind::Telescoping => format!(
            "Telescoping product: Π({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        ProductEvaluationKind::FactorizedTelescoping => format!(
            "Factorized telescoping product: Π({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        ProductEvaluationKind::ProductOfFirstIntegers => format!(
            "Product of first integers: Π({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        ProductEvaluationKind::ProductOfPowers => format!(
            "Product of powers: Π({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        ProductEvaluationKind::ProductOfConstant => format!(
            "Product of constant factor: Π({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        ProductEvaluationKind::FiniteDirect { start, end } => format!(
            "product({}, {}, {}, {})",
            render_expr(call.term),
            call.var_name,
            start,
            end
        ),
        ProductEvaluationKind::DivergentInfinite => format!(
            "Divergent infinite product: Π({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
    }
}

fn append_equivalence_summary(lines: &mut Vec<String>, equivalence: &crate::EquivalenceResult) {
    let formatted = crate::format_equivalence_result_lines(equivalence);
    if let Some(first) = formatted.first() {
        lines.push(format!("Equivalence: {first}"));
    }
    lines.extend(formatted.into_iter().skip(1));
}

fn append_derive_requires_lines(
    lines: &mut Vec<String>,
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    derived_expr: ExprId,
) {
    let mut diagnostics = crate::Diagnostics::new();
    diagnostics.extend_required(
        crate::infer_implicit_domain(ctx, source_expr, crate::ValueDomain::RealOnly)
            .conditions()
            .iter()
            .cloned(),
        crate::RequireOrigin::InputImplicit,
    );
    diagnostics.extend_required(
        crate::infer_implicit_domain(ctx, derived_expr, crate::ValueDomain::RealOnly)
            .conditions()
            .iter()
            .cloned(),
        crate::RequireOrigin::OutputImplicit,
    );
    diagnostics.dedup_and_sort(ctx);

    let rendered = cas_solver_core::assumption_render::format_diagnostics_requires_lines(
        ctx,
        &diagnostics,
        Some(derived_expr),
        cas_solver_core::domain_condition::RequiresDisplayLevel::All,
        false,
    );
    if rendered.is_empty() {
        return;
    }
    lines.push("ℹ️ Requires:".to_string());
    lines.extend(rendered);
}

#[cfg(test)]
mod tests {
    use super::{
        derive_semantic_match, evaluate_derive_request_with_session,
        evaluate_derive_resolved_input, generate_planner_candidate_stages,
        humanize_derive_cli_step_rule_suffixes, try_bounded_multistage_derive,
        try_fast_direct_hyperbolic_derive, try_supported_derive_strategies_inner, DeriveStatus,
        DeriveStrategy,
    };
    use cas_session_core::eval::StatelessEvalSession;

    #[test]
    fn derive_cli_rule_suffix_humanizer_keeps_empty_description_step_lines_visible() {
        let mut lines = vec!["2.   [Cancel Reciprocal Exponents]".to_string()];

        humanize_derive_cli_step_rule_suffixes(&mut lines);

        assert_eq!(lines, vec!["2.   [Deshacer raíz y potencia]"]);
    }

    #[test]
    fn planner_candidate_generation_includes_hyperbolic_bridge_stage() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse(
            "sinh(2*x)*cosh(x)+cosh(2*x)*sinh(x)",
            &mut simplifier.context,
        )
        .expect("parse source");
        let target = cas_parser::parse("sinh(3*x)", &mut simplifier.context).expect("parse target");

        let stages = generate_planner_candidate_stages(
            &mut simplifier,
            source,
            target,
            false,
            &crate::SimplifyOptions::default(),
        );

        assert!(stages.iter().any(|stage| derive_semantic_match(
            &mut simplifier,
            stage.expr,
            target
        )));
    }

    #[test]
    #[ignore = "planner candidate generation for this trig bridge still overflows stack in debug; direct trig bridge coverage remains in derive::trig tests"]
    fn planner_candidate_generation_includes_trig_bridge_stage() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("sin(2*x)*cos(x)+cos(2*x)*sin(x)", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("sin(3*x)", &mut simplifier.context).expect("parse target");

        let stages = generate_planner_candidate_stages(
            &mut simplifier,
            source,
            target,
            false,
            &crate::SimplifyOptions::default(),
        );

        assert!(stages.iter().any(|stage| derive_semantic_match(
            &mut simplifier,
            stage.expr,
            target
        )));
    }

    #[test]
    #[ignore = "planner candidate generation for additive phase-shift bridges is too expensive in debug; direct additive phase-shift coverage remains in derive::trig tests"]
    fn planner_candidate_generation_includes_trig_additive_phase_shift_bridge_stage() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("sin(x)+cos(x)+sin(y)+cos(y)", &mut simplifier.context)
            .expect("parse source");
        let intermediate =
            cas_parser::parse("sqrt(2)*sin(x+pi/4)+sin(y)+cos(y)", &mut simplifier.context)
                .expect("parse intermediate");

        let stages = generate_planner_candidate_stages(
            &mut simplifier,
            source,
            intermediate,
            false,
            &crate::SimplifyOptions::default(),
        );

        assert!(stages.iter().any(|stage| derive_semantic_match(
            &mut simplifier,
            stage.expr,
            intermediate
        )));
    }

    #[test]
    fn planner_prefers_single_bridge_stage_over_semantic_cleanup_tail() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("sinh(x)+sinh(y)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("2*sinh((x+y)/2)*cosh((x-y)/2)", &mut simplifier.context)
            .expect("parse target");

        let derived = try_bounded_multistage_derive(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
        )
        .expect("planner should derive hyperbolic sum-to-product target");

        assert_eq!(derived.2, DeriveStrategy::Planner);
        assert_eq!(derived.0, target);
        assert!(!derived.1.is_empty());
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Product-to-Sum Identity");
    }

    #[test]
    fn run_expand_stage_rewrites_hyperbolic_sinh_difference_target_aware() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("sinh(x-y)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("sinh(x)*cosh(y)-sinh(y)*cosh(x)", &mut simplifier.context)
            .expect("parse target");

        let stage = super::run_expand_stage(&mut simplifier, source, target, true);
        assert_eq!(stage.expr, target);
        assert!(!stage.steps.is_empty());
    }

    #[test]
    fn run_expand_stage_rewrites_sophie_germain_product_target_aware() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse(
            "(x^2 - 2*x*y + 2*y^2)*(x^2 + 2*x*y + 2*y^2)",
            &mut simplifier.context,
        )
        .expect("parse source");
        let target =
            cas_parser::parse("x^4 + 4*y^4", &mut simplifier.context).expect("parse target");

        let stage = super::run_expand_stage(&mut simplifier, source, target, true);
        assert_eq!(stage.expr, target);
        assert_eq!(stage.steps.len(), 1);
        assert_eq!(stage.steps[0].rule_name, "Sophie Germain Identity");
    }

    #[test]
    fn truncate_steps_after_first_target_match_uses_global_after_snapshot() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("(sin(x)+cos(x))^2", &mut ctx).expect("parse source");
        let local_before = cas_parser::parse("2*sin(x)*cos(x)+1", &mut ctx).expect("parse local");
        let local_after = cas_parser::parse("sin(2*x)", &mut ctx).expect("parse local after");
        let target = cas_parser::parse("1+sin(2*x)", &mut ctx).expect("parse target");
        let extra = cas_parser::parse("2*sin(x)*cos(x)", &mut ctx).expect("parse extra");

        let mut steps = vec![
            crate::Step::with_snapshots(
                "contract double angle",
                "Double Angle Identity",
                local_before,
                local_after,
                Vec::new(),
                Some(&ctx),
                source,
                target,
            ),
            crate::Step::with_snapshots(
                "backtrack",
                "Double Angle Identity",
                local_after,
                extra,
                Vec::new(),
                Some(&ctx),
                target,
                extra,
            ),
        ];

        super::truncate_steps_after_first_target_match(&mut steps, target, &ctx);
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].global_after, Some(target));
    }

    #[test]
    fn prune_semantically_noop_derive_steps_drops_pure_canonicalize_tail() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("sinh(x)+sinh(y)", &mut ctx).expect("parse source");
        let target =
            cas_parser::parse("2*sinh((x+y)/2)*cosh((x-y)/2)", &mut ctx).expect("parse target");

        let steps = vec![
            crate::Step::with_snapshots(
                "Hyperbolic sum-to-product",
                "Hyperbolic Product-to-Sum Identity",
                source,
                target,
                Vec::new(),
                Some(&ctx),
                source,
                target,
            ),
            crate::Step::with_snapshots(
                "canonicalize",
                "Canonicalize Multiplication",
                target,
                target,
                Vec::new(),
                Some(&ctx),
                target,
                target,
            ),
        ];

        let pruned = super::prune_semantically_noop_derive_steps(steps, &ctx);
        assert_eq!(pruned.len(), 1);
        assert_eq!(pruned[0].rule_name, "Hyperbolic Product-to-Sum Identity");
    }

    #[test]
    fn direct_derive_prefers_expand_for_hyperbolic_sinh_difference_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("sinh(x-y)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("sinh(x)*cosh(y)-sinh(y)*cosh(x)", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            false,
        )
        .expect("direct derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::Expand);
        assert!(!derived.1.is_empty());
    }

    #[test]
    fn direct_derive_expands_fractional_binomial_square_without_generic_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("(x + 1/2)^2", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("x^2 + x + 1/4", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::Expand);
        assert!(
            derived
                .1
                .iter()
                .any(|step| step.rule_name == "Binomial Expansion"),
            "expected a binomial expansion step, got {:?}",
            derived
                .1
                .iter()
                .map(|step| step.rule_name.as_str())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn direct_derive_expands_cosine_fourth_power_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("cos(x)^4", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("(3+4*cos(2*x)+cos(4*x))/8", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            false,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
    }

    #[test]
    fn direct_derive_prefers_combine_like_terms_for_simple_duplicate_sum() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("x + x", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("2*x", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::CombineLikeTerms);
    }

    #[test]
    fn direct_derive_rewrites_negative_pythagorean_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("-sin(x)^2", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("cos(x)^2 - 1", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            false,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigRewrite);
    }

    #[test]
    fn derive_eval_resolved_input_handles_cosine_fourth_power() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("cos(x)^4", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("(3+4*cos(2*x)+cos(4*x))/8", &mut simplifier.context)
            .expect("parse target");

        let output = evaluate_derive_resolved_input(
            &mut simplifier,
            source,
            target,
            false,
            crate::SimplifyOptions::default(),
        );

        assert_eq!(output.derived_expr, target);
        assert!(matches!(
            output.status,
            DeriveStatus::Derived {
                strategy: DeriveStrategy::TrigExpand
            }
        ));
    }

    #[test]
    fn derive_request_with_session_handles_cosine_fourth_power() {
        let mut engine = crate::Engine::new();
        let options = crate::EvalOptions {
            steps_mode: crate::StepsMode::Off,
            ..crate::EvalOptions::default()
        };
        let mut session = StatelessEvalSession::<
            crate::EvalOptions,
            crate::DomainMode,
            crate::RequiredItem,
            crate::Step,
            crate::Diagnostics,
        >::new(options);

        let parsed =
            cas_parser::parse("cos(x)^4", &mut engine.simplifier.context).expect("parse source");
        let target = cas_parser::parse("(3+4*cos(2*x)+cos(4*x))/8", &mut engine.simplifier.context)
            .expect("parse target");

        let output = evaluate_derive_request_with_session(
            &mut engine,
            &mut session,
            "derive cos(x)^4, (3+4*cos(2*x)+cos(4*x))/8".to_string(),
            parsed,
            target,
            false,
        )
        .expect("derive request should succeed");

        match output.result {
            crate::EvalResult::Expr(expr) => assert_eq!(expr, target),
            other => panic!("unexpected derive result: {other:?}"),
        }
    }

    #[test]
    fn derive_request_with_session_handles_negative_pythagorean_target() {
        let mut engine = crate::Engine::new();
        let options = crate::EvalOptions {
            steps_mode: crate::StepsMode::Off,
            ..crate::EvalOptions::default()
        };
        let mut session = StatelessEvalSession::<
            crate::EvalOptions,
            crate::DomainMode,
            crate::RequiredItem,
            crate::Step,
            crate::Diagnostics,
        >::new(options);

        let parsed =
            cas_parser::parse("-sin(x)^2", &mut engine.simplifier.context).expect("parse source");
        let target = cas_parser::parse("cos(x)^2 - 1", &mut engine.simplifier.context)
            .expect("parse target");

        let output = evaluate_derive_request_with_session(
            &mut engine,
            &mut session,
            "derive -sin(x)^2, cos(x)^2-1".to_string(),
            parsed,
            target,
            false,
        )
        .expect("derive request should succeed");

        match output.result {
            crate::EvalResult::Expr(expr) => assert_eq!(expr, target),
            other => panic!("unexpected derive result: {other:?}"),
        }
    }

    #[test]
    fn derive_request_with_session_handles_trig_sum_to_triple_angle_bridge() {
        let mut engine = crate::Engine::new();
        let options = crate::EvalOptions {
            steps_mode: crate::StepsMode::Off,
            ..crate::EvalOptions::default()
        };
        let mut session = StatelessEvalSession::<
            crate::EvalOptions,
            crate::DomainMode,
            crate::RequiredItem,
            crate::Step,
            crate::Diagnostics,
        >::new(options);

        let parsed = cas_parser::parse(
            "sin(2*x)*cos(x)+cos(2*x)*sin(x)",
            &mut engine.simplifier.context,
        )
        .expect("parse source");
        let target =
            cas_parser::parse("sin(3*x)", &mut engine.simplifier.context).expect("parse target");

        let output = evaluate_derive_request_with_session(
            &mut engine,
            &mut session,
            "derive sin(2*x)*cos(x)+cos(2*x)*sin(x), sin(3*x)".to_string(),
            parsed,
            target,
            false,
        )
        .expect("derive request should succeed");

        match output.result {
            crate::EvalResult::Expr(expr) => assert_eq!(expr, target),
            other => panic!("unexpected derive result: {other:?}"),
        }
    }

    #[test]
    fn derive_request_with_session_handles_trig_sum_to_triple_angle_polynomial() {
        let mut engine = crate::Engine::new();
        let options = crate::EvalOptions {
            steps_mode: crate::StepsMode::Off,
            ..crate::EvalOptions::default()
        };
        let mut session = StatelessEvalSession::<
            crate::EvalOptions,
            crate::DomainMode,
            crate::RequiredItem,
            crate::Step,
            crate::Diagnostics,
        >::new(options);

        let parsed = cas_parser::parse(
            "sin(2*x)*cos(x)+cos(2*x)*sin(x)",
            &mut engine.simplifier.context,
        )
        .expect("parse source");
        let target = cas_parser::parse("3*sin(x)-4*sin(x)^3", &mut engine.simplifier.context)
            .expect("parse target");

        let output = evaluate_derive_request_with_session(
            &mut engine,
            &mut session,
            "derive sin(2*x)*cos(x)+cos(2*x)*sin(x), 3*sin(x)-4*sin(x)^3".to_string(),
            parsed,
            target,
            false,
        )
        .expect("derive request should succeed");

        match output.result {
            crate::EvalResult::Expr(expr) => assert_eq!(expr, target),
            other => panic!("unexpected derive result: {other:?}"),
        }
    }

    #[test]
    fn direct_derive_prefers_expand_for_hyperbolic_sinh_sum_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("sinh(x+y)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("sinh(x)*cosh(y)+sinh(y)*cosh(x)", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            false,
        )
        .expect("direct derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::Expand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(
            derived.1[0].rule_name,
            "Hyperbolic Angle Sum/Difference Identity"
        );
    }

    #[test]
    fn direct_derive_prefers_expand_for_recursive_hyperbolic_cosh_sum_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("cosh(6*x)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse(
            "cosh(5*x)*cosh(x)+sinh(5*x)*sinh(x)",
            &mut simplifier.context,
        )
        .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            false,
        )
        .expect("direct derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::Expand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(
            derived.1[0].rule_name,
            "Hyperbolic Angle Sum/Difference Identity"
        );
    }

    #[test]
    fn bounded_multistage_derive_reaches_hyperbolic_sum_to_split_exponential_products() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse(
            "sinh(2*x)*cosh(x)+cosh(2*x)*sinh(x)",
            &mut simplifier.context,
        )
        .expect("parse source");
        let target = cas_parser::parse("(e^x*e^(2*x)-e^(-x)*e^(-2*x))/2", &mut simplifier.context)
            .expect("parse target");

        let derived = try_bounded_multistage_derive(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
        )
        .expect("planner should derive target");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::Planner);
        assert!(!derived.1.is_empty());
    }

    #[test]
    #[ignore = "bounded planner derivation still overflows stack in debug; direct trig bridge and CLI derive coverage remain in narrower tests"]
    fn bounded_multistage_derive_reaches_trig_sum_to_triple_angle_polynomial() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("sin(2*x)*cos(x)+cos(2*x)*sin(x)", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("3*sin(x)-4*sin(x)^3", &mut simplifier.context)
            .expect("parse target");

        let derived = try_bounded_multistage_derive(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
        )
        .expect("planner should derive trig target");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::Planner);
        assert!(!derived.1.is_empty());
    }

    #[test]
    fn direct_derive_contracts_phase_shift_sum_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("sin(x)+cos(x)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("sqrt(2)*sin(x+pi/4)", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigContract);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Phase Shift Identity");
    }

    #[test]
    fn direct_derive_expands_phase_shift_sum_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("sqrt(2)*sin(x+pi/4)", &mut simplifier.context)
            .expect("parse source");
        let target =
            cas_parser::parse("sin(x)+cos(x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Phase Shift Identity");
    }

    #[test]
    fn direct_derive_contracts_scaled_phase_shift_sum_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("2*sin(x)+2*cos(x)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("2*sqrt(2)*sin(x+pi/4)", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigContract);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Phase Shift Identity");
    }

    #[test]
    fn direct_derive_expands_scaled_phase_shift_sum_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("2*sqrt(2)*sin(x+pi/4)", &mut simplifier.context)
            .expect("parse source");
        let target =
            cas_parser::parse("2*sin(x)+2*cos(x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Phase Shift Identity");
    }

    #[test]
    fn direct_derive_contracts_exact_third_phase_shift_sum_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("2*sin(x)+2*sqrt(3)*cos(x)", &mut simplifier.context)
            .expect("parse source");
        let target =
            cas_parser::parse("4*sin(x+pi/3)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigContract);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Phase Shift Identity");
    }

    #[test]
    fn direct_derive_expands_exact_third_phase_shift_sum_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("4*sin(x+pi/3)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("2*sin(x)+2*sqrt(3)*cos(x)", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Phase Shift Identity");
    }

    #[test]
    fn direct_derive_contracts_exact_sixth_phase_shift_sum_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("sqrt(3)*sin(x)+cos(x)", &mut simplifier.context)
            .expect("parse source");
        let target =
            cas_parser::parse("2*sin(x+pi/6)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigContract);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Phase Shift Identity");
    }

    #[test]
    fn direct_derive_expands_exact_sixth_phase_shift_sum_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("2*sin(x+pi/6)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("sqrt(3)*sin(x)+cos(x)", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Phase Shift Identity");
    }

    #[test]
    fn direct_derive_contracts_general_phase_shift_sum_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("3*sin(x)+4*cos(x)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("5*sin(x+arctan(4/3))", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigContract);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Phase Shift Identity");
    }

    #[test]
    fn direct_derive_expands_general_phase_shift_sum_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("5*sin(x+arctan(4/3))", &mut simplifier.context)
            .expect("parse source");
        let target =
            cas_parser::parse("3*sin(x)+4*cos(x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Phase Shift Identity");
    }

    #[test]
    fn direct_derive_contracts_general_phase_shift_sum_with_passthrough_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("3*sin(x)+4*cos(x)+a", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("5*sin(x+arctan(4/3))+a", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigContract);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Phase Shift Identity");
    }

    #[test]
    fn direct_derive_contracts_phase_shift_shifted_sine_to_shifted_cosine_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("sqrt(2)*sin(x+pi/4)", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("sqrt(2)*cos(x-pi/4)", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigContract);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Phase Shift Identity");
    }

    #[test]
    fn direct_derive_contracts_phase_shift_shifted_terms_with_passthrough_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("sqrt(2)*sin(x+pi/4)+a", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("sqrt(2)*cos(x-pi/4)+a", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigContract);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Phase Shift Identity");
    }

    #[test]
    fn direct_derive_contracts_general_phase_shift_shifted_sine_to_shifted_cosine_without_planner()
    {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("5*sin(x+arctan(4/3))", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("5*cos(x-arctan(3/4))", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigContract);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Phase Shift Identity");
    }

    #[test]
    fn direct_derive_contracts_general_phase_shift_shifted_terms_with_passthrough_without_planner()
    {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("5*sin(x+arctan(4/3))+a", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("5*cos(x-arctan(3/4))+a", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigContract);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Phase Shift Identity");
    }

    #[test]
    fn direct_derive_rewrites_perfect_square_root_with_passthrough_without_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("sqrt(a^2 + 2*a*b + b^2)+c", &mut simplifier.context)
            .expect("parse source");
        let target =
            cas_parser::parse("abs(a+b)+c", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::RadicalRewrite);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Sqrt Perfect Square");
    }

    #[test]
    fn direct_derive_rewrites_sqrt_squared_symbol_without_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("sqrt(x^2)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("abs(x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::RadicalRewrite);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Sqrt Perfect Square");
    }

    #[test]
    fn direct_derive_rewrites_square_of_square_root_without_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("sqrt(x)^2", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("x", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::RadicalRewrite);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Square of Square Root");
        assert_eq!(
            derived.1[0]
                .meta
                .as_ref()
                .expect("expected step metadata")
                .required_conditions
                .len(),
            1
        );
    }

    #[test]
    fn direct_derive_rewrites_hyperbolic_double_angle_with_passthrough_without_expand() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("2*sinh(x)*cosh(x)+a", &mut simplifier.context)
            .expect("parse source");
        let target =
            cas_parser::parse("sinh(2*x)+a", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::HyperbolicRewrite);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Double-Angle Identity");
    }

    #[test]
    fn direct_derive_rewrites_hyperbolic_pythagorean_with_passthrough_without_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("cosh(x)^2-sinh(x)^2+a", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("1+a", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::HyperbolicRewrite);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Pythagorean Identity");
    }

    #[test]
    fn direct_derive_contracts_hyperbolic_quotient_without_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("sinh(x)/cosh(x)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("tanh(x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::HyperbolicRewrite);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Quotient Identity");
        assert_eq!(
            derived.1[0].description,
            "Recognize sinh(u) / cosh(u) as tanh(u)"
        );
    }

    #[test]
    fn direct_derive_rewrites_hyperbolic_half_angle_squares_with_specific_rule() {
        for (source, target) in [
            ("cosh(x/2)^2", "(cosh(x)+1)/2"),
            ("sinh(x/2)^2", "(cosh(x)-1)/2"),
        ] {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let source = cas_parser::parse(source, &mut simplifier.context).expect("parse source");
            let target = cas_parser::parse(target, &mut simplifier.context).expect("parse target");

            let derived = try_supported_derive_strategies_inner(
                &mut simplifier,
                source,
                target,
                true,
                &crate::SimplifyOptions::default(),
                true,
                true,
            )
            .expect("derive should succeed");

            assert_eq!(derived.0, target);
            assert_eq!(derived.2, DeriveStrategy::HyperbolicRewrite);
            assert_eq!(derived.1.len(), 1);
            assert_eq!(derived.1[0].rule_name, "Hyperbolic Half-Angle Squares");
        }
    }

    #[test]
    fn direct_derive_rewrites_consecutive_factorial_ratio_with_passthrough_without_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("(n+1)!/n!+a", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("n+1+a", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::FactorialRewrite);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Consecutive Factorial Ratio");
    }

    #[test]
    fn direct_derive_rewrites_safe_arcsin_arctan_composition_without_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("asin(x/sqrt(x^2 + 1))", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("arctan(x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::InverseTrigRewrite);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Inverse Trig Composition");
        let substeps = derived.1[0].substeps();
        assert_eq!(substeps.len(), 3);
        assert_eq!(
            substeps[0].title,
            "Reconocer el argumento como seno de una arctangente"
        );
        assert!(substeps[0]
            .lines
            .iter()
            .any(|line| line.contains("sin(arctan(x))")));
        assert!(substeps[1]
            .lines
            .iter()
            .any(|line| line.contains("arcsin(sin(arctan(x)))")));
        assert!(substeps[2]
            .lines
            .iter()
            .any(|line| line.contains("-> arctan(x)")));
    }

    #[test]
    fn direct_derive_rewrites_direct_inverse_trig_compositions_without_generic_simplify() {
        for (source_text, target_text) in [
            ("sin(arcsin(x))", "x"),
            ("cos(arccos(x))", "x"),
            ("tan(arctan(x))", "x"),
        ] {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let source =
                cas_parser::parse(source_text, &mut simplifier.context).expect("parse source");
            let target =
                cas_parser::parse(target_text, &mut simplifier.context).expect("parse target");

            let derived = try_supported_derive_strategies_inner(
                &mut simplifier,
                source,
                target,
                true,
                &crate::SimplifyOptions::default(),
                true,
                true,
            )
            .unwrap_or_else(|| panic!("derive should succeed for {source_text}"));

            assert_eq!(derived.0, target);
            assert_eq!(derived.2, DeriveStrategy::InverseTrigRewrite);
            assert_eq!(derived.1.len(), 1);
            assert_eq!(derived.1[0].rule_name, "Inverse Trig Composition");
        }
    }

    #[test]
    fn direct_derive_rewrites_arcsin_arccos_sum_without_generic_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("arcsin(x)+arccos(x)", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("pi/2", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::InverseTrigRewrite);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Inverse Trig Sum Identity");
    }

    #[test]
    fn direct_derive_rewrites_arctan_right_triangle_projections_without_simplify() {
        for (source_text, target_text) in [
            ("sin(arctan(x))", "x/sqrt(1+x^2)"),
            ("cos(arctan(x))", "1/sqrt(1+x^2)"),
        ] {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let source =
                cas_parser::parse(source_text, &mut simplifier.context).expect("parse source");
            let target =
                cas_parser::parse(target_text, &mut simplifier.context).expect("parse target");

            let derived = try_supported_derive_strategies_inner(
                &mut simplifier,
                source,
                target,
                true,
                &crate::SimplifyOptions::default(),
                true,
                true,
            )
            .expect("derive should succeed");

            assert_eq!(derived.0, target);
            assert_eq!(derived.2, DeriveStrategy::InverseTrigRewrite);
            assert_eq!(derived.1.len(), 1);
            assert_eq!(derived.1[0].rule_name, "Inverse Trig Composition");
        }
    }

    #[test]
    fn direct_derive_rewrites_arcsin_arccos_complement_projections_without_simplify() {
        for (source_text, target_text) in [
            ("cos(arcsin(x))", "sqrt(1-x^2)"),
            ("sin(arccos(x))", "sqrt(1-x^2)"),
        ] {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let source =
                cas_parser::parse(source_text, &mut simplifier.context).expect("parse source");
            let target =
                cas_parser::parse(target_text, &mut simplifier.context).expect("parse target");

            let derived = try_supported_derive_strategies_inner(
                &mut simplifier,
                source,
                target,
                true,
                &crate::SimplifyOptions::default(),
                true,
                true,
            )
            .expect("derive should succeed");

            assert_eq!(derived.0, target);
            assert_eq!(derived.2, DeriveStrategy::InverseTrigRewrite);
            assert_eq!(derived.1.len(), 1);
            assert_eq!(derived.1[0].rule_name, "Inverse Trig Composition");
        }
    }

    #[test]
    fn direct_derive_rewrites_tan_arcsin_projection_without_trig_expand() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("tan(arcsin(x))", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("x/sqrt(1-x^2)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::InverseTrigRewrite);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Inverse Trig Composition");
    }

    #[test]
    fn direct_derive_labels_calculus_diff_without_generic_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("diff(arcsin(2*x-1)/2,x)", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("1/(2*sqrt(x)*sqrt(1-x))", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::CalculusDiff);
        assert!(derived.1.iter().any(|step| {
            matches!(
                step.rule_name.as_str(),
                "Symbolic Differentiation" | "Calcular la derivada"
            )
        }));
    }

    #[test]
    fn direct_derive_combines_radical_like_terms_without_generic_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("sqrt(8)+sqrt(2)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("3*sqrt(2)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::CombineLikeTerms);
        assert!(derived
            .1
            .iter()
            .any(|step| step.rule_name == "Combine Like Terms"));
    }

    #[test]
    fn direct_derive_combines_subtracted_radical_like_terms_without_generic_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("sqrt(18)-sqrt(2)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("2*sqrt(2)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::CombineLikeTerms);
        assert!(derived
            .1
            .iter()
            .any(|step| step.rule_name == "Combine Like Terms"));
    }

    #[test]
    fn direct_derive_labels_log_inverse_power_tower_without_generic_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("x^(ln(y)/ln(x))", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("y", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::LogInversePower);
        assert_eq!(derived.1.len(), 2);
        assert_eq!(derived.1[0].rule_name, "Log Inverse Power");
        assert_eq!(derived.1[1].rule_name, "Exponential-Log Inverse");
    }

    #[test]
    fn direct_derive_labels_log_exp_inverses_without_generic_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("ln(exp(x))", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("x", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::ExponentialRewrite);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Log-Exp Inverse");

        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("exp(ln(x))", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("x", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::ExponentialRewrite);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Exponential-Log Inverse");
    }

    #[test]
    fn direct_derive_labels_log_exp_power_inverse_as_two_exponential_steps() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("ln(exp(x)^2)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("2*x", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::ExponentialRewrite);
        assert_eq!(derived.1.len(), 2);
        assert_eq!(derived.1[0].rule_name, "Power of a Power");
        assert_eq!(derived.1[1].rule_name, "Log-Exp Inverse");
    }

    #[test]
    fn direct_derive_labels_atanh_square_ratio_log_without_generic_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("atanh((x^2 - 1)/(x^2 + 1))", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("ln(x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::HyperbolicRewrite);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Inverse Hyperbolic Log Identity");
    }

    #[test]
    fn direct_derive_labels_hyperbolic_composition_without_generic_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("sinh(asinh(x))", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("x", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::HyperbolicRewrite);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Composition");
    }

    #[test]
    fn direct_derive_labels_hyperbolic_special_values_without_generic_simplify() {
        let cases = [
            ("sinh(0)", "0"),
            ("cosh(0)", "1"),
            ("tanh(0)", "0"),
            ("asinh(0)", "0"),
            ("atanh(0)", "0"),
            ("acosh(1)", "0"),
        ];

        for (source_text, target_text) in cases {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let source =
                cas_parser::parse(source_text, &mut simplifier.context).expect("parse source");
            let target =
                cas_parser::parse(target_text, &mut simplifier.context).expect("parse target");

            let derived = try_supported_derive_strategies_inner(
                &mut simplifier,
                source,
                target,
                true,
                &crate::SimplifyOptions::default(),
                true,
                true,
            )
            .unwrap_or_else(|| panic!("derive should succeed for {source_text}"));

            assert_eq!(derived.0, target);
            assert_eq!(derived.2, DeriveStrategy::HyperbolicRewrite);
            assert_eq!(derived.1.len(), 1);
            assert_eq!(derived.1[0].rule_name, "Evaluate Hyperbolic Functions");
        }
    }

    #[test]
    fn direct_derive_labels_trig_special_values_without_generic_simplify() {
        let cases = [
            ("sin(0)", "0"),
            ("cos(0)", "1"),
            ("tan(0)", "0"),
            ("asin(0)", "0"),
            ("acos(1)", "0"),
            ("atan(0)", "0"),
            ("arctan(sqrt(3))", "pi/3"),
            ("cos(2*pi/3)", "-1/2"),
            ("cos(3*pi/4)", "-sqrt(2)/2"),
            ("cos(5*pi/6)", "-sqrt(3)/2"),
            ("sec(pi/4)", "sqrt(2)"),
            ("csc(pi/6)", "2"),
            ("cot(pi/4)", "1"),
        ];

        for (source_text, target_text) in cases {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let source =
                cas_parser::parse(source_text, &mut simplifier.context).expect("parse source");
            let target =
                cas_parser::parse(target_text, &mut simplifier.context).expect("parse target");

            let derived = try_supported_derive_strategies_inner(
                &mut simplifier,
                source,
                target,
                true,
                &crate::SimplifyOptions::default(),
                true,
                true,
            )
            .unwrap_or_else(|| panic!("derive should succeed for {source_text}"));

            assert_eq!(derived.0, target);
            assert_eq!(derived.2, DeriveStrategy::TrigRewrite);
            assert_eq!(derived.1.len(), 1);
            assert_eq!(derived.1[0].rule_name, "Evaluate Trigonometric Functions");
        }
    }

    #[test]
    fn direct_derive_cancels_difference_of_squares_fraction_with_passthrough_without_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("(a^2-b^2)/(a-b)+c", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("a+b+c", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::FractionCancel);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(
            derived.1[0].rule_name,
            "Pre-order Difference of Squares Cancel"
        );
    }

    #[test]
    fn direct_derive_cancels_difference_of_cubes_fraction_with_passthrough_without_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("(a^3-b^3)/(a-b)+c", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("a^2+a*b+b^2+c", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::FractionCancel);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(
            derived.1[0].rule_name,
            "Cancel Sum/Difference of Cubes Fraction"
        );
    }

    #[test]
    fn direct_derive_rewrites_odd_half_power_with_passthrough_without_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("sqrt(x^3)+a", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("abs(x)*sqrt(x)+a", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::OddHalfPowerExpand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Expand Odd Half Power");
    }

    #[test]
    fn direct_derive_rewrites_higher_odd_half_power_with_passthrough_without_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("sqrt(x^7)+a", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("x^3*sqrt(x)+a", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::OddHalfPowerExpand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Expand Odd Half Power");
    }

    #[test]
    fn direct_derive_rewrites_reciprocal_trig_product_with_passthrough_without_simplify() {
        for (source, target, description) in [
            ("tan(x)*cot(x)+a", "1+a", "Recognize tan(u) · cot(u) = 1"),
            ("sin(x)*csc(x)", "1", "Recognize sin(u) · csc(u) = 1"),
            ("cos(x)*sec(x)", "1", "Recognize cos(u) · sec(u) = 1"),
        ] {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let source = cas_parser::parse(source, &mut simplifier.context).expect("parse source");
            let target = cas_parser::parse(target, &mut simplifier.context).expect("parse target");

            let derived = try_supported_derive_strategies_inner(
                &mut simplifier,
                source,
                target,
                true,
                &crate::SimplifyOptions::default(),
                true,
                true,
            )
            .expect("derive should succeed");

            assert_eq!(derived.0, target);
            assert_eq!(derived.2, DeriveStrategy::TrigRewrite);
            assert_eq!(derived.1.len(), 1);
            assert_eq!(derived.1[0].rule_name, "Reciprocal Product Identity");
            assert_eq!(derived.1[0].description, description);
        }
    }

    #[test]
    fn direct_derive_contracts_reciprocal_trig_root_forms_without_simplify() {
        for (source, target, description) in [
            ("1/cos(x)", "sec(x)", "Recognize 1 / cos(u) as sec(u)"),
            ("1/sin(x)", "csc(x)", "Recognize 1 / sin(u) as csc(u)"),
            (
                "cos(x)/sin(x)",
                "cot(x)",
                "Recognize cos(u) / sin(u) as cot(u)",
            ),
        ] {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let source = cas_parser::parse(source, &mut simplifier.context).expect("parse source");
            let target = cas_parser::parse(target, &mut simplifier.context).expect("parse target");

            let derived = try_supported_derive_strategies_inner(
                &mut simplifier,
                source,
                target,
                true,
                &crate::SimplifyOptions::default(),
                true,
                true,
            )
            .expect("derive should succeed");

            assert_eq!(derived.0, target);
            assert_eq!(derived.2, DeriveStrategy::TrigContract);
            assert_eq!(derived.1.len(), 1);
            assert_eq!(derived.1[0].rule_name, "Reciprocal Trig Identity");
            assert_eq!(derived.1[0].description, description);
        }
    }

    #[test]
    fn direct_derive_rewrites_csc_cot_pythagorean_to_one_without_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("csc(x)^2 - cot(x)^2", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("1", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigRewrite);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Reciprocal Pythagorean Identity");
    }

    #[test]
    fn direct_derive_expands_tangent_to_sine_over_cosine_without_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("tan(x)", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("sin(x)/cos(x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Trig Expansion");
    }

    #[test]
    fn direct_derive_expands_secant_to_reciprocal_cosine_without_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("sec(x)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("1/cos(x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Reciprocal Trig Identity");
        assert_eq!(derived.1[0].description, "Expand sec(u) as 1 / cos(u)");
    }

    #[test]
    fn direct_derive_expands_cosecant_to_reciprocal_sine_without_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("csc(x)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("1/sin(x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Reciprocal Trig Identity");
        assert_eq!(derived.1[0].description, "Expand csc(u) as 1 / sin(u)");
    }

    #[test]
    fn direct_derive_expands_cotangent_to_cosine_over_sine_without_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("cot(x)", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("cos(x)/sin(x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Reciprocal Trig Identity");
        assert_eq!(derived.1[0].description, "Expand cot(u) as cos(u) / sin(u)");
    }

    #[test]
    fn direct_derive_rewrites_negative_tangent_parity_with_specific_rule() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("tan(-x)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("-tan(x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Trig Parity (Odd/Even)");
        assert_eq!(
            derived.1[0].description,
            "Apply a trigonometric odd/even parity identity"
        );
    }

    #[test]
    fn direct_derive_rewrites_negative_hyperbolic_parity_with_specific_rule() {
        for (source_text, target_text) in [
            ("sinh(-x)", "-sinh(x)"),
            ("cosh(-x)", "cosh(x)"),
            ("tanh(-x)", "-tanh(x)"),
        ] {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let source =
                cas_parser::parse(source_text, &mut simplifier.context).expect("parse source");
            let target =
                cas_parser::parse(target_text, &mut simplifier.context).expect("parse target");

            let derived = try_supported_derive_strategies_inner(
                &mut simplifier,
                source,
                target,
                true,
                &crate::SimplifyOptions::default(),
                true,
                true,
            )
            .expect("derive should succeed");

            assert_eq!(derived.0, target, "{source_text} -> {target_text}");
            assert_eq!(derived.2, DeriveStrategy::HyperbolicRewrite);
            assert_eq!(derived.1.len(), 1);
            assert_eq!(derived.1[0].rule_name, "Hyperbolic Parity (Odd/Even)");
            assert_eq!(
                derived.1[0].description,
                "Apply a hyperbolic odd/even parity identity"
            );
        }
    }

    #[test]
    fn direct_derive_rewrites_cofunction_sine_cosine_pair_with_specific_rule() {
        for (source_text, target_text) in [("sin(pi/2 - x)", "cos(x)"), ("cos(pi/2 - x)", "sin(x)")]
        {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let source =
                cas_parser::parse(source_text, &mut simplifier.context).expect("parse source");
            let target =
                cas_parser::parse(target_text, &mut simplifier.context).expect("parse target");

            let derived = try_supported_derive_strategies_inner(
                &mut simplifier,
                source,
                target,
                true,
                &crate::SimplifyOptions::default(),
                true,
                true,
            )
            .expect("derive should succeed");

            assert_eq!(derived.0, target, "{source_text} -> {target_text}");
            assert_eq!(derived.2, DeriveStrategy::TrigExpand);
            assert_eq!(derived.1.len(), 1);
            assert_eq!(derived.1[0].rule_name, "Cofunction Identity");
            assert_eq!(
                derived.1[0].description,
                "Apply a sine/cosine cofunction identity"
            );
        }
    }

    #[test]
    fn trig_contract_stage_matches_general_phase_shift_shifted_terms_with_passthrough() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("5*sin(x+arctan(4/3))+a", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("5*cos(x-arctan(3/4))+a", &mut simplifier.context)
            .expect("parse target");

        let stage = super::run_trig_contract_stage(&mut simplifier, source, target, true);

        assert!(super::derive_target_match(
            &mut simplifier,
            stage.expr,
            target
        ));
        assert_eq!(stage.steps.len(), 1);
        assert_eq!(stage.steps[0].rule_name, "Phase Shift Identity");
    }

    #[test]
    fn direct_derive_contracts_half_scaled_sine_double_angle_without_generic_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("sin(x)*cos(x)", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("sin(2*x)/2", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigContract);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Double Angle Expansion");
    }

    #[test]
    fn direct_derive_expands_general_phase_shift_sum_with_passthrough_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("5*sin(x+arctan(4/3))+a", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("3*sin(x)+4*cos(x)+a", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Phase Shift Identity");
    }

    #[test]
    fn direct_derive_contracts_phase_shift_sum_with_passthrough_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("sin(x)+cos(x)+a", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("sqrt(2)*sin(x+pi/4)+a", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigContract);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Phase Shift Identity");
    }

    #[test]
    fn direct_derive_expands_scaled_phase_shift_sum_with_passthrough_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("2*sqrt(2)*sin(x+pi/4)+a", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("2*sin(x)+2*cos(x)+a", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Phase Shift Identity");
    }

    #[test]
    fn direct_derive_reaches_repeated_phase_shift_pair_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("sin(x)+cos(x)+sin(y)+cos(y)", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse(
            "sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4)",
            &mut simplifier.context,
        )
        .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigContract);
        assert_eq!(derived.1.len(), 2);
        assert!(derived
            .1
            .iter()
            .all(|step| step.rule_name == "Phase Shift Identity"));
    }

    #[test]
    fn direct_derive_expands_repeated_phase_shift_pair_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse(
            "sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4)",
            &mut simplifier.context,
        )
        .expect("parse source");
        let target = cas_parser::parse("sin(x)+cos(x)+sin(y)+cos(y)", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
        assert_eq!(derived.1.len(), 2);
        assert!(derived
            .1
            .iter()
            .all(|step| step.rule_name == "Phase Shift Identity"));
    }

    #[test]
    fn direct_derive_rewrites_reverse_dirichlet_kernel_without_trig_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("sin(5*x/2)/sin(x/2)", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("1 + 2*cos(x) + 2*cos(2*x)", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::IntegratePrep);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Dirichlet Kernel Identity");
    }

    #[test]
    fn bounded_multistage_derive_reaches_log_contracted_grouped_power_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("ln(x^2)+ln(y^2)", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("ln((x*y)^2)", &mut simplifier.context).expect("parse target");

        let derived = try_bounded_multistage_derive(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
        )
        .expect("planner should derive grouped log-power target");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::Planner);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Log Contraction");
    }

    #[test]
    fn direct_derive_contracts_grouped_log_power_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("ln(x^2)+ln(y^2)", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("ln((x*y)^2)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::LogContract);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Log Contraction");
    }

    #[test]
    fn bounded_multistage_derive_reaches_abs_log_product_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("2*ln(abs(x))+2*ln(abs(y))", &mut simplifier.context)
            .expect("parse source");
        let target =
            cas_parser::parse("2*ln(abs(x*y))", &mut simplifier.context).expect("parse target");

        let derived = try_bounded_multistage_derive(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
        )
        .expect("planner should derive abs log product target");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::Planner);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Log Contraction");
    }

    #[test]
    fn direct_derive_contracts_abs_log_product_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("2*ln(abs(x))+2*ln(abs(y))", &mut simplifier.context)
            .expect("parse source");
        let target =
            cas_parser::parse("2*ln(abs(x*y))", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::LogContract);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Log Contraction");
    }

    #[test]
    fn bounded_multistage_derive_reaches_general_base_log_grouped_power_target() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("2*log(b,x)+2*log(b,y)", &mut simplifier.context)
            .expect("parse source");
        let target =
            cas_parser::parse("log(b,(x*y)^2)", &mut simplifier.context).expect("parse target");

        let derived = try_bounded_multistage_derive(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
        )
        .expect("planner should derive grouped general-base log target");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::Planner);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Log Contraction");
    }

    #[test]
    fn direct_derive_contracts_grouped_general_base_log_power_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("2*log(b,x)+2*log(b,y)", &mut simplifier.context)
            .expect("parse source");
        let target =
            cas_parser::parse("log(b,(x*y)^2)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::LogContract);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Log Contraction");
    }

    #[test]
    fn direct_derive_contracts_change_of_base_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("ln(x)/ln(2)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("log(2, x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::LogContract);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Change of Base");
        let substeps = derived.1[0].substeps();
        assert_eq!(substeps.len(), 3);
        assert_eq!(substeps[0].title, "Leer el argumento desde el numerador");
        assert!(substeps[0].lines.iter().any(|line| line.contains("ln(x)")));
        assert_eq!(substeps[1].title, "Leer la base desde el denominador");
        assert!(substeps[1].lines.iter().any(|line| line.contains("ln(2)")));
        assert_eq!(
            substeps[2].title,
            "Reconstruir el logaritmo de base indicada"
        );
    }

    #[test]
    fn direct_derive_contracts_grouped_log_power_with_passthrough_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("ln(x^2)+ln(y^2)+a", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("ln((x*y)^2)+a", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::LogContract);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Log Contraction");
    }

    #[test]
    fn direct_derive_contracts_abs_log_product_with_passthrough_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("2*ln(abs(x))+2*ln(abs(y))+a", &mut simplifier.context)
            .expect("parse source");
        let target =
            cas_parser::parse("2*ln(abs(x*y))+a", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::LogContract);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Log Contraction");
    }

    #[test]
    fn direct_derive_contracts_grouped_general_base_log_power_with_passthrough_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("2*log(b,x)+2*log(b,y)+a", &mut simplifier.context)
            .expect("parse source");
        let target =
            cas_parser::parse("log(b,(x*y)^2)+a", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::LogContract);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Log Contraction");
    }

    #[test]
    fn direct_derive_expands_grouped_log_power_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("ln((x*y)^2)", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("ln(x^2)+ln(y^2)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::LogExpand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "expand_log");
    }

    #[test]
    fn direct_derive_expands_grouped_abs_log_product_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("2*ln(abs(x*y))", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("2*ln(abs(x))+2*ln(abs(y))", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::LogExpand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "expand_log");
    }

    #[test]
    fn direct_derive_expands_grouped_general_base_log_power_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("log(b,(x*y)^2)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("2*log(b,x)+2*log(b,y)", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::LogExpand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "expand_log");
    }

    #[test]
    fn direct_derive_expands_change_of_base_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("log(2, x)", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("ln(x)/ln(2)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::LogExpand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Change of Base");
        let substeps = derived.1[0].substeps();
        assert_eq!(substeps.len(), 3);
        assert_eq!(substeps[0].title, "Poner el argumento en el numerador");
        assert!(substeps[0].lines.iter().any(|line| line.contains("ln(x)")));
        assert_eq!(substeps[1].title, "Poner la base en el denominador");
        assert!(substeps[1].lines.iter().any(|line| line.contains("ln(2)")));
        assert_eq!(substeps[2].title, "Formar el cociente de cambio de base");
    }

    #[test]
    fn direct_derive_expands_change_of_base_chain_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("log(b, c)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("log(b, a)*log(a, c)", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::LogExpand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Change of Base");
        assert!(derived.1[0].substeps().is_empty());
    }

    #[test]
    fn direct_derive_expands_log_product_then_simplifies_log_root() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("ln(sqrt(x)*y)", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("ln(x)/2 + ln(y)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::LogExpand);
        assert_eq!(derived.1.len(), 2);
        assert_eq!(derived.1[0].rule_name, "expand_log");
        assert_eq!(derived.1[1].rule_name, "Evaluate Logarithms");
    }

    #[test]
    fn direct_derive_expands_grouped_log_power_with_passthrough_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("ln((x*y)^2)+a", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("ln(x^2)+ln(y^2)+a", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::LogExpand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "expand_log");
    }

    #[test]
    fn direct_derive_expands_grouped_abs_log_product_with_passthrough_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("2*ln(abs(x*y))+a", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("2*ln(abs(x))+2*ln(abs(y))+a", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::LogExpand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "expand_log");
    }

    #[test]
    fn direct_derive_expands_grouped_general_base_log_power_with_passthrough_without_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("log(b,(x*y)^2)+a", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("2*log(b,x)+2*log(b,y)+a", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::LogExpand);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "expand_log");
    }

    #[test]
    fn direct_derive_factors_log_argument_before_expanding_without_generic_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("log(x^2-y^2)", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("log(x-y)+log(x+y)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::LogExpand);
        assert_eq!(derived.1.len(), 2);
        assert_eq!(derived.1[0].rule_name, "Factorization");
        assert_eq!(derived.1[1].rule_name, "expand_log");
    }

    #[test]
    fn direct_derive_factors_log_quotient_argument_before_expanding_without_generic_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("log((x^2-y^2)/(u*v))", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("log(x-y)+log(x+y)-log(u)-log(v)", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::LogExpand);
        assert_eq!(derived.1.len(), 2);
        assert_eq!(derived.1[0].rule_name, "Factorization");
        assert_eq!(derived.1[1].rule_name, "expand_log");
    }

    #[test]
    fn prefers_direct_product_to_sum_expansion_over_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("2*sin(2*x)*cos(x)", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("sin(3*x)+sin(x)", &mut simplifier.context).expect("parse target");

        let planner = try_bounded_multistage_derive(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
        )
        .expect("planner should derive product-to-sum target");

        assert_eq!(planner.2, DeriveStrategy::Planner);
        assert_eq!(planner.0, target);
        assert!(!planner.1.is_empty());

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Product-to-Sum Identity");
    }

    #[test]
    fn prefers_direct_cosine_product_to_sum_expansion_over_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("2*cos(2*x)*cos(x)", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("cos(3*x)+cos(x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Product-to-Sum Identity");
    }

    #[test]
    fn prefers_direct_sine_product_to_sum_difference_expansion_over_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("2*sin(2*x)*sin(x)", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("cos(x)-cos(3*x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Product-to-Sum Identity");
    }

    #[test]
    fn prefers_trig_expand_targeted_additive_triple_angle_chain_over_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("2*sin(2*x)*sin(x)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("cos(x)-4*cos(x)^3+3*cos(x)", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 2);
        assert_eq!(derived.1[0].rule_name, "Product-to-Sum Identity");
        assert_eq!(derived.1[1].rule_name, "Triple Angle Expansion");
    }

    #[test]
    fn prefers_trig_expand_combined_additive_triple_angle_chain_for_cosine_sum_polynomial() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("2*cos(2*x)*cos(x)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("4*cos(x)^3-2*cos(x)", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 2);
        assert_eq!(derived.1[0].rule_name, "Product-to-Sum Identity");
        assert_eq!(derived.1[1].rule_name, "Triple Angle Expansion");
    }

    #[test]
    fn prefers_trig_expand_combined_additive_triple_angle_chain_for_cosine_difference_polynomial() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("2*sin(2*x)*sin(x)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("4*cos(x)-4*cos(x)^3", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 2);
        assert_eq!(derived.1[0].rule_name, "Product-to-Sum Identity");
        assert_eq!(derived.1[1].rule_name, "Triple Angle Expansion");
    }

    #[test]
    fn prefers_trig_expand_combined_additive_triple_angle_chain_for_sine_difference_mixed_polynomial(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("2*cos(2*x)*sin(x)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("4*cos(x)^2*sin(x)-2*sin(x)", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 2);
        assert_eq!(derived.1[0].rule_name, "Product-to-Sum Identity");
        assert_eq!(derived.1[1].rule_name, "Triple Angle Expansion");
    }

    #[test]
    fn prefers_trig_expand_product_to_sum_triple_angle_chain_with_passthrough_term() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("2*sin(2*x)*sin(x)+a", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("4*cos(x)-4*cos(x)^3+a", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 2);
        assert_eq!(derived.1[0].rule_name, "Product-to-Sum Identity");
        assert_eq!(derived.1[1].rule_name, "Triple Angle Expansion");
    }

    #[test]
    fn prefers_trig_expand_mixed_product_to_sum_triple_angle_chain_with_passthrough_term() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("2*cos(2*x)*sin(x)+a", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("4*cos(x)^2*sin(x)-2*sin(x)+a", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 2);
        assert_eq!(derived.1[0].rule_name, "Product-to-Sum Identity");
        assert_eq!(derived.1[1].rule_name, "Triple Angle Expansion");
    }

    #[test]
    fn prefers_direct_mixed_cosine_difference_double_angle_expansion_over_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("2*sin(2*x)*sin(x)", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("4*sin(x)^2*cos(x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Double Angle Expansion");
    }

    #[test]
    fn direct_derive_expands_inverse_trig_double_angle_projections() {
        for (source, target) in [
            ("sin(2*arcsin(x))", "2*x*sqrt(1-x^2)"),
            ("cos(2*arcsin(x))", "1-2*x^2"),
            ("sin(2*arccos(x))", "2*x*sqrt(1-x^2)"),
            ("cos(2*arccos(x))", "2*x^2-1"),
            ("sin(2*arctan(x))", "2*x/(1+x^2)"),
            ("cos(2*arctan(x))", "(1-x^2)/(1+x^2)"),
        ] {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let source = cas_parser::parse(source, &mut simplifier.context).expect("parse source");
            let target = cas_parser::parse(target, &mut simplifier.context).expect("parse target");

            let derived = try_supported_derive_strategies_inner(
                &mut simplifier,
                source,
                target,
                true,
                &crate::SimplifyOptions::default(),
                true,
                true,
            )
            .expect("derive should succeed");

            assert_eq!(derived.2, DeriveStrategy::TrigExpand);
            assert_eq!(derived.0, target);
            assert_eq!(derived.1.len(), 1);
            assert_eq!(derived.1[0].rule_name, "Double Angle Expansion");
        }
    }

    #[test]
    fn direct_derive_expands_simplified_argument_half_angle_tangent_variants() {
        for (source, target) in [
            ("tan(x/2)", "sin(x)/(1+cos(x))"),
            ("tan(x/2)", "(1-cos(x))/sin(x)"),
        ] {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let source = cas_parser::parse(source, &mut simplifier.context).expect("parse source");
            let target = cas_parser::parse(target, &mut simplifier.context).expect("parse target");

            let derived = try_supported_derive_strategies_inner(
                &mut simplifier,
                source,
                target,
                true,
                &crate::SimplifyOptions::default(),
                true,
                true,
            )
            .expect("derive should succeed");

            assert_eq!(derived.2, DeriveStrategy::TrigExpand);
            assert_eq!(derived.0, target);
            assert_eq!(derived.1.len(), 1);
            assert_eq!(derived.1[0].rule_name, "Half-Angle Tangent Identity");
        }
    }

    #[test]
    fn direct_derive_expands_tangent_half_angle_substitution_without_generic_simplify() {
        for (source, target) in [
            ("sin(x)", "2*tan(x/2)/(1+tan(x/2)^2)"),
            ("cos(x)", "(1-tan(x/2)^2)/(1+tan(x/2)^2)"),
        ] {
            let mut simplifier = crate::Simplifier::with_default_rules();
            let source = cas_parser::parse(source, &mut simplifier.context).expect("parse source");
            let target = cas_parser::parse(target, &mut simplifier.context).expect("parse target");

            let derived = try_supported_derive_strategies_inner(
                &mut simplifier,
                source,
                target,
                true,
                &crate::SimplifyOptions::default(),
                true,
                true,
            )
            .expect("derive should succeed");

            assert_eq!(derived.2, DeriveStrategy::TrigExpand);
            assert_eq!(derived.0, target);
            assert_eq!(derived.1.len(), 1);
            assert_eq!(derived.1[0].rule_name, "Half-Angle Tangent Identity");
        }
    }

    #[test]
    fn bounded_multistage_derive_reaches_split_exponential_products_from_hyperbolic_bridge_node() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("sinh(x+2*x)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("(e^x*e^(2*x)-e^(-x)*e^(-2*x))/2", &mut simplifier.context)
            .expect("parse target");

        let derived = try_bounded_multistage_derive(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
        )
        .expect("planner should derive target from bridge node");

        assert_eq!(derived.0, target);
        assert_eq!(derived.2, DeriveStrategy::Planner);
        assert!(!derived.1.is_empty());
    }

    #[test]
    fn prefers_expand_for_hyperbolic_product_to_sum_triple_angle_polynomial() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("2*sinh(2*x)*cosh(x)", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("4*sinh(x)+4*sinh(x)^3", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::Expand);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Product-to-Sum Identity");
    }

    #[test]
    fn prefers_direct_hyperbolic_product_to_sum_expansion_over_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("2*sinh(2*x)*cosh(x)", &mut simplifier.context)
            .expect("parse source");
        let target =
            cas_parser::parse("sinh(3*x)+sinh(x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::Expand);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Product-to-Sum Identity");
    }

    #[test]
    fn prefers_direct_hyperbolic_sum_to_product_contraction_over_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("sinh(3*x)-sinh(x)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("2*cosh(2*x)*sinh(x)", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::Expand);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Product-to-Sum Identity");
    }

    #[test]
    fn prefers_direct_exact_hyperbolic_sum_to_product_xy_over_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("sinh(x)+sinh(y)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("2*sinh((x+y)/2)*cosh((x-y)/2)", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::Expand);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Product-to-Sum Identity");
    }

    #[test]
    fn prefers_expand_for_hyperbolic_cosh_product_to_sum_triple_angle_polynomial() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("2*sinh(2*x)*sinh(x)", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("4*cosh(x)^3-4*cosh(x)", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::Expand);
        assert_eq!(derived.0, target);
        // Soundness fix (cosh(3x)−cosh(x) no longer collapses to 0) lets the
        // simplifier normalize the product-to-sum output straight to the cubic
        // polynomial in one pass, so the standalone case now lands in a single
        // expand step. (The passthrough variant below still surfaces the two
        // discrete steps.) Tracked as a follow-up: restore the explicit
        // triple-angle step in this bare derive narrative.
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Product-to-Sum Identity");
    }

    #[test]
    fn prefers_expand_for_hyperbolic_product_to_sum_polynomial_with_passthrough_term() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("2*sinh(2*x)*sinh(x)+a", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("4*cosh(x)^3-4*cosh(x)+a", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::Expand);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 2);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Product-to-Sum Identity");
        assert_eq!(derived.1[1].rule_name, "Hyperbolic Triple-Angle Identity");
    }

    #[test]
    fn prefers_direct_hyperbolic_cosh_double_angle_expansion_to_sinh_cubic_polynomial() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("2*cosh(2*x)*sinh(x)", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("2*sinh(x)+4*sinh(x)^3", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::HyperbolicRewrite);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Double-Angle Identity");
    }

    #[test]
    fn prefers_direct_hyperbolic_cosh_double_angle_expansion_to_sinh_mixed_polynomial() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("2*cosh(2*x)*sinh(x)", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("4*cosh(x)^2*sinh(x)-2*sinh(x)", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::HyperbolicRewrite);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Double-Angle Identity");
    }

    #[test]
    fn prefers_direct_hyperbolic_cosh_double_angle_contraction_from_sinh_mixed_polynomial() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("4*cosh(x)^2*sinh(x)-2*sinh(x)", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("2*cosh(2*x)*sinh(x)", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::HyperbolicRewrite);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Double-Angle Identity");
    }

    #[test]
    fn prefers_direct_hyperbolic_cosh_double_angle_contraction_from_cosh_mixed_polynomial() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("2*cosh(x)+4*sinh(x)^2*cosh(x)", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("2*cosh(2*x)*cosh(x)", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::HyperbolicRewrite);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Double-Angle Identity");
    }

    #[test]
    fn prefers_direct_hyperbolic_cosh_double_angle_expansion_to_cosh_mixed_polynomial() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("2*cosh(2*x)*cosh(x)", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("2*cosh(x)+4*sinh(x)^2*cosh(x)", &mut simplifier.context)
            .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::HyperbolicRewrite);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Double-Angle Identity");
    }

    #[test]
    fn fast_direct_hyperbolic_derive_rejects_pure_polynomial_pair() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("(a+b+c)^3", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse(
            "a^3 + b^3 + c^3 + 3*a^2*b + 3*a^2*c + 3*a*b^2 + 6*a*b*c + 3*a*c^2 + 3*b^2*c + 3*b*c^2",
            &mut simplifier.context,
        )
        .expect("parse target");

        assert!(
            try_fast_direct_hyperbolic_derive(&mut simplifier, source, target, true).is_none(),
            "pure polynomial derive should not enter the fast hyperbolic path"
        );
    }

    #[test]
    fn prefers_direct_scaled_cosh_from_exponential_sum() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("exp(2*x)+exp(-2*x)", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("2*cosh(2*x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::HyperbolicRewrite);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Exponential Identity");
    }

    #[test]
    fn prefers_direct_scaled_cosh_from_exponential_reciprocal_pair() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("exp(x)+1/exp(x)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("2*cosh(x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::HyperbolicRewrite);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Exponential Identity");
    }

    #[test]
    fn prefers_direct_negative_hyperbolic_decomposition_to_exp_neg_over_planner() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("cosh(x)-sinh(x)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse("exp(-x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::HyperbolicRewrite);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Exponential Identity");
    }

    #[test]
    fn prefers_direct_scaled_cosh_to_exponential_sum() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("2*cosh(2*x)", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("exp(2*x)+exp(-2*x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::HyperbolicRewrite);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Exponential Identity");
    }

    #[test]
    fn prefers_direct_scaled_sinh_to_exponential_difference() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("2*sinh(x)", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("exp(x)-exp(-x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::HyperbolicRewrite);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Exponential Identity");
    }

    #[test]
    fn prefers_direct_presentational_hyperbolic_reciprocal_ratio_bridge_over_semantic_match() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("tanh(x)", &mut simplifier.context).expect("parse source");
        let target = cas_parser::parse(
            "(exp(x)-1/exp(x))/(exp(x)+1/exp(x))",
            &mut simplifier.context,
        )
        .expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::HyperbolicRewrite);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Exponential Identity");
    }

    #[test]
    fn prefers_direct_scaled_cosh_reciprocal_exponential_bridge() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("2*cosh(x)", &mut simplifier.context).expect("parse source");
        let target =
            cas_parser::parse("exp(x)+1/exp(x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::HyperbolicRewrite);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Exponential Identity");
    }

    #[test]
    fn prefers_direct_half_cosh_reciprocal_exponential_recognition_bridge() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = cas_parser::parse("(exp(x)+1/exp(x))/2", &mut simplifier.context)
            .expect("parse source");
        let target = cas_parser::parse("cosh(x)", &mut simplifier.context).expect("parse target");

        let derived = try_supported_derive_strategies_inner(
            &mut simplifier,
            source,
            target,
            true,
            &crate::SimplifyOptions::default(),
            true,
            true,
        )
        .expect("derive should succeed");

        assert_eq!(derived.2, DeriveStrategy::HyperbolicRewrite);
        assert_eq!(derived.0, target);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Exponential Identity");
    }
}
