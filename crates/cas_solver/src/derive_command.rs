use crate::derive::{
    classify_target_profile, extract_factored_division_target,
    generate_hyperbolic_additive_term_bridge_rewrites, generate_hyperbolic_bridge_rewrites,
    generate_trig_additive_term_bridge_rewrites, generate_trig_bridge_rewrites,
    looks_like_factored_target, ordered_strategies_for_target, phase_shift_target_match,
    presentational_target_match, run_combine_like_terms_rewrite,
    should_try_hyperbolic_planner_before_simplify, should_try_trig_planner_before_simplify,
    strong_target_match, try_build_combined_fraction_from_fold_add,
    try_rewrite_collect_monomial_target_aware,
    try_rewrite_consecutive_factorial_ratio_target_aware,
    try_rewrite_exact_fraction_cancel_target_aware, try_rewrite_expanded_target_aware,
    try_rewrite_exponential_sum_diff_target_aware, try_rewrite_fraction_combination_target_aware,
    try_rewrite_fraction_expansion_target_aware, try_rewrite_hyperbolic_simplify_target_aware,
    try_rewrite_integrate_prep_target_aware, try_rewrite_log_argument_factorization_target_aware,
    try_rewrite_log_contraction_target_aware, try_rewrite_log_contraction_to_target_aware,
    try_rewrite_log_expansion_target_aware, try_rewrite_log_simplify_target_aware,
    try_rewrite_nested_fraction_target_aware, try_rewrite_odd_half_power_to_target_aware,
    try_rewrite_power_merge_target_aware, try_rewrite_pythagorean_factor_form_target_aware,
    try_rewrite_radical_target_aware, try_rewrite_rationalized_target_aware,
    try_rewrite_shifted_double_angle_target_aware,
    try_rewrite_shifted_reciprocal_pythagorean_target_aware, try_rewrite_solve_prep_target_aware,
    try_rewrite_trig_contraction_target_aware, try_rewrite_trig_expansion,
    try_rewrite_trig_identity_to_one_target_aware, DeriveHyperbolicRewriteKind, DeriveStrategy,
    DeriveTargetForm, ExpandRewriteKind, RationalizeRewriteKind,
};

use cas_ast::{Expr, ExprId};
use cas_engine::NormalFormGoal;
use cas_math::inverse_trig_composition_support::try_plan_inverse_atan_reciprocal_add_expr;
use cas_math::summation_support::{
    try_plan_finite_product_evaluation, try_plan_finite_sum_evaluation, FiniteAggregateCall,
    ProductEvaluationKind, SumEvaluationKind,
};
use cas_solver_core::engine_event_collector::EngineEventCollector;
use cas_solver_core::engine_events::EngineEvent;
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

#[derive(Debug, Clone)]
struct DeriveEvalOutput {
    resolved_expr: ExprId,
    target_expr: ExprId,
    derived_expr: ExprId,
    steps: Vec<crate::Step>,
    status: DeriveStatus,
}

/// Format `derive` parse/resolve errors for user-facing output.
pub fn format_derive_eval_error_message(error: &DeriveEvalError) -> String {
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
    let planner_context_snapshot = allow_planner.then(|| simplifier.context.clone());
    let profile = classify_target_profile(&mut simplifier.context, resolved_expr, target_expr);

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
    let allow_planner_preference = allow_planner
        && (should_try_trig_planner_before_simplify(
            &mut simplifier.context,
            resolved_expr,
            target_expr,
        ) || should_try_hyperbolic_planner_before_simplify(
            &mut simplifier.context,
            resolved_expr,
            target_expr,
        ));

    if !prefer_simplify_first
        && !matches!(
            profile.form,
            DeriveTargetForm::FactoredWithDivision { .. }
                | DeriveTargetForm::FractionExpanded
                | DeriveTargetForm::FiniteAggregateEvaluated
        )
        && looks_like_factored_target(&mut simplifier.context, target_expr)
    {
        let stage = run_factored_stage(
            simplifier,
            resolved_expr,
            collect_steps,
            simplify_options.clone(),
        );
        if derive_target_match(simplifier, stage.expr, target_expr)
            || derive_semantic_match(simplifier, stage.expr, target_expr)
        {
            let mut stage = stage;
            retarget_stage_output(&mut stage, target_expr);
            let steps =
                finalize_steps(resolved_expr, target_expr, vec![stage], &simplifier.context);
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
            let steps =
                finalize_steps(resolved_expr, target_expr, vec![stage], &simplifier.context);
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

    for strategy in ordered_strategies_for_target(&profile) {
        match strategy {
            DeriveStrategy::Planner => continue,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                    if derive_target_match(simplifier, stage.expr, target_expr) {
                        let mut stage = stage;
                        retarget_stage_output(&mut stage, target_expr);
                        let steps = finalize_steps(
                            resolved_expr,
                            target_expr,
                            vec![first_stage, stage],
                            &simplifier.context,
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
                if derive_target_match(simplifier, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::LogExpand));
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::SimplifyThenTrigContract));
                }
            }
            DeriveStrategy::Rationalize => {
                let stage = rationalize_stage.get_or_insert_with(|| {
                    run_rationalize_stage(simplifier, resolved_expr, target_expr, collect_steps)
                });
                if derive_target_match(simplifier, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![stage],
                        &simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::Rationalize));
                }

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
                        &simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::Rationalize));
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
                    );
                    return Some((target_expr, steps, DeriveStrategy::Expand));
                }
            }
            DeriveStrategy::Factor => {
                let stage = factor_stage.get_or_insert_with(|| {
                    run_factored_stage(
                        simplifier,
                        resolved_expr,
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
                        &simplifier.context,
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
                    if strong_target_match(&mut simplifier.context, stage.expr, target_expr) {
                        let mut stage = stage;
                        retarget_stage_output(&mut stage, target_expr);
                        let steps = finalize_steps(
                            resolved_expr,
                            target_expr,
                            vec![first_stage, stage],
                            &simplifier.context,
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
                if strong_target_match(&mut simplifier.context, stage.expr, target_expr) {
                    let mut stage = stage.clone();
                    retarget_stage_output(&mut stage, target_expr);
                    let steps = finalize_steps(
                        resolved_expr,
                        target_expr,
                        vec![first_stage.clone(), stage],
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
                        &simplifier.context,
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
        let steps = finalize_steps(resolved_expr, target_expr, vec![stage], &simplifier.context);
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

    let steps = finalize_planner_steps(resolved_expr, target_expr, stages, &simplifier.context);
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

        if derive_target_match(simplifier, stage.expr, target_expr) {
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
        if let Some(focus_before) = rewrite.focus_before {
            step.meta_mut().before_local = Some(focus_before);
        }
        if let Some(focus_after) = rewrite.focus_after {
            step.meta_mut().after_local = Some(focus_after);
        } else if rewrite.intermediate != rewrite.rewritten {
            step.meta_mut().after_local = Some(rewrite.intermediate);
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
        if let Some(focus_before) = rewrite.focus_before {
            step.meta_mut().before_local = Some(focus_before);
        }
        if let Some(focus_after) = rewrite.focus_after {
            step.meta_mut().after_local = Some(focus_after);
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
    let Some(plan) =
        try_plan_inverse_atan_reciprocal_add_expr(&mut simplifier.context, expr, false)
    else {
        return DeriveStageOutput {
            expr,
            steps: Vec::new(),
        };
    };

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
            return DeriveStageOutput {
                expr,
                steps: Vec::new(),
            };
        }
        simplified
    };

    let steps = if collect_steps {
        let mut step = crate::Step::with_snapshots(
            &plan.desc,
            "Inverse Tan Relations",
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

    DeriveStageOutput {
        expr: rewritten,
        steps,
    }
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
            before_local,
            after_local,
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
    _collect_steps: bool,
    _simplify_options: crate::SimplifyOptions,
) -> DeriveStageOutput {
    let factored_expr = cas_math::factor::factor(&mut simplifier.context, expr);
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
    ctx: &cas_ast::Context,
) -> Vec<crate::Step> {
    let stage_count = stages.len();
    let cleaned = prune_semantically_noop_derive_steps(
        prune_adjacent_inverse_derive_steps(clean_derive_stage_steps(stages), ctx),
        ctx,
    );

    let mut steps = if stage_count > 1 {
        cleaned
    } else {
        match cas_solver_core::step_optimization_runtime::optimize_steps_semantic(
            cleaned.clone(),
            ctx,
            original_expr,
            final_expr,
        ) {
            cas_solver_core::step_optimization_runtime::StepOptimizationResult::Steps(steps) => {
                steps
            }
            cas_solver_core::step_optimization_runtime::StepOptimizationResult::NoSimplificationNeeded => {
                cleaned
            }
        }
    };
    steps = prune_adjacent_inverse_derive_steps(steps, ctx);
    steps = prune_semantically_noop_derive_steps(steps, ctx);

    if stage_count > 1 {
        truncate_steps_after_first_presentational_target_match(&mut steps, final_expr, ctx);
    } else {
        truncate_steps_after_first_target_match(&mut steps, final_expr, ctx);
    }

    if let Some(last_step) = steps.last_mut() {
        last_step.global_after = Some(final_expr);
    }

    steps
}

fn finalize_planner_steps(
    _original_expr: ExprId,
    final_expr: ExprId,
    stages: Vec<DeriveStageOutput>,
    ctx: &cas_ast::Context,
) -> Vec<crate::Step> {
    let mut steps = prune_semantically_noop_derive_steps(
        prune_adjacent_inverse_derive_steps(clean_derive_stage_steps(stages), ctx),
        ctx,
    );
    truncate_steps_after_first_presentational_target_match(&mut steps, final_expr, ctx);

    if let Some(last_step) = steps.last_mut() {
        last_step.global_after = Some(final_expr);
    }

    steps
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
            retarget_final_after_line(&mut lines, &target);
            lines.insert(1, format!("Target: {target}"));
            lines.insert(2, format!("Strategy: {}", strategy.label()));
            append_derive_requires_lines(&mut lines, ctx, output.derived_expr);
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

fn retarget_final_after_line(lines: &mut [String], target: &str) {
    if let Some(index) = lines
        .iter()
        .rposition(|line| line.trim_start().starts_with("After:"))
    {
        lines[index] = format!("   After: {target}");
    }
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
        SumEvaluationKind::FiniteDirect { start, end } => format!(
            "sum({}, {}, {}, {})",
            render_expr(call.term),
            call.var_name,
            start,
            end
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
        ProductEvaluationKind::FiniteDirect { start, end } => format!(
            "product({}, {}, {}, {})",
            render_expr(call.term),
            call.var_name,
            start,
            end
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
    derived_expr: ExprId,
) {
    let domain = crate::infer_implicit_domain(ctx, derived_expr, crate::ValueDomain::RealOnly);
    let conditions: Vec<_> = domain.conditions().iter().cloned().collect();
    if conditions.is_empty() {
        return;
    }
    let rendered =
        cas_solver_core::domain_normalization::render_conditions_normalized(ctx, &conditions);
    if rendered.is_empty() {
        return;
    }
    lines.push("ℹ️ Requires:".to_string());
    lines.extend(rendered.into_iter().map(|line| format!("  • {line}")));
}

#[cfg(test)]
mod tests {
    use super::{
        derive_semantic_match, generate_planner_candidate_stages, try_bounded_multistage_derive,
        try_supported_derive_strategies_inner, DeriveStrategy,
    };

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
        assert_eq!(derived.1.len(), 1);
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
    fn direct_derive_rewrites_reciprocal_trig_product_with_passthrough_without_simplify() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source =
            cas_parser::parse("tan(x)*cot(x)+a", &mut simplifier.context).expect("parse source");
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
        assert_eq!(derived.2, DeriveStrategy::TrigRewrite);
        assert_eq!(derived.1.len(), 1);
        assert_eq!(derived.1[0].rule_name, "Reciprocal Product Identity");
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
        assert_eq!(derived.2, DeriveStrategy::TrigExpand);
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
        assert_eq!(derived.1.len(), 2);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Product-to-Sum Identity");
        assert_eq!(derived.1[1].rule_name, "Hyperbolic Triple-Angle Identity");
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
        assert_eq!(derived.1.len(), 2);
        assert_eq!(derived.1[0].rule_name, "Hyperbolic Product-to-Sum Identity");
        assert_eq!(derived.1[1].rule_name, "Hyperbolic Triple-Angle Identity");
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
