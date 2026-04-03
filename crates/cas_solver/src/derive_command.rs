use crate::derive::{
    classify_target_profile, extract_factored_division_target, ordered_strategies_for_target,
    presentational_target_match, strong_target_match, try_build_combined_fraction_from_fold_add,
    try_rewrite_collect_monomial_target_aware, try_rewrite_exact_fraction_cancel_target_aware,
    try_rewrite_expanded_target_aware, try_rewrite_fraction_combination_target_aware,
    try_rewrite_fraction_expansion_target_aware, try_rewrite_integrate_prep_target_aware,
    try_rewrite_log_contraction_target_aware, try_rewrite_log_expansion_target_aware,
    try_rewrite_odd_half_power_target_aware, try_rewrite_power_merge_target_aware,
    try_rewrite_pythagorean_factor_form_target_aware, try_rewrite_solve_prep_target_aware,
    try_rewrite_trig_contraction_target_aware, try_rewrite_trig_expansion,
    try_rewrite_trig_identity_to_one_target_aware, DeriveStrategy, DeriveTargetForm,
    ExpandRewriteKind,
};

use cas_ast::{Expr, ExprId};
use cas_engine::NormalFormGoal;
use cas_math::summation_support::{
    try_plan_finite_sum_evaluation, FiniteAggregateCall, SumEvaluationKind,
};
use cas_solver_core::engine_event_collector::EngineEventCollector;
use cas_solver_core::engine_events::EngineEvent;

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
        DeriveStrategy::TrigContract => (
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
                    "Derive unavailable: cannot prove the two expressions are equivalent."
                }
                crate::EquivalenceResult::True
                | crate::EquivalenceResult::ConditionalTrue { .. } => "Derive unavailable.",
            };
            return Err(detail.to_string());
        }
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
    simplify_options: crate::SimplifyOptions,
) -> DeriveEvalOutput {
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
    ) {
        let steps = if collect_steps
            && steps.is_empty()
            && !presentational_target_match(&mut simplifier.context, resolved_expr, derived_expr)
        {
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
}

fn try_supported_derive_strategies_inner(
    simplifier: &mut crate::Simplifier,
    resolved_expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
    simplify_options: &crate::SimplifyOptions,
    allow_factor_with_division: bool,
) -> Option<(ExprId, Vec<crate::Step>, DeriveStrategy)> {
    let profile = classify_target_profile(&mut simplifier.context, resolved_expr, target_expr);

    let mut simplify_stage: Option<DeriveStageOutput> = None;
    let mut integrate_prep_stage: Option<DeriveStageOutput> = None;
    let mut solve_prep_stage: Option<DeriveStageOutput> = None;
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
    let mut simplify_then_factor_with_division_stage: Option<Option<DeriveStageOutput>> = None;
    let mut simplify_then_odd_half_power_stage: Option<DeriveStageOutput> = None;
    let mut simplify_then_expand_stage: Option<DeriveStageOutput> = None;
    let mut simplify_then_collect_stage: Option<Option<DeriveStageOutput>> = None;
    let mut simplify_then_factor_stage: Option<DeriveStageOutput> = None;
    let prefer_simplify_first = is_finite_aggregate_source(&simplifier.context, resolved_expr);

    if !prefer_simplify_first
        && !matches!(
            profile.form,
            DeriveTargetForm::FactoredWithDivision { .. } | DeriveTargetForm::FractionExpanded
        )
        && looks_like_explicit_factored_target(&mut simplifier.context, target_expr)
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
            return Some((target_expr, steps, DeriveStrategy::Simplify));
        }
    }

    for strategy in ordered_strategies_for_target(&profile) {
        match strategy {
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
            DeriveStrategy::LogExpand => {
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
                    run_log_contract_stage(simplifier, resolved_expr, collect_steps)
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
                    run_log_contract_stage(simplifier, first_stage.expr, collect_steps)
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
                if strong_target_match(&mut simplifier.context, direct_stage.expr, target_expr) {
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
                    return Some((target_expr, steps, DeriveStrategy::TrigContract));
                }
            }
            DeriveStrategy::Rationalize => {
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
                    run_odd_half_power_stage(simplifier, resolved_expr, collect_steps)
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
                if !matches!(profile.form, DeriveTargetForm::Expanded) {
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
                    run_odd_half_power_stage(simplifier, first_stage.expr, collect_steps)
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

    None
}

fn run_expand_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    if let Some(rewrite) =
        try_rewrite_expanded_target_aware(&mut simplifier.context, expr, target_expr)
    {
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

fn build_expand_step(
    ctx: &cas_ast::Context,
    before: ExprId,
    after: ExprId,
    kind: ExpandRewriteKind,
) -> crate::Step {
    let (description, rule_name) = match kind {
        ExpandRewriteKind::BinomialPower => ("Expand the binomial power", "Binomial Expansion"),
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
        if let Some(canonicalized) = rewrite.canonicalized {
            let mut root_step = crate::Step::with_snapshots(
                "sqrt(x) = x^(1/2)",
                "Canonicalize Roots",
                expr,
                canonicalized,
                Vec::new(),
                Some(&simplifier.context),
                expr,
                canonicalized,
            );
            root_step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
            root_step.category = cas_solver_core::step_types::StepCategory::Simplify;
            steps.push(root_step);

            let mut merge_step = crate::Step::with_snapshots(
                "Combine powers with same base (n-ary)",
                "Combine powers with same base (n-ary)",
                canonicalized,
                rewrite.rewritten,
                Vec::new(),
                Some(&simplifier.context),
                canonicalized,
                rewrite.rewritten,
            );
            merge_step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
            merge_step.category = cas_solver_core::step_types::StepCategory::Simplify;
            steps.push(merge_step);
        } else {
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
    collect_steps: bool,
) -> DeriveStageOutput {
    let rewritten_source = if let Some(rewritten) =
        try_rewrite_odd_half_power_target_aware(&mut simplifier.context, expr)
    {
        rewritten
    } else {
        let (silently_simplified, _steps, _stats) = simplifier.simplify_with_stats(
            expr,
            crate::SimplifyOptions {
                collect_steps: false,
                ..crate::SimplifyOptions::default()
            },
        );
        let Some(rewritten) =
            try_rewrite_odd_half_power_target_aware(&mut simplifier.context, silently_simplified)
        else {
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
    let Some(rewrite) = try_rewrite_trig_expansion(&mut simplifier.context, expr, target_expr)
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
        vec![step]
    } else {
        Vec::new()
    };

    DeriveStageOutput {
        expr: rewrite.rewritten,
        steps,
    }
}

fn run_trig_contract_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    let Some(rewrite) =
        try_rewrite_trig_contraction_target_aware(&mut simplifier.context, expr, target_expr)
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
            plan.rewritten,
            Vec::new(),
            Some(&simplifier.context),
            expr,
            plan.rewritten,
        );
        step.importance = cas_solver_core::step_types::ImportanceLevel::Medium;
        step.category = cas_solver_core::step_types::StepCategory::Expand;
        step.meta_mut().assumption_events.extend(assumed_positive);
        steps.push(step);
    }
    steps.append(&mut cleanup.steps);

    DeriveStageOutput {
        expr: cleanup.expr,
        steps,
    }
}

fn run_log_contract_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    collect_steps: bool,
) -> DeriveStageOutput {
    let Some(rewritten) = try_rewrite_log_contraction_target_aware(&mut simplifier.context, expr)
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
        try_run_finite_summation_target_aware_stage(simplifier, expr, target_expr, collect_steps)
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
        try_rewrite_exact_fraction_cancel_target_aware(&mut simplifier.context, expr, target_expr)
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
            if rewrite.intermediate != rewrite.rewritten {
                step.meta_mut().after_local = Some(rewrite.intermediate);
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

    simplify_options.collect_steps = collect_steps;
    run_stage(simplifier, expr, collect_steps, simplify_options)
}

fn try_run_finite_summation_target_aware_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
) -> Option<DeriveStageOutput> {
    let plan = try_plan_finite_sum_evaluation(&mut simplifier.context, expr, 1000)?;
    let mut temp = crate::Simplifier::with_default_rules();
    temp.context = simplifier.context.clone();
    let (simplified, _) = temp.simplify(plan.candidate);
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
    let raw_steps: Vec<_> = stages.into_iter().flat_map(|stage| stage.steps).collect();
    let cleaned = cas_solver_core::eval_step_pipeline::clean_eval_steps(
        raw_steps,
        |s: &crate::Step| s.before,
        |s: &crate::Step| s.after,
        |s: &crate::Step| s.before_local(),
        |s: &crate::Step| s.after_local(),
        |s: &crate::Step| s.global_after,
        |s: &mut crate::Step, gb| s.global_before = Some(gb),
    );

    let mut steps = match cas_solver_core::step_optimization_runtime::optimize_steps_semantic(
        cleaned.clone(),
        ctx,
        original_expr,
        final_expr,
    ) {
        cas_solver_core::step_optimization_runtime::StepOptimizationResult::Steps(steps) => steps,
        cas_solver_core::step_optimization_runtime::StepOptimizationResult::NoSimplificationNeeded => {
            cleaned
        }
    };

    truncate_steps_after_first_target_match(&mut steps, final_expr, ctx);

    if let Some(last_step) = steps.last_mut() {
        last_step.global_after = Some(final_expr);
    }

    steps
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
                    "Derive unavailable: cannot prove the two expressions are equivalent."
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
    let (simplified, _steps, _stats) =
        temp.simplify_with_stats(difference, crate::SimplifyOptions::default());
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

fn looks_like_explicit_factored_target(ctx: &mut cas_ast::Context, target_expr: ExprId) -> bool {
    match ctx.get(target_expr) {
        Expr::Pow(base, exp) => {
            matches!(ctx.get(*base), Expr::Add(_, _) | Expr::Sub(_, _))
                && matches!(ctx.get(*exp), Expr::Number(n) if n.is_integer() && n.to_integer() >= 2.into())
        }
        _ => {
            if !ctx.is_mul_commutative(target_expr) {
                return false;
            }

            let factors = cas_math::trig_roots_flatten::flatten_mul_chain(ctx, target_expr);
            if factors.len() < 2 {
                return false;
            }

            let mut non_numeric_factors = 0usize;
            let mut has_additive_factor = false;

            for factor in factors {
                match ctx.get(factor) {
                    Expr::Number(_) | Expr::Constant(_) => {}
                    Expr::Add(_, _) | Expr::Sub(_, _) => {
                        non_numeric_factors += 1;
                        has_additive_factor = true;
                    }
                    Expr::Pow(base, exp)
                        if matches!(ctx.get(*base), Expr::Add(_, _) | Expr::Sub(_, _))
                            && matches!(ctx.get(*exp), Expr::Number(n) if n.is_integer() && n.to_integer() >= 2.into()) =>
                    {
                        non_numeric_factors += 1;
                        has_additive_factor = true;
                    }
                    _ => non_numeric_factors += 1,
                }
            }

            has_additive_factor && non_numeric_factors >= 2
        }
    }
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
