use std::cmp::Ordering;
use std::collections::BTreeSet;

use cas_ast::ExprId;
use cas_solver_core::engine_event_collector::EngineEventCollector;
use cas_solver_core::engine_events::EngineEvent;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeriveEvalError {
    Parse(crate::ParseExprPairError),
    Resolve(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeriveStrategy {
    Simplify,
    Collect,
    Factor,
    SimplifyThenCollect,
    SimplifyThenFactor,
}

impl DeriveStrategy {
    fn label(self) -> &'static str {
        match self {
            Self::Simplify => "simplify",
            Self::Collect => "collect",
            Self::Factor => "factor",
            Self::SimplifyThenCollect => "simplify -> collect",
            Self::SimplifyThenFactor => "simplify -> factor",
        }
    }
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
    if strong_target_match(&mut simplifier.context, resolved_expr, target_expr) {
        return DeriveEvalOutput {
            resolved_expr,
            target_expr,
            derived_expr: target_expr,
            steps: Vec::new(),
            status: DeriveStatus::AlreadyAtTarget,
        };
    }

    if let Some((derived_expr, steps, strategy)) = try_supported_derive_strategies(
        simplifier,
        resolved_expr,
        target_expr,
        collect_steps,
        &simplify_options,
    ) {
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

fn try_supported_derive_strategies(
    simplifier: &mut crate::Simplifier,
    resolved_expr: ExprId,
    target_expr: ExprId,
    collect_steps: bool,
    simplify_options: &crate::SimplifyOptions,
) -> Option<(ExprId, Vec<crate::Step>, DeriveStrategy)> {
    let simplify_stage = run_simplify_stage(
        simplifier,
        resolved_expr,
        collect_steps,
        simplify_options.clone(),
    );
    if strong_target_match(&mut simplifier.context, simplify_stage.expr, target_expr) {
        let mut simplify_stage = simplify_stage;
        retarget_stage_output(&mut simplify_stage, target_expr);
        let steps = finalize_steps(
            resolved_expr,
            target_expr,
            vec![simplify_stage],
            &simplifier.context,
        );
        return Some((target_expr, steps, DeriveStrategy::Simplify));
    }

    let collect_stage = run_collect_stage(simplifier, resolved_expr, target_expr);
    if let Some(mut collect_stage) = collect_stage {
        retarget_stage_output(&mut collect_stage, target_expr);
        let steps = finalize_steps(
            resolved_expr,
            target_expr,
            vec![collect_stage],
            &simplifier.context,
        );
        return Some((target_expr, steps, DeriveStrategy::Collect));
    }

    let factored_stage = run_factored_stage(
        simplifier,
        resolved_expr,
        collect_steps,
        simplify_options.clone(),
    );
    if strong_target_match(&mut simplifier.context, factored_stage.expr, target_expr) {
        let mut factored_stage = factored_stage;
        retarget_stage_output(&mut factored_stage, target_expr);
        let steps = finalize_steps(
            resolved_expr,
            target_expr,
            vec![factored_stage],
            &simplifier.context,
        );
        return Some((target_expr, steps, DeriveStrategy::Factor));
    }

    let simplify_then_collect_stage =
        run_collect_stage(simplifier, simplify_stage.expr, target_expr);
    if let Some(mut simplify_then_collect_stage) = simplify_then_collect_stage {
        retarget_stage_output(&mut simplify_then_collect_stage, target_expr);
        let steps = finalize_steps(
            resolved_expr,
            target_expr,
            vec![simplify_stage, simplify_then_collect_stage],
            &simplifier.context,
        );
        return Some((target_expr, steps, DeriveStrategy::SimplifyThenCollect));
    }

    let simplify_then_factor_stage = run_factored_stage(
        simplifier,
        simplify_stage.expr,
        collect_steps,
        simplify_options.clone(),
    );
    if strong_target_match(
        &mut simplifier.context,
        simplify_then_factor_stage.expr,
        target_expr,
    ) {
        let mut simplify_then_factor_stage = simplify_then_factor_stage;
        retarget_stage_output(&mut simplify_then_factor_stage, target_expr);
        let steps = finalize_steps(
            resolved_expr,
            target_expr,
            vec![simplify_stage, simplify_then_factor_stage],
            &simplifier.context,
        );
        return Some((target_expr, steps, DeriveStrategy::SimplifyThenFactor));
    }

    None
}

fn run_simplify_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    collect_steps: bool,
    mut simplify_options: crate::SimplifyOptions,
) -> DeriveStageOutput {
    simplify_options.collect_steps = collect_steps;
    run_stage(simplifier, expr, collect_steps, simplify_options)
}

fn run_collect_stage(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveStageOutput> {
    for var_name in collect_candidate_variables(&simplifier.context, expr, target_expr) {
        let Some(rewrite) =
            cas_engine::try_collect_by_var(&mut simplifier.context, expr, &var_name)
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

    match cas_solver_core::step_optimization_runtime::optimize_steps_semantic(
        cleaned,
        ctx,
        original_expr,
        final_expr,
    ) {
        cas_solver_core::step_optimization_runtime::StepOptimizationResult::Steps(steps) => steps,
        cas_solver_core::step_optimization_runtime::StepOptimizationResult::NoSimplificationNeeded => {
            Vec::new()
        }
    }
}

fn strong_target_match(ctx: &mut cas_ast::Context, actual: ExprId, target: ExprId) -> bool {
    if actual == target {
        return true;
    }

    let normalized_actual = cas_math::canonical_forms::normalize_core(ctx, actual);
    let normalized_target = cas_math::canonical_forms::normalize_core(ctx, target);
    if cas_ast::ordering::compare_expr(ctx, normalized_actual, normalized_target) == Ordering::Equal
    {
        return true;
    }

    if commutative_mul_multiset_match(ctx, actual, target) {
        return true;
    }

    cas_math::semantic_equality::SemanticEqualityChecker::new(ctx)
        .are_equal_for_cycle_check(actual, target)
}

fn commutative_mul_multiset_match(
    ctx: &mut cas_ast::Context,
    actual: ExprId,
    target: ExprId,
) -> bool {
    if !ctx.is_mul_commutative(actual) || !ctx.is_mul_commutative(target) {
        return false;
    }

    let actual_factors = cas_math::trig_roots_flatten::flatten_mul_chain(ctx, actual);
    let target_factors = cas_math::trig_roots_flatten::flatten_mul_chain(ctx, target);
    if actual_factors.len() != target_factors.len() {
        return false;
    }

    let checker = cas_math::semantic_equality::SemanticEqualityChecker::new(ctx);
    let mut used = vec![false; target_factors.len()];
    for actual_factor in actual_factors {
        let mut matched = false;
        for (index, target_factor) in target_factors.iter().enumerate() {
            if used[index] {
                continue;
            }
            if checker.are_equal_for_cycle_check(actual_factor, *target_factor) {
                used[index] = true;
                matched = true;
                break;
            }
        }
        if !matched {
            return false;
        }
    }

    true
}

fn retarget_stage_output(stage: &mut DeriveStageOutput, target_expr: ExprId) {
    let previous_expr = stage.expr;
    stage.expr = target_expr;

    if let Some(last_step) = stage.steps.last_mut() {
        last_step.after = target_expr;
        if last_step.global_after.is_some() {
            last_step.global_after = Some(target_expr);
        }
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
            lines.insert(1, format!("Target: {target}"));
            lines.insert(2, format!("Strategy: {}", strategy.label()));
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

fn append_equivalence_summary(lines: &mut Vec<String>, equivalence: &crate::EquivalenceResult) {
    let formatted = crate::format_equivalence_result_lines(equivalence);
    if let Some(first) = formatted.first() {
        lines.push(format!("Equivalence: {first}"));
    }
    lines.extend(formatted.into_iter().skip(1));
}

fn collect_candidate_variables(
    ctx: &cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Vec<String> {
    let source_vars = cas_ast::collect_variables(ctx, source_expr);
    let target_vars = cas_ast::collect_variables(ctx, target_expr);

    source_vars
        .intersection(&target_vars)
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}
