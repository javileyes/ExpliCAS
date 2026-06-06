//! Post-processing: diagnostics assembly, session caching, and output construction.
//!
//! After action dispatch produces a raw result, this module assembles the
//! unified `Diagnostics`, updates session cache, and constructs the final `EvalOutput`.

use super::*;

type EngineSimplifiedUpdate = cas_session_core::eval::SimplifiedUpdate<
    crate::DomainMode,
    crate::diagnostics::RequiredItem,
    crate::step::Step,
>;

fn push_step_requires_with_display_dedup(
    ctx: &cas_ast::Context,
    steps: &[crate::Step],
    diagnostics: &mut crate::diagnostics::Diagnostics,
) {
    for cond in cas_session_core::eval::collect_step_items_with_display_dedup(
        steps,
        |step| step.required_conditions().to_vec(),
        |cond| cond.display(ctx),
        |cond| cond,
    ) {
        diagnostics.push_required(cond, crate::diagnostics::RequireOrigin::RewriteAirbag);
    }
}

fn push_solver_requires(
    solver_required: &[crate::ImplicitCondition],
    diagnostics: &mut crate::diagnostics::Diagnostics,
) {
    for cond in solver_required {
        diagnostics.push_required(
            cond.clone(),
            crate::diagnostics::RequireOrigin::EquationDerived,
        );
    }
}

fn push_rewrite_requires(
    rewrite_required: &[crate::ImplicitCondition],
    diagnostics: &mut crate::diagnostics::Diagnostics,
) {
    for cond in rewrite_required {
        diagnostics.push_required(
            cond.clone(),
            crate::diagnostics::RequireOrigin::RewriteAirbag,
        );
    }
}

fn eval_result_first_expr(result: &EvalResult) -> Option<ExprId> {
    match result {
        EvalResult::Expr(e) => Some(*e),
        EvalResult::Set(exprs) => exprs.first().copied(),
        EvalResult::SolutionSet(solution_set) => {
            use cas_ast::SolutionSet;
            match solution_set {
                SolutionSet::Discrete(vec) => vec.first().copied(),
                _ => None,
            }
        }
        _ => None,
    }
}

fn present_calculus_required_condition(
    ctx: &mut cas_ast::Context,
    condition: crate::ImplicitCondition,
) -> crate::ImplicitCondition {
    fn present_expr(ctx: &mut cas_ast::Context, expr: ExprId) -> ExprId {
        crate::rules::calculus::compact_double_angle_sine_products_for_calculus_presentation(
            ctx, expr,
        )
        .unwrap_or(expr)
    }

    match condition {
        crate::ImplicitCondition::Positive(expr) => {
            crate::ImplicitCondition::Positive(present_expr(ctx, expr))
        }
        crate::ImplicitCondition::NonNegative(expr) => {
            crate::ImplicitCondition::NonNegative(present_expr(ctx, expr))
        }
        crate::ImplicitCondition::NonZero(expr) => crate::ImplicitCondition::NonZero(expr),
        crate::ImplicitCondition::LowerBound(expr, bound) => {
            crate::ImplicitCondition::LowerBound(expr, bound)
        }
    }
}

fn resolved_is_calculus_call(ctx: &cas_ast::Context, resolved: ExprId) -> bool {
    matches!(
        ctx.get(resolved),
        cas_ast::Expr::Function(fn_id, _) if matches!(ctx.sym_name(*fn_id), "diff" | "integrate" | "limit")
    )
}

fn compact_trig_log_source_residual_condition_aliases(
    ctx: &mut cas_ast::Context,
    resolved: ExprId,
    diagnostics: &mut crate::diagnostics::Diagnostics,
) {
    use std::collections::HashSet;

    let aliases =
        cas_math::reciprocal_trig_log_domain::integrate_reciprocal_trig_log_source_condition_aliases(
            ctx, resolved,
        );
    if aliases.is_empty() {
        return;
    }

    let nonzero_fingerprints: HashSet<_> = diagnostics
        .requires
        .iter()
        .filter_map(|item| match &item.cond {
            crate::ImplicitCondition::NonZero(expr) => Some(crate::expr_fingerprint(ctx, *expr)),
            _ => None,
        })
        .collect();
    let redundant_aliases: Vec<_> = aliases
        .into_iter()
        .filter(|alias| {
            nonzero_fingerprints.contains(&crate::expr_fingerprint(ctx, alias.source_pole))
        })
        .map(|alias| alias.factored_pole)
        .collect();
    let redundant_fingerprints: HashSet<_> = redundant_aliases
        .iter()
        .map(|factored_pole| crate::expr_fingerprint(ctx, *factored_pole))
        .collect();
    let redundant_displays: HashSet<_> = redundant_aliases
        .iter()
        .map(|factored_pole| crate::ImplicitCondition::NonZero(*factored_pole).display(ctx))
        .collect();
    if redundant_fingerprints.is_empty() {
        return;
    }

    diagnostics.requires.retain(|item| match &item.cond {
        crate::ImplicitCondition::NonZero(expr) => {
            !redundant_fingerprints.contains(&crate::expr_fingerprint(ctx, *expr))
                && !redundant_displays.contains(&item.cond.display(ctx))
        }
        _ => true,
    });
}

fn compact_public_trig_log_source_residual_condition_aliases(
    ctx: &mut cas_ast::Context,
    resolved: ExprId,
    conditions: &mut Vec<crate::ImplicitCondition>,
) {
    use std::collections::HashSet;

    let aliases =
        cas_math::reciprocal_trig_log_domain::integrate_reciprocal_trig_log_source_condition_aliases(
            ctx, resolved,
        );
    if aliases.is_empty() {
        return;
    }

    let nonzero_fingerprints: HashSet<_> = conditions
        .iter()
        .filter_map(|condition| match condition {
            crate::ImplicitCondition::NonZero(expr) => Some(crate::expr_fingerprint(ctx, *expr)),
            _ => None,
        })
        .collect();
    let nonzero_displays: HashSet<_> = conditions
        .iter()
        .filter_map(|condition| match condition {
            crate::ImplicitCondition::NonZero(_) => Some(condition_display_key(ctx, condition)),
            _ => None,
        })
        .collect();
    let redundant_aliases: Vec<_> = aliases
        .into_iter()
        .filter(|alias| {
            nonzero_fingerprints.contains(&crate::expr_fingerprint(ctx, alias.source_pole))
                || nonzero_displays.contains(&condition_display_key(
                    ctx,
                    &crate::ImplicitCondition::NonZero(alias.source_pole),
                ))
        })
        .map(|alias| alias.factored_pole)
        .collect();
    let redundant_fingerprints: HashSet<_> = redundant_aliases
        .iter()
        .map(|factored_pole| crate::expr_fingerprint(ctx, *factored_pole))
        .collect();
    let redundant_displays: HashSet<_> = redundant_aliases
        .iter()
        .map(|factored_pole| {
            condition_display_key(ctx, &crate::ImplicitCondition::NonZero(*factored_pole))
        })
        .collect();
    if redundant_fingerprints.is_empty() {
        return;
    }

    conditions.retain(|condition| match condition {
        crate::ImplicitCondition::NonZero(expr) => {
            !redundant_fingerprints.contains(&crate::expr_fingerprint(ctx, *expr))
                && !redundant_displays.contains(&condition_display_key(ctx, condition))
        }
        _ => true,
    });
}

fn condition_display_key(ctx: &cas_ast::Context, condition: &crate::ImplicitCondition) -> String {
    condition.display(ctx).replace(" * ", "·")
}

fn present_calculus_required_diagnostics(
    ctx: &mut cas_ast::Context,
    diagnostics: &mut crate::diagnostics::Diagnostics,
) {
    for item in &mut diagnostics.requires {
        item.cond = present_calculus_required_condition(ctx, item.cond.clone());
    }
}

fn push_structural_requires(
    ctx: &mut cas_ast::Context,
    resolved: ExprId,
    result: &EvalResult,
    diagnostics: &mut crate::diagnostics::Diagnostics,
) {
    use crate::infer_implicit_domain;

    let resolved_is_calculus_call = resolved_is_calculus_call(ctx, resolved);

    let input_domain =
        infer_implicit_domain(ctx, resolved, crate::semantics::ValueDomain::RealOnly);
    for cond in input_domain.conditions() {
        if let crate::ImplicitCondition::NonZero(expr) = cond {
            if let Some(required_conditions) =
                crate::calculus_residual_support::shifted_integrate_resolved_reciprocal_half_power_residual_passthrough_nonzero_required_conditions(
                    ctx, *expr,
                )
            {
                for required in required_conditions {
                    diagnostics.push_required(
                        required,
                        crate::diagnostics::RequireOrigin::InputImplicit,
                    );
                }
                continue;
            }
            if let Some(required_conditions) =
                crate::calculus_residual_support::shifted_integral_residual_passthrough_nonzero_required_conditions(
                    ctx, *expr,
                )
            {
                for required in required_conditions {
                    diagnostics.push_required(
                        required,
                        crate::diagnostics::RequireOrigin::InputImplicit,
                    );
                }
                continue;
            }
        }
        let cond = if resolved_is_calculus_call {
            present_calculus_required_condition(ctx, cond.clone())
        } else {
            cond.clone()
        };
        diagnostics.push_required(cond, crate::diagnostics::RequireOrigin::InputImplicit);
    }
    push_intrinsic_function_requires(
        ctx,
        resolved,
        crate::diagnostics::RequireOrigin::InputImplicit,
        diagnostics,
    );

    if let Some(result_id) = eval_result_first_expr(result) {
        let output_domain =
            infer_implicit_domain(ctx, result_id, crate::semantics::ValueDomain::RealOnly);
        for cond in output_domain.conditions() {
            if let crate::ImplicitCondition::NonZero(expr) = cond {
                if let Some(required_conditions) =
                    crate::calculus_residual_support::shifted_integral_residual_passthrough_nonzero_required_conditions(
                        ctx, *expr,
                    )
                {
                    for required in required_conditions {
                        diagnostics.push_required(
                            required,
                            crate::diagnostics::RequireOrigin::OutputImplicit,
                        );
                    }
                    continue;
                }
            }
            let cond = if resolved_is_calculus_call {
                present_calculus_required_condition(ctx, cond.clone())
            } else {
                cond.clone()
            };
            diagnostics.push_required(cond, crate::diagnostics::RequireOrigin::OutputImplicit);
        }
        if !resolved_is_calculus_call {
            push_intrinsic_function_requires(
                ctx,
                result_id,
                crate::diagnostics::RequireOrigin::OutputImplicit,
                diagnostics,
            );
        }
    }
}

fn push_intrinsic_function_requires(
    ctx: &mut cas_ast::Context,
    root: ExprId,
    origin: crate::diagnostics::RequireOrigin,
    diagnostics: &mut crate::diagnostics::Diagnostics,
) {
    let mut stack = vec![(root, false)];
    while let Some((expr, inside_calculus_call)) = stack.pop() {
        match ctx.get(expr).clone() {
            cas_ast::Expr::Function(fn_id, args) => {
                let builtin = ctx.builtin_of(fn_id);
                if !inside_calculus_call
                    && args.len() == 1
                    && !cas_ast::collect_variables(ctx, args[0]).is_empty()
                {
                    let requirement = match builtin {
                        Some(cas_ast::BuiltinFn::Tan | cas_ast::BuiltinFn::Sec) => {
                            let denominator =
                                ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![args[0]]);
                            Some(crate::ImplicitCondition::NonZero(denominator))
                        }
                        Some(cas_ast::BuiltinFn::Cot | cas_ast::BuiltinFn::Csc) => {
                            let denominator =
                                ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![args[0]]);
                            Some(crate::ImplicitCondition::NonZero(denominator))
                        }
                        Some(
                            cas_ast::BuiltinFn::Arcsin
                            | cas_ast::BuiltinFn::Asin
                            | cas_ast::BuiltinFn::Arccos
                            | cas_ast::BuiltinFn::Acos,
                        ) => Some(inverse_unit_interval_intrinsic_requirement(
                            ctx,
                            args[0],
                            UnitIntervalBoundary::Closed,
                        )),
                        Some(cas_ast::BuiltinFn::Atanh) => {
                            Some(inverse_unit_interval_intrinsic_requirement(
                                ctx,
                                args[0],
                                UnitIntervalBoundary::Open,
                            ))
                        }
                        _ => None,
                    };
                    if let Some(requirement) = requirement {
                        diagnostics.push_required(requirement, origin);
                    }
                }
                let enters_calculus = matches!(ctx.sym_name(fn_id), "diff" | "integrate" | "limit");
                stack.extend(
                    args.into_iter()
                        .map(|arg| (arg, inside_calculus_call || enters_calculus)),
                );
            }
            cas_ast::Expr::Add(left, right)
            | cas_ast::Expr::Sub(left, right)
            | cas_ast::Expr::Mul(left, right)
            | cas_ast::Expr::Div(left, right) => {
                stack.push((left, inside_calculus_call));
                stack.push((right, inside_calculus_call));
            }
            cas_ast::Expr::Pow(base, exponent) => {
                stack.push((base, inside_calculus_call));
                stack.push((exponent, inside_calculus_call));
            }
            cas_ast::Expr::Neg(inner) | cas_ast::Expr::Hold(inner) => {
                stack.push((inner, inside_calculus_call));
            }
            cas_ast::Expr::Matrix { data, .. } => {
                stack.extend(data.into_iter().map(|arg| (arg, inside_calculus_call)));
            }
            cas_ast::Expr::Number(_)
            | cas_ast::Expr::Variable(_)
            | cas_ast::Expr::Constant(_)
            | cas_ast::Expr::SessionRef(_) => {}
        }
    }
}

enum UnitIntervalBoundary {
    Closed,
    Open,
}

fn sqrt_like_radicand_for_intrinsic_domain(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    if let Some(radicand) = cas_math::expr_extract::extract_sqrt_argument_view(ctx, expr) {
        return Some(radicand);
    }

    let cas_ast::Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    match ctx.get(*exponent) {
        cas_ast::Expr::Number(value)
            if value == &num_rational::BigRational::new(1.into(), 2.into()) =>
        {
            Some(*base)
        }
        _ => None,
    }
}

fn inverse_unit_interval_intrinsic_requirement(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
    boundary: UnitIntervalBoundary,
) -> crate::ImplicitCondition {
    let one = ctx.num(1);
    let bounded = if let Some(radicand) = sqrt_like_radicand_for_intrinsic_domain(ctx, arg) {
        ctx.add(cas_ast::Expr::Sub(one, radicand))
    } else {
        let two = ctx.num(2);
        let square = ctx.add(cas_ast::Expr::Pow(arg, two));
        ctx.add(cas_ast::Expr::Sub(one, square))
    };
    match boundary {
        UnitIntervalBoundary::Closed => crate::ImplicitCondition::NonNegative(bounded),
        UnitIntervalBoundary::Open => crate::ImplicitCondition::Positive(bounded),
    }
}

fn push_blocked_hints(
    blocked_hints: &[crate::BlockedHint],
    diagnostics: &mut crate::diagnostics::Diagnostics,
) {
    for hint in blocked_hints {
        diagnostics.push_blocked(hint.clone());
    }
}

fn has_empty_real_domain_diff_block(blocked_hints: &[crate::BlockedHint]) -> bool {
    blocked_hints.iter().any(|hint| {
        hint.rule == "Symbolic Differentiation"
            && hint
                .suggestion
                .contains("real domain is empty; no real derivative is exposed")
    })
}

fn is_inverse_reciprocal_trig_empty_domain_diff_residual(
    ctx: &cas_ast::Context,
    result: &EvalResult,
) -> bool {
    let Some(result_expr) = eval_result_first_expr(result) else {
        return false;
    };
    let Some(call) = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, result_expr)
    else {
        return false;
    };
    let mut target = call.target;
    while let cas_ast::Expr::Hold(inner) = ctx.get(target) {
        target = *inner;
    }
    let cas_ast::Expr::Function(fn_id, args) = ctx.get(target) else {
        return false;
    };
    if args.len() != 1 {
        return false;
    }
    if !matches!(
        ctx.builtin_of(*fn_id),
        Some(
            cas_ast::BuiltinFn::Arcsin
                | cas_ast::BuiltinFn::Asin
                | cas_ast::BuiltinFn::Arccos
                | cas_ast::BuiltinFn::Acos
        )
    ) {
        return false;
    }
    let cas_ast::Expr::Function(arg_fn_id, arg_args) = ctx.get(args[0]) else {
        return false;
    };
    arg_args.len() == 1
        && matches!(
            ctx.builtin_of(*arg_fn_id),
            Some(cas_ast::BuiltinFn::Sec | cas_ast::BuiltinFn::Csc)
        )
}

fn eval_result_is_undefined(ctx: &cas_ast::Context, result: &EvalResult) -> bool {
    matches!(
        eval_result_first_expr(result).map(|expr| ctx.get(expr)),
        Some(cas_ast::Expr::Constant(cas_ast::Constant::Undefined))
    )
}

fn public_required_conditions(
    ctx: &cas_ast::Context,
    result: &EvalResult,
    diagnostics: &crate::diagnostics::Diagnostics,
    blocked_hints: &[crate::BlockedHint],
) -> Vec<crate::ImplicitCondition> {
    if eval_result_is_undefined(ctx, result) {
        return Vec::new();
    }
    if has_empty_real_domain_diff_block(blocked_hints)
        && is_inverse_reciprocal_trig_empty_domain_diff_residual(ctx, result)
    {
        return Vec::new();
    }

    diagnostics.required_conditions()
}

fn push_assumed_from_steps(
    steps: &[crate::Step],
    diagnostics: &mut crate::diagnostics::Diagnostics,
) {
    for step in steps {
        for event in step.assumption_events() {
            diagnostics.push_assumed(event.clone());
        }
    }
}

fn build_simplified_cache_update(
    ctx: &cas_ast::Context,
    result: &EvalResult,
    options: &crate::options::EvalOptions,
    diagnostics: &crate::diagnostics::Diagnostics,
    steps: &[crate::Step],
) -> Option<EngineSimplifiedUpdate> {
    match result {
        EvalResult::Expr(simplified_expr)
            if !cas_math::poly_result::is_poly_result(ctx, *simplified_expr) =>
        {
            Some(EngineSimplifiedUpdate {
                domain: options.shared.semantics.domain_mode,
                expr: *simplified_expr,
                requires: diagnostics.requires.clone(),
                steps: Some(std::sync::Arc::new(steps.to_vec())),
            })
        }
        _ => None,
    }
}

struct EvalDiagnosticsInput<'a> {
    ctx: &'a mut cas_ast::Context,
    resolved: ExprId,
    result: &'a EvalResult,
    steps: &'a [crate::Step],
    solver_required: &'a [crate::ImplicitCondition],
    rewrite_required: &'a [crate::ImplicitCondition],
    blocked_hints: &'a [crate::BlockedHint],
    inherited_diagnostics: &'a crate::diagnostics::Diagnostics,
}

fn build_eval_diagnostics(input: EvalDiagnosticsInput<'_>) -> crate::diagnostics::Diagnostics {
    let EvalDiagnosticsInput {
        ctx,
        resolved,
        result,
        steps,
        solver_required,
        rewrite_required,
        blocked_hints,
        inherited_diagnostics,
    } = input;
    let mut diagnostics = crate::diagnostics::Diagnostics::new();

    // Each source gets its proper provenance/origin classification.
    push_step_requires_with_display_dedup(ctx, steps, &mut diagnostics);
    push_solver_requires(solver_required, &mut diagnostics);
    push_rewrite_requires(rewrite_required, &mut diagnostics);
    push_structural_requires(ctx, resolved, result, &mut diagnostics);
    push_blocked_hints(blocked_hints, &mut diagnostics);
    push_assumed_from_steps(steps, &mut diagnostics);

    // Track provenance when reusing cached/session entries.
    diagnostics.inherit_requires_from(inherited_diagnostics);

    if resolved_is_calculus_call(ctx, resolved) {
        present_calculus_required_diagnostics(ctx, &mut diagnostics);
        compact_trig_log_source_residual_condition_aliases(ctx, resolved, &mut diagnostics);
    }

    // Stable output ordering and trivial-condition filtering.
    diagnostics.dedup_and_sort(ctx);
    diagnostics
}

impl Engine {
    /// Build unified diagnostics, update session cache, and construct final `EvalOutput`.
    ///
    /// This is the post-processing stage after action dispatch has produced raw results.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_output<StoreT>(
        &mut self,
        stored_id: Option<u64>,
        parsed: ExprId,
        resolved: ExprId,
        result: EvalResult,
        domain_warnings: Vec<DomainWarning>,
        steps: Vec<crate::Step>,
        solve_steps: Vec<crate::api::SolveStep>,
        solver_assumptions: Vec<crate::AssumptionRecord>,
        output_scopes: Vec<cas_formatter::display_transforms::ScopeTag>,
        solver_required: Vec<crate::ImplicitCondition>,
        rewrite_required: Vec<crate::ImplicitCondition>,
        inherited_diagnostics: crate::diagnostics::Diagnostics,
        store: &mut StoreT,
        options: &crate::options::EvalOptions,
    ) -> Result<EvalOutput, anyhow::Error>
    where
        StoreT: cas_session_core::eval::TypedEvalStore<
            crate::DomainMode,
            crate::diagnostics::RequiredItem,
            crate::step::Step,
            crate::diagnostics::Diagnostics,
        >,
    {
        // Collect blocked hints from simplifier
        let blocked_hints = self.simplifier.take_blocked_hints();

        let mut diagnostics = build_eval_diagnostics(EvalDiagnosticsInput {
            ctx: &mut self.simplifier.context,
            resolved,
            result: &result,
            steps: &steps,
            solver_required: &solver_required,
            rewrite_required: &rewrite_required,
            blocked_hints: &blocked_hints,
            inherited_diagnostics: &inherited_diagnostics,
        });
        if has_empty_real_domain_diff_block(&blocked_hints)
            && is_inverse_reciprocal_trig_empty_domain_diff_residual(
                &self.simplifier.context,
                &result,
            )
        {
            diagnostics.requires.clear();
        }

        // Update stored entry with final diagnostics and optional simplified cache.
        // This keeps cache write policy centralized in session-core helpers.
        let simplified_update = build_simplified_cache_update(
            &self.simplifier.context,
            &result,
            options,
            &diagnostics,
            &steps,
        );
        cas_session_core::eval::apply_post_dispatch_store_updates(
            store,
            stored_id,
            diagnostics.clone(),
            simplified_update,
        );

        // Legacy field: extract conditions from diagnostics for backward compatibility
        // Tests and some code paths still use output.required_conditions
        let required_conditions = public_required_conditions(
            &self.simplifier.context,
            &result,
            &diagnostics,
            &blocked_hints,
        );
        let required_conditions =
            cas_solver_core::domain_normalization::normalize_and_dedupe_conditions(
                &mut self.simplifier.context,
                &required_conditions,
            );
        let mut required_conditions = required_conditions;
        compact_public_trig_log_source_residual_condition_aliases(
            &mut self.simplifier.context,
            parsed,
            &mut required_conditions,
        );

        // V2.9.9: Convert raw steps to display-ready steps via unified pipeline.
        // This is the ONLY place DisplayEvalSteps is constructed from raw steps.
        // The pipeline removes no-ops and prepares steps for all renderers.
        let display_steps = cas_solver_core::eval_step_pipeline::to_display_eval_steps(steps);

        Ok(EvalOutput {
            stored_id,
            parsed,
            resolved,
            result,
            domain_warnings,
            steps: display_steps,
            solve_steps,
            solver_assumptions,
            output_scopes,
            required_conditions,
            blocked_hints,
            diagnostics,
        })
    }
}
