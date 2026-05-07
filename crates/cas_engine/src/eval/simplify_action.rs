//! Simplify action handler for `Engine::eval`.

use super::*;
use cas_ast::{BuiltinFn, Expr};

fn expr_contains_any_builtin_local(
    ctx: &cas_ast::Context,
    root: ExprId,
    builtins: &[BuiltinFn],
) -> bool {
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Function(fn_id, args) => {
                if builtins
                    .iter()
                    .any(|builtin| ctx.is_builtin(*fn_id, *builtin))
                {
                    return true;
                }
                stack.extend(args.iter().copied());
            }
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Div(lhs, rhs)
            | Expr::Pow(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

fn expr_contains_hyperbolic_builtin_local(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    expr_contains_any_builtin_local(
        ctx,
        expr,
        &[
            BuiltinFn::Sinh,
            BuiltinFn::Cosh,
            BuiltinFn::Tanh,
            BuiltinFn::Asinh,
            BuiltinFn::Acosh,
            BuiltinFn::Atanh,
        ],
    )
}

fn expr_is_named_function_call_local(ctx: &cas_ast::Context, expr: ExprId, names: &[&str]) -> bool {
    let Expr::Function(fn_id, _) = ctx.get(expr) else {
        return false;
    };
    names.iter().any(|name| ctx.sym_name(*fn_id) == *name)
}

fn expr_contains_named_function_local(
    ctx: &cas_ast::Context,
    root: ExprId,
    names: &[&str],
) -> bool {
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Function(fn_id, args) => {
                if names.iter().any(|name| ctx.sym_name(*fn_id) == *name) {
                    return true;
                }
                stack.extend(args.iter().copied());
            }
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Div(lhs, rhs)
            | Expr::Pow(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

fn expr_is_post_calculus_residual_candidate_local(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Neg(_)
    )
}

fn collapse_redundant_post_calculus_trace_if_direct_step_is_compact(
    ctx: &mut cas_ast::Context,
    resolved: ExprId,
    presented: ExprId,
    steps: &mut Vec<crate::Step>,
) -> bool {
    let Some(first) = steps.first() else {
        return false;
    };
    if first.rule_name.as_str() != "Symbolic Differentiation" {
        return false;
    }

    let Some(first_presented) =
        crate::rules::calculus::try_post_calculus_presentation(ctx, resolved, first.after)
    else {
        return false;
    };

    let first_presented_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: first_presented,
        }
    );
    let final_presented_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: presented,
        }
    );
    if first_presented_display != final_presented_display {
        return false;
    }

    steps.truncate(1);
    if let Some(first) = steps.first_mut() {
        first.after = presented;
        first.global_after = Some(presented);
    }
    true
}

impl Engine {
    /// Handle `EvalAction::Expand`.
    ///
    /// Actualmente usa el mismo simplificador principal; se mantiene separado
    /// para que el dispatch quede desacoplado del detalle de implementación.
    pub(super) fn eval_expand(
        &mut self,
        options: &crate::options::EvalOptions,
        resolved: ExprId,
    ) -> Result<ActionResult, anyhow::Error> {
        let (res, steps) = self.simplifier.simplify(resolved);
        let warnings = collect_domain_warnings(
            &self.simplifier.context,
            options.shared.semantics.value_domain,
            res,
            &steps,
        );
        let rewrite_required = self.simplifier.take_required_conditions();
        Ok((
            EvalResult::Expr(res),
            warnings,
            steps,
            vec![],
            vec![],
            vec![],
            vec![],
            rewrite_required,
        ))
    }

    /// Handle `EvalAction::Simplify`: tool dispatch, simplification, const fold, domain classification.
    pub(super) fn eval_simplify(
        &mut self,
        options: &crate::options::EvalOptions,
        resolved: ExprId,
    ) -> Result<ActionResult, anyhow::Error> {
        let effective_opts = self.effective_options(options, resolved);
        let profile = self.profile_cache.get_or_build(&effective_opts);
        let inherited_allow_numerical_verification = self.simplifier.allow_numerical_verification;
        let inherited_debug_mode = self.simplifier.debug_mode;
        let inherited_step_listener = self.simplifier.replace_step_listener(None);
        let mut ctx_simplifier = Simplifier::from_profile_with_context(
            profile,
            std::mem::take(&mut self.simplifier.context),
        );
        let preserve_hidden_solve_fast_paths = effective_opts.shared.context_mode
            == crate::options::ContextMode::Solve
            && effective_opts.shared.semantics.domain_mode == crate::DomainMode::Strict;
        let runtime_steps_mode = match effective_opts.steps_mode {
            crate::options::StepsMode::Off if preserve_hidden_solve_fast_paths => {
                crate::options::StepsMode::Off
            }
            crate::options::StepsMode::Off
                if !expr_contains_hyperbolic_builtin_local(&ctx_simplifier.context, resolved) =>
            {
                crate::options::StepsMode::Compact
            }
            mode => mode,
        };
        ctx_simplifier.set_steps_mode(runtime_steps_mode);
        ctx_simplifier.allow_numerical_verification = inherited_allow_numerical_verification;
        ctx_simplifier.debug_mode = inherited_debug_mode;
        ctx_simplifier.set_step_listener(inherited_step_listener);

        let mut simplify_opts = effective_opts.to_simplify_options();

        let (expr_to_simplify, expand_log_events) =
            if let Expr::Function(fn_id, args) = ctx_simplifier.context.get(resolved).clone() {
                match ctx_simplifier.context.sym_name(fn_id) {
                    "collect" => {
                        simplify_opts.goal = crate::semantics::NormalFormGoal::Collected;
                        (resolved, Vec::new())
                    }
                    "expand_log" if args.len() == 1 => {
                        simplify_opts.goal = crate::semantics::NormalFormGoal::ExpandedLog;
                        crate::rules::logarithms::expand_logs_with_assumptions(
                            &mut ctx_simplifier.context,
                            args[0],
                            effective_opts.shared.semantics.domain_mode,
                            effective_opts.shared.semantics.value_domain,
                        )
                        .unwrap_or((args[0], Vec::new()))
                    }
                    _ => (resolved, Vec::new()),
                }
            } else {
                (resolved, Vec::new())
            };

        let (mut res, mut steps, stats) =
            ctx_simplifier.simplify_with_stats(expr_to_simplify, simplify_opts.clone());

        let hyperbolic_zero_rewrite =
            crate::rules::hyperbolic::try_build_atanh_square_ratio_log_zero_rewrite(
                &mut ctx_simplifier.context,
                expr_to_simplify,
            )
            .map(|rewrite| (expr_to_simplify, rewrite))
            .or_else(|| {
                crate::rules::hyperbolic::try_build_atanh_square_ratio_log_zero_rewrite(
                    &mut ctx_simplifier.context,
                    res,
                )
                .map(|rewrite| (res, rewrite))
            });
        if let Some((rewrite_source, rewrite)) = hyperbolic_zero_rewrite {
            let final_expr = rewrite.final_expr();
            res = final_expr;
            if effective_opts.steps_mode != crate::options::StepsMode::Off {
                steps.clear();
                let mut step = crate::Step::new(
                    &rewrite.description,
                    &rewrite.description,
                    rewrite_source,
                    final_expr,
                    Vec::new(),
                    Some(&ctx_simplifier.context),
                );
                step.meta_mut().required_conditions = rewrite.required_conditions.clone();
                steps.push(step);
            }
        }

        if !expand_log_events.is_empty() {
            if steps.is_empty() {
                let mut step = crate::Step::new(
                    "Log expansion",
                    "expand_log",
                    resolved,
                    res,
                    Vec::new(),
                    Some(&ctx_simplifier.context),
                );
                step.meta_mut().assumption_events.extend(expand_log_events);
                steps.push(step);
            } else {
                steps[0]
                    .meta_mut()
                    .assumption_events
                    .extend(expand_log_events);
            }
        }

        let calculus_call_names = ["diff", "integrate", "int"];
        let embedded_calculus_residual =
            expr_contains_named_function_local(
                &ctx_simplifier.context,
                expr_to_simplify,
                &calculus_call_names,
            ) && !expr_is_named_function_call_local(
                &ctx_simplifier.context,
                expr_to_simplify,
                &calculus_call_names,
            ) && !expr_contains_named_function_local(
                &ctx_simplifier.context,
                res,
                &calculus_call_names,
            ) && expr_is_post_calculus_residual_candidate_local(&ctx_simplifier.context, res);

        if embedded_calculus_residual {
            let first_pass_required = ctx_simplifier.take_required_conditions();
            let (post_residual_res, _post_residual_steps, _post_residual_stats) =
                ctx_simplifier.simplify_with_stats(res, simplify_opts);
            let second_pass_required = ctx_simplifier.take_required_conditions();
            ctx_simplifier.extend_required_conditions(
                first_pass_required.into_iter().chain(second_pass_required),
            );

            if post_residual_res != res {
                if effective_opts.steps_mode != crate::options::StepsMode::Off {
                    let mut post_residual_step = crate::Step::new(
                        "Post-calculus residual simplification",
                        "Re-simplify residual after resolving calculus calls",
                        res,
                        post_residual_res,
                        Vec::new(),
                        Some(&ctx_simplifier.context),
                    );
                    post_residual_step.importance = crate::ImportanceLevel::Medium;
                    steps.push(post_residual_step);
                }
                res = post_residual_res;
            }
        }

        if let Some(presented) = crate::rules::calculus::try_post_calculus_presentation(
            &mut ctx_simplifier.context,
            resolved,
            res,
        ) {
            if effective_opts.steps_mode != crate::options::StepsMode::Off
                && presented != res
                && !collapse_redundant_post_calculus_trace_if_direct_step_is_compact(
                    &mut ctx_simplifier.context,
                    resolved,
                    presented,
                    &mut steps,
                )
            {
                let mut presentation_step = crate::Step::new(
                    "Post-calculus presentation",
                    "Present calculus result in compact form",
                    res,
                    presented,
                    Vec::new(),
                    Some(&ctx_simplifier.context),
                );
                presentation_step.importance = crate::ImportanceLevel::Medium;
                steps.push(presentation_step);
            }
            res = presented;
        }

        if effective_opts.const_fold == crate::const_fold::ConstFoldMode::Safe {
            let mut budget = crate::budget::Budget::preset_cli();
            let cfg = crate::semantics::EvalConfig {
                value_domain: effective_opts.shared.semantics.value_domain,
                branch: effective_opts.shared.semantics.branch,
                ..Default::default()
            };
            if let Ok(fold_result) = crate::const_fold::fold_constants(
                &mut ctx_simplifier.context,
                res,
                &cfg,
                effective_opts.const_fold,
                &mut budget,
            ) {
                res = fold_result.expr;
            }
        }

        if let Some(presented) = crate::rules::calculus::try_post_calculus_presentation(
            &mut ctx_simplifier.context,
            resolved,
            res,
        ) {
            res = presented;
        } else if let Some(presented) = crate::rules::calculus::try_calculus_result_presentation(
            &mut ctx_simplifier.context,
            res,
        ) {
            res = presented;
        }

        self.simplifier
            .extend_blocked_hints(ctx_simplifier.take_blocked_hints());
        let rewrite_required = ctx_simplifier.take_required_conditions();
        let restored_step_listener = ctx_simplifier.replace_step_listener(None);
        self.simplifier.context = ctx_simplifier.context;
        self.simplifier.set_step_listener(restored_step_listener);

        {
            use crate::{classify_assumptions_in_place, infer_implicit_domain, DomainContext};

            let input_domain = infer_implicit_domain(
                &self.simplifier.context,
                resolved,
                effective_opts.shared.semantics.value_domain,
            );
            let global_requires: Vec<_> = input_domain.conditions().iter().cloned().collect();
            let mut dc = DomainContext::new(global_requires);

            for step in &mut steps {
                classify_assumptions_in_place(
                    &self.simplifier.context,
                    &mut dc,
                    &mut step.meta_mut().assumption_events,
                );
            }

            if effective_opts.shared.semantics.domain_mode == crate::DomainMode::Assume {
                for step in &mut steps {
                    if step.rule_name == "Log-Exp Inverse" {
                        for event in &mut step.meta_mut().assumption_events {
                            if matches!(event.kind, crate::AssumptionKind::DerivedFromRequires)
                                && matches!(event.key, crate::AssumptionKey::Positive { .. })
                            {
                                // `log(b, b^x) -> x` in Assume mode should still surface the
                                // user-visible positivity assumption on the symbolic base, even
                                // when that positivity is already implicit in the source log.
                                event.kind = crate::AssumptionKind::HeuristicAssumption;
                            }
                        }
                    }
                }
            }
        }

        let mut warnings = collect_domain_warnings(
            &self.simplifier.context,
            effective_opts.shared.semantics.value_domain,
            res,
            &steps,
        );

        if stats.timed_out {
            let message = match effective_opts.time_budget_ms {
                Some(ms) => format!(
                    "Partial result: simplification stopped after reaching the {} ms time budget.",
                    ms
                ),
                None => "Partial result: simplification stopped after reaching the time budget."
                    .to_string(),
            };
            let timeout_warning = DomainWarning {
                message,
                rule_name: "Simplification Time Budget".to_string(),
            };
            if !warnings.iter().any(|warning| warning == &timeout_warning) {
                warnings.push(timeout_warning);
            }
        }

        if effective_opts.shared.semantics.value_domain == crate::semantics::ValueDomain::RealOnly
            && cas_math::numeric_eval::contains_i(&self.simplifier.context, resolved)
        {
            let i_warning = DomainWarning {
                message: "To use complex arithmetic (i² = -1), run: semantics set value complex"
                    .to_string(),
                rule_name: "Imaginary Usage Warning".to_string(),
            };
            if !warnings.iter().any(|w| w.message == i_warning.message) {
                warnings.push(i_warning);
            }
        }

        Ok((
            EvalResult::Expr(res),
            warnings,
            steps,
            vec![],
            vec![],
            vec![],
            vec![],
            rewrite_required,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    #[test]
    fn eval_stateless_steps_off_handles_triple_sine_plus_rational_against_hyperbolic_pythagorean_regression(
    ) {
        let mut engine = Engine::new();
        let expr_text = "(sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))";
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Auto;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let output = engine
            .eval_stateless(
                options,
                crate::EvalRequest {
                    raw_input: expr_text.to_string(),
                    parsed,
                    action: crate::EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = output.result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_triple_sine_plus_rational_against_hyperbolic_pythagorean_regression(
    ) {
        let mut engine = Engine::new();
        let expr_text = "(sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))";
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_preserves_post_calculus_sqrt_tan_presentation() {
        let mut engine = Engine::new();
        let expr_text = "3*tan(sqrt(3*x+1))/(2*sqrt(3*x+1))";
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let options = crate::options::EvalOptions::default();

        let (result, ..) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };

        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))"
        );
    }

    #[test]
    fn with_profile_simplify_steps_off_handles_triple_sine_plus_rational_against_hyperbolic_pythagorean_regression(
    ) {
        let expr_text = "(sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))";
        let mut probe_engine = Engine::new();
        let probe_parsed = parse(expr_text, &mut probe_engine.simplifier.context)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        let effective_options = probe_engine.effective_options(&options, probe_parsed);
        let mut simplifier = crate::Simplifier::with_profile(&effective_options);
        simplifier.set_collect_steps(false);
        let parsed = parse(expr_text, &mut simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let (result, _steps) =
            simplifier.simplify_with_options(parsed, effective_options.to_simplify_options());
        assert_eq!(
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_diff_tan_affine_avoids_pre_diff_trig_expansion_timeout() {
        let mut engine = Engine::new();
        let expr_text = "diff(tan(2*x+1), x)";
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        options.time_budget_ms = Some(50);

        let output = engine
            .eval_stateless(
                options,
                crate::EvalRequest {
                    raw_input: expr_text.to_string(),
                    parsed,
                    action: crate::EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = output.result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "2 / cos(2 * x + 1)^2"
        );
        assert!(
            output
                .domain_warnings
                .iter()
                .all(|warning| warning.rule_name != "Simplification Time Budget"),
            "unexpected timeout warning: {:?}",
            output.domain_warnings
        );
        let required_display: Vec<_> = output
            .required_conditions
            .iter()
            .map(|condition| condition.display(&engine.simplifier.context))
            .collect();
        assert_eq!(
            required_display
                .iter()
                .filter(|condition| condition.as_str() == "cos(2 * x + 1) ≠ 0")
                .count(),
            1,
            "expected exactly one displayed cos-domain condition, got: {required_display:?}"
        );
    }

    #[test]
    fn eval_simplify_steps_off_diff_scaled_atanh_surd_polynomial_uses_direct_route() {
        let cases = [
            "diff(1/3 * atanh(1/3 * sqrt(3) * (x^2 + 2*x + 1)) * sqrt(3), x)",
            "diff(atanh((x^2 + 2*x + 1)/sqrt(3))/sqrt(3), x)",
        ];

        for expr_text in cases {
            let mut engine = Engine::new();
            let parsed = parse(expr_text, &mut engine.simplifier.context)
                .unwrap_or_else(|e| panic!("{e:?}"));
            let mut options = crate::options::EvalOptions::default();
            options.steps_mode = crate::options::StepsMode::Off;
            options.shared.context_mode = crate::options::ContextMode::Standard;
            options.shared.semantics.domain_mode = crate::DomainMode::Generic;
            options.time_budget_ms = Some(50);

            let output = engine
                .eval_stateless(
                    options,
                    crate::EvalRequest {
                        raw_input: expr_text.to_string(),
                        parsed,
                        action: crate::EvalAction::Simplify,
                        auto_store: false,
                    },
                )
                .unwrap_or_else(|e| panic!("{e:?}"));

            let crate::EvalResult::Expr(result) = output.result else {
                panic!("expected expression result");
            };
            assert_eq!(
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: result,
                }
                .to_string(),
                "(2 * x + 2) / (3 - (x + 1)^4)",
                "input: {expr_text}"
            );
            assert!(
                output
                    .domain_warnings
                    .iter()
                    .all(|warning| warning.rule_name != "Simplification Time Budget"),
                "unexpected timeout warning for {expr_text}: {:?}",
                output.domain_warnings
            );
        }
    }

    #[test]
    fn eval_simplify_integrate_scaled_denominator_square_preserves_required_domain() {
        let mut engine = Engine::new();
        let expr_text = "integrate((2*x+1)/(3*(x^2+x-1)^2), x)";
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let output = engine
            .eval_stateless(
                options,
                crate::EvalRequest {
                    raw_input: expr_text.to_string(),
                    parsed,
                    action: crate::EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .unwrap_or_else(|e| panic!("{e:?}"));

        let required_display = crate::render_conditions_normalized(
            &mut engine.simplifier.context,
            &output.required_conditions,
        );
        assert_eq!(
            required_display,
            vec!["x^2 + x - 1 ≠ 0".to_string()],
            "unexpected required_conditions: {required_display:?}"
        );
    }

    #[test]
    fn eval_simplify_steps_off_diff_tan_square_avoids_pre_diff_trig_expansion_timeout() {
        let mut engine = Engine::new();
        let expr_text = "diff(tan(2*x+1)^2, x)";
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        options.time_budget_ms = Some(50);

        let output = engine
            .eval_stateless(
                options,
                crate::EvalRequest {
                    raw_input: expr_text.to_string(),
                    parsed,
                    action: crate::EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = output.result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "(tan(2 * x + 1) * 4)/cos(2 * x + 1)^2"
        );
        assert!(
            output
                .domain_warnings
                .iter()
                .all(|warning| warning.rule_name != "Simplification Time Budget"),
            "unexpected timeout warning: {:?}",
            output.domain_warnings
        );
        let required_display: Vec<_> = output
            .required_conditions
            .iter()
            .map(|condition| condition.display(&engine.simplifier.context))
            .collect();
        assert_eq!(
            required_display
                .iter()
                .filter(|condition| condition.as_str() == "cos(2 * x + 1) ≠ 0")
                .count(),
            1,
            "expected exactly one displayed cos-domain condition, got: {required_display:?}"
        );
    }

    #[test]
    fn eval_simplify_steps_off_diff_variable_times_tan_avoids_pre_diff_trig_expansion_timeout() {
        let mut engine = Engine::new();
        let expr_text = "diff(x*tan(2*x+1), x)";
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        options.time_budget_ms = Some(50);

        let output = engine
            .eval_stateless(
                options,
                crate::EvalRequest {
                    raw_input: expr_text.to_string(),
                    parsed,
                    action: crate::EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = output.result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "tan(2 * x + 1) + (x * 2)/cos(2 * x + 1)^2"
        );
        assert!(
            output
                .domain_warnings
                .iter()
                .all(|warning| warning.rule_name != "Simplification Time Budget"),
            "unexpected timeout warning: {:?}",
            output.domain_warnings
        );
        let required_display: Vec<_> = output
            .required_conditions
            .iter()
            .map(|condition| condition.display(&engine.simplifier.context))
            .collect();
        assert_eq!(
            required_display
                .iter()
                .filter(|condition| condition.as_str() == "cos(2 * x + 1) ≠ 0")
                .count(),
            1,
            "expected exactly one displayed cos-domain condition, got: {required_display:?}"
        );
    }

    #[test]
    fn eval_simplify_steps_off_diff_scaled_variable_times_tan_avoids_post_diff_trig_timeout() {
        let mut engine = Engine::new();
        let expr_text = "diff(2*x*tan(2*x+1), x)";
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        options.time_budget_ms = Some(50);

        let output = engine
            .eval_stateless(
                options,
                crate::EvalRequest {
                    raw_input: expr_text.to_string(),
                    parsed,
                    action: crate::EvalAction::Simplify,
                    auto_store: false,
                },
            )
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = output.result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "2 * tan(2 * x + 1) + (x * 4)/cos(2 * x + 1)^2"
        );
        assert!(
            output
                .domain_warnings
                .iter()
                .all(|warning| warning.rule_name != "Simplification Time Budget"),
            "unexpected timeout warning: {:?}",
            output.domain_warnings
        );
        let required_display: Vec<_> = output
            .required_conditions
            .iter()
            .map(|condition| condition.display(&engine.simplifier.context))
            .collect();
        assert_eq!(
            required_display
                .iter()
                .filter(|condition| condition.as_str() == "cos(2 * x + 1) ≠ 0")
                .count(),
            1,
            "expected exactly one displayed cos-domain condition, got: {required_display:?}"
        );
    }

    #[test]
    fn eval_simplify_steps_off_diff_shifted_linear_times_tan_keeps_product_rule_shape() {
        let cases = [
            (
                "diff((x+1)*tan(2*x+1), x)",
                "tan(2 * x + 1) + (2 * x + 2) / cos(2 * x + 1)^2",
            ),
            (
                "diff((3*x+2)*tan(2*x+1), x)",
                "3 * tan(2 * x + 1) + (6 * x + 4) / cos(2 * x + 1)^2",
            ),
        ];

        for (expr_text, expected) in cases {
            let mut engine = Engine::new();
            let parsed = parse(expr_text, &mut engine.simplifier.context)
                .unwrap_or_else(|e| panic!("{e:?}"));
            let mut options = crate::options::EvalOptions::default();
            options.steps_mode = crate::options::StepsMode::Off;
            options.shared.context_mode = crate::options::ContextMode::Standard;
            options.shared.semantics.domain_mode = crate::DomainMode::Generic;
            options.time_budget_ms = Some(50);

            let output = engine
                .eval_stateless(
                    options,
                    crate::EvalRequest {
                        raw_input: expr_text.to_string(),
                        parsed,
                        action: crate::EvalAction::Simplify,
                        auto_store: false,
                    },
                )
                .unwrap_or_else(|e| panic!("{e:?}"));

            let crate::EvalResult::Expr(result) = output.result else {
                panic!("expected expression result");
            };
            assert_eq!(
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: result,
                }
                .to_string(),
                expected,
                "unexpected result for {expr_text}"
            );
            assert!(
                output
                    .domain_warnings
                    .iter()
                    .all(|warning| warning.rule_name != "Simplification Time Budget"),
                "input {expr_text}: unexpected timeout warning: {:?}",
                output.domain_warnings
            );
            let required_display: Vec<_> = output
                .required_conditions
                .iter()
                .map(|condition| condition.display(&engine.simplifier.context))
                .collect();
            assert_eq!(
                required_display
                    .iter()
                    .filter(|condition| condition.as_str() == "cos(2 * x + 1) ≠ 0")
                    .count(),
                1,
                "input {expr_text}: expected exactly one displayed cos-domain condition, got: {required_display:?}"
            );
        }
    }

    #[test]
    fn eval_simplify_steps_off_diff_shifted_linear_times_tanh_avoids_timeout() {
        let cases = [
            (
                "diff((x+1)*tanh(2*x+1), x)",
                "tanh(2 * x + 1) + (2 * x + 2) / cosh(2 * x + 1)^2",
            ),
            (
                "diff((3*x+2)*tanh(2*x+1), x)",
                "3 * tanh(2 * x + 1) + (6 * x + 4) / cosh(2 * x + 1)^2",
            ),
        ];

        for (expr_text, expected) in cases {
            let mut engine = Engine::new();
            let parsed = parse(expr_text, &mut engine.simplifier.context)
                .unwrap_or_else(|e| panic!("{e:?}"));
            let mut options = crate::options::EvalOptions::default();
            options.steps_mode = crate::options::StepsMode::Off;
            options.shared.context_mode = crate::options::ContextMode::Standard;
            options.shared.semantics.domain_mode = crate::DomainMode::Generic;
            options.time_budget_ms = Some(50);

            let output = engine
                .eval_stateless(
                    options,
                    crate::EvalRequest {
                        raw_input: expr_text.to_string(),
                        parsed,
                        action: crate::EvalAction::Simplify,
                        auto_store: false,
                    },
                )
                .unwrap_or_else(|e| panic!("{e:?}"));

            let crate::EvalResult::Expr(result) = output.result else {
                panic!("expected expression result");
            };
            assert_eq!(
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: result,
                }
                .to_string(),
                expected,
                "unexpected result for {expr_text}"
            );
            assert!(
                output
                    .domain_warnings
                    .iter()
                    .all(|warning| warning.rule_name != "Simplification Time Budget"),
                "input {expr_text}: unexpected timeout warning: {:?}",
                output.domain_warnings
            );
            let required_display: Vec<_> = output
                .required_conditions
                .iter()
                .map(|condition| condition.display(&engine.simplifier.context))
                .collect();
            assert_eq!(
                required_display
                    .iter()
                    .filter(|condition| condition.as_str() == "cosh(2 * x + 1) ≠ 0")
                    .count(),
                1,
                "input {expr_text}: expected exactly one displayed cosh-domain condition, got: {required_display:?}"
            );
        }
    }

    #[test]
    fn eval_simplify_steps_off_diff_shifted_linear_times_cot_keeps_product_rule_shape() {
        let cases = [
            (
                "diff((x+1)*cot(2*x+1), x)",
                "cot(2 * x + 1) - (2 * x + 2) / sin(2 * x + 1)^2",
            ),
            (
                "diff((3*x+2)*cot(2*x+1), x)",
                "3 * cot(2 * x + 1) - (6 * x + 4) / sin(2 * x + 1)^2",
            ),
        ];

        for (expr_text, expected) in cases {
            let mut engine = Engine::new();
            let parsed = parse(expr_text, &mut engine.simplifier.context)
                .unwrap_or_else(|e| panic!("{e:?}"));
            let mut options = crate::options::EvalOptions::default();
            options.steps_mode = crate::options::StepsMode::Off;
            options.shared.context_mode = crate::options::ContextMode::Standard;
            options.shared.semantics.domain_mode = crate::DomainMode::Generic;
            options.time_budget_ms = Some(50);

            let output = engine
                .eval_stateless(
                    options,
                    crate::EvalRequest {
                        raw_input: expr_text.to_string(),
                        parsed,
                        action: crate::EvalAction::Simplify,
                        auto_store: false,
                    },
                )
                .unwrap_or_else(|e| panic!("{e:?}"));

            let crate::EvalResult::Expr(result) = output.result else {
                panic!("expected expression result");
            };
            assert_eq!(
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: result,
                }
                .to_string(),
                expected,
                "unexpected result for {expr_text}"
            );
            assert!(
                output
                    .domain_warnings
                    .iter()
                    .all(|warning| warning.rule_name != "Simplification Time Budget"),
                "input {expr_text}: unexpected timeout warning: {:?}",
                output.domain_warnings
            );
            let required_display: Vec<_> = output
                .required_conditions
                .iter()
                .map(|condition| condition.display(&engine.simplifier.context))
                .collect();
            assert_eq!(
                required_display
                    .iter()
                    .filter(|condition| condition.as_str() == "sin(2 * x + 1) ≠ 0")
                    .count(),
                1,
                "input {expr_text}: expected exactly one displayed sin-domain condition, got: {required_display:?}"
            );
        }
    }

    #[test]
    fn eval_simplify_steps_off_diff_shifted_linear_times_sec_csc_avoids_timeout() {
        let cases = [
            (
                "diff((x+1)*sec(2*x+1), x)",
                "(cos(2 * x + 1) + 2 * sin(2 * x + 1) + 2 * x * sin(2 * x + 1)) / cos(2 * x + 1)^2",
                "cos(2 * x + 1) ≠ 0",
            ),
            (
                "diff((3*x+2)*sec(2*x+1), x)",
                "(3 * cos(2 * x + 1) + 4 * sin(2 * x + 1) + 6 * x * sin(2 * x + 1)) / cos(2 * x + 1)^2",
                "cos(2 * x + 1) ≠ 0",
            ),
            (
                "diff((x+1)*csc(2*x+1), x)",
                "csc(2 * x + 1) - cos(2 * x + 1) * (2 * x + 2) / sin(2 * x + 1)^2",
                "sin(2 * x + 1) ≠ 0",
            ),
            (
                "diff((3*x+2)*csc(2*x+1), x)",
                "3 * csc(2 * x + 1) - cos(2 * x + 1) * (6 * x + 4) / sin(2 * x + 1)^2",
                "sin(2 * x + 1) ≠ 0",
            ),
        ];

        for (expr_text, expected, expected_condition) in cases {
            let mut engine = Engine::new();
            let parsed = parse(expr_text, &mut engine.simplifier.context)
                .unwrap_or_else(|e| panic!("{e:?}"));
            let mut options = crate::options::EvalOptions::default();
            options.steps_mode = crate::options::StepsMode::Off;
            options.shared.context_mode = crate::options::ContextMode::Standard;
            options.shared.semantics.domain_mode = crate::DomainMode::Generic;
            options.time_budget_ms = Some(500);

            let output = engine
                .eval_stateless(
                    options,
                    crate::EvalRequest {
                        raw_input: expr_text.to_string(),
                        parsed,
                        action: crate::EvalAction::Simplify,
                        auto_store: false,
                    },
                )
                .unwrap_or_else(|e| panic!("{e:?}"));

            let crate::EvalResult::Expr(result) = output.result else {
                panic!("expected expression result");
            };
            assert_eq!(
                DisplayExpr {
                    context: &engine.simplifier.context,
                    id: result,
                }
                .to_string(),
                expected,
                "unexpected result for {expr_text}"
            );
            assert!(
                output
                    .domain_warnings
                    .iter()
                    .all(|warning| warning.rule_name != "Simplification Time Budget"),
                "input {expr_text}: unexpected timeout warning: {:?}",
                output.domain_warnings
            );
            let required_display: Vec<_> = output
                .required_conditions
                .iter()
                .map(|condition| condition.display(&engine.simplifier.context))
                .collect();
            assert_eq!(
                required_display
                    .iter()
                    .filter(|condition| condition.as_str() == expected_condition)
                    .count(),
                1,
                "input {expr_text}: expected exactly one displayed domain condition, got: {required_display:?}"
            );
        }
    }

    #[test]
    fn eval_simplify_steps_off_handles_triple_sine_against_polynomial_partner_regression() {
        let expr_text =
            "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1)";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn with_profile_simplify_steps_off_handles_triple_sine_against_polynomial_partner_regression() {
        let expr_text =
            "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1)";
        let mut probe_engine = Engine::new();
        let probe_parsed = parse(expr_text, &mut probe_engine.simplifier.context)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        let effective_options = probe_engine.effective_options(&options, probe_parsed);
        let mut simplifier = crate::Simplifier::with_profile(&effective_options);
        simplifier.set_collect_steps(false);
        let parsed = parse(expr_text, &mut simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let (result, _steps) =
            simplifier.simplify_with_options(parsed, effective_options.to_simplify_options());
        assert_eq!(
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_triple_sine_against_polynomial_plus_rational_regression() {
        let expr_text =
            "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (x/(1 + x/(1-x)) - x + x^2)";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn with_profile_simplify_steps_off_handles_triple_sine_against_polynomial_plus_rational_regression(
    ) {
        let expr_text =
            "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (x/(1 + x/(1-x)) - x + x^2)";
        let mut probe_engine = Engine::new();
        let probe_parsed = parse(expr_text, &mut probe_engine.simplifier.context)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        let effective_options = probe_engine.effective_options(&options, probe_parsed);
        let mut simplifier = crate::Simplifier::with_profile(&effective_options);
        simplifier.set_collect_steps(false);
        let parsed = parse(expr_text, &mut simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let (result, _steps) =
            simplifier.simplify_with_options(parsed, effective_options.to_simplify_options());
        assert_eq!(
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_triple_sine_against_polynomial_plus_hyperbolic_regression() {
        let expr_text =
            "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn with_profile_simplify_steps_off_handles_triple_sine_against_polynomial_plus_hyperbolic_regression(
    ) {
        let expr_text =
            "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))";
        let mut probe_engine = Engine::new();
        let probe_parsed = parse(expr_text, &mut probe_engine.simplifier.context)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        let effective_options = probe_engine.effective_options(&options, probe_parsed);
        let mut simplifier = crate::Simplifier::with_profile(&effective_options);
        simplifier.set_collect_steps(false);
        let parsed = parse(expr_text, &mut simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let (result, _steps) =
            simplifier.simplify_with_options(parsed, effective_options.to_simplify_options());
        assert_eq!(
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_hyperbolic_sum_against_telescoping_sum_regression() {
        let expr_text =
            "(sinh(x+y) - (sinh(x)*cosh(y) + cosh(x)*sinh(y))) + (1/(u*(u+1)) - 1/u + 1/(u+1))";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn with_profile_simplify_steps_off_handles_hyperbolic_sum_against_telescoping_sum_regression() {
        let expr_text =
            "(sinh(x+y) - (sinh(x)*cosh(y) + cosh(x)*sinh(y))) + (1/(u*(u+1)) - 1/u + 1/(u+1))";
        let mut probe_engine = Engine::new();
        let probe_parsed = parse(expr_text, &mut probe_engine.simplifier.context)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        let effective_options = probe_engine.effective_options(&options, probe_parsed);
        let mut simplifier = crate::Simplifier::with_profile(&effective_options);
        simplifier.set_collect_steps(false);
        let parsed = parse(expr_text, &mut simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let (result, _steps) =
            simplifier.simplify_with_options(parsed, effective_options.to_simplify_options());
        assert_eq!(
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_trig_square_cube_substitution_pair_regression() {
        let expr_text = "(((sin(u)^2)^3 - 1) / ((sin(u)^2) - 1)) - ((sin(u)^4) + (sin(u)^2) + 1)";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_negative_double_cos_square_diff_passthrough_forward_regression(
    ) {
        let expr_text = "((sin(x)^2 - cos(x)^2) + m) - ((-cos(2*x)) + m)";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn with_profile_simplify_steps_off_handles_negative_double_cos_square_diff_passthrough_forward_regression(
    ) {
        let expr_text = "((sin(x)^2 - cos(x)^2) + m) - ((-cos(2*x)) + m)";
        let mut probe_engine = Engine::new();
        let probe_parsed = parse(expr_text, &mut probe_engine.simplifier.context)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        let effective_options = probe_engine.effective_options(&options, probe_parsed);
        let mut simplifier = crate::Simplifier::with_profile(&effective_options);
        simplifier.set_collect_steps(false);
        let parsed = parse(expr_text, &mut simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let (result, _steps) =
            simplifier.simplify_with_options(parsed, effective_options.to_simplify_options());
        assert_eq!(
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_squared_pythagorean_passthrough_forward_regression() {
        let expr_text = "(((sin(x)^2 + cos(x)^2)^2) + m) - (((1)^2) + m)";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_negative_double_sine_passthrough_forward_regression() {
        let expr_text = "((-2*sin(x)*cos(x)) + m) - ((-sin(2*x)) + m)";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_keeps_strict_solve_difference_of_cubes_fast_path() {
        let expr_text = "(x^3 - y^3) / (x - y)";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Solve;
        options.shared.semantics.domain_mode = crate::DomainMode::Strict;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        let result_str = DisplayExpr {
            context: &engine.simplifier.context,
            id: result,
        }
        .to_string();
        assert!(
            result_str == "x^2 + y^2 + x * y" || result_str == "x^2 + y^2 + y * x",
            "expected strict solve steps-off cubes fast path, got: {result_str}"
        );
    }

    #[test]
    fn with_profile_simplify_steps_off_handles_negative_double_sine_passthrough_forward_regression()
    {
        let expr_text = "((-2*sin(x)*cos(x)) + m) - ((-sin(2*x)) + m)";
        let mut probe_engine = Engine::new();
        let probe_parsed = parse(expr_text, &mut probe_engine.simplifier.context)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        let effective_options = probe_engine.effective_options(&options, probe_parsed);
        let mut simplifier = crate::Simplifier::with_profile(&effective_options);
        simplifier.set_collect_steps(false);
        let parsed = parse(expr_text, &mut simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let (result, _steps) =
            simplifier.simplify_with_options(parsed, effective_options.to_simplify_options());
        assert_eq!(
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_sophie_germain_passthrough_forward_regression() {
        let expr_text = "((x^4 + 4*y^4) + m) - (((x^2 - 2*x*y + 2*y^2)*(x^2 + 2*x*y + 2*y^2)) + m)";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_trig_ratio_alias_passthrough_forward_regression() {
        let expr_text = "((sin(2*x)/cos(x+x)) + m) - ((tan(2*x)) + m)";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_half_angle_tan_zero_difference_regression() {
        let expr_text = "(1 - cos(2*x))/sin(2*x) - tan(x)";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_morrie_scaled_difference_regression() {
        let expr_text = "k*(cos(x)*cos(2*x)*cos(4*x)) - k*(sin(8*x)/(8*sin(x)))";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_steps_off_handles_full_mixed_identity_regression() {
        let expr_text =
            "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (ln(sqrt((1+sin(y))/(1-sin(y)))) - atanh(sin(y))) + (x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))";
        let mut engine = Engine::new();
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;

        let (
            result,
            _warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn with_profile_simplify_steps_off_handles_full_mixed_identity_regression() {
        let expr_text =
            "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (ln(sqrt((1+sin(y))/(1-sin(y)))) - atanh(sin(y))) + (x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))";
        let mut probe_engine = Engine::new();
        let probe_parsed = parse(expr_text, &mut probe_engine.simplifier.context)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        let effective_options = probe_engine.effective_options(&options, probe_parsed);
        let mut simplifier = crate::Simplifier::with_profile(&effective_options);
        simplifier.set_collect_steps(false);
        let parsed = parse(expr_text, &mut simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let (result, _steps) =
            simplifier.simplify_with_options(parsed, effective_options.to_simplify_options());
        assert_eq!(
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn with_profile_simplify_steps_off_handles_polynomial_against_hyperbolic_pythagorean_regression(
    ) {
        let expr_text =
            "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))";
        let mut probe_engine = Engine::new();
        let probe_parsed = parse(expr_text, &mut probe_engine.simplifier.context)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        let effective_options = probe_engine.effective_options(&options, probe_parsed);
        let mut simplifier = crate::Simplifier::with_profile(&effective_options);
        simplifier.set_collect_steps(false);
        let parsed = parse(expr_text, &mut simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let (result, _steps) =
            simplifier.simplify_with_options(parsed, effective_options.to_simplify_options());
        assert_eq!(
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn with_default_rules_simplify_steps_off_handles_triple_sine_against_polynomial_partner_regression(
    ) {
        let expr_text =
            "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1)";
        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.set_collect_steps(false);
        let parsed = parse(expr_text, &mut simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        let (result, _steps) =
            simplifier.simplify_with_options(parsed, options.to_simplify_options());
        assert_eq!(
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn from_profile_simplify_steps_off_handles_triple_sine_plus_rational_against_hyperbolic_pythagorean_regression(
    ) {
        let expr_text = "(sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))";
        let mut probe_engine = Engine::new();
        let probe_parsed = parse(expr_text, &mut probe_engine.simplifier.context)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        let effective_options = probe_engine.effective_options(&options, probe_parsed);
        let mut cache = crate::profile_cache::ProfileCache::new();
        let profile = cache.get_or_build(&effective_options);
        let mut simplifier =
            crate::Simplifier::from_profile_with_context(profile, cas_ast::Context::new());
        simplifier.set_collect_steps(false);
        let parsed = parse(expr_text, &mut simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let (result, _steps) =
            simplifier.simplify_with_options(parsed, effective_options.to_simplify_options());
        assert_eq!(
            DisplayExpr {
                context: &simplifier.context,
                id: result,
            }
            .to_string(),
            "0"
        );
    }

    #[test]
    fn eval_simplify_surfaces_partial_result_warning_when_time_budget_is_hit() {
        let mut engine = Engine::new();
        let expr_text = "a + b";
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.time_budget_ms = Some(0);

        let (
            result,
            warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        assert_eq!(
            DisplayExpr {
                context: &engine.simplifier.context,
                id: result,
            }
            .to_string(),
            "a + b"
        );
        assert!(
            warnings.iter().any(|warning| {
                warning.rule_name == "Simplification Time Budget"
                    && warning.message.contains("Partial result")
            }),
            "expected partial-result timeout warning, got: {warnings:?}"
        );
    }

    #[test]
    fn eval_simplify_surfaces_partial_result_warning_when_root_shortcut_nested_timeout_is_hit() {
        let mut engine = Engine::new();
        let expr_text = "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (log(sqrt((1+sin(y))/(1-sin(y)))) - atanh(sin(y))) + (x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))";
        let parsed =
            parse(expr_text, &mut engine.simplifier.context).unwrap_or_else(|e| panic!("{e:?}"));
        let mut options = crate::options::EvalOptions::default();
        options.steps_mode = crate::options::StepsMode::Off;
        options.shared.context_mode = crate::options::ContextMode::Standard;
        options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        options.time_budget_ms = Some(1);

        let (
            result,
            warnings,
            _steps,
            _solve_steps,
            _assumptions,
            _scopes,
            _required,
            _rewrite_required,
        ) = engine
            .eval_simplify(&options, parsed)
            .unwrap_or_else(|e| panic!("{e:?}"));

        let crate::EvalResult::Expr(result) = result else {
            panic!("expected expression result");
        };
        let rendered = DisplayExpr {
            context: &engine.simplifier.context,
            id: result,
        }
        .to_string();
        assert!(
            !rendered.is_empty(),
            "expected a partial result to be returned before timeout"
        );
        assert!(
            warnings.iter().any(|warning| {
                warning.rule_name == "Simplification Time Budget"
                    && warning.message.contains("Partial result")
            }),
            "expected nested root-shortcut timeout warning, got: {warnings:?}"
        );
    }
}
