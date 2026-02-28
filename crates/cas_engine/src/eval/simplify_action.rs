//! Simplify action handler for `Engine::eval`.

use super::*;

impl Engine {
    /// Handle `EvalAction::Expand`.
    ///
    /// Actualmente usa el mismo simplificador principal; se mantiene separado
    /// para que el dispatch quede desacoplado del detalle de implementación.
    pub(super) fn eval_expand(&mut self, resolved: ExprId) -> Result<ActionResult, anyhow::Error> {
        let (res, steps) = self.simplifier.simplify(resolved);
        let warnings = collect_domain_warnings(&steps);
        Ok((
            EvalResult::Expr(res),
            warnings,
            steps,
            vec![],
            vec![],
            vec![],
            vec![],
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

        let mut ctx_simplifier = Simplifier::from_profile(profile);
        ctx_simplifier.context = std::mem::take(&mut self.simplifier.context);

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
                        )
                    }
                    _ => (resolved, Vec::new()),
                }
            } else {
                (resolved, Vec::new())
            };

        let (mut res, mut steps, _stats) =
            ctx_simplifier.simplify_with_stats(expr_to_simplify, simplify_opts);

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

        self.simplifier
            .extend_blocked_hints(ctx_simplifier.take_blocked_hints());
        self.simplifier.context = ctx_simplifier.context;

        {
            use crate::implicit_domain::{
                classify_assumptions_in_place, infer_implicit_domain, DomainContext,
            };

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
        }

        let mut warnings = collect_domain_warnings(&steps);

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
        ))
    }
}
