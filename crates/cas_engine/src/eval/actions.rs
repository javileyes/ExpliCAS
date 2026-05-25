//! Action handlers used by `Engine::eval` dispatch.
//!
//! This keeps non-simplify actions (solve/equiv/limit) out of the main
//! dispatch orchestrator so the pipeline flow stays compact.

use super::*;

fn is_simple_negative_var_affine_witness(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var: ExprId,
    var_name: &str,
) -> bool {
    match ctx.get(expr) {
        cas_ast::Expr::Neg(inner) => *inner == var,
        cas_ast::Expr::Sub(left, right) => {
            *right == var && !cas_math::expr_predicates::contains_named_var(ctx, *left, var_name)
        }
        cas_ast::Expr::Add(left, right) => {
            (is_simple_negative_var_affine_witness(ctx, *left, var, var_name)
                && !cas_math::expr_predicates::contains_named_var(ctx, *right, var_name))
                || (is_simple_negative_var_affine_witness(ctx, *right, var, var_name)
                    && !cas_math::expr_predicates::contains_named_var(ctx, *left, var_name))
        }
        _ => false,
    }
}

fn is_simple_positive_var_affine_witness(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var: ExprId,
    var_name: &str,
) -> bool {
    match ctx.get(expr) {
        _ if expr == var => true,
        cas_ast::Expr::Add(left, right) => {
            (*left == var && !cas_math::expr_predicates::contains_named_var(ctx, *right, var_name))
                || (*right == var
                    && !cas_math::expr_predicates::contains_named_var(ctx, *left, var_name))
        }
        cas_ast::Expr::Sub(left, right) => {
            *left == var && !cas_math::expr_predicates::contains_named_var(ctx, *right, var_name)
        }
        _ => false,
    }
}

fn limit_domain_path_warning(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var: ExprId,
    var_name: &str,
    approach: crate::limits::Approach,
) -> Option<DomainWarning> {
    let input_domain =
        crate::infer_implicit_domain(ctx, expr, crate::semantics::ValueDomain::RealOnly);

    let required = match approach {
        crate::limits::Approach::NegInfinity => input_domain.conditions().iter().find_map(|cond| {
            let witness = match cond {
                crate::ImplicitCondition::Positive(witness)
                | crate::ImplicitCondition::NonNegative(witness) => *witness,
                crate::ImplicitCondition::LowerBound(_, _) => return None,
                crate::ImplicitCondition::NonZero(_) => return None,
            };

            is_simple_positive_var_affine_witness(ctx, witness, var, var_name)
                .then(|| cond.display(ctx))
        }),
        crate::limits::Approach::PosInfinity => input_domain.conditions().iter().find_map(|cond| {
            let witness = match cond {
                crate::ImplicitCondition::Positive(witness)
                | crate::ImplicitCondition::NonNegative(witness) => *witness,
                crate::ImplicitCondition::LowerBound(_, _) => return None,
                crate::ImplicitCondition::NonZero(_) => return None,
            };

            is_simple_negative_var_affine_witness(ctx, witness, var, var_name)
                .then(|| cond.display(ctx))
        }),
        crate::limits::Approach::Finite(_) => return None,
    }?;

    let approach_display = match approach {
        crate::limits::Approach::NegInfinity => "-infinity",
        crate::limits::Approach::PosInfinity => "infinity",
        crate::limits::Approach::Finite(_) => return None,
    };

    Some(DomainWarning {
        message: format!(
            "Limit path conflicts with the input domain: {var_name} -> {approach_display} while the expression requires {required}"
        ),
        rule_name: "Limit Domain Path".to_string(),
    })
}

impl Engine {
    /// Route one already-resolved action to its concrete handler.
    pub(super) fn dispatch_eval_action(
        &mut self,
        options: &crate::options::EvalOptions,
        action: EvalAction,
        resolved: ExprId,
        resolved_equiv_other: Option<ExprId>,
    ) -> Result<ActionResult, anyhow::Error> {
        match action {
            EvalAction::Simplify => self.eval_simplify(options, resolved),
            EvalAction::Expand => self.eval_expand(options, resolved),
            EvalAction::Solve { var } => self.eval_solve(options, resolved, &var),
            EvalAction::Equiv { .. } => {
                let resolved_other = resolved_equiv_other
                    .ok_or_else(|| anyhow::anyhow!("Missing resolved equivalence operand"))?;
                self.eval_equiv(resolved, resolved_other)
            }
            EvalAction::Limit { var, approach } => {
                self.eval_limit(options, resolved, &var, approach)
            }
        }
    }

    /// Handle `EvalAction::Solve`: equation construction, solver invocation.
    pub(super) fn eval_solve(
        &mut self,
        options: &crate::options::EvalOptions,
        resolved: ExprId,
        var: &str,
    ) -> Result<ActionResult, anyhow::Error> {
        let eq_to_solve = cas_solver_core::solve_entry::equation_from_expr_or_zero(
            &mut self.simplifier.context,
            resolved,
        );

        let solver_opts = cas_solver_core::solver_options::SolverOptions::from_eval_config(
            options.shared.semantics,
            options.budget,
        );

        let sol_result = crate::api::solve_with_display_steps(
            &eq_to_solve,
            var,
            &mut self.simplifier,
            solver_opts,
        );

        match sol_result {
            Ok((solution_set, display_steps, diagnostics)) => {
                let solve_steps = display_steps.0;

                let solver_assumptions =
                    if options.shared.assumption_reporting == crate::AssumptionReporting::Off {
                        vec![]
                    } else {
                        diagnostics.assumed_records.clone()
                    };

                let warnings: Vec<DomainWarning> = vec![];
                let eval_res = EvalResult::SolutionSet(solution_set);
                let output_scopes = diagnostics.output_scopes;
                let solver_required = diagnostics.required;

                Ok((
                    eval_res,
                    warnings,
                    vec![],
                    solve_steps,
                    solver_assumptions,
                    output_scopes,
                    solver_required,
                    vec![],
                ))
            }
            Err(e) => Err(anyhow::anyhow!("Solver error: {}", e)),
        }
    }

    /// Handle `EvalAction::Equiv`: resolve other expression and check equivalence.
    pub(super) fn eval_equiv(
        &mut self,
        resolved: ExprId,
        resolved_other: ExprId,
    ) -> Result<ActionResult, anyhow::Error> {
        let are_eq = self.simplifier.are_equivalent(resolved, resolved_other);
        let rhs_domain = crate::infer_implicit_domain(
            &self.simplifier.context,
            resolved_other,
            crate::semantics::ValueDomain::RealOnly,
        );
        let solver_required = rhs_domain.conditions().iter().cloned().collect();
        Ok((
            EvalResult::Bool(are_eq),
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            solver_required,
            vec![],
        ))
    }

    /// Handle `EvalAction::Limit`: compute limit using the limits engine.
    pub(super) fn eval_limit(
        &mut self,
        options: &crate::options::EvalOptions,
        resolved: ExprId,
        var: &str,
        approach: crate::limits::Approach,
    ) -> Result<ActionResult, anyhow::Error> {
        use crate::limits::{limit, LimitOptions};

        let var_id = self.simplifier.context.var(var);
        let opts = LimitOptions::default();
        let mut budget = crate::budget::Budget::preset_cli();

        match limit(
            &mut self.simplifier.context,
            resolved,
            var_id,
            approach,
            &opts,
            &mut budget,
        ) {
            Ok(result) => {
                let mut steps = Vec::new();
                if !matches!(options.steps_mode, crate::options::StepsMode::Off) {
                    let (rule_name, description) = if result.warning.is_some() {
                        (
                            "Conservar límite residual",
                            "Conservar el límite sin resolver porque la política segura no lo decide",
                        )
                    } else {
                        match approach {
                            crate::limits::Approach::Finite(_) => (
                                "Evaluar límite finito",
                                "Evaluar el límite finito con política conservadora",
                            ),
                            crate::limits::Approach::PosInfinity
                            | crate::limits::Approach::NegInfinity => (
                                "Evaluar límite en infinito",
                                "Evaluar el límite en infinito con política conservadora",
                            ),
                        }
                    };
                    let mut step = crate::Step::new(
                        description,
                        rule_name,
                        resolved,
                        result.expr,
                        Vec::new(),
                        Some(&self.simplifier.context),
                    );
                    step.importance = crate::ImportanceLevel::Medium;
                    step.category = crate::StepCategory::Limits;
                    steps.push(step);
                }
                let mut warnings: Vec<DomainWarning> = result
                    .warning
                    .into_iter()
                    .map(|message| DomainWarning {
                        message,
                        rule_name: "Limit Evaluation".to_string(),
                    })
                    .collect();
                if !warnings.is_empty() {
                    if let Some(warning) = limit_domain_path_warning(
                        &self.simplifier.context,
                        resolved,
                        var_id,
                        var,
                        approach,
                    ) {
                        warnings.push(warning);
                    }
                }
                Ok((
                    EvalResult::Expr(result.expr),
                    warnings,
                    steps,
                    vec![],
                    vec![],
                    vec![],
                    vec![],
                    vec![],
                ))
            }
            Err(e) => Err(anyhow::anyhow!("Limit error: {}", e)),
        }
    }
}
