//! Action handlers used by `Engine::eval` dispatch.
//!
//! This keeps non-simplify actions (solve/equiv/limit) out of the main
//! dispatch orchestrator so the pipeline flow stays compact.

use super::*;
use cas_math::limit_types::FiniteLimitSide;
use cas_math::polynomial::Polynomial;
use cas_solver_core::rule_names::RULE_CONSERVAR_LIMITE_RESIDUAL;
use num_rational::BigRational;
use num_traits::{Signed, Zero};

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

fn polynomial_local_order_and_derivative(
    polynomial: &Polynomial,
    point: &BigRational,
) -> Option<(usize, BigRational)> {
    let mut current = polynomial.clone();
    for order in 0..=polynomial.degree() {
        let value = current.eval(point);
        if !value.is_zero() {
            return Some((order, value));
        }
        current = current.derivative();
    }
    None
}

fn finite_one_sided_polynomial_tail_is_negative(
    polynomial: &Polynomial,
    point: &BigRational,
    side: FiniteLimitSide,
) -> Option<bool> {
    let (order, derivative_value) = polynomial_local_order_and_derivative(polynomial, point)?;
    Some(
        derivative_value.is_positive()
            != (side == FiniteLimitSide::Right || order.is_multiple_of(2)),
    )
}

fn finite_one_sided_rational_tail_is_negative(
    ctx: &cas_ast::Context,
    witness: ExprId,
    var_name: &str,
    point: &BigRational,
    side: FiniteLimitSide,
) -> Option<bool> {
    let cas_ast::Expr::Div(num, den) = ctx.get(witness).clone() else {
        return None;
    };
    let numerator = Polynomial::from_expr(ctx, num, var_name).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var_name).ok()?;
    let denominator_value = denominator.eval(point);
    if denominator_value.is_zero() {
        return None;
    }
    let numerator_tail_negative =
        finite_one_sided_polynomial_tail_is_negative(&numerator, point, side)?;
    Some(numerator_tail_negative ^ denominator_value.is_negative())
}

fn shifted_domain_witness_polynomial(
    ctx: &cas_ast::Context,
    witness: ExprId,
    var_name: &str,
    lower_bound: &BigRational,
) -> Option<Polynomial> {
    let poly = Polynomial::from_expr(ctx, witness, var_name).ok()?;
    if lower_bound.is_zero() {
        return Some(poly);
    }

    let mut coeffs = poly.coeffs;
    if coeffs.is_empty() {
        coeffs.push(-lower_bound.clone());
    } else {
        coeffs[0] -= lower_bound;
    }
    Some(Polynomial::new(coeffs, poly.var))
}

fn finite_one_sided_domain_path_conflict(
    ctx: &cas_ast::Context,
    cond: &crate::ImplicitCondition,
    var_name: &str,
    point_value: &BigRational,
    side: FiniteLimitSide,
) -> bool {
    let (witness, lower_bound, strict_positive) = match cond {
        crate::ImplicitCondition::Positive(witness) => (*witness, BigRational::zero(), true),
        crate::ImplicitCondition::NonNegative(witness) => (*witness, BigRational::zero(), false),
        crate::ImplicitCondition::LowerBound(witness, lower_bound) => {
            (*witness, lower_bound.clone(), false)
        }
        crate::ImplicitCondition::NonZero(_) => return false,
    };

    let Some(poly) = shifted_domain_witness_polynomial(ctx, witness, var_name, &lower_bound) else {
        return lower_bound.is_zero()
            && finite_one_sided_rational_tail_is_negative(
                ctx,
                witness,
                var_name,
                point_value,
                side,
            )
            .unwrap_or(false);
    };
    let value_at_point = poly.eval(point_value);
    if value_at_point.is_negative() {
        return true;
    }
    if value_at_point.is_positive() {
        return false;
    }
    if poly.is_zero() {
        return strict_positive;
    }

    finite_one_sided_polynomial_tail_is_negative(&poly, point_value, side).unwrap_or(false)
}

fn finite_one_sided_approach_display(
    ctx: &cas_ast::Context,
    point: ExprId,
    side: FiniteLimitSide,
) -> String {
    let point_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: point,
        }
    );
    match side {
        FiniteLimitSide::Left => format!("{point_display} from the left"),
        FiniteLimitSide::Right => format!("{point_display} from the right"),
    }
}

fn finite_one_sided_inverse_interval_condition(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<crate::ImplicitCondition> {
    let cas_ast::Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let one = ctx.num(1);
    let two = ctx.num(2);
    let square = ctx.add(cas_ast::Expr::Pow(args[0], two));
    let bounded = ctx.add(cas_ast::Expr::Sub(one, square));
    match ctx.builtin_of(fn_id) {
        Some(
            cas_ast::BuiltinFn::Arcsin
            | cas_ast::BuiltinFn::Asin
            | cas_ast::BuiltinFn::Arccos
            | cas_ast::BuiltinFn::Acos,
        ) => Some(crate::ImplicitCondition::NonNegative(bounded)),
        Some(cas_ast::BuiltinFn::Atanh) => Some(crate::ImplicitCondition::Positive(bounded)),
        _ => None,
    }
}

fn cleanup_residual_limit_output_expr(ctx: &mut cas_ast::Context, expr: ExprId) -> ExprId {
    match ctx.get(expr).clone() {
        cas_ast::Expr::Add(lhs, rhs) => {
            let lhs = cleanup_residual_limit_output_expr(ctx, lhs);
            let rhs = cleanup_residual_limit_output_expr(ctx, rhs);
            if cas_math::expr_predicates::is_zero_expr(ctx, lhs) {
                rhs
            } else if cas_math::expr_predicates::is_zero_expr(ctx, rhs) {
                lhs
            } else {
                ctx.add(cas_ast::Expr::Add(lhs, rhs))
            }
        }
        cas_ast::Expr::Sub(lhs, rhs) => {
            let lhs = cleanup_residual_limit_output_expr(ctx, lhs);
            let rhs = cleanup_residual_limit_output_expr(ctx, rhs);
            if cas_math::expr_predicates::is_zero_expr(ctx, rhs) {
                lhs
            } else {
                ctx.add(cas_ast::Expr::Sub(lhs, rhs))
            }
        }
        cas_ast::Expr::Mul(lhs, rhs) => {
            let lhs = cleanup_residual_limit_output_expr(ctx, lhs);
            let rhs = cleanup_residual_limit_output_expr(ctx, rhs);
            if cas_math::expr_predicates::is_one_expr(ctx, lhs) {
                rhs
            } else if cas_math::expr_predicates::is_one_expr(ctx, rhs) {
                lhs
            } else {
                ctx.add(cas_ast::Expr::Mul(lhs, rhs))
            }
        }
        cas_ast::Expr::Div(lhs, rhs) => {
            let lhs = cleanup_residual_limit_output_expr(ctx, lhs);
            let rhs = cleanup_residual_limit_output_expr(ctx, rhs);
            if cas_math::expr_predicates::is_one_expr(ctx, rhs) {
                lhs
            } else {
                ctx.add(cas_ast::Expr::Div(lhs, rhs))
            }
        }
        cas_ast::Expr::Pow(base, exp) => {
            let base = cleanup_residual_limit_output_expr(ctx, base);
            let exp = cleanup_residual_limit_output_expr(ctx, exp);
            ctx.add(cas_ast::Expr::Pow(base, exp))
        }
        cas_ast::Expr::Neg(inner) => {
            let inner = cleanup_residual_limit_output_expr(ctx, inner);
            if cas_math::expr_predicates::is_zero_expr(ctx, inner) {
                inner
            } else {
                ctx.add(cas_ast::Expr::Neg(inner))
            }
        }
        cas_ast::Expr::Function(fn_id, args) => {
            let args = args
                .into_iter()
                .map(|arg| cleanup_residual_limit_output_expr(ctx, arg))
                .collect();
            ctx.add(cas_ast::Expr::Function(fn_id, args))
        }
        cas_ast::Expr::Matrix { rows, cols, data } => {
            let data = data
                .into_iter()
                .map(|item| cleanup_residual_limit_output_expr(ctx, item))
                .collect();
            ctx.add(cas_ast::Expr::Matrix { rows, cols, data })
        }
        cas_ast::Expr::Hold(inner) => {
            let inner = cleanup_residual_limit_output_expr(ctx, inner);
            ctx.add(cas_ast::Expr::Hold(inner))
        }
        cas_ast::Expr::Number(_)
        | cas_ast::Expr::Constant(_)
        | cas_ast::Expr::Variable(_)
        | cas_ast::Expr::SessionRef(_) => expr,
    }
}

fn limit_domain_path_warning(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    var: ExprId,
    var_name: &str,
    approach: crate::limits::Approach,
) -> Option<DomainWarning> {
    let input_domain =
        crate::infer_implicit_domain(ctx, expr, crate::semantics::ValueDomain::RealOnly);

    let required = match approach {
        crate::limits::Approach::FiniteOneSided(point, side) => {
            let cas_ast::Expr::Number(point_value) = ctx.get(point) else {
                return None;
            };
            let point_value = point_value.clone();
            let mut conditions: Vec<_> = input_domain.conditions().iter().cloned().collect();
            if let Some(cond) = finite_one_sided_inverse_interval_condition(ctx, expr) {
                conditions.push(cond);
            }
            conditions.iter().find_map(|cond| {
                finite_one_sided_domain_path_conflict(ctx, cond, var_name, &point_value, side)
                    .then(|| cond.display(ctx))
            })
        }
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
        crate::limits::Approach::FiniteOneSided(point, side) => {
            return Some(DomainWarning {
                message: format!(
                    "Limit path conflicts with the input domain: {var_name} -> {} while the expression requires {required}",
                    finite_one_sided_approach_display(ctx, point, side)
                ),
                rule_name: "Limit Domain Path".to_string(),
            });
        }
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
                self.eval_equiv(options, resolved, resolved_other)
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
        options: &crate::options::EvalOptions,
        resolved: ExprId,
        resolved_other: ExprId,
    ) -> Result<ActionResult, anyhow::Error> {
        // Soundness: `equiv` must report `true` only from an EXACT symbolic proof,
        // never a numeric coincidence. `are_equivalent`'s numeric fallback confirms
        // equivalence from a SINGLE f64 probe — every variable gets the SAME value
        // (collapsing `equiv(x,y)` onto the diagonal x=y) under an absolute 1e-9
        // epsilon (so `equiv(x/1e12,0)` and probe-tangent pairs read as zero) — which
        // certifies non-equivalent expressions as equal, violating the project rule
        // that soundness gates must be exact. So: (1) disable numeric confirmation —
        // a probe may never CONFIRM equivalence; (2) recover genuine identities the
        // raw simplifier leaves in a non-cancelling form (e.g. derivative identities
        // like d/dx √(sec x) = √(sec x)·tan(x)/2) by checking whether the FULL
        // evaluator reduces `a - b` to exactly 0 — an exact symbolic zero, not numeric.
        let saved_numeric = self.simplifier.allow_numerical_verification;
        self.simplifier.allow_numerical_verification = false;
        let are_eq = self.simplifier.are_equivalent(resolved, resolved_other)
            || self.equiv_difference_evaluates_to_zero(options, resolved, resolved_other);
        self.simplifier.allow_numerical_verification = saved_numeric;
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

    /// True when the FULL evaluator reduces `a - b` to exactly `0`.
    ///
    /// This is the sound, exact recovery path for `equiv`: the bare simplifier can
    /// leave an identity in a non-cancelling form (e.g. it differentiates
    /// `sqrt(sec(x))` to `sec(x)·tan(x)/(2·sqrt(sec(x)))` without collapsing
    /// `sec(x)/sqrt(sec(x))` to `sqrt(sec(x))`), so `are_equivalent` misses it; the
    /// `eval_simplify` pipeline applies the extra normalization that reduces the
    /// difference to `0`. The result is a symbolic `Number(0)` — never a numeric
    /// probe — so it stays an exact equivalence gate.
    fn equiv_difference_evaluates_to_zero(
        &mut self,
        options: &crate::options::EvalOptions,
        a: ExprId,
        b: ExprId,
    ) -> bool {
        let diff = self.simplifier.context.add(cas_ast::Expr::Sub(a, b));
        match self.eval_simplify(options, diff) {
            Ok((EvalResult::Expr(result), ..)) => {
                matches!(self.simplifier.context.get(result), cas_ast::Expr::Number(n) if n.is_zero())
            }
            _ => false,
        }
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
                let folded = cas_math::infinity_support::fold_infinity_saturation(
                    &mut self.simplifier.context,
                    result.expr,
                );
                let output_expr = if result.warning.is_some() {
                    cleanup_residual_limit_output_expr(&mut self.simplifier.context, folded)
                } else {
                    folded
                };
                let mut steps = Vec::new();
                if !matches!(options.steps_mode, crate::options::StepsMode::Off) {
                    let (rule_name, description) = if result.warning.is_some() {
                        (
                            RULE_CONSERVAR_LIMITE_RESIDUAL,
                            "Conservar el límite sin resolver porque la política segura no lo decide",
                        )
                    } else {
                        match approach {
                            crate::limits::Approach::Finite(_) => (
                                "Evaluar límite finito",
                                "Evaluar el límite finito con política conservadora",
                            ),
                            crate::limits::Approach::FiniteOneSided(_, _) => (
                                "Evaluar límite unilateral finito",
                                "Evaluar el límite finito unilateral con política conservadora",
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
                        output_expr,
                        Vec::new(),
                        Some(&self.simplifier.context),
                    );
                    step.importance = crate::ImportanceLevel::Medium;
                    step.category = crate::StepCategory::Limits;
                    // Record the approached point for finite limits so didactic narration can
                    // soundly recognise an indeterminate 0/0 at that point.
                    if let crate::limits::Approach::Finite(point) = approach {
                        step.meta_mut().limit_point = Some(point);
                    }
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
                        &mut self.simplifier.context,
                        resolved,
                        var_id,
                        var,
                        approach,
                    ) {
                        warnings.push(warning);
                    }
                }
                Ok((
                    EvalResult::Expr(output_expr),
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
