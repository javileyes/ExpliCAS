use cas_api_models::{
    AssumptionDto, BlockedHintDto, EvalLimitApproach, EvalSpecialCommand, RequiredConditionWire,
    WarningWire,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_formatter::DisplayExpr;
use cas_solver_core::domain_normalization::normalize_and_dedupe_conditions;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use std::collections::HashSet;

use crate::eval_output_condition_filter::AssumedConditionFilter;

pub(crate) fn collect_output_warnings(
    domain_warnings: &[crate::DomainWarning],
    assumptions_used: &[AssumptionDto],
) -> Vec<WarningWire> {
    let assumed_display: HashSet<&str> = assumptions_used
        .iter()
        .map(|assumption| assumption.display.as_str())
        .collect();
    domain_warnings
        .iter()
        .filter(|w| !assumed_display.contains(w.message.as_str()))
        .map(|w| WarningWire {
            rule: w.rule_name.clone(),
            assumption: w.message.clone(),
        })
        .collect()
}

pub(crate) fn collect_output_required_conditions(
    required_conditions: &[crate::ImplicitCondition],
    ctx: &mut Context,
    assumptions_used: &[AssumptionDto],
    raw_input: &str,
    result_display: Option<&str>,
) -> Vec<RequiredConditionWire> {
    let assumed_filter = AssumedConditionFilter::from_assumptions(assumptions_used);
    let normalized = normalize_and_dedupe_conditions(ctx, required_conditions);
    let visible_conditions = visible_required_conditions_after_public_suppression(
        ctx,
        &normalized,
        &assumed_filter,
        raw_input,
        result_display,
    );

    let wires: Vec<_> = visible_conditions
        .iter()
        .filter(|cond| !required_condition_wire_is_redundant(ctx, cond, &visible_conditions))
        .filter(|cond| !assumed_filter.covers_required_condition(ctx, cond))
        .map(|cond| {
            let cond = *cond;
            let (kind, expr_id) = match cond {
                crate::ImplicitCondition::NonNegative(e) => ("NonNegative", *e),
                crate::ImplicitCondition::LowerBound(e, _) => ("LowerBound", *e),
                crate::ImplicitCondition::Positive(e) => ("Positive", *e),
                crate::ImplicitCondition::NonZero(e) => ("NonZero", *e),
            };
            let expr_str = expr_display(ctx, expr_id);
            let expr_display =
                apply_input_inverse_trig_alias_preferences(&expr_str, raw_input, result_display);
            RequiredConditionWire {
                kind: kind.to_string(),
                expr_display,
                expr_canonical: expr_str,
            }
        })
        .collect();
    dedupe_sqrt_half_power_condition_wires(wires)
}

fn required_condition_wire_is_redundant(
    ctx: &Context,
    cond: &crate::ImplicitCondition,
    visible_conditions: &[&crate::ImplicitCondition],
) -> bool {
    let crate::ImplicitCondition::NonZero(nonzero_expr) = cond else {
        return matches!(cond, crate::ImplicitCondition::Positive(positive_expr) if
            reciprocal_trig_log_positive_quotient_condition_is_redundant(
                ctx,
                *positive_expr,
                visible_conditions
            )
        );
    };

    if calculus_nonzero_condition_is_redundant(ctx, *nonzero_expr, visible_conditions) {
        return true;
    }
    if sqrt_like_unary_nonzero_condition_is_redundant(ctx, *nonzero_expr, visible_conditions) {
        return true;
    }

    reciprocal_trig_log_argument_condition_is_redundant(ctx, *nonzero_expr, visible_conditions)
}

pub(crate) fn collect_output_required_display(
    required_conditions: &[crate::ImplicitCondition],
    ctx: &mut Context,
    assumptions_used: &[AssumptionDto],
    raw_input: &str,
    result_display: Option<&str>,
) -> Vec<String> {
    let assumed_filter = AssumedConditionFilter::from_assumptions(assumptions_used);
    let normalized = normalize_and_dedupe_conditions(ctx, required_conditions);
    let visible_conditions = visible_required_conditions_after_public_suppression(
        ctx,
        &normalized,
        &assumed_filter,
        raw_input,
        result_display,
    );

    let displays: Vec<_> = visible_conditions
        .iter()
        .filter(|cond| !required_condition_is_redundant(ctx, cond, &visible_conditions))
        .map(|cond| {
            apply_input_inverse_trig_alias_preferences(
                &cond.display(ctx),
                raw_input,
                result_display,
            )
        })
        .collect();
    dedupe_sqrt_half_power_required_displays(displays)
}

fn dedupe_sqrt_half_power_condition_wires(
    wires: Vec<RequiredConditionWire>,
) -> Vec<RequiredConditionWire> {
    wires
        .iter()
        .enumerate()
        .filter_map(|(idx, wire)| {
            let key = sqrt_half_power_display_key(&wire.expr_display);
            let zero_set_key = sqrt_half_power_zero_set_display_key(&wire.expr_display);
            let has_prior_equivalent_half_power_display = wires.iter().take(idx).any(|other| {
                other.kind == wire.kind && sqrt_half_power_display_key(&other.expr_display) == key
            });
            let has_prior_equivalent_zero_set_display = zero_set_key.as_ref().is_some_and(|key| {
                wires.iter().take(idx).any(|other| {
                    other.kind == wire.kind
                        && sqrt_half_power_zero_set_display_key(&other.expr_display).as_ref()
                            == Some(key)
                })
            });
            let has_preferred_sqrt_display = wires.iter().enumerate().any(|(other_idx, other)| {
                other_idx != idx
                    && other.kind == wire.kind
                    && other.expr_display.contains("sqrt(")
                    && (sqrt_half_power_display_key(&other.expr_display) == key
                        || sqrt_half_power_zero_set_display_key(&other.expr_display).is_some_and(
                            |other_key| zero_set_key.as_ref().is_some_and(|key| other_key == *key),
                        ))
            });
            (!has_prior_equivalent_zero_set_display
                && !(display_contains_half_power(&wire.expr_display)
                    && (has_preferred_sqrt_display || has_prior_equivalent_half_power_display)))
                .then_some(wire.clone())
        })
        .collect()
}

fn dedupe_sqrt_half_power_required_displays(displays: Vec<String>) -> Vec<String> {
    displays
        .iter()
        .enumerate()
        .filter_map(|(idx, display)| {
            let key = sqrt_half_power_display_key(display);
            let zero_set_key = sqrt_half_power_zero_set_display_key(display);
            let has_prior_equivalent_half_power_display = displays
                .iter()
                .take(idx)
                .any(|other| sqrt_half_power_display_key(other) == key);
            let has_prior_equivalent_zero_set_display = zero_set_key.as_ref().is_some_and(|key| {
                displays
                    .iter()
                    .take(idx)
                    .any(|other| sqrt_half_power_zero_set_display_key(other).as_ref() == Some(key))
            });
            let has_preferred_sqrt_display =
                displays.iter().enumerate().any(|(other_idx, other)| {
                    other_idx != idx
                        && other.contains("sqrt(")
                        && (sqrt_half_power_display_key(other) == key
                            || sqrt_half_power_zero_set_display_key(other).is_some_and(
                                |other_key| {
                                    zero_set_key.as_ref().is_some_and(|key| other_key == *key)
                                },
                            ))
                });
            (!has_prior_equivalent_zero_set_display
                && !(display_contains_half_power(display)
                    && (has_preferred_sqrt_display || has_prior_equivalent_half_power_display)))
                .then(|| display.clone())
        })
        .collect()
}

fn display_contains_half_power(display: &str) -> bool {
    display.contains("^(1/2)") || display.contains("^(1 / 2)")
}

fn sqrt_half_power_display_key(display: &str) -> String {
    let mut normalized = display.to_string();
    while let Some((power_start, power_len)) = next_half_power_display(&normalized) {
        let Some((base_start, base_text)) =
            display_base_before_half_power(&normalized, power_start)
        else {
            break;
        };
        let replacement = format!("sqrt({base_text})");
        normalized.replace_range(base_start..power_start + power_len, &replacement);
    }
    normalized
}

fn next_half_power_display(display: &str) -> Option<(usize, usize)> {
    ["^(1/2)", "^(1 / 2)"]
        .into_iter()
        .filter_map(|needle| display.find(needle).map(|idx| (idx, needle.len())))
        .min_by_key(|(idx, _)| *idx)
}

fn sqrt_half_power_zero_set_display_key(display: &str) -> Option<String> {
    let normalized = sqrt_half_power_display_key(display);
    for builtin in [
        "cos", "sin", "tan", "cot", "sec", "csc", "cosh", "sinh", "tanh",
    ] {
        let prefix = format!("{builtin}(");
        for suffix in [") ≠ 0", ")"] {
            if !normalized.starts_with(&prefix) || !normalized.ends_with(suffix) {
                continue;
            }

            let arg = &normalized[prefix.len()..normalized.len() - suffix.len()];
            let Some((left, right)) = split_top_level_subtraction_display(arg) else {
                continue;
            };
            let mut parts = [left.trim().to_string(), right.trim().to_string()];
            parts.sort();
            return Some(format!("{builtin}({} - {}){suffix}", parts[0], parts[1]));
        }
    }

    None
}

fn split_top_level_subtraction_display(display: &str) -> Option<(&str, &str)> {
    let mut depth = 0usize;
    for (idx, ch) in display.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            '-' if depth == 0 => return Some((&display[..idx], &display[idx + ch.len_utf8()..])),
            _ => {}
        }
    }
    None
}

fn display_base_before_half_power(display: &str, power_start: usize) -> Option<(usize, String)> {
    let prefix = &display[..power_start];
    let trimmed_end = prefix.trim_end().len();
    if trimmed_end == 0 {
        return None;
    }

    if prefix[..trimmed_end].ends_with(')') {
        let mut depth = 0usize;
        for (idx, ch) in prefix[..trimmed_end].char_indices().rev() {
            match ch {
                ')' => depth += 1,
                '(' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        return Some((idx, prefix[idx + 1..trimmed_end - 1].to_string()));
                    }
                }
                _ => {}
            }
        }
        return None;
    }

    let mut start = trimmed_end;
    for (idx, ch) in prefix[..trimmed_end].char_indices().rev() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            start = idx;
        } else {
            break;
        }
    }
    (start < trimmed_end).then(|| (start, prefix[start..trimmed_end].to_string()))
}

fn visible_required_conditions_after_public_suppression<'a>(
    ctx: &Context,
    normalized: &'a [crate::ImplicitCondition],
    assumed_filter: &AssumedConditionFilter,
    raw_input: &str,
    result_display: Option<&str>,
) -> Vec<&'a crate::ImplicitCondition> {
    if result_display.map(str::trim_start) == Some("undefined")
        && normalized
            .iter()
            .any(|cond| required_condition_is_impossible(ctx, cond))
    {
        return Vec::new();
    }

    let infinity_tail = resolved_infinity_limit_tail(raw_input, result_display);
    normalized
        .iter()
        .filter(|cond| !assumed_filter.covers_required_condition(ctx, cond))
        .filter(|cond| !should_suppress_public_required_condition(ctx, cond, result_display))
        .filter(|cond| {
            !infinity_tail.as_ref().is_some_and(|tail| {
                required_condition_is_eventually_true_on_infinity_tail(
                    ctx,
                    cond,
                    &tail.var,
                    tail.approach,
                )
            })
        })
        .collect()
}

#[derive(Clone, Copy)]
enum InfinityTailApproach {
    Pos,
    Neg,
}

struct InfinityLimitTail {
    var: String,
    approach: InfinityTailApproach,
}

fn resolved_infinity_limit_tail(
    raw_input: &str,
    result_display: Option<&str>,
) -> Option<InfinityLimitTail> {
    let result_display = result_display?.trim_start();
    if result_display == "undefined" || result_display.starts_with("limit(") {
        return None;
    }

    let Some(EvalSpecialCommand::Limit { var, approach, .. }) =
        cas_api_models::parse_eval_special_command(raw_input)
    else {
        return None;
    };
    let approach = match approach {
        EvalLimitApproach::PosInfinity => InfinityTailApproach::Pos,
        EvalLimitApproach::NegInfinity => InfinityTailApproach::Neg,
        EvalLimitApproach::Finite(_)
        | EvalLimitApproach::FiniteFromLeft(_)
        | EvalLimitApproach::FiniteFromRight(_) => return None,
    };
    Some(InfinityLimitTail { var, approach })
}

fn required_condition_is_eventually_true_on_infinity_tail(
    ctx: &Context,
    cond: &crate::ImplicitCondition,
    var: &str,
    approach: InfinityTailApproach,
) -> bool {
    match cond {
        crate::ImplicitCondition::Positive(expr) | crate::ImplicitCondition::NonNegative(expr) => {
            affine_expr_tends_positive_on_tail(ctx, *expr, var, approach)
                || polynomial_in_limit_var_tends_positive_on_tail(ctx, *expr, var, approach)
                || rational_in_limit_var_is_eventually_positive_on_tail(ctx, *expr, var, approach)
        }
        crate::ImplicitCondition::LowerBound(expr, lower) => {
            if polynomial_in_limit_var_tends_positive_on_tail(ctx, *expr, var, approach)
                || rational_in_limit_var_tends_positive_infinity_on_tail(ctx, *expr, var, approach)
                || rational_in_limit_var_is_eventually_above_lower_on_tail(
                    ctx, *expr, lower, var, approach,
                )
            {
                return true;
            }
            let Some(form) = affine_form_in_limit_var(ctx, *expr, var) else {
                return false;
            };
            affine_slope_tends_positive_on_tail(&form.slope, approach)
                || (form.slope.is_zero() && form.intercept >= *lower)
        }
        crate::ImplicitCondition::NonZero(expr) => {
            affine_form_in_limit_var(ctx, *expr, var).is_some_and(|form| !form.slope.is_zero())
                || polynomial_in_limit_var_has_eventual_nonzero_tail(ctx, *expr, var)
                || rational_in_limit_var_has_eventual_nonzero_tail(ctx, *expr, var)
        }
    }
}

fn affine_expr_tends_positive_on_tail(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    approach: InfinityTailApproach,
) -> bool {
    affine_form_in_limit_var(ctx, expr, var)
        .is_some_and(|form| affine_slope_tends_positive_on_tail(&form.slope, approach))
}

fn affine_slope_tends_positive_on_tail(
    slope: &BigRational,
    approach: InfinityTailApproach,
) -> bool {
    match approach {
        InfinityTailApproach::Pos => slope.is_positive(),
        InfinityTailApproach::Neg => slope.is_negative(),
    }
}

fn polynomial_in_limit_var_tends_positive_on_tail(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    approach: InfinityTailApproach,
) -> bool {
    let Ok(poly) = cas_math::polynomial::Polynomial::from_expr(ctx, expr, var) else {
        return false;
    };
    if poly.is_zero() || poly.degree() == 0 {
        return false;
    }

    let leading_coeff = poly.leading_coeff();
    match approach {
        InfinityTailApproach::Pos => leading_coeff.is_positive(),
        InfinityTailApproach::Neg if poly.degree() % 2 == 0 => leading_coeff.is_positive(),
        InfinityTailApproach::Neg => leading_coeff.is_negative(),
    }
}

fn polynomial_in_limit_var_has_eventual_nonzero_tail(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> bool {
    cas_math::polynomial::Polynomial::from_expr(ctx, expr, var)
        .ok()
        .is_some_and(|poly| !poly.is_zero() && poly.degree() > 0)
}

#[derive(Clone)]
struct RationalLimitVarForm {
    numerator: cas_math::polynomial::Polynomial,
    denominator: cas_math::polynomial::Polynomial,
}

impl RationalLimitVarForm {
    fn from_polynomial(poly: cas_math::polynomial::Polynomial) -> Self {
        Self {
            denominator: cas_math::polynomial::Polynomial::one(poly.var.clone()),
            numerator: poly,
        }
    }

    fn neg(self) -> Self {
        Self {
            numerator: self.numerator.neg(),
            denominator: self.denominator,
        }
    }

    fn add(self, rhs: Self) -> Self {
        Self {
            numerator: self
                .numerator
                .mul(&rhs.denominator)
                .add(&rhs.numerator.mul(&self.denominator)),
            denominator: self.denominator.mul(&rhs.denominator),
        }
    }

    fn sub(self, rhs: Self) -> Self {
        self.add(rhs.neg())
    }

    fn mul(self, rhs: Self) -> Self {
        Self {
            numerator: self.numerator.mul(&rhs.numerator),
            denominator: self.denominator.mul(&rhs.denominator),
        }
    }

    fn div(self, rhs: Self) -> Option<Self> {
        if rhs.numerator.is_zero() {
            return None;
        }
        Some(Self {
            numerator: self.numerator.mul(&rhs.denominator),
            denominator: self.denominator.mul(&rhs.numerator),
        })
    }

    fn sub_constant(self, value: &BigRational) -> Self {
        let denominator = self.denominator;
        let constant =
            cas_math::polynomial::Polynomial::new(vec![value.clone()], denominator.var.clone());
        let scaled_denominator = denominator.mul(&constant);
        Self {
            numerator: self.numerator.sub(&scaled_denominator),
            denominator,
        }
    }
}

fn rational_form_in_limit_var(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<RationalLimitVarForm> {
    if let Ok(poly) = cas_math::polynomial::Polynomial::from_expr(ctx, expr, var) {
        return Some(RationalLimitVarForm::from_polynomial(poly));
    }

    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Neg(inner) => Some(rational_form_in_limit_var(ctx, *inner, var)?.neg()),
        Expr::Add(left, right) => Some(
            rational_form_in_limit_var(ctx, *left, var)?
                .add(rational_form_in_limit_var(ctx, *right, var)?),
        ),
        Expr::Sub(left, right) => Some(
            rational_form_in_limit_var(ctx, *left, var)?
                .sub(rational_form_in_limit_var(ctx, *right, var)?),
        ),
        Expr::Mul(left, right) => Some(
            rational_form_in_limit_var(ctx, *left, var)?
                .mul(rational_form_in_limit_var(ctx, *right, var)?),
        ),
        Expr::Div(left, right) => rational_form_in_limit_var(ctx, *left, var)?
            .div(rational_form_in_limit_var(ctx, *right, var)?),
        Expr::Hold(inner) => rational_form_in_limit_var(ctx, *inner, var),
        _ => None,
    }
}

fn rational_in_limit_var_tail_sign(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    approach: InfinityTailApproach,
) -> Option<bool> {
    let form = rational_form_in_limit_var(ctx, expr, var)?;
    rational_form_tail_sign(&form, approach)
}

fn rational_form_tail_sign(
    form: &RationalLimitVarForm,
    approach: InfinityTailApproach,
) -> Option<bool> {
    if form.numerator.is_zero() || form.denominator.is_zero() {
        return None;
    }

    let leading_ratio = form.numerator.leading_coeff() / form.denominator.leading_coeff();
    let degree_delta = form.numerator.degree().abs_diff(form.denominator.degree());
    Some(match approach {
        InfinityTailApproach::Pos => leading_ratio.is_positive(),
        InfinityTailApproach::Neg if degree_delta.is_multiple_of(2) => leading_ratio.is_positive(),
        InfinityTailApproach::Neg => leading_ratio.is_negative(),
    })
}

fn rational_in_limit_var_is_eventually_positive_on_tail(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    approach: InfinityTailApproach,
) -> bool {
    rational_in_limit_var_tail_sign(ctx, expr, var, approach) == Some(true)
}

fn rational_in_limit_var_is_eventually_above_lower_on_tail(
    ctx: &Context,
    expr: ExprId,
    lower: &BigRational,
    var: &str,
    approach: InfinityTailApproach,
) -> bool {
    let Some(form) = rational_form_in_limit_var(ctx, expr, var) else {
        return false;
    };
    rational_form_tail_sign(&form.sub_constant(lower), approach) == Some(true)
}

fn rational_in_limit_var_tends_positive_infinity_on_tail(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    approach: InfinityTailApproach,
) -> bool {
    let Some(form) = rational_form_in_limit_var(ctx, expr, var) else {
        return false;
    };
    if form.numerator.is_zero()
        || form.denominator.is_zero()
        || form.numerator.degree() <= form.denominator.degree()
    {
        return false;
    }

    rational_in_limit_var_tail_sign(ctx, expr, var, approach) == Some(true)
}

fn rational_in_limit_var_has_eventual_nonzero_tail(ctx: &Context, expr: ExprId, var: &str) -> bool {
    rational_form_in_limit_var(ctx, expr, var).is_some_and(|form| {
        !form.numerator.is_zero()
            && !form.denominator.is_zero()
            && (form.numerator.degree() > 0 || form.denominator.degree() > 0)
    })
}

#[derive(Clone)]
struct LimitVarAffineForm {
    slope: BigRational,
    intercept: BigRational,
}

impl LimitVarAffineForm {
    fn constant(value: BigRational) -> Self {
        Self {
            slope: BigRational::zero(),
            intercept: value,
        }
    }

    fn variable() -> Self {
        Self {
            slope: BigRational::one(),
            intercept: BigRational::zero(),
        }
    }

    fn scale(self, factor: BigRational) -> Self {
        Self {
            slope: self.slope * factor.clone(),
            intercept: self.intercept * factor,
        }
    }

    fn add(self, rhs: Self) -> Self {
        Self {
            slope: self.slope + rhs.slope,
            intercept: self.intercept + rhs.intercept,
        }
    }
}

fn affine_form_in_limit_var(ctx: &Context, expr: ExprId, var: &str) -> Option<LimitVarAffineForm> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Number(value) => Some(LimitVarAffineForm::constant(value.clone())),
        Expr::Variable(symbol) if ctx.sym_name(*symbol) == var => {
            Some(LimitVarAffineForm::variable())
        }
        Expr::Variable(_) | Expr::Constant(_) | Expr::Function(_, _) | Expr::Matrix { .. } => None,
        Expr::SessionRef(_) => None,
        Expr::Hold(inner) => affine_form_in_limit_var(ctx, *inner, var),
        Expr::Neg(inner) => {
            affine_form_in_limit_var(ctx, *inner, var).map(|form| form.scale(-BigRational::one()))
        }
        Expr::Add(left, right) => Some(
            affine_form_in_limit_var(ctx, *left, var)?
                .add(affine_form_in_limit_var(ctx, *right, var)?),
        ),
        Expr::Sub(left, right) => Some(
            affine_form_in_limit_var(ctx, *left, var)?
                .add(affine_form_in_limit_var(ctx, *right, var)?.scale(-BigRational::one())),
        ),
        Expr::Mul(left, right) => {
            let left_form = affine_form_in_limit_var(ctx, *left, var)?;
            let right_form = affine_form_in_limit_var(ctx, *right, var)?;
            if left_form.slope.is_zero() {
                Some(right_form.scale(left_form.intercept))
            } else if right_form.slope.is_zero() {
                Some(left_form.scale(right_form.intercept))
            } else {
                None
            }
        }
        Expr::Div(left, right) => {
            let numerator = affine_form_in_limit_var(ctx, *left, var)?;
            let denominator = affine_form_in_limit_var(ctx, *right, var)?;
            if !denominator.slope.is_zero() || denominator.intercept.is_zero() {
                return None;
            }
            Some(numerator.scale(BigRational::one() / denominator.intercept))
        }
        Expr::Pow(_, _) => None,
    }
}

fn required_condition_is_impossible(ctx: &Context, cond: &crate::ImplicitCondition) -> bool {
    match cond {
        crate::ImplicitCondition::Positive(expr) => {
            let mut scratch = ctx.clone();
            cas_math::calculus_domain_support::positive_condition_is_impossible_over_reals(
                &mut scratch,
                *expr,
                12,
            )
        }
        crate::ImplicitCondition::NonNegative(expr) => {
            let mut scratch = ctx.clone();
            cas_math::calculus_domain_support::nonnegative_condition_is_impossible_over_reals(
                &mut scratch,
                *expr,
                12,
            )
        }
        _ => false,
    }
}

fn should_suppress_public_required_condition(
    ctx: &Context,
    cond: &crate::ImplicitCondition,
    result_display: Option<&str>,
) -> bool {
    let Some(display) = result_display.map(str::trim_start) else {
        return false;
    };
    if !(display.starts_with("limit(") || display == "undefined") {
        return false;
    }

    if display == "undefined" && required_condition_is_impossible(ctx, cond) {
        return true;
    }

    matches!(
        cond,
        crate::ImplicitCondition::NonZero(expr) if is_integer_literal(ctx, *expr, 0)
    )
}

fn required_condition_is_redundant(
    ctx: &Context,
    cond: &crate::ImplicitCondition,
    visible_conditions: &[&crate::ImplicitCondition],
) -> bool {
    let crate::ImplicitCondition::NonZero(nonzero_expr) = cond else {
        return matches!(cond, crate::ImplicitCondition::Positive(positive_expr) if
            reciprocal_trig_log_positive_quotient_condition_is_redundant(
                ctx,
                *positive_expr,
                visible_conditions
            )
        );
    };

    if unit_interval_nonzero_condition_is_redundant(ctx, *nonzero_expr, visible_conditions) {
        return true;
    }

    if calculus_nonzero_condition_is_redundant(ctx, *nonzero_expr, visible_conditions) {
        return true;
    }
    if sqrt_like_unary_nonzero_condition_is_redundant(ctx, *nonzero_expr, visible_conditions) {
        return true;
    }

    reciprocal_trig_log_argument_condition_is_redundant(ctx, *nonzero_expr, visible_conditions)
}

#[derive(Clone, Copy)]
enum ReciprocalTrigLogPositiveKind {
    SecTan,
    CscCot,
}

fn reciprocal_trig_log_positive_quotient_condition_is_redundant(
    ctx: &Context,
    positive_expr: ExprId,
    visible_conditions: &[&crate::ImplicitCondition],
) -> bool {
    if reciprocal_trig_log_positive_quotient_display_condition_is_redundant(
        ctx,
        positive_expr,
        visible_conditions,
    ) {
        return true;
    }

    let Some((kind, arg)) = reciprocal_trig_log_positive_quotient_arg(ctx, positive_expr) else {
        return false;
    };

    visible_conditions.iter().any(|candidate| {
        let crate::ImplicitCondition::Positive(candidate_expr) = candidate else {
            return false;
        };
        if *candidate_expr == positive_expr {
            return false;
        }

        let candidate_arg = match kind {
            ReciprocalTrigLogPositiveKind::SecTan => sec_tan_sum_arg(ctx, *candidate_expr),
            ReciprocalTrigLogPositiveKind::CscCot => csc_cot_difference_arg(ctx, *candidate_expr),
        };
        candidate_arg.is_some_and(|candidate_arg| {
            cas_math::expr_domain::exprs_equivalent(ctx, candidate_arg, arg)
        })
    })
}

fn reciprocal_trig_log_positive_quotient_display_condition_is_redundant(
    ctx: &Context,
    positive_expr: ExprId,
    visible_conditions: &[&crate::ImplicitCondition],
) -> bool {
    let positive_display = expr_display(ctx, positive_expr);
    visible_conditions.iter().any(|candidate| {
        let crate::ImplicitCondition::Positive(candidate_expr) = candidate else {
            return false;
        };
        if *candidate_expr == positive_expr {
            return false;
        }

        if let Some(arg) = sec_tan_sum_arg(ctx, *candidate_expr) {
            let arg_display = expr_display(ctx, arg);
            return positive_display == format!("(sin({arg_display}) + 1) / cos({arg_display})")
                || positive_display == format!("(1 + sin({arg_display})) / cos({arg_display})");
        }

        if let Some(arg) = csc_cot_difference_arg(ctx, *candidate_expr) {
            let arg_display = expr_display(ctx, arg);
            return positive_display == format!("(1 - cos({arg_display})) / sin({arg_display})");
        }

        false
    })
}

fn reciprocal_trig_log_positive_quotient_arg(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ReciprocalTrigLogPositiveKind, ExprId)> {
    sec_tan_compact_quotient_arg(ctx, expr)
        .map(|arg| (ReciprocalTrigLogPositiveKind::SecTan, arg))
        .or_else(|| {
            csc_cot_compact_quotient_arg(ctx, expr)
                .map(|arg| (ReciprocalTrigLogPositiveKind::CscCot, arg))
        })
}

fn sec_tan_compact_quotient_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let denominator_arg = unary_builtin_arg(ctx, *denominator, BuiltinFn::Cos)?;
    let numerator_arg = one_plus_unary_builtin_arg(ctx, *numerator, BuiltinFn::Sin)?;
    cas_math::expr_domain::exprs_equivalent(ctx, denominator_arg, numerator_arg)
        .then_some(denominator_arg)
}

fn csc_cot_compact_quotient_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let denominator_arg = unary_builtin_arg(ctx, *denominator, BuiltinFn::Sin)?;
    let numerator_arg = one_minus_unary_builtin_arg(ctx, *numerator, BuiltinFn::Cos)?;
    cas_math::expr_domain::exprs_equivalent(ctx, denominator_arg, numerator_arg)
        .then_some(denominator_arg)
}

fn one_plus_unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };

    if is_integer_literal(ctx, *left, 1) {
        unary_builtin_arg(ctx, *right, builtin)
    } else if is_integer_literal(ctx, *right, 1) {
        unary_builtin_arg(ctx, *left, builtin)
    } else {
        None
    }
}

fn one_minus_unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Sub(left, right) if is_integer_literal(ctx, *left, 1) => {
            unary_builtin_arg(ctx, *right, builtin)
        }
        Expr::Add(left, right) if is_integer_literal(ctx, *left, 1) => {
            negated_unary_builtin_arg(ctx, *right, builtin)
        }
        Expr::Add(left, right) if is_integer_literal(ctx, *right, 1) => {
            negated_unary_builtin_arg(ctx, *left, builtin)
        }
        _ => None,
    }
}

fn unit_interval_nonzero_condition_is_redundant(
    ctx: &Context,
    nonzero_expr: ExprId,
    visible_conditions: &[&crate::ImplicitCondition],
) -> bool {
    visible_conditions.iter().any(|candidate| {
        let crate::ImplicitCondition::NonNegative(gap_expr) = candidate else {
            return false;
        };
        let Some(denominator) = exterior_unit_interval_denominator(ctx, *gap_expr) else {
            return false;
        };
        cas_math::expr_domain::exprs_equivalent(ctx, denominator, nonzero_expr)
    })
}

fn reciprocal_trig_log_argument_condition_is_redundant(
    ctx: &Context,
    nonzero_expr: ExprId,
    visible_conditions: &[&crate::ImplicitCondition],
) -> bool {
    let Some((required_builtin, arg)) = reciprocal_trig_log_argument_denominator(ctx, nonzero_expr)
    else {
        return false;
    };

    visible_conditions.iter().any(|candidate| {
        let crate::ImplicitCondition::NonZero(candidate_expr) = candidate else {
            return false;
        };
        unary_builtin_arg(ctx, *candidate_expr, required_builtin).is_some_and(|candidate_arg| {
            cas_math::expr_domain::exprs_equivalent(ctx, candidate_arg, arg)
        })
    })
}

fn calculus_nonzero_condition_is_redundant(
    ctx: &Context,
    nonzero_expr: ExprId,
    visible_conditions: &[&crate::ImplicitCondition],
) -> bool {
    if !expr_contains_calculus_call(ctx, nonzero_expr) {
        return false;
    }

    visible_conditions.iter().any(|candidate| {
        let crate::ImplicitCondition::NonZero(candidate_expr) = candidate else {
            return false;
        };
        *candidate_expr != nonzero_expr
            && (cas_math::expr_domain::exprs_equivalent(ctx, *candidate_expr, nonzero_expr)
                || nonzero_condition_is_candidate_plus_antiderivative_residual(
                    ctx,
                    nonzero_expr,
                    *candidate_expr,
                ))
    })
}

fn sqrt_like_unary_nonzero_condition_is_redundant(
    ctx: &Context,
    nonzero_expr: ExprId,
    visible_conditions: &[&crate::ImplicitCondition],
) -> bool {
    let Some((required_builtin, arg)) = sqrt_like_unary_condition_arg(ctx, nonzero_expr) else {
        return false;
    };
    if !expr_contains_positive_half_power(ctx, arg) || expr_contains_sqrt_call(ctx, arg) {
        return false;
    }

    visible_conditions.iter().any(|candidate| {
        let crate::ImplicitCondition::NonZero(candidate_expr) = candidate else {
            return false;
        };
        if *candidate_expr == nonzero_expr {
            return false;
        }

        let Some((candidate_builtin, candidate_arg)) =
            sqrt_like_unary_condition_arg(ctx, *candidate_expr)
        else {
            return false;
        };

        candidate_builtin == required_builtin
            && expr_contains_sqrt_call(ctx, candidate_arg)
            && (sqrt_like_args_equivalent(ctx, candidate_arg, arg)
                || sqrt_like_unary_zero_set_args_equivalent(
                    ctx,
                    required_builtin,
                    candidate_arg,
                    arg,
                ))
    })
}

fn sqrt_like_unary_zero_set_args_equivalent(
    ctx: &Context,
    builtin: BuiltinFn,
    left: ExprId,
    right: ExprId,
) -> bool {
    if !matches!(
        builtin,
        BuiltinFn::Cos
            | BuiltinFn::Sin
            | BuiltinFn::Tan
            | BuiltinFn::Cot
            | BuiltinFn::Sec
            | BuiltinFn::Csc
            | BuiltinFn::Cosh
            | BuiltinFn::Sinh
            | BuiltinFn::Tanh
    ) {
        return false;
    }

    sqrt_like_args_are_negations(ctx, left, right)
}

fn sqrt_like_args_are_negations(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    let left = cas_ast::hold::unwrap_hold(ctx, left);
    let right = cas_ast::hold::unwrap_hold(ctx, right);

    if let Expr::Neg(left_inner) = ctx.get(left) {
        return sqrt_like_args_equivalent(ctx, *left_inner, right);
    }
    if let Expr::Neg(right_inner) = ctx.get(right) {
        return sqrt_like_args_equivalent(ctx, left, *right_inner);
    }

    match (ctx.get(left), ctx.get(right)) {
        (Expr::Sub(left_a, left_b), Expr::Sub(right_a, right_b)) => {
            sqrt_like_args_equivalent(ctx, *left_a, *right_b)
                && sqrt_like_args_equivalent(ctx, *left_b, *right_a)
        }
        _ => false,
    }
}

fn sqrt_like_unary_condition_arg(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    for builtin in [
        BuiltinFn::Cos,
        BuiltinFn::Sin,
        BuiltinFn::Tan,
        BuiltinFn::Cot,
        BuiltinFn::Sec,
        BuiltinFn::Csc,
        BuiltinFn::Cosh,
        BuiltinFn::Sinh,
        BuiltinFn::Tanh,
    ] {
        if let Some(arg) = unary_builtin_arg(ctx, expr, builtin) {
            if expr_contains_sqrt_like_form(ctx, arg) {
                return Some((builtin, arg));
            }
        }
    }
    None
}

fn expr_contains_sqrt_like_form(ctx: &Context, expr: ExprId) -> bool {
    if unary_builtin_arg(ctx, expr, BuiltinFn::Sqrt).is_some() {
        return true;
    }
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Function(_, args) => args
            .iter()
            .any(|arg| expr_contains_sqrt_like_form(ctx, *arg)),
        Expr::Pow(_, exp) if is_positive_half_literal(ctx, *exp) => true,
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            expr_contains_sqrt_like_form(ctx, *left) || expr_contains_sqrt_like_form(ctx, *right)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => expr_contains_sqrt_like_form(ctx, *inner),
        Expr::Matrix { data, .. } => data
            .iter()
            .any(|entry| expr_contains_sqrt_like_form(ctx, *entry)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => false,
    }
}

fn sqrt_like_args_equivalent(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    if cas_ast::ordering::compare_expr(ctx, left, right) == std::cmp::Ordering::Equal
        || cas_math::expr_domain::exprs_equivalent(ctx, left, right)
    {
        return true;
    }

    let left = cas_ast::hold::unwrap_hold(ctx, left);
    let right = cas_ast::hold::unwrap_hold(ctx, right);
    if let (Some(left_arg), Expr::Pow(right_base, right_exp)) = (
        unary_builtin_arg(ctx, left, BuiltinFn::Sqrt),
        ctx.get(right),
    ) {
        if is_positive_half_literal(ctx, *right_exp) {
            return sqrt_like_args_equivalent(ctx, left_arg, *right_base);
        }
    }
    if let (Expr::Pow(left_base, left_exp), Some(right_arg)) = (
        ctx.get(left),
        unary_builtin_arg(ctx, right, BuiltinFn::Sqrt),
    ) {
        if is_positive_half_literal(ctx, *left_exp) {
            return sqrt_like_args_equivalent(ctx, *left_base, right_arg);
        }
    }

    match (ctx.get(left), ctx.get(right)) {
        (Expr::Function(left_fn, left_args), Expr::Pow(right_base, right_exp))
            if ctx.is_builtin(*left_fn, BuiltinFn::Sqrt)
                && left_args.len() == 1
                && is_positive_half_literal(ctx, *right_exp) =>
        {
            sqrt_like_args_equivalent(ctx, left_args[0], *right_base)
        }
        (Expr::Pow(left_base, left_exp), Expr::Function(right_fn, right_args))
            if is_positive_half_literal(ctx, *left_exp)
                && ctx.is_builtin(*right_fn, BuiltinFn::Sqrt)
                && right_args.len() == 1 =>
        {
            sqrt_like_args_equivalent(ctx, *left_base, right_args[0])
        }
        (Expr::Function(left_fn, left_args), Expr::Function(right_fn, right_args))
            if left_fn == right_fn && left_args.len() == right_args.len() =>
        {
            left_args
                .iter()
                .zip(right_args.iter())
                .all(|(left_arg, right_arg)| sqrt_like_args_equivalent(ctx, *left_arg, *right_arg))
        }
        (Expr::Add(left_a, left_b), Expr::Add(right_a, right_b))
        | (Expr::Sub(left_a, left_b), Expr::Sub(right_a, right_b))
        | (Expr::Mul(left_a, left_b), Expr::Mul(right_a, right_b))
        | (Expr::Div(left_a, left_b), Expr::Div(right_a, right_b))
        | (Expr::Pow(left_a, left_b), Expr::Pow(right_a, right_b)) => {
            sqrt_like_args_equivalent(ctx, *left_a, *right_a)
                && sqrt_like_args_equivalent(ctx, *left_b, *right_b)
        }
        (Expr::Neg(left_inner), Expr::Neg(right_inner))
        | (Expr::Hold(left_inner), Expr::Hold(right_inner)) => {
            sqrt_like_args_equivalent(ctx, *left_inner, *right_inner)
        }
        _ => false,
    }
}

fn expr_contains_sqrt_call(ctx: &Context, expr: ExprId) -> bool {
    if unary_builtin_arg(ctx, expr, BuiltinFn::Sqrt).is_some() {
        return true;
    }
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Function(_, args) => args.iter().any(|arg| expr_contains_sqrt_call(ctx, *arg)),
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            expr_contains_sqrt_call(ctx, *left) || expr_contains_sqrt_call(ctx, *right)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => expr_contains_sqrt_call(ctx, *inner),
        Expr::Matrix { data, .. } => data
            .iter()
            .any(|entry| expr_contains_sqrt_call(ctx, *entry)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => false,
    }
}

fn expr_contains_positive_half_power(ctx: &Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Pow(_, exp) if is_positive_half_literal(ctx, *exp) => true,
        Expr::Function(_, args) => args
            .iter()
            .any(|arg| expr_contains_positive_half_power(ctx, *arg)),
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            expr_contains_positive_half_power(ctx, *left)
                || expr_contains_positive_half_power(ctx, *right)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => expr_contains_positive_half_power(ctx, *inner),
        Expr::Matrix { data, .. } => data
            .iter()
            .any(|entry| expr_contains_positive_half_power(ctx, *entry)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => false,
    }
}

fn is_positive_half_literal(ctx: &Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(value) if *value == BigRational::new(1.into(), 2.into())
    )
}

#[derive(Clone, Copy)]
struct SignedTerm {
    expr: ExprId,
    positive: bool,
}

fn nonzero_condition_is_candidate_plus_antiderivative_residual(
    ctx: &Context,
    nonzero_expr: ExprId,
    candidate_expr: ExprId,
) -> bool {
    let mut terms = Vec::new();
    collect_signed_add_terms(ctx, nonzero_expr, true, &mut terms);

    let mut candidate_terms = Vec::new();
    collect_signed_add_terms(ctx, candidate_expr, true, &mut candidate_terms);

    for candidate_term in candidate_terms {
        let Some(index) = terms.iter().position(|term| {
            term.positive == candidate_term.positive
                && cas_math::expr_domain::exprs_equivalent(ctx, term.expr, candidate_term.expr)
        }) else {
            return false;
        };
        terms.remove(index);
    }

    signed_terms_are_antiderivative_residual(ctx, &terms)
}

fn collect_signed_add_terms(
    ctx: &Context,
    expr: ExprId,
    positive: bool,
    terms: &mut Vec<SignedTerm>,
) {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            collect_signed_add_terms(ctx, *left, positive, terms);
            collect_signed_add_terms(ctx, *right, positive, terms);
        }
        Expr::Sub(left, right) => {
            collect_signed_add_terms(ctx, *left, positive, terms);
            collect_signed_add_terms(ctx, *right, !positive, terms);
        }
        Expr::Neg(inner) => collect_signed_add_terms(ctx, *inner, !positive, terms),
        Expr::Mul(left, right) if is_integer_literal(ctx, *left, -1) => {
            collect_signed_add_terms(ctx, *right, !positive, terms)
        }
        Expr::Mul(left, right) if is_integer_literal(ctx, *right, -1) => {
            collect_signed_add_terms(ctx, *left, !positive, terms)
        }
        _ => terms.push(SignedTerm { expr, positive }),
    }
}

fn signed_terms_are_antiderivative_residual(ctx: &Context, terms: &[SignedTerm]) -> bool {
    if terms.len() != 2 || terms[0].positive == terms[1].positive {
        return false;
    }

    diff_integrate_integrand(ctx, terms[0].expr).is_some_and(|integrand| {
        cas_math::expr_domain::exprs_equivalent(ctx, integrand, terms[1].expr)
    }) || diff_integrate_integrand(ctx, terms[1].expr).is_some_and(|integrand| {
        cas_math::expr_domain::exprs_equivalent(ctx, integrand, terms[0].expr)
    })
}

fn diff_integrate_integrand(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Function(diff_fn, diff_args) = ctx.get(expr) else {
        return None;
    };
    if ctx.sym_name(*diff_fn) != "diff" || diff_args.len() != 2 {
        return None;
    }

    let Expr::Function(integrate_fn, integrate_args) =
        ctx.get(cas_ast::hold::unwrap_hold(ctx, diff_args[0]))
    else {
        return None;
    };
    if ctx.sym_name(*integrate_fn) != "integrate" || integrate_args.len() != 2 {
        return None;
    }

    cas_math::expr_domain::exprs_equivalent(ctx, diff_args[1], integrate_args[1])
        .then_some(integrate_args[0])
}

fn expr_contains_calculus_call(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Function(fn_id, args) => {
            matches!(ctx.sym_name(*fn_id), "diff" | "integrate" | "limit")
                || args
                    .iter()
                    .any(|arg| expr_contains_calculus_call(ctx, *arg))
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            expr_contains_calculus_call(ctx, *left) || expr_contains_calculus_call(ctx, *right)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => expr_contains_calculus_call(ctx, *inner),
        Expr::Matrix { data, .. } => data
            .iter()
            .any(|entry| expr_contains_calculus_call(ctx, *entry)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => false,
    }
}

fn reciprocal_trig_log_argument_denominator(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    sec_tan_sum_arg(ctx, expr)
        .or_else(|| sec_tan_quotient_sum_arg(ctx, expr))
        .map(|arg| (BuiltinFn::Cos, arg))
        .or_else(|| {
            csc_cot_difference_arg(ctx, expr)
                .or_else(|| csc_cot_quotient_difference_arg(ctx, expr))
                .map(|arg| (BuiltinFn::Sin, arg))
        })
}

fn sec_tan_sum_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };
    unordered_same_arg_unary_pair(ctx, *left, BuiltinFn::Sec, *right, BuiltinFn::Tan)
}

fn csc_cot_difference_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Sub(left, right) => {
            same_arg_unary_pair(ctx, *left, BuiltinFn::Csc, *right, BuiltinFn::Cot)
        }
        Expr::Add(left, right) => unary_builtin_arg(ctx, *left, BuiltinFn::Csc)
            .zip(negated_unary_builtin_arg(ctx, *right, BuiltinFn::Cot))
            .or_else(|| {
                unary_builtin_arg(ctx, *right, BuiltinFn::Csc).zip(negated_unary_builtin_arg(
                    ctx,
                    *left,
                    BuiltinFn::Cot,
                ))
            })
            .and_then(|(left_arg, right_arg)| {
                cas_math::expr_domain::exprs_equivalent(ctx, left_arg, right_arg)
                    .then_some(left_arg)
            }),
        _ => None,
    }
}

fn sec_tan_quotient_sum_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };
    reciprocal_plus_ratio_arg(ctx, *left, *right, BuiltinFn::Sin, BuiltinFn::Cos)
        .or_else(|| reciprocal_plus_ratio_arg(ctx, *right, *left, BuiltinFn::Sin, BuiltinFn::Cos))
}

fn csc_cot_quotient_difference_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Sub(left, right) => {
            reciprocal_plus_ratio_arg(ctx, *left, *right, BuiltinFn::Cos, BuiltinFn::Sin)
        }
        Expr::Add(left, right) => reciprocal_plus_ratio_arg_with_negated_ratio(
            ctx,
            *left,
            *right,
            BuiltinFn::Cos,
            BuiltinFn::Sin,
        )
        .or_else(|| {
            reciprocal_plus_ratio_arg_with_negated_ratio(
                ctx,
                *right,
                *left,
                BuiltinFn::Cos,
                BuiltinFn::Sin,
            )
        }),
        _ => None,
    }
}

fn reciprocal_plus_ratio_arg(
    ctx: &Context,
    reciprocal: ExprId,
    ratio: ExprId,
    numerator_builtin: BuiltinFn,
    denominator_builtin: BuiltinFn,
) -> Option<ExprId> {
    let denominator_arg = reciprocal_builtin_denominator_arg(ctx, reciprocal, denominator_builtin)?;
    let ratio_arg =
        ratio_builtin_denominator_arg(ctx, ratio, numerator_builtin, denominator_builtin)?;
    cas_math::expr_domain::exprs_equivalent(ctx, denominator_arg, ratio_arg)
        .then_some(denominator_arg)
}

fn reciprocal_plus_ratio_arg_with_negated_ratio(
    ctx: &Context,
    reciprocal: ExprId,
    ratio: ExprId,
    numerator_builtin: BuiltinFn,
    denominator_builtin: BuiltinFn,
) -> Option<ExprId> {
    let ratio = match ctx.get(cas_ast::hold::unwrap_hold(ctx, ratio)) {
        Expr::Neg(inner) => *inner,
        Expr::Mul(left, right) if is_integer_literal(ctx, *left, -1) => *right,
        Expr::Mul(left, right) if is_integer_literal(ctx, *right, -1) => *left,
        _ => return None,
    };
    reciprocal_plus_ratio_arg(
        ctx,
        reciprocal,
        ratio,
        numerator_builtin,
        denominator_builtin,
    )
}

fn reciprocal_builtin_denominator_arg(
    ctx: &Context,
    expr: ExprId,
    denominator_builtin: BuiltinFn,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    if is_integer_literal(ctx, *numerator, 1) {
        unary_builtin_arg(ctx, *denominator, denominator_builtin)
    } else {
        None
    }
}

fn ratio_builtin_denominator_arg(
    ctx: &Context,
    expr: ExprId,
    numerator_builtin: BuiltinFn,
    denominator_builtin: BuiltinFn,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    same_arg_unary_pair(
        ctx,
        *numerator,
        numerator_builtin,
        *denominator,
        denominator_builtin,
    )
}

fn unordered_same_arg_unary_pair(
    ctx: &Context,
    left: ExprId,
    left_builtin: BuiltinFn,
    right: ExprId,
    right_builtin: BuiltinFn,
) -> Option<ExprId> {
    same_arg_unary_pair(ctx, left, left_builtin, right, right_builtin)
        .or_else(|| same_arg_unary_pair(ctx, right, left_builtin, left, right_builtin))
}

fn same_arg_unary_pair(
    ctx: &Context,
    left: ExprId,
    left_builtin: BuiltinFn,
    right: ExprId,
    right_builtin: BuiltinFn,
) -> Option<ExprId> {
    let left_arg = unary_builtin_arg(ctx, left, left_builtin)?;
    let right_arg = unary_builtin_arg(ctx, right, right_builtin)?;
    cas_math::expr_domain::exprs_equivalent(ctx, left_arg, right_arg).then_some(left_arg)
}

fn unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() == 1 && ctx.is_builtin(*fn_id, builtin) {
        Some(args[0])
    } else {
        None
    }
}

fn negated_unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Neg(inner) => unary_builtin_arg(ctx, *inner, builtin),
        Expr::Mul(left, right) if is_integer_literal(ctx, *left, -1) => {
            unary_builtin_arg(ctx, *right, builtin)
        }
        Expr::Mul(left, right) if is_integer_literal(ctx, *right, -1) => {
            unary_builtin_arg(ctx, *left, builtin)
        }
        _ => None,
    }
}

fn exterior_unit_interval_denominator(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let base = unit_interval_base(ctx, expr)?;
    match ctx.get(base) {
        Expr::Div(numerator, denominator) if is_integer_literal(ctx, *numerator, 1) => {
            Some(*denominator)
        }
        Expr::Pow(inner, exponent) if is_integer_literal(ctx, *exponent, -1) => Some(*inner),
        _ => None,
    }
}

fn unit_interval_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Sub(left, right) if is_integer_literal(ctx, *left, 1) => squared_base(ctx, *right),
        Expr::Add(left, right) if is_integer_literal(ctx, *left, 1) => {
            negated_squared_base(ctx, *right)
        }
        Expr::Add(left, right) if is_integer_literal(ctx, *right, 1) => {
            negated_squared_base(ctx, *left)
        }
        _ => None,
    }
}

fn squared_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    if is_integer_literal(ctx, *exponent, 2) {
        Some(*base)
    } else {
        None
    }
}

fn negated_squared_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Neg(inner) => squared_base(ctx, *inner),
        Expr::Mul(left, right) if is_integer_literal(ctx, *left, -1) => squared_base(ctx, *right),
        Expr::Mul(left, right) if is_integer_literal(ctx, *right, -1) => squared_base(ctx, *left),
        _ => None,
    }
}

fn is_integer_literal(ctx: &Context, expr: ExprId, value: i64) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(n) if n == &num_rational::BigRational::from_integer(value.into())
    )
}

pub(crate) fn collect_output_blocked_hints(
    ctx: &mut Context,
    resolved: cas_ast::ExprId,
    required_conditions: &[crate::ImplicitCondition],
    blocked_hints: &[crate::BlockedHint],
) -> Vec<BlockedHintDto> {
    let normalized_required_conditions = normalize_and_dedupe_conditions(ctx, required_conditions);
    crate::filter_blocked_hints_for_eval(
        ctx,
        resolved,
        &normalized_required_conditions,
        blocked_hints,
    )
    .iter()
    .map(|hint| BlockedHintDto {
        rule: hint.rule.clone(),
        requires: vec![crate::format_blocked_hint_condition(ctx, hint)],
        tip: hint.suggestion.to_string(),
    })
    .collect()
}

pub(crate) fn collect_output_assumptions_used(steps: &[crate::Step]) -> Vec<AssumptionDto> {
    let mut seen: HashSet<(String, String, String)> = HashSet::new();
    let mut assumptions = Vec::new();

    for step in steps {
        for event in step.assumption_events() {
            if !matches!(
                event.kind,
                cas_solver_core::assumption_model::AssumptionKind::HeuristicAssumption
            ) {
                continue;
            }

            let kind = event.key.kind().to_string();
            let rule = step.rule_name.to_string();
            let display = event.message.clone();
            let expr_canonical = event.expr_display.clone();
            if !seen.insert((kind.clone(), expr_canonical.clone(), display.clone())) {
                continue;
            }

            assumptions.push(AssumptionDto {
                kind,
                display,
                expr_canonical,
                rule,
            });
        }
    }

    assumptions
}

fn expr_display(ctx: &Context, expr_id: cas_ast::ExprId) -> String {
    DisplayExpr {
        context: ctx,
        id: expr_id,
    }
    .to_string()
}

pub(crate) fn apply_input_inverse_trig_alias_preferences(
    display: &str,
    raw_input: &str,
    result_display: Option<&str>,
) -> String {
    let mut adjusted = display.to_string();
    let result_lookup = result_display.map(normalize_alias_lookup_text);
    for (short, long) in [
        ("asin", "arcsin"),
        ("acos", "arccos"),
        ("atan", "arctan"),
        ("asec", "arcsec"),
        ("acsc", "arccsc"),
        ("acot", "arccot"),
    ] {
        adjusted = apply_single_input_inverse_trig_alias_preference(
            &adjusted,
            raw_input,
            result_lookup.as_deref(),
            short,
            long,
        );
    }
    adjusted
}

fn apply_single_input_inverse_trig_alias_preference(
    display: &str,
    raw_input: &str,
    result_lookup: Option<&str>,
    short: &str,
    long: &str,
) -> String {
    let long_call_prefix = format!("{long}(");
    let raw_lookup = normalize_alias_lookup_text(raw_input);
    let mut out = String::with_capacity(display.len());
    let mut cursor = 0;

    while let Some(relative_start) = display[cursor..].find(&long_call_prefix) {
        let start = cursor + relative_start;
        let Some(end) = matching_call_end(display, start + long.len()) else {
            break;
        };

        out.push_str(&display[cursor..start]);
        let long_call = &display[start..end];
        let short_call = format!("{short}{}", &long_call[long.len()..]);
        if result_lookup_contains_call(result_lookup, &short_call) {
            out.push_str(&short_call);
        } else if result_lookup_contains_call(result_lookup, long_call) {
            out.push_str(long_call);
        } else if raw_input_contains_short_alias_call(&raw_lookup, short, &short_call) {
            out.push_str(&short_call);
        } else {
            out.push_str(long_call);
        }
        cursor = end;
    }

    out.push_str(&display[cursor..]);
    out
}

fn result_lookup_contains_call(result_lookup: Option<&str>, call: &str) -> bool {
    result_lookup.is_some_and(|lookup| lookup.contains(&normalize_alias_lookup_text(call)))
}

fn raw_input_contains_short_alias_call(raw_lookup: &str, short: &str, short_call: &str) -> bool {
    let short_call_lookup = normalize_alias_lookup_text(short_call);
    if raw_lookup.contains(&short_call_lookup) {
        return true;
    }

    let Some(short_call_arg) = call_argument(&short_call_lookup, short.len()) else {
        return false;
    };
    let short_call_arg = strip_redundant_outer_parens(short_call_arg);
    let short_call_prefix = format!("{short}(");
    let mut cursor = 0;

    while let Some(relative_start) = raw_lookup[cursor..].find(&short_call_prefix) {
        let start = cursor + relative_start;
        let Some(end) = matching_call_end(raw_lookup, start + short.len()) else {
            break;
        };
        let raw_call = &raw_lookup[start..end];
        if call_argument(raw_call, short.len()).map(strip_redundant_outer_parens)
            == Some(short_call_arg)
        {
            return true;
        }
        cursor = end;
    }

    false
}

fn call_argument(call: &str, name_len: usize) -> Option<&str> {
    let open_paren = name_len;
    let end = matching_call_end(call, open_paren)?;
    if end != call.len() {
        return None;
    }
    Some(&call[open_paren + 1..end - 1])
}

fn strip_redundant_outer_parens(mut text: &str) -> &str {
    loop {
        if !text.starts_with('(') || !text.ends_with(')') {
            return text;
        }
        if matching_call_end(text, 0) != Some(text.len()) {
            return text;
        }
        text = &text[1..text.len() - 1];
    }
}

fn matching_call_end(text: &str, open_paren: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.get(open_paren) != Some(&b'(') {
        return None;
    }
    let mut depth = 0usize;
    for (offset, byte) in bytes[open_paren..].iter().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(open_paren + offset + 1);
                }
            }
            _ => {}
        }
    }
    None
}

fn normalize_alias_lookup_text(text: &str) -> String {
    text.chars()
        .filter_map(|ch| {
            if ch.is_whitespace() {
                None
            } else if ch == '·' {
                Some('*')
            } else {
                Some(ch)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedupe_required_displays_removes_opposite_shifted_sqrt_zero_set() {
        let displays = vec![
            "sinh(b - sqrt(x)) ≠ 0".to_string(),
            "x > 0".to_string(),
            "sinh(sqrt(x) - b) ≠ 0".to_string(),
        ];

        assert_eq!(
            dedupe_sqrt_half_power_required_displays(displays),
            vec!["sinh(b - sqrt(x)) ≠ 0".to_string(), "x > 0".to_string()]
        );
    }

    #[test]
    fn dedupe_condition_wires_removes_opposite_shifted_sqrt_zero_set() {
        let wires = vec![
            RequiredConditionWire {
                kind: "NonZero".to_string(),
                expr_display: "sinh(b - sqrt(x))".to_string(),
                expr_canonical: "sinh(b - sqrt(x))".to_string(),
            },
            RequiredConditionWire {
                kind: "Positive".to_string(),
                expr_display: "x".to_string(),
                expr_canonical: "x".to_string(),
            },
            RequiredConditionWire {
                kind: "NonZero".to_string(),
                expr_display: "sinh(sqrt(x) - b)".to_string(),
                expr_canonical: "sinh(sqrt(x) - b)".to_string(),
            },
        ];

        let deduped = dedupe_sqrt_half_power_condition_wires(wires);
        let rendered = deduped
            .iter()
            .map(|wire| format!("{}:{}", wire.kind, wire.expr_display))
            .collect::<Vec<_>>();

        assert_eq!(rendered, vec!["NonZero:sinh(b - sqrt(x))", "Positive:x"]);
    }
}
