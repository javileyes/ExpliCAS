//! Condition normalization, deduplication, and dominance rules.

use crate::domain_condition::ImplicitCondition;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_domain::{
    exprs_equivalent, exprs_equivalent_up_to_sign, is_abs_of, is_odd_power_of,
    is_positive_multiple_of, is_positive_power_of_base, is_product_dominated_by_positives,
};
use cas_math::expr_extract::{
    extract_abs_argument_view, extract_log_base_argument_view, extract_sqrt_argument_view,
};
use cas_math::expr_nary::{build_balanced_mul, mul_leaves};
use cas_math::expr_normalization::{
    extract_even_positive_power_base, normalize_condition_expr as normalize_condition_expr_math,
    normalize_condition_expr_preserve_sign,
};
use cas_math::factor::factor;
use cas_math::multipoly::MultiPoly;
use cas_math::numeric_eval::as_rational_const;
use cas_math::polynomial::Polynomial;
use cas_math::prove_sign::{prove_nonnegative_depth_with, prove_positive_depth_with};
use cas_math::root_forms::{try_rewrite_simplify_square_root_expr, SimplifySquareRootRewriteKind};
use cas_math::tri_proof::TriProof;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

const DISPLAY_SIGN_PROOF_DEPTH: usize = 12;

/// Normalize an expression for display in conditions.
pub fn normalize_condition_expr(ctx: &mut Context, expr: ExprId) -> ExprId {
    normalize_condition_expr_math(ctx, expr)
}

/// Normalize a condition for display (applies normalization to the inner expression).
pub fn normalize_condition(ctx: &mut Context, cond: &ImplicitCondition) -> ImplicitCondition {
    if let ImplicitCondition::NonNegative(e) = cond {
        if let Some(compact) = compact_positive_sqrt_lower_bound_for_display(ctx, *e) {
            return normalize_condition(ctx, &ImplicitCondition::NonNegative(compact));
        }
    }

    if let ImplicitCondition::Positive(e) = cond {
        if let Some(denominator) = positive_reciprocal_denominator(ctx, *e) {
            return normalize_condition(ctx, &ImplicitCondition::Positive(denominator));
        }

        if let Some(base) = positive_odd_power_base(ctx, *e) {
            let normalized_base = normalize_condition_expr_preserve_sign(ctx, base);
            return ImplicitCondition::Positive(normalized_base);
        }

        if let Some(base) = positive_condition_equivalent_nonzero_base(ctx, *e) {
            let normalized_base = normalize_condition_expr(ctx, base);
            return ImplicitCondition::NonZero(normalized_base);
        }

        if let Some(compact) = compact_positive_power_gap_for_display(ctx, *e) {
            return ImplicitCondition::Positive(compact);
        }

        if let Some(compact) = compact_positive_sqrt_lower_bound_for_display(ctx, *e) {
            return normalize_condition(ctx, &ImplicitCondition::Positive(compact));
        }
    }

    let normalized_expr = match cond {
        ImplicitCondition::NonNegative(e) => normalize_condition_expr_preserve_sign(ctx, *e),
        ImplicitCondition::Positive(e) => normalize_condition_expr_preserve_sign(ctx, *e),
        ImplicitCondition::NonZero(e) => {
            let stripped = strip_nonzero_scalar_factors_for_display(ctx, *e);
            normalize_nonzero_condition_expr_for_display(ctx, stripped)
        }
    };

    match cond {
        ImplicitCondition::NonNegative(_) => {
            if let Some(compact) = compact_positive_power_gap_for_display(ctx, normalized_expr) {
                ImplicitCondition::NonNegative(compact)
            } else {
                ImplicitCondition::NonNegative(normalized_expr)
            }
        }
        ImplicitCondition::Positive(_) => {
            if let Some(compact) = compact_positive_power_gap_for_display(ctx, normalized_expr) {
                ImplicitCondition::Positive(compact)
            } else {
                ImplicitCondition::Positive(normalized_expr)
            }
        }
        ImplicitCondition::NonZero(_) => ImplicitCondition::NonZero(normalized_expr),
    }
}

fn positive_reciprocal_denominator(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };

    as_rational_const(ctx, *num)
        .is_some_and(|constant| constant.is_positive())
        .then_some(*den)
}

fn positive_odd_power_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let Expr::Number(exp_num) = ctx.get(*exp) else {
        return None;
    };

    if !exp_num.is_integer() {
        return None;
    }

    let exp_int = exp_num.to_integer();
    let zero: num_bigint::BigInt = 0.into();
    let two: num_bigint::BigInt = 2.into();
    let one: num_bigint::BigInt = 1.into();

    (exp_int > zero && (&exp_int % &two) == one).then_some(*base)
}

fn positive_condition_equivalent_nonzero_base(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    // Positive(x^even) -> NonZero(x) because x^(2k) > 0 <=> x != 0 for reals.
    if let Some(base) = extract_even_positive_power_base(ctx, expr) {
        return Some(base);
    }

    // Positive(sqrt(x^even)) -> NonZero(x) because sqrt(x^(2k)) > 0 <=> x != 0.
    if let Some(base) = positive_sqrt_even_power_base(ctx, expr) {
        return Some(base);
    }

    if let Some(base) = positive_perfect_square_base_from_sqrt_rewrite(ctx, expr) {
        return Some(base);
    }

    if let Some(base) = positive_condition_equivalent_nonzero_base_from_factorable(ctx, expr) {
        return Some(base);
    }

    let normalized = normalize_condition_expr_preserve_sign(ctx, expr);
    if normalized == expr {
        return None;
    }

    positive_sqrt_even_power_base(ctx, normalized)
        .or_else(|| positive_perfect_square_base_from_sqrt_rewrite(ctx, normalized))
        .or_else(|| positive_condition_equivalent_nonzero_base_from_factorable(ctx, normalized))
}

fn positive_condition_equivalent_nonzero_base_from_factorable(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let factored = factor(ctx, expr);
    let factors: Vec<_> = mul_leaves(ctx, factored).into_iter().collect();
    if factors.len() <= 1 {
        return None;
    }

    let mut candidate_base = None;
    for (factor_index, factor_expr) in factors.iter().copied().enumerate() {
        let Some(base) = positive_multiple_even_power_base(ctx, factor_expr) else {
            continue;
        };

        if !factors.iter().enumerate().all(|(other_index, other)| {
            other_index == factor_index || is_intrinsically_positive_real(ctx, *other)
        }) {
            continue;
        }

        if candidate_base.replace(base).is_some() {
            return None;
        }
    }

    candidate_base
}

fn positive_condition_equivalent_nonzero_conditions(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<Vec<ImplicitCondition>> {
    if let Some(base) = extract_even_positive_power_base(ctx, expr) {
        return Some(expand_nonzero_condition_for_display(ctx, base));
    }

    if let Some(base) = positive_sqrt_even_power_base(ctx, expr) {
        return Some(expand_nonzero_condition_for_display(ctx, base));
    }

    if let Some(base) = positive_perfect_square_base_from_sqrt_rewrite(ctx, expr) {
        return Some(expand_nonzero_condition_for_display(ctx, base));
    }

    if let Some(conditions) =
        positive_condition_equivalent_nonzero_conditions_from_factorable(ctx, expr)
    {
        return Some(conditions);
    }

    let normalized = normalize_condition_expr_preserve_sign(ctx, expr);
    if normalized == expr {
        return None;
    }

    if let Some(base) = positive_sqrt_even_power_base(ctx, normalized) {
        return Some(expand_nonzero_condition_for_display(ctx, base));
    }

    if let Some(base) = positive_perfect_square_base_from_sqrt_rewrite(ctx, normalized) {
        return Some(expand_nonzero_condition_for_display(ctx, base));
    }

    positive_condition_equivalent_nonzero_conditions_from_factorable(ctx, normalized)
}

fn positive_sqrt_even_power_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let sqrt_arg = extract_sqrt_like_base(ctx, expr)?;
    extract_even_positive_power_base(ctx, sqrt_arg)
}

fn positive_perfect_square_base_from_sqrt_rewrite(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Add(_, _) | Expr::Sub(_, _) => {}
        _ => return None,
    }

    let sqrt_expr = ctx.call_builtin(BuiltinFn::Sqrt, vec![expr]);
    let rewrite = try_rewrite_simplify_square_root_expr(ctx, sqrt_expr)?;
    (rewrite.kind == SimplifySquareRootRewriteKind::PerfectSquare)
        .then(|| extract_abs_argument_view(ctx, rewrite.rewritten))
        .flatten()
}

fn positive_condition_equivalent_nonzero_conditions_from_factorable(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<Vec<ImplicitCondition>> {
    let factored = factor(ctx, expr);
    let factors: Vec<_> = mul_leaves(ctx, factored).into_iter().collect();
    if factors.len() <= 1 {
        return None;
    }

    let mut expanded = Vec::new();
    for factor_expr in factors {
        if let Some(base) = positive_multiple_even_power_base(ctx, factor_expr) {
            for condition in expand_nonzero_condition_for_display(ctx, base) {
                if !expanded
                    .iter()
                    .any(|existing| conditions_equivalent(ctx, existing, &condition))
                {
                    expanded.push(condition);
                }
            }
            continue;
        }

        if !is_intrinsically_positive_real(ctx, factor_expr) {
            return None;
        }
    }

    (!expanded.is_empty()).then_some(expanded)
}

fn strip_nonzero_scalar_factors_for_display(ctx: &mut Context, expr: ExprId) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => strip_nonzero_scalar_factors_for_display(ctx, inner),
        Expr::Mul(_, _) => {
            let mut symbolic_factors = Vec::new();
            for factor in mul_leaves(ctx, expr) {
                if as_rational_const(ctx, factor).is_none() {
                    symbolic_factors.push(strip_nonzero_scalar_factors_for_display(ctx, factor));
                }
            }

            if symbolic_factors.is_empty() {
                expr
            } else {
                build_balanced_mul(ctx, &symbolic_factors)
            }
        }
        Expr::Div(num, den) => {
            let numerator_is_numeric = as_rational_const(ctx, num).is_some();
            let denominator_is_numeric = as_rational_const(ctx, den).is_some();
            match (numerator_is_numeric, denominator_is_numeric) {
                (true, false) => strip_nonzero_scalar_factors_for_display(ctx, den),
                (false, true) => strip_nonzero_scalar_factors_for_display(ctx, num),
                _ => expr,
            }
        }
        _ => expr,
    }
}

fn is_intrinsically_positive_real(ctx: &Context, expr: ExprId) -> bool {
    prove_positive_depth_with(
        ctx,
        expr,
        DISPLAY_SIGN_PROOF_DEPTH,
        true,
        |_inner_ctx, _inner_expr, _inner_depth| TriProof::Unknown,
    )
    .is_proven()
}

fn is_intrinsically_nonnegative_real(ctx: &Context, expr: ExprId) -> bool {
    prove_nonnegative_depth_with(
        ctx,
        expr,
        DISPLAY_SIGN_PROOF_DEPTH,
        true,
        |_inner_ctx, _inner_expr, _inner_depth| TriProof::Unknown,
    )
    .is_proven()
}

fn compact_positive_power_gap_for_display(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    if let Some(compact) = compact_structured_positive_power_gap_for_display(ctx, expr) {
        return Some(compact);
    }

    if let Some(compact) = compact_expanded_monic_low_degree_power_gap_for_display(ctx, expr) {
        return Some(compact);
    }

    if let Some(compact) = compact_expanded_shifted_fourth_gap_for_display(ctx, expr) {
        return Some(compact);
    }

    if let Some(compact) =
        compact_expanded_negative_scaled_low_degree_power_gap_for_display(ctx, expr)
    {
        return Some(compact);
    }

    compact_expanded_negative_monic_low_degree_power_gap_for_display(ctx, expr)
}

fn sqrt_positive_lower_shift_parts(ctx: &Context, expr: ExprId) -> Option<(ExprId, BigRational)> {
    match ctx.get(expr).clone() {
        Expr::Sub(left, right) => {
            let shift = as_rational_const(ctx, right)?;
            shift.is_positive().then_some((left, shift))
        }
        Expr::Add(left, right) => {
            if let Some(value) = as_rational_const(ctx, left).filter(|value| value.is_negative()) {
                return Some((right, -value));
            }
            if let Some(value) = as_rational_const(ctx, right).filter(|value| value.is_negative()) {
                return Some((left, -value));
            }
            None
        }
        _ => None,
    }
}

fn compact_positive_sqrt_lower_bound_for_display(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let (sqrt_expr, shift) = sqrt_positive_lower_shift_parts(ctx, expr)?;
    let base = extract_sqrt_like_base(ctx, sqrt_expr)?;
    let shift_squared = shift.clone() * shift;
    let boundary = ctx.add(Expr::Number(shift_squared));
    let gap = ctx.add(Expr::Sub(base, boundary));
    Some(normalize_condition_expr_preserve_sign(ctx, gap))
}

fn normalize_nonzero_condition_expr_for_display(ctx: &mut Context, expr: ExprId) -> ExprId {
    if let Some(base) = nonzero_integer_power_base_for_display(ctx, expr) {
        return normalize_nonzero_condition_expr_for_display(ctx, base);
    }

    if let Some(base) = compact_expanded_monic_low_degree_power_for_display(ctx, expr) {
        return normalize_nonzero_condition_expr_for_display(ctx, base);
    }

    let normalized = normalize_condition_expr(ctx, expr);
    let normalized =
        primitive_nonzero_polynomial_for_display(ctx, normalized).unwrap_or(normalized);
    if let Some(base) = nonzero_integer_power_base_for_display(ctx, normalized) {
        return normalize_nonzero_condition_expr_for_display(ctx, base);
    }

    if let Some(base) = compact_expanded_monic_low_degree_power_for_display(ctx, normalized) {
        return normalize_nonzero_condition_expr_for_display(ctx, base);
    }

    compact_nonzero_power_gap_for_display(ctx, expr)
        .or_else(|| compact_nonzero_power_gap_for_display(ctx, normalized))
        .unwrap_or(normalized)
}

fn nonzero_integer_power_base_for_display(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let Expr::Number(exp_num) = ctx.get(*exp) else {
        return None;
    };

    (exp_num.is_integer() && !exp_num.is_zero()).then_some(*base)
}

fn primitive_nonzero_polynomial_for_display(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    use cas_math::multipoly::{multipoly_from_expr, multipoly_to_expr, PolyBudget};

    let budget = PolyBudget {
        max_terms: 50,
        max_total_degree: 20,
        max_pow_exp: 10,
    };
    let poly = multipoly_from_expr(ctx, expr, &budget).ok()?;
    if poly.is_zero() || poly.is_constant() {
        return None;
    }

    let (content, primitive) = poly.primitive_part();
    if content.is_zero() || content.is_one() {
        return None;
    }

    Some(multipoly_to_expr(&primitive, ctx))
}

fn compact_nonzero_power_gap_for_display(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    if let Some(compact) = compact_positive_power_gap_for_display(ctx, expr) {
        return Some(compact);
    }

    let negated = ctx.add(Expr::Neg(expr));
    compact_positive_power_gap_for_display(ctx, negated)
}

fn compact_structured_positive_power_gap_for_display(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Sub(left, right) = ctx.get(expr).clone() else {
        return None;
    };
    as_rational_const(ctx, left)?;

    let compact_right = compact_high_additive_power_for_display(ctx, right)?;
    Some(ctx.add(Expr::Sub(left, compact_right)))
}

fn compact_high_additive_power_for_display(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let (base, outer_exp) = positive_integer_power_parts(ctx, expr)?;
    let (compact_base, inner_exp) = positive_integer_power_parts(ctx, base).unwrap_or((base, 1));
    let combined_exp = outer_exp.checked_mul(inner_exp)?;

    if combined_exp < 4 || !is_small_additive_shape(ctx, compact_base) {
        return None;
    }

    let exp = ctx.num(combined_exp);
    Some(ctx.add(Expr::Pow(compact_base, exp)))
}

fn compact_expanded_shifted_fourth_gap_for_display(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let vars = cas_ast::collect_variables(ctx, expr);
    if vars.len() != 1 {
        return None;
    }
    let var = vars.iter().next()?;
    let poly = Polynomial::from_expr(ctx, expr, var).ok()?;
    if poly.degree() != 4 {
        return None;
    }

    let coeff = |degree: usize| {
        poly.coeffs
            .get(degree)
            .cloned()
            .unwrap_or_else(BigRational::zero)
    };
    let c4 = coeff(4);
    if c4 != -BigRational::one() {
        return None;
    }

    let four = BigRational::from_integer(4.into());
    let six = BigRational::from_integer(6.into());
    let shift = -coeff(3) / four.clone();
    if shift.is_zero() {
        return None;
    }

    let shift_sq = shift.clone() * shift.clone();
    let shift_cu = shift_sq.clone() * shift.clone();
    let shift_fourth = shift_sq.clone() * shift_sq.clone();
    if coeff(2) != -(six * shift_sq) || coeff(1) != -(four * shift_cu) {
        return None;
    }

    let width = coeff(0) + shift_fourth;
    if !width.is_positive() {
        return None;
    }

    let width_expr = ctx.add(Expr::Number(width));
    let base = shifted_var_expr_for_display(ctx, var, &shift);
    let exp = ctx.num(4);
    let power = ctx.add(Expr::Pow(base, exp));
    Some(ctx.add(Expr::Sub(width_expr, power)))
}

fn compact_expanded_monic_low_degree_power_gap_for_display(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let variables = cas_ast::collect_variables(ctx, expr);
    if variables.len() != 1 {
        return None;
    }
    let var = variables.iter().next()?;
    let poly = Polynomial::from_expr(ctx, expr, var.as_str()).ok()?;
    let degree = poly.degree();
    if degree < 4 || poly.leading_coeff() != BigRational::one() {
        return None;
    }

    for power in 4..=8_i64 {
        let power_usize = power as usize;
        if degree % power_usize != 0 {
            continue;
        }
        let base_degree = degree / power_usize;
        if !(1..=2).contains(&base_degree) {
            continue;
        }

        let candidate = monic_power_root_candidate(&poly, var, power, base_degree)?;
        let candidate_power = polynomial_positive_power(&candidate, power);
        let residual = poly.sub(&candidate_power);
        if residual.is_zero() || residual.degree() > 0 {
            continue;
        }
        let residual_constant = residual.coeffs.first()?.clone();
        if residual_constant.is_zero() {
            continue;
        }

        let base = candidate.to_expr(ctx);
        let exp = ctx.num(power);
        let power_expr = ctx.add(Expr::Pow(base, exp));
        let offset = ctx.add(Expr::Number(residual_constant.abs()));
        return Some(if residual_constant.is_positive() {
            ctx.add(Expr::Add(power_expr, offset))
        } else {
            ctx.add(Expr::Sub(power_expr, offset))
        });
    }

    None
}

fn compact_expanded_negative_monic_low_degree_power_gap_for_display(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let variables = cas_ast::collect_variables(ctx, expr);
    if variables.len() != 1 {
        return None;
    }
    let var = variables.iter().next()?;
    let poly = Polynomial::from_expr(ctx, expr, var.as_str()).ok()?;
    let degree = poly.degree();
    if degree < 4 || poly.leading_coeff() != -BigRational::one() {
        return None;
    }

    let neg_poly = poly.neg();
    for power in 2..=8_i64 {
        let power_usize = power as usize;
        if degree % power_usize != 0 {
            continue;
        }
        let base_degree = degree / power_usize;
        if !matches!((power, base_degree), (2, 2) | (4..=8, 1 | 2)) {
            continue;
        }

        let candidate = monic_power_root_candidate(&neg_poly, var, power, base_degree)?;
        if power == 2 && !polynomial_has_additive_shape(&candidate) {
            continue;
        }
        let candidate_power = polynomial_positive_power(&candidate, power);
        let residual = neg_poly.sub(&candidate_power);
        if residual.is_zero() || residual.degree() > 0 {
            continue;
        }
        let residual_constant = residual.coeffs.first()?.clone();
        if !residual_constant.is_negative() {
            continue;
        }

        let base = candidate.to_expr(ctx);
        let exp = ctx.num(power);
        let power_expr = ctx.add(Expr::Pow(base, exp));
        let offset = ctx.add(Expr::Number(residual_constant.abs()));
        return Some(ctx.add(Expr::Sub(offset, power_expr)));
    }

    None
}

fn compact_expanded_negative_scaled_low_degree_power_gap_for_display(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let variables = cas_ast::collect_variables(ctx, expr);
    if variables.len() != 1 {
        return None;
    }
    let var = variables.iter().next()?;
    let poly = Polynomial::from_expr(ctx, expr, var.as_str()).ok()?;
    let degree = poly.degree();
    if degree < 4 || !poly.leading_coeff().is_negative() {
        return None;
    }

    let scale = -poly.leading_coeff();
    if scale.is_one() || !scale.is_integer() {
        return None;
    }

    let normalized_positive = poly.neg().div_scalar(&scale);
    for power in 2..=8_i64 {
        let power_usize = power as usize;
        if degree % power_usize != 0 {
            continue;
        }
        let base_degree = degree / power_usize;
        if !matches!((power, base_degree), (2, 2) | (4..=8, 1 | 2)) {
            continue;
        }

        let candidate = monic_power_root_candidate(&normalized_positive, var, power, base_degree)?;
        if power == 2 && !polynomial_has_additive_shape(&candidate) {
            continue;
        }

        let candidate_power = polynomial_positive_power(&candidate, power);
        let residual = normalized_positive.sub(&candidate_power);
        if residual.is_zero() || residual.degree() > 0 {
            continue;
        }
        let residual_constant = residual.coeffs.first()?.clone();
        if !residual_constant.is_negative() {
            continue;
        }

        let offset = ctx.add(Expr::Number(residual_constant.abs() * scale.clone()));
        let base = candidate.to_expr(ctx);
        let exp = ctx.num(power);
        let power_expr = ctx.add(Expr::Pow(base, exp));
        let scale_expr = ctx.add(Expr::Number(scale.clone()));
        let scaled_power = ctx.add(Expr::Mul(scale_expr, power_expr));
        return Some(ctx.add(Expr::Sub(offset, scaled_power)));
    }

    None
}

fn polynomial_has_additive_shape(poly: &Polynomial) -> bool {
    let nonzero_terms = poly.coeffs.iter().filter(|coeff| !coeff.is_zero()).count();
    nonzero_terms >= 2
}

fn shifted_var_expr_for_display(ctx: &mut Context, var: &str, shift: &BigRational) -> ExprId {
    let var_expr = ctx.var(var);
    if shift.is_zero() {
        return var_expr;
    }

    let shift_expr = ctx.add(Expr::Number(shift.abs()));
    if shift.is_positive() {
        ctx.add(Expr::Add(var_expr, shift_expr))
    } else {
        ctx.add(Expr::Sub(var_expr, shift_expr))
    }
}

fn positive_integer_power_parts(ctx: &Context, expr: ExprId) -> Option<(ExprId, i64)> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let Expr::Number(exp_num) = ctx.get(*exp) else {
        return None;
    };

    if !exp_num.is_integer() {
        return None;
    }

    let exp_i64 = exp_num.to_integer().to_i64()?;
    (exp_i64 > 0).then_some((*base, exp_i64))
}

fn compact_expanded_monic_low_degree_power_for_display(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let variables = cas_ast::collect_variables(ctx, expr);
    if variables.len() != 1 {
        return None;
    }
    let var = variables.iter().next()?;
    let poly = Polynomial::from_expr(ctx, expr, var.as_str()).ok()?;
    let degree = poly.degree();
    if degree < 2 || poly.leading_coeff() != BigRational::one() {
        return None;
    }

    for power in 2..=8_i64 {
        let power_usize = power as usize;
        if degree % power_usize != 0 {
            continue;
        }
        let base_degree = degree / power_usize;
        if !(1..=2).contains(&base_degree) {
            continue;
        }

        let candidate = monic_power_root_candidate(&poly, var, power, base_degree)?;
        if polynomial_positive_power(&candidate, power) == poly {
            return Some(candidate.to_expr(ctx));
        }
    }

    None
}

fn monic_power_root_candidate(
    poly: &Polynomial,
    var: &str,
    power: i64,
    base_degree: usize,
) -> Option<Polynomial> {
    let coeff = |degree: usize| {
        poly.coeffs
            .get(degree)
            .cloned()
            .unwrap_or_else(BigRational::zero)
    };
    let degree = poly.degree();
    let power_rat = BigRational::from_integer(power.into());

    match base_degree {
        1 => {
            let constant = coeff(degree - 1) / power_rat;
            Some(Polynomial::new(
                vec![constant, BigRational::one()],
                var.to_string(),
            ))
        }
        2 => {
            let linear = coeff(degree - 1) / power_rat.clone();
            let pair_count = BigRational::from_integer(((power * (power - 1)) / 2).into());
            let constant =
                (coeff(degree - 2) - pair_count * linear.clone() * linear.clone()) / power_rat;
            Some(Polynomial::new(
                vec![constant, linear, BigRational::one()],
                var.to_string(),
            ))
        }
        _ => None,
    }
}

fn polynomial_positive_power(poly: &Polynomial, power: i64) -> Polynomial {
    let mut out = Polynomial::one(poly.var.clone());
    for _ in 0..power {
        out = out.mul(poly);
    }
    out
}

fn is_small_additive_shape(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
        && additive_leaf_count_up_to(ctx, expr, 2).is_some()
}

fn additive_leaf_count_up_to(ctx: &Context, expr: ExprId, limit: usize) -> Option<usize> {
    if limit == 0 {
        return None;
    }

    match ctx.get(expr) {
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            let left_count = additive_leaf_count_up_to(ctx, *left, limit)?;
            let remaining = limit.checked_sub(left_count)?;
            let right_count = additive_leaf_count_up_to(ctx, *right, remaining)?;
            let total = left_count + right_count;
            (total <= limit).then_some(total)
        }
        _ => Some(1),
    }
}

fn is_nonnegative_under_display_conditions_or_factored(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
    skip_index: usize,
    expr: ExprId,
    depth: usize,
) -> bool {
    if is_nonnegative_under_display_conditions(ctx, conditions, skip_index, expr, depth) {
        return true;
    }

    if unit_reciprocal_square_gap_is_nonnegative(ctx, conditions, skip_index, expr, depth) {
        return true;
    }

    if depth == 0 {
        return false;
    }

    let factored = factor(ctx, expr);
    factored != expr
        && is_nonnegative_under_display_conditions(ctx, conditions, skip_index, factored, depth - 1)
}

fn unit_reciprocal_square_gap_is_nonnegative(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
    skip_index: usize,
    expr: ExprId,
    depth: usize,
) -> bool {
    if depth == 0 {
        return false;
    }

    let Some(base) = one_minus_unit_reciprocal_square_base(ctx, expr) else {
        return false;
    };

    if !is_positive_under_display_conditions(ctx, conditions, skip_index, base, depth - 1) {
        return false;
    }

    let two = ctx.num(2);
    let base_squared = ctx.add(Expr::Pow(base, two));
    let one = ctx.num(1);
    let gap = ctx.add(Expr::Sub(base_squared, one));
    let normalized_gap = normalize_condition_expr_preserve_sign(ctx, gap);

    is_nonnegative_under_display_conditions(ctx, conditions, skip_index, normalized_gap, depth - 1)
}

fn one_minus_unit_reciprocal_square_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Sub(left, right) if as_rational_const(ctx, *left).is_some_and(|n| n.is_one()) => {
            unit_reciprocal_square_base(ctx, *right)
        }
        Expr::Add(left, right) => negative_unit_reciprocal_square_base(ctx, *right)
            .filter(|_| as_rational_const(ctx, *left).is_some_and(|n| n.is_one()))
            .or_else(|| {
                negative_unit_reciprocal_square_base(ctx, *left)
                    .filter(|_| as_rational_const(ctx, *right).is_some_and(|n| n.is_one()))
            }),
        _ => None,
    }
}

fn negative_unit_reciprocal_square_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Neg(inner) => unit_reciprocal_square_base(ctx, *inner),
        Expr::Mul(left, right)
            if as_rational_const(ctx, *left).is_some_and(|n| n == -BigRational::one()) =>
        {
            unit_reciprocal_square_base(ctx, *right)
        }
        Expr::Mul(left, right)
            if as_rational_const(ctx, *right).is_some_and(|n| n == -BigRational::one()) =>
        {
            unit_reciprocal_square_base(ctx, *left)
        }
        _ => None,
    }
}

fn unit_reciprocal_square_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Div(num, den) if as_rational_const(ctx, *num).is_some_and(|n| n.is_one()) => {
            square_base(ctx, *den)
        }
        Expr::Pow(inner, exp)
            if as_rational_const(ctx, *exp)
                .is_some_and(|n| n.is_integer() && n.to_integer() == 2.into()) =>
        {
            unit_reciprocal_base(ctx, *inner)
        }
        Expr::Pow(base, exp)
            if as_rational_const(ctx, *exp)
                .is_some_and(|n| n.is_integer() && n.to_integer() == (-2).into()) =>
        {
            Some(*base)
        }
        _ => None,
    }
}

fn unit_reciprocal_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Div(num, den) if as_rational_const(ctx, *num).is_some_and(|n| n.is_one()) => {
            Some(*den)
        }
        Expr::Pow(base, exp)
            if as_rational_const(ctx, *exp)
                .is_some_and(|n| n.is_integer() && n.to_integer() == (-1).into()) =>
        {
            Some(*base)
        }
        _ => None,
    }
}

fn square_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp)
            if as_rational_const(ctx, *exp)
                .is_some_and(|n| n.is_integer() && n.to_integer() == 2.into()) =>
        {
            Some(*base)
        }
        _ => None,
    }
}

fn extract_sqrt_like_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Some(arg) = extract_sqrt_argument_view(ctx, expr) {
        return Some(arg);
    }

    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let half = num_rational::BigRational::new(1.into(), 2.into());
    (as_rational_const(ctx, *exp).as_ref() == Some(&half)).then_some(*base)
}

fn exprs_equivalent_or_same_sqrt_like_base(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    exprs_equivalent_up_to_sign(ctx, left, right)
        || match (
            extract_sqrt_like_base(ctx, left),
            extract_sqrt_like_base(ctx, right),
        ) {
            (Some(left_base), Some(right_base)) => exprs_equivalent(ctx, left_base, right_base),
            _ => false,
        }
}

fn positive_ordered_exprs_equivalent(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    is_positive_multiple_of(ctx, left, right) || is_positive_multiple_of(ctx, right, left)
}

fn positive_exprs_equivalent_or_same_sqrt_like_base(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    positive_ordered_exprs_equivalent(ctx, left, right)
        || match (
            extract_sqrt_like_base(ctx, left),
            extract_sqrt_like_base(ctx, right),
        ) {
            (Some(left_base), Some(right_base)) => exprs_equivalent(ctx, left_base, right_base),
            _ => false,
        }
}

fn positive_target_from_nonzero_and_nonnegative(
    ctx: &mut Context,
    nonzero_expr: ExprId,
    nonnegative_expr: ExprId,
) -> Option<ExprId> {
    let nonzero_core = extract_abs_argument_view(ctx, nonzero_expr).unwrap_or(nonzero_expr);

    if exprs_equivalent_up_to_sign(ctx, nonzero_core, nonnegative_expr) {
        return Some(nonnegative_expr);
    }

    if exprs_equivalent_up_to_nonzero_scalar(ctx, nonzero_core, nonnegative_expr) {
        return Some(nonnegative_expr);
    }

    if let Some(nonzero_base) = extract_sqrt_like_base(ctx, nonzero_core) {
        if positive_ordered_exprs_equivalent(ctx, nonzero_base, nonnegative_expr) {
            return Some(nonnegative_expr);
        }
    }

    if nonzero_product_contains_nonnegative_factor_with_positive_cofactors(
        ctx,
        nonzero_core,
        nonnegative_expr,
    ) {
        return Some(nonnegative_expr);
    }

    None
}

fn nonzero_product_contains_nonnegative_factor_with_positive_cofactors(
    ctx: &mut Context,
    nonzero_expr: ExprId,
    nonnegative_expr: ExprId,
) -> bool {
    let factored = factor(ctx, nonzero_expr);
    let mut factors = Vec::new();
    collect_nonzero_atomic_factors(ctx, factored, &mut factors);
    if factors.len() <= 1 {
        return polynomial_nonzero_quotient_is_intrinsically_positive(
            ctx,
            nonzero_expr,
            nonnegative_expr,
        );
    }

    factors
        .iter()
        .enumerate()
        .any(|(factor_index, factor_expr)| {
            exprs_equivalent_up_to_nonzero_scalar(ctx, *factor_expr, nonnegative_expr)
                && factors.iter().enumerate().all(|(other_index, other_expr)| {
                    other_index == factor_index || is_intrinsically_positive_real(ctx, *other_expr)
                })
        })
        || polynomial_nonzero_quotient_is_intrinsically_positive(
            ctx,
            nonzero_expr,
            nonnegative_expr,
        )
}

fn polynomial_nonzero_quotient_is_intrinsically_positive(
    ctx: &mut Context,
    nonzero_expr: ExprId,
    nonnegative_expr: ExprId,
) -> bool {
    let mut variables = cas_ast::collect_variables(ctx, nonzero_expr);
    variables.extend(cas_ast::collect_variables(ctx, nonnegative_expr));
    if variables.len() != 1 {
        return false;
    }

    let Some(var) = variables.iter().next() else {
        return false;
    };
    let Ok(nonzero_poly) = Polynomial::from_expr(ctx, nonzero_expr, var.as_str()) else {
        return false;
    };
    let Ok(nonnegative_poly) = Polynomial::from_expr(ctx, nonnegative_expr, var.as_str()) else {
        return false;
    };
    let Ok((quotient, remainder)) = nonzero_poly.div_rem(&nonnegative_poly) else {
        return false;
    };
    if !remainder.is_zero() {
        return false;
    }

    let quotient_expr = quotient.to_expr(ctx);
    is_intrinsically_positive_real(ctx, quotient_expr)
}

fn combine_nonzero_nonnegative_into_positive(
    ctx: &mut Context,
    conditions: &mut Vec<ImplicitCondition>,
) {
    let mut to_remove: Vec<usize> = Vec::new();
    let mut to_add: Vec<ImplicitCondition> = Vec::new();

    for (i, cond) in conditions.iter().enumerate() {
        let ImplicitCondition::NonZero(nz_expr) = cond else {
            continue;
        };

        for (j, other) in conditions.iter().enumerate() {
            if i == j {
                continue;
            }

            let ImplicitCondition::NonNegative(nn_expr) = other else {
                continue;
            };

            let Some(target) =
                positive_target_from_nonzero_and_nonnegative(ctx, *nz_expr, *nn_expr)
            else {
                continue;
            };

            to_remove.push(i);
            to_remove.push(j);

            let positive = normalize_condition(ctx, &ImplicitCondition::Positive(target));
            let already_present = conditions.iter().enumerate().any(|(idx, existing)| {
                !to_remove.contains(&idx) && conditions_equivalent(ctx, existing, &positive)
            }) || to_add
                .iter()
                .any(|existing| conditions_equivalent(ctx, existing, &positive));
            if !already_present {
                to_add.push(positive);
            }
            break;
        }
    }

    to_remove.sort();
    to_remove.dedup();
    for idx in to_remove.into_iter().rev() {
        conditions.remove(idx);
    }
    conditions.extend(to_add);
}

fn combine_factored_nonzero_nonnegative_into_positive(
    ctx: &mut Context,
    conditions: &mut Vec<ImplicitCondition>,
) {
    let mut to_remove: Vec<usize> = Vec::new();
    let mut to_add: Vec<ImplicitCondition> = Vec::new();

    for (i, cond) in conditions.iter().enumerate() {
        let ImplicitCondition::NonNegative(nn_expr) = cond else {
            continue;
        };

        let factored = factor(ctx, *nn_expr);
        let mut factors = Vec::new();
        collect_nonzero_atomic_factors(ctx, factored, &mut factors);
        if factors.is_empty() {
            continue;
        }

        let mut matched_indices = Vec::new();
        let mut all_factors_matched = true;
        for factor_expr in factors {
            let Some((idx, _)) = conditions.iter().enumerate().find(|(idx, candidate)| {
                *idx != i
                    && !matched_indices.contains(idx)
                    && matches!(
                        candidate,
                        ImplicitCondition::NonZero(nz_expr)
                            if exprs_equivalent_or_same_sqrt_like_base(ctx, *nz_expr, factor_expr)
                                || exprs_equivalent_up_to_nonzero_scalar(ctx, *nz_expr, factor_expr)
                    )
            }) else {
                all_factors_matched = false;
                break;
            };
            matched_indices.push(idx);
        }

        if !all_factors_matched || matched_indices.is_empty() {
            continue;
        }

        if let Some(expanded_positive) =
            positive_condition_equivalent_nonzero_conditions(ctx, *nn_expr)
        {
            let expanded_already_present = expanded_positive.iter().all(|expanded| {
                conditions.iter().enumerate().any(|(idx, candidate)| {
                    idx != i && conditions_equivalent(ctx, candidate, expanded)
                })
            });
            if expanded_already_present {
                to_remove.push(i);
                continue;
            }
        }

        to_remove.push(i);
        to_remove.extend(matched_indices);

        let positive = normalize_condition(ctx, &ImplicitCondition::Positive(*nn_expr));
        let already_present = conditions.iter().enumerate().any(|(idx, existing)| {
            !to_remove.contains(&idx) && conditions_equivalent(ctx, existing, &positive)
        }) || to_add
            .iter()
            .any(|existing| conditions_equivalent(ctx, existing, &positive));
        if !already_present {
            to_add.push(positive);
        }
    }

    to_remove.sort();
    to_remove.dedup();
    for idx in to_remove.into_iter().rev() {
        conditions.remove(idx);
    }
    conditions.extend(to_add);
}

fn dedupe_conditions_in_place(ctx: &Context, conditions: &mut Vec<ImplicitCondition>) {
    let mut deduped = Vec::new();
    for condition in conditions.drain(..) {
        if !deduped
            .iter()
            .any(|existing| conditions_equivalent(ctx, existing, &condition))
        {
            deduped.push(condition);
        }
    }
    *conditions = deduped;
}

fn rewrite_sign_quotients_with_positive_components(
    ctx: &mut Context,
    conditions: &mut Vec<ImplicitCondition>,
) {
    let mut replacements: Vec<(usize, ImplicitCondition)> = Vec::new();

    for (idx, condition) in conditions.iter().enumerate() {
        let replacement = match condition {
            ImplicitCondition::NonNegative(expr) => match ctx.get(*expr) {
                Expr::Div(num, den) => Some((idx, false, *num, *den)),
                _ => None,
            },
            ImplicitCondition::Positive(expr) => match ctx.get(*expr) {
                Expr::Div(num, den) => Some((idx, true, *num, *den)),
                _ => None,
            },
            ImplicitCondition::NonZero(_) => None,
        };

        let Some((idx, is_positive, num, den)) = replacement else {
            continue;
        };

        if is_positive_under_display_conditions_or_factored(
            ctx,
            conditions,
            idx,
            den,
            DISPLAY_SIGN_PROOF_DEPTH,
        ) {
            let replacement = if is_positive {
                normalize_condition(ctx, &ImplicitCondition::Positive(num))
            } else {
                normalize_condition(ctx, &ImplicitCondition::NonNegative(num))
            };
            replacements.push((idx, replacement));
            continue;
        }

        if is_positive_under_display_conditions_or_factored(
            ctx,
            conditions,
            idx,
            num,
            DISPLAY_SIGN_PROOF_DEPTH,
        ) {
            replacements.push((
                idx,
                normalize_condition(ctx, &ImplicitCondition::Positive(den)),
            ));
        }
    }

    if replacements.is_empty() {
        return;
    }

    for (idx, replacement) in replacements {
        conditions[idx] = replacement;
    }

    dedupe_conditions_in_place(ctx, conditions);
}

fn is_positive_under_display_conditions_or_factored(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
    skip_index: usize,
    expr: ExprId,
    depth: usize,
) -> bool {
    if is_positive_under_display_conditions(ctx, conditions, skip_index, expr, depth) {
        return true;
    }

    if depth == 0 {
        return false;
    }

    let factored = factor(ctx, expr);
    factored != expr
        && is_positive_under_display_conditions(ctx, conditions, skip_index, factored, depth - 1)
}

fn nonzero_is_dominated_by_positive_condition(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
    expr: ExprId,
) -> bool {
    let stripped = strip_nonzero_scalar_factors_for_display(ctx, expr);
    let normalized_expr = normalize_condition_expr(ctx, stripped);

    conditions.iter().any(|condition| {
        let ImplicitCondition::Positive(pos_expr) = condition else {
            return false;
        };
        let normalized_positive = normalize_condition_expr_preserve_sign(ctx, *pos_expr);
        exprs_equivalent_up_to_nonzero_scalar(ctx, normalized_expr, normalized_positive)
            || positive_condition_dominates_affine_nonzero_offset(
                ctx,
                normalized_positive,
                normalized_expr,
            )
            || positive_polynomial_condition_contains_nonzero_factor(
                ctx,
                normalized_positive,
                normalized_expr,
            )
    })
}

fn positive_condition_dominates_affine_nonzero_offset(
    ctx: &Context,
    positive_expr: ExprId,
    nonzero_expr: ExprId,
) -> bool {
    use cas_math::multipoly::{multipoly_from_expr, PolyBudget};

    let budget = PolyBudget {
        max_terms: 16,
        max_total_degree: 1,
        max_pow_exp: 1,
    };
    let (Ok(positive_poly), Ok(nonzero_poly)) = (
        multipoly_from_expr(ctx, positive_expr, &budget),
        multipoly_from_expr(ctx, nonzero_expr, &budget),
    ) else {
        return false;
    };
    if positive_poly.vars != nonzero_poly.vars {
        return false;
    }

    let Some(scale) = nonzero_nonconstant_scale(&positive_poly, &nonzero_poly) else {
        return false;
    };
    let scaled_positive = positive_poly.mul_scalar(&scale);
    let Ok(offset_poly) = nonzero_poly.sub(&scaled_positive) else {
        return false;
    };

    offset_poly.constant_value().is_some_and(|offset| {
        (scale.is_positive() && offset >= BigRational::zero())
            || (scale.is_negative() && offset <= BigRational::zero())
    })
}

fn positive_condition_dominates_affine_positive_offset(
    ctx: &Context,
    positive_expr: ExprId,
    derived_positive_expr: ExprId,
) -> bool {
    use cas_math::multipoly::{multipoly_from_expr, PolyBudget};

    let budget = PolyBudget {
        max_terms: 16,
        max_total_degree: 1,
        max_pow_exp: 1,
    };
    let (Ok(positive_poly), Ok(derived_poly)) = (
        multipoly_from_expr(ctx, positive_expr, &budget),
        multipoly_from_expr(ctx, derived_positive_expr, &budget),
    ) else {
        return false;
    };
    if positive_poly.vars != derived_poly.vars {
        return false;
    }

    let Some(scale) = nonzero_nonconstant_scale(&positive_poly, &derived_poly) else {
        return false;
    };
    if !scale.is_positive() {
        return false;
    }

    let scaled_positive = positive_poly.mul_scalar(&scale);
    let Ok(offset_poly) = derived_poly.sub(&scaled_positive) else {
        return false;
    };

    offset_poly
        .constant_value()
        .is_some_and(|offset| offset >= BigRational::zero())
}

fn one_minus_positive_const_over_expr_parts(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational)> {
    match ctx.get(expr).clone() {
        Expr::Sub(left, right) if is_one_constant(ctx, left) => {
            positive_const_over_expr_parts(ctx, right)
        }
        Expr::Add(left, right) if is_one_constant(ctx, left) => {
            negative_const_over_expr_parts(ctx, right)
        }
        Expr::Add(left, right) if is_one_constant(ctx, right) => {
            negative_const_over_expr_parts(ctx, left)
        }
        _ => None,
    }
}

fn positive_const_over_expr_parts(ctx: &Context, expr: ExprId) -> Option<(ExprId, BigRational)> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let constant = as_rational_const(ctx, num)?;
    constant.is_positive().then_some((den, constant))
}

fn negative_const_over_expr_parts(ctx: &Context, expr: ExprId) -> Option<(ExprId, BigRational)> {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => positive_const_over_expr_parts(ctx, inner),
        Expr::Div(num, den) => {
            let constant = as_rational_const(ctx, num)?;
            constant.is_negative().then_some((den, -constant))
        }
        _ => None,
    }
}

fn positive_condition_dominates_reciprocal_offset_nonnegative(
    ctx: &mut Context,
    positive_expr: ExprId,
    nonnegative_expr: ExprId,
) -> bool {
    let Some((den, offset)) = one_minus_positive_const_over_expr_parts(ctx, nonnegative_expr)
    else {
        return false;
    };

    let offset_expr = ctx.add(Expr::Number(offset));
    let gap = ctx.add(Expr::Sub(den, offset_expr));
    let normalized_positive = normalize_condition_expr_preserve_sign(ctx, positive_expr);
    let normalized_gap = normalize_condition_expr_preserve_sign(ctx, gap);

    exprs_equivalent_up_to_nonzero_scalar(ctx, normalized_positive, normalized_gap)
}

fn nonzero_nonconstant_scale(source: &MultiPoly, target: &MultiPoly) -> Option<BigRational> {
    let mut scale: Option<BigRational> = None;
    let mut saw_nonconstant = false;

    for (source_coeff, source_mono) in &source.terms {
        if source_mono.iter().all(|exp| *exp == 0) {
            continue;
        }
        saw_nonconstant = true;
        let (target_coeff, _) = target
            .terms
            .iter()
            .find(|(_, target_mono)| target_mono == source_mono)?;
        let term_scale = target_coeff / source_coeff;
        if term_scale.is_zero() {
            return None;
        }
        match &scale {
            Some(existing) if existing != &term_scale => return None,
            Some(_) => {}
            None => scale = Some(term_scale),
        }
    }

    if !saw_nonconstant {
        return None;
    }

    for (_, target_mono) in &target.terms {
        if target_mono.iter().all(|exp| *exp == 0) {
            continue;
        }
        if !source
            .terms
            .iter()
            .any(|(_, source_mono)| source_mono == target_mono)
        {
            return None;
        }
    }

    scale
}

fn exprs_equivalent_up_to_nonzero_scalar(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    use cas_math::multipoly::{multipoly_from_expr, PolyBudget};

    if exprs_equivalent_up_to_sign(ctx, left, right) {
        return true;
    }

    let budget = PolyBudget {
        max_terms: 50,
        max_total_degree: 20,
        max_pow_exp: 10,
    };

    let (Ok(left_poly), Ok(right_poly)) = (
        multipoly_from_expr(ctx, left, &budget),
        multipoly_from_expr(ctx, right, &budget),
    ) else {
        return false;
    };

    if left_poly.is_zero() || right_poly.is_zero() {
        return false;
    }

    let (_, left_primitive) = left_poly.primitive_part();
    let (_, right_primitive) = right_poly.primitive_part();
    left_primitive == right_primitive || left_primitive == right_primitive.neg()
}

fn positive_polynomial_condition_contains_nonzero_factor(
    ctx: &Context,
    positive_expr: ExprId,
    nonzero_expr: ExprId,
) -> bool {
    use cas_math::multipoly::{multipoly_from_expr, PolyBudget};

    if !is_polynomial_condition_syntax(ctx, positive_expr)
        || !is_polynomial_condition_syntax(ctx, nonzero_expr)
    {
        return false;
    }

    let budget = PolyBudget {
        max_terms: 50,
        max_total_degree: 20,
        max_pow_exp: 10,
    };
    let (Ok(positive_poly), Ok(nonzero_poly)) = (
        multipoly_from_expr(ctx, positive_expr, &budget),
        multipoly_from_expr(ctx, nonzero_expr, &budget),
    ) else {
        return false;
    };

    if positive_poly.is_zero()
        || positive_poly.is_constant()
        || nonzero_poly.is_zero()
        || nonzero_poly.is_constant()
        || positive_poly.vars.len() != 1
    {
        return false;
    }

    let mut vars = positive_poly.vars.clone();
    vars.extend(nonzero_poly.vars.iter().cloned());
    vars.sort();
    vars.dedup();

    let positive_poly = positive_poly.align_vars(&vars);
    let nonzero_poly = nonzero_poly.align_vars(&vars);
    positive_poly.div_exact(&nonzero_poly).is_some()
}

fn is_polynomial_condition_syntax(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) => true,
        Expr::Neg(inner) => is_polynomial_condition_syntax(ctx, *inner),
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            is_polynomial_condition_syntax(ctx, *left)
                && is_polynomial_condition_syntax(ctx, *right)
        }
        Expr::Pow(base, exp) => {
            let Expr::Number(power) = ctx.get(*exp) else {
                return false;
            };
            power.is_integer() && !power.is_negative() && is_polynomial_condition_syntax(ctx, *base)
        }
        Expr::Div(_, _)
        | Expr::Function(_, _)
        | Expr::Matrix { .. }
        | Expr::SessionRef(_)
        | Expr::Hold(_) => false,
    }
}

fn nonzero_is_dominated_by_nonzero_factors(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
    skip_index: usize,
    expr: ExprId,
) -> bool {
    use cas_math::multipoly::{multipoly_from_expr, PolyBudget};

    let budget = PolyBudget {
        max_terms: 50,
        max_total_degree: 20,
        max_pow_exp: 10,
    };

    let normalized_expr = normalize_condition_expr(ctx, expr);
    let Ok(mut remaining) = multipoly_from_expr(ctx, normalized_expr, &budget) else {
        return false;
    };
    if remaining.is_zero() {
        return false;
    }

    let mut known_nonzero_factors = Vec::new();
    for (idx, condition) in conditions.iter().enumerate() {
        if idx == skip_index {
            continue;
        }

        let ImplicitCondition::NonZero(other_expr) = condition else {
            continue;
        };
        let normalized_other = normalize_condition_expr(ctx, *other_expr);
        let Ok(factor_poly) = multipoly_from_expr(ctx, normalized_other, &budget) else {
            continue;
        };
        if factor_poly.is_zero()
            || factor_poly.is_constant()
            || !factor_poly.vars.iter().all(|var| {
                remaining
                    .vars
                    .iter()
                    .any(|remaining_var| remaining_var == var)
            })
        {
            continue;
        }
        known_nonzero_factors.push(factor_poly);
    }

    if known_nonzero_factors.is_empty() {
        return false;
    }

    let (_, remaining_primitive) = remaining.primitive_part();
    known_nonzero_factors.retain(|factor_poly| {
        let (_, factor_primitive) = factor_poly.primitive_part();
        factor_primitive != remaining_primitive && factor_primitive != remaining_primitive.neg()
    });

    if known_nonzero_factors.is_empty() {
        return false;
    }

    let max_divisions = remaining.total_degree() as usize;
    for _ in 0..max_divisions {
        if remaining.is_constant() {
            return remaining
                .constant_value()
                .is_some_and(|constant| !constant.is_zero());
        }

        let mut divided = false;
        for factor_poly in &known_nonzero_factors {
            let aligned_factor = factor_poly.align_vars(&remaining.vars);
            if let Some(quotient) = remaining.div_exact(&aligned_factor) {
                if quotient == remaining {
                    continue;
                }
                remaining = quotient;
                divided = true;
                break;
            }
        }

        if !divided {
            break;
        }
    }

    remaining.is_constant()
        && remaining
            .constant_value()
            .is_some_and(|constant| !constant.is_zero())
}

fn positive_even_power_gap_forces_nonzero(
    ctx: &Context,
    positive_expr: ExprId,
    nonzero_expr: ExprId,
) -> bool {
    if let Some(base) = positive_even_power_gap_base(ctx, positive_expr) {
        if exprs_equivalent_up_to_sign(ctx, base, nonzero_expr) {
            return true;
        }
    }

    positive_quadratic_gap_forces_nonzero(ctx, positive_expr, nonzero_expr)
}

fn positive_quadratic_gap_forces_nonzero(
    ctx: &Context,
    positive_expr: ExprId,
    nonzero_expr: ExprId,
) -> bool {
    use cas_math::multipoly::{multipoly_from_expr, PolyBudget};

    let budget = PolyBudget {
        max_terms: 50,
        max_total_degree: 20,
        max_pow_exp: 10,
    };

    let (Ok(positive_poly), Ok(nonzero_poly)) = (
        multipoly_from_expr(ctx, positive_expr, &budget),
        multipoly_from_expr(ctx, nonzero_expr, &budget),
    ) else {
        return false;
    };

    if nonzero_poly.is_zero() {
        return false;
    }

    let mut vars = positive_poly.vars.clone();
    vars.extend(nonzero_poly.vars.iter().cloned());
    vars.sort();
    vars.dedup();

    let positive_poly = positive_poly.align_vars(&vars);
    let nonzero_poly = nonzero_poly.align_vars(&vars);
    let Ok(nonzero_square) = nonzero_poly.mul(&nonzero_poly, &budget) else {
        return false;
    };

    let Some(scale) = nonconstant_polynomial_scale(&positive_poly, &nonzero_square) else {
        return false;
    };
    if !scale.is_positive() {
        return false;
    }

    let scaled_square = nonzero_square.mul_scalar(&scale);
    let Ok(remainder) = positive_poly.sub(&scaled_square) else {
        return false;
    };
    remainder
        .constant_value()
        .is_some_and(|constant| constant.is_negative())
}

fn nonconstant_polynomial_scale(source: &MultiPoly, target: &MultiPoly) -> Option<BigRational> {
    if source.vars != target.vars {
        return None;
    }

    let zero_monomial = vec![0; source.vars.len()];
    let source_terms = source.to_map();
    let target_terms = target.to_map();
    let mut scale: Option<BigRational> = None;

    for (monomial, target_coeff) in &target_terms {
        if monomial == &zero_monomial {
            continue;
        }
        let source_coeff = source_terms.get(monomial)?;
        let candidate = source_coeff.clone() / target_coeff.clone();
        if let Some(existing) = &scale {
            if existing != &candidate {
                return None;
            }
        } else {
            scale = Some(candidate);
        }
    }

    for monomial in source_terms.keys() {
        if monomial != &zero_monomial && !target_terms.contains_key(monomial) {
            return None;
        }
    }

    scale
}

fn positive_even_power_gap_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Sub(left, right) => as_rational_const(ctx, *right)
            .is_some_and(|constant| constant.is_positive())
            .then(|| positive_multiple_even_power_base(ctx, *left))
            .flatten(),
        Expr::Add(left, right) => even_power_minus_positive_constant(ctx, *left, *right)
            .or_else(|| even_power_minus_positive_constant(ctx, *right, *left)),
        _ => None,
    }
}

fn even_power_minus_positive_constant(
    ctx: &Context,
    power_term: ExprId,
    constant_term: ExprId,
) -> Option<ExprId> {
    if !as_rational_const(ctx, constant_term).is_some_and(|constant| constant.is_negative()) {
        return None;
    }

    positive_multiple_even_power_base(ctx, power_term)
}

fn positive_multiple_even_power_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Some(base) = extract_even_positive_power_base(ctx, expr) {
        return Some(base);
    }

    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };

    positive_scalar_times_even_power_base(ctx, *left, *right)
        .or_else(|| positive_scalar_times_even_power_base(ctx, *right, *left))
}

fn positive_scalar_times_even_power_base(
    ctx: &Context,
    scalar: ExprId,
    power_term: ExprId,
) -> Option<ExprId> {
    if !as_rational_const(ctx, scalar).is_some_and(|constant| constant.is_positive()) {
        return None;
    }

    extract_even_positive_power_base(ctx, power_term)
}

fn has_nonzero_display_condition(
    ctx: &Context,
    conditions: &[ImplicitCondition],
    skip_index: usize,
    expr: ExprId,
) -> bool {
    conditions.iter().enumerate().any(|(idx, cond)| {
        idx != skip_index
            && matches!(
                cond,
                ImplicitCondition::NonZero(other)
                    if exprs_equivalent_or_same_sqrt_like_base(ctx, *other, expr)
            )
    })
}

fn has_nonnegative_display_condition(
    ctx: &Context,
    conditions: &[ImplicitCondition],
    skip_index: usize,
    expr: ExprId,
) -> bool {
    conditions.iter().enumerate().any(|(idx, cond)| {
        idx != skip_index
            && matches!(
                cond,
                ImplicitCondition::NonNegative(other) | ImplicitCondition::Positive(other)
                    if positive_exprs_equivalent_or_same_sqrt_like_base(ctx, *other, expr)
            )
    })
}

fn has_positive_display_condition(
    ctx: &Context,
    conditions: &[ImplicitCondition],
    skip_index: usize,
    expr: ExprId,
) -> bool {
    conditions.iter().enumerate().any(|(idx, cond)| {
        idx != skip_index
            && matches!(
                cond,
                ImplicitCondition::Positive(other)
                    if positive_exprs_equivalent_or_same_sqrt_like_base(ctx, *other, expr)
            )
    })
}

fn is_nonnegative_under_display_conditions(
    ctx: &Context,
    conditions: &[ImplicitCondition],
    skip_index: usize,
    expr: ExprId,
    depth: usize,
) -> bool {
    if depth == 0 {
        return false;
    }

    if has_nonnegative_display_condition(ctx, conditions, skip_index, expr)
        || is_intrinsically_nonnegative_real(ctx, expr)
    {
        return true;
    }

    if let Some(base) = extract_sqrt_like_base(ctx, expr) {
        return is_nonnegative_under_display_conditions(
            ctx,
            conditions,
            skip_index,
            base,
            depth - 1,
        );
    }

    match ctx.get(expr) {
        Expr::Add(left, right) => {
            is_nonnegative_under_display_conditions(ctx, conditions, skip_index, *left, depth - 1)
                && is_nonnegative_under_display_conditions(
                    ctx,
                    conditions,
                    skip_index,
                    *right,
                    depth - 1,
                )
        }
        Expr::Mul(left, right) => {
            is_nonnegative_under_display_conditions(ctx, conditions, skip_index, *left, depth - 1)
                && is_nonnegative_under_display_conditions(
                    ctx,
                    conditions,
                    skip_index,
                    *right,
                    depth - 1,
                )
        }
        Expr::Div(num, den) => {
            is_nonnegative_under_display_conditions(ctx, conditions, skip_index, *num, depth - 1)
                && is_positive_under_display_conditions(
                    ctx,
                    conditions,
                    skip_index,
                    *den,
                    depth - 1,
                )
        }
        _ => false,
    }
}

fn is_positive_under_display_conditions(
    ctx: &Context,
    conditions: &[ImplicitCondition],
    skip_index: usize,
    expr: ExprId,
    depth: usize,
) -> bool {
    if depth == 0 {
        return false;
    }

    if has_positive_display_condition(ctx, conditions, skip_index, expr)
        || is_intrinsically_positive_real(ctx, expr)
        || (has_nonzero_display_condition(ctx, conditions, skip_index, expr)
            && is_nonnegative_under_display_conditions(
                ctx,
                conditions,
                skip_index,
                expr,
                depth - 1,
            ))
    {
        return true;
    }

    if let Some(base) = extract_even_positive_power_base(ctx, expr) {
        if has_nonzero_display_condition(ctx, conditions, skip_index, base) {
            return true;
        }
    }

    if let Some(base) = extract_sqrt_like_base(ctx, expr) {
        if is_positive_under_display_conditions(ctx, conditions, skip_index, base, depth - 1) {
            return true;
        }
    }

    if let Some(arg) = extract_abs_argument_view(ctx, expr) {
        if has_nonzero_display_condition(ctx, conditions, skip_index, arg)
            || is_positive_under_display_conditions(ctx, conditions, skip_index, arg, depth - 1)
        {
            return true;
        }
    }

    match ctx.get(expr) {
        Expr::Add(left, right) => {
            (is_positive_under_display_conditions(ctx, conditions, skip_index, *left, depth - 1)
                && is_nonnegative_under_display_conditions(
                    ctx,
                    conditions,
                    skip_index,
                    *right,
                    depth - 1,
                ))
                || (is_positive_under_display_conditions(
                    ctx,
                    conditions,
                    skip_index,
                    *right,
                    depth - 1,
                ) && is_nonnegative_under_display_conditions(
                    ctx,
                    conditions,
                    skip_index,
                    *left,
                    depth - 1,
                ))
        }
        Expr::Mul(left, right) | Expr::Div(left, right) => {
            is_positive_under_display_conditions(ctx, conditions, skip_index, *left, depth - 1)
                && is_positive_under_display_conditions(
                    ctx,
                    conditions,
                    skip_index,
                    *right,
                    depth - 1,
                )
        }
        _ => false,
    }
}

/// Check if two conditions are equivalent.
pub fn conditions_equivalent(
    ctx: &Context,
    c1: &ImplicitCondition,
    c2: &ImplicitCondition,
) -> bool {
    match (c1, c2) {
        (ImplicitCondition::NonNegative(a), ImplicitCondition::NonNegative(b))
        | (ImplicitCondition::Positive(a), ImplicitCondition::Positive(b)) => {
            positive_ordered_exprs_equivalent(ctx, *a, *b)
        }
        (ImplicitCondition::NonZero(a), ImplicitCondition::NonZero(b)) => {
            exprs_equivalent_up_to_sign(ctx, *a, *b)
        }
        _ => false,
    }
}

fn conditions_same_display(ctx: &Context, c1: &ImplicitCondition, c2: &ImplicitCondition) -> bool {
    c1.display(ctx) == c2.display(ctx)
}

/// Normalize and deduplicate a list of conditions for display.
pub fn normalize_and_dedupe_conditions(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
) -> Vec<ImplicitCondition> {
    let mut result: Vec<ImplicitCondition> = Vec::new();

    for cond in conditions {
        if condition_is_intrinsically_satisfied(ctx, cond) {
            continue;
        }

        if let ImplicitCondition::NonZero(expr) = cond {
            if nonzero_is_dominated_by_positive_condition(ctx, conditions, *expr) {
                continue;
            }
        }

        for normalized in expand_condition_for_display(ctx, cond) {
            let is_duplicate = result.iter().any(|existing| {
                conditions_equivalent(ctx, existing, &normalized)
                    || conditions_same_display(ctx, existing, &normalized)
            });

            if !is_duplicate {
                result.push(normalized);
            }
        }
    }

    combine_nonzero_nonnegative_into_positive(ctx, &mut result);
    rewrite_sign_quotients_with_positive_components(ctx, &mut result);
    combine_nonzero_nonnegative_into_positive(ctx, &mut result);
    combine_factored_nonzero_nonnegative_into_positive(ctx, &mut result);
    apply_dominance_rules(ctx, &mut result);
    result
}

fn condition_is_intrinsically_satisfied(ctx: &Context, cond: &ImplicitCondition) -> bool {
    match cond {
        ImplicitCondition::Positive(expr) => {
            is_intrinsically_positive_real(ctx, *expr)
                || positive_quadratic_power_gap_is_intrinsic(ctx, *expr)
        }
        ImplicitCondition::NonNegative(expr) => {
            is_intrinsically_nonnegative_real(ctx, *expr)
                || is_intrinsically_positive_real(ctx, *expr)
                || positive_quadratic_power_gap_is_intrinsic(ctx, *expr)
        }
        ImplicitCondition::NonZero(expr) => is_intrinsically_positive_real(ctx, *expr),
    }
}

fn positive_quadratic_power_gap_is_intrinsic(ctx: &Context, expr: ExprId) -> bool {
    let vars = cas_ast::collect_variables(ctx, expr);
    if vars.len() != 1 {
        return false;
    }
    let Some(var) = vars.iter().next() else {
        return false;
    };
    let Ok(poly) = Polynomial::from_expr(ctx, expr, var.as_str()) else {
        return false;
    };
    if poly.degree() < 4 || !poly.leading_coeff().is_positive() {
        return false;
    }

    for power in [2_i64, 4, 6, 8] {
        let power_usize = power as usize;
        if poly.degree() % power_usize != 0 {
            continue;
        }
        let base_degree = poly.degree() / power_usize;
        if base_degree != 2 {
            continue;
        }
        let Some(candidate) = quadratic_power_root_candidate(&poly, var, power) else {
            continue;
        };

        let candidate_power = polynomial_positive_power(&candidate, power);
        let gap = candidate_power.sub(&poly);
        if gap.degree() != 0 {
            continue;
        }
        let gap_value = gap
            .coeffs
            .first()
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if !gap_value.is_positive() {
            continue;
        }
        let Some(threshold) = exact_positive_rational_nth_root(&gap_value, power) else {
            continue;
        };

        if quadratic_minimum_strictly_exceeds(&candidate, &threshold) {
            return true;
        }
    }

    false
}

fn quadratic_power_root_candidate(poly: &Polynomial, var: &str, power: i64) -> Option<Polynomial> {
    let coeff = |degree: usize| {
        poly.coeffs
            .get(degree)
            .cloned()
            .unwrap_or_else(BigRational::zero)
    };
    let degree = poly.degree();
    let leading = poly.leading_coeff();
    let a = exact_positive_rational_nth_root(&leading, power)?;
    let power_rat = BigRational::from_integer(power.into());
    let a_to_power_minus_one = rational_positive_power(&a, power - 1);
    let linear_denom = power_rat.clone() * a_to_power_minus_one.clone();
    if linear_denom.is_zero() {
        return None;
    }
    let b = coeff(degree - 1) / linear_denom.clone();
    let pair_count = BigRational::from_integer(((power * (power - 1)) / 2).into());
    let a_to_power_minus_two = rational_positive_power(&a, power - 2);
    let c = (coeff(degree - 2) - pair_count * a_to_power_minus_two * b.clone() * b.clone())
        / linear_denom;

    Some(Polynomial::new(vec![c, b, a], var.to_string()))
}

fn exact_positive_rational_nth_root(value: &BigRational, power: i64) -> Option<BigRational> {
    if power <= 0 || !value.is_positive() {
        return None;
    }
    let power_u32 = u32::try_from(power).ok()?;
    let num_root = value.numer().nth_root(power_u32);
    let den_root = value.denom().nth_root(power_u32);
    if num_root.pow(power_u32) == *value.numer() && den_root.pow(power_u32) == *value.denom() {
        Some(BigRational::new(num_root, den_root))
    } else {
        None
    }
}

fn rational_positive_power(value: &BigRational, power: i64) -> BigRational {
    let mut out = BigRational::one();
    for _ in 0..power {
        out *= value.clone();
    }
    out
}

fn quadratic_minimum_strictly_exceeds(poly: &Polynomial, threshold: &BigRational) -> bool {
    if poly.degree() != 2 {
        return false;
    }
    let a = poly
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if !a.is_positive() {
        return false;
    }
    let b = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let c = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let four = BigRational::from_integer(4.into());
    let minimum = c - (&b * &b) / (four * a);
    minimum > *threshold
}

fn expand_condition_for_display(
    ctx: &mut Context,
    cond: &ImplicitCondition,
) -> Vec<ImplicitCondition> {
    match cond {
        ImplicitCondition::NonZero(expr) => expand_nonzero_condition_for_display(ctx, *expr),
        ImplicitCondition::Positive(expr) => {
            if let Some(arg) = extract_abs_argument_view(ctx, *expr) {
                return expand_nonzero_condition_for_display(ctx, arg);
            }
            if let Some(expanded) = expand_positive_quotient_condition_for_display(ctx, *expr) {
                return expanded;
            }
            if let Some(expanded) = positive_condition_equivalent_nonzero_conditions(ctx, *expr) {
                return expanded;
            }
            let normalized = normalize_condition(ctx, cond);
            if condition_is_intrinsically_satisfied(ctx, &normalized) {
                Vec::new()
            } else {
                vec![normalized]
            }
        }
        _ => {
            let normalized = normalize_condition(ctx, cond);
            if condition_is_intrinsically_satisfied(ctx, &normalized) {
                Vec::new()
            } else {
                vec![normalized]
            }
        }
    }
}

fn expand_positive_quotient_condition_for_display(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<Vec<ImplicitCondition>> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };
    let numerator_nonzero_conditions = positive_condition_equivalent_nonzero_conditions(ctx, num);
    let denominator_nonzero_conditions = positive_condition_equivalent_nonzero_conditions(ctx, den);

    let mut expanded = match numerator_nonzero_conditions {
        Some(conditions) => conditions,
        None if denominator_nonzero_conditions.is_some() => {
            expand_condition_for_display(ctx, &ImplicitCondition::Positive(num))
        }
        None => return None,
    };

    let denominator_conditions = match denominator_nonzero_conditions {
        Some(conditions) => conditions,
        None => vec![normalize_condition(ctx, &ImplicitCondition::Positive(den))],
    };
    for condition in denominator_conditions {
        if !expanded
            .iter()
            .any(|existing| conditions_equivalent(ctx, existing, &condition))
        {
            expanded.push(condition);
        }
    }

    Some(expanded)
}

fn expand_nonzero_condition_for_display(ctx: &mut Context, expr: ExprId) -> Vec<ImplicitCondition> {
    let core_expr = extract_abs_argument_view(ctx, expr).unwrap_or(expr);
    let stripped_expr = strip_nonzero_scalar_factors_for_display(ctx, core_expr);

    if let Some(sinh_expr) = tanh_nonzero_equivalent_sinh(ctx, stripped_expr) {
        return expand_nonzero_condition_for_display(ctx, sinh_expr);
    }

    if let Some(arg_minus_one) = log_nonzero_argument_offset(ctx, stripped_expr) {
        return expand_nonzero_condition_for_display(ctx, arg_minus_one);
    }

    if let Some(expanded) = expand_abs_unit_offset_nonzero_for_display(ctx, stripped_expr) {
        return expanded;
    }

    if let Some(expanded) =
        expand_sqrt_even_power_unit_offset_nonzero_for_display(ctx, stripped_expr)
    {
        return expanded;
    }

    if let Some(base) = extract_sqrt_like_base(ctx, stripped_expr) {
        return expand_condition_for_display(ctx, &ImplicitCondition::Positive(base));
    }

    let normalized_expr = normalize_condition_expr(ctx, stripped_expr);

    if let Some(sinh_expr) = tanh_nonzero_equivalent_sinh(ctx, normalized_expr) {
        return expand_nonzero_condition_for_display(ctx, sinh_expr);
    }

    if let Some(arg_minus_one) = log_nonzero_argument_offset(ctx, normalized_expr) {
        return expand_nonzero_condition_for_display(ctx, arg_minus_one);
    }

    if let Some(expanded) = expand_abs_unit_offset_nonzero_for_display(ctx, normalized_expr) {
        return expanded;
    }

    if let Some(expanded) =
        expand_sqrt_even_power_unit_offset_nonzero_for_display(ctx, normalized_expr)
    {
        return expanded;
    }

    if let Some(base) = extract_sqrt_like_base(ctx, normalized_expr) {
        return expand_condition_for_display(ctx, &ImplicitCondition::Positive(base));
    }

    if let Some(base) = extract_even_positive_power_base(ctx, normalized_expr) {
        return expand_nonzero_condition_for_display(ctx, base);
    }

    if let Some(expanded) = expand_common_factor_sum_nonzero_for_display(ctx, normalized_expr) {
        return expanded;
    }

    let factored = factor(ctx, normalized_expr);
    let mut atomic_factors = Vec::new();
    collect_nonzero_atomic_factors(ctx, factored, &mut atomic_factors);

    if atomic_factors.is_empty() {
        return vec![ImplicitCondition::NonZero(
            normalize_nonzero_condition_expr_for_display(ctx, normalized_expr),
        )];
    }

    if atomic_factors.len() == 1 {
        let atomic = normalize_nonzero_condition_expr_for_display(ctx, atomic_factors[0]);
        return vec![ImplicitCondition::NonZero(atomic)];
    }

    let mut expanded = Vec::new();
    for factor_expr in atomic_factors {
        for cond in expand_nonzero_condition_for_display(ctx, factor_expr) {
            if !expanded
                .iter()
                .any(|existing| conditions_equivalent(ctx, existing, &cond))
            {
                expanded.push(cond);
            }
        }
    }

    if expanded.is_empty() {
        vec![ImplicitCondition::NonZero(
            normalize_nonzero_condition_expr_for_display(ctx, normalized_expr),
        )]
    } else {
        expanded
    }
}

fn expand_abs_unit_offset_nonzero_for_display(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<Vec<ImplicitCondition>> {
    let abs_arg = match ctx.get(expr) {
        Expr::Sub(left, right) if is_one_constant(ctx, *right) => {
            extract_abs_argument_view(ctx, *left)
        }
        Expr::Sub(left, right) if is_one_constant(ctx, *left) => {
            extract_abs_argument_view(ctx, *right)
        }
        _ => None,
    }?;

    if is_intrinsically_nonnegative_real(ctx, abs_arg) {
        let boundary = even_power_unit_offset_nonzero_boundary(ctx, abs_arg).unwrap_or_else(|| {
            let one = ctx.num(1);
            ctx.add(Expr::Sub(abs_arg, one))
        });
        return Some(expand_nonzero_condition_for_display(ctx, boundary));
    }

    expand_signed_unit_boundaries_nonzero_for_display(ctx, abs_arg)
}

fn even_power_unit_offset_nonzero_boundary(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let base = extract_even_positive_power_base(ctx, expr)?;
    let two = ctx.num(2);
    let base_squared = ctx.add(Expr::Pow(base, two));
    let one = ctx.num(1);
    Some(ctx.add(Expr::Sub(base_squared, one)))
}

fn expand_sqrt_even_power_unit_offset_nonzero_for_display(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<Vec<ImplicitCondition>> {
    let sqrt_arg = match ctx.get(expr) {
        Expr::Sub(left, right) if is_one_constant(ctx, *right) => {
            extract_sqrt_like_base(ctx, *left)
        }
        Expr::Sub(left, right) if is_one_constant(ctx, *left) => {
            extract_sqrt_like_base(ctx, *right)
        }
        _ => None,
    }?;
    let base = extract_even_positive_power_base(ctx, sqrt_arg)?;

    expand_signed_unit_boundaries_nonzero_for_display(ctx, base)
}

fn expand_signed_unit_boundaries_nonzero_for_display(
    ctx: &mut Context,
    base: ExprId,
) -> Option<Vec<ImplicitCondition>> {
    let one = ctx.num(1);
    let lower_boundary = ctx.add(Expr::Sub(base, one));
    let upper_boundary = ctx.add(Expr::Add(base, one));
    let mut expanded = Vec::new();

    for boundary in [lower_boundary, upper_boundary] {
        for cond in expand_nonzero_condition_for_display(ctx, boundary) {
            if !expanded
                .iter()
                .any(|existing| conditions_equivalent(ctx, existing, &cond))
            {
                expanded.push(cond);
            }
        }
    }

    (!expanded.is_empty()).then_some(expanded)
}

fn is_one_constant(ctx: &Context, expr: ExprId) -> bool {
    as_rational_const(ctx, expr).is_some_and(|constant| constant.is_one())
}

fn tanh_nonzero_equivalent_sinh(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let arg = match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.builtin_of(*fn_id) == Some(BuiltinFn::Tanh) && args.len() == 1 =>
        {
            args[0]
        }
        _ => return None,
    };

    Some(ctx.call_builtin(BuiltinFn::Sinh, vec![arg]))
}

fn expand_common_factor_sum_nonzero_for_display(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<Vec<ImplicitCondition>> {
    let (left, right, is_subtraction) = additive_common_factor_terms(ctx, expr)?;

    let left_factors: Vec<_> = mul_leaves(ctx, left).into_iter().collect();
    let mut right_factors: Vec<_> = mul_leaves(ctx, right).into_iter().collect();
    let mut common_factors = Vec::new();
    let mut left_remaining = Vec::new();

    for left_factor in left_factors {
        if let Some(right_index) = right_factors
            .iter()
            .position(|right_factor| exprs_equivalent(ctx, left_factor, *right_factor))
        {
            common_factors.push(left_factor);
            right_factors.remove(right_index);
        } else {
            left_remaining.push(left_factor);
        }
    }

    if common_factors.is_empty() {
        return None;
    }

    let common_expr = build_mul_or_one(ctx, &common_factors);
    let left_residual = build_mul_or_one(ctx, &left_remaining);
    let right_residual = build_mul_or_one(ctx, &right_factors);
    let residual_expr = if is_subtraction {
        ctx.add(Expr::Sub(left_residual, right_residual))
    } else {
        ctx.add(Expr::Add(left_residual, right_residual))
    };

    let mut expanded = Vec::new();
    for factor_expr in [common_expr, residual_expr] {
        if as_rational_const(ctx, factor_expr).is_some_and(|constant| !constant.is_zero()) {
            continue;
        }

        for cond in expand_nonzero_condition_for_display(ctx, factor_expr) {
            if !expanded
                .iter()
                .any(|existing| conditions_equivalent(ctx, existing, &cond))
            {
                expanded.push(cond);
            }
        }
    }

    (!expanded.is_empty()).then_some(expanded)
}

fn additive_common_factor_terms(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId, bool)> {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            if let Expr::Neg(inner) = ctx.get(*right) {
                return Some((*left, *inner, true));
            }
            if let Expr::Neg(inner) = ctx.get(*left) {
                return Some((*right, *inner, true));
            }
            Some((*left, *right, false))
        }
        Expr::Sub(left, right) => Some((*left, *right, true)),
        _ => None,
    }
}

fn build_mul_or_one(ctx: &mut Context, factors: &[ExprId]) -> ExprId {
    if factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, factors)
    }
}

fn log_nonzero_argument_offset(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let arg = if let Some((_base_opt, arg)) = extract_log_base_argument_view(ctx, expr) {
        arg
    } else {
        let Expr::Function(fn_id, args) = ctx.get(expr) else {
            return None;
        };
        match (ctx.sym_name(*fn_id), args.as_slice()) {
            ("ln" | "log10", [arg]) => *arg,
            ("log", [arg]) => *arg,
            ("log", [_base, arg]) => *arg,
            _ => return None,
        }
    };
    let one = ctx.num(1);
    Some(ctx.add(Expr::Sub(arg, one)))
}

fn collect_nonzero_atomic_factors(ctx: &Context, expr: ExprId, factors: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Mul(l, r) => {
            collect_nonzero_atomic_factors(ctx, *l, factors);
            collect_nonzero_atomic_factors(ctx, *r, factors);
        }
        Expr::Div(num, den) => {
            collect_nonzero_atomic_factors(ctx, *num, factors);
            collect_nonzero_atomic_factors(ctx, *den, factors);
        }
        Expr::Neg(inner) => collect_nonzero_atomic_factors(ctx, *inner, factors),
        Expr::Pow(base, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() && n.is_positive() {
                    factors.push(*base);
                    return;
                }
            }
            factors.push(expr);
        }
        Expr::Number(_) => {}
        _ => factors.push(expr),
    }
}

fn is_factorial_of_arg(ctx: &Context, expr: ExprId, arg: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(ctx.sym_name(*fn_id), "fact" | "factorial")
                && exprs_equivalent(ctx, args[0], arg)
    )
}

fn as_unit_reciprocal_denominator(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Div(num, den) if as_rational_const(ctx, *num).is_some_and(|n| n.is_one()) => {
            Some(*den)
        }
        _ => None,
    }
}

fn extract_unit_reciprocal_sum_denominators(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };
    let left_den = as_unit_reciprocal_denominator(ctx, *left)?;
    let right_den = as_unit_reciprocal_denominator(ctx, *right)?;
    Some((left_den, right_den))
}

fn is_sum_of_terms(ctx: &Context, expr: ExprId, a: ExprId, b: ExprId) -> bool {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return false;
    };

    (exprs_equivalent_up_to_sign(ctx, *left, a) && exprs_equivalent_up_to_sign(ctx, *right, b))
        || (exprs_equivalent_up_to_sign(ctx, *left, b)
            && exprs_equivalent_up_to_sign(ctx, *right, a))
}

fn reciprocal_sum_nonzero_is_dominated(
    ctx: &Context,
    conditions: &[ImplicitCondition],
    skip_index: usize,
    expr: ExprId,
) -> bool {
    let Some((left_den, right_den)) = extract_unit_reciprocal_sum_denominators(ctx, expr) else {
        return false;
    };

    let mut has_left = false;
    let mut has_right = false;
    let mut has_sum = false;

    for (idx, condition) in conditions.iter().enumerate() {
        if idx == skip_index {
            continue;
        }
        let ImplicitCondition::NonZero(other_expr) = condition else {
            continue;
        };

        if exprs_equivalent_up_to_sign(ctx, *other_expr, left_den) {
            has_left = true;
        } else if exprs_equivalent_up_to_sign(ctx, *other_expr, right_den) {
            has_right = true;
        } else if is_sum_of_terms(ctx, *other_expr, left_den, right_den) {
            has_sum = true;
        }
    }

    has_left && has_right && has_sum
}

fn unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    let expr = extract_abs_argument_view(ctx, expr).unwrap_or(expr);
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.builtin_of(*fn_id) == Some(builtin) && args.len() == 1 =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

fn trig_unit_offset_arg(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    fn trig_arg(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
        unary_builtin_arg(ctx, expr, BuiltinFn::Sin)
            .map(|arg| (BuiltinFn::Sin, arg))
            .or_else(|| {
                unary_builtin_arg(ctx, expr, BuiltinFn::Cos).map(|arg| (BuiltinFn::Cos, arg))
            })
    }

    match ctx.get(expr) {
        Expr::Add(left, right) if is_unit_constant(ctx, *left) => trig_arg(ctx, *right),
        Expr::Add(left, right) if is_unit_constant(ctx, *right) => trig_arg(ctx, *left),
        Expr::Sub(left, right) if is_one_constant(ctx, *left) => trig_arg(ctx, *right),
        Expr::Sub(left, right) if is_one_constant(ctx, *right) => trig_arg(ctx, *left),
        _ => None,
    }
}

fn is_unit_constant(ctx: &Context, expr: ExprId) -> bool {
    as_rational_const(ctx, expr).is_some_and(|constant| constant.abs().is_one())
}

fn trig_perpendicular_nonzero_is_present(
    ctx: &Context,
    conditions: &[ImplicitCondition],
    skip_index: usize,
    offset_builtin: BuiltinFn,
    arg: ExprId,
) -> bool {
    let perpendicular_builtin = match offset_builtin {
        BuiltinFn::Sin => BuiltinFn::Cos,
        BuiltinFn::Cos => BuiltinFn::Sin,
        _ => return false,
    };

    conditions.iter().enumerate().any(|(idx, condition)| {
        if idx == skip_index {
            return false;
        }
        let ImplicitCondition::NonZero(other_expr) = condition else {
            return false;
        };
        let Some(other_arg) = unary_builtin_arg(ctx, *other_expr, perpendicular_builtin) else {
            return false;
        };
        positive_ordered_exprs_equivalent(ctx, arg, other_arg)
    })
}

fn trig_unit_offset_nonzero_is_dominated(
    ctx: &Context,
    conditions: &[ImplicitCondition],
    skip_index: usize,
    expr: ExprId,
) -> bool {
    let Some((offset_builtin, arg)) = trig_unit_offset_arg(ctx, expr) else {
        return false;
    };

    trig_perpendicular_nonzero_is_present(ctx, conditions, skip_index, offset_builtin, arg)
}

fn sqrt_lower_nonzero_boundary(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let (sqrt_expr, shift) = sqrt_positive_lower_shift_parts(ctx, expr)?;
    let sqrt_arg = extract_sqrt_like_base(ctx, sqrt_expr)?;
    let shift_squared = shift.clone() * shift;
    let boundary = ctx.add(Expr::Number(shift_squared));
    Some(ctx.add(Expr::Sub(sqrt_arg, boundary)))
}

fn positive_condition_dominates_sqrt_lower_nonzero(
    ctx: &mut Context,
    positive_expr: ExprId,
    nonzero_expr: ExprId,
) -> bool {
    let Some(boundary) = sqrt_lower_nonzero_boundary(ctx, nonzero_expr) else {
        return false;
    };

    positive_ordered_exprs_equivalent(ctx, positive_expr, boundary)
}

fn nonzero_condition_dominates_sqrt_lower_nonzero(
    ctx: &mut Context,
    nonzero_expr: ExprId,
    shifted_sqrt_expr: ExprId,
) -> bool {
    let Some(boundary) = sqrt_lower_nonzero_boundary(ctx, shifted_sqrt_expr) else {
        return false;
    };
    let normalized_boundary = normalize_nonzero_condition_expr_for_display(ctx, boundary);
    let factored_boundary = factor(ctx, normalized_boundary);

    exprs_equivalent_up_to_sign(ctx, nonzero_expr, normalized_boundary)
        || nonzero_integer_power_base_for_display(ctx, normalized_boundary)
            .is_some_and(|base| exprs_equivalent_up_to_sign(ctx, nonzero_expr, base))
        || exprs_equivalent_up_to_sign(ctx, nonzero_expr, factored_boundary)
        || nonzero_integer_power_base_for_display(ctx, factored_boundary)
            .is_some_and(|base| exprs_equivalent_up_to_sign(ctx, nonzero_expr, base))
}

fn apply_dominance_rules(ctx: &mut Context, conditions: &mut Vec<ImplicitCondition>) {
    let mut to_remove: Vec<usize> = Vec::new();

    for (i, cond) in conditions.iter().enumerate() {
        for (j, other) in conditions.iter().enumerate() {
            if i == j {
                continue;
            }

            match (cond, other) {
                (ImplicitCondition::NonZero(nz_expr), ImplicitCondition::Positive(pos_expr)) => {
                    if exprs_equivalent(ctx, *nz_expr, *pos_expr)
                        || exprs_equivalent_up_to_sign(ctx, *nz_expr, *pos_expr)
                        || is_abs_of(ctx, *nz_expr, *pos_expr)
                        || is_abs_of(ctx, *pos_expr, *nz_expr)
                        || is_positive_power_of_base(ctx, *nz_expr, *pos_expr)
                        || positive_even_power_gap_forces_nonzero(ctx, *pos_expr, *nz_expr)
                        || positive_condition_dominates_affine_nonzero_offset(
                            ctx, *pos_expr, *nz_expr,
                        )
                        || positive_condition_dominates_sqrt_lower_nonzero(ctx, *pos_expr, *nz_expr)
                        || positive_polynomial_condition_contains_nonzero_factor(
                            ctx, *pos_expr, *nz_expr,
                        )
                    {
                        to_remove.push(i);
                        break;
                    }
                }
                (
                    ImplicitCondition::Positive(derived_expr),
                    ImplicitCondition::Positive(pos_expr),
                ) => {
                    if is_abs_of(ctx, *derived_expr, *pos_expr)
                        || is_positive_power_of_base(ctx, *derived_expr, *pos_expr)
                        || positive_condition_dominates_affine_positive_offset(
                            ctx,
                            *pos_expr,
                            *derived_expr,
                        )
                    {
                        to_remove.push(i);
                        break;
                    }
                }
                (
                    ImplicitCondition::NonNegative(nn_expr),
                    ImplicitCondition::Positive(pos_expr),
                ) => {
                    if exprs_equivalent(ctx, *nn_expr, *pos_expr)
                        || is_odd_power_of(ctx, *nn_expr, *pos_expr)
                        || is_positive_multiple_of(ctx, *nn_expr, *pos_expr)
                        || positive_condition_dominates_reciprocal_offset_nonnegative(
                            ctx, *pos_expr, *nn_expr,
                        )
                    {
                        to_remove.push(i);
                        break;
                    }
                }
                (
                    ImplicitCondition::NonNegative(derived_expr),
                    ImplicitCondition::NonNegative(base_expr),
                ) => {
                    if is_odd_power_of(ctx, *derived_expr, *base_expr) {
                        to_remove.push(i);
                        break;
                    }
                }
                (ImplicitCondition::NonZero(nz_expr), ImplicitCondition::NonNegative(nn_expr)) => {
                    if is_factorial_of_arg(ctx, *nz_expr, *nn_expr) {
                        to_remove.push(i);
                        break;
                    }
                }
                (
                    ImplicitCondition::NonZero(nz_expr),
                    ImplicitCondition::NonZero(other_nz_expr),
                ) => {
                    if nonzero_condition_dominates_sqrt_lower_nonzero(ctx, *other_nz_expr, *nz_expr)
                    {
                        to_remove.push(i);
                        break;
                    }
                }
                _ => {}
            }
        }

        if !to_remove.contains(&i) {
            if let ImplicitCondition::Positive(pos_expr) = cond {
                if is_positive_under_display_conditions(
                    ctx,
                    conditions,
                    i,
                    *pos_expr,
                    DISPLAY_SIGN_PROOF_DEPTH,
                ) {
                    to_remove.push(i);
                    continue;
                }
            }

            if let ImplicitCondition::NonNegative(nn_expr) = cond {
                if is_nonnegative_under_display_conditions_or_factored(
                    ctx,
                    conditions,
                    i,
                    *nn_expr,
                    DISPLAY_SIGN_PROOF_DEPTH,
                ) {
                    to_remove.push(i);
                    continue;
                }
            }

            if let ImplicitCondition::NonZero(nz_expr) = cond {
                if is_positive_under_display_conditions(
                    ctx,
                    conditions,
                    i,
                    *nz_expr,
                    DISPLAY_SIGN_PROOF_DEPTH,
                ) {
                    to_remove.push(i);
                    continue;
                }

                if nonzero_is_dominated_by_nonzero_factors(ctx, conditions, i, *nz_expr) {
                    to_remove.push(i);
                    continue;
                }

                if reciprocal_sum_nonzero_is_dominated(ctx, conditions, i, *nz_expr) {
                    to_remove.push(i);
                    continue;
                }

                if trig_unit_offset_nonzero_is_dominated(ctx, conditions, i, *nz_expr) {
                    to_remove.push(i);
                    continue;
                }
            }

            if let ImplicitCondition::Positive(prod_expr) = cond {
                let other_positive_exprs: Vec<ExprId> = conditions
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, condition)| {
                        if idx == i {
                            return None;
                        }
                        match condition {
                            ImplicitCondition::Positive(e) => Some(*e),
                            _ => None,
                        }
                    })
                    .collect();

                if is_product_dominated_by_positives(ctx, *prod_expr, &other_positive_exprs) {
                    to_remove.push(i);
                    continue;
                }

                let factored = factor(ctx, *prod_expr);
                if factored != *prod_expr
                    && is_product_dominated_by_positives(ctx, factored, &other_positive_exprs)
                {
                    to_remove.push(i);
                }
            }
        }
    }

    to_remove.sort();
    to_remove.dedup();
    for i in to_remove.into_iter().rev() {
        conditions.remove(i);
    }
}

/// Render conditions for display, applying normalization and deduplication.
pub fn render_conditions_normalized(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
) -> Vec<String> {
    let normalized = normalize_and_dedupe_conditions(ctx, conditions);
    normalized.iter().map(|c| c.display(ctx)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn unit_reciprocal_square_gap_proves_nonnegative_for_positive_shifted_square_base() {
        let mut ctx = Context::new();
        let expr = parse("1 - 1/(x^2 + 1)^2", &mut ctx).expect("parse expression");

        let base = one_minus_unit_reciprocal_square_base(&ctx, expr).expect("extract base");
        let two = ctx.num(2);
        let base_squared = ctx.add(Expr::Pow(base, two));
        let one = ctx.num(1);
        let gap = ctx.add(Expr::Sub(base_squared, one));
        let normalized_gap = normalize_condition_expr_preserve_sign(&mut ctx, gap);
        assert!(is_positive_under_display_conditions(
            &ctx,
            &[],
            0,
            base,
            DISPLAY_SIGN_PROOF_DEPTH
        ));
        assert!(is_nonnegative_under_display_conditions(
            &ctx,
            &[],
            0,
            normalized_gap,
            DISPLAY_SIGN_PROOF_DEPTH
        ));
        assert!(unit_reciprocal_square_gap_is_nonnegative(
            &mut ctx,
            &[],
            0,
            expr,
            DISPLAY_SIGN_PROOF_DEPTH
        ));
    }

    #[test]
    fn nonnegative_factorial_argument_dominates_factorial_nonzero_display_condition() {
        let mut ctx = Context::new();
        let n = parse("n", &mut ctx).expect("parse n");
        let fact_n = parse("n!", &mut ctx).expect("parse n!");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonNegative(n),
                ImplicitCondition::NonZero(fact_n),
            ],
        );

        assert_eq!(normalized, vec![ImplicitCondition::NonNegative(n)]);
    }

    #[test]
    fn nonzero_constant_multiple_normalizes_to_base_condition() {
        let mut ctx = Context::new();
        let scaled = parse("2*a", &mut ctx).expect("parse scaled");
        let base = parse("a", &mut ctx).expect("parse base");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(scaled),
                ImplicitCondition::NonZero(base),
            ],
        );

        assert_eq!(normalized, vec![ImplicitCondition::NonZero(base)]);
    }

    #[test]
    fn nonzero_fractional_multiple_normalizes_to_base_condition() {
        let mut ctx = Context::new();
        let scaled = parse("a/2", &mut ctx).expect("parse scaled");
        let base = parse("a", &mut ctx).expect("parse base");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(scaled),
                ImplicitCondition::NonZero(base),
            ],
        );

        assert_eq!(normalized, vec![ImplicitCondition::NonZero(base)]);
    }

    #[test]
    fn nonzero_scaled_square_and_expanded_denominator_preserve_base_condition() {
        let mut ctx = Context::new();
        let scaled_square = parse("3*(x^2+x-1)^2", &mut ctx).expect("parse scaled square");
        let expanded_scaled = parse("3*x^2+3*x-3", &mut ctx).expect("parse expanded scaled");
        let base = parse("x^2+x-1", &mut ctx).expect("parse base");

        let inputs = [
            ImplicitCondition::NonZero(scaled_square),
            ImplicitCondition::NonZero(expanded_scaled),
            ImplicitCondition::NonZero(base),
        ];
        let normalized = normalize_and_dedupe_conditions(&mut ctx, &inputs);
        assert_eq!(normalized.len(), 1, "got: {:?}", normalized);
        assert!(conditions_equivalent(
            &ctx,
            &normalized[0],
            &ImplicitCondition::NonZero(base)
        ));
    }

    #[test]
    fn nonzero_expanded_low_degree_polynomial_power_normalizes_to_base() {
        let cases = [
            "x^6 + 3*x^5 - 5*x^3 + 3*x - 1",
            "x^8 + 4*x^7 + 2*x^6 - 8*x^5 - 5*x^4 + 8*x^3 + 2*x^2 - 4*x + 1",
        ];

        for input in cases {
            let mut ctx = Context::new();
            let expanded = parse(input, &mut ctx).expect("parse expanded");
            let base = parse("x^2 + x - 1", &mut ctx).expect("parse base");

            let normalized =
                normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::NonZero(expanded)]);

            assert_eq!(normalized.len(), 1, "input: {input}");
            assert!(
                conditions_equivalent(&ctx, &normalized[0], &ImplicitCondition::NonZero(base)),
                "input: {input}, got: {:?}",
                normalized
            );
        }
    }

    #[test]
    fn positive_abs_dominates_base_nonzero_condition() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("parse x");
        let abs_x = parse("abs(x)", &mut ctx).expect("parse abs(x)");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(x),
                ImplicitCondition::Positive(abs_x),
            ],
        );

        assert_eq!(normalized, vec![ImplicitCondition::NonZero(x)]);
    }

    #[test]
    fn positive_expanded_perfect_square_collapses_to_base_nonzero_condition() {
        let mut ctx = Context::new();

        for (input, expected_base) in [("x^2 + 2*x + 1", "x + 1"), ("4*x^2 + 4*x + 1", "2*x + 1")] {
            let square = parse(input, &mut ctx).expect("parse square");
            let base = parse(expected_base, &mut ctx).expect("parse base");

            let normalized =
                normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::Positive(square)]);

            assert_eq!(
                normalized.len(),
                1,
                "input: {input}, got: {:?}",
                normalized
                    .iter()
                    .map(|cond| cond.display(&ctx))
                    .collect::<Vec<_>>()
            );
            assert!(
                conditions_equivalent(&ctx, &normalized[0], &ImplicitCondition::NonZero(base)),
                "input: {input}, expected NonZero({expected_base}), got: {:?}",
                normalized
                    .iter()
                    .map(|cond| cond.display(&ctx))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn atomic_positive_factors_dominate_composite_log_argument_positive_condition() {
        let mut ctx = Context::new();
        let t = parse("t", &mut ctx).expect("parse t");
        let y = parse("y", &mut ctx).expect("parse y");
        let z = parse("z", &mut ctx).expect("parse z");
        let x = parse("x", &mut ctx).expect("parse x");
        let abs_x = parse("abs(x)", &mut ctx).expect("parse abs(x)");
        let composite = parse("y*x^2/(t*z)", &mut ctx).expect("parse composite");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(t),
                ImplicitCondition::Positive(y),
                ImplicitCondition::Positive(composite),
                ImplicitCondition::Positive(z),
                ImplicitCondition::Positive(abs_x),
            ],
        );

        assert_eq!(
            normalized,
            vec![
                ImplicitCondition::Positive(t),
                ImplicitCondition::Positive(y),
                ImplicitCondition::Positive(z),
                ImplicitCondition::NonZero(x),
            ]
        );
    }

    #[test]
    fn factored_positive_factors_dominate_unfactored_difference_of_squares_condition() {
        let mut ctx = Context::new();
        let x_plus_y = parse("x + y", &mut ctx).expect("parse x + y");
        let x_minus_y = parse("x - y", &mut ctx).expect("parse x - y");
        let composite = parse("x^2 - y^2", &mut ctx).expect("parse composite");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(x_plus_y),
                ImplicitCondition::Positive(x_minus_y),
                ImplicitCondition::Positive(composite),
            ],
        );

        assert_eq!(normalized.len(), 2);
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(x_plus_y))
        }));
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(x_minus_y))
        }));
    }

    #[test]
    fn reciprocal_sum_nonzero_is_dominated_by_atomic_denominators_and_sum() {
        let mut ctx = Context::new();
        let reciprocal_sum = parse("1/a + 1/b", &mut ctx).expect("parse reciprocal sum");
        let sum = parse("a + b", &mut ctx).expect("parse sum");
        let a = parse("a", &mut ctx).expect("parse a");
        let b = parse("b", &mut ctx).expect("parse b");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(reciprocal_sum),
                ImplicitCondition::NonZero(sum),
                ImplicitCondition::NonZero(a),
                ImplicitCondition::NonZero(b),
            ],
        );

        assert_eq!(normalized.len(), 3);
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(sum)) }));
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(a)) }));
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(b)) }));
    }

    #[test]
    fn intrinsically_positive_nonzero_display_condition_is_dropped() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x^2 + 1)^3", &mut ctx).expect("parse expr");

        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::NonZero(expr)]);

        assert!(normalized.is_empty());
    }

    #[test]
    fn real_cosh_nonzero_display_condition_is_dropped() {
        let mut ctx = Context::new();
        let cosh_expr = parse("cosh(sqrt(x))", &mut ctx).expect("parse cosh");
        let sinh_expr = parse("sinh(sqrt(x))", &mut ctx).expect("parse sinh");

        let cosh_normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::NonZero(cosh_expr)]);
        assert!(
            cosh_normalized.is_empty(),
            "real cosh is strictly positive, got: {cosh_normalized:?}"
        );

        let sinh_normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::NonZero(sinh_expr)]);
        assert_eq!(sinh_normalized, vec![ImplicitCondition::NonZero(sinh_expr)]);
    }

    #[test]
    fn nonzero_log_display_condition_normalizes_to_argument_not_one() {
        let mut ctx = Context::new();
        let ln_y = parse("ln(y)", &mut ctx).expect("parse ln(y)");
        let y = parse("y", &mut ctx).expect("parse y");
        let y_minus_one = parse("y - 1", &mut ctx).expect("parse y - 1");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(ln_y),
                ImplicitCondition::Positive(y),
            ],
        );

        assert_eq!(normalized.len(), 2);
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(y_minus_one))
        }));
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(y)) }));
    }

    #[test]
    fn nonzero_tanh_display_condition_normalizes_to_sinh_nonzero() {
        let mut ctx = Context::new();
        let tanh_expr = parse("tanh(2*x + 1)", &mut ctx).expect("parse tanh");
        let sinh_expr = parse("sinh(2*x + 1)", &mut ctx).expect("parse sinh");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(tanh_expr),
                ImplicitCondition::NonZero(sinh_expr),
            ],
        );

        assert_eq!(normalized.len(), 1);
        assert!(conditions_equivalent(
            &ctx,
            &normalized[0],
            &ImplicitCondition::NonZero(sinh_expr)
        ));
    }

    #[test]
    fn nonzero_common_log_square_sum_normalizes_to_base_boundary() {
        let mut ctx = Context::new();
        let composite = parse("ln(x)^2 + x*ln(x)^2", &mut ctx).expect("parse composite condition");
        let x = parse("x", &mut ctx).expect("parse x");
        let x_minus_one = parse("x - 1", &mut ctx).expect("parse x - 1");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(composite),
                ImplicitCondition::Positive(x),
            ],
        );

        assert_eq!(normalized.len(), 2);
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x_minus_one))
        }));
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(x)) }));
    }

    #[test]
    fn nonzero_common_log_square_difference_normalizes_under_positive_base() {
        let mut ctx = Context::new();
        let composite =
            parse("x^3*ln(x)^2 - x*ln(x)^2", &mut ctx).expect("parse composite condition");
        let x = parse("x", &mut ctx).expect("parse x");
        let x_minus_one = parse("x - 1", &mut ctx).expect("parse x - 1");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(x_minus_one),
                ImplicitCondition::Positive(x),
                ImplicitCondition::NonZero(composite),
            ],
        );

        assert_eq!(normalized.len(), 2);
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x_minus_one))
        }));
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(x)) }));
    }

    #[test]
    fn nonzero_log_of_intrinsically_positive_quadratic_normalizes_to_base_nonzero() {
        let mut ctx = Context::new();
        let ln_quad = parse("ln(y^2 + 1)", &mut ctx).expect("parse ln(y^2 + 1)");
        let y = parse("y", &mut ctx).expect("parse y");

        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::NonZero(ln_quad)]);

        assert_eq!(normalized, vec![ImplicitCondition::NonZero(y)]);
    }

    #[test]
    fn positive_sum_under_other_display_conditions_drops_composite_nonzero_condition() {
        let mut ctx = Context::new();
        let reciprocal_sum = parse("1/sqrt(u) + 1", &mut ctx).expect("parse reciprocal sum");
        let transformed_den =
            parse("u^(1/2) + u", &mut ctx).expect("parse transformed denominator");
        let sqrt_u = parse("sqrt(u)", &mut ctx).expect("parse sqrt(u)");
        let u = parse("u", &mut ctx).expect("parse u");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(reciprocal_sum),
                ImplicitCondition::NonZero(sqrt_u),
                ImplicitCondition::NonNegative(u),
                ImplicitCondition::NonZero(transformed_den),
            ],
        );

        assert_eq!(normalized, vec![ImplicitCondition::Positive(u)]);
    }

    #[test]
    fn intrinsically_nonnegative_display_condition_is_dropped() {
        let mut ctx = Context::new();
        let expr = parse("2*sqrt(x^2 + 1) + x^2 + 2", &mut ctx).expect("parse expr");

        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::NonNegative(expr)]);

        assert!(normalized.is_empty());
    }

    #[test]
    fn intrinsically_positive_square_plus_constant_condition_is_dropped() {
        let mut ctx = Context::new();
        let expr = parse("x^4 + 2*x^3 + 3*x^2 + 2*x + 8", &mut ctx).expect("parse expr");

        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::Positive(expr)]);

        assert!(normalized.is_empty());
    }

    #[test]
    fn factored_perfect_square_nonnegative_display_condition_is_dropped() {
        let mut ctx = Context::new();
        let expr = parse("a^2 + 2*a*b + b^2", &mut ctx).expect("parse expr");

        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::NonNegative(expr)]);

        assert!(normalized.is_empty());
    }

    #[test]
    fn nonzero_sqrt_and_nonnegative_base_combine_into_positive_base() {
        let mut ctx = Context::new();
        let sqrt_u = parse("sqrt(u)", &mut ctx).expect("parse sqrt(u)");
        let u = parse("u", &mut ctx).expect("parse u");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(sqrt_u),
                ImplicitCondition::NonNegative(u),
            ],
        );

        assert_eq!(normalized, vec![ImplicitCondition::Positive(u)]);
    }

    #[test]
    fn nonzero_scaled_sqrt_and_nonnegative_base_combine_into_positive_base() {
        let mut ctx = Context::new();
        let sqrt_scaled = parse("sqrt(3*x)", &mut ctx).expect("parse sqrt(3*x)");
        let x = parse("x", &mut ctx).expect("parse x");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(sqrt_scaled),
                ImplicitCondition::NonNegative(x),
            ],
        );

        assert_eq!(normalized, vec![ImplicitCondition::Positive(x)]);
    }

    #[test]
    fn positive_sqrt_unit_lower_boundary_dominates_nonzero_shifted_sqrt() {
        let mut ctx = Context::new();
        let nonzero = parse("sqrt(x)-1", &mut ctx).expect("parse nonzero");
        let positive = parse("x-1", &mut ctx).expect("parse positive");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(nonzero),
                ImplicitCondition::Positive(positive),
            ],
        );

        assert_eq!(normalized.len(), 1);
        assert!(conditions_equivalent(
            &ctx,
            &normalized[0],
            &ImplicitCondition::Positive(positive)
        ));
    }

    #[test]
    fn positive_scaled_sqrt_unit_lower_boundary_dominates_nonzero_shifted_sqrt() {
        let mut ctx = Context::new();
        let nonzero = parse("sqrt(2*x)-1", &mut ctx).expect("parse nonzero");
        let positive = parse("2*x-1", &mut ctx).expect("parse positive");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(nonzero),
                ImplicitCondition::Positive(positive),
            ],
        );

        assert_eq!(normalized.len(), 1);
        assert!(conditions_equivalent(
            &ctx,
            &normalized[0],
            &ImplicitCondition::Positive(positive)
        ));
    }

    #[test]
    fn positive_sqrt_lower_boundary_compacts_nonunit_shift() {
        let mut ctx = Context::new();
        let shifted = parse("sqrt(x)-2", &mut ctx).expect("parse shifted");
        let boundary = parse("x-4", &mut ctx).expect("parse boundary");

        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::Positive(shifted)]);

        assert_eq!(normalized.len(), 1);
        assert!(conditions_equivalent(
            &ctx,
            &normalized[0],
            &ImplicitCondition::Positive(boundary)
        ));
    }

    #[test]
    fn positive_scaled_sqrt_lower_boundary_dominates_nonzero_nonunit_shift() {
        let mut ctx = Context::new();
        let nonzero = parse("sqrt(2*x)-2", &mut ctx).expect("parse nonzero");
        let positive = parse("2*x-4", &mut ctx).expect("parse positive");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(nonzero),
                ImplicitCondition::Positive(positive),
            ],
        );

        assert_eq!(normalized.len(), 1);
        assert!(conditions_equivalent(
            &ctx,
            &normalized[0],
            &ImplicitCondition::Positive(positive)
        ));
    }

    #[test]
    fn nonzero_boundary_dominates_degenerate_sqrt_lower_nonzero() {
        let mut ctx = Context::new();
        let shifted_sqrt = parse("sqrt(x^2+4)-2", &mut ctx).expect("parse shifted sqrt");
        let boundary = parse("x", &mut ctx).expect("parse boundary");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(boundary),
                ImplicitCondition::NonZero(shifted_sqrt),
            ],
        );

        assert_eq!(normalized.len(), 1);
        assert!(conditions_equivalent(
            &ctx,
            &normalized[0],
            &ImplicitCondition::NonZero(boundary)
        ));
    }

    #[test]
    fn nonzero_affine_boundary_dominates_scaled_degenerate_sqrt_lower_nonzero() {
        let mut ctx = Context::new();
        let shifted_sqrt = parse("sqrt((2*x+1)^2+4)-2", &mut ctx).expect("parse shifted sqrt");
        let boundary = parse("2*x+1", &mut ctx).expect("parse boundary");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(boundary),
                ImplicitCondition::NonZero(shifted_sqrt),
            ],
        );

        assert_eq!(normalized.len(), 1);
        assert!(conditions_equivalent(
            &ctx,
            &normalized[0],
            &ImplicitCondition::NonZero(boundary)
        ));
    }

    #[test]
    fn positive_shifted_affine_dominates_later_nonzero_shift() {
        let mut ctx = Context::new();
        let positive = parse("x + 1", &mut ctx).expect("parse positive");
        let nonzero = parse("x + 2", &mut ctx).expect("parse nonzero");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(nonzero),
                ImplicitCondition::Positive(positive),
            ],
        );

        assert_eq!(normalized.len(), 1);
        assert!(conditions_equivalent(
            &ctx,
            &normalized[0],
            &ImplicitCondition::Positive(positive)
        ));
    }

    #[test]
    fn positive_shifted_affine_keeps_earlier_nonzero_shift() {
        let mut ctx = Context::new();
        let positive = parse("x + 1", &mut ctx).expect("parse positive");
        let nonzero = parse("x", &mut ctx).expect("parse nonzero");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(nonzero),
                ImplicitCondition::Positive(positive),
            ],
        );

        assert!(normalized.iter().any(|condition| conditions_equivalent(
            &ctx,
            condition,
            &ImplicitCondition::NonZero(nonzero)
        )));
        assert!(normalized.iter().any(|condition| conditions_equivalent(
            &ctx,
            condition,
            &ImplicitCondition::Positive(positive)
        )));
    }

    #[test]
    fn positive_shifted_affine_dominates_opposite_oriented_nonzero_shift() {
        let mut ctx = Context::new();
        let positive = parse("3 - 2*x", &mut ctx).expect("parse positive");
        let nonzero = parse("x - 2", &mut ctx).expect("parse nonzero");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(nonzero),
                ImplicitCondition::Positive(positive),
            ],
        );

        assert_eq!(normalized.len(), 1);
        assert!(conditions_equivalent(
            &ctx,
            &normalized[0],
            &ImplicitCondition::Positive(positive)
        ));
    }

    #[test]
    fn positive_shifted_affine_keeps_crossing_opposite_oriented_nonzero_shift() {
        let mut ctx = Context::new();
        let positive = parse("3 - 2*x", &mut ctx).expect("parse positive");
        let nonzero = parse("x - 1", &mut ctx).expect("parse nonzero");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(nonzero),
                ImplicitCondition::Positive(positive),
            ],
        );

        assert!(normalized.iter().any(|condition| conditions_equivalent(
            &ctx,
            condition,
            &ImplicitCondition::NonZero(nonzero)
        )));
        assert!(normalized.iter().any(|condition| conditions_equivalent(
            &ctx,
            condition,
            &ImplicitCondition::Positive(positive)
        )));
    }

    #[test]
    fn factored_boundary_nonzeros_and_nonnegative_base_combine_into_positive_base() {
        let mut ctx = Context::new();
        let base = parse("1 - x^2", &mut ctx).expect("parse base");
        let left_boundary = parse("x - 1", &mut ctx).expect("parse left boundary");
        let right_boundary = parse("x + 1", &mut ctx).expect("parse right boundary");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonNegative(base),
                ImplicitCondition::NonZero(left_boundary),
                ImplicitCondition::NonZero(right_boundary),
            ],
        );

        assert_eq!(normalized.len(), 1);
        assert!(conditions_equivalent(
            &ctx,
            &normalized[0],
            &ImplicitCondition::Positive(base)
        ));
    }

    #[test]
    fn nonzero_positive_cofactor_and_nonnegative_factor_combine_into_positive_factor() {
        let mut ctx = Context::new();
        let nonzero_product =
            parse("(x^2 + 1/2) * (x^4 + x^2 - 3/4)", &mut ctx).expect("parse product");
        let scaled_gap =
            parse("4*x^4 + 4*x^2 - 3", &mut ctx).expect("parse scaled nonnegative gap");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(nonzero_product),
                ImplicitCondition::NonNegative(scaled_gap),
            ],
        );

        assert_eq!(normalized.len(), 1, "got: {normalized:?}");
        assert!(conditions_equivalent(
            &ctx,
            &normalized[0],
            &ImplicitCondition::Positive(scaled_gap)
        ));
    }

    #[test]
    fn nonzero_boundary_base_is_dominated_by_positive_base_before_factor_display() {
        let mut ctx = Context::new();
        let base = parse("1 - x^2", &mut ctx).expect("parse base");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(base),
                ImplicitCondition::Positive(base),
            ],
        );

        assert_eq!(normalized.len(), 1);
        assert!(conditions_equivalent(
            &ctx,
            &normalized[0],
            &ImplicitCondition::Positive(base)
        ));
    }

    #[test]
    fn positive_base_dominates_positive_scaled_nonnegative_polynomial() {
        let mut ctx = Context::new();
        let scaled = parse("-4*x^2 - 4*x", &mut ctx).expect("parse scaled");
        let base = parse("-x^2 - x", &mut ctx).expect("parse base");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonNegative(scaled),
                ImplicitCondition::Positive(base),
            ],
        );

        assert_eq!(normalized.len(), 1);
        assert!(conditions_equivalent(
            &ctx,
            &normalized[0],
            &ImplicitCondition::Positive(base)
        ));
    }

    #[test]
    fn positive_shifted_high_power_gap_preserves_compact_display() {
        let mut ctx = Context::new();
        let gap = parse("3 - ((x + 1)^2)^2", &mut ctx).expect("parse shifted power gap");

        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::Positive(gap)]);
        let rendered: Vec<String> = normalized
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect();

        assert_eq!(rendered, vec!["3 - (x + 1)^4 > 0".to_string()]);
    }

    #[test]
    fn positive_expanded_shifted_fourth_gap_preserves_compact_display() {
        let mut ctx = Context::new();
        let gap = parse("2 - x^4 - 4*x^3 - 6*x^2 - 4*x", &mut ctx).expect("parse expanded gap");

        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::Positive(gap)]);
        let rendered: Vec<String> = normalized
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect();

        assert_eq!(rendered, vec!["3 - (x + 1)^4 > 0".to_string()]);
    }

    #[test]
    fn positive_expanded_negative_monic_quadratic_square_gap_preserves_compact_display() {
        let mut ctx = Context::new();
        let gap = parse("6 - x^4 - 2*x^3 - 3*x^2 - 2*x", &mut ctx)
            .expect("parse expanded quadratic-square gap");

        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::Positive(gap)]);
        let rendered: Vec<String> = normalized
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect();

        assert_eq!(rendered, vec!["7 - (x^2 + x + 1)^2 > 0".to_string()]);
    }

    #[test]
    fn positive_shifted_fourth_gap_dominates_expanded_nonnegative_gap() {
        let mut ctx = Context::new();
        let positive_gap = parse("3 - (x + 1)^4", &mut ctx).expect("parse positive gap");
        let nonnegative_gap =
            parse("2 - x^4 - 4*x^3 - 6*x^2 - 4*x", &mut ctx).expect("parse nonnegative gap");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(positive_gap),
                ImplicitCondition::NonNegative(nonnegative_gap),
            ],
        );
        let rendered: Vec<String> = normalized
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect();

        assert_eq!(rendered, vec!["3 - (x + 1)^4 > 0".to_string()]);
    }

    #[test]
    fn nonzero_shifted_high_power_gap_preserves_compact_display_up_to_sign() {
        let cases = ["3 - (x + 1)^4", "x^4 + 4*x^3 + 6*x^2 + 4*x - 2"];

        for input in cases {
            let mut ctx = Context::new();
            let gap = parse(input, &mut ctx).expect("parse gap");

            let normalized =
                normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::NonZero(gap)]);
            let rendered: Vec<String> = normalized
                .iter()
                .map(|condition| condition.display(&ctx))
                .collect();

            assert!(
                rendered == vec!["3 - (x + 1)^4 ≠ 0".to_string()]
                    || rendered == vec!["(x + 1)^4 - 3 ≠ 0".to_string()],
                "input: {input}, rendered: {rendered:?}"
            );
        }
    }

    #[test]
    fn positive_affine_square_gap_keeps_expanded_display() {
        let mut ctx = Context::new();
        let gap = parse("1 - (2*x + 1)^2", &mut ctx).expect("parse affine square gap");

        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::Positive(gap)]);
        let rendered: Vec<String> = normalized
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect();

        assert_eq!(rendered, vec!["-x^2 - x > 0".to_string()]);
    }

    #[test]
    fn positive_factorable_gap_dominates_boundary_nonzeros() {
        let mut ctx = Context::new();
        let positive_base = parse("1 - x^2", &mut ctx).expect("parse positive base");
        let left_boundary = parse("x - 1", &mut ctx).expect("parse left boundary");
        let right_boundary = parse("x + 1", &mut ctx).expect("parse right boundary");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(left_boundary),
                ImplicitCondition::NonZero(right_boundary),
                ImplicitCondition::Positive(positive_base),
            ],
        );

        assert_eq!(normalized.len(), 1);
        assert!(conditions_equivalent(
            &ctx,
            &normalized[0],
            &ImplicitCondition::Positive(positive_base)
        ));
    }

    #[test]
    fn positive_factored_polynomial_dominates_scaled_boundary_nonzeros() {
        let mut ctx = Context::new();
        let positive_base = parse("-4*x^2 - 4*x", &mut ctx).expect("parse positive base");
        let x = parse("x", &mut ctx).expect("parse x");
        let scaled_boundary = parse("2*x + 2", &mut ctx).expect("parse scaled boundary");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(x),
                ImplicitCondition::NonZero(scaled_boundary),
                ImplicitCondition::Positive(positive_base),
            ],
        );

        assert_eq!(normalized.len(), 1);
        assert!(conditions_equivalent(
            &ctx,
            &normalized[0],
            &ImplicitCondition::Positive(positive_base)
        ));
    }

    #[test]
    fn nonzero_abs_sqrt_and_nonnegative_base_combine_into_positive_base() {
        let mut ctx = Context::new();
        let abs_sqrt_u = parse("abs(sqrt(u))", &mut ctx).expect("parse abs(sqrt(u))");
        let u = parse("u", &mut ctx).expect("parse u");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(abs_sqrt_u),
                ImplicitCondition::NonNegative(u),
            ],
        );

        assert_eq!(normalized, vec![ImplicitCondition::Positive(u)]);
    }

    #[test]
    fn positive_abs_sqrt_is_dominated_by_positive_base() {
        let mut ctx = Context::new();
        let abs_sqrt_u = parse("abs(sqrt(u))", &mut ctx).expect("parse abs(sqrt(u))");
        let u = parse("u", &mut ctx).expect("parse u");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(abs_sqrt_u),
                ImplicitCondition::Positive(u),
            ],
        );

        assert_eq!(normalized, vec![ImplicitCondition::Positive(u)]);
    }

    #[test]
    fn nonzero_abs_product_expands_to_atomic_factors() {
        let mut ctx = Context::new();
        let abs_product = parse("abs((x-1)*(x+1))", &mut ctx).expect("parse abs product");
        let x_minus_1 = parse("x - 1", &mut ctx).expect("parse x - 1");
        let x_plus_1 = parse("x + 1", &mut ctx).expect("parse x + 1");

        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::NonZero(abs_product)]);

        assert_eq!(normalized.len(), 2);
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x_minus_1))
        }));
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x_plus_1))
        }));
    }

    #[test]
    fn positive_abs_product_expands_to_atomic_factors() {
        let mut ctx = Context::new();
        let abs_product = parse("abs(x*y)", &mut ctx).expect("parse abs product");
        let x = parse("x", &mut ctx).expect("parse x");
        let y = parse("y", &mut ctx).expect("parse y");

        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::Positive(abs_product)]);

        assert_eq!(normalized.len(), 2);
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x)) }));
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(y)) }));
    }

    #[test]
    fn positive_sqrt_even_power_expands_to_atomic_nonzero_base() {
        let mut ctx = Context::new();
        let x_minus_one = parse("x - 1", &mut ctx).expect("parse x - 1");
        let x_plus_one = parse("x + 1", &mut ctx).expect("parse x + 1");

        for input in ["sqrt((x^2 - 1)^2)", "((x^2 - 1)^2)^(1/2)"] {
            let expr = parse(input, &mut ctx).expect("parse sqrt even power");
            let normalized =
                normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::Positive(expr)]);

            assert_eq!(normalized.len(), 2, "input: {input}, got: {normalized:?}");
            assert!(normalized.iter().any(|cond| {
                conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x_minus_one))
            }));
            assert!(normalized.iter().any(|cond| {
                conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x_plus_one))
            }));
        }
    }

    #[test]
    fn nonzero_abs_unit_offset_expands_to_signed_boundaries() {
        let mut ctx = Context::new();
        let x_minus_one = parse("x - 1", &mut ctx).expect("parse x - 1");
        let x_plus_one = parse("x + 1", &mut ctx).expect("parse x + 1");

        for input in ["abs(x) - 1", "1 - abs(x)"] {
            let expr = parse(input, &mut ctx).expect("parse abs unit offset");
            let normalized =
                normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::NonZero(expr)]);

            assert_eq!(normalized.len(), 2, "input: {input}, got: {normalized:?}");
            assert!(normalized.iter().any(|cond| {
                conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x_minus_one))
            }));
            assert!(normalized.iter().any(|cond| {
                conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x_plus_one))
            }));
        }
    }

    #[test]
    fn nonzero_abs_nonnegative_unit_offset_drops_impossible_positive_boundary() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("parse x");
        let x_square_minus_two = parse("x^2 - 2", &mut ctx).expect("parse x^2 - 2");
        let impossible_boundary =
            parse("x^4 + 2 - 2*x^2", &mut ctx).expect("parse impossible boundary");

        for input in ["abs((x^2 - 1)^2) - 1", "1 - abs((x^2 - 1)^2)"] {
            let expr = parse(input, &mut ctx).expect("parse abs nonnegative unit offset");
            let normalized =
                normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::NonZero(expr)]);

            assert_eq!(normalized.len(), 2, "input: {input}, got: {normalized:?}");
            assert!(normalized
                .iter()
                .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x)) }));
            assert!(normalized.iter().any(|cond| {
                conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x_square_minus_two))
            }));
            assert!(
                !normalized.iter().any(|cond| {
                    conditions_equivalent(
                        &ctx,
                        cond,
                        &ImplicitCondition::NonZero(impossible_boundary),
                    )
                }),
                "input: {input}, impossible nonzero boundary should be dropped: {normalized:?}"
            );
        }
    }

    #[test]
    fn nonzero_abs_nonnegative_higher_even_power_unit_offset_drops_positive_factor_boundary() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("parse x");
        let x_square_minus_two = parse("x^2 - 2", &mut ctx).expect("parse x^2 - 2");
        let mixed_positive_factor_boundary =
            parse("x^6 + 6*x^2 - 4*x^4 - 4", &mut ctx).expect("parse mixed boundary");

        for input in ["abs((x^2 - 1)^4) - 1", "1 - abs((x^2 - 1)^4)"] {
            let expr = parse(input, &mut ctx).expect("parse higher abs nonnegative unit offset");
            let normalized =
                normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::NonZero(expr)]);

            assert_eq!(normalized.len(), 2, "input: {input}, got: {normalized:?}");
            assert!(normalized
                .iter()
                .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x)) }));
            assert!(normalized.iter().any(|cond| {
                conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x_square_minus_two))
            }));
            assert!(
                !normalized.iter().any(|cond| {
                    conditions_equivalent(
                        &ctx,
                        cond,
                        &ImplicitCondition::NonZero(mixed_positive_factor_boundary),
                    )
                }),
                "input: {input}, positive factor boundary should not leak: {normalized:?}"
            );
        }
    }

    #[test]
    fn nonzero_sqrt_even_power_unit_offset_expands_to_signed_boundaries() {
        let mut ctx = Context::new();
        let x_minus_one = parse("x - 1", &mut ctx).expect("parse x - 1");
        let x_plus_one = parse("x + 1", &mut ctx).expect("parse x + 1");

        for input in [
            "sqrt(x^2) - 1",
            "1 - sqrt(x^2)",
            "(x^2)^(1/2) - 1",
            "1 - (x^2)^(1/2)",
        ] {
            let expr = parse(input, &mut ctx).expect("parse sqrt unit offset");
            let normalized =
                normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::NonZero(expr)]);

            assert_eq!(normalized.len(), 2, "input: {input}, got: {normalized:?}");
            assert!(normalized.iter().any(|cond| {
                conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x_minus_one))
            }));
            assert!(normalized.iter().any(|cond| {
                conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x_plus_one))
            }));
        }
    }

    #[test]
    fn nonzero_abs_quotient_expands_to_atomic_numerator_and_denominator() {
        let mut ctx = Context::new();
        let abs_quotient = parse("abs(x/(x+1))", &mut ctx).expect("parse abs quotient");
        let x = parse("x", &mut ctx).expect("parse x");
        let x_plus_1 = parse("x + 1", &mut ctx).expect("parse x + 1");

        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::NonZero(abs_quotient)]);

        assert_eq!(normalized.len(), 2);
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x)) }));
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x_plus_1))
        }));
    }

    #[test]
    fn positive_abs_quotient_expands_to_atomic_numerator_and_denominator() {
        let mut ctx = Context::new();
        let abs_quotient = parse("abs((x-1)/(x+1))", &mut ctx).expect("parse abs quotient");
        let x_minus_1 = parse("x - 1", &mut ctx).expect("parse x - 1");
        let x_plus_1 = parse("x + 1", &mut ctx).expect("parse x + 1");

        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::Positive(abs_quotient)]);

        assert_eq!(normalized.len(), 2);
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x_minus_1))
        }));
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x_plus_1))
        }));
    }

    #[test]
    fn trig_unit_offset_nonzero_is_dominated_by_perpendicular_nonzero_condition() {
        let mut ctx = Context::new();
        let sec_log_arg =
            parse("abs((sin(2*x+1)+1)/cos(2*x+1))", &mut ctx).expect("parse sec log arg");
        let cos_arg = parse("cos(2*x+1)", &mut ctx).expect("parse cos arg");
        let csc_log_arg =
            parse("abs((cos(2*x+1)-1)/sin(2*x+1))", &mut ctx).expect("parse csc log arg");
        let sin_arg = parse("sin(2*x+1)", &mut ctx).expect("parse sin arg");

        let sec_normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(sec_log_arg),
                ImplicitCondition::NonZero(cos_arg),
            ],
        );
        assert_eq!(sec_normalized, vec![ImplicitCondition::NonZero(cos_arg)]);

        let csc_normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(csc_log_arg),
                ImplicitCondition::NonZero(sin_arg),
            ],
        );
        assert_eq!(csc_normalized, vec![ImplicitCondition::NonZero(sin_arg)]);

        let cos_offset = parse("cos(1+2*x)-1", &mut ctx).expect("parse reordered cos offset");
        let abs_sin_arg = parse("abs(sin(2*x+1))", &mut ctx).expect("parse abs sin");
        let direct_csc_normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(cos_offset),
                ImplicitCondition::NonZero(abs_sin_arg),
            ],
        );
        assert_eq!(
            direct_csc_normalized,
            vec![ImplicitCondition::NonZero(sin_arg)]
        );
    }

    #[test]
    fn trig_unit_offset_nonzero_stays_when_perpendicular_condition_is_absent() {
        let mut ctx = Context::new();
        let sin_unit_offset = parse("sin(2*x+1)+1", &mut ctx).expect("parse sin offset");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[ImplicitCondition::NonZero(sin_unit_offset)],
        );

        assert_eq!(
            normalized,
            vec![ImplicitCondition::NonZero(sin_unit_offset)]
        );
    }

    #[test]
    fn nonnegative_odd_power_is_dominated_by_nonnegative_base() {
        let mut ctx = Context::new();
        let x_cubed = parse("x^3", &mut ctx).expect("parse x^3");
        let x = parse("x", &mut ctx).expect("parse x");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonNegative(x_cubed),
                ImplicitCondition::NonNegative(x),
            ],
        );

        assert_eq!(normalized, vec![ImplicitCondition::NonNegative(x)]);
    }

    #[test]
    fn positive_odd_power_normalizes_to_positive_base() {
        let mut ctx = Context::new();
        let x_cubed = parse("x^3", &mut ctx).expect("parse x^3");
        let x = parse("x", &mut ctx).expect("parse x");
        let x_plus_one = parse("x + 1", &mut ctx).expect("parse x + 1");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(x_cubed),
                ImplicitCondition::NonZero(x),
                ImplicitCondition::NonZero(x_plus_one),
            ],
        );

        assert_eq!(normalized, vec![ImplicitCondition::Positive(x)]);
    }

    #[test]
    fn positive_reciprocal_normalizes_to_positive_denominator_and_dominates_product() {
        let mut ctx = Context::new();
        let reciprocal_y = parse("1/y", &mut ctx).expect("parse 1/y");
        let product = parse("x*y", &mut ctx).expect("parse product");
        let x = parse("x", &mut ctx).expect("parse x");
        let y = parse("y", &mut ctx).expect("parse y");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(reciprocal_y),
                ImplicitCondition::Positive(product),
                ImplicitCondition::Positive(x),
            ],
        );

        assert_eq!(normalized.len(), 2);
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(x)) }));
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(y)) }));
    }
}
