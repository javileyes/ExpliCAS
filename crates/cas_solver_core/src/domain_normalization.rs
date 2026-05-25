//! Condition normalization, deduplication, and dominance rules.

use crate::domain_condition::ImplicitCondition;
use cas_ast::{ordering::compare_expr, BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_domain::{
    exprs_equivalent, exprs_equivalent_up_to_sign, is_abs_of, is_odd_power_of,
    is_positive_multiple_of, is_positive_power_of_base, is_product_dominated_by_positives,
    positive_expr_implies_lower_bound,
};
use cas_math::expr_extract::{
    extract_abs_argument_view, extract_log_base_argument_view, extract_sqrt_argument_view,
};
use cas_math::expr_nary::{
    add_terms_signed, build_balanced_add, build_balanced_mul, mul_leaves, AddView, Sign,
};
use cas_math::expr_normalization::{
    extract_even_positive_power_base, normalize_condition_expr as normalize_condition_expr_math,
    normalize_condition_expr_preserve_sign,
};
use cas_math::factor::factor;
use cas_math::multipoly::MultiPoly;
use cas_math::numeric_eval::as_rational_const;
use cas_math::perfect_square_support::rational_sqrt;
use cas_math::polynomial::Polynomial;
use cas_math::prove_sign::{prove_nonnegative_depth_with, prove_positive_depth_with};
use cas_math::root_forms::{try_rewrite_simplify_square_root_expr, SimplifySquareRootRewriteKind};
use cas_math::tri_proof::TriProof;
use cas_math::trig_eval_table_support::lookup_trig_or_inverse;
use cas_math::trig_values::TrigValue;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::collections::BTreeMap;

const DISPLAY_SIGN_PROOF_DEPTH: usize = 12;
const CONDITION_DISPLAY_SIGNATURE_MAX_TERMS: usize = 12;
const CONDITION_DISPLAY_SIGNATURE_MAX_FACTORS: usize = 8;

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
        if let Some(arg) = extract_abs_argument_view(ctx, *e) {
            return normalize_condition(ctx, &ImplicitCondition::NonZero(arg));
        }

        if let Some(equivalent) = positive_trig_condition_equivalent_expr(ctx, *e) {
            return normalize_condition(ctx, &ImplicitCondition::Positive(equivalent));
        }

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
        ImplicitCondition::LowerBound(e, _) => *e,
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
        ImplicitCondition::LowerBound(_, lower) => {
            ImplicitCondition::LowerBound(normalized_expr, lower.clone())
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

fn positive_trig_condition_equivalent_expr(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let (builtin, arg) = {
        let Expr::Function(fn_id, args) = ctx.get(expr) else {
            return None;
        };
        if args.len() != 1 {
            return None;
        }
        (ctx.builtin_of(*fn_id)?, args[0])
    };

    match builtin {
        BuiltinFn::Sec => Some(ctx.call_builtin(BuiltinFn::Cos, vec![arg])),
        BuiltinFn::Csc => Some(ctx.call_builtin(BuiltinFn::Sin, vec![arg])),
        BuiltinFn::Tan => {
            let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
            let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
            Some(ctx.add(Expr::Div(sin_arg, cos_arg)))
        }
        BuiltinFn::Cot => {
            let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
            let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
            Some(ctx.add(Expr::Div(cos_arg, sin_arg)))
        }
        _ => None,
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
    // Positive(x^(2k)) -> NonZero(x), including k < 0, because real even
    // integer powers are positive exactly away from the base's zero set.
    if let Some(base) = extract_even_nonzero_integer_power_base(ctx, expr) {
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
    if let Some(base) = extract_even_nonzero_integer_power_base(ctx, expr) {
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

fn extract_even_nonzero_integer_power_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let Expr::Number(exp_num) = ctx.get(*exp) else {
        return None;
    };
    if !exp_num.is_integer() || exp_num.is_zero() {
        return None;
    }

    let exp_int = exp_num.to_integer();
    let two: num_bigint::BigInt = 2.into();
    (&exp_int % &two == num_bigint::BigInt::zero()).then_some(*base)
}

fn positive_sqrt_even_power_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let sqrt_arg = extract_sqrt_like_base(ctx, expr)?;
    extract_even_nonzero_integer_power_base(ctx, sqrt_arg)
}

fn nonnegative_negative_even_power_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let Expr::Number(exp_num) = ctx.get(*exp) else {
        return None;
    };
    if !exp_num.is_integer() || exp_num.is_zero() {
        return None;
    }

    let exp_int = exp_num.to_integer();
    let two: num_bigint::BigInt = 2.into();
    let zero: num_bigint::BigInt = 0.into();
    (exp_int < zero && (&exp_int % &two == num_bigint::BigInt::zero())).then_some(*base)
}

fn positive_reciprocal_requires_nonzero_guard(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return false;
    };
    if !as_rational_const(ctx, *num).is_some_and(|constant| constant.is_positive()) {
        return false;
    }

    extract_abs_argument_view(ctx, *den).is_some()
        || extract_even_nonzero_integer_power_base(ctx, *den).is_some()
        || extract_sqrt_like_base(ctx, *den)
            .and_then(|radicand| extract_even_nonzero_integer_power_base(ctx, radicand))
            .is_some()
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

    if let Some(compact) = compact_scaled_sqrt_square_gap_for_display(ctx, expr) {
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

fn scaled_sqrt_radicand_for_display(ctx: &Context, expr: ExprId) -> Option<(BigRational, ExprId)> {
    let (sign, expr) = match ctx.get(expr) {
        Expr::Neg(inner) => (-BigRational::one(), *inner),
        _ => (BigRational::one(), expr),
    };

    if let Some(radicand) = extract_sqrt_like_base(ctx, expr) {
        return Some((sign, radicand));
    }

    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };

    let mut scale = sign;
    let mut radicand = None;
    for factor in mul_leaves(ctx, expr) {
        if let Some(value) = as_rational_const(ctx, factor) {
            scale *= value;
            continue;
        }

        let factor_radicand = extract_sqrt_like_base(ctx, factor)?;
        if radicand.replace(factor_radicand).is_some() {
            return None;
        }
    }

    Some((scale, radicand?))
}

fn shifted_unit_interval_sqrt_arg_radicand_for_display(
    ctx: &Context,
    expr: ExprId,
) -> Option<ExprId> {
    let two = BigRational::from_integer(2.into());
    match ctx.get(expr) {
        Expr::Sub(left, right) => {
            if as_rational_const(ctx, *right) == Some(BigRational::one()) {
                let (scale, radicand) = scaled_sqrt_radicand_for_display(ctx, *left)?;
                if scale == two {
                    return Some(radicand);
                }
            }

            if as_rational_const(ctx, *left) == Some(BigRational::one()) {
                let (scale, radicand) = scaled_sqrt_radicand_for_display(ctx, *right)?;
                if scale == two {
                    return Some(radicand);
                }
            }
        }
        Expr::Add(left, right) => {
            for (constant_side, sqrt_side) in [(*left, *right), (*right, *left)] {
                let Some(constant) = as_rational_const(ctx, constant_side) else {
                    continue;
                };
                if constant.abs() != BigRational::one() {
                    continue;
                }
                let (scale, radicand) = scaled_sqrt_radicand_for_display(ctx, sqrt_side)?;
                if scale.abs() == two {
                    return Some(radicand);
                }
            }
        }
        _ => {}
    }

    None
}

fn shifted_unit_interval_sqrt_open_interval_parts_for_display(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let squared_arg = match ctx.get(expr) {
        Expr::Sub(left, right) if as_rational_const(ctx, *left) == Some(BigRational::one()) => {
            square_base(ctx, *right)?
        }
        Expr::Add(left, right) => {
            let square_expr = if as_rational_const(ctx, *left) == Some(BigRational::one()) {
                negative_inner(ctx, *right)?
            } else if as_rational_const(ctx, *right) == Some(BigRational::one()) {
                negative_inner(ctx, *left)?
            } else {
                return None;
            };
            square_base(ctx, square_expr)?
        }
        _ => return None,
    };
    let radicand = shifted_unit_interval_sqrt_arg_radicand_for_display(ctx, squared_arg)?;
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let gap = ctx.add(Expr::Sub(sqrt_radicand, radicand));
    let gap = normalize_condition_expr_preserve_sign(ctx, gap);

    Some((radicand, gap))
}

fn negative_inner(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Neg(inner) => Some(*inner),
        _ => None,
    }
}

fn normalize_nonzero_condition_expr_for_display(ctx: &mut Context, expr: ExprId) -> ExprId {
    if let Some(compact) =
        compact_diff_integral_residual_passthrough_for_condition_display(ctx, expr)
    {
        return normalize_nonzero_condition_expr_for_display(ctx, compact);
    }

    if let Some(sinh_expr) = normalize_sinh_nonzero_expr_for_display(ctx, expr) {
        return sinh_expr;
    }

    if let Some(base) = nonzero_integer_power_base_for_display(ctx, expr) {
        return normalize_nonzero_condition_expr_for_display(ctx, base);
    }

    if let Some(base) = compact_expanded_monic_low_degree_power_for_display(ctx, expr) {
        return normalize_nonzero_condition_expr_for_display(ctx, base);
    }

    let normalized = normalize_condition_expr(ctx, expr);
    let normalized =
        primitive_nonzero_polynomial_for_display(ctx, normalized).unwrap_or(normalized);
    if let Some(sinh_expr) = normalize_sinh_nonzero_expr_for_display(ctx, normalized) {
        return sinh_expr;
    }

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

fn normalize_sinh_nonzero_expr_for_display(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let arg = match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.builtin_of(*fn_id) == Some(BuiltinFn::Sinh) && args.len() == 1 =>
        {
            args[0]
        }
        _ => return None,
    };

    let arg = normalize_sinh_nonzero_argument_for_display(ctx, arg);
    Some(ctx.call_builtin(BuiltinFn::Sinh, vec![arg]))
}

fn normalize_sinh_nonzero_argument_for_display(ctx: &mut Context, arg: ExprId) -> ExprId {
    if let Some(sqrt_expr) = normalize_sqrt_argument_preserve_sign_for_display(ctx, arg) {
        return sqrt_expr;
    }

    let normalized = normalize_condition_expr(ctx, arg);
    normalize_sqrt_argument_preserve_sign_for_display(ctx, normalized).unwrap_or(normalized)
}

fn normalize_sqrt_argument_preserve_sign_for_display(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let radicand = extract_sqrt_argument_view(ctx, expr)?;
    let radicand = normalize_condition_expr_preserve_sign(ctx, radicand);
    Some(ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]))
}

fn compact_diff_integral_residual_passthrough_for_condition_display(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() < 3 {
        return None;
    }

    for (diff_index, (diff_term, diff_sign)) in terms.iter().copied().enumerate() {
        let Some(integrand) = diff_of_matching_integral_term(ctx, diff_term) else {
            continue;
        };

        for (target_index, (target_term, target_sign)) in terms.iter().copied().enumerate() {
            if target_index == diff_index || target_sign == diff_sign {
                continue;
            }
            if !exprs_equivalent(ctx, integrand, target_term) {
                continue;
            }

            let mut remaining = Vec::with_capacity(terms.len().saturating_sub(2));
            for (index, (term, sign)) in terms.iter().copied().enumerate() {
                if index == diff_index || index == target_index {
                    continue;
                }
                remaining.push(match sign {
                    Sign::Pos => term,
                    Sign::Neg => ctx.add(Expr::Neg(term)),
                });
            }

            return Some(match remaining.as_slice() {
                [] => ctx.add(Expr::Number(BigRational::zero())),
                [single] => *single,
                _ => build_balanced_add(ctx, &remaining),
            });
        }
    }

    None
}

fn diff_of_matching_integral_term(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(diff_fn, diff_args) = ctx.get(expr) else {
        return None;
    };
    if ctx.sym_name(*diff_fn) != "diff" || diff_args.len() != 2 {
        return None;
    }

    let Expr::Function(integrate_fn, integrate_args) = ctx.get(diff_args[0]) else {
        return None;
    };
    if ctx.sym_name(*integrate_fn) != "integrate" || integrate_args.len() != 2 {
        return None;
    }
    if !exprs_equivalent(ctx, diff_args[1], integrate_args[1]) {
        return None;
    }

    Some(integrate_args[0])
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

fn compact_scaled_sqrt_square_gap_for_display(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let square_expr = match ctx.get(expr).clone() {
        Expr::Sub(left, right) if as_rational_const(ctx, left).is_some_and(|n| n.is_one()) => right,
        Expr::Add(left, right) => {
            if as_rational_const(ctx, left).is_some_and(|n| n.is_one()) {
                negative_inner(ctx, right)?
            } else if as_rational_const(ctx, right).is_some_and(|n| n.is_one()) {
                negative_inner(ctx, left)?
            } else {
                return None;
            }
        }
        _ => return None,
    };

    let square_base = square_base(ctx, square_expr)?;
    let (scale, radicand) = scaled_sqrt_radicand_for_display(ctx, square_base)?;
    if scale.is_zero() {
        return None;
    }

    let scale_squared = scale.clone() * scale;
    let boundary = ctx.add(Expr::Number(BigRational::one() / scale_squared));
    let gap = ctx.add(Expr::Sub(boundary, radicand));
    Some(normalize_condition_expr_preserve_sign(ctx, gap))
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
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
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
    is_positive_multiple_of(ctx, left, right)
        || is_positive_multiple_of(ctx, right, left)
        || inverse_trig_alias_calls_equivalent(ctx, left, right)
        || positive_distributive_display_signatures_equivalent(ctx, left, right)
}

fn inverse_trig_alias_calls_equivalent(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    let (Expr::Function(left_fn, left_args), Expr::Function(right_fn, right_args)) =
        (ctx.get(left), ctx.get(right))
    else {
        return false;
    };
    let (Some(left_builtin), Some(right_builtin)) =
        (ctx.builtin_of(*left_fn), ctx.builtin_of(*right_fn))
    else {
        return false;
    };

    inverse_trig_alias_builtins_equivalent(left_builtin, right_builtin)
        && left_args.len() == right_args.len()
        && left_args
            .iter()
            .zip(right_args)
            .all(|(left_arg, right_arg)| exprs_equivalent(ctx, *left_arg, *right_arg))
}

fn inverse_trig_alias_builtins_equivalent(left: BuiltinFn, right: BuiltinFn) -> bool {
    matches!(
        (left, right),
        (BuiltinFn::Asin, BuiltinFn::Arcsin)
            | (BuiltinFn::Arcsin, BuiltinFn::Asin)
            | (BuiltinFn::Acos, BuiltinFn::Arccos)
            | (BuiltinFn::Arccos, BuiltinFn::Acos)
            | (BuiltinFn::Atan, BuiltinFn::Arctan)
            | (BuiltinFn::Arctan, BuiltinFn::Atan)
            | (BuiltinFn::Asec, BuiltinFn::Arcsec)
            | (BuiltinFn::Arcsec, BuiltinFn::Asec)
            | (BuiltinFn::Acsc, BuiltinFn::Arccsc)
            | (BuiltinFn::Arccsc, BuiltinFn::Acsc)
            | (BuiltinFn::Acot, BuiltinFn::Arccot)
            | (BuiltinFn::Arccot, BuiltinFn::Acot)
    )
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
            ImplicitCondition::LowerBound(_, _) => None,
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

fn nonzero_expansion_has_uncovered_condition(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
    skip_index: usize,
    expr: ExprId,
) -> bool {
    let original = ImplicitCondition::NonZero(expr);
    expand_condition_for_display(ctx, &original)
        .into_iter()
        .filter(|expanded| !conditions_equivalent(ctx, expanded, &original))
        .any(|expanded| {
            !conditions.iter().enumerate().any(|(idx, condition)| {
                idx != skip_index
                    && (conditions_equivalent(ctx, condition, &expanded)
                        || conditions_same_display(ctx, condition, &expanded))
            })
        })
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
            || positive_condition_dominates_sqrt_nonzero(ctx, normalized_positive, normalized_expr)
            || positive_condition_dominates_positive_constant_shift_nonzero(
                ctx,
                normalized_positive,
                normalized_expr,
            )
            || positive_condition_plus_nonnegative_square_dominates_nonzero(
                ctx,
                normalized_positive,
                normalized_expr,
            )
            || positive_condition_times_nonnegative_square_plus_positive_constant_dominates_nonzero(
                ctx,
                normalized_positive,
                normalized_expr,
            )
            || positive_sqrt_square_gap_dominates_nonzero(ctx, normalized_positive, normalized_expr)
            || positive_condition_dominates_affine_nonzero_offset(
                ctx,
                normalized_positive,
                normalized_expr,
            )
            || positive_log_condition_dominates_argument_minus_one_nonzero(
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

fn positive_condition_plus_nonnegative_square_dominates_nonzero(
    ctx: &mut Context,
    positive_expr: ExprId,
    nonzero_expr: ExprId,
) -> bool {
    let positive_expr = normalize_condition_expr_preserve_sign(ctx, positive_expr);
    let stripped_nonzero_expr = strip_nonzero_scalar_factors_for_display(ctx, nonzero_expr);
    let nonzero_expr = normalize_condition_expr(ctx, stripped_nonzero_expr);
    let positive_terms = AddView::from_expr(ctx, positive_expr)
        .terms
        .into_iter()
        .collect::<Vec<_>>();
    let nonzero_terms = AddView::from_expr(ctx, nonzero_expr)
        .terms
        .into_iter()
        .collect::<Vec<_>>();

    let mut remaining_terms = nonzero_terms.clone();
    if remove_matching_signed_terms(ctx, &mut remaining_terms, &positive_terms)
        && single_positive_even_square_term_is_nonnegative(ctx, &remaining_terms)
    {
        return true;
    }

    let mut remaining_terms = nonzero_terms;
    remove_matching_signed_term(ctx, &mut remaining_terms, positive_expr, Sign::Pos)
        && single_positive_even_square_term_is_nonnegative(ctx, &remaining_terms)
}

fn positive_condition_times_nonnegative_square_plus_positive_constant_dominates_nonzero(
    ctx: &mut Context,
    positive_expr: ExprId,
    nonzero_expr: ExprId,
) -> bool {
    let positive_expr = normalize_condition_expr_preserve_sign(ctx, positive_expr);
    let stripped_nonzero_expr = strip_nonzero_scalar_factors_for_display(ctx, nonzero_expr);
    let nonzero_expr = normalize_condition_expr(ctx, stripped_nonzero_expr);
    let mut remaining_terms = AddView::from_expr(ctx, nonzero_expr)
        .terms
        .into_iter()
        .collect::<Vec<_>>();

    if !remove_positive_constant_shift(ctx, &mut remaining_terms) {
        return false;
    }

    let positive_terms = AddView::from_expr(ctx, positive_expr)
        .terms
        .into_iter()
        .collect::<Vec<_>>();
    positive_terms_match_nonnegative_square_product(ctx, &remaining_terms, &positive_terms)
        || positive_terms_match_nonnegative_square_product(
            ctx,
            &remaining_terms,
            &[(positive_expr, Sign::Pos)],
        )
}

fn positive_condition_times_nonnegative_square_plus_positive_constant_dominates_positive(
    ctx: &mut Context,
    positive_expr: ExprId,
    derived_positive_expr: ExprId,
) -> bool {
    positive_condition_times_nonnegative_square_plus_positive_constant_dominates_nonzero(
        ctx,
        positive_expr,
        derived_positive_expr,
    )
}

fn positive_sqrt_square_gap_dominates_nonzero(
    ctx: &mut Context,
    positive_expr: ExprId,
    nonzero_expr: ExprId,
) -> bool {
    let Some(gap) = sqrt_square_gap_equivalent_expr(ctx, positive_expr) else {
        return false;
    };

    exprs_equivalent_up_to_nonzero_scalar(ctx, gap, nonzero_expr)
}

fn sqrt_square_gap_equivalent_expr(ctx: &mut Context, positive_expr: ExprId) -> Option<ExprId> {
    let (constant, square_expr) = positive_constant_minus_square_parts(ctx, positive_expr)?;
    let payload = sqrt_product_square_payload(ctx, square_expr)?;
    let gap = ctx.add(Expr::Sub(constant, payload));
    Some(normalize_condition_expr_preserve_sign(ctx, gap))
}

fn positive_constant_minus_square_parts(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr).clone() {
        Expr::Sub(left, right) if as_rational_const(ctx, left).is_some_and(|n| n.is_positive()) => {
            Some((left, right))
        }
        Expr::Add(left, right) => {
            if as_rational_const(ctx, left).is_some_and(|n| n.is_positive()) {
                let square = negative_inner(ctx, right)?;
                return Some((left, square));
            }
            if as_rational_const(ctx, right).is_some_and(|n| n.is_positive()) {
                let square = negative_inner(ctx, left)?;
                return Some((right, square));
            }
            None
        }
        _ => None,
    }
}

fn sqrt_product_square_payload(ctx: &mut Context, square_expr: ExprId) -> Option<ExprId> {
    let square_base = square_base(ctx, square_expr)?;
    let mut payload_factors = Vec::new();
    let mut saw_sqrt = false;

    for factor in mul_leaves(ctx, square_base) {
        if let Some(radicand) = extract_sqrt_like_base(ctx, factor) {
            payload_factors.push(radicand);
            saw_sqrt = true;
            continue;
        }

        let exponent = ctx.num(2);
        payload_factors.push(ctx.add(Expr::Pow(factor, exponent)));
    }

    if !saw_sqrt {
        return None;
    }

    let payload = match payload_factors.as_slice() {
        [] => return None,
        [single] => *single,
        _ => build_balanced_mul(ctx, &payload_factors),
    };
    Some(normalize_condition_expr_preserve_sign(ctx, payload))
}

fn remove_positive_constant_shift(
    ctx: &Context,
    remaining_terms: &mut Vec<(ExprId, Sign)>,
) -> bool {
    let Some(index) = remaining_terms.iter().position(|(term, sign)| {
        *sign == Sign::Pos
            && cas_math::numeric_eval::as_rational_const(ctx, *term)
                .is_some_and(|value| value.is_positive())
    }) else {
        return false;
    };
    remaining_terms.remove(index);
    !remaining_terms
        .iter()
        .any(|(term, _)| cas_math::numeric_eval::as_rational_const(ctx, *term).is_some())
}

#[derive(Clone)]
struct NonnegativeSquareProductTerm {
    sign: Sign,
    scale: BigRational,
    square_base: ExprId,
    payload: ExprId,
}

fn positive_terms_match_nonnegative_square_product(
    ctx: &mut Context,
    product_terms: &[(ExprId, Sign)],
    positive_terms: &[(ExprId, Sign)],
) -> bool {
    if product_terms.len() != positive_terms.len() || product_terms.is_empty() {
        return false;
    }

    let mut product_terms = product_terms
        .iter()
        .copied()
        .map(|(term, sign)| split_nonnegative_square_product_payload(ctx, term, sign))
        .collect::<Option<Vec<_>>>();
    let Some(mut product_terms) = product_terms.take() else {
        return false;
    };
    let Some(anchor) = product_terms.first().cloned() else {
        return false;
    };

    for (positive_term, positive_sign) in positive_terms {
        let Some(index) = product_terms.iter().position(|product_term| {
            product_term.sign == *positive_sign
                && product_term.scale == anchor.scale
                && exprs_equivalent_up_to_sign(ctx, product_term.square_base, anchor.square_base)
                && exprs_equivalent(ctx, product_term.payload, *positive_term)
        }) else {
            return false;
        };
        product_terms.remove(index);
    }

    product_terms.is_empty()
}

fn split_nonnegative_square_product_payload(
    ctx: &mut Context,
    term: ExprId,
    sign: Sign,
) -> Option<NonnegativeSquareProductTerm> {
    let mut scale = BigRational::one();
    let mut square_base = None;
    let mut payload_factors = Vec::new();

    for factor in mul_leaves(ctx, term) {
        if let Some(value) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
            scale *= value;
            continue;
        }

        if let Some(base) = positive_multiple_even_power_base(ctx, factor) {
            if square_base.replace(base).is_some() {
                return None;
            }
            continue;
        }

        payload_factors.push(factor);
    }

    if !scale.is_positive() {
        return None;
    }

    let payload = match payload_factors.as_slice() {
        [] => ctx.add(Expr::Number(BigRational::one())),
        [single] => *single,
        _ => build_balanced_mul(ctx, &payload_factors),
    };

    Some(NonnegativeSquareProductTerm {
        sign,
        scale,
        square_base: square_base?,
        payload,
    })
}

fn remove_matching_signed_terms(
    ctx: &Context,
    remaining_terms: &mut Vec<(ExprId, Sign)>,
    expected_terms: &[(ExprId, Sign)],
) -> bool {
    for (expected_expr, expected_sign) in expected_terms {
        if !remove_matching_signed_term(ctx, remaining_terms, *expected_expr, *expected_sign) {
            return false;
        }
    }
    true
}

fn remove_matching_signed_term(
    ctx: &Context,
    remaining_terms: &mut Vec<(ExprId, Sign)>,
    expected_expr: ExprId,
    expected_sign: Sign,
) -> bool {
    let Some(index) = remaining_terms.iter().position(|(term, sign)| {
        *sign == expected_sign && exprs_equivalent(ctx, *term, expected_expr)
    }) else {
        return false;
    };
    remaining_terms.remove(index);
    true
}

fn single_positive_even_square_term_is_nonnegative(
    ctx: &mut Context,
    remaining_terms: &[(ExprId, Sign)],
) -> bool {
    let [(term, Sign::Pos)] = remaining_terms else {
        return false;
    };
    positive_multiple_even_power_base(ctx, *term).is_some()
}

fn positive_condition_dominates_positive_constant_shift_nonzero(
    ctx: &mut Context,
    positive_expr: ExprId,
    nonzero_expr: ExprId,
) -> bool {
    let positive_terms = AddView::from_expr(ctx, positive_expr).terms;
    let mut remaining_terms = AddView::from_expr(ctx, nonzero_expr)
        .terms
        .into_iter()
        .collect::<Vec<_>>();

    for (positive_term, positive_sign) in positive_terms {
        let Some(index) = remaining_terms.iter().position(|(term, sign)| {
            *sign == positive_sign && exprs_equivalent(ctx, *term, positive_term)
        }) else {
            return false;
        };
        remaining_terms.remove(index);
    }

    if remaining_terms.is_empty() {
        return false;
    }

    let mut shift = BigRational::zero();
    for (term, sign) in remaining_terms {
        let Some(value) = as_rational_const(ctx, term) else {
            return false;
        };
        match sign {
            Sign::Pos => shift += value,
            Sign::Neg => shift -= value,
        }
    }

    shift.is_positive()
}

fn positive_condition_dominates_sqrt_nonzero(
    ctx: &mut Context,
    positive_expr: ExprId,
    nonzero_expr: ExprId,
) -> bool {
    let Some(radicand) = extract_sqrt_like_base(ctx, nonzero_expr) else {
        return false;
    };
    let normalized_radicand = normalize_condition_expr(ctx, radicand);

    exprs_equivalent_up_to_nonzero_scalar(ctx, normalized_radicand, positive_expr)
        || positive_condition_dominates_affine_nonzero_offset(
            ctx,
            positive_expr,
            normalized_radicand,
        )
        || positive_log_condition_dominates_argument_minus_one_nonzero(
            ctx,
            positive_expr,
            normalized_radicand,
        )
        || positive_polynomial_condition_contains_nonzero_factor(
            ctx,
            positive_expr,
            normalized_radicand,
        )
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

fn positive_log_condition_dominates_argument_minus_one_nonzero(
    ctx: &mut Context,
    positive_expr: ExprId,
    nonzero_expr: ExprId,
) -> bool {
    let Some((base_opt, arg)) = extract_log_base_argument_view(ctx, positive_expr) else {
        return false;
    };
    if base_opt.is_some() {
        return false;
    }

    let one = ctx.num(1);
    let arg_minus_one = ctx.add(Expr::Sub(arg, one));
    let normalized_boundary = normalize_condition_expr(ctx, arg_minus_one);
    let normalized_nonzero = normalize_condition_expr(ctx, nonzero_expr);

    exprs_equivalent_up_to_sign(ctx, normalized_nonzero, normalized_boundary)
}

fn affine_root(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    affine_root_and_slope(ctx, expr).map(|(root, _)| root)
}

fn affine_root_and_slope(ctx: &Context, expr: ExprId) -> Option<(BigRational, BigRational)> {
    let vars = cas_ast::collect_variables(ctx, expr);
    if vars.len() != 1 {
        return None;
    }
    let var_name = vars.iter().next()?;
    let poly = Polynomial::from_expr(ctx, expr, var_name.as_str()).ok()?;
    if poly.degree() != 1 {
        return None;
    }
    let slope = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if slope.is_zero() {
        return None;
    }
    let constant = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    Some((-constant / slope.clone(), slope))
}

fn positive_affine_product_dominates_affine_nonzero(
    ctx: &Context,
    positive_expr: ExprId,
    nonzero_expr: ExprId,
) -> bool {
    let factors: Vec<_> = mul_leaves(ctx, positive_expr).into_iter().collect();
    if factors.len() != 2 {
        return false;
    }

    let Some((left_root, left_slope)) = affine_root_and_slope(ctx, factors[0]) else {
        return false;
    };
    let Some((right_root, right_slope)) = affine_root_and_slope(ctx, factors[1]) else {
        return false;
    };
    if left_root == right_root {
        return false;
    }
    if !(left_slope * right_slope).is_positive() {
        return false;
    }
    let Some(nonzero_root) = affine_root(ctx, nonzero_expr) else {
        return false;
    };

    let (lower, upper) = if left_root < right_root {
        (left_root, right_root)
    } else {
        (right_root, left_root)
    };
    lower <= nonzero_root && nonzero_root <= upper
}

fn quadratic_roots_with_leading(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, BigRational, BigRational)> {
    let vars = cas_ast::collect_variables(ctx, expr);
    if vars.len() != 1 {
        return None;
    }
    let var_name = vars.iter().next()?;
    let poly = Polynomial::from_expr(ctx, expr, var_name.as_str()).ok()?;
    if poly.degree() != 2 {
        return None;
    }
    let a = poly
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
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
    if a.is_zero() {
        return None;
    }

    let four = BigRational::from_integer(4.into());
    let two = BigRational::from_integer(2.into());
    let discriminant = b.clone() * b.clone() - four * a.clone() * c;
    let root_disc = rational_sqrt(&discriminant)?;
    let leading_sign = a.signum();
    let denominator = two * a;
    if denominator.is_zero() {
        return None;
    }

    Some((
        leading_sign,
        (-b.clone() - root_disc.clone()) / denominator.clone(),
        (-b + root_disc) / denominator,
    ))
}

fn positive_quadratic_dominates_affine_nonzero(
    ctx: &Context,
    positive_expr: ExprId,
    nonzero_expr: ExprId,
) -> bool {
    let Some((leading_sign, left_root, right_root)) =
        quadratic_roots_with_leading(ctx, positive_expr)
    else {
        return false;
    };
    if left_root == right_root {
        return false;
    }
    if !leading_sign.is_positive() {
        return false;
    }
    let Some(nonzero_root) = affine_root(ctx, nonzero_expr) else {
        return false;
    };

    let (lower, upper) = if left_root < right_root {
        (left_root, right_root)
    } else {
        (right_root, left_root)
    };
    lower <= nonzero_root && nonzero_root <= upper
}

fn positive_affine_product_dominates_quotient_nonnegative(
    ctx: &mut Context,
    positive_expr: ExprId,
    quotient_expr: ExprId,
) -> bool {
    let Expr::Div(num, den) = ctx.get(quotient_expr).clone() else {
        return false;
    };

    let vars = cas_ast::collect_variables(ctx, quotient_expr);
    if vars.len() != 1 {
        return false;
    }
    let Some(var_name) = vars.iter().next() else {
        return false;
    };
    let Ok(num_poly) = Polynomial::from_expr(ctx, num, var_name.as_str()) else {
        return false;
    };
    let Ok(den_poly) = Polynomial::from_expr(ctx, den, var_name.as_str()) else {
        return false;
    };
    if num_poly.degree() != 1 || den_poly.degree() != 1 {
        return false;
    }

    let product = cas_math::expr_nary::build_balanced_mul(ctx, &[num, den]);
    exprs_equivalent_up_to_nonzero_scalar(ctx, positive_expr, product)
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

fn positive_condition_dominates_affine_unit_ratio_gap(
    ctx: &mut Context,
    positive_expr: ExprId,
    derived_positive_expr: ExprId,
) -> bool {
    let Some((num, den)) = one_minus_ratio_parts(ctx, derived_positive_expr) else {
        return false;
    };

    let normalized_positive = normalize_condition_expr_preserve_sign(ctx, positive_expr);
    let normalized_num = normalize_condition_expr_preserve_sign(ctx, num);
    if !positive_condition_dominates_affine_positive_offset(
        ctx,
        normalized_positive,
        normalized_num,
    ) {
        return false;
    }

    let mut vars = cas_ast::collect_variables(ctx, normalized_num);
    vars.extend(cas_ast::collect_variables(ctx, den));
    if vars.len() != 1 {
        return false;
    }
    let Some(var_name) = vars.iter().next() else {
        return false;
    };

    let Ok(num_poly) = Polynomial::from_expr(ctx, normalized_num, var_name.as_str()) else {
        return false;
    };
    let Ok(den_poly) = Polynomial::from_expr(ctx, den, var_name.as_str()) else {
        return false;
    };
    let gap_poly = den_poly.sub(&num_poly);
    gap_poly.degree() == 0
        && gap_poly
            .coeffs
            .first()
            .is_some_and(|offset| offset.is_positive())
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

fn positive_product_polynomial_is_dominated_by_positive_factor_pair(
    ctx: &Context,
    product_expr: ExprId,
    known_positive_exprs: &[ExprId],
) -> bool {
    use cas_math::multipoly::{multipoly_from_expr, PolyBudget};

    if known_positive_exprs.len() < 2 {
        return false;
    }

    let budget = PolyBudget {
        max_terms: 64,
        max_total_degree: 8,
        max_pow_exp: 8,
    };
    let Ok(product_poly) = multipoly_from_expr(ctx, product_expr, &budget) else {
        return false;
    };
    if product_poly.is_zero() || product_poly.total_degree() < 2 {
        return false;
    }

    for (left_idx, left_expr) in known_positive_exprs.iter().enumerate() {
        let Ok(left_poly) = multipoly_from_expr(ctx, *left_expr, &budget) else {
            continue;
        };
        if !left_poly.vars.iter().all(|var| {
            product_poly
                .vars
                .iter()
                .any(|product_var| product_var == var)
        }) {
            continue;
        }
        let left_poly = left_poly.align_vars(&product_poly.vars);

        for right_expr in known_positive_exprs.iter().skip(left_idx + 1) {
            let Ok(right_poly) = multipoly_from_expr(ctx, *right_expr, &budget) else {
                continue;
            };
            if !right_poly.vars.iter().all(|var| {
                product_poly
                    .vars
                    .iter()
                    .any(|product_var| product_var == var)
            }) {
                continue;
            }
            let right_poly = right_poly.align_vars(&product_poly.vars);
            let Ok(candidate_product) = left_poly.mul(&right_poly, &budget) else {
                continue;
            };

            if polynomial_is_positive_scalar_multiple(&candidate_product, &product_poly) {
                return true;
            }
        }
    }

    false
}

fn polynomial_is_positive_scalar_multiple(source: &MultiPoly, target: &MultiPoly) -> bool {
    if source.vars != target.vars || source.is_zero() || target.is_zero() {
        return false;
    }

    let mut scale: Option<BigRational> = None;
    for (source_coeff, source_mono) in &source.terms {
        let Some((target_coeff, _)) = target
            .terms
            .iter()
            .find(|(_, target_mono)| target_mono == source_mono)
        else {
            return false;
        };
        let term_scale = target_coeff / source_coeff;
        match &scale {
            Some(existing) if existing != &term_scale => return false,
            Some(_) => {}
            None => scale = Some(term_scale),
        }
    }

    if target.terms.iter().any(|(_, target_mono)| {
        !source
            .terms
            .iter()
            .any(|(_, source_mono)| source_mono == target_mono)
    }) {
        return false;
    }

    scale.is_some_and(|value| value.is_positive())
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

    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if as_rational_const(ctx, *exp).is_some_and(|n| n.is_positive())
            && is_nonnegative_under_display_conditions(
                ctx,
                conditions,
                skip_index,
                *base,
                depth - 1,
            )
        {
            return true;
        }
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

    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if as_rational_const(ctx, *exp).is_some_and(|n| n.is_positive())
            && is_positive_under_display_conditions(ctx, conditions, skip_index, *base, depth - 1)
        {
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
            nonzero_exprs_equivalent_for_display(ctx, *a, *b)
        }
        _ => false,
    }
}

fn nonzero_exprs_equivalent_for_display(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    nonzero_exprs_algebraically_equivalent_for_display(ctx, left, right)
        || same_unary_zero_set_conditions_equivalent_for_display(ctx, left, right)
        || supported_integral_condition_exprs_equivalent_for_display(ctx, left, right)
}

fn nonzero_exprs_algebraically_equivalent_for_display(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    exprs_equivalent_up_to_sign(ctx, left, right)
        || distributive_display_signatures_equivalent(ctx, left, right)
}

fn same_unary_zero_set_conditions_equivalent_for_display(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    let Some((left_builtin, left_arg)) = unary_zero_set_condition_arg(ctx, left) else {
        return false;
    };
    let Some((right_builtin, right_arg)) = unary_zero_set_condition_arg(ctx, right) else {
        return false;
    };

    left_builtin == right_builtin
        && zero_set_condition_args_equivalent_for_display(ctx, left_arg, right_arg)
}

fn unary_zero_set_condition_arg(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let builtin = ctx.builtin_of(*fn_id)?;
    zero_set_condition_builtin_is_sign_stable(builtin).then_some((builtin, args[0]))
}

fn zero_set_condition_builtin_is_sign_stable(builtin: BuiltinFn) -> bool {
    matches!(builtin, BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Sinh)
}

fn zero_set_condition_args_equivalent_for_display(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    match (
        extract_sqrt_argument_view(ctx, left),
        extract_sqrt_argument_view(ctx, right),
    ) {
        (Some(left_radicand), Some(right_radicand)) => {
            exprs_equivalent(ctx, left_radicand, right_radicand)
        }
        _ => exprs_equivalent_up_to_sign(ctx, left, right),
    }
}

fn supported_integral_condition_exprs_equivalent_for_display(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    supported_integral_condition_rewrite_matches(ctx, left, right)
        || supported_integral_condition_rewrite_matches(ctx, right, left)
}

fn supported_integral_condition_rewrite_matches(
    ctx: &Context,
    source: ExprId,
    target: ExprId,
) -> bool {
    if !contains_integrate_call_for_condition_display(ctx, source, 12) {
        return false;
    }

    let mut rewrite_ctx = ctx.clone();
    let Some(rewritten_source) =
        rewrite_supported_integrate_calls_for_condition_display(&mut rewrite_ctx, source, 12)
    else {
        return false;
    };

    nonzero_exprs_algebraically_equivalent_for_display(&rewrite_ctx, rewritten_source, target)
}

fn contains_integrate_call_for_condition_display(
    ctx: &Context,
    expr: ExprId,
    depth: usize,
) -> bool {
    if depth == 0 {
        return false;
    }

    match ctx.get(expr) {
        Expr::Function(fn_id, args) if ctx.sym_name(*fn_id) == "integrate" => {
            matches!(args.len(), 1 | 2)
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            contains_integrate_call_for_condition_display(ctx, *left, depth - 1)
                || contains_integrate_call_for_condition_display(ctx, *right, depth - 1)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            contains_integrate_call_for_condition_display(ctx, *inner, depth - 1)
        }
        Expr::Function(_, args) => args
            .iter()
            .any(|arg| contains_integrate_call_for_condition_display(ctx, *arg, depth - 1)),
        _ => false,
    }
}

fn rewrite_supported_integrate_calls_for_condition_display(
    ctx: &mut Context,
    expr: ExprId,
    depth: usize,
) -> Option<ExprId> {
    if depth == 0 {
        return None;
    }

    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if ctx.sym_name(fn_id) == "integrate" => {
            let (target, var_name) = match args.as_slice() {
                [target] => (*target, "x".to_string()),
                [target, var_expr] => match ctx.get(*var_expr) {
                    Expr::Variable(var_sym) => (*target, ctx.sym_name(*var_sym).to_string()),
                    _ => return None,
                },
                _ => return None,
            };
            cas_math::symbolic_integration_support::integrate_symbolic_expr(ctx, target, &var_name)
        }
        Expr::Add(left, right) => {
            let new_left =
                rewrite_supported_integrate_calls_for_condition_display(ctx, left, depth - 1);
            let new_right =
                rewrite_supported_integrate_calls_for_condition_display(ctx, right, depth - 1);
            if new_left.is_none() && new_right.is_none() {
                return None;
            }
            Some(ctx.add(Expr::Add(
                new_left.unwrap_or(left),
                new_right.unwrap_or(right),
            )))
        }
        Expr::Sub(left, right) => {
            let new_left =
                rewrite_supported_integrate_calls_for_condition_display(ctx, left, depth - 1);
            let new_right =
                rewrite_supported_integrate_calls_for_condition_display(ctx, right, depth - 1);
            if new_left.is_none() && new_right.is_none() {
                return None;
            }
            Some(ctx.add(Expr::Sub(
                new_left.unwrap_or(left),
                new_right.unwrap_or(right),
            )))
        }
        Expr::Mul(left, right) => {
            let new_left =
                rewrite_supported_integrate_calls_for_condition_display(ctx, left, depth - 1);
            let new_right =
                rewrite_supported_integrate_calls_for_condition_display(ctx, right, depth - 1);
            if new_left.is_none() && new_right.is_none() {
                return None;
            }
            Some(ctx.add(Expr::Mul(
                new_left.unwrap_or(left),
                new_right.unwrap_or(right),
            )))
        }
        Expr::Div(left, right) => {
            let new_left =
                rewrite_supported_integrate_calls_for_condition_display(ctx, left, depth - 1);
            let new_right =
                rewrite_supported_integrate_calls_for_condition_display(ctx, right, depth - 1);
            if new_left.is_none() && new_right.is_none() {
                return None;
            }
            Some(ctx.add(Expr::Div(
                new_left.unwrap_or(left),
                new_right.unwrap_or(right),
            )))
        }
        Expr::Pow(left, right) => {
            let new_left =
                rewrite_supported_integrate_calls_for_condition_display(ctx, left, depth - 1);
            let new_right =
                rewrite_supported_integrate_calls_for_condition_display(ctx, right, depth - 1);
            if new_left.is_none() && new_right.is_none() {
                return None;
            }
            Some(ctx.add(Expr::Pow(
                new_left.unwrap_or(left),
                new_right.unwrap_or(right),
            )))
        }
        Expr::Neg(inner) => {
            let new_inner =
                rewrite_supported_integrate_calls_for_condition_display(ctx, inner, depth - 1)?;
            Some(ctx.add(Expr::Neg(new_inner)))
        }
        Expr::Hold(inner) => {
            let new_inner =
                rewrite_supported_integrate_calls_for_condition_display(ctx, inner, depth - 1)?;
            Some(ctx.add(Expr::Hold(new_inner)))
        }
        Expr::Function(fn_id, args) => {
            let mut changed = false;
            let mut new_args = Vec::with_capacity(args.len());
            for arg in args {
                if let Some(new_arg) =
                    rewrite_supported_integrate_calls_for_condition_display(ctx, arg, depth - 1)
                {
                    changed = true;
                    new_args.push(new_arg);
                } else {
                    new_args.push(arg);
                }
            }
            changed.then(|| ctx.add(Expr::Function(fn_id, new_args)))
        }
        _ => None,
    }
}

fn distributive_display_signatures_equivalent(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    let Some(left_signature) = bounded_distributive_display_signature(ctx, left) else {
        return false;
    };
    let Some(right_signature) = bounded_distributive_display_signature(ctx, right) else {
        return false;
    };

    signatures_equal_up_to_sign(&left_signature, &right_signature)
}

fn positive_distributive_display_signatures_equivalent(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    let Some(left_signature) = bounded_distributive_display_signature(ctx, left) else {
        return false;
    };
    let Some(right_signature) = bounded_distributive_display_signature(ctx, right) else {
        return false;
    };

    left_signature == right_signature
}

type DisplayTermSignature = Vec<String>;
type DisplayLinearSignature = BTreeMap<DisplayTermSignature, BigRational>;

fn bounded_distributive_display_signature(
    ctx: &Context,
    expr: ExprId,
) -> Option<DisplayLinearSignature> {
    let terms = product_display_signature_terms(ctx, expr)?;
    if terms.len() > CONDITION_DISPLAY_SIGNATURE_MAX_TERMS {
        return None;
    }

    let mut signature = DisplayLinearSignature::new();
    for (coeff, factors) in terms {
        if coeff.is_zero() {
            continue;
        }
        let entry = signature.entry(factors).or_insert_with(BigRational::zero);
        *entry = entry.clone() + coeff;
    }
    signature.retain(|_, coeff| !coeff.is_zero());

    (!signature.is_empty()).then_some(signature)
}

fn product_display_signature_terms(
    ctx: &Context,
    expr: ExprId,
) -> Option<Vec<(BigRational, DisplayTermSignature)>> {
    let mut terms = vec![(BigRational::one(), DisplayTermSignature::new())];

    for factor in mul_leaves(ctx, expr) {
        let factor_terms = factor_display_signature_terms(ctx, factor)?;
        let mut next_terms = Vec::new();

        for (base_coeff, base_factors) in &terms {
            for (factor_coeff, factor_factors) in &factor_terms {
                let mut combined_factors = base_factors.clone();
                combined_factors.extend(factor_factors.iter().cloned());
                if combined_factors.len() > CONDITION_DISPLAY_SIGNATURE_MAX_FACTORS {
                    return None;
                }
                combined_factors.sort();
                next_terms.push((base_coeff.clone() * factor_coeff.clone(), combined_factors));
                if next_terms.len() > CONDITION_DISPLAY_SIGNATURE_MAX_TERMS {
                    return None;
                }
            }
        }

        terms = next_terms;
    }

    Some(terms)
}

fn factor_display_signature_terms(
    ctx: &Context,
    factor: ExprId,
) -> Option<Vec<(BigRational, DisplayTermSignature)>> {
    let (factor_coeff, factor_expr) = signed_numeric_factor(ctx, factor)?;
    let Some(factor_expr) = factor_expr else {
        return Some(vec![(factor_coeff, DisplayTermSignature::new())]);
    };

    let additive_terms = AddView::from_expr(ctx, factor_expr).terms;
    if additive_terms.len() > 1 {
        let mut distributed = Vec::new();
        for (term_expr, term_sign) in additive_terms {
            let signed_coeff = match term_sign {
                Sign::Pos => factor_coeff.clone(),
                Sign::Neg => -factor_coeff.clone(),
            };
            for (term_coeff, term_factors) in product_display_signature_terms(ctx, term_expr)? {
                distributed.push((signed_coeff.clone() * term_coeff, term_factors));
                if distributed.len() > CONDITION_DISPLAY_SIGNATURE_MAX_TERMS {
                    return None;
                }
            }
        }
        return Some(distributed);
    }

    Some(vec![(
        factor_coeff,
        vec![condition_factor_key(ctx, factor_expr)],
    )])
}

fn signed_numeric_factor(ctx: &Context, expr: ExprId) -> Option<(BigRational, Option<ExprId>)> {
    if let Some(number) = as_rational_const(ctx, expr) {
        return Some((number, None));
    }

    if let Expr::Neg(inner) = ctx.get(expr) {
        let (coeff, inner_expr) = signed_numeric_factor(ctx, *inner)?;
        return Some((-coeff, inner_expr));
    }

    Some((BigRational::one(), Some(expr)))
}

fn condition_factor_key(ctx: &Context, expr: ExprId) -> String {
    use cas_formatter::DisplayExpr;

    if let Some(base) = extract_sqrt_like_base(ctx, expr) {
        return format!(
            "sqrt({})",
            DisplayExpr {
                context: ctx,
                id: base,
            }
        );
    }

    DisplayExpr {
        context: ctx,
        id: expr,
    }
    .to_string()
}

fn signatures_equal_up_to_sign(
    left: &DisplayLinearSignature,
    right: &DisplayLinearSignature,
) -> bool {
    left == right
        || (left.len() == right.len()
            && left
                .iter()
                .all(|(factors, coeff)| right.get(factors).is_some_and(|rhs| rhs == &-coeff)))
}

fn conditions_same_display(ctx: &Context, c1: &ImplicitCondition, c2: &ImplicitCondition) -> bool {
    c1.display(ctx) == c2.display(ctx)
}

fn should_prefer_condition_display(
    ctx: &Context,
    existing: &ImplicitCondition,
    candidate: &ImplicitCondition,
) -> bool {
    match (existing, candidate) {
        (
            ImplicitCondition::Positive(existing_expr),
            ImplicitCondition::Positive(candidate_expr),
        )
        | (
            ImplicitCondition::NonNegative(existing_expr),
            ImplicitCondition::NonNegative(candidate_expr),
        ) => should_prefer_inverse_trig_alias_display(ctx, *existing_expr, *candidate_expr),
        (ImplicitCondition::NonZero(existing_expr), ImplicitCondition::NonZero(candidate_expr)) => {
            should_prefer_compact_same_unary_zero_set_display(ctx, *existing_expr, *candidate_expr)
        }
        _ => false,
    }
}

fn should_prefer_compact_same_unary_zero_set_display(
    ctx: &Context,
    existing: ExprId,
    candidate: ExprId,
) -> bool {
    same_unary_zero_set_conditions_equivalent_for_display(ctx, existing, candidate)
        && condition_expr_display_len(ctx, candidate) < condition_expr_display_len(ctx, existing)
}

fn condition_expr_display_len(ctx: &Context, expr: ExprId) -> usize {
    use cas_formatter::DisplayExpr;

    DisplayExpr {
        context: ctx,
        id: expr,
    }
    .to_string()
    .len()
}

fn should_prefer_inverse_trig_alias_display(
    ctx: &Context,
    existing: ExprId,
    candidate: ExprId,
) -> bool {
    if !inverse_trig_alias_calls_equivalent(ctx, existing, candidate) {
        return false;
    }

    let Some(existing_builtin) = function_builtin(ctx, existing) else {
        return false;
    };
    let Some(candidate_builtin) = function_builtin(ctx, candidate) else {
        return false;
    };

    is_long_inverse_trig_alias(existing_builtin) && is_short_inverse_trig_alias(candidate_builtin)
}

fn function_builtin(ctx: &Context, expr: ExprId) -> Option<BuiltinFn> {
    let Expr::Function(fn_id, _) = ctx.get(expr) else {
        return None;
    };
    ctx.builtin_of(*fn_id)
}

fn is_short_inverse_trig_alias(builtin: BuiltinFn) -> bool {
    matches!(
        builtin,
        BuiltinFn::Asin
            | BuiltinFn::Acos
            | BuiltinFn::Atan
            | BuiltinFn::Asec
            | BuiltinFn::Acsc
            | BuiltinFn::Acot
    )
}

fn is_long_inverse_trig_alias(builtin: BuiltinFn) -> bool {
    matches!(
        builtin,
        BuiltinFn::Arcsin
            | BuiltinFn::Arccos
            | BuiltinFn::Arctan
            | BuiltinFn::Arcsec
            | BuiltinFn::Arccsc
            | BuiltinFn::Arccot
    )
}

/// Normalize and deduplicate a list of conditions for display.
pub fn normalize_and_dedupe_conditions(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
) -> Vec<ImplicitCondition> {
    let mut result: Vec<ImplicitCondition> = Vec::new();

    for (idx, cond) in conditions.iter().enumerate() {
        if condition_is_intrinsically_satisfied(ctx, cond) {
            continue;
        }

        if let ImplicitCondition::NonZero(expr) = cond {
            let dominated = is_positive_under_display_conditions_or_factored(
                ctx,
                conditions,
                idx,
                *expr,
                DISPLAY_SIGN_PROOF_DEPTH,
            ) || nonzero_is_dominated_by_positive_condition(ctx, conditions, *expr);
            if dominated && !nonzero_expansion_has_uncovered_condition(ctx, conditions, idx, *expr)
            {
                continue;
            }
        }

        for normalized in expand_condition_for_display(ctx, cond) {
            let duplicate_index = result.iter().position(|existing| {
                conditions_equivalent(ctx, existing, &normalized)
                    || conditions_same_display(ctx, existing, &normalized)
            });

            if let Some(duplicate_index) = duplicate_index {
                if should_prefer_condition_display(ctx, &result[duplicate_index], &normalized) {
                    result[duplicate_index] = normalized;
                }
            } else {
                result.push(normalized);
            }
        }
    }

    combine_nonzero_nonnegative_into_positive(ctx, &mut result);
    rewrite_sign_quotients_with_positive_components(ctx, &mut result);
    combine_nonzero_nonnegative_into_positive(ctx, &mut result);
    combine_factored_nonzero_nonnegative_into_positive(ctx, &mut result);
    apply_dominance_rules(ctx, &mut result);
    drop_acosh_log_argument_positive_dominated_by_branch_gap(ctx, &mut result);
    drop_nonzero_conditions_with_matching_positive_condition(ctx, &mut result);
    if result.is_empty() {
        restore_nonzero_domain_hole_guards(ctx, conditions, &mut result);
    }
    result
}

fn restore_nonzero_domain_hole_guards(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
    result: &mut Vec<ImplicitCondition>,
) {
    for cond in conditions {
        let restored = match cond {
            ImplicitCondition::NonZero(expr)
                if !condition_is_intrinsically_satisfied(ctx, cond) =>
            {
                expand_nonzero_condition_for_display(ctx, *expr)
            }
            ImplicitCondition::Positive(expr)
                if positive_sqrt_even_power_base(ctx, *expr).is_some()
                    || positive_reciprocal_requires_nonzero_guard(ctx, *expr) =>
            {
                expand_condition_for_display(ctx, cond)
            }
            _ => Vec::new(),
        };

        for condition in restored {
            if condition_is_intrinsically_satisfied(ctx, &condition) {
                continue;
            }
            if !result.iter().any(|existing| {
                conditions_equivalent(ctx, existing, &condition)
                    || conditions_same_display(ctx, existing, &condition)
            }) {
                result.push(condition);
            }
        }
    }
}

fn condition_is_intrinsically_satisfied(ctx: &mut Context, cond: &ImplicitCondition) -> bool {
    match cond {
        ImplicitCondition::Positive(expr) => {
            positive_sqrt_even_power_base(ctx, *expr).is_none()
                && !positive_reciprocal_requires_nonzero_guard(ctx, *expr)
                && (is_intrinsically_positive_real(ctx, *expr)
                    || positive_quadratic_power_gap_is_intrinsic(ctx, *expr)
                    || positive_sqrt_polynomial_gap_sum_is_intrinsic(ctx, *expr))
        }
        ImplicitCondition::NonNegative(expr) => {
            is_intrinsically_nonnegative_real(ctx, *expr)
                || is_intrinsically_positive_real(ctx, *expr)
                || positive_quadratic_power_gap_is_intrinsic(ctx, *expr)
                || unit_nonnegative_ratio_gap_is_intrinsic(ctx, *expr)
        }
        ImplicitCondition::LowerBound(expr, lower) => match ctx.get(*expr) {
            Expr::Number(value) => value >= lower,
            _ => false,
        },
        ImplicitCondition::NonZero(expr) => {
            nonnegative_negative_even_power_base(ctx, *expr).is_none()
                && (is_intrinsically_positive_real(ctx, *expr)
                    || nonzero_trig_table_condition_is_intrinsic(ctx, *expr))
        }
    }
}

fn nonzero_trig_table_condition_is_intrinsic(ctx: &mut Context, expr: ExprId) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return false;
    };
    let Some(builtin) = ctx.builtin_of(*fn_id) else {
        return false;
    };
    if args.len() != 1
        || !matches!(
            builtin,
            BuiltinFn::Sin
                | BuiltinFn::Cos
                | BuiltinFn::Tan
                | BuiltinFn::Sec
                | BuiltinFn::Csc
                | BuiltinFn::Cot
        )
    {
        return false;
    }

    let arg = structurally_cancel_additive_inverse_terms(ctx, args[0]);
    lookup_trig_or_inverse(ctx, builtin.name(), arg)
        .is_some_and(|hit| !matches!(hit.value, TrigValue::Zero | TrigValue::Undefined))
}

fn structurally_cancel_additive_inverse_terms(ctx: &mut Context, expr: ExprId) -> ExprId {
    if !matches!(
        ctx.get(expr),
        Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Neg(_)
    ) {
        return expr;
    }

    let terms: Vec<_> = add_terms_signed(ctx, expr).into_iter().collect();
    if terms.len() < 2 {
        return expr;
    }

    let mut cancelled = vec![false; terms.len()];
    for i in 0..terms.len() {
        if cancelled[i] {
            continue;
        }
        let (term, sign) = terms[i];
        for j in (i + 1)..terms.len() {
            if cancelled[j] {
                continue;
            }
            let (other_term, other_sign) = terms[j];
            if other_sign == sign.negate() && compare_expr(ctx, term, other_term).is_eq() {
                cancelled[i] = true;
                cancelled[j] = true;
                break;
            }
        }
    }

    let mut rebuilt_terms = Vec::new();
    for (idx, (term, sign)) in terms.into_iter().enumerate() {
        if cancelled[idx] {
            continue;
        }
        match sign {
            Sign::Pos => rebuilt_terms.push(term),
            Sign::Neg => rebuilt_terms.push(ctx.add(Expr::Neg(term))),
        }
    }

    build_balanced_add(ctx, &rebuilt_terms)
}

fn unit_nonnegative_ratio_gap_is_intrinsic(ctx: &Context, expr: ExprId) -> bool {
    let Some((num, den)) = one_minus_ratio_parts(ctx, expr) else {
        return false;
    };
    if !is_intrinsically_nonnegative_real(ctx, num) {
        return false;
    }

    let mut vars = cas_ast::collect_variables(ctx, num);
    vars.extend(cas_ast::collect_variables(ctx, den));
    if vars.len() != 1 {
        return false;
    }
    let Some(var_name) = vars.iter().next() else {
        return false;
    };

    let Ok(num_poly) = Polynomial::from_expr(ctx, num, var_name.as_str()) else {
        return false;
    };
    let Ok(den_poly) = Polynomial::from_expr(ctx, den, var_name.as_str()) else {
        return false;
    };
    let gap_poly = den_poly.sub(&num_poly);
    gap_poly.degree() == 0
        && gap_poly
            .coeffs
            .first()
            .is_some_and(|coeff| coeff.is_positive())
}

fn one_minus_ratio_parts(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr) {
        Expr::Sub(left, right)
            if as_rational_const(ctx, *left).is_some_and(|value| value.is_one()) =>
        {
            ratio_parts(ctx, *right)
        }
        Expr::Add(left, right) => {
            if as_rational_const(ctx, *left).is_some_and(|value| value.is_one()) {
                return negative_ratio_parts(ctx, *right);
            }
            if as_rational_const(ctx, *right).is_some_and(|value| value.is_one()) {
                return negative_ratio_parts(ctx, *left);
            }
            None
        }
        _ => None,
    }
}

fn ratio_parts(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr) {
        Expr::Div(num, den) => Some((*num, *den)),
        _ => None,
    }
}

fn negative_ratio_parts(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr) {
        Expr::Neg(inner) => ratio_parts(ctx, *inner),
        Expr::Mul(left, right)
            if as_rational_const(ctx, *left).is_some_and(|value| value == -BigRational::one()) =>
        {
            ratio_parts(ctx, *right)
        }
        Expr::Mul(left, right)
            if as_rational_const(ctx, *right).is_some_and(|value| value == -BigRational::one()) =>
        {
            ratio_parts(ctx, *left)
        }
        _ => None,
    }
}

fn positive_sqrt_polynomial_gap_sum_is_intrinsic(ctx: &Context, expr: ExprId) -> bool {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
    if terms.len() < 2 {
        return false;
    }

    let vars = cas_ast::collect_variables(ctx, expr);
    if vars.len() != 1 {
        return false;
    }
    let Some(var) = vars.iter().next() else {
        return false;
    };

    let mut radicand = None;
    let mut term_poly = Polynomial::zero(var.to_string());
    for (term, sign) in terms {
        if let Some(term_radicand) = sqrt_or_half_power_argument(ctx, term) {
            if sign == Sign::Neg {
                return false;
            }
            if radicand.is_some() {
                return false;
            }
            radicand = Some(term_radicand);
        } else {
            let Ok(mut next_poly) = Polynomial::from_expr(ctx, term, var.as_str()) else {
                return false;
            };
            if sign == Sign::Neg {
                next_poly = next_poly.neg();
            }
            term_poly = term_poly.add(&next_poly);
        }
    }

    let Some(radicand) = radicand else {
        return false;
    };
    let Ok(radicand_poly) = Polynomial::from_expr(ctx, radicand, var.as_str()) else {
        return false;
    };
    let gap = radicand_poly.sub(&term_poly.mul(&term_poly));
    gap.degree() == 0 && gap.coeffs.first().is_some_and(|value| value.is_positive())
}

fn sqrt_or_half_power_argument(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Some(arg) = extract_sqrt_argument_view(ctx, expr) {
        return Some(arg);
    }

    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    (as_rational_const(ctx, *exp) == Some(BigRational::new(1.into(), 2.into()))).then_some(*base)
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
        ImplicitCondition::NonNegative(expr) => {
            if let Some(base) = nonnegative_negative_even_power_base(ctx, *expr) {
                return expand_nonzero_condition_for_display(ctx, base);
            }

            let normalized = normalize_condition(ctx, cond);
            if condition_is_intrinsically_satisfied(ctx, &normalized) {
                Vec::new()
            } else {
                vec![normalized]
            }
        }
        ImplicitCondition::LowerBound(_, _) => {
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
    if let Some(expanded) = expand_positive_affine_partition_quotient_condition(ctx, expr) {
        return Some(expanded);
    }
    if let Some(expanded) = expand_positive_affine_quotient_as_product_condition(ctx, expr) {
        return Some(expanded);
    }

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

fn expand_positive_affine_quotient_as_product_condition(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<Vec<ImplicitCondition>> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let vars = cas_ast::collect_variables(ctx, expr);
    if vars.len() != 1 {
        return None;
    }
    let var_name = vars.iter().next()?;
    let num_poly = Polynomial::from_expr(ctx, num, var_name.as_str()).ok()?;
    let den_poly = Polynomial::from_expr(ctx, den, var_name.as_str()).ok()?;
    if num_poly.degree() != 1 || den_poly.degree() != 1 {
        return None;
    }

    let product = cas_math::expr_nary::build_balanced_mul(ctx, &[num, den]);
    Some(vec![normalize_condition(
        ctx,
        &ImplicitCondition::Positive(product),
    )])
}

fn expand_positive_affine_partition_quotient_condition(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<Vec<ImplicitCondition>> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let vars = cas_ast::collect_variables(ctx, expr);
    if vars.len() != 1 {
        return None;
    }
    let var_name = vars.iter().next()?;
    let num_poly = Polynomial::from_expr(ctx, num, var_name.as_str()).ok()?;
    let den_poly = Polynomial::from_expr(ctx, den, var_name.as_str()).ok()?;
    if num_poly.degree() > 1 || den_poly.degree() > 1 {
        return None;
    }

    let partition_sum = num_poly.add(&den_poly);
    if partition_sum.degree() != 0 {
        return None;
    }
    let partition_total = partition_sum
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if !partition_total.is_positive() {
        return None;
    }

    let numerator_positive = normalize_condition(ctx, &ImplicitCondition::Positive(num));
    let denominator_positive = normalize_condition(ctx, &ImplicitCondition::Positive(den));
    Some(vec![numerator_positive, denominator_positive])
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

    let normalized_arg = normalize_sinh_nonzero_argument_for_display(ctx, arg);
    Some(ctx.call_builtin(BuiltinFn::Sinh, vec![normalized_arg]))
}

fn expand_common_factor_sum_nonzero_for_display(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<Vec<ImplicitCondition>> {
    if let Some(expanded) = expand_nary_common_factor_sum_nonzero_for_display(ctx, expr) {
        return Some(expanded);
    }

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

fn expand_nary_common_factor_sum_nonzero_for_display(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<Vec<ImplicitCondition>> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 3 {
        return None;
    }

    let mut term_factors: Vec<(Vec<ExprId>, Sign)> = view
        .terms
        .iter()
        .map(|(term, sign)| (mul_leaves(ctx, *term).into_iter().collect(), *sign))
        .collect();

    if term_factors.iter().any(|(factors, _)| factors.is_empty()) {
        return None;
    }

    let mut common_factors = Vec::new();
    for candidate in term_factors[0].0.clone() {
        if as_rational_const(ctx, candidate).is_some_and(|constant| !constant.is_zero()) {
            continue;
        }

        let mut matched_indices = Vec::with_capacity(term_factors.len());
        for (factors, _) in &term_factors {
            let Some(index) = factors
                .iter()
                .position(|factor| exprs_equivalent(ctx, candidate, *factor))
            else {
                matched_indices.clear();
                break;
            };
            matched_indices.push(index);
        }

        if matched_indices.len() != term_factors.len() {
            continue;
        }

        for ((factors, _), index) in term_factors.iter_mut().zip(matched_indices.into_iter()) {
            factors.remove(index);
        }
        common_factors.push(candidate);
    }

    if common_factors.is_empty() {
        return None;
    }

    let common_expr = build_mul_or_one(ctx, &common_factors);
    let mut residual_terms = Vec::with_capacity(term_factors.len());
    for (factors, sign) in term_factors {
        let term = build_mul_or_one(ctx, &factors);
        residual_terms.push(match sign {
            Sign::Pos => term,
            Sign::Neg => ctx.add(Expr::Neg(term)),
        });
    }
    let residual_expr = build_balanced_add(ctx, &residual_terms);

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

fn positive_condition_dominates_sqrt_lower_positive(
    ctx: &mut Context,
    positive_expr: ExprId,
    derived_positive_expr: ExprId,
) -> bool {
    let Some(boundary) = sqrt_lower_nonzero_boundary(ctx, derived_positive_expr) else {
        return false;
    };

    positive_ordered_exprs_equivalent(ctx, positive_expr, boundary)
}

fn positive_condition_dominates_acosh_sqrt_gap_nonzero(
    ctx: &mut Context,
    positive_expr: ExprId,
    nonzero_expr: ExprId,
) -> bool {
    if positive_polynomial_gap_dominates_square_gap_sqrt_nonzero(ctx, positive_expr, nonzero_expr) {
        return true;
    }

    let Some((polynomial_term, gap_value)) = acosh_sqrt_gap_nonzero_parts(ctx, nonzero_expr) else {
        return false;
    };

    condition_implies_acosh_log_argument_positive(
        ctx,
        &ImplicitCondition::Positive(positive_expr),
        polynomial_term,
        &gap_value,
    ) || {
        let negated_polynomial_term = ctx.add(Expr::Neg(polynomial_term));
        condition_implies_acosh_log_argument_positive(
            ctx,
            &ImplicitCondition::Positive(positive_expr),
            negated_polynomial_term,
            &gap_value,
        )
    }
}

fn acosh_sqrt_gap_nonzero_parts(
    ctx: &mut Context,
    nonzero_expr: ExprId,
) -> Option<(ExprId, BigRational)> {
    let nonzero_expr = cas_ast::hold::unwrap_internal_hold(ctx, nonzero_expr);
    let radicand = extract_sqrt_like_base(ctx, nonzero_expr)?;
    acosh_radicand_gap_parts(ctx, radicand)
}

fn positive_polynomial_gap_dominates_square_gap_sqrt_nonzero(
    ctx: &mut Context,
    positive_expr: ExprId,
    nonzero_expr: ExprId,
) -> bool {
    let nonzero_expr = cas_ast::hold::unwrap_internal_hold(ctx, nonzero_expr);
    let Some(radicand) = extract_sqrt_like_base(ctx, nonzero_expr) else {
        return false;
    };
    let positive_expr = cas_ast::hold::unwrap_internal_hold(ctx, positive_expr);

    let mut vars = cas_ast::collect_variables(ctx, radicand);
    vars.extend(cas_ast::collect_variables(ctx, positive_expr));
    if vars.len() != 1 {
        return false;
    }
    let Some(var_name) = vars.iter().next() else {
        return false;
    };

    let Ok(radicand_poly) = Polynomial::from_expr(ctx, radicand, var_name.as_str()) else {
        return false;
    };
    let Ok(gap_poly) = Polynomial::from_expr(ctx, positive_expr, var_name.as_str()) else {
        return false;
    };
    if gap_poly.is_zero() {
        return false;
    }

    let gap_squared = gap_poly.mul(&gap_poly);
    let remainder_after_square = radicand_poly.sub(&gap_squared);
    let Ok((quotient, remainder)) = remainder_after_square.div_rem(&gap_poly) else {
        return false;
    };

    if !remainder.is_zero() || quotient.degree() != 0 {
        return false;
    }

    quotient
        .coeffs
        .first()
        .is_some_and(|constant| constant.is_positive())
}

fn positive_condition_dominates_acosh_radicand_nonnegative(
    ctx: &mut Context,
    positive_expr: ExprId,
    nonnegative_expr: ExprId,
) -> bool {
    let Some((polynomial_term, gap_value)) = acosh_radicand_gap_parts(ctx, nonnegative_expr) else {
        return false;
    };

    condition_implies_acosh_log_argument_positive(
        ctx,
        &ImplicitCondition::Positive(positive_expr),
        polynomial_term,
        &gap_value,
    ) || {
        let negated_polynomial_term = ctx.add(Expr::Neg(polynomial_term));
        condition_implies_acosh_log_argument_positive(
            ctx,
            &ImplicitCondition::Positive(positive_expr),
            negated_polynomial_term,
            &gap_value,
        )
    }
}

fn acosh_radicand_gap_parts(ctx: &mut Context, radicand: ExprId) -> Option<(ExprId, BigRational)> {
    let radicand = cas_ast::hold::unwrap_internal_hold(ctx, radicand);
    let vars = cas_ast::collect_variables(ctx, radicand);
    if vars.len() != 1 {
        return None;
    }
    let var_name = vars.iter().next()?;
    let radicand_poly = Polynomial::from_expr(ctx, radicand, var_name.as_str()).ok()?;
    if radicand_poly.degree() != 2 {
        return None;
    }

    let quadratic = radicand_poly.coeffs.get(2)?;
    let linear = radicand_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let constant = radicand_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let slope = exact_positive_rational_nth_root(quadratic, 2)?;
    let two = BigRational::from_integer(2.into());
    let shift = linear / (two * slope.clone());
    let gap_value = shift.clone() * shift.clone() - constant;
    if !gap_value.is_positive() {
        return None;
    }

    let polynomial_term = Polynomial::new(vec![shift, slope], var_name.to_string()).to_expr(ctx);
    Some((polynomial_term, gap_value))
}

fn positive_condition_dominates_shifted_unit_sqrt_open_interval(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
    skip_index: usize,
    positive_expr: ExprId,
    open_interval_expr: ExprId,
) -> bool {
    let Some((radicand, compact_gap)) =
        shifted_unit_interval_sqrt_open_interval_parts_for_display(ctx, open_interval_expr)
    else {
        return false;
    };

    let normalized_positive = normalize_condition_expr_preserve_sign(ctx, positive_expr);
    let gap_matches = positive_ordered_exprs_equivalent(ctx, normalized_positive, compact_gap)
        || conditions_same_display(
            ctx,
            &ImplicitCondition::Positive(normalized_positive),
            &ImplicitCondition::Positive(compact_gap),
        );
    if !gap_matches {
        return false;
    }

    conditions.iter().enumerate().any(|(idx, condition)| {
        if idx == skip_index {
            return false;
        }
        let ImplicitCondition::Positive(other_expr) = condition else {
            return false;
        };
        positive_ordered_exprs_equivalent(ctx, *other_expr, radicand)
    })
}

fn positive_condition_present_equiv(
    ctx: &Context,
    conditions: &[ImplicitCondition],
    skip_index: usize,
    target: ExprId,
) -> bool {
    conditions.iter().enumerate().any(|(idx, condition)| {
        if idx == skip_index {
            return false;
        }
        let ImplicitCondition::Positive(expr) = condition else {
            return false;
        };
        positive_ordered_exprs_equivalent(ctx, *expr, target)
            || conditions_same_display(
                ctx,
                &ImplicitCondition::Positive(*expr),
                &ImplicitCondition::Positive(target),
            )
    })
}

fn unit_sqrt_gap_positive_parts(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return None;
    }

    let mut radicand = None;
    let mut other = None;
    for (term, sign) in terms {
        if let Some((mut scale, term_radicand)) = scaled_sqrt_radicand_for_display(ctx, term) {
            if sign == Sign::Neg {
                scale = -scale;
            }
            if !scale.is_one() || radicand.replace(term_radicand).is_some() {
                return None;
            }
        } else if other.replace((term, sign)).is_some() {
            return None;
        }
    }

    let radicand = radicand?;
    let (other_term, other_sign) = other?;
    let signed_other = if other_sign == Sign::Neg {
        ctx.add(Expr::Neg(other_term))
    } else {
        other_term
    };
    let neg_radicand = ctx.add(Expr::Neg(radicand));
    if !exprs_equivalent(ctx, signed_other, neg_radicand) {
        return None;
    }

    let one = ctx.num(1);
    let boundary = ctx.add(Expr::Sub(one, radicand));
    Some((radicand, boundary))
}

fn positive_unit_sqrt_gap_dominated_by_atomic_bounds(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
    skip_index: usize,
    gap_expr: ExprId,
) -> bool {
    let Some((radicand, boundary)) = unit_sqrt_gap_positive_parts(ctx, gap_expr) else {
        return false;
    };

    positive_condition_present_equiv(ctx, conditions, skip_index, radicand)
        && positive_condition_present_equiv(ctx, conditions, skip_index, boundary)
}

fn unit_sqrt_boundary_for_nonzero(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return None;
    }

    let mut constant = BigRational::zero();
    let mut radicand = None;
    let mut sqrt_scale = BigRational::zero();
    for (term, sign) in terms {
        if let Some(value) = as_rational_const(ctx, term) {
            constant += if sign == Sign::Neg { -value } else { value };
            continue;
        }
        let (mut scale, term_radicand) = scaled_sqrt_radicand_for_display(ctx, term)?;
        if sign == Sign::Neg {
            scale = -scale;
        }
        sqrt_scale += scale;
        if radicand.replace(term_radicand).is_some() {
            return None;
        }
    }

    if constant.abs() != BigRational::one() || sqrt_scale.abs() != BigRational::one() {
        return None;
    }

    let radicand = radicand?;
    let one = ctx.num(1);
    let boundary = ctx.add(Expr::Sub(one, radicand));
    Some((radicand, boundary))
}

fn positive_bounds_dominate_unit_sqrt_nonzero(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
    skip_index: usize,
    nonzero_expr: ExprId,
) -> bool {
    let Some((radicand, boundary)) = unit_sqrt_boundary_for_nonzero(ctx, nonzero_expr) else {
        return false;
    };

    positive_condition_present_equiv(ctx, conditions, skip_index, radicand)
        && positive_condition_present_equiv(ctx, conditions, skip_index, boundary)
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

fn nonzero_condition_dominates_lower_bound(
    ctx: &mut Context,
    nonzero_expr: ExprId,
    bounded_expr: ExprId,
    lower: &BigRational,
) -> bool {
    let lower_expr = ctx.add(Expr::Number(lower.clone()));
    let gap = ctx.add(Expr::Sub(bounded_expr, lower_expr));
    let normalized_gap = normalize_condition_expr(ctx, gap);
    let factored_gap = factor(ctx, normalized_gap);

    for candidate in [gap, normalized_gap, factored_gap] {
        if nonzero_integer_power_base_for_display(ctx, candidate)
            .is_some_and(|base| exprs_equivalent_up_to_sign(ctx, nonzero_expr, base))
        {
            return true;
        }
    }

    false
}

fn positive_condition_dominates_reciprocal_surd_lower_bound(
    ctx: &mut Context,
    positive_expr: ExprId,
    bounded_expr: ExprId,
    lower: &BigRational,
) -> bool {
    if !lower.is_one() {
        return false;
    }

    let mut rational_coeff = BigRational::one();
    let mut sqrt_arg = None;
    let mut payload_factors = Vec::new();
    collect_reciprocal_surd_lower_bound_parts(
        ctx,
        bounded_expr,
        &mut rational_coeff,
        &mut sqrt_arg,
        &mut payload_factors,
    );

    let Some(sqrt_arg) = sqrt_arg else {
        return false;
    };
    if payload_factors.is_empty() || rational_coeff != BigRational::one() / sqrt_arg.clone() {
        return false;
    }

    let payload = build_balanced_mul(ctx, &payload_factors);
    let sqrt_arg_expr = ctx.add(Expr::Number(sqrt_arg));
    let sqrt_expr = ctx.call_builtin(BuiltinFn::Sqrt, vec![sqrt_arg_expr]);
    let expected_gap = ctx.add(Expr::Sub(payload, sqrt_expr));
    let normalized_gap = normalize_condition_expr_preserve_sign(ctx, expected_gap);

    positive_ordered_exprs_equivalent(ctx, positive_expr, expected_gap)
        || positive_ordered_exprs_equivalent(ctx, positive_expr, normalized_gap)
        || conditions_same_display(
            ctx,
            &ImplicitCondition::Positive(positive_expr),
            &ImplicitCondition::Positive(expected_gap),
        )
        || conditions_same_display(
            ctx,
            &ImplicitCondition::Positive(positive_expr),
            &ImplicitCondition::Positive(normalized_gap),
        )
}

fn drop_acosh_log_argument_positive_dominated_by_branch_gap(
    ctx: &mut Context,
    conditions: &mut Vec<ImplicitCondition>,
) {
    let mut to_remove = Vec::new();

    for (idx, condition) in conditions.iter().enumerate() {
        let Some((polynomial_term, gap_value)) = acosh_log_argument_positive_parts(ctx, condition)
        else {
            continue;
        };

        if conditions.iter().enumerate().any(|(other_idx, other)| {
            other_idx != idx
                && condition_implies_acosh_log_argument_positive(
                    ctx,
                    other,
                    polynomial_term,
                    &gap_value,
                )
        }) {
            to_remove.push(idx);
        }
    }

    to_remove.sort_unstable();
    to_remove.dedup();
    for idx in to_remove.into_iter().rev() {
        conditions.remove(idx);
    }
}

fn acosh_log_argument_positive_parts(
    ctx: &mut Context,
    condition: &ImplicitCondition,
) -> Option<(ExprId, BigRational)> {
    let expr = match condition {
        ImplicitCondition::Positive(expr) => *expr,
        ImplicitCondition::LowerBound(expr, lower) if lower.is_zero() => *expr,
        _ => return None,
    };

    let mut radicand = None;
    let mut polynomial_terms = Vec::new();
    for (term, sign) in AddView::from_expr(ctx, expr).terms {
        if let Some(term_radicand) = extract_sqrt_like_base(ctx, term) {
            if sign == Sign::Neg {
                return None;
            }
            if radicand.replace(term_radicand).is_some() {
                return None;
            }
        } else {
            polynomial_terms.push((term, sign));
        }
    }

    let radicand = radicand?;
    if polynomial_terms.is_empty() {
        return None;
    }
    let mut vars = cas_ast::collect_variables(ctx, radicand);
    for (term, _) in &polynomial_terms {
        vars.extend(cas_ast::collect_variables(ctx, *term));
    }
    if vars.len() != 1 {
        return None;
    }
    let var_name = vars.iter().next()?;
    let radicand_poly = Polynomial::from_expr(ctx, radicand, var_name.as_str()).ok()?;
    let mut polynomial_poly = Polynomial::zero(var_name.to_string());
    for (term, sign) in polynomial_terms {
        let mut term_poly = Polynomial::from_expr(ctx, term, var_name.as_str()).ok()?;
        if sign == Sign::Neg {
            term_poly = term_poly.neg();
        }
        polynomial_poly = polynomial_poly.add(&term_poly);
    }
    if polynomial_poly.is_zero() {
        return None;
    }
    let polynomial_term = polynomial_poly.to_expr(ctx);
    let square_gap = polynomial_poly.mul(&polynomial_poly).sub(&radicand_poly);
    if square_gap.degree() != 0 {
        return None;
    }
    let gap_value = square_gap.coeffs.first()?;
    gap_value
        .is_positive()
        .then(|| (polynomial_term, gap_value.clone()))
}

fn condition_implies_acosh_log_argument_positive(
    ctx: &mut Context,
    condition: &ImplicitCondition,
    polynomial_term: ExprId,
    gap_value: &BigRational,
) -> bool {
    if let Some(boundary) = exact_positive_rational_nth_root(gap_value, 2) {
        match condition {
            ImplicitCondition::LowerBound(expr, lower) => {
                return (lower == &boundary && exprs_equivalent(ctx, *expr, polynomial_term))
                    || lower_bound_implies_affine_branch_gap(
                        ctx,
                        *expr,
                        lower,
                        polynomial_term,
                        &boundary,
                    );
            }
            ImplicitCondition::Positive(expr) => {
                let boundary_expr = ctx.add(Expr::Number(boundary));
                let expected_gap = ctx.add(Expr::Sub(polynomial_term, boundary_expr));
                if exprs_equivalent(ctx, *expr, expected_gap)
                    || positive_ordered_exprs_equivalent(ctx, *expr, expected_gap)
                    || conditions_same_display(
                        ctx,
                        condition,
                        &ImplicitCondition::Positive(expected_gap),
                    )
                {
                    return true;
                }
            }
            _ => {}
        }
    }

    let gap_expr = ctx.add(Expr::Number(gap_value.clone()));
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap_expr]);
    let expected_gap = ctx.add(Expr::Sub(polynomial_term, sqrt_gap));
    match condition {
        ImplicitCondition::Positive(expr) => {
            exprs_equivalent(ctx, *expr, expected_gap)
                || positive_ordered_exprs_equivalent(ctx, *expr, expected_gap)
                || conditions_same_display(
                    ctx,
                    condition,
                    &ImplicitCondition::Positive(expected_gap),
                )
        }
        _ => false,
    }
}

fn lower_bound_implies_affine_branch_gap(
    ctx: &mut Context,
    bounded_expr: ExprId,
    lower: &BigRational,
    polynomial_term: ExprId,
    boundary: &BigRational,
) -> bool {
    let mut vars = cas_ast::collect_variables(ctx, bounded_expr);
    vars.extend(cas_ast::collect_variables(ctx, polynomial_term));
    if vars.len() != 1 {
        return false;
    }
    let Some(var_name) = vars.iter().next() else {
        return false;
    };
    let Ok(bounded_poly) = Polynomial::from_expr(ctx, bounded_expr, var_name.as_str()) else {
        return false;
    };
    let Ok(branch_poly) = Polynomial::from_expr(ctx, polynomial_term, var_name.as_str()) else {
        return false;
    };
    if bounded_poly.degree() != 1 || branch_poly.degree() != 1 {
        return false;
    }

    let Some(bounded_slope) = bounded_poly.coeffs.get(1).filter(|value| !value.is_zero()) else {
        return false;
    };
    let Some(branch_slope) = branch_poly.coeffs.get(1).filter(|value| !value.is_zero()) else {
        return false;
    };
    let scale = branch_slope / bounded_slope;
    if !scale.is_positive() {
        return false;
    }

    let bounded_const = bounded_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let branch_const = branch_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let shift = branch_const - scale.clone() * bounded_const;
    let expected_lower = (boundary - shift) / scale;

    lower == &expected_lower
}

fn collect_reciprocal_surd_lower_bound_parts(
    ctx: &Context,
    expr: ExprId,
    rational_coeff: &mut BigRational,
    sqrt_arg: &mut Option<BigRational>,
    payload_factors: &mut Vec<ExprId>,
) {
    if let Some(value) = as_rational_const(ctx, expr) {
        *rational_coeff *= value;
        return;
    }

    if let Some(arg) = extract_sqrt_like_base(ctx, expr) {
        if let Some(value) = as_rational_const(ctx, arg) {
            if value.is_positive() && sqrt_arg.is_none() {
                *sqrt_arg = Some(value);
                return;
            }
        }
    }

    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            collect_reciprocal_surd_lower_bound_parts(
                ctx,
                *left,
                rational_coeff,
                sqrt_arg,
                payload_factors,
            );
            collect_reciprocal_surd_lower_bound_parts(
                ctx,
                *right,
                rational_coeff,
                sqrt_arg,
                payload_factors,
            );
        }
        Expr::Div(num, den) => {
            if let Some(value) = as_rational_const(ctx, *den) {
                collect_reciprocal_surd_lower_bound_parts(
                    ctx,
                    *num,
                    rational_coeff,
                    sqrt_arg,
                    payload_factors,
                );
                *rational_coeff /= value;
            } else if let Some(arg) = extract_sqrt_like_base(ctx, *den) {
                if let Some(value) = as_rational_const(ctx, arg) {
                    if value.is_positive() && sqrt_arg.is_none() {
                        collect_reciprocal_surd_lower_bound_parts(
                            ctx,
                            *num,
                            rational_coeff,
                            sqrt_arg,
                            payload_factors,
                        );
                        *sqrt_arg = Some(value.clone());
                        *rational_coeff /= value;
                    } else {
                        payload_factors.push(expr);
                    }
                } else {
                    payload_factors.push(expr);
                }
            } else {
                payload_factors.push(expr);
            }
        }
        _ => payload_factors.push(expr),
    }
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
                        || positive_condition_dominates_positive_constant_shift_nonzero(
                            ctx, *pos_expr, *nz_expr,
                        )
                        || positive_condition_plus_nonnegative_square_dominates_nonzero(
                            ctx, *pos_expr, *nz_expr,
                        )
                        || positive_condition_times_nonnegative_square_plus_positive_constant_dominates_nonzero(
                            ctx, *pos_expr, *nz_expr,
                        )
                        || positive_sqrt_square_gap_dominates_nonzero(ctx, *pos_expr, *nz_expr)
                        || positive_log_condition_dominates_argument_minus_one_nonzero(
                            ctx, *pos_expr, *nz_expr,
                        )
                        || positive_affine_product_dominates_affine_nonzero(
                            ctx, *pos_expr, *nz_expr,
                        )
                        || positive_quadratic_dominates_affine_nonzero(ctx, *pos_expr, *nz_expr)
                        || positive_condition_dominates_sqrt_lower_nonzero(ctx, *pos_expr, *nz_expr)
                        || positive_condition_dominates_acosh_sqrt_gap_nonzero(
                            ctx, *pos_expr, *nz_expr,
                        )
                        || positive_bounds_dominate_unit_sqrt_nonzero(ctx, conditions, i, *nz_expr)
                        || positive_polynomial_condition_contains_nonzero_factor(
                            ctx, *pos_expr, *nz_expr,
                        )
                        || positive_trig_quotient_condition_dominates_nonzero(
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
                        || positive_condition_dominates_sqrt_lower_positive(
                            ctx,
                            *pos_expr,
                            *derived_expr,
                        )
                        || positive_condition_dominates_affine_unit_ratio_gap(
                            ctx,
                            *pos_expr,
                            *derived_expr,
                        )
                        || positive_condition_dominates_shifted_unit_sqrt_open_interval(
                            ctx,
                            conditions,
                            i,
                            *pos_expr,
                            *derived_expr,
                        )
                        || positive_unit_sqrt_gap_dominated_by_atomic_bounds(
                            ctx,
                            conditions,
                            i,
                            *derived_expr,
                        )
                        || positive_condition_dominates_acosh_radicand_nonnegative(
                            ctx,
                            *pos_expr,
                            *derived_expr,
                        )
                        || positive_condition_times_nonnegative_square_plus_positive_constant_dominates_positive(
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
                        || condition_factor_key(ctx, *nn_expr)
                            == condition_factor_key(ctx, *pos_expr)
                        || is_odd_power_of(ctx, *nn_expr, *pos_expr)
                        || is_positive_multiple_of(ctx, *nn_expr, *pos_expr)
                        || positive_affine_product_dominates_quotient_nonnegative(
                            ctx, *pos_expr, *nn_expr,
                        )
                        || positive_condition_dominates_reciprocal_offset_nonnegative(
                            ctx, *pos_expr, *nn_expr,
                        )
                        || positive_condition_dominates_acosh_radicand_nonnegative(
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
                (
                    ImplicitCondition::NonNegative(nonnegative_expr),
                    ImplicitCondition::LowerBound(lower_expr, lower),
                ) => {
                    if !lower.is_negative() && exprs_equivalent(ctx, *nonnegative_expr, *lower_expr)
                    {
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
                    ImplicitCondition::LowerBound(bounded_expr, lower),
                    ImplicitCondition::Positive(pos_expr),
                ) => {
                    if positive_expr_implies_lower_bound(ctx, *pos_expr, *bounded_expr, lower)
                        || positive_condition_dominates_reciprocal_surd_lower_bound(
                            ctx,
                            *pos_expr,
                            *bounded_expr,
                            lower,
                        )
                    {
                        to_remove.push(i);
                        break;
                    }
                }
                (
                    ImplicitCondition::LowerBound(bounded_expr, lower),
                    ImplicitCondition::NonZero(nz_expr),
                ) => {
                    if nonzero_condition_dominates_lower_bound(ctx, *nz_expr, *bounded_expr, lower)
                    {
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

                if is_product_dominated_by_positives(ctx, *nn_expr, &other_positive_exprs) {
                    to_remove.push(i);
                    continue;
                }

                if positive_product_is_dominated_by_known_or_intrinsic_positive_factors(
                    ctx,
                    *nn_expr,
                    &other_positive_exprs,
                ) {
                    to_remove.push(i);
                    continue;
                }

                let factored = factor(ctx, *nn_expr);
                if factored != *nn_expr
                    && is_product_dominated_by_positives(ctx, factored, &other_positive_exprs)
                {
                    to_remove.push(i);
                    continue;
                }

                if factored != *nn_expr
                    && positive_product_is_dominated_by_known_or_intrinsic_positive_factors(
                        ctx,
                        factored,
                        &other_positive_exprs,
                    )
                {
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

                if positive_product_is_dominated_by_known_or_intrinsic_positive_factors(
                    ctx,
                    *prod_expr,
                    &other_positive_exprs,
                ) {
                    to_remove.push(i);
                    continue;
                }

                let factored = factor(ctx, *prod_expr);
                if factored != *prod_expr
                    && is_product_dominated_by_positives(ctx, factored, &other_positive_exprs)
                {
                    to_remove.push(i);
                    continue;
                }

                if factored != *prod_expr
                    && positive_product_is_dominated_by_known_or_intrinsic_positive_factors(
                        ctx,
                        factored,
                        &other_positive_exprs,
                    )
                {
                    to_remove.push(i);
                    continue;
                }

                if positive_product_polynomial_is_dominated_by_positive_factor_pair(
                    ctx,
                    *prod_expr,
                    &other_positive_exprs,
                ) {
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

fn positive_trig_quotient_condition_dominates_nonzero(
    ctx: &mut Context,
    positive_expr: ExprId,
    nonzero_expr: ExprId,
) -> bool {
    let Some((num_arg, den_arg)) = sin_cos_positive_quotient_args(ctx, positive_expr) else {
        return false;
    };
    if !exprs_equivalent(ctx, num_arg, den_arg) {
        return false;
    }

    let num_call = ctx.call_builtin(BuiltinFn::Sin, vec![num_arg]);
    let den_call = ctx.call_builtin(BuiltinFn::Cos, vec![den_arg]);
    let cot_num_call = ctx.call_builtin(BuiltinFn::Cos, vec![num_arg]);
    let cot_den_call = ctx.call_builtin(BuiltinFn::Sin, vec![den_arg]);
    if exprs_equivalent(ctx, nonzero_expr, num_call)
        || exprs_equivalent(ctx, nonzero_expr, den_call)
        || exprs_equivalent(ctx, nonzero_expr, cot_num_call)
        || exprs_equivalent(ctx, nonzero_expr, cot_den_call)
    {
        return true;
    }

    let two = ctx.num(2);
    let double_arg = ctx.add(Expr::Mul(two, num_arg));
    let sin_double = ctx.call_builtin(BuiltinFn::Sin, vec![double_arg]);
    exprs_equivalent_up_to_sign(ctx, nonzero_expr, sin_double)
        || sine_double_angle_matches_arg(ctx, nonzero_expr, num_arg)
}

fn sin_cos_positive_quotient_args(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let (num_fn, num_arg) = sin_or_cos_arg(ctx, *num)?;
    let (den_fn, den_arg) = sin_or_cos_arg(ctx, *den)?;
    match (num_fn, den_fn) {
        (BuiltinFn::Sin, BuiltinFn::Cos) | (BuiltinFn::Cos, BuiltinFn::Sin) => {
            Some((num_arg, den_arg))
        }
        _ => None,
    }
}

fn sin_or_cos_arg(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => match ctx.builtin_of(*fn_id) {
            Some(BuiltinFn::Sin) => Some((BuiltinFn::Sin, args[0])),
            Some(BuiltinFn::Cos) => Some((BuiltinFn::Cos, args[0])),
            _ => None,
        },
        _ => None,
    }
}

fn sine_double_angle_matches_arg(ctx: &Context, expr: ExprId, arg: ExprId) -> bool {
    let Some(sin_arg) = unary_builtin_arg(ctx, expr, BuiltinFn::Sin) else {
        return false;
    };
    let Expr::Mul(left, right) = ctx.get(sin_arg) else {
        return false;
    };

    (is_two_constant(ctx, *left) && exprs_equivalent_or_same_sqrt_like_base(ctx, *right, arg))
        || (is_two_constant(ctx, *right)
            && exprs_equivalent_or_same_sqrt_like_base(ctx, *left, arg))
}

fn is_two_constant(ctx: &Context, expr: ExprId) -> bool {
    as_rational_const(ctx, expr)
        .is_some_and(|constant| constant == BigRational::from_integer(2.into()))
}

fn drop_nonzero_conditions_with_matching_positive_condition(
    ctx: &Context,
    conditions: &mut Vec<ImplicitCondition>,
) {
    let mut positive_keys = std::collections::HashSet::new();
    for condition in conditions.iter() {
        if let ImplicitCondition::Positive(expr) = condition {
            positive_keys.insert(condition_factor_key(ctx, *expr));
        }
    }
    if positive_keys.is_empty() {
        return;
    }

    conditions.retain(|condition| match condition {
        ImplicitCondition::NonZero(expr) => {
            !positive_keys.contains(&condition_factor_key(ctx, *expr))
        }
        _ => true,
    });
}

fn positive_product_is_dominated_by_known_or_intrinsic_positive_factors(
    ctx: &mut Context,
    product_expr: ExprId,
    known_positive_exprs: &[ExprId],
) -> bool {
    if known_positive_exprs.is_empty() {
        return false;
    }

    if positive_additive_product_is_dominated_by_known_factor_and_intrinsic_remainder(
        ctx,
        product_expr,
        known_positive_exprs,
    ) {
        return true;
    }

    if positive_even_power_gap_is_dominated_by_surd_lower_gap(
        ctx,
        product_expr,
        known_positive_exprs,
    ) {
        return true;
    }

    if positive_polynomial_product_is_dominated_by_known_factor_and_intrinsic_remainder(
        ctx,
        product_expr,
        known_positive_exprs,
    ) {
        return true;
    }

    let factors = mul_leaves(ctx, product_expr);
    if factors.len() < 2 {
        return false;
    }

    let mut saw_known_positive_factor = false;
    for factor_expr in factors {
        if factor_is_known_or_intrinsic_positive(ctx, factor_expr, known_positive_exprs) {
            saw_known_positive_factor |= known_positive_exprs
                .iter()
                .any(|known| positive_condition_factor_matches_known(ctx, factor_expr, *known));
            continue;
        }

        let factored_factor = factor(ctx, factor_expr);
        if factored_factor != factor_expr {
            let mut factored_saw_known = false;
            let all_factored_parts_positive =
                mul_leaves(ctx, factored_factor).into_iter().all(|part| {
                    let part_is_known = known_positive_exprs
                        .iter()
                        .any(|known| positive_condition_factor_matches_known(ctx, part, *known));
                    factored_saw_known |= part_is_known;
                    part_is_known
                        || as_rational_const(ctx, part).is_some_and(|value| value.is_positive())
                        || is_intrinsically_positive_real(ctx, part)
                        || positive_quadratic_power_gap_is_intrinsic(ctx, part)
                });
            if all_factored_parts_positive {
                saw_known_positive_factor |= factored_saw_known;
                continue;
            }
        }

        if is_intrinsically_positive_real(ctx, factor_expr)
            || positive_quadratic_power_gap_is_intrinsic(ctx, factor_expr)
        {
            continue;
        }

        return false;
    }

    saw_known_positive_factor
}

fn factor_is_known_or_intrinsic_positive(
    ctx: &mut Context,
    factor: ExprId,
    known_positive_exprs: &[ExprId],
) -> bool {
    as_rational_const(ctx, factor).is_some_and(|value| value.is_positive())
        || known_positive_exprs
            .iter()
            .any(|known| positive_condition_factor_matches_known(ctx, factor, *known))
        || is_intrinsically_positive_real(ctx, factor)
        || positive_quadratic_power_gap_is_intrinsic(ctx, factor)
}

fn positive_condition_factor_matches_known(
    ctx: &mut Context,
    factor: ExprId,
    known: ExprId,
) -> bool {
    exprs_equivalent(ctx, factor, known)
        || sqrt_like_gap_conditions_equivalent(ctx, factor, known)
        || condition_exprs_equivalent_after_sqrt_half_power_normalization(ctx, factor, known)
        || {
            let normalized_factor = normalize_condition_expr_preserve_sign(ctx, factor);
            let normalized_known = normalize_condition_expr_preserve_sign(ctx, known);
            exprs_equivalent(ctx, normalized_factor, known)
                || exprs_equivalent(ctx, factor, normalized_known)
                || exprs_equivalent(ctx, normalized_factor, normalized_known)
                || condition_exprs_equivalent_after_sqrt_half_power_normalization(
                    ctx,
                    normalized_factor,
                    normalized_known,
                )
        }
        || is_positive_power_of_base(ctx, factor, known)
        || is_abs_of(ctx, factor, known)
        || positive_condition_dominates_affine_positive_offset(ctx, known, factor)
        || positive_condition_dominates_sqrt_lower_positive(ctx, known, factor)
}

fn sqrt_like_gap_conditions_equivalent(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    let Some((left_root_base, left_rhs)) = sqrt_like_gap_parts(ctx, left) else {
        return false;
    };
    let Some((right_root_base, right_rhs)) = sqrt_like_gap_parts(ctx, right) else {
        return false;
    };

    exprs_equivalent(ctx, left_root_base, right_root_base)
        && exprs_equivalent(ctx, left_rhs, right_rhs)
}

fn sqrt_like_gap_parts(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr) {
        Expr::Sub(left, right) => extract_sqrt_like_base(ctx, *left).map(|base| (base, *right)),
        Expr::Add(left, right) => match (ctx.get(*left), ctx.get(*right)) {
            (_, Expr::Neg(negated)) => {
                extract_sqrt_like_base(ctx, *left).map(|base| (base, *negated))
            }
            (Expr::Neg(negated), _) => {
                extract_sqrt_like_base(ctx, *right).map(|base| (base, *negated))
            }
            _ => None,
        },
        _ => None,
    }
}

fn condition_exprs_equivalent_after_sqrt_half_power_normalization(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    let left_norm = rewrite_sqrt_calls_as_half_powers_for_condition_match(ctx, left);
    let right_norm = rewrite_sqrt_calls_as_half_powers_for_condition_match(ctx, right);
    if left_norm == left && right_norm == right {
        return false;
    }

    if exprs_equivalent(ctx, left_norm, right_norm) {
        return true;
    }

    let left_display_norm = normalize_condition_expr_preserve_sign(ctx, left_norm);
    let right_display_norm = normalize_condition_expr_preserve_sign(ctx, right_norm);
    exprs_equivalent(ctx, left_display_norm, right_display_norm)
}

fn rewrite_sqrt_calls_as_half_powers_for_condition_match(
    ctx: &mut Context,
    expr: ExprId,
) -> ExprId {
    let rewritten = match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let new_left = rewrite_sqrt_calls_as_half_powers_for_condition_match(ctx, left);
            let new_right = rewrite_sqrt_calls_as_half_powers_for_condition_match(ctx, right);
            if new_left == left && new_right == right {
                return expr;
            }
            Expr::Add(new_left, new_right)
        }
        Expr::Sub(left, right) => {
            let new_left = rewrite_sqrt_calls_as_half_powers_for_condition_match(ctx, left);
            let new_right = rewrite_sqrt_calls_as_half_powers_for_condition_match(ctx, right);
            if new_left == left && new_right == right {
                return expr;
            }
            Expr::Sub(new_left, new_right)
        }
        Expr::Mul(left, right) => {
            let new_left = rewrite_sqrt_calls_as_half_powers_for_condition_match(ctx, left);
            let new_right = rewrite_sqrt_calls_as_half_powers_for_condition_match(ctx, right);
            if new_left == left && new_right == right {
                return expr;
            }
            Expr::Mul(new_left, new_right)
        }
        Expr::Div(left, right) => {
            let new_left = rewrite_sqrt_calls_as_half_powers_for_condition_match(ctx, left);
            let new_right = rewrite_sqrt_calls_as_half_powers_for_condition_match(ctx, right);
            if new_left == left && new_right == right {
                return expr;
            }
            Expr::Div(new_left, new_right)
        }
        Expr::Pow(base, exp) => {
            let new_base = rewrite_sqrt_calls_as_half_powers_for_condition_match(ctx, base);
            let rewritten_exp = rewrite_sqrt_calls_as_half_powers_for_condition_match(ctx, exp);
            let new_exp = as_rational_const(ctx, rewritten_exp)
                .map(|value| ctx.add(Expr::Number(value)))
                .unwrap_or(rewritten_exp);
            if new_base == base && new_exp == exp {
                return expr;
            }
            Expr::Pow(new_base, new_exp)
        }
        Expr::Neg(inner) => {
            let new_inner = rewrite_sqrt_calls_as_half_powers_for_condition_match(ctx, inner);
            if new_inner == inner {
                return expr;
            }
            Expr::Neg(new_inner)
        }
        Expr::Hold(inner) => {
            let new_inner = rewrite_sqrt_calls_as_half_powers_for_condition_match(ctx, inner);
            if new_inner == inner {
                return expr;
            }
            Expr::Hold(new_inner)
        }
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Sqrt) =>
        {
            let new_arg = rewrite_sqrt_calls_as_half_powers_for_condition_match(ctx, args[0]);
            let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
            Expr::Pow(new_arg, half)
        }
        Expr::Function(fn_id, args) => {
            let mut changed = false;
            let mut new_args = Vec::with_capacity(args.len());
            for arg in args {
                let new_arg = rewrite_sqrt_calls_as_half_powers_for_condition_match(ctx, arg);
                changed |= new_arg != arg;
                new_args.push(new_arg);
            }
            if !changed {
                return expr;
            }
            Expr::Function(fn_id, new_args)
        }
        Expr::Matrix { rows, cols, data } => {
            let mut changed = false;
            let mut new_data = Vec::with_capacity(data.len());
            for item in data {
                let new_item = rewrite_sqrt_calls_as_half_powers_for_condition_match(ctx, item);
                changed |= new_item != item;
                new_data.push(new_item);
            }
            if !changed {
                return expr;
            }
            Expr::Matrix {
                rows,
                cols,
                data: new_data,
            }
        }
        _ => return expr,
    };

    ctx.add(rewritten)
}

fn positive_even_power_gap_is_dominated_by_surd_lower_gap(
    ctx: &mut Context,
    product_expr: ExprId,
    known_positive_exprs: &[ExprId],
) -> bool {
    for known_positive in known_positive_exprs {
        let Some((base, radicand)) = surd_lower_gap_parts(ctx, *known_positive) else {
            continue;
        };

        let two = ctx.num(2);
        let base_squared = ctx.add(Expr::Pow(base, two));
        let candidate_gap = ctx.add(Expr::Sub(base_squared, radicand));
        if exprs_equivalent(ctx, product_expr, candidate_gap) {
            return true;
        }

        let normalized_product = normalize_condition_expr_preserve_sign(ctx, product_expr);
        let normalized_candidate = normalize_condition_expr_preserve_sign(ctx, candidate_gap);
        if exprs_equivalent(ctx, normalized_product, normalized_candidate) {
            return true;
        }
    }

    false
}

fn surd_lower_gap_parts(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr) {
        Expr::Sub(base, rhs) => {
            positive_rational_sqrt_radicand(ctx, *rhs).map(|radicand| (*base, radicand))
        }
        Expr::Add(left, right) => match (ctx.get(*left), ctx.get(*right)) {
            (Expr::Neg(negated), _) => {
                positive_rational_sqrt_radicand(ctx, *negated).map(|radicand| (*right, radicand))
            }
            (_, Expr::Neg(negated)) => {
                positive_rational_sqrt_radicand(ctx, *negated).map(|radicand| (*left, radicand))
            }
            _ => None,
        },
        _ => None,
    }
}

fn positive_rational_sqrt_radicand(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let radicand = extract_sqrt_like_base(ctx, expr)?;
    as_rational_const(ctx, radicand)
        .is_some_and(|value| value.is_positive())
        .then_some(radicand)
}

fn positive_polynomial_product_is_dominated_by_known_factor_and_intrinsic_remainder(
    ctx: &mut Context,
    product_expr: ExprId,
    known_positive_exprs: &[ExprId],
) -> bool {
    use cas_math::multipoly::{multipoly_from_expr, multipoly_to_expr, PolyBudget};

    let budget = PolyBudget {
        max_terms: 64,
        max_total_degree: 8,
        max_pow_exp: 4,
    };

    let Ok(product_poly) = multipoly_from_expr(ctx, product_expr, &budget) else {
        return false;
    };
    if product_poly.is_zero() || product_poly.is_constant() {
        return false;
    }

    for known_positive in known_positive_exprs {
        let Ok(known_poly) = multipoly_from_expr(ctx, *known_positive, &budget) else {
            continue;
        };
        if known_poly.is_zero()
            || known_poly.is_constant()
            || !known_poly.vars.iter().all(|var| {
                product_poly
                    .vars
                    .iter()
                    .any(|product_var| product_var == var)
            })
        {
            continue;
        }

        let aligned_known = known_poly.align_vars(&product_poly.vars);
        let Some(quotient) = product_poly.div_exact(&aligned_known) else {
            continue;
        };
        if quotient == product_poly || quotient.is_zero() {
            continue;
        }

        let quotient_expr = multipoly_to_expr(&quotient, ctx);
        if is_intrinsically_positive_real(ctx, quotient_expr)
            || positive_quadratic_power_gap_is_intrinsic(ctx, quotient_expr)
        {
            return true;
        }
    }

    false
}

fn positive_additive_product_is_dominated_by_known_factor_and_intrinsic_remainder(
    ctx: &mut Context,
    expr: ExprId,
    known_positive_exprs: &[ExprId],
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return false;
    }

    for known_positive in known_positive_exprs {
        let mut quotient_terms = Vec::with_capacity(view.terms.len());
        let mut saw_known_factor = false;
        for (term, sign) in view.terms.iter().copied() {
            let Some(quotient_term) =
                strip_known_positive_factor_from_product(ctx, term, *known_positive)
            else {
                quotient_terms.clear();
                break;
            };
            saw_known_factor = true;
            quotient_terms.push(match sign {
                cas_math::expr_nary::Sign::Pos => quotient_term,
                cas_math::expr_nary::Sign::Neg => ctx.add(Expr::Neg(quotient_term)),
            });
        }

        if !saw_known_factor || quotient_terms.is_empty() {
            continue;
        }

        let quotient = build_balanced_add(ctx, &quotient_terms);
        if is_intrinsically_positive_real(ctx, quotient)
            || positive_quadratic_power_gap_is_intrinsic(ctx, quotient)
        {
            return true;
        }
    }

    false
}

fn strip_known_positive_factor_from_product(
    ctx: &mut Context,
    product: ExprId,
    known_positive: ExprId,
) -> Option<ExprId> {
    if exprs_equivalent(ctx, product, known_positive) {
        return Some(ctx.add(Expr::Number(BigRational::one())));
    }

    let factors = mul_leaves(ctx, product);
    for (factor_index, factor) in factors.iter().copied().enumerate() {
        if !exprs_equivalent(ctx, factor, known_positive) {
            continue;
        }

        let remaining: Vec<_> = factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, factor)| (index != factor_index).then_some(factor))
            .collect();
        return Some(build_balanced_mul(ctx, &remaining));
    }

    None
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
    fn unit_nonnegative_ratio_gap_is_intrinsic_for_positive_constant_gap() {
        let mut ctx = Context::new();
        let expr = parse("1 - x^2/(x^2 + 3)", &mut ctx).expect("parse expression");
        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::NonNegative(expr)]);

        assert!(
            normalized.is_empty(),
            "1 - n^2/(n^2+c) is intrinsically nonnegative for c > 0"
        );
    }

    #[test]
    fn scaled_sqrt_square_gap_renders_as_open_interval_condition() {
        let mut ctx = Context::new();
        let positive_x = parse("x", &mut ctx).expect("parse x");
        let nonzero_boundary = parse("x - 4", &mut ctx).expect("parse boundary");
        let branch_gap = parse("1 - (1/2*x^(1/2))^2", &mut ctx).expect("parse scaled sqrt gap");

        let rendered = render_conditions_normalized(
            &mut ctx,
            &[
                ImplicitCondition::Positive(positive_x),
                ImplicitCondition::NonZero(nonzero_boundary),
                ImplicitCondition::Positive(branch_gap),
            ],
        );

        assert_eq!(rendered, vec!["x > 0", "x < 4"]);

        let mut ctx = Context::new();
        let scaled_branch_gap =
            parse("1 - (2*sqrt(x))^2", &mut ctx).expect("parse expanded scaled sqrt gap");

        let rendered = render_conditions_normalized(
            &mut ctx,
            &[ImplicitCondition::Positive(scaled_branch_gap)],
        );

        assert_eq!(rendered, vec!["x < 1/4"]);
    }

    #[test]
    fn reciprocal_surd_lower_bound_is_dominated_by_matching_positive_gap() {
        let mut ctx = Context::new();
        let bounded = parse("sqrt(5)*(x^2+x)/5", &mut ctx).expect("parse bounded expression");
        let gap = parse("x^2+x-sqrt(5)", &mut ctx).expect("parse positive gap");

        let rendered = render_conditions_normalized(
            &mut ctx,
            &[
                ImplicitCondition::LowerBound(bounded, BigRational::one()),
                ImplicitCondition::Positive(gap),
            ],
        );

        assert_eq!(rendered, vec!["x^2 + x - sqrt(5) > 0"]);

        let bounded = parse("(x^2+x)/sqrt(5)", &mut ctx).expect("parse bounded expression");
        let gap = parse("x^2+x-sqrt(5)", &mut ctx).expect("parse positive gap");

        let rendered = render_conditions_normalized(
            &mut ctx,
            &[
                ImplicitCondition::LowerBound(bounded, BigRational::one()),
                ImplicitCondition::Positive(gap),
            ],
        );

        assert_eq!(rendered, vec!["x^2 + x - sqrt(5) > 0"]);
    }

    #[test]
    fn lower_bound_dominates_matching_nonnegative_condition() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("parse x");

        let rendered = render_conditions_normalized(
            &mut ctx,
            &[
                ImplicitCondition::LowerBound(x, BigRational::one()),
                ImplicitCondition::NonNegative(x),
            ],
        );

        assert_eq!(rendered, vec!["x ≥ 1"]);
    }

    #[test]
    fn acosh_log_argument_positive_is_dominated_by_branch_gap() {
        let mut ctx = Context::new();
        let half_power_arg = parse("(x^2-1)^(1/2)+x", &mut ctx).expect("parse half-power log arg");
        let sqrt_arg = parse("sqrt(x^2-1)+x", &mut ctx).expect("parse sqrt log arg");
        let branch_gap = parse("x-1", &mut ctx).expect("parse branch gap");

        let rendered = render_conditions_normalized(
            &mut ctx,
            &[
                ImplicitCondition::Positive(half_power_arg),
                ImplicitCondition::Positive(sqrt_arg),
                ImplicitCondition::Positive(branch_gap),
            ],
        );

        assert_eq!(rendered, vec!["x > 1"]);

        let mut ctx = Context::new();
        let half_power_arg =
            parse("(4*x^2+4*x-3)^(1/2)+2*x+1", &mut ctx).expect("parse affine half-power log arg");
        let sqrt_arg =
            parse("sqrt((2*x+1)^2-4)+2*x+1", &mut ctx).expect("parse affine sqrt log arg");
        let sqrt_gap = parse("sqrt((2*x+1)^2-4)", &mut ctx).expect("parse affine sqrt gap");
        let branch_gap = parse("2*x-1", &mut ctx).expect("parse affine branch gap");

        let rendered = render_conditions_normalized(
            &mut ctx,
            &[
                ImplicitCondition::Positive(half_power_arg),
                ImplicitCondition::Positive(sqrt_arg),
                ImplicitCondition::NonZero(sqrt_gap),
                ImplicitCondition::Positive(branch_gap),
            ],
        );

        assert_eq!(rendered, vec!["x > 1/2"]);

        let mut ctx = Context::new();
        let sqrt_gap = parse("sqrt((x^2+x+1)^2-4)", &mut ctx).expect("parse quadratic sqrt gap");
        let branch_gap = parse("x^2+x-1", &mut ctx).expect("parse quadratic branch gap");

        let rendered = render_conditions_normalized(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(sqrt_gap),
                ImplicitCondition::Positive(branch_gap),
            ],
        );

        assert_eq!(
            rendered,
            vec!["x < -1/2 - sqrt(5)/2 or x > -1/2 + sqrt(5)/2"]
        );

        let mut ctx = Context::new();
        let sqrt_arg =
            parse("sqrt((2*x+1)^2-4)-(2*x+1)", &mut ctx).expect("parse negative sqrt log arg");
        let sqrt_gap = parse("sqrt((2*x+1)^2-4)", &mut ctx).expect("parse negative sqrt gap");
        let radicand_gap = parse("(2*x+1)^2-4", &mut ctx).expect("parse negative radicand gap");
        let branch_gap = parse("-2*x-3", &mut ctx).expect("parse negative branch gap");

        let rendered = render_conditions_normalized(
            &mut ctx,
            &[
                ImplicitCondition::Positive(sqrt_arg),
                ImplicitCondition::NonZero(sqrt_gap),
                ImplicitCondition::NonNegative(radicand_gap),
                ImplicitCondition::Positive(branch_gap),
            ],
        );

        assert_eq!(rendered, vec!["x < -3/2"]);
    }

    #[test]
    fn positive_sqrt_lower_gap_dominates_positive_product_condition() {
        let mut ctx = Context::new();
        let branch_gap = parse("x-2", &mut ctx).expect("parse branch gap");
        let sqrt_product = parse("1/2*x*(sqrt(x/2)-1)", &mut ctx).expect("parse sqrt product");
        let half_power_product =
            parse("1/2*x*((1/2*x)^(1/2)-1)", &mut ctx).expect("parse half-power product");

        let rendered = render_conditions_normalized(
            &mut ctx,
            &[
                ImplicitCondition::Positive(branch_gap),
                ImplicitCondition::Positive(sqrt_product),
                ImplicitCondition::Positive(half_power_product),
            ],
        );

        assert_eq!(rendered, vec!["x > 2"]);
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
    fn nonzero_distributive_trig_denominators_dedupe_for_display() {
        let mut ctx = Context::new();
        let compact =
            parse("2*x*cos(x)+(x^2-2)*sin(x)+c", &mut ctx).expect("parse compact denominator");
        let expanded = parse("2*x*cos(x)+sin(x)*x^2+c-2*sin(x)", &mut ctx).expect("parse expanded");
        let reordered = parse("sin(x)*(x^2-2)+2*x*cos(x)+c", &mut ctx).expect("parse reordered");

        let rendered = render_conditions_normalized(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(compact),
                ImplicitCondition::NonZero(expanded),
                ImplicitCondition::NonZero(reordered),
            ],
        );

        assert_eq!(
            rendered,
            vec!["sin(x) * (x^2 - 2) + 2 * x * cos(x) + c ≠ 0"]
        );
    }

    #[test]
    fn nonzero_supported_integral_denominator_dedupes_against_antiderivative_condition() {
        let mut ctx = Context::new();
        let antiderivative =
            parse("2*x*cos(x)+(x^2-2)*sin(x)+c", &mut ctx).expect("parse antiderivative");
        let original = parse("integrate(x^2*cos(x),x)+c", &mut ctx).expect("parse original");

        let rendered = render_conditions_normalized(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(antiderivative),
                ImplicitCondition::NonZero(original),
            ],
        );

        assert_eq!(
            rendered,
            vec!["sin(x) * (x^2 - 2) + 2 * x * cos(x) + c ≠ 0"]
        );
    }

    #[test]
    fn nonzero_supported_integral_denominator_without_peer_stays_visible() {
        let mut ctx = Context::new();
        let original = parse("integrate(x^2*cos(x),x)+c", &mut ctx).expect("parse original");

        let rendered =
            render_conditions_normalized(&mut ctx, &[ImplicitCondition::NonZero(original)]);

        assert_eq!(rendered, vec!["integrate(cos(x) * x^2, x) + c ≠ 0"]);
    }

    #[test]
    fn nonzero_diff_integral_residual_passthrough_denominator_compacts_for_display() {
        let mut ctx = Context::new();
        let denominator = parse(
            "diff(integrate(1/(sqrt(2*x)*sqrt(2*x+6)),x),x)-1/(sqrt(2*x)*sqrt(2*x+6))+x+2",
            &mut ctx,
        )
        .expect("parse denominator");
        let positive_x = parse("x", &mut ctx).expect("parse x");

        let rendered = render_conditions_normalized(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(denominator),
                ImplicitCondition::Positive(positive_x),
            ],
        );

        assert_eq!(rendered, vec!["x > 0"]);
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
    fn positive_affine_factors_dominate_expanded_product_condition() {
        let mut ctx = Context::new();
        let left_gap = parse("1 - x", &mut ctx).expect("parse left gap");
        let right_gap = parse("2*x - 1", &mut ctx).expect("parse right gap");
        let expanded_product = parse("3*x - 2*x^2 - 1", &mut ctx).expect("parse product");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(expanded_product),
                ImplicitCondition::Positive(left_gap),
                ImplicitCondition::Positive(right_gap),
            ],
        );

        assert_eq!(normalized.len(), 2);
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(left_gap))
        }));
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(right_gap))
        }));
    }

    #[test]
    fn positive_known_factor_dominates_product_with_intrinsic_positive_factor() {
        let mut ctx = Context::new();
        let asinh_arg = parse("asinh(2*x+1)", &mut ctx).expect("parse positive factor");
        let product = parse(
            "2*asinh(2*x+1) + 4*x*asinh(2*x+1) + 4*asinh(2*x+1)*x^2",
            &mut ctx,
        )
        .expect("parse expanded product");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(product),
                ImplicitCondition::Positive(asinh_arg),
            ],
        );

        assert_eq!(
            normalized,
            vec![ImplicitCondition::Positive(asinh_arg)],
            "got: {:?}",
            normalized
                .iter()
                .map(|condition| condition.display(&ctx))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn positive_variable_dominates_positive_product_plus_offset_nonzero_condition() {
        let mut ctx = Context::new();
        let positive_var = parse("x", &mut ctx).expect("parse positive variable");
        let denominator =
            parse("x^5 + 2*x^3 + x + 1", &mut ctx).expect("parse positive product plus offset");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(positive_var),
                ImplicitCondition::NonZero(denominator),
            ],
        );

        assert_eq!(
            normalized,
            vec![ImplicitCondition::Positive(positive_var)],
            "got: {:?}",
            normalized
                .iter()
                .map(|condition| condition.display(&ctx))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn positive_polynomial_known_factor_dominates_product_with_positive_quotient() {
        let mut ctx = Context::new();
        let known_factor = parse("x^2 + x - 2", &mut ctx).expect("parse known factor");
        let product = parse("x^4 + 2*x^3 + x^2 - 4", &mut ctx).expect("parse polynomial product");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(known_factor),
                ImplicitCondition::Positive(product),
            ],
        );

        assert_eq!(
            normalized.len(),
            1,
            "got: {:?}",
            normalized
                .iter()
                .map(|condition| condition.display(&ctx))
                .collect::<Vec<_>>()
        );
        assert!(conditions_equivalent(
            &ctx,
            &normalized[0],
            &ImplicitCondition::Positive(known_factor)
        ));
    }

    #[test]
    fn positive_surd_lower_gap_dominates_matching_even_power_gap() {
        let mut ctx = Context::new();
        let known_factor = parse("x^2 + x - sqrt(5)", &mut ctx).expect("parse known factor");
        let product =
            parse("x^4 + 2*x^3 + x^2 - 5", &mut ctx).expect("parse matching even power gap");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(known_factor),
                ImplicitCondition::Positive(product),
            ],
        );

        assert_eq!(
            normalized.len(),
            1,
            "got: {:?}",
            normalized
                .iter()
                .map(|condition| condition.display(&ctx))
                .collect::<Vec<_>>()
        );
        assert!(conditions_equivalent(
            &ctx,
            &normalized[0],
            &ImplicitCondition::Positive(known_factor)
        ));

        let nonmatching_product =
            parse("x^4 + 2*x^3 + x^2 - 6", &mut ctx).expect("parse nonmatching even power gap");
        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(known_factor),
                ImplicitCondition::Positive(nonmatching_product),
            ],
        );

        assert_eq!(normalized.len(), 2);
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
    fn positive_unary_log_dominates_argument_boundary_nonzero() {
        let mut ctx = Context::new();
        let ln_x = parse("ln(x)", &mut ctx).expect("parse ln(x)");
        let x = parse("x", &mut ctx).expect("parse x");
        let x_minus_one = parse("x - 1", &mut ctx).expect("parse x - 1");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(x_minus_one),
                ImplicitCondition::Positive(ln_x),
                ImplicitCondition::Positive(x),
            ],
        );

        assert_eq!(normalized.len(), 2, "got: {normalized:?}");
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(ln_x)) }));
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(x)) }));
        assert!(
            !normalized.iter().any(|cond| {
                conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x_minus_one))
            }),
            "ln(x) > 0 implies x > 1, so x - 1 != 0 is redundant: {normalized:?}"
        );
    }

    #[test]
    fn positive_unary_log_dominates_opposite_oriented_argument_boundary_nonzero() {
        let mut ctx = Context::new();
        let ln_x = parse("ln(x)", &mut ctx).expect("parse ln(x)");
        let one_minus_x = parse("1 - x", &mut ctx).expect("parse 1 - x");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(one_minus_x),
                ImplicitCondition::Positive(ln_x),
            ],
        );

        assert_eq!(normalized.len(), 1, "got: {normalized:?}");
        assert!(conditions_equivalent(
            &ctx,
            &normalized[0],
            &ImplicitCondition::Positive(ln_x)
        ));
    }

    #[test]
    fn positive_unknown_base_log_keeps_argument_boundary_nonzero() {
        let mut ctx = Context::new();
        let log_b_x = parse("log(b, x)", &mut ctx).expect("parse log(b, x)");
        let x_minus_one = parse("x - 1", &mut ctx).expect("parse x - 1");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(x_minus_one),
                ImplicitCondition::Positive(log_b_x),
            ],
        );

        assert_eq!(normalized.len(), 2, "got: {normalized:?}");
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::NonZero(x_minus_one))
        }));
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(log_b_x))
        }));
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
    fn nonzero_tanh_display_condition_normalizes_argument_before_sinh_dedupe() {
        let mut ctx = Context::new();
        let tanh_expr = parse("tanh(x^2 + 0)", &mut ctx).expect("parse tanh");
        let sinh_expr = parse("sinh(x^2)", &mut ctx).expect("parse sinh");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(sinh_expr),
                ImplicitCondition::NonZero(tanh_expr),
            ],
        );
        let rendered: Vec<_> = normalized
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect();

        assert_eq!(rendered, vec!["sinh(x^2) ≠ 0"]);
    }

    #[test]
    fn nonzero_sinh_display_condition_normalizes_sqrt_argument_before_dedupe() {
        let mut ctx = Context::new();
        let source = parse("sinh(sqrt(x + 0))", &mut ctx).expect("parse source sinh");
        let target = parse("sinh(sqrt(x))", &mut ctx).expect("parse target sinh");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(source),
                ImplicitCondition::NonZero(target),
            ],
        );
        let rendered: Vec<_> = normalized
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect();

        assert_eq!(rendered, vec!["sinh(sqrt(x)) ≠ 0"]);
    }

    #[test]
    fn nonzero_sinh_display_condition_preserves_sqrt_radicand_orientation() {
        let mut ctx = Context::new();
        let source = parse("sinh(sqrt(1 - x + 0))", &mut ctx).expect("parse source sinh");

        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::NonZero(source)]);
        let rendered: Vec<_> = normalized
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect();

        assert_eq!(rendered, vec!["sinh(sqrt(1 - x)) ≠ 0"]);
    }

    #[test]
    fn nonzero_same_trig_display_conditions_dedupe_equivalent_sqrt_argument() {
        let mut ctx = Context::new();
        let source = parse("cos(sqrt(x + 0))", &mut ctx).expect("parse source cos");
        let target = parse("cos(sqrt(x))", &mut ctx).expect("parse target cos");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(source),
                ImplicitCondition::NonZero(target),
            ],
        );
        let rendered: Vec<_> = normalized
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect();

        assert_eq!(rendered, vec!["cos(sqrt(x)) ≠ 0"]);
    }

    #[test]
    fn nonzero_same_trig_display_condition_preserves_standalone_source_argument() {
        let mut ctx = Context::new();
        let source = parse("cos(sqrt(x + 0))", &mut ctx).expect("parse source cos");

        let normalized =
            normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::NonZero(source)]);
        let rendered: Vec<_> = normalized
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect();

        assert_eq!(rendered, vec!["cos(sqrt(x + 0)) ≠ 0"]);
    }

    #[test]
    fn nonzero_same_trig_display_equivalence_preserves_sqrt_radicand_orientation() {
        let mut ctx = Context::new();
        let source = parse("cos(sqrt(1 - x + 0))", &mut ctx).expect("parse source cos");
        let opposite = parse("cos(sqrt(x - 1))", &mut ctx).expect("parse opposite cos");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(source),
                ImplicitCondition::NonZero(opposite),
            ],
        );
        let rendered: Vec<_> = normalized
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect();

        assert!(
            !conditions_equivalent(
                &ctx,
                &ImplicitCondition::NonZero(source),
                &ImplicitCondition::NonZero(opposite)
            ),
            "opposite sqrt radicands must not be deduped: {rendered:?}"
        );
        assert_eq!(normalized.len(), 2, "got: {rendered:?}");
    }

    #[test]
    fn positive_sinh_condition_dominates_matching_tanh_nonzero_boundary() {
        let mut ctx = Context::new();
        let tanh_expr = parse("tanh(sqrt(x))", &mut ctx).expect("parse tanh");
        let sinh_expr = parse("sinh(sqrt(x))", &mut ctx).expect("parse sinh");
        let x = parse("x", &mut ctx).expect("parse x");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(sinh_expr),
                ImplicitCondition::NonZero(tanh_expr),
                ImplicitCondition::Positive(x),
            ],
        );

        assert_eq!(normalized.len(), 2, "got: {normalized:?}");
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(sinh_expr))
        }));
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(x)) }));
    }

    #[test]
    fn positive_sinh_condition_dominates_matching_sinh_nonzero_boundary() {
        let mut ctx = Context::new();
        let sinh_expr = parse("sinh(sqrt(x))", &mut ctx).expect("parse sinh");
        let x = parse("x", &mut ctx).expect("parse x");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(sinh_expr),
                ImplicitCondition::NonZero(sinh_expr),
                ImplicitCondition::Positive(x),
            ],
        );

        assert_eq!(normalized.len(), 2, "got: {normalized:?}");
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(sinh_expr))
        }));
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(x)) }));
    }

    #[test]
    fn positive_reciprocal_sinh_condition_dominates_product_denominator_nonzero_boundary() {
        let mut ctx = Context::new();
        let reciprocal_sinh = parse("1/sinh(sqrt(x))", &mut ctx).expect("parse reciprocal sinh");
        let denominator = parse("2*tanh(sqrt(x))*sqrt(x)", &mut ctx).expect("parse denominator");
        let x = parse("x", &mut ctx).expect("parse x");
        let sinh_expr = parse("sinh(sqrt(x))", &mut ctx).expect("parse sinh");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(reciprocal_sinh),
                ImplicitCondition::NonZero(denominator),
                ImplicitCondition::Positive(x),
            ],
        );

        assert_eq!(normalized.len(), 2, "got: {normalized:?}");
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(sinh_expr))
        }));
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(x)) }));
    }

    #[test]
    fn positive_sec_condition_normalizes_to_positive_cos_boundary() {
        let mut ctx = Context::new();
        let sec_expr = parse("sec(sqrt(x))", &mut ctx).expect("parse sec");
        let cos_expr = parse("cos(sqrt(x))", &mut ctx).expect("parse cos");
        let x = parse("x", &mut ctx).expect("parse x");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(sec_expr),
                ImplicitCondition::Positive(cos_expr),
                ImplicitCondition::Positive(x),
            ],
        );

        assert_eq!(normalized.len(), 2, "got: {normalized:?}");
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(cos_expr))
        }));
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(x)) }));
    }

    #[test]
    fn positive_csc_condition_normalizes_to_positive_sin_boundary() {
        let mut ctx = Context::new();
        let csc_expr = parse("csc(sqrt(x))", &mut ctx).expect("parse csc");
        let sin_expr = parse("sin(sqrt(x))", &mut ctx).expect("parse sin");
        let x = parse("x", &mut ctx).expect("parse x");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(csc_expr),
                ImplicitCondition::Positive(sin_expr),
                ImplicitCondition::Positive(x),
            ],
        );

        assert_eq!(normalized.len(), 2, "got: {normalized:?}");
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(sin_expr))
        }));
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(x)) }));
    }

    #[test]
    fn positive_tan_condition_normalizes_to_positive_sin_cos_quotient() {
        let mut ctx = Context::new();
        let tan_expr = parse("tan(sqrt(x))", &mut ctx).expect("parse tan");
        let quotient = parse("sin(sqrt(x))/cos(sqrt(x))", &mut ctx).expect("parse quotient");
        let double_angle = parse("sin(2*sqrt(x))", &mut ctx).expect("parse double angle");
        let x = parse("x", &mut ctx).expect("parse x");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(tan_expr),
                ImplicitCondition::Positive(quotient),
                ImplicitCondition::NonZero(double_angle),
                ImplicitCondition::Positive(x),
            ],
        );

        assert_eq!(normalized.len(), 2, "got: {normalized:?}");
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(quotient))
        }));
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(x)) }));
    }

    #[test]
    fn positive_cot_condition_normalizes_to_positive_cos_sin_quotient() {
        let mut ctx = Context::new();
        let cot_expr = parse("cot(sqrt(x))", &mut ctx).expect("parse cot");
        let quotient = parse("cos(sqrt(x))/sin(sqrt(x))", &mut ctx).expect("parse quotient");
        let double_angle = parse("sin(2*sqrt(x))", &mut ctx).expect("parse double angle");
        let x = parse("x", &mut ctx).expect("parse x");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(cot_expr),
                ImplicitCondition::Positive(quotient),
                ImplicitCondition::NonZero(double_angle),
                ImplicitCondition::Positive(x),
            ],
        );

        assert_eq!(normalized.len(), 2, "got: {normalized:?}");
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(quotient))
        }));
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(x)) }));
    }

    #[test]
    fn positive_trig_quotient_dominates_atomic_nonzero_boundaries() {
        let mut ctx = Context::new();
        let quotient =
            parse("sin(sqrt(3*x+1))/cos(sqrt(3*x+1))", &mut ctx).expect("parse quotient");
        let sin_boundary = parse("sin(sqrt(3*x+1))", &mut ctx).expect("parse sin boundary");
        let cos_boundary = parse("cos(sqrt(3*x+1))", &mut ctx).expect("parse cos boundary");
        let double_angle = parse("sin(2*(3*x+1)^(1/2))", &mut ctx).expect("parse double angle");
        let radicand = parse("3*x+1", &mut ctx).expect("parse radicand");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(sin_boundary),
                ImplicitCondition::NonZero(cos_boundary),
                ImplicitCondition::NonZero(double_angle),
                ImplicitCondition::Positive(quotient),
                ImplicitCondition::Positive(radicand),
            ],
        );

        assert_eq!(normalized.len(), 2, "got: {normalized:?}");
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(quotient))
        }));
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(radicand))
        }));
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
    fn positive_fractional_power_sum_under_positive_base_drops_nonzero_condition() {
        let mut ctx = Context::new();
        let denominator = parse("x^(3/2) + 4*x^(1/2)", &mut ctx).expect("parse denominator");
        let x = parse("x", &mut ctx).expect("parse x");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(x),
                ImplicitCondition::NonZero(denominator),
            ],
        );

        assert_eq!(normalized, vec![ImplicitCondition::Positive(x)]);
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
    fn positive_sqrt_polynomial_gap_sum_condition_is_dropped() {
        let mut ctx = Context::new();
        for input in [
            "sqrt(x^2 + 1) + x",
            "sqrt(x^2 + 1) - x",
            "sqrt(x^4 + 1) + x^2",
            "sqrt(x^4 + 1) - x^2",
            "sqrt((2*x+1)^2 + 4) + 2*x + 1",
            "sqrt((2*x+1)^2 + 4) - (2*x + 1)",
        ] {
            let expr = parse(input, &mut ctx).expect("parse expr");

            let normalized =
                normalize_and_dedupe_conditions(&mut ctx, &[ImplicitCondition::Positive(expr)]);

            assert!(
                normalized.is_empty(),
                "positive sqrt-polynomial gap should be intrinsic for {input}"
            );
        }
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
    fn positive_acosh_and_gap_dominate_split_radical_product_nonnegative() {
        let mut ctx = Context::new();
        let acosh_x = parse("acosh(x)", &mut ctx).expect("parse acosh(x)");
        let gap = parse("x - 1", &mut ctx).expect("parse x - 1");
        let product = parse("acosh(x)*(x^2 - 1)", &mut ctx).expect("parse acosh product gap");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(gap),
                ImplicitCondition::Positive(acosh_x),
                ImplicitCondition::NonNegative(product),
            ],
        );

        assert_eq!(normalized.len(), 2);
        assert!(normalized
            .iter()
            .any(|cond| { conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(gap)) }));
        assert!(normalized.iter().any(|cond| {
            conditions_equivalent(&ctx, cond, &ImplicitCondition::Positive(acosh_x))
        }));
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
    fn compact_shifted_unit_sqrt_gap_dominates_raw_inverse_trig_open_interval_guard() {
        let mut ctx = Context::new();
        let raw_open_interval =
            parse("1 - (2*sqrt(2*x)-1)^2", &mut ctx).expect("parse raw open interval");
        let radicand = parse("2*x", &mut ctx).expect("parse radicand");
        let compact_gap = parse("sqrt(2*x)-2*x", &mut ctx).expect("parse compact shifted sqrt gap");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(raw_open_interval),
                ImplicitCondition::Positive(radicand),
                ImplicitCondition::Positive(compact_gap),
            ],
        );
        let rendered: Vec<String> = normalized
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect();

        assert_eq!(
            rendered,
            vec!["x > 0".to_string(), "sqrt(2 * x) - 2 * x > 0".to_string()]
        );
    }

    #[test]
    fn compact_shifted_unit_pow_half_gap_dominates_raw_inverse_trig_open_interval_guard() {
        let mut ctx = Context::new();
        let raw_open_interval =
            parse("1 - (2*(2*x)^(1/2)-1)^2", &mut ctx).expect("parse raw open interval");
        let radicand = parse("x", &mut ctx).expect("parse normalized radicand");
        let compact_gap = parse("sqrt(2*x)-2*x", &mut ctx).expect("parse compact shifted sqrt gap");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(raw_open_interval),
                ImplicitCondition::Positive(radicand),
                ImplicitCondition::Positive(compact_gap),
            ],
        );
        let rendered: Vec<String> = normalized
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect();

        assert_eq!(
            rendered,
            vec!["x > 0".to_string(), "sqrt(2 * x) - 2 * x > 0".to_string()]
        );
    }

    #[test]
    fn positive_affine_numerator_dominates_shifted_unit_ratio_gap() {
        let mut ctx = Context::new();
        let numerator = parse("x+1", &mut ctx).expect("parse numerator");
        let raw_open_interval =
            parse("1 - (x+1)/(x+3)", &mut ctx).expect("parse raw open interval");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(numerator),
                ImplicitCondition::Positive(raw_open_interval),
            ],
        );
        let rendered: Vec<String> = normalized
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect();

        assert_eq!(rendered, vec!["x > -1".to_string()]);
    }

    #[test]
    fn positive_affine_quotient_displays_exterior_interval_and_dominates_internal_nonzeros() {
        let mut ctx = Context::new();
        let quotient = parse("(x+1)/(x+3)", &mut ctx).expect("parse quotient");
        let quotient_nonnegative =
            parse("(x+1)/(x+3)", &mut ctx).expect("parse nonnegative quotient");
        let interior_boundary = parse("x+2", &mut ctx).expect("parse interior boundary");
        let denominator_boundary = parse("x+3", &mut ctx).expect("parse denominator boundary");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonNegative(quotient_nonnegative),
                ImplicitCondition::Positive(quotient),
                ImplicitCondition::NonZero(interior_boundary),
                ImplicitCondition::NonZero(denominator_boundary),
            ],
        );
        let rendered: Vec<String> = normalized
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect();

        assert_eq!(rendered, vec!["x < -3 or x > -1".to_string()]);
    }

    #[test]
    fn scaled_positive_affine_quotient_dominates_internal_affine_nonzero() {
        let mut ctx = Context::new();
        let quotient = parse("(2*x+1)/(x+3)", &mut ctx).expect("parse quotient");
        let quotient_nonnegative =
            parse("(2*x+1)/(x+3)", &mut ctx).expect("parse nonnegative quotient");
        let interior_boundary = parse("3*x+4", &mut ctx).expect("parse interior boundary");
        let denominator_boundary = parse("x+3", &mut ctx).expect("parse denominator boundary");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonNegative(quotient_nonnegative),
                ImplicitCondition::Positive(quotient),
                ImplicitCondition::NonZero(interior_boundary),
                ImplicitCondition::NonZero(denominator_boundary),
            ],
        );
        let rendered: Vec<String> = normalized
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect();

        assert_eq!(rendered, vec!["x < -3 or x > -1/2".to_string()]);
    }

    #[test]
    fn bounded_positive_quadratic_does_not_dominate_interior_nonzero() {
        let mut ctx = Context::new();
        let interval = parse("4-(x+1)^2", &mut ctx).expect("parse interval");
        let interior_boundary = parse("x+2", &mut ctx).expect("parse interior boundary");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(interval),
                ImplicitCondition::NonZero(interior_boundary),
            ],
        );
        let rendered: Vec<String> = normalized
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect();

        assert_eq!(
            rendered,
            vec!["-3 < x < 1".to_string(), "x ≠ -2".to_string()]
        );
    }

    #[test]
    fn inverse_trig_alias_positive_conditions_dedupe() {
        let mut ctx = Context::new();
        let short = parse("asin(2*x+1)", &mut ctx).expect("parse short alias");
        let long = parse("arcsin(2*x+1)", &mut ctx).expect("parse long alias");
        let gap = parse("-x^2-x", &mut ctx).expect("parse interval gap");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(long),
                ImplicitCondition::Positive(gap),
                ImplicitCondition::Positive(short),
            ],
        );
        let rendered: Vec<String> = normalized
            .iter()
            .map(|condition| condition.display(&ctx))
            .collect();

        assert_eq!(
            normalized
                .iter()
                .filter(|condition| matches!(condition, ImplicitCondition::Positive(expr) if inverse_trig_alias_calls_equivalent(&ctx, *expr, short) || exprs_equivalent(&ctx, *expr, short)))
                .count(),
            1,
            "expected a single asin/arcsin positivity guard, got: {rendered:?}"
        );
        assert!(
            normalized.iter().any(
                |condition| matches!(condition, ImplicitCondition::Positive(expr) if exprs_equivalent(&ctx, *expr, short))
            ),
            "expected the retained alias display to prefer asin(...), got: {rendered:?}"
        );
        assert!(normalized.iter().any(|condition| conditions_equivalent(
            &ctx,
            condition,
            &ImplicitCondition::Positive(gap)
        )));
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
    fn positive_composite_base_dominates_positive_constant_shift_nonzero() {
        let mut ctx = Context::new();
        let base = parse("tan(x)+sqrt(x)+1/sqrt(x)+x", &mut ctx).expect("parse base");
        let shifted = parse("tan(x)+sqrt(x)+1/sqrt(x)+x+1", &mut ctx).expect("parse shifted");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(base),
                ImplicitCondition::NonZero(shifted),
            ],
        );

        assert_eq!(normalized.len(), 1, "got: {normalized:?}");
        assert!(conditions_equivalent(
            &ctx,
            &normalized[0],
            &ImplicitCondition::Positive(base)
        ));
    }

    #[test]
    fn positive_composite_base_keeps_negative_constant_shift_nonzero() {
        let mut ctx = Context::new();
        let base = parse("tan(x)+sqrt(x)+1/sqrt(x)+x", &mut ctx).expect("parse base");
        let shifted = parse("tan(x)+sqrt(x)+1/sqrt(x)+x-1", &mut ctx).expect("parse shifted");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::Positive(base),
                ImplicitCondition::NonZero(shifted),
            ],
        );

        assert!(normalized.iter().any(|condition| conditions_equivalent(
            &ctx,
            condition,
            &ImplicitCondition::Positive(base)
        )));
        assert!(normalized.iter().any(|condition| conditions_equivalent(
            &ctx,
            condition,
            &ImplicitCondition::NonZero(shifted)
        )));
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

        assert_eq!(rendered, vec!["-1 < x < 0".to_string()]);
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
    fn nonzero_sqrt_is_dominated_by_positive_radicand_context() {
        let mut ctx = Context::new();
        let sqrt_x = parse("sqrt(x)", &mut ctx).expect("parse sqrt(x)");
        let sqrt_x_plus_one = parse("sqrt(x+1)", &mut ctx).expect("parse sqrt(x+1)");
        let held_sqrt_x = cas_ast::hold::wrap_hold(&mut ctx, sqrt_x);
        let held_sqrt_x_plus_one = cas_ast::hold::wrap_hold(&mut ctx, sqrt_x_plus_one);
        let x = parse("x", &mut ctx).expect("parse x");

        let normalized = normalize_and_dedupe_conditions(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(held_sqrt_x),
                ImplicitCondition::NonZero(held_sqrt_x_plus_one),
                ImplicitCondition::Positive(x),
            ],
        );

        assert_eq!(normalized, vec![ImplicitCondition::Positive(x)]);
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

    #[test]
    fn nonzero_common_log_power_sum_uses_atomic_guards_and_positive_base() {
        let mut ctx = Context::new();
        let base = parse("x^2+x-1", &mut ctx).expect("parse base");
        let composite = parse(
            "x*ln(x^2+x-1)^2 + x^2*ln(x^2+x-1)^2 - ln(x^2+x-1)^2",
            &mut ctx,
        )
        .expect("parse composite");
        let x_minus_one = parse("x - 1", &mut ctx).expect("parse x - 1");
        let x_plus_two = parse("x + 2", &mut ctx).expect("parse x + 2");

        let rendered = render_conditions_normalized(
            &mut ctx,
            &[
                ImplicitCondition::NonZero(composite),
                ImplicitCondition::NonZero(x_minus_one),
                ImplicitCondition::NonZero(x_plus_two),
                ImplicitCondition::Positive(base),
            ],
        );

        assert_eq!(
            rendered,
            vec![
                "x ≠ 1".to_string(),
                "x ≠ -2".to_string(),
                "x < -1/2 - sqrt(5)/2 or x > -1/2 + sqrt(5)/2".to_string(),
            ]
        );
    }
}
