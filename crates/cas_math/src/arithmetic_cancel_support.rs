//! Pattern helpers for arithmetic self-cancellation rewrites.

use crate::expr_nary::{mul_leaves, AddView, Sign};
use crate::numeric_eval::as_rational_const;
use crate::semantic_equality::SemanticEqualityChecker;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::cmp::Ordering;
use std::collections::BTreeMap;

fn extract_abs_sub_like_pair(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Abs) || args.len() != 1 {
        return None;
    }

    let mut scratch = ctx.clone();
    crate::expr_sub_like::extract_sub_like_pair(&mut scratch, args[0])
}

fn match_abs_sub_mirror_expr(ctx: &Context, lhs: ExprId, rhs: ExprId) -> bool {
    let Some((l1, l2)) = extract_abs_sub_like_pair(ctx, lhs) else {
        return false;
    };
    let Some((r1, r2)) = extract_abs_sub_like_pair(ctx, rhs) else {
        return false;
    };

    let checker = SemanticEqualityChecker::new(ctx);
    checker.are_equal(l1, r2) && checker.are_equal(l2, r1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArithmeticCancelRewrite {
    pub rewritten: ExprId,
    pub inner: ExprId,
}

fn is_angle_identity_arg_shape(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
}

fn builtin_trig_angle_identity_shape(ctx: &Context, expr: ExprId) -> Option<BuiltinFn> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let builtin = ctx.builtin_of(*fn_id)?;
    match builtin {
        BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Tan => {
            is_angle_identity_arg_shape(ctx, args[0]).then_some(builtin)
        }
        _ => None,
    }
}

fn should_skip_semantic_self_cancel_check(ctx: &Context, lhs: ExprId, rhs: ExprId) -> bool {
    matches!(
        (
            builtin_trig_angle_identity_shape(ctx, lhs),
            builtin_trig_angle_identity_shape(ctx, rhs),
        ),
        (Some(lhs_builtin), Some(rhs_builtin)) if lhs_builtin == rhs_builtin
    )
}

/// Match `a - a` using semantic equality.
///
/// Returns the representative inner term when both sides are semantically equal.
pub fn match_sub_self_semantic_expr(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Sub(lhs, rhs) = ctx.get(expr) else {
        return None;
    };

    if sqrt_product_term_is_one(ctx, *lhs) && is_one_expr(ctx, *rhs) {
        return Some(*lhs);
    }
    if is_one_expr(ctx, *lhs) && sqrt_product_term_is_one(ctx, *rhs) {
        return Some(*rhs);
    }
    if let (Some(lhs_coeff), Some(rhs_coeff)) = (
        scaled_sqrt_product_one_coeff(ctx, *lhs),
        rational_const_expr(ctx, *rhs),
    ) {
        if lhs_coeff == rhs_coeff {
            return Some(*lhs);
        }
    }
    if let (Some(lhs_coeff), Some(rhs_coeff)) = (
        rational_const_expr(ctx, *lhs),
        scaled_sqrt_product_one_coeff(ctx, *rhs),
    ) {
        if lhs_coeff == rhs_coeff {
            return Some(*rhs);
        }
    }
    if scaled_sqrt_product_one_term_matches_product(ctx, *lhs, *rhs) {
        return Some(*lhs);
    }
    if scaled_sqrt_product_one_term_matches_product(ctx, *rhs, *lhs) {
        return Some(*rhs);
    }

    if match_abs_sub_mirror_expr(ctx, *lhs, *rhs) {
        return Some(*lhs);
    }

    if should_skip_semantic_self_cancel_check(ctx, *lhs, *rhs) {
        return None;
    }

    if affine_sqrt_fraction_terms_equal(ctx, *lhs, *rhs) {
        return Some(*lhs);
    }

    if sqrt_product_terms_equal(ctx, *lhs, *rhs) {
        return Some(*lhs);
    }

    let checker = SemanticEqualityChecker::new(ctx);
    if checker.are_equal(*lhs, *rhs) {
        Some(*lhs)
    } else {
        None
    }
}

/// Match additive inverse patterns:
/// - `a + (-a)`
/// - `(-a) + a`
///
/// Returns `a` when matched.
pub fn match_add_inverse_expr(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Add(l, r) = ctx.get(expr) else {
        return None;
    };

    let checker = SemanticEqualityChecker::new(ctx);

    if sqrt_product_term_is_one(ctx, *l) && is_negative_one_expr(ctx, *r) {
        return Some(*l);
    }
    if sqrt_product_term_is_one(ctx, *r) && is_negative_one_expr(ctx, *l) {
        return Some(*r);
    }
    if let (Some(l_coeff), Some(r_coeff)) = (
        scaled_sqrt_product_one_coeff(ctx, *l),
        rational_const_expr(ctx, *r),
    ) {
        if l_coeff + r_coeff == BigRational::zero() {
            return Some(*l);
        }
    }
    if let (Some(l_coeff), Some(r_coeff)) = (
        rational_const_expr(ctx, *l),
        scaled_sqrt_product_one_coeff(ctx, *r),
    ) {
        if l_coeff + r_coeff == BigRational::zero() {
            return Some(*r);
        }
    }
    if scaled_sqrt_product_one_term_cancels_product(ctx, *l, *r) {
        return Some(*l);
    }
    if scaled_sqrt_product_one_term_cancels_product(ctx, *r, *l) {
        return Some(*r);
    }

    if let Expr::Neg(neg_inner) = ctx.get(*r) {
        if sqrt_product_term_is_one(ctx, *l) && is_one_expr(ctx, *neg_inner) {
            return Some(*l);
        }
        if match_abs_sub_mirror_expr(ctx, *neg_inner, *l) {
            return Some(*l);
        }
        if *neg_inner == *l {
            return Some(*l);
        }
        if !should_skip_semantic_self_cancel_check(ctx, *neg_inner, *l)
            && checker.are_equal(*neg_inner, *l)
        {
            return Some(*l);
        }
    }
    if let Expr::Neg(neg_inner) = ctx.get(*l) {
        if sqrt_product_term_is_one(ctx, *r) && is_one_expr(ctx, *neg_inner) {
            return Some(*r);
        }
        if match_abs_sub_mirror_expr(ctx, *neg_inner, *r) {
            return Some(*r);
        }
        if *neg_inner == *r {
            return Some(*r);
        }
        if !should_skip_semantic_self_cancel_check(ctx, *neg_inner, *r)
            && checker.are_equal(*neg_inner, *r)
        {
            return Some(*r);
        }
    }

    if affine_sqrt_fraction_terms_cancel(ctx, *l, *r) {
        return Some(*l);
    }

    if sqrt_product_terms_cancel(ctx, *l, *r) {
        return Some(*l);
    }

    None
}

#[derive(Debug, Clone)]
struct NormalizedAffineSqrtFractionTerm {
    coeff: BigRational,
    radicand_poly: crate::polynomial::Polynomial,
    radicand_power: BigRational,
    denominator_factors: Vec<crate::polynomial::Polynomial>,
}

fn affine_sqrt_fraction_terms_cancel(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    let Some(left) = normalize_affine_sqrt_fraction_term(ctx, left) else {
        return false;
    };
    let Some(right) = normalize_affine_sqrt_fraction_term(ctx, right) else {
        return false;
    };

    left.radicand_poly == right.radicand_poly
        && left.radicand_power == right.radicand_power
        && polynomial_factor_multisets_match(&left.denominator_factors, &right.denominator_factors)
        && (left.coeff + right.coeff).is_zero()
}

fn affine_sqrt_fraction_terms_equal(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    let Some(left) = normalize_affine_sqrt_fraction_term(ctx, left) else {
        return false;
    };
    let Some(right) = normalize_affine_sqrt_fraction_term(ctx, right) else {
        return false;
    };

    left.radicand_poly == right.radicand_poly
        && left.radicand_power == right.radicand_power
        && polynomial_factor_multisets_match(&left.denominator_factors, &right.denominator_factors)
        && left.coeff == right.coeff
}

fn normalize_affine_sqrt_fraction_term(
    ctx: &Context,
    expr: ExprId,
) -> Option<NormalizedAffineSqrtFractionTerm> {
    let (sign, expr) = strip_negated_term(ctx, expr);
    let mut coeff = sign;
    let (numerator_factors, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (
            crate::expr_nary::mul_leaves(ctx, *num)
                .into_iter()
                .collect::<Vec<_>>(),
            *den,
        ),
        Expr::Mul(_, _) => {
            let mut numerator_factors = Vec::new();
            let mut denominator = None;
            for raw_factor in crate::expr_nary::mul_leaves(ctx, expr) {
                let (factor_sign, factor) = strip_negated_term(ctx, raw_factor);
                coeff *= factor_sign;

                if let Some(value) = crate::numeric_eval::as_rational_const(ctx, factor) {
                    coeff *= value.clone();
                    continue;
                }

                match ctx.get(factor) {
                    Expr::Div(num, den) if denominator.is_none() => {
                        denominator = Some(*den);
                        numerator_factors.extend(crate::expr_nary::mul_leaves(ctx, *num));
                    }
                    Expr::Div(_, _) => return None,
                    _ => numerator_factors.push(factor),
                }
            }
            (numerator_factors, denominator?)
        }
        _ => return None,
    };

    let mut radicand = None;
    let mut radicand_poly = None;
    let mut radicand_power = BigRational::zero();

    for raw_factor in numerator_factors {
        let (factor_sign, factor) = strip_negated_term(ctx, raw_factor);
        coeff *= factor_sign;

        if let Some(value) = crate::numeric_eval::as_rational_const(ctx, factor) {
            coeff *= value.clone();
            continue;
        }

        let (candidate_radicand, power) = affine_sqrt_power_factor(ctx, factor)?;
        let candidate_poly = affine_polynomial_for_single_var(ctx, candidate_radicand)?;
        match &radicand_poly {
            Some(existing) if existing != &candidate_poly => return None,
            Some(_) => {}
            None => {
                radicand = Some(candidate_radicand);
                radicand_poly = Some(candidate_poly);
            }
        }
        radicand_power += power;
    }

    let _radicand = radicand?;
    let radicand_poly = radicand_poly?;
    let var = radicand_poly.var.clone();
    let mut denominator_factors = Vec::new();

    for factor in crate::expr_nary::mul_leaves(ctx, den) {
        if let Some(value) = crate::numeric_eval::as_rational_const(ctx, factor) {
            if value.is_zero() {
                return None;
            }
            coeff /= value.clone();
            continue;
        }

        let factor_poly = crate::polynomial::Polynomial::from_expr(ctx, factor, &var).ok()?;
        if let Some(ratio) = constant_polynomial_ratio_local(&radicand_poly, &factor_poly) {
            coeff *= ratio;
            radicand_power -= BigRational::one();
            continue;
        }

        let (scale, normalized) = normalize_affine_denominator_factor(factor_poly)?;
        if scale.is_zero() {
            return None;
        }
        coeff /= scale;
        denominator_factors.push(normalized);
    }

    if radicand_power.is_zero() {
        return None;
    }

    Some(NormalizedAffineSqrtFractionTerm {
        coeff,
        radicand_poly,
        radicand_power,
        denominator_factors,
    })
}

fn strip_negated_term(ctx: &Context, expr: ExprId) -> (BigRational, ExprId) {
    match ctx.get(expr) {
        Expr::Neg(inner) => (BigRational::from_integer((-1).into()), *inner),
        _ => (BigRational::one(), expr),
    }
}

fn affine_sqrt_power_factor(ctx: &Context, expr: ExprId) -> Option<(ExprId, BigRational)> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) =>
        {
            Some((args[0], BigRational::new(1.into(), 2.into())))
        }
        Expr::Pow(base, exp) => {
            let power = crate::numeric_eval::as_rational_const(ctx, *exp)?.clone();
            if power == BigRational::new(1.into(), 2.into())
                || power == BigRational::new((-1).into(), 2.into())
            {
                Some((*base, power))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn affine_polynomial_for_single_var(
    ctx: &Context,
    expr: ExprId,
) -> Option<crate::polynomial::Polynomial> {
    let vars = cas_ast::collect_variables(ctx, expr);
    if vars.len() != 1 {
        return None;
    }
    let var = vars.iter().next()?;
    let poly = crate::polynomial::Polynomial::from_expr(ctx, expr, var).ok()?;
    (poly.degree() == 1).then_some(poly)
}

fn normalize_affine_denominator_factor(
    poly: crate::polynomial::Polynomial,
) -> Option<(BigRational, crate::polynomial::Polynomial)> {
    if poly.degree() != 1 {
        return None;
    }
    let scale = poly.coeffs.get(1).cloned()?;
    if scale.is_zero() {
        return None;
    }
    let normalized = poly.div_scalar(&scale);
    Some((scale, normalized))
}

fn constant_polynomial_ratio_local(
    numerator: &crate::polynomial::Polynomial,
    denominator: &crate::polynomial::Polynomial,
) -> Option<BigRational> {
    if denominator.is_zero() {
        return None;
    }

    let pivot = denominator
        .coeffs
        .iter()
        .position(|coeff| !coeff.is_zero())?;
    let numerator_pivot = numerator
        .coeffs
        .get(pivot)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let scale = numerator_pivot / denominator.coeffs[pivot].clone();
    let len = numerator.coeffs.len().max(denominator.coeffs.len());

    for idx in 0..len {
        let left = numerator
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let right = denominator
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero)
            * scale.clone();
        if left != right {
            return None;
        }
    }

    Some(scale)
}

fn polynomial_factor_multisets_match(
    left: &[crate::polynomial::Polynomial],
    right: &[crate::polynomial::Polynomial],
) -> bool {
    if left.len() != right.len() {
        return false;
    }

    let mut used = vec![false; right.len()];
    'outer: for left_factor in left {
        for (idx, right_factor) in right.iter().enumerate() {
            if !used[idx] && left_factor == right_factor {
                used[idx] = true;
                continue 'outer;
            }
        }
        return false;
    }
    true
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct AffinePolyKey {
    var: String,
    coeffs: Vec<BigRational>,
}

#[derive(Debug, Clone)]
struct NormalizedSqrtProductTerm {
    sign: i8,
    squared_scale: BigRational,
    affine_powers: BTreeMap<AffinePolyKey, i32>,
    saw_root_like: bool,
}

impl Default for NormalizedSqrtProductTerm {
    fn default() -> Self {
        Self {
            sign: 1,
            squared_scale: BigRational::one(),
            affine_powers: BTreeMap::new(),
            saw_root_like: false,
        }
    }
}

fn sqrt_product_terms_equal(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    let Some(left) = normalize_sqrt_product_term(ctx, left) else {
        return false;
    };
    let Some(right) = normalize_sqrt_product_term(ctx, right) else {
        return false;
    };

    left.sign == right.sign
        && left.squared_scale == right.squared_scale
        && left.affine_powers == right.affine_powers
}

fn sqrt_product_terms_cancel(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    let Some(left) = normalize_sqrt_product_term(ctx, left) else {
        return false;
    };
    let Some(right) = normalize_sqrt_product_term(ctx, right) else {
        return false;
    };

    left.sign == -right.sign
        && left.squared_scale == right.squared_scale
        && left.affine_powers == right.affine_powers
}

fn normalize_sqrt_product_term(ctx: &Context, expr: ExprId) -> Option<NormalizedSqrtProductTerm> {
    let mut normalized = NormalizedSqrtProductTerm::default();
    accumulate_sqrt_product_factor(ctx, expr, BigRational::one(), &mut normalized)?;
    normalized.affine_powers.retain(|_, power| *power != 0);
    normalized.saw_root_like.then_some(normalized)
}

fn sqrt_product_term_is_one(ctx: &Context, expr: ExprId) -> bool {
    if sqrt_product_reciprocal_pair_is_one(ctx, expr) {
        return true;
    }
    if sqrt_split_product_reciprocal_unit_is_one(ctx, expr) {
        return true;
    }

    let Some(normalized) = normalize_sqrt_product_term(ctx, expr) else {
        return false;
    };
    normalized.sign == 1 && normalized.squared_scale.is_one() && normalized.affine_powers.is_empty()
}

fn sqrt_product_reciprocal_pair_is_one(ctx: &Context, expr: ExprId) -> bool {
    let factors = crate::expr_nary::mul_leaves(ctx, expr);
    if factors.len() != 2 {
        return false;
    }

    sqrt_product_reciprocal_pair_factors_are_one(ctx, factors[0], factors[1])
}

fn sqrt_product_reciprocal_pair_factors_are_one(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    let Some((base_a, exp_a)) = sqrt_like_base_and_exponent(ctx, left) else {
        return false;
    };
    let Some((base_b, exp_b)) = sqrt_like_base_and_exponent(ctx, right) else {
        return false;
    };

    let half = BigRational::new(1.into(), 2.into());
    let neg_half = -half.clone();
    ((exp_a == half && exp_b == neg_half) || (exp_a == neg_half && exp_b == half))
        && crate::poly_compare::poly_eq(ctx, base_a, base_b)
}

fn sqrt_split_product_reciprocal_unit_is_one(ctx: &Context, expr: ExprId) -> bool {
    let factors = crate::expr_nary::mul_leaves(ctx, expr);
    sqrt_split_product_unit_factor_indices(ctx, &factors)
        .is_some_and(|indices| indices.len() == factors.len())
}

fn sqrt_split_product_unit_factor_indices(ctx: &Context, factors: &[ExprId]) -> Option<Vec<usize>> {
    for (product_index, product_factor) in factors.iter().enumerate() {
        let Some(product_leaves) = positive_sqrt_product_leaves(ctx, *product_factor) else {
            continue;
        };
        if product_leaves.len() < 2 || product_leaves.len() > 3 {
            continue;
        }

        let checker = SemanticEqualityChecker::new(ctx);
        let mut used = vec![product_index];
        for product_leaf in product_leaves {
            let reciprocal_index =
                factors
                    .iter()
                    .enumerate()
                    .find_map(|(factor_index, factor)| {
                        if used.contains(&factor_index) {
                            return None;
                        }
                        reciprocal_sqrt_base(ctx, *factor)
                            .filter(|base| checker.are_equal(product_leaf, *base))
                            .map(|_| factor_index)
                    })?;
            used.push(reciprocal_index);
        }

        used.sort_unstable();
        return Some(used);
    }

    None
}

fn positive_sqrt_product_leaves(ctx: &Context, expr: ExprId) -> Option<Vec<ExprId>> {
    let (base, exp) = sqrt_like_base_and_exponent(ctx, expr)?;
    (exp == BigRational::new(1.into(), 2.into())).then(|| {
        crate::expr_nary::mul_leaves(ctx, base)
            .into_iter()
            .collect()
    })
}

fn reciprocal_sqrt_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let (base, exp) = sqrt_like_base_and_exponent(ctx, expr)?;
    (exp == BigRational::new((-1).into(), 2.into())).then_some(base)
}

#[derive(Debug, Clone)]
struct ProductTerm {
    coeff: BigRational,
    factors: Vec<ExprId>,
}

fn scaled_sqrt_product_one_coeff(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    let term = scaled_sqrt_product_one_term(ctx, expr)?;
    term.factors.is_empty().then_some(term.coeff)
}

fn scaled_sqrt_product_one_term_matches_product(
    ctx: &Context,
    unit_expr: ExprId,
    product_expr: ExprId,
) -> bool {
    let Some(unit_term) = scaled_sqrt_product_one_term(ctx, unit_expr) else {
        return false;
    };
    let Some(product_term) = rational_product_term(ctx, product_expr) else {
        return false;
    };

    unit_term.coeff == product_term.coeff
        && expr_factor_multisets_equal(ctx, &unit_term.factors, &product_term.factors)
}

fn scaled_sqrt_product_one_term_cancels_product(
    ctx: &Context,
    unit_expr: ExprId,
    product_expr: ExprId,
) -> bool {
    let Some(unit_term) = scaled_sqrt_product_one_term(ctx, unit_expr) else {
        return false;
    };
    let Some(product_term) = rational_product_term(ctx, product_expr) else {
        return false;
    };

    unit_term.coeff + product_term.coeff == BigRational::zero()
        && expr_factor_multisets_equal(ctx, &unit_term.factors, &product_term.factors)
}

fn scaled_sqrt_product_one_term(ctx: &Context, expr: ExprId) -> Option<ProductTerm> {
    let mut term = rational_product_term(ctx, expr)?;
    remove_one_sqrt_product_unit(ctx, &mut term.factors).then_some(term)
}

fn rational_product_term(ctx: &Context, expr: ExprId) -> Option<ProductTerm> {
    let (sign, expr) = strip_negated_term(ctx, expr);
    let mut coeff = sign;
    let mut factors = Vec::new();

    for raw_factor in crate::expr_nary::mul_leaves(ctx, expr) {
        let (factor_sign, factor) = strip_negated_term(ctx, raw_factor);
        coeff *= factor_sign;

        if let Some(value) = crate::numeric_eval::as_rational_const(ctx, factor) {
            coeff *= value.clone();
        } else {
            factors.push(factor);
        }
    }

    Some(ProductTerm { coeff, factors })
}

fn remove_one_sqrt_product_unit(ctx: &Context, factors: &mut Vec<ExprId>) -> bool {
    if let Some(indices) = sqrt_split_product_unit_factor_indices(ctx, factors) {
        for index in indices.into_iter().rev() {
            factors.remove(index);
        }
        return true;
    }

    for i in 0..factors.len() {
        for j in (i + 1)..factors.len() {
            if sqrt_product_reciprocal_pair_factors_are_one(ctx, factors[i], factors[j]) {
                factors.remove(j);
                factors.remove(i);
                return true;
            }
        }
    }

    if let Some(index) = factors
        .iter()
        .position(|factor| sqrt_product_term_is_one(ctx, *factor))
    {
        factors.remove(index);
        return true;
    }

    false
}

fn expr_factor_multisets_equal(ctx: &Context, left: &[ExprId], right: &[ExprId]) -> bool {
    if left.len() != right.len() {
        return false;
    }

    let checker = SemanticEqualityChecker::new(ctx);
    let mut used = vec![false; right.len()];
    'outer: for left_factor in left {
        for (index, right_factor) in right.iter().enumerate() {
            if used[index] {
                continue;
            }
            if left_factor == right_factor || checker.are_equal(*left_factor, *right_factor) {
                used[index] = true;
                continue 'outer;
            }
        }

        return false;
    }

    true
}

fn sqrt_like_base_and_exponent(ctx: &Context, expr: ExprId) -> Option<(ExprId, BigRational)> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            let exp = crate::numeric_eval::as_rational_const(ctx, *exp)?;
            (exp == BigRational::new(1.into(), 2.into())
                || exp == BigRational::new((-1).into(), 2.into()))
            .then_some((*base, exp))
        }
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) =>
        {
            Some((args[0], BigRational::new(1.into(), 2.into())))
        }
        _ => None,
    }
}

fn is_one_expr(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value.is_one())
}

fn is_negative_one_expr(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if *value == -BigRational::one())
}

fn rational_const_expr(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    let (sign, expr) = strip_negated_term(ctx, expr);
    crate::numeric_eval::as_rational_const(ctx, expr).map(|value| sign * value.clone())
}

fn accumulate_sqrt_product_factor(
    ctx: &Context,
    expr: ExprId,
    power: BigRational,
    normalized: &mut NormalizedSqrtProductTerm,
) -> Option<()> {
    match ctx.get(expr) {
        Expr::Neg(inner) => {
            let integer_power = rational_to_i32(&power)?;
            if integer_power % 2 != 0 {
                normalized.sign = -normalized.sign;
            }
            accumulate_sqrt_product_factor(ctx, *inner, power, normalized)
        }
        Expr::Number(value) => accumulate_rational_sqrt_product_factor(value, power, normalized),
        Expr::Mul(left, right) => {
            if !power.is_integer() {
                return accumulate_polynomial_sqrt_product_factor(ctx, expr, power, normalized);
            }
            accumulate_sqrt_product_factor(ctx, *left, power.clone(), normalized)?;
            accumulate_sqrt_product_factor(ctx, *right, power, normalized)
        }
        Expr::Div(num, den) => {
            if !power.is_integer() {
                return accumulate_polynomial_div_sqrt_product_factor(
                    ctx, *num, *den, power, normalized,
                );
            }
            accumulate_sqrt_product_factor(ctx, *num, power.clone(), normalized)?;
            accumulate_sqrt_product_factor(ctx, *den, -power, normalized)
        }
        Expr::Pow(base, exp) => {
            let exp_value = crate::numeric_eval::as_rational_const(ctx, *exp)?.clone();
            if exp_value == BigRational::new(1.into(), 2.into())
                || exp_value == BigRational::new((-1).into(), 2.into())
            {
                normalized.saw_root_like = true;
            }
            accumulate_sqrt_product_factor(ctx, *base, power * exp_value, normalized)
        }
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) =>
        {
            normalized.saw_root_like = true;
            accumulate_sqrt_product_factor(
                ctx,
                args[0],
                power * BigRational::new(1.into(), 2.into()),
                normalized,
            )
        }
        _ => accumulate_affine_sqrt_product_factor(ctx, expr, power, normalized),
    }
}

fn accumulate_rational_sqrt_product_factor(
    value: &BigRational,
    power: BigRational,
    normalized: &mut NormalizedSqrtProductTerm,
) -> Option<()> {
    if value.is_zero() {
        return None;
    }

    let mut value = value.clone();
    if value.is_negative() {
        let integer_power = rational_to_i32(&power)?;
        if integer_power % 2 != 0 {
            normalized.sign = -normalized.sign;
        }
        value = value.abs();
    }

    let squared_power = rational_to_i32(&(power * BigRational::from_integer(2.into())))?;
    multiply_squared_scale(&mut normalized.squared_scale, &value, squared_power);
    Some(())
}

fn accumulate_affine_sqrt_product_factor(
    ctx: &Context,
    expr: ExprId,
    power: BigRational,
    normalized: &mut NormalizedSqrtProductTerm,
) -> Option<()> {
    let vars = cas_ast::collect_variables(ctx, expr);
    if vars.len() != 1 {
        return None;
    }
    let var = vars.iter().next()?;
    let poly = crate::polynomial::Polynomial::from_expr(ctx, expr, var).ok()?;
    if poly.degree() != 1 {
        return None;
    }

    accumulate_sqrt_product_polynomial(poly, power, normalized)
}

fn accumulate_polynomial_div_sqrt_product_factor(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    power: BigRational,
    normalized: &mut NormalizedSqrtProductTerm,
) -> Option<()> {
    if let Some(value) = crate::numeric_eval::as_rational_const(ctx, num) {
        accumulate_rational_sqrt_product_factor(&value, power.clone(), normalized)?;
        return accumulate_polynomial_sqrt_product_factor(ctx, den, -power, normalized);
    }

    if let Some(value) = crate::numeric_eval::as_rational_const(ctx, den) {
        accumulate_polynomial_sqrt_product_factor(ctx, num, power.clone(), normalized)?;
        return accumulate_rational_sqrt_product_factor(&value, -power, normalized);
    }

    None
}

fn accumulate_polynomial_sqrt_product_factor(
    ctx: &Context,
    expr: ExprId,
    power: BigRational,
    normalized: &mut NormalizedSqrtProductTerm,
) -> Option<()> {
    let vars = cas_ast::collect_variables(ctx, expr);
    if vars.len() != 1 {
        return None;
    }
    let var = vars.iter().next()?;
    let poly = crate::polynomial::Polynomial::from_expr(ctx, expr, var).ok()?;
    if poly.degree() < 1 {
        return None;
    }

    accumulate_sqrt_product_polynomial(poly, power, normalized)
}

fn accumulate_sqrt_product_polynomial(
    mut poly: crate::polynomial::Polynomial,
    power: BigRational,
    normalized: &mut NormalizedSqrtProductTerm,
) -> Option<()> {
    let squared_power = rational_to_i32(&(power * BigRational::from_integer(2.into())))?;
    let content = poly.content();
    if content.is_zero() {
        return None;
    }
    if !content.is_one() {
        multiply_squared_scale(&mut normalized.squared_scale, &content, squared_power);
        poly = poly.div_scalar(&content);
    }

    let key = AffinePolyKey {
        var: poly.var,
        coeffs: poly.coeffs,
    };
    let entry = normalized.affine_powers.entry(key).or_insert(0);
    *entry += squared_power;
    Some(())
}

fn rational_to_i32(value: &BigRational) -> Option<i32> {
    if !value.is_integer() {
        return None;
    }
    value.to_integer().try_into().ok()
}

fn multiply_squared_scale(scale: &mut BigRational, base: &BigRational, exponent: i32) {
    let factor = rational_pow_i32(base, exponent);
    *scale *= factor;
}

fn rational_pow_i32(base: &BigRational, exponent: i32) -> BigRational {
    if exponent == 0 {
        return BigRational::one();
    }

    let mut result = BigRational::one();
    for _ in 0..exponent.unsigned_abs() {
        result *= base.clone();
    }
    if exponent < 0 {
        BigRational::one() / result
    } else {
        result
    }
}

/// True when `expr` carries a literal non-finite or undefined value — an
/// `Infinity`/`Undefined` constant, or a division with a provably-zero
/// denominator — anywhere in its tree.
///
/// Subtracting such a term from itself does NOT cancel to `0`: `inf - inf`,
/// `(1/0) - (1/0)` and `undefined - undefined` are indeterminate, not zero. Every
/// structural additive-cancellation path must decline when this holds, so this
/// is the shared gate for the cancel rewrites here and for the orchestrator's
/// exact-zero / common-scale collapse shortcuts in `cas_engine`.
/// True when `expr` is provably equal to zero over the reals — so a division by
/// it is `c/0` (undefined) and a cancellation/fold against it is unsound. Combines:
/// - numeric folding (`as_rational_const`): `1 - 1`, `1^2 - 1`, `2^2 - 4`, `1*0`;
/// - structural additive cancellation (`is_structurally_zero`): `x - x`,
///   `x^2 - x^2`, telescoping sums, literal `0`;
/// - a product with a provably-zero factor: `0*x`, `(x - x)*y`.
///
/// - a polynomial identity that normalizes to the ZERO polynomial: `x*x - x^2`,
///   `2x - x - x`, `(x - 1)(x + 1) - (x^2 - 1)`.
///
/// Conservative: returns `false` when zero-ness is not provable (a bare variable,
/// `x - 1`, …), so it never blocks a legitimate division. EXACT — numeric folding,
/// structural cancellation and exact rational polynomial normalization (no float,
/// no random-point probing), so it never reports a false zero. A polynomial that is
/// the zero polynomial is zero for ALL values of its variables, so a denominator
/// equal to it is `c/0` everywhere.
pub fn is_provably_zero(ctx: &Context, expr: ExprId) -> bool {
    if let Some(rat) = exact_rational_value(ctx, expr) {
        return rat.is_zero();
    }
    if crate::expr_relations::is_structurally_zero(ctx, expr) {
        return true;
    }
    if let Expr::Mul(l, r) = ctx.get(expr) {
        return is_provably_zero(ctx, *l) || is_provably_zero(ctx, *r);
    }
    // `0 / c = 0` for a NONZERO constant `c`: `(sin^2+cos^2-1)/2`. (A symbolic or
    // zero denominator is left alone — `0/0` is undefined, not zero.)
    if let Expr::Div(num, den) = ctx.get(expr) {
        if as_rational_const(ctx, *den).is_some_and(|d| !d.is_zero()) && is_provably_zero(ctx, *num)
        {
            return true;
        }
    }
    // `0^n = 0` for a strictly-positive exponent: `(x*x - x^2)^2`, `(2x-x-x)^3`.
    // (`0^0` is indeterminate and `0^(neg)` is undefined, so require `exp > 0`.)
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        let (base, exp) = (*base, *exp);
        if is_provably_zero(ctx, base)
            && exact_rational_value(ctx, exp).is_some_and(|e| e.is_positive())
        {
            return true;
        }
    }
    // Polynomial identity zero: only attempt on additive nodes (a single
    // variable/power/function can never be identically zero), and only after the
    // cheaper checks above. Bounded by `PolyBudget`; a non-polynomial expression
    // (a function like `sin`, a division, an over-budget power) converts to an
    // error and falls through, so trig/transcendental identities are NOT claimed.
    if matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        // Bounded budget: the default caps `max_pow_exp` at 2, which misses cube /
        // quartic identities like `(x+1)^3 - x^3 - 3x^2 - 3x - 1`. Raise it enough to
        // expand the common cases; `max_terms` still caps multivariate blow-up, so an
        // over-budget denominator returns an error and falls through (sound — it is
        // just left un-flagged, never wrongly flagged).
        let budget = crate::multipoly::PolyBudget {
            max_terms: 256,
            max_total_degree: 64,
            max_pow_exp: 12,
        };
        if let Ok(mut poly) = crate::multipoly::multipoly_from_expr(ctx, expr, &budget) {
            poly.normalize();
            if poly.is_zero() {
                return true;
            }
        }
        // Transcendental Pythagorean identities (the multipoly check declines trig):
        // `k·sin²+k·cos²−k`, `k·cosh²−k·sinh²−k`, `k·sec²−k·tan²−k`, `k·csc²−k·cot²−k`.
        if is_pythagorean_identity_zero(ctx, expr) {
            return true;
        }
    }
    false
}

/// Extract `(signed_coeff, builtin, arg)` from a term equal to `c · f(arg)^2` where
/// `f` is a trig/hyperbolic builtin and `c` a rational — accepting BOTH spellings
/// `f(arg)^2` and `f(arg)*f(arg)` (and any mix whose total power on a single
/// `f(arg)` is exactly 2). Folds a leading `Neg` and numeric factors into `c`.
/// Returns `None` for anything else (a different total power, two distinct
/// functions/arguments, a non-trig factor).
fn extract_squared_builtin_term(
    ctx: &Context,
    term: ExprId,
) -> Option<(BigRational, BuiltinFn, ExprId)> {
    let mut coef = BigRational::one();
    let mut working = term;
    if let Expr::Neg(inner) = ctx.get(working) {
        coef = -coef;
        working = *inner;
    }
    // Peel division by a nonzero numeric constant into the coefficient: `sin^2/2`.
    while let Expr::Div(num, den) = ctx.get(working) {
        let Some(d) = as_rational_const(ctx, *den) else {
            break;
        };
        if d.is_zero() {
            return None;
        }
        coef /= d;
        working = *num;
    }

    let mut fn_match: Option<(BuiltinFn, ExprId)> = None;
    let mut total_power: i64 = 0;
    for factor in mul_leaves(ctx, working) {
        if let Some(n) = as_rational_const(ctx, factor) {
            coef *= n;
            continue;
        }
        // Resolve this factor to a (function-base, integer power).
        let (base, power) = match ctx.get(factor) {
            Expr::Pow(b, e) => {
                let p = as_rational_const(ctx, *e)?;
                if !p.is_integer() {
                    return None;
                }
                (*b, p.to_integer().to_i64()?)
            }
            _ => (factor, 1),
        };
        let Expr::Function(fn_id, args) = ctx.get(base) else {
            return None;
        };
        if args.len() != 1 {
            return None;
        }
        let builtin = ctx.builtin_of(*fn_id)?;
        if !matches!(
            builtin,
            BuiltinFn::Sin
                | BuiltinFn::Cos
                | BuiltinFn::Cosh
                | BuiltinFn::Sinh
                | BuiltinFn::Sec
                | BuiltinFn::Tan
                | BuiltinFn::Csc
                | BuiltinFn::Cot
        ) {
            return None;
        }
        match &fn_match {
            None => fn_match = Some((builtin, args[0])),
            Some((b, a)) => {
                if *b != builtin || compare_expr(ctx, *a, args[0]) != Ordering::Equal {
                    return None; // two distinct trig atoms in one term
                }
            }
        }
        total_power += power;
    }

    let (b, arg) = fn_match?;
    (total_power == 2).then_some((coef, b, arg))
}

/// True when `expr` is identically zero by a Pythagorean identity over the reals:
/// `k·sin²(x) + k·cos²(x) − k`, `k·cosh²(x) − k·sinh²(x) − k`,
/// `k·sec²(x) − k·tan²(x) − k`, `k·csc²(x) − k·cot²(x) − k` — for any rational `k`
/// and any term order/sign. Exact: rational-coefficient bookkeeping, no float, no
/// random-point probing, so it never reports a false zero. It requires EXACTLY the
/// two squared terms (same argument) plus a numeric constant; anything else (a
/// stray term, a different argument, a different identity such as a double-angle or
/// `e^(ln x) − x`) declines — those remain honest residuals.
fn is_pythagorean_identity_zero(ctx: &Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    let mut squared: Vec<(BigRational, BuiltinFn, ExprId)> = Vec::new();
    let mut constant = BigRational::zero();
    for (term, sign) in view.terms {
        let s = if sign == Sign::Neg {
            -BigRational::one()
        } else {
            BigRational::one()
        };
        if let Some(c) = as_rational_const(ctx, term) {
            constant += s * c;
        } else if let Some((coeff, builtin, arg)) = extract_squared_builtin_term(ctx, term) {
            squared.push((s * coeff, builtin, arg));
        } else {
            return false;
        }
    }
    if squared.len() != 2 {
        return false;
    }
    let (c1, b1, a1) = &squared[0];
    let (c2, b2, a2) = &squared[1];
    if compare_expr(ctx, *a1, *a2) != Ordering::Equal {
        return false;
    }
    let pair_is = |x: BuiltinFn, y: BuiltinFn| (*b1 == x && *b2 == y) || (*b1 == y && *b2 == x);
    // sin²+cos²=1: same coefficient on both squares, constant = -k.
    if pair_is(BuiltinFn::Sin, BuiltinFn::Cos) {
        return c1 == c2 && constant == -c1.clone();
    }
    // cosh²-sinh²=1, sec²-tan²=1, csc²-cot²=1: the "positive" square's coefficient is
    // the negative of the other's, and the constant is its negation.
    for (pos, neg) in [
        (BuiltinFn::Cosh, BuiltinFn::Sinh),
        (BuiltinFn::Sec, BuiltinFn::Tan),
        (BuiltinFn::Csc, BuiltinFn::Cot),
    ] {
        if pair_is(pos, neg) {
            let (cp, cn) = if *b1 == pos { (c1, c2) } else { (c2, c1) };
            return *cp == -cn.clone() && constant == -cp.clone();
        }
    }
    false
}

/// Exact rational value of a fully-numeric (variable-free) expression, including
/// integer-exponent powers (`1^2 - 1`, `2^2 - 4`) which `as_rational_const`
/// declines. Returns `None` for anything not exactly rational (a variable, an
/// irrational power like `2^(1/2)`, an indeterminate `0^0`, a zero denominator).
/// Exact only — no float evaluation — so it never reports a false zero.
fn exact_rational_value(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Number(n) => Some(n.clone()),
        Expr::Neg(a) => Some(-exact_rational_value(ctx, *a)?),
        Expr::Add(a, b) => Some(exact_rational_value(ctx, *a)? + exact_rational_value(ctx, *b)?),
        Expr::Sub(a, b) => Some(exact_rational_value(ctx, *a)? - exact_rational_value(ctx, *b)?),
        Expr::Mul(a, b) => Some(exact_rational_value(ctx, *a)? * exact_rational_value(ctx, *b)?),
        Expr::Div(a, b) => {
            let d = exact_rational_value(ctx, *b)?;
            if d.is_zero() {
                return None;
            }
            Some(exact_rational_value(ctx, *a)? / d)
        }
        Expr::Pow(base, exp) => {
            let base = exact_rational_value(ctx, *base)?;
            let exp = exact_rational_value(ctx, *exp)?;
            if !exp.is_integer() {
                return None; // irrational / radical power — not exactly rational
            }
            let e = exp.to_integer().to_i32()?;
            if base.is_zero() && e <= 0 {
                return None; // 0^0 indeterminate, 0^(neg) undefined
            }
            let mut acc = BigRational::one();
            for _ in 0..e.unsigned_abs() {
                acc *= &base;
            }
            Some(if e < 0 { BigRational::one() / acc } else { acc })
        }
        _ => None,
    }
}

pub fn expr_carries_nonfinite_or_undefined(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Constant(cas_ast::Constant::Infinity | cas_ast::Constant::Undefined) => true,
        Expr::Div(num, den) => {
            is_provably_zero(ctx, *den)
                || expr_carries_nonfinite_or_undefined(ctx, *num)
                || expr_carries_nonfinite_or_undefined(ctx, *den)
        }
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Pow(a, b) => {
            expr_carries_nonfinite_or_undefined(ctx, *a)
                || expr_carries_nonfinite_or_undefined(ctx, *b)
        }
        Expr::Neg(a) | Expr::Hold(a) => expr_carries_nonfinite_or_undefined(ctx, *a),
        Expr::Function(_, args) => args
            .iter()
            .any(|&c| expr_carries_nonfinite_or_undefined(ctx, c)),
        _ => false,
    }
}

/// True when rewriting `before` into `after` would unsoundly drop a non-finite or
/// undefined value.
///
/// `before` is an additive node (`Add`/`Sub`) that carries a literal non-finite
/// or undefined value, but `after` no longer does. Every such rewrite is a
/// cancellation/collapse that silently turned an *indeterminate* difference
/// (`inf - inf`, `x/0 - x/0`, `sqrt(inf) - sqrt(inf)`, `undefined - undefined`,
/// `ln(inf) - ln(inf) + 7`) into a purely finite value — which is unsound.
///
/// This is the universal backstop applied at every rewrite-acceptance point
/// (`Rule::apply` and the orchestrator root-shortcut dispatch), because the same
/// "this additive combination is zero" conclusion is reached by a large family of
/// independent rules and shortcuts. A *sound* resolution either keeps the value
/// non-finite (`inf + 1 -> inf`) or folds to `undefined` — both of which still
/// "carry", so they are never blocked. Function/quotient *evaluations* such as
/// `atan(inf) -> pi/2` or `1/inf -> 0` operate on non-additive nodes, so they are
/// never blocked either.
///
/// The second clause covers division by a PROVABLY-zero denominator: a quotient
/// `c/(x-x)` is `c/0` (undefined), so a fraction-simplification shortcut that
/// cancels its zero factor — `(x^2-x^2)/(x-x) -> x+x`, `(3x-3x)/(x-x) -> 3` — is
/// unsound and must be rejected. This is scoped to a *provably-zero* denominator
/// (not any non-finite-bearing `Div`), so a legitimate evaluation like `1/inf -> 0`
/// (where `inf` is non-zero) is never blocked.
pub fn rewrite_unsoundly_drops_nonfinite(ctx: &Context, before: ExprId, after: ExprId) -> bool {
    if expr_carries_nonfinite_or_undefined(ctx, after) {
        return false;
    }
    match ctx.get(before) {
        Expr::Add(_, _) | Expr::Sub(_, _) => expr_carries_nonfinite_or_undefined(ctx, before),
        Expr::Div(_, den) => is_provably_zero(ctx, *den),
        _ => false,
    }
}

/// Rewrite `a - a` to `0`, returning the cancelled inner expression.
pub fn try_rewrite_sub_self_zero_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ArithmeticCancelRewrite> {
    if expr_carries_nonfinite_or_undefined(ctx, expr) {
        return None;
    }
    let inner = match_sub_self_semantic_expr(ctx, expr)?;
    Some(ArithmeticCancelRewrite {
        rewritten: ctx.num(0),
        inner,
    })
}

/// Rewrite `a + (-a)` (or `(-a) + a`) to `0`, returning `a`.
pub fn try_rewrite_add_inverse_zero_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ArithmeticCancelRewrite> {
    if expr_carries_nonfinite_or_undefined(ctx, expr) {
        return None;
    }
    let inner = match_add_inverse_expr(ctx, expr)?;
    Some(ArithmeticCancelRewrite {
        rewritten: ctx.num(0),
        inner,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        expr_carries_nonfinite_or_undefined, is_provably_zero, match_add_inverse_expr,
        match_sub_self_semantic_expr, rewrite_unsoundly_drops_nonfinite,
        try_rewrite_add_inverse_zero_expr, try_rewrite_sub_self_zero_expr,
    };
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    #[test]
    fn rewrite_filter_blocks_nonfinite_additive_drop_but_allows_evaluations() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let undef = ctx.add(cas_ast::Expr::Constant(cas_ast::Constant::Undefined));

        // Additive non-finite collapses to a finite value -> BLOCKED.
        for src in [
            "inf - inf",
            "sqrt(inf) - sqrt(inf)",
            "ln(inf) - ln(inf) + 7",
            "x/0 - x/0 + y/0 - y/0",
            "sin(undefined) - sin(undefined)",
        ] {
            let before = parse(src, &mut ctx).expect("parse");
            assert!(
                rewrite_unsoundly_drops_nonfinite(&ctx, before, zero),
                "`{src}` -> 0 must be flagged as an unsound drop"
            );
        }

        // Folding the SAME non-finite difference to `undefined` is allowed (after
        // still carries the non-finite marker).
        let inf_minus_inf = parse("inf - inf", &mut ctx).expect("parse");
        assert!(!rewrite_unsoundly_drops_nonfinite(
            &ctx,
            inf_minus_inf,
            undef
        ));

        // A non-ADDITIVE node carrying non-finite (a function/quotient evaluation
        // like `atan(inf) -> pi/2` or `1/inf -> 0`) is never blocked.
        let atan_inf = parse("atan(inf)", &mut ctx).expect("parse");
        let pi_half = parse("pi/2", &mut ctx).expect("parse");
        assert!(!rewrite_unsoundly_drops_nonfinite(&ctx, atan_inf, pi_half));

        // A purely finite additive expression simplifying is never blocked.
        let finite = parse("2*x + 3*x", &mut ctx).expect("parse");
        let five_x = parse("5*x", &mut ctx).expect("parse");
        assert!(!rewrite_unsoundly_drops_nonfinite(&ctx, finite, five_x));

        // A `Div` with a PROVABLY-zero denominator folding to a finite value is
        // blocked (`(x^2-x^2)/(x-x) -> x+x`), but `1/inf -> 0` (denominator non-zero)
        // and a legit quotient (`(x^2-1)/(x-1) -> x+1`) are NOT.
        let sq_over_zero = parse("(x^2-x^2)/(x-x)", &mut ctx).expect("parse");
        let x_plus_x = parse("x+x", &mut ctx).expect("parse");
        assert!(rewrite_unsoundly_drops_nonfinite(
            &ctx,
            sq_over_zero,
            x_plus_x
        ));
        let one_over_inf = parse("1/inf", &mut ctx).expect("parse");
        let zero = ctx.num(0);
        assert!(!rewrite_unsoundly_drops_nonfinite(&ctx, one_over_inf, zero));
        let legit = parse("(x^2-1)/(x-1)", &mut ctx).expect("parse");
        let x_plus_1 = parse("x+1", &mut ctx).expect("parse");
        assert!(!rewrite_unsoundly_drops_nonfinite(&ctx, legit, x_plus_1));
    }

    #[test]
    fn detects_nonfinite_and_undefined_terms() {
        let mut ctx = Context::new();
        for src in [
            "inf",
            "undefined",
            "1/0",
            "x/0",
            "x/0 - x/0",
            "inf*x",
            "2 + inf",
            // R3-3: division by a PROVABLY (not literally) zero denominator.
            "1/(x-x)",
            "1/(x^2-x^2)",
            "1/(0*x)",
            "1/(x-x) - 1/(x-x)",
        ] {
            let expr = parse(src, &mut ctx).expect("parse");
            assert!(
                expr_carries_nonfinite_or_undefined(&ctx, expr),
                "expected `{src}` to carry a non-finite/undefined value"
            );
        }
        for src in [
            "x",
            "x - x",
            "sin(x)",
            "1/x",
            "x/2 - x/2",
            "a/c - b/c",
            "1/(x-1)",
        ] {
            let expr = parse(src, &mut ctx).expect("parse");
            assert!(
                !expr_carries_nonfinite_or_undefined(&ctx, expr),
                "expected `{src}` to be finite/defined"
            );
        }
    }

    #[test]
    fn is_provably_zero_detects_structural_numeric_and_product_zeros() {
        let mut ctx = Context::new();
        for src in [
            // numeric / structural / product-with-zero-factor
            "0",
            "1-1",
            "2-2",
            "x-x",
            "x^2-x^2",
            "0*x",
            "x*0",
            "(x-x)*y",
            // polynomial identities (zero only after expansion/normalization)
            "x*x - x^2",
            "2*x - x - x",
            "(x-1)*(x+1) - (x^2-1)",
            "(x+1)^3 - x^3 - 3*x^2 - 3*x - 1",
            "(x+y)^2 - x^2 - 2*x*y - y^2",
            // powers of an identically-zero polynomial: 0^n = 0
            "(x*x - x^2)^2",
            "(2*x - x - x)^3",
            // Pythagorean identities, both `^2` and `f*f` spellings, any k/order/arg
            "sin(x)^2 + cos(x)^2 - 1",
            "cos(t)^2 + sin(t)^2 - 1",
            "1 - sin(x)^2 - cos(x)^2",
            "3*sin(x)^2 + 3*cos(x)^2 - 3",
            "sin(x)*sin(x) + cos(x)*cos(x) - 1",
            "sin(2*y)^2 + cos(2*y)^2 - 1",
            "cosh(x)^2 - sinh(x)^2 - 1",
            "cosh(x)*cosh(x) - sinh(x)*sinh(x) - 1",
            "sec(x)^2 - tan(x)^2 - 1",
            "csc(x)^2 - cot(x)^2 - 1",
            // fractional coefficients, both `f^2/c` and `(…)/c` spellings
            "sin(x)^2/2 + cos(x)^2/2 - 1/2",
            "(sin(x)^2 + cos(x)^2 - 1)/2",
            "cosh(x)^2/3 - sinh(x)^2/3 - 1/3",
        ] {
            let expr = parse(src, &mut ctx).expect("parse");
            assert!(
                is_provably_zero(&ctx, expr),
                "`{src}` must be provably zero"
            );
        }
        for src in [
            // never a false positive on a nonzero (even close-looking) expression
            "x",
            "x-1",
            "1",
            "x*y",
            "x^2",
            "2",
            "x*x - x^2 + 1",
            "2*x - x",
            "(x-1)^2",
            "(x+1)^3",
            "(x+1)^4 - x^4",
            // Pythagorean-adjacent but NONzero: must never be flagged.
            "sin(x)^2 + cos(x)^2",
            "sin(x)*sin(x) + cos(x)*cos(x)",
            "sin(x)^2 + cos(x)^2 + 1",
            "sin(x)^2 - cos(x)^2",
            "2*sin(x)^2 + cos(x)^2 - 1",
            "sin(x)^2 + cos(y)^2 - 1",
            "sin(x)^2 + cos(x)^4 - 1",
            "cosh(x)^2 - sinh(x)^2 + 1",
            "sin(x)*cos(x) - 1",
        ] {
            let expr = parse(src, &mut ctx).expect("parse");
            assert!(
                !is_provably_zero(&ctx, expr),
                "`{src}` must NOT be provably zero"
            );
        }
    }

    #[test]
    fn declines_sub_self_cancellation_for_division_by_zero() {
        // `x/0 - x/0` must NOT cancel to 0: `x/0` is undefined, so the difference
        // is indeterminate, not zero.
        let mut ctx = Context::new();
        let expr = parse("x/0 - x/0", &mut ctx).expect("parse");
        assert!(try_rewrite_sub_self_zero_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn declines_add_inverse_cancellation_for_infinity() {
        // `inf + (-inf)` is indeterminate, not 0.
        let mut ctx = Context::new();
        let expr = parse("inf + (-inf)", &mut ctx).expect("parse");
        assert!(try_rewrite_add_inverse_zero_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn still_cancels_finite_sub_self() {
        // Regression: ordinary finite self-subtraction must still cancel.
        let mut ctx = Context::new();
        let expr = parse("sin(x)/2 - sin(x)/2", &mut ctx).expect("parse");
        assert!(try_rewrite_sub_self_zero_expr(&mut ctx, expr).is_some());
    }

    #[test]
    fn detects_sub_self_symbolic() {
        let mut ctx = Context::new();
        let expr = parse("tan(3*x)-tan(3*x)", &mut ctx).expect("parse");
        assert!(match_sub_self_semantic_expr(&ctx, expr).is_some());
    }

    #[test]
    fn detects_sub_self_with_abs_sub_mirror_forms() {
        let mut ctx = Context::new();
        let expr = parse(
            "abs((1/(x - 1) + 1/(x + 1)) - 1) - abs(1 - (1/(x - 1) + 1/(x + 1)))",
            &mut ctx,
        )
        .expect("parse");
        assert!(match_sub_self_semantic_expr(&ctx, expr).is_some());
    }

    #[test]
    fn detects_sub_self_with_abs_sub_mirror_runtime_shape() {
        let mut ctx = Context::new();
        let expr = parse(
            "abs((2*x)/(x^2 - 1) - 1) - abs(1 - 2*x/(x^2 - 1))",
            &mut ctx,
        )
        .expect("parse");
        assert!(match_sub_self_semantic_expr(&ctx, expr).is_some());
    }

    #[test]
    fn detects_add_inverse_both_orders() {
        let mut ctx = Context::new();
        let expr1 = parse("a+(-a)", &mut ctx).expect("parse");
        assert!(match_add_inverse_expr(&ctx, expr1).is_some());

        let expr2 = parse("(-a)+a", &mut ctx).expect("parse");
        assert!(match_add_inverse_expr(&ctx, expr2).is_some());
    }

    #[test]
    fn detects_add_inverse_with_abs_sub_mirror_forms() {
        let mut ctx = Context::new();
        let expr = parse(
            "abs((2*x)/(x^2 - 1) - 1) + (-abs(1 - 2*x/(x^2 - 1)))",
            &mut ctx,
        )
        .expect("parse");
        assert!(match_add_inverse_expr(&ctx, expr).is_some());
    }

    #[test]
    fn detects_add_inverse_with_semantically_equal_trig_wrappers() {
        let mut ctx = Context::new();
        let expr = parse(
            "sin((2*u + 1)/(u*(u+1))) + (-sin((2*u + 1)/(u^2 + u)))",
            &mut ctx,
        )
        .expect("parse");
        assert!(match_add_inverse_expr(&ctx, expr).is_some());
    }

    #[test]
    fn detects_affine_sqrt_fraction_sign_oriented_sub_self() {
        let mut ctx = Context::new();
        let expr = parse(
            "-(5-3*x)^(1/2)/((10-6*x)*(x-2)) - 3*(5-3*x)^(-1/2)/(12-6*x)",
            &mut ctx,
        )
        .expect("parse");
        assert!(match_sub_self_semantic_expr(&ctx, expr).is_some());
    }

    #[test]
    fn detects_affine_sqrt_fraction_sign_oriented_add_inverse() {
        let mut ctx = Context::new();
        let expr = parse(
            "-(5-3*x)^(1/2)/((10-6*x)*(x-2)) + (-3*(5-3*x)^(-1/2)/(12-6*x))",
            &mut ctx,
        )
        .expect("parse");
        assert!(match_add_inverse_expr(&ctx, expr).is_some());
    }

    #[test]
    fn detects_affine_sqrt_fraction_with_external_div_coefficient() {
        let mut ctx = Context::new();
        let expr = parse(
            "-(5-3*x)^(1/2)/((10-6*x)*(x-2)) + (-3*((5-3*x)^(-1/2)/(12-6*x)))",
            &mut ctx,
        )
        .expect("parse");
        assert!(match_add_inverse_expr(&ctx, expr).is_some());
    }

    #[test]
    fn detects_sqrt_product_fraction_with_non_square_rational_content() {
        let mut ctx = Context::new();
        let expr = parse(
            "(1/2 / ((-x-1)*(2*x+3)))^(1/2) - ((-2*x-2)*(2*x+3))^(-1/2)",
            &mut ctx,
        )
        .expect("parse");
        assert!(match_sub_self_semantic_expr(&ctx, expr).is_some());
    }

    #[test]
    fn detects_sqrt_product_fraction_add_inverse_with_non_square_rational_content() {
        let mut ctx = Context::new();
        let expr = parse(
            "(1/2 / ((-x-1)*(2*x+3)))^(1/2) + (-(((-2*x-2)*(2*x+3))^(-1/2)))",
            &mut ctx,
        )
        .expect("parse");
        assert!(match_add_inverse_expr(&ctx, expr).is_some());
    }

    #[test]
    fn detects_reciprocal_sqrt_product_minus_one_with_polynomial_equivalent_bases() {
        let mut ctx = Context::new();
        let expr = parse("(x*(1-x))^((1/2))*(x-x^2)^((-1/2)) - 1", &mut ctx).expect("parse");
        assert!(match_sub_self_semantic_expr(&ctx, expr).is_some());
    }

    #[test]
    fn detects_reciprocal_sqrt_product_add_negative_one_with_polynomial_equivalent_bases() {
        let mut ctx = Context::new();
        let product = parse("(x*(1-x))^((1/2))*(x-x^2)^((-1/2))", &mut ctx).expect("parse");
        let one = ctx.num(1);
        let neg_one = ctx.add(cas_ast::Expr::Neg(one));
        let expr = ctx.add_raw(cas_ast::Expr::Add(product, neg_one));
        assert!(match_add_inverse_expr(&ctx, expr).is_some());
    }

    #[test]
    fn detects_scaled_reciprocal_sqrt_product_minus_matching_constant() {
        let mut ctx = Context::new();
        let expr = parse("2*(x*(1-x))^((-1/2))*(x-x^2)^((1/2)) - 2", &mut ctx).expect("parse");
        assert!(match_sub_self_semantic_expr(&ctx, expr).is_some());
    }

    #[test]
    fn detects_scaled_reciprocal_sqrt_product_add_negative_matching_constant() {
        let mut ctx = Context::new();
        let expr = parse("2*(x*(1-x))^((-1/2))*(x-x^2)^((1/2)) + (-2)", &mut ctx).expect("parse");
        assert!(match_add_inverse_expr(&ctx, expr).is_some());
    }

    #[test]
    fn detects_symbolic_scaled_reciprocal_sqrt_product_minus_matching_product() {
        let mut ctx = Context::new();
        let expr = parse("2*a*(x*(1-x))^((-1/2))*(x-x^2)^((1/2)) - 2*a", &mut ctx).expect("parse");
        assert!(match_sub_self_semantic_expr(&ctx, expr).is_some());
    }

    #[test]
    fn detects_symbolic_scaled_reciprocal_sqrt_product_add_negative_matching_product() {
        let mut ctx = Context::new();
        let expr =
            parse("2*a*(x*(1-x))^((-1/2))*(x-x^2)^((1/2)) + (-2*a)", &mut ctx).expect("parse");
        assert!(match_add_inverse_expr(&ctx, expr).is_some());
    }

    #[test]
    fn detects_split_reciprocal_sqrt_product_minus_one() {
        let mut ctx = Context::new();
        let expr = parse(
            "x^(-1/2)*(x*(x^(1/2)-x))^(1/2)*(x^(1/2)-x)^(-1/2) - 1",
            &mut ctx,
        )
        .expect("parse");
        assert!(match_sub_self_semantic_expr(&ctx, expr).is_some());
    }

    #[test]
    fn detects_symbolic_scaled_split_reciprocal_sqrt_product_minus_matching_product() {
        let mut ctx = Context::new();
        let expr = parse(
            "a*x^(-1/2)*(x*(x^(1/2)-x))^(1/2)*(x^(1/2)-x)^(-1/2) - a",
            &mut ctx,
        )
        .expect("parse");
        assert!(match_sub_self_semantic_expr(&ctx, expr).is_some());
    }

    #[test]
    fn rejects_symbolic_scaled_reciprocal_sqrt_product_with_mismatched_product() {
        let mut ctx = Context::new();
        let expr = parse("2*a*(x*(1-x))^((-1/2))*(x-x^2)^((1/2)) - 2*b", &mut ctx).expect("parse");
        assert!(match_sub_self_semantic_expr(&ctx, expr).is_none());
    }

    #[test]
    fn rejects_sqrt_product_split_without_shared_domain_proof() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x*y) - sqrt(x)*sqrt(y)", &mut ctx).expect("parse");
        assert!(match_sub_self_semantic_expr(&ctx, expr).is_none());
    }

    #[test]
    fn rejects_split_of_polynomial_product_under_root() {
        let mut ctx = Context::new();
        let expr = parse("sqrt((x+1)*(x+2)) - sqrt(x+1)*sqrt(x+2)", &mut ctx).expect("parse");
        assert!(match_sub_self_semantic_expr(&ctx, expr).is_none());
    }

    #[test]
    fn rewrites_sub_self_to_zero() {
        let mut ctx = Context::new();
        let expr = parse("x-x", &mut ctx).expect("parse");
        let rewrite = try_rewrite_sub_self_zero_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.rewritten
                }
            ),
            "0"
        );
    }

    #[test]
    fn rewrites_add_inverse_to_zero() {
        let mut ctx = Context::new();
        let expr = parse("x+(-x)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_add_inverse_zero_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.rewritten
                }
            ),
            "0"
        );
    }

    #[test]
    fn rewrites_affine_sqrt_fraction_sign_oriented_sub_self_to_zero() {
        let mut ctx = Context::new();
        let expr = parse(
            "-(5-3*x)^(1/2)/((10-6*x)*(x-2)) - 3*(5-3*x)^(-1/2)/(12-6*x)",
            &mut ctx,
        )
        .expect("parse");
        let rewrite = try_rewrite_sub_self_zero_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.rewritten
                }
            ),
            "0"
        );
    }
}
