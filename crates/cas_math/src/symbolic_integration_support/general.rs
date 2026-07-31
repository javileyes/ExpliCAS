//! `symbolic_integration_support`: familia `general`.
//!
//! Ver la cabecera de `symbolic_integration_support.rs` para el contexto.

use super::*;

pub(super) fn real_domain_is_empty_or_nonfinite_for_integration(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    real_domain_is_empty_or_nonfinite_over_reals(
        ctx,
        expr,
        SYMBOLIC_INTEGRATION_DOMAIN_PROOF_DEPTH,
        SYMBOLIC_INTEGRATION_DOMAIN_SCAN_DEPTH,
    )
}

pub(super) fn is_negative_half(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => *n == BigRational::new((-1).into(), 2.into()),
        Expr::Div(num, den) => is_number(ctx, *num, -1) && is_number(ctx, *den, 2),
        Expr::Neg(inner) => match ctx.get(*inner) {
            Expr::Number(n) => *n == BigRational::new(1.into(), 2.into()),
            Expr::Div(num, den) => is_number(ctx, *num, 1) && is_number(ctx, *den, 2),
            _ => false,
        },
        _ => false,
    }
}

pub(super) fn is_positive_half(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => *n == BigRational::new(1.into(), 2.into()),
        Expr::Div(num, den) => is_number(ctx, *num, 1) && is_number(ctx, *den, 2),
        _ => false,
    }
}

/// Flatten an additive tree (`Add`/`Sub`/`Neg`) into signed leaf terms `(is_positive, term)`.
pub(super) fn signed_additive_terms(ctx: &Context, expr: ExprId) -> Vec<(bool, ExprId)> {
    fn walk(ctx: &Context, e: ExprId, positive: bool, out: &mut Vec<(bool, ExprId)>) {
        match ctx.get(e) {
            Expr::Add(a, b) => {
                walk(ctx, *a, positive, out);
                walk(ctx, *b, positive, out);
            }
            Expr::Sub(a, b) => {
                walk(ctx, *a, positive, out);
                walk(ctx, *b, !positive, out);
            }
            Expr::Neg(inner) => walk(ctx, *inner, !positive, out),
            _ => out.push((positive, e)),
        }
    }
    let mut out = Vec::new();
    walk(ctx, expr, true, &mut out);
    out
}

/// `C(n, k)` for small non-negative `n, k`.
pub(super) fn binomial_i64(n: i64, k: i64) -> i64 {
    if k < 0 || k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result: i64 = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

pub(super) fn indexed_matching_unary_builtin_factor(
    ctx: &Context,
    factors: &[ExprId],
    builtin: BuiltinFn,
    arg: ExprId,
    excluded: &[usize],
) -> Option<usize> {
    factors.iter().enumerate().find_map(|(idx, factor)| {
        if excluded.contains(&idx) {
            return None;
        }
        let factor_arg = unary_builtin_arg(ctx, *factor, builtin)?;
        (compare_expr(ctx, factor_arg, arg) == Ordering::Equal).then_some(idx)
    })
}

pub(super) fn signed_unary_builtin_arg(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<(ExprId, BigRational)> {
    match ctx.get(expr) {
        Expr::Neg(inner) => {
            unary_builtin_arg(ctx, *inner, builtin).map(|arg| (arg, -BigRational::one()))
        }
        _ => unary_builtin_arg(ctx, expr, builtin).map(|arg| (arg, BigRational::one())),
    }
}

pub(super) fn divide_by_coeff_unless_one(
    ctx: &mut Context,
    integral: ExprId,
    coeff: ExprId,
) -> ExprId {
    let is_coeff_one = if let Expr::Number(n) = ctx.get(coeff) {
        n.is_one()
    } else {
        false
    };

    if is_coeff_one {
        integral
    } else {
        ctx.add(Expr::Div(integral, coeff))
    }
}

pub(super) fn divide_by_coeff_unless_one_preserving_presentation(
    ctx: &mut Context,
    integral: ExprId,
    coeff: ExprId,
) -> ExprId {
    let scaled = divide_by_coeff_unless_one(ctx, integral, coeff);
    if scaled == integral {
        scaled
    } else {
        cas_ast::hold::wrap_hold(ctx, scaled)
    }
}

pub(super) fn negate_integration_result(ctx: &mut Context, expr: ExprId) -> ExprId {
    let unheld = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let negated = match ctx.get(unheld).clone() {
        Expr::Neg(inner) => inner,
        Expr::Mul(left, right) if is_reciprocal_trig_call(ctx, right) => {
            let negative_scale = negate_scalar_expr(ctx, left);
            mul2_raw(ctx, negative_scale, right)
        }
        Expr::Mul(left, right) if is_reciprocal_trig_call(ctx, left) => {
            let negative_scale = negate_scalar_expr(ctx, right);
            mul2_raw(ctx, negative_scale, left)
        }
        Expr::Div(num, den) => match ctx.get(num).clone() {
            Expr::Neg(inner) => ctx.add(Expr::Div(inner, den)),
            _ => ctx.add(Expr::Neg(unheld)),
        },
        _ => ctx.add(Expr::Neg(unheld)),
    };

    if unheld == expr {
        negated
    } else {
        cas_ast::hold::wrap_hold(ctx, negated)
    }
}

pub(super) fn additive_var_dependent_part(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let terms = crate::expr_nary::add_terms_signed(ctx, expr);
    if terms.len() <= 1 {
        return None;
    }

    let mut dependent_terms = Vec::new();
    let mut removed_independent_term = false;
    for (term, sign) in terms {
        if contains_named_var(ctx, term, var) {
            dependent_terms.push(match sign {
                Sign::Pos => term,
                Sign::Neg => ctx.add(Expr::Neg(term)),
            });
        } else {
            removed_independent_term = true;
        }
    }

    if !removed_independent_term || dependent_terms.is_empty() {
        return None;
    }

    Some(build_balanced_add(ctx, &dependent_terms))
}

pub(super) fn remove_matching_factor(
    ctx: &Context,
    factors: &[ExprId],
    target: ExprId,
) -> Option<Vec<ExprId>> {
    let (index, _) = factors
        .iter()
        .enumerate()
        .find(|(_, factor)| compare_expr(ctx, **factor, target) == Ordering::Equal)?;
    Some(
        factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != index).then_some(*factor))
            .collect(),
    )
}

pub(super) fn positive_one_plus_non_one_term(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };

    if is_number(ctx, *left, 1) {
        return Some(*right);
    }
    if is_number(ctx, *right, 1) {
        return Some(*left);
    }

    None
}

/// `exp/sin/cos` (or `e^(.)`) of a pure quadratic `c x^2` (no linear/constant
/// term), so substituting `u = x^2` keeps the argument an affine function of u.
pub(super) fn is_elementary_function_of_pure_quadratic(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let inner = match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(
                    ctx.builtin_of(*fn_id),
                    Some(BuiltinFn::Exp | BuiltinFn::Sin | BuiltinFn::Cos)
                ) =>
        {
            args[0]
        }
        Expr::Pow(base, exp) if matches!(ctx.get(*base), Expr::Constant(Constant::E)) => *exp,
        _ => return false,
    };
    let Ok(poly) = Polynomial::from_expr(ctx, inner, var) else {
        return false;
    };
    poly.degree() == 2
        && poly.coeffs.first().is_none_or(|c| c.is_zero())
        && poly.coeffs.get(1).is_none_or(|c| c.is_zero())
}

pub(super) fn descending_factorial_ratio(top: u32, bottom: u32) -> BigRational {
    let mut acc = BigRational::one();
    for value in (bottom + 1)..=top {
        acc *= BigRational::from_integer(value.into());
    }
    acc
}

pub(super) fn negate_term_for_compact_integration_sum(ctx: &mut Context, term: ExprId) -> ExprId {
    let term = cas_ast::hold::unwrap_hold(ctx, term);
    if let Expr::Number(value) = ctx.get(term).clone() {
        return ctx.add(Expr::Number(-value));
    }
    if let Expr::Div(num, den) = ctx.get(term).clone() {
        let num = negate_term_for_compact_integration_sum(ctx, num);
        return ctx.add(Expr::Div(num, den));
    }

    let factors = mul_leaves(ctx, term);
    if factors.len() > 1 {
        let mut replaced = false;
        let mut negated_factors = Vec::with_capacity(factors.len());
        for factor in factors {
            if !replaced {
                if let Expr::Number(value) = ctx.get(factor).clone() {
                    negated_factors.push(ctx.add(Expr::Number(-value)));
                    replaced = true;
                    continue;
                }
            }
            negated_factors.push(factor);
        }
        if replaced {
            return build_balanced_mul(ctx, &negated_factors);
        }
    }

    ctx.add(Expr::Neg(term))
}

pub(super) fn is_positive_leading_quadratic(poly: &Polynomial) -> bool {
    if poly.degree() != 2 || poly.coeffs.len() != 3 || !poly.coeffs[2].is_positive() {
        return false;
    }
    true
}

pub(super) fn is_strictly_positive_quadratic(poly: &Polynomial) -> bool {
    if !is_positive_leading_quadratic(poly) {
        return false;
    }

    let four = BigRational::from_integer(4.into());
    let discriminant = poly.coeffs[1].clone() * poly.coeffs[1].clone()
        - four * poly.coeffs[2].clone() * poly.coeffs[0].clone();
    discriminant.is_negative()
}

pub(super) fn signed_term(ctx: &mut Context, term: ExprId, sign: Sign) -> ExprId {
    match sign {
        Sign::Pos => term,
        Sign::Neg => ctx.add(Expr::Neg(term)),
    }
}

pub(super) fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

pub(super) fn split_variable_free_scale_from_product(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    if !matches!(ctx.get(expr), Expr::Mul(_, _)) {
        return None;
    }

    let mut scale_factors = Vec::new();
    let mut cofactor_factors = Vec::new();
    for factor in mul_leaves(ctx, expr) {
        if contains_named_var(ctx, factor, var) {
            cofactor_factors.push(factor);
        } else {
            scale_factors.push(factor);
        }
    }

    if scale_factors.is_empty() || cofactor_factors.is_empty() {
        return None;
    }

    let scale = build_balanced_mul(ctx, &scale_factors);
    let cofactor = build_balanced_mul(ctx, &cofactor_factors);
    Some((scale, cofactor))
}

pub(super) fn is_neg_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    matches!(ctx.get(expr), Expr::Neg(inner) if is_var(ctx, *inner, var))
}

fn collect_required_conditions_from(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
    collectors: &[RequiredConditionCollector],
    conditions: &mut Vec<ExprId>,
) {
    for collector in collectors {
        match collector {
            RequiredConditionCollector::Optional(collector) => {
                conditions.extend(collector(ctx, expr, var));
            }
            RequiredConditionCollector::Multi(collector) => {
                conditions.extend(collector(ctx, expr, var));
            }
        }
    }
}

pub fn integrate_symbolic_required_nonzero_conditions(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Vec<ExprId> {
    if let Some(inner) = constant_scaled_integrand_inner(ctx, expr, var) {
        return integrate_symbolic_required_nonzero_conditions(ctx, inner, var);
    }

    let mut conditions = Vec::new();
    collect_required_conditions_from(
        ctx,
        expr,
        var,
        REQUIRED_NONZERO_CONDITION_COLLECTORS_BEFORE_RESIDUAL_SCAN,
        &mut conditions,
    );
    conditions.extend(residual_trig_pole_required_nonzero_conditions(ctx, expr));
    collect_required_conditions_from(
        ctx,
        expr,
        var,
        REQUIRED_NONZERO_CONDITION_COLLECTORS_AFTER_RESIDUAL_SCAN,
        &mut conditions,
    );

    dedup_required_conditions(ctx, conditions, var)
}

fn dedup_required_conditions(ctx: &mut Context, conditions: Vec<ExprId>, var: &str) -> Vec<ExprId> {
    let all_conditions = conditions.clone();
    let mut unique = Vec::new();
    for condition in conditions {
        if tanh_nonzero_dominated_by_sinh_nonzero(ctx, condition, &all_conditions) {
            continue;
        }
        if nonzero_condition_is_proven_for_symbolic_integration(ctx, condition) {
            continue;
        }
        if positive_constant_radius_quadratic_denominator_is_structurally_nonzero(
            ctx, condition, var,
        ) {
            continue;
        }
        if unique
            .iter()
            .any(|existing| compare_expr(ctx, *existing, condition) == Ordering::Equal)
        {
            continue;
        }
        unique.push(condition);
    }
    unique
}

fn nonzero_condition_is_proven_for_symbolic_integration(ctx: &mut Context, expr: ExprId) -> bool {
    if crate::calculus_domain_support::positive_condition_is_proven_over_reals(
        ctx,
        expr,
        SYMBOLIC_INTEGRATION_DOMAIN_PROOF_DEPTH,
    ) {
        return true;
    }

    match ctx.get(expr).clone() {
        Expr::Number(value) => !value.is_zero(),
        Expr::Constant(Constant::Pi | Constant::E | Constant::Phi) => true,
        Expr::Neg(inner) => nonzero_condition_is_proven_for_symbolic_integration(ctx, inner),
        Expr::Mul(_, _) => mul_leaves(ctx, expr)
            .into_iter()
            .all(|factor| nonzero_condition_is_proven_for_symbolic_integration(ctx, factor)),
        Expr::Div(num, den) => {
            nonzero_condition_is_proven_for_symbolic_integration(ctx, num)
                && nonzero_condition_is_proven_for_symbolic_integration(ctx, den)
        }
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Sqrt) =>
        {
            crate::calculus_domain_support::positive_condition_is_proven_over_reals(
                ctx,
                args[0],
                SYMBOLIC_INTEGRATION_DOMAIN_PROOF_DEPTH,
            )
        }
        _ => false,
    }
}

pub fn integrate_symbolic_required_positive_conditions(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Vec<ExprId> {
    let mut conditions = Vec::new();
    collect_required_conditions_from(
        ctx,
        expr,
        var,
        REQUIRED_POSITIVE_CONDITION_COLLECTORS,
        &mut conditions,
    );
    conditions
}

pub(super) fn cancel_matching_factor_from_product(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
) -> Option<ExprId> {
    if integral_factors_match(ctx, numerator, denominator) {
        return Some(ctx.add(Expr::Number(BigRational::one())));
    }

    let mut factors = mul_leaves(ctx, numerator);
    for idx in 0..factors.len() {
        if integral_factors_match(ctx, factors[idx], denominator) {
            factors.remove(idx);
            return Some(build_product_from_factors(ctx, &factors));
        }
    }

    None
}

pub(super) fn integral_factors_match(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    if compare_expr(ctx, left, right) == Ordering::Equal {
        return true;
    }

    let Some(left_radicand) = sqrt_like_radicand(ctx, left) else {
        return false;
    };
    let Some(right_radicand) = sqrt_like_radicand(ctx, right) else {
        return false;
    };
    compare_expr(ctx, left_radicand, right_radicand) == Ordering::Equal
}

pub(super) fn build_product_from_factors(ctx: &mut Context, factors: &[ExprId]) -> ExprId {
    match factors {
        [] => ctx.add(Expr::Number(BigRational::one())),
        [single] => *single,
        _ => build_balanced_mul(ctx, factors),
    }
}

pub(super) fn combine_factor_signs(outer: Sign, factor: Sign) -> Sign {
    if outer == Sign::Neg {
        factor.negate()
    } else {
        factor
    }
}

pub(super) fn chain_antiderivative_supported(builtin: BuiltinFn) -> bool {
    matches!(
        builtin,
        BuiltinFn::Exp | BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Sinh | BuiltinFn::Cosh
    )
}

/// `F(g)` where `F` is the elementary antiderivative of the unary `f`: ∫exp=exp, ∫cos=sin,
/// ∫sin=−cos, ∫sinh=cosh, ∫cosh=sinh.
pub(super) fn unary_chain_antiderivative(
    ctx: &mut Context,
    builtin: BuiltinFn,
    inner: ExprId,
) -> Option<ExprId> {
    Some(match builtin {
        BuiltinFn::Exp => {
            let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
            ctx.add(Expr::Pow(e, inner))
        }
        BuiltinFn::Cos => ctx.call_builtin(BuiltinFn::Sin, vec![inner]),
        BuiltinFn::Sin => {
            let cos = ctx.call_builtin(BuiltinFn::Cos, vec![inner]);
            negate_scalar_expr(ctx, cos)
        }
        BuiltinFn::Sinh => ctx.call_builtin(BuiltinFn::Cosh, vec![inner]),
        BuiltinFn::Cosh => ctx.call_builtin(BuiltinFn::Sinh, vec![inner]),
        _ => return None,
    })
}

/// Strip a top-level sign from `expr`, returning the unsigned core and whether it was negated
/// (handles `Neg(x)` and a negative `Number`).
pub(super) fn usub_strip_top_neg(ctx: &mut Context, expr: ExprId) -> (ExprId, bool) {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => (inner, true),
        Expr::Number(n) if n.is_negative() => (ctx.add(Expr::Number(-n)), true),
        _ => (expr, false),
    }
}
