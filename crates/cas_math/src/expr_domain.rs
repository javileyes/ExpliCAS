//! Domain-oriented expression equivalence and implication helpers.

use crate::expr_extract::extract_abs_argument_view;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Zero;

/// Check if two expressions are equivalent using polynomial comparison.
pub fn exprs_equivalent(ctx: &Context, e1: ExprId, e2: ExprId) -> bool {
    use crate::multipoly::{multipoly_from_expr, PolyBudget};

    if e1 == e2 {
        return true;
    }

    // Quick check: same variable name
    if let (Expr::Variable(name1), Expr::Variable(name2)) = (ctx.get(e1), ctx.get(e2)) {
        if name1 == name2 {
            return true;
        }
    }

    let budget = PolyBudget {
        max_terms: 50,
        max_total_degree: 20,
        max_pow_exp: 10,
    };

    if let (Ok(p1), Ok(p2)) = (
        multipoly_from_expr(ctx, e1, &budget),
        multipoly_from_expr(ctx, e2, &budget),
    ) {
        return p1 == p2;
    }

    false
}

/// Check if two expressions are equivalent up to global sign.
///
/// This treats `E` and `-E` as equivalent, which is useful for predicates like
/// `E != 0` where multiplying by `-1` does not change truth value.
pub fn exprs_equivalent_up_to_sign(ctx: &Context, e1: ExprId, e2: ExprId) -> bool {
    use crate::multipoly::{multipoly_from_expr, PolyBudget};

    if e1 == e2 {
        return true;
    }

    let budget = PolyBudget {
        max_terms: 50,
        max_total_degree: 20,
        max_pow_exp: 10,
    };

    if let (Ok(p1), Ok(p2)) = (
        multipoly_from_expr(ctx, e1, &budget),
        multipoly_from_expr(ctx, e2, &budget),
    ) {
        return p1 == p2 || p1 == p2.neg();
    }

    false
}

/// Check if `source` is `target^(odd positive integer)`.
pub fn is_odd_power_of(ctx: &Context, source: ExprId, target: ExprId) -> bool {
    if let Expr::Pow(base, exp) = ctx.get(source) {
        if let Expr::Number(n) = ctx.get(*exp) {
            if n.is_integer() {
                let exp_int = n.to_integer();
                let two: num_bigint::BigInt = 2.into();
                let zero: num_bigint::BigInt = 0.into();
                let one: num_bigint::BigInt = 1.into();
                if &exp_int % &two == one && exp_int > zero {
                    return exprs_equivalent(ctx, *base, target);
                }
            }
        }
    }
    false
}

/// Check if `expr` is a power whose base is equivalent to `base`.
///
/// This is used for positive-domain implication:
/// if `base > 0`, then `base^exp > 0` for any real exponent `exp`.
pub fn is_positive_power_of_base(ctx: &Context, expr: ExprId, base: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Pow(pow_base, _) if exprs_equivalent(ctx, *pow_base, base)
    )
}

/// Check if `source` is `k*target` where `k > 0`.
pub fn is_positive_multiple_of(ctx: &Context, source: ExprId, target: ExprId) -> bool {
    use crate::multipoly::{multipoly_from_expr, PolyBudget};
    use num_traits::Zero;

    if exprs_equivalent(ctx, source, target) {
        return true;
    }

    if let Expr::Mul(l, r) = ctx.get(source) {
        if let Expr::Number(n) = ctx.get(*l) {
            let zero = num_rational::BigRational::zero();
            if *n > zero && exprs_equivalent(ctx, *r, target) {
                return true;
            }
        }
        if let Expr::Number(n) = ctx.get(*r) {
            let zero = num_rational::BigRational::zero();
            if *n > zero && exprs_equivalent(ctx, *l, target) {
                return true;
            }
        }
    }

    let budget = PolyBudget {
        max_terms: 50,
        max_total_degree: 20,
        max_pow_exp: 10,
    };

    let (Ok(source_poly), Ok(target_poly)) = (
        multipoly_from_expr(ctx, source, &budget),
        multipoly_from_expr(ctx, target, &budget),
    ) else {
        return false;
    };

    if source_poly.is_zero() || target_poly.is_zero() {
        return false;
    }

    let (_, source_primitive) = source_poly.primitive_part();
    let (_, target_primitive) = target_poly.primitive_part();
    source_primitive == target_primitive
}

/// Check whether a known lower-bound expression implies another lower-bound.
///
/// If `source >= 0` and `target = source + c` with `c >= 0`, then
/// `target >= 0`. The same relation also preserves strict positivity when the
/// caller has `source > 0`.
pub fn is_nonnegative_shift_from_source(ctx: &Context, target: ExprId, source: ExprId) -> bool {
    if exprs_equivalent(ctx, target, source) {
        return true;
    }

    let Some(delta) = constant_difference(ctx, target, source) else {
        return false;
    };
    delta >= BigRational::zero()
}

/// Check whether `source >= lower` implies `target >= 0`.
///
/// This recognizes exact affine/polynomial forms where
/// `target = k * (source - lower) + c` with `k > 0` and `c >= 0`.
pub fn lower_bound_implies_nonnegative(
    ctx: &Context,
    target: ExprId,
    source: ExprId,
    lower: &BigRational,
) -> bool {
    lower_bound_implication_margin(ctx, target, source, lower)
        .is_some_and(|margin| margin >= BigRational::zero())
}

/// Check whether `source >= lower` implies `target > 0`.
///
/// This is true for the same exact form as [`lower_bound_implies_nonnegative`]
/// when the remaining margin is strictly positive.
pub fn lower_bound_implies_positive(
    ctx: &Context,
    target: ExprId,
    source: ExprId,
    lower: &BigRational,
) -> bool {
    lower_bound_implication_margin(ctx, target, source, lower)
        .is_some_and(|margin| margin > BigRational::zero())
}

/// Check whether `source >= lower` plus a known non-zero denominator implies a
/// reciprocal-shaped target is positive.
///
/// If `source >= lower` proves `den >= 0`, and `den != 0` is already known,
/// then `1/den > 0`. Likewise, if it proves `-den >= 0`, then `-1/den > 0`.
/// This keeps reciprocal sign cleanup domain-safe by requiring the separate
/// non-zero witness instead of treating a weak lower bound as strict.
pub fn lower_bound_and_nonzero_implies_reciprocal_positive(
    ctx: &Context,
    target: ExprId,
    source: ExprId,
    lower: &BigRational,
    known_nonzeros: &[ExprId],
) -> bool {
    if let Some(sign) =
        reciprocal_product_sign_under_lower_bound(ctx, target, source, lower, known_nonzeros)
    {
        return sign > BigRational::zero();
    }

    let Some((denominator, denominator_sign)) = reciprocal_denominator_sign(ctx, target) else {
        return false;
    };

    if !known_nonzeros
        .iter()
        .any(|known| exprs_equivalent_up_to_sign(ctx, *known, denominator))
    {
        return false;
    }

    lower_bound_signed_implication_margin(ctx, denominator, denominator_sign, source, lower)
        .is_some_and(|margin| margin >= BigRational::zero())
}

/// Check whether a non-zero base proves a reciprocal even-power target positive.
///
/// For real `u != 0`, `1/u^(2k) > 0`. This helper is intentionally limited to
/// positive numeric numerators and positive even integer denominator powers.
pub fn nonzero_implies_reciprocal_even_power_positive(
    ctx: &Context,
    target: ExprId,
    known_nonzero: ExprId,
) -> bool {
    let Some((base, numerator_sign, exponent)) = reciprocal_power_base_sign(ctx, target) else {
        return false;
    };
    let two: BigInt = 2.into();
    let zero: BigInt = 0.into();

    numerator_sign > BigRational::zero()
        && exponent > zero
        && (&exponent % &two).is_zero()
        && exprs_equivalent_up_to_sign(ctx, base, known_nonzero)
}

/// Check whether `positive_expr > 0` implies `bounded_expr >= lower`.
///
/// This is true when `positive_expr` is an exact positive multiple of
/// `bounded_expr - lower`.
pub fn positive_expr_implies_lower_bound(
    ctx: &Context,
    positive_expr: ExprId,
    bounded_expr: ExprId,
    lower: &BigRational,
) -> bool {
    lower_bound_implication_margin(ctx, positive_expr, bounded_expr, lower)
        .is_some_and(|margin| margin.is_zero())
}

/// Check whether a positive absolute-value quotient implies a negated argument
/// is nonnegative/positive.
///
/// For real `u`, `-|u|/u > 0` and `|u|/(-u) > 0` both force `u < 0`, hence
/// `-u > 0`. This helper keeps the implication structural and scoped to the
/// quotient shape that appears after domain-safe log contraction.
pub fn positive_abs_quotient_implies_negated_arg_positive(
    ctx: &Context,
    positive_expr: ExprId,
    negated_target: ExprId,
) -> bool {
    let Some(target_arg) = negated_arg(ctx, negated_target) else {
        return false;
    };

    positive_abs_quotient_matches_arg(ctx, positive_expr, target_arg)
}

fn positive_abs_quotient_matches_arg(ctx: &Context, expr: ExprId, target_arg: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Div(num, den) => {
            is_negated_abs_of(ctx, *num, target_arg) && exprs_equivalent(ctx, *den, target_arg)
                || is_abs_of(ctx, *num, target_arg)
                    && negated_arg(ctx, *den)
                        .is_some_and(|den_arg| exprs_equivalent(ctx, den_arg, target_arg))
        }
        Expr::Neg(inner) => match ctx.get(*inner) {
            Expr::Div(num, den) => {
                is_abs_of(ctx, *num, target_arg) && exprs_equivalent(ctx, *den, target_arg)
            }
            _ => false,
        },
        _ => false,
    }
}

fn is_negated_abs_of(ctx: &Context, expr: ExprId, target_arg: ExprId) -> bool {
    negated_arg(ctx, expr).is_some_and(|inner| is_abs_of(ctx, inner, target_arg))
}

fn negated_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Neg(inner) => Some(*inner),
        _ => None,
    }
}

fn lower_bound_implication_margin(
    ctx: &Context,
    target: ExprId,
    source: ExprId,
    lower: &BigRational,
) -> Option<BigRational> {
    lower_bound_signed_implication_margin(
        ctx,
        target,
        BigRational::from_integer(1.into()),
        source,
        lower,
    )
}

fn lower_bound_signed_implication_margin(
    ctx: &Context,
    target: ExprId,
    target_sign: BigRational,
    source: ExprId,
    lower: &BigRational,
) -> Option<BigRational> {
    use crate::multipoly::{multipoly_from_expr, MultiPoly, PolyBudget};

    let budget = PolyBudget {
        max_terms: 50,
        max_total_degree: 20,
        max_pow_exp: 10,
    };

    let target_poly = multipoly_from_expr(ctx, target, &budget)
        .ok()?
        .mul_scalar(&target_sign);
    let source_poly = multipoly_from_expr(ctx, source, &budget).ok()?;
    let mut vars = target_poly.vars.clone();
    for var in &source_poly.vars {
        if !vars.contains(var) {
            vars.push(var.clone());
        }
    }
    vars.sort();

    let target_poly = target_poly.align_vars(&vars);
    let lower_poly = MultiPoly::from_const(lower.clone()).align_vars(&vars);
    let base_poly = source_poly.align_vars(&vars).sub(&lower_poly).ok()?;
    if base_poly.is_zero() {
        return None;
    }

    let mut ratio: Option<BigRational> = None;
    for (base_coeff, mono) in &base_poly.terms {
        if mono.iter().all(|&exp| exp == 0) {
            continue;
        }
        let target_coeff = target_poly
            .terms
            .iter()
            .find(|(_, target_mono)| target_mono == mono)
            .map(|(coeff, _)| coeff.clone())
            .unwrap_or_else(BigRational::zero);
        let current = target_coeff / base_coeff;
        if current <= BigRational::zero() {
            return None;
        }
        match &ratio {
            Some(existing) if existing != &current => return None,
            Some(_) => {}
            None => ratio = Some(current),
        }
    }

    let ratio = ratio?;
    for (target_coeff, mono) in &target_poly.terms {
        if mono.iter().all(|&exp| exp == 0) {
            continue;
        }
        let base_coeff = base_poly
            .terms
            .iter()
            .find(|(_, base_mono)| base_mono == mono)
            .map(|(coeff, _)| coeff.clone())
            .unwrap_or_else(BigRational::zero);
        if target_coeff != &(ratio.clone() * base_coeff) {
            return None;
        }
    }

    let scaled_base = base_poly.mul_scalar(&ratio);
    target_poly.sub(&scaled_base).ok()?.constant_value()
}

fn reciprocal_denominator_sign(ctx: &Context, expr: ExprId) -> Option<(ExprId, BigRational)> {
    let (denominator, numerator_sign, exponent) = reciprocal_power_base_sign(ctx, expr)?;
    let two: BigInt = 2.into();
    let one: BigInt = 1.into();

    if (&exponent % &two) != one {
        return None;
    }

    Some((denominator, numerator_sign))
}

fn reciprocal_product_sign_under_lower_bound(
    ctx: &Context,
    expr: ExprId,
    source: ExprId,
    lower: &BigRational,
    known_nonzeros: &[ExprId],
) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Neg(inner) => Some(-reciprocal_product_sign_under_lower_bound(
            ctx,
            *inner,
            source,
            lower,
            known_nonzeros,
        )?),
        Expr::Div(num, den) => {
            let Expr::Number(numerator) = ctx.get(*num) else {
                return None;
            };
            let numerator_sign = rational_sign(numerator)?;
            let denominator_sign = denominator_product_sign_under_lower_bound(
                ctx,
                *den,
                source,
                lower,
                known_nonzeros,
            )?;
            Some(numerator_sign * denominator_sign)
        }
        _ => None,
    }
}

fn denominator_product_sign_under_lower_bound(
    ctx: &Context,
    denominator: ExprId,
    source: ExprId,
    lower: &BigRational,
    known_nonzeros: &[ExprId],
) -> Option<BigRational> {
    match ctx.get(denominator) {
        Expr::Number(value) => rational_sign(value),
        Expr::Neg(inner) => Some(-denominator_product_sign_under_lower_bound(
            ctx,
            *inner,
            source,
            lower,
            known_nonzeros,
        )?),
        Expr::Mul(left, right) => {
            let left_sign = denominator_product_sign_under_lower_bound(
                ctx,
                *left,
                source,
                lower,
                known_nonzeros,
            )?;
            let right_sign = denominator_product_sign_under_lower_bound(
                ctx,
                *right,
                source,
                lower,
                known_nonzeros,
            )?;
            Some(left_sign * right_sign)
        }
        Expr::Pow(base, exp) => {
            let Expr::Number(n) = ctx.get(*exp) else {
                return None;
            };
            if !n.is_integer() {
                return None;
            }
            let exponent = n.to_integer();
            if exponent <= 0.into() {
                return None;
            }
            factor_power_sign_under_lower_bound(
                ctx,
                *base,
                &exponent,
                source,
                lower,
                known_nonzeros,
            )
        }
        _ => factor_power_sign_under_lower_bound(
            ctx,
            denominator,
            &BigInt::from(1),
            source,
            lower,
            known_nonzeros,
        ),
    }
}

fn factor_power_sign_under_lower_bound(
    ctx: &Context,
    base: ExprId,
    exponent: &BigInt,
    source: ExprId,
    lower: &BigRational,
    known_nonzeros: &[ExprId],
) -> Option<BigRational> {
    if !known_nonzeros
        .iter()
        .any(|known| exprs_equivalent_up_to_sign(ctx, *known, base))
    {
        return None;
    }

    let two: BigInt = 2.into();
    if (exponent % &two).is_zero() {
        return Some(BigRational::from_integer(1.into()));
    }

    if lower_bound_signed_implication_margin(
        ctx,
        base,
        BigRational::from_integer(1.into()),
        source,
        lower,
    )
    .is_some_and(|margin| margin >= BigRational::zero())
    {
        return Some(BigRational::from_integer(1.into()));
    }

    if lower_bound_signed_implication_margin(
        ctx,
        base,
        BigRational::from_integer((-1).into()),
        source,
        lower,
    )
    .is_some_and(|margin| margin >= BigRational::zero())
    {
        return Some(BigRational::from_integer((-1).into()));
    }

    None
}

fn rational_sign(value: &BigRational) -> Option<BigRational> {
    if value > &BigRational::zero() {
        Some(BigRational::from_integer(1.into()))
    } else if value < &BigRational::zero() {
        Some(BigRational::from_integer((-1).into()))
    } else {
        None
    }
}

fn reciprocal_power_base_sign(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational, BigInt)> {
    match ctx.get(expr) {
        Expr::Neg(inner) => {
            let (denominator, sign, exponent) = reciprocal_power_base_sign(ctx, *inner)?;
            Some((denominator, -sign, exponent))
        }
        Expr::Div(num, den) => {
            let Expr::Number(numerator) = ctx.get(*num) else {
                return None;
            };
            let numerator_sign = if numerator > &BigRational::zero() {
                BigRational::from_integer(1.into())
            } else if numerator < &BigRational::zero() {
                BigRational::from_integer((-1).into())
            } else {
                return None;
            };
            let (base, exponent) = denominator_power_base(ctx, *den)?;
            Some((base, numerator_sign, exponent))
        }
        _ => None,
    }
}

fn denominator_power_base(ctx: &Context, denominator: ExprId) -> Option<(ExprId, BigInt)> {
    match ctx.get(denominator) {
        Expr::Pow(base, exp) => {
            let Expr::Number(n) = ctx.get(*exp) else {
                return None;
            };
            if !n.is_integer() {
                return None;
            }
            let exponent = n.to_integer();
            if exponent <= 0.into() {
                return None;
            }
            Some((*base, exponent))
        }
        _ => Some((denominator, 1.into())),
    }
}

fn constant_difference(ctx: &Context, left: ExprId, right: ExprId) -> Option<BigRational> {
    use crate::multipoly::{multipoly_from_expr, PolyBudget};

    let budget = PolyBudget {
        max_terms: 50,
        max_total_degree: 20,
        max_pow_exp: 10,
    };

    let left_poly = multipoly_from_expr(ctx, left, &budget).ok()?;
    let right_poly = multipoly_from_expr(ctx, right, &budget).ok()?;
    let mut vars = left_poly.vars.clone();
    for var in &right_poly.vars {
        if !vars.contains(var) {
            vars.push(var.clone());
        }
    }
    vars.sort();

    let left_poly = left_poly.align_vars(&vars);
    let right_poly = right_poly.align_vars(&vars);
    left_poly.sub(&right_poly).ok()?.constant_value()
}

/// Check if `expr` is `abs(inner)`.
pub fn is_abs_of(ctx: &Context, expr: ExprId, inner: ExprId) -> bool {
    if let Some(arg) = extract_abs_argument_view(ctx, expr) {
        return exprs_equivalent(ctx, arg, inner);
    }
    false
}

/// Check if a positive product condition is dominated by known positive bases.
pub fn is_product_dominated_by_positives(
    ctx: &Context,
    prod_expr: ExprId,
    known_positives: &[ExprId],
) -> bool {
    if positive_component_count(ctx, prod_expr) < 2 {
        return false;
    }

    is_positive_expr_dominated_by_positives(ctx, prod_expr, known_positives)
}

fn positive_component_count(ctx: &Context, expr: ExprId) -> usize {
    match ctx.get(expr) {
        Expr::Mul(l, r) | Expr::Div(l, r) => {
            positive_component_count(ctx, *l) + positive_component_count(ctx, *r)
        }
        Expr::Pow(_, exp) => match ctx.get(*exp) {
            Expr::Number(n) if n.is_integer() && n.to_integer() > 0.into() => 1,
            _ => 1,
        },
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Abs) && args.len() == 1 =>
        {
            positive_component_count(ctx, args[0])
        }
        _ => 1,
    }
}

fn is_positive_expr_dominated_by_positives(
    ctx: &Context,
    expr: ExprId,
    known_positives: &[ExprId],
) -> bool {
    if known_positives
        .iter()
        .any(|pos| exprs_equivalent(ctx, expr, *pos))
    {
        return true;
    }

    match ctx.get(expr) {
        Expr::Mul(l, r) | Expr::Div(l, r) => {
            is_positive_expr_dominated_by_positives(ctx, *l, known_positives)
                && is_positive_expr_dominated_by_positives(ctx, *r, known_positives)
        }
        Expr::Pow(base, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() {
                    let exp_int = n.to_integer();
                    let zero: num_bigint::BigInt = 0.into();
                    if exp_int > zero {
                        let two: num_bigint::BigInt = 2.into();
                        if (&exp_int % &two) == 0.into() {
                            return known_positives.iter().any(|pos| {
                                exprs_equivalent(ctx, *base, *pos) || is_abs_of(ctx, *pos, *base)
                            }) || is_positive_expr_dominated_by_positives(
                                ctx,
                                *base,
                                known_positives,
                            );
                        }

                        return is_positive_expr_dominated_by_positives(
                            ctx,
                            *base,
                            known_positives,
                        );
                    }
                }
            }
            false
        }
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Abs) && args.len() == 1 =>
        {
            is_positive_expr_dominated_by_positives(ctx, args[0], known_positives)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use cas_parser::parse;

    #[test]
    fn polynomial_equivalence_detects_reordered_sum() {
        let mut ctx = Context::new();
        let e1 = parse("x+1", &mut ctx).expect("parse");
        let e2 = parse("1+x", &mut ctx).expect("parse");
        assert!(exprs_equivalent(&ctx, e1, e2));
    }

    #[test]
    fn equivalence_up_to_sign_detects_negation() {
        let mut ctx = Context::new();
        let e1 = parse("x+1", &mut ctx).expect("parse");
        let e2 = parse("-(x+1)", &mut ctx).expect("parse");
        let e3 = parse("x+2", &mut ctx).expect("parse");

        assert!(exprs_equivalent_up_to_sign(&ctx, e1, e2));
        assert!(!exprs_equivalent_up_to_sign(&ctx, e1, e3));
    }

    #[test]
    fn odd_power_detection_works() {
        let mut ctx = Context::new();
        let source = parse("b^3", &mut ctx).expect("parse");
        let target = parse("b", &mut ctx).expect("parse");
        let even = parse("b^4", &mut ctx).expect("parse");
        assert!(is_odd_power_of(&ctx, source, target));
        assert!(!is_odd_power_of(&ctx, even, target));
    }

    #[test]
    fn positive_power_of_base_detects_symbolic_exponent() {
        let mut ctx = Context::new();
        let base = parse("x", &mut ctx).expect("parse");
        let symbolic_pow = parse("x^n", &mut ctx).expect("parse");
        let other = parse("y", &mut ctx).expect("parse");

        assert!(is_positive_power_of_base(&ctx, symbolic_pow, base));
        assert!(!is_positive_power_of_base(&ctx, symbolic_pow, other));
    }

    #[test]
    fn positive_multiple_detection_works() {
        let mut ctx = Context::new();
        let source = parse("4*x", &mut ctx).expect("parse");
        let target = parse("x", &mut ctx).expect("parse");
        let neg_source = parse("-2*x", &mut ctx).expect("parse");
        assert!(is_positive_multiple_of(&ctx, source, target));
        assert!(!is_positive_multiple_of(&ctx, neg_source, target));
    }

    #[test]
    fn positive_multiple_detection_matches_polynomial_scalar_content() {
        let mut ctx = Context::new();
        let source = parse("-4*x^2 - 4*x", &mut ctx).expect("parse source");
        let target = parse("-x^2 - x", &mut ctx).expect("parse target");
        let opposite = parse("4*x^2 + 4*x", &mut ctx).expect("parse opposite");

        assert!(is_positive_multiple_of(&ctx, source, target));
        assert!(!is_positive_multiple_of(&ctx, opposite, target));
    }

    #[test]
    fn abs_detection_works() {
        let mut ctx = Context::new();
        let abs_expr = parse("abs(x)", &mut ctx).expect("parse");
        let x = parse("x", &mut ctx).expect("parse");
        let y = parse("y", &mut ctx).expect("parse");
        assert!(is_abs_of(&ctx, abs_expr, x));
        assert!(!is_abs_of(&ctx, abs_expr, y));
    }

    #[test]
    fn product_dominance_requires_all_factors() {
        let mut ctx = Context::new();
        let product = parse("a^2*b^3", &mut ctx).expect("parse");
        let a = parse("a", &mut ctx).expect("parse");
        let b = parse("b", &mut ctx).expect("parse");
        assert!(is_product_dominated_by_positives(&ctx, product, &[a, b]));
        assert!(!is_product_dominated_by_positives(&ctx, product, &[a]));
    }

    #[test]
    fn quotient_dominance_accepts_even_power_covered_by_abs() {
        let mut ctx = Context::new();
        let quotient = parse("y*x^2/(t*z)", &mut ctx).expect("parse");
        let y = parse("y", &mut ctx).expect("parse");
        let t = parse("t", &mut ctx).expect("parse");
        let z = parse("z", &mut ctx).expect("parse");
        let abs_x = parse("abs(x)", &mut ctx).expect("parse");

        assert!(is_product_dominated_by_positives(
            &ctx,
            quotient,
            &[y, t, z, abs_x]
        ));
    }
}
