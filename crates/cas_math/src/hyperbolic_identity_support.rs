use crate::expr_predicates::is_two_expr;
use crate::trig_roots_flatten::{extract_double_angle_arg_relaxed, extract_triple_angle_arg};
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HyperbolicPythagoreanValue {
    One,
    NegativeOne,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HyperbolicIdentityRewriteKind {
    PythagoreanOne,
    PythagoreanNegativeOne,
    SinhCoshToTanh,
    CoshSinhToReciprocalTanh,
    TanhToSinhCosh,
    SinhDoubleAngleExpansion,
    TanhDoubleAngleExpansion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyperbolicIdentityRewrite {
    pub rewritten: ExprId,
    pub kind: HyperbolicIdentityRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SinhCoshToExpRewriteKind {
    Sum,
    CoshMinusSinh,
    SinhMinusCosh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SinhCoshToExpRewrite {
    pub rewritten: ExprId,
    pub kind: SinhCoshToExpRewriteKind,
}

fn e_pow(ctx: &mut Context, arg: ExprId) -> ExprId {
    let e = ctx.add(Expr::Constant(Constant::E));
    ctx.add(Expr::Pow(e, arg))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HyperbolicDoubleAngleRewriteKind {
    Sum,
    SubChain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyperbolicDoubleAngleRewrite {
    pub rewritten: ExprId,
    pub kind: HyperbolicDoubleAngleRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyperbolicTanhPythagoreanRewrite {
    pub rewritten: ExprId,
}

fn extract_coeff_tanh_pow2(ctx: &Context, term: ExprId) -> Option<(BigRational, ExprId)> {
    let mut coeff = BigRational::one();
    let mut working = term;

    if let Expr::Neg(inner) = ctx.get(term) {
        coeff = -coeff;
        working = *inner;
    }

    let mut tanh_arg: Option<ExprId> = None;
    for factor in crate::expr_nary::mul_leaves(ctx, working) {
        if let Expr::Number(n) = ctx.get(factor) {
            coeff *= n.clone();
            continue;
        }

        let Expr::Pow(base, exp) = ctx.get(factor) else {
            return None;
        };
        if !is_two_expr(ctx, *exp) {
            return None;
        }
        let Expr::Function(fn_id, args) = ctx.get(*base) else {
            return None;
        };
        if !ctx.is_builtin(*fn_id, BuiltinFn::Tanh) || args.len() != 1 || tanh_arg.is_some() {
            return None;
        }
        tanh_arg = Some(args[0]);
    }

    Some((coeff, tanh_arg?))
}

fn multiply_numeric_coeff(ctx: &mut Context, coeff: &BigRational, body: ExprId) -> ExprId {
    if coeff.is_one() {
        return body;
    }
    let coeff_expr = ctx.add(Expr::Number(coeff.clone()));
    ctx.add(Expr::Mul(coeff_expr, body))
}

struct SignedFactors {
    negative: bool,
    factors: Vec<ExprId>,
}

fn push_normalized_factor(
    ctx: &mut Context,
    mut factor: ExprId,
    negative: &mut bool,
    factors: &mut Vec<ExprId>,
) -> bool {
    loop {
        match ctx.get(factor).clone() {
            Expr::Neg(inner) => {
                *negative = !*negative;
                factor = inner;
            }
            Expr::Number(value) => {
                if value.is_zero() {
                    return false;
                }
                if value < BigRational::zero() {
                    *negative = !*negative;
                    let positive = -value;
                    if !positive.is_one() {
                        factors.push(ctx.add(Expr::Number(positive)));
                    }
                } else if !value.is_one() {
                    factors.push(factor);
                }
                return true;
            }
            _ => {
                factors.push(factor);
                return true;
            }
        }
    }
}

fn signed_factors_for_term(
    ctx: &mut Context,
    term: ExprId,
    sign: crate::expr_nary::Sign,
) -> Option<SignedFactors> {
    let mut negative = sign == crate::expr_nary::Sign::Neg;
    let mut factors = Vec::new();
    for factor in crate::expr_nary::mul_leaves(ctx, term) {
        if !push_normalized_factor(ctx, factor, &mut negative, &mut factors) {
            return None;
        }
    }
    Some(SignedFactors { negative, factors })
}

fn signed_factors_to_expr(ctx: &mut Context, factors: &[ExprId], negative: bool) -> ExprId {
    let product = if factors.is_empty() {
        ctx.num(1)
    } else if factors.len() == 1 {
        factors[0]
    } else {
        crate::expr_nary::build_balanced_mul(ctx, factors)
    };

    if negative {
        match ctx.get(product).clone() {
            Expr::Number(value) => ctx.add(Expr::Number(-value)),
            Expr::Neg(inner) => inner,
            _ => ctx.add(Expr::Neg(product)),
        }
    } else {
        product
    }
}

fn tanh_squared_arg(ctx: &Context, factor: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(factor) else {
        return None;
    };
    if !is_two_expr(ctx, *exp) {
        return None;
    }
    let Expr::Function(fn_id, args) = ctx.get(*base) else {
        return None;
    };
    (ctx.is_builtin(*fn_id, BuiltinFn::Tanh) && args.len() == 1).then_some(args[0])
}

fn is_integer_number(ctx: &Context, expr: ExprId, value: i64) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(n) if *n == BigRational::from_integer(value.into())
    )
}

fn hyperbolic_power_arg(
    ctx: &Context,
    factor: ExprId,
    builtin: BuiltinFn,
    power: i64,
) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(factor) else {
        return None;
    };
    if !is_integer_number(ctx, *exp, power) {
        return None;
    }
    let Expr::Function(fn_id, args) = ctx.get(*base) else {
        return None;
    };
    (ctx.is_builtin(*fn_id, builtin) && args.len() == 1).then_some(args[0])
}

fn cosh_tanh_fourth_denominator(ctx: &Context, den: ExprId) -> Option<(ExprId, BigRational)> {
    let mut cosh_arg = None;
    let mut tanh_arg = None;
    let mut coefficient = BigRational::one();

    for factor in crate::expr_nary::mul_leaves(ctx, den) {
        if let Expr::Number(value) = ctx.get(factor) {
            if value.is_zero() {
                return None;
            }
            coefficient *= value.clone();
            continue;
        }
        if let Some(arg) = hyperbolic_power_arg(ctx, factor, BuiltinFn::Cosh, 2) {
            if cosh_arg.replace(arg).is_some() {
                return None;
            }
            continue;
        }
        if let Some(arg) = hyperbolic_power_arg(ctx, factor, BuiltinFn::Tanh, 4) {
            if tanh_arg.replace(arg).is_some() {
                return None;
            }
            continue;
        }
        return None;
    }

    let cosh_arg = cosh_arg?;
    let tanh_arg = tanh_arg?;
    (compare_expr(ctx, cosh_arg, tanh_arg) == Ordering::Equal).then_some((cosh_arg, coefficient))
}

fn cosh_tanh_square_denominator(ctx: &Context, den: ExprId) -> Option<(ExprId, BigRational)> {
    let mut cosh_arg = None;
    let mut tanh_arg = None;
    let mut coefficient = BigRational::one();

    for factor in crate::expr_nary::mul_leaves(ctx, den) {
        if let Expr::Number(value) = ctx.get(factor) {
            if value.is_zero() {
                return None;
            }
            coefficient *= value.clone();
            continue;
        }
        if let Some(arg) = hyperbolic_power_arg(ctx, factor, BuiltinFn::Cosh, 2) {
            if cosh_arg.replace(arg).is_some() {
                return None;
            }
            continue;
        }
        if let Some(arg) = hyperbolic_power_arg(ctx, factor, BuiltinFn::Tanh, 2) {
            if tanh_arg.replace(arg).is_some() {
                return None;
            }
            continue;
        }
        return None;
    }

    let cosh_arg = cosh_arg?;
    let tanh_arg = tanh_arg?;
    (compare_expr(ctx, cosh_arg, tanh_arg) == Ordering::Equal).then_some((cosh_arg, coefficient))
}

fn hyperbolic_power_denominator_arg(
    ctx: &Context,
    den: ExprId,
    builtin: BuiltinFn,
    power: i64,
) -> Option<(ExprId, BigRational)> {
    let mut arg = None;
    let mut coefficient = BigRational::one();

    for factor in crate::expr_nary::mul_leaves(ctx, den) {
        if let Expr::Number(value) = ctx.get(factor) {
            if value.is_zero() {
                return None;
            }
            coefficient *= value.clone();
            continue;
        }
        if let Some(current_arg) = hyperbolic_power_arg(ctx, factor, builtin, power) {
            if arg.replace(current_arg).is_some() {
                return None;
            }
            continue;
        }
        return None;
    }

    Some((arg?, coefficient))
}

fn factor_multisets_match(ctx: &Context, left: &[ExprId], right: &[ExprId]) -> bool {
    if left.len() != right.len() {
        return false;
    }

    let mut used = vec![false; right.len()];
    'outer: for left_factor in left {
        for (idx, right_factor) in right.iter().enumerate() {
            if !used[idx] && compare_expr(ctx, *left_factor, *right_factor) == Ordering::Equal {
                used[idx] = true;
                continue 'outer;
            }
        }
        return false;
    }

    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CschFourthVerificationKind {
    CoshTanhFourth,
    SinhSquare,
    SinhFourth,
}

struct CschFourthVerificationTerm {
    kind: CschFourthVerificationKind,
    arg: ExprId,
    negative: bool,
    factors: Vec<ExprId>,
}

fn csch_verification_term_with_denominator_coefficient(
    ctx: &mut Context,
    mut signed: SignedFactors,
    kind: CschFourthVerificationKind,
    arg: ExprId,
    denominator_coefficient: BigRational,
) -> Option<CschFourthVerificationTerm> {
    if denominator_coefficient.is_zero() {
        return None;
    }

    let mut numerator_coefficient = BigRational::one();
    let mut factors = Vec::new();
    for factor in signed.factors {
        if let Expr::Number(value) = ctx.get(factor) {
            numerator_coefficient *= value.clone();
        } else {
            factors.push(factor);
        }
    }

    let mut coefficient = numerator_coefficient / denominator_coefficient;
    if coefficient < BigRational::zero() {
        signed.negative = !signed.negative;
        coefficient = -coefficient;
    }
    if !coefficient.is_one() {
        let coefficient_expr = ctx.add(Expr::Number(coefficient));
        factors.insert(0, coefficient_expr);
    }

    Some(CschFourthVerificationTerm {
        kind,
        arg,
        negative: signed.negative,
        factors,
    })
}

fn scaled_division_factors_for_csch_verification_term(
    ctx: &mut Context,
    term: ExprId,
    sign: crate::expr_nary::Sign,
) -> Option<(SignedFactors, ExprId)> {
    if let Expr::Div(num, den) = ctx.get(term).clone() {
        return Some((signed_factors_for_term(ctx, num, sign)?, den));
    }

    let mut division = None;
    let mut cofactor_factors = Vec::new();
    for factor in crate::expr_nary::mul_leaves(ctx, term) {
        if let Expr::Div(num, den) = ctx.get(factor).clone() {
            if division.replace((num, den)).is_some() {
                return None;
            }
            continue;
        }
        cofactor_factors.push(factor);
    }

    let (num, den) = division?;
    let mut signed = signed_factors_for_term(ctx, num, sign)?;
    for factor in cofactor_factors {
        if !push_normalized_factor(ctx, factor, &mut signed.negative, &mut signed.factors) {
            return None;
        }
    }

    Some((signed, den))
}

fn extract_csch_fourth_verification_term(
    ctx: &mut Context,
    term: ExprId,
    sign: crate::expr_nary::Sign,
) -> Option<CschFourthVerificationTerm> {
    let (signed, den) = scaled_division_factors_for_csch_verification_term(ctx, term, sign)?;

    if let Some((arg, denominator_coefficient)) = cosh_tanh_fourth_denominator(ctx, den) {
        return csch_verification_term_with_denominator_coefficient(
            ctx,
            signed,
            CschFourthVerificationKind::CoshTanhFourth,
            arg,
            denominator_coefficient,
        );
    }
    if let Some((arg, denominator_coefficient)) =
        hyperbolic_power_denominator_arg(ctx, den, BuiltinFn::Sinh, 2)
    {
        return csch_verification_term_with_denominator_coefficient(
            ctx,
            signed,
            CschFourthVerificationKind::SinhSquare,
            arg,
            denominator_coefficient,
        );
    }
    if let Some((arg, denominator_coefficient)) = cosh_tanh_square_denominator(ctx, den) {
        return csch_verification_term_with_denominator_coefficient(
            ctx,
            signed,
            CschFourthVerificationKind::SinhSquare,
            arg,
            denominator_coefficient,
        );
    }
    if let Some((arg, denominator_coefficient)) =
        hyperbolic_power_denominator_arg(ctx, den, BuiltinFn::Sinh, 4)
    {
        return csch_verification_term_with_denominator_coefficient(
            ctx,
            signed,
            CschFourthVerificationKind::SinhFourth,
            arg,
            denominator_coefficient,
        );
    }

    None
}

fn signed_add_term_to_expr(
    ctx: &mut Context,
    term: ExprId,
    sign: crate::expr_nary::Sign,
) -> ExprId {
    match sign {
        crate::expr_nary::Sign::Pos => term,
        crate::expr_nary::Sign::Neg => ctx.add(Expr::Neg(term)),
    }
}

fn build_add_from_terms(ctx: &mut Context, terms: &[ExprId]) -> ExprId {
    if terms.is_empty() {
        return ctx.num(0);
    }
    let mut acc = terms[0];
    for &term in terms.iter().skip(1) {
        acc = ctx.add(Expr::Add(acc, term));
    }
    acc
}

/// Detects the bounded verifier residual:
/// `1/(cosh(u)^2*tanh(u)^4) - 1/sinh(u)^4 - 1/sinh(u)^2 -> 0`.
///
/// The matcher also accepts a shared multiplicative cofactor and the opposite
/// sign orientation. It avoids expanding hyperbolic sums while verifying
/// `coth(u) - coth(u)^3/3` antiderivatives.
pub fn try_rewrite_csch_fourth_tanh_verification_add_chain(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HyperbolicTanhPythagoreanRewrite> {
    let terms = crate::expr_nary::add_terms_signed(ctx, expr);
    if terms.len() < 3 {
        return None;
    }

    let parsed_terms = terms
        .iter()
        .map(|&(term, sign)| extract_csch_fourth_verification_term(ctx, term, sign))
        .collect::<Vec<_>>();

    for (cosh_idx, cosh_term) in parsed_terms.iter().enumerate() {
        let Some(cosh_term) = cosh_term else {
            continue;
        };
        if cosh_term.kind != CschFourthVerificationKind::CoshTanhFourth {
            continue;
        }

        let mut sinh_square_idx = None;
        let mut sinh_fourth_idx = None;
        for (idx, candidate) in parsed_terms.iter().enumerate() {
            if idx == cosh_idx {
                continue;
            }
            let Some(candidate) = candidate else {
                continue;
            };
            if candidate.negative == cosh_term.negative
                || compare_expr(ctx, candidate.arg, cosh_term.arg) != Ordering::Equal
                || !factor_multisets_match(ctx, &candidate.factors, &cosh_term.factors)
            {
                continue;
            }
            match candidate.kind {
                CschFourthVerificationKind::SinhSquare if sinh_square_idx.is_none() => {
                    sinh_square_idx = Some(idx);
                }
                CschFourthVerificationKind::SinhFourth if sinh_fourth_idx.is_none() => {
                    sinh_fourth_idx = Some(idx);
                }
                _ => {}
            }
        }

        let (Some(sinh_square_idx), Some(sinh_fourth_idx)) = (sinh_square_idx, sinh_fourth_idx)
        else {
            continue;
        };

        let mut remaining = Vec::new();
        for (idx, &(term, sign)) in terms.iter().enumerate() {
            if idx != cosh_idx && idx != sinh_square_idx && idx != sinh_fourth_idx {
                remaining.push(signed_add_term_to_expr(ctx, term, sign));
            }
        }
        let rewritten = build_add_from_terms(ctx, &remaining);
        return Some(HyperbolicTanhPythagoreanRewrite { rewritten });
    }

    None
}

fn try_rewrite_tanh_pythagorean_common_cofactor(
    ctx: &mut Context,
    terms: &[(ExprId, crate::expr_nary::Sign)],
) -> Option<HyperbolicTanhPythagoreanRewrite> {
    let mut signed_terms = Vec::new();
    let mut tanh_terms = Vec::new();

    for (idx, &(term, sign)) in terms.iter().enumerate() {
        let signed = signed_factors_for_term(ctx, term, sign)?;
        let mut tanh_idx = None;
        let mut tanh_arg = None;
        for (factor_idx, factor) in signed.factors.iter().enumerate() {
            if let Some(arg) = tanh_squared_arg(ctx, *factor) {
                if tanh_idx.is_some() {
                    tanh_idx = None;
                    tanh_arg = None;
                    break;
                }
                tanh_idx = Some(factor_idx);
                tanh_arg = Some(arg);
            }
        }

        if let (Some(tanh_idx), Some(arg)) = (tanh_idx, tanh_arg) {
            let residual_factors = signed
                .factors
                .iter()
                .enumerate()
                .filter_map(|(factor_idx, factor)| (factor_idx != tanh_idx).then_some(*factor))
                .collect::<Vec<_>>();
            tanh_terms.push((idx, signed.negative, residual_factors, arg));
        }

        signed_terms.push(signed);
    }

    for (tanh_term_idx, tanh_negative, residual_factors, arg) in tanh_terms {
        for (base_idx, base) in signed_terms.iter().enumerate() {
            if base_idx == tanh_term_idx || base.negative == tanh_negative {
                continue;
            }
            if !factor_multisets_match(ctx, &base.factors, &residual_factors) {
                continue;
            }

            let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
            let two = ctx.num(2);
            let cosh_squared = ctx.add(Expr::Pow(cosh, two));
            let one = ctx.num(1);
            let sech_squared = ctx.add(Expr::Div(one, cosh_squared));
            let cofactor = signed_factors_to_expr(ctx, &base.factors, base.negative);
            let replacement = if matches!(ctx.get(cofactor), Expr::Number(n) if n.is_one()) {
                sech_squared
            } else {
                ctx.add(Expr::Mul(cofactor, sech_squared))
            };

            let mut new_terms: Vec<ExprId> = Vec::new();
            for (idx, signed) in signed_terms.iter().enumerate() {
                if idx != base_idx && idx != tanh_term_idx {
                    new_terms.push(signed_factors_to_expr(
                        ctx,
                        &signed.factors,
                        signed.negative,
                    ));
                }
            }
            new_terms.push(replacement);

            let rewritten = if new_terms.len() == 1 {
                new_terms[0]
            } else {
                let mut acc = new_terms[0];
                for &term in new_terms.iter().skip(1) {
                    acc = ctx.add(Expr::Add(acc, term));
                }
                acc
            };
            return Some(HyperbolicTanhPythagoreanRewrite { rewritten });
        }
    }

    None
}

/// Detects hyperbolic Pythagorean subtraction forms:
/// - `cosh(x)^2 - sinh(x)^2` -> `1`
/// - `sinh(x)^2 - cosh(x)^2` -> `-1`
pub fn detect_hyperbolic_pythagorean_sub(
    ctx: &Context,
    expr: ExprId,
) -> Option<HyperbolicPythagoreanValue> {
    let Expr::Sub(l, r) = ctx.get(expr) else {
        return None;
    };
    let (l, r) = (*l, *r);

    let (Expr::Pow(l_base, l_exp), Expr::Pow(r_base, r_exp)) = (ctx.get(l), ctx.get(r)) else {
        return None;
    };
    let (l_base, l_exp, r_base, r_exp) = (*l_base, *l_exp, *r_base, *r_exp);

    if !is_two_expr(ctx, l_exp) || !is_two_expr(ctx, r_exp) {
        return None;
    }

    let (Expr::Function(l_fn, l_args), Expr::Function(r_fn, r_args)) =
        (ctx.get(l_base), ctx.get(r_base))
    else {
        return None;
    };

    if l_args.len() != 1 || r_args.len() != 1 {
        return None;
    }
    if compare_expr(ctx, l_args[0], r_args[0]) != Ordering::Equal {
        return None;
    }

    if ctx.is_builtin(*l_fn, BuiltinFn::Cosh) && ctx.is_builtin(*r_fn, BuiltinFn::Sinh) {
        return Some(HyperbolicPythagoreanValue::One);
    }

    if ctx.is_builtin(*l_fn, BuiltinFn::Sinh) && ctx.is_builtin(*r_fn, BuiltinFn::Cosh) {
        return Some(HyperbolicPythagoreanValue::NegativeOne);
    }

    None
}

/// Detect and rewrite:
/// - `cosh(x)^2 - sinh(x)^2` -> `1`
/// - `sinh(x)^2 - cosh(x)^2` -> `-1`
pub fn try_rewrite_hyperbolic_pythagorean_sub_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HyperbolicIdentityRewrite> {
    match detect_hyperbolic_pythagorean_sub(ctx, expr)? {
        HyperbolicPythagoreanValue::One => Some(HyperbolicIdentityRewrite {
            rewritten: ctx.num(1),
            kind: HyperbolicIdentityRewriteKind::PythagoreanOne,
        }),
        HyperbolicPythagoreanValue::NegativeOne => Some(HyperbolicIdentityRewrite {
            rewritten: ctx.num(-1),
            kind: HyperbolicIdentityRewriteKind::PythagoreanNegativeOne,
        }),
    }
}

/// Detect and rewrite:
/// - `sinh(x) + cosh(x)` or `cosh(x) + sinh(x)` to `exp(x)`
/// - `cosh(x) - sinh(x)` to `exp(-x)`
/// - `sinh(x) - cosh(x)` to `-exp(-x)`
pub fn try_rewrite_sinh_cosh_to_exp(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<SinhCoshToExpRewrite> {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            let (Expr::Function(l_fn, l_args), Expr::Function(r_fn, r_args)) =
                (ctx.get(l), ctx.get(r))
            else {
                return None;
            };
            if l_args.len() != 1 || r_args.len() != 1 {
                return None;
            }
            let is_sinh_plus_cosh =
                ctx.is_builtin(*l_fn, BuiltinFn::Sinh) && ctx.is_builtin(*r_fn, BuiltinFn::Cosh);
            let is_cosh_plus_sinh =
                ctx.is_builtin(*l_fn, BuiltinFn::Cosh) && ctx.is_builtin(*r_fn, BuiltinFn::Sinh);

            if (is_sinh_plus_cosh || is_cosh_plus_sinh)
                && compare_expr(ctx, l_args[0], r_args[0]) == Ordering::Equal
            {
                let exp_x = e_pow(ctx, l_args[0]);
                return Some(SinhCoshToExpRewrite {
                    rewritten: exp_x,
                    kind: SinhCoshToExpRewriteKind::Sum,
                });
            }
            None
        }
        Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            let (Expr::Function(l_fn, l_args), Expr::Function(r_fn, r_args)) =
                (ctx.get(l), ctx.get(r))
            else {
                return None;
            };
            if l_args.len() != 1 || r_args.len() != 1 {
                return None;
            }
            let is_cosh_minus_sinh =
                ctx.is_builtin(*l_fn, BuiltinFn::Cosh) && ctx.is_builtin(*r_fn, BuiltinFn::Sinh);
            let is_sinh_minus_cosh =
                ctx.is_builtin(*l_fn, BuiltinFn::Sinh) && ctx.is_builtin(*r_fn, BuiltinFn::Cosh);

            if !(is_cosh_minus_sinh || is_sinh_minus_cosh)
                || compare_expr(ctx, l_args[0], r_args[0]) != Ordering::Equal
            {
                return None;
            }

            let neg_arg = ctx.add(Expr::Neg(l_args[0]));
            let exp_neg_x = e_pow(ctx, neg_arg);
            if is_cosh_minus_sinh {
                Some(SinhCoshToExpRewrite {
                    rewritten: exp_neg_x,
                    kind: SinhCoshToExpRewriteKind::CoshMinusSinh,
                })
            } else {
                Some(SinhCoshToExpRewrite {
                    rewritten: ctx.add(Expr::Neg(exp_neg_x)),
                    kind: SinhCoshToExpRewriteKind::SinhMinusCosh,
                })
            }
        }
        _ => None,
    }
}

/// Detect and rewrite:
/// `cosh(x)^2 + sinh(x)^2` (or swapped) to `cosh(2x)`.
pub fn try_rewrite_hyperbolic_double_angle_sum(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HyperbolicDoubleAngleRewrite> {
    let Expr::Add(l, r) = ctx.get(expr) else {
        return None;
    };
    let (l, r) = (*l, *r);

    let (Expr::Pow(l_base, l_exp), Expr::Pow(r_base, r_exp)) = (ctx.get(l), ctx.get(r)) else {
        return None;
    };
    let (l_base, l_exp, r_base, r_exp) = (*l_base, *l_exp, *r_base, *r_exp);

    if !is_two_expr(ctx, l_exp) || !is_two_expr(ctx, r_exp) {
        return None;
    }

    let (Expr::Function(l_fn, l_args), Expr::Function(r_fn, r_args)) =
        (ctx.get(l_base), ctx.get(r_base))
    else {
        return None;
    };
    if l_args.len() != 1 || r_args.len() != 1 {
        return None;
    }

    let is_cosh_sinh =
        ctx.is_builtin(*l_fn, BuiltinFn::Cosh) && ctx.is_builtin(*r_fn, BuiltinFn::Sinh);
    let is_sinh_cosh =
        ctx.is_builtin(*l_fn, BuiltinFn::Sinh) && ctx.is_builtin(*r_fn, BuiltinFn::Cosh);

    if !(is_cosh_sinh || is_sinh_cosh) || compare_expr(ctx, l_args[0], r_args[0]) != Ordering::Equal
    {
        return None;
    }

    let x = l_args[0];
    let two = ctx.num(2);
    let two_x = ctx.add(Expr::Mul(two, x));
    let cosh_2x = ctx.call_builtin(BuiltinFn::Cosh, vec![two_x]);
    Some(HyperbolicDoubleAngleRewrite {
        rewritten: cosh_2x,
        kind: HyperbolicDoubleAngleRewriteKind::Sum,
    })
}

/// Detect and rewrite additive-chain subtraction form:
/// `cosh(2x) - cosh²(x) - sinh²(x) -> 0`.
///
/// Works on canonicalized add chains where subtraction appears as `Add(..., Neg(...))`.
pub fn try_rewrite_hyperbolic_double_angle_sub_chain(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HyperbolicDoubleAngleRewrite> {
    let mut terms = Vec::new();
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        if let Expr::Add(l, r) = ctx.get(id) {
            stack.push(*l);
            stack.push(*r);
        } else {
            terms.push(id);
        }
    }

    if terms.len() < 3 {
        return None;
    }

    let mut cosh_double = None;
    for (i, &t) in terms.iter().enumerate() {
        if let Expr::Function(fn_id, args) = ctx.get(t) {
            if ctx.builtin_of(*fn_id) == Some(BuiltinFn::Cosh) && args.len() == 1 {
                if let Some(x) = extract_double_angle_arg_relaxed(ctx, args[0]) {
                    cosh_double = Some((i, x));
                    break;
                }
            }
            if cosh_double.is_some() {
                break;
            }
        }
    }

    let as_neg_hyp_squared = |e: ExprId| -> Option<(ExprId, bool)> {
        if let Expr::Neg(inner) = ctx.get(e) {
            if let Expr::Pow(base, exp) = ctx.get(*inner) {
                if is_two_expr(ctx, *exp) {
                    if let Expr::Function(fn_id, args) = ctx.get(*base) {
                        if args.len() == 1 {
                            if ctx.is_builtin(*fn_id, BuiltinFn::Cosh) {
                                return Some((args[0], true));
                            }
                            if ctx.is_builtin(*fn_id, BuiltinFn::Sinh) {
                                return Some((args[0], false));
                            }
                        }
                    }
                }
            }
        }
        None
    };

    let (cosh_idx, x_arg) = cosh_double?;

    let mut neg_cosh_idx = None;
    let mut neg_sinh_idx = None;

    for (i, &t) in terms.iter().enumerate() {
        if i == cosh_idx {
            continue;
        }
        if let Some((arg, is_cosh)) = as_neg_hyp_squared(t) {
            if compare_expr(ctx, arg, x_arg) == Ordering::Equal {
                if is_cosh && neg_cosh_idx.is_none() {
                    neg_cosh_idx = Some(i);
                } else if !is_cosh && neg_sinh_idx.is_none() {
                    neg_sinh_idx = Some(i);
                }
            }
        }
    }

    let nc_idx = neg_cosh_idx?;
    let ns_idx = neg_sinh_idx?;

    let mut remaining: Vec<ExprId> = terms
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != cosh_idx && i != nc_idx && i != ns_idx)
        .map(|(_, &t)| t)
        .collect();

    let rewritten = if remaining.is_empty() {
        ctx.num(0)
    } else {
        let mut result = remaining.pop().expect("non-empty");
        while let Some(t) = remaining.pop() {
            result = ctx.add(Expr::Add(t, result));
        }
        result
    };

    Some(HyperbolicDoubleAngleRewrite {
        rewritten,
        kind: HyperbolicDoubleAngleRewriteKind::SubChain,
    })
}

/// Detect and rewrite additive-chain form of:
/// `1 - tanh(x)^2 -> 1/cosh(x)^2`.
///
/// Works on flattened additive forms such as `1 + (-tanh(x)^2) + rest`.
pub fn try_rewrite_tanh_pythagorean_add_chain(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HyperbolicTanhPythagoreanRewrite> {
    let terms = crate::expr_nary::add_terms_signed(ctx, expr);
    if terms.len() < 2 {
        return None;
    }

    let mut constants: Vec<(usize, BigRational)> = Vec::new();
    let mut tanh_terms: Vec<(usize, BigRational, ExprId)> = Vec::new();

    for (i, &(term, sign)) in terms.iter().enumerate() {
        let sign_coeff = BigRational::from_integer(sign.to_i32().into());
        if let Expr::Number(n) = ctx.get(term) {
            let coeff = n.clone() * sign_coeff;
            if !coeff.is_zero() {
                constants.push((i, coeff));
            }
            continue;
        }

        if let Some((coeff, arg)) = extract_coeff_tanh_pow2(ctx, term) {
            let coeff = coeff * sign_coeff;
            if !coeff.is_zero() {
                tanh_terms.push((i, coeff, arg));
            }
        }
    }

    let mut matched: Option<(usize, usize, BigRational, ExprId)> = None;
    for (constant_i, constant_coeff) in &constants {
        for (tanh_i, tanh_coeff, arg) in &tanh_terms {
            if constant_i == tanh_i {
                continue;
            }
            if tanh_coeff == &(-constant_coeff.clone()) {
                matched = Some((*constant_i, *tanh_i, constant_coeff.clone(), *arg));
                break;
            }
        }
        if matched.is_some() {
            break;
        }
    }

    let Some((constant_i, tanh_i, coeff, arg)) = matched else {
        return try_rewrite_tanh_pythagorean_common_cofactor(ctx, &terms);
    };

    let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
    let two = ctx.num(2);
    let cosh_squared = ctx.add(Expr::Pow(cosh, two));
    let one = ctx.num(1);
    let sech_squared = ctx.add(Expr::Div(one, cosh_squared));
    let replacement = multiply_numeric_coeff(ctx, &coeff, sech_squared);

    let mut new_terms: Vec<ExprId> = Vec::new();
    for (i, &(term, sign)) in terms.iter().enumerate() {
        if i != constant_i && i != tanh_i {
            let signed_term = match sign {
                crate::expr_nary::Sign::Pos => term,
                crate::expr_nary::Sign::Neg => ctx.add(Expr::Neg(term)),
            };
            new_terms.push(signed_term);
        }
    }
    new_terms.push(replacement);

    let rewritten = if new_terms.len() == 1 {
        new_terms[0]
    } else {
        let mut acc = new_terms[0];
        for &term in new_terms.iter().skip(1) {
            acc = ctx.add(Expr::Add(acc, term));
        }
        acc
    };

    Some(HyperbolicTanhPythagoreanRewrite { rewritten })
}

/// Detect and rewrite `sinh(x)/cosh(x) -> tanh(x)`.
pub fn try_rewrite_sinh_cosh_to_tanh(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let num = *num;
    let den = *den;

    let Expr::Function(num_name, num_args) = ctx.get(num) else {
        return None;
    };
    if !ctx.is_builtin(*num_name, BuiltinFn::Sinh) || num_args.len() != 1 {
        return None;
    }

    let Expr::Function(den_name, den_args) = ctx.get(den) else {
        return None;
    };
    if !ctx.is_builtin(*den_name, BuiltinFn::Cosh) || den_args.len() != 1 {
        return None;
    }

    if compare_expr(ctx, num_args[0], den_args[0]) != Ordering::Equal {
        return None;
    }

    Some(ctx.call_builtin(BuiltinFn::Tanh, vec![num_args[0]]))
}

/// Detect and rewrite `cosh(x)/sinh(x) -> 1/tanh(x)`.
pub fn try_rewrite_cosh_sinh_to_reciprocal_tanh(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let num = *num;
    let den = *den;

    let Expr::Function(num_name, num_args) = ctx.get(num) else {
        return None;
    };
    if !ctx.is_builtin(*num_name, BuiltinFn::Cosh) || num_args.len() != 1 {
        return None;
    }

    let Expr::Function(den_name, den_args) = ctx.get(den) else {
        return None;
    };
    if !ctx.is_builtin(*den_name, BuiltinFn::Sinh) || den_args.len() != 1 {
        return None;
    }

    if compare_expr(ctx, num_args[0], den_args[0]) != Ordering::Equal {
        return None;
    }
    let arg = num_args[0];

    let one = ctx.num(1);
    let tanh = ctx.call_builtin(BuiltinFn::Tanh, vec![arg]);
    Some(ctx.add(Expr::Div(one, tanh)))
}

/// Detect and rewrite `sinh(x)/cosh(x) -> tanh(x)` with canonical description.
pub fn try_rewrite_sinh_cosh_to_tanh_identity_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HyperbolicIdentityRewrite> {
    let rewritten = try_rewrite_sinh_cosh_to_tanh(ctx, expr)?;
    Some(HyperbolicIdentityRewrite {
        rewritten,
        kind: HyperbolicIdentityRewriteKind::SinhCoshToTanh,
    })
}

/// Detect and rewrite `cosh(x)/sinh(x) -> 1/tanh(x)` with canonical description.
pub fn try_rewrite_cosh_sinh_to_reciprocal_tanh_identity_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HyperbolicIdentityRewrite> {
    let rewritten = try_rewrite_cosh_sinh_to_reciprocal_tanh(ctx, expr)?;
    Some(HyperbolicIdentityRewrite {
        rewritten,
        kind: HyperbolicIdentityRewriteKind::CoshSinhToReciprocalTanh,
    })
}

/// Detect and rewrite `tanh(x) -> sinh(x)/cosh(x)`.
///
/// Guarded to preserve:
/// - direct composition simplifications like `tanh(atanh(x))`
/// - odd-function normalization `tanh(-x) -> -tanh(x)`
pub fn try_rewrite_tanh_to_sinh_cosh(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Tanh) || args.len() != 1 {
        return None;
    }
    let x = args[0];

    // Preserve composition rules for inverse hyperbolic arguments.
    if let Expr::Function(inner_fn, _) = ctx.get(x) {
        if ctx.is_builtin(*inner_fn, BuiltinFn::Atanh)
            || ctx.is_builtin(*inner_fn, BuiltinFn::Asinh)
            || ctx.is_builtin(*inner_fn, BuiltinFn::Acosh)
        {
            return None;
        }
    }

    // Preserve odd-function rewrite path tanh(-x) -> -tanh(x).
    if matches!(ctx.get(x), Expr::Neg(_)) {
        return None;
    }

    let sinh_x = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
    let cosh_x = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
    Some(ctx.add(Expr::Div(sinh_x, cosh_x)))
}

/// Detect and rewrite `tanh(x) -> sinh(x)/cosh(x)` with canonical description.
pub fn try_rewrite_tanh_to_sinh_cosh_identity_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HyperbolicIdentityRewrite> {
    let rewritten = try_rewrite_tanh_to_sinh_cosh(ctx, expr)?;
    Some(HyperbolicIdentityRewrite {
        rewritten,
        kind: HyperbolicIdentityRewriteKind::TanhToSinhCosh,
    })
}

/// Detect and rewrite `sinh(2x) -> 2*sinh(x)*cosh(x)`.
pub fn try_rewrite_sinh_double_angle_expansion(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Sinh) || args.len() != 1 {
        return None;
    }

    let x = extract_double_angle_arg_relaxed(ctx, args[0])?;
    let two = ctx.num(2);
    let sinh_x = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
    let cosh_x = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
    let sinh_cosh = crate::expr_rewrite::smart_mul(ctx, sinh_x, cosh_x);
    Some(crate::expr_rewrite::smart_mul(ctx, two, sinh_cosh))
}

/// Detect and rewrite `sinh(2x) -> 2*sinh(x)*cosh(x)` with canonical description.
pub fn try_rewrite_sinh_double_angle_expansion_identity_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HyperbolicIdentityRewrite> {
    let rewritten = try_rewrite_sinh_double_angle_expansion(ctx, expr)?;
    Some(HyperbolicIdentityRewrite {
        rewritten,
        kind: HyperbolicIdentityRewriteKind::SinhDoubleAngleExpansion,
    })
}

/// Detect and rewrite `tanh(2x) -> 2*tanh(x)/(1 + tanh(x)^2)`.
pub fn try_rewrite_tanh_double_angle_expansion(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Tanh) || args.len() != 1 {
        return None;
    }

    let x = extract_double_angle_arg_relaxed(ctx, args[0])?;
    let two = ctx.num(2);
    let one = ctx.num(1);
    let tanh_x = ctx.call_builtin(BuiltinFn::Tanh, vec![x]);
    let numerator = crate::expr_rewrite::smart_mul(ctx, two, tanh_x);
    let tanh_sq = ctx.add(Expr::Pow(tanh_x, two));
    let denominator = ctx.add(Expr::Add(one, tanh_sq));
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

/// Detect and rewrite `tanh(2x) -> 2*tanh(x)/(1 + tanh(x)^2)` with canonical description.
pub fn try_rewrite_tanh_double_angle_expansion_identity_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HyperbolicIdentityRewrite> {
    let rewritten = try_rewrite_tanh_double_angle_expansion(ctx, expr)?;
    Some(HyperbolicIdentityRewrite {
        rewritten,
        kind: HyperbolicIdentityRewriteKind::TanhDoubleAngleExpansion,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HyperbolicTripleAngleRewriteKind {
    Sinh,
    Cosh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyperbolicTripleAngleRewrite {
    pub rewritten: ExprId,
    pub kind: HyperbolicTripleAngleRewriteKind,
}

/// Detect and rewrite hyperbolic triple-angle expansions.
///
/// - `sinh(3x) -> 3*sinh(x) + 4*sinh(x)^3`
/// - `cosh(3x) -> 4*cosh(x)^3 - 3*cosh(x)`
pub fn try_rewrite_hyperbolic_triple_angle(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HyperbolicTripleAngleRewrite> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let x = extract_triple_angle_arg(ctx, args[0])?;

    // Expand only for trivial argument forms to avoid expression blow-up.
    let is_simple = |id: ExprId| {
        matches!(
            ctx.get(id),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_)
        )
    };
    match ctx.get(x) {
        Expr::Variable(_) | Expr::Constant(_) | Expr::Number(_) => {}
        Expr::Mul(l, r) => {
            if !(is_simple(*l) && is_simple(*r)) {
                return None;
            }
        }
        Expr::Neg(inner) => {
            if !is_simple(*inner) {
                return None;
            }
        }
        _ => return None,
    }

    match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Sinh) => {
            let three = ctx.num(3);
            let four = ctx.num(4);
            let exp_three = ctx.num(3);
            let sinh_x = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
            let term1 = crate::expr_rewrite::smart_mul(ctx, three, sinh_x);
            let sinh_cubed = ctx.add(Expr::Pow(sinh_x, exp_three));
            let term2 = crate::expr_rewrite::smart_mul(ctx, four, sinh_cubed);
            let rewritten = ctx.add(Expr::Add(term1, term2));
            Some(HyperbolicTripleAngleRewrite {
                rewritten,
                kind: HyperbolicTripleAngleRewriteKind::Sinh,
            })
        }
        Some(BuiltinFn::Cosh) => {
            let three = ctx.num(3);
            let four = ctx.num(4);
            let exp_three = ctx.num(3);
            let cosh_x = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
            let cosh_cubed = ctx.add(Expr::Pow(cosh_x, exp_three));
            let term1 = crate::expr_rewrite::smart_mul(ctx, four, cosh_cubed);
            let term2 = crate::expr_rewrite::smart_mul(ctx, three, cosh_x);
            let rewritten = ctx.add(Expr::Sub(term1, term2));
            Some(HyperbolicTripleAngleRewrite {
                rewritten,
                kind: HyperbolicTripleAngleRewriteKind::Cosh,
            })
        }
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecognizeHyperbolicFromExpRewriteKind {
    CoshHalf,
    SinhHalf,
    NegSinhHalf,
    CoshDirect,
    SinhDirect,
    NegSinhDirect,
    TanhRatio,
    NegTanhRatio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecognizeHyperbolicFromExpRewrite {
    pub rewritten: ExprId,
    pub kind: RecognizeHyperbolicFromExpRewriteKind,
}

fn build_recognize_hyperbolic_from_exp_rewrite(
    ctx: &mut Context,
    arg: ExprId,
    is_cosh: bool,
    positive_first: bool,
    direct: bool,
) -> RecognizeHyperbolicFromExpRewrite {
    let base = if is_cosh {
        ctx.call_builtin(BuiltinFn::Cosh, vec![arg])
    } else {
        ctx.call_builtin(BuiltinFn::Sinh, vec![arg])
    };
    let two = ctx.num(2);
    let neg_two = ctx.num(-2);

    let (rewritten, kind) = match (is_cosh, direct, positive_first) {
        (true, false, _) => (base, RecognizeHyperbolicFromExpRewriteKind::CoshHalf),
        (true, true, _) => (
            ctx.add(Expr::Mul(two, base)),
            RecognizeHyperbolicFromExpRewriteKind::CoshDirect,
        ),
        (false, false, true) => (base, RecognizeHyperbolicFromExpRewriteKind::SinhHalf),
        (false, false, false) => (
            ctx.add(Expr::Neg(base)),
            RecognizeHyperbolicFromExpRewriteKind::NegSinhHalf,
        ),
        (false, true, true) => (
            ctx.add(Expr::Mul(two, base)),
            RecognizeHyperbolicFromExpRewriteKind::SinhDirect,
        ),
        (false, true, false) => (
            ctx.add(Expr::Mul(neg_two, base)),
            RecognizeHyperbolicFromExpRewriteKind::NegSinhDirect,
        ),
    };

    RecognizeHyperbolicFromExpRewrite { rewritten, kind }
}

/// Detect and rewrite exponential definitions of hyperbolic functions:
/// - `(e^x + e^(-x))/2` or `(1/2)*(...)` -> `cosh(x)`
/// - `(e^x - e^(-x))/2` or `(1/2)*(...)` -> `sinh(x)`
/// - `(e^(-x) - e^x)/2` or `(1/2)*(...)` -> `-sinh(x)`
/// - `e^x + e^(-x)` (or `e^x + 1/e^x`) -> `2*cosh(x)`
/// - `e^x - e^(-x)` (or `e^x - 1/e^x`) -> `2*sinh(x)`
/// - `(e^x - e^(-x))/(e^x + e^(-x))` -> `tanh(x)` (or negated variant)
pub fn try_rewrite_recognize_hyperbolic_from_exp(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<RecognizeHyperbolicFromExpRewrite> {
    // Pattern 1: Div(sum_or_diff, 2)
    if let Expr::Div(num, den) = ctx.get(expr) {
        if is_two_expr(ctx, *den) {
            if let Some((arg, is_cosh, positive_first)) =
                crate::hyperbolic_exp_support::extract_exp_pair(ctx, *num)
            {
                return Some(build_recognize_hyperbolic_from_exp_rewrite(
                    ctx,
                    arg,
                    is_cosh,
                    positive_first,
                    false,
                ));
            }
        }
    }

    // Pattern 2: Mul(1/2, sum_or_diff) or Mul(sum_or_diff, 1/2)
    if let Expr::Mul(l, r) = ctx.get(expr) {
        let sum_id = if crate::expr_predicates::is_half_expr(ctx, *l) {
            Some(*r)
        } else if crate::expr_predicates::is_half_expr(ctx, *r) {
            Some(*l)
        } else {
            None
        };
        if let Some(sum_id) = sum_id {
            if let Some((arg, is_cosh, positive_first)) =
                crate::hyperbolic_exp_support::extract_exp_pair(ctx, sum_id)
            {
                return Some(build_recognize_hyperbolic_from_exp_rewrite(
                    ctx,
                    arg,
                    is_cosh,
                    positive_first,
                    false,
                ));
            }
        }
    }

    // Pattern 3: e^x +/- e^(-x) -> 2*cosh(x) / +/-2*sinh(x).
    if let Some((arg, is_cosh, positive_first)) =
        crate::hyperbolic_exp_support::extract_exp_pair(ctx, expr)
    {
        return Some(build_recognize_hyperbolic_from_exp_rewrite(
            ctx,
            arg,
            is_cosh,
            positive_first,
            true,
        ));
    }

    // Pattern 4: (e^x - e^(-x)) / (e^x + e^(-x)) -> tanh(x) (or -tanh(x)).
    if let Expr::Div(num, den) = ctx.get(expr) {
        if let Some((num_arg, false, num_positive_first)) =
            crate::hyperbolic_exp_support::extract_exp_pair(ctx, *num)
        {
            if let Some((den_arg, true, _)) =
                crate::hyperbolic_exp_support::extract_exp_pair(ctx, *den)
            {
                if compare_expr(ctx, num_arg, den_arg) == Ordering::Equal {
                    let tanh_x = ctx.call_builtin(BuiltinFn::Tanh, vec![num_arg]);
                    if num_positive_first {
                        return Some(RecognizeHyperbolicFromExpRewrite {
                            rewritten: tanh_x,
                            kind: RecognizeHyperbolicFromExpRewriteKind::TanhRatio,
                        });
                    } else {
                        return Some(RecognizeHyperbolicFromExpRewrite {
                            rewritten: ctx.add(Expr::Neg(tanh_x)),
                            kind: RecognizeHyperbolicFromExpRewriteKind::NegTanhRatio,
                        });
                    }
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{
        detect_hyperbolic_pythagorean_sub, try_rewrite_cosh_sinh_to_reciprocal_tanh,
        try_rewrite_cosh_sinh_to_reciprocal_tanh_identity_expr,
        try_rewrite_csch_fourth_tanh_verification_add_chain,
        try_rewrite_hyperbolic_double_angle_sub_chain, try_rewrite_hyperbolic_double_angle_sum,
        try_rewrite_hyperbolic_pythagorean_sub_expr, try_rewrite_hyperbolic_triple_angle,
        try_rewrite_recognize_hyperbolic_from_exp, try_rewrite_sinh_cosh_to_exp,
        try_rewrite_sinh_cosh_to_tanh, try_rewrite_sinh_cosh_to_tanh_identity_expr,
        try_rewrite_sinh_double_angle_expansion,
        try_rewrite_sinh_double_angle_expansion_identity_expr,
        try_rewrite_tanh_double_angle_expansion,
        try_rewrite_tanh_double_angle_expansion_identity_expr,
        try_rewrite_tanh_pythagorean_add_chain, try_rewrite_tanh_to_sinh_cosh,
        try_rewrite_tanh_to_sinh_cosh_identity_expr, HyperbolicDoubleAngleRewriteKind,
        HyperbolicIdentityRewriteKind, HyperbolicPythagoreanValue,
        HyperbolicTripleAngleRewriteKind, RecognizeHyperbolicFromExpRewriteKind,
        SinhCoshToExpRewriteKind,
    };
    use cas_ast::ordering::compare_expr;
    use cas_ast::{BuiltinFn, Context, Expr, ExprId};
    use cas_formatter::render_expr;
    use cas_parser::parse;
    use std::cmp::Ordering;

    fn matches_any_expected_arg(ctx: &Context, arg: ExprId, expected: &[ExprId]) -> bool {
        expected
            .iter()
            .any(|candidate| compare_expr(ctx, arg, *candidate) == Ordering::Equal)
    }

    fn assert_two_builtin_product_with_shared_arg(
        ctx: &Context,
        two: ExprId,
        expr: ExprId,
        lhs_builtin: BuiltinFn,
        rhs_builtin: BuiltinFn,
        expected_args: &[ExprId],
    ) {
        let Expr::Mul(lhs, rhs) = ctx.get(expr) else {
            panic!("expected outer multiplication");
        };
        let inner_mul = if *lhs == two {
            *rhs
        } else if *rhs == two {
            *lhs
        } else {
            panic!("expected numeric factor 2");
        };

        let Expr::Mul(m1, m2) = ctx.get(inner_mul) else {
            panic!("expected inner multiplication");
        };
        let (f1, f2) = (*m1, *m2);
        let Expr::Function(fn1, args1) = ctx.get(f1) else {
            panic!("expected builtin factor");
        };
        let Expr::Function(fn2, args2) = ctx.get(f2) else {
            panic!("expected builtin factor");
        };
        assert_eq!(args1.len(), 1);
        assert_eq!(args2.len(), 1);
        let builtins_match = (ctx.builtin_of(*fn1) == Some(lhs_builtin)
            && ctx.builtin_of(*fn2) == Some(rhs_builtin))
            || (ctx.builtin_of(*fn1) == Some(rhs_builtin)
                && ctx.builtin_of(*fn2) == Some(lhs_builtin));
        assert!(
            builtins_match,
            "expected {:?}/{:?} factors",
            lhs_builtin, rhs_builtin
        );
        assert!(
            matches_any_expected_arg(ctx, args1[0], expected_args),
            "unexpected first argument: {}",
            render_expr(ctx, args1[0])
        );
        assert!(
            matches_any_expected_arg(ctx, args2[0], expected_args),
            "unexpected second argument: {}",
            render_expr(ctx, args2[0])
        );
    }

    #[test]
    fn detects_cosh2_minus_sinh2() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
        let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
        let lhs = ctx.add(Expr::Pow(cosh, two));
        let rhs = ctx.add(Expr::Pow(sinh, two));
        let expr = ctx.add(Expr::Sub(lhs, rhs));

        assert_eq!(
            detect_hyperbolic_pythagorean_sub(&ctx, expr),
            Some(HyperbolicPythagoreanValue::One)
        );
    }

    #[test]
    fn detects_sinh2_minus_cosh2() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
        let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
        let lhs = ctx.add(Expr::Pow(sinh, two));
        let rhs = ctx.add(Expr::Pow(cosh, two));
        let expr = ctx.add(Expr::Sub(lhs, rhs));

        assert_eq!(
            detect_hyperbolic_pythagorean_sub(&ctx, expr),
            Some(HyperbolicPythagoreanValue::NegativeOne)
        );
    }

    #[test]
    fn rewrites_hyperbolic_pythagorean_sub_with_desc() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
        let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
        let lhs = ctx.add(Expr::Pow(cosh, two));
        let rhs = ctx.add(Expr::Pow(sinh, two));
        let expr = ctx.add(Expr::Sub(lhs, rhs));
        let rewrite = try_rewrite_hyperbolic_pythagorean_sub_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, HyperbolicIdentityRewriteKind::PythagoreanOne);
    }

    #[test]
    fn rewrites_scaled_tanh_pythagorean_add_chain() {
        let mut ctx = Context::new();
        let expr = parse("6 - 6*tanh(2*x + 1)^2", &mut ctx).expect("expr");
        let expected = parse("6 * (1 / cosh(2*x + 1)^2)", &mut ctx).expect("expected");

        let rewrite = try_rewrite_tanh_pythagorean_add_chain(&mut ctx, expr).expect("rewrite");

        assert!(
            compare_expr(&ctx, rewrite.rewritten, expected) == Ordering::Equal,
            "expected {}, got {}",
            render_expr(&ctx, expected),
            render_expr(&ctx, rewrite.rewritten)
        );
    }

    #[test]
    fn rewrites_symbolic_cofactor_tanh_pythagorean_add_chain() {
        let mut ctx = Context::new();
        let expr = parse("6*k*x - 6*k*x*tanh(x^2 + b)^2", &mut ctx).expect("expr");
        let expected = parse("6*k*x * (1 / cosh(x^2 + b)^2)", &mut ctx).expect("expected");

        let rewrite = try_rewrite_tanh_pythagorean_add_chain(&mut ctx, expr).expect("rewrite");

        assert!(
            compare_expr(&ctx, rewrite.rewritten, expected) == Ordering::Equal,
            "expected {}, got {}",
            render_expr(&ctx, expected),
            render_expr(&ctx, rewrite.rewritten)
        );
    }

    #[test]
    fn rewrites_csch_fourth_tanh_verification_add_chain_for_affine_argument() {
        let mut ctx = Context::new();
        let expr = parse(
            "1/(cosh(2*x+1)^2*tanh(2*x+1)^4) - 1/sinh(2*x+1)^4 - 1/sinh(2*x+1)^2",
            &mut ctx,
        )
        .expect("expr");

        let rewrite =
            try_rewrite_csch_fourth_tanh_verification_add_chain(&mut ctx, expr).expect("rewrite");

        assert_eq!(render_expr(&ctx, rewrite.rewritten), "0");
    }

    #[test]
    fn rewrites_csch_fourth_tanh_verification_add_chain_with_symbolic_cofactor() {
        let mut ctx = Context::new();
        let expr = parse(
            "2*k*x/(cosh(x^2+b)^2*tanh(x^2+b)^4) - 2*k*x/sinh(x^2+b)^4 - 2*k*x/sinh(x^2+b)^2",
            &mut ctx,
        )
        .expect("expr");

        let rewrite =
            try_rewrite_csch_fourth_tanh_verification_add_chain(&mut ctx, expr).expect("rewrite");

        assert_eq!(render_expr(&ctx, rewrite.rewritten), "0");
    }

    #[test]
    fn rewrites_csch_fourth_tanh_verification_add_chain_with_tanh_square_denominator() {
        let mut ctx = Context::new();
        let expr = parse(
            "a/(cosh(u)^2*tanh(u)^4) - a/sinh(u)^4 - a/(cosh(u)^2*tanh(u)^2)",
            &mut ctx,
        )
        .expect("expr");

        let rewrite =
            try_rewrite_csch_fourth_tanh_verification_add_chain(&mut ctx, expr).expect("rewrite");

        assert_eq!(render_expr(&ctx, rewrite.rewritten), "0");
    }

    #[test]
    fn rewrites_csch_fourth_tanh_verification_add_chain_with_denominator_coefficient() {
        let mut ctx = Context::new();
        let expr = parse(
            "18*k*x/(9*cosh(x^2+b)^2*tanh(x^2+b)^4) - 2*k*x/sinh(x^2+b)^4 - 2*k*x/(cosh(x^2+b)^2*tanh(x^2+b)^2)",
            &mut ctx,
        )
        .expect("expr");

        let rewrite =
            try_rewrite_csch_fourth_tanh_verification_add_chain(&mut ctx, expr).expect("rewrite");

        assert_eq!(render_expr(&ctx, rewrite.rewritten), "0");
    }

    #[test]
    fn rewrites_csch_fourth_tanh_verification_add_chain_with_external_fraction_scale() {
        let mut ctx = Context::new();
        let expr = parse(
            "2*(k*x/(cosh(x^2+b)^2*tanh(x^2+b)^4)) - 2*(k*x/sinh(x^2+b)^4) - 2*(k*x/(cosh(x^2+b)^2*tanh(x^2+b)^2))",
            &mut ctx,
        )
        .expect("expr");

        let rewrite =
            try_rewrite_csch_fourth_tanh_verification_add_chain(&mut ctx, expr).expect("rewrite");

        assert_eq!(render_expr(&ctx, rewrite.rewritten), "0");
    }

    #[test]
    fn rejects_mismatched_arguments() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let two = ctx.num(2);
        let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
        let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![y]);
        let lhs = ctx.add(Expr::Pow(cosh, two));
        let rhs = ctx.add(Expr::Pow(sinh, two));
        let expr = ctx.add(Expr::Sub(lhs, rhs));

        assert_eq!(detect_hyperbolic_pythagorean_sub(&ctx, expr), None);
    }

    #[test]
    fn rewrites_sinh_plus_cosh_to_exp() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
        let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
        let expr = ctx.add(Expr::Add(sinh, cosh));

        let rewrite = try_rewrite_sinh_cosh_to_exp(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, SinhCoshToExpRewriteKind::Sum);
    }

    #[test]
    fn rewrites_cosh_minus_sinh_to_exp_neg() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
        let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
        let expr = ctx.add(Expr::Sub(cosh, sinh));

        let rewrite = try_rewrite_sinh_cosh_to_exp(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, SinhCoshToExpRewriteKind::CoshMinusSinh);
    }

    #[test]
    fn rewrites_double_angle_sum() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
        let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
        let cosh_sq = ctx.add(Expr::Pow(cosh, two));
        let sinh_sq = ctx.add(Expr::Pow(sinh, two));
        let expr = ctx.add(Expr::Add(cosh_sq, sinh_sq));

        let rewrite = try_rewrite_hyperbolic_double_angle_sum(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, HyperbolicDoubleAngleRewriteKind::Sum);
    }

    #[test]
    fn rewrites_double_angle_sub_chain_to_zero() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let two_x = ctx.add(Expr::Mul(two, x));
        let cosh_2x = ctx.call_builtin(BuiltinFn::Cosh, vec![two_x]);
        let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
        let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
        let cosh_sq = ctx.add(Expr::Pow(cosh, two));
        let sinh_sq = ctx.add(Expr::Pow(sinh, two));
        let neg_cosh_sq = ctx.add(Expr::Neg(cosh_sq));
        let neg_sinh_sq = ctx.add(Expr::Neg(sinh_sq));
        let tail = ctx.add(Expr::Add(neg_cosh_sq, neg_sinh_sq));
        let expr = ctx.add(Expr::Add(cosh_2x, tail));

        let rewrite =
            try_rewrite_hyperbolic_double_angle_sub_chain(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, HyperbolicDoubleAngleRewriteKind::SubChain);
        let zero = num_rational::BigRational::from_integer(0.into());
        assert!(matches!(ctx.get(rewrite.rewritten), Expr::Number(n) if n == &zero));
    }

    #[test]
    fn rewrites_tanh_pythagorean_add_chain() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let tanh_x = ctx.call_builtin(BuiltinFn::Tanh, vec![x]);
        let tanh_sq = ctx.add(Expr::Pow(tanh_x, two));
        let neg_tanh_sq = ctx.add(Expr::Neg(tanh_sq));
        let expr = ctx.add(Expr::Add(one, neg_tanh_sq));

        let rewrite = try_rewrite_tanh_pythagorean_add_chain(&mut ctx, expr).expect("rewrite");
        assert_eq!(render_expr(&ctx, rewrite.rewritten), "1 / cosh(x)^2");
    }

    #[test]
    fn rewrites_sinh_div_cosh_to_tanh() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
        let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
        let expr = ctx.add(Expr::Div(sinh, cosh));

        let rewrite = try_rewrite_sinh_cosh_to_tanh(&mut ctx, expr).expect("rewrite");
        let Expr::Function(fn_id, args) = ctx.get(rewrite) else {
            panic!("expected function");
        };
        assert!(ctx.is_builtin(*fn_id, BuiltinFn::Tanh));
        assert_eq!(args, &vec![x]);
    }

    #[test]
    fn rewrites_sinh_div_cosh_to_tanh_with_desc() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
        let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
        let expr = ctx.add(Expr::Div(sinh, cosh));
        let rewrite = try_rewrite_sinh_cosh_to_tanh_identity_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, HyperbolicIdentityRewriteKind::SinhCoshToTanh);
    }

    #[test]
    fn rewrites_cosh_div_sinh_to_reciprocal_tanh() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
        let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
        let expr = ctx.add(Expr::Div(cosh, sinh));

        let rewrite = try_rewrite_cosh_sinh_to_reciprocal_tanh(&mut ctx, expr).expect("rewrite");
        assert_eq!(render_expr(&ctx, rewrite), "1 / tanh(x)");
    }

    #[test]
    fn rewrites_cosh_div_sinh_to_reciprocal_tanh_with_desc() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
        let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
        let expr = ctx.add(Expr::Div(cosh, sinh));
        let rewrite = try_rewrite_cosh_sinh_to_reciprocal_tanh_identity_expr(&mut ctx, expr)
            .expect("rewrite");
        assert_eq!(
            rewrite.kind,
            HyperbolicIdentityRewriteKind::CoshSinhToReciprocalTanh
        );
    }

    #[test]
    fn rewrites_tanh_to_sinh_div_cosh() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let expr = ctx.call_builtin(BuiltinFn::Tanh, vec![x]);

        let rewrite = try_rewrite_tanh_to_sinh_cosh(&mut ctx, expr).expect("rewrite");
        let Expr::Div(num, den) = ctx.get(rewrite) else {
            panic!("expected division");
        };
        let Expr::Function(num_fn, num_args) = ctx.get(*num) else {
            panic!("expected function numerator");
        };
        let Expr::Function(den_fn, den_args) = ctx.get(*den) else {
            panic!("expected function denominator");
        };
        assert!(ctx.is_builtin(*num_fn, BuiltinFn::Sinh));
        assert!(ctx.is_builtin(*den_fn, BuiltinFn::Cosh));
        assert_eq!(num_args, &vec![x]);
        assert_eq!(den_args, &vec![x]);
    }

    #[test]
    fn rewrites_tanh_to_sinh_div_cosh_with_desc() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let expr = ctx.call_builtin(BuiltinFn::Tanh, vec![x]);
        let rewrite = try_rewrite_tanh_to_sinh_cosh_identity_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, HyperbolicIdentityRewriteKind::TanhToSinhCosh);
    }

    #[test]
    fn rewrites_sinh_double_angle_expansion() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let two_x = ctx.add(Expr::Mul(two, x));
        let expr = ctx.call_builtin(BuiltinFn::Sinh, vec![two_x]);

        let rewrite = try_rewrite_sinh_double_angle_expansion(&mut ctx, expr).expect("rewrite");
        let Expr::Mul(lhs, rhs) = ctx.get(rewrite) else {
            panic!("expected outer multiplication");
        };
        let (two_factor, inner_mul) = if *lhs == two {
            (*lhs, *rhs)
        } else if *rhs == two {
            (*rhs, *lhs)
        } else {
            panic!("expected numeric factor 2");
        };
        assert_eq!(two_factor, two);

        let Expr::Mul(m1, m2) = ctx.get(inner_mul) else {
            panic!("expected inner multiplication");
        };
        let (f1, f2) = (*m1, *m2);
        let Expr::Function(fn1, args1) = ctx.get(f1) else {
            panic!("expected function factor");
        };
        let Expr::Function(fn2, args2) = ctx.get(f2) else {
            panic!("expected function factor");
        };
        assert_eq!(args1, &vec![x]);
        assert_eq!(args2, &vec![x]);
        let is_sinh_cosh =
            ctx.is_builtin(*fn1, BuiltinFn::Sinh) && ctx.is_builtin(*fn2, BuiltinFn::Cosh);
        let is_cosh_sinh =
            ctx.is_builtin(*fn1, BuiltinFn::Cosh) && ctx.is_builtin(*fn2, BuiltinFn::Sinh);
        assert!(is_sinh_cosh || is_cosh_sinh, "expected sinh/cosh factors");
    }

    #[test]
    fn rewrites_sinh_double_angle_expansion_with_desc() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let two_x = ctx.add(Expr::Mul(two, x));
        let expr = ctx.call_builtin(BuiltinFn::Sinh, vec![two_x]);
        let rewrite =
            try_rewrite_sinh_double_angle_expansion_identity_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            rewrite.kind,
            HyperbolicIdentityRewriteKind::SinhDoubleAngleExpansion
        );
    }

    #[test]
    fn rewrites_sinh_double_angle_expansion_for_additive_argument() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let expr = parse("sinh(2*x + 2*pi)", &mut ctx).expect("expr");
        let expected = [
            parse("x + pi", &mut ctx).expect("expected"),
            parse("pi + x", &mut ctx).expect("expected variant"),
        ];
        let rewrite = try_rewrite_sinh_double_angle_expansion(&mut ctx, expr).expect("rewrite");
        assert_two_builtin_product_with_shared_arg(
            &ctx,
            two,
            rewrite,
            BuiltinFn::Sinh,
            BuiltinFn::Cosh,
            &expected,
        );
    }

    #[test]
    fn rewrites_sinh_double_angle_expansion_for_divisible_rational_argument() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let expr = parse("sinh(4*u/(u^2 - 1))", &mut ctx).expect("expr");
        let expected = [
            parse("2*u/(u^2 - 1)", &mut ctx).expect("expected"),
            parse("(u*2)/(u^2 - 1)", &mut ctx).expect("expected variant"),
        ];
        let rewrite = try_rewrite_sinh_double_angle_expansion(&mut ctx, expr).expect("rewrite");
        assert_two_builtin_product_with_shared_arg(
            &ctx,
            two,
            rewrite,
            BuiltinFn::Sinh,
            BuiltinFn::Cosh,
            &expected,
        );
    }

    #[test]
    fn rewrites_sinh_double_angle_expansion_for_expanded_linear_argument() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let expr = parse("sinh(4*u + 6)", &mut ctx).expect("expr");
        let expected = [
            parse("2*u + 3", &mut ctx).expect("expected"),
            parse("3 + 2*u", &mut ctx).expect("expected variant"),
        ];
        let rewrite = try_rewrite_sinh_double_angle_expansion(&mut ctx, expr).expect("rewrite");
        assert_two_builtin_product_with_shared_arg(
            &ctx,
            two,
            rewrite,
            BuiltinFn::Sinh,
            BuiltinFn::Cosh,
            &expected,
        );
    }

    #[test]
    fn rewrites_tanh_double_angle_expansion() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let two_x = ctx.add(Expr::Mul(two, x));
        let expr = ctx.call_builtin(BuiltinFn::Tanh, vec![two_x]);

        let rewrite = try_rewrite_tanh_double_angle_expansion(&mut ctx, expr).expect("rewrite");
        let Expr::Div(num, den) = ctx.get(rewrite) else {
            panic!("expected division");
        };
        assert_eq!(render_expr(&ctx, *num), "2 * tanh(x)");
        let den_render = render_expr(&ctx, *den);
        assert!(
            den_render == "1 + tanh(x)^2" || den_render == "tanh(x)^2 + 1",
            "unexpected denominator: {den_render}"
        );
    }

    #[test]
    fn rewrites_tanh_double_angle_expansion_with_desc() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let two_x = ctx.add(Expr::Mul(two, x));
        let expr = ctx.call_builtin(BuiltinFn::Tanh, vec![two_x]);
        let rewrite =
            try_rewrite_tanh_double_angle_expansion_identity_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            rewrite.kind,
            HyperbolicIdentityRewriteKind::TanhDoubleAngleExpansion
        );
    }

    #[test]
    fn rewrites_hyperbolic_triple_angle_sinh() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let three = ctx.num(3);
        let three_x = ctx.add(Expr::Mul(three, x));
        let expr = ctx.call_builtin(BuiltinFn::Sinh, vec![three_x]);

        let rewrite = try_rewrite_hyperbolic_triple_angle(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, HyperbolicTripleAngleRewriteKind::Sinh);
    }

    #[test]
    fn rewrites_hyperbolic_triple_angle_cosh() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let three = ctx.num(3);
        let three_x = ctx.add(Expr::Mul(three, x));
        let expr = ctx.call_builtin(BuiltinFn::Cosh, vec![three_x]);

        let rewrite = try_rewrite_hyperbolic_triple_angle(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, HyperbolicTripleAngleRewriteKind::Cosh);
    }

    #[test]
    fn recognize_from_exp_rewrites_div_by_two() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let neg_x = ctx.add(Expr::Neg(x));
        let exp_x = ctx.add(Expr::Pow(e, x));
        let e2 = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let exp_neg_x = ctx.add(Expr::Pow(e2, neg_x));
        let sum = ctx.add(Expr::Add(exp_x, exp_neg_x));
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Div(sum, two));

        let rewrite = try_rewrite_recognize_hyperbolic_from_exp(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            rewrite.kind,
            RecognizeHyperbolicFromExpRewriteKind::CoshHalf
        );
    }

    #[test]
    fn recognize_from_exp_rewrites_direct_sum_with_reciprocal() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let exp_x = ctx.add(Expr::Pow(e, x));
        let one = ctx.num(1);
        let reciprocal = ctx.add(Expr::Div(one, exp_x));
        let expr = ctx.add(Expr::Add(exp_x, reciprocal));

        let rewrite = try_rewrite_recognize_hyperbolic_from_exp(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            rewrite.kind,
            RecognizeHyperbolicFromExpRewriteKind::CoshDirect
        );
    }

    #[test]
    fn recognize_from_exp_rewrites_direct_difference_with_reciprocal() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let exp_x = ctx.add(Expr::Pow(e, x));
        let one = ctx.num(1);
        let reciprocal = ctx.add(Expr::Div(one, exp_x));
        let expr = ctx.add(Expr::Sub(exp_x, reciprocal));

        let rewrite = try_rewrite_recognize_hyperbolic_from_exp(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            rewrite.kind,
            RecognizeHyperbolicFromExpRewriteKind::SinhDirect
        );
    }

    #[test]
    fn recognize_from_exp_rewrites_tanh_ratio() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let neg_x = ctx.add(Expr::Neg(x));
        let exp_x = ctx.add(Expr::Pow(e, x));
        let e2 = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let exp_neg_x = ctx.add(Expr::Pow(e2, neg_x));
        let num = ctx.add(Expr::Sub(exp_x, exp_neg_x));
        let den = ctx.add(Expr::Add(exp_x, exp_neg_x));
        let expr = ctx.add(Expr::Div(num, den));

        let rewrite = try_rewrite_recognize_hyperbolic_from_exp(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            rewrite.kind,
            RecognizeHyperbolicFromExpRewriteKind::TanhRatio
        );
    }
}
