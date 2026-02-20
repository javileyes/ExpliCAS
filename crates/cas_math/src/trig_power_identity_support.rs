use crate::expr_nary::mul_leaves;
use crate::numeric_eval::as_rational_const;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::One;
use std::cmp::Ordering;

const SIN_COS_BUILTINS: [BuiltinFn; 2] = [BuiltinFn::Sin, BuiltinFn::Cos];
const TAN_COT_BUILTINS: [BuiltinFn; 2] = [BuiltinFn::Tan, BuiltinFn::Cot];

fn extract_trig_pow_n_from_set(
    ctx: &Context,
    term: ExprId,
    n: i64,
    allowed: &[BuiltinFn],
) -> Option<(ExprId, &'static str)> {
    if let Expr::Pow(base, exp) = ctx.get(term) {
        if let Expr::Number(pow) = ctx.get(*exp) {
            if pow.is_integer() && *pow.numer() == n.into() {
                if let Expr::Function(fn_id, args) = ctx.get(*base) {
                    if args.len() == 1 {
                        let builtin = ctx.builtin_of(*fn_id)?;
                        if allowed.contains(&builtin) {
                            return Some((args[0], builtin.name()));
                        }
                    }
                }
            }
        }
    }
    None
}

fn extract_trig_pow_n(ctx: &Context, term: ExprId, n: i64) -> Option<(ExprId, &'static str)> {
    extract_trig_pow_n_from_set(ctx, term, n, &SIN_COS_BUILTINS)
}

pub fn extract_trig_pow2(ctx: &Context, term: ExprId) -> Option<(ExprId, &'static str)> {
    extract_trig_pow_n(ctx, term, 2)
}

pub fn extract_trig_pow4(ctx: &Context, term: ExprId) -> Option<(ExprId, &'static str)> {
    extract_trig_pow_n(ctx, term, 4)
}

pub fn extract_trig_pow6(ctx: &Context, term: ExprId) -> Option<(ExprId, &'static str)> {
    extract_trig_pow_n(ctx, term, 6)
}

fn extract_coeff_trig_pow_n(
    ctx: &Context,
    term: ExprId,
    n: i64,
    allowed: &[BuiltinFn],
) -> Option<(BigRational, &'static str, ExprId)> {
    let mut coef = BigRational::one();
    let mut working = term;

    if let Expr::Neg(inner) = ctx.get(term) {
        coef = -coef;
        working = *inner;
    }

    let factors = mul_leaves(ctx, working);
    let mut trig_match: Option<(&'static str, ExprId)> = None;

    for factor in factors {
        if let Expr::Number(num) = ctx.get(factor) {
            coef *= num.clone();
            continue;
        }

        if let Some((arg, name)) = extract_trig_pow_n_from_set(ctx, factor, n, allowed) {
            if trig_match.is_some() {
                return None;
            }
            trig_match = Some((name, arg));
            continue;
        }

        return None;
    }

    let (name, arg) = trig_match?;
    Some((coef, name, arg))
}

/// Extract `(coefficient, trig_name, argument)` from `k * sin(arg)^4` or `k * cos(arg)^4`.
pub fn extract_coeff_trig_pow4(
    ctx: &Context,
    term: ExprId,
) -> Option<(BigRational, &'static str, ExprId)> {
    extract_coeff_trig_pow_n(ctx, term, 4, &SIN_COS_BUILTINS)
}

/// Extract `(coefficient, trig_name, argument)` from `k * sin(arg)^2` or `k * cos(arg)^2`.
pub fn extract_coeff_trig_pow2(
    ctx: &Context,
    term: ExprId,
) -> Option<(BigRational, &'static str, ExprId)> {
    extract_coeff_trig_pow_n(ctx, term, 2, &SIN_COS_BUILTINS)
}

/// Extract `(coefficient, trig_name, argument)` from `k * tan(arg)^2` or `k * cot(arg)^2`.
pub fn extract_coeff_tan_or_cot_pow2(
    ctx: &Context,
    term: ExprId,
) -> Option<(BigRational, &'static str, ExprId)> {
    extract_coeff_trig_pow_n(ctx, term, 2, &TAN_COT_BUILTINS)
}

/// Extract all `(is_sin, argument, coefficient_factors)` candidates from a term containing
/// trigonometric squares.
///
/// For `cos(u)^2*sin(x)^2`, this returns two candidates:
/// - `sin(x)^2` with residual `[cos(u)^2]`
/// - `cos(u)^2` with residual `[sin(x)^2]`
///
/// For higher powers `trig^n` (`n >= 3`), this decomposes as `trig^(n-2) * trig^2`,
/// so `sin(x)^3` contributes a `sin(x)^2` candidate with residual `[sin(x)]`.
pub fn extract_all_trig_squared_candidates(
    ctx: &mut Context,
    term: ExprId,
) -> Vec<(bool, ExprId, Vec<ExprId>)> {
    let mut results = Vec::new();

    {
        let mut case1_info = None;
        if let Expr::Pow(base, exp) = ctx.get(term) {
            let base = *base;
            let exp = *exp;
            if let Expr::Number(n) = ctx.get(exp) {
                if let Expr::Function(fn_id, args) = ctx.get(base) {
                    let builtin = ctx.builtin_of(*fn_id);
                    if args.len() == 1 && matches!(builtin, Some(BuiltinFn::Sin | BuiltinFn::Cos)) {
                        let is_sin = matches!(builtin, Some(BuiltinFn::Sin));
                        let arg = args[0];
                        let two = num_rational::BigRational::from_integer(2.into());
                        if *n == two {
                            case1_info = Some((is_sin, arg, base, None));
                        } else if *n > two && n.is_integer() {
                            let remainder = n - &two;
                            case1_info = Some((is_sin, arg, base, Some(remainder)));
                        }
                    }
                }
            }
        }

        if let Some((is_sin, arg, base, remainder_opt)) = case1_info {
            match remainder_opt {
                None => {
                    results.push((is_sin, arg, vec![]));
                }
                Some(remainder) => {
                    let one = num_rational::BigRational::from_integer(1.into());
                    let leftover = if remainder == one {
                        base
                    } else {
                        let remainder_id = ctx.add(Expr::Number(remainder));
                        ctx.add(Expr::Pow(base, remainder_id))
                    };
                    results.push((is_sin, arg, vec![leftover]));
                }
            }
            return results;
        }
        if matches!(ctx.get(term), Expr::Pow(_, _)) {
            return results;
        }
    }

    if let Expr::Mul(_, _) = ctx.get(term) {
        let mut factors = Vec::new();
        let mut stack = vec![term];
        while let Some(curr) = stack.pop() {
            if let Expr::Mul(l, r) = ctx.get(curr) {
                stack.push(*r);
                stack.push(*l);
            } else {
                factors.push(curr);
            }
        }

        struct CandidateInfo {
            factor_idx: usize,
            is_sin: bool,
            arg: ExprId,
            base: ExprId,
            remainder: Option<num_rational::BigRational>,
        }
        let mut candidates_info: Vec<CandidateInfo> = Vec::new();

        for (i, &f) in factors.iter().enumerate() {
            if let Expr::Pow(base, exp) = ctx.get(f) {
                let base = *base;
                let exp = *exp;
                if let Expr::Number(n) = ctx.get(exp) {
                    if let Expr::Function(fn_id, args) = ctx.get(base) {
                        let builtin = ctx.builtin_of(*fn_id);
                        if args.len() == 1
                            && matches!(builtin, Some(BuiltinFn::Sin | BuiltinFn::Cos))
                        {
                            let is_sin = matches!(builtin, Some(BuiltinFn::Sin));
                            let arg = args[0];
                            let two = num_rational::BigRational::from_integer(2.into());
                            if *n == two {
                                candidates_info.push(CandidateInfo {
                                    factor_idx: i,
                                    is_sin,
                                    arg,
                                    base,
                                    remainder: None,
                                });
                            } else if *n > two && n.is_integer() {
                                let rem = n - &two;
                                candidates_info.push(CandidateInfo {
                                    factor_idx: i,
                                    is_sin,
                                    arg,
                                    base,
                                    remainder: Some(rem),
                                });
                            }
                        }
                    }
                }
            }
        }

        for info in candidates_info {
            let mut coef_factors: Vec<ExprId> = factors
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != info.factor_idx)
                .map(|(_, &g)| g)
                .collect();

            if let Some(remainder) = info.remainder {
                let one = num_rational::BigRational::from_integer(1.into());
                let leftover = if remainder == one {
                    info.base
                } else {
                    let remainder_id = ctx.add(Expr::Number(remainder));
                    ctx.add(Expr::Pow(info.base, remainder_id))
                };
                coef_factors.push(leftover);
            }

            results.push((info.is_sin, info.arg, coef_factors));
        }
    }

    results
}

/// Return all possible decompositions of a term as:
/// `(is_sin, trig_argument, numeric_coefficient, residual_factors)`.
///
/// For `2*cos(x)^2*sin(u)^2`, this yields two entries:
/// - treating `cos(x)^2` as the chosen trig-square
/// - treating `sin(u)^2` as the chosen trig-square
pub fn decompose_term_with_residual_multi(
    ctx: &Context,
    term: ExprId,
) -> Vec<(bool, ExprId, BigRational, Vec<ExprId>)> {
    let mut factors = Vec::new();
    let mut stack = vec![term];
    let mut is_negated = false;

    if let Expr::Neg(inner) = ctx.get(term) {
        is_negated = true;
        stack = vec![*inner];
    }

    while let Some(curr) = stack.pop() {
        match ctx.get(curr) {
            Expr::Mul(l, r) => {
                stack.push(*r);
                stack.push(*l);
            }
            Expr::Neg(inner) => {
                is_negated = !is_negated;
                stack.push(*inner);
            }
            _ => factors.push(curr),
        }
    }

    let mut trig_indices: Vec<(usize, bool, ExprId)> = Vec::new();
    for (i, &f) in factors.iter().enumerate() {
        if let Expr::Pow(base, exp) = ctx.get(f) {
            if let Expr::Number(n) = ctx.get(*exp) {
                if *n == BigRational::from_integer(2.into()) {
                    if let Expr::Function(fn_id, args) = ctx.get(*base) {
                        let builtin = ctx.builtin_of(*fn_id);
                        if args.len() == 1
                            && matches!(builtin, Some(BuiltinFn::Sin | BuiltinFn::Cos))
                        {
                            trig_indices.push((
                                i,
                                matches!(builtin, Some(BuiltinFn::Sin)),
                                args[0],
                            ));
                        }
                    }
                }
            }
        }
    }

    let mut results = Vec::new();
    let base_numeric_coef = if is_negated {
        -BigRational::one()
    } else {
        BigRational::one()
    };

    for (trig_idx, is_sin, arg) in trig_indices {
        let mut numeric_coef = base_numeric_coef.clone();
        let mut residual: Vec<ExprId> = Vec::new();

        for (i, &f) in factors.iter().enumerate() {
            if i == trig_idx {
                continue;
            }
            if let Expr::Number(n) = ctx.get(f) {
                numeric_coef *= n.clone();
            } else {
                residual.push(f);
            }
        }

        results.push((is_sin, arg, numeric_coef, residual));
    }

    results.truncate(6);
    results
}

/// Flatten a product term and separate numeric coefficient from non-numeric factors.
pub fn extract_as_product(ctx: &Context, term: ExprId) -> Option<(Vec<ExprId>, BigRational)> {
    let mut factors = Vec::new();
    let mut stack = vec![term];
    let mut is_negated = false;

    if let Expr::Neg(inner) = ctx.get(term) {
        is_negated = true;
        stack = vec![*inner];
    }

    while let Some(curr) = stack.pop() {
        match ctx.get(curr) {
            Expr::Mul(l, r) => {
                stack.push(*r);
                stack.push(*l);
            }
            Expr::Neg(inner) => {
                is_negated = !is_negated;
                stack.push(*inner);
            }
            _ => factors.push(curr),
        }
    }

    let mut numeric_coef = BigRational::one();
    let mut non_numeric: Vec<ExprId> = Vec::new();

    for &f in &factors {
        if let Expr::Number(n) = ctx.get(f) {
            numeric_coef *= n.clone();
        } else {
            non_numeric.push(f);
        }
    }

    if is_negated {
        numeric_coef = -numeric_coef;
    }

    Some((non_numeric, numeric_coef))
}

/// Flatten a term into `(is_negated, factors)`.
///
/// For trig powers `sin/cos` with integer exponent `n >= 3`, decomposes
/// `trig^n` into `trig^(n-2)` and `trig^2` to expose a square factor.
pub fn flatten_with_trig_decomp(ctx: &mut Context, term: ExprId) -> (bool, Vec<ExprId>) {
    use num_traits::{One, Signed};

    let mut is_neg = false;
    let mut factors = Vec::new();
    let mut stack = vec![term];

    if let Expr::Neg(inner) = ctx.get(term) {
        is_neg = true;
        stack = vec![*inner];
    }

    while let Some(curr) = stack.pop() {
        match ctx.get(curr) {
            Expr::Mul(l, r) => {
                stack.push(*r);
                stack.push(*l);
            }
            Expr::Neg(inner) => {
                is_neg = !is_neg;
                stack.push(*inner);
            }
            _ => factors.push(curr),
        }
    }

    let mut final_factors = Vec::with_capacity(factors.len());
    for f in factors {
        if let Expr::Number(n) = ctx.get(f) {
            if n.is_negative() {
                is_neg = !is_neg;
                let abs_val = -n.clone();
                if abs_val == BigRational::one() {
                    continue;
                }
                let abs_id = ctx.add(Expr::Number(abs_val));
                final_factors.push(abs_id);
                continue;
            }
        }
        final_factors.push(f);
    }
    let factors = final_factors;

    let mut decomposed = Vec::new();
    for &f in &factors {
        if let Expr::Pow(base, exp) = ctx.get(f) {
            let base = *base;
            let exp = *exp;
            if let Expr::Number(n) = ctx.get(exp) {
                let two = BigRational::from_integer(2.into());
                if *n > two && n.is_integer() {
                    if let Expr::Function(fn_id, args) = ctx.get(base) {
                        let builtin = ctx.builtin_of(*fn_id);
                        if matches!(builtin, Some(BuiltinFn::Sin | BuiltinFn::Cos))
                            && args.len() == 1
                        {
                            let remainder = n - &two;
                            let leftover = if remainder == BigRational::one() {
                                base
                            } else {
                                let rem_id = ctx.add(Expr::Number(remainder));
                                ctx.add(Expr::Pow(base, rem_id))
                            };
                            let two_id = ctx.num(2);
                            let trig_sq = ctx.add(Expr::Pow(base, two_id));
                            decomposed.push(leftover);
                            decomposed.push(trig_sq);
                            continue;
                        }
                    }
                }
            }
        }
        decomposed.push(f);
    }

    (is_neg, decomposed)
}

fn extract_sin_or_cos_square(ctx: &Context, factor: ExprId) -> Option<(BuiltinFn, ExprId)> {
    if let Expr::Pow(base, exp) = ctx.get(factor) {
        if let Expr::Number(n) = ctx.get(*exp) {
            if *n == BigRational::from_integer(2.into()) {
                if let Expr::Function(fn_id, args) = ctx.get(*base) {
                    if args.len() == 1 {
                        let builtin = ctx.builtin_of(*fn_id)?;
                        if matches!(builtin, BuiltinFn::Sin | BuiltinFn::Cos) {
                            return Some((builtin, args[0]));
                        }
                    }
                }
            }
        }
    }
    None
}

fn complementary_sin_cos(builtin: BuiltinFn) -> Option<BuiltinFn> {
    match builtin {
        BuiltinFn::Sin => Some(BuiltinFn::Cos),
        BuiltinFn::Cos => Some(BuiltinFn::Sin),
        _ => None,
    }
}

fn find_single_extra_factor(
    ctx: &Context,
    smaller: &[ExprId],
    bigger: &[ExprId],
) -> Option<ExprId> {
    if bigger.len() != smaller.len() + 1 {
        return None;
    }

    let mut smaller_sorted = smaller.to_vec();
    let mut bigger_sorted = bigger.to_vec();
    smaller_sorted.sort_by(|a, b| cas_ast::ordering::compare_expr(ctx, *a, *b));
    bigger_sorted.sort_by(|a, b| cas_ast::ordering::compare_expr(ctx, *a, *b));

    let mut extra_factor = None;
    let mut smaller_index = 0usize;
    let mut bigger_index = 0usize;
    let mut mismatches = 0usize;

    while bigger_index < bigger_sorted.len() {
        if smaller_index < smaller_sorted.len()
            && cas_ast::ordering::compare_expr(
                ctx,
                smaller_sorted[smaller_index],
                bigger_sorted[bigger_index],
            ) == Ordering::Equal
        {
            smaller_index += 1;
            bigger_index += 1;
            continue;
        }

        mismatches += 1;
        if mismatches > 1 {
            return None;
        }
        extra_factor = Some(bigger_sorted[bigger_index]);
        bigger_index += 1;
    }

    if smaller_index != smaller_sorted.len() {
        return None;
    }

    extra_factor
}

/// Try to match `small_term + big_term` where `big_term = -small_term * trig(x)^2`.
/// Returns `(rewritten_expr, trig, other)` for the identity:
/// `R - R*trig(x)^2 = R*other(x)^2`.
pub fn try_high_power_pythagorean(
    ctx: &mut Context,
    small_term: ExprId,
    big_term: ExprId,
) -> Option<(ExprId, BuiltinFn, BuiltinFn)> {
    let (small_neg, small_factors) = flatten_with_trig_decomp(ctx, small_term);
    let (big_neg, big_factors) = flatten_with_trig_decomp(ctx, big_term);

    if small_neg == big_neg {
        return None;
    }

    let extra_factor = find_single_extra_factor(ctx, &small_factors, &big_factors)?;
    let (trig, arg) = extract_sin_or_cos_square(ctx, extra_factor)?;
    let other = complementary_sin_cos(trig)?;

    let other_fn = ctx.call_builtin(other, vec![arg]);
    let two = ctx.num(2);
    let other_sq = ctx.add(Expr::Pow(other_fn, two));
    let result = crate::expr_rewrite::smart_mul(ctx, small_term, other_sq);

    Some((result, trig, other))
}

/// Check if `(c_term, t_term)` matches `k - k*trig(x)^2` with `trig ∈ {sin, cos}`.
/// Returns `(rewritten_expr, trig, other)` for the identity:
/// `k - k*trig(x)^2 = k*other(x)^2`.
pub fn check_pythagorean_pattern(
    ctx: &mut Context,
    c_term: ExprId,
    t_term: ExprId,
) -> Option<(ExprId, BuiltinFn, BuiltinFn)> {
    let (base_term, is_neg) = if let Expr::Neg(inner) = ctx.get(t_term) {
        (*inner, true)
    } else {
        (t_term, false)
    };

    let mut factors = Vec::new();
    let mut stack = vec![base_term];
    while let Some(curr) = stack.pop() {
        if let Expr::Mul(l, r) = ctx.get(curr) {
            stack.push(*r);
            stack.push(*l);
        } else {
            factors.push(curr);
        }
    }

    let mut trig_index = None;
    let mut trig_builtin = None;
    let mut trig_arg = None;
    for (i, &factor) in factors.iter().enumerate() {
        if let Some((builtin, arg)) = extract_sin_or_cos_square(ctx, factor) {
            trig_index = Some(i);
            trig_builtin = Some(builtin);
            trig_arg = Some(arg);
            break;
        }
    }

    let trig_idx = trig_index?;
    let trig = trig_builtin?;
    let arg = trig_arg?;
    let other = complementary_sin_cos(trig)?;

    let mut coeff_factors = Vec::new();
    if is_neg {
        coeff_factors.push(ctx.num(-1));
    }
    for (i, &factor) in factors.iter().enumerate() {
        if i != trig_idx {
            coeff_factors.push(factor);
        }
    }

    let coeff = if coeff_factors.is_empty() {
        ctx.num(1)
    } else {
        let mut acc = coeff_factors[0];
        for factor in coeff_factors.iter().skip(1) {
            acc = crate::expr_rewrite::smart_mul(ctx, acc, *factor);
        }
        acc
    };

    let neg_coeff = if let Expr::Number(n) = ctx.get(coeff) {
        ctx.add(Expr::Number(-n.clone()))
    } else if let Expr::Neg(inner) = ctx.get(coeff) {
        *inner
    } else {
        ctx.add(Expr::Neg(coeff))
    };

    if cas_ast::ordering::compare_expr(ctx, c_term, neg_coeff) != Ordering::Equal {
        return None;
    }

    let other_fn = ctx.call_builtin(other, vec![arg]);
    let two = ctx.num(2);
    let other_sq = ctx.add(Expr::Pow(other_fn, two));
    let result = crate::expr_rewrite::smart_mul(ctx, c_term, other_sq);
    Some((result, trig, other))
}

/// Extract `coeff * sin(arg)^2 * cos(arg)^2` from a product term.
/// Returns `(coeff, arg)` when both squared trig factors share the same argument.
pub fn extract_sin2_cos2_product(ctx: &mut Context, term: ExprId) -> Option<(ExprId, ExprId)> {
    let factors = mul_leaves(ctx, term);
    if factors.len() < 2 {
        return None;
    }

    let mut sin2_arg: Option<ExprId> = None;
    let mut cos2_arg: Option<ExprId> = None;
    let mut other_factors: Vec<ExprId> = Vec::new();

    for factor in factors {
        if let Some((arg, name)) = extract_trig_pow2(ctx, factor) {
            match name {
                "sin" if sin2_arg.is_none() => sin2_arg = Some(arg),
                "cos" if cos2_arg.is_none() => cos2_arg = Some(arg),
                _ => other_factors.push(factor),
            }
        } else {
            other_factors.push(factor);
        }
    }

    let sin_arg = sin2_arg?;
    let cos_arg = cos2_arg?;
    if cas_ast::ordering::compare_expr(ctx, sin_arg, cos_arg) != Ordering::Equal {
        return None;
    }

    let coeff = if other_factors.is_empty() {
        ctx.num(1)
    } else if other_factors.len() == 1 {
        other_factors[0]
    } else {
        let mut acc = other_factors[0];
        for factor in &other_factors[1..] {
            acc = ctx.add(Expr::Mul(acc, *factor));
        }
        acc
    };

    Some((coeff, sin_arg))
}

/// Check if a coefficient expression is exactly 3.
pub fn coeff_is_three(ctx: &mut Context, coeff: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(coeff) {
        return n.is_integer() && *n.numer() == 3.into();
    }
    if let Some(v) = as_rational_const(ctx, coeff) {
        return v == num_rational::BigRational::from_integer(3.into());
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn extract_pow_variants_detect_sin_and_cos() {
        let mut ctx = Context::new();
        let s2 = parse("sin(x)^2", &mut ctx).expect("s2");
        let c4 = parse("cos(y)^4", &mut ctx).expect("c4");
        let s6 = parse("sin(z)^6", &mut ctx).expect("s6");

        assert!(extract_trig_pow2(&ctx, s2).is_some());
        assert!(extract_trig_pow4(&ctx, c4).is_some());
        assert!(extract_trig_pow6(&ctx, s6).is_some());
    }

    #[test]
    fn extract_all_trig_squared_candidates_handles_two_trig_squares() {
        let mut ctx = Context::new();
        let term = parse("cos(u)^2*sin(x)^2", &mut ctx).expect("term");
        let u = parse("u", &mut ctx).expect("u");
        let x = parse("x", &mut ctx).expect("x");

        let candidates = extract_all_trig_squared_candidates(&mut ctx, term);
        assert_eq!(candidates.len(), 2);

        let mut saw_sin = false;
        let mut saw_cos = false;
        for (is_sin, arg, residual) in candidates {
            assert_eq!(residual.len(), 1);
            if is_sin {
                saw_sin = true;
                assert_eq!(
                    cas_ast::ordering::compare_expr(&ctx, arg, x),
                    Ordering::Equal
                );
            } else {
                saw_cos = true;
                assert_eq!(
                    cas_ast::ordering::compare_expr(&ctx, arg, u),
                    Ordering::Equal
                );
            }
        }

        assert!(saw_sin && saw_cos);
    }

    #[test]
    fn extract_all_trig_squared_candidates_decomposes_higher_power() {
        let mut ctx = Context::new();
        let term = parse("sin(x)^3", &mut ctx).expect("term");
        let x = parse("x", &mut ctx).expect("x");
        let sin_x = parse("sin(x)", &mut ctx).expect("sin_x");

        let candidates = extract_all_trig_squared_candidates(&mut ctx, term);
        assert_eq!(candidates.len(), 1);
        let (is_sin, arg, residual) = &candidates[0];

        assert!(*is_sin);
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, *arg, x),
            Ordering::Equal
        );
        assert_eq!(residual.len(), 1);
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, residual[0], sin_x),
            Ordering::Equal
        );
    }

    #[test]
    fn decompose_term_with_residual_multi_returns_both_trig_choices() {
        let mut ctx = Context::new();
        let term = parse("2*cos(x)^2*sin(u)^2", &mut ctx).expect("term");
        let x = parse("x", &mut ctx).expect("x");
        let u = parse("u", &mut ctx).expect("u");

        let decomps = decompose_term_with_residual_multi(&ctx, term);
        assert_eq!(decomps.len(), 2);

        let mut saw_sin_u = false;
        let mut saw_cos_x = false;
        for (is_sin, arg, coeff, residual) in decomps {
            assert_eq!(coeff, BigRational::from_integer(2.into()));
            assert_eq!(residual.len(), 1);
            if is_sin {
                saw_sin_u = true;
                assert_eq!(
                    cas_ast::ordering::compare_expr(&ctx, arg, u),
                    Ordering::Equal
                );
            } else {
                saw_cos_x = true;
                assert_eq!(
                    cas_ast::ordering::compare_expr(&ctx, arg, x),
                    Ordering::Equal
                );
            }
        }

        assert!(saw_sin_u && saw_cos_x);
    }

    #[test]
    fn extract_as_product_splits_numeric_coefficient() {
        let mut ctx = Context::new();
        let term = parse("-2*a*b", &mut ctx).expect("term");
        let (factors, coeff) = extract_as_product(&ctx, term).expect("decomposition");

        assert_eq!(coeff, BigRational::from_integer((-2).into()));
        assert_eq!(factors.len(), 2);
    }

    #[test]
    fn flatten_with_trig_decomp_splits_trig_cubes_and_sign() {
        let mut ctx = Context::new();
        let term = parse("-4*sin(x)^3", &mut ctx).expect("term");
        let four = parse("4", &mut ctx).expect("four");
        let sin_x = parse("sin(x)", &mut ctx).expect("sin_x");
        let sin_x_sq = parse("sin(x)^2", &mut ctx).expect("sin_x_sq");

        let (is_neg, factors) = flatten_with_trig_decomp(&mut ctx, term);

        assert!(is_neg);
        assert_eq!(factors.len(), 3);
        assert!(factors
            .iter()
            .any(|&f| cas_ast::ordering::compare_expr(&ctx, f, four) == Ordering::Equal));
        assert!(factors
            .iter()
            .any(|&f| cas_ast::ordering::compare_expr(&ctx, f, sin_x) == Ordering::Equal));
        assert!(factors
            .iter()
            .any(|&f| cas_ast::ordering::compare_expr(&ctx, f, sin_x_sq) == Ordering::Equal));
    }

    #[test]
    fn try_high_power_pythagorean_rewrites_trig_cube_pattern() {
        let mut ctx = Context::new();
        let small = parse("sin(x)", &mut ctx).expect("small");
        let big = parse("-sin(x)^3", &mut ctx).expect("big");
        let expected = parse("sin(x)*cos(x)^2", &mut ctx).expect("expected");

        let (result, trig, other) =
            try_high_power_pythagorean(&mut ctx, small, big).expect("pattern match");
        assert_eq!(trig, BuiltinFn::Sin);
        assert_eq!(other, BuiltinFn::Cos);
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, result, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn check_pythagorean_pattern_rewrites_one_minus_sin_sq() {
        let mut ctx = Context::new();
        let c_term = parse("1", &mut ctx).expect("c_term");
        let t_term = parse("-sin(x)^2", &mut ctx).expect("t_term");
        let expected = parse("cos(x)^2", &mut ctx).expect("expected");

        let (result, trig, other) =
            check_pythagorean_pattern(&mut ctx, c_term, t_term).expect("pattern match");
        assert_eq!(trig, BuiltinFn::Sin);
        assert_eq!(other, BuiltinFn::Cos);
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, result, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn extract_coeff_trig_pow4_detects_sign_and_coefficient() {
        let mut ctx = Context::new();
        let term1 = parse("3*sin(t)^4", &mut ctx).expect("term1");
        let term2 = parse("-cos(t)^4", &mut ctx).expect("term2");
        let t = parse("t", &mut ctx).expect("t");

        let (coef1, name1, arg1) = extract_coeff_trig_pow4(&ctx, term1).expect("term1 match");
        let (coef2, name2, arg2) = extract_coeff_trig_pow4(&ctx, term2).expect("term2 match");

        assert_eq!(coef1, BigRational::from_integer(3.into()));
        assert_eq!(name1, "sin");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, arg1, t),
            Ordering::Equal
        );

        assert_eq!(coef2, BigRational::from_integer((-1).into()));
        assert_eq!(name2, "cos");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, arg2, t),
            Ordering::Equal
        );
    }

    #[test]
    fn extract_coeff_trig_pow4_rejects_non_numeric_residuals() {
        let mut ctx = Context::new();
        let bad1 = parse("x*sin(t)^4", &mut ctx).expect("bad1");
        let bad2 = parse("sin(t)^4*cos(t)^4", &mut ctx).expect("bad2");

        assert!(extract_coeff_trig_pow4(&ctx, bad1).is_none());
        assert!(extract_coeff_trig_pow4(&ctx, bad2).is_none());
    }

    #[test]
    fn extract_coeff_trig_pow2_detects_sign_and_coefficient() {
        let mut ctx = Context::new();
        let term1 = parse("2*sin(t)^2", &mut ctx).expect("term1");
        let term2 = parse("-3*cos(t)^2", &mut ctx).expect("term2");
        let t = parse("t", &mut ctx).expect("t");

        let (coef1, name1, arg1) = extract_coeff_trig_pow2(&ctx, term1).expect("term1 match");
        let (coef2, name2, arg2) = extract_coeff_trig_pow2(&ctx, term2).expect("term2 match");

        assert_eq!(coef1, BigRational::from_integer(2.into()));
        assert_eq!(name1, "sin");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, arg1, t),
            Ordering::Equal
        );

        assert_eq!(coef2, BigRational::from_integer((-3).into()));
        assert_eq!(name2, "cos");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, arg2, t),
            Ordering::Equal
        );
    }

    #[test]
    fn extract_coeff_tan_or_cot_pow2_detects_sign_and_coefficient() {
        let mut ctx = Context::new();
        let term1 = parse("4*tan(u)^2", &mut ctx).expect("term1");
        let term2 = parse("-cot(u)^2", &mut ctx).expect("term2");
        let u = parse("u", &mut ctx).expect("u");

        let (coef1, name1, arg1) = extract_coeff_tan_or_cot_pow2(&ctx, term1).expect("term1 match");
        let (coef2, name2, arg2) = extract_coeff_tan_or_cot_pow2(&ctx, term2).expect("term2 match");

        assert_eq!(coef1, BigRational::from_integer(4.into()));
        assert_eq!(name1, "tan");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, arg1, u),
            Ordering::Equal
        );

        assert_eq!(coef2, BigRational::from_integer((-1).into()));
        assert_eq!(name2, "cot");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, arg2, u),
            Ordering::Equal
        );
    }

    #[test]
    fn extract_sin2_cos2_product_finds_coeff_and_arg() {
        let mut ctx = Context::new();
        let term = parse("3*sin(t)^2*cos(t)^2", &mut ctx).expect("term");
        let (coeff, arg) = extract_sin2_cos2_product(&mut ctx, term).expect("match");
        let three = parse("3", &mut ctx).expect("three");
        let t = parse("t", &mut ctx).expect("t");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, coeff, three),
            Ordering::Equal
        );
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, arg, t),
            Ordering::Equal
        );
        assert!(coeff_is_three(&mut ctx, coeff));
    }
}
