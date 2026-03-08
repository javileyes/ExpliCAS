//! Support for diff-squares denominator rationalization.
//!
//! Rewrites fractions with binomial square-root denominators by multiplying by the
//! conjugate:
//! `num / (a ± b)` -> `num*(a ∓ b) / (a^2 - b^2)`, when at least one side is a square root.

use crate::build::mul2_raw;
use crate::expr_destructure::{as_add, as_sub};
use crate::expr_terms::{
    build_sum, collect_additive_terms_flat_add as collect_additive_terms, contains_irrational,
};
use crate::fraction_factors::collect_mul_factors_flat as collect_mul_factors;
use crate::fraction_forms::split_binomial_den;
use crate::root_forms::extract_root_base_and_index as extract_root_base;
use cas_ast::views::FractionParts;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::ToPrimitive;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RationalizeDiffSquaresRewrite {
    pub rewritten: ExprId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalizeNthRootBinomialRewrite {
    pub rewritten: ExprId,
    pub desc: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CancelNthRootBinomialFactorRewrite {
    pub rewritten: ExprId,
    pub desc: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneralizedRationalizationRewrite {
    pub rewritten: ExprId,
    pub desc: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SqrtConjugateCollapseMatch {
    pub rewritten: ExprId,
    pub other: ExprId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SqrtConjugateCollapseGate {
    pub allow: bool,
    pub assumed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SqrtConjugateCollapseRewrite {
    pub rewritten: ExprId,
    pub assumed_nonnegative_target: Option<ExprId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RationalizeProductDenominatorRewrite {
    pub rewritten: ExprId,
}

/// Try to rationalize a binomial denominator with square roots using difference of squares.
pub fn try_rewrite_rationalize_denominator_diff_squares_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<RationalizeDiffSquaresRewrite> {
    let fp = FractionParts::from(&*ctx, expr);
    if !fp.is_fraction() {
        return None;
    }

    let (num, den, _) = fp.to_num_den(ctx);
    let (l, r, is_add, r_is_abs_one) = split_binomial_den(ctx, den)?;

    // Difference-of-squares rationalization applies only to square roots.
    let is_sqrt_root = |e: ExprId| -> bool {
        match ctx.get(e) {
            Expr::Pow(_, exp) => {
                if let Expr::Number(n) = ctx.get(*exp) {
                    if !n.is_integer() && n.denom() == &num_bigint::BigInt::from(2) {
                        return true;
                    }
                }
                false
            }
            Expr::Function(fn_id, _) => ctx.is_builtin(*fn_id, BuiltinFn::Sqrt),
            _ => false,
        }
    };

    let l_sqrt = is_sqrt_root(l);
    let r_sqrt = is_sqrt_root(r);
    if !l_sqrt && !r_sqrt {
        return None;
    }

    let conjugate = if is_add {
        ctx.add(Expr::Sub(l, r))
    } else {
        ctx.add(Expr::Add(l, r))
    };

    let new_num = mul2_raw(ctx, num, conjugate);

    let two = ctx.num(2);
    let one = ctx.num(1);
    let l_sq = ctx.add(Expr::Pow(l, two));
    let r_sq = if r_is_abs_one {
        one
    } else {
        ctx.add(Expr::Pow(r, two))
    };
    let new_den = ctx.add(Expr::Sub(l_sq, r_sq));

    let rewritten = ctx.add(Expr::Div(new_num, new_den));
    Some(RationalizeDiffSquaresRewrite { rewritten })
}

fn ordinal(n: u32) -> &'static str {
    match n {
        3 => "cube",
        4 => "4th",
        5 => "5th",
        6 => "6th",
        7 => "7th",
        8 => "8th",
        _ => "nth",
    }
}

/// Rationalize binomial denominators with nth roots (`n >= 3`) using geometric sums.
pub fn try_rewrite_rationalize_nth_root_binomial_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<RationalizeNthRootBinomialRewrite> {
    let fp = FractionParts::from(&*ctx, expr);
    if !fp.is_fraction() {
        return None;
    }

    let (num, den, _) = fp.to_num_den(ctx);

    let extract_nth_root = |e: ExprId| -> Option<(ExprId, u32)> {
        if let Expr::Pow(base, exp) = ctx.get(e) {
            if let Expr::Number(ev) = ctx.get(*exp) {
                if ev.numer() == &num_bigint::BigInt::from(1) {
                    if let Some(denom) = ev.denom().to_u32() {
                        if denom >= 3 {
                            return Some((*base, denom));
                        }
                    }
                }
            }
        }
        None
    };

    let mut sign_flip = false;
    let (t, r, base, n, is_sub) = if let Some((l, r_side)) = as_add(ctx, den) {
        if let Some((base, n)) = extract_nth_root(l) {
            (l, r_side, base, n, false)
        } else if let Some((base, n)) = extract_nth_root(r_side) {
            (r_side, l, base, n, false)
        } else {
            return None;
        }
    } else if let Some((l, r_side)) = as_sub(ctx, den) {
        if let Some((base, n)) = extract_nth_root(l) {
            (l, r_side, base, n, true)
        } else if let Some((base, n)) = extract_nth_root(r_side) {
            sign_flip = true;
            (r_side, l, base, n, true)
        } else {
            return None;
        }
    } else {
        return None;
    };

    if n > 8 {
        return None;
    }

    let mut m_terms: Vec<ExprId> = Vec::new();
    for k in 0..n {
        let exp_t = n - 1 - k;
        let exp_r = k;

        let t_part = if exp_t == 0 {
            ctx.num(1)
        } else if exp_t == 1 {
            t
        } else {
            let exp_val = num_rational::BigRational::new(
                num_bigint::BigInt::from(exp_t),
                num_bigint::BigInt::from(n),
            );
            let exp_node = ctx.add(Expr::Number(exp_val));
            ctx.add(Expr::Pow(base, exp_node))
        };

        let r_part = if exp_r == 0 {
            ctx.num(1)
        } else if exp_r == 1 {
            r
        } else {
            let exp_node = ctx.num(exp_r as i64);
            ctx.add(Expr::Pow(r, exp_node))
        };

        let mut term = mul2_raw(ctx, t_part, r_part);
        if !is_sub && k % 2 == 1 {
            term = ctx.add(Expr::Neg(term));
        }
        m_terms.push(term);
    }

    let multiplier = build_sum(ctx, &m_terms);
    let mut new_num = mul2_raw(ctx, num, multiplier);
    if sign_flip {
        new_num = ctx.add(Expr::Neg(new_num));
    }

    let r_to_n = {
        let exp_node = ctx.num(n as i64);
        ctx.add(Expr::Pow(r, exp_node))
    };

    let new_den = if is_sub || n % 2 == 0 {
        ctx.add(Expr::Sub(base, r_to_n))
    } else {
        ctx.add(Expr::Add(base, r_to_n))
    };

    let rewritten = ctx.add(Expr::Div(new_num, new_den));
    Some(RationalizeNthRootBinomialRewrite {
        rewritten,
        desc: format!("Rationalize {} root binomial (geometric sum)", ordinal(n)),
    })
}

/// Cancel nth-root binomial factors:
/// `(u ± r^n) / (u^(1/n) ± r)` -> geometric series quotient.
pub fn try_rewrite_cancel_nth_root_binomial_factor_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<CancelNthRootBinomialFactorRewrite> {
    let fp = FractionParts::from(&*ctx, expr);
    if !fp.is_fraction() {
        return None;
    }
    let (num, den, _) = fp.to_num_den(ctx);

    let (left, right, den_is_add) = if let Some((l, r)) = as_add(ctx, den) {
        (l, r, true)
    } else if let Some((l, r)) = as_sub(ctx, den) {
        (l, r, false)
    } else {
        return None;
    };

    let extract_nth_root = |e: ExprId| -> Option<(ExprId, u32)> {
        if let Expr::Pow(base, exp) = ctx.get(e) {
            if let Expr::Number(ev) = ctx.get(*exp) {
                if ev.numer() == &num_bigint::BigInt::from(1) {
                    if let Some(denom) = ev.denom().to_u32() {
                        if denom >= 2 {
                            return Some((*base, denom));
                        }
                    }
                }
            }
        }
        None
    };

    let (t, r, u, n) = if let Some((base, denom)) = extract_nth_root(left) {
        (left, right, base, denom)
    } else if let Some((base, denom)) = extract_nth_root(right) {
        (right, left, base, denom)
    } else {
        return None;
    };

    let r_val = match ctx.get(r) {
        Expr::Number(rv) => rv.clone(),
        _ => return None,
    };
    if n > 8 {
        return None;
    }

    let r_to_n = r_val.pow(n as i32);
    let (expected_num_is_add, expected_r_val) = if den_is_add {
        if n % 2 == 1 {
            (true, r_to_n.clone())
        } else {
            return None;
        }
    } else {
        (false, r_to_n.clone())
    };

    let (num_left, num_right, num_is_add) = if let Some((l, rr)) = as_add(ctx, num) {
        (l, rr, true)
    } else if let Some((l, rr)) = as_sub(ctx, num) {
        (l, rr, false)
    } else {
        return None;
    };
    if num_is_add != expected_num_is_add {
        return None;
    }

    let actual_r_n =
        if cas_ast::ordering::compare_expr(ctx, num_left, u) == std::cmp::Ordering::Equal {
            num_right
        } else if cas_ast::ordering::compare_expr(ctx, num_right, u) == std::cmp::Ordering::Equal {
            num_left
        } else {
            return None;
        };

    let actual_r_n_val = match ctx.get(actual_r_n) {
        Expr::Number(v) => v.clone(),
        _ => return None,
    };
    if actual_r_n_val != expected_r_val {
        return None;
    }

    let mut terms: Vec<ExprId> = Vec::new();
    for k in 0..n {
        let exp_t = n - 1 - k;
        let exp_r = k;

        let t_part = if exp_t == 0 {
            ctx.num(1)
        } else if exp_t == 1 {
            t
        } else {
            let exp_val = num_rational::BigRational::new(
                num_bigint::BigInt::from(exp_t),
                num_bigint::BigInt::from(n),
            );
            let exp_node = ctx.add(Expr::Number(exp_val));
            ctx.add(Expr::Pow(u, exp_node))
        };

        let r_part = if exp_r == 0 {
            ctx.num(1)
        } else {
            let r_pow_k = r_val.pow(exp_r as i32);
            ctx.add(Expr::Number(r_pow_k))
        };

        let mut term = mul2_raw(ctx, t_part, r_part);
        if den_is_add && k % 2 == 1 {
            term = ctx.add(Expr::Neg(term));
        }
        terms.push(term);
    }

    let rewritten = build_sum(ctx, &terms);
    Some(CancelNthRootBinomialFactorRewrite {
        rewritten,
        desc: format!("Cancel {} root binomial factor", ordinal(n)),
    })
}

/// Rationalize denominator with 3+ additive terms by grouping and conjugation.
///
/// Strategy:
/// - `num / (t1 + t2 + ... + tn)` with `n >= 3` and at least one irrational term.
/// - Group as `g = t1 + ... + t(n-1)` and `last = tn`.
/// - Multiply by conjugate `(g - last) / (g - last)`:
///   `num*(g-last) / (g^2 - last^2)`.
/// - Expand denominator to expose potential simplifications.
pub fn try_rewrite_generalized_rationalization_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<GeneralizedRationalizationRewrite> {
    let fp = FractionParts::from(&*ctx, expr);
    if !fp.is_fraction() {
        return None;
    }

    let (num, den, _) = fp.to_num_den(ctx);
    let terms = collect_additive_terms(ctx, den);

    // Binary case is handled by dedicated diff-squares rules.
    if terms.len() < 3 {
        return None;
    }

    if !terms.iter().any(|&t| contains_irrational(ctx, t)) {
        return None;
    }

    let last_term = terms[terms.len() - 1];
    let group = build_sum(ctx, &terms[..terms.len() - 1]);
    let conjugate = ctx.add(Expr::Sub(group, last_term));

    let new_num = mul2_raw(ctx, num, conjugate);

    let two = ctx.num(2);
    let group_sq = ctx.add(Expr::Pow(group, two));
    let last_sq = ctx.add(Expr::Pow(last_term, two));
    let new_den = ctx.add(Expr::Sub(group_sq, last_sq));
    let new_den_expanded = crate::expand_ops::expand(ctx, new_den);

    let rewritten = ctx.add(Expr::Div(new_num, new_den_expanded));
    Some(GeneralizedRationalizationRewrite {
        rewritten,
        desc: format!(
            "Rationalize: group {} terms and multiply by conjugate",
            terms.len()
        ),
    })
}

/// Rationalize denominators that are products containing a root factor.
///
/// Examples:
/// - `num / (a * sqrt(b)) -> num*sqrt(b) / (a*b)`
/// - `num / root(b, n) -> num*root(b, n) / b` (when `b` is non-numeric)
///
/// Numeric-radicand cases are intentionally skipped because power rules
/// generally produce cleaner canonical forms.
pub fn try_rewrite_rationalize_product_denominator_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<RationalizeProductDenominatorRewrite> {
    let fp = FractionParts::from(&*ctx, expr);
    if !fp.is_fraction() {
        return None;
    }

    let (num, den, _) = fp.to_num_den(ctx);
    let factors = collect_mul_factors(ctx, den);

    let mut root_factor = None;
    let mut non_root_factors = Vec::new();
    for &factor in &factors {
        if extract_root_base(ctx, factor).is_some() && root_factor.is_none() {
            root_factor = Some(factor);
        } else {
            non_root_factors.push(factor);
        }
    }
    let root = root_factor?;

    if non_root_factors.is_empty() {
        let (radicand, _index) = extract_root_base(ctx, root)?;
        let is_binomial_radical = matches!(ctx.get(radicand), Expr::Add(_, _) | Expr::Sub(_, _));
        if is_binomial_radical && contains_irrational(ctx, num) {
            return None;
        }
        if matches!(ctx.get(radicand), Expr::Number(_)) {
            return None;
        }

        let new_num = mul2_raw(ctx, num, root);
        let rewritten = ctx.add(Expr::Div(new_num, radicand));
        return Some(RationalizeProductDenominatorRewrite { rewritten });
    }

    if let Some((radicand, _index)) = extract_root_base(ctx, root) {
        if matches!(ctx.get(radicand), Expr::Number(_)) {
            return None;
        }
    }

    let (radicand, index) = extract_root_base(ctx, root)?;
    let index_int = match ctx.get(index) {
        Expr::Number(n) if n.is_integer() => n.to_integer(),
        _ => return None,
    };
    if index_int <= BigInt::from(1) {
        return None;
    }

    let conjugate_exp = BigRational::new(&index_int - BigInt::from(1), index_int.clone());
    let conjugate_exp_id = ctx.add(Expr::Number(conjugate_exp));
    let conjugate_power = ctx.add(Expr::Pow(radicand, conjugate_exp_id));

    let new_num = mul2_raw(ctx, num, conjugate_power);
    let mut new_den = radicand;
    for &factor in &non_root_factors {
        new_den = mul2_raw(ctx, new_den, factor);
    }

    let rewritten = ctx.add(Expr::Div(new_num, new_den));
    Some(RationalizeProductDenominatorRewrite { rewritten })
}

/// Match and build `sqrt(A) * B -> sqrt(B)` when `A` and `B` are conjugates.
///
/// This function performs only structural matching and candidate construction.
/// Domain gating (`B >= 0`) is handled by caller policy.
pub fn try_match_sqrt_conjugate_collapse_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<SqrtConjugateCollapseMatch> {
    use cas_ast::views::MulChainView;

    if !matches!(ctx.get(expr), Expr::Mul(_, _)) {
        return None;
    }
    let mv = MulChainView::from(&*ctx, expr);
    if mv.factors.len() != 2 {
        return None;
    }

    let unwrap_sqrt = |e: ExprId| -> Option<ExprId> {
        match ctx.get(e) {
            Expr::Pow(base, exp) => {
                if let Expr::Number(n) = ctx.get(*exp) {
                    let half = BigRational::new(1.into(), 2.into());
                    if n == &half {
                        return Some(*base);
                    }
                }
                None
            }
            Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 =>
            {
                Some(args[0])
            }
            _ => None,
        }
    };

    let (sqrt_arg, other) = if let Some(a) = unwrap_sqrt(mv.factors[0]) {
        (a, mv.factors[1])
    } else if let Some(a) = unwrap_sqrt(mv.factors[1]) {
        (a, mv.factors[0])
    } else {
        return None;
    };

    struct SignedBinomial {
        p: ExprId,
        s: ExprId,
        s_positive: bool,
    }
    let parse_signed_binomial = |e: ExprId| -> Option<SignedBinomial> {
        match ctx.get(e) {
            Expr::Add(l, r) => {
                if let Expr::Neg(inner) = ctx.get(*r) {
                    Some(SignedBinomial {
                        p: *l,
                        s: *inner,
                        s_positive: false,
                    })
                } else {
                    Some(SignedBinomial {
                        p: *l,
                        s: *r,
                        s_positive: true,
                    })
                }
            }
            Expr::Sub(l, r) => Some(SignedBinomial {
                p: *l,
                s: *r,
                s_positive: false,
            }),
            _ => None,
        }
    };

    let a_bin = parse_signed_binomial(sqrt_arg)?;
    let b_bin = parse_signed_binomial(other)?;

    let p_matches =
        cas_ast::ordering::compare_expr(ctx, a_bin.p, b_bin.p) == std::cmp::Ordering::Equal;
    let s_matches =
        cas_ast::ordering::compare_expr(ctx, a_bin.s, b_bin.s) == std::cmp::Ordering::Equal;
    let signs_opposite = a_bin.s_positive != b_bin.s_positive;
    if !p_matches || !s_matches || !signs_opposite {
        return None;
    }

    // Require surd term in conjugate form.
    unwrap_sqrt(a_bin.s)?;

    let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
    let rewritten = ctx.add(Expr::Pow(other, half));
    Some(SqrtConjugateCollapseMatch { rewritten, other })
}

/// Apply sqrt-conjugate collapse with caller-provided nonnegative gating policy.
pub fn try_rewrite_sqrt_conjugate_collapse_expr_with<FGate>(
    ctx: &mut Context,
    expr: ExprId,
    mut gate: FGate,
) -> Option<SqrtConjugateCollapseRewrite>
where
    FGate: FnMut(&mut Context, ExprId) -> SqrtConjugateCollapseGate,
{
    let matched = try_match_sqrt_conjugate_collapse_expr(ctx, expr)?;
    let decision = gate(ctx, matched.other);
    if !decision.allow {
        return None;
    }
    let assumed_nonnegative_target = if decision.assumed {
        Some(matched.other)
    } else {
        None
    };
    Some(SqrtConjugateCollapseRewrite {
        rewritten: matched.rewritten,
        assumed_nonnegative_target,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        try_match_sqrt_conjugate_collapse_expr, try_rewrite_cancel_nth_root_binomial_factor_expr,
        try_rewrite_generalized_rationalization_expr,
        try_rewrite_rationalize_denominator_diff_squares_expr,
        try_rewrite_rationalize_product_denominator_expr,
        try_rewrite_sqrt_conjugate_collapse_expr_with, SqrtConjugateCollapseGate,
    };
    use crate::expr_terms::contains_irrational;
    use cas_ast::views::FractionParts;
    use cas_ast::Context;
    use cas_ast::Expr;
    use cas_parser::parse;

    #[test]
    fn rewrites_sqrt_plus_one_denominator() {
        let mut ctx = Context::new();
        let expr = parse("1/(sqrt(x)+1)", &mut ctx).expect("parse");
        let rewrite =
            try_rewrite_rationalize_denominator_diff_squares_expr(&mut ctx, expr).expect("rewrite");
        let fp = FractionParts::from(&ctx, rewrite.rewritten);
        assert!(fp.is_fraction());
        let (_, den, _) = fp.to_num_den(&mut ctx);
        assert!(!contains_irrational(&ctx, den));
    }

    #[test]
    fn rejects_non_sqrt_binomial_denominator() {
        let mut ctx = Context::new();
        let expr = parse("1/(x+1)", &mut ctx).expect("parse");
        assert!(try_rewrite_rationalize_denominator_diff_squares_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn rejects_non_fraction_expression() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x)+1", &mut ctx).expect("parse");
        assert!(try_rewrite_rationalize_denominator_diff_squares_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn cancels_nth_root_binomial_factor_difference() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let one_third = ctx.rational(1, 3);
        let x_cubert = ctx.add(Expr::Pow(x, one_third));
        let numerator = ctx.add(Expr::Add(x, one));
        let denominator = ctx.add(Expr::Add(x_cubert, one));
        let expr = ctx.add(Expr::Div(numerator, denominator));
        let rewrite = try_rewrite_cancel_nth_root_binomial_factor_expr(&mut ctx, expr)
            .expect("rewrite should apply");
        let rendered = cas_formatter::render_expr(&ctx, rewrite.rewritten);
        assert!(rendered.contains("x^(2/3)"));
        assert_eq!(rewrite.desc, "Cancel cube root binomial factor");
    }

    #[test]
    fn rejects_cancel_when_numerator_not_matching_power_pattern() {
        let mut ctx = Context::new();
        let expr = parse("(x+2)/(x^(1/3)+1)", &mut ctx).expect("parse");
        assert!(try_rewrite_cancel_nth_root_binomial_factor_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn matches_sqrt_conjugate_collapse_pattern() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x+sqrt(y))*(x-sqrt(y))", &mut ctx).expect("parse");
        let matched = try_match_sqrt_conjugate_collapse_expr(&mut ctx, expr);
        assert!(matched.is_some());
    }

    #[test]
    fn rewrites_generalized_rationalization_three_term_denominator() {
        let mut ctx = Context::new();
        let expr = parse("1/(1+sqrt(2)+sqrt(3))", &mut ctx).expect("parse");
        let rewrite = try_rewrite_generalized_rationalization_expr(&mut ctx, expr)
            .expect("generalized rewrite should apply");
        let rendered = cas_formatter::render_expr(&ctx, rewrite.rewritten);
        assert!(rendered.contains("sqrt(2)") || rendered.contains("sqrt(3)"));
        assert!(rewrite.desc.contains("group 3 terms"));
    }

    #[test]
    fn rejects_generalized_rationalization_two_term_denominator() {
        let mut ctx = Context::new();
        let expr = parse("1/(sqrt(x)+1)", &mut ctx).expect("parse");
        assert!(try_rewrite_generalized_rationalization_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn rewrites_rationalize_product_denominator_variable_radicand() {
        let mut ctx = Context::new();
        let expr = parse("1/(x*sqrt(y))", &mut ctx).expect("parse");
        let rewrite = try_rewrite_rationalize_product_denominator_expr(&mut ctx, expr)
            .expect("product-den rationalization should apply");
        let rendered = cas_formatter::render_expr(&ctx, rewrite.rewritten);
        assert!(rendered.contains("sqrt(y)") || rendered.contains("y^(1/2)"));
    }

    #[test]
    fn rejects_rationalize_product_denominator_numeric_radicand() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(2)/(2*2^(1/3))", &mut ctx).expect("parse");
        assert!(try_rewrite_rationalize_product_denominator_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn rewrites_sqrt_conjugate_collapse_when_gate_allows() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x+sqrt(y))*(x-sqrt(y))", &mut ctx).expect("parse");
        let rewrite =
            try_rewrite_sqrt_conjugate_collapse_expr_with(&mut ctx, expr, |_ctx, _other| {
                SqrtConjugateCollapseGate {
                    allow: true,
                    assumed: true,
                }
            })
            .expect("rewrite");
        assert!(rewrite.assumed_nonnegative_target.is_some());
    }

    #[test]
    fn blocks_sqrt_conjugate_collapse_when_gate_blocks() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x+sqrt(y))*(x-sqrt(y))", &mut ctx).expect("parse");
        let rewrite =
            try_rewrite_sqrt_conjugate_collapse_expr_with(&mut ctx, expr, |_ctx, _other| {
                SqrtConjugateCollapseGate {
                    allow: false,
                    assumed: false,
                }
            });
        assert!(rewrite.is_none());
    }
}
