use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_nary::{build_balanced_mul, AddView, Sign};
use cas_math::expr_rewrite::smart_mul;
use cas_math::hyperbolic_core_support::{
    try_eval_hyperbolic_special_value, try_rewrite_hyperbolic_composition,
};
use cas_math::hyperbolic_identity_support::{
    try_rewrite_hyperbolic_double_angle_sub_chain, try_rewrite_hyperbolic_double_angle_sum,
    try_rewrite_hyperbolic_pythagorean_sub_expr, try_rewrite_hyperbolic_triple_angle,
    try_rewrite_sinh_cosh_to_exp, try_rewrite_sinh_cosh_to_tanh_identity_expr,
    try_rewrite_sinh_double_angle_expansion_identity_expr,
    try_rewrite_tanh_double_angle_expansion_identity_expr,
    try_rewrite_tanh_to_sinh_cosh_identity_expr,
};
use cas_math::hyperbolic_negative_support::try_rewrite_hyperbolic_negative_expr;
use num_rational::BigRational;
use num_traits::One;
use std::cmp::Ordering;

enum HyperbolicCubicZeroMatch {
    ProductTriple {
        scale: ExprId,
        arg: ExprId,
        target: ExprId,
    },
    ExpandedPythagorean {
        focus_before: ExprId,
        factorized: ExprId,
        rewritten: ExprId,
    },
}

fn format_hyperbolic_negative_desc(
    kind: cas_math::hyperbolic_negative_support::HyperbolicNegativeRewriteKind,
) -> &'static str {
    match kind {
        cas_math::hyperbolic_negative_support::HyperbolicNegativeRewriteKind::SinhExplicitNeg => {
            "sinh(-x) = -sinh(x)"
        }
        cas_math::hyperbolic_negative_support::HyperbolicNegativeRewriteKind::CoshExplicitNeg => {
            "cosh(-x) = cosh(x)"
        }
        cas_math::hyperbolic_negative_support::HyperbolicNegativeRewriteKind::TanhExplicitNeg => {
            "tanh(-x) = -tanh(x)"
        }
        cas_math::hyperbolic_negative_support::HyperbolicNegativeRewriteKind::AsinhExplicitNeg => {
            "asinh(-x) = -asinh(x)"
        }
        cas_math::hyperbolic_negative_support::HyperbolicNegativeRewriteKind::AtanhExplicitNeg => {
            "atanh(-x) = -atanh(x)"
        }
        cas_math::hyperbolic_negative_support::HyperbolicNegativeRewriteKind::SinhCanonicalSub => {
            "sinh(a−b) = −sinh(b−a)"
        }
        cas_math::hyperbolic_negative_support::HyperbolicNegativeRewriteKind::CoshCanonicalSub => {
            "cosh(a−b) = cosh(b−a)"
        }
        cas_math::hyperbolic_negative_support::HyperbolicNegativeRewriteKind::TanhCanonicalSub => {
            "tanh(a−b) = −tanh(b−a)"
        }
        cas_math::hyperbolic_negative_support::HyperbolicNegativeRewriteKind::AsinhCanonicalSub => {
            "asinh(a−b) = −asinh(b−a)"
        }
        cas_math::hyperbolic_negative_support::HyperbolicNegativeRewriteKind::AtanhCanonicalSub => {
            "atanh(a−b) = −atanh(b−a)"
        }
    }
}

fn format_hyperbolic_special_value_desc(ctx: &Context, expr: ExprId) -> Option<&'static str> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Sinh) => Some("sinh(0) = 0"),
        Some(BuiltinFn::Tanh) => Some("tanh(0) = 0"),
        Some(BuiltinFn::Cosh) => Some("cosh(0) = 1"),
        Some(BuiltinFn::Asinh) => Some("asinh(0) = 0"),
        Some(BuiltinFn::Atanh) => Some("atanh(0) = 0"),
        Some(BuiltinFn::Acosh) => Some("acosh(1) = 0"),
        _ => None,
    }
}

fn format_hyperbolic_composition_desc(ctx: &Context, expr: ExprId) -> Option<&'static str> {
    let Expr::Function(outer_fn, outer_args) = ctx.get(expr) else {
        return None;
    };
    if outer_args.len() != 1 {
        return None;
    }
    let Expr::Function(inner_fn, inner_args) = ctx.get(outer_args[0]) else {
        return None;
    };
    if inner_args.len() != 1 {
        return None;
    }
    match (ctx.builtin_of(*outer_fn), ctx.builtin_of(*inner_fn)) {
        (Some(BuiltinFn::Sinh), Some(BuiltinFn::Asinh)) => Some("sinh(asinh(x)) = x"),
        (Some(BuiltinFn::Cosh), Some(BuiltinFn::Acosh)) => Some("cosh(acosh(x)) = x"),
        (Some(BuiltinFn::Tanh), Some(BuiltinFn::Atanh)) => Some("tanh(atanh(x)) = x"),
        (Some(BuiltinFn::Asinh), Some(BuiltinFn::Sinh)) => Some("asinh(sinh(x)) = x"),
        (Some(BuiltinFn::Acosh), Some(BuiltinFn::Cosh)) => Some("acosh(cosh(x)) = x"),
        (Some(BuiltinFn::Atanh), Some(BuiltinFn::Tanh)) => Some("atanh(tanh(x)) = x"),
        _ => None,
    }
}

fn format_hyperbolic_identity_desc(
    kind: cas_math::hyperbolic_identity_support::HyperbolicIdentityRewriteKind,
) -> &'static str {
    match kind {
        cas_math::hyperbolic_identity_support::HyperbolicIdentityRewriteKind::PythagoreanOne => {
            "cosh²(x) - sinh²(x) = 1"
        }
        cas_math::hyperbolic_identity_support::HyperbolicIdentityRewriteKind::PythagoreanNegativeOne => {
            "sinh²(x) - cosh²(x) = -1"
        }
        cas_math::hyperbolic_identity_support::HyperbolicIdentityRewriteKind::SinhCoshToTanh => {
            "sinh(x)/cosh(x) = tanh(x)"
        }
        cas_math::hyperbolic_identity_support::HyperbolicIdentityRewriteKind::TanhToSinhCosh => {
            "tanh(x) = sinh(x)/cosh(x)"
        }
        cas_math::hyperbolic_identity_support::HyperbolicIdentityRewriteKind::SinhDoubleAngleExpansion => {
            "sinh(2x) = 2·sinh(x)·cosh(x)"
        }
        cas_math::hyperbolic_identity_support::HyperbolicIdentityRewriteKind::TanhDoubleAngleExpansion => {
            "tanh(2x) = 2·tanh(x)/(1+tanh²(x))"
        }
    }
}

fn format_sinh_cosh_to_exp_desc(
    kind: cas_math::hyperbolic_identity_support::SinhCoshToExpRewriteKind,
) -> &'static str {
    match kind {
        cas_math::hyperbolic_identity_support::SinhCoshToExpRewriteKind::Sum => {
            "sinh(x) + cosh(x) = e^x"
        }
        cas_math::hyperbolic_identity_support::SinhCoshToExpRewriteKind::CoshMinusSinh => {
            "cosh(x) - sinh(x) = e^(-x)"
        }
        cas_math::hyperbolic_identity_support::SinhCoshToExpRewriteKind::SinhMinusCosh => {
            "sinh(x) - cosh(x) = -e^(-x)"
        }
    }
}

fn format_hyperbolic_double_angle_desc(
    kind: cas_math::hyperbolic_identity_support::HyperbolicDoubleAngleRewriteKind,
) -> &'static str {
    match kind {
        cas_math::hyperbolic_identity_support::HyperbolicDoubleAngleRewriteKind::Sum => {
            "cosh²(x) + sinh²(x) = cosh(2x)"
        }
        cas_math::hyperbolic_identity_support::HyperbolicDoubleAngleRewriteKind::SubChain => {
            "cosh(2x) - cosh²(x) - sinh²(x) = 0"
        }
    }
}

fn format_hyperbolic_triple_angle_desc(
    kind: cas_math::hyperbolic_identity_support::HyperbolicTripleAngleRewriteKind,
) -> &'static str {
    match kind {
        cas_math::hyperbolic_identity_support::HyperbolicTripleAngleRewriteKind::Sinh => {
            "sinh(3x) → 3sinh(x) + 4sinh³(x)"
        }
        cas_math::hyperbolic_identity_support::HyperbolicTripleAngleRewriteKind::Cosh => {
            "cosh(3x) → 4cosh³(x) - 3cosh(x)"
        }
    }
}

fn format_recognize_hyperbolic_from_exp_desc(
    kind: cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind,
) -> &'static str {
    match kind {
        cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::CoshHalf => {
            "(e^x + e^(-x))/2 = cosh(x)"
        }
        cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::SinhHalf => {
            "(e^x - e^(-x))/2 = sinh(x)"
        }
        cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::NegSinhHalf => {
            "(e^(-x) - e^x)/2 = -sinh(x)"
        }
        cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::CoshDirect => {
            "e^x + e^(-x) = 2cosh(x)"
        }
        cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::SinhDirect => {
            "e^x - e^(-x) = 2sinh(x)"
        }
        cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::NegSinhDirect => {
            "e^(-x) - e^x = -2sinh(x)"
        }
        cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::TanhRatio => {
            "(e^x - e^(-x))/(e^x + e^(-x)) = tanh(x)"
        }
        cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::NegTanhRatio => {
            "(e^(-x) - e^x)/(e^x + e^(-x)) = -tanh(x)"
        }
    }
}

fn strip_hyperbolic_term_negation(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => Some(inner),
        Expr::Number(n) if n < BigRational::from_integer(0.into()) => {
            Some(ctx.add(Expr::Number(-n)))
        }
        _ => None,
    }
}

fn normalize_signed_hyperbolic_add_term(
    ctx: &mut Context,
    term_expr: ExprId,
    term_sign: Sign,
) -> (ExprId, Sign) {
    if let Some(positive_expr) = strip_hyperbolic_term_negation(ctx, term_expr) {
        return (positive_expr, term_sign.negate());
    }

    match ctx.get(term_expr).clone() {
        Expr::Mul(lhs, rhs) => {
            if let Some(positive_lhs) = strip_hyperbolic_term_negation(ctx, lhs) {
                return (
                    build_scaled_expr(ctx, positive_lhs, rhs),
                    term_sign.negate(),
                );
            }
            if let Some(positive_rhs) = strip_hyperbolic_term_negation(ctx, rhs) {
                return (
                    build_scaled_expr(ctx, positive_rhs, lhs),
                    term_sign.negate(),
                );
            }
            (term_expr, term_sign)
        }
        _ => (term_expr, term_sign),
    }
}

fn hyperbolic_exprs_match(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> bool {
    if compare_expr(ctx, lhs, rhs) == Ordering::Equal
        || cas_math::expr_domain::exprs_equivalent(ctx, lhs, rhs)
    {
        return true;
    }

    let lhs_normalized = cas_math::canonical_forms::normalize_core(ctx, lhs);
    let rhs_normalized = cas_math::canonical_forms::normalize_core(ctx, rhs);
    compare_expr(ctx, lhs_normalized, rhs_normalized) == Ordering::Equal
        || cas_math::expr_domain::exprs_equivalent(ctx, lhs_normalized, rhs_normalized)
}

fn extract_n_times_base(ctx: &Context, expr: ExprId, n: i64) -> Option<ExprId> {
    let Expr::Mul(l, r) = ctx.get(expr) else {
        return None;
    };
    let expected = BigRational::from_integer(n.into());
    if let Expr::Number(k) = ctx.get(*l) {
        if *k == expected {
            return Some(*r);
        }
    }
    if let Expr::Number(k) = ctx.get(*r) {
        if *k == expected {
            return Some(*l);
        }
    }
    None
}

fn extract_numeric_scale_factors(ctx: &Context, expr: ExprId) -> (BigRational, Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Neg(inner) => {
            let (coeff, factors) = extract_numeric_scale_factors(ctx, *inner);
            (-coeff, factors)
        }
        Expr::Mul(_, _) => {
            let mut coeff = BigRational::from_integer(1.into());
            let mut factors = Vec::new();
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Expr::Number(n) = ctx.get(factor) {
                    coeff *= n.clone();
                } else {
                    factors.push(factor);
                }
            }
            (coeff, factors)
        }
        Expr::Number(n) => (n.clone(), Vec::new()),
        _ => (BigRational::from_integer(1.into()), vec![expr]),
    }
}

fn extract_hyperbolic_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    (ctx.is_builtin(*fn_id, builtin) && args.len() == 1).then_some(args[0])
}

fn build_scaled_expr(ctx: &mut Context, scale: ExprId, expr: ExprId) -> ExprId {
    let one = ctx.num(1);
    if compare_expr(ctx, scale, one) == Ordering::Equal {
        expr
    } else {
        smart_mul(ctx, scale, expr)
    }
}

fn extract_scaled_double_sinh_product_identity(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let (numeric_coeff, factors) = extract_numeric_scale_factors(ctx, expr);
    let mut residual_factors = Vec::new();
    let mut sinh_args = Vec::new();

    for factor in factors {
        if let Some(arg) = extract_hyperbolic_arg(ctx, factor, BuiltinFn::Sinh) {
            sinh_args.push(arg);
        } else {
            residual_factors.push(factor);
        }
    }

    if sinh_args.len() != 2 {
        return None;
    }

    let base_arg = if let Some(base) = extract_n_times_base(ctx, sinh_args[0], 2) {
        if compare_expr(ctx, base, sinh_args[1]) == Ordering::Equal {
            sinh_args[1]
        } else {
            return None;
        }
    } else if let Some(base) = extract_n_times_base(ctx, sinh_args[1], 2) {
        if compare_expr(ctx, base, sinh_args[0]) == Ordering::Equal {
            sinh_args[0]
        } else {
            return None;
        }
    } else {
        return None;
    };

    let scale_numeric = numeric_coeff / BigRational::from_integer(2.into());
    let mut scale_factors = residual_factors;
    if scale_numeric != BigRational::one() || scale_factors.is_empty() {
        scale_factors.insert(0, ctx.add(Expr::Number(scale_numeric)));
    }

    let scale = if scale_factors.len() == 1 {
        scale_factors[0]
    } else {
        build_balanced_mul(ctx, &scale_factors)
    };

    Some((scale, base_arg))
}

fn build_hyperbolic_product_to_sum_intermediate(
    ctx: &mut Context,
    scale: ExprId,
    arg: ExprId,
) -> ExprId {
    let three = ctx.num(3);
    let triple_arg = smart_mul(ctx, three, arg);
    let cosh_triple = ctx.call_builtin(BuiltinFn::Cosh, vec![triple_arg]);
    let cosh_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
    let diff = ctx.add(Expr::Sub(cosh_triple, cosh_arg));
    build_scaled_expr(ctx, scale, diff)
}

fn build_hyperbolic_cubic_cosh_polynomial(ctx: &mut Context, scale: ExprId, arg: ExprId) -> ExprId {
    let cosh_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
    let three = ctx.num(3);
    let four = ctx.num(4);
    let cosh_cubed = ctx.add(Expr::Pow(cosh_arg, three));
    let cubic = smart_mul(ctx, four, cosh_cubed);
    let linear = smart_mul(ctx, four, cosh_arg);
    let diff = ctx.add(Expr::Sub(cubic, linear));
    build_scaled_expr(ctx, scale, diff)
}

fn build_hyperbolic_scaled_cosh_term(
    ctx: &mut Context,
    scale: ExprId,
    arg: ExprId,
    power: i64,
) -> ExprId {
    let cosh_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
    let four = ctx.num(4);
    let term = if power == 1 {
        smart_mul(ctx, four, cosh_arg)
    } else {
        let exponent = ctx.num(power);
        let pow = ctx.add(Expr::Pow(cosh_arg, exponent));
        smart_mul(ctx, four, pow)
    };
    build_scaled_expr(ctx, scale, term)
}

fn matches_hyperbolic_linear_minus_cubic_terms(
    ctx: &mut Context,
    terms: &[(ExprId, Sign)],
    scale: ExprId,
    arg: ExprId,
) -> bool {
    if terms.len() != 2 {
        return false;
    }

    let linear_target = build_hyperbolic_scaled_cosh_term(ctx, scale, arg, 1);
    let cubic_target = build_hyperbolic_scaled_cosh_term(ctx, scale, arg, 3);
    let mut saw_linear = false;
    let mut saw_cubic = false;

    for (term_expr, term_sign) in terms.iter().copied() {
        let (normalized_expr, normalized_sign) =
            normalize_signed_hyperbolic_add_term(ctx, term_expr, term_sign);
        if normalized_sign == Sign::Pos
            && !saw_linear
            && hyperbolic_exprs_match(ctx, normalized_expr, linear_target)
        {
            saw_linear = true;
            continue;
        }
        if normalized_sign == Sign::Neg
            && !saw_cubic
            && hyperbolic_exprs_match(ctx, normalized_expr, cubic_target)
        {
            saw_cubic = true;
            continue;
        }
        return false;
    }

    saw_linear && saw_cubic
}

fn matches_hyperbolic_cubic_minus_linear_terms(
    ctx: &mut Context,
    terms: &[(ExprId, Sign)],
    scale: ExprId,
    arg: ExprId,
) -> bool {
    if terms.len() != 2 {
        return false;
    }

    let linear_target = build_hyperbolic_scaled_cosh_term(ctx, scale, arg, 1);
    let cubic_target = build_hyperbolic_scaled_cosh_term(ctx, scale, arg, 3);
    let mut saw_linear = false;
    let mut saw_cubic = false;

    for (term_expr, term_sign) in terms.iter().copied() {
        let (normalized_expr, normalized_sign) =
            normalize_signed_hyperbolic_add_term(ctx, term_expr, term_sign);
        if normalized_sign == Sign::Pos
            && !saw_cubic
            && hyperbolic_exprs_match(ctx, normalized_expr, cubic_target)
        {
            saw_cubic = true;
            continue;
        }
        if normalized_sign == Sign::Neg
            && !saw_linear
            && hyperbolic_exprs_match(ctx, normalized_expr, linear_target)
        {
            saw_linear = true;
            continue;
        }
        return false;
    }

    saw_linear && saw_cubic
}

fn extract_scaled_expanded_hyperbolic_product_term(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let (numeric_coeff, factors) = extract_numeric_scale_factors(ctx, expr);
    let mut residual_factors = Vec::new();
    let mut cosh_arg = None;
    let mut sinh_arg = None;
    let mut sinh_count = 0;

    for factor in factors {
        if let Some(arg) = extract_hyperbolic_arg(ctx, factor, BuiltinFn::Cosh) {
            if cosh_arg.replace(arg).is_some() {
                return None;
            }
            continue;
        }

        if let Some(arg) = extract_hyperbolic_arg(ctx, factor, BuiltinFn::Sinh) {
            if let Some(existing) = sinh_arg {
                if compare_expr(ctx, existing, arg) != Ordering::Equal {
                    return None;
                }
            } else {
                sinh_arg = Some(arg);
            }
            sinh_count += 1;
            continue;
        }

        if let Expr::Pow(base, exponent) = ctx.get(factor) {
            let Expr::Number(n) = ctx.get(*exponent) else {
                residual_factors.push(factor);
                continue;
            };
            if *n == BigRational::from_integer(2.into()) {
                if let Some(arg) = extract_hyperbolic_arg(ctx, *base, BuiltinFn::Sinh) {
                    if let Some(existing) = sinh_arg {
                        if compare_expr(ctx, existing, arg) != Ordering::Equal {
                            return None;
                        }
                    } else {
                        sinh_arg = Some(arg);
                    }
                    sinh_count += 2;
                    continue;
                }
            }
        }

        residual_factors.push(factor);
    }

    let arg = cosh_arg?;
    if compare_expr(ctx, arg, sinh_arg?) != Ordering::Equal || sinh_count != 2 {
        return None;
    }

    let scale_numeric = numeric_coeff / BigRational::from_integer(4.into());
    let mut scale_factors = residual_factors;
    if scale_numeric != BigRational::one() || scale_factors.is_empty() {
        scale_factors.insert(0, ctx.add(Expr::Number(scale_numeric)));
    }

    let scale = if scale_factors.len() == 1 {
        scale_factors[0]
    } else {
        build_balanced_mul(ctx, &scale_factors)
    };

    Some((scale, arg))
}

fn build_hyperbolic_factored_pythagorean_intermediate(
    ctx: &mut Context,
    scale: ExprId,
    arg: ExprId,
) -> ExprId {
    let cosh_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
    let sinh_arg = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
    let two = ctx.num(2);
    let sinh_sq = ctx.add(Expr::Pow(sinh_arg, two));
    let one = ctx.num(1);
    let inner = ctx.add(Expr::Add(sinh_sq, one));
    let product = smart_mul(ctx, cosh_arg, inner);
    build_scaled_expr(ctx, scale, product)
}

fn match_hyperbolic_product_triple_identity_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HyperbolicCubicZeroMatch> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    for index in 0..view.terms.len() {
        let (term_expr, term_sign) =
            normalize_signed_hyperbolic_add_term(ctx, view.terms[index].0, view.terms[index].1);
        let Some((scale, arg)) = extract_scaled_double_sinh_product_identity(ctx, term_expr) else {
            continue;
        };
        let target_expr = build_hyperbolic_cubic_cosh_polynomial(ctx, scale, arg);
        let remaining_terms: Vec<_> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(other_index, term)| (other_index != index).then_some(term))
            .collect();
        if remaining_terms.is_empty() {
            continue;
        }

        let matches = match term_sign {
            Sign::Pos => {
                matches_hyperbolic_linear_minus_cubic_terms(ctx, &remaining_terms, scale, arg)
            }
            Sign::Neg => {
                matches_hyperbolic_cubic_minus_linear_terms(ctx, &remaining_terms, scale, arg)
            }
        };
        if matches {
            return Some(HyperbolicCubicZeroMatch::ProductTriple {
                scale,
                arg,
                target: target_expr,
            });
        }
    }

    if view.terms.len() != 3 {
        return None;
    }

    for index in 0..view.terms.len() {
        let (term_expr, term_sign) =
            normalize_signed_hyperbolic_add_term(ctx, view.terms[index].0, view.terms[index].1);
        if term_sign != Sign::Pos {
            continue;
        }
        let Some((scale, arg)) = extract_scaled_expanded_hyperbolic_product_term(ctx, term_expr)
        else {
            continue;
        };
        let remaining_terms: Vec<_> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(other_index, term)| (other_index != index).then_some(term))
            .collect();
        if !matches_hyperbolic_linear_minus_cubic_terms(ctx, &remaining_terms, scale, arg) {
            continue;
        }

        let linear_target = build_hyperbolic_scaled_cosh_term(ctx, scale, arg, 1);
        let focus_before = ctx.add(Expr::Add(term_expr, linear_target));
        let factorized = build_hyperbolic_factored_pythagorean_intermediate(ctx, scale, arg);
        let rewritten = build_hyperbolic_scaled_cosh_term(ctx, scale, arg, 3);
        return Some(HyperbolicCubicZeroMatch::ExpandedPythagorean {
            focus_before,
            factorized,
            rewritten,
        });
    }

    None
}

fn build_hyperbolic_product_triple_identity_zero_rewrite(
    ctx: &mut Context,
    whole_expr: ExprId,
    scale: ExprId,
    arg: ExprId,
    target: ExprId,
) -> Rewrite {
    let product_sum = build_hyperbolic_product_to_sum_intermediate(ctx, scale, arg);
    let product_sum_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: product_sum
        }
    );
    let target_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: target
        }
    );

    Rewrite::with_local(
        ctx.num(0),
        "Hyperbolic Product-to-Sum and Triple-Angle Identity",
        whole_expr,
        ctx.num(0),
    )
    .substep(
        "Usar 2·sinh(A)·sinh(B) = cosh(A+B) - cosh(A-B)",
        vec![format!("Se obtiene {product_sum_display}.")],
    )
    .substep(
        "Usar cosh(3u) = 4·cosh(u)^3 - 3·cosh(u)",
        vec![format!("La expresión se reescribe como {target_display}.")],
    )
    .substep(
        "Cancelar términos iguales",
        vec![
            "Tras reconocer la misma forma en el otro lado, toda la expresión se anula."
                .to_string(),
        ],
    )
}

fn build_hyperbolic_expanded_pythagorean_zero_rewrite(
    ctx: &mut Context,
    focus_before: ExprId,
    factorized: ExprId,
    rewritten: ExprId,
) -> Rewrite {
    let factorized_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: factorized
        }
    );
    let rewritten_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: rewritten
        }
    );

    Rewrite::with_local(
        ctx.num(0),
        "Hyperbolic Pythagorean Identity Cancellation Bridge",
        focus_before,
        rewritten,
    )
    .substep(
        "Sacar factor común",
        vec![format!("Sacar factor común para obtener {factorized_display}.")],
    )
    .substep(
        "Usar sinh(u)^2 + 1 = cosh(u)^2",
        vec![format!("Así se obtiene {rewritten_display}.")],
    )
    .substep(
        "Cancelar términos iguales",
        vec![
            "Tras la reescritura, el término restante es exactamente el opuesto y toda la expresión se anula."
                .to_string(),
        ],
    )
}

// ==================== Hyperbolic Function Rules ====================

// Rule 1: Evaluate hyperbolic functions at special values
define_rule!(
    EvaluateHyperbolicRule,
    "Evaluate Hyperbolic Functions",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_eval_hyperbolic_special_value(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_hyperbolic_special_value_desc(ctx, expr)?))
    }
);

// Rule 2: Composition identities - sinh(asinh(x)) = x, etc.
// HIGH PRIORITY: Must run BEFORE TanhToSinhCoshRule - ensured by registration order
define_rule!(
    HyperbolicCompositionRule,
    "Hyperbolic Composition",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Analytic
    ),
    |ctx, expr, _parent_ctx| {
        let rewrite = try_rewrite_hyperbolic_composition(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_hyperbolic_composition_desc(ctx, expr)?))
    }
);

// Rule 3: Negative argument identities
// Handles both explicit Neg(x) and Sub(a,b) where a < b canonically.
// V2.16: Extended to catch Sub patterns like sinh(1-u²) → -sinh(u²-1).
define_rule!(
    HyperbolicNegativeRule,
    "Hyperbolic Negative Argument",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_hyperbolic_negative_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_hyperbolic_negative_desc(rewrite.kind)))
    }
);

// Rule 4: Hyperbolic Pythagorean identity: cosh²(x) - sinh²(x) = 1
define_rule!(
    HyperbolicPythagoreanRule,
    "Hyperbolic Pythagorean Identity",
    Some(crate::target_kind::TargetKindSet::SUB),
    |ctx, expr| {
        let rewrite = try_rewrite_hyperbolic_pythagorean_sub_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_hyperbolic_identity_desc(rewrite.kind)))
    }
);

// Rule 4b: sinh(x) + cosh(x) = exp(x), cosh(x) - sinh(x) = exp(-x)
// Inverse of RecognizeHyperbolicFromExpRule — collapses hyperbolic sums/diffs to exp.
define_rule!(
    SinhCoshToExpRule,
    "Hyperbolic Sum to Exponential",
    Some(crate::target_kind::TargetKindSet::ADD.union(crate::target_kind::TargetKindSet::SUB)),
    |ctx, expr| {
        let rewrite = try_rewrite_sinh_cosh_to_exp(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_sinh_cosh_to_exp_desc(rewrite.kind)))
    }
);

// Rule 4c: exact cancellation bridge for
// 2*sinh(2x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)) -> 0
// and scaled/factored variants of the same family.
pub struct HyperbolicProductTripleAngleIdentityZeroRule;

impl crate::rule::Rule for HyperbolicProductTripleAngleIdentityZeroRule {
    fn name(&self) -> &str {
        "Hyperbolic Pythagorean Identity Cancellation Bridge Residual"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        match match_hyperbolic_product_triple_identity_zero(ctx, expr)? {
            HyperbolicCubicZeroMatch::ProductTriple { scale, arg, target } => {
                Some(build_hyperbolic_product_triple_identity_zero_rewrite(
                    ctx, expr, scale, arg, target,
                ))
            }
            HyperbolicCubicZeroMatch::ExpandedPythagorean {
                focus_before,
                factorized,
                rewritten,
            } => Some(build_hyperbolic_expanded_pythagorean_zero_rewrite(
                ctx,
                focus_before,
                factorized,
                rewritten,
            )),
        }
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::ADD_SUB)
    }

    fn priority(&self) -> i32 {
        200
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

// Rule 5: Hyperbolic double angle identity: cosh²(x) + sinh²(x) = cosh(2x)
// This direction collapses two squared terms into a single term, reducing complexity.
// The inverse (expansion) is not implemented to avoid loops.
define_rule!(
    HyperbolicDoubleAngleRule,
    "Hyperbolic Double Angle",
    Some(crate::target_kind::TargetKindSet::ADD),
    |ctx, expr| {
        let rewrite = try_rewrite_hyperbolic_double_angle_sum(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten).desc(format_hyperbolic_double_angle_desc(rewrite.kind)),
        )
    }
);

// Rule: tanh(x) → sinh(x) / cosh(x)
// This is the hyperbolic analogue of tan(x) → sin(x) / cos(x)
// GUARD: Skip if argument is inverse hyperbolic (let composition rule handle tanh(atanh(x)) → x)
define_rule!(
    TanhToSinhCoshRule,
    "tanh(x) = sinh(x)/cosh(x)",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_tanh_to_sinh_cosh_identity_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_hyperbolic_identity_desc(rewrite.kind)))
    }
);

// Rule: sinh(2x) → 2·sinh(x)·cosh(x)
// Expansion of double angle for sinh
define_rule!(
    SinhDoubleAngleExpansionRule,
    "sinh(2x) = 2·sinh(x)·cosh(x)",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_sinh_double_angle_expansion_identity_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_hyperbolic_identity_desc(rewrite.kind)))
    }
);

// Rule: tanh(2x) → 2·tanh(x)/(1+tanh(x)^2)
define_rule!(
    TanhDoubleAngleExpansionRule,
    "tanh(2x) = 2·tanh(x)/(1+tanh(x)^2)",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_tanh_double_angle_expansion_identity_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_hyperbolic_identity_desc(rewrite.kind)))
    }
);

// Rule: cosh(2x) - cosh²(x) - sinh²(x) → 0
// After canonicalization, Sub nodes become Add+Neg, so the actual pattern is:
//   Add-chain containing: cosh(2x), Neg(cosh²(x)), Neg(sinh²(x))
// When found, these three terms cancel to 0 and are removed from the sum.
define_rule!(
    HyperbolicDoubleAngleSubRule,
    "Hyperbolic Double Angle Subtraction",
    Some(crate::target_kind::TargetKindSet::ADD),
    |ctx, expr| {
        let rewrite = try_rewrite_hyperbolic_double_angle_sub_chain(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten).desc(format_hyperbolic_double_angle_desc(rewrite.kind)),
        )
    }
);

// Rule: sinh(x) / cosh(x) → tanh(x)
// Contraction rule (inverse of TanhToSinhCoshRule) - safe direction that doesn't break composition tests
define_rule!(
    SinhCoshToTanhRule,
    "sinh(x)/cosh(x) = tanh(x)",
    Some(crate::target_kind::TargetKindSet::DIV),
    |ctx, expr| {
        let rewrite = try_rewrite_sinh_cosh_to_tanh_identity_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_hyperbolic_identity_desc(rewrite.kind)))
    }
);

// ==================== Recognize Hyperbolic From Exponential ====================

// Rule 5: Recognize hyperbolic functions from exponential definitions
// (e^x + e^(-x))/2 → cosh(x)
// (e^x - e^(-x))/2 → sinh(x)
// (e^(-x) - e^x)/2 → -sinh(x)
define_rule!(
    RecognizeHyperbolicFromExpRule,
    "Recognize Hyperbolic from Exponential",
    importance: crate::step::ImportanceLevel::Medium,
    |ctx, expr| {
        let rewrite = cas_math::hyperbolic_identity_support::try_rewrite_recognize_hyperbolic_from_exp(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten)
                .desc(format_recognize_hyperbolic_from_exp_desc(rewrite.kind))
        )
    }
);

// Rule: Hyperbolic triple angle expansion
// sinh(3x) → 3·sinh(x) + 4·sinh³(x)
// cosh(3x) → 4·cosh³(x) - 3·cosh(x)
// Note: sinh uses + (not -) because sinh is always positive for positive x
define_rule!(
    HyperbolicTripleAngleRule,
    "Hyperbolic Triple Angle Identity",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_hyperbolic_triple_angle(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten).desc(format_hyperbolic_triple_angle_desc(rewrite.kind)),
        )
    }
);

/// Register all hyperbolic function rules
pub fn register(simplifier: &mut crate::engine::Simplifier) {
    simplifier.add_rule(Box::new(EvaluateHyperbolicRule));
    simplifier.add_rule(Box::new(HyperbolicCompositionRule));
    simplifier.add_rule(Box::new(HyperbolicNegativeRule));
    simplifier.add_rule(Box::new(HyperbolicPythagoreanRule));
    simplifier.add_rule(Box::new(SinhCoshToExpRule));
    simplifier.add_rule(Box::new(HyperbolicProductTripleAngleIdentityZeroRule));
    simplifier.add_rule(Box::new(HyperbolicDoubleAngleRule));
    simplifier.add_rule(Box::new(HyperbolicDoubleAngleSubRule));
    // DISABLED: TanhToSinhCoshRule breaks tanh(atanh(x))→x and tanh(-x)→-tanh(x) paths
    // simplifier.add_rule(Box::new(TanhToSinhCoshRule)); // tanh(x) → sinh(x)/cosh(x)
    simplifier.add_rule(Box::new(SinhCoshToTanhRule)); // sinh(x)/cosh(x) → tanh(x) (contraction)
    simplifier.add_rule(Box::new(SinhDoubleAngleExpansionRule)); // sinh(2x) → 2sinh(x)cosh(x)
    simplifier.add_rule(Box::new(TanhDoubleAngleExpansionRule)); // tanh(2x) → 2tanh(x)/(1+tanh(x)^2)
    simplifier.add_rule(Box::new(RecognizeHyperbolicFromExpRule));
    simplifier.add_rule(Box::new(HyperbolicTripleAngleRule)); // sinh(3x), cosh(3x)
}

#[cfg(test)]
mod tests {
    use super::{match_hyperbolic_product_triple_identity_zero, HyperbolicCubicZeroMatch};
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn matches_hyperbolic_product_triple_identity_zero_expr() {
        let mut ctx = Context::new();
        let expr = parse("2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))", &mut ctx)
            .unwrap_or_else(|err| panic!("parse: {err}"));

        let matched = match_hyperbolic_product_triple_identity_zero(&mut ctx, expr);
        assert!(matches!(
            matched,
            Some(HyperbolicCubicZeroMatch::ProductTriple { .. })
        ));
    }

    #[test]
    fn matches_hyperbolic_expanded_pythagorean_zero_expr() {
        let mut ctx = Context::new();
        let expr = parse("4*cosh(x)*sinh(x)^2 + 4*cosh(x) - 4*cosh(x)^3", &mut ctx)
            .unwrap_or_else(|err| panic!("parse: {err}"));

        let matched = match_hyperbolic_product_triple_identity_zero(&mut ctx, expr);
        assert!(matches!(
            matched,
            Some(HyperbolicCubicZeroMatch::ExpandedPythagorean { .. })
        ));
    }
}
