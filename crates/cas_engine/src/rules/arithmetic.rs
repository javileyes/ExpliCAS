use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Expr};
use cas_math::arithmetic_cancel_support::{
    try_rewrite_add_inverse_zero_expr, try_rewrite_sub_self_zero_expr,
};
use cas_math::arithmetic_rule_support::{
    try_rewrite_add_zero_expr, try_rewrite_combine_constants_expr, try_rewrite_mul_one_expr,
    try_rewrite_normalize_mul_neg_expr, try_rewrite_simplify_numeric_exponents_expr,
};
use cas_math::arithmetic_zero_support::{match_div_zero_numerator_pattern, match_mul_zero_pattern};
use cas_math::expr_destructure::{as_div, as_mul, as_pow};
use cas_math::expr_extract::extract_i64_integer;
use cas_math::expr_nary::{build_balanced_add, build_balanced_mul, AddView, Sign};
use cas_math::expr_predicates::is_minus_one_expr;
use cas_math::expr_rewrite::smart_mul;
use cas_math::logarithm_inverse_support::{make_log_expr, try_extract_log_parts};
use cas_math::trig_roots_flatten::flatten_mul_chain;
use cas_math::trig_sum_product_support::{
    build_avg_with_simplifier, build_half_diff_with_simplifier, extract_trig_two_term_diff,
    extract_trig_two_term_sum, normalize_for_even_fn, try_rewrite_sum_to_product_contraction_expr,
    TrigSumToProductContractionRewriteKind,
};
use num_rational::BigRational;
use num_traits::Signed;
use num_traits::{One, Zero};
use std::cmp::Ordering;

fn canonicalize_nested_integer_powers(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let rebuilt = match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => {
            let lhs = canonicalize_nested_integer_powers(ctx, lhs);
            let rhs = canonicalize_nested_integer_powers(ctx, rhs);
            ctx.add(Expr::Add(lhs, rhs))
        }
        Expr::Sub(lhs, rhs) => {
            let lhs = canonicalize_nested_integer_powers(ctx, lhs);
            let rhs = canonicalize_nested_integer_powers(ctx, rhs);
            ctx.add(Expr::Sub(lhs, rhs))
        }
        Expr::Mul(lhs, rhs) => {
            let lhs = canonicalize_nested_integer_powers(ctx, lhs);
            let rhs = canonicalize_nested_integer_powers(ctx, rhs);
            ctx.add(Expr::Mul(lhs, rhs))
        }
        Expr::Div(lhs, rhs) => {
            let lhs = canonicalize_nested_integer_powers(ctx, lhs);
            let rhs = canonicalize_nested_integer_powers(ctx, rhs);
            ctx.add(Expr::Div(lhs, rhs))
        }
        Expr::Pow(base, exp) => {
            let base = canonicalize_nested_integer_powers(ctx, base);
            let exp = canonicalize_nested_integer_powers(ctx, exp);
            let pow = ctx.add(Expr::Pow(base, exp));
            cas_math::rational_canonicalization_support::try_rewrite_nested_pow_canonical_expr(
                ctx, pow,
            )
            .map(|rewrite| rewrite.rewritten)
            .unwrap_or(pow)
        }
        Expr::Neg(inner) => {
            let inner = canonicalize_nested_integer_powers(ctx, inner);
            ctx.add(Expr::Neg(inner))
        }
        Expr::Function(name, args) => {
            let args = args
                .into_iter()
                .map(|arg| canonicalize_nested_integer_powers(ctx, arg))
                .collect();
            ctx.add(Expr::Function(name, args))
        }
        Expr::Matrix { rows, cols, data } => {
            let data = data
                .into_iter()
                .map(|arg| canonicalize_nested_integer_powers(ctx, arg))
                .collect();
            ctx.add(Expr::Matrix { rows, cols, data })
        }
        Expr::Hold(inner) => {
            let inner = canonicalize_nested_integer_powers(ctx, inner);
            ctx.add(Expr::Hold(inner))
        }
        Expr::SessionRef(id) => ctx.add(Expr::SessionRef(id)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) => expr,
    };

    if rebuilt == expr {
        expr
    } else {
        rebuilt
    }
}

fn collect_add_terms(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
    out: &mut Vec<cas_ast::ExprId>,
) {
    match ctx.get(expr) {
        Expr::Add(lhs, rhs) => {
            collect_add_terms(ctx, *lhs, out);
            collect_add_terms(ctx, *rhs, out);
        }
        _ => out.push(expr),
    }
}

fn expr_contains_any_builtin(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
    builtins: &[BuiltinFn],
) -> bool {
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Function(fn_id, args) => {
                if builtins
                    .iter()
                    .any(|builtin| ctx.is_builtin(*fn_id, *builtin))
                {
                    return true;
                }
                stack.extend(args.iter().copied());
            }
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Div(lhs, rhs)
            | Expr::Pow(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

fn maybe_trig_sum_to_product_zero_candidate(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    expr_contains_any_builtin(ctx, expr, &[BuiltinFn::Sin, BuiltinFn::Cos])
}

fn maybe_hyperbolic_angle_sum_diff_zero_candidate(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
) -> bool {
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Function(fn_id, args)
                if (ctx.is_builtin(*fn_id, BuiltinFn::Sinh)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Cosh))
                    && args.len() == 1 =>
            {
                if matches!(ctx.get(args[0]), Expr::Add(_, _) | Expr::Sub(_, _)) {
                    return true;
                }
                stack.push(args[0]);
            }
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Div(lhs, rhs)
            | Expr::Pow(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

fn extract_cosh_power_shape(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, i64)> {
    let positive_expr = match ctx.get(expr) {
        Expr::Neg(inner) => (-1, *inner),
        _ => (1, expr),
    }
    .1;

    let mut cosh_arg = None;
    let mut cosh_power = None;

    for factor in cas_math::expr_nary::mul_leaves(ctx, positive_expr) {
        match ctx.get(factor) {
            Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, BuiltinFn::Cosh) && args.len() == 1 =>
            {
                if cosh_arg.is_some() {
                    return None;
                }
                cosh_arg = Some(args[0]);
                cosh_power = Some(1);
            }
            Expr::Pow(base, exponent) => {
                let Expr::Function(fn_id, args) = ctx.get(*base) else {
                    continue;
                };
                if !ctx.is_builtin(*fn_id, BuiltinFn::Cosh) || args.len() != 1 {
                    continue;
                }
                if cosh_arg.is_some() {
                    return None;
                }
                cosh_arg = Some(args[0]);
                cosh_power = Some(extract_i64_integer(ctx, *exponent)?);
            }
            _ => {}
        }
    }

    Some((cosh_arg?, cosh_power?))
}

fn extract_signed_cosh_power(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(i64, cas_ast::ExprId, cas_ast::ExprId, i64)> {
    let (sign, positive_expr) = match ctx.get(expr) {
        Expr::Neg(inner) => (-1, *inner),
        _ => (1, expr),
    };

    let mut coefficient_factors = smallvec::SmallVec::<[cas_ast::ExprId; 8]>::new();
    let mut cosh_arg = None;
    let mut cosh_power = None;

    for factor in cas_math::expr_nary::mul_leaves(ctx, positive_expr) {
        match ctx.get(factor) {
            Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, BuiltinFn::Cosh) && args.len() == 1 =>
            {
                if cosh_arg.is_some() {
                    return None;
                }
                cosh_arg = Some(args[0]);
                cosh_power = Some(1);
            }
            Expr::Pow(base, exponent) => {
                let Expr::Function(fn_id, args) = ctx.get(*base) else {
                    coefficient_factors.push(factor);
                    continue;
                };
                if !ctx.is_builtin(*fn_id, BuiltinFn::Cosh) || args.len() != 1 {
                    coefficient_factors.push(factor);
                    continue;
                }
                if cosh_arg.is_some() {
                    return None;
                }
                cosh_arg = Some(args[0]);
                cosh_power = Some(extract_i64_integer(ctx, *exponent)?);
            }
            _ => coefficient_factors.push(factor),
        }
    }

    let coefficient = if coefficient_factors.is_empty() {
        ctx.num(1)
    } else {
        let coefficient_vec = coefficient_factors.into_vec();
        build_balanced_mul(ctx, &coefficient_vec)
    };

    Some((sign, coefficient, cosh_arg?, cosh_power?))
}

fn apply_sign_to_expr(
    ctx: &mut cas_ast::Context,
    sign: i64,
    expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    if sign < 0 {
        ctx.add(Expr::Neg(expr))
    } else {
        expr
    }
}

fn maybe_hyperbolic_pythagorean_factor_zero_candidate(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    let mut seen_linear = Vec::new();
    let mut seen_cubic = Vec::new();

    for (term_expr, _sign) in view.terms {
        if let Some((arg, power)) = extract_cosh_power_shape(ctx, term_expr) {
            match power {
                1 => seen_linear.push(arg),
                3 => seen_cubic.push(arg),
                _ => {}
            }
        }
    }

    seen_linear.iter().any(|linear_arg| {
        seen_cubic
            .iter()
            .any(|cubic_arg| compare_expr(ctx, *linear_arg, *cubic_arg) == Ordering::Equal)
    })
}

fn expr_contains_sqrt_or_half_power(ctx: &cas_ast::Context, root: cas_ast::ExprId) -> bool {
    let mut stack = vec![root];
    let half = num_rational::BigRational::new(1.into(), 2.into());

    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 =>
            {
                return true;
            }
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Pow(base, exp) => {
                if matches!(ctx.get(*exp), Expr::Number(n) if *n == half) {
                    return true;
                }
                stack.push(*base);
                stack.push(*exp);
            }
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Div(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }

    false
}

fn maybe_odd_half_power_zero_candidate(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Sub(lhs, rhs) => {
            expr_contains_sqrt_or_half_power(ctx, *lhs)
                || expr_contains_sqrt_or_half_power(ctx, *rhs)
        }
        Expr::Add(lhs, rhs) => match ctx.get(*rhs) {
            Expr::Neg(inner) => {
                expr_contains_sqrt_or_half_power(ctx, *lhs)
                    || expr_contains_sqrt_or_half_power(ctx, *inner)
            }
            _ => false,
        },
        _ => false,
    }
}

fn exprs_equal_up_to_add_term_order(
    ctx: &cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    let mut lhs_terms = Vec::new();
    let mut rhs_terms = Vec::new();
    collect_add_terms(ctx, lhs, &mut lhs_terms);
    collect_add_terms(ctx, rhs, &mut rhs_terms);
    if lhs_terms.len() != rhs_terms.len() {
        return false;
    }

    let mut lhs_terms: Vec<_> = lhs_terms
        .into_iter()
        .map(|term| {
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: term
                }
            )
        })
        .collect();
    let mut rhs_terms: Vec<_> = rhs_terms
        .into_iter()
        .map(|term| {
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: term
                }
            )
        })
        .collect();

    lhs_terms.sort();
    rhs_terms.sort();
    lhs_terms == rhs_terms
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OddHalfPowerProductForm {
    base: cas_ast::ExprId,
    outside_power: i64,
}

fn try_rewrite_odd_half_power_target_aware(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if let Some(rewrite) = cas_math::root_forms::try_rewrite_odd_half_power_expr(ctx, expr) {
        return Some(rewrite.rewritten);
    }

    let normalized = cas_math::canonical_forms::normalize_core(ctx, expr);
    if normalized == expr {
        return None;
    }

    cas_math::root_forms::try_rewrite_odd_half_power_expr(ctx, normalized)
        .map(|rewrite| rewrite.rewritten)
}

fn run_default_simplify(ctx: &mut cas_ast::Context, expr: cas_ast::ExprId) -> cas_ast::ExprId {
    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let (rewritten, _steps, _stats) = simplifier.simplify_with_stats(
        expr,
        crate::SimplifyOptions {
            suppress_depth_overflow_warnings: true,
            ..crate::SimplifyOptions::default()
        },
    );
    std::mem::swap(&mut simplifier.context, ctx);
    rewritten
}

fn try_rewrite_odd_half_power_with_optional_simplify(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if let Some(rewritten) = try_rewrite_odd_half_power_target_aware(ctx, expr) {
        return Some(rewritten);
    }

    let simplified = run_default_simplify(ctx, expr);
    if simplified == expr {
        return None;
    }

    try_rewrite_odd_half_power_target_aware(ctx, simplified)
}

fn extract_sqrt_argument(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> Option<cas_ast::ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            let half = num_rational::BigRational::new(1.into(), 2.into());
            match ctx.get(*exp) {
                Expr::Number(n) if *n == half => Some(*base),
                _ => None,
            }
        }
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

fn abs_argument(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> Option<cas_ast::ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Abs) && args.len() == 1 =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

fn small_positive_integer_value(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> Option<i64> {
    match ctx.get(expr) {
        Expr::Number(n)
            if n.is_integer() && *n > num_rational::BigRational::from_integer(0.into()) =>
        {
            n.to_integer().try_into().ok()
        }
        _ => None,
    }
}

fn extract_odd_half_power_outer_factor(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, i64)> {
    if let Some(inner) = abs_argument(ctx, expr) {
        return Some((inner, 1));
    }

    match ctx.get(expr) {
        Expr::Pow(base, exponent) => {
            let power = small_positive_integer_value(ctx, *exponent)?;
            if let Some(inner) = abs_argument(ctx, *base) {
                Some((inner, power))
            } else {
                Some((*base, power))
            }
        }
        _ => Some((expr, 1)),
    }
}

fn extract_odd_half_power_product_form(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<OddHalfPowerProductForm> {
    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    for (sqrt_index, sqrt_factor) in factors.iter().copied().enumerate() {
        let Some(base) = extract_sqrt_argument(ctx, sqrt_factor) else {
            continue;
        };
        let outer_factor = factors[1 - sqrt_index];
        let Some((outer_base, outside_power)) =
            extract_odd_half_power_outer_factor(ctx, outer_factor)
        else {
            continue;
        };
        if compare_expr(ctx, outer_base, base) == Ordering::Equal {
            return Some(OddHalfPowerProductForm {
                base,
                outside_power,
            });
        }
    }

    None
}

fn odd_half_power_domain_equivalent_target_match(
    ctx: &cas_ast::Context,
    rewritten: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let rewritten_form = extract_odd_half_power_product_form(ctx, rewritten)?;
    let target_form = extract_odd_half_power_product_form(ctx, target_expr)?;
    (rewritten_form.outside_power == target_form.outside_power
        && compare_expr(ctx, rewritten_form.base, target_form.base) == Ordering::Equal)
        .then_some(rewritten_form.base)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OddHalfPowerCancellationMatch {
    focus_before: cas_ast::ExprId,
    focus_after: cas_ast::ExprId,
    rewritten_expr: cas_ast::ExprId,
    base: Option<cas_ast::ExprId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LogAbsMulDivCancellationMatch {
    focus_after: cas_ast::ExprId,
    components: [(cas_ast::ExprId, Sign); 2],
}

#[derive(Debug, Clone)]
struct LogPowerProductCancellationMatch {
    raw_focus_after: cas_ast::ExprId,
    focus_after: cas_ast::ExprId,
    components: Vec<(cas_ast::ExprId, Sign)>,
    needs_power_split: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HyperbolicPythagoreanFactorCancellationMatch {
    local_before: cas_ast::ExprId,
    local_after: cas_ast::ExprId,
    mode: HyperbolicPythagoreanFactorCancellationMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HyperbolicPythagoreanFactorCancellationMode {
    FactorThenRewrite {
        factorized: cas_ast::ExprId,
        rewritten: cas_ast::ExprId,
    },
    AlreadyFactored,
}

fn try_match_odd_half_power_cancellation_side(
    ctx: &mut cas_ast::Context,
    focus_expr: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
) -> Option<OddHalfPowerCancellationMatch> {
    let rewritten = try_rewrite_odd_half_power_with_optional_simplify(ctx, focus_expr)?;
    if compare_expr(ctx, rewritten, target_expr) == Ordering::Equal {
        return Some(OddHalfPowerCancellationMatch {
            focus_before: focus_expr,
            focus_after: target_expr,
            rewritten_expr: target_expr,
            base: None,
        });
    }

    let base = odd_half_power_domain_equivalent_target_match(ctx, rewritten, target_expr)?;
    Some(OddHalfPowerCancellationMatch {
        focus_before: focus_expr,
        focus_after: target_expr,
        rewritten_expr: target_expr,
        base: Some(base),
    })
}

fn build_scaled_expr(
    ctx: &mut cas_ast::Context,
    scale: cas_ast::ExprId,
    expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let one = ctx.num(1);
    if compare_expr(ctx, scale, one) == Ordering::Equal {
        expr
    } else {
        ctx.add(Expr::Mul(scale, expr))
    }
}

fn strip_term_negation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => Some(inner),
        Expr::Number(n) if n < num_rational::BigRational::from_integer(0.into()) => {
            Some(ctx.add(Expr::Number(-n)))
        }
        _ => None,
    }
}

fn normalize_signed_add_term(
    ctx: &mut cas_ast::Context,
    term_expr: cas_ast::ExprId,
    term_sign: Sign,
) -> (cas_ast::ExprId, Sign) {
    if let Some(positive_expr) = strip_term_negation(ctx, term_expr) {
        return (positive_expr, term_sign.negate());
    }

    match ctx.get(term_expr).clone() {
        Expr::Mul(lhs, rhs) => {
            if let Some(positive_lhs) = strip_term_negation(ctx, lhs) {
                return (
                    build_scaled_expr(ctx, positive_lhs, rhs),
                    term_sign.negate(),
                );
            }
            if let Some(positive_rhs) = strip_term_negation(ctx, rhs) {
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

fn extract_scaled_log_abs_mul_div(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, cas_ast::ExprId)> {
    let extract_direct = |ctx: &mut cas_ast::Context,
                          expr: cas_ast::ExprId|
     -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
        let (base, arg) = try_extract_log_parts(ctx, expr)?;
        let inner = abs_argument(ctx, arg)?;
        Some((base, inner))
    };

    if let Some((base, inner)) = extract_direct(ctx, expr) {
        let one = ctx.num(1);
        return Some((one, base, inner));
    }

    match ctx.get(expr).clone() {
        Expr::Mul(lhs, rhs) => {
            if let Some((base, inner)) = extract_direct(ctx, lhs) {
                Some((rhs, base, inner))
            } else if let Some((base, inner)) = extract_direct(ctx, rhs) {
                Some((lhs, base, inner))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn try_match_log_abs_mul_div_cancellation_side(
    ctx: &mut cas_ast::Context,
    focus_expr: cas_ast::ExprId,
) -> Option<LogAbsMulDivCancellationMatch> {
    let (scale, log_base, inner) = extract_scaled_log_abs_mul_div(ctx, focus_expr)?;
    if let Some((lhs, rhs)) = as_mul(ctx, inner) {
        let lhs_abs = ctx.call_builtin(BuiltinFn::Abs, vec![lhs]);
        let rhs_abs = ctx.call_builtin(BuiltinFn::Abs, vec![rhs]);
        let lhs_log = make_log_expr(ctx, log_base, lhs_abs);
        let rhs_log = make_log_expr(ctx, log_base, rhs_abs);
        let expanded = ctx.add(Expr::Add(lhs_log, rhs_log));
        let focus_after = build_scaled_expr(ctx, scale, expanded);
        return Some(LogAbsMulDivCancellationMatch {
            focus_after,
            components: [
                (build_scaled_expr(ctx, scale, lhs_log), Sign::Pos),
                (build_scaled_expr(ctx, scale, rhs_log), Sign::Pos),
            ],
        });
    }

    if let Some((num, den)) = as_div(ctx, inner) {
        let num_abs = ctx.call_builtin(BuiltinFn::Abs, vec![num]);
        let den_abs = ctx.call_builtin(BuiltinFn::Abs, vec![den]);
        let num_log = make_log_expr(ctx, log_base, num_abs);
        let den_log = make_log_expr(ctx, log_base, den_abs);
        let expanded = ctx.add(Expr::Sub(num_log, den_log));
        let focus_after = build_scaled_expr(ctx, scale, expanded);
        return Some(LogAbsMulDivCancellationMatch {
            focus_after,
            components: [
                (build_scaled_expr(ctx, scale, num_log), Sign::Pos),
                (build_scaled_expr(ctx, scale, den_log), Sign::Neg),
            ],
        });
    }

    None
}

fn normalize_log_argument_for_cancellation(
    ctx: &mut cas_ast::Context,
    arg: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, bool)> {
    if let Some((pow_base, pow_exp)) = as_pow(ctx, arg) {
        let power = small_positive_integer_value(ctx, pow_exp)?;
        let normalized_arg = if power % 2 == 0 {
            ctx.call_builtin(BuiltinFn::Abs, vec![pow_base])
        } else {
            pow_base
        };
        let one = ctx.num(1);
        return Some((
            normalized_arg,
            pow_exp,
            compare_expr(ctx, pow_exp, one) != Ordering::Equal,
        ));
    }

    let one = ctx.num(1);
    Some((arg, one, false))
}

fn extract_scaled_log_term_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, cas_ast::ExprId, bool)> {
    let mut log_factor = None;
    let mut scale_factors = Vec::new();
    for factor in flatten_mul_chain(ctx, expr) {
        if try_extract_log_parts(ctx, factor).is_some() {
            if log_factor.is_some() {
                return None;
            }
            log_factor = Some(factor);
        } else {
            scale_factors.push(factor);
        }
    }

    let log_factor = log_factor?;
    let (log_base, log_arg) = try_extract_log_parts(ctx, log_factor)?;
    let (normalized_arg, power_scale, changed_by_power) =
        normalize_log_argument_for_cancellation(ctx, log_arg)?;

    let scale = match scale_factors.len() {
        0 => ctx.num(1),
        1 => scale_factors[0],
        _ => build_balanced_mul(ctx, &scale_factors),
    };
    let one = ctx.num(1);
    let total_scale = if compare_expr(ctx, power_scale, one) == Ordering::Equal {
        scale
    } else if compare_expr(ctx, scale, one) == Ordering::Equal {
        power_scale
    } else {
        smart_mul(ctx, scale, power_scale)
    };

    Some((total_scale, log_base, normalized_arg, changed_by_power))
}

fn normalize_log_term_expr_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let (scale, log_base, arg, _) = extract_scaled_log_term_for_cancellation(ctx, expr)?;
    let log_expr = make_log_expr(ctx, log_base, arg);
    Some(build_scaled_expr(ctx, scale, log_expr))
}

fn build_signed_add_expr(
    ctx: &mut cas_ast::Context,
    terms: &[(cas_ast::ExprId, Sign)],
) -> cas_ast::ExprId {
    let signed_terms: Vec<_> = terms
        .iter()
        .map(|(term, sign)| match sign {
            Sign::Pos => *term,
            Sign::Neg => ctx.add(Expr::Neg(*term)),
        })
        .collect();

    match signed_terms.len() {
        0 => ctx.num(0),
        1 => signed_terms[0],
        _ => build_balanced_add(ctx, &signed_terms),
    }
}

fn try_match_log_product_power_cancellation_side(
    ctx: &mut cas_ast::Context,
    focus_expr: cas_ast::ExprId,
) -> Option<LogPowerProductCancellationMatch> {
    let factors = flatten_mul_chain(ctx, focus_expr);
    let mut log_factor = None;
    let mut scale_factors = Vec::new();
    for factor in factors {
        if try_extract_log_parts(ctx, factor).is_some() {
            if log_factor.is_some() {
                return None;
            }
            log_factor = Some(factor);
        } else {
            scale_factors.push(factor);
        }
    }

    let log_factor = log_factor?;
    let (log_base, log_arg) = try_extract_log_parts(ctx, log_factor)?;
    if abs_argument(ctx, log_arg).is_some() {
        return None;
    }

    let scale = match scale_factors.len() {
        0 => ctx.num(1),
        1 => scale_factors[0],
        _ => build_balanced_mul(ctx, &scale_factors),
    };

    let raw_factor_terms: Vec<(cas_ast::ExprId, Sign)> = if as_mul(ctx, log_arg).is_some() {
        let mut terms = Vec::new();
        for factor in flatten_mul_chain(ctx, log_arg) {
            let log_factor = make_log_expr(ctx, log_base, factor);
            terms.push((build_scaled_expr(ctx, scale, log_factor), Sign::Pos));
        }
        terms
    } else if let Some((num, den)) = as_div(ctx, log_arg) {
        let log_num = make_log_expr(ctx, log_base, num);
        let log_den = make_log_expr(ctx, log_base, den);
        vec![
            (build_scaled_expr(ctx, scale, log_num), Sign::Pos),
            (build_scaled_expr(ctx, scale, log_den), Sign::Neg),
        ]
    } else {
        return None;
    };

    let mut normalized_terms = Vec::with_capacity(raw_factor_terms.len());
    let mut needs_power_split = false;
    for (raw_component, sign) in &raw_factor_terms {
        let normalized_component = normalize_log_term_expr_for_cancellation(ctx, *raw_component)?;
        if compare_expr(ctx, normalized_component, *raw_component) != Ordering::Equal {
            needs_power_split = true;
        }
        normalized_terms.push((normalized_component, *sign));
    }

    Some(LogPowerProductCancellationMatch {
        raw_focus_after: build_signed_add_expr(ctx, &raw_factor_terms),
        focus_after: build_signed_add_expr(ctx, &normalized_terms),
        components: normalized_terms,
        needs_power_split,
    })
}

fn rebuild_subtractive_expr(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
    was_add_with_neg: bool,
) -> cas_ast::ExprId {
    if was_add_with_neg {
        let neg_rhs = ctx.add(Expr::Neg(rhs));
        ctx.add(Expr::Add(lhs, neg_rhs))
    } else {
        ctx.add(Expr::Sub(lhs, rhs))
    }
}

fn exprs_match_for_cancellation(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    if compare_expr(ctx, lhs, rhs) == Ordering::Equal
        || cas_math::expr_domain::exprs_equivalent(ctx, lhs, rhs)
        || exprs_equal_up_to_add_term_order(ctx, lhs, rhs)
    {
        return true;
    }

    let lhs_normalized = cas_math::canonical_forms::normalize_core(ctx, lhs);
    let rhs_normalized = cas_math::canonical_forms::normalize_core(ctx, rhs);
    compare_expr(ctx, lhs_normalized, rhs_normalized) == Ordering::Equal
        || cas_math::expr_domain::exprs_equivalent(ctx, lhs_normalized, rhs_normalized)
        || exprs_equal_up_to_add_term_order(ctx, lhs_normalized, rhs_normalized)
}

fn exprs_match_after_default_simplify(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    if exprs_match_for_cancellation(ctx, lhs, rhs) {
        return true;
    }

    let lhs_simplified = run_default_simplify(ctx, lhs);
    let rhs_simplified = run_default_simplify(ctx, rhs);
    exprs_match_for_cancellation(ctx, lhs_simplified, rhs_simplified)
}

fn expr_matches_negation_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    target: cas_ast::ExprId,
) -> bool {
    let neg_target = ctx.add(Expr::Neg(target));
    exprs_match_for_cancellation(ctx, expr, neg_target)
}

fn expr_matches_negation_after_default_simplify(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    target: cas_ast::ExprId,
) -> bool {
    let neg_target = ctx.add(Expr::Neg(target));
    exprs_match_after_default_simplify(ctx, expr, neg_target)
}

fn signed_term_expr(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    sign: Sign,
) -> cas_ast::ExprId {
    match sign {
        Sign::Pos => expr,
        Sign::Neg => ctx.add(Expr::Neg(expr)),
    }
}

fn build_signed_sum_expr(
    ctx: &mut cas_ast::Context,
    terms: &[(cas_ast::ExprId, Sign)],
) -> cas_ast::ExprId {
    let Some((first_expr, first_sign)) = terms.first().copied() else {
        return ctx.num(0);
    };
    let mut acc = signed_term_expr(ctx, first_expr, first_sign);
    for (expr, sign) in terms.iter().copied().skip(1) {
        let term = signed_term_expr(ctx, expr, sign);
        acc = ctx.add(Expr::Add(acc, term));
    }
    acc
}

fn try_rewrite_trig_sum_to_product_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, &'static str)> {
    if let Some(rewrite) = try_rewrite_sum_to_product_contraction_expr(ctx, expr) {
        let description = match rewrite.kind {
            TrigSumToProductContractionRewriteKind::SinSum => "Expand sine sum to product",
            TrigSumToProductContractionRewriteKind::SinDiff => "Expand sine difference to product",
            TrigSumToProductContractionRewriteKind::CosSum => "Expand cosine sum to product",
            TrigSumToProductContractionRewriteKind::CosDiff => {
                "Expand cosine difference to product"
            }
        };
        return Some((rewrite.rewritten, description));
    }

    let two = ctx.num(2);

    if let Some((arg_a, arg_b)) = extract_trig_two_term_sum(ctx, expr, "sin") {
        let avg_arg = build_avg_with_simplifier(ctx, arg_a, arg_b, crate::collect::collect);
        let half_diff_arg =
            build_half_diff_with_simplifier(ctx, arg_a, arg_b, false, crate::collect::collect);
        let sin_avg = ctx.call_builtin(BuiltinFn::Sin, vec![avg_arg]);
        let cos_half_diff = ctx.call_builtin(BuiltinFn::Cos, vec![half_diff_arg]);
        let product = smart_mul(ctx, sin_avg, cos_half_diff);
        return Some((smart_mul(ctx, two, product), "Expand sine sum to product"));
    }

    if let Some((arg_a, arg_b)) = extract_trig_two_term_diff(ctx, expr, "sin") {
        let avg_arg = build_avg_with_simplifier(ctx, arg_a, arg_b, crate::collect::collect);
        let half_diff_arg =
            build_half_diff_with_simplifier(ctx, arg_a, arg_b, false, crate::collect::collect);
        let cos_avg = ctx.call_builtin(BuiltinFn::Cos, vec![avg_arg]);
        let sin_half_diff = ctx.call_builtin(BuiltinFn::Sin, vec![half_diff_arg]);
        let product = smart_mul(ctx, cos_avg, sin_half_diff);
        return Some((
            smart_mul(ctx, two, product),
            "Expand sine difference to product",
        ));
    }

    if let Some((arg_a, arg_b)) = extract_trig_two_term_sum(ctx, expr, "cos") {
        let avg_arg = build_avg_with_simplifier(ctx, arg_a, arg_b, crate::collect::collect);
        let half_diff_arg =
            build_half_diff_with_simplifier(ctx, arg_a, arg_b, false, crate::collect::collect);
        let half_diff_arg = normalize_for_even_fn(ctx, half_diff_arg);
        let cos_avg = ctx.call_builtin(BuiltinFn::Cos, vec![avg_arg]);
        let cos_half_diff = ctx.call_builtin(BuiltinFn::Cos, vec![half_diff_arg]);
        let product = smart_mul(ctx, cos_avg, cos_half_diff);
        return Some((smart_mul(ctx, two, product), "Expand cosine sum to product"));
    }

    if let Some((arg_a, arg_b)) = extract_trig_two_term_diff(ctx, expr, "cos") {
        let avg_arg = build_avg_with_simplifier(ctx, arg_a, arg_b, crate::collect::collect);
        let half_diff_arg =
            build_half_diff_with_simplifier(ctx, arg_a, arg_b, false, crate::collect::collect);
        let sin_avg = ctx.call_builtin(BuiltinFn::Sin, vec![avg_arg]);
        let sin_half_diff = ctx.call_builtin(BuiltinFn::Sin, vec![half_diff_arg]);
        let product = smart_mul(ctx, sin_avg, sin_half_diff);
        let two_product = smart_mul(ctx, two, product);
        return Some((
            ctx.add(Expr::Neg(two_product)),
            "Expand cosine difference to product",
        ));
    }

    None
}

fn try_rewrite_hyperbolic_angle_sum_diff_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let rewritten = cas_math::expand_ops::expand(ctx, expr);
    (rewritten != expr).then_some(rewritten)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrigPhaseShiftCancellationMode {
    LinearToShifted,
    ShiftedToLinear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TrigPhaseShiftCancellationMatch {
    local_before: cas_ast::ExprId,
    local_after: cas_ast::ExprId,
    mode: TrigPhaseShiftCancellationMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GeneralPhaseShiftTermData {
    coeff: cas_ast::ExprId,
    trig_fn: BuiltinFn,
    base_arg: cas_ast::ExprId,
    ratio: cas_ast::ExprId,
    subtract_shift: bool,
    global_sign: i8,
}

fn maybe_trig_phase_shift_zero_candidate(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    expr_contains_any_builtin(ctx, expr, &[BuiltinFn::Sin, BuiltinFn::Cos])
        && expr_contains_any_builtin(ctx, expr, &[BuiltinFn::Atan, BuiltinFn::Arctan])
}

fn strip_unit_negation_for_phase_shift(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    match ctx.get(expr) {
        Expr::Neg(inner) => return Some(*inner),
        Expr::Number(n) if n.is_negative() => return Some(ctx.add(Expr::Number(-n))),
        Expr::Mul(left, right) if is_minus_one_expr(ctx, *left) => return Some(*right),
        Expr::Mul(left, right) if is_minus_one_expr(ctx, *right) => return Some(*left),
        Expr::Mul(_, _) => {}
        _ => return None,
    }

    let factors = flatten_mul_chain(ctx, expr);
    for (index, factor) in factors.iter().copied().enumerate() {
        let replacement = match ctx.get(factor).clone() {
            Expr::Neg(inner) => Some(inner),
            Expr::Number(n) if n.is_negative() => Some(ctx.add(Expr::Number(-n))),
            _ => None,
        };
        let Some(replacement) = replacement else {
            continue;
        };

        let mut rebuilt = None;
        for (other_index, other) in factors.iter().copied().enumerate() {
            let current = if other_index == index {
                replacement
            } else {
                other
            };
            rebuilt = Some(match rebuilt {
                Some(acc) => smart_mul(ctx, acc, current),
                None => current,
            });
        }
        return rebuilt;
    }

    None
}

fn extract_sin_or_cos_linear_term_for_phase_shift(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BuiltinFn, cas_ast::ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
        Some((BuiltinFn::Sin, args[0]))
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
        Some((BuiltinFn::Cos, args[0]))
    } else {
        None
    }
}

fn extract_scaled_sin_or_cos_linear_term_for_phase_shift(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BuiltinFn, cas_ast::ExprId, cas_ast::ExprId)> {
    if let Some((trig_fn, arg)) = extract_sin_or_cos_linear_term_for_phase_shift(ctx, expr) {
        return Some((trig_fn, arg, ctx.num(1)));
    }

    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    let mut trig_term = None;
    let mut coeff_factors = Vec::new();
    for factor in factors {
        if let Some((trig_fn, arg)) = extract_sin_or_cos_linear_term_for_phase_shift(ctx, factor) {
            if trig_term.is_some() {
                return None;
            }
            trig_term = Some((trig_fn, arg));
        } else {
            coeff_factors.push(factor);
        }
    }

    let (trig_fn, arg) = trig_term?;
    if coeff_factors.is_empty() {
        return None;
    }

    let coeff = coeff_factors
        .into_iter()
        .fold(ctx.num(1), |acc, factor| smart_mul(ctx, acc, factor));
    Some((trig_fn, arg, coeff))
}

fn normalize_phase_shift_coefficient_sign_for_cancellation(
    ctx: &mut cas_ast::Context,
    coeff: cas_ast::ExprId,
    sign: i8,
) -> (cas_ast::ExprId, i8) {
    if let Some(inner) = strip_unit_negation_for_phase_shift(ctx, coeff) {
        (inner, -sign)
    } else {
        (coeff, sign)
    }
}

fn is_atan_call_for_phase_shift(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return false;
    };
    args.len() == 1
        && matches!(
            ctx.builtin_of(*fn_id),
            Some(BuiltinFn::Atan | BuiltinFn::Arctan)
        )
}

fn extract_general_phase_shift_argument_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, bool)> {
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            if is_atan_call_for_phase_shift(ctx, left) {
                Some((right, left, false))
            } else if is_atan_call_for_phase_shift(ctx, right) {
                Some((left, right, false))
            } else {
                None
            }
        }
        Expr::Sub(left, right) => {
            is_atan_call_for_phase_shift(ctx, right).then_some((left, right, true))
        }
        _ => None,
    }
}

fn extract_weighted_phase_shift_linear_combination_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, cas_ast::ExprId, i8, i8)> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
    if terms.len() != 2 {
        return None;
    }

    let mut sin_term = None;
    let mut cos_term = None;

    for (term, sign) in terms {
        let signed = match sign {
            Sign::Pos => 1,
            Sign::Neg => -1,
        };
        let (trig_fn, arg, coeff) =
            extract_scaled_sin_or_cos_linear_term_for_phase_shift(ctx, term)?;
        let (coeff, signed) =
            normalize_phase_shift_coefficient_sign_for_cancellation(ctx, coeff, signed);

        match trig_fn {
            BuiltinFn::Sin => {
                if sin_term.is_some() {
                    return None;
                }
                sin_term = Some((arg, coeff, signed));
            }
            BuiltinFn::Cos => {
                if cos_term.is_some() {
                    return None;
                }
                cos_term = Some((arg, coeff, signed));
            }
            _ => return None,
        }
    }

    let (sin_arg, sin_coeff, sin_sign) = sin_term?;
    let (cos_arg, cos_coeff, cos_sign) = cos_term?;
    if compare_expr(ctx, sin_arg, cos_arg) != Ordering::Equal {
        return None;
    }

    Some((sin_arg, sin_coeff, cos_coeff, sin_sign, cos_sign))
}

fn extract_general_phase_shift_term_data_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<GeneralPhaseShiftTermData> {
    let (global_sign, positive_expr) =
        if let Some(inner) = strip_unit_negation_for_phase_shift(ctx, expr) {
            (-1_i8, inner)
        } else {
            (1_i8, expr)
        };

    if let Some((trig_fn, raw_arg)) =
        extract_sin_or_cos_linear_term_for_phase_shift(ctx, positive_expr)
    {
        let (base_arg, ratio, subtract_shift) =
            extract_general_phase_shift_argument_for_cancellation(ctx, raw_arg)?;
        return Some(GeneralPhaseShiftTermData {
            coeff: ctx.num(1),
            trig_fn,
            base_arg,
            ratio,
            subtract_shift,
            global_sign,
        });
    }

    let factors = flatten_mul_chain(ctx, positive_expr);
    if factors.len() < 2 {
        return None;
    }

    let mut trig_term = None;
    let mut coeff_factors = Vec::new();
    for factor in factors {
        if trig_term.is_none() {
            if let Some((trig_fn, raw_arg)) =
                extract_sin_or_cos_linear_term_for_phase_shift(ctx, factor)
            {
                trig_term = Some((trig_fn, raw_arg));
                continue;
            }
        }
        coeff_factors.push(factor);
    }

    let (trig_fn, raw_arg) = trig_term?;
    let (base_arg, ratio, subtract_shift) =
        extract_general_phase_shift_argument_for_cancellation(ctx, raw_arg)?;
    let coeff = coeff_factors
        .into_iter()
        .fold(ctx.num(1), |acc, factor| smart_mul(ctx, acc, factor));
    let (coeff, global_sign) =
        normalize_phase_shift_coefficient_sign_for_cancellation(ctx, coeff, global_sign);

    Some(GeneralPhaseShiftTermData {
        coeff,
        trig_fn,
        base_arg,
        ratio,
        subtract_shift,
        global_sign,
    })
}

fn build_weighted_phase_shift_linear_combination_for_cancellation(
    ctx: &mut cas_ast::Context,
    arg: cas_ast::ExprId,
    sin_coeff: cas_ast::ExprId,
    cos_coeff: cas_ast::ExprId,
    sin_sign: i8,
    cos_sign: i8,
) -> cas_ast::ExprId {
    let sin_call = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
    let cos_call = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
    let sin_term = smart_mul(ctx, sin_coeff, sin_call);
    let cos_term = smart_mul(ctx, cos_coeff, cos_call);
    let signed_sin = if sin_sign < 0 {
        ctx.add(Expr::Neg(sin_term))
    } else {
        sin_term
    };
    let signed_cos = if cos_sign < 0 {
        ctx.add(Expr::Neg(cos_term))
    } else {
        cos_term
    };

    ctx.add(Expr::Add(signed_sin, signed_cos))
}

fn build_general_phase_shift_linear_combination_for_cancellation(
    ctx: &mut cas_ast::Context,
    data: GeneralPhaseShiftTermData,
) -> Option<cas_ast::ExprId> {
    let Expr::Function(atan_fn, atan_args) = ctx.get(data.ratio) else {
        return None;
    };
    if atan_args.len() != 1
        || !matches!(
            ctx.builtin_of(*atan_fn),
            Some(BuiltinFn::Atan | BuiltinFn::Arctan)
        )
    {
        return None;
    }

    let ratio_arg = atan_args[0];
    let (sin_shift, _) = cas_math::trig_inverse_expansion_support::expand_trig_inverse_composition(
        ctx, "sin", "arctan", ratio_arg,
    )?;
    let (cos_shift, _) = cas_math::trig_inverse_expansion_support::expand_trig_inverse_composition(
        ctx, "cos", "arctan", ratio_arg,
    )?;

    let sin_coeff = smart_mul(ctx, data.coeff, cos_shift);
    let cos_coeff = smart_mul(ctx, data.coeff, sin_shift);
    let (sin_sign, cos_sign) = match (data.trig_fn, data.subtract_shift) {
        (BuiltinFn::Sin, false) => (data.global_sign, data.global_sign),
        (BuiltinFn::Sin, true) => (data.global_sign, -data.global_sign),
        (BuiltinFn::Cos, false) => (-data.global_sign, data.global_sign),
        (BuiltinFn::Cos, true) => (data.global_sign, data.global_sign),
        _ => return None,
    };
    let rewritten = build_weighted_phase_shift_linear_combination_for_cancellation(
        ctx,
        data.base_arg,
        sin_coeff,
        cos_coeff,
        sin_sign,
        cos_sign,
    );

    Some(run_default_simplify(ctx, rewritten))
}

fn build_general_phase_shift_sine_term_candidate_for_cancellation(
    ctx: &mut cas_ast::Context,
    arg: cas_ast::ExprId,
    sin_coeff: cas_ast::ExprId,
    cos_coeff: cas_ast::ExprId,
    sin_sign: i8,
    cos_sign: i8,
) -> cas_ast::ExprId {
    let ratio_raw = ctx.add(Expr::Div(cos_coeff, sin_coeff));
    let ratio = run_default_simplify(ctx, ratio_raw);
    let shift = ctx.call_builtin(BuiltinFn::Arctan, vec![ratio]);
    let shifted_arg = if sin_sign == cos_sign {
        ctx.add(Expr::Add(arg, shift))
    } else {
        ctx.add(Expr::Sub(arg, shift))
    };

    let sin_sq = smart_mul(ctx, sin_coeff, sin_coeff);
    let cos_sq = smart_mul(ctx, cos_coeff, cos_coeff);
    let amplitude_sq = ctx.add(Expr::Add(sin_sq, cos_sq));
    let amplitude_raw = ctx.call_builtin(BuiltinFn::Sqrt, vec![amplitude_sq]);
    let amplitude = run_default_simplify(ctx, amplitude_raw);
    let shifted_sine = ctx.call_builtin(BuiltinFn::Sin, vec![shifted_arg]);
    let rewritten = smart_mul(ctx, amplitude, shifted_sine);
    let rewritten = if sin_sign < 0 {
        ctx.add(Expr::Neg(rewritten))
    } else {
        rewritten
    };

    run_default_simplify(ctx, rewritten)
}

fn build_general_phase_shift_cosine_term_candidate_for_cancellation(
    ctx: &mut cas_ast::Context,
    arg: cas_ast::ExprId,
    sin_coeff: cas_ast::ExprId,
    cos_coeff: cas_ast::ExprId,
    sin_sign: i8,
    cos_sign: i8,
) -> cas_ast::ExprId {
    let ratio_raw = ctx.add(Expr::Div(sin_coeff, cos_coeff));
    let ratio = run_default_simplify(ctx, ratio_raw);
    let shift = ctx.call_builtin(BuiltinFn::Arctan, vec![ratio]);
    let shifted_arg = if sin_sign == cos_sign {
        ctx.add(Expr::Sub(arg, shift))
    } else {
        ctx.add(Expr::Add(arg, shift))
    };

    let sin_sq = smart_mul(ctx, sin_coeff, sin_coeff);
    let cos_sq = smart_mul(ctx, cos_coeff, cos_coeff);
    let amplitude_sq = ctx.add(Expr::Add(sin_sq, cos_sq));
    let amplitude_raw = ctx.call_builtin(BuiltinFn::Sqrt, vec![amplitude_sq]);
    let amplitude = run_default_simplify(ctx, amplitude_raw);
    let shifted_cosine = ctx.call_builtin(BuiltinFn::Cos, vec![shifted_arg]);
    let rewritten = smart_mul(ctx, amplitude, shifted_cosine);
    let rewritten = if cos_sign < 0 {
        ctx.add(Expr::Neg(rewritten))
    } else {
        rewritten
    };

    run_default_simplify(ctx, rewritten)
}

fn try_find_trig_phase_shift_cancellation_match(
    ctx: &mut cas_ast::Context,
    focus_expr: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
    target_is_negated: bool,
) -> Option<TrigPhaseShiftCancellationMatch> {
    let matches_target = |ctx: &mut cas_ast::Context, candidate: cas_ast::ExprId| {
        if target_is_negated {
            expr_matches_negation_after_default_simplify(ctx, candidate, target_expr)
        } else {
            exprs_match_after_default_simplify(ctx, candidate, target_expr)
        }
    };

    if let Some((arg, sin_coeff, cos_coeff, sin_sign, cos_sign)) =
        extract_weighted_phase_shift_linear_combination_for_cancellation(ctx, focus_expr)
    {
        let sine_candidate = build_general_phase_shift_sine_term_candidate_for_cancellation(
            ctx, arg, sin_coeff, cos_coeff, sin_sign, cos_sign,
        );
        if matches_target(ctx, sine_candidate) {
            return Some(TrigPhaseShiftCancellationMatch {
                local_before: focus_expr,
                local_after: sine_candidate,
                mode: TrigPhaseShiftCancellationMode::LinearToShifted,
            });
        }

        let cosine_candidate = build_general_phase_shift_cosine_term_candidate_for_cancellation(
            ctx, arg, sin_coeff, cos_coeff, sin_sign, cos_sign,
        );
        if matches_target(ctx, cosine_candidate) {
            return Some(TrigPhaseShiftCancellationMatch {
                local_before: focus_expr,
                local_after: cosine_candidate,
                mode: TrigPhaseShiftCancellationMode::LinearToShifted,
            });
        }
    }

    if let Some(data) = extract_general_phase_shift_term_data_for_cancellation(ctx, focus_expr) {
        let expanded = build_general_phase_shift_linear_combination_for_cancellation(ctx, data)?;
        if matches_target(ctx, expanded) {
            return Some(TrigPhaseShiftCancellationMatch {
                local_before: focus_expr,
                local_after: expanded,
                mode: TrigPhaseShiftCancellationMode::ShiftedToLinear,
            });
        }
    }

    None
}

fn build_trig_phase_shift_zero_rewrite(
    ctx: &mut cas_ast::Context,
    rewrite_match: TrigPhaseShiftCancellationMatch,
) -> Rewrite {
    let focus_after_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: rewrite_match.local_after
        }
    );

    let rewrite = Rewrite::with_local(
        ctx.num(0),
        "Phase Shift Identity",
        rewrite_match.local_before,
        rewrite_match.local_after,
    );

    match rewrite_match.mode {
        TrigPhaseShiftCancellationMode::LinearToShifted => rewrite
            .substep(
                "Reescribir la combinación lineal",
                vec![format!("Se obtiene {focus_after_display}.")],
            )
            .substep(
                "Cancelar términos iguales",
                vec![
                    "Tras aplicar la identidad de desfase, el término restante es exactamente el opuesto y toda la expresión se anula."
                        .to_string(),
                ],
            ),
        TrigPhaseShiftCancellationMode::ShiftedToLinear => rewrite
            .substep(
                "Expandir el término desplazado",
                vec![format!("Se obtiene {focus_after_display}.")],
            )
            .substep(
                "Cancelar términos iguales",
                vec![
                    "Tras expandir el término desplazado, el resto de la expresión es exactamente el opuesto y el resultado es 0."
                        .to_string(),
                ],
            ),
    }
}

fn extract_scaled_double_sine_product_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 3 {
        return None;
    }

    let mut numeric_coeff = BigRational::one();
    let mut residual_factors = Vec::new();
    let mut sine_args = Vec::new();

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }

        if let Some((BuiltinFn::Sin, arg)) =
            extract_sin_or_cos_linear_term_for_phase_shift(ctx, factor)
        {
            sine_args.push(arg);
            continue;
        }

        residual_factors.push(factor);
    }

    if sine_args.len() != 2 || numeric_coeff.is_zero() {
        return None;
    }

    let two = ctx.num(2);
    let try_match_pair = |ctx: &mut cas_ast::Context,
                          left: cas_ast::ExprId,
                          right: cas_ast::ExprId|
     -> Option<cas_ast::ExprId> {
        let doubled_right = smart_mul(ctx, two, right);
        exprs_match_for_cancellation(ctx, left, doubled_right).then_some(right)
    };

    let base_arg = try_match_pair(ctx, sine_args[0], sine_args[1])
        .or_else(|| try_match_pair(ctx, sine_args[1], sine_args[0]))?;

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

    Some((run_default_simplify(ctx, scale), base_arg))
}

fn build_trig_sine_product_to_sum_intermediate(
    ctx: &mut cas_ast::Context,
    scale: cas_ast::ExprId,
    arg: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let three = ctx.num(3);
    let triple_arg = smart_mul(ctx, three, arg);
    let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
    let cos_triple = ctx.call_builtin(BuiltinFn::Cos, vec![triple_arg]);
    let base = ctx.add(Expr::Sub(cos_arg, cos_triple));
    build_scaled_expr(ctx, scale, base)
}

fn build_trig_sine_product_cubic_polynomial(
    ctx: &mut cas_ast::Context,
    scale: cas_ast::ExprId,
    arg: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
    let three = ctx.num(3);
    let four = ctx.num(4);
    let cos_cubed = ctx.add(Expr::Pow(cos_arg, three));
    let linear = smart_mul(ctx, four, cos_arg);
    let cubic = smart_mul(ctx, four, cos_cubed);
    let base = ctx.add(Expr::Sub(linear, cubic));
    build_scaled_expr(ctx, scale, base)
}

fn build_trig_sine_product_cosine_cubic_target(
    ctx: &mut cas_ast::Context,
    scale: cas_ast::ExprId,
    arg: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let scaled = build_trig_sine_product_cubic_polynomial(ctx, scale, arg);
    run_default_simplify(ctx, scaled)
}

fn build_trig_sine_product_triple_angle_zero_rewrite(
    ctx: &mut cas_ast::Context,
    scope_expr: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
    scale: cas_ast::ExprId,
    arg: cas_ast::ExprId,
) -> Rewrite {
    let product_sum = build_trig_sine_product_to_sum_intermediate(ctx, scale, arg);
    let cubic_polynomial = build_trig_sine_product_cubic_polynomial(ctx, scale, arg);
    let product_sum_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: product_sum
        }
    );
    let cubic_polynomial_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: cubic_polynomial
        }
    );
    let target_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: target_expr
        }
    );

    Rewrite::with_local(
        ctx.num(0),
        "Product-to-Sum and Triple-Angle Identity",
        scope_expr,
        ctx.num(0),
    )
    .substep(
        "Convertir producto de senos en diferencia de cosenos",
        vec![format!("Se obtiene {product_sum_display}.")],
    )
    .substep(
        "Usar cos(3u) = 4·cos(u)^3 - 3·cos(u)",
        vec![format!(
            "La expresión se reescribe como {cubic_polynomial_display}."
        )],
    )
    .substep(
        "Usar 1 - cos(u)^2 = sin(u)^2",
        vec![format!("Así se obtiene {target_display}.")],
    )
    .substep(
        "Cancelar términos iguales",
        vec![
            "Tras reconocer la misma forma en el otro lado, toda la expresión se anula."
                .to_string(),
        ],
    )
}

fn try_build_exact_trig_sine_product_triple_angle_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    for index in 0..view.terms.len() {
        let (term_expr, term_sign) =
            normalize_signed_add_term(ctx, view.terms[index].0, view.terms[index].1);
        let Some((scale, arg)) =
            extract_scaled_double_sine_product_for_cancellation(ctx, term_expr)
        else {
            continue;
        };
        let target_expr = build_trig_sine_product_cosine_cubic_target(ctx, scale, arg);
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
        let remaining_expr = build_signed_sum_expr(ctx, &remaining_terms);

        let matches = match term_sign {
            Sign::Pos => {
                expr_matches_negation_after_default_simplify(ctx, remaining_expr, target_expr)
            }
            Sign::Neg => exprs_match_after_default_simplify(ctx, remaining_expr, target_expr),
        };
        if matches {
            return Some(build_trig_sine_product_triple_angle_zero_rewrite(
                ctx,
                expr,
                target_expr,
                scale,
                arg,
            ));
        }
    }

    None
}

fn maybe_trig_square_zero_candidate(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    expr_contains_any_builtin(ctx, expr, &[BuiltinFn::Sin, BuiltinFn::Cos])
}

fn extract_trig_binomial_square_identity_data(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, bool)> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    if extract_i64_integer(ctx, *exponent)? != 2 {
        return None;
    }

    let (left, right, is_sum) = match ctx.get(*base) {
        Expr::Add(left, right) => (*left, *right, true),
        Expr::Sub(left, right) => (*left, *right, false),
        _ => return None,
    };

    let lhs = extract_sin_or_cos_linear_term_for_phase_shift(ctx, left)?;
    let rhs = extract_sin_or_cos_linear_term_for_phase_shift(ctx, right)?;
    if compare_expr(ctx, lhs.1, rhs.1) != Ordering::Equal {
        return None;
    }

    let trig_kinds = [lhs.0, rhs.0];
    if !trig_kinds.contains(&BuiltinFn::Sin) || !trig_kinds.contains(&BuiltinFn::Cos) {
        return None;
    }

    Some((lhs.1, is_sum))
}

fn build_trig_square_double_angle_term(
    ctx: &mut cas_ast::Context,
    arg: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let two = ctx.num(2);
    let doubled_arg = smart_mul(ctx, two, arg);
    ctx.call_builtin(BuiltinFn::Sin, vec![doubled_arg])
}

fn build_trig_binomial_square_target(
    ctx: &mut cas_ast::Context,
    arg: cas_ast::ExprId,
    is_sum: bool,
) -> cas_ast::ExprId {
    let double_angle = build_trig_square_double_angle_term(ctx, arg);
    let one = ctx.num(1);
    let target = if is_sum {
        ctx.add(Expr::Add(one, double_angle))
    } else {
        ctx.add(Expr::Sub(one, double_angle))
    };
    run_default_simplify(ctx, target)
}

fn build_trig_binomial_square_expansion(
    ctx: &mut cas_ast::Context,
    arg: cas_ast::ExprId,
    is_sum: bool,
) -> cas_ast::ExprId {
    let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
    let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
    let two = ctx.num(2);
    let sin_sq = ctx.add(Expr::Pow(sin_arg, two));
    let cos_sq = ctx.add(Expr::Pow(cos_arg, two));
    let sin_cos = smart_mul(ctx, sin_arg, cos_arg);
    let cross = smart_mul(ctx, two, sin_cos);
    let partial = if is_sum {
        ctx.add(Expr::Add(sin_sq, cross))
    } else {
        ctx.add(Expr::Sub(sin_sq, cross))
    };
    ctx.add(Expr::Add(partial, cos_sq))
}

fn build_trig_binomial_square_reduced(
    ctx: &mut cas_ast::Context,
    arg: cas_ast::ExprId,
    is_sum: bool,
) -> cas_ast::ExprId {
    let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
    let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
    let two = ctx.num(2);
    let sin_cos = smart_mul(ctx, sin_arg, cos_arg);
    let cross = smart_mul(ctx, two, sin_cos);
    let one = ctx.num(1);
    if is_sum {
        ctx.add(Expr::Add(one, cross))
    } else {
        ctx.add(Expr::Sub(one, cross))
    }
}

fn build_trig_square_zero_rewrite(
    ctx: &mut cas_ast::Context,
    square_expr: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
    arg: cas_ast::ExprId,
    is_sum: bool,
) -> Rewrite {
    let expanded = build_trig_binomial_square_expansion(ctx, arg, is_sum);
    let reduced = build_trig_binomial_square_reduced(ctx, arg, is_sum);
    let expanded_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: expanded
        }
    );
    let reduced_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: reduced
        }
    );
    let target_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: target_expr
        }
    );

    Rewrite::with_local(ctx.num(0), "Trig Square Identity", square_expr, target_expr)
        .substep(
            "Expandir el binomio",
            vec![format!("Se obtiene {expanded_display}.")],
        )
        .substep(
            "Usar sin(u)^2 + cos(u)^2 = 1",
            vec![format!("Así queda {reduced_display}.")],
        )
        .substep(
            "Usar 2 · sin(u) · cos(u) = sin(2u)",
            vec![format!("La expresión se convierte en {target_display}.")],
        )
        .substep(
            "Cancelar términos iguales",
            vec![
                "Tras reconocer la misma identidad en el otro lado, toda la expresión se anula."
                    .to_string(),
            ],
        )
}

fn try_build_exact_trig_square_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    for index in 0..view.terms.len() {
        let (term_expr, term_sign) = view.terms[index];
        let Some((arg, is_sum)) = extract_trig_binomial_square_identity_data(ctx, term_expr) else {
            continue;
        };
        let target_expr = build_trig_binomial_square_target(ctx, arg, is_sum);
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
        let remaining_expr = build_signed_sum_expr(ctx, &remaining_terms);

        let matches = match term_sign {
            Sign::Pos => {
                expr_matches_negation_after_default_simplify(ctx, remaining_expr, target_expr)
            }
            Sign::Neg => exprs_match_after_default_simplify(ctx, remaining_expr, target_expr),
        };
        if matches {
            return Some(build_trig_square_zero_rewrite(
                ctx,
                term_expr,
                target_expr,
                arg,
                is_sum,
            ));
        }
    }

    None
}

fn try_rewrite_hyperbolic_pythagorean_factor_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<HyperbolicPythagoreanFactorCancellationMatch> {
    let (lhs, rhs) = match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) => (lhs, rhs),
        Expr::Add(lhs, rhs) => match ctx.get(rhs).clone() {
            Expr::Neg(inner) => (lhs, inner),
            _ => match ctx.get(lhs).clone() {
                Expr::Neg(inner) => (rhs, inner),
                _ => return None,
            },
        },
        _ => return None,
    };

    if let (
        Some((lhs_sign, lhs_coeff, lhs_arg, lhs_power)),
        Some((rhs_sign, rhs_coeff, rhs_arg, rhs_power)),
    ) = (
        extract_signed_cosh_power(ctx, lhs),
        extract_signed_cosh_power(ctx, rhs),
    ) {
        if compare_expr(ctx, lhs_arg, rhs_arg) == Ordering::Equal
            && lhs_sign == rhs_sign
            && exprs_match_for_cancellation(ctx, lhs_coeff, rhs_coeff)
        {
            let coeff_with_sign = apply_sign_to_expr(ctx, lhs_sign, lhs_coeff);
            let oriented_coeff = match (lhs_power, rhs_power) {
                (3, 1) => coeff_with_sign,
                (1, 3) => ctx.add(Expr::Neg(coeff_with_sign)),
                _ => return None,
            };

            let cosh_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![lhs_arg]);
            let sinh_arg = ctx.call_builtin(BuiltinFn::Sinh, vec![lhs_arg]);
            let two = ctx.num(2);
            let one = ctx.num(1);
            let cosh_sq = ctx.add(Expr::Pow(cosh_arg, two));
            let sinh_sq = ctx.add(Expr::Pow(sinh_arg, two));
            let factorized_inner = ctx.add(Expr::Sub(cosh_sq, one));
            let factorized_product = smart_mul(ctx, cosh_arg, factorized_inner);
            let rewritten_product = smart_mul(ctx, cosh_arg, sinh_sq);
            let factorized = smart_mul(ctx, oriented_coeff, factorized_product);
            let rewritten = smart_mul(ctx, oriented_coeff, rewritten_product);

            return Some(HyperbolicPythagoreanFactorCancellationMatch {
                local_before: factorized,
                local_after: rewritten,
                mode: HyperbolicPythagoreanFactorCancellationMode::FactorThenRewrite {
                    factorized,
                    rewritten,
                },
            });
        }
    }

    try_rewrite_factored_hyperbolic_pythagorean_for_cancellation(ctx, lhs, rhs)
        .or_else(|| try_rewrite_factored_hyperbolic_pythagorean_for_cancellation(ctx, rhs, lhs))
}

fn extract_sinh_sq_multiplier(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    arg: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let mut coefficient_factors = smallvec::SmallVec::<[cas_ast::ExprId; 8]>::new();
    let mut seen_sinh_sq = false;

    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        match ctx.get(factor) {
            Expr::Pow(base, exponent) => {
                let Expr::Function(fn_id, args) = ctx.get(*base) else {
                    coefficient_factors.push(factor);
                    continue;
                };
                if !ctx.is_builtin(*fn_id, BuiltinFn::Sinh)
                    || args.len() != 1
                    || compare_expr(ctx, args[0], arg) != Ordering::Equal
                    || extract_i64_integer(ctx, *exponent)? != 2
                {
                    coefficient_factors.push(factor);
                    continue;
                }
                if seen_sinh_sq {
                    return None;
                }
                seen_sinh_sq = true;
            }
            _ => coefficient_factors.push(factor),
        }
    }

    if !seen_sinh_sq {
        return None;
    }

    Some(if coefficient_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &coefficient_factors.into_vec())
    })
}

fn match_sinh_sq_plus_one_multiple(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    arg: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    for candidate in [expr, cas_math::canonical_forms::normalize_core(ctx, expr)] {
        let view = AddView::from_expr(ctx, candidate);
        if view.terms.len() != 2 || view.terms.iter().any(|(_, sign)| *sign != Sign::Pos) {
            continue;
        }

        for squared_index in 0..2 {
            let const_index = 1 - squared_index;
            let squared_term = view.terms[squared_index].0;
            let const_term = view.terms[const_index].0;
            let Some(coeff) = extract_sinh_sq_multiplier(ctx, squared_term, arg) else {
                continue;
            };
            if exprs_match_for_cancellation(ctx, coeff, const_term) {
                return Some(coeff);
            }
        }
    }

    None
}

fn extract_factored_hyperbolic_linear_term(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(i64, cas_ast::ExprId, cas_ast::ExprId, cas_ast::ExprId)> {
    let (sign, positive_expr) = match ctx.get(expr) {
        Expr::Neg(inner) => (-1, *inner),
        _ => (1, expr),
    };

    let mut outer_coeff_factors = smallvec::SmallVec::<[cas_ast::ExprId; 8]>::new();
    let mut cosh_arg = None;
    let mut additive_factor = None;

    for factor in cas_math::expr_nary::mul_leaves(ctx, positive_expr) {
        match ctx.get(factor) {
            Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, BuiltinFn::Cosh) && args.len() == 1 =>
            {
                if cosh_arg.is_some() {
                    return None;
                }
                cosh_arg = Some(args[0]);
            }
            Expr::Add(_, _) | Expr::Sub(_, _) => {
                if additive_factor.is_some() {
                    return None;
                }
                additive_factor = Some(factor);
            }
            _ => outer_coeff_factors.push(factor),
        }
    }

    let arg = cosh_arg?;
    let additive = additive_factor?;
    let inner_coeff = match_sinh_sq_plus_one_multiple(ctx, additive, arg)?;
    let outer_coeff = if outer_coeff_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &outer_coeff_factors.into_vec())
    };

    Some((
        sign,
        smart_mul(ctx, outer_coeff, inner_coeff),
        arg,
        positive_expr,
    ))
}

fn try_rewrite_factored_hyperbolic_pythagorean_for_cancellation(
    ctx: &mut cas_ast::Context,
    linear_expr: cas_ast::ExprId,
    cubic_expr: cas_ast::ExprId,
) -> Option<HyperbolicPythagoreanFactorCancellationMatch> {
    let (linear_sign, linear_coeff, linear_arg, linear_before_positive) =
        extract_factored_hyperbolic_linear_term(ctx, linear_expr)?;
    let (cubic_sign, cubic_coeff, cubic_arg, cubic_power) =
        extract_signed_cosh_power(ctx, cubic_expr)?;

    if cubic_power != 3
        || linear_sign != cubic_sign
        || compare_expr(ctx, linear_arg, cubic_arg) != Ordering::Equal
        || !exprs_match_for_cancellation(ctx, linear_coeff, cubic_coeff)
    {
        return None;
    }

    let cosh_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![linear_arg]);
    let three = ctx.num(3);
    let cosh_cubed = ctx.add(Expr::Pow(cosh_arg, three));
    let rewritten_positive = smart_mul(ctx, cubic_coeff, cosh_cubed);

    Some(HyperbolicPythagoreanFactorCancellationMatch {
        local_before: apply_sign_to_expr(ctx, linear_sign, linear_before_positive),
        local_after: apply_sign_to_expr(ctx, cubic_sign, rewritten_positive),
        mode: HyperbolicPythagoreanFactorCancellationMode::AlreadyFactored,
    })
}

fn try_build_exact_trig_sum_to_product_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return None;
    }

    for first_index in 0..view.terms.len() {
        for second_index in (first_index + 1)..view.terms.len() {
            let focus_terms = [view.terms[first_index], view.terms[second_index]];
            let focus_expr = build_signed_sum_expr(ctx, &focus_terms);
            let Some((rewritten, description)) =
                try_rewrite_trig_sum_to_product_for_cancellation(ctx, focus_expr)
            else {
                continue;
            };

            let Some(remaining_index) =
                (0..view.terms.len()).find(|index| *index != first_index && *index != second_index)
            else {
                continue;
            };
            let remaining_expr = signed_term_expr(
                ctx,
                view.terms[remaining_index].0,
                view.terms[remaining_index].1,
            );

            if expr_matches_negation_after_default_simplify(ctx, rewritten, remaining_expr) {
                return Some(
                    Rewrite::with_local(ctx.num(0), description, focus_expr, rewritten).substep(
                        "Cancelar términos iguales",
                        vec![
                            "Tras aplicar la identidad, el término restante es el opuesto y toda la expresión se anula."
                                .to_string(),
                        ],
                    ),
                );
            }
        }
    }

    None
}

fn try_build_exact_trig_phase_shift_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    for first_index in 0..view.terms.len() {
        for second_index in (first_index + 1)..view.terms.len() {
            let focus_terms = [view.terms[first_index], view.terms[second_index]];
            let focus_expr = build_signed_sum_expr(ctx, &focus_terms);
            let remaining_terms: Vec<_> = view
                .terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| {
                    (index != first_index && index != second_index).then_some(term)
                })
                .collect();
            if remaining_terms.is_empty() {
                continue;
            }
            let remaining_expr = build_signed_sum_expr(ctx, &remaining_terms);

            if let Some(rewrite_match) =
                try_find_trig_phase_shift_cancellation_match(ctx, focus_expr, remaining_expr, true)
            {
                return Some(build_trig_phase_shift_zero_rewrite(ctx, rewrite_match));
            }
        }
    }

    None
}

fn try_build_exact_hyperbolic_angle_sum_diff_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return None;
    }

    for focus_index in 0..view.terms.len() {
        let (focus_expr, focus_sign) = view.terms[focus_index];
        if focus_sign != Sign::Pos {
            continue;
        }

        let Some(rewritten) =
            try_rewrite_hyperbolic_angle_sum_diff_for_cancellation(ctx, focus_expr)
        else {
            continue;
        };

        let remaining_terms: Vec<_> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, term)| (index != focus_index).then_some(term))
            .collect();
        let remaining_expr = build_signed_sum_expr(ctx, &remaining_terms);

        let neg_rewritten = ctx.add(Expr::Neg(rewritten));
        if expr_matches_negation_for_cancellation(ctx, rewritten, remaining_expr)
            || exprs_match_after_default_simplify(ctx, neg_rewritten, remaining_expr)
        {
            return Some(
                Rewrite::with_local(
                    ctx.num(0),
                    "Expand hyperbolic angle sum/difference",
                    focus_expr,
                    rewritten,
                )
                .substep(
                    "Cancelar términos iguales",
                    vec![
                        "Tras aplicar la identidad, el resto de la expresión es exactamente el opuesto y el resultado es 0."
                            .to_string(),
                    ],
                ),
            );
        }
    }

    None
}

fn try_build_exact_hyperbolic_pythagorean_factor_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    for first_index in 0..view.terms.len() {
        for second_index in (first_index + 1)..view.terms.len() {
            let focus_terms = [view.terms[first_index], view.terms[second_index]];
            let focus_expr = build_signed_sum_expr(ctx, &focus_terms);
            let Some(rewrite_match) =
                try_rewrite_hyperbolic_pythagorean_factor_for_cancellation(ctx, focus_expr)
            else {
                continue;
            };

            let remaining_terms: Vec<_> = view
                .terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| {
                    (index != first_index && index != second_index).then_some(term)
                })
                .collect();
            if remaining_terms.is_empty() {
                continue;
            }

            let remaining_expr = build_signed_sum_expr(ctx, &remaining_terms);
            if !expr_matches_negation_after_default_simplify(
                ctx,
                rewrite_match.local_after,
                remaining_expr,
            ) {
                continue;
            }

            return Some(build_hyperbolic_pythagorean_factor_zero_rewrite(
                ctx,
                rewrite_match,
            ));
        }
    }

    None
}

fn build_hyperbolic_pythagorean_factor_zero_rewrite(
    ctx: &mut cas_ast::Context,
    rewrite_match: HyperbolicPythagoreanFactorCancellationMatch,
) -> Rewrite {
    let focus_after_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: rewrite_match.local_after
        }
    );

    let rewrite = Rewrite::with_local(
        ctx.num(0),
        "Factor out cosh and apply hyperbolic Pythagorean identity",
        rewrite_match.local_before,
        rewrite_match.local_after,
    );

    match rewrite_match.mode {
        HyperbolicPythagoreanFactorCancellationMode::FactorThenRewrite {
            factorized,
            rewritten: _,
        } => {
            let factorized_display = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: factorized
                }
            );
            rewrite
                .substep(
                    "Sacar factor común",
                    vec![format!("Sacar factor común para obtener {factorized_display}.")],
                )
                .substep(
                    "Usar cosh(u)^2 - 1 = sinh(u)^2",
                    vec![format!("Así se obtiene {focus_after_display}.")],
                )
                .substep(
                    "Cancelar términos iguales",
                    vec![
                        "Tras la reescritura, el término restante es exactamente el opuesto y toda la expresión se anula."
                            .to_string(),
                    ],
                )
        }
        HyperbolicPythagoreanFactorCancellationMode::AlreadyFactored => rewrite
            .substep(
                "Usar sinh(u)^2 + 1 = cosh(u)^2",
                vec![format!("Así se obtiene {focus_after_display}.")],
            )
            .substep(
                "Cancelar términos iguales",
                vec![
                    "Tras la reescritura, el término restante es exactamente el opuesto y toda la expresión se anula."
                        .to_string(),
                ],
            ),
    }
}

fn build_hyperbolic_pythagorean_factor_root_zero_rewrite(
    ctx: &mut cas_ast::Context,
    whole_expr: cas_ast::ExprId,
    rewrite_match: HyperbolicPythagoreanFactorCancellationMatch,
) -> Rewrite {
    let focus_after_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: rewrite_match.local_after
        }
    );

    let rewrite = Rewrite::with_local(
        ctx.num(0),
        "Factor out cosh and apply hyperbolic Pythagorean identity",
        whole_expr,
        ctx.num(0),
    );

    match rewrite_match.mode {
        HyperbolicPythagoreanFactorCancellationMode::FactorThenRewrite {
            factorized,
            rewritten: _,
        } => {
            let factorized_display = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: factorized
                }
            );
            rewrite
                .substep(
                    "Sacar factor común",
                    vec![format!("Sacar factor común para obtener {factorized_display}.")],
                )
                .substep(
                    "Usar cosh(u)^2 - 1 = sinh(u)^2",
                    vec![format!("Así se obtiene {focus_after_display}.")],
                )
                .substep(
                    "Cancelar términos iguales",
                    vec![
                        "Tras la reescritura, el término restante es exactamente el opuesto y toda la expresión se anula."
                            .to_string(),
                    ],
                )
        }
        HyperbolicPythagoreanFactorCancellationMode::AlreadyFactored => rewrite
            .substep(
                "Usar sinh(u)^2 + 1 = cosh(u)^2",
                vec![format!("Así se obtiene {focus_after_display}.")],
            )
            .substep(
                "Cancelar términos iguales",
                vec![
                    "Tras la reescritura, el término restante es exactamente el opuesto y toda la expresión se anula."
                        .to_string(),
                ],
            ),
    }
}

define_rule!(
    ExpandTrigSumToProductToEnableCancellationRule,
    "Sum-to-Product Identity Cancellation Bridge",
    Some(crate::target_kind::TargetKindSet::ADD.union(crate::target_kind::TargetKindSet::SUB)),
    crate::phase::PhaseMask::POST,
    priority: 510,
    |ctx, expr| {
        if !maybe_trig_sum_to_product_zero_candidate(ctx, expr) {
            return None;
        }

        if let Some(rewrite) = try_build_exact_trig_sum_to_product_zero_scope_rewrite(ctx, expr) {
            return Some(rewrite);
        }

        match ctx.get(expr).clone() {
            Expr::Sub(lhs, rhs) => {
                if let Some((rewritten, description)) =
                    try_rewrite_trig_sum_to_product_for_cancellation(ctx, lhs)
                {
                    if exprs_match_after_default_simplify(ctx, rewritten, rhs) {
                        return Some(Rewrite::with_local(
                            ctx.add(Expr::Sub(rhs, rhs)),
                            description,
                            lhs,
                            rhs,
                        ));
                    }
                }

                if let Some((rewritten, description)) =
                    try_rewrite_trig_sum_to_product_for_cancellation(ctx, rhs)
                {
                    if exprs_match_after_default_simplify(ctx, rewritten, lhs) {
                        return Some(Rewrite::with_local(
                            ctx.add(Expr::Sub(lhs, lhs)),
                            description,
                            rhs,
                            lhs,
                        ));
                    }
                }

                None
            }
            Expr::Add(lhs, rhs) => {
                if let Some((rewritten, description)) =
                    try_rewrite_trig_sum_to_product_for_cancellation(ctx, lhs)
                {
                    if expr_matches_negation_after_default_simplify(ctx, rewritten, rhs) {
                        return Some(Rewrite::with_local(
                            ctx.add(Expr::Add(rewritten, rhs)),
                            description,
                            lhs,
                            rewritten,
                        ));
                    }
                }

                if let Some((rewritten, description)) =
                    try_rewrite_trig_sum_to_product_for_cancellation(ctx, rhs)
                {
                    if expr_matches_negation_after_default_simplify(ctx, rewritten, lhs) {
                        return Some(Rewrite::with_local(
                            ctx.add(Expr::Add(lhs, rewritten)),
                            description,
                            rhs,
                            rewritten,
                        ));
                    }
                }

                None
            }
            _ => None,
        }
    }
);

define_rule!(
    ExpandTrigSineProductTripleAngleToEnableCancellationRule,
    "Product-to-Sum and Triple-Angle Identity",
    Some(crate::target_kind::TargetKindSet::ADD.union(crate::target_kind::TargetKindSet::SUB)),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    priority: 510,
    |ctx, expr| {
        if !maybe_trig_square_zero_candidate(ctx, expr) {
            return None;
        }

        try_build_exact_trig_sine_product_triple_angle_zero_scope_rewrite(ctx, expr)
    }
);

define_rule!(
    ExpandTrigSquareIdentityToEnableCancellationRule,
    "Trig Square Identity",
    Some(crate::target_kind::TargetKindSet::ADD.union(crate::target_kind::TargetKindSet::SUB)),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    priority: 510,
    |ctx, expr| {
        if !maybe_trig_square_zero_candidate(ctx, expr) {
            return None;
        }

        try_build_exact_trig_square_zero_scope_rewrite(ctx, expr)
    }
);

define_rule!(
    ExpandTrigPhaseShiftToEnableCancellationRule,
    "Phase Shift Identity",
    Some(crate::target_kind::TargetKindSet::ADD.union(crate::target_kind::TargetKindSet::SUB)),
    crate::phase::PhaseMask::POST,
    priority: 510,
    |ctx, expr| {
        if !maybe_trig_phase_shift_zero_candidate(ctx, expr) {
            return None;
        }

        if let Some(rewrite) = try_build_exact_trig_phase_shift_zero_scope_rewrite(ctx, expr) {
            return Some(rewrite);
        }

        match ctx.get(expr).clone() {
            Expr::Sub(lhs, rhs) => {
                let rewrite_match =
                    try_find_trig_phase_shift_cancellation_match(ctx, lhs, rhs, false)
                        .or_else(|| try_find_trig_phase_shift_cancellation_match(ctx, rhs, lhs, false))?;
                Some(build_trig_phase_shift_zero_rewrite(ctx, rewrite_match))
            }
            Expr::Add(lhs, rhs) => {
                let rewrite_match =
                    try_find_trig_phase_shift_cancellation_match(ctx, lhs, rhs, true)
                        .or_else(|| try_find_trig_phase_shift_cancellation_match(ctx, rhs, lhs, true))?;
                Some(build_trig_phase_shift_zero_rewrite(ctx, rewrite_match))
            }
            _ => None,
        }
    }
);

define_rule!(
    ExpandHyperbolicPythagoreanFactorToEnableCancellationRule,
    "Hyperbolic Pythagorean Identity Cancellation Bridge",
    Some(crate::target_kind::TargetKindSet::ADD.union(crate::target_kind::TargetKindSet::SUB)),
    crate::phase::PhaseMask::CORE
        | crate::phase::PhaseMask::TRANSFORM
        | crate::phase::PhaseMask::POST,
    priority: 510,
    |ctx, expr, parent_ctx| {
        if !maybe_hyperbolic_pythagorean_factor_zero_candidate(ctx, expr) {
            return None;
        }

        if let Some(rewrite) =
            try_build_exact_hyperbolic_pythagorean_factor_zero_scope_rewrite(ctx, expr)
        {
            return Some(rewrite);
        }

        if parent_ctx.depth() == 0 {
            if let Some(rewrite_match) =
                try_rewrite_hyperbolic_pythagorean_factor_for_cancellation(ctx, expr)
            {
                return Some(build_hyperbolic_pythagorean_factor_root_zero_rewrite(
                    ctx,
                    expr,
                    rewrite_match,
                ));
            }
        }

        None
    }
);

define_rule!(
    ExpandHyperbolicAngleSumDiffToEnableCancellationRule,
    "Hyperbolic Angle Sum/Difference Identity",
    Some(crate::target_kind::TargetKindSet::ADD.union(crate::target_kind::TargetKindSet::SUB)),
    crate::phase::PhaseMask::POST,
    priority: 510,
    |ctx, expr| {
        if !maybe_hyperbolic_angle_sum_diff_zero_candidate(ctx, expr) {
            return None;
        }

        if let Some(rewrite) =
            try_build_exact_hyperbolic_angle_sum_diff_zero_scope_rewrite(ctx, expr)
        {
            return Some(rewrite);
        }

        match ctx.get(expr).clone() {
            Expr::Sub(lhs, rhs) => {
                if let Some(rewritten) =
                    try_rewrite_hyperbolic_angle_sum_diff_for_cancellation(ctx, lhs)
                {
                    if exprs_match_after_default_simplify(ctx, rewritten, rhs) {
                        return Some(Rewrite::with_local(
                            ctx.add(Expr::Sub(rhs, rhs)),
                            "Expand hyperbolic angle sum/difference",
                            lhs,
                            rhs,
                        ));
                    }
                }

                if let Some(rewritten) =
                    try_rewrite_hyperbolic_angle_sum_diff_for_cancellation(ctx, rhs)
                {
                    if exprs_match_after_default_simplify(ctx, rewritten, lhs) {
                        return Some(Rewrite::with_local(
                            ctx.add(Expr::Sub(lhs, lhs)),
                            "Expand hyperbolic angle sum/difference",
                            rhs,
                            lhs,
                        ));
                    }
                }

                None
            }
            Expr::Add(lhs, rhs) => {
                if let Some(rewritten) =
                    try_rewrite_hyperbolic_angle_sum_diff_for_cancellation(ctx, lhs)
                {
                    if expr_matches_negation_for_cancellation(ctx, rewritten, rhs) {
                        return Some(Rewrite::with_local(
                            ctx.add(Expr::Add(rewritten, rhs)),
                            "Expand hyperbolic angle sum/difference",
                            lhs,
                            rewritten,
                        ));
                    }
                }

                if let Some(rewritten) =
                    try_rewrite_hyperbolic_angle_sum_diff_for_cancellation(ctx, rhs)
                {
                    if expr_matches_negation_for_cancellation(ctx, rewritten, lhs) {
                        return Some(Rewrite::with_local(
                            ctx.add(Expr::Add(lhs, rewritten)),
                            "Expand hyperbolic angle sum/difference",
                            rhs,
                            rewritten,
                        ));
                    }
                }

                None
            }
            _ => None,
        }
    }
);

define_rule!(
    AddZeroRule,
    "Identity Property of Addition",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_add_zero_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.description))
    }
);

define_rule!(
    MulOneRule,
    "Identity Property of Multiplication",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_mul_one_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.description))
    }
);

// MulZeroRule: 0*e → 0
// Domain Mode Policy: 0*e → 0 changes the domain of definition if e can be undefined.
// Uses ConditionClass taxonomy:
// - Strict: only apply if other factor has no undefined risk
// - Generic: apply with Defined(e) assumption (Definability class)
// - Assume: apply with Defined(e) assumption
define_rule!(
    MulZeroRule,
    "Zero Property of Multiplication",
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        let pattern = match_mul_zero_pattern(ctx, expr)?;
        let other = pattern.other;
        let has_risk = crate::collect::has_undefined_risk(ctx, other);
        let allowed = cas_solver_core::undefined_risk_policy_support::allow_cancellation_with_undefined_risk_mode_flags(
            matches!(parent_ctx.domain_mode(), crate::DomainMode::Assume),
            matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict),
            has_risk,
        );

        if !allowed {
            return None; // Strict mode: don't simplify if has risk
        }

        // Build assumption events if has risk and allowed
        let assumption_events: smallvec::SmallVec<[crate::AssumptionEvent; 1]> = if has_risk {
            smallvec::smallvec![crate::AssumptionEvent::defined(ctx, other)]
        } else {
            smallvec::SmallVec::new()
        };

        let description = if pattern.zero_on_lhs {
            "0 * x = 0".to_string()
        } else {
            "x * 0 = 0".to_string()
        };


        let zero = ctx.num(0);
        Some(Rewrite::new(zero).desc(description).assume_all(assumption_events))
    }
);

// DivZeroRule: 0/d → 0
// Domain Mode Policy: 0/d → 0 changes the domain of definition if d can be 0.
// Uses unified DomainOracle via oracle_allows_with_hint:
// - Strict: only apply if prove_nonzero(d) == Proven
// - Generic: apply with NonZero(d) assumption (Definability class)
// - Assume: apply with NonZero(d) assumption
define_rule!(
    DivZeroRule,
    "Zero Property of Division",
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::Proof;
        use crate::Predicate;

        let pattern = match_div_zero_numerator_pattern(ctx, expr)?;
        let den = pattern.denominator;

        // Special case: 0/0 → undefined (all modes)
        if pattern.denominator_is_literal_zero {
            let undef = ctx.add(Expr::Constant(cas_ast::Constant::Undefined));
            return Some(Rewrite::new(undef).desc("0/0 is undefined"));
        }

        // Use unified oracle for NonZero condition (Definability class)
        let decision = crate::oracle_allows_with_hint(
            ctx,
            parent_ctx.domain_mode(),
            parent_ctx.value_domain(),
            &Predicate::NonZero(den),
            "Zero Property of Division",
        );

        if !decision.allow {
            return None; // Strict mode: don't simplify if not proven
        }

        // Build assumption events if needed
        let den_proof = crate::helpers::prove_nonzero(ctx, den);
        let assumption_events: smallvec::SmallVec<[crate::AssumptionEvent; 1]> = if decision.assumption.is_some() && den_proof != Proof::Proven {
            smallvec::smallvec![crate::AssumptionEvent::nonzero(ctx, den)]
        } else {
            smallvec::SmallVec::new()
        };

        let zero = ctx.num(0);
        Some(Rewrite::new(zero).desc("0 / d = 0").assume_all(assumption_events))
    }
);

define_rule!(
    CombineConstantsRule,
    "Combine Constants",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_combine_constants_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.description))
    }
);

// =============================================================================
// SubSelfToZeroRule: a - a = 0 (Short-circuit)
// =============================================================================
//
// V2.14.45: This rule MUST fire before expansion rules like TanToSinCosRule.
// Without this, tan(3x) - tan(3x) would expand both tans and fail to cancel.
// Uses priority 500 to ensure it runs first.
//
// Domain Policy: Same as AddInverseRule - check for undefined subexpressions.
// Uses compare_expr for structural equality (handles tan(3x) == tan(3·x)).
// =============================================================================
define_rule!(
    ExpandOddHalfPowerToEnableCancellationRule,
    "Expand Odd Half Power",
    Some(crate::target_kind::TargetKindSet::ADD.union(crate::target_kind::TargetKindSet::SUB)),
    crate::phase::PhaseMask::POST,
    priority: 510,
    |ctx, expr| {
        if !maybe_odd_half_power_zero_candidate(ctx, expr) {
            return None;
        }

        let (lhs, rhs, was_add_with_neg) = match ctx.get(expr) {
            Expr::Sub(lhs, rhs) => (*lhs, *rhs, false),
            Expr::Add(lhs, rhs) => match ctx.get(*rhs) {
                Expr::Neg(inner) => (*lhs, *inner, true),
                _ => return None,
            },
            _ => return None,
        };

        let candidate = if let Some(matched) =
            try_match_odd_half_power_cancellation_side(ctx, lhs, rhs)
        {
            let new_expr = rebuild_subtractive_expr(ctx, matched.rewritten_expr, rhs, was_add_with_neg);
            let mut rewrite = Rewrite::with_local(
                new_expr,
                "Rewrite an odd half-integer power using a square root",
                matched.focus_before,
                matched.focus_after,
            );
            if let Some(base) = matched.base {
                rewrite = rewrite.requires(crate::ImplicitCondition::NonNegative(base));
            }
            Some(rewrite)
        } else if let Some(matched) =
            try_match_odd_half_power_cancellation_side(ctx, rhs, lhs)
        {
            let new_expr = rebuild_subtractive_expr(ctx, lhs, matched.rewritten_expr, was_add_with_neg);
            let mut rewrite = Rewrite::with_local(
                new_expr,
                "Rewrite an odd half-integer power using a square root",
                matched.focus_before,
                matched.focus_after,
            );
            if let Some(base) = matched.base {
                rewrite = rewrite.requires(crate::ImplicitCondition::NonNegative(base));
            }
            Some(rewrite)
        } else {
            None
        }?;

        Some(candidate)
    }
);

define_rule!(
    ExpandLogProductPowerToEnableCancellationRule,
    "Expand Log Product Power",
    Some(crate::target_kind::TargetKindSet::ADD_SUB),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    priority: 511,
    |ctx, expr| {
        let view = AddView::from_expr(ctx, expr);
        if view.terms.len() < 3 {
            return None;
        }

        for (focus_index, (raw_focus_expr, raw_focus_sign)) in view.terms.iter().copied().enumerate() {
            let (focus_expr, focus_sign) =
                normalize_signed_add_term(ctx, raw_focus_expr, raw_focus_sign);
            let Some(matched) = try_match_log_product_power_cancellation_side(ctx, focus_expr) else {
                continue;
            };

            let mut used_indices = Vec::new();
            let mut all_components_found = true;
            for (component_expr, component_sign) in &matched.components {
                let expected_sign = if focus_sign == Sign::Pos {
                    component_sign.negate()
                } else {
                    *component_sign
                };

                let mut found_index = None;
                for (term_index, (term_expr, term_sign)) in view.terms.iter().copied().enumerate() {
                    if term_index == focus_index || used_indices.contains(&term_index) {
                        continue;
                    }
                    let (normalized_term_expr, normalized_term_sign) =
                        normalize_signed_add_term(ctx, term_expr, term_sign);
                    if normalized_term_sign != expected_sign {
                        continue;
                    }
                    let comparable_term_expr =
                        normalize_log_term_expr_for_cancellation(ctx, normalized_term_expr)
                            .unwrap_or(normalized_term_expr);
                    if compare_expr(ctx, comparable_term_expr, *component_expr) == Ordering::Equal {
                        found_index = Some(term_index);
                        break;
                    }
                }

                if let Some(term_index) = found_index {
                    used_indices.push(term_index);
                } else {
                    all_components_found = false;
                    break;
                }
            }

            if !all_components_found || used_indices.len() + 1 != view.terms.len() {
                continue;
            }

            let zero = ctx.num(0);
            let focus_before_display = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: raw_focus_expr
                }
            );
            let raw_focus_after_display = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: matched.raw_focus_after
                }
            );
            let focus_after_display = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: matched.focus_after
                }
            );

            let mut rewrite = Rewrite::with_local(
                zero,
                "Log expansion followed by exact cancellation",
                expr,
                zero,
            )
            .substep(
                "Expandir el logaritmo del producto o del cociente",
                vec![format!(
                    "Reescribir {focus_before_display} como {raw_focus_after_display}."
                )],
            );

            if matched.needs_power_split {
                rewrite = rewrite.substep(
                    "Sacar exponentes fuera del logaritmo cuando sea necesario",
                    vec![format!(
                        "Reescribir {raw_focus_after_display} como {focus_after_display}."
                    )],
                );
            }

            return Some(rewrite.substep(
                "Cancelar términos iguales",
                vec![
                    "Tras la expansión, los términos opuestos se anulan y el resultado es 0."
                        .to_string(),
                ],
            ));
        }

        None
    }
);

define_rule!(
    ExpandLogAbsMulDivToEnableCancellationRule,
    "Expand Log Abs Mul/Div",
    Some(crate::target_kind::TargetKindSet::ADD_SUB),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    priority: 510,
    |ctx, expr| {
        let view = AddView::from_expr(ctx, expr);
        if view.terms.len() < 3 {
            return None;
        }

        for (focus_index, (raw_focus_expr, raw_focus_sign)) in view.terms.iter().copied().enumerate() {
            let (focus_expr, focus_sign) =
                normalize_signed_add_term(ctx, raw_focus_expr, raw_focus_sign);
            let Some(matched) = try_match_log_abs_mul_div_cancellation_side(ctx, focus_expr) else {
                continue;
            };

            let mut used_indices = Vec::new();
            let mut all_components_found = true;
            for (component_expr, component_sign) in matched.components {
                let expected_sign = if focus_sign == Sign::Pos {
                    component_sign.negate()
                } else {
                    component_sign
                };

                let mut found_index = None;
                for (term_index, (term_expr, term_sign)) in view.terms.iter().copied().enumerate() {
                    if term_index == focus_index || used_indices.contains(&term_index) {
                        continue;
                    }
                    let (normalized_term_expr, normalized_term_sign) =
                        normalize_signed_add_term(ctx, term_expr, term_sign);
                    if normalized_term_sign != expected_sign {
                        continue;
                    }
                    if compare_expr(ctx, normalized_term_expr, component_expr) == Ordering::Equal {
                        found_index = Some(term_index);
                        break;
                    }
                }

                if let Some(term_index) = found_index {
                    used_indices.push(term_index);
                } else {
                    all_components_found = false;
                    break;
                }
            }

            if !all_components_found {
                continue;
            }

            let exact_identity_scope = used_indices.len() + 1 == view.terms.len();

            if exact_identity_scope {
                let zero = ctx.num(0);
                let focus_before_display = format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: ctx,
                        id: raw_focus_expr
                    }
                );
                let focus_after_display = format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: ctx,
                        id: matched.focus_after
                    }
                );

                return Some(
                    Rewrite::with_local(
                        zero,
                        "Log expansion followed by exact cancellation",
                        expr,
                        zero,
                    )
                    .substep(
                        "Expandir el logaritmo del producto o del cociente",
                        vec![format!(
                            "Reescribir {focus_before_display} como {focus_after_display}."
                        )],
                    )
                    .substep(
                        "Cancelar términos iguales",
                        vec![
                            "Tras la expansión, los términos opuestos se anulan y el resultado es 0."
                                .to_string(),
                        ],
                    ),
                );
            }

            let mut rebuilt_terms = smallvec::SmallVec::<[(cas_ast::ExprId, Sign); 8]>::new();
            for (term_index, (term_expr, term_sign)) in view.terms.iter().copied().enumerate() {
                if term_index == focus_index {
                    for (component_expr, component_sign) in matched.components {
                        let global_sign = if focus_sign == Sign::Pos {
                            component_sign
                        } else {
                            component_sign.negate()
                        };
                        rebuilt_terms.push((component_expr, global_sign));
                    }
                    continue;
                }
                rebuilt_terms.push((term_expr, term_sign));
            }

            let new_expr = AddView {
                root: expr,
                terms: rebuilt_terms,
            }
            .rebuild(ctx);
            let focus_after_local = if focus_sign == Sign::Pos {
                matched.focus_after
            } else {
                ctx.add(Expr::Neg(matched.focus_after))
            };

            return Some(Rewrite::with_local(
                new_expr,
                "Log expansion",
                raw_focus_expr,
                focus_after_local,
            ));
        }

        None
    }
);

define_rule!(
    SubSelfToZeroRule,
    "Subtraction Self-Cancel",
    priority: 500, // High priority: before any expansion rules
    |ctx, expr, parent_ctx| {
        let rewrite = try_rewrite_sub_self_zero_expr(ctx, expr)?;
        let allow =
            cas_solver_core::undefined_risk_policy_support::allow_cancellation_with_undefined_risk_mode_flags(
                matches!(parent_ctx.domain_mode(), crate::DomainMode::Assume),
                matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict),
                crate::collect::has_undefined_risk(ctx, rewrite.inner),
            );
        if !allow {
            return None;
        }

        Some(Rewrite::new(rewrite.rewritten).desc("a - a = 0"))
    }
);

define_rule!(
    SubtractExpandedSumDiffCubesQuotientRule,
    "Subtract Expanded Sum/Difference of Cubes Quotient",
    priority: 500,
    |ctx, expr, parent_ctx| {
        use crate::{ImplicitCondition, Predicate};

        let (lhs, rhs) = match ctx.get(expr) {
            Expr::Sub(lhs, rhs) => (*lhs, *rhs),
            Expr::Add(lhs, rhs) => match (ctx.get(*lhs), ctx.get(*rhs)) {
                (_, Expr::Neg(inner)) => (*lhs, *inner),
                (Expr::Neg(inner), _) => (*rhs, *inner),
                _ => return None,
            },
            _ => return None,
        };

        let (num, den) = match ctx.get(lhs) {
            Expr::Div(num, den) => (*num, *den),
            _ => return None,
        };

        let decision = crate::oracle_allows_with_hint(
            ctx,
            parent_ctx.domain_mode(),
            parent_ctx.value_domain(),
            &Predicate::NonZero(den),
            "Subtract Expanded Sum/Difference of Cubes Quotient",
        );
        if !decision.allow {
            return None;
        }

        let plan = crate::rules::algebra::fractions::try_plan_sum_diff_of_cubes_in_num(
            ctx, num, den, false,
        )?;

        let cancelled = canonicalize_nested_integer_powers(ctx, plan.cancelled_result);
        let rhs = canonicalize_nested_integer_powers(ctx, rhs);
        if !(cas_math::expr_domain::exprs_equivalent(ctx, cancelled, rhs)
            || exprs_equal_up_to_add_term_order(ctx, cancelled, rhs))
        {
            return None;
        }

        Some(
            Rewrite::new(ctx.num(0))
                .desc("((a^3 ± b^3)/(a ± b)) - expanded quotient = 0")
                .requires(ImplicitCondition::NonZero(den)),
        )
    }
);

// AddInverseRule: a + (-a) = 0
// Domain Mode Policy: Like other cancellation rules, we must respect domain_mode
// because if `a` can be undefined (e.g., x/(x+1) when x=-1), then a + (-a)
// is undefined, not 0.
// - Strict: only if `a` contains no potentially-undefined subexpressions (no variable denominator)
// - Assume: always apply (educational mode assumption: all expressions are defined)
// - Generic: same as Assume
//
// V2.12.13: REMOVED redundant "is defined" assumption event.
// The individual Div operations already emit NonZero(denominator) as Requires.
// Showing "a is defined" here is redundant and confusing.
define_rule!(AddInverseRule, "Add Inverse", |ctx, expr, parent_ctx| {
    let rewrite = try_rewrite_add_inverse_zero_expr(ctx, expr)?;
    let allow =
        cas_solver_core::undefined_risk_policy_support::allow_cancellation_with_undefined_risk_mode_flags(
            matches!(parent_ctx.domain_mode(), crate::DomainMode::Assume),
            matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict),
            crate::collect::has_undefined_risk(ctx, rewrite.inner),
        );
    if !allow {
        return None;
    }

    // V2.12.13: No assumption events - the division conditions are already
    // tracked as Requires from the original Div operations.
    // Adding "a is defined" here is redundant and clutters the output.
    Some(Rewrite::new(rewrite.rewritten).desc("a + (-a) = 0"))
});

#[cfg(test)]
mod tests {
    use super::{
        canonicalize_nested_integer_powers, exprs_equal_up_to_add_term_order,
        extract_scaled_double_sine_product_for_cancellation,
        ExpandHyperbolicAngleSumDiffToEnableCancellationRule,
        ExpandHyperbolicPythagoreanFactorToEnableCancellationRule,
        ExpandLogAbsMulDivToEnableCancellationRule, ExpandLogProductPowerToEnableCancellationRule,
        ExpandOddHalfPowerToEnableCancellationRule, ExpandTrigPhaseShiftToEnableCancellationRule,
        ExpandTrigSineProductTripleAngleToEnableCancellationRule,
        ExpandTrigSquareIdentityToEnableCancellationRule,
        ExpandTrigSumToProductToEnableCancellationRule, SubSelfToZeroRule,
        SubtractExpandedSumDiffCubesQuotientRule,
    };
    use crate::parent_context::ParentContext;
    use crate::rule::Rule;
    use crate::DomainMode;
    use cas_ast::{Context, Expr};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    #[test]
    fn subtraction_self_cancel_rule_matches_abs_sub_mirror_in_generic() {
        let mut ctx = Context::new();
        let expr = parse(
            "abs((2*u)/(u^2 - 1) - 1) - abs(1 - 2*u/(u^2 - 1))",
            &mut ctx,
        )
        .unwrap_or_else(|err| panic!("parse: {err}"));

        let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
        let rule = SubSelfToZeroRule;
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
    }

    #[test]
    fn expand_odd_half_power_to_enable_cancellation_rule_matches_nonnegative_target() {
        let mut ctx = Context::new();
        let expr =
            parse("sqrt(x^5) - x^2*sqrt(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
        let expected = parse("x^2*sqrt(x) - x^2*sqrt(x)", &mut ctx)
            .unwrap_or_else(|err| panic!("parse expected: {err}"));

        let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
        let rule = ExpandOddHalfPowerToEnableCancellationRule;
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: expected
                }
            )
        );
        assert_eq!(rewrite.required_conditions.len(), 1);
    }

    #[test]
    fn expand_odd_half_power_to_enable_cancellation_rule_matches_reversed_side() {
        let mut ctx = Context::new();
        let expr =
            parse("x^3*sqrt(x) - sqrt(x^7)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
        let expected = parse("x^3*sqrt(x) - x^3*sqrt(x)", &mut ctx)
            .unwrap_or_else(|err| panic!("parse expected: {err}"));

        let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
        let rule = ExpandOddHalfPowerToEnableCancellationRule;
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: expected
                }
            )
        );
        assert_eq!(rewrite.required_conditions.len(), 1);
    }

    #[test]
    fn expand_log_abs_mul_div_to_enable_cancellation_rule_matches_scaled_ln_product() {
        let mut ctx = Context::new();
        let expr = parse("2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y))", &mut ctx)
            .unwrap_or_else(|err| panic!("parse: {err}"));

        let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
        let rule = ExpandLogAbsMulDivToEnableCancellationRule;
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
        assert_eq!(rewrite.substeps.len(), 2);
    }

    #[test]
    fn expand_log_abs_mul_div_to_enable_cancellation_rule_matches_scaled_log_product() {
        let mut ctx = Context::new();
        let expr = parse("2*log(abs(x*y)) - 2*log(abs(x)) - 2*log(abs(y))", &mut ctx)
            .unwrap_or_else(|err| panic!("parse: {err}"));

        let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
        let rule = ExpandLogAbsMulDivToEnableCancellationRule;
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
        assert_eq!(rewrite.substeps.len(), 2);
    }

    #[test]
    fn expand_log_product_power_to_enable_cancellation_rule_matches_mixed_power_product() {
        let mut ctx = Context::new();
        let expr = parse("ln(x^3) + ln(y^2) - ln(x^3 * y^2)", &mut ctx)
            .unwrap_or_else(|err| panic!("parse: {err}"));

        let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
        let rule = ExpandLogProductPowerToEnableCancellationRule;
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
        assert_eq!(rewrite.substeps.len(), 3);
    }

    #[test]
    fn expand_trig_sum_to_product_to_enable_cancellation_rule_matches_symbolic_sine_sum() {
        let mut ctx = Context::new();
        let expr = parse("sin(x) + sin(y) - 2*sin((x+y)/2)*cos((x-y)/2)", &mut ctx)
            .unwrap_or_else(|err| panic!("parse: {err}"));

        let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
        let rule = ExpandTrigSumToProductToEnableCancellationRule;
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
        assert_eq!(rewrite.description, "Expand sine sum to product");
    }

    #[test]
    fn expand_trig_sum_to_product_to_enable_cancellation_rule_matches_symbolic_cosine_difference() {
        let mut ctx = Context::new();
        let expr = parse("cos(x) - cos(y) + 2*sin((x+y)/2)*sin((x-y)/2)", &mut ctx)
            .unwrap_or_else(|err| panic!("parse: {err}"));

        let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
        let rule = ExpandTrigSumToProductToEnableCancellationRule;
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
        assert_eq!(rewrite.description, "Expand cosine difference to product");
    }

    #[test]
    fn expand_trig_phase_shift_to_enable_cancellation_rule_matches_general_shifted_sine() {
        let mut ctx = Context::new();
        let expr = parse("3*sin(x) + 4*cos(x) - 5*sin(x + arctan(4/3))", &mut ctx)
            .unwrap_or_else(|err| panic!("parse: {err}"));

        let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
        let rule = ExpandTrigPhaseShiftToEnableCancellationRule;
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
        assert_eq!(rewrite.description, "Phase Shift Identity");
        assert_eq!(rewrite.substeps.len(), 2);
    }

    #[test]
    fn expand_trig_square_identity_to_enable_cancellation_rule_matches_binomial_square() {
        let mut ctx = Context::new();
        let expr = parse("(sin(x) + cos(x))^2 - (1 + sin(2*x))", &mut ctx)
            .unwrap_or_else(|err| panic!("parse: {err}"));

        let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
        let rule = ExpandTrigSquareIdentityToEnableCancellationRule;
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
        assert_eq!(rewrite.description, "Trig Square Identity");
        assert_eq!(rewrite.substeps.len(), 4);
    }

    #[test]
    fn extract_scaled_double_sine_product_for_cancellation_matches_canonical_order() {
        let mut ctx = Context::new();
        let expr =
            parse("2*sin(2*x)*sin(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

        let (scale, arg) = extract_scaled_double_sine_product_for_cancellation(&mut ctx, expr)
            .unwrap_or_else(|| panic!("extract"));

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: scale
                }
            ),
            "1"
        );
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: arg
                }
            ),
            "x"
        );
    }

    #[test]
    fn expand_trig_sine_product_triple_angle_to_enable_cancellation_rule_matches_cubic_cosine_residual(
    ) {
        let mut ctx = Context::new();
        let expr = parse("2*sin(2*x)*sin(x) - (4*cos(x) - 4*cos(x)^3)", &mut ctx)
            .unwrap_or_else(|err| panic!("parse: {err}"));

        let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
        let rule = ExpandTrigSineProductTripleAngleToEnableCancellationRule;
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
        assert_eq!(
            rewrite.description,
            "Product-to-Sum and Triple-Angle Identity"
        );
        assert_eq!(rewrite.substeps.len(), 4);
    }

    #[test]
    fn expand_hyperbolic_angle_sum_diff_to_enable_cancellation_rule_matches_sinh_sum() {
        let mut ctx = Context::new();
        let expr = parse("sinh(x+y) - (sinh(x)*cosh(y) + cosh(x)*sinh(y))", &mut ctx)
            .unwrap_or_else(|err| panic!("parse: {err}"));

        let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
        let rule = ExpandHyperbolicAngleSumDiffToEnableCancellationRule;
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
        assert_eq!(
            rewrite.description,
            "Expand hyperbolic angle sum/difference"
        );
    }

    #[test]
    fn expand_hyperbolic_pythagorean_factor_to_enable_cancellation_rule_matches_cubic_residual() {
        let mut ctx = Context::new();
        let expr = parse("4*cosh(x)*sinh(x)^2 + 4*cosh(x) - 4*cosh(x)^3", &mut ctx)
            .unwrap_or_else(|err| panic!("parse: {err}"));

        let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
        let rule = ExpandHyperbolicPythagoreanFactorToEnableCancellationRule;
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
        assert_eq!(rewrite.substeps.len(), 3);
    }

    #[test]
    fn expand_hyperbolic_pythagorean_factor_to_enable_cancellation_rule_matches_symbolic_coefficient(
    ) {
        let mut ctx = Context::new();
        let expr = parse(
            "4*a*cosh(x)*sinh(x)^2 + 4*a*cosh(x) - 4*a*cosh(x)^3",
            &mut ctx,
        )
        .unwrap_or_else(|err| panic!("parse: {err}"));

        let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
        let rule = ExpandHyperbolicPythagoreanFactorToEnableCancellationRule;
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
        assert_eq!(rewrite.substeps.len(), 3);
    }

    #[test]
    fn subtract_expanded_sum_diff_cubes_quotient_rule_matches_trig_square_cube_residual() {
        let mut ctx = Context::new();
        let expr = parse(
            "((sin(u)^2)^3 - 1)/((sin(u)^2) - 1) - ((sin(u)^2)^2 + (sin(u)^2) + 1)",
            &mut ctx,
        )
        .unwrap_or_else(|err| panic!("parse: {err}"));

        let (lhs, rhs) = match ctx.get(expr) {
            Expr::Sub(lhs, rhs) => (*lhs, *rhs),
            Expr::Add(lhs, rhs) => match (ctx.get(*lhs), ctx.get(*rhs)) {
                (_, Expr::Neg(inner)) => (*lhs, *inner),
                (Expr::Neg(inner), _) => (*rhs, *inner),
                _ => panic!("unexpected add form"),
            },
            other => panic!("unexpected root: {other:?}"),
        };
        let (num, den) = match ctx.get(lhs) {
            Expr::Div(num, den) => (*num, *den),
            other => panic!("unexpected lhs: {other:?}"),
        };
        let plan = crate::rules::algebra::fractions::try_plan_sum_diff_of_cubes_in_num(
            &mut ctx, num, den, false,
        )
        .unwrap_or_else(|| panic!("plan"));
        let cancelled = canonicalize_nested_integer_powers(&mut ctx, plan.cancelled_result);
        let rhs = canonicalize_nested_integer_powers(&mut ctx, rhs);
        assert!(
            cas_math::expr_domain::exprs_equivalent(&ctx, cancelled, rhs)
                || exprs_equal_up_to_add_term_order(&ctx, cancelled, rhs),
            "cancelled={} rhs={}",
            DisplayExpr {
                context: &ctx,
                id: cancelled
            },
            DisplayExpr {
                context: &ctx,
                id: rhs
            }
        );

        let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
        let rule = SubtractExpandedSumDiffCubesQuotientRule;
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
    }
}

// Simplify sums of fractions in exponents: x^(1/2 + 1/3) → x^(5/6)
// This makes the fraction sum visible as a step in the timeline.
define_rule!(
    SimplifyNumericExponentsRule,
    "Sum Exponents",
    |ctx, expr| {
        let rewrite = try_rewrite_simplify_numeric_exponents_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.description))
    }
);

// =============================================================================
// NormalizeMulNegRule: Lift Neg out of Mul for canonical form
// =============================================================================
//
// Canonical form: Neg should be at the TOP of Mul, not buried inside.
// This unlocks cancellations in Add like: a*(-b) + (-a)*b → Neg(a*b) + Neg(a*b) → -2*a*b
//
// Rewrites:
// - Mul(Neg(a), b) → Neg(Mul(a, b))
// - Mul(a, Neg(b)) → Neg(Mul(a, b))
// - Mul(Neg(a), Neg(b)) → Mul(a, b)  (double negation cancels)
//
// This is idempotent and always reduces complexity.
// =============================================================================
define_rule!(
    NormalizeMulNegRule,
    "Normalize Negation in Product",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_normalize_mul_neg_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.description))
    }
);

pub fn register(simplifier: &mut crate::Simplifier) {
    // High-priority short-circuit rules first
    simplifier.add_rule(Box::new(
        ExpandTrigSineProductTripleAngleToEnableCancellationRule,
    ));
    simplifier.add_rule(Box::new(ExpandTrigSquareIdentityToEnableCancellationRule));
    simplifier.add_rule(Box::new(ExpandTrigPhaseShiftToEnableCancellationRule));
    simplifier.add_rule(Box::new(ExpandTrigSumToProductToEnableCancellationRule));
    simplifier.add_rule(Box::new(
        ExpandHyperbolicPythagoreanFactorToEnableCancellationRule,
    ));
    simplifier.add_rule(Box::new(
        ExpandHyperbolicAngleSumDiffToEnableCancellationRule,
    ));
    simplifier.add_rule(Box::new(ExpandOddHalfPowerToEnableCancellationRule));
    simplifier.add_rule(Box::new(ExpandLogProductPowerToEnableCancellationRule));
    simplifier.add_rule(Box::new(ExpandLogAbsMulDivToEnableCancellationRule));
    simplifier.add_rule(Box::new(SubSelfToZeroRule)); // priority 500: before expansion
    simplifier.add_rule(Box::new(SubtractExpandedSumDiffCubesQuotientRule));

    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));
    simplifier.add_rule(Box::new(NormalizeMulNegRule)); // Lift Neg out of Mul for canonical form
    simplifier.add_rule(Box::new(MulZeroRule));
    simplifier.add_rule(Box::new(DivZeroRule));
    simplifier.add_rule(Box::new(CombineConstantsRule));
    simplifier.add_rule(Box::new(SimplifyNumericExponentsRule));
    simplifier.add_rule(Box::new(AddInverseRule));
}
