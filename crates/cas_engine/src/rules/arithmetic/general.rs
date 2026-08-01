//! `arithmetic`: familia `general`.
//!
//! Ver la cabecera de `arithmetic.rs` para el contexto.

use super::*;

pub(super) fn additive_view_has_exact_duplicate_or_canceling_terms(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 3 {
        return false;
    }

    for (idx, (left_expr, _left_sign)) in view.terms.iter().enumerate() {
        if view
            .terms
            .iter()
            .skip(idx + 1)
            .any(|(right_expr, _)| compare_expr(ctx, *left_expr, *right_expr) == Ordering::Equal)
        {
            return true;
        }
    }

    false
}

pub(super) fn expr_contains_factorial_call(ctx: &cas_ast::Context, root: cas_ast::ExprId) -> bool {
    let mut stack = vec![root];

    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Function(fn_id, args)
                if args.len() == 1 && matches!(ctx.sym_name(*fn_id), "fact" | "factorial") =>
            {
                return true;
            }
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Pow(base, exp) => {
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

/// Arm the per-pipeline probe budget and ambient probe value domain for a
/// TOP-LEVEL simplify pipeline (nested probe pipelines run with nesting > 0
/// and must not re-arm them). The returned guard restores the previous state
/// when the pipeline exits.
pub(crate) fn enter_default_simplify_probe_budget_scope(
    value_domain: crate::semantics::ValueDomain,
) -> DefaultSimplifyProbeBudgetScope {
    if default_simplify_nesting_depth() == 0 {
        let saved = DEFAULT_SIMPLIFY_PROBES_LEFT.with(|left| left.get());
        DEFAULT_SIMPLIFY_PROBES_LEFT.with(|left| left.set(Some(DEFAULT_SIMPLIFY_PROBE_BUDGET)));
        let saved_value_domain = DEFAULT_SIMPLIFY_PROBE_VALUE_DOMAIN.with(|vd| {
            let previous = vd.get();
            vd.set(value_domain);
            previous
        });
        DEFAULT_SIMPLIFY_PROBE_MEMO.with(|memo| {
            let mut memo = memo.borrow_mut();
            if memo.len() > 4096 {
                memo.clear();
            }
        });
        DefaultSimplifyProbeBudgetScope {
            saved: Some(saved),
            saved_value_domain: Some(saved_value_domain),
        }
    } else {
        DefaultSimplifyProbeBudgetScope {
            saved: None,
            saved_value_domain: None,
        }
    }
}

pub(super) fn abs_argument(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Abs) && args.len() == 1 =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

pub(super) fn build_signed_add_expr(
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

pub(super) fn rebuild_subtractive_expr(
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

pub(super) fn exprs_match_shallow_noncall_for_cancellation(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    if expr_contains_any_function_call(ctx, lhs)
        || expr_contains_any_function_call(ctx, rhs)
        || expr_contains_division_node(ctx, lhs)
        || expr_contains_division_node(ctx, rhs)
    {
        return false;
    }

    compare_expr(ctx, lhs, rhs) == Ordering::Equal
        || exprs_equal_up_to_add_term_order(ctx, lhs, rhs)
        || exprs_equal_up_to_mul_factor_order_and_sign(ctx, lhs, rhs)
}

pub(super) fn build_unsigned_sum_expr(
    ctx: &mut cas_ast::Context,
    terms: &[cas_ast::ExprId],
) -> cas_ast::ExprId {
    match terms {
        [] => ctx.num(0),
        [single] => *single,
        _ => build_balanced_add(ctx, terms),
    }
}

fn additive_term_is_negative_like(
    ctx: &cas_ast::Context,
    term_expr: cas_ast::ExprId,
    term_sign: Sign,
) -> bool {
    term_sign == Sign::Neg
        || matches!(ctx.get(term_expr), Expr::Neg(_))
        || matches!(ctx.get(term_expr), Expr::Number(n) if n.is_negative())
}

pub(super) fn additive_scope_has_negative_term(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    AddView::from_expr(ctx, expr)
        .terms
        .into_iter()
        .any(|(term_expr, term_sign)| additive_term_is_negative_like(ctx, term_expr, term_sign))
}

pub(super) fn combine_additive_numeric_constants_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    let mut saw_numeric = false;
    let mut numeric_sum = BigRational::zero();
    let mut rebuilt_terms = Vec::with_capacity(view.terms.len());
    for (term_expr, term_sign) in view.terms {
        if let Expr::Number(value) = ctx.get(term_expr).clone() {
            saw_numeric = true;
            match term_sign {
                Sign::Pos => numeric_sum += value,
                Sign::Neg => numeric_sum -= value,
            }
            continue;
        }

        rebuilt_terms.push((term_expr, term_sign));
    }

    if !saw_numeric {
        return None;
    }

    if !numeric_sum.is_zero() {
        let (sign, magnitude) = if numeric_sum < BigRational::zero() {
            (Sign::Neg, -numeric_sum)
        } else {
            (Sign::Pos, numeric_sum)
        };
        rebuilt_terms.push((ctx.add(Expr::Number(magnitude)), sign));
    }

    let rebuilt = build_signed_sum_expr(ctx, &rebuilt_terms);
    (compare_expr(ctx, rebuilt, expr) != Ordering::Equal).then_some(rebuilt)
}

/// True when `expr` is provably non-finite or undefined over the reals — it
/// contains an `infinity`/`undefined` constant, or a division by a provably-zero
/// denominator (`x/0`, `0/0`). Such a term must NOT cancel against a copy of
/// itself: `inf - inf`, `undefined - undefined`, and `(1/0) - (1/0)` are all
/// indeterminate/undefined, NOT `0`.
/// True when `expr` carries a literal non-finite or undefined value — an
/// `Infinity`/`Undefined` constant, or a division with a provably-zero
/// denominator — anywhere in its tree. Subtracting such a term from itself does
/// NOT cancel to `0` (`inf - inf`, `(1/0) - (1/0)` and `undefined - undefined`
/// are indeterminate, not zero), so every structural additive-cancellation path
/// must decline when this holds. Shared by the additive-pair cancellation rule
/// and the orchestrator's exact-zero equivalence shortcut.
pub(crate) fn additive_term_is_nonfinite_or_undefined(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    cas_math::arithmetic_cancel_support::expr_carries_nonfinite_or_undefined(ctx, expr)
}

pub(crate) fn try_rewrite_exact_additive_term_cancellation_expr(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() == 2 {
        let (lhs_expr, lhs_sign) = normalize_signed_add_term(ctx, view.terms[0].0, view.terms[0].1);
        let (rhs_expr, rhs_sign) = normalize_signed_add_term(ctx, view.terms[1].0, view.terms[1].1);
        if lhs_sign != rhs_sign
            && exprs_match_for_cancellation(ctx, lhs_expr, rhs_expr)
            && !additive_term_is_nonfinite_or_undefined(ctx, lhs_expr)
        {
            return Some(ctx.num(0));
        }
        return None;
    }
    if view.terms.len() < 3 {
        return None;
    }

    let normalized_terms: Vec<_> = view
        .terms
        .iter()
        .copied()
        .map(|(term_expr, term_sign)| normalize_signed_add_term(ctx, term_expr, term_sign))
        .collect();

    let mut used = vec![false; normalized_terms.len()];
    let mut rebuilt_terms = Vec::new();
    let mut changed = false;

    for (index, (term_expr, term_sign)) in normalized_terms.iter().copied().enumerate() {
        if used[index] {
            continue;
        }

        let opposite_index = if additive_term_is_nonfinite_or_undefined(ctx, term_expr) {
            None
        } else {
            normalized_terms.iter().copied().enumerate().find_map(
                |(other_index, (other_expr, other_sign))| {
                    (other_index != index
                        && !used[other_index]
                        && other_sign != term_sign
                        && compare_expr(ctx, term_expr, other_expr) == Ordering::Equal)
                        .then_some(other_index)
                },
            )
        };

        if let Some(other_index) = opposite_index {
            used[index] = true;
            used[other_index] = true;
            changed = true;
            continue;
        }

        used[index] = true;
        rebuilt_terms.push((term_expr, term_sign));
    }

    if !changed {
        return None;
    }

    let rebuilt = build_signed_sum_expr(ctx, &rebuilt_terms);
    let rebuilt =
        combine_additive_numeric_constants_for_cancellation(ctx, rebuilt).unwrap_or(rebuilt);
    (compare_expr(ctx, rebuilt, expr) != Ordering::Equal).then_some(rebuilt)
}

pub(super) fn binomial_i64_for_cancellation(n: u32, k: u32) -> Option<i64> {
    if k > n {
        return None;
    }

    let k = k.min(n - k);
    let mut result: u128 = 1;
    for i in 1..=k {
        let numerator = u128::from(n - k + i);
        let denominator = u128::from(i);
        result = result.checked_mul(numerator)? / denominator;
    }

    i64::try_from(result).ok()
}

pub(super) fn apply_numeric_scale_for_cancellation(
    ctx: &mut cas_ast::Context,
    coeff: &BigRational,
    expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    if coeff.is_zero() {
        return ctx.num(0);
    }
    if coeff.is_one() {
        return expr;
    }
    if coeff == &BigRational::from_integer(BigInt::from(-1_i32)) {
        return ctx.add(Expr::Neg(expr));
    }

    if let Expr::Div(numerator, denominator) = ctx.get(expr).clone() {
        if let Expr::Number(den) = ctx.get(denominator) {
            let combined = coeff / den.clone();
            if combined.is_one() {
                return numerator;
            }
            if combined == BigRational::from_integer(BigInt::from(-1_i32)) {
                return ctx.add(Expr::Neg(numerator));
            }
            let combined_expr = ctx.add(Expr::Number(combined));
            return smart_mul(ctx, combined_expr, numerator);
        }
    }

    let coeff_expr = ctx.add(Expr::Number(coeff.clone()));
    smart_mul(ctx, coeff_expr, expr)
}

fn canonicalize_commutative_mul_term_for_fast_match(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let view = MulView::from_expr(ctx, expr);
    if view.len() <= 1 || !view.commutative {
        expr
    } else {
        view.rebuild(ctx)
    }
}

pub(super) fn signed_terms_match_multiset(
    ctx: &mut cas_ast::Context,
    actual: &[(cas_ast::ExprId, Sign)],
    expected: &[(cas_ast::ExprId, Sign)],
) -> bool {
    if actual.len() != expected.len() {
        return false;
    }

    let canonical_actual: Vec<_> = actual
        .iter()
        .copied()
        .map(|(expr, sign)| {
            (
                canonicalize_commutative_mul_term_for_fast_match(ctx, expr),
                sign,
            )
        })
        .collect();
    let canonical_expected: Vec<_> = expected
        .iter()
        .copied()
        .map(|(expr, sign)| {
            (
                canonicalize_commutative_mul_term_for_fast_match(ctx, expr),
                sign,
            )
        })
        .collect();

    let mut used = vec![false; actual.len()];
    for (expected_expr, expected_sign) in &canonical_expected {
        let mut found = false;
        for (index, (actual_expr, actual_sign)) in canonical_actual.iter().copied().enumerate() {
            if used[index] || actual_sign != *expected_sign {
                continue;
            }
            if compare_expr(ctx, actual_expr, *expected_expr) == Ordering::Equal {
                used[index] = true;
                found = true;
                break;
            }
        }
        if !found {
            return false;
        }
    }

    true
}

pub(super) fn is_positive_one_expr(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if *n == BigRational::one())
}

pub(super) fn extract_literal_rational_for_cancellation(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Number(n) => Some(n.clone()),
        Expr::Neg(inner) => extract_literal_rational_for_cancellation(ctx, *inner).map(|n| -n),
        Expr::Div(numerator, denominator) => {
            let numerator = extract_literal_rational_for_cancellation(ctx, *numerator)?;
            let denominator = extract_literal_rational_for_cancellation(ctx, *denominator)?;
            Some(numerator / denominator)
        }
        _ => None,
    }
}

pub(super) fn scale_with_add_sign(scale: BigRational, sign: Sign) -> BigRational {
    match sign {
        Sign::Pos => scale,
        Sign::Neg => -scale,
    }
}

pub(super) fn bare_variable_name(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> Option<String> {
    match ctx.get(expr) {
        Expr::Variable(sym_id) => Some(ctx.sym_name(*sym_id).to_string()),
        _ => None,
    }
}

fn collect_non_division_linear_variable_names_from_term(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
    names: &mut std::collections::BTreeSet<String>,
) {
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Variable(sym_id) => {
                names.insert(ctx.sym_name(*sym_id).to_string());
            }
            Expr::Pow(base, exp)
                if extract_i64_integer(ctx, *exp) == Some(2)
                    && matches!(ctx.get(*base), Expr::Variable(_)) => {}
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Pow(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            Expr::Div(num, _) => stack.push(*num),
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }
}

pub(super) fn collect_direct_additive_linear_variable_names(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
) -> Vec<String> {
    let mut names = std::collections::BTreeSet::new();
    let view = AddView::from_expr(ctx, root);
    for (term_expr, _) in view.terms {
        collect_non_division_linear_variable_names_from_term(ctx, term_expr, &mut names);
    }
    names.into_iter().collect()
}

pub(super) fn is_positive_one_half_expr(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(n) if *n == BigRational::new(BigInt::from(1), BigInt::from(2))
    )
}

pub(super) fn extract_positive_half_scaled_base_expr(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    match ctx.get(expr) {
        Expr::Div(num, den) if extract_i64_integer(ctx, *den) == Some(2) => Some(*num),
        Expr::Mul(lhs, rhs) if is_positive_one_half_expr(ctx, *lhs) => Some(*rhs),
        Expr::Mul(lhs, rhs) if is_positive_one_half_expr(ctx, *rhs) => Some(*lhs),
        _ => None,
    }
}

pub(super) fn expr_contains_pi_constant(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Constant(c) => matches!(c, cas_ast::Constant::Pi),
        Expr::Add(lhs, rhs)
        | Expr::Sub(lhs, rhs)
        | Expr::Mul(lhs, rhs)
        | Expr::Div(lhs, rhs)
        | Expr::Pow(lhs, rhs) => {
            expr_contains_pi_constant(ctx, *lhs) || expr_contains_pi_constant(ctx, *rhs)
        }
        Expr::Neg(inner) => expr_contains_pi_constant(ctx, *inner),
        Expr::Function(_, args) => args.iter().any(|arg| expr_contains_pi_constant(ctx, *arg)),
        _ => false,
    }
}

pub(super) fn build_pi_over_for_cancellation(
    ctx: &mut cas_ast::Context,
    denominator: i64,
) -> cas_ast::ExprId {
    let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
    let denom = ctx.num(denominator);
    ctx.add(Expr::Div(pi, denom))
}

pub(super) fn split_out_small_integer_factor_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    value: i64,
) -> Option<cas_ast::ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    if let Some(index) = factors
        .iter()
        .position(|factor| extract_i64_integer(ctx, *factor) == Some(value))
    {
        return Some(
            factors
                .into_iter()
                .enumerate()
                .filter_map(|(i, factor)| (i != index).then_some(factor))
                .fold(ctx.num(1), |acc, factor| smart_mul(ctx, acc, factor)),
        );
    }

    for (index, factor) in factors.iter().copied().enumerate() {
        let Expr::Number(n) = ctx.get(factor) else {
            continue;
        };
        let divisor = BigRational::from_integer(value.into());
        let quotient = n / &divisor;
        if !quotient.is_integer() {
            continue;
        }

        let quotient_id = ctx.add(Expr::Number(quotient));
        let rebuilt = factors
            .iter()
            .copied()
            .enumerate()
            .map(|(i, existing)| if i == index { quotient_id } else { existing })
            .fold(ctx.num(1), |acc, factor| smart_mul(ctx, acc, factor));
        return Some(rebuilt);
    }

    None
}

pub(super) fn extract_scaled_double_sine_product_for_cancellation(
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

pub(super) fn extract_binary_product_with_sum_factor(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<BinaryProductWithSumFactor> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    for (sum_factor, common_factor) in [(factors[0], factors[1]), (factors[1], factors[0])] {
        let sum_terms = AddView::from_expr(ctx, sum_factor).terms;
        if sum_terms.len() != 2 {
            continue;
        }
        return Some((common_factor, sum_terms[0], sum_terms[1]));
    }

    None
}

pub(super) fn combine_signs(lhs: Sign, rhs: Sign) -> Sign {
    if lhs == rhs {
        Sign::Pos
    } else {
        Sign::Neg
    }
}

pub(super) fn term_matches_binary_product(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    first: cas_ast::ExprId,
    second: cas_ast::ExprId,
) -> bool {
    let factors = flatten_mul_chain(ctx, expr);
    factors.len() == 2
        && ((compare_expr(ctx, factors[0], first) == Ordering::Equal
            && compare_expr(ctx, factors[1], second) == Ordering::Equal)
            || (compare_expr(ctx, factors[0], second) == Ordering::Equal
                && compare_expr(ctx, factors[1], first) == Ordering::Equal))
}

pub(super) fn extract_exact_double_sine_product_args(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 3 {
        return None;
    }

    let mut saw_two = false;
    let mut sin_args = Vec::new();
    for factor in factors {
        match ctx.get(factor) {
            Expr::Number(n) if *n == BigRational::from_integer(2.into()) && !saw_two => {
                saw_two = true;
            }
            Expr::Function(fn_id, args)
                if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Sin) =>
            {
                sin_args.push(args[0]);
            }
            _ => return None,
        }
    }

    if saw_two && sin_args.len() == 2 {
        Some((sin_args[0], sin_args[1]))
    } else {
        None
    }
}

pub(super) fn extract_exact_double_sine_product_args_from_signed_expr(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => extract_exact_double_sine_product_args(ctx, inner),
        _ => extract_exact_double_sine_product_args(ctx, expr),
    }
}

pub(super) fn strip_common_factor_from_term(
    ctx: &mut cas_ast::Context,
    term_expr: cas_ast::ExprId,
    common_factor: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let term_factors = flatten_mul_chain(ctx, term_expr);
    let common_factors = flatten_mul_chain(ctx, common_factor);
    if term_factors.is_empty() || common_factors.is_empty() {
        return None;
    }

    let mut used = vec![false; term_factors.len()];
    for common in common_factors {
        let matched_index = term_factors
            .iter()
            .enumerate()
            .find_map(|(index, factor)| {
                (!used[index] && compare_expr(ctx, *factor, common) == Ordering::Equal)
                    .then_some(index)
            })?;
        used[matched_index] = true;
    }

    let residual_factors: Vec<_> = term_factors
        .into_iter()
        .enumerate()
        .filter_map(|(index, factor)| (!used[index]).then_some(factor))
        .collect();
    Some(build_mul_expr_from_factors(ctx, &residual_factors))
}

pub(super) fn extract_two_term_common_scale_difference_cores(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, cas_ast::ExprId)> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let (common_factor, _) = extract_common_multiplicative_residual_sum(ctx, expr)?;
    let lhs_residual = strip_common_factor_from_term(ctx, view.terms[0].0, common_factor)?;
    let rhs_residual = strip_common_factor_from_term(ctx, view.terms[1].0, common_factor)?;
    let lhs_core = apply_sign_to_expr(ctx, sign_to_i64(view.terms[0].1), lhs_residual);
    let rhs_core = apply_sign_to_expr(
        ctx,
        sign_to_i64(view.terms[1].1).checked_neg()?,
        rhs_residual,
    );
    Some((common_factor, lhs_core, rhs_core))
}

fn positive_assumption_subject_for_abs_like_expr(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if let Some(inner) = abs_argument(ctx, expr) {
        return Some(inner);
    }

    let radicand = extract_square_root_base(ctx, expr)?;
    match ctx.get(radicand).clone() {
        Expr::Pow(base, exponent) if extract_i64_integer(ctx, exponent) == Some(2) => Some(base),
        Expr::Mul(lhs, rhs) if exprs_match_for_cancellation(ctx, lhs, rhs) => Some(lhs),
        _ => None,
    }
}

fn positive_assumption_subject_for_abs_like_core_pair(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if let Some(subject) = positive_assumption_subject_for_abs_like_expr(ctx, lhs_core) {
        if exprs_match_for_cancellation(ctx, subject, rhs_core) {
            return Some(subject);
        }
    }

    if let Some(subject) = positive_assumption_subject_for_abs_like_expr(ctx, rhs_core) {
        if exprs_match_for_cancellation(ctx, subject, lhs_core) {
            return Some(subject);
        }
    }

    None
}

pub(crate) fn common_scale_abs_like_positive_assumption_event(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    parent_ctx: &ParentContext,
) -> Option<crate::AssumptionEvent> {
    if !matches!(parent_ctx.domain_mode(), crate::DomainMode::Assume) {
        return None;
    }

    let (_common_factor, lhs_core, rhs_core) =
        extract_two_term_common_scale_difference_cores(ctx, expr)?;
    let subject = positive_assumption_subject_for_abs_like_core_pair(ctx, lhs_core, rhs_core)?;
    if crate::helpers::prove_positive(ctx, subject, parent_ctx.value_domain())
        == crate::Proof::Proven
    {
        return None;
    }

    Some(crate::AssumptionEvent::positive_assumed(ctx, subject))
}

/// EXACT soundness witness for the "collapse to 0" routes: if `expr` is a rational/algebraic form
/// whose value at a generic rational assignment of its free variables is provably NON-ZERO, then it
/// is NOT identically zero and must never be collapsed to `0`. Returns `true` on such a witness.
/// Transcendental sub-terms (`sin`, `exp`, surds …) make `as_rational_const` return `None`, so those
/// routes keep their existing behaviour — this guard only ever VETOES a wrong collapse, never forces
/// one. Catches e.g. `1/(x²−1) − 1/(x−1)` (value −2/3 at x=2), which a pattern matcher mistook for a
/// common-scale cancellation.
/// Exact rational evaluation of a constant expression, including INTEGER powers (which
/// `as_rational_const` declines). `None` for any transcendental/non-rational sub-term, a division by
/// zero, or a non-integer / oversized exponent.
pub(super) fn eval_exact_rational(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
    depth: usize,
) -> Option<num_rational::BigRational> {
    use num_traits::{One, Zero};
    if depth == 0 {
        return None;
    }
    match ctx.get(expr) {
        cas_ast::Expr::Number(n) => Some(n.clone()),
        cas_ast::Expr::Neg(inner) => Some(-eval_exact_rational(ctx, *inner, depth - 1)?),
        cas_ast::Expr::Add(l, r) => Some(
            eval_exact_rational(ctx, *l, depth - 1)? + eval_exact_rational(ctx, *r, depth - 1)?,
        ),
        cas_ast::Expr::Sub(l, r) => Some(
            eval_exact_rational(ctx, *l, depth - 1)? - eval_exact_rational(ctx, *r, depth - 1)?,
        ),
        cas_ast::Expr::Mul(l, r) => Some(
            eval_exact_rational(ctx, *l, depth - 1)? * eval_exact_rational(ctx, *r, depth - 1)?,
        ),
        cas_ast::Expr::Div(l, r) => {
            let d = eval_exact_rational(ctx, *r, depth - 1)?;
            if d.is_zero() {
                return None;
            }
            Some(eval_exact_rational(ctx, *l, depth - 1)? / d)
        }
        cas_ast::Expr::Pow(base, exponent) => {
            let exp = eval_exact_rational(ctx, *exponent, depth - 1)?;
            if !exp.is_integer() {
                return None;
            }
            let e = exp.to_integer().to_i64()?;
            if e.unsigned_abs() > 64 {
                return None; // avoid blow-up
            }
            let b = eval_exact_rational(ctx, *base, depth - 1)?;
            if e == 0 {
                return Some(num_rational::BigRational::one());
            }
            let factor = if e < 0 {
                if b.is_zero() {
                    return None;
                }
                num_rational::BigRational::one() / b
            } else {
                b
            };
            let mut acc = num_rational::BigRational::one();
            for _ in 0..e.unsigned_abs() {
                acc *= &factor;
            }
            Some(acc)
        }
        _ => None,
    }
}

pub(super) fn strip_positive_one_passthrough(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let view = AddView::from_expr(ctx, expr);
    let mut stripped = false;
    let mut residual_terms = Vec::new();

    for (term_expr, term_sign) in view.terms {
        let is_positive_one = term_sign == Sign::Pos
            && matches!(
                ctx.get(term_expr),
                Expr::Number(n) if *n == num_rational::BigRational::from_integer(1.into())
            );

        if is_positive_one && !stripped {
            stripped = true;
            continue;
        }
        residual_terms.push((term_expr, term_sign));
    }

    if !stripped || residual_terms.is_empty() {
        return None;
    }

    Some(build_signed_sum_expr(ctx, &residual_terms))
}

pub(super) fn try_build_two_term_core_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let (lhs_core, rhs_core) = extract_two_term_core_difference(ctx, expr)?;

    try_build_direct_core_equivalence_rewrite(ctx, lhs_core, rhs_core)
}

pub(super) fn strip_trivial_one_product_factors_for_core_difference(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() <= 1 {
        return expr;
    }

    let retained: Vec<_> = factors
        .iter()
        .copied()
        .filter(|factor| !is_one_expr(ctx, *factor))
        .collect();
    if retained.len() == factors.len() {
        expr
    } else {
        build_mul_expr_from_factors(ctx, &retained)
    }
}

pub(super) fn normalize_core_difference_term(
    ctx: &mut cas_ast::Context,
    term_expr: cas_ast::ExprId,
    term_sign: Sign,
) -> (cas_ast::ExprId, Sign) {
    let (term_expr, term_sign) =
        normalize_signed_add_term_for_fast_match(ctx, term_expr, term_sign);
    (
        strip_trivial_one_product_factors_for_core_difference(ctx, term_expr),
        term_sign,
    )
}

pub(super) fn try_build_direct_dirichlet_core_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    let lhs_minus_rhs = ctx.add(Expr::Sub(lhs_core, rhs_core));
    if let Some(result) = try_dirichlet_kernel_identity(ctx, lhs_minus_rhs) {
        return Some(
            Rewrite::with_local(ctx.num(0), "Dirichlet Kernel Identity", lhs_core, rhs_core)
                .requires(crate::ImplicitCondition::NonZero(
                    build_dirichlet_nonzero_condition_expr(ctx, result),
                )),
        );
    }

    let rhs_minus_lhs = ctx.add(Expr::Sub(rhs_core, lhs_core));
    let result = try_dirichlet_kernel_identity(ctx, rhs_minus_lhs)?;
    Some(
        Rewrite::with_local(ctx.num(0), "Dirichlet Kernel Identity", rhs_core, lhs_core).requires(
            crate::ImplicitCondition::NonZero(build_dirichlet_nonzero_condition_expr(ctx, result)),
        ),
    )
}

pub(super) fn maybe_integrate_prep_exact_additive_candidate(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    if !expr_contains_division_node(ctx, expr)
        || !expr_contains_any_builtin(ctx, expr, &[BuiltinFn::Sin, BuiltinFn::Cos])
    {
        return false;
    }

    try_build_direct_integrate_prep_exact_zero_scope_rewrite(ctx, expr).is_some()
        || try_build_exact_dirichlet_zero_scope_rewrite(ctx, expr).is_some()
}

pub(super) fn try_build_direct_finite_product_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(plan) = try_plan_finite_product_evaluation(ctx, source, 1000) else {
            continue;
        };

        if exprs_match_for_cancellation(ctx, plan.candidate, target)
            || exprs_match_after_default_simplify(ctx, plan.candidate, target)
        {
            let description = match plan.kind {
                ProductEvaluationKind::Telescoping
                | ProductEvaluationKind::FactorizedTelescoping => "Finite Telescoping Product",
                ProductEvaluationKind::ProductOfFirstIntegers
                | ProductEvaluationKind::ProductOfPowers
                | ProductEvaluationKind::ProductOfConstant => "Finite Product Closed Form",
                ProductEvaluationKind::FiniteDirect { .. } => "Finite Product",
                ProductEvaluationKind::DivergentInfinite => "Divergent Infinite Product",
            };
            return Some(Rewrite::with_local(ctx.num(0), description, source, target));
        }
    }

    None
}

fn finite_sum_evaluation_description(kind: &SumEvaluationKind) -> &'static str {
    match kind {
        SumEvaluationKind::Telescoping => "Finite Telescoping Sum",
        SumEvaluationKind::SumOfFirstIntegers
        | SumEvaluationKind::SumOfSquares
        | SumEvaluationKind::SumOfCubes
        | SumEvaluationKind::SumOfConstant
        | SumEvaluationKind::GeometricPower
        | SumEvaluationKind::PolynomialLinearity => "Finite Sum Closed Form",
        SumEvaluationKind::FiniteDirect { .. } => "Finite Sum",
        SumEvaluationKind::DivergentInfinite => "Divergent Infinite Series",
        SumEvaluationKind::ConvergentInfinite => "Convergent Geometric Series",
        SumEvaluationKind::UndefinedPole => "Undefined (pole in range)",
    }
}

fn try_plan_scaled_finite_sum_evaluation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, SumEvaluationPlan)> {
    if let Some(plan) = try_plan_finite_sum_evaluation(ctx, expr, 1000) {
        let one = ctx.num(1);
        return Some((one, plan));
    }

    let view = MulView::from_expr(ctx, expr);
    if view.factors.len() < 2 {
        return None;
    }

    for (index, factor) in view.factors.iter().copied().enumerate() {
        let Some(plan) = try_plan_finite_sum_evaluation(ctx, factor, 1000) else {
            continue;
        };

        let scale_factors: Vec<_> = view
            .factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(factor_index, factor)| (factor_index != index).then_some(factor))
            .collect();
        return Some((build_mul_expr_from_factors(ctx, &scale_factors), plan));
    }

    None
}

fn finite_evaluation_candidate_matches_target(
    ctx: &mut cas_ast::Context,
    candidate: cas_ast::ExprId,
    target: cas_ast::ExprId,
) -> bool {
    if exprs_match_for_cancellation(ctx, candidate, target)
        || exprs_match_after_default_simplify(ctx, candidate, target)
    {
        return true;
    }

    if finite_evaluation_fraction_parts_match(ctx, candidate, target) {
        return true;
    }

    let residual = ctx.add(Expr::Sub(candidate, target));
    try_build_exact_zero_identity_rewrite_direct_impl(ctx, residual, false).is_some()
        || try_build_exact_zero_common_scaled_difference_rewrite(ctx, residual).is_some()
}

pub(super) fn try_build_direct_finite_sum_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((scale, plan)) = try_plan_scaled_finite_sum_evaluation(ctx, source) else {
            continue;
        };
        let candidate = smart_mul(ctx, scale, plan.candidate);

        if finite_evaluation_candidate_matches_target(ctx, candidate, target) {
            let description = finite_sum_evaluation_description(&plan.kind);
            return Some(Rewrite::with_local(ctx.num(0), description, source, target));
        }
    }

    None
}

pub(super) fn is_simple_symbolic_scale_factor_for_cancellation(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    match ctx.get(expr) {
        Expr::Variable(_) | Expr::SessionRef(_) => true,
        Expr::Pow(base, exp) => {
            matches!(ctx.get(*base), Expr::Variable(_) | Expr::SessionRef(_))
                && extract_i64_integer(ctx, *exp).is_some_and(|value| value > 0)
        }
        _ => false,
    }
}

fn distribute_symbolic_scale_sum_term_for_cancellation(
    ctx: &mut cas_ast::Context,
    scale_expr: cas_ast::ExprId,
    term_expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let Expr::Div(numerator, denominator) = ctx.get(term_expr).clone() else {
        return smart_mul(ctx, scale_expr, term_expr);
    };

    if compare_expr(ctx, denominator, scale_expr) == Ordering::Equal {
        numerator
    } else {
        smart_mul(ctx, scale_expr, term_expr)
    }
}

fn try_rewrite_single_symbolic_scale_sum_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let (scale_expr, sum_expr) =
        if is_simple_symbolic_scale_factor_for_cancellation(ctx, factors[0])
            && matches!(ctx.get(factors[1]), Expr::Add(_, _) | Expr::Sub(_, _))
        {
            (factors[0], factors[1])
        } else if is_simple_symbolic_scale_factor_for_cancellation(ctx, factors[1])
            && matches!(ctx.get(factors[0]), Expr::Add(_, _) | Expr::Sub(_, _))
        {
            (factors[1], factors[0])
        } else {
            return None;
        };

    let sum_terms = AddView::from_expr(ctx, sum_expr).terms;
    if !(2..=4).contains(&sum_terms.len()) {
        return None;
    }

    let distributed_terms: Vec<_> = sum_terms
        .into_iter()
        .map(|(term_expr, term_sign)| {
            let distributed_term =
                distribute_symbolic_scale_sum_term_for_cancellation(ctx, scale_expr, term_expr);
            let (normalized_expr, normalized_sign) =
                normalize_signed_add_term(ctx, distributed_term, Sign::Pos);
            let combined_sign = match term_sign {
                Sign::Pos => normalized_sign,
                Sign::Neg => normalized_sign.negate(),
            };
            (normalized_expr, combined_sign)
        })
        .collect();

    let distributed_expr = build_signed_sum_expr(ctx, &distributed_terms);
    (compare_expr(ctx, distributed_expr, expr) != Ordering::Equal).then_some(distributed_expr)
}

fn collect_distributed_single_symbolic_scale_sum_terms_for_fast_match(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Vec<(cas_ast::ExprId, Sign)>> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let (scale_expr, sum_expr) =
        if is_simple_symbolic_scale_factor_for_cancellation(ctx, factors[0])
            && matches!(ctx.get(factors[1]), Expr::Add(_, _) | Expr::Sub(_, _))
        {
            (factors[0], factors[1])
        } else if is_simple_symbolic_scale_factor_for_cancellation(ctx, factors[1])
            && matches!(ctx.get(factors[0]), Expr::Add(_, _) | Expr::Sub(_, _))
        {
            (factors[1], factors[0])
        } else {
            return None;
        };

    let sum_terms = AddView::from_expr(ctx, sum_expr).terms;
    if !(2..=4).contains(&sum_terms.len()) {
        return None;
    }

    Some(
        sum_terms
            .into_iter()
            .map(|(term_expr, term_sign)| {
                let distributed_term =
                    distribute_symbolic_scale_sum_term_for_cancellation(ctx, scale_expr, term_expr);
                let (normalized_expr, normalized_sign) =
                    normalize_signed_add_term_for_fast_match(ctx, distributed_term, Sign::Pos);
                let combined_sign = match term_sign {
                    Sign::Pos => normalized_sign,
                    Sign::Neg => normalized_sign.negate(),
                };
                (normalized_expr, combined_sign)
            })
            .collect(),
    )
}

pub(super) fn grouped_symbolic_scale_sum_matches_target_for_cancellation(
    ctx: &mut cas_ast::Context,
    grouped_expr: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
) -> bool {
    let grouped_terms = AddView::from_expr(ctx, grouped_expr).terms;
    if !(2..=4).contains(&grouped_terms.len()) {
        return false;
    }

    let mut distributed_terms = Vec::new();
    for (term_expr, term_sign) in grouped_terms {
        let Some(rewritten_terms) =
            collect_distributed_single_symbolic_scale_sum_terms_for_fast_match(ctx, term_expr)
        else {
            return false;
        };
        distributed_terms.extend(rewritten_terms.into_iter().map(|(expr, sign)| {
            let combined_sign = match term_sign {
                Sign::Pos => sign,
                Sign::Neg => sign.negate(),
            };
            (expr, combined_sign)
        }));
    }

    let target_terms: Vec<_> = AddView::from_expr(ctx, target_expr)
        .terms
        .into_iter()
        .map(|(term_expr, term_sign)| {
            normalize_signed_add_term_for_fast_match(ctx, term_expr, term_sign)
        })
        .collect();
    if distributed_terms.len() != target_terms.len() {
        return false;
    }

    let mut used_target = vec![false; target_terms.len()];
    for (distributed_expr, distributed_sign) in distributed_terms {
        let Some(target_index) =
            target_terms
                .iter()
                .enumerate()
                .find_map(|(index, (target_expr, target_sign))| {
                    (!used_target[index]
                        && distributed_sign == *target_sign
                        && exprs_match_for_cancellation_leaf(ctx, distributed_expr, *target_expr))
                    .then_some(index)
                })
        else {
            return false;
        };
        used_target[target_index] = true;
    }

    true
}

pub(super) fn try_rewrite_simple_symbolic_scale_sum_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if let Some(rewritten) = try_rewrite_single_symbolic_scale_sum_for_cancellation(ctx, expr) {
        return Some(rewritten);
    }

    let sum_terms = AddView::from_expr(ctx, expr).terms;
    if !(2..=4).contains(&sum_terms.len()) {
        return None;
    }

    let mut rewritten_terms = Vec::with_capacity(sum_terms.len());
    for (term_expr, term_sign) in sum_terms {
        let rewritten_term =
            try_rewrite_single_symbolic_scale_sum_for_cancellation(ctx, term_expr)?;
        rewritten_terms.push(normalize_signed_add_term(ctx, rewritten_term, term_sign));
    }

    let rewritten_expr = build_signed_sum_expr(ctx, &rewritten_terms);
    (compare_expr(ctx, rewritten_expr, expr) != Ordering::Equal).then_some(rewritten_expr)
}

pub(super) fn extract_unary_builtin_arg(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
    builtin: BuiltinFn,
) -> Option<cas_ast::ExprId> {
    match ctx.get(expr) {
        Expr::Function(name, args) if ctx.is_builtin(*name, builtin) && args.len() == 1 => {
            Some(args[0])
        }
        _ => None,
    }
}

pub(super) fn extract_two_times_factor_arg(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    if extract_i64_integer(ctx, factors[0]) == Some(2) {
        Some(factors[1])
    } else if extract_i64_integer(ctx, factors[1]) == Some(2) {
        Some(factors[0])
    } else {
        None
    }
}

pub(super) fn positive_two_term_sum_matches_terms(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 || terms.iter().any(|(_, sign)| *sign != Sign::Pos) {
        return false;
    }

    let first = terms[0].0;
    let second = terms[1].0;
    (exprs_match_for_cancellation(ctx, first, lhs)
        && exprs_match_for_cancellation(ctx, second, rhs))
        || (exprs_match_for_cancellation(ctx, first, rhs)
            && exprs_match_for_cancellation(ctx, second, lhs))
}

pub(super) fn product_has_top_level_additive_factor(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    flatten_mul_chain(ctx, expr).iter().copied().any(|factor| {
        let factor = match ctx.get(factor) {
            Expr::Neg(inner) => *inner,
            _ => factor,
        };
        matches!(ctx.get(factor), Expr::Add(_, _) | Expr::Sub(_, _))
    })
}

pub(super) fn exprs_match_with_local_default_simplify(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    exprs_match_for_cancellation(ctx, lhs, rhs) || exprs_match_after_default_simplify(ctx, lhs, rhs)
}

pub(super) fn split_matching_factor_from_product(
    ctx: &mut cas_ast::Context,
    product: cas_ast::ExprId,
    expected_factor: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if exprs_match_for_cancellation(ctx, product, expected_factor) {
        return Some(ctx.num(1));
    }

    let view = MulView::from_expr(ctx, product);
    for (index, factor) in view.factors.iter().copied().enumerate() {
        let factor_scale = if exprs_match_for_cancellation(ctx, factor, expected_factor) {
            Some(ctx.num(1))
        } else {
            split_scaled_additive_factor_matching_expected(ctx, factor, expected_factor)
        };
        let Some(factor_scale) = factor_scale else {
            continue;
        };

        let mut remaining_factors: Vec<_> = view
            .factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(factor_index, factor)| (factor_index != index).then_some(factor))
            .collect();
        let one = ctx.num(1);
        if compare_expr(ctx, factor_scale, one) != Ordering::Equal {
            remaining_factors.push(factor_scale);
        }
        return Some(build_scale_from_factors(ctx, &remaining_factors));
    }

    None
}

fn split_scaled_additive_factor_matching_expected(
    ctx: &mut cas_ast::Context,
    factor: cas_ast::ExprId,
    expected_factor: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let (common_factor, residual_expr) = extract_common_multiplicative_residual_sum(ctx, factor)?;
    exprs_match_for_cancellation(ctx, residual_expr, expected_factor).then_some(common_factor)
}

pub(super) fn build_scale_from_factors(
    ctx: &mut cas_ast::Context,
    factors: &[cas_ast::ExprId],
) -> cas_ast::ExprId {
    if factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, factors)
    }
}

pub(super) fn signed_division_parts(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Some(parts) = as_div(ctx, expr) {
        return Some(parts);
    }

    let Expr::Neg(inner) = ctx.get(expr).clone() else {
        return scaled_signed_division_parts(ctx, expr);
    };
    let inner = cas_ast::hold::unwrap_internal_hold(ctx, inner);
    let (numerator, denominator) = as_div(ctx, inner)?;
    Some((ctx.add(Expr::Neg(numerator)), denominator))
}

fn scaled_signed_division_parts(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let view = MulView::from_expr(ctx, expr);
    if view.factors.len() < 2 {
        return None;
    }

    let mut scale = BigRational::one();
    let mut division_parts = None;
    for factor in view.factors {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
            continue;
        }

        if division_parts.is_some() {
            return None;
        }
        let factor = cas_ast::hold::unwrap_internal_hold(ctx, factor);
        division_parts = as_div(ctx, factor);
        division_parts?;
    }

    let (numerator, denominator) = division_parts?;
    scale_division_numerator_denominator_by_rational(ctx, numerator, denominator, scale)
}

pub(super) fn reject_atomic_noncall_pair_before_default_simplify(
    ctx: &cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<bool> {
    if expr_is_atomic_noncall(ctx, lhs_core) && expr_is_atomic_noncall(ctx, rhs_core) {
        return Some(false);
    }

    None
}

fn extract_scaled_symbolic_atom_for_default_simplify_reject(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, BigRational)> {
    match ctx.get(expr).clone() {
        Expr::Variable(_) | Expr::SessionRef(_) | Expr::Constant(_) => {
            return Some((expr, BigRational::one()));
        }
        Expr::Neg(inner) => {
            let (atom, scale) =
                extract_scaled_symbolic_atom_for_default_simplify_reject(ctx, inner)?;
            return Some((atom, -scale));
        }
        Expr::Mul(_, _) => {}
        _ => return None,
    }

    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let mut atom = None;
    let mut scale = BigRational::one();
    for factor in factors {
        if let Some(literal_scale) = extract_literal_rational_for_cancellation(ctx, factor) {
            scale *= literal_scale;
            continue;
        }

        let (factor_atom, factor_scale) =
            extract_scaled_symbolic_atom_for_default_simplify_reject(ctx, factor)?;
        if atom.replace(factor_atom).is_some() {
            return None;
        }
        scale *= factor_scale;
    }

    Some((atom?, scale))
}

pub(super) fn reject_scaled_symbolic_atom_mismatch_before_default_simplify(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<bool> {
    let (lhs_atom, lhs_scale) =
        extract_scaled_symbolic_atom_for_default_simplify_reject(ctx, lhs_core)?;
    let (rhs_atom, rhs_scale) =
        extract_scaled_symbolic_atom_for_default_simplify_reject(ctx, rhs_core)?;

    if compare_expr(ctx, lhs_atom, rhs_atom) == Ordering::Equal && lhs_scale != rhs_scale {
        Some(false)
    } else {
        None
    }
}

pub(super) fn has_builtin_on_either_side(
    ctx: &cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
    builtin: BuiltinFn,
) -> bool {
    expr_contains_any_builtin(ctx, lhs_core, &[builtin])
        || expr_contains_any_builtin(ctx, rhs_core, &[builtin])
}

pub(super) fn has_div_on_either_side(
    ctx: &cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> bool {
    matches!(ctx.get(lhs_core), Expr::Div(_, _)) || matches!(ctx.get(rhs_core), Expr::Div(_, _))
}

pub(super) fn extract_atomic_noncall_factor_set(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Vec<cas_ast::ExprId>> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.is_empty()
        || !factors
            .iter()
            .all(|factor| expr_is_atomic_noncall(ctx, *factor))
    {
        return None;
    }

    let mut unique = Vec::new();
    for factor in factors {
        if unique
            .iter()
            .any(|existing| compare_expr(ctx, *existing, factor) == Ordering::Equal)
        {
            continue;
        }
        unique.push(factor);
    }
    Some(unique)
}

pub(super) fn extract_shared_additive_passthrough_difference_cores(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    fn extract_cores_from_term_lists(
        ctx: &mut cas_ast::Context,
        lhs_terms: Vec<(cas_ast::ExprId, Sign)>,
        rhs_terms: Vec<(cas_ast::ExprId, Sign)>,
    ) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
        if lhs_terms.is_empty() || rhs_terms.is_empty() {
            return None;
        }

        let mut rhs_used = vec![false; rhs_terms.len()];
        let mut lhs_remaining = Vec::new();
        let mut matched_any = false;

        for (lhs_expr, lhs_sign) in lhs_terms {
            let mut matched_index = None;
            for (rhs_index, (rhs_expr, rhs_sign)) in rhs_terms.iter().copied().enumerate() {
                if rhs_used[rhs_index] || lhs_sign != rhs_sign {
                    continue;
                }
                if compare_expr(ctx, lhs_expr, rhs_expr) == Ordering::Equal {
                    matched_index = Some(rhs_index);
                    break;
                }
            }

            if let Some(rhs_index) = matched_index {
                rhs_used[rhs_index] = true;
                matched_any = true;
            } else {
                lhs_remaining.push((lhs_expr, lhs_sign));
            }
        }

        if !matched_any {
            return None;
        }

        let rhs_remaining: Vec<_> = rhs_terms
            .into_iter()
            .enumerate()
            .filter_map(|(index, term)| (!rhs_used[index]).then_some(term))
            .collect();

        if lhs_remaining.is_empty() || rhs_remaining.is_empty() {
            return None;
        }

        let lhs_core = build_signed_sum_expr(ctx, &lhs_remaining);
        let rhs_core = build_signed_sum_expr(ctx, &rhs_remaining);
        Some((lhs_core, rhs_core))
    }

    if let Some((lhs, rhs)) = match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) => Some((lhs, rhs)),
        Expr::Add(lhs, rhs) => match ctx.get(rhs).clone() {
            Expr::Neg(inner) => Some((lhs, inner)),
            _ => None,
        },
        _ => None,
    } {
        let lhs_terms: Vec<_> = AddView::from_expr(ctx, lhs)
            .terms
            .iter()
            .copied()
            .map(|(term_expr, term_sign)| normalize_signed_add_term(ctx, term_expr, term_sign))
            .collect();
        let rhs_terms: Vec<_> = AddView::from_expr(ctx, rhs)
            .terms
            .iter()
            .copied()
            .map(|(term_expr, term_sign)| normalize_signed_add_term(ctx, term_expr, term_sign))
            .collect();
        if let Some(cores) = extract_cores_from_term_lists(ctx, lhs_terms, rhs_terms) {
            return Some(cores);
        }
    }

    let normalized_terms: Vec<_> = AddView::from_expr(ctx, expr)
        .terms
        .iter()
        .copied()
        .map(|(term_expr, term_sign)| normalize_signed_add_term(ctx, term_expr, term_sign))
        .collect();
    if normalized_terms.len() < 2 {
        return None;
    }

    let lhs_terms: Vec<_> = normalized_terms
        .iter()
        .copied()
        .filter_map(|(term_expr, term_sign)| {
            (term_sign == Sign::Pos).then_some((term_expr, Sign::Pos))
        })
        .collect();
    let rhs_terms: Vec<_> = normalized_terms
        .iter()
        .copied()
        .filter_map(|(term_expr, term_sign)| {
            (term_sign == Sign::Neg).then_some((term_expr, Sign::Pos))
        })
        .collect();
    extract_cores_from_term_lists(ctx, lhs_terms, rhs_terms)
}

pub(super) fn has_plausible_shared_additive_passthrough_difference_shape(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    if let Some((lhs, rhs)) = match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) => Some((lhs, rhs)),
        Expr::Add(lhs, rhs) => match ctx.get(rhs).clone() {
            Expr::Neg(inner) => Some((lhs, inner)),
            _ => None,
        },
        _ => None,
    } {
        return AddView::from_expr(ctx, lhs).terms.len() >= 2
            && AddView::from_expr(ctx, rhs).terms.len() >= 2;
    }

    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() < 4 {
        return false;
    }

    let positive_count = terms.iter().filter(|(_, sign)| *sign == Sign::Pos).count();
    let negative_count = terms.iter().filter(|(_, sign)| *sign == Sign::Neg).count();
    positive_count >= 2 && negative_count >= 2
}

pub(super) fn try_build_structural_cancel_subset_passthrough_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 3 {
        return None;
    }

    let normalized_terms: Vec<_> = view
        .terms
        .iter()
        .copied()
        .map(|(term_expr, term_sign)| normalize_signed_add_term(ctx, term_expr, term_sign))
        .collect();

    for first_index in 0..normalized_terms.len().saturating_sub(1) {
        for second_index in (first_index + 1)..normalized_terms.len() {
            let (first_expr, first_sign) = normalized_terms[first_index];
            let (second_expr, second_sign) = normalized_terms[second_index];
            if first_sign == second_sign
                || compare_expr(ctx, first_expr, second_expr) != Ordering::Equal
            {
                continue;
            }

            let passthrough_terms: Vec<_> = normalized_terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| {
                    (index != first_index && index != second_index).then_some(term)
                })
                .collect();
            if passthrough_terms.is_empty() {
                continue;
            }

            let subset_terms = [
                normalized_terms[first_index],
                normalized_terms[second_index],
            ];
            return Some(
                Rewrite::with_local(
                    build_signed_sum_expr(ctx, &passthrough_terms),
                    "Cancelar términos iguales",
                    build_signed_sum_expr(ctx, &subset_terms),
                    ctx.num(0),
                )
                .substep(
                    "Cancelar términos iguales",
                    vec![
                        "Dos términos opuestos se anulan y solo queda el resto de la suma."
                            .to_string(),
                    ],
                ),
            );
        }
    }

    None
}

fn match_one_plus_term(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> Option<cas_ast::ExprId> {
    let Expr::Add(lhs, rhs) = ctx.get(expr) else {
        return None;
    };
    if matches!(
        ctx.get(*lhs),
        Expr::Number(n) if *n == BigRational::from_integer(1.into())
    ) {
        return Some(*rhs);
    }
    if matches!(
        ctx.get(*rhs),
        Expr::Number(n) if *n == BigRational::from_integer(1.into())
    ) {
        return Some(*lhs);
    }
    None
}

pub(super) fn match_mul_by_one_plus_term(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let Expr::Mul(lhs, rhs) = ctx.get(expr) else {
        return None;
    };
    if let Some(extra) = match_one_plus_term(ctx, *lhs) {
        return Some((*rhs, extra));
    }
    if let Some(extra) = match_one_plus_term(ctx, *rhs) {
        return Some((*lhs, extra));
    }
    None
}

pub(super) fn term_matches_structural_product(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    match ctx.get(expr) {
        Expr::Mul(a, b) => {
            (compare_expr(ctx, *a, lhs) == Ordering::Equal
                && compare_expr(ctx, *b, rhs) == Ordering::Equal)
                || (compare_expr(ctx, *a, rhs) == Ordering::Equal
                    && compare_expr(ctx, *b, lhs) == Ordering::Equal)
        }
        Expr::Pow(base, exp)
            if compare_expr(ctx, lhs, rhs) == Ordering::Equal
                && compare_expr(ctx, *base, lhs) == Ordering::Equal
                && matches!(ctx.get(*exp), Expr::Number(n) if *n == BigRational::from_integer(2.into())) =>
        {
            true
        }
        _ => false,
    }
}
