use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_nary::{add_terms_signed, Sign};
use cas_math::expr_rewrite::smart_mul;
use cas_math::factor::factor;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LogFamily {
    Ln,
    Log10,
    LogBase(ExprId),
}

#[derive(Debug, Clone)]
struct ScaledLogTerm {
    family: LogFamily,
    arg: ExprId,
    coeff: BigRational,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeriveLogSimplifyRewriteKind {
    EvenPower,
    Power,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeriveLogSimplifyRewrite {
    pub(crate) rewritten: ExprId,
    pub(crate) kind: DeriveLogSimplifyRewriteKind,
}

impl DeriveLogSimplifyRewriteKind {
    pub(crate) fn description(self) -> &'static str {
        match self {
            Self::EvenPower => "Recognize an even power inside the logarithm",
            Self::Power => "log(b, x^y) = y * log(b, x)",
        }
    }

    pub(crate) fn rule_name(self) -> &'static str {
        match self {
            Self::EvenPower => "Factor Perfect Square in Logarithm",
            Self::Power => "Evaluate Logarithms",
        }
    }
}

pub(crate) fn try_rewrite_log_simplify_target_aware(
    ctx: &mut Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveLogSimplifyRewrite> {
    if let Some(plan) =
        cas_math::logarithm_inverse_support::try_plan_log_even_power_abs_expr(ctx, expr)
    {
        if super::strong_target_match(ctx, plan.with_abs_rewrite, target_expr) {
            return Some(DeriveLogSimplifyRewrite {
                rewritten: target_expr,
                kind: DeriveLogSimplifyRewriteKind::EvenPower,
            });
        }
    }

    if try_rewrite_log_power_target_aware(ctx, expr, target_expr).is_some() {
        return Some(DeriveLogSimplifyRewrite {
            rewritten: target_expr,
            kind: DeriveLogSimplifyRewriteKind::Power,
        });
    }

    None
}

fn try_rewrite_log_power_target_aware(
    ctx: &mut Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<ExprId> {
    let source = extract_plain_log_term(ctx, expr, BigRational::one())?;
    let target = extract_scaled_log_term(ctx, target_expr, Sign::Pos)?;
    if !same_log_family(ctx, source.family, target.family) {
        return None;
    }

    if !target.coeff.is_integer() || target.coeff <= BigRational::one() {
        return None;
    }

    let Expr::Pow(base, exp) = ctx.get(source.arg) else {
        return None;
    };
    let base = *base;
    let exp = *exp;
    let Expr::Number(source_coeff) = ctx.get(exp) else {
        return None;
    };

    if compare_expr(ctx, base, target.arg) != Ordering::Equal || source_coeff != &target.coeff {
        return None;
    }

    Some(target_expr)
}

pub(crate) fn try_rewrite_log_contraction_target_aware(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let terms = add_terms_signed(ctx, expr);
    if terms.len() < 2 {
        return None;
    }

    let mut parsed_terms = Vec::with_capacity(terms.len());
    for (term, sign) in terms {
        parsed_terms.push(extract_scaled_log_term(ctx, term, sign)?);
    }

    let first_family = parsed_terms.first()?.family;
    if !parsed_terms
        .iter()
        .all(|term| same_log_family(ctx, term.family, first_family))
    {
        return None;
    }

    let mut numerator_factors = Vec::new();
    let mut denominator_factors = Vec::new();

    for term in parsed_terms {
        if term.coeff.is_zero() {
            continue;
        }
        let magnitude = term.coeff.abs();
        let factor = build_powered_log_factor(ctx, term.arg, &magnitude);
        if term.coeff.is_negative() {
            denominator_factors.push(factor);
        } else {
            numerator_factors.push(factor);
        }
    }

    if numerator_factors.is_empty() && denominator_factors.is_empty() {
        return None;
    }

    let rewritten = match (numerator_factors.is_empty(), denominator_factors.is_empty()) {
        (false, true) => {
            let product = build_product(ctx, &numerator_factors);
            make_log_expr(ctx, first_family, product)
        }
        (true, false) => {
            let product = build_product(ctx, &denominator_factors);
            let inner = make_log_expr(ctx, first_family, product);
            ctx.add(Expr::Neg(inner))
        }
        (false, false) => {
            let numerator = build_product(ctx, &numerator_factors);
            let denominator = build_product(ctx, &denominator_factors);
            let combined_arg = ctx.add(Expr::Div(numerator, denominator));
            make_log_expr(ctx, first_family, combined_arg)
        }
        (true, true) => return None,
    };

    Some(rewritten)
}

pub(crate) fn try_rewrite_log_contraction_to_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<ExprId> {
    if let Some(rewritten) = try_rewrite_log_contraction_target_aware(ctx, source_expr) {
        if super::strong_target_match(ctx, rewritten, target_expr) {
            return Some(target_expr);
        }

        if log_contracted_forms_match(ctx, rewritten, target_expr) {
            return Some(target_expr);
        }
    }

    try_rewrite_log_contraction_additive_target_aware(ctx, source_expr, target_expr)
}

pub(crate) fn try_rewrite_log_expansion_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<ExprId> {
    if let Some(contracted) = try_rewrite_log_contraction_target_aware(ctx, target_expr) {
        if super::strong_target_match(ctx, contracted, source_expr)
            || log_contracted_forms_match(ctx, source_expr, contracted)
        {
            return Some(target_expr);
        }
    }

    try_rewrite_log_expansion_additive_target_aware(ctx, source_expr, target_expr)
}

pub(crate) fn try_rewrite_log_argument_factorization_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<(ExprId, ExprId, ExprId)> {
    if let Some(rewrite) =
        try_rewrite_log_argument_factorization_core_target_aware(ctx, source_expr, target_expr)
    {
        return Some(rewrite);
    }

    try_rewrite_log_argument_factorization_additive_target_aware(ctx, source_expr, target_expr)
}

fn try_rewrite_log_argument_factorization_core_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<(ExprId, ExprId, ExprId)> {
    let source = extract_plain_log_term(ctx, source_expr, BigRational::one())?;
    for (factored_arg, focus_before, focus_after) in
        candidate_factored_log_arguments(ctx, source.arg)
    {
        let rewritten = make_log_expr(ctx, source.family, factored_arg);
        if try_rewrite_log_expansion_target_aware(ctx, rewritten, target_expr).is_some() {
            return Some((rewritten, focus_before, focus_after));
        }
    }

    None
}

fn candidate_factored_log_arguments(
    ctx: &mut Context,
    source_arg: ExprId,
) -> Vec<(ExprId, ExprId, ExprId)> {
    let mut candidates = Vec::new();

    let factored_arg = factor(ctx, source_arg);
    if compare_expr(ctx, factored_arg, source_arg) != Ordering::Equal {
        candidates.push((factored_arg, source_arg, factored_arg));
    }

    if let Expr::Div(num, den) = ctx.get(source_arg).clone() {
        let factored_num = factor(ctx, num);
        let factored_den = factor(ctx, den);

        if compare_expr(ctx, factored_num, num) != Ordering::Equal {
            let rewritten = ctx.add(Expr::Div(factored_num, den));
            candidates.push((rewritten, num, factored_num));
        }

        if compare_expr(ctx, factored_den, den) != Ordering::Equal {
            let rewritten = ctx.add(Expr::Div(num, factored_den));
            candidates.push((rewritten, den, factored_den));
        }

        if compare_expr(ctx, factored_num, num) != Ordering::Equal
            || compare_expr(ctx, factored_den, den) != Ordering::Equal
        {
            let rewritten = ctx.add(Expr::Div(factored_num, factored_den));
            if compare_expr(ctx, rewritten, source_arg) != Ordering::Equal {
                candidates.push((rewritten, source_arg, rewritten));
            }
        }
    }

    candidates
}

fn try_rewrite_log_argument_factorization_additive_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<(ExprId, ExprId, ExprId)> {
    let source_terms = signed_additive_terms(ctx, source_expr);
    let target_terms = signed_additive_terms(ctx, target_expr);

    if source_terms.len() < 2 || target_terms.len() <= source_terms.len() {
        return None;
    }

    let target_limit = 1usize.checked_shl(target_terms.len() as u32)?;
    for (source_index, source_focus) in source_terms.iter().copied().enumerate() {
        for mask in 1..target_limit {
            let selected = mask.count_ones() as usize;
            if selected < 2 || selected == target_terms.len() {
                continue;
            }

            let target_focus = build_additive_expr_from_signed_terms(
                ctx,
                target_terms
                    .iter()
                    .enumerate()
                    .filter_map(|(index, term)| ((mask & (1usize << index)) != 0).then_some(*term)),
            );

            let Some((rewritten_focus, focus_before, focus_after)) =
                try_rewrite_log_argument_factorization_core_target_aware(
                    ctx,
                    source_focus,
                    target_focus,
                )
            else {
                continue;
            };

            let source_passthrough =
                collect_passthrough_terms_excluding_index(&source_terms, source_index);
            let target_passthrough = collect_passthrough_terms(&target_terms, mask);
            if !additive_term_multiset_matches(ctx, &source_passthrough, &target_passthrough) {
                continue;
            }

            let rebuilt = build_additive_expr_from_signed_terms(
                ctx,
                source_terms.iter().enumerate().map(|(index, term)| {
                    if index == source_index {
                        rewritten_focus
                    } else {
                        *term
                    }
                }),
            );
            return Some((rebuilt, focus_before, focus_after));
        }
    }

    None
}

fn try_rewrite_log_contraction_additive_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<ExprId> {
    let source_terms = signed_additive_terms(ctx, source_expr);
    let target_terms = signed_additive_terms(ctx, target_expr);

    if source_terms.len() < 3 || target_terms.len() >= source_terms.len() {
        return None;
    }

    let source_limit = 1usize.checked_shl(source_terms.len() as u32)?;
    for mask in 1..source_limit {
        let selected = mask.count_ones() as usize;
        if selected < 2 || selected == source_terms.len() {
            continue;
        }

        let source_focus = build_additive_expr_from_signed_terms(
            ctx,
            source_terms
                .iter()
                .enumerate()
                .filter_map(|(index, term)| ((mask & (1usize << index)) != 0).then_some(*term)),
        );

        for (target_index, target_focus) in target_terms.iter().copied().enumerate() {
            if !matches_contracting_additive_log_focus(ctx, source_focus, target_focus) {
                continue;
            }

            let source_passthrough = collect_passthrough_terms(&source_terms, mask);
            let target_passthrough =
                collect_passthrough_terms_excluding_index(&target_terms, target_index);
            if additive_term_multiset_matches(ctx, &source_passthrough, &target_passthrough) {
                return Some(target_expr);
            }
        }
    }

    None
}

fn try_rewrite_log_expansion_additive_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<ExprId> {
    let source_terms = signed_additive_terms(ctx, source_expr);
    let target_terms = signed_additive_terms(ctx, target_expr);

    if source_terms.len() < 2 || target_terms.len() <= source_terms.len() {
        return None;
    }

    let target_limit = 1usize.checked_shl(target_terms.len() as u32)?;
    for (source_index, source_focus) in source_terms.iter().copied().enumerate() {
        for mask in 1..target_limit {
            let selected = mask.count_ones() as usize;
            if selected < 2 || selected == target_terms.len() {
                continue;
            }

            let target_focus = build_additive_expr_from_signed_terms(
                ctx,
                target_terms
                    .iter()
                    .enumerate()
                    .filter_map(|(index, term)| ((mask & (1usize << index)) != 0).then_some(*term)),
            );

            if try_rewrite_log_expansion_target_aware(ctx, source_focus, target_focus).is_none() {
                continue;
            }

            let source_passthrough =
                collect_passthrough_terms_excluding_index(&source_terms, source_index);
            let target_passthrough = collect_passthrough_terms(&target_terms, mask);
            if additive_term_multiset_matches(ctx, &source_passthrough, &target_passthrough) {
                return Some(target_expr);
            }
        }
    }

    None
}

fn extract_scaled_log_term(ctx: &Context, expr: ExprId, sign: Sign) -> Option<ScaledLogTerm> {
    let sign_coeff = match sign {
        Sign::Pos => BigRational::one(),
        Sign::Neg => -BigRational::one(),
    };

    match ctx.get(expr) {
        Expr::Neg(inner) => {
            let mut parsed = extract_scaled_log_term(ctx, *inner, Sign::Pos)?;
            parsed.coeff = -parsed.coeff;
            parsed.coeff *= sign_coeff;
            Some(parsed)
        }
        Expr::Mul(left, right) => {
            let (numeric, log_expr) = match (ctx.get(*left), ctx.get(*right)) {
                (Expr::Number(n), _) => (n.clone(), *right),
                (_, Expr::Number(n)) => (n.clone(), *left),
                _ => return extract_plain_log_term(ctx, expr, sign_coeff),
            };
            let mut parsed = extract_plain_log_term(ctx, log_expr, BigRational::one())?;
            parsed.coeff = sign_coeff * numeric;
            Some(parsed)
        }
        _ => extract_plain_log_term(ctx, expr, sign_coeff),
    }
}

fn extract_plain_log_term(
    ctx: &Context,
    expr: ExprId,
    coeff: BigRational,
) -> Option<ScaledLogTerm> {
    let (family, arg) = match ctx.get(expr) {
        Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, BuiltinFn::Ln) && args.len() == 1 => {
            (LogFamily::Ln, args[0])
        }
        Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, BuiltinFn::Log) => {
            match args.as_slice() {
                [arg] => (LogFamily::Log10, *arg),
                [base, arg] => (LogFamily::LogBase(*base), *arg),
                _ => return None,
            }
        }
        _ => return None,
    };

    Some(ScaledLogTerm { family, arg, coeff })
}

fn same_log_family(ctx: &Context, lhs: LogFamily, rhs: LogFamily) -> bool {
    match (lhs, rhs) {
        (LogFamily::Ln, LogFamily::Ln) => true,
        (LogFamily::Log10, LogFamily::Log10) => true,
        (LogFamily::LogBase(lhs), LogFamily::LogBase(rhs)) => {
            compare_expr(ctx, lhs, rhs) == Ordering::Equal
        }
        _ => false,
    }
}

fn log_contracted_forms_match(ctx: &mut Context, actual: ExprId, target: ExprId) -> bool {
    let Some(actual) = extract_scaled_log_term(ctx, actual, Sign::Pos) else {
        return false;
    };
    let Some(target) = extract_scaled_log_term(ctx, target, Sign::Pos) else {
        return false;
    };

    if !same_log_family(ctx, actual.family, target.family) {
        return false;
    }

    if actual.coeff.is_negative() != target.coeff.is_negative() {
        return false;
    }

    let Some(actual_arg) = fold_log_coefficient_into_argument(ctx, actual.arg, &actual.coeff.abs())
    else {
        return false;
    };
    let Some(target_arg) = fold_log_coefficient_into_argument(ctx, target.arg, &target.coeff.abs())
    else {
        return false;
    };

    log_argument_match(ctx, actual_arg, target_arg)
}

fn fold_log_coefficient_into_argument(
    ctx: &mut Context,
    arg: ExprId,
    coeff: &BigRational,
) -> Option<ExprId> {
    if coeff.is_zero() {
        return None;
    }
    if coeff.is_one() {
        return Some(arg);
    }
    if !coeff.is_integer() || coeff.is_negative() {
        return None;
    }
    Some(build_powered_log_factor(ctx, arg, coeff))
}

fn log_argument_match(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> bool {
    if super::strong_target_match(ctx, lhs, rhs) {
        return true;
    }

    let lhs = normalize_log_argument_for_contraction_match(ctx, lhs);
    let rhs = normalize_log_argument_for_contraction_match(ctx, rhs);
    compare_expr(ctx, lhs, rhs) == Ordering::Equal
}

fn normalize_log_argument_for_contraction_match(ctx: &mut Context, expr: ExprId) -> ExprId {
    let mut factors = Vec::new();
    collect_log_argument_factors(ctx, expr, 1, &mut factors);
    build_log_argument_from_factors(ctx, factors)
}

fn signed_additive_terms(ctx: &mut Context, expr: ExprId) -> Vec<ExprId> {
    add_terms_signed(ctx, expr)
        .into_iter()
        .map(|(term, sign)| apply_sign_to_term(ctx, term, sign))
        .collect()
}

fn collect_passthrough_terms(terms: &[ExprId], included_mask: usize) -> Vec<ExprId> {
    terms
        .iter()
        .enumerate()
        .filter_map(|(index, term)| ((included_mask & (1usize << index)) == 0).then_some(*term))
        .collect()
}

fn collect_passthrough_terms_excluding_index(
    terms: &[ExprId],
    excluded_index: usize,
) -> Vec<ExprId> {
    terms
        .iter()
        .enumerate()
        .filter_map(|(index, term)| (index != excluded_index).then_some(*term))
        .collect()
}

fn additive_term_multiset_matches(
    ctx: &mut Context,
    lhs_terms: &[ExprId],
    rhs_terms: &[ExprId],
) -> bool {
    if lhs_terms.len() != rhs_terms.len() {
        return false;
    }

    let mut lhs = lhs_terms.to_vec();
    let mut rhs = rhs_terms.to_vec();
    lhs.sort_by(|left, right| compare_expr(ctx, *left, *right));
    rhs.sort_by(|left, right| compare_expr(ctx, *left, *right));

    lhs.iter()
        .zip(rhs.iter())
        .all(|(left, right)| compare_expr(ctx, *left, *right) == Ordering::Equal)
}

fn build_additive_expr_from_signed_terms(
    ctx: &mut Context,
    terms: impl IntoIterator<Item = ExprId>,
) -> ExprId {
    let mut iter = terms.into_iter();
    let Some(first) = iter.next() else {
        return ctx.num(0);
    };

    iter.fold(first, |acc, term| ctx.add(Expr::Add(acc, term)))
}

fn apply_sign_to_term(ctx: &mut Context, term: ExprId, sign: Sign) -> ExprId {
    match sign {
        Sign::Pos => term,
        Sign::Neg => ctx.add(Expr::Neg(term)),
    }
}

fn matches_contracting_additive_log_focus(
    ctx: &mut Context,
    source_focus: ExprId,
    target_focus: ExprId,
) -> bool {
    try_rewrite_log_contraction_target_aware(ctx, source_focus).is_some_and(|rewritten| {
        super::strong_target_match(ctx, rewritten, target_focus)
            || log_contracted_forms_match(ctx, rewritten, target_focus)
    })
}

fn collect_log_argument_factors(
    ctx: &mut Context,
    expr: ExprId,
    mult: i64,
    out: &mut Vec<(ExprId, i64)>,
) {
    if mult == 0 {
        return;
    }

    match ctx.get(expr).clone() {
        Expr::Mul(left, right) => {
            collect_log_argument_factors(ctx, left, mult, out);
            collect_log_argument_factors(ctx, right, mult, out);
        }
        Expr::Div(num, den) => {
            collect_log_argument_factors(ctx, num, mult, out);
            collect_log_argument_factors(ctx, den, -mult, out);
        }
        Expr::Pow(base, exp) => {
            let Some(k) = integer_i64(ctx, exp) else {
                let normalized = cas_math::canonical_forms::normalize_core(ctx, expr);
                out.push((normalized, mult));
                return;
            };
            let total = mult.saturating_mul(k);
            if total == 0 {
                return;
            }

            if let Expr::Function(fn_id, args) = ctx.get(base) {
                if ctx.is_builtin(*fn_id, BuiltinFn::Abs) && args.len() == 1 && total % 2 == 0 {
                    collect_log_argument_factors(ctx, args[0], total, out);
                    return;
                }
            }

            collect_log_argument_factors(ctx, base, total, out);
        }
        _ => {
            let normalized = cas_math::canonical_forms::normalize_core(ctx, expr);
            out.push((normalized, mult));
        }
    }
}

fn build_log_argument_from_factors(ctx: &mut Context, mut factors: Vec<(ExprId, i64)>) -> ExprId {
    if factors.is_empty() {
        return ctx.num(1);
    }

    factors.sort_by(|(lhs, _), (rhs, _)| compare_expr(ctx, *lhs, *rhs));

    let mut merged: Vec<(ExprId, i64)> = Vec::with_capacity(factors.len());
    for (base, exp) in factors {
        if exp == 0 {
            continue;
        }
        if let Some((last_base, last_exp)) = merged.last_mut() {
            if compare_expr(ctx, *last_base, base) == Ordering::Equal {
                *last_exp += exp;
                continue;
            }
        }
        merged.push((base, exp));
    }

    let mut numerator = Vec::new();
    let mut denominator = Vec::new();
    for (base, exp) in merged {
        if exp > 0 {
            numerator.push(build_integer_power_factor(ctx, base, exp));
        } else if exp < 0 {
            denominator.push(build_integer_power_factor(ctx, base, -exp));
        }
    }

    match (numerator.is_empty(), denominator.is_empty()) {
        (true, true) => ctx.num(1),
        (false, true) => build_product(ctx, &numerator),
        (true, false) => {
            let one = ctx.num(1);
            let den = build_product(ctx, &denominator);
            ctx.add(Expr::Div(one, den))
        }
        (false, false) => {
            let num = build_product(ctx, &numerator);
            let den = build_product(ctx, &denominator);
            ctx.add(Expr::Div(num, den))
        }
    }
}

fn build_integer_power_factor(ctx: &mut Context, base: ExprId, exp: i64) -> ExprId {
    if exp == 1 {
        return base;
    }

    let exponent = ctx.add(Expr::Number(BigRational::from_integer(exp.into())));
    ctx.add(Expr::Pow(base, exponent))
}

fn integer_i64(ctx: &Context, expr: ExprId) -> Option<i64> {
    match ctx.get(expr) {
        Expr::Number(n) if n.is_integer() => n.to_integer().try_into().ok(),
        Expr::Neg(inner) => integer_i64(ctx, *inner).map(|v| -v),
        _ => None,
    }
}

fn make_log_expr(ctx: &mut Context, family: LogFamily, arg: ExprId) -> ExprId {
    match family {
        LogFamily::Ln => ctx.call_builtin(BuiltinFn::Ln, vec![arg]),
        LogFamily::Log10 => ctx.call_builtin(BuiltinFn::Log, vec![arg]),
        LogFamily::LogBase(base) => ctx.call_builtin(BuiltinFn::Log, vec![base, arg]),
    }
}

fn build_product(ctx: &mut Context, factors: &[ExprId]) -> ExprId {
    let mut iter = factors.iter().copied();
    let Some(first) = iter.next() else {
        return ctx.num(1);
    };
    iter.fold(first, |acc, factor| smart_mul(ctx, acc, factor))
}

fn build_powered_log_factor(ctx: &mut Context, arg: ExprId, exponent: &BigRational) -> ExprId {
    if exponent.is_one() {
        return arg;
    }

    if let Some(rewritten) = rewrite_even_abs_power(ctx, arg, exponent) {
        return rewritten;
    }

    if let Some(rewritten) = multiply_existing_power_exponent(ctx, arg, exponent) {
        return rewritten;
    }

    let exponent_id = ctx.add(Expr::Number(exponent.clone()));
    ctx.add(Expr::Pow(arg, exponent_id))
}

fn rewrite_even_abs_power(
    ctx: &mut Context,
    arg: ExprId,
    exponent: &BigRational,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(arg) else {
        return None;
    };
    let fn_id = *fn_id;
    let inner = *args.first()?;
    if !ctx.is_builtin(fn_id, BuiltinFn::Abs) || args.len() != 1 || !exponent.is_integer() {
        return None;
    }

    let integer = exponent.to_integer();
    if integer <= 0.into() || (&integer % 2) != 0.into() {
        return None;
    }

    let exponent_id = ctx.add(Expr::Number(exponent.clone()));
    Some(ctx.add(Expr::Pow(inner, exponent_id)))
}

fn multiply_existing_power_exponent(
    ctx: &mut Context,
    arg: ExprId,
    exponent: &BigRational,
) -> Option<ExprId> {
    let Expr::Pow(base, inner_exp) = ctx.get(arg) else {
        return None;
    };
    let base = *base;
    let inner_exp = *inner_exp;
    let Expr::Number(inner_number) = ctx.get(inner_exp) else {
        return None;
    };

    let multiplied = inner_number.clone() * exponent.clone();
    let exponent_id = ctx.add(Expr::Number(multiplied));
    Some(ctx.add(Expr::Pow(base, exponent_id)))
}

#[cfg(test)]
mod tests {
    use super::{
        try_rewrite_log_argument_factorization_target_aware,
        try_rewrite_log_contraction_additive_target_aware,
        try_rewrite_log_contraction_target_aware, try_rewrite_log_contraction_to_target_aware,
        try_rewrite_log_expansion_additive_target_aware, try_rewrite_log_expansion_target_aware,
        try_rewrite_log_simplify_target_aware, DeriveLogSimplifyRewriteKind,
    };
    use cas_ast::Context;
    use cas_math::semantic_equality::SemanticEqualityChecker;
    use cas_parser::parse;

    #[test]
    fn contracts_tabulated_log_targets_aware() {
        let cases = [
            ("ln(x) + ln(y)", "ln(x*y)"),
            ("ln(x) - ln(y)", "ln(x/y)"),
            ("ln(x) + ln(y) - ln(z)", "ln((x*y)/z)"),
            ("2*ln(abs(x)) + ln(y) - ln(z) - ln(t)", "ln((x^2*y)/(z*t))"),
            ("3*ln(x) + 2*ln(abs(y))", "ln(x^3*y^2)"),
            ("3*ln(x) - 2*ln(y)", "ln(x^3/y^2)"),
            ("log(2, x) - log(2, y)", "log(2, x/y)"),
            (
                "2*log(b, x) + 3*log(b, y) - 2*log(b, z) - log(b, t)",
                "log(b, (x^2*y^3)/(z^2*t))",
            ),
            ("3*log(2, x) - 2*log(2, y)", "log(2, x^3/y^2)"),
        ];

        for (expr_text, expected_text) in cases {
            let mut ctx = Context::new();
            let expr = parse(expr_text, &mut ctx).expect("expr");
            let rewritten =
                try_rewrite_log_contraction_target_aware(&mut ctx, expr).expect("rewrite");
            let expected = parse(expected_text, &mut ctx).expect("expected");
            let checker = SemanticEqualityChecker::new(&ctx);

            assert!(
                checker.are_equal(rewritten, expected),
                "expected `{expr_text}` to contract to `{expected_text}`"
            );
        }
    }

    #[test]
    fn expands_tabulated_log_targets_aware() {
        let cases = [
            ("ln(x*y)", "ln(x) + ln(y)"),
            ("ln(x/y)", "ln(x) - ln(y)"),
            ("ln((x*y)/z)", "ln(x) + ln(y) - ln(z)"),
            ("ln((x^2*y)/(z*t))", "2*ln(abs(x)) + ln(y) - ln(z) - ln(t)"),
            ("log(b, (x*y)/z)", "log(b, x) + log(b, y) - log(b, z)"),
            (
                "log(b, (x^2*y^3)/(z^2*t))",
                "2*log(b, x) + 3*log(b, y) - 2*log(b, z) - log(b, t)",
            ),
            ("ln(x^3*y^2)", "ln(x^3) + ln(y^2)"),
            ("ln((x*y)^2)", "ln(x^2)+ln(y^2)"),
            ("2*ln(abs(x*y))", "2*ln(abs(x))+2*ln(abs(y))"),
            ("log(b,(x*y)^2)", "2*log(b,x)+2*log(b,y)"),
        ];

        for (source, target) in cases {
            let mut ctx = Context::new();
            let source = parse(source, &mut ctx).expect("source");
            let target = parse(target, &mut ctx).expect("target");
            let rewritten =
                try_rewrite_log_expansion_target_aware(&mut ctx, source, target).expect("rewrite");

            assert_eq!(rewritten, target);
        }
    }

    #[test]
    fn contracts_grouped_log_targets_aware() {
        let cases = [
            ("ln(x^2)+ln(y^2)", "ln((x*y)^2)"),
            ("2*ln(abs(x))+2*ln(abs(y))", "2*ln(abs(x*y))"),
            ("2*log(b, x)+2*log(b, y)", "log(b, (x*y)^2)"),
        ];

        for (source_text, target_text) in cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewritten = try_rewrite_log_contraction_to_target_aware(&mut ctx, source, target)
                .expect("rewrite");

            assert_eq!(rewritten, target);
        }
    }

    #[test]
    fn contracts_grouped_log_targets_with_passthrough_aware() {
        let cases = [
            ("ln(x^2)+ln(y^2)+a", "ln((x*y)^2)+a"),
            ("2*ln(abs(x))+2*ln(abs(y))+a", "2*ln(abs(x*y))+a"),
            ("2*log(b, x)+2*log(b, y)+a", "log(b, (x*y)^2)+a"),
        ];

        for (source_text, target_text) in cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewritten =
                try_rewrite_log_contraction_additive_target_aware(&mut ctx, source, target)
                    .expect("rewrite");

            assert_eq!(rewritten, target);
        }
    }

    #[test]
    fn contracts_grouped_log_targets_with_passthrough_via_public_target_aware_api() {
        let mut ctx = Context::new();
        let source = parse("ln(x^2)+ln(y^2)+a", &mut ctx).expect("source");
        let target = parse("ln((x*y)^2)+a", &mut ctx).expect("target");

        let rewritten =
            try_rewrite_log_contraction_to_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewritten, target);
    }

    #[test]
    fn expands_grouped_log_targets_with_passthrough_aware() {
        let cases = [
            ("ln((x*y)^2)+a", "ln(x^2)+ln(y^2)+a"),
            ("2*ln(abs(x*y))+a", "2*ln(abs(x))+2*ln(abs(y))+a"),
            ("log(b,(x*y)^2)+a", "2*log(b,x)+2*log(b,y)+a"),
        ];

        for (source_text, target_text) in cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewritten =
                try_rewrite_log_expansion_additive_target_aware(&mut ctx, source, target)
                    .expect("rewrite");

            assert_eq!(rewritten, target);
        }
    }

    #[test]
    fn expands_grouped_log_targets_with_passthrough_via_public_target_aware_api() {
        let mut ctx = Context::new();
        let source = parse("ln((x*y)^2)+a", &mut ctx).expect("source");
        let target = parse("ln(x^2)+ln(y^2)+a", &mut ctx).expect("target");

        let rewritten =
            try_rewrite_log_expansion_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewritten, target);
    }

    #[test]
    fn rejects_log_expansion_target_that_does_not_contract_back() {
        let mut ctx = Context::new();
        let source = parse("ln(x^3*y^2)", &mut ctx).expect("source");
        let target = parse("ln(x^3) + ln(y)", &mut ctx).expect("target");

        assert!(try_rewrite_log_expansion_target_aware(&mut ctx, source, target).is_none());
    }

    #[test]
    fn rewrites_even_log_power_to_abs_target_aware() {
        let mut ctx = Context::new();
        let source = parse("ln(x^4)", &mut ctx).expect("source");
        let target = parse("4*ln(abs(x))", &mut ctx).expect("target");

        let rewrite =
            try_rewrite_log_simplify_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.rewritten, target);
        assert_eq!(rewrite.kind, DeriveLogSimplifyRewriteKind::EvenPower);
    }

    #[test]
    fn rewrites_log_power_to_scaled_log_target_aware() {
        let mut ctx = Context::new();
        let source = parse("log(b, x^3)", &mut ctx).expect("source");
        let target = parse("3*log(b, x)", &mut ctx).expect("target");

        let rewrite =
            try_rewrite_log_simplify_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.rewritten, target);
        assert_eq!(rewrite.kind, DeriveLogSimplifyRewriteKind::Power);
    }

    #[test]
    fn factors_log_argument_difference_of_squares_with_passthrough_target_aware() {
        let mut ctx = Context::new();
        let source = parse("log(x^2-y^2)+a", &mut ctx).expect("source");
        let target = parse("log(x-y)+log(x+y)+a", &mut ctx).expect("target");

        let (rewritten, focus_before, focus_after) =
            try_rewrite_log_argument_factorization_target_aware(&mut ctx, source, target)
                .expect("rewrite");

        let expected_rewritten = parse("log((x-y)*(x+y))+a", &mut ctx).expect("expected");
        let expected_before = parse("x^2-y^2", &mut ctx).expect("expected");
        let expected_after = parse("(x-y)*(x+y)", &mut ctx).expect("expected");
        let checker = SemanticEqualityChecker::new(&ctx);

        assert!(checker.are_equal(rewritten, expected_rewritten));
        assert_eq!(focus_before, expected_before);
        assert!(checker.are_equal(focus_after, expected_after));
    }

    #[test]
    fn factors_log_quotient_argument_difference_of_squares_target_aware() {
        let mut ctx = Context::new();
        let source = parse("log((x^2-y^2)/(u*v))", &mut ctx).expect("source");
        let target = parse("log(x-y)+log(x+y)-log(u)-log(v)", &mut ctx).expect("target");

        let (rewritten, focus_before, focus_after) =
            try_rewrite_log_argument_factorization_target_aware(&mut ctx, source, target)
                .expect("rewrite");

        let expected_rewritten = parse("log(((x-y)*(x+y))/(u*v))", &mut ctx).expect("expected");
        let expected_before = parse("x^2-y^2", &mut ctx).expect("expected");
        let expected_after = parse("(x-y)*(x+y)", &mut ctx).expect("expected");
        let checker = SemanticEqualityChecker::new(&ctx);

        assert!(checker.are_equal(rewritten, expected_rewritten));
        assert_eq!(focus_before, expected_before);
        assert!(checker.are_equal(focus_after, expected_after));
    }
}
