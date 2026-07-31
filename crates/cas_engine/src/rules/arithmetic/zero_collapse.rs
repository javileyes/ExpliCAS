//! `arithmetic`: familia `zero_collapse`.
//!
//! Ver la cabecera de `arithmetic.rs` para el contexto.

use super::*;

pub(super) fn is_default_simplified_zero_expr(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_zero())
}

pub(super) fn build_small_zero_partition_expr(
    ctx: &mut cas_ast::Context,
    terms: &[(cas_ast::ExprId, Sign)],
) -> cas_ast::ExprId {
    let mut positive_terms = Vec::new();
    let mut negative_terms = Vec::new();
    for (expr, sign) in terms.iter().copied() {
        match sign {
            Sign::Pos => positive_terms.push(expr),
            Sign::Neg => negative_terms.push(expr),
        }
    }

    if positive_terms.len() == 1
        && negative_terms.len() >= 2
        && expr_is_sin_ratio_term(ctx, positive_terms[0])
    {
        let kernel_sum = build_unsigned_sum_expr(ctx, &negative_terms);
        return ctx.add(Expr::Sub(positive_terms[0], kernel_sum));
    }
    if negative_terms.len() == 1
        && positive_terms.len() >= 2
        && expr_is_sin_ratio_term(ctx, negative_terms[0])
    {
        let kernel_sum = build_unsigned_sum_expr(ctx, &positive_terms);
        return ctx.add(Expr::Sub(kernel_sum, negative_terms[0]));
    }

    build_signed_sum_expr(ctx, terms)
}

pub(super) fn try_build_fast_small_polynomial_expansion_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    let normalized_terms: Vec<_> = view
        .terms
        .iter()
        .copied()
        .map(|(term_expr, term_sign)| normalize_signed_add_term(ctx, term_expr, term_sign))
        .collect();

    for focus_index in 0..normalized_terms.len() {
        let (focus_expr, focus_sign) = normalized_terms[focus_index];
        let policy = SmallPowExpandPolicy {
            max_vars: 3,
            ..SmallPowExpandPolicy::default()
        };
        let Some(expanded_positive) = try_expand_small_pow_sum_expr(ctx, focus_expr, policy) else {
            continue;
        };
        if compare_expr(ctx, expanded_positive, focus_expr) == Ordering::Equal {
            continue;
        }

        let normalized_expanded_positive = normalize_additive_scope_expr(ctx, expanded_positive);
        let expected_remaining: Vec<_> = AddView::from_expr(ctx, normalized_expanded_positive)
            .terms
            .iter()
            .copied()
            .map(|(term_expr, term_sign)| {
                let (term_expr, term_sign) =
                    normalize_signed_add_term_for_fast_match(ctx, term_expr, term_sign);
                let global_sign = if focus_sign == Sign::Pos {
                    term_sign.negate()
                } else {
                    term_sign
                };
                (term_expr, global_sign)
            })
            .collect();

        let remaining_terms: Vec<_> = normalized_terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, term)| (index != focus_index).then_some(term))
            .collect();

        if !signed_terms_match_multiset(ctx, &remaining_terms, &expected_remaining) {
            continue;
        }

        return Some(Rewrite::with_local(
            ctx.num(0),
            "Expand binomial/trinomial power",
            focus_expr,
            apply_sign_to_expr(ctx, i64::from(focus_sign.to_i32()), expanded_positive),
        ));
    }

    None
}

pub(super) fn maybe_small_polynomial_expand_zero_candidate(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    fn walk(ctx: &cas_ast::Context, expr: cas_ast::ExprId, depth: usize) -> bool {
        if depth > 4 {
            return false;
        }

        match ctx.get(expr) {
            Expr::Pow(base, exp) => {
                if matches!(ctx.get(*base), Expr::Add(_, _) | Expr::Sub(_, _)) {
                    if let Some(power) = extract_i64_integer(ctx, *exp) {
                        let base_term_count = AddView::from_expr(ctx, *base).terms.len();
                        if (2..=6).contains(&power) && (2..=3).contains(&base_term_count) {
                            return true;
                        }
                    }
                }

                walk(ctx, *base, depth + 1) || walk(ctx, *exp, depth + 1)
            }
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Div(lhs, rhs) => walk(ctx, *lhs, depth + 1) || walk(ctx, *rhs, depth + 1),
            Expr::Neg(inner) | Expr::Hold(inner) => walk(ctx, *inner, depth + 1),
            Expr::Function(_, args) => args.iter().copied().any(|arg| walk(ctx, arg, depth + 1)),
            Expr::Matrix { data, .. } => {
                data.iter().copied().any(|item| walk(ctx, item, depth + 1))
            }
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => false,
        }
    }

    walk(ctx, expr, 0)
}

pub(crate) fn try_build_small_direct_zero_core_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    let term_count = view.terms.len();
    if !(2..=4).contains(&term_count) {
        return None;
    }
    if !additive_scope_has_negative_term(ctx, expr) {
        return None;
    }
    if let Some(rewrite) = try_build_exact_opposite_pair_zero_rewrite(ctx, expr) {
        return Some(rewrite);
    }

    let solve_prep_candidate = maybe_solve_prep_exact_additive_candidate(ctx, expr);
    let integrate_prep_candidate = maybe_integrate_prep_exact_additive_candidate(ctx, expr);
    let unit_fraction_trig_denominator_candidate =
        maybe_unit_fraction_trig_denominator_equivalence_zero_candidate(ctx, expr);
    let direct_zero_identity_candidate = has_direct_small_zero_identity_candidate(ctx, expr);
    if term_count > 3 && !solve_prep_candidate && !integrate_prep_candidate {
        return None;
    }
    if term_count == 3 {
        if let Some(rewrite) = try_build_small_structural_poly_zero_core_rewrite(ctx, expr) {
            return Some(rewrite);
        }
    }
    let has_trig_or_hyper = expr_contains_any_builtin(
        ctx,
        expr,
        &[
            BuiltinFn::Sin,
            BuiltinFn::Cos,
            BuiltinFn::Tan,
            BuiltinFn::Cot,
            BuiltinFn::Sec,
            BuiltinFn::Csc,
            BuiltinFn::Sinh,
            BuiltinFn::Cosh,
            BuiltinFn::Tanh,
        ],
    );
    let has_division = expr_contains_division_node(ctx, expr);
    let has_supported_shape = solve_prep_candidate
        || integrate_prep_candidate
        || unit_fraction_trig_denominator_candidate
        || direct_zero_identity_candidate
        || ((has_trig_or_hyper
            || has_division
            || expr_contains_sqrt_or_half_power(ctx, expr)
            || expr_contains_factorial_call(ctx, expr))
            && !(has_trig_or_hyper && has_division));
    if !has_supported_shape {
        return None;
    }

    if solve_prep_candidate {
        if let Some(rewrite) = try_build_fast_solve_prep_exact_zero_scope_rewrite(ctx, expr) {
            return Some(rewrite);
        }
    }
    if integrate_prep_candidate {
        if let Some(rewrite) = try_build_direct_integrate_prep_exact_zero_scope_rewrite(ctx, expr) {
            return Some(rewrite);
        }
        if let Some(rewrite) = try_build_exact_dirichlet_zero_scope_rewrite(ctx, expr) {
            return Some(rewrite);
        }
    }
    if has_trig_or_hyper {
        if let Some(rewrite) =
            try_build_exact_trig_product_to_sum_sin_sin_three_term_zero_rewrite(ctx, expr)
        {
            return Some(rewrite);
        }
        if let Some(rewrite) = try_build_small_tan_cot_product_zero_core_rewrite(ctx, expr) {
            return Some(rewrite);
        }
        if let Some(rewrite) = try_build_small_tan_cot_sec_csc_zero_core_rewrite(ctx, expr) {
            return Some(rewrite);
        }
        if let Some(rewrite) = try_build_small_sec_tan_pythagorean_zero_core_rewrite(ctx, expr) {
            return Some(rewrite);
        }
        if let Some(rewrite) = try_build_small_csc_cot_pythagorean_zero_core_rewrite(ctx, expr) {
            return Some(rewrite);
        }
    }
    if expr_contains_division_node(ctx, expr) {
        if let Some(rewrite) =
            try_build_unit_fraction_trig_denominator_equivalence_zero_core_rewrite(ctx, expr)
        {
            return Some(rewrite);
        }
        if let Some(rewrite) =
            try_build_direct_reciprocal_sum_difference_nested_fraction_zero_scope_rewrite(ctx, expr)
        {
            return Some(rewrite);
        }
        if let Some(rewrite) = try_build_direct_nested_fraction_zero_scope_rewrite(ctx, expr) {
            return Some(rewrite);
        }
        if let Some(rewrite) =
            try_build_small_difference_square_partial_fraction_zero_core_rewrite(ctx, expr)
        {
            return Some(rewrite);
        }
        if let Some(rewrite) =
            try_build_small_rationalized_sum_of_sqrts_zero_core_rewrite(ctx, expr)
        {
            return Some(rewrite);
        }
    }
    if expr_contains_factorial_call(ctx, expr) {
        if let Some(rewrite) = try_build_small_factorial_zero_core_rewrite(ctx, expr) {
            return Some(rewrite);
        }
    }
    let direct_add_term_count = AddView::from_expr(ctx, expr).terms.len();
    if (3..=4).contains(&direct_add_term_count) {
        if let Some(rewrite) =
            try_build_direct_reciprocal_half_power_linear_residual_zero_rewrite(ctx, expr)
        {
            return Some(rewrite);
        }
    }
    if expr_contains_sqrt_or_half_power(ctx, expr) {
        if let Some(rewrite) = try_build_small_symbolic_root_denesting_zero_core_rewrite(ctx, expr)
        {
            return Some(rewrite);
        }
        if let Some(rewrite) = try_build_small_odd_half_power_zero_core_rewrite(ctx, expr) {
            return Some(rewrite);
        }
    }

    let rewrite = try_build_exact_zero_identity_rewrite_direct_impl(ctx, expr, false)?;
    let zero = ctx.num(0);
    (compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal).then_some(rewrite)
}

fn try_build_exact_opposite_pair_zero_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return None;
    }
    let (lhs_expr, lhs_sign) = normalize_signed_add_term(ctx, terms[0].0, terms[0].1);
    let (rhs_expr, rhs_sign) = normalize_signed_add_term(ctx, terms[1].0, terms[1].1);
    if lhs_sign == rhs_sign || !exprs_match_for_cancellation(ctx, lhs_expr, rhs_expr) {
        return None;
    }

    Some(Rewrite::with_local(
        ctx.num(0),
        RULE_CANCEL_EXACT_ADDITIVE_PAIRS,
        expr,
        ctx.num(0),
    ))
}

fn has_plausible_small_zero_partition_term_shape(terms: &[(cas_ast::ExprId, Sign)]) -> bool {
    (2..=4).contains(&terms.len()) && terms.iter().any(|(_, sign)| *sign == Sign::Neg)
}

fn has_plausible_small_zero_partition_core_shape(
    ctx: &mut cas_ast::Context,
    candidate: cas_ast::ExprId,
) -> bool {
    let terms = AddView::from_expr(ctx, candidate).terms;
    has_plausible_small_zero_partition_term_shape(&terms)
}

pub(super) fn small_zero_additive_combination_supported_partition_core(
    ctx: &mut cas_ast::Context,
    candidate: cas_ast::ExprId,
) -> bool {
    let terms = AddView::from_expr(ctx, candidate).terms;
    let has_trig_or_hyper = expr_contains_any_builtin(
        ctx,
        candidate,
        &[
            BuiltinFn::Sin,
            BuiltinFn::Cos,
            BuiltinFn::Tan,
            BuiltinFn::Cot,
            BuiltinFn::Sec,
            BuiltinFn::Csc,
            BuiltinFn::Sinh,
            BuiltinFn::Cosh,
            BuiltinFn::Tanh,
        ],
    );
    let has_division = expr_contains_division_node(ctx, candidate);
    let has_radical = expr_contains_sqrt_or_half_power(ctx, candidate);
    let has_factorial = expr_contains_factorial_call(ctx, candidate);
    let has_solve_prep = maybe_solve_prep_exact_additive_candidate(ctx, candidate);
    let has_integrate_prep = maybe_integrate_prep_exact_additive_candidate(ctx, candidate);
    let has_log = expr_contains_any_builtin(
        ctx,
        candidate,
        &[
            BuiltinFn::Ln,
            BuiltinFn::Log,
            BuiltinFn::Log2,
            BuiltinFn::Log10,
        ],
    );
    let has_structural_poly_zero = terms.len() == 3
        && try_build_small_structural_poly_zero_core_rewrite(ctx, candidate).is_some();
    let has_direct_zero_identity = has_direct_small_zero_identity_candidate(ctx, candidate)
        && try_build_exact_zero_identity_rewrite_direct_impl(ctx, candidate, false).is_some();
    let has_log_zero_identity = has_log
        && (maybe_log_product_power_zero_candidate(ctx, candidate)
            || maybe_log_abs_mul_div_zero_candidate(ctx, candidate))
        && try_build_exact_zero_identity_rewrite_direct_impl(ctx, candidate, false).is_some();
    let has_top_level_division_with_radical_denominator = terms.iter().any(|(term_expr, _)| {
        let Some((_, denominator)) = as_div(ctx, *term_expr) else {
            return false;
        };
        expr_contains_sqrt_or_half_power(ctx, denominator)
    });

    if !has_trig_or_hyper && has_radical && has_log {
        return false;
    }

    if !has_trig_or_hyper
        && has_radical
        && has_division
        && !has_solve_prep
        && !has_integrate_prep
        && !has_top_level_division_with_radical_denominator
    {
        return false;
    }

    if !has_trig_or_hyper
        && has_division
        && !has_radical
        && !has_factorial
        && !has_solve_prep
        && !has_integrate_prep
    {
        let all_terms_are_division_like = terms
            .iter()
            .all(|(term_expr, _)| expr_contains_division_node(ctx, *term_expr));
        if !all_terms_are_division_like {
            return false;
        }
    }

    (has_trig_or_hyper
        || has_division
        || has_radical
        || has_factorial
        || has_solve_prep
        || has_integrate_prep
        || has_structural_poly_zero
        || has_log_zero_identity
        || has_direct_zero_identity)
        && (!has_trig_or_hyper || !has_division || has_integrate_prep || has_direct_zero_identity)
}

fn direct_small_zero_additive_combination_terms(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Vec<(cas_ast::ExprId, Sign)> {
    let terms = AddView::from_expr(ctx, expr).terms;
    let mut flattened = Vec::with_capacity(terms.len());

    for (term_expr, term_sign) in terms {
        let (term_expr, term_sign) = normalize_signed_add_term(ctx, term_expr, term_sign);
        let nested_terms = AddView::from_expr(ctx, term_expr).terms;
        if nested_terms.len() <= 1 {
            flattened.push((term_expr, term_sign));
            continue;
        }

        flattened.extend(nested_terms.into_iter().map(|(nested_expr, nested_sign)| {
            (nested_expr, combine_signs(term_sign, nested_sign))
        }));
    }

    flattened
}

fn has_direct_small_zero_identity_candidate(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    AddView::from_expr(ctx, expr).terms.len() <= 4
        && expr_contains_division_node(ctx, expr)
        && expr_contains_any_builtin(ctx, expr, &[BuiltinFn::Sin, BuiltinFn::Cos])
}

pub(crate) fn maybe_direct_small_zero_additive_combination_candidate(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let terms = direct_small_zero_additive_combination_terms(ctx, expr);
    if !(4..=small_zero_additive_combination_max_terms(ctx, expr)).contains(&terms.len()) {
        return false;
    }

    for subset_len in 2..=4 {
        if subset_len >= terms.len() {
            continue;
        }
        let mut stack = vec![(0usize, Vec::<usize>::new())];
        while let Some((next_index, chosen)) = stack.pop() {
            if chosen.len() == subset_len {
                if !chosen.contains(&0) {
                    continue;
                }

                let first_terms: Vec<_> = terms
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(index, term)| chosen.contains(&index).then_some(term))
                    .collect();
                let second_terms: Vec<_> = terms
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(index, term)| (!chosen.contains(&index)).then_some(term))
                    .collect();
                if !(2..=4).contains(&second_terms.len()) {
                    continue;
                }
                if !has_plausible_small_zero_partition_term_shape(&first_terms)
                    || !has_plausible_small_zero_partition_term_shape(&second_terms)
                {
                    continue;
                }

                let first_expr = build_small_zero_partition_expr(ctx, &first_terms);
                let second_expr = build_small_zero_partition_expr(ctx, &second_terms);
                if small_zero_additive_combination_supported_partition_core(ctx, first_expr)
                    && small_zero_additive_combination_supported_partition_core(ctx, second_expr)
                {
                    return true;
                }
                continue;
            }

            let remaining_slots = subset_len - chosen.len();
            let max_start = terms.len().saturating_sub(remaining_slots);
            for index in (next_index..=max_start).rev() {
                let mut next_chosen = chosen.clone();
                next_chosen.push(index);
                stack.push((index + 1, next_chosen));
            }
        }
    }

    false
}

pub(super) fn small_zero_additive_combination_max_terms(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> usize {
    if expr_contains_division_node(ctx, expr)
        && expr_contains_any_builtin(ctx, expr, &[BuiltinFn::Sin, BuiltinFn::Cos])
    {
        7
    } else {
        6
    }
}

pub(super) fn try_build_small_structural_poly_zero_core_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    try_build_structural_three_term_poly_zero_rewrite(ctx, expr)
        .or_else(|| try_build_structural_difference_squares_zero_rewrite(ctx, expr))
        .or_else(|| try_build_structural_common_factor_zero_rewrite(ctx, expr))
        .or_else(|| try_build_exact_zero_identity_rewrite_direct_impl(ctx, expr, false))
}

fn try_build_structural_common_factor_zero_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 3 {
        return None;
    }

    for target_index in 0..terms.len() {
        let (target_expr, target_sign) = terms[target_index];
        let Some((common_factor, (left_term, left_sign), (right_term, right_sign))) =
            extract_binary_product_with_sum_factor(ctx, target_expr)
        else {
            continue;
        };
        let others: Vec<_> = terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, term)| (index != target_index).then_some(term))
            .collect();
        let expected_left_sign = combine_signs(target_sign, left_sign).negate();
        let expected_right_sign = combine_signs(target_sign, right_sign).negate();

        let first_matches_left = others[0].1 == expected_left_sign
            && others[1].1 == expected_right_sign
            && term_matches_binary_product(ctx, others[0].0, common_factor, left_term)
            && term_matches_binary_product(ctx, others[1].0, common_factor, right_term);
        let first_matches_right = others[0].1 == expected_right_sign
            && others[1].1 == expected_left_sign
            && term_matches_binary_product(ctx, others[0].0, common_factor, right_term)
            && term_matches_binary_product(ctx, others[1].0, common_factor, left_term);
        if first_matches_left || first_matches_right {
            return Some(
                Rewrite::with_local(ctx.num(0), "Common Factor", expr, ctx.num(0)).substep(
                    "Reconocer factor común",
                    vec![
                        "Los dos productos comparten factor y coinciden con su forma factorizada, así que el residuo vale 0.".to_string(),
                    ],
                ),
            );
        }
    }

    None
}

pub(crate) fn try_build_direct_small_zero_additive_combination_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let build_combined_rewrite = |ctx: &mut cas_ast::Context,
                                  first_expr: cas_ast::ExprId,
                                  second_expr: cas_ast::ExprId| {
        if !has_plausible_small_zero_partition_core_shape(ctx, first_expr)
            || !has_plausible_small_zero_partition_core_shape(ctx, second_expr)
            || !small_zero_additive_combination_supported_partition_core(ctx, first_expr)
            || !small_zero_additive_combination_supported_partition_core(ctx, second_expr)
        {
            return None;
        }

        let first_zero_rewrite = try_build_small_direct_zero_core_rewrite(ctx, first_expr)?;
        let second_zero_rewrite = try_build_small_direct_zero_core_rewrite(ctx, second_expr)?;
        let description = if first_zero_rewrite.description == second_zero_rewrite.description {
            first_zero_rewrite.description.clone()
        } else {
            "Exact Zero Core Composition".into()
        };

        let mut rewrite = Rewrite::with_local(ctx.num(0), description, expr, ctx.num(0))
                .requires_all(first_zero_rewrite.required_conditions.clone())
                .requires_all(second_zero_rewrite.required_conditions.clone())
                .assume_all(first_zero_rewrite.assumption_events.clone())
                .assume_all(second_zero_rewrite.assumption_events.clone())
                .substep(
                    "Anular el primer core",
                    vec![
                        "El primer término compuesto es una identidad exacta pequeña, así que vale 0."
                            .to_string(),
                    ],
                )
                .substep(
                    "Anular el segundo core",
                    vec![
                        "El segundo término compuesto también es una identidad exacta pequeña, así que vale 0."
                            .to_string(),
                    ],
                )
                .substep(
                    "Sumar ceros",
                    vec!["Tras sustituir ambos cores por 0, toda la combinación vale 0.".to_string()],
                );

        if let Some(poly_proof) = first_zero_rewrite.poly_proof.clone() {
            rewrite = rewrite.poly_proof(poly_proof);
        } else if let Some(poly_proof) = second_zero_rewrite.poly_proof.clone() {
            rewrite = rewrite.poly_proof(poly_proof);
        }

        Some(rewrite)
    };

    if let Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) = ctx.get(expr) {
        if let Some(rewrite) = build_combined_rewrite(ctx, *lhs, *rhs) {
            return Some(rewrite);
        }
    }

    if !maybe_direct_small_zero_additive_combination_candidate(ctx, expr) {
        return None;
    }

    let terms = direct_small_zero_additive_combination_terms(ctx, expr);
    if !(4..=small_zero_additive_combination_max_terms(ctx, expr)).contains(&terms.len()) {
        return None;
    }

    for subset_len in 2..=4 {
        if subset_len >= terms.len() {
            continue;
        }
        let mut stack = vec![(0usize, Vec::<usize>::new())];
        while let Some((next_index, chosen)) = stack.pop() {
            if chosen.len() == subset_len {
                if !chosen.contains(&0) {
                    continue;
                }

                let first_terms: Vec<_> = terms
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(index, term)| chosen.contains(&index).then_some(term))
                    .collect();
                let second_terms: Vec<_> = terms
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(index, term)| (!chosen.contains(&index)).then_some(term))
                    .collect();
                if !(2..=4).contains(&second_terms.len()) {
                    continue;
                }
                if !has_plausible_small_zero_partition_term_shape(&first_terms)
                    || !has_plausible_small_zero_partition_term_shape(&second_terms)
                {
                    continue;
                }

                let first_expr = build_small_zero_partition_expr(ctx, &first_terms);
                let second_expr = build_small_zero_partition_expr(ctx, &second_terms);
                if let Some(rewrite) = build_combined_rewrite(ctx, first_expr, second_expr) {
                    return Some(rewrite);
                }
                continue;
            }

            let remaining_slots = subset_len - chosen.len();
            let max_start = terms.len().saturating_sub(remaining_slots);
            for index in (next_index..=max_start).rev() {
                let mut next_chosen = chosen.clone();
                next_chosen.push(index);
                stack.push((index + 1, next_chosen));
            }
        }
    }

    None
}

pub(super) fn maybe_exact_zero_additive_candidate(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let term_count = AddView::from_expr(ctx, expr).terms.len();
    let small_polynomial_expand_candidate = maybe_small_polynomial_expand_zero_candidate(ctx, expr);
    if small_polynomial_expand_candidate {
        return (2..=20).contains(&term_count);
    }
    if term_count > 6 && term_count <= small_zero_additive_combination_max_terms(ctx, expr) {
        return true;
    }
    if !(2..=6).contains(&term_count) {
        return false;
    }

    let solve_prep_candidate = maybe_solve_prep_exact_additive_candidate(ctx, expr);
    let trig_power_reduction_candidate = maybe_trig_power_reduction_zero_candidate(ctx, expr);
    let trig_candidate = maybe_trig_sum_to_product_zero_candidate(ctx, expr)
        || maybe_trig_square_zero_candidate(ctx, expr)
        || maybe_trig_phase_shift_zero_candidate(ctx, expr)
        || expr_contains_any_builtin(
            ctx,
            expr,
            &[
                BuiltinFn::Sin,
                BuiltinFn::Cos,
                BuiltinFn::Tan,
                BuiltinFn::Cot,
                BuiltinFn::Sec,
                BuiltinFn::Csc,
            ],
        )
        || expr_contains_any_builtin(
            ctx,
            expr,
            &[BuiltinFn::Sinh, BuiltinFn::Cosh, BuiltinFn::Tanh],
        )
        || maybe_hyperbolic_angle_sum_diff_zero_candidate(ctx, expr)
        || maybe_hyperbolic_pythagorean_factor_zero_candidate(ctx, expr)
        || expr_contains_any_builtin(
            ctx,
            expr,
            &[
                BuiltinFn::Ln,
                BuiltinFn::Log,
                BuiltinFn::Log10,
                BuiltinFn::Abs,
            ],
        );

    if term_count == 2 {
        return trig_power_reduction_candidate || solve_prep_candidate;
    }

    trig_candidate || solve_prep_candidate
}

pub(super) fn maybe_exact_zero_common_scaled_difference_candidate(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let term_count = AddView::from_expr(ctx, expr).terms.len();
    if !(2..=4).contains(&term_count) {
        return false;
    }
    if term_count > 2 && additive_view_has_exact_duplicate_or_canceling_terms(ctx, expr) {
        return false;
    }

    if term_count == 2 {
        let view = AddView::from_expr(ctx, expr);
        let mut denominator = None;
        let mut all_terms_are_divisions = true;
        for (term_expr, _term_sign) in view.terms {
            let Expr::Div(_num, den) = ctx.get(term_expr) else {
                all_terms_are_divisions = false;
                break;
            };
            if let Some(existing_den) = denominator {
                if compare_expr(ctx, *den, existing_den) != Ordering::Equal {
                    all_terms_are_divisions = false;
                    break;
                }
            } else {
                denominator = Some(*den);
            }
        }

        if all_terms_are_divisions && denominator.is_some() {
            return true;
        }

        if expr_contains_sqrt_or_half_power(ctx, expr) && expr_contains_division_node(ctx, expr) {
            return true;
        }
    }

    expr_contains_any_builtin(
        ctx,
        expr,
        &[
            BuiltinFn::Sin,
            BuiltinFn::Cos,
            BuiltinFn::Tan,
            BuiltinFn::Cot,
            BuiltinFn::Sec,
            BuiltinFn::Csc,
            BuiltinFn::Sinh,
            BuiltinFn::Cosh,
            BuiltinFn::Tanh,
            BuiltinFn::Ln,
            BuiltinFn::Log,
            BuiltinFn::Log2,
            BuiltinFn::Log10,
            BuiltinFn::Abs,
        ],
    ) || maybe_solve_prep_common_scale_candidate(ctx, expr)
}

pub(super) fn try_build_exact_zero_identity_rewrite_direct_impl(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    allow_direct_small_zero_combination: bool,
) -> Option<Rewrite> {
    let parent_ctx = ParentContext::root().with_domain_mode(crate::DomainMode::Generic);
    let zero = ctx.num(0);
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    let expr_sample = profiling.then(|| render_expr_for_orchestrator_profile(ctx, expr));
    let two_term_cores = extract_two_term_core_difference(ctx, expr);
    let direct_term_count = AddView::from_expr(ctx, expr).terms.len();
    let has_direct_trig_expr = expr_contains_direct_trig_builtin(ctx, expr);
    if expr_contains_direct_hyperbolic_builtin(ctx, expr)
        && reject_linear_hyperbolic_combination_before_zero_scope(ctx, expr)
    {
        return None;
    }

    let try_rule = |ctx: &mut cas_ast::Context, rewrite: Option<Rewrite>| -> Option<Rewrite> {
        let rewrite = rewrite?;
        exprs_match_after_default_simplify(ctx, rewrite.final_expr(), zero).then_some(rewrite)
    };

    if let Some((lhs_core, rhs_core)) = two_term_cores {
        if has_direct_trig_builtin_on_either_side(ctx, lhs_core, rhs_core) {
            if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
                profiling,
                "rule.direct_identity.try.two_term_trig_power_reduction_early",
                &expr_sample,
                || {
                    try_build_direct_trig_power_reduction_equivalence_rewrite(
                        ctx, lhs_core, rhs_core,
                    )
                },
            ) {
                return Some(rewrite);
            }
        }
    }
    if has_direct_trig_expr && maybe_exact_trig_equivalence_zero_scope_candidate(ctx, expr) {
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.sin_sin_three_term_early",
            &expr_sample,
            || try_build_exact_trig_product_to_sum_sin_sin_three_term_zero_rewrite(ctx, expr),
        ) {
            return Some(rewrite);
        }
    }
    if (3..=4).contains(&direct_term_count) {
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.zero_scope_reciprocal_half_power_linear_residual",
            &expr_sample,
            || try_build_direct_reciprocal_half_power_linear_residual_zero_rewrite(ctx, expr),
        ) {
            return Some(rewrite);
        }
    }
    if expr_contains_sqrt_or_half_power(ctx, expr) {
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.symbolic_root_denesting",
            &expr_sample,
            || try_build_small_symbolic_root_denesting_zero_core_rewrite(ctx, expr),
        ) {
            return Some(rewrite);
        }
    }

    if allow_direct_small_zero_combination
        && maybe_direct_small_zero_additive_combination_candidate(ctx, expr)
    {
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.small_zero_additive_combination",
            &expr_sample,
            || try_build_direct_small_zero_additive_combination_rewrite(ctx, expr),
        ) {
            return Some(rewrite);
        }
    }
    if maybe_fraction_telescoping_zero_scope_candidate(ctx, expr) {
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.fraction_telescoping_zero_scope",
            &expr_sample,
            || try_build_direct_fraction_telescoping_zero_scope_rewrite(ctx, expr),
        ) {
            return Some(rewrite);
        }
    }

    if let Some((lhs_core, rhs_core)) = two_term_cores {
        if expr_contains_named_function_for_profile(ctx, lhs_core, &["sum"])
            || expr_contains_named_function_for_profile(ctx, rhs_core, &["sum"])
        {
            if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
                profiling,
                "rule.direct_identity.try.two_term_finite_sum",
                &expr_sample,
                || try_build_direct_finite_sum_equivalence_rewrite(ctx, lhs_core, rhs_core),
            ) {
                return Some(rewrite);
            }
        }
    }

    if let Some((lhs_core, rhs_core)) = two_term_cores {
        let has_direct_trig_core = has_direct_trig_builtin_on_either_side(ctx, lhs_core, rhs_core);
        let maybe_tanh_exp_pair =
            maybe_two_term_tanh_exp_equivalence_candidate(ctx, lhs_core, rhs_core);
        let maybe_trig_product_to_sum_pair =
            maybe_two_term_trig_product_to_sum_equivalence_candidate(ctx, lhs_core, rhs_core);
        if has_direct_trig_core {
            if maybe_trig_product_to_sum_pair {
                if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
                    profiling,
                    "rule.direct_identity.try.two_term_trig_product_to_sum",
                    &expr_sample,
                    || {
                        try_build_direct_trig_product_to_sum_equivalence_rewrite(
                            ctx, lhs_core, rhs_core,
                        )
                    },
                ) {
                    return Some(rewrite);
                }
            }
            if maybe_two_term_trig_sum_to_product_equivalence_candidate(ctx, lhs_core, rhs_core) {
                if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
                    profiling,
                    "rule.direct_identity.try.two_term_trig_sum_to_product",
                    &expr_sample,
                    || {
                        try_build_direct_trig_sum_to_product_equivalence_rewrite(
                            ctx, lhs_core, rhs_core,
                        )
                    },
                ) {
                    return Some(rewrite);
                }
            }
        }
        if maybe_tanh_exp_pair {
            if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
                profiling,
                "rule.direct_identity.try.two_term_tanh_exp",
                &expr_sample,
                || {
                    try_build_direct_tanh_exp_definition_equivalence_rewrite(
                        ctx, lhs_core, rhs_core,
                    )
                },
            ) {
                return Some(rewrite);
            }
        }
        if has_direct_trig_core {
            if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
                profiling,
                "rule.direct_identity.try.two_term_trig_power_reduction",
                &expr_sample,
                || {
                    try_build_direct_trig_power_reduction_equivalence_rewrite(
                        ctx, lhs_core, rhs_core,
                    )
                },
            ) {
                return Some(rewrite);
            }
            if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
                profiling,
                "rule.direct_identity.try.two_term_mixed_double_angle_poly",
                &expr_sample,
                || {
                    try_build_direct_trig_mixed_double_angle_polynomial_equivalence_rewrite(
                        ctx, lhs_core, rhs_core,
                    )
                },
            ) {
                return Some(rewrite);
            }
            if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
                profiling,
                "rule.direct_identity.try.two_term_trig_square",
                &expr_sample,
                || try_build_direct_trig_square_equivalence_rewrite(ctx, lhs_core, rhs_core),
            ) {
                return Some(rewrite);
            }
        }
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.two_term_sinh_cubic",
            &expr_sample,
            || {
                try_build_direct_hyperbolic_sinh_cubic_polynomial_equivalence_rewrite(
                    ctx, lhs_core, rhs_core,
                )
            },
        ) {
            return Some(rewrite);
        }
        if expr_contains_hyperbolic_builtin(ctx, lhs_core)
            || expr_contains_hyperbolic_builtin(ctx, rhs_core)
        {
            if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
                profiling,
                "rule.direct_identity.try.two_term_safe_hyperbolic",
                &expr_sample,
                || {
                    try_build_direct_safe_hyperbolic_core_equivalence_rewrite(
                        ctx, lhs_core, rhs_core,
                    )
                },
            ) {
                return Some(rewrite);
            }
        }
    }

    if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
        profiling,
        "rule.direct_identity.try.difference_of_equivalent_square_bases",
        &expr_sample,
        || try_build_difference_of_equivalent_square_bases_zero_rewrite(ctx, expr),
    ) {
        return Some(rewrite);
    }

    if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
        profiling,
        "rule.direct_identity.try.zero_scope_sinh_cubic",
        &expr_sample,
        || try_build_exact_zero_hyperbolic_sinh_cubic_polynomial_zero_scope_rewrite(ctx, expr),
    ) {
        return Some(rewrite);
    }

    if has_direct_trig_expr && maybe_exact_trig_equivalence_zero_scope_candidate(ctx, expr) {
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.zero_scope_fast_recursive_trig",
            &expr_sample,
            || try_build_fast_recursive_trig_angle_sum_diff_zero_scope_rewrite(ctx, expr),
        ) {
            return Some(rewrite);
        }
    }

    if has_direct_trig_expr && maybe_exact_trig_equivalence_zero_scope_candidate(ctx, expr) {
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.sin_sin_three_term_late",
            &expr_sample,
            || try_build_exact_trig_product_to_sum_sin_sin_three_term_zero_rewrite(ctx, expr),
        ) {
            return Some(rewrite);
        }
    }

    if maybe_small_polynomial_expand_zero_candidate(ctx, expr) {
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.zero_scope_fast_small_polynomial",
            &expr_sample,
            || try_build_fast_small_polynomial_expansion_zero_scope_rewrite(ctx, expr),
        ) {
            return Some(rewrite);
        }
    }

    if maybe_solve_prep_exact_additive_candidate(ctx, expr) {
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.zero_scope_fast_solve_prep",
            &expr_sample,
            || try_build_fast_solve_prep_exact_zero_scope_rewrite(ctx, expr),
        ) {
            return Some(rewrite);
        }
    }

    if let Some((lhs_core, rhs_core)) = two_term_cores {
        let has_direct_trig_core = has_direct_trig_builtin_on_either_side(ctx, lhs_core, rhs_core);
        if has_direct_trig_core {
            if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
                profiling,
                "rule.direct_identity.try.two_term_phase_shift_quarter_pair",
                &expr_sample,
                || {
                    try_build_direct_trig_exact_quarter_phase_shift_pair_equivalence_rewrite(
                        ctx, lhs_core, rhs_core,
                    )
                },
            ) {
                return Some(rewrite);
            }
        }
        if direct_term_count == 2
            && maybe_trig_phase_shift_zero_candidate(ctx, expr)
            && !binary_add_pair_is_surface_plain_trig_against_shift_signal_for_phase_shift(
                ctx, lhs_core, rhs_core,
            )
        {
            if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
                profiling,
                "rule.direct_identity.try.two_term_phase_shift_identity",
                &expr_sample,
                || {
                    try_find_trig_phase_shift_cancellation_match(ctx, lhs_core, rhs_core, false)
                        .or_else(|| {
                            try_find_trig_phase_shift_cancellation_match(
                                ctx, rhs_core, lhs_core, false,
                            )
                        })
                        .map(|_| {
                            Rewrite::with_local(zero, "Phase Shift Identity", lhs_core, rhs_core)
                        })
                },
            ) {
                return Some(rewrite);
            }
        }
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.two_term_finite_product",
            &expr_sample,
            || try_build_direct_finite_product_equivalence_rewrite(ctx, lhs_core, rhs_core),
        ) {
            return Some(rewrite);
        }
        if has_direct_trig_core {
            if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
                profiling,
                "rule.direct_identity.try.two_term_cos_product_telescoping_early",
                &expr_sample,
                || {
                    try_build_direct_cos_product_telescoping_equivalence_rewrite(
                        ctx, lhs_core, rhs_core,
                    )
                },
            ) {
                return Some(rewrite);
            }
            if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
                profiling,
                "rule.direct_identity.try.two_term_dirichlet_early",
                &expr_sample,
                || try_build_direct_dirichlet_core_equivalence_rewrite(ctx, lhs_core, rhs_core),
            ) {
                return Some(rewrite);
            }
        }
    }

    if has_direct_trig_expr && maybe_exact_trig_equivalence_zero_scope_candidate(ctx, expr) {
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.zero_scope_dirichlet",
            &expr_sample,
            || try_build_exact_dirichlet_zero_scope_rewrite(ctx, expr),
        ) {
            return Some(rewrite);
        }
    }

    if maybe_trig_square_zero_candidate(ctx, expr) {
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.zero_scope_trig_square",
            &expr_sample,
            || try_build_exact_trig_square_zero_scope_rewrite(ctx, expr),
        ) {
            return Some(rewrite);
        }
    }

    if maybe_trig_sum_to_product_zero_candidate(ctx, expr) {
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.zero_scope_trig_sum_to_product",
            &expr_sample,
            || try_build_exact_trig_sum_to_product_zero_scope_rewrite(ctx, expr),
        ) {
            return Some(rewrite);
        }
    }

    if has_direct_trig_expr && maybe_exact_trig_equivalence_zero_scope_candidate(ctx, expr) {
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.zero_scope_cos_double_angle_poly",
            &expr_sample,
            || try_build_exact_zero_trig_cos_double_angle_polynomial_zero_scope_rewrite(ctx, expr),
        ) {
            return Some(rewrite);
        }
    }

    if has_direct_trig_expr
        && maybe_exact_trig_equivalence_zero_scope_candidate(ctx, expr)
        && maybe_trig_double_angle_cos_variant_zero_scope_candidate(ctx, expr)
    {
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.zero_scope_double_angle_cos_variant",
            &expr_sample,
            || try_build_exact_zero_trig_double_angle_cos_variant_zero_scope_rewrite(ctx, expr),
        ) {
            return Some(rewrite);
        }
    }

    if has_direct_trig_expr && maybe_exact_trig_equivalence_zero_scope_candidate(ctx, expr) {
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.zero_scope_mixed_double_angle_poly",
            &expr_sample,
            || {
                try_build_exact_zero_trig_mixed_double_angle_polynomial_zero_scope_rewrite(
                    ctx, expr,
                )
            },
        ) {
            return Some(rewrite);
        }
    }

    if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
        profiling,
        "rule.direct_identity.try.zero_scope_sinh_cubic_late",
        &expr_sample,
        || try_build_exact_zero_hyperbolic_sinh_cubic_polynomial_zero_scope_rewrite(ctx, expr),
    ) {
        return Some(rewrite);
    }

    if has_direct_trig_expr
        && maybe_exact_trig_equivalence_zero_scope_candidate(ctx, expr)
        && maybe_trig_embedded_double_angle_factor_zero_scope_candidate(ctx, expr)
    {
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.zero_scope_embedded_double_angle_factor",
            &expr_sample,
            || try_build_exact_zero_trig_embedded_double_angle_factor_zero_scope_rewrite(ctx, expr),
        ) {
            return Some(rewrite);
        }
    }

    if let Some((lhs_core, rhs_core)) = two_term_cores {
        let has_direct_trig_core = has_direct_trig_builtin_on_either_side(ctx, lhs_core, rhs_core);
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.two_term_negative_even_root_power_reciprocal",
            &expr_sample,
            || {
                try_build_direct_negative_even_root_power_reciprocal_rewrite(
                    ctx, lhs_core, rhs_core,
                )
            },
        ) {
            return Some(rewrite);
        }
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.two_term_reciprocal_half_power_product",
            &expr_sample,
            || try_build_direct_reciprocal_half_power_product_rewrite(ctx, lhs_core, rhs_core),
        ) {
            return Some(rewrite);
        }
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.two_term_scaled_reciprocal_half_power_product",
            &expr_sample,
            || {
                try_build_direct_scaled_reciprocal_half_power_product_rewrite(
                    ctx, lhs_core, rhs_core,
                )
            },
        ) {
            return Some(rewrite);
        }
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.two_term_reciprocal_half_power_quotient_product_one",
            &expr_sample,
            || {
                try_build_direct_reciprocal_half_power_quotient_product_one_rewrite(
                    ctx, lhs_core, rhs_core,
                )
            },
        ) {
            return Some(rewrite);
        }
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.two_term_reciprocal_half_power_quotient_over_base",
            &expr_sample,
            || {
                try_build_direct_reciprocal_half_power_quotient_over_base_rewrite(
                    ctx, lhs_core, rhs_core,
                )
            },
        ) {
            return Some(rewrite);
        }
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.two_term_reciprocal_half_power_shared_denominator",
            &expr_sample,
            || {
                try_build_direct_reciprocal_half_power_shared_denominator_rewrite(
                    ctx, lhs_core, rhs_core,
                )
            },
        ) {
            return Some(rewrite);
        }
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.two_term_scaled_reciprocal_half_power_shared_denominator",
            &expr_sample,
            || {
                try_build_direct_scaled_reciprocal_half_power_shared_denominator_rewrite(
                    ctx, lhs_core, rhs_core,
                )
            },
        ) {
            return Some(rewrite);
        }
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.two_term_scaled_reciprocal_half_power_over_base",
            &expr_sample,
            || {
                try_build_direct_scaled_reciprocal_half_power_over_base_rewrite(
                    ctx, lhs_core, rhs_core,
                )
            },
        ) {
            return Some(rewrite);
        }
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.two_term_common_sqrt_denominator_fraction",
            &expr_sample,
            || try_build_direct_common_sqrt_denominator_fraction_rewrite(ctx, lhs_core, rhs_core),
        ) {
            return Some(rewrite);
        }
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.two_term_sqrt_over_base_fraction",
            &expr_sample,
            || try_build_direct_sqrt_over_base_fraction_rewrite(ctx, lhs_core, rhs_core),
        ) {
            return Some(rewrite);
        }
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.two_term_rationalized_common_sqrt_denominator_fraction",
            &expr_sample,
            || {
                try_build_direct_rationalized_common_sqrt_denominator_fraction_rewrite(
                    ctx, lhs_core, rhs_core,
                )
            },
        ) {
            return Some(rewrite);
        }
        if maybe_two_term_hyperbolic_direct_identity_candidate(ctx, lhs_core, rhs_core) {
            if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
                profiling,
                "rule.direct_identity.try.two_term_hyperbolic_direct_core_equivalence",
                &expr_sample,
                || try_build_direct_core_equivalence_rewrite(ctx, lhs_core, rhs_core),
            ) {
                return Some(rewrite);
            }
        }
        if has_direct_trig_core {
            if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
                profiling,
                "rule.direct_identity.try.two_term_cos_diff_sin_diff_quotient",
                &expr_sample,
                || {
                    try_build_direct_trig_cos_diff_sin_diff_quotient_equivalence_rewrite(
                        ctx, lhs_core, rhs_core,
                    )
                },
            ) {
                return Some(rewrite);
            }
            if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
                profiling,
                "rule.direct_identity.try.two_term_cos_product_telescoping_late",
                &expr_sample,
                || {
                    try_build_direct_cos_product_telescoping_equivalence_rewrite(
                        ctx, lhs_core, rhs_core,
                    )
                },
            ) {
                return Some(rewrite);
            }
            if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
                profiling,
                "rule.direct_identity.try.two_term_dirichlet_late",
                &expr_sample,
                || try_build_direct_dirichlet_core_equivalence_rewrite(ctx, lhs_core, rhs_core),
            ) {
                return Some(rewrite);
            }
            if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
                profiling,
                "rule.direct_identity.try.two_term_double_angle_contraction",
                &expr_sample,
                || {
                    try_build_direct_trig_double_angle_contraction_equivalence_rewrite(
                        ctx, lhs_core, rhs_core,
                    )
                },
            ) {
                return Some(rewrite);
            }
            if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
                profiling,
                "rule.direct_identity.try.two_term_double_angle_cos_variant",
                &expr_sample,
                || {
                    try_build_direct_trig_double_angle_cos_variant_equivalence_rewrite(
                        ctx, lhs_core, rhs_core,
                    )
                },
            ) {
                return Some(rewrite);
            }
            if maybe_two_term_embedded_double_angle_expansion_candidate(ctx, lhs_core, rhs_core) {
                if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
                    profiling,
                    "rule.direct_identity.try.two_term_embedded_double_angle_expansion",
                    &expr_sample,
                    || {
                        try_build_direct_trig_embedded_double_angle_expansion_equivalence_rewrite(
                            ctx, lhs_core, rhs_core,
                        )
                    },
                ) {
                    return Some(rewrite);
                }
            }
            if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
                profiling,
                "rule.direct_identity.try.two_term_multi_angle",
                &expr_sample,
                || try_build_direct_multi_angle_equivalence_rewrite(ctx, lhs_core, rhs_core),
            ) {
                return Some(rewrite);
            }
        }
    }

    if maybe_exact_trig_equivalence_zero_scope_candidate(ctx, expr) {
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.zero_scope_exact_trig_equivalence",
            &expr_sample,
            || {
                let candidate = try_build_exact_trig_equivalence_zero_scope_rewrite(ctx, expr);
                try_rule(ctx, candidate)
            },
        ) {
            return Some(rewrite);
        }
    }

    if maybe_trig_sum_to_product_zero_candidate(ctx, expr) {
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.expand_trig_sum_to_product",
            &expr_sample,
            || {
                let candidate =
                    ExpandTrigSumToProductToEnableCancellationRule.apply(ctx, expr, &parent_ctx);
                try_rule(ctx, candidate)
            },
        ) {
            return Some(rewrite);
        }
    }

    if maybe_trig_square_zero_candidate(ctx, expr) {
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.expand_trig_square",
            &expr_sample,
            || {
                let square_candidate =
                    ExpandTrigSquareIdentityToEnableCancellationRule.apply(ctx, expr, &parent_ctx);
                try_rule(ctx, square_candidate)
            },
        ) {
            return Some(rewrite);
        }
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.expand_triple_angle",
            &expr_sample,
            || {
                let triple_angle_candidate =
                    ExpandTrigSineProductTripleAngleToEnableCancellationRule.apply(
                        ctx,
                        expr,
                        &parent_ctx,
                    );
                try_rule(ctx, triple_angle_candidate)
            },
        ) {
            return Some(rewrite);
        }
    }

    if maybe_trig_phase_shift_zero_candidate(ctx, expr) {
        if extract_two_term_core_difference(ctx, expr).is_some_and(|(lhs_core, rhs_core)| {
            binary_add_pair_is_surface_plain_trig_against_shift_signal_for_phase_shift(
                ctx, lhs_core, rhs_core,
            )
        }) {
            return None;
        }
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.expand_trig_phase_shift",
            &expr_sample,
            || {
                let candidate =
                    ExpandTrigPhaseShiftToEnableCancellationRule.apply(ctx, expr, &parent_ctx);
                try_rule(ctx, candidate)
            },
        ) {
            return Some(rewrite);
        }
    }

    if expr_contains_direct_hyperbolic_builtin(ctx, expr) {
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.zero_scope_fast_hyperbolic",
            &expr_sample,
            || try_build_fast_hyperbolic_zero_scope_rewrite(ctx, expr),
        ) {
            return Some(rewrite);
        }
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.zero_scope_exact_hyperbolic_equivalence",
            &expr_sample,
            || {
                let candidate =
                    try_build_exact_hyperbolic_equivalence_zero_scope_rewrite(ctx, expr);
                try_rule(ctx, candidate)
            },
        ) {
            return Some(rewrite);
        }
    }

    if maybe_hyperbolic_angle_sum_diff_zero_candidate(ctx, expr) {
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.expand_hyperbolic_angle_sum_diff",
            &expr_sample,
            || {
                let candidate = ExpandHyperbolicAngleSumDiffToEnableCancellationRule.apply(
                    ctx,
                    expr,
                    &parent_ctx,
                );
                try_rule(ctx, candidate)
            },
        ) {
            return Some(rewrite);
        }
    }

    if maybe_hyperbolic_pythagorean_factor_zero_candidate(ctx, expr) {
        if let Some(rewrite) =
            run_profiled_exact_zero_direct_identity_probe(
                profiling,
                "rule.direct_identity.try.expand_hyperbolic_pythagorean_factor",
                &expr_sample,
                || {
                    let candidate = ExpandHyperbolicPythagoreanFactorToEnableCancellationRule
                        .apply(ctx, expr, &parent_ctx);
                    try_rule(ctx, candidate)
                },
            )
        {
            return Some(rewrite);
        }
    }

    if maybe_solve_prep_exact_additive_candidate(ctx, expr) {
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.zero_scope_exact_solve_prep",
            &expr_sample,
            || {
                let candidate = try_build_exact_solve_prep_zero_scope_rewrite(ctx, expr);
                try_rule(ctx, candidate)
            },
        ) {
            return Some(rewrite);
        }
    }

    if maybe_log_product_power_zero_candidate(ctx, expr) {
        if let Some(rewrite) = run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.expand_log_product_power",
            &expr_sample,
            || {
                let log_power_candidate =
                    ExpandLogProductPowerToEnableCancellationRule.apply(ctx, expr, &parent_ctx);
                try_rule(ctx, log_power_candidate)
            },
        ) {
            return Some(rewrite);
        }
    }

    if maybe_log_abs_mul_div_zero_candidate(ctx, expr) {
        run_profiled_exact_zero_direct_identity_probe(
            profiling,
            "rule.direct_identity.try.expand_log_abs_mul_div",
            &expr_sample,
            || {
                let log_abs_candidate =
                    ExpandLogAbsMulDivToEnableCancellationRule.apply(ctx, expr, &parent_ctx);
                try_rule(ctx, log_abs_candidate)
            },
        )
    } else {
        None
    }
}

pub(super) fn build_common_scale_exact_zero_rewrite(
    ctx: &mut cas_ast::Context,
    whole_expr: cas_ast::ExprId,
    common_factor: cas_ast::ExprId,
    residual_expr: cas_ast::ExprId,
    child_rewrite: Rewrite,
) -> Rewrite {
    let factorized_difference = smart_mul(ctx, common_factor, residual_expr);
    let factorized_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: factorized_difference
        }
    );

    let mut rewrite = Rewrite::with_local(
        ctx.num(0),
        child_rewrite.description.clone(),
        whole_expr,
        ctx.num(0),
    )
    .requires_all(child_rewrite.required_conditions.clone())
    .assume_all(child_rewrite.assumption_events.clone());

    if let Some(poly_proof) = child_rewrite.poly_proof.clone() {
        rewrite = rewrite.poly_proof(poly_proof);
    }

    rewrite = rewrite.substep(
        "Sacar factor común",
        vec![format!("Se obtiene {factorized_display}.")],
    );
    rewrite.substeps.extend(child_rewrite.substeps.clone());
    rewrite
}

pub(super) fn try_build_exact_zero_common_scaled_difference_rewrite_with_context(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    parent_ctx: &ParentContext,
) -> Option<Rewrite> {
    let mut rewrite = try_build_exact_zero_common_scaled_difference_rewrite(ctx, expr)?;
    if rewrite.assumption_events.is_empty() {
        if let Some(event) = common_scale_abs_like_positive_assumption_event(ctx, expr, parent_ctx)
        {
            rewrite = rewrite.assume(event);
        }
    }
    Some(rewrite)
}

pub(super) fn strip_single_additive_zero_term(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() < 2 {
        return None;
    }

    let zero = ctx.num(0);
    let mut removed = false;
    let mut remaining = Vec::with_capacity(terms.len().saturating_sub(1));
    for (term_expr, term_sign) in terms {
        if !removed && compare_expr(ctx, term_expr, zero) == Ordering::Equal {
            removed = true;
            continue;
        }
        remaining.push((term_expr, term_sign));
    }

    if !removed || remaining.is_empty() {
        return None;
    }

    Some(build_signed_sum_expr(ctx, &remaining))
}

pub(super) fn additive_scope_contains_zero_term(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() < 2 {
        return false;
    }

    let zero = ctx.num(0);
    terms
        .into_iter()
        .any(|(term_expr, _term_sign)| compare_expr(ctx, term_expr, zero) == Ordering::Equal)
}

fn try_find_exact_zero_additive_factor_in_product(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, Rewrite)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for factor in factors {
        if !matches!(ctx.get(factor), Expr::Add(_, _) | Expr::Sub(_, _)) {
            continue;
        }
        if let Some((lhs_core, rhs_core)) = extract_two_term_core_difference(ctx, factor) {
            if is_atanh_common_log_definition_mismatch_pair(ctx, lhs_core, rhs_core) {
                continue;
            }
            if expr_contains_sqrt_or_half_power(ctx, factor) {
                if let Some(child_rewrite) =
                    try_build_direct_scaled_reciprocal_half_power_product_rewrite(
                        ctx, lhs_core, rhs_core,
                    )
                {
                    return Some((factor, child_rewrite));
                }
            }
        }

        if let Some(child_rewrite) = try_build_small_direct_zero_core_rewrite(ctx, factor) {
            return Some((factor, child_rewrite));
        }
        let child_rewrite = try_build_exact_zero_identity_rewrite(ctx, factor)?;
        return Some((factor, child_rewrite));
    }

    None
}

fn build_exact_zero_product_factor_rewrite(
    ctx: &mut cas_ast::Context,
    local_factor: cas_ast::ExprId,
    child_rewrite: Rewrite,
) -> Rewrite {
    let mut rewrite = Rewrite::with_local(
        ctx.num(0),
        child_rewrite.description.clone(),
        local_factor,
        ctx.num(0),
    )
    .requires_all(child_rewrite.required_conditions.clone())
    .assume_all(child_rewrite.assumption_events.clone());

    if let Some(poly_proof) = child_rewrite.poly_proof.clone() {
        rewrite = rewrite.poly_proof(poly_proof);
    }

    rewrite.substeps = child_rewrite.substeps.clone();
    rewrite
}

pub(crate) fn common_scaled_difference_has_exact_nonzero_witness(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    use num_traits::Zero;

    let var_names: Vec<String> = cas_ast::traversal::collect_variables(ctx, expr)
        .into_iter()
        .collect();
    // Generic rationals that avoid the small-integer poles typical of these denominators.
    let samples: [(i64, i64); 5] = [(7, 2), (11, 3), (13, 5), (17, 4), (23, 6)];
    for round in 0..samples.len() {
        let mut substituted = expr;
        for (i, name) in var_names.iter().enumerate() {
            let var_node = ctx.var(name);
            let (n, d) = samples[(round + i) % samples.len()];
            let value = num_rational::BigRational::new((n + i as i64).into(), d.into());
            let value_node = ctx.add(cas_ast::Expr::Number(value));
            substituted =
                cas_ast::traversal::substitute_expr_by_id(ctx, substituted, var_node, value_node);
        }
        if let Some(value) = eval_exact_rational(ctx, substituted, 64) {
            if !value.is_zero() {
                return true; // exact non-zero value → the difference is NOT identically zero
            }
        }
        // `None` (transcendental, or a pole at this sample) cannot disprove zero — try the next.
    }
    false
}

pub(crate) fn try_build_exact_zero_common_scaled_difference_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    // A term carrying a literal non-finite or undefined value never cancels with
    // itself: `inf - inf`, `(1/0) - (1/0)` and `undefined - undefined` are
    // indeterminate, not zero. Decline so the difference stays symbolic.
    if additive_term_is_nonfinite_or_undefined(ctx, expr) {
        return None;
    }
    // SOUNDNESS GATE: never collapse to 0 a rational expression that exactly evaluates to a non-zero
    // value at a generic point (`1/(x²−1) − 1/(x−1)` is `−x/(x²−1)`, not 0).
    if common_scaled_difference_has_exact_nonzero_witness(ctx, expr) {
        return None;
    }

    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    let expr_sample = profiling.then(|| render_expr_for_orchestrator_profile(ctx, expr));
    let profile_route = |label: &'static str| {
        if profiling {
            let _ =
                run_profiled_orchestrator_option_section(label, expr_sample.clone(), || Some(()));
        }
    };

    if let Some(rewrite) = try_build_exact_zero_same_denominator_rewrite(ctx, expr) {
        profile_route("rule.common_scale_zero.route.same_denominator");
        return Some(rewrite);
    }

    if let Some((lhs_core, rhs_core)) = extract_two_term_core_difference(ctx, expr) {
        if let Some(rewrite) =
            try_build_direct_scaled_reciprocal_half_power_shared_denominator_rewrite(
                ctx, lhs_core, rhs_core,
            )
        {
            profile_route(
                "rule.common_scale_zero.route.scaled_reciprocal_half_power_shared_denominator",
            );
            return Some(rewrite);
        }
        if let Some(rewrite) =
            try_build_direct_scaled_reciprocal_half_power_over_base_rewrite(ctx, lhs_core, rhs_core)
        {
            profile_route("rule.common_scale_zero.route.scaled_reciprocal_half_power_over_base");
            return Some(rewrite);
        }
    }

    if let Some((common_factor, lhs_core, rhs_core)) =
        extract_two_term_common_scale_difference_cores(ctx, expr)
    {
        if same_arg_sin_cos_core_pair(ctx, lhs_core, rhs_core) {
            return None;
        }
        let residual_expr = ctx.add(Expr::Sub(lhs_core, rhs_core));
        if let Some(child_rewrite) =
            try_build_repeated_trig_phase_shift_pair_zero_rewrite(ctx, residual_expr)
        {
            profile_route("rule.common_scale_zero.route.focused_phase_shift_pair");
            return Some(build_common_scale_exact_zero_rewrite(
                ctx,
                expr,
                common_factor,
                residual_expr,
                child_rewrite,
            ));
        }
        if let Some(child_rewrite) =
            try_build_exact_zero_shared_passthrough_difference_rewrite(ctx, residual_expr)
        {
            profile_route("rule.common_scale_zero.route.focused_shared_passthrough");
            return Some(build_common_scale_exact_zero_rewrite(
                ctx,
                expr,
                common_factor,
                residual_expr,
                child_rewrite,
            ));
        }
        if let Some(child_rewrite) =
            try_build_direct_tanh_exp_definition_equivalence_rewrite(ctx, lhs_core, rhs_core)
        {
            profile_route("rule.common_scale_zero.route.focused_tanh_exp");
            return Some(build_common_scale_exact_zero_rewrite(
                ctx,
                expr,
                common_factor,
                residual_expr,
                child_rewrite,
            ));
        }
        if let Some(child_rewrite) =
            try_build_direct_trig_square_equivalence_rewrite(ctx, lhs_core, rhs_core)
        {
            profile_route("rule.common_scale_zero.route.focused_trig_square");
            return Some(build_common_scale_exact_zero_rewrite(
                ctx,
                expr,
                common_factor,
                residual_expr,
                child_rewrite,
            ));
        }
        if let Some(child_rewrite) =
            try_build_direct_hyperbolic_sinh_cubic_polynomial_equivalence_rewrite(
                ctx, lhs_core, rhs_core,
            )
        {
            profile_route("rule.common_scale_zero.route.focused_sinh_cubic");
            return Some(build_common_scale_exact_zero_rewrite(
                ctx,
                expr,
                common_factor,
                residual_expr,
                child_rewrite,
            ));
        }
        if let Some(child_rewrite) =
            try_build_direct_trig_exact_quarter_phase_shift_pair_equivalence_rewrite(
                ctx, lhs_core, rhs_core,
            )
        {
            profile_route("rule.common_scale_zero.route.focused_phase_shift_quarter_pair");
            return Some(build_common_scale_exact_zero_rewrite(
                ctx,
                expr,
                common_factor,
                residual_expr,
                child_rewrite,
            ));
        }
        if let Some(child_rewrite) =
            try_build_direct_safe_hyperbolic_core_equivalence_rewrite(ctx, lhs_core, rhs_core)
        {
            profile_route("rule.common_scale_zero.route.focused_safe_hyperbolic");
            return Some(build_common_scale_exact_zero_rewrite(
                ctx,
                expr,
                common_factor,
                residual_expr,
                child_rewrite,
            ));
        }
    }

    let (common_factor, residual_expr) = extract_common_multiplicative_residual_sum(ctx, expr)?;
    if same_arg_sin_cos_additive_pair(ctx, residual_expr) {
        return None;
    }
    let residual_term_count = AddView::from_expr(ctx, residual_expr).terms.len();
    if residual_term_count == 2 {
        if let Some((lhs_core, rhs_core)) = extract_two_term_core_difference(ctx, residual_expr) {
            if let Some(rewrite_match) =
                try_find_trig_phase_shift_cancellation_match(ctx, lhs_core, rhs_core, false)
            {
                if matches!(
                    rewrite_match.mode,
                    TrigPhaseShiftCancellationMode::ShiftedToShifted
                ) {
                    profile_route(
                        "rule.common_scale_zero.route.residual_phase_shift_identity_shifted_to_shifted",
                    );
                    let child_rewrite = build_trig_phase_shift_zero_rewrite(ctx, rewrite_match);
                    return Some(build_common_scale_exact_zero_rewrite(
                        ctx,
                        expr,
                        common_factor,
                        residual_expr,
                        child_rewrite,
                    ));
                }
            }
            if let Some(child_rewrite) =
                try_build_direct_core_equivalence_rewrite(ctx, lhs_core, rhs_core)
            {
                profile_route("rule.common_scale_zero.route.residual_direct_core_equivalence");
                return Some(build_common_scale_exact_zero_rewrite(
                    ctx,
                    expr,
                    common_factor,
                    residual_expr,
                    child_rewrite,
                ));
            }
        }
    }
    if let Some(child_rewrite) =
        try_build_repeated_trig_phase_shift_pair_zero_rewrite(ctx, residual_expr)
    {
        profile_route("rule.common_scale_zero.route.residual_phase_shift_pair");
        return Some(build_common_scale_exact_zero_rewrite(
            ctx,
            expr,
            common_factor,
            residual_expr,
            child_rewrite,
        ));
    }
    if let Some(child_rewrite) =
        try_build_exact_zero_shared_passthrough_difference_rewrite(ctx, residual_expr)
    {
        profile_route("rule.common_scale_zero.route.residual_shared_passthrough");
        return Some(build_common_scale_exact_zero_rewrite(
            ctx,
            expr,
            common_factor,
            residual_expr,
            child_rewrite,
        ));
    }
    if residual_term_count <= 4 {
        if let Some(child_rewrite) =
            try_build_exact_zero_identity_rewrite_direct(ctx, residual_expr)
        {
            let zero = ctx.num(0);
            if compare_expr(ctx, child_rewrite.final_expr(), zero) == Ordering::Equal {
                profile_route("rule.common_scale_zero.route.residual_direct_identity");
                return Some(build_common_scale_exact_zero_rewrite(
                    ctx,
                    expr,
                    common_factor,
                    residual_expr,
                    child_rewrite,
                ));
            }
        }
    }
    if let Some(child_rewrite) =
        try_build_stripped_zero_log_identity_child_rewrite(ctx, residual_expr).or_else(|| {
            try_build_fast_multiterm_hyperbolic_residual_child_rewrite(ctx, residual_expr)
        })
    {
        profile_route("rule.common_scale_zero.route.residual_log_or_multiterm_hyperbolic");
        return Some(build_common_scale_exact_zero_rewrite(
            ctx,
            expr,
            common_factor,
            residual_expr,
            child_rewrite,
        ));
    }
    if let Some((lhs_core, rhs_core)) = extract_two_term_core_difference(ctx, residual_expr) {
        if let Some(child_rewrite) =
            try_build_direct_safe_hyperbolic_core_equivalence_rewrite(ctx, lhs_core, rhs_core)
        {
            profile_route("rule.common_scale_zero.route.residual_safe_hyperbolic");
            return Some(build_common_scale_exact_zero_rewrite(
                ctx,
                expr,
                common_factor,
                residual_expr,
                child_rewrite,
            ));
        }
    }

    let normalized_residual = normalize_additive_scope_expr(ctx, residual_expr);
    let child_rewrite = try_build_fast_trig_residual_identity_child_rewrite(ctx, residual_expr)
        .inspect(|_| {
            profile_route("rule.common_scale_zero.route.tail_fast_trig_raw");
        })
        .or_else(|| {
            try_build_fast_trig_residual_identity_child_rewrite(ctx, normalized_residual).inspect(
                |_| {
                    profile_route("rule.common_scale_zero.route.tail_fast_trig_normalized");
                },
            )
        })
        .or_else(|| {
            try_build_exact_zero_shared_passthrough_difference_rewrite(ctx, residual_expr).inspect(
                |_| {
                    profile_route("rule.common_scale_zero.route.tail_shared_passthrough");
                },
            )
        })
        .or_else(|| {
            try_build_fast_small_polynomial_residual_child_rewrite(ctx, residual_expr).inspect(
                |_| {
                    profile_route("rule.common_scale_zero.route.tail_fast_small_poly_raw");
                },
            )
        })
        .or_else(|| {
            try_build_fast_small_polynomial_residual_child_rewrite(ctx, normalized_residual)
                .inspect(|_| {
                    profile_route("rule.common_scale_zero.route.tail_fast_small_poly_normalized");
                })
        })
        .or_else(|| {
            try_build_two_term_core_equivalence_rewrite(ctx, residual_expr).inspect(|_| {
                profile_route("rule.common_scale_zero.route.tail_two_term_core_equivalence");
            })
        })
        .or_else(|| {
            try_build_exact_zero_identity_rewrite(ctx, residual_expr).inspect(|_| {
                profile_route("rule.common_scale_zero.route.tail_exact_zero_identity_raw");
            })
        })
        .or_else(|| {
            try_build_exact_zero_identity_rewrite(ctx, normalized_residual).inspect(|_| {
                profile_route("rule.common_scale_zero.route.tail_exact_zero_identity_normalized");
            })
        })?;
    Some(build_common_scale_exact_zero_rewrite(
        ctx,
        expr,
        common_factor,
        residual_expr,
        child_rewrite,
    ))
}

pub(super) fn try_build_exact_zero_product_factor_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    parent_ctx: &ParentContext,
) -> Option<Rewrite> {
    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };

    let allow = cas_solver_core::undefined_risk_policy_support::allow_cancellation_with_undefined_risk_mode_flags(
        matches!(parent_ctx.domain_mode(), crate::DomainMode::Assume),
        matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict),
        crate::collect::has_undefined_risk(ctx, expr),
    );
    if !allow {
        return None;
    }
    if product_has_variable_scaled_direct_trig_or_hyperbolic_additive_factor(ctx, expr) {
        return None;
    }

    let (local_factor, child_rewrite) = try_find_exact_zero_additive_factor_in_product(ctx, expr)?;
    Some(build_exact_zero_product_factor_rewrite(
        ctx,
        local_factor,
        child_rewrite,
    ))
}

pub(super) fn build_dirichlet_nonzero_condition_expr(
    ctx: &mut cas_ast::Context,
    result: cas_math::telescoping_dirichlet::DirichletKernelResult,
) -> cas_ast::ExprId {
    let scaled_base = if result.base_multiplier == 1 {
        result.base_var
    } else {
        let multiplier = ctx.num(result.base_multiplier as i64);
        smart_mul(ctx, multiplier, result.base_var)
    };
    let two = ctx.num(2);
    let half_angle = ctx.add(Expr::Div(scaled_base, two));
    ctx.call_builtin(BuiltinFn::Sin, vec![half_angle])
}

pub(super) fn try_build_exact_dirichlet_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let result = try_dirichlet_kernel_identity(ctx, expr)?;
    Some(
        Rewrite::with_local(ctx.num(0), "Dirichlet Kernel Identity", expr, ctx.num(0)).requires(
            crate::ImplicitCondition::NonZero(build_dirichlet_nonzero_condition_expr(ctx, result)),
        ),
    )
}

pub(super) fn try_build_direct_integrate_prep_exact_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let (lhs_core, rhs_core) = extract_two_term_core_difference(ctx, expr)?;
    try_build_direct_cos_product_telescoping_equivalence_rewrite(ctx, lhs_core, rhs_core)
        .or_else(|| try_build_direct_dirichlet_core_equivalence_rewrite(ctx, lhs_core, rhs_core))
}

fn try_build_small_factorial_zero_core_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let zero = ctx.num(0);
    let (lhs_core, rhs_core) = extract_two_term_core_difference(ctx, expr)?;

    for (ratio_side, target_side) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) =
            cas_math::number_theory_support::try_rewrite_consecutive_factorial_ratio_expr(
                ctx, ratio_side,
            )
        else {
            continue;
        };

        if exprs_match_for_cancellation(ctx, rewrite.rewritten, target_side)
            || exprs_match_after_default_simplify(ctx, rewrite.rewritten, target_side)
        {
            return Some(
                Rewrite::with_local(zero, "Cancel consecutive factorials", ratio_side, target_side)
                    .requires(crate::ImplicitCondition::NonNegative(
                        rewrite.factorial_arg_requires_nonnegative,
                    ))
                    .substep(
                        "Expandir el factorial superior hasta llegar al factorial inferior",
                        vec![
                            "La razón de factoriales consecutivos se desarrolla hasta exponer el factorial común."
                                .to_string(),
                        ],
                    )
                    .substep(
                        "Cancelar el factorial común",
                        vec![
                            "Después de cancelar el factorial compartido, ambos lados coinciden exactamente."
                                .to_string(),
                        ],
                    ),
            );
        }
    }

    None
}

fn build_exact_zero_subset_passthrough_rewrite(
    ctx: &mut cas_ast::Context,
    subset_expr: cas_ast::ExprId,
    passthrough_terms: &[(cas_ast::ExprId, Sign)],
    child_rewrite: Rewrite,
) -> Rewrite {
    let mut rewrite = Rewrite::with_local(
        build_signed_sum_expr(ctx, passthrough_terms),
        child_rewrite.description.clone(),
        subset_expr,
        ctx.num(0),
    )
    .requires_all(child_rewrite.required_conditions.clone())
    .assume_all(child_rewrite.assumption_events.clone());

    if let Some(poly_proof) = child_rewrite.poly_proof.clone() {
        rewrite = rewrite.poly_proof(poly_proof);
    }

    rewrite.substeps = child_rewrite.substeps.clone();
    rewrite
}

pub(super) fn try_build_structural_cancel_then_exact_zero_subset_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 4 {
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

            let passthrough_expr = build_signed_sum_expr(ctx, &passthrough_terms);
            if let Some(rewrite) =
                try_build_repeated_trig_phase_shift_pair_zero_rewrite(ctx, passthrough_expr)
            {
                return Some(rewrite);
            }
            let child_rewrite = try_build_exact_zero_identity_rewrite(ctx, passthrough_expr)?;

            let mut rewrite = Rewrite::with_local(
                ctx.num(0),
                child_rewrite.description.clone(),
                passthrough_expr,
                ctx.num(0),
            )
            .requires_all(child_rewrite.required_conditions.clone())
            .assume_all(child_rewrite.assumption_events.clone());

            if let Some(poly_proof) = child_rewrite.poly_proof.clone() {
                rewrite = rewrite.poly_proof(poly_proof);
            }

            rewrite.substeps = child_rewrite.substeps.clone();
            return Some(rewrite);
        }
    }

    None
}

fn matches_structural_three_term_zero_pattern(
    ctx: &cas_ast::Context,
    target_expr: cas_ast::ExprId,
    other_a: cas_ast::ExprId,
    other_b: cas_ast::ExprId,
) -> bool {
    let Some((common, extra)) = match_mul_by_one_plus_term(ctx, target_expr) else {
        return false;
    };
    (compare_expr(ctx, other_a, common) == Ordering::Equal
        && term_matches_structural_product(ctx, other_b, common, extra))
        || (compare_expr(ctx, other_b, common) == Ordering::Equal
            && term_matches_structural_product(ctx, other_a, common, extra))
}

pub(super) fn try_build_structural_three_term_poly_zero_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return None;
    }

    let normalized_terms: Vec<_> = view
        .terms
        .iter()
        .copied()
        .map(|(term_expr, term_sign)| normalize_signed_add_term(ctx, term_expr, term_sign))
        .collect();

    for target_index in 0..normalized_terms.len() {
        let (target_expr, target_sign) = normalized_terms[target_index];
        let others: Vec<_> = normalized_terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, term)| (index != target_index).then_some(term))
            .collect();
        if others.len() != 2 {
            continue;
        }
        if others.iter().any(|(_, sign)| *sign != target_sign.negate()) {
            continue;
        }

        if matches_structural_three_term_zero_pattern(ctx, target_expr, others[0].0, others[1].0) {
            return Some(
                Rewrite::with_local(
                    ctx.num(0),
                    "Polynomial equality: expressions cancel to 0",
                    expr,
                    ctx.num(0),
                )
                .substep(
                    "Reconocer cancelación exacta de tres términos",
                    vec![
                        "Un término coincide exactamente con la suma de los otros dos, así que toda la expresión vale 0.".to_string(),
                    ],
                ),
            );
        }
    }

    None
}

pub(super) fn try_build_exact_zero_three_term_subset_passthrough_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 4 {
        return None;
    }

    for first_index in 0..view.terms.len().saturating_sub(2) {
        for second_index in (first_index + 1)..view.terms.len().saturating_sub(1) {
            for third_index in (second_index + 1)..view.terms.len() {
                let subset_terms = [
                    view.terms[first_index],
                    view.terms[second_index],
                    view.terms[third_index],
                ];
                let subset_expr = build_signed_sum_expr(ctx, &subset_terms);
                let Some(child_rewrite) = try_build_exact_zero_identity_rewrite(ctx, subset_expr)
                else {
                    continue;
                };

                let passthrough_terms: Vec<_> = view
                    .terms
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(index, term)| {
                        (index != first_index && index != second_index && index != third_index)
                            .then_some(term)
                    })
                    .collect();

                if passthrough_terms.is_empty() {
                    continue;
                }

                return Some(build_exact_zero_subset_passthrough_rewrite(
                    ctx,
                    subset_expr,
                    &passthrough_terms,
                    child_rewrite,
                ));
            }
        }
    }

    None
}
