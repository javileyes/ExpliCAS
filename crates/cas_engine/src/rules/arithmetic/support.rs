//! `arithmetic`: familia `support` — y, desde D1b (2026-08), el hogar físico
//! de la **API del motor de cancelación**.
//!
//! Los 18 helpers alcanzados desde ≥12 de los 25 entries `define_rule!`
//! (inventario `docs/DESACOPLO_D1_INVENTARIO_2026-08.md`) viven aquí, más los
//! PROMOVIDOS por los peldaños D1c (el umbral 12 descubrió la API; la
//! semántica la cierra): `canonicalize_nested_integer_powers` (D1c-1, el
//! canonicalizador-para-comparar del veredicto),
//! `additive_term_is_nonfinite_or_undefined` y
//! `combine_additive_numeric_constants_for_cancellation` (D1c-2, guard de
//! no-finitos y combinador de constantes del candidato);
//! `expr_contains_any_builtin`, `expr_matches_negation_after_default_simplify`,
//! `abs_argument`, `small_positive_integer_value`, `extract_sqrt_argument` y
//! `expr_contains_sqrt_or_half_power` (D1c-3/4, detectores y extractores
//! estructurales neutros — cuatro ya vivían aquí sin declarar);
//! `apply_sign_to_expr` y `expr_matches_negation_for_cancellation`
//! (D1c-5/6, aplicador de signo y matcher de negación — ya vivían aquí);
//! `build_signed_add_expr` (D1c-7/8, constructor hermano de
//! `build_signed_sum_expr`); y los NUEVE neutros de D1c-9 que ya vivían aquí
//! sin declarar (`additive_scopes_match_after_default_simplify`,
//! `build_mul_expr_from_factors`, `default_simplify_nesting_depth`,
//! `expr_contains_symbolic_atom_for_cancellation`,
//! `extract_two_term_core_difference`, `negate_additive_scope_expr`,
//! `normalize_additive_scope_expr`, `normalize_signed_add_term_for_fast_match`,
//! `sign_to_i64`); y los NUEVE neutros de cancelación de D1c-10 traídos de
//! general.rs (`distribute_symbolic_scale_sum_term_for_cancellation`,
//! `extract_literal_rational_for_cancellation`,
//! `is_simple_symbolic_scale_factor_for_cancellation`,
//! `normalize_core_difference_term`,
//! `split_out_small_integer_factor_for_cancellation`,
//! `strip_common_factor_from_term`,
//! `strip_trivial_one_product_factors_for_core_difference`,
//! `try_rewrite_{simple,single}_symbolic_scale_sum_for_cancellation`). Los
//! helpers de PERFILADO viven en `profiling.rs` (infra transversal
//! declarada, D1c-9) y los angulares en su familia (π, √2 y 1/2 son
//! constantes ANGULARES: viven en `phase_shift.rs` desde D1c-10). NO
//! promovidos a propósito: `extract_sin_or_cos_linear_term_for_phase_shift`,
//! `maybe_trig_square_zero_candidate` y
//! `split_linear_angle_term_for_phase_shift_cancellation` son de FAMILIA
//! angular/trig — semilla del submódulo de esa familia (peldaños D1c 9-12);
//! la familia hyperbolic ya tiene el suyo declarado (`hyperbolic.rs`,
//! D1c-5/6). Tres grupos:
//! veredicto de equivalencia-para-cancelación (`exprs_match_for_*`,
//! `exprs_equal_up_to_*`, `canonicalize_nested_integer_powers`),
//! candidato/colección (`collect_add_terms`,
//! `collect_signed_mul_factors`, `signed_term_expr`,
//! `normalize_signed_add_term`, `strip_term_negation`,
//! `term_has_matrix_product_factor`) y rewrite/entorno (`build_signed_sum_expr`,
//! `build_scaled_expr`, `run_default_simplify`,
//! `ambient_pipeline_value_domain`). Los disparadores migrados por D1c deben
//! importar de AQUÍ (y de sus exclusivos) — no del resto de internals; la
//! meta medible es que su «arrastre» (tabla del inventario) baje a cero.
//!
//! Ver la cabecera de `arithmetic.rs` para el contexto del troceo.

use super::*;

pub(super) fn expr_contains_any_builtin(
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

pub(super) fn expr_contains_symbolic_atom_for_cancellation(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
) -> bool {
    let mut stack = vec![root];

    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Variable(_) | Expr::SessionRef(_) => return true,
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
            Expr::Number(_) | Expr::Constant(_) => {}
        }
    }

    false
}

pub(super) fn expr_contains_any_function_call(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
) -> bool {
    let mut stack = vec![root];

    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Function(_, _) => {
                return true;
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

pub(super) fn apply_sign_to_expr(
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

pub(super) fn expr_contains_sqrt_or_half_power(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
) -> bool {
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

pub(super) fn default_simplify_nesting_depth() -> usize {
    DEFAULT_SIMPLIFY_NESTING.with(|depth| depth.get())
}

pub(super) fn run_default_simplify(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    struct DefaultSimplifyNestingGuard;

    impl Drop for DefaultSimplifyNestingGuard {
        fn drop(&mut self) {
            DEFAULT_SIMPLIFY_NESTING.with(|depth| {
                depth.set(depth.get().saturating_sub(1));
            });
        }
    }

    let nesting = DEFAULT_SIMPLIFY_NESTING.with(|depth| {
        let current = depth.get();
        depth.set(current + 1);
        current
    });
    let _nesting_guard = DefaultSimplifyNestingGuard;

    // Speculative exact-zero probes may nest at most TWO default
    // simplifies: observed successful matches happen at nesting 0-1
    // (the phase-shift quotient pair needs one nested probe inside its
    // full-pipeline probe); nesting 2-3 only burns CPU. The
    // double-angle/power-reduction probe pair otherwise regenerates
    // cos(4x)+1 one level deeper each round at x20-40 the work,
    // hanging sums like sin(x)^2 cos(x)^2 - sin(x)^4 indefinitely.
    if nesting >= 2 {
        return expr;
    }

    // Breadth cap: the subset-enumeration probes each launch a
    // simplify here; past the per-pipeline budget they fall back to
    // the syntactic fast path. Outside an armed pipeline scope (unit
    // contexts) the budget is inactive.
    // Memo hit: replay the earlier probe result without consuming budget or
    // nesting (each expr is served the strength of its FIRST probe, which the
    // decaying budget makes the strongest one it would ever get).
    let probe_value_domain = ambient_pipeline_value_domain();
    let memo_key = (ctx.instance_tag(), expr, probe_value_domain);
    if let Some(cached) =
        DEFAULT_SIMPLIFY_PROBE_MEMO.with(|memo| memo.borrow().get(&memo_key).copied())
    {
        return cached;
    }

    let mut force_local = false;
    match DEFAULT_SIMPLIFY_PROBES_LEFT.with(|left| left.get()) {
        Some(0) => return expr,
        Some(probes_left) => {
            DEFAULT_SIMPLIFY_PROBES_LEFT.with(|left| left.set(Some(probes_left - 1)));
            // Only the first FULL_PROBE_BUDGET probes may launch a
            // full fresh pipeline: a full pipeline per probe is what
            // turned the subset enumeration into a hang (16 probes x
            // 1-2s pipelines on sin^4 + cos^4 - 1 + 2 sin^2 cos^2).
            force_local =
                probes_left <= DEFAULT_SIMPLIFY_PROBE_BUDGET - DEFAULT_SIMPLIFY_FULL_PROBE_BUDGET;
        }
        None => {}
    }

    if nesting > 0 || force_local {
        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.set_collect_steps(false);
        simplifier.set_sticky_value_domain(probe_value_domain);
        std::mem::swap(&mut simplifier.context, ctx);
        let pattern_marks = crate::pattern_marks::PatternMarks::new();
        let rewritten = crate::with_suppressed_depth_overflow_warnings(|| {
            let (core, _) = simplifier.local_simplify_with_phase(
                expr,
                &pattern_marks,
                crate::phase::SimplifyPhase::Core,
            );
            let (transform, _) = simplifier.local_simplify_with_phase(
                core,
                &pattern_marks,
                crate::phase::SimplifyPhase::Transform,
            );
            let (post, _) = simplifier.local_simplify_with_phase(
                transform,
                &pattern_marks,
                crate::phase::SimplifyPhase::PostCleanup,
            );
            post
        });
        std::mem::swap(&mut simplifier.context, ctx);
        DEFAULT_SIMPLIFY_PROBE_MEMO.with(|memo| memo.borrow_mut().insert(memo_key, rewritten));
        return rewritten;
    }

    let mut simplifier = crate::Simplifier::with_default_rules();
    simplifier.set_collect_steps(false);
    simplifier.set_sticky_value_domain(probe_value_domain);
    std::mem::swap(&mut simplifier.context, ctx);
    let mut probe_options = crate::SimplifyOptions {
        suppress_depth_overflow_warnings: true,
        ..crate::SimplifyOptions::default()
    };
    probe_options.shared.semantics.value_domain = probe_value_domain;
    let (rewritten, _steps, _stats) = simplifier.simplify_with_stats(expr, probe_options);
    std::mem::swap(&mut simplifier.context, ctx);
    DEFAULT_SIMPLIFY_PROBE_MEMO.with(|memo| memo.borrow_mut().insert(memo_key, rewritten));
    rewritten
}

pub(super) fn is_zero_after_default_simplify(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let zero = ctx.num(0);
    let simplified = run_default_simplify(ctx, expr);
    compare_expr(ctx, simplified, zero) == Ordering::Equal
}

pub(super) fn small_positive_integer_value(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<i64> {
    match ctx.get(expr) {
        Expr::Number(n)
            if n.is_integer() && *n > num_rational::BigRational::from_integer(0.into()) =>
        {
            n.to_integer().try_into().ok()
        }
        _ => None,
    }
}

pub(super) fn build_scaled_expr(
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

pub(super) fn normalize_signed_add_term(
    ctx: &mut cas_ast::Context,
    term_expr: cas_ast::ExprId,
    term_sign: Sign,
) -> (cas_ast::ExprId, Sign) {
    let unheld = cas_ast::hold::unwrap_internal_hold(ctx, term_expr);
    if unheld != term_expr {
        return normalize_signed_add_term(ctx, unheld, term_sign);
    }

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
        Expr::Div(num, den) => {
            if let Some(positive_num) = strip_term_negation(ctx, num) {
                return (ctx.add(Expr::Div(positive_num, den)), term_sign.negate());
            }
            (term_expr, term_sign)
        }
        _ => (term_expr, term_sign),
    }
}

pub(super) fn normalize_signed_add_term_for_fast_match(
    ctx: &mut cas_ast::Context,
    term_expr: cas_ast::ExprId,
    term_sign: Sign,
) -> (cas_ast::ExprId, Sign) {
    let (term_expr, term_sign) = normalize_signed_add_term(ctx, term_expr, term_sign);
    let factors = flatten_mul_chain(ctx, term_expr);
    if factors.len() <= 1 {
        return (term_expr, term_sign);
    }

    for (index, factor) in factors.iter().copied().enumerate() {
        let Some(positive_factor) = strip_term_negation(ctx, factor) else {
            continue;
        };

        let mut rebuilt_factors = factors.clone();
        rebuilt_factors[index] = positive_factor;
        let rebuilt = if rebuilt_factors.len() == 1 {
            rebuilt_factors[0]
        } else {
            build_balanced_mul(ctx, &rebuilt_factors)
        };
        return (rebuilt, term_sign.negate());
    }

    (term_expr, term_sign)
}

pub(super) fn exprs_match_for_cancellation(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    if let Some(hit) = CANCELLATION_MATCH_MEMO.with(|m| m.borrow().get(&(lhs, rhs)).copied()) {
        return hit;
    }
    let result = exprs_match_for_cancellation_uncached(ctx, lhs, rhs);
    CANCELLATION_MATCH_MEMO.with(|m| {
        m.borrow_mut().insert((lhs, rhs), result);
    });
    result
}

pub(super) fn exprs_match_for_cancellation_leaf(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    if term_has_matrix_product_factor(ctx, lhs) || term_has_matrix_product_factor(ctx, rhs) {
        // Non-commutative matrix product present: only order-preserving
        // structural equality is sound (see `term_has_matrix_product_factor`).
        return compare_expr(ctx, lhs, rhs) == Ordering::Equal;
    }
    if compare_expr(ctx, lhs, rhs) == Ordering::Equal
        || cas_math::expr_domain::exprs_equivalent(ctx, lhs, rhs)
        || exprs_equal_up_to_add_term_order(ctx, lhs, rhs)
        || exprs_equal_up_to_mul_factor_order_and_sign(ctx, lhs, rhs)
    {
        return true;
    }

    let lhs_normalized = cas_math::canonical_forms::normalize_core(ctx, lhs);
    let rhs_normalized = cas_math::canonical_forms::normalize_core(ctx, rhs);
    compare_expr(ctx, lhs_normalized, rhs_normalized) == Ordering::Equal
        || cas_math::expr_domain::exprs_equivalent(ctx, lhs_normalized, rhs_normalized)
        || exprs_equal_up_to_add_term_order(ctx, lhs_normalized, rhs_normalized)
        || exprs_equal_up_to_mul_factor_order_and_sign(ctx, lhs_normalized, rhs_normalized)
}

pub(super) fn exprs_match_after_default_simplify(
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

pub(super) fn expr_contains_hyperbolic_builtin(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    expr_contains_any_builtin(
        ctx,
        expr,
        &[
            BuiltinFn::Sinh,
            BuiltinFn::Cosh,
            BuiltinFn::Tanh,
            BuiltinFn::Asinh,
            BuiltinFn::Acosh,
            BuiltinFn::Atanh,
        ],
    )
}

pub(super) fn expr_matches_negation_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    target: cas_ast::ExprId,
) -> bool {
    let neg_target = ctx.add(Expr::Neg(target));
    exprs_match_for_cancellation(ctx, expr, neg_target)
}

pub(super) fn expr_matches_negation_after_default_simplify(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    target: cas_ast::ExprId,
) -> bool {
    let neg_target = ctx.add(Expr::Neg(target));
    exprs_match_after_default_simplify(ctx, expr, neg_target)
}

pub(super) fn build_signed_sum_expr(
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

pub(super) fn normalize_additive_scope_expr(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let terms = AddView::from_expr(ctx, expr).terms;
    build_signed_sum_expr(ctx, &terms)
}

pub(super) fn additive_scopes_match_after_default_simplify(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    let lhs_normalized = normalize_additive_scope_expr(ctx, lhs);
    let rhs_normalized = normalize_additive_scope_expr(ctx, rhs);
    let lhs_terms = AddView::from_expr(ctx, lhs_normalized).terms;
    let rhs_terms = AddView::from_expr(ctx, rhs_normalized).terms;
    if lhs_terms.len() != rhs_terms.len() {
        return false;
    }

    let lhs_signed_terms: Vec<_> = lhs_terms
        .into_iter()
        .map(|(term_expr, sign)| signed_term_expr(ctx, term_expr, sign))
        .collect();
    let rhs_signed_terms: Vec<_> = rhs_terms
        .into_iter()
        .map(|(term_expr, sign)| signed_term_expr(ctx, term_expr, sign))
        .collect();

    let mut used_rhs = vec![false; rhs_signed_terms.len()];
    for lhs_term in lhs_signed_terms {
        let Some(match_index) =
            rhs_signed_terms
                .iter()
                .enumerate()
                .find_map(|(index, rhs_term)| {
                    (!used_rhs[index]
                        && exprs_match_after_default_simplify(ctx, lhs_term, *rhs_term))
                    .then_some(index)
                })
        else {
            return false;
        };
        used_rhs[match_index] = true;
    }

    true
}

pub(super) fn negate_additive_scope_expr(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let terms = AddView::from_expr(ctx, expr).terms;
    let negated_terms: Vec<_> = terms
        .into_iter()
        .map(|(term_expr, sign)| (term_expr, sign.negate()))
        .collect();
    build_signed_sum_expr(ctx, &negated_terms)
}

pub(super) fn try_build_fast_small_polynomial_residual_child_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    maybe_small_polynomial_expand_zero_candidate(ctx, expr)
        .then(|| try_build_fast_small_polynomial_expansion_zero_scope_rewrite(ctx, expr))
        .flatten()
}

pub(super) fn try_build_direct_trig_square_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((arg, is_sum)) = extract_trig_binomial_square_identity_data(ctx, source) else {
            continue;
        };
        let target_candidate = build_trig_binomial_square_target(ctx, arg, is_sum);
        if exprs_match_for_cancellation(ctx, target_candidate, target)
            || exprs_match_after_default_simplify(ctx, target_candidate, target)
        {
            return Some(build_direct_trig_square_equivalence_rewrite(
                ctx, lhs_core, rhs_core, target, arg, is_sum,
            ));
        }
    }

    None
}

pub(super) fn expr_contains_division_node(ctx: &cas_ast::Context, root: cas_ast::ExprId) -> bool {
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Div(_, _) => return true,
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Pow(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    false
}

pub(crate) fn try_build_exact_zero_identity_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    if let Some((lhs_core, rhs_core)) = extract_two_term_core_difference(ctx, expr) {
        if is_atanh_common_log_definition_mismatch_pair(ctx, lhs_core, rhs_core) {
            return None;
        }
    }

    if let Some(rewrite) = try_build_exact_zero_identity_rewrite_direct(ctx, expr) {
        return Some(rewrite);
    }

    let view = AddView::from_expr(ctx, expr);
    if !(2..=4).contains(&view.terms.len()) {
        return None;
    }

    let flipped_terms: Vec<_> = view
        .terms
        .iter()
        .map(|(term_expr, term_sign)| (*term_expr, term_sign.negate()))
        .collect();
    let flipped_expr = build_signed_sum_expr(ctx, &flipped_terms);
    let child_rewrite = try_build_exact_zero_identity_rewrite_direct(ctx, flipped_expr)?;

    let mut rewrite = Rewrite::with_local(
        ctx.num(0),
        child_rewrite.description.clone(),
        expr,
        ctx.num(0),
    )
    .requires_all(child_rewrite.required_conditions.clone())
    .assume_all(child_rewrite.assumption_events.clone());

    if let Some(poly_proof) = child_rewrite.poly_proof.clone() {
        rewrite = rewrite.poly_proof(poly_proof);
    }

    rewrite.substeps = child_rewrite.substeps.clone();
    Some(rewrite)
}

pub(super) fn build_mul_expr_from_factors(
    ctx: &mut cas_ast::Context,
    factors: &[cas_ast::ExprId],
) -> cas_ast::ExprId {
    match factors {
        [] => ctx.num(1),
        [single] => *single,
        _ => build_balanced_mul(ctx, factors),
    }
}

pub(super) fn extract_common_multiplicative_residual_sum(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
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

    let first_factors = flatten_mul_chain(ctx, normalized_terms.first()?.0);
    if first_factors.is_empty() {
        return None;
    }

    let mut used_by_term: Vec<Vec<bool>> = normalized_terms
        .iter()
        .map(|(term_expr, _)| vec![false; flatten_mul_chain(ctx, *term_expr).len()])
        .collect();
    let factor_lists: Vec<Vec<_>> = normalized_terms
        .iter()
        .map(|(term_expr, _)| flatten_mul_chain(ctx, *term_expr))
        .collect();

    let mut common = Vec::new();
    for first_factor in first_factors {
        let mut matched_indexes = Vec::new();
        let mut all_match = true;

        for (term_index, factors) in factor_lists.iter().enumerate().skip(1) {
            let Some(factor_index) =
                factors
                    .iter()
                    .enumerate()
                    .find_map(|(factor_index, factor)| {
                        (!used_by_term[term_index][factor_index]
                            && compare_expr(ctx, *factor, first_factor) == Ordering::Equal)
                            .then_some(factor_index)
                    })
            else {
                all_match = false;
                break;
            };
            matched_indexes.push((term_index, factor_index));
        }

        if !all_match {
            continue;
        }

        common.push(first_factor);
        for (term_index, factor_index) in matched_indexes {
            used_by_term[term_index][factor_index] = true;
        }

        if let Some(first_index) =
            factor_lists[0]
                .iter()
                .enumerate()
                .find_map(|(factor_index, factor)| {
                    (!used_by_term[0][factor_index]
                        && compare_expr(ctx, *factor, first_factor) == Ordering::Equal)
                        .then_some(factor_index)
                })
        {
            used_by_term[0][first_index] = true;
        }
    }

    if common.is_empty() {
        return None;
    }

    let residual_terms: Vec<_> = normalized_terms
        .iter()
        .enumerate()
        .map(|(term_index, (_term_expr, term_sign))| {
            let residual_factors: Vec<_> = factor_lists[term_index]
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(factor_index, factor)| {
                    (!used_by_term[term_index][factor_index]).then_some(factor)
                })
                .collect();
            (
                build_mul_expr_from_factors(ctx, &residual_factors),
                *term_sign,
            )
        })
        .collect();

    let common_factor = build_mul_expr_from_factors(ctx, &common);
    let residual_expr = build_signed_sum_expr(ctx, &residual_terms);
    let one = ctx.num(1);
    if compare_expr(ctx, common_factor, one) == Ordering::Equal
        || compare_expr(ctx, residual_expr, expr) == Ordering::Equal
    {
        return None;
    }
    Some((common_factor, residual_expr))
}

pub(super) fn try_build_stripped_zero_log_identity_child_rewrite(
    ctx: &mut cas_ast::Context,
    residual_expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let stripped_residual = strip_single_additive_zero_term(ctx, residual_expr)?;
    if !expr_contains_any_builtin(
        ctx,
        stripped_residual,
        &[
            BuiltinFn::Ln,
            BuiltinFn::Log,
            BuiltinFn::Log2,
            BuiltinFn::Log10,
            BuiltinFn::Abs,
        ],
    ) {
        return None;
    }

    let child_rewrite = try_build_exact_zero_identity_rewrite(ctx, stripped_residual)?;
    let zero = ctx.num(0);
    (compare_expr(ctx, child_rewrite.final_expr(), zero) == Ordering::Equal)
        .then_some(child_rewrite)
}

pub(super) fn try_build_exact_zero_identity_rewrite_direct(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    try_build_exact_zero_identity_rewrite_direct_impl(ctx, expr, true)
}

pub(super) fn sign_to_i64(sign: Sign) -> i64 {
    match sign {
        Sign::Pos => 1,
        Sign::Neg => -1,
    }
}

pub(crate) fn extract_two_term_core_difference(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) => {
            let (lhs_expr, lhs_sign) = normalize_core_difference_term(ctx, lhs, Sign::Pos);
            let (rhs_expr, rhs_sign) = normalize_core_difference_term(ctx, rhs, Sign::Pos);
            Some((
                apply_sign_to_expr(ctx, sign_to_i64(lhs_sign), lhs_expr),
                apply_sign_to_expr(ctx, sign_to_i64(rhs_sign), rhs_expr),
            ))
        }
        Expr::Add(lhs, rhs) => match ctx.get(rhs).clone() {
            Expr::Neg(inner) => {
                let (lhs_expr, lhs_sign) = normalize_core_difference_term(ctx, lhs, Sign::Pos);
                let (rhs_expr, rhs_sign) = normalize_core_difference_term(ctx, inner, Sign::Pos);
                Some((
                    apply_sign_to_expr(ctx, sign_to_i64(lhs_sign), lhs_expr),
                    apply_sign_to_expr(ctx, sign_to_i64(rhs_sign), rhs_expr),
                ))
            }
            _ => {
                let terms = AddView::from_expr(ctx, expr).terms;
                if terms.len() != 2 {
                    return None;
                }
                let (first_expr, first_sign) =
                    normalize_core_difference_term(ctx, terms[0].0, terms[0].1);
                let (second_expr, second_sign) =
                    normalize_core_difference_term(ctx, terms[1].0, terms[1].1);
                Some((
                    apply_sign_to_expr(ctx, sign_to_i64(first_sign), first_expr),
                    apply_sign_to_expr(ctx, sign_to_i64(second_sign).checked_neg()?, second_expr),
                ))
            }
        },
        _ => {
            let terms = AddView::from_expr(ctx, expr).terms;
            if terms.len() != 2 {
                return None;
            }
            let (first_expr, first_sign) =
                normalize_core_difference_term(ctx, terms[0].0, terms[0].1);
            let (second_expr, second_sign) =
                normalize_core_difference_term(ctx, terms[1].0, terms[1].1);
            Some((
                apply_sign_to_expr(ctx, sign_to_i64(first_sign), first_expr),
                apply_sign_to_expr(ctx, sign_to_i64(second_sign).checked_neg()?, second_expr),
            ))
        }
    }
}

pub(super) fn try_build_direct_trig_cos_diff_sin_diff_quotient_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((rewritten, den)) =
            try_rewrite_trig_cos_diff_sin_diff_quotient_for_cancellation(ctx, source)
        else {
            continue;
        };

        if exprs_match_for_cancellation(ctx, rewritten, target)
            || exprs_match_after_default_simplify(ctx, rewritten, target)
        {
            return Some(
                Rewrite::with_local(ctx.num(0), "Cos-Diff / Sin-Diff Quotient", source, target)
                    .requires(crate::ImplicitCondition::NonZero(den)),
            );
        }
    }

    None
}

pub(super) fn try_build_direct_trig_exact_quarter_phase_shift_pair_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (linear_side, shifted_side) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(linear_groups) =
            extract_structural_unit_linear_phase_shift_pair_side(ctx, linear_side)
        else {
            continue;
        };
        let Some(shifted_groups) =
            extract_structural_unit_exact_quarter_shifted_phase_shift_pair_side(ctx, shifted_side)
        else {
            continue;
        };

        if !structural_unit_phase_shift_pair_groups_match(ctx, &linear_groups, &shifted_groups) {
            continue;
        }

        return Some(
            Rewrite::with_local(ctx.num(0), "Phase Shift Identity", lhs_core, rhs_core)
                .substep(
                    "Aplicar identidad de desfase",
                    vec![
                        "Reescribir cada par sin(u) + cos(u) como sqrt(2)·sin(u + pi/4)."
                            .to_string(),
                    ],
                )
                .substep(
                    "Cancelar términos iguales",
                    vec![
                        "Tras reescribir ambos pares, los dos lados quedan idénticos y la diferencia se anula."
                            .to_string(),
                    ],
                ),
        );
    }

    None
}

pub(super) fn try_build_direct_trig_double_angle_contraction_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewritten) =
            try_rewrite_signed_double_angle_contraction_for_cancellation(ctx, source)
        else {
            continue;
        };

        if exprs_match_for_cancellation(ctx, rewritten, target)
            || exprs_match_after_default_simplify(ctx, rewritten, target)
        {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Double Angle Contraction",
                lhs_core,
                rhs_core,
            ));
        }
    }

    None
}

pub(super) fn try_build_direct_hyperbolic_sinh_cubic_polynomial_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewritten) =
            try_rewrite_hyperbolic_product_sum_sinh_cubic_polynomial_for_cancellation(ctx, source)
        else {
            continue;
        };

        if exprs_match_for_cancellation(ctx, rewritten, target)
            || exprs_match_after_default_simplify(ctx, rewritten, target)
        {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Hyperbolic Product-to-Sum and Triple-Angle Identity",
                lhs_core,
                rhs_core,
            ));
        }
    }

    None
}

pub(super) fn try_build_direct_tanh_exp_definition_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewritten) = try_rewrite_tanh_exp_definition_for_cancellation(ctx, source) else {
            continue;
        };

        if exprs_match_for_cancellation(ctx, rewritten, target) {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Recognize Hyperbolic from Exponential",
                lhs_core,
                rhs_core,
            ));
        }
    }

    None
}

pub(crate) fn try_build_direct_sub_fraction_combination_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        if !matches!(ctx.get(target), Expr::Div(_, _)) {
            continue;
        }
        if let Some(rewritten) =
            try_rewrite_symbolic_difference_squares_telescoping_for_cancellation(ctx, source)
        {
            if exprs_match_for_cancellation(ctx, rewritten, target) {
                return Some(Rewrite::with_local(
                    ctx.num(0),
                    "Subtract Fractions",
                    lhs_core,
                    rhs_core,
                ));
            }
        }
        let Some(rewritten) =
            try_rewrite_scaled_sub_fraction_combination_for_cancellation(ctx, source)
        else {
            continue;
        };
        let residual = ctx.add(Expr::Sub(rewritten, target));

        if exprs_match_for_cancellation(ctx, rewritten, target)
            || exprs_match_after_default_simplify(ctx, rewritten, target)
            || is_zero_after_default_simplify(ctx, residual)
        {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Subtract Fractions",
                lhs_core,
                rhs_core,
            ));
        }
    }

    None
}

pub(super) fn try_build_direct_safe_hyperbolic_core_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    let nested_default_simplify = default_simplify_nesting_depth() > 0;
    if !expr_contains_hyperbolic_builtin(ctx, lhs_core)
        && !expr_contains_hyperbolic_builtin(ctx, rhs_core)
    {
        return None;
    }

    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let source_has_direct_hyperbolic = expr_contains_direct_hyperbolic_builtin(ctx, source);
        let source_has_atanh = expr_contains_any_builtin(ctx, source, &[BuiltinFn::Atanh]);
        if source_has_atanh
            && !source_has_direct_hyperbolic
            && !expr_contains_any_builtin(
                ctx,
                target,
                &[BuiltinFn::Ln, BuiltinFn::Log, BuiltinFn::Log10],
            )
        {
            continue;
        }

        let Some((rewritten, description)) =
            try_rewrite_safe_direct_hyperbolic_equivalence_for_cancellation(ctx, source)
        else {
            continue;
        };

        if exprs_match_for_cancellation(ctx, rewritten, target)
            || (!nested_default_simplify
                && exprs_match_after_default_simplify(ctx, rewritten, target))
        {
            return Some(Rewrite::with_local(
                ctx.num(0),
                description,
                lhs_core,
                rhs_core,
            ));
        }
    }

    None
}

pub(super) fn try_build_direct_core_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    let pair_sample = profiling.then(|| {
        format!(
            "{}  ||  {}",
            render_expr_for_orchestrator_profile(ctx, lhs_core),
            render_expr_for_orchestrator_profile(ctx, rhs_core)
        )
    });
    let profile_route = |label: &'static str| {
        if profiling {
            let _ =
                run_profiled_orchestrator_option_section(label, pair_sample.clone(), || Some(()));
        }
    };

    if exprs_match_shallow_noncall_for_cancellation(ctx, lhs_core, rhs_core) {
        profile_route("rule.direct_core_equivalence.route.direct_match");
        return Some(Rewrite::with_local(
            ctx.num(0),
            "Equivalent Residual Cancellation",
            lhs_core,
            rhs_core,
        ));
    }

    if let Some(rewrite) =
        try_build_direct_reciprocal_half_power_shared_denominator_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route(
            "rule.direct_core_equivalence.route.reciprocal_half_power_shared_denominator",
        );
        return Some(rewrite);
    }

    if let Some(rewrite) = try_build_direct_scaled_reciprocal_half_power_shared_denominator_rewrite(
        ctx, lhs_core, rhs_core,
    ) {
        profile_route(
            "rule.direct_core_equivalence.route.scaled_reciprocal_half_power_shared_denominator",
        );
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_scaled_reciprocal_half_power_over_base_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.scaled_reciprocal_half_power_over_base");
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_negative_even_root_power_reciprocal_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.negative_even_root_power_reciprocal");
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_reciprocal_half_power_product_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.reciprocal_half_power_product");
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_scaled_reciprocal_half_power_product_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.scaled_reciprocal_half_power_product");
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_common_sqrt_denominator_fraction_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.common_sqrt_denominator_fraction");
        return Some(rewrite);
    }

    if let Some(rewrite) = try_build_direct_sqrt_over_base_fraction_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.sqrt_over_base_fraction");
        return Some(rewrite);
    }

    if let Some(rewrite) = try_build_direct_rationalized_common_sqrt_denominator_fraction_rewrite(
        ctx, lhs_core, rhs_core,
    ) {
        profile_route(
            "rule.direct_core_equivalence.route.rationalized_common_sqrt_denominator_fraction",
        );
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_tanh_exp_definition_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.tanh_exp");
        return Some(rewrite);
    }

    if let Some(rewrite) = try_build_direct_trig_square_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.trig_square");
        return Some(rewrite);
    }
    if let Some(rewrite) =
        try_build_direct_trig_product_to_sum_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.trig_product_to_sum");
        return Some(rewrite);
    }
    if let Some(rewrite) =
        try_build_direct_trig_sum_to_product_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.trig_sum_to_product");
        return Some(rewrite);
    }

    if let Some(rewrite) = try_build_direct_trig_exact_quarter_phase_shift_pair_equivalence_rewrite(
        ctx, lhs_core, rhs_core,
    ) {
        profile_route("rule.direct_core_equivalence.route.phase_shift_pair");
        return Some(rewrite);
    }

    let maybe_phase_shift_pair_residual =
        if expr_contains_any_builtin(ctx, lhs_core, &[BuiltinFn::Sin, BuiltinFn::Cos])
            || expr_contains_any_builtin(ctx, rhs_core, &[BuiltinFn::Sin, BuiltinFn::Cos])
        {
            let residual_expr = ctx.add(Expr::Sub(lhs_core, rhs_core));
            (AddView::from_expr(ctx, residual_expr).terms.len() == 6).then_some(residual_expr)
        } else {
            None
        };

    if let Some(residual_expr) = maybe_phase_shift_pair_residual {
        if let Some(rewrite) =
            try_build_repeated_trig_phase_shift_pair_zero_rewrite(ctx, residual_expr)
        {
            profile_route("rule.direct_core_equivalence.route.repeated_phase_shift_pair");
            return Some(rewrite);
        }
    }

    let has_hyperbolic_core = expr_contains_hyperbolic_builtin(ctx, lhs_core)
        || expr_contains_hyperbolic_builtin(ctx, rhs_core);
    if has_hyperbolic_core {
        if let Some(rewrite) =
            try_build_direct_safe_hyperbolic_core_equivalence_rewrite(ctx, lhs_core, rhs_core)
        {
            profile_route("rule.direct_core_equivalence.route.safe_hyperbolic");
            return Some(rewrite);
        }
    }

    if exprs_match_for_cancellation(ctx, lhs_core, rhs_core) {
        profile_route("rule.direct_core_equivalence.route.direct_match");
        return Some(Rewrite::with_local(
            ctx.num(0),
            "Equivalent Residual Cancellation",
            lhs_core,
            rhs_core,
        ));
    }

    if let Some(rewrite) = try_build_direct_trig_cos_diff_sin_diff_quotient_equivalence_rewrite(
        ctx, lhs_core, rhs_core,
    ) {
        profile_route("rule.direct_core_equivalence.route.cos_diff_sin_diff_quotient");
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_sum_diff_cubes_quotient_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.sum_diff_cubes_quotient");
        return Some(rewrite);
    }

    if try_find_trig_phase_shift_cancellation_match(ctx, lhs_core, rhs_core, false)
        .or_else(|| try_find_trig_phase_shift_cancellation_match(ctx, rhs_core, lhs_core, false))
        .is_some()
    {
        profile_route("rule.direct_core_equivalence.route.phase_shift_identity");
        return Some(Rewrite::with_local(
            ctx.num(0),
            "Phase Shift Identity",
            lhs_core,
            rhs_core,
        ));
    }

    let residual_expr = ctx.add(Expr::Sub(lhs_core, rhs_core));
    if let Some(rewrite) = try_build_repeated_trig_phase_shift_pair_zero_rewrite(ctx, residual_expr)
    {
        profile_route("rule.direct_core_equivalence.route.repeated_phase_shift_residual");
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_cos_product_telescoping_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.cos_product_telescoping");
        return Some(rewrite);
    }

    if let Some(rewrite) = try_build_direct_finite_sum_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.finite_sum");
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_dirichlet_core_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.dirichlet");
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_finite_product_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.finite_product");
        return Some(rewrite);
    }
    if let Some(rewrite) =
        try_build_direct_trig_power_reduction_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.trig_power_reduction");
        return Some(rewrite);
    }
    if let Some(rewrite) =
        try_build_direct_trig_double_angle_contraction_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.double_angle_contraction");
        return Some(rewrite);
    }
    if let Some(rewrite) = try_build_direct_trig_cos_double_angle_polynomial_equivalence_rewrite(
        ctx, lhs_core, rhs_core,
    ) {
        profile_route("rule.direct_core_equivalence.route.cos_double_angle_poly");
        return Some(rewrite);
    }
    if let Some(rewrite) = try_build_direct_trig_mixed_double_angle_polynomial_equivalence_rewrite(
        ctx, lhs_core, rhs_core,
    ) {
        profile_route("rule.direct_core_equivalence.route.mixed_double_angle_poly");
        return Some(rewrite);
    }
    if let Some(rewrite) =
        try_build_direct_trig_double_angle_cos_variant_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.double_angle_cos_variant");
        return Some(rewrite);
    }
    if maybe_two_term_embedded_double_angle_expansion_candidate(ctx, lhs_core, rhs_core) {
        if let Some(rewrite) =
            try_build_direct_trig_embedded_double_angle_expansion_equivalence_rewrite(
                ctx, lhs_core, rhs_core,
            )
        {
            profile_route("rule.direct_core_equivalence.route.embedded_double_angle");
            return Some(rewrite);
        }
    }
    if let Some(rewrite) = try_build_direct_multi_angle_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.multi_angle");
        return Some(rewrite);
    }
    if let Some(rewrite) =
        try_build_direct_recursive_trig_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.recursive_trig");
        return Some(rewrite);
    }

    if let Some((rewritten, description)) =
        try_rewrite_exact_trig_equivalence_for_cancellation(ctx, lhs_core)
    {
        if exprs_match_for_cancellation(ctx, rewritten, rhs_core)
            || exprs_match_after_default_simplify(ctx, rewritten, rhs_core)
        {
            profile_route("rule.direct_core_equivalence.route.exact_trig_lhs");
            return Some(Rewrite::with_local(
                ctx.num(0),
                description,
                lhs_core,
                rhs_core,
            ));
        }
    }
    if let Some((rewritten, description)) =
        try_rewrite_exact_trig_equivalence_for_cancellation(ctx, rhs_core)
    {
        if exprs_match_for_cancellation(ctx, rewritten, lhs_core)
            || exprs_match_after_default_simplify(ctx, rewritten, lhs_core)
        {
            profile_route("rule.direct_core_equivalence.route.exact_trig_rhs");
            return Some(Rewrite::with_local(
                ctx.num(0),
                description,
                lhs_core,
                rhs_core,
            ));
        }
    }

    let has_tanh_core = expr_contains_any_builtin(ctx, lhs_core, &[BuiltinFn::Tanh])
        || expr_contains_any_builtin(ctx, rhs_core, &[BuiltinFn::Tanh]);
    if has_tanh_core {
        if let Some((rewritten, description)) =
            try_rewrite_exact_hyperbolic_equivalence_for_cancellation(ctx, lhs_core)
        {
            if exprs_match_for_cancellation(ctx, rewritten, rhs_core)
                || exprs_match_after_default_simplify(ctx, rewritten, rhs_core)
            {
                profile_route("rule.direct_core_equivalence.route.hyperbolic_lhs_tanh");
                return Some(Rewrite::with_local(
                    ctx.num(0),
                    description,
                    lhs_core,
                    rhs_core,
                ));
            }
        }
        if let Some((rewritten, description)) =
            try_rewrite_exact_hyperbolic_equivalence_for_cancellation(ctx, rhs_core)
        {
            if exprs_match_for_cancellation(ctx, rewritten, lhs_core)
                || exprs_match_after_default_simplify(ctx, rewritten, lhs_core)
            {
                profile_route("rule.direct_core_equivalence.route.hyperbolic_rhs_tanh");
                return Some(Rewrite::with_local(
                    ctx.num(0),
                    description,
                    lhs_core,
                    rhs_core,
                ));
            }
        }
    }

    if classify_symbolic_scale_sum_profile_detail(ctx, lhs_core) == "grouped_multi_scale"
        && grouped_symbolic_scale_sum_matches_target_for_cancellation(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.symbolic_scale_sum_lhs");
        return Some(Rewrite::with_local(
            ctx.num(0),
            "Equivalent Residual Cancellation",
            lhs_core,
            rhs_core,
        ));
    }

    if let Some(rewritten) = try_rewrite_simple_symbolic_scale_sum_for_cancellation(ctx, lhs_core) {
        if exprs_match_for_cancellation(ctx, rewritten, rhs_core) {
            profile_route("rule.direct_core_equivalence.route.symbolic_scale_sum_lhs");
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Equivalent Residual Cancellation",
                lhs_core,
                rhs_core,
            ));
        }
    }

    if classify_symbolic_scale_sum_profile_detail(ctx, rhs_core) == "grouped_multi_scale"
        && grouped_symbolic_scale_sum_matches_target_for_cancellation(ctx, rhs_core, lhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.symbolic_scale_sum_rhs");
        return Some(Rewrite::with_local(
            ctx.num(0),
            "Equivalent Residual Cancellation",
            lhs_core,
            rhs_core,
        ));
    }

    if let Some(rewritten) = try_rewrite_simple_symbolic_scale_sum_for_cancellation(ctx, rhs_core) {
        if exprs_match_for_cancellation(ctx, rewritten, lhs_core) {
            profile_route("rule.direct_core_equivalence.route.symbolic_scale_sum_rhs");
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Equivalent Residual Cancellation",
                lhs_core,
                rhs_core,
            ));
        }
    }

    if let Some(rewrite) =
        try_build_direct_trig_reciprocal_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.trig_reciprocal");
        return Some(rewrite);
    }

    if let Some(rewrite) = try_build_direct_trig_ratio_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.trig_ratio");
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_log_expansion_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.log_expansion");
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_build_direct_log_chain_product_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.log_chain_product");
        return Some(rewrite);
    }

    if let Some(false) =
        reject_noncall_vs_surface_symbolic_trig_before_default_simplify(ctx, lhs_core, rhs_core)
    {
        profile_route(
            "rule.direct_core_equivalence.route.default_simplify_noncall_surface_trig_reject",
        );
        return None;
    }

    if let Some(false) = reject_atomic_noncall_pair_before_default_simplify(ctx, lhs_core, rhs_core)
    {
        profile_route(
            "rule.direct_core_equivalence.route.default_simplify_atomic_noncall_pair_reject",
        );
        return None;
    }

    if let Some(false) =
        reject_scaled_symbolic_atom_mismatch_before_default_simplify(ctx, lhs_core, rhs_core)
    {
        profile_route(
            "rule.direct_core_equivalence.route.default_simplify_scaled_symbolic_atom_mismatch_reject",
        );
        return None;
    }

    if let Some(false) =
        reject_noncall_product_vs_division_shared_numerator_scale_before_default_simplify(
            ctx, lhs_core, rhs_core,
        )
    {
        profile_route(
            "rule.direct_core_equivalence.route.default_simplify_product_division_shared_scale_reject",
        );
        return None;
    }

    if let Some(false) =
        reject_surface_plain_cross_trig_pair_before_default_simplify(ctx, lhs_core, rhs_core)
    {
        profile_route(
            "rule.direct_core_equivalence.route.default_simplify_plain_cross_trig_reject",
        );
        return None;
    }

    if let Some(false) = reject_shifted_surface_trig_symbolic_base_mismatch_before_default_simplify(
        ctx, lhs_core, rhs_core,
    ) {
        profile_route(
            "rule.direct_core_equivalence.route.default_simplify_shifted_surface_trig_symbolic_base_mismatch_reject",
        );
        return None;
    }

    if let Some(false) = reject_scaled_surface_trig_power_vs_numeric_atom_before_default_simplify(
        ctx, lhs_core, rhs_core,
    ) {
        profile_route(
            "rule.direct_core_equivalence.route.default_simplify_surface_trig_power_numeric_atom_reject",
        );
        return None;
    }

    if let Some(false) =
        reject_plain_surface_trig_power_gap_before_default_simplify(ctx, lhs_core, rhs_core)
    {
        profile_route(
            "rule.direct_core_equivalence.route.default_simplify_surface_trig_power_gap_reject",
        );
        return None;
    }

    if let Some(false) =
        reject_hyperbolic_additive_mismatch_before_default_simplify(ctx, lhs_core, rhs_core)
    {
        profile_route(
            "rule.direct_core_equivalence.route.default_simplify_hyperbolic_additive_mismatch_reject",
        );
        return None;
    }

    if let Some(false) =
        reject_obvious_hyperbolic_pair_before_default_simplify(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.direct_core_equivalence.route.default_simplify_hyperbolic_pair_reject");
        return None;
    }

    if let Some(false) = reject_negated_log_pair_without_reciprocal_shape_before_default_simplify(
        ctx, lhs_core, rhs_core,
    ) {
        profile_route(
            "rule.direct_core_equivalence.route.default_simplify_negated_log_nonreciprocal_reject",
        );
        return None;
    }

    if has_hyperbolic_core {
        if let Some((rewritten, description)) =
            try_rewrite_exact_hyperbolic_equivalence_for_cancellation(ctx, lhs_core)
        {
            if exprs_match_for_cancellation(ctx, rewritten, rhs_core) {
                profile_route("rule.direct_core_equivalence.route.hyperbolic_exact_pre_default");
                return Some(Rewrite::with_local(
                    ctx.num(0),
                    description,
                    lhs_core,
                    rhs_core,
                ));
            }
        }
        if let Some((rewritten, description)) =
            try_rewrite_exact_hyperbolic_equivalence_for_cancellation(ctx, rhs_core)
        {
            if exprs_match_for_cancellation(ctx, rewritten, lhs_core) {
                profile_route("rule.direct_core_equivalence.route.hyperbolic_exact_pre_default");
                return Some(Rewrite::with_local(
                    ctx.num(0),
                    description,
                    lhs_core,
                    rhs_core,
                ));
            }
        }
    }

    let default_simplify_match = if profiling {
        let label = direct_core_default_simplify_profile_label(ctx, lhs_core, rhs_core);
        run_profiled_orchestrator_option_section(label, pair_sample.clone(), || {
            exprs_match_after_default_simplify(ctx, lhs_core, rhs_core).then_some(())
        })
        .is_some()
    } else {
        exprs_match_after_default_simplify(ctx, lhs_core, rhs_core)
    };
    if default_simplify_match {
        profile_route("rule.direct_core_equivalence.route.default_simplify_match");
        return Some(Rewrite::with_local(
            ctx.num(0),
            "Equivalent Residual Cancellation",
            lhs_core,
            rhs_core,
        ));
    }

    if !has_tanh_core {
        if let Some((rewritten, description)) =
            try_rewrite_exact_hyperbolic_equivalence_for_cancellation(ctx, lhs_core)
        {
            if exprs_match_for_cancellation(ctx, rewritten, rhs_core)
                || exprs_match_after_default_simplify(ctx, rewritten, rhs_core)
            {
                profile_route("rule.direct_core_equivalence.route.hyperbolic_lhs");
                return Some(Rewrite::with_local(
                    ctx.num(0),
                    description,
                    lhs_core,
                    rhs_core,
                ));
            }
        }
        if let Some((rewritten, description)) =
            try_rewrite_exact_hyperbolic_equivalence_for_cancellation(ctx, rhs_core)
        {
            if exprs_match_for_cancellation(ctx, rewritten, lhs_core)
                || exprs_match_after_default_simplify(ctx, rewritten, lhs_core)
            {
                profile_route("rule.direct_core_equivalence.route.hyperbolic_rhs");
                return Some(Rewrite::with_local(
                    ctx.num(0),
                    description,
                    lhs_core,
                    rhs_core,
                ));
            }
        }
    }

    None
}

pub(super) fn expr_is_atomic_noncall(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Variable(_) | Expr::SessionRef(_) | Expr::Number(_) | Expr::Constant(_) => true,
        Expr::Neg(inner) => matches!(
            ctx.get(*inner),
            Expr::Variable(_) | Expr::SessionRef(_) | Expr::Number(_) | Expr::Constant(_)
        ),
        _ => false,
    }
}

pub(super) fn try_build_fast_trig_residual_identity_child_rewrite(
    ctx: &mut cas_ast::Context,
    residual_expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let residual_expr = if let Some((lhs_core, rhs_core)) =
        extract_two_term_core_difference(ctx, residual_expr)
    {
        if let Some(rewrite) =
            try_build_direct_trig_exact_quarter_phase_shift_pair_equivalence_rewrite(
                ctx, lhs_core, rhs_core,
            )
        {
            return Some(rewrite);
        }
        if let Some(rewrite) = try_build_direct_trig_cos_diff_sin_diff_quotient_equivalence_rewrite(
            ctx, lhs_core, rhs_core,
        ) {
            return Some(rewrite);
        }
        if let Some(rewrite) =
            try_build_direct_trig_product_to_sum_equivalence_rewrite(ctx, lhs_core, rhs_core)
        {
            return Some(rewrite);
        }
        if let Some(rewrite) =
            try_build_direct_trig_sum_to_product_equivalence_rewrite(ctx, lhs_core, rhs_core)
        {
            return Some(rewrite);
        }
        if let Some(rewrite) = try_build_direct_trig_double_angle_cos_variant_equivalence_rewrite(
            ctx, lhs_core, rhs_core,
        ) {
            return Some(rewrite);
        }
        if let Some(rewrite) = try_build_direct_trig_double_angle_contraction_equivalence_rewrite(
            ctx, lhs_core, rhs_core,
        ) {
            return Some(rewrite);
        }
        ctx.add(Expr::Sub(lhs_core, rhs_core))
    } else {
        residual_expr
    };

    let term_count = AddView::from_expr(ctx, residual_expr).terms.len();
    if !(2..=4).contains(&term_count) {
        return None;
    }
    if !expr_contains_any_builtin(
        ctx,
        residual_expr,
        &[
            BuiltinFn::Sin,
            BuiltinFn::Cos,
            BuiltinFn::Tan,
            BuiltinFn::Cot,
            BuiltinFn::Sec,
            BuiltinFn::Csc,
        ],
    ) {
        return None;
    }
    if expr_contains_plain_trig_angle_identity_term(ctx, residual_expr) {
        return None;
    }

    let rewrite = try_build_exact_zero_identity_rewrite_direct(ctx, residual_expr)?;
    let zero = ctx.num(0);
    (compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal).then_some(rewrite)
}

pub(super) fn try_build_fast_multiterm_hyperbolic_residual_child_rewrite(
    ctx: &mut cas_ast::Context,
    residual_expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let term_count = AddView::from_expr(ctx, residual_expr).terms.len();
    if !(3..=4).contains(&term_count) {
        return None;
    }
    if !expr_contains_hyperbolic_builtin(ctx, residual_expr) {
        return None;
    }

    let rewrite = try_build_exact_zero_identity_rewrite(ctx, residual_expr)?;
    let zero = ctx.num(0);
    (compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal).then_some(rewrite)
}

pub(super) fn try_build_repeated_trig_phase_shift_pair_zero_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let normalized_expr = normalize_additive_scope_expr(ctx, expr);
    let view = AddView::from_expr(ctx, normalized_expr);
    if view.terms.len() != 6 {
        return None;
    }

    if let Some(rewrite) =
        try_build_fast_repeated_trig_phase_shift_pair_zero_rewrite(ctx, &view.terms)
    {
        return Some(rewrite);
    }

    let zero = ctx.num(0);

    for first_index in 0..view.terms.len().saturating_sub(2) {
        for second_index in (first_index + 1)..view.terms.len().saturating_sub(1) {
            for third_index in (second_index + 1)..view.terms.len() {
                let first_terms = [
                    view.terms[first_index],
                    view.terms[second_index],
                    view.terms[third_index],
                ];
                let first_expr = build_signed_sum_expr(ctx, &first_terms);
                let Some(first_rewrite) =
                    try_build_exact_trig_phase_shift_zero_scope_rewrite(ctx, first_expr)
                else {
                    continue;
                };
                if compare_expr(ctx, first_rewrite.final_expr(), zero) != Ordering::Equal {
                    continue;
                }

                let remaining_terms: Vec<_> = view
                    .terms
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(index, term)| {
                        (index != first_index && index != second_index && index != third_index)
                            .then_some(term)
                    })
                    .collect();
                if remaining_terms.len() != 3 {
                    continue;
                }

                let second_expr = build_signed_sum_expr(ctx, &remaining_terms);
                let Some(second_rewrite) =
                    try_build_exact_trig_phase_shift_zero_scope_rewrite(ctx, second_expr)
                else {
                    continue;
                };
                if compare_expr(ctx, second_rewrite.final_expr(), zero) != Ordering::Equal {
                    continue;
                }

                let mut rewrite = Rewrite::with_local(
                    zero,
                    first_rewrite.description.clone(),
                    normalized_expr,
                    zero,
                )
                .requires_all(first_rewrite.required_conditions.clone())
                .requires_all(second_rewrite.required_conditions.clone())
                .assume_all(first_rewrite.assumption_events.clone())
                .assume_all(second_rewrite.assumption_events.clone());

                if let Some(poly_proof) = first_rewrite.poly_proof.clone() {
                    rewrite = rewrite.poly_proof(poly_proof);
                }
                let mut substeps = if first_rewrite.substeps.is_empty() {
                    vec![build_phase_shift_zero_substep(ctx, first_expr)]
                } else {
                    first_rewrite.substeps.clone()
                };
                if second_rewrite.substeps.is_empty() {
                    substeps.push(build_phase_shift_zero_substep(ctx, second_expr));
                } else {
                    substeps.extend(second_rewrite.substeps.clone());
                }
                rewrite.substeps = substeps;

                return Some(rewrite);
            }
        }
    }

    None
}

pub(super) fn try_build_exact_zero_shared_passthrough_difference_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    if !has_plausible_shared_additive_passthrough_difference_shape(ctx, expr) {
        return None;
    }

    let (lhs_core, rhs_core) = extract_shared_additive_passthrough_difference_cores(ctx, expr)?;
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    let pair_sample = profiling.then(|| {
        format!(
            "{}  ||  {}",
            render_expr_for_orchestrator_profile(ctx, lhs_core),
            render_expr_for_orchestrator_profile(ctx, rhs_core)
        )
    });
    let profile_route = |label: &'static str| {
        if profiling {
            let _ =
                run_profiled_orchestrator_option_section(label, pair_sample.clone(), || Some(()));
        }
    };
    let residual_expr = ctx.add(Expr::Sub(lhs_core, rhs_core));
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.sub_fraction",
        &pair_sample,
        || try_build_direct_sub_fraction_combination_equivalence_rewrite(ctx, lhs_core, rhs_core),
    ) {
        profile_route("rule.shared_passthrough.route.sub_fraction");
        return Some(rewrite);
    }
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.tanh_exp",
        &pair_sample,
        || try_build_direct_tanh_exp_definition_equivalence_rewrite(ctx, lhs_core, rhs_core),
    ) {
        profile_route("rule.shared_passthrough.route.tanh_exp");
        return Some(rewrite);
    }
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.trig_product_to_sum",
        &pair_sample,
        || try_build_direct_trig_product_to_sum_equivalence_rewrite(ctx, lhs_core, rhs_core),
    ) {
        profile_route("rule.shared_passthrough.route.trig_product_to_sum");
        return Some(rewrite);
    }
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.trig_sum_to_product",
        &pair_sample,
        || try_build_direct_trig_sum_to_product_equivalence_rewrite(ctx, lhs_core, rhs_core),
    ) {
        profile_route("rule.shared_passthrough.route.trig_sum_to_product");
        return Some(rewrite);
    }
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.trig_square",
        &pair_sample,
        || try_build_direct_trig_square_equivalence_rewrite(ctx, lhs_core, rhs_core),
    ) {
        profile_route("rule.shared_passthrough.route.trig_square");
        return Some(rewrite);
    }
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.sinh_cubic",
        &pair_sample,
        || {
            try_build_direct_hyperbolic_sinh_cubic_polynomial_equivalence_rewrite(
                ctx, lhs_core, rhs_core,
            )
        },
    ) {
        profile_route("rule.shared_passthrough.route.sinh_cubic");
        return Some(rewrite);
    }
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.repeated_phase_shift_pair",
        &pair_sample,
        || try_build_repeated_trig_phase_shift_pair_zero_rewrite(ctx, residual_expr),
    ) {
        profile_route("rule.shared_passthrough.route.repeated_phase_shift_pair");
        return Some(rewrite);
    }
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.phase_shift_quarter_pair",
        &pair_sample,
        || {
            try_build_direct_trig_exact_quarter_phase_shift_pair_equivalence_rewrite(
                ctx, lhs_core, rhs_core,
            )
        },
    ) {
        profile_route("rule.shared_passthrough.route.phase_shift_quarter_pair");
        return Some(rewrite);
    }
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.safe_hyperbolic",
        &pair_sample,
        || try_build_direct_safe_hyperbolic_core_equivalence_rewrite(ctx, lhs_core, rhs_core),
    ) {
        profile_route("rule.shared_passthrough.route.safe_hyperbolic");
        return Some(rewrite);
    }
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.repeated_phase_shift_pair_late",
        &pair_sample,
        || try_build_repeated_trig_phase_shift_pair_zero_rewrite(ctx, residual_expr),
    ) {
        profile_route("rule.shared_passthrough.route.repeated_phase_shift_pair_late");
        return Some(rewrite);
    }
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.direct_identity",
        &pair_sample,
        || try_build_exact_zero_identity_rewrite_direct(ctx, residual_expr),
    ) {
        let zero = ctx.num(0);
        if compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal {
            profile_route("rule.shared_passthrough.route.direct_identity");
            return Some(rewrite);
        }
    }
    if let Some(rewrite) = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.square_base_equivalence",
        &pair_sample,
        || try_build_shared_passthrough_square_base_equivalence_rewrite(ctx, lhs_core, rhs_core),
    ) {
        profile_route("rule.shared_passthrough.route.square_base_equivalence");
        return Some(rewrite);
    }
    let normalized_residual = normalize_additive_scope_expr(ctx, residual_expr);

    let child_rewrite = run_profiled_shared_passthrough_probe(
        profiling,
        "rule.shared_passthrough.try.tail_fast_multiterm_hyperbolic",
        &pair_sample,
        || try_build_fast_multiterm_hyperbolic_residual_child_rewrite(ctx, residual_expr),
    )
    .inspect(|_| {
        profile_route("rule.shared_passthrough.route.tail_fast_multiterm_hyperbolic");
    })
    .or_else(|| {
        run_profiled_shared_passthrough_probe(
            profiling,
            "rule.shared_passthrough.try.tail_safe_hyperbolic",
            &pair_sample,
            || try_build_direct_safe_hyperbolic_core_equivalence_rewrite(ctx, lhs_core, rhs_core),
        )
        .inspect(|_| {
            profile_route("rule.shared_passthrough.route.tail_safe_hyperbolic");
        })
    })
    .or_else(|| {
        run_profiled_shared_passthrough_probe(
            profiling,
            "rule.shared_passthrough.try.tail_stripped_zero_log",
            &pair_sample,
            || try_build_stripped_zero_log_identity_child_rewrite(ctx, residual_expr),
        )
        .inspect(|_| {
            profile_route("rule.shared_passthrough.route.tail_stripped_zero_log");
        })
    })
    .or_else(|| {
        run_profiled_shared_passthrough_probe(
            profiling,
            "rule.shared_passthrough.try.tail_stripped_zero_log_normalized",
            &pair_sample,
            || try_build_stripped_zero_log_identity_child_rewrite(ctx, normalized_residual),
        )
        .inspect(|_| {
            profile_route("rule.shared_passthrough.route.tail_stripped_zero_log_normalized");
        })
    })
    .or_else(|| {
        run_profiled_shared_passthrough_probe(
            profiling,
            "rule.shared_passthrough.try.tail_fast_trig_residual",
            &pair_sample,
            || try_build_fast_trig_residual_identity_child_rewrite(ctx, residual_expr),
        )
        .inspect(|_| {
            profile_route("rule.shared_passthrough.route.tail_fast_trig_residual");
        })
    })
    .or_else(|| {
        run_profiled_shared_passthrough_probe(
            profiling,
            "rule.shared_passthrough.try.tail_fast_trig_residual_normalized",
            &pair_sample,
            || try_build_fast_trig_residual_identity_child_rewrite(ctx, normalized_residual),
        )
        .inspect(|_| {
            profile_route("rule.shared_passthrough.route.tail_fast_trig_residual_normalized");
        })
    })
    .or_else(|| {
        run_profiled_shared_passthrough_probe(
            profiling,
            "rule.shared_passthrough.try.tail_fast_small_polynomial",
            &pair_sample,
            || try_build_fast_small_polynomial_residual_child_rewrite(ctx, residual_expr),
        )
        .inspect(|_| {
            profile_route("rule.shared_passthrough.route.tail_fast_small_polynomial");
        })
    })
    .or_else(|| {
        run_profiled_shared_passthrough_probe(
            profiling,
            "rule.shared_passthrough.try.tail_fast_small_polynomial_normalized",
            &pair_sample,
            || try_build_fast_small_polynomial_residual_child_rewrite(ctx, normalized_residual),
        )
        .inspect(|_| {
            profile_route("rule.shared_passthrough.route.tail_fast_small_polynomial_normalized");
        })
    })
    .or_else(|| {
        run_profiled_shared_passthrough_probe(
            profiling,
            "rule.shared_passthrough.try.tail_direct_core_equivalence",
            &pair_sample,
            || try_build_direct_core_equivalence_rewrite(ctx, lhs_core, rhs_core),
        )
        .inspect(|_| {
            profile_route("rule.shared_passthrough.route.tail_direct_core_equivalence");
            profile_shared_passthrough_tail_direct_core_family(
                ctx,
                lhs_core,
                rhs_core,
                pair_sample.clone(),
            );
        })
    })
    .or_else(|| {
        run_profiled_shared_passthrough_probe(
            profiling,
            "rule.shared_passthrough.try.tail_exact_zero_identity",
            &pair_sample,
            || try_build_exact_zero_identity_rewrite(ctx, residual_expr),
        )
        .inspect(|_| {
            profile_route("rule.shared_passthrough.route.tail_exact_zero_identity");
        })
    })?;
    let zero = ctx.num(0);

    (compare_expr(ctx, child_rewrite.final_expr(), zero) == Ordering::Equal)
        .then_some(child_rewrite)
}

pub(super) fn exprs_equal_up_to_same_denominator(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    let Expr::Div(lhs_num, lhs_den) = ctx.get(lhs).clone() else {
        return false;
    };
    let Expr::Div(rhs_num, rhs_den) = ctx.get(rhs).clone() else {
        return false;
    };

    compare_expr(ctx, lhs_den, rhs_den) == Ordering::Equal
        && (exprs_match_for_cancellation_leaf(ctx, lhs_num, rhs_num)
            || exprs_equal_up_to_add_term_multiset_for_cancellation(ctx, lhs_num, rhs_num))
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
pub(super) fn exprs_equal_up_to_add_term_order(
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
pub(super) fn exprs_equal_up_to_add_term_multiset_for_cancellation(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    let lhs_terms = AddView::from_expr(ctx, lhs).terms;
    let rhs_terms = AddView::from_expr(ctx, rhs).terms;
    if lhs_terms.len() != rhs_terms.len() {
        return false;
    }

    let lhs_signed_terms: Vec<_> = lhs_terms
        .into_iter()
        .map(|(term_expr, sign)| signed_term_expr(ctx, term_expr, sign))
        .collect();
    let rhs_signed_terms: Vec<_> = rhs_terms
        .into_iter()
        .map(|(term_expr, sign)| signed_term_expr(ctx, term_expr, sign))
        .collect();

    let mut used_rhs = vec![false; rhs_signed_terms.len()];
    for lhs_term in lhs_signed_terms {
        let Some(match_index) =
            rhs_signed_terms
                .iter()
                .enumerate()
                .find_map(|(index, rhs_term)| {
                    (!used_rhs[index]
                        && exprs_match_for_cancellation_leaf(ctx, lhs_term, *rhs_term))
                    .then_some(index)
                })
        else {
            return false;
        };
        used_rhs[match_index] = true;
    }

    true
}
fn collect_signed_mul_factors(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    sign: &mut i8,
    out: &mut Vec<cas_ast::ExprId>,
) {
    match ctx.get(expr).clone() {
        Expr::Mul(lhs, rhs) => {
            collect_signed_mul_factors(ctx, lhs, sign, out);
            collect_signed_mul_factors(ctx, rhs, sign, out);
        }
        Expr::Neg(inner) => {
            *sign *= -1;
            collect_signed_mul_factors(ctx, inner, sign, out);
        }
        Expr::Number(n) if n < num_rational::BigRational::from_integer(0.into()) => {
            *sign *= -1;
            out.push(ctx.add(Expr::Number(-n)));
        }
        _ => out.push(expr),
    }
}
pub(super) fn exprs_equal_up_to_mul_factor_order_and_sign(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    let mut lhs_sign = 1_i8;
    let mut rhs_sign = 1_i8;
    let mut lhs_factors = Vec::new();
    let mut rhs_factors = Vec::new();

    collect_signed_mul_factors(ctx, lhs, &mut lhs_sign, &mut lhs_factors);
    collect_signed_mul_factors(ctx, rhs, &mut rhs_sign, &mut rhs_factors);

    if lhs_sign != rhs_sign || lhs_factors.len() != rhs_factors.len() {
        return false;
    }

    // Matrix multiplication is non-commutative, so two products are equal only
    // when their factors line up in the SAME order: `A·B` and `B·A` are
    // generally different. Sorting the factor lists (the commutative-ring
    // assumption) would wrongly collapse the commutator `A·B − B·A` to 0.
    // When any factor is a matrix, keep the factors in evaluation order.
    let has_matrix_factor = lhs_factors
        .iter()
        .chain(rhs_factors.iter())
        .any(|factor| matches!(ctx.get(*factor), Expr::Matrix { .. }));
    if !has_matrix_factor {
        lhs_factors.sort_by(|a, b| compare_expr(ctx, *a, *b));
        rhs_factors.sort_by(|a, b| compare_expr(ctx, *a, *b));
    }

    lhs_factors
        .iter()
        .zip(rhs_factors.iter())
        .all(|(lhs_factor, rhs_factor)| {
            compare_expr(ctx, *lhs_factor, *rhs_factor) == Ordering::Equal
        })
}
/// The value domain of the ENCLOSING top-level pipeline (armed in
/// `simplify_pipeline_inner` alongside the probe budget; RealOnly outside any
/// armed pipeline, e.g. unit contexts). Speculative probe pipelines AND the
/// structural real-only zero-identity matchers consult it — the matcher
/// signatures (`(ctx, expr) -> bool`, ~45 fns) predate the value-domain axis,
/// and this ambient carries the axis to them without rewriting every helper
/// (audit 2026-07-30, causa C2).
pub(crate) fn ambient_pipeline_value_domain() -> crate::semantics::ValueDomain {
    DEFAULT_SIMPLIFY_PROBE_VALUE_DOMAIN.with(|vd| vd.get())
}
pub(super) fn strip_term_negation(
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
/// Returns true if `root` contains a matrix literal that participates as a
/// factor in a multiplication, division, or power — i.e. a non-commutative
/// matrix product. Matrix literals that appear only as additive terms
/// (`A + B`) or as standalone values do NOT count, so the commutative
/// add-reordering matchers below stay enabled for matrix sums.
///
/// Matrix multiplication is non-commutative (`A·B ≠ B·A` in general), so any
/// matcher that reorders multiplicative factors (the sorted factor compare,
/// `normalize_core`, `exprs_equivalent`) is UNSOUND when a term is a matrix
/// product: it would treat `A·B` and `B·A` as equal and cancel the commutator
/// `A·B − B·A` to 0. When this predicate holds we restrict cancellation to the
/// order-preserving `compare_expr`, which still cancels genuinely identical
/// products (`A·B − A·B → 0`) while leaving `A·B − B·A` to evaluate to the
/// true commutator.
pub(crate) fn term_has_matrix_product_factor(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
) -> bool {
    // `in_product` marks subtrees reached through a multiplicative factor
    // position (Mul/Div operand, or the base of a power).
    let mut stack = vec![(root, false)];
    while let Some((expr, in_product)) = stack.pop() {
        // Some callers compare expressions carrying non-arena sentinel ExprIds
        // (e.g. `ln_base_sentinel()`), whose index is out of bounds. Skip them:
        // a sentinel can never be a matrix literal, and dereferencing it panics.
        let Some(node) = ctx.nodes.get(expr.index()) else {
            continue;
        };
        match node {
            Expr::Matrix { data, .. } => {
                if in_product {
                    return true;
                }
                // Matrix entries are independent scalar expressions.
                for &entry in data.iter() {
                    stack.push((entry, false));
                }
            }
            Expr::Mul(lhs, rhs) | Expr::Div(lhs, rhs) => {
                stack.push((*lhs, true));
                stack.push((*rhs, true));
            }
            Expr::Pow(base, exponent) => {
                // A power is a repeated product (`M^n = M·M·…`), so a matrix base
                // is a non-commutative product factor even at the root (e.g. the
                // difference-of-squares factoring `M^2 − N^2 = (M−N)(M+N)` is
                // unsound when `M·N ≠ N·M`). The exponent is a scalar position.
                stack.push((*base, true));
                stack.push((*exponent, false));
            }
            Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => {
                stack.push((*lhs, false));
                stack.push((*rhs, false));
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push((*inner, in_product)),
            Expr::Function(_, args) => {
                for &arg in args.iter() {
                    stack.push((arg, false));
                }
            }
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    false
}
pub(super) fn exprs_match_for_cancellation_uncached(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    if term_has_matrix_product_factor(ctx, lhs) || term_has_matrix_product_factor(ctx, rhs) {
        // Non-commutative matrix product present: only order-preserving
        // structural equality is sound (see `term_has_matrix_product_factor`).
        return compare_expr(ctx, lhs, rhs) == Ordering::Equal;
    }
    if compare_expr(ctx, lhs, rhs) == Ordering::Equal
        || cas_math::expr_domain::exprs_equivalent(ctx, lhs, rhs)
        || exprs_equal_up_to_add_term_order(ctx, lhs, rhs)
        || exprs_equal_up_to_add_term_multiset_for_cancellation(ctx, lhs, rhs)
        || exprs_equal_up_to_mul_factor_order_and_sign(ctx, lhs, rhs)
        || exprs_equal_up_to_same_denominator(ctx, lhs, rhs)
    {
        return true;
    }

    let lhs_unheld = cas_ast::hold::unwrap_internal_hold(ctx, lhs);
    let rhs_unheld = cas_ast::hold::unwrap_internal_hold(ctx, rhs);
    if (lhs_unheld != lhs || rhs_unheld != rhs)
        && (compare_expr(ctx, lhs_unheld, rhs_unheld) == Ordering::Equal
            || cas_math::expr_domain::exprs_equivalent(ctx, lhs_unheld, rhs_unheld)
            || exprs_equal_up_to_add_term_order(ctx, lhs_unheld, rhs_unheld)
            || exprs_equal_up_to_add_term_multiset_for_cancellation(ctx, lhs_unheld, rhs_unheld)
            || exprs_equal_up_to_mul_factor_order_and_sign(ctx, lhs_unheld, rhs_unheld)
            || exprs_equal_up_to_same_denominator(ctx, lhs_unheld, rhs_unheld))
    {
        return true;
    }

    let lhs_normalized = cas_math::canonical_forms::normalize_core(ctx, lhs);
    let rhs_normalized = cas_math::canonical_forms::normalize_core(ctx, rhs);
    compare_expr(ctx, lhs_normalized, rhs_normalized) == Ordering::Equal
        || cas_math::expr_domain::exprs_equivalent(ctx, lhs_normalized, rhs_normalized)
        || exprs_equal_up_to_add_term_order(ctx, lhs_normalized, rhs_normalized)
        || exprs_equal_up_to_add_term_multiset_for_cancellation(ctx, lhs_normalized, rhs_normalized)
        || exprs_equal_up_to_mul_factor_order_and_sign(ctx, lhs_normalized, rhs_normalized)
        || exprs_equal_up_to_same_denominator(ctx, lhs_normalized, rhs_normalized)
}
pub(super) fn signed_term_expr(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    sign: Sign,
) -> cas_ast::ExprId {
    match sign {
        Sign::Pos => expr,
        Sign::Neg => ctx.add(Expr::Neg(expr)),
    }
}

pub(super) fn canonicalize_nested_integer_powers(
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

pub(super) fn extract_sqrt_argument(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
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

pub(super) fn distribute_symbolic_scale_sum_term_for_cancellation(
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
