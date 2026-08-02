//! Orquestador: familia `support` (troceo P1) — declarada en D2-1 (2026-08-02)
//! **API interna del pipeline de shortcuts** del orquestador.
//!
//! Aquí viven las primitivas COMPARTIDAS entre las familias de shortcuts
//! (las 41 que la medición de la campaña destapó — llamadas desde ≥4
//! familias, 606 aristas entrantes — más las promovidas por D2-1:
//! `extract_common_multiplicative_residual_sum_root`,
//! `extract_direct_two_linear_shift_product_root`,
//! `build_root_shortcut_step_from_rewrite`). Doctrina calcada de
//! `rules/arithmetic/support.rs` (D1): un helper compartido entre familias
//! con contenido NEUTRO vive aquí y se declara; uno con contenido de
//! familia vive en el fichero de su familia (el fichero ES la frontera —
//! `trig`, `trig_angles`, `hyperbolic`, `logs_exp`, `fractions`,
//! `zero_detection`, `radicals_powers`, `pairing`). Los flujos entre
//! familias del pipeline (pairing→trig_angles, zero_detection→trig_angles…)
//! son PARTE DEL DISEÑO del orquestador — este directorio es un pipeline
//! entrelazado, no un motor con disparadores; la métrica de progreso es el
//! % de aristas intra (baseline 28,1%, `decoupling_metrics_baseline.json`)
//! y la auditoría es el callgraph del arnés D0, no un invariante de cierre.
//!
//! Ver la cabecera de `orchestrator.rs` para el contexto del troceo.

use super::*;

pub(super) fn run_profiled_root_shortcut<T>(
    name: &'static str,
    run: impl FnOnce() -> Option<T>,
) -> Option<T> {
    if !crate::orchestrator_shortcut_profiler::should_profile_orchestrator_shortcut(name) {
        return run();
    }

    let start = web_time::Instant::now();
    let result = run();
    crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_attempt(
        name,
        result.is_some(),
        start.elapsed(),
    );
    result
}

pub(super) fn render_expr_for_orchestrator_profile(ctx: &Context, expr: ExprId) -> String {
    crate::orchestrator_shortcut_profiler::render_expr_shape_for_orchestrator_profile(ctx, expr)
}

pub(super) fn is_symbolic_atom(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Variable(_) => true,
        // `∞` and `undefined` are NOT finite symbolic atoms: they ABSORB / PROPAGATE rather than stay
        // in a symbolic binomial. Treating `∞` as an atom let the plain-mode "symbolic atom + literal"
        // shortcut return `∞ + 1` unevaluated (diverging from `--steps`, which absorbs it to `∞`).
        // The genuine finite constants (`π`, `e`, `i`) are still atoms (`π + 1` stays `π + 1`).
        Expr::Constant(c) => !matches!(
            c,
            cas_ast::Constant::Infinity | cas_ast::Constant::Undefined
        ),
        _ => false,
    }
}

pub(super) fn expr_contains_any_builtin_local(
    ctx: &Context,
    root: ExprId,
    builtins: &[BuiltinFn],
) -> bool {
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Function(fn_id, args) => {
                if let Some(builtin) = ctx.builtin_of(*fn_id) {
                    if builtins.contains(&builtin) {
                        return true;
                    }
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
            Expr::Neg(inner) => stack.push(*inner),
            _ => {}
        }
    }
    false
}

pub(super) fn expr_contains_division_node_local(ctx: &Context, root: ExprId) -> bool {
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

pub(super) fn expr_contains_sqrt_or_half_power_local(ctx: &Context, root: ExprId) -> bool {
    let mut stack = vec![root];
    let half = BigRational::new(1.into(), 2.into());

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
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    false
}

pub(super) fn build_root_shortcut_parent_ctx(
    options: &crate::phase::SimplifyOptions,
    ctx: &Context,
    expr: ExprId,
) -> crate::parent_context::ParentContext {
    crate::parent_context::ParentContext::root()
        .with_domain_mode(options.shared.semantics.domain_mode)
        .with_value_domain(options.shared.semantics.value_domain)
        .with_inv_trig(options.shared.semantics.inv_trig)
        .with_goal(options.goal)
        .with_context_mode(options.shared.context_mode)
        .with_simplify_purpose(options.simplify_purpose)
        .with_autoexpand_binomials(options.shared.autoexpand_binomials)
        .with_heuristic_poly(options.shared.heuristic_poly)
        .with_expand_mode_flag(options.expand_mode)
        .with_root_expr(ctx, expr)
}

pub(super) fn finish_standard_root_shortcut(
    _ctx: &Context,
    before: ExprId,
    rewrite: crate::rule::Rewrite,
    rule_name: &'static str,
    collect_steps: bool,
) -> (ExprId, Vec<Step>) {
    let result = rewrite.final_expr();
    let mut shortcut_steps = Vec::new();
    if collect_steps {
        let mut step = Step::new_compact(&rewrite.description, rule_name, before, rewrite.new_expr);
        step.global_before = Some(before);
        step.global_after = Some(rewrite.new_expr);
        step.importance = crate::step::ImportanceLevel::High;
        shortcut_steps.push(step);

        let mut current = rewrite.new_expr;
        for chained in rewrite.chained {
            let mut chain_step = Step::new_compact(
                chained.description.as_ref(),
                rule_name,
                current,
                chained.after,
            );
            chain_step.global_before = Some(current);
            chain_step.global_after = Some(chained.after);
            chain_step.importance = chained
                .importance
                .unwrap_or(crate::step::ImportanceLevel::High);
            shortcut_steps.push(chain_step);
            current = chained.after;
        }
    }
    (result, shortcut_steps)
}

pub(super) fn build_root_shortcut_compact_step(
    before: ExprId,
    after: ExprId,
    description: &'static str,
    rule_name: &'static str,
) -> Step {
    let mut step = Step::new_compact(description, rule_name, before, after);
    step.global_before = Some(before);
    step.global_after = Some(after);
    step.importance = crate::step::ImportanceLevel::High;
    step
}

pub(super) fn finish_root_shortcut_with_rewrite_meta(
    ctx: &Context,
    before: ExprId,
    rewrite: crate::rule::Rewrite,
    rule_name: &'static str,
    collect_steps: bool,
) -> (ExprId, Vec<Step>) {
    let result = rewrite.final_expr();
    let mut shortcut_steps = Vec::new();
    if collect_steps {
        shortcut_steps.push(build_root_shortcut_step_from_rewrite(
            ctx, before, &rewrite, rule_name,
        ));

        let mut current = rewrite.new_expr;
        for chained in &rewrite.chained {
            let mut chain_step = Step::with_snapshots(
                chained.description.as_ref(),
                rule_name,
                current,
                chained.after,
                smallvec::SmallVec::<[crate::step::PathStep; 8]>::new(),
                Some(ctx),
                current,
                chained.after,
            );
            chain_step.importance = chained
                .importance
                .unwrap_or(crate::step::ImportanceLevel::High);
            {
                let meta = chain_step.meta_mut();
                meta.before_local = chained.before_local;
                meta.after_local = chained.after_local;
                meta.assumption_events = chained.assumption_events.clone();
                meta.required_conditions = chained.required_conditions.clone();
                meta.poly_proof = chained.poly_proof.clone();
                meta.is_chained = true;
            }
            shortcut_steps.push(chain_step);
            current = chained.after;
        }
    }
    (result, shortcut_steps)
}

pub(super) fn build_signed_sum_expr_root(ctx: &mut Context, terms: &[(ExprId, Sign)]) -> ExprId {
    let Some((first_expr, first_sign)) = terms.first().copied() else {
        return ctx.num(0);
    };
    let mut acc = if first_sign == Sign::Neg {
        ctx.add(Expr::Neg(first_expr))
    } else {
        first_expr
    };
    for (expr, sign) in terms.iter().copied().skip(1) {
        let term = if sign == Sign::Neg {
            ctx.add(Expr::Neg(expr))
        } else {
            expr
        };
        acc = ctx.add(Expr::Add(acc, term));
    }
    acc
}

pub(super) fn normalize_signed_add_term_root(
    ctx: &mut Context,
    term_expr: ExprId,
    term_sign: Sign,
) -> (ExprId, Sign) {
    match ctx.get(term_expr).clone() {
        Expr::Neg(inner) => (inner, flip_add_sign_root(term_sign)),
        Expr::Number(n) if n < BigRational::zero() => {
            (ctx.add(Expr::Number(-n)), flip_add_sign_root(term_sign))
        }
        _ => {
            let (coeff, base) = extract_coef_and_base(ctx, term_expr);
            if coeff < BigRational::zero() {
                let normalized = if coeff == BigRational::from_integer((-1).into()) {
                    base
                } else {
                    let positive_coeff = ctx.add(Expr::Number(-coeff));
                    smart_mul(ctx, positive_coeff, base)
                };
                (normalized, flip_add_sign_root(term_sign))
            } else {
                (term_expr, term_sign)
            }
        }
    }
}

pub(super) fn extract_plain_sin_or_cos_arg_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
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

pub(super) fn matches_direct_recursive_hyperbolic_sinh_sum_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (single_expr, expanded_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((BuiltinFn::Sinh, angle_arg)) =
            extract_plain_sinh_or_cosh_arg_root(ctx, single_expr)
        else {
            continue;
        };

        let view = AddView::from_expr(ctx, expanded_expr);
        if view.terms.len() != 2 || !view.terms.iter().all(|(_, sign)| *sign == Sign::Pos) {
            continue;
        }

        let Some(((lhs_fn_a, lhs_arg_a), (lhs_fn_b, lhs_arg_b))) =
            extract_plain_hyperbolic_product_pair_args_root(ctx, view.terms[0].0)
        else {
            continue;
        };
        let Some(((rhs_fn_a, rhs_arg_a), (rhs_fn_b, rhs_arg_b))) =
            extract_plain_hyperbolic_product_pair_args_root(ctx, view.terms[1].0)
        else {
            continue;
        };

        let lhs_is_sinh_cosh = matches!(
            (lhs_fn_a, lhs_fn_b),
            (BuiltinFn::Sinh, BuiltinFn::Cosh) | (BuiltinFn::Cosh, BuiltinFn::Sinh)
        );
        let rhs_is_sinh_cosh = matches!(
            (rhs_fn_a, rhs_fn_b),
            (BuiltinFn::Sinh, BuiltinFn::Cosh) | (BuiltinFn::Cosh, BuiltinFn::Sinh)
        );
        if !lhs_is_sinh_cosh || !rhs_is_sinh_cosh {
            continue;
        }

        let lhs_sinh_arg = if lhs_fn_a == BuiltinFn::Sinh {
            lhs_arg_a
        } else {
            lhs_arg_b
        };
        let lhs_cosh_arg = if lhs_fn_a == BuiltinFn::Cosh {
            lhs_arg_a
        } else {
            lhs_arg_b
        };
        let rhs_sinh_arg = if rhs_fn_a == BuiltinFn::Sinh {
            rhs_arg_a
        } else {
            rhs_arg_b
        };
        let rhs_cosh_arg = if rhs_fn_a == BuiltinFn::Cosh {
            rhs_arg_a
        } else {
            rhs_arg_b
        };

        if compare_expr(ctx, lhs_sinh_arg, rhs_cosh_arg) != Ordering::Equal
            || compare_expr(ctx, lhs_cosh_arg, rhs_sinh_arg) != Ordering::Equal
        {
            continue;
        }

        if matches_angle_sum_or_diff_arg_root(ctx, angle_arg, lhs_sinh_arg, lhs_cosh_arg, true) {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_recursive_hyperbolic_cosh_sum_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (single_expr, expanded_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((BuiltinFn::Cosh, angle_arg)) =
            extract_plain_sinh_or_cosh_arg_root(ctx, single_expr)
        else {
            continue;
        };

        let view = AddView::from_expr(ctx, expanded_expr);
        if view.terms.len() != 2 || !view.terms.iter().all(|(_, sign)| *sign == Sign::Pos) {
            continue;
        }

        let Some(((lhs_fn_a, lhs_arg_a), (lhs_fn_b, lhs_arg_b))) =
            extract_plain_hyperbolic_product_pair_args_root(ctx, view.terms[0].0)
        else {
            continue;
        };
        let Some(((rhs_fn_a, rhs_arg_a), (rhs_fn_b, rhs_arg_b))) =
            extract_plain_hyperbolic_product_pair_args_root(ctx, view.terms[1].0)
        else {
            continue;
        };

        let lhs_is_cosh_cosh = lhs_fn_a == BuiltinFn::Cosh && lhs_fn_b == BuiltinFn::Cosh;
        let rhs_is_sinh_sinh = rhs_fn_a == BuiltinFn::Sinh && rhs_fn_b == BuiltinFn::Sinh;
        let lhs_is_sinh_sinh = lhs_fn_a == BuiltinFn::Sinh && lhs_fn_b == BuiltinFn::Sinh;
        let rhs_is_cosh_cosh = rhs_fn_a == BuiltinFn::Cosh && rhs_fn_b == BuiltinFn::Cosh;

        let (arg_u, arg_v) = if lhs_is_cosh_cosh && rhs_is_sinh_sinh {
            if !matches_unordered_expr_pair_root(ctx, lhs_arg_a, lhs_arg_b, rhs_arg_a, rhs_arg_b) {
                continue;
            }
            (lhs_arg_a, lhs_arg_b)
        } else if lhs_is_sinh_sinh && rhs_is_cosh_cosh {
            if !matches_unordered_expr_pair_root(ctx, lhs_arg_a, lhs_arg_b, rhs_arg_a, rhs_arg_b) {
                continue;
            }
            (rhs_arg_a, rhs_arg_b)
        } else {
            continue;
        };

        if matches_angle_sum_or_diff_arg_root(ctx, angle_arg, arg_u, arg_v, true) {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_cos_square_diff_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    matches_direct_negative_double_cos_square_diff_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_cos_minus_sin_square_diff_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_positive_double_cos_square_diff_pair_root(ctx, lhs_core, rhs_core)
}

pub(super) fn matches_unordered_expr_pair_root(
    ctx: &Context,
    lhs_a: ExprId,
    lhs_b: ExprId,
    rhs_a: ExprId,
    rhs_b: ExprId,
) -> bool {
    (compare_expr(ctx, lhs_a, rhs_a) == Ordering::Equal
        && compare_expr(ctx, lhs_b, rhs_b) == Ordering::Equal)
        || (compare_expr(ctx, lhs_a, rhs_b) == Ordering::Equal
            && compare_expr(ctx, lhs_b, rhs_a) == Ordering::Equal)
}

pub(super) fn matches_direct_angle_sum_diff_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (angle_expr, product_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((angle_fn, angle_arg)) = extract_plain_sin_or_cos_arg_root(ctx, angle_expr) else {
            continue;
        };

        let view = AddView::from_expr(ctx, product_expr);
        if view.terms.len() != 2 {
            continue;
        }

        if angle_fn == BuiltinFn::Sin {
            let mut first_pair = None;
            let mut first_sign = None;
            let mut second_pair = None;
            let mut second_sign = None;
            let mut bad_term = false;

            for (term_expr, term_sign) in view.terms {
                let Some(pair) = extract_plain_mixed_sin_cos_product_pair_args_root(ctx, term_expr)
                else {
                    bad_term = true;
                    break;
                };
                if first_pair.is_none() {
                    first_pair = Some(pair);
                    first_sign = Some(term_sign);
                } else if second_pair.is_none() {
                    second_pair = Some(pair);
                    second_sign = Some(term_sign);
                } else {
                    bad_term = true;
                    break;
                }
            }

            let (
                Some((first_sin_arg, first_cos_arg)),
                Some(first_sign),
                Some((second_sin_arg, second_cos_arg)),
                Some(second_sign),
            ) = (first_pair, first_sign, second_pair, second_sign)
            else {
                continue;
            };
            if bad_term
                || !matches_unordered_expr_pair_root(
                    ctx,
                    first_sin_arg,
                    first_cos_arg,
                    second_sin_arg,
                    second_cos_arg,
                )
            {
                continue;
            }

            let is_sum = match (first_sign, second_sign) {
                (Sign::Pos, Sign::Pos) => true,
                (Sign::Pos, Sign::Neg) | (Sign::Neg, Sign::Pos) => false,
                _ => continue,
            };
            if matches_angle_sum_or_diff_arg_root(
                ctx,
                angle_arg,
                first_sin_arg,
                first_cos_arg,
                is_sum,
            ) {
                return true;
            }

            continue;
        }

        if angle_fn != BuiltinFn::Cos {
            continue;
        }

        let mut cos_pair = None;
        let mut sin_pair = None;
        let mut sin_sign = None;
        let mut bad_term = false;

        for (term_expr, term_sign) in view.terms {
            if let Some(pair) =
                extract_plain_trig_product_pair_args_root(ctx, term_expr, BuiltinFn::Cos)
            {
                if cos_pair.is_some() || term_sign != Sign::Pos {
                    bad_term = true;
                    break;
                }
                cos_pair = Some(pair);
                continue;
            }

            if let Some(pair) =
                extract_plain_trig_product_pair_args_root(ctx, term_expr, BuiltinFn::Sin)
            {
                if sin_pair.is_some() {
                    bad_term = true;
                    break;
                }
                sin_pair = Some(pair);
                sin_sign = Some(term_sign);
                continue;
            }

            bad_term = true;
            break;
        }

        let (Some((cos_lhs, cos_rhs)), Some((sin_lhs, sin_rhs)), Some(sin_sign)) =
            (cos_pair, sin_pair, sin_sign)
        else {
            continue;
        };
        if bad_term || !matches_unordered_expr_pair_root(ctx, cos_lhs, cos_rhs, sin_lhs, sin_rhs) {
            continue;
        }

        let is_sum = sin_sign == Sign::Neg;
        if matches_angle_sum_or_diff_arg_root(ctx, angle_arg, cos_lhs, cos_rhs, is_sum) {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_hyperbolic_cosh_cubic_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut double_angle = None;
    let mut linear_cosh = None;
    let mut cubic_cosh = None;

    for (term_expr, term_sign) in view.terms {
        if let Some(arg) = extract_scaled_sinh_double_angle_sinh_term_arg_root(ctx, term_expr) {
            if double_angle.is_some() || term_sign != Sign::Pos {
                return false;
            }
            double_angle = Some(arg);
            continue;
        }

        if let Some(arg) = extract_scaled_plain_cosh_term_arg_root(ctx, term_expr) {
            if linear_cosh.is_some() || term_sign != Sign::Pos {
                return false;
            }
            linear_cosh = Some(arg);
            continue;
        }

        if let Some(arg) = extract_scaled_cosh_cubic_term_arg_root(ctx, term_expr) {
            if cubic_cosh.is_some() || term_sign != Sign::Neg {
                return false;
            }
            cubic_cosh = Some(arg);
            continue;
        }

        return false;
    }

    match (double_angle, linear_cosh, cubic_cosh) {
        (Some(double_arg), Some(linear_arg), Some(cubic_arg)) => {
            compare_expr(ctx, double_arg, linear_arg) == Ordering::Equal
                && compare_expr(ctx, double_arg, cubic_arg) == Ordering::Equal
        }
        _ => false,
    }
}

pub(super) fn extract_unary_builtin_arg_root(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(name, args) if ctx.is_builtin(*name, builtin) && args.len() == 1 => {
            Some(args[0])
        }
        _ => None,
    }
}

pub(super) fn matches_direct_small_zero_identity_root(ctx: &mut Context, expr: ExprId) -> bool {
    matches_direct_quotient_pair_zero_difference_root(ctx, expr)
        || matches_direct_half_angle_square_zero_identity_root(ctx, expr)
        || matches_direct_trig_binomial_square_zero_identity_root(ctx, expr)
        || matches_direct_positive_double_cos_square_diff_zero_identity_root(ctx, expr)
        || matches_direct_negative_double_cos_square_diff_zero_identity_root(ctx, expr)
        || matches_direct_tan_cot_product_zero_identity_root(ctx, expr)
        || matches_direct_tan_cot_sec_csc_zero_identity_root(ctx, expr)
        || matches_direct_sec_tan_pythagorean_zero_identity_root(ctx, expr)
        || matches_direct_csc_cot_pythagorean_zero_identity_root(ctx, expr)
        || matches_direct_squared_exact_one_zero_identity_root(ctx, expr)
        || matches_direct_trig_sine_double_angle_zero_identity_root(ctx, expr)
        || matches_direct_trig_product_to_sum_sin_sin_zero_identity_root(ctx, expr)
        || matches_direct_trig_product_to_sum_sin_cos_zero_identity_root(ctx, expr)
        || matches_direct_trig_product_to_sum_cos_cos_zero_identity_root(ctx, expr)
        || matches_direct_trig_mixed_double_angle_zero_identity_root(ctx, expr)
        || matches_direct_nested_fraction_simplified_zero_identity_root(ctx, expr)
        || matches_direct_trig_cubic_cosine_zero_identity_root(ctx, expr)
        || matches_direct_sum_diff_cubes_quotient_zero_identity_root(ctx, expr)
        || matches_direct_sqrt_perfect_square_abs_zero_identity_root(ctx, expr)
        || matches_direct_odd_half_power_zero_scope_root(ctx, expr)
        || matches_direct_odd_half_power_zero_identity_root(ctx, expr)
        || matches_direct_perfect_square_trinomial_zero_identity_root(ctx, expr)
        || matches_direct_log_product_contract_zero_identity_root(ctx, expr)
        || matches_direct_log_square_product_split_zero_identity_root(ctx, expr)
        || matches_direct_log_difference_squares_split_zero_identity_root(ctx, expr)
        || matches_direct_ln_abs_product_split_zero_identity_root(ctx, expr)
        || matches_direct_affine_common_denominator_zero_identity_root(ctx, expr)
        || matches_direct_small_polynomial_zero_identity_root(ctx, expr)
        || matches_direct_consecutive_telescoping_fraction_zero_identity_root(ctx, expr)
        || matches_direct_small_rational_zero_identity_root(ctx, expr)
        || matches_direct_symbolic_trig_sum_to_product_zero_identity_root(ctx, expr)
        || matches_direct_three_term_phase_shift_zero_subset_root(ctx, expr)
        || matches_direct_general_phase_shift_zero_identity_root(ctx, expr)
        || matches_direct_hyperbolic_exp_sum_zero_identity_root(ctx, expr)
        || matches_direct_recursive_hyperbolic_sinh_sum_zero_identity_root(ctx, expr)
        || matches_direct_recursive_hyperbolic_cosh_sum_zero_identity_root(ctx, expr)
        || matches_direct_hyperbolic_cosh_cubic_zero_identity_root(ctx, expr)
        || matches_direct_atanh_square_ratio_log_zero_identity_root(ctx, expr)
        || matches_direct_inverse_trig_composition_zero_identity_root(ctx, expr)
        || matches_direct_hyperbolic_pythagorean_zero_identity_root(ctx, expr)
        || matches_direct_mixed_pythagorean_zero_identity_root(ctx, expr)
        || matches_direct_exp_hyperbolic_double_identity_root(ctx, expr)
}

pub(super) fn matches_direct_small_zero_or_known_pair_base_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    if matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        let add_view = AddView::from_expr(ctx, expr);

        if expr_contains_factorial_call_local(ctx, expr)
            && crate::rules::arithmetic::try_build_small_direct_zero_core_rewrite(ctx, expr)
                .is_some()
        {
            return true;
        }

        if add_view.terms.len() == 3 {
            let has_trig = expr_contains_trig_builtin_local(ctx, expr);
            let has_hyperbolic = expr_contains_hyperbolic_builtin_local(ctx, expr);
            let has_log = expr_contains_log_builtin_local(ctx, expr);
            let has_division = expr_contains_division_node_local(ctx, expr);

            if has_trig
                && !has_hyperbolic
                && !has_log
                && matches_direct_tan_cot_sec_csc_zero_identity_root(ctx, expr)
            {
                return true;
            }

            if has_trig
                && !has_hyperbolic
                && !has_log
                && is_potential_direct_three_term_phase_shift_zero_subset_root(ctx, expr)
                && (matches_direct_numeric_general_phase_shift_zero_identity_root(ctx, expr)
                    || matches_direct_three_term_phase_shift_zero_subset_root(ctx, expr)
                    || matches_direct_general_phase_shift_zero_identity_root(ctx, expr))
            {
                return true;
            }

            if has_trig
                && !has_hyperbolic
                && !has_log
                && matches_direct_trig_cubic_cosine_zero_identity_root(ctx, expr)
            {
                return true;
            }

            if has_trig
                && !has_hyperbolic
                && !has_log
                && matches_direct_symbolic_trig_sum_to_product_zero_identity_root(ctx, expr)
            {
                return true;
            }

            if has_hyperbolic
                && !has_trig
                && !has_log
                && matches_direct_hyperbolic_cosh_cubic_zero_identity_root(ctx, expr)
            {
                return true;
            }

            if has_log
                && !has_trig
                && !has_hyperbolic
                && matches_direct_log_product_contract_zero_identity_root(ctx, expr)
            {
                return true;
            }

            if has_log
                && !has_trig
                && !has_hyperbolic
                && (matches_direct_log_square_product_split_zero_identity_root(ctx, expr)
                    || matches_direct_ln_abs_product_split_zero_identity_root(ctx, expr))
            {
                return true;
            }

            if !has_trig
                && !has_hyperbolic
                && !has_log
                && matches_geometric_difference_terms_root(ctx, &add_view.terms)
            {
                return true;
            }

            if !has_trig
                && !has_hyperbolic
                && !has_log
                && matches_direct_sophie_germain_zero_identity_root(ctx, expr)
            {
                return true;
            }

            if has_division
                && !has_trig
                && !has_hyperbolic
                && !has_log
                && matches_direct_depth_three_unit_continued_fraction_zero_identity_terms_root(
                    ctx,
                    &add_view.terms,
                )
            {
                return true;
            }
        }
    }

    match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) => {
            if matches_known_direct_pair_root(ctx, lhs, rhs)
                || matches_direct_half_angle_binomial_square_pair_root(ctx, lhs, rhs)
            {
                return true;
            }
        }
        Expr::Add(lhs, rhs) => {
            let Some((pos, neg)) = (match (ctx.get(lhs), ctx.get(rhs)) {
                (Expr::Neg(inner), _) => Some((rhs, *inner)),
                (_, Expr::Neg(inner)) => Some((lhs, *inner)),
                _ => None,
            }) else {
                return matches_direct_small_zero_identity_root(ctx, expr);
            };
            if matches_known_direct_pair_root(ctx, pos, neg)
                || matches_direct_half_angle_binomial_square_pair_root(ctx, pos, neg)
            {
                return true;
            }
        }
        _ => {}
    }

    matches_direct_small_zero_identity_root(ctx, expr)
}

pub(super) fn strip_multiplicative_one_root(ctx: &mut Context, expr: ExprId) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Mul(_, _) => {
            let factors = flatten_mul_chain(ctx, expr);
            let original_len = factors.len();
            let filtered: smallvec::SmallVec<[ExprId; 4]> = factors
                .into_iter()
                .filter(|factor| extract_i64_integer(ctx, *factor) != Some(1))
                .collect();
            if filtered.len() == 1 {
                filtered[0]
            } else if !filtered.is_empty() && filtered.len() < original_len {
                build_mul_expr_from_factors_root(ctx, &filtered)
            } else {
                expr
            }
        }
        Expr::Div(numerator, denominator) => {
            let normalized_numerator = strip_multiplicative_one_root(ctx, numerator);
            if compare_expr(ctx, normalized_numerator, numerator) == Ordering::Equal {
                expr
            } else {
                ctx.add(Expr::Div(normalized_numerator, denominator))
            }
        }
        _ => expr,
    }
}

pub(super) fn build_mul_expr_from_factors_root(ctx: &mut Context, factors: &[ExprId]) -> ExprId {
    match factors {
        [] => ctx.num(1),
        [single] => *single,
        _ => build_balanced_mul(ctx, factors),
    }
}

pub(super) fn try_standard_exact_zero_equivalence_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    // A term carrying a literal non-finite or undefined value never cancels with
    // itself: `inf - inf`, `(1/0) - (1/0)` and `undefined - undefined` are
    // indeterminate, not zero. Decline the whole exact-zero collapse so these
    // stay symbolic rather than folding to `0`.
    if crate::rules::arithmetic::additive_term_is_nonfinite_or_undefined(ctx, expr) {
        return None;
    }

    // Matrix multiplication is non-commutative, but these exact-zero routes treat
    // products as commutative factor multisets (e.g. `pairwise_matches` accepts
    // both factor pairings). That would collapse the commutator `A·B − B·A` to 0
    // even though it is generally nonzero. Decline the whole exact-zero collapse
    // whenever a matrix participates as a product factor so matrix differences
    // evaluate to their true value; genuine identities (`A·B − A·B`) still
    // collapse through the order-preserving matchers downstream.
    if crate::rules::arithmetic::term_has_matrix_product_factor(ctx, expr) {
        return None;
    }

    if matches_direct_log_square_product_split_zero_identity_root(ctx, expr)
        || matches_direct_ln_abs_product_split_zero_identity_root(ctx, expr)
    {
        let zero = ctx.num(0);
        record_profiled_orchestrator_route_hit(ctx, expr, "root.exact_zero.route.direct_log_split");
        return Some(run_named_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            zero,
            "Expandir logaritmos y cancelar términos iguales",
            "Expandir logaritmos y cancelar términos iguales",
            collect_steps,
        ));
    }

    profile_root_exact_zero_multiterm_trig_numeric_subset_status(
        options,
        ctx,
        expr,
        "root.exact_zero.entry.multiterm_trig_numeric_subset",
    );

    if is_guarded_small_zero_composition_candidate_root(ctx, expr) {
        let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);
        match ctx.get(expr) {
            Expr::Mul(_, _) => {
                let rule = crate::rules::arithmetic::CollapseExactZeroProductFactorRule;
                if let Some(rewrite) = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx) {
                    record_profiled_orchestrator_route_hit(
                        ctx,
                        expr,
                        "root.exact_zero.route.guarded_small_zero_product_rule",
                    );
                    return Some(finish_root_shortcut_with_rewrite_meta(
                        ctx,
                        expr,
                        rewrite,
                        "Collapse Zero Product via Exact Residual",
                        collect_steps,
                    ));
                }
            }
            Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => {
                if matches_direct_trig_cubic_cosine_pair_root(ctx, *lhs, *rhs) {
                    let zero = ctx.num(0);
                    record_profiled_orchestrator_route_hit(
                        ctx,
                        expr,
                        "root.exact_zero.route.guarded_small_zero_trig_cubic_pair",
                    );
                    return Some(run_named_rebuilt_root_shortcut_simplify(
                        options,
                        ctx,
                        expr,
                        zero,
                        "Collapse Exact Zero Additive Subexpression",
                        "Collapse Exact Zero Additive Subexpression",
                        collect_steps,
                    ));
                }
                let rule = crate::rules::arithmetic::CollapseExactZeroThreeTermSubsetRule;
                if let Some(rewrite) = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx) {
                    record_profiled_orchestrator_route_hit(
                        ctx,
                        expr,
                        "root.exact_zero.route.guarded_small_zero_three_term_subset_rule",
                    );
                    return Some(finish_root_shortcut_with_rewrite_meta(
                        ctx,
                        expr,
                        rewrite,
                        "Collapse Exact Zero Additive Subexpression",
                        collect_steps,
                    ));
                }
            }
            _ => {}
        }
    }

    let binary_zero_pair = match ctx.get(expr) {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) => Some((*lhs, *rhs)),
        _ => None,
    };
    if let Some((lhs, rhs)) = binary_zero_pair {
        if matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
            && matches_direct_trig_cubic_cosine_pair_root(ctx, lhs, rhs)
        {
            let zero = ctx.num(0);
            record_profiled_orchestrator_route_hit(
                ctx,
                expr,
                "root.exact_zero.route.binary_trig_cubic_pair",
            );
            return Some(run_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                zero,
                collect_steps,
            ));
        }

        if matches_direct_cos_square_diff_pair_root(ctx, lhs, rhs) {
            let zero = ctx.num(0);
            record_profiled_orchestrator_route_hit(
                ctx,
                expr,
                "root.exact_zero.route.binary_cos_square_diff_pair",
            );
            return Some(run_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                zero,
                collect_steps,
            ));
        }

        let lhs_is_direct_zero = matches_direct_small_zero_identity_root(ctx, lhs);
        let rhs_is_direct_zero = matches_direct_small_zero_identity_root(ctx, rhs);
        let lhs_is_small_trig_zero = is_small_trig_or_hyperbolic_zero_child(options, ctx, lhs);
        let rhs_is_small_trig_zero = is_small_trig_or_hyperbolic_zero_child(options, ctx, rhs);

        if lhs_is_direct_zero && rhs_is_direct_zero {
            let zero = ctx.num(0);
            let rewrite =
                crate::rule::Rewrite::with_local(zero, "Exact Zero Core Composition", expr, zero);
            record_profiled_orchestrator_route_hit(
                ctx,
                expr,
                "root.exact_zero.route.binary_direct_zero_pair",
            );
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Exact Zero Additive Subexpression",
                collect_steps,
            ));
        }

        if (lhs_is_direct_zero && rhs_is_small_trig_zero)
            || (rhs_is_direct_zero && lhs_is_small_trig_zero)
        {
            let zero = ctx.num(0);
            record_profiled_orchestrator_route_hit(
                ctx,
                expr,
                "root.exact_zero.route.binary_direct_plus_small_trig",
            );
            return Some(run_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                zero,
                collect_steps,
            ));
        }
    }

    let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);
    let common_scale_rule = crate::rules::arithmetic::CollapseExactZeroCommonScaledDifferenceRule;

    if let Some((_common_factor, residual_expr)) =
        extract_common_multiplicative_residual_sum_root(ctx, expr)
    {
        if matches_direct_small_zero_or_known_pair_residual_root(ctx, residual_expr) {
            record_profiled_orchestrator_route_hit(
                ctx,
                expr,
                "root.exact_zero.route.common_scale_known_residual",
            );
            return Some(finish_common_scale_zero_shortcut_with_domain_meta(
                ctx,
                expr,
                &parent_ctx,
                collect_steps,
            ));
        }
    }

    if is_same_denominator_difference_root(ctx, expr) {
        if let Some((_den, lhs_core, rhs_core)) =
            extract_same_denominator_direct_pair_root(ctx, expr)
        {
            if matches_known_direct_pair_root(ctx, lhs_core, rhs_core)
                || matches_direct_half_angle_binomial_square_pair_root(ctx, lhs_core, rhs_core)
            {
                let zero = ctx.num(0);
                record_profiled_orchestrator_route_hit(
                    ctx,
                    expr,
                    "root.exact_zero.route.same_denominator_direct_pair",
                );
                return Some(run_common_scale_rebuilt_root_shortcut_simplify(
                    options,
                    ctx,
                    expr,
                    zero,
                    collect_steps,
                ));
            }

            if let Some((lhs_residual, rhs_residual)) =
                extract_shared_additive_passthrough_pair_cores_root(ctx, lhs_core, rhs_core)
            {
                if matches_known_direct_pair_root(ctx, lhs_residual, rhs_residual)
                    || matches_direct_half_angle_binomial_square_pair_root(
                        ctx,
                        lhs_residual,
                        rhs_residual,
                    )
                {
                    let zero = ctx.num(0);
                    record_profiled_orchestrator_route_hit(
                        ctx,
                        expr,
                        "root.exact_zero.route.same_denominator_passthrough_pair",
                    );
                    return Some(run_common_scale_rebuilt_root_shortcut_simplify(
                        options,
                        ctx,
                        expr,
                        zero,
                        collect_steps,
                    ));
                }
            }
        }

        if let Some(rewrite) = crate::rule::Rule::apply(&common_scale_rule, ctx, expr, &parent_ctx)
        {
            let zero = ctx.num(0);
            if compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal {
                record_profiled_orchestrator_route_hit(
                    ctx,
                    expr,
                    "root.exact_zero.route.same_denominator_rule",
                );
                return Some(finish_root_shortcut_with_rewrite_meta(
                    ctx,
                    expr,
                    rewrite,
                    "Collapse Common-Scale Equivalent Difference",
                    collect_steps,
                ));
            }
        }
    }

    if matches_direct_two_factor_product_pair_zero_difference_root(ctx, expr)
        || matches_direct_quotient_pair_zero_difference_root(ctx, expr)
    {
        let zero = ctx.num(0);
        let rewrite =
            crate::rule::Rewrite::with_local(zero, "Equivalent Residual Cancellation", expr, zero);
        record_profiled_orchestrator_route_hit(
            ctx,
            expr,
            "root.exact_zero.route.two_factor_or_quotient_pair",
        );
        return Some(finish_root_shortcut_with_rewrite_meta(
            ctx,
            expr,
            rewrite,
            "Collapse Common-Scale Equivalent Difference",
            collect_steps,
        ));
    }

    if let Some((result, shortcut_steps)) =
        try_standard_subtract_expanded_sum_diff_cubes_quotient_shortcut(
            options,
            ctx,
            expr,
            collect_steps,
        )
    {
        record_profiled_orchestrator_route_hit(
            ctx,
            expr,
            "root.exact_zero.route.sum_diff_cubes_quotient",
        );
        return Some((result, shortcut_steps));
    }

    if let Some((lhs_core, rhs_core)) =
        extract_shared_additive_passthrough_sub_cores_root(ctx, expr)
    {
        if matches_direct_small_pow_expansion_pair_root(ctx, lhs_core, rhs_core) {
            let zero = ctx.num(0);
            record_profiled_orchestrator_route_hit(
                ctx,
                expr,
                "root.exact_zero.route.shared_passthrough_small_pow",
            );
            return Some(run_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                zero,
                collect_steps,
            ));
        }
    }

    let direct_rule = crate::rules::arithmetic::CollapseExactZeroThreeTermSubsetRule;
    if let Some(rewrite) = crate::rule::Rule::apply(&direct_rule, ctx, expr, &parent_ctx) {
        let zero = ctx.num(0);
        if compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal {
            record_profiled_orchestrator_route_hit(
                ctx,
                expr,
                "root.exact_zero.route.three_term_subset_rule",
            );
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Exact Zero Additive Subexpression",
                collect_steps,
            ));
        }
    }

    if let Some(rewrite) = crate::rule::Rule::apply(&common_scale_rule, ctx, expr, &parent_ctx) {
        let zero = ctx.num(0);
        if compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal {
            record_profiled_orchestrator_route_hit(
                ctx,
                expr,
                "root.exact_zero.route.common_scale_rule",
            );
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Common-Scale Equivalent Difference",
                collect_steps,
            ));
        }
    }

    if let Some(result) =
        try_standard_common_scale_exact_zero_shortcut_fallback(options, ctx, expr, collect_steps)
    {
        record_profiled_orchestrator_route_hit(
            ctx,
            expr,
            "root.exact_zero.route.common_scale_fallback",
        );
        return Some(result);
    }

    None
}

pub(super) fn run_rebuilt_root_shortcut_simplify(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    before: ExprId,
    rewritten: ExprId,
    collect_steps: bool,
) -> (ExprId, Vec<Step>) {
    run_named_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        before,
        rewritten,
        "Collapse Exact Zero Additive Subexpression",
        "Collapse Exact Zero Additive Subexpression",
        collect_steps,
    )
}

pub(super) fn run_named_rebuilt_root_shortcut_simplify(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    before: ExprId,
    rewritten: ExprId,
    local_desc: &'static str,
    rule_name: &'static str,
    collect_steps: bool,
) -> (ExprId, Vec<Step>) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let (result, inner_steps, _stats) = simplifier.simplify_with_stats(
        rewritten,
        crate::SimplifyOptions {
            suppress_depth_overflow_warnings: true,
            ..options.clone()
        },
    );
    std::mem::swap(&mut simplifier.context, ctx);
    let (result, mut closure_steps) = if let Some((closed, steps)) =
        try_finalize_trivial_additive_closure_root(options, ctx, result, collect_steps)
    {
        (closed, steps)
    } else {
        (result, Vec::new())
    };

    let mut shortcut_steps = Vec::new();
    if collect_steps {
        let mut step = Step::new_compact(local_desc, rule_name, before, rewritten);
        step.global_before = Some(before);
        step.global_after = Some(rewritten);
        step.importance = crate::step::ImportanceLevel::High;
        shortcut_steps.push(step);
        shortcut_steps.extend(inner_steps);
        shortcut_steps.append(&mut closure_steps);
    }

    (result, shortcut_steps)
}

pub(super) fn isolated_simplify_expr_if_changed(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let _nesting_guard = enter_isolated_simplify_probe()?;
    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let (rewritten, _steps, _stats) = simplifier.simplify_with_stats(
        expr,
        crate::SimplifyOptions {
            collect_steps: false,
            suppress_depth_overflow_warnings: true,
            ..options.clone()
        },
    );
    std::mem::swap(&mut simplifier.context, ctx);
    (compare_expr(ctx, rewritten, expr) != Ordering::Equal).then_some(rewritten)
}

pub(super) fn isolated_simplify_rewrites_to_target(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    target: ExprId,
) -> bool {
    let memo_key = (
        ctx.instance_tag(),
        isolated_probe_options_fingerprint(options),
        expr,
        target,
    );
    if let Some(cached) =
        ISOLATED_SIMPLIFY_PROBE_MEMO.with(|memo| memo.borrow().get(&memo_key).copied())
    {
        return cached;
    }

    let Some(_nesting_guard) = enter_isolated_simplify_probe() else {
        return false;
    };
    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let mut orchestrator = Orchestrator::new();
    orchestrator.options = SimplifyOptions {
        collect_steps: false,
        suppress_depth_overflow_warnings: true,
        ..options.clone()
    };
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    std::mem::swap(&mut simplifier.context, ctx);
    let reaches_target = compare_expr(ctx, rewritten, target) == Ordering::Equal;

    ISOLATED_SIMPLIFY_PROBE_MEMO.with(|memo| {
        let mut memo = memo.borrow_mut();
        if memo.len() > 8192 {
            memo.clear();
        }
        memo.insert(memo_key, reaches_target);
    });
    reaches_target
}

pub(super) fn isolated_simplify_rewrites_to_zero(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let zero = ctx.num(0);
    isolated_simplify_rewrites_to_target(options, ctx, expr, zero)
}

pub(super) fn expr_contains_hyperbolic_builtin_local(ctx: &Context, expr: ExprId) -> bool {
    expr_contains_any_builtin_local(
        ctx,
        expr,
        &[BuiltinFn::Sinh, BuiltinFn::Cosh, BuiltinFn::Tanh],
    )
}

pub(super) fn expr_contains_trig_builtin_local(ctx: &Context, expr: ExprId) -> bool {
    expr_contains_any_builtin_local(
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
}

pub(super) fn expr_contains_trig_or_hyperbolic_builtin_local(ctx: &Context, expr: ExprId) -> bool {
    expr_contains_trig_builtin_local(ctx, expr) || expr_contains_hyperbolic_builtin_local(ctx, expr)
}

pub(super) fn expr_contains_log_builtin_local(ctx: &Context, expr: ExprId) -> bool {
    expr_contains_any_builtin_local(
        ctx,
        expr,
        &[
            BuiltinFn::Ln,
            BuiltinFn::Log,
            BuiltinFn::Log10,
            BuiltinFn::Abs,
        ],
    )
}

pub(super) fn is_supported_nested_zero_child_partner(ctx: &Context, expr: ExprId) -> bool {
    expr_contains_log_builtin_local(ctx, expr)
        || is_supported_nonlog_additive_nested_zero_child_partner(ctx, expr)
}

pub(super) fn supported_nested_zero_partner_rewrites_to_zero(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    if !is_supported_nested_zero_child_partner(ctx, expr) {
        return false;
    }

    let family = supported_nested_zero_child_partner_profile_family(ctx, expr);
    let skip_direct_small_zero =
        family == "nonlog_additive" && is_plain_division_difference_root(ctx, expr);

    let direct_small_zero = if skip_direct_small_zero {
        false
    } else if let Some(label) =
        supported_nested_zero_partner_try_profile_label(family, "direct_small_zero")
    {
        if crate::orchestrator_shortcut_profiler::should_profile_orchestrator_shortcut(label) {
            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                label,
                render_expr_for_orchestrator_profile(ctx, expr),
            );
        }
        run_profiled_orchestrator_bool_section(label, || {
            try_standard_direct_small_zero_identity_shortcut(options, ctx, expr, false).is_some()
        })
    } else {
        try_standard_direct_small_zero_identity_shortcut(options, ctx, expr, false).is_some()
    };
    if direct_small_zero {
        return true;
    }

    let symbolic_root_denesting = if expr_contains_sqrt_or_half_power_local(ctx, expr) {
        if let Some(label) =
            supported_nested_zero_partner_try_profile_label(family, "symbolic_root_denesting")
        {
            if crate::orchestrator_shortcut_profiler::should_profile_orchestrator_shortcut(label) {
                crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                    label,
                    render_expr_for_orchestrator_profile(ctx, expr),
                );
            }
            run_profiled_orchestrator_bool_section(label, || {
                try_standard_symbolic_root_denesting_subset_zero_shortcut(options, ctx, expr, false)
                    .is_some()
            })
        } else {
            try_standard_symbolic_root_denesting_subset_zero_shortcut(options, ctx, expr, false)
                .is_some()
        }
    } else {
        false
    };
    if symbolic_root_denesting {
        return true;
    }

    let atanh_square_ratio_log = if let Some(label) =
        supported_nested_zero_partner_try_profile_label(family, "atanh_square_ratio_log")
    {
        if crate::orchestrator_shortcut_profiler::should_profile_orchestrator_shortcut(label) {
            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                label,
                render_expr_for_orchestrator_profile(ctx, expr),
            );
        }
        run_profiled_orchestrator_bool_section(label, || {
            try_standard_atanh_square_ratio_log_subset_zero_shortcut(options, ctx, expr, false)
                .is_some()
        })
    } else {
        try_standard_atanh_square_ratio_log_subset_zero_shortcut(options, ctx, expr, false)
            .is_some()
    };
    if atanh_square_ratio_log {
        return true;
    }

    let exact_zero_equivalence = if let Some(label) =
        supported_nested_zero_partner_try_profile_label(family, "exact_zero_equivalence")
    {
        if crate::orchestrator_shortcut_profiler::should_profile_orchestrator_shortcut(label) {
            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                label,
                render_expr_for_orchestrator_profile(ctx, expr),
            );
        }
        run_profiled_orchestrator_bool_section(label, || {
            try_standard_exact_zero_equivalence_shortcut(options, ctx, expr, false).is_some()
        })
    } else {
        try_standard_exact_zero_equivalence_shortcut(options, ctx, expr, false).is_some()
    };
    if exact_zero_equivalence {
        return true;
    }

    if !should_try_supported_nested_zero_partner_isolated_simplify(ctx, expr) {
        return false;
    }

    if let Some(label) =
        supported_nested_zero_partner_try_profile_label(family, "isolated_simplify")
    {
        if crate::orchestrator_shortcut_profiler::should_profile_orchestrator_shortcut(label) {
            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                label,
                render_expr_for_orchestrator_profile(ctx, expr),
            );
        }
        run_profiled_orchestrator_bool_section(label, || {
            isolated_simplify_rewrites_to_zero(options, ctx, expr)
        })
    } else {
        isolated_simplify_rewrites_to_zero(options, ctx, expr)
    }
}

pub(super) fn canonicalize_direct_pair_factor_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    if let Some(rewritten) = extract_special_angle_exact_value_root(ctx, expr) {
        return Some(rewritten);
    }
    if cas_ast::collect_variables(ctx, expr).is_empty() && cas_ast::count_nodes(ctx, expr) <= 16 {
        if let Some(rewritten) =
            isolated_simplify_expr_if_changed(&isolated_probe_options(), ctx, expr)
        {
            return Some(strip_multiplicative_one_root(ctx, rewritten));
        }
    }
    if let Some(rewrite) = try_rewrite_trig_phase_shift_function_expr(ctx, expr) {
        let normalized = strip_multiplicative_one_root(ctx, rewrite.rewritten);
        return Some(canonicalize_even_cos_in_simple_expr_root(ctx, normalized));
    }
    if let Some(plan) =
        cas_math::inverse_trig_composition_support::try_plan_inverse_trig_composition_expr(
            ctx, expr, false, false,
        )
    {
        return Some(strip_multiplicative_one_root(ctx, plan.rewritten));
    }
    if let Some(rewrite) = try_rewrite_trig_inverse_composition_expr(ctx, expr) {
        return Some(strip_multiplicative_one_root(ctx, rewrite.rewritten));
    }
    if let Some(rewrite) = try_rewrite_exponential_log_inverse_expr(ctx, expr) {
        return Some(strip_multiplicative_one_root(ctx, rewrite.rewritten));
    }
    if let Some((trig_fn, full_arg)) = extract_direct_abs_trig_half_angle_target_root(ctx, expr) {
        return Some(build_direct_sqrt_abs_trig_half_angle_target_root(
            ctx, trig_fn, full_arg,
        ));
    }
    if let Some(log_inverse_match) =
        cas_math::logarithm_inverse_support::try_match_log_exp_inverse_expr(ctx, expr)
    {
        match log_inverse_match {
            cas_math::logarithm_inverse_support::LogExpInverseMatch::Numeric {
                rewritten, ..
            } => {
                return Some(strip_multiplicative_one_root(ctx, rewritten));
            }
            cas_math::logarithm_inverse_support::LogExpInverseMatch::Symbolic {
                base,
                exponent,
            } => {
                let e = ctx.add(Expr::Constant(Constant::E));
                if compare_expr(ctx, base, e) == Ordering::Equal {
                    return Some(strip_multiplicative_one_root(ctx, exponent));
                }
            }
        }
    }
    if let Some(base) = extract_addition_of_successive_unit_fractions_arg_root(ctx, expr) {
        return Some(build_collapsed_successive_unit_fractions_expr_root(
            ctx, base,
        ));
    }
    if let Some(base) = extract_consecutive_telescoping_fraction_difference_arg_root(ctx, expr) {
        return Some(build_consecutive_telescoping_fraction_difference_expr_root(
            ctx, base,
        ));
    }
    if let Some(base) = extract_collapsed_successive_unit_fractions_arg_root(ctx, expr) {
        return Some(build_collapsed_successive_unit_fractions_expr_root(
            ctx, base,
        ));
    }
    if let Some(denominator) = extract_unit_fraction_denominator_root(ctx, expr) {
        if let Some(base) = extract_consecutive_product_core_root(ctx, denominator) {
            return Some(build_consecutive_telescoping_fraction_difference_expr_root(
                ctx, base,
            ));
        }
    }
    if let Some(factored) = cas_math::factor::factor_perfect_square_trinomial(ctx, expr) {
        return Some(factored);
    }
    // REAL-ONLY (mirrors SimplifySquareRootRule): the helper emits |·| forms of
    // symbolic squares (`√(x²) → |x|`), false over ℂ (`√(i²) = i ≠ |i|`). The
    // canonicalizer's signature predates the value-domain axis; the ambient
    // pipeline domain carries it here (audit 2026-07-30, corrección (c) del
    // refutador de S4-002 — último consumidor publish-capaz sin guarda).
    if crate::rules::arithmetic::ambient_pipeline_value_domain()
        == crate::semantics::ValueDomain::RealOnly
    {
        if let Some(rewrite) = try_rewrite_simplify_square_root_expr(ctx, expr) {
            return Some(strip_multiplicative_one_root(ctx, rewrite.rewritten));
        }
    }
    if let Some(canonical) = try_rewrite_canonical_root_expr(ctx, expr) {
        if let Some(extract) =
            try_rewrite_extract_perfect_power_from_radicand_expr(ctx, canonical.rewritten)
        {
            return Some(strip_multiplicative_one_root(ctx, extract.rewritten));
        }
        return Some(strip_multiplicative_one_root(ctx, canonical.rewritten));
    }
    if let Some(numerator) = extract_div_by_two_numerator_root(ctx, expr) {
        if let Some(rewrite) = try_rewrite_sum_to_product_contraction_expr(ctx, numerator) {
            if rewrite.kind
                == cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinSum
            {
                if let Some((sin_arg, cos_arg)) =
                    extract_scaled_trig_sin_cos_product_args_root(ctx, rewrite.rewritten)
                {
                    let canonical_cos_arg = canonicalize_even_cos_arg_root(ctx, cos_arg);
                    return Some(build_plain_trig_sin_cos_product_root(
                        ctx,
                        sin_arg,
                        canonical_cos_arg,
                    ));
                }
            }
        }
    }
    if let Some(rewrite) = try_rewrite_sum_to_product_contraction_expr(ctx, expr) {
        match rewrite.kind {
            cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinSum
            | cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinDiff => {
                if let Some((sin_arg, cos_arg)) =
                    extract_scaled_trig_sin_cos_product_args_root(ctx, rewrite.rewritten)
                {
                    let canonical_cos_arg = canonicalize_even_cos_arg_root(ctx, cos_arg);
                    return Some(build_scaled_trig_sin_cos_product_root(
                        ctx,
                        sin_arg,
                        canonical_cos_arg,
                    ));
                }
            }
            _ => {}
        }
    }
    if let Some((sin_arg, cos_arg)) = extract_scaled_trig_sin_cos_product_args_root(ctx, expr) {
        let canonical_cos_arg = canonicalize_even_cos_arg_root(ctx, cos_arg);
        return Some(build_scaled_trig_sin_cos_product_root(
            ctx,
            sin_arg,
            canonical_cos_arg,
        ));
    }
    if let Some(rewritten) = rewrite_direct_trig_product_to_sum_double_angle_target_root(ctx, expr)
    {
        return Some(rewritten);
    }
    if let Some(rewrite) = try_rewrite_product_to_sum_expr(ctx, expr) {
        return Some(rewrite.rewritten);
    }
    if let Some(rewrite) = try_rewrite_angle_sum_fraction_to_tan_expr(ctx, expr) {
        return Some(rewrite.rewritten);
    }
    if let Some((lhs_arg, rhs_arg)) = extract_direct_tan_angle_sum_target_root(ctx, expr) {
        return Some(build_tan_angle_sum_fraction_root(ctx, lhs_arg, rhs_arg));
    }
    if let Some(rewrite) = try_rewrite_tan_to_sin_cos_function_expr(ctx, expr) {
        return Some(rewrite.rewritten);
    }
    if let Some(rewrite) = try_rewrite_double_angle_function_expr(ctx, expr) {
        return Some(rewrite.rewritten);
    }
    if let Some((lhs_arg, rhs_arg)) = extract_direct_tangent_addition_target_root(ctx, expr) {
        return Some(build_tangent_addition_fraction_root(ctx, lhs_arg, rhs_arg));
    }
    if let Some(rewrite) = try_rewrite_triple_angle_expr(ctx, expr) {
        return Some(rewrite.rewritten);
    }
    if let Some(rewrite) = try_rewrite_hyperbolic_triple_angle(ctx, expr) {
        return Some(rewrite.rewritten);
    }
    if let Some(rewrite) = try_rewrite_hyperbolic_double_angle_sum(ctx, expr) {
        return Some(rewrite.rewritten);
    }
    if let Some((arg, is_sum)) = extract_direct_hyperbolic_exp_sum_target_root(ctx, expr) {
        return Some(build_direct_hyperbolic_exp_sum_target_root(
            ctx, arg, is_sum,
        ));
    }
    if let Some(rewrite) = try_rewrite_recognize_hyperbolic_from_exp(ctx, expr) {
        return Some(rewrite.rewritten);
    }
    if let Some(arg) = extract_direct_tanh_pythagorean_identity_arg_root(ctx, expr) {
        return Some(build_tanh_pythagorean_target_root(ctx, arg));
    }
    if let Some(rewritten) = try_rewrite_tanh_double_angle_expansion(ctx, expr) {
        return Some(rewritten);
    }
    if let Some(rewritten) = try_rewrite_tanh_to_sinh_cosh(ctx, expr) {
        return Some(rewritten);
    }
    if let Some(rewrite) =
        cas_math::root_den_rationalize_support::try_rewrite_rationalize_cube_root_den_expr(
            ctx, expr,
        )
    {
        return Some(strip_multiplicative_one_root(ctx, rewrite.rewritten));
    }
    if let Some((hyperbolic_fn, arg)) =
        extract_direct_hyperbolic_half_angle_square_target_root(ctx, expr)
    {
        return Some(build_plain_hyperbolic_half_angle_pow2_root(
            ctx,
            hyperbolic_fn,
            arg,
        ));
    }
    if let Some(arg) = extract_scaled_double_angle_sin_square_target_root(ctx, expr) {
        return Some(build_plain_sin_cos_square_product_root(ctx, arg));
    }
    if let Some(arg) = extract_direct_positive_double_cos_square_diff_target_root(ctx, expr) {
        return Some(build_positive_cos_double_angle_expr_root(ctx, arg));
    }
    if let Some(arg) = extract_direct_cos_fourth_power_reduction_target_root(ctx, expr) {
        return Some(build_plain_trig_pow4_root(ctx, BuiltinFn::Cos, arg));
    }
    if let Some(rewritten) = rewrite_sum_of_squares_product_root(ctx, expr) {
        return Some(rewritten);
    }
    None
}

pub(super) fn factor_known_small_polynomial_partner_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    if let Some(factored) = factor_short_geometric_sum_partner_root(ctx, expr) {
        if compare_expr(ctx, factored, expr) != Ordering::Equal {
            return Some(factored);
        }
    }

    if let Some(factored) =
        cas_math::factor::factor_perfect_square_trinomial(ctx, expr).or_else(|| {
            let view = AddView::from_expr(ctx, expr);
            build_direct_perfect_square_from_terms_root(ctx, &view.terms)
        })
    {
        if compare_expr(ctx, factored, expr) != Ordering::Equal {
            return Some(factored);
        }
    }

    let rewrite = try_rewrite_automatic_factor_expr(ctx, expr)?;
    let factored = strip_multiplicative_one_root(ctx, rewrite.rewritten);
    if compare_expr(ctx, factored, expr) == Ordering::Equal {
        return None;
    }

    if extract_direct_two_linear_shift_product_root(ctx, factored).is_some()
        || extract_direct_three_linear_shift_product_root(ctx, factored).is_some()
        || matches_direct_small_pow_expansion_pair_root(ctx, expr, factored)
    {
        return Some(factored);
    }

    None
}

pub(super) fn is_function_free_arithmetic_expr_root(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) => true,
        Expr::Neg(inner) => is_function_free_arithmetic_expr_root(ctx, *inner),
        Expr::Add(lhs, rhs)
        | Expr::Sub(lhs, rhs)
        | Expr::Mul(lhs, rhs)
        | Expr::Div(lhs, rhs)
        | Expr::Pow(lhs, rhs) => {
            is_function_free_arithmetic_expr_root(ctx, *lhs)
                && is_function_free_arithmetic_expr_root(ctx, *rhs)
        }
        Expr::Function(_, _) | Expr::Matrix { .. } | Expr::SessionRef(_) | Expr::Hold(_) => false,
    }
}

pub(super) fn extract_shared_additive_passthrough_sub_cores_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let (lhs, rhs) = match ctx.get(expr) {
        Expr::Sub(lhs, rhs) => (*lhs, *rhs),
        Expr::Add(lhs, rhs) => match ctx.get(*rhs) {
            Expr::Neg(inner) => (*lhs, *inner),
            _ => return None,
        },
        _ => return None,
    };
    extract_shared_additive_passthrough_pair_cores_root(ctx, lhs, rhs)
}

pub(super) fn try_build_chunk_pair_zero_shortcut_steps_root(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    first_chunk: ExprId,
    second_chunk: ExprId,
) -> Option<Vec<Step>> {
    let zero = ctx.num(0);
    let first_steps = build_recursive_or_leaf_zero_chunk_steps_root(options, ctx, first_chunk)?;
    let second_steps = build_recursive_or_leaf_zero_chunk_steps_root(options, ctx, second_chunk)?;

    let mut stitched_steps = Vec::new();
    let mut current_global = expr;
    for step in first_steps {
        let internal_after = step.global_after.unwrap_or(zero);
        let parent_after =
            merge_additive_zero_chunk_residual_root(ctx, internal_after, second_chunk);
        let mut stitched = step.clone();
        stitched.global_before = Some(current_global);
        stitched.global_after = Some(parent_after);
        stitched_steps.push(stitched);
        current_global = parent_after;
    }

    for step in second_steps {
        let internal_after = step.global_after.unwrap_or(zero);
        let mut stitched = step.clone();
        stitched.global_before = Some(current_global);
        stitched.global_after = Some(internal_after);
        stitched_steps.push(stitched);
        current_global = internal_after;
    }

    Some(stitched_steps)
}

pub(super) fn extract_common_multiplicative_residual_sum_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let factor_lists: Vec<Vec<_>> = view
        .terms
        .iter()
        .map(|(term_expr, _)| flatten_mul_chain(ctx, *term_expr))
        .collect();
    if factor_lists.iter().any(Vec::is_empty) {
        return None;
    }

    let mut used_by_term = factor_lists
        .iter()
        .map(|factors| vec![false; factors.len()])
        .collect::<Vec<_>>();
    let mut common = Vec::new();

    for (first_index, first_factor) in factor_lists[0].iter().copied().enumerate() {
        let Some(second_index) =
            factor_lists[1]
                .iter()
                .enumerate()
                .find_map(|(candidate_index, factor)| {
                    (!used_by_term[1][candidate_index]
                        && compare_expr(ctx, *factor, first_factor) == Ordering::Equal)
                        .then_some(candidate_index)
                })
        else {
            continue;
        };

        common.push(first_factor);
        used_by_term[0][first_index] = true;
        used_by_term[1][second_index] = true;
    }

    if common.is_empty() {
        return None;
    }

    let residual_terms: Vec<_> = view
        .terms
        .iter()
        .copied()
        .enumerate()
        .map(|(term_index, (_term_expr, term_sign))| {
            let residual_factors = factor_lists[term_index]
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(factor_index, factor)| {
                    (!used_by_term[term_index][factor_index]).then_some(factor)
                })
                .collect::<Vec<_>>();
            (
                build_mul_expr_from_factors_root(ctx, &residual_factors),
                term_sign,
            )
        })
        .collect();

    let common_factor = build_mul_expr_from_factors_root(ctx, &common);
    let residual_expr = build_signed_sum_expr_root(ctx, &residual_terms);
    let one = ctx.num(1);
    if compare_expr(ctx, common_factor, one) == Ordering::Equal
        || compare_expr(ctx, residual_expr, expr) == Ordering::Equal
    {
        return None;
    }
    Some((common_factor, residual_expr))
}

pub(super) fn extract_direct_two_linear_shift_product_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<BigRational>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let (lhs_base, lhs_constant) = extract_base_plus_constant_root(ctx, factors[0])?;
    let (rhs_base, rhs_constant) = extract_base_plus_constant_root(ctx, factors[1])?;
    if compare_expr(ctx, lhs_base, rhs_base) != Ordering::Equal {
        return None;
    }

    let mut constants = vec![lhs_constant, rhs_constant];
    constants.sort();
    Some((lhs_base, constants))
}

pub(super) fn build_root_shortcut_step_from_rewrite(
    ctx: &Context,
    before: ExprId,
    rewrite: &crate::rule::Rewrite,
    rule_name: &'static str,
) -> Step {
    let mut step = Step::with_snapshots(
        &rewrite.description,
        rule_name,
        before,
        rewrite.new_expr,
        smallvec::SmallVec::<[crate::step::PathStep; 8]>::new(),
        Some(ctx),
        before,
        rewrite.new_expr,
    );
    step.importance = crate::step::ImportanceLevel::High;
    {
        let meta = step.meta_mut();
        meta.before_local = rewrite.before_local;
        meta.after_local = rewrite.after_local;
        meta.assumption_events = rewrite.assumption_events.clone();
        meta.required_conditions = rewrite.required_conditions.clone();
        meta.poly_proof = rewrite.poly_proof.clone();
        meta.substeps = rewrite.substeps.clone();
    }
    step
}
