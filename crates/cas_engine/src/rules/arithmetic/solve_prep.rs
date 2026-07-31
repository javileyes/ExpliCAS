//! `arithmetic`: familia `solve_prep`.
//!
//! Ver la cabecera de `arithmetic.rs` para el contexto.

use super::*;

pub(super) fn classify_solve_prep_coeff_shape(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> SolvePrepCoeffShape {
    let contains_div = contains_division_like_term(ctx, expr);
    match ctx.get(expr) {
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) => SolvePrepCoeffShape::Atom,
        Expr::Neg(inner) => match ctx.get(*inner) {
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) => SolvePrepCoeffShape::NegAtom,
            _ if contains_div => SolvePrepCoeffShape::NegWithDiv,
            _ => SolvePrepCoeffShape::NegOther,
        },
        Expr::Add(_, _) | Expr::Sub(_, _) if contains_div => SolvePrepCoeffShape::AddSubWithDiv,
        Expr::Add(_, _) | Expr::Sub(_, _) => SolvePrepCoeffShape::AddSubNoDiv,
        Expr::Div(_, _) => SolvePrepCoeffShape::Div,
        Expr::Mul(_, _) if contains_div => SolvePrepCoeffShape::MulWithDiv,
        Expr::Mul(_, _) => SolvePrepCoeffShape::MulNoDiv,
        Expr::Pow(_, _) => SolvePrepCoeffShape::Pow,
        Expr::Function(_, _) => SolvePrepCoeffShape::Function,
        Expr::Hold(_) => SolvePrepCoeffShape::Hold,
        Expr::Matrix { .. } => SolvePrepCoeffShape::Matrix,
        Expr::SessionRef(_) => SolvePrepCoeffShape::SessionRef,
    }
}

pub(super) fn classify_solve_prep_simplify_c_shape_label(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> &'static str {
    classify_solve_prep_coeff_shape(ctx, expr).simplify_c_profile_label()
}

pub(super) fn try_build_fast_solve_prep_exact_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();

    let term_count = view.terms.len();
    for subset_len in [3usize, 2, 1] {
        if subset_len >= term_count {
            break;
        }

        for mask in 1usize..(1usize << term_count) {
            if mask.count_ones() as usize != subset_len {
                continue;
            }

            let focus_terms: Vec<_> = view
                .terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| (((mask >> index) & 1) == 1).then_some(term))
                .collect();
            let focus_expr = build_signed_sum_expr(ctx, &focus_terms);

            let remaining_terms: Vec<_> = view
                .terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| (((mask >> index) & 1) == 0).then_some(term))
                .collect();
            if remaining_terms.is_empty() {
                continue;
            }

            let remaining_expr = build_signed_sum_expr(ctx, &remaining_terms);
            let pair_sample = profiling.then(|| {
                format!(
                    "{}  ||  {}",
                    render_expr_for_orchestrator_profile(ctx, focus_expr),
                    render_expr_for_orchestrator_profile(ctx, remaining_expr)
                )
            });
            let remaining_has_shifted_square = expr_contains_shifted_square(ctx, remaining_expr);
            if profiling {
                let label = if remaining_has_shifted_square {
                    "rule.fast_solve_prep.gate.remaining_shifted_square"
                } else {
                    "rule.fast_solve_prep.gate.remaining_no_shifted_square"
                };
                let _ =
                    run_profiled_orchestrator_option_section(label, pair_sample.clone(), || {
                        Some(())
                    });
            }
            if !remaining_has_shifted_square {
                continue;
            }
            if profiling {
                let overlap = has_plausible_solve_prep_focus_remaining_variable_overlap(
                    ctx,
                    focus_expr,
                    remaining_expr,
                );
                let label = if overlap {
                    "rule.fast_solve_prep.gate.focus_remaining_var_overlap"
                } else {
                    "rule.fast_solve_prep.gate.focus_remaining_var_mismatch"
                };
                let _ =
                    run_profiled_orchestrator_option_section(label, pair_sample.clone(), || {
                        Some(())
                    });
            }
            let neg_remaining = negate_additive_scope_expr(ctx, remaining_expr);
            let mut canonical_neg_remaining = None;

            let rewrite_matches = match run_profiled_fast_solve_prep_probe(
                profiling,
                "rule.fast_solve_prep.try.collect_rewrites",
                &pair_sample,
                || {
                    Some(
                        collect_exact_solve_prep_equivalence_rewrites_for_cancellation(
                            ctx, focus_expr,
                        ),
                    )
                },
            ) {
                Some(matches) => matches,
                None => continue,
            };

            for rewrite_match in rewrite_matches {
                if exprs_match_for_cancellation(ctx, rewrite_match.rewritten, neg_remaining) {
                    if profiling {
                        let _ = run_profiled_orchestrator_option_section(
                            "rule.fast_solve_prep.route.direct_neg_match",
                            pair_sample.clone(),
                            || Some(()),
                        );
                    }
                    return Some(
                        Rewrite::with_local(
                            ctx.num(0),
                            "Complete the Square",
                            rewrite_match.local_before,
                            rewrite_match.local_after,
                        )
                        .requires(crate::ImplicitCondition::NonZero(
                            rewrite_match.nonzero_expr,
                        )),
                    );
                }

                let canonical_neg_remaining = *canonical_neg_remaining
                    .get_or_insert_with(|| normalize_additive_scope_expr(ctx, neg_remaining));
                let canonical_rewritten =
                    normalize_additive_scope_expr(ctx, rewrite_match.rewritten);
                let candidate_total = ctx.add(Expr::Add(rewrite_match.rewritten, remaining_expr));
                let canonical_match =
                    compare_expr(ctx, canonical_rewritten, canonical_neg_remaining)
                        == Ordering::Equal;
                let default_simplify_match = !canonical_match
                    && exprs_match_after_default_simplify(
                        ctx,
                        rewrite_match.rewritten,
                        neg_remaining,
                    );
                let total_zero_match = !canonical_match
                    && !default_simplify_match
                    && is_zero_after_default_simplify(ctx, candidate_total);
                if !(canonical_match || default_simplify_match || total_zero_match) {
                    continue;
                }
                if profiling {
                    let route_label = if canonical_match {
                        "rule.fast_solve_prep.route.canonical_neg_match"
                    } else if default_simplify_match {
                        "rule.fast_solve_prep.route.default_simplify_match"
                    } else {
                        "rule.fast_solve_prep.route.candidate_total_zero"
                    };
                    let _ = run_profiled_orchestrator_option_section(
                        route_label,
                        pair_sample.clone(),
                        || Some(()),
                    );
                    if default_simplify_match {
                        let _ = run_profiled_orchestrator_option_section(
                            rewrite_match.build_route.default_simplify_profile_label(),
                            pair_sample.clone(),
                            || Some(()),
                        );
                    } else if total_zero_match {
                        let _ = run_profiled_orchestrator_option_section(
                            rewrite_match
                                .build_route
                                .candidate_total_zero_profile_label(),
                            pair_sample.clone(),
                            || Some(()),
                        );
                    }
                }

                return Some(
                    Rewrite::with_local(
                        ctx.num(0),
                        "Complete the Square",
                        rewrite_match.local_before,
                        rewrite_match.local_after,
                    )
                    .requires(crate::ImplicitCondition::NonZero(
                        rewrite_match.nonzero_expr,
                    )),
                );
            }
        }
    }

    None
}

pub(super) fn classify_solve_prep_build_route(
    ctx: &mut cas_ast::Context,
    a: cas_ast::ExprId,
    b: cas_ast::ExprId,
) -> SolvePrepBuildRoute {
    if let Some(positive_a) = strip_unit_negation_for_phase_shift(ctx, a) {
        if is_direct_complete_square_symbolic_scale_expr(ctx, positive_a)
            && strip_unit_negation_for_phase_shift(ctx, b).is_none()
        {
            SolvePrepBuildRoute::NegSymbolic
        } else {
            SolvePrepBuildRoute::NegGeneric
        }
    } else if extract_positive_half_scaled_base_expr(ctx, a).is_some() {
        SolvePrepBuildRoute::PosHalf
    } else if is_direct_complete_square_symbolic_scale_expr(ctx, a)
        && strip_unit_negation_for_phase_shift(ctx, b).is_none()
    {
        SolvePrepBuildRoute::PosSymbolic
    } else {
        SolvePrepBuildRoute::PosGeneric
    }
}

/// Clear the per-pipeline solve-prep gate memos (see the thread_local doc).
pub(crate) fn clear_solve_prep_gate_memos() {
    VARIABLE_SQUARE_GATE_MEMO.with(|m| m.borrow_mut().clear());
    SHIFTED_SQUARE_GATE_MEMO.with(|m| m.borrow_mut().clear());
    CANCELLATION_MATCH_MEMO.with(|m| m.borrow_mut().clear());
}

pub(super) fn collect_solve_prep_candidate_variable_names(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
) -> Vec<String> {
    let shifted_primary_vars = collect_shifted_square_primary_variable_names(ctx, root);
    if !shifted_primary_vars.is_empty() {
        return shifted_primary_vars;
    }

    let direct_additive_square_vars = collect_direct_additive_square_variable_names(ctx, root);
    if !direct_additive_square_vars.is_empty() {
        let direct_additive_linear_vars = collect_direct_additive_linear_variable_names(ctx, root);
        let filtered_square_vars: Vec<_> = direct_additive_square_vars
            .into_iter()
            .filter(|var| {
                direct_additive_linear_vars
                    .iter()
                    .any(|linear| linear == var)
            })
            .collect();
        if !filtered_square_vars.is_empty() {
            return filtered_square_vars;
        }
    }

    collect_squared_variable_names(ctx, root)
        .into_iter()
        .filter(|var| expr_contains_named_var_outside_simple_square(ctx, root, var))
        .collect()
}

pub(super) fn has_plausible_solve_prep_focus_remaining_variable_overlap(
    ctx: &cas_ast::Context,
    focus_expr: cas_ast::ExprId,
    remaining_expr: cas_ast::ExprId,
) -> bool {
    let raw_quadratic_vars = collect_squared_variable_names(ctx, focus_expr);
    if raw_quadratic_vars.is_empty() {
        return false;
    }

    let shifted_square_vars = collect_shifted_square_primary_variable_names(ctx, remaining_expr);
    if shifted_square_vars.is_empty() {
        return false;
    }

    raw_quadratic_vars
        .iter()
        .any(|var| shifted_square_vars.iter().any(|shifted| shifted == var))
}

fn is_solve_prep_focus_term_candidate(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Div(num, _) => expr_contains_variable_square(ctx, *num),
        Expr::Neg(inner) => is_solve_prep_focus_term_candidate(ctx, *inner),
        Expr::Mul(_, _) => false,
        _ => expr_contains_variable_square(ctx, expr),
    }
}

pub(crate) fn maybe_solve_prep_exact_additive_candidate(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    let has_raw_quadratic = view
        .terms
        .iter()
        .any(|(term_expr, _)| is_solve_prep_focus_term_candidate(ctx, *term_expr));
    let has_shifted_square = view
        .terms
        .iter()
        .any(|(term_expr, _)| expr_contains_shifted_square(ctx, *term_expr));

    has_raw_quadratic && has_shifted_square
}

pub(super) fn maybe_solve_prep_common_scale_candidate(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    let has_raw_quadratic = view
        .terms
        .iter()
        .any(|(term_expr, _)| expr_contains_variable_square(ctx, *term_expr));
    let has_shifted_square = view
        .terms
        .iter()
        .any(|(term_expr, _)| expr_contains_shifted_square(ctx, *term_expr));

    has_raw_quadratic && has_shifted_square
}

fn collect_exact_solve_prep_core_rewrites_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Vec<SolvePrepExactEquivalenceRewrite> {
    let vars = collect_solve_prep_candidate_variable_names(ctx, expr);
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    if profiling {
        let label = match vars.len() {
            0 => "rule.solve_prep.collect.vars.none",
            1 => "rule.solve_prep.collect.vars.single",
            _ => "rule.solve_prep.collect.vars.multi",
        };
        let _ = run_profiled_orchestrator_option_section(
            label,
            Some(render_expr_for_orchestrator_profile(ctx, expr)),
            || Some(()),
        );
    }
    let mut rewrites = Vec::new();
    for var in vars {
        let Some((rewritten, nonzero_expr, build_route)) =
            build_complete_square_candidate_for_var_for_cancellation(ctx, expr, &var)
        else {
            continue;
        };

        rewrites.push(SolvePrepExactEquivalenceRewrite {
            rewritten,
            local_before: expr,
            local_after: rewritten,
            nonzero_expr,
            build_route,
        });
    }

    rewrites
}

#[cfg(test)]
pub(super) fn try_rewrite_exact_solve_prep_equivalence_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<SolvePrepExactEquivalenceRewrite> {
    collect_exact_solve_prep_equivalence_rewrites_for_cancellation(ctx, expr)
        .into_iter()
        .next()
}

pub(super) fn collect_exact_solve_prep_equivalence_rewrites_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Vec<SolvePrepExactEquivalenceRewrite> {
    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        return collect_exact_solve_prep_equivalence_rewrites_for_cancellation(ctx, num)
            .into_iter()
            .map(|child| SolvePrepExactEquivalenceRewrite {
                rewritten: ctx.add(Expr::Div(child.rewritten, den)),
                local_before: child.local_before,
                local_after: child.local_after,
                nonzero_expr: child.nonzero_expr,
                build_route: child.build_route,
            })
            .collect();
    }

    collect_exact_solve_prep_core_rewrites_for_cancellation(ctx, expr)
}

pub(super) fn try_build_exact_solve_prep_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    let term_count = view.terms.len();

    for subset_len in [3usize, 2, 1] {
        if subset_len >= term_count {
            break;
        }

        for mask in 1usize..(1usize << term_count) {
            if mask.count_ones() as usize != subset_len {
                continue;
            }

            let focus_terms: Vec<_> = view
                .terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| (((mask >> index) & 1) == 1).then_some(term))
                .collect();
            let focus_expr = build_signed_sum_expr(ctx, &focus_terms);

            let remaining_terms: Vec<_> = view
                .terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| (((mask >> index) & 1) == 0).then_some(term))
                .collect();
            if remaining_terms.is_empty() {
                continue;
            }

            let remaining_expr = build_signed_sum_expr(ctx, &remaining_terms);
            for rewrite_match in
                collect_exact_solve_prep_equivalence_rewrites_for_cancellation(ctx, focus_expr)
            {
                let candidate_total = ctx.add(Expr::Add(rewrite_match.rewritten, remaining_expr));
                let neg_rewritten = ctx.add(Expr::Neg(rewrite_match.rewritten));
                let distributed_neg_rewritten =
                    negate_additive_scope_expr(ctx, rewrite_match.rewritten);
                if !(expr_matches_negation_for_cancellation(
                    ctx,
                    rewrite_match.rewritten,
                    remaining_expr,
                ) || expr_matches_negation_after_default_simplify(
                    ctx,
                    rewrite_match.rewritten,
                    remaining_expr,
                ) || exprs_match_after_default_simplify(ctx, neg_rewritten, remaining_expr)
                    || exprs_match_after_default_simplify(
                        ctx,
                        distributed_neg_rewritten,
                        remaining_expr,
                    )
                    || additive_scopes_match_after_default_simplify(
                        ctx,
                        distributed_neg_rewritten,
                        remaining_expr,
                    )
                    || is_zero_after_default_simplify(ctx, candidate_total))
                {
                    continue;
                }

                return Some(
                    Rewrite::with_local(
                        ctx.num(0),
                        "Complete the Square",
                        rewrite_match.local_before,
                        rewrite_match.local_after,
                    )
                    .requires(crate::ImplicitCondition::NonZero(
                        rewrite_match.nonzero_expr,
                    )),
                );
            }
        }
    }

    None
}
