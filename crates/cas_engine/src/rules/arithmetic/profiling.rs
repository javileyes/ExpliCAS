//! `arithmetic`: familia `profiling`.
//!
//! Ver la cabecera de `arithmetic.rs` para el contexto.

use super::*;

pub(super) fn run_profiled_orchestrator_section<T>(
    name: &'static str,
    sample: Option<String>,
    body: impl FnOnce() -> T,
    is_hit: impl FnOnce(&T) -> bool,
) -> T {
    if !crate::orchestrator_shortcut_profiler::should_profile_orchestrator_shortcut(name) {
        return body();
    }

    if let Some(sample) = sample {
        crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(name, sample);
    }

    let start = Instant::now();
    let result = body();
    crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_attempt(
        name,
        is_hit(&result),
        start.elapsed(),
    );
    result
}

pub(super) fn run_profiled_exact_zero_direct_identity_probe(
    profiling: bool,
    name: &'static str,
    sample: &Option<String>,
    body: impl FnOnce() -> Option<Rewrite>,
) -> Option<Rewrite> {
    if profiling {
        run_profiled_orchestrator_option_section(name, sample.clone(), body)
    } else {
        body()
    }
}

pub(super) fn run_profiled_fast_solve_prep_probe<T>(
    profiling: bool,
    name: &'static str,
    sample: &Option<String>,
    body: impl FnOnce() -> Option<T>,
) -> Option<T> {
    if profiling {
        run_profiled_orchestrator_option_section(name, sample.clone(), body)
    } else {
        body()
    }
}

pub(super) fn extract_profiled_solve_prep_nonzero_quadratic_coefficients(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    var: &str,
    profiling: bool,
    sample: &Option<String>,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, cas_ast::ExprId)> {
    let (a, b, c) = if profiling {
        run_profiled_orchestrator_option_section(
            "rule.solve_prep.extract.raw_coeffs",
            sample.clone(),
            || extract_quadratic_coefficients(ctx, expr, var),
        )?
    } else {
        extract_quadratic_coefficients(ctx, expr, var)?
    };
    let sim_a = if profiling {
        run_profiled_orchestrator_option_section(
            "rule.solve_prep.extract.simplify_a",
            sample.clone(),
            || Some(run_default_simplify(ctx, a)),
        )?
    } else {
        run_default_simplify(ctx, a)
    };
    let sim_b = if profiling {
        run_profiled_orchestrator_option_section(
            "rule.solve_prep.extract.simplify_b",
            sample.clone(),
            || Some(run_default_simplify(ctx, b)),
        )?
    } else {
        run_default_simplify(ctx, b)
    };
    let predicted_route = classify_solve_prep_build_route(ctx, sim_a, sim_b);
    if profiling && predicted_route == SolvePrepBuildRoute::PosGeneric {
        let _ = run_profiled_orchestrator_option_section(
            classify_solve_prep_coeff_shape(ctx, sim_a).simplify_a_profile_label(),
            sample.clone(),
            || Some(()),
        );
        let _ = run_profiled_orchestrator_option_section(
            classify_solve_prep_coeff_shape(ctx, sim_b).simplify_b_profile_label(),
            sample.clone(),
            || Some(()),
        );
        let focus_shape_label = if expr_contains_shifted_square(ctx, expr) {
            "rule.solve_prep.extract.pos_generic.focus_shifted_square"
        } else {
            "rule.solve_prep.extract.pos_generic.focus_no_shifted_square"
        };
        let _ =
            run_profiled_orchestrator_option_section(
                focus_shape_label,
                sample.clone(),
                || Some(()),
            );
    }
    let should_defer_c_simplify = matches!(
        predicted_route,
        SolvePrepBuildRoute::NegSymbolic
            | SolvePrepBuildRoute::PosHalf
            | SolvePrepBuildRoute::PosSymbolic
    );
    let sim_c = if profiling {
        if should_defer_c_simplify {
            let _ = run_profiled_orchestrator_option_section(
                predicted_route.defer_simplify_c_profile_label(),
                sample.clone(),
                || Some(()),
            );
            c
        } else {
            run_profiled_orchestrator_option_section(
                "rule.solve_prep.extract.simplify_c",
                sample.clone(),
                || {
                    if predicted_route == SolvePrepBuildRoute::PosGeneric {
                        run_profiled_orchestrator_option_section(
                            predicted_route.simplify_c_profile_label(),
                            sample.clone(),
                            || {
                                run_profiled_orchestrator_option_section(
                                    classify_solve_prep_simplify_c_shape_label(ctx, c),
                                    sample.clone(),
                                    || Some(run_default_simplify(ctx, c)),
                                )
                            },
                        )
                    } else {
                        run_profiled_orchestrator_option_section(
                            predicted_route.simplify_c_profile_label(),
                            sample.clone(),
                            || Some(run_default_simplify(ctx, c)),
                        )
                    }
                },
            )?
        }
    } else if should_defer_c_simplify {
        c
    } else {
        run_default_simplify(ctx, c)
    };
    let a_is_zero = if profiling {
        run_profiled_orchestrator_section(
            "rule.solve_prep.extract.zero_check_a",
            sample.clone(),
            || is_default_simplified_zero_expr(ctx, sim_a),
            |is_zero| *is_zero,
        )
    } else {
        is_default_simplified_zero_expr(ctx, sim_a)
    };
    if profiling {
        let gate_label = if a_is_zero {
            "rule.solve_prep.extract.reject_zero_a"
        } else {
            "rule.solve_prep.extract.keep_nonzero_a"
        };
        let _ = run_profiled_orchestrator_option_section(gate_label, sample.clone(), || Some(()));
    }
    if a_is_zero {
        return None;
    }

    Some((sim_a, sim_b, sim_c))
}

pub(super) fn exact_phase_shift_arg_relation_label_for_profile(
    ctx: &mut cas_ast::Context,
    left_arg: cas_ast::ExprId,
    right_arg: cas_ast::ExprId,
) -> &'static str {
    if compare_expr(ctx, left_arg, right_arg) == Ordering::Equal {
        return "exact_match";
    }

    if matches!(
        (ctx.get(left_arg), ctx.get(right_arg)),
        (Expr::Variable(_), Expr::Variable(_))
            | (Expr::Variable(_), Expr::SessionRef(_))
            | (Expr::SessionRef(_), Expr::Variable(_))
            | (Expr::SessionRef(_), Expr::SessionRef(_))
    ) {
        return "symbolic_leaf_mismatch";
    }

    if exprs_match_for_cancellation_leaf(ctx, left_arg, right_arg) {
        "leaf_equivalent_match"
    } else {
        "other_mismatch"
    }
}

pub(super) fn profile_binary_add_surface_pair_shape_for_phase_shift(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
    sample: Option<String>,
) {
    let lhs_plain_algebraic = is_surface_plain_algebraic_term_for_phase_shift(ctx, lhs);
    let rhs_plain_algebraic = is_surface_plain_algebraic_term_for_phase_shift(ctx, rhs);

    if crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled() {
        let label = match (lhs_plain_algebraic, rhs_plain_algebraic) {
            (true, true) => {
                "rule.phase_shift.binary_add_match.surface_pair_shape.both_plain_algebraic"
            }
            (true, false) => {
                "rule.phase_shift.binary_add_match.surface_pair_shape.lhs_plain_algebraic"
            }
            (false, true) => {
                "rule.phase_shift.binary_add_match.surface_pair_shape.rhs_plain_algebraic"
            }
            (false, false) => {
                "rule.phase_shift.binary_add_match.surface_pair_shape.neither_plain_algebraic"
            }
        };
        let _ = run_profiled_orchestrator_option_section(label, sample, || Some(()));
    }
}

pub(super) fn profile_binary_add_term_family_pair_for_phase_shift(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
    sample: Option<String>,
) {
    if !crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled() {
        return;
    }

    let lhs_family = classify_binary_add_term_family_for_phase_shift(ctx, lhs);
    let rhs_family = classify_binary_add_term_family_for_phase_shift(ctx, rhs);
    let label = match (lhs_family, rhs_family) {
        ("exact_shifted_surface", "exact_shifted_surface") => {
            "rule.phase_shift.binary_add_match.term_family.exact_exact"
        }
        ("exact_shifted_surface", "general_shifted_surface") => {
            "rule.phase_shift.binary_add_match.term_family.exact_general"
        }
        ("general_shifted_surface", "exact_shifted_surface") => {
            "rule.phase_shift.binary_add_match.term_family.general_exact"
        }
        ("exact_shifted_surface", "plain_surface_trig") => {
            "rule.phase_shift.binary_add_match.term_family.exact_plain"
        }
        ("plain_surface_trig", "exact_shifted_surface") => {
            "rule.phase_shift.binary_add_match.term_family.plain_exact"
        }
        ("exact_shifted_surface", "other_trig") => {
            "rule.phase_shift.binary_add_match.term_family.exact_other_trig"
        }
        ("other_trig", "exact_shifted_surface") => {
            "rule.phase_shift.binary_add_match.term_family.other_trig_exact"
        }
        ("exact_shifted_surface", "non_trig") => {
            "rule.phase_shift.binary_add_match.term_family.exact_non_trig"
        }
        ("non_trig", "exact_shifted_surface") => {
            "rule.phase_shift.binary_add_match.term_family.non_trig_exact"
        }
        ("general_shifted_surface", "general_shifted_surface") => {
            "rule.phase_shift.binary_add_match.term_family.general_general"
        }
        ("general_shifted_surface", "plain_surface_trig") => {
            "rule.phase_shift.binary_add_match.term_family.general_plain"
        }
        ("plain_surface_trig", "general_shifted_surface") => {
            "rule.phase_shift.binary_add_match.term_family.plain_general"
        }
        ("general_shifted_surface", "non_trig") => {
            "rule.phase_shift.binary_add_match.term_family.general_non_trig"
        }
        ("non_trig", "general_shifted_surface") => {
            "rule.phase_shift.binary_add_match.term_family.non_trig_general"
        }
        ("plain_surface_trig", "plain_surface_trig") => {
            "rule.phase_shift.binary_add_match.term_family.plain_plain"
        }
        ("plain_surface_trig", "other_trig") => {
            "rule.phase_shift.binary_add_match.term_family.plain_other_trig"
        }
        ("other_trig", "plain_surface_trig") => {
            "rule.phase_shift.binary_add_match.term_family.other_trig_plain"
        }
        ("plain_surface_trig", "non_trig") => {
            "rule.phase_shift.binary_add_match.term_family.plain_non_trig"
        }
        ("non_trig", "plain_surface_trig") => {
            "rule.phase_shift.binary_add_match.term_family.non_trig_plain"
        }
        ("other_trig", "other_trig") => {
            "rule.phase_shift.binary_add_match.term_family.other_trig_other_trig"
        }
        ("other_trig", "non_trig") => {
            "rule.phase_shift.binary_add_match.term_family.other_trig_non_trig"
        }
        ("non_trig", "other_trig") => {
            "rule.phase_shift.binary_add_match.term_family.non_trig_other_trig"
        }
        ("non_trig", "non_trig") => {
            "rule.phase_shift.binary_add_match.term_family.non_trig_non_trig"
        }
        _ => "rule.phase_shift.binary_add_match.term_family.other",
    };
    let _ = run_profiled_orchestrator_option_section(label, sample.clone(), || Some(()));
}

pub(super) fn profile_generated_candidate_target_shape_for_phase_shift(
    ctx: &mut cas_ast::Context,
    target_expr: cas_ast::ExprId,
    target_is_negated: bool,
    sample: Option<String>,
) {
    if !crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled() {
        return;
    }

    let adjusted_target = if target_is_negated {
        strip_unit_negation_for_phase_shift(ctx, target_expr).unwrap_or(target_expr)
    } else {
        target_expr
    };

    let label = if extract_exact_phase_shift_term_data_for_cancellation(ctx, target_expr).is_some()
    {
        "rule.phase_shift.route.exact_try.generated_candidate.target_exact_shifted"
    } else if extract_general_phase_shift_term_data_for_cancellation(ctx, target_expr).is_some() {
        "rule.phase_shift.route.exact_try.generated_candidate.target_general_shifted"
    } else if extract_surface_scaled_trig_term_for_phase_shift(ctx, adjusted_target).is_some() {
        "rule.phase_shift.route.exact_try.generated_candidate.target_surface_trig_nonshift"
    } else if expr_contains_any_builtin(ctx, target_expr, &[BuiltinFn::Sin, BuiltinFn::Cos]) {
        "rule.phase_shift.route.exact_try.generated_candidate.target_other_trig"
    } else {
        "rule.phase_shift.route.exact_try.generated_candidate.target_non_trig"
    };

    let _ = run_profiled_orchestrator_option_section(label, sample.clone(), || Some(()));
}

pub(super) fn profile_exact_phase_shift_pair_relation_for_phase_shift(
    ctx: &mut cas_ast::Context,
    left: ExactPhaseShiftTermData,
    right: ExactPhaseShiftTermData,
    right_is_negated_relative_to_left: bool,
    label_prefix: &'static str,
    sample: Option<String>,
) {
    if !crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled() {
        return;
    }

    let label = exact_phase_shift_pair_relation_label_for_cancellation(
        ctx,
        left,
        right,
        right_is_negated_relative_to_left,
    );

    let _ = run_profiled_orchestrator_option_section(
        match label {
            "arg_mismatch" => "rule.phase_shift.exact_pair_relation.arg_mismatch",
            "sign_mismatch" => "rule.phase_shift.exact_pair_relation.sign_mismatch",
            "coeff_mismatch" => "rule.phase_shift.exact_pair_relation.coeff_mismatch",
            "signature_match" => "rule.phase_shift.exact_pair_relation.signature_match",
            _ => unreachable!(),
        },
        sample.clone(),
        || Some(()),
    );
    let _ = run_profiled_orchestrator_option_section(label_prefix, sample, || Some(()));
}

pub(super) fn profile_binary_add_productive_term_family_gate_for_phase_shift(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
    sample: Option<String>,
) {
    if !crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled() {
        return;
    }

    let lhs_family = classify_binary_add_term_family_for_phase_shift(ctx, lhs);
    let rhs_family = classify_binary_add_term_family_for_phase_shift(ctx, rhs);
    let label = match (lhs_family, rhs_family) {
        _ if phase_shift_term_families_are_single_plain_against_shifted(lhs_family, rhs_family) => {
            "rule.phase_shift.binary_add_match.productive_term_family.single_plain_shifted_reject"
        }
        ("other_trig", "other_trig") => {
            "rule.phase_shift.binary_add_match.productive_term_family.other_trig_other_trig_reject"
        }
        ("exact_shifted_surface", "exact_shifted_surface") => {
            let exact_relation = extract_exact_phase_shift_term_data_for_cancellation(ctx, lhs)
                .zip(extract_exact_phase_shift_term_data_for_cancellation(
                    ctx, rhs,
                ))
                .map(|(left, right)| {
                    exact_phase_shift_pair_relation_label_for_cancellation(ctx, left, right, true)
                });

            match exact_relation {
                Some("signature_match") => {
                    "rule.phase_shift.binary_add_match.productive_term_family.exact_exact_productive"
                }
                Some("arg_mismatch") => {
                    "rule.phase_shift.binary_add_match.productive_term_family.exact_exact_arg_mismatch"
                }
                Some("sign_mismatch") => {
                    "rule.phase_shift.binary_add_match.productive_term_family.exact_exact_sign_mismatch"
                }
                Some("coeff_mismatch") => {
                    "rule.phase_shift.binary_add_match.productive_term_family.exact_exact_coeff_mismatch"
                }
                _ => "rule.phase_shift.binary_add_match.productive_term_family.exact_exact_other",
            }
        }
        _ => "rule.phase_shift.binary_add_match.productive_term_family.other",
    };

    let _ = run_profiled_orchestrator_option_section(label, sample.clone(), || Some(()));
}

pub(super) fn classify_phase_shift_nontrig_entry_detail_for_profile(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> &'static str {
    if try_rewrite_simple_symbolic_scale_sum_for_cancellation(ctx, lhs)
        .map(|rewritten| exprs_match_for_cancellation(ctx, rewritten, rhs))
        .unwrap_or(false)
    {
        "rule.phase_shift.route.entry_pair_shape.both_non_trig.symbolic_scale_sum_lhs"
    } else if try_rewrite_simple_symbolic_scale_sum_for_cancellation(ctx, rhs)
        .map(|rewritten| exprs_match_for_cancellation(ctx, rewritten, lhs))
        .unwrap_or(false)
    {
        "rule.phase_shift.route.entry_pair_shape.both_non_trig.symbolic_scale_sum_rhs"
    } else if matches!(ctx.get(lhs), Expr::Add(_, _) | Expr::Sub(_, _))
        && matches!(ctx.get(rhs), Expr::Add(_, _) | Expr::Sub(_, _))
    {
        "rule.phase_shift.route.entry_pair_shape.both_non_trig.both_additive"
    } else if contains_division_like_term(ctx, lhs) || contains_division_like_term(ctx, rhs) {
        "rule.phase_shift.route.entry_pair_shape.both_non_trig.division_like"
    } else {
        "rule.phase_shift.route.entry_pair_shape.both_non_trig.other"
    }
}

pub(super) fn classify_phase_shift_exact_scope_nontrig_detail_for_profile(
    ctx: &mut cas_ast::Context,
    focus_expr: cas_ast::ExprId,
    remaining_expr: cas_ast::ExprId,
) -> &'static str {
    if try_rewrite_simple_symbolic_scale_sum_for_cancellation(ctx, focus_expr)
        .map(|rewritten| exprs_match_for_cancellation(ctx, rewritten, remaining_expr))
        .unwrap_or(false)
    {
        "rule.phase_shift.exact_scope_pair_shape.both_non_trig.symbolic_scale_sum_focus"
    } else if try_rewrite_simple_symbolic_scale_sum_for_cancellation(ctx, remaining_expr)
        .map(|rewritten| exprs_match_for_cancellation(ctx, rewritten, focus_expr))
        .unwrap_or(false)
    {
        "rule.phase_shift.exact_scope_pair_shape.both_non_trig.symbolic_scale_sum_remaining"
    } else if matches!(ctx.get(focus_expr), Expr::Add(_, _) | Expr::Sub(_, _))
        && matches!(ctx.get(remaining_expr), Expr::Add(_, _) | Expr::Sub(_, _))
    {
        "rule.phase_shift.exact_scope_pair_shape.both_non_trig.both_additive"
    } else if matches!(ctx.get(focus_expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        "rule.phase_shift.exact_scope_pair_shape.both_non_trig.focus_additive"
    } else if matches!(ctx.get(remaining_expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        "rule.phase_shift.exact_scope_pair_shape.both_non_trig.remaining_additive"
    } else if contains_division_like_term(ctx, focus_expr)
        || contains_division_like_term(ctx, remaining_expr)
    {
        "rule.phase_shift.exact_scope_pair_shape.both_non_trig.division_like"
    } else {
        "rule.phase_shift.exact_scope_pair_shape.both_non_trig.other"
    }
}

pub(super) fn phase_shift_supported_arg_fallback_profile_label(denominator: i64) -> &'static str {
    match denominator {
        4 => "rule.phase_shift.supported_arg.normalized_simplify_fallback.quarter",
        3 => "rule.phase_shift.supported_arg.normalized_simplify_fallback.third",
        6 => "rule.phase_shift.supported_arg.normalized_simplify_fallback.sixth",
        _ => "rule.phase_shift.supported_arg.normalized_simplify_fallback.other",
    }
}

pub(super) fn profile_shifted_generated_candidate_target_family_for_phase_shift(
    ctx: &mut cas_ast::Context,
    target_expr: cas_ast::ExprId,
    sample: Option<String>,
) {
    if !crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled() {
        return;
    }

    let label = if let Some(data) =
        extract_general_phase_shift_term_data_for_cancellation(ctx, target_expr)
    {
        match data.trig_fn {
            BuiltinFn::Sin => {
                "rule.phase_shift.route.shifted_try.generated_candidate.target_general_sin"
            }
            BuiltinFn::Cos => {
                "rule.phase_shift.route.shifted_try.generated_candidate.target_general_cos"
            }
            _ => "rule.phase_shift.route.shifted_try.generated_candidate.target_other",
        }
    } else if expr_contains_any_builtin(ctx, target_expr, &[BuiltinFn::Sin, BuiltinFn::Cos]) {
        "rule.phase_shift.route.shifted_try.generated_candidate.target_other_trig"
    } else {
        "rule.phase_shift.route.shifted_try.generated_candidate.target_non_trig"
    };

    let _ = run_profiled_orchestrator_option_section(label, sample.clone(), || Some(()));
}

pub(super) fn linear_focus_phase_shift_compare_profile_label(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    target_is_negated: bool,
) -> &'static str {
    let positive_expr = strip_unit_negation_for_phase_shift(ctx, expr).unwrap_or(expr);
    let mut trig_factor = extract_sin_or_cos_linear_term_for_phase_shift(ctx, positive_expr);

    if trig_factor.is_none() {
        for factor in flatten_mul_chain(ctx, positive_expr) {
            if let Some(found) = extract_sin_or_cos_linear_term_for_phase_shift(ctx, factor) {
                trig_factor = Some(found);
                break;
            }
        }
    }

    let Some((trig_fn, raw_arg)) = trig_factor else {
        return if target_is_negated {
            "rule.phase_shift.linear_focus.compare_candidate.negated.other"
        } else {
            "rule.phase_shift.linear_focus.compare_candidate.direct.other"
        };
    };
    let Some((_base_arg, kind, subtract_shift)) =
        extract_supported_phase_shift_argument_for_cancellation(ctx, trig_fn, raw_arg)
    else {
        return if target_is_negated {
            "rule.phase_shift.linear_focus.compare_candidate.negated.other"
        } else {
            "rule.phase_shift.linear_focus.compare_candidate.direct.other"
        };
    };

    match (target_is_negated, trig_fn, kind, subtract_shift) {
        (false, BuiltinFn::Sin, PhaseShiftKindForCancellation::Quarter, false) => {
            "rule.phase_shift.linear_focus.dir.sin_q_add"
        }
        (false, BuiltinFn::Sin, PhaseShiftKindForCancellation::Quarter, true) => {
            "rule.phase_shift.linear_focus.dir.sin_q_sub"
        }
        (false, BuiltinFn::Cos, PhaseShiftKindForCancellation::Quarter, false) => {
            "rule.phase_shift.linear_focus.dir.cos_q_add"
        }
        (false, BuiltinFn::Cos, PhaseShiftKindForCancellation::Quarter, true) => {
            "rule.phase_shift.linear_focus.dir.cos_q_sub"
        }
        (false, BuiltinFn::Sin, PhaseShiftKindForCancellation::Third, false) => {
            "rule.phase_shift.linear_focus.dir.sin_t_add"
        }
        (false, BuiltinFn::Sin, PhaseShiftKindForCancellation::Third, true) => {
            "rule.phase_shift.linear_focus.dir.sin_t_sub"
        }
        (false, BuiltinFn::Cos, PhaseShiftKindForCancellation::Third, false) => {
            "rule.phase_shift.linear_focus.dir.cos_t_add"
        }
        (false, BuiltinFn::Cos, PhaseShiftKindForCancellation::Third, true) => {
            "rule.phase_shift.linear_focus.dir.cos_t_sub"
        }
        (false, BuiltinFn::Sin, PhaseShiftKindForCancellation::Sixth, false) => {
            "rule.phase_shift.linear_focus.dir.sin_s_add"
        }
        (false, BuiltinFn::Sin, PhaseShiftKindForCancellation::Sixth, true) => {
            "rule.phase_shift.linear_focus.dir.sin_s_sub"
        }
        (false, BuiltinFn::Cos, PhaseShiftKindForCancellation::Sixth, false) => {
            "rule.phase_shift.linear_focus.dir.cos_s_add"
        }
        (false, BuiltinFn::Cos, PhaseShiftKindForCancellation::Sixth, true) => {
            "rule.phase_shift.linear_focus.dir.cos_s_sub"
        }
        (true, BuiltinFn::Sin, PhaseShiftKindForCancellation::Quarter, false) => {
            "rule.phase_shift.linear_focus.neg.sin_q_add"
        }
        (true, BuiltinFn::Sin, PhaseShiftKindForCancellation::Quarter, true) => {
            "rule.phase_shift.linear_focus.neg.sin_q_sub"
        }
        (true, BuiltinFn::Cos, PhaseShiftKindForCancellation::Quarter, false) => {
            "rule.phase_shift.linear_focus.neg.cos_q_add"
        }
        (true, BuiltinFn::Cos, PhaseShiftKindForCancellation::Quarter, true) => {
            "rule.phase_shift.linear_focus.neg.cos_q_sub"
        }
        (true, BuiltinFn::Sin, PhaseShiftKindForCancellation::Third, false) => {
            "rule.phase_shift.linear_focus.neg.sin_t_add"
        }
        (true, BuiltinFn::Sin, PhaseShiftKindForCancellation::Third, true) => {
            "rule.phase_shift.linear_focus.neg.sin_t_sub"
        }
        (true, BuiltinFn::Cos, PhaseShiftKindForCancellation::Third, false) => {
            "rule.phase_shift.linear_focus.neg.cos_t_add"
        }
        (true, BuiltinFn::Cos, PhaseShiftKindForCancellation::Third, true) => {
            "rule.phase_shift.linear_focus.neg.cos_t_sub"
        }
        (true, BuiltinFn::Sin, PhaseShiftKindForCancellation::Sixth, false) => {
            "rule.phase_shift.linear_focus.neg.sin_s_add"
        }
        (true, BuiltinFn::Sin, PhaseShiftKindForCancellation::Sixth, true) => {
            "rule.phase_shift.linear_focus.neg.sin_s_sub"
        }
        (true, BuiltinFn::Cos, PhaseShiftKindForCancellation::Sixth, false) => {
            "rule.phase_shift.linear_focus.neg.cos_s_add"
        }
        (true, BuiltinFn::Cos, PhaseShiftKindForCancellation::Sixth, true) => {
            "rule.phase_shift.linear_focus.neg.cos_s_sub"
        }
        _ => {
            if target_is_negated {
                "rule.phase_shift.linear_focus.neg.other"
            } else {
                "rule.phase_shift.linear_focus.dir.other"
            }
        }
    }
}

pub(super) fn profiled_direct_default_simplify_fallback_for_phase_shift_compare(
    ctx: &mut cas_ast::Context,
    focus_expr: cas_ast::ExprId,
    candidate: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
    cached_target_simplified: &mut Option<cas_ast::ExprId>,
) -> bool {
    let compare_sample = Some(format!(
        "{}  =>  {}  ||  {}",
        render_expr_for_orchestrator_profile(ctx, focus_expr),
        render_expr_for_orchestrator_profile(ctx, candidate),
        render_expr_for_orchestrator_profile(ctx, target_expr)
    ));

    let candidate_simplified = run_profiled_orchestrator_section(
        "rule.phase_shift.linear_focus.compare_candidate.direct.fallback.simplify_candidate",
        compare_sample.clone(),
        || run_default_simplify(ctx, candidate),
        |_| true,
    );
    let target_simplified = if let Some(existing) = *cached_target_simplified {
        existing
    } else {
        let simplified = run_profiled_orchestrator_section(
            "rule.phase_shift.linear_focus.compare_candidate.direct.fallback.simplify_target",
            compare_sample.clone(),
            || run_default_simplify(ctx, target_expr),
            |_| true,
        );
        *cached_target_simplified = Some(simplified);
        simplified
    };

    run_profiled_orchestrator_option_section(
        "rule.phase_shift.linear_focus.compare_candidate.direct.fallback.post_compare",
        compare_sample,
        || exprs_match_for_cancellation(ctx, candidate_simplified, target_simplified).then_some(()),
    )
    .is_some()
}

pub(super) fn profiled_negated_default_simplify_fallback_for_phase_shift_compare(
    ctx: &mut cas_ast::Context,
    focus_expr: cas_ast::ExprId,
    candidate: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
    cached_target_simplified: &mut Option<cas_ast::ExprId>,
) -> bool {
    let compare_sample = Some(format!(
        "{}  =>  {}  ||  {}",
        render_expr_for_orchestrator_profile(ctx, focus_expr),
        render_expr_for_orchestrator_profile(ctx, candidate),
        render_expr_for_orchestrator_profile(ctx, target_expr)
    ));

    let candidate_simplified = run_profiled_orchestrator_section(
        "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.simplify_candidate",
        compare_sample.clone(),
        || run_default_simplify(ctx, candidate),
        |_| true,
    );
    let target_simplified = if let Some(existing) = *cached_target_simplified {
        existing
    } else {
        let simplified = run_profiled_orchestrator_section(
            "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.simplify_target",
            compare_sample.clone(),
            || {
                let neg_target = ctx.add(Expr::Neg(target_expr));
                run_default_simplify(ctx, neg_target)
            },
            |_| true,
        );
        *cached_target_simplified = Some(simplified);
        simplified
    };

    profile_linear_focus_negated_fallback_post_simplify_relation_for_phase_shift(
        ctx,
        candidate_simplified,
        target_simplified,
        compare_sample.clone(),
    );

    run_profiled_orchestrator_option_section(
        "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.post_compare",
        compare_sample,
        || exprs_match_for_cancellation(ctx, candidate_simplified, target_simplified).then_some(()),
    )
    .is_some()
}

pub(super) fn profile_linear_focus_negated_fallback_target_relation_for_phase_shift(
    ctx: &mut cas_ast::Context,
    candidate: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
    sample: Option<String>,
) {
    if !crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled() {
        return;
    }

    let label = if let Some((target_fn, target_arg, _, _)) =
        extract_surface_scaled_trig_term_for_phase_shift(ctx, target_expr)
    {
        if extract_supported_phase_shift_argument_for_cancellation(ctx, target_fn, target_arg)
            .is_some()
        {
            "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.target.exact_shifted"
        } else if expr_contains_pi_constant(ctx, target_arg) {
            "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.target.plain_with_pi"
        } else if let Some((candidate_fn, candidate_arg, _, _)) =
            extract_surface_scaled_trig_term_for_phase_shift(ctx, candidate)
        {
            if let Some((candidate_base, _, _)) =
                extract_supported_phase_shift_argument_for_cancellation(
                    ctx,
                    candidate_fn,
                    candidate_arg,
                )
            {
                let base_arg_matches =
                    exact_phase_shift_args_match_for_cancellation(ctx, candidate_base, target_arg);
                match (candidate_fn == target_fn, base_arg_matches) {
                    (true, true) => {
                        "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.target.plain_same_fn_base_arg_match"
                    }
                    (true, false) => {
                        "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.target.plain_same_fn_base_arg_mismatch"
                    }
                    (false, true) => {
                        "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.target.plain_cross_fn_base_arg_match"
                    }
                    (false, false) => {
                        "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.target.plain_cross_fn_base_arg_mismatch"
                    }
                }
            } else {
                "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.target.other_trig"
            }
        } else {
            "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.target.other_trig"
        }
    } else if expr_contains_any_builtin(ctx, target_expr, &[BuiltinFn::Sin, BuiltinFn::Cos]) {
        "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.target.other_trig"
    } else {
        "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.target.non_trig"
    };

    let _ = run_profiled_orchestrator_option_section(label, sample.clone(), || Some(()));
}

fn profile_linear_focus_negated_fallback_post_simplify_relation_for_phase_shift(
    ctx: &mut cas_ast::Context,
    candidate_simplified: cas_ast::ExprId,
    target_simplified: cas_ast::ExprId,
    sample: Option<String>,
) {
    if !crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled() {
        return;
    }

    let label = if exprs_match_for_cancellation(ctx, candidate_simplified, target_simplified) {
        "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.post_simplify_relation.exact_match"
    } else if let Some((target_fn, target_arg, _, _)) =
        extract_surface_scaled_trig_term_for_phase_shift(ctx, target_simplified)
    {
        if extract_supported_phase_shift_argument_for_cancellation(ctx, target_fn, target_arg)
            .is_some()
        {
            "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.post_simplify_relation.target_exact_shifted"
        } else if expr_contains_pi_constant(ctx, target_arg) {
            "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.post_simplify_relation.target_plain_with_pi"
        } else if let Some((candidate_fn, candidate_arg, _, _)) =
            extract_surface_scaled_trig_term_for_phase_shift(ctx, candidate_simplified)
        {
            if let Some((candidate_base, _, _)) =
                extract_supported_phase_shift_argument_for_cancellation(
                    ctx,
                    candidate_fn,
                    candidate_arg,
                )
            {
                let base_arg_matches =
                    exact_phase_shift_args_match_for_cancellation(ctx, candidate_base, target_arg);
                match (candidate_fn == target_fn, base_arg_matches) {
                    (true, true) => {
                        "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.post_simplify_relation.plain_same_fn_base_arg_match"
                    }
                    (true, false) => {
                        "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.post_simplify_relation.plain_same_fn_base_arg_mismatch"
                    }
                    (false, true) => {
                        "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.post_simplify_relation.plain_cross_fn_base_arg_match"
                    }
                    (false, false) => {
                        "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.post_simplify_relation.plain_cross_fn_base_arg_mismatch"
                    }
                }
            } else {
                "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.post_simplify_relation.target_other_trig"
            }
        } else {
            "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.post_simplify_relation.target_other_trig"
        }
    } else if expr_contains_any_builtin(ctx, target_simplified, &[BuiltinFn::Sin, BuiltinFn::Cos]) {
        "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.post_simplify_relation.target_other_trig"
    } else {
        "rule.phase_shift.linear_focus.compare_candidate.negated.fallback.post_simplify_relation.target_non_trig"
    };

    let _ = run_profiled_orchestrator_option_section(label, sample.clone(), || Some(()));
}

pub(crate) fn classify_exact_zero_common_scale_route_profile_family(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<ExactZeroCommonScaleRouteProfileFamily> {
    if try_build_exact_zero_same_denominator_rewrite(ctx, expr).is_some() {
        return Some(ExactZeroCommonScaleRouteProfileFamily::SameDenominator);
    }

    if let Some((common_factor, lhs_core, rhs_core)) =
        extract_two_term_common_scale_difference_cores(ctx, expr)
    {
        let residual_expr = ctx.add(Expr::Sub(lhs_core, rhs_core));
        if try_build_repeated_trig_phase_shift_pair_zero_rewrite(ctx, residual_expr).is_some()
            || try_build_exact_zero_shared_passthrough_difference_rewrite(ctx, residual_expr)
                .is_some()
            || try_build_direct_tanh_exp_definition_equivalence_rewrite(ctx, lhs_core, rhs_core)
                .is_some()
            || try_build_direct_trig_square_equivalence_rewrite(ctx, lhs_core, rhs_core).is_some()
            || try_build_direct_hyperbolic_sinh_cubic_polynomial_equivalence_rewrite(
                ctx, lhs_core, rhs_core,
            )
            .is_some()
            || try_build_direct_trig_exact_quarter_phase_shift_pair_equivalence_rewrite(
                ctx, lhs_core, rhs_core,
            )
            .is_some()
            || try_build_direct_safe_hyperbolic_core_equivalence_rewrite(ctx, lhs_core, rhs_core)
                .is_some()
        {
            let _ = common_factor;
            return Some(ExactZeroCommonScaleRouteProfileFamily::Other);
        }
    }

    let (_common_factor, residual_expr) = extract_common_multiplicative_residual_sum(ctx, expr)?;
    let residual_term_count = AddView::from_expr(ctx, residual_expr).terms.len();
    if residual_term_count == 2 {
        if let Some((lhs_core, rhs_core)) = extract_two_term_core_difference(ctx, residual_expr) {
            if try_build_direct_core_equivalence_rewrite(ctx, lhs_core, rhs_core).is_some() {
                return Some(ExactZeroCommonScaleRouteProfileFamily::ResidualDirect);
            }
        }
    }
    if try_build_repeated_trig_phase_shift_pair_zero_rewrite(ctx, residual_expr).is_some()
        || try_build_exact_zero_shared_passthrough_difference_rewrite(ctx, residual_expr).is_some()
        || (residual_term_count <= 4
            && try_build_exact_zero_identity_rewrite_direct(ctx, residual_expr).is_some_and(
                |child_rewrite| {
                    let zero = ctx.num(0);
                    compare_expr(ctx, child_rewrite.final_expr(), zero) == Ordering::Equal
                },
            ))
        || try_build_stripped_zero_log_identity_child_rewrite(ctx, residual_expr)
            .or_else(|| {
                try_build_fast_multiterm_hyperbolic_residual_child_rewrite(ctx, residual_expr)
            })
            .is_some()
        || extract_two_term_core_difference(ctx, residual_expr).is_some_and(
            |(lhs_core, rhs_core)| {
                try_build_direct_safe_hyperbolic_core_equivalence_rewrite(ctx, lhs_core, rhs_core)
                    .is_some()
            },
        )
    {
        return Some(ExactZeroCommonScaleRouteProfileFamily::Other);
    }

    let normalized_residual = normalize_additive_scope_expr(ctx, residual_expr);
    if try_build_fast_trig_residual_identity_child_rewrite(ctx, residual_expr).is_some() {
        return Some(ExactZeroCommonScaleRouteProfileFamily::TailFastTrigRaw);
    }
    if try_build_fast_trig_residual_identity_child_rewrite(ctx, normalized_residual).is_some() {
        return Some(ExactZeroCommonScaleRouteProfileFamily::TailFastTrigNormalized);
    }
    if try_build_exact_zero_shared_passthrough_difference_rewrite(ctx, residual_expr).is_some()
        || try_build_fast_small_polynomial_residual_child_rewrite(ctx, residual_expr).is_some()
        || try_build_fast_small_polynomial_residual_child_rewrite(ctx, normalized_residual)
            .is_some()
    {
        return Some(ExactZeroCommonScaleRouteProfileFamily::Other);
    }
    if try_build_two_term_core_equivalence_rewrite(ctx, residual_expr).is_some() {
        return Some(ExactZeroCommonScaleRouteProfileFamily::TailTwoTermCoreEquivalence);
    }
    if try_build_exact_zero_identity_rewrite(ctx, residual_expr).is_some()
        || try_build_exact_zero_identity_rewrite(ctx, normalized_residual).is_some()
    {
        return Some(ExactZeroCommonScaleRouteProfileFamily::Other);
    }

    None
}

pub(super) fn classify_positive_one_passthrough_profile_kind(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> PositiveOnePassthroughProfileKind {
    let view = AddView::from_expr(ctx, expr);
    let positive_one_count = view
        .terms
        .iter()
        .filter(|(term_expr, term_sign)| {
            *term_sign == Sign::Pos
                && matches!(
                    ctx.get(*term_expr),
                    Expr::Number(n) if *n == num_rational::BigRational::from_integer(1.into())
                )
        })
        .count();

    if view.terms.len() == 1 {
        return if positive_one_count == 1 {
            PositiveOnePassthroughProfileKind::SinglePositiveOne
        } else {
            PositiveOnePassthroughProfileKind::SingleOther
        };
    }

    if positive_one_count == 0 {
        return PositiveOnePassthroughProfileKind::AddNoPositiveOne;
    }

    if positive_one_count == view.terms.len() {
        return PositiveOnePassthroughProfileKind::AddOnlyPositiveOne;
    }

    PositiveOnePassthroughProfileKind::Strippable
}

fn classify_add_no_positive_one_profile_detail(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> AddNoPositiveOneProfileDetail {
    let minus_one = num_rational::BigRational::from_integer((-1).into());
    let view = AddView::from_expr(ctx, expr);
    let mut saw_other_numeric = false;

    for (term_expr, term_sign) in view.terms {
        let Expr::Number(number) = ctx.get(term_expr) else {
            continue;
        };
        let signed_number = match term_sign {
            Sign::Pos => number.clone(),
            Sign::Neg => -number.clone(),
        };

        if signed_number == minus_one {
            return AddNoPositiveOneProfileDetail::NegativeOne;
        }
        saw_other_numeric = true;
    }

    if saw_other_numeric {
        AddNoPositiveOneProfileDetail::OtherNumeric
    } else {
        AddNoPositiveOneProfileDetail::NonNumeric
    }
}

pub(super) fn profile_shifted_quotient_exact_one_gate_side(
    side: &'static str,
    kind: PositiveOnePassthroughProfileKind,
    sample: Option<String>,
) {
    if !crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled() {
        return;
    }

    let label = match (side, kind) {
        ("numerator", PositiveOnePassthroughProfileKind::Strippable) => {
            "rule.shifted_quotient.exact_one.gate.numerator.strippable"
        }
        ("numerator", PositiveOnePassthroughProfileKind::SinglePositiveOne) => {
            "rule.shifted_quotient.exact_one.gate.numerator.single_positive_one"
        }
        ("numerator", PositiveOnePassthroughProfileKind::SingleOther) => {
            "rule.shifted_quotient.exact_one.gate.numerator.single_other"
        }
        ("numerator", PositiveOnePassthroughProfileKind::AddNoPositiveOne) => {
            "rule.shifted_quotient.exact_one.gate.numerator.add_no_positive_one"
        }
        ("numerator", PositiveOnePassthroughProfileKind::AddOnlyPositiveOne) => {
            "rule.shifted_quotient.exact_one.gate.numerator.add_only_positive_one"
        }
        ("denominator", PositiveOnePassthroughProfileKind::Strippable) => {
            "rule.shifted_quotient.exact_one.gate.denominator.strippable"
        }
        ("denominator", PositiveOnePassthroughProfileKind::SinglePositiveOne) => {
            "rule.shifted_quotient.exact_one.gate.denominator.single_positive_one"
        }
        ("denominator", PositiveOnePassthroughProfileKind::SingleOther) => {
            "rule.shifted_quotient.exact_one.gate.denominator.single_other"
        }
        ("denominator", PositiveOnePassthroughProfileKind::AddNoPositiveOne) => {
            "rule.shifted_quotient.exact_one.gate.denominator.add_no_positive_one"
        }
        ("denominator", PositiveOnePassthroughProfileKind::AddOnlyPositiveOne) => {
            "rule.shifted_quotient.exact_one.gate.denominator.add_only_positive_one"
        }
        _ => unreachable!(),
    };

    let _ = run_profiled_orchestrator_option_section(label, sample, || Some(()));
}

pub(super) fn profile_shifted_quotient_exact_one_gate_add_no_positive_one_detail(
    ctx: &cas_ast::Context,
    side: &'static str,
    expr: cas_ast::ExprId,
    sample: Option<String>,
) {
    if !crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled() {
        return;
    }

    let detail = classify_add_no_positive_one_profile_detail(ctx, expr);
    let label = match (side, detail) {
        ("numerator", AddNoPositiveOneProfileDetail::NegativeOne) => {
            "rule.shifted_quotient.exact_one.gate.numerator.add_no_positive_one.negative_one"
        }
        ("numerator", AddNoPositiveOneProfileDetail::OtherNumeric) => {
            "rule.shifted_quotient.exact_one.gate.numerator.add_no_positive_one.other_numeric"
        }
        ("numerator", AddNoPositiveOneProfileDetail::NonNumeric) => {
            "rule.shifted_quotient.exact_one.gate.numerator.add_no_positive_one.non_numeric"
        }
        ("denominator", AddNoPositiveOneProfileDetail::NegativeOne) => {
            "rule.shifted_quotient.exact_one.gate.denominator.add_no_positive_one.negative_one"
        }
        ("denominator", AddNoPositiveOneProfileDetail::OtherNumeric) => {
            "rule.shifted_quotient.exact_one.gate.denominator.add_no_positive_one.other_numeric"
        }
        ("denominator", AddNoPositiveOneProfileDetail::NonNumeric) => {
            "rule.shifted_quotient.exact_one.gate.denominator.add_no_positive_one.non_numeric"
        }
        _ => unreachable!(),
    };

    let _ = run_profiled_orchestrator_option_section(label, sample, || Some(()));
}

fn classify_single_symbolic_scale_sum_profile_detail(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<&'static str> {
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

    let has_matching_reciprocal_tail = sum_terms.iter().any(|(term_expr, _)| {
        let Expr::Div(_, denominator) = ctx.get(*term_expr).clone() else {
            return false;
        };
        compare_expr(ctx, denominator, scale_expr) == Ordering::Equal
    });
    if !has_matching_reciprocal_tail {
        return Some("single_scale_plain");
    }

    Some(match ctx.get(scale_expr) {
        Expr::Variable(_) | Expr::SessionRef(_) => "linear_reciprocal_tail",
        Expr::Pow(_, exp) if extract_i64_integer(ctx, *exp).is_some_and(|value| value > 1) => {
            "power_reciprocal_tail"
        }
        _ => "single_scale_other",
    })
}

pub(super) fn classify_symbolic_scale_sum_profile_detail(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> &'static str {
    if let Some(label) = classify_single_symbolic_scale_sum_profile_detail(ctx, expr) {
        return label;
    }

    let sum_terms = AddView::from_expr(ctx, expr).terms;
    if (2..=4).contains(&sum_terms.len())
        && sum_terms.iter().all(|(term_expr, _)| {
            classify_single_symbolic_scale_sum_profile_detail(ctx, *term_expr).is_some()
        })
    {
        return "grouped_multi_scale";
    }

    "other"
}

fn classify_direct_core_equivalence_profile_family(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> DirectCoreEquivalenceProfileFamily {
    if exprs_match_for_cancellation(ctx, lhs_core, rhs_core) {
        DirectCoreEquivalenceProfileFamily::DirectMatch
    } else if try_rewrite_simple_symbolic_scale_sum_for_cancellation(ctx, lhs_core)
        .map(|rewritten| exprs_match_for_cancellation(ctx, rewritten, rhs_core))
        .unwrap_or(false)
    {
        DirectCoreEquivalenceProfileFamily::SymbolicScaleSumLhs
    } else if try_rewrite_simple_symbolic_scale_sum_for_cancellation(ctx, rhs_core)
        .map(|rewritten| exprs_match_for_cancellation(ctx, rewritten, lhs_core))
        .unwrap_or(false)
    {
        DirectCoreEquivalenceProfileFamily::SymbolicScaleSumRhs
    } else if try_build_direct_log_chain_product_equivalence_rewrite(ctx, lhs_core, rhs_core)
        .is_some()
    {
        DirectCoreEquivalenceProfileFamily::LogChainProduct
    } else if try_build_direct_log_expansion_equivalence_rewrite(ctx, lhs_core, rhs_core).is_some()
    {
        DirectCoreEquivalenceProfileFamily::LogExpansion
    } else if try_build_direct_trig_reciprocal_equivalence_rewrite(ctx, lhs_core, rhs_core)
        .is_some()
    {
        DirectCoreEquivalenceProfileFamily::TrigReciprocal
    } else if try_build_direct_trig_cos_diff_sin_diff_quotient_equivalence_rewrite(
        ctx, lhs_core, rhs_core,
    )
    .is_some()
    {
        DirectCoreEquivalenceProfileFamily::CosDiffSinDiffQuotient
    } else if try_build_direct_sum_diff_cubes_quotient_equivalence_rewrite(ctx, lhs_core, rhs_core)
        .is_some()
    {
        DirectCoreEquivalenceProfileFamily::SumDiffCubesQuotient
    } else if try_find_trig_phase_shift_cancellation_match(ctx, lhs_core, rhs_core, false)
        .or_else(|| try_find_trig_phase_shift_cancellation_match(ctx, rhs_core, lhs_core, false))
        .is_some()
    {
        DirectCoreEquivalenceProfileFamily::PhaseShiftIdentity
    } else if try_build_direct_cos_product_telescoping_equivalence_rewrite(ctx, lhs_core, rhs_core)
        .is_some()
    {
        DirectCoreEquivalenceProfileFamily::CosProductTelescoping
    } else if try_build_direct_finite_sum_equivalence_rewrite(ctx, lhs_core, rhs_core).is_some() {
        DirectCoreEquivalenceProfileFamily::FiniteSum
    } else if try_build_direct_finite_product_equivalence_rewrite(ctx, lhs_core, rhs_core).is_some()
    {
        DirectCoreEquivalenceProfileFamily::FiniteProduct
    } else if try_build_direct_trig_power_reduction_equivalence_rewrite(ctx, lhs_core, rhs_core)
        .is_some()
    {
        DirectCoreEquivalenceProfileFamily::TrigPowerReduction
    } else if try_build_direct_trig_double_angle_contraction_equivalence_rewrite(
        ctx, lhs_core, rhs_core,
    )
    .is_some()
    {
        DirectCoreEquivalenceProfileFamily::DoubleAngleContraction
    } else if exprs_match_after_default_simplify(ctx, lhs_core, rhs_core) {
        DirectCoreEquivalenceProfileFamily::DefaultSimplify
    } else {
        DirectCoreEquivalenceProfileFamily::Other
    }
}

pub(crate) fn classify_exact_zero_common_scale_residual_direct_profile_label(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<&'static str> {
    let (_common_factor, residual_expr) = extract_common_multiplicative_residual_sum(ctx, expr)?;
    let (lhs_core, rhs_core) = extract_two_term_core_difference(ctx, residual_expr)?;
    let family = classify_direct_core_equivalence_profile_family(ctx, lhs_core, rhs_core);

    Some(match family {
        DirectCoreEquivalenceProfileFamily::DirectMatch => {
            "rule.direct_core_equivalence.family.direct_match"
        }
        DirectCoreEquivalenceProfileFamily::SymbolicScaleSumLhs => {
            "rule.direct_core_equivalence.family.symbolic_scale_sum_lhs"
        }
        DirectCoreEquivalenceProfileFamily::SymbolicScaleSumRhs => {
            "rule.direct_core_equivalence.family.symbolic_scale_sum_rhs"
        }
        DirectCoreEquivalenceProfileFamily::LogExpansion => {
            "rule.direct_core_equivalence.family.log_expansion"
        }
        DirectCoreEquivalenceProfileFamily::LogChainProduct => {
            "rule.direct_core_equivalence.family.log_chain_product"
        }
        DirectCoreEquivalenceProfileFamily::TrigReciprocal => {
            "rule.direct_core_equivalence.family.trig_reciprocal"
        }
        DirectCoreEquivalenceProfileFamily::CosDiffSinDiffQuotient => {
            "rule.direct_core_equivalence.family.cos_diff_sin_diff_quotient"
        }
        DirectCoreEquivalenceProfileFamily::SumDiffCubesQuotient => {
            "rule.direct_core_equivalence.family.sum_diff_cubes_quotient"
        }
        DirectCoreEquivalenceProfileFamily::PhaseShiftIdentity => {
            "rule.direct_core_equivalence.family.phase_shift_identity"
        }
        DirectCoreEquivalenceProfileFamily::CosProductTelescoping => {
            "rule.direct_core_equivalence.family.cos_product_telescoping"
        }
        DirectCoreEquivalenceProfileFamily::FiniteSum => {
            "rule.direct_core_equivalence.family.finite_sum"
        }
        DirectCoreEquivalenceProfileFamily::FiniteProduct => {
            "rule.direct_core_equivalence.family.finite_product"
        }
        DirectCoreEquivalenceProfileFamily::TrigPowerReduction => {
            "rule.direct_core_equivalence.family.trig_power_reduction"
        }
        DirectCoreEquivalenceProfileFamily::DoubleAngleContraction => {
            "rule.direct_core_equivalence.family.double_angle_contraction"
        }
        DirectCoreEquivalenceProfileFamily::DefaultSimplify => {
            direct_core_default_simplify_profile_label(ctx, lhs_core, rhs_core)
        }
        DirectCoreEquivalenceProfileFamily::Other => "rule.direct_core_equivalence.family.other",
    })
}

pub(crate) fn classify_exact_zero_common_scale_residual_direct_other_profile_label(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<&'static str> {
    let (_common_factor, residual_expr) = extract_common_multiplicative_residual_sum(ctx, expr)?;
    let (lhs_core, rhs_core) = extract_two_term_core_difference(ctx, residual_expr)?;
    let family = classify_direct_core_equivalence_profile_family(ctx, lhs_core, rhs_core);

    Some(match family {
        DirectCoreEquivalenceProfileFamily::PhaseShiftIdentity => {
            "root.div.03g2c7c2b3a.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.phase_shift_identity"
        }
        DirectCoreEquivalenceProfileFamily::DefaultSimplify => {
            match direct_core_default_simplify_profile_label(ctx, lhs_core, rhs_core) {
                "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.other" => {
                    "root.div.03g2c7c2b3b.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.default_simplify_non_hyperbolic_other"
                }
                _ => {
                    "root.div.03g2c7c2b3c.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.default_simplify_other"
                }
            }
        }
        DirectCoreEquivalenceProfileFamily::Other => {
            "root.div.03g2c7c2b3d.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.direct_core_other"
        }
        _ => {
            "root.div.03g2c7c2b3e.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.remaining_family"
        }
    })
}

pub(crate) fn classify_exact_zero_common_scale_residual_direct_phase_shift_identity_profile_label(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<&'static str> {
    let (_common_factor, residual_expr) = extract_common_multiplicative_residual_sum(ctx, expr)?;
    let (lhs_core, rhs_core) = extract_two_term_core_difference(ctx, residual_expr)?;

    let classify_match = |direction: &'static str,
                          rewrite_match: TrigPhaseShiftCancellationMatch| {
        match (
            direction,
            rewrite_match.mode,
        ) {
            ("forward", TrigPhaseShiftCancellationMode::LinearToShifted) => {
                "root.div.03g2c7c2b3a1.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.phase_shift_identity.forward_linear_to_shifted"
            }
            ("forward", TrigPhaseShiftCancellationMode::ShiftedToLinear) => {
                "root.div.03g2c7c2b3a2.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.phase_shift_identity.forward_shifted_to_linear"
            }
            ("forward", TrigPhaseShiftCancellationMode::ShiftedToShifted) => {
                "root.div.03g2c7c2b3a3.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.phase_shift_identity.forward_shifted_to_shifted"
            }
            ("reverse", TrigPhaseShiftCancellationMode::LinearToShifted) => {
                "root.div.03g2c7c2b3a4.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.phase_shift_identity.reverse_linear_to_shifted"
            }
            ("reverse", TrigPhaseShiftCancellationMode::ShiftedToLinear) => {
                "root.div.03g2c7c2b3a5.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.phase_shift_identity.reverse_shifted_to_linear"
            }
            ("reverse", TrigPhaseShiftCancellationMode::ShiftedToShifted) => {
                "root.div.03g2c7c2b3a6.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.phase_shift_identity.reverse_shifted_to_shifted"
            }
            _ => unreachable!(),
        }
    };

    if let Some(rewrite_match) =
        try_find_trig_phase_shift_cancellation_match(ctx, lhs_core, rhs_core, false)
    {
        return Some(classify_match("forward", rewrite_match));
    }
    if let Some(rewrite_match) =
        try_find_trig_phase_shift_cancellation_match(ctx, rhs_core, lhs_core, false)
    {
        return Some(classify_match("reverse", rewrite_match));
    }

    None
}

fn expr_contains_fractional_power_for_profile(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            matches!(ctx.get(*exp), Expr::Number(n) if !n.is_integer())
                || expr_contains_fractional_power_for_profile(ctx, *base)
                || expr_contains_fractional_power_for_profile(ctx, *exp)
        }
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) | Expr::Div(lhs, rhs) => {
            expr_contains_fractional_power_for_profile(ctx, *lhs)
                || expr_contains_fractional_power_for_profile(ctx, *rhs)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            expr_contains_fractional_power_for_profile(ctx, *inner)
        }
        Expr::Function(_, args) => args
            .iter()
            .copied()
            .any(|arg| expr_contains_fractional_power_for_profile(ctx, arg)),
        Expr::Matrix { data, .. } => data
            .iter()
            .copied()
            .any(|entry| expr_contains_fractional_power_for_profile(ctx, entry)),
        _ => false,
    }
}

fn expr_is_radical_like_for_profile(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    expr_contains_any_builtin(ctx, expr, &[BuiltinFn::Sqrt])
        || expr_contains_fractional_power_for_profile(ctx, expr)
}

fn matches_signed_double_angle_contraction_profile_pair(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> bool {
    let (lhs_expr, lhs_sign) = normalize_signed_add_term_for_fast_match(ctx, lhs_core, Sign::Pos);
    let (rhs_expr, rhs_sign) = normalize_signed_add_term_for_fast_match(ctx, rhs_core, Sign::Pos);
    lhs_sign == rhs_sign
        && try_build_direct_trig_double_angle_contraction_equivalence_rewrite(
            ctx, lhs_expr, rhs_expr,
        )
        .is_some()
}

fn classify_quotient_cancel_profile_shape(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<QuotientCancelProfileShape> {
    let (num, den) = as_div(ctx, expr)?;

    if expr_is_radical_like_for_profile(ctx, expr)
        || expr_is_radical_like_for_profile(ctx, num)
        || expr_is_radical_like_for_profile(ctx, den)
    {
        return Some(QuotientCancelProfileShape::Radical);
    }

    if matches!(ctx.get(num), Expr::Add(_, _) | Expr::Sub(_, _))
        || matches!(ctx.get(den), Expr::Add(_, _) | Expr::Sub(_, _))
    {
        return Some(QuotientCancelProfileShape::Polynomial);
    }

    if !matches!(ctx.get(num), Expr::Div(_, _))
        && !matches!(ctx.get(den), Expr::Div(_, _))
        && !matches!(ctx.get(num), Expr::Add(_, _) | Expr::Sub(_, _))
        && !matches!(ctx.get(den), Expr::Add(_, _) | Expr::Sub(_, _))
    {
        return Some(QuotientCancelProfileShape::Monomial);
    }

    Some(QuotientCancelProfileShape::Other)
}

fn quotient_cancel_profile_label(
    ctx: &cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> &'static str {
    let lhs_shape = classify_quotient_cancel_profile_shape(ctx, lhs_core);
    let rhs_shape = classify_quotient_cancel_profile_shape(ctx, rhs_core);

    match (lhs_shape, rhs_shape) {
        (Some(QuotientCancelProfileShape::Radical), Some(_))
        | (Some(_), Some(QuotientCancelProfileShape::Radical)) => {
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.radical_pair"
        }
        (Some(QuotientCancelProfileShape::Radical), None)
        | (None, Some(QuotientCancelProfileShape::Radical)) => {
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.radical_single"
        }
        (Some(QuotientCancelProfileShape::Polynomial), Some(_))
        | (Some(_), Some(QuotientCancelProfileShape::Polynomial)) => {
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_pair"
        }
        (Some(QuotientCancelProfileShape::Polynomial), None)
        | (None, Some(QuotientCancelProfileShape::Polynomial)) => {
            classify_polynomial_single_quotient_cancel_profile_pair(ctx, lhs_core, rhs_core)
                .unwrap_or(
                    "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_single.other",
                )
        }
        (Some(QuotientCancelProfileShape::Monomial), Some(_))
        | (Some(_), Some(QuotientCancelProfileShape::Monomial)) => {
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.monomial_pair"
        }
        (Some(QuotientCancelProfileShape::Monomial), None)
        | (None, Some(QuotientCancelProfileShape::Monomial)) => {
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.monomial_single"
        }
        (Some(_), Some(_)) => {
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.other_pair"
        }
        _ => "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.other_single",
    }
}

fn classify_polynomial_single_quotient_cancel_profile_pair(
    ctx: &cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<&'static str> {
    for (quot_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((num, den)) = as_div(ctx, quot_expr) else {
            continue;
        };
        if !matches!(ctx.get(num), Expr::Add(_, _) | Expr::Sub(_, _)) {
            continue;
        }

        if MulView::from_expr(ctx, den).len() >= 2 {
            let target_terms = AddView::from_expr(ctx, target_expr).terms;
            if target_terms.len() >= 2
                && target_terms
                    .iter()
                    .all(|(term, _)| matches!(ctx.get(*term), Expr::Div(_, _)))
            {
                let has_full_denominator_tail = target_terms
                    .iter()
                    .filter_map(|(term, _)| as_div(ctx, *term).map(|(_, term_den)| term_den))
                    .any(|term_den| compare_expr(ctx, term_den, den) == Ordering::Equal);

                return Some(if has_full_denominator_tail {
                    "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_single.product_denominator_plus_tail"
                } else {
                    "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_single.product_denominator_split"
                });
            }

            if target_terms.len() >= 2
                && target_terms.iter().all(|(_, sign)| *sign == Sign::Pos)
                && target_terms
                    .iter()
                    .filter(|(term, _)| as_div(ctx, *term).is_none())
                    .count()
                    == 1
                && target_terms
                    .iter()
                    .filter(|(term, _)| as_div(ctx, *term).is_some())
                    .count()
                    >= 1
            {
                return Some(
                    "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_single.product_denominator_with_whole_term",
                );
            }
        }

        let target_terms = AddView::from_expr(ctx, target_expr).terms;
        if target_terms.len() == 2 {
            let first_div = as_div(ctx, target_terms[0].0);
            let second_div = as_div(ctx, target_terms[1].0);
            if let (Some((_, first_den)), Some((_, second_den))) = (first_div, second_div) {
                if target_terms[0].1 != target_terms[1].1
                    && compare_expr(ctx, first_den, second_den) == Ordering::Equal
                {
                    return Some(
                        "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_single.same_denominator_difference",
                    );
                }
            }

            if (first_div.is_some() && second_div.is_none())
                || (first_div.is_none() && second_div.is_some())
            {
                return Some(
                    "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_single.whole_fraction_pair",
                );
            }
        }

        if !contains_division_like_term(ctx, target_expr) {
            return Some(
                "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_single.nonfraction_target",
            );
        }
    }

    None
}

fn is_top_level_additive_for_profile(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
}

fn expr_contains_hyperbolic_builtin_for_profile(
    ctx: &cas_ast::Context,
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

pub(super) fn expr_contains_log_builtin_for_profile(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    expr_contains_any_builtin(
        ctx,
        expr,
        &[
            BuiltinFn::Ln,
            BuiltinFn::Log,
            BuiltinFn::Log2,
            BuiltinFn::Log10,
        ],
    )
}

fn classify_log_arg_profile_shape(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> LogArgProfileShape {
    match ctx.get(expr) {
        Expr::Variable(_) | Expr::SessionRef(_) | Expr::Number(_) | Expr::Constant(_) => {
            LogArgProfileShape::Atom
        }
        Expr::Neg(inner) => classify_log_arg_profile_shape(ctx, *inner),
        Expr::Add(_, _) | Expr::Sub(_, _) => LogArgProfileShape::AddSub,
        Expr::Mul(_, _) => LogArgProfileShape::Mul,
        Expr::Div(_, _) => LogArgProfileShape::Div,
        Expr::Pow(_, _) => LogArgProfileShape::Pow,
        Expr::Function(fn_id, _) if ctx.is_builtin(*fn_id, BuiltinFn::Abs) => {
            LogArgProfileShape::Abs
        }
        Expr::Function(_, _) => LogArgProfileShape::Function,
        _ => LogArgProfileShape::Other,
    }
}

fn extract_log_profile_member(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<LogProfileMember> {
    let (negated, inner) = match ctx.get(expr) {
        Expr::Neg(inner) => (true, *inner),
        _ => (false, expr),
    };
    let (_base, arg) = try_extract_log_parts(ctx, inner)?;
    let kind = match ctx.get(inner) {
        Expr::Function(fn_id, args) if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Ln) => {
            LogProfileKind::Ln
        }
        _ => LogProfileKind::GeneralBase,
    };
    Some(LogProfileMember {
        negated,
        kind,
        arg_shape: classify_log_arg_profile_shape(ctx, arg),
    })
}

fn classify_nonhyperbolic_log_pair_profile_label(
    ctx: &cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> &'static str {
    let lhs = extract_log_profile_member(ctx, lhs_core);
    let rhs = extract_log_profile_member(ctx, rhs_core);
    let Some(lhs) = lhs else {
        return "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_pair.other";
    };
    let Some(rhs) = rhs else {
        return "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_pair.other";
    };

    let pair_arg_shape = |a: LogArgProfileShape, b: LogArgProfileShape| match (a, b) {
        (LogArgProfileShape::Atom, LogArgProfileShape::Div)
        | (LogArgProfileShape::Div, LogArgProfileShape::Atom) => "atomic_div",
        (LogArgProfileShape::Atom, LogArgProfileShape::Pow)
        | (LogArgProfileShape::Pow, LogArgProfileShape::Atom) => "atomic_pow",
        (LogArgProfileShape::Pow, LogArgProfileShape::Pow) => "pow_pair",
        (LogArgProfileShape::Mul, LogArgProfileShape::Pow)
        | (LogArgProfileShape::Pow, LogArgProfileShape::Mul) => "mul_pow",
        (LogArgProfileShape::Mul, LogArgProfileShape::Mul) => "mul_pair",
        (LogArgProfileShape::Div, LogArgProfileShape::Div) => "div_pair",
        (LogArgProfileShape::Abs, LogArgProfileShape::Abs) => "abs_pair",
        (LogArgProfileShape::AddSub, LogArgProfileShape::AddSub) => "addsub_pair",
        _ => "other",
    };

    match (lhs.negated, rhs.negated, lhs.kind, rhs.kind) {
        (false, true, LogProfileKind::Ln, LogProfileKind::Ln)
        | (true, false, LogProfileKind::Ln, LogProfileKind::Ln) => match pair_arg_shape(
            lhs.arg_shape,
            rhs.arg_shape,
        ) {
            "atomic_div" => {
                "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_pair.negated_ln.atomic_div"
            }
            "atomic_pow" => {
                "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_pair.negated_ln.atomic_pow"
            }
            "pow_pair" => {
                "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_pair.negated_ln.pow_pair"
            }
            _ => {
                "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_pair.negated_ln.other"
            }
        },
        (false, true, LogProfileKind::GeneralBase, LogProfileKind::GeneralBase)
        | (true, false, LogProfileKind::GeneralBase, LogProfileKind::GeneralBase) => {
            match pair_arg_shape(lhs.arg_shape, rhs.arg_shape) {
                "atomic_div" => {
                    "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_pair.negated_general_base.atomic_div"
                }
                "atomic_pow" => {
                    "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_pair.negated_general_base.atomic_pow"
                }
                "pow_pair" => {
                    "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_pair.negated_general_base.pow_pair"
                }
                _ => {
                    "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_pair.negated_general_base.other"
                }
            }
        }
        (false, false, LogProfileKind::Ln, LogProfileKind::Ln) => {
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_pair.ln_pair"
        }
        (false, false, LogProfileKind::GeneralBase, LogProfileKind::GeneralBase) => {
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_pair.general_base_pair"
        }
        (true, true, _, _) => {
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_pair.negated_pair"
        }
        _ => "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_pair.mixed_base_or_shape",
    }
}

pub(super) fn expr_contains_named_function_for_profile(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
    names: &[&str],
) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) => {
            names.iter().any(|name| ctx.sym_name(*fn_id) == *name)
                || args
                    .iter()
                    .copied()
                    .any(|arg| expr_contains_named_function_for_profile(ctx, arg, names))
        }
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) | Expr::Div(lhs, rhs) => {
            expr_contains_named_function_for_profile(ctx, *lhs, names)
                || expr_contains_named_function_for_profile(ctx, *rhs, names)
        }
        Expr::Pow(base, exp) => {
            expr_contains_named_function_for_profile(ctx, *base, names)
                || expr_contains_named_function_for_profile(ctx, *exp, names)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            expr_contains_named_function_for_profile(ctx, *inner, names)
        }
        Expr::Matrix { data, .. } => data
            .iter()
            .copied()
            .any(|entry| expr_contains_named_function_for_profile(ctx, entry, names)),
        _ => false,
    }
}

pub(super) fn extract_shifted_surface_trig_base_arg_for_profile(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BuiltinFn, cas_ast::ExprId, cas_ast::ExprId)> {
    let (trig_fn, trig_arg, coeff, _sign) =
        extract_surface_scaled_trig_term_for_phase_shift(ctx, expr)?;

    if let Some((base_arg, _, _)) =
        extract_supported_phase_shift_argument_for_cancellation(ctx, trig_fn, trig_arg)
    {
        return Some((trig_fn, base_arg, coeff));
    }

    if let Some((base_arg, _, _)) =
        extract_general_phase_shift_argument_for_cancellation(ctx, trig_arg)
    {
        return Some((trig_fn, base_arg, coeff));
    }

    None
}

fn matches_shifted_surface_trig_mismatch_profile_pair(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> bool {
    let Some((lhs_fn, lhs_base_arg, lhs_coeff)) =
        extract_shifted_surface_trig_base_arg_for_profile(ctx, lhs_core)
    else {
        return false;
    };
    let Some((rhs_fn, rhs_base_arg, rhs_coeff)) =
        extract_shifted_surface_trig_base_arg_for_profile(ctx, rhs_core)
    else {
        return false;
    };

    lhs_fn == rhs_fn
        && compare_expr(ctx, lhs_coeff, rhs_coeff) == Ordering::Equal
        && expr_contains_symbolic_atom_for_cancellation(ctx, lhs_base_arg)
        && expr_contains_symbolic_atom_for_cancellation(ctx, rhs_base_arg)
        && !exprs_match_for_cancellation_leaf(ctx, lhs_base_arg, rhs_base_arg)
}

fn matches_plain_surface_trig_power_gap_profile_pair(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> bool {
    for (plain_expr, power_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((plain_fn, plain_arg)) =
            extract_sin_or_cos_linear_term_for_phase_shift(ctx, plain_expr)
        else {
            continue;
        };
        let Expr::Pow(power_base, power_exp) = ctx.get(power_expr).clone() else {
            continue;
        };
        let Some((power_fn, power_arg)) =
            extract_sin_or_cos_linear_term_for_phase_shift(ctx, power_base)
        else {
            continue;
        };
        let Some(power) = small_positive_integer_value(ctx, power_exp) else {
            continue;
        };
        if power >= 2
            && plain_fn == power_fn
            && exprs_match_for_cancellation_leaf(ctx, plain_arg, power_arg)
        {
            return true;
        }
    }

    false
}

fn classify_finite_series_vs_other_profile_pair(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> &'static str {
    for series_expr in [lhs_core, rhs_core] {
        if let Some(plan) = try_plan_finite_sum_evaluation(ctx, series_expr, 1000) {
            return match plan.kind {
                SumEvaluationKind::Telescoping => {
                    "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.sum_telescoping"
                }
                SumEvaluationKind::SumOfFirstIntegers => {
                    "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.sum_first_integers"
                }
                SumEvaluationKind::SumOfSquares => {
                    "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.sum_squares"
                }
                SumEvaluationKind::SumOfCubes => {
                    "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.sum_cubes"
                }
                SumEvaluationKind::SumOfConstant => {
                    "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.sum_constant"
                }
                SumEvaluationKind::GeometricPower => {
                    "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.sum_geometric_power"
                }
                SumEvaluationKind::PolynomialLinearity => {
                    "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.sum_polynomial_linearity"
                }
                SumEvaluationKind::FiniteDirect { .. } => {
                    "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.sum_direct"
                }
                SumEvaluationKind::DivergentInfinite => {
                    "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.sum_divergent_infinite"
                }
                SumEvaluationKind::ConvergentInfinite => {
                    "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.sum_convergent_infinite"
                }
                SumEvaluationKind::UndefinedPole => {
                    "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.sum_undefined_pole"
                }
            };
        }

        if try_plan_finite_product_evaluation(ctx, series_expr, 1000).is_some() {
            return "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.product_evaluable";
        }
    }

    "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.other"
}

pub(super) fn extract_hyperbolic_linear_term_for_profile(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BuiltinFn, cas_ast::ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    if ctx.is_builtin(*fn_id, BuiltinFn::Sinh) {
        Some((BuiltinFn::Sinh, args[0]))
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cosh) {
        Some((BuiltinFn::Cosh, args[0]))
    } else {
        None
    }
}

fn extract_single_hyperbolic_linear_or_cubic_term_for_profile(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BuiltinFn, cas_ast::ExprId, i64)> {
    let expr = strip_unit_negation_for_phase_shift(ctx, expr).unwrap_or(expr);
    if let Some((builtin, arg)) = extract_hyperbolic_linear_term_for_profile(ctx, expr) {
        return Some((builtin, arg, 1));
    }

    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    let mut hyperbolic_term = None;
    for factor in factors {
        if let Some((builtin, arg)) = extract_hyperbolic_linear_term_for_profile(ctx, factor) {
            if hyperbolic_term.is_some() {
                return None;
            }
            hyperbolic_term = Some((builtin, arg, 1));
            continue;
        }

        let Expr::Pow(base, exp) = ctx.get(factor).clone() else {
            if expr_contains_any_function_call(ctx, factor) {
                return None;
            }
            continue;
        };
        let Some((builtin, arg)) = extract_hyperbolic_linear_term_for_profile(ctx, base) else {
            if expr_contains_any_function_call(ctx, factor) {
                return None;
            }
            continue;
        };
        let power = small_positive_integer_value(ctx, exp)?;
        if power != 3 || hyperbolic_term.is_some() {
            return None;
        }
        hyperbolic_term = Some((builtin, arg, power));
    }

    hyperbolic_term
}

fn extract_two_factor_hyperbolic_product_for_profile(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<[(BuiltinFn, cas_ast::ExprId); 2]> {
    let expr = strip_unit_negation_for_phase_shift(ctx, expr).unwrap_or(expr);
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let first = extract_hyperbolic_linear_term_for_profile(ctx, factors[0])?;
    let second = extract_hyperbolic_linear_term_for_profile(ctx, factors[1])?;
    Some([first, second])
}

fn matches_hyperbolic_cosh_cubic_linear_profile_pair(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> bool {
    for (cubic_expr, linear_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((cubic_builtin, cubic_arg, cubic_power)) =
            extract_single_hyperbolic_linear_or_cubic_term_for_profile(ctx, cubic_expr)
        else {
            continue;
        };
        let Some((linear_builtin, linear_arg, linear_power)) =
            extract_single_hyperbolic_linear_or_cubic_term_for_profile(ctx, linear_expr)
        else {
            continue;
        };

        if cubic_builtin == BuiltinFn::Cosh
            && linear_builtin == BuiltinFn::Cosh
            && cubic_power == 3
            && linear_power == 1
            && compare_expr(ctx, cubic_arg, linear_arg) == Ordering::Equal
        {
            return true;
        }
    }

    false
}

fn matches_hyperbolic_cross_swap_profile_pair(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> bool {
    let Some(lhs_terms) = extract_two_factor_hyperbolic_product_for_profile(ctx, lhs_core) else {
        return false;
    };
    let Some(rhs_terms) = extract_two_factor_hyperbolic_product_for_profile(ctx, rhs_core) else {
        return false;
    };

    let lhs_is_cross = lhs_terms[0].0 != lhs_terms[1].0;
    let rhs_is_cross = rhs_terms[0].0 != rhs_terms[1].0;
    if !lhs_is_cross || !rhs_is_cross {
        return false;
    }

    (compare_expr(ctx, lhs_terms[0].1, rhs_terms[0].1) == Ordering::Equal
        && compare_expr(ctx, lhs_terms[1].1, rhs_terms[1].1) == Ordering::Equal
        && lhs_terms[0].0 != rhs_terms[0].0
        && lhs_terms[1].0 != rhs_terms[1].0)
        || (compare_expr(ctx, lhs_terms[0].1, rhs_terms[1].1) == Ordering::Equal
            && compare_expr(ctx, lhs_terms[1].1, rhs_terms[0].1) == Ordering::Equal
            && lhs_terms[0].0 != rhs_terms[1].0
            && lhs_terms[1].0 != rhs_terms[0].0)
}

fn matches_hyperbolic_square_product_gap_profile_pair(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> bool {
    for (cosh_expr, sinh_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(cosh_terms) = extract_two_factor_hyperbolic_product_for_profile(ctx, cosh_expr)
        else {
            continue;
        };
        let Some(sinh_terms) = extract_two_factor_hyperbolic_product_for_profile(ctx, sinh_expr)
        else {
            continue;
        };

        if cosh_terms
            .iter()
            .all(|(builtin, _)| *builtin == BuiltinFn::Cosh)
            && sinh_terms
                .iter()
                .all(|(builtin, _)| *builtin == BuiltinFn::Sinh)
            && compare_expr(ctx, cosh_terms[0].1, sinh_terms[0].1) == Ordering::Equal
            && compare_expr(ctx, cosh_terms[1].1, sinh_terms[1].1) == Ordering::Equal
        {
            return true;
        }

        if cosh_terms
            .iter()
            .all(|(builtin, _)| *builtin == BuiltinFn::Cosh)
            && sinh_terms
                .iter()
                .all(|(builtin, _)| *builtin == BuiltinFn::Sinh)
            && compare_expr(ctx, cosh_terms[0].1, sinh_terms[1].1) == Ordering::Equal
            && compare_expr(ctx, cosh_terms[1].1, sinh_terms[0].1) == Ordering::Equal
        {
            return true;
        }
    }

    false
}

fn extract_symbolic_half_power_merge_source_for_profile(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let mut half_power_base = None;
    let mut other_power = None;
    for factor in factors {
        if let Some(base) = extract_sqrt_argument(ctx, factor) {
            if half_power_base.is_some() {
                return None;
            }
            half_power_base = Some(base);
            continue;
        }

        let Expr::Pow(base, exp) = ctx.get(factor).clone() else {
            return None;
        };
        if other_power.is_some() {
            return None;
        }
        other_power = Some((base, exp));
    }

    let half_power_base = half_power_base?;
    let (power_base, power_exp) = other_power?;
    (compare_expr(ctx, half_power_base, power_base) == Ordering::Equal)
        .then_some((half_power_base, power_exp))
}

fn matches_symbolic_half_power_merge_profile_pair(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((base, power_exp)) =
            extract_symbolic_half_power_merge_source_for_profile(ctx, source_expr)
        else {
            continue;
        };

        let Expr::Pow(target_base, target_exp) = ctx.get(target_expr).clone() else {
            continue;
        };
        if compare_expr(ctx, base, target_base) != Ordering::Equal {
            continue;
        }

        let half = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let expected_exp_raw = ctx.add(Expr::Add(power_exp, half));
        let expected_exp = run_default_simplify(ctx, expected_exp_raw);
        if exprs_match_for_cancellation_leaf(ctx, expected_exp, target_exp)
            || exprs_match_after_default_simplify(ctx, expected_exp, target_exp)
        {
            return true;
        }
    }

    false
}

fn is_negated_additive_noncall_profile_expr(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let Some(stripped) = strip_unit_negation_for_phase_shift(ctx, expr) else {
        return false;
    };
    !expr_contains_any_function_call(ctx, stripped)
        && is_top_level_additive_for_profile(ctx, stripped)
}

fn is_simple_symbolic_multiplicative_noncall_profile_expr(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let expr = strip_unit_negation_for_phase_shift(ctx, expr).unwrap_or(expr);
    !expr_contains_any_function_call(ctx, expr)
        && !is_top_level_additive_for_profile(ctx, expr)
        && expr_contains_symbolic_atom_for_cancellation(ctx, expr)
        && matches!(ctx.get(expr), Expr::Mul(_, _) | Expr::Div(_, _))
}

fn classify_noncall_multiplicative_profile_pair(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> &'static str {
    let difference = ctx.add(Expr::Sub(lhs_core, rhs_core));
    if let Some((_common_factor, residual_expr)) =
        extract_common_multiplicative_residual_sum(ctx, difference)
    {
        let residual_terms = AddView::from_expr(ctx, residual_expr).terms;
        let normalized_terms: Vec<_> = residual_terms
            .iter()
            .copied()
            .map(|(term_expr, term_sign)| normalize_signed_add_term(ctx, term_expr, term_sign).0)
            .collect();
        let atomic_terms = normalized_terms
            .iter()
            .filter(|term| expr_is_atomic_noncall(ctx, **term))
            .count();
        let division_terms = normalized_terms
            .iter()
            .filter(|term| contains_division_like_term(ctx, **term))
            .count();

        if residual_terms.len() == 2 && atomic_terms == 2 {
            return "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.shared_scale_atomic_tail";
        }

        if residual_terms.len() == 2 && atomic_terms == 1 && division_terms == 1 {
            return "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.shared_scale_division_tail";
        }

        return "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.shared_scale_other";
    }

    let lhs_expr = strip_unit_negation_for_phase_shift(ctx, lhs_core).unwrap_or(lhs_core);
    let rhs_expr = strip_unit_negation_for_phase_shift(ctx, rhs_core).unwrap_or(rhs_core);
    if matches_noncall_product_vs_division_shared_numerator_scale_profile_pair(
        ctx, lhs_expr, rhs_expr,
    ) {
        return "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.product_vs_division_shared_numerator_scale";
    }

    let lhs_factors = flatten_mul_chain(ctx, lhs_expr);
    let rhs_factors = flatten_mul_chain(ctx, rhs_expr);
    if lhs_factors.len() == 2
        && rhs_factors.len() == 2
        && lhs_factors
            .iter()
            .all(|factor| expr_is_atomic_noncall(ctx, *factor))
        && rhs_factors
            .iter()
            .all(|factor| expr_is_atomic_noncall(ctx, *factor))
    {
        return "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.cross_atomic_product";
    }

    "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.other"
}

fn matches_noncall_product_vs_division_shared_numerator_scale_profile_pair(
    ctx: &mut cas_ast::Context,
    lhs_expr: cas_ast::ExprId,
    rhs_expr: cas_ast::ExprId,
) -> bool {
    for (product_expr, division_expr) in [(lhs_expr, rhs_expr), (rhs_expr, lhs_expr)] {
        let product_factors = flatten_mul_chain(ctx, product_expr);
        if product_factors.len() != 2
            || !product_factors
                .iter()
                .all(|factor| expr_is_atomic_noncall(ctx, *factor))
        {
            continue;
        }

        let Some((numerator, denominator)) = as_div(ctx, division_expr) else {
            continue;
        };
        if !expr_is_atomic_noncall(ctx, denominator) {
            continue;
        }

        let numerator_factors = flatten_mul_chain(ctx, numerator);
        if numerator_factors.len() != 2
            || !numerator_factors
                .iter()
                .all(|factor| expr_is_atomic_noncall(ctx, *factor))
        {
            continue;
        }

        if product_factors.iter().any(|product_factor| {
            numerator_factors.iter().any(|numerator_factor| {
                compare_expr(ctx, *product_factor, *numerator_factor) == Ordering::Equal
            })
        }) {
            return true;
        }
    }

    false
}

fn classify_power_merge_exponent_profile_kind(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> PowerMergeExponentProfileKind {
    match cas_ast::views::as_rational_const(ctx, expr, 8) {
        Some(value) if value.is_integer() => PowerMergeExponentProfileKind::Integer,
        Some(_) => PowerMergeExponentProfileKind::Fractional,
        None => PowerMergeExponentProfileKind::Symbolic,
    }
}

fn extract_noncall_power_merge_source_profile(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, PowerMergeExponentProfileKind)> {
    let expr = strip_unit_negation_for_phase_shift(ctx, expr).unwrap_or(expr);
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 || factors.len() > 12 {
        return None;
    }

    let mut base = None;
    let mut saw_fractional = false;
    let mut saw_symbolic = false;
    for factor in factors {
        let (factor_base, exponent_kind) = match ctx.get(factor).clone() {
            Expr::Pow(base, exp) => (base, classify_power_merge_exponent_profile_kind(ctx, exp)),
            _ => (factor, PowerMergeExponentProfileKind::Integer),
        };

        if let Some(existing_base) = base {
            if compare_expr(ctx, existing_base, factor_base) != Ordering::Equal {
                return None;
            }
        } else {
            base = Some(factor_base);
        }

        match exponent_kind {
            PowerMergeExponentProfileKind::Integer => {}
            PowerMergeExponentProfileKind::Fractional => saw_fractional = true,
            PowerMergeExponentProfileKind::Symbolic => saw_symbolic = true,
        }
    }

    let exponent_kind = if saw_fractional {
        PowerMergeExponentProfileKind::Fractional
    } else if saw_symbolic {
        PowerMergeExponentProfileKind::Symbolic
    } else {
        PowerMergeExponentProfileKind::Integer
    };

    Some((base?, exponent_kind))
}

fn extract_noncall_power_merge_target_profile(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, PowerMergeExponentProfileKind)> {
    let expr = strip_unit_negation_for_phase_shift(ctx, expr).unwrap_or(expr);
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };

    Some((base, classify_power_merge_exponent_profile_kind(ctx, exp)))
}

fn classify_noncall_power_merge_profile_pair(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<&'static str> {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((source_base, source_kind)) =
            extract_noncall_power_merge_source_profile(ctx, source_expr)
        else {
            continue;
        };
        let Some((target_base, target_kind)) =
            extract_noncall_power_merge_target_profile(ctx, target_expr)
        else {
            continue;
        };

        if compare_expr(ctx, source_base, target_base) != Ordering::Equal {
            continue;
        }

        let has_fractional = matches!(source_kind, PowerMergeExponentProfileKind::Fractional)
            || matches!(target_kind, PowerMergeExponentProfileKind::Fractional);
        if has_fractional {
            return Some(
                "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_power_merge_fractional",
            );
        }

        let has_symbolic = matches!(source_kind, PowerMergeExponentProfileKind::Symbolic)
            || matches!(target_kind, PowerMergeExponentProfileKind::Symbolic);
        if has_symbolic {
            return Some(
                "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_power_merge_symbolic",
            );
        }

        return Some(
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_power_merge_integer",
        );
    }

    None
}

fn default_simplify_other_profile_label(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> &'static str {
    let lhs_has_hyperbolic = expr_contains_hyperbolic_builtin_for_profile(ctx, lhs_core);
    let rhs_has_hyperbolic = expr_contains_hyperbolic_builtin_for_profile(ctx, rhs_core);

    if lhs_has_hyperbolic || rhs_has_hyperbolic {
        let lhs_additive = is_top_level_additive_for_profile(ctx, lhs_core);
        let rhs_additive = is_top_level_additive_for_profile(ctx, rhs_core);

        if lhs_additive ^ rhs_additive {
            return classify_hyperbolic_additive_mismatch_profile_pair(ctx, lhs_core, rhs_core)
                .unwrap_or(
                    "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_additive_mismatch",
                );
        }

        if lhs_has_hyperbolic ^ rhs_has_hyperbolic {
            return classify_hyperbolic_vs_nonhyperbolic_profile_pair(ctx, lhs_core, rhs_core)
                .unwrap_or(
                    "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic",
                );
        }

        if matches_hyperbolic_cosh_cubic_linear_profile_pair(ctx, lhs_core, rhs_core) {
            return "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_pair.cosh_cubic_linear";
        }

        if matches_hyperbolic_cross_swap_profile_pair(ctx, lhs_core, rhs_core) {
            return "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_pair.cross_swap";
        }

        if matches_hyperbolic_square_product_gap_profile_pair(ctx, lhs_core, rhs_core) {
            return "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_pair.square_product_gap";
        }

        return "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_pair";
    }

    let lhs_has_log = expr_contains_log_builtin_for_profile(ctx, lhs_core);
    let rhs_has_log = expr_contains_log_builtin_for_profile(ctx, rhs_core);
    if lhs_has_log || rhs_has_log {
        let lhs_additive = is_top_level_additive_for_profile(ctx, lhs_core);
        let rhs_additive = is_top_level_additive_for_profile(ctx, rhs_core);

        if lhs_additive ^ rhs_additive {
            return "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_additive_mismatch";
        }

        if lhs_has_log ^ rhs_has_log {
            return "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_vs_nonlog";
        }

        return classify_nonhyperbolic_log_pair_profile_label(ctx, lhs_core, rhs_core);
    }

    if matches_signed_double_angle_contraction_profile_pair(ctx, lhs_core, rhs_core) {
        return "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.signed_double_angle";
    }

    let lhs_has_call = expr_contains_any_function_call(ctx, lhs_core);
    let rhs_has_call = expr_contains_any_function_call(ctx, rhs_core);
    if !lhs_has_call && !rhs_has_call {
        let lhs_additive = is_top_level_additive_for_profile(ctx, lhs_core);
        let rhs_additive = is_top_level_additive_for_profile(ctx, rhs_core);

        if expr_is_atomic_noncall(ctx, lhs_core) && expr_is_atomic_noncall(ctx, rhs_core) {
            return "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_atom_pair";
        }

        if lhs_additive ^ rhs_additive {
            return "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_additive_mismatch";
        }

        let lhs_negated_additive = is_negated_additive_noncall_profile_expr(ctx, lhs_core);
        let rhs_negated_additive = is_negated_additive_noncall_profile_expr(ctx, rhs_core);
        if lhs_negated_additive ^ rhs_negated_additive {
            return "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_negated_additive_mismatch";
        }

        let lhs_multiplicative =
            is_simple_symbolic_multiplicative_noncall_profile_expr(ctx, lhs_core);
        let rhs_multiplicative =
            is_simple_symbolic_multiplicative_noncall_profile_expr(ctx, rhs_core);
        if lhs_multiplicative && rhs_multiplicative {
            return classify_noncall_multiplicative_profile_pair(ctx, lhs_core, rhs_core);
        }

        if lhs_multiplicative ^ rhs_multiplicative {
            if let Some(label) = classify_noncall_power_merge_profile_pair(ctx, lhs_core, rhs_core)
            {
                return label;
            }
            return "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_vs_other";
        }

        return "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_other";
    }

    let lhs_has_finite_series =
        expr_contains_named_function_for_profile(ctx, lhs_core, &["sum", "product"]);
    let rhs_has_finite_series =
        expr_contains_named_function_for_profile(ctx, rhs_core, &["sum", "product"]);
    if lhs_has_finite_series || rhs_has_finite_series {
        if lhs_has_finite_series && rhs_has_finite_series {
            return "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_pair";
        }

        return classify_finite_series_vs_other_profile_pair(ctx, lhs_core, rhs_core);
    }

    if matches_shifted_surface_trig_mismatch_profile_pair(ctx, lhs_core, rhs_core) {
        return "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.shifted_surface_trig_mismatch";
    }

    if matches_plain_surface_trig_power_gap_profile_pair(ctx, lhs_core, rhs_core) {
        return "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.surface_trig_power_gap";
    }

    if matches_symbolic_half_power_merge_profile_pair(ctx, lhs_core, rhs_core) {
        return "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.symbolic_half_power_merge";
    }

    let lhs_has_fractional_power = expr_contains_fractional_power_for_profile(ctx, lhs_core);
    let rhs_has_fractional_power = expr_contains_fractional_power_for_profile(ctx, rhs_core);
    if lhs_has_fractional_power || rhs_has_fractional_power {
        let lhs_has_abs = expr_contains_any_builtin(ctx, lhs_core, &[BuiltinFn::Abs]);
        let rhs_has_abs = expr_contains_any_builtin(ctx, rhs_core, &[BuiltinFn::Abs]);

        if lhs_has_abs || rhs_has_abs {
            return "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.fractional_power_abs";
        }

        return "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.fractional_power";
    }

    "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.other"
}

fn classify_hyperbolic_vs_nonhyperbolic_profile_pair(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<&'static str> {
    let one = ctx.num(1);

    for (hyper_expr, other_expr, hyper_first) in
        [(lhs_core, rhs_core, true), (rhs_core, lhs_core, false)]
    {
        if !expr_contains_hyperbolic_builtin_for_profile(ctx, hyper_expr)
            || expr_contains_hyperbolic_builtin_for_profile(ctx, other_expr)
        {
            continue;
        }

        let hyper_negated = strip_unit_negation_for_phase_shift(ctx, hyper_expr).is_some();
        let simple_hyper =
            extract_single_hyperbolic_linear_or_small_power_term_for_reject(ctx, hyper_expr)
                .filter(|(_, hyper_arg, _)| {
                    expr_is_symbolic_leaf_for_hyperbolic_reject(ctx, *hyper_arg)
                });

        if compare_expr(ctx, other_expr, one) == Ordering::Equal {
            return Some(match (hyper_first, simple_hyper, hyper_negated) {
                (true, Some((BuiltinFn::Cosh, _, 2)), false) => {
                    "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_one.cosh_square"
                }
                (true, Some(_), _) => {
                    "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_one.simple_other"
                }
                (true, None, _) => {
                    "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_one.composite"
                }
                (false, Some((BuiltinFn::Tanh, _, 2)), false) => {
                    "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic.tanh_square"
                }
                (false, Some((BuiltinFn::Cosh, _, 1)), true) => {
                    "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic.neg_cosh_linear"
                }
                (false, Some(_), _) => {
                    "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic.simple_other"
                }
                (false, None, _) => {
                    "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic.composite"
                }
            });
        }

        if expr_is_atomic_noncall(ctx, other_expr) {
            return Some(match (hyper_first, simple_hyper, hyper_negated, is_minus_one_expr(ctx, other_expr))
            {
                (true, Some((BuiltinFn::Sinh, _, 2)), false, _) => {
                    "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_atomic_noncall.sinh_square"
                }
                (true, Some(_), _, _) => {
                    "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_atomic_noncall.simple_other"
                }
                (true, None, _, _) => {
                    "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_atomic_noncall.composite"
                }
                (false, Some((BuiltinFn::Cosh, _, 1)), false, true) => {
                    "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.atomic_noncall_vs_hyperbolic.neg_one_cosh_linear"
                }
                (false, Some(_), _, _) => {
                    "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.atomic_noncall_vs_hyperbolic.simple_other"
                }
                (false, None, _, _) => {
                    "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.atomic_noncall_vs_hyperbolic.composite"
                }
            });
        }

        return Some(
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.other",
        );
    }

    None
}

pub(super) fn extract_hyperbolic_square_atomic_tail_profile_pair(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BuiltinFn, cas_ast::ExprId, cas_ast::ExprId, Sign)> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return None;
    }

    let mut hyper_square = None;
    let mut atomic_tail = None;

    for (term, sign) in terms {
        if let Some((builtin, arg, power)) =
            extract_single_hyperbolic_linear_or_small_power_term_for_reject(ctx, term)
        {
            if power == 2 && hyper_square.is_none() {
                hyper_square = Some((builtin, arg));
                continue;
            }
        }

        if expr_is_atomic_noncall(ctx, term) && atomic_tail.is_none() {
            atomic_tail = Some((term, sign));
            continue;
        }

        return None;
    }

    let (builtin, arg) = hyper_square?;
    let (tail_expr, tail_sign) = atomic_tail?;
    Some((builtin, arg, tail_expr, tail_sign))
}

fn extract_additive_sinh_cosh_sum_profile_arg(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 || terms.iter().any(|(_, sign)| *sign != Sign::Pos) {
        return None;
    }

    let first = extract_hyperbolic_linear_term_for_profile(ctx, terms[0].0)?;
    let second = extract_hyperbolic_linear_term_for_profile(ctx, terms[1].0)?;
    if compare_expr(ctx, first.1, second.1) != Ordering::Equal {
        return None;
    }

    let has_sinh = matches!(first.0, BuiltinFn::Sinh) || matches!(second.0, BuiltinFn::Sinh);
    let has_cosh = matches!(first.0, BuiltinFn::Cosh) || matches!(second.0, BuiltinFn::Cosh);
    (has_sinh && has_cosh).then_some(first.1)
}

fn classify_hyperbolic_additive_mismatch_profile_pair(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<&'static str> {
    let one = ctx.num(1);

    for (additive_expr, other_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        if !is_top_level_additive_for_profile(ctx, additive_expr)
            || is_top_level_additive_for_profile(ctx, other_expr)
        {
            continue;
        }

        if let Some((add_builtin, add_arg, tail_expr, tail_sign)) =
            extract_hyperbolic_square_atomic_tail_profile_pair(ctx, additive_expr)
        {
            if let Some((other_builtin, other_arg, other_power)) =
                extract_single_hyperbolic_linear_or_small_power_term_for_reject(ctx, other_expr)
            {
                if other_power == 2
                    && add_builtin != other_builtin
                    && compare_expr(ctx, add_arg, other_arg) == Ordering::Equal
                {
                    if tail_sign == Sign::Neg
                        && compare_expr(ctx, tail_expr, one) == Ordering::Equal
                    {
                        return Some(
                            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_additive_mismatch.conjugate_square_minus_one",
                        );
                    }

                    return Some(
                        "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_additive_mismatch.conjugate_square_atomic_tail",
                    );
                }
            }
        }

        if let Some(sum_arg) = extract_additive_sinh_cosh_sum_profile_arg(ctx, additive_expr) {
            if let Some(exp_arg) = extract_exp_argument(ctx, other_expr) {
                if compare_expr(ctx, sum_arg, exp_arg) == Ordering::Equal {
                    return Some(
                        "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_additive_mismatch.exp_vs_sinh_cosh_sum",
                    );
                }
            }
        }

        return Some(
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_additive_mismatch.other",
        );
    }

    None
}

pub(super) fn direct_core_default_simplify_profile_label(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> &'static str {
    let has_tanh = has_builtin_on_either_side(ctx, lhs_core, rhs_core, BuiltinFn::Tanh);
    let has_cosh = has_builtin_on_either_side(ctx, lhs_core, rhs_core, BuiltinFn::Cosh);
    if has_tanh && has_cosh {
        return "rule.direct_core_equivalence.default_simplify.family.tanh_cosh";
    }

    let has_exp = has_builtin_on_either_side(ctx, lhs_core, rhs_core, BuiltinFn::Exp);
    let has_sinh = has_builtin_on_either_side(ctx, lhs_core, rhs_core, BuiltinFn::Sinh);
    if has_exp && (has_sinh || has_cosh) {
        return "rule.direct_core_equivalence.default_simplify.family.exp_hyperbolic";
    }

    let has_arctan = has_builtin_on_either_side(ctx, lhs_core, rhs_core, BuiltinFn::Arctan);
    if has_arctan {
        return "rule.direct_core_equivalence.default_simplify.family.inverse_trig";
    }

    let has_abs = has_builtin_on_either_side(ctx, lhs_core, rhs_core, BuiltinFn::Abs);
    let has_sqrt = has_builtin_on_either_side(ctx, lhs_core, rhs_core, BuiltinFn::Sqrt);
    if has_abs && has_sqrt {
        return "rule.direct_core_equivalence.default_simplify.family.abs_sqrt";
    }

    let has_tan = has_builtin_on_either_side(ctx, lhs_core, rhs_core, BuiltinFn::Tan);
    let has_cot = has_builtin_on_either_side(ctx, lhs_core, rhs_core, BuiltinFn::Cot);
    if has_div_on_either_side(ctx, lhs_core, rhs_core) && (has_tan || has_cot) {
        return direct_core_trig_ratio_profile_label(ctx, lhs_core, rhs_core);
    }

    if has_div_on_either_side(ctx, lhs_core, rhs_core) {
        return quotient_cancel_profile_label(ctx, lhs_core, rhs_core);
    }

    default_simplify_other_profile_label(ctx, lhs_core, rhs_core)
}

fn direct_core_trig_ratio_profile_label(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> &'static str {
    if try_match_half_angle_tan_equivalence(ctx, lhs_core, rhs_core).is_some()
        || try_match_half_angle_tan_equivalence(ctx, rhs_core, lhs_core).is_some()
    {
        "rule.direct_core_equivalence.default_simplify.family.trig_ratio.half_angle_tan"
    } else {
        "rule.direct_core_equivalence.default_simplify.family.trig_ratio.other"
    }
}

fn scoped_direct_core_default_simplify_profile_label(
    scope: DirectCoreDefaultSimplifyProfileScope,
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> &'static str {
    let base = direct_core_default_simplify_profile_label(ctx, lhs_core, rhs_core);

    match (scope, base) {
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.tanh_cosh",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.tanh_cosh",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.exp_hyperbolic",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.exp_hyperbolic",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.inverse_trig",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.inverse_trig",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.abs_sqrt",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.abs_sqrt",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.trig_ratio.half_angle_tan",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.trig_ratio.half_angle_tan",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.trig_ratio.other",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.trig_ratio.other",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.radical_pair",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.quotient_cancel.radical_pair",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.radical_single",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.quotient_cancel.radical_single",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_pair",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.quotient_cancel.polynomial_pair",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_single",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.quotient_cancel.polynomial_single",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_single.product_denominator_split",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.quotient_cancel.polynomial_single.product_denominator_split",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_single.product_denominator_plus_tail",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.quotient_cancel.polynomial_single.product_denominator_plus_tail",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_single.product_denominator_with_whole_term",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.quotient_cancel.polynomial_single.product_denominator_with_whole_term",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_single.same_denominator_difference",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.quotient_cancel.polynomial_single.same_denominator_difference",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_single.whole_fraction_pair",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.quotient_cancel.polynomial_single.whole_fraction_pair",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_single.nonfraction_target",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.quotient_cancel.polynomial_single.nonfraction_target",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_single.other",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.quotient_cancel.polynomial_single.other",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.monomial_pair",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.quotient_cancel.monomial_pair",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.monomial_single",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.quotient_cancel.monomial_single",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.other_pair",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.quotient_cancel.other_pair",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.other_single",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.quotient_cancel.other_single",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_pair.cosh_cubic_linear",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_pair.cosh_cubic_linear",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_pair.cross_swap",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_pair.cross_swap",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_pair.square_product_gap",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_pair.square_product_gap",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_additive_mismatch.conjugate_square_minus_one",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_additive_mismatch.conjugate_square_minus_one",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_additive_mismatch.conjugate_square_atomic_tail",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_additive_mismatch.conjugate_square_atomic_tail",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_additive_mismatch.exp_vs_sinh_cosh_sum",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_additive_mismatch.exp_vs_sinh_cosh_sum",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_additive_mismatch.other",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_additive_mismatch.other",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_additive_mismatch",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_additive_mismatch",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_pair",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_pair",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_one.cosh_square",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_one.cosh_square",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_one.simple_other",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_one.simple_other",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_one.composite",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_one.composite",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_one",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_one",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic.tanh_square",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic.tanh_square",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic.neg_cosh_linear",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic.neg_cosh_linear",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic.simple_other",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic.simple_other",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic.composite",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic.composite",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_atomic_noncall.sinh_square",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_atomic_noncall.sinh_square",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_atomic_noncall.simple_other",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_atomic_noncall.simple_other",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_atomic_noncall.composite",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_atomic_noncall.composite",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_atomic_noncall",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_atomic_noncall",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.atomic_noncall_vs_hyperbolic.neg_one_cosh_linear",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.atomic_noncall_vs_hyperbolic.neg_one_cosh_linear",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.atomic_noncall_vs_hyperbolic.simple_other",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.atomic_noncall_vs_hyperbolic.simple_other",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.atomic_noncall_vs_hyperbolic.composite",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.atomic_noncall_vs_hyperbolic.composite",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.atomic_noncall_vs_hyperbolic",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.atomic_noncall_vs_hyperbolic",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.other",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.other",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_additive_mismatch",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.log_additive_mismatch",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_vs_nonlog",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.log_vs_nonlog",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_pair",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.log_pair",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.signed_double_angle",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.signed_double_angle",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_atom_pair",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.noncall_atom_pair",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_additive_mismatch",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.noncall_additive_mismatch",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_negated_additive_mismatch",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.noncall_negated_additive_mismatch",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.product_vs_division_shared_numerator_scale",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.product_vs_division_shared_numerator_scale",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.cross_atomic_product",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.cross_atomic_product",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.shared_scale_atomic_tail",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.shared_scale_atomic_tail",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.shared_scale_division_tail",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.shared_scale_division_tail",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.shared_scale_other",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.shared_scale_other",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.other",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.other",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_vs_other",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_vs_other",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_power_merge_integer",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.noncall_power_merge_integer",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_power_merge_symbolic",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.noncall_power_merge_symbolic",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_power_merge_fractional",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.noncall_power_merge_fractional",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_other",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.noncall_other",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_pair",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.finite_series_pair",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.sum_telescoping",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.sum_telescoping",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.sum_direct",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.sum_direct",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.product_evaluable",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.product_evaluable",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.other",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.other",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_vs_other",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.finite_series_vs_other",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.shifted_surface_trig_mismatch",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.shifted_surface_trig_mismatch",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.surface_trig_power_gap",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.surface_trig_power_gap",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.symbolic_half_power_merge",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.symbolic_half_power_merge",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.fractional_power_abs",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.fractional_power_abs",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.fractional_power",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.fractional_power",
        (
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.other",
        ) => "rule.same_denominator_zero.tail_direct_core.default_simplify.family.other.non_hyperbolic.other",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.tanh_cosh",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.tanh_cosh",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.exp_hyperbolic",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.exp_hyperbolic",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.inverse_trig",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.inverse_trig",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.abs_sqrt",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.abs_sqrt",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.trig_ratio.half_angle_tan",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.trig_ratio.half_angle_tan",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.trig_ratio.other",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.trig_ratio.other",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.radical_pair",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.quotient_cancel.radical_pair",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.radical_single",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.quotient_cancel.radical_single",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_pair",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.quotient_cancel.polynomial_pair",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_single",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.quotient_cancel.polynomial_single",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_single.product_denominator_split",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.quotient_cancel.polynomial_single.product_denominator_split",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_single.product_denominator_plus_tail",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.quotient_cancel.polynomial_single.product_denominator_plus_tail",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_single.product_denominator_with_whole_term",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.quotient_cancel.polynomial_single.product_denominator_with_whole_term",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_single.same_denominator_difference",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.quotient_cancel.polynomial_single.same_denominator_difference",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_single.whole_fraction_pair",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.quotient_cancel.polynomial_single.whole_fraction_pair",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_single.nonfraction_target",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.quotient_cancel.polynomial_single.nonfraction_target",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.polynomial_single.other",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.quotient_cancel.polynomial_single.other",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.monomial_pair",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.quotient_cancel.monomial_pair",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.monomial_single",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.quotient_cancel.monomial_single",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.other_pair",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.quotient_cancel.other_pair",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.quotient_cancel.other_single",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.quotient_cancel.other_single",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_pair.cosh_cubic_linear",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_pair.cosh_cubic_linear",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_pair.cross_swap",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_pair.cross_swap",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_pair.square_product_gap",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_pair.square_product_gap",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_additive_mismatch.conjugate_square_minus_one",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_additive_mismatch.conjugate_square_minus_one",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_additive_mismatch.conjugate_square_atomic_tail",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_additive_mismatch.conjugate_square_atomic_tail",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_additive_mismatch.exp_vs_sinh_cosh_sum",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_additive_mismatch.exp_vs_sinh_cosh_sum",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_additive_mismatch.other",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_additive_mismatch.other",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_additive_mismatch",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_additive_mismatch",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_pair",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_pair",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_one.cosh_square",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_one.cosh_square",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_one.simple_other",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_one.simple_other",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_one.composite",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_one.composite",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_one",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_one",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic.tanh_square",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic.tanh_square",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic.neg_cosh_linear",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic.neg_cosh_linear",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic.simple_other",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic.simple_other",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic.composite",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic.composite",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.one_vs_hyperbolic",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_atomic_noncall.sinh_square",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_atomic_noncall.sinh_square",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_atomic_noncall.simple_other",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_atomic_noncall.simple_other",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_atomic_noncall.composite",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_atomic_noncall.composite",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_atomic_noncall",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.hyperbolic_vs_atomic_noncall",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.atomic_noncall_vs_hyperbolic.neg_one_cosh_linear",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.atomic_noncall_vs_hyperbolic.neg_one_cosh_linear",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.atomic_noncall_vs_hyperbolic.simple_other",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.atomic_noncall_vs_hyperbolic.simple_other",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.atomic_noncall_vs_hyperbolic.composite",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.atomic_noncall_vs_hyperbolic.composite",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.atomic_noncall_vs_hyperbolic",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.atomic_noncall_vs_hyperbolic",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.other",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic.other",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.hyperbolic_vs_nonhyperbolic",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.hyperbolic_vs_nonhyperbolic",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_additive_mismatch",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.log_additive_mismatch",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_vs_nonlog",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.log_vs_nonlog",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.log_pair",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.log_pair",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.signed_double_angle",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.signed_double_angle",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_atom_pair",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.noncall_atom_pair",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_additive_mismatch",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.noncall_additive_mismatch",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_negated_additive_mismatch",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.noncall_negated_additive_mismatch",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.product_vs_division_shared_numerator_scale",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.product_vs_division_shared_numerator_scale",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.cross_atomic_product",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.cross_atomic_product",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.shared_scale_atomic_tail",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.shared_scale_atomic_tail",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.shared_scale_division_tail",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.shared_scale_division_tail",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.shared_scale_other",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.shared_scale_other",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.other",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_pair.other",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_vs_other",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.noncall_multiplicative_vs_other",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_power_merge_integer",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.noncall_power_merge_integer",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_power_merge_symbolic",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.noncall_power_merge_symbolic",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_power_merge_fractional",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.noncall_power_merge_fractional",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.noncall_other",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.noncall_other",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_pair",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.finite_series_pair",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.sum_telescoping",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.sum_telescoping",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.sum_direct",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.sum_direct",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.product_evaluable",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.product_evaluable",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.other",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.finite_series_vs_other.other",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.finite_series_vs_other",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.finite_series_vs_other",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.shifted_surface_trig_mismatch",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.shifted_surface_trig_mismatch",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.surface_trig_power_gap",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.surface_trig_power_gap",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.symbolic_half_power_merge",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.symbolic_half_power_merge",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.fractional_power_abs",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.fractional_power_abs",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.fractional_power",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.fractional_power",
        (
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.other",
        ) => "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.other",
        _ => unreachable!(),
    }
}

pub(super) fn profile_same_denominator_tail_direct_core_equivalence_family(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
    sample: Option<String>,
) {
    if !crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled() {
        return;
    }

    let family = classify_direct_core_equivalence_profile_family(ctx, lhs_core, rhs_core);
    let label = match family {
        DirectCoreEquivalenceProfileFamily::DirectMatch => {
            "rule.same_denominator_zero.tail_direct_core.family.direct_match"
        }
        DirectCoreEquivalenceProfileFamily::SymbolicScaleSumLhs => {
            "rule.same_denominator_zero.tail_direct_core.family.symbolic_scale_sum_lhs"
        }
        DirectCoreEquivalenceProfileFamily::SymbolicScaleSumRhs => {
            "rule.same_denominator_zero.tail_direct_core.family.symbolic_scale_sum_rhs"
        }
        DirectCoreEquivalenceProfileFamily::LogExpansion => {
            "rule.same_denominator_zero.tail_direct_core.family.log_expansion"
        }
        DirectCoreEquivalenceProfileFamily::LogChainProduct => {
            "rule.same_denominator_zero.tail_direct_core.family.log_chain_product"
        }
        DirectCoreEquivalenceProfileFamily::TrigReciprocal => {
            "rule.same_denominator_zero.tail_direct_core.family.trig_reciprocal"
        }
        DirectCoreEquivalenceProfileFamily::CosDiffSinDiffQuotient => {
            "rule.same_denominator_zero.tail_direct_core.family.cos_diff_sin_diff_quotient"
        }
        DirectCoreEquivalenceProfileFamily::SumDiffCubesQuotient => {
            "rule.same_denominator_zero.tail_direct_core.family.sum_diff_cubes_quotient"
        }
        DirectCoreEquivalenceProfileFamily::PhaseShiftIdentity => {
            "rule.same_denominator_zero.tail_direct_core.family.phase_shift_identity"
        }
        DirectCoreEquivalenceProfileFamily::CosProductTelescoping => {
            "rule.same_denominator_zero.tail_direct_core.family.cos_product_telescoping"
        }
        DirectCoreEquivalenceProfileFamily::FiniteSum => {
            "rule.same_denominator_zero.tail_direct_core.family.finite_sum"
        }
        DirectCoreEquivalenceProfileFamily::FiniteProduct => {
            "rule.same_denominator_zero.tail_direct_core.family.finite_product"
        }
        DirectCoreEquivalenceProfileFamily::TrigPowerReduction => {
            "rule.same_denominator_zero.tail_direct_core.family.trig_power_reduction"
        }
        DirectCoreEquivalenceProfileFamily::DoubleAngleContraction => {
            "rule.same_denominator_zero.tail_direct_core.family.double_angle_contraction"
        }
        DirectCoreEquivalenceProfileFamily::DefaultSimplify => {
            "rule.same_denominator_zero.tail_direct_core.family.default_simplify"
        }
        DirectCoreEquivalenceProfileFamily::Other => {
            "rule.same_denominator_zero.tail_direct_core.family.other"
        }
    };

    let _ = run_profiled_orchestrator_option_section(label, sample.clone(), || Some(()));
    if matches!(family, DirectCoreEquivalenceProfileFamily::DefaultSimplify) {
        let detail_label = scoped_direct_core_default_simplify_profile_label(
            DirectCoreDefaultSimplifyProfileScope::SameDenominatorTail,
            ctx,
            lhs_core,
            rhs_core,
        );
        let _ = run_profiled_orchestrator_option_section(detail_label, sample.clone(), || Some(()));
    }
}

pub(super) fn profile_shifted_quotient_exact_one_direct_core_equivalence_family(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
    sample: Option<String>,
) {
    if !crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled() {
        return;
    }

    let family = classify_direct_core_equivalence_profile_family(ctx, lhs_core, rhs_core);
    let label = match family {
        DirectCoreEquivalenceProfileFamily::DirectMatch => {
            "rule.shifted_quotient.exact_one.direct_core.family.direct_match"
        }
        DirectCoreEquivalenceProfileFamily::SymbolicScaleSumLhs => {
            match classify_symbolic_scale_sum_profile_detail(ctx, lhs_core) {
                "linear_reciprocal_tail" => {
                    "rule.shifted_quotient.exact_one.direct_core.family.symbolic_scale_sum_lhs.linear_reciprocal_tail"
                }
                "power_reciprocal_tail" => {
                    "rule.shifted_quotient.exact_one.direct_core.family.symbolic_scale_sum_lhs.power_reciprocal_tail"
                }
                "single_scale_plain" => {
                    "rule.shifted_quotient.exact_one.direct_core.family.symbolic_scale_sum_lhs.single_scale_plain"
                }
                "single_scale_other" => {
                    "rule.shifted_quotient.exact_one.direct_core.family.symbolic_scale_sum_lhs.single_scale_other"
                }
                "grouped_multi_scale" => {
                    "rule.shifted_quotient.exact_one.direct_core.family.symbolic_scale_sum_lhs.grouped_multi_scale"
                }
                _ => "rule.shifted_quotient.exact_one.direct_core.family.symbolic_scale_sum_lhs.other",
            }
        }
        DirectCoreEquivalenceProfileFamily::SymbolicScaleSumRhs => {
            match classify_symbolic_scale_sum_profile_detail(ctx, rhs_core) {
                "linear_reciprocal_tail" => {
                    "rule.shifted_quotient.exact_one.direct_core.family.symbolic_scale_sum_rhs.linear_reciprocal_tail"
                }
                "power_reciprocal_tail" => {
                    "rule.shifted_quotient.exact_one.direct_core.family.symbolic_scale_sum_rhs.power_reciprocal_tail"
                }
                "single_scale_plain" => {
                    "rule.shifted_quotient.exact_one.direct_core.family.symbolic_scale_sum_rhs.single_scale_plain"
                }
                "single_scale_other" => {
                    "rule.shifted_quotient.exact_one.direct_core.family.symbolic_scale_sum_rhs.single_scale_other"
                }
                "grouped_multi_scale" => {
                    "rule.shifted_quotient.exact_one.direct_core.family.symbolic_scale_sum_rhs.grouped_multi_scale"
                }
                _ => "rule.shifted_quotient.exact_one.direct_core.family.symbolic_scale_sum_rhs.other",
            }
        }
        DirectCoreEquivalenceProfileFamily::LogExpansion => {
            "rule.shifted_quotient.exact_one.direct_core.family.log_expansion"
        }
        DirectCoreEquivalenceProfileFamily::LogChainProduct => {
            "rule.shifted_quotient.exact_one.direct_core.family.log_chain_product"
        }
        DirectCoreEquivalenceProfileFamily::TrigReciprocal => {
            "rule.shifted_quotient.exact_one.direct_core.family.trig_reciprocal"
        }
        DirectCoreEquivalenceProfileFamily::CosDiffSinDiffQuotient => {
            "rule.shifted_quotient.exact_one.direct_core.family.cos_diff_sin_diff_quotient"
        }
        DirectCoreEquivalenceProfileFamily::SumDiffCubesQuotient => {
            "rule.shifted_quotient.exact_one.direct_core.family.sum_diff_cubes_quotient"
        }
        DirectCoreEquivalenceProfileFamily::PhaseShiftIdentity => {
            "rule.shifted_quotient.exact_one.direct_core.family.phase_shift_identity"
        }
        DirectCoreEquivalenceProfileFamily::CosProductTelescoping => {
            "rule.shifted_quotient.exact_one.direct_core.family.cos_product_telescoping"
        }
        DirectCoreEquivalenceProfileFamily::FiniteSum => {
            "rule.shifted_quotient.exact_one.direct_core.family.finite_sum"
        }
        DirectCoreEquivalenceProfileFamily::FiniteProduct => {
            "rule.shifted_quotient.exact_one.direct_core.family.finite_product"
        }
        DirectCoreEquivalenceProfileFamily::TrigPowerReduction => {
            "rule.shifted_quotient.exact_one.direct_core.family.trig_power_reduction"
        }
        DirectCoreEquivalenceProfileFamily::DoubleAngleContraction => {
            "rule.shifted_quotient.exact_one.direct_core.family.double_angle_contraction"
        }
        DirectCoreEquivalenceProfileFamily::DefaultSimplify => {
            "rule.shifted_quotient.exact_one.direct_core.family.default_simplify"
        }
        DirectCoreEquivalenceProfileFamily::Other => {
            "rule.shifted_quotient.exact_one.direct_core.family.other"
        }
    };

    let _ = run_profiled_orchestrator_option_section(label, sample.clone(), || Some(()));
    if matches!(family, DirectCoreEquivalenceProfileFamily::DefaultSimplify) {
        let detail_label = scoped_direct_core_default_simplify_profile_label(
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            ctx,
            lhs_core,
            rhs_core,
        );
        let _ = run_profiled_orchestrator_option_section(detail_label, sample.clone(), || Some(()));
    }
}

pub(super) fn profile_shifted_quotient_exact_one_rule_apply_pair_family(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
    sample: Option<String>,
) {
    if !crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled() {
        return;
    }

    let family = classify_direct_core_equivalence_profile_family(ctx, lhs_core, rhs_core);
    let label = match family {
        DirectCoreEquivalenceProfileFamily::DirectMatch => "sq1.rule_apply.family.direct_match",
        DirectCoreEquivalenceProfileFamily::SymbolicScaleSumLhs => {
            "sq1.rule_apply.family.symbolic_scale_sum_lhs"
        }
        DirectCoreEquivalenceProfileFamily::SymbolicScaleSumRhs => {
            "sq1.rule_apply.family.symbolic_scale_sum_rhs"
        }
        DirectCoreEquivalenceProfileFamily::LogExpansion => "sq1.rule_apply.family.log_expansion",
        DirectCoreEquivalenceProfileFamily::LogChainProduct => {
            "sq1.rule_apply.family.log_chain_product"
        }
        DirectCoreEquivalenceProfileFamily::TrigReciprocal => {
            "sq1.rule_apply.family.trig_reciprocal"
        }
        DirectCoreEquivalenceProfileFamily::CosDiffSinDiffQuotient => {
            "sq1.rule_apply.family.cos_diff_sin_diff_quotient"
        }
        DirectCoreEquivalenceProfileFamily::SumDiffCubesQuotient => {
            "sq1.rule_apply.family.sum_diff_cubes_quotient"
        }
        DirectCoreEquivalenceProfileFamily::PhaseShiftIdentity => {
            "sq1.rule_apply.family.phase_shift_identity"
        }
        DirectCoreEquivalenceProfileFamily::CosProductTelescoping => {
            "sq1.rule_apply.family.cos_product_telescoping"
        }
        DirectCoreEquivalenceProfileFamily::FiniteSum => "sq1.rule_apply.family.finite_sum",
        DirectCoreEquivalenceProfileFamily::FiniteProduct => "sq1.rule_apply.family.finite_product",
        DirectCoreEquivalenceProfileFamily::TrigPowerReduction => {
            "sq1.rule_apply.family.trig_power_reduction"
        }
        DirectCoreEquivalenceProfileFamily::DoubleAngleContraction => {
            "sq1.rule_apply.family.double_angle_contraction"
        }
        DirectCoreEquivalenceProfileFamily::DefaultSimplify => {
            "sq1.rule_apply.family.default_simplify"
        }
        DirectCoreEquivalenceProfileFamily::Other => "sq1.rule_apply.family.other",
    };

    let _ = run_profiled_orchestrator_option_section(label, sample.clone(), || Some(()));
    match family {
        DirectCoreEquivalenceProfileFamily::SymbolicScaleSumLhs => {
            let detail_label = match classify_symbolic_scale_sum_profile_detail(ctx, lhs_core) {
                "linear_reciprocal_tail" => {
                    "sq1.rule_apply.family.symbolic_scale_sum_lhs.linear_reciprocal_tail"
                }
                "power_reciprocal_tail" => {
                    "sq1.rule_apply.family.symbolic_scale_sum_lhs.power_reciprocal_tail"
                }
                "single_scale_plain" => {
                    "sq1.rule_apply.family.symbolic_scale_sum_lhs.single_scale_plain"
                }
                "single_scale_other" => {
                    "sq1.rule_apply.family.symbolic_scale_sum_lhs.single_scale_other"
                }
                "grouped_multi_scale" => {
                    "sq1.rule_apply.family.symbolic_scale_sum_lhs.grouped_multi_scale"
                }
                _ => "sq1.rule_apply.family.symbolic_scale_sum_lhs.other",
            };
            let _ =
                run_profiled_orchestrator_option_section(detail_label, sample.clone(), || Some(()));
        }
        DirectCoreEquivalenceProfileFamily::SymbolicScaleSumRhs => {
            let detail_label = match classify_symbolic_scale_sum_profile_detail(ctx, rhs_core) {
                "linear_reciprocal_tail" => {
                    "sq1.rule_apply.family.symbolic_scale_sum_rhs.linear_reciprocal_tail"
                }
                "power_reciprocal_tail" => {
                    "sq1.rule_apply.family.symbolic_scale_sum_rhs.power_reciprocal_tail"
                }
                "single_scale_plain" => {
                    "sq1.rule_apply.family.symbolic_scale_sum_rhs.single_scale_plain"
                }
                "single_scale_other" => {
                    "sq1.rule_apply.family.symbolic_scale_sum_rhs.single_scale_other"
                }
                "grouped_multi_scale" => {
                    "sq1.rule_apply.family.symbolic_scale_sum_rhs.grouped_multi_scale"
                }
                _ => "sq1.rule_apply.family.symbolic_scale_sum_rhs.other",
            };
            let _ =
                run_profiled_orchestrator_option_section(detail_label, sample.clone(), || Some(()));
        }
        DirectCoreEquivalenceProfileFamily::DefaultSimplify => {
            let scoped_label = scoped_direct_core_default_simplify_profile_label(
                DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
                ctx,
                lhs_core,
                rhs_core,
            );
            let detail_label = if scoped_label
                .starts_with(
                    "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.quotient_cancel.",
                )
            {
                "sq1.rule_apply.family.default_simplify.detail.quotient_cancel"
            } else if scoped_label.starts_with(
                "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.log",
            ) {
                "sq1.rule_apply.family.default_simplify.detail.log"
            } else if scoped_label == "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.inverse_trig" {
                "sq1.rule_apply.family.default_simplify.detail.inverse_trig"
            } else if scoped_label == "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.abs_sqrt" {
                "sq1.rule_apply.family.default_simplify.detail.abs_sqrt"
            } else if scoped_label.starts_with(
                "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.trig_ratio.",
            ) {
                "sq1.rule_apply.family.default_simplify.detail.trig_ratio"
            } else if scoped_label
                .contains(".hyperbolic")
                || scoped_label
                    == "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.tanh_cosh"
                || scoped_label
                    == "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.exp_hyperbolic"
            {
                "sq1.rule_apply.family.default_simplify.detail.hyperbolic"
            } else if scoped_label.starts_with(
                "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.",
            ) {
                "sq1.rule_apply.family.default_simplify.detail.other_non_hyperbolic"
            } else {
                "sq1.rule_apply.family.default_simplify.detail.other"
            };
            let _ =
                run_profiled_orchestrator_option_section(detail_label, sample.clone(), || Some(()));
        }
        _ => {}
    }
}

pub(super) fn profile_shifted_quotient_exact_one_route_pair_family(
    route_prefix: &'static str,
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
    sample: Option<String>,
) {
    if !crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled() {
        return;
    }

    let family = classify_direct_core_equivalence_profile_family(ctx, lhs_core, rhs_core);
    let label = match family {
        DirectCoreEquivalenceProfileFamily::DirectMatch => {
            "rule.shifted_quotient.exact_one.route.family.direct_match"
        }
        DirectCoreEquivalenceProfileFamily::SymbolicScaleSumLhs => {
            match classify_symbolic_scale_sum_profile_detail(ctx, lhs_core) {
                "linear_reciprocal_tail" => {
                    "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.linear_reciprocal_tail"
                }
                "power_reciprocal_tail" => {
                    "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.power_reciprocal_tail"
                }
                "single_scale_plain" => {
                    "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.single_scale_plain"
                }
                "single_scale_other" => {
                    "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.single_scale_other"
                }
                "grouped_multi_scale" => {
                    "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.grouped_multi_scale"
                }
                _ => "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.other",
            }
        }
        DirectCoreEquivalenceProfileFamily::SymbolicScaleSumRhs => {
            match classify_symbolic_scale_sum_profile_detail(ctx, rhs_core) {
                "linear_reciprocal_tail" => {
                    "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.linear_reciprocal_tail"
                }
                "power_reciprocal_tail" => {
                    "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.power_reciprocal_tail"
                }
                "single_scale_plain" => {
                    "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.single_scale_plain"
                }
                "single_scale_other" => {
                    "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.single_scale_other"
                }
                "grouped_multi_scale" => {
                    "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.grouped_multi_scale"
                }
                _ => "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.other",
            }
        }
        DirectCoreEquivalenceProfileFamily::LogExpansion => {
            "rule.shifted_quotient.exact_one.route.family.log_expansion"
        }
        DirectCoreEquivalenceProfileFamily::LogChainProduct => {
            "rule.shifted_quotient.exact_one.route.family.log_chain_product"
        }
        DirectCoreEquivalenceProfileFamily::DefaultSimplify => {
            let scoped_label = scoped_direct_core_default_simplify_profile_label(
                DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
                ctx,
                lhs_core,
                rhs_core,
            );
            if scoped_label.starts_with(
                "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.quotient_cancel.",
            ) {
                "rule.shifted_quotient.exact_one.route.family.default_simplify.quotient_cancel"
            } else if scoped_label.starts_with(
                "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.other.non_hyperbolic.log",
            ) {
                "rule.shifted_quotient.exact_one.route.family.default_simplify.log"
            } else if scoped_label
                == "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.abs_sqrt"
            {
                "rule.shifted_quotient.exact_one.route.family.default_simplify.abs_sqrt"
            } else if scoped_label
                == "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.inverse_trig"
            {
                "rule.shifted_quotient.exact_one.route.family.default_simplify.inverse_trig"
            } else if scoped_label.contains(".hyperbolic")
                || scoped_label
                    == "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.tanh_cosh"
                || scoped_label
                    == "rule.shifted_quotient.exact_one.direct_core.default_simplify.family.exp_hyperbolic"
            {
                "rule.shifted_quotient.exact_one.route.family.default_simplify.hyperbolic"
            } else {
                "rule.shifted_quotient.exact_one.route.family.default_simplify.other"
            }
        }
        DirectCoreEquivalenceProfileFamily::TrigReciprocal => {
            "rule.shifted_quotient.exact_one.route.family.trig_reciprocal"
        }
        DirectCoreEquivalenceProfileFamily::CosDiffSinDiffQuotient => {
            "rule.shifted_quotient.exact_one.route.family.cos_diff_sin_diff_quotient"
        }
        DirectCoreEquivalenceProfileFamily::SumDiffCubesQuotient => {
            "rule.shifted_quotient.exact_one.route.family.sum_diff_cubes_quotient"
        }
        DirectCoreEquivalenceProfileFamily::PhaseShiftIdentity => {
            "rule.shifted_quotient.exact_one.route.family.phase_shift_identity"
        }
        DirectCoreEquivalenceProfileFamily::CosProductTelescoping => {
            "rule.shifted_quotient.exact_one.route.family.cos_product_telescoping"
        }
        DirectCoreEquivalenceProfileFamily::FiniteSum => {
            "rule.shifted_quotient.exact_one.route.family.finite_sum"
        }
        DirectCoreEquivalenceProfileFamily::FiniteProduct => {
            "rule.shifted_quotient.exact_one.route.family.finite_product"
        }
        DirectCoreEquivalenceProfileFamily::TrigPowerReduction => {
            "rule.shifted_quotient.exact_one.route.family.trig_power_reduction"
        }
        DirectCoreEquivalenceProfileFamily::DoubleAngleContraction => {
            "rule.shifted_quotient.exact_one.route.family.double_angle_contraction"
        }
        DirectCoreEquivalenceProfileFamily::Other => {
            "rule.shifted_quotient.exact_one.route.family.other"
        }
    };

    let full_label = match route_prefix {
        "shared_passthrough_residual" => match label {
            "rule.shifted_quotient.exact_one.route.family.direct_match" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.direct_match"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.linear_reciprocal_tail" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.symbolic_scale_sum_lhs.linear_reciprocal_tail"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.power_reciprocal_tail" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.symbolic_scale_sum_lhs.power_reciprocal_tail"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.single_scale_plain" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.symbolic_scale_sum_lhs.single_scale_plain"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.single_scale_other" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.symbolic_scale_sum_lhs.single_scale_other"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.grouped_multi_scale" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.symbolic_scale_sum_lhs.grouped_multi_scale"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.other" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.symbolic_scale_sum_lhs.other"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.linear_reciprocal_tail" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.symbolic_scale_sum_rhs.linear_reciprocal_tail"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.power_reciprocal_tail" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.symbolic_scale_sum_rhs.power_reciprocal_tail"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.single_scale_plain" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.symbolic_scale_sum_rhs.single_scale_plain"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.single_scale_other" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.symbolic_scale_sum_rhs.single_scale_other"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.grouped_multi_scale" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.symbolic_scale_sum_rhs.grouped_multi_scale"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.other" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.symbolic_scale_sum_rhs.other"
            }
            "rule.shifted_quotient.exact_one.route.family.log_expansion" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.log_expansion"
            }
            "rule.shifted_quotient.exact_one.route.family.log_chain_product" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.log_chain_product"
            }
            "rule.shifted_quotient.exact_one.route.family.default_simplify.quotient_cancel" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.default_simplify.quotient_cancel"
            }
            "rule.shifted_quotient.exact_one.route.family.default_simplify.log" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.default_simplify.log"
            }
            "rule.shifted_quotient.exact_one.route.family.default_simplify.abs_sqrt" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.default_simplify.abs_sqrt"
            }
            "rule.shifted_quotient.exact_one.route.family.default_simplify.inverse_trig" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.default_simplify.inverse_trig"
            }
            "rule.shifted_quotient.exact_one.route.family.default_simplify.hyperbolic" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.default_simplify.hyperbolic"
            }
            "rule.shifted_quotient.exact_one.route.family.default_simplify.other" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.default_simplify.other"
            }
            "rule.shifted_quotient.exact_one.route.family.trig_reciprocal" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.trig_reciprocal"
            }
            "rule.shifted_quotient.exact_one.route.family.cos_diff_sin_diff_quotient" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.cos_diff_sin_diff_quotient"
            }
            "rule.shifted_quotient.exact_one.route.family.sum_diff_cubes_quotient" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.sum_diff_cubes_quotient"
            }
            "rule.shifted_quotient.exact_one.route.family.phase_shift_identity" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.phase_shift_identity"
            }
            "rule.shifted_quotient.exact_one.route.family.cos_product_telescoping" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.cos_product_telescoping"
            }
            "rule.shifted_quotient.exact_one.route.family.finite_sum" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.finite_sum"
            }
            "rule.shifted_quotient.exact_one.route.family.finite_product" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.finite_product"
            }
            "rule.shifted_quotient.exact_one.route.family.trig_power_reduction" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.trig_power_reduction"
            }
            "rule.shifted_quotient.exact_one.route.family.double_angle_contraction" => {
                "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.double_angle_contraction"
            }
            _ => "rule.shifted_quotient.exact_one.route.shared_passthrough_residual.family.other",
        },
        "exact_zero_direct_residual" => match label {
            "rule.shifted_quotient.exact_one.route.family.direct_match" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.direct_match"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.linear_reciprocal_tail" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.symbolic_scale_sum_lhs.linear_reciprocal_tail"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.power_reciprocal_tail" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.symbolic_scale_sum_lhs.power_reciprocal_tail"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.single_scale_plain" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.symbolic_scale_sum_lhs.single_scale_plain"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.single_scale_other" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.symbolic_scale_sum_lhs.single_scale_other"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.grouped_multi_scale" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.symbolic_scale_sum_lhs.grouped_multi_scale"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.other" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.symbolic_scale_sum_lhs.other"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.linear_reciprocal_tail" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.symbolic_scale_sum_rhs.linear_reciprocal_tail"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.power_reciprocal_tail" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.symbolic_scale_sum_rhs.power_reciprocal_tail"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.single_scale_plain" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.symbolic_scale_sum_rhs.single_scale_plain"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.single_scale_other" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.symbolic_scale_sum_rhs.single_scale_other"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.grouped_multi_scale" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.symbolic_scale_sum_rhs.grouped_multi_scale"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.other" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.symbolic_scale_sum_rhs.other"
            }
            "rule.shifted_quotient.exact_one.route.family.log_expansion" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.log_expansion"
            }
            "rule.shifted_quotient.exact_one.route.family.log_chain_product" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.log_chain_product"
            }
            "rule.shifted_quotient.exact_one.route.family.default_simplify.quotient_cancel" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.default_simplify.quotient_cancel"
            }
            "rule.shifted_quotient.exact_one.route.family.default_simplify.log" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.default_simplify.log"
            }
            "rule.shifted_quotient.exact_one.route.family.default_simplify.abs_sqrt" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.default_simplify.abs_sqrt"
            }
            "rule.shifted_quotient.exact_one.route.family.default_simplify.inverse_trig" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.default_simplify.inverse_trig"
            }
            "rule.shifted_quotient.exact_one.route.family.default_simplify.hyperbolic" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.default_simplify.hyperbolic"
            }
            "rule.shifted_quotient.exact_one.route.family.default_simplify.other" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.default_simplify.other"
            }
            "rule.shifted_quotient.exact_one.route.family.trig_reciprocal" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.trig_reciprocal"
            }
            "rule.shifted_quotient.exact_one.route.family.cos_diff_sin_diff_quotient" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.cos_diff_sin_diff_quotient"
            }
            "rule.shifted_quotient.exact_one.route.family.sum_diff_cubes_quotient" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.sum_diff_cubes_quotient"
            }
            "rule.shifted_quotient.exact_one.route.family.phase_shift_identity" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.phase_shift_identity"
            }
            "rule.shifted_quotient.exact_one.route.family.cos_product_telescoping" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.cos_product_telescoping"
            }
            "rule.shifted_quotient.exact_one.route.family.finite_sum" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.finite_sum"
            }
            "rule.shifted_quotient.exact_one.route.family.finite_product" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.finite_product"
            }
            "rule.shifted_quotient.exact_one.route.family.trig_power_reduction" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.trig_power_reduction"
            }
            "rule.shifted_quotient.exact_one.route.family.double_angle_contraction" => {
                "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.double_angle_contraction"
            }
            _ => "rule.shifted_quotient.exact_one.route.exact_zero_direct_residual.family.other",
        },
        "direct_core_equivalence" => match label {
            "rule.shifted_quotient.exact_one.route.family.direct_match" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.direct_match"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.linear_reciprocal_tail" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.symbolic_scale_sum_lhs.linear_reciprocal_tail"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.power_reciprocal_tail" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.symbolic_scale_sum_lhs.power_reciprocal_tail"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.single_scale_plain" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.symbolic_scale_sum_lhs.single_scale_plain"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.single_scale_other" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.symbolic_scale_sum_lhs.single_scale_other"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.grouped_multi_scale" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.symbolic_scale_sum_lhs.grouped_multi_scale"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_lhs.other" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.symbolic_scale_sum_lhs.other"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.linear_reciprocal_tail" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.symbolic_scale_sum_rhs.linear_reciprocal_tail"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.power_reciprocal_tail" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.symbolic_scale_sum_rhs.power_reciprocal_tail"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.single_scale_plain" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.symbolic_scale_sum_rhs.single_scale_plain"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.single_scale_other" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.symbolic_scale_sum_rhs.single_scale_other"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.grouped_multi_scale" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.symbolic_scale_sum_rhs.grouped_multi_scale"
            }
            "rule.shifted_quotient.exact_one.route.family.symbolic_scale_sum_rhs.other" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.symbolic_scale_sum_rhs.other"
            }
            "rule.shifted_quotient.exact_one.route.family.log_expansion" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.log_expansion"
            }
            "rule.shifted_quotient.exact_one.route.family.log_chain_product" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.log_chain_product"
            }
            "rule.shifted_quotient.exact_one.route.family.default_simplify.quotient_cancel" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.default_simplify.quotient_cancel"
            }
            "rule.shifted_quotient.exact_one.route.family.default_simplify.log" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.default_simplify.log"
            }
            "rule.shifted_quotient.exact_one.route.family.default_simplify.abs_sqrt" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.default_simplify.abs_sqrt"
            }
            "rule.shifted_quotient.exact_one.route.family.default_simplify.inverse_trig" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.default_simplify.inverse_trig"
            }
            "rule.shifted_quotient.exact_one.route.family.default_simplify.hyperbolic" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.default_simplify.hyperbolic"
            }
            "rule.shifted_quotient.exact_one.route.family.default_simplify.other" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.default_simplify.other"
            }
            "rule.shifted_quotient.exact_one.route.family.trig_reciprocal" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.trig_reciprocal"
            }
            "rule.shifted_quotient.exact_one.route.family.cos_diff_sin_diff_quotient" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.cos_diff_sin_diff_quotient"
            }
            "rule.shifted_quotient.exact_one.route.family.sum_diff_cubes_quotient" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.sum_diff_cubes_quotient"
            }
            "rule.shifted_quotient.exact_one.route.family.phase_shift_identity" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.phase_shift_identity"
            }
            "rule.shifted_quotient.exact_one.route.family.cos_product_telescoping" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.cos_product_telescoping"
            }
            "rule.shifted_quotient.exact_one.route.family.finite_sum" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.finite_sum"
            }
            "rule.shifted_quotient.exact_one.route.family.finite_product" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.finite_product"
            }
            "rule.shifted_quotient.exact_one.route.family.trig_power_reduction" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.trig_power_reduction"
            }
            "rule.shifted_quotient.exact_one.route.family.double_angle_contraction" => {
                "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.double_angle_contraction"
            }
            _ => "rule.shifted_quotient.exact_one.route.direct_core_equivalence.family.other",
        },
        _ => return,
    };

    let _ = run_profiled_orchestrator_option_section(full_label, sample.clone(), || Some(()));
    if matches!(family, DirectCoreEquivalenceProfileFamily::DefaultSimplify) {
        let scoped_label = scoped_direct_core_default_simplify_profile_label(
            DirectCoreDefaultSimplifyProfileScope::ShiftedQuotientExactOne,
            ctx,
            lhs_core,
            rhs_core,
        );
        let _ = run_profiled_orchestrator_option_section(scoped_label, sample, || Some(()));
    }
}

pub(super) fn profile_shared_passthrough_tail_direct_core_family(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
    sample: Option<String>,
) {
    if !crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled() {
        return;
    }

    let family = classify_direct_core_equivalence_profile_family(ctx, lhs_core, rhs_core);
    let label = match family {
        DirectCoreEquivalenceProfileFamily::DirectMatch => {
            "rule.shared_passthrough.tail_direct_core.family.direct_match"
        }
        DirectCoreEquivalenceProfileFamily::SymbolicScaleSumLhs => {
            "rule.shared_passthrough.tail_direct_core.family.symbolic_scale_sum_lhs"
        }
        DirectCoreEquivalenceProfileFamily::SymbolicScaleSumRhs => {
            "rule.shared_passthrough.tail_direct_core.family.symbolic_scale_sum_rhs"
        }
        DirectCoreEquivalenceProfileFamily::LogExpansion => {
            "rule.shared_passthrough.tail_direct_core.family.log_expansion"
        }
        DirectCoreEquivalenceProfileFamily::LogChainProduct => {
            "rule.shared_passthrough.tail_direct_core.family.log_chain_product"
        }
        DirectCoreEquivalenceProfileFamily::TrigReciprocal => {
            "rule.shared_passthrough.tail_direct_core.family.trig_reciprocal"
        }
        DirectCoreEquivalenceProfileFamily::CosDiffSinDiffQuotient => {
            "rule.shared_passthrough.tail_direct_core.family.cos_diff_sin_diff_quotient"
        }
        DirectCoreEquivalenceProfileFamily::SumDiffCubesQuotient => {
            "rule.shared_passthrough.tail_direct_core.family.sum_diff_cubes_quotient"
        }
        DirectCoreEquivalenceProfileFamily::PhaseShiftIdentity => {
            "rule.shared_passthrough.tail_direct_core.family.phase_shift_identity"
        }
        DirectCoreEquivalenceProfileFamily::CosProductTelescoping => {
            "rule.shared_passthrough.tail_direct_core.family.cos_product_telescoping"
        }
        DirectCoreEquivalenceProfileFamily::FiniteSum => {
            "rule.shared_passthrough.tail_direct_core.family.finite_sum"
        }
        DirectCoreEquivalenceProfileFamily::FiniteProduct => {
            "rule.shared_passthrough.tail_direct_core.family.finite_product"
        }
        DirectCoreEquivalenceProfileFamily::TrigPowerReduction => {
            "rule.shared_passthrough.tail_direct_core.family.trig_power_reduction"
        }
        DirectCoreEquivalenceProfileFamily::DoubleAngleContraction => {
            "rule.shared_passthrough.tail_direct_core.family.double_angle_contraction"
        }
        DirectCoreEquivalenceProfileFamily::DefaultSimplify => {
            "rule.shared_passthrough.tail_direct_core.family.default_simplify"
        }
        DirectCoreEquivalenceProfileFamily::Other => {
            "rule.shared_passthrough.tail_direct_core.family.other"
        }
    };

    let _ = run_profiled_orchestrator_option_section(label, sample.clone(), || Some(()));
    match family {
        DirectCoreEquivalenceProfileFamily::SymbolicScaleSumLhs => {
            let detail_label = match classify_symbolic_scale_sum_profile_detail(ctx, lhs_core) {
                "linear_reciprocal_tail" => {
                    "rule.shared_passthrough.tail_direct_core.family.symbolic_scale_sum_lhs.linear_reciprocal_tail"
                }
                "power_reciprocal_tail" => {
                    "rule.shared_passthrough.tail_direct_core.family.symbolic_scale_sum_lhs.power_reciprocal_tail"
                }
                "single_scale_plain" => {
                    "rule.shared_passthrough.tail_direct_core.family.symbolic_scale_sum_lhs.single_scale_plain"
                }
                "single_scale_other" => {
                    "rule.shared_passthrough.tail_direct_core.family.symbolic_scale_sum_lhs.single_scale_other"
                }
                "grouped_multi_scale" => {
                    "rule.shared_passthrough.tail_direct_core.family.symbolic_scale_sum_lhs.grouped_multi_scale"
                }
                _ => "rule.shared_passthrough.tail_direct_core.family.symbolic_scale_sum_lhs.other",
            };
            let _ =
                run_profiled_orchestrator_option_section(detail_label, sample.clone(), || Some(()));
        }
        DirectCoreEquivalenceProfileFamily::SymbolicScaleSumRhs => {
            let detail_label = match classify_symbolic_scale_sum_profile_detail(ctx, rhs_core) {
                "linear_reciprocal_tail" => {
                    "rule.shared_passthrough.tail_direct_core.family.symbolic_scale_sum_rhs.linear_reciprocal_tail"
                }
                "power_reciprocal_tail" => {
                    "rule.shared_passthrough.tail_direct_core.family.symbolic_scale_sum_rhs.power_reciprocal_tail"
                }
                "single_scale_plain" => {
                    "rule.shared_passthrough.tail_direct_core.family.symbolic_scale_sum_rhs.single_scale_plain"
                }
                "single_scale_other" => {
                    "rule.shared_passthrough.tail_direct_core.family.symbolic_scale_sum_rhs.single_scale_other"
                }
                "grouped_multi_scale" => {
                    "rule.shared_passthrough.tail_direct_core.family.symbolic_scale_sum_rhs.grouped_multi_scale"
                }
                _ => "rule.shared_passthrough.tail_direct_core.family.symbolic_scale_sum_rhs.other",
            };
            let _ =
                run_profiled_orchestrator_option_section(detail_label, sample.clone(), || Some(()));
        }
        DirectCoreEquivalenceProfileFamily::DefaultSimplify => {
            let detail_label = direct_core_default_simplify_profile_label(ctx, lhs_core, rhs_core);
            let _ = run_profiled_orchestrator_option_section(detail_label, sample, || Some(()));
        }
        _ => {}
    }
}

pub(super) fn run_profiled_shifted_quotient_exact_one_route(
    profiling: bool,
    name: &'static str,
    sample: &Option<String>,
    body: impl FnOnce() -> Option<Rewrite>,
) -> Option<Rewrite> {
    if profiling {
        run_profiled_orchestrator_option_section(name, sample.clone(), body)
    } else {
        body()
    }
}

pub(super) fn run_profiled_shared_passthrough_probe(
    profiling: bool,
    name: &'static str,
    sample: &Option<String>,
    body: impl FnOnce() -> Option<Rewrite>,
) -> Option<Rewrite> {
    if profiling {
        run_profiled_orchestrator_option_section(name, sample.clone(), body)
    } else {
        body()
    }
}
