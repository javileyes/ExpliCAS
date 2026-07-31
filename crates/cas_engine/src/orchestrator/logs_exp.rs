//! Orquestador: familia `logs_exp` (troceo P1).
//!
//! Ver la cabecera de `orchestrator.rs` para el contexto.

use super::*;

pub(super) fn is_hot_log_split_zero_side_root(ctx: &mut Context, expr: ExprId) -> bool {
    let flags = scan_hot_direct_small_zero_family_flags_root(ctx, expr);
    matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
        && AddView::from_expr(ctx, expr).terms.len() == 3
        && flags.has_log
        && !flags.has_division
        && !flags.has_trig
        && !flags.has_hyperbolic
        && (matches_direct_log_product_contract_zero_identity_root(ctx, expr)
            || matches_direct_log_square_product_split_zero_identity_root(ctx, expr)
            || matches_direct_ln_abs_product_split_zero_identity_root(ctx, expr)
            || matches_direct_log_difference_squares_split_zero_identity_root(ctx, expr))
}

pub(super) fn matches_direct_small_zero_log_split_division_hot_pair_root(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> bool {
    for (log_side, other_side) in [(lhs, rhs), (rhs, lhs)] {
        if !is_hot_log_split_zero_side_root(ctx, log_side)
            || !expr_contains_division_node_local(ctx, other_side)
            || expr_contains_log_builtin_local(ctx, other_side)
            || expr_contains_trig_builtin_local(ctx, other_side)
            || expr_contains_hyperbolic_builtin_local(ctx, other_side)
        {
            continue;
        }

        if matches_direct_consecutive_telescoping_fraction_zero_identity_root(ctx, other_side) {
            return true;
        }

        if extract_small_quotient_cancel_zero_candidate_root(ctx, other_side).is_some() {
            return true;
        }

        if has_sum_diff_cubes_quotient_term_root(ctx, other_side)
            && matches_direct_sum_diff_cubes_quotient_zero_identity_root(ctx, other_side)
        {
            return true;
        }
    }

    false
}

pub(super) fn is_nested_additive_log_residual_pair_root(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> bool {
    let residual_difference = ctx.add(Expr::Sub(lhs, rhs));
    is_nested_additive_pair_root(ctx, residual_difference)
        && expr_contains_any_builtin_local(
            ctx,
            residual_difference,
            &[
                BuiltinFn::Ln,
                BuiltinFn::Log,
                BuiltinFn::Log10,
                BuiltinFn::Abs,
            ],
        )
}

pub(super) fn try_hidden_solve_root_exp_ln_shortcut(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let rewrite = try_rewrite_exponential_log_inverse_expr(ctx, expr)?;
    if is_symbolic_atom(ctx, rewrite.rewritten) {
        Some(rewrite.rewritten)
    } else {
        None
    }
}

fn extract_direct_exponential_product_pair_args_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }
    let lhs = extract_exp_argument(ctx, factors[0])?;
    let rhs = extract_exp_argument(ctx, factors[1])?;
    Some(sort_direct_pair_args_root(ctx, lhs, rhs))
}

fn extract_direct_exponential_sum_pair_args_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let sum_expr = extract_exp_argument(ctx, expr)?;
    let view = AddView::from_expr(ctx, sum_expr);
    if view.terms.len() != 2 {
        return None;
    }
    let mut args = Vec::with_capacity(2);
    for (term_expr, term_sign) in view.terms {
        if term_sign != Sign::Pos {
            return None;
        }
        args.push(term_expr);
    }
    Some(sort_direct_pair_args_root(ctx, args[0], args[1]))
}

pub(super) fn matches_direct_exponential_combination_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (product_expr, combined_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((prod_lhs, prod_rhs)) =
            extract_direct_exponential_product_pair_args_root(ctx, product_expr)
        else {
            continue;
        };
        let Some((sum_lhs, sum_rhs)) =
            extract_direct_exponential_sum_pair_args_root(ctx, combined_expr)
        else {
            continue;
        };
        if compare_expr(ctx, prod_lhs, sum_lhs) == Ordering::Equal
            && compare_expr(ctx, prod_rhs, sum_rhs) == Ordering::Equal
        {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_log_square_product_split_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    for candidate_index in 0..view.terms.len() {
        let (candidate_expr, candidate_sign) = view.terms[candidate_index];
        let Some((candidate_base_opt, candidate_arg)) =
            cas_math::expr_extract::extract_log_base_argument_view(ctx, candidate_expr)
        else {
            continue;
        };
        let Some(candidate_product_base) = extract_square_power_base_root(ctx, candidate_arg)
        else {
            continue;
        };
        let Some((factor_a, factor_b)) = extract_mul_pair_root(ctx, candidate_product_base) else {
            continue;
        };

        let mut saw_factor_a = false;
        let mut saw_factor_b = false;
        let mut ok = true;

        for (other_index, (other_expr, other_sign)) in view.terms.iter().copied().enumerate() {
            if other_index == candidate_index {
                continue;
            }
            if other_sign == candidate_sign {
                ok = false;
                break;
            }

            let Some(other_base) =
                extract_log_square_split_factor_arg_root(ctx, other_expr, candidate_base_opt)
            else {
                ok = false;
                break;
            };

            if !saw_factor_a && compare_expr(ctx, other_base, factor_a) == Ordering::Equal {
                saw_factor_a = true;
                continue;
            }
            if !saw_factor_b && compare_expr(ctx, other_base, factor_b) == Ordering::Equal {
                saw_factor_b = true;
                continue;
            }

            ok = false;
            break;
        }

        if ok && saw_factor_a && saw_factor_b {
            return true;
        }
    }

    false
}

fn extract_log_square_split_factor_arg_root(
    ctx: &mut Context,
    expr: ExprId,
    expected_base_opt: Option<ExprId>,
) -> Option<ExprId> {
    if let Some((base_opt, arg)) = cas_math::expr_extract::extract_log_base_argument_view(ctx, expr)
    {
        if base_opt == expected_base_opt {
            if let Some(base) = extract_square_power_base_root(ctx, arg) {
                return Some(base);
            }
        }
    }

    let (coeff, base_expr) = extract_coef_and_base(ctx, expr);
    if coeff != BigRational::from_integer(2.into()) {
        return None;
    }

    let (base_opt, arg) = cas_math::expr_extract::extract_log_base_argument_view(ctx, base_expr)?;
    (base_opt == expected_base_opt).then_some(arg)
}

pub(super) fn matches_direct_log_product_contract_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    for candidate_index in 0..view.terms.len() {
        let (candidate_expr, candidate_sign) = view.terms[candidate_index];
        let Some((candidate_base_opt, candidate_arg)) =
            cas_math::expr_extract::extract_log_base_argument_view(ctx, candidate_expr)
        else {
            continue;
        };
        let Some((factor_a, factor_b)) = extract_mul_pair_root(ctx, candidate_arg) else {
            continue;
        };

        let mut saw_factor_a = false;
        let mut saw_factor_b = false;
        let mut ok = true;

        for (other_index, (other_expr, other_sign)) in view.terms.iter().copied().enumerate() {
            if other_index == candidate_index {
                continue;
            }
            if other_sign == candidate_sign {
                ok = false;
                break;
            }

            let Some((other_base_opt, other_arg)) =
                cas_math::expr_extract::extract_log_base_argument_view(ctx, other_expr)
            else {
                ok = false;
                break;
            };
            if other_base_opt != candidate_base_opt {
                ok = false;
                break;
            }

            if !saw_factor_a && compare_expr(ctx, other_arg, factor_a) == Ordering::Equal {
                saw_factor_a = true;
                continue;
            }
            if !saw_factor_b && compare_expr(ctx, other_arg, factor_b) == Ordering::Equal {
                saw_factor_b = true;
                continue;
            }

            ok = false;
            break;
        }

        if ok && saw_factor_a && saw_factor_b {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_log_difference_squares_split_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    for candidate_index in 0..view.terms.len() {
        let (candidate_expr, candidate_sign) = view.terms[candidate_index];
        let Some((candidate_base_opt, candidate_arg)) =
            cas_math::expr_extract::extract_log_base_argument_view(ctx, candidate_expr)
        else {
            continue;
        };
        let Some((factor_a, factor_b)) =
            extract_difference_of_square_bases_root(ctx, candidate_arg)
                .map(|(positive_base, negative_base)| {
                    (
                        ctx.add(Expr::Sub(positive_base, negative_base)),
                        ctx.add(Expr::Add(positive_base, negative_base)),
                    )
                })
                .or_else(|| {
                    let factored_arg =
                        cas_math::factor::factor_difference_squares(ctx, candidate_arg)?;
                    extract_mul_pair_root(ctx, factored_arg)
                })
        else {
            continue;
        };

        let mut saw_factor_a = false;
        let mut saw_factor_b = false;
        let mut ok = true;

        for (other_index, (other_expr, other_sign)) in view.terms.iter().copied().enumerate() {
            if other_index == candidate_index {
                continue;
            }
            if other_sign == candidate_sign {
                ok = false;
                break;
            }

            let Some((other_base_opt, other_arg)) =
                cas_math::expr_extract::extract_log_base_argument_view(ctx, other_expr)
            else {
                ok = false;
                break;
            };
            if other_base_opt != candidate_base_opt {
                ok = false;
                break;
            }

            if !saw_factor_a && compare_expr(ctx, other_arg, factor_a) == Ordering::Equal {
                saw_factor_a = true;
                continue;
            }
            if !saw_factor_b && compare_expr(ctx, other_arg, factor_b) == Ordering::Equal {
                saw_factor_b = true;
                continue;
            }

            ok = false;
            break;
        }

        if ok && saw_factor_a && saw_factor_b {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_ln_abs_product_split_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    // REAL-ONLY: `ln|X·Y| ≡ ln|X| + ln|Y|` and `ln(X²) ≡ 2·ln|X|` are
    // real-domain identities (over ℂ, ln(i²) = iπ ≠ 0 = 2·ln|i|); ambient
    // axis, same rationale as the sqrt-abs matcher (ficha S4-001).
    if crate::rules::arithmetic::ambient_pipeline_value_domain()
        != crate::semantics::ValueDomain::RealOnly
    {
        return false;
    }
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    for candidate_index in 0..view.terms.len() {
        let (candidate_expr, candidate_sign) = view.terms[candidate_index];
        let (candidate_coeff, candidate_base) = extract_coef_and_base(ctx, candidate_expr);
        if candidate_coeff != BigRational::from_integer(2.into()) {
            continue;
        }
        let Some((candidate_base_opt, candidate_log_arg)) =
            cas_math::expr_extract::extract_log_base_argument_view(ctx, candidate_base)
        else {
            continue;
        };
        if candidate_base_opt.is_some() {
            continue;
        }
        let Some(candidate_abs_arg) =
            cas_math::expr_extract::extract_abs_argument_view(ctx, candidate_log_arg)
        else {
            continue;
        };
        let Some((factor_a, factor_b)) = extract_mul_pair_root(ctx, candidate_abs_arg) else {
            continue;
        };

        let mut saw_factor_a = false;
        let mut saw_factor_b = false;
        let mut ok = true;

        for (other_index, (other_expr, other_sign)) in view.terms.iter().copied().enumerate() {
            if other_index == candidate_index {
                continue;
            }
            if other_sign == candidate_sign {
                ok = false;
                break;
            }

            let (other_coeff, other_base) = extract_coef_and_base(ctx, other_expr);
            if other_coeff != BigRational::from_integer(2.into()) {
                ok = false;
                break;
            }

            let Some((other_base_opt, other_log_arg)) =
                cas_math::expr_extract::extract_log_base_argument_view(ctx, other_base)
            else {
                ok = false;
                break;
            };
            if other_base_opt.is_some() {
                ok = false;
                break;
            }
            let Some(other_abs_arg) =
                cas_math::expr_extract::extract_abs_argument_view(ctx, other_log_arg)
            else {
                ok = false;
                break;
            };

            if !saw_factor_a && compare_expr(ctx, other_abs_arg, factor_a) == Ordering::Equal {
                saw_factor_a = true;
                continue;
            }
            if !saw_factor_b && compare_expr(ctx, other_abs_arg, factor_b) == Ordering::Equal {
                saw_factor_b = true;
                continue;
            }

            ok = false;
            break;
        }

        if ok && saw_factor_a && saw_factor_b {
            return true;
        }
    }

    false
}

fn matches_direct_log_cancellation_zero_identity_root(ctx: &mut Context, expr: ExprId) -> bool {
    matches_direct_log_product_contract_zero_identity_root(ctx, expr)
        || matches_direct_log_square_product_split_zero_identity_root(ctx, expr)
        || matches_direct_log_difference_squares_split_zero_identity_root(ctx, expr)
        || matches_direct_ln_abs_product_split_zero_identity_root(ctx, expr)
}

pub(super) fn contains_direct_log_cancellation_zero_group_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    if matches_direct_log_cancellation_zero_identity_root(ctx, expr) {
        return true;
    }

    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return false;
    }

    let terms = AddView::from_expr(ctx, expr).terms;
    if !(4..=11).contains(&terms.len()) {
        return false;
    }

    for first_index in 0..terms.len().saturating_sub(2) {
        for second_index in (first_index + 1)..terms.len().saturating_sub(1) {
            for third_index in (second_index + 1)..terms.len() {
                let group_terms = [terms[first_index], terms[second_index], terms[third_index]];
                let group_expr = build_signed_sum_expr_root(ctx, &group_terms);
                if matches_direct_log_cancellation_zero_identity_root(ctx, group_expr) {
                    return true;
                }
            }
        }
    }

    false
}

pub(super) fn rewrite_small_exp_product_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let e = ctx.add(Expr::Constant(Constant::E));
    for (left, right) in [(factors[0], factors[1]), (factors[1], factors[0])] {
        if compare_expr(ctx, left, e) != Ordering::Equal {
            continue;
        }
        let exp_arg = extract_exp_argument(ctx, right)?;
        let one = ctx.num(1);
        let shifted_arg = ctx.add(Expr::Add(exp_arg, one));
        return Some(ctx.call_builtin(BuiltinFn::Exp, vec![shifted_arg]));
    }

    None
}

pub(super) fn try_hidden_solve_root_log_power_base_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    domain_mode: crate::DomainMode,
    value_domain: crate::ValueDomain,
) -> Option<(ExprId, Vec<Step>)> {
    use crate::{ImplicitCondition, Proof};

    let planned = try_rewrite_log_power_base_numeric_expr(ctx, expr)?;
    let mode = log_exp_inverse_policy_mode_from_flags(
        matches!(domain_mode, crate::DomainMode::Assume),
        matches!(domain_mode, crate::DomainMode::Strict),
    );
    let one = ctx.num(1);
    let base_positive_proven = if domain_mode.is_generic() {
        matches!(
            prove_positive_literal_fast(ctx, planned.base_core),
            Some(Proof::Proven)
        )
    } else {
        crate::prove_positive(ctx, planned.base_core, value_domain) == Proof::Proven
    };
    let policy = plan_log_power_base_numeric_policy(
        mode,
        value_domain == crate::ValueDomain::ComplexEnabled,
        base_positive_proven,
        cas_ast::ordering::compare_expr(ctx, planned.base_core, one) == std::cmp::Ordering::Equal,
    );

    let cas_math::logarithm_inverse_support::LogPowerBasePolicyPlan::Rewrite {
        require_positive_base,
        require_nonzero_base_minus_one: _,
    } = policy
    else {
        return None;
    };

    if !require_positive_base {
        return Some((planned.rewritten, Vec::new()));
    }

    let mut step = Step::new_compact(
        "log(a^m, a^n) = n/m",
        "Log Power Base",
        expr,
        planned.rewritten,
    );
    step.soundness = crate::SoundnessLabel::EquivalenceUnderIntroducedRequires;
    {
        let meta = step.meta_mut();
        if require_positive_base {
            meta.required_conditions
                .push(ImplicitCondition::Positive(planned.base_core));
        }
    }

    Some((planned.rewritten, vec![step]))
}
