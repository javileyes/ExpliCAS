use std::collections::BTreeSet;

use cas_ast::{BuiltinFn, Expr, ExprId};
use cas_engine::NormalFormGoal;
use cas_math::expr_extract::extract_i64_multiplier_and_base_factors;
use cas_math::expr_nary::build_balanced_mul;
use cas_math::inverse_trig_composition_support::{
    try_plan_inverse_atan_reciprocal_add_expr, try_plan_inverse_trig_composition_expr,
    try_plan_inverse_trig_sum_add_expr,
};
use cas_math::summation_support::{
    try_plan_finite_product_evaluation, try_plan_finite_sum_evaluation,
};

use super::{
    contains_phase_shift_term, detect_factor_out_with_division_target,
    generate_trig_additive_term_bridge_rewrites, generate_trig_bridge_rewrites,
    looks_like_fraction_expanded_target, looks_like_mixed_fraction_target,
    looks_like_telescoping_fraction_target, looks_rationalizable_source,
    matches_exact_hyperbolic_sum_to_product_target, phase_shift_target_match,
    should_try_trig_planner_before_simplify, strong_target_match,
    try_build_combined_fraction_from_fold_add, try_plan_log_exp_power_inverse_target_aware,
    try_rewrite_collect_monomial_target_aware, try_rewrite_combine_like_terms_target_aware,
    try_rewrite_consecutive_factorial_ratio_target_aware,
    try_rewrite_exact_fraction_cancel_target_aware, try_rewrite_expanded_target_aware,
    try_rewrite_exponential_sum_diff_target_aware, try_rewrite_factored_target_aware,
    try_rewrite_fraction_combination_target_aware, try_rewrite_fraction_expansion_target_aware,
    try_rewrite_hyperbolic_exponential_bridge_target_aware,
    try_rewrite_hyperbolic_simplify_target_aware, try_rewrite_integrate_prep_target_aware,
    try_rewrite_log_argument_factorization_target_aware,
    try_rewrite_log_contraction_to_target_aware, try_rewrite_log_expansion_target_aware,
    try_rewrite_nested_fraction_target_aware, try_rewrite_odd_half_power_target_aware,
    try_rewrite_odd_half_power_to_target_aware, try_rewrite_power_merge_target_aware,
    try_rewrite_pythagorean_factor_form_target_aware, try_rewrite_radical_target_aware,
    try_rewrite_shifted_reciprocal_pythagorean_target_aware, try_rewrite_solve_prep_target_aware,
    try_rewrite_trig_contraction_target_aware, try_rewrite_trig_expansion,
    try_rewrite_trig_identity_to_one_target_aware, DeriveTargetForm,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeriveTargetProfile {
    pub(crate) form: DeriveTargetForm,
    pub(crate) shared_vars: Vec<String>,
}

pub(crate) fn classify_target_profile(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> DeriveTargetProfile {
    let shared_vars = collect_candidate_variables(ctx, source_expr, target_expr);

    if looks_like_finite_aggregate_source(ctx, source_expr)
        && detect_finite_aggregate_target(ctx, source_expr, target_expr)
    {
        return DeriveTargetProfile {
            form: DeriveTargetForm::FiniteAggregateEvaluated,
            shared_vars,
        };
    }

    if detect_combined_like_terms_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::LikeTermsCombined,
            shared_vars,
        };
    }

    if detect_factorial_rewritten_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::FactorialRewritten,
            shared_vars,
        };
    }

    if detect_inverse_trig_rewritten_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::InverseTrigRewritten,
            shared_vars,
        };
    }

    if detect_fraction_cancelled_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::FractionCancelled,
            shared_vars,
        };
    }

    if detect_nested_fraction_simplified_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::NestedFractionSimplified,
            shared_vars,
        };
    }

    if detect_radical_rewritten_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::RadicalRewritten,
            shared_vars,
        };
    }

    if let Some(var_name) = detect_factor_with_division_target(ctx, target_expr, &shared_vars) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::FactoredWithDivision { var: var_name },
            shared_vars,
        };
    }

    if let Some(var_name) = detect_collect_target(ctx, source_expr, target_expr, &shared_vars) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::Collected { var: var_name },
            shared_vars,
        };
    }

    if detect_log_expanded_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::LogExpanded,
            shared_vars,
        };
    }

    if detect_integrate_prepared_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::IntegratePrepared,
            shared_vars,
        };
    }

    if detect_finite_aggregate_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::FiniteAggregateEvaluated,
            shared_vars,
        };
    }

    if detect_log_contracted_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::LogContracted,
            shared_vars,
        };
    }

    if detect_exponential_rewritten_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::ExponentialRewritten,
            shared_vars,
        };
    }

    if detect_hyperbolic_product_sum_expanded_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::Expanded,
            shared_vars,
        };
    }

    if detect_hyperbolic_rewritten_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::HyperbolicRewritten,
            shared_vars,
        };
    }

    if detect_trig_rewritten_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::TrigRewritten,
            shared_vars,
        };
    }

    if detect_trig_expanded_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::TrigExpanded,
            shared_vars,
        };
    }

    if detect_trig_contracted_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::TrigContracted,
            shared_vars,
        };
    }

    if detect_fraction_expanded_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::FractionExpanded,
            shared_vars,
        };
    }

    if detect_fraction_decomposed_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::FractionDecomposed,
            shared_vars,
        };
    }

    if detect_fraction_combined_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::FractionCombined,
            shared_vars,
        };
    }

    if detect_power_merged_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::PowerMerged,
            shared_vars,
        };
    }

    if detect_odd_half_power_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::OddHalfPowerExpanded,
            shared_vars,
        };
    }

    if detect_factored_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::Factored,
            shared_vars,
        };
    }

    if detect_solve_prepared_target(ctx, source_expr, target_expr, &shared_vars) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::SolvePrepared,
            shared_vars,
        };
    }

    if detect_rationalized_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::Rationalized,
            shared_vars,
        };
    }

    if detect_expanded_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::Expanded,
            shared_vars,
        };
    }

    if detect_simplified_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::Simplified,
            shared_vars,
        };
    }

    DeriveTargetProfile {
        form: DeriveTargetForm::Unknown,
        shared_vars,
    }
}

fn detect_collect_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
    shared_vars: &[String],
) -> Option<String> {
    for var_name in shared_vars {
        let Some(rewrite) = cas_engine::try_collect_by_var(ctx, source_expr, var_name) else {
            continue;
        };
        if strong_target_match(ctx, rewrite.rewritten, target_expr) {
            return Some(var_name.clone());
        }
    }

    if let Some(rewrite) = try_rewrite_collect_monomial_target_aware(ctx, source_expr, target_expr)
    {
        return Some(rewrite.focus_label);
    }
    None
}

fn detect_factored_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    if try_rewrite_factored_target_aware(ctx, source_expr, target_expr).is_some() {
        return true;
    }

    if !looks_like_factored_target(ctx, target_expr) {
        return false;
    }
    let factored = cas_math::factor::factor(ctx, source_expr);
    factored != source_expr
        && (strong_target_match(ctx, factored, target_expr)
            || simplified_difference_matches_zero(ctx, factored, target_expr))
}

fn detect_log_expanded_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    if try_rewrite_log_argument_factorization_target_aware(ctx, source_expr, target_expr).is_some()
    {
        return true;
    }

    if try_rewrite_log_expansion_target_aware(ctx, source_expr, target_expr).is_some() {
        return true;
    }

    if !looks_log_expandable_source(ctx, source_expr) {
        return false;
    }

    let expanded = run_log_expanded_nf(ctx, source_expr);
    expanded != source_expr && strong_target_match(ctx, expanded, target_expr)
}

fn detect_integrate_prepared_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    try_rewrite_integrate_prep_target_aware(ctx, source_expr, target_expr).is_some()
}

fn detect_solve_prepared_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
    shared_vars: &[String],
) -> bool {
    try_rewrite_solve_prep_target_aware(ctx, source_expr, target_expr, shared_vars).is_some()
}

fn detect_log_contracted_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    if try_rewrite_log_contraction_to_target_aware(ctx, source_expr, target_expr).is_some() {
        return true;
    }

    if !looks_like_log_contracted_target(ctx, source_expr, target_expr) {
        return false;
    }

    let simplified = run_default_simplify(ctx, source_expr);
    if simplified != source_expr {
        if strong_target_match(ctx, simplified, target_expr) {
            return true;
        }

        if try_rewrite_log_contraction_to_target_aware(ctx, simplified, target_expr).is_some() {
            return true;
        }
    }

    false
}

fn detect_trig_expanded_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    if try_rewrite_trig_contraction_target_aware(ctx, source_expr, target_expr).is_some() {
        return false;
    }

    if detect_inverse_trig_double_angle_expanded_target(ctx, source_expr, target_expr) {
        return true;
    }

    if contains_phase_shift_term(ctx, target_expr)
        && generate_trig_additive_term_bridge_rewrites(ctx, source_expr)
            .into_iter()
            .any(|rewrite| {
                rewrite.kind.rule_name() == "Phase Shift Identity"
                    && phase_shift_target_match(ctx, rewrite.rewritten, target_expr)
            })
    {
        return false;
    }

    if let Some(expanded) = run_trig_expand_towards_target(ctx, source_expr, target_expr) {
        if strong_target_match(ctx, expanded, target_expr) {
            return true;
        }
    }

    if should_try_trig_planner_before_simplify(ctx, source_expr, target_expr) {
        return false;
    }

    let simplified = run_default_simplify(ctx, source_expr);
    if simplified == source_expr {
        return false;
    }

    let Some(expanded) = run_trig_expand_towards_target(ctx, simplified, target_expr) else {
        return false;
    };
    strong_target_match(ctx, expanded, target_expr)
}

fn detect_inverse_trig_double_angle_expanded_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    let Some((outer, inverse, x)) = inverse_trig_double_angle_source_parts(ctx, source_expr) else {
        return false;
    };

    let two = ctx.num(2);
    let one = ctx.num(1);
    let x_squared = ctx.add(Expr::Pow(x, two));
    let expected = match (outer, inverse) {
        (BuiltinFn::Sin, BuiltinFn::Arcsin | BuiltinFn::Arccos) => {
            let radicand = ctx.add(Expr::Sub(one, x_squared));
            let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
            build_balanced_mul(ctx, &[two, x, sqrt_radicand])
        }
        (BuiltinFn::Cos, BuiltinFn::Arcsin) => {
            let double_x_squared = build_balanced_mul(ctx, &[two, x_squared]);
            ctx.add(Expr::Sub(one, double_x_squared))
        }
        (BuiltinFn::Cos, BuiltinFn::Arccos) => {
            let double_x_squared = build_balanced_mul(ctx, &[two, x_squared]);
            ctx.add(Expr::Sub(double_x_squared, one))
        }
        _ => return false,
    };

    strong_target_match(ctx, expected, target_expr)
}

fn inverse_trig_double_angle_source_parts(
    ctx: &cas_ast::Context,
    source_expr: ExprId,
) -> Option<(BuiltinFn, BuiltinFn, ExprId)> {
    let Expr::Function(outer_fn, outer_args) = ctx.get(source_expr) else {
        return None;
    };
    if outer_args.len() != 1 {
        return None;
    }
    let outer = ctx.builtin_of(*outer_fn)?;
    if !matches!(outer, BuiltinFn::Sin | BuiltinFn::Cos) {
        return None;
    }

    let (multiple, base_factors) = extract_i64_multiplier_and_base_factors(ctx, outer_args[0]);
    if multiple != 2 {
        return None;
    }
    let base_factors = base_factors.into_vec();
    if base_factors.len() != 1 {
        return None;
    }
    let inverse_call = base_factors[0];

    let Expr::Function(inverse_fn, inverse_args) = ctx.get(inverse_call) else {
        return None;
    };
    if inverse_args.len() != 1 {
        return None;
    }
    let inverse = match ctx.builtin_of(*inverse_fn)? {
        BuiltinFn::Arcsin | BuiltinFn::Asin => BuiltinFn::Arcsin,
        BuiltinFn::Arccos | BuiltinFn::Acos => BuiltinFn::Arccos,
        _ => return None,
    };

    Some((outer, inverse, inverse_args[0]))
}

fn detect_trig_contracted_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    if try_rewrite_trig_contraction_target_aware(ctx, source_expr, target_expr).is_some() {
        return true;
    }

    if generate_trig_bridge_rewrites(ctx, source_expr)
        .into_iter()
        .any(|rewrite| {
            rewrite.kind.rule_name() == "Phase Shift Identity"
                && phase_shift_target_match(ctx, rewrite.rewritten, target_expr)
        })
    {
        return true;
    }

    if generate_trig_additive_term_bridge_rewrites(ctx, source_expr)
        .into_iter()
        .any(|rewrite| {
            rewrite.kind.rule_name() == "Phase Shift Identity"
                && phase_shift_target_match(ctx, rewrite.rewritten, target_expr)
        })
    {
        return true;
    }

    let Some(target_expanded) = run_trig_expand_default(ctx, target_expr) else {
        return false;
    };

    if strong_target_match(ctx, source_expr, target_expanded) {
        return true;
    }

    if should_try_trig_planner_before_simplify(ctx, source_expr, target_expr) {
        return false;
    }

    let simplified = run_default_simplify(ctx, source_expr);
    if simplified == source_expr {
        return false;
    }

    strong_target_match(ctx, simplified, target_expr)
        || try_rewrite_trig_contraction_target_aware(ctx, simplified, target_expr).is_some()
}

fn detect_expanded_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    if !looks_expandable_source(ctx, source_expr) {
        return false;
    }

    if looks_like_hyperbolic_target_family(ctx, source_expr, target_expr)
        && contains_exponential_like(ctx, target_expr)
        && !contains_hyperbolic_fn(ctx, target_expr)
    {
        return false;
    }

    let Some(rewrite) = try_rewrite_expanded_target_aware(ctx, source_expr, target_expr) else {
        return false;
    };

    if cas_math::expr_predicates::contains_division_like_term(ctx, source_expr)
        && !matches!(rewrite.kind, super::ExpandRewriteKind::BinomialPower)
    {
        return false;
    }

    looks_like_expanded_target(ctx, target_expr)
        || matches!(rewrite.kind, super::ExpandRewriteKind::BinomialPower)
}

fn detect_fraction_expanded_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let expanded = try_rewrite_fraction_expansion_target_aware(
        &mut simplifier,
        source_expr,
        target_expr,
        crate::SimplifyOptions {
            suppress_depth_overflow_warnings: true,
            ..crate::SimplifyOptions::default()
        },
    )
    .map(|rewrite| rewrite.rewritten);
    std::mem::swap(&mut simplifier.context, ctx);

    let Some(expanded) = expanded else {
        return false;
    };

    strong_target_match(ctx, expanded, target_expr)
        && (looks_like_fraction_expanded_target(ctx, target_expr)
            || looks_like_telescoping_fraction_target(ctx, target_expr)
            || !matches!(ctx.get(target_expr), cas_ast::Expr::Div(_, _)))
}

fn detect_fraction_decomposed_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    if !matches!(ctx.get(source_expr), cas_ast::Expr::Div(_, _)) {
        return false;
    }

    if !looks_like_mixed_fraction_target(ctx, target_expr) {
        return false;
    }

    let Some(recombined) = try_build_combined_fraction_from_fold_add(ctx, target_expr) else {
        return false;
    };

    strong_target_match(ctx, recombined, source_expr)
        || simplified_difference_matches_zero(ctx, recombined, source_expr)
}

fn detect_fraction_combined_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    try_rewrite_fraction_combination_target_aware(ctx, source_expr, target_expr).is_some()
}

fn detect_odd_half_power_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    if try_rewrite_odd_half_power_to_target_aware(ctx, source_expr, target_expr).is_some() {
        return true;
    }

    if let Some(rewritten) = try_rewrite_odd_half_power_target_aware(ctx, source_expr) {
        if strong_target_match(ctx, rewritten, target_expr) {
            return true;
        }
    }

    let simplified = run_default_simplify(ctx, source_expr);
    if simplified == source_expr {
        return false;
    }

    let Some(rewritten) = try_rewrite_odd_half_power_target_aware(ctx, simplified) else {
        return false;
    };

    strong_target_match(ctx, rewritten, target_expr)
}

fn detect_power_merged_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    if try_rewrite_exponential_sum_diff_target_aware(ctx, source_expr, target_expr).is_some() {
        return false;
    }

    try_rewrite_power_merge_target_aware(ctx, source_expr, target_expr).is_some()
}

fn detect_finite_aggregate_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    if let Some(plan) = try_plan_finite_sum_evaluation(ctx, source_expr, 1000) {
        let simplified = run_default_simplify(ctx, plan.candidate);
        if strong_target_match(ctx, simplified, target_expr)
            || simplified_difference_matches_zero(ctx, simplified, target_expr)
        {
            return true;
        }
    }

    if let Some(plan) = try_plan_finite_product_evaluation(ctx, source_expr, 1000) {
        let simplified = run_default_simplify(ctx, plan.candidate);
        if strong_target_match(ctx, simplified, target_expr)
            || simplified_difference_matches_zero(ctx, simplified, target_expr)
        {
            return true;
        }
    }

    false
}

fn looks_like_finite_aggregate_source(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    let cas_ast::Expr::Function(fn_id, args) = ctx.get(expr) else {
        return false;
    };
    args.len() == 4 && matches!(ctx.sym_name(*fn_id), "sum" | "product")
}

fn detect_exponential_rewritten_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    try_rewrite_exponential_sum_diff_target_aware(ctx, source_expr, target_expr).is_some()
        || try_plan_log_exp_power_inverse_target_aware(ctx, source_expr, target_expr).is_some()
}

fn detect_fraction_cancelled_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    try_rewrite_exact_fraction_cancel_target_aware(ctx, source_expr, target_expr).is_some()
}

fn detect_radical_rewritten_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    try_rewrite_radical_target_aware(ctx, source_expr, target_expr).is_some()
}

fn detect_hyperbolic_rewritten_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    if !looks_like_hyperbolic_target_family(ctx, source_expr, target_expr) {
        return false;
    }

    if contains_exponential_like(ctx, target_expr) && !contains_hyperbolic_fn(ctx, target_expr) {
        return try_rewrite_hyperbolic_exponential_bridge_target_aware(
            ctx,
            source_expr,
            target_expr,
        )
        .is_some();
    }

    try_rewrite_hyperbolic_simplify_target_aware(ctx, source_expr, target_expr).is_some()
}

fn detect_hyperbolic_product_sum_expanded_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    if !looks_like_hyperbolic_target_family(ctx, source_expr, target_expr) {
        return false;
    }

    if contains_exponential_like(ctx, target_expr) && !contains_hyperbolic_fn(ctx, target_expr) {
        return false;
    }

    if matches_exact_hyperbolic_product_sum_expanded_target(ctx, source_expr, target_expr) {
        return true;
    }

    if try_rewrite_hyperbolic_simplify_target_aware(ctx, source_expr, target_expr).is_some() {
        return false;
    }

    let Some(rewrite) = try_rewrite_expanded_target_aware(ctx, source_expr, target_expr) else {
        return false;
    };

    matches!(rewrite.kind, super::ExpandRewriteKind::HyperbolicProductSum)
}

fn matches_exact_hyperbolic_product_sum_expanded_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    matches_exact_hyperbolic_sum_to_product_target(ctx, source_expr, target_expr)
}

fn looks_like_hyperbolic_target_family(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    contains_hyperbolic_fn(ctx, source_expr)
        || contains_hyperbolic_fn(ctx, target_expr)
        || contains_exponential_like(ctx, source_expr)
        || contains_exponential_like(ctx, target_expr)
}

fn contains_hyperbolic_fn(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            Expr::Function(fn_id, args) => {
                if matches!(
                    ctx.builtin_of(*fn_id),
                    Some(
                        BuiltinFn::Sinh
                            | BuiltinFn::Cosh
                            | BuiltinFn::Tanh
                            | BuiltinFn::Asinh
                            | BuiltinFn::Acosh
                            | BuiltinFn::Atanh
                    )
                ) {
                    return true;
                }
                stack.extend(args.iter().copied());
            }
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::Pow(left, right) => {
                stack.push(*left);
                stack.push(*right);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }

    false
}

fn contains_exponential_like(ctx: &mut cas_ast::Context, expr: ExprId) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        if cas_math::expr_extract::extract_exp_argument(ctx, current).is_some() {
            return true;
        }

        match ctx.get(current) {
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::Pow(left, right) => {
                stack.push(*left);
                stack.push(*right);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }

    false
}

fn detect_trig_rewritten_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    try_rewrite_trig_identity_to_one_target_aware(ctx, source_expr, target_expr).is_some()
        || try_rewrite_shifted_reciprocal_pythagorean_target_aware(ctx, source_expr, target_expr)
            .is_some()
        || try_rewrite_pythagorean_factor_form_target_aware(ctx, source_expr, target_expr).is_some()
}

fn detect_factor_with_division_target(
    ctx: &mut cas_ast::Context,
    target_expr: ExprId,
    shared_vars: &[String],
) -> Option<String> {
    detect_factor_out_with_division_target(ctx, target_expr, shared_vars)
}

fn detect_factorial_rewritten_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    try_rewrite_consecutive_factorial_ratio_target_aware(ctx, source_expr, target_expr).is_some()
}

fn detect_combined_like_terms_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    try_rewrite_combine_like_terms_target_aware(ctx, source_expr, target_expr).is_some()
}

fn detect_nested_fraction_simplified_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    try_rewrite_nested_fraction_target_aware(ctx, source_expr, target_expr).is_some()
}

fn detect_inverse_trig_rewritten_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    if let Some(plan) = try_plan_inverse_trig_composition_expr(ctx, source_expr, false, true)
        .or_else(|| try_plan_inverse_trig_composition_expr(ctx, source_expr, false, false))
    {
        if strong_target_match(ctx, plan.rewritten, target_expr) {
            return true;
        }
        let simplified = run_default_simplify(ctx, plan.rewritten);
        if simplified != plan.rewritten && strong_target_match(ctx, simplified, target_expr) {
            return true;
        }
    }

    if let Some(plan) = try_plan_inverse_trig_sum_add_expr(ctx, source_expr) {
        if strong_target_match(ctx, plan.final_result, target_expr) {
            return true;
        }
        let simplified = run_default_simplify(ctx, plan.final_result);
        if simplified != plan.final_result && strong_target_match(ctx, simplified, target_expr) {
            return true;
        }
    }

    let Some(plan) = try_plan_inverse_atan_reciprocal_add_expr(ctx, source_expr, false) else {
        return false;
    };
    if strong_target_match(ctx, plan.final_result, target_expr) {
        return true;
    }
    let simplified = run_default_simplify(ctx, plan.final_result);
    simplified != plan.final_result && strong_target_match(ctx, simplified, target_expr)
}

fn detect_rationalized_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    if !looks_rationalizable_source(ctx, source_expr) {
        return false;
    }

    if denominator_still_has_root_like(ctx, target_expr) {
        return false;
    }

    let simplified = run_default_simplify(ctx, source_expr);
    simplified != source_expr && strong_target_match(ctx, simplified, target_expr)
}

fn detect_simplified_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    let normalized_source = cas_math::canonical_forms::normalize_core(ctx, source_expr);
    normalized_source != source_expr && strong_target_match(ctx, normalized_source, target_expr)
}

fn run_default_simplify(ctx: &mut cas_ast::Context, source_expr: ExprId) -> ExprId {
    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let simplify_options = crate::SimplifyOptions {
        suppress_depth_overflow_warnings: true,
        ..crate::SimplifyOptions::default()
    };
    let (rewritten, _steps, _stats) = simplifier.simplify_with_stats(source_expr, simplify_options);
    std::mem::swap(&mut simplifier.context, ctx);
    rewritten
}

fn simplified_difference_matches_zero(
    ctx: &mut cas_ast::Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    let zero = ctx.num(0);
    let difference = ctx.add(cas_ast::Expr::Sub(left, right));
    let simplified = run_default_simplify(ctx, difference);
    strong_target_match(ctx, simplified, zero)
}

fn run_log_expanded_nf(ctx: &mut cas_ast::Context, source_expr: ExprId) -> ExprId {
    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);

    let expanded = cas_math::logarithm_inverse_support::expand_logs_collect_positive_assumptions(
        &mut simplifier.context,
        source_expr,
    )
    .rewritten;

    let simplify_options = crate::SimplifyOptions {
        collect_steps: false,
        goal: NormalFormGoal::ExpandedLog,
        suppress_depth_overflow_warnings: true,
        ..Default::default()
    };
    let (rewritten, _steps, _stats) = simplifier.simplify_with_stats(expanded, simplify_options);

    std::mem::swap(&mut simplifier.context, ctx);
    rewritten
}

fn run_trig_expand_towards_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<ExprId> {
    if let Some(rewrite) = try_rewrite_trig_expansion(ctx, source_expr, target_expr) {
        return Some(rewrite.rewritten);
    }

    for bridge in generate_trig_bridge_rewrites(ctx, source_expr) {
        if strong_target_match(ctx, bridge.rewritten, target_expr) {
            return Some(bridge.rewritten);
        }

        let Some(rewrite) = try_rewrite_trig_expansion(ctx, bridge.rewritten, target_expr) else {
            continue;
        };
        if strong_target_match(ctx, rewrite.rewritten, target_expr) {
            return Some(rewrite.rewritten);
        }
    }

    for bridge in generate_trig_additive_term_bridge_rewrites(ctx, source_expr) {
        if strong_target_match(ctx, bridge.rewritten, target_expr) {
            return Some(bridge.rewritten);
        }

        let Some(rewrite) = try_rewrite_trig_expansion(ctx, bridge.rewritten, target_expr) else {
            continue;
        };
        if strong_target_match(ctx, rewrite.rewritten, target_expr) {
            return Some(rewrite.rewritten);
        }
    }

    None
}

fn run_trig_expand_default(ctx: &mut cas_ast::Context, source_expr: ExprId) -> Option<ExprId> {
    try_rewrite_trig_expansion(ctx, source_expr, source_expr).map(|rewrite| rewrite.rewritten)
}

fn denominator_still_has_root_like(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    let cas_ast::Expr::Div(_, denominator) = ctx.get(expr) else {
        return false;
    };
    contains_root_like(ctx, *denominator)
}

fn contains_root_like(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        cas_ast::Expr::Pow(_, exp) => {
            matches!(ctx.get(*exp), cas_ast::Expr::Number(n) if !n.is_integer())
        }
        cas_ast::Expr::Function(name, args)
            if (ctx.is_builtin(*name, cas_ast::BuiltinFn::Sqrt)
                || ctx.is_builtin(*name, cas_ast::BuiltinFn::Root))
                && !args.is_empty() =>
        {
            true
        }
        cas_ast::Expr::Add(left, right)
        | cas_ast::Expr::Sub(left, right)
        | cas_ast::Expr::Mul(left, right)
        | cas_ast::Expr::Div(left, right) => {
            contains_root_like(ctx, *left) || contains_root_like(ctx, *right)
        }
        cas_ast::Expr::Neg(inner) | cas_ast::Expr::Hold(inner) => contains_root_like(ctx, *inner),
        cas_ast::Expr::Function(_, args) => args.iter().any(|arg| contains_root_like(ctx, *arg)),
        cas_ast::Expr::Matrix { data, .. } => data.iter().any(|arg| contains_root_like(ctx, *arg)),
        cas_ast::Expr::Number(_)
        | cas_ast::Expr::Constant(_)
        | cas_ast::Expr::Variable(_)
        | cas_ast::Expr::SessionRef(_) => false,
    }
}

fn collect_candidate_variables(
    ctx: &cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Vec<String> {
    let source_vars = cas_ast::collect_variables(ctx, source_expr);
    let target_vars = cas_ast::collect_variables(ctx, target_expr);

    source_vars
        .intersection(&target_vars)
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(crate) fn looks_like_factored_target(ctx: &mut cas_ast::Context, target_expr: ExprId) -> bool {
    if is_factor_power_target(ctx, target_expr) {
        return true;
    }

    if !ctx.is_mul_commutative(target_expr) {
        return false;
    }

    let factors = cas_math::trig_roots_flatten::flatten_mul_chain(ctx, target_expr);
    if factors.len() < 2 {
        return false;
    }

    let mut non_numeric_factors = 0usize;
    let mut has_additive_factor = false;

    for factor in factors {
        if matches!(
            ctx.get(factor),
            cas_ast::Expr::Number(_) | cas_ast::Expr::Constant(_)
        ) {
            continue;
        }

        non_numeric_factors += 1;
        if is_additive_factor_shape(ctx, factor) {
            has_additive_factor = true;
        }
    }

    has_additive_factor && non_numeric_factors >= 2
}

fn is_factor_power_target(ctx: &mut cas_ast::Context, expr: ExprId) -> bool {
    match ctx.get(expr).clone() {
        cas_ast::Expr::Pow(base, exp) => {
            is_additive_factor_shape(ctx, base) && is_positive_integer_exponent(ctx, exp, 2)
        }
        _ => false,
    }
}

fn is_additive_factor_shape(ctx: &mut cas_ast::Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        cas_ast::Expr::Add(_, _) | cas_ast::Expr::Sub(_, _) => true,
        cas_ast::Expr::Neg(inner) => {
            matches!(
                ctx.get(*inner),
                cas_ast::Expr::Add(_, _) | cas_ast::Expr::Sub(_, _)
            )
        }
        cas_ast::Expr::Pow(base, exp) => {
            matches!(
                ctx.get(*base),
                cas_ast::Expr::Add(_, _) | cas_ast::Expr::Sub(_, _)
            ) && is_positive_integer_exponent(ctx, *exp, 2)
        }
        _ => false,
    }
}

fn is_positive_integer_exponent(ctx: &mut cas_ast::Context, expr: ExprId, min_value: i64) -> bool {
    matches!(
        ctx.get(expr),
        cas_ast::Expr::Number(n) if n.is_integer() && n.to_integer() >= min_value.into()
    )
}

fn looks_expandable_source(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            cas_ast::Expr::Mul(l, r) => {
                if matches!(
                    ctx.get(*l),
                    cas_ast::Expr::Add(_, _) | cas_ast::Expr::Sub(_, _)
                ) || matches!(
                    ctx.get(*r),
                    cas_ast::Expr::Add(_, _) | cas_ast::Expr::Sub(_, _)
                ) {
                    return true;
                }
                stack.push(*l);
                stack.push(*r);
            }
            cas_ast::Expr::Div(num, den) => {
                if matches!(
                    ctx.get(*num),
                    cas_ast::Expr::Add(_, _) | cas_ast::Expr::Sub(_, _)
                ) || matches!(
                    ctx.get(*den),
                    cas_ast::Expr::Add(_, _) | cas_ast::Expr::Sub(_, _)
                ) {
                    return true;
                }
                stack.push(*num);
                stack.push(*den);
            }
            cas_ast::Expr::Pow(base, exp) => {
                if matches!(
                    ctx.get(*base),
                    cas_ast::Expr::Add(_, _) | cas_ast::Expr::Sub(_, _) | cas_ast::Expr::Mul(_, _)
                ) {
                    if let cas_ast::Expr::Number(_) = ctx.get(*exp) {
                        return true;
                    }
                }
                stack.push(*base);
                stack.push(*exp);
            }
            cas_ast::Expr::Add(l, r) | cas_ast::Expr::Sub(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            cas_ast::Expr::Neg(inner) | cas_ast::Expr::Hold(inner) => stack.push(*inner),
            cas_ast::Expr::Function(fn_id, args) => {
                let additive_arg = args.first().copied().filter(|arg| {
                    matches!(
                        ctx.get(*arg),
                        cas_ast::Expr::Add(_, _) | cas_ast::Expr::Sub(_, _)
                    )
                });
                if additive_arg.is_some()
                    && matches!(
                        ctx.builtin_of(*fn_id),
                        Some(cas_ast::BuiltinFn::Sinh) | Some(cas_ast::BuiltinFn::Cosh)
                    )
                {
                    return true;
                }
                let multiplicative_arg = args
                    .first()
                    .copied()
                    .filter(|arg| matches!(ctx.get(*arg), cas_ast::Expr::Mul(_, _)));
                if multiplicative_arg.is_some()
                    && matches!(
                        ctx.builtin_of(*fn_id),
                        Some(cas_ast::BuiltinFn::Sinh) | Some(cas_ast::BuiltinFn::Cosh)
                    )
                {
                    return true;
                }
                stack.extend(args.iter().copied());
            }
            cas_ast::Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            cas_ast::Expr::Number(_)
            | cas_ast::Expr::Constant(_)
            | cas_ast::Expr::Variable(_)
            | cas_ast::Expr::SessionRef(_) => {}
        }
    }
    false
}

fn looks_like_expanded_target(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    fn strip_sign(ctx: &cas_ast::Context, mut expr: ExprId) -> ExprId {
        while let cas_ast::Expr::Neg(inner) = ctx.get(expr) {
            expr = *inner;
        }
        expr
    }

    fn collect_terms(ctx: &cas_ast::Context, expr: ExprId, out: &mut Vec<ExprId>) {
        match ctx.get(expr) {
            cas_ast::Expr::Add(l, r) | cas_ast::Expr::Sub(l, r) => {
                collect_terms(ctx, *l, out);
                collect_terms(ctx, *r, out);
            }
            _ => out.push(strip_sign(ctx, expr)),
        }
    }

    if !matches!(
        ctx.get(expr),
        cas_ast::Expr::Add(_, _) | cas_ast::Expr::Sub(_, _)
    ) {
        return false;
    }

    let mut terms = Vec::new();
    collect_terms(ctx, expr, &mut terms);
    if terms.len() < 2 {
        return false;
    }

    if terms.len() >= 3 {
        return true;
    }

    terms.into_iter().any(|term| {
        matches!(
            ctx.get(term),
            cas_ast::Expr::Mul(_, _) | cas_ast::Expr::Pow(_, _)
        )
    })
}

fn looks_log_expandable_source(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            cas_ast::Expr::Function(fn_id, args)
                if matches!(
                    ctx.builtin_of(*fn_id),
                    Some(cas_ast::BuiltinFn::Ln) | Some(cas_ast::BuiltinFn::Log)
                ) =>
            {
                let candidate_arg = match args.as_slice() {
                    [arg] => Some(*arg),
                    [_, arg] => Some(*arg),
                    _ => None,
                };
                if candidate_arg.is_some_and(|arg| {
                    matches!(
                        ctx.get(arg),
                        cas_ast::Expr::Mul(_, _)
                            | cas_ast::Expr::Div(_, _)
                            | cas_ast::Expr::Pow(_, _)
                    )
                }) {
                    return true;
                }
                stack.extend(args.iter().copied());
            }
            cas_ast::Expr::Add(l, r)
            | cas_ast::Expr::Sub(l, r)
            | cas_ast::Expr::Mul(l, r)
            | cas_ast::Expr::Div(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            cas_ast::Expr::Pow(base, exp) => {
                stack.push(*base);
                stack.push(*exp);
            }
            cas_ast::Expr::Neg(inner) | cas_ast::Expr::Hold(inner) => stack.push(*inner),
            cas_ast::Expr::Function(_, args) => stack.extend(args.iter().copied()),
            cas_ast::Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            cas_ast::Expr::Number(_)
            | cas_ast::Expr::Constant(_)
            | cas_ast::Expr::Variable(_)
            | cas_ast::Expr::SessionRef(_) => {}
        }
    }
    false
}

fn looks_like_log_contracted_target(
    ctx: &cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    let source_logs = count_log_calls(ctx, source_expr);
    let target_logs = count_log_calls(ctx, target_expr);
    source_logs >= 2 && target_logs >= 1 && target_logs < source_logs
}

fn count_log_calls(ctx: &cas_ast::Context, expr: ExprId) -> usize {
    let mut count = 0usize;
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            cas_ast::Expr::Function(fn_id, args) => {
                if matches!(
                    ctx.builtin_of(*fn_id),
                    Some(cas_ast::BuiltinFn::Ln) | Some(cas_ast::BuiltinFn::Log)
                ) {
                    count += 1;
                }
                stack.extend(args.iter().copied());
            }
            cas_ast::Expr::Add(l, r)
            | cas_ast::Expr::Sub(l, r)
            | cas_ast::Expr::Mul(l, r)
            | cas_ast::Expr::Div(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            cas_ast::Expr::Pow(base, exp) => {
                stack.push(*base);
                stack.push(*exp);
            }
            cas_ast::Expr::Neg(inner) | cas_ast::Expr::Hold(inner) => stack.push(*inner),
            cas_ast::Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            cas_ast::Expr::Number(_)
            | cas_ast::Expr::Constant(_)
            | cas_ast::Expr::Variable(_)
            | cas_ast::Expr::SessionRef(_) => {}
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::{classify_target_profile, DeriveTargetForm};

    fn classify(source: &str, target: &str) -> super::DeriveTargetProfile {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse(source, &mut ctx).expect("parse source");
        let target = cas_parser::parse(target, &mut ctx).expect("parse target");
        classify_target_profile(&mut ctx, source, target)
    }

    #[test]
    fn classifies_tabulated_collect_targets() {
        let cases = [
            ("a*x + b*x + c", "(a + b)*x + c", &["x"][..]),
            ("a*y + b*y + c", "(a + b)*y + c", &["y"][..]),
            (
                "a*x^2 + b*x + c*x^2 + d*x + e*x^2 + f",
                "(a + c + e)*x^2 + (b + d)*x + f",
                &["x"][..],
            ),
            ("x*y + x*z + w", "x*(y + z) + w", &["x"][..]),
            ("a*x*y + b*x*y + c", "(a + b)*x*y + c", &["x", "y"][..]),
            (
                "a*x*y + b*x*y + c*x*z + d*x*z + e",
                "(a + b)*x*y + (c + d)*x*z + e",
                &["x", "y"][..],
            ),
        ];

        for (source, target, var_fragments) in cases {
            let profile = classify(source, target);
            match profile.form {
                DeriveTargetForm::Collected { var } => {
                    for fragment in var_fragments {
                        assert!(
                            var.contains(fragment),
                            "expected collected focus `{var}` to contain `{fragment}`"
                        );
                    }
                }
                other => panic!("expected collect target, got {other:?}"),
            }
        }
    }

    #[test]
    fn classifies_quadratic_passthrough_collect_target() {
        let profile = classify("x^2 + 2*x + 1", "x*(x + 2) + 1");
        assert_eq!(profile.form, DeriveTargetForm::LikeTermsCombined);
    }

    #[test]
    fn classifies_factor_target() {
        let profile = classify("a^2 - b^2", "(a - b)*(a + b)");
        assert_eq!(profile.form, DeriveTargetForm::Factored);
    }

    #[test]
    fn classifies_additive_passthrough_factor_target() {
        let profile = classify("a+x^2-1", "a+(x-1)*(x+1)");
        assert_eq!(profile.form, DeriveTargetForm::Factored);
    }

    #[test]
    fn classifies_perfect_square_factor_target() {
        let profile = classify("x^2 + 2*x + 1", "(x + 1)^2");
        assert_eq!(profile.form, DeriveTargetForm::Factored);
    }

    #[test]
    fn classifies_negated_perfect_square_factor_target() {
        let profile = classify("x^2 + 2*x + 1", "(-(x + 1))^2");
        assert_eq!(profile.form, DeriveTargetForm::Factored);
    }

    #[test]
    fn classifies_sophie_germain_factor_target() {
        let profile = classify("x^4 + 4*y^4", "(x^2 - 2*x*y + 2*y^2)*(x^2 + 2*x*y + 2*y^2)");
        assert_eq!(profile.form, DeriveTargetForm::Factored);
    }

    #[test]
    fn classifies_sign_distributed_difference_of_squares_target() {
        let profile = classify("x^2 - 1", "(1 - x)*(-x - 1)");
        assert_eq!(profile.form, DeriveTargetForm::Factored);
    }

    type SolvePrepClassifyCase = (&'static str, &'static str);

    fn assert_tabulated_solve_prep_classifies(cases: &[SolvePrepClassifyCase]) {
        for (source, target) in cases {
            let profile = classify(source, target);
            assert_eq!(
                profile.form,
                DeriveTargetForm::SolvePrepared,
                "expected solve-prep classification for `{source}` -> `{target}`"
            );
        }
    }

    #[test]
    fn classifies_tabulated_solve_prep_monic_targets() {
        assert_tabulated_solve_prep_classifies(&[
            ("x^2 + 6*x + 5", "(x+3)^2 - 4"),
            ("x^2 + 2*b*x + c", "(x+b)^2 + c - b^2"),
            ("x^2 + 3*x + 1", "(x+3/2)^2 - 5/4"),
        ]);
    }

    #[test]
    fn classifies_tabulated_solve_prep_symbolic_positive_targets() {
        assert_tabulated_solve_prep_classifies(&[
            ("a*x^2 + b*x + c", "a*(x + b/(2*a))^2 + c - b^2/(4*a)"),
            ("a*y^2 + b*y + c", "a*(y + b/(2*a))^2 + c - b^2/(4*a)"),
        ]);
    }

    #[test]
    fn classifies_tabulated_solve_prep_negative_linear_targets() {
        assert_tabulated_solve_prep_classifies(&[(
            "a*x^2 - b*x + c",
            "a*(x - b/(2*a))^2 + c - b^2/(4*a)",
        )]);
    }

    #[test]
    fn classifies_tabulated_solve_prep_negative_leading_targets() {
        assert_tabulated_solve_prep_classifies(&[("-x^2 + b*x + c", "-(x - b/2)^2 + c + b^2/4")]);
    }

    #[test]
    fn classifies_tabulated_solve_prep_fractional_targets() {
        assert_tabulated_solve_prep_classifies(&[(
            "(a/2)*x^2 + b*x + c",
            "(a/2)*(x + b/a)^2 + c - b^2/(2*a)",
        )]);
    }

    #[test]
    fn classifies_expanded_target() {
        let profile = classify("(x + 1)^2", "x^2 + 2*x + 1");
        assert_eq!(profile.form, DeriveTargetForm::Expanded);
    }

    #[test]
    fn classifies_fractional_binomial_square_as_expanded_target() {
        let profile = classify("(x + 1/2)^2", "x^2 + x + 1/4");
        assert_eq!(profile.form, DeriveTargetForm::Expanded);
    }

    #[test]
    fn classifies_expanded_target_when_binomial_expansion_simplifies_to_single_term() {
        let profile = classify("(a+b)^2 - a^2 - 2*a*b", "b^2");
        assert_eq!(profile.form, DeriveTargetForm::Expanded);
    }

    #[test]
    fn classifies_hyperbolic_expanded_target() {
        let profile = classify("sinh(x+y)", "sinh(x)*cosh(y) + cosh(x)*sinh(y)");
        assert_eq!(profile.form, DeriveTargetForm::Expanded);
    }

    #[test]
    fn classifies_hyperbolic_expanded_difference_target() {
        let profile = classify("sinh(x-y)", "sinh(x)*cosh(y) - sinh(y)*cosh(x)");
        assert_eq!(profile.form, DeriveTargetForm::Expanded);
    }

    #[test]
    fn classifies_recursive_hyperbolic_expanded_target() {
        let profile = classify("sinh(6*x)", "sinh(5*x)*cosh(x) + cosh(5*x)*sinh(x)");
        assert_eq!(profile.form, DeriveTargetForm::Expanded);
    }

    #[test]
    fn classifies_hyperbolic_simplified_double_angle_backward_targets() {
        for (source, target) in [
            ("2*cosh(x)^2 - 1", "cosh(2*x)"),
            ("2*sinh(x)^2 + 1", "cosh(2*x)"),
        ] {
            let profile = classify(source, target);
            assert_eq!(
                profile.form,
                DeriveTargetForm::HyperbolicRewritten,
                "expected hyperbolic rewrite classification for `{source}` -> `{target}`"
            );
        }
    }

    #[test]
    fn classifies_hyperbolic_half_angle_square_targets() {
        for (source, target) in [
            ("cosh(x/2)^2", "(cosh(x)+1)/2"),
            ("sinh(x/2)^2", "(cosh(x)-1)/2"),
        ] {
            let profile = classify(source, target);
            assert_eq!(
                profile.form,
                DeriveTargetForm::HyperbolicRewritten,
                "expected hyperbolic half-angle target classification for `{source}` -> `{target}`"
            );
        }
    }

    #[test]
    fn classifies_hyperbolic_simplified_angle_sum_diff_targets() {
        for (source, target) in [
            ("sinh(x)*cosh(y)+cosh(x)*sinh(y)", "sinh(x+y)"),
            ("cosh(x)*cosh(y)+sinh(x)*sinh(y)", "cosh(x+y)"),
            ("sinh(x)*cosh(y)-cosh(x)*sinh(y)", "sinh(x-y)"),
            ("cosh(x)*cosh(y)-sinh(x)*sinh(y)", "cosh(x-y)"),
        ] {
            let profile = classify(source, target);
            assert_eq!(
                profile.form,
                DeriveTargetForm::HyperbolicRewritten,
                "expected hyperbolic rewrite classification for `{source}` -> `{target}`"
            );
        }
    }

    #[test]
    fn classifies_exact_hyperbolic_sum_to_product_targets_as_expanded() {
        for (source, target) in [
            ("sinh(x)+sinh(y)", "2*sinh((x+y)/2)*cosh((x-y)/2)"),
            ("cosh(x)+cosh(y)", "2*cosh((x+y)/2)*cosh((x-y)/2)"),
            ("cosh(x)-cosh(y)", "2*sinh((x+y)/2)*sinh((x-y)/2)"),
        ] {
            let profile = classify(source, target);
            assert_eq!(
                profile.form,
                DeriveTargetForm::Expanded,
                "expected exact hyperbolic product/sum target to classify as expanded for `{source}` -> `{target}`"
            );
        }
    }

    #[test]
    fn classifies_hyperbolic_product_to_sum_passthrough_target_as_expanded() {
        let profile = classify("2*sinh(2*x)*sinh(x)+a", "4*cosh(x)^3-4*cosh(x)+a");
        assert_eq!(profile.form, DeriveTargetForm::Expanded);
    }

    #[test]
    fn classifies_hyperbolic_exponential_identity_targets() {
        for (source, target) in [
            ("exp(x)+exp(-x)", "2*cosh(x)"),
            ("tanh(x)", "(e^x - e^(-x))/(e^x + e^(-x))"),
        ] {
            let profile = classify(source, target);
            assert_eq!(
                profile.form,
                DeriveTargetForm::HyperbolicRewritten,
                "expected hyperbolic rewrite classification for `{source}` -> `{target}`"
            );
        }
    }

    #[test]
    fn classifies_tabulated_log_expanded_targets() {
        let cases = [
            ("ln(x*y)", "ln(x) + ln(y)"),
            ("ln(x/y)", "ln(x) - ln(y)"),
            ("ln((x*y)/z)", "ln(x) + ln(y) - ln(z)"),
            ("ln((x^2*y)/(z*t))", "2*ln(abs(x)) + ln(y) - ln(z) - ln(t)"),
            ("ln(x^2)", "2*ln(abs(x))"),
            ("log(b, (x*y)/z)", "log(b, x) + log(b, y) - log(b, z)"),
            (
                "log(b, (x^2*y^3)/(z^2*t))",
                "2*log(b, x) + 3*log(b, y) - 2*log(b, z) - log(b, t)",
            ),
            ("log(b, x^3)", "3*log(b, x)"),
            ("ln(x^3*y^2)", "ln(x^3) + ln(y^2)"),
        ];

        for (source, target) in cases {
            let profile = classify(source, target);
            assert_eq!(
                profile.form,
                DeriveTargetForm::LogExpanded,
                "expected log-expanded classification for `{source}` -> `{target}`",
            );
        }
    }

    #[test]
    fn classifies_tabulated_log_contracted_targets() {
        let cases = [
            ("ln(x) + ln(y)", "ln(x*y)"),
            ("ln(x) - ln(y)", "ln(x/y)"),
            ("ln(x) + ln(y) - ln(z)", "ln((x*y)/z)"),
            ("2*ln(abs(x))", "ln(x^2)"),
            ("2*ln(abs(x)) + ln(y) - ln(z) - ln(t)", "ln((x^2*y)/(z*t))"),
            ("3*ln(x) + 2*ln(abs(y))", "ln(x^3*y^2)"),
            ("3*ln(x) - 2*ln(y)", "ln(x^3/y^2)"),
            ("3*log(2, x)", "log(2, x^3)"),
            ("ln(x^2)+ln(y^2)", "ln((x*y)^2)"),
            ("2*ln(abs(x))+2*ln(abs(y))", "2*ln(abs(x*y))"),
            ("log(2, x) - log(2, y)", "log(2, x/y)"),
            ("2*log(b, x)+2*log(b, y)", "log(b, (x*y)^2)"),
            (
                "2*log(b, x) + 3*log(b, y) - 2*log(b, z) - log(b, t)",
                "log(b, (x^2*y^3)/(z^2*t))",
            ),
        ];

        for (source, target) in cases {
            let profile = classify(source, target);
            assert_eq!(
                profile.form,
                DeriveTargetForm::LogContracted,
                "expected log-contracted classification for `{source}` -> `{target}`",
            );
        }
    }

    #[test]
    fn classifies_tabulated_exponential_rewritten_targets() {
        let cases = [
            ("exp(x+y)", "exp(x)*exp(y)"),
            ("exp(x)*exp(y)+a", "exp(x+y)+a"),
        ];

        for (source, target) in cases {
            let profile = classify(source, target);
            assert_eq!(
                profile.form,
                DeriveTargetForm::ExponentialRewritten,
                "expected exponential rewrite classification for `{source}` -> `{target}`",
            );
        }
    }

    #[test]
    fn classifies_tabulated_finite_aggregate_targets() {
        let cases = [
            ("product((k+1)/k, k, 1, n)", "n+1"),
            ("sum(1/(k*(k+1)), k, 1, n)", "1 - 1/(n+1)"),
            (
                "sum(1/((a*k+b+c)*(a*k+b+c+a)), k, m, n)",
                "1/a*(1/(a*m+b+c) - 1/(a*n+a+b+c))",
            ),
        ];

        for (source, target) in cases {
            let profile = classify(source, target);
            assert_eq!(
                profile.form,
                DeriveTargetForm::FiniteAggregateEvaluated,
                "expected finite aggregate classification for `{source}` -> `{target}`",
            );
        }
    }

    #[test]
    fn classifies_tabulated_like_terms_combined_targets() {
        let profile = classify("2*x + 3*x + 0", "5*x");
        assert_eq!(profile.form, DeriveTargetForm::LikeTermsCombined);
    }

    #[test]
    fn classifies_tabulated_factorial_rewritten_targets() {
        for (source, target) in [("(n+1)!/n!", "n+1"), ("(n+1)!/n!+a", "n+1+a")] {
            let profile = classify(source, target);
            assert_eq!(profile.form, DeriveTargetForm::FactorialRewritten);
        }
    }

    #[test]
    fn classifies_tabulated_inverse_trig_rewritten_targets() {
        for (source, target) in [
            ("arctan(a) + arctan(1/a)", "pi/2"),
            ("sin(arcsin(x))", "x"),
            ("cos(arccos(x))", "x"),
            ("tan(arctan(x))", "x"),
            ("asin(x/sqrt(x^2 + 1))", "arctan(x)"),
            ("arcsin(x) + arccos(x)", "pi/2"),
            ("sin(arctan(x))", "x/sqrt(1+x^2)"),
            ("cos(arctan(x))", "1/sqrt(1+x^2)"),
            ("cos(arcsin(x))", "sqrt(1-x^2)"),
            ("sin(arccos(x))", "sqrt(1-x^2)"),
            ("tan(arcsin(x))", "x/sqrt(1-x^2)"),
        ] {
            let profile = classify(source, target);
            assert_eq!(profile.form, DeriveTargetForm::InverseTrigRewritten);
        }
    }

    #[test]
    fn classifies_tabulated_fraction_cancelled_targets() {
        for (source, target) in [
            ("(a^3-b^3)/(a-b)", "a^2+a*b+b^2"),
            ("(a^2-b^2)/(a-b)+c", "a+b+c"),
            ("(a^3-b^3)/(a-b)+c", "a^2+a*b+b^2+c"),
        ] {
            let profile = classify(source, target);
            assert_eq!(profile.form, DeriveTargetForm::FractionCancelled);
        }
    }

    #[test]
    fn classifies_tabulated_nested_fraction_targets() {
        for (source, target) in [
            ("1/(1/a)", "a"),
            ("1/(1/a + 1/b)", "(a*b)/(a+b)"),
            ("a + 1/(1/x + 1/y)", "a + (x*y)/(x+y)"),
            ("a - 1/(1/x + 1/y)", "a - (x*y)/(x+y)"),
        ] {
            let profile = classify(source, target);
            assert_eq!(profile.form, DeriveTargetForm::NestedFractionSimplified);
        }
    }

    #[test]
    fn classifies_tabulated_radical_rewritten_targets() {
        let profile = classify("sqrt(a^2 + 2*a*b + b^2)", "abs(a+b)");
        assert_eq!(profile.form, DeriveTargetForm::RadicalRewritten);
    }

    #[test]
    fn classifies_direct_sqrt_squared_symbol_as_radical_rewritten_target() {
        let profile = classify("sqrt(x^2)", "abs(x)");
        assert_eq!(profile.form, DeriveTargetForm::RadicalRewritten);
    }

    #[test]
    fn classifies_square_of_square_root_as_radical_rewritten_target() {
        let profile = classify("sqrt(x)^2", "x");
        assert_eq!(profile.form, DeriveTargetForm::RadicalRewritten);
    }

    #[test]
    fn classifies_tabulated_radical_rewritten_targets_with_passthrough() {
        let profile = classify("sqrt(a^2 + 2*a*b + b^2)+c", "abs(a+b)+c");
        assert_eq!(profile.form, DeriveTargetForm::RadicalRewritten);
    }

    #[test]
    fn classifies_tabulated_trig_expanded_targets() {
        let cases = [
            ("sin(2*x)", "2*sin(x)*cos(x)"),
            ("sin(2*arcsin(x))", "2*x*sqrt(1-x^2)"),
            ("cos(2*arcsin(x))", "1-2*x^2"),
            ("sin(2*arccos(x))", "2*x*sqrt(1-x^2)"),
            ("cos(2*arccos(x))", "2*x^2-1"),
            ("sec(x)", "1/cos(x)"),
            ("tan(x)", "(1-cos(2*x))/sin(2*x)"),
            ("tan(x/2)", "sin(x)/(1+cos(x))"),
            ("tan(x/2)", "(1-cos(x))/sin(x)"),
            ("sin(2*a*x)", "2*sin(a*x)*cos(a*x)"),
            ("sin(4*x)", "4*sin(x)*cos(x)^3 - 4*sin(x)^3*cos(x)"),
            ("cos(4*x)", "8*cos(x)^4 - 8*cos(x)^2 + 1"),
            ("sin(x)^4", "(3-4*cos(2*x)+cos(4*x))/8"),
            ("cos(x)^4", "(3+4*cos(2*x)+cos(4*x))/8"),
            ("sin(x)^2*cos(x)^2", "(1-cos(4*x))/8"),
            ("sin(x)+sin(y)", "2*sin((x+y)/2)*cos((x-y)/2)"),
            ("cos(x)+cos(y)", "2*cos((x+y)/2)*cos((x-y)/2)"),
            ("cos(x)-cos(y)", "-2*sin((x+y)/2)*sin((x-y)/2)"),
            ("2*sin(2*x)*sin(x)", "4*cos(x)-4*cos(x)^3"),
            ("2*cos(2*x)*cos(x)", "4*cos(x)^3-2*cos(x)"),
            ("2*cos(2*x)*sin(x)", "4*cos(x)^2*sin(x)-2*sin(x)"),
            ("2*sin(2*x)*sin(x)+a", "4*cos(x)-4*cos(x)^3+a"),
            ("2*cos(2*x)*sin(x)+a", "4*cos(x)^2*sin(x)-2*sin(x)+a"),
            ("sqrt(2)*sin(x+pi/4)+a", "sin(x)+cos(x)+a"),
            ("4*sin(x+pi/3)", "2*sin(x)+2*sqrt(3)*cos(x)"),
            ("2*sin(x+pi/6)", "sqrt(3)*sin(x)+cos(x)"),
        ];

        for (source, target) in cases {
            let profile = classify(source, target);
            assert_eq!(
                profile.form,
                DeriveTargetForm::TrigExpanded,
                "expected trig-expanded classification for `{source}` -> `{target}`",
            );
        }
    }

    #[test]
    fn classifies_tabulated_trig_rewritten_targets() {
        let cases = [
            ("1 - sin(x)^2", "cos(x)^2"),
            ("tan(x)*cot(x)", "1"),
            ("tan(x)*cot(x)+a", "1+a"),
            ("sec(x)^2 - 1", "tan(x)^2"),
        ];

        for (source, target) in cases {
            let profile = classify(source, target);
            assert_eq!(
                profile.form,
                DeriveTargetForm::TrigRewritten,
                "expected trig-rewritten classification for `{source}` -> `{target}`",
            );
        }
    }

    #[test]
    fn classifies_tabulated_trig_contracted_targets() {
        let cases = [
            ("(sin(2*x))/(cos(2*x))", "tan(2*x)"),
            ("x*sin(x^2)/cos(x^2)", "x*tan(x^2)"),
            ("1+x*sin(x^2)/cos(x^2)", "1+x*tan(x^2)"),
            ("1/cos(x)", "sec(x)"),
            ("1 + tan(x)^2", "sec(x)^2"),
            ("2*sin(a*x)*cos(a*x)", "sin(2*a*x)"),
            ("sin(x)^2*cos(x)^2", "sin(2*x)^2/4"),
            ("4*sin(x)*cos(x)^3 - 4*sin(x)^3*cos(x)", "sin(4*x)"),
            ("(1-cos(2*a*x))/sin(2*a*x)", "tan(a*x)"),
            ("sin(x)+cos(x)", "sqrt(2)*sin(x+pi/4)"),
        ];

        for (source, target) in cases {
            let profile = classify(source, target);
            assert_eq!(
                profile.form,
                DeriveTargetForm::TrigContracted,
                "expected trig-contracted classification for `{source}` -> `{target}`",
            );
        }
    }

    type RationalizedClassifyCase = (&'static str, &'static str);

    fn assert_rationalized_classification(cases: &[RationalizedClassifyCase]) {
        for (source, target) in cases {
            let profile = classify(source, target);
            assert_eq!(
                profile.form,
                DeriveTargetForm::Rationalized,
                "expected rationalized classification for `{source}` -> `{target}`"
            );
        }
    }

    #[test]
    fn classifies_tabulated_rationalized_numeric_targets() {
        assert_rationalized_classification(&[
            ("1/(sqrt(x)-1)", "(sqrt(x)+1)/(x-1)"),
            ("1/(sqrt(x)-2)", "(sqrt(x)+2)/(x-4)"),
        ]);
    }

    #[test]
    fn classifies_representative_rationalized_zero_target() {
        assert_rationalized_classification(&[("1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1)", "0")]);
    }

    #[test]
    fn classifies_tabulated_fraction_expanded_targets() {
        let cases = [
            ("(a*y+b*x)/(x*y)", "a/x + b/y"),
            ("1/(n*(n+2))", "1/2*(1/n - 1/(n+2))"),
            ("1/(n*(n-2))", "1/2*(1/(n-2) - 1/n)"),
            ("1/((2*n+1)*(2*n+3))", "1/2*(1/(2*n+1) - 1/(2*n+3))"),
            ("1/((a*n+b)*(a*n+c))", "1/(c-b)*(1/(a*n+b) - 1/(a*n+c))"),
            ("1/(x^2-a^2)", "1/(2*a)*(1/(x-a) - 1/(x+a))"),
        ];

        for (source, target) in cases {
            let profile = classify(source, target);
            assert_eq!(
                profile.form,
                DeriveTargetForm::FractionExpanded,
                "expected fraction expansion classification for `{source}` -> `{target}`",
            );
        }
    }

    #[test]
    fn classifies_tabulated_fraction_decomposed_targets() {
        let cases = [
            ("(x+1)/(x-1)", "1 + 2/(x-1)"),
            ("(a*x+b)/(x+c)", "a + (b-a*c)/(x+c)"),
            ("(x+a)/(x+b)", "1 + (a-b)/(x+b)"),
            ("(a*x+b)/(c*x+d)", "a/c + (b-a*d/c)/(c*x+d)"),
            ("(x+a)/(c*x+d)", "1/c + (a-d/c)/(c*x+d)"),
            ("(a*x+b)/(d-c*x)", "-a/c + (b+a*d/c)/(d-c*x)"),
            ("(x+a)/(d-c*x)", "-1/c + (a+d/c)/(d-c*x)"),
        ];

        for (source, target) in cases {
            let profile = classify(source, target);
            assert_eq!(
                profile.form,
                DeriveTargetForm::FractionDecomposed,
                "expected fraction decomposition classification for `{source}` -> `{target}`",
            );
        }
    }

    #[test]
    fn classifies_tabulated_fraction_combined_targets() {
        let cases = [
            ("1 + 2/(x-1)", "(x+1)/(x-1)"),
            ("a/c + (b-a*d/c)/(c*x+d)", "(a*x+b)/(c*x+d)"),
            ("1/c + (a-d/c)/(c*x+d)", "(x+a)/(c*x+d)"),
            ("-a/c + (b+a*d/c)/(d-c*x)", "(a*x+b)/(d-c*x)"),
            ("-1/c + (a+d/c)/(d-c*x)", "(x+a)/(d-c*x)"),
        ];

        for (source, target) in cases {
            let profile = classify(source, target);
            assert_eq!(
                profile.form,
                DeriveTargetForm::FractionCombined,
                "expected fraction combination classification for `{source}` -> `{target}`",
            );
        }
    }

    #[test]
    fn classifies_tabulated_odd_half_power_targets() {
        let direct_cases = [
            ("x^(3/2)", "abs(x)*sqrt(x)"),
            ("x^(5/2)", "abs(x)^2*sqrt(x)"),
            ("x^(11/2)", "abs(x)^5*sqrt(x)"),
            ("y^(13/2)", "abs(y)^6*sqrt(y)"),
        ];
        for (source, target) in direct_cases {
            let profile = classify(source, target);
            assert_eq!(
                profile.form,
                DeriveTargetForm::OddHalfPowerExpanded,
                "expected odd-half-power target for {source} -> {target}"
            );
        }

        let simplify_cases = [
            ("sqrt(x^3)", "abs(x)*sqrt(x)"),
            ("sqrt(x^3)+a", "abs(x)*sqrt(x)+a"),
            ("sqrt(x^5)", "abs(x)^2*sqrt(x)"),
            ("sqrt(y^9)", "abs(y)^4*sqrt(y)"),
        ];
        for (source, target) in simplify_cases {
            let profile = classify(source, target);
            assert_eq!(
                profile.form,
                DeriveTargetForm::OddHalfPowerExpanded,
                "expected odd-half-power target after simplify for {source} -> {target}"
            );
        }
    }

    #[test]
    fn classifies_tabulated_power_merged_targets() {
        let cases = [
            ("x^(3/4)*x^(1/4)", "x"),
            ("x*x^(1/3)", "x^(4/3)"),
            ("x^a*x^b", "x^(a+b)"),
            ("x*x^a", "x^(a+1)"),
            ("sqrt(x)*x^a", "x^(a+1/2)"),
            ("x^a*x^b*x^c*x^d", "x^(a+b+c+d)"),
        ];

        for (source, target) in cases {
            let profile = classify(source, target);
            assert_eq!(
                profile.form,
                DeriveTargetForm::PowerMerged,
                "expected power-merged target for {source} -> {target}"
            );
        }
    }

    #[test]
    fn does_not_classify_simple_product_as_factored_target() {
        let profile = classify("x + x", "2*x");
        assert_eq!(profile.form, DeriveTargetForm::LikeTermsCombined);
    }

    #[test]
    fn does_not_classify_plain_simplification_as_expanded_target() {
        let profile = classify("x + x", "2*x");
        assert_ne!(profile.form, DeriveTargetForm::Expanded);
    }

    #[test]
    fn does_not_classify_fraction_cancellation_as_expanded_target() {
        let profile = classify("(a^2 - b^2)/(a - b)", "a + b");
        assert_ne!(profile.form, DeriveTargetForm::Expanded);
    }

    #[test]
    fn classifies_tabulated_factored_with_division_targets() {
        let cases = [
            ("a*x + b*x + c", "x*(a + b + c/x)", "x"),
            (
                "a*x^4 + b*x^3 + c*x^2 + d",
                "x^2*(a*x^2 + b*x + c + d/x^2)",
                "x",
            ),
        ];

        for (source, target, var) in cases {
            let profile = classify(source, target);
            assert_eq!(
                profile.form,
                DeriveTargetForm::FactoredWithDivision {
                    var: var.to_string()
                },
                "expected factor-with-division target for {source} -> {target}"
            );
        }
    }
}
