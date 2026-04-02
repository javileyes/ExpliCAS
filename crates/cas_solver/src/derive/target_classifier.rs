use std::collections::BTreeSet;

use cas_ast::ExprId;
use cas_engine::NormalFormGoal;

use super::{
    detect_factor_out_with_division_target, looks_like_fraction_expanded_target,
    looks_like_mixed_fraction_target, looks_like_telescoping_fraction_target,
    looks_rationalizable_source, strong_target_match, try_build_combined_fraction_from_fold_add,
    try_rewrite_expanded_target_aware, try_rewrite_fraction_combination_target_aware,
    try_rewrite_fraction_expansion_target_aware, try_rewrite_integrate_prep_target_aware,
    try_rewrite_log_contraction_target_aware, try_rewrite_log_expansion_target_aware,
    try_rewrite_odd_half_power_target_aware, try_rewrite_power_merge_target_aware,
    try_rewrite_solve_prep_target_aware, try_rewrite_trig_contraction_target_aware,
    try_rewrite_trig_expansion, DeriveTargetForm,
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

    if detect_log_contracted_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::LogContracted,
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

    if detect_rationalized_target(ctx, source_expr, target_expr) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::Rationalized,
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

    if let Some(var_name) = detect_factor_with_division_target(ctx, target_expr, &shared_vars) {
        return DeriveTargetProfile {
            form: DeriveTargetForm::FactoredWithDivision { var: var_name },
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
    None
}

fn detect_factored_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
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
    if !looks_log_expandable_source(ctx, source_expr) {
        return false;
    }

    if try_rewrite_log_expansion_target_aware(ctx, source_expr, target_expr).is_some() {
        return true;
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
    if !looks_like_log_contracted_target(ctx, source_expr, target_expr) {
        return false;
    }

    if let Some(rewritten) = try_rewrite_log_contraction_target_aware(ctx, source_expr) {
        if strong_target_match(ctx, rewritten, target_expr) {
            return true;
        }
    }

    let simplified = run_default_simplify(ctx, source_expr);
    if simplified != source_expr {
        if strong_target_match(ctx, simplified, target_expr) {
            return true;
        }

        if let Some(rewritten) = try_rewrite_log_contraction_target_aware(ctx, simplified) {
            if strong_target_match(ctx, rewritten, target_expr) {
                return true;
            }
        }
    }

    false
}

fn detect_trig_expanded_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    if let Some(expanded) = run_trig_expand_towards_target(ctx, source_expr, target_expr) {
        if strong_target_match(ctx, expanded, target_expr) {
            return true;
        }
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

fn detect_trig_contracted_target(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    if try_rewrite_trig_contraction_target_aware(ctx, source_expr, target_expr).is_some() {
        return true;
    }

    let Some(target_expanded) = run_trig_expand_default(ctx, target_expr) else {
        return false;
    };

    if strong_target_match(ctx, source_expr, target_expanded) {
        return true;
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
    if cas_math::expr_predicates::contains_division_like_term(ctx, source_expr)
        || !looks_expandable_source(ctx, source_expr)
        || !looks_like_expanded_target(ctx, target_expr)
    {
        return false;
    }

    try_rewrite_expanded_target_aware(ctx, source_expr, target_expr).is_some()
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
        crate::SimplifyOptions::default(),
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
    if !looks_like_mixed_fraction_target(ctx, target_expr) {
        return false;
    }

    let Some(recombined) = try_build_combined_fraction_from_fold_add(ctx, target_expr) else {
        return false;
    };

    strong_target_match(ctx, recombined, source_expr)
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
    try_rewrite_power_merge_target_aware(ctx, source_expr, target_expr).is_some()
}

fn detect_factor_with_division_target(
    ctx: &mut cas_ast::Context,
    target_expr: ExprId,
    shared_vars: &[String],
) -> Option<String> {
    detect_factor_out_with_division_target(ctx, target_expr, shared_vars)
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
    let (rewritten, _steps, _stats) =
        simplifier.simplify_with_stats(source_expr, crate::SimplifyOptions::default());
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
    try_rewrite_trig_expansion(ctx, source_expr, target_expr).map(|rewrite| rewrite.rewritten)
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

fn looks_like_factored_target(ctx: &mut cas_ast::Context, target_expr: ExprId) -> bool {
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
                        cas_ast::Expr::Mul(_, _) | cas_ast::Expr::Div(_, _)
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
    fn classifies_collect_target() {
        let profile = classify("a*x + b*x + c", "(a + b)*x + c");
        assert_eq!(
            profile.form,
            DeriveTargetForm::Collected {
                var: "x".to_string()
            }
        );
    }

    #[test]
    fn classifies_factor_target() {
        let profile = classify("x^2 - 1", "(x - 1)*(x + 1)");
        assert_eq!(profile.form, DeriveTargetForm::Factored);
    }

    #[test]
    fn classifies_perfect_square_factor_target() {
        let profile = classify("x^2 + 2*x + 1", "(x + 1)^2");
        assert_eq!(profile.form, DeriveTargetForm::Factored);
    }

    #[test]
    fn classifies_sophie_germain_factor_target() {
        let profile = classify("x^4 + 4*y^4", "(x^2 - 2*x*y + 2*y^2)*(x^2 + 2*x*y + 2*y^2)");
        assert_eq!(profile.form, DeriveTargetForm::Factored);
    }

    #[test]
    fn classifies_complete_square_target_as_solve_prep() {
        let profile = classify("x^2 + 6*x + 5", "(x+3)^2 - 4");
        assert_eq!(profile.form, DeriveTargetForm::SolvePrepared);
    }

    #[test]
    fn classifies_expanded_target() {
        let profile = classify("(x + 1)^2", "x^2 + 2*x + 1");
        assert_eq!(profile.form, DeriveTargetForm::Expanded);
    }

    #[test]
    fn classifies_log_expanded_target() {
        let profile = classify("ln(x^2*y)", "ln(y) + 2*ln(abs(x))");
        assert_eq!(profile.form, DeriveTargetForm::LogExpanded);
    }

    #[test]
    fn classifies_log_contracted_target() {
        let profile = classify("ln(x) + ln(y)", "ln(x*y)");
        assert_eq!(profile.form, DeriveTargetForm::LogContracted);
    }

    #[test]
    fn classifies_log_contracted_target_with_powers() {
        let profile = classify("ln(x^3) + ln(y^2)", "ln(x^3*y^2)");
        assert_eq!(profile.form, DeriveTargetForm::LogContracted);
    }

    #[test]
    fn classifies_trig_expanded_target() {
        let profile = classify("sin(2*x)", "2*sin(x)*cos(x)");
        assert_eq!(profile.form, DeriveTargetForm::TrigExpanded);
    }

    #[test]
    fn classifies_trig_contracted_target() {
        let profile = classify("(sin(2*x))/(cos(2*x))", "tan(2*x)");
        assert_eq!(profile.form, DeriveTargetForm::TrigContracted);
    }

    #[test]
    fn classifies_rationalized_target() {
        let profile = classify("1/(sqrt(x)-1)", "(sqrt(x)+1)/(x-1)");
        assert_eq!(profile.form, DeriveTargetForm::Rationalized);
    }

    #[test]
    fn classifies_fraction_expanded_target() {
        let profile = classify("(x+y)/(x*y)", "1/x + 1/y");
        assert_eq!(profile.form, DeriveTargetForm::FractionExpanded);
    }

    #[test]
    fn classifies_scaled_telescoping_fraction_expanded_target() {
        let profile = classify("1/(n*(n+2))", "1/2*(1/n - 1/(n+2))");
        assert_eq!(profile.form, DeriveTargetForm::FractionExpanded);
    }

    #[test]
    fn classifies_shifted_scaled_telescoping_fraction_expanded_target() {
        let profile = classify("1/(n*(n-2))", "1/2*(1/(n-2) - 1/n)");
        assert_eq!(profile.form, DeriveTargetForm::FractionExpanded);
    }

    #[test]
    fn classifies_affine_telescoping_fraction_expanded_target() {
        let profile = classify("1/((2*n+1)*(2*n+3))", "1/2*(1/(2*n+1) - 1/(2*n+3))");
        assert_eq!(profile.form, DeriveTargetForm::FractionExpanded);
    }

    #[test]
    fn classifies_affine_shifted_telescoping_fraction_expanded_target() {
        let profile = classify("1/((2*n-1)*(2*n+1))", "1/2*(1/(2*n-1) - 1/(2*n+1))");
        assert_eq!(profile.form, DeriveTargetForm::FractionExpanded);
    }

    #[test]
    fn classifies_affine_coeff_three_telescoping_fraction_expanded_target() {
        let profile = classify("1/((3*n+2)*(3*n+5))", "1/3*(1/(3*n+2) - 1/(3*n+5))");
        assert_eq!(profile.form, DeriveTargetForm::FractionExpanded);
    }

    #[test]
    fn classifies_affine_coeff_three_shifted_telescoping_fraction_expanded_target() {
        let profile = classify("1/((3*n-1)*(3*n+2))", "1/3*(1/(3*n-1) - 1/(3*n+2))");
        assert_eq!(profile.form, DeriveTargetForm::FractionExpanded);
    }

    #[test]
    fn classifies_affine_symbolic_coeff_telescoping_fraction_expanded_target() {
        let profile = classify("1/((a*n+2)*(a*n+5))", "1/3*(1/(a*n+2) - 1/(a*n+5))");
        assert_eq!(profile.form, DeriveTargetForm::FractionExpanded);
    }

    #[test]
    fn classifies_affine_symbolic_coeff_shifted_telescoping_fraction_expanded_target() {
        let profile = classify("1/((a*n-1)*(a*n+2))", "1/3*(1/(a*n-1) - 1/(a*n+2))");
        assert_eq!(profile.form, DeriveTargetForm::FractionExpanded);
    }

    #[test]
    fn classifies_symbolic_shift_gap_telescoping_fraction_expanded_target() {
        let profile = classify("1/((n+a)*(n+b))", "1/(b-a)*(1/(n+a) - 1/(n+b))");
        assert_eq!(profile.form, DeriveTargetForm::FractionExpanded);
    }

    #[test]
    fn classifies_affine_symbolic_shift_gap_telescoping_fraction_expanded_target() {
        let profile = classify("1/((a*n+b)*(a*n+c))", "1/(c-b)*(1/(a*n+b) - 1/(a*n+c))");
        assert_eq!(profile.form, DeriveTargetForm::FractionExpanded);
    }

    #[test]
    fn classifies_fraction_decomposed_target() {
        let profile = classify("(x+1)/(x-1)", "1 + 2/(x-1)");
        assert_eq!(profile.form, DeriveTargetForm::FractionDecomposed);
    }

    #[test]
    fn classifies_fraction_combined_target() {
        let profile = classify("1 + 2/(x-1)", "(x+1)/(x-1)");
        assert_eq!(profile.form, DeriveTargetForm::FractionCombined);
    }

    #[test]
    fn classifies_odd_half_power_target() {
        let profile = classify("x^(3/2)", "abs(x)*sqrt(x)");
        assert_eq!(profile.form, DeriveTargetForm::OddHalfPowerExpanded);
    }

    #[test]
    fn classifies_odd_half_power_target_after_simplify() {
        let profile = classify("sqrt(x^3)", "abs(x)*sqrt(x)");
        assert_eq!(profile.form, DeriveTargetForm::OddHalfPowerExpanded);
    }

    #[test]
    fn does_not_classify_simple_product_as_factored_target() {
        let profile = classify("x + x", "2*x");
        assert_eq!(profile.form, DeriveTargetForm::Unknown);
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
    fn classify_unknown_for_equivalent_but_unsupported_target() {
        let profile = classify("a*x + b*x + c", "x*(a + b + c/x)");
        assert_eq!(
            profile.form,
            DeriveTargetForm::FactoredWithDivision {
                var: "x".to_string()
            }
        );
    }
}
