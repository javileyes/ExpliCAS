use super::strong_target_match;
use cas_ast::views::as_rational_const;
use cas_ast::{Context, Expr, ExprId};
use cas_math::difference_of_squares_support::{
    try_plan_difference_of_squares_division_expr, DifferenceOfSquaresDivisionPolicy,
};
use cas_math::distribution_division_support::try_rewrite_div_distribution_simplifying_expr;
use cas_math::expr_destructure::{as_add, as_div, as_sub};
use cas_math::expr_nary::build_balanced_add;
use cas_math::expr_nary::{self, AddView, Sign};
use cas_math::expr_predicates::{contains_division_like_term, contains_named_var};
use cas_math::fold_add_build_support::try_build_fold_add_fraction_rewrite;
use cas_math::fold_add_fraction_support::extract_fold_add_operands;
use cas_math::fraction_add_rewrite_support::{
    plan_add_fraction_rewrite_with, AddFractionRewriteInput,
};
use cas_math::fraction_add_rule_support::try_plan_sub_term_matches_denom_rewrite;
use cas_math::fraction_combine_policy_support::try_plan_same_denominator_combination_with;
use cas_math::fraction_factors::{
    try_rewrite_cancel_common_factors_expr_with, CancelCommonFactorsGate,
};
use cas_math::fraction_pair_support::extract_fraction_pair;
use cas_math::fraction_sub_rewrite_support::plan_sub_fraction_rewrite_with;
use cas_math::nested_fraction_support::try_rewrite_simplify_nested_fraction_expr;
use cas_math::poly_compare::poly_eq;
use cas_math::symbolic_integration_support::get_linear_coeffs;
use cas_math::trig_roots_flatten::flatten_add_sub_chain;
use num_rational::BigRational;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FractionExpansionKind {
    Distribution,
    TelescopingFraction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FractionExpansionRewrite {
    pub(crate) intermediate: ExprId,
    pub(crate) rewritten: ExprId,
    pub(crate) kind: FractionExpansionKind,
    pub(crate) focus_before: Option<ExprId>,
    pub(crate) focus_after: Option<ExprId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FractionCombinationKind {
    TelescopingFraction,
    MixedFraction,
    SameDenominator,
    AddFractions,
    SameDenominatorSub,
    SubtractFractions,
    SubTermMatchesDenominator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FractionCombinationRewrite {
    pub(crate) rewritten: ExprId,
    pub(crate) kind: FractionCombinationKind,
    pub(crate) focus_before: Option<ExprId>,
    pub(crate) focus_after: Option<ExprId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExactFractionCancelKind {
    CommonFactor,
    DifferenceOfSquares,
    PerfectSquarePlus,
    PerfectSquareMinus,
    SumDiffCubes,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExactFractionCancelRewrite {
    pub(crate) intermediate: ExprId,
    pub(crate) rewritten: ExprId,
    pub(crate) kind: ExactFractionCancelKind,
    pub(crate) required_conditions: Vec<crate::ImplicitCondition>,
    pub(crate) focus_before: Option<ExprId>,
    pub(crate) focus_after: Option<ExprId>,
}

impl FractionCombinationKind {
    pub(crate) fn description(self) -> &'static str {
        match self {
            Self::TelescopingFraction => {
                "Recompose the telescoping partial fractions into a single fraction"
            }
            Self::MixedFraction => "Combine the whole part with the remaining fraction",
            Self::SameDenominator => "Combine fractions that already share the same denominator",
            Self::AddFractions => "Combine two fractions into a single denominator",
            Self::SameDenominatorSub => {
                "Combine fractions with the same denominator into one subtraction"
            }
            Self::SubtractFractions => "Subtract two fractions into a single denominator",
            Self::SubTermMatchesDenominator => {
                "Put the term and the fraction over the same denominator"
            }
        }
    }

    pub(crate) fn rule_name(self) -> &'static str {
        match self {
            Self::TelescopingFraction => "Telescoping Fraction Combine",
            Self::MixedFraction => "Mixed Fraction Combine",
            Self::SameDenominator => "Combine Same Denominator Fractions",
            Self::AddFractions => "Add Fractions",
            Self::SameDenominatorSub => "Combine Same Denominator Sub",
            Self::SubtractFractions => "Subtract Fractions",
            Self::SubTermMatchesDenominator => "Combine Same Denominator Sub",
        }
    }
}

impl FractionExpansionKind {
    pub(crate) fn description(self) -> &'static str {
        match self {
            Self::Distribution => "Distribute a sum over the common denominator",
            Self::TelescopingFraction => "Split into telescoping partial fractions",
        }
    }

    pub(crate) fn rule_name(self) -> &'static str {
        match self {
            Self::Distribution => "Distribute Division",
            Self::TelescopingFraction => "Telescoping Fraction Split",
        }
    }
}

impl ExactFractionCancelKind {
    pub(crate) fn description(self) -> &'static str {
        match self {
            Self::CommonFactor => "Cancel common factor",
            Self::DifferenceOfSquares => "Cancel common factor",
            Self::PerfectSquarePlus => "Cancel common factor",
            Self::PerfectSquareMinus => "Cancel common factor",
            Self::SumDiffCubes => {
                "Factor numerator as a sum or difference of cubes and cancel the common factor"
            }
        }
    }

    pub(crate) fn rule_name(self) -> &'static str {
        match self {
            Self::CommonFactor => "Pre-order Common Factor Cancel",
            Self::DifferenceOfSquares => "Pre-order Difference of Squares Cancel",
            Self::PerfectSquarePlus => "Simplify Nested Fraction",
            Self::PerfectSquareMinus => "Pre-order Perfect Square Minus Cancel",
            Self::SumDiffCubes => "Cancel Sum/Difference of Cubes Fraction",
        }
    }

    pub(crate) fn local_snapshots(
        self,
        source_expr: ExprId,
        intermediate: ExprId,
        rewritten: ExprId,
    ) -> (ExprId, ExprId) {
        match self {
            Self::CommonFactor => (source_expr, rewritten),
            Self::DifferenceOfSquares => (source_expr, rewritten),
            Self::PerfectSquarePlus => (source_expr, rewritten),
            Self::PerfectSquareMinus => (source_expr, intermediate),
            Self::SumDiffCubes => (source_expr, rewritten),
        }
    }
}

pub(crate) fn looks_like_fraction_expanded_target(ctx: &mut Context, expr: ExprId) -> bool {
    let terms = flatten_add_sub_chain(ctx, expr);
    if terms.len() < 2 {
        return false;
    }

    terms
        .into_iter()
        .all(|term| is_fraction_like_term(ctx, term))
}

pub(crate) fn looks_like_telescoping_fraction_target(ctx: &mut Context, expr: ExprId) -> bool {
    try_plan_telescoping_fraction_combination(ctx, expr).is_some()
}

pub(crate) fn looks_like_mixed_fraction_target(ctx: &Context, expr: ExprId) -> bool {
    let mut temp_ctx = ctx.clone();
    try_build_combined_fraction_from_fold_add(&mut temp_ctx, expr).is_some()
}

pub(crate) fn try_rewrite_nested_fraction_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<ExprId> {
    let rewritten = try_rewrite_simplify_nested_fraction_expr(ctx, source_expr)
        .map(|rewrite| rewrite.rewritten)
        .or_else(|| try_rewrite_unit_over_unit_fraction(ctx, source_expr))?;

    if strong_target_match(ctx, rewritten, target_expr) {
        return Some(rewritten);
    }

    let normalized = cas_math::canonical_forms::normalize_core(ctx, rewritten);
    if strong_target_match(ctx, normalized, target_expr) {
        return Some(rewritten);
    }

    strip_division_by_one(ctx, rewritten)
        .filter(|stripped| strong_target_match(ctx, *stripped, target_expr))
}

fn try_rewrite_unit_over_unit_fraction(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    if !is_one(ctx, *numerator) {
        return None;
    }
    let Expr::Div(inner_numerator, inner_denominator) = ctx.get(*denominator) else {
        return None;
    };
    is_one(ctx, *inner_numerator).then_some(*inner_denominator)
}

fn strip_division_by_one(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    is_one(ctx, *denominator).then_some(*numerator)
}

pub(crate) fn try_build_combined_fraction_from_fold_add(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };
    if let Some(ops) = extract_fold_add_operands(ctx, *left, *right) {
        if !contains_division_like_term(ctx, ops.term) {
            return try_build_fold_add_fraction_rewrite(
                ctx,
                expr,
                ops.term,
                ops.numerator,
                ops.denominator,
            );
        }
    }

    try_build_combined_fraction_from_scaled_fold_add(ctx, *left, *right)
}

fn try_build_combined_fraction_from_scaled_fold_add(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    let candidates = [(left, right), (right, left)];

    for (whole_term, remainder_term) in candidates {
        let Expr::Div(whole_num, whole_den) = ctx.get(whole_term) else {
            continue;
        };
        let Expr::Div(rem_num, rem_den) = ctx.get(remainder_term) else {
            continue;
        };
        let whole_num = *whole_num;
        let whole_den = *whole_den;
        let rem_num = *rem_num;
        let rem_den = *rem_den;

        let candidate_vars = cas_ast::collect_variables(ctx, rem_den);
        for var_name in candidate_vars {
            if contains_named_var(ctx, whole_num, &var_name)
                || contains_named_var(ctx, whole_den, &var_name)
                || contains_named_var(ctx, rem_num, &var_name)
            {
                continue;
            }

            let Some((linear_coeff, offset)) = get_linear_coeffs(ctx, rem_den, &var_name) else {
                continue;
            };
            let zero = ctx.num(0);
            if strong_target_match(ctx, linear_coeff, zero) {
                continue;
            }
            let coeff_matches_whole_den = strong_target_match(ctx, linear_coeff, whole_den)
                || poly_eq(ctx, linear_coeff, whole_den);
            let neg_whole_den = ctx.add(Expr::Neg(whole_den));
            let coeff_matches_neg_whole_den = strong_target_match(ctx, linear_coeff, neg_whole_den)
                || poly_eq(ctx, linear_coeff, neg_whole_den);
            if !coeff_matches_whole_den && !coeff_matches_neg_whole_den {
                continue;
            }

            let var_expr = ctx.var(&var_name);
            let signed_whole_num = if coeff_matches_neg_whole_den {
                ctx.add(Expr::Neg(whole_num))
            } else {
                whole_num
            };
            let whole_times_var = ctx.add(Expr::Mul(signed_whole_num, var_expr));
            let whole_times_offset = ctx.add(Expr::Mul(whole_num, offset));
            let lifted_offset = ctx.add(Expr::Div(whole_times_offset, whole_den));
            let numerator_tail = ctx.add(Expr::Add(lifted_offset, rem_num));
            let numerator = ctx.add(Expr::Add(whole_times_var, numerator_tail));
            return Some(ctx.add(Expr::Div(numerator, rem_den)));
        }
    }

    None
}

pub(crate) fn try_rewrite_fraction_combination_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<FractionCombinationRewrite> {
    if let Some((passthrough_terms, source_focus_terms, target_focus_terms)) =
        extract_additive_passthrough_terms(ctx, source_expr, target_expr)
    {
        let source_focus = build_balanced_add(ctx, &source_focus_terms);
        let target_focus = build_balanced_add(ctx, &target_focus_terms);
        if let Some(rewrite) =
            try_rewrite_fraction_combination_core_target_aware(ctx, source_focus, target_focus)
        {
            let mut rebuilt_terms = passthrough_terms;
            rebuilt_terms.push(rewrite.rewritten);
            let rebuilt = build_balanced_add(ctx, &rebuilt_terms);
            if strong_target_match(ctx, rebuilt, target_expr) {
                return Some(FractionCombinationRewrite {
                    rewritten: rebuilt,
                    kind: rewrite.kind,
                    focus_before: Some(source_focus),
                    focus_after: Some(rewrite.rewritten),
                });
            }
        }
    }

    try_rewrite_fraction_combination_core_target_aware(ctx, source_expr, target_expr)
}

fn try_rewrite_fraction_combination_core_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<FractionCombinationRewrite> {
    let mut candidates = Vec::with_capacity(6);
    let is_subtraction = as_sub(ctx, source_expr).is_some();

    if let Some(rewritten) = try_plan_telescoping_fraction_combination(ctx, source_expr) {
        candidates.push(FractionCombinationRewrite {
            rewritten,
            kind: FractionCombinationKind::TelescopingFraction,
            focus_before: None,
            focus_after: None,
        });
    }

    if let Some(rewritten) = try_build_combined_fraction_from_fold_add(ctx, source_expr) {
        candidates.push(FractionCombinationRewrite {
            rewritten,
            kind: FractionCombinationKind::MixedFraction,
            focus_before: None,
            focus_after: None,
        });
    }

    if is_subtraction {
        if let Some(rewritten) = try_plan_same_denominator_subtraction(ctx, source_expr) {
            candidates.push(FractionCombinationRewrite {
                rewritten,
                kind: FractionCombinationKind::SameDenominatorSub,
                focus_before: None,
                focus_after: None,
            });
        }

        if let Some(rewritten) = try_plan_sub_fraction_pair(ctx, source_expr) {
            candidates.push(FractionCombinationRewrite {
                rewritten,
                kind: FractionCombinationKind::SubtractFractions,
                focus_before: None,
                focus_after: None,
            });
        }

        if let Some(rewritten) = try_plan_sub_term_matches_denominator(ctx, source_expr) {
            candidates.push(FractionCombinationRewrite {
                rewritten,
                kind: FractionCombinationKind::SubTermMatchesDenominator,
                focus_before: None,
                focus_after: None,
            });
        }
    }

    if let Some(plan) =
        try_plan_same_denominator_combination_with(ctx, source_expr, false, false, |_ctx, _den| {
            false
        })
    {
        candidates.push(FractionCombinationRewrite {
            rewritten: plan.build.result,
            kind: FractionCombinationKind::SameDenominator,
            focus_before: None,
            focus_after: None,
        });
    }

    if let Some(rewritten) = try_plan_add_fraction_pair(ctx, source_expr) {
        candidates.push(FractionCombinationRewrite {
            rewritten,
            kind: FractionCombinationKind::AddFractions,
            focus_before: None,
            focus_after: None,
        });
    }

    candidates.into_iter().find(|candidate| {
        target_matches_fraction_combination(ctx, candidate.rewritten, target_expr)
    })
}

fn try_plan_telescoping_fraction_combination(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let (u, u_plus_gap, _gap) = telescoping_fraction_split_base_and_gap(ctx, expr)?;
    let denominator = ctx.add(Expr::Mul(u, u_plus_gap));
    let one = ctx.num(1);
    Some(ctx.add(Expr::Div(one, denominator)))
}

fn telescoping_fraction_split_base_and_gap(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, ExprId)> {
    let (u, u_plus_gap, gap_expr) = extract_telescoping_split_components(ctx, expr)?;
    additive_gap_relation_holds(ctx, u, gap_expr, u_plus_gap).then_some((u, u_plus_gap, gap_expr))
}

fn extract_telescoping_split_components(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, ExprId)> {
    if let Some((numerator, denominator)) = as_div(ctx, expr) {
        if let Some((u, u_plus_gap)) = extract_telescoping_fraction_core(ctx, numerator) {
            return Some((u, u_plus_gap, denominator));
        }
    }

    let factors = expr_nary::mul_leaves(ctx, expr);
    for core_index in 0..factors.len() {
        let Some((u, u_plus_gap)) = extract_telescoping_fraction_core(ctx, factors[core_index])
        else {
            continue;
        };
        let gap_expr = match extract_gap_expr_without_index(ctx, &factors, core_index) {
            Some(gap_expr) => gap_expr,
            None => continue,
        };
        return Some((u, u_plus_gap, gap_expr));
    }

    None
}

fn extract_telescoping_fraction_core(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return None;
    }

    let mut saw_u = None;
    let mut saw_u_plus_gap = None;
    for (term, sign) in terms {
        let (numerator, denominator) = as_div(ctx, term)?;
        if !is_one(ctx, numerator) {
            return None;
        }

        match sign {
            Sign::Pos => saw_u = Some(denominator),
            Sign::Neg => saw_u_plus_gap = Some(denominator),
        }
    }

    Some((saw_u?, saw_u_plus_gap?))
}

fn extract_gap_expr_without_index(
    ctx: &mut Context,
    factors: &[ExprId],
    skip_index: usize,
) -> Option<ExprId> {
    let retained = factors
        .iter()
        .enumerate()
        .filter_map(|(index, factor)| (index != skip_index).then_some(*factor))
        .collect::<Vec<_>>();
    match retained.as_slice() {
        [] => Some(ctx.num(1)),
        [single] => extract_unit_reciprocal_denominator(ctx, *single),
        _ => None,
    }
}

fn extract_unit_reciprocal_denominator(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    if let Some((numerator, denominator)) = as_div(ctx, expr) {
        return is_one(ctx, numerator).then_some(denominator);
    }

    let value = as_rational_const(ctx, expr, 4)?;
    if *value.numer() != 1.into() {
        return None;
    }
    Some(ctx.add(Expr::Number(BigRational::from_integer(
        value.denom().clone(),
    ))))
}

fn additive_signature(ctx: &Context, expr: ExprId) -> (Vec<(ExprId, i32)>, BigRational) {
    let mut terms: Vec<(ExprId, i32)> = Vec::new();
    let mut constant = BigRational::from_integer(0.into());

    for (term, sign) in AddView::from_expr(ctx, expr).terms {
        if let Some(value) = as_rational_const(ctx, term, 4) {
            match sign {
                Sign::Pos => constant += value,
                Sign::Neg => constant -= value,
            }
        } else {
            let delta = match sign {
                Sign::Pos => 1,
                Sign::Neg => -1,
            };
            if let Some((_, count)) = terms.iter_mut().find(|(existing, _)| {
                cas_ast::ordering::compare_expr(ctx, *existing, term) == std::cmp::Ordering::Equal
            }) {
                *count += delta;
            } else {
                terms.push((term, delta));
            }
        }
    }

    terms.retain(|(_, count)| *count != 0);
    terms.sort_by(|(left_expr, _), (right_expr, _)| {
        cas_ast::ordering::compare_expr(ctx, *left_expr, *right_expr)
    });

    (terms, constant)
}

fn additive_gap_relation_holds(
    ctx: &mut Context,
    base: ExprId,
    gap: ExprId,
    target: ExprId,
) -> bool {
    let (base_terms, base_constant) = additive_signature(ctx, base);
    let (gap_terms, gap_constant) = additive_signature(ctx, gap);
    let (target_terms, target_constant) = additive_signature(ctx, target);

    let mut combined_terms = base_terms;
    for (term, count) in gap_terms {
        if let Some((_, existing_count)) = combined_terms.iter_mut().find(|(existing, _)| {
            cas_ast::ordering::compare_expr(ctx, *existing, term) == std::cmp::Ordering::Equal
        }) {
            *existing_count += count;
        } else {
            combined_terms.push((term, count));
        }
    }
    combined_terms.retain(|(_, count)| *count != 0);
    combined_terms.sort_by(|(left_expr, _), (right_expr, _)| {
        cas_ast::ordering::compare_expr(ctx, *left_expr, *right_expr)
    });

    if combined_terms == target_terms
        && base_constant.clone() + gap_constant.clone() == target_constant
    {
        return true;
    }

    let combined = ctx.add(Expr::Add(base, gap));
    poly_eq(ctx, combined, target)
}

fn is_one(ctx: &Context, expr: ExprId) -> bool {
    as_rational_const(ctx, expr, 4)
        .is_some_and(|value| value == BigRational::from_integer(1.into()))
}

pub(crate) fn try_rewrite_fraction_expansion(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    simplify_options: crate::SimplifyOptions,
) -> Option<FractionExpansionRewrite> {
    let rewrite = try_rewrite_div_distribution_simplifying_expr(&mut simplifier.context, expr)
        .map(|rewrite| rewrite.rewritten)
        .or_else(|| distribute_fraction_numerator_terms(&mut simplifier.context, expr))?;
    let rewritten = simplify_fraction_sum_terms_individually(simplifier, rewrite, simplify_options);
    Some(FractionExpansionRewrite {
        intermediate: rewrite,
        rewritten,
        kind: FractionExpansionKind::Distribution,
        focus_before: None,
        focus_after: None,
    })
}

pub(crate) fn try_rewrite_fraction_expansion_target_aware(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    target_expr: ExprId,
    simplify_options: crate::SimplifyOptions,
) -> Option<FractionExpansionRewrite> {
    if let Some(rewrite) = try_rewrite_telescoping_fraction_expansion_target_aware(
        &mut simplifier.context,
        expr,
        target_expr,
    ) {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_fraction_expansion(simplifier, expr, simplify_options.clone())
    {
        if strong_target_match(&mut simplifier.context, rewrite.rewritten, target_expr) {
            return Some(rewrite);
        }
    }

    let (passthrough_terms, source_focus_terms, target_focus_terms) =
        extract_additive_passthrough_terms(&mut simplifier.context, expr, target_expr)?;
    let source_focus = build_balanced_add(&mut simplifier.context, &source_focus_terms);
    let target_focus = build_balanced_add(&mut simplifier.context, &target_focus_terms);
    let focus_rewrite = try_rewrite_fraction_expansion(simplifier, source_focus, simplify_options)?;

    if !strong_target_match(
        &mut simplifier.context,
        focus_rewrite.rewritten,
        target_focus,
    ) {
        return None;
    }

    let mut rebuilt_terms = passthrough_terms.clone();
    rebuilt_terms.push(focus_rewrite.intermediate);
    let rebuilt_intermediate = build_balanced_add(&mut simplifier.context, &rebuilt_terms);

    let mut rebuilt_terms = passthrough_terms;
    rebuilt_terms.push(focus_rewrite.rewritten);
    let rebuilt = build_balanced_add(&mut simplifier.context, &rebuilt_terms);
    if !strong_target_match(&mut simplifier.context, rebuilt, target_expr) {
        return None;
    }

    Some(FractionExpansionRewrite {
        intermediate: rebuilt_intermediate,
        rewritten: rebuilt,
        kind: focus_rewrite.kind,
        focus_before: Some(source_focus),
        focus_after: Some(focus_rewrite.intermediate),
    })
}

fn try_rewrite_telescoping_fraction_expansion_target_aware(
    ctx: &mut Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<FractionExpansionRewrite> {
    let rewritten = try_plan_telescoping_fraction_combination(ctx, target_expr)?;
    if !strong_target_match(ctx, rewritten, expr) {
        return None;
    }

    Some(FractionExpansionRewrite {
        intermediate: target_expr,
        rewritten: target_expr,
        kind: FractionExpansionKind::TelescopingFraction,
        focus_before: None,
        focus_after: None,
    })
}

pub(crate) fn try_rewrite_exact_fraction_cancel_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<ExactFractionCancelRewrite> {
    if let Some((passthrough_terms, source_focus_terms, target_focus_terms)) =
        extract_additive_passthrough_terms(ctx, source_expr, target_expr)
    {
        let source_focus = build_balanced_add(ctx, &source_focus_terms);
        let target_focus = build_balanced_add(ctx, &target_focus_terms);

        if let Some(rewrite) =
            try_rewrite_exact_fraction_cancel_core_target_aware(ctx, source_focus, target_focus)
        {
            let mut rebuilt_terms = passthrough_terms;
            rebuilt_terms.push(rewrite.rewritten);
            let rebuilt = build_balanced_add(ctx, &rebuilt_terms);

            if strong_target_match(ctx, rebuilt, target_expr) {
                let (focus_before, focus_after) = rewrite.kind.local_snapshots(
                    source_focus,
                    rewrite.intermediate,
                    rewrite.rewritten,
                );

                return Some(ExactFractionCancelRewrite {
                    intermediate: rewrite.intermediate,
                    rewritten: rebuilt,
                    kind: rewrite.kind,
                    required_conditions: rewrite.required_conditions,
                    focus_before: Some(focus_before),
                    focus_after: Some(focus_after),
                });
            }
        }
    }

    try_rewrite_exact_fraction_cancel_core_target_aware(ctx, source_expr, target_expr)
}

fn try_rewrite_exact_fraction_cancel_core_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<ExactFractionCancelRewrite> {
    let rewrites = [
        try_rewrite_common_factor_fraction_cancel(ctx, source_expr),
        try_rewrite_difference_of_squares_fraction_cancel(ctx, source_expr),
        try_rewrite_perfect_square_plus_fraction_cancel(ctx, source_expr),
        try_rewrite_perfect_square_minus_fraction_cancel(ctx, source_expr),
        try_rewrite_sum_diff_cubes_fraction_cancel(ctx, source_expr),
        try_rewrite_fraction_cancel_via_target_product(ctx, source_expr, target_expr),
    ];

    rewrites
        .into_iter()
        .flatten()
        .find(|rewrite| fraction_cancel_target_match(ctx, rewrite.rewritten, target_expr))
}

fn fraction_cancel_target_match(
    ctx: &mut Context,
    actual_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    if strong_target_match(ctx, actual_expr, target_expr) {
        return true;
    }

    poly_eq(ctx, actual_expr, target_expr)
        || cas_math::semantic_equality::SemanticEqualityChecker::new(ctx)
            .are_equal(actual_expr, target_expr)
}

fn try_rewrite_fraction_cancel_via_target_product(
    ctx: &mut Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<ExactFractionCancelRewrite> {
    let (numerator, denominator) = as_div(ctx, expr)?;
    let factored_numerator = ctx.add(Expr::Mul(target_expr, denominator));

    if !(strong_target_match(ctx, factored_numerator, numerator)
        || poly_eq(ctx, factored_numerator, numerator))
    {
        return None;
    }

    let intermediate = ctx.add(Expr::Div(factored_numerator, denominator));
    Some(ExactFractionCancelRewrite {
        intermediate,
        rewritten: target_expr,
        kind: ExactFractionCancelKind::CommonFactor,
        required_conditions: vec![crate::ImplicitCondition::NonZero(denominator)],
        focus_before: None,
        focus_after: None,
    })
}

fn simplify_fraction_sum_terms_individually(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    mut simplify_options: crate::SimplifyOptions,
) -> ExprId {
    simplify_options.collect_steps = false;

    let terms = flatten_add_sub_chain(&mut simplifier.context, expr);
    if terms.len() < 2 {
        return expr;
    }

    let simplified_terms = terms
        .into_iter()
        .map(|term| {
            // A first cancellation can expose a second direct factor cancellation,
            // so settle each distributed term locally before rebuilding the sum.
            let mut current = term;
            for _ in 0..3 {
                let (rewritten, _steps, _stats) =
                    simplifier.simplify_with_stats(current, simplify_options.clone());
                if rewritten == current {
                    break;
                }
                current = rewritten;
            }
            current
        })
        .collect::<Vec<_>>();

    build_balanced_add(&mut simplifier.context, &simplified_terms)
}

fn try_rewrite_perfect_square_minus_fraction_cancel(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExactFractionCancelRewrite> {
    let (numerator, denominator) = as_div(ctx, expr)?;
    let (a, b) = extract_subtraction_terms(ctx, denominator)?;

    let two = ctx.num(2);
    let a_sq = ctx.add(Expr::Pow(a, two));
    let b_sq = square_expr(ctx, b)?;
    let two_ab = build_scaled_product(ctx, 2, a, b);
    let neg_two_ab = ctx.add(Expr::Neg(two_ab));
    let tail = ctx.add(Expr::Add(neg_two_ab, b_sq));
    let expected = ctx.add(Expr::Add(a_sq, tail));

    if !(numerator == expected || poly_eq(ctx, numerator, expected)) {
        return None;
    }

    let a_minus_b = ctx.add(Expr::Sub(a, b));
    let squared = ctx.add(Expr::Pow(a_minus_b, two));
    let intermediate = ctx.add(Expr::Div(squared, denominator));
    Some(ExactFractionCancelRewrite {
        intermediate,
        rewritten: a_minus_b,
        kind: ExactFractionCancelKind::PerfectSquareMinus,
        required_conditions: vec![crate::ImplicitCondition::NonZero(denominator)],
        focus_before: None,
        focus_after: None,
    })
}

fn try_rewrite_perfect_square_plus_fraction_cancel(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExactFractionCancelRewrite> {
    let (numerator, denominator) = as_div(ctx, expr)?;
    let (a, b) = extract_addition_terms(ctx, denominator)?;

    let a_sq = square_expr(ctx, a)?;
    let b_sq = square_expr(ctx, b)?;
    let two_ab = build_scaled_product(ctx, 2, a, b);
    let tail = ctx.add(Expr::Add(two_ab, b_sq));
    let expected = ctx.add(Expr::Add(a_sq, tail));

    if !(numerator == expected || poly_eq(ctx, numerator, expected)) {
        return None;
    }

    let rewritten = ctx.add(Expr::Add(a, b));
    Some(ExactFractionCancelRewrite {
        intermediate: expr,
        rewritten,
        kind: ExactFractionCancelKind::PerfectSquarePlus,
        required_conditions: vec![crate::ImplicitCondition::NonZero(denominator)],
        focus_before: None,
        focus_after: None,
    })
}

fn try_rewrite_common_factor_fraction_cancel(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExactFractionCancelRewrite> {
    let rewrite = try_rewrite_cancel_common_factors_expr_with(
        ctx,
        expr,
        |_ctx, _nonzero_base, emit_assumption| CancelCommonFactorsGate {
            allow: true,
            assumed: emit_assumption,
        },
    )?;

    Some(ExactFractionCancelRewrite {
        intermediate: expr,
        rewritten: rewrite.rewritten,
        kind: ExactFractionCancelKind::CommonFactor,
        required_conditions: rewrite
            .assumed_nonzero_targets
            .into_iter()
            .map(crate::ImplicitCondition::NonZero)
            .collect(),
        focus_before: None,
        focus_after: None,
    })
}

fn try_rewrite_difference_of_squares_fraction_cancel(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExactFractionCancelRewrite> {
    let (numerator, denominator) = as_div(ctx, expr)?;
    let plan = try_plan_difference_of_squares_division_expr(
        ctx,
        numerator,
        denominator,
        DifferenceOfSquaresDivisionPolicy::default(),
    )?;

    Some(ExactFractionCancelRewrite {
        intermediate: plan.intermediate,
        rewritten: plan.final_result,
        kind: ExactFractionCancelKind::DifferenceOfSquares,
        required_conditions: vec![crate::ImplicitCondition::NonZero(denominator)],
        focus_before: None,
        focus_after: None,
    })
}

fn try_rewrite_sum_diff_cubes_fraction_cancel(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExactFractionCancelRewrite> {
    let (numerator, denominator) = as_div(ctx, expr)?;
    let mut steps = Vec::new();
    let rewritten = cas_engine::rules::algebra::try_sum_diff_of_cubes_preorder(
        ctx,
        expr,
        numerator,
        denominator,
        false,
        &mut steps,
        &[],
    )?;

    Some(ExactFractionCancelRewrite {
        intermediate: expr,
        rewritten,
        kind: ExactFractionCancelKind::SumDiffCubes,
        required_conditions: vec![crate::ImplicitCondition::NonZero(denominator)],
        focus_before: None,
        focus_after: None,
    })
}

fn extract_subtraction_terms(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    if let Some((left, right)) = as_sub(ctx, expr) {
        return Some((left, right));
    }

    let (left, right) = as_add(ctx, expr)?;
    if let Expr::Neg(inner) = ctx.get(right) {
        return Some((left, *inner));
    }
    if let Expr::Neg(inner) = ctx.get(left) {
        return Some((right, *inner));
    }
    None
}

fn extract_addition_terms(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let (left, right) = as_add(ctx, expr)?;
    if matches!(ctx.get(left), Expr::Neg(_)) || matches!(ctx.get(right), Expr::Neg(_)) {
        return None;
    }
    Some((left, right))
}

fn square_expr(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let two = ctx.num(2);
    Some(ctx.add(Expr::Pow(expr, two)))
}

fn build_scaled_product(ctx: &mut Context, factor: i64, left: ExprId, right: ExprId) -> ExprId {
    let product = ctx.add(Expr::Mul(left, right));
    if factor == 1 {
        product
    } else {
        let factor_expr = ctx.num(factor);
        ctx.add(Expr::Mul(factor_expr, product))
    }
}

fn is_fraction_like_term(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Div(_, _) => true,
        Expr::Neg(inner) => matches!(ctx.get(*inner), Expr::Div(_, _)),
        _ => false,
    }
}

fn distribute_fraction_numerator_terms(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let (numer, denom) = as_div(ctx, expr)?;
    let numerator_terms = AddView::from_expr(ctx, numer);
    if numerator_terms.terms.len() < 2 {
        return None;
    }

    let distributed_terms = numerator_terms
        .terms
        .iter()
        .map(|&(term, sign)| {
            let div_term = ctx.add(Expr::Div(term, denom));
            match sign {
                Sign::Pos => div_term,
                Sign::Neg => ctx.add(Expr::Neg(div_term)),
            }
        })
        .collect::<Vec<_>>();

    Some(build_balanced_add(ctx, &distributed_terms))
}

fn try_plan_add_fraction_pair(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let (left, right) = as_add(ctx, expr)?;
    let pair = extract_fraction_pair(ctx, left, right);
    if !pair.is_frac1 || !pair.is_frac2 {
        return None;
    }

    let plan = plan_add_fraction_rewrite_with(
        ctx,
        AddFractionRewriteInput {
            expr,
            l: left,
            r: right,
            n1: pair.n1,
            d1: pair.d1,
            n2: pair.n2,
            d2: pair.d2,
            same_sign: pair.sign1 == pair.sign2,
            inside_trig: false,
        },
        cas_math::expand_ops::expand,
    )?;

    Some(plan.rewritten)
}

fn try_plan_same_denominator_subtraction(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let (left, right) = as_sub(ctx, expr)?;
    let pair = extract_fraction_pair(ctx, left, right);
    if !pair.is_frac1 || !pair.is_frac2 {
        return None;
    }
    if !strong_target_match(ctx, pair.d1, pair.d2) {
        return None;
    }

    let plan = plan_sub_fraction_rewrite_with(
        ctx,
        pair.n1,
        pair.n2,
        pair.d1,
        pair.d2,
        cas_math::expand_ops::expand,
    );

    Some(plan.rewritten)
}

fn try_plan_sub_fraction_pair(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let (left, right) = as_sub(ctx, expr)?;
    let pair = extract_fraction_pair(ctx, left, right);
    if !pair.is_frac1 || !pair.is_frac2 {
        return None;
    }

    let plan = plan_sub_fraction_rewrite_with(
        ctx,
        pair.n1,
        pair.n2,
        pair.d1,
        pair.d2,
        cas_math::expand_ops::expand,
    );

    Some(plan.rewritten)
}

fn try_plan_sub_term_matches_denominator(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let plan = try_plan_sub_term_matches_denom_rewrite(ctx, expr, false)?;
    Some(plan.rewritten)
}

fn target_matches_fraction_combination(
    ctx: &mut Context,
    rewritten: ExprId,
    target_expr: ExprId,
) -> bool {
    if strong_target_match(ctx, rewritten, target_expr) {
        return true;
    }

    let Some((rewritten_num, rewritten_den)) = as_div(ctx, rewritten) else {
        return false;
    };
    let Some((target_num, target_den)) = as_div(ctx, target_expr) else {
        return false;
    };

    (strong_target_match(ctx, rewritten_num, target_num) || poly_eq(ctx, rewritten_num, target_num))
        && (strong_target_match(ctx, rewritten_den, target_den)
            || poly_eq(ctx, rewritten_den, target_den))
        || simplified_difference_matches_zero(ctx, rewritten, target_expr)
}

fn simplified_difference_matches_zero(ctx: &mut Context, left: ExprId, right: ExprId) -> bool {
    let difference = ctx.add(Expr::Sub(left, right));
    let mut temp = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut temp.context, ctx);
    let (simplified, _steps, _stats) = temp.simplify_with_stats(
        difference,
        crate::SimplifyOptions {
            suppress_depth_overflow_warnings: true,
            ..crate::SimplifyOptions::default()
        },
    );
    std::mem::swap(&mut temp.context, ctx);

    let zero = ctx.num(0);
    strong_target_match(ctx, simplified, zero)
}

fn extract_additive_passthrough_terms(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<(Vec<ExprId>, Vec<ExprId>, Vec<ExprId>)> {
    let source_terms = signed_add_terms(ctx, source_expr);
    let target_terms = signed_add_terms(ctx, target_expr);
    if source_terms.len() < 2 || target_terms.len() < 2 {
        return None;
    }

    let mut target_used = vec![false; target_terms.len()];
    let mut passthrough_terms = Vec::new();
    let mut source_focus_terms = Vec::new();

    for source_term in &source_terms {
        let mut matched_index = None;
        for (index, target_term) in target_terms.iter().enumerate() {
            if target_used[index] {
                continue;
            }
            if strong_target_match(ctx, *source_term, *target_term) {
                matched_index = Some(index);
                break;
            }
        }

        if let Some(index) = matched_index {
            target_used[index] = true;
            passthrough_terms.push(*source_term);
        } else {
            source_focus_terms.push(*source_term);
        }
    }

    let target_focus_terms = target_terms
        .into_iter()
        .enumerate()
        .filter_map(|(index, term)| (!target_used[index]).then_some(term))
        .collect::<Vec<_>>();

    if passthrough_terms.is_empty()
        || source_focus_terms.is_empty()
        || target_focus_terms.is_empty()
    {
        return None;
    }

    Some((passthrough_terms, source_focus_terms, target_focus_terms))
}

fn signed_add_terms(ctx: &mut Context, expr: ExprId) -> Vec<ExprId> {
    flatten_add_sub_chain(ctx, expr)
}

#[cfg(test)]
mod tests {
    use super::{
        looks_like_fraction_expanded_target, looks_like_mixed_fraction_target, strong_target_match,
        try_build_combined_fraction_from_fold_add, try_rewrite_exact_fraction_cancel_target_aware,
        try_rewrite_fraction_combination_target_aware, try_rewrite_fraction_expansion,
        try_rewrite_fraction_expansion_target_aware, try_rewrite_nested_fraction_target_aware,
        ExactFractionCancelKind, FractionCombinationKind, FractionExpansionKind,
    };
    use crate::runtime::Simplifier;
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn detects_fraction_expanded_target_shape() {
        let mut ctx = Context::new();
        let expr = parse("a/x + b/x", &mut ctx).expect("parse");
        assert!(looks_like_fraction_expanded_target(&mut ctx, expr));
    }

    #[test]
    fn rejects_non_fraction_additive_target_shape() {
        let mut ctx = Context::new();
        let expr = parse("a + b/x", &mut ctx).expect("parse");
        assert!(!looks_like_fraction_expanded_target(&mut ctx, expr));
    }

    #[test]
    fn rewrites_nested_fraction_targets_aware() {
        let cases = [
            ("1/(1/a)", "a"),
            ("1/(1/a + 1/b)", "(a*b)/(a+b)"),
            ("a/(b + c/d)", "a*d/(b*d+c)"),
            ("(a + b/c)/d", "(a*c+b)/(c*d)"),
        ];

        for (source_text, target_text) in cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewritten = try_rewrite_nested_fraction_target_aware(&mut ctx, source, target)
                .expect("nested fraction rewrite");

            assert!(
                strong_target_match(&mut ctx, rewritten, target),
                "expected nested fraction rewrite for `{source_text}` -> `{target_text}`",
            );
        }
    }

    #[test]
    fn detects_mixed_fraction_target_shape() {
        let mut ctx = Context::new();
        let expr = parse("1 + 2/(x-1)", &mut ctx).expect("parse");
        assert!(looks_like_mixed_fraction_target(&ctx, expr));
    }

    #[test]
    fn detects_scaled_affine_mixed_fraction_target_shape() {
        let mut ctx = Context::new();
        let expr = parse("a/c + (b-a*d/c)/(c*x+d)", &mut ctx).expect("parse");
        assert!(looks_like_mixed_fraction_target(&ctx, expr));
    }

    #[test]
    fn detects_negative_scaled_affine_mixed_fraction_target_shape() {
        let mut ctx = Context::new();
        let expr = parse("-a/c + (b+a*d/c)/(d-c*x)", &mut ctx).expect("parse");
        assert!(looks_like_mixed_fraction_target(&ctx, expr));
    }

    #[test]
    fn combines_fold_add_fraction_pattern() {
        let mut ctx = Context::new();
        let expr = parse("1 + 2/(x-1)", &mut ctx).expect("parse");
        let rewritten = try_build_combined_fraction_from_fold_add(&mut ctx, expr).expect("rewrite");
        let expected = parse("(x+1)/(x-1)", &mut ctx).expect("expected");
        let checker = cas_math::semantic_equality::SemanticEqualityChecker::new(&ctx);
        assert!(checker.are_equal(rewritten, expected));
    }

    #[test]
    fn combines_scaled_affine_fold_add_fraction_pattern() {
        let mut ctx = Context::new();
        let expr = parse("a/c + (b-a*d/c)/(c*x+d)", &mut ctx).expect("parse");
        let rewritten = try_build_combined_fraction_from_fold_add(&mut ctx, expr).expect("rewrite");
        let expected = parse("(a*x+b)/(c*x+d)", &mut ctx).expect("expected");
        assert!(super::simplified_difference_matches_zero(
            &mut ctx, rewritten, expected
        ));
    }

    #[test]
    fn combines_negative_scaled_affine_fold_add_fraction_pattern() {
        let mut ctx = Context::new();
        let expr = parse("-a/c + (b+a*d/c)/(d-c*x)", &mut ctx).expect("parse");
        let rewritten = try_build_combined_fraction_from_fold_add(&mut ctx, expr).expect("rewrite");
        let expected = parse("(a*x+b)/(d-c*x)", &mut ctx).expect("expected");
        assert!(super::simplified_difference_matches_zero(
            &mut ctx, rewritten, expected
        ));
    }

    #[test]
    fn rewrites_fraction_expansion_with_termwise_cleanup() {
        let mut simplifier = Simplifier::with_default_rules();
        let expr = parse("(x+y)/(x*y)", &mut simplifier.context).expect("parse");
        let rewrite = try_rewrite_fraction_expansion(
            &mut simplifier,
            expr,
            crate::runtime::SimplifyOptions::default(),
        )
        .expect("rewrite");

        let text = rendered(&simplifier.context, rewrite.rewritten);
        assert!(text.contains("1 / x") || text.contains("1 / y"));
    }

    #[test]
    fn combines_tabulated_fraction_targets_aware() {
        let cases = [
            (
                "a/x + b/x",
                "(a+b)/x",
                FractionCombinationKind::SameDenominator,
            ),
            (
                "1/x + 1/y",
                "(x+y)/(x*y)",
                FractionCombinationKind::AddFractions,
            ),
            (
                "a/c + (b-a*d/c)/(c*x+d)",
                "(a*x+b)/(c*x+d)",
                FractionCombinationKind::MixedFraction,
            ),
            (
                "a/x - b/x",
                "(a-b)/x",
                FractionCombinationKind::SameDenominatorSub,
            ),
            (
                "1/x - 1/y",
                "(y-x)/(x*y)",
                FractionCombinationKind::SubtractFractions,
            ),
            (
                "a - b/a",
                "(a^2-b)/a",
                FractionCombinationKind::SubTermMatchesDenominator,
            ),
        ];

        for (source_text, target_text, expected_kind) in cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_fraction_combination_target_aware(&mut ctx, source, target)
                .expect("rewrite");

            assert_eq!(
                rewrite.kind, expected_kind,
                "expected `{source_text}` -> `{target_text}` to use {expected_kind:?}",
            );
        }
    }

    #[test]
    fn combines_tabulated_telescoping_fraction_targets_aware() {
        let cases = [
            ("1/n - 1/(n+1)", "1/(n*(n+1))"),
            ("1/2*(1/n - 1/(n+2))", "1/(n*(n+2))"),
            ("1/2*(1/(n-2) - 1/n)", "1/(n*(n-2))"),
            ("1/2*(1/(2*n+1) - 1/(2*n+3))", "1/((2*n+1)*(2*n+3))"),
            ("1/(c-b)*(1/(a*n+b) - 1/(a*n+c))", "1/((a*n+b)*(a*n+c))"),
            ("1/(2*a)*(1/(x-a) - 1/(x+a))", "1/(x^2-a^2)"),
        ];

        for (source_text, target_text) in cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_fraction_combination_target_aware(&mut ctx, source, target)
                .expect("rewrite");

            assert_eq!(
                rewrite.kind,
                FractionCombinationKind::TelescopingFraction,
                "expected `{source_text}` -> `{target_text}` to combine telescopically",
            );
        }
    }

    #[test]
    fn combines_same_denominator_fraction_subset_with_passthrough_target_aware() {
        let mut ctx = Context::new();
        let source = parse("1 + a/d + b/d + c/d", &mut ctx).expect("parse");
        let target = parse("1 + (a+b+c)/d", &mut ctx).expect("parse");
        let rewrite = try_rewrite_fraction_combination_target_aware(&mut ctx, source, target)
            .expect("rewrite");

        assert_eq!(rewrite.kind, FractionCombinationKind::SameDenominator);
        assert!(rewrite.focus_before.is_some());
        assert!(rewrite.focus_after.is_some());
    }

    #[test]
    fn expands_fraction_subset_with_passthrough_target_aware() {
        let mut simplifier = Simplifier::with_default_rules();
        let source = parse("1 + (a+b+c)/d", &mut simplifier.context).expect("parse");
        let target = parse("1 + a/d + b/d + c/d", &mut simplifier.context).expect("target");
        let rewrite = try_rewrite_fraction_expansion_target_aware(
            &mut simplifier,
            source,
            target,
            crate::runtime::SimplifyOptions::default(),
        )
        .expect("rewrite");

        assert!(rewrite.focus_before.is_some());
        assert!(rewrite.focus_after.is_some());
        let checker =
            cas_math::semantic_equality::SemanticEqualityChecker::new(&simplifier.context);
        assert!(checker.are_equal(rewrite.rewritten, target));
    }

    #[test]
    fn cancels_difference_of_squares_fraction_with_passthrough_target_aware() {
        let mut ctx = Context::new();
        let source = parse("(a^2-b^2)/(a-b)+c", &mut ctx).expect("parse source");
        let target = parse("a+b+c", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_exact_fraction_cancel_target_aware(&mut ctx, source, target)
            .expect("rewrite");

        assert_eq!(rewrite.kind, ExactFractionCancelKind::DifferenceOfSquares);
        assert!(rewrite.focus_before.is_some());
        assert!(rewrite.focus_after.is_some());
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn cancels_difference_of_cubes_fraction_with_passthrough_target_aware() {
        let mut ctx = Context::new();
        let source = parse("(a^3-b^3)/(a-b)+c", &mut ctx).expect("parse source");
        let target = parse("a^2+a*b+b^2+c", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_exact_fraction_cancel_target_aware(&mut ctx, source, target)
            .expect("rewrite");

        assert_eq!(rewrite.kind, ExactFractionCancelKind::SumDiffCubes);
        assert!(rewrite.focus_before.is_some());
        assert!(rewrite.focus_after.is_some());
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn expands_tabulated_fraction_targets_aware() {
        let cases = [
            ("(a+b)/x", "a/x + b/x"),
            ("(x+y)/(x*y)", "1/x + 1/y"),
            ("(a*x+b)/(c*x)", "a/c + b/(c*x)"),
            ("(a*d+b)/d", "a + b/d"),
            ("(a*x+b*y)/(x*y)", "a/y + b/x"),
            ("(a*x+b*y+c*z)/(x*y*z)", "a/(y*z) + b/(x*z) + c/(x*y)"),
            ("(a*x+b*y+c)/(x*y)", "a/y + b/x + c/(x*y)"),
            (
                "(a*x*y+b*y*z+c*x*z+d)/(x*y*z)",
                "a/z + b/x + c/y + d/(x*y*z)",
            ),
            ("(a*x*y+b*x*z+c*y*z+d*x*y*z)/(x*y*z)", "a/z + b/y + c/x + d"),
        ];

        for (source_text, target_text) in cases {
            let mut simplifier = Simplifier::with_default_rules();
            let source = parse(source_text, &mut simplifier.context).expect("source");
            let target = parse(target_text, &mut simplifier.context).expect("target");
            let rewrite = try_rewrite_fraction_expansion_target_aware(
                &mut simplifier,
                source,
                target,
                crate::runtime::SimplifyOptions::default(),
            )
            .expect("rewrite");

            let checker =
                cas_math::semantic_equality::SemanticEqualityChecker::new(&simplifier.context);
            assert!(
                checker.are_equal(rewrite.rewritten, target),
                "expected `{source_text}` -> `{target_text}` to rewrite fraction expansion target-aware",
            );
        }
    }

    #[test]
    fn expands_tabulated_telescoping_fraction_targets_aware() {
        let cases = [
            ("1/(n*(n+2))", "1/2*(1/n - 1/(n+2))"),
            ("1/(n*(n-2))", "1/2*(1/(n-2) - 1/n)"),
            ("1/((2*n+1)*(2*n+3))", "1/2*(1/(2*n+1) - 1/(2*n+3))"),
            ("1/((a*n+b)*(a*n+c))", "1/(c-b)*(1/(a*n+b) - 1/(a*n+c))"),
            ("1/(x^2-1)", "1/2*(1/(x-1) - 1/(x+1))"),
            ("1/(x^2-a^2)", "1/(2*a)*(1/(x-a) - 1/(x+a))"),
        ];

        for (source_text, target_text) in cases {
            let mut simplifier = Simplifier::with_default_rules();
            let source = parse(source_text, &mut simplifier.context).expect("source");
            let target = parse(target_text, &mut simplifier.context).expect("target");
            let rewrite = try_rewrite_fraction_expansion_target_aware(
                &mut simplifier,
                source,
                target,
                crate::runtime::SimplifyOptions::default(),
            )
            .expect("rewrite");

            assert_eq!(
                rewrite.kind,
                FractionExpansionKind::TelescopingFraction,
                "expected `{source_text}` -> `{target_text}` to use telescoping fraction expansion",
            );
            let checker =
                cas_math::semantic_equality::SemanticEqualityChecker::new(&simplifier.context);
            assert!(checker.are_equal(rewrite.rewritten, target));
        }
    }
}
