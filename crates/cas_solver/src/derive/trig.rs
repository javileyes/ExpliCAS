use super::{presentational_target_match, strong_target_match};
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Constant, Expr, ExprId};
use cas_math::expr_nary::{add_terms_signed, Sign};
use cas_math::expr_rewrite::smart_mul;
use cas_math::trig_roots_flatten::flatten_mul_chain;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeriveTrigRewriteKind {
    TrigSpecialValue,
    TanToSinCos,
    RecognizeSinOverCosAsTan,
    ExpandSecToRecipCos,
    ExpandCscToRecipSin,
    ExpandCotToCosSin,
    RecognizeRecipCosAsSec,
    RecognizeRecipSinAsCsc,
    RecognizeCosOverSinAsCot,
    ReciprocalProductToOne,
    RecognizeCosDiffOverSinDiffAsTan,
    PythagoreanChainToOne,
    SecTanPythagoreanToOne,
    CscCotPythagoreanToOne,
    RecognizeSecSquaredMinusOneAsTanSquared,
    ExpandTanSquaredToSecSquaredMinusOne,
    RecognizeCscSquaredMinusOneAsCotSquared,
    ExpandCotSquaredToCscSquaredMinusOne,
    RecognizeOneMinusSecSquaredAsNegTanSquared,
    ExpandNegTanSquaredToOneMinusSecSquared,
    RecognizeOneMinusCscSquaredAsNegCotSquared,
    ExpandNegCotSquaredToOneMinusCscSquared,
    AngleSumDiff,
    TangentAngleSumDiffExpand,
    TangentAngleSumDiffContract,
    RecursiveAngleSumDiff,
    HalfAngleTangent,
    HalfAngleTangentExpandOneMinusCosOverSin,
    HalfAngleTangentExpandSinOverOnePlusCos,
    TangentHalfAngleSubstitutionSin,
    TangentHalfAngleSubstitutionCos,
    HalfAngleSinSquaredExpand,
    HalfAngleCosSquaredExpand,
    SinCosSquareSum,
    SinCosSquareDiff,
    PowerReductionSinFourth,
    PowerReductionCosFourth,
    PowerReductionSinSixth,
    PowerReductionCosSixth,
    PowerReductionSinEighth,
    PowerReductionCosEighth,
    PowerReductionSinTenth,
    PowerReductionCosTenth,
    PowerReductionSinTwelfth,
    PowerReductionCosTwelfth,
    PowerReductionSinFourteenth,
    PowerReductionCosFourteenth,
    PowerReductionSinSixteenth,
    PowerReductionCosSixteenth,
    PowerReductionSinEighteenth,
    PowerReductionCosEighteenth,
    PowerReductionSinTwentieth,
    PowerReductionCosTwentieth,
    PowerReductionSinTwentySecond,
    PowerReductionCosTwentySecond,
    PowerReductionSinHigherEven,
    PowerReductionCosHigherEven,
    PowerReductionSinCosSquares,
    HalfAngleSinSquaredContract,
    HalfAngleCosSquaredContract,
    HalfAngleNegSinSquaredExpand,
    HalfAngleNegCosSquaredExpand,
    HalfAngleNegSinSquaredContract,
    HalfAngleNegCosSquaredContract,
    DoubleAngleSin,
    DoubleAngleCos,
    DoubleAngleTanExpand,
    DoubleAngleTanContract,
    TripleAngleSin,
    TripleAngleCos,
    TripleAngleTan,
    QuintupleAngleSin,
    QuintupleAngleCos,
    DoubleAngleNegSinExpand,
    DoubleAngleNegSinContract,
    DoubleAngleCosOneMinusTwoSinSq,
    DoubleAngleCosTwoCosSqMinusOne,
    DoubleAngleNegCosExpand,
    DoubleAngleNegCosContract,
    DoubleAngleCosOneMinusCosToTwoSinSq,
    DoubleAngleCosOnePlusCosToTwoCosSq,
    DoubleAngleCosMinusOneToNegTwoSinSq,
    DoubleAngleCosTwoSinSqToOneMinusCos,
    DoubleAngleCosTwoCosSqToOnePlusCos,
    DoubleAngleCosNegTwoSinSqToCosMinusOne,
    ExpandSecSquared,
    ExpandCscSquared,
    RecognizeSecSquared,
    RecognizeCscSquared,
    ProductToSumSinCos,
    ProductToSumCosSin,
    ProductToSumCosCos,
    ProductToSumSinSin,
    SumToProductSinSum,
    SumToProductSinDiff,
    SumToProductCosSum,
    SumToProductCosDiff,
    PhaseShiftIdentity,
    TrigOddEvenParity,
    CofunctionIdentity,
    LinearAngleArgumentSimplify,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeriveTrigRewrite {
    pub(crate) rewritten: ExprId,
    pub(crate) kind: DeriveTrigRewriteKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DerivePythagoreanFactorRewrite {
    pub(crate) rewritten: ExprId,
    pub(crate) description: String,
}

pub(crate) fn generate_trig_bridge_rewrites(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Vec<DeriveTrigRewrite> {
    let mut rewrites = Vec::new();

    if let Some(rewrite) = rewrite_angle_sum_diff_bridge_expr(ctx, expr, false) {
        push_unique_trig_bridge_rewrite(&mut rewrites, rewrite);
    }

    if let Some(positive_expr) = strip_unit_negation(ctx, expr) {
        if let Some(rewrite) = rewrite_angle_sum_diff_bridge_expr(ctx, positive_expr, true) {
            push_unique_trig_bridge_rewrite(&mut rewrites, rewrite);
        }
    }

    if let Some(rewrite) =
        cas_math::trig_sum_product_support::try_rewrite_product_to_sum_expr(ctx, expr)
    {
        let normalized = normalize_trig_bridge_expr(ctx, rewrite.rewritten);
        if compare_expr(ctx, normalized, rewrite.rewritten) != Ordering::Equal {
            push_unique_trig_bridge_rewrite(
                &mut rewrites,
                DeriveTrigRewrite {
                    rewritten: normalized,
                    kind: map_product_to_sum_kind(rewrite.kind),
                },
            );
        }

        push_unique_trig_bridge_rewrite(
            &mut rewrites,
            DeriveTrigRewrite {
                rewritten: rewrite.rewritten,
                kind: map_product_to_sum_kind(rewrite.kind),
            },
        );
    }

    for rewrite in generate_phase_shift_bridge_rewrites(ctx, expr) {
        push_unique_trig_bridge_rewrite(&mut rewrites, rewrite);
    }

    if let Some(rewrite) = rewrite_linear_angle_argument_bridge_expr(ctx, expr) {
        push_unique_trig_bridge_rewrite(&mut rewrites, rewrite);
    }

    rewrites
}

pub(crate) fn generate_trig_additive_term_bridge_rewrites(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Vec<DeriveTrigRewrite> {
    let terms = add_terms_signed(ctx, expr);
    if terms.len() <= 1 {
        return Vec::new();
    }

    let mut rewrites = Vec::new();
    generate_combined_additive_triple_angle_rewrites(ctx, &terms, &mut rewrites);
    generate_combined_additive_phase_shift_rewrites(ctx, &terms, &mut rewrites);

    for (index, (term, sign)) in terms.iter().enumerate() {
        for candidate in generate_trig_bridge_rewrites(ctx, *term) {
            let signed_rewritten_term = apply_sign_to_term(ctx, candidate.rewritten, *sign);
            let rewritten = rebuild_additive_terms_with_rewritten_term(
                ctx,
                &terms,
                index,
                signed_rewritten_term,
            );
            push_unique_trig_bridge_rewrite(
                &mut rewrites,
                DeriveTrigRewrite {
                    rewritten,
                    kind: candidate.kind,
                },
            );
        }

        for candidate in generate_trig_multi_angle_term_rewrites(ctx, *term) {
            let signed_rewritten_term = apply_sign_to_term(ctx, candidate.rewritten, *sign);
            let rewritten = rebuild_additive_terms_with_rewritten_term(
                ctx,
                &terms,
                index,
                signed_rewritten_term,
            );
            push_unique_trig_bridge_rewrite(
                &mut rewrites,
                DeriveTrigRewrite {
                    rewritten,
                    kind: candidate.kind,
                },
            );
        }
    }

    rewrites
}

pub(crate) fn should_try_trig_planner_before_simplify(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> bool {
    if !contains_circular_trig_fn(ctx, expr) && !contains_circular_trig_fn(ctx, target_expr) {
        return false;
    }

    if try_rewrite_trig_expansion(ctx, expr, target_expr).is_some()
        || try_rewrite_trig_contraction_target_aware(ctx, expr, target_expr).is_some()
    {
        return false;
    }

    if has_direct_additive_phase_shift_chain(ctx, expr, target_expr) {
        return false;
    }

    let mut rewrites = generate_trig_bridge_rewrites(ctx, expr);
    rewrites.extend(generate_trig_additive_term_bridge_rewrites(ctx, expr));
    if rewrites.is_empty() {
        return false;
    }

    for rewrite in &rewrites {
        if strong_target_match(ctx, rewrite.rewritten, target_expr)
            || (rewrite.kind.rule_name() == "Phase Shift Identity"
                && phase_shift_target_match(ctx, rewrite.rewritten, target_expr))
        {
            return false;
        }

        if try_rewrite_trig_expansion(ctx, rewrite.rewritten, target_expr).is_some()
            || try_rewrite_trig_contraction_target_aware(ctx, rewrite.rewritten, target_expr)
                .is_some()
        {
            return false;
        }
    }

    true
}

fn has_direct_additive_phase_shift_chain(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> bool {
    generate_trig_additive_term_bridge_rewrites(ctx, expr)
        .into_iter()
        .filter(|rewrite| rewrite.kind.rule_name() == "Phase Shift Identity")
        .any(|first| {
            generate_trig_additive_term_bridge_rewrites(ctx, first.rewritten)
                .into_iter()
                .filter(|rewrite| rewrite.kind.rule_name() == "Phase Shift Identity")
                .any(|second| phase_shift_target_match(ctx, second.rewritten, target_expr))
        })
}

fn contains_circular_trig_fn(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            Expr::Function(fn_id, args) => {
                if matches!(
                    ctx.builtin_of(*fn_id),
                    Some(
                        BuiltinFn::Sin
                            | BuiltinFn::Cos
                            | BuiltinFn::Tan
                            | BuiltinFn::Sec
                            | BuiltinFn::Csc
                            | BuiltinFn::Cot
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

fn push_unique_trig_bridge_rewrite(
    rewrites: &mut Vec<DeriveTrigRewrite>,
    rewrite: DeriveTrigRewrite,
) {
    if rewrites.iter().any(|existing| existing == &rewrite) {
        return;
    }

    rewrites.push(rewrite);
}

fn generate_trig_multi_angle_term_rewrites(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Vec<DeriveTrigRewrite> {
    let mut rewrites = Vec::new();

    if let Some(rewrite) =
        cas_math::trig_multi_angle_support::try_rewrite_double_angle_function_expr(ctx, expr)
    {
        if let Some(kind) = map_trig_multi_angle_bridge_kind(rewrite.kind) {
            rewrites.push(DeriveTrigRewrite {
                rewritten: rewrite.rewritten,
                kind,
            });
        }
    }

    if let Some(rewrite) =
        cas_math::trig_multi_angle_support::try_rewrite_triple_angle_expr(ctx, expr)
    {
        if let Some(kind) = map_trig_multi_angle_bridge_kind(rewrite.kind) {
            rewrites.push(DeriveTrigRewrite {
                rewritten: rewrite.rewritten,
                kind,
            });
        }
    }

    if let Some(rewrite) =
        cas_math::trig_multi_angle_support::try_rewrite_quintuple_angle_expr(ctx, expr)
    {
        if let Some(kind) = map_trig_multi_angle_bridge_kind(rewrite.kind) {
            rewrites.push(DeriveTrigRewrite {
                rewritten: rewrite.rewritten,
                kind,
            });
        }
    }

    if let Some(rewrite) =
        cas_math::trig_multi_angle_support::try_rewrite_recursive_trig_expansion_expr(ctx, expr)
    {
        rewrites.push(DeriveTrigRewrite {
            rewritten: rewrite.rewritten,
            kind: DeriveTrigRewriteKind::RecursiveAngleSumDiff,
        });
    }

    rewrites
}

fn map_trig_multi_angle_bridge_kind(
    kind: cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind,
) -> Option<DeriveTrigRewriteKind> {
    match kind {
        cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::DoubleSin => {
            Some(DeriveTrigRewriteKind::DoubleAngleSin)
        }
        cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::DoubleCos => {
            Some(DeriveTrigRewriteKind::DoubleAngleCos)
        }
        cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::TripleSin => {
            Some(DeriveTrigRewriteKind::TripleAngleSin)
        }
        cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::TripleCos => {
            Some(DeriveTrigRewriteKind::TripleAngleCos)
        }
        cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::TripleTan => {
            Some(DeriveTrigRewriteKind::TripleAngleTan)
        }
        cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::QuintupleSin => {
            Some(DeriveTrigRewriteKind::QuintupleAngleSin)
        }
        cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::QuintupleCos => {
            Some(DeriveTrigRewriteKind::QuintupleAngleCos)
        }
        _ => None,
    }
}

fn rebuild_additive_terms_with_rewritten_term(
    ctx: &mut cas_ast::Context,
    terms: &[(ExprId, Sign)],
    rewritten_index: usize,
    rewritten_term: ExprId,
) -> ExprId {
    let mut rebuilt_terms = Vec::with_capacity(terms.len());
    for (index, (term, sign)) in terms.iter().enumerate() {
        let current = if index == rewritten_index {
            rewritten_term
        } else {
            apply_sign_to_term(ctx, *term, *sign)
        };
        rebuilt_terms.push(current);
    }

    let mut iter = rebuilt_terms.into_iter();
    let Some(first) = iter.next() else {
        return ctx.num(0);
    };

    iter.fold(first, |acc, term| ctx.add(Expr::Add(acc, term)))
}

fn rebuild_additive_terms_with_combined_terms(
    ctx: &mut cas_ast::Context,
    terms: &[(ExprId, Sign)],
    first_index: usize,
    second_index: usize,
    combined_term: ExprId,
) -> ExprId {
    let keep_index = first_index.min(second_index);
    let skip_index = first_index.max(second_index);
    let mut rebuilt_terms = Vec::with_capacity(terms.len().saturating_sub(1));

    for (index, (term, sign)) in terms.iter().enumerate() {
        if index == keep_index {
            rebuilt_terms.push(combined_term);
            continue;
        }
        if index == skip_index {
            continue;
        }

        rebuilt_terms.push(apply_sign_to_term(ctx, *term, *sign));
    }

    let mut iter = rebuilt_terms.into_iter();
    let Some(first) = iter.next() else {
        return ctx.num(0);
    };

    iter.fold(first, |acc, term| ctx.add(Expr::Add(acc, term)))
}

fn apply_sign_to_term(ctx: &mut cas_ast::Context, term: ExprId, sign: Sign) -> ExprId {
    match sign {
        Sign::Pos => term,
        Sign::Neg => ctx.add(Expr::Neg(term)),
    }
}

fn compare_signed_additive_term(
    ctx: &mut cas_ast::Context,
    lhs: (ExprId, Sign),
    rhs: (ExprId, Sign),
) -> Ordering {
    let lhs_sign_rank = match lhs.1 {
        Sign::Pos => 0,
        Sign::Neg => 1,
    };
    let rhs_sign_rank = match rhs.1 {
        Sign::Pos => 0,
        Sign::Neg => 1,
    };

    lhs_sign_rank
        .cmp(&rhs_sign_rank)
        .then_with(|| compare_expr(ctx, lhs.0, rhs.0))
}

fn signed_additive_passthrough_terms_match(
    ctx: &mut cas_ast::Context,
    lhs_terms: &[(ExprId, Sign)],
    lhs_excluded_index: usize,
    rhs_terms: &[(ExprId, Sign)],
    rhs_excluded_index: usize,
) -> bool {
    let lhs = lhs_terms
        .iter()
        .enumerate()
        .filter_map(|(index, term)| (index != lhs_excluded_index).then_some(*term))
        .collect::<Vec<_>>();
    let rhs = rhs_terms
        .iter()
        .enumerate()
        .filter_map(|(index, term)| (index != rhs_excluded_index).then_some(*term))
        .collect::<Vec<_>>();

    signed_additive_terms_match(ctx, &lhs, &rhs)
}

fn collect_signed_passthrough_terms(
    terms: &[(ExprId, Sign)],
    included_mask: usize,
) -> Vec<(ExprId, Sign)> {
    terms
        .iter()
        .enumerate()
        .filter_map(|(index, term)| ((included_mask & (1usize << index)) == 0).then_some(*term))
        .collect()
}

fn signed_additive_terms_match(
    ctx: &mut cas_ast::Context,
    lhs_terms: &[(ExprId, Sign)],
    rhs_terms: &[(ExprId, Sign)],
) -> bool {
    let mut lhs = lhs_terms.to_vec();
    let mut rhs = rhs_terms.to_vec();

    if lhs.len() != rhs.len() {
        return false;
    }

    lhs.sort_by(|left, right| compare_signed_additive_term(ctx, *left, *right));
    rhs.sort_by(|left, right| compare_signed_additive_term(ctx, *left, *right));

    lhs.iter().zip(rhs.iter()).all(|(left, right)| {
        left.1 == right.1 && compare_expr(ctx, left.0, right.0) == Ordering::Equal
    })
}

fn build_additive_expr_from_terms_and_signs(
    ctx: &mut cas_ast::Context,
    terms: impl IntoIterator<Item = (ExprId, Sign)>,
) -> ExprId {
    let mut iter = terms.into_iter();
    let Some((first_term, first_sign)) = iter.next() else {
        return ctx.num(0);
    };

    let mut acc = apply_sign_to_term(ctx, first_term, first_sign);
    for (term, sign) in iter {
        acc = match sign {
            Sign::Pos => ctx.add(Expr::Add(acc, term)),
            Sign::Neg => ctx.add(Expr::Sub(acc, term)),
        };
    }

    acc
}

fn generate_combined_additive_triple_angle_rewrites(
    ctx: &mut cas_ast::Context,
    terms: &[(ExprId, Sign)],
    rewrites: &mut Vec<DeriveTrigRewrite>,
) {
    for first_index in 0..terms.len() {
        for second_index in (first_index + 1)..terms.len() {
            let Some((kind, trig_fn, base_arg, base_coeff, triple_coeff)) =
                combined_triple_angle_data(ctx, terms[first_index], terms[second_index])
            else {
                continue;
            };

            let combined_term = build_combined_triple_angle_polynomial(
                ctx,
                trig_fn,
                base_arg,
                base_coeff,
                triple_coeff,
            );
            let rewritten = rebuild_additive_terms_with_combined_terms(
                ctx,
                terms,
                first_index,
                second_index,
                combined_term,
            );
            push_unique_trig_bridge_rewrite(rewrites, DeriveTrigRewrite { rewritten, kind });

            let mixed_term = build_mixed_triple_angle_polynomial(
                ctx,
                trig_fn,
                base_arg,
                base_coeff,
                triple_coeff,
            );
            let mixed_rewritten = rebuild_additive_terms_with_combined_terms(
                ctx,
                terms,
                first_index,
                second_index,
                mixed_term,
            );
            push_unique_trig_bridge_rewrite(
                rewrites,
                DeriveTrigRewrite {
                    rewritten: mixed_rewritten,
                    kind,
                },
            );
        }
    }
}

fn generate_combined_additive_phase_shift_rewrites(
    ctx: &mut cas_ast::Context,
    terms: &[(ExprId, Sign)],
    rewrites: &mut Vec<DeriveTrigRewrite>,
) {
    for first_index in 0..terms.len() {
        for second_index in (first_index + 1)..terms.len() {
            let first_signed = apply_sign_to_term(ctx, terms[first_index].0, terms[first_index].1);
            let second_signed =
                apply_sign_to_term(ctx, terms[second_index].0, terms[second_index].1);
            let pair_expr = ctx.add(Expr::Add(first_signed, second_signed));

            for candidate in generate_phase_shift_bridge_rewrites(ctx, pair_expr) {
                let rewritten = rebuild_additive_terms_with_combined_terms(
                    ctx,
                    terms,
                    first_index,
                    second_index,
                    candidate.rewritten,
                );
                push_unique_trig_bridge_rewrite(
                    rewrites,
                    DeriveTrigRewrite {
                        rewritten,
                        kind: candidate.kind,
                    },
                );
            }
        }
    }
}

fn combined_triple_angle_data(
    ctx: &mut cas_ast::Context,
    first: (ExprId, Sign),
    second: (ExprId, Sign),
) -> Option<(DeriveTrigRewriteKind, BuiltinFn, ExprId, i64, i64)> {
    let (first_term, first_sign) = first;
    let (second_term, second_sign) = second;
    let (first_fn, first_arg) = sin_or_cos_call_arg(ctx, first_term)?;
    let (second_fn, second_arg) = sin_or_cos_call_arg(ctx, second_term)?;
    if first_fn != second_fn {
        return None;
    }

    if let Some(base_arg) = triple_scaled_arg_base(ctx, second_arg) {
        if compare_expr(ctx, first_arg, base_arg) == Ordering::Equal {
            return Some((
                triple_angle_kind_for_fn(first_fn)?,
                first_fn,
                base_arg,
                sign_to_coeff(first_sign),
                sign_to_coeff(second_sign),
            ));
        }
    }

    if let Some(base_arg) = triple_scaled_arg_base(ctx, first_arg) {
        if compare_expr(ctx, second_arg, base_arg) == Ordering::Equal {
            return Some((
                triple_angle_kind_for_fn(first_fn)?,
                first_fn,
                base_arg,
                sign_to_coeff(second_sign),
                sign_to_coeff(first_sign),
            ));
        }
    }

    None
}

fn build_combined_triple_angle_polynomial(
    ctx: &mut cas_ast::Context,
    trig_fn: BuiltinFn,
    base_arg: ExprId,
    base_coeff: i64,
    triple_coeff: i64,
) -> ExprId {
    let trig_call = ctx.call_builtin(trig_fn, vec![base_arg]);
    let cubic_term = pow3(ctx, trig_call);
    let (cubic_coeff, linear_coeff) = match trig_fn {
        BuiltinFn::Cos => (4 * triple_coeff, base_coeff - 3 * triple_coeff),
        BuiltinFn::Sin => (-4 * triple_coeff, base_coeff + 3 * triple_coeff),
        _ => unreachable!("only sin/cos should reach combined triple-angle bridge"),
    };

    build_scaled_additive_terms(ctx, &[(cubic_coeff, cubic_term), (linear_coeff, trig_call)])
}

fn build_mixed_triple_angle_polynomial(
    ctx: &mut cas_ast::Context,
    trig_fn: BuiltinFn,
    base_arg: ExprId,
    base_coeff: i64,
    triple_coeff: i64,
) -> ExprId {
    let sin = ctx.call_builtin(BuiltinFn::Sin, vec![base_arg]);
    let cos = ctx.call_builtin(BuiltinFn::Cos, vec![base_arg]);

    match trig_fn {
        BuiltinFn::Sin => {
            let cos_sq = pow2(ctx, cos);
            let mixed_term = smart_mul(ctx, cos_sq, sin);
            let mixed_coeff = 4 * triple_coeff;
            let linear_coeff = base_coeff - triple_coeff;
            build_scaled_additive_terms(ctx, &[(mixed_coeff, mixed_term), (linear_coeff, sin)])
        }
        BuiltinFn::Cos => {
            let sin_sq = pow2(ctx, sin);
            let mixed_term = smart_mul(ctx, sin_sq, cos);
            let mixed_coeff = -4 * triple_coeff;
            let linear_coeff = base_coeff + triple_coeff;
            build_scaled_additive_terms(ctx, &[(mixed_coeff, mixed_term), (linear_coeff, cos)])
        }
        _ => unreachable!("only sin/cos should reach mixed combined triple-angle bridge"),
    }
}

fn build_scaled_additive_terms(
    ctx: &mut cas_ast::Context,
    scaled_terms: &[(i64, ExprId)],
) -> ExprId {
    let mut acc = None;

    for (coeff, term) in scaled_terms.iter().copied() {
        if coeff == 0 {
            continue;
        }

        let scaled = scale_term_by_small_integer(ctx, term, coeff.abs());
        acc = Some(match acc {
            None if coeff < 0 => ctx.add(Expr::Neg(scaled)),
            None => scaled,
            Some(existing) if coeff < 0 => ctx.add(Expr::Sub(existing, scaled)),
            Some(existing) => ctx.add(Expr::Add(existing, scaled)),
        });
    }

    acc.unwrap_or_else(|| ctx.num(0))
}

fn scale_term_by_small_integer(ctx: &mut cas_ast::Context, expr: ExprId, coeff: i64) -> ExprId {
    match coeff {
        0 => ctx.num(0),
        1 => expr,
        _ => {
            let coeff_expr = ctx.num(coeff);
            smart_mul(ctx, coeff_expr, expr)
        }
    }
}

fn sin_or_cos_call_arg(ctx: &cas_ast::Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
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

fn triple_scaled_arg_base(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };

    if is_small_integer(ctx, *left, 3) {
        Some(*right)
    } else if is_small_integer(ctx, *right, 3) {
        Some(*left)
    } else {
        None
    }
}

fn triple_angle_kind_for_fn(trig_fn: BuiltinFn) -> Option<DeriveTrigRewriteKind> {
    match trig_fn {
        BuiltinFn::Sin => Some(DeriveTrigRewriteKind::TripleAngleSin),
        BuiltinFn::Cos => Some(DeriveTrigRewriteKind::TripleAngleCos),
        _ => None,
    }
}

fn sign_to_coeff(sign: Sign) -> i64 {
    match sign {
        Sign::Pos => 1,
        Sign::Neg => -1,
    }
}

impl DeriveTrigRewriteKind {
    pub(crate) fn description(self) -> &'static str {
        match self {
            Self::TrigSpecialValue => "Evaluate a trigonometric function at a special input",
            Self::TanToSinCos => "Expand tangent to sine over cosine",
            Self::RecognizeSinOverCosAsTan => "Recognize sin(u) / cos(u) as tan(u)",
            Self::ExpandSecToRecipCos => "Expand sec(u) as 1 / cos(u)",
            Self::ExpandCscToRecipSin => "Expand csc(u) as 1 / sin(u)",
            Self::ExpandCotToCosSin => "Expand cot(u) as cos(u) / sin(u)",
            Self::RecognizeRecipCosAsSec => "Recognize 1 / cos(u) as sec(u)",
            Self::RecognizeRecipSinAsCsc => "Recognize 1 / sin(u) as csc(u)",
            Self::RecognizeCosOverSinAsCot => "Recognize cos(u) / sin(u) as cot(u)",
            Self::RecognizeCosDiffOverSinDiffAsTan => {
                "Recognize a cosine/sine difference quotient as tan((A+B)/2)"
            }
            Self::ReciprocalProductToOne => "Recognize tan(u) · cot(u) = 1",
            Self::PythagoreanChainToOne => "Recognize sin²(u) + cos²(u) = 1",
            Self::SecTanPythagoreanToOne => "Recognize sec²(u) - tan²(u) = 1",
            Self::CscCotPythagoreanToOne => "Recognize csc²(u) - cot²(u) = 1",
            Self::RecognizeSecSquaredMinusOneAsTanSquared => "Recognize sec²(u) - 1 as tan²(u)",
            Self::ExpandTanSquaredToSecSquaredMinusOne => "Expand tan²(u) as sec²(u) - 1",
            Self::RecognizeCscSquaredMinusOneAsCotSquared => "Recognize csc²(u) - 1 as cot²(u)",
            Self::ExpandCotSquaredToCscSquaredMinusOne => "Expand cot²(u) as csc²(u) - 1",
            Self::RecognizeOneMinusSecSquaredAsNegTanSquared => "Recognize 1 - sec²(u) as -tan²(u)",
            Self::ExpandNegTanSquaredToOneMinusSecSquared => "Expand -tan²(u) as 1 - sec²(u)",
            Self::RecognizeOneMinusCscSquaredAsNegCotSquared => "Recognize 1 - csc²(u) as -cot²(u)",
            Self::ExpandNegCotSquaredToOneMinusCscSquared => "Expand -cot²(u) as 1 - csc²(u)",
            Self::AngleSumDiff => "Expand or contract an angle sum/difference trig identity",
            Self::TangentAngleSumDiffExpand => "Expand tangent angle sum/difference form",
            Self::TangentAngleSumDiffContract => "Recognize tangent angle sum/difference form",
            Self::RecursiveAngleSumDiff => {
                "Expand a trig multiple angle recursively via angle addition"
            }
            Self::HalfAngleTangent => "Contract half-angle tangent quotient",
            Self::HalfAngleTangentExpandOneMinusCosOverSin => {
                "Expand tan(u) as (1 - cos(2u))/sin(2u)"
            }
            Self::HalfAngleTangentExpandSinOverOnePlusCos => {
                "Expand tan(u) as sin(2u)/(1 + cos(2u))"
            }
            Self::TangentHalfAngleSubstitutionSin => {
                "Rewrite sin(u) using the tangent half-angle substitution"
            }
            Self::TangentHalfAngleSubstitutionCos => {
                "Rewrite cos(u) using the tangent half-angle substitution"
            }
            Self::HalfAngleSinSquaredExpand => "Expand sin²(u) as (1 - cos(2u))/2",
            Self::HalfAngleCosSquaredExpand => "Expand cos²(u) as (1 + cos(2u))/2",
            Self::SinCosSquareSum => "Expand (sin(u) + cos(u))² as 1 + sin(2u)",
            Self::SinCosSquareDiff => "Expand (sin(u) - cos(u))² as 1 - sin(2u)",
            Self::PowerReductionSinFourth => "Reduce sin⁴(u) using power-reduction identities",
            Self::PowerReductionCosFourth => "Reduce cos⁴(u) using power-reduction identities",
            Self::PowerReductionSinSixth => "Reduce sin⁶(u) using power-reduction identities",
            Self::PowerReductionCosSixth => "Reduce cos⁶(u) using power-reduction identities",
            Self::PowerReductionSinEighth => "Reduce sin⁸(u) using power-reduction identities",
            Self::PowerReductionCosEighth => "Reduce cos⁸(u) using power-reduction identities",
            Self::PowerReductionSinTenth => "Reduce sin¹⁰(u) using power-reduction identities",
            Self::PowerReductionCosTenth => "Reduce cos¹⁰(u) using power-reduction identities",
            Self::PowerReductionSinTwelfth => "Reduce sin¹²(u) using power-reduction identities",
            Self::PowerReductionCosTwelfth => "Reduce cos¹²(u) using power-reduction identities",
            Self::PowerReductionSinFourteenth => "Reduce sin¹⁴(u) using power-reduction identities",
            Self::PowerReductionCosFourteenth => "Reduce cos¹⁴(u) using power-reduction identities",
            Self::PowerReductionSinSixteenth => "Reduce sin¹⁶(u) using power-reduction identities",
            Self::PowerReductionCosSixteenth => "Reduce cos¹⁶(u) using power-reduction identities",
            Self::PowerReductionSinEighteenth => "Reduce sin¹⁸(u) using power-reduction identities",
            Self::PowerReductionCosEighteenth => "Reduce cos¹⁸(u) using power-reduction identities",
            Self::PowerReductionSinTwentieth => "Reduce sin²⁰(u) using power-reduction identities",
            Self::PowerReductionCosTwentieth => "Reduce cos²⁰(u) using power-reduction identities",
            Self::PowerReductionSinTwentySecond => {
                "Reduce sin²²(u) using power-reduction identities"
            }
            Self::PowerReductionCosTwentySecond => {
                "Reduce cos²²(u) using power-reduction identities"
            }
            Self::PowerReductionSinHigherEven => {
                "Reduce higher even powers of sin(u) using power-reduction identities"
            }
            Self::PowerReductionCosHigherEven => {
                "Reduce higher even powers of cos(u) using power-reduction identities"
            }
            Self::PowerReductionSinCosSquares => {
                "Reduce sin²(u)·cos²(u) using power-reduction identities"
            }
            Self::HalfAngleSinSquaredContract => "Recognize (1 - cos(2u))/2 as sin²(u)",
            Self::HalfAngleCosSquaredContract => "Recognize (1 + cos(2u))/2 as cos²(u)",
            Self::HalfAngleNegSinSquaredExpand => "Expand -sin²(u) using the half-angle identity",
            Self::HalfAngleNegCosSquaredExpand => "Expand -cos²(u) using the half-angle identity",
            Self::HalfAngleNegSinSquaredContract => {
                "Recognize a negated half-angle sine-square form"
            }
            Self::HalfAngleNegCosSquaredContract => {
                "Recognize a negated half-angle cosine-square form"
            }
            Self::DoubleAngleSin => "Expand double-angle sine",
            Self::DoubleAngleCos => "Expand double-angle cosine",
            Self::DoubleAngleTanExpand => "Expand tangent double-angle form",
            Self::DoubleAngleTanContract => "Recognize tangent double-angle form",
            Self::TripleAngleSin => "Expand or contract sine triple-angle form",
            Self::TripleAngleCos => "Expand or contract cosine triple-angle form",
            Self::TripleAngleTan => "Expand or contract tangent triple-angle form",
            Self::QuintupleAngleSin => "Expand or contract sine quintuple-angle form",
            Self::QuintupleAngleCos => "Expand or contract cosine quintuple-angle form",
            Self::DoubleAngleNegSinExpand => "Expand -sin(2u) using the double-angle identity",
            Self::DoubleAngleNegSinContract => "Recognize a negated sine double-angle form",
            Self::DoubleAngleCosOneMinusTwoSinSq => "Expand cosine double-angle as 1 - 2·sin(u)^2",
            Self::DoubleAngleCosTwoCosSqMinusOne => "Expand cosine double-angle as 2·cos(u)^2 - 1",
            Self::DoubleAngleNegCosExpand => "Expand -cos(2u) using the double-angle identity",
            Self::DoubleAngleNegCosContract => "Recognize a negated cosine double-angle form",
            Self::DoubleAngleCosOneMinusCosToTwoSinSq => "Recognize 1 - cos(2u) as 2·sin(u)^2",
            Self::DoubleAngleCosOnePlusCosToTwoCosSq => "Recognize 1 + cos(2u) as 2·cos(u)^2",
            Self::DoubleAngleCosMinusOneToNegTwoSinSq => "Recognize cos(2u) - 1 as -2·sin(u)^2",
            Self::DoubleAngleCosTwoSinSqToOneMinusCos => "Expand 2·sin(u)^2 as 1 - cos(2u)",
            Self::DoubleAngleCosTwoCosSqToOnePlusCos => "Expand 2·cos(u)^2 as 1 + cos(2u)",
            Self::DoubleAngleCosNegTwoSinSqToCosMinusOne => "Expand -2·sin(u)^2 as cos(2u) - 1",
            Self::ExpandSecSquared => "Expand sec²(u) as 1 + tan(u)^2",
            Self::ExpandCscSquared => "Expand csc²(u) as 1 + cot(u)^2",
            Self::RecognizeSecSquared => "Recognize 1 + tan²(u) as sec²(u)",
            Self::RecognizeCscSquared => "Recognize 1 + cot²(u) as csc²(u)",
            Self::ProductToSumSinCos => "Expand 2·sin(A)·cos(B) into sin(A+B) + sin(A-B)",
            Self::ProductToSumCosSin => "Expand 2·cos(A)·sin(B) into sin(A+B) - sin(A-B)",
            Self::ProductToSumCosCos => "Expand 2·cos(A)·cos(B) into cos(A+B) + cos(A-B)",
            Self::ProductToSumSinSin => "Expand 2·sin(A)·sin(B) into cos(A-B) - cos(A+B)",
            Self::SumToProductSinSum => "Expand sine sum to product",
            Self::SumToProductSinDiff => "Expand sine difference to product",
            Self::SumToProductCosSum => "Expand cosine sum to product",
            Self::SumToProductCosDiff => "Expand cosine difference to product",
            Self::PhaseShiftIdentity => {
                "Rewrite exact sine/cosine linear combinations using a phase shift"
            }
            Self::TrigOddEvenParity => "Apply a trigonometric odd/even parity identity",
            Self::CofunctionIdentity => "Apply a sine/cosine cofunction identity",
            Self::LinearAngleArgumentSimplify => {
                "Simplify linear angle arguments inside trig functions"
            }
        }
    }

    pub(crate) fn rule_name(self) -> &'static str {
        match self {
            Self::TrigSpecialValue => "Evaluate Trigonometric Functions",
            Self::TanToSinCos => "Trig Expansion",
            Self::RecognizeSinOverCosAsTan | Self::RecognizeCosDiffOverSinDiffAsTan => {
                "Trig Quotient"
            }
            Self::ExpandSecToRecipCos
            | Self::ExpandCscToRecipSin
            | Self::ExpandCotToCosSin
            | Self::RecognizeRecipCosAsSec
            | Self::RecognizeRecipSinAsCsc
            | Self::RecognizeCosOverSinAsCot => "Reciprocal Trig Identity",
            Self::PythagoreanChainToOne => "Pythagorean Chain Identity",
            Self::ReciprocalProductToOne => "Reciprocal Product Identity",
            Self::SecTanPythagoreanToOne
            | Self::CscCotPythagoreanToOne
            | Self::RecognizeSecSquaredMinusOneAsTanSquared
            | Self::ExpandTanSquaredToSecSquaredMinusOne
            | Self::RecognizeCscSquaredMinusOneAsCotSquared
            | Self::RecognizeOneMinusSecSquaredAsNegTanSquared
            | Self::ExpandNegTanSquaredToOneMinusSecSquared
            | Self::RecognizeOneMinusCscSquaredAsNegCotSquared
            | Self::ExpandNegCotSquaredToOneMinusCscSquared
            | Self::ExpandCotSquaredToCscSquaredMinusOne => "Reciprocal Pythagorean Identity",
            Self::AngleSumDiff | Self::RecursiveAngleSumDiff => "Angle Sum/Diff Identity",
            Self::TangentAngleSumDiffExpand | Self::TangentAngleSumDiffContract => {
                "Tangent Angle Sum/Diff Identity"
            }
            Self::HalfAngleTangent
            | Self::HalfAngleTangentExpandOneMinusCosOverSin
            | Self::HalfAngleTangentExpandSinOverOnePlusCos
            | Self::TangentHalfAngleSubstitutionSin
            | Self::TangentHalfAngleSubstitutionCos => "Half-Angle Tangent Identity",
            Self::HalfAngleSinSquaredExpand
            | Self::HalfAngleCosSquaredExpand
            | Self::HalfAngleSinSquaredContract
            | Self::HalfAngleCosSquaredContract
            | Self::HalfAngleNegSinSquaredExpand
            | Self::HalfAngleNegCosSquaredExpand
            | Self::HalfAngleNegSinSquaredContract
            | Self::HalfAngleNegCosSquaredContract => "Half-Angle Square Identity",
            Self::SinCosSquareSum | Self::SinCosSquareDiff => "Trig Square Identity",
            Self::PowerReductionSinFourth
            | Self::PowerReductionCosFourth
            | Self::PowerReductionSinSixth
            | Self::PowerReductionCosSixth
            | Self::PowerReductionSinEighth
            | Self::PowerReductionCosEighth
            | Self::PowerReductionSinTenth
            | Self::PowerReductionCosTenth
            | Self::PowerReductionSinTwelfth
            | Self::PowerReductionCosTwelfth
            | Self::PowerReductionSinFourteenth
            | Self::PowerReductionCosFourteenth
            | Self::PowerReductionSinSixteenth
            | Self::PowerReductionCosSixteenth
            | Self::PowerReductionSinEighteenth
            | Self::PowerReductionCosEighteenth
            | Self::PowerReductionSinTwentieth
            | Self::PowerReductionCosTwentieth
            | Self::PowerReductionSinTwentySecond
            | Self::PowerReductionCosTwentySecond
            | Self::PowerReductionSinHigherEven
            | Self::PowerReductionCosHigherEven
            | Self::PowerReductionSinCosSquares => "Power Reduction Identity",
            Self::DoubleAngleSin
            | Self::DoubleAngleNegSinExpand
            | Self::DoubleAngleNegSinContract => "Double Angle Expansion",
            Self::DoubleAngleCos
            | Self::DoubleAngleCosOneMinusTwoSinSq
            | Self::DoubleAngleCosTwoCosSqMinusOne
            | Self::DoubleAngleNegCosExpand
            | Self::DoubleAngleNegCosContract
            | Self::DoubleAngleCosOneMinusCosToTwoSinSq
            | Self::DoubleAngleCosOnePlusCosToTwoCosSq
            | Self::DoubleAngleCosMinusOneToNegTwoSinSq
            | Self::DoubleAngleCosTwoSinSqToOneMinusCos
            | Self::DoubleAngleCosTwoCosSqToOnePlusCos
            | Self::DoubleAngleCosNegTwoSinSqToCosMinusOne => "Double Angle Expansion",
            Self::DoubleAngleTanExpand | Self::DoubleAngleTanContract => {
                "Tangent Double-Angle Identity"
            }
            Self::TripleAngleSin | Self::TripleAngleCos | Self::TripleAngleTan => {
                "Triple Angle Expansion"
            }
            Self::QuintupleAngleSin | Self::QuintupleAngleCos => "Quintuple Angle Identity",
            Self::ExpandSecSquared => "Expand Secant Squared",
            Self::ExpandCscSquared => "Expand Cosecant Squared",
            Self::RecognizeSecSquared => "Recognize Secant Squared",
            Self::RecognizeCscSquared => "Recognize Cosecant Squared",
            Self::ProductToSumSinCos
            | Self::ProductToSumCosSin
            | Self::ProductToSumCosCos
            | Self::ProductToSumSinSin => "Product-to-Sum Identity",
            Self::SumToProductSinSum
            | Self::SumToProductSinDiff
            | Self::SumToProductCosSum
            | Self::SumToProductCosDiff => "Sum-to-Product Identity",
            Self::PhaseShiftIdentity => "Phase Shift Identity",
            Self::TrigOddEvenParity => "Trig Parity (Odd/Even)",
            Self::CofunctionIdentity => "Cofunction Identity",
            Self::LinearAngleArgumentSimplify => "Linear Angle Simplification",
        }
    }
}

pub(crate) fn try_rewrite_trig_expansion(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    if let Some(rewrite) = try_rewrite_trig_odd_even_parity_target_aware(ctx, expr, target_expr) {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_rewrite_cofunction_phase_shift_target_aware(ctx, expr, target_expr) {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_rewrite_phase_shift_expansion_target_aware(ctx, expr, target_expr) {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_normalized_double_angle_expansion_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_negated_angle_sum_diff_expansion_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_rewrite_angle_sum_diff_expansion_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_tangent_angle_sum_diff_expansion_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_negated_half_angle_tangent_expansion_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_half_angle_tangent_expansion_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_tangent_half_angle_substitution_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_tangent_double_angle_expansion_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_negated_reciprocal_trig_expansion_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_reciprocal_trig_expansion_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_rewrite_sin_cos_binomial_square_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_half_angle_square_expansion_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_rewrite_power_reduction_target_aware(ctx, expr, target_expr) {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_negated_half_angle_square_expansion_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_sec_csc_squared_expansion_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_mixed_sine_double_angle_expansion_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_rewrite_product_to_sum_target_aware(ctx, expr, target_expr) {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_rewrite_sum_to_product_target_aware(ctx, expr, target_expr) {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_additive_triple_angle_expansion_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_rewrite_negated_sin_double_angle_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_negated_triple_angle_expansion_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_negated_quintuple_angle_expansion_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_negated_recursive_angle_expansion_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_rewrite_negated_cos_double_angle_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_rewrite_cos_double_angle_target_aware(ctx, expr, target_expr) {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_rewrite_triple_angle_expansion_target_aware(ctx, expr, target_expr) {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_quintuple_angle_expansion_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_recursive_angle_expansion_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_rewrite_negated_tan_to_sin_cos_target_aware(ctx, expr, target_expr) {
        return Some(rewrite);
    }

    if let Some(plan) =
        cas_math::trig_canonicalization_support::try_rewrite_tan_to_sin_cos_function_expr(ctx, expr)
    {
        return Some(DeriveTrigRewrite {
            rewritten: plan.rewritten,
            kind: DeriveTrigRewriteKind::TanToSinCos,
        });
    }

    let rewrite =
        cas_math::trig_multi_angle_support::try_rewrite_double_angle_function_expr(ctx, expr)?;
    let kind = match rewrite.kind {
        cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::DoubleSin => {
            DeriveTrigRewriteKind::DoubleAngleSin
        }
        cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::DoubleCos => {
            DeriveTrigRewriteKind::DoubleAngleCos
        }
        _ => return None,
    };

    Some(DeriveTrigRewrite {
        rewritten: rewrite.rewritten,
        kind,
    })
}

pub(crate) fn try_rewrite_trig_special_value_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    if let Some(rewrite) =
        cas_math::trig_eval_table_support::try_rewrite_trig_eval_table_expr(ctx, expr)
    {
        if !matches!(
            rewrite.kind,
            cas_math::trig_eval_table_support::TrigEvalRewriteKind::Table(_)
        ) {
            return None;
        }
        if !strong_target_match(ctx, rewrite.rewritten, target_expr) {
            return None;
        }

        return Some(DeriveTrigRewrite {
            rewritten: rewrite.rewritten,
            kind: DeriveTrigRewriteKind::TrigSpecialValue,
        });
    }

    let rewrite =
        cas_math::trig_core_identity_support::try_rewrite_legacy_evaluate_trig_expr(ctx, expr)?;
    if matches!(
        rewrite.kind,
        cas_math::trig_core_identity_support::LegacyTrigEvalRewriteKind::SinNegative
            | cas_math::trig_core_identity_support::LegacyTrigEvalRewriteKind::CosNegative
            | cas_math::trig_core_identity_support::LegacyTrigEvalRewriteKind::TanNegative
    ) {
        return None;
    }
    if !strong_target_match(ctx, rewrite.rewritten, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten: rewrite.rewritten,
        kind: DeriveTrigRewriteKind::TrigSpecialValue,
    })
}

fn try_rewrite_trig_odd_even_parity_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let rewritten = normalize_trig_negative_parity_tree(ctx, expr);
    if rewritten == expr || !trig_presentational_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten,
        kind: DeriveTrigRewriteKind::TrigOddEvenParity,
    })
}

fn try_rewrite_circular_trig_odd_even_parity_expr(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<cas_math::trig_core_identity_support::TrigOddEvenParityRewrite> {
    let rewrite =
        cas_math::trig_core_identity_support::try_rewrite_trig_odd_even_parity_expr(ctx, expr)?;
    matches!(
        rewrite.fn_name.as_str(),
        "sin" | "cos" | "tan" | "sec" | "csc" | "cot"
    )
    .then_some(rewrite)
}

fn try_rewrite_normalized_double_angle_expansion_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let normalized_expr = rewrite_trig_linear_angle_arguments(ctx, expr);
    if normalized_expr == expr {
        return None;
    }

    let rewrite = cas_math::trig_multi_angle_support::try_rewrite_double_angle_function_expr(
        ctx,
        normalized_expr,
    )?;
    let kind = match rewrite.kind {
        cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::DoubleSin => {
            DeriveTrigRewriteKind::DoubleAngleSin
        }
        cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::DoubleCos => {
            DeriveTrigRewriteKind::DoubleAngleCos
        }
        _ => return None,
    };

    if !strong_target_match(ctx, rewrite.rewritten, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten: rewrite.rewritten,
        kind,
    })
}

fn trig_presentational_target_match(
    ctx: &mut cas_ast::Context,
    actual_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    strong_target_match(ctx, actual_expr, target_expr)
        || presentational_target_match(ctx, actual_expr, target_expr)
}

pub(crate) fn phase_shift_target_match(
    ctx: &mut cas_ast::Context,
    actual_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    if trig_presentational_target_match(ctx, actual_expr, target_expr) {
        return true;
    }

    let canonical_actual = canonicalize_phase_shift_inverse_trig_aliases(ctx, actual_expr);
    let canonical_target = canonicalize_phase_shift_inverse_trig_aliases(ctx, target_expr);
    let simplified_actual = run_phase_shift_match_simplify(ctx, canonical_actual);
    let simplified_target = run_phase_shift_match_simplify(ctx, canonical_target);
    trig_presentational_target_match(ctx, simplified_actual, simplified_target)
}

fn canonicalize_phase_shift_inverse_trig_aliases(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let left = canonicalize_phase_shift_inverse_trig_aliases(ctx, left);
            let right = canonicalize_phase_shift_inverse_trig_aliases(ctx, right);
            ctx.add(Expr::Add(left, right))
        }
        Expr::Sub(left, right) => {
            let left = canonicalize_phase_shift_inverse_trig_aliases(ctx, left);
            let right = canonicalize_phase_shift_inverse_trig_aliases(ctx, right);
            ctx.add(Expr::Sub(left, right))
        }
        Expr::Mul(left, right) => {
            let left = canonicalize_phase_shift_inverse_trig_aliases(ctx, left);
            let right = canonicalize_phase_shift_inverse_trig_aliases(ctx, right);
            ctx.add(Expr::Mul(left, right))
        }
        Expr::Div(left, right) => {
            let left = canonicalize_phase_shift_inverse_trig_aliases(ctx, left);
            let right = canonicalize_phase_shift_inverse_trig_aliases(ctx, right);
            ctx.add(Expr::Div(left, right))
        }
        Expr::Pow(left, right) => {
            let left = canonicalize_phase_shift_inverse_trig_aliases(ctx, left);
            let right = canonicalize_phase_shift_inverse_trig_aliases(ctx, right);
            ctx.add(Expr::Pow(left, right))
        }
        Expr::Neg(inner) => {
            let inner = canonicalize_phase_shift_inverse_trig_aliases(ctx, inner);
            ctx.add(Expr::Neg(inner))
        }
        Expr::Hold(inner) => {
            let inner = canonicalize_phase_shift_inverse_trig_aliases(ctx, inner);
            ctx.add(Expr::Hold(inner))
        }
        Expr::Function(fn_id, args) => {
            let rewritten_args = args
                .into_iter()
                .map(|arg| canonicalize_phase_shift_inverse_trig_aliases(ctx, arg))
                .collect::<Vec<_>>();
            match ctx.builtin_of(fn_id) {
                Some(BuiltinFn::Atan | BuiltinFn::Arctan) => {
                    ctx.call_builtin(BuiltinFn::Arctan, rewritten_args)
                }
                Some(builtin) => ctx.call_builtin(builtin, rewritten_args),
                None => ctx.add(Expr::Function(fn_id, rewritten_args)),
            }
        }
        Expr::Matrix { .. }
        | Expr::Number(_)
        | Expr::Constant(_)
        | Expr::Variable(_)
        | Expr::SessionRef(_) => expr,
    }
}

fn map_product_to_sum_kind(
    kind: cas_math::trig_sum_product_support::TrigProductToSumRewriteKind,
) -> DeriveTrigRewriteKind {
    match kind {
        cas_math::trig_sum_product_support::TrigProductToSumRewriteKind::SinCos => {
            DeriveTrigRewriteKind::ProductToSumSinCos
        }
        cas_math::trig_sum_product_support::TrigProductToSumRewriteKind::CosSin => {
            DeriveTrigRewriteKind::ProductToSumCosSin
        }
        cas_math::trig_sum_product_support::TrigProductToSumRewriteKind::CosCos => {
            DeriveTrigRewriteKind::ProductToSumCosCos
        }
        cas_math::trig_sum_product_support::TrigProductToSumRewriteKind::SinSin => {
            DeriveTrigRewriteKind::ProductToSumSinSin
        }
    }
}

fn rewrite_linear_angle_argument_bridge_expr(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let rewritten = normalize_trig_bridge_expr(ctx, expr);
    if compare_expr(ctx, rewritten, expr) == Ordering::Equal {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten,
        kind: DeriveTrigRewriteKind::LinearAngleArgumentSimplify,
    })
}

fn normalize_trig_bridge_expr(ctx: &mut cas_ast::Context, expr: ExprId) -> ExprId {
    let rewritten = rewrite_trig_linear_angle_arguments(ctx, expr);
    normalize_trig_negative_parity_tree(ctx, rewritten)
}

fn rewrite_trig_linear_angle_arguments(ctx: &mut cas_ast::Context, expr: ExprId) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let rewritten_left = rewrite_trig_linear_angle_arguments(ctx, left);
            let rewritten_right = rewrite_trig_linear_angle_arguments(ctx, right);
            if rewritten_left == left && rewritten_right == right {
                expr
            } else {
                ctx.add(Expr::Add(rewritten_left, rewritten_right))
            }
        }
        Expr::Sub(left, right) => {
            let rewritten_left = rewrite_trig_linear_angle_arguments(ctx, left);
            let rewritten_right = rewrite_trig_linear_angle_arguments(ctx, right);
            if rewritten_left == left && rewritten_right == right {
                expr
            } else {
                ctx.add(Expr::Sub(rewritten_left, rewritten_right))
            }
        }
        Expr::Neg(inner) => {
            let rewritten_inner = rewrite_trig_linear_angle_arguments(ctx, inner);
            if rewritten_inner == inner {
                expr
            } else {
                ctx.add(Expr::Neg(rewritten_inner))
            }
        }
        Expr::Mul(left, right) => {
            let rewritten_left = rewrite_trig_linear_angle_arguments(ctx, left);
            let rewritten_right = rewrite_trig_linear_angle_arguments(ctx, right);
            if rewritten_left == left && rewritten_right == right {
                expr
            } else {
                ctx.add(Expr::Mul(rewritten_left, rewritten_right))
            }
        }
        Expr::Div(left, right) => {
            let rewritten_left = rewrite_trig_linear_angle_arguments(ctx, left);
            let rewritten_right = rewrite_trig_linear_angle_arguments(ctx, right);
            if rewritten_left == left && rewritten_right == right {
                expr
            } else {
                ctx.add(Expr::Div(rewritten_left, rewritten_right))
            }
        }
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let Some(builtin) = ctx.builtin_of(fn_id) else {
                return expr;
            };
            if !matches!(
                builtin,
                BuiltinFn::Sin
                    | BuiltinFn::Cos
                    | BuiltinFn::Tan
                    | BuiltinFn::Sec
                    | BuiltinFn::Csc
                    | BuiltinFn::Cot
            ) {
                return expr;
            }

            let rewritten_arg = rewrite_linear_angle_expr(ctx, args[0]);
            if rewritten_arg == args[0] {
                expr
            } else {
                ctx.add(Expr::Function(fn_id, vec![rewritten_arg]))
            }
        }
        _ => expr,
    }
}

fn rewrite_linear_angle_expr(ctx: &mut cas_ast::Context, expr: ExprId) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let rewritten_left = rewrite_linear_angle_expr(ctx, left);
            let rewritten_right = rewrite_linear_angle_expr(ctx, right);
            combine_linear_angle_terms(ctx, rewritten_left, rewritten_right, false)
        }
        Expr::Sub(left, right) => {
            let rewritten_left = rewrite_linear_angle_expr(ctx, left);
            let rewritten_right = rewrite_linear_angle_expr(ctx, right);
            combine_linear_angle_terms(ctx, rewritten_left, rewritten_right, true)
        }
        Expr::Neg(inner) => {
            let rewritten_inner = rewrite_linear_angle_expr(ctx, inner);
            if rewritten_inner == inner {
                expr
            } else {
                ctx.add(Expr::Neg(rewritten_inner))
            }
        }
        _ => expr,
    }
}

fn try_rewrite_reciprocal_trig_expansion_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    if let Some(rewrite) =
        cas_math::trig_canonicalization_support::try_rewrite_sec_to_recip_cos_function_expr(
            ctx, expr,
        )
    {
        if strong_target_match(ctx, rewrite.rewritten, target_expr) {
            return Some(DeriveTrigRewrite {
                rewritten: rewrite.rewritten,
                kind: DeriveTrigRewriteKind::ExpandSecToRecipCos,
            });
        }
    }

    if let Some(rewrite) =
        cas_math::trig_canonicalization_support::try_rewrite_csc_to_recip_sin_function_expr(
            ctx, expr,
        )
    {
        if strong_target_match(ctx, rewrite.rewritten, target_expr) {
            return Some(DeriveTrigRewrite {
                rewritten: rewrite.rewritten,
                kind: DeriveTrigRewriteKind::ExpandCscToRecipSin,
            });
        }
    }

    if let Some(rewrite) =
        cas_math::trig_canonicalization_support::try_rewrite_cot_to_cos_sin_function_expr(ctx, expr)
    {
        if strong_target_match(ctx, rewrite.rewritten, target_expr) {
            return Some(DeriveTrigRewrite {
                rewritten: rewrite.rewritten,
                kind: DeriveTrigRewriteKind::ExpandCotToCosSin,
            });
        }
    }

    None
}

fn try_rewrite_additive_triple_angle_expansion_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    generate_trig_additive_term_bridge_rewrites(ctx, expr)
        .into_iter()
        .find(|rewrite| {
            matches!(
                rewrite.kind,
                DeriveTrigRewriteKind::TripleAngleSin
                    | DeriveTrigRewriteKind::TripleAngleCos
                    | DeriveTrigRewriteKind::TripleAngleTan
            ) && presentational_target_match(ctx, rewrite.rewritten, target_expr)
        })
}

fn try_rewrite_triple_angle_expansion_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let rewrite = cas_math::trig_multi_angle_support::try_rewrite_triple_angle_expr(ctx, expr)?;
    if !strong_target_match(ctx, rewrite.rewritten, target_expr) {
        return None;
    }

    let kind = match rewrite.kind {
        cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::TripleSin => {
            DeriveTrigRewriteKind::TripleAngleSin
        }
        cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::TripleCos => {
            DeriveTrigRewriteKind::TripleAngleCos
        }
        cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::TripleTan => {
            DeriveTrigRewriteKind::TripleAngleTan
        }
        _ => return None,
    };

    Some(DeriveTrigRewrite {
        rewritten: rewrite.rewritten,
        kind,
    })
}

fn try_rewrite_negated_triple_angle_expansion_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let positive_expr = strip_unit_negation(ctx, expr)?;
    let rewrite =
        cas_math::trig_multi_angle_support::try_rewrite_triple_angle_expr(ctx, positive_expr)?;
    let rewritten = match rewrite.kind {
        cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::TripleTan => {
            negate_preserving_fraction_numerator(ctx, rewrite.rewritten)
        }
        _ => ctx.add(Expr::Neg(rewrite.rewritten)),
    };
    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    let kind = match rewrite.kind {
        cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::TripleSin => {
            DeriveTrigRewriteKind::TripleAngleSin
        }
        cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::TripleCos => {
            DeriveTrigRewriteKind::TripleAngleCos
        }
        cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::TripleTan => {
            DeriveTrigRewriteKind::TripleAngleTan
        }
        _ => return None,
    };

    Some(DeriveTrigRewrite { rewritten, kind })
}

fn try_rewrite_quintuple_angle_expansion_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let rewrite = cas_math::trig_multi_angle_support::try_rewrite_quintuple_angle_expr(ctx, expr)?;
    if !strong_target_match(ctx, rewrite.rewritten, target_expr) {
        return None;
    }

    let kind = match rewrite.kind {
        cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::QuintupleSin => {
            DeriveTrigRewriteKind::QuintupleAngleSin
        }
        cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::QuintupleCos => {
            DeriveTrigRewriteKind::QuintupleAngleCos
        }
        _ => return None,
    };

    Some(DeriveTrigRewrite {
        rewritten: rewrite.rewritten,
        kind,
    })
}

fn try_rewrite_negated_quintuple_angle_expansion_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let positive_expr = strip_unit_negation(ctx, expr)?;
    let rewrite =
        cas_math::trig_multi_angle_support::try_rewrite_quintuple_angle_expr(ctx, positive_expr)?;
    let rewritten = ctx.add(Expr::Neg(rewrite.rewritten));
    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    let kind = match rewrite.kind {
        cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::QuintupleSin => {
            DeriveTrigRewriteKind::QuintupleAngleSin
        }
        cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::QuintupleCos => {
            DeriveTrigRewriteKind::QuintupleAngleCos
        }
        _ => return None,
    };

    Some(DeriveTrigRewrite { rewritten, kind })
}

fn try_rewrite_angle_sum_diff_expansion_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let rewritten = rewrite_angle_sum_diff_expr(ctx, expr)?;

    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten,
        kind: DeriveTrigRewriteKind::AngleSumDiff,
    })
}

fn try_rewrite_negated_angle_sum_diff_expansion_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let positive_expr = strip_unit_negation(ctx, expr)?;
    let positive_rewritten = rewrite_angle_sum_diff_expr(ctx, positive_expr)?;
    let rewritten = ctx.add(Expr::Neg(positive_rewritten));
    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten,
        kind: DeriveTrigRewriteKind::AngleSumDiff,
    })
}

fn rewrite_angle_sum_diff_expr(ctx: &mut cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let (lhs, rhs, is_sum) = match ctx.get(args[0]) {
        Expr::Add(lhs, rhs) => (*lhs, *rhs, true),
        Expr::Sub(lhs, rhs) => (*lhs, *rhs, false),
        _ => return None,
    };

    if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
        let sin_lhs = ctx.call_builtin(BuiltinFn::Sin, vec![lhs]);
        let cos_lhs = ctx.call_builtin(BuiltinFn::Cos, vec![lhs]);
        let sin_rhs = ctx.call_builtin(BuiltinFn::Sin, vec![rhs]);
        let cos_rhs = ctx.call_builtin(BuiltinFn::Cos, vec![rhs]);
        let left = smart_mul(ctx, sin_lhs, cos_rhs);
        let right = smart_mul(ctx, cos_lhs, sin_rhs);
        if is_sum {
            Some(ctx.add(Expr::Add(left, right)))
        } else {
            Some(ctx.add(Expr::Sub(left, right)))
        }
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
        let sin_lhs = ctx.call_builtin(BuiltinFn::Sin, vec![lhs]);
        let cos_lhs = ctx.call_builtin(BuiltinFn::Cos, vec![lhs]);
        let sin_rhs = ctx.call_builtin(BuiltinFn::Sin, vec![rhs]);
        let cos_rhs = ctx.call_builtin(BuiltinFn::Cos, vec![rhs]);
        let left = smart_mul(ctx, cos_lhs, cos_rhs);
        let right = smart_mul(ctx, sin_lhs, sin_rhs);
        if is_sum {
            Some(ctx.add(Expr::Sub(left, right)))
        } else {
            Some(ctx.add(Expr::Add(left, right)))
        }
    } else {
        None
    }
}

fn try_rewrite_tangent_angle_sum_diff_expansion_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let (lhs, rhs, is_sum) = tangent_angle_sum_diff_args(ctx, expr)?;
    let rewritten = build_tangent_angle_sum_diff_fraction(ctx, lhs, rhs, is_sum);

    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten,
        kind: DeriveTrigRewriteKind::TangentAngleSumDiffExpand,
    })
}

fn tangent_angle_sum_diff_args(
    ctx: &cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, bool)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Tan) || args.len() != 1 {
        return None;
    }

    match ctx.get(args[0]) {
        Expr::Add(lhs, rhs) => Some((*lhs, *rhs, true)),
        Expr::Sub(lhs, rhs) => Some((*lhs, *rhs, false)),
        _ => None,
    }
}

fn build_tangent_angle_sum_diff_fraction(
    ctx: &mut cas_ast::Context,
    lhs: ExprId,
    rhs: ExprId,
    is_sum: bool,
) -> ExprId {
    let tan_lhs = ctx.call_builtin(BuiltinFn::Tan, vec![lhs]);
    let tan_rhs = ctx.call_builtin(BuiltinFn::Tan, vec![rhs]);
    let numerator = if is_sum {
        ctx.add(Expr::Add(tan_lhs, tan_rhs))
    } else {
        ctx.add(Expr::Sub(tan_lhs, tan_rhs))
    };

    let tan_lhs = ctx.call_builtin(BuiltinFn::Tan, vec![lhs]);
    let tan_rhs = ctx.call_builtin(BuiltinFn::Tan, vec![rhs]);
    let product = smart_mul(ctx, tan_lhs, tan_rhs);
    let one = ctx.num(1);
    let denominator = if is_sum {
        ctx.add(Expr::Sub(one, product))
    } else {
        ctx.add(Expr::Add(one, product))
    };

    ctx.add(Expr::Div(numerator, denominator))
}

fn try_rewrite_recursive_angle_expansion_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let rewrite =
        cas_math::trig_multi_angle_support::try_rewrite_recursive_trig_expansion_expr(ctx, expr)?;
    if !strong_target_match(ctx, rewrite.rewritten, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten: rewrite.rewritten,
        kind: DeriveTrigRewriteKind::RecursiveAngleSumDiff,
    })
}

fn try_rewrite_negated_recursive_angle_expansion_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let positive_expr = strip_unit_negation(ctx, expr)?;
    let rewrite = cas_math::trig_multi_angle_support::try_rewrite_recursive_trig_expansion_expr(
        ctx,
        positive_expr,
    )?;
    let rewritten = ctx.add(Expr::Neg(rewrite.rewritten));
    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten,
        kind: DeriveTrigRewriteKind::RecursiveAngleSumDiff,
    })
}

fn try_rewrite_negated_reciprocal_trig_expansion_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let positive_expr = strip_unit_negation(ctx, expr)?;

    if let Some(rewrite) =
        cas_math::trig_canonicalization_support::try_rewrite_sec_to_recip_cos_function_expr(
            ctx,
            positive_expr,
        )
    {
        let rewritten = negate_preserving_fraction_numerator(ctx, rewrite.rewritten);
        if strong_target_match(ctx, rewritten, target_expr) {
            return Some(DeriveTrigRewrite {
                rewritten,
                kind: DeriveTrigRewriteKind::ExpandSecToRecipCos,
            });
        }
    }

    if let Some(rewrite) =
        cas_math::trig_canonicalization_support::try_rewrite_csc_to_recip_sin_function_expr(
            ctx,
            positive_expr,
        )
    {
        let rewritten = negate_preserving_fraction_numerator(ctx, rewrite.rewritten);
        if strong_target_match(ctx, rewritten, target_expr) {
            return Some(DeriveTrigRewrite {
                rewritten,
                kind: DeriveTrigRewriteKind::ExpandCscToRecipSin,
            });
        }
    }

    if let Some(rewrite) =
        cas_math::trig_canonicalization_support::try_rewrite_cot_to_cos_sin_function_expr(
            ctx,
            positive_expr,
        )
    {
        let rewritten = negate_preserving_fraction_numerator(ctx, rewrite.rewritten);
        if strong_target_match(ctx, rewritten, target_expr) {
            return Some(DeriveTrigRewrite {
                rewritten,
                kind: DeriveTrigRewriteKind::ExpandCotToCosSin,
            });
        }
    }

    None
}

fn try_rewrite_negated_tan_to_sin_cos_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let positive_expr = strip_unit_negation(ctx, expr)?;
    let rewrite =
        cas_math::trig_canonicalization_support::try_rewrite_tan_to_sin_cos_function_expr(
            ctx,
            positive_expr,
        )?;
    let rewritten = negate_preserving_fraction_numerator(ctx, rewrite.rewritten);
    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten,
        kind: DeriveTrigRewriteKind::TanToSinCos,
    })
}

fn try_rewrite_sec_csc_squared_expansion_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let (trig_fn, arg) = trig_square(ctx, expr)?;
    let one = ctx.num(1);
    let rewritten = match trig_fn {
        BuiltinFn::Sec => {
            let tan_call = ctx.call_builtin(BuiltinFn::Tan, vec![arg]);
            let tan_sq = pow2(ctx, tan_call);
            ctx.add(Expr::Add(one, tan_sq))
        }
        BuiltinFn::Csc => {
            let cot_call = ctx.call_builtin(BuiltinFn::Cot, vec![arg]);
            let cot_sq = pow2(ctx, cot_call);
            ctx.add(Expr::Add(one, cot_sq))
        }
        _ => return None,
    };

    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten,
        kind: match trig_fn {
            BuiltinFn::Sec => DeriveTrigRewriteKind::ExpandSecSquared,
            BuiltinFn::Csc => DeriveTrigRewriteKind::ExpandCscSquared,
            _ => unreachable!("only sec/csc are valid here"),
        },
    })
}

pub(crate) fn try_rewrite_pythagorean_factor_form_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DerivePythagoreanFactorRewrite> {
    for candidate in [
        build_forward_pythagorean_factor_candidate(ctx, expr),
        build_reverse_pythagorean_factor_candidate(ctx, expr),
        build_negated_reverse_pythagorean_factor_candidate(ctx, expr),
    ] {
        let Some(rewrite) = candidate else {
            continue;
        };
        if strong_target_match(ctx, rewrite.rewritten, target_expr) {
            return Some(rewrite);
        }
    }

    None
}

fn try_rewrite_half_angle_square_expansion_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let (trig_fn, arg) = trig_square(ctx, expr)?;
    let kind = match trig_fn {
        BuiltinFn::Sin => DeriveTrigRewriteKind::HalfAngleSinSquaredExpand,
        BuiltinFn::Cos => DeriveTrigRewriteKind::HalfAngleCosSquaredExpand,
        _ => return None,
    };

    let one = ctx.num(1);
    let two = ctx.num(2);
    let raw_double_arg = smart_mul(ctx, two, arg);
    let double_arg = cas_math::canonical_forms::normalize_core(ctx, raw_double_arg);
    let cos_double_arg = ctx.call_builtin(BuiltinFn::Cos, vec![double_arg]);
    let numerator = match trig_fn {
        BuiltinFn::Sin => ctx.add(Expr::Sub(one, cos_double_arg)),
        BuiltinFn::Cos => ctx.add(Expr::Add(one, cos_double_arg)),
        _ => unreachable!("only sin/cos are valid here"),
    };
    let rewritten = ctx.add(Expr::Div(numerator, two));

    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite { rewritten, kind })
}

fn try_rewrite_power_reduction_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    if let Some((trig_fn, arg)) = trig_fourth_power(ctx, expr) {
        let (rewritten, kind) = match trig_fn {
            BuiltinFn::Sin => (
                build_sin_fourth_power_reduction_rewritten(ctx, arg),
                DeriveTrigRewriteKind::PowerReductionSinFourth,
            ),
            BuiltinFn::Cos => (
                build_cos_fourth_power_reduction_rewritten(ctx, arg),
                DeriveTrigRewriteKind::PowerReductionCosFourth,
            ),
            _ => return None,
        };

        if strong_target_match(ctx, rewritten, target_expr) {
            return Some(DeriveTrigRewrite { rewritten, kind });
        }
    }

    if let Some((trig_fn, arg)) = trig_sixth_power(ctx, expr) {
        let (rewritten, kind) = match trig_fn {
            BuiltinFn::Sin => (
                build_sin_sixth_power_reduction_rewritten(ctx, arg),
                DeriveTrigRewriteKind::PowerReductionSinSixth,
            ),
            BuiltinFn::Cos => (
                build_cos_sixth_power_reduction_rewritten(ctx, arg),
                DeriveTrigRewriteKind::PowerReductionCosSixth,
            ),
            _ => return None,
        };

        if strong_target_match(ctx, rewritten, target_expr) {
            return Some(DeriveTrigRewrite { rewritten, kind });
        }
    }

    if let Some((trig_fn, arg)) = trig_eighth_power(ctx, expr) {
        let (rewritten, kind) = match trig_fn {
            BuiltinFn::Sin => (
                build_sin_eighth_power_reduction_rewritten(ctx, arg),
                DeriveTrigRewriteKind::PowerReductionSinEighth,
            ),
            BuiltinFn::Cos => (
                build_cos_eighth_power_reduction_rewritten(ctx, arg),
                DeriveTrigRewriteKind::PowerReductionCosEighth,
            ),
            _ => return None,
        };

        if strong_target_match(ctx, rewritten, target_expr) {
            return Some(DeriveTrigRewrite { rewritten, kind });
        }
    }

    if let Some((trig_fn, arg)) = trig_tenth_power(ctx, expr) {
        let (rewritten, kind) = match trig_fn {
            BuiltinFn::Sin => (
                build_sin_tenth_power_reduction_rewritten(ctx, arg),
                DeriveTrigRewriteKind::PowerReductionSinTenth,
            ),
            BuiltinFn::Cos => (
                build_cos_tenth_power_reduction_rewritten(ctx, arg),
                DeriveTrigRewriteKind::PowerReductionCosTenth,
            ),
            _ => return None,
        };

        if strong_target_match(ctx, rewritten, target_expr) {
            return Some(DeriveTrigRewrite { rewritten, kind });
        }
    }

    if let Some((trig_fn, arg)) = trig_twelfth_power(ctx, expr) {
        let (rewritten, kind) = match trig_fn {
            BuiltinFn::Sin => (
                build_sin_twelfth_power_reduction_rewritten(ctx, arg),
                DeriveTrigRewriteKind::PowerReductionSinTwelfth,
            ),
            BuiltinFn::Cos => (
                build_cos_twelfth_power_reduction_rewritten(ctx, arg),
                DeriveTrigRewriteKind::PowerReductionCosTwelfth,
            ),
            _ => return None,
        };

        if strong_target_match(ctx, rewritten, target_expr) {
            return Some(DeriveTrigRewrite { rewritten, kind });
        }
    }

    if let Some((trig_fn, arg)) = trig_fourteenth_power(ctx, expr) {
        let (rewritten, kind) = match trig_fn {
            BuiltinFn::Sin => (
                build_sin_fourteenth_power_reduction_rewritten(ctx, arg),
                DeriveTrigRewriteKind::PowerReductionSinFourteenth,
            ),
            BuiltinFn::Cos => (
                build_cos_fourteenth_power_reduction_rewritten(ctx, arg),
                DeriveTrigRewriteKind::PowerReductionCosFourteenth,
            ),
            _ => return None,
        };

        if strong_target_match(ctx, rewritten, target_expr) {
            return Some(DeriveTrigRewrite { rewritten, kind });
        }
    }

    if let Some((trig_fn, arg)) = trig_sixteenth_power(ctx, expr) {
        let (rewritten, kind) = match trig_fn {
            BuiltinFn::Sin => (
                build_sin_sixteenth_power_reduction_rewritten(ctx, arg),
                DeriveTrigRewriteKind::PowerReductionSinSixteenth,
            ),
            BuiltinFn::Cos => (
                build_cos_sixteenth_power_reduction_rewritten(ctx, arg),
                DeriveTrigRewriteKind::PowerReductionCosSixteenth,
            ),
            _ => return None,
        };

        if strong_target_match(ctx, rewritten, target_expr) {
            return Some(DeriveTrigRewrite { rewritten, kind });
        }
    }

    if let Some((trig_fn, arg)) = trig_eighteenth_power(ctx, expr) {
        let (rewritten, kind) = match trig_fn {
            BuiltinFn::Sin => (
                build_sin_eighteenth_power_reduction_rewritten(ctx, arg),
                DeriveTrigRewriteKind::PowerReductionSinEighteenth,
            ),
            BuiltinFn::Cos => (
                build_cos_eighteenth_power_reduction_rewritten(ctx, arg),
                DeriveTrigRewriteKind::PowerReductionCosEighteenth,
            ),
            _ => return None,
        };

        if strong_target_match(ctx, rewritten, target_expr) {
            return Some(DeriveTrigRewrite { rewritten, kind });
        }
    }

    if let Some((trig_fn, arg)) = trig_twentieth_power(ctx, expr) {
        let (rewritten, kind) = match trig_fn {
            BuiltinFn::Sin => (
                build_sin_twentieth_power_reduction_rewritten(ctx, arg),
                DeriveTrigRewriteKind::PowerReductionSinTwentieth,
            ),
            BuiltinFn::Cos => (
                build_cos_twentieth_power_reduction_rewritten(ctx, arg),
                DeriveTrigRewriteKind::PowerReductionCosTwentieth,
            ),
            _ => return None,
        };

        if strong_target_match(ctx, rewritten, target_expr) {
            return Some(DeriveTrigRewrite { rewritten, kind });
        }
    }

    if let Some((trig_fn, arg)) = trig_twenty_second_power(ctx, expr) {
        let (rewritten, kind) = match trig_fn {
            BuiltinFn::Sin => (
                build_sin_twenty_second_power_reduction_rewritten(ctx, arg),
                DeriveTrigRewriteKind::PowerReductionSinTwentySecond,
            ),
            BuiltinFn::Cos => (
                build_cos_twenty_second_power_reduction_rewritten(ctx, arg),
                DeriveTrigRewriteKind::PowerReductionCosTwentySecond,
            ),
            _ => return None,
        };

        if strong_target_match(ctx, rewritten, target_expr) {
            return Some(DeriveTrigRewrite { rewritten, kind });
        }
    }

    if let Some((trig_fn, arg, power)) = trig_higher_even_power(ctx, expr) {
        let rewritten = build_higher_even_power_reduction_rewritten(ctx, arg, power, trig_fn)?;
        let kind = match trig_fn {
            BuiltinFn::Sin => DeriveTrigRewriteKind::PowerReductionSinHigherEven,
            BuiltinFn::Cos => DeriveTrigRewriteKind::PowerReductionCosHigherEven,
            _ => return None,
        };

        if strong_target_match(ctx, rewritten, target_expr) {
            return Some(DeriveTrigRewrite { rewritten, kind });
        }
    }

    if let Some(arg) = trig_square_product_same_arg(ctx, expr) {
        let rewritten = build_sin_cos_square_product_reduction_rewritten(ctx, arg);
        if strong_target_match(ctx, rewritten, target_expr) {
            return Some(DeriveTrigRewrite {
                rewritten,
                kind: DeriveTrigRewriteKind::PowerReductionSinCosSquares,
            });
        }
    }

    None
}

fn try_rewrite_sin_cos_binomial_square_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let (arg, same_sign) = sin_cos_binomial_square_same_arg(ctx, expr)?;
    let rewritten = build_sin_cos_binomial_square_rewritten(ctx, arg, same_sign);
    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten,
        kind: if same_sign {
            DeriveTrigRewriteKind::SinCosSquareSum
        } else {
            DeriveTrigRewriteKind::SinCosSquareDiff
        },
    })
}

fn try_rewrite_negated_half_angle_square_expansion_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let positive_expr = strip_unit_negation(ctx, expr)?;
    let (trig_fn, arg) = trig_square(ctx, positive_expr)?;
    let kind = match trig_fn {
        BuiltinFn::Sin => DeriveTrigRewriteKind::HalfAngleNegSinSquaredExpand,
        BuiltinFn::Cos => DeriveTrigRewriteKind::HalfAngleNegCosSquaredExpand,
        _ => return None,
    };

    let rewritten = build_negated_half_angle_square_rewritten(ctx, trig_fn, arg)?;

    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite { rewritten, kind })
}

fn try_rewrite_half_angle_tangent_expansion_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Tan) || args.len() != 1 {
        return None;
    }

    let arg = args[0];
    let one = ctx.num(1);
    let double_arg = doubled_half_angle_tangent_argument(ctx, arg);
    let sin_double = ctx.call_builtin(BuiltinFn::Sin, vec![double_arg]);
    let cos_double = ctx.call_builtin(BuiltinFn::Cos, vec![double_arg]);

    let numerator_a = ctx.add(Expr::Sub(one, cos_double));
    let candidate_a = ctx.add(Expr::Div(numerator_a, sin_double));
    if strong_target_match(ctx, candidate_a, target_expr) {
        return Some(DeriveTrigRewrite {
            rewritten: candidate_a,
            kind: DeriveTrigRewriteKind::HalfAngleTangentExpandOneMinusCosOverSin,
        });
    }

    let denominator_b = ctx.add(Expr::Add(one, cos_double));
    let candidate_b = ctx.add(Expr::Div(sin_double, denominator_b));
    if strong_target_match(ctx, candidate_b, target_expr) {
        return Some(DeriveTrigRewrite {
            rewritten: candidate_b,
            kind: DeriveTrigRewriteKind::HalfAngleTangentExpandSinOverOnePlusCos,
        });
    }

    None
}

fn try_rewrite_negated_half_angle_tangent_expansion_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let positive_expr = strip_unit_negation(ctx, expr)?;
    let Expr::Function(fn_id, args) = ctx.get(positive_expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Tan) || args.len() != 1 {
        return None;
    }

    let arg = args[0];
    let one = ctx.num(1);
    let double_arg = doubled_half_angle_tangent_argument(ctx, arg);
    let sin_double = ctx.call_builtin(BuiltinFn::Sin, vec![double_arg]);
    let cos_double = ctx.call_builtin(BuiltinFn::Cos, vec![double_arg]);

    let numerator_a = ctx.add(Expr::Sub(one, cos_double));
    let positive_a = ctx.add(Expr::Div(numerator_a, sin_double));
    let candidate_a = negate_preserving_fraction_numerator(ctx, positive_a);
    if strong_target_match(ctx, candidate_a, target_expr) {
        return Some(DeriveTrigRewrite {
            rewritten: candidate_a,
            kind: DeriveTrigRewriteKind::HalfAngleTangentExpandOneMinusCosOverSin,
        });
    }

    let denominator_b = ctx.add(Expr::Add(one, cos_double));
    let positive_b = ctx.add(Expr::Div(sin_double, denominator_b));
    let candidate_b = negate_preserving_fraction_numerator(ctx, positive_b);
    if strong_target_match(ctx, candidate_b, target_expr) {
        return Some(DeriveTrigRewrite {
            rewritten: candidate_b,
            kind: DeriveTrigRewriteKind::HalfAngleTangentExpandSinOverOnePlusCos,
        });
    }

    None
}

fn doubled_half_angle_tangent_argument(ctx: &mut cas_ast::Context, arg: ExprId) -> ExprId {
    if let Expr::Div(numerator, denominator) = ctx.get(arg).clone() {
        if is_small_integer(ctx, denominator, 2) {
            return numerator;
        }
    }

    let two = ctx.num(2);
    let raw_double_arg = smart_mul(ctx, two, arg);
    rewrite_linear_angle_expr(ctx, raw_double_arg)
}

fn try_rewrite_tangent_half_angle_substitution_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let (trig_fn, arg) = {
        let Expr::Function(fn_id, args) = ctx.get(expr) else {
            return None;
        };
        if args.len() != 1 {
            return None;
        }
        if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
            (BuiltinFn::Sin, args[0])
        } else if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
            (BuiltinFn::Cos, args[0])
        } else {
            return None;
        }
    };

    let half_arg = tangent_half_angle_substitution_argument(ctx, arg);
    let rewritten = match trig_fn {
        BuiltinFn::Sin => build_tangent_half_angle_sine_substitution(ctx, half_arg),
        BuiltinFn::Cos => build_tangent_half_angle_cosine_substitution(ctx, half_arg),
        _ => return None,
    };

    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    let kind = match trig_fn {
        BuiltinFn::Sin => DeriveTrigRewriteKind::TangentHalfAngleSubstitutionSin,
        BuiltinFn::Cos => DeriveTrigRewriteKind::TangentHalfAngleSubstitutionCos,
        _ => return None,
    };

    Some(DeriveTrigRewrite { rewritten, kind })
}

fn tangent_half_angle_substitution_argument(ctx: &mut cas_ast::Context, arg: ExprId) -> ExprId {
    let normalized_arg = rewrite_linear_angle_expr(ctx, arg);
    if let Some(inner) = extract_double_angle_inner(ctx, normalized_arg) {
        return inner;
    }

    let two = ctx.num(2);
    ctx.add(Expr::Div(normalized_arg, two))
}

fn build_tangent_half_angle_sine_substitution(
    ctx: &mut cas_ast::Context,
    half_arg: ExprId,
) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let tan_half = ctx.call_builtin(BuiltinFn::Tan, vec![half_arg]);
    let numerator = smart_mul(ctx, two, tan_half);
    let tan_sq = pow2(ctx, tan_half);
    let denominator = ctx.add(Expr::Add(one, tan_sq));
    ctx.add(Expr::Div(numerator, denominator))
}

fn build_tangent_half_angle_cosine_substitution(
    ctx: &mut cas_ast::Context,
    half_arg: ExprId,
) -> ExprId {
    let one = ctx.num(1);
    let tan_half = ctx.call_builtin(BuiltinFn::Tan, vec![half_arg]);
    let tan_sq = pow2(ctx, tan_half);
    let numerator = ctx.add(Expr::Sub(one, tan_sq));
    let denominator = ctx.add(Expr::Add(one, tan_sq));
    ctx.add(Expr::Div(numerator, denominator))
}

fn try_rewrite_tangent_double_angle_expansion_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let arg = {
        let Expr::Function(fn_id, args) = ctx.get(expr) else {
            return None;
        };
        if !ctx.is_builtin(*fn_id, BuiltinFn::Tan) || args.len() != 1 {
            return None;
        }
        args[0]
    };

    let normalized_arg = rewrite_linear_angle_expr(ctx, arg);
    let half_arg = extract_double_angle_inner(ctx, normalized_arg)?;
    let rewritten = build_tangent_double_angle_fraction(ctx, half_arg);

    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten,
        kind: DeriveTrigRewriteKind::DoubleAngleTanExpand,
    })
}

fn build_tangent_double_angle_fraction(ctx: &mut cas_ast::Context, arg: ExprId) -> ExprId {
    let two = ctx.num(2);
    let one = ctx.num(1);
    let tan_arg = ctx.call_builtin(BuiltinFn::Tan, vec![arg]);
    let numerator = smart_mul(ctx, two, tan_arg);
    let tan_sq = pow2(ctx, tan_arg);
    let denominator = ctx.add(Expr::Sub(one, tan_sq));
    ctx.add(Expr::Div(numerator, denominator))
}

fn build_forward_pythagorean_factor_candidate(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<DerivePythagoreanFactorRewrite> {
    let Expr::Sub(lhs, rhs) = ctx.get(expr) else {
        return None;
    };
    if !is_small_integer(ctx, *lhs, 1) {
        return None;
    }

    let (trig_fn, arg) = trig_square(ctx, *rhs)?;
    let other_fn = complementary_trig_fn(trig_fn)?;
    let other_trig = ctx.call_builtin(other_fn, vec![arg]);
    Some(DerivePythagoreanFactorRewrite {
        rewritten: pow2(ctx, other_trig),
        description: format!(
            "1 - {}²(x) = {}²(x)",
            trig_name(trig_fn),
            trig_name(other_fn)
        ),
    })
}

fn build_reverse_pythagorean_factor_candidate(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<DerivePythagoreanFactorRewrite> {
    let (trig_fn, arg) = trig_square(ctx, expr)?;
    let other_fn = complementary_trig_fn(trig_fn)?;
    let other_trig = ctx.call_builtin(other_fn, vec![arg]);
    let one = ctx.num(1);
    let other_square = pow2(ctx, other_trig);
    Some(DerivePythagoreanFactorRewrite {
        rewritten: ctx.add(Expr::Sub(one, other_square)),
        description: format!(
            "1 - {}²(x) = {}²(x)",
            trig_name(other_fn),
            trig_name(trig_fn)
        ),
    })
}

fn build_negated_reverse_pythagorean_factor_candidate(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<DerivePythagoreanFactorRewrite> {
    let positive_expr = strip_unit_negation(ctx, expr)?;
    let (trig_fn, arg) = trig_square(ctx, positive_expr)?;
    let other_fn = complementary_trig_fn(trig_fn)?;
    let other_trig = ctx.call_builtin(other_fn, vec![arg]);
    let one = ctx.num(1);
    let other_square = pow2(ctx, other_trig);

    Some(DerivePythagoreanFactorRewrite {
        rewritten: ctx.add(Expr::Sub(other_square, one)),
        description: format!(
            "{}²(x) - 1 = -{}²(x)",
            trig_name(other_fn),
            trig_name(trig_fn)
        ),
    })
}

fn trig_square(ctx: &cas_ast::Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if !is_small_integer(ctx, *exp, 2) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(*base) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let trig_fn = if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
        BuiltinFn::Sin
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
        BuiltinFn::Cos
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Sec) {
        BuiltinFn::Sec
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Csc) {
        BuiltinFn::Csc
    } else {
        return None;
    };

    Some((trig_fn, args[0]))
}

fn trig_fourth_power(ctx: &cas_ast::Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    trig_even_power(ctx, expr, 4)
}

fn trig_sixth_power(ctx: &cas_ast::Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    trig_even_power(ctx, expr, 6)
}

fn trig_eighth_power(ctx: &cas_ast::Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    trig_even_power(ctx, expr, 8)
}

fn trig_tenth_power(ctx: &cas_ast::Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    trig_even_power(ctx, expr, 10)
}

fn trig_twelfth_power(ctx: &cas_ast::Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    trig_even_power(ctx, expr, 12)
}

fn trig_fourteenth_power(ctx: &cas_ast::Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    trig_even_power(ctx, expr, 14)
}

fn trig_sixteenth_power(ctx: &cas_ast::Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    trig_even_power(ctx, expr, 16)
}

fn trig_eighteenth_power(ctx: &cas_ast::Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    trig_even_power(ctx, expr, 18)
}

fn trig_twentieth_power(ctx: &cas_ast::Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    trig_even_power(ctx, expr, 20)
}

fn trig_twenty_second_power(ctx: &cas_ast::Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    trig_even_power(ctx, expr, 22)
}

fn trig_higher_even_power(
    ctx: &cas_ast::Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId, u32)> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let Expr::Number(n) = ctx.get(*exp) else {
        return None;
    };
    if !n.is_integer() {
        return None;
    }

    let power = n.to_integer().to_u32()?;
    if power <= 22 || power % 2 != 0 {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(*base) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let trig_fn = if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
        BuiltinFn::Sin
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
        BuiltinFn::Cos
    } else {
        return None;
    };

    Some((trig_fn, args[0], power))
}

fn trig_even_power(
    ctx: &cas_ast::Context,
    expr: ExprId,
    power: i64,
) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if !is_small_integer(ctx, *exp, power) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(*base) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let trig_fn = if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
        BuiltinFn::Sin
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
        BuiltinFn::Cos
    } else {
        return None;
    };

    Some((trig_fn, args[0]))
}

fn trig_square_product_same_arg(ctx: &mut cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let mut sin_arg = None;
    let mut cos_arg = None;
    for factor in factors {
        let (trig_fn, arg) = trig_square(ctx, factor)?;
        match trig_fn {
            BuiltinFn::Sin => sin_arg = Some(arg),
            BuiltinFn::Cos => cos_arg = Some(arg),
            _ => return None,
        }
    }

    let sin_arg = sin_arg?;
    let cos_arg = cos_arg?;
    (compare_expr(ctx, sin_arg, cos_arg) == Ordering::Equal).then_some(sin_arg)
}

fn sin_cos_binomial_square_same_arg(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, bool)> {
    let (base, exp) = match ctx.get(expr) {
        Expr::Pow(base, exp) => (*base, *exp),
        _ => return None,
    };
    if !is_small_integer(ctx, exp, 2) {
        return None;
    }

    let base = strip_unit_negation(ctx, base).unwrap_or(base);
    let terms = add_terms_signed(ctx, base);
    if terms.len() != 2 {
        return None;
    }

    let mut sin_term = None;
    let mut cos_term = None;
    for (term, sign) in terms {
        let (trig_fn, arg) = extract_sin_or_cos_linear_term(ctx, term)?;
        match trig_fn {
            BuiltinFn::Sin if sin_term.is_none() => sin_term = Some((arg, sign)),
            BuiltinFn::Cos if cos_term.is_none() => cos_term = Some((arg, sign)),
            _ => return None,
        }
    }

    let (sin_arg, sin_sign) = sin_term?;
    let (cos_arg, cos_sign) = cos_term?;
    if compare_expr(ctx, sin_arg, cos_arg) != Ordering::Equal {
        return None;
    }

    Some((sin_arg, sin_sign == cos_sign))
}

fn extract_sin_or_cos_linear_term(
    ctx: &cas_ast::Context,
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

fn build_sin_fourth_power_reduction_rewritten(ctx: &mut cas_ast::Context, arg: ExprId) -> ExprId {
    build_fourth_power_reduction_rewritten(ctx, arg, true)
}

fn build_cos_fourth_power_reduction_rewritten(ctx: &mut cas_ast::Context, arg: ExprId) -> ExprId {
    build_fourth_power_reduction_rewritten(ctx, arg, false)
}

fn build_sin_sixth_power_reduction_rewritten(ctx: &mut cas_ast::Context, arg: ExprId) -> ExprId {
    build_sixth_power_reduction_rewritten(ctx, arg, true)
}

fn build_cos_sixth_power_reduction_rewritten(ctx: &mut cas_ast::Context, arg: ExprId) -> ExprId {
    build_sixth_power_reduction_rewritten(ctx, arg, false)
}

fn build_sin_eighth_power_reduction_rewritten(ctx: &mut cas_ast::Context, arg: ExprId) -> ExprId {
    build_eighth_power_reduction_rewritten(ctx, arg, true)
}

fn build_cos_eighth_power_reduction_rewritten(ctx: &mut cas_ast::Context, arg: ExprId) -> ExprId {
    build_eighth_power_reduction_rewritten(ctx, arg, false)
}

fn build_sin_tenth_power_reduction_rewritten(ctx: &mut cas_ast::Context, arg: ExprId) -> ExprId {
    build_tenth_power_reduction_rewritten(ctx, arg, true)
}

fn build_cos_tenth_power_reduction_rewritten(ctx: &mut cas_ast::Context, arg: ExprId) -> ExprId {
    build_tenth_power_reduction_rewritten(ctx, arg, false)
}

fn build_sin_twelfth_power_reduction_rewritten(ctx: &mut cas_ast::Context, arg: ExprId) -> ExprId {
    build_twelfth_power_reduction_rewritten(ctx, arg, true)
}

fn build_cos_twelfth_power_reduction_rewritten(ctx: &mut cas_ast::Context, arg: ExprId) -> ExprId {
    build_twelfth_power_reduction_rewritten(ctx, arg, false)
}

fn build_sin_fourteenth_power_reduction_rewritten(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
) -> ExprId {
    build_fourteenth_power_reduction_rewritten(ctx, arg, true)
}

fn build_cos_fourteenth_power_reduction_rewritten(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
) -> ExprId {
    build_fourteenth_power_reduction_rewritten(ctx, arg, false)
}

fn build_sin_sixteenth_power_reduction_rewritten(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
) -> ExprId {
    build_sixteenth_power_reduction_rewritten(ctx, arg, true)
}

fn build_cos_sixteenth_power_reduction_rewritten(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
) -> ExprId {
    build_sixteenth_power_reduction_rewritten(ctx, arg, false)
}

fn build_sin_eighteenth_power_reduction_rewritten(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
) -> ExprId {
    build_eighteenth_power_reduction_rewritten(ctx, arg, true)
}

fn build_cos_eighteenth_power_reduction_rewritten(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
) -> ExprId {
    build_eighteenth_power_reduction_rewritten(ctx, arg, false)
}

fn build_sin_twentieth_power_reduction_rewritten(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
) -> ExprId {
    build_twentieth_power_reduction_rewritten(ctx, arg, true)
}

fn build_cos_twentieth_power_reduction_rewritten(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
) -> ExprId {
    build_twentieth_power_reduction_rewritten(ctx, arg, false)
}

fn build_sin_twenty_second_power_reduction_rewritten(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
) -> ExprId {
    build_twenty_second_power_reduction_rewritten(ctx, arg, true)
}

fn build_cos_twenty_second_power_reduction_rewritten(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
) -> ExprId {
    build_twenty_second_power_reduction_rewritten(ctx, arg, false)
}

fn build_higher_even_power_reduction_rewritten(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
    power: u32,
    trig_fn: BuiltinFn,
) -> Option<ExprId> {
    if power <= 22 || !power.is_multiple_of(2) || power > 62 {
        return None;
    }
    if trig_fn != BuiltinFn::Sin && trig_fn != BuiltinFn::Cos {
        return None;
    }

    let half = power / 2;
    let denominator = i64::checked_shl(1, power - 1)?;
    let constant_coeff = binomial_i64(power, half)?.checked_div(2)?;

    let mut numerator = ctx.num(constant_coeff);
    for j in 1..=half {
        let coeff = binomial_i64(power, half - j)?;
        let term = if coeff == 1 {
            build_cos_multiple(ctx, arg, i64::from(2 * j))
        } else {
            build_scaled_cos_multiple(ctx, arg, i64::from(2 * j), coeff)
        };

        numerator = match trig_fn {
            BuiltinFn::Sin if j % 2 == 1 => ctx.add(Expr::Sub(numerator, term)),
            BuiltinFn::Sin | BuiltinFn::Cos => ctx.add(Expr::Add(numerator, term)),
            _ => unreachable!("only sin/cos are valid here"),
        };
    }

    let denominator = ctx.num(denominator);
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn build_fourth_power_reduction_rewritten(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
    is_sin: bool,
) -> ExprId {
    let numerator = if is_sin {
        let three = ctx.num(3);
        let four_cos_double = build_scaled_cos_multiple(ctx, arg, 2, 4);
        let cos_quadruple = build_cos_multiple(ctx, arg, 4);
        let leading = ctx.add(Expr::Sub(three, four_cos_double));
        ctx.add(Expr::Add(leading, cos_quadruple))
    } else {
        let three = ctx.num(3);
        let four_cos_double = build_scaled_cos_multiple(ctx, arg, 2, 4);
        let leading = ctx.add(Expr::Add(three, four_cos_double));
        let cos_quadruple = build_cos_multiple(ctx, arg, 4);
        ctx.add(Expr::Add(leading, cos_quadruple))
    };

    let eight = ctx.num(8);
    ctx.add(Expr::Div(numerator, eight))
}

fn build_sixth_power_reduction_rewritten(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
    is_sin: bool,
) -> ExprId {
    let ten = ctx.num(10);
    let fifteen_cos_double = build_scaled_cos_multiple(ctx, arg, 2, 15);
    let six_cos_quadruple = build_scaled_cos_multiple(ctx, arg, 4, 6);
    let cos_sextuple = build_cos_multiple(ctx, arg, 6);

    let numerator = if is_sin {
        let leading = ctx.add(Expr::Sub(ten, fifteen_cos_double));
        let middle = ctx.add(Expr::Add(leading, six_cos_quadruple));
        ctx.add(Expr::Sub(middle, cos_sextuple))
    } else {
        let leading = ctx.add(Expr::Add(ten, fifteen_cos_double));
        let middle = ctx.add(Expr::Add(leading, six_cos_quadruple));
        ctx.add(Expr::Add(middle, cos_sextuple))
    };

    let thirty_two = ctx.num(32);
    ctx.add(Expr::Div(numerator, thirty_two))
}

fn build_eighth_power_reduction_rewritten(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
    is_sin: bool,
) -> ExprId {
    let thirty_five = ctx.num(35);
    let fifty_six_cos_double = build_scaled_cos_multiple(ctx, arg, 2, 56);
    let twenty_eight_cos_quadruple = build_scaled_cos_multiple(ctx, arg, 4, 28);
    let eight_cos_sextuple = build_scaled_cos_multiple(ctx, arg, 6, 8);
    let cos_octuple = build_cos_multiple(ctx, arg, 8);

    let numerator = if is_sin {
        let leading = ctx.add(Expr::Sub(thirty_five, fifty_six_cos_double));
        let middle = ctx.add(Expr::Add(leading, twenty_eight_cos_quadruple));
        let tail = ctx.add(Expr::Sub(middle, eight_cos_sextuple));
        ctx.add(Expr::Add(tail, cos_octuple))
    } else {
        let leading = ctx.add(Expr::Add(thirty_five, fifty_six_cos_double));
        let middle = ctx.add(Expr::Add(leading, twenty_eight_cos_quadruple));
        let tail = ctx.add(Expr::Add(middle, eight_cos_sextuple));
        ctx.add(Expr::Add(tail, cos_octuple))
    };

    let one_twenty_eight = ctx.num(128);
    ctx.add(Expr::Div(numerator, one_twenty_eight))
}

fn build_tenth_power_reduction_rewritten(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
    is_sin: bool,
) -> ExprId {
    let one_twenty_six = ctx.num(126);
    let two_ten_cos_double = build_scaled_cos_multiple(ctx, arg, 2, 210);
    let one_twenty_cos_quadruple = build_scaled_cos_multiple(ctx, arg, 4, 120);
    let forty_five_cos_sextuple = build_scaled_cos_multiple(ctx, arg, 6, 45);
    let ten_cos_octuple = build_scaled_cos_multiple(ctx, arg, 8, 10);
    let cos_decuple = build_cos_multiple(ctx, arg, 10);

    let numerator = if is_sin {
        let leading = ctx.add(Expr::Sub(one_twenty_six, two_ten_cos_double));
        let second = ctx.add(Expr::Add(leading, one_twenty_cos_quadruple));
        let third = ctx.add(Expr::Sub(second, forty_five_cos_sextuple));
        let fourth = ctx.add(Expr::Add(third, ten_cos_octuple));
        ctx.add(Expr::Sub(fourth, cos_decuple))
    } else {
        let leading = ctx.add(Expr::Add(one_twenty_six, two_ten_cos_double));
        let second = ctx.add(Expr::Add(leading, one_twenty_cos_quadruple));
        let third = ctx.add(Expr::Add(second, forty_five_cos_sextuple));
        let fourth = ctx.add(Expr::Add(third, ten_cos_octuple));
        ctx.add(Expr::Add(fourth, cos_decuple))
    };

    let five_twelve = ctx.num(512);
    ctx.add(Expr::Div(numerator, five_twelve))
}

fn build_twelfth_power_reduction_rewritten(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
    is_sin: bool,
) -> ExprId {
    let four_sixty_two = ctx.num(462);
    let seven_ninety_two_cos_double = build_scaled_cos_multiple(ctx, arg, 2, 792);
    let four_ninety_five_cos_quadruple = build_scaled_cos_multiple(ctx, arg, 4, 495);
    let two_twenty_cos_sextuple = build_scaled_cos_multiple(ctx, arg, 6, 220);
    let sixty_six_cos_octuple = build_scaled_cos_multiple(ctx, arg, 8, 66);
    let twelve_cos_decuple = build_scaled_cos_multiple(ctx, arg, 10, 12);
    let cos_twelvefold = build_cos_multiple(ctx, arg, 12);

    let numerator = if is_sin {
        let leading = ctx.add(Expr::Sub(four_sixty_two, seven_ninety_two_cos_double));
        let second = ctx.add(Expr::Add(leading, four_ninety_five_cos_quadruple));
        let third = ctx.add(Expr::Sub(second, two_twenty_cos_sextuple));
        let fourth = ctx.add(Expr::Add(third, sixty_six_cos_octuple));
        let fifth = ctx.add(Expr::Sub(fourth, twelve_cos_decuple));
        ctx.add(Expr::Add(fifth, cos_twelvefold))
    } else {
        let leading = ctx.add(Expr::Add(four_sixty_two, seven_ninety_two_cos_double));
        let second = ctx.add(Expr::Add(leading, four_ninety_five_cos_quadruple));
        let third = ctx.add(Expr::Add(second, two_twenty_cos_sextuple));
        let fourth = ctx.add(Expr::Add(third, sixty_six_cos_octuple));
        let fifth = ctx.add(Expr::Add(fourth, twelve_cos_decuple));
        ctx.add(Expr::Add(fifth, cos_twelvefold))
    };

    let two_zero_four_eight = ctx.num(2048);
    ctx.add(Expr::Div(numerator, two_zero_four_eight))
}

fn build_fourteenth_power_reduction_rewritten(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
    is_sin: bool,
) -> ExprId {
    let one_seven_one_six = ctx.num(1716);
    let three_zero_zero_three_cos_double = build_scaled_cos_multiple(ctx, arg, 2, 3003);
    let two_zero_zero_two_cos_quadruple = build_scaled_cos_multiple(ctx, arg, 4, 2002);
    let one_zero_zero_one_cos_sextuple = build_scaled_cos_multiple(ctx, arg, 6, 1001);
    let three_sixty_four_cos_octuple = build_scaled_cos_multiple(ctx, arg, 8, 364);
    let ninety_one_cos_decuple = build_scaled_cos_multiple(ctx, arg, 10, 91);
    let fourteen_cos_twelvefold = build_scaled_cos_multiple(ctx, arg, 12, 14);
    let cos_fourteenfold = build_cos_multiple(ctx, arg, 14);

    let numerator = if is_sin {
        let leading = ctx.add(Expr::Sub(
            one_seven_one_six,
            three_zero_zero_three_cos_double,
        ));
        let second = ctx.add(Expr::Add(leading, two_zero_zero_two_cos_quadruple));
        let third = ctx.add(Expr::Sub(second, one_zero_zero_one_cos_sextuple));
        let fourth = ctx.add(Expr::Add(third, three_sixty_four_cos_octuple));
        let fifth = ctx.add(Expr::Sub(fourth, ninety_one_cos_decuple));
        let sixth = ctx.add(Expr::Add(fifth, fourteen_cos_twelvefold));
        ctx.add(Expr::Sub(sixth, cos_fourteenfold))
    } else {
        let leading = ctx.add(Expr::Add(
            one_seven_one_six,
            three_zero_zero_three_cos_double,
        ));
        let second = ctx.add(Expr::Add(leading, two_zero_zero_two_cos_quadruple));
        let third = ctx.add(Expr::Add(second, one_zero_zero_one_cos_sextuple));
        let fourth = ctx.add(Expr::Add(third, three_sixty_four_cos_octuple));
        let fifth = ctx.add(Expr::Add(fourth, ninety_one_cos_decuple));
        let sixth = ctx.add(Expr::Add(fifth, fourteen_cos_twelvefold));
        ctx.add(Expr::Add(sixth, cos_fourteenfold))
    };

    let eight_one_nine_two = ctx.num(8192);
    ctx.add(Expr::Div(numerator, eight_one_nine_two))
}

fn build_sixteenth_power_reduction_rewritten(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
    is_sin: bool,
) -> ExprId {
    let six_four_three_five = ctx.num(6435);
    let eleven_four_forty_cos_double = build_scaled_cos_multiple(ctx, arg, 2, 11440);
    let eight_zero_zero_eight_cos_quadruple = build_scaled_cos_multiple(ctx, arg, 4, 8008);
    let four_three_six_eight_cos_sextuple = build_scaled_cos_multiple(ctx, arg, 6, 4368);
    let one_eight_two_zero_cos_octuple = build_scaled_cos_multiple(ctx, arg, 8, 1820);
    let five_sixty_cos_decuple = build_scaled_cos_multiple(ctx, arg, 10, 560);
    let one_twenty_cos_twelvefold = build_scaled_cos_multiple(ctx, arg, 12, 120);
    let sixteen_cos_fourteenfold = build_scaled_cos_multiple(ctx, arg, 14, 16);
    let cos_sixteenfold = build_cos_multiple(ctx, arg, 16);

    let numerator = if is_sin {
        let first = ctx.add(Expr::Sub(six_four_three_five, eleven_four_forty_cos_double));
        let second = ctx.add(Expr::Add(first, eight_zero_zero_eight_cos_quadruple));
        let third = ctx.add(Expr::Sub(second, four_three_six_eight_cos_sextuple));
        let fourth = ctx.add(Expr::Add(third, one_eight_two_zero_cos_octuple));
        let fifth = ctx.add(Expr::Sub(fourth, five_sixty_cos_decuple));
        let sixth = ctx.add(Expr::Add(fifth, one_twenty_cos_twelvefold));
        let seventh = ctx.add(Expr::Sub(sixth, sixteen_cos_fourteenfold));
        ctx.add(Expr::Add(seventh, cos_sixteenfold))
    } else {
        let first = ctx.add(Expr::Add(six_four_three_five, eleven_four_forty_cos_double));
        let second = ctx.add(Expr::Add(first, eight_zero_zero_eight_cos_quadruple));
        let third = ctx.add(Expr::Add(second, four_three_six_eight_cos_sextuple));
        let fourth = ctx.add(Expr::Add(third, one_eight_two_zero_cos_octuple));
        let fifth = ctx.add(Expr::Add(fourth, five_sixty_cos_decuple));
        let sixth = ctx.add(Expr::Add(fifth, one_twenty_cos_twelvefold));
        let seventh = ctx.add(Expr::Add(sixth, sixteen_cos_fourteenfold));
        ctx.add(Expr::Add(seventh, cos_sixteenfold))
    };

    let three_two_seven_six_eight = ctx.num(32768);
    ctx.add(Expr::Div(numerator, three_two_seven_six_eight))
}

fn build_eighteenth_power_reduction_rewritten(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
    is_sin: bool,
) -> ExprId {
    let twenty_four_three_ten = ctx.num(24310);
    let forty_three_seven_fifty_eight_cos_double = build_scaled_cos_multiple(ctx, arg, 2, 43758);
    let thirty_one_eight_twenty_four_cos_quadruple = build_scaled_cos_multiple(ctx, arg, 4, 31824);
    let eighteen_five_sixty_four_cos_sextuple = build_scaled_cos_multiple(ctx, arg, 6, 18564);
    let eight_five_sixty_eight_cos_octuple = build_scaled_cos_multiple(ctx, arg, 8, 8568);
    let three_zero_sixty_cos_decuple = build_scaled_cos_multiple(ctx, arg, 10, 3060);
    let eight_sixteen_cos_twelvefold = build_scaled_cos_multiple(ctx, arg, 12, 816);
    let one_fifty_three_cos_fourteenfold = build_scaled_cos_multiple(ctx, arg, 14, 153);
    let eighteen_cos_sixteenfold = build_scaled_cos_multiple(ctx, arg, 16, 18);
    let cos_eighteenfold = build_cos_multiple(ctx, arg, 18);

    let numerator = if is_sin {
        let first = ctx.add(Expr::Sub(
            twenty_four_three_ten,
            forty_three_seven_fifty_eight_cos_double,
        ));
        let second = ctx.add(Expr::Add(first, thirty_one_eight_twenty_four_cos_quadruple));
        let third = ctx.add(Expr::Sub(second, eighteen_five_sixty_four_cos_sextuple));
        let fourth = ctx.add(Expr::Add(third, eight_five_sixty_eight_cos_octuple));
        let fifth = ctx.add(Expr::Sub(fourth, three_zero_sixty_cos_decuple));
        let sixth = ctx.add(Expr::Add(fifth, eight_sixteen_cos_twelvefold));
        let seventh = ctx.add(Expr::Sub(sixth, one_fifty_three_cos_fourteenfold));
        let eighth = ctx.add(Expr::Add(seventh, eighteen_cos_sixteenfold));
        ctx.add(Expr::Sub(eighth, cos_eighteenfold))
    } else {
        let first = ctx.add(Expr::Add(
            twenty_four_three_ten,
            forty_three_seven_fifty_eight_cos_double,
        ));
        let second = ctx.add(Expr::Add(first, thirty_one_eight_twenty_four_cos_quadruple));
        let third = ctx.add(Expr::Add(second, eighteen_five_sixty_four_cos_sextuple));
        let fourth = ctx.add(Expr::Add(third, eight_five_sixty_eight_cos_octuple));
        let fifth = ctx.add(Expr::Add(fourth, three_zero_sixty_cos_decuple));
        let sixth = ctx.add(Expr::Add(fifth, eight_sixteen_cos_twelvefold));
        let seventh = ctx.add(Expr::Add(sixth, one_fifty_three_cos_fourteenfold));
        let eighth = ctx.add(Expr::Add(seventh, eighteen_cos_sixteenfold));
        ctx.add(Expr::Add(eighth, cos_eighteenfold))
    };

    let one_three_one_zero_seven_two = ctx.num(131072);
    ctx.add(Expr::Div(numerator, one_three_one_zero_seven_two))
}

fn build_twentieth_power_reduction_rewritten(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
    is_sin: bool,
) -> ExprId {
    let ninety_two_three_seventy_eight = ctx.num(92378);
    let one_sixty_seven_nine_sixty_cos_double = build_scaled_cos_multiple(ctx, arg, 2, 167960);
    let one_twenty_five_nine_seventy_cos_quadruple = build_scaled_cos_multiple(ctx, arg, 4, 125970);
    let seventy_seven_five_twenty_cos_sextuple = build_scaled_cos_multiple(ctx, arg, 6, 77520);
    let thirty_eight_seven_sixty_cos_octuple = build_scaled_cos_multiple(ctx, arg, 8, 38760);
    let fifteen_five_zero_four_cos_decuple = build_scaled_cos_multiple(ctx, arg, 10, 15504);
    let four_eight_four_five_cos_twelvefold = build_scaled_cos_multiple(ctx, arg, 12, 4845);
    let one_one_four_zero_cos_fourteenfold = build_scaled_cos_multiple(ctx, arg, 14, 1140);
    let one_nine_zero_cos_sixteenfold = build_scaled_cos_multiple(ctx, arg, 16, 190);
    let twenty_cos_eighteenfold = build_scaled_cos_multiple(ctx, arg, 18, 20);
    let cos_twentyfold = build_cos_multiple(ctx, arg, 20);

    let numerator = if is_sin {
        let first = ctx.add(Expr::Sub(
            ninety_two_three_seventy_eight,
            one_sixty_seven_nine_sixty_cos_double,
        ));
        let second = ctx.add(Expr::Add(first, one_twenty_five_nine_seventy_cos_quadruple));
        let third = ctx.add(Expr::Sub(second, seventy_seven_five_twenty_cos_sextuple));
        let fourth = ctx.add(Expr::Add(third, thirty_eight_seven_sixty_cos_octuple));
        let fifth = ctx.add(Expr::Sub(fourth, fifteen_five_zero_four_cos_decuple));
        let sixth = ctx.add(Expr::Add(fifth, four_eight_four_five_cos_twelvefold));
        let seventh = ctx.add(Expr::Sub(sixth, one_one_four_zero_cos_fourteenfold));
        let eighth = ctx.add(Expr::Add(seventh, one_nine_zero_cos_sixteenfold));
        let ninth = ctx.add(Expr::Sub(eighth, twenty_cos_eighteenfold));
        ctx.add(Expr::Add(ninth, cos_twentyfold))
    } else {
        let first = ctx.add(Expr::Add(
            ninety_two_three_seventy_eight,
            one_sixty_seven_nine_sixty_cos_double,
        ));
        let second = ctx.add(Expr::Add(first, one_twenty_five_nine_seventy_cos_quadruple));
        let third = ctx.add(Expr::Add(second, seventy_seven_five_twenty_cos_sextuple));
        let fourth = ctx.add(Expr::Add(third, thirty_eight_seven_sixty_cos_octuple));
        let fifth = ctx.add(Expr::Add(fourth, fifteen_five_zero_four_cos_decuple));
        let sixth = ctx.add(Expr::Add(fifth, four_eight_four_five_cos_twelvefold));
        let seventh = ctx.add(Expr::Add(sixth, one_one_four_zero_cos_fourteenfold));
        let eighth = ctx.add(Expr::Add(seventh, one_nine_zero_cos_sixteenfold));
        let ninth = ctx.add(Expr::Add(eighth, twenty_cos_eighteenfold));
        ctx.add(Expr::Add(ninth, cos_twentyfold))
    };

    let five_two_four_two_eight_eight = ctx.num(524288);
    ctx.add(Expr::Div(numerator, five_two_four_two_eight_eight))
}

fn build_twenty_second_power_reduction_rewritten(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
    is_sin: bool,
) -> ExprId {
    let three_five_two_seven_one_six = ctx.num(352716);
    let six_four_six_six_four_six_cos_double = build_scaled_cos_multiple(ctx, arg, 2, 646646);
    let four_nine_seven_four_two_zero_cos_quadruple =
        build_scaled_cos_multiple(ctx, arg, 4, 497420);
    let three_one_nine_seven_seven_zero_cos_sextuple =
        build_scaled_cos_multiple(ctx, arg, 6, 319770);
    let one_seven_zero_five_four_four_cos_octuple = build_scaled_cos_multiple(ctx, arg, 8, 170544);
    let seventy_four_six_thirteen_cos_decuple = build_scaled_cos_multiple(ctx, arg, 10, 74613);
    let twenty_six_three_three_four_cos_twelvefold = build_scaled_cos_multiple(ctx, arg, 12, 26334);
    let seven_three_one_five_cos_fourteenfold = build_scaled_cos_multiple(ctx, arg, 14, 7315);
    let one_five_four_zero_cos_sixteenfold = build_scaled_cos_multiple(ctx, arg, 16, 1540);
    let two_three_one_cos_eighteenfold = build_scaled_cos_multiple(ctx, arg, 18, 231);
    let twenty_two_cos_twentyfold = build_scaled_cos_multiple(ctx, arg, 20, 22);
    let cos_twenty_twofold = build_cos_multiple(ctx, arg, 22);

    let numerator = if is_sin {
        let first = ctx.add(Expr::Sub(
            three_five_two_seven_one_six,
            six_four_six_six_four_six_cos_double,
        ));
        let second = ctx.add(Expr::Add(
            first,
            four_nine_seven_four_two_zero_cos_quadruple,
        ));
        let third = ctx.add(Expr::Sub(
            second,
            three_one_nine_seven_seven_zero_cos_sextuple,
        ));
        let fourth = ctx.add(Expr::Add(third, one_seven_zero_five_four_four_cos_octuple));
        let fifth = ctx.add(Expr::Sub(fourth, seventy_four_six_thirteen_cos_decuple));
        let sixth = ctx.add(Expr::Add(fifth, twenty_six_three_three_four_cos_twelvefold));
        let seventh = ctx.add(Expr::Sub(sixth, seven_three_one_five_cos_fourteenfold));
        let eighth = ctx.add(Expr::Add(seventh, one_five_four_zero_cos_sixteenfold));
        let ninth = ctx.add(Expr::Sub(eighth, two_three_one_cos_eighteenfold));
        let tenth = ctx.add(Expr::Add(ninth, twenty_two_cos_twentyfold));
        ctx.add(Expr::Sub(tenth, cos_twenty_twofold))
    } else {
        let first = ctx.add(Expr::Add(
            three_five_two_seven_one_six,
            six_four_six_six_four_six_cos_double,
        ));
        let second = ctx.add(Expr::Add(
            first,
            four_nine_seven_four_two_zero_cos_quadruple,
        ));
        let third = ctx.add(Expr::Add(
            second,
            three_one_nine_seven_seven_zero_cos_sextuple,
        ));
        let fourth = ctx.add(Expr::Add(third, one_seven_zero_five_four_four_cos_octuple));
        let fifth = ctx.add(Expr::Add(fourth, seventy_four_six_thirteen_cos_decuple));
        let sixth = ctx.add(Expr::Add(fifth, twenty_six_three_three_four_cos_twelvefold));
        let seventh = ctx.add(Expr::Add(sixth, seven_three_one_five_cos_fourteenfold));
        let eighth = ctx.add(Expr::Add(seventh, one_five_four_zero_cos_sixteenfold));
        let ninth = ctx.add(Expr::Add(eighth, two_three_one_cos_eighteenfold));
        let tenth = ctx.add(Expr::Add(ninth, twenty_two_cos_twentyfold));
        ctx.add(Expr::Add(tenth, cos_twenty_twofold))
    };

    let two_zero_nine_seven_one_five_two = ctx.num(2097152);
    ctx.add(Expr::Div(numerator, two_zero_nine_seven_one_five_two))
}

fn binomial_i64(n: u32, k: u32) -> Option<i64> {
    if k > n {
        return None;
    }

    let k = k.min(n - k);
    let mut result: u128 = 1;
    for i in 1..=k {
        let numerator = u128::from(n - k + i);
        let denominator = u128::from(i);
        result = result.checked_mul(numerator)? / denominator;
    }

    i64::try_from(result).ok()
}

fn build_sin_cos_square_product_reduction_rewritten(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
) -> ExprId {
    let one = ctx.num(1);
    let cos_quadruple = build_cos_multiple(ctx, arg, 4);
    let numerator = ctx.add(Expr::Sub(one, cos_quadruple));
    let eight = ctx.num(8);
    ctx.add(Expr::Div(numerator, eight))
}

fn build_sin_cos_binomial_square_rewritten(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
    same_sign: bool,
) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let double_arg = smart_mul(ctx, two, arg);
    let sin_double = ctx.call_builtin(BuiltinFn::Sin, vec![double_arg]);

    if same_sign {
        ctx.add(Expr::Add(one, sin_double))
    } else {
        ctx.add(Expr::Sub(one, sin_double))
    }
}

fn build_cos_multiple(ctx: &mut cas_ast::Context, arg: ExprId, factor: i64) -> ExprId {
    let factor_expr = ctx.num(factor);
    let scaled_arg = smart_mul(ctx, factor_expr, arg);
    ctx.call_builtin(BuiltinFn::Cos, vec![scaled_arg])
}

fn build_scaled_cos_multiple(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
    angle_factor: i64,
    coeff: i64,
) -> ExprId {
    let cos_term = build_cos_multiple(ctx, arg, angle_factor);
    let coeff_expr = ctx.num(coeff);
    smart_mul(ctx, coeff_expr, cos_term)
}

fn complementary_trig_fn(trig_fn: BuiltinFn) -> Option<BuiltinFn> {
    match trig_fn {
        BuiltinFn::Sin => Some(BuiltinFn::Cos),
        BuiltinFn::Cos => Some(BuiltinFn::Sin),
        _ => None,
    }
}

fn trig_name(trig_fn: BuiltinFn) -> &'static str {
    match trig_fn {
        BuiltinFn::Sin => "sin",
        BuiltinFn::Cos => "cos",
        BuiltinFn::Sec => "sec",
        BuiltinFn::Csc => "csc",
        _ => unreachable!("only supported trig names should reach here"),
    }
}

fn try_rewrite_product_to_sum_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    if let Some(rewrite) =
        cas_math::trig_sum_product_support::try_rewrite_product_to_sum_expr(ctx, expr)
    {
        if trig_presentational_target_match(ctx, rewrite.rewritten, target_expr) {
            return Some(DeriveTrigRewrite {
                rewritten: rewrite.rewritten,
                kind: map_product_to_sum_kind(rewrite.kind),
            });
        }

        let normalized = normalize_trig_bridge_expr(ctx, rewrite.rewritten);
        if trig_presentational_target_match(ctx, normalized, target_expr) {
            return Some(DeriveTrigRewrite {
                rewritten: normalized,
                kind: map_product_to_sum_kind(rewrite.kind),
            });
        }
    }

    try_rewrite_product_to_sum_commuted_mixed_target_aware(ctx, expr, target_expr)
}

fn try_rewrite_product_to_sum_commuted_mixed_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 3 {
        return None;
    }

    let mut two_idx = None;
    let mut sin_factor = None;
    let mut cos_factor = None;

    for (idx, factor) in factors.iter().copied().enumerate() {
        match ctx.get(factor) {
            Expr::Number(_) if is_small_integer(ctx, factor, 2) && two_idx.is_none() => {
                two_idx = Some(idx);
            }
            Expr::Function(fn_id, args)
                if args.len() == 1
                    && ctx.is_builtin(*fn_id, BuiltinFn::Sin)
                    && sin_factor.is_none() =>
            {
                sin_factor = Some((idx, args[0]));
            }
            Expr::Function(fn_id, args)
                if args.len() == 1
                    && ctx.is_builtin(*fn_id, BuiltinFn::Cos)
                    && cos_factor.is_none() =>
            {
                cos_factor = Some((idx, args[0]));
            }
            _ => {}
        }
    }

    let (two_idx, (sin_idx, sin_arg), (cos_idx, cos_arg)) = match (two_idx, sin_factor, cos_factor)
    {
        (Some(two_idx), Some(sin_factor), Some(cos_factor)) => (two_idx, sin_factor, cos_factor),
        _ => return None,
    };

    let mut remaining = Vec::new();
    for (idx, factor) in factors.iter().copied().enumerate() {
        if idx != two_idx && idx != sin_idx && idx != cos_idx {
            remaining.push(factor);
        }
    }

    let cos_sin = build_product_to_sum_candidate(
        ctx,
        cos_arg,
        sin_arg,
        &remaining,
        DeriveTrigRewriteKind::ProductToSumCosSin,
    );
    if trig_presentational_target_match(ctx, cos_sin.rewritten, target_expr) {
        return Some(cos_sin);
    }
    let normalized_cos_sin = DeriveTrigRewrite {
        rewritten: normalize_trig_bridge_expr(ctx, cos_sin.rewritten),
        kind: cos_sin.kind,
    };
    if trig_presentational_target_match(ctx, normalized_cos_sin.rewritten, target_expr) {
        return Some(normalized_cos_sin);
    }

    let sin_cos = build_product_to_sum_candidate(
        ctx,
        sin_arg,
        cos_arg,
        &remaining,
        DeriveTrigRewriteKind::ProductToSumSinCos,
    );
    if trig_presentational_target_match(ctx, sin_cos.rewritten, target_expr) {
        return Some(sin_cos);
    }
    let normalized_sin_cos = DeriveTrigRewrite {
        rewritten: normalize_trig_bridge_expr(ctx, sin_cos.rewritten),
        kind: sin_cos.kind,
    };
    if trig_presentational_target_match(ctx, normalized_sin_cos.rewritten, target_expr) {
        return Some(normalized_sin_cos);
    }

    None
}

fn build_product_to_sum_candidate(
    ctx: &mut cas_ast::Context,
    arg_a: ExprId,
    arg_b: ExprId,
    remaining: &[ExprId],
    kind: DeriveTrigRewriteKind,
) -> DeriveTrigRewrite {
    let sum_arg = ctx.add(Expr::Add(arg_a, arg_b));
    let diff_arg = ctx.add(Expr::Sub(arg_a, arg_b));

    let rewritten = match kind {
        DeriveTrigRewriteKind::ProductToSumSinCos => {
            let sin_sum = ctx.call_builtin(BuiltinFn::Sin, vec![sum_arg]);
            let sin_diff = ctx.call_builtin(BuiltinFn::Sin, vec![diff_arg]);
            ctx.add(Expr::Add(sin_sum, sin_diff))
        }
        DeriveTrigRewriteKind::ProductToSumCosSin => {
            let sin_sum = ctx.call_builtin(BuiltinFn::Sin, vec![sum_arg]);
            let sin_diff = ctx.call_builtin(BuiltinFn::Sin, vec![diff_arg]);
            ctx.add(Expr::Sub(sin_sum, sin_diff))
        }
        DeriveTrigRewriteKind::ProductToSumCosCos => {
            let cos_sum = ctx.call_builtin(BuiltinFn::Cos, vec![sum_arg]);
            let cos_diff = ctx.call_builtin(BuiltinFn::Cos, vec![diff_arg]);
            ctx.add(Expr::Add(cos_sum, cos_diff))
        }
        DeriveTrigRewriteKind::ProductToSumSinSin => {
            let cos_sum = ctx.call_builtin(BuiltinFn::Cos, vec![sum_arg]);
            let cos_diff = ctx.call_builtin(BuiltinFn::Cos, vec![diff_arg]);
            ctx.add(Expr::Sub(cos_diff, cos_sum))
        }
        _ => unreachable!("only product-to-sum rewrite kinds are valid here"),
    };

    let rewritten = remaining
        .iter()
        .copied()
        .fold(rewritten, |acc, factor| smart_mul(ctx, acc, factor));

    DeriveTrigRewrite { rewritten, kind }
}

pub(crate) fn try_rewrite_trig_contraction_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    if let Some(rewrite) = try_rewrite_phase_shift_contraction_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_normalized_reciprocal_trig_contraction_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_negated_angle_sum_diff_contraction_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_angle_sum_diff_contraction_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_tangent_angle_sum_diff_contraction_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_negated_reciprocal_trig_contraction_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_reciprocal_trig_contraction_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_tangent_double_angle_contraction_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_half_angle_square_contraction_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_negated_half_angle_square_contraction_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        cas_math::trig_power_identity_support::try_rewrite_recognize_sec_squared_add_expr(ctx, expr)
    {
        if strong_target_match(ctx, rewrite.rewritten, target_expr) {
            return Some(DeriveTrigRewrite {
                rewritten: rewrite.rewritten,
                kind: DeriveTrigRewriteKind::RecognizeSecSquared,
            });
        }
    }

    if let Some(rewrite) =
        cas_math::trig_power_identity_support::try_rewrite_recognize_csc_squared_add_expr(ctx, expr)
    {
        if strong_target_match(ctx, rewrite.rewritten, target_expr) {
            return Some(DeriveTrigRewrite {
                rewritten: rewrite.rewritten,
                kind: DeriveTrigRewriteKind::RecognizeCscSquared,
            });
        }
    }

    if let Some(rewrite) =
        try_rewrite_negated_half_angle_tangent_contraction_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_mixed_sine_double_angle_product_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_half_scaled_sine_double_angle_contraction_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        cas_math::trig_contraction_support::try_rewrite_half_angle_tangent_div_expr(ctx, expr)
    {
        if strong_target_match(ctx, rewrite.rewritten, target_expr) {
            return Some(DeriveTrigRewrite {
                rewritten: rewrite.rewritten,
                kind: DeriveTrigRewriteKind::HalfAngleTangent,
            });
        }
    }

    if let Some(rewrite) =
        cas_math::trig_contraction_support::try_rewrite_double_angle_contraction_expr(ctx, expr)
    {
        if strong_target_match(ctx, rewrite.rewritten, target_expr) {
            if let Some(kind) = match rewrite.kind {
                cas_math::trig_contraction_support::TrigContractionRewriteKind::DoubleAngleSin => {
                    Some(DeriveTrigRewriteKind::DoubleAngleSin)
                }
                cas_math::trig_contraction_support::TrigContractionRewriteKind::DoubleAngleCos => {
                    Some(DeriveTrigRewriteKind::DoubleAngleCos)
                }
                _ => None,
            } {
                return Some(DeriveTrigRewrite {
                    rewritten: rewrite.rewritten,
                    kind,
                });
            }
        }
    }

    if let Some(rewrite) = try_rewrite_triple_angle_contraction_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_quintuple_angle_contraction_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_negated_sin_double_angle_contraction_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_negated_triple_angle_contraction_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_negated_quintuple_angle_contraction_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        cas_math::trig_contraction_support::try_rewrite_cos2x_additive_contraction_expr(ctx, expr)
    {
        if strong_target_match(ctx, rewrite.rewritten, target_expr) {
            if let Some(kind) = match rewrite.kind {
                cas_math::trig_contraction_support::TrigContractionRewriteKind::Cos2xAdditiveSin => {
                    Some(DeriveTrigRewriteKind::DoubleAngleCosOneMinusTwoSinSq)
                }
                cas_math::trig_contraction_support::TrigContractionRewriteKind::Cos2xAdditiveCos => {
                    Some(DeriveTrigRewriteKind::DoubleAngleCosTwoCosSqMinusOne)
                }
                _ => None,
            } {
                return Some(DeriveTrigRewrite {
                    rewritten: rewrite.rewritten,
                    kind,
                });
            }
        }
    }

    if let Some(rewrite) =
        try_rewrite_negated_cos_double_angle_contraction_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    None
}

fn try_rewrite_phase_shift_contraction_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    if contains_phase_shift_term(ctx, expr) && contains_phase_shift_term(ctx, target_expr) {
        for rewrite in generate_phase_shift_bridge_rewrites(ctx, expr) {
            if phase_shift_target_match(ctx, rewrite.rewritten, target_expr) {
                return Some(DeriveTrigRewrite {
                    rewritten: target_expr,
                    kind: DeriveTrigRewriteKind::PhaseShiftIdentity,
                });
            }
        }
    }

    let (source_arg, source_sin_coeff, source_cos_coeff, source_sin_sign, source_cos_sign) =
        extract_weighted_phase_shift_linear_combination(ctx, expr)?;
    let source_linear = build_weighted_phase_shift_linear_combination(
        ctx,
        source_arg,
        source_sin_coeff,
        source_cos_coeff,
        source_sin_sign,
        source_cos_sign,
    )?;
    let source_linear = run_phase_shift_match_simplify(ctx, source_linear);

    if let Some(target_data) = extract_general_phase_shift_term_data(ctx, target_expr) {
        let expanded = build_general_phase_shift_linear_combination(ctx, target_data)?;
        let expanded = run_phase_shift_match_simplify(ctx, expanded);
        if phase_shift_target_match(ctx, source_linear, expanded) {
            return Some(DeriveTrigRewrite {
                rewritten: target_expr,
                kind: DeriveTrigRewriteKind::PhaseShiftIdentity,
            });
        }
    }

    if let Some((arg, coeff, kind, sin_sign, cos_sign)) =
        extract_phase_shift_linear_combination(ctx, expr)
    {
        if let Some((target_arg, target_coeff, target_kind, target_sin_sign, target_cos_sign)) =
            extract_phase_shift_term_data(ctx, target_expr)
        {
            let expanded = build_phase_shift_linear_combination(
                ctx,
                target_coeff,
                target_arg,
                target_kind,
                target_sin_sign,
                target_cos_sign,
            );
            if strong_target_match(ctx, source_linear, expanded) {
                return Some(DeriveTrigRewrite {
                    rewritten: target_expr,
                    kind: DeriveTrigRewriteKind::PhaseShiftIdentity,
                });
            }
        }

        for candidate in build_phase_shift_candidates(ctx, coeff, arg, kind, sin_sign, cos_sign) {
            if strong_target_match(ctx, candidate, target_expr) {
                return Some(DeriveTrigRewrite {
                    rewritten: target_expr,
                    kind: DeriveTrigRewriteKind::PhaseShiftIdentity,
                });
            }
        }
    }

    None
}

fn try_rewrite_phase_shift_expansion_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let (target_arg, target_sin_coeff, target_cos_coeff, target_sin_sign, target_cos_sign) =
        extract_weighted_phase_shift_linear_combination(ctx, target_expr)?;
    let target_linear = build_weighted_phase_shift_linear_combination(
        ctx,
        target_arg,
        target_sin_coeff,
        target_cos_coeff,
        target_sin_sign,
        target_cos_sign,
    )?;
    let target_linear = run_phase_shift_match_simplify(ctx, target_linear);

    if let Some(source_data) = extract_general_phase_shift_term_data(ctx, expr) {
        let expanded = build_general_phase_shift_linear_combination(ctx, source_data)?;
        let expanded = run_phase_shift_match_simplify(ctx, expanded);
        if trig_presentational_target_match(ctx, expanded, target_linear) {
            return Some(DeriveTrigRewrite {
                rewritten: target_expr,
                kind: DeriveTrigRewriteKind::PhaseShiftIdentity,
            });
        }
    }

    if let Some((arg, coeff, kind, sin_sign, cos_sign)) =
        extract_phase_shift_linear_combination(ctx, target_expr)
    {
        if let Some((source_arg, source_coeff, source_kind, source_sin_sign, source_cos_sign)) =
            extract_phase_shift_term_data(ctx, expr)
        {
            let expanded = build_phase_shift_linear_combination(
                ctx,
                source_coeff,
                source_arg,
                source_kind,
                source_sin_sign,
                source_cos_sign,
            );
            if strong_target_match(ctx, expanded, target_linear) {
                return Some(DeriveTrigRewrite {
                    rewritten: target_expr,
                    kind: DeriveTrigRewriteKind::PhaseShiftIdentity,
                });
            }
        }

        for candidate in build_phase_shift_candidates(ctx, coeff, arg, kind, sin_sign, cos_sign) {
            if strong_target_match(ctx, expr, candidate) {
                return Some(DeriveTrigRewrite {
                    rewritten: target_expr,
                    kind: DeriveTrigRewriteKind::PhaseShiftIdentity,
                });
            }
        }
    }

    None
}

fn try_rewrite_cofunction_phase_shift_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let rewrite =
        cas_math::trig_phase_shift_support::try_rewrite_trig_phase_shift_function_expr(ctx, expr)?;
    if !matches!(
        rewrite.shift,
        cas_math::trig_phase_shift_support::TrigPhaseShiftKind::PiOver2
            | cas_math::trig_phase_shift_support::TrigPhaseShiftKind::NegPiOver2
            | cas_math::trig_phase_shift_support::TrigPhaseShiftKind::ThreePiOver2
            | cas_math::trig_phase_shift_support::TrigPhaseShiftKind::NegThreePiOver2
    ) {
        return None;
    }

    let rewritten = run_phase_shift_match_simplify(ctx, rewrite.rewritten);
    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten: target_expr,
        kind: DeriveTrigRewriteKind::CofunctionIdentity,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhaseShiftKind {
    Quarter,
    Sixth,
    Third,
}

fn extract_phase_shift_linear_combination(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, PhaseShiftKind, i8, i8)> {
    let (sin_arg, sin_coeff, cos_coeff, sin_sign, cos_sign) =
        extract_weighted_phase_shift_linear_combination(ctx, expr)?;

    let three = ctx.num(3);
    let sqrt_three = ctx.call_builtin(BuiltinFn::Sqrt, vec![three]);
    let (coeff, kind) = if compare_expr(ctx, sin_coeff, cos_coeff) == Ordering::Equal {
        (sin_coeff, PhaseShiftKind::Quarter)
    } else {
        let sin_times_sqrt_three = smart_mul(ctx, sin_coeff, sqrt_three);
        let cos_times_sqrt_three = smart_mul(ctx, cos_coeff, sqrt_three);
        if strong_target_match(ctx, sin_times_sqrt_three, cos_coeff) {
            (sin_coeff, PhaseShiftKind::Third)
        } else if strong_target_match(ctx, cos_times_sqrt_three, sin_coeff) {
            (cos_coeff, PhaseShiftKind::Sixth)
        } else {
            return None;
        }
    };

    Some((sin_arg, coeff, kind, sin_sign, cos_sign))
}

fn extract_weighted_phase_shift_linear_combination(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, ExprId, i8, i8)> {
    let terms = add_terms_signed(ctx, expr);
    if terms.len() != 2 {
        return None;
    }

    let mut sin_term: Option<(ExprId, ExprId, i8)> = None;
    let mut cos_term: Option<(ExprId, ExprId, i8)> = None;

    for (term, sign) in terms {
        let signed = match sign {
            Sign::Pos => 1,
            Sign::Neg => -1,
        };

        let (trig_fn, arg, coeff) = extract_scaled_sin_or_cos_linear_term(ctx, term)?;
        let (coeff, signed) = normalize_phase_shift_coefficient_sign(ctx, coeff, signed);
        match trig_fn {
            BuiltinFn::Sin => {
                if sin_term.is_some() {
                    return None;
                }
                sin_term = Some((arg, coeff, signed));
            }
            BuiltinFn::Cos => {
                if cos_term.is_some() {
                    return None;
                }
                cos_term = Some((arg, coeff, signed));
            }
            _ => return None,
        }
    }

    let (sin_arg, sin_coeff, sin_sign) = sin_term?;
    let (cos_arg, cos_coeff, cos_sign) = cos_term?;
    if compare_expr(ctx, sin_arg, cos_arg) != Ordering::Equal {
        return None;
    }

    Some((sin_arg, sin_coeff, cos_coeff, sin_sign, cos_sign))
}

fn build_phase_shift_candidates(
    ctx: &mut cas_ast::Context,
    coeff: ExprId,
    arg: ExprId,
    kind: PhaseShiftKind,
    sin_sign: i8,
    cos_sign: i8,
) -> Vec<ExprId> {
    let target_linear =
        build_phase_shift_linear_combination(ctx, coeff, arg, kind, sin_sign, cos_sign);
    let mut candidates = Vec::new();

    for candidate in generate_phase_shift_term_candidates_for_kind(ctx, coeff, arg, kind) {
        let Some((
            candidate_arg,
            candidate_coeff,
            candidate_kind,
            candidate_sin_sign,
            candidate_cos_sign,
        )) = extract_phase_shift_term_data(ctx, candidate)
        else {
            continue;
        };
        let rebuilt = build_phase_shift_linear_combination(
            ctx,
            candidate_coeff,
            candidate_arg,
            candidate_kind,
            candidate_sin_sign,
            candidate_cos_sign,
        );
        if !strong_target_match(ctx, rebuilt, target_linear) {
            continue;
        }
        if candidates
            .iter()
            .any(|existing| compare_expr(ctx, *existing, candidate) == Ordering::Equal)
        {
            continue;
        }
        candidates.push(candidate);
    }

    candidates
}

fn build_general_phase_shift_cosine_term_candidate(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
    sin_coeff: ExprId,
    cos_coeff: ExprId,
    sin_sign: i8,
    cos_sign: i8,
) -> Option<ExprId> {
    let ratio_raw = ctx.add(Expr::Div(sin_coeff, cos_coeff));
    let ratio = run_phase_shift_match_simplify(ctx, ratio_raw);
    let shift = ctx.call_builtin(BuiltinFn::Arctan, vec![ratio]);
    let shifted_arg = if sin_sign == cos_sign {
        ctx.add(Expr::Sub(arg, shift))
    } else {
        ctx.add(Expr::Add(arg, shift))
    };

    let sin_sq = smart_mul(ctx, sin_coeff, sin_coeff);
    let cos_sq = smart_mul(ctx, cos_coeff, cos_coeff);
    let amplitude_sq = ctx.add(Expr::Add(sin_sq, cos_sq));
    let amplitude_raw = ctx.call_builtin(BuiltinFn::Sqrt, vec![amplitude_sq]);
    let amplitude = run_phase_shift_match_simplify(ctx, amplitude_raw);
    let shifted_cosine = ctx.call_builtin(BuiltinFn::Cos, vec![shifted_arg]);
    let rewritten = smart_mul(ctx, amplitude, shifted_cosine);
    let rewritten = if cos_sign < 0 {
        ctx.add(Expr::Neg(rewritten))
    } else {
        rewritten
    };

    Some(run_phase_shift_match_simplify(ctx, rewritten))
}

fn generate_general_phase_shift_term_candidates(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
    sin_coeff: ExprId,
    cos_coeff: ExprId,
    sin_sign: i8,
    cos_sign: i8,
) -> Vec<ExprId> {
    let mut candidates = Vec::new();

    if let Some(rewritten) =
        build_general_phase_shift_term_candidate(ctx, arg, sin_coeff, cos_coeff, sin_sign, cos_sign)
    {
        candidates.push(rewritten);
    }

    if let Some(rewritten) = build_general_phase_shift_cosine_term_candidate(
        ctx, arg, sin_coeff, cos_coeff, sin_sign, cos_sign,
    ) {
        if candidates
            .iter()
            .all(|existing| compare_expr(ctx, *existing, rewritten) != Ordering::Equal)
        {
            candidates.push(rewritten);
        }
    }

    candidates
}

fn push_phase_shift_term_bridge_rewrites_from_linear_combination(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    linear_expr: ExprId,
    rewrites: &mut Vec<DeriveTrigRewrite>,
) {
    if let Some((arg, coeff, kind, sin_sign, cos_sign)) =
        extract_phase_shift_linear_combination(ctx, linear_expr)
    {
        for rewritten in build_phase_shift_candidates(ctx, coeff, arg, kind, sin_sign, cos_sign) {
            if compare_expr(ctx, rewritten, source_expr) == Ordering::Equal {
                continue;
            }

            push_unique_trig_bridge_rewrite(
                rewrites,
                DeriveTrigRewrite {
                    rewritten,
                    kind: DeriveTrigRewriteKind::PhaseShiftIdentity,
                },
            );
        }

        return;
    }

    if let Some((arg, sin_coeff, cos_coeff, sin_sign, cos_sign)) =
        extract_weighted_phase_shift_linear_combination(ctx, linear_expr)
    {
        for rewritten in generate_general_phase_shift_term_candidates(
            ctx, arg, sin_coeff, cos_coeff, sin_sign, cos_sign,
        ) {
            if compare_expr(ctx, rewritten, source_expr) == Ordering::Equal {
                continue;
            }

            push_unique_trig_bridge_rewrite(
                rewrites,
                DeriveTrigRewrite {
                    rewritten,
                    kind: DeriveTrigRewriteKind::PhaseShiftIdentity,
                },
            );
        }
    }
}

fn generate_phase_shift_bridge_rewrites(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Vec<DeriveTrigRewrite> {
    let mut rewrites = Vec::new();

    if let Some((arg, coeff, kind, sin_sign, cos_sign)) =
        extract_phase_shift_linear_combination(ctx, expr)
    {
        for rewritten in build_phase_shift_candidates(ctx, coeff, arg, kind, sin_sign, cos_sign) {
            push_unique_trig_bridge_rewrite(
                &mut rewrites,
                DeriveTrigRewrite {
                    rewritten,
                    kind: DeriveTrigRewriteKind::PhaseShiftIdentity,
                },
            );
        }
    }

    if let Some((arg, sin_coeff, cos_coeff, sin_sign, cos_sign)) =
        extract_weighted_phase_shift_linear_combination(ctx, expr)
    {
        for rewritten in generate_general_phase_shift_term_candidates(
            ctx, arg, sin_coeff, cos_coeff, sin_sign, cos_sign,
        ) {
            push_unique_trig_bridge_rewrite(
                &mut rewrites,
                DeriveTrigRewrite {
                    rewritten,
                    kind: DeriveTrigRewriteKind::PhaseShiftIdentity,
                },
            );
        }
    }

    if let Some((arg, coeff, kind, sin_sign, cos_sign)) = extract_phase_shift_term_data(ctx, expr) {
        let rewritten =
            build_phase_shift_linear_combination(ctx, coeff, arg, kind, sin_sign, cos_sign);
        push_unique_trig_bridge_rewrite(
            &mut rewrites,
            DeriveTrigRewrite {
                rewritten,
                kind: DeriveTrigRewriteKind::PhaseShiftIdentity,
            },
        );

        push_phase_shift_term_bridge_rewrites_from_linear_combination(
            ctx,
            expr,
            rewritten,
            &mut rewrites,
        );
    }

    if let Some(data) = extract_general_phase_shift_term_data(ctx, expr) {
        if let Some(rewritten) = build_general_phase_shift_linear_combination(ctx, data) {
            let rewritten = run_phase_shift_match_simplify(ctx, rewritten);
            push_unique_trig_bridge_rewrite(
                &mut rewrites,
                DeriveTrigRewrite {
                    rewritten,
                    kind: DeriveTrigRewriteKind::PhaseShiftIdentity,
                },
            );

            push_phase_shift_term_bridge_rewrites_from_linear_combination(
                ctx,
                expr,
                rewritten,
                &mut rewrites,
            );
        }
    }

    rewrites
}

fn extract_scaled_sin_or_cos_linear_term(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId, ExprId)> {
    if let Some((trig_fn, arg)) = extract_sin_or_cos_linear_term(ctx, expr) {
        return Some((trig_fn, arg, ctx.num(1)));
    }

    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    let mut trig_term: Option<(BuiltinFn, ExprId)> = None;
    let mut coeff_factors = Vec::new();

    for factor in factors {
        if let Some((trig_fn, arg)) = extract_sin_or_cos_linear_term(ctx, factor) {
            if trig_term.is_some() {
                return None;
            }
            trig_term = Some((trig_fn, arg));
        } else {
            coeff_factors.push(factor);
        }
    }

    let (trig_fn, arg) = trig_term?;
    if coeff_factors.is_empty() {
        return None;
    }

    let coeff = coeff_factors
        .into_iter()
        .fold(ctx.num(1), |acc, factor| smart_mul(ctx, acc, factor));

    Some((trig_fn, arg, coeff))
}

fn extract_phase_shift_term_data(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, PhaseShiftKind, i8, i8)> {
    let (global_sign, positive_expr) = if let Some(inner) = strip_unit_negation(ctx, expr) {
        (-1_i8, inner)
    } else {
        (1_i8, expr)
    };

    let factors = flatten_mul_chain(ctx, positive_expr);
    if factors.len() < 2 {
        return None;
    }

    let mut trig_factor = None;
    let mut has_sqrt_two = false;
    let mut coeff_factors = Vec::new();

    for factor in factors {
        if trig_factor.is_none() {
            if let Some((trig_fn, arg)) = extract_sin_or_cos_linear_term(ctx, factor) {
                trig_factor = Some((trig_fn, arg));
                continue;
            }
        }

        if !has_sqrt_two && is_sqrt_two(ctx, factor) {
            has_sqrt_two = true;
            continue;
        }

        coeff_factors.push(factor);
    }

    let (trig_fn, raw_arg) = trig_factor?;
    let (base_arg, kind, subtract_shift) =
        extract_supported_phase_shift_argument(ctx, trig_fn, raw_arg)?;

    let coeff_expr = coeff_factors
        .into_iter()
        .fold(ctx.num(1), |acc, factor| smart_mul(ctx, acc, factor));

    let coeff = match kind {
        PhaseShiftKind::Quarter => {
            if !has_sqrt_two {
                return None;
            }
            coeff_expr
        }
        PhaseShiftKind::Sixth | PhaseShiftKind::Third => {
            split_out_small_integer_factor(ctx, coeff_expr, 2)?
        }
    };

    let (sin_sign, cos_sign) = match (trig_fn, subtract_shift) {
        (BuiltinFn::Sin, false) => (global_sign, global_sign),
        (BuiltinFn::Sin, true) => (global_sign, -global_sign),
        (BuiltinFn::Cos, false) => (-global_sign, global_sign),
        (BuiltinFn::Cos, true) => (global_sign, global_sign),
        _ => return None,
    };

    Some((base_arg, coeff, kind, sin_sign, cos_sign))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GeneralPhaseShiftTermData {
    coeff: ExprId,
    trig_fn: BuiltinFn,
    base_arg: ExprId,
    ratio: ExprId,
    subtract_shift: bool,
    global_sign: i8,
}

fn extract_general_phase_shift_term_data(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<GeneralPhaseShiftTermData> {
    let (global_sign, positive_expr) = if let Some(inner) = strip_unit_negation(ctx, expr) {
        (-1_i8, inner)
    } else {
        (1_i8, expr)
    };

    if let Some((trig_fn, raw_arg)) = extract_sin_or_cos_linear_term(ctx, positive_expr) {
        let (base_arg, ratio, subtract_shift) = extract_general_phase_shift_argument(ctx, raw_arg)?;
        return Some(GeneralPhaseShiftTermData {
            coeff: ctx.num(1),
            trig_fn,
            base_arg,
            ratio,
            subtract_shift,
            global_sign,
        });
    }

    let factors = flatten_mul_chain(ctx, positive_expr);
    if factors.len() < 2 {
        return None;
    }

    let mut trig_term = None;
    let mut coeff_factors = Vec::new();

    for factor in factors {
        if trig_term.is_none() {
            if let Some((trig_fn, raw_arg)) = extract_sin_or_cos_linear_term(ctx, factor) {
                trig_term = Some((trig_fn, raw_arg));
                continue;
            }
        }
        coeff_factors.push(factor);
    }

    let (trig_fn, raw_arg) = trig_term?;
    let (base_arg, ratio, subtract_shift) = extract_general_phase_shift_argument(ctx, raw_arg)?;
    let coeff = coeff_factors
        .into_iter()
        .fold(ctx.num(1), |acc, factor| smart_mul(ctx, acc, factor));
    let (coeff, global_sign) = normalize_phase_shift_coefficient_sign(ctx, coeff, global_sign);

    Some(GeneralPhaseShiftTermData {
        coeff,
        trig_fn,
        base_arg,
        ratio,
        subtract_shift,
        global_sign,
    })
}

pub(crate) fn contains_phase_shift_term(ctx: &mut cas_ast::Context, expr: ExprId) -> bool {
    if extract_phase_shift_term_data(ctx, expr).is_some()
        || extract_general_phase_shift_term_data(ctx, expr).is_some()
    {
        return true;
    }

    add_terms_signed(ctx, expr)
        .into_iter()
        .any(|(term, _sign)| {
            extract_phase_shift_term_data(ctx, term).is_some()
                || extract_general_phase_shift_term_data(ctx, term).is_some()
        })
}

fn normalize_phase_shift_coefficient_sign(
    ctx: &mut cas_ast::Context,
    coeff: ExprId,
    sign: i8,
) -> (ExprId, i8) {
    if let Some(inner) = strip_unit_negation(ctx, coeff) {
        (inner, -sign)
    } else {
        (coeff, sign)
    }
}

fn build_phase_shift_linear_combination(
    ctx: &mut cas_ast::Context,
    coeff: ExprId,
    arg: ExprId,
    kind: PhaseShiftKind,
    sin_sign: i8,
    cos_sign: i8,
) -> ExprId {
    let three = ctx.num(3);
    let sqrt_three = ctx.call_builtin(BuiltinFn::Sqrt, vec![three]);
    let (sin_coeff, cos_coeff) = match kind {
        PhaseShiftKind::Quarter => (coeff, coeff),
        PhaseShiftKind::Third => (coeff, smart_mul(ctx, coeff, sqrt_three)),
        PhaseShiftKind::Sixth => (smart_mul(ctx, coeff, sqrt_three), coeff),
    };

    let sin_call = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
    let cos_call = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
    let sin_term = smart_mul(ctx, sin_coeff, sin_call);
    let cos_term = smart_mul(ctx, cos_coeff, cos_call);
    let signed_sin = if sin_sign < 0 {
        ctx.add(Expr::Neg(sin_term))
    } else {
        sin_term
    };
    let signed_cos = if cos_sign < 0 {
        ctx.add(Expr::Neg(cos_term))
    } else {
        cos_term
    };

    ctx.add(Expr::Add(signed_sin, signed_cos))
}

fn build_weighted_phase_shift_linear_combination(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
    sin_coeff: ExprId,
    cos_coeff: ExprId,
    sin_sign: i8,
    cos_sign: i8,
) -> Option<ExprId> {
    let sin_call = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
    let cos_call = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
    let sin_term = smart_mul(ctx, sin_coeff, sin_call);
    let cos_term = smart_mul(ctx, cos_coeff, cos_call);
    let signed_sin = if sin_sign < 0 {
        ctx.add(Expr::Neg(sin_term))
    } else {
        sin_term
    };
    let signed_cos = if cos_sign < 0 {
        ctx.add(Expr::Neg(cos_term))
    } else {
        cos_term
    };

    Some(ctx.add(Expr::Add(signed_sin, signed_cos)))
}

fn build_general_phase_shift_linear_combination(
    ctx: &mut cas_ast::Context,
    data: GeneralPhaseShiftTermData,
) -> Option<ExprId> {
    let Expr::Function(atan_fn, atan_args) = ctx.get(data.ratio) else {
        return None;
    };
    if atan_args.len() != 1
        || !matches!(
            ctx.builtin_of(*atan_fn),
            Some(BuiltinFn::Atan | BuiltinFn::Arctan)
        )
    {
        return None;
    }
    let ratio_arg = atan_args[0];
    let (sin_shift, _) = cas_math::trig_inverse_expansion_support::expand_trig_inverse_composition(
        ctx, "sin", "arctan", ratio_arg,
    )?;
    let (cos_shift, _) = cas_math::trig_inverse_expansion_support::expand_trig_inverse_composition(
        ctx, "cos", "arctan", ratio_arg,
    )?;

    let sin_coeff = smart_mul(ctx, data.coeff, cos_shift);
    let cos_coeff = smart_mul(ctx, data.coeff, sin_shift);

    let (sin_sign, cos_sign) = match (data.trig_fn, data.subtract_shift) {
        (BuiltinFn::Sin, false) => (data.global_sign, data.global_sign),
        (BuiltinFn::Sin, true) => (data.global_sign, -data.global_sign),
        (BuiltinFn::Cos, false) => (-data.global_sign, data.global_sign),
        (BuiltinFn::Cos, true) => (data.global_sign, data.global_sign),
        _ => return None,
    };

    build_weighted_phase_shift_linear_combination(
        ctx,
        data.base_arg,
        sin_coeff,
        cos_coeff,
        sin_sign,
        cos_sign,
    )
}

fn build_general_phase_shift_term_candidate(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
    sin_coeff: ExprId,
    cos_coeff: ExprId,
    sin_sign: i8,
    cos_sign: i8,
) -> Option<ExprId> {
    let ratio_raw = ctx.add(Expr::Div(cos_coeff, sin_coeff));
    let ratio = run_phase_shift_match_simplify(ctx, ratio_raw);
    let shift = ctx.call_builtin(BuiltinFn::Arctan, vec![ratio]);
    let shifted_arg = if sin_sign == cos_sign {
        ctx.add(Expr::Add(arg, shift))
    } else {
        ctx.add(Expr::Sub(arg, shift))
    };

    let sin_sq = smart_mul(ctx, sin_coeff, sin_coeff);
    let cos_sq = smart_mul(ctx, cos_coeff, cos_coeff);
    let amplitude_sq = ctx.add(Expr::Add(sin_sq, cos_sq));
    let amplitude_raw = ctx.call_builtin(BuiltinFn::Sqrt, vec![amplitude_sq]);
    let amplitude = run_phase_shift_match_simplify(ctx, amplitude_raw);
    let shifted_sine = ctx.call_builtin(BuiltinFn::Sin, vec![shifted_arg]);
    let rewritten = smart_mul(ctx, amplitude, shifted_sine);
    let rewritten = if sin_sign < 0 {
        ctx.add(Expr::Neg(rewritten))
    } else {
        rewritten
    };

    Some(run_phase_shift_match_simplify(ctx, rewritten))
}

fn run_phase_shift_match_simplify(ctx: &mut cas_ast::Context, expr: ExprId) -> ExprId {
    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let (rewritten, _steps, _stats) = simplifier.simplify_with_stats(
        expr,
        crate::SimplifyOptions {
            suppress_depth_overflow_warnings: true,
            ..crate::SimplifyOptions::default()
        },
    );
    std::mem::swap(&mut simplifier.context, ctx);
    rewritten
}

fn generate_phase_shift_term_candidates_for_kind(
    ctx: &mut cas_ast::Context,
    coeff: ExprId,
    arg: ExprId,
    kind: PhaseShiftKind,
) -> Vec<ExprId> {
    let amplitude = match kind {
        PhaseShiftKind::Quarter => {
            let two = ctx.num(2);
            ctx.call_builtin(BuiltinFn::Sqrt, vec![two])
        }
        PhaseShiftKind::Sixth | PhaseShiftKind::Third => ctx.num(2),
    };
    let scaled_coeff = smart_mul(ctx, coeff, amplitude);
    let mut candidates = Vec::new();

    match kind {
        PhaseShiftKind::Quarter => {
            generate_signed_shifted_phase_terms(
                ctx,
                scaled_coeff,
                arg,
                BuiltinFn::Sin,
                4,
                &mut candidates,
            );
            generate_signed_shifted_phase_terms(
                ctx,
                scaled_coeff,
                arg,
                BuiltinFn::Cos,
                4,
                &mut candidates,
            );
        }
        PhaseShiftKind::Third => {
            generate_signed_shifted_phase_terms(
                ctx,
                scaled_coeff,
                arg,
                BuiltinFn::Sin,
                3,
                &mut candidates,
            );
            generate_signed_shifted_phase_terms(
                ctx,
                scaled_coeff,
                arg,
                BuiltinFn::Cos,
                6,
                &mut candidates,
            );
        }
        PhaseShiftKind::Sixth => {
            generate_signed_shifted_phase_terms(
                ctx,
                scaled_coeff,
                arg,
                BuiltinFn::Sin,
                6,
                &mut candidates,
            );
            generate_signed_shifted_phase_terms(
                ctx,
                scaled_coeff,
                arg,
                BuiltinFn::Cos,
                3,
                &mut candidates,
            );
        }
    }

    candidates
}

fn generate_signed_shifted_phase_terms(
    ctx: &mut cas_ast::Context,
    coeff: ExprId,
    arg: ExprId,
    trig_fn: BuiltinFn,
    denominator: i64,
    candidates: &mut Vec<ExprId>,
) {
    let plus = build_scaled_shifted_phase_term(ctx, coeff, arg, trig_fn, denominator, false, false);
    let minus = build_scaled_shifted_phase_term(ctx, coeff, arg, trig_fn, denominator, true, false);
    let neg_plus =
        build_scaled_shifted_phase_term(ctx, coeff, arg, trig_fn, denominator, false, true);
    let neg_minus =
        build_scaled_shifted_phase_term(ctx, coeff, arg, trig_fn, denominator, true, true);

    for candidate in [plus, minus, neg_plus, neg_minus] {
        if candidates
            .iter()
            .any(|existing| compare_expr(ctx, *existing, candidate) == Ordering::Equal)
        {
            continue;
        }
        candidates.push(candidate);
    }
}

fn build_scaled_shifted_phase_term(
    ctx: &mut cas_ast::Context,
    coeff: ExprId,
    arg: ExprId,
    trig_fn: BuiltinFn,
    denominator: i64,
    subtract_shift: bool,
    negate: bool,
) -> ExprId {
    let shift = build_pi_over(ctx, denominator);
    let shifted_arg = combine_linear_angle_terms(ctx, arg, shift, subtract_shift);
    let trig_call = ctx.call_builtin(trig_fn, vec![shifted_arg]);
    let scaled = smart_mul(ctx, coeff, trig_call);
    if negate {
        ctx.add(Expr::Neg(scaled))
    } else {
        scaled
    }
}

fn extract_supported_phase_shift_argument(
    ctx: &mut cas_ast::Context,
    trig_fn: BuiltinFn,
    expr: ExprId,
) -> Option<(ExprId, PhaseShiftKind, bool)> {
    let normalized = rewrite_linear_angle_expr(ctx, expr);
    for denominator in [4_i64, 3_i64, 6_i64] {
        let shift = build_pi_over(ctx, denominator);
        let matched = match ctx.get(normalized).clone() {
            Expr::Add(left, right) => {
                if compare_expr(ctx, left, shift) == Ordering::Equal {
                    Some((right, false))
                } else if compare_expr(ctx, right, shift) == Ordering::Equal {
                    Some((left, false))
                } else {
                    None
                }
            }
            Expr::Sub(left, right) => {
                (compare_expr(ctx, right, shift) == Ordering::Equal).then_some((left, true))
            }
            _ => None,
        };

        let Some((base_arg, subtract_shift)) = matched else {
            continue;
        };

        let kind = match (trig_fn, denominator) {
            (BuiltinFn::Sin, 4) | (BuiltinFn::Cos, 4) => PhaseShiftKind::Quarter,
            (BuiltinFn::Sin, 3) | (BuiltinFn::Cos, 6) => PhaseShiftKind::Third,
            (BuiltinFn::Sin, 6) | (BuiltinFn::Cos, 3) => PhaseShiftKind::Sixth,
            _ => continue,
        };

        return Some((base_arg, kind, subtract_shift));
    }

    None
}

fn extract_general_phase_shift_argument(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, bool)> {
    let normalized = rewrite_linear_angle_expr(ctx, expr);
    match ctx.get(normalized).clone() {
        Expr::Add(left, right) => {
            if is_atan_call(ctx, left) {
                Some((right, left, false))
            } else if is_atan_call(ctx, right) {
                Some((left, right, false))
            } else {
                None
            }
        }
        Expr::Sub(left, right) => is_atan_call(ctx, right).then_some((left, right, true)),
        _ => None,
    }
}

fn is_atan_call(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return false;
    };
    args.len() == 1
        && matches!(
            ctx.builtin_of(*fn_id),
            Some(BuiltinFn::Atan | BuiltinFn::Arctan)
        )
}

fn is_sqrt_two(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) =>
        {
            is_small_integer(ctx, args[0], 2)
        }
        _ => false,
    }
}

fn build_pi_over(ctx: &mut cas_ast::Context, denominator: i64) -> ExprId {
    let pi = ctx.add(Expr::Constant(Constant::Pi));
    let denom = ctx.num(denominator);
    ctx.add(Expr::Div(pi, denom))
}

fn split_out_small_integer_factor(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    value: i64,
) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    if let Some(index) = factors
        .iter()
        .position(|factor| is_small_integer(ctx, *factor, value))
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
        if !n.is_integer() {
            continue;
        }
        let integer = n.to_integer();
        let Ok(integer_i64): Result<i64, _> = integer.try_into() else {
            continue;
        };
        if integer_i64 % value != 0 {
            continue;
        }
        let quotient = integer_i64 / value;
        let mut remaining = Vec::new();
        for (i, other) in factors.iter().copied().enumerate() {
            if i == index {
                continue;
            }
            remaining.push(other);
        }
        if quotient != 1 {
            remaining.push(ctx.num(quotient));
        }
        return Some(
            remaining
                .into_iter()
                .fold(ctx.num(1), |acc, factor| smart_mul(ctx, acc, factor)),
        );
    }

    None
}

fn try_rewrite_normalized_reciprocal_trig_contraction_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let normalized_expr = rewrite_trig_linear_angle_arguments(ctx, expr);
    if normalized_expr == expr {
        return None;
    }

    let rewrite = cas_math::trig_canonicalization_support::try_rewrite_trig_quotient_div_expr(
        ctx,
        normalized_expr,
    )?;

    if !strong_target_match(ctx, rewrite.rewritten, target_expr) {
        return None;
    }

    let kind = match rewrite.kind {
        Some(
            cas_math::trig_canonicalization_support::TrigCanonicalRewriteKind::SinOverCosToTan,
        ) => DeriveTrigRewriteKind::RecognizeSinOverCosAsTan,
        Some(
            cas_math::trig_canonicalization_support::TrigCanonicalRewriteKind::OneOverCosToSec,
        ) => DeriveTrigRewriteKind::RecognizeRecipCosAsSec,
        Some(
            cas_math::trig_canonicalization_support::TrigCanonicalRewriteKind::OneOverSinToCsc,
        ) => DeriveTrigRewriteKind::RecognizeRecipSinAsCsc,
        Some(
            cas_math::trig_canonicalization_support::TrigCanonicalRewriteKind::CosOverSinToCot,
        ) => DeriveTrigRewriteKind::RecognizeCosOverSinAsCot,
        _ => return None,
    };

    Some(DeriveTrigRewrite {
        rewritten: rewrite.rewritten,
        kind,
    })
}

fn try_rewrite_mixed_sine_double_angle_product_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let (mixed_kind, arg) = extract_scaled_mixed_sine_double_angle_term(ctx, expr, 4)?;
    let two = ctx.num(2);
    let double_arg = smart_mul(ctx, two, arg);
    let sin_double = ctx.call_builtin(BuiltinFn::Sin, vec![double_arg]);
    let linear = match mixed_kind {
        BuiltinFn::Sin => ctx.call_builtin(BuiltinFn::Sin, vec![arg]),
        BuiltinFn::Cos => ctx.call_builtin(BuiltinFn::Cos, vec![arg]),
        _ => return None,
    };
    let inner_product = smart_mul(ctx, sin_double, linear);
    let candidate = smart_mul(ctx, two, inner_product);

    if !strong_target_match(ctx, candidate, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten: target_expr,
        kind: DeriveTrigRewriteKind::DoubleAngleSin,
    })
}

fn try_rewrite_half_scaled_sine_double_angle_contraction_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let TrigTwoFactorPattern::SinCos(left_arg, right_arg) =
        extract_trig_two_factor_product(ctx, expr)?
    else {
        return None;
    };
    if compare_expr(ctx, left_arg, right_arg) != Ordering::Equal {
        return None;
    }

    let two = ctx.num(2);
    let double_arg = smart_mul(ctx, two, left_arg);
    let sin_double = ctx.call_builtin(BuiltinFn::Sin, vec![double_arg]);
    let denominator = ctx.num(2);
    let candidate = ctx.add(Expr::Div(sin_double, denominator));

    if !strong_target_match(ctx, candidate, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten: target_expr,
        kind: DeriveTrigRewriteKind::DoubleAngleSin,
    })
}

fn try_rewrite_triple_angle_contraction_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let rewrite = try_rewrite_triple_angle_contraction_expr_with_trivial_angles(ctx, expr)?;
    if !strong_target_match(ctx, rewrite.rewritten, target_expr) {
        return None;
    }

    Some(rewrite)
}

fn try_rewrite_negated_triple_angle_contraction_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let positive_expr = strip_top_level_or_div_numerator_negation(ctx, expr)?;
    let rewrite =
        try_rewrite_triple_angle_contraction_expr_with_trivial_angles(ctx, positive_expr)?;
    let rewritten = ctx.add(Expr::Neg(rewrite.rewritten));
    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten,
        kind: rewrite.kind,
    })
}

fn try_rewrite_quintuple_angle_contraction_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let rewrite = try_rewrite_quintuple_angle_contraction_expr(ctx, expr)?;
    if !strong_target_match(ctx, rewrite.rewritten, target_expr) {
        return None;
    }

    Some(rewrite)
}

fn try_rewrite_negated_quintuple_angle_contraction_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let positive_expr = strip_unit_negation(ctx, expr)?;
    let rewrite = try_rewrite_quintuple_angle_contraction_expr(ctx, positive_expr)?;
    let rewritten = ctx.add(Expr::Neg(rewrite.rewritten));
    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten,
        kind: rewrite.kind,
    })
}

fn try_rewrite_angle_sum_diff_contraction_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let rewritten = rewrite_angle_sum_diff_contraction_expr(ctx, expr, target_expr, false)?;
    Some(DeriveTrigRewrite {
        rewritten,
        kind: DeriveTrigRewriteKind::AngleSumDiff,
    })
}

fn try_rewrite_negated_angle_sum_diff_contraction_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let positive_expr = strip_unit_negation(ctx, expr)?;
    let rewritten = rewrite_angle_sum_diff_contraction_expr(ctx, positive_expr, target_expr, true)?;

    Some(DeriveTrigRewrite {
        rewritten,
        kind: DeriveTrigRewriteKind::AngleSumDiff,
    })
}

fn rewrite_angle_sum_diff_bridge_expr(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    negate_output: bool,
) -> Option<DeriveTrigRewrite> {
    let finalize = |ctx: &mut cas_ast::Context, rewritten: ExprId| {
        let rewritten = if negate_output {
            ctx.add(Expr::Neg(rewritten))
        } else {
            rewritten
        };

        Some(DeriveTrigRewrite {
            rewritten,
            kind: DeriveTrigRewriteKind::AngleSumDiff,
        })
    };

    let expr_kind = match ctx.get(expr) {
        Expr::Add(left, right) => Some((true, *left, *right)),
        Expr::Sub(left, right) => Some((false, *left, *right)),
        _ => None,
    };

    match expr_kind {
        Some((true, left, right)) => {
            let left_pattern = extract_trig_two_factor_product(ctx, left)?;
            let right_pattern = extract_trig_two_factor_product(ctx, right)?;

            if let (TrigTwoFactorPattern::SinCos(a, b), TrigTwoFactorPattern::SinCos(c, d)) =
                (left_pattern, right_pattern)
            {
                if compare_expr(ctx, a, d) == Ordering::Equal
                    && compare_expr(ctx, b, c) == Ordering::Equal
                {
                    let sum_ab = combine_linear_angle_terms(ctx, a, b, false);
                    let candidate = ctx.call_builtin(BuiltinFn::Sin, vec![sum_ab]);
                    return finalize(ctx, candidate);
                }
            }

            let cos_sin_pair = match (left_pattern, right_pattern) {
                (TrigTwoFactorPattern::CosCos(a, b), TrigTwoFactorPattern::SinSin(c, d))
                | (TrigTwoFactorPattern::SinSin(c, d), TrigTwoFactorPattern::CosCos(a, b)) => {
                    Some((a, b, c, d))
                }
                _ => None,
            }?;

            let (a, b, c, d) = cos_sin_pair;
            if same_unordered_pair(ctx, a, b, c, d) {
                let diff_ab = combine_linear_angle_terms(ctx, a, b, true);
                let candidate = ctx.call_builtin(BuiltinFn::Cos, vec![diff_ab]);
                return finalize(ctx, candidate);
            }
        }
        Some((false, left, right)) => {
            let left_pattern = extract_trig_two_factor_product(ctx, left)?;
            let right_pattern = extract_trig_two_factor_product(ctx, right)?;

            if let (TrigTwoFactorPattern::SinCos(a, b), TrigTwoFactorPattern::SinCos(c, d)) =
                (left_pattern, right_pattern)
            {
                if compare_expr(ctx, a, d) == Ordering::Equal
                    && compare_expr(ctx, b, c) == Ordering::Equal
                {
                    let diff_ab = combine_linear_angle_terms(ctx, a, b, true);
                    let candidate = ctx.call_builtin(BuiltinFn::Sin, vec![diff_ab]);
                    return finalize(ctx, candidate);
                }
            }

            if let (TrigTwoFactorPattern::CosCos(a, b), TrigTwoFactorPattern::SinSin(c, d)) =
                (left_pattern, right_pattern)
            {
                if same_unordered_pair(ctx, a, b, c, d) {
                    let sum_ab = combine_linear_angle_terms(ctx, a, b, false);
                    let candidate = ctx.call_builtin(BuiltinFn::Cos, vec![sum_ab]);
                    return finalize(ctx, candidate);
                }
            }
        }
        None => {}
    }

    None
}

fn rewrite_angle_sum_diff_contraction_expr(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
    negate_output: bool,
) -> Option<ExprId> {
    let try_candidate = |ctx: &mut cas_ast::Context, candidate: ExprId, target_expr: ExprId| {
        let candidate = if negate_output {
            ctx.add(Expr::Neg(candidate))
        } else {
            candidate
        };
        if strong_target_match(ctx, candidate, target_expr) {
            Some(candidate)
        } else {
            None
        }
    };

    let expr_kind = match ctx.get(expr) {
        Expr::Add(left, right) => Some((true, *left, *right)),
        Expr::Sub(left, right) => Some((false, *left, *right)),
        _ => None,
    };

    match expr_kind {
        Some((true, left, right)) => {
            let left_pattern = extract_trig_two_factor_product(ctx, left)?;
            let right_pattern = extract_trig_two_factor_product(ctx, right)?;

            if let (TrigTwoFactorPattern::SinCos(a, b), TrigTwoFactorPattern::SinCos(c, d)) =
                (left_pattern, right_pattern)
            {
                if compare_expr(ctx, a, d) == Ordering::Equal
                    && compare_expr(ctx, b, c) == Ordering::Equal
                {
                    let sum_ab = combine_linear_angle_terms(ctx, a, b, false);
                    let candidate = ctx.call_builtin(BuiltinFn::Sin, vec![sum_ab]);
                    if let Some(candidate) = try_candidate(ctx, candidate, target_expr) {
                        return Some(candidate);
                    }
                }
            }

            let cos_sin_pair = match (left_pattern, right_pattern) {
                (TrigTwoFactorPattern::CosCos(a, b), TrigTwoFactorPattern::SinSin(c, d))
                | (TrigTwoFactorPattern::SinSin(c, d), TrigTwoFactorPattern::CosCos(a, b)) => {
                    Some((a, b, c, d))
                }
                _ => None,
            };
            if let Some((a, b, c, d)) = cos_sin_pair {
                if same_unordered_pair(ctx, a, b, c, d) {
                    let diff_ab = combine_linear_angle_terms(ctx, a, b, true);
                    let candidate = ctx.call_builtin(BuiltinFn::Cos, vec![diff_ab]);
                    if let Some(candidate) = try_candidate(ctx, candidate, target_expr) {
                        return Some(candidate);
                    }

                    let diff_ba = combine_linear_angle_terms(ctx, b, a, true);
                    let candidate = ctx.call_builtin(BuiltinFn::Cos, vec![diff_ba]);
                    if let Some(candidate) = try_candidate(ctx, candidate, target_expr) {
                        return Some(candidate);
                    }
                }
            }
        }
        Some((false, left, right)) => {
            let left_pattern = extract_trig_two_factor_product(ctx, left)?;
            let right_pattern = extract_trig_two_factor_product(ctx, right)?;

            if let (TrigTwoFactorPattern::SinCos(a, b), TrigTwoFactorPattern::SinCos(c, d)) =
                (left_pattern, right_pattern)
            {
                if compare_expr(ctx, a, d) == Ordering::Equal
                    && compare_expr(ctx, b, c) == Ordering::Equal
                {
                    let diff_ab = combine_linear_angle_terms(ctx, a, b, true);
                    let candidate = ctx.call_builtin(BuiltinFn::Sin, vec![diff_ab]);
                    if let Some(candidate) = try_candidate(ctx, candidate, target_expr) {
                        return Some(candidate);
                    }

                    let diff_ba = combine_linear_angle_terms(ctx, b, a, true);
                    let candidate = ctx.call_builtin(BuiltinFn::Sin, vec![diff_ba]);
                    if let Some(candidate) = try_candidate(ctx, candidate, target_expr) {
                        return Some(candidate);
                    }
                }
            }

            if let (TrigTwoFactorPattern::CosCos(a, b), TrigTwoFactorPattern::SinSin(c, d)) =
                (left_pattern, right_pattern)
            {
                if same_unordered_pair(ctx, a, b, c, d) {
                    let sum_ab = combine_linear_angle_terms(ctx, a, b, false);
                    let candidate = ctx.call_builtin(BuiltinFn::Cos, vec![sum_ab]);
                    if let Some(candidate) = try_candidate(ctx, candidate, target_expr) {
                        return Some(candidate);
                    }
                }
            }
        }
        None => {}
    }

    None
}

fn try_rewrite_tangent_angle_sum_diff_contraction_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let (numerator, denominator) = (*numerator, *denominator);
    let (lhs, rhs, is_sum) = tangent_sum_diff_numerator_args(ctx, numerator)?;

    let denominator_matches = if is_sum {
        matches_one_minus_tangent_product(ctx, denominator, lhs, rhs)
    } else {
        matches_one_plus_tangent_product(ctx, denominator, lhs, rhs)
    };
    if !denominator_matches {
        return None;
    }

    let argument = if is_sum {
        ctx.add(Expr::Add(lhs, rhs))
    } else {
        ctx.add(Expr::Sub(lhs, rhs))
    };
    let rewritten = ctx.call_builtin(BuiltinFn::Tan, vec![argument]);

    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten,
        kind: DeriveTrigRewriteKind::TangentAngleSumDiffContract,
    })
}

fn tangent_sum_diff_numerator_args(
    ctx: &cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, bool)> {
    match ctx.get(expr) {
        Expr::Add(lhs, rhs) => Some((tan_call_arg(ctx, *lhs)?, tan_call_arg(ctx, *rhs)?, true)),
        Expr::Sub(lhs, rhs) => Some((tan_call_arg(ctx, *lhs)?, tan_call_arg(ctx, *rhs)?, false)),
        _ => None,
    }
}

fn matches_one_minus_tangent_product(
    ctx: &cas_ast::Context,
    expr: ExprId,
    lhs: ExprId,
    rhs: ExprId,
) -> bool {
    let Expr::Sub(one, product) = ctx.get(expr) else {
        return false;
    };
    is_small_integer(ctx, *one, 1) && matches_tangent_product(ctx, *product, lhs, rhs)
}

fn matches_one_plus_tangent_product(
    ctx: &cas_ast::Context,
    expr: ExprId,
    lhs: ExprId,
    rhs: ExprId,
) -> bool {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return false;
    };

    if is_small_integer(ctx, *left, 1) {
        return matches_tangent_product(ctx, *right, lhs, rhs);
    }
    if is_small_integer(ctx, *right, 1) {
        return matches_tangent_product(ctx, *left, lhs, rhs);
    }
    false
}

fn matches_tangent_product(ctx: &cas_ast::Context, expr: ExprId, lhs: ExprId, rhs: ExprId) -> bool {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return false;
    };
    let Some(left_arg) = tan_call_arg(ctx, *left) else {
        return false;
    };
    let Some(right_arg) = tan_call_arg(ctx, *right) else {
        return false;
    };

    same_unordered_pair(ctx, left_arg, right_arg, lhs, rhs)
}

fn try_rewrite_quintuple_angle_contraction_expr(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let signed_terms = add_terms_signed(ctx, expr);
    if signed_terms.len() < 3 {
        return None;
    }

    struct TrigTerm {
        index: usize,
        coeff: BigRational,
        builtin: BuiltinFn,
        arg: ExprId,
        power: i64,
    }

    fn decompose_trig_term(
        ctx: &cas_ast::Context,
        term: ExprId,
        sign: Sign,
    ) -> Option<(BigRational, BuiltinFn, ExprId, i64)> {
        let outer_sign = BigRational::from_integer(sign.to_i32().into());

        let (inner, neg_sign) = if let Expr::Neg(i) = ctx.get(term) {
            (*i, BigRational::from_integer((-1).into()))
        } else {
            (term, BigRational::from_integer(1.into()))
        };

        let (coeff, core) = match ctx.get(inner) {
            Expr::Mul(l, r) => {
                if let Expr::Number(n) = ctx.get(*l) {
                    (n.clone(), *r)
                } else if let Expr::Number(n) = ctx.get(*r) {
                    (n.clone(), *l)
                } else {
                    (BigRational::from_integer(1.into()), inner)
                }
            }
            _ => (BigRational::from_integer(1.into()), inner),
        };
        let final_coeff = outer_sign * neg_sign * coeff;

        let (base, power) = if let Expr::Pow(b, e) = ctx.get(core) {
            if let Expr::Number(n) = ctx.get(*e) {
                if n.is_integer() {
                    (*b, n.to_integer().try_into().ok().unwrap_or(0i64))
                } else {
                    return None;
                }
            } else {
                return None;
            }
        } else {
            (core, 1i64)
        };

        if let Expr::Function(fn_id, args) = ctx.get(base) {
            if args.len() == 1 {
                if let Some(b @ (BuiltinFn::Sin | BuiltinFn::Cos)) = ctx.builtin_of(*fn_id) {
                    return Some((final_coeff, b, args[0], power));
                }
            }
        }
        None
    }

    let mut trig_terms = Vec::new();
    for (idx, (term, sign)) in signed_terms.iter().enumerate() {
        if let Some((coeff, builtin, arg, power)) = decompose_trig_term(ctx, *term, *sign) {
            if matches!(power, 1 | 3 | 5) {
                trig_terms.push(TrigTerm {
                    index: idx,
                    coeff,
                    builtin,
                    arg,
                    power,
                });
            }
        }
    }

    for linear in &trig_terms {
        if linear.power != 1 {
            continue;
        }

        let Some(cubic) = trig_terms.iter().find(|term| {
            term.power == 3
                && term.builtin == linear.builtin
                && compare_expr(ctx, term.arg, linear.arg) == Ordering::Equal
        }) else {
            continue;
        };
        let Some(fifth) = trig_terms.iter().find(|term| {
            term.power == 5
                && term.builtin == linear.builtin
                && compare_expr(ctx, term.arg, linear.arg) == Ordering::Equal
        }) else {
            continue;
        };

        let five = BigRational::from_integer(5.into());
        let sixteen = BigRational::from_integer(16.into());
        let twenty = BigRational::from_integer(20.into());
        let scale = &linear.coeff / &five;
        if cubic.coeff != -(scale.clone() * &twenty) || fifth.coeff != scale.clone() * &sixteen {
            continue;
        }

        let five_id = ctx.num(5);
        let quintuple_arg = smart_mul(ctx, five_id, linear.arg);
        let contracted = ctx.call_builtin(linear.builtin, vec![quintuple_arg]);
        let one = BigRational::from_integer(1.into());
        let neg_one = -&one;
        let scaled = if scale == one {
            contracted
        } else if scale == neg_one {
            ctx.add(Expr::Neg(contracted))
        } else {
            let scale_id = ctx.add(Expr::Number(scale.clone()));
            smart_mul(ctx, scale_id, contracted)
        };

        let kind = match linear.builtin {
            BuiltinFn::Sin => DeriveTrigRewriteKind::QuintupleAngleSin,
            BuiltinFn::Cos => DeriveTrigRewriteKind::QuintupleAngleCos,
            _ => continue,
        };

        if signed_terms.len() == 3 {
            return Some(DeriveTrigRewrite {
                rewritten: scaled,
                kind,
            });
        }

        let mut new_terms = Vec::new();
        for (idx, (term, sign)) in signed_terms.iter().enumerate() {
            if idx != linear.index && idx != cubic.index && idx != fifth.index {
                if *sign == Sign::Neg {
                    new_terms.push(ctx.add(Expr::Neg(*term)));
                } else {
                    new_terms.push(*term);
                }
            }
        }
        new_terms.push(scaled);

        let mut rewritten = new_terms[0];
        for &term in new_terms.iter().skip(1) {
            rewritten = ctx.add(Expr::Add(rewritten, term));
        }

        return Some(DeriveTrigRewrite { rewritten, kind });
    }

    None
}

fn try_rewrite_triple_angle_contraction_expr_with_trivial_angles(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    if let Some(rewrite) =
        cas_math::trig_multi_angle_support::try_rewrite_triple_angle_contraction_expr(ctx, expr)
    {
        let kind = match rewrite.kind {
            cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::TripleContractionSin => {
                DeriveTrigRewriteKind::TripleAngleSin
            }
            cas_math::trig_multi_angle_support::TrigMultiAngleRewriteKind::TripleContractionCos => {
                DeriveTrigRewriteKind::TripleAngleCos
            }
            _ => return None,
        };

        return Some(DeriveTrigRewrite {
            rewritten: rewrite.rewritten,
            kind,
        });
    }

    if let Some(rewrite) = try_rewrite_tangent_triple_angle_contraction_expr(ctx, expr) {
        return Some(rewrite);
    }

    try_rewrite_trivial_triple_angle_contraction_expr(ctx, expr)
}

fn try_rewrite_tangent_triple_angle_contraction_expr(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };

    let arg = extract_tan_triple_angle_numerator_arg(ctx, *numerator)?;
    if !matches_tan_triple_angle_denominator(ctx, *denominator, arg) {
        return None;
    }

    let three = ctx.num(3);
    let triple_arg = smart_mul(ctx, three, arg);
    let rewritten = ctx.call_builtin(BuiltinFn::Tan, vec![triple_arg]);

    Some(DeriveTrigRewrite {
        rewritten,
        kind: DeriveTrigRewriteKind::TripleAngleTan,
    })
}

fn try_rewrite_trivial_triple_angle_contraction_expr(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let signed_terms = add_terms_signed(ctx, expr);
    if signed_terms.len() < 2 {
        return None;
    }

    struct TrigTerm {
        index: usize,
        coeff: BigRational,
        builtin: BuiltinFn,
        arg: ExprId,
        power: i64,
    }

    fn decompose_trig_term(
        ctx: &cas_ast::Context,
        term: ExprId,
        sign: Sign,
    ) -> Option<(BigRational, BuiltinFn, ExprId, i64)> {
        let outer_sign = BigRational::from_integer(sign.to_i32().into());

        let (inner, neg_sign) = if let Expr::Neg(i) = ctx.get(term) {
            (*i, BigRational::from_integer((-1).into()))
        } else {
            (term, BigRational::from_integer(1.into()))
        };

        let (coeff, core) = match ctx.get(inner) {
            Expr::Mul(l, r) => {
                if let Expr::Number(n) = ctx.get(*l) {
                    (n.clone(), *r)
                } else if let Expr::Number(n) = ctx.get(*r) {
                    (n.clone(), *l)
                } else {
                    (BigRational::from_integer(1.into()), inner)
                }
            }
            _ => (BigRational::from_integer(1.into()), inner),
        };
        let final_coeff = outer_sign * neg_sign * coeff;

        let (base, power) = if let Expr::Pow(b, e) = ctx.get(core) {
            if let Expr::Number(n) = ctx.get(*e) {
                if n.is_integer() {
                    (*b, n.to_integer().try_into().ok().unwrap_or(0i64))
                } else {
                    return None;
                }
            } else {
                return None;
            }
        } else {
            (core, 1i64)
        };

        if let Expr::Function(fn_id, args) = ctx.get(base) {
            if args.len() == 1 {
                if let Some(b @ (BuiltinFn::Sin | BuiltinFn::Cos)) = ctx.builtin_of(*fn_id) {
                    return Some((final_coeff, b, args[0], power));
                }
            }
        }
        None
    }

    let mut trig_terms = Vec::new();
    for (idx, (term, sign)) in signed_terms.iter().enumerate() {
        if let Some((coeff, builtin, arg, power)) = decompose_trig_term(ctx, *term, *sign) {
            if power == 1 || power == 3 {
                trig_terms.push(TrigTerm {
                    index: idx,
                    coeff,
                    builtin,
                    arg,
                    power,
                });
            }
        }
    }

    for i in 0..trig_terms.len() {
        for j in 0..trig_terms.len() {
            if i == j {
                continue;
            }

            let linear = &trig_terms[i];
            let cubic = &trig_terms[j];
            if linear.builtin != cubic.builtin || linear.power != 1 || cubic.power != 3 {
                continue;
            }
            if compare_expr(ctx, linear.arg, cubic.arg) != Ordering::Equal {
                continue;
            }

            let three = BigRational::from_integer(3.into());
            let four = BigRational::from_integer(4.into());
            let matches_pattern = match linear.builtin {
                BuiltinFn::Sin => (&linear.coeff * &four) == (-(&cubic.coeff) * &three),
                BuiltinFn::Cos => (&cubic.coeff * &three) == (-(&linear.coeff) * &four),
                _ => false,
            };
            if !matches_pattern {
                continue;
            }

            let scale = match linear.builtin {
                BuiltinFn::Sin => &linear.coeff / &three,
                BuiltinFn::Cos => &cubic.coeff / &four,
                _ => continue,
            };

            let three_id = ctx.num(3);
            let triple_arg = smart_mul(ctx, three_id, linear.arg);
            let contracted = ctx.call_builtin(linear.builtin, vec![triple_arg]);
            let one = BigRational::from_integer(1.into());
            let neg_one = -&one;
            let scaled = if scale == one {
                contracted
            } else if scale == neg_one {
                ctx.add(Expr::Neg(contracted))
            } else {
                let scale_id = ctx.add(Expr::Number(scale));
                smart_mul(ctx, scale_id, contracted)
            };

            let kind = match linear.builtin {
                BuiltinFn::Sin => DeriveTrigRewriteKind::TripleAngleSin,
                BuiltinFn::Cos => DeriveTrigRewriteKind::TripleAngleCos,
                _ => continue,
            };

            if signed_terms.len() == 2 {
                return Some(DeriveTrigRewrite {
                    rewritten: scaled,
                    kind,
                });
            }

            let mut new_terms = Vec::new();
            for (idx, (term, sign)) in signed_terms.iter().enumerate() {
                if idx != linear.index && idx != cubic.index {
                    if *sign == Sign::Neg {
                        new_terms.push(ctx.add(Expr::Neg(*term)));
                    } else {
                        new_terms.push(*term);
                    }
                }
            }
            new_terms.push(scaled);

            let mut rewritten = new_terms[0];
            for &term in new_terms.iter().skip(1) {
                rewritten = ctx.add(Expr::Add(rewritten, term));
            }

            return Some(DeriveTrigRewrite { rewritten, kind });
        }
    }

    None
}

fn try_rewrite_negated_half_angle_tangent_contraction_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let positive_expr = strip_top_level_or_div_numerator_negation(ctx, expr)?;
    let rewrite = cas_math::trig_contraction_support::try_rewrite_half_angle_tangent_div_expr(
        ctx,
        positive_expr,
    )?;
    let rewritten = ctx.add(Expr::Neg(rewrite.rewritten));

    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten,
        kind: DeriveTrigRewriteKind::HalfAngleTangent,
    })
}

fn try_rewrite_trig_identity_term_to_one_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let one = ctx.num(1);
    if !strong_target_match(ctx, one, target_expr) {
        return None;
    }

    if let Some(rewrite) =
        cas_math::trig_canonicalization_support::try_rewrite_reciprocal_product_identity_expr(
            ctx, expr,
        )
    {
        return Some(DeriveTrigRewrite {
            rewritten: rewrite.rewritten,
            kind: DeriveTrigRewriteKind::ReciprocalProductToOne,
        });
    }

    if let Some(rewrite) =
        cas_math::trig_power_identity_support::try_rewrite_pythagorean_chain_add_expr(ctx, expr)
    {
        return Some(DeriveTrigRewrite {
            rewritten: rewrite.rewritten,
            kind: DeriveTrigRewriteKind::PythagoreanChainToOne,
        });
    }

    if let Some(rewrite) =
        cas_math::trig_canonicalization_support::try_rewrite_sec_tan_pythagorean_identity_expr(
            ctx, expr,
        )
    {
        return Some(DeriveTrigRewrite {
            rewritten: rewrite.rewritten,
            kind: DeriveTrigRewriteKind::SecTanPythagoreanToOne,
        });
    }

    if let Some(rewrite) =
        cas_math::trig_canonicalization_support::try_rewrite_csc_cot_pythagorean_identity_expr(
            ctx, expr,
        )
    {
        return Some(DeriveTrigRewrite {
            rewritten: rewrite.rewritten,
            kind: DeriveTrigRewriteKind::CscCotPythagoreanToOne,
        });
    }

    None
}

pub(crate) fn try_rewrite_trig_identity_to_one_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    if let Some(rewrite) =
        try_rewrite_trig_identity_term_to_one_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    let source_terms = add_terms_signed(ctx, expr);
    let target_terms = add_terms_signed(ctx, target_expr);
    if source_terms.len() <= 1 || target_terms.len() <= 1 {
        return None;
    }

    let one = ctx.num(1);
    for (source_index, (source_term, source_sign)) in source_terms.iter().copied().enumerate() {
        let Some(rewrite) =
            try_rewrite_trig_identity_term_to_one_target_aware(ctx, source_term, one)
        else {
            continue;
        };

        for (target_index, (target_term, target_sign)) in target_terms.iter().copied().enumerate() {
            if source_sign != target_sign || !strong_target_match(ctx, one, target_term) {
                continue;
            }

            if signed_additive_passthrough_terms_match(
                ctx,
                &source_terms,
                source_index,
                &target_terms,
                target_index,
            ) {
                return Some(DeriveTrigRewrite {
                    rewritten: target_expr,
                    kind: rewrite.kind,
                });
            }
        }
    }

    if let Some(source_limit) = 1usize.checked_shl(source_terms.len() as u32) {
        for mask in 1..source_limit {
            if mask.count_ones() < 2 {
                continue;
            }

            let source_focus = build_additive_expr_from_terms_and_signs(
                ctx,
                source_terms
                    .iter()
                    .enumerate()
                    .filter_map(|(index, term)| ((mask & (1usize << index)) != 0).then_some(*term)),
            );
            let Some(rewrite) =
                try_rewrite_trig_identity_term_to_one_target_aware(ctx, source_focus, one)
            else {
                continue;
            };

            let source_passthrough = collect_signed_passthrough_terms(&source_terms, mask);
            for (target_index, (target_term, target_sign)) in
                target_terms.iter().copied().enumerate()
            {
                if target_sign != Sign::Pos || !strong_target_match(ctx, one, target_term) {
                    continue;
                }

                let target_passthrough = target_terms
                    .iter()
                    .enumerate()
                    .filter_map(|(index, term)| (index != target_index).then_some(*term))
                    .collect::<Vec<_>>();
                if signed_additive_terms_match(ctx, &source_passthrough, &target_passthrough) {
                    return Some(DeriveTrigRewrite {
                        rewritten: target_expr,
                        kind: rewrite.kind,
                    });
                }
            }
        }
    }

    None
}

pub(crate) fn try_rewrite_shifted_double_angle_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let one = ctx.num(1);
    let two = ctx.num(2);

    if let Expr::Add(left, right) = ctx.get(expr) {
        let (cos_term, one_term) = if is_small_integer(ctx, *left, 1) {
            (*right, *left)
        } else if is_small_integer(ctx, *right, 1) {
            (*left, *right)
        } else {
            (expr, expr)
        };

        if cos_term != expr && is_small_integer(ctx, one_term, 1) {
            if let Some(arg) = double_angle_cos_inner(ctx, cos_term) {
                let cos_fn = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
                let cos_sq = pow2(ctx, cos_fn);
                let candidate = ctx.add(Expr::Mul(two, cos_sq));
                if strong_target_match(ctx, candidate, target_expr) {
                    return Some(DeriveTrigRewrite {
                        rewritten: candidate,
                        kind: DeriveTrigRewriteKind::DoubleAngleCosOnePlusCosToTwoCosSq,
                    });
                }
            }
        }
    }

    if let Expr::Sub(left, right) = ctx.get(expr) {
        let (left, right) = (*left, *right);

        if is_small_integer(ctx, left, 1) {
            if let Some(arg) = double_angle_cos_inner(ctx, right) {
                let sin_fn = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
                let sin_sq = pow2(ctx, sin_fn);
                let candidate = ctx.add(Expr::Mul(two, sin_sq));
                if strong_target_match(ctx, candidate, target_expr) {
                    return Some(DeriveTrigRewrite {
                        rewritten: candidate,
                        kind: DeriveTrigRewriteKind::DoubleAngleCosOneMinusCosToTwoSinSq,
                    });
                }
            }
        }

        if is_small_integer(ctx, right, 1) {
            if let Some(arg) = double_angle_cos_inner(ctx, left) {
                let sin_fn = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
                let sin_sq = pow2(ctx, sin_fn);
                let neg_two = ctx.num(-2);
                let candidate = smart_mul(ctx, neg_two, sin_sq);
                if strong_target_match(ctx, candidate, target_expr) {
                    return Some(DeriveTrigRewrite {
                        rewritten: candidate,
                        kind: DeriveTrigRewriteKind::DoubleAngleCosMinusOneToNegTwoSinSq,
                    });
                }
            }
        }
    }

    if let Some((BuiltinFn::Sin, arg)) = scaled_sin_or_cos_square(ctx, expr, 2) {
        let double_arg_raw = smart_mul(ctx, two, arg);
        let double_arg = run_phase_shift_match_simplify(ctx, double_arg_raw);
        let cos_double = ctx.call_builtin(BuiltinFn::Cos, vec![double_arg]);
        let candidate = ctx.add(Expr::Sub(one, cos_double));
        if strong_target_match(ctx, candidate, target_expr) {
            return Some(DeriveTrigRewrite {
                rewritten: candidate,
                kind: DeriveTrigRewriteKind::DoubleAngleCosTwoSinSqToOneMinusCos,
            });
        }
    }

    if let Some((BuiltinFn::Cos, arg)) = scaled_sin_or_cos_square(ctx, expr, 2) {
        let double_arg_raw = smart_mul(ctx, two, arg);
        let double_arg = run_phase_shift_match_simplify(ctx, double_arg_raw);
        let cos_double = ctx.call_builtin(BuiltinFn::Cos, vec![double_arg]);
        let candidate = ctx.add(Expr::Add(one, cos_double));
        if strong_target_match(ctx, candidate, target_expr) {
            return Some(DeriveTrigRewrite {
                rewritten: candidate,
                kind: DeriveTrigRewriteKind::DoubleAngleCosTwoCosSqToOnePlusCos,
            });
        }
    }

    if let Some((BuiltinFn::Sin, arg)) = scaled_sin_or_cos_square(ctx, expr, -2) {
        let double_arg_raw = smart_mul(ctx, two, arg);
        let double_arg = run_phase_shift_match_simplify(ctx, double_arg_raw);
        let cos_double = ctx.call_builtin(BuiltinFn::Cos, vec![double_arg]);
        let candidate = ctx.add(Expr::Sub(cos_double, one));
        if strong_target_match(ctx, candidate, target_expr) {
            return Some(DeriveTrigRewrite {
                rewritten: candidate,
                kind: DeriveTrigRewriteKind::DoubleAngleCosNegTwoSinSqToCosMinusOne,
            });
        }
    }

    None
}

pub(crate) fn try_rewrite_shifted_reciprocal_pythagorean_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    if let Expr::Sub(left, right) = ctx.get(expr) {
        let (left, right) = (*left, *right);
        if is_small_integer(ctx, left, 1) {
            if let Some((BuiltinFn::Sec, arg)) = trig_square(ctx, right) {
                let tan = ctx.call_builtin(BuiltinFn::Tan, vec![arg]);
                let tan_sq = pow2(ctx, tan);
                let neg_tan_sq = ctx.add(Expr::Neg(tan_sq));
                if strong_target_match(ctx, neg_tan_sq, target_expr) {
                    return Some(DeriveTrigRewrite {
                        rewritten: neg_tan_sq,
                        kind: DeriveTrigRewriteKind::RecognizeOneMinusSecSquaredAsNegTanSquared,
                    });
                }
            }

            if let Some((BuiltinFn::Csc, arg)) = trig_square(ctx, right) {
                let cot = ctx.call_builtin(BuiltinFn::Cot, vec![arg]);
                let cot_sq = pow2(ctx, cot);
                let neg_cot_sq = ctx.add(Expr::Neg(cot_sq));
                if strong_target_match(ctx, neg_cot_sq, target_expr) {
                    return Some(DeriveTrigRewrite {
                        rewritten: neg_cot_sq,
                        kind: DeriveTrigRewriteKind::RecognizeOneMinusCscSquaredAsNegCotSquared,
                    });
                }
            }
        }

        if is_small_integer(ctx, right, 1) {
            if let Some((BuiltinFn::Sec, arg)) = trig_square(ctx, left) {
                let tan = ctx.call_builtin(BuiltinFn::Tan, vec![arg]);
                let tan_sq = pow2(ctx, tan);
                if strong_target_match(ctx, tan_sq, target_expr) {
                    return Some(DeriveTrigRewrite {
                        rewritten: tan_sq,
                        kind: DeriveTrigRewriteKind::RecognizeSecSquaredMinusOneAsTanSquared,
                    });
                }
            }

            if let Some((BuiltinFn::Csc, arg)) = trig_square(ctx, left) {
                let cot = ctx.call_builtin(BuiltinFn::Cot, vec![arg]);
                let cot_sq = pow2(ctx, cot);
                if strong_target_match(ctx, cot_sq, target_expr) {
                    return Some(DeriveTrigRewrite {
                        rewritten: cot_sq,
                        kind: DeriveTrigRewriteKind::RecognizeCscSquaredMinusOneAsCotSquared,
                    });
                }
            }
        }
    }

    if let Some(positive_expr) = strip_unit_negation(ctx, expr) {
        if let Some((BuiltinFn::Tan, arg)) = tan_or_cot_square(ctx, positive_expr) {
            let sec = ctx.call_builtin(BuiltinFn::Sec, vec![arg]);
            let sec_sq = pow2(ctx, sec);
            let one = ctx.num(1);
            let candidate = ctx.add(Expr::Sub(one, sec_sq));
            if strong_target_match(ctx, candidate, target_expr) {
                return Some(DeriveTrigRewrite {
                    rewritten: candidate,
                    kind: DeriveTrigRewriteKind::ExpandNegTanSquaredToOneMinusSecSquared,
                });
            }
        }

        if let Some((BuiltinFn::Cot, arg)) = tan_or_cot_square(ctx, positive_expr) {
            let csc = ctx.call_builtin(BuiltinFn::Csc, vec![arg]);
            let csc_sq = pow2(ctx, csc);
            let one = ctx.num(1);
            let candidate = ctx.add(Expr::Sub(one, csc_sq));
            if strong_target_match(ctx, candidate, target_expr) {
                return Some(DeriveTrigRewrite {
                    rewritten: candidate,
                    kind: DeriveTrigRewriteKind::ExpandNegCotSquaredToOneMinusCscSquared,
                });
            }
        }
    }

    if let Some((BuiltinFn::Tan, arg)) = tan_or_cot_square(ctx, expr) {
        let sec = ctx.call_builtin(BuiltinFn::Sec, vec![arg]);
        let sec_sq = pow2(ctx, sec);
        let one = ctx.num(1);
        let candidate = ctx.add(Expr::Sub(sec_sq, one));
        if strong_target_match(ctx, candidate, target_expr) {
            return Some(DeriveTrigRewrite {
                rewritten: candidate,
                kind: DeriveTrigRewriteKind::ExpandTanSquaredToSecSquaredMinusOne,
            });
        }
    }

    if let Some((BuiltinFn::Cot, arg)) = tan_or_cot_square(ctx, expr) {
        let csc = ctx.call_builtin(BuiltinFn::Csc, vec![arg]);
        let csc_sq = pow2(ctx, csc);
        let one = ctx.num(1);
        let candidate = ctx.add(Expr::Sub(csc_sq, one));
        if strong_target_match(ctx, candidate, target_expr) {
            return Some(DeriveTrigRewrite {
                rewritten: candidate,
                kind: DeriveTrigRewriteKind::ExpandCotSquaredToCscSquaredMinusOne,
            });
        }
    }

    None
}

fn double_angle_cos_inner(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Cos) || args.len() != 1 {
        return None;
    }
    extract_double_angle_inner(ctx, args[0])
}

fn scaled_sin_or_cos_square(
    ctx: &cas_ast::Context,
    expr: ExprId,
    scale: i64,
) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };

    let squared = if is_small_integer(ctx, *left, scale) {
        *right
    } else if is_small_integer(ctx, *right, scale) {
        *left
    } else {
        return None;
    };

    let (trig_fn, arg) = trig_square(ctx, squared)?;
    match trig_fn {
        BuiltinFn::Sin | BuiltinFn::Cos => Some((trig_fn, arg)),
        _ => None,
    }
}

fn extract_scaled_mixed_sine_double_angle_term(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    scale: i64,
) -> Option<(BuiltinFn, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut coeff = 1_i64;
    let mut sin_sq_arg = None;
    let mut cos_sq_arg = None;
    let mut sin_arg = None;
    let mut cos_arg = None;

    for factor in factors {
        match ctx.get(factor) {
            Expr::Number(n) if n.is_integer() => {
                let integer = n.to_integer();
                let integer = i64::try_from(integer).ok()?;
                coeff = coeff.checked_mul(integer)?;
            }
            Expr::Pow(base, exponent) if is_small_integer(ctx, *exponent, 2) => {
                match trig_fn_arg(ctx, *base)? {
                    (BuiltinFn::Sin, arg) => sin_sq_arg = Some(arg),
                    (BuiltinFn::Cos, arg) => cos_sq_arg = Some(arg),
                    _ => return None,
                }
            }
            _ => match trig_fn_arg(ctx, factor)? {
                (BuiltinFn::Sin, arg) => sin_arg = Some(arg),
                (BuiltinFn::Cos, arg) => cos_arg = Some(arg),
                _ => return None,
            },
        }
    }

    if coeff != scale {
        return None;
    }

    if let (Some(arg), Some(other_arg)) = (sin_sq_arg, cos_arg) {
        if compare_expr(ctx, arg, other_arg) == Ordering::Equal {
            return Some((BuiltinFn::Sin, arg));
        }
    }

    if let (Some(arg), Some(other_arg)) = (cos_sq_arg, sin_arg) {
        if compare_expr(ctx, arg, other_arg) == Ordering::Equal {
            return Some((BuiltinFn::Cos, arg));
        }
    }

    None
}

fn extract_scaled_sine_double_angle_linear_term(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    scale: i64,
) -> Option<(BuiltinFn, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut coeff = 1_i64;
    let mut sin_double_arg = None;
    let mut sin_arg = None;
    let mut cos_arg = None;

    for factor in factors {
        match ctx.get(factor) {
            Expr::Number(n) if n.is_integer() => {
                let integer = n.to_integer();
                let integer = i64::try_from(integer).ok()?;
                coeff = coeff.checked_mul(integer)?;
            }
            _ => match trig_fn_arg(ctx, factor)? {
                (BuiltinFn::Sin, arg) => {
                    if let Some(double_inner) = extract_double_angle_inner(ctx, arg) {
                        sin_double_arg = Some(double_inner);
                    } else {
                        sin_arg = Some(arg);
                    }
                }
                (BuiltinFn::Cos, arg) => cos_arg = Some(arg),
                _ => return None,
            },
        }
    }

    if coeff != scale {
        return None;
    }

    if let (Some(arg), Some(other_arg)) = (sin_double_arg, sin_arg) {
        if compare_expr(ctx, arg, other_arg) == Ordering::Equal {
            return Some((BuiltinFn::Sin, arg));
        }
    }

    if let (Some(arg), Some(other_arg)) = (sin_double_arg, cos_arg) {
        if compare_expr(ctx, arg, other_arg) == Ordering::Equal {
            return Some((BuiltinFn::Cos, arg));
        }
    }

    None
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrigTwoFactorPattern {
    SinCos(ExprId, ExprId),
    CosCos(ExprId, ExprId),
    SinSin(ExprId, ExprId),
}

fn extract_trig_two_factor_product(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<TrigTwoFactorPattern> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let first = trig_fn_arg(ctx, factors[0])?;
    let second = trig_fn_arg(ctx, factors[1])?;

    match (first, second) {
        ((BuiltinFn::Sin, a), (BuiltinFn::Cos, b)) | ((BuiltinFn::Cos, b), (BuiltinFn::Sin, a)) => {
            Some(TrigTwoFactorPattern::SinCos(a, b))
        }
        ((BuiltinFn::Cos, a), (BuiltinFn::Cos, b)) => Some(TrigTwoFactorPattern::CosCos(a, b)),
        ((BuiltinFn::Sin, a), (BuiltinFn::Sin, b)) => Some(TrigTwoFactorPattern::SinSin(a, b)),
        _ => None,
    }
}

fn trig_fn_arg(ctx: &cas_ast::Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Sin) => Some((BuiltinFn::Sin, args[0])),
        Some(BuiltinFn::Cos) => Some((BuiltinFn::Cos, args[0])),
        _ => None,
    }
}

fn same_unordered_pair(ctx: &cas_ast::Context, a: ExprId, b: ExprId, c: ExprId, d: ExprId) -> bool {
    (compare_expr(ctx, a, c) == Ordering::Equal && compare_expr(ctx, b, d) == Ordering::Equal)
        || (compare_expr(ctx, a, d) == Ordering::Equal
            && compare_expr(ctx, b, c) == Ordering::Equal)
}

fn combine_linear_angle_terms(
    ctx: &mut cas_ast::Context,
    left: ExprId,
    right: ExprId,
    subtract: bool,
) -> ExprId {
    let (left_coeff, left_base) = split_linear_angle_term(ctx, left);
    let (right_coeff, right_base) = split_linear_angle_term(ctx, right);
    if compare_expr(ctx, left_base, right_base) != Ordering::Equal {
        return if subtract {
            ctx.add(Expr::Sub(left, right))
        } else {
            ctx.add(Expr::Add(left, right))
        };
    }

    let coeff = if subtract {
        left_coeff - right_coeff
    } else {
        left_coeff + right_coeff
    };
    let zero = BigRational::from_integer(0.into());
    let one = BigRational::from_integer(1.into());
    let neg_one = -one.clone();

    if coeff == zero {
        return ctx.num(0);
    }
    if coeff == one {
        return left_base;
    }
    if coeff == neg_one {
        return ctx.add(Expr::Neg(left_base));
    }

    let coeff_id = ctx.add(Expr::Number(coeff));
    smart_mul(ctx, coeff_id, left_base)
}

fn split_linear_angle_term(ctx: &cas_ast::Context, expr: ExprId) -> (BigRational, ExprId) {
    match ctx.get(expr) {
        Expr::Neg(inner) => {
            let (coeff, base) = split_linear_angle_term(ctx, *inner);
            (-coeff, base)
        }
        Expr::Mul(left, right) => {
            if let Expr::Number(n) = ctx.get(*left) {
                (n.clone(), *right)
            } else if let Expr::Number(n) = ctx.get(*right) {
                (n.clone(), *left)
            } else {
                (BigRational::from_integer(1.into()), expr)
            }
        }
        _ => (BigRational::from_integer(1.into()), expr),
    }
}

fn build_linear_angle_term(ctx: &mut cas_ast::Context, coeff: BigRational, base: ExprId) -> ExprId {
    let zero = BigRational::from_integer(0.into());
    let one = BigRational::from_integer(1.into());
    let neg_one = -one.clone();

    if coeff == zero {
        return ctx.num(0);
    }
    if coeff == one {
        return base;
    }
    if coeff == neg_one {
        return ctx.add(Expr::Neg(base));
    }

    let coeff_id = ctx.add(Expr::Number(coeff));
    smart_mul(ctx, coeff_id, base)
}

fn half_sum_linear_angle_terms(
    ctx: &mut cas_ast::Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    let (left_coeff, left_base) = split_linear_angle_term(ctx, left);
    let (right_coeff, right_base) = split_linear_angle_term(ctx, right);
    if compare_expr(ctx, left_base, right_base) != Ordering::Equal {
        return None;
    }

    let two = BigRational::from_integer(2.into());
    Some(build_linear_angle_term(
        ctx,
        (left_coeff + right_coeff) / two,
        left_base,
    ))
}

fn extract_trig_difference_args(
    ctx: &cas_ast::Context,
    expr: ExprId,
    trig_fn: BuiltinFn,
) -> Option<(ExprId, ExprId)> {
    let Expr::Sub(left, right) = ctx.get(expr) else {
        return None;
    };
    let (left_fn, left_arg) = trig_fn_arg(ctx, *left)?;
    let (right_fn, right_arg) = trig_fn_arg(ctx, *right)?;
    if left_fn == trig_fn && right_fn == trig_fn {
        Some((left_arg, right_arg))
    } else {
        None
    }
}

fn strip_unit_negation(ctx: &mut cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Neg(inner) => return Some(*inner),
        Expr::Number(n) if n.is_negative() => return Some(ctx.add(Expr::Number(-n))),
        Expr::Mul(left, right) if is_small_integer(ctx, *left, -1) => return Some(*right),
        Expr::Mul(left, right) if is_small_integer(ctx, *right, -1) => return Some(*left),
        Expr::Mul(_, _) => {}
        _ => return None,
    }

    let factors = flatten_mul_chain(ctx, expr);
    for (idx, factor) in factors.iter().copied().enumerate() {
        let replacement = match ctx.get(factor).clone() {
            Expr::Neg(inner) => Some(inner),
            Expr::Number(n) if n.is_negative() => Some(ctx.add(Expr::Number(-n))),
            _ => None,
        };

        let Some(replacement) = replacement else {
            continue;
        };

        let mut rebuilt = None;
        for (j, other) in factors.iter().copied().enumerate() {
            let current = if j == idx { replacement } else { other };
            rebuilt = Some(match rebuilt {
                Some(acc) => smart_mul(ctx, acc, current),
                None => current,
            });
        }
        return rebuilt;
    }

    None
}

fn strip_top_level_or_div_numerator_negation(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<ExprId> {
    if let Some(inner) = strip_unit_negation(ctx, expr) {
        return Some(inner);
    }

    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let (numerator, denominator) = (*numerator, *denominator);
    let positive_numerator = strip_unit_negation(ctx, numerator)?;
    Some(ctx.add(Expr::Div(positive_numerator, denominator)))
}

fn negate_preserving_fraction_numerator(ctx: &mut cas_ast::Context, expr: ExprId) -> ExprId {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return ctx.add(Expr::Neg(expr));
    };
    let (numerator, denominator) = (*numerator, *denominator);
    let negated_numerator = ctx.add(Expr::Neg(numerator));
    ctx.add(Expr::Div(negated_numerator, denominator))
}

fn tan_or_cot_square(ctx: &cas_ast::Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if !is_small_integer(ctx, *exp, 2) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(*base) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    if ctx.is_builtin(*fn_id, BuiltinFn::Tan) {
        Some((BuiltinFn::Tan, args[0]))
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cot) {
        Some((BuiltinFn::Cot, args[0]))
    } else {
        None
    }
}

fn tan_call_arg(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if ctx.is_builtin(*fn_id, BuiltinFn::Tan) && args.len() == 1 {
        Some(args[0])
    } else {
        None
    }
}

fn tan_power_arg(ctx: &cas_ast::Context, expr: ExprId, power: i64) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if !is_small_integer(ctx, *exp, power) {
        return None;
    }
    tan_call_arg(ctx, *base)
}

fn extract_tan_triple_angle_numerator_arg(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Sub(left, right) = ctx.get(expr) else {
        return None;
    };

    let Expr::Mul(lhs, rhs) = ctx.get(*left) else {
        return None;
    };
    let linear_term = if is_small_integer(ctx, *lhs, 3) {
        *rhs
    } else if is_small_integer(ctx, *rhs, 3) {
        *lhs
    } else {
        return None;
    };
    let linear_arg = tan_call_arg(ctx, linear_term)?;
    let cubic_arg = tan_power_arg(ctx, *right, 3)?;
    if compare_expr(ctx, linear_arg, cubic_arg) == Ordering::Equal {
        Some(linear_arg)
    } else {
        None
    }
}

fn matches_tan_triple_angle_denominator(
    ctx: &cas_ast::Context,
    expr: ExprId,
    expected_arg: ExprId,
) -> bool {
    let Expr::Sub(left, right) = ctx.get(expr) else {
        return false;
    };
    if !is_small_integer(ctx, *left, 1) {
        return false;
    }

    let Expr::Mul(lhs, rhs) = ctx.get(*right) else {
        return false;
    };
    let squared = if is_small_integer(ctx, *lhs, 3) {
        *rhs
    } else if is_small_integer(ctx, *rhs, 3) {
        *lhs
    } else {
        return false;
    };

    let Some(arg) = tan_power_arg(ctx, squared, 2) else {
        return false;
    };
    compare_expr(ctx, arg, expected_arg) == Ordering::Equal
}

fn try_rewrite_negated_cos_double_angle_contraction_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let two = ctx.num(2);
    let build_candidate = |ctx: &mut cas_ast::Context, arg: ExprId| {
        let double_arg = smart_mul(ctx, two, arg);
        let cos_double = ctx.call_builtin(BuiltinFn::Cos, vec![double_arg]);
        let candidate = ctx.add(Expr::Neg(cos_double));
        strong_target_match(ctx, candidate, target_expr).then_some(DeriveTrigRewrite {
            rewritten: candidate,
            kind: DeriveTrigRewriteKind::DoubleAngleNegCosContract,
        })
    };

    let terms = add_terms_signed(ctx, expr);
    if terms.len() != 2 {
        return None;
    }

    let mut positive_term = None;
    let mut negative_term = None;
    for (term, sign) in terms {
        match sign {
            Sign::Pos if positive_term.is_none() => positive_term = Some(term),
            Sign::Neg if negative_term.is_none() => negative_term = Some(term),
            _ => return None,
        }
    }

    let (Some(positive_term), Some(negative_term)) = (positive_term, negative_term) else {
        return None;
    };

    if let (Some((BuiltinFn::Sin, sin_arg)), Some((BuiltinFn::Cos, cos_arg))) = (
        trig_square(ctx, positive_term),
        trig_square(ctx, negative_term),
    ) {
        if strong_target_match(ctx, sin_arg, cos_arg) {
            return build_candidate(ctx, sin_arg);
        }
    }

    if let Some((BuiltinFn::Sin, arg)) = scaled_sin_or_cos_square(ctx, positive_term, 2) {
        if is_small_integer(ctx, negative_term, 1) {
            return build_candidate(ctx, arg);
        }
    }

    if is_small_integer(ctx, positive_term, 1) {
        if let Some((BuiltinFn::Cos, arg)) = scaled_sin_or_cos_square(ctx, negative_term, 2) {
            return build_candidate(ctx, arg);
        }
    }

    None
}

fn try_rewrite_negated_sin_double_angle_contraction_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let positive_expr = strip_unit_negation(ctx, expr)?;
    let rewrite = cas_math::trig_contraction_support::try_rewrite_double_angle_contraction_expr(
        ctx,
        positive_expr,
    )?;

    let negated = ctx.add(Expr::Neg(rewrite.rewritten));
    if !strong_target_match(ctx, negated, target_expr) {
        return None;
    }

    match rewrite.kind {
        cas_math::trig_contraction_support::TrigContractionRewriteKind::DoubleAngleSin => {
            Some(DeriveTrigRewrite {
                rewritten: negated,
                kind: DeriveTrigRewriteKind::DoubleAngleNegSinContract,
            })
        }
        _ => None,
    }
}

fn try_rewrite_reciprocal_trig_contraction_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    if let Some(rewrite) =
        try_rewrite_cos_diff_over_sin_diff_quotient_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    let rewrite =
        cas_math::trig_canonicalization_support::try_rewrite_trig_quotient_div_expr(ctx, expr)?;

    if !strong_target_match(ctx, rewrite.rewritten, target_expr) {
        return None;
    }

    let kind = match rewrite.kind {
        Some(
            cas_math::trig_canonicalization_support::TrigCanonicalRewriteKind::SinOverCosToTan,
        ) => DeriveTrigRewriteKind::RecognizeSinOverCosAsTan,
        Some(
            cas_math::trig_canonicalization_support::TrigCanonicalRewriteKind::OneOverCosToSec,
        ) => DeriveTrigRewriteKind::RecognizeRecipCosAsSec,
        Some(
            cas_math::trig_canonicalization_support::TrigCanonicalRewriteKind::OneOverSinToCsc,
        ) => DeriveTrigRewriteKind::RecognizeRecipSinAsCsc,
        Some(
            cas_math::trig_canonicalization_support::TrigCanonicalRewriteKind::CosOverSinToCot,
        ) => DeriveTrigRewriteKind::RecognizeCosOverSinAsCot,
        _ => return None,
    };

    Some(DeriveTrigRewrite {
        rewritten: rewrite.rewritten,
        kind,
    })
}

fn try_rewrite_tangent_double_angle_contraction_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let rewrite =
        cas_math::trig_contraction_support::try_rewrite_tan_double_angle_contraction_expr(
            ctx, expr,
        )?;

    if !strong_target_match(ctx, rewrite.rewritten, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten: rewrite.rewritten,
        kind: DeriveTrigRewriteKind::DoubleAngleTanContract,
    })
}

fn try_rewrite_cos_diff_over_sin_diff_quotient_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let (num_left, num_right) = extract_trig_difference_args(ctx, *numerator, BuiltinFn::Cos)?;
    let (den_left, den_right) = extract_trig_difference_args(ctx, *denominator, BuiltinFn::Sin)?;
    let average_arg = half_sum_linear_angle_terms(ctx, num_left, num_right)?;

    let candidate = if compare_expr(ctx, num_left, den_right) == Ordering::Equal
        && compare_expr(ctx, num_right, den_left) == Ordering::Equal
    {
        ctx.call_builtin(BuiltinFn::Tan, vec![average_arg])
    } else if compare_expr(ctx, num_left, den_left) == Ordering::Equal
        && compare_expr(ctx, num_right, den_right) == Ordering::Equal
    {
        let tan = ctx.call_builtin(BuiltinFn::Tan, vec![average_arg]);
        ctx.add(Expr::Neg(tan))
    } else {
        return None;
    };

    if !strong_target_match(ctx, candidate, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten: candidate,
        kind: DeriveTrigRewriteKind::RecognizeCosDiffOverSinDiffAsTan,
    })
}

fn try_rewrite_negated_reciprocal_trig_contraction_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let positive_expr = strip_top_level_or_div_numerator_negation(ctx, expr)?;
    let rewrite = cas_math::trig_canonicalization_support::try_rewrite_trig_quotient_div_expr(
        ctx,
        positive_expr,
    )?;
    let rewritten = ctx.add(Expr::Neg(rewrite.rewritten));

    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    let kind = match rewrite.kind {
        Some(
            cas_math::trig_canonicalization_support::TrigCanonicalRewriteKind::SinOverCosToTan,
        ) => DeriveTrigRewriteKind::RecognizeSinOverCosAsTan,
        Some(
            cas_math::trig_canonicalization_support::TrigCanonicalRewriteKind::OneOverCosToSec,
        ) => DeriveTrigRewriteKind::RecognizeRecipCosAsSec,
        Some(
            cas_math::trig_canonicalization_support::TrigCanonicalRewriteKind::OneOverSinToCsc,
        ) => DeriveTrigRewriteKind::RecognizeRecipSinAsCsc,
        Some(
            cas_math::trig_canonicalization_support::TrigCanonicalRewriteKind::CosOverSinToCot,
        ) => DeriveTrigRewriteKind::RecognizeCosOverSinAsCot,
        _ => return None,
    };

    Some(DeriveTrigRewrite { rewritten, kind })
}

fn try_rewrite_half_angle_square_contraction_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let (trig_fn, arg) = extract_half_angle_square_additive_form(ctx, expr)?;
    let trig_call = ctx.call_builtin(trig_fn, vec![arg]);
    let rewritten = pow2(ctx, trig_call);

    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    let kind = match trig_fn {
        BuiltinFn::Sin => DeriveTrigRewriteKind::HalfAngleSinSquaredContract,
        BuiltinFn::Cos => DeriveTrigRewriteKind::HalfAngleCosSquaredContract,
        _ => unreachable!("only sin/cos are valid here"),
    };

    Some(DeriveTrigRewrite { rewritten, kind })
}

fn try_rewrite_negated_half_angle_square_contraction_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let (trig_fn, arg) = if let Some(positive_expr) = strip_unit_negation(ctx, expr) {
        extract_half_angle_square_additive_form(ctx, positive_expr)?
    } else {
        extract_negated_half_angle_square_additive_form(ctx, expr)?
    };
    let trig_call = ctx.call_builtin(trig_fn, vec![arg]);
    let positive = pow2(ctx, trig_call);
    let rewritten = ctx.add(Expr::Neg(positive));

    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    let kind = match trig_fn {
        BuiltinFn::Sin => DeriveTrigRewriteKind::HalfAngleNegSinSquaredContract,
        BuiltinFn::Cos => DeriveTrigRewriteKind::HalfAngleNegCosSquaredContract,
        _ => unreachable!("only sin/cos are valid here"),
    };

    Some(DeriveTrigRewrite { rewritten, kind })
}

fn build_negated_half_angle_square_rewritten(
    ctx: &mut cas_ast::Context,
    trig_fn: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let double_arg = smart_mul(ctx, two, arg);
    let cos_double_arg = ctx.call_builtin(BuiltinFn::Cos, vec![double_arg]);

    let numerator = match trig_fn {
        BuiltinFn::Sin => ctx.add(Expr::Sub(one, cos_double_arg)),
        BuiltinFn::Cos => ctx.add(Expr::Add(one, cos_double_arg)),
        _ => return None,
    };
    let negated_numerator = ctx.add(Expr::Neg(numerator));
    Some(ctx.add(Expr::Div(negated_numerator, two)))
}

fn extract_negated_half_angle_square_additive_form(
    ctx: &cas_ast::Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    if let Expr::Div(numerator, denominator) = ctx.get(expr) {
        if is_small_integer(ctx, *denominator, 2) {
            return extract_negated_half_angle_square_numerator(ctx, *numerator);
        }
    }

    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };

    if is_one_half(ctx, *left) {
        return extract_negated_half_angle_square_numerator(ctx, *right);
    }
    if is_one_half(ctx, *right) {
        return extract_negated_half_angle_square_numerator(ctx, *left);
    }

    None
}

fn extract_negated_half_angle_square_numerator(
    ctx: &cas_ast::Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    match ctx.get(expr) {
        Expr::Sub(lhs, rhs) if is_small_integer(ctx, *rhs, 1) => {
            extract_negated_half_angle_square_leading_term(ctx, *lhs)
        }
        Expr::Neg(inner) => extract_half_angle_square_numerator(ctx, *inner),
        _ => None,
    }
}

fn extract_negated_half_angle_square_leading_term(
    ctx: &cas_ast::Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Cos) && args.len() == 1 =>
        {
            let half_arg = extract_double_angle_inner(ctx, args[0])?;
            Some((BuiltinFn::Sin, half_arg))
        }
        Expr::Neg(inner) => {
            let Expr::Function(fn_id, args) = ctx.get(*inner) else {
                return None;
            };
            if !ctx.is_builtin(*fn_id, BuiltinFn::Cos) || args.len() != 1 {
                return None;
            }
            let half_arg = extract_double_angle_inner(ctx, args[0])?;
            Some((BuiltinFn::Cos, half_arg))
        }
        _ => None,
    }
}

fn extract_half_angle_square_additive_form(
    ctx: &cas_ast::Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    if let Expr::Div(numerator, denominator) = ctx.get(expr) {
        if is_small_integer(ctx, *denominator, 2) {
            return extract_half_angle_square_numerator(ctx, *numerator);
        }
    }

    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };

    if is_one_half(ctx, *left) {
        return extract_half_angle_square_numerator(ctx, *right);
    }
    if is_one_half(ctx, *right) {
        return extract_half_angle_square_numerator(ctx, *left);
    }

    None
}

fn extract_half_angle_square_numerator(
    ctx: &cas_ast::Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    let (kind, cosine_term) = match ctx.get(expr) {
        Expr::Sub(lhs, rhs) if is_small_integer(ctx, *lhs, 1) => (BuiltinFn::Sin, *rhs),
        Expr::Add(lhs, rhs) if is_small_integer(ctx, *lhs, 1) => (BuiltinFn::Cos, *rhs),
        Expr::Add(lhs, rhs) if is_small_integer(ctx, *rhs, 1) => (BuiltinFn::Cos, *lhs),
        _ => return None,
    };

    let Expr::Function(fn_id, args) = ctx.get(cosine_term) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Cos) || args.len() != 1 {
        return None;
    }

    let half_arg = extract_double_angle_inner(ctx, args[0])?;
    Some((kind, half_arg))
}

fn extract_double_angle_inner(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };

    if is_small_integer(ctx, *left, 2) {
        return Some(*right);
    }
    if is_small_integer(ctx, *right, 2) {
        return Some(*left);
    }

    None
}

fn try_rewrite_cos_double_angle_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Cos) || args.len() != 1 {
        return None;
    }

    let inner = args[0];
    let Expr::Mul(left, right) = ctx.get(inner) else {
        return None;
    };
    let doubled = if is_small_integer(ctx, *left, 2) {
        *right
    } else if is_small_integer(ctx, *right, 2) {
        *left
    } else {
        return None;
    };

    let sin_call = ctx.call_builtin(BuiltinFn::Sin, vec![doubled]);
    let sin_sq = pow2(ctx, sin_call);
    let cos_call = ctx.call_builtin(BuiltinFn::Cos, vec![doubled]);
    let cos_sq = pow2(ctx, cos_call);
    let one = ctx.num(1);
    let two = ctx.num(2);

    let two_sin_sq = ctx.add(Expr::Mul(two, sin_sq));
    let one_minus_two_sin_sq = ctx.add(Expr::Sub(one, two_sin_sq));
    if strong_target_match(ctx, one_minus_two_sin_sq, target_expr) {
        return Some(DeriveTrigRewrite {
            rewritten: one_minus_two_sin_sq,
            kind: DeriveTrigRewriteKind::DoubleAngleCosOneMinusTwoSinSq,
        });
    }

    let two_cos_sq = ctx.add(Expr::Mul(two, cos_sq));
    let two_cos_sq_minus_one = ctx.add(Expr::Sub(two_cos_sq, one));
    if strong_target_match(ctx, two_cos_sq_minus_one, target_expr) {
        return Some(DeriveTrigRewrite {
            rewritten: two_cos_sq_minus_one,
            kind: DeriveTrigRewriteKind::DoubleAngleCosTwoCosSqMinusOne,
        });
    }

    None
}

fn try_rewrite_negated_cos_double_angle_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let inner_cos = strip_unit_negation(ctx, expr)?;
    let doubled = double_angle_cos_inner(ctx, inner_cos)?;

    let sin_call = ctx.call_builtin(BuiltinFn::Sin, vec![doubled]);
    let sin_sq = pow2(ctx, sin_call);
    let cos_call = ctx.call_builtin(BuiltinFn::Cos, vec![doubled]);
    let cos_sq = pow2(ctx, cos_call);
    let one = ctx.num(1);
    let two = ctx.num(2);

    let two_sin_sq = ctx.add(Expr::Mul(two, sin_sq));
    let two_sin_sq_minus_one = ctx.add(Expr::Sub(two_sin_sq, one));
    if strong_target_match(ctx, two_sin_sq_minus_one, target_expr) {
        return Some(DeriveTrigRewrite {
            rewritten: two_sin_sq_minus_one,
            kind: DeriveTrigRewriteKind::DoubleAngleNegCosExpand,
        });
    }

    let two_cos_sq = ctx.add(Expr::Mul(two, cos_sq));
    let one_minus_two_cos_sq = ctx.add(Expr::Sub(one, two_cos_sq));
    if strong_target_match(ctx, one_minus_two_cos_sq, target_expr) {
        return Some(DeriveTrigRewrite {
            rewritten: one_minus_two_cos_sq,
            kind: DeriveTrigRewriteKind::DoubleAngleNegCosExpand,
        });
    }

    let sin_sq_minus_cos_sq = ctx.add(Expr::Sub(sin_sq, cos_sq));
    if strong_target_match(ctx, sin_sq_minus_cos_sq, target_expr) {
        return Some(DeriveTrigRewrite {
            rewritten: sin_sq_minus_cos_sq,
            kind: DeriveTrigRewriteKind::DoubleAngleNegCosExpand,
        });
    }

    None
}

fn try_rewrite_negated_sin_double_angle_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let inner_sin = strip_unit_negation(ctx, expr)?;
    let Expr::Function(fn_id, args) = ctx.get(inner_sin) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Sin) || args.len() != 1 {
        return None;
    }

    let doubled = extract_double_angle_inner(ctx, args[0])?;
    let sin_call = ctx.call_builtin(BuiltinFn::Sin, vec![doubled]);
    let cos_call = ctx.call_builtin(BuiltinFn::Cos, vec![doubled]);
    let sin_cos = smart_mul(ctx, sin_call, cos_call);
    let neg_two = ctx.num(-2);
    let rewritten = smart_mul(ctx, neg_two, sin_cos);

    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten,
        kind: DeriveTrigRewriteKind::DoubleAngleNegSinExpand,
    })
}

fn try_rewrite_mixed_sine_double_angle_expansion_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let (mixed_kind, arg) = extract_scaled_sine_double_angle_linear_term(ctx, expr, 2)?;
    let four = ctx.num(4);
    let sin_call = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
    let cos_call = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
    let rewritten = match mixed_kind {
        BuiltinFn::Sin => {
            let sin_sq = pow2(ctx, sin_call);
            let inner_product = smart_mul(ctx, sin_sq, cos_call);
            smart_mul(ctx, four, inner_product)
        }
        BuiltinFn::Cos => {
            let cos_sq = pow2(ctx, cos_call);
            let inner_product = smart_mul(ctx, cos_sq, sin_call);
            smart_mul(ctx, four, inner_product)
        }
        _ => return None,
    };

    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    Some(DeriveTrigRewrite {
        rewritten: target_expr,
        kind: DeriveTrigRewriteKind::DoubleAngleSin,
    })
}

fn try_rewrite_sum_to_product_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    if let Some(rewrite) =
        cas_math::trig_sum_product_support::try_rewrite_sum_to_product_contraction_expr(ctx, expr)
    {
        let rewritten =
            normalize_sum_to_product_candidate(ctx, rewrite.rewritten, rewrite.kind, target_expr)?;

        let kind = match rewrite.kind {
            cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinSum => {
                DeriveTrigRewriteKind::SumToProductSinSum
            }
            cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinDiff => {
                DeriveTrigRewriteKind::SumToProductSinDiff
            }
            cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::CosSum => {
                DeriveTrigRewriteKind::SumToProductCosSum
            }
            cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::CosDiff => {
                DeriveTrigRewriteKind::SumToProductCosDiff
            }
        };

        return Some(DeriveTrigRewrite { rewritten, kind });
    }

    try_rewrite_generic_sum_to_product_target_aware(ctx, expr, target_expr)
}

fn try_rewrite_generic_sum_to_product_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    if let Some((rewritten, kind)) = build_generic_sum_to_product_candidates(ctx, expr)
        .into_iter()
        .next()
    {
        let rewritten = normalize_sum_to_product_candidate(
            ctx,
            rewritten,
            map_derive_sum_to_product_kind(kind),
            target_expr,
        )?;
        return Some(DeriveTrigRewrite { rewritten, kind });
    }

    None
}

fn build_generic_sum_to_product_candidates(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Vec<(ExprId, DeriveTrigRewriteKind)> {
    let mut candidates = Vec::new();

    if let Some((arg_a, arg_b)) =
        cas_math::trig_sum_product_support::extract_trig_two_term_sum(ctx, expr, "sin")
    {
        candidates.push((
            build_sum_to_product_rewritten(
                ctx,
                arg_a,
                arg_b,
                DeriveTrigRewriteKind::SumToProductSinSum,
            ),
            DeriveTrigRewriteKind::SumToProductSinSum,
        ));
    }

    if let Some((arg_a, arg_b)) =
        cas_math::trig_sum_product_support::extract_trig_two_term_diff(ctx, expr, "sin")
    {
        candidates.push((
            build_sum_to_product_rewritten(
                ctx,
                arg_a,
                arg_b,
                DeriveTrigRewriteKind::SumToProductSinDiff,
            ),
            DeriveTrigRewriteKind::SumToProductSinDiff,
        ));
    }

    if let Some((arg_a, arg_b)) =
        cas_math::trig_sum_product_support::extract_trig_two_term_sum(ctx, expr, "cos")
    {
        candidates.push((
            build_sum_to_product_rewritten(
                ctx,
                arg_a,
                arg_b,
                DeriveTrigRewriteKind::SumToProductCosSum,
            ),
            DeriveTrigRewriteKind::SumToProductCosSum,
        ));
    }

    if let Some((arg_a, arg_b)) =
        cas_math::trig_sum_product_support::extract_trig_two_term_diff(ctx, expr, "cos")
    {
        candidates.push((
            build_sum_to_product_rewritten(
                ctx,
                arg_a,
                arg_b,
                DeriveTrigRewriteKind::SumToProductCosDiff,
            ),
            DeriveTrigRewriteKind::SumToProductCosDiff,
        ));
    }

    candidates
}

fn build_sum_to_product_rewritten(
    ctx: &mut cas_ast::Context,
    arg_a: ExprId,
    arg_b: ExprId,
    kind: DeriveTrigRewriteKind,
) -> ExprId {
    let two = ctx.num(2);
    let sum_arg = ctx.add(Expr::Add(arg_a, arg_b));
    let diff_arg = ctx.add(Expr::Sub(arg_a, arg_b));
    let avg_arg = ctx.add(Expr::Div(sum_arg, two));
    let half_diff_arg = ctx.add(Expr::Div(diff_arg, two));

    match kind {
        DeriveTrigRewriteKind::SumToProductSinSum => {
            let sin_avg = ctx.call_builtin(BuiltinFn::Sin, vec![avg_arg]);
            let cos_half_diff = ctx.call_builtin(BuiltinFn::Cos, vec![half_diff_arg]);
            let product = ctx.add(Expr::Mul(sin_avg, cos_half_diff));
            smart_mul(ctx, two, product)
        }
        DeriveTrigRewriteKind::SumToProductSinDiff => {
            let cos_avg = ctx.call_builtin(BuiltinFn::Cos, vec![avg_arg]);
            let sin_half_diff = ctx.call_builtin(BuiltinFn::Sin, vec![half_diff_arg]);
            let product = ctx.add(Expr::Mul(cos_avg, sin_half_diff));
            smart_mul(ctx, two, product)
        }
        DeriveTrigRewriteKind::SumToProductCosSum => {
            let cos_avg = ctx.call_builtin(BuiltinFn::Cos, vec![avg_arg]);
            let cos_half_diff = ctx.call_builtin(BuiltinFn::Cos, vec![half_diff_arg]);
            let product = ctx.add(Expr::Mul(cos_avg, cos_half_diff));
            smart_mul(ctx, two, product)
        }
        DeriveTrigRewriteKind::SumToProductCosDiff => {
            let sin_avg = ctx.call_builtin(BuiltinFn::Sin, vec![avg_arg]);
            let sin_half_diff = ctx.call_builtin(BuiltinFn::Sin, vec![half_diff_arg]);
            let product = ctx.add(Expr::Mul(sin_avg, sin_half_diff));
            let two_times_product = smart_mul(ctx, two, product);
            ctx.add(Expr::Neg(two_times_product))
        }
        _ => unreachable!("expected sum-to-product rewrite kind"),
    }
}

fn map_derive_sum_to_product_kind(
    kind: DeriveTrigRewriteKind,
) -> cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind {
    match kind {
        DeriveTrigRewriteKind::SumToProductSinSum => {
            cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinSum
        }
        DeriveTrigRewriteKind::SumToProductSinDiff => {
            cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinDiff
        }
        DeriveTrigRewriteKind::SumToProductCosSum => {
            cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::CosSum
        }
        DeriveTrigRewriteKind::SumToProductCosDiff => {
            cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::CosDiff
        }
        _ => unreachable!("expected sum-to-product rewrite kind"),
    }
}

fn normalize_sum_to_product_candidate(
    ctx: &mut cas_ast::Context,
    rewritten: ExprId,
    kind: cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind,
    target_expr: ExprId,
) -> Option<ExprId> {
    if trig_presentational_target_match(ctx, rewritten, target_expr) {
        return Some(rewritten);
    }

    let normalized = normalize_trig_negative_parity_tree(ctx, rewritten);
    if trig_presentational_target_match(ctx, normalized, target_expr) {
        return Some(normalized);
    }

    if matches!(
        kind,
        cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinDiff
            | cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::CosDiff
    ) {
        let negated = ctx.add(Expr::Neg(normalized));
        if trig_presentational_target_match(ctx, negated, target_expr) {
            return Some(negated);
        }
    }

    None
}

fn normalize_trig_negative_parity_tree(ctx: &mut cas_ast::Context, expr: ExprId) -> ExprId {
    if let Some(rewrite) = try_rewrite_circular_trig_odd_even_parity_expr(ctx, expr) {
        return normalize_trig_negative_parity_tree(ctx, rewrite.rewritten);
    }

    match ctx.get(expr).clone() {
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => expr,
        Expr::Add(left, right) => rebuild_binary(ctx, expr, Expr::Add, left, right),
        Expr::Sub(left, right) => rebuild_binary(ctx, expr, Expr::Sub, left, right),
        Expr::Mul(left, right) => rebuild_binary(ctx, expr, Expr::Mul, left, right),
        Expr::Div(left, right) => rebuild_binary(ctx, expr, Expr::Div, left, right),
        Expr::Pow(base, exp) => rebuild_binary(ctx, expr, Expr::Pow, base, exp),
        Expr::Neg(inner) => {
            let rewritten = normalize_trig_negative_parity_tree(ctx, inner);
            if rewritten == inner {
                expr
            } else {
                ctx.add(Expr::Neg(rewritten))
            }
        }
        Expr::Function(name, args) => {
            let rewritten_args = args
                .iter()
                .map(|arg| normalize_trig_negative_parity_tree(ctx, *arg))
                .collect::<Vec<_>>();
            if rewritten_args == args {
                expr
            } else {
                ctx.add(Expr::Function(name, rewritten_args))
            }
        }
        Expr::Matrix { rows, cols, data } => {
            let rewritten_data = data
                .iter()
                .map(|item| normalize_trig_negative_parity_tree(ctx, *item))
                .collect::<Vec<_>>();
            if rewritten_data == data {
                expr
            } else {
                ctx.add(Expr::Matrix {
                    rows,
                    cols,
                    data: rewritten_data,
                })
            }
        }
        Expr::Hold(inner) => {
            let rewritten = normalize_trig_negative_parity_tree(ctx, inner);
            if rewritten == inner {
                expr
            } else {
                ctx.add(Expr::Hold(rewritten))
            }
        }
    }
}

fn rebuild_binary(
    ctx: &mut cas_ast::Context,
    original: ExprId,
    ctor: fn(ExprId, ExprId) -> Expr,
    left: ExprId,
    right: ExprId,
) -> ExprId {
    let rewritten_left = normalize_trig_negative_parity_tree(ctx, left);
    let rewritten_right = normalize_trig_negative_parity_tree(ctx, right);
    if rewritten_left == left && rewritten_right == right {
        original
    } else {
        ctx.add(ctor(rewritten_left, rewritten_right))
    }
}

fn pow2(ctx: &mut cas_ast::Context, expr: ExprId) -> ExprId {
    let two = ctx.num(2);
    ctx.add(Expr::Pow(expr, two))
}

fn pow3(ctx: &mut cas_ast::Context, expr: ExprId) -> ExprId {
    let three = ctx.num(3);
    ctx.add(Expr::Pow(expr, three))
}

fn is_small_integer(ctx: &cas_ast::Context, expr: ExprId, value: i64) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_integer() && n.to_integer() == value.into())
}

fn is_one_half(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if *n == BigRational::new(1.into(), 2.into()))
}

#[cfg(test)]
mod tests {
    use super::{
        generate_trig_additive_term_bridge_rewrites, generate_trig_bridge_rewrites,
        should_try_trig_planner_before_simplify, try_rewrite_product_to_sum_target_aware,
        try_rewrite_pythagorean_factor_form_target_aware,
        try_rewrite_shifted_double_angle_target_aware,
        try_rewrite_shifted_reciprocal_pythagorean_target_aware,
        try_rewrite_sum_to_product_target_aware, try_rewrite_trig_contraction_target_aware,
        try_rewrite_trig_expansion, try_rewrite_trig_identity_to_one_target_aware,
        try_rewrite_trig_special_value_target_aware, DeriveTrigRewriteKind,
    };
    use crate::derive::{presentational_target_match, strong_target_match};
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn target_aware_trig_rewrite_recognizes_special_values() {
        let cases = [
            ("sin(0)", "0"),
            ("cos(0)", "1"),
            ("tan(0)", "0"),
            ("asin(0)", "0"),
            ("acos(1)", "0"),
            ("atan(0)", "0"),
            ("arctan(sqrt(3))", "pi/3"),
            ("cos(2*pi/3)", "-1/2"),
            ("cos(3*pi/4)", "-sqrt(2)/2"),
            ("cos(5*pi/6)", "-sqrt(3)/2"),
            ("sec(pi/4)", "sqrt(2)"),
            ("csc(pi/6)", "2"),
            ("cot(pi/4)", "1"),
        ];

        for (source_text, target_text) in cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_trig_special_value_target_aware(&mut ctx, source, target)
                .unwrap_or_else(|| panic!("special-value rewrite for `{source_text}`"));

            assert_eq!(rewrite.kind, DeriveTrigRewriteKind::TrigSpecialValue);
            assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
        }
    }

    #[test]
    fn target_aware_trig_special_value_rejects_parity_rewrites() {
        let mut ctx = Context::new();
        let source = parse("tan(-x)", &mut ctx).expect("source");
        let target = parse("-tan(x)", &mut ctx).expect("target");

        assert!(try_rewrite_trig_special_value_target_aware(&mut ctx, source, target).is_none());
    }

    #[test]
    fn rewrites_negative_trig_parity_variants_target_aware() {
        for (source_text, target_text) in [
            ("tan(-x)", "-tan(x)"),
            ("sec(-x)", "sec(x)"),
            ("csc(-x)+a", "-csc(x)+a"),
        ] {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target)
                .unwrap_or_else(|| panic!("rewrite for `{source_text}` -> `{target_text}`"));

            assert_eq!(rewrite.kind, DeriveTrigRewriteKind::TrigOddEvenParity);
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }
    }

    #[test]
    fn rewrites_half_angle_tangent_variants_target_aware() {
        let contraction_cases = [
            (
                "(1-cos(2*x))/sin(2*x)",
                "tan(x)",
                DeriveTrigRewriteKind::HalfAngleTangent,
            ),
            (
                "sin(2*x)/(1+cos(2*x))",
                "tan(x)",
                DeriveTrigRewriteKind::HalfAngleTangent,
            ),
        ];

        for (source_text, target_text, expected_kind) in contraction_cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_trig_contraction_target_aware(&mut ctx, source, target)
                .unwrap_or_else(|| panic!("rewrite for `{source_text}` -> `{target_text}`"));

            assert_eq!(rewrite.kind, expected_kind);
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }

        let expansion_cases = [
            (
                "tan(x)",
                "(1-cos(2*x))/sin(2*x)",
                DeriveTrigRewriteKind::HalfAngleTangentExpandOneMinusCosOverSin,
            ),
            (
                "tan(x)",
                "sin(2*x)/(1+cos(2*x))",
                DeriveTrigRewriteKind::HalfAngleTangentExpandSinOverOnePlusCos,
            ),
            (
                "tan(x/2)",
                "sin(x)/(1+cos(x))",
                DeriveTrigRewriteKind::HalfAngleTangentExpandSinOverOnePlusCos,
            ),
        ];

        for (source_text, target_text, expected_kind) in expansion_cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }
    }

    #[test]
    fn rewrites_tangent_half_angle_substitution_variants_target_aware() {
        for (source_text, target_text, expected_kind) in [
            (
                "sin(x)",
                "2*tan(x/2)/(1+tan(x/2)^2)",
                DeriveTrigRewriteKind::TangentHalfAngleSubstitutionSin,
            ),
            (
                "cos(x)",
                "(1-tan(x/2)^2)/(1+tan(x/2)^2)",
                DeriveTrigRewriteKind::TangentHalfAngleSubstitutionCos,
            ),
            (
                "sin(2*x)",
                "2*tan(x)/(1+tan(x)^2)",
                DeriveTrigRewriteKind::TangentHalfAngleSubstitutionSin,
            ),
        ] {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target)
                .unwrap_or_else(|| panic!("rewrite for `{source_text}` -> `{target_text}`"));

            assert_eq!(rewrite.kind, expected_kind);
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }
    }

    #[test]
    fn rewrites_tangent_double_angle_variants_target_aware() {
        let mut ctx = Context::new();
        let source = parse("tan(2*x)", &mut ctx).expect("source");
        let target = parse("2*tan(x)/(1-tan(x)^2)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::DoubleAngleTanExpand);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));

        let mut ctx = Context::new();
        let source = parse("2*tan(x)/(1-tan(x)^2)", &mut ctx).expect("source");
        let target = parse("tan(2*x)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::DoubleAngleTanContract);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_negative_half_angle_tangent_variants_target_aware() {
        let contraction_cases = [
            (
                "-(1-cos(2*x))/sin(2*x)",
                "-tan(x)",
                DeriveTrigRewriteKind::HalfAngleTangent,
            ),
            (
                "-sin(2*x)/(1+cos(2*x))",
                "-tan(x)",
                DeriveTrigRewriteKind::HalfAngleTangent,
            ),
        ];

        for (source_text, target_text, expected_kind) in contraction_cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_trig_contraction_target_aware(&mut ctx, source, target)
                .unwrap_or_else(|| panic!("rewrite for `{source_text}` -> `{target_text}`"));

            assert_eq!(rewrite.kind, expected_kind);
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }

        let expansion_cases = [
            (
                "-tan(x)",
                "-(1-cos(2*x))/sin(2*x)",
                DeriveTrigRewriteKind::HalfAngleTangentExpandOneMinusCosOverSin,
            ),
            (
                "-tan(x)",
                "-sin(2*x)/(1+cos(2*x))",
                DeriveTrigRewriteKind::HalfAngleTangentExpandSinOverOnePlusCos,
            ),
        ];

        for (source_text, target_text, expected_kind) in expansion_cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }
    }

    #[test]
    fn contracts_tabulated_scaled_trig_targets_aware() {
        let cases = [
            (
                "2*sin(a*x)*cos(a*x)",
                "sin(2*a*x)",
                DeriveTrigRewriteKind::DoubleAngleSin,
            ),
            (
                "sin(2*a*x)/cos(2*a*x)",
                "tan(2*a*x)",
                DeriveTrigRewriteKind::RecognizeSinOverCosAsTan,
            ),
            (
                "1/cos(a*x)",
                "sec(a*x)",
                DeriveTrigRewriteKind::RecognizeRecipCosAsSec,
            ),
            (
                "1/sin(a*x)",
                "csc(a*x)",
                DeriveTrigRewriteKind::RecognizeRecipSinAsCsc,
            ),
            (
                "cos(a*x)/sin(a*x)",
                "cot(a*x)",
                DeriveTrigRewriteKind::RecognizeCosOverSinAsCot,
            ),
            (
                "1 + tan(a*x)^2",
                "sec(a*x)^2",
                DeriveTrigRewriteKind::RecognizeSecSquared,
            ),
            (
                "1 + cot(a*x)^2",
                "csc(a*x)^2",
                DeriveTrigRewriteKind::RecognizeCscSquared,
            ),
        ];

        for (source_text, target_text, expected_kind) in cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_trig_contraction_target_aware(&mut ctx, source, target)
                .unwrap_or_else(|| {
                    panic!(
                        "expected scaled trig contraction for `{source_text}` -> `{target_text}`"
                    )
                });

            assert_eq!(
                rewrite.kind, expected_kind,
                "unexpected rewrite kind for `{source_text}` -> `{target_text}`"
            );
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }
    }

    #[test]
    fn rewrites_sec_reciprocal_expansion_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sec(x)", &mut ctx).expect("source");
        let target = parse("1/cos(x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::ExpandSecToRecipCos);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn expands_angle_sum_sine_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(x+y)", &mut ctx).expect("source");
        let target = parse("sin(x)*cos(y)+cos(x)*sin(y)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::AngleSumDiff);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn generates_trig_bridge_rewrite_for_sine_angle_sum_contraction() {
        let mut ctx = Context::new();
        let source = parse("sin(2*x)*cos(x)+cos(2*x)*sin(x)", &mut ctx).expect("source");
        let target = parse("sin(3*x)", &mut ctx).expect("target");
        let rewrites = generate_trig_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::AngleSumDiff
                && strong_target_match(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_trig_bridge_rewrite_for_cosine_angle_sum_contraction() {
        let mut ctx = Context::new();
        let source = parse("cos(2*x)*cos(x)-sin(2*x)*sin(x)", &mut ctx).expect("source");
        let target = parse("cos(3*x)", &mut ctx).expect("target");
        let rewrites = generate_trig_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::AngleSumDiff
                && strong_target_match(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn contracts_angle_sum_sine_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(x)*cos(y)+cos(x)*sin(y)", &mut ctx).expect("source");
        let target = parse("sin(x+y)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::AngleSumDiff);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_tangent_angle_sum_diff_variants_target_aware() {
        for (source_text, target_text) in [
            ("tan(x+y)", "(tan(x)+tan(y))/(1-tan(x)*tan(y))"),
            ("tan(x-y)", "(tan(x)-tan(y))/(1+tan(x)*tan(y))"),
        ] {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

            assert_eq!(
                rewrite.kind,
                DeriveTrigRewriteKind::TangentAngleSumDiffExpand
            );
            assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
        }

        for (source_text, target_text) in [
            ("(tan(x)+tan(y))/(1-tan(x)*tan(y))", "tan(x+y)"),
            ("(tan(x)-tan(y))/(1+tan(x)*tan(y))", "tan(x-y)"),
        ] {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_trig_contraction_target_aware(&mut ctx, source, target)
                .expect("rewrite");

            assert_eq!(
                rewrite.kind,
                DeriveTrigRewriteKind::TangentAngleSumDiffContract
            );
            assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
        }
    }

    #[test]
    fn expands_recursive_sine_six_x_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(6*x)", &mut ctx).expect("source");
        let target = parse("sin(5*x)*cos(x)+cos(5*x)*sin(x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::RecursiveAngleSumDiff);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn contracts_recursive_sine_six_x_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(5*x)*cos(x)+cos(5*x)*sin(x)", &mut ctx).expect("source");
        let target = parse("sin(6*x)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::AngleSumDiff);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn contracts_triple_angle_sine_target_aware() {
        let mut ctx = Context::new();
        let source = parse("3*sin(x)-4*sin(x)^3", &mut ctx).expect("source");
        let target = parse("sin(3*x)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::TripleAngleSin);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn expands_triple_angle_sine_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(3*x)", &mut ctx).expect("source");
        let target = parse("3*sin(x)-4*sin(x)^3", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::TripleAngleSin);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn contracts_triple_angle_cosine_target_aware() {
        let mut ctx = Context::new();
        let source = parse("4*cos(x)^3-3*cos(x)", &mut ctx).expect("source");
        let target = parse("cos(3*x)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::TripleAngleCos);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn expands_triple_angle_cosine_target_aware() {
        let mut ctx = Context::new();
        let source = parse("cos(3*x)", &mut ctx).expect("source");
        let target = parse("4*cos(x)^3-3*cos(x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::TripleAngleCos);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn contracts_triple_angle_tangent_target_aware() {
        let mut ctx = Context::new();
        let source = parse("(3*tan(x)-tan(x)^3)/(1-3*tan(x)^2)", &mut ctx).expect("source");
        let target = parse("tan(3*x)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::TripleAngleTan);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn expands_triple_angle_tangent_target_aware() {
        let mut ctx = Context::new();
        let source = parse("tan(3*x)", &mut ctx).expect("source");
        let target = parse("(3*tan(x)-tan(x)^3)/(1-3*tan(x)^2)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::TripleAngleTan);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_negative_triple_angle_variants_target_aware() {
        let contraction_cases = [
            (
                "-(3*sin(x)-4*sin(x)^3)",
                "-sin(3*x)",
                DeriveTrigRewriteKind::TripleAngleSin,
            ),
            (
                "-(4*cos(x)^3-3*cos(x))",
                "-cos(3*x)",
                DeriveTrigRewriteKind::TripleAngleCos,
            ),
            (
                "-(3*tan(x)-tan(x)^3)/(1-3*tan(x)^2)",
                "-tan(3*x)",
                DeriveTrigRewriteKind::TripleAngleTan,
            ),
        ];

        for (source_text, target_text, expected_kind) in contraction_cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_trig_contraction_target_aware(&mut ctx, source, target)
                .expect("rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }

        let expansion_cases = [
            (
                "-sin(3*x)",
                "-(3*sin(x)-4*sin(x)^3)",
                DeriveTrigRewriteKind::TripleAngleSin,
            ),
            (
                "-cos(3*x)",
                "-(4*cos(x)^3-3*cos(x))",
                DeriveTrigRewriteKind::TripleAngleCos,
            ),
            (
                "-tan(3*x)",
                "-(3*tan(x)-tan(x)^3)/(1-3*tan(x)^2)",
                DeriveTrigRewriteKind::TripleAngleTan,
            ),
        ];

        for (source_text, target_text, expected_kind) in expansion_cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }
    }

    #[test]
    fn expands_quintuple_angle_sine_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(5*x)", &mut ctx).expect("source");
        let target = parse("5*sin(x)-20*sin(x)^3+16*sin(x)^5", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::QuintupleAngleSin);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn expands_quintuple_angle_cosine_target_aware() {
        let mut ctx = Context::new();
        let source = parse("cos(5*x)", &mut ctx).expect("source");
        let target = parse("16*cos(x)^5-20*cos(x)^3+5*cos(x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::QuintupleAngleCos);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn contracts_quintuple_angle_sine_target_aware() {
        let mut ctx = Context::new();
        let source = parse("5*sin(x)-20*sin(x)^3+16*sin(x)^5", &mut ctx).expect("source");
        let target = parse("sin(5*x)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::QuintupleAngleSin);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn contracts_quintuple_angle_cosine_target_aware() {
        let mut ctx = Context::new();
        let source = parse("16*cos(x)^5-20*cos(x)^3+5*cos(x)", &mut ctx).expect("source");
        let target = parse("cos(5*x)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::QuintupleAngleCos);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_negative_quintuple_angle_variants_target_aware() {
        let expansion_cases = [
            (
                "-sin(5*x)",
                "-(5*sin(x)-20*sin(x)^3+16*sin(x)^5)",
                DeriveTrigRewriteKind::QuintupleAngleSin,
            ),
            (
                "-cos(5*x)",
                "-(16*cos(x)^5-20*cos(x)^3+5*cos(x))",
                DeriveTrigRewriteKind::QuintupleAngleCos,
            ),
        ];

        for (source_text, target_text, expected_kind) in expansion_cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }

        let contraction_cases = [
            (
                "-(5*sin(x)-20*sin(x)^3+16*sin(x)^5)",
                "-sin(5*x)",
                DeriveTrigRewriteKind::QuintupleAngleSin,
            ),
            (
                "-(16*cos(x)^5-20*cos(x)^3+5*cos(x))",
                "-cos(5*x)",
                DeriveTrigRewriteKind::QuintupleAngleCos,
            ),
        ];

        for (source_text, target_text, expected_kind) in contraction_cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_trig_contraction_target_aware(&mut ctx, source, target)
                .expect("rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }
    }

    #[test]
    fn expands_tabulated_scaled_trig_targets_aware() {
        let cases = [
            (
                "tan(2*a*x)",
                "sin(2*a*x)/cos(2*a*x)",
                DeriveTrigRewriteKind::TanToSinCos,
            ),
            (
                "sec(a*x)",
                "1/cos(a*x)",
                DeriveTrigRewriteKind::ExpandSecToRecipCos,
            ),
            (
                "csc(a*x)",
                "1/sin(a*x)",
                DeriveTrigRewriteKind::ExpandCscToRecipSin,
            ),
            (
                "cot(a*x)",
                "cos(a*x)/sin(a*x)",
                DeriveTrigRewriteKind::ExpandCotToCosSin,
            ),
        ];

        for (source_text, target_text, expected_kind) in cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite =
                try_rewrite_trig_expansion(&mut ctx, source, target).unwrap_or_else(|| {
                    panic!("expected scaled trig expansion for `{source_text}` -> `{target_text}`")
                });

            assert_eq!(
                rewrite.kind, expected_kind,
                "unexpected rewrite kind for `{source_text}` -> `{target_text}`"
            );
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }
    }

    #[test]
    fn rewrites_cot_quotient_expansion_target_aware() {
        let mut ctx = Context::new();
        let source = parse("cot(x)", &mut ctx).expect("source");
        let target = parse("cos(x)/sin(x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::ExpandCotToCosSin);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn recognizes_reciprocal_cosine_as_sec_target_aware() {
        let mut ctx = Context::new();
        let source = parse("1/cos(x)", &mut ctx).expect("source");
        let target = parse("sec(x)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::RecognizeRecipCosAsSec);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn recognizes_sine_over_cosine_as_tan_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(x)/cos(x)", &mut ctx).expect("source");
        let target = parse("tan(x)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::RecognizeSinOverCosAsTan
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn recognizes_sine_over_cosine_as_tan_after_linear_angle_normalization() {
        let mut ctx = Context::new();
        let source = parse("sin(2*x)/cos(x+x)", &mut ctx).expect("source");
        let target = parse("tan(2*x)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::RecognizeSinOverCosAsTan
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn recognizes_cosine_over_sine_as_cot_target_aware() {
        let mut ctx = Context::new();
        let source = parse("cos(x)/sin(x)", &mut ctx).expect("source");
        let target = parse("cot(x)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::RecognizeCosOverSinAsCot
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn recognizes_cosine_difference_over_sine_difference_as_tan_target_aware() {
        let mut ctx = Context::new();
        let source = parse("(cos(x)-cos(3*x))/(sin(3*x)-sin(x))", &mut ctx).expect("source");
        let target = parse("tan(2*x)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::RecognizeCosDiffOverSinDiffAsTan
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_reciprocal_trig_product_to_one_target_aware() {
        let mut ctx = Context::new();
        let source = parse("tan(x)*cot(x)", &mut ctx).expect("source");
        let target = parse("1", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_identity_to_one_target_aware(&mut ctx, source, target).expect("rw");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::ReciprocalProductToOne);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_reciprocal_trig_product_to_one_with_passthrough_target_aware() {
        let mut ctx = Context::new();
        let source = parse("tan(x)*cot(x)+a", &mut ctx).expect("source");
        let target = parse("1+a", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_identity_to_one_target_aware(&mut ctx, source, target).expect("rw");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::ReciprocalProductToOne);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_sec_tan_pythagorean_to_one_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sec(x)^2 - tan(x)^2", &mut ctx).expect("source");
        let target = parse("1", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_identity_to_one_target_aware(&mut ctx, source, target).expect("rw");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::SecTanPythagoreanToOne);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_sec_tan_pythagorean_to_one_with_passthrough_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sec(x)^2 - tan(x)^2 + a", &mut ctx).expect("source");
        let target = parse("1+a", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_identity_to_one_target_aware(&mut ctx, source, target).expect("rw");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::SecTanPythagoreanToOne);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_pythagorean_chain_to_one_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(x)^2 + cos(x)^2", &mut ctx).expect("source");
        let target = parse("1", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_identity_to_one_target_aware(&mut ctx, source, target).expect("rw");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::PythagoreanChainToOne);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_csc_cot_pythagorean_to_one_target_aware() {
        let mut ctx = Context::new();
        let source = parse("csc(x)^2 - cot(x)^2", &mut ctx).expect("source");
        let target = parse("1", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_identity_to_one_target_aware(&mut ctx, source, target).expect("rw");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::CscCotPythagoreanToOne);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_half_angle_tangent_expansion_target_aware() {
        let mut ctx = Context::new();
        let source = parse("tan(x)", &mut ctx).expect("source");
        let target = parse("(1-cos(2*x))/sin(2*x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::HalfAngleTangentExpandOneMinusCosOverSin
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_double_angle_expansion_after_linear_angle_normalization() {
        let mut ctx = Context::new();
        let source = parse("sin(x+x)", &mut ctx).expect("source");
        let target = parse("2*sin(x)*cos(x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::DoubleAngleSin);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_negative_reciprocal_trig_contractions_target_aware() {
        let cases = [
            (
                "-sin(x)/cos(x)",
                "-tan(x)",
                DeriveTrigRewriteKind::RecognizeSinOverCosAsTan,
            ),
            (
                "-1/cos(x)",
                "-sec(x)",
                DeriveTrigRewriteKind::RecognizeRecipCosAsSec,
            ),
            (
                "-1/sin(x)",
                "-csc(x)",
                DeriveTrigRewriteKind::RecognizeRecipSinAsCsc,
            ),
            (
                "-cos(x)/sin(x)",
                "-cot(x)",
                DeriveTrigRewriteKind::RecognizeCosOverSinAsCot,
            ),
        ];

        for (source_text, target_text, expected_kind) in cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_trig_contraction_target_aware(&mut ctx, source, target)
                .expect("rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }
    }

    #[test]
    fn rewrites_negative_reciprocal_trig_expansions_target_aware() {
        let cases = [
            (
                "-tan(x)",
                "-sin(x)/cos(x)",
                DeriveTrigRewriteKind::TanToSinCos,
            ),
            (
                "-sec(x)",
                "-1/cos(x)",
                DeriveTrigRewriteKind::ExpandSecToRecipCos,
            ),
            (
                "-csc(x)",
                "-1/sin(x)",
                DeriveTrigRewriteKind::ExpandCscToRecipSin,
            ),
            (
                "-cot(x)",
                "-cos(x)/sin(x)",
                DeriveTrigRewriteKind::ExpandCotToCosSin,
            ),
        ];

        for (source_text, target_text, expected_kind) in cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }
    }

    #[test]
    fn rewrites_half_angle_square_expansion_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(x)^2", &mut ctx).expect("source");
        let target = parse("(1-cos(2*x))/2", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::HalfAngleSinSquaredExpand
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn contracts_half_angle_square_target_aware() {
        let mut ctx = Context::new();
        let source = parse("(1+cos(2*x))/2", &mut ctx).expect("source");
        let target = parse("cos(x)^2", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::HalfAngleCosSquaredContract
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_negative_half_angle_square_variants_target_aware() {
        let expansion_cases = [
            (
                "-sin(x)^2",
                "-(1-cos(2*x))/2",
                DeriveTrigRewriteKind::HalfAngleNegSinSquaredExpand,
            ),
            (
                "-cos(x)^2",
                "-(1+cos(2*x))/2",
                DeriveTrigRewriteKind::HalfAngleNegCosSquaredExpand,
            ),
        ];

        for (source_text, target_text, expected_kind) in expansion_cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }

        let contraction_cases = [
            (
                "-(1-cos(2*x))/2",
                "-sin(x)^2",
                DeriveTrigRewriteKind::HalfAngleNegSinSquaredContract,
            ),
            (
                "-(1+cos(2*x))/2",
                "-cos(x)^2",
                DeriveTrigRewriteKind::HalfAngleNegCosSquaredContract,
            ),
        ];

        for (source_text, target_text, expected_kind) in contraction_cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_trig_contraction_target_aware(&mut ctx, source, target)
                .expect("rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }
    }

    #[test]
    fn rewrites_product_to_sum_cos_sin_even_when_mul_is_canonicalized() {
        let mut ctx = Context::new();
        let source = parse("2*cos(x)*sin(y)", &mut ctx).expect("source");
        let target = parse("sin(x+y)-sin(x-y)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_product_to_sum_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::ProductToSumCosSin);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn generates_product_to_sum_bridge_for_sine_cosine_with_linear_angles() {
        let mut ctx = Context::new();
        let source = parse("2*sin(2*x)*cos(x)", &mut ctx).expect("source");
        let bridge_target = parse("sin(2*x+x)+sin(2*x-x)", &mut ctx).expect("target");
        let rewrites = generate_trig_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::ProductToSumSinCos
                && presentational_target_match(&mut ctx, rewrite.rewritten, bridge_target)
        }));
    }

    #[test]
    fn generates_normalized_product_to_sum_bridge_for_cosine_cosine_with_linear_angles() {
        let mut ctx = Context::new();
        let source = parse("2*cos(2*x)*cos(x)", &mut ctx).expect("source");
        let bridge_target = parse("cos(3*x)+cos(x)", &mut ctx).expect("target");
        let rewrites = generate_trig_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::ProductToSumCosCos
                && presentational_target_match(&mut ctx, rewrite.rewritten, bridge_target)
        }));
    }

    #[test]
    fn generates_normalized_product_to_sum_bridge_for_sine_sine_with_linear_angles() {
        let mut ctx = Context::new();
        let source = parse("2*sin(2*x)*sin(x)", &mut ctx).expect("source");
        let bridge_target = parse("cos(x)-cos(3*x)", &mut ctx).expect("target");
        let rewrites = generate_trig_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::ProductToSumSinSin
                && presentational_target_match(&mut ctx, rewrite.rewritten, bridge_target)
        }));
    }

    #[test]
    fn generates_targeted_additive_triple_angle_bridge_for_cosine_difference_polynomial() {
        let mut ctx = Context::new();
        let source = parse("cos(x)-cos(3*x)", &mut ctx).expect("source");
        let target = parse("cos(x)-4*cos(x)^3+3*cos(x)", &mut ctx).expect("target");
        let rewrites = generate_trig_additive_term_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::TripleAngleCos
                && strong_target_match(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_combined_additive_triple_angle_bridge_for_cosine_sum_polynomial() {
        let mut ctx = Context::new();
        let source = parse("cos(x)+cos(3*x)", &mut ctx).expect("source");
        let target = parse("4*cos(x)^3-2*cos(x)", &mut ctx).expect("target");
        let rewrites = generate_trig_additive_term_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::TripleAngleCos
                && strong_target_match(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_combined_additive_triple_angle_bridge_for_cosine_difference_polynomial() {
        let mut ctx = Context::new();
        let source = parse("cos(x)-cos(3*x)", &mut ctx).expect("source");
        let target = parse("4*cos(x)-4*cos(x)^3", &mut ctx).expect("target");
        let rewrites = generate_trig_additive_term_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::TripleAngleCos
                && strong_target_match(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_combined_additive_triple_angle_bridge_for_sine_difference_mixed_polynomial() {
        let mut ctx = Context::new();
        let source = parse("sin(3*x)-sin(x)", &mut ctx).expect("source");
        let target = parse("4*cos(x)^2*sin(x)-2*sin(x)", &mut ctx).expect("target");
        let rewrites = generate_trig_additive_term_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::TripleAngleSin
                && strong_target_match(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_combined_additive_triple_angle_bridge_for_cosine_difference_mixed_polynomial() {
        let mut ctx = Context::new();
        let source = parse("cos(x)-cos(3*x)", &mut ctx).expect("source");
        let target = parse("4*sin(x)^2*cos(x)", &mut ctx).expect("target");
        let rewrites = generate_trig_additive_term_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::TripleAngleCos
                && strong_target_match(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_phase_shift_bridge_for_scaled_shifted_sine_term() {
        let mut ctx = Context::new();
        let source = parse("2*sqrt(2)*sin(x+pi/4)", &mut ctx).expect("source");
        let target = parse("2*sin(x)+2*cos(x)", &mut ctx).expect("target");
        let rewrites = generate_trig_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::PhaseShiftIdentity
                && presentational_target_match(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_additive_phase_shift_bridge_with_passthrough_term() {
        let mut ctx = Context::new();
        let source = parse("sin(x)+cos(x)+a", &mut ctx).expect("source");
        let target = parse("sqrt(2)*sin(x+pi/4)+a", &mut ctx).expect("target");
        let rewrites = generate_trig_additive_term_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::PhaseShiftIdentity
                && strong_target_match(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_scaled_additive_phase_shift_bridge_with_passthrough_term() {
        let mut ctx = Context::new();
        let source = parse("2*sin(x)+2*cos(x)+a", &mut ctx).expect("source");
        let target = parse("2*sqrt(2)*sin(x+pi/4)+a", &mut ctx).expect("target");
        let rewrites = generate_trig_additive_term_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::PhaseShiftIdentity
                && strong_target_match(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_general_phase_shift_bridge_for_shifted_sine_term() {
        let mut ctx = Context::new();
        let source = parse("5*sin(x+arctan(4/3))", &mut ctx).expect("source");
        let target = parse("3*sin(x)+4*cos(x)", &mut ctx).expect("target");
        let rewrites = generate_trig_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::PhaseShiftIdentity
                && presentational_target_match(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_phase_shift_bridge_for_shifted_sine_to_shifted_cosine_term() {
        let mut ctx = Context::new();
        let source = parse("sqrt(2)*sin(x+pi/4)", &mut ctx).expect("source");
        let target = parse("sqrt(2)*cos(x-pi/4)", &mut ctx).expect("target");
        let rewrites = generate_trig_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::PhaseShiftIdentity
                && presentational_target_match(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_general_phase_shift_bridge_for_shifted_sine_to_shifted_cosine_term() {
        let mut ctx = Context::new();
        let source = parse("5*sin(x+arctan(4/3))", &mut ctx).expect("source");
        let target = parse("5*cos(x-arctan(3/4))", &mut ctx).expect("target");
        let rewrites = generate_trig_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::PhaseShiftIdentity
                && super::phase_shift_target_match(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_phase_shift_bridge_between_shifted_terms_with_passthrough() {
        let mut ctx = Context::new();
        let source = parse("sqrt(2)*sin(x+pi/4)+a", &mut ctx).expect("source");
        let target = parse("sqrt(2)*cos(x-pi/4)+a", &mut ctx).expect("target");
        let rewrites = generate_trig_additive_term_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::PhaseShiftIdentity
                && presentational_target_match(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_general_phase_shift_bridge_between_shifted_terms_with_passthrough() {
        let mut ctx = Context::new();
        let source = parse("5*sin(x+arctan(4/3))+a", &mut ctx).expect("source");
        let target = parse("5*cos(x-arctan(3/4))+a", &mut ctx).expect("target");
        let rewrites = generate_trig_additive_term_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::PhaseShiftIdentity
                && super::phase_shift_target_match(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_general_additive_phase_shift_bridge_with_passthrough_term() {
        let mut ctx = Context::new();
        let source = parse("3*sin(x)+4*cos(x)+a", &mut ctx).expect("source");
        let target = parse("5*sin(x+arctan(4/3))+a", &mut ctx).expect("target");
        let rewrites = generate_trig_additive_term_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::PhaseShiftIdentity
                && presentational_target_match(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn general_shifted_phase_shift_with_passthrough_does_not_prefer_trig_planner() {
        let mut ctx = Context::new();
        let source = parse("5*sin(x+arctan(4/3))+a", &mut ctx).expect("source");
        let target = parse("5*cos(x-arctan(3/4))+a", &mut ctx).expect("target");

        assert!(!should_try_trig_planner_before_simplify(
            &mut ctx, source, target
        ));
    }

    #[test]
    fn generates_linear_angle_simplification_bridge_for_trig_arguments() {
        let mut ctx = Context::new();
        let source = parse("sin(2*x+x)+sin(2*x-x)", &mut ctx).expect("source");
        let target = parse("sin(3*x)+sin(x)", &mut ctx).expect("target");
        let rewrites = generate_trig_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::LinearAngleArgumentSimplify
                && presentational_target_match(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_product_to_sum_bridge_for_cosine_cosine_with_linear_angles() {
        let mut ctx = Context::new();
        let source = parse("2*cos(2*x)*cos(x)", &mut ctx).expect("source");
        let bridge_target = parse("cos(3*x)+cos(x)", &mut ctx).expect("target");
        let rewrites = generate_trig_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::ProductToSumCosCos
                && presentational_target_match(&mut ctx, rewrite.rewritten, bridge_target)
        }));
    }

    #[test]
    fn generates_linear_angle_simplification_bridge_for_cosine_arguments_with_even_parity() {
        let mut ctx = Context::new();
        let source = parse("cos(2*x+x)+cos(2*x-x)", &mut ctx).expect("source");
        let target = parse("cos(3*x)+cos(x)", &mut ctx).expect("target");
        let rewrites = generate_trig_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::LinearAngleArgumentSimplify
                && presentational_target_match(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_product_to_sum_bridge_for_sine_sine_with_linear_angles() {
        let mut ctx = Context::new();
        let source = parse("2*sin(2*x)*sin(x)", &mut ctx).expect("source");
        let bridge_target = parse("cos(x)-cos(3*x)", &mut ctx).expect("target");
        let rewrites = generate_trig_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::ProductToSumSinSin
                && presentational_target_match(&mut ctx, rewrite.rewritten, bridge_target)
        }));
    }

    #[test]
    fn generates_linear_angle_simplification_bridge_for_sine_sine_cosine_difference() {
        let mut ctx = Context::new();
        let source = parse("cos(2*x-x)-cos(2*x+x)", &mut ctx).expect("source");
        let target = parse("cos(x)-cos(3*x)", &mut ctx).expect("target");
        let rewrites = generate_trig_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::LinearAngleArgumentSimplify
                && presentational_target_match(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn does_not_try_trig_planner_preference_for_hyperbolic_difference_expansion() {
        let mut ctx = Context::new();
        let source = parse("sinh(x-y)", &mut ctx).expect("source");
        let target = parse("sinh(x)*cosh(y)-sinh(y)*cosh(x)", &mut ctx).expect("target");

        assert!(!should_try_trig_planner_before_simplify(
            &mut ctx, source, target
        ));
    }

    #[test]
    fn generates_additive_product_to_sum_bridge_with_passthrough_term() {
        let mut ctx = Context::new();
        let source = parse("2*sin(2*x)*sin(x)+a", &mut ctx).expect("source");
        let target = parse("cos(x)-cos(3*x)+a", &mut ctx).expect("target");
        let rewrites = generate_trig_additive_term_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveTrigRewriteKind::ProductToSumSinSin
                && presentational_target_match(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn does_not_try_trig_planner_preference_when_linear_angle_then_double_angle_is_direct() {
        let mut ctx = Context::new();
        let source = parse("sin(x+x)", &mut ctx).expect("source");
        let target = parse("2*sin(x)*cos(x)", &mut ctx).expect("target");

        assert!(!should_try_trig_planner_before_simplify(
            &mut ctx, source, target
        ));
    }

    #[test]
    fn does_not_try_trig_planner_preference_when_additive_bridge_then_triple_angle_is_direct() {
        let mut ctx = Context::new();
        let source = parse("2*sin(2*x)*sin(x)+a", &mut ctx).expect("source");
        let target = parse("4*cos(x)-4*cos(x)^3+a", &mut ctx).expect("target");

        assert!(!should_try_trig_planner_before_simplify(
            &mut ctx, source, target
        ));
    }

    #[test]
    fn does_not_try_trig_planner_preference_when_repeated_phase_shift_chain_is_direct() {
        let mut ctx = Context::new();
        let source = parse("sin(x)+cos(x)+sin(y)+cos(y)", &mut ctx).expect("source");
        let target = parse("sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4)", &mut ctx).expect("target");

        assert!(!should_try_trig_planner_before_simplify(
            &mut ctx, source, target
        ));
    }

    #[test]
    fn rewrites_sum_to_product_when_target_product_order_is_canonicalized() {
        let mut ctx = Context::new();
        let source = parse("sin(3*x)+sin(x)", &mut ctx).expect("source");
        let target = parse("2*sin(2*x)*cos(x)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_sum_to_product_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::SumToProductSinSum);
        assert!(presentational_target_match(
            &mut ctx,
            rewrite.rewritten,
            target
        ));
    }

    #[test]
    fn rewrites_sum_to_product_sine_sum_when_half_difference_is_negative() {
        let mut ctx = Context::new();
        let source = parse("sin(x)+sin(5*x)", &mut ctx).expect("source");
        let target = parse("2*sin(3*x)*cos(2*x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::SumToProductSinSum);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_general_sum_to_product_sine_sum_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(x)+sin(y)", &mut ctx).expect("source");
        let target = parse("2*sin((x+y)/2)*cos((x-y)/2)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::SumToProductSinSum);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_sum_to_product_cosine_sum_when_half_difference_is_negative() {
        let mut ctx = Context::new();
        let source = parse("cos(x)+cos(5*x)", &mut ctx).expect("source");
        let target = parse("2*cos(3*x)*cos(2*x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::SumToProductCosSum);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_general_sum_to_product_cosine_sum_target_aware() {
        let mut ctx = Context::new();
        let source = parse("cos(x)+cos(y)", &mut ctx).expect("source");
        let target = parse("2*cos((x+y)/2)*cos((x-y)/2)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::SumToProductCosSum);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_sum_to_product_cosine_difference_target_aware() {
        let mut ctx = Context::new();
        let source = parse("cos(5*x)-cos(x)", &mut ctx).expect("source");
        let target = parse("-2*sin(3*x)*sin(2*x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::SumToProductCosDiff);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_general_sum_to_product_cosine_difference_target_aware() {
        let mut ctx = Context::new();
        let source = parse("cos(x)-cos(y)", &mut ctx).expect("source");
        let target = parse("-2*sin((x+y)/2)*sin((x-y)/2)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::SumToProductCosDiff);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_phase_shift_sum_to_shifted_sine_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(x)+cos(x)", &mut ctx).expect("source");
        let target = parse("sqrt(2)*sin(x+pi/4)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::PhaseShiftIdentity);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_phase_shift_difference_to_shifted_sine_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(x)-cos(x)", &mut ctx).expect("source");
        let target = parse("sqrt(2)*sin(x-pi/4)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::PhaseShiftIdentity);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_phase_shift_shifted_sine_to_sum_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sqrt(2)*sin(x+pi/4)", &mut ctx).expect("source");
        let target = parse("sin(x)+cos(x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::PhaseShiftIdentity);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_phase_shift_shifted_cosine_to_sum_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sqrt(2)*cos(x-pi/4)", &mut ctx).expect("source");
        let target = parse("sin(x)+cos(x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::PhaseShiftIdentity);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_cofunction_phase_shift_targets_before_generic_simplify() {
        for (source_text, target_text) in [
            ("sin(pi/2 - x)", "cos(x)"),
            ("cos(pi/2 - x)", "sin(x)"),
            ("sin(pi/2 + x)", "cos(x)"),
            ("cos(pi/2 + x)", "-sin(x)"),
        ] {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

            assert_eq!(rewrite.kind, DeriveTrigRewriteKind::CofunctionIdentity);
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected cofunction target match for `{source_text}` -> `{target_text}`"
            );
        }
    }

    #[test]
    fn rewrites_scaled_phase_shift_sum_to_shifted_sine_target_aware() {
        let mut ctx = Context::new();
        let source = parse("2*sin(x)+2*cos(x)", &mut ctx).expect("source");
        let target = parse("2*sqrt(2)*sin(x+pi/4)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::PhaseShiftIdentity);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_symbolically_scaled_phase_shift_sum_to_shifted_sine_target_aware() {
        let mut ctx = Context::new();
        let source = parse("a*sin(x)+a*cos(x)", &mut ctx).expect("source");
        let target = parse("a*sqrt(2)*sin(x+pi/4)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::PhaseShiftIdentity);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_scaled_phase_shift_shifted_sine_to_sum_target_aware() {
        let mut ctx = Context::new();
        let source = parse("2*sqrt(2)*sin(x+pi/4)", &mut ctx).expect("source");
        let target = parse("2*sin(x)+2*cos(x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::PhaseShiftIdentity);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_exact_third_phase_shift_sum_to_shifted_sine_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(x)+sqrt(3)*cos(x)", &mut ctx).expect("source");
        let target = parse("2*sin(x+pi/3)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::PhaseShiftIdentity);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_exact_third_phase_shift_shifted_sine_to_sum_target_aware() {
        let mut ctx = Context::new();
        let source = parse("4*sin(x+pi/3)", &mut ctx).expect("source");
        let target = parse("2*sin(x)+2*sqrt(3)*cos(x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::PhaseShiftIdentity);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_general_phase_shift_sum_to_shifted_sine_target_aware() {
        let mut ctx = Context::new();
        let source = parse("3*sin(x)+4*cos(x)", &mut ctx).expect("source");
        let target = parse("5*sin(x+arctan(4/3))", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::PhaseShiftIdentity);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_general_phase_shift_shifted_sine_to_sum_target_aware() {
        let mut ctx = Context::new();
        let source = parse("5*sin(x+arctan(4/3))", &mut ctx).expect("source");
        let target = parse("3*sin(x)+4*cos(x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::PhaseShiftIdentity);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_phase_shift_shifted_sine_to_shifted_cosine_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sqrt(2)*sin(x+pi/4)", &mut ctx).expect("source");
        let target = parse("sqrt(2)*cos(x-pi/4)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::PhaseShiftIdentity);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_general_phase_shift_shifted_sine_to_shifted_cosine_target_aware() {
        let mut ctx = Context::new();
        let source = parse("5*sin(x+arctan(4/3))", &mut ctx).expect("source");
        let target = parse("5*cos(x-arctan(3/4))", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::PhaseShiftIdentity);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_tabulated_pythagorean_factor_form_targets_aware() {
        let cases = [
            ("1 - sin(x)^2", "cos(x)^2", "1 - sin²(x) = cos²(x)"),
            ("1 - cos(x)^2", "sin(x)^2", "1 - cos²(x) = sin²(x)"),
            ("sin(x)^2", "1 - cos(x)^2", "1 - cos²(x) = sin²(x)"),
            ("cos(x)^2", "1 - sin(x)^2", "1 - sin²(x) = cos²(x)"),
            ("-sin(x)^2", "cos(x)^2 - 1", "cos²(x) - 1 = -sin²(x)"),
            ("-cos(x)^2", "sin(x)^2 - 1", "sin²(x) - 1 = -cos²(x)"),
        ];

        for (source_text, target_text, expected_description) in cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_pythagorean_factor_form_target_aware(
                &mut ctx, source, target,
            )
            .unwrap_or_else(|| {
                panic!(
                    "expected pythagorean factor-form rewrite for `{source_text}` -> `{target_text}`"
                )
            });

            assert_eq!(
                rewrite.description, expected_description,
                "unexpected description for `{source_text}` -> `{target_text}`"
            );
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }
    }

    #[test]
    fn rewrites_shifted_reciprocal_pythagorean_variants_target_aware() {
        let cases = [
            (
                "sec(x)^2 - 1",
                "tan(x)^2",
                DeriveTrigRewriteKind::RecognizeSecSquaredMinusOneAsTanSquared,
            ),
            (
                "tan(x)^2",
                "sec(x)^2 - 1",
                DeriveTrigRewriteKind::ExpandTanSquaredToSecSquaredMinusOne,
            ),
            (
                "csc(x)^2 - 1",
                "cot(x)^2",
                DeriveTrigRewriteKind::RecognizeCscSquaredMinusOneAsCotSquared,
            ),
            (
                "cot(x)^2",
                "csc(x)^2 - 1",
                DeriveTrigRewriteKind::ExpandCotSquaredToCscSquaredMinusOne,
            ),
        ];

        for (source_text, target_text, expected_kind) in cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite =
                try_rewrite_shifted_reciprocal_pythagorean_target_aware(&mut ctx, source, target)
                    .expect("rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }
    }

    #[test]
    fn rewrites_negative_shifted_reciprocal_pythagorean_variants_target_aware() {
        let cases = [
            (
                "1 - sec(x)^2",
                "-tan(x)^2",
                DeriveTrigRewriteKind::RecognizeOneMinusSecSquaredAsNegTanSquared,
            ),
            (
                "-tan(x)^2",
                "1 - sec(x)^2",
                DeriveTrigRewriteKind::ExpandNegTanSquaredToOneMinusSecSquared,
            ),
            (
                "1 - csc(x)^2",
                "-cot(x)^2",
                DeriveTrigRewriteKind::RecognizeOneMinusCscSquaredAsNegCotSquared,
            ),
            (
                "-cot(x)^2",
                "1 - csc(x)^2",
                DeriveTrigRewriteKind::ExpandNegCotSquaredToOneMinusCscSquared,
            ),
        ];

        for (source_text, target_text, expected_kind) in cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite =
                try_rewrite_shifted_reciprocal_pythagorean_target_aware(&mut ctx, source, target)
                    .expect("rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }
    }

    #[test]
    fn rewrites_shifted_double_angle_to_two_cos_sq_target_aware() {
        let mut ctx = Context::new();
        let source = parse("cos(2*x) + 1", &mut ctx).expect("source");
        let target = parse("2*cos(x)^2", &mut ctx).expect("target");
        let rewrite = try_rewrite_shifted_double_angle_target_aware(&mut ctx, source, target)
            .expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::DoubleAngleCosOnePlusCosToTwoCosSq
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_shifted_double_angle_from_two_sin_sq_target_aware() {
        for (source_text, target_text, expected_kind) in [
            (
                "2*sin(x)^2",
                "1 - cos(2*x)",
                DeriveTrigRewriteKind::DoubleAngleCosTwoSinSqToOneMinusCos,
            ),
            (
                "2*sin(x/2)^2",
                "1 - cos(x)",
                DeriveTrigRewriteKind::DoubleAngleCosTwoSinSqToOneMinusCos,
            ),
            (
                "2*cos(x/2)^2",
                "1 + cos(x)",
                DeriveTrigRewriteKind::DoubleAngleCosTwoCosSqToOnePlusCos,
            ),
        ] {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_shifted_double_angle_target_aware(&mut ctx, source, target)
                .expect("rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }
    }

    #[test]
    fn rewrites_negative_shifted_double_angle_variants_target_aware() {
        let cases = [
            (
                "cos(2*x) - 1",
                "-2*sin(x)^2",
                DeriveTrigRewriteKind::DoubleAngleCosMinusOneToNegTwoSinSq,
            ),
            (
                "-2*sin(x)^2",
                "cos(2*x) - 1",
                DeriveTrigRewriteKind::DoubleAngleCosNegTwoSinSqToCosMinusOne,
            ),
        ];

        for (source_text, target_text, expected_kind) in cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_shifted_double_angle_target_aware(&mut ctx, source, target)
                .expect("rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }
    }

    #[test]
    fn rewrites_negative_cosine_double_angle_variants_target_aware() {
        let contraction_cases = [
            (
                "sin(x)^2 - cos(x)^2",
                "-cos(2*x)",
                DeriveTrigRewriteKind::DoubleAngleNegCosContract,
            ),
            (
                "2*sin(x)^2 - 1",
                "-cos(2*x)",
                DeriveTrigRewriteKind::DoubleAngleNegCosContract,
            ),
        ];

        for (source_text, target_text, expected_kind) in contraction_cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_trig_contraction_target_aware(&mut ctx, source, target)
                .unwrap_or_else(|| panic!("rewrite for `{source_text}` -> `{target_text}`"));

            assert_eq!(rewrite.kind, expected_kind);
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }

        let expansion_cases = [
            (
                "-cos(2*x)",
                "sin(x)^2 - cos(x)^2",
                DeriveTrigRewriteKind::DoubleAngleNegCosExpand,
            ),
            (
                "-cos(2*x)",
                "2*sin(x)^2 - 1",
                DeriveTrigRewriteKind::DoubleAngleNegCosExpand,
            ),
        ];

        for (source_text, target_text, expected_kind) in expansion_cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }
    }

    #[test]
    fn contracts_negative_angle_sum_diff_variants_target_aware() {
        let cases = [
            (
                "-(sin(x)*cos(y)-cos(x)*sin(y))",
                "-sin(x-y)",
                DeriveTrigRewriteKind::AngleSumDiff,
            ),
            (
                "-(cos(5*x)*cos(x)-sin(5*x)*sin(x))",
                "-cos(6*x)",
                DeriveTrigRewriteKind::AngleSumDiff,
            ),
        ];

        for (source_text, target_text, expected_kind) in cases {
            let mut ctx = Context::new();
            let source = parse(source_text, &mut ctx).expect("source");
            let target = parse(target_text, &mut ctx).expect("target");
            let rewrite = try_rewrite_trig_contraction_target_aware(&mut ctx, source, target)
                .expect("rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert!(
                strong_target_match(&mut ctx, rewrite.rewritten, target),
                "expected strong target match for `{source_text}` -> `{target_text}`"
            );
        }
    }

    #[test]
    fn contracts_negative_sine_double_angle_target_aware() {
        let mut ctx = Context::new();
        let source = parse("-2*sin(x)*cos(x)", &mut ctx).expect("source");
        let target = parse("-sin(2*x)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::DoubleAngleNegSinContract
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn contracts_mixed_sine_double_angle_product_target_aware() {
        let mut ctx = Context::new();
        let source = parse("4*sin(x)^2*cos(x)", &mut ctx).expect("source");
        let target = parse("2*sin(2*x)*sin(x)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::DoubleAngleSin);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn contracts_half_scaled_sine_double_angle_product_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(x)*cos(x)", &mut ctx).expect("source");
        let target = parse("sin(2*x)/2", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::DoubleAngleSin);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn contracts_mixed_cosine_double_angle_product_target_aware() {
        let mut ctx = Context::new();
        let source = parse("4*cos(x)^2*sin(x)", &mut ctx).expect("source");
        let target = parse("2*sin(2*x)*cos(x)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::DoubleAngleSin);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn expands_negative_sine_double_angle_target_aware() {
        let mut ctx = Context::new();
        let source = parse("-sin(2*x)", &mut ctx).expect("source");
        let target = parse("-2*sin(x)*cos(x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::DoubleAngleNegSinExpand);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn expands_mixed_sine_double_angle_product_target_aware() {
        let mut ctx = Context::new();
        let source = parse("2*sin(2*x)*sin(x)", &mut ctx).expect("source");
        let target = parse("4*sin(x)^2*cos(x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::DoubleAngleSin);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn expands_mixed_cosine_double_angle_product_target_aware() {
        let mut ctx = Context::new();
        let source = parse("2*sin(2*x)*cos(x)", &mut ctx).expect("source");
        let target = parse("4*cos(x)^2*sin(x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::DoubleAngleSin);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_sec_squared_expansion_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sec(x)^2", &mut ctx).expect("source");
        let target = parse("1 + tan(x)^2", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::ExpandSecSquared);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_sine_fourth_power_reduction_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(x)^4", &mut ctx).expect("source");
        let target = parse("(3-4*cos(2*x)+cos(4*x))/8", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::PowerReductionSinFourth);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_cosine_fourth_power_reduction_target_aware() {
        let mut ctx = Context::new();
        let source = parse("cos(x)^4", &mut ctx).expect("source");
        let target = parse("(3+4*cos(2*x)+cos(4*x))/8", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::PowerReductionCosFourth);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_sine_cosine_square_product_reduction_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(x)^2*cos(x)^2", &mut ctx).expect("source");
        let target = parse("(1-cos(4*x))/8", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::PowerReductionSinCosSquares
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_sine_sixth_power_reduction_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(x)^6", &mut ctx).expect("source");
        let target = parse("(10-15*cos(2*x)+6*cos(4*x)-cos(6*x))/32", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::PowerReductionSinSixth);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_cosine_sixth_power_reduction_target_aware() {
        let mut ctx = Context::new();
        let source = parse("cos(x)^6", &mut ctx).expect("source");
        let target = parse("(10+15*cos(2*x)+6*cos(4*x)+cos(6*x))/32", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::PowerReductionCosSixth);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_sine_eighth_power_reduction_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(x)^8", &mut ctx).expect("source");
        let target = parse(
            "(35-56*cos(2*x)+28*cos(4*x)-8*cos(6*x)+cos(8*x))/128",
            &mut ctx,
        )
        .expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::PowerReductionSinEighth);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_cosine_eighth_power_reduction_target_aware() {
        let mut ctx = Context::new();
        let source = parse("cos(x)^8", &mut ctx).expect("source");
        let target = parse(
            "(35+56*cos(2*x)+28*cos(4*x)+8*cos(6*x)+cos(8*x))/128",
            &mut ctx,
        )
        .expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::PowerReductionCosEighth);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_sine_tenth_power_reduction_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(x)^10", &mut ctx).expect("source");
        let target = parse(
            "(126-210*cos(2*x)+120*cos(4*x)-45*cos(6*x)+10*cos(8*x)-cos(10*x))/512",
            &mut ctx,
        )
        .expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::PowerReductionSinTenth);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_cosine_tenth_power_reduction_target_aware() {
        let mut ctx = Context::new();
        let source = parse("cos(x)^10", &mut ctx).expect("source");
        let target = parse(
            "(126+210*cos(2*x)+120*cos(4*x)+45*cos(6*x)+10*cos(8*x)+cos(10*x))/512",
            &mut ctx,
        )
        .expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::PowerReductionCosTenth);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_sine_twelfth_power_reduction_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(x)^12", &mut ctx).expect("source");
        let target = parse(
            "(462-792*cos(2*x)+495*cos(4*x)-220*cos(6*x)+66*cos(8*x)-12*cos(10*x)+cos(12*x))/2048",
            &mut ctx,
        )
        .expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::PowerReductionSinTwelfth
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_cosine_twelfth_power_reduction_target_aware() {
        let mut ctx = Context::new();
        let source = parse("cos(x)^12", &mut ctx).expect("source");
        let target = parse(
            "(462+792*cos(2*x)+495*cos(4*x)+220*cos(6*x)+66*cos(8*x)+12*cos(10*x)+cos(12*x))/2048",
            &mut ctx,
        )
        .expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::PowerReductionCosTwelfth
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_sine_fourteenth_power_reduction_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(x)^14", &mut ctx).expect("source");
        let target = parse(
            "(1716-3003*cos(2*x)+2002*cos(4*x)-1001*cos(6*x)+364*cos(8*x)-91*cos(10*x)+14*cos(12*x)-cos(14*x))/8192",
            &mut ctx,
        )
        .expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::PowerReductionSinFourteenth
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_cosine_fourteenth_power_reduction_target_aware() {
        let mut ctx = Context::new();
        let source = parse("cos(x)^14", &mut ctx).expect("source");
        let target = parse(
            "(1716+3003*cos(2*x)+2002*cos(4*x)+1001*cos(6*x)+364*cos(8*x)+91*cos(10*x)+14*cos(12*x)+cos(14*x))/8192",
            &mut ctx,
        )
        .expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::PowerReductionCosFourteenth
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_sine_sixteenth_power_reduction_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(x)^16", &mut ctx).expect("source");
        let target = parse(
            "(6435-11440*cos(2*x)+8008*cos(4*x)-4368*cos(6*x)+1820*cos(8*x)-560*cos(10*x)+120*cos(12*x)-16*cos(14*x)+cos(16*x))/32768",
            &mut ctx,
        )
        .expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::PowerReductionSinSixteenth
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_cosine_sixteenth_power_reduction_target_aware() {
        let mut ctx = Context::new();
        let source = parse("cos(x)^16", &mut ctx).expect("source");
        let target = parse(
            "(6435+11440*cos(2*x)+8008*cos(4*x)+4368*cos(6*x)+1820*cos(8*x)+560*cos(10*x)+120*cos(12*x)+16*cos(14*x)+cos(16*x))/32768",
            &mut ctx,
        )
        .expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::PowerReductionCosSixteenth
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_sine_eighteenth_power_reduction_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(x)^18", &mut ctx).expect("source");
        let target = parse(
            "(24310-43758*cos(2*x)+31824*cos(4*x)-18564*cos(6*x)+8568*cos(8*x)-3060*cos(10*x)+816*cos(12*x)-153*cos(14*x)+18*cos(16*x)-cos(18*x))/131072",
            &mut ctx,
        )
        .expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::PowerReductionSinEighteenth
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_cosine_eighteenth_power_reduction_target_aware() {
        let mut ctx = Context::new();
        let source = parse("cos(x)^18", &mut ctx).expect("source");
        let target = parse(
            "(24310+43758*cos(2*x)+31824*cos(4*x)+18564*cos(6*x)+8568*cos(8*x)+3060*cos(10*x)+816*cos(12*x)+153*cos(14*x)+18*cos(16*x)+cos(18*x))/131072",
            &mut ctx,
        )
        .expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::PowerReductionCosEighteenth
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_sine_twentieth_power_reduction_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(x)^20", &mut ctx).expect("source");
        let target = parse(
            "(92378-167960*cos(2*x)+125970*cos(4*x)-77520*cos(6*x)+38760*cos(8*x)-15504*cos(10*x)+4845*cos(12*x)-1140*cos(14*x)+190*cos(16*x)-20*cos(18*x)+cos(20*x))/524288",
            &mut ctx,
        )
        .expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::PowerReductionSinTwentieth
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_cosine_twentieth_power_reduction_target_aware() {
        let mut ctx = Context::new();
        let source = parse("cos(x)^20", &mut ctx).expect("source");
        let target = parse(
            "(92378+167960*cos(2*x)+125970*cos(4*x)+77520*cos(6*x)+38760*cos(8*x)+15504*cos(10*x)+4845*cos(12*x)+1140*cos(14*x)+190*cos(16*x)+20*cos(18*x)+cos(20*x))/524288",
            &mut ctx,
        )
        .expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::PowerReductionCosTwentieth
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_sine_twenty_second_power_reduction_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(x)^22", &mut ctx).expect("source");
        let target = parse(
            "(352716-646646*cos(2*x)+497420*cos(4*x)-319770*cos(6*x)+170544*cos(8*x)-74613*cos(10*x)+26334*cos(12*x)-7315*cos(14*x)+1540*cos(16*x)-231*cos(18*x)+22*cos(20*x)-cos(22*x))/2097152",
            &mut ctx,
        )
        .expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::PowerReductionSinTwentySecond
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_cosine_twenty_second_power_reduction_target_aware() {
        let mut ctx = Context::new();
        let source = parse("cos(x)^22", &mut ctx).expect("source");
        let target = parse(
            "(352716+646646*cos(2*x)+497420*cos(4*x)+319770*cos(6*x)+170544*cos(8*x)+74613*cos(10*x)+26334*cos(12*x)+7315*cos(14*x)+1540*cos(16*x)+231*cos(18*x)+22*cos(20*x)+cos(22*x))/2097152",
            &mut ctx,
        )
        .expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::PowerReductionCosTwentySecond
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_sine_twenty_fourth_power_reduction_target_aware() {
        let mut ctx = Context::new();
        let source = parse("sin(x)^24", &mut ctx).expect("source");
        let target = parse(
            "(1352078-2496144*cos(2*x)+1961256*cos(4*x)-1307504*cos(6*x)+735471*cos(8*x)-346104*cos(10*x)+134596*cos(12*x)-42504*cos(14*x)+10626*cos(16*x)-2024*cos(18*x)+276*cos(20*x)-24*cos(22*x)+cos(24*x))/8388608",
            &mut ctx,
        )
        .expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::PowerReductionSinHigherEven
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_cosine_twenty_fourth_power_reduction_target_aware() {
        let mut ctx = Context::new();
        let source = parse("cos(x)^24", &mut ctx).expect("source");
        let target = parse(
            "(1352078+2496144*cos(2*x)+1961256*cos(4*x)+1307504*cos(6*x)+735471*cos(8*x)+346104*cos(10*x)+134596*cos(12*x)+42504*cos(14*x)+10626*cos(16*x)+2024*cos(18*x)+276*cos(20*x)+24*cos(22*x)+cos(24*x))/8388608",
            &mut ctx,
        )
        .expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::PowerReductionCosHigherEven
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_sine_plus_cosine_square_target_aware() {
        let mut ctx = Context::new();
        let source = parse("(sin(x)+cos(x))^2", &mut ctx).expect("source");
        let target = parse("1+sin(2*x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::SinCosSquareSum);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_sine_minus_cosine_square_target_aware() {
        let mut ctx = Context::new();
        let source = parse("(sin(x)-cos(x))^2", &mut ctx).expect("source");
        let target = parse("1-sin(2*x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::SinCosSquareDiff);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_cosine_minus_sine_square_target_aware() {
        let mut ctx = Context::new();
        let source = parse("(cos(x)-sin(x))^2", &mut ctx).expect("source");
        let target = parse("1-sin(2*x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::SinCosSquareDiff);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn recognizes_sec_squared_target_aware() {
        let mut ctx = Context::new();
        let source = parse("1 + tan(x)^2", &mut ctx).expect("source");
        let target = parse("sec(x)^2", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::RecognizeSecSquared);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }
}
