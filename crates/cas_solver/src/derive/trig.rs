use super::{presentational_target_match, strong_target_match};
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Expr, ExprId};
use cas_math::expr_nary::{add_terms_signed, Sign};
use cas_math::expr_rewrite::smart_mul;
use cas_math::trig_roots_flatten::flatten_mul_chain;
use num_rational::BigRational;
use num_traits::Signed;
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeriveTrigRewriteKind {
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
    RecursiveAngleSumDiff,
    HalfAngleTangent,
    HalfAngleTangentExpandOneMinusCosOverSin,
    HalfAngleTangentExpandSinOverOnePlusCos,
    HalfAngleSinSquaredExpand,
    HalfAngleCosSquaredExpand,
    HalfAngleSinSquaredContract,
    HalfAngleCosSquaredContract,
    HalfAngleNegSinSquaredExpand,
    HalfAngleNegCosSquaredExpand,
    HalfAngleNegSinSquaredContract,
    HalfAngleNegCosSquaredContract,
    DoubleAngleSin,
    DoubleAngleCos,
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

    for (index, (term, sign)) in terms.iter().enumerate() {
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

    let rewrites = generate_trig_bridge_rewrites(ctx, expr);
    if rewrites.is_empty() {
        return false;
    }

    for rewrite in &rewrites {
        if rewrite.kind == DeriveTrigRewriteKind::LinearAngleArgumentSimplify
            && (try_rewrite_trig_expansion(ctx, rewrite.rewritten, target_expr).is_some()
                || try_rewrite_trig_contraction_target_aware(ctx, rewrite.rewritten, target_expr)
                    .is_some())
        {
            return false;
        }
    }

    true
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
            Self::HalfAngleSinSquaredExpand => "Expand sin²(u) as (1 - cos(2u))/2",
            Self::HalfAngleCosSquaredExpand => "Expand cos²(u) as (1 + cos(2u))/2",
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
            Self::LinearAngleArgumentSimplify => {
                "Simplify linear angle arguments inside trig functions"
            }
        }
    }

    pub(crate) fn rule_name(self) -> &'static str {
        match self {
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
            Self::HalfAngleTangent
            | Self::HalfAngleTangentExpandOneMinusCosOverSin
            | Self::HalfAngleTangentExpandSinOverOnePlusCos => "Half-Angle Tangent Identity",
            Self::HalfAngleSinSquaredExpand
            | Self::HalfAngleCosSquaredExpand
            | Self::HalfAngleSinSquaredContract
            | Self::HalfAngleCosSquaredContract
            | Self::HalfAngleNegSinSquaredExpand
            | Self::HalfAngleNegCosSquaredExpand
            | Self::HalfAngleNegSinSquaredContract
            | Self::HalfAngleNegCosSquaredContract => "Half-Angle Square Identity",
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
            Self::LinearAngleArgumentSimplify => "Linear Angle Simplification",
        }
    }
}

pub(crate) fn try_rewrite_trig_expansion(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
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
        try_rewrite_negated_reciprocal_trig_expansion_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_reciprocal_trig_expansion_target_aware(ctx, expr, target_expr)
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
        try_rewrite_half_angle_square_expansion_target_aware(ctx, expr, target_expr)
    {
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
    let double_arg = smart_mul(ctx, two, arg);
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
    let two = ctx.num(2);
    let one = ctx.num(1);
    let double_arg = smart_mul(ctx, two, arg);
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
    let two = ctx.num(2);
    let one = ctx.num(1);
    let double_arg = smart_mul(ctx, two, arg);
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
            let kind = match rewrite.kind {
                cas_math::trig_contraction_support::TrigContractionRewriteKind::DoubleAngleSin => {
                    DeriveTrigRewriteKind::DoubleAngleSin
                }
                cas_math::trig_contraction_support::TrigContractionRewriteKind::DoubleAngleCos => {
                    DeriveTrigRewriteKind::DoubleAngleCos
                }
                _ => return None,
            };
            return Some(DeriveTrigRewrite {
                rewritten: rewrite.rewritten,
                kind,
            });
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
            let kind = match rewrite.kind {
                cas_math::trig_contraction_support::TrigContractionRewriteKind::Cos2xAdditiveSin => {
                    DeriveTrigRewriteKind::DoubleAngleCosOneMinusTwoSinSq
                }
                cas_math::trig_contraction_support::TrigContractionRewriteKind::Cos2xAdditiveCos => {
                    DeriveTrigRewriteKind::DoubleAngleCosTwoCosSqMinusOne
                }
                _ => return None,
            };
            return Some(DeriveTrigRewrite {
                rewritten: rewrite.rewritten,
                kind,
            });
        }
    }

    if let Some(rewrite) =
        try_rewrite_negated_cos_double_angle_contraction_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
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

pub(crate) fn try_rewrite_trig_identity_to_one_target_aware(
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
        let double_arg = smart_mul(ctx, two, arg);
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
        let double_arg = smart_mul(ctx, two, arg);
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
        let double_arg = smart_mul(ctx, two, arg);
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

    if let Expr::Sub(left, right) = ctx.get(expr) {
        let (left, right) = (*left, *right);

        if let (Some((BuiltinFn::Sin, sin_arg)), Some((BuiltinFn::Cos, cos_arg))) =
            (trig_square(ctx, left), trig_square(ctx, right))
        {
            if strong_target_match(ctx, sin_arg, cos_arg) {
                let double_arg = smart_mul(ctx, two, sin_arg);
                let cos_double = ctx.call_builtin(BuiltinFn::Cos, vec![double_arg]);
                let candidate = ctx.add(Expr::Neg(cos_double));
                if strong_target_match(ctx, candidate, target_expr) {
                    return Some(DeriveTrigRewrite {
                        rewritten: candidate,
                        kind: DeriveTrigRewriteKind::DoubleAngleNegCosContract,
                    });
                }
            }
        }

        if let Some((BuiltinFn::Sin, arg)) = scaled_sin_or_cos_square(ctx, left, 2) {
            if is_small_integer(ctx, right, 1) {
                let double_arg = smart_mul(ctx, two, arg);
                let cos_double = ctx.call_builtin(BuiltinFn::Cos, vec![double_arg]);
                let candidate = ctx.add(Expr::Neg(cos_double));
                if strong_target_match(ctx, candidate, target_expr) {
                    return Some(DeriveTrigRewrite {
                        rewritten: candidate,
                        kind: DeriveTrigRewriteKind::DoubleAngleNegCosContract,
                    });
                }
            }
        }

        if is_small_integer(ctx, left, 1) {
            if let Some((BuiltinFn::Cos, arg)) = scaled_sin_or_cos_square(ctx, right, 2) {
                let double_arg = smart_mul(ctx, two, arg);
                let cos_double = ctx.call_builtin(BuiltinFn::Cos, vec![double_arg]);
                let candidate = ctx.add(Expr::Neg(cos_double));
                if strong_target_match(ctx, candidate, target_expr) {
                    return Some(DeriveTrigRewrite {
                        rewritten: candidate,
                        kind: DeriveTrigRewriteKind::DoubleAngleNegCosContract,
                    });
                }
            }
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
    let rewrite =
        cas_math::trig_sum_product_support::try_rewrite_sum_to_product_contraction_expr(ctx, expr)?;
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

    Some(DeriveTrigRewrite { rewritten, kind })
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
    if let Some(rewrite) =
        cas_math::trig_core_identity_support::try_rewrite_trig_odd_even_parity_expr(ctx, expr)
    {
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
        DeriveTrigRewriteKind,
    };
    use crate::derive::{presentational_target_match, strong_target_match};
    use cas_ast::Context;
    use cas_parser::parse;

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
                .expect("rewrite");

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
                .expect("rewrite");

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
    fn does_not_try_trig_planner_preference_when_linear_angle_then_double_angle_is_direct() {
        let mut ctx = Context::new();
        let source = parse("sin(x+x)", &mut ctx).expect("source");
        let target = parse("2*sin(x)*cos(x)", &mut ctx).expect("target");

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
    fn rewrites_sum_to_product_cosine_sum_when_half_difference_is_negative() {
        let mut ctx = Context::new();
        let source = parse("cos(x)+cos(5*x)", &mut ctx).expect("source");
        let target = parse("2*cos(3*x)*cos(2*x)", &mut ctx).expect("target");
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
        let mut ctx = Context::new();
        let source = parse("2*sin(x)^2", &mut ctx).expect("source");
        let target = parse("1 - cos(2*x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_shifted_double_angle_target_aware(&mut ctx, source, target)
            .expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::DoubleAngleCosTwoSinSqToOneMinusCos
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
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
                .expect("rewrite");

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
