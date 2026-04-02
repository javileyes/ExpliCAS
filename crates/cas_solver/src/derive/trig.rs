use super::strong_target_match;
use cas_ast::{BuiltinFn, Expr, ExprId};
use cas_math::expr_rewrite::smart_mul;
use cas_math::trig_roots_flatten::flatten_mul_chain;
use num_rational::BigRational;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeriveTrigRewriteKind {
    TanToSinCos,
    ExpandSecToRecipCos,
    ExpandCscToRecipSin,
    ExpandCotToCosSin,
    RecognizeRecipCosAsSec,
    RecognizeRecipSinAsCsc,
    RecognizeCosOverSinAsCot,
    ReciprocalProductToOne,
    SecTanPythagoreanToOne,
    CscCotPythagoreanToOne,
    HalfAngleTangent,
    HalfAngleTangentExpandOneMinusCosOverSin,
    HalfAngleTangentExpandSinOverOnePlusCos,
    HalfAngleSinSquaredExpand,
    HalfAngleCosSquaredExpand,
    HalfAngleSinSquaredContract,
    HalfAngleCosSquaredContract,
    DoubleAngleSin,
    DoubleAngleCos,
    DoubleAngleCosOneMinusTwoSinSq,
    DoubleAngleCosTwoCosSqMinusOne,
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

impl DeriveTrigRewriteKind {
    pub(crate) fn description(self) -> &'static str {
        match self {
            Self::TanToSinCos => "Expand tangent to sine over cosine",
            Self::ExpandSecToRecipCos => "Expand sec(u) as 1 / cos(u)",
            Self::ExpandCscToRecipSin => "Expand csc(u) as 1 / sin(u)",
            Self::ExpandCotToCosSin => "Expand cot(u) as cos(u) / sin(u)",
            Self::RecognizeRecipCosAsSec => "Recognize 1 / cos(u) as sec(u)",
            Self::RecognizeRecipSinAsCsc => "Recognize 1 / sin(u) as csc(u)",
            Self::RecognizeCosOverSinAsCot => "Recognize cos(u) / sin(u) as cot(u)",
            Self::ReciprocalProductToOne => "Recognize tan(u) · cot(u) = 1",
            Self::SecTanPythagoreanToOne => "Recognize sec²(u) - tan²(u) = 1",
            Self::CscCotPythagoreanToOne => "Recognize csc²(u) - cot²(u) = 1",
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
            Self::DoubleAngleSin => "Expand double-angle sine",
            Self::DoubleAngleCos => "Expand double-angle cosine",
            Self::DoubleAngleCosOneMinusTwoSinSq => "Expand cosine double-angle as 1 - 2·sin(u)^2",
            Self::DoubleAngleCosTwoCosSqMinusOne => "Expand cosine double-angle as 2·cos(u)^2 - 1",
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
        }
    }

    pub(crate) fn rule_name(self) -> &'static str {
        match self {
            Self::TanToSinCos => "Trig Expansion",
            Self::ExpandSecToRecipCos
            | Self::ExpandCscToRecipSin
            | Self::ExpandCotToCosSin
            | Self::RecognizeRecipCosAsSec
            | Self::RecognizeRecipSinAsCsc
            | Self::RecognizeCosOverSinAsCot => "Reciprocal Trig Identity",
            Self::ReciprocalProductToOne => "Reciprocal Product Identity",
            Self::SecTanPythagoreanToOne | Self::CscCotPythagoreanToOne => {
                "Reciprocal Pythagorean Identity"
            }
            Self::HalfAngleTangent
            | Self::HalfAngleTangentExpandOneMinusCosOverSin
            | Self::HalfAngleTangentExpandSinOverOnePlusCos => "Half-Angle Tangent Identity",
            Self::HalfAngleSinSquaredExpand
            | Self::HalfAngleCosSquaredExpand
            | Self::HalfAngleSinSquaredContract
            | Self::HalfAngleCosSquaredContract => "Half-Angle Square Identity",
            Self::DoubleAngleSin => "Double Angle Expansion",
            Self::DoubleAngleCos
            | Self::DoubleAngleCosOneMinusTwoSinSq
            | Self::DoubleAngleCosTwoCosSqMinusOne => "Double Angle Expansion",
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
        }
    }
}

pub(crate) fn try_rewrite_trig_expansion(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    if let Some(rewrite) =
        try_rewrite_reciprocal_trig_expansion_target_aware(ctx, expr, target_expr)
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
        try_rewrite_sec_csc_squared_expansion_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_rewrite_product_to_sum_target_aware(ctx, expr, target_expr) {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_rewrite_sum_to_product_target_aware(ctx, expr, target_expr) {
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

    if let Some(rewrite) = try_rewrite_cos_double_angle_target_aware(ctx, expr, target_expr) {
        return Some(rewrite);
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
        if strong_target_match(ctx, rewrite.rewritten, target_expr) {
            let kind = match rewrite.kind {
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
            };

            return Some(DeriveTrigRewrite {
                rewritten: rewrite.rewritten,
                kind,
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
    if strong_target_match(ctx, cos_sin.rewritten, target_expr) {
        return Some(cos_sin);
    }

    let sin_cos = build_product_to_sum_candidate(
        ctx,
        sin_arg,
        cos_arg,
        &remaining,
        DeriveTrigRewriteKind::ProductToSumSinCos,
    );
    if strong_target_match(ctx, sin_cos.rewritten, target_expr) {
        return Some(sin_cos);
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

    None
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

fn try_rewrite_reciprocal_trig_contraction_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveTrigRewrite> {
    let rewrite =
        cas_math::trig_canonicalization_support::try_rewrite_trig_quotient_div_expr(ctx, expr)?;

    if !strong_target_match(ctx, rewrite.rewritten, target_expr) {
        return None;
    }

    let kind = match rewrite.kind {
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
    if strong_target_match(ctx, rewritten, target_expr) {
        return Some(rewritten);
    }

    let normalized = normalize_trig_negative_parity_tree(ctx, rewritten);
    if strong_target_match(ctx, normalized, target_expr) {
        return Some(normalized);
    }

    if matches!(
        kind,
        cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinDiff
            | cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::CosDiff
    ) {
        let negated = ctx.add(Expr::Neg(normalized));
        if strong_target_match(ctx, negated, target_expr) {
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

fn is_small_integer(ctx: &cas_ast::Context, expr: ExprId, value: i64) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_integer() && n.to_integer() == value.into())
}

fn is_one_half(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if *n == BigRational::new(1.into(), 2.into()))
}

#[cfg(test)]
mod tests {
    use super::{
        try_rewrite_product_to_sum_target_aware, try_rewrite_pythagorean_factor_form_target_aware,
        try_rewrite_trig_contraction_target_aware, try_rewrite_trig_expansion,
        try_rewrite_trig_identity_to_one_target_aware, DeriveTrigRewriteKind,
    };
    use crate::derive::strong_target_match;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn contracts_half_angle_tangent_target_aware() {
        let mut ctx = Context::new();
        let source = parse("(1-cos(2*x))/sin(2*x)", &mut ctx).expect("source");
        let target = parse("tan(x)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_trig_contraction_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.kind, DeriveTrigRewriteKind::HalfAngleTangent);
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
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
    fn rewrites_half_angle_tangent_alt_expansion_target_aware() {
        let mut ctx = Context::new();
        let source = parse("tan(x)", &mut ctx).expect("source");
        let target = parse("sin(2*x)/(1+cos(2*x))", &mut ctx).expect("target");
        let rewrite = try_rewrite_trig_expansion(&mut ctx, source, target).expect("rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveTrigRewriteKind::HalfAngleTangentExpandSinOverOnePlusCos
        );
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
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
    fn rewrites_pythagorean_factor_form_target_aware_to_complementary_square() {
        let mut ctx = Context::new();
        let source = parse("1 - sin(x)^2", &mut ctx).expect("source");
        let target = parse("cos(x)^2", &mut ctx).expect("target");
        let rewrite = try_rewrite_pythagorean_factor_form_target_aware(&mut ctx, source, target)
            .expect("rewrite");

        assert_eq!(rewrite.description, "1 - sin²(x) = cos²(x)");
        assert!(strong_target_match(&mut ctx, rewrite.rewritten, target));
    }

    #[test]
    fn rewrites_pythagorean_factor_form_target_aware_from_square_to_complement() {
        let mut ctx = Context::new();
        let source = parse("sin(x)^2", &mut ctx).expect("source");
        let target = parse("1 - cos(x)^2", &mut ctx).expect("target");
        let rewrite = try_rewrite_pythagorean_factor_form_target_aware(&mut ctx, source, target)
            .expect("rewrite");

        assert_eq!(rewrite.description, "1 - cos²(x) = sin²(x)");
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
