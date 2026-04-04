use super::{presentational_target_match, strong_target_match};
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Expr, ExprId};
use cas_math::expr_destructure::as_div;
use cas_math::expr_nary::{add_terms_signed, Sign};
use cas_math::expr_relations::is_negation;
use cas_math::expr_rewrite::smart_mul;
use cas_math::trig_roots_flatten::flatten_mul_chain;
use num_traits::{One, Signed};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeriveHyperbolicRewriteKind {
    PythagoreanOne,
    PythagoreanNegativeOne,
    TanhPythagorean,
    ExpandTanhPythagorean,
    RecognizeCoshSquaredMinusOneAsSinhSquared,
    ExpandSinhSquaredToCoshSquaredMinusOne,
    RecognizeOnePlusSinhSquaredAsCoshSquared,
    ExpandCoshSquaredToOnePlusSinhSquared,
    SinhCoshToTanh,
    TanhToSinhCosh,
    SinhCoshToExp,
    CoshMinusSinhToExpNeg,
    SinhMinusCoshToNegExpNeg,
    ExpandExpToSinhPlusCosh,
    ExpandExpNegToCoshMinusSinh,
    ExpandNegExpNegToSinhMinusCosh,
    RecognizeCoshFromExp,
    RecognizeSinhFromExp,
    RecognizeNegSinhFromExp,
    RecognizeDoubleCoshFromExp,
    RecognizeDoubleSinhFromExp,
    RecognizeTanhFromExp,
    RecognizeNegTanhFromExp,
    ExpandCoshToExpHalfSum,
    ExpandSinhToExpHalfDiff,
    ExpandNegSinhToExpHalfNegDiff,
    ExpandDoubleCoshToExpSum,
    ExpandDoubleSinhToExpDiff,
    ExpandTanhToExpRatio,
    ExpandNegTanhToExpNegRatio,
    ExpandHyperbolicCoshHalfAngleSquare,
    ExpandHyperbolicSinhHalfAngleSquare,
    ContractHyperbolicCoshHalfAngleSquare,
    ContractHyperbolicSinhHalfAngleSquare,
    ExpandCoshDoubleAngleAsSum,
    ExpandCoshDoubleAngleAsTwoCoshSqMinusOne,
    ExpandCoshDoubleAngleAsOnePlusTwoSinhSq,
    ExpandNegativeCoshDoubleAngleAsOneMinusTwoCoshSq,
    ExpandNegativeCoshDoubleAngleAsNegativeOneMinusTwoSinhSq,
    RecognizeTwoCoshSqMinusOneAsCoshDoubleAngle,
    RecognizeTwoSinhSqPlusOneAsCoshDoubleAngle,
    RecognizeOneMinusTwoCoshSqAsNegativeCoshDoubleAngle,
    RecognizeNegativeOneMinusTwoSinhSqAsNegativeCoshDoubleAngle,
    RecognizeCoshDoubleAngleMinusOneAsTwoSinhSq,
    RecognizeCoshDoubleAnglePlusOneAsTwoCoshSq,
    RecognizeOneMinusCoshDoubleAngleAsNegativeTwoSinhSq,
    RecognizeNegativeOneMinusCoshDoubleAngleAsNegativeTwoCoshSq,
    ExpandTwoSinhSqToCoshDoubleAngleMinusOne,
    ExpandTwoCoshSqToCoshDoubleAnglePlusOne,
    ExpandNegativeTwoSinhSqToOneMinusCoshDoubleAngle,
    ExpandNegativeTwoCoshSqToNegativeOneMinusCoshDoubleAngle,
    DoubleAngleCoshSum,
    DoubleAngleSubChainToZero,
    ContractSinhAngleSumDiff,
    ContractCoshAngleSumDiff,
    ExpandTanhAngleSumDiff,
    ContractTanhAngleSumDiff,
    ExpandTanhTripleAngle,
    ContractTanhTripleAngle,
    ContractSinhDoubleAngle,
    ContractTanhDoubleAngle,
    ContractSinhTripleAngle,
    ContractCoshTripleAngle,
    ProductToSumSinhCosh,
    ProductToSumCoshCosh,
    ProductToSumSinhSinh,
    SumToProductSinhCosh,
    SumToProductCoshCosh,
    SumToProductSinhSinh,
    ExpandCombinedSinhTripleAngle,
    ExpandCombinedCoshTripleAngle,
    ContractCombinedSinhTripleAngle,
    ContractCombinedCoshTripleAngle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeriveHyperbolicRewrite {
    pub(crate) rewritten: ExprId,
    pub(crate) kind: DeriveHyperbolicRewriteKind,
}

pub(crate) fn generate_hyperbolic_bridge_rewrites(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Vec<DeriveHyperbolicRewrite> {
    let mut rewrites = Vec::new();

    if let Some((rewritten, kind)) = rewrite_hyperbolic_exponential_bridge_expr(ctx, expr) {
        push_unique_hyperbolic_bridge_rewrite(&mut rewrites, rewritten, kind);
    }

    if let Some((rewritten, kind)) = rewrite_negated_hyperbolic_exponential_bridge_expr(ctx, expr) {
        push_unique_hyperbolic_bridge_rewrite(&mut rewrites, rewritten, kind);
    }

    if let Some((rewritten, kind)) =
        rewrite_hyperbolic_reciprocal_exponential_bridge_expr(ctx, expr)
    {
        push_unique_hyperbolic_bridge_rewrite(&mut rewrites, rewritten, kind);
    }

    if let Some((rewritten, kind)) =
        rewrite_negated_hyperbolic_reciprocal_exponential_bridge_expr(ctx, expr)
    {
        push_unique_hyperbolic_bridge_rewrite(&mut rewrites, rewritten, kind);
    }

    if let Some((rewritten, kind)) = rewrite_scaled_hyperbolic_exponential_bridge_expr(ctx, expr) {
        push_unique_hyperbolic_bridge_rewrite(&mut rewrites, rewritten, kind);
    }

    if let Some((rewritten, kind)) =
        rewrite_half_hyperbolic_from_exponentials_bridge_expr(ctx, expr)
    {
        push_unique_hyperbolic_bridge_rewrite(&mut rewrites, rewritten, kind);
    }

    if let Some((rewritten, kind)) =
        rewrite_scaled_hyperbolic_reciprocal_exponential_bridge_expr(ctx, expr)
    {
        push_unique_hyperbolic_bridge_rewrite(&mut rewrites, rewritten, kind);
    }

    if let Some((rewritten, kind)) = rewrite_hyperbolic_product_to_sum_bridge_expr(ctx, expr, false)
    {
        push_unique_hyperbolic_bridge_rewrite(&mut rewrites, rewritten, kind);
    }

    if let Some((rewritten, kind)) = rewrite_hyperbolic_sum_to_product_bridge_expr(ctx, expr, false)
    {
        push_unique_hyperbolic_bridge_rewrite(&mut rewrites, rewritten, kind);
    }

    if let Some(positive_expr) = strip_unit_negation(ctx, expr) {
        if let Some((rewritten, kind)) =
            rewrite_hyperbolic_product_to_sum_bridge_expr(ctx, positive_expr, true)
        {
            push_unique_hyperbolic_bridge_rewrite(&mut rewrites, rewritten, kind);
        }

        if let Some((rewritten, kind)) =
            rewrite_hyperbolic_sum_to_product_bridge_expr(ctx, positive_expr, true)
        {
            push_unique_hyperbolic_bridge_rewrite(&mut rewrites, rewritten, kind);
        }
    }

    if let Some((rewritten, kind)) = rewrite_hyperbolic_angle_sum_diff_bridge_expr(ctx, expr, false)
    {
        push_unique_hyperbolic_bridge_rewrite(&mut rewrites, rewritten, kind);
    }

    if let Some(positive_expr) = strip_unit_negation(ctx, expr) {
        if let Some((rewritten, kind)) =
            rewrite_hyperbolic_angle_sum_diff_bridge_expr(ctx, positive_expr, true)
        {
            push_unique_hyperbolic_bridge_rewrite(&mut rewrites, rewritten, kind);
        }
    }

    generate_combined_additive_hyperbolic_triple_angle_rewrites(ctx, expr, &mut rewrites);
    generate_reverse_combined_additive_hyperbolic_triple_angle_rewrites(ctx, expr, &mut rewrites);

    rewrites
}

pub(crate) fn generate_hyperbolic_additive_term_bridge_rewrites(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Vec<DeriveHyperbolicRewrite> {
    let terms = add_terms_signed(ctx, expr);
    if terms.len() <= 1 {
        return Vec::new();
    }

    let mut rewrites = Vec::new();
    generate_combined_additive_hyperbolic_triple_angle_rewrites(ctx, expr, &mut rewrites);
    generate_reverse_combined_additive_hyperbolic_triple_angle_rewrites(ctx, expr, &mut rewrites);

    for (index, (term, sign)) in terms.iter().enumerate() {
        for candidate in generate_hyperbolic_bridge_rewrites(ctx, *term) {
            let signed_rewritten_term = apply_sign_to_term(ctx, candidate.rewritten, *sign);
            let rewritten = rebuild_additive_terms_with_rewritten_term(
                ctx,
                &terms,
                index,
                signed_rewritten_term,
            );
            push_unique_hyperbolic_bridge_rewrite(&mut rewrites, rewritten, candidate.kind);
        }
    }

    rewrites
}

fn push_unique_hyperbolic_bridge_rewrite(
    rewrites: &mut Vec<DeriveHyperbolicRewrite>,
    rewritten: ExprId,
    kind: DeriveHyperbolicRewriteKind,
) {
    if rewrites
        .iter()
        .any(|rewrite| rewrite.rewritten == rewritten && rewrite.kind == kind)
    {
        return;
    }

    rewrites.push(DeriveHyperbolicRewrite { rewritten, kind });
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

fn rewrite_hyperbolic_exponential_bridge_expr(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, DeriveHyperbolicRewriteKind)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    build_hyperbolic_exponential_bridge(ctx, ctx.builtin_of(*fn_id)?, args[0], false)
}

fn rewrite_negated_hyperbolic_exponential_bridge_expr(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, DeriveHyperbolicRewriteKind)> {
    let positive_expr = strip_unit_negation(ctx, expr)?;
    let Expr::Function(fn_id, args) = ctx.get(positive_expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    build_hyperbolic_exponential_bridge(ctx, ctx.builtin_of(*fn_id)?, args[0], true)
}

fn rewrite_hyperbolic_reciprocal_exponential_bridge_expr(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, DeriveHyperbolicRewriteKind)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    build_hyperbolic_reciprocal_exponential_bridge(ctx, ctx.builtin_of(*fn_id)?, args[0], false)
}

fn rewrite_negated_hyperbolic_reciprocal_exponential_bridge_expr(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, DeriveHyperbolicRewriteKind)> {
    let positive_expr = strip_unit_negation(ctx, expr)?;
    let Expr::Function(fn_id, args) = ctx.get(positive_expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    build_hyperbolic_reciprocal_exponential_bridge(ctx, ctx.builtin_of(*fn_id)?, args[0], true)
}

fn build_hyperbolic_exponential_bridge(
    ctx: &mut cas_ast::Context,
    builtin: BuiltinFn,
    arg: ExprId,
    negated: bool,
) -> Option<(ExprId, DeriveHyperbolicRewriteKind)> {
    let exp_arg = ctx.call_builtin(BuiltinFn::Exp, vec![arg]);
    let neg_arg = ctx.add(Expr::Neg(arg));
    let exp_neg_arg = ctx.call_builtin(BuiltinFn::Exp, vec![neg_arg]);
    let two = ctx.num(2);
    let exp_sum = ctx.add(Expr::Add(exp_arg, exp_neg_arg));
    let exp_diff = ctx.add(Expr::Sub(exp_arg, exp_neg_arg));
    let neg_exp_diff = ctx.add(Expr::Sub(exp_neg_arg, exp_arg));
    let exp_half_sum = ctx.add(Expr::Div(exp_sum, two));
    let neg_exp_half_sum = negate_preserving_fraction_numerator(ctx, exp_half_sum);
    let exp_half_diff = ctx.add(Expr::Div(exp_diff, two));
    let neg_exp_half_diff = negate_preserving_fraction_numerator(ctx, exp_half_diff);

    let (candidate, kind, should_simplify) = match (builtin, negated) {
        (BuiltinFn::Cosh, false) => (
            exp_half_sum,
            DeriveHyperbolicRewriteKind::ExpandCoshToExpHalfSum,
            false,
        ),
        (BuiltinFn::Cosh, true) => (
            neg_exp_half_sum,
            DeriveHyperbolicRewriteKind::ExpandCoshToExpHalfSum,
            false,
        ),
        (BuiltinFn::Sinh, false) => (
            exp_half_diff,
            DeriveHyperbolicRewriteKind::ExpandSinhToExpHalfDiff,
            false,
        ),
        (BuiltinFn::Sinh, true) => (
            neg_exp_half_diff,
            DeriveHyperbolicRewriteKind::ExpandNegSinhToExpHalfNegDiff,
            false,
        ),
        (BuiltinFn::Tanh, false) => (
            ctx.add(Expr::Div(exp_diff, exp_sum)),
            DeriveHyperbolicRewriteKind::ExpandTanhToExpRatio,
            false,
        ),
        (BuiltinFn::Tanh, true) => (
            ctx.add(Expr::Div(neg_exp_diff, exp_sum)),
            DeriveHyperbolicRewriteKind::ExpandNegTanhToExpNegRatio,
            false,
        ),
        _ => return None,
    };

    let rewritten = if should_simplify {
        run_default_simplify(ctx, candidate)
    } else {
        candidate
    };

    Some((rewritten, kind))
}

fn build_hyperbolic_reciprocal_exponential_bridge(
    ctx: &mut cas_ast::Context,
    builtin: BuiltinFn,
    arg: ExprId,
    negated: bool,
) -> Option<(ExprId, DeriveHyperbolicRewriteKind)> {
    let exp_arg = ctx.call_builtin(BuiltinFn::Exp, vec![arg]);
    let one = ctx.num(1);
    let reciprocal = ctx.add(Expr::Div(one, exp_arg));
    let two = ctx.num(2);
    let exp_sum = ctx.add(Expr::Add(exp_arg, reciprocal));
    let exp_diff = ctx.add(Expr::Sub(exp_arg, reciprocal));
    let neg_exp_diff = ctx.add(Expr::Sub(reciprocal, exp_arg));
    let exp_half_sum = ctx.add(Expr::Div(exp_sum, two));
    let neg_exp_half_sum = negate_preserving_fraction_numerator(ctx, exp_half_sum);
    let exp_half_diff = ctx.add(Expr::Div(exp_diff, two));
    let neg_exp_half_diff = negate_preserving_fraction_numerator(ctx, exp_half_diff);

    let (candidate, kind) = match (builtin, negated) {
        (BuiltinFn::Cosh, false) => (
            exp_half_sum,
            DeriveHyperbolicRewriteKind::ExpandCoshToExpHalfSum,
        ),
        (BuiltinFn::Cosh, true) => (
            neg_exp_half_sum,
            DeriveHyperbolicRewriteKind::ExpandCoshToExpHalfSum,
        ),
        (BuiltinFn::Sinh, false) => (
            exp_half_diff,
            DeriveHyperbolicRewriteKind::ExpandSinhToExpHalfDiff,
        ),
        (BuiltinFn::Sinh, true) => (
            neg_exp_half_diff,
            DeriveHyperbolicRewriteKind::ExpandNegSinhToExpHalfNegDiff,
        ),
        (BuiltinFn::Tanh, false) => (
            ctx.add(Expr::Div(exp_diff, exp_sum)),
            DeriveHyperbolicRewriteKind::ExpandTanhToExpRatio,
        ),
        (BuiltinFn::Tanh, true) => (
            ctx.add(Expr::Div(neg_exp_diff, exp_sum)),
            DeriveHyperbolicRewriteKind::ExpandNegTanhToExpNegRatio,
        ),
        _ => return None,
    };

    Some((candidate, kind))
}

fn rewrite_scaled_hyperbolic_exponential_bridge_expr(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, DeriveHyperbolicRewriteKind)> {
    if let Some((rewritten, kind)) = recognize_scaled_hyperbolic_from_exponentials(ctx, expr) {
        return Some((rewritten, kind));
    }

    if let Some((rewritten, kind)) = expand_scaled_hyperbolic_to_exponentials(ctx, expr) {
        return Some((rewritten, kind));
    }

    None
}

fn rewrite_half_hyperbolic_from_exponentials_bridge_expr(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, DeriveHyperbolicRewriteKind)> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    if !cas_math::expr_predicates::is_two_expr(ctx, *denominator) {
        return None;
    }

    let (arg, positive_arg_sign, negative_arg_sign) =
        extract_exponential_pair_signs(ctx, *numerator)?;

    match (positive_arg_sign, negative_arg_sign) {
        (Sign::Pos, Sign::Pos) => Some((
            build_canonical_hyperbolic_call(ctx, BuiltinFn::Cosh, arg),
            DeriveHyperbolicRewriteKind::RecognizeCoshFromExp,
        )),
        (Sign::Pos, Sign::Neg) => Some((
            build_canonical_hyperbolic_call(ctx, BuiltinFn::Sinh, arg),
            DeriveHyperbolicRewriteKind::RecognizeSinhFromExp,
        )),
        (Sign::Neg, Sign::Pos) => {
            let positive = build_canonical_hyperbolic_call(ctx, BuiltinFn::Sinh, arg);
            Some((
                ctx.add(Expr::Neg(positive)),
                DeriveHyperbolicRewriteKind::RecognizeNegSinhFromExp,
            ))
        }
        (Sign::Neg, Sign::Neg) => {
            let positive = build_canonical_hyperbolic_call(ctx, BuiltinFn::Cosh, arg);
            Some((
                ctx.add(Expr::Neg(positive)),
                DeriveHyperbolicRewriteKind::RecognizeCoshFromExp,
            ))
        }
    }
}

fn rewrite_scaled_hyperbolic_reciprocal_exponential_bridge_expr(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, DeriveHyperbolicRewriteKind)> {
    let (builtin, arg, negated) = extract_signed_scaled_hyperbolic_call(ctx, expr, 2)?;
    let exp_arg = ctx.call_builtin(BuiltinFn::Exp, vec![arg]);
    let one = ctx.num(1);
    let reciprocal = ctx.add(Expr::Div(one, exp_arg));
    let exp_sum = ctx.add(Expr::Add(exp_arg, reciprocal));
    let exp_diff = ctx.add(Expr::Sub(exp_arg, reciprocal));
    let neg_exp_diff = ctx.add(Expr::Sub(reciprocal, exp_arg));

    match (builtin, negated) {
        (BuiltinFn::Cosh, false) => Some((
            exp_sum,
            DeriveHyperbolicRewriteKind::ExpandDoubleCoshToExpSum,
        )),
        (BuiltinFn::Cosh, true) => Some((
            ctx.add(Expr::Neg(exp_sum)),
            DeriveHyperbolicRewriteKind::ExpandDoubleCoshToExpSum,
        )),
        (BuiltinFn::Sinh, false) => Some((
            exp_diff,
            DeriveHyperbolicRewriteKind::ExpandDoubleSinhToExpDiff,
        )),
        (BuiltinFn::Sinh, true) => Some((
            neg_exp_diff,
            DeriveHyperbolicRewriteKind::ExpandDoubleSinhToExpDiff,
        )),
        _ => None,
    }
}

fn recognize_scaled_hyperbolic_from_exponentials(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, DeriveHyperbolicRewriteKind)> {
    let terms = add_terms_signed(ctx, expr);
    if terms.len() != 2 {
        return None;
    }

    let (first_term, first_sign) = terms[0];
    let (second_term, second_sign) = terms[1];
    let first_arg = extract_exp_call_arg(ctx, first_term)?;
    let second_arg = extract_exp_call_arg(ctx, second_term)?;
    if !is_negation(ctx, first_arg, second_arg) {
        return None;
    }

    let magnitude = ctx.num(2);

    match (first_sign, second_sign) {
        (Sign::Pos, Sign::Pos) => {
            let cosh = build_canonical_hyperbolic_call(ctx, BuiltinFn::Cosh, first_arg);
            Some((
                smart_mul(ctx, magnitude, cosh),
                DeriveHyperbolicRewriteKind::RecognizeDoubleCoshFromExp,
            ))
        }
        (Sign::Pos, Sign::Neg) => {
            let sinh = build_canonical_hyperbolic_call(ctx, BuiltinFn::Sinh, first_arg);
            Some((
                smart_mul(ctx, magnitude, sinh),
                DeriveHyperbolicRewriteKind::RecognizeDoubleSinhFromExp,
            ))
        }
        (Sign::Neg, Sign::Pos) => {
            let sinh = build_canonical_hyperbolic_call(ctx, BuiltinFn::Sinh, second_arg);
            Some((
                smart_mul(ctx, magnitude, sinh),
                DeriveHyperbolicRewriteKind::RecognizeDoubleSinhFromExp,
            ))
        }
        (Sign::Neg, Sign::Neg) => {
            let cosh = build_canonical_hyperbolic_call(ctx, BuiltinFn::Cosh, first_arg);
            let doubled = smart_mul(ctx, magnitude, cosh);
            Some((
                ctx.add(Expr::Neg(doubled)),
                DeriveHyperbolicRewriteKind::RecognizeDoubleCoshFromExp,
            ))
        }
    }
}

fn expand_scaled_hyperbolic_to_exponentials(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, DeriveHyperbolicRewriteKind)> {
    let (builtin, arg, negated) = extract_signed_scaled_hyperbolic_call(ctx, expr, 2)?;
    let exp_arg = ctx.call_builtin(BuiltinFn::Exp, vec![arg]);
    let neg_arg = ctx.add(Expr::Neg(arg));
    let exp_neg_arg = ctx.call_builtin(BuiltinFn::Exp, vec![neg_arg]);

    let candidate = match builtin {
        BuiltinFn::Cosh => {
            let sum = ctx.add(Expr::Add(exp_arg, exp_neg_arg));
            if negated {
                ctx.add(Expr::Neg(sum))
            } else {
                sum
            }
        }
        BuiltinFn::Sinh => {
            if negated {
                ctx.add(Expr::Sub(exp_neg_arg, exp_arg))
            } else {
                ctx.add(Expr::Sub(exp_arg, exp_neg_arg))
            }
        }
        _ => return None,
    };

    Some((
        candidate,
        match builtin {
            BuiltinFn::Cosh => DeriveHyperbolicRewriteKind::ExpandDoubleCoshToExpSum,
            BuiltinFn::Sinh => DeriveHyperbolicRewriteKind::ExpandDoubleSinhToExpDiff,
            _ => unreachable!("only sinh/cosh should reach scaled hyperbolic exponential bridge"),
        },
    ))
}

fn extract_exp_call_arg(ctx: &mut cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    if let Some(arg) = cas_math::expr_extract::extract_exp_argument(ctx, expr) {
        return Some(arg);
    }

    let (numerator, denominator) = as_div(ctx, expr)?;
    match ctx.get(numerator) {
        Expr::Number(n) if n.is_one() => {}
        _ => return None,
    }

    let arg = cas_math::expr_extract::extract_exp_argument(ctx, denominator)?;
    Some(ctx.add(Expr::Neg(arg)))
}

fn extract_signed_scaled_hyperbolic_call(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    scale: i64,
) -> Option<(BuiltinFn, ExprId, bool)> {
    let (positive_expr, mut negated) = if let Some(inner) = strip_unit_negation(ctx, expr) {
        (inner, true)
    } else {
        (expr, false)
    };

    let factors = flatten_mul_chain(ctx, positive_expr);
    if factors.len() != 2 {
        return None;
    }

    let mut saw_scale = false;
    let mut hyperbolic = None;

    for factor in factors {
        match ctx.get(factor) {
            Expr::Number(n) if n.is_integer() => {
                let integer = n.to_integer();
                if integer == scale.into() {
                    saw_scale = true;
                } else if integer == (-scale).into() {
                    saw_scale = true;
                    negated = !negated;
                } else {
                    return None;
                }
            }
            Expr::Function(fn_id, args)
                if args.len() == 1
                    && matches!(
                        ctx.builtin_of(*fn_id),
                        Some(BuiltinFn::Sinh | BuiltinFn::Cosh)
                    ) =>
            {
                hyperbolic = Some((ctx.builtin_of(*fn_id).expect("builtin"), args[0]));
            }
            _ => return None,
        }
    }

    let (builtin, arg) = hyperbolic?;
    if !saw_scale {
        return None;
    }

    Some((builtin, arg, negated))
}

pub(crate) fn should_try_hyperbolic_planner_before_simplify(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> bool {
    let expr_has_exp = contains_exponential_like(ctx, expr);
    let expr_has_hyper = contains_hyperbolic_fn(ctx, expr);
    let target_has_exp = contains_exponential_like(ctx, target_expr);
    let target_has_hyper = contains_hyperbolic_fn(ctx, target_expr);

    if !((expr_has_exp && target_has_hyper) || (expr_has_hyper && target_has_exp)) {
        return false;
    }

    if try_rewrite_hyperbolic_exponential_bridge_target_aware(ctx, expr, target_expr).is_some() {
        return false;
    }

    let rewrites = generate_hyperbolic_bridge_rewrites(ctx, expr);
    if rewrites.is_empty() {
        return false;
    }

    rewrites.iter().any(|rewrite| {
        rewrite.kind.is_exponential_bridge()
            && matches_target_modulo_simplify(ctx, rewrite.rewritten, target_expr)
    })
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

fn contains_hyperbolic_fn(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            Expr::Function(fn_id, args) => {
                if matches!(
                    ctx.builtin_of(*fn_id),
                    Some(BuiltinFn::Sinh | BuiltinFn::Cosh | BuiltinFn::Tanh)
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

impl DeriveHyperbolicRewriteKind {
    fn is_exponential_bridge(self) -> bool {
        matches!(
            self,
            Self::SinhCoshToExp
                | Self::CoshMinusSinhToExpNeg
                | Self::SinhMinusCoshToNegExpNeg
                | Self::ExpandExpToSinhPlusCosh
                | Self::ExpandExpNegToCoshMinusSinh
                | Self::ExpandNegExpNegToSinhMinusCosh
                | Self::RecognizeCoshFromExp
                | Self::RecognizeSinhFromExp
                | Self::RecognizeNegSinhFromExp
                | Self::RecognizeDoubleCoshFromExp
                | Self::RecognizeDoubleSinhFromExp
                | Self::RecognizeTanhFromExp
                | Self::RecognizeNegTanhFromExp
                | Self::ExpandCoshToExpHalfSum
                | Self::ExpandSinhToExpHalfDiff
                | Self::ExpandNegSinhToExpHalfNegDiff
                | Self::ExpandDoubleCoshToExpSum
                | Self::ExpandDoubleSinhToExpDiff
                | Self::ExpandTanhToExpRatio
                | Self::ExpandNegTanhToExpNegRatio
        )
    }

    pub(crate) fn description(self) -> &'static str {
        match self {
            Self::PythagoreanOne => "Recognize cosh(u)^2 - sinh(u)^2 = 1",
            Self::PythagoreanNegativeOne => "Recognize sinh(u)^2 - cosh(u)^2 = -1",
            Self::TanhPythagorean => "Recognize 1 - tanh(u)^2 as 1 / cosh(u)^2",
            Self::ExpandTanhPythagorean => "Expand 1 / cosh(u)^2 as 1 - tanh(u)^2",
            Self::RecognizeCoshSquaredMinusOneAsSinhSquared => {
                "Recognize cosh(u)^2 - 1 as sinh(u)^2"
            }
            Self::ExpandSinhSquaredToCoshSquaredMinusOne => "Expand sinh(u)^2 as cosh(u)^2 - 1",
            Self::RecognizeOnePlusSinhSquaredAsCoshSquared => {
                "Recognize 1 + sinh(u)^2 as cosh(u)^2"
            }
            Self::ExpandCoshSquaredToOnePlusSinhSquared => "Expand cosh(u)^2 as 1 + sinh(u)^2",
            Self::SinhCoshToTanh => "Recognize sinh(u) / cosh(u) as tanh(u)",
            Self::TanhToSinhCosh => "Expand tanh(u) as sinh(u) / cosh(u)",
            Self::SinhCoshToExp => "Recognize sinh(u) + cosh(u) as exp(u)",
            Self::CoshMinusSinhToExpNeg => "Recognize cosh(u) - sinh(u) as exp(-u)",
            Self::SinhMinusCoshToNegExpNeg => "Recognize sinh(u) - cosh(u) as -exp(-u)",
            Self::ExpandExpToSinhPlusCosh => "Expand exp(u) as sinh(u) + cosh(u)",
            Self::ExpandExpNegToCoshMinusSinh => "Expand exp(-u) as cosh(u) - sinh(u)",
            Self::ExpandNegExpNegToSinhMinusCosh => "Expand -exp(-u) as sinh(u) - cosh(u)",
            Self::RecognizeCoshFromExp => "Recognize (exp(u) + exp(-u)) / 2 as cosh(u)",
            Self::RecognizeSinhFromExp => "Recognize (exp(u) - exp(-u)) / 2 as sinh(u)",
            Self::RecognizeNegSinhFromExp => "Recognize (exp(-u) - exp(u)) / 2 as -sinh(u)",
            Self::RecognizeDoubleCoshFromExp => "Recognize exp(u) + exp(-u) as 2·cosh(u)",
            Self::RecognizeDoubleSinhFromExp => "Recognize exp(u) - exp(-u) as 2·sinh(u)",
            Self::RecognizeTanhFromExp => {
                "Recognize (exp(u) - exp(-u)) / (exp(u) + exp(-u)) as tanh(u)"
            }
            Self::RecognizeNegTanhFromExp => {
                "Recognize (exp(-u) - exp(u)) / (exp(u) + exp(-u)) as -tanh(u)"
            }
            Self::ExpandCoshToExpHalfSum => "Expand cosh(u) as (exp(u) + exp(-u)) / 2",
            Self::ExpandSinhToExpHalfDiff => "Expand sinh(u) as (exp(u) - exp(-u)) / 2",
            Self::ExpandNegSinhToExpHalfNegDiff => "Expand -sinh(u) as (exp(-u) - exp(u)) / 2",
            Self::ExpandDoubleCoshToExpSum => "Expand 2·cosh(u) as exp(u) + exp(-u)",
            Self::ExpandDoubleSinhToExpDiff => "Expand 2·sinh(u) as exp(u) - exp(-u)",
            Self::ExpandTanhToExpRatio => {
                "Expand tanh(u) as (exp(u) - exp(-u)) / (exp(u) + exp(-u))"
            }
            Self::ExpandNegTanhToExpNegRatio => {
                "Expand -tanh(u) as (exp(-u) - exp(u)) / (exp(u) + exp(-u))"
            }
            Self::ExpandHyperbolicCoshHalfAngleSquare => "Expand cosh(u/2)^2 as (cosh(u) + 1) / 2",
            Self::ExpandHyperbolicSinhHalfAngleSquare => "Expand sinh(u/2)^2 as (cosh(u) - 1) / 2",
            Self::ContractHyperbolicCoshHalfAngleSquare => {
                "Recognize (cosh(u) + 1) / 2 as cosh(u/2)^2"
            }
            Self::ContractHyperbolicSinhHalfAngleSquare => {
                "Recognize (cosh(u) - 1) / 2 as sinh(u/2)^2"
            }
            Self::ExpandCoshDoubleAngleAsSum => "Expand cosh(2u) as cosh(u)^2 + sinh(u)^2",
            Self::ExpandCoshDoubleAngleAsTwoCoshSqMinusOne => "Expand cosh(2u) as 2·cosh(u)^2 - 1",
            Self::ExpandCoshDoubleAngleAsOnePlusTwoSinhSq => "Expand cosh(2u) as 1 + 2·sinh(u)^2",
            Self::ExpandNegativeCoshDoubleAngleAsOneMinusTwoCoshSq => {
                "Expand -cosh(2u) as 1 - 2·cosh(u)^2"
            }
            Self::ExpandNegativeCoshDoubleAngleAsNegativeOneMinusTwoSinhSq => {
                "Expand -cosh(2u) as -1 - 2·sinh(u)^2"
            }
            Self::RecognizeTwoCoshSqMinusOneAsCoshDoubleAngle => {
                "Recognize 2·cosh(u)^2 - 1 as cosh(2u)"
            }
            Self::RecognizeTwoSinhSqPlusOneAsCoshDoubleAngle => {
                "Recognize 2·sinh(u)^2 + 1 as cosh(2u)"
            }
            Self::RecognizeOneMinusTwoCoshSqAsNegativeCoshDoubleAngle => {
                "Recognize 1 - 2·cosh(u)^2 as -cosh(2u)"
            }
            Self::RecognizeNegativeOneMinusTwoSinhSqAsNegativeCoshDoubleAngle => {
                "Recognize -1 - 2·sinh(u)^2 as -cosh(2u)"
            }
            Self::RecognizeCoshDoubleAngleMinusOneAsTwoSinhSq => {
                "Recognize cosh(2u) - 1 as 2·sinh(u)^2"
            }
            Self::RecognizeCoshDoubleAnglePlusOneAsTwoCoshSq => {
                "Recognize cosh(2u) + 1 as 2·cosh(u)^2"
            }
            Self::RecognizeOneMinusCoshDoubleAngleAsNegativeTwoSinhSq => {
                "Recognize 1 - cosh(2u) as -2·sinh(u)^2"
            }
            Self::RecognizeNegativeOneMinusCoshDoubleAngleAsNegativeTwoCoshSq => {
                "Recognize -1 - cosh(2u) as -2·cosh(u)^2"
            }
            Self::ExpandTwoSinhSqToCoshDoubleAngleMinusOne => "Expand 2·sinh(u)^2 as cosh(2u) - 1",
            Self::ExpandTwoCoshSqToCoshDoubleAnglePlusOne => "Expand 2·cosh(u)^2 as cosh(2u) + 1",
            Self::ExpandNegativeTwoSinhSqToOneMinusCoshDoubleAngle => {
                "Expand -2·sinh(u)^2 as 1 - cosh(2u)"
            }
            Self::ExpandNegativeTwoCoshSqToNegativeOneMinusCoshDoubleAngle => {
                "Expand -2·cosh(u)^2 as -1 - cosh(2u)"
            }
            Self::DoubleAngleCoshSum => "Recognize cosh(u)^2 + sinh(u)^2 as cosh(2u)",
            Self::DoubleAngleSubChainToZero => "Recognize cosh(2u) - cosh(u)^2 - sinh(u)^2 = 0",
            Self::ContractSinhAngleSumDiff => {
                "Recognize sinh(u)·cosh(v) ± cosh(u)·sinh(v) as sinh(u ± v)"
            }
            Self::ContractCoshAngleSumDiff => {
                "Recognize cosh(u)·cosh(v) ± sinh(u)·sinh(v) as cosh(u ± v)"
            }
            Self::ExpandTanhAngleSumDiff => {
                "Expand tanh(u ± v) as (tanh(u) ± tanh(v)) / (1 ± tanh(u)·tanh(v))"
            }
            Self::ContractTanhAngleSumDiff => {
                "Recognize (tanh(u) ± tanh(v)) / (1 ± tanh(u)·tanh(v)) as tanh(u ± v)"
            }
            Self::ExpandTanhTripleAngle => {
                "Expand tanh(3u) as (3·tanh(u) + tanh(u)^3) / (1 + 3·tanh(u)^2)"
            }
            Self::ContractTanhTripleAngle => {
                "Recognize (3·tanh(u) + tanh(u)^3) / (1 + 3·tanh(u)^2) as tanh(3u)"
            }
            Self::ContractSinhDoubleAngle => "Recognize 2·sinh(u)·cosh(u) as sinh(2u)",
            Self::ContractTanhDoubleAngle => "Recognize 2·tanh(u)/(1 + tanh(u)^2) as tanh(2u)",
            Self::ContractSinhTripleAngle => "Recognize 3·sinh(u) + 4·sinh(u)^3 as sinh(3u)",
            Self::ContractCoshTripleAngle => "Recognize 4·cosh(u)^3 - 3·cosh(u) as cosh(3u)",
            Self::ProductToSumSinhCosh => {
                "Expand 2·sinh(u)·cosh(v) as sinh(u + v) + sinh(u - v)"
            }
            Self::ProductToSumCoshCosh => {
                "Expand 2·cosh(u)·cosh(v) as cosh(u + v) + cosh(u - v)"
            }
            Self::ProductToSumSinhSinh => {
                "Expand 2·sinh(u)·sinh(v) as cosh(u + v) - cosh(u - v)"
            }
            Self::SumToProductSinhCosh => {
                "Contract sinh(u) ± sinh(v) into 2·cosh((u + v)/2)·sinh((u - v)/2) or 2·sinh((u + v)/2)·cosh((u - v)/2)"
            }
            Self::SumToProductCoshCosh => {
                "Contract cosh(u) + cosh(v) into 2·cosh((u + v)/2)·cosh((u - v)/2)"
            }
            Self::SumToProductSinhSinh => {
                "Contract cosh(u) - cosh(v) into 2·sinh((u + v)/2)·sinh((u - v)/2)"
            }
            Self::ExpandCombinedSinhTripleAngle => {
                "Combine sinh(u) ± sinh(3u) using the hyperbolic triple-angle identity"
            }
            Self::ExpandCombinedCoshTripleAngle => {
                "Combine cosh(u) ± cosh(3u) using the hyperbolic triple-angle identity"
            }
            Self::ContractCombinedSinhTripleAngle => {
                "Split a sinh triple-angle polynomial into sinh(u) ± sinh(3u)"
            }
            Self::ContractCombinedCoshTripleAngle => {
                "Split a cosh triple-angle polynomial into cosh(u) ± cosh(3u)"
            }
        }
    }

    pub(crate) fn rule_name(self) -> &'static str {
        match self {
            Self::PythagoreanOne
            | Self::PythagoreanNegativeOne
            | Self::TanhPythagorean
            | Self::ExpandTanhPythagorean
            | Self::RecognizeCoshSquaredMinusOneAsSinhSquared
            | Self::ExpandSinhSquaredToCoshSquaredMinusOne
            | Self::RecognizeOnePlusSinhSquaredAsCoshSquared
            | Self::ExpandCoshSquaredToOnePlusSinhSquared => "Hyperbolic Pythagorean Identity",
            Self::SinhCoshToTanh | Self::TanhToSinhCosh => "Hyperbolic Quotient Identity",
            Self::SinhCoshToExp
            | Self::CoshMinusSinhToExpNeg
            | Self::SinhMinusCoshToNegExpNeg
            | Self::ExpandExpToSinhPlusCosh
            | Self::ExpandExpNegToCoshMinusSinh
            | Self::ExpandNegExpNegToSinhMinusCosh
            | Self::RecognizeCoshFromExp
            | Self::RecognizeSinhFromExp
            | Self::RecognizeNegSinhFromExp
            | Self::RecognizeDoubleCoshFromExp
            | Self::RecognizeDoubleSinhFromExp
            | Self::RecognizeTanhFromExp
            | Self::RecognizeNegTanhFromExp
            | Self::ExpandCoshToExpHalfSum
            | Self::ExpandSinhToExpHalfDiff
            | Self::ExpandNegSinhToExpHalfNegDiff
            | Self::ExpandDoubleCoshToExpSum
            | Self::ExpandDoubleSinhToExpDiff
            | Self::ExpandTanhToExpRatio
            | Self::ExpandNegTanhToExpNegRatio => "Hyperbolic Exponential Identity",
            Self::ExpandHyperbolicCoshHalfAngleSquare
            | Self::ExpandHyperbolicSinhHalfAngleSquare
            | Self::ContractHyperbolicCoshHalfAngleSquare
            | Self::ContractHyperbolicSinhHalfAngleSquare => "Hyperbolic Half-Angle Squares",
            Self::ExpandCoshDoubleAngleAsSum
            | Self::ExpandCoshDoubleAngleAsTwoCoshSqMinusOne
            | Self::ExpandCoshDoubleAngleAsOnePlusTwoSinhSq
            | Self::ExpandNegativeCoshDoubleAngleAsOneMinusTwoCoshSq
            | Self::ExpandNegativeCoshDoubleAngleAsNegativeOneMinusTwoSinhSq
            | Self::RecognizeTwoCoshSqMinusOneAsCoshDoubleAngle
            | Self::RecognizeTwoSinhSqPlusOneAsCoshDoubleAngle
            | Self::RecognizeOneMinusTwoCoshSqAsNegativeCoshDoubleAngle
            | Self::RecognizeNegativeOneMinusTwoSinhSqAsNegativeCoshDoubleAngle
            | Self::RecognizeCoshDoubleAngleMinusOneAsTwoSinhSq
            | Self::RecognizeCoshDoubleAnglePlusOneAsTwoCoshSq
            | Self::RecognizeOneMinusCoshDoubleAngleAsNegativeTwoSinhSq
            | Self::RecognizeNegativeOneMinusCoshDoubleAngleAsNegativeTwoCoshSq
            | Self::ExpandTwoSinhSqToCoshDoubleAngleMinusOne
            | Self::ExpandTwoCoshSqToCoshDoubleAnglePlusOne
            | Self::ExpandNegativeTwoSinhSqToOneMinusCoshDoubleAngle
            | Self::ExpandNegativeTwoCoshSqToNegativeOneMinusCoshDoubleAngle
            | Self::DoubleAngleCoshSum
            | Self::DoubleAngleSubChainToZero
            | Self::ContractSinhDoubleAngle
            | Self::ContractTanhDoubleAngle => "Hyperbolic Double-Angle Identity",
            Self::ContractSinhAngleSumDiff
            | Self::ContractCoshAngleSumDiff
            | Self::ExpandTanhAngleSumDiff
            | Self::ContractTanhAngleSumDiff => "Hyperbolic Angle Sum/Difference Identity",
            Self::ExpandTanhTripleAngle
            | Self::ContractTanhTripleAngle
            | Self::ContractSinhTripleAngle
            | Self::ContractCoshTripleAngle
            | Self::ExpandCombinedSinhTripleAngle
            | Self::ExpandCombinedCoshTripleAngle
            | Self::ContractCombinedSinhTripleAngle
            | Self::ContractCombinedCoshTripleAngle => "Hyperbolic Triple-Angle Identity",
            Self::ProductToSumSinhCosh
            | Self::ProductToSumCoshCosh
            | Self::ProductToSumSinhSinh
            | Self::SumToProductSinhCosh
            | Self::SumToProductCoshCosh
            | Self::SumToProductSinhSinh => "Hyperbolic Product-to-Sum Identity",
        }
    }
}

fn rewrite_hyperbolic_product_to_sum_bridge_expr(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    negate_output: bool,
) -> Option<(ExprId, DeriveHyperbolicRewriteKind)> {
    let finalize =
        |ctx: &mut cas_ast::Context, candidate: ExprId, kind: DeriveHyperbolicRewriteKind| {
            let rewritten = if negate_output {
                ctx.add(Expr::Neg(candidate))
            } else {
                candidate
            };
            Some((rewritten, kind))
        };

    let (kind, candidate) = match extract_scaled_hyperbolic_two_factor_product(ctx, expr, 2)? {
        ScaledHyperbolicProductPattern::SinhCosh(left, right) => {
            let sum_expr = ctx.add(Expr::Add(left, right));
            let diff_expr = ctx.add(Expr::Sub(left, right));
            let sum = run_default_simplify(ctx, sum_expr);
            let diff = run_default_simplify(ctx, diff_expr);
            let sum_term = build_canonical_hyperbolic_call(ctx, BuiltinFn::Sinh, sum);
            let diff_term = build_canonical_hyperbolic_call(ctx, BuiltinFn::Sinh, diff);
            (
                DeriveHyperbolicRewriteKind::ProductToSumSinhCosh,
                ctx.add(Expr::Add(sum_term, diff_term)),
            )
        }
        ScaledHyperbolicProductPattern::CoshCosh(left, right) => {
            let sum_expr = ctx.add(Expr::Add(left, right));
            let diff_expr = ctx.add(Expr::Sub(left, right));
            let sum = run_default_simplify(ctx, sum_expr);
            let diff = run_default_simplify(ctx, diff_expr);
            let sum_term = build_canonical_hyperbolic_call(ctx, BuiltinFn::Cosh, sum);
            let diff_term = build_canonical_hyperbolic_call(ctx, BuiltinFn::Cosh, diff);
            (
                DeriveHyperbolicRewriteKind::ProductToSumCoshCosh,
                ctx.add(Expr::Add(sum_term, diff_term)),
            )
        }
        ScaledHyperbolicProductPattern::SinhSinh(left, right) => {
            let sum_expr = ctx.add(Expr::Add(left, right));
            let diff_expr = ctx.add(Expr::Sub(left, right));
            let sum = run_default_simplify(ctx, sum_expr);
            let diff = run_default_simplify(ctx, diff_expr);
            let sum_term = build_canonical_hyperbolic_call(ctx, BuiltinFn::Cosh, sum);
            let diff_term = build_canonical_hyperbolic_call(ctx, BuiltinFn::Cosh, diff);
            (
                DeriveHyperbolicRewriteKind::ProductToSumSinhSinh,
                ctx.add(Expr::Sub(sum_term, diff_term)),
            )
        }
    };

    finalize(ctx, candidate, kind)
}

fn rewrite_hyperbolic_sum_to_product_bridge_expr(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    negate_output: bool,
) -> Option<(ExprId, DeriveHyperbolicRewriteKind)> {
    let terms = add_terms_signed(ctx, expr);
    if terms.len() != 2 {
        return None;
    }

    let (first_fn, first_arg) = hyperbolic_call_arg(ctx, terms[0].0)?;
    let (second_fn, second_arg) = hyperbolic_call_arg(ctx, terms[1].0)?;
    if first_fn != second_fn {
        return None;
    }

    let two = ctx.num(2);

    let (candidate, kind) = match first_fn {
        BuiltinFn::Sinh => {
            if terms[0].1 == terms[1].1 {
                let avg = half_sum(ctx, first_arg, second_arg);
                let half_diff = half_diff(ctx, first_arg, second_arg);
                let avg_sinh = build_canonical_hyperbolic_call(ctx, BuiltinFn::Sinh, avg);
                let diff_cosh = build_canonical_hyperbolic_call(ctx, BuiltinFn::Cosh, half_diff);
                let scaled_avg = smart_mul(ctx, two, avg_sinh);
                (
                    smart_mul(ctx, scaled_avg, diff_cosh),
                    DeriveHyperbolicRewriteKind::SumToProductSinhCosh,
                )
            } else {
                let (positive_arg, negative_arg) = if terms[0].1 == Sign::Pos {
                    (first_arg, second_arg)
                } else {
                    (second_arg, first_arg)
                };
                let avg = half_sum(ctx, positive_arg, negative_arg);
                let half_diff = half_diff(ctx, positive_arg, negative_arg);
                let avg_cosh = build_canonical_hyperbolic_call(ctx, BuiltinFn::Cosh, avg);
                let diff_sinh = build_canonical_hyperbolic_call(ctx, BuiltinFn::Sinh, half_diff);
                let scaled_avg = smart_mul(ctx, two, avg_cosh);
                (
                    smart_mul(ctx, scaled_avg, diff_sinh),
                    DeriveHyperbolicRewriteKind::SumToProductSinhCosh,
                )
            }
        }
        BuiltinFn::Cosh => {
            if terms[0].1 == terms[1].1 {
                let avg = half_sum(ctx, first_arg, second_arg);
                let half_diff = half_diff(ctx, first_arg, second_arg);
                let avg_cosh = build_canonical_hyperbolic_call(ctx, BuiltinFn::Cosh, avg);
                let diff_cosh = build_canonical_hyperbolic_call(ctx, BuiltinFn::Cosh, half_diff);
                let scaled_avg = smart_mul(ctx, two, avg_cosh);
                (
                    smart_mul(ctx, scaled_avg, diff_cosh),
                    DeriveHyperbolicRewriteKind::SumToProductCoshCosh,
                )
            } else {
                let (positive_arg, negative_arg) = if terms[0].1 == Sign::Pos {
                    (first_arg, second_arg)
                } else {
                    (second_arg, first_arg)
                };
                let avg = half_sum(ctx, positive_arg, negative_arg);
                let half_diff = half_diff(ctx, positive_arg, negative_arg);
                let avg_sinh = build_canonical_hyperbolic_call(ctx, BuiltinFn::Sinh, avg);
                let diff_sinh = build_canonical_hyperbolic_call(ctx, BuiltinFn::Sinh, half_diff);
                let scaled_avg = smart_mul(ctx, two, avg_sinh);
                (
                    smart_mul(ctx, scaled_avg, diff_sinh),
                    DeriveHyperbolicRewriteKind::SumToProductSinhSinh,
                )
            }
        }
        _ => return None,
    };

    let rewritten = if negate_output {
        ctx.add(Expr::Neg(candidate))
    } else {
        candidate
    };

    Some((rewritten, kind))
}

pub(crate) fn try_rewrite_hyperbolic_simplify_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    if let Some(rewrite) =
        try_rewrite_hyperbolic_exponential_bridge_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_rewrite_negated_cosh_from_exp_target_aware(ctx, expr, target_expr) {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_rewrite_negated_tanh_pythagorean_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_rewrite_tanh_pythagorean_target_aware(ctx, expr, target_expr) {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        cas_math::hyperbolic_identity_support::try_rewrite_hyperbolic_pythagorean_sub_expr(
            ctx, expr,
        )
    {
        if strong_target_match(ctx, rewrite.rewritten, target_expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: rewrite.rewritten,
                kind: match rewrite.kind {
                    cas_math::hyperbolic_identity_support::HyperbolicIdentityRewriteKind::PythagoreanOne => {
                        DeriveHyperbolicRewriteKind::PythagoreanOne
                    }
                    cas_math::hyperbolic_identity_support::HyperbolicIdentityRewriteKind::PythagoreanNegativeOne => {
                        DeriveHyperbolicRewriteKind::PythagoreanNegativeOne
                    }
                    _ => unreachable!("pythagorean rewrite should only produce pythagorean kinds"),
                },
            });
        }
    }

    if let Some(rewrite) =
        cas_math::hyperbolic_identity_support::try_rewrite_tanh_pythagorean_add_chain(ctx, expr)
    {
        if matches_target_modulo_simplify(ctx, rewrite.rewritten, target_expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: rewrite.rewritten,
                kind: DeriveHyperbolicRewriteKind::TanhPythagorean,
            });
        }
    }

    if let Some(rewrite) =
        cas_math::hyperbolic_identity_support::try_rewrite_tanh_pythagorean_add_chain(
            ctx,
            target_expr,
        )
    {
        if matches_target_modulo_simplify(ctx, rewrite.rewritten, expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: target_expr,
                kind: DeriveHyperbolicRewriteKind::ExpandTanhPythagorean,
            });
        }
    }

    if let Some(rewrite) =
        try_rewrite_negated_inverse_cosh_square_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_rewrite_inverse_cosh_square_target_aware(ctx, expr, target_expr) {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_hyperbolic_pythagorean_factor_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    let simplified_target = run_default_simplify(ctx, target_expr);
    if simplified_target != target_expr {
        if let Some(rewrite) =
            cas_math::hyperbolic_identity_support::try_rewrite_tanh_pythagorean_add_chain(
                ctx,
                simplified_target,
            )
        {
            if matches_target_modulo_simplify(ctx, rewrite.rewritten, expr) {
                return Some(DeriveHyperbolicRewrite {
                    rewritten: target_expr,
                    kind: DeriveHyperbolicRewriteKind::ExpandTanhPythagorean,
                });
            }
        }
    }

    if let Some(rewrite) =
        try_rewrite_negated_sinh_cosh_to_tanh_identity_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_rewrite_negated_hyperbolic_angle_sum_diff_contraction_target_aware(
        ctx,
        expr,
        target_expr,
    ) {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_hyperbolic_angle_sum_diff_contraction_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_negated_tanh_angle_sum_diff_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_mixed_sinh_double_angle_product_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_rewrite_tanh_angle_sum_diff_target_aware(ctx, expr, target_expr) {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_negated_tanh_triple_angle_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_rewrite_tanh_triple_angle_target_aware(ctx, expr, target_expr) {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        cas_math::hyperbolic_identity_support::try_rewrite_sinh_cosh_to_tanh_identity_expr(
            ctx, expr,
        )
    {
        if strong_target_match(ctx, rewrite.rewritten, target_expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: rewrite.rewritten,
                kind: DeriveHyperbolicRewriteKind::SinhCoshToTanh,
            });
        }
    }

    if let Some(rewrite) =
        try_rewrite_negated_tanh_to_sinh_cosh_identity_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        cas_math::hyperbolic_identity_support::try_rewrite_tanh_to_sinh_cosh_identity_expr(
            ctx, expr,
        )
    {
        if strong_target_match(ctx, rewrite.rewritten, target_expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: rewrite.rewritten,
                kind: DeriveHyperbolicRewriteKind::TanhToSinhCosh,
            });
        }
    }

    if let Some(rewrite) =
        cas_math::hyperbolic_identity_support::try_rewrite_sinh_cosh_to_exp(ctx, expr)
    {
        if strong_target_match(ctx, rewrite.rewritten, target_expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: rewrite.rewritten,
                kind: match rewrite.kind {
                    cas_math::hyperbolic_identity_support::SinhCoshToExpRewriteKind::Sum => {
                        DeriveHyperbolicRewriteKind::SinhCoshToExp
                    }
                    cas_math::hyperbolic_identity_support::SinhCoshToExpRewriteKind::CoshMinusSinh => {
                        DeriveHyperbolicRewriteKind::CoshMinusSinhToExpNeg
                    }
                    cas_math::hyperbolic_identity_support::SinhCoshToExpRewriteKind::SinhMinusCosh => {
                        DeriveHyperbolicRewriteKind::SinhMinusCoshToNegExpNeg
                    }
                },
            });
        }
    }

    if let Some(rewrite) =
        cas_math::hyperbolic_identity_support::try_rewrite_sinh_cosh_to_exp(ctx, target_expr)
    {
        if matches_target_modulo_simplify(ctx, rewrite.rewritten, expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: target_expr,
                kind: match rewrite.kind {
                    cas_math::hyperbolic_identity_support::SinhCoshToExpRewriteKind::Sum => {
                        DeriveHyperbolicRewriteKind::ExpandExpToSinhPlusCosh
                    }
                    cas_math::hyperbolic_identity_support::SinhCoshToExpRewriteKind::CoshMinusSinh => {
                        DeriveHyperbolicRewriteKind::ExpandExpNegToCoshMinusSinh
                    }
                    cas_math::hyperbolic_identity_support::SinhCoshToExpRewriteKind::SinhMinusCosh => {
                        DeriveHyperbolicRewriteKind::ExpandNegExpNegToSinhMinusCosh
                    }
                },
            });
        }
    }

    if let Some(rewrite) =
        cas_math::hyperbolic_identity_support::try_rewrite_recognize_hyperbolic_from_exp(ctx, expr)
    {
        if strong_target_match(ctx, rewrite.rewritten, target_expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: rewrite.rewritten,
                kind: match rewrite.kind {
                    cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::CoshHalf => {
                        DeriveHyperbolicRewriteKind::RecognizeCoshFromExp
                    }
                    cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::SinhHalf => {
                        DeriveHyperbolicRewriteKind::RecognizeSinhFromExp
                    }
                    cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::NegSinhHalf => {
                        DeriveHyperbolicRewriteKind::RecognizeNegSinhFromExp
                    }
                    cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::TanhRatio => {
                        DeriveHyperbolicRewriteKind::RecognizeTanhFromExp
                    }
                    cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::NegTanhRatio => {
                        DeriveHyperbolicRewriteKind::RecognizeNegTanhFromExp
                    }
                },
            });
        }
    }

    if let Some(rewrite) = try_rewrite_tanh_exponential_ratio_target_aware(ctx, expr, target_expr) {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_expand_tanh_to_exponential_ratio_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        cas_math::hyperbolic_identity_support::try_rewrite_recognize_hyperbolic_from_exp(
            ctx,
            target_expr,
        )
    {
        if matches_target_modulo_simplify(ctx, rewrite.rewritten, expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: target_expr,
                kind: match rewrite.kind {
                    cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::CoshHalf => {
                        DeriveHyperbolicRewriteKind::ExpandCoshToExpHalfSum
                    }
                    cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::SinhHalf => {
                        DeriveHyperbolicRewriteKind::ExpandSinhToExpHalfDiff
                    }
                    cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::NegSinhHalf => {
                        DeriveHyperbolicRewriteKind::ExpandNegSinhToExpHalfNegDiff
                    }
                    cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::TanhRatio => {
                        DeriveHyperbolicRewriteKind::ExpandTanhToExpRatio
                    }
                    cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::NegTanhRatio => {
                        DeriveHyperbolicRewriteKind::ExpandNegTanhToExpNegRatio
                    }
                },
            });
        }
    }

    if let Some(rewrite) =
        cas_math::trig_half_angle_support::try_rewrite_hyperbolic_half_angle_squares_expr(ctx, expr)
    {
        if matches_target_modulo_simplify(ctx, rewrite.rewritten, target_expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: rewrite.rewritten,
                kind: match rewrite.kind {
                    cas_math::trig_half_angle_support::HalfAngleSquareRewriteKind::HyperbolicCosh => {
                        DeriveHyperbolicRewriteKind::ExpandHyperbolicCoshHalfAngleSquare
                    }
                    cas_math::trig_half_angle_support::HalfAngleSquareRewriteKind::HyperbolicSinh => {
                        DeriveHyperbolicRewriteKind::ExpandHyperbolicSinhHalfAngleSquare
                    }
                    _ => unreachable!("hyperbolic half-angle rewrite should stay hyperbolic"),
                },
            });
        }
    }

    if let Some(rewrite) =
        try_rewrite_negated_hyperbolic_half_angle_square_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        cas_math::trig_half_angle_support::try_rewrite_hyperbolic_half_angle_squares_expr(
            ctx,
            target_expr,
        )
    {
        if matches_target_modulo_simplify(ctx, rewrite.rewritten, expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: target_expr,
                kind: match rewrite.kind {
                    cas_math::trig_half_angle_support::HalfAngleSquareRewriteKind::HyperbolicCosh => {
                        DeriveHyperbolicRewriteKind::ContractHyperbolicCoshHalfAngleSquare
                    }
                    cas_math::trig_half_angle_support::HalfAngleSquareRewriteKind::HyperbolicSinh => {
                        DeriveHyperbolicRewriteKind::ContractHyperbolicSinhHalfAngleSquare
                    }
                    _ => unreachable!("hyperbolic half-angle rewrite should stay hyperbolic"),
                },
            });
        }
    }

    if let Some(rewrite) =
        try_rewrite_negated_cosh_double_angle_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_rewrite_cosh_double_angle_target_aware(ctx, expr, target_expr) {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_mixed_cosh_double_angle_product_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_shifted_cosh_double_angle_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        cas_math::hyperbolic_identity_support::try_rewrite_hyperbolic_double_angle_sum(ctx, expr)
    {
        if strong_target_match(ctx, rewrite.rewritten, target_expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: rewrite.rewritten,
                kind: DeriveHyperbolicRewriteKind::DoubleAngleCoshSum,
            });
        }
    }

    if let Some(rewrite) =
        cas_math::hyperbolic_identity_support::try_rewrite_hyperbolic_double_angle_sub_chain(
            ctx, expr,
        )
    {
        if strong_target_match(ctx, rewrite.rewritten, target_expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: rewrite.rewritten,
                kind: DeriveHyperbolicRewriteKind::DoubleAngleSubChainToZero,
            });
        }
    }

    if let Some(rewrite) =
        try_rewrite_negated_sinh_double_angle_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        try_rewrite_negated_tanh_double_angle_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    if let Some(rewrite) =
        cas_math::hyperbolic_identity_support::try_rewrite_sinh_double_angle_expansion_identity_expr(
            ctx,
            target_expr,
        )
    {
        if strong_target_match(ctx, rewrite.rewritten, expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: target_expr,
                kind: DeriveHyperbolicRewriteKind::ContractSinhDoubleAngle,
            });
        }
    }

    if let Some(rewrite) =
        cas_math::hyperbolic_identity_support::try_rewrite_tanh_double_angle_expansion_identity_expr(
            ctx,
            target_expr,
        )
    {
        if strong_target_match(ctx, rewrite.rewritten, expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: target_expr,
                kind: DeriveHyperbolicRewriteKind::ContractTanhDoubleAngle,
            });
        }
    }

    if let Some(rewrite) =
        cas_math::hyperbolic_identity_support::try_rewrite_hyperbolic_triple_angle(ctx, target_expr)
    {
        if strong_target_match(ctx, rewrite.rewritten, expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: target_expr,
                kind: match rewrite.kind {
                    cas_math::hyperbolic_identity_support::HyperbolicTripleAngleRewriteKind::Sinh => {
                        DeriveHyperbolicRewriteKind::ContractSinhTripleAngle
                    }
                    cas_math::hyperbolic_identity_support::HyperbolicTripleAngleRewriteKind::Cosh => {
                        DeriveHyperbolicRewriteKind::ContractCoshTripleAngle
                    }
                },
            });
        }
    }

    if let Some(rewrite) =
        try_rewrite_negated_hyperbolic_triple_angle_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    None
}

fn try_rewrite_hyperbolic_exponential_bridge_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    generate_hyperbolic_bridge_rewrites(ctx, expr)
        .into_iter()
        .find(|rewrite| {
            rewrite.kind.is_exponential_bridge()
                && strong_target_match(ctx, rewrite.rewritten, target_expr)
        })
}

fn try_rewrite_mixed_sinh_double_angle_product_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    let (mixed_kind, arg) = extract_scaled_mixed_sinh_double_angle_term(ctx, expr, 4)?;
    let two = ctx.num(2);
    let double_arg = smart_mul(ctx, two, arg);
    let sinh_double = ctx.call_builtin(BuiltinFn::Sinh, vec![double_arg]);
    let linear = match mixed_kind {
        BuiltinFn::Sinh => ctx.call_builtin(BuiltinFn::Sinh, vec![arg]),
        BuiltinFn::Cosh => ctx.call_builtin(BuiltinFn::Cosh, vec![arg]),
        _ => return None,
    };
    let inner_product = smart_mul(ctx, sinh_double, linear);
    let candidate = smart_mul(ctx, two, inner_product);

    if !strong_target_match(ctx, candidate, target_expr) {
        return None;
    }

    Some(DeriveHyperbolicRewrite {
        rewritten: target_expr,
        kind: DeriveHyperbolicRewriteKind::ContractSinhDoubleAngle,
    })
}

fn try_rewrite_mixed_cosh_double_angle_product_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    if let Some((linear_builtin, arg)) = extract_scaled_mixed_cosh_double_angle_term(ctx, expr, 2) {
        if let Some(kind) =
            match_mixed_cosh_double_angle_target(ctx, linear_builtin, arg, target_expr)
        {
            return Some(DeriveHyperbolicRewrite {
                rewritten: target_expr,
                kind,
            });
        }
    }

    if let Some((linear_builtin, arg)) =
        extract_scaled_mixed_cosh_double_angle_term(ctx, target_expr, 2)
    {
        if let Some(kind) = match_mixed_cosh_double_angle_target(ctx, linear_builtin, arg, expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: target_expr,
                kind,
            });
        }
    }

    None
}

fn match_mixed_cosh_double_angle_target(
    ctx: &mut cas_ast::Context,
    linear_builtin: BuiltinFn,
    arg: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewriteKind> {
    let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
    let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
    let sinh_sq = pow2(ctx, sinh);
    let cosh_sq = pow2(ctx, cosh);

    match linear_builtin {
        BuiltinFn::Sinh => {
            let mixed_term = smart_mul(ctx, cosh_sq, sinh);
            let mixed_candidate = build_scaled_additive_terms(ctx, &[(4, mixed_term), (-2, sinh)]);
            if strong_target_match(ctx, mixed_candidate, target_expr) {
                return Some(DeriveHyperbolicRewriteKind::ExpandCoshDoubleAngleAsTwoCoshSqMinusOne);
            }

            let cubic_term = pow3(ctx, sinh);
            let cubic_candidate = build_scaled_additive_terms(ctx, &[(2, sinh), (4, cubic_term)]);
            if strong_target_match(ctx, cubic_candidate, target_expr) {
                return Some(DeriveHyperbolicRewriteKind::ExpandCoshDoubleAngleAsOnePlusTwoSinhSq);
            }
        }
        BuiltinFn::Cosh => {
            let mixed_term = smart_mul(ctx, sinh_sq, cosh);
            let mixed_candidate = build_scaled_additive_terms(ctx, &[(2, cosh), (4, mixed_term)]);
            if strong_target_match(ctx, mixed_candidate, target_expr) {
                return Some(DeriveHyperbolicRewriteKind::ExpandCoshDoubleAngleAsOnePlusTwoSinhSq);
            }

            let cubic_term = pow3(ctx, cosh);
            let cubic_candidate = build_scaled_additive_terms(ctx, &[(4, cubic_term), (-2, cosh)]);
            if strong_target_match(ctx, cubic_candidate, target_expr) {
                return Some(DeriveHyperbolicRewriteKind::ExpandCoshDoubleAngleAsTwoCoshSqMinusOne);
            }
        }
        _ => return None,
    }

    None
}

fn try_rewrite_tanh_exponential_ratio_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    let (rewritten, kind) = rewrite_tanh_exponential_ratio_expr(ctx, expr)?;
    if strong_target_match(ctx, rewritten, target_expr) {
        return Some(DeriveHyperbolicRewrite { rewritten, kind });
    }
    None
}

fn try_expand_tanh_to_exponential_ratio_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    let (rewritten, recognize_kind) = rewrite_tanh_exponential_ratio_expr(ctx, target_expr)?;
    if !strong_target_match(ctx, rewritten, expr) {
        return None;
    }

    Some(DeriveHyperbolicRewrite {
        rewritten: target_expr,
        kind: match recognize_kind {
            DeriveHyperbolicRewriteKind::RecognizeTanhFromExp => {
                DeriveHyperbolicRewriteKind::ExpandTanhToExpRatio
            }
            DeriveHyperbolicRewriteKind::RecognizeNegTanhFromExp => {
                DeriveHyperbolicRewriteKind::ExpandNegTanhToExpNegRatio
            }
            _ => return None,
        },
    })
}

fn rewrite_tanh_exponential_ratio_expr(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, DeriveHyperbolicRewriteKind)> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let (num, den) = (*num, *den);

    let (arg, num_pos_sign, num_neg_sign) = extract_exponential_pair_signs(ctx, num)?;
    let (den_arg, den_pos_sign, den_neg_sign) = extract_exponential_pair_signs(ctx, den)?;
    if compare_expr(ctx, arg, den_arg) != Ordering::Equal {
        return None;
    }
    if den_pos_sign != Sign::Pos || den_neg_sign != Sign::Pos {
        return None;
    }

    let tanh = ctx.call_builtin(BuiltinFn::Tanh, vec![arg]);
    match (num_pos_sign, num_neg_sign) {
        (Sign::Pos, Sign::Neg) => Some((tanh, DeriveHyperbolicRewriteKind::RecognizeTanhFromExp)),
        (Sign::Neg, Sign::Pos) => Some((
            ctx.add(Expr::Neg(tanh)),
            DeriveHyperbolicRewriteKind::RecognizeNegTanhFromExp,
        )),
        _ => None,
    }
}

fn extract_exponential_pair_signs(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(ExprId, Sign, Sign)> {
    let terms = add_terms_signed(ctx, expr);
    if terms.len() != 2 {
        return None;
    }

    let first_arg = extract_exp_call_arg(ctx, terms[0].0)?;
    let second_arg = extract_exp_call_arg(ctx, terms[1].0)?;
    if !is_negation(ctx, first_arg, second_arg) {
        return None;
    }

    let canonical_arg = if strip_top_level_negative_factor(ctx, first_arg).is_some() {
        second_arg
    } else {
        first_arg
    };

    let mut positive_arg_sign = None;
    let mut negative_arg_sign = None;
    for (term, sign) in terms {
        let arg = extract_exp_call_arg(ctx, term)?;
        if compare_expr(ctx, arg, canonical_arg) == Ordering::Equal {
            positive_arg_sign = Some(sign);
        } else if is_negation(ctx, arg, canonical_arg) {
            negative_arg_sign = Some(sign);
        }
    }

    Some((canonical_arg, positive_arg_sign?, negative_arg_sign?))
}

pub(crate) fn try_rewrite_hyperbolic_expansion_target_aware(
    ctx: &mut cas_ast::Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewriteKind> {
    if let Some(kind) =
        try_rewrite_hyperbolic_product_sum_target_aware(ctx, source_expr, target_expr)
    {
        return Some(kind);
    }

    if let Some(rewrite) = try_rewrite_negated_hyperbolic_angle_sum_diff_contraction_target_aware(
        ctx,
        target_expr,
        source_expr,
    ) {
        return Some(rewrite.kind);
    }

    if let Some(rewrite) = try_rewrite_hyperbolic_angle_sum_diff_contraction_target_aware(
        ctx,
        target_expr,
        source_expr,
    ) {
        return Some(rewrite.kind);
    }

    None
}

fn try_rewrite_hyperbolic_product_sum_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewriteKind> {
    for rewrite in generate_hyperbolic_bridge_rewrites(ctx, expr) {
        if !matches!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ProductToSumSinhCosh
                | DeriveHyperbolicRewriteKind::ProductToSumCoshCosh
                | DeriveHyperbolicRewriteKind::ProductToSumSinhSinh
                | DeriveHyperbolicRewriteKind::SumToProductSinhCosh
                | DeriveHyperbolicRewriteKind::SumToProductCoshCosh
                | DeriveHyperbolicRewriteKind::SumToProductSinhSinh
        ) {
            continue;
        }

        if presentational_target_match(ctx, rewrite.rewritten, target_expr) {
            return Some(rewrite.kind);
        }
    }

    None
}

fn try_rewrite_negated_hyperbolic_half_angle_square_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    if let Some(positive_expr) = strip_unit_negation(ctx, expr) {
        if let Some(rewrite) =
            cas_math::trig_half_angle_support::try_rewrite_hyperbolic_half_angle_squares_expr(
                ctx,
                positive_expr,
            )
        {
            let negated_rewritten = negate_preserving_fraction_numerator(ctx, rewrite.rewritten);
            if matches_target_modulo_simplify(ctx, negated_rewritten, target_expr) {
                return Some(DeriveHyperbolicRewrite {
                    rewritten: target_expr,
                    kind: match rewrite.kind {
                        cas_math::trig_half_angle_support::HalfAngleSquareRewriteKind::HyperbolicCosh => {
                            DeriveHyperbolicRewriteKind::ExpandHyperbolicCoshHalfAngleSquare
                        }
                        cas_math::trig_half_angle_support::HalfAngleSquareRewriteKind::HyperbolicSinh => {
                            DeriveHyperbolicRewriteKind::ExpandHyperbolicSinhHalfAngleSquare
                        }
                        _ => unreachable!("hyperbolic half-angle rewrite should stay hyperbolic"),
                    },
                });
            }
        }
    }

    if let Some(positive_target) = strip_unit_negation(ctx, target_expr) {
        if let Some(rewrite) =
            cas_math::trig_half_angle_support::try_rewrite_hyperbolic_half_angle_squares_expr(
                ctx,
                positive_target,
            )
        {
            let negated_rewritten = negate_preserving_fraction_numerator(ctx, rewrite.rewritten);
            if matches_target_modulo_simplify(ctx, negated_rewritten, expr) {
                return Some(DeriveHyperbolicRewrite {
                    rewritten: target_expr,
                    kind: match rewrite.kind {
                        cas_math::trig_half_angle_support::HalfAngleSquareRewriteKind::HyperbolicCosh => {
                            DeriveHyperbolicRewriteKind::ContractHyperbolicCoshHalfAngleSquare
                        }
                        cas_math::trig_half_angle_support::HalfAngleSquareRewriteKind::HyperbolicSinh => {
                            DeriveHyperbolicRewriteKind::ContractHyperbolicSinhHalfAngleSquare
                        }
                        _ => unreachable!("hyperbolic half-angle rewrite should stay hyperbolic"),
                    },
                });
            }
        }
    }

    None
}

fn try_rewrite_negated_sinh_double_angle_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    if let Some(positive_target) = strip_unit_negation(ctx, target_expr) {
        if let Some(rewrite) =
            cas_math::hyperbolic_identity_support::try_rewrite_sinh_double_angle_expansion_identity_expr(
                ctx,
                positive_target,
            )
        {
            let negated_rewritten = ctx.add(Expr::Neg(rewrite.rewritten));
            if matches_target_modulo_simplify(ctx, negated_rewritten, expr) {
                return Some(DeriveHyperbolicRewrite {
                    rewritten: target_expr,
                    kind: DeriveHyperbolicRewriteKind::ContractSinhDoubleAngle,
                });
            }
        }
    }

    if let Some(positive_expr) = strip_unit_negation(ctx, expr) {
        if let Some(rewrite) =
            cas_math::hyperbolic_identity_support::try_rewrite_sinh_double_angle_expansion_identity_expr(
                ctx,
                positive_expr,
            )
        {
            let negated_rewritten = ctx.add(Expr::Neg(rewrite.rewritten));
            if matches_target_modulo_simplify(ctx, negated_rewritten, target_expr) {
                return Some(DeriveHyperbolicRewrite {
                    rewritten: target_expr,
                    kind: DeriveHyperbolicRewriteKind::ContractSinhDoubleAngle,
                });
            }
        }
    }

    None
}

fn try_rewrite_negated_tanh_double_angle_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    if let Some(positive_target) = strip_unit_negation(ctx, target_expr) {
        if let Some(rewrite) =
            cas_math::hyperbolic_identity_support::try_rewrite_tanh_double_angle_expansion_identity_expr(
                ctx,
                positive_target,
            )
        {
            let negated_rewritten = negate_preserving_fraction_numerator(ctx, rewrite.rewritten);
            if matches_target_modulo_simplify(ctx, negated_rewritten, expr) {
                return Some(DeriveHyperbolicRewrite {
                    rewritten: target_expr,
                    kind: DeriveHyperbolicRewriteKind::ContractTanhDoubleAngle,
                });
            }
        }
    }

    if let Some(positive_expr) = strip_unit_negation(ctx, expr) {
        if let Some(rewrite) =
            cas_math::hyperbolic_identity_support::try_rewrite_tanh_double_angle_expansion_identity_expr(
                ctx,
                positive_expr,
            )
        {
            let negated_rewritten = negate_preserving_fraction_numerator(ctx, rewrite.rewritten);
            if matches_target_modulo_simplify(ctx, negated_rewritten, target_expr) {
                return Some(DeriveHyperbolicRewrite {
                    rewritten: target_expr,
                    kind: DeriveHyperbolicRewriteKind::ContractTanhDoubleAngle,
                });
            }
        }
    }

    None
}

fn try_rewrite_negated_hyperbolic_triple_angle_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    if let Some(positive_target) = strip_unit_negation(ctx, target_expr) {
        if let Some(rewrite) =
            cas_math::hyperbolic_identity_support::try_rewrite_hyperbolic_triple_angle(
                ctx,
                positive_target,
            )
        {
            let negated_rewritten = ctx.add(Expr::Neg(rewrite.rewritten));
            if matches_target_modulo_simplify(ctx, negated_rewritten, expr) {
                return Some(DeriveHyperbolicRewrite {
                    rewritten: target_expr,
                    kind: match rewrite.kind {
                        cas_math::hyperbolic_identity_support::HyperbolicTripleAngleRewriteKind::Sinh => {
                            DeriveHyperbolicRewriteKind::ContractSinhTripleAngle
                        }
                        cas_math::hyperbolic_identity_support::HyperbolicTripleAngleRewriteKind::Cosh => {
                            DeriveHyperbolicRewriteKind::ContractCoshTripleAngle
                        }
                    },
                });
            }
        }
    }

    if let Some(positive_expr) = strip_unit_negation(ctx, expr) {
        if let Some(rewrite) =
            cas_math::hyperbolic_identity_support::try_rewrite_hyperbolic_triple_angle(
                ctx,
                positive_expr,
            )
        {
            let negated_rewritten = ctx.add(Expr::Neg(rewrite.rewritten));
            if matches_target_modulo_simplify(ctx, negated_rewritten, target_expr) {
                return Some(DeriveHyperbolicRewrite {
                    rewritten: target_expr,
                    kind: match rewrite.kind {
                        cas_math::hyperbolic_identity_support::HyperbolicTripleAngleRewriteKind::Sinh => {
                            DeriveHyperbolicRewriteKind::ContractSinhTripleAngle
                        }
                        cas_math::hyperbolic_identity_support::HyperbolicTripleAngleRewriteKind::Cosh => {
                            DeriveHyperbolicRewriteKind::ContractCoshTripleAngle
                        }
                    },
                });
            }
        }
    }

    None
}

fn try_rewrite_negated_cosh_from_exp_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    if let Some(positive_expr) = strip_top_level_or_div_numerator_negation(ctx, expr) {
        if let Some(rewrite) =
            cas_math::hyperbolic_identity_support::try_rewrite_recognize_hyperbolic_from_exp(
                ctx,
                positive_expr,
            )
        {
            if matches!(
                rewrite.kind,
                cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::CoshHalf
            ) {
                let negated_rewritten = ctx.add(Expr::Neg(rewrite.rewritten));
                if matches_target_modulo_simplify(ctx, negated_rewritten, target_expr) {
                    return Some(DeriveHyperbolicRewrite {
                        rewritten: target_expr,
                        kind: DeriveHyperbolicRewriteKind::RecognizeCoshFromExp,
                    });
                }
            }
        }
    }

    if let Some(positive_target) = strip_top_level_or_div_numerator_negation(ctx, target_expr) {
        if let Some(rewrite) =
            cas_math::hyperbolic_identity_support::try_rewrite_recognize_hyperbolic_from_exp(
                ctx,
                positive_target,
            )
        {
            if matches!(
                rewrite.kind,
                cas_math::hyperbolic_identity_support::RecognizeHyperbolicFromExpRewriteKind::CoshHalf
            ) {
                let positive_expr = strip_unit_negation(ctx, expr)?;
                if matches_target_modulo_simplify(ctx, rewrite.rewritten, positive_expr) {
                    return Some(DeriveHyperbolicRewrite {
                        rewritten: target_expr,
                        kind: DeriveHyperbolicRewriteKind::ExpandCoshToExpHalfSum,
                    });
                }
            }
        }
    }

    None
}

fn try_rewrite_inverse_cosh_square_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    if !cas_math::expr_predicates::is_one_expr(ctx, *numerator) {
        return None;
    }

    let Expr::Pow(base, exponent) = ctx.get(*denominator) else {
        return None;
    };
    if !cas_math::expr_predicates::is_two_expr(ctx, *exponent) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(*base) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Cosh) || args.len() != 1 {
        return None;
    }

    let tanh = ctx.call_builtin(BuiltinFn::Tanh, vec![args[0]]);
    let tanh_sq = pow2(ctx, tanh);
    let one = ctx.num(1);
    let candidate = ctx.add(Expr::Sub(one, tanh_sq));
    if !matches_target_modulo_simplify(ctx, candidate, target_expr) {
        return None;
    }

    Some(DeriveHyperbolicRewrite {
        rewritten: target_expr,
        kind: DeriveHyperbolicRewriteKind::ExpandTanhPythagorean,
    })
}

fn try_rewrite_negated_tanh_pythagorean_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    let Expr::Sub(left, right) = ctx.get(expr) else {
        return None;
    };
    if !cas_math::expr_predicates::is_one_expr(ctx, *right) {
        return None;
    }

    let Expr::Pow(base, exponent) = ctx.get(*left) else {
        return None;
    };
    if !cas_math::expr_predicates::is_two_expr(ctx, *exponent) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(*base) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Tanh) || args.len() != 1 {
        return None;
    }

    let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![args[0]]);
    let one = ctx.num(1);
    let cosh_sq = pow2(ctx, cosh);
    let positive_reciprocal = ctx.add(Expr::Div(one, cosh_sq));
    let candidate = negate_preserving_fraction_numerator(ctx, positive_reciprocal);
    if !matches_target_modulo_simplify(ctx, candidate, target_expr) {
        return None;
    }

    Some(DeriveHyperbolicRewrite {
        rewritten: target_expr,
        kind: DeriveHyperbolicRewriteKind::TanhPythagorean,
    })
}

fn try_rewrite_tanh_pythagorean_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    let Expr::Sub(left, right) = ctx.get(expr) else {
        return None;
    };
    if !cas_math::expr_predicates::is_one_expr(ctx, *left) {
        return None;
    }

    let Expr::Pow(base, exponent) = ctx.get(*right) else {
        return None;
    };
    if !cas_math::expr_predicates::is_two_expr(ctx, *exponent) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(*base) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Tanh) || args.len() != 1 {
        return None;
    }

    let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![args[0]]);
    let one = ctx.num(1);
    let cosh_sq = pow2(ctx, cosh);
    let candidate = ctx.add(Expr::Div(one, cosh_sq));
    if !matches_target_modulo_simplify(ctx, candidate, target_expr) {
        return None;
    }

    Some(DeriveHyperbolicRewrite {
        rewritten: target_expr,
        kind: DeriveHyperbolicRewriteKind::TanhPythagorean,
    })
}

fn try_rewrite_negated_inverse_cosh_square_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    let positive_expr = strip_top_level_or_div_numerator_negation(ctx, expr)?;

    let Expr::Div(numerator, denominator) = ctx.get(positive_expr) else {
        return None;
    };
    if !cas_math::expr_predicates::is_one_expr(ctx, *numerator) {
        return None;
    }

    let Expr::Pow(base, exponent) = ctx.get(*denominator) else {
        return None;
    };
    if !cas_math::expr_predicates::is_two_expr(ctx, *exponent) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(*base) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Cosh) || args.len() != 1 {
        return None;
    }

    let tanh = ctx.call_builtin(BuiltinFn::Tanh, vec![args[0]]);
    let tanh_sq = pow2(ctx, tanh);
    let one = ctx.num(1);
    let candidate = ctx.add(Expr::Sub(tanh_sq, one));
    if !matches_target_modulo_simplify(ctx, candidate, target_expr) {
        return None;
    }

    Some(DeriveHyperbolicRewrite {
        rewritten: target_expr,
        kind: DeriveHyperbolicRewriteKind::ExpandTanhPythagorean,
    })
}

fn try_rewrite_hyperbolic_pythagorean_factor_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    let target = run_default_simplify(ctx, target_expr);
    let one = ctx.num(1);

    let as_hyperbolic_square =
        |ctx: &cas_ast::Context, expr: ExprId| -> Option<(BuiltinFn, ExprId)> {
            let Expr::Pow(base, exponent) = ctx.get(expr) else {
                return None;
            };
            if !cas_math::expr_predicates::is_two_expr(ctx, *exponent) {
                return None;
            }
            let Expr::Function(fn_id, args) = ctx.get(*base) else {
                return None;
            };
            if args.len() != 1 {
                return None;
            }
            let builtin = ctx.builtin_of(*fn_id)?;
            match builtin {
                BuiltinFn::Sinh | BuiltinFn::Cosh => Some((builtin, args[0])),
                _ => None,
            }
        };

    if let Expr::Sub(left, right) = ctx.get(expr) {
        if cas_math::expr_predicates::is_one_expr(ctx, *right) {
            if let Some((BuiltinFn::Cosh, arg)) = as_hyperbolic_square(ctx, *left) {
                let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
                let sinh_sq = pow2(ctx, sinh);
                if matches_target_modulo_simplify(ctx, sinh_sq, target_expr) {
                    return Some(DeriveHyperbolicRewrite {
                        rewritten: target_expr,
                        kind:
                            DeriveHyperbolicRewriteKind::RecognizeCoshSquaredMinusOneAsSinhSquared,
                    });
                }
            }
        }
    }

    if let Some((BuiltinFn::Sinh, arg)) = as_hyperbolic_square(ctx, expr) {
        let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
        let cosh_sq = pow2(ctx, cosh);
        let candidate = ctx.add(Expr::Sub(cosh_sq, one));
        if matches_target_modulo_simplify(ctx, candidate, target_expr)
            || matches_target_modulo_simplify(ctx, candidate, target)
        {
            return Some(DeriveHyperbolicRewrite {
                rewritten: target_expr,
                kind: DeriveHyperbolicRewriteKind::ExpandSinhSquaredToCoshSquaredMinusOne,
            });
        }
    }

    if let Expr::Add(left, right) = ctx.get(expr) {
        let (square_term, one_term) = if cas_math::expr_predicates::is_one_expr(ctx, *left) {
            (*right, *left)
        } else if cas_math::expr_predicates::is_one_expr(ctx, *right) {
            (*left, *right)
        } else {
            (expr, expr)
        };

        if square_term != expr && cas_math::expr_predicates::is_one_expr(ctx, one_term) {
            if let Some((BuiltinFn::Sinh, arg)) = as_hyperbolic_square(ctx, square_term) {
                let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
                let cosh_sq = pow2(ctx, cosh);
                if matches_target_modulo_simplify(ctx, cosh_sq, target_expr) {
                    return Some(DeriveHyperbolicRewrite {
                        rewritten: target_expr,
                        kind: DeriveHyperbolicRewriteKind::RecognizeOnePlusSinhSquaredAsCoshSquared,
                    });
                }
            }
        }
    }

    if let Some((BuiltinFn::Cosh, arg)) = as_hyperbolic_square(ctx, expr) {
        let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
        let sinh_sq = pow2(ctx, sinh);
        let candidate = ctx.add(Expr::Add(one, sinh_sq));
        if matches_target_modulo_simplify(ctx, candidate, target_expr)
            || matches_target_modulo_simplify(ctx, candidate, target)
        {
            return Some(DeriveHyperbolicRewrite {
                rewritten: target_expr,
                kind: DeriveHyperbolicRewriteKind::ExpandCoshSquaredToOnePlusSinhSquared,
            });
        }
    }

    None
}

fn try_rewrite_negated_sinh_cosh_to_tanh_identity_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    let positive_expr = strip_top_level_or_div_numerator_negation(ctx, expr)?;
    let rewrite =
        cas_math::hyperbolic_identity_support::try_rewrite_sinh_cosh_to_tanh_identity_expr(
            ctx,
            positive_expr,
        )?;
    let rewritten = ctx.add(Expr::Neg(rewrite.rewritten));
    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    Some(DeriveHyperbolicRewrite {
        rewritten,
        kind: DeriveHyperbolicRewriteKind::SinhCoshToTanh,
    })
}

fn try_rewrite_hyperbolic_angle_sum_diff_contraction_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    let (rewritten, kind) =
        rewrite_hyperbolic_angle_sum_diff_contraction_expr(ctx, expr, target_expr, false)?;
    Some(DeriveHyperbolicRewrite { rewritten, kind })
}

fn try_rewrite_negated_hyperbolic_angle_sum_diff_contraction_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    let positive_expr = strip_unit_negation(ctx, expr)?;
    let (rewritten, kind) =
        rewrite_hyperbolic_angle_sum_diff_contraction_expr(ctx, positive_expr, target_expr, true)?;
    Some(DeriveHyperbolicRewrite { rewritten, kind })
}

fn rewrite_hyperbolic_angle_sum_diff_contraction_expr(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
    negate_output: bool,
) -> Option<(ExprId, DeriveHyperbolicRewriteKind)> {
    let try_candidate = |ctx: &mut cas_ast::Context,
                         candidate: ExprId,
                         target_expr: ExprId,
                         kind: DeriveHyperbolicRewriteKind| {
        let candidate = if negate_output {
            ctx.add(Expr::Neg(candidate))
        } else {
            candidate
        };
        if matches_target_modulo_simplify(ctx, candidate, target_expr) {
            Some((target_expr, kind))
        } else {
            None
        }
    };

    match ctx.get(expr) {
        Expr::Add(left, right) => {
            let left_pattern = extract_hyperbolic_two_factor_product(ctx, *left)?;
            let right_pattern = extract_hyperbolic_two_factor_product(ctx, *right)?;

            if let (
                HyperbolicTwoFactorPattern::SinhCosh(a, b),
                HyperbolicTwoFactorPattern::SinhCosh(c, d),
            ) = (left_pattern, right_pattern)
            {
                if compare_expr(ctx, a, d) == Ordering::Equal
                    && compare_expr(ctx, b, c) == Ordering::Equal
                {
                    let sum = ctx.add(Expr::Add(a, b));
                    let candidate = ctx.call_builtin(BuiltinFn::Sinh, vec![sum]);
                    if let Some(result) = try_candidate(
                        ctx,
                        candidate,
                        target_expr,
                        DeriveHyperbolicRewriteKind::ContractSinhAngleSumDiff,
                    ) {
                        return Some(result);
                    }
                }
            }

            let cosh_sinh_pair = match (left_pattern, right_pattern) {
                (
                    HyperbolicTwoFactorPattern::CoshCosh(a, b),
                    HyperbolicTwoFactorPattern::SinhSinh(c, d),
                )
                | (
                    HyperbolicTwoFactorPattern::SinhSinh(c, d),
                    HyperbolicTwoFactorPattern::CoshCosh(a, b),
                ) => Some((a, b, c, d)),
                _ => None,
            }?;

            let (a, b, c, d) = cosh_sinh_pair;
            if same_unordered_pair(ctx, a, b, c, d) {
                let sum = ctx.add(Expr::Add(a, b));
                let candidate = ctx.call_builtin(BuiltinFn::Cosh, vec![sum]);
                return try_candidate(
                    ctx,
                    candidate,
                    target_expr,
                    DeriveHyperbolicRewriteKind::ContractCoshAngleSumDiff,
                );
            }

            None
        }
        Expr::Sub(left, right) => {
            let left_pattern = extract_hyperbolic_two_factor_product(ctx, *left)?;
            let right_pattern = extract_hyperbolic_two_factor_product(ctx, *right)?;

            if let (
                HyperbolicTwoFactorPattern::SinhCosh(a, b),
                HyperbolicTwoFactorPattern::SinhCosh(c, d),
            ) = (left_pattern, right_pattern)
            {
                if compare_expr(ctx, a, d) == Ordering::Equal
                    && compare_expr(ctx, b, c) == Ordering::Equal
                {
                    let diff = ctx.add(Expr::Sub(a, b));
                    let candidate = ctx.call_builtin(BuiltinFn::Sinh, vec![diff]);
                    if let Some(result) = try_candidate(
                        ctx,
                        candidate,
                        target_expr,
                        DeriveHyperbolicRewriteKind::ContractSinhAngleSumDiff,
                    ) {
                        return Some(result);
                    }
                }
            }

            if let (
                HyperbolicTwoFactorPattern::CoshCosh(a, b),
                HyperbolicTwoFactorPattern::SinhSinh(c, d),
            ) = (left_pattern, right_pattern)
            {
                if same_unordered_pair(ctx, a, b, c, d) {
                    let diff_ab = ctx.add(Expr::Sub(a, b));
                    let candidate_ab = ctx.call_builtin(BuiltinFn::Cosh, vec![diff_ab]);
                    if let Some(result) = try_candidate(
                        ctx,
                        candidate_ab,
                        target_expr,
                        DeriveHyperbolicRewriteKind::ContractCoshAngleSumDiff,
                    ) {
                        return Some(result);
                    }

                    let diff_ba = ctx.add(Expr::Sub(b, a));
                    let candidate_ba = ctx.call_builtin(BuiltinFn::Cosh, vec![diff_ba]);
                    return try_candidate(
                        ctx,
                        candidate_ba,
                        target_expr,
                        DeriveHyperbolicRewriteKind::ContractCoshAngleSumDiff,
                    );
                }
            }

            None
        }
        _ => None,
    }
}

fn rewrite_hyperbolic_angle_sum_diff_bridge_expr(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    negate_output: bool,
) -> Option<(ExprId, DeriveHyperbolicRewriteKind)> {
    let finalize =
        |ctx: &mut cas_ast::Context, candidate: ExprId, kind: DeriveHyperbolicRewriteKind| {
            let rewritten = if negate_output {
                ctx.add(Expr::Neg(candidate))
            } else {
                candidate
            };
            Some((rewritten, kind))
        };

    match ctx.get(expr) {
        Expr::Add(left, right) => {
            let left_pattern = extract_hyperbolic_two_factor_product(ctx, *left)?;
            let right_pattern = extract_hyperbolic_two_factor_product(ctx, *right)?;

            if let (
                HyperbolicTwoFactorPattern::SinhCosh(a, b),
                HyperbolicTwoFactorPattern::SinhCosh(c, d),
            ) = (left_pattern, right_pattern)
            {
                if compare_expr(ctx, a, d) == Ordering::Equal
                    && compare_expr(ctx, b, c) == Ordering::Equal
                {
                    let sum = ctx.add(Expr::Add(a, b));
                    let candidate = ctx.call_builtin(BuiltinFn::Sinh, vec![sum]);
                    return finalize(
                        ctx,
                        candidate,
                        DeriveHyperbolicRewriteKind::ContractSinhAngleSumDiff,
                    );
                }
            }

            let cosh_sinh_pair = match (left_pattern, right_pattern) {
                (
                    HyperbolicTwoFactorPattern::CoshCosh(a, b),
                    HyperbolicTwoFactorPattern::SinhSinh(c, d),
                )
                | (
                    HyperbolicTwoFactorPattern::SinhSinh(c, d),
                    HyperbolicTwoFactorPattern::CoshCosh(a, b),
                ) => Some((a, b, c, d)),
                _ => None,
            }?;

            let (a, b, c, d) = cosh_sinh_pair;
            if same_unordered_pair(ctx, a, b, c, d) {
                let sum = ctx.add(Expr::Add(a, b));
                let candidate = ctx.call_builtin(BuiltinFn::Cosh, vec![sum]);
                return finalize(
                    ctx,
                    candidate,
                    DeriveHyperbolicRewriteKind::ContractCoshAngleSumDiff,
                );
            }

            None
        }
        Expr::Sub(left, right) => {
            let left_pattern = extract_hyperbolic_two_factor_product(ctx, *left)?;
            let right_pattern = extract_hyperbolic_two_factor_product(ctx, *right)?;

            if let (
                HyperbolicTwoFactorPattern::SinhCosh(a, b),
                HyperbolicTwoFactorPattern::SinhCosh(c, d),
            ) = (left_pattern, right_pattern)
            {
                if compare_expr(ctx, a, d) == Ordering::Equal
                    && compare_expr(ctx, b, c) == Ordering::Equal
                {
                    let diff = ctx.add(Expr::Sub(a, b));
                    let candidate = ctx.call_builtin(BuiltinFn::Sinh, vec![diff]);
                    return finalize(
                        ctx,
                        candidate,
                        DeriveHyperbolicRewriteKind::ContractSinhAngleSumDiff,
                    );
                }
            }

            if let (
                HyperbolicTwoFactorPattern::CoshCosh(a, b),
                HyperbolicTwoFactorPattern::SinhSinh(c, d),
            ) = (left_pattern, right_pattern)
            {
                if same_unordered_pair(ctx, a, b, c, d) {
                    let (lhs, rhs) = if compare_expr(ctx, a, b) == Ordering::Greater {
                        (b, a)
                    } else {
                        (a, b)
                    };
                    let diff = ctx.add(Expr::Sub(lhs, rhs));
                    let candidate = ctx.call_builtin(BuiltinFn::Cosh, vec![diff]);
                    return finalize(
                        ctx,
                        candidate,
                        DeriveHyperbolicRewriteKind::ContractCoshAngleSumDiff,
                    );
                }
            }

            None
        }
        _ => None,
    }
}

fn try_rewrite_negated_tanh_to_sinh_cosh_identity_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    let positive_expr = strip_unit_negation(ctx, expr)?;
    let rewrite =
        cas_math::hyperbolic_identity_support::try_rewrite_tanh_to_sinh_cosh_identity_expr(
            ctx,
            positive_expr,
        )?;
    let rewritten = negate_preserving_fraction_numerator(ctx, rewrite.rewritten);
    if !strong_target_match(ctx, rewritten, target_expr) {
        return None;
    }

    Some(DeriveHyperbolicRewrite {
        rewritten,
        kind: DeriveHyperbolicRewriteKind::TanhToSinhCosh,
    })
}

fn try_rewrite_tanh_angle_sum_diff_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    if let Some(candidate) = expand_tanh_angle_sum_diff_expr(ctx, expr) {
        if matches_target_modulo_simplify(ctx, candidate, target_expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: target_expr,
                kind: DeriveHyperbolicRewriteKind::ExpandTanhAngleSumDiff,
            });
        }
    }

    if let Some(candidate) = contract_tanh_angle_sum_diff_expr(ctx, expr) {
        if matches_target_modulo_simplify(ctx, candidate, target_expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: target_expr,
                kind: DeriveHyperbolicRewriteKind::ContractTanhAngleSumDiff,
            });
        }
    }

    None
}

fn try_rewrite_negated_tanh_angle_sum_diff_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    let positive_expr = strip_top_level_or_div_numerator_negation(ctx, expr)?;

    if let Some(candidate) = expand_tanh_angle_sum_diff_expr(ctx, positive_expr) {
        let negated_candidate = negate_preserving_fraction_numerator(ctx, candidate);
        if matches_target_modulo_simplify(ctx, negated_candidate, target_expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: target_expr,
                kind: DeriveHyperbolicRewriteKind::ExpandTanhAngleSumDiff,
            });
        }
    }

    if let Some(candidate) = contract_tanh_angle_sum_diff_expr(ctx, positive_expr) {
        let negated_candidate = ctx.add(Expr::Neg(candidate));
        if matches_target_modulo_simplify(ctx, negated_candidate, target_expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: target_expr,
                kind: DeriveHyperbolicRewriteKind::ContractTanhAngleSumDiff,
            });
        }
    }

    None
}

fn try_rewrite_tanh_triple_angle_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    if let Some(candidate) = expand_tanh_triple_angle_expr(ctx, expr) {
        if matches_target_modulo_simplify(ctx, candidate, target_expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: target_expr,
                kind: DeriveHyperbolicRewriteKind::ExpandTanhTripleAngle,
            });
        }
    }

    if let Some(candidate) = contract_tanh_triple_angle_expr(ctx, expr) {
        if matches_target_modulo_simplify(ctx, candidate, target_expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: target_expr,
                kind: DeriveHyperbolicRewriteKind::ContractTanhTripleAngle,
            });
        }
    }

    None
}

fn try_rewrite_negated_tanh_triple_angle_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    let positive_expr = strip_top_level_or_div_numerator_negation(ctx, expr)?;

    if let Some(candidate) = expand_tanh_triple_angle_expr(ctx, positive_expr) {
        let negated_candidate = negate_preserving_fraction_numerator(ctx, candidate);
        if matches_target_modulo_simplify(ctx, negated_candidate, target_expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: target_expr,
                kind: DeriveHyperbolicRewriteKind::ExpandTanhTripleAngle,
            });
        }
    }

    if let Some(candidate) = contract_tanh_triple_angle_expr(ctx, positive_expr) {
        let negated_candidate = ctx.add(Expr::Neg(candidate));
        if matches_target_modulo_simplify(ctx, negated_candidate, target_expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: target_expr,
                kind: DeriveHyperbolicRewriteKind::ContractTanhTripleAngle,
            });
        }
    }

    None
}

fn try_rewrite_negated_cosh_double_angle_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    if let Some(positive_expr) = strip_unit_negation(ctx, expr) {
        let Expr::Function(fn_id, args) = ctx.get(positive_expr) else {
            return None;
        };
        if !ctx.is_builtin(*fn_id, BuiltinFn::Cosh) || args.len() != 1 {
            return None;
        }

        let inner = cas_math::trig_roots_flatten::extract_double_angle_arg_relaxed(ctx, args[0])?;
        let cosh_inner = ctx.call_builtin(BuiltinFn::Cosh, vec![inner]);
        let sinh_inner = ctx.call_builtin(BuiltinFn::Sinh, vec![inner]);
        let cosh_sq = pow2(ctx, cosh_inner);
        let sinh_sq = pow2(ctx, sinh_inner);
        let one = ctx.num(1);
        let neg_one = ctx.num(-1);
        let two = ctx.num(2);

        let two_cosh_sq = ctx.add(Expr::Mul(two, cosh_sq));
        let one_minus_two_cosh_sq = ctx.add(Expr::Sub(one, two_cosh_sq));
        if matches_target_modulo_simplify(ctx, one_minus_two_cosh_sq, target_expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: one_minus_two_cosh_sq,
                kind: DeriveHyperbolicRewriteKind::ExpandNegativeCoshDoubleAngleAsOneMinusTwoCoshSq,
            });
        }

        let two_sinh_sq = ctx.add(Expr::Mul(two, sinh_sq));
        let negative_one_minus_two_sinh_sq = ctx.add(Expr::Sub(neg_one, two_sinh_sq));
        if matches_target_modulo_simplify(ctx, negative_one_minus_two_sinh_sq, target_expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: negative_one_minus_two_sinh_sq,
                kind: DeriveHyperbolicRewriteKind::ExpandNegativeCoshDoubleAngleAsNegativeOneMinusTwoSinhSq,
            });
        }
    }

    if let Expr::Sub(left, right) = ctx.get(expr) {
        let (left, right) = (*left, *right);

        if cas_math::expr_predicates::is_one_expr(ctx, left) {
            if let Some((BuiltinFn::Cosh, arg)) = scaled_hyperbolic_square(ctx, right, 2) {
                let two = ctx.num(2);
                let two_arg = ctx.add(Expr::Mul(two, arg));
                let cosh_two_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![two_arg]);
                let candidate = ctx.add(Expr::Neg(cosh_two_arg));
                if matches_target_modulo_simplify(ctx, candidate, target_expr) {
                    return Some(DeriveHyperbolicRewrite {
                        rewritten: target_expr,
                        kind: DeriveHyperbolicRewriteKind::RecognizeOneMinusTwoCoshSqAsNegativeCoshDoubleAngle,
                    });
                }
            }
        }

        if is_small_integer(ctx, left, -1) {
            if let Some((BuiltinFn::Sinh, arg)) = scaled_hyperbolic_square(ctx, right, 2) {
                let two = ctx.num(2);
                let two_arg = ctx.add(Expr::Mul(two, arg));
                let cosh_two_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![two_arg]);
                let candidate = ctx.add(Expr::Neg(cosh_two_arg));
                if matches_target_modulo_simplify(ctx, candidate, target_expr) {
                    return Some(DeriveHyperbolicRewrite {
                        rewritten: target_expr,
                        kind: DeriveHyperbolicRewriteKind::RecognizeNegativeOneMinusTwoSinhSqAsNegativeCoshDoubleAngle,
                    });
                }
            }
        }
    }

    None
}

fn try_rewrite_cosh_double_angle_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Cosh) || args.len() != 1 {
        return None;
    }

    let inner = cas_math::trig_roots_flatten::extract_double_angle_arg_relaxed(ctx, args[0])?;
    let cosh_inner = ctx.call_builtin(BuiltinFn::Cosh, vec![inner]);
    let sinh_inner = ctx.call_builtin(BuiltinFn::Sinh, vec![inner]);
    let cosh_sq = pow2(ctx, cosh_inner);
    let sinh_sq = pow2(ctx, sinh_inner);
    let one = ctx.num(1);
    let two = ctx.num(2);

    let cosh_sum = ctx.add(Expr::Add(cosh_sq, sinh_sq));
    if matches_target_modulo_simplify(ctx, cosh_sum, target_expr) {
        return Some(DeriveHyperbolicRewrite {
            rewritten: cosh_sum,
            kind: DeriveHyperbolicRewriteKind::ExpandCoshDoubleAngleAsSum,
        });
    }

    let two_cosh_sq = ctx.add(Expr::Mul(two, cosh_sq));
    let two_cosh_sq_minus_one = ctx.add(Expr::Sub(two_cosh_sq, one));
    if matches_target_modulo_simplify(ctx, two_cosh_sq_minus_one, target_expr) {
        return Some(DeriveHyperbolicRewrite {
            rewritten: two_cosh_sq_minus_one,
            kind: DeriveHyperbolicRewriteKind::ExpandCoshDoubleAngleAsTwoCoshSqMinusOne,
        });
    }

    let two_sinh_sq = ctx.add(Expr::Mul(two, sinh_sq));
    let one_plus_two_sinh_sq = ctx.add(Expr::Add(one, two_sinh_sq));
    if matches_target_modulo_simplify(ctx, one_plus_two_sinh_sq, target_expr) {
        return Some(DeriveHyperbolicRewrite {
            rewritten: one_plus_two_sinh_sq,
            kind: DeriveHyperbolicRewriteKind::ExpandCoshDoubleAngleAsOnePlusTwoSinhSq,
        });
    }

    None
}

fn try_rewrite_shifted_cosh_double_angle_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveHyperbolicRewrite> {
    let one = ctx.num(1);
    let two = ctx.num(2);

    if let Expr::Sub(left, right) = ctx.get(expr) {
        let (left, right) = (*left, *right);

        if cas_math::expr_predicates::is_one_expr(ctx, left) {
            if let Expr::Function(fn_id, args) = ctx.get(right) {
                if ctx.is_builtin(*fn_id, BuiltinFn::Cosh) && args.len() == 1 {
                    let inner = cas_math::trig_roots_flatten::extract_double_angle_arg_relaxed(
                        ctx, args[0],
                    )?;
                    let sinh_inner = ctx.call_builtin(BuiltinFn::Sinh, vec![inner]);
                    let sinh_sq = pow2(ctx, sinh_inner);
                    let neg_two = ctx.num(-2);
                    let candidate = ctx.add(Expr::Mul(neg_two, sinh_sq));
                    if matches_target_modulo_simplify(ctx, candidate, target_expr) {
                        return Some(DeriveHyperbolicRewrite {
                            rewritten: target_expr,
                            kind: DeriveHyperbolicRewriteKind::RecognizeOneMinusCoshDoubleAngleAsNegativeTwoSinhSq,
                        });
                    }
                }
            }
        }

        if is_small_integer(ctx, left, -1) {
            if let Expr::Function(fn_id, args) = ctx.get(right) {
                if ctx.is_builtin(*fn_id, BuiltinFn::Cosh) && args.len() == 1 {
                    let inner = cas_math::trig_roots_flatten::extract_double_angle_arg_relaxed(
                        ctx, args[0],
                    )?;
                    let cosh_inner = ctx.call_builtin(BuiltinFn::Cosh, vec![inner]);
                    let cosh_sq = pow2(ctx, cosh_inner);
                    let neg_two = ctx.num(-2);
                    let candidate = ctx.add(Expr::Mul(neg_two, cosh_sq));
                    if matches_target_modulo_simplify(ctx, candidate, target_expr) {
                        return Some(DeriveHyperbolicRewrite {
                            rewritten: target_expr,
                            kind: DeriveHyperbolicRewriteKind::RecognizeNegativeOneMinusCoshDoubleAngleAsNegativeTwoCoshSq,
                        });
                    }
                }
            }
        }

        if cas_math::expr_predicates::is_one_expr(ctx, right) {
            if let Expr::Function(fn_id, args) = ctx.get(left) {
                if ctx.is_builtin(*fn_id, BuiltinFn::Cosh) && args.len() == 1 {
                    let inner = cas_math::trig_roots_flatten::extract_double_angle_arg_relaxed(
                        ctx, args[0],
                    )?;
                    let sinh_inner = ctx.call_builtin(BuiltinFn::Sinh, vec![inner]);
                    let sinh_sq = pow2(ctx, sinh_inner);
                    let candidate = ctx.add(Expr::Mul(two, sinh_sq));
                    if matches_target_modulo_simplify(ctx, candidate, target_expr) {
                        return Some(DeriveHyperbolicRewrite {
                            rewritten: target_expr,
                            kind: DeriveHyperbolicRewriteKind::RecognizeCoshDoubleAngleMinusOneAsTwoSinhSq,
                        });
                    }
                }
            }

            if let Some((BuiltinFn::Cosh, arg)) = scaled_hyperbolic_square(ctx, left, 2) {
                let two_arg = ctx.add(Expr::Mul(two, arg));
                let candidate = ctx.call_builtin(BuiltinFn::Cosh, vec![two_arg]);
                if matches_target_modulo_simplify(ctx, candidate, target_expr) {
                    return Some(DeriveHyperbolicRewrite {
                        rewritten: target_expr,
                        kind:
                            DeriveHyperbolicRewriteKind::RecognizeTwoCoshSqMinusOneAsCoshDoubleAngle,
                    });
                }
            }
        }
    }

    if let Expr::Add(left, right) = ctx.get(expr) {
        let (cosh_term, one_term) = if cas_math::expr_predicates::is_one_expr(ctx, *left) {
            (*right, *left)
        } else if cas_math::expr_predicates::is_one_expr(ctx, *right) {
            (*left, *right)
        } else {
            return None;
        };

        if cas_math::expr_predicates::is_one_expr(ctx, one_term) {
            if let Expr::Function(fn_id, args) = ctx.get(cosh_term) {
                if ctx.is_builtin(*fn_id, BuiltinFn::Cosh) && args.len() == 1 {
                    let inner = cas_math::trig_roots_flatten::extract_double_angle_arg_relaxed(
                        ctx, args[0],
                    )?;
                    let cosh_inner = ctx.call_builtin(BuiltinFn::Cosh, vec![inner]);
                    let cosh_sq = pow2(ctx, cosh_inner);
                    let candidate = ctx.add(Expr::Mul(two, cosh_sq));
                    if matches_target_modulo_simplify(ctx, candidate, target_expr) {
                        return Some(DeriveHyperbolicRewrite {
                            rewritten: target_expr,
                            kind: DeriveHyperbolicRewriteKind::RecognizeCoshDoubleAnglePlusOneAsTwoCoshSq,
                        });
                    }
                }
            }

            if let Some((BuiltinFn::Sinh, arg)) = scaled_hyperbolic_square(ctx, cosh_term, 2) {
                let two_arg = ctx.add(Expr::Mul(two, arg));
                let candidate = ctx.call_builtin(BuiltinFn::Cosh, vec![two_arg]);
                if matches_target_modulo_simplify(ctx, candidate, target_expr) {
                    return Some(DeriveHyperbolicRewrite {
                        rewritten: target_expr,
                        kind:
                            DeriveHyperbolicRewriteKind::RecognizeTwoSinhSqPlusOneAsCoshDoubleAngle,
                    });
                }
            }
        }
    }

    if let Some((BuiltinFn::Cosh, arg)) = scaled_hyperbolic_square(ctx, expr, -2) {
        let two_arg = ctx.add(Expr::Mul(two, arg));
        let cosh_two_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![two_arg]);
        let neg_one = ctx.num(-1);
        let candidate = ctx.add(Expr::Sub(neg_one, cosh_two_arg));
        if matches_target_modulo_simplify(ctx, candidate, target_expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: target_expr,
                kind: DeriveHyperbolicRewriteKind::ExpandNegativeTwoCoshSqToNegativeOneMinusCoshDoubleAngle,
            });
        }
    }

    if let Some((BuiltinFn::Sinh, arg)) = scaled_hyperbolic_square(ctx, expr, -2) {
        let two_arg = ctx.add(Expr::Mul(two, arg));
        let cosh_two_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![two_arg]);
        let candidate = ctx.add(Expr::Sub(one, cosh_two_arg));
        if matches_target_modulo_simplify(ctx, candidate, target_expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: target_expr,
                kind: DeriveHyperbolicRewriteKind::ExpandNegativeTwoSinhSqToOneMinusCoshDoubleAngle,
            });
        }
    }

    if let Some((BuiltinFn::Sinh, arg)) = scaled_hyperbolic_square(ctx, expr, 2) {
        let two_arg = ctx.add(Expr::Mul(two, arg));
        let cosh_two_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![two_arg]);
        let candidate = ctx.add(Expr::Sub(cosh_two_arg, one));
        if matches_target_modulo_simplify(ctx, candidate, target_expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: target_expr,
                kind: DeriveHyperbolicRewriteKind::ExpandTwoSinhSqToCoshDoubleAngleMinusOne,
            });
        }
    }

    if let Some((BuiltinFn::Cosh, arg)) = scaled_hyperbolic_square(ctx, expr, 2) {
        let two_arg = ctx.add(Expr::Mul(two, arg));
        let cosh_two_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![two_arg]);
        let candidate = ctx.add(Expr::Add(cosh_two_arg, one));
        if matches_target_modulo_simplify(ctx, candidate, target_expr) {
            return Some(DeriveHyperbolicRewrite {
                rewritten: target_expr,
                kind: DeriveHyperbolicRewriteKind::ExpandTwoCoshSqToCoshDoubleAnglePlusOne,
            });
        }
    }

    None
}

fn scaled_hyperbolic_square(
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

    let Expr::Pow(base, exponent) = ctx.get(squared) else {
        return None;
    };
    if !cas_math::expr_predicates::is_two_expr(ctx, *exponent) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(*base) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let builtin = ctx.builtin_of(*fn_id)?;
    match builtin {
        BuiltinFn::Sinh | BuiltinFn::Cosh => Some((builtin, args[0])),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HyperbolicTwoFactorPattern {
    SinhCosh(ExprId, ExprId),
    CoshCosh(ExprId, ExprId),
    SinhSinh(ExprId, ExprId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScaledHyperbolicProductPattern {
    SinhCosh(ExprId, ExprId),
    CoshCosh(ExprId, ExprId),
    SinhSinh(ExprId, ExprId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CombinedHyperbolicTripleAngleData {
    kind: DeriveHyperbolicRewriteKind,
    hyperbolic_fn: BuiltinFn,
    base_arg: ExprId,
    base_coeff: i64,
    triple_coeff: i64,
}

fn generate_combined_additive_hyperbolic_triple_angle_rewrites(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    rewrites: &mut Vec<DeriveHyperbolicRewrite>,
) {
    let terms = add_terms_signed(ctx, expr);
    if terms.len() <= 1 {
        return;
    }

    for first_index in 0..terms.len() {
        for second_index in (first_index + 1)..terms.len() {
            let Some(data) =
                combined_hyperbolic_triple_angle_data(ctx, terms[first_index], terms[second_index])
            else {
                continue;
            };

            let standard_term = build_combined_hyperbolic_triple_angle_polynomial(
                ctx,
                data.hyperbolic_fn,
                data.base_arg,
                data.base_coeff,
                data.triple_coeff,
            );
            let standard_rewritten = rebuild_additive_terms_with_combined_terms(
                ctx,
                &terms,
                first_index,
                second_index,
                standard_term,
            );
            push_unique_hyperbolic_bridge_rewrite(rewrites, standard_rewritten, data.kind);

            let mixed_term = build_mixed_hyperbolic_triple_angle_polynomial(
                ctx,
                data.hyperbolic_fn,
                data.base_arg,
                data.base_coeff,
                data.triple_coeff,
            );
            let mixed_rewritten = rebuild_additive_terms_with_combined_terms(
                ctx,
                &terms,
                first_index,
                second_index,
                mixed_term,
            );
            push_unique_hyperbolic_bridge_rewrite(rewrites, mixed_rewritten, data.kind);
        }
    }
}

fn generate_reverse_combined_additive_hyperbolic_triple_angle_rewrites(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    rewrites: &mut Vec<DeriveHyperbolicRewrite>,
) {
    let terms = add_terms_signed(ctx, expr);
    if terms.len() <= 1 {
        return;
    }

    for first_index in 0..terms.len() {
        for second_index in (first_index + 1)..terms.len() {
            let Some(data) = reverse_combined_hyperbolic_triple_angle_data(
                ctx,
                terms[first_index],
                terms[second_index],
            ) else {
                continue;
            };

            let three = ctx.num(3);
            let triple_arg = smart_mul(ctx, three, data.base_arg);
            let base_term = build_canonical_hyperbolic_call(ctx, data.hyperbolic_fn, data.base_arg);
            let triple_term = build_canonical_hyperbolic_call(ctx, data.hyperbolic_fn, triple_arg);
            let combined = build_scaled_additive_terms(
                ctx,
                &[
                    (data.base_coeff, base_term),
                    (data.triple_coeff, triple_term),
                ],
            );
            let rewritten = rebuild_additive_terms_with_combined_terms(
                ctx,
                &terms,
                first_index,
                second_index,
                combined,
            );
            push_unique_hyperbolic_bridge_rewrite(rewrites, rewritten, data.kind);
        }
    }
}

fn combined_hyperbolic_triple_angle_data(
    ctx: &mut cas_ast::Context,
    first: (ExprId, Sign),
    second: (ExprId, Sign),
) -> Option<CombinedHyperbolicTripleAngleData> {
    let (first_term, first_sign) = first;
    let (second_term, second_sign) = second;
    let (first_fn, first_arg) = hyperbolic_call_arg(ctx, first_term)?;
    let (second_fn, second_arg) = hyperbolic_call_arg(ctx, second_term)?;
    if first_fn != second_fn {
        return None;
    }

    if let Some(base_arg) = triple_scaled_arg_base(ctx, second_arg) {
        if compare_expr(ctx, first_arg, base_arg) == Ordering::Equal {
            return Some(CombinedHyperbolicTripleAngleData {
                kind: combined_hyperbolic_triple_angle_kind(first_fn)?,
                hyperbolic_fn: first_fn,
                base_arg,
                base_coeff: sign_to_coeff(first_sign),
                triple_coeff: sign_to_coeff(second_sign),
            });
        }
    }

    if let Some(base_arg) = triple_scaled_arg_base(ctx, first_arg) {
        if compare_expr(ctx, second_arg, base_arg) == Ordering::Equal {
            return Some(CombinedHyperbolicTripleAngleData {
                kind: combined_hyperbolic_triple_angle_kind(first_fn)?,
                hyperbolic_fn: first_fn,
                base_arg,
                base_coeff: sign_to_coeff(second_sign),
                triple_coeff: sign_to_coeff(first_sign),
            });
        }
    }

    None
}

fn build_combined_hyperbolic_triple_angle_polynomial(
    ctx: &mut cas_ast::Context,
    hyperbolic_fn: BuiltinFn,
    base_arg: ExprId,
    base_coeff: i64,
    triple_coeff: i64,
) -> ExprId {
    let hyperbolic_call = ctx.call_builtin(hyperbolic_fn, vec![base_arg]);
    let cubic_term = pow3(ctx, hyperbolic_call);
    let (cubic_coeff, linear_coeff) = match hyperbolic_fn {
        BuiltinFn::Sinh => (4 * triple_coeff, base_coeff + 3 * triple_coeff),
        BuiltinFn::Cosh => (4 * triple_coeff, base_coeff - 3 * triple_coeff),
        _ => unreachable!("only sinh/cosh should reach combined hyperbolic triple-angle bridge"),
    };

    build_scaled_additive_terms(
        ctx,
        &[(cubic_coeff, cubic_term), (linear_coeff, hyperbolic_call)],
    )
}

fn build_mixed_hyperbolic_triple_angle_polynomial(
    ctx: &mut cas_ast::Context,
    hyperbolic_fn: BuiltinFn,
    base_arg: ExprId,
    base_coeff: i64,
    triple_coeff: i64,
) -> ExprId {
    let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![base_arg]);
    let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![base_arg]);

    match hyperbolic_fn {
        BuiltinFn::Sinh => {
            let cosh_sq = pow2(ctx, cosh);
            let mixed_term = smart_mul(ctx, cosh_sq, sinh);
            let mixed_coeff = 4 * triple_coeff;
            let linear_coeff = base_coeff - triple_coeff;
            build_scaled_additive_terms(ctx, &[(mixed_coeff, mixed_term), (linear_coeff, sinh)])
        }
        BuiltinFn::Cosh => {
            let sinh_sq = pow2(ctx, sinh);
            let mixed_term = smart_mul(ctx, sinh_sq, cosh);
            let mixed_coeff = 4 * triple_coeff;
            let linear_coeff = base_coeff + triple_coeff;
            build_scaled_additive_terms(ctx, &[(mixed_coeff, mixed_term), (linear_coeff, cosh)])
        }
        _ => unreachable!("only sinh/cosh should reach mixed combined triple-angle bridge"),
    }
}

fn reverse_combined_hyperbolic_triple_angle_data(
    ctx: &mut cas_ast::Context,
    first: (ExprId, Sign),
    second: (ExprId, Sign),
) -> Option<CombinedHyperbolicTripleAngleData> {
    reverse_combined_hyperbolic_triple_angle_data_ordered(ctx, first, second)
        .or_else(|| reverse_combined_hyperbolic_triple_angle_data_ordered(ctx, second, first))
}

fn reverse_combined_hyperbolic_triple_angle_data_ordered(
    ctx: &mut cas_ast::Context,
    linear: (ExprId, Sign),
    other: (ExprId, Sign),
) -> Option<CombinedHyperbolicTripleAngleData> {
    let (linear_builtin, linear_arg, linear_coeff) =
        extract_scaled_hyperbolic_linear_term(ctx, linear.0)?;
    let signed_linear_coeff = linear_coeff * sign_to_coeff(linear.1);

    if let Some((cubic_builtin, cubic_arg, cubic_coeff)) =
        extract_scaled_hyperbolic_cubic_term(ctx, other.0)
    {
        if linear_builtin == cubic_builtin
            && compare_expr(ctx, linear_arg, cubic_arg) == Ordering::Equal
        {
            let signed_cubic_coeff = cubic_coeff * sign_to_coeff(other.1);
            if signed_cubic_coeff % 4 == 0 {
                let triple_coeff = signed_cubic_coeff / 4;
                let base_coeff = match linear_builtin {
                    BuiltinFn::Sinh => signed_linear_coeff - 3 * triple_coeff,
                    BuiltinFn::Cosh => signed_linear_coeff + 3 * triple_coeff,
                    _ => return None,
                };
                return Some(CombinedHyperbolicTripleAngleData {
                    kind: reverse_combined_hyperbolic_triple_angle_kind(linear_builtin)?,
                    hyperbolic_fn: linear_builtin,
                    base_arg: linear_arg,
                    base_coeff,
                    triple_coeff,
                });
            }
        }
    }

    let (mixed_builtin, mixed_arg, mixed_coeff) =
        extract_scaled_mixed_hyperbolic_term(ctx, other.0)?;
    if linear_builtin != mixed_builtin
        || compare_expr(ctx, linear_arg, mixed_arg) != Ordering::Equal
    {
        return None;
    }

    let signed_mixed_coeff = mixed_coeff * sign_to_coeff(other.1);
    if signed_mixed_coeff % 4 != 0 {
        return None;
    }

    let triple_coeff = signed_mixed_coeff / 4;
    let base_coeff = match linear_builtin {
        BuiltinFn::Sinh => signed_linear_coeff + triple_coeff,
        BuiltinFn::Cosh => signed_linear_coeff - triple_coeff,
        _ => return None,
    };

    Some(CombinedHyperbolicTripleAngleData {
        kind: reverse_combined_hyperbolic_triple_angle_kind(linear_builtin)?,
        hyperbolic_fn: linear_builtin,
        base_arg: linear_arg,
        base_coeff,
        triple_coeff,
    })
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

fn hyperbolic_call_arg(ctx: &cas_ast::Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
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

fn combined_hyperbolic_triple_angle_kind(
    builtin: BuiltinFn,
) -> Option<DeriveHyperbolicRewriteKind> {
    match builtin {
        BuiltinFn::Sinh => Some(DeriveHyperbolicRewriteKind::ExpandCombinedSinhTripleAngle),
        BuiltinFn::Cosh => Some(DeriveHyperbolicRewriteKind::ExpandCombinedCoshTripleAngle),
        _ => None,
    }
}

fn reverse_combined_hyperbolic_triple_angle_kind(
    builtin: BuiltinFn,
) -> Option<DeriveHyperbolicRewriteKind> {
    match builtin {
        BuiltinFn::Sinh => Some(DeriveHyperbolicRewriteKind::ContractCombinedSinhTripleAngle),
        BuiltinFn::Cosh => Some(DeriveHyperbolicRewriteKind::ContractCombinedCoshTripleAngle),
        _ => None,
    }
}

fn sign_to_coeff(sign: Sign) -> i64 {
    match sign {
        Sign::Pos => 1,
        Sign::Neg => -1,
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

fn build_canonical_hyperbolic_call(
    ctx: &mut cas_ast::Context,
    builtin: BuiltinFn,
    arg: ExprId,
) -> ExprId {
    match builtin {
        BuiltinFn::Cosh => {
            let arg = strip_top_level_negative_factor(ctx, arg).unwrap_or(arg);
            ctx.call_builtin(BuiltinFn::Cosh, vec![arg])
        }
        BuiltinFn::Sinh => {
            if let Some(arg) = strip_top_level_negative_factor(ctx, arg) {
                let positive = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
                ctx.add(Expr::Neg(positive))
            } else {
                ctx.call_builtin(BuiltinFn::Sinh, vec![arg])
            }
        }
        _ => unreachable!("only sinh/cosh should reach hyperbolic call canonicalization"),
    }
}

fn half_sum(ctx: &mut cas_ast::Context, left: ExprId, right: ExprId) -> ExprId {
    let two = ctx.num(2);
    let sum = ctx.add(Expr::Add(left, right));
    let average = ctx.add(Expr::Div(sum, two));
    run_default_simplify(ctx, average)
}

fn half_diff(ctx: &mut cas_ast::Context, left: ExprId, right: ExprId) -> ExprId {
    let two = ctx.num(2);
    let diff = ctx.add(Expr::Sub(left, right));
    let half_difference = ctx.add(Expr::Div(diff, two));
    run_default_simplify(ctx, half_difference)
}

fn strip_top_level_negative_factor(ctx: &mut cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    if let Some(inner) = strip_unit_negation(ctx, expr) {
        return Some(inner);
    }

    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            if let Expr::Number(n) = ctx.get(*left) {
                if n.is_negative() {
                    let right = *right;
                    let coeff = -n.clone();
                    let positive = ctx.add(Expr::Number(coeff));
                    return Some(smart_mul(ctx, positive, right));
                }
            }
            if let Expr::Number(n) = ctx.get(*right) {
                if n.is_negative() {
                    let left = *left;
                    let coeff = -n.clone();
                    let positive = ctx.add(Expr::Number(coeff));
                    return Some(smart_mul(ctx, left, positive));
                }
            }
            None
        }
        Expr::Number(n) if n.is_negative() => Some(ctx.add(Expr::Number(-n.clone()))),
        _ => None,
    }
}

fn extract_hyperbolic_two_factor_product(
    ctx: &cas_ast::Context,
    expr: ExprId,
) -> Option<HyperbolicTwoFactorPattern> {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };
    let (Expr::Function(left_fn, left_args), Expr::Function(right_fn, right_args)) =
        (ctx.get(*left), ctx.get(*right))
    else {
        return None;
    };
    if left_args.len() != 1 || right_args.len() != 1 {
        return None;
    }

    let left_builtin = ctx.builtin_of(*left_fn)?;
    let right_builtin = ctx.builtin_of(*right_fn)?;
    match (left_builtin, right_builtin) {
        (BuiltinFn::Sinh, BuiltinFn::Cosh) => Some(HyperbolicTwoFactorPattern::SinhCosh(
            left_args[0],
            right_args[0],
        )),
        (BuiltinFn::Cosh, BuiltinFn::Sinh) => Some(HyperbolicTwoFactorPattern::SinhCosh(
            right_args[0],
            left_args[0],
        )),
        (BuiltinFn::Cosh, BuiltinFn::Cosh) => Some(HyperbolicTwoFactorPattern::CoshCosh(
            left_args[0],
            right_args[0],
        )),
        (BuiltinFn::Sinh, BuiltinFn::Sinh) => Some(HyperbolicTwoFactorPattern::SinhSinh(
            left_args[0],
            right_args[0],
        )),
        _ => None,
    }
}

fn extract_scaled_hyperbolic_two_factor_product(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    scale: i64,
) -> Option<ScaledHyperbolicProductPattern> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut saw_scale = false;
    let mut hyperbolic_factors = Vec::new();

    for factor in factors {
        if !saw_scale && is_small_integer(ctx, factor, scale) {
            saw_scale = true;
            continue;
        }

        let (builtin, arg) = hyperbolic_call_arg(ctx, factor)?;
        hyperbolic_factors.push((builtin, arg));
    }

    if !saw_scale || hyperbolic_factors.len() != 2 {
        return None;
    }

    match (hyperbolic_factors[0].0, hyperbolic_factors[1].0) {
        (BuiltinFn::Sinh, BuiltinFn::Cosh) => Some(ScaledHyperbolicProductPattern::SinhCosh(
            hyperbolic_factors[0].1,
            hyperbolic_factors[1].1,
        )),
        (BuiltinFn::Cosh, BuiltinFn::Sinh) => Some(ScaledHyperbolicProductPattern::SinhCosh(
            hyperbolic_factors[1].1,
            hyperbolic_factors[0].1,
        )),
        (BuiltinFn::Cosh, BuiltinFn::Cosh) => Some(ScaledHyperbolicProductPattern::CoshCosh(
            hyperbolic_factors[0].1,
            hyperbolic_factors[1].1,
        )),
        (BuiltinFn::Sinh, BuiltinFn::Sinh) => Some(ScaledHyperbolicProductPattern::SinhSinh(
            hyperbolic_factors[0].1,
            hyperbolic_factors[1].1,
        )),
        _ => None,
    }
}

fn extract_scaled_mixed_sinh_double_angle_term(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    scale: i64,
) -> Option<(BuiltinFn, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut coeff = 1_i64;
    let mut sinh_sq_arg = None;
    let mut cosh_sq_arg = None;
    let mut sinh_arg = None;
    let mut cosh_arg = None;

    for factor in factors {
        match ctx.get(factor) {
            Expr::Number(n) if n.is_integer() => {
                let integer = n.to_integer();
                let integer = i64::try_from(integer).ok()?;
                coeff = coeff.checked_mul(integer)?;
            }
            Expr::Pow(base, exponent) if is_small_integer(ctx, *exponent, 2) => {
                match hyperbolic_call_arg(ctx, *base)? {
                    (BuiltinFn::Sinh, arg) => sinh_sq_arg = Some(arg),
                    (BuiltinFn::Cosh, arg) => cosh_sq_arg = Some(arg),
                    _ => return None,
                }
            }
            _ => match hyperbolic_call_arg(ctx, factor)? {
                (BuiltinFn::Sinh, arg) => sinh_arg = Some(arg),
                (BuiltinFn::Cosh, arg) => cosh_arg = Some(arg),
                _ => return None,
            },
        }
    }

    if coeff != scale {
        return None;
    }

    if let (Some(arg), Some(other_arg)) = (sinh_sq_arg, cosh_arg) {
        if compare_expr(ctx, arg, other_arg) == Ordering::Equal {
            return Some((BuiltinFn::Sinh, arg));
        }
    }

    if let (Some(arg), Some(other_arg)) = (cosh_sq_arg, sinh_arg) {
        if compare_expr(ctx, arg, other_arg) == Ordering::Equal {
            return Some((BuiltinFn::Cosh, arg));
        }
    }

    None
}

fn extract_scaled_mixed_cosh_double_angle_term(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    scale: i64,
) -> Option<(BuiltinFn, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut coeff = 1_i64;
    let mut double_arg = None;
    let mut linear = None;

    for factor in factors {
        match ctx.get(factor) {
            Expr::Number(n) if n.is_integer() => {
                let integer = n.to_integer();
                let integer = i64::try_from(integer).ok()?;
                coeff = coeff.checked_mul(integer)?;
            }
            Expr::Function(fn_id, args) => {
                if args.len() != 1 {
                    return None;
                }

                let builtin = ctx.builtin_of(*fn_id)?;
                let arg = args[0];
                match builtin {
                    BuiltinFn::Cosh => {
                        if let Some(base_arg) =
                            cas_math::trig_roots_flatten::extract_double_angle_arg_relaxed(ctx, arg)
                        {
                            if double_arg.is_some() {
                                return None;
                            }
                            double_arg = Some(base_arg);
                            continue;
                        }

                        if linear.is_some() {
                            return None;
                        }
                        linear = Some((BuiltinFn::Cosh, arg));
                    }
                    BuiltinFn::Sinh => {
                        if linear.is_some() {
                            return None;
                        }
                        linear = Some((BuiltinFn::Sinh, arg));
                    }
                    _ => return None,
                }
            }
            _ => return None,
        }
    }

    if coeff != scale {
        return None;
    }

    let double_arg = double_arg?;
    let (linear_builtin, linear_arg) = linear?;
    if compare_expr(ctx, double_arg, linear_arg) != Ordering::Equal {
        return None;
    }

    Some((linear_builtin, double_arg))
}

fn extract_scaled_hyperbolic_linear_term(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId, i64)> {
    let (coeff, core) = split_integer_coefficient(ctx, expr)?;
    let (builtin, arg) = hyperbolic_call_arg(ctx, core)?;
    Some((builtin, arg, coeff))
}

fn extract_scaled_hyperbolic_cubic_term(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId, i64)> {
    let (coeff, core) = split_integer_coefficient(ctx, expr)?;
    let Expr::Pow(base, exponent) = ctx.get(core) else {
        return None;
    };
    if !is_small_integer(ctx, *exponent, 3) {
        return None;
    }
    let (builtin, arg) = hyperbolic_call_arg(ctx, *base)?;
    Some((builtin, arg, coeff))
}

fn extract_scaled_mixed_hyperbolic_term(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId, i64)> {
    let (coeff, core) = split_integer_coefficient(ctx, expr)?;
    let factors = flatten_mul_chain(ctx, core);
    if factors.len() != 2 {
        return None;
    }

    let first = extract_hyperbolic_square_factor(ctx, factors[0]);
    let second = extract_hyperbolic_square_factor(ctx, factors[1]);

    match (first, hyperbolic_call_arg(ctx, factors[1])) {
        (Some((BuiltinFn::Cosh, arg)), Some((BuiltinFn::Sinh, other_arg)))
            if compare_expr(ctx, arg, other_arg) == Ordering::Equal =>
        {
            Some((BuiltinFn::Sinh, arg, coeff))
        }
        (Some((BuiltinFn::Sinh, arg)), Some((BuiltinFn::Cosh, other_arg)))
            if compare_expr(ctx, arg, other_arg) == Ordering::Equal =>
        {
            Some((BuiltinFn::Cosh, arg, coeff))
        }
        _ => match (second, hyperbolic_call_arg(ctx, factors[0])) {
            (Some((BuiltinFn::Cosh, arg)), Some((BuiltinFn::Sinh, other_arg)))
                if compare_expr(ctx, arg, other_arg) == Ordering::Equal =>
            {
                Some((BuiltinFn::Sinh, arg, coeff))
            }
            (Some((BuiltinFn::Sinh, arg)), Some((BuiltinFn::Cosh, other_arg)))
                if compare_expr(ctx, arg, other_arg) == Ordering::Equal =>
            {
                Some((BuiltinFn::Cosh, arg, coeff))
            }
            _ => None,
        },
    }
}

fn extract_hyperbolic_square_factor(
    ctx: &cas_ast::Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    if !is_small_integer(ctx, *exponent, 2) {
        return None;
    }
    hyperbolic_call_arg(ctx, *base)
}

fn split_integer_coefficient(ctx: &mut cas_ast::Context, expr: ExprId) -> Option<(i64, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut coeff = 1_i64;
    let mut non_numeric = Vec::new();

    for factor in factors {
        match ctx.get(factor) {
            Expr::Number(n) if n.is_integer() => {
                let integer = n.to_integer();
                let integer = i64::try_from(integer).ok()?;
                coeff = coeff.checked_mul(integer)?;
            }
            _ => non_numeric.push(factor),
        }
    }

    let core = match non_numeric.as_slice() {
        [] => return None,
        [single] => *single,
        [first, rest @ ..] => rest
            .iter()
            .copied()
            .fold(*first, |acc, factor| smart_mul(ctx, acc, factor)),
    };

    Some((coeff, core))
}

fn expand_tanh_angle_sum_diff_expr(ctx: &mut cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Tanh) || args.len() != 1 {
        return None;
    }

    let (is_sum, left, right) = match ctx.get(args[0]) {
        Expr::Add(left, right) => (true, *left, *right),
        Expr::Sub(left, right) => (false, *left, *right),
        _ => return None,
    };

    let tanh_left = ctx.call_builtin(BuiltinFn::Tanh, vec![left]);
    let tanh_right = ctx.call_builtin(BuiltinFn::Tanh, vec![right]);
    let numerator = if is_sum {
        ctx.add(Expr::Add(tanh_left, tanh_right))
    } else {
        ctx.add(Expr::Sub(tanh_left, tanh_right))
    };
    let product = ctx.add(Expr::Mul(tanh_left, tanh_right));
    let one = ctx.num(1);
    let denominator = if is_sum {
        ctx.add(Expr::Add(one, product))
    } else {
        ctx.add(Expr::Sub(one, product))
    };
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn contract_tanh_angle_sum_diff_expr(ctx: &mut cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };

    let (is_sum, left_num, right_num) = match ctx.get(*numerator) {
        Expr::Add(left, right) => (true, *left, *right),
        Expr::Sub(left, right) => (false, *left, *right),
        _ => return None,
    };

    let left_arg = tanh_arg(ctx, left_num)?;
    let right_arg = tanh_arg(ctx, right_num)?;

    let denominator_matches = match ctx.get(*denominator) {
        Expr::Add(left, right) if is_sum => {
            (cas_math::expr_predicates::is_one_expr(ctx, *left)
                && tanh_product_matches_pair(ctx, *right, left_arg, right_arg))
                || (cas_math::expr_predicates::is_one_expr(ctx, *right)
                    && tanh_product_matches_pair(ctx, *left, left_arg, right_arg))
        }
        Expr::Sub(left, right) if !is_sum => {
            cas_math::expr_predicates::is_one_expr(ctx, *left)
                && tanh_product_matches_pair(ctx, *right, left_arg, right_arg)
        }
        _ => false,
    };

    if !denominator_matches {
        return None;
    }

    let angle = if is_sum {
        ctx.add(Expr::Add(left_arg, right_arg))
    } else {
        ctx.add(Expr::Sub(left_arg, right_arg))
    };
    Some(ctx.call_builtin(BuiltinFn::Tanh, vec![angle]))
}

fn tanh_arg(ctx: &cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Tanh) || args.len() != 1 {
        return None;
    }
    Some(args[0])
}

fn tanh_product_matches_pair(
    ctx: &cas_ast::Context,
    expr: ExprId,
    left_arg: ExprId,
    right_arg: ExprId,
) -> bool {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return false;
    };
    let Some(left_tanh_arg) = tanh_arg(ctx, *left) else {
        return false;
    };
    let Some(right_tanh_arg) = tanh_arg(ctx, *right) else {
        return false;
    };
    same_unordered_pair(ctx, left_tanh_arg, right_tanh_arg, left_arg, right_arg)
}

fn extract_tanh_power(ctx: &cas_ast::Context, expr: ExprId, power: i64) -> Option<ExprId> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    if !is_small_integer(ctx, *exponent, power) {
        return None;
    }
    tanh_arg(ctx, *base)
}

fn extract_scaled_tanh_power(
    ctx: &cas_ast::Context,
    expr: ExprId,
    scale: i64,
    power: i64,
) -> Option<ExprId> {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };

    if is_small_integer(ctx, *left, scale) {
        return if power == 1 {
            tanh_arg(ctx, *right)
        } else {
            extract_tanh_power(ctx, *right, power)
        };
    }

    if is_small_integer(ctx, *right, scale) {
        return if power == 1 {
            tanh_arg(ctx, *left)
        } else {
            extract_tanh_power(ctx, *left, power)
        };
    }

    None
}

fn expand_tanh_triple_angle_expr(ctx: &mut cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Tanh) || args.len() != 1 {
        return None;
    }

    let inner = cas_math::trig_roots_flatten::extract_triple_angle_arg_relaxed(ctx, args[0])?;
    let tanh_inner = ctx.call_builtin(BuiltinFn::Tanh, vec![inner]);
    let tanh_sq = pow2(ctx, tanh_inner);
    let three = ctx.num(3);
    let tanh_cube = ctx.add(Expr::Pow(tanh_inner, three));
    let three_tanh = ctx.add(Expr::Mul(three, tanh_inner));
    let numerator = ctx.add(Expr::Add(tanh_cube, three_tanh));
    let three_tanh_sq = ctx.add(Expr::Mul(three, tanh_sq));
    let one = ctx.num(1);
    let denominator = ctx.add(Expr::Add(three_tanh_sq, one));
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn contract_tanh_triple_angle_expr(ctx: &mut cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };

    let Expr::Add(num_left, num_right) = ctx.get(*numerator) else {
        return None;
    };
    let inner = match (
        extract_tanh_power(ctx, *num_left, 3),
        extract_scaled_tanh_power(ctx, *num_right, 3, 1),
    ) {
        (Some(left_inner), Some(right_inner))
            if compare_expr(ctx, left_inner, right_inner) == Ordering::Equal =>
        {
            left_inner
        }
        _ => match (
            extract_tanh_power(ctx, *num_right, 3),
            extract_scaled_tanh_power(ctx, *num_left, 3, 1),
        ) {
            (Some(left_inner), Some(right_inner))
                if compare_expr(ctx, left_inner, right_inner) == Ordering::Equal =>
            {
                left_inner
            }
            _ => return None,
        },
    };

    let Expr::Add(den_left, den_right) = ctx.get(*denominator) else {
        return None;
    };
    let denominator_matches = (is_small_integer(ctx, *den_left, 1)
        && extract_scaled_tanh_power(ctx, *den_right, 3, 2)
            .is_some_and(|arg| compare_expr(ctx, arg, inner) == Ordering::Equal))
        || (is_small_integer(ctx, *den_right, 1)
            && extract_scaled_tanh_power(ctx, *den_left, 3, 2)
                .is_some_and(|arg| compare_expr(ctx, arg, inner) == Ordering::Equal));
    if !denominator_matches {
        return None;
    }

    let three = ctx.num(3);
    let triple_inner = ctx.add(Expr::Mul(three, inner));
    Some(ctx.call_builtin(BuiltinFn::Tanh, vec![triple_inner]))
}

fn same_unordered_pair(ctx: &cas_ast::Context, a: ExprId, b: ExprId, c: ExprId, d: ExprId) -> bool {
    (compare_expr(ctx, a, c) == Ordering::Equal && compare_expr(ctx, b, d) == Ordering::Equal)
        || (compare_expr(ctx, a, d) == Ordering::Equal
            && compare_expr(ctx, b, c) == Ordering::Equal)
}

fn is_small_integer(ctx: &cas_ast::Context, expr: ExprId, value: i64) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_integer() && n.to_integer() == value.into())
}

fn pow2(ctx: &mut cas_ast::Context, base: ExprId) -> ExprId {
    let two = ctx.num(2);
    ctx.add(Expr::Pow(base, two))
}

fn pow3(ctx: &mut cas_ast::Context, base: ExprId) -> ExprId {
    let three = ctx.num(3);
    ctx.add(Expr::Pow(base, three))
}

fn strip_unit_negation(ctx: &mut cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Neg(inner) => Some(*inner),
        Expr::Number(n) if n.is_negative() => Some(ctx.add(Expr::Number(-n))),
        Expr::Mul(left, right) if is_small_integer(ctx, *left, -1) => Some(*right),
        Expr::Mul(left, right) if is_small_integer(ctx, *right, -1) => Some(*left),
        _ => None,
    }
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

fn matches_target_modulo_simplify(ctx: &mut cas_ast::Context, left: ExprId, right: ExprId) -> bool {
    strong_target_match(ctx, left, right) || simplified_difference_matches_zero(ctx, left, right)
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

fn run_default_simplify(ctx: &mut cas_ast::Context, expr: ExprId) -> ExprId {
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

#[cfg(test)]
mod tests {
    use super::{
        generate_hyperbolic_additive_term_bridge_rewrites, generate_hyperbolic_bridge_rewrites,
        matches_target_modulo_simplify, should_try_hyperbolic_planner_before_simplify,
        try_rewrite_hyperbolic_simplify_target_aware, DeriveHyperbolicRewriteKind,
    };

    #[test]
    fn target_aware_hyperbolic_rewrite_contracts_sinh_double_angle() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("2*sinh(x)*cosh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("sinh(2*x)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ContractSinhDoubleAngle
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_contracts_mixed_sinh_double_angle_product() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("4*sinh(x)^2*cosh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("2*sinh(2*x)*sinh(x)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ContractSinhDoubleAngle
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_contracts_mixed_cosh_double_angle_product() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("4*cosh(x)^2*sinh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("2*sinh(2*x)*cosh(x)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ContractSinhDoubleAngle
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_expands_mixed_cosh_double_angle_product_to_sinh_cubic() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("2*cosh(2*x)*sinh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("2*sinh(x)+4*sinh(x)^3", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ExpandCoshDoubleAngleAsOnePlusTwoSinhSq
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_expands_mixed_cosh_double_angle_product_to_sinh_mixed() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("2*cosh(2*x)*sinh(x)", &mut ctx).expect("parse source");
        let target =
            cas_parser::parse("4*cosh(x)^2*sinh(x)-2*sinh(x)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ExpandCoshDoubleAngleAsTwoCoshSqMinusOne
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_expands_mixed_cosh_double_angle_product_to_cosh_mixed() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("2*cosh(2*x)*cosh(x)", &mut ctx).expect("parse source");
        let target =
            cas_parser::parse("2*cosh(x)+4*sinh(x)^2*cosh(x)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ExpandCoshDoubleAngleAsOnePlusTwoSinhSq
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_contracts_mixed_cosh_square_polynomial_to_product() {
        let mut ctx = cas_ast::Context::new();
        let source =
            cas_parser::parse("4*cosh(x)^2*sinh(x)-2*sinh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("2*cosh(2*x)*sinh(x)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ExpandCoshDoubleAngleAsTwoCoshSqMinusOne
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_contracts_mixed_sinh_square_polynomial_to_product() {
        let mut ctx = cas_ast::Context::new();
        let source =
            cas_parser::parse("2*cosh(x)+4*sinh(x)^2*cosh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("2*cosh(2*x)*cosh(x)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ExpandCoshDoubleAngleAsOnePlusTwoSinhSq
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_contracts_sinh_angle_sum() {
        let mut ctx = cas_ast::Context::new();
        let source =
            cas_parser::parse("sinh(x)*cosh(y)+cosh(x)*sinh(y)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("sinh(x+y)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ContractSinhAngleSumDiff
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_contracts_cosh_angle_sum() {
        let mut ctx = cas_ast::Context::new();
        let source =
            cas_parser::parse("cosh(x)*cosh(y)+sinh(x)*sinh(y)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("cosh(x+y)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ContractCoshAngleSumDiff
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_contracts_sinh_angle_difference() {
        let mut ctx = cas_ast::Context::new();
        let source =
            cas_parser::parse("sinh(x)*cosh(y)-cosh(x)*sinh(y)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("sinh(x-y)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ContractSinhAngleSumDiff
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_contracts_negative_cosh_angle_difference() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("-(cosh(x)*cosh(y)-sinh(x)*sinh(y))", &mut ctx)
            .expect("parse source");
        let target = cas_parser::parse("-cosh(x-y)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ContractCoshAngleSumDiff
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_expands_tanh_angle_sum() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("tanh(x+y)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("(tanh(x)+tanh(y))/(1+tanh(x)*tanh(y))", &mut ctx)
            .expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ExpandTanhAngleSumDiff
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_contracts_tanh_angle_sum() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("(tanh(x)+tanh(y))/(1+tanh(x)*tanh(y))", &mut ctx)
            .expect("parse source");
        let target = cas_parser::parse("tanh(x+y)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ContractTanhAngleSumDiff
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_expands_tanh_triple_angle() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("tanh(3*x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("(3*tanh(x)+tanh(x)^3)/(1+3*tanh(x)^2)", &mut ctx)
            .expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ExpandTanhTripleAngle
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_contracts_tanh_triple_angle() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("(3*tanh(x)+tanh(x)^3)/(1+3*tanh(x)^2)", &mut ctx)
            .expect("parse source");
        let target = cas_parser::parse("tanh(3*x)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ContractTanhTripleAngle
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn generates_hyperbolic_product_to_sum_bridge_for_sinh_cosh() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("2*sinh(2*x)*cosh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("sinh(3*x)+sinh(x)", &mut ctx).expect("parse target");
        let rewrites = generate_hyperbolic_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::ProductToSumSinhCosh
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_hyperbolic_product_to_sum_bridge_for_cosh_sinh_with_odd_parity_canonicalization() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("2*cosh(2*x)*sinh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("sinh(3*x)-sinh(x)", &mut ctx).expect("parse target");
        let rewrites = generate_hyperbolic_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::ProductToSumSinhCosh
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_combined_hyperbolic_triple_angle_bridge_for_sinh_sum_polynomial() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("sinh(3*x)+sinh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("4*sinh(x)+4*sinh(x)^3", &mut ctx).expect("parse target");
        let rewrites = generate_hyperbolic_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::ExpandCombinedSinhTripleAngle
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_combined_hyperbolic_triple_angle_bridge_for_sinh_difference_mixed_polynomial() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("sinh(3*x)-sinh(x)", &mut ctx).expect("parse source");
        let target =
            cas_parser::parse("4*cosh(x)^2*sinh(x)-2*sinh(x)", &mut ctx).expect("parse target");
        let rewrites = generate_hyperbolic_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::ExpandCombinedSinhTripleAngle
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_reverse_combined_hyperbolic_triple_angle_bridge_for_sinh_difference_mixed_polynomial(
    ) {
        let mut ctx = cas_ast::Context::new();
        let source =
            cas_parser::parse("4*cosh(x)^2*sinh(x)-2*sinh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("sinh(3*x)-sinh(x)", &mut ctx).expect("parse target");
        let rewrites = generate_hyperbolic_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::ContractCombinedSinhTripleAngle
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_combined_hyperbolic_triple_angle_bridge_for_cosh_difference_polynomial() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("cosh(3*x)-cosh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("4*cosh(x)^3-4*cosh(x)", &mut ctx).expect("parse target");
        let rewrites = generate_hyperbolic_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::ExpandCombinedCoshTripleAngle
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_reverse_combined_hyperbolic_triple_angle_bridge_for_cosh_sum_mixed_polynomial() {
        let mut ctx = cas_ast::Context::new();
        let source =
            cas_parser::parse("2*cosh(x)+4*sinh(x)^2*cosh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("cosh(3*x)+cosh(x)", &mut ctx).expect("parse target");
        let rewrites = generate_hyperbolic_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::ContractCombinedCoshTripleAngle
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_hyperbolic_sum_to_product_bridge_for_sinh_difference() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("sinh(3*x)-sinh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("2*cosh(2*x)*sinh(x)", &mut ctx).expect("parse target");
        let rewrites = generate_hyperbolic_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::SumToProductSinhCosh
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_hyperbolic_sum_to_product_bridge_for_cosh_sum() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("cosh(3*x)+cosh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("2*cosh(2*x)*cosh(x)", &mut ctx).expect("parse target");
        let rewrites = generate_hyperbolic_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::SumToProductCoshCosh
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_hyperbolic_additive_term_bridge_with_passthrough() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("2*sinh(2*x)*sinh(x)+a", &mut ctx).expect("parse source");
        let target = cas_parser::parse("cosh(3*x)-cosh(x)+a", &mut ctx).expect("parse target");
        let rewrites = generate_hyperbolic_additive_term_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::ProductToSumSinhSinh
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_rewrites_negative_multi_angle_variants() {
        let cases = [
            (
                "-2*sinh(x)*cosh(x)",
                "-sinh(2*x)",
                DeriveHyperbolicRewriteKind::ContractSinhDoubleAngle,
            ),
            (
                "-sinh(2*x)",
                "-2*sinh(x)*cosh(x)",
                DeriveHyperbolicRewriteKind::ContractSinhDoubleAngle,
            ),
            (
                "-2*tanh(x)/(1+tanh(x)^2)",
                "-tanh(2*x)",
                DeriveHyperbolicRewriteKind::ContractTanhDoubleAngle,
            ),
            (
                "-tanh(2*x)",
                "-2*tanh(x)/(1+tanh(x)^2)",
                DeriveHyperbolicRewriteKind::ContractTanhDoubleAngle,
            ),
            (
                "-(3*sinh(x)+4*sinh(x)^3)",
                "-sinh(3*x)",
                DeriveHyperbolicRewriteKind::ContractSinhTripleAngle,
            ),
            (
                "-sinh(3*x)",
                "-(3*sinh(x)+4*sinh(x)^3)",
                DeriveHyperbolicRewriteKind::ContractSinhTripleAngle,
            ),
            (
                "-(4*cosh(x)^3-3*cosh(x))",
                "-cosh(3*x)",
                DeriveHyperbolicRewriteKind::ContractCoshTripleAngle,
            ),
            (
                "-cosh(3*x)",
                "-(4*cosh(x)^3-3*cosh(x))",
                DeriveHyperbolicRewriteKind::ContractCoshTripleAngle,
            ),
        ];

        for (source_text, target_text, expected_kind) in cases {
            let mut ctx = cas_ast::Context::new();
            let source = cas_parser::parse(source_text, &mut ctx).expect("parse source");
            let target = cas_parser::parse(target_text, &mut ctx).expect("parse target");
            let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
                .expect("expected hyperbolic target-aware rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert!(matches_target_modulo_simplify(
                &mut ctx,
                rewrite.rewritten,
                target
            ));
        }
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_recognizes_cosh_from_exponentials() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("(e^x + e^(-x))/2", &mut ctx).expect("parse source");
        let target = cas_parser::parse("cosh(x)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::RecognizeCoshFromExp
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_rewrites_negative_cosh_exponential_variants() {
        let cases = [
            (
                "-(e^x + e^(-x))/2",
                "-cosh(x)",
                DeriveHyperbolicRewriteKind::RecognizeCoshFromExp,
            ),
            (
                "-cosh(x)",
                "-(e^x + e^(-x))/2",
                DeriveHyperbolicRewriteKind::ExpandCoshToExpHalfSum,
            ),
        ];

        for (source_text, target_text, expected_kind) in cases {
            let mut ctx = cas_ast::Context::new();
            let source = cas_parser::parse(source_text, &mut ctx).expect("parse source");
            let target = cas_parser::parse(target_text, &mut ctx).expect("parse target");
            let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
                .expect("expected hyperbolic target-aware rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert!(matches_target_modulo_simplify(
                &mut ctx,
                rewrite.rewritten,
                target
            ));
        }
    }

    #[test]
    fn generates_hyperbolic_bridge_rewrite_for_double_cosh_from_exponentials() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("exp(2*x)+exp(-2*x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("2*cosh(2*x)", &mut ctx).expect("parse target");
        let rewrites = generate_hyperbolic_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::RecognizeDoubleCoshFromExp
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_hyperbolic_bridge_rewrite_for_double_cosh_from_exponential_reciprocal_pair() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("exp(x)+1/exp(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("2*cosh(x)", &mut ctx).expect("parse target");
        let rewrites = generate_hyperbolic_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::RecognizeDoubleCoshFromExp
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_hyperbolic_bridge_rewrite_for_tanh_reciprocal_exponential_definition() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("tanh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("(exp(x)-1/exp(x))/(exp(x)+1/exp(x))", &mut ctx)
            .expect("parse target");
        let rewrites = generate_hyperbolic_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::ExpandTanhToExpRatio
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_hyperbolic_bridge_rewrite_for_cosh_reciprocal_exponential_definition() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("cosh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("(exp(x)+1/exp(x))/2", &mut ctx).expect("parse target");
        let rewrites = generate_hyperbolic_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::ExpandCoshToExpHalfSum
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_hyperbolic_bridge_rewrite_for_scaled_cosh_reciprocal_exponential_definition() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("2*cosh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("exp(x)+1/exp(x)", &mut ctx).expect("parse target");
        let rewrites = generate_hyperbolic_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::ExpandDoubleCoshToExpSum
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_hyperbolic_bridge_rewrite_for_scaled_sinh_reciprocal_exponential_definition() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("2*sinh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("exp(x)-1/exp(x)", &mut ctx).expect("parse target");
        let rewrites = generate_hyperbolic_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::ExpandDoubleSinhToExpDiff
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_hyperbolic_bridge_rewrite_for_negative_scaled_cosh_reciprocal_exponential_definition(
    ) {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("-2*cosh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("-(exp(x)+1/exp(x))", &mut ctx).expect("parse target");
        let rewrites = generate_hyperbolic_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::ExpandDoubleCoshToExpSum
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_hyperbolic_bridge_rewrite_for_negative_cosh_reciprocal_exponential_definition() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("-cosh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("-(exp(x)+1/exp(x))/2", &mut ctx).expect("parse target");
        let rewrites = generate_hyperbolic_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::ExpandCoshToExpHalfSum
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_hyperbolic_bridge_rewrite_for_negative_sinh_reciprocal_exponential_definition() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("-sinh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("-(exp(x)-1/exp(x))/2", &mut ctx).expect("parse target");
        let rewrites = generate_hyperbolic_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::ExpandNegSinhToExpHalfNegDiff
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn prefers_direct_hyperbolic_rewrite_for_exponential_reciprocal_pair() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("exp(x)+1/exp(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("2*cosh(x)", &mut ctx).expect("parse target");

        assert!(!should_try_hyperbolic_planner_before_simplify(
            &mut ctx, source, target
        ));
    }

    #[test]
    fn prefers_direct_hyperbolic_rewrite_for_scaled_sinh_to_exponential_difference() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("2*sinh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("exp(x)-exp(-x)", &mut ctx).expect("parse target");

        assert!(!should_try_hyperbolic_planner_before_simplify(
            &mut ctx, source, target
        ));
    }

    #[test]
    fn generates_hyperbolic_bridge_rewrite_for_double_sinh_from_exponentials() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("exp(2*x)-exp(-2*x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("2*sinh(2*x)", &mut ctx).expect("parse target");
        let rewrites = generate_hyperbolic_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::RecognizeDoubleSinhFromExp
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_hyperbolic_bridge_rewrite_for_double_cosh_to_exponentials() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("2*cosh(2*x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("exp(2*x)+exp(-2*x)", &mut ctx).expect("parse target");
        let rewrites = generate_hyperbolic_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::ExpandDoubleCoshToExpSum
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_hyperbolic_bridge_rewrite_for_negative_double_sinh_to_exponentials() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("-2*sinh(2*x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("exp(-2*x)-exp(2*x)", &mut ctx).expect("parse target");
        let rewrites = generate_hyperbolic_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::ExpandDoubleSinhToExpDiff
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_contracts_cosh_triple_angle() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("4*cosh(x)^3 - 3*cosh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("cosh(3*x)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ContractCoshTripleAngle
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_contracts_cosh_half_angle_square() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("(cosh(x)+1)/2", &mut ctx).expect("parse source");
        let target = cas_parser::parse("cosh(x/2)^2", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ContractHyperbolicCoshHalfAngleSquare
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_recognizes_pythagorean_identity_variants() {
        let cases = [
            (
                "cosh(x)^2 - sinh(x)^2",
                "1",
                DeriveHyperbolicRewriteKind::PythagoreanOne,
            ),
            (
                "sinh(x)^2 - cosh(x)^2",
                "-1",
                DeriveHyperbolicRewriteKind::PythagoreanNegativeOne,
            ),
        ];

        for (source_text, target_text, expected_kind) in cases {
            let mut ctx = cas_ast::Context::new();
            let source = cas_parser::parse(source_text, &mut ctx).expect("parse source");
            let target = cas_parser::parse(target_text, &mut ctx).expect("parse target");
            let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
                .expect("expected hyperbolic target-aware rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert!(matches_target_modulo_simplify(
                &mut ctx,
                rewrite.rewritten,
                target
            ));
        }
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_contracts_sinh_half_angle_square() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("(cosh(x)-1)/2", &mut ctx).expect("parse source");
        let target = cas_parser::parse("sinh(x/2)^2", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ContractHyperbolicSinhHalfAngleSquare
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_rewrites_negative_half_angle_square_variants() {
        let cases = [
            (
                "-(cosh(x)+1)/2",
                "-cosh(x/2)^2",
                DeriveHyperbolicRewriteKind::ContractHyperbolicCoshHalfAngleSquare,
            ),
            (
                "-cosh(x/2)^2",
                "-(cosh(x)+1)/2",
                DeriveHyperbolicRewriteKind::ExpandHyperbolicCoshHalfAngleSquare,
            ),
            (
                "-(cosh(x)-1)/2",
                "-sinh(x/2)^2",
                DeriveHyperbolicRewriteKind::ContractHyperbolicSinhHalfAngleSquare,
            ),
            (
                "-sinh(x/2)^2",
                "-(cosh(x)-1)/2",
                DeriveHyperbolicRewriteKind::ExpandHyperbolicSinhHalfAngleSquare,
            ),
        ];

        for (source_text, target_text, expected_kind) in cases {
            let mut ctx = cas_ast::Context::new();
            let source = cas_parser::parse(source_text, &mut ctx).expect("parse source");
            let target = cas_parser::parse(target_text, &mut ctx).expect("parse target");
            let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
                .expect("expected hyperbolic target-aware rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert_eq!(rewrite.rewritten, target);
        }
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_expands_cosh_double_angle_to_sum() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("cosh(2*x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("cosh(x)^2 + sinh(x)^2", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ExpandCoshDoubleAngleAsSum
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_expands_cosh_double_angle_to_two_cosh_sq_minus_one() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("cosh(2*x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("2*cosh(x)^2 - 1", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ExpandCoshDoubleAngleAsTwoCoshSqMinusOne
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_expands_cosh_double_angle_to_one_plus_two_sinh_sq() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("cosh(2*x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("1 + 2*sinh(x)^2", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ExpandCoshDoubleAngleAsOnePlusTwoSinhSq
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_rewrites_negative_cosh_double_angle_variants() {
        let cases = [
            (
                "-cosh(2*x)",
                "1 - 2*cosh(x)^2",
                DeriveHyperbolicRewriteKind::ExpandNegativeCoshDoubleAngleAsOneMinusTwoCoshSq,
            ),
            (
                "1 - 2*cosh(x)^2",
                "-cosh(2*x)",
                DeriveHyperbolicRewriteKind::RecognizeOneMinusTwoCoshSqAsNegativeCoshDoubleAngle,
            ),
            (
                "-cosh(2*x)",
                "-1 - 2*sinh(x)^2",
                DeriveHyperbolicRewriteKind::ExpandNegativeCoshDoubleAngleAsNegativeOneMinusTwoSinhSq,
            ),
            (
                "-1 - 2*sinh(x)^2",
                "-cosh(2*x)",
                DeriveHyperbolicRewriteKind::RecognizeNegativeOneMinusTwoSinhSqAsNegativeCoshDoubleAngle,
            ),
        ];

        for (source_text, target_text, expected_kind) in cases {
            let mut ctx = cas_ast::Context::new();
            let source = cas_parser::parse(source_text, &mut ctx).expect("parse source");
            let target = cas_parser::parse(target_text, &mut ctx).expect("parse target");
            let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
                .expect("expected hyperbolic target-aware rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert_eq!(rewrite.rewritten, target);
        }
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_expands_exp_to_sinh_plus_cosh() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("exp(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("sinh(x) + cosh(x)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ExpandExpToSinhPlusCosh
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_expands_exp_neg_to_cosh_minus_sinh() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("exp(-x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("cosh(x) - sinh(x)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ExpandExpNegToCoshMinusSinh
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_expands_neg_exp_neg_to_sinh_minus_cosh() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("-exp(-x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("sinh(x) - cosh(x)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ExpandNegExpNegToSinhMinusCosh
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_expands_inverse_cosh_square_to_tanh_pythagorean() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("1/cosh(x)^2", &mut ctx).expect("parse source");
        let target = cas_parser::parse("1 - tanh(x)^2", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ExpandTanhPythagorean
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_contracts_tanh_pythagorean_to_inverse_cosh_square() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("1 - tanh(x)^2", &mut ctx).expect("parse source");
        let target = cas_parser::parse("1/cosh(x)^2", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(rewrite.kind, DeriveHyperbolicRewriteKind::TanhPythagorean);
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_rewrites_negative_tanh_identity_variants() {
        let cases = [
            (
                "tanh(x)^2 - 1",
                "-1/cosh(x)^2",
                DeriveHyperbolicRewriteKind::TanhPythagorean,
            ),
            (
                "-1/cosh(x)^2",
                "tanh(x)^2 - 1",
                DeriveHyperbolicRewriteKind::ExpandTanhPythagorean,
            ),
            (
                "-sinh(x)/cosh(x)",
                "-tanh(x)",
                DeriveHyperbolicRewriteKind::SinhCoshToTanh,
            ),
            (
                "-tanh(x)",
                "-sinh(x)/cosh(x)",
                DeriveHyperbolicRewriteKind::TanhToSinhCosh,
            ),
        ];

        for (source_text, target_text, expected_kind) in cases {
            let mut ctx = cas_ast::Context::new();
            let source = cas_parser::parse(source_text, &mut ctx).expect("parse source");
            let target = cas_parser::parse(target_text, &mut ctx).expect("parse target");
            let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
                .expect("expected hyperbolic target-aware rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert_eq!(rewrite.rewritten, target);
        }
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_expands_sinh_to_exponential_definition() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("sinh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("(e^x - e^(-x))/2", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ExpandSinhToExpHalfDiff
        );
        assert!(matches_target_modulo_simplify(
            &mut ctx,
            rewrite.rewritten,
            target
        ));
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_expands_tanh_to_exponential_definition() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("tanh(x)", &mut ctx).expect("parse source");
        let target =
            cas_parser::parse("(e^x - e^(-x))/(e^x + e^(-x))", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ExpandTanhToExpRatio
        );
        assert!(matches_target_modulo_simplify(
            &mut ctx,
            rewrite.rewritten,
            target
        ));
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_expands_tanh_to_reciprocal_exponential_definition() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("tanh(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("(exp(x)-1/exp(x))/(exp(x)+1/exp(x))", &mut ctx)
            .expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ExpandTanhToExpRatio
        );
        assert!(matches_target_modulo_simplify(
            &mut ctx,
            rewrite.rewritten,
            target
        ));
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_recognizes_scaled_tanh_from_exponential_ratio() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("(e^(2*x)-e^(-2*x))/(e^(2*x)+e^(-2*x))", &mut ctx)
            .expect("parse source");
        let target = cas_parser::parse("tanh(2*x)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::RecognizeTanhFromExp
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_recognizes_cosh_squared_minus_one_as_sinh_squared() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("cosh(x)^2 - 1", &mut ctx).expect("parse source");
        let target = cas_parser::parse("sinh(x)^2", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::RecognizeCoshSquaredMinusOneAsSinhSquared
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_expands_sinh_squared_to_cosh_squared_minus_one() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("sinh(x)^2", &mut ctx).expect("parse source");
        let target = cas_parser::parse("cosh(x)^2 - 1", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ExpandSinhSquaredToCoshSquaredMinusOne
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_recognizes_one_plus_sinh_squared_as_cosh_squared() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("1 + sinh(x)^2", &mut ctx).expect("parse source");
        let target = cas_parser::parse("cosh(x)^2", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::RecognizeOnePlusSinhSquaredAsCoshSquared
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_expands_cosh_squared_to_one_plus_sinh_squared() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("cosh(x)^2", &mut ctx).expect("parse source");
        let target = cas_parser::parse("1 + sinh(x)^2", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ExpandCoshSquaredToOnePlusSinhSquared
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_recognizes_cosh_double_angle_minus_one_as_two_sinh_sq() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("cosh(2*x) - 1", &mut ctx).expect("parse source");
        let target = cas_parser::parse("2*sinh(x)^2", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::RecognizeCoshDoubleAngleMinusOneAsTwoSinhSq
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_expands_two_sinh_sq_to_cosh_double_angle_minus_one() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("2*sinh(x)^2", &mut ctx).expect("parse source");
        let target = cas_parser::parse("cosh(2*x) - 1", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ExpandTwoSinhSqToCoshDoubleAngleMinusOne
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_rewrites_negative_shifted_cosh_double_angle_variants() {
        let cases = [
            (
                "1 - cosh(2*x)",
                "-2*sinh(x)^2",
                DeriveHyperbolicRewriteKind::RecognizeOneMinusCoshDoubleAngleAsNegativeTwoSinhSq,
            ),
            (
                "-2*sinh(x)^2",
                "1 - cosh(2*x)",
                DeriveHyperbolicRewriteKind::ExpandNegativeTwoSinhSqToOneMinusCoshDoubleAngle,
            ),
            (
                "-1 - cosh(2*x)",
                "-2*cosh(x)^2",
                DeriveHyperbolicRewriteKind::RecognizeNegativeOneMinusCoshDoubleAngleAsNegativeTwoCoshSq,
            ),
            (
                "-2*cosh(x)^2",
                "-1 - cosh(2*x)",
                DeriveHyperbolicRewriteKind::ExpandNegativeTwoCoshSqToNegativeOneMinusCoshDoubleAngle,
            ),
        ];

        for (source_text, target_text, expected_kind) in cases {
            let mut ctx = cas_ast::Context::new();
            let source = cas_parser::parse(source_text, &mut ctx).expect("parse source");
            let target = cas_parser::parse(target_text, &mut ctx).expect("parse target");
            let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
                .expect("expected hyperbolic target-aware rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert_eq!(rewrite.rewritten, target);
        }
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_recognizes_cosh_double_angle_plus_one_as_two_cosh_sq() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("cosh(2*x) + 1", &mut ctx).expect("parse source");
        let target = cas_parser::parse("2*cosh(x)^2", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::RecognizeCoshDoubleAnglePlusOneAsTwoCoshSq
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_expands_two_cosh_sq_to_cosh_double_angle_plus_one() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("2*cosh(x)^2", &mut ctx).expect("parse source");
        let target = cas_parser::parse("cosh(2*x) + 1", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::ExpandTwoCoshSqToCoshDoubleAnglePlusOne
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_recognizes_two_cosh_sq_minus_one_as_cosh_double_angle() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("2*cosh(x)^2 - 1", &mut ctx).expect("parse source");
        let target = cas_parser::parse("cosh(2*x)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::RecognizeTwoCoshSqMinusOneAsCoshDoubleAngle
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn target_aware_hyperbolic_rewrite_recognizes_two_sinh_sq_plus_one_as_cosh_double_angle() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("2*sinh(x)^2 + 1", &mut ctx).expect("parse source");
        let target = cas_parser::parse("cosh(2*x)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_hyperbolic_simplify_target_aware(&mut ctx, source, target)
            .expect("expected hyperbolic target-aware rewrite");

        assert_eq!(
            rewrite.kind,
            DeriveHyperbolicRewriteKind::RecognizeTwoSinhSqPlusOneAsCoshDoubleAngle
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn generates_hyperbolic_bridge_rewrite_for_sinh_angle_sum_diff_contraction() {
        let mut ctx = cas_ast::Context::new();
        let source =
            cas_parser::parse("sinh(2*x)*cosh(x)+cosh(2*x)*sinh(x)", &mut ctx).expect("parse");
        let target = cas_parser::parse("sinh(3*x)", &mut ctx).expect("parse target");
        let rewrites = generate_hyperbolic_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::ContractSinhAngleSumDiff
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }

    #[test]
    fn generates_hyperbolic_bridge_rewrite_for_sinh_exponential_definition() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("sinh(3*x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("(e^(3*x)-e^(-3*x))/2", &mut ctx).expect("parse target");
        let rewrites = generate_hyperbolic_bridge_rewrites(&mut ctx, source);

        assert!(rewrites.iter().any(|rewrite| {
            rewrite.kind == DeriveHyperbolicRewriteKind::ExpandSinhToExpHalfDiff
                && matches_target_modulo_simplify(&mut ctx, rewrite.rewritten, target)
        }));
    }
}
