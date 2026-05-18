//! Factor-based cancellation and rationalization rules.
//!
//! Contains `CancelCommonFactorsRule` (cancel shared factors in num/den),
//! `RationalizeProductDenominatorRule` (rationalize product denominators),
//! and supporting helper functions.

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::{ChainedRewrite, Rewrite};
use cas_ast::views::FractionParts;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::fraction_factors::collect_mul_factors_flat;
use cas_math::fraction_factors::try_rewrite_cancel_common_factors_expr_with;
use cas_math::fraction_factors::CancelCommonFactorsGate;
use cas_math::fraction_power_cancel_support::try_rewrite_cancel_identical_fraction_expr;
use cas_math::rationalize_diff_squares_support::try_rewrite_rationalize_product_denominator_expr;
use cas_math::root_forms::extract_square_root_base;

define_rule!(
    RationalizeProductDenominatorRule,
    "Rationalize Product Denominator",
    None,
    PhaseMask::RATIONALIZE,
    |ctx, expr| {
        if is_post_calculus_shifted_tan_sqrt_denominator(ctx, expr)
            || is_post_calculus_reciprocal_trig_sqrt_denominator(ctx, expr)
        {
            return None;
        }
        let rewrite = try_rewrite_rationalize_product_denominator_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc("Rationalize product denominator"))
    }
);

fn is_post_calculus_reciprocal_trig_sqrt_denominator(ctx: &mut Context, expr: ExprId) -> bool {
    let fp = FractionParts::from(&*ctx, expr);
    if !fp.is_fraction() {
        return false;
    }

    let (_num, den, _) = fp.to_num_den(ctx);
    let factors = collect_mul_factors_flat(ctx, den);
    let Some(root_base) = factors
        .iter()
        .find_map(|factor| extract_square_root_base(ctx, *factor))
    else {
        return false;
    };

    factors
        .iter()
        .any(|factor| is_sin_or_cos_of_sqrt_base(ctx, *factor, root_base))
}

fn is_post_calculus_shifted_tan_sqrt_denominator(ctx: &mut Context, expr: ExprId) -> bool {
    let fp = FractionParts::from(&*ctx, expr);
    if !fp.is_fraction() {
        return false;
    }

    let (_num, den, _) = fp.to_num_den(ctx);
    let factors = collect_mul_factors_flat(ctx, den);
    let Some(root_base) = factors
        .iter()
        .find_map(|factor| extract_square_root_base(ctx, *factor))
    else {
        return false;
    };

    let has_cos_square = factors
        .iter()
        .any(|factor| is_cos_square_of_sqrt_base(ctx, *factor, root_base));
    let has_shifted_tan = factors
        .iter()
        .any(|factor| is_shifted_tan_of_sqrt_base(ctx, *factor, root_base));

    has_cos_square && has_shifted_tan
}

fn is_sin_or_cos_of_sqrt_base(ctx: &Context, expr: ExprId, root_base: ExprId) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return false;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(BuiltinFn::Sin | BuiltinFn::Cos)
        )
    {
        return false;
    }
    extract_square_root_base(ctx, args[0])
        .is_some_and(|base| cas_ast::ordering::compare_expr(ctx, base, root_base).is_eq())
}

fn is_cos_square_of_sqrt_base(ctx: &Context, expr: ExprId, root_base: ExprId) -> bool {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return false;
    };
    if cas_ast::views::as_rational_const(ctx, *exp, 8)
        .is_none_or(|value| value != num_rational::BigRational::from_integer(2.into()))
    {
        return false;
    }

    let Expr::Function(fn_id, args) = ctx.get(*base) else {
        return false;
    };
    if args.len() != 1 || ctx.builtin_of(*fn_id) != Some(BuiltinFn::Cos) {
        return false;
    }
    extract_square_root_base(ctx, args[0])
        .is_some_and(|base| cas_ast::ordering::compare_expr(ctx, base, root_base).is_eq())
}

fn is_shifted_tan_of_sqrt_base(ctx: &Context, expr: ExprId, root_base: ExprId) -> bool {
    let (left, right) = match ctx.get(expr) {
        Expr::Add(left, right) | Expr::Sub(left, right) => (*left, *right),
        _ => return false,
    };

    (cas_ast::views::as_rational_const(ctx, left, 8).is_some()
        && is_tan_of_sqrt_base(ctx, right, root_base))
        || (cas_ast::views::as_rational_const(ctx, right, 8).is_some()
            && is_tan_of_sqrt_base(ctx, left, root_base))
}

fn is_tan_of_sqrt_base(ctx: &Context, expr: ExprId, root_base: ExprId) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return false;
    };
    if args.len() != 1 || ctx.builtin_of(*fn_id) != Some(BuiltinFn::Tan) {
        return false;
    }
    extract_square_root_base(ctx, args[0])
        .is_some_and(|base| cas_ast::ordering::compare_expr(ctx, base, root_base).is_eq())
}

define_rule!(
    CancelCommonFactorsRule,
    "Cancel Common Factors",
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::ImplicitCondition;
        use crate::Predicate;

        // Capture domain mode once at start
        let domain_mode = parent_ctx.domain_mode();
        // NOTE: Pythagorean identity simplification (k - k*sin² → k*cos²) has been
        // extracted to TrigPythagoreanSimplifyRule for pedagogical clarity.
        // CancelCommonFactorsRule now does pure factor cancellation.

        let rewrite = try_rewrite_cancel_common_factors_expr_with(
            ctx,
            expr,
            |ctx, nonzero_base, _emit_assumption| {
                let decision = crate::oracle_allows_with_hint(
                    ctx,
                    domain_mode,
                    parent_ctx.value_domain(),
                    &Predicate::NonZero(nonzero_base),
                    "Cancel Common Factors",
                );
                CancelCommonFactorsGate {
                    allow: decision.allow,
                    assumed: decision.assumption.is_some(),
                }
            },
        )?;
        let assumption_events: smallvec::SmallVec<[crate::AssumptionEvent; 1]> =
            rewrite
                .assumed_nonzero_targets
                .into_iter()
                .map(|target| crate::AssumptionEvent::nonzero(ctx, target))
                .collect();

        let mut out = Rewrite::new(rewrite.rewritten)
            .desc("Cancel common factors")
            .local(expr, rewrite.rewritten)
            .assume_all(assumption_events);

        // When factor cancellation exposes an identical residual fraction like x/x,
        // finish the closure here so Assume/Generic recover the symbolic NonZero
        // target instead of stopping at the partially cancelled form.
        if let Some(plan) = try_rewrite_cancel_identical_fraction_expr(ctx, rewrite.rewritten) {
            let decision = crate::oracle_allows_with_hint(
                ctx,
                domain_mode,
                parent_ctx.value_domain(),
                &Predicate::NonZero(plan.nonzero_target),
                "Cancel Common Factors",
            );

            if decision.allow {
                let mut cancel = ChainedRewrite::new(plan.rewritten)
                    .desc("Cancel: P/P -> 1")
                    .local(rewrite.rewritten, plan.rewritten)
                    .requires(ImplicitCondition::NonZero(plan.nonzero_target));
                for event in decision.assumption_events(ctx, plan.nonzero_target) {
                    cancel = cancel.assume(event);
                }
                out = out.chain(cancel);
            }
        }

        Some(out)
    }
);
