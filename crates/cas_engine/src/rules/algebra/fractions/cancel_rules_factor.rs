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
use cas_math::polynomial::Polynomial;
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
            || is_post_calculus_unit_interval_sqrt_product_denominator(ctx, expr)
            || is_post_calculus_sqrt_variable_polynomial_denominator(ctx, expr)
            || is_post_calculus_trig_elementary_sqrt_denominator(ctx, expr)
            || is_post_calculus_tan_sum_sqrt_denominator_with_matching_cos_square(ctx, expr)
        {
            return None;
        }
        let rewrite = try_rewrite_rationalize_product_denominator_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc("Rationalize product denominator"))
    }
);

fn is_post_calculus_sqrt_variable_polynomial_denominator(ctx: &mut Context, expr: ExprId) -> bool {
    let fp = FractionParts::from(&*ctx, expr);
    if !fp.is_fraction() {
        return false;
    }

    let (num, den, _) = fp.to_num_den(ctx);
    let factors = collect_mul_factors_flat(ctx, den);
    let Some((sqrt_base, var_name)) = factors.iter().find_map(|factor| {
        let base = extract_square_root_base(ctx, *factor)?;
        positive_scaled_variable_root_base(ctx, base).map(|var_name| (base, var_name))
    }) else {
        return false;
    };

    let Ok(numerator_poly) = Polynomial::from_expr(ctx, num, &var_name) else {
        return false;
    };
    if numerator_poly.degree() == 0 || numerator_poly.degree() > 1 {
        return false;
    }

    factors.iter().any(|factor| {
        if cas_ast::views::as_rational_const(ctx, *factor, 8).is_some() {
            return false;
        }
        if extract_square_root_base(ctx, *factor)
            .is_some_and(|base| cas_ast::ordering::compare_expr(ctx, base, sqrt_base).is_eq())
        {
            return false;
        }

        Polynomial::from_expr(ctx, *factor, &var_name)
            .is_ok_and(|poly| poly.degree() >= 1 && poly.degree() <= 3)
    })
}

fn positive_scaled_variable_root_base(ctx: &Context, base: ExprId) -> Option<String> {
    if let Expr::Variable(sym_id) = ctx.get(base) {
        return Some(ctx.sym_name(*sym_id).to_string());
    }

    let factors = collect_mul_factors_flat(ctx, base);
    if factors.len() < 2 {
        return None;
    }

    let mut var_name = None;
    for factor in factors {
        if let Expr::Variable(sym_id) = ctx.get(factor) {
            if var_name.is_some() {
                return None;
            }
            var_name = Some(ctx.sym_name(*sym_id).to_string());
            continue;
        }

        let value = cas_ast::views::as_rational_const(ctx, factor, 8)?;
        if value <= num_rational::BigRational::from_integer(0.into()) {
            return None;
        }
    }

    var_name
}

fn is_post_calculus_trig_elementary_sqrt_denominator(ctx: &mut Context, expr: ExprId) -> bool {
    let fp = FractionParts::from(&*ctx, expr);
    if !fp.is_fraction() {
        return false;
    }

    let (_num, den, _) = fp.to_num_den(ctx);
    collect_mul_factors_flat(ctx, den)
        .into_iter()
        .filter_map(|factor| extract_square_root_base(ctx, factor))
        .any(|base| additive_base_has_trig_and_elementary_term(ctx, base))
}

fn is_post_calculus_tan_sum_sqrt_denominator_with_matching_cos_square(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let fp = FractionParts::from(&*ctx, expr);
    if !fp.is_fraction() {
        return false;
    }

    let (_num, den, _) = fp.to_num_den(ctx);
    let factors = collect_mul_factors_flat(ctx, den);
    let Some((root_base, tan_arg)) = factors.iter().find_map(|factor| {
        let root_base = extract_square_root_base(ctx, *factor)?;
        additive_base_tan_arg(ctx, root_base).map(|tan_arg| (root_base, tan_arg))
    }) else {
        return false;
    };

    factors.iter().any(|factor| {
        !extract_square_root_base(ctx, *factor)
            .is_some_and(|base| cas_ast::ordering::compare_expr(ctx, base, root_base).is_eq())
            && is_cos_square_of_arg(ctx, *factor, tan_arg)
    })
}

fn additive_base_tan_arg(ctx: &Context, base: ExprId) -> Option<ExprId> {
    let terms = cas_math::expr_nary::add_terms_no_sign(ctx, base);
    if terms.len() < 2 || terms.len() > 8 {
        return None;
    }

    let mut tan_arg = None;
    for term in terms {
        let core = single_non_numeric_factor(ctx, term).unwrap_or(term);
        if let Some(arg) = tan_arg_of_term(ctx, core) {
            if tan_arg.is_some_and(|existing| {
                !cas_ast::ordering::compare_expr(ctx, existing, arg).is_eq()
            }) {
                return None;
            }
            tan_arg = Some(arg);
        }
    }
    tan_arg
}

fn tan_arg_of_term(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    (args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Tan)).then_some(args[0])
}

fn is_cos_square_of_arg(ctx: &Context, expr: ExprId, arg: ExprId) -> bool {
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
    args.len() == 1
        && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Cos)
        && cas_ast::ordering::compare_expr(ctx, args[0], arg).is_eq()
}

fn additive_base_has_trig_and_elementary_term(ctx: &Context, base: ExprId) -> bool {
    let terms = cas_math::expr_nary::add_terms_no_sign(ctx, base);
    if terms.len() < 2 || terms.len() > 8 {
        return false;
    }

    let mut has_trig = false;
    let mut has_elementary = false;
    for term in terms {
        let core = single_non_numeric_factor(ctx, term).unwrap_or(term);
        has_trig |= contains_sin_or_cos(ctx, core, 0);
        has_elementary |= is_ln_exp_or_sqrt_variable_term(ctx, core);
    }

    has_trig && has_elementary
}

fn single_non_numeric_factor(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let mut core = None;
    for factor in collect_mul_factors_flat(ctx, expr) {
        if cas_ast::views::as_rational_const(ctx, factor, 8).is_some() {
            continue;
        }
        if core.replace(factor).is_some() {
            return None;
        }
    }
    core
}

fn contains_sin_or_cos(ctx: &Context, expr: ExprId, depth: u8) -> bool {
    if depth > 6 {
        return false;
    }

    match ctx.get(expr) {
        Expr::Function(fn_id, args) => {
            matches!(
                ctx.builtin_of(*fn_id),
                Some(BuiltinFn::Sin | BuiltinFn::Cos)
            ) || args
                .iter()
                .any(|arg| contains_sin_or_cos(ctx, *arg, depth + 1))
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            contains_sin_or_cos(ctx, *left, depth + 1)
                || contains_sin_or_cos(ctx, *right, depth + 1)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => contains_sin_or_cos(ctx, *inner, depth + 1),
        _ => false,
    }
}

fn is_ln_exp_or_sqrt_variable_term(ctx: &Context, expr: ExprId) -> bool {
    if extract_square_root_base(ctx, expr)
        .is_some_and(|base| matches!(ctx.get(base), Expr::Variable(_)))
    {
        return true;
    }

    match ctx.get(expr) {
        Expr::Function(fn_id, args) => {
            args.len() == 1
                && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Ln | BuiltinFn::Exp))
        }
        Expr::Pow(base, _) => matches!(ctx.get(*base), Expr::Constant(cas_ast::Constant::E)),
        _ => false,
    }
}

fn is_post_calculus_unit_interval_sqrt_product_denominator(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let fp = FractionParts::from(&*ctx, expr);
    if !fp.is_fraction() {
        return false;
    }

    let (num, den, _) = fp.to_num_den(ctx);
    if cas_ast::views::as_rational_const(ctx, num, 8).is_none() {
        return false;
    }

    let factors = collect_mul_factors_flat(ctx, den);
    let sqrt_bases: Vec<ExprId> = factors
        .iter()
        .filter_map(|factor| extract_square_root_base(ctx, *factor))
        .collect();

    sqrt_bases.iter().enumerate().any(|(index, left)| {
        sqrt_bases
            .iter()
            .skip(index + 1)
            .any(|right| are_unit_interval_complements(ctx, *left, *right))
    })
}

fn are_unit_interval_complements(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    is_one_minus_expr(ctx, left, right) || is_one_minus_expr(ctx, right, left)
}

fn is_one_minus_expr(ctx: &Context, expr: ExprId, inner: ExprId) -> bool {
    let Expr::Sub(left, right) = ctx.get(expr) else {
        return false;
    };

    cas_ast::views::as_rational_const(ctx, *left, 8)
        .is_some_and(|value| value == num_rational::BigRational::from_integer(1.into()))
        && cas_ast::ordering::compare_expr(ctx, *right, inner).is_eq()
}

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
