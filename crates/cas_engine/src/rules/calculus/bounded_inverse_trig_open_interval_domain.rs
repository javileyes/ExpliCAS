use super::atanh_open_interval_domain::arg_is_proven_outside_open_unit_interval;
use super::presentation_utils::squared_expr;
use super::scalar_presentation::subtract_from_one_for_calculus_presentation;
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::{One, Signed};

const CALCULUS_DOMAIN_PROOF_DEPTH: usize = 12;

pub(super) fn bounded_inverse_trig_known_empty_open_interval_gap(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    match ctx.get(target).clone() {
        Expr::Neg(inner) | Expr::Hold(inner) => {
            return bounded_inverse_trig_known_empty_open_interval_gap(ctx, inner, var_name);
        }
        Expr::Div(num, den) if cas_ast::views::as_rational_const(ctx, den, 8).is_some() => {
            return bounded_inverse_trig_known_empty_open_interval_gap(ctx, num, var_name);
        }
        Expr::Mul(left, right) if cas_ast::views::as_rational_const(ctx, left, 8).is_some() => {
            return bounded_inverse_trig_known_empty_open_interval_gap(ctx, right, var_name);
        }
        Expr::Mul(left, right) if cas_ast::views::as_rational_const(ctx, right, 8).is_some() => {
            return bounded_inverse_trig_known_empty_open_interval_gap(ctx, left, var_name);
        }
        Expr::Add(left, right) if domain_policy_constant_offset(ctx, left) => {
            return bounded_inverse_trig_known_empty_open_interval_gap(ctx, right, var_name);
        }
        Expr::Add(left, right) if domain_policy_constant_offset(ctx, right) => {
            return bounded_inverse_trig_known_empty_open_interval_gap(ctx, left, var_name);
        }
        Expr::Sub(left, right) if domain_policy_constant_offset(ctx, left) => {
            return bounded_inverse_trig_known_empty_open_interval_gap(ctx, right, var_name);
        }
        Expr::Sub(left, right) if domain_policy_constant_offset(ctx, right) => {
            return bounded_inverse_trig_known_empty_open_interval_gap(ctx, left, var_name);
        }
        _ => {}
    }

    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(fn_id),
            Some(BuiltinFn::Arcsin | BuiltinFn::Asin | BuiltinFn::Arccos | BuiltinFn::Acos)
        )
    {
        return None;
    }

    let arg = args[0];
    if bounded_inverse_trig_finite_constant_domain_arg(ctx, arg) {
        return None;
    }

    let gap = bounded_inverse_trig_open_interval_gap(ctx, arg);
    let mut scratch = ctx.clone();
    if cas_math::calculus_domain_support::bounded_inverse_real_domain_rejection_over_reals(
        &mut scratch,
        ctx.builtin_of(fn_id),
        &args,
        CALCULUS_DOMAIN_PROOF_DEPTH,
    )
    .is_some()
        || (contains_named_var(ctx, arg, var_name)
            && arg_is_proven_outside_open_unit_interval(ctx, arg))
    {
        return Some(gap);
    }

    None
}

fn bounded_inverse_trig_finite_constant_domain_arg(ctx: &Context, arg: ExprId) -> bool {
    cas_math::numeric_eval::as_rational_const(ctx, arg)
        .is_some_and(|value| value.abs() <= BigRational::one())
}

fn domain_policy_constant_offset(ctx: &Context, expr: ExprId) -> bool {
    cas_ast::views::as_rational_const(ctx, expr, 8).is_some()
        || matches!(ctx.get(expr), Expr::Constant(Constant::Pi | Constant::E))
}

fn bounded_inverse_trig_open_interval_gap(ctx: &mut Context, arg: ExprId) -> ExprId {
    if let Expr::Neg(inner) = ctx.get(arg) {
        return bounded_inverse_trig_open_interval_gap(ctx, *inner);
    }
    if let Some(radicand) = extract_square_root_base(ctx, arg) {
        return subtract_from_one_for_calculus_presentation(ctx, radicand);
    }

    let arg_square = squared_expr(ctx, arg);
    subtract_from_one_for_calculus_presentation(ctx, arg_square)
}

pub(super) fn inverse_reciprocal_trig_bounded_trig_empty_open_interval_gap(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, bool)> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let (trig_arg, should_emit_hint) = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Arcsec | BuiltinFn::Asec | BuiltinFn::Arccsc | BuiltinFn::Acsc) => {
            variable_dependent_direct_bounded_trig_arg(ctx, args[0], var_name)?;
            (args[0], true)
        }
        Some(BuiltinFn::Arcsin | BuiltinFn::Asin | BuiltinFn::Arccos | BuiltinFn::Acos) => {
            let canonical = variable_dependent_reciprocal_bounded_trig_arg(ctx, args[0], var_name)?;
            (args[0], canonical)
        }
        _ => return None,
    };

    let two = ctx.num(2);
    let trig_square = ctx.add(Expr::Pow(trig_arg, two));
    match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Arcsec | BuiltinFn::Asec | BuiltinFn::Arccsc | BuiltinFn::Acsc) => {
            let one = ctx.num(1);
            Some((ctx.add(Expr::Sub(trig_square, one)), should_emit_hint))
        }
        Some(BuiltinFn::Arcsin | BuiltinFn::Asin | BuiltinFn::Arccos | BuiltinFn::Acos) => {
            let one = ctx.num(1);
            Some((ctx.add(Expr::Sub(one, trig_square)), should_emit_hint))
        }
        _ => None,
    }
}

fn variable_dependent_direct_bounded_trig_arg(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<()> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    (args.len() == 1
        && matches!(
            ctx.builtin_of(*fn_id),
            Some(BuiltinFn::Sin | BuiltinFn::Cos)
        )
        && contains_named_var(ctx, args[0], var_name))
    .then_some(())
}

fn variable_dependent_reciprocal_bounded_trig_arg(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<bool> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(
                    ctx.builtin_of(*fn_id),
                    Some(BuiltinFn::Sec | BuiltinFn::Csc)
                )
                && contains_named_var(ctx, args[0], var_name) =>
        {
            Some(true)
        }
        Expr::Div(num, den)
            if cas_ast::views::as_rational_const(ctx, *num, 8)
                .is_some_and(|value| value.is_one()) =>
        {
            variable_dependent_direct_bounded_trig_arg(ctx, *den, var_name).map(|()| false)
        }
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, *exp, 8)
                .is_some_and(|value| value == -BigRational::one()) =>
        {
            variable_dependent_direct_bounded_trig_arg(ctx, *base, var_name).map(|()| false)
        }
        _ => None,
    }
}
