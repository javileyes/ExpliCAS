//! Reciprocal-trig policy helpers for symbolic integration.
//!
//! This module owns the small detection and primitive-construction policy for
//! routes based on `1/cos(u)^2`, `1/sin(u)^2`, `sec(u) tan(u)`,
//! `csc(u) cot(u)`, and trig log-derivative numerator pairing including
//! source-side `tan`/`cot` evidence. Higher-level integration routes stay in
//! `symbolic_integration_support` so route order remains explicit there.

use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::One;
use std::cmp::Ordering;

pub(crate) fn reciprocal_trig_square_parts(
    ctx: &Context,
    den: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    let (base, exp) = match ctx.get(den) {
        Expr::Pow(base, exp) => (*base, *exp),
        _ => return None,
    };
    if !is_number(ctx, exp, 2) {
        return None;
    }

    let (builtin, arg) = match ctx.get(base) {
        Expr::Function(fn_id, args) if args.len() == 1 => (ctx.builtin_of(*fn_id)?, args[0]),
        _ => return None,
    };
    match builtin {
        BuiltinFn::Cos | BuiltinFn::Sin => Some((builtin, arg)),
        _ => None,
    }
}

pub(crate) fn indexed_reciprocal_trig_square_parts(
    ctx: &Context,
    factors: &[ExprId],
) -> Option<(usize, (BuiltinFn, ExprId))> {
    factors.iter().enumerate().find_map(|(idx, factor)| {
        reciprocal_trig_square_parts(ctx, *factor).map(|parts| (idx, parts))
    })
}

#[derive(Clone, Copy)]
pub(crate) struct ReciprocalTrigDerivativePolicy {
    denominator_builtin: BuiltinFn,
    numerator_builtin: BuiltinFn,
    reciprocal_builtin: BuiltinFn,
    derivative_builtin: BuiltinFn,
    antiderivative_negative: bool,
}

impl ReciprocalTrigDerivativePolicy {
    pub(crate) fn denominator_builtin(self) -> BuiltinFn {
        self.denominator_builtin
    }

    pub(crate) fn numerator_builtin(self) -> BuiltinFn {
        self.numerator_builtin
    }

    pub(crate) fn derivative_builtin(self) -> BuiltinFn {
        self.derivative_builtin
    }

    fn reciprocal_builtin(self) -> BuiltinFn {
        self.reciprocal_builtin
    }

    fn antiderivative_negative(self) -> bool {
        self.antiderivative_negative
    }
}

pub(crate) fn reciprocal_trig_derivative_policy(
    denominator_builtin: BuiltinFn,
) -> Option<ReciprocalTrigDerivativePolicy> {
    match denominator_builtin {
        BuiltinFn::Cos => Some(ReciprocalTrigDerivativePolicy {
            denominator_builtin,
            numerator_builtin: BuiltinFn::Sin,
            reciprocal_builtin: BuiltinFn::Sec,
            derivative_builtin: BuiltinFn::Tan,
            antiderivative_negative: false,
        }),
        BuiltinFn::Sin => Some(ReciprocalTrigDerivativePolicy {
            denominator_builtin,
            numerator_builtin: BuiltinFn::Cos,
            reciprocal_builtin: BuiltinFn::Csc,
            derivative_builtin: BuiltinFn::Cot,
            antiderivative_negative: true,
        }),
        _ => None,
    }
}

pub(crate) fn reciprocal_trig_derivative_policy_from_reciprocal(
    reciprocal_builtin: BuiltinFn,
) -> Option<ReciprocalTrigDerivativePolicy> {
    match reciprocal_builtin {
        BuiltinFn::Sec => reciprocal_trig_derivative_policy(BuiltinFn::Cos),
        BuiltinFn::Csc => reciprocal_trig_derivative_policy(BuiltinFn::Sin),
        _ => None,
    }
}

pub(crate) fn build_reciprocal_trig_derivative_integral(
    ctx: &mut Context,
    policy: ReciprocalTrigDerivativePolicy,
    arg: ExprId,
) -> ExprId {
    let reciprocal = ctx.call_builtin(policy.reciprocal_builtin(), vec![arg]);
    if policy.antiderivative_negative() {
        ctx.add(Expr::Neg(reciprocal))
    } else {
        reciprocal
    }
}

pub(crate) fn reciprocal_trig_derivative_base_antiderivative(
    ctx: &mut Context,
    denominator_builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    let policy = reciprocal_trig_derivative_policy(denominator_builtin)?;
    Some(build_reciprocal_trig_derivative_integral(ctx, policy, arg))
}

pub(crate) fn is_reciprocal_trig_call(ctx: &Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Sec | BuiltinFn::Csc))
    )
}

pub(crate) fn reciprocal_trig_denominator_call(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    let (builtin, arg) = match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => (ctx.builtin_of(*fn_id)?, args[0]),
        _ => return None,
    };
    match builtin {
        BuiltinFn::Cos | BuiltinFn::Sin => Some((builtin, arg)),
        _ => None,
    }
}

pub(crate) fn indexed_reciprocal_trig_denominator_call(
    ctx: &Context,
    factors: &[ExprId],
) -> Option<(usize, (BuiltinFn, ExprId))> {
    factors.iter().enumerate().find_map(|(idx, factor)| {
        reciprocal_trig_denominator_call(ctx, *factor).map(|parts| (idx, parts))
    })
}

pub(crate) fn trig_log_derivative_numerator_builtin(
    denominator_builtin: BuiltinFn,
) -> Option<BuiltinFn> {
    match denominator_builtin {
        BuiltinFn::Cos => Some(BuiltinFn::Sin),
        BuiltinFn::Sin => Some(BuiltinFn::Cos),
        _ => None,
    }
}

pub(crate) fn indexed_trig_log_derivative_numerator_factor(
    ctx: &Context,
    factors: &[ExprId],
    denominator_builtin: BuiltinFn,
    expected_arg: ExprId,
) -> Option<usize> {
    let numerator_builtin = trig_log_derivative_numerator_builtin(denominator_builtin)?;
    factors.iter().enumerate().find_map(|(idx, factor)| {
        unary_builtin_arg(ctx, *factor, numerator_builtin)
            .is_some_and(|arg| compare_expr(ctx, arg, expected_arg) == Ordering::Equal)
            .then_some(idx)
    })
}

pub(crate) fn indexed_trig_log_derivative_raw_numerator_factor(
    ctx: &Context,
    factors: &[ExprId],
) -> Option<(BuiltinFn, ExprId, usize, BigRational)> {
    factors.iter().enumerate().find_map(|(idx, factor)| {
        if let Some((arg, sign)) = signed_unary_builtin_arg(ctx, *factor, BuiltinFn::Tan) {
            return Some((BuiltinFn::Cos, arg, idx, sign));
        }
        if let Some((arg, sign)) = signed_unary_builtin_arg(ctx, *factor, BuiltinFn::Cot) {
            return Some((BuiltinFn::Sin, arg, idx, sign));
        }
        None
    })
}

pub(crate) fn indexed_trig_pole_builtin_factor(
    ctx: &Context,
    factors: &[ExprId],
) -> Option<(usize, BuiltinFn, ExprId)> {
    factors.iter().enumerate().find_map(|(idx, factor)| {
        trig_pole_builtin_arg(ctx, *factor).map(|(builtin, arg)| (idx, builtin, arg))
    })
}

pub(crate) fn has_trig_pole_builtin_factor_except(
    ctx: &Context,
    factors: &[ExprId],
    excluded_index: usize,
) -> bool {
    factors
        .iter()
        .enumerate()
        .any(|(idx, factor)| idx != excluded_index && trig_pole_builtin_arg(ctx, *factor).is_some())
}

pub(crate) fn reciprocal_trig_reciprocal_parts_from_denominator(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    let (denominator_builtin, arg) = reciprocal_trig_denominator_call(ctx, expr)?;
    let policy = reciprocal_trig_derivative_policy(denominator_builtin)?;
    Some((policy.reciprocal_builtin(), arg))
}

pub(crate) fn build_reciprocal_trig_log_argument(
    ctx: &mut Context,
    reciprocal_builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    match reciprocal_builtin {
        BuiltinFn::Sec => {
            let primary = ctx.call_builtin(BuiltinFn::Sec, vec![arg]);
            let companion = ctx.call_builtin(BuiltinFn::Tan, vec![arg]);
            Some(ctx.add(Expr::Add(primary, companion)))
        }
        BuiltinFn::Csc => {
            let primary = ctx.call_builtin(BuiltinFn::Csc, vec![arg]);
            let companion = ctx.call_builtin(BuiltinFn::Cot, vec![arg]);
            Some(ctx.add(Expr::Sub(primary, companion)))
        }
        _ => None,
    }
}

pub(crate) fn trig_pole_nonzero_builtin(builtin: BuiltinFn) -> Option<BuiltinFn> {
    match builtin {
        BuiltinFn::Tan | BuiltinFn::Sec => Some(BuiltinFn::Cos),
        BuiltinFn::Cot | BuiltinFn::Csc => Some(BuiltinFn::Sin),
        _ => None,
    }
}

pub(crate) fn build_trig_pole_nonzero_condition(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    let nonzero_builtin = trig_pole_nonzero_builtin(builtin)?;
    Some(ctx.call_builtin(nonzero_builtin, vec![arg]))
}

pub(crate) fn build_reciprocal_trig_denominator_nonzero_condition(
    ctx: &mut Context,
    denominator_builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    match denominator_builtin {
        BuiltinFn::Cos | BuiltinFn::Sin => Some(ctx.call_builtin(denominator_builtin, vec![arg])),
        _ => None,
    }
}

fn is_number(ctx: &Context, expr: ExprId, value: i64) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if *n == BigRational::from_integer(value.into()))
}

fn unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(builtin) =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

fn trig_pole_builtin_arg(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let (builtin, arg) = match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => (ctx.builtin_of(*fn_id)?, args[0]),
        _ => return None,
    };
    matches!(
        builtin,
        BuiltinFn::Tan | BuiltinFn::Cot | BuiltinFn::Sec | BuiltinFn::Csc
    )
    .then_some((builtin, arg))
}

fn signed_unary_builtin_arg(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<(ExprId, BigRational)> {
    match ctx.get(expr) {
        Expr::Neg(inner) => {
            unary_builtin_arg(ctx, *inner, builtin).map(|arg| (arg, -BigRational::one()))
        }
        _ => unary_builtin_arg(ctx, expr, builtin).map(|arg| (arg, BigRational::one())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn detects_reciprocal_trig_square_denominators() {
        let mut ctx = Context::new();
        let cos_den = parse("cos(2*x+1)^2", &mut ctx).unwrap();
        let sin_den = parse("sin(2*x+1)^2", &mut ctx).unwrap();
        let tan_den = parse("tan(2*x+1)^2", &mut ctx).unwrap();

        let (cos_builtin, cos_arg) = reciprocal_trig_square_parts(&ctx, cos_den).unwrap();
        let (sin_builtin, sin_arg) = reciprocal_trig_square_parts(&ctx, sin_den).unwrap();

        assert_eq!(cos_builtin, BuiltinFn::Cos);
        assert_eq!(rendered(&ctx, cos_arg), "2 * x + 1");
        assert_eq!(sin_builtin, BuiltinFn::Sin);
        assert_eq!(rendered(&ctx, sin_arg), "2 * x + 1");
        assert!(reciprocal_trig_square_parts(&ctx, tan_den).is_none());
    }

    #[test]
    fn maps_derivative_policy_and_base_antiderivative() {
        let mut ctx = Context::new();
        let arg = parse("2*x+1", &mut ctx).unwrap();

        let sec_integral =
            reciprocal_trig_derivative_base_antiderivative(&mut ctx, BuiltinFn::Cos, arg).unwrap();
        let csc_integral =
            reciprocal_trig_derivative_base_antiderivative(&mut ctx, BuiltinFn::Sin, arg).unwrap();

        assert_eq!(rendered(&ctx, sec_integral), "sec(2 * x + 1)");
        assert_eq!(rendered(&ctx, csc_integral), "-csc(2 * x + 1)");
    }

    #[test]
    fn maps_reciprocal_builtin_to_derivative_policy() {
        let sec_policy = reciprocal_trig_derivative_policy_from_reciprocal(BuiltinFn::Sec).unwrap();
        let csc_policy = reciprocal_trig_derivative_policy_from_reciprocal(BuiltinFn::Csc).unwrap();

        assert_eq!(sec_policy.denominator_builtin(), BuiltinFn::Cos);
        assert_eq!(sec_policy.derivative_builtin(), BuiltinFn::Tan);
        assert_eq!(csc_policy.denominator_builtin(), BuiltinFn::Sin);
        assert_eq!(csc_policy.derivative_builtin(), BuiltinFn::Cot);
        assert!(reciprocal_trig_derivative_policy_from_reciprocal(BuiltinFn::Tan).is_none());
    }

    #[test]
    fn detects_reciprocal_trig_calls() {
        let mut ctx = Context::new();
        let sec_expr = parse("sec(2*x+1)", &mut ctx).unwrap();
        let csc_expr = parse("csc(2*x+1)", &mut ctx).unwrap();
        let tan_expr = parse("tan(2*x+1)", &mut ctx).unwrap();

        assert!(is_reciprocal_trig_call(&ctx, sec_expr));
        assert!(is_reciprocal_trig_call(&ctx, csc_expr));
        assert!(!is_reciprocal_trig_call(&ctx, tan_expr));
    }

    #[test]
    fn maps_denominator_calls_to_reciprocal_trig_policy_parts() {
        let mut ctx = Context::new();
        let cos_den = parse("cos(2*x+1)", &mut ctx).unwrap();
        let sin_den = parse("sin(2*x+1)", &mut ctx).unwrap();
        let tan_den = parse("tan(2*x+1)", &mut ctx).unwrap();

        let (cos_builtin, cos_arg) = reciprocal_trig_denominator_call(&ctx, cos_den).unwrap();
        let (sin_builtin, sin_arg) = reciprocal_trig_denominator_call(&ctx, sin_den).unwrap();
        let (sec_builtin, sec_arg) =
            reciprocal_trig_reciprocal_parts_from_denominator(&ctx, cos_den).unwrap();
        let (csc_builtin, csc_arg) =
            reciprocal_trig_reciprocal_parts_from_denominator(&ctx, sin_den).unwrap();

        assert_eq!(cos_builtin, BuiltinFn::Cos);
        assert_eq!(rendered(&ctx, cos_arg), "2 * x + 1");
        assert_eq!(sin_builtin, BuiltinFn::Sin);
        assert_eq!(rendered(&ctx, sin_arg), "2 * x + 1");
        assert_eq!(sec_builtin, BuiltinFn::Sec);
        assert_eq!(rendered(&ctx, sec_arg), "2 * x + 1");
        assert_eq!(csc_builtin, BuiltinFn::Csc);
        assert_eq!(rendered(&ctx, csc_arg), "2 * x + 1");
        assert!(reciprocal_trig_denominator_call(&ctx, tan_den).is_none());
        assert!(reciprocal_trig_reciprocal_parts_from_denominator(&ctx, tan_den).is_none());
    }

    #[test]
    fn maps_trig_log_derivative_numerator_factor_from_denominator() {
        let mut ctx = Context::new();
        let arg = parse("2*x+1", &mut ctx).unwrap();
        let wrong_arg = parse("x", &mut ctx).unwrap();
        let factors = [
            parse("3", &mut ctx).unwrap(),
            parse("sin(2*x+1)", &mut ctx).unwrap(),
        ];

        assert_eq!(
            trig_log_derivative_numerator_builtin(BuiltinFn::Cos),
            Some(BuiltinFn::Sin)
        );
        assert_eq!(
            trig_log_derivative_numerator_builtin(BuiltinFn::Sin),
            Some(BuiltinFn::Cos)
        );
        assert_eq!(
            indexed_trig_log_derivative_numerator_factor(&ctx, &factors, BuiltinFn::Cos, arg),
            Some(1)
        );
        assert_eq!(
            indexed_trig_log_derivative_numerator_factor(&ctx, &factors, BuiltinFn::Cos, wrong_arg),
            None
        );
        assert_eq!(
            indexed_trig_log_derivative_numerator_factor(&ctx, &factors, BuiltinFn::Tan, arg),
            None
        );
    }

    #[test]
    fn maps_raw_tan_cot_log_derivative_numerator_factors() {
        let mut ctx = Context::new();
        let tan_factors = [
            parse("3", &mut ctx).unwrap(),
            parse("-tan(2*x+1)", &mut ctx).unwrap(),
        ];
        let cot_factors = [
            parse("5", &mut ctx).unwrap(),
            parse("cot(x^2)", &mut ctx).unwrap(),
        ];
        let invalid_factors = [parse("sin(x)", &mut ctx).unwrap()];

        let (tan_den_builtin, tan_arg, tan_index, tan_sign) =
            indexed_trig_log_derivative_raw_numerator_factor(&ctx, &tan_factors).unwrap();
        let (cot_den_builtin, cot_arg, cot_index, cot_sign) =
            indexed_trig_log_derivative_raw_numerator_factor(&ctx, &cot_factors).unwrap();

        assert_eq!(tan_den_builtin, BuiltinFn::Cos);
        assert_eq!(rendered(&ctx, tan_arg), "2 * x + 1");
        assert_eq!(tan_index, 1);
        assert_eq!(tan_sign, -BigRational::one());
        assert_eq!(cot_den_builtin, BuiltinFn::Sin);
        assert_eq!(rendered(&ctx, cot_arg), "x^2");
        assert_eq!(cot_index, 1);
        assert_eq!(cot_sign, BigRational::one());
        assert!(indexed_trig_log_derivative_raw_numerator_factor(&ctx, &invalid_factors).is_none());
    }

    #[test]
    fn maps_trig_pole_builtin_factor_without_widening_to_sin_cos() {
        let mut ctx = Context::new();
        let factors = [
            parse("2", &mut ctx).unwrap(),
            parse("sec(2*x+1)", &mut ctx).unwrap(),
        ];
        let invalid_factors = [
            parse("2", &mut ctx).unwrap(),
            parse("cos(2*x+1)", &mut ctx).unwrap(),
        ];

        let (idx, builtin, arg) = indexed_trig_pole_builtin_factor(&ctx, &factors).unwrap();

        assert_eq!(idx, 1);
        assert_eq!(builtin, BuiltinFn::Sec);
        assert_eq!(rendered(&ctx, arg), "2 * x + 1");
        assert!(!has_trig_pole_builtin_factor_except(&ctx, &factors, 1));
        assert!(indexed_trig_pole_builtin_factor(&ctx, &invalid_factors).is_none());
    }

    #[test]
    fn detects_additional_trig_pole_builtin_factor_except_selected_index() {
        let mut ctx = Context::new();
        let factors = [
            parse("tan(x)", &mut ctx).unwrap(),
            parse("3", &mut ctx).unwrap(),
            parse("csc(x)", &mut ctx).unwrap(),
        ];

        assert!(has_trig_pole_builtin_factor_except(&ctx, &factors, 0));
        assert!(has_trig_pole_builtin_factor_except(&ctx, &factors, 2));
        assert!(!has_trig_pole_builtin_factor_except(&ctx, &factors[..1], 0));
    }

    #[test]
    fn builds_reciprocal_trig_log_arguments() {
        let mut ctx = Context::new();
        let arg = parse("2*x+1", &mut ctx).unwrap();

        let sec_arg = build_reciprocal_trig_log_argument(&mut ctx, BuiltinFn::Sec, arg).unwrap();
        let csc_arg = build_reciprocal_trig_log_argument(&mut ctx, BuiltinFn::Csc, arg).unwrap();

        assert_eq!(rendered(&ctx, sec_arg), "tan(2 * x + 1) + sec(2 * x + 1)");
        assert_eq!(rendered(&ctx, csc_arg), "csc(2 * x + 1) - cot(2 * x + 1)");
        assert!(build_reciprocal_trig_log_argument(&mut ctx, BuiltinFn::Tan, arg).is_none());
    }

    #[test]
    fn builds_trig_nonzero_conditions_without_widening_pole_policy() {
        let mut ctx = Context::new();
        let arg = parse("2*x+1", &mut ctx).unwrap();

        let sec_pole = build_trig_pole_nonzero_condition(&mut ctx, BuiltinFn::Sec, arg).unwrap();
        let cot_pole = build_trig_pole_nonzero_condition(&mut ctx, BuiltinFn::Cot, arg).unwrap();
        let cos_den =
            build_reciprocal_trig_denominator_nonzero_condition(&mut ctx, BuiltinFn::Cos, arg)
                .unwrap();
        let sin_den =
            build_reciprocal_trig_denominator_nonzero_condition(&mut ctx, BuiltinFn::Sin, arg)
                .unwrap();

        assert_eq!(rendered(&ctx, sec_pole), "cos(2 * x + 1)");
        assert_eq!(rendered(&ctx, cot_pole), "sin(2 * x + 1)");
        assert_eq!(rendered(&ctx, cos_den), "cos(2 * x + 1)");
        assert_eq!(rendered(&ctx, sin_den), "sin(2 * x + 1)");
        assert!(build_trig_pole_nonzero_condition(&mut ctx, BuiltinFn::Cos, arg).is_none());
        assert!(
            build_reciprocal_trig_denominator_nonzero_condition(&mut ctx, BuiltinFn::Tan, arg)
                .is_none()
        );
    }
}
