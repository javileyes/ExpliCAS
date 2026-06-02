//! Hyperbolic reciprocal policy helpers for symbolic integration.
//!
//! This module owns only the small detection and primitive-construction policy
//! for `1/cosh(u)^n`, `1/sinh(u)^n`, and derivative-product reciprocal
//! hyperbolic routes. Scaled primitive presentation primitives are injected by
//! the caller so route order and broader integration formatting stay explicit.

use crate::build::mul2_raw;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::One;

pub(crate) fn reciprocal_hyperbolic_power_arg(
    ctx: &Context,
    den: ExprId,
    builtin: BuiltinFn,
    power: i64,
) -> Option<ExprId> {
    let (base, exp) = match ctx.get(den) {
        Expr::Pow(base, exp) => (*base, *exp),
        _ => return None,
    };
    if !is_number(ctx, exp, power) {
        return None;
    }

    match ctx.get(base) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(builtin) =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

fn reciprocal_hyperbolic_square_arg(
    ctx: &Context,
    den: ExprId,
    builtin: BuiltinFn,
) -> Option<ExprId> {
    reciprocal_hyperbolic_power_arg(ctx, den, builtin, 2)
}

pub(crate) fn reciprocal_hyperbolic_cosh_square_arg(ctx: &Context, den: ExprId) -> Option<ExprId> {
    reciprocal_hyperbolic_square_arg(ctx, den, BuiltinFn::Cosh)
}

pub(crate) fn reciprocal_hyperbolic_sinh_square_arg(ctx: &Context, den: ExprId) -> Option<ExprId> {
    reciprocal_hyperbolic_square_arg(ctx, den, BuiltinFn::Sinh)
}

pub(crate) fn reciprocal_hyperbolic_square_parts(
    ctx: &Context,
    factor: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    if let Some(arg) = reciprocal_hyperbolic_cosh_square_arg(ctx, factor) {
        return Some((BuiltinFn::Cosh, arg));
    }
    if let Some(arg) = reciprocal_hyperbolic_sinh_square_arg(ctx, factor) {
        return Some((BuiltinFn::Sinh, arg));
    }
    None
}

#[derive(Clone, Copy)]
pub(crate) struct HyperbolicReciprocalDerivativePolicy {
    pub(crate) denominator_builtin: BuiltinFn,
    pub(crate) numerator_builtin: BuiltinFn,
}

#[derive(Clone, Copy)]
pub(crate) struct HyperbolicReciprocalTablePolicy {
    pub(crate) denominator_builtin: BuiltinFn,
    pub(crate) power: HyperbolicReciprocalTablePower,
}

#[derive(Clone, Copy)]
pub(crate) struct HyperbolicReciprocalPrimitiveScaleOps {
    scale_rational_term: fn(&mut Context, BigRational, ExprId) -> ExprId,
    rational_over_expr: fn(&mut Context, BigRational, ExprId) -> ExprId,
    scale_expr_reciprocal_integration_result: fn(&mut Context, ExprId, ExprId) -> ExprId,
    negate_scalar_expr: fn(&mut Context, ExprId) -> ExprId,
}

impl HyperbolicReciprocalPrimitiveScaleOps {
    pub(crate) fn new(
        scale_rational_term: fn(&mut Context, BigRational, ExprId) -> ExprId,
        rational_over_expr: fn(&mut Context, BigRational, ExprId) -> ExprId,
        scale_expr_reciprocal_integration_result: fn(&mut Context, ExprId, ExprId) -> ExprId,
        negate_scalar_expr: fn(&mut Context, ExprId) -> ExprId,
    ) -> Self {
        Self {
            scale_rational_term,
            rational_over_expr,
            scale_expr_reciprocal_integration_result,
            negate_scalar_expr,
        }
    }

    fn scale_rational_term(&self, ctx: &mut Context, scale: BigRational, term: ExprId) -> ExprId {
        (self.scale_rational_term)(ctx, scale, term)
    }

    fn rational_over_expr(
        &self,
        ctx: &mut Context,
        numerator: BigRational,
        denominator: ExprId,
    ) -> ExprId {
        (self.rational_over_expr)(ctx, numerator, denominator)
    }

    fn scale_expr_reciprocal_integration_result(
        &self,
        ctx: &mut Context,
        scale: ExprId,
        expr: ExprId,
    ) -> ExprId {
        (self.scale_expr_reciprocal_integration_result)(ctx, scale, expr)
    }

    fn negate_scalar_expr(&self, ctx: &mut Context, expr: ExprId) -> ExprId {
        (self.negate_scalar_expr)(ctx, expr)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HyperbolicReciprocalTablePower {
    Square,
    Fourth,
}

impl HyperbolicReciprocalTablePower {
    pub(crate) fn exponent(self) -> i64 {
        match self {
            HyperbolicReciprocalTablePower::Square => 2,
            HyperbolicReciprocalTablePower::Fourth => 4,
        }
    }
}

pub(crate) fn hyperbolic_reciprocal_table_policy(
    denominator_builtin: BuiltinFn,
    power: i64,
) -> Option<HyperbolicReciprocalTablePolicy> {
    if !matches!(denominator_builtin, BuiltinFn::Cosh | BuiltinFn::Sinh) {
        return None;
    }
    let power = match power {
        2 => HyperbolicReciprocalTablePower::Square,
        4 => HyperbolicReciprocalTablePower::Fourth,
        _ => return None,
    };
    Some(HyperbolicReciprocalTablePolicy {
        denominator_builtin,
        power,
    })
}

pub(crate) fn hyperbolic_reciprocal_derivative_policy(
    denominator_builtin: BuiltinFn,
) -> Option<HyperbolicReciprocalDerivativePolicy> {
    match denominator_builtin {
        BuiltinFn::Cosh => Some(HyperbolicReciprocalDerivativePolicy {
            denominator_builtin,
            numerator_builtin: BuiltinFn::Sinh,
        }),
        BuiltinFn::Sinh => Some(HyperbolicReciprocalDerivativePolicy {
            denominator_builtin,
            numerator_builtin: BuiltinFn::Cosh,
        }),
        _ => None,
    }
}

pub(crate) fn build_hyperbolic_denominator_nonzero_condition(
    ctx: &mut Context,
    denominator_builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    match denominator_builtin {
        BuiltinFn::Cosh | BuiltinFn::Sinh => Some(ctx.call_builtin(denominator_builtin, vec![arg])),
        _ => None,
    }
}

pub(crate) fn build_hyperbolic_reciprocal_derivative_integral(
    ctx: &mut Context,
    policy: HyperbolicReciprocalDerivativePolicy,
    arg: ExprId,
) -> ExprId {
    let den_arg = ctx.call_builtin(policy.denominator_builtin, vec![arg]);
    let one = ctx.num(1);
    let reciprocal = ctx.add(Expr::Div(one, den_arg));
    ctx.add(Expr::Neg(reciprocal))
}

pub(crate) fn build_hyperbolic_reciprocal_table_integral(
    ctx: &mut Context,
    policy: HyperbolicReciprocalTablePolicy,
    arg: ExprId,
    scale: ExprId,
    scale_ops: HyperbolicReciprocalPrimitiveScaleOps,
) -> ExprId {
    match policy.power {
        HyperbolicReciprocalTablePower::Square => {
            build_hyperbolic_reciprocal_square_integral(ctx, policy, arg, scale, scale_ops)
        }
        HyperbolicReciprocalTablePower::Fourth => {
            build_hyperbolic_reciprocal_fourth_integral(ctx, policy, arg, scale, scale_ops)
        }
    }
}

fn build_hyperbolic_reciprocal_square_integral(
    ctx: &mut Context,
    policy: HyperbolicReciprocalTablePolicy,
    arg: ExprId,
    scale: ExprId,
    scale_ops: HyperbolicReciprocalPrimitiveScaleOps,
) -> ExprId {
    match policy.denominator_builtin {
        BuiltinFn::Cosh => {
            let tanh_arg = ctx.call_builtin(BuiltinFn::Tanh, vec![arg]);
            scale_ops.scale_expr_reciprocal_integration_result(ctx, scale, tanh_arg)
        }
        BuiltinFn::Sinh => {
            let cosh_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
            let sinh_arg = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
            let quotient = ctx.add(Expr::Div(cosh_arg, sinh_arg));
            if let Expr::Number(value) = ctx.get(scale).clone() {
                if value.is_one() {
                    return ctx.add(Expr::Neg(quotient));
                }

                let scale_expr = ctx.add(Expr::Number(value));
                let negative_scale_expr = ctx.add(Expr::Neg(scale_expr));
                return mul2_raw(ctx, negative_scale_expr, quotient);
            }

            let negative_scale = scale_ops.negate_scalar_expr(ctx, scale);
            scale_ops.scale_expr_reciprocal_integration_result(ctx, negative_scale, quotient)
        }
        _ => unreachable!("hyperbolic reciprocal table policy only supports sinh/cosh"),
    }
}

fn build_hyperbolic_reciprocal_fourth_integral(
    ctx: &mut Context,
    policy: HyperbolicReciprocalTablePolicy,
    arg: ExprId,
    scale: ExprId,
    scale_ops: HyperbolicReciprocalPrimitiveScaleOps,
) -> ExprId {
    match policy.denominator_builtin {
        BuiltinFn::Cosh => {
            build_hyperbolic_cosh_reciprocal_fourth_integral(ctx, arg, scale, scale_ops)
        }
        BuiltinFn::Sinh => {
            build_hyperbolic_sinh_reciprocal_fourth_integral(ctx, arg, scale, scale_ops)
        }
        _ => unreachable!("hyperbolic reciprocal table policy only supports sinh/cosh"),
    }
}

fn build_hyperbolic_cosh_reciprocal_fourth_integral(
    ctx: &mut Context,
    arg: ExprId,
    scale: ExprId,
    scale_ops: HyperbolicReciprocalPrimitiveScaleOps,
) -> ExprId {
    let tanh_arg = ctx.call_builtin(BuiltinFn::Tanh, vec![arg]);
    let three = ctx.num(3);
    let tanh_cubed = ctx.add(Expr::Pow(tanh_arg, three));
    if let Expr::Number(scale_value) = ctx.get(scale).clone() {
        let linear_term = scale_ops.scale_rational_term(ctx, scale_value.clone(), tanh_arg);
        let cubic_term = scale_ops.scale_rational_term(
            ctx,
            -scale_value / BigRational::from_integer(3.into()),
            tanh_cubed,
        );
        return ctx.add(Expr::Add(linear_term, cubic_term));
    }

    let linear_term = scale_ops.scale_expr_reciprocal_integration_result(ctx, scale, tanh_arg);
    let cubic_scale =
        scale_ops.scale_rational_term(ctx, BigRational::new(1.into(), 3.into()), scale);
    let cubic_term =
        scale_ops.scale_expr_reciprocal_integration_result(ctx, cubic_scale, tanh_cubed);
    ctx.add(Expr::Sub(linear_term, cubic_term))
}

fn build_hyperbolic_sinh_reciprocal_fourth_integral(
    ctx: &mut Context,
    arg: ExprId,
    scale: ExprId,
    scale_ops: HyperbolicReciprocalPrimitiveScaleOps,
) -> ExprId {
    let tanh_arg = ctx.call_builtin(BuiltinFn::Tanh, vec![arg]);
    let one = ctx.num(1);
    let coth_arg = ctx.add(Expr::Div(one, tanh_arg));
    let three = ctx.num(3);
    let tanh_cubed = ctx.add(Expr::Pow(tanh_arg, three));
    let one = ctx.num(1);
    let coth_cubed = ctx.add(Expr::Div(one, tanh_cubed));

    if let Expr::Number(scale_value) = ctx.get(scale).clone() {
        let linear_term = scale_ops.rational_over_expr(ctx, scale_value.clone(), tanh_arg);
        let cubic_term = scale_ops.rational_over_expr(
            ctx,
            -scale_value / BigRational::from_integer(3.into()),
            tanh_cubed,
        );
        return ctx.add(Expr::Add(linear_term, cubic_term));
    }

    let linear_term = scale_ops.scale_expr_reciprocal_integration_result(ctx, scale, coth_arg);
    let cubic_scale =
        scale_ops.scale_rational_term(ctx, BigRational::new(1.into(), 3.into()), scale);
    let cubic_term =
        scale_ops.scale_expr_reciprocal_integration_result(ctx, cubic_scale, coth_cubed);
    ctx.add(Expr::Sub(linear_term, cubic_term))
}

fn is_number(ctx: &Context, expr: ExprId, value: i64) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if *n == BigRational::from_integer(value.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    fn test_scale_rational_term(ctx: &mut Context, scale: BigRational, term: ExprId) -> ExprId {
        if scale.is_one() {
            return term;
        }
        if scale == BigRational::from_integer((-1).into()) {
            return ctx.add(Expr::Neg(term));
        }

        let scale = ctx.add(Expr::Number(scale));
        mul2_raw(ctx, scale, term)
    }

    fn test_rational_over_expr(
        ctx: &mut Context,
        numerator: BigRational,
        denominator: ExprId,
    ) -> ExprId {
        let numerator = ctx.add(Expr::Number(numerator));
        ctx.add(Expr::Div(numerator, denominator))
    }

    fn test_scale_expr_reciprocal_integration_result(
        ctx: &mut Context,
        scale: ExprId,
        expr: ExprId,
    ) -> ExprId {
        if is_number(ctx, scale, 1) {
            expr
        } else {
            mul2_raw(ctx, scale, expr)
        }
    }

    fn test_negate_scalar_expr(ctx: &mut Context, expr: ExprId) -> ExprId {
        match ctx.get(expr).clone() {
            Expr::Number(value) => ctx.add(Expr::Number(-value)),
            Expr::Neg(inner) => inner,
            _ => ctx.add(Expr::Neg(expr)),
        }
    }

    fn test_scale_ops() -> HyperbolicReciprocalPrimitiveScaleOps {
        HyperbolicReciprocalPrimitiveScaleOps::new(
            test_scale_rational_term,
            test_rational_over_expr,
            test_scale_expr_reciprocal_integration_result,
            test_negate_scalar_expr,
        )
    }

    #[test]
    fn detects_hyperbolic_reciprocal_power_arguments() {
        let mut ctx = Context::new();
        let cosh_square = parse("cosh(2*x+1)^2", &mut ctx).unwrap();
        let sinh_fourth = parse("sinh(2*x+1)^4", &mut ctx).unwrap();
        let tanh_square = parse("tanh(2*x+1)^2", &mut ctx).unwrap();

        let cosh_arg =
            reciprocal_hyperbolic_cosh_square_arg(&ctx, cosh_square).expect("cosh square");
        let sinh_arg = reciprocal_hyperbolic_power_arg(&ctx, sinh_fourth, BuiltinFn::Sinh, 4)
            .expect("sinh fourth");

        assert_eq!(rendered(&ctx, cosh_arg), "2 * x + 1");
        assert_eq!(rendered(&ctx, sinh_arg), "2 * x + 1");
        let (square_builtin, square_arg) =
            reciprocal_hyperbolic_square_parts(&ctx, cosh_square).expect("cosh square parts");
        assert_eq!(square_builtin, BuiltinFn::Cosh);
        assert_eq!(rendered(&ctx, square_arg), "2 * x + 1");
        assert!(reciprocal_hyperbolic_cosh_square_arg(&ctx, tanh_square).is_none());
        assert!(reciprocal_hyperbolic_square_parts(&ctx, tanh_square).is_none());
    }

    #[test]
    fn maps_table_policy_for_supported_powers_only() {
        let cosh_square = hyperbolic_reciprocal_table_policy(BuiltinFn::Cosh, 2).unwrap();
        let sinh_fourth = hyperbolic_reciprocal_table_policy(BuiltinFn::Sinh, 4).unwrap();

        assert_eq!(cosh_square.denominator_builtin, BuiltinFn::Cosh);
        assert_eq!(cosh_square.power, HyperbolicReciprocalTablePower::Square);
        assert_eq!(cosh_square.power.exponent(), 2);
        assert_eq!(sinh_fourth.denominator_builtin, BuiltinFn::Sinh);
        assert_eq!(sinh_fourth.power, HyperbolicReciprocalTablePower::Fourth);
        assert_eq!(sinh_fourth.power.exponent(), 4);
        assert!(hyperbolic_reciprocal_table_policy(BuiltinFn::Tanh, 2).is_none());
        assert!(hyperbolic_reciprocal_table_policy(BuiltinFn::Cosh, 3).is_none());
    }

    #[test]
    fn builds_table_primitives_inside_hyperbolic_policy() {
        let mut ctx = Context::new();
        let arg = parse("x", &mut ctx).unwrap();
        let scale = ctx.num(1);

        let cosh_square = build_hyperbolic_reciprocal_table_integral(
            &mut ctx,
            hyperbolic_reciprocal_table_policy(BuiltinFn::Cosh, 2).unwrap(),
            arg,
            scale,
            test_scale_ops(),
        );
        let sinh_square = build_hyperbolic_reciprocal_table_integral(
            &mut ctx,
            hyperbolic_reciprocal_table_policy(BuiltinFn::Sinh, 2).unwrap(),
            arg,
            scale,
            test_scale_ops(),
        );
        let cosh_fourth = build_hyperbolic_reciprocal_table_integral(
            &mut ctx,
            hyperbolic_reciprocal_table_policy(BuiltinFn::Cosh, 4).unwrap(),
            arg,
            scale,
            test_scale_ops(),
        );
        let sinh_fourth = build_hyperbolic_reciprocal_table_integral(
            &mut ctx,
            hyperbolic_reciprocal_table_policy(BuiltinFn::Sinh, 4).unwrap(),
            arg,
            scale,
            test_scale_ops(),
        );

        assert_eq!(rendered(&ctx, cosh_square), "tanh(x)");
        assert_eq!(rendered(&ctx, sinh_square), "-cosh(x) / sinh(x)");
        assert_eq!(rendered(&ctx, cosh_fourth), "tanh(x) - 1/3 * tanh(x)^3");
        assert_eq!(rendered(&ctx, sinh_fourth), "1 / tanh(x) - 1/3 / tanh(x)^3");
    }

    #[test]
    fn builds_symbolic_scaled_table_primitives_inside_hyperbolic_policy() {
        let mut ctx = Context::new();
        let arg = parse("x", &mut ctx).unwrap();
        let scale = parse("a", &mut ctx).unwrap();

        let cosh_square = build_hyperbolic_reciprocal_table_integral(
            &mut ctx,
            hyperbolic_reciprocal_table_policy(BuiltinFn::Cosh, 2).unwrap(),
            arg,
            scale,
            test_scale_ops(),
        );
        let sinh_square = build_hyperbolic_reciprocal_table_integral(
            &mut ctx,
            hyperbolic_reciprocal_table_policy(BuiltinFn::Sinh, 2).unwrap(),
            arg,
            scale,
            test_scale_ops(),
        );
        let cosh_fourth = build_hyperbolic_reciprocal_table_integral(
            &mut ctx,
            hyperbolic_reciprocal_table_policy(BuiltinFn::Cosh, 4).unwrap(),
            arg,
            scale,
            test_scale_ops(),
        );
        let sinh_fourth = build_hyperbolic_reciprocal_table_integral(
            &mut ctx,
            hyperbolic_reciprocal_table_policy(BuiltinFn::Sinh, 4).unwrap(),
            arg,
            scale,
            test_scale_ops(),
        );

        assert_eq!(rendered(&ctx, cosh_square), "a * tanh(x)");
        assert_eq!(rendered(&ctx, sinh_square), "-cosh(x) * a/sinh(x)");
        assert_eq!(
            rendered(&ctx, cosh_fourth),
            "a * tanh(x) - 1/3 * a * tanh(x)^3"
        );
        assert_eq!(
            rendered(&ctx, sinh_fourth),
            "(1 * a)/tanh(x) - (1 * a * 1/3)/tanh(x)^3"
        );
    }

    #[test]
    fn maps_derivative_policy_and_base_integral() {
        let mut ctx = Context::new();
        let arg = parse("2*x+1", &mut ctx).unwrap();
        let policy = hyperbolic_reciprocal_derivative_policy(BuiltinFn::Cosh).unwrap();

        let integral = build_hyperbolic_reciprocal_derivative_integral(&mut ctx, policy, arg);

        assert_eq!(policy.numerator_builtin, BuiltinFn::Sinh);
        assert_eq!(rendered(&ctx, integral), "-1 / cosh(2 * x + 1)");
        assert!(hyperbolic_reciprocal_derivative_policy(BuiltinFn::Tanh).is_none());
    }

    #[test]
    fn builds_hyperbolic_denominator_conditions_without_trig_leakage() {
        let mut ctx = Context::new();
        let arg = parse("sqrt(x)", &mut ctx).unwrap();

        let cosh_condition =
            build_hyperbolic_denominator_nonzero_condition(&mut ctx, BuiltinFn::Cosh, arg)
                .expect("cosh condition");
        let sinh_condition =
            build_hyperbolic_denominator_nonzero_condition(&mut ctx, BuiltinFn::Sinh, arg)
                .expect("sinh condition");

        assert_eq!(rendered(&ctx, cosh_condition), "cosh(sqrt(x))");
        assert_eq!(rendered(&ctx, sinh_condition), "sinh(sqrt(x))");
        assert!(
            build_hyperbolic_denominator_nonzero_condition(&mut ctx, BuiltinFn::Cos, arg).is_none()
        );
        assert!(
            build_hyperbolic_denominator_nonzero_condition(&mut ctx, BuiltinFn::Tanh, arg)
                .is_none()
        );
    }
}
