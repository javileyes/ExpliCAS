//! Log-result construction helpers for symbolic integration.

use crate::build::mul2_raw;
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

pub(crate) fn ln_abs(ctx: &mut Context, arg: ExprId) -> ExprId {
    let abs_arg = ctx.call_builtin(BuiltinFn::Abs, vec![arg]);
    ctx.call_builtin(BuiltinFn::Ln, vec![abs_arg])
}

pub(crate) fn scaled_ln_abs_product_form(
    ctx: &mut Context,
    arg: ExprId,
    scale: BigRational,
) -> ExprId {
    let log_abs = ln_abs(ctx, arg);
    if scale.is_one() {
        return log_abs;
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    mul2_raw(ctx, scale_expr, log_abs)
}

pub(crate) fn scaled_ln_abs_with_negative_shortcut(
    ctx: &mut Context,
    arg: ExprId,
    scale: BigRational,
) -> ExprId {
    let log_abs = ln_abs(ctx, arg);
    if scale.is_one() {
        return log_abs;
    }
    if scale == -BigRational::one() {
        return ctx.add(Expr::Neg(log_abs));
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    mul2_raw(ctx, scale_expr, log_abs)
}

pub(crate) fn constant_base_log_derivative_correction(
    ctx: &mut Context,
    base_ln: Option<ExprId>,
) -> ExprId {
    match base_ln {
        Some(base_ln) => {
            let one = ctx.num(1);
            ctx.add(Expr::Div(one, base_ln))
        }
        None => ctx.num(1),
    }
}

pub(crate) fn positive_integer_constant_log_base_ln(ctx: &mut Context, base_value: i64) -> ExprId {
    debug_assert!(base_value > 1);
    let base = ctx.num(base_value);
    ctx.call_builtin(BuiltinFn::Ln, vec![base])
}

pub(crate) fn positive_integer_constant_log_base_derivative_correction(
    ctx: &mut Context,
    base_value: i64,
) -> ExprId {
    let base_ln = positive_integer_constant_log_base_ln(ctx, base_value);
    constant_base_log_derivative_correction(ctx, Some(base_ln))
}

pub(crate) fn affine_constant_base_log_antiderivative_from_slope(
    ctx: &mut Context,
    log_expr: ExprId,
    arg: ExprId,
    base_ln: Option<ExprId>,
    slope: ExprId,
    slope_value: Option<BigRational>,
) -> Option<ExprId> {
    if slope_value.is_some_and(|value| value.is_zero()) {
        return None;
    }

    let reciprocal_base_ln = constant_base_log_derivative_correction(ctx, base_ln);
    let log_minus_reciprocal_base_ln = ctx.add(Expr::Sub(log_expr, reciprocal_base_ln));
    let integral = crate::build::mul2_raw(ctx, arg, log_minus_reciprocal_base_ln);

    if matches!(ctx.get(slope), Expr::Number(n) if n.is_one()) {
        Some(integral)
    } else {
        Some(ctx.add(Expr::Div(integral, slope)))
    }
}

pub(crate) fn valid_constant_log_base_ln_from_rational_value(
    ctx: &mut Context,
    base: ExprId,
    rational_value: Option<BigRational>,
) -> Option<Option<ExprId>> {
    if matches!(ctx.get(base), Expr::Constant(Constant::E)) {
        return Some(None);
    }

    let value = rational_value?;
    if !value.is_positive() || value.is_one() {
        return None;
    }

    Some(Some(ctx.call_builtin(BuiltinFn::Ln, vec![base])))
}

#[cfg(test)]
mod tests {
    use super::{
        affine_constant_base_log_antiderivative_from_slope,
        constant_base_log_derivative_correction,
        positive_integer_constant_log_base_derivative_correction,
        positive_integer_constant_log_base_ln, scaled_ln_abs_product_form,
        scaled_ln_abs_with_negative_shortcut, valid_constant_log_base_ln_from_rational_value,
    };
    use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
    use cas_formatter::DisplayExpr;
    use num_rational::BigRational;

    fn rational(value: i64) -> BigRational {
        BigRational::new(value.into(), 1.into())
    }

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn builds_positive_integer_constant_log_base_ln() {
        let mut ctx = Context::new();

        let base_two_ln = positive_integer_constant_log_base_ln(&mut ctx, 2);
        assert_eq!(rendered(&ctx, base_two_ln), "ln(2)");

        let base_ten_ln = positive_integer_constant_log_base_ln(&mut ctx, 10);
        assert_eq!(rendered(&ctx, base_ten_ln), "ln(10)");
    }

    #[test]
    fn builds_positive_integer_constant_log_base_derivative_correction() {
        let mut ctx = Context::new();

        let base_two_correction =
            positive_integer_constant_log_base_derivative_correction(&mut ctx, 2);
        assert_eq!(rendered(&ctx, base_two_correction), "1 / ln(2)");

        let base_ten_correction =
            positive_integer_constant_log_base_derivative_correction(&mut ctx, 10);
        assert_eq!(rendered(&ctx, base_ten_correction), "1 / ln(10)");
    }

    #[test]
    fn builds_scaled_ln_abs_product_form() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        let unit = scaled_ln_abs_product_form(&mut ctx, x, rational(1));
        assert_eq!(rendered(&ctx, unit), "ln(|x|)");

        let scaled = scaled_ln_abs_product_form(&mut ctx, x, rational(2));
        assert_eq!(rendered(&ctx, scaled), "2 * ln(|x|)");
    }

    #[test]
    fn builds_scaled_ln_abs_with_negative_shortcut() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        let negative = scaled_ln_abs_with_negative_shortcut(&mut ctx, x, rational(-1));
        assert_eq!(rendered(&ctx, negative), "-ln(|x|)");

        let scaled = scaled_ln_abs_with_negative_shortcut(&mut ctx, x, rational(-2));
        assert_eq!(rendered(&ctx, scaled), "-2 * ln(|x|)");
    }

    #[test]
    fn builds_affine_constant_base_log_antiderivative_for_unit_slope() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let log_expr = ctx.call_builtin(BuiltinFn::Ln, vec![x]);
        let slope = ctx.num(1);

        let integral = affine_constant_base_log_antiderivative_from_slope(
            &mut ctx,
            log_expr,
            x,
            None,
            slope,
            Some(rational(1)),
        )
        .expect("unit slope log primitive");

        assert_eq!(rendered(&ctx, integral), "x * (ln(x) - 1)");
    }

    #[test]
    fn builds_affine_constant_base_log_antiderivative_for_scaled_slope() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let arg = ctx.add(Expr::Mul(two, x));
        let log_expr = ctx.call_builtin(BuiltinFn::Log, vec![two, arg]);
        let base_ln = ctx.call_builtin(BuiltinFn::Ln, vec![two]);

        let integral = affine_constant_base_log_antiderivative_from_slope(
            &mut ctx,
            log_expr,
            arg,
            Some(base_ln),
            two,
            Some(rational(2)),
        )
        .expect("scaled slope log primitive");

        assert_eq!(
            rendered(&ctx, integral),
            "2 * x * (log(2, 2 * x) - 1 / ln(2)) / 2"
        );
    }

    #[test]
    fn rejects_zero_slope_affine_log_antiderivative() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let log_expr = ctx.call_builtin(BuiltinFn::Ln, vec![x]);
        let zero = ctx.num(0);

        assert!(affine_constant_base_log_antiderivative_from_slope(
            &mut ctx,
            log_expr,
            x,
            None,
            zero,
            Some(rational(0)),
        )
        .is_none());
    }

    #[test]
    fn accepts_e_base_as_natural_log_without_extra_ln_factor() {
        let mut ctx = Context::new();
        let e = ctx.add(Expr::Constant(Constant::E));

        let base_ln =
            valid_constant_log_base_ln_from_rational_value(&mut ctx, e, None).expect("valid e");

        assert!(base_ln.is_none());
        let correction = constant_base_log_derivative_correction(&mut ctx, base_ln);
        assert_eq!(rendered(&ctx, correction), "1");
    }

    #[test]
    fn accepts_positive_rational_base_other_than_one() {
        let mut ctx = Context::new();
        let two = ctx.num(2);

        let base_ln =
            valid_constant_log_base_ln_from_rational_value(&mut ctx, two, Some(rational(2)))
                .expect("valid positive base")
                .expect("non-e base should carry ln(base)");

        assert_eq!(rendered(&ctx, base_ln), "ln(2)");
        let correction = constant_base_log_derivative_correction(&mut ctx, Some(base_ln));
        assert_eq!(rendered(&ctx, correction), "1 / ln(2)");
    }

    #[test]
    fn rejects_invalid_rational_log_bases() {
        for value in [1, 0, -2] {
            let mut ctx = Context::new();
            let base = ctx.num(value);

            assert!(
                valid_constant_log_base_ln_from_rational_value(
                    &mut ctx,
                    base,
                    Some(rational(value))
                )
                .is_none(),
                "base {value} should be invalid over the reals"
            );
        }
    }

    #[test]
    fn rejects_non_e_base_without_rational_evidence() {
        let mut ctx = Context::new();
        let symbolic_base = ctx.var("a");

        assert!(
            valid_constant_log_base_ln_from_rational_value(&mut ctx, symbolic_base, None).is_none()
        );
    }
}
