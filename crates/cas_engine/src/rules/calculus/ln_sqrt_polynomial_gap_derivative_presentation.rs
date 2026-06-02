//! Derivative presentation for `ln(sqrt(polynomial) + polynomial)` positive gaps.
//!
//! This module owns the asinh-style log-root route where the radicand differs
//! from the polynomial square by a positive constant gap.

use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::scalar_presentation::scale_expr_for_calculus_presentation;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use cas_math::root_forms::extract_square_root_base;
use num_traits::Signed;

pub(super) fn ln_sqrt_polynomial_gap_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let terms = cas_math::expr_nary::add_terms_signed(ctx, args[0]);
    let mut radicand = None;
    let mut polynomial_term_poly = Polynomial::zero(var_name.to_string());
    for (term, sign) in terms {
        if let Some(term_radicand) = extract_square_root_base(ctx, term) {
            if sign == cas_math::expr_nary::Sign::Neg {
                return None;
            }
            if radicand.is_some() {
                return None;
            }
            radicand = Some(term_radicand);
        } else {
            let mut term_poly = polynomial_radicand_for_calculus_presentation(ctx, term, var_name)?;
            if sign == cas_math::expr_nary::Sign::Neg {
                term_poly = term_poly.neg();
            }
            polynomial_term_poly = polynomial_term_poly.add(&term_poly);
        }
    }

    let radicand = radicand?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let positive_gap = radicand_poly.sub(&polynomial_term_poly.mul(&polynomial_term_poly));
    if positive_gap.degree() != 0
        || positive_gap
            .coeffs
            .first()
            .is_none_or(|value| !value.is_positive())
    {
        return None;
    }

    let derivative_poly = polynomial_term_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let numerator = scale_expr_for_calculus_presentation(ctx, derivative_content, derivative_core);
    let denominator = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::ln_sqrt_polynomial_gap_derivative_presentation;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn ln_sqrt_polynomial_gap_keeps_asinh_style_derivative() {
        let mut ctx = Context::new();
        let target = parse("ln(sqrt(x^2+1)+x)", &mut ctx).unwrap();
        let derivative =
            ln_sqrt_polynomial_gap_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "1 / sqrt(x^2 + 1)");

        let target = parse("ln(sqrt(x^4+1)-x^2)", &mut ctx).unwrap();
        let derivative =
            ln_sqrt_polynomial_gap_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "-2 * x / sqrt(x^4 + 1)");
    }
}
