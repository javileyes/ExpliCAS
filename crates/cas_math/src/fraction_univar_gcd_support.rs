//! Support for univariate GCD-based fraction reduction.
//!
//! This module keeps polynomial reduction mechanics in `cas_math` and leaves
//! domain-policy decisions to higher layers.

use crate::build::mul2_raw;
use crate::canonical_forms::normalize_core;
use crate::numeric::gcd_rational;
use crate::polynomial::Polynomial;
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Zero};

#[derive(Debug, Clone)]
pub struct UnivariateFractionGcdReduction {
    pub new_num: ExprId,
    pub new_den: ExprId,
    pub gcd_expr: ExprId,
    pub numeric_gcd: BigRational,
    pub partial_num: Option<ExprId>,
    pub partial_den: Option<ExprId>,
}

#[derive(Debug, Clone, Copy)]
pub struct FractionCancelForms {
    pub result: ExprId,
    pub result_norm: ExprId,
    pub factored_form_norm: ExprId,
    pub numerator_is_zero: bool,
}

/// Build reduced result expression for `(new_num/new_den)`, collapsing `/1`.
pub fn build_reduced_fraction_result(
    ctx: &mut Context,
    new_num: ExprId,
    new_den: ExprId,
) -> ExprId {
    if let Expr::Number(n) = ctx.get(new_den) {
        if n.is_one() {
            return new_num;
        }
    }
    ctx.add(Expr::Div(new_num, new_den))
}

/// Build factored display/cancellation forms and normalized variants.
pub fn build_fraction_cancel_forms(
    ctx: &mut Context,
    new_num: ExprId,
    new_den: ExprId,
    gcd_expr: ExprId,
) -> FractionCancelForms {
    let numerator_is_zero = matches!(ctx.get(new_num), Expr::Number(n) if n.is_zero());
    let result = build_reduced_fraction_result(ctx, new_num, new_den);

    let factored_num = mul2_raw(ctx, new_num, gcd_expr);
    let factored_den = if let Expr::Number(n) = ctx.get(new_den) {
        if n.is_one() {
            gcd_expr
        } else {
            mul2_raw(ctx, new_den, gcd_expr)
        }
    } else {
        mul2_raw(ctx, new_den, gcd_expr)
    };
    let factored_form = ctx.add(Expr::Div(factored_num, factored_den));

    let factored_form_norm = normalize_core(ctx, factored_form);
    let result_norm = normalize_core(ctx, result);

    FractionCancelForms {
        result,
        result_norm,
        factored_form_norm,
        numerator_is_zero,
    }
}

/// Try univariate GCD reduction for a fraction `num/den` in variable `var`.
///
/// Returns `None` for unsupported inputs, trivial GCD, or inexact division.
pub fn try_univariate_fraction_gcd_reduction(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<UnivariateFractionGcdReduction> {
    let p_num = Polynomial::from_expr(ctx, num, var).ok()?;
    let p_den = Polynomial::from_expr(ctx, den, var).ok()?;

    if p_den.is_zero() {
        return None;
    }

    let poly_gcd = p_num.gcd(&p_den);

    let content_num = p_num.content();
    let content_den = p_den.content();
    let numeric_gcd = gcd_rational(content_num, content_den);

    let scalar = Polynomial::new(vec![numeric_gcd.clone()], var.to_string());
    let full_gcd = poly_gcd.mul(&scalar);

    if full_gcd.degree() == 0 && full_gcd.leading_coeff().is_one() {
        return None;
    }

    let (new_num_poly, rem_num) = p_num.div_rem(&full_gcd).ok()?;
    let (new_den_poly, rem_den) = p_den.div_rem(&full_gcd).ok()?;
    if !rem_num.is_zero() || !rem_den.is_zero() {
        return None;
    }

    let partial = if !numeric_gcd.is_one() && !numeric_gcd.is_zero() {
        let partial_num = p_num.div_scalar(&numeric_gcd).to_expr(ctx);
        let partial_den = p_den.div_scalar(&numeric_gcd).to_expr(ctx);
        Some((partial_num, partial_den))
    } else {
        None
    };

    let new_num = new_num_poly.to_expr(ctx);
    let new_den = new_den_poly.to_expr(ctx);
    let gcd_expr = full_gcd.to_expr(ctx);

    Some(UnivariateFractionGcdReduction {
        new_num,
        new_den,
        gcd_expr,
        numeric_gcd,
        partial_num: partial.map(|p| p.0),
        partial_den: partial.map(|p| p.1),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        build_fraction_cancel_forms, build_reduced_fraction_result,
        try_univariate_fraction_gcd_reduction,
    };
    use crate::poly_compare::poly_eq;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn reduce_fraction_result_collapses_denominator_one() {
        let mut ctx = Context::new();
        let num = parse("x+1", &mut ctx).expect("parse num");
        let den = parse("1", &mut ctx).expect("parse den");
        let result = build_reduced_fraction_result(&mut ctx, num, den);
        assert!(poly_eq(&ctx, result, num));
    }

    #[test]
    fn build_cancel_forms_marks_zero_numerator() {
        let mut ctx = Context::new();
        let new_num = parse("0", &mut ctx).expect("parse");
        let new_den = parse("x+1", &mut ctx).expect("parse");
        let gcd = parse("x-1", &mut ctx).expect("parse");
        let forms = build_fraction_cancel_forms(&mut ctx, new_num, new_den, gcd);
        assert!(forms.numerator_is_zero);
    }

    #[test]
    fn univariate_reduction_basic() {
        let mut ctx = Context::new();
        let num = parse("x^2-1", &mut ctx).expect("parse num");
        let den = parse("x-1", &mut ctx).expect("parse den");
        let reduced =
            try_univariate_fraction_gcd_reduction(&mut ctx, num, den, "x").expect("reduce");
        let expected_num = parse("x+1", &mut ctx).expect("expected num");
        let expected_den = parse("1", &mut ctx).expect("expected den");
        let expected_gcd = parse("x-1", &mut ctx).expect("expected gcd");
        assert!(poly_eq(&ctx, reduced.new_num, expected_num));
        assert!(poly_eq(&ctx, reduced.new_den, expected_den));
        assert!(poly_eq(&ctx, reduced.gcd_expr, expected_gcd));
    }

    #[test]
    fn univariate_reduction_trivial_gcd_returns_none() {
        let mut ctx = Context::new();
        let num = parse("x+1", &mut ctx).expect("parse num");
        let den = parse("x+2", &mut ctx).expect("parse den");
        assert!(try_univariate_fraction_gcd_reduction(&mut ctx, num, den, "x").is_none());
    }

    #[test]
    fn univariate_reduction_exposes_numeric_partial() {
        let mut ctx = Context::new();
        let num = parse("27*x^3", &mut ctx).expect("parse num");
        let den = parse("9*x", &mut ctx).expect("parse den");
        let reduced =
            try_univariate_fraction_gcd_reduction(&mut ctx, num, den, "x").expect("reduce");
        assert!(reduced.partial_num.is_some());
        assert!(reduced.partial_den.is_some());
    }
}
