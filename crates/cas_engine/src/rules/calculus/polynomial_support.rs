use cas_ast::{Context, Expr, ExprId};
use cas_math::multipoly::{multipoly_from_expr, multipoly_to_expr, MultiPoly, PolyBudget};
use cas_math::polynomial::Polynomial;
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

pub(super) fn nonzero_affine_variable_derivative(
    ctx: &Context,
    target: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let budget = PolyBudget {
        max_terms: 8,
        max_total_degree: 1,
        max_pow_exp: 1,
    };
    let poly = multipoly_from_expr(ctx, target, &budget).ok()?;
    if poly.vars.len() != 1 || poly.vars[0] != var_name || poly.total_degree() > 1 {
        return None;
    }

    let mut linear_coeff = BigRational::zero();
    for (coeff, mono) in &poly.terms {
        match mono.as_slice() {
            [0] => {}
            [1] => linear_coeff += coeff.clone(),
            _ => return None,
        }
    }

    (!linear_coeff.is_zero()).then_some(linear_coeff)
}

pub(super) fn polynomial_radicand_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<Polynomial> {
    let poly = Polynomial::from_expr(ctx, expr, var_name).ok()?;
    if poly.degree() > 4 || poly.coeffs.len() > 8 {
        return None;
    }
    Some(poly)
}

pub(super) fn polynomial_is_strictly_positive_everywhere(poly: &Polynomial) -> bool {
    if polynomial_has_positive_constant_and_nonnegative_even_terms(poly) {
        return true;
    }
    match poly.degree() {
        0 => poly
            .coeffs
            .first()
            .is_some_and(|constant| constant.is_positive()),
        2 => strictly_positive_quadratic_on_reals(poly),
        _ => false,
    }
}

fn polynomial_has_positive_constant_and_nonnegative_even_terms(poly: &Polynomial) -> bool {
    if !poly
        .coeffs
        .first()
        .is_some_and(|constant| constant.is_positive())
    {
        return false;
    }
    poly.coeffs
        .iter()
        .enumerate()
        .skip(1)
        .all(|(power, coeff)| coeff.is_zero() || (power % 2 == 0 && !coeff.is_negative()))
}

pub(super) fn strictly_positive_quadratic_on_reals(poly: &Polynomial) -> bool {
    if poly.degree() != 2 {
        return false;
    }

    let a = poly
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if !a.is_positive() {
        return false;
    }

    let b = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let c = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let four = BigRational::from_integer(4.into());
    let discriminant = &b * &b - four * a * c;
    discriminant.is_negative()
}

pub(super) fn square_of_strictly_positive_quadratic_arg(
    ctx: &Context,
    arg: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(arg) else {
        return None;
    };
    let two = BigRational::from_integer(2.into());
    if cas_ast::views::as_rational_const(ctx, *exp, 8).as_ref() != Some(&two) {
        return None;
    }

    let base_poly = polynomial_radicand_for_calculus_presentation(ctx, *base, var_name)?;
    strictly_positive_quadratic_on_reals(&base_poly).then_some(*base)
}

pub(super) fn polynomial_derivative_expr_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let poly = polynomial_radicand_for_calculus_presentation(ctx, expr, var_name)?;
    Some(poly.derivative().to_expr(ctx))
}

pub(super) fn expanded_polynomial_expr_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    max_total_degree: u32,
) -> ExprId {
    let budget = PolyBudget {
        max_terms: 16,
        max_total_degree,
        max_pow_exp: max_total_degree,
    };

    multipoly_from_expr(ctx, expr, &budget)
        .map(|poly| multipoly_to_expr(&poly, ctx))
        .unwrap_or(expr)
}

pub(super) fn rational_polynomial_content_for_calculus_presentation(
    poly: &Polynomial,
) -> BigRational {
    let mut numer_gcd: Option<BigInt> = None;
    let mut denom_lcm = BigInt::one();

    for coeff in &poly.coeffs {
        if coeff.is_zero() {
            continue;
        }
        let numer = coeff.numer().abs();
        let denom = coeff.denom().clone();
        numer_gcd = Some(match numer_gcd {
            Some(gcd) => gcd.gcd(&numer),
            None => numer,
        });
        denom_lcm = denom_lcm.lcm(&denom);
    }

    match numer_gcd {
        Some(numer_gcd) if !numer_gcd.is_zero() => BigRational::new(numer_gcd, denom_lcm),
        _ => BigRational::zero(),
    }
}

pub(super) fn scale_polynomial_for_calculus_presentation(
    poly: &Polynomial,
    coeff: &BigRational,
) -> Polynomial {
    Polynomial::new(
        poly.coeffs.iter().map(|term| term * coeff).collect(),
        poly.var.clone(),
    )
}

pub(super) fn split_polynomial_content_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> (ExprId, BigRational) {
    let budget = PolyBudget {
        max_terms: 8,
        max_total_degree: 4,
        max_pow_exp: 4,
    };

    let Ok(poly) = multipoly_from_expr(ctx, expr, &budget) else {
        return (expr, BigRational::one());
    };
    let (content, primitive) = poly.primitive_part();
    if content.is_zero() || content.is_one() {
        return (expr, BigRational::one());
    }

    (multipoly_to_expr(&primitive, ctx), content)
}

pub(super) fn multipoly_denominator_lcm_for_calculus_presentation(poly: &MultiPoly) -> BigInt {
    poly.terms
        .iter()
        .fold(BigInt::one(), |acc, (coeff, _)| acc.lcm(coeff.denom()))
}

pub(super) fn multipoly_has_integer_coefficients_for_calculus_presentation(
    poly: &MultiPoly,
) -> bool {
    poly.terms.iter().all(|(coeff, _)| coeff.denom().is_one())
}

pub(super) fn polynomial_positive_content_for_calculus_presentation(
    poly: &Polynomial,
) -> Option<BigRational> {
    let mut denominator_lcm = BigInt::one();
    let mut integer_gcd: Option<BigInt> = None;

    for coeff in poly.coeffs.iter().filter(|coeff| !coeff.is_zero()) {
        denominator_lcm = denominator_lcm.lcm(coeff.denom());
    }
    if denominator_lcm.is_zero() {
        return None;
    }

    for coeff in poly.coeffs.iter().filter(|coeff| !coeff.is_zero()) {
        let scaled = coeff * BigRational::from_integer(denominator_lcm.clone());
        let value = scaled.to_integer().abs();
        integer_gcd = Some(match integer_gcd {
            Some(existing) => existing.gcd(&value),
            None => value,
        });
    }

    let integer_gcd = integer_gcd?;
    if integer_gcd.is_zero() {
        return None;
    }
    Some(BigRational::new(integer_gcd, denominator_lcm))
}
