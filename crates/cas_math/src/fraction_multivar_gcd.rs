//! Multivariate GCD helpers specialized for fraction cancellation.

use crate::multipoly::{
    gcd_multivar_layer2, gcd_multivar_layer25, multipoly_from_expr, multipoly_to_expr, GcdBudget,
    GcdLayer, Layer25Budget, MultiPoly, PolyBudget,
};
use cas_ast::{Context, ExprId};
use num_rational::BigRational;
use num_traits::One;

fn gcd_rational_integers_only(a: BigRational, b: BigRational) -> BigRational {
    if a.is_integer() && b.is_integer() {
        use num_integer::Integer;
        let num_a = a.to_integer();
        let num_b = b.to_integer();
        return BigRational::from_integer(num_a.gcd(&num_b));
    }
    BigRational::one()
}

/// Try to compute a non-trivial multivariate GCD for `num/den`.
///
/// Returns:
/// - `None` when conversion fails, inputs are effectively univariate, or GCD is trivial.
/// - `Some((new_num, new_den, gcd_expr, layer))` when a factor is extracted.
pub fn try_multivar_gcd(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
) -> Option<(ExprId, ExprId, ExprId, GcdLayer)> {
    let budget = PolyBudget::default();

    let p_num = multipoly_from_expr(ctx, num, &budget).ok()?;
    let p_den = multipoly_from_expr(ctx, den, &budget).ok()?;

    // Keep current behavior: only run this path for multivariate numerators.
    if p_num.vars.len() <= 1 {
        return None;
    }

    let (p_num, p_den) = if p_num.vars != p_den.vars {
        let mut all_vars: Vec<String> = p_num
            .vars
            .iter()
            .chain(p_den.vars.iter())
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        all_vars.sort();
        (p_num.align_vars(&all_vars), p_den.align_vars(&all_vars))
    } else {
        (p_num, p_den)
    };

    let mono_gcd = p_num.monomial_gcd_with(&p_den).ok()?;
    let has_mono_gcd = mono_gcd.iter().any(|&e| e > 0);

    let content_num = p_num.content();
    let content_den = p_den.content();
    let content_gcd = gcd_rational_integers_only(content_num.clone(), content_den.clone());
    let has_content_gcd = !content_gcd.is_one();

    if !has_mono_gcd && !has_content_gcd {
        if p_num.vars.len() >= 2 {
            let gcd_budget = GcdBudget::default();
            if let Some(gcd_poly) = gcd_multivar_layer2(&p_num, &p_den, &gcd_budget) {
                if !gcd_poly.is_one() && !gcd_poly.is_constant() {
                    let q_num = p_num.div_exact(&gcd_poly)?;
                    let q_den = p_den.div_exact(&gcd_poly)?;
                    let new_num = multipoly_to_expr(&q_num, ctx);
                    let new_den = multipoly_to_expr(&q_den, ctx);
                    let gcd_expr = multipoly_to_expr(&gcd_poly, ctx);
                    return Some((new_num, new_den, gcd_expr, GcdLayer::Layer2HeuristicSeeds));
                }
            }

            let layer25_budget = Layer25Budget::default();
            if let Some(gcd_poly) = gcd_multivar_layer25(&p_num, &p_den, &layer25_budget) {
                if !gcd_poly.is_one() && !gcd_poly.is_constant() {
                    let q_num = p_num.div_exact(&gcd_poly)?;
                    let q_den = p_den.div_exact(&gcd_poly)?;
                    let new_num = multipoly_to_expr(&q_num, ctx);
                    let new_den = multipoly_to_expr(&q_den, ctx);
                    let gcd_expr = multipoly_to_expr(&gcd_poly, ctx);
                    return Some((new_num, new_den, gcd_expr, GcdLayer::Layer25TensorGrid));
                }
            }
        }
        return None;
    }

    let (p_num, p_den) = if has_mono_gcd {
        (
            p_num.div_monomial_exact(&mono_gcd)?,
            p_den.div_monomial_exact(&mono_gcd)?,
        )
    } else {
        (p_num, p_den)
    };

    let (p_num, p_den) = if has_content_gcd {
        (
            p_num.div_scalar_exact(&content_gcd)?,
            p_den.div_scalar_exact(&content_gcd)?,
        )
    } else {
        (p_num, p_den)
    };

    let mut gcd_poly = MultiPoly::one(p_num.vars.clone());
    if has_mono_gcd {
        gcd_poly = gcd_poly.mul_monomial(&mono_gcd).ok()?;
    }
    if has_content_gcd {
        gcd_poly = gcd_poly.mul_scalar(&content_gcd);
    }

    let new_num = multipoly_to_expr(&p_num, ctx);
    let new_den = multipoly_to_expr(&p_den, ctx);
    let gcd_expr = multipoly_to_expr(&gcd_poly, ctx);

    Some((new_num, new_den, gcd_expr, GcdLayer::Layer1MonomialContent))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly_compare::poly_eq;
    use cas_parser::parse;

    #[test]
    fn multivar_layer1_extracts_common_factor() {
        let mut ctx = Context::new();
        let num = parse("x*y + x*z", &mut ctx).expect("parse num");
        let den = parse("x", &mut ctx).expect("parse den");

        let (new_num, new_den, gcd, layer) =
            try_multivar_gcd(&mut ctx, num, den).expect("should extract");

        assert_eq!(layer, GcdLayer::Layer1MonomialContent);
        let expected_num = parse("y+z", &mut ctx).expect("parse expected_num");
        let expected_den = parse("1", &mut ctx).expect("parse expected_den");
        let expected_gcd = parse("x", &mut ctx).expect("parse expected_gcd");
        assert!(poly_eq(&ctx, new_num, expected_num));
        assert!(poly_eq(&ctx, new_den, expected_den));
        assert!(poly_eq(&ctx, gcd, expected_gcd));
    }

    #[test]
    fn univariate_returns_none() {
        let mut ctx = Context::new();
        let num = parse("x^2-1", &mut ctx).expect("parse num");
        let den = parse("x-1", &mut ctx).expect("parse den");
        assert!(try_multivar_gcd(&mut ctx, num, den).is_none());
    }
}
