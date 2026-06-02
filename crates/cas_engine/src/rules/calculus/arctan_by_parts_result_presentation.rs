use super::presentation_utils::unwrap_internal_hold_for_calculus;
use super::scalar_presentation::fold_numeric_mul_constants_for_hold;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::Zero;

struct ArctanAffineByPartsTerm {
    arg: ExprId,
    arg_poly: Polynomial,
    cofactor_poly: Polynomial,
}

struct LnAffineByPartsTerm {
    arg_poly: Polynomial,
    coefficient: BigRational,
}

fn apply_additive_sign_to_poly(poly: Polynomial, sign: cas_math::expr_nary::Sign) -> Polynomial {
    match sign {
        cas_math::expr_nary::Sign::Pos => poly,
        cas_math::expr_nary::Sign::Neg => poly.neg(),
    }
}

fn arctan_affine_by_parts_arctan_term(
    ctx: &Context,
    term: ExprId,
    sign: cas_math::expr_nary::Sign,
    var_name: &str,
) -> Option<ArctanAffineByPartsTerm> {
    let term = cas_ast::hold::unwrap_internal_hold(ctx, term);
    if let Expr::Div(num, den) = ctx.get(term).clone() {
        let denominator = cas_ast::views::as_rational_const(ctx, den, 8)?;
        if denominator.is_zero() {
            return None;
        }
        let mut term = arctan_affine_by_parts_arctan_term(ctx, num, sign, var_name)?;
        term.cofactor_poly = term.cofactor_poly.div_scalar(&denominator);
        return Some(term);
    }

    let factors = cas_math::expr_nary::MulView::from_expr(ctx, term).factors;
    let mut arctan_arg = None;
    let mut cofactor_poly = Polynomial::one(var_name.to_string());

    for factor in factors {
        let factor = cas_ast::hold::unwrap_internal_hold(ctx, factor);
        if let Expr::Function(fn_id, args) = ctx.get(factor) {
            if args.len() == 1
                && matches!(
                    ctx.builtin_of(*fn_id),
                    Some(BuiltinFn::Arctan | BuiltinFn::Atan)
                )
            {
                if arctan_arg.replace(args[0]).is_some() {
                    return None;
                }
                continue;
            }
        }

        let factor_poly = Polynomial::from_expr(ctx, factor, var_name).ok()?;
        cofactor_poly = cofactor_poly.mul(&factor_poly);
    }

    let arg = arctan_arg?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var_name).ok()?;
    if arg_poly.degree() != 1 {
        return None;
    }

    Some(ArctanAffineByPartsTerm {
        arg,
        arg_poly,
        cofactor_poly: apply_additive_sign_to_poly(cofactor_poly, sign),
    })
}

fn arctan_affine_by_parts_ln_term(
    ctx: &Context,
    term: ExprId,
    sign: cas_math::expr_nary::Sign,
    var_name: &str,
) -> Option<LnAffineByPartsTerm> {
    let term = cas_ast::hold::unwrap_internal_hold(ctx, term);
    if let Expr::Div(num, den) = ctx.get(term).clone() {
        let denominator = cas_ast::views::as_rational_const(ctx, den, 8)?;
        if denominator.is_zero() {
            return None;
        }
        let mut term = arctan_affine_by_parts_ln_term(ctx, num, sign, var_name)?;
        term.coefficient /= denominator;
        return Some(term);
    }

    let factors = cas_math::expr_nary::MulView::from_expr(ctx, term).factors;
    let mut ln_arg = None;
    let mut coefficient_poly = Polynomial::one(var_name.to_string());

    for factor in factors {
        let factor = cas_ast::hold::unwrap_internal_hold(ctx, factor);
        if let Expr::Function(fn_id, args) = ctx.get(factor) {
            if ctx.builtin_of(*fn_id) == Some(BuiltinFn::Ln) && args.len() == 1 {
                if ln_arg.replace(args[0]).is_some() {
                    return None;
                }
                continue;
            }
        }

        let factor_poly = Polynomial::from_expr(ctx, factor, var_name).ok()?;
        coefficient_poly = coefficient_poly.mul(&factor_poly);
    }

    let ln_arg = ln_arg?;
    let coefficient_poly = apply_additive_sign_to_poly(coefficient_poly, sign);
    if coefficient_poly.degree() != 0 {
        return None;
    }

    Some(LnAffineByPartsTerm {
        arg_poly: Polynomial::from_expr(ctx, ln_arg, var_name).ok()?,
        coefficient: coefficient_poly
            .coeffs
            .first()
            .cloned()
            .unwrap_or_else(BigRational::zero),
    })
}

fn scale_polynomial(poly: &Polynomial, scale: &BigRational) -> Polynomial {
    Polynomial::new(
        poly.coeffs.iter().map(|coeff| coeff * scale).collect(),
        poly.var.clone(),
    )
}

fn polynomial_arctan_product(ctx: &mut Context, poly: &Polynomial, arg: ExprId) -> ExprId {
    if poly.is_zero() {
        return ctx.num(0);
    }

    let one = Polynomial::one(poly.var.clone());
    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arg]);
    if *poly == one {
        return arctan;
    }
    if *poly == one.neg() {
        return ctx.add(Expr::Neg(arctan));
    }

    let poly_expr = poly.to_expr(ctx);
    ctx.add(Expr::Mul(poly_expr, arctan))
}

pub(super) fn arctan_affine_by_parts_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let target = unwrap_internal_hold_for_calculus(ctx, target);
    match ctx.get(target).clone() {
        Expr::Mul(left, right) => {
            if cas_ast::views::as_rational_const(ctx, left, 8).is_some() {
                let derivative = arctan_affine_by_parts_compact_derivative(ctx, right, var_name)?;
                let scaled = ctx.add(Expr::Mul(left, derivative));
                return Some(fold_numeric_mul_constants_for_hold(ctx, scaled));
            }
            if cas_ast::views::as_rational_const(ctx, right, 8).is_some() {
                let derivative = arctan_affine_by_parts_compact_derivative(ctx, left, var_name)?;
                let scaled = ctx.add(Expr::Mul(derivative, right));
                return Some(fold_numeric_mul_constants_for_hold(ctx, scaled));
            }
        }
        Expr::Div(num, den) if cas_ast::views::as_rational_const(ctx, den, 8).is_some() => {
            let derivative = arctan_affine_by_parts_compact_derivative(ctx, num, var_name)?;
            let scaled = ctx.add(Expr::Div(derivative, den));
            return Some(fold_numeric_mul_constants_for_hold(ctx, scaled));
        }
        _ => {}
    }

    let terms = cas_math::expr_nary::AddView::from_expr(ctx, target).terms;
    if terms.len() < 2 {
        return None;
    }

    let mut arctan_term: Option<ArctanAffineByPartsTerm> = None;
    let mut ln_term: Option<LnAffineByPartsTerm> = None;
    let mut remainder_poly = Polynomial::zero(var_name.to_string());

    for (term, sign) in terms {
        if let Some(term) = arctan_affine_by_parts_arctan_term(ctx, term, sign, var_name) {
            if let Some(existing) = &mut arctan_term {
                if existing.arg_poly != term.arg_poly {
                    return None;
                }
                existing.cofactor_poly = existing.cofactor_poly.add(&term.cofactor_poly);
            } else {
                arctan_term = Some(term);
            }
            continue;
        }

        if let Some(term) = arctan_affine_by_parts_ln_term(ctx, term, sign, var_name) {
            if let Some(existing) = &mut ln_term {
                if existing.arg_poly != term.arg_poly {
                    return None;
                }
                existing.coefficient += term.coefficient;
            } else {
                ln_term = Some(term);
            }
            continue;
        }

        let term_poly = Polynomial::from_expr(ctx, term, var_name).ok()?;
        remainder_poly = remainder_poly.add(&apply_additive_sign_to_poly(term_poly, sign));
    }

    let arctan_term = arctan_term?;
    let ln_term = ln_term?;
    let derivative_poly = arctan_term.arg_poly.derivative();
    if derivative_poly.degree() != 0 || derivative_poly.is_zero() {
        return None;
    }
    let linear_coeff = derivative_poly.coeffs.first()?.clone();
    if linear_coeff.is_zero() {
        return None;
    }

    let expected_ln_arg_poly = arctan_term
        .arg_poly
        .mul(&arctan_term.arg_poly)
        .add(&Polynomial::one(var_name.to_string()));
    if ln_term.arg_poly != expected_ln_arg_poly {
        return None;
    }

    let rational_numerator = scale_polynomial(&arctan_term.cofactor_poly, &linear_coeff)
        .add(&scale_polynomial(
            &arctan_term.arg_poly,
            &(BigRational::from_integer(2.into()) * &ln_term.coefficient * &linear_coeff),
        ))
        .add(&remainder_poly.derivative().mul(&expected_ln_arg_poly));
    if !rational_numerator.is_zero() {
        return None;
    }

    let arctan_cofactor_derivative = arctan_term.cofactor_poly.derivative();
    Some(polynomial_arctan_product(
        ctx,
        &arctan_cofactor_derivative,
        arctan_term.arg,
    ))
}

#[cfg(test)]
mod tests {
    use super::arctan_affine_by_parts_compact_derivative;
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn arctan_affine_by_parts_compact_derivative_accepts_polynomial_remainder() {
        let mut ctx = Context::new();
        let expr = parse(
            "((x^3+2)*arctan(1-x))/3 + ln(x^2+2-2*x)/3 + x^2/6 + 2*x/3",
            &mut ctx,
        )
        .unwrap();
        let derivative = arctan_affine_by_parts_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "arctan(1 - x) * x^2");

        let normalized = parse(
            "1/6*(2*ln(x^2+2-2*x) + 2*arctan(1-x)*x^3 + 4*arctan(1-x) + x^2 + 4*x)",
            &mut ctx,
        )
        .unwrap();
        let derivative =
            arctan_affine_by_parts_compact_derivative(&mut ctx, normalized, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "x^2 * arctan(1 - x)");
    }

    #[test]
    fn arctan_affine_by_parts_compact_derivative_runs_in_diff_pipeline() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "diff(((x^3+2)*arctan(1-x))/3 + ln(x^2+2-2*x)/3 + x^2/6 + 2*x/3, x)",
            &mut simplifier.context,
        )
        .unwrap();
        let (result, _steps) = simplifier.simplify(expr);

        assert_eq!(rendered(&simplifier.context, result), "arctan(1 - x) * x^2");
    }
}
