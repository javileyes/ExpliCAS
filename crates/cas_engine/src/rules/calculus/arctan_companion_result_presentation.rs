use super::presentation_utils::unwrap_internal_hold_for_calculus;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::{One, Zero};

struct LnTermForCalculusPresentation {
    arg: ExprId,
    arg_poly: Polynomial,
    coefficient: BigRational,
}

fn extract_ln_term_for_calculus_presentation(
    ctx: &mut Context,
    term: ExprId,
    var_name: &str,
) -> Option<LnTermForCalculusPresentation> {
    let term = unwrap_internal_hold_for_calculus(ctx, term);
    if let Expr::Neg(inner) = ctx.get(term).clone() {
        let mut extracted = extract_ln_term_for_calculus_presentation(ctx, inner, var_name)?;
        extracted.coefficient = -extracted.coefficient;
        return Some(extracted);
    }
    if let Expr::Div(num, den) = ctx.get(term).clone() {
        let denominator = cas_ast::views::as_rational_const(ctx, den, 8)?;
        if denominator.is_zero() {
            return None;
        }
        let mut extracted = extract_ln_term_for_calculus_presentation(ctx, num, var_name)?;
        extracted.coefficient /= denominator;
        return Some(extracted);
    }

    let mut ln_arg = None;
    let mut coefficient = BigRational::one();
    for factor in cas_math::expr_nary::MulView::from_expr(ctx, term).factors {
        let factor = unwrap_internal_hold_for_calculus(ctx, factor);
        if let Expr::Function(fn_id, args) = ctx.get(factor).clone() {
            if ctx.builtin_of(fn_id) == Some(BuiltinFn::Ln) && args.len() == 1 {
                if ln_arg.replace(args[0]).is_some() {
                    return None;
                }
                continue;
            }
        }

        let factor_value = cas_ast::views::as_rational_const(ctx, factor, 8)?;
        coefficient *= factor_value;
    }

    let arg = ln_arg?;
    Some(LnTermForCalculusPresentation {
        arg,
        arg_poly: Polynomial::from_expr(ctx, arg, var_name).ok()?,
        coefficient,
    })
}

fn build_scaled_ln_for_calculus_presentation(
    ctx: &mut Context,
    coefficient: &BigRational,
    arg: ExprId,
) -> Option<ExprId> {
    if coefficient.is_zero() {
        return None;
    }

    let ln = ctx.call_builtin(BuiltinFn::Ln, vec![arg]);
    if coefficient.is_one() {
        return Some(ln);
    }
    if *coefficient == -BigRational::one() {
        return Some(ctx.add(Expr::Neg(ln)));
    }

    let coefficient = ctx.add(Expr::Number(coefficient.clone()));
    Some(ctx.add(Expr::Mul(coefficient, ln)))
}

pub(super) fn ln_polynomial_coefficient_degree_for_calculus_presentation(
    ctx: &mut Context,
    term: ExprId,
    var_name: &str,
) -> Option<usize> {
    let term = unwrap_internal_hold_for_calculus(ctx, term);
    if let Expr::Neg(inner) = ctx.get(term).clone() {
        return ln_polynomial_coefficient_degree_for_calculus_presentation(ctx, inner, var_name);
    }

    let mut ln_seen = false;
    let mut coefficient_factors = Vec::new();
    for factor in cas_math::expr_nary::MulView::from_expr(ctx, term).factors {
        let factor = unwrap_internal_hold_for_calculus(ctx, factor);
        if let Expr::Function(fn_id, args) = ctx.get(factor).clone() {
            if ctx.builtin_of(fn_id) == Some(BuiltinFn::Ln) && args.len() == 1 {
                if ln_seen {
                    return None;
                }
                ln_seen = true;
                continue;
            }
        }
        coefficient_factors.push(factor);
    }
    if !ln_seen {
        return None;
    }

    let coefficient = if coefficient_factors.is_empty() {
        ctx.num(1)
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &coefficient_factors)
    };
    Some(
        Polynomial::from_expr(ctx, coefficient, var_name)
            .ok()?
            .degree(),
    )
}

pub(super) fn compact_arctan_presentation_other_terms(
    ctx: &mut Context,
    terms: Vec<ExprId>,
    var_name: &str,
) -> Vec<ExprId> {
    let mut polynomial_sum = Polynomial::zero(var_name.to_string());
    let mut ln_groups: Vec<LnTermForCalculusPresentation> = Vec::new();
    let mut passthrough = Vec::new();

    for term in terms {
        if let Some(ln_term) = extract_ln_term_for_calculus_presentation(ctx, term, var_name) {
            if let Some(existing) = ln_groups
                .iter_mut()
                .find(|existing| existing.arg_poly == ln_term.arg_poly)
            {
                existing.coefficient += ln_term.coefficient;
            } else {
                ln_groups.push(ln_term);
            }
            continue;
        }

        if let Ok(poly) = Polynomial::from_expr(ctx, term, var_name) {
            polynomial_sum = polynomial_sum.add(&poly);
            continue;
        }

        passthrough.push(term);
    }

    let mut out = Vec::new();
    for ln_term in ln_groups {
        if let Some(term) =
            build_scaled_ln_for_calculus_presentation(ctx, &ln_term.coefficient, ln_term.arg)
        {
            out.push(term);
        }
    }
    if !polynomial_sum.is_zero() {
        out.push(polynomial_sum.to_expr(ctx));
    }
    out.extend(passthrough);
    out
}
