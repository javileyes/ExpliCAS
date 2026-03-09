//! Unified planning support for fraction GCD rewrites.
//!
//! Produces structural rewrite plans (multivariate or univariate) while leaving
//! domain-policy decisions to higher layers.

use crate::expr_nary::{AddView, Sign};
use crate::fraction_multivar_gcd::try_multivar_gcd;
use crate::fraction_univar_gcd_support::{
    build_fraction_cancel_forms, try_univariate_fraction_gcd_reduction, FractionCancelForms,
};
use crate::multipoly::GcdLayer;
use crate::trig_linear_support::extract_coef_and_base;
use cas_ast::{collect_variables, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Zero};

fn signed_term_coeff_and_base(
    ctx: &Context,
    term: (ExprId, Sign),
) -> Option<(BigRational, ExprId)> {
    let (mut coeff, base) = extract_coef_and_base(ctx, term.0);
    if coeff.is_zero() {
        return None;
    }
    if term.1 == Sign::Neg {
        coeff = -coeff;
    }
    Some((coeff, base))
}

fn try_structural_scalar_multiple_fraction_reduction(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
) -> Option<(ExprId, ExprId, ExprId)> {
    let num_view = AddView::from_expr(ctx, num);
    let den_view = AddView::from_expr(ctx, den);
    if num_view.terms.len() < 2 || num_view.terms.len() != den_view.terms.len() {
        return None;
    }

    let mut den_terms: Vec<(BigRational, ExprId, bool)> = den_view
        .terms
        .iter()
        .copied()
        .map(|term| signed_term_coeff_and_base(ctx, term).map(|(coeff, base)| (coeff, base, false)))
        .collect::<Option<_>>()?;

    let mut ratio: Option<BigRational> = None;
    for term in num_view.terms.iter().copied() {
        let (num_coeff, num_base) = signed_term_coeff_and_base(ctx, term)?;
        let Some((den_coeff, _, used)) =
            den_terms.iter_mut().find(|(den_coeff, den_base, used)| {
                !*used
                    && !den_coeff.is_zero()
                    && cas_ast::ordering::compare_expr(ctx, num_base, *den_base)
                        == std::cmp::Ordering::Equal
            })
        else {
            return None;
        };
        let candidate_ratio = den_coeff.clone() / num_coeff;
        if candidate_ratio.is_zero() {
            return None;
        }
        match &ratio {
            Some(existing) if existing != &candidate_ratio => return None,
            None => ratio = Some(candidate_ratio),
            _ => {}
        }
        *used = true;
    }

    let ratio = ratio?;
    if ratio.is_one() {
        return None;
    }

    let result_ratio = BigRational::one() / ratio;
    let new_num = ctx.add(Expr::Number(BigRational::from_integer(
        result_ratio.numer().clone(),
    )));
    let new_den = ctx.add(Expr::Number(BigRational::from_integer(
        result_ratio.denom().clone(),
    )));
    Some((new_num, new_den, num))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FractionGcdRoute {
    StructuralScalarMultiple,
    Multivar { layer: GcdLayer },
    Univar,
}

#[derive(Debug, Clone)]
pub struct FractionGcdRewritePlan {
    pub route: FractionGcdRoute,
    pub gcd_expr: ExprId,
    pub forms: FractionCancelForms,
    pub strict_partial_result: Option<ExprId>,
    pub strict_partial_numeric_gcd: Option<BigRational>,
}

/// Try the exact additive scalar-multiple fast path only.
pub fn try_plan_structural_scalar_multiple_fraction_rewrite(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    include_factored_form: bool,
) -> Option<FractionGcdRewritePlan> {
    let (new_num, new_den, gcd_expr) =
        try_structural_scalar_multiple_fraction_reduction(ctx, num, den)?;
    let forms = build_fraction_cancel_forms(ctx, new_num, new_den, gcd_expr, include_factored_form);
    Some(FractionGcdRewritePlan {
        route: FractionGcdRoute::StructuralScalarMultiple,
        gcd_expr,
        forms,
        strict_partial_result: None,
        strict_partial_numeric_gcd: None,
    })
}

/// Try to plan a GCD-based fraction rewrite for `num/den` within `expr`.
///
/// Selection policy:
/// - First, short-circuit exact scalar-multiple additive fractions structurally.
/// - Prefer multivariate path when `expr` contains more than one variable.
/// - Fall back to univariate path when exactly one variable is present.
/// - Return `None` when no non-trivial GCD rewrite is available.
pub fn try_plan_fraction_gcd_rewrite(
    ctx: &mut Context,
    expr: ExprId,
    num: ExprId,
    den: ExprId,
    include_factored_form: bool,
) -> Option<FractionGcdRewritePlan> {
    if let Some(plan) =
        try_plan_structural_scalar_multiple_fraction_rewrite(ctx, num, den, include_factored_form)
    {
        return Some(plan);
    }

    let vars = collect_variables(ctx, expr);

    if vars.len() > 1 {
        let (new_num, new_den, gcd_expr, layer) = try_multivar_gcd(ctx, num, den)?;
        let forms =
            build_fraction_cancel_forms(ctx, new_num, new_den, gcd_expr, include_factored_form);
        return Some(FractionGcdRewritePlan {
            route: FractionGcdRoute::Multivar { layer },
            gcd_expr,
            forms,
            strict_partial_result: None,
            strict_partial_numeric_gcd: None,
        });
    }

    if vars.len() != 1 {
        return None;
    }
    let var = vars.iter().next()?;
    let reduced = try_univariate_fraction_gcd_reduction(ctx, num, den, var)?;
    let forms = build_fraction_cancel_forms(
        ctx,
        reduced.new_num,
        reduced.new_den,
        reduced.gcd_expr,
        include_factored_form,
    );

    let strict_partial_result = match (reduced.partial_num, reduced.partial_den) {
        (Some(n), Some(d)) => Some(ctx.add(Expr::Div(n, d))),
        _ => None,
    };
    let strict_partial_numeric_gcd = strict_partial_result.map(|_| reduced.numeric_gcd.clone());

    Some(FractionGcdRewritePlan {
        route: FractionGcdRoute::Univar,
        gcd_expr: reduced.gcd_expr,
        forms,
        strict_partial_result,
        strict_partial_numeric_gcd,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        try_plan_fraction_gcd_rewrite, try_plan_structural_scalar_multiple_fraction_rewrite,
        FractionGcdRoute,
    };
    use crate::multipoly::GcdLayer;
    use crate::poly_compare::poly_eq;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn plan_univar_path_and_partial_result() {
        let mut ctx = Context::new();
        let expr = parse("(27*x^3)/(9*x)", &mut ctx).expect("parse");
        let num = parse("27*x^3", &mut ctx).expect("parse");
        let den = parse("9*x", &mut ctx).expect("parse");
        let plan = try_plan_fraction_gcd_rewrite(&mut ctx, expr, num, den, true).expect("plan");
        assert!(matches!(plan.route, FractionGcdRoute::Univar));
        assert!(plan.strict_partial_result.is_some());
        assert!(plan.strict_partial_numeric_gcd.is_some());
    }

    #[test]
    fn plan_multivar_path() {
        let mut ctx = Context::new();
        let expr = parse("(x*y+x*z)/x", &mut ctx).expect("parse");
        let num = parse("x*y+x*z", &mut ctx).expect("parse");
        let den = parse("x", &mut ctx).expect("parse");
        let plan = try_plan_fraction_gcd_rewrite(&mut ctx, expr, num, den, true).expect("plan");
        assert!(matches!(
            plan.route,
            FractionGcdRoute::Multivar {
                layer: GcdLayer::Layer1MonomialContent
            }
        ));
        let expected = parse("y+z", &mut ctx).expect("expected");
        assert!(poly_eq(&ctx, plan.forms.result_norm, expected));
    }

    #[test]
    fn plan_multivar_structural_scalar_multiple_fast_path() {
        let mut ctx = Context::new();
        let expr = parse("(2*x + 2*y)/(4*y + 4*x)", &mut ctx).expect("parse");
        let num = parse("2*x + 2*y", &mut ctx).expect("parse");
        let den = parse("4*y + 4*x", &mut ctx).expect("parse");
        let plan = try_plan_fraction_gcd_rewrite(&mut ctx, expr, num, den, true).expect("plan");
        assert!(matches!(
            plan.route,
            FractionGcdRoute::StructuralScalarMultiple
        ));
        let expected = parse("1/2", &mut ctx).expect("expected");
        assert!(poly_eq(&ctx, plan.forms.result_norm, expected));
        assert_eq!(plan.gcd_expr, num);
    }

    #[test]
    fn structural_scalar_multiple_helper_matches_full_plan_shape() {
        let mut ctx = Context::new();
        let num = parse("2*x + 2*y", &mut ctx).expect("parse");
        let den = parse("4*y + 4*x", &mut ctx).expect("parse");
        let plan = try_plan_structural_scalar_multiple_fraction_rewrite(&mut ctx, num, den, false)
            .expect("plan");
        assert!(matches!(
            plan.route,
            FractionGcdRoute::StructuralScalarMultiple
        ));
        assert!(plan.forms.factored_form_norm.is_none());
        let expected = parse("1/2", &mut ctx).expect("expected");
        assert!(poly_eq(&ctx, plan.forms.result_norm, expected));
    }

    #[test]
    fn plan_can_skip_factored_form_for_plain_runtime() {
        let mut ctx = Context::new();
        let expr = parse("(2*x + 2*y)/(4*y + 4*x)", &mut ctx).expect("parse");
        let num = parse("2*x + 2*y", &mut ctx).expect("parse");
        let den = parse("4*y + 4*x", &mut ctx).expect("parse");
        let plan = try_plan_fraction_gcd_rewrite(&mut ctx, expr, num, den, false).expect("plan");
        assert!(plan.forms.factored_form_norm.is_none());
        let expected = parse("1/2", &mut ctx).expect("expected");
        assert!(poly_eq(&ctx, plan.forms.result_norm, expected));
    }

    #[test]
    fn no_plan_when_gcd_trivial() {
        let mut ctx = Context::new();
        let expr = parse("(x+1)/(x+2)", &mut ctx).expect("parse");
        let num = parse("x+1", &mut ctx).expect("parse");
        let den = parse("x+2", &mut ctx).expect("parse");
        assert!(try_plan_fraction_gcd_rewrite(&mut ctx, expr, num, den, true).is_none());
    }

    #[test]
    fn no_plan_for_non_polynomial_symbolic_power_fraction() {
        let mut ctx = Context::new();
        let expr = parse("(a^x)/a", &mut ctx).expect("parse");
        let num = parse("a^x", &mut ctx).expect("parse");
        let den = parse("a", &mut ctx).expect("parse");
        assert!(try_plan_fraction_gcd_rewrite(&mut ctx, expr, num, den, false).is_none());
    }
}
