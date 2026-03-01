//! Unified planning support for fraction GCD rewrites.
//!
//! Produces structural rewrite plans (multivariate or univariate) while leaving
//! domain-policy decisions to higher layers.

use crate::fraction_multivar_gcd::try_multivar_gcd;
use crate::fraction_univar_gcd_support::{
    build_fraction_cancel_forms, try_univariate_fraction_gcd_reduction, FractionCancelForms,
};
use crate::multipoly::GcdLayer;
use cas_ast::{collect_variables, Context, Expr, ExprId};
use num_rational::BigRational;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FractionGcdRoute {
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

/// Try to plan a GCD-based fraction rewrite for `num/den` within `expr`.
///
/// Selection policy:
/// - Prefer multivariate path when `expr` contains more than one variable.
/// - Fall back to univariate path when exactly one variable is present.
/// - Return `None` when no non-trivial GCD rewrite is available.
pub fn try_plan_fraction_gcd_rewrite(
    ctx: &mut Context,
    expr: ExprId,
    num: ExprId,
    den: ExprId,
) -> Option<FractionGcdRewritePlan> {
    let vars = collect_variables(ctx, expr);

    if vars.len() > 1 {
        let (new_num, new_den, gcd_expr, layer) = try_multivar_gcd(ctx, num, den)?;
        let forms = build_fraction_cancel_forms(ctx, new_num, new_den, gcd_expr);
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
    let forms =
        build_fraction_cancel_forms(ctx, reduced.new_num, reduced.new_den, reduced.gcd_expr);

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

/// Build description text for a factorization step by GCD.
///
/// Caller provides expression rendering to keep this module independent from
/// formatter crates.
pub fn format_factor_by_gcd_desc_with<FRender>(gcd_expr: ExprId, mut render_expr: FRender) -> String
where
    FRender: FnMut(ExprId) -> String,
{
    format!("Factor by GCD: {}", render_expr(gcd_expr))
}

#[cfg(test)]
mod tests {
    use super::{format_factor_by_gcd_desc_with, try_plan_fraction_gcd_rewrite, FractionGcdRoute};
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
        let plan = try_plan_fraction_gcd_rewrite(&mut ctx, expr, num, den).expect("plan");
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
        let plan = try_plan_fraction_gcd_rewrite(&mut ctx, expr, num, den).expect("plan");
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
    fn no_plan_when_gcd_trivial() {
        let mut ctx = Context::new();
        let expr = parse("(x+1)/(x+2)", &mut ctx).expect("parse");
        let num = parse("x+1", &mut ctx).expect("parse");
        let den = parse("x+2", &mut ctx).expect("parse");
        assert!(try_plan_fraction_gcd_rewrite(&mut ctx, expr, num, den).is_none());
    }

    #[test]
    fn factor_desc_contains_prefix() {
        let desc =
            format_factor_by_gcd_desc_with(cas_ast::ExprId::from_raw(42), |id| format!("{:?}", id));
        assert!(desc.starts_with("Factor by GCD:"));
    }
}
