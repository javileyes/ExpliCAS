//! Structural planning helpers for `(A^2 - B^2) / (A ± B)` rewrites.

use crate::multipoly::{multipoly_from_expr, multipoly_to_expr, PolyBudget};
use cas_ast::{Context, Expr, ExprId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DifferenceOfSquaresDivisionPolicy {
    pub max_terms: usize,
    pub max_total_degree: u32,
    pub max_pow_exp: u32,
}

impl Default for DifferenceOfSquaresDivisionPolicy {
    fn default() -> Self {
        Self {
            max_terms: 50,
            max_total_degree: 6,
            max_pow_exp: 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DenMatch {
    AMinusB,
    APlusB,
    BMinusA,
    BPlusA,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DifferenceOfSquaresDivisionPlan {
    pub factored_numerator: ExprId,
    pub intermediate_with_orig_den: ExprId,
    pub den_simplified: ExprId,
    pub intermediate: ExprId,
    pub final_result: ExprId,
}

/// Plan rewrite for `(A² - B²) / (A ± B)` style expressions.
///
/// Returns `None` when shape/polynomial checks do not match.
pub fn try_plan_difference_of_squares_division_expr(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
    policy: DifferenceOfSquaresDivisionPolicy,
) -> Option<DifferenceOfSquaresDivisionPlan> {
    // Numerator must match A² - B²:
    // - Sub(Pow(A,2), Pow(B,2))
    // - Add(Pow(A,2), Neg(Pow(B,2)))
    let (a, b) = match ctx.get(numerator) {
        Expr::Sub(left, right) => {
            let a_opt = extract_squared_base(ctx, *left)?;
            let b_opt = extract_squared_base(ctx, *right)?;
            (a_opt, b_opt)
        }
        Expr::Add(left, right) => {
            let a_opt = extract_squared_base(ctx, *left)?;
            if let Expr::Neg(inner) = ctx.get(*right) {
                let b_opt = extract_squared_base(ctx, *inner)?;
                (a_opt, b_opt)
            } else {
                return None;
            }
        }
        _ => return None,
    };

    // Quick structural prefilter: denominator should be additive.
    if !matches!(ctx.get(denominator), Expr::Sub(_, _) | Expr::Add(_, _)) {
        return None;
    }

    let budget = PolyBudget {
        max_terms: policy.max_terms,
        max_total_degree: policy.max_total_degree,
        max_pow_exp: policy.max_pow_exp,
    };

    let a_minus_b_raw = ctx.add(Expr::Sub(a, b));
    let a_plus_b_raw = ctx.add(Expr::Add(a, b));

    let den_poly = multipoly_from_expr(ctx, denominator, &budget).ok()?;
    let a_minus_b_poly = multipoly_from_expr(ctx, a_minus_b_raw, &budget).ok()?;
    let a_plus_b_poly = multipoly_from_expr(ctx, a_plus_b_raw, &budget).ok()?;

    let den_match = if den_poly == a_minus_b_poly {
        DenMatch::AMinusB
    } else if den_poly == a_minus_b_poly.neg() {
        DenMatch::BMinusA
    } else if den_poly == a_plus_b_poly {
        DenMatch::APlusB
    } else if den_poly == a_plus_b_poly.neg() {
        DenMatch::BPlusA
    } else {
        return None;
    };

    let a_minus_b = if let Ok(p) = multipoly_from_expr(ctx, a_minus_b_raw, &budget) {
        multipoly_to_expr(&p, ctx)
    } else {
        a_minus_b_raw
    };
    let a_plus_b = if let Ok(p) = multipoly_from_expr(ctx, a_plus_b_raw, &budget) {
        multipoly_to_expr(&p, ctx)
    } else {
        a_plus_b_raw
    };
    let den_simplified = if let Ok(p) = multipoly_from_expr(ctx, denominator, &budget) {
        multipoly_to_expr(&p, ctx)
    } else {
        denominator
    };

    let factored_numerator = ctx.add(Expr::Mul(a_minus_b, a_plus_b));
    let intermediate_with_orig_den = ctx.add(Expr::Div(factored_numerator, denominator));
    let intermediate = ctx.add(Expr::Div(factored_numerator, den_simplified));

    let final_result = match den_match {
        DenMatch::AMinusB => a_plus_b,
        DenMatch::BMinusA => ctx.add(Expr::Neg(a_plus_b)),
        DenMatch::APlusB | DenMatch::BPlusA => a_minus_b,
    };

    Some(DifferenceOfSquaresDivisionPlan {
        factored_numerator,
        intermediate_with_orig_den,
        den_simplified,
        intermediate,
        final_result,
    })
}

fn extract_squared_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*exp) {
            if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into()) {
                return Some(*base);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{try_plan_difference_of_squares_division_expr, DifferenceOfSquaresDivisionPolicy};
    use cas_ast::{Context, Expr};
    use cas_parser::parse;

    #[test]
    fn plans_basic_difference_of_squares_division() {
        let mut ctx = Context::new();
        let expr = parse("(x^2 - 1^2) / (x - 1)", &mut ctx).expect("parse");
        let (num, den) = match ctx.get(expr) {
            Expr::Div(n, d) => (*n, *d),
            other => panic!("expected div, got {other:?}"),
        };
        let plan = try_plan_difference_of_squares_division_expr(
            &mut ctx,
            num,
            den,
            DifferenceOfSquaresDivisionPolicy::default(),
        )
        .expect("plan");
        let rendered = cas_formatter::render_expr(&ctx, plan.final_result);
        assert_eq!(rendered, "x + 1");
    }

    #[test]
    fn plans_symbolic_additive_denominator_by_polynomial_equality() {
        let mut ctx = Context::new();
        let expr = parse(
            "((x + 2*y)^2 - (3*x - y)^2) / ((x + 2*y) - (3*x - y))",
            &mut ctx,
        )
        .expect("parse");
        let (num, den) = match ctx.get(expr) {
            Expr::Div(n, d) => (*n, *d),
            other => panic!("expected div, got {other:?}"),
        };
        let plan = try_plan_difference_of_squares_division_expr(
            &mut ctx,
            num,
            den,
            DifferenceOfSquaresDivisionPolicy::default(),
        )
        .expect("plan");
        let rendered = cas_formatter::render_expr(&ctx, plan.final_result);
        assert!(rendered.contains("x"));
        assert!(rendered.contains("y"));
    }

    #[test]
    fn rejects_non_matching_denominator() {
        let mut ctx = Context::new();
        let expr = parse("(x^2 - 1) / (x + 2)", &mut ctx).expect("parse");
        let (num, den) = match ctx.get(expr) {
            Expr::Div(n, d) => (*n, *d),
            other => panic!("expected div, got {other:?}"),
        };
        let plan = try_plan_difference_of_squares_division_expr(
            &mut ctx,
            num,
            den,
            DifferenceOfSquaresDivisionPolicy::default(),
        );
        assert!(plan.is_none());
    }
}
