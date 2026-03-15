//! Structural planning helpers for `(A^2 - B^2) / (A ± B)` rewrites.

use crate::abs_support::try_unwrap_abs_arg;
use crate::multipoly::{multipoly_from_expr, multipoly_to_expr, PolyBudget};
use crate::perfect_square_support::rational_sqrt;
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DifferenceOfSquaresDivisionPolicy {
    pub max_terms: usize,
    pub max_total_degree: u32,
    pub max_pow_exp: u32,
    pub allow_abs_square_equiv: bool,
}

impl Default for DifferenceOfSquaresDivisionPolicy {
    fn default() -> Self {
        Self {
            max_terms: 50,
            max_total_degree: 6,
            max_pow_exp: 4,
            allow_abs_square_equiv: false,
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
enum ReciprocalNumeratorMatch {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReciprocalDifferenceOfSquaresPlan {
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
    let (a, b) = match ctx.get(numerator).clone() {
        Expr::Sub(left, right) => {
            let a_opt = extract_squared_base(ctx, left)?;
            let b_opt = extract_squared_base(ctx, right)?;
            (a_opt, b_opt)
        }
        Expr::Add(left, right) => {
            let a_opt = extract_squared_base(ctx, left)?;
            if let Expr::Neg(inner) = ctx.get(right).clone() {
                let b_opt = extract_squared_base(ctx, inner)?;
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

    // Exact hidden-path hotspot: avoid polynomial conversion when the
    // denominator is already the raw `A-B` / `A+B` factor.
    if denominator == a_minus_b_raw || denominator == a_plus_b_raw {
        let factored_numerator = ctx.add(Expr::Mul(a_minus_b_raw, a_plus_b_raw));
        let intermediate = ctx.add(Expr::Div(factored_numerator, denominator));
        let final_result = if denominator == a_minus_b_raw {
            a_plus_b_raw
        } else {
            a_minus_b_raw
        };
        return Some(DifferenceOfSquaresDivisionPlan {
            factored_numerator,
            intermediate_with_orig_den: intermediate,
            den_simplified: denominator,
            intermediate,
            final_result,
        });
    }

    if let Some((rep_a, rep_b, den_match)) =
        match_denominator_pair(ctx, denominator, a, b, policy.allow_abs_square_equiv)
    {
        let a_minus_b = ctx.add(Expr::Sub(rep_a, rep_b));
        let a_plus_b = ctx.add(Expr::Add(rep_a, rep_b));
        let factored_numerator = ctx.add(Expr::Mul(a_minus_b, a_plus_b));
        let intermediate = ctx.add(Expr::Div(factored_numerator, denominator));
        let final_result = match den_match {
            DenMatch::AMinusB => a_plus_b,
            DenMatch::BMinusA => ctx.add(Expr::Neg(a_plus_b)),
            DenMatch::APlusB | DenMatch::BPlusA => a_minus_b,
        };
        return Some(DifferenceOfSquaresDivisionPlan {
            factored_numerator,
            intermediate_with_orig_den: intermediate,
            den_simplified: denominator,
            intermediate,
            final_result,
        });
    }

    let a_minus_b_poly = multipoly_from_expr(ctx, a_minus_b_raw, &budget).ok()?;
    let a_plus_b_poly = multipoly_from_expr(ctx, a_plus_b_raw, &budget).ok()?;
    let exact_den_match = if denominator == a_minus_b_raw {
        Some(DenMatch::AMinusB)
    } else if denominator == a_plus_b_raw {
        Some(DenMatch::APlusB)
    } else {
        None
    };

    let den_match = if let Some(exact) = exact_den_match {
        exact
    } else {
        let den_poly = multipoly_from_expr(ctx, denominator, &budget).ok()?;
        if den_poly == a_minus_b_poly {
            DenMatch::AMinusB
        } else if den_poly == a_minus_b_poly.neg() {
            DenMatch::BMinusA
        } else if den_poly == a_plus_b_poly {
            DenMatch::APlusB
        } else if den_poly == a_plus_b_poly.neg() {
            DenMatch::BPlusA
        } else {
            return None;
        }
    };

    let a_minus_b = multipoly_to_expr(&a_minus_b_poly, ctx);
    let a_plus_b = multipoly_to_expr(&a_plus_b_poly, ctx);
    let den_simplified = if exact_den_match.is_some() {
        denominator
    } else if matches!(den_match, DenMatch::BMinusA | DenMatch::BPlusA) {
        match den_match {
            DenMatch::BMinusA => ctx.add(Expr::Neg(a_minus_b)),
            DenMatch::BPlusA => ctx.add(Expr::Neg(a_plus_b)),
            _ => unreachable!(),
        }
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

/// Plan rewrite for `(A ± B) / (A² - B²)` style expressions.
pub fn try_plan_reciprocal_difference_of_squares_expr(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
) -> Option<ReciprocalDifferenceOfSquaresPlan> {
    let (a, b, den_sign) = match ctx.get(denominator).clone() {
        Expr::Sub(left, right) => {
            let a = extract_squared_base(ctx, left)?;
            let b = extract_squared_base(ctx, right)?;
            (a, b, 1_i32)
        }
        Expr::Add(left, right) => {
            if let Expr::Neg(inner) = ctx.get(right).clone() {
                let a = extract_squared_base(ctx, left)?;
                let b = extract_squared_base(ctx, inner)?;
                (a, b, 1_i32)
            } else if let Some(b) = extract_negative_squared_base(ctx, right) {
                let a = extract_squared_base(ctx, left)?;
                (a, b, 1_i32)
            } else if let Expr::Neg(inner) = ctx.get(left).clone() {
                let a = extract_squared_base(ctx, inner)?;
                let b = extract_squared_base(ctx, right)?;
                (a, b, -1_i32)
            } else if let Some(a) = extract_negative_squared_base(ctx, left) {
                let b = extract_squared_base(ctx, right)?;
                (a, b, -1_i32)
            } else {
                return None;
            }
        }
        _ => return None,
    };

    let num_match = match_additive_pair(ctx, numerator, a, b)?;
    let a_minus_b = ctx.add(Expr::Sub(a, b));
    let a_plus_b = ctx.add(Expr::Add(a, b));

    let (result_den, result_sign) = match num_match {
        ReciprocalNumeratorMatch::AMinusB => (a_plus_b, den_sign),
        ReciprocalNumeratorMatch::BMinusA => (a_plus_b, -den_sign),
        ReciprocalNumeratorMatch::APlusB | ReciprocalNumeratorMatch::BPlusA => {
            (a_minus_b, den_sign)
        }
    };

    let one = ctx.num(1);
    let reciprocal = ctx.add(Expr::Div(one, result_den));
    let final_result = if result_sign < 0 {
        ctx.add(Expr::Neg(reciprocal))
    } else {
        reciprocal
    };

    Some(ReciprocalDifferenceOfSquaresPlan { final_result })
}

fn extract_squared_base(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*exp) {
            if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into()) {
                return Some(*base);
            }
        }
    }
    if let Expr::Number(n) = ctx.get(expr) {
        if let Some(root) = rational_sqrt(n) {
            return Some(ctx.add(Expr::Number(root)));
        }
    }
    None
}

fn extract_negative_squared_base(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        return extract_squared_base(ctx, inner);
    }
    if let Expr::Number(n) = ctx.get(expr) {
        let negated = -n.clone();
        let root = rational_sqrt(&negated)?;
        return Some(ctx.add(Expr::Number(root)));
    }
    None
}

fn square_base_equivalent(
    ctx: &Context,
    lhs: ExprId,
    rhs: ExprId,
    allow_abs_square_equiv: bool,
) -> bool {
    if compare_expr(ctx, lhs, rhs) == Ordering::Equal {
        return true;
    }
    if !allow_abs_square_equiv {
        return false;
    }
    match (try_unwrap_abs_arg(ctx, lhs), try_unwrap_abs_arg(ctx, rhs)) {
        (Some(inner), _) if compare_expr(ctx, inner, rhs) == Ordering::Equal => true,
        (_, Some(inner)) if compare_expr(ctx, lhs, inner) == Ordering::Equal => true,
        (Some(left_inner), Some(right_inner)) => {
            compare_expr(ctx, left_inner, right_inner) == Ordering::Equal
        }
        _ => false,
    }
}

fn match_denominator_pair(
    ctx: &Context,
    expr: ExprId,
    a: ExprId,
    b: ExprId,
    allow_abs_square_equiv: bool,
) -> Option<(ExprId, ExprId, DenMatch)> {
    match ctx.get(expr) {
        Expr::Sub(left, right) => {
            if square_base_equivalent(ctx, *left, a, allow_abs_square_equiv)
                && square_base_equivalent(ctx, *right, b, allow_abs_square_equiv)
            {
                Some((*left, *right, DenMatch::AMinusB))
            } else if square_base_equivalent(ctx, *left, b, allow_abs_square_equiv)
                && square_base_equivalent(ctx, *right, a, allow_abs_square_equiv)
            {
                Some((*right, *left, DenMatch::BMinusA))
            } else {
                None
            }
        }
        Expr::Add(left, right) => {
            if square_base_equivalent(ctx, *left, a, allow_abs_square_equiv)
                && square_base_equivalent(ctx, *right, b, allow_abs_square_equiv)
            {
                Some((*left, *right, DenMatch::APlusB))
            } else if square_base_equivalent(ctx, *left, a, allow_abs_square_equiv)
                && matches_neg_equivalent(ctx, *right, b)
            {
                Some((*left, b, DenMatch::AMinusB))
            } else if square_base_equivalent(ctx, *left, b, allow_abs_square_equiv)
                && square_base_equivalent(ctx, *right, a, allow_abs_square_equiv)
            {
                Some((*right, *left, DenMatch::BPlusA))
            } else if square_base_equivalent(ctx, *left, b, allow_abs_square_equiv)
                && matches_neg_equivalent(ctx, *right, a)
            {
                Some((a, *left, DenMatch::BMinusA))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn match_additive_pair(
    ctx: &Context,
    expr: ExprId,
    a: ExprId,
    b: ExprId,
) -> Option<ReciprocalNumeratorMatch> {
    match ctx.get(expr) {
        Expr::Sub(left, right) => {
            if compare_expr(ctx, *left, a) == Ordering::Equal
                && compare_expr(ctx, *right, b) == Ordering::Equal
            {
                Some(ReciprocalNumeratorMatch::AMinusB)
            } else if compare_expr(ctx, *left, b) == Ordering::Equal
                && compare_expr(ctx, *right, a) == Ordering::Equal
            {
                Some(ReciprocalNumeratorMatch::BMinusA)
            } else {
                None
            }
        }
        Expr::Add(left, right) => {
            if compare_expr(ctx, *left, a) == Ordering::Equal
                && matches_neg_equivalent(ctx, *right, b)
            {
                return Some(ReciprocalNumeratorMatch::AMinusB);
            }
            if compare_expr(ctx, *left, b) == Ordering::Equal
                && matches_neg_equivalent(ctx, *right, a)
            {
                return Some(ReciprocalNumeratorMatch::BMinusA);
            }
            if matches_neg_equivalent(ctx, *left, a)
                && compare_expr(ctx, *right, b) == Ordering::Equal
            {
                return Some(ReciprocalNumeratorMatch::BMinusA);
            }
            if matches_neg_equivalent(ctx, *left, b)
                && compare_expr(ctx, *right, a) == Ordering::Equal
            {
                return Some(ReciprocalNumeratorMatch::AMinusB);
            }
            if compare_expr(ctx, *left, a) == Ordering::Equal
                && compare_expr(ctx, *right, b) == Ordering::Equal
            {
                Some(ReciprocalNumeratorMatch::APlusB)
            } else if compare_expr(ctx, *left, b) == Ordering::Equal
                && compare_expr(ctx, *right, a) == Ordering::Equal
            {
                Some(ReciprocalNumeratorMatch::BPlusA)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn matches_neg_equivalent(ctx: &Context, expr: ExprId, target: ExprId) -> bool {
    if let Expr::Neg(inner) = ctx.get(expr) {
        return compare_expr(ctx, *inner, target) == Ordering::Equal;
    }
    match (ctx.get(expr), ctx.get(target)) {
        (Expr::Number(lhs), Expr::Number(rhs)) => lhs == &(-rhs.clone()),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        try_plan_difference_of_squares_division_expr,
        try_plan_reciprocal_difference_of_squares_expr, DifferenceOfSquaresDivisionPolicy,
    };
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

    #[test]
    fn plans_reciprocal_difference_of_squares_with_opaque_atom() {
        let mut ctx = Context::new();
        let expr = parse("(arctan(u) - 1) / (arctan(u)^2 - 1)", &mut ctx).expect("parse");
        let (num, den) = match ctx.get(expr) {
            Expr::Div(n, d) => (*n, *d),
            other => panic!("expected div, got {other:?}"),
        };
        let plan =
            try_plan_reciprocal_difference_of_squares_expr(&mut ctx, num, den).expect("plan");
        let rendered = cas_formatter::render_expr(&ctx, plan.final_result);
        assert_eq!(rendered, "1 / (arctan(u) + 1)");
    }

    #[test]
    fn plans_reciprocal_difference_of_squares_with_canonicalized_add_neg_numerator() {
        let mut ctx = Context::new();
        let expr = parse("(arctan(u) + (-1)) / (arctan(u)^2 + (-1))", &mut ctx).expect("parse");
        let (num, den) = match ctx.get(expr) {
            Expr::Div(n, d) => (*n, *d),
            other => panic!("expected div, got {other:?}"),
        };
        let plan =
            try_plan_reciprocal_difference_of_squares_expr(&mut ctx, num, den).expect("plan");
        let rendered = cas_formatter::render_expr(&ctx, plan.final_result);
        assert_eq!(rendered, "1 / (arctan(u) + 1)");
    }

    #[test]
    fn plans_difference_of_squares_with_abs_denominator_when_allowed() {
        let mut ctx = Context::new();
        let expr = parse("(u^2 - 4) / (abs(u) + 2)", &mut ctx).expect("parse");
        let (num, den) = match ctx.get(expr) {
            Expr::Div(n, d) => (*n, *d),
            other => panic!("expected div, got {other:?}"),
        };
        let plan = try_plan_difference_of_squares_division_expr(
            &mut ctx,
            num,
            den,
            DifferenceOfSquaresDivisionPolicy {
                allow_abs_square_equiv: true,
                ..DifferenceOfSquaresDivisionPolicy::default()
            },
        )
        .expect("plan");
        let rendered = cas_formatter::render_expr(&ctx, plan.final_result);
        assert_eq!(rendered, "|u| - 2");
    }

    #[test]
    fn rejects_difference_of_squares_with_abs_denominator_when_disallowed() {
        let mut ctx = Context::new();
        let expr = parse("(u^2 - 4) / (abs(u) + 2)", &mut ctx).expect("parse");
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
