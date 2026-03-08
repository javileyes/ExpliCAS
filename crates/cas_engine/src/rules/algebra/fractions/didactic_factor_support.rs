use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_destructure::{as_add, as_neg, as_pow, as_sub};
use num_rational::BigRational;
use std::cmp::Ordering;

use cas_math::expr_rewrite::smart_mul;
use cas_math::numeric::as_i64;
use cas_math::poly_compare::poly_eq;

/// Plan for `(a^2 + 2ab + b^2)/(a+b)^2` denominator expansion prior to cancel.
#[derive(Clone, Debug)]
pub struct ExpandBinomialSquareDenCancelPlan {
    pub rewritten: ExprId,
    pub expanded_denominator: ExprId,
}

/// Plan for `(a^2 - b^2)/(a±b)` factorization + cancel.
#[derive(Clone, Debug)]
pub struct DifferenceOfSquaresCancelPlan {
    pub rewritten: ExprId,
    pub factored_numerator: ExprId,
}

/// Plan for `(a^2 - 2ab + b^2)/(a-b)` recognition as `(a-b)^2/(a-b)`.
#[derive(Clone, Debug)]
pub struct PerfectSquareMinusCancelPlan {
    pub rewritten: ExprId,
    pub factored_numerator: ExprId,
}

/// Plan for `(a^3 ± b^3)/(a±b)` factorization into `(a±b)(a^2 ∓ ab + b^2)/(a±b)`.
#[derive(Clone, Debug)]
pub struct SumDiffCubesCancelPlan {
    pub rewritten: ExprId,
    pub factored_numerator: ExprId,
    pub desc: &'static str,
}

/// Plan for `P^m / P^n -> P^(m-n)` while preserving base structure.
#[derive(Clone, Debug)]
pub struct PowerQuotientPreserveFormPlan {
    pub rewritten: ExprId,
    pub desc: String,
}

/// Unified didactic-cancellation plan for fraction rewrites in preferred order.
#[derive(Clone, Debug)]
pub struct FractionDidacticCancelPlan {
    pub rewritten: ExprId,
    pub local_after: ExprId,
    pub desc: String,
}

/// Detect `(a^2 + 2ab + b^2)/(a+b)^2` and plan denominator expansion.
pub fn try_plan_expand_binomial_square_in_den_for_cancel(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
) -> Option<ExpandBinomialSquareDenCancelPlan> {
    let (base, exp) = as_pow(ctx, den)?;
    if as_i64(ctx, exp)? != 2 {
        return None;
    }

    let (a, b) = as_add(ctx, base)?;

    let two = ctx.num(2);
    let exp_two = ctx.num(2);
    let a_sq = ctx.add(Expr::Pow(a, exp_two));
    let exp_two_b = ctx.num(2);
    let b_sq = ctx.add(Expr::Pow(b, exp_two_b));
    let a_times_b = smart_mul(ctx, a, b);
    let two_ab = smart_mul(ctx, two, a_times_b);
    let middle_sum = ctx.add(Expr::Add(two_ab, b_sq));
    let expanded = ctx.add(Expr::Add(a_sq, middle_sum));

    if !poly_eq(ctx, num, expanded) {
        return None;
    }

    let rewritten = ctx.add(Expr::Div(num, expanded));
    Some(ExpandBinomialSquareDenCancelPlan {
        rewritten,
        expanded_denominator: expanded,
    })
}

/// Detect `(a^2 - b^2)/(a+b)` or `(a^2 - b^2)/(a-b)` and plan factored cancellation.
pub fn try_plan_difference_of_squares_in_num(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
) -> Option<DifferenceOfSquaresCancelPlan> {
    let (a, b) = if let Some((left, right)) = as_sub(ctx, num) {
        let (a, exp_a) = as_pow(ctx, left)?;
        if as_i64(ctx, exp_a)? != 2 {
            return None;
        }

        if let Some((b, exp_b)) = as_pow(ctx, right) {
            if as_i64(ctx, exp_b)? != 2 {
                return None;
            }
            (a, b)
        } else if let Expr::Number(n) = ctx.get(right) {
            let sqrt_n = try_integer_sqrt(n)?;
            let b = ctx.add(Expr::Number(sqrt_n));
            (a, b)
        } else {
            return None;
        }
    } else if let Some((left, right)) = as_add(ctx, num) {
        let (a, exp_a) = as_pow(ctx, left)?;
        if as_i64(ctx, exp_a)? != 2 {
            return None;
        }

        let neg_inner = as_neg(ctx, right)?;
        if let Some((b, exp_b)) = as_pow(ctx, neg_inner) {
            if as_i64(ctx, exp_b)? != 2 {
                return None;
            }
            (a, b)
        } else if let Expr::Number(n) = ctx.get(neg_inner) {
            let sqrt_n = try_integer_sqrt(n)?;
            let b = ctx.add(Expr::Number(sqrt_n));
            (a, b)
        } else {
            return None;
        }
    } else {
        return None;
    };

    let den_matches_a_plus_b = if let Some((da, db)) = as_add(ctx, den) {
        (compare_expr(ctx, da, a) == Ordering::Equal && compare_expr(ctx, db, b) == Ordering::Equal)
            || (compare_expr(ctx, da, b) == Ordering::Equal
                && compare_expr(ctx, db, a) == Ordering::Equal)
    } else {
        false
    };

    let den_matches_a_minus_b = if let Some((da, db)) = as_sub(ctx, den) {
        compare_expr(ctx, da, a) == Ordering::Equal && compare_expr(ctx, db, b) == Ordering::Equal
    } else {
        false
    };

    if !den_matches_a_plus_b && !den_matches_a_minus_b {
        return None;
    }

    let a_minus_b = ctx.add(Expr::Sub(a, b));
    let a_plus_b = ctx.add(Expr::Add(a, b));
    let factored_num = smart_mul(ctx, a_minus_b, a_plus_b);
    let rewritten = if den_matches_a_plus_b {
        a_minus_b
    } else {
        a_plus_b
    };

    Some(DifferenceOfSquaresCancelPlan {
        rewritten,
        factored_numerator: factored_num,
    })
}

/// Detect `(a^2 - 2ab + b^2)/(a-b)` and plan rewrite to `(a-b)^2/(a-b)`.
pub fn try_plan_perfect_square_minus_in_num(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
) -> Option<PerfectSquareMinusCancelPlan> {
    let (a, b) = if let Some((a, b)) = as_sub(ctx, den) {
        (a, b)
    } else if let Some((left, right)) = as_add(ctx, den) {
        if let Some(neg_inner) = as_neg(ctx, right) {
            (left, neg_inner)
        } else if let Some(neg_inner) = as_neg(ctx, left) {
            (right, neg_inner)
        } else {
            return None;
        }
    } else {
        return None;
    };

    let two = ctx.num(2);
    let exp_two = ctx.num(2);
    let a_sq = ctx.add(Expr::Pow(a, exp_two));
    let exp_two_b = ctx.num(2);
    let b_sq = ctx.add(Expr::Pow(b, exp_two_b));
    let a_b = smart_mul(ctx, a, b);
    let two_ab = smart_mul(ctx, two, a_b);
    let neg_two_ab = ctx.add(Expr::Neg(two_ab));
    let inner_sum = ctx.add(Expr::Add(neg_two_ab, b_sq));
    let expected_num = ctx.add(Expr::Add(a_sq, inner_sum));

    if !poly_eq(ctx, num, expected_num) {
        return None;
    }

    let a_minus_b = ctx.add(Expr::Sub(a, b));
    let exp_for_square = ctx.num(2);
    let factored_num = ctx.add(Expr::Pow(a_minus_b, exp_for_square));
    let rewritten = ctx.add(Expr::Div(factored_num, den));

    Some(PerfectSquareMinusCancelPlan {
        rewritten,
        factored_numerator: factored_num,
    })
}

/// Detect `(a^3 - b^3)/(a-b)` or `(a^3 + b^3)/(a+b)` and plan didactic factorization.
pub fn try_plan_sum_diff_of_cubes_in_num(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
) -> Option<SumDiffCubesCancelPlan> {
    let (a, b, is_difference) = if let Some((left, right)) = as_sub(ctx, num) {
        let (a, exp_a) = as_pow(ctx, left)?;
        if as_i64(ctx, exp_a)? != 3 {
            return None;
        }

        let (b, exp_b) = as_pow(ctx, right)?;
        if as_i64(ctx, exp_b)? != 3 {
            return None;
        }

        (a, b, true)
    } else if let Some((left, right)) = as_add(ctx, num) {
        let (a, exp_a) = as_pow(ctx, left)?;
        if as_i64(ctx, exp_a)? != 3 {
            return None;
        }

        if let Some(neg_inner) = as_neg(ctx, right) {
            let (b, exp_b) = as_pow(ctx, neg_inner)?;
            if as_i64(ctx, exp_b)? != 3 {
                return None;
            }
            (a, b, true)
        } else {
            let (b, exp_b) = as_pow(ctx, right)?;
            if as_i64(ctx, exp_b)? != 3 {
                return None;
            }
            (a, b, false)
        }
    } else {
        return None;
    };

    let den_is_a_minus_b = if let Some((da, db)) = as_sub(ctx, den) {
        poly_eq(ctx, da, a) && poly_eq(ctx, db, b)
    } else if let Some((left, right)) = as_add(ctx, den) {
        if let Some(neg_inner) = as_neg(ctx, right) {
            poly_eq(ctx, left, a) && poly_eq(ctx, neg_inner, b)
        } else if let Some(neg_inner) = as_neg(ctx, left) {
            poly_eq(ctx, right, a) && poly_eq(ctx, neg_inner, b)
        } else {
            false
        }
    } else {
        false
    };

    let den_is_a_plus_b = if let Some((da, db)) = as_add(ctx, den) {
        if as_neg(ctx, da).is_some() || as_neg(ctx, db).is_some() {
            false
        } else {
            (poly_eq(ctx, da, a) && poly_eq(ctx, db, b))
                || (poly_eq(ctx, da, b) && poly_eq(ctx, db, a))
        }
    } else {
        false
    };

    if is_difference && !den_is_a_minus_b {
        return None;
    }
    if !is_difference && !den_is_a_plus_b {
        return None;
    }

    let exp_two = ctx.num(2);
    let a_sq = ctx.add(Expr::Pow(a, exp_two));
    let exp_two_b = ctx.num(2);
    let b_sq = ctx.add(Expr::Pow(b, exp_two_b));
    let ab = smart_mul(ctx, a, b);

    let result = if is_difference {
        let inner = ctx.add(Expr::Add(ab, b_sq));
        ctx.add(Expr::Add(a_sq, inner))
    } else {
        let neg_ab = ctx.add(Expr::Neg(ab));
        let inner = ctx.add(Expr::Add(neg_ab, b_sq));
        ctx.add(Expr::Add(a_sq, inner))
    };

    let linear_factor = if is_difference {
        ctx.add(Expr::Sub(a, b))
    } else {
        ctx.add(Expr::Add(a, b))
    };
    let factored_num = smart_mul(ctx, linear_factor, result);
    let rewritten = ctx.add(Expr::Div(factored_num, den));

    let desc = if is_difference {
        "Factor: a³ - b³ = (a-b)(a² + ab + b²)"
    } else {
        "Factor: a³ + b³ = (a+b)(a² - ab + b²)"
    };

    Some(SumDiffCubesCancelPlan {
        rewritten,
        factored_numerator: factored_num,
        desc,
    })
}

/// Detect `P^m / P^n` (or implicit exponent 1) and plan `P^(m-n)` for `m>n`.
pub fn try_plan_power_quotient_preserve_form(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
) -> Option<PowerQuotientPreserveFormPlan> {
    let (num_base, num_exp) = if let Some((base, exp)) = as_pow(ctx, num) {
        (base, Some(exp))
    } else {
        (num, None)
    };

    let (den_base, den_exp) = if let Some((base, exp)) = as_pow(ctx, den) {
        (base, Some(exp))
    } else {
        (den, None)
    };

    if !poly_eq(ctx, num_base, den_base) {
        return None;
    }

    let num_exp_val = match num_exp {
        Some(e) => as_i64(ctx, e)?,
        None => 1,
    };
    let den_exp_val = match den_exp {
        Some(e) => as_i64(ctx, e)?,
        None => 1,
    };

    if num_exp_val <= den_exp_val {
        return None;
    }

    let result_exp = num_exp_val - den_exp_val;
    let rewritten = if result_exp == 1 {
        num_base
    } else {
        let exp_expr = ctx.num(result_exp);
        ctx.add(Expr::Pow(num_base, exp_expr))
    };

    let desc = format!(
        "Cancel: P^{} / P^{} → P^{}",
        num_exp_val, den_exp_val, result_exp
    );

    Some(PowerQuotientPreserveFormPlan { rewritten, desc })
}

/// Try all didactic fraction-cancellation plans in canonical priority order.
///
/// Priority:
/// 1) expand `(a+b)^2` denominator for visible `P/P` cancellation
/// 2) difference of squares
/// 3) perfect-square-minus recognition
/// 4) sum/difference of cubes
/// 5) power quotient preservation
pub fn try_plan_fraction_didactic_cancel(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
) -> Option<FractionDidacticCancelPlan> {
    if let Some(plan) = try_plan_expand_binomial_square_in_den_for_cancel(ctx, num, den) {
        return Some(FractionDidacticCancelPlan {
            rewritten: plan.rewritten,
            local_after: plan.expanded_denominator,
            desc: "Expand: (a+b)² → a² + 2ab + b²".to_string(),
        });
    }

    if let Some(plan) = try_plan_difference_of_squares_in_num(ctx, num, den) {
        return Some(FractionDidacticCancelPlan {
            rewritten: plan.rewritten,
            local_after: plan.factored_numerator,
            desc: "Factor and cancel: a² - b² = (a-b)(a+b)".to_string(),
        });
    }

    if let Some(plan) = try_plan_perfect_square_minus_in_num(ctx, num, den) {
        return Some(FractionDidacticCancelPlan {
            rewritten: plan.rewritten,
            local_after: plan.factored_numerator,
            desc: "Recognize: a² - 2ab + b² = (a-b)²".to_string(),
        });
    }

    if let Some(plan) = try_plan_sum_diff_of_cubes_in_num(ctx, num, den) {
        return Some(FractionDidacticCancelPlan {
            rewritten: plan.rewritten,
            local_after: plan.factored_numerator,
            desc: plan.desc.to_string(),
        });
    }

    if let Some(plan) = try_plan_power_quotient_preserve_form(ctx, num, den) {
        return Some(FractionDidacticCancelPlan {
            rewritten: plan.rewritten,
            local_after: plan.rewritten,
            desc: plan.desc,
        });
    }

    None
}

/// If `n` is a positive integer perfect square, return `sqrt(n)`.
fn try_integer_sqrt(n: &BigRational) -> Option<BigRational> {
    use num_bigint::BigInt;
    use num_traits::Zero;

    if !n.is_integer() {
        return None;
    }
    let val = n.to_integer();
    if val <= BigInt::zero() {
        return None;
    }

    let root = val.sqrt();
    if &root * &root == val {
        Some(BigRational::from_integer(root))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{
        try_plan_difference_of_squares_in_num, try_plan_expand_binomial_square_in_den_for_cancel,
        try_plan_fraction_didactic_cancel, try_plan_perfect_square_minus_in_num,
        try_plan_power_quotient_preserve_form, try_plan_sum_diff_of_cubes_in_num,
    };
    use cas_ast::ordering::compare_expr;
    use cas_ast::{Context, Expr};
    use std::cmp::Ordering;

    #[test]
    fn plans_expand_binomial_square_denominator() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let two = ctx.num(2);
        let x_sq = ctx.add(Expr::Pow(x, two));
        let two2 = ctx.num(2);
        let xy = ctx.add(Expr::Mul(x, y));
        let two_xy = ctx.add(Expr::Mul(two2, xy));
        let two3 = ctx.num(2);
        let y_sq = ctx.add(Expr::Pow(y, two3));
        let rhs = ctx.add(Expr::Add(two_xy, y_sq));
        let num = ctx.add(Expr::Add(x_sq, rhs));

        let base = ctx.add(Expr::Add(x, y));
        let two4 = ctx.num(2);
        let den = ctx.add(Expr::Pow(base, two4));

        let plan = try_plan_expand_binomial_square_in_den_for_cancel(&mut ctx, num, den)
            .expect("plan should exist");
        let expected = ctx.add(Expr::Div(num, plan.expanded_denominator));
        assert_eq!(
            compare_expr(&ctx, plan.rewritten, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn plans_difference_of_squares_cancel() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let two = ctx.num(2);
        let x_sq = ctx.add(Expr::Pow(x, two));
        let two2 = ctx.num(2);
        let y_sq = ctx.add(Expr::Pow(y, two2));
        let num = ctx.add(Expr::Sub(x_sq, y_sq));
        let den = ctx.add(Expr::Add(x, y));

        let plan =
            try_plan_difference_of_squares_in_num(&mut ctx, num, den).expect("plan should exist");
        let expected = ctx.add(Expr::Sub(x, y));
        assert_eq!(
            compare_expr(&ctx, plan.rewritten, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn plans_perfect_square_minus_in_num() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let two = ctx.num(2);
        let x_sq = ctx.add(Expr::Pow(x, two));
        let two2 = ctx.num(2);
        let y_sq = ctx.add(Expr::Pow(y, two2));
        let two3 = ctx.num(2);
        let xy = ctx.add(Expr::Mul(x, y));
        let two_xy = ctx.add(Expr::Mul(two3, xy));
        let neg_two_xy = ctx.add(Expr::Neg(two_xy));
        let rhs = ctx.add(Expr::Add(neg_two_xy, y_sq));
        let num = ctx.add(Expr::Add(x_sq, rhs));
        let den = ctx.add(Expr::Sub(x, y));

        let plan =
            try_plan_perfect_square_minus_in_num(&mut ctx, num, den).expect("plan should exist");
        let two4 = ctx.num(2);
        let x_minus_y = ctx.add(Expr::Sub(x, y));
        let num_expected = ctx.add(Expr::Pow(x_minus_y, two4));
        let expected = ctx.add(Expr::Div(num_expected, den));
        assert_eq!(
            compare_expr(&ctx, plan.rewritten, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn plans_sum_diff_of_cubes_in_num() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let three = ctx.num(3);
        let x_cubed = ctx.add(Expr::Pow(x, three));
        let three2 = ctx.num(3);
        let y_cubed = ctx.add(Expr::Pow(y, three2));
        let num = ctx.add(Expr::Sub(x_cubed, y_cubed));
        let den = ctx.add(Expr::Sub(x, y));

        let plan =
            try_plan_sum_diff_of_cubes_in_num(&mut ctx, num, den).expect("plan should exist");
        let expected_desc = "Factor: a³ - b³ = (a-b)(a² + ab + b²)";
        assert_eq!(plan.desc, expected_desc);

        let expected = ctx.add(Expr::Div(plan.factored_numerator, den));
        assert_eq!(
            compare_expr(&ctx, plan.rewritten, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn plans_power_quotient_preserve_form() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_minus_one = ctx.add(Expr::Sub(x, one));
        let four = ctx.num(4);
        let two = ctx.num(2);
        let num = ctx.add(Expr::Pow(x_minus_one, four));
        let den = ctx.add(Expr::Pow(x_minus_one, two));

        let plan =
            try_plan_power_quotient_preserve_form(&mut ctx, num, den).expect("plan should exist");
        let expected = ctx.add(Expr::Pow(x_minus_one, two));
        assert_eq!(
            compare_expr(&ctx, plan.rewritten, expected),
            Ordering::Equal
        );
        assert_eq!(plan.desc, "Cancel: P^4 / P^2 → P^2");
    }

    #[test]
    fn dispatcher_returns_first_matching_plan() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let two = ctx.num(2);
        let x_sq = ctx.add(Expr::Pow(x, two));
        let two2 = ctx.num(2);
        let xy = ctx.add(Expr::Mul(x, y));
        let two_xy = ctx.add(Expr::Mul(two2, xy));
        let two3 = ctx.num(2);
        let y_sq = ctx.add(Expr::Pow(y, two3));
        let rhs = ctx.add(Expr::Add(two_xy, y_sq));
        let num = ctx.add(Expr::Add(x_sq, rhs));

        let base = ctx.add(Expr::Add(x, y));
        let two4 = ctx.num(2);
        let den = ctx.add(Expr::Pow(base, two4));

        let plan = try_plan_fraction_didactic_cancel(&mut ctx, num, den).expect("plan");
        assert_eq!(plan.desc, "Expand: (a+b)² → a² + 2ab + b²");
    }
}
