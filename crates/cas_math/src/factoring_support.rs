//! Support rewrites for algebraic factoring rules.

use crate::expr_nary::{add_leaves, mul_leaves};
use crate::expr_predicates::contains_variable;
use crate::expr_relations::{
    conjugate_add_sub_pair as is_conjugate_pair,
    conjugate_nary_add_sub_pair as is_nary_conjugate_pair, is_structurally_zero, poly_equal,
};
use crate::expr_rewrite::smart_mul;
use crate::numeric::gcd_rational;
use cas_ast::ordering::compare_expr;
use cas_ast::{count_nodes, Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DifferenceOfSquaresProductRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FactorFunctionRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactorCommonIntegerFromAddRewrite {
    pub rewritten: ExprId,
    pub gcd_int: BigInt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SumThreeCubesZeroRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FactorDifferenceSquaresNaryRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutomaticFactorRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

/// Rewrite conjugate products:
/// - `(a-b)(a+b) -> a^2-b^2`
/// - `(U+V)(U-V) -> U^2-V^2`
/// - scans n-ary products to find one conjugate pair.
pub fn try_rewrite_difference_of_squares_product_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<DifferenceOfSquaresProductRewrite> {
    fn nary_candidate_matches_factor_pair(
        ctx: &mut Context,
        left: ExprId,
        right: ExprId,
        u: ExprId,
        v: ExprId,
    ) -> bool {
        let u_plus_v = ctx.add(Expr::Add(u, v));
        let u_minus_v = ctx.add(Expr::Sub(u, v));

        // Require each original factor to be polynomially equivalent to one of
        // the reconstructed conjugates, in opposite sign arrangement.
        let plus_minus = poly_equal(ctx, left, u_plus_v) && poly_equal(ctx, right, u_minus_v);
        let minus_plus = poly_equal(ctx, left, u_minus_v) && poly_equal(ctx, right, u_plus_v);
        plus_minus || minus_plus
    }

    let (left, right) = match ctx.get(expr) {
        Expr::Mul(l, r) => (*l, *r),
        _ => return None,
    };

    if let Some((a, b)) = is_conjugate_pair(ctx, left, right) {
        let two = ctx.num(2);
        let a_squared = ctx.add(Expr::Pow(a, two));
        let b_squared = ctx.add(Expr::Pow(b, two));
        let rewritten = ctx.add(Expr::Sub(a_squared, b_squared));
        return Some(DifferenceOfSquaresProductRewrite {
            rewritten,
            desc: "(a-b)(a+b) = a² - b²",
        });
    }

    if let Some((u, v)) = is_nary_conjugate_pair(ctx, left, right) {
        if nary_candidate_matches_factor_pair(ctx, left, right, u, v) {
            let two = ctx.num(2);
            let u_squared = ctx.add(Expr::Pow(u, two));
            let v_squared = ctx.add(Expr::Pow(v, two));
            let rewritten = ctx.add(Expr::Sub(u_squared, v_squared));
            return Some(DifferenceOfSquaresProductRewrite {
                rewritten,
                desc: "(U+V)(U-V) = U² - V² (conjugate product)",
            });
        }
    }

    let factors = mul_leaves(ctx, expr);
    if factors.len() < 3 {
        return None;
    }

    for i in 0..factors.len() {
        for j in (i + 1)..factors.len() {
            let fi = factors[i];
            let fj = factors[j];

            let conjugate = if let Some(pair) = is_conjugate_pair(ctx, fi, fj) {
                Some((pair, false))
            } else {
                is_nary_conjugate_pair(ctx, fi, fj).map(|pair| (pair, true))
            };

            if let Some(((a, b), from_nary)) = conjugate {
                if from_nary && !nary_candidate_matches_factor_pair(ctx, fi, fj, a, b) {
                    continue;
                }

                let two = ctx.num(2);
                let a_squared = ctx.add(Expr::Pow(a, two));
                let b_squared = ctx.add(Expr::Pow(b, two));
                let dos = ctx.add(Expr::Sub(a_squared, b_squared));

                let mut rewritten = dos;
                for (k, &factor) in factors.iter().enumerate() {
                    if k != i && k != j {
                        rewritten = ctx.add(Expr::Mul(rewritten, factor));
                    }
                }
                return Some(DifferenceOfSquaresProductRewrite {
                    rewritten,
                    desc: "(a-b)(a+b)·… = (a²-b²)·… (n-ary scan)",
                });
            }
        }
    }

    None
}

/// Rewrite `factor(expr)` function calls into held factored forms.
pub fn try_rewrite_factor_function_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<FactorFunctionRewrite> {
    let (fn_id, args) = match ctx.get(expr) {
        Expr::Function(fn_id, args) => (*fn_id, args),
        _ => return None,
    };

    if ctx.sym_name(fn_id) != "factor" || args.len() != 1 {
        return None;
    }

    let arg = args[0];
    let factored = crate::factor::factor(ctx, arg);
    if factored == arg {
        return None;
    }

    let rewritten = cas_ast::hold::wrap_hold(ctx, factored);
    Some(FactorFunctionRewrite {
        rewritten,
        desc: "Factorization",
    })
}

fn get_integer_coefficient(ctx: &Context, term: ExprId) -> Option<BigRational> {
    match ctx.get(term) {
        Expr::Number(n) if n.is_integer() => Some(n.clone()),
        Expr::Mul(a, b) => {
            if let Expr::Number(n) = ctx.get(*a) {
                if n.is_integer() {
                    return Some(n.clone());
                }
            }
            if let Expr::Number(n) = ctx.get(*b) {
                if n.is_integer() {
                    return Some(n.clone());
                }
            }
            None
        }
        Expr::Neg(inner) => get_integer_coefficient(ctx, *inner).map(|value| -value),
        _ => None,
    }
}

fn divide_term_by_rational(ctx: &mut Context, term: ExprId, divisor: &BigRational) -> ExprId {
    match ctx.get(term) {
        Expr::Number(n) => {
            let divided = n / divisor;
            ctx.add(Expr::Number(divided))
        }
        Expr::Mul(a, b) => {
            let (a, b) = (*a, *b);
            if let Expr::Number(n) = ctx.get(a) {
                let divided = n / divisor;
                if divided.is_one() {
                    return b;
                }
                let num = ctx.add(Expr::Number(divided));
                return ctx.add_raw(Expr::Mul(num, b));
            }
            if let Expr::Number(n) = ctx.get(b) {
                let divided = n / divisor;
                if divided.is_one() {
                    return a;
                }
                let num = ctx.add(Expr::Number(divided));
                return ctx.add_raw(Expr::Mul(a, num));
            }
            term
        }
        Expr::Neg(inner) => {
            let divided = divide_term_by_rational(ctx, *inner, divisor);
            ctx.add(Expr::Neg(divided))
        }
        _ => term,
    }
}

/// Rewrite binary additions by factoring common integer GCD:
/// - `2*sqrt(2) - 2 -> 2*(sqrt(2)-1)`
pub fn try_rewrite_factor_common_integer_from_add_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<FactorCommonIntegerFromAddRewrite> {
    let (left, right) = match ctx.get(expr) {
        Expr::Add(l, r) => (*l, *r),
        _ => return None,
    };

    let coef_left = get_integer_coefficient(ctx, left)?;
    let coef_right = get_integer_coefficient(ctx, right)?;

    if contains_variable(ctx, left) || contains_variable(ctx, right) {
        return None;
    }

    let gcd = gcd_rational(coef_left.abs(), coef_right.abs());
    if gcd <= BigRational::one() {
        return None;
    }

    let gcd_int = gcd.to_integer();
    if gcd_int <= BigInt::from(1) {
        return None;
    }

    let new_left = divide_term_by_rational(ctx, left, &gcd);
    let new_right = divide_term_by_rational(ctx, right, &gcd);
    let inner = ctx.add_raw(Expr::Add(new_left, new_right));
    let gcd_expr = ctx.add(Expr::Number(gcd));
    let rewritten = ctx.add_raw(Expr::Mul(gcd_expr, inner));

    Some(FactorCommonIntegerFromAddRewrite { rewritten, gcd_int })
}

/// Rewrite `x^3 + y^3 + z^3` into `3xyz` when `x+y+z = 0`.
pub fn try_rewrite_sum_three_cubes_zero_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<SumThreeCubesZeroRewrite> {
    let terms = add_leaves(ctx, expr);
    if terms.len() != 3 {
        return None;
    }

    let mut bases = Vec::with_capacity(3);
    for &term in &terms {
        let (base, is_negated_cube) = match ctx.get(term) {
            Expr::Pow(base, exp_id) => {
                if let Expr::Number(n) = ctx.get(*exp_id) {
                    if n.is_integer() && n.to_integer() == BigInt::from(3) {
                        (*base, false)
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }
            Expr::Neg(inner) => {
                let inner = *inner;
                if let Expr::Pow(base, exp_id) = ctx.get(inner) {
                    if let Expr::Number(n) = ctx.get(*exp_id) {
                        if n.is_integer() && n.to_integer() == BigInt::from(3) {
                            (*base, true)
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        if is_negated_cube {
            bases.push(ctx.add(Expr::Neg(base)));
        } else {
            bases.push(base);
        }
    }

    let xy = ctx.add(Expr::Add(bases[0], bases[1]));
    let sum_bases = ctx.add(Expr::Add(xy, bases[2]));
    if !is_structurally_zero(ctx, sum_bases) {
        return None;
    }

    let three = ctx.num(3);
    let xy_mul = smart_mul(ctx, bases[0], bases[1]);
    let xyz_mul = smart_mul(ctx, xy_mul, bases[2]);
    let simplified = smart_mul(ctx, three, xyz_mul);
    let rewritten = cas_ast::hold::wrap_hold(ctx, simplified);

    Some(SumThreeCubesZeroRewrite {
        rewritten,
        desc: "x³ + y³ + z³ = 3xyz (when x + y + z = 0)",
    })
}

/// Try pairwise difference-of-squares factoring in n-ary sums.
pub fn try_rewrite_factor_difference_squares_nary_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<FactorDifferenceSquaresNaryRewrite> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let terms = add_leaves(ctx, expr);
    for i in 0..terms.len() {
        for j in 0..terms.len() {
            if i == j {
                continue;
            }

            let t1 = terms[i];
            let t2 = terms[j];
            let pair = ctx.add(Expr::Add(t1, t2));
            let factored = crate::factor::factor_difference_squares(ctx, pair)?;

            let old_count = count_nodes(ctx, pair);
            let new_count = count_nodes(ctx, factored);
            let is_mul = matches!(ctx.get(factored), Expr::Mul(_, _));
            let allowed = if is_mul {
                new_count < old_count
            } else {
                new_count <= old_count
            };
            if !allowed {
                continue;
            }

            let mut new_terms = Vec::new();
            new_terms.push(factored);
            for (k, &term) in terms.iter().enumerate() {
                if k != i && k != j {
                    new_terms.push(term);
                }
            }

            if new_terms.is_empty() {
                return Some(FactorDifferenceSquaresNaryRewrite {
                    rewritten: ctx.num(0),
                    desc: "Factor difference of squares (Empty)",
                });
            }

            let mut rewritten = new_terms[0];
            for term in new_terms.iter().skip(1) {
                rewritten = ctx.add(Expr::Add(rewritten, *term));
            }
            return Some(FactorDifferenceSquaresNaryRewrite {
                rewritten,
                desc: "Factor difference of squares (N-ary)",
            });
        }
    }

    None
}

/// Heuristic automatic factorization with node-count guard.
pub fn try_rewrite_automatic_factor_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<AutomaticFactorRewrite> {
    match ctx.get(expr) {
        Expr::Add(_, _) | Expr::Sub(_, _) => {}
        _ => return None,
    }

    if let Some(poly_factored) = crate::factor::factor_polynomial(ctx, expr) {
        if poly_factored != expr {
            let old_count = count_nodes(ctx, expr);
            let new_count = count_nodes(ctx, poly_factored);
            if new_count < old_count && compare_expr(ctx, poly_factored, expr) != Ordering::Equal {
                return Some(AutomaticFactorRewrite {
                    rewritten: poly_factored,
                    desc: "Automatic Factorization (Reduced Size)",
                });
            }
        }
    }

    if let Some(diff_squares) = crate::factor::factor_difference_squares(ctx, expr) {
        if diff_squares != expr {
            let old_count = count_nodes(ctx, expr);
            let new_count = count_nodes(ctx, diff_squares);
            if new_count < old_count {
                return Some(AutomaticFactorRewrite {
                    rewritten: diff_squares,
                    desc: "Automatic Factorization (Diff Squares)",
                });
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{
        try_rewrite_automatic_factor_expr, try_rewrite_difference_of_squares_product_expr,
        try_rewrite_factor_common_integer_from_add_expr,
        try_rewrite_factor_difference_squares_nary_expr, try_rewrite_factor_function_expr,
        try_rewrite_sum_three_cubes_zero_expr,
    };
    use cas_ast::{BuiltinFn, Context, Expr};
    use cas_parser::parse;
    #[test]
    fn difference_of_squares_support_matches_basic_conjugate_product() {
        let mut ctx = Context::new();
        let expr = parse("(x+1)*(x-1)", &mut ctx).expect("parse");
        let rewrite =
            try_rewrite_difference_of_squares_product_expr(&mut ctx, expr).expect("rewrite");
        assert!(matches!(ctx.get(rewrite.rewritten), Expr::Sub(_, _)));
    }

    #[test]
    fn difference_of_squares_support_rejects_false_positive_nary_pair() {
        let mut ctx = Context::new();
        // Not a true conjugate pair around the same center:
        // ((u^2+1)-1) * ((u^2+1)+1) == u^2 * (u^2+2), not u^4-1.
        let expr = parse("((u^2+1)-1)*((u^2+1)+1)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_difference_of_squares_product_expr(&mut ctx, expr);
        assert!(rewrite.is_none(), "must not contract to U^2-V^2");
    }

    #[test]
    fn factor_function_support_rewrites_factor_call() {
        let mut ctx = Context::new();
        let expr = parse("factor(x^2-1)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_factor_function_expr(&mut ctx, expr).expect("rewrite");
        assert!(cas_ast::hold::is_hold(&ctx, rewrite.rewritten));
    }

    #[test]
    fn factor_common_integer_support_extracts_gcd() {
        let mut ctx = Context::new();
        let expr = parse("2*sqrt(2)+4", &mut ctx).expect("parse");
        let rewrite =
            try_rewrite_factor_common_integer_from_add_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.gcd_int, 2.into());
        assert!(matches!(ctx.get(rewrite.rewritten), Expr::Mul(_, _)));
    }

    #[test]
    fn sum_three_cubes_zero_support_matches_cyclic_identity() {
        let mut ctx = Context::new();
        let expr = parse("(a-b)^3 + (b-c)^3 + (c-a)^3", &mut ctx).expect("parse");
        let rewrite = try_rewrite_sum_three_cubes_zero_expr(&mut ctx, expr).expect("rewrite");
        assert!(cas_ast::hold::is_hold(&ctx, rewrite.rewritten));
    }

    #[test]
    fn factor_difference_squares_nary_support_rewrites_matching_pair() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sin_x = ctx.call_builtin(BuiltinFn::Sin, vec![x]);
        let cos_x = ctx.call_builtin(BuiltinFn::Cos, vec![x]);
        let two = ctx.num(2);
        let sin_sq = ctx.add(Expr::Pow(sin_x, two));
        let cos_sq = ctx.add(Expr::Pow(cos_x, two));
        let neg_cos_sq = ctx.add(Expr::Neg(cos_sq));
        let expr = ctx.add_raw(Expr::Add(sin_sq, neg_cos_sq));
        let rewrite = try_rewrite_factor_difference_squares_nary_expr(&mut ctx, expr);
        assert!(rewrite.is_none());
    }

    #[test]
    fn automatic_factor_support_rewrites_simple_binomial_square_difference() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sin_x = ctx.call_builtin(BuiltinFn::Sin, vec![x]);
        let cos_x = ctx.call_builtin(BuiltinFn::Cos, vec![x]);
        let two = ctx.num(2);
        let sin_sq = ctx.add(Expr::Pow(sin_x, two));
        let cos_sq = ctx.add(Expr::Pow(cos_x, two));
        let neg_cos_sq = ctx.add(Expr::Neg(cos_sq));
        let expr = ctx.add_raw(Expr::Add(sin_sq, neg_cos_sq));
        let rewrite = try_rewrite_automatic_factor_expr(&mut ctx, expr);
        assert!(rewrite.is_none());
    }
}
