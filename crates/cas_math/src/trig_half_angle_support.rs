use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HalfAngleSquareRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CotHalfAngleDifferenceRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

/// Check if `arg` represents `u/2` and return `u`.
/// Supports `Mul(1/2, u)`, `Mul(u, 1/2)` and `Div(u, 2)`.
pub fn is_half_angle(ctx: &Context, arg: ExprId) -> Option<ExprId> {
    match ctx.get(arg) {
        Expr::Mul(l, r) => {
            let half = num_rational::BigRational::new(1.into(), 2.into());
            if let Expr::Number(n) = ctx.get(*l) {
                if *n == half {
                    return Some(*r);
                }
            }
            if let Expr::Number(n) = ctx.get(*r) {
                if *n == half {
                    return Some(*l);
                }
            }
            None
        }
        Expr::Div(num, den) => {
            if let Expr::Number(d) = ctx.get(*den) {
                if *d == num_rational::BigRational::from_integer(2.into()) {
                    return Some(*num);
                }
            }
            None
        }
        _ => None,
    }
}

/// Check if `expr` is `tan(u/2)` and return `u`.
pub fn extract_tan_half_angle(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Tan)) && args.len() == 1 {
            return is_half_angle(ctx, args[0]);
        }
    }
    None
}

/// If `expr` is `sin(u/2)` or `cos(u/2)`, returns `(u, is_sin)`.
pub fn extract_trig_half_angle(ctx: &Context, expr: ExprId) -> Option<(ExprId, bool)> {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if args.len() == 1 {
            let builtin = ctx.builtin_of(*fn_id);
            let is_sin = matches!(builtin, Some(BuiltinFn::Sin));
            let is_cos = matches!(builtin, Some(BuiltinFn::Cos));
            if is_sin || is_cos {
                if let Some(full_angle) = is_half_angle(ctx, args[0]) {
                    return Some((full_angle, is_sin));
                }
            }
        }
    }
    None
}

/// Extract coefficient and cot argument from a term.
/// Returns `(coefficient_opt, cot_arg, is_positive)` where `coefficient_opt=None` means `1`.
pub fn extract_cot_term(ctx: &Context, term: ExprId) -> Option<(Option<ExprId>, ExprId, bool)> {
    let term_data = ctx.get(term);

    let (inner_term, is_positive) = match term_data {
        Expr::Neg(inner) => (*inner, false),
        _ => (term, true),
    };

    let inner_data = ctx.get(inner_term);

    if let Expr::Function(fn_id, args) = inner_data {
        if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cot)) && args.len() == 1 {
            return Some((None, args[0], is_positive));
        }
    }

    if let Expr::Mul(l, r) = inner_data {
        if let Expr::Function(fn_id, args) = ctx.get(*r) {
            if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cot)) && args.len() == 1 {
                return Some((Some(*l), args[0], is_positive));
            }
        }
        if let Expr::Function(fn_id, args) = ctx.get(*l) {
            if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cot)) && args.len() == 1 {
                return Some((Some(*r), args[0], is_positive));
            }
        }
    }

    None
}

/// Rewrite cotangent half-angle differences:
/// - `cot(u/2) - cot(u) -> 1/sin(u)`
/// - `-cot(u/2) + cot(u) -> -1/sin(u)`
/// - Supports coefficient form `k*cot(u/2) - k*cot(u)`.
/// - Works on additive chains by replacing just the matching pair.
pub fn try_rewrite_cot_half_angle_difference_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<CotHalfAngleDifferenceRewrite> {
    let terms: Vec<ExprId> = if crate::expr_destructure::as_add(ctx, expr).is_some() {
        crate::expr_nary::add_leaves(ctx, expr).to_vec()
    } else if let Some((l, r)) = crate::expr_destructure::as_sub(ctx, expr) {
        vec![l, r]
    } else {
        return None;
    };

    if terms.len() < 2 {
        return None;
    }

    let is_explicit_sub = matches!(ctx.get(expr), Expr::Sub(_, _));

    struct CotTerm {
        index: usize,
        coeff: Option<ExprId>,
        arg: ExprId,
        is_positive: bool,
    }

    let mut cot_terms = Vec::new();

    if is_explicit_sub {
        if let Some((c1, arg1, _)) = extract_cot_term(ctx, terms[0]) {
            cot_terms.push(CotTerm {
                index: 0,
                coeff: c1,
                arg: arg1,
                is_positive: true,
            });
        }
        if let Some((c2, arg2, sign2)) = extract_cot_term(ctx, terms[1]) {
            cot_terms.push(CotTerm {
                index: 1,
                coeff: c2,
                arg: arg2,
                is_positive: !sign2,
            });
        }
    } else {
        for (i, &term) in terms.iter().enumerate() {
            if let Some((c, arg, is_pos)) = extract_cot_term(ctx, term) {
                cot_terms.push(CotTerm {
                    index: i,
                    coeff: c,
                    arg,
                    is_positive: is_pos,
                });
            }
        }
    }

    for i in 0..cot_terms.len() {
        for j in 0..cot_terms.len() {
            if i == j {
                continue;
            }

            let t_half = &cot_terms[i];
            let t_full = &cot_terms[j];

            let Some(full_angle) = is_half_angle(ctx, t_half.arg) else {
                continue;
            };
            if compare_expr(ctx, full_angle, t_full.arg) != Ordering::Equal {
                continue;
            }

            let coeffs_match = match (&t_half.coeff, &t_full.coeff) {
                (None, None) => true,
                (Some(c1), Some(c2)) => compare_expr(ctx, *c1, *c2) == Ordering::Equal,
                _ => false,
            };
            if !coeffs_match {
                continue;
            }

            let one = ctx.num(1);
            let sin_u = ctx.call_builtin(BuiltinFn::Sin, vec![t_full.arg]);
            let base = ctx.add(Expr::Div(one, sin_u));

            let (result, desc) = if t_half.is_positive && !t_full.is_positive {
                (base, "cot(u/2) - cot(u) = 1/sin(u)")
            } else if !t_half.is_positive && t_full.is_positive {
                (ctx.add(Expr::Neg(base)), "-cot(u/2) + cot(u) = -1/sin(u)")
            } else {
                continue;
            };

            let final_result = if let Some(c) = t_half.coeff {
                crate::expr_rewrite::smart_mul(ctx, c, result)
            } else {
                result
            };

            if is_explicit_sub && terms.len() == 2 {
                return Some(CotHalfAngleDifferenceRewrite {
                    rewritten: final_result,
                    desc,
                });
            }

            let mut new_terms: Vec<ExprId> = Vec::new();
            for (k, &term) in terms.iter().enumerate() {
                if k != t_half.index && k != t_full.index {
                    new_terms.push(term);
                }
            }
            new_terms.push(final_result);

            let mut new_expr = new_terms[0];
            for &term in new_terms.iter().skip(1) {
                new_expr = ctx.add(Expr::Add(new_expr, term));
            }

            return Some(CotHalfAngleDifferenceRewrite {
                rewritten: new_expr,
                desc,
            });
        }
    }

    None
}

/// Rewrite hyperbolic half-angle squares:
/// - `cosh(x/2)^2 -> (cosh(x)+1)/2`
/// - `sinh(x/2)^2 -> (cosh(x)-1)/2`
pub fn try_rewrite_hyperbolic_half_angle_squares_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HalfAngleSquareRewrite> {
    let (base, exp) = match ctx.get(expr) {
        Expr::Pow(base, exp) => (*base, *exp),
        _ => return None,
    };

    let Expr::Number(n) = ctx.get(exp) else {
        return None;
    };
    if *n != num_rational::BigRational::from_integer(2.into()) {
        return None;
    }

    let (fn_id, args) = if let Expr::Function(fn_id, args) = ctx.get(base) {
        (*fn_id, args)
    } else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let builtin = ctx.builtin_of(fn_id);
    let is_cosh = matches!(builtin, Some(BuiltinFn::Cosh));
    let is_sinh = matches!(builtin, Some(BuiltinFn::Sinh));
    if !is_cosh && !is_sinh {
        return None;
    }

    let x = is_half_angle(ctx, args[0])?;
    let cosh_x = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
    let one = ctx.num(1);
    let half = ctx.add(Expr::Number(num_rational::BigRational::new(
        1.into(),
        2.into(),
    )));

    if is_cosh {
        let sum = ctx.add(Expr::Add(cosh_x, one));
        let rewritten = ctx.add(Expr::Mul(half, sum));
        Some(HalfAngleSquareRewrite {
            rewritten,
            desc: "cosh²(x/2) = (cosh(x)+1)/2",
        })
    } else {
        let diff = ctx.add(Expr::Sub(cosh_x, one));
        let rewritten = ctx.add(Expr::Mul(half, diff));
        Some(HalfAngleSquareRewrite {
            rewritten,
            desc: "sinh²(x/2) = (cosh(x)-1)/2",
        })
    }
}

/// Rewrite trig half-angle squares:
/// - `sin(x/2)^2 -> (1 - cos(x))/2`
/// - `cos(x/2)^2 -> (1 + cos(x))/2`
/// - `sin(x/2)*sin(x/2) -> (1 - cos(x))/2`
/// - `cos(x/2)*cos(x/2) -> (1 + cos(x))/2`
pub fn try_rewrite_trig_half_angle_squares_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HalfAngleSquareRewrite> {
    let matched = if let Expr::Pow(base, exp) = ctx.get(expr) {
        let Expr::Number(n) = ctx.get(*exp) else {
            return None;
        };
        if *n != num_rational::BigRational::from_integer(2.into()) {
            return None;
        }
        extract_trig_half_angle(ctx, *base)
    } else if let Expr::Mul(l, r) = ctx.get(expr) {
        let (l, r) = (*l, *r);
        if let Some((angle_l, is_sin_l)) = extract_trig_half_angle(ctx, l) {
            if let Some((angle_r, is_sin_r)) = extract_trig_half_angle(ctx, r) {
                if is_sin_l == is_sin_r && compare_expr(ctx, angle_l, angle_r) == Ordering::Equal {
                    Some((angle_l, is_sin_l))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    }?;

    let (full_angle, is_sin) = matched;
    let cos_x = ctx.call_builtin(BuiltinFn::Cos, vec![full_angle]);
    let one = ctx.num(1);
    let half = ctx.add(Expr::Number(num_rational::BigRational::new(
        1.into(),
        2.into(),
    )));

    if is_sin {
        let diff = ctx.add(Expr::Sub(one, cos_x));
        let rewritten = ctx.add(Expr::Mul(half, diff));
        Some(HalfAngleSquareRewrite {
            rewritten,
            desc: "sin²(x/2) = (1 - cos(x))/2",
        })
    } else {
        let sum = ctx.add(Expr::Add(one, cos_x));
        let rewritten = ctx.add(Expr::Mul(half, sum));
        Some(HalfAngleSquareRewrite {
            rewritten,
            desc: "cos²(x/2) = (1 + cos(x))/2",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn is_half_angle_recognizes_mul_and_div_forms() {
        let mut ctx = Context::new();
        let div = parse("x/2", &mut ctx).expect("x/2");
        let x = parse("x", &mut ctx).expect("x");
        let half = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let mul = ctx.add(Expr::Mul(half, x));

        assert_eq!(
            cas_ast::ordering::compare_expr(
                &ctx,
                is_half_angle(&ctx, div).expect("div half-angle"),
                x
            ),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            cas_ast::ordering::compare_expr(
                &ctx,
                is_half_angle(&ctx, mul).expect("mul half-angle"),
                x
            ),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn extract_tan_half_angle_matches_tan_of_half_arg() {
        let mut ctx = Context::new();
        let expr = parse("tan(x/2)", &mut ctx).expect("tan(x/2)");
        let x = parse("x", &mut ctx).expect("x");

        assert_eq!(
            cas_ast::ordering::compare_expr(
                &ctx,
                extract_tan_half_angle(&ctx, expr).expect("tan half-angle"),
                x
            ),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn extract_trig_half_angle_distinguishes_sin_and_cos() {
        let mut ctx = Context::new();
        let sin_expr = parse("sin(x/2)", &mut ctx).expect("sin(x/2)");
        let cos_expr = parse("cos(x/2)", &mut ctx).expect("cos(x/2)");

        let sin = extract_trig_half_angle(&ctx, sin_expr).expect("sin half-angle");
        let cos = extract_trig_half_angle(&ctx, cos_expr).expect("cos half-angle");

        assert!(sin.1);
        assert!(!cos.1);
    }

    #[test]
    fn extract_cot_term_handles_plain_negated_and_scaled_terms() {
        let mut ctx = Context::new();
        let plain = parse("cot(x)", &mut ctx).expect("cot(x)");
        let neg = parse("-cot(y)", &mut ctx).expect("-cot(y)");
        let scaled = parse("3*cot(z)", &mut ctx).expect("3*cot(z)");

        assert!(extract_cot_term(&ctx, plain).is_some());
        assert!(extract_cot_term(&ctx, neg).is_some_and(|(_, _, positive)| !positive));
        assert!(extract_cot_term(&ctx, scaled).is_some_and(|(c, _, _)| c.is_some()));
    }

    #[test]
    fn rewrites_cot_half_angle_difference_basic_forms() {
        let mut ctx = Context::new();
        let expr1 = parse("cot(x/2) - cot(x)", &mut ctx).expect("expr1");
        let expr2 = parse("-cot(x/2) + cot(x)", &mut ctx).expect("expr2");
        let expected1 = parse("1/sin(x)", &mut ctx).expect("expected1");
        let expected2 = parse("-(1/sin(x))", &mut ctx).expect("expected2");

        let rw1 = try_rewrite_cot_half_angle_difference_expr(&mut ctx, expr1).expect("rw1");
        let rw2 = try_rewrite_cot_half_angle_difference_expr(&mut ctx, expr2).expect("rw2");

        assert_eq!(
            compare_expr(&ctx, rw1.rewritten, expected1),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            compare_expr(&ctx, rw2.rewritten, expected2),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn rewrites_cot_half_angle_difference_with_shared_coefficient() {
        let mut ctx = Context::new();
        let expr = parse("3*cot(x/2) - 3*cot(x)", &mut ctx).expect("expr");
        let expected = parse("3*(1/sin(x))", &mut ctx).expect("expected");
        let rw = try_rewrite_cot_half_angle_difference_expr(&mut ctx, expr).expect("rw");
        assert_eq!(
            compare_expr(&ctx, rw.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn rewrites_trig_half_angle_square_pow_form() {
        let mut ctx = Context::new();
        let expr = parse("sin(x/2)^2", &mut ctx).expect("expr");
        let rewrite = try_rewrite_trig_half_angle_squares_expr(&mut ctx, expr).expect("rewrite");
        let rewritten_str = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert_eq!(rewritten_str, "1/2 * (1 - cos(x))");
    }

    #[test]
    fn rewrites_trig_half_angle_square_mul_form() {
        let mut ctx = Context::new();
        let expr = parse("cos(x/2)*cos(x/2)", &mut ctx).expect("expr");
        let rewrite = try_rewrite_trig_half_angle_squares_expr(&mut ctx, expr).expect("rewrite");
        let rewritten_str = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert_eq!(rewritten_str, "1/2 * (cos(x) + 1)");
    }

    #[test]
    fn rewrites_hyperbolic_half_angle_square() {
        let mut ctx = Context::new();
        let expr = parse("sinh(x/2)^2", &mut ctx).expect("expr");
        let rewrite =
            try_rewrite_hyperbolic_half_angle_squares_expr(&mut ctx, expr).expect("rewrite");
        let rewritten_str = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert_eq!(rewritten_str, "1/2 * (cosh(x) - 1)");
    }
}
