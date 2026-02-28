use cas_ast::{Context, Expr, ExprId};
use num_traits::{Signed, Zero};
use std::cmp::Ordering;

/// Extract a semantic subtraction pair `(a, b)` from subtraction-like syntax.
///
/// Supported forms:
/// - `Sub(a, b)`
/// - `Add(a, Neg(b))`
/// - `Add(Neg(b), a)`
/// - `Add(a, Number(-k))` (rewritten as `(a, k)`)
/// - `Add(Number(-k), a)` (rewritten as `(a, k)`)
pub fn extract_sub_like_pair(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr) {
        Expr::Sub(a, b) => Some((*a, *b)),
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            if let Expr::Neg(x) = ctx.get(r) {
                Some((l, *x))
            } else if let Expr::Neg(x) = ctx.get(l) {
                Some((r, *x))
            } else if let Expr::Number(n) = ctx.get(r) {
                if n.is_negative() {
                    let pos_k = ctx.add(Expr::Number(-n.clone()));
                    Some((l, pos_k))
                } else {
                    None
                }
            } else if let Expr::Number(n) = ctx.get(l) {
                if n.is_negative() {
                    let pos_k = ctx.add(Expr::Number(-n.clone()));
                    Some((r, pos_k))
                } else {
                    None
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Build a canonical subtraction-like expression: `a + (-b)`.
pub fn build_sub_like_expr(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    let neg_b = ctx.add(Expr::Neg(b));
    ctx.add(Expr::Add(a, neg_b))
}

/// Peel a semantic negation from an expression.
///
/// Returns `(core, was_negation)` where `was_negation` is true for:
/// - `Neg(t)` -> core `t`
/// - `Mul(-1, t)` or `Mul(t, -1)` -> core `t`
/// - sub-like forms with reversed canonical orientation (treated as negated)
pub fn peel_negation_sub_like(ctx: &mut Context, id: ExprId) -> (ExprId, bool) {
    match ctx.get(id) {
        Expr::Neg(inner) => (*inner, true),
        Expr::Mul(l, r) => {
            let minus_one = num_rational::BigRational::from_integer((-1).into());
            if let Expr::Number(n) = ctx.get(*l) {
                if *n == minus_one {
                    return (*r, true);
                }
            }
            if let Expr::Number(n) = ctx.get(*r) {
                if *n == minus_one {
                    return (*l, true);
                }
            }
            (id, false)
        }
        _ => {
            if let Some((a, b)) = extract_sub_like_pair(ctx, id) {
                if cas_ast::ordering::compare_expr(ctx, a, b) == Ordering::Less {
                    return (id, true);
                }
            }
            (id, false)
        }
    }
}

/// Build an "un-negated" canonical form from a negated/sub-like expression.
///
/// - sub-like `a-b` with reversed canonical orientation -> `b-a` (as `Add(b, Neg(a))`)
/// - `Neg(t)` -> `t`
/// - `Mul(-1,t)`/`Mul(t,-1)` -> `t`
/// - otherwise returns original `id`
pub fn build_unnegated_sub_like_expr(ctx: &mut Context, id: ExprId) -> ExprId {
    if let Some((a, b)) = extract_sub_like_pair(ctx, id) {
        return build_sub_like_expr(ctx, b, a);
    }

    match ctx.get(id) {
        Expr::Neg(inner) => *inner,
        Expr::Mul(l, r) => {
            let minus_one = num_rational::BigRational::from_integer((-1).into());
            if let Expr::Number(n) = ctx.get(*l) {
                if *n == minus_one {
                    return *r;
                }
            }
            if let Expr::Number(n) = ctx.get(*r) {
                if *n == minus_one {
                    return *l;
                }
            }
            id
        }
        _ => id,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NegCoeffFlipRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

/// Rewrite negative-coefficient binomial forms:
/// - `(-k) * (a-b)` -> `k * (b-a)`
/// - `(-k) * (x * (a-b))` -> `k * (x * (b-a))`
/// - `(-k) * ((a-b) * x)` -> `k * ((b-a) * x)`
pub fn try_rewrite_neg_coeff_flip_binomial_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<NegCoeffFlipRewrite> {
    use num_traits::Signed;
    let factors = crate::expr_nary::mul_factors(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    let mut neg_coeff_idx = None;
    let mut neg_coeff = None;
    for (idx, factor) in factors.iter().enumerate() {
        if let Expr::Number(n) = ctx.get(*factor) {
            if n.is_negative() {
                if neg_coeff_idx.is_some() {
                    return None;
                }
                neg_coeff_idx = Some(idx);
                neg_coeff = Some(n.clone());
            }
        }
    }
    let neg_coeff_idx = neg_coeff_idx?;
    let neg_coeff = neg_coeff?;

    let mut sub_like_idx = None;
    let mut sub_like_pair = None;
    for (idx, factor) in factors.iter().enumerate() {
        if idx == neg_coeff_idx {
            continue;
        }
        if let Some((a, b)) = extract_sub_like_pair(ctx, *factor) {
            sub_like_idx = Some(idx);
            sub_like_pair = Some((a, b));
            break;
        }
    }
    let sub_like_idx = sub_like_idx?;
    let (a, b) = sub_like_pair?;

    let pos_n = ctx.add(Expr::Number(-neg_coeff));
    let b_minus_a = build_sub_like_expr(ctx, b, a);

    let mut rewritten_factors: Vec<ExprId> = factors.into_iter().collect();
    rewritten_factors[neg_coeff_idx] = pos_n;
    rewritten_factors[sub_like_idx] = b_minus_a;

    let mut iter = rewritten_factors.into_iter();
    let mut rewritten = iter.next()?;
    for f in iter {
        rewritten = crate::build::mul2_raw(ctx, rewritten, f);
    }

    Some(NegCoeffFlipRewrite {
        rewritten,
        desc: "(-k) * (...) * (a-b) → k * (...) * (b-a)",
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizeSignsRewrite {
    pub rewritten: ExprId,
    pub desc: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormalizeBinomialOrderRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NegSubFlipRewrite {
    pub inner: ExprId,
    pub rewritten: ExprId,
    pub desc: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalizeDivRewrite {
    pub rewritten: ExprId,
    pub desc: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalizeNegationRewrite {
    pub rewritten: ExprId,
    pub desc: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CancelFractionSignsRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

/// Rewrite additive negative-constant forms to subtraction:
/// - `(-c) + x` -> `x - c`
/// - `x + (-c)` -> `x - c`
///
/// for `c > 0`.
pub fn try_rewrite_add_negative_constant_to_sub_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<NormalizeSignsRewrite> {
    let Expr::Add(l, r) = ctx.get(expr) else {
        return None;
    };
    let (l, r) = (*l, *r);

    if let Expr::Neg(inner_neg) = ctx.get(l) {
        if let Expr::Number(n) = ctx.get(*inner_neg) {
            if n.is_positive() {
                let n_clone = n.clone();
                let rewritten = ctx.add(Expr::Sub(r, *inner_neg));
                return Some(NormalizeSignsRewrite {
                    rewritten,
                    desc: format!("-{} + x -> x - {}", n_clone, n_clone),
                });
            }
        }
    }

    if let Expr::Neg(inner_neg) = ctx.get(r) {
        if let Expr::Number(n) = ctx.get(*inner_neg) {
            if n.is_positive() {
                let n_clone = n.clone();
                let rewritten = ctx.add(Expr::Sub(l, *inner_neg));
                return Some(NormalizeSignsRewrite {
                    rewritten,
                    desc: format!("x + (-{}) -> x - {}", n_clone, n_clone),
                });
            }
        }
    }

    None
}

/// Canonicalize basic negation/negative-sign forms:
/// - `a - b` -> `a + (-b)`
/// - `-(number)` -> normalized numeric literal
/// - `-(-x)` -> `x`
/// - `-(a+b)` -> `-a + -b`
/// - `-(c*x)` with numeric `c` -> `(-c)*x`
/// - `(-a)*b` / `a*(-b)` -> `-(a*b)` (or numeric-coeff form)
/// - `a/(-b)` -> `-(a/b)`
pub fn try_rewrite_canonicalize_negation_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<CanonicalizeNegationRewrite> {
    use crate::expr_destructure::{as_add, as_div, as_mul, as_neg, as_sub};

    // 1. a - b -> a + (-b)
    if let Some((lhs, rhs)) = as_sub(ctx, expr) {
        let neg_rhs = ctx.add(Expr::Neg(rhs));
        let rewritten = ctx.add(Expr::Add(lhs, neg_rhs));
        return Some(CanonicalizeNegationRewrite {
            rewritten,
            desc: "Convert Subtraction to Addition (a - b -> a + (-b))".to_string(),
        });
    }

    // 2. Negation canonical forms
    if let Some(inner) = as_neg(ctx, expr) {
        if let Expr::Number(n) = ctx.get(inner) {
            let n_display = n.clone();
            let neg_n = -n_display.clone();
            let normalized_n = if neg_n.is_zero() {
                num_rational::BigRational::from_integer(0.into())
            } else {
                neg_n
            };
            let rewritten = ctx.add(Expr::Number(normalized_n.clone()));
            return Some(CanonicalizeNegationRewrite {
                rewritten,
                desc: format!("-({}) = {}", n_display, normalized_n),
            });
        }

        if let Some(double_inner) = as_neg(ctx, inner) {
            return Some(CanonicalizeNegationRewrite {
                rewritten: double_inner,
                desc: "-(-x) = x".to_string(),
            });
        }

        if let Some((lhs, rhs)) = as_add(ctx, inner) {
            let neg_lhs = if let Expr::Number(n) = ctx.get(lhs) {
                ctx.add(Expr::Number(-n.clone()))
            } else {
                ctx.add(Expr::Neg(lhs))
            };
            let neg_rhs = if let Expr::Number(n) = ctx.get(rhs) {
                ctx.add(Expr::Number(-n.clone()))
            } else {
                ctx.add(Expr::Neg(rhs))
            };
            let rewritten = ctx.add(Expr::Add(neg_lhs, neg_rhs));
            return Some(CanonicalizeNegationRewrite {
                rewritten,
                desc: "-(a + b) = -a - b".to_string(),
            });
        }

        if let Some((lhs, rhs)) = as_mul(ctx, inner) {
            if let Expr::Number(n) = ctx.get(lhs) {
                let n_display = n.clone();
                let neg_n = -n_display.clone();
                let neg_n_expr = ctx.add(Expr::Number(neg_n.clone()));
                let rewritten = crate::build::mul2_raw(ctx, neg_n_expr, rhs);
                return Some(CanonicalizeNegationRewrite {
                    rewritten,
                    desc: format!("-({} * x) = {} * x", n_display, neg_n),
                });
            }
        }
    }

    // 3. Multiplication negative-sign pullout
    if let Some((lhs, rhs)) = as_mul(ctx, expr) {
        if let Some(inner_l) = as_neg(ctx, lhs) {
            let new_mul = crate::build::mul2_raw(ctx, inner_l, rhs);
            let rewritten = ctx.add(Expr::Neg(new_mul));
            return Some(CanonicalizeNegationRewrite {
                rewritten,
                desc: "(-a) * b = -(a * b)".to_string(),
            });
        }

        if let Some(inner_r) = as_neg(ctx, rhs) {
            if let Expr::Number(n) = ctx.get(lhs) {
                let n_display = n.clone();
                let neg_n = -n_display.clone();
                let neg_n_expr = ctx.add(Expr::Number(neg_n.clone()));
                let rewritten = crate::build::mul2_raw(ctx, neg_n_expr, inner_r);
                return Some(CanonicalizeNegationRewrite {
                    rewritten,
                    desc: format!("{} * (-x) = {} * x", n_display, neg_n),
                });
            }

            let new_mul = crate::build::mul2_raw(ctx, lhs, inner_r);
            let rewritten = ctx.add(Expr::Neg(new_mul));
            return Some(CanonicalizeNegationRewrite {
                rewritten,
                desc: "a * (-b) = -(a * b)".to_string(),
            });
        }
    }

    // 4. a / (-b) -> -(a / b)
    if let Some((lhs, rhs)) = as_div(ctx, expr) {
        if let Some(inner_r) = as_neg(ctx, rhs) {
            let new_div = ctx.add(Expr::Div(lhs, inner_r));
            let rewritten = ctx.add(Expr::Neg(new_div));
            return Some(CanonicalizeNegationRewrite {
                rewritten,
                desc: "a / (-b) = -(a / b)".to_string(),
            });
        }
    }

    None
}

/// Rewrite `(y-x)` into `-(x-y)` when `x < y` in canonical order.
///
/// Matches subtraction-like shape encoded as `Add(y, Neg(x))`.
pub fn try_rewrite_normalize_binomial_order_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<NormalizeBinomialOrderRewrite> {
    let Expr::Add(l, r) = ctx.get(expr) else {
        return None;
    };
    let (l, r) = (*l, *r);
    let Expr::Neg(inner) = ctx.get(r) else {
        return None;
    };
    let inner = *inner;

    if cas_ast::ordering::compare_expr(ctx, inner, l) != Ordering::Less {
        return None;
    }

    // (y-x) -> -(x-y) as Neg(Add(x, Neg(y)))
    let neg_l = ctx.add(Expr::Neg(l));
    let inner_minus_l = ctx.add(Expr::Add(inner, neg_l));
    let rewritten = ctx.add(Expr::Neg(inner_minus_l));
    Some(NormalizeBinomialOrderRewrite {
        rewritten,
        desc: "(y-x) -> -(x-y) for canonical order",
    })
}

/// Rewrite `-(a-b)` into canonical orientation `(b-a)` only when `a > b`.
pub fn try_rewrite_neg_sub_flip_expr(ctx: &mut Context, expr: ExprId) -> Option<NegSubFlipRewrite> {
    let Expr::Neg(inner) = ctx.get(expr) else {
        return None;
    };
    let inner = *inner;
    let (a, b) = extract_sub_like_pair(ctx, inner)?;

    if cas_ast::ordering::compare_expr(ctx, a, b) != Ordering::Greater {
        return None;
    }

    let rewritten = build_sub_like_expr(ctx, b, a);
    Some(NegSubFlipRewrite {
        inner,
        rewritten,
        desc: "-(a - b) → (b - a) (canonical orientation)",
    })
}

/// Rewrite division by numeric constant into multiplication by reciprocal,
/// with special flattening for nested quotients:
/// - `(a/b) / c` -> `a / (b*c)`
/// - `x / c` -> `(1/c) * x`
pub fn try_rewrite_canonicalize_div_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<CanonicalizeDivRewrite> {
    let Expr::Div(lhs, rhs) = ctx.get(expr) else {
        return None;
    };
    let (lhs, rhs) = (*lhs, *rhs);

    let Expr::Number(n) = ctx.get(rhs) else {
        return None;
    };
    if n.is_zero() {
        return None;
    }
    let n = n.clone();

    // (a/b) / c -> a / (b*c)
    if let Expr::Div(inner_num, inner_den) = ctx.get(lhs).clone() {
        let c_expr = ctx.add(Expr::Number(n.clone()));
        let new_den = crate::expr_rewrite::smart_mul(ctx, inner_den, c_expr);
        let rewritten = ctx.add(Expr::Div(inner_num, new_den));
        return Some(CanonicalizeDivRewrite {
            rewritten,
            desc: format!("(a/b) / {} = a / (b·{})", n, n),
        });
    }

    // x / c -> (1/c) * x
    let inv = n.recip();
    let inv_expr = ctx.add(Expr::Number(inv));
    let rewritten = crate::expr_rewrite::smart_mul(ctx, inv_expr, lhs);
    Some(CanonicalizeDivRewrite {
        rewritten,
        desc: format!("x / {} = (1/{}) * x", n, n),
    })
}

/// Rewrite `(-A)/(-B)` into `A/B` by canceling double sign.
pub fn try_rewrite_cancel_fraction_signs_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<CancelFractionSignsRewrite> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let num_id = *num;
    let den_id = *den;

    let (_, num_is_neg) = peel_negation_sub_like(ctx, num_id);
    let (_, den_is_neg) = peel_negation_sub_like(ctx, den_id);
    if !(num_is_neg && den_is_neg) {
        return None;
    }

    let new_num = build_unnegated_sub_like_expr(ctx, num_id);
    let new_den = build_unnegated_sub_like_expr(ctx, den_id);
    let rewritten = ctx.add(Expr::Div(new_num, new_den));

    Some(CancelFractionSignsRewrite {
        rewritten,
        desc: "(-A)/(-B) = A/B (cancel double sign)",
    })
}

#[cfg(test)]
mod tests {
    use super::{
        build_sub_like_expr, build_unnegated_sub_like_expr, extract_sub_like_pair,
        peel_negation_sub_like, try_rewrite_add_negative_constant_to_sub_expr,
        try_rewrite_cancel_fraction_signs_expr, try_rewrite_canonicalize_div_expr,
        try_rewrite_canonicalize_negation_expr, try_rewrite_neg_coeff_flip_binomial_expr,
        try_rewrite_neg_sub_flip_expr, try_rewrite_normalize_binomial_order_expr,
    };
    use cas_ast::{Context, Expr};
    use std::cmp::Ordering;

    #[test]
    fn extracts_sub_pair_from_sub_node() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let expr = ctx.add(Expr::Sub(a, b));
        assert_eq!(extract_sub_like_pair(&mut ctx, expr), Some((a, b)));
    }

    #[test]
    fn extracts_sub_pair_from_add_neg_forms() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let neg_b = ctx.add(Expr::Neg(b));
        let left = ctx.add(Expr::Add(a, neg_b));
        let right = ctx.add(Expr::Add(neg_b, a));

        assert_eq!(extract_sub_like_pair(&mut ctx, left), Some((a, b)));
        assert_eq!(extract_sub_like_pair(&mut ctx, right), Some((a, b)));
    }

    #[test]
    fn extracts_sub_pair_from_add_negative_number_forms() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let neg_three = ctx.num(-3);
        let left = ctx.add(Expr::Add(x, neg_three));
        let right = ctx.add(Expr::Add(neg_three, x));
        let three = num_rational::BigRational::from_integer(3.into());

        let (a1, b1) = extract_sub_like_pair(&mut ctx, left).expect("left");
        let (a2, b2) = extract_sub_like_pair(&mut ctx, right).expect("right");

        assert_eq!(a1, x);
        assert_eq!(a2, x);
        assert!(matches!(ctx.get(b1), Expr::Number(n) if n == &three));
        assert!(matches!(ctx.get(b2), Expr::Number(n) if n == &three));
    }

    #[test]
    fn builds_canonical_sub_like_form() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let sub_like = build_sub_like_expr(&mut ctx, a, b);

        let Expr::Add(left, right) = ctx.get(sub_like) else {
            panic!("expected Add");
        };
        assert_eq!(*left, a);
        let Expr::Neg(inner) = ctx.get(*right) else {
            panic!("expected Neg on RHS");
        };
        assert_eq!(*inner, b);
    }

    #[test]
    fn peels_explicit_and_mul_negation_forms() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let neg_x = ctx.add(Expr::Neg(x));
        let minus_one = ctx.num(-1);
        let mul_form = ctx.add(Expr::Mul(minus_one, x));

        assert_eq!(peel_negation_sub_like(&mut ctx, neg_x), (x, true));
        assert_eq!(peel_negation_sub_like(&mut ctx, mul_form), (x, true));
    }

    #[test]
    fn builds_unnegated_from_sub_like_and_explicit_neg() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let sub_like = ctx.add(Expr::Sub(a, b));
        let neg_a = ctx.add(Expr::Neg(a));

        let unneg_sub = build_unnegated_sub_like_expr(&mut ctx, sub_like);
        let Expr::Add(lhs, rhs) = ctx.get(unneg_sub) else {
            panic!("expected Add for unnegated sub-like");
        };
        assert_eq!(*lhs, b);
        let Expr::Neg(inner) = ctx.get(*rhs) else {
            panic!("expected Neg rhs for unnegated sub-like");
        };
        assert_eq!(*inner, a);

        assert_eq!(build_unnegated_sub_like_expr(&mut ctx, neg_a), a);
    }

    #[test]
    fn rewrites_negative_coeff_binomial_forms() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let a = ctx.var("a");
        let b = ctx.var("b");
        let minus_two = ctx.num(-2);

        let a_minus_b = ctx.add(Expr::Sub(a, b));
        let direct = ctx.add(Expr::Mul(minus_two, a_minus_b));
        let rewrite1 = try_rewrite_neg_coeff_flip_binomial_expr(&mut ctx, direct).expect("direct");
        assert!(rewrite1.desc.contains("(-k)"));

        let nested = ctx.add(Expr::Mul(x, a_minus_b));
        let expr2 = ctx.add(Expr::Mul(minus_two, nested));
        let rewrite2 =
            try_rewrite_neg_coeff_flip_binomial_expr(&mut ctx, expr2).expect("nested right");
        assert!(rewrite2.desc.contains("(-k)"));

        let nested_left = ctx.add(Expr::Mul(a_minus_b, x));
        let expr3 = ctx.add(Expr::Mul(minus_two, nested_left));
        let rewrite3 =
            try_rewrite_neg_coeff_flip_binomial_expr(&mut ctx, expr3).expect("nested left");
        assert!(rewrite3.desc.contains("(-k)"));
    }

    #[test]
    fn rewrites_add_negative_constant_to_sub() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let three = ctx.num(3);
        let neg_three = ctx.add_raw(Expr::Neg(three));

        let left_form = ctx.add_raw(Expr::Add(neg_three, x));
        let right_form = ctx.add_raw(Expr::Add(x, neg_three));

        let r1 =
            try_rewrite_add_negative_constant_to_sub_expr(&mut ctx, left_form).expect("left form");
        let r2 = try_rewrite_add_negative_constant_to_sub_expr(&mut ctx, right_form)
            .expect("right form");

        let expected = ctx.add(Expr::Sub(x, three));
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, r1.rewritten, expected),
            Ordering::Equal
        );
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, r2.rewritten, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn rewrites_normalize_binomial_order() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        // y-x encoded as Add(y, Neg(x))
        let neg_x = ctx.add(Expr::Neg(x));
        let expr = ctx.add(Expr::Add(y, neg_x));

        let rewrite =
            try_rewrite_normalize_binomial_order_expr(&mut ctx, expr).expect("rewrite expected");

        // Expected: -(x-y) => Neg(Add(x, Neg(y)))
        let neg_y = ctx.add(Expr::Neg(y));
        let x_minus_y = ctx.add(Expr::Add(x, neg_y));
        let expected = ctx.add(Expr::Neg(x_minus_y));
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn rewrites_neg_sub_flip_only_for_noncanonical_inner() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");

        // -(y-x) should flip to (x-y) because y > x
        let y_minus_x = ctx.add(Expr::Sub(y, x));
        let neg_noncanonical = ctx.add(Expr::Neg(y_minus_x));
        let rewrite =
            try_rewrite_neg_sub_flip_expr(&mut ctx, neg_noncanonical).expect("flip expected");
        let expected = build_sub_like_expr(&mut ctx, x, y);
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            Ordering::Equal
        );

        // -(x-y) should NOT flip (inner already canonical)
        let x_minus_y = ctx.add(Expr::Sub(x, y));
        let neg_canonical = ctx.add(Expr::Neg(x_minus_y));
        assert!(try_rewrite_neg_sub_flip_expr(&mut ctx, neg_canonical).is_none());
    }

    #[test]
    fn rewrites_canonicalize_division_forms() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let x = ctx.var("x");
        let c = ctx.num(5);

        let inner = ctx.add(Expr::Div(a, b));
        let nested = ctx.add(Expr::Div(inner, c));
        let r1 = try_rewrite_canonicalize_div_expr(&mut ctx, nested).expect("nested");
        let mul_bc = crate::expr_rewrite::smart_mul(&mut ctx, b, c);
        let expected_nested = ctx.add(Expr::Div(a, mul_bc));
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, r1.rewritten, expected_nested),
            Ordering::Equal
        );

        let simple = ctx.add(Expr::Div(x, c));
        let r2 = try_rewrite_canonicalize_div_expr(&mut ctx, simple).expect("simple");
        let inv = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            5.into(),
        )));
        let expected_simple = crate::expr_rewrite::smart_mul(&mut ctx, inv, x);
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, r2.rewritten, expected_simple),
            Ordering::Equal
        );
    }

    #[test]
    fn rewrites_canonicalize_negation_sub_and_division() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");

        let sub = ctx.add(Expr::Sub(a, b));
        let r1 = try_rewrite_canonicalize_negation_expr(&mut ctx, sub).expect("sub rewrite");
        let expected_sub = build_sub_like_expr(&mut ctx, a, b);
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, r1.rewritten, expected_sub),
            Ordering::Equal
        );

        let neg_b = ctx.add(Expr::Neg(b));
        let div = ctx.add(Expr::Div(a, neg_b));
        let r2 = try_rewrite_canonicalize_negation_expr(&mut ctx, div).expect("div rewrite");
        let div_ab = ctx.add(Expr::Div(a, b));
        let expected_div = ctx.add(Expr::Neg(div_ab));
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, r2.rewritten, expected_div),
            Ordering::Equal
        );
    }

    #[test]
    fn rewrites_cancel_fraction_signs() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let one = ctx.num(1);
        let x_minus_1 = build_sub_like_expr(&mut ctx, x, one);
        let y_minus_1 = build_sub_like_expr(&mut ctx, y, one);

        let neg_num = ctx.add(Expr::Neg(x_minus_1));
        let neg_den = ctx.add(Expr::Neg(y_minus_1));
        let expr = ctx.add(Expr::Div(neg_num, neg_den));
        let rewrite = try_rewrite_cancel_fraction_signs_expr(&mut ctx, expr).expect("rewrite");

        let expected = ctx.add(Expr::Div(x_minus_1, y_minus_1));
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            Ordering::Equal
        );
    }
}
