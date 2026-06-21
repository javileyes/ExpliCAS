//! Root-shape helpers over AST expressions.

use crate::abs_support::try_unwrap_abs_arg;
use crate::numeric::gcd_rational;
use crate::perfect_square_support::{extract_square_root_of_term, rational_sqrt};
use crate::sqrt_square_support::{detect_sqrt_square_pattern, SqrtSquarePattern};
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalRootRewrite {
    pub rewritten: ExprId,
    pub kind: CanonicalRootRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonicalRootRewriteKind {
    SqrtEvenPower,
    SqrtUnary,
    SqrtWithIndex,
    RootWithIndex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OddHalfPowerRewrite {
    pub rewritten: ExprId,
    pub numerator: i64,
    pub abs_power: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenestSqrtAddSqrtRewrite {
    pub rewritten: ExprId,
    pub desc: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootDenestingRewrite {
    pub rewritten: ExprId,
    pub kind: RootDenestingRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootDenestingRewriteKind {
    DenestSquareRoot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtractPerfectPowerFromRadicandRewrite {
    pub rewritten: ExprId,
    pub kind: ExtractPerfectPowerFromRadicandKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractPerfectPowerFromRadicandKind {
    ExtractPerfectSquare,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimplifySquareRootRewrite {
    pub rewritten: ExprId,
    pub kind: SimplifySquareRootRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimplifySquareRootRewriteKind {
    PerfectSquare,
    SquareRootFactors,
    AdditiveCommonFactor,
    QuotientOfSquares,
}

fn get_integer_coefficient_for_root_add(ctx: &Context, term: ExprId) -> Option<BigRational> {
    match ctx.get(term) {
        Expr::Number(n) => {
            if n.is_integer() {
                Some(n.clone())
            } else {
                None
            }
        }
        Expr::Mul(l, r) => {
            if let Expr::Number(n) = ctx.get(*l) {
                if n.is_integer() {
                    return Some(n.clone());
                }
            }
            if let Expr::Number(n) = ctx.get(*r) {
                if n.is_integer() {
                    return Some(n.clone());
                }
            }
            Some(BigRational::from_integer(1.into()))
        }
        Expr::Neg(inner) => get_integer_coefficient_for_root_add(ctx, *inner).map(|v| -v),
        _ => Some(BigRational::from_integer(1.into())),
    }
}

fn divide_root_add_term_by_rational(
    ctx: &mut Context,
    term: ExprId,
    divisor: &BigRational,
) -> ExprId {
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
            let divided = divide_root_add_term_by_rational(ctx, *inner, divisor);
            ctx.add(Expr::Neg(divided))
        }
        _ => term,
    }
}

fn rebuild_add_from_signed_terms(
    ctx: &mut Context,
    terms: &[(ExprId, crate::expr_nary::Sign)],
) -> Option<ExprId> {
    use crate::expr_nary::Sign;

    let mut iter = terms.iter();
    let (first_expr, first_sign) = iter.next().copied()?;
    let mut acc = if first_sign == Sign::Pos {
        first_expr
    } else {
        ctx.add(Expr::Neg(first_expr))
    };

    for (expr, sign) in iter.copied() {
        acc = match sign {
            Sign::Pos => ctx.add(Expr::Add(acc, expr)),
            Sign::Neg => ctx.add(Expr::Sub(acc, expr)),
        };
    }
    Some(acc)
}

fn try_rewrite_extract_additive_common_square_factor_expr(
    ctx: &mut Context,
    arg: ExprId,
) -> Option<ExprId> {
    use crate::expr_nary::AddView;

    let terms = AddView::from_expr(ctx, arg).terms;
    if terms.len() < 2 {
        return None;
    }

    let mut gcd: Option<BigRational> = None;
    for (term, _) in &terms {
        let coef = get_integer_coefficient_for_root_add(ctx, *term)?;
        let coef_abs = coef.abs();
        if coef_abs.is_zero() {
            continue;
        }
        gcd = Some(match gcd {
            None => coef_abs,
            Some(current) => gcd_rational(current, coef_abs),
        });
    }

    let gcd = gcd?;
    if !gcd.is_integer() || gcd <= BigRational::from_integer(1.into()) {
        return None;
    }
    let sqrt_gcd = rational_sqrt(&gcd)?;
    if !sqrt_gcd.is_integer() || sqrt_gcd <= BigRational::from_integer(1.into()) {
        return None;
    }

    let mut new_terms = Vec::with_capacity(terms.len());
    for (term, sign) in terms {
        let divided = divide_root_add_term_by_rational(ctx, term, &gcd);
        new_terms.push((divided, sign));
    }
    let inner = rebuild_add_from_signed_terms(ctx, &new_terms)?;
    let coeff_expr = ctx.add(Expr::Number(sqrt_gcd));
    let sqrt_inner = ctx.call_builtin(BuiltinFn::Sqrt, vec![inner]);
    Some(ctx.add(Expr::Mul(coeff_expr, sqrt_inner)))
}

fn try_rewrite_extract_additive_common_square_factor_reciprocal_root_expr(
    ctx: &mut Context,
    arg: ExprId,
    exp_id: ExprId,
) -> Option<ExprId> {
    use crate::expr_nary::AddView;

    let terms = AddView::from_expr(ctx, arg).terms;
    if terms.len() < 2 {
        return None;
    }

    let mut gcd: Option<BigRational> = None;
    for (term, _) in &terms {
        let coef = get_integer_coefficient_for_root_add(ctx, *term)?;
        let coef_abs = coef.abs();
        if coef_abs.is_zero() {
            continue;
        }
        gcd = Some(match gcd {
            None => coef_abs,
            Some(current) => gcd_rational(current, coef_abs),
        });
    }

    let gcd = gcd?;
    if !gcd.is_integer() || gcd <= BigRational::from_integer(1.into()) {
        return None;
    }
    let sqrt_gcd = rational_sqrt(&gcd)?;
    if !sqrt_gcd.is_integer() || sqrt_gcd <= BigRational::from_integer(1.into()) {
        return None;
    }

    let mut new_terms = Vec::with_capacity(terms.len());
    for (term, sign) in terms {
        let divided = divide_root_add_term_by_rational(ctx, term, &gcd);
        new_terms.push((divided, sign));
    }
    let inner = rebuild_add_from_signed_terms(ctx, &new_terms)?;
    let reciprocal_coeff = BigRational::one() / sqrt_gcd;
    let coeff_expr = ctx.add(Expr::Number(reciprocal_coeff));
    let inner_root = ctx.add(Expr::Pow(inner, exp_id));
    Some(ctx.add(Expr::Mul(coeff_expr, inner_root)))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenestCubeQuadraticRewrite {
    pub rewritten: ExprId,
    pub desc: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CubicConjugateTrapRewrite {
    pub rewritten: ExprId,
    pub desc: String,
}

fn as_sqrt_like(ctx: &Context, e: ExprId) -> Option<ExprId> {
    match ctx.get(e) {
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            Some(args[0])
        }
        Expr::Pow(base, exp) => {
            match ctx.get(*exp) {
                Expr::Number(n) => {
                    if *n.numer() == 1.into() && *n.denom() == 2.into() {
                        return Some(*base);
                    }
                }
                Expr::Div(num, den) => {
                    if let (Expr::Number(n_num), Expr::Number(n_den)) =
                        (ctx.get(*num), ctx.get(*den))
                    {
                        if n_num.is_one() && n_den.is_integer() && *n_den.numer() == 2.into() {
                            return Some(*base);
                        }
                    }
                }
                _ => {}
            }
            None
        }
        _ => None,
    }
}

/// Extract base/radicand from square-root-like forms.
///
/// Recognizes:
/// - `sqrt(x)` -> `x`
/// - `x^(1/2)` -> `x`
pub fn extract_square_root_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    as_sqrt_like(ctx, expr)
}

/// Rewrite nested square roots of the form `sqrt(a + sqrt(b))` into
/// `sqrt(m) + sqrt(n)` when `m,n` are non-negative rationals:
/// `m = (a + sqrt(a^2-b))/2`, `n = (a - sqrt(a^2-b))/2`.
pub fn try_rewrite_denest_sqrt_add_sqrt_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<DenestSqrtAddSqrtRewrite> {
    let inner = as_sqrt_like(ctx, expr)?;
    let (left, right, is_add) = match ctx.get(inner) {
        Expr::Add(l, r) => (*l, *r, true),
        Expr::Sub(l, r) => (*l, *r, false),
        _ => return None,
    };

    if !is_add {
        return None;
    }

    let extract_ab =
        |ctx: &Context, a_term: ExprId, surd_term: ExprId| -> Option<(BigRational, BigRational)> {
            let Expr::Number(a) = ctx.get(a_term) else {
                return None;
            };
            let b_inner = as_sqrt_like(ctx, surd_term)?;
            let Expr::Number(b) = ctx.get(b_inner) else {
                return None;
            };
            Some((a.clone(), b.clone()))
        };

    let (a_val, b_val) = extract_ab(ctx, left, right).or_else(|| extract_ab(ctx, right, left))?;
    let disc = &a_val * &a_val - &b_val;
    let disc_sqrt = rational_sqrt(&disc)?;

    let two = BigRational::from_integer(2.into());
    let m = (&a_val + &disc_sqrt) / &two;
    let n = (&a_val - &disc_sqrt) / &two;

    if m.is_negative() || n.is_negative() {
        return None;
    }

    let m_expr = ctx.add(Expr::Number(m.clone()));
    let n_expr = ctx.add(Expr::Number(n.clone()));
    let half_m = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
    let half_n = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
    let sqrt_m = ctx.add(Expr::Pow(m_expr, half_m));
    let sqrt_n = ctx.add(Expr::Pow(n_expr, half_n));
    let rewritten = ctx.add(Expr::Add(sqrt_m, sqrt_n));

    Some(DenestSqrtAddSqrtRewrite {
        rewritten,
        desc: format!("Denest nested square root: √(a+√b) = √({}) + √({})", m, n),
    })
}

/// Rewrite `sqrt(A ± C*sqrt(D))` / `(A ± C*sqrt(D))^(1/2)` by denesting when
/// `A^2 - C^2*D` is a non-negative perfect square.
pub fn try_rewrite_root_denesting_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<RootDenestingRewrite> {
    enum RootShape {
        Sqrt(ExprId),
        HalfPow(ExprId),
        Other,
    }
    let shape = match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            RootShape::Sqrt(args[0])
        }
        Expr::Pow(b, e) => {
            if let Expr::Number(n) = ctx.get(*e) {
                if *n.numer() == 1.into() && *n.denom() == 2.into() {
                    RootShape::HalfPow(*b)
                } else {
                    RootShape::Other
                }
            } else {
                RootShape::Other
            }
        }
        _ => RootShape::Other,
    };

    let inner = match shape {
        RootShape::Sqrt(i) | RootShape::HalfPow(i) => i,
        RootShape::Other => return None,
    };

    let (a, b, is_add) = match ctx.get(inner) {
        Expr::Add(l, r) => (*l, *r, true),
        Expr::Sub(l, r) => (*l, *r, false),
        _ => return None,
    };

    fn analyze_sqrt_term(ctx: &Context, e: ExprId) -> Option<(Option<ExprId>, ExprId)> {
        match ctx.get(e) {
            Expr::Function(fname, fargs)
                if ctx.is_builtin(*fname, BuiltinFn::Sqrt) && fargs.len() == 1 =>
            {
                Some((None, fargs[0]))
            }
            Expr::Pow(b, exp) => {
                if let Expr::Number(n) = ctx.get(*exp) {
                    if *n.numer() == 3.into() && *n.denom() == 2.into() {
                        return Some((Some(*b), *b));
                    }
                }
                None
            }
            Expr::Mul(l, r) => {
                if let Some(inner) = as_sqrt_like(ctx, *r) {
                    return Some((Some(*l), inner));
                }
                if let Some(inner) = as_sqrt_like(ctx, *l) {
                    return Some((Some(*r), inner));
                }
                None
            }
            _ => None,
        }
    }

    let check_permutation = |ctx: &mut Context, term_a: ExprId, term_b: ExprId| -> Option<ExprId> {
        let (c_opt, d) = analyze_sqrt_term(ctx, term_b)?;
        let c = c_opt.unwrap_or_else(|| ctx.num(1));

        let (Expr::Number(val_a), Expr::Number(val_c), Expr::Number(val_d)) =
            (ctx.get(term_a), ctx.get(c), ctx.get(d))
        else {
            return None;
        };
        let val_a = val_a.clone();
        let val_c = val_c.clone();
        let val_d = val_d.clone();

        let val_beff = &val_c * &val_c * &val_d;
        let val_delta = &val_a * &val_a - &val_beff;
        if val_delta < BigRational::zero() || !val_delta.is_integer() {
            return None;
        }
        let int_delta = val_delta.to_integer();
        let sqrt_delta = int_delta.sqrt();
        if sqrt_delta.clone() * sqrt_delta.clone() != int_delta {
            return None;
        }

        let z_val = ctx.add(Expr::Number(BigRational::from_integer(sqrt_delta)));
        let two = ctx.num(2);
        let term1_num = ctx.add(Expr::Add(term_a, z_val));
        let term2_num = ctx.add(Expr::Sub(term_a, z_val));
        let term1_frac = ctx.add(Expr::Div(term1_num, two));
        let term2_frac = ctx.add(Expr::Div(term2_num, two));
        let term1 = ctx.call_builtin(BuiltinFn::Sqrt, vec![term1_frac]);
        let term2 = ctx.call_builtin(BuiltinFn::Sqrt, vec![term2_frac]);

        let c_is_negative = val_c < BigRational::zero();
        let effective_sub = if is_add {
            c_is_negative
        } else {
            !c_is_negative
        };
        Some(if effective_sub {
            ctx.add(Expr::Sub(term1, term2))
        } else {
            ctx.add(Expr::Add(term1, term2))
        })
    };

    let rewritten = check_permutation(ctx, a, b).or_else(|| check_permutation(ctx, b, a))?;
    Some(RootDenestingRewrite {
        rewritten,
        kind: RootDenestingRewriteKind::DenestSquareRoot,
    })
}

/// Extract largest perfect `n`-th-power factor from radicand in `(c*rest)^(1/n)`.
///
/// Examples:
/// - `(4*x)^(1/2) -> 2*x^(1/2)`
/// - `(8*x)^(1/2) -> 2*(2*x)^(1/2)`
/// - `((1/4)*x)^(1/2) -> (1/2)*x^(1/2)`
/// - `((1/4)*x)^(-1/2) -> 2*x^(-1/2)`
/// - `(16*x)^(1/4) -> 2*x^(1/4)`
pub fn try_rewrite_extract_perfect_power_from_radicand_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExtractPerfectPowerFromRadicandRewrite> {
    let (base, exp_id) = match ctx.get(expr) {
        Expr::Pow(b, e) => (*b, *e),
        _ => return None,
    };

    #[derive(Clone, Copy)]
    enum RootExponentSign {
        Positive,
        Negative,
    }

    let exp_value = cas_ast::views::as_rational_const(ctx, exp_id, 4)?;
    let exponent_sign = if exp_value.numer() == &1.into() {
        RootExponentSign::Positive
    } else if exp_value.numer() == &(-1).into() {
        RootExponentSign::Negative
    } else {
        return None;
    };
    let d = exp_value.denom();
    let d_u32 = d.to_u32_digits().1.first().copied()?;
    let root_index = if d_u32 >= 2 && d.sign() == num_bigint::Sign::Plus && d.bits() <= 32 {
        d_u32
    } else {
        return None;
    };

    if matches!(exponent_sign, RootExponentSign::Negative) && root_index == 2 {
        if let Some(rewritten) = try_rewrite_scaled_unit_square_reciprocal_root(ctx, base, exp_id) {
            return Some(ExtractPerfectPowerFromRadicandRewrite {
                rewritten,
                kind: ExtractPerfectPowerFromRadicandKind::ExtractPerfectSquare,
            });
        }
        if let Some(rewritten) =
            try_rewrite_scaled_square_plus_unit_reciprocal_root(ctx, base, exp_id)
        {
            return Some(ExtractPerfectPowerFromRadicandRewrite {
                rewritten,
                kind: ExtractPerfectPowerFromRadicandKind::ExtractPerfectSquare,
            });
        }
        if let Some(rewritten) =
            try_rewrite_extract_additive_common_square_factor_reciprocal_root_expr(
                ctx, base, exp_id,
            )
        {
            return Some(ExtractPerfectPowerFromRadicandRewrite {
                rewritten,
                kind: ExtractPerfectPowerFromRadicandKind::ExtractPerfectSquare,
            });
        }
    }

    let (num_val, rest) = match ctx.get(base) {
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            if let Expr::Number(n) = ctx.get(l) {
                (n.clone(), r)
            } else if let Expr::Number(n) = ctx.get(r) {
                (n.clone(), l)
            } else {
                return None;
            }
        }
        _ => return None,
    };

    let (outside_coeff, new_coeff) = extract_positive_rational_power_factor(&num_val, root_index)?;

    let extracted_coeff = match exponent_sign {
        RootExponentSign::Positive => outside_coeff,
        RootExponentSign::Negative => reciprocal_positive_rational(&outside_coeff),
    };
    let k_expr = ctx.add(Expr::Number(extracted_coeff));
    let new_radicand = if new_coeff.is_one() {
        rest
    } else {
        let new_coeff_expr = ctx.add(Expr::Number(new_coeff));
        ctx.add(Expr::Mul(new_coeff_expr, rest))
    };
    let new_root = ctx.add(Expr::Pow(new_radicand, exp_id));
    let simplified_root = try_rewrite_simplify_square_root_expr(ctx, new_root)
        .map(|rewrite| rewrite.rewritten)
        .unwrap_or(new_root);
    let rewritten = if root_index == 2 {
        try_extract_square_factors_from_mixed_radicand(ctx, k_expr, new_radicand, exp_id)
            .unwrap_or_else(|| ctx.add(Expr::Mul(k_expr, simplified_root)))
    } else {
        ctx.add(Expr::Mul(k_expr, simplified_root))
    };

    Some(ExtractPerfectPowerFromRadicandRewrite {
        rewritten,
        kind: ExtractPerfectPowerFromRadicandKind::ExtractPerfectSquare,
    })
}

fn try_rewrite_scaled_unit_square_reciprocal_root(
    ctx: &mut Context,
    base: ExprId,
    exp_id: ExprId,
) -> Option<ExprId> {
    let (left, square_term) = match ctx.get(base) {
        Expr::Sub(left, square_term) => (*left, *square_term),
        _ => return None,
    };
    let Expr::Number(left_num) = ctx.get(left) else {
        return None;
    };
    if !left_num.is_one() {
        return None;
    }

    let (scale, inner) = positive_scaled_square_base(ctx, square_term)?;
    if scale.is_one() {
        return None;
    }

    let scale_squared = &scale * &scale;
    let offset = BigRational::one() / scale_squared;
    let outside = reciprocal_positive_rational(&scale);
    let two = ctx.num(2);
    let inner_square = ctx.add(Expr::Pow(inner, two));
    let offset_expr = ctx.add(Expr::Number(offset));
    let new_base = ctx.add(Expr::Sub(offset_expr, inner_square));
    let new_root = ctx.add(Expr::Pow(new_base, exp_id));
    if outside.is_one() {
        Some(new_root)
    } else {
        let outside_expr = ctx.add(Expr::Number(outside));
        Some(ctx.add(Expr::Mul(outside_expr, new_root)))
    }
}

fn try_rewrite_scaled_square_plus_unit_reciprocal_root(
    ctx: &mut Context,
    base: ExprId,
    exp_id: ExprId,
) -> Option<ExprId> {
    let square_term = match ctx.get(base) {
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            match (ctx.get(l), ctx.get(r)) {
                (Expr::Number(n), _) if n.is_one() => r,
                (_, Expr::Number(n)) if n.is_one() => l,
                _ => return None,
            }
        }
        _ => return None,
    };

    let (scale, inner) = positive_scaled_square_base(ctx, square_term)?;
    if scale.is_one() {
        return None;
    }

    let scale_squared = &scale * &scale;
    let offset = BigRational::one() / scale_squared;
    let outside = reciprocal_positive_rational(&scale);
    let two = ctx.num(2);
    let inner_square = ctx.add(Expr::Pow(inner, two));
    let offset_expr = ctx.add(Expr::Number(offset));
    let new_base = ctx.add(Expr::Add(offset_expr, inner_square));
    let new_root = ctx.add(Expr::Pow(new_base, exp_id));
    if outside.is_one() {
        Some(new_root)
    } else {
        let outside_expr = ctx.add(Expr::Number(outside));
        Some(ctx.add(Expr::Mul(outside_expr, new_root)))
    }
}

fn positive_scaled_square_base(
    ctx: &Context,
    square_term: ExprId,
) -> Option<(BigRational, ExprId)> {
    let (square_base, square_exp) = match ctx.get(square_term) {
        Expr::Pow(square_base, square_exp) => (*square_base, *square_exp),
        _ => return None,
    };
    let Expr::Number(exp_num) = ctx.get(square_exp) else {
        return None;
    };
    if *exp_num != BigRational::from_integer(2.into()) {
        return None;
    }

    match ctx.get(square_base) {
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            if let Some(scale) = positive_rational_const(ctx, l) {
                Some((scale, r))
            } else {
                positive_rational_const(ctx, r).map(|scale| (scale, l))
            }
        }
        Expr::Div(num, den) => {
            let Expr::Number(den_num) = ctx.get(*den) else {
                return None;
            };
            if !den_num.is_positive() {
                return None;
            }
            Some((reciprocal_positive_rational(den_num), *num))
        }
        _ => None,
    }
}

fn positive_rational_const(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    let value = cas_ast::views::as_rational_const(ctx, expr, 4)?;
    value.is_positive().then_some(value)
}

fn extract_positive_rational_power_factor(
    value: &BigRational,
    root_index: u32,
) -> Option<(BigRational, BigRational)> {
    if !value.is_positive() {
        return None;
    }

    let (outside_num, remaining_num) =
        extract_positive_integer_power_factor(value.numer(), root_index);
    let (outside_den, remaining_den) =
        extract_positive_integer_power_factor(value.denom(), root_index);
    let outside = BigRational::new(outside_num, outside_den);
    if outside.is_one() {
        return None;
    }

    Some((outside, BigRational::new(remaining_num, remaining_den)))
}

fn reciprocal_positive_rational(value: &BigRational) -> BigRational {
    BigRational::new(value.denom().clone(), value.numer().clone())
}

fn extract_positive_integer_power_factor(value: &BigInt, root_index: u32) -> (BigInt, BigInt) {
    let mut outside = BigInt::from(1);
    let mut remaining = value.clone();
    let mut trial = BigInt::from(2);

    loop {
        if &trial * &trial > remaining {
            break;
        }
        let mut count: u32 = 0;
        while (&remaining % &trial).is_zero() {
            remaining /= &trial;
            count += 1;
        }
        let extracted = count / root_index;
        for _ in 0..extracted {
            outside *= &trial;
        }
        trial += 1;
    }

    let mut outside_power = BigInt::from(1);
    for _ in 0..root_index {
        outside_power *= &outside;
    }

    (outside, value / outside_power)
}

fn try_extract_square_factors_from_mixed_radicand(
    ctx: &mut Context,
    outside_numeric: ExprId,
    radicand: ExprId,
    exp_id: ExprId,
) -> Option<ExprId> {
    let Expr::Mul(_, _) = ctx.get(radicand) else {
        return None;
    };

    let mut outside_factors = vec![outside_numeric];
    let mut inside_factors = Vec::new();
    let mut changed = false;

    for factor in crate::expr_nary::mul_factors(ctx, radicand) {
        if let Some(square_factor) = try_extract_abs_of_square_factorized_base(ctx, factor) {
            outside_factors.push(square_factor);
            changed = true;
        } else {
            inside_factors.push(factor);
        }
    }

    if !changed || inside_factors.is_empty() {
        return None;
    }

    let inside = crate::expr_nary::build_balanced_mul(ctx, &inside_factors);
    let inside_root = ctx.add(Expr::Pow(inside, exp_id));
    outside_factors.push(inside_root);
    Some(crate::expr_nary::build_balanced_mul(ctx, &outside_factors))
}

/// Simplify square roots of perfect-square polynomials.
///
/// Main rewrite:
/// - `sqrt((dx+e)^2) -> abs(dx+e)`
///
/// Fallback for repeated linear factors:
/// - `sqrt((f)^(2k)) -> abs(f)^k`
/// - `sqrt((f)^(2k+1)) -> abs(f)^k * sqrt(f)`
pub fn try_rewrite_simplify_square_root_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<SimplifySquareRootRewrite> {
    use crate::expr_rewrite::smart_mul;
    use crate::polynomial::Polynomial;

    let arg = match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            Some(args[0])
        }
        Expr::Pow(b, e) => match ctx.get(*e) {
            Expr::Number(n) if *n.numer() == 1.into() && *n.denom() == 2.into() => Some(*b),
            _ => None,
        },
        _ => None,
    }?;

    if let Expr::Div(num, den) = ctx.get(arg).clone() {
        if let (Some(num_abs), Some(den_abs)) = (
            try_extract_abs_of_square_base(ctx, num),
            try_extract_abs_of_square_base(ctx, den),
        ) {
            let num_inner = try_unwrap_abs_arg(ctx, num_abs)?;
            let den_inner = try_unwrap_abs_arg(ctx, den_abs)?;
            let (num_inner, den_inner) =
                normalize_common_rational_content_in_quotient(ctx, num_inner, den_inner);
            let quotient = ctx.add(Expr::Div(num_inner, den_inner));
            let rewritten = ctx.call_builtin(BuiltinFn::Abs, vec![quotient]);
            return Some(SimplifySquareRootRewrite {
                rewritten,
                kind: SimplifySquareRootRewriteKind::QuotientOfSquares,
            });
        }
    }

    if let Some(rewritten) = try_extract_abs_of_square_factorized_base(ctx, arg) {
        return Some(SimplifySquareRootRewrite {
            rewritten,
            kind: SimplifySquareRootRewriteKind::PerfectSquare,
        });
    }

    match ctx.get(arg) {
        Expr::Add(_, _) | Expr::Sub(_, _) => {}
        _ => return None,
    }

    let vars = cas_ast::collect_variables(ctx, arg);
    if vars.len() != 1 {
        return None;
    }
    let var = vars.iter().next()?;
    let poly = Polynomial::from_expr(ctx, arg, var).ok()?;

    if poly.degree() == 2 && poly.coeffs.len() >= 3 {
        let a = poly.coeffs.get(2).cloned();
        let b = poly.coeffs.get(1).cloned();
        let c = poly.coeffs.first().cloned();
        if let (Some(a), Some(b), Some(c)) = (a, b, c) {
            let four = BigRational::from_integer(4.into());
            let discriminant = b.clone() * b.clone() - four * a.clone() * c.clone();
            if discriminant.is_zero() {
                if let (Some(d), Some(_)) = (rational_sqrt(&a), rational_sqrt(&c)) {
                    let two = BigRational::from_integer(2.into());
                    let e = if d.is_zero() {
                        rational_sqrt(&c).unwrap_or_else(BigRational::zero)
                    } else {
                        b.clone() / (two * d.clone())
                    };

                    let var_expr = ctx.var(var);
                    let d_expr = ctx.add(Expr::Number(d.clone()));
                    let e_expr = ctx.add(Expr::Number(e.clone()));
                    let one = BigRational::from_integer(1.into());
                    let dx = if d == one {
                        var_expr
                    } else {
                        smart_mul(ctx, d_expr, var_expr)
                    };
                    let linear = if e.is_zero() {
                        dx
                    } else {
                        ctx.add(Expr::Add(dx, e_expr))
                    };
                    let rewritten = ctx.call_builtin(BuiltinFn::Abs, vec![linear]);
                    return Some(SimplifySquareRootRewrite {
                        rewritten,
                        kind: SimplifySquareRootRewriteKind::PerfectSquare,
                    });
                }
            }
        }
    }

    if let Some(rewritten) = try_rewrite_extract_additive_common_square_factor_expr(ctx, arg) {
        return Some(SimplifySquareRootRewrite {
            rewritten,
            kind: SimplifySquareRootRewriteKind::AdditiveCommonFactor,
        });
    }

    let factors = poly.factor_rational_roots();
    if factors.is_empty() {
        return None;
    }
    let first = &factors[0];
    if !factors.iter().all(|f| f == first) {
        return None;
    }
    let count = factors.len() as u32;
    if count < 2 {
        return None;
    }

    let base = first.to_expr(ctx);
    let k = count / 2;
    let rem = count % 2;
    let abs_base = wrap_abs_if_needed(ctx, base);
    let term1 = if k == 1 {
        abs_base
    } else {
        let k_expr = ctx.num(k as i64);
        ctx.add(Expr::Pow(abs_base, k_expr))
    };
    if rem == 0 {
        return Some(SimplifySquareRootRewrite {
            rewritten: term1,
            kind: SimplifySquareRootRewriteKind::PerfectSquare,
        });
    }

    let sqrt_base = ctx.call_builtin(BuiltinFn::Sqrt, vec![base]);
    let rewritten = smart_mul(ctx, term1, sqrt_base);
    Some(SimplifySquareRootRewrite {
        rewritten,
        kind: SimplifySquareRootRewriteKind::SquareRootFactors,
    })
}

fn wrap_abs_if_needed(ctx: &mut Context, expr: ExprId) -> ExprId {
    if crate::abs_support::is_sum_of_nonnegative(ctx, expr) {
        expr
    } else {
        ctx.call_builtin(BuiltinFn::Abs, vec![expr])
    }
}

fn try_extract_abs_of_square_factorized_base(ctx: &mut Context, base: ExprId) -> Option<ExprId> {
    if let Some(pattern) = detect_sqrt_square_pattern(ctx, base) {
        let arg = match pattern {
            SqrtSquarePattern::PowSquare { arg } | SqrtSquarePattern::RepeatedMul { arg } => arg,
        };
        return Some(wrap_abs_if_needed(ctx, arg));
    }

    if let Some(root) = extract_square_root_of_term(ctx, base) {
        return Some(wrap_abs_if_needed(ctx, root));
    }

    if let Some(rewritten) = try_extract_abs_of_reciprocal_square_base(ctx, base) {
        return Some(rewritten);
    }

    if let Some(rewritten) = try_extract_abs_of_generic_shifted_unit_square(ctx, base) {
        return Some(rewritten);
    }

    if let Some(rewritten) = try_extract_abs_of_combined_shifted_unit_square(ctx, base) {
        return Some(rewritten);
    }

    if let Some(rewritten) = try_extract_abs_of_univariate_square_polynomial(ctx, base) {
        return Some(rewritten);
    }

    let factored = crate::factor::factor(ctx, base);
    if factored != base {
        if let Some(pattern) = detect_sqrt_square_pattern(ctx, factored) {
            let arg = match pattern {
                SqrtSquarePattern::PowSquare { arg } | SqrtSquarePattern::RepeatedMul { arg } => {
                    arg
                }
            };
            return Some(wrap_abs_if_needed(ctx, arg));
        }
        if let Some(root) = extract_square_root_of_term(ctx, factored) {
            return Some(wrap_abs_if_needed(ctx, root));
        }
        if let Some(rewritten) = try_extract_abs_of_reciprocal_square_base(ctx, factored) {
            return Some(rewritten);
        }
        if let Some(rewritten) = try_extract_abs_of_generic_shifted_unit_square(ctx, factored) {
            return Some(rewritten);
        }
        if let Some(rewritten) = try_extract_abs_of_combined_shifted_unit_square(ctx, factored) {
            return Some(rewritten);
        }
        if let Some(rewritten) = try_extract_abs_of_univariate_square_polynomial(ctx, factored) {
            return Some(rewritten);
        }
    }

    None
}

fn try_extract_linear_two_times_candidate(ctx: &Context, term: ExprId) -> Option<ExprId> {
    let Expr::Mul(left, right) = ctx.get(term) else {
        return None;
    };
    match (ctx.get(*left), ctx.get(*right)) {
        (Expr::Number(n), _) if *n == BigRational::from_integer(2.into()) => Some(*right),
        (_, Expr::Number(n)) if *n == BigRational::from_integer(2.into()) => Some(*left),
        _ => None,
    }
}

fn square_term_matches_candidate(
    ctx: &mut Context,
    square_term: ExprId,
    candidate: ExprId,
) -> bool {
    use std::cmp::Ordering;
    fn expr_eq(ctx: &Context, left: ExprId, right: ExprId) -> bool {
        compare_expr(ctx, left, right) == Ordering::Equal
    }

    if let Some(pattern) = detect_sqrt_square_pattern(ctx, square_term) {
        let arg = match pattern {
            SqrtSquarePattern::PowSquare { arg } | SqrtSquarePattern::RepeatedMul { arg } => arg,
        };
        if expr_eq(&*ctx, arg, candidate) {
            return true;
        }
    }

    let two = ctx.num(2);
    let candidate_sq = ctx.add(Expr::Pow(candidate, two));
    if expr_eq(&*ctx, square_term, candidate_sq) {
        return true;
    }

    if let (Some(square_arg), Some(candidate_arg)) = (
        crate::expr_extract::extract_exp_argument(&*ctx, square_term),
        crate::expr_extract::extract_exp_argument(&*ctx, candidate),
    ) {
        if let Some(inner) =
            crate::trig_roots_flatten::extract_double_angle_arg_relaxed(ctx, square_arg)
        {
            if expr_eq(&*ctx, inner, candidate_arg) {
                return true;
            }
        }
    }

    match ctx.get(candidate).clone() {
        Expr::Div(num, den) => {
            let one = ctx.num(1);
            if expr_eq(&*ctx, num, one) {
                if let Some(radicand) = extract_square_root_base(ctx, den) {
                    let reciprocal_num = ctx.num(1);
                    let reciprocal = ctx.add(Expr::Div(reciprocal_num, radicand));
                    if expr_eq(&*ctx, square_term, reciprocal) {
                        return true;
                    }
                    let neg_one = ctx.add(Expr::Number(BigRational::from_integer((-1).into())));
                    let inverse_pow = ctx.add(Expr::Pow(radicand, neg_one));
                    if expr_eq(&*ctx, square_term, inverse_pow) {
                        return true;
                    }
                }
            }
        }
        Expr::Pow(base, exp) => match ctx.get(exp) {
            Expr::Number(n) if *n == BigRational::new((-1).into(), 2.into()) => {
                let reciprocal_num = ctx.num(1);
                let reciprocal = ctx.add(Expr::Div(reciprocal_num, base));
                if expr_eq(&*ctx, square_term, reciprocal) {
                    return true;
                }
                let neg_one = ctx.add(Expr::Number(BigRational::from_integer((-1).into())));
                let inverse_pow = ctx.add(Expr::Pow(base, neg_one));
                if expr_eq(&*ctx, square_term, inverse_pow) {
                    return true;
                }
            }
            _ => {}
        },
        _ => {}
    }

    false
}

fn try_extract_abs_of_reciprocal_square_base(ctx: &mut Context, base: ExprId) -> Option<ExprId> {
    let one = ctx.num(1);
    match ctx.get(base).clone() {
        Expr::Div(num, den) if compare_expr(&*ctx, num, one) == std::cmp::Ordering::Equal => {
            let sqrt_den = ctx.call_builtin(BuiltinFn::Sqrt, vec![den]);
            let reciprocal = ctx.add(Expr::Div(one, sqrt_den));
            Some(ctx.call_builtin(BuiltinFn::Abs, vec![reciprocal]))
        }
        Expr::Pow(inner, exp) => match ctx.get(exp) {
            Expr::Number(n) if *n == BigRational::from_integer((-1).into()) => {
                let sqrt_inner = ctx.call_builtin(BuiltinFn::Sqrt, vec![inner]);
                let reciprocal = ctx.add(Expr::Div(one, sqrt_inner));
                Some(ctx.call_builtin(BuiltinFn::Abs, vec![reciprocal]))
            }
            _ => None,
        },
        _ => None,
    }
}

fn try_extract_abs_of_generic_shifted_unit_square(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    use crate::expr_nary::{AddView, Sign};

    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 3 {
        return None;
    }

    let mut square_term = None;
    let mut linear = None;
    let mut constant_one = false;

    for (term, sign) in terms {
        if sign == Sign::Pos && matches!(ctx.get(term), Expr::Number(n) if n.is_one()) {
            if constant_one {
                return None;
            }
            constant_one = true;
            continue;
        }

        if let Some(candidate) = try_extract_linear_two_times_candidate(ctx, term) {
            if linear.is_some() {
                return None;
            }
            linear = Some((candidate, sign));
            continue;
        }

        if sign == Sign::Pos {
            if square_term.is_some() {
                return None;
            }
            square_term = Some(term);
            continue;
        }

        return None;
    }

    let (candidate, sign) = linear?;
    let square_term = square_term?;
    if !constant_one || !square_term_matches_candidate(ctx, square_term, candidate) {
        return None;
    }

    let one = ctx.num(1);
    let inner = match sign {
        Sign::Pos => ctx.add(Expr::Add(candidate, one)),
        Sign::Neg => ctx.add(Expr::Sub(candidate, one)),
    };
    Some(ctx.call_builtin(BuiltinFn::Abs, vec![inner]))
}

fn combined_one_plus_square_term_matches_candidate(
    ctx: &mut Context,
    term: ExprId,
    candidate: ExprId,
) -> bool {
    let Expr::Div(num, den) = ctx.get(term).clone() else {
        return false;
    };

    let one = ctx.num(1);
    let reciprocal = ctx.add(Expr::Div(one, den));
    if !square_term_matches_candidate(ctx, reciprocal, candidate) {
        let neg_one = ctx.add(Expr::Number(BigRational::from_integer((-1).into())));
        let inverse_pow = ctx.add(Expr::Pow(den, neg_one));
        if !square_term_matches_candidate(ctx, inverse_pow, candidate) {
            return false;
        }
    }

    let expected_num = ctx.add(Expr::Add(den, one));
    compare_expr(&*ctx, num, expected_num) == std::cmp::Ordering::Equal
}

fn try_extract_abs_of_combined_shifted_unit_square(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    use crate::expr_nary::{AddView, Sign};

    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return None;
    }

    for idx in 0..2 {
        let (linear_term, linear_sign) = terms[idx];
        let (combined_term, combined_sign) = terms[1 - idx];
        if combined_sign != Sign::Pos {
            continue;
        }

        let Some(candidate) = try_extract_linear_two_times_candidate(ctx, linear_term) else {
            continue;
        };
        if !combined_one_plus_square_term_matches_candidate(ctx, combined_term, candidate) {
            continue;
        }

        let one = ctx.num(1);
        let inner = match linear_sign {
            Sign::Pos => ctx.add(Expr::Add(candidate, one)),
            Sign::Neg => ctx.add(Expr::Sub(candidate, one)),
        };
        return Some(ctx.call_builtin(BuiltinFn::Abs, vec![inner]));
    }

    None
}

fn try_extract_abs_of_square_base(ctx: &mut Context, base: ExprId) -> Option<ExprId> {
    if let Some(rewritten) = try_extract_abs_of_square_factorized_base(ctx, base) {
        return Some(rewritten);
    }

    let sqrt_expr = ctx.call_builtin(BuiltinFn::Sqrt, vec![base]);
    let rewrite = try_rewrite_simplify_square_root_expr(ctx, sqrt_expr)?;
    try_unwrap_abs_arg(ctx, rewrite.rewritten)?;
    Some(rewrite.rewritten)
}

fn normalize_common_rational_content_in_quotient(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
) -> (ExprId, ExprId) {
    use crate::polynomial::Polynomial;
    use num_traits::{One, Zero};

    let vars_num = cas_ast::collect_variables(ctx, num);
    let vars_den = cas_ast::collect_variables(ctx, den);
    if vars_num.len() != 1 || vars_den.len() != 1 || vars_num != vars_den {
        return (num, den);
    }

    let Some(var) = vars_num.iter().next() else {
        return (num, den);
    };

    let Ok(num_poly) = Polynomial::from_expr(ctx, num, var) else {
        return (num, den);
    };
    let Ok(den_poly) = Polynomial::from_expr(ctx, den, var) else {
        return (num, den);
    };

    let rational_content = |poly: &Polynomial| -> BigRational {
        let mut gcd_num = None::<BigInt>;
        let mut lcm_den = BigInt::from(1);
        for coeff in &poly.coeffs {
            if coeff.is_zero() {
                continue;
            }
            let numer = coeff.numer().abs().clone();
            let denom = coeff.denom().clone();
            gcd_num = Some(match gcd_num {
                Some(existing) => existing.gcd(&numer),
                None => numer,
            });
            lcm_den = lcm_den.lcm(&denom);
        }

        match gcd_num {
            Some(g) => BigRational::new(g, lcm_den),
            None => BigRational::zero(),
        }
    };

    let num_content = rational_content(&num_poly);
    let den_content = rational_content(&den_poly);
    if num_content.is_zero() || den_content.is_zero() {
        return (num, den);
    }

    let mut new_num_poly = num_poly.clone();
    let mut new_den_poly = den_poly.clone();
    let mut changed = false;

    let common = if num_content.abs() == den_content.abs() {
        num_content.abs()
    } else {
        gcd_rational(num_content.abs(), den_content.abs())
    };
    if !common.is_zero() && !common.is_one() {
        new_num_poly = new_num_poly.div_scalar(&common);
        new_den_poly = new_den_poly.div_scalar(&common);
        changed = true;
    }

    if new_den_poly.leading_coeff() < BigRational::zero() {
        new_num_poly = new_num_poly.neg();
        new_den_poly = new_den_poly.neg();
        changed = true;
    }

    if !changed {
        return (num, den);
    }

    (new_num_poly.to_expr(ctx), new_den_poly.to_expr(ctx))
}

fn try_extract_abs_of_univariate_square_polynomial(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    use crate::expr_nary::{AddView, Sign};
    use crate::expr_rewrite::smart_mul;
    use crate::polynomial::Polynomial;

    fn build_mul(ctx: &mut Context, factors: Vec<ExprId>) -> ExprId {
        let mut iter = factors.into_iter();
        let Some(first) = iter.next() else {
            return ctx.num(1);
        };
        iter.fold(first, |acc, factor| ctx.add(Expr::Mul(acc, factor)))
    }

    fn build_signed_add(ctx: &mut Context, terms: &[(ExprId, Sign)]) -> ExprId {
        let mut iter = terms.iter();
        let Some((first_term, first_sign)) = iter.next() else {
            return ctx.num(0);
        };
        let mut acc = if *first_sign == Sign::Neg {
            ctx.add(Expr::Neg(*first_term))
        } else {
            *first_term
        };
        for (term, sign) in iter {
            acc = if *sign == Sign::Neg {
                ctx.add(Expr::Sub(acc, *term))
            } else {
                ctx.add(Expr::Add(acc, *term))
            };
        }
        acc
    }

    fn compare_signed_term_multiset(
        ctx: &Context,
        left: &[(ExprId, Sign)],
        right: &[(ExprId, Sign)],
    ) -> bool {
        fn terms_match(ctx: &Context, left: ExprId, right: ExprId) -> bool {
            if cas_ast::ordering::compare_expr(ctx, left, right) == std::cmp::Ordering::Equal {
                return true;
            }

            let left_factors = crate::expr_nary::mul_factors(ctx, left);
            let right_factors = crate::expr_nary::mul_factors(ctx, right);
            if left_factors.len() != right_factors.len() {
                return false;
            }

            let mut used = vec![false; right_factors.len()];
            for left_factor in left_factors {
                let mut matched = false;
                for (idx, right_factor) in right_factors.iter().enumerate() {
                    if used[idx] {
                        continue;
                    }
                    if cas_ast::ordering::compare_expr(ctx, left_factor, *right_factor)
                        == std::cmp::Ordering::Equal
                    {
                        used[idx] = true;
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    return false;
                }
            }

            true
        }

        if left.len() != right.len() {
            return false;
        }

        let mut used = vec![false; right.len()];
        for (left_term, left_sign) in left {
            let mut matched = false;
            for (idx, (right_term, right_sign)) in right.iter().enumerate() {
                if used[idx] || left_sign != right_sign {
                    continue;
                }
                if terms_match(ctx, *left_term, *right_term) {
                    used[idx] = true;
                    matched = true;
                    break;
                }
            }
            if !matched {
                return false;
            }
        }

        true
    }

    fn extract_term_degree_in_var(
        ctx: &mut Context,
        term: ExprId,
        var: &str,
    ) -> Option<(u32, ExprId)> {
        let mut degree = 0u32;
        let mut coeff_factors = Vec::new();

        for factor in crate::expr_nary::mul_factors(ctx, term) {
            match ctx.get(factor) {
                Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var => {
                    degree = degree.checked_add(1)?;
                }
                Expr::Pow(base, exp) => match (ctx.get(*base), ctx.get(*exp)) {
                    (Expr::Variable(sym_id), Expr::Number(n))
                        if ctx.sym_name(*sym_id) == var
                            && n.is_integer()
                            && *n >= BigRational::zero() =>
                    {
                        let power: u32 = n.to_integer().try_into().ok()?;
                        degree = degree.checked_add(power)?;
                    }
                    _ => {
                        if cas_ast::collect_variables(ctx, factor).contains(var) {
                            return None;
                        }
                        coeff_factors.push(factor);
                    }
                },
                _ => {
                    if cas_ast::collect_variables(ctx, factor).contains(var) {
                        return None;
                    }
                    coeff_factors.push(factor);
                }
            }
        }

        Some((degree, build_mul(ctx, coeff_factors)))
    }

    fn try_extract_abs_of_symbolic_shifted_unit_square(
        ctx: &mut Context,
        expr: ExprId,
        var: &str,
        var_expr: ExprId,
    ) -> Option<ExprId> {
        let mut quadratic_terms = Vec::new();
        let mut linear_terms = Vec::new();
        let mut constant_terms = Vec::new();

        for (term, sign) in AddView::from_expr(ctx, expr).terms {
            let (degree, coeff) = extract_term_degree_in_var(ctx, term, var)?;
            match degree {
                2 => quadratic_terms.push((coeff, sign)),
                1 => linear_terms.push((term, sign)),
                0 => constant_terms.push((term, sign)),
                _ => return None,
            }
        }

        if quadratic_terms.len() != 1 {
            return None;
        }
        let (quadratic_coeff, quadratic_sign) = quadratic_terms[0];
        let one = ctx.num(1);
        if quadratic_sign != Sign::Pos
            || cas_ast::ordering::compare_expr(ctx, quadratic_coeff, one)
                != std::cmp::Ordering::Equal
        {
            return None;
        }

        let constant_expr = build_signed_add(ctx, &constant_terms);
        let (lhs, rhs, rhs_is_sub) =
            crate::perfect_square_support::try_match_perfect_square_trinomial(ctx, constant_expr)?;
        let shift = if rhs_is_sub {
            ctx.add(Expr::Sub(lhs, rhs))
        } else {
            ctx.add(Expr::Add(lhs, rhs))
        };

        let two = ctx.num(2);
        let mut expected_same = Vec::new();
        let mut expected_negated = Vec::new();
        for (term, sign) in AddView::from_expr(ctx, shift).terms {
            let var_term = smart_mul(ctx, var_expr, term);
            let expected = smart_mul(ctx, two, var_term);
            expected_same.push((expected, sign));
            expected_negated.push((
                expected,
                if sign == Sign::Pos {
                    Sign::Neg
                } else {
                    Sign::Pos
                },
            ));
        }

        let inner = if compare_signed_term_multiset(ctx, &linear_terms, &expected_same) {
            ctx.add(Expr::Add(var_expr, shift))
        } else if compare_signed_term_multiset(ctx, &linear_terms, &expected_negated) {
            ctx.add(Expr::Sub(var_expr, shift))
        } else {
            return None;
        };

        Some(ctx.call_builtin(BuiltinFn::Abs, vec![inner]))
    }

    let vars = cas_ast::collect_variables(ctx, expr);
    if vars.len() != 1 {
        return None;
    }
    let var = vars.iter().next()?;
    let var_expr = ctx.var(var);

    if let Some(rewritten) =
        try_extract_abs_of_symbolic_shifted_unit_square(ctx, expr, var, var_expr)
    {
        return Some(rewritten);
    }

    let poly = Polynomial::from_expr(ctx, expr, var).ok()?;

    if poly.degree() == 2 && poly.coeffs.len() >= 3 {
        let a = poly.coeffs.get(2).cloned();
        let b = poly.coeffs.get(1).cloned();
        let c = poly.coeffs.first().cloned();
        if let (Some(a), Some(b), Some(c)) = (a, b, c) {
            let four = BigRational::from_integer(4.into());
            let discriminant = b.clone() * b.clone() - four * a.clone() * c.clone();
            if discriminant.is_zero() {
                if let (Some(d), Some(_)) = (rational_sqrt(&a), rational_sqrt(&c)) {
                    let two = BigRational::from_integer(2.into());
                    let e = if d.is_zero() {
                        rational_sqrt(&c).unwrap_or_else(BigRational::zero)
                    } else {
                        b.clone() / (two * d.clone())
                    };

                    let d_expr = ctx.add(Expr::Number(d.clone()));
                    let e_expr = ctx.add(Expr::Number(e.clone()));
                    let one = BigRational::from_integer(1.into());
                    let dx = if d == one {
                        var_expr
                    } else {
                        smart_mul(ctx, d_expr, var_expr)
                    };
                    let linear = if e.is_zero() {
                        dx
                    } else {
                        ctx.add(Expr::Add(dx, e_expr))
                    };
                    return Some(ctx.call_builtin(BuiltinFn::Abs, vec![linear]));
                }
            }
        }
    }

    if poly.degree() == 4 && poly.coeffs.len() >= 5 {
        let a2 = poly.coeffs.get(4).cloned()?;
        let b2 = poly.coeffs.get(3).cloned()?;
        let c2 = poly.coeffs.get(2).cloned()?;
        let d2 = poly.coeffs.get(1).cloned()?;
        let e2 = poly.coeffs.first().cloned()?;
        let two = BigRational::from_integer(2.into());
        let a = rational_sqrt(&a2)?;
        let c_abs = rational_sqrt(&e2)?;

        let c_candidates = if c_abs.is_zero() {
            vec![c_abs]
        } else {
            vec![c_abs.clone(), -c_abs]
        };

        for c in c_candidates {
            let b = if a.is_zero() {
                continue;
            } else {
                b2.clone() / (two.clone() * a.clone())
            };
            if d2 != two.clone() * b.clone() * c.clone() {
                continue;
            }
            if c2 != b.clone() * b.clone() + two.clone() * a.clone() * c.clone() {
                continue;
            }

            let mut terms = Vec::new();
            if !a.is_zero() {
                let a_expr = ctx.add(Expr::Number(a.clone()));
                let two_expr = ctx.num(2);
                let x_sq = ctx.add(Expr::Pow(var_expr, two_expr));
                terms.push(if a == BigRational::from_integer(1.into()) {
                    x_sq
                } else {
                    smart_mul(ctx, a_expr, x_sq)
                });
            }
            if !b.is_zero() {
                let b_expr = ctx.add(Expr::Number(b.clone()));
                terms.push(if b == BigRational::from_integer(1.into()) {
                    var_expr
                } else if b == BigRational::from_integer((-1).into()) {
                    ctx.add(Expr::Neg(var_expr))
                } else {
                    smart_mul(ctx, b_expr, var_expr)
                });
            }
            if !c.is_zero() {
                terms.push(ctx.add(Expr::Number(c.clone())));
            }
            if terms.is_empty() {
                continue;
            }

            let mut inner = terms[0];
            for term in terms.into_iter().skip(1) {
                inner = ctx.add(Expr::Add(inner, term));
            }
            return Some(ctx.call_builtin(BuiltinFn::Abs, vec![inner]));
        }
    }

    None
}

fn split_linear_surd(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, BigRational, BigRational)> {
    fn extract_coef_surd(ctx: &Context, term: ExprId) -> Option<(BigRational, BigRational)> {
        if let Some(radicand) = as_sqrt_like(ctx, term) {
            if let Expr::Number(n) = ctx.get(radicand) {
                return Some((BigRational::from_integer(1.into()), n.clone()));
            }
        }

        if let Expr::Mul(l, r) = ctx.get(term) {
            if let Expr::Number(b) = ctx.get(*l) {
                if let Some(radicand) = as_sqrt_like(ctx, *r) {
                    if let Expr::Number(n) = ctx.get(radicand) {
                        return Some((b.clone(), n.clone()));
                    }
                }
            }
            if let Expr::Number(b) = ctx.get(*r) {
                if let Some(radicand) = as_sqrt_like(ctx, *l) {
                    if let Expr::Number(n) = ctx.get(radicand) {
                        return Some((b.clone(), n.clone()));
                    }
                }
            }
        }
        None
    }

    match ctx.get(expr) {
        Expr::Add(l, r) => {
            if let Expr::Number(a) = ctx.get(*l) {
                if let Some((b, n)) = extract_coef_surd(ctx, *r) {
                    return Some((a.clone(), b, n));
                }
            }
            if let Expr::Number(a) = ctx.get(*r) {
                if let Some((b, n)) = extract_coef_surd(ctx, *l) {
                    return Some((a.clone(), b, n));
                }
            }
            if let Expr::Neg(neg_inner) = ctx.get(*r) {
                if let Expr::Number(a) = ctx.get(*l) {
                    if let Some((b, n)) = extract_coef_surd(ctx, *neg_inner) {
                        return Some((a.clone(), -b, n));
                    }
                }
            }
            None
        }
        Expr::Sub(l, r) => {
            if let Expr::Number(a) = ctx.get(*l) {
                if let Some((b, n)) = extract_coef_surd(ctx, *r) {
                    return Some((a.clone(), -b, n));
                }
            }
            if let Expr::Number(a) = ctx.get(*r) {
                if let Some((b, n)) = extract_coef_surd(ctx, *l) {
                    return Some((-a.clone(), b, n));
                }
            }
            None
        }
        _ => None,
    }
}

/// Reduce `expr` to the single quadratic-surd normal form `A + B·sqrt(n)` with
/// `A`, `B` exact rationals and `n` a non-negative rational, returning
/// `(A, B, n)`.
///
/// Handles rational constants, a single `sqrt(rational ≥ 0)` surd, the
/// rational-linear closure over it (`Neg`, `rational · surd`, `surd / rational`,
/// `Add`/`Sub` of compatible parts), and a rational base raised to a
/// half-integer power (so the reciprocal-surd spelling `b^(-1/2)` is handled).
/// Returns `None` for anything outside a single real quadratic surd — nested
/// radicals, two distinct radicands, surd products of distinct radicands, `sqrt`
/// of a negative or non-rational radicand, or any transcendental atom. Soundness
/// callers must treat `None` as "not provable" and never act on it.
pub fn as_linear_surd(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, BigRational, BigRational)> {
    fn normalize(
        a: BigRational,
        b: BigRational,
        n: BigRational,
    ) -> (BigRational, BigRational, BigRational) {
        if b.is_zero() {
            (a, BigRational::zero(), BigRational::zero())
        } else {
            (a, b, n)
        }
    }
    fn combine(
        l: (BigRational, BigRational, BigRational),
        r: (BigRational, BigRational, BigRational),
    ) -> Option<(BigRational, BigRational, BigRational)> {
        let a = &l.0 + &r.0;
        if l.1.is_zero() {
            return Some(normalize(a, r.1, r.2));
        }
        if r.1.is_zero() {
            return Some(normalize(a, l.1, l.2));
        }
        if l.2 != r.2 {
            return None; // two distinct surds: not a single quadratic surd
        }
        Some(normalize(a, &l.1 + &r.1, l.2))
    }

    // Pure rational constant -> A.
    if let Some(a) = crate::numeric_eval::as_rational_const(ctx, expr) {
        return Some((a, BigRational::zero(), BigRational::zero()));
    }
    // A single sqrt(rational >= 0) surd -> 1·sqrt(n).
    if let Some(radicand) = as_sqrt_like(ctx, expr) {
        let n = crate::numeric_eval::as_rational_const(ctx, radicand)?;
        if n.is_negative() {
            return None; // even root of a negative is not a real value
        }
        return Some(normalize(BigRational::zero(), BigRational::one(), n));
    }
    match ctx.get(expr) {
        Expr::Neg(inner) => {
            let (a, b, n) = as_linear_surd(ctx, *inner)?;
            Some((-a, -b, n))
        }
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            if let Some(c) = crate::numeric_eval::as_rational_const(ctx, l) {
                let (a, b, n) = as_linear_surd(ctx, r)?;
                return Some(normalize(&c * &a, &c * &b, n));
            }
            if let Some(c) = crate::numeric_eval::as_rational_const(ctx, r) {
                let (a, b, n) = as_linear_surd(ctx, l)?;
                return Some(normalize(&c * &a, &c * &b, n));
            }
            None // surd * surd would leave the quadratic field
        }
        Expr::Div(l, r) => {
            let (l, r) = (*l, *r);
            let c = crate::numeric_eval::as_rational_const(ctx, r)?;
            if c.is_zero() {
                return None;
            }
            let (a, b, n) = as_linear_surd(ctx, l)?;
            Some(normalize(&a / &c, &b / &c, n))
        }
        Expr::Add(l, r) => combine(as_linear_surd(ctx, *l)?, as_linear_surd(ctx, *r)?),
        Expr::Sub(l, r) => {
            let lhs = as_linear_surd(ctx, *l)?;
            let rhs = as_linear_surd(ctx, *r)?;
            combine(lhs, (-rhs.0, -rhs.1, rhs.2))
        }
        // A rational base raised to a HALF-INTEGER power stays inside ℚ(√b):
        // `b^(k/2) = b^((k−1)/2)·√b` for odd k (needs b ≥ 0), or a pure rational
        // for even k. This covers the reciprocal-surd spelling `b^(-1/2)` the
        // solver emits for the negative branch of a symmetric quadratic
        // (`-N·N^(-1/2) = -√N`). The plain `b^(1/2)` / `sqrt(b)` form is already
        // handled by the `as_sqrt_like` branch above.
        Expr::Pow(base, exp) => {
            fn ratio_pow(b: &BigRational, m: i32) -> BigRational {
                let mut acc = BigRational::one();
                for _ in 0..m.unsigned_abs() {
                    acc = &acc * b;
                }
                if m < 0 {
                    BigRational::one() / acc
                } else {
                    acc
                }
            }
            let e = crate::numeric_eval::as_rational_const(ctx, *exp)?;
            let b = crate::numeric_eval::as_rational_const(ctx, *base)?;
            let two_e = &e + &e;
            if !two_e.is_integer() {
                return None;
            }
            let k = two_e.to_integer().to_i64()?;
            if !(-64..=64).contains(&k) {
                return None; // keep the integer power small and bounded
            }
            if k % 2 == 0 {
                let m = (k / 2) as i32;
                if b.is_zero() && m < 0 {
                    return None; // 0 to a negative power is undefined
                }
                return Some((ratio_pow(&b, m), BigRational::zero(), BigRational::zero()));
            }
            if b.is_negative() {
                return None; // an even root of a negative is not a real surd
            }
            if b.is_zero() {
                return (k > 0).then(|| {
                    (
                        BigRational::zero(),
                        BigRational::zero(),
                        BigRational::zero(),
                    )
                });
            }
            let m = ((k - 1) / 2) as i32;
            Some(normalize(BigRational::zero(), ratio_pow(&b, m), b))
        }
        _ => None,
    }
}

/// Exact sign of `expr` versus zero, when `expr` reduces to a single quadratic
/// surd (see [`as_linear_surd`]). `Some(Less | Equal | Greater)` is a PROOF over
/// the reals; `None` means the sign could not be decided exactly and callers
/// must not act on it. Immune to floating-point error — e.g.
/// `93222358·sqrt(2) − 131836323` is correctly `Less` (never spuriously
/// `Equal`), because `A + B·sqrt(n)` with `B ≠ 0` and `n` not a perfect square
/// is irrational and thus never zero.
pub fn provable_sign_vs_zero(ctx: &Context, expr: ExprId) -> Option<std::cmp::Ordering> {
    let zero = BigRational::zero();
    let (a, b, n) = as_linear_surd(ctx, expr)?;
    // B == 0 (or n == 0): the value is the rational A.
    if b.is_zero() || n.is_zero() {
        return Some(a.cmp(&zero));
    }
    // A == 0: the value is B·sqrt(n); sign is sign(B) since sqrt(n) > 0 for n > 0.
    if a.is_zero() {
        return Some(b.cmp(&zero));
    }
    let sa = a.cmp(&zero);
    let sb = b.cmp(&zero);
    if sa == sb {
        // A and B share a sign -> the value has that sign.
        return Some(sa);
    }
    // Opposite signs: sign(A + B·sqrt(n)) = sign(B) · sign(B^2·n − A^2).
    let b2n = b.clone() * b.clone() * n.clone();
    let a2 = a.clone() * a.clone();
    let inner = b2n.cmp(&a2);
    Some(if b.is_negative() {
        inner.reverse()
    } else {
        inner
    })
}

/// `base^k` for `k ≥ 0` (exact rational power).
fn pow_rat_nonneg(base: &BigRational, k: i64) -> BigRational {
    let mut acc = BigRational::one();
    for _ in 0..k {
        acc *= base;
    }
    acc
}

/// PROVABLE rational lower/upper bounds for a CONSTANT expression built from rationals, `e`, `π`,
/// and their sums, differences, products, positive-integer powers, and division by a positive
/// rational. The constant bounds `2.718 < e < 2.719` and `3.141 < π < 3.142` are exact mathematical
/// facts and interval arithmetic preserves enclosure, so a comparison decided by these bounds is
/// EXACT (never an f64 estimate). `None` for any shape outside this grammar (a free variable, a
/// negative-base power, division by a non-positive value, …).
fn rational_bounds(ctx: &Context, expr: ExprId) -> Option<(BigRational, BigRational)> {
    use crate::numeric_eval::as_rational_const;
    let r = |n: i64, d: i64| BigRational::new(n.into(), d.into());
    if let Some(v) = as_rational_const(ctx, expr) {
        return Some((v.clone(), v));
    }
    match ctx.get(expr) {
        Expr::Constant(Constant::E) => Some((r(2718, 1000), r(2719, 1000))),
        Expr::Constant(Constant::Pi) => Some((r(3141, 1000), r(3142, 1000))),
        Expr::Neg(a) => {
            let (lo, hi) = rational_bounds(ctx, *a)?;
            Some((-hi, -lo))
        }
        Expr::Add(a, b) => {
            let (al, ah) = rational_bounds(ctx, *a)?;
            let (bl, bh) = rational_bounds(ctx, *b)?;
            Some((al + bl, ah + bh))
        }
        Expr::Sub(a, b) => {
            let (al, ah) = rational_bounds(ctx, *a)?;
            let (bl, bh) = rational_bounds(ctx, *b)?;
            Some((al - bh, ah - bl))
        }
        Expr::Mul(a, b) => {
            let (al, ah) = rational_bounds(ctx, *a)?;
            let (bl, bh) = rational_bounds(ctx, *b)?;
            let prods = [&al * &bl, &al * &bh, &ah * &bl, &ah * &bh];
            let lo = prods.iter().min().cloned()?;
            let hi = prods.iter().max().cloned()?;
            Some((lo, hi))
        }
        Expr::Pow(base, exp) => {
            let n = as_rational_const(ctx, *exp)?;
            if !n.is_integer() {
                return None;
            }
            let k = n.to_integer().to_i64()?;
            if !(0..=16).contains(&k) {
                return None; // keep the power small/bounded
            }
            let (bl, bh) = rational_bounds(ctx, *base)?;
            if bl.is_negative() {
                return None; // only a nonnegative base, to avoid even/odd sign juggling
            }
            Some((pow_rat_nonneg(&bl, k), pow_rat_nonneg(&bh, k)))
        }
        Expr::Div(a, b) => {
            let c = as_rational_const(ctx, *b)?;
            if !c.is_positive() {
                return None;
            }
            let (al, ah) = rational_bounds(ctx, *a)?;
            Some((al / &c, ah / &c))
        }
        _ => None,
    }
}

/// Provable sign of `expr − threshold` for a constant `expr` (decided by exact rational bounds).
fn provable_const_minus_rational_sign(
    ctx: &Context,
    expr: ExprId,
    threshold: &BigRational,
) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering;
    let (lo, hi) = rational_bounds(ctx, expr)?;
    if &lo > threshold {
        return Some(Ordering::Greater);
    }
    if &hi < threshold {
        return Some(Ordering::Less);
    }
    if &lo == threshold && &hi == threshold {
        return Some(Ordering::Equal);
    }
    None
}

/// Like [`as_linear_surd`] but the radicand may be ANY constant expression (e.g. `9 + 4e`), returned
/// as `(A, B, radicand)` for `A + B·√radicand` (`radicand = None` when `B = 0`). Only the `√`
/// (`as_sqrt_like`) spelling of the surd is decomposed.
fn as_linear_surd_expr(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, BigRational, Option<ExprId>)> {
    use crate::numeric_eval::as_rational_const;
    fn combine(
        ctx: &Context,
        l: (BigRational, BigRational, Option<ExprId>),
        r: (BigRational, BigRational, Option<ExprId>),
    ) -> Option<(BigRational, BigRational, Option<ExprId>)> {
        let a = &l.0 + &r.0;
        if l.1.is_zero() {
            return Some((a, r.1.clone(), if r.1.is_zero() { None } else { r.2 }));
        }
        if r.1.is_zero() {
            return Some((a, l.1, l.2));
        }
        let (lr, rr) = (l.2?, r.2?);
        if compare_expr(ctx, lr, rr) != std::cmp::Ordering::Equal {
            return None; // two distinct surds: not a single quadratic surd
        }
        let b = &l.1 + &r.1;
        Some((a, b.clone(), if b.is_zero() { None } else { Some(lr) }))
    }
    if let Some(a) = as_rational_const(ctx, expr) {
        return Some((a, BigRational::zero(), None));
    }
    if let Some(radicand) = as_sqrt_like(ctx, expr) {
        return Some((BigRational::zero(), BigRational::one(), Some(radicand)));
    }
    match ctx.get(expr) {
        Expr::Neg(inner) => {
            let (a, b, rad) = as_linear_surd_expr(ctx, *inner)?;
            Some((-a, -b, rad))
        }
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            if let Some(c) = as_rational_const(ctx, l) {
                let (a, b, rad) = as_linear_surd_expr(ctx, r)?;
                return Some((&c * &a, &c * &b, rad));
            }
            if let Some(c) = as_rational_const(ctx, r) {
                let (a, b, rad) = as_linear_surd_expr(ctx, l)?;
                return Some((&c * &a, &c * &b, rad));
            }
            None
        }
        Expr::Div(l, r) => {
            let (l, r) = (*l, *r);
            let c = as_rational_const(ctx, r)?;
            if c.is_zero() {
                return None;
            }
            let (a, b, rad) = as_linear_surd_expr(ctx, l)?;
            Some((&a / &c, &b / &c, rad))
        }
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            combine(
                ctx,
                as_linear_surd_expr(ctx, l)?,
                as_linear_surd_expr(ctx, r)?,
            )
        }
        Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            let rhs = as_linear_surd_expr(ctx, r)?;
            combine(ctx, as_linear_surd_expr(ctx, l)?, (-rhs.0, -rhs.1, rhs.2))
        }
        _ => None,
    }
}

/// Provable sign vs 0 of a quadratic surd `A + B·√R` whose radicand `R` is a constant expression
/// that may involve `e`/`π` (the transcendental case [`provable_sign_vs_zero`] declines, since it
/// needs a RATIONAL radicand). Decides the sign by `sign(A + B·√R) = sign(B)·sign(R − (A/B)²)` (for
/// opposite signs of `A`, `B`), proving the radicand comparison with [`provable_const_minus_rational_sign`].
/// EXACT: never a float estimate — `None` when the comparison is not provable, so a valid root is
/// never dropped.
pub fn provable_sign_vs_zero_const_radicand(
    ctx: &Context,
    expr: ExprId,
) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering;
    let zero = BigRational::zero();
    let (a, b, radicand) = as_linear_surd_expr(ctx, expr)?;
    let Some(radicand) = radicand else {
        return Some(a.cmp(&zero)); // pure rational
    };
    // √R requires R > 0 (and √R > 0); decline when R's positivity is not provable.
    if provable_const_minus_rational_sign(ctx, radicand, &zero)? != Ordering::Greater {
        return None;
    }
    if b.is_zero() {
        return Some(a.cmp(&zero));
    }
    if a.is_zero() {
        return Some(b.cmp(&zero)); // B·√R, R > 0
    }
    let sa = a.cmp(&zero);
    let sb = b.cmp(&zero);
    if sa == sb {
        return Some(sa);
    }
    // Opposite signs: sign(A + B·√R) = sign(B) · sign(R − (A/B)²).
    let threshold = (&a * &a) / (&b * &b);
    let inner = provable_const_minus_rational_sign(ctx, radicand, &threshold)?;
    Some(if b.is_negative() {
        inner.reverse()
    } else {
        inner
    })
}

fn solve_cube_in_quadratic_field(
    a: &BigRational,
    b: &BigRational,
    n: &BigRational,
) -> Option<(BigRational, BigRational)> {
    use num_bigint::BigInt;
    use num_traits::Zero;

    if n <= &BigRational::zero() {
        return None;
    }

    let a_approx: f64 = a.numer().to_string().parse().unwrap_or(f64::MAX);
    let b_approx: f64 = b.numer().to_string().parse().unwrap_or(f64::MAX);
    if a_approx.abs() > 1e12 || b_approx.abs() > 1e12 {
        return None;
    }

    let denoms: [i64; 7] = [1, 2, 3, 4, 6, 8, 12];
    let max_num: i64 = 10;
    let three = BigRational::from_integer(3.into());

    for &denom in &denoms {
        let denom_big = BigInt::from(denom);
        for num in -max_num..=max_num {
            if num == 0 {
                continue;
            }

            let y = BigRational::new(BigInt::from(num), denom_big.clone());
            let y_squared = &y * &y;
            let y_cubed = &y_squared * &y;

            let b_over_y = b / &y;
            let n_y_sq = n * &y_squared;
            let x_squared = (&b_over_y - &n_y_sq) / &three;
            if x_squared.is_negative() {
                continue;
            }

            if let Some(x_pos) = rational_sqrt(&x_squared) {
                for x in [x_pos.clone(), -x_pos.clone()] {
                    let x_cubed = &x * &x * &x;
                    let term_3xy2n = &three * &x * &y_squared * n;
                    let lhs_a = &x_cubed + &term_3xy2n;

                    let x_sq = &x * &x;
                    let term_3x2y = &three * &x_sq * &y;
                    let term_y3n = &y_cubed * n;
                    let lhs_b = &term_3x2y + &term_y3n;

                    if &lhs_a == a && &lhs_b == b {
                        return Some((x, y));
                    }
                }
            }
        }
    }
    None
}

/// Rewrite perfect cube roots in quadratic fields:
/// - `(A + B*sqrt(n))^(1/3) -> x + y*sqrt(n)` when rational `x,y` satisfy
///   `(x + y*sqrt(n))^3 = A + B*sqrt(n)`.
pub fn try_rewrite_denest_cube_quadratic_field_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<DenestCubeQuadraticRewrite> {
    use num_traits::Zero;

    let base = extract_cube_root_base(ctx, expr)?;

    let (a, b, n) = split_linear_surd(ctx, base)?;
    if b.is_zero() {
        return None;
    }

    let (x, y) = solve_cube_in_quadratic_field(&a, &b, &n)?;

    let x_expr = ctx.add(Expr::Number(x.clone()));
    let y_expr = ctx.add(Expr::Number(y.clone()));
    let n_expr = ctx.add(Expr::Number(n.clone()));
    let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
    let sqrt_n = ctx.add(Expr::Pow(n_expr, half));

    let rewritten = if y.is_zero() {
        x_expr
    } else if x.is_zero() {
        ctx.add(Expr::Mul(y_expr, sqrt_n))
    } else {
        let y_sqrt_n = ctx.add(Expr::Mul(y_expr, sqrt_n));
        ctx.add(Expr::Add(x_expr, y_sqrt_n))
    };

    Some(DenestCubeQuadraticRewrite {
        rewritten,
        desc: format!(
            "Denest cube root in quadratic field: ∛(A+B√n) = {} + {}√{}",
            x, y, n
        ),
    })
}

/// Rewrite `∛(m+t) + ∛(m-t)` when it evaluates to a rational root.
pub fn try_rewrite_cubic_conjugate_identity_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<CubicConjugateTrapRewrite> {
    let (left, right) = match ctx.get(expr) {
        Expr::Add(l, r) => (*l, *r),
        _ => return None,
    };

    let base_a = extract_cube_root_base(ctx, left)?;
    let base_b = extract_cube_root_base(ctx, right)?;
    let (m, t) = conjugate_numeric_surd_pair(ctx, base_a, base_b)?;

    let two = BigRational::from_integer(2.into());
    let m_value = match ctx.get(m) {
        Expr::Number(n) => n.clone(),
        _ => return None,
    };
    let s_value = &two * &m_value;

    let t_squared_value = surd_square_rational(ctx, t)?;
    let ab_value = &m_value * &m_value - &t_squared_value;
    let p_value = rational_cbrt_exact(&ab_value)?;

    let three = BigRational::from_integer(3.into());
    let p_coef = -&three * &p_value;
    let q_coef = -&s_value;

    if p_coef <= BigRational::zero() {
        return None;
    }

    let root = find_rational_root_depressed_cubic(&p_coef, &q_coef)?;
    let rewritten = ctx.add(Expr::Number(root.clone()));
    Some(CubicConjugateTrapRewrite {
        rewritten,
        desc: format!("Cubic conjugate identity: ∛(m+t) + ∛(m-t) = {}", root),
    })
}

/// Rewrite root syntax into canonical power syntax.
///
/// Supported rewrites:
/// - `sqrt(x^2k)` -> `|x|^k`
/// - `sqrt(x)` -> `x^(1/2)`
/// - `sqrt(x, n)` -> `x^(1/n)` (numeric index only)
/// - `root(x, n)` -> `x^(1/n)` (numeric index only)
pub fn try_rewrite_canonical_root_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<CanonicalRootRewrite> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    let fn_id = *fn_id;
    let args = args.clone();

    if ctx.is_builtin(fn_id, BuiltinFn::Sqrt) {
        if args.len() == 1 {
            let arg = args[0];

            if let Expr::Pow(base, exp) = ctx.get(arg) {
                let (base, exp) = (*base, *exp);
                if let Expr::Number(n) = ctx.get(exp) {
                    if n.is_integer() && n.to_integer().is_even() {
                        let two = ctx.num(2);
                        let k = ctx.add(Expr::Div(exp, two));
                        let abs_base = ctx.call_builtin(BuiltinFn::Abs, vec![base]);
                        let rewritten = ctx.add(Expr::Pow(abs_base, k));
                        return Some(CanonicalRootRewrite {
                            rewritten,
                            kind: CanonicalRootRewriteKind::SqrtEvenPower,
                        });
                    }
                }
            }

            let half = ctx.rational(1, 2);
            let rewritten = ctx.add(Expr::Pow(arg, half));
            return Some(CanonicalRootRewrite {
                rewritten,
                kind: CanonicalRootRewriteKind::SqrtUnary,
            });
        }

        if args.len() == 2 {
            let (base_arg, index) = (args[0], args[1]);
            if !matches!(ctx.get(index), Expr::Number(_)) {
                return None;
            }
            let one = ctx.num(1);
            let exp = ctx.add(Expr::Div(one, index));
            let rewritten = ctx.add(Expr::Pow(base_arg, exp));
            return Some(CanonicalRootRewrite {
                rewritten,
                kind: CanonicalRootRewriteKind::SqrtWithIndex,
            });
        }
    }

    if ctx.is_builtin(fn_id, BuiltinFn::Root) && args.len() == 2 {
        let (base_arg, index) = (args[0], args[1]);
        if !matches!(ctx.get(index), Expr::Number(_)) {
            return None;
        }
        let one = ctx.num(1);
        let exp = ctx.add(Expr::Div(one, index));
        let rewritten = ctx.add(Expr::Pow(base_arg, exp));
        return Some(CanonicalRootRewrite {
            rewritten,
            kind: CanonicalRootRewriteKind::RootWithIndex,
        });
    }

    None
}

/// Rewrite odd half-integer powers into absolute-value times square-root form.
///
/// Pattern:
/// - `x^(n/2)` with odd integer `n >= 3`
///
/// Result:
/// - `|x|^k * sqrt(x)` where `k = (n-1)/2`
pub fn try_rewrite_odd_half_power_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<OddHalfPowerRewrite> {
    let (base, exp) = match ctx.get(expr) {
        Expr::Pow(base, exp) => (*base, *exp),
        _ => return None,
    };

    let Expr::Number(exp_num) = ctx.get(exp) else {
        return None;
    };
    let numer = exp_num.numer().to_i64()?;
    let denom = exp_num.denom().to_i64()?;
    if denom != 2 || numer < 3 || numer % 2 == 0 {
        return None;
    }

    let k = (numer - 1) / 2;
    let abs_base = ctx.call_builtin(BuiltinFn::Abs, vec![base]);
    let sqrt_base = ctx.call_builtin(BuiltinFn::Sqrt, vec![base]);
    let rewritten = if k == 1 {
        ctx.add(Expr::Mul(abs_base, sqrt_base))
    } else {
        let k_expr = ctx.num(k);
        let abs_pow_k = ctx.add(Expr::Pow(abs_base, k_expr));
        ctx.add(Expr::Mul(abs_pow_k, sqrt_base))
    };

    Some(OddHalfPowerRewrite {
        rewritten,
        numerator: numer,
        abs_power: k,
    })
}

/// Extract `(radicand, index)` when `expr` is a root-like form.
///
/// Recognizes:
/// - `sqrt(x)` as `(x, 2)`
/// - `x^(1/k)` where exponent is numeric `1/k`
/// - `x^(1/k)` where exponent is structural `Div(1, k)`
pub fn extract_root_base_and_index(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let sqrt_arg = match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            Some(args[0])
        }
        _ => None,
    };
    if let Some(arg) = sqrt_arg {
        return Some((arg, ctx.num(2)));
    }

    let (base, exp) = match ctx.get(expr) {
        Expr::Pow(base, exp) => (*base, *exp),
        _ => return None,
    };

    if let Some(n) = crate::numeric::as_number(ctx, exp) {
        if !n.is_integer() && n.numer().is_one() {
            let k_expr = ctx.add(Expr::Number(BigRational::from_integer(n.denom().clone())));
            return Some((base, k_expr));
        }
    }

    if let Expr::Div(num_exp, den_exp) = ctx.get(exp) {
        if let Some(n) = crate::numeric::as_number(ctx, *num_exp) {
            if n.is_one() {
                return Some((base, *den_exp));
            }
        }
    }

    None
}

/// Extract positive integer `n` when `expr` is a numeric square root.
///
/// Recognizes:
/// - `sqrt(n)`
/// - `n^(1/2)` (including equivalent rational constants)
pub fn extract_numeric_sqrt_radicand(ctx: &Context, expr: ExprId) -> Option<i64> {
    use cas_ast::views::as_rational_const;

    let base = match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            args[0]
        }
        Expr::Pow(base, exp) => {
            let exp_val = as_rational_const(ctx, *exp, 8)?;
            let half = BigRational::new(1.into(), 2.into());
            if exp_val != half {
                return None;
            }
            *base
        }
        _ => return None,
    };

    if let Expr::Number(n) = ctx.get(base) {
        if n.is_integer() {
            return n.numer().to_i64().filter(|&x| x > 0);
        }
    }
    None
}

/// Extracts perfect `k`-th-power factors from integer `n`.
///
/// Returns `(outside, inside)` such that:
/// - `outside^k * inside == n`
/// - `inside` has no remaining `k`-th-power factors
pub fn extract_root_factor(n: &BigInt, k: u32) -> (BigInt, BigInt) {
    if n.is_zero() {
        return (BigInt::zero(), BigInt::one());
    }
    if n.is_one() {
        return (BigInt::one(), BigInt::one());
    }

    let sign = if n.is_negative() { -1 } else { 1 };
    let mut n_abs = n.abs();

    let mut outside = BigInt::one();
    let mut inside = BigInt::one();

    // Trial division for factor 2 first.
    let mut count = 0;
    while n_abs.is_even() {
        count += 1;
        n_abs /= 2;
    }
    if count > 0 {
        let out_exp = count / k;
        let in_exp = count % k;
        if out_exp > 0 {
            outside *= BigInt::from(2).pow(out_exp);
        }
        if in_exp > 0 {
            inside *= BigInt::from(2).pow(in_exp);
        }
    }

    let mut d = BigInt::from(3);
    while &d * &d <= n_abs {
        if (&n_abs % &d).is_zero() {
            let mut count = 0;
            while (&n_abs % &d).is_zero() {
                count += 1;
                n_abs /= &d;
            }
            let out_exp = count / k;
            let in_exp = count % k;
            if out_exp > 0 {
                outside *= d.pow(out_exp);
            }
            if in_exp > 0 {
                inside *= d.pow(in_exp);
            }
        }
        d += 2;
    }

    if n_abs > BigInt::one() {
        inside *= n_abs;
    }

    // Preserve sign: odd roots keep sign outside, even roots keep sign inside.
    if sign == -1 {
        if !k.is_multiple_of(2) {
            outside = -outside;
        } else {
            inside = -inside;
        }
    }

    (outside, inside)
}

/// Check if distributing a fractional root exponent (`1/n`) over `expr` is safe.
///
/// Safe cases:
/// - Purely numeric subexpressions.
/// - Symbolic factors whose exponents are integer multiples of `n`.
pub fn can_distribute_root_safely(ctx: &Context, expr: ExprId, root_index: &BigInt) -> bool {
    match ctx.get(expr) {
        Expr::Number(_) => true,
        Expr::Variable(_) | Expr::Constant(_) => root_index == &BigInt::from(1),
        Expr::Pow(base, exp) => {
            if is_purely_numeric(ctx, *base) {
                return true;
            }
            if let Expr::Number(exp_num) = ctx.get(*exp) {
                if exp_num.is_integer() {
                    let exp_int = exp_num.to_integer();
                    return (&exp_int % root_index).is_zero();
                }
            }
            false
        }
        Expr::Mul(l, r) | Expr::Div(l, r) => {
            can_distribute_root_safely(ctx, *l, root_index)
                && can_distribute_root_safely(ctx, *r, root_index)
        }
        Expr::Neg(inner) => can_distribute_root_safely(ctx, *inner, root_index),
        _ => false,
    }
}

fn is_purely_numeric(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(_) => true,
        Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => false,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            is_purely_numeric(ctx, *l) && is_purely_numeric(ctx, *r)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => is_purely_numeric(ctx, *inner),
        Expr::Function(_, args) => args.iter().all(|a| is_purely_numeric(ctx, *a)),
        Expr::Matrix { data, .. } => data.iter().all(|e| is_purely_numeric(ctx, *e)),
    }
}

fn is_surd_like(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            true
        }
        Expr::Pow(_, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                *n.numer() == 1.into() && *n.denom() == 2.into()
            } else {
                false
            }
        }
        Expr::Mul(l, r) => is_surd_like(ctx, *l) || is_surd_like(ctx, *r),
        Expr::Neg(inner) => is_surd_like(ctx, *inner),
        _ => false,
    }
}

/// Split an expression as `m ± t`, where `m` is numeric and `t` is surd-like.
///
/// Returns `(m, t, sign)`, where `sign = +1` means `m+t` and `sign = -1` means `m-t`.
pub fn split_numeric_plus_surd(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId, i32)> {
    let is_numeric = |e: ExprId| matches!(ctx.get(e), Expr::Number(_));

    match ctx.get(expr) {
        Expr::Add(l, r) => {
            if let Expr::Neg(neg_inner) = ctx.get(*r) {
                if is_numeric(*l) && is_surd_like(ctx, *neg_inner) {
                    return Some((*l, *neg_inner, -1));
                }
                if is_surd_like(ctx, *l) && is_numeric(*neg_inner) {
                    return None;
                }
            }

            if is_numeric(*l) && is_surd_like(ctx, *r) {
                return Some((*l, *r, 1));
            }
            if is_surd_like(ctx, *l) && is_numeric(*r) {
                return Some((*r, *l, 1));
            }
            None
        }
        Expr::Sub(l, r) => {
            if is_numeric(*l) && is_surd_like(ctx, *r) {
                return Some((*l, *r, -1));
            }
            if is_surd_like(ctx, *l) && is_numeric(*r) {
                return None;
            }
            None
        }
        _ => None,
    }
}

/// Check whether two expressions are conjugates in the `m ± t` form.
///
/// Returns `(m, t)` when both sides share the same `m` and `t` and opposite sign.
pub fn conjugate_numeric_surd_pair(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, ExprId)> {
    use cas_ast::ordering::compare_expr;
    use std::cmp::Ordering;

    let (m1, t1, sign1) = split_numeric_plus_surd(ctx, left)?;
    let (m2, t2, sign2) = split_numeric_plus_surd(ctx, right)?;

    if compare_expr(ctx, m1, m2) != Ordering::Equal {
        return None;
    }
    if compare_expr(ctx, t1, t2) != Ordering::Equal {
        return None;
    }
    if sign1 + sign2 != 0 {
        return None;
    }

    Some((m1, t1))
}

/// Extract base from `Pow(base, 1/3)`.
pub fn extract_cube_root_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        match ctx.get(*exp) {
            Expr::Number(n) => {
                if *n.numer() == 1.into() && *n.denom() == 3.into() {
                    return Some(*base);
                }
            }
            Expr::Div(num, den) => {
                if let (Expr::Number(n_num), Expr::Number(n_den)) = (ctx.get(*num), ctx.get(*den)) {
                    if n_num.is_one() && n_den.is_integer() && *n_den.numer() == 3.into() {
                        return Some(*base);
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// Exact real cube-root for rational inputs, if the input is a perfect cube.
pub fn rational_cbrt_exact(r: &BigRational) -> Option<BigRational> {
    let neg = r.is_negative();
    let abs_r = if neg { -r.clone() } else { r.clone() };

    if abs_r.is_zero() {
        return Some(BigRational::from_integer(0.into()));
    }

    let numer = abs_r.numer().clone();
    let denom = abs_r.denom().clone();

    let numer_cbrt = numer.cbrt();
    if &numer_cbrt * &numer_cbrt * &numer_cbrt != numer {
        return None;
    }

    let denom_cbrt = denom.cbrt();
    if &denom_cbrt * &denom_cbrt * &denom_cbrt != denom {
        return None;
    }

    let result = BigRational::new(numer_cbrt, denom_cbrt);
    if neg {
        Some(-result)
    } else {
        Some(result)
    }
}

/// Exact rational `n`-th root of `r`, or `None` when it is not rational. An even
/// root of a negative is undefined over the reals (returns `None`); an odd root
/// keeps the sign. Generalizes `rational_sqrt` (n=2) and `rational_cbrt_exact`
/// (n=3).
pub fn rational_nth_root(r: &BigRational, n: u32) -> Option<BigRational> {
    if n == 0 {
        return None;
    }
    if n == 1 {
        return Some(r.clone());
    }
    let neg = r.is_negative();
    if neg && n.is_multiple_of(2) {
        return None; // even root of a negative is not real
    }
    let abs_r = if neg { -r.clone() } else { r.clone() };
    if abs_r.is_zero() {
        return Some(BigRational::from_integer(0.into()));
    }
    let numer = abs_r.numer().clone();
    let denom = abs_r.denom().clone();
    let numer_root = numer.nth_root(n);
    if num_traits::pow(numer_root.clone(), n as usize) != numer {
        return None;
    }
    let denom_root = denom.nth_root(n);
    if num_traits::pow(denom_root.clone(), n as usize) != denom {
        return None;
    }
    let result = BigRational::new(numer_root, denom_root);
    Some(if neg { -result } else { result })
}

/// Compute `t²` when `t` is numeric, `sqrt(d)`, `d^(1/2)`, or `k*sqrt(d)`.
pub fn surd_square_rational(ctx: &Context, t: ExprId) -> Option<BigRational> {
    match ctx.get(t) {
        Expr::Number(n) => Some(n * n),
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            if let Expr::Number(d) = ctx.get(args[0]) {
                Some(d.clone())
            } else {
                None
            }
        }
        Expr::Pow(base, exp) => {
            if let Expr::Number(e) = ctx.get(*exp) {
                if *e.numer() == 1.into() && *e.denom() == 2.into() {
                    if let Expr::Number(d) = ctx.get(*base) {
                        return Some(d.clone());
                    }
                }
            }
            None
        }
        Expr::Mul(l, r) => {
            let try_extract = |coef: ExprId, surd: ExprId| -> Option<BigRational> {
                let k = if let Expr::Number(n) = ctx.get(coef) {
                    n.clone()
                } else {
                    return None;
                };

                let d = match ctx.get(surd) {
                    Expr::Function(fn_id, args)
                        if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 =>
                    {
                        if let Expr::Number(n) = ctx.get(args[0]) {
                            n.clone()
                        } else {
                            return None;
                        }
                    }
                    Expr::Pow(base, exp) => {
                        if let Expr::Number(e) = ctx.get(*exp) {
                            if *e.numer() == 1.into() && *e.denom() == 2.into() {
                                if let Expr::Number(n) = ctx.get(*base) {
                                    n.clone()
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

                Some(&k * &k * &d)
            };

            try_extract(*l, *r).or_else(|| try_extract(*r, *l))
        }
        _ => None,
    }
}

/// Find a rational root of the depressed cubic `x³ + p·x + q = 0`.
///
/// Uses Rational Root Theorem after clearing denominators.
pub fn find_rational_root_depressed_cubic(p: &BigRational, q: &BigRational) -> Option<BigRational> {
    use num_bigint::BigInt;

    if q.is_zero() {
        return Some(BigRational::zero());
    }

    let lcm_denom = num_integer::lcm(p.denom().clone(), q.denom().clone());
    let leading_coef = lcm_denom.clone();
    let constant_coef = q * BigRational::from_integer(lcm_denom.clone());
    let constant_int = constant_coef.to_integer();

    let c_abs = if constant_int.is_negative() {
        -constant_int.clone()
    } else {
        constant_int.clone()
    };
    let a_abs = if leading_coef.is_negative() {
        -leading_coef.clone()
    } else {
        leading_coef.clone()
    };

    fn small_divisors(n: &BigInt, limit: i64) -> Vec<BigInt> {
        let mut divs = Vec::new();
        if n.is_zero() {
            return vec![BigInt::from(1)];
        }
        let n_abs = if n.is_negative() {
            -n.clone()
        } else {
            n.clone()
        };
        for d in 1..=limit {
            let bd = BigInt::from(d);
            if &n_abs % &bd == BigInt::zero() {
                divs.push(bd.clone());
                let quotient = &n_abs / &bd;
                if !divs.contains(&quotient) {
                    divs.push(quotient);
                }
            }
        }
        if divs.is_empty() {
            divs.push(BigInt::from(1));
        }
        divs
    }

    let c_divisors = small_divisors(&c_abs, 50);
    let a_divisors = small_divisors(&a_abs, 20);

    for d in &c_divisors {
        for e in &a_divisors {
            for sign in &[1i32, -1i32] {
                let candidate = if *sign == 1 {
                    BigRational::new(d.clone(), e.clone())
                } else {
                    -BigRational::new(d.clone(), e.clone())
                };

                let x2 = &candidate * &candidate;
                let x3 = &x2 * &candidate;
                let val = &x3 + p * &candidate + q;

                if val.is_zero() {
                    return Some(candidate);
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly_compare::poly_eq;
    use cas_parser::parse;

    #[test]
    fn provable_sign_const_radicand_decides_transcendental_surds() {
        use std::cmp::Ordering;
        let mut ctx = Context::new();
        // `(3 − √(9+4e))/2 < 0` (the extraneous root of `ln(x)+ln(x-3)=1`): proven because
        // `√(9+4e) > 3 ⟺ 9+4e > 9 ⟺ 4e > 0`.
        let neg = parse("(3 - sqrt(9 + 4*e))/2", &mut ctx).expect("neg root");
        assert_eq!(
            provable_sign_vs_zero_const_radicand(&ctx, neg),
            Some(Ordering::Less)
        );
        // `(3 + √(9+4e))/2 > 0` (the valid root): both terms positive.
        let pos = parse("(3 + sqrt(9 + 4*e))/2", &mut ctx).expect("pos root");
        assert_eq!(
            provable_sign_vs_zero_const_radicand(&ctx, pos),
            Some(Ordering::Greater)
        );
        // `√π − 1 > 0` (π > 1, so √π > 1), decided by the rational bound `π > 3.141`.
        let sp = parse("sqrt(pi) - 1", &mut ctx).expect("sqrt pi - 1");
        assert_eq!(
            provable_sign_vs_zero_const_radicand(&ctx, sp),
            Some(Ordering::Greater)
        );
        // `2 − √(e²+1) < 0` (`e² > 3`, decided by `e > 2.718` ⇒ `e² > 7.38`).
        let sq = parse("2 - sqrt(e^2 + 1)", &mut ctx).expect("sqrt(e^2+1)");
        assert_eq!(
            provable_sign_vs_zero_const_radicand(&ctx, sq),
            Some(Ordering::Less)
        );
        // A radicand that is provably NEGATIVE (`e − π < 0`) is not a real surd → decline (keep).
        let mixed = parse("sqrt(e - pi)", &mut ctx).expect("sqrt(e-pi)");
        assert_eq!(provable_sign_vs_zero_const_radicand(&ctx, mixed), None);
    }

    #[test]
    fn as_linear_surd_extracts_and_signs_exactly() {
        use std::cmp::Ordering;
        let mut ctx = Context::new();

        let rat = |a: i64, b: i64| BigRational::new(a.into(), b.into());

        // sqrt(29) -> 0 + 1·sqrt(29).
        let s29 = parse("sqrt(29)", &mut ctx).expect("sqrt29");
        assert_eq!(
            as_linear_surd(&ctx, s29),
            Some((rat(0, 1), rat(1, 1), rat(29, 1)))
        );

        // 1/2·(-sqrt(29) - 5) -> -5/2 - 1/2·sqrt(29); strictly negative.
        let r_neg = parse("1/2*(-sqrt(29) - 5)", &mut ctx).expect("rneg");
        assert_eq!(
            as_linear_surd(&ctx, r_neg),
            Some((rat(-5, 2), rat(-1, 2), rat(29, 1)))
        );
        assert_eq!(provable_sign_vs_zero(&ctx, r_neg), Some(Ordering::Less));

        // 1/2·(sqrt(29) - 5): A=-5/2, B=1/2; B^2·29=29/4 > A^2=25/4 -> positive.
        let r_pos = parse("1/2*(sqrt(29) - 5)", &mut ctx).expect("rpos");
        assert_eq!(provable_sign_vs_zero(&ctx, r_pos), Some(Ordering::Greater));

        // sqrt(x)*sqrt(x-1)=2 roots: 1/2·(1 - sqrt(17)) < 0, 1/2·(1 + sqrt(17)) > 0.
        let s_lo = parse("1/2*(1 - sqrt(17))", &mut ctx).expect("slo");
        let s_hi = parse("1/2*(1 + sqrt(17))", &mut ctx).expect("shi");
        assert_eq!(provable_sign_vs_zero(&ctx, s_lo), Some(Ordering::Less));
        assert_eq!(provable_sign_vs_zero(&ctx, s_hi), Some(Ordering::Greater));

        // For ln(x)+ln(x+5)=0, the extraneous root makes x+5 negative too.
        let xp5_at_lo = parse("1/2*(-sqrt(29) - 5) + 5", &mut ctx).expect("xp5");
        assert_eq!(provable_sign_vs_zero(&ctx, xp5_at_lo), Some(Ordering::Less));

        // The adversarial convergent: 93222358·sqrt(2) - 131836323 is irrational,
        // so it is PROVABLY nonzero, never spuriously Equal (f64 rounds it to 0.0).
        // p²−2q² = 1 ⇒ q·sqrt(2) < p ⇒ strictly negative. The soundness-critical
        // fact is that it is NOT Equal, so a NonZero condition is satisfied (kept).
        let conv = parse("93222358*sqrt(2) - 131836323", &mut ctx).expect("conv");
        assert_eq!(provable_sign_vs_zero(&ctx, conv), Some(Ordering::Less));
        assert_ne!(provable_sign_vs_zero(&ctx, conv), Some(Ordering::Equal));

        // Reciprocal-surd spelling: the solver emits the negative branch of a
        // symmetric quadratic as -N·N^(-1/2) = -√N. `-13·13^(-1/2)` must reduce
        // to -√13 = (0, -1, 13), and `-13·13^(-1/2) - 2` must be provably Less.
        let recip = parse("-13*13^(-1/2)", &mut ctx).expect("recip");
        assert_eq!(
            as_linear_surd(&ctx, recip),
            Some((rat(0, 1), rat(-1, 1), rat(13, 1)))
        );
        let recip_shift = parse("-13*13^(-1/2) - 2", &mut ctx).expect("recip_shift");
        assert_eq!(
            provable_sign_vs_zero(&ctx, recip_shift),
            Some(Ordering::Less)
        );
        // b^(-1/2) itself is positive; b^(3/2) = b·√b is positive.
        let inv = parse("13^(-1/2)", &mut ctx).expect("inv");
        assert_eq!(provable_sign_vs_zero(&ctx, inv), Some(Ordering::Greater));
        let three_halves = parse("4^(3/2)", &mut ctx).expect("th"); // = 4·√4 = 8 > 0
        assert_eq!(
            provable_sign_vs_zero(&ctx, three_halves),
            Some(Ordering::Greater)
        );

        // Exact zero is detected: 1 - sqrt(1) = 0.
        let z = parse("1 - sqrt(1)", &mut ctx).expect("zero");
        assert_eq!(provable_sign_vs_zero(&ctx, z), Some(Ordering::Equal));

        // Pure rationals.
        let neg = parse("-5/2", &mut ctx).expect("neg");
        assert_eq!(provable_sign_vs_zero(&ctx, neg), Some(Ordering::Less));

        // Two distinct surds / transcendental atoms -> None (conservative KEEP).
        let two_surds = parse("sqrt(2) + sqrt(3)", &mut ctx).expect("two");
        assert_eq!(as_linear_surd(&ctx, two_surds), None);
        let trig = parse("sin(x) + 1", &mut ctx).expect("trig");
        assert_eq!(provable_sign_vs_zero(&ctx, trig), None);

        // sqrt of a negative is not a real linear surd -> None.
        let neg_rad = parse("sqrt(-2)", &mut ctx).expect("negrad");
        assert_eq!(as_linear_surd(&ctx, neg_rad), None);
    }

    fn eval_small_rat(ctx: &Context, id: ExprId) -> Option<BigRational> {
        match ctx.get(id) {
            Expr::Number(n) => Some(n.clone()),
            Expr::Div(l, r) => {
                let den = eval_small_rat(ctx, *r)?;
                if den == BigRational::from_integer(0.into()) {
                    None
                } else {
                    Some(eval_small_rat(ctx, *l)? / den)
                }
            }
            _ => None,
        }
    }

    #[test]
    fn extract_from_sqrt_function() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x)", &mut ctx).expect("parse");
        let (radicand, index) = extract_root_base_and_index(&mut ctx, expr).expect("root");

        let x = parse("x", &mut ctx).expect("parse x");
        let two = parse("2", &mut ctx).expect("parse 2");
        assert!(poly_eq(&ctx, radicand, x));
        assert!(poly_eq(&ctx, index, two));
    }

    #[test]
    fn extract_from_fractional_power() {
        let mut ctx = Context::new();
        let expr = parse("x^(1/3)", &mut ctx).expect("parse");
        let (radicand, index) = extract_root_base_and_index(&mut ctx, expr).expect("root");

        let x = parse("x", &mut ctx).expect("parse x");
        let three = parse("3", &mut ctx).expect("parse 3");
        assert!(poly_eq(&ctx, radicand, x));
        assert!(poly_eq(&ctx, index, three));
    }

    #[test]
    fn reject_non_unit_numerator_exponent() {
        let mut ctx = Context::new();
        let expr = parse("x^(2/3)", &mut ctx).expect("parse");
        assert!(extract_root_base_and_index(&mut ctx, expr).is_none());
    }

    #[test]
    fn extract_numeric_sqrt_radicand_recognizes_forms() {
        let mut ctx = Context::new();
        let sqrt_fn = parse("sqrt(7)", &mut ctx).expect("parse sqrt");
        let sqrt_pow = parse("7^(1/2)", &mut ctx).expect("parse pow");
        let symbolic = parse("sqrt(x)", &mut ctx).expect("parse symbolic");

        assert_eq!(extract_numeric_sqrt_radicand(&ctx, sqrt_fn), Some(7));
        assert_eq!(extract_numeric_sqrt_radicand(&ctx, sqrt_pow), Some(7));
        assert_eq!(extract_numeric_sqrt_radicand(&ctx, symbolic), None);
    }

    #[test]
    fn extract_square_root_base_detects_builtin_and_pow_forms() {
        let mut ctx = Context::new();
        let sqrt_fn = parse("sqrt(x+1)", &mut ctx).expect("parse sqrt");
        let sqrt_pow = parse("(x+1)^(1/2)", &mut ctx).expect("parse pow");
        assert!(extract_square_root_base(&ctx, sqrt_fn).is_some());
        assert!(extract_square_root_base(&ctx, sqrt_pow).is_some());
    }

    #[test]
    fn split_numeric_plus_surd_detects_add_and_sub() {
        let mut ctx = Context::new();
        let plus = parse("5+sqrt(2)", &mut ctx).expect("plus");
        let minus = parse("5-sqrt(2)", &mut ctx).expect("minus");

        let (m_plus, t_plus, sign_plus) = split_numeric_plus_surd(&ctx, plus).expect("split +");
        let (m_minus, t_minus, sign_minus) = split_numeric_plus_surd(&ctx, minus).expect("split -");

        assert!(poly_eq(&ctx, m_plus, m_minus));
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, t_plus, t_minus),
            std::cmp::Ordering::Equal
        );
        assert_eq!(sign_plus, 1);
        assert_eq!(sign_minus, -1);
    }

    #[test]
    fn conjugate_numeric_surd_pair_detects_match() {
        let mut ctx = Context::new();
        let left = parse("3+sqrt(7)", &mut ctx).expect("left");
        let right = parse("3-sqrt(7)", &mut ctx).expect("right");
        assert!(conjugate_numeric_surd_pair(&ctx, left, right).is_some());
    }

    #[test]
    fn extract_cube_root_base_detects_pow_one_third() {
        let mut ctx = Context::new();
        let expr = parse("(2+sqrt(5))^(1/3)", &mut ctx).expect("expr");
        assert!(extract_cube_root_base(&ctx, expr).is_some());
    }

    #[test]
    fn rational_cbrt_exact_handles_perfect_cube() {
        let r = BigRational::new(8.into(), 27.into());
        let root = rational_cbrt_exact(&r).expect("root");
        assert_eq!(root, BigRational::new(2.into(), 3.into()));
    }

    #[test]
    fn surd_square_rational_handles_scaled_sqrt() {
        let mut ctx = Context::new();
        let expr = parse("3*sqrt(2)", &mut ctx).expect("expr");
        let t2 = surd_square_rational(&ctx, expr).expect("t2");
        assert_eq!(t2, BigRational::from_integer(18.into()));
    }

    #[test]
    fn find_rational_root_depressed_cubic_finds_simple_root() {
        // x^3 - 3x - 2 = 0 has rational roots 2 and -1
        let p = BigRational::from_integer((-3).into());
        let q = BigRational::from_integer((-2).into());
        let root = find_rational_root_depressed_cubic(&p, &q).expect("root");
        assert!(
            root == BigRational::from_integer(2.into())
                || root == BigRational::from_integer((-1).into())
        );
    }

    #[test]
    fn cubic_conjugate_identity_rewrite_matches_known_case() {
        let mut ctx = Context::new();
        let expr = parse("(2 + sqrt(5))^(1/3) + (2 - sqrt(5))^(1/3)", &mut ctx).expect("expr");
        let rewrite = try_rewrite_cubic_conjugate_identity_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            eval_small_rat(&ctx, rewrite.rewritten),
            Some(BigRational::from_integer(1.into()))
        );
        assert!(rewrite.desc.contains("Cubic conjugate identity"));
    }

    #[test]
    fn cubic_conjugate_identity_rewrite_rejects_non_conjugate_pair() {
        let mut ctx = Context::new();
        let expr = parse("(2 + sqrt(5))^(1/3) + (2 + sqrt(5))^(1/3)", &mut ctx).expect("expr");
        assert!(try_rewrite_cubic_conjugate_identity_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn extract_root_factor_preserves_sign_by_parity() {
        use num_bigint::BigInt;

        let n = BigInt::from(-72);

        let (outside_even, inside_even) = extract_root_factor(&n, 2);
        assert_eq!(outside_even, BigInt::from(6));
        assert_eq!(inside_even, BigInt::from(-2));

        let (outside_odd, inside_odd) = extract_root_factor(&n, 3);
        assert_eq!(outside_odd, BigInt::from(-2));
        assert_eq!(inside_odd, BigInt::from(9));
    }

    #[test]
    fn can_distribute_root_safely_accepts_multiple_powers_only() {
        let mut ctx = Context::new();
        let safe = parse("x^2*9", &mut ctx).expect("safe");
        let unsafe_expr = parse("x*9", &mut ctx).expect("unsafe");

        assert!(can_distribute_root_safely(&ctx, safe, &2.into()));
        assert!(!can_distribute_root_safely(&ctx, unsafe_expr, &2.into()));
    }

    #[test]
    fn canonical_root_rewrite_handles_sqrt_and_root_forms() {
        let mut ctx = Context::new();

        let sqrt_expr = parse("sqrt(x)", &mut ctx).expect("sqrt");
        let sqrt_rewrite =
            try_rewrite_canonical_root_expr(&mut ctx, sqrt_expr).expect("rewrite sqrt");
        let x = parse("x", &mut ctx).expect("x");
        let Expr::Pow(sqrt_base, sqrt_exp) = ctx.get(sqrt_rewrite.rewritten) else {
            panic!("sqrt rewrite should be Pow");
        };
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, *sqrt_base, x),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            eval_small_rat(&ctx, *sqrt_exp),
            Some(BigRational::new(1.into(), 2.into()))
        );

        let root_expr = parse("root(x, 3)", &mut ctx).expect("root");
        let root_rewrite =
            try_rewrite_canonical_root_expr(&mut ctx, root_expr).expect("rewrite root");
        let Expr::Pow(root_base, root_exp) = ctx.get(root_rewrite.rewritten) else {
            panic!("root rewrite should be Pow");
        };
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, *root_base, x),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            eval_small_rat(&ctx, *root_exp),
            Some(BigRational::new(1.into(), 3.into()))
        );
    }

    #[test]
    fn canonical_root_rewrite_even_power_maps_to_abs_power() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x^4)", &mut ctx).expect("expr");
        let rewrite = try_rewrite_canonical_root_expr(&mut ctx, expr).expect("rewrite");
        let x = parse("x", &mut ctx).expect("x");
        let Expr::Pow(base, exp) = ctx.get(rewrite.rewritten) else {
            panic!("expected Pow");
        };
        let Expr::Function(fn_id, args) = ctx.get(*base) else {
            panic!("expected abs(base)");
        };
        assert!(ctx.is_builtin(*fn_id, BuiltinFn::Abs));
        assert_eq!(args.len(), 1);
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, args[0], x),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            eval_small_rat(&ctx, *exp),
            Some(BigRational::from_integer(2.into()))
        );
    }

    #[test]
    fn odd_half_power_rewrite_builds_abs_times_sqrt() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let exp = ctx.rational(5, 2);
        let expr = ctx.add(Expr::Pow(x, exp));
        let rewrite = try_rewrite_odd_half_power_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.numerator, 5);
        assert_eq!(rewrite.abs_power, 2);

        let rendered = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert!(rendered.contains("|x|^2"));
        assert!(rendered.contains("sqrt(x)") || rendered.contains("√x"));
    }

    #[test]
    fn odd_half_power_rewrite_scales_across_multiple_odd_numerators() {
        let cases = [
            ("x", "abs(x)*sqrt(x)", 3_i64, 1_i64),
            ("x", "abs(x)^2*sqrt(x)", 5, 2),
            ("x", "abs(x)^5*sqrt(x)", 11, 5),
            ("y", "abs(y)^6*sqrt(y)", 13, 6),
            ("x", "abs(x)^7*sqrt(x)", 15, 7),
        ];

        for (base_name, expected, numer, abs_power) in cases {
            let mut ctx = Context::new();
            let base = ctx.var(base_name);
            let exp = ctx.rational(numer, 2);
            let expr = ctx.add(Expr::Pow(base, exp));
            let rewrite = try_rewrite_odd_half_power_expr(&mut ctx, expr).expect("rewrite");
            assert_eq!(
                rewrite.numerator, numer,
                "wrong numerator for {base_name}^({numer}/2)"
            );
            assert_eq!(
                rewrite.abs_power, abs_power,
                "wrong abs power for {base_name}^({numer}/2)"
            );

            let expected_expr = parse(expected, &mut ctx).expect("expected");
            assert_eq!(
                cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected_expr),
                std::cmp::Ordering::Equal,
                "unexpected rewrite for {base_name}^({numer}/2)"
            );
        }
    }

    #[test]
    fn denest_sqrt_add_sqrt_rewrite_matches_known_case() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(3 + sqrt(5))", &mut ctx).expect("expr");
        let rewrite = try_rewrite_denest_sqrt_add_sqrt_expr(&mut ctx, expr).expect("rewrite");

        let Expr::Add(l, r) = ctx.get(rewrite.rewritten) else {
            panic!("denest rewrite should build sum of two roots");
        };
        let mut roots = Vec::new();
        for term in [*l, *r] {
            let Expr::Pow(base, exp) = ctx.get(term) else {
                panic!("term should be a power");
            };
            assert_eq!(
                eval_small_rat(&ctx, *exp),
                Some(BigRational::new(1.into(), 2.into()))
            );
            let Expr::Number(n) = ctx.get(*base) else {
                panic!("root base should be numeric");
            };
            roots.push(n.clone());
        }
        assert!(roots.contains(&BigRational::new(1.into(), 2.into())));
        assert!(roots.contains(&BigRational::new(5.into(), 2.into())));
    }

    #[test]
    fn denest_sqrt_add_sqrt_rewrite_rejects_subtraction_shape() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(3 - 2*sqrt(2))", &mut ctx).expect("expr");
        assert!(try_rewrite_denest_sqrt_add_sqrt_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn denest_cube_quadratic_rewrite_matches_known_case() {
        let mut ctx = Context::new();
        let expr = parse("(26 + 15*sqrt(3))^(1/3)", &mut ctx).expect("expr");
        let rewrite =
            try_rewrite_denest_cube_quadratic_field_expr(&mut ctx, expr).expect("rewrite");

        fn is_one(ctx: &Context, id: ExprId) -> bool {
            matches!(
                ctx.get(id),
                Expr::Number(n) if *n == BigRational::from_integer(1.into())
            )
        }
        fn is_two(ctx: &Context, id: ExprId) -> bool {
            matches!(
                ctx.get(id),
                Expr::Number(n) if *n == BigRational::from_integer(2.into())
            )
        }
        fn is_sqrt_three(ctx: &Context, id: ExprId) -> bool {
            match ctx.get(id) {
                Expr::Pow(base, exp) => {
                    matches!(
                        (ctx.get(*base), ctx.get(*exp)),
                        (Expr::Number(n), Expr::Number(e))
                            if *n == BigRational::from_integer(3.into())
                                && *e == BigRational::new(1.into(), 2.into())
                    )
                }
                Expr::Mul(l, r) => {
                    (is_one(ctx, *l) && is_sqrt_three(ctx, *r))
                        || (is_one(ctx, *r) && is_sqrt_three(ctx, *l))
                }
                _ => false,
            }
        }

        let Expr::Add(l, r) = ctx.get(rewrite.rewritten) else {
            panic!("cube denest rewrite should build an additive form");
        };
        assert!(
            (is_two(&ctx, *l) && is_sqrt_three(&ctx, *r))
                || (is_two(&ctx, *r) && is_sqrt_three(&ctx, *l))
        );
    }

    #[test]
    fn denest_cube_quadratic_rewrite_rejects_nonmatching_input() {
        let mut ctx = Context::new();
        let expr = parse("(5 + sqrt(6))^(1/3)", &mut ctx).expect("expr");
        assert!(try_rewrite_denest_cube_quadratic_field_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn root_denesting_rewrite_matches_nested_surd_form() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(4 + sqrt(7))", &mut ctx).expect("expr");
        let rewrite = try_rewrite_root_denesting_expr(&mut ctx, expr).expect("rewrite");
        let rendered = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert!(rendered.contains("/ 2"), "rendered={rendered}");
        assert_eq!(rendered.matches("sqrt(").count(), 2, "rendered={rendered}");
    }

    #[test]
    fn root_denesting_rewrite_rejects_non_perfect_delta() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(4 + sqrt(10))", &mut ctx).expect("expr");
        assert!(try_rewrite_root_denesting_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn extract_perfect_power_from_radicand_rewrite_sqrt_case() {
        let mut ctx = Context::new();
        let expr = parse("(8*x)^(1/2)", &mut ctx).expect("expr");
        let expected = parse("2*(2*x)^(1/2)", &mut ctx).expect("expected");
        let rewrite =
            try_rewrite_extract_perfect_power_from_radicand_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn extract_perfect_power_from_radicand_rewrite_rational_sqrt_factor_case() {
        let mut ctx = Context::new();
        let rest = parse("4-x^4", &mut ctx).expect("rest");
        let quarter = ctx.add(Expr::Number(BigRational::new(1.into(), 4.into())));
        let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
        let base = ctx.add(Expr::Mul(quarter, rest));
        let expr = ctx.add(Expr::Pow(base, half));
        let expected_coeff = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
        let expected_root = ctx.add(Expr::Pow(rest, half));
        let expected = ctx.add(Expr::Mul(expected_coeff, expected_root));
        let rewrite =
            try_rewrite_extract_perfect_power_from_radicand_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn extract_perfect_power_from_radicand_rewrite_reciprocal_rational_sqrt_factor_case() {
        let mut ctx = Context::new();
        let rest = parse("3-x^2-2*x", &mut ctx).expect("rest");
        let quarter = ctx.add(Expr::Number(BigRational::new(1.into(), 4.into())));
        let neg_half = ctx.add(Expr::Number(BigRational::new((-1).into(), 2.into())));
        let base = ctx.add(Expr::Mul(quarter, rest));
        let expr = ctx.add(Expr::Pow(base, neg_half));
        let expected_coeff = ctx.add(Expr::Number(BigRational::from_integer(2.into())));
        let expected_root = ctx.add(Expr::Pow(rest, neg_half));
        let expected = ctx.add(Expr::Mul(expected_coeff, expected_root));
        let rewrite =
            try_rewrite_extract_perfect_power_from_radicand_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn extract_perfect_power_from_radicand_rewrite_reciprocal_additive_square_factor_case() {
        let mut ctx = Context::new();
        let expr = parse("(4*x - 4*x^2)^(-1/2)", &mut ctx).expect("expr");
        let rewrite =
            try_rewrite_extract_perfect_power_from_radicand_expr(&mut ctx, expr).expect("rewrite");
        let Expr::Mul(coeff, root) = ctx.get(rewrite.rewritten) else {
            panic!("rewrite should extract a reciprocal square-root coefficient");
        };
        let (coeff, root) = (*coeff, *root);
        assert_eq!(
            eval_small_rat(&ctx, coeff),
            Some(BigRational::new(1.into(), 2.into()))
        );
        let Expr::Pow(base, exp) = ctx.get(root) else {
            panic!("rewrite should retain a reciprocal square-root factor");
        };
        let (base, exp) = (*base, *exp);
        let expected_base = parse("x - x^2", &mut ctx).expect("expected base");
        assert!(poly_eq(&ctx, base, expected_base));
        assert_eq!(
            eval_small_rat(&ctx, exp),
            Some(BigRational::new((-1).into(), 2.into()))
        );
    }

    #[test]
    fn extract_perfect_power_from_radicand_rewrite_scaled_unit_square_reciprocal_case() {
        let mut ctx = Context::new();
        let expr = parse("(1 - (1/2*(x+1))^2)^(-1/2)", &mut ctx).expect("expr");
        let expected = parse("2*(4 - (x+1)^2)^(-1/2)", &mut ctx).expect("expected");
        let rewrite =
            try_rewrite_extract_perfect_power_from_radicand_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn extract_perfect_power_from_radicand_rewrite_scaled_square_plus_unit_reciprocal_case() {
        let mut ctx = Context::new();
        let expr = parse("(1 + (1/2*(x+1))^2)^(-1/2)", &mut ctx).expect("expr");
        let expected = parse("2*(4 + (x+1)^2)^(-1/2)", &mut ctx).expect("expected");
        let rewrite =
            try_rewrite_extract_perfect_power_from_radicand_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn extract_perfect_power_from_radicand_rewrite_sqrt_mixed_square_factor_case() {
        let mut ctx = Context::new();
        let expr = parse("(8*x^2)^(1/2)", &mut ctx).expect("expr");
        let expected = parse("2*abs(x)*2^(1/2)", &mut ctx).expect("expected");
        let rewrite =
            try_rewrite_extract_perfect_power_from_radicand_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn extract_perfect_power_from_radicand_rewrite_fourth_root_case() {
        let mut ctx = Context::new();
        let expr = parse("(16*x)^(1/4)", &mut ctx).expect("expr");
        let expected = parse("2*x^(1/4)", &mut ctx).expect("expected");
        let rewrite =
            try_rewrite_extract_perfect_power_from_radicand_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn extract_perfect_power_from_radicand_rewrite_rejects_without_factor() {
        let mut ctx = Context::new();
        let expr = parse("(2*x)^(1/2)", &mut ctx).expect("expr");
        assert!(try_rewrite_extract_perfect_power_from_radicand_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn simplify_square_root_rewrite_trinomial_to_abs_linear() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x^2 + 2*x + 1)", &mut ctx).expect("expr");
        let expected = parse("abs(x + 1)", &mut ctx).expect("expected");
        let rewrite = try_rewrite_simplify_square_root_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn simplify_square_root_rewrite_root_ctx_shifted_unit_square() {
        let mut ctx = Context::new();
        let expr = parse("sqrt((1/sqrt(u))^2 + 2*(1/sqrt(u)) + 1)", &mut ctx).expect("expr");
        let expected = parse("abs((1/sqrt(u)) + 1)", &mut ctx).expect("expected");
        let rewrite = try_rewrite_simplify_square_root_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn simplify_square_root_rewrite_symbolic_phase_shift_linear() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(pi^2 + u^2 + 2*u + 2*pi*u + 1 + 2*pi)", &mut ctx).expect("expr");
        let expected = parse("abs(u + pi + 1)", &mut ctx).expect("expected");
        let rewrite = try_rewrite_simplify_square_root_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn simplify_square_root_rewrite_exp_shifted_unit_square() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(e^(2*u) + 2*e^u + 1)", &mut ctx).expect("expr");
        let expected = parse("abs(e^u + 1)", &mut ctx).expect("expected");
        let rewrite = try_rewrite_simplify_square_root_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn simplify_square_root_rewrite_rejects_non_perfect_square_poly() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x^2 + 1)", &mut ctx).expect("expr");
        assert!(try_rewrite_simplify_square_root_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn simplify_square_root_rewrite_extracts_additive_common_square_factor() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(4*x^2 + 4)", &mut ctx).expect("expr");
        let expected = parse("2*sqrt(x^2 + 1)", &mut ctx).expect("expected");
        let rewrite = try_rewrite_simplify_square_root_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn simplify_square_root_rewrite_extracts_quotient_of_perfect_squares() {
        let mut ctx = Context::new();
        let expr = parse("sqrt((4*x^2 + 4*x + 1)/(x^2 + 2*x + 1))", &mut ctx).expect("expr");
        let expected = parse("abs((2*x + 1)/(x + 1))", &mut ctx).expect("expected");
        let rewrite = try_rewrite_simplify_square_root_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn simplify_square_root_rewrite_extracts_factored_quotient_of_squares() {
        let mut ctx = Context::new();
        let expr = parse(
            "sqrt((x^4 + 6*x^3 + 11*x^2 + 6*x + 1)/(x^4 + 2*x^3 + x^2))",
            &mut ctx,
        )
        .expect("expr");
        let expected = parse("abs((x^2 + 3*x + 1)/(x^2 + x))", &mut ctx).expect("expected");
        let rewrite = try_rewrite_simplify_square_root_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }
}
