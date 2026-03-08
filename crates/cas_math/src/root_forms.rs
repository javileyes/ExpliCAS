//! Root-shape helpers over AST expressions.

use crate::perfect_square_support::rational_sqrt;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
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
/// - `(16*x)^(1/4) -> 2*x^(1/4)`
pub fn try_rewrite_extract_perfect_power_from_radicand_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExtractPerfectPowerFromRadicandRewrite> {
    let (base, exp_id) = match ctx.get(expr) {
        Expr::Pow(b, e) => (*b, *e),
        _ => return None,
    };

    let root_index: u32 = match ctx.get(exp_id) {
        Expr::Number(n) if n.numer() == &1.into() => {
            let d = n.denom();
            let d_u32 = d.to_u32_digits().1.first().copied()?;
            if d_u32 >= 2 && d.sign() == num_bigint::Sign::Plus && d.bits() <= 32 {
                d_u32
            } else {
                return None;
            }
        }
        Expr::Div(num, den) => {
            let Expr::Number(num_n) = ctx.get(*num) else {
                return None;
            };
            let Expr::Number(den_n) = ctx.get(*den) else {
                return None;
            };
            if !num_n.is_one() || !den_n.is_integer() || den_n <= &BigRational::zero() {
                return None;
            }
            let d = den_n.to_integer();
            let d_u32 = d.to_u32_digits().1.first().copied()?;
            if d_u32 >= 2 && d.sign() == num_bigint::Sign::Plus && d.bits() <= 32 {
                d_u32
            } else {
                return None;
            }
        }
        _ => return None,
    };

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

    if !num_val.is_positive() || !num_val.is_integer() {
        return None;
    }

    let int_val = num_val.to_integer();
    let mut k = num_bigint::BigInt::from(1);
    let mut remaining = int_val.clone();
    let mut trial: num_bigint::BigInt = 2.into();
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
        if extracted > 0 {
            for _ in 0..extracted {
                k *= &trial;
            }
        }
        trial += 1;
    }
    if k == num_bigint::BigInt::from(1) {
        return None;
    }

    let mut k_power = num_bigint::BigInt::from(1);
    for _ in 0..root_index {
        k_power *= &k;
    }
    let new_coeff = &int_val / &k_power;

    let k_expr = ctx.add(Expr::Number(BigRational::from_integer(k)));
    let new_radicand = if new_coeff == num_bigint::BigInt::from(1) {
        rest
    } else {
        let new_coeff_expr = ctx.add(Expr::Number(BigRational::from_integer(new_coeff)));
        ctx.add(Expr::Mul(new_coeff_expr, rest))
    };
    let new_root = ctx.add(Expr::Pow(new_radicand, exp_id));
    let rewritten = ctx.add(Expr::Mul(k_expr, new_root));

    Some(ExtractPerfectPowerFromRadicandRewrite {
        rewritten,
        kind: ExtractPerfectPowerFromRadicandKind::ExtractPerfectSquare,
    })
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
    let abs_base = ctx.call_builtin(BuiltinFn::Abs, vec![base]);
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
    fn simplify_square_root_rewrite_rejects_non_perfect_square_poly() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x^2 + 1)", &mut ctx).expect("expr");
        assert!(try_rewrite_simplify_square_root_expr(&mut ctx, expr).is_none());
    }
}
