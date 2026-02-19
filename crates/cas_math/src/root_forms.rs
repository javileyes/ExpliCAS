//! Root-shape helpers over AST expressions.

use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

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
}
