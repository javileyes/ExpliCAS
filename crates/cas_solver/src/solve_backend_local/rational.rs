//! `solve_backend_local`: familia `rational`.
//!
//! Ver la cabecera de `solve_backend_local.rs` para el contexto.

use super::*;

/// True when the equation contains a denominator that is provably zero for ALL
/// `x` — `x/0`, `1/0`, or `x/(x-x)` — so the equation is identically UNDEFINED
/// over ℝ and therefore has NO real solution. Without this guard the isolation
/// logic cancels or eliminates the undefined term and fabricates a spurious
/// `All real numbers` (`solve(x/0=5) → ℝ`, `solve(x=1/0) → ℝ`) or an
/// impossible-conditioned identity (`solve(x/(x-x)=0) → ℝ if 0 ≠ 0`).
///
/// "Provably zero everywhere" is decided EXACTLY: each `Div` denominator is
/// simplified and accepted only when it folds to the rational constant `0`
/// (covers the literal `0`, `x-x`, `0*x`, …). A denominator that merely vanishes
/// at some points (`x` in `3/x`, `x-1` in `1/(x-1)`) does NOT match — those are
/// legitimate excluded points, not an undefined equation. Unfoldable denominators
/// keep the prior behaviour (conservative: never a false "No solution").
pub(super) fn equation_has_identically_zero_denominator(
    simplifier: &mut Simplifier,
    eq: &Equation,
) -> bool {
    fn any_zero_denominator(simplifier: &mut Simplifier, expr: ExprId) -> bool {
        match simplifier.context.get(expr).clone() {
            Expr::Div(num, den) => {
                denominator_is_identically_zero(simplifier, den)
                    || any_zero_denominator(simplifier, num)
                    || any_zero_denominator(simplifier, den)
            }
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Pow(a, b) => {
                any_zero_denominator(simplifier, a) || any_zero_denominator(simplifier, b)
            }
            Expr::Neg(a) | Expr::Hold(a) => any_zero_denominator(simplifier, a),
            Expr::Function(_, args) => args
                .into_iter()
                .any(|c| any_zero_denominator(simplifier, c)),
            _ => false,
        }
    }
    any_zero_denominator(simplifier, eq.lhs) || any_zero_denominator(simplifier, eq.rhs)
}

/// True when `den` simplifies to the exact rational constant `0` (identically
/// zero everywhere). EXACT — `as_rational_const` never falls back to a float.
fn denominator_is_identically_zero(simplifier: &mut Simplifier, den: ExprId) -> bool {
    use num_traits::Zero;
    let (simplified, _) = simplifier.simplify(den);
    cas_math::numeric_eval::as_rational_const(&simplifier.context, simplified)
        .is_some_and(|r| r.is_zero())
}

/// Express `e` as a ratio of polynomials `(num, den)` in `var`, combining sums/differences/products/
/// quotients/integer-powers over a common denominator WITHOUT cancelling shared factors (so every
/// genuine pole stays in `den`, which the caller's numeric verification relies on). The denominator is
/// the PRODUCT of the sub-denominators, not their lcm — this only raises the MULTIPLICITY of existing
/// poles (never introduces a new pole location, since each factor is a real denominator of `e`), and the
/// caller's `P/D {op} 0` sign analysis is invariant under multiplying both `P` and `D` by the same
/// factor, so the candidate stays exact. Returns `None` if any leaf is not a polynomial in `var` (a
/// fractional power `x^(1/2)`, a transcendental, …) so such inputs decline cleanly.
pub(super) fn rational_function_of(
    ctx: &mut Context,
    e: ExprId,
    var: &str,
    depth: usize,
) -> Option<(
    cas_math::polynomial::Polynomial,
    cas_math::polynomial::Polynomial,
)> {
    use cas_math::numeric_eval::as_rational_const;
    use cas_math::polynomial::Polynomial;
    use num_traits::{One, ToPrimitive};

    if depth > 48 {
        return None;
    }
    let one = || Polynomial::new(vec![num_rational::BigRational::one()], var.to_string());

    match ctx.get(e).clone() {
        // (nl/dl) ± (nr/dr) = (nl·dr ± nr·dl) / (dl·dr) — the `Add` case is what lets a sum such as
        // `x + 1/x` reach the reliable rational path (instead of declining and falling to a generic
        // path that drops the inequality operator).
        Expr::Add(l, r) => {
            let (nl, dl) = rational_function_of(ctx, l, var, depth + 1)?;
            let (nr, dr) = rational_function_of(ctx, r, var, depth + 1)?;
            Some((nl.mul(&dr).add(&nr.mul(&dl)), dl.mul(&dr)))
        }
        Expr::Sub(l, r) => {
            let (nl, dl) = rational_function_of(ctx, l, var, depth + 1)?;
            let (nr, dr) = rational_function_of(ctx, r, var, depth + 1)?;
            Some((nl.mul(&dr).sub(&nr.mul(&dl)), dl.mul(&dr)))
        }
        Expr::Mul(l, r) => {
            let (nl, dl) = rational_function_of(ctx, l, var, depth + 1)?;
            let (nr, dr) = rational_function_of(ctx, r, var, depth + 1)?;
            Some((nl.mul(&nr), dl.mul(&dr)))
        }
        Expr::Div(l, r) => {
            let (nl, dl) = rational_function_of(ctx, l, var, depth + 1)?;
            let (nr, dr) = rational_function_of(ctx, r, var, depth + 1)?;
            Some((nl.mul(&dr), dl.mul(&nr)))
        }
        Expr::Neg(inner) => {
            let (n, d) = rational_function_of(ctx, inner, var, depth + 1)?;
            let neg_one = Polynomial::new(vec![-num_rational::BigRational::one()], var.to_string());
            Some((n.mul(&neg_one), d))
        }
        Expr::Pow(base, exp) => {
            if let Some(k) = as_rational_const(ctx, exp) {
                if k.is_integer() {
                    let exponent = k.to_i64()?;
                    let magnitude = exponent.unsigned_abs() as usize;
                    if magnitude > 12 {
                        return None; // bound degree growth (matches the rational-inequality degree cap)
                    }
                    let (nb, db) = rational_function_of(ctx, base, var, depth + 1)?;
                    let raise = |p: &Polynomial| {
                        let mut acc = one();
                        for _ in 0..magnitude {
                            acc = acc.mul(p);
                        }
                        acc
                    };
                    let (np, dp) = (raise(&nb), raise(&db));
                    // A negative exponent sends the base to the opposite side (`x^(-2)` → `1/x²`).
                    return Some(if exponent < 0 { (dp, np) } else { (np, dp) });
                }
            }
            // Non-integer / symbolic exponent: only sound if the whole power is itself a polynomial
            // in `var` (it is not, for `x^(1/2)`), so this declines via `from_expr`.
            let p = Polynomial::from_expr(ctx, e, var).ok()?;
            Some((p, one()))
        }
        _ => {
            let p = Polynomial::from_expr(ctx, e, var).ok()?;
            Some((p, one()))
        }
    }
}

/// Solve an EQUATION that is a polynomial of degree ≥ 2 in `x^(1/q)` for some
/// integer `q ≥ 2`: `x` appears only as positive rational powers with common
/// denominator `q` (e.g. `x - 3·√x + 2 = 0`, a quadratic in `√x`, or
/// `x^(2/3) - x^(1/3) - 2 = 0`, a quadratic in `x^(1/3)`). Substitute `u = x^(1/q)`,
/// solve the polynomial in `u`, then back-substitute `x^(1/q) = u_root` — the
/// recursive solver finishes each with the correct real-root domain (even `q`
/// drops negative `u_root`, odd `q` keeps it). Without this, the isolation path
/// reorients to `x = f(x)` and leaks a malformed `solve(...)` residual while
/// dropping every root.
pub(super) fn try_solve_rational_power_polynomial(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
    steps_out: &mut Vec<crate::SolveStep>,
) -> Option<SolutionSet> {
    use num_bigint::BigInt;
    use num_integer::Integer;
    use num_rational::BigRational;
    use num_traits::One;

    if eq.op != cas_ast::RelOp::Eq {
        return None;
    }
    // Simplify the difference so radicals canonicalize to `x^(p/q)` powers.
    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (expr, _) = simplifier.simplify(diff);

    let mut exps: Vec<BigRational> = Vec::new();
    if !collect_x_power_exponents(&simplifier.context, expr, var, &mut exps) || exps.is_empty() {
        return None;
    }
    let q_big = exps.iter().fold(BigInt::one(), |acc, e| acc.lcm(e.denom()));
    if q_big <= BigInt::one() {
        return None; // q == 1: a plain polynomial in x, owned by the normal path
    }

    let u_var = "__rps_u";
    let u_expr = rebuild_x_powers_as_u(&mut simplifier.context, expr, var, u_var, &q_big);

    // Back-substitution atom is `x^(1/q)`; `solve_polynomial_in_atom` enforces the
    // degree-≥2 gate, solves for u, and back-substitutes with the real-root domain.
    let recip_q = simplifier
        .context
        .add(Expr::Number(BigRational::new(BigInt::one(), q_big)));
    let x = simplifier.context.var(var);
    let atom = simplifier.context.add(Expr::Pow(x, recip_q));
    solve_polynomial_in_atom(simplifier, u_expr, u_var, var, atom, steps_out)
}

/// Solve an equation that is a LAURENT polynomial in `x^(1/q)` — a root mixed
/// with its RECIPROCAL, e.g. `√x − 1/√x = 1`, `√x + 1/√x = 5/2`. Collect the
/// Laurent map `x^(p/q) → u^p` (`u = x^(1/q)`), shift every exponent up by
/// `−min_k` (multiply through by `u^(−min_k)`, a positive real ⇒ no real root
/// lost) to get a POLYNOMIAL in `u` built term-by-term (a `Mul(...)·u^K` does not
/// auto-distribute), then hand it to `solve_polynomial_in_atom`, which solves for
/// `u` and back-substitutes `x^(1/q) = u_root` (the root domain drops `u < 0` for
/// even `q`, keeps it for odd `q`). The shift places a nonzero coefficient at
/// `u^0`, so no spurious `u = 0` is introduced.
///
/// Without this the isolation reorients to `x = (…)^(1/(1/2))` and leaks a
/// malformed `solve(...)` residual. Pure-positive-power forms (`x − 3√x + 2`) are
/// owned by [`try_solve_rational_power_polynomial`]: this needs a genuine
/// negative exponent to clear.
pub(super) fn try_solve_rational_power_laurent(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
    steps_out: &mut Vec<crate::SolveStep>,
) -> Option<SolutionSet> {
    use num_bigint::BigInt;
    use num_integer::Integer;
    use num_rational::BigRational;
    use num_traits::{One, ToPrimitive, Zero};
    use std::collections::BTreeMap;

    if eq.op != cas_ast::RelOp::Eq {
        return None;
    }
    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (expr, _) = simplifier.simplify(diff);

    // `simplify` may COMBINE the Laurent over a common denominator into
    // `Div(N, x^m)` (`x^(1/3) − 1/x^(1/3) → (x^(4/3) − x^(2/3))/x`). Since
    // `N/D = 0 ⟺ N = 0` off the pole `D = 0` (which the negative-power domain
    // already excludes), collect `N` and subtract the denominator monomial's
    // exponent from every term, restoring the original Laurent's negative powers.
    let (numer, den_exp, den_coeff) = match simplifier.context.get(expr) {
        Expr::Div(n, d) => {
            let (n, d) = (*n, *d);
            match x_root_laurent_leaf(&simplifier.context, d, var) {
                Some((de, dc)) if !dc.is_zero() => (n, de, dc),
                _ => (expr, BigRational::zero(), BigRational::one()),
            }
        }
        _ => (expr, BigRational::zero(), BigRational::one()),
    };

    let mut pairs: Vec<(BigRational, BigRational)> = Vec::new();
    if !collect_x_root_laurent_pairs(&simplifier.context, numer, var, true, &mut pairs)
        || pairs.is_empty()
    {
        return None;
    }
    if !den_exp.is_zero() || !den_coeff.is_one() {
        for (e, c) in pairs.iter_mut() {
            *e -= &den_exp;
            *c /= &den_coeff;
        }
    }
    let q_big = pairs
        .iter()
        .fold(BigInt::one(), |acc, (e, _)| acc.lcm(e.denom()));
    if q_big <= BigInt::one() {
        return None; // integer / Laurent-in-x — owned by the normal paths
    }
    let q_rat = BigRational::from(q_big.clone());
    // Laurent map `k → coeff`, `k = q·exponent` (integer). Fold repeats.
    let mut map: BTreeMap<i64, BigRational> = BTreeMap::new();
    for (e, c) in &pairs {
        let k = (e * &q_rat).to_integer().to_i64()?;
        *map.entry(k).or_insert_with(BigRational::zero) += c;
    }
    map.retain(|_, c| !c.is_zero());
    let min_k = *map.keys().next()?;
    let max_k = *map.keys().next_back()?;
    // Require a genuine reciprocal (`min_k < 0`) and span ≥ 2 (a proper quadratic
    // in `u` after shifting). Pure-positive is owned by the sibling handler.
    if min_k >= 0 || max_k - min_k < 2 {
        return None;
    }
    // Build `Σ coeff·u^(k − min_k)` directly — a polynomial in `u`.
    let u = simplifier.context.var("__rpl_u");
    let mut u_expr = simplifier.context.num(0);
    for (k, c) in &map {
        let coeff = simplifier.context.add(Expr::Number(c.clone()));
        let shift = simplifier.context.num(k - min_k);
        let power = simplifier.context.add(Expr::Pow(u, shift));
        let term = simplifier.context.add(Expr::Mul(coeff, power));
        u_expr = simplifier.context.add(Expr::Add(u_expr, term));
    }

    let recip_q = simplifier
        .context
        .add(Expr::Number(BigRational::new(BigInt::one(), q_big)));
    let x = simplifier.context.var(var);
    let atom = simplifier.context.add(Expr::Pow(x, recip_q));
    solve_polynomial_in_atom(simplifier, u_expr, "__rpl_u", var, atom, steps_out)
}

/// Rational coefficient of an additive term that is a RATIONAL multiple of the bare variable
/// (`x` -> 1, `k·x`/`x·k` -> k, `x/k` -> 1/k, `Neg(t)` -> -coeff). Returns `None` if the term is not
/// a rational multiple of `x` alone (e.g. `x^2`, `x·y`, `a·x` with symbolic `a`).
pub(super) fn rational_coeff_of_bare_var(
    ctx: &cas_ast::Context,
    term: ExprId,
    var: &str,
) -> Option<num_rational::BigRational> {
    use cas_solver_core::isolation_utils::contains_var;
    use num_traits::{One as _, Zero as _};
    match ctx.get(term) {
        Expr::Neg(inner) => rational_coeff_of_bare_var(ctx, *inner, var).map(|c| -c),
        Expr::Variable(sym) if ctx.sym_name(*sym) == var => Some(num_rational::BigRational::one()),
        Expr::Div(num, den) if !contains_var(ctx, *den, var) => {
            let d = cas_math::numeric_eval::as_rational_const(ctx, *den)?;
            if d.is_zero() {
                return None;
            }
            rational_coeff_of_bare_var(ctx, *num, var).map(|c| c / d)
        }
        Expr::Mul(l, r) => {
            let (var_side, coeff_side) = if contains_var(ctx, *l, var) {
                (*l, *r)
            } else {
                (*r, *l)
            };
            if contains_var(ctx, coeff_side, var) {
                return None; // x·x etc.
            }
            let coeff = cas_math::numeric_eval::as_rational_const(ctx, coeff_side)?;
            rational_coeff_of_bare_var(ctx, var_side, var).map(|c| c * coeff)
        }
        _ => None,
    }
}

/// Peel a leading rational coefficient (including `Neg` as `-1` and nested `Mul`s of constants) off
/// `e`, returning `(coefficient, core)` with `e = coefficient · core`. `cos(x)^2 - 1` simplifies to
/// `-(sin(x)^2)`, so a `Neg` wrapper must be peeled for the squared-trig detector to see the trig.
pub(super) fn peel_rational_coefficient(
    ctx: &Context,
    e: ExprId,
) -> (num_rational::BigRational, ExprId) {
    use cas_math::numeric_eval::as_rational_const;
    use num_rational::BigRational;
    use num_traits::One;
    match ctx.get(e) {
        Expr::Neg(inner) => {
            let (c, core) = peel_rational_coefficient(ctx, *inner);
            (-c, core)
        }
        Expr::Mul(l, r) => {
            if let Some(a) = as_rational_const(ctx, *l) {
                let (c, core) = peel_rational_coefficient(ctx, *r);
                (a * c, core)
            } else if let Some(a) = as_rational_const(ctx, *r) {
                let (c, core) = peel_rational_coefficient(ctx, *l);
                (a * c, core)
            } else {
                (BigRational::one(), e)
            }
        }
        _ => (BigRational::one(), e),
    }
}

/// Build an exact rational constant expression.
pub(super) fn rational_to_expr(ctx: &mut Context, r: &num_rational::BigRational) -> ExprId {
    ctx.add(Expr::Number(r.clone()))
}

/// `period / π` as a positive rational, or `None` if `period` is not a rational multiple of π.
pub(super) fn period_as_rational_multiple_of_pi(
    simplifier: &mut Simplifier,
    period: ExprId,
) -> Option<num_rational::BigRational> {
    let pi = simplifier.context.add(Expr::Constant(Constant::Pi));
    let ratio = simplifier.context.add(Expr::Div(period, pi));
    let (ratio, _) = simplifier.simplify(ratio);
    cas_math::numeric_eval::as_rational_const(&simplifier.context, ratio)
}

pub(super) fn factor_monic_quartic_into_rational_quadratics(
    b: i64,
    c: i64,
    d: i64,
    e: i64,
) -> Option<((i64, i64), (i64, i64))> {
    if e == 0 {
        return None; // x is a factor ⇒ a rational root the caller already peeled
    }
    let abs_e = e.unsigned_abs();
    if abs_e > 1_000_000 {
        return None; // keep the divisor enumeration bounded
    }
    for mag in 1..=abs_e {
        if !abs_e.is_multiple_of(mag) {
            continue;
        }
        for q in [mag as i64, -(mag as i64)] {
            let s = e / q; // exact: q divides e
            if s == q {
                // The two quadratics share a constant term `q = s` (e.g. a perfect square like
                // `(x²-3)² = (x²-3)(x²-3)` with `q = s = -3`). The general formula below divides by
                // `s - q = 0`, so it skipped this case — which silently dropped the roots of a SQUARED
                // (or equal-constant) irreducible quadratic factor. Solve it directly: with `q = s`,
                //   p·s + r·q = q·(p + r) = q·b  ⇒ requires  d == q·b,
                //   q + s + p·r = 2q + p·r = c   ⇒  p·r = c - 2q,  and  p + r = b,
                // so `p, r` are the integer roots of `t² - b·t + (c - 2q) = 0`.
                if d != q * b {
                    continue;
                }
                let disc = b * b - 4 * (c - 2 * q);
                let root = match exact_i64_sqrt(disc) {
                    Some(v) => v,
                    None => continue,
                };
                if (b + root).rem_euclid(2) != 0 {
                    continue; // p, r would not be integers
                }
                let p = (b + root) / 2;
                let r = (b - root) / 2;
                if q + s + p * r == c && p + r == b {
                    return Some(((p, q), (r, s)));
                }
                continue;
            }
            let numerator = d - q * b;
            let denom = s - q;
            if numerator % denom != 0 {
                continue;
            }
            let p = numerator / denom;
            let r = b - p;
            if q + s + p * r == c && p + r == b {
                return Some(((p, q), (r, s)));
            }
        }
    }
    None
}
