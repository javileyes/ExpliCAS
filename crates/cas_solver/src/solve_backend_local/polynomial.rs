//! `solve_backend_local`: familia `polynomial`.
//!
//! Ver la cabecera de `solve_backend_local.rs` para el contexto.

use super::*;

/// True when the equation is `c/poly = 0` for a nonzero constant `c` — which has
/// NO real solution (a nonzero constant over anything is never zero; the points
/// where the denominator vanishes make it undefined, not zero). Detected by
/// simplifying `lhs - rhs` to a single fraction with a nonzero-constant numerator.
///
/// Short-circuiting this BEFORE the isolation logic avoids the solver dividing by
/// zero (`poly = c/0 = ∞`), which otherwise fabricates `{∞}` or a malformed nested
/// `solve(x = ∞ - x^2, x)` for denominators like `x^2 + x + 1`.
pub(super) fn equation_is_nonzero_const_over_polynomial(
    simplifier: &mut Simplifier,
    eq: &Equation,
) -> bool {
    use num_traits::Zero;
    if eq.op != cas_ast::RelOp::Eq {
        return false;
    }
    // RAW-tree check first, BEFORE simplify: `c / g = 0` with a nonzero constant `c` has no
    // solution regardless of `g` (where defined the value is nonzero; the poles are undefined).
    // The simplifier RATIONALIZES a surd-affine denominator through its conjugate and plants a
    // numerator root there, so the post-simplify check below missed it and the solver returned
    // the conjugate as a root (`-2/3/(2x+√2) = 0 → {2^(-1/2)}`).
    if cas_math::numeric_eval::as_rational_const(&simplifier.context, eq.rhs)
        .is_some_and(|r| r.is_zero())
    {
        let mut node = eq.lhs;
        while let Expr::Neg(inner) = simplifier.context.get(node) {
            node = *inner;
        }
        if let Expr::Div(num, _den) = simplifier.context.get(node) {
            let num = *num;
            if cas_math::numeric_eval::as_rational_const(&simplifier.context, num)
                .is_some_and(|r| !r.is_zero())
            {
                return true;
            }
        }
    }
    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (simplified, _) = simplifier.simplify(diff);
    let Expr::Div(num, _den) = simplifier.context.get(simplified) else {
        return false;
    };
    let num = *num;
    cas_math::numeric_eval::as_rational_const(&simplifier.context, num)
        .is_some_and(|r| !r.is_zero())
}

/// True when `e` is an AFFINE function of `var` — a degree-1 polynomial `a·x + b` (`x`, `x-1`,
/// `2x+3`). The non-monotonicity of a fractional power is invariant under such a shift/scale, so
/// `(x-1)^(2/3)` is a symmetric valley exactly like `x^(2/3)`.
pub(super) fn is_affine_degree_one(ctx: &Context, e: ExprId, var: &str) -> bool {
    cas_math::polynomial::Polynomial::from_expr(ctx, e, var)
        .map(|p| p.degree() == 1)
        .unwrap_or(false)
}

/// Decompose `e` into `coeff·(α)^exp + addconst` where `α` is an AFFINE function of `var` (`a·x + b`),
/// `coeff`/`addconst` are rational constants, and `exp` is a rational constant. Returns
/// `(coeff, α, exp, addconst)`. Handles a leading coefficient, an additive constant on either side, and
/// `Neg`. Returns `None` for anything else (a sum of two powers, a non-affine base, …).
pub(super) fn extract_affine_power_term(
    ctx: &Context,
    e: ExprId,
    var: &str,
) -> Option<(
    num_rational::BigRational,
    ExprId,
    num_rational::BigRational,
    num_rational::BigRational,
)> {
    use cas_ast::ordering::compare_expr;
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;
    use num_rational::BigRational;
    use num_traits::{One, Zero};
    // Two affine power terms are LIKE (combinable) when they share the affine base and the exponent.
    let like =
        |ctx: &Context, a1: ExprId, x1: &BigRational, a2: ExprId, x2: &BigRational| -> bool {
            x1 == x2 && compare_expr(ctx, a1, a2) == std::cmp::Ordering::Equal
        };
    match ctx.get(e) {
        Expr::Neg(inner) => {
            let (c, a, x, d) = extract_affine_power_term(ctx, *inner, var)?;
            Some((-c, a, x, -d))
        }
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            match (contains_var(ctx, l, var), contains_var(ctx, r, var)) {
                (true, false) => {
                    let (c, a, x, d) = extract_affine_power_term(ctx, l, var)?;
                    Some((c, a, x, d + as_rational_const(ctx, r)?))
                }
                (false, true) => {
                    let (c, a, x, d) = extract_affine_power_term(ctx, r, var)?;
                    Some((c, a, x, d + as_rational_const(ctx, l)?))
                }
                // Both sides carry the variable: combine LIKE power terms
                // (`x^(2/3) + x^(2/3) → 2·x^(2/3)`), which the standalone simplifier folds but the raw
                // solve LHS does not. Unlike bases/exponents stay `None` (left to the other paths).
                (true, true) => {
                    let (c1, a1, x1, d1) = extract_affine_power_term(ctx, l, var)?;
                    let (c2, a2, x2, d2) = extract_affine_power_term(ctx, r, var)?;
                    like(ctx, a1, &x1, a2, &x2).then(|| (c1 + c2, a1, x1, d1 + d2))
                }
                (false, false) => None,
            }
        }
        Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            match (contains_var(ctx, l, var), contains_var(ctx, r, var)) {
                (true, false) => {
                    let (c, a, x, d) = extract_affine_power_term(ctx, l, var)?;
                    Some((c, a, x, d - as_rational_const(ctx, r)?))
                }
                (false, true) => {
                    // `cst − (c·αˣ + d) = −c·αˣ + (cst − d)`
                    let (c, a, x, d) = extract_affine_power_term(ctx, r, var)?;
                    Some((-c, a, x, as_rational_const(ctx, l)? - d))
                }
                (true, true) => {
                    let (c1, a1, x1, d1) = extract_affine_power_term(ctx, l, var)?;
                    let (c2, a2, x2, d2) = extract_affine_power_term(ctx, r, var)?;
                    like(ctx, a1, &x1, a2, &x2).then(|| (c1 - c2, a1, x1, d1 - d2))
                }
                (false, false) => None,
            }
        }
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            match (contains_var(ctx, l, var), contains_var(ctx, r, var)) {
                (true, false) => {
                    let (c, a, x, d) = extract_affine_power_term(ctx, l, var)?;
                    let f = as_rational_const(ctx, r)?;
                    Some((c * &f, a, x, d * f))
                }
                (false, true) => {
                    let (c, a, x, d) = extract_affine_power_term(ctx, r, var)?;
                    let f = as_rational_const(ctx, l)?;
                    Some((c * &f, a, x, d * f))
                }
                _ => None,
            }
        }
        Expr::Pow(base, exp) => {
            let (base, exp) = (*base, *exp);
            if is_affine_degree_one(ctx, base, var) {
                let x = as_rational_const(ctx, exp)?;
                Some((BigRational::one(), base, x, BigRational::zero()))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// The coefficient of `var` in an AFFINE exponent: `Some(0)` for a constant,
/// `Some(k)` for `k*var + b`; `None` if the exponent is not affine in `var`.
pub(super) fn exponent_linear_rate(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<num_rational::BigRational> {
    use cas_solver_core::isolation_utils::contains_var;
    use num_traits::Zero;
    if !contains_var(ctx, expr, var) {
        return Some(num_rational::BigRational::zero());
    }
    match ctx.get(expr).clone() {
        Expr::Variable(s) if ctx.sym_name(s) == var => {
            Some(num_rational::BigRational::from_integer(1.into()))
        }
        Expr::Mul(l, r) => {
            if let Some(c) = cas_math::numeric_eval::as_rational_const(ctx, l) {
                return exponent_linear_rate(ctx, r, var).map(|rate| c * rate);
            }
            if let Some(c) = cas_math::numeric_eval::as_rational_const(ctx, r) {
                return exponent_linear_rate(ctx, l, var).map(|rate| rate * c);
            }
            None
        }
        Expr::Add(l, r) => {
            Some(exponent_linear_rate(ctx, l, var)? + exponent_linear_rate(ctx, r, var)?)
        }
        Expr::Sub(l, r) => {
            Some(exponent_linear_rate(ctx, l, var)? - exponent_linear_rate(ctx, r, var)?)
        }
        Expr::Neg(inner) => exponent_linear_rate(ctx, inner, var).map(|rate| -rate),
        _ => None,
    }
}

/// Accumulate `expr` into an affine form `k·x + Σ const_terms` in `var`, walking `Add`/`Sub`/`Neg`
/// so signed constant parts and a `Sub`-form variable term (`a - x`) are captured. `sign_positive`
/// carries the running sign of this subtree. Returns `false` (caller bails) if the variable appears
/// in a non-rational-linear position. Const terms are recorded with their sign for exact rebuild.
pub(super) fn collect_affine_terms_in_var(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var: &str,
    sign_positive: bool,
    k: &mut num_rational::BigRational,
    const_terms: &mut Vec<(ExprId, bool)>,
) -> bool {
    use cas_solver_core::isolation_utils::contains_var;
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            collect_affine_terms_in_var(ctx, *l, var, sign_positive, k, const_terms)
                && collect_affine_terms_in_var(ctx, *r, var, sign_positive, k, const_terms)
        }
        Expr::Sub(l, r) => {
            collect_affine_terms_in_var(ctx, *l, var, sign_positive, k, const_terms)
                && collect_affine_terms_in_var(ctx, *r, var, !sign_positive, k, const_terms)
        }
        Expr::Neg(inner) => {
            collect_affine_terms_in_var(ctx, *inner, var, !sign_positive, k, const_terms)
        }
        _ => {
            if contains_var(ctx, expr, var) {
                match rational_coeff_of_bare_var(ctx, expr, var) {
                    Some(coeff) => {
                        if sign_positive {
                            *k += coeff;
                        } else {
                            *k -= coeff;
                        }
                        true
                    }
                    None => false,
                }
            } else {
                const_terms.push((expr, sign_positive));
                true
            }
        }
    }
}

/// `sin(x)=c` / `cos(x)=c` / `tan(x)=c` (bare trig of the solve variable, constant `c`) has an
/// INFINITE periodic family of roots; the unary-inverse path rewrites to `x = arctan(c)` and returns
/// only the principal root (`solve(tan(x)=1)→{π/4}`, dropping `+kπ`). Emit the full family as
/// `SolutionSet::Periodic { base, period }`:
///   tan(x)=c → {arctan(c) + kπ}        (period π, every constant c)
///   sin(x)=c → {arcsin(c) + …}         (period π for c=0, 2π for c=±1; other c are TWO families and
///   cos(x)=c → {arccos(c) + …}          cannot be a single `Periodic`, so they decline)
/// Only fires for an EQUATION (inequalities correctly residual elsewhere). `arcsin/arccos/arctan`
/// fold to the exact bound (`arctan(1)→π/4`, `arccos(0)→π/2`) via the simplifier.
/// The positive rational `a` of an argument `a·x` (`x → 1`, `2·x → 2`), else `None`. Used so the
/// periodic trig guard handles a SCALED argument `trig(a·x)=c`. An affine offset (`a·x+b`) or a
/// non-positive coefficient declines (kept clean: the family set is sign-insensitive but renders
/// awkwardly, and an offset shifts the base — out of this guard's scope).
/// Extract the AFFINE argument `a·x + b` (positive rational slope `a`, rational offset `b`) of a trig
/// call, so `sin(x − 1)`, `cos(2x + 1)` etc. are recognised — not only the pure `a·x` form. Returns
/// `(a, b)` with `a > 0`. Declines a non-affine argument (`x²`, `√x`) or a non-rational offset.
/// Affine argument `a·x + b` where the slope `a` is a VAR-FREE expression
/// with PROVABLY POSITIVE sign (π, 2π, √2, e, q·π …) and `b` is var-free —
/// the symbolic generalization of [`positive_affine_arg_of_var`] for the
/// final-audit family `sin(π·x) = 1` (the rational-only gate declined and
/// the principal-root isolation asserted `{ 1/2 }` as the complete answer).
/// Returns `(a_expr, b_expr)` simplified. Exactness: affinity is decided by
/// a vanishing second difference (exact rational or the linear-surd zero
/// oracle), positivity by `provable_const_sign` — no f64 anywhere.
pub(super) fn symbolic_positive_affine_arg_of_var(
    simplifier: &mut Simplifier,
    arg: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;
    let xvar = simplifier.context.var(var);
    let sample = |simplifier: &mut Simplifier, k: i64| -> ExprId {
        let kn = simplifier.context.num(k);
        let s = substitute_expr_by_id(&mut simplifier.context, arg, xvar, kn);
        simplifier.simplify(s).0
    };
    let g0 = sample(simplifier, 0);
    if contains_var(&simplifier.context, g0, var) {
        return None;
    }
    let g1 = sample(simplifier, 1);
    let g2 = sample(simplifier, 2);
    let a_raw = simplifier.context.add(Expr::Sub(g1, g0));
    let (a_expr, _) = simplifier.simplify(a_raw);
    if contains_var(&simplifier.context, a_expr, var) {
        return None;
    }
    // Second difference must vanish EXACTLY (affine): rational fold or the
    // linear-surd sign oracle; undecidable ⇒ decline (sound).
    let two_g1 = simplifier.context.add(Expr::Add(g1, g1));
    let g2_plus_g0 = simplifier.context.add(Expr::Add(g2, g0));
    let second = simplifier.context.add(Expr::Sub(g2_plus_g0, two_g1));
    let (second, _) = simplifier.simplify(second);
    let second_is_zero = match as_rational_const(&simplifier.context, second) {
        Some(r) => num_traits::Zero::is_zero(&r),
        None => {
            cas_math::root_forms::provable_sign_vs_zero(&simplifier.context, second)
                == Some(std::cmp::Ordering::Equal)
        }
    };
    if !second_is_zero {
        return None;
    }
    // Positive-slope convention, proven exactly (π-lattice, surds, e-powers).
    match cas_math::const_sign::provable_const_sign(&simplifier.context, a_expr) {
        Some(cas_math::const_sign::ConstSign::Positive) => Some((a_expr, g0)),
        _ => None,
    }
}

pub(super) fn positive_affine_arg_of_var(
    ctx: &Context,
    arg: ExprId,
    var: &str,
) -> Option<(num_rational::BigRational, num_rational::BigRational)> {
    use cas_math::polynomial::Polynomial;
    use num_traits::{Signed, Zero};
    let poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    if poly.degree() != 1 {
        return None;
    }
    let a = poly.coeffs.get(1).cloned()?;
    let b = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(num_rational::BigRational::zero);
    if !a.is_positive() {
        return None; // keep the positive-slope convention; a < 0 left to the existing path
    }
    Some((a, b))
}

/// Build `lo < g < hi` (or its complement) for an AFFINE `g = k·x + b` with
/// SYMMETRIC symbolic bounds `±hi_u` (`hi_u ≥ 0` by construction: it is
/// `ar*(r)` for `r ≥ 0`). The x-endpoints `(±hi_u − b)/k` are ordered by the
/// RATIONAL slope's sign — never by a symbolic comparator (the F7 trap the
/// core set algebra cannot avoid). Non-affine `g` declines.
pub(super) fn build_affine_symmetric_band_or_complement(
    simplifier: &mut Simplifier,
    g_poly: &cas_math::polynomial::Polynomial,
    hi_u: ExprId,
    op: cas_ast::RelOp,
) -> Option<SolutionSet> {
    use cas_ast::{BoundType, Interval, RelOp};
    use cas_solver_core::solution_set::{neg_inf, pos_inf};
    use num_traits::{Signed, Zero};

    if g_poly.degree() != 1 {
        return None;
    }
    let b = g_poly.coeffs[0].clone();
    let k = g_poly.coeffs[1].clone();
    if k.is_zero() {
        return None;
    }
    let lo_u = {
        let n = simplifier.context.add(Expr::Neg(hi_u));
        simplifier.simplify(n).0
    };
    let mut endpoint = |u: ExprId| -> ExprId {
        let b_expr = simplifier.context.add(Expr::Number(b.clone()));
        let k_expr = simplifier.context.add(Expr::Number(k.clone()));
        let shifted = simplifier.context.add(Expr::Sub(u, b_expr));
        let scaled = simplifier.context.add(Expr::Div(shifted, k_expr));
        simplifier.simplify(scaled).0
    };
    let (lo, hi) = if k.is_positive() {
        (endpoint(lo_u), endpoint(hi_u))
    } else {
        (endpoint(hi_u), endpoint(lo_u))
    };
    match op {
        RelOp::Lt | RelOp::Leq => {
            let bt = if matches!(op, RelOp::Lt) {
                BoundType::Open
            } else {
                BoundType::Closed
            };
            Some(SolutionSet::Continuous(Interval {
                min: lo,
                min_type: bt.clone(),
                max: hi,
                max_type: bt,
            }))
        }
        RelOp::Gt | RelOp::Geq => {
            let bt = if matches!(op, RelOp::Gt) {
                BoundType::Open
            } else {
                BoundType::Closed
            };
            let ninf = neg_inf(&mut simplifier.context);
            let pinf = pos_inf(&mut simplifier.context);
            Some(SolutionSet::Union(vec![
                Interval {
                    min: ninf,
                    min_type: BoundType::Open,
                    max: lo,
                    max_type: bt.clone(),
                },
                Interval {
                    min: hi,
                    min_type: bt,
                    max: pinf,
                    max_type: BoundType::Open,
                },
            ]))
        }
        _ => None,
    }
}

/// Extract the affine coefficients of `g = a·x + b` (slope `a` a nonzero rational, intercept `b` a
/// `var`-free expression) by sampling `g` at `x ∈ {0, 1, 2}`. Returns `None` if `g` is not affine in
/// `var` (the second difference is nonzero) or the slope is not a nonzero rational.
pub(super) fn affine_coefficients(
    simplifier: &mut Simplifier,
    g: ExprId,
    var: &str,
) -> Option<(num_rational::BigRational, ExprId)> {
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;
    let xvar = simplifier.context.var(var);
    let sample = |simplifier: &mut Simplifier, k: i64| -> ExprId {
        let kn = simplifier.context.num(k);
        let s = substitute_expr_by_id(&mut simplifier.context, g, xvar, kn);
        simplifier.simplify(s).0
    };
    let g0 = sample(simplifier, 0);
    if contains_var(&simplifier.context, g0, var) {
        return None; // intercept still depends on the variable ⇒ not affine
    }
    let g1 = sample(simplifier, 1);
    let g2 = sample(simplifier, 2);
    // slope `a = g(1) − g(0)`; second difference `g(2) − 2·g(1) + g(0)` must vanish (affine).
    let a_expr = simplifier.context.add(Expr::Sub(g1, g0));
    let (a_expr, _) = simplifier.simplify(a_expr);
    let a = as_rational_const(&simplifier.context, a_expr)?;
    if num_traits::Zero::is_zero(&a) {
        return None;
    }
    let two_g1 = simplifier.context.add(Expr::Add(g1, g1));
    let g2_plus_g0 = simplifier.context.add(Expr::Add(g2, g0));
    let second = simplifier.context.add(Expr::Sub(g2_plus_g0, two_g1));
    let (second, _) = simplifier.simplify(second);
    // The simplifier can leave an EXACTLY-ZERO surd combination uncollected
    // (`(√2−1) + (√2+1) − 2·√2` stays `√2+√2−√2−√2`), which `as_rational_const`
    // cannot read; decide exact zero via the linear-surd oracle. Undecidable ⇒
    // treat as not affine (sound decline).
    let second_is_zero = match as_rational_const(&simplifier.context, second) {
        Some(r) => num_traits::Zero::is_zero(&r),
        None => {
            cas_math::root_forms::provable_sign_vs_zero(&simplifier.context, second)
                == Some(std::cmp::Ordering::Equal)
        }
    };
    if !second_is_zero {
        return None;
    }
    Some((a, g0))
}

/// Map a periodic/discrete solution set for the atom `u = a·x + b` back to `x` via `x = (u − b)/a`:
/// each base becomes `(base − b)/a` and the period scales by `1/|a|` (kept positive). Declines a
/// non-periodic/non-discrete set (nothing to map soundly).
pub(super) fn map_solution_through_affine(
    simplifier: &mut Simplifier,
    sol: SolutionSet,
    a: &num_rational::BigRational,
    b: ExprId,
) -> Option<SolutionSet> {
    use num_traits::Signed;
    let a_num = simplifier.context.add(Expr::Number(a.clone()));
    let a_abs = simplifier.context.add(Expr::Number(a.abs()));
    let map_point = |simplifier: &mut Simplifier, base: ExprId| -> ExprId {
        let shifted = simplifier.context.add(Expr::Sub(base, b));
        let scaled = simplifier.context.add(Expr::Div(shifted, a_num));
        simplifier.simplify(scaled).0
    };
    match sol {
        SolutionSet::Empty => Some(SolutionSet::Empty),
        SolutionSet::Periodic { bases, period } => {
            let new_bases: Vec<ExprId> = bases
                .into_iter()
                .map(|base| map_point(simplifier, base))
                .collect();
            let scaled_period = simplifier.context.add(Expr::Div(period, a_abs));
            let (new_period, _) = simplifier.simplify(scaled_period);
            let mut new_bases = new_bases;
            dedup_bases_modulo_period(simplifier, &mut new_bases, new_period);
            Some(SolutionSet::Periodic {
                bases: new_bases,
                period: new_period,
            })
        }
        SolutionSet::Discrete(points) => {
            let mapped = points
                .into_iter()
                .map(|p| map_point(simplifier, p))
                .collect();
            Some(SolutionSet::Discrete(mapped))
        }
        _ => None,
    }
}

/// Map an interval bound through the inverse affine `x = (u − b)/a` (`a` rational ≠ 0,
/// `b` a constant ExprId). Infinities map to the ±∞ matching the sign of `a`.
pub(super) fn map_bound_through_inverse_affine(
    simplifier: &mut Simplifier,
    bound: ExprId,
    a: &num_rational::BigRational,
    b: ExprId,
) -> ExprId {
    use cas_solver_core::solution_set::{is_infinity, is_neg_infinity, neg_inf, pos_inf};
    use num_traits::Signed;
    let a_positive = a.is_positive();
    if is_infinity(&simplifier.context, bound) {
        return if a_positive {
            pos_inf(&mut simplifier.context)
        } else {
            neg_inf(&mut simplifier.context)
        };
    }
    if is_neg_infinity(&simplifier.context, bound) {
        return if a_positive {
            neg_inf(&mut simplifier.context)
        } else {
            pos_inf(&mut simplifier.context)
        };
    }
    let a_num = simplifier.context.add(Expr::Number(a.clone()));
    let shifted = simplifier.context.add(Expr::Sub(bound, b));
    let scaled = simplifier.context.add(Expr::Div(shifted, a_num));
    simplifier.simplify(scaled).0
}

/// Map a solution set in `u`-space back through `x = (u − b)/a`. A negative `a` reverses
/// interval orientation. Only structural sets are mapped; anything else declines.
pub(super) fn map_set_through_inverse_affine(
    simplifier: &mut Simplifier,
    set: SolutionSet,
    a: &num_rational::BigRational,
    b: ExprId,
) -> Option<SolutionSet> {
    use num_traits::Signed;
    let map_interval = |simplifier: &mut Simplifier, iv: cas_ast::Interval| -> cas_ast::Interval {
        let new_min = map_bound_through_inverse_affine(simplifier, iv.min, a, b);
        let new_max = map_bound_through_inverse_affine(simplifier, iv.max, a, b);
        if a.is_positive() {
            cas_ast::Interval {
                min: new_min,
                min_type: iv.min_type,
                max: new_max,
                max_type: iv.max_type,
            }
        } else {
            cas_ast::Interval {
                min: new_max,
                min_type: iv.max_type,
                max: new_min,
                max_type: iv.min_type,
            }
        }
    };
    Some(match set {
        SolutionSet::Empty => SolutionSet::Empty,
        SolutionSet::AllReals => SolutionSet::AllReals,
        SolutionSet::Discrete(points) => SolutionSet::Discrete(
            points
                .into_iter()
                .map(|p| map_bound_through_inverse_affine(simplifier, p, a, b))
                .collect(),
        ),
        SolutionSet::Continuous(iv) => SolutionSet::Continuous(map_interval(simplifier, iv)),
        SolutionSet::Union(ivs) => {
            let mut mapped: Vec<cas_ast::Interval> = ivs
                .into_iter()
                .map(|iv| map_interval(simplifier, iv))
                .collect();
            if !a.is_positive() {
                mapped.reverse(); // keep ascending order after the flip
            }
            SolutionSet::Union(mapped)
        }
        _ => return None, // Residual / Conditional / Periodic: nothing sound to map here
    })
}

/// Degree of a single additive TERM as a polynomial in `var` (`3·p·x²` -> 2), with
/// coefficients possibly SYMBOLIC. Returns `None` if `var` occurs in any non-polynomial
/// position (inside a function, a non-integer/negative exponent, a denominator). A leading
/// `Neg` carries no degree. Mirrors `root_forms::extract_term_degree_in_var` on an immutable
/// context.
fn term_degree_in_var(ctx: &cas_ast::Context, term: ExprId, var: &str) -> Option<u32> {
    if let Expr::Neg(inner) = ctx.get(term) {
        return term_degree_in_var(ctx, *inner, var);
    }
    let mut degree = 0u32;
    for factor in cas_math::expr_nary::mul_leaves(ctx, term) {
        match ctx.get(factor) {
            Expr::Neg(inner) => {
                degree = degree.checked_add(term_degree_in_var(ctx, *inner, var)?)?;
            }
            Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var => {
                degree = degree.checked_add(1)?;
            }
            Expr::Pow(base, exp) => match (ctx.get(*base), ctx.get(*exp)) {
                (Expr::Variable(sym_id), Expr::Number(n))
                    if ctx.sym_name(*sym_id) == var
                        && n.is_integer()
                        && *n >= num_rational::BigRational::from_integer(0.into()) =>
                {
                    let power: u32 = n.to_integer().try_into().ok()?;
                    degree = degree.checked_add(power)?;
                }
                _ => {
                    if cas_ast::collect_variables(ctx, factor).contains(var) {
                        return None;
                    }
                }
            },
            _ => {
                if cas_ast::collect_variables(ctx, factor).contains(var) {
                    return None;
                }
            }
        }
    }
    Some(degree)
}

/// Degree of `expr` as a polynomial in `var` (max over its additive terms), coefficients
/// possibly symbolic. `None` if `expr` is not a clean polynomial in `var` (see
/// [`term_degree_in_var`]). Used to recognise a leaked degree>=3 symbolic-coefficient
/// polynomial equation and echo it honestly instead of the self-referential radical.
pub(super) fn symbolic_poly_degree_in_var(
    ctx: &cas_ast::Context,
    expr: ExprId,
    var: &str,
) -> Option<u32> {
    let mut max_degree = 0u32;
    for term in cas_math::expr_nary::add_leaves(ctx, expr) {
        max_degree = max_degree.max(term_degree_in_var(ctx, term, var)?);
    }
    Some(max_degree)
}

/// Solve a BIQUADRATIC equation `a·x⁴ + b·x² + c = 0` (no odd-degree terms) by the substitution
/// `z = x²`: solve the quadratic `a·z² + b·z + c = 0`, then for each NON-NEGATIVE real `z` root take
/// `x = ±√z`. The normal path only handles biquadratics whose `x`-roots are rational; when they are
/// surds (`x⁴-8x²+15 → {±√3, ±√5}`, `x⁴-2x²-3 → {±√3}`) it leaks a circular residual.
///
/// EXACT (R2; auditoría 2026-07-30, ficha Q3c-005): every drop/keep decision here is exact — the
/// discriminant sign on `BigRational` and the `z`-root signs via the constant sign oracle on the
/// CONSTRUCTED `z` expression. The old f64 chain (disc sign in f64 → `z_f` sign with `1e-12` →
/// f64 back-substitution with `1e-9·scale` → value-dedup `1e-9`) FABRICATED `Empty` when
/// cancellation flipped the disc sign (`x⁴ − 2⁵⁰-ish·x² + …` with 4 existing roots → «No
/// solution»), and its docstring's safety claim was falsified: the failure mode is not a wrong
/// VALUE but a wrong SET. Roots need no back-substitution — the quadratic formula over exact
/// coefficients is an identity — and dedup is structural on the simplified interned ids. An
/// undecidable `z` sign declines (`None`, honest residual): a probe never fabricates a set.
/// Returns `None` for a non-biquadratic quartic or `Empty` (exact) when no real root exists.
pub(super) fn try_solve_biquadratic(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
    is_real_only: bool,
) -> Option<SolutionSet> {
    use cas_math::polynomial::Polynomial;
    use cas_solver_core::quadratic_formula::sqrt_expr;
    use num_rational::BigRational;
    use num_traits::{Signed, Zero};

    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (diff, _) = simplifier.simplify(diff);
    let poly = Polynomial::from_expr(&simplifier.context, diff, var).ok()?;
    if poly.degree() != 4 {
        return None;
    }
    let a = poly.coeffs[4].clone();
    let b = poly.coeffs[2].clone();
    let c = poly.coeffs[0].clone();
    // Biquadratic ⇒ the odd-degree coefficients vanish.
    if a.is_zero() || !poly.coeffs[3].is_zero() || !poly.coeffs[1].is_zero() {
        return None;
    }

    // Quadratic `a·z² + b·z + c` in `z = x²`: discriminant sign decided EXACTLY.
    let r = |n: i64| BigRational::from_integer(n.into());
    let disc = &b * &b - &a * &c * r(4);
    if disc.is_negative() {
        if is_real_only {
            // Complex z roots ⇒ no real x. Exact, so Empty is sound.
            return Some(SolutionSet::Empty);
        }
        // ComplexEnabled: both z roots are complex conjugates; build the four
        // `x = ±√z` EXACTLY — `√disc` over the negative rational folds to the
        // i-form downstream and the nested `√(u + i·v)` stays as the exact
        // closed form (the audit's b>0 twin pinned exactly that output). No
        // sign decisions are needed over ℂ, so no oracle and no f64.
        let ctx = &mut simplifier.context;
        let num = |ctx: &mut cas_ast::Context, v: BigRational| ctx.add(Expr::Number(v));
        let disc_node = num(ctx, disc);
        let sqrt_disc = sqrt_expr(ctx, disc_node);
        let neg_b = num(ctx, -&b);
        let two_a = num(ctx, &a * r(2));
        let mut raw_roots: Vec<ExprId> = Vec::new();
        for sgn in [1i8, -1i8] {
            let signed = if sgn > 0 {
                sqrt_disc
            } else {
                ctx.add(Expr::Neg(sqrt_disc))
            };
            let z_numer = ctx.add(Expr::Add(neg_b, signed));
            let z_expr = ctx.add(Expr::Div(z_numer, two_a));
            let sqrt_z = sqrt_expr(ctx, z_expr);
            let neg_sqrt_z = ctx.add(Expr::Neg(sqrt_z));
            raw_roots.push(sqrt_z);
            raw_roots.push(neg_sqrt_z);
        }
        let mut roots: Vec<ExprId> = Vec::new();
        for raw in raw_roots {
            let (root, _) = simplifier.simplify(raw);
            if !roots.contains(&root) {
                roots.push(root);
            }
        }
        return Some(SolutionSet::Discrete(roots));
    }

    // Build the exact `z = (−b ± √disc)/(2a)` and decide each z's SIGN with
    // the exact constant oracle on the constructed expression.
    let ctx = &mut simplifier.context;
    let num = |ctx: &mut cas_ast::Context, v: BigRational| ctx.add(Expr::Number(v));
    let disc_node = num(ctx, disc);
    let sqrt_disc = sqrt_expr(ctx, disc_node);
    let neg_b = num(ctx, -&b);
    let two_a = num(ctx, &a * r(2));
    let mut raw_roots: Vec<ExprId> = Vec::new();
    for s in [1i8, -1i8] {
        let signed = if s > 0 {
            sqrt_disc
        } else {
            ctx.add(Expr::Neg(sqrt_disc))
        };
        let z_numer = ctx.add(Expr::Add(neg_b, signed));
        let z_expr = ctx.add(Expr::Div(z_numer, two_a));
        let negative_z = match cas_math::const_sign::provable_const_sign(ctx, z_expr) {
            Some(cas_math::const_sign::ConstSign::Negative) => true,
            Some(_) => false,
            // Undecidable sign: never guess a set from it — decline to the
            // honest residual.
            None => return None,
        };
        if negative_z && is_real_only {
            continue; // z < 0 ⇒ x² = z has no real solution
        }
        // ComplexEnabled with z < 0: `±√z` folds to the pure-imaginary pair
        // downstream — exact either way, no separate handling needed.
        let sqrt_z = sqrt_expr(ctx, z_expr);
        let neg_sqrt_z = ctx.add(Expr::Neg(sqrt_z));
        raw_roots.push(sqrt_z);
        raw_roots.push(neg_sqrt_z);
    }

    // Roots hold by CONSTRUCTION (quadratic formula over exact coefficients);
    // dedup structurally on the simplified interned ids (disc = 0 double
    // root ⇒ identical expressions ⇒ identical ids).
    let mut roots: Vec<ExprId> = Vec::new();
    for raw in raw_roots {
        let (root, _) = simplifier.simplify(raw);
        if !roots.contains(&root) {
            roots.push(root);
        }
    }
    if roots.is_empty() {
        // Only reachable when real-only and every z was PROVABLY negative.
        return Some(SolutionSet::Empty);
    }
    Some(SolutionSet::Discrete(roots))
}

/// Solve a polynomial whose deflated quotient (after peeling rational roots) is a degree-4 factor that
/// splits into two rational quadratics — `x⁵-5x³+x²-5 = (x+1)(x²-5)(x²-x+1)` loses the `±√5` roots of
/// `x²-5` because the higher-degree path drops the quartic factor. This peels the rational roots,
/// factors the monic quartic quotient into `(x²+px+q)(x²+rx+s)`, solves each quadratic for its REAL
/// roots `(−p ± √(p²−4q))/2`, and returns the complete real set `rational_roots ∪ {quadratic roots}`.
/// Every root is verified by numeric back-substitution. An irreducible quartic quotient declines.
pub(super) fn try_solve_polynomial_with_quartic_factor(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
    is_real_only: bool,
) -> Option<SolutionSet> {
    use cas_math::polynomial::Polynomial;
    use cas_solver_core::quadratic_formula::sqrt_expr;
    use cas_solver_core::rational_roots::{find_rational_roots, rational_to_expr};
    use num_rational::BigRational;
    use num_traits::{ToPrimitive, Zero};
    const MAX_CANDIDATES: usize = 256;

    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (diff, _) = simplifier.simplify(diff);
    let poly = Polynomial::from_expr(&simplifier.context, diff, var).ok()?;
    if poly.degree() < 4 {
        return None;
    }
    let (rational_roots, quotient) = find_rational_roots(poly.coeffs.clone(), MAX_CANDIDATES);
    // The deflated quotient must be a degree-4 factor.
    if quotient.len() != 5 {
        return None;
    }
    // Normalize the quotient to MONIC. Dividing a polynomial by its (nonzero) leading coefficient
    // preserves its roots, so a content / scalar-multiple factor — `2·(x²-3)²` from
    // `2(x²-3)²(x-1)=0`, or the `4·(x²-3)²` of `(2x²-6)²(x-1)=0` — reduces to the monic `x⁴-6x²+9`
    // that `factor_monic_quartic_into_rational_quadratics` reads; otherwise the non-monic leading
    // coefficient made the factorizer decline and the repeated factor's irrational roots vanished.
    let lead = quotient[4].clone();
    if lead.is_zero() {
        return None;
    }
    let monic: Vec<BigRational> = quotient.iter().map(|cf| cf / &lead).collect();
    let int_of = |r: &BigRational| -> Option<i64> {
        if r.is_integer() {
            r.to_i64()
        } else {
            None
        }
    };
    let e = int_of(&monic[0])?;
    let d = int_of(&monic[1])?;
    let c = int_of(&monic[2])?;
    let b = int_of(&monic[3])?;
    let ((p1, q1), (p2, q2)) = factor_monic_quartic_into_rational_quadratics(b, c, d, e)?;

    // Solve each monic quadratic `x² + p·x + q` for its real roots `(−p ± √(p²−4q))/2`.
    let mut raw_roots: Vec<ExprId> = Vec::new();
    // Under ComplexEnabled the disc<0 conjugate pairs are emitted too (exact
    // roots of exact rational quadratic factors; `√(negative)` folds to the
    // i-form downstream). They bypass the f64 back-substitution below, which
    // rejects `i`.
    let mut exact_complex_roots: Vec<ExprId> = Vec::new();
    for (p, q) in [(p1, q1), (p2, q2)] {
        let disc = p * p - 4 * q;
        if disc < 0 && is_real_only {
            continue; // complex roots ⇒ no real solution from this factor
        }
        let ctx = &mut simplifier.context;
        let disc_node = ctx.add(Expr::Number(BigRational::from_integer(disc.into())));
        let sqrt_disc = sqrt_expr(ctx, disc_node);
        let neg_p = ctx.add(Expr::Number(BigRational::from_integer((-p).into())));
        let two = ctx.num(2);
        let plus = ctx.add(Expr::Add(neg_p, sqrt_disc));
        let minus = ctx.add(Expr::Sub(neg_p, sqrt_disc));
        let r_plus = ctx.add(Expr::Div(plus, two));
        let r_minus = ctx.add(Expr::Div(minus, two));
        if disc < 0 {
            exact_complex_roots.push(r_plus);
            exact_complex_roots.push(r_minus);
        } else {
            raw_roots.push(r_plus);
            raw_roots.push(r_minus);
        }
    }

    // Distinct rational roots (with multiplicity from `find_rational_roots`).
    let mut distinct_rationals: Vec<BigRational> = Vec::new();
    for root in &rational_roots {
        if !distinct_rationals.contains(root) {
            distinct_rationals.push(root.clone());
        }
    }
    let mut roots: Vec<ExprId> = distinct_rationals
        .iter()
        .map(|root| rational_to_expr(&mut simplifier.context, root))
        .collect();

    // Roots hold by CONSTRUCTION (exact rational roots from deflation; exact
    // quadratic formula over the integer factors — the discriminant sign at
    // the factor split above is already exact i64). The old f64
    // back-substitution with `1e-9·scale` and the value-dedup `1e-9` were
    // drop/keep decisions on data that is exact end to end (R2, auditoría
    // 2026-07-30 ficha Q3c-005): with large coefficients they discarded true
    // roots and, combined with `Empty` below, fabricated a wrong SET. Dedup
    // is structural on the simplified interned ids.
    for raw in raw_roots.into_iter().chain(exact_complex_roots) {
        let (root, _) = simplifier.simplify(raw);
        if !roots.contains(&root) {
            roots.push(root);
        }
    }
    if roots.is_empty() {
        // Exact everywhere above, so an empty set is a PROVEN empty set
        // (real-only with every factor's discriminant negative and no
        // rational roots).
        return Some(SolutionSet::Empty);
    }
    Some(SolutionSet::Discrete(roots))
}

/// Recover the degenerate `coefficient = 0` branch of a PARAMETRIC linear equation whose solution is a
/// constant. `a·x = a` (and `2a·x = 2a`, `a·x = 2a`, `a²·x = a²`) cancels the shared symbolic factor and
/// returns a bare `{1}`/`{2}`, silently dropping the `a ≠ 0` guard and the `a = 0 ⇒ ℝ` case — whereas
/// the structurally identical compound `(a-1)·x = a-1` correctly emits both. Re-applies the canonical
/// `build_linear_solution_set` branch logic.
///
/// Scoped tightly so it never disturbs an ordinary solve: it fires ONLY when the result is a single
/// NUMERIC root (so the coefficient genuinely cancelled) and the linear coefficient is NOT a non-zero
/// number (i.e. it is parametric). `2x = 4 → {2}` (numeric coefficient) and `a·x = b → {b/a}`
/// (non-numeric root) are both left untouched.
pub(super) fn try_parametric_linear_degenerate_branch(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
    root: ExprId,
) -> Option<SolutionSet> {
    use cas_ast::{Case, ConditionPredicate, ConditionSet};
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;
    use cas_solver_core::linear_form::linear_form;

    if !matches!(eq.op, cas_ast::RelOp::Eq) {
        return None;
    }
    // The solution must be a pure numeric constant — the tell that the coefficient cancelled.
    as_rational_const(&simplifier.context, root)?;

    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (diff, _) = simplifier.simplify(diff);
    let lf = linear_form(&mut simplifier.context, diff, var)?;
    let (coef, _) = simplifier.simplify(lf.coef);
    // The coefficient must be PARAMETRIC: a numeric coefficient (non-zero ⇒ ordinary equation needing
    // no branch; zero ⇒ not a linear solve in `var`) and a coefficient still containing the solve
    // variable are both left to the normal path.
    if as_rational_const(&simplifier.context, coef).is_some()
        || contains_var(&simplifier.context, coef, var)
    {
        return None;
    }
    // The equation is `coef·x = coef·root` (the numeric `root` solves `coef·x + constant = 0`, so
    // `constant = −coef·root`). Hence `root` is the unique solution when `coef ≠ 0`, and when
    // `coef = 0` the equation degenerates to `0 = 0` ⇒ all reals. Emit that two-case split — the guard
    // the bare `{root}` silently dropped.
    let nonzero_case = Case::new(
        ConditionSet::single(ConditionPredicate::NonZero(coef)),
        SolutionSet::Discrete(vec![root]),
    );
    let zero_case = Case::new(
        ConditionSet::single(ConditionPredicate::EqZero(coef)),
        SolutionSet::AllReals,
    );
    Some(SolutionSet::Conditional(vec![nonzero_case, zero_case]))
}

/// For a polynomial equation `p(x) = 0`, peel its rational roots and — if the deflated quotient is an
/// irreducible cubic — solve that cubic exactly (radical form for Δ > 0, trigonometric form for the
/// Δ < 0 *casus irreducibilis*). Returns the complete real set `rational_roots ∪ {cubic real roots}`,
/// or `None` when no degree-3 quotient remains (or it has Δ = 0). This closes BOTH the standalone
/// irreducible cubic (`x³+x²+3 = 0`, `x³-3x+1 = 0`) and the higher-degree case where the cubic factor
/// was dropped (`x⁴+x³+3x = x·(x³+x²+3)`).
/// Adapt an ASSOCIATED-`= 0` root recovery (cubic factor / quartic factor /
/// biquadratic) to the equation's relational op at its replace-the-incumbent
/// call sites. Those helpers solve `P = 0` and return its root set: under `=`
/// that IS the answer; under `!=` it is the exact set of NON-solutions —
/// publishing it verbatim caused the Cardano / casus-irreducibilis / quintic
/// wrong answers (`x³+x+1 ≠ 0 → {root}`, `x⁵−5x³+x²−5 ≠ 0 → {−1, ±√5}`).
/// Under `!=` a discrete recovery flips to its complement when the roots
/// order exactly, otherwise the honest incumbent (residual) stays; a rootless
/// recovery means the polynomial never vanishes, so `!= 0` holds everywhere.
/// Order inequalities keep the pre-existing replace behavior untouched
/// (their sign-analysis path answers before these recoveries fire today).
pub(super) fn adapt_associated_root_recovery_to_op(
    simplifier: &mut Simplifier,
    eq: &Equation,
    recovered: SolutionSet,
    incumbent: SolutionSet,
) -> SolutionSet {
    match eq.op {
        cas_ast::RelOp::Neq => match &recovered {
            SolutionSet::Discrete(roots) => cas_solver_core::solution_set::all_reals_except_points(
                &mut simplifier.context,
                roots,
            )
            .unwrap_or(incumbent),
            SolutionSet::Empty => SolutionSet::AllReals,
            _ => incumbent,
        },
        _ => recovered,
    }
}

pub(super) fn try_solve_polynomial_with_cubic_factor(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_math::polynomial::Polynomial;
    use cas_solver_core::rational_roots::{find_rational_roots, rational_to_expr};
    const MAX_CANDIDATES: usize = 256;

    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (diff, _) = simplifier.simplify(diff);
    let poly = Polynomial::from_expr(&simplifier.context, diff, var).ok()?;
    if poly.degree() < 3 {
        return None;
    }
    let (rational_roots, quotient) = find_rational_roots(poly.coeffs.clone(), MAX_CANDIDATES);
    // The deflated quotient must be exactly a cubic (degree 3 -> 4 coefficients).
    if quotient.len() != 4 {
        return None;
    }
    let cubic_roots = build_cubic_real_roots(
        simplifier,
        &quotient[3],
        &quotient[2],
        &quotient[1],
        &quotient[0],
    )?;
    // `find_rational_roots` returns roots WITH multiplicity (`x²·(…)` yields `0` twice); the engine
    // reports a DISTINCT-root set (`(x+1)³ → {-1}`), so dedup before emitting. The cubic roots are the
    // roots of an IRREDUCIBLE cubic, hence irrational — they can never collide with a rational root.
    let mut distinct_rationals: Vec<num_rational::BigRational> = Vec::new();
    for root in &rational_roots {
        if !distinct_rationals.contains(root) {
            distinct_rationals.push(root.clone());
        }
    }
    let mut roots: Vec<ExprId> = distinct_rationals
        .iter()
        .map(|root| rational_to_expr(&mut simplifier.context, root))
        .collect();
    roots.extend(cubic_roots);
    Some(SolutionSet::Discrete(roots))
}

/// Recover the degenerate `content = 0` branch of a PARAMETRIC polynomial product: the sibling of
/// [`try_parametric_linear_degenerate_branch`] for higher degrees. `y·(x−1)·(x+2) = 0` solved to a
/// bare `{−2, 1}` — the polynomial path divides the var-free parametric factor away, silently
/// dropping the `y = 0 ⇒ ℝ` case the TWO-factor spelling correctly emits. Fires only when:
/// - the simplified difference is a top-level product with ≥ 1 var-free PARAMETRIC factor
///   (symbolic, not a rational constant — `2·(x−1)(x+2)` needs no branch), and
/// - every var-carrying factor is POLYNOMIAL in `var` (total domain, so the `content = 0` branch
///   is exactly ℝ — a radical factor's zero branch would need the expression's domain instead,
///   which stays a named stepping stone), and
/// - the incumbent is the unconditional shape the division produced (`Discrete`/`Empty`).
pub(super) fn try_parametric_content_factor_branch(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
    incumbent: &SolutionSet,
) -> Option<SolutionSet> {
    use cas_ast::{Case, ConditionPredicate, ConditionSet};
    use cas_math::numeric_eval::as_rational_const;
    use cas_math::polynomial::Polynomial;
    use cas_solver_core::isolation_utils::contains_var;
    use num_traits::Zero;

    if !matches!(eq.op, cas_ast::RelOp::Eq) {
        return None;
    }
    if !matches!(incumbent, SolutionSet::Discrete(_) | SolutionSet::Empty) {
        return None;
    }
    // RAW tree on purpose: the simplifier EXPANDS `y·(x−1)·(x+2)` into the sum
    // `y·x² + y·x − 2y`, destroying exactly the product structure this hook
    // detects (the recurring «si simplify colapsa la estructura, árbol CRUDO»
    // discipline). The zero side must be a literal 0 for the raw other side to
    // BE the product.
    let zero_side = |ctx: &Context, e: ExprId| {
        cas_math::numeric_eval::as_rational_const(ctx, e).is_some_and(|c| c.is_zero())
    };
    let product = if zero_side(&simplifier.context, eq.rhs) {
        eq.lhs
    } else if zero_side(&simplifier.context, eq.lhs) {
        eq.rhs
    } else {
        return None;
    };

    // Split the top-level product into var-free and var-carrying factors.
    fn collect_factors(ctx: &Context, e: ExprId, out: &mut Vec<ExprId>) {
        match ctx.get(e) {
            Expr::Mul(a, b) => {
                let (a, b) = (*a, *b);
                collect_factors(ctx, a, out);
                collect_factors(ctx, b, out);
            }
            Expr::Neg(inner) => {
                let inner = *inner;
                collect_factors(ctx, inner, out);
            }
            _ => out.push(e),
        }
    }
    let mut factors = Vec::new();
    collect_factors(&simplifier.context, product, &mut factors);
    if factors.len() < 2 {
        return None;
    }
    let (var_free, var_carrying): (Vec<ExprId>, Vec<ExprId>) = factors
        .into_iter()
        .partition(|&f| !contains_var(&simplifier.context, f, var));
    if var_free.is_empty() || var_carrying.is_empty() {
        return None;
    }
    // Every var-carrying factor must be polynomial (total domain).
    for &q in &var_carrying {
        Polynomial::from_expr(&simplifier.context, q, var).ok()?;
    }
    // The content: product of the var-free factors. Parametric only — a
    // rational constant can never be zero-or-not, and a content still
    // carrying `var` was not var-free by construction.
    let mut content = var_free[0];
    for &f in &var_free[1..] {
        content = simplifier.context.add(Expr::Mul(content, f));
    }
    let (content, _) = simplifier.simplify(content);
    if as_rational_const(&simplifier.context, content).is_some() {
        return None;
    }
    if cas_ast::collect_variables(&simplifier.context, content).is_empty() {
        return None;
    }
    let nonzero_case = Case::new(
        ConditionSet::single(ConditionPredicate::NonZero(content)),
        incumbent.clone(),
    );
    let zero_case = Case::new(
        ConditionSet::single(ConditionPredicate::EqZero(content)),
        SolutionSet::AllReals,
    );
    Some(SolutionSet::Conditional(vec![nonzero_case, zero_case]))
}
