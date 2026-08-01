//! `solve_backend_local`: familia `inequalities`.
//!
//! Ver la cabecera de `solve_backend_local.rs` para el contexto.

use super::*;

/// Replace the result of a `sin(x)`/`cos(x)` inequality whose threshold is PROVABLY out of the
/// `[-1, 1]` range with the exact `ℝ` / `∅` answer (the generic monotonic inversion otherwise emits a
/// finite ray, sometimes with a non-real `arcsin(c)` endpoint). Only the unambiguous cases are
/// decided: a strictly out-of-range `c`, or the closed boundary (`c = 1` with `≤`/`>`, `c = -1` with
/// `≥`/`<`). The "touch" boundaries (`cos(x) < 1`, `cos(x) ≥ 1`, …) and `c ∈ (-1, 1)` exclude/include
/// only the periodic extremal points, which `ℝ`/`∅` cannot express, so they are left unchanged for
/// the residual path. Equations and non-bare-trig LHS are untouched.
pub(super) fn intersect_inequality_with_trig_range(
    ctx: &Context,
    eq: &Equation,
    var: &str,
    set: SolutionSet,
) -> SolutionSet {
    use cas_ast::RelOp;
    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return set;
    }
    if !bare_sin_or_cos_of_var(ctx, eq.lhs, var) {
        return set;
    }
    let Some(region) = classify_trig_threshold(ctx, eq.rhs) else {
        return set;
    };
    match (region, eq.op.clone()) {
        // c > 1: sin/cos < c always true; > c never.
        (TrigThresholdRegion::AboveRange, RelOp::Lt | RelOp::Leq) => SolutionSet::AllReals,
        (TrigThresholdRegion::AboveRange, RelOp::Gt | RelOp::Geq) => SolutionSet::Empty,
        // c < -1: sin/cos > c always; < c never.
        (TrigThresholdRegion::BelowRange, RelOp::Gt | RelOp::Geq) => SolutionSet::AllReals,
        (TrigThresholdRegion::BelowRange, RelOp::Lt | RelOp::Leq) => SolutionSet::Empty,
        // c = 1: `≤ 1` always true; `> 1` never. (`< 1` / `≥ 1` touch periodic points -> residual.)
        (TrigThresholdRegion::AtUpperBound, RelOp::Leq) => SolutionSet::AllReals,
        (TrigThresholdRegion::AtUpperBound, RelOp::Gt) => SolutionSet::Empty,
        // c = -1: `≥ -1` always; `< -1` never. (`> -1` / `≤ -1` touch -> residual.)
        (TrigThresholdRegion::AtLowerBound, RelOp::Geq) => SolutionSet::AllReals,
        (TrigThresholdRegion::AtLowerBound, RelOp::Lt) => SolutionSet::Empty,
        _ => set,
    }
}

/// Intersect a monotonic-function inequality result with the function's real
/// argument-domain, which the inversion drops — `solve(sqrt(x)<2) → [0,4)` (not
/// `(-∞,4)`), `solve(ln(x)<0) → (0,1)`, `solve(log(2,x)<3) → (0,8)`. EXACT and
/// EQ-safe: runs ONLY for the four inequality ops, ONLY when the LHS is
/// `√(x)`/`ln(x)`/`log(b,x)` over the BARE solve variable. It also folds the
/// even-root RANGE (`√ ≥ 0`), where squaring the threshold is invalid:
/// `sqrt(x)<-1 → ∅`, `sqrt(x)>-1 → [0,∞)`, `sqrt(x)<=0 → {0}`.
///
/// The half-line bound is simplified before intersecting so the interval gate is
/// an exact numeric comparison, never structural. A COMPOUND argument
/// (`sqrt(x-1)`) or a function on the RHS is an honest residual (returned as-is).
pub(super) fn intersect_inequality_with_function_domain(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
    set: SolutionSet,
) -> SolutionSet {
    use cas_ast::{BoundType, Interval, RelOp};
    use cas_solver_core::solution_set::{intersect_solution_sets, pos_inf};

    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return set;
    }
    if !matches!(set, SolutionSet::Continuous(_) | SolutionSet::Union(_)) {
        return set;
    }
    let Some((kind, arg)) = detect_monotonic_lhs(&simplifier.context, eq.lhs) else {
        return set;
    };

    // Argument domain over ℝ: even root → `{arg ≥ 0}`, ln/log → `{arg > 0}`.
    // For the BARE solve variable this is the half-line `[0,∞)` / `(0,∞)`. For a
    // COMPOUND argument (`√(x-1)`, `√(2x-1)`, `√(x²-4)`) the out-of-domain region
    // was previously KEPT (the inversion only constrained `arg` against the
    // threshold), so `√(x-1) < 3` wrongly returned `(-∞, 10)` instead of `[1, 10)`
    // — a wrong answer including points where the radicand is negative. Solve the
    // domain inequality `arg {≥,>} 0` for the variable so that region is excluded.
    let arg_is_var = matches!(simplifier.context.get(arg), Expr::Variable(s)
        if simplifier.context.sym_name(*s) == var);
    let domain = if arg_is_var {
        let domain_min_type = match kind {
            MonotonicFn::EvenRoot => BoundType::Closed,
            MonotonicFn::Log => BoundType::Open,
        };
        let ctx = &mut simplifier.context;
        let zero = ctx.num(0);
        let inf = pos_inf(ctx);
        SolutionSet::Continuous(Interval {
            min: zero,
            min_type: domain_min_type,
            max: inf,
            max_type: BoundType::Open,
        })
    } else {
        let domain_op = match kind {
            MonotonicFn::EvenRoot => RelOp::Geq,
            MonotonicFn::Log => RelOp::Gt,
        };
        let zero = simplifier.context.num(0);
        let domain_eq = Equation {
            lhs: arg,
            rhs: zero,
            op: domain_op,
        };
        // `arg` is non-radical here (the radical/log was peeled by
        // `detect_monotonic_lhs`), so this recursion is bounded. Bail to the
        // unchanged set only when the domain cannot be reduced to a clean
        // interval set (an honest no-worse-than-before fallback).
        match crate::solver_entrypoints_solve::solve(&domain_eq, var, simplifier) {
            Ok((
                d @ (SolutionSet::Continuous(_)
                | SolutionSet::Union(_)
                | SolutionSet::Empty
                | SolutionSet::AllReals),
                _,
            )) => d,
            _ => return set,
        }
    };

    // Even-root RANGE correction (`√ ≥ 0`): inverting squares the threshold `c`,
    // which is unsound when `c` is on the wrong side of 0 — handle those directly.
    if let MonotonicFn::EvenRoot = kind {
        // Decide the sign of `c` EXACTLY: a rational directly, else a constant linear surd
        // (`−√2`) via `provable_sign_vs_zero`, else the general exact value-bounds oracle
        // (`−e^(1/3)`). Without these paths, `√x < −√2` fell through to the (unsound)
        // squaring branch and returned `[0, 2)` instead of No solution.
        let sign = cas_math::numeric_eval::as_rational_const(&simplifier.context, eq.rhs)
            .map(|c| c.cmp(&num_rational::BigRational::from_integer(0.into())))
            .or_else(|| cas_math::root_forms::provable_sign_vs_zero(&simplifier.context, eq.rhs))
            .or_else(|| {
                use cas_math::const_sign::{provable_const_sign, ConstSign};
                Some(match provable_const_sign(&simplifier.context, eq.rhs)? {
                    ConstSign::Negative => std::cmp::Ordering::Less,
                    ConstSign::Zero => std::cmp::Ordering::Equal,
                    ConstSign::Positive => std::cmp::Ordering::Greater,
                })
            });
        if let Some(ord) = sign {
            let (neg, pos) = (
                ord == std::cmp::Ordering::Less,
                ord == std::cmp::Ordering::Greater,
            );
            match eq.op {
                // √ < c≤0 and √ ≤ c<0 are impossible (√ ≥ 0).
                RelOp::Lt if !pos => return SolutionSet::Empty,
                RelOp::Leq if neg => return SolutionSet::Empty,
                // √ > c<0 and √ ≥ c≤0 hold across the whole domain.
                RelOp::Gt if neg => return domain,
                RelOp::Geq if !pos => return domain,
                _ => {}
            }
        }
    }

    // Valid side: simplify the half-line bound (so the gate is exact), intersect.
    let set = simplify_solution_bounds(simplifier, set);
    intersect_solution_sets(&simplifier.context, set, domain)
}

/// Convert a `Discrete` solution set to degenerate closed intervals `[p, p]` so
/// `union_solution_sets` (which merges interval LISTS) keeps the points instead
/// of dropping them as a non-interval operand. Other variants pass through.
fn discrete_to_intervals(set: SolutionSet) -> SolutionSet {
    use cas_ast::domain::Interval;
    match set {
        SolutionSet::Discrete(points) => {
            let intervals: Vec<Interval> =
                points.into_iter().map(|p| Interval::closed(p, p)).collect();
            match intervals.len() {
                0 => SolutionSet::Empty,
                1 => SolutionSet::Continuous(intervals.into_iter().next().unwrap()),
                _ => SolutionSet::Union(intervals),
            }
        }
        other => other,
    }
}

/// If every interval of `set` is a degenerate point `[p, p]`, present it as a
/// `Discrete` set (`{p, …}`) — the engine's idiom for finite point sets — rather
/// than `[p, p] U …`. A mixed point/interval result (e.g. `{-2} ∪ [0, ∞)`) has no
/// `Discrete` representation and is left as-is.
fn collapse_degenerate_intervals(ctx: &Context, set: SolutionSet) -> SolutionSet {
    use cas_ast::domain::BoundType;
    use cas_solver_core::solution_set::compare_values;
    use std::cmp::Ordering;
    let intervals: &[cas_ast::domain::Interval] = match &set {
        SolutionSet::Continuous(i) => std::slice::from_ref(i),
        SolutionSet::Union(u) => u.as_slice(),
        _ => return set,
    };
    if intervals.is_empty()
        || !intervals.iter().all(|i| {
            i.min_type == BoundType::Closed
                && i.max_type == BoundType::Closed
                && compare_values(ctx, i.min, i.max) == Ordering::Equal
        })
    {
        return set;
    }
    SolutionSet::Discrete(intervals.iter().map(|i| i.min).collect())
}

/// Split a rational `lhs` into numerator/denominator POLYNOMIALS in `var` WITHOUT cancelling shared
/// factors (see [`rational_function_of`]). Handles sums (`x + 1/x`), products, quotients, negations and
/// integer powers; declines (returns `None`) for any non-polynomial leaf.
pub(super) fn split_rational_inequality_lhs(
    ctx: &mut Context,
    lhs: ExprId,
    var: &str,
) -> Option<(
    cas_math::polynomial::Polynomial,
    cas_math::polynomial::Polynomial,
)> {
    rational_function_of(ctx, lhs, var, 0)
}

/// Solve `N / D {op} c` with a polynomial denominator `D` (degree ≥ 1) and a var-free RHS `c`. With
/// `P = N − c·D`, the relation is `P/D {op} 0`: `P {op} 0` on the region `D > 0` and `P {flip op} 0`
/// on `D < 0` (poles `D = 0` excluded by the strict sign regions). This keeps every sub-solve to
/// `deg(P)`/`deg(D)` (≤ 4) — multiplying out to `(N−c·D)·D {op} 0` would push the polynomial degree
/// past the inequality solver's reliable range. A simpler shortcut otherwise reciprocates both sides
/// WITHOUT flipping (`1/(x²+1) < 1/2 → (-1,1)`, `1/x³ < 8 → (-∞,1/2)`, both wrong).
pub(super) fn try_solve_rational_constant_inequality(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    use cas_math::numeric_eval::as_rational_const;
    use cas_math::polynomial::Polynomial;
    use cas_solver_core::solution_set::{intersect_solution_sets, union_solution_sets};

    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    // Split the ORIGINAL `lhs` into numerator/denominator polynomials WITHOUT cancelling shared
    // factors (`x/(x³−x)` keeps `den = x³−x` and its pole at 0; `simplify` would cancel `x` and drop
    // the pole, which the verification — evaluating `den` — relies on). The splitter also folds the
    // reciprocal-power form `x^(-n)` / `c·x^(-n)` into `num/den` so `x^(-2) > 4` routes here too.
    let (num_poly, den_poly) = split_rational_inequality_lhs(&mut simplifier.context, eq.lhs, var)?;
    if den_poly.degree() < 1 {
        return None; // a constant denominator is the ordinary path's job
    }
    let c = as_rational_const(&simplifier.context, eq.rhs)?;

    let c_den = Polynomial::new(
        den_poly.coeffs.iter().map(|k| k * &c).collect(),
        var.to_string(),
    );
    let p_poly = num_poly.sub(&c_den); // P = N − c·D
                                       // `P ≡ 0` means `N/D = c` identically (off the poles) — a constant relation, not a genuine
                                       // inequality. Leave it to the dedicated removable-pole path, which renders the guarded
                                       // `R∖{poles}` Conditional (`(2x−4)/(x−2) ≥ 2`).
    if p_poly.is_zero() {
        return None;
    }
    // The interval algebra can mis-solve high-degree pieces, but the numeric verification gate below
    // now orders `n`-th-root bounds exactly AND the sign-analysis recovery handles `xⁿ - c` with a
    // rational root and a positive-definite residual, so a wrong candidate is REJECTED (declined)
    // rather than returned. Allow up to degree 12 — enough for reciprocal power inequalities `c/xⁿ`
    // through `1/x¹²` — and let verification be the soundness net for anything it cannot confirm.
    if p_poly.degree() > 12 || den_poly.degree() > 12 {
        return None;
    }

    // `P/D {op} 0`: keep `op` where `D > 0`, flip it where `D < 0`; `D = 0` excluded.
    let den_expr = den_poly.to_expr(&mut simplifier.context);
    let zero = simplifier.context.num(0);
    let d_pos = solve_relation_set(simplifier, var, den_expr, zero, RelOp::Gt)?;
    let d_neg = solve_relation_set(simplifier, var, den_expr, zero, RelOp::Lt)?;
    let p_same = solve_poly_sign(simplifier, var, &p_poly, eq.op.clone())?;
    let p_flip = solve_poly_sign(simplifier, var, &p_poly, flip_inequality(eq.op.clone()))?;
    let part_pos = intersect_solution_sets(&simplifier.context, p_same, d_pos);
    let part_neg = intersect_solution_sets(&simplifier.context, p_flip, d_neg);
    let candidate = union_solution_sets(&simplifier.context, part_pos, part_neg);

    // SOUNDNESS GATE. The sign-split is exact, but the interval algebra (intersection/union) is not
    // fully reliable: it mis-orders cube/fourth-root bounds, drops isolated points, and can fill a
    // punctured union. So never trust the candidate structurally — verify it numerically. Its
    // membership must match the truth of `N(r)/D(r) {op} c` at every rational sample `r` (a pole
    // `D(r) = 0` puts `r` outside the domain). Membership is decided EXACTLY for rational and
    // quadratic-surd (`A + B·√n`, incl. `φ`) bounds; a higher-surd bound the check cannot order makes
    // verification fail, so the case declines and keeps its prior behaviour instead of gaining a
    // fresh wrong answer.
    if rational_inequality_candidate_verifies(
        &simplifier.context,
        &candidate,
        &num_poly,
        &den_poly,
        &c,
        eq.op.clone(),
    ) {
        Some(candidate)
    } else {
        None
    }
}

/// Numerically verify a `N/D {op} c` candidate. Returns `true` iff candidate membership matches the
/// truth of `N(r)/D(r) {op} c` at every rational sample `r` (a pole `D(r) = 0` makes the relation
/// false — `r` is outside the domain). Returns `false` if any bound is not rational or a quadratic
/// surd the membership test can order exactly, or if any sample disagrees.
fn rational_inequality_candidate_verifies(
    ctx: &Context,
    candidate: &SolutionSet,
    num_poly: &cas_math::polynomial::Polynomial,
    den_poly: &cas_math::polynomial::Polynomial,
    c: &num_rational::BigRational,
    op: cas_ast::RelOp,
) -> bool {
    use cas_ast::{BoundType, Constant, Interval, RelOp};
    use cas_math::root_forms::as_linear_surd;
    use cas_solver_core::solution_set::{is_infinity, is_neg_infinity};
    use num_rational::BigRational;
    use num_traits::Zero;
    use std::cmp::Ordering;

    // The quadratic-surd form `a + b·√n` of a bound (the golden ratio `φ = ½ + ½·√5` is emitted as
    // the bare `Φ`/`−Φ` constant, which `as_linear_surd` leaves unfolded). `None` => not orderable.
    fn bound_surd(ctx: &Context, e: ExprId) -> Option<(BigRational, BigRational, BigRational)> {
        if let Some(t) = as_linear_surd(ctx, e) {
            return Some(t);
        }
        let half = BigRational::new(1.into(), 2.into());
        let five = BigRational::from_integer(5.into());
        match ctx.get(e) {
            Expr::Constant(Constant::Phi) => Some((half.clone(), half, five)),
            Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Phi)) => {
                Some((-half.clone(), -half, five))
            }
            _ => None,
        }
    }
    // `± q^(1/n)` (a real `n`-th root of a non-negative rational `q`, possibly negated): the bound
    // shape produced by reciprocal power inequalities (`1/x³ > 2 → x < (1/2)^(1/3)`). Returns the
    // radicand `q ≥ 0`, the root `n ≥ 2`, and whether the whole bound is negated.
    fn bound_nth_root(ctx: &Context, e: ExprId) -> Option<(BigRational, u32, bool)> {
        use num_traits::{One, Signed, ToPrimitive, Zero};
        match ctx.get(e) {
            Expr::Neg(inner) => {
                let (q, n, neg) = bound_nth_root(ctx, *inner)?;
                Some((q, n, !neg))
            }
            Expr::Pow(base, exp) => {
                let er = cas_math::numeric_eval::as_rational_const(ctx, *exp)?;
                // Exponent must be `1/n` with `n ≥ 2`.
                if !er.numer().is_one() {
                    return None;
                }
                let n: u32 = er.denom().to_u32()?;
                if n < 2 {
                    return None;
                }
                let q = cas_math::numeric_eval::as_rational_const(ctx, *base)?;
                if q.is_zero() || q.is_positive() {
                    Some((q, n, false))
                } else if n % 2 == 1 {
                    // (−q)^(1/n) for odd n is the real root −(q^(1/n)).
                    Some((-q, n, true))
                } else {
                    None // even root of a negative: not real
                }
            }
            _ => None,
        }
    }
    // `r {?} bound`, exact. `None` if the bound is a surd we cannot order.
    fn cmp_to_bound(ctx: &Context, r: &BigRational, e: ExprId) -> Option<Ordering> {
        if let Some((a, b, n)) = bound_surd(ctx, e) {
            return Some(cmp_rational_to_quadratic_surd(r, &a, &b, &n));
        }
        let (q, n, neg) = bound_nth_root(ctx, e)?;
        Some(cmp_rational_to_nth_root(r, &q, n, neg))
    }
    fn interval_member(ctx: &Context, iv: &Interval, r: &BigRational) -> Option<bool> {
        let lo_ok = if is_neg_infinity(ctx, iv.min) {
            true
        } else {
            match cmp_to_bound(ctx, r, iv.min)? {
                Ordering::Greater => true,
                Ordering::Equal => iv.min_type == BoundType::Closed,
                Ordering::Less => false,
            }
        };
        let hi_ok = if is_infinity(ctx, iv.max) {
            true
        } else {
            match cmp_to_bound(ctx, r, iv.max)? {
                Ordering::Less => true,
                Ordering::Equal => iv.max_type == BoundType::Closed,
                Ordering::Greater => false,
            }
        };
        Some(lo_ok && hi_ok)
    }
    fn member(ctx: &Context, set: &SolutionSet, r: &BigRational) -> Option<bool> {
        match set {
            SolutionSet::Empty => Some(false),
            SolutionSet::AllReals => Some(true),
            SolutionSet::Discrete(pts) => {
                let mut hit = false;
                for p in pts {
                    if cmp_to_bound(ctx, r, *p)? == Ordering::Equal {
                        hit = true;
                    }
                }
                Some(hit)
            }
            SolutionSet::Continuous(iv) => interval_member(ctx, iv, r),
            SolutionSet::Union(ivs) => {
                let mut hit = false;
                for iv in ivs {
                    if interval_member(ctx, iv, r)? {
                        hit = true;
                    }
                }
                Some(hit)
            }
            _ => None, // Residual/Conditional: cannot verify → decline
        }
    }

    for k in -90i64..=90 {
        let r = BigRational::new(k.into(), 6.into());
        let d = den_poly.eval(&r);
        let truth = if d.is_zero() {
            false
        } else {
            let v = num_poly.eval(&r) / d;
            match op {
                RelOp::Lt => v < *c,
                RelOp::Leq => v <= *c,
                RelOp::Gt => v > *c,
                RelOp::Geq => v >= *c,
                _ => return false,
            }
        };
        match member(ctx, candidate, &r) {
            Some(m) if m == truth => {}
            _ => return false,
        }
    }
    true
}

pub(super) fn try_solve_radical_inequality(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    use cas_math::polynomial::Polynomial;
    use cas_solver_core::solution_set::{intersect_solution_sets, union_solution_sets};

    let op = eq.op.clone();
    if !matches!(op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (d, _) = simplifier.simplify(diff);

    let (s, coeff, f, rest) = collect_radical_split(&simplifier.context, d, var)?;
    // A coefficiented radical (`2√x + 1 < y`) is OUTSIDE this handler's scope:
    // its numeric cases already normalize to the unit form upstream
    // (`2√x < 4` → `[0, 4)`), so widening here would duplicate an owner, and
    // the parametric ones were never solved by anyone. Decline keeps today's
    // behavior byte-identical; the equation-side range-condition publisher is
    // the coefficient-aware consumer.
    if !coeff.is_unit() {
        return None;
    }
    // The radicand and the remainder must be sqrt-free (no nested / second radical
    // or a coefficiented radical hiding in `rest`).
    if expr_contains_sqrt(&simplifier.context, f) {
        return None;
    }
    // SOUNDNESS GATE: require a polynomial radicand of degree ≤ 2. A linear or
    // quadratic `f` has rational or quadratic-surd domain endpoints (`f ≥ 0`), and
    // every endpoint comparison in the case-split intersections is then between
    // quadratic surds — which `compare_values` now orders EXACTLY (including two
    // DISTINCT radicands, e.g. domain `√6` against constraint `√2−1`). A cubic or
    // higher radicand can have non-quadratic-surd roots that `as_surd_value` does
    // not model, so the intersection could mis-order them; decline those.
    match Polynomial::from_expr(&simplifier.context, f, var) {
        Ok(p) if p.degree() <= 2 => {}
        _ => return None,
    }
    let mut r = simplifier.context.num(0);
    for (sg, term) in rest {
        r = if sg >= 0 {
            simplifier.context.add(Expr::Add(r, term))
        } else {
            simplifier.context.add(Expr::Sub(r, term))
        };
    }
    if expr_contains_sqrt(&simplifier.context, r) {
        return None;
    }

    // `s·√f + r {op} 0`  ⇒  `√f {eff_op} g`.
    let (g, eff_op) = if s >= 0 {
        let neg_r = simplifier.context.add(Expr::Neg(r));
        (neg_r, op)
    } else {
        (r, flip_inequality(op))
    };

    // SOUNDNESS GATE: the RHS `g` must be AFFINE (degree ≤ 1). The `f ≶ g²` branch
    // is solved as `f − g²`, whose degree is `max(deg f, 2·deg g)`; with `deg f ≤ 2`
    // and `deg g ≤ 1` it stays ≤ 2, so its roots are quadratic surds that
    // `compare_values` orders exactly. A quadratic-or-higher `g` makes `g²` quartic+
    // (e.g. `√(9-x²) < x²` ⇒ `9-x² < x⁴`), whose roots are NOT quadratic surds —
    // `as_surd_value` returns `None` and the intersection mis-orders them. Decline.
    match Polynomial::from_expr(&simplifier.context, g, var) {
        Ok(p) if p.degree() <= 1 => {}
        _ => return None,
    }

    let zero = simplifier.context.num(0);
    // Build g² as an EXPANDED polynomial (not `Pow(g, 2)`): the simplifier keeps a
    // sloped affine RHS in factored form (`(1/2)x+5` ⇒ `1/2·(x+10)`), and squaring
    // that as `Pow(·, 2)` makes the downstream `f − g²` polynomial extraction drop
    // the squared outer rational factor — `√(x²-4) < (1/2)x+5` then wrongly leaked
    // `No solution`. The expanded form `1/4·x² + 5·x + 25` extracts cleanly.
    let g2 = {
        let g_poly = Polynomial::from_expr(&simplifier.context, g, var).ok()?;
        let g2_poly = g_poly.mul(&g_poly);
        g2_poly.to_expr(&mut simplifier.context)
    };
    // `f ≥ 0` can be a single POINT for a negative-definite radicand (`-x²` ⇒ {0});
    // present it as a degenerate interval so the case-split intersections keep it (a
    // bare `Discrete` operand collapses to ∅ in `intersect_solution_sets`).
    let f_nonneg = discrete_to_intervals(solve_relation_set(simplifier, var, f, zero, RelOp::Geq)?);

    // Solve by the case split. The non-strict (≤,≥) branches use CLOSED
    // sub-inequalities — these naturally close finite endpoints at the boundary
    // `√f = g`. The only ones that escape are *detached* touch points (e.g.
    // `√(x+3) ≤ -x-3` is exactly `{-3}` where `√0 = 0 = -x-3`), which the interval
    // intersection silently drops as a degenerate overlap; we recover those by
    // unioning `solve(√f = g)`. (The closed result has no finite OPEN endpoint, so
    // adding the boundary can never hit the `merge_intervals` min-not-extended
    // gap — that only bites when a closed point meets an open endpoint.)
    let closed_with_boundary =
        |simplifier: &mut Simplifier, core: SolutionSet| -> Option<SolutionSet> {
            // Boundary `√f = g` ⟺ `f = g² ∧ g ≥ 0` (`f = g² ≥ 0` is automatic). Solve
            // the POLYNOMIAL equation `f = g²` and keep roots with `g ≥ 0`: this avoids
            // the single-radical EQUATION solver, which leaks a residual on a fractional
            // RHS (`√(x²+4) = (1/3)x+2`), and reuses the already-expanded `g²`.
            let roots = solve_relation_set(simplifier, var, f, g2, RelOp::Eq)?;
            let boundary = keep_roots_with_g_nonneg(simplifier, var, roots, g);
            let boundary = discrete_to_intervals(boundary);
            let merged = union_solution_sets(&simplifier.context, boundary, core);
            Some(collapse_degenerate_intervals(&simplifier.context, merged))
        };

    let result = match eff_op {
        RelOp::Lt => {
            // f ≥ 0 ∧ g > 0 ∧ f < g²  (strict: open branches, no boundary point)
            let g_pos = solve_g_sign_condition(simplifier, var, g, RelOp::Gt)?;
            let f_lt = solve_relation_set(simplifier, var, f, g2, RelOp::Lt)?;
            let i = intersect_solution_sets(&simplifier.context, f_nonneg, g_pos);
            intersect_solution_sets(&simplifier.context, i, f_lt)
        }
        RelOp::Gt => {
            // f ≥ 0 ∧ (g < 0 ∨ f > g²)  (strict)
            let g_neg = solve_g_sign_condition(simplifier, var, g, RelOp::Lt)?;
            let f_gt = solve_relation_set(simplifier, var, f, g2, RelOp::Gt)?;
            let u = union_solution_sets(&simplifier.context, g_neg, f_gt);
            intersect_solution_sets(&simplifier.context, f_nonneg, u)
        }
        RelOp::Leq => {
            // f ≥ 0 ∧ g ≥ 0 ∧ f ≤ g²  (closed) ∪ detached `√f = g` points
            let g_nonneg = solve_g_sign_condition(simplifier, var, g, RelOp::Geq)?;
            let f_le = solve_relation_set(simplifier, var, f, g2, RelOp::Leq)?;
            let i = intersect_solution_sets(&simplifier.context, f_nonneg, g_nonneg);
            let core = intersect_solution_sets(&simplifier.context, i, f_le);
            closed_with_boundary(simplifier, core)?
        }
        RelOp::Geq => {
            // f ≥ 0 ∧ (g < 0 ∨ f ≥ g²)  (closed) ∪ detached `√f = g` points
            let g_neg = solve_g_sign_condition(simplifier, var, g, RelOp::Lt)?;
            let f_ge = solve_relation_set(simplifier, var, f, g2, RelOp::Geq)?;
            let u = union_solution_sets(&simplifier.context, g_neg, f_ge);
            let core = intersect_solution_sets(&simplifier.context, f_nonneg, u);
            closed_with_boundary(simplifier, core)?
        }
        _ => return None,
    };
    Some(result)
}

/// Solve a polynomial-in-`ln(arg)` INEQUALITY `P(ln(x)) {op} 0` (`ln(x)^2 - 3·ln(x) + 2 < 0`, the
/// pure-square `ln(x)^2 - 4 < 0`, …) which the isolation path mis-reported as "No solution". Substitute
/// `u = ln(arg)`, solve the polynomial inequality `P(u) {op} 0` for the u-set, then map each u-interval
/// back through `ln` (a strictly increasing bijection `(0,∞) → ℝ`): `a < ln(x) < b ⟺ e^a < x < e^b`,
/// done by solving the single-`ln` bound relations and intersecting/uniting.
pub(super) fn try_solve_polynomial_in_log_inequality(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::{BoundType, BuiltinFn, Constant, Interval, RelOp};
    use cas_math::polynomial::Polynomial;
    use cas_solver_core::solution_set::{is_infinity, is_neg_infinity, neg_inf, pos_inf};
    use num_rational::BigRational;
    use num_traits::Zero;
    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (expr, _) = simplifier.simplify(diff); // P(ln(g))
    let atom = find_log_atom_containing_var(&simplifier.context, expr, var)?;
    // The atom must be `ln(g)` with `g` AFFINE in the variable (`g = a·x + b`, a ≠ 0). The back-sub
    // `u = ln(g) ∈ (p, q) ⟺ g ∈ (e^p, e^q) ⟺ x ∈ ((e^p − b)/a, (e^q − b)/a)` is then an affine image of
    // the exponential band (the bounds swap when a < 0). The bare `ln(x)` case is just `a = 1, b = 0`.
    let g_arg = match simplifier.context.get(atom) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && simplifier.context.is_builtin(*fn_id, BuiltinFn::Ln) =>
        {
            args[0]
        }
        _ => return None,
    };
    let g_poly = Polynomial::from_expr(&simplifier.context, g_arg, var).ok()?;
    if g_poly.degree() != 1 {
        return None; // non-affine argument (`ln(x²)`, `ln(sin x)`) — left to other paths
    }
    let a = g_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero); // slope
    let b = g_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero); // intercept
    if a.is_zero() {
        return None;
    }
    let u_var = "__lns_u";
    let u = simplifier.context.var(u_var);
    let u_expr = substitute_expr_by_id(&mut simplifier.context, expr, atom, u);
    if expr_contains_named_var(&simplifier.context, u_expr, var) {
        return None; // a second distinct log atom (or x elsewhere) remains
    }
    // EXPAND: the simplifier factors a difference of squares (`ln(x)^2 - 4 → (ln(x)-2)(ln(x)+2)`), which
    // `Polynomial::from_expr` cannot read; expanding restores the `u^2 - 4` monomial form.
    let u_expr = cas_math::expand_ops::expand(&mut simplifier.context, u_expr);
    // Degree ≥ 2 in u — a single `ln` (degree 1) is the ordinary monotonic isolation's job.
    if Polynomial::from_expr(&simplifier.context, u_expr, u_var)
        .ok()?
        .degree()
        < 2
    {
        return None;
    }
    let zero = simplifier.context.num(0);
    let u_eq = Equation {
        lhs: u_expr,
        rhs: zero,
        op: eq.op.clone(),
    };
    let (u_set, _) = crate::solver_entrypoints_solve::solve(&u_eq, u_var, simplifier).ok()?;

    // Map the u-set through `x = (e^u − b)/a`. `AllReals` (e.g. `ln(g)^2 + 1 > 0`) is the full band
    // `u ∈ (−∞, +∞)`, which maps to the DOMAIN `g > 0` — an affine half-line at `−b/a`, NOT `x > 0` —
    // so it runs through the same mapping as an open `(−∞, +∞)` interval.
    let u_intervals: Vec<Interval> = match u_set {
        SolutionSet::Empty => return Some(SolutionSet::Empty),
        SolutionSet::AllReals => {
            let lo = neg_inf(&mut simplifier.context);
            let hi = pos_inf(&mut simplifier.context);
            vec![Interval {
                min: lo,
                min_type: BoundType::Open,
                max: hi,
                max_type: BoundType::Open,
            }]
        }
        SolutionSet::Continuous(iv) => vec![iv],
        SolutionSet::Union(v) => v,
        _ => return None, // Discrete / Conditional / unsolved: leave to the existing path
    };
    // `g = e^u` (`e^(−∞) = 0`, `e^(+∞) = +∞`). Building the bound directly avoids the bound-comparator
    // (which could not order `1/e²` against `e²` and collapsed the band to ∅).
    let exp_of = |simplifier: &mut Simplifier, bound: ExprId| -> ExprId {
        let e = simplifier.context.add(Expr::Constant(Constant::E));
        let p = simplifier.context.add(Expr::Pow(e, bound));
        simplifier.simplify(p).0
    };
    // `x = (g − b)/a` for a finite g-bound.
    let affine_x = |simplifier: &mut Simplifier,
                    g_bound: ExprId,
                    a: &BigRational,
                    b: &BigRational|
     -> ExprId {
        let b_node = simplifier.context.add(Expr::Number(b.clone()));
        let diff = simplifier.context.add(Expr::Sub(g_bound, b_node));
        let a_node = simplifier.context.add(Expr::Number(a.clone()));
        let q = simplifier.context.add(Expr::Div(diff, a_node));
        simplifier.simplify(q).0
    };
    let a_pos = a > BigRational::zero();
    let mut x_intervals: Vec<Interval> = Vec::with_capacity(u_intervals.len());
    for iv in u_intervals {
        // x-image of the LOWER u-endpoint (`g = e^u`, `e^(−∞) = 0`, so always finite).
        let g_lo = if is_neg_infinity(&simplifier.context, iv.min) {
            simplifier.context.num(0)
        } else {
            exp_of(simplifier, iv.min)
        };
        let x_lo_img = affine_x(simplifier, g_lo, &a, &b);
        // x-image of the UPPER u-endpoint (finite, or `+∞ ↦ +∞` (a>0) / `−∞` (a<0)).
        let x_hi_img = if is_infinity(&simplifier.context, iv.max) {
            if a_pos {
                pos_inf(&mut simplifier.context)
            } else {
                neg_inf(&mut simplifier.context)
            }
        } else {
            let g_hi = exp_of(simplifier, iv.max);
            affine_x(simplifier, g_hi, &a, &b)
        };
        // Increasing (a>0): the lower u-endpoint is the lower x-bound. Decreasing (a<0): swapped.
        let interval = if a_pos {
            Interval {
                min: x_lo_img,
                min_type: iv.min_type,
                max: x_hi_img,
                max_type: iv.max_type,
            }
        } else {
            Interval {
                min: x_hi_img,
                min_type: iv.max_type,
                max: x_lo_img,
                max_type: iv.min_type,
            }
        };
        x_intervals.push(interval);
    }
    Some(if x_intervals.len() == 1 {
        SolutionSet::Continuous(x_intervals.pop().unwrap())
    } else {
        SolutionSet::Union(x_intervals)
    })
}

/// Solve a polynomial-in-`x^(1/q)` INEQUALITY (`x − 3·√x + 2 < 0`, a quadratic in `√x`;
/// `x^(2/3) − x^(1/3) − 2 < 0`, a quadratic in `x^(1/3)`) which the isolation path mis-reports as an
/// honest-but-incomplete residual. Substitute `u = x^(1/q)`, solve the polynomial inequality `P(u) {op}
/// 0` for the u-set, then map each u-interval back through `x = u^q` (monotonic increasing on the valid
/// u-domain): even `q` ⇒ `u ≥ 0` (so the u-set is first intersected with `[0, ∞)` and `x ≥ 0`); odd `q`
/// ⇒ all reals. Mirrors [`try_solve_rational_power_polynomial`] (its equation sibling).
pub(super) fn try_solve_rational_power_polynomial_inequality(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::{BoundType, Interval, RelOp};
    use cas_math::polynomial::Polynomial;
    use cas_solver_core::solution_set::{
        intersect_solution_sets, is_infinity, is_neg_infinity, neg_inf, pos_inf,
    };
    use num_bigint::BigInt;
    use num_integer::Integer;
    use num_rational::BigRational;
    use num_traits::One;

    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (expr, _) = simplifier.simplify(diff); // radicals canonicalize to `x^(p/q)`
    let mut exps: Vec<BigRational> = Vec::new();
    if !collect_x_power_exponents(&simplifier.context, expr, var, &mut exps) || exps.is_empty() {
        return None;
    }
    let q_big = exps.iter().fold(BigInt::one(), |acc, e| acc.lcm(e.denom()));
    if q_big <= BigInt::one() {
        return None; // q == 1: a plain polynomial inequality, owned by the normal path
    }
    let u_var = "__rps_u";
    let u_expr = rebuild_x_powers_as_u(&mut simplifier.context, expr, var, u_var, &q_big);
    // Degree ≥ 2 in u — a single power (degree 1) is the ordinary monotonic isolation's job.
    if Polynomial::from_expr(&simplifier.context, u_expr, u_var)
        .ok()?
        .degree()
        < 2
    {
        return None;
    }
    let zero = simplifier.context.num(0);
    let u_eq = Equation {
        lhs: u_expr,
        rhs: zero,
        op: eq.op.clone(),
    };
    let (u_set, _) = crate::solver_entrypoints_solve::solve(&u_eq, u_var, simplifier).ok()?;

    // `u = x^(1/q)`: even q ⇒ `u ≥ 0` (and `x ≥ 0`); odd q ⇒ all reals.
    let q_even = q_big.is_even();
    let u_set = if q_even {
        let lo = simplifier.context.num(0);
        let hi = pos_inf(&mut simplifier.context);
        let dom = SolutionSet::Continuous(Interval {
            min: lo,
            min_type: BoundType::Closed,
            max: hi,
            max_type: BoundType::Open,
        });
        intersect_solution_sets(&simplifier.context, u_set, dom)
    } else {
        u_set
    };
    let u_intervals: Vec<Interval> = match u_set {
        SolutionSet::Empty => return Some(SolutionSet::Empty),
        SolutionSet::AllReals => {
            // Every real `u` satisfies: `x` is the whole u-domain image (`[0, ∞)` for even q, ℝ for odd).
            if q_even {
                let lo = simplifier.context.num(0);
                let hi = pos_inf(&mut simplifier.context);
                return Some(SolutionSet::Continuous(Interval {
                    min: lo,
                    min_type: BoundType::Closed,
                    max: hi,
                    max_type: BoundType::Open,
                }));
            }
            return Some(SolutionSet::AllReals);
        }
        SolutionSet::Continuous(iv) => vec![iv],
        SolutionSet::Union(v) => v,
        _ => return None, // Discrete / Conditional / unsolved: leave to the existing path
    };
    // `x = u^q` is increasing on the valid u-domain, so each `(p, r) ↦ (p^q, r^q)` keeps its order and
    // bound types. Building the power directly avoids the bound-comparator.
    let pow_q = |simplifier: &mut Simplifier, bound: ExprId| -> ExprId {
        let qn = simplifier
            .context
            .add(Expr::Number(BigRational::from(q_big.clone())));
        let p = simplifier.context.add(Expr::Pow(bound, qn));
        simplifier.simplify(p).0
    };
    let mut x_intervals: Vec<Interval> = Vec::with_capacity(u_intervals.len());
    for iv in u_intervals {
        let (min, min_type) = if is_neg_infinity(&simplifier.context, iv.min) {
            (neg_inf(&mut simplifier.context), BoundType::Open) // odd q only
        } else {
            (pow_q(simplifier, iv.min), iv.min_type)
        };
        let (max, max_type) = if is_infinity(&simplifier.context, iv.max) {
            (pos_inf(&mut simplifier.context), BoundType::Open)
        } else {
            (pow_q(simplifier, iv.max), iv.max_type)
        };
        x_intervals.push(Interval {
            min,
            min_type,
            max,
            max_type,
        });
    }
    Some(if x_intervals.len() == 1 {
        SolutionSet::Continuous(x_intervals.pop().unwrap())
    } else {
        SolutionSet::Union(x_intervals)
    })
}

/// Decline a `log(x, c) {op} k` inequality (the variable is the BASE) to an honest residual: `logₓ(c)
/// = ln(c)/ln(x)` is non-monotonic (decreasing on `x > 1`, sign change at `x = 1`), so the engine's
/// monotonic log isolation emits a WRONG ray (and a `1/0 → undefined` bound when `k = 0`). With no
/// exact split representation, the sound outcome is a residual, not a fabricated interval. Equations
/// and constant-base `log(c, x)` (monotonic, solvable) are untouched.
pub(super) fn try_decline_variable_base_log_inequality(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    if !is_variable_base_log(&simplifier.context, eq.lhs, var) {
        return None;
    }
    Some(cas_solver_core::solve_outcome::residual_solution_set(
        &mut simplifier.context,
        eq.lhs,
        eq.rhs,
        eq.op.clone(),
        var,
    ))
}

/// Decline a PERIODIC trig inequality (`sin`/`cos`/`tan` of `var`) to an honest residual: its true
/// solution is an infinite PERIODIC UNION which the `SolutionSet` enum cannot represent, so the
/// monotonic inversion otherwise emits a single wrong ray. The bare `sin(x)`/`cos(x)` cases with a
/// threshold PROVABLY outside `[-1, 1]` are EXCLUDED — they are answered exactly (`ℝ`/`∅`) by the
/// trig-range guard after `solve_inner`, so they must not be pre-empted here. Equations are untouched
/// (op gate).
pub(super) fn try_decline_periodic_trig_inequality(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    let ctx = &simplifier.context;
    // Orientation-blind (PIU design-review P0): with the trig on the RHS
    // (`1/2 < sin(x)`, `2 < tan(x)`) the LHS-only check used to fall through
    // to the generic monotonic inversion, which asserted a WRONG ray like
    // `(π/6, ∞)`. Normalize to trig-on-LHS (swapping sides flips the
    // operator) and treat both orientations identically.
    let (lhs, rhs, op) = if contains_trig_of_var(ctx, eq.lhs, var) {
        (eq.lhs, eq.rhs, eq.op.clone())
    } else if contains_trig_of_var(ctx, eq.rhs, var) {
        (eq.rhs, eq.lhs, flip_inequality(eq.op.clone()))
    } else {
        return None;
    };
    // A bare sin/cos with an out-of-range / boundary threshold is solved exactly downstream — leave it.
    if bare_sin_or_cos_of_var(ctx, lhs, var) && classify_trig_threshold(ctx, rhs).is_some() {
        return None;
    }
    Some(cas_solver_core::solve_outcome::residual_solution_set(
        &mut simplifier.context,
        lhs,
        rhs,
        op,
        var,
    ))
}

/// A bare `sin(x)`/`cos(x)` inequality whose threshold is EXACTLY the range boundary `±1` (so the
/// generic monotonic inversion emits a wrong ray like `sin(x) ≥ 1 → [π/2, ∞)`). Two sub-cases:
/// - The TOUCH side (`sin(x) ≥ 1`, `sin(x) ≤ -1`, `cos(x) ≥ 1`, `cos(x) ≤ -1`) holds only where the
///   trig EQUALS the extreme value, so it reduces to the boundary equation `trig(x) = ±1` and returns
///   its periodic point set (`{π/2 + 2kπ}`) — exactly representable as `Periodic`.
/// - The COMPLEMENT side (`sin(x) < 1`, `cos(x) > -1`, …) is `ℝ` minus those periodic points, which
///   the `SolutionSet` enum cannot represent, so it declines to an honest residual (better than the
///   wrong ray). The other combinations (`sin(x) ≤ 1 → ℝ`, `sin(x) > 1 → ∅`) are answered by the
///   trig-range guard and are left untouched here.
pub(super) fn try_solve_boundary_trig_inequality(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    // Orientation-blind (PIU design-review P0): `1 > sin(x)` is the same
    // complement case as `sin(x) < 1` — normalize trig-on-LHS (swapping
    // sides flips the operator) so the RHS orientation cannot fall through
    // to the generic monotonic inversion's wrong ray.
    let (lhs, rhs, op) = if bare_sin_or_cos_of_var(&simplifier.context, eq.lhs, var) {
        (eq.lhs, eq.rhs, eq.op.clone())
    } else if bare_sin_or_cos_of_var(&simplifier.context, eq.rhs, var) {
        (eq.rhs, eq.lhs, flip_inequality(eq.op.clone()))
    } else {
        return None;
    };
    let region = classify_trig_threshold(&simplifier.context, rhs)?;
    match (region, op.clone()) {
        // sin(x) ≥ 1  ⇔  sin(x) = 1 ; cos(x) ≤ -1  ⇔  cos(x) = -1 : the periodic touch points.
        (TrigThresholdRegion::AtUpperBound, RelOp::Geq)
        | (TrigThresholdRegion::AtLowerBound, RelOp::Leq) => {
            let reduced = Equation {
                lhs,
                rhs,
                op: RelOp::Eq,
            };
            try_solve_periodic_trig_equation(&reduced, var, simplifier)
        }
        // sin(x) < 1 / sin(x) > -1: the COMPLEMENT ℝ∖{touch points}, not representable -> residual.
        (TrigThresholdRegion::AtUpperBound, RelOp::Lt)
        | (TrigThresholdRegion::AtLowerBound, RelOp::Gt) => {
            Some(cas_solver_core::solve_outcome::residual_solution_set(
                &mut simplifier.context,
                lhs,
                rhs,
                op,
                var,
            ))
        }
        // ≤ 1 → ℝ, > 1 → ∅, and the strictly out-of-range cases: left to the trig-range guard.
        _ => None,
    }
}

/// Solve an even-numerator VALLEY power inequality `c·(a·x+b)^(p/q) + d {op} k` exactly (p EVEN,
/// e = p/q > 0). Since `(α)^(p/q) = |α|^(p/q)` and that is increasing in `|α|`, the relation reduces to
/// `|α| {op'} ((k−d)/c)^(q/p)` (op' flips when c < 0), which splits into two linear pieces of the affine
/// argument. The reciprocal valleys (`e < 0`) are left to the decline. Returns `None` for the
/// non-valley shapes so the surrounding dispatch keeps its other behaviour.
pub(super) fn try_solve_even_power_valley_inequality(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::solution_set::{intersect_solution_sets, union_solution_sets};
    use num_integer::Integer;
    use num_rational::BigRational;
    use num_traits::{One, Signed, Zero};

    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    let k = as_rational_const(&simplifier.context, eq.rhs)?;
    let (c, alpha, exp, d) = extract_affine_power_term(&simplifier.context, eq.lhs, var)?;
    if c.is_zero() {
        return None;
    }
    // VALLEY only: e = p/q > 0 with EVEN numerator (so q is odd and `α^(p/q)` is defined for all α).
    if exp.denom().is_one() || !exp.is_positive() || !exp.numer().is_even() {
        return None;
    }
    // `(α)^e {op} m`, with `m = (k − d)/c` and `op` flipped if c < 0.
    let m = (&k - &d) / &c;
    let op = if c.is_negative() {
        flip_inequality(eq.op.clone())
    } else {
        eq.op.clone()
    };

    // `|α|^e {op} m`, `|α|^e ≥ 0`.  Handle the `m ≤ 0` degenerate cases, then the main `m > 0` reduction
    // `|α| {op} m^(q/p)`.
    let zero = simplifier.context.num(0);
    if !m.is_positive() {
        // m < 0: `|α|^e ≥ 0 > m` everywhere; m = 0: `|α|^e = 0` only at α = 0.
        return Some(match (&op, m.is_zero()) {
            (RelOp::Gt, false) | (RelOp::Geq, _) => SolutionSet::AllReals, // > m<0, ≥ m≤0
            (RelOp::Lt, _) | (RelOp::Leq, false) => SolutionSet::Empty,    // < m≤0, ≤ m<0
            (RelOp::Gt, true) => {
                // |α|^e > 0 ⟺ α ≠ 0.
                let lo = solve_relation_set(simplifier, var, alpha, zero, RelOp::Lt)?;
                let hi = solve_relation_set(simplifier, var, alpha, zero, RelOp::Gt)?;
                union_solution_sets(&simplifier.context, lo, hi)
            }
            (RelOp::Leq, true) => solve_relation_set(simplifier, var, alpha, zero, RelOp::Eq)?, // α = 0
            _ => return None,
        });
    }
    // m > 0: bound `B = m^(q/p) ≥ 0`.
    let m_expr = simplifier.context.add(Expr::Number(m));
    let qp = BigRational::new(exp.denom().clone(), exp.numer().abs());
    let qp_expr = simplifier.context.add(Expr::Number(qp));
    let bound = simplifier.context.add(Expr::Pow(m_expr, qp_expr));
    let (bound, _) = simplifier.simplify(bound);
    let neg_bound = simplifier.context.add(Expr::Neg(bound));
    let (neg_bound, _) = simplifier.simplify(neg_bound);
    // `|α| {op} B`: outside-the-band union for >, ≥; inside-the-band intersection for <, ≤.
    match op {
        RelOp::Gt | RelOp::Geq => {
            let hi = solve_relation_set(simplifier, var, alpha, bound, op.clone())?; // α {>,≥} B
            let lo = solve_relation_set(simplifier, var, alpha, neg_bound, flip_inequality(op))?; // α {<,≤} −B
            Some(union_solution_sets(&simplifier.context, lo, hi))
        }
        RelOp::Lt | RelOp::Leq => {
            let hi = solve_relation_set(simplifier, var, alpha, bound, op.clone())?; // α {<,≤} B
            let lo = solve_relation_set(simplifier, var, alpha, neg_bound, flip_inequality(op))?; // α {>,≥} −B
            Some(intersect_solution_sets(&simplifier.context, lo, hi))
        }
        _ => None,
    }
}

/// Decline a power-monomial inequality `c·x^e {op} k` whose exponent makes the engine's monotonic
/// isolation UNSOUND, to an honest residual. The isolation treats `x^e` as globally monotonic and
/// emits a single ray — correct ONLY when `e > 0` with an ODD numerator (a strictly monotonic power).
/// It is WRONG (1) for an EVEN numerator — `x^(2/3) = |x|^(2/3)` is a symmetric valley, so
/// `x^(2/3) > 2` truly has TWO rays `(−∞,−2√2)∪(2√2,∞)` but isolation drops the negative one; and
/// (2) for a NEGATIVE non-integer exponent (`1/x^(1/3)`, `1/√x`) — a reciprocal fractional power with
/// a pole at 0 and a sign jump that isolation mishandles (it returns the complement ray, or includes
/// the pole). Integer-exponent reciprocals (`1/x³`, `1/x²`) are EXCLUDED — they are solved exactly by
/// the rational-constant path. Only a rational-constant RHS is handled (the audited shape); equations
/// are untouched (op gate). Correctly solving the two-ray valleys and the reciprocal fractional powers
/// is the next capability rung; declining keeps the engine SOUND until then.
pub(super) fn try_decline_unsound_power_monomial_inequality(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    use cas_math::numeric_eval::as_rational_const;
    use num_integer::Integer;
    use num_traits::{One, Zero};
    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    // RHS must be a rational constant (the audited `x^e {op} k` shape).
    as_rational_const(&simplifier.context, eq.rhs)?;
    let exp = pure_power_monomial_exponent(&simplifier.context, eq.lhs, var)?;
    // Integer exponents (`x²`, `1/x³`) are owned by the polynomial / rational-constant paths.
    if exp.denom().is_one() {
        return None;
    }
    let numerator_even = exp.numer().is_even();
    let negative = exp < num_rational::BigRational::zero();
    if !(numerator_even || negative) {
        return None; // e > 0 with odd numerator: strictly monotonic, solved correctly — keep.
    }
    Some(cas_solver_core::solve_outcome::residual_solution_set(
        &mut simplifier.context,
        eq.lhs,
        eq.rhs,
        eq.op.clone(),
        var,
    ))
}

/// A degree-2 exponential inequality collapsed onto one side,
/// `A*base^(2x) + B*base^x {op} c` with NO `base^0` constant term (so `base^x`
/// factors out cleanly), is — since `base^x > 0` — equivalent to the single
/// exponential `base^x {op'} (-B/A)` (`op'` flips when `A < 0`). The single-side
/// terminal answers that even for a SYMBOLIC threshold (`e`, `pi`), where the
/// polynomial-in-u inequality solver rejects the symbolic coefficient. Fixes the
/// silent `e^(2x) - e*e^x < 0 -> {1}` (truth `(-inf,1)`) and the loud
/// `e^(2x) - pi*e^x < 0` "symbolic coefficient" error.
pub(super) fn try_solve_factorable_exponential_inequality(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: CoreSolverOptions,
    ctx: &SolveCtx,
) -> Option<SolutionSet> {
    use cas_solver_core::isolation_utils::contains_var;
    use num_traits::{Signed, Zero};

    if !matches!(
        eq.op,
        cas_ast::RelOp::Lt | cas_ast::RelOp::Leq | cas_ast::RelOp::Gt | cas_ast::RelOp::Geq
    ) {
        return None;
    }
    // Only the collapsed form (RHS constant in var). This also prevents re-entry
    // on the two-sided `base^x {op} threshold` this guard emits.
    if contains_var(&simplifier.context, eq.rhs, var) {
        return None;
    }

    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (diff, _) = simplifier.simplify(diff);

    let zero = simplifier.context.num(0);
    let atom = cas_solver_core::substitution::detect_exponential_substitution(
        &mut simplifier.context,
        diff,
        zero,
        var,
        true,
    )?;

    // Reduce by the positive factor base^x: expand(diff / base^x). A clean
    // factor-out (no `base^0` constant in the original) yields `A*base^x + B`;
    // a leftover `base^(-x)` term makes `collect_*` decline, so the constant-term
    // family is left to the substitution path.
    let quotient = simplifier.context.add(Expr::Div(diff, atom));
    let expanded = cas_math::expand_ops::expand(&mut simplifier.context, quotient);
    let (reduced, _) = simplifier.simplify(expanded);

    let mut atom_coeff = num_rational::BigRational::zero();
    let mut const_terms: Vec<(bool, ExprId)> = Vec::new();
    let collected = collect_linear_exponential_atom_terms(
        &simplifier.context,
        reduced,
        atom,
        var,
        true,
        &mut atom_coeff,
        &mut const_terms,
    );
    if collected.is_none() {
        // The cofactor is not linear in base^x. If it is still a CLEAN polynomial
        // in base^x (no `base^(-x)` term), the original had no `base^0` constant,
        // so this is a degree-3+ factor-out (`e^(3x)-e*e^x` -> `e^(2x)-e`): since
        // base^x > 0, re-solve `cofactor {op} 0` — the non-unit-exponent guard
        // (which runs before this one) answers the single `base^(k*x)` cofactor.
        // A leftover `base^(-x)` means a real constant term (e.g. B3
        // `e^(2x)-3e^x+2`), which the substitution path owns -> decline.
        if exponential_has_negative_rate(&simplifier.context, reduced, var) {
            return None;
        }
        let zero_rhs = simplifier.context.num(0);
        let reduced_eq = Equation {
            lhs: reduced,
            rhs: zero_rhs,
            op: eq.op.clone(),
        };
        return solve_local_core(&reduced_eq, var, simplifier, opts, ctx)
            .ok()
            .map(|(set, _)| set);
    }
    if atom_coeff.is_zero() {
        return None;
    }

    // threshold = -B / A, with B the signed sum of the constant terms.
    let mut b_sum = simplifier.context.num(0);
    for (positive, term) in const_terms {
        b_sum = if positive {
            simplifier.context.add(Expr::Add(b_sum, term))
        } else {
            simplifier.context.add(Expr::Sub(b_sum, term))
        };
    }
    let neg_b = simplifier.context.add(Expr::Neg(b_sum));
    let a_expr = simplifier.context.add(Expr::Number(atom_coeff.clone()));
    let threshold = simplifier.context.add(Expr::Div(neg_b, a_expr));
    let (threshold, _) = simplifier.simplify(threshold);

    // Dividing the relation by A flips the operator when A < 0.
    let op = if atom_coeff.is_positive() {
        eq.op.clone()
    } else {
        flip_inequality(eq.op.clone())
    };

    let reduced_eq = Equation {
        lhs: atom,
        rhs: threshold,
        op,
    };
    let (set, _) = solve_local_core(&reduced_eq, var, simplifier, opts, ctx).ok()?;
    Some(set)
}

/// A single exponential with a NON-UNIT integer exponent, `a*base^(k*x) + c {op} m`
/// (k >= 2, base > 1, RHS constant). The unit-exponent terminal cannot isolate
/// `base^(k*x)`, so isolate it to `base^(k*x) {op'} threshold` and, since
/// `base^(k*x)` is strictly increasing, recover the monotone ray from the
/// boundary EQUATION `base^(k*x) = threshold` (which the equation solver handles:
/// `solve(e^(2x)=e) -> {1/2}`). This is what lets the degree-3 factor-out cofactor
/// `e^(2x) - e < 0` resolve to `(-inf, 1/2)` — and never rewrites `base^(k*x)` to
/// `(base^k)^x` (the simplifier renormalizes that back, re-entering forever).
pub(super) fn try_solve_nonunit_exponential_inequality(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: CoreSolverOptions,
    ctx: &SolveCtx,
) -> Option<SolutionSet> {
    use cas_solver_core::isolation_utils::contains_var;
    use num_traits::{Signed, Zero};

    if !matches!(
        eq.op,
        cas_ast::RelOp::Lt | cas_ast::RelOp::Leq | cas_ast::RelOp::Gt | cas_ast::RelOp::Geq
    ) {
        return None;
    }
    // RHS constant in var (also blocks re-entry on the `base^x {op} c` shape the
    // boundary equation routes through).
    if contains_var(&simplifier.context, eq.rhs, var) {
        return None;
    }

    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (diff, _) = simplifier.simplify(diff);

    // The single exponential atom `base^(k*x)` — the REAL term, never the unit
    // atom `detect_exponential_substitution` would synthesize.
    let atom = find_first_exponential(&simplifier.context, diff, var)?;
    let pattern = cas_solver_core::isolation_utils::match_exponential_var_in_exponent(
        &simplifier.context,
        atom,
        var,
    )?;
    let base = pattern.base;
    let rate = exponent_linear_rate(&simplifier.context, pattern.exponent, var)?;
    let two = num_rational::BigRational::from_integer(2.into());
    if !rate.is_integer() || rate < two {
        return None;
    }

    // Isolate: the relation must be linear in `base^(k*x)` — `a*atom + c`. (A
    // second distinct exponential makes this decline, leaving the two-exponential
    // forms to the factor-out / substitution paths.)
    let mut a = num_rational::BigRational::zero();
    let mut const_terms: Vec<(bool, ExprId)> = Vec::new();
    collect_linear_exponential_atom_terms(
        &simplifier.context,
        diff,
        atom,
        var,
        true,
        &mut a,
        &mut const_terms,
    )?;
    if a.is_zero() {
        return None;
    }

    // threshold = -c / a, with c the signed sum of the constants.
    let mut c_sum = simplifier.context.num(0);
    for (positive, term) in const_terms {
        c_sum = if positive {
            simplifier.context.add(Expr::Add(c_sum, term))
        } else {
            simplifier.context.add(Expr::Sub(c_sum, term))
        };
    }
    let neg_c = simplifier.context.add(Expr::Neg(c_sum));
    let a_expr = simplifier.context.add(Expr::Number(a.clone()));
    let threshold = simplifier.context.add(Expr::Div(neg_c, a_expr));
    let (threshold, _) = simplifier.simplify(threshold);

    // Dividing the relation by `a` flips the operator when `a < 0`.
    let op = if a.is_positive() {
        eq.op.clone()
    } else {
        flip_inequality(eq.op.clone())
    };

    // SOUNDNESS: base must be provably > 1 (strictly increasing). EXACT: `e` and
    // `pi` are the known mathematical constants > 1 (no f64); a numeric base is
    // compared exactly. Anything else (fractional/symbolic) declines.
    let one = num_rational::BigRational::from_integer(1.into());
    let base_above_one = matches!(
        simplifier.context.get(base),
        Expr::Constant(cas_ast::Constant::E | cas_ast::Constant::Pi)
    ) || cas_math::numeric_eval::as_rational_const(&simplifier.context, base)
        .is_some_and(|value| value > one);
    if !base_above_one {
        return None;
    }

    // A provably non-positive threshold resolves by sign with no boundary
    // (`base^(k*x) > 0` always). This covers a symbolic-negative threshold like
    // `-e` (the boundary equation `base^(k*x) = -e` cannot prove no-real-solution
    // for a symbolic RHS, so it must be settled here): e.g. `e^(3x)+e*e^x < 0`
    // -> cofactor `e^(2x)+e < 0` -> threshold `-e` <= 0 -> No solution.
    if threshold_provably_nonpositive(&simplifier.context, threshold) {
        return Some(match op {
            cas_ast::RelOp::Gt | cas_ast::RelOp::Geq => SolutionSet::AllReals,
            _ => SolutionSet::Empty,
        });
    }

    // The boundary equation `base^(k*x) = threshold` decides the threshold sign
    // for us (delegated to the working equation solver, so it handles ANY
    // threshold — `2`, `e`, `e^2`, `sqrt(2)`, `2*e`, ...):
    //   - threshold > 0  => one real root x0; the monotone (base>1) ray `x {op} x0`.
    //   - threshold <= 0 => no real root (base^(k*x) > 0 always); resolve by sign.
    //   - anything else  => decline (unknown-sign symbolic threshold).
    let boundary_eq = Equation {
        lhs: atom,
        rhs: threshold,
        op: cas_ast::RelOp::Eq,
    };
    let (set, _) = solve_local_core(&boundary_eq, var, simplifier, opts, ctx).ok()?;
    match set {
        SolutionSet::Discrete(values) if values.len() == 1 => {
            Some(cas_solver_core::solution_set::isolated_var_solution(
                &mut simplifier.context,
                values[0],
                op,
            ))
        }
        SolutionSet::Empty => Some(match op {
            cas_ast::RelOp::Gt | cas_ast::RelOp::Geq => SolutionSet::AllReals,
            _ => SolutionSet::Empty,
        }),
        _ => None,
    }
}

/// A single-exponential inequality `a*base^x + c {op} k` (linear in `base^x`,
/// constant RHS) is isolated to the pure single exponential `base^x {op'} (k-c)/a`
/// (op' flips when a < 0), which the single-side terminal answers for EVERY base
/// and threshold — including a fractional base or a negative threshold
/// (`(1/2)^x - 4 > 0 -> (1/2)^x > 4 -> (-inf,-2)`; `(1/2)^x + 1 > 0 ->
/// (1/2)^x > -1 -> all reals`). Doing the isolation here (before the strategy
/// substitution, which would decline a fractional base to a residual) keeps the
/// additive family correct for all bases. A pure `base^x {op} k` (a==1, no
/// constant) is left to the terminal directly, which also prevents re-entry.
pub(super) fn try_isolate_single_exponential_inequality(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: CoreSolverOptions,
    ctx: &SolveCtx,
) -> Option<SolutionSet> {
    use cas_solver_core::isolation_utils::contains_var;
    use num_traits::{Signed, Zero};

    if !matches!(
        eq.op,
        cas_ast::RelOp::Lt | cas_ast::RelOp::Leq | cas_ast::RelOp::Gt | cas_ast::RelOp::Geq
    ) {
        return None;
    }
    if contains_var(&simplifier.context, eq.rhs, var) {
        return None;
    }

    let zero = simplifier.context.num(0);
    let atom = cas_solver_core::substitution::detect_exponential_substitution(
        &mut simplifier.context,
        eq.lhs,
        zero,
        var,
        true,
    )?;

    // lhs must be linear in base^x: `a*base^x + c` (a rational != 0, c constant).
    // A `base^(2x)` term makes the collect decline -> the degree-2 paths own it.
    let mut atom_coeff = num_rational::BigRational::zero();
    let mut const_terms: Vec<(bool, ExprId)> = Vec::new();
    collect_linear_exponential_atom_terms(
        &simplifier.context,
        eq.lhs,
        atom,
        var,
        true,
        &mut atom_coeff,
        &mut const_terms,
    )?;
    if atom_coeff.is_zero() {
        return None;
    }
    // Already a pure single exponential `base^x {op} k`: leave it to the terminal
    // (also prevents re-entry on the relation this guard emits).
    if atom_coeff == num_rational::BigRational::from_integer(1.into()) && const_terms.is_empty() {
        return None;
    }

    // threshold = (k - c) / a, with c the signed sum of the constant terms.
    let mut c_sum = simplifier.context.num(0);
    for (positive, term) in const_terms {
        c_sum = if positive {
            simplifier.context.add(Expr::Add(c_sum, term))
        } else {
            simplifier.context.add(Expr::Sub(c_sum, term))
        };
    }
    let k_minus_c = simplifier.context.add(Expr::Sub(eq.rhs, c_sum));
    let a_expr = simplifier.context.add(Expr::Number(atom_coeff.clone()));
    let threshold = simplifier.context.add(Expr::Div(k_minus_c, a_expr));
    let (threshold, _) = simplifier.simplify(threshold);

    let op = if atom_coeff.is_positive() {
        eq.op.clone()
    } else {
        flip_inequality(eq.op.clone())
    };

    let reduced_eq = Equation {
        lhs: atom,
        rhs: threshold,
        op,
    };
    let (set, _) = solve_local_core(&reduced_eq, var, simplifier, opts, ctx).ok()?;
    Some(set)
}

/// `|g(x)| {op} c` with a CONSTANT `c` is reduced to the polynomial inequalities on
/// the two sides of the abs, which the engine already solves correctly — the abs
/// *split* otherwise drops the operator and returns the boundary equation
/// (`|x^2-2x| < 1` -> "No solution"; `<=` -> the boundary points only). For `c > 0`:
///   `|g| < c`  <=>  `g < c` AND `g > -c`      `|g| > c`  <=>  `g > c` OR `g < -c`
/// and the `c <= 0` degenerate cases resolve by sign (`|g| >= 0` always). Declines
/// (-> the existing abs/isolation paths) for a sum of abs, a non-constant RHS, a
/// symbolic `c`, or a `g` whose polynomial-inequality solve is not concrete.
/// `A/|g(x)| ⋚ c` (A a nonzero rational constant, c constant): rewrite to the
/// exact twin `|A/g| ⋚ c` (A > 0; the A < 0 case negates and flips), which the
/// abs-threshold handler solves with correct pole puncturing — `1/|x| > 2` →
/// `(−1/2, 0) ∪ (0, 1/2)`, `1/|x| > 0` → `ℝ \ {0}`.
pub(super) fn try_solve_reciprocal_abs_inequality(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: CoreSolverOptions,
    ctx: &SolveCtx,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    use cas_solver_core::isolation_utils::contains_var;
    use num_traits::Signed;

    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    // Match `A / abs(g)` on one side (after simplify), constant on the other.
    let recip_abs = |ctx_: &Context, e: ExprId| -> Option<(num_rational::BigRational, ExprId)> {
        let (coeff, core) = peel_rational_coefficient(ctx_, e);
        if num_traits::Zero::is_zero(&coeff) {
            return None;
        }
        // `coeff · abs(g)^(−1)` (the canonical reciprocal shape) …
        if let Expr::Pow(base, exp) = ctx_.get(core) {
            let (base, exp) = (*base, *exp);
            let minus_one = num_rational::BigRational::from_integer((-1).into());
            if cas_math::numeric_eval::as_rational_const(ctx_, exp) == Some(minus_one) {
                if let Some(g) = match_abs_argument(ctx_, base) {
                    return Some((coeff, g));
                }
            }
        }
        // … or a literal `A / abs(g)` division.
        if let Expr::Div(num, den) = ctx_.get(core) {
            let (num, den) = (*num, *den);
            let a = cas_math::numeric_eval::as_rational_const(ctx_, num)?;
            if let Some(g) = match_abs_argument(ctx_, den) {
                return Some((coeff * a, g));
            }
        }
        None
    };

    let (lhs, _) = simplifier.simplify(eq.lhs);
    let (rhs, _) = simplifier.simplify(eq.rhs);
    let (a_coeff, g, c_expr, op) = if let Some((a, g)) = recip_abs(&simplifier.context, lhs) {
        if contains_var(&simplifier.context, rhs, var) {
            return None;
        }
        (a, g, rhs, eq.op.clone())
    } else if let Some((a, g)) = recip_abs(&simplifier.context, rhs) {
        if contains_var(&simplifier.context, lhs, var) {
            return None;
        }
        (a, g, lhs, flip_inequality(eq.op.clone()))
    } else {
        return None;
    };
    if !contains_var(&simplifier.context, g, var) {
        return None;
    }

    // A < 0: A/|g| = −(|A|/|g|); negate both sides (flips the operator).
    let (abs_a, op, c_expr) = if a_coeff.is_negative() {
        let neg_c = simplifier.context.add(Expr::Neg(c_expr));
        let (neg_c, _) = simplifier.simplify(neg_c);
        (-a_coeff, flip_inequality(op), neg_c)
    } else {
        (a_coeff, op, c_expr)
    };

    let a_expr = simplifier.context.add(Expr::Number(abs_a));
    let inner = simplifier.context.add(Expr::Div(a_expr, g));

    // c ≤ 0: the sign settles it (|A/g| > 0 wherever defined). Delegate to the
    // abs-threshold sign shortcut, which handles these without touching the
    // inner rational (verified: `1/|x| > 0` → ℝ∖{0}, `1/|x| > −1` → ℝ∖{0}).
    let c_val = cas_math::numeric_eval::as_rational_const(&simplifier.context, c_expr)?;
    if !c_val.is_positive() {
        let abs_call = simplifier.context.call("abs", vec![inner]);
        let reduced = Equation {
            lhs: abs_call,
            rhs: c_expr,
            op,
        };
        return try_solve_abs_threshold_inequality(&reduced, var, simplifier, opts, ctx);
    }

    // c > 0: solve the two rational relations on h = A/g DIRECTLY — the
    // const-over-g path punctures poles correctly (`2/x > 4` → (0, 1/2)).
    // (Routing through the abs-threshold instead re-normalizes `|A/g|` back to
    // `A/|g|` and falls into the pole-less path this handler exists to fix.)
    let neg_c = simplifier.context.add(Expr::Neg(c_expr));
    let (neg_c, _) = simplifier.simplify(neg_c);
    let mut solve_rel = |lhs: ExprId, rhs: ExprId, op: RelOp| -> Option<SolutionSet> {
        let rel = Equation { lhs, rhs, op };
        crate::solver_entrypoints_solve::solve(&rel, var, simplifier)
            .ok()
            .map(|(set, _)| set)
    };
    match op {
        RelOp::Gt | RelOp::Geq => {
            // |h| ⋛ c ⇔ h ⋛ c ∪ h ⋚ −c
            let (lo, hi) = if matches!(op, RelOp::Gt) {
                (RelOp::Lt, RelOp::Gt)
            } else {
                (RelOp::Leq, RelOp::Geq)
            };
            let upper = solve_rel(inner, c_expr, hi)?;
            let lower = solve_rel(inner, neg_c, lo)?;
            Some(cas_solver_core::solution_set::union_solution_sets(
                &simplifier.context,
                upper,
                lower,
            ))
        }
        RelOp::Lt | RelOp::Leq => {
            // |h| ⋚ c ⇔ h ⋚ c ∩ h ⋛ −c
            let (lo, hi) = if matches!(op, RelOp::Lt) {
                (RelOp::Gt, RelOp::Lt)
            } else {
                (RelOp::Geq, RelOp::Leq)
            };
            let upper = solve_rel(inner, c_expr, hi)?;
            let lower = solve_rel(inner, neg_c, lo)?;
            Some(cas_solver_core::solution_set::intersect_solution_sets(
                &simplifier.context,
                upper,
                lower,
            ))
        }
        _ => None,
    }
}

/// `|f(x)| {op} |g(x)|` with POLYNOMIAL arguments and an order operator
/// (`|x²−1| < |x+1|`): both sides are non-negative, so the relation is EXACTLY
/// `f² {op} g²` — one polynomial inequality, delegated to its correct owner via
/// the exact expanded difference `f² − g² {op} 0`. No handler owned this shape
/// (the single-abs handler requires ONE distinct abs, the threshold handler a
/// CONSTANT side, the multi-abs handler affine arguments), so it fell to the
/// generic path: "No solution" for `<`, boundary-only degenerate points for `≤`,
/// a mangled conditional leak for `>`. Linear-vs-linear (`|2x+1| > |x−1|`)
/// already has a correct owner and stays declined (degree gate).
pub(super) fn try_solve_abs_vs_abs_polynomial_inequality(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<Result<SolutionSet, CasError>> {
    use cas_ast::RelOp;
    use cas_math::polynomial::Polynomial;
    use cas_solver_core::isolation_utils::contains_var;

    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    let f_arg = match_abs_argument(&simplifier.context, eq.lhs)?;
    let g_arg = match_abs_argument(&simplifier.context, eq.rhs)?;
    if !contains_var(&simplifier.context, f_arg, var)
        || !contains_var(&simplifier.context, g_arg, var)
    {
        return None; // a constant side is the threshold handler's job
    }
    let (Ok(f_poly), Ok(g_poly)) = (
        Polynomial::from_expr(&simplifier.context, f_arg, var),
        Polynomial::from_expr(&simplifier.context, g_arg, var),
    ) else {
        // NON-POLYNOMIAL argument (`|ln(x)| < |x|`, `|e^x − 1| < |x|`): the generic
        // path fabricated a false "No solution" for `<` and mangled conditional
        // leaks for the other relations — DECLINE honestly. (The squared reduction
        // is still an EQUIVALENCE here, but the resulting transcendental relation
        // has no owner yet; this is the declared next step.)
        return Some(Err(CasError::SolverError(
            "relations between absolute values of non-polynomial expressions are not yet supported"
                .to_string(),
        )));
    };
    if f_poly.degree().max(g_poly.degree()) < 2 {
        return None; // affine-vs-affine already has a correct owner
    }
    // p = f² − g², built with EXACT polynomial arithmetic so it arrives expanded
    // and canonically collected (an unexpanded Mul shape can defeat the recursive
    // solver — the F17 lesson).
    let p_poly = f_poly.mul(&f_poly).sub(&g_poly.mul(&g_poly));
    let p_expr = p_poly.to_expr(&mut simplifier.context);
    let zero = simplifier.context.num(0);
    let set = solve_relation_set(simplifier, var, p_expr, zero, eq.op.clone())?;
    is_concrete_solution_set(&set).then_some(Ok(set))
}

/// `f(x)·g(x) {op} 0` where at least one factor is NOT polynomial-parseable
/// (`(x−1)·ln(x) < 0`, `x·e^x > 0`): split on the factor signs on the RAW tree.
/// `f·g < 0 ⟺ (f>0 ∧ g<0) ∪ (f<0 ∧ g>0)` and `f·g > 0 ⟺` the matching-signs
/// union; non-strict operators add the in-domain roots of `f·g = 0` (owned by the
/// equation path, which already filters domain). This must run on the RAW form:
/// the solve prepass DISTRIBUTES the product (`x·ln(x) − ln(x)`), after which the
/// Mul-isolation fallback divides by the variable-carrying factor KEEPING the
/// operator direction — `is_known_negative` is a constant oracle, so an unproven
/// sign silently became "assume positive" (`(x−1)·ln(x) < 0` → `(0, 1)`, truth:
/// no solution — both factors share the root x = 1 and the same sign elsewhere).
/// Polynomial·polynomial products stay with the polynomial sign-analysis owner.
pub(super) fn try_solve_product_inequality_sign_split(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    use cas_math::numeric_eval::as_rational_const;
    use cas_math::polynomial::Polynomial;
    use cas_solver_core::isolation_utils::contains_var;
    use cas_solver_core::solution_set::{intersect_solution_sets, union_solution_sets};
    use num_traits::Zero;

    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    // RHS must be literal zero: `f·g ⋚ k ≠ 0` does not reduce casewise.
    let k = as_rational_const(&simplifier.context, eq.rhs)?;
    if !k.is_zero() {
        return None;
    }
    // Peel leading negations into the operator; expect `Mul(f, g)` underneath.
    let mut op = eq.op.clone();
    let mut node = eq.lhs;
    while let Expr::Neg(inner) = simplifier.context.get(node) {
        node = *inner;
        op = cas_solver_core::isolation_utils::flip_inequality(op);
    }
    let (f, g) = match simplifier.context.get(node) {
        Expr::Mul(l, r) => (*l, *r),
        _ => return None,
    };
    // Both factors must carry the variable (a constant factor is ordinary
    // isolation), and at least one must be non-polynomial — the polynomial
    // product is owned by the correct sign-analysis path.
    if !contains_var(&simplifier.context, f, var) || !contains_var(&simplifier.context, g, var) {
        return None;
    }
    if Polynomial::from_expr(&simplifier.context, f, var).is_ok()
        && Polynomial::from_expr(&simplifier.context, g, var).is_ok()
    {
        return None;
    }

    fn interval_like(s: &SolutionSet) -> bool {
        matches!(
            s,
            SolutionSet::Empty
                | SolutionSet::AllReals
                | SolutionSet::Continuous(_)
                | SolutionSet::Union(_)
                | SolutionSet::Discrete(_)
        )
    }
    // Strictness carries into the factor sub-solves: for non-strict operators a zero
    // of EITHER factor (inside the other factor's domain) has product 0 and is
    // covered because `f = 0` belongs to both `f ≥ 0` and `f ≤ 0` — no separate
    // root union is needed (re-merging degenerate points into open endpoints is
    // exactly where boundary points get lost).
    let (op_pos, op_neg) = if matches!(op, RelOp::Gt | RelOp::Lt) {
        (RelOp::Gt, RelOp::Lt)
    } else {
        (RelOp::Geq, RelOp::Leq)
    };
    let zero = simplifier.context.num(0);
    let f_pos = solve_relation_set(simplifier, var, f, zero, op_pos.clone())?;
    let f_neg = solve_relation_set(simplifier, var, f, zero, op_neg.clone())?;
    let g_pos = solve_relation_set(simplifier, var, g, zero, op_pos)?;
    let g_neg = solve_relation_set(simplifier, var, g, zero, op_neg)?;
    if !(interval_like(&f_pos)
        && interval_like(&f_neg)
        && interval_like(&g_pos)
        && interval_like(&g_neg))
    {
        return None;
    }
    // SOUNDNESS gate (audit 2026-07-30, U1b-001): the interval algebra below
    // resolves min/max with `compare_values`, whose last resort is STRUCTURAL
    // order. With free-parameter endpoints (`a`, `b`) that order is invented —
    // `solve((x-a)*(x-b)<0)` published `(a, b)` unconditionally (false for
    // a ≥ b, and the `(x-a)^2*(x-b)` variant silently collapsed the `x ≠ a`
    // puncture). If ANY pair of finite endpoints across the four branch sets
    // has no exact value order, DECLINE — the expanded spelling of the same
    // inequality already declines honestly, and this route must not do worse.
    // Pairs a real oracle decides (numeric, surds, constant differences like
    // `a−3` vs `a+3`) pass through and keep their exact assembly.
    {
        let mut endpoints: Vec<cas_ast::ExprId> = Vec::new();
        for set in [&f_pos, &f_neg, &g_pos, &g_neg] {
            let intervals: &[cas_ast::Interval] = match set {
                SolutionSet::Continuous(interval) => std::slice::from_ref(interval),
                SolutionSet::Union(intervals) => intervals,
                _ => continue,
            };
            for interval in intervals {
                for bound in [interval.min, interval.max] {
                    if !cas_solver_core::solution_set::is_infinity(&simplifier.context, bound)
                        && !cas_solver_core::solution_set::is_neg_infinity(
                            &simplifier.context,
                            bound,
                        )
                        && !endpoints.contains(&bound)
                    {
                        endpoints.push(bound);
                    }
                }
            }
        }
        for (i, &low) in endpoints.iter().enumerate() {
            for &high in &endpoints[i + 1..] {
                cas_solver_core::solution_set::try_compare_values(&simplifier.context, low, high)?;
            }
        }
    }
    let want_positive = matches!(op, RelOp::Gt | RelOp::Geq);
    let (case_a, case_b) = if want_positive {
        (
            intersect_solution_sets(&simplifier.context, f_pos, g_pos),
            intersect_solution_sets(&simplifier.context, f_neg, g_neg),
        )
    } else {
        (
            intersect_solution_sets(&simplifier.context, f_pos, g_neg),
            intersect_solution_sets(&simplifier.context, f_neg, g_pos),
        )
    };
    Some(union_solution_sets(&simplifier.context, case_a, case_b))
}

pub(super) fn try_solve_abs_threshold_inequality(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: CoreSolverOptions,
    ctx: &SolveCtx,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    use cas_solver_core::isolation_utils::contains_var;
    use num_traits::{Signed, Zero};

    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    let (lhs, _) = simplifier.simplify(eq.lhs);
    let (rhs, _) = simplifier.simplify(eq.rhs);

    // Normalize to `abs(g) {op} c`, flipping the operator if the abs is on the right.
    let (g, op) = if let Some(g) = match_abs_argument(&simplifier.context, lhs) {
        if contains_var(&simplifier.context, rhs, var) {
            return None;
        }
        (g, eq.op.clone())
    } else if let Some(g) = match_abs_argument(&simplifier.context, rhs) {
        if contains_var(&simplifier.context, lhs, var) {
            return None;
        }
        (g, flip_inequality(eq.op.clone()))
    } else {
        return None;
    };
    if !contains_var(&simplifier.context, g, var) {
        return None;
    }
    // `c` is whichever side is constant.
    let c_expr = if match_abs_argument(&simplifier.context, lhs).is_some() {
        rhs
    } else {
        lhs
    };
    let c_value = cas_math::numeric_eval::as_rational_const(&simplifier.context, c_expr)?;

    // c <= 0: |g| >= 0, so the sign settles it with no boundary.
    if c_value.is_negative() {
        return Some(match op {
            RelOp::Gt | RelOp::Geq => SolutionSet::AllReals,
            _ => SolutionSet::Empty,
        });
    }
    let zero = simplifier.context.num(0);
    if c_value.is_zero() {
        return Some(match op {
            RelOp::Lt => SolutionSet::Empty,
            RelOp::Geq => SolutionSet::AllReals,
            // |g| <= 0  <=>  g = 0.
            RelOp::Leq => solve_concrete_side(g, zero, RelOp::Eq, var, simplifier, opts, ctx)?,
            // |g| > 0  <=>  g > 0 OR g < 0  (i.e. g != 0).
            RelOp::Gt => {
                let pos = solve_concrete_side(g, zero, RelOp::Gt, var, simplifier, opts, ctx)?;
                let neg = solve_concrete_side(g, zero, RelOp::Lt, var, simplifier, opts, ctx)?;
                cas_solver_core::solution_set::union_solution_sets(&simplifier.context, pos, neg)
            }
            _ => return None,
        });
    }

    // SYMBOLIC-CENTER band (F7 2026-07-14): `|k·x + b| {op} c` with a SYMBOLIC constant term `b`
    // (`|x - a| < 3`) reduces to `k·x+b {op} ±c`, whose endpoints `(±c − b)/k` are symbolic. The
    // set-algebra intersect/union cannot order two symbolic endpoints, so it collapses `<`/`<=` to
    // Empty and `>`/`>=` to AllReals. But the endpoint order is KNOWN from sign(k), so build the
    // band/rays DIRECTLY here. Numeric-center forms fall through to the existing (pinned) reduction.
    'symbolic_center: {
        use num_traits::Signed as _;
        // g = k·x + b with k a nonzero rational and b the (possibly symbolic) constant part. The
        // additive walk handles `Sub`/`Neg` (e.g. `|a - x|` extracts k = -1, b = a) which the
        // plain `add_leaves` split would miss.
        let mut k = num_rational::BigRational::zero();
        let mut const_terms: Vec<(ExprId, bool)> = Vec::new();
        if !collect_affine_terms_in_var(&simplifier.context, g, var, true, &mut k, &mut const_terms)
        {
            break 'symbolic_center;
        }
        if k.is_zero() {
            break 'symbolic_center;
        }
        // b as an expression; only take the DIRECT path when it is genuinely symbolic (the numeric
        // center keeps its existing owner and pinned fixtures).
        let mut b_expr = simplifier.context.num(0);
        for (t, positive) in &const_terms {
            b_expr = if *positive {
                simplifier.context.add(Expr::Add(b_expr, *t))
            } else {
                simplifier.context.add(Expr::Sub(b_expr, *t))
            };
        }
        let (b_expr, _) = simplifier.simplify(b_expr);
        if cas_math::numeric_eval::as_rational_const(&simplifier.context, b_expr).is_some() {
            break 'symbolic_center; // numeric center -> existing path
        }
        let k_expr = simplifier.context.add(Expr::Number(k.clone()));
        let neg_c_expr = simplifier.context.add(Expr::Number(-c_value.clone()));
        // x where g = +c and g = -c.
        let at_pos = {
            let n = simplifier.context.add(Expr::Sub(c_expr, b_expr));
            let d = simplifier.context.add(Expr::Div(n, k_expr));
            simplifier.simplify(d).0
        };
        let at_neg = {
            let n = simplifier.context.add(Expr::Sub(neg_c_expr, b_expr));
            let d = simplifier.context.add(Expr::Div(n, k_expr));
            simplifier.simplify(d).0
        };
        // Order endpoints by the sign of k (g increasing iff k > 0).
        let (lo, hi) = if k.is_positive() {
            (at_neg, at_pos)
        } else {
            (at_pos, at_neg)
        };
        let neg_inf = cas_solver_core::solution_set::neg_inf(&mut simplifier.context);
        let pos_inf = cas_solver_core::solution_set::pos_inf(&mut simplifier.context);
        let set = match op {
            RelOp::Lt => SolutionSet::Continuous(cas_ast::Interval {
                min: lo,
                min_type: cas_ast::BoundType::Open,
                max: hi,
                max_type: cas_ast::BoundType::Open,
            }),
            RelOp::Leq => SolutionSet::Continuous(cas_ast::Interval {
                min: lo,
                min_type: cas_ast::BoundType::Closed,
                max: hi,
                max_type: cas_ast::BoundType::Closed,
            }),
            RelOp::Gt => SolutionSet::Union(vec![
                cas_ast::Interval {
                    min: neg_inf,
                    min_type: cas_ast::BoundType::Open,
                    max: lo,
                    max_type: cas_ast::BoundType::Open,
                },
                cas_ast::Interval {
                    min: hi,
                    min_type: cas_ast::BoundType::Open,
                    max: pos_inf,
                    max_type: cas_ast::BoundType::Open,
                },
            ]),
            RelOp::Geq => SolutionSet::Union(vec![
                cas_ast::Interval {
                    min: neg_inf,
                    min_type: cas_ast::BoundType::Open,
                    max: lo,
                    max_type: cas_ast::BoundType::Closed,
                },
                cas_ast::Interval {
                    min: hi,
                    min_type: cas_ast::BoundType::Closed,
                    max: pos_inf,
                    max_type: cas_ast::BoundType::Open,
                },
            ]),
            _ => break 'symbolic_center,
        };
        return Some(set);
    }

    // c > 0: reduce to the two-sided inequality / the outside union.
    let neg_c = simplifier.context.add(Expr::Number(-c_value));
    let (upper_op, lower_op, conj) = match op {
        RelOp::Lt => (RelOp::Lt, RelOp::Gt, true),
        RelOp::Leq => (RelOp::Leq, RelOp::Geq, true),
        RelOp::Gt => (RelOp::Gt, RelOp::Lt, false),
        RelOp::Geq => (RelOp::Geq, RelOp::Leq, false),
        _ => return None,
    };
    let upper = solve_concrete_side(g, c_expr, upper_op, var, simplifier, opts, ctx)?;
    let lower = solve_concrete_side(g, neg_c, lower_op, var, simplifier, opts, ctx)?;
    // ORDER GUARD (2026-07-31, cubic abs — mirror of the sign-split
    // handler's): the combination below runs through the core set algebra,
    // whose endpoint order falls back to a VALUE-BLIND structural compare.
    // When the sign-split handler declines a cubic (undecidable
    // casus-irreducibilis TRIG endpoints), the relation falls HERE and the
    // same blind combination bridged disjoint regions (`|x³−4x| < 2`).
    // Any undecidable endpoint pair declines to an honest residual instead.
    let mut endpoints: Vec<ExprId> = Vec::new();
    for set in [&upper, &lower] {
        collect_finite_set_endpoints(&simplifier.context, set, &mut endpoints);
    }
    for i in 0..endpoints.len() {
        for j in (i + 1)..endpoints.len() {
            if cas_solver_core::solution_set::try_compare_values(
                &simplifier.context,
                endpoints[i],
                endpoints[j],
            )
            .is_none()
            {
                // Honest echo of the ORIGINAL relation (see the sign-split
                // handler's guard: falling through mangles the residual).
                return Some(cas_solver_core::solve_outcome::residual_solution_set(
                    &mut simplifier.context,
                    eq.lhs,
                    eq.rhs,
                    eq.op.clone(),
                    var,
                ));
            }
        }
    }
    let result = if conj {
        cas_solver_core::solution_set::intersect_solution_sets(&simplifier.context, upper, lower)
    } else {
        cas_solver_core::solution_set::union_solution_sets(&simplifier.context, upper, lower)
    };
    Some(result)
}

/// `coeff · ln(x)^2 {op} c` (constant `c`) is non-monotonic in `x`, so the log-isolation
/// path collapses it to the boundary equation and reports "All real numbers if x > 0"
/// (`ln(x)^2 > 1` -> wrong; truth `(0, 1/e) U (e, ∞)`). Reduce to the two SINGLE-`ln`
/// inequalities, which the engine solves exactly: with `u = ln(x)`,
///   `u^2 > t` (t>0) <=> `u > √t` OR `u < -√t`      `u^2 < t` <=> `-√t < u < √t`,
/// and the single-`ln` solver carries the `x > 0` domain through `x = e^u`. `t <= 0`
/// resolves by sign on the domain `(0, ∞)`. Only fires for a bare `ln(x)` (natural log
/// of the solve variable); `log_b(x)^2` already routes correctly or honestly residuals.
pub(super) fn try_solve_ln_square_inequality(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: CoreSolverOptions,
    ctx: &SolveCtx,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    use num_rational::BigRational;
    use num_traits::{One, Signed, Zero};

    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    let (lhs, _) = simplifier.simplify(eq.lhs);
    let (rhs, _) = simplifier.simplify(eq.rhs);
    let var_id = simplifier.context.var(var);

    // `coeff · ln(var)^2 {op} c`, flipping the operator if the square is on the right.
    let (coeff, ln_expr, op, c_value) = if let Some((coeff, ln_expr)) =
        match_ln_var_squared_with_coeff(&simplifier.context, lhs, var_id)
    {
        let c = cas_math::numeric_eval::as_rational_const(&simplifier.context, rhs)?;
        (coeff, ln_expr, eq.op.clone(), c)
    } else if let Some((coeff, ln_expr)) =
        match_ln_var_squared_with_coeff(&simplifier.context, rhs, var_id)
    {
        let c = cas_math::numeric_eval::as_rational_const(&simplifier.context, lhs)?;
        (coeff, ln_expr, flip_inequality(eq.op.clone()), c)
    } else {
        return None;
    };
    if coeff.is_zero() {
        return None;
    }
    // `ln^2 {op'} t`, `t = c / coeff` (flip the operator when `coeff < 0`).
    let t = c_value / &coeff;
    let op = if coeff.is_negative() {
        flip_inequality(op)
    } else {
        op
    };

    // `ln(x)^2 >= 0`, so a non-positive `t` settles by sign on the domain `(0, ∞)`.
    if t.is_negative() {
        return Some(match op {
            RelOp::Gt | RelOp::Geq => {
                cas_solver_core::solution_set::open_positive_domain(&mut simplifier.context)
            }
            _ => SolutionSet::Empty,
        });
    }
    if t.is_zero() {
        let zero = simplifier.context.num(0);
        return Some(match op {
            RelOp::Lt => SolutionSet::Empty,
            RelOp::Geq => {
                cas_solver_core::solution_set::open_positive_domain(&mut simplifier.context)
            }
            // ln(x)^2 <= 0  <=>  ln(x) = 0.
            RelOp::Leq => solve_concrete_side(ln_expr, zero, RelOp::Eq, var, simplifier, opts, ctx)
                .unwrap_or_else(|| {
                    cas_solver_core::solve_outcome::residual_solution_set(
                        &mut simplifier.context,
                        eq.lhs,
                        eq.rhs,
                        eq.op.clone(),
                        var,
                    )
                }),
            // ln(x)^2 > 0  <=>  ln(x) != 0.
            RelOp::Gt => {
                match (
                    solve_concrete_side(ln_expr, zero, RelOp::Gt, var, simplifier, opts, ctx),
                    solve_concrete_side(ln_expr, zero, RelOp::Lt, var, simplifier, opts, ctx),
                ) {
                    (Some(p), Some(n)) => cas_solver_core::solution_set::union_solution_sets(
                        &simplifier.context,
                        p,
                        n,
                    ),
                    _ => cas_solver_core::solve_outcome::residual_solution_set(
                        &mut simplifier.context,
                        eq.lhs,
                        eq.rhs,
                        eq.op.clone(),
                        var,
                    ),
                }
            }
            _ => return None,
        });
    }

    // t > 0: r = √t; reduce to the two single-`ln` inequalities around ±r.
    let t_expr = simplifier.context.add(Expr::Number(t));
    let half = simplifier
        .context
        .add(Expr::Number(BigRational::new(1.into(), 2.into())));
    let sqrt_t = simplifier.context.add(Expr::Pow(t_expr, half));
    let neg_one = simplifier.context.add(Expr::Number(-BigRational::one()));
    let neg_sqrt_t = simplifier.context.add(Expr::Mul(neg_one, sqrt_t));

    let (upper_op, lower_op, combine_union) = match op {
        RelOp::Gt => (RelOp::Gt, RelOp::Lt, true),
        RelOp::Geq => (RelOp::Geq, RelOp::Leq, true),
        RelOp::Lt => (RelOp::Lt, RelOp::Gt, false),
        RelOp::Leq => (RelOp::Leq, RelOp::Geq, false),
        _ => return None,
    };
    // `upper` is the `ln(x) {>,<} √t` half (the larger-`x` ray `(e^√t, ∞)` for `>`, the
    // `(0, e^√t)` cap for `<`); `lower` is the `ln(x) {<,>} -√t` half. Both are single
    // `(…)` intervals whose ENDS are `e^{±√t}` — bounds containing the constant `E`,
    // which `union_solution_sets`/`intersect_solution_sets` cannot order (they fold via
    // the rational-only `as_rational_const`, so they would mis-merge `(0,1/e) ∪ (e,∞)`
    // into `(0,∞)`). Combine them DIRECTLY: for `>`/`≥` the two halves are disjoint and
    // already ordered (`e^{-√t} < e^{√t}`); for `<`/`≤` the result is the single band
    // `(e^{-√t}, e^{√t})`.
    let upper = solve_concrete_side(ln_expr, sqrt_t, upper_op, var, simplifier, opts, ctx);
    let lower = solve_concrete_side(ln_expr, neg_sqrt_t, lower_op, var, simplifier, opts, ctx);
    let residual = |simplifier: &mut Simplifier| {
        cas_solver_core::solve_outcome::residual_solution_set(
            &mut simplifier.context,
            eq.lhs,
            eq.rhs,
            eq.op.clone(),
            var,
        )
    };
    let (Some(SolutionSet::Continuous(iv_upper)), Some(SolutionSet::Continuous(iv_lower))) =
        (upper, lower)
    else {
        return Some(residual(simplifier));
    };
    Some(if combine_union {
        // `ln(x) < -√t` -> `(0, e^{-√t})` (small x), then `ln(x) > √t` -> `(e^{√t}, ∞)`.
        SolutionSet::Union(vec![iv_lower, iv_upper])
    } else {
        // `(e^{-√t}, e^{√t})`: low end from the `ln(x) > -√t` half, high end from `< √t`.
        SolutionSet::Continuous(cas_ast::Interval {
            min: iv_lower.min,
            min_type: iv_lower.min_type,
            max: iv_upper.max,
            max_type: iv_upper.max_type,
        })
    })
}

/// `A·sin/cos(g(x)) ⋚ c` where `|c/A| ≥ 1`: the bounded range of sin/cos
/// settles the relation exactly — attained boundaries reduce to the periodic
/// EQUATION (owned by `try_solve_periodic_trig_equation`, multiple-angle
/// capable), unattainable ones to ∅/ℝ. `|c/A| < 1` returns None (honest
/// decline: the answer is a periodic union of intervals the SolutionSet
/// cannot yet represent).
pub(super) fn try_solve_trig_weak_boundary_inequality(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<(SolutionSet, Vec<crate::SolveStep>)> {
    // Same shape-preservation gate as the periodic handler, plus the
    // angle-sum expansion (`sin(x+π/3) → sin·cos + cos·sin`) which would
    // destroy the shifted-argument match before the range check runs.
    let mut added: Vec<&'static str> = Vec::new();
    for rule in MULTIPLE_ANGLE_EXPANSION_RULES
        .iter()
        .copied()
        .chain(std::iter::once("Angle Sum/Diff Identity"))
    {
        if !simplifier.is_rule_disabled(rule) {
            simplifier.disable_rule(rule);
            added.push(rule);
        }
    }
    let out = try_solve_trig_weak_boundary_inequality_ungated(eq, var, simplifier);
    for rule in added {
        simplifier.enable_rule(rule);
    }
    out
}

fn try_solve_trig_weak_boundary_inequality_ungated(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<(SolutionSet, Vec<crate::SolveStep>)> {
    use cas_ast::{BuiltinFn, RelOp};
    use cas_solver_core::isolation_utils::contains_var;
    use num_rational::BigRational;
    use num_traits::{One, Signed};

    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    let (lhs, _) = simplifier.simplify(eq.lhs);
    let (rhs, _) = simplifier.simplify(eq.rhs);

    // Match `A·sin/cos(g)` on one side, a rational constant on the other.
    let bounded_trig = |ctx: &Context, e: ExprId| -> Option<(BigRational, BuiltinFn, ExprId)> {
        let (coeff, core) = peel_rational_coefficient(ctx, e);
        if num_traits::Zero::is_zero(&coeff) {
            return None;
        }
        if let Expr::Function(fn_id, args) = ctx.get(core) {
            if args.len() == 1 && contains_var(ctx, args[0], var) {
                if let Some(f) = ctx.builtin_of(*fn_id) {
                    if matches!(f, BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Tan) {
                        return Some((coeff, f, args[0]));
                    }
                }
            }
        }
        None
    };
    // Peel an outer additive rational constant from a candidate side:
    // `A·trig(g) + d ⋚ c` matches as the core `A·trig(g)` with the constant
    // `d` moved to the other side (`r = (c − d)/A`). Without this the shape
    // `3·cos(x) + 1 ≥ 4` never reached this handler at all (design §5.2).
    let peel_additive =
        |ctx: &Context, e: ExprId| -> Option<(BigRational, ExprId, cas_math::expr_nary::Sign)> {
            let view = cas_math::expr_nary::AddView::from_expr(ctx, e);
            if view.terms.len() < 2 {
                return Some((
                    BigRational::from_integer(0.into()),
                    e,
                    cas_math::expr_nary::Sign::Pos,
                ));
            }
            let mut d = BigRational::from_integer(0.into());
            let mut core: Option<(ExprId, cas_math::expr_nary::Sign)> = None;
            for (term, sign) in view.terms.iter() {
                if let Some(n) = cas_math::numeric_eval::as_rational_const(ctx, *term) {
                    match sign {
                        cas_math::expr_nary::Sign::Pos => d += n,
                        cas_math::expr_nary::Sign::Neg => d -= n,
                    }
                } else if core.is_none() {
                    core = Some((*term, *sign));
                } else {
                    return None; // more than one non-constant term
                }
            }
            let (core_expr, core_sign) = core?;
            Some((d, core_expr, core_sign))
        };
    let match_side = |ctx: &Context,
                      e: ExprId|
     -> Option<(BigRational, cas_ast::BuiltinFn, ExprId, BigRational)> {
        // Try the bare shape first, then the additive-peeled core. A
        // Neg-signed core folds its sign into the coefficient (no interning
        // needed): `1 − 2·cos(x)` peels to d=1, core=2·cos(x), sign=Neg ⇒
        // A = −2.
        if let Some((a, f, g)) = bounded_trig(ctx, e) {
            return Some((a, f, g, BigRational::from_integer(0.into())));
        }
        let (d, core, core_sign) = peel_additive(ctx, e)?;
        if num_traits::Zero::is_zero(&d) {
            return None; // nothing peeled — bare match already failed
        }
        let (a, f, g) = bounded_trig(ctx, core)?;
        let a = match core_sign {
            cas_math::expr_nary::Sign::Pos => a,
            cas_math::expr_nary::Sign::Neg => -a,
        };
        Some((a, f, g, d))
    };
    let (a_coeff, trig_fn, arg, c_expr, d_shift, op) =
        if let Some((a, f, g, d)) = match_side(&simplifier.context, lhs) {
            if contains_var(&simplifier.context, rhs, var) {
                return None;
            }
            (a, f, g, rhs, d, eq.op.clone())
        } else if let Some((a, f, g, d)) = match_side(&simplifier.context, rhs) {
            if contains_var(&simplifier.context, lhs, var) {
                return None;
            }
            (a, f, g, lhs, d, flip_inequality(eq.op.clone()))
        } else {
            return None;
        };
    let c_val = cas_math::numeric_eval::as_rational_const(&simplifier.context, c_expr)? - d_shift;

    // Normalize to `trig(g) ⋚' r` (dividing by A flips the operator when A < 0).
    let r = c_val / &a_coeff;
    let op = if a_coeff.is_negative() {
        flip_inequality(op)
    } else {
        op
    };
    // tan branches BEFORE the |r| ladder (design §5, panel-mandated): its
    // range is ℝ, so no threshold is exterior or weak — the window table
    // applies to EVERY rational r (`tan(x) ≥ 2` must never hit the
    // `r > 1 → Empty` arm below, which encodes sin/cos range semantics).
    if matches!(trig_fn, BuiltinFn::Tan) {
        return try_emit_trig_interior_interval_union(simplifier, trig_fn, arg, &r, &op, var);
    }
    let one = BigRational::one();
    if r.clone().abs() < one {
        // Interior threshold |r| < 1: the exact answer is a periodic union
        // of intervals — emit it via the analytic window table (design §5).
        return try_emit_trig_interior_interval_union(simplifier, trig_fn, arg, &r, &op, var);
    }

    // Boundary/exterior: settled by sin/cos ∈ [−1, 1].
    let boundary_equation = |simplifier: &mut Simplifier, value: i32| -> Option<SolutionSet> {
        let val = simplifier.context.num(value.into());
        let trig_call = simplifier.context.call_builtin(trig_fn, vec![arg]);
        let reduced = Equation {
            lhs: trig_call,
            rhs: val,
            op: RelOp::Eq,
        };
        // Full pipeline (not just the periodic handler): a SYMBOLIC shift
        // (`sin(x+π/3) = 1`) is owned by the shifted-argument handler. The
        // reduced relation is an EQUATION, so this cannot re-enter the
        // weak-boundary handler.
        let (set, _) = crate::solver_entrypoints_solve::solve(&reduced, var, simplifier).ok()?;
        // An unresolved residual would echo a mutated equation as the answer
        // to the ORIGINAL inequality — decline instead (honest residual keeps
        // the true operator). A FINITE Discrete set is declined too: a trig
        // boundary equation over an argument containing the variable always
        // has an infinite (periodic) solution family when it has any, so a
        // finite set means the equation path dropped periodicity (the
        // irrational-coefficient family: sin(π·x) = 1 → { 1/2 } loses
        // { 1/2 + 2k } — final-audit finding) and must not be asserted as
        // the complete answer to the inequality.
        if matches!(set, SolutionSet::Residual(_) | SolutionSet::Discrete(_)) {
            return None;
        }
        Some(set)
    };
    match op {
        RelOp::Geq => {
            if r == one {
                boundary_equation(simplifier, 1).map(|set| (set, Vec::new())) // t ≥ 1 ⇔ t = 1
            } else if r > one {
                Some((SolutionSet::Empty, Vec::new()))
            } else {
                Some((SolutionSet::AllReals, Vec::new())) // r ≤ −1: always true
            }
        }
        RelOp::Gt => {
            if r >= one {
                Some((SolutionSet::Empty, Vec::new())) // t > 1 (or more) is unattainable
            } else if r < -&one {
                Some((SolutionSet::AllReals, Vec::new()))
            } else {
                // r = −1: the complement ℝ ∖ {touch points} — the table with
                // r = −1 yields exactly the punctured line (len == period,
                // both ends open): sin u > −1 → (−π/2, 3π/2).
                try_emit_trig_interior_interval_union(simplifier, trig_fn, arg, &r, &op, var)
            }
        }
        RelOp::Leq => {
            if r == -&one {
                boundary_equation(simplifier, -1).map(|set| (set, Vec::new())) // t ≤ −1 ⇔ t = −1
            } else if r < -&one {
                Some((SolutionSet::Empty, Vec::new()))
            } else {
                Some((SolutionSet::AllReals, Vec::new())) // r ≥ 1: always true
            }
        }
        RelOp::Lt => {
            if r <= -&one {
                Some((SolutionSet::Empty, Vec::new()))
            } else if r > one {
                Some((SolutionSet::AllReals, Vec::new()))
            } else {
                // r = 1: complement — punctured line via the table
                // (sin u < 1 → (π/2, 5π/2); cos u < 1 → (0, 2π)).
                try_emit_trig_interior_interval_union(simplifier, trig_fn, arg, &r, &op, var)
            }
        }
        _ => None,
    }
}

/// P2 producer (design §5): `trig(g) ⋚ r` with `|r| < 1` and `g = a·x + b`
/// affine → the exact `PeriodicIntervalUnion` via the analytic window table
/// in u-space mapped back through the inverse affine transform.
///
/// Every guard failure returns `None` (the orientation-blind decline chain
/// then emits the honest operator-preserving residual).
fn try_emit_trig_interior_interval_union(
    simplifier: &mut Simplifier,
    trig_fn: cas_ast::BuiltinFn,
    arg: ExprId,
    r: &num_rational::BigRational,
    op: &cas_ast::RelOp,
    var: &str,
) -> Option<(SolutionSet, Vec<crate::SolveStep>)> {
    use cas_ast::{BoundType, BuiltinFn, Interval, RelOp};

    // Affine gate (design §5.1): `bounded_trig` only checked contains_var, so
    // non-affine args (`sin(x²)`, `sin(sin(x))`) reach this slot and MUST
    // decline — the window table is only valid for monotone affine u.
    let (a, b_intercept) = affine_coefficients(simplifier, arg, var)?;

    // Exact inverse-trig endpoint; the simplifier folds table angles
    // (`arcsin(1/2) → π/6`) and leaves `arcsin(1/3)` symbolic — both fine.
    let r_expr = rational_to_expr(&mut simplifier.context, r);
    let inv_name = match trig_fn {
        BuiltinFn::Sin => "arcsin",
        BuiltinFn::Cos => "arccos",
        BuiltinFn::Tan => "arctan",
        _ => return None,
    };
    let inv_call = simplifier.context.call(inv_name, vec![r_expr]);
    let inv = simplifier.simplify(inv_call).0;

    let pi = simplifier
        .context
        .add(Expr::Constant(cas_ast::Constant::Pi));
    let two = simplifier.context.num(2);
    let two_pi_raw = simplifier.context.add(Expr::Mul(two, pi));
    let two_pi = simplifier.simplify(two_pi_raw).0;

    let simp_add = |simplifier: &mut Simplifier, x: ExprId, y: ExprId| -> ExprId {
        let e = simplifier.context.add(Expr::Add(x, y));
        simplifier.simplify(e).0
    };
    let simp_sub = |simplifier: &mut Simplifier, x: ExprId, y: ExprId| -> ExprId {
        let e = simplifier.context.add(Expr::Sub(x, y));
        simplifier.simplify(e).0
    };
    let simp_neg = |simplifier: &mut Simplifier, x: ExprId| -> ExprId {
        let e = simplifier.context.add(Expr::Neg(x));
        simplifier.simplify(e).0
    };

    // Analytic u-window (design §5 table). Closedness is PER ENDPOINT:
    // strict ops open both ends; non-strict close both for sin/cos (their
    // windows never touch an asymptote) but tan's asymptote end stays Open
    // ALWAYS (`tan u ≥ r` → [arctan r, π/2)).
    let closed = matches!(op, RelOp::Geq | RelOp::Leq);
    let bt = if closed {
        BoundType::Closed
    } else {
        BoundType::Open
    };
    let (u_lo, u_lo_type, u_hi, u_hi_type) = match (trig_fn, op) {
        // sin u > r on (arcsin r, π − arcsin r)
        (BuiltinFn::Sin, RelOp::Gt | RelOp::Geq) => {
            let hi = simp_sub(simplifier, pi, inv);
            (inv, bt.clone(), hi, bt)
        }
        // sin u < r on (π − arcsin r, 2π + arcsin r)
        (BuiltinFn::Sin, RelOp::Lt | RelOp::Leq) => {
            let lo = simp_sub(simplifier, pi, inv);
            let hi = simp_add(simplifier, two_pi, inv);
            (lo, bt.clone(), hi, bt)
        }
        // cos u > r on (−arccos r, arccos r)
        (BuiltinFn::Cos, RelOp::Gt | RelOp::Geq) => {
            let lo = simp_neg(simplifier, inv);
            (lo, bt.clone(), inv, bt)
        }
        // cos u < r on (arccos r, 2π − arccos r)
        (BuiltinFn::Cos, RelOp::Lt | RelOp::Leq) => {
            let hi = simp_sub(simplifier, two_pi, inv);
            (inv, bt.clone(), hi, bt)
        }
        // tan u > r on (arctan r, π/2) — the asymptote end is Open ALWAYS.
        (BuiltinFn::Tan, RelOp::Gt | RelOp::Geq) => {
            let two_e = simplifier.context.num(2);
            let half_pi_raw = simplifier.context.add(Expr::Div(pi, two_e));
            let half_pi = simplifier.simplify(half_pi_raw).0;
            (inv, bt, half_pi, BoundType::Open)
        }
        // tan u < r on (−π/2, arctan r)
        (BuiltinFn::Tan, RelOp::Lt | RelOp::Leq) => {
            let two_e = simplifier.context.num(2);
            let half_pi_raw = simplifier.context.add(Expr::Div(pi, two_e));
            let half_pi = simplifier.simplify(half_pi_raw).0;
            let neg_half = simp_neg(simplifier, half_pi);
            (neg_half, BoundType::Open, inv, bt)
        }
        _ => return None,
    };

    // Inverse affine map x = (u − b)/a: endpoints move as PAIRS
    // (value, BoundType) — swap for a < 0, the BoundType travels WITH its
    // endpoint (design §5, panel-corrected; normative precedent
    // `map_set_through_inverse_affine`). Period T_x = T_u / |a|.
    let a_expr = rational_to_expr(&mut simplifier.context, &a);
    let map_endpoint = |simplifier: &mut Simplifier, u: ExprId| -> ExprId {
        let shifted = simplifier.context.add(Expr::Sub(u, b_intercept));
        let scaled = simplifier.context.add(Expr::Div(shifted, a_expr));
        simplifier.simplify(scaled).0
    };
    let x_lo_raw = map_endpoint(simplifier, u_lo);
    let x_hi_raw = map_endpoint(simplifier, u_hi);
    let (x_lo, x_lo_type, x_hi, x_hi_type) = if num_traits::Signed::is_negative(&a) {
        // Endpoints swap as (value, BoundType) PAIRS under a decreasing map.
        (x_hi_raw, u_hi_type, x_lo_raw, u_lo_type)
    } else {
        (x_lo_raw, u_lo_type, x_hi_raw, u_hi_type)
    };
    let abs_a = num_traits::Signed::abs(&a);
    let abs_a_expr = rational_to_expr(&mut simplifier.context, &abs_a);
    let period_u = if matches!(trig_fn, BuiltinFn::Tan) {
        pi
    } else {
        two_pi
    };
    let period_raw = simplifier.context.add(Expr::Div(period_u, abs_a_expr));
    let period = simplifier.simplify(period_raw).0;

    let window = Interval {
        min: x_lo,
        min_type: x_lo_type,
        max: x_hi,
        max_type: x_hi_type,
    };

    // Numeric emission airbag (design §5, panel-amended semantics): sample
    // the ORIGINAL relation at window-relative u fractions mapped to x —
    // never at endpoints (f64 ulp), never deciding on inconclusive samples.
    // It can only DEGRADE to a decline, never widen the set.
    if !interior_window_samples_consistent(simplifier, trig_fn, arg, r, op, u_lo, u_hi, &a, var) {
        return None;
    }

    // The three facts the student needs are already computed here: the
    // one-period inversion, the base window where the relation holds, and the
    // translation by the period. They were being thrown away — the inequality
    // returned a correct interval union and narrated NOTHING, while its `=`
    // twin narrated fine (audit: 12 of 16 inequality rows mute vs 2 of 19
    // equation rows).
    let mut steps = Vec::new();
    let x_var = simplifier.context.var(var);
    steps.push(crate::SolveStep::new(
        "Invert the trig function on one period".to_string(),
        Equation {
            lhs: simplifier.context.var("u"),
            rhs: inv,
            op: cas_ast::RelOp::Eq,
        },
        crate::ImportanceLevel::Medium,
    ));
    steps.push(crate::SolveStep::new(
        "Base window where the relation holds".to_string(),
        Equation {
            lhs: x_var,
            rhs: window.min,
            op: match window.min_type {
                BoundType::Closed => cas_ast::RelOp::Geq,
                _ => cas_ast::RelOp::Gt,
            },
        },
        crate::ImportanceLevel::Medium,
    ));
    steps.push(crate::SolveStep::new(
        "Translate the window by the period (k any integer)".to_string(),
        Equation {
            lhs: x_var,
            rhs: {
                let k_var = simplifier.context.var("k");
                let k_t = simplifier.context.add(Expr::Mul(k_var, period));
                simplifier.context.add(Expr::Add(window.min, k_t))
            },
            op: cas_ast::RelOp::Eq,
        },
        crate::ImportanceLevel::Medium,
    ));

    Some((
        SolutionSet::PeriodicIntervalUnion {
            windows: vec![window],
            period,
        },
        steps,
    ))
}

/// PIU P3b: `A / trig(g) ⋚ c` (Div or `trig^(−1)` shapes, either side).
/// Normalizes to `1/s ⋚ r` (s = trig(g), r = c/A, flip for A < 0), splits by
/// the sign of `r` into window relations on `s`, sub-solves each through the
/// full pipeline (the P2/P3a producers), and combines with the circular
/// same-period algebra. Any sub-result outside {∅, ℝ, PIU} declines.
/// `A·trig(g)² ⋚ c` and `A·|trig(g)| ⋚ c` (sin/cos/tan): reduce the even
/// power / absolute value to a sign case split on `trig(g)` and combine the
/// windows with the circular same-period algebra.
///   `sin(x)² < 1/4` ⟺ `|sin(x)| < 1/2` ⟺ `sin(x) > −1/2 ∩ sin(x) < 1/2`
///   `cos(x)² > 1/2` ⟺ `|cos(x)| > √2/2` ⟺ `cos > √2/2 ∪ cos < −√2/2`
/// Point-set outcomes (`sin(x)² ≥ 1` ⟺ `sin(x) = ±1`) fall out as an honest
/// residual: the sub-solves return `Periodic`, which the window combiner
/// declines rather than mis-handle.
pub(super) fn try_solve_even_power_or_abs_trig_inequality(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<SolutionSet> {
    use cas_ast::{BuiltinFn, RelOp};
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;
    use num_rational::BigRational;
    use num_traits::{Signed, Zero};

    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    // Detect on the RAW tree: `simplify` rewrites `tan(x)²` into
    // `sin(x)²/cos(x)²`, which would hide the `Pow(tan, 2)` shape. The
    // constant side is read exactly either way.
    let lhs = eq.lhs;
    let rhs = eq.rhs;

    let trig_of = |ctx_: &Context, e: ExprId| -> Option<(BuiltinFn, ExprId)> {
        if let Expr::Function(fn_id, args) = ctx_.get(e) {
            if args.len() == 1 && contains_var(ctx_, args[0], var) {
                if let Some(f) = ctx_.builtin_of(*fn_id) {
                    // Tanh/Cosh/Sinh (F4 hyperbolic member): total domain,
                    // no poles — the square/abs reduction and its edge arms
                    // apply verbatim; their sub-solves settle through
                    // `try_solve_hyperbolic_range_edge_inequality` (range
                    // edges exactly, interior thresholds by ar*-inversion).
                    if matches!(
                        f,
                        BuiltinFn::Sin
                            | BuiltinFn::Cos
                            | BuiltinFn::Tan
                            | BuiltinFn::Tanh
                            | BuiltinFn::Cosh
                            | BuiltinFn::Sinh
                    ) {
                        return Some((f, args[0]));
                    }
                }
            }
        }
        None
    };
    // Match `A·trig(g)²` (is_square = true) or `A·|trig(g)|` (false).
    let detect = |ctx_: &Context, e: ExprId| -> Option<(BigRational, BuiltinFn, ExprId, bool)> {
        let (coeff, core) = peel_rational_coefficient(ctx_, e);
        if coeff.is_zero() {
            return None;
        }
        if let Expr::Pow(base, exp) = ctx_.get(core) {
            if as_rational_const(ctx_, *exp) == Some(BigRational::from_integer(2.into())) {
                if let Some((f, g)) = trig_of(ctx_, *base) {
                    return Some((coeff, f, g, true));
                }
            }
        }
        if let Some(inner) = match_abs_argument(ctx_, core) {
            if let Some((f, g)) = trig_of(ctx_, inner) {
                return Some((coeff, f, g, false));
            }
        }
        None
    };
    let (a_coeff, trig_fn, g, is_square, c_expr, op) =
        if let Some((a, f, gg, sq)) = detect(&simplifier.context, lhs) {
            if contains_var(&simplifier.context, rhs, var) {
                return None;
            }
            (a, f, gg, sq, rhs, eq.op.clone())
        } else if let Some((a, f, gg, sq)) = detect(&simplifier.context, rhs) {
            if contains_var(&simplifier.context, lhs, var) {
                return None;
            }
            (a, f, gg, sq, lhs, flip_inequality(eq.op.clone()))
        } else {
            // Not a direct square/abs: try the reciprocal shapes (`sec²`,
            // `csc²`, `A/trig²`, `|sec|`, `A/|trig|`) before giving up.
            return try_solve_reciprocal_square_or_abs_trig_inequality(eq, var, simplifier);
        };
    let c_val = as_rational_const(&simplifier.context, c_expr)?;

    // Divide by A: `trig² ⋚ t` (square) or `|trig| ⋚ t` (abs), flipping for A<0.
    let t = c_val / &a_coeff;
    let op = if a_coeff.is_negative() {
        flip_inequality(op)
    } else {
        op
    };
    solve_trig_square_or_abs_rel(simplifier, trig_fn, g, is_square, op, t, var)
}

/// F4 (frontier-audit 2026-07-14): reciprocal-square / reciprocal-abs trig
/// inequalities — `sec(g)² ⋚ c`, `csc(g)² ⋚ c`, `A/trig(g)² ⋚ c`,
/// `trig(g)^(−2) ⋚ c`, `|sec(g)| ⋚ c`, `A/|trig(g)| ⋚ c`, sin/cos bases only
/// (the tan/cot pair keeps declining until cot gets its own window table,
/// mirroring the power-1 reciprocal handler). Inverts `A/T ⋚ c` to relations
/// on `T = trig²` (or `|trig|`) by the sign case split of the power-1
/// handler; the definedness conjunct `T > 0` IS the pole puncture, solved
/// through the existing punctured-line reduction (`trig > 0 ∪ trig < 0`), so
/// `sec(x)² > 2` excludes the poles `π/2 + kπ` exactly.
fn try_solve_reciprocal_square_or_abs_trig_inequality(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<SolutionSet> {
    use cas_ast::{BuiltinFn, RelOp};
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;
    use num_rational::BigRational;
    use num_traits::{One, Signed, Zero};

    let lhs = eq.lhs;
    let rhs = eq.rhs;

    // `(sin|cos, arg)` of a direct sin/cos call (a reciprocal's DENOMINATOR)…
    let sin_cos_of = |ctx_: &Context, e2: ExprId| -> Option<(BuiltinFn, ExprId)> {
        if let Expr::Function(fn_id, args) = ctx_.get(e2) {
            if args.len() == 1 && contains_var(ctx_, args[0], var) {
                if let Some(f) = ctx_.builtin_of(*fn_id) {
                    if matches!(f, BuiltinFn::Sin | BuiltinFn::Cos) {
                        return Some((f, args[0]));
                    }
                }
            }
        }
        None
    };
    // …or of the refolded reciprocal call: `sec = 1/cos`, `csc = 1/sin`.
    let recip_base_of = |ctx_: &Context, e2: ExprId| -> Option<(BuiltinFn, ExprId)> {
        if let Expr::Function(fn_id, args) = ctx_.get(e2) {
            if args.len() == 1 && contains_var(ctx_, args[0], var) {
                match ctx_.builtin_of(*fn_id) {
                    Some(BuiltinFn::Sec) => return Some((BuiltinFn::Cos, args[0])),
                    Some(BuiltinFn::Csc) => return Some((BuiltinFn::Sin, args[0])),
                    _ => {}
                }
            }
        }
        None
    };
    // Match `A · (1/T)` with T = trig(g)² (is_square = true) or |trig(g)|.
    let detect_recip =
        |ctx_: &Context, e: ExprId| -> Option<(BigRational, BuiltinFn, ExprId, bool)> {
            let (coeff, core) = peel_rational_coefficient(ctx_, e);
            if coeff.is_zero() {
                return None;
            }
            let two = BigRational::from_integer(2.into());
            match ctx_.get(core) {
                Expr::Pow(base, exp) => {
                    let (base, exp) = (*base, *exp);
                    let ev = as_rational_const(ctx_, exp)?;
                    if ev == two {
                        // sec(g)² = 1/cos(g)², csc(g)² = 1/sin(g)².
                        let (f, g) = recip_base_of(ctx_, base)?;
                        Some((coeff, f, g, true))
                    } else if ev == -two {
                        // sin(g)^(−2) = 1/sin(g)².
                        let (f, g) = sin_cos_of(ctx_, base)?;
                        Some((coeff, f, g, true))
                    } else {
                        None
                    }
                }
                Expr::Div(num, den) => {
                    let (num, den) = (*num, *den);
                    let a = as_rational_const(ctx_, num)?;
                    if a.is_zero() {
                        return None;
                    }
                    let (den_coeff, den_core) = peel_rational_coefficient(ctx_, den);
                    if den_coeff.is_zero() {
                        return None;
                    }
                    let scale = coeff * a / den_coeff;
                    if let Expr::Pow(base, exp) = ctx_.get(den_core) {
                        let (base, exp) = (*base, *exp);
                        if as_rational_const(ctx_, exp)? == two {
                            let (f, g) = sin_cos_of(ctx_, base)?;
                            return Some((scale, f, g, true));
                        }
                        return None;
                    }
                    let inner = match_abs_argument(ctx_, den_core)?;
                    let (f, g) = sin_cos_of(ctx_, inner)?;
                    Some((scale, f, g, false))
                }
                _ => {
                    // |sec(g)| = 1/|cos(g)|, |csc(g)| = 1/|sin(g)|.
                    let inner = match_abs_argument(ctx_, core)?;
                    let (f, g) = recip_base_of(ctx_, inner)?;
                    Some((coeff, f, g, false))
                }
            }
        };

    let (a_coeff, trig_fn, g, is_square, c_expr, op) =
        if let Some((a, f, gg, sq)) = detect_recip(&simplifier.context, lhs) {
            if contains_var(&simplifier.context, rhs, var) {
                return None;
            }
            (a, f, gg, sq, rhs, eq.op.clone())
        } else if let Some((a, f, gg, sq)) = detect_recip(&simplifier.context, rhs) {
            if contains_var(&simplifier.context, lhs, var) {
                return None;
            }
            (a, f, gg, sq, lhs, flip_inequality(eq.op.clone()))
        } else {
            return None;
        };
    let c_val = as_rational_const(&simplifier.context, c_expr)?;

    // A/T ⋚ c ⟺ 1/T ⋚' r with r = c/A (dividing by A flips for A < 0).
    let r = c_val / &a_coeff;
    let op = if a_coeff.is_negative() {
        flip_inequality(op)
    } else {
        op
    };

    // Case split on `1/T ⋚ r`: T ≥ 0 always, and 1/T is defined ⟺ T > 0, so
    // 1/T is positive everywhere it exists.
    //   r > 0:  `>` ⟺ 0 < T < 1/r    `≥` ⟺ 0 < T ≤ 1/r
    //           `<` ⟺ T > 1/r        `≤` ⟺ T ≥ 1/r
    //   r ≤ 0:  `>`/`≥` ⟺ T > 0 (every defined point)   `<`/`≤` ⟺ ∅
    let zero = BigRational::zero();
    let parts: Vec<(RelOp, BigRational)> = if r.is_positive() {
        let inv_r = BigRational::one() / &r;
        match op {
            RelOp::Gt => vec![(RelOp::Gt, zero), (RelOp::Lt, inv_r)],
            RelOp::Geq => vec![(RelOp::Gt, zero), (RelOp::Leq, inv_r)],
            RelOp::Lt => vec![(RelOp::Gt, inv_r)],
            RelOp::Leq => vec![(RelOp::Geq, inv_r)],
            _ => return None,
        }
    } else {
        match op {
            RelOp::Gt | RelOp::Geq => vec![(RelOp::Gt, zero)],
            RelOp::Lt | RelOp::Leq => return Some(SolutionSet::Empty),
            _ => return None,
        }
    };

    // All multi-part splits above are conjunctions (0 < T ∧ T < 1/r).
    let mut acc: Option<SolutionSet> = None;
    for (sub_op, t) in parts {
        let set = solve_trig_square_or_abs_rel(simplifier, trig_fn, g, is_square, sub_op, t, var)?;
        acc = Some(match acc {
            None => set,
            Some(prev) => combine_piu_sets(simplifier, prev, set, true)?,
        });
    }
    acc
}

/// Hyperbolic RANGE-edge inequalities (F4 hyperbolic member, frontier-audit
/// 2026-07-14): `tanh(g) ⋚ c` for |c| ≥ 1 and `cosh(g) ⋚ c` for c ≤ 1 are
/// decided EXACTLY by the function's range — `tanh: ℝ → (−1, 1)` (strict:
/// the bounds are never attained) and `cosh: ℝ → [1, ∞)` (1 attained exactly
/// where g = 0) — with no inversion machinery. The argument must be a
/// NON-CONSTANT POLYNOMIAL in `var` so its domain is all of ℝ: for
/// `tanh(ln(x)) < 1` the true set is `(0, ∞)`, not ℝ, and the guard must
/// decline. Thresholds strictly inside the range (|c| < 1 for tanh, c > 1
/// for cosh) still decline honestly — the ar-function inversion is a named
/// follow-up, and `sinh` (full range) has no edge at all.
pub(super) fn try_solve_hyperbolic_range_edge_inequality(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<SolutionSet> {
    use cas_ast::{BuiltinFn, RelOp};
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;
    use num_rational::BigRational;

    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    let hyper_of = |ctx_: &Context, e: ExprId| -> Option<(BuiltinFn, ExprId)> {
        if let Expr::Function(fn_id, args) = ctx_.get(e) {
            if args.len() == 1 && contains_var(ctx_, args[0], var) {
                if let Some(f) = ctx_.builtin_of(*fn_id) {
                    if matches!(f, BuiltinFn::Tanh | BuiltinFn::Cosh | BuiltinFn::Sinh) {
                        return Some((f, args[0]));
                    }
                }
            }
        }
        None
    };
    // Also match the SQUARE `hyper(g)²` and ABS `|hyper(g)|` spellings: their
    // even-power reduction produces two rays with SYMBOLIC ar*-endpoints that
    // the set algebra cannot order (the F7 trap), so they reduce HERE — the
    // inverse of an odd increasing function keeps `−ar(r) < ar(r)` true by
    // the math, never by a comparator.
    let hyper_sq_abs = |ctx_: &Context, e: ExprId| -> Option<(BuiltinFn, ExprId, bool)> {
        if let Expr::Pow(base, exp) = ctx_.get(e) {
            if as_rational_const(ctx_, *exp) == Some(BigRational::from_integer(2.into())) {
                let (f, g) = hyper_of(ctx_, *base)?;
                return Some((f, g, true));
            }
            return None;
        }
        let inner = match_abs_argument(ctx_, e)?;
        let (f, g) = hyper_of(ctx_, inner)?;
        Some((f, g, false))
    };
    let mut square_mode: Option<bool> = None; // Some(is_square) when sq/abs form
    let (hyper_fn, g, c_expr, op) = if let Some((f, g)) = hyper_of(&simplifier.context, eq.lhs) {
        if contains_var(&simplifier.context, eq.rhs, var) {
            return None;
        }
        (f, g, eq.rhs, eq.op.clone())
    } else if let Some((f, g)) = hyper_of(&simplifier.context, eq.rhs) {
        if contains_var(&simplifier.context, eq.lhs, var) {
            return None;
        }
        (f, g, eq.lhs, flip_inequality(eq.op.clone()))
    } else if let Some((f, g, sq)) = hyper_sq_abs(&simplifier.context, eq.lhs) {
        if contains_var(&simplifier.context, eq.rhs, var) {
            return None;
        }
        square_mode = Some(sq);
        (f, g, eq.rhs, eq.op.clone())
    } else if let Some((f, g, sq)) = hyper_sq_abs(&simplifier.context, eq.rhs) {
        if contains_var(&simplifier.context, eq.lhs, var) {
            return None;
        }
        square_mode = Some(sq);
        (f, g, eq.lhs, flip_inequality(eq.op.clone()))
    } else {
        return None;
    };
    // The even-power split hands its branches UNSIMPLIFIED bounds
    // (`tanh(x)² < 1` → branch `tanh(x) < sqrt(1)`): fold before reading.
    let (c_expr, _) = simplifier.simplify(c_expr);
    let c = as_rational_const(&simplifier.context, c_expr)?;

    // SQUARE/ABS reduction to the power-1 relation `hyper(g) {op} …` or the
    // even-function band: `T = hyper²` (or `|hyper|`) is compared against the
    // rational `t`, and every case lands on machinery below with a KNOWN
    // endpoint order.
    if let Some(is_square) = square_mode {
        return solve_hyperbolic_square_or_abs_rel(simplifier, hyper_fn, g, is_square, op, c, var);
    }
    // Total-domain gate: a polynomial argument is defined on all of ℝ.
    let g_poly = cas_math::polynomial::Polynomial::from_expr(&simplifier.context, g, var).ok()?;
    if g_poly.degree() < 1 {
        return None;
    }
    let one = BigRational::from_integer(1.into());
    match hyper_fn {
        BuiltinFn::Tanh => {
            // tanh(g) ∈ (−1, 1) for every real g, both bounds strict.
            if c >= one {
                match op {
                    RelOp::Lt | RelOp::Leq => Some(SolutionSet::AllReals),
                    RelOp::Gt | RelOp::Geq => Some(SolutionSet::Empty),
                    _ => None,
                }
            } else if c <= -one {
                match op {
                    RelOp::Gt | RelOp::Geq => Some(SolutionSet::AllReals),
                    RelOp::Lt | RelOp::Leq => Some(SolutionSet::Empty),
                    _ => None,
                }
            } else {
                // INTERIOR threshold |c| < 1: tanh is strictly increasing on
                // its total domain, so `tanh(g) {op} c ⟺ g {op} atanh(c)` —
                // a single monotone relation the linear/poly solver isolates
                // without any symbolic-endpoint set algebra.
                let inv = simplifier
                    .context
                    .call_builtin(BuiltinFn::Atanh, vec![c_expr]);
                let (inv, _) = simplifier.simplify(inv);
                solve_relation_set(simplifier, var, g, inv, op)
            }
        }
        BuiltinFn::Sinh => {
            // sinh: strictly increasing bijection ℝ → ℝ — invert for ANY c.
            let inv = simplifier
                .context
                .call_builtin(BuiltinFn::Asinh, vec![c_expr]);
            let (inv, _) = simplifier.simplify(inv);
            solve_relation_set(simplifier, var, g, inv, op)
        }
        BuiltinFn::Cosh => {
            // cosh(g) ∈ [1, ∞), with cosh(g) = 1 ⟺ g = 0.
            if c < one {
                match op {
                    RelOp::Gt | RelOp::Geq => Some(SolutionSet::AllReals),
                    RelOp::Lt | RelOp::Leq => Some(SolutionSet::Empty),
                    _ => None,
                }
            } else if c == one {
                match op {
                    RelOp::Geq => Some(SolutionSet::AllReals),
                    RelOp::Lt => Some(SolutionSet::Empty),
                    // cosh(g) > 1 ⟺ g ≠ 0; cosh(g) ≤ 1 ⟺ g = 0: both need
                    // g's zero set — delegate to the full solver on the
                    // POLYNOMIAL relation (total domain, no recursion into
                    // this handler: the sub-relations carry no hyperbolic).
                    RelOp::Gt => {
                        let zero = simplifier.context.num(0);
                        let lo = solve_relation_set(simplifier, var, g, zero, RelOp::Lt)?;
                        let hi = solve_relation_set(simplifier, var, g, zero, RelOp::Gt)?;
                        Some(cas_solver_core::solution_set::union_solution_sets(
                            &simplifier.context,
                            lo,
                            hi,
                        ))
                    }
                    RelOp::Leq => {
                        let zero = simplifier.context.num(0);
                        solve_relation_set(simplifier, var, g, zero, RelOp::Eq)
                    }
                    _ => None,
                }
            } else {
                // c > 1: cosh is EVEN with minimum 1 at g = 0, so
                // `cosh(g) {op} c ⟺ |g| {op} acosh(c)` — a band or its
                // complement with SYMBOLIC endpoints ±acosh(c). The core set
                // algebra cannot order symbolic endpoints (the F7 trap), so
                // the band is built DIRECTLY for an AFFINE g = k·x + b,
                // oriented by the RATIONAL slope's sign; higher degrees
                // decline honestly.
                let a_pos = simplifier
                    .context
                    .call_builtin(BuiltinFn::Acosh, vec![c_expr]);
                let (a_pos, _) = simplifier.simplify(a_pos);
                build_affine_symmetric_band_or_complement(simplifier, &g_poly, a_pos, op)
            }
        }
        _ => None,
    }
}

pub(super) fn try_solve_reciprocal_trig_inequality(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<SolutionSet> {
    use cas_ast::{BuiltinFn, RelOp};
    use cas_solver_core::isolation_utils::contains_var;
    use num_rational::BigRational;
    use num_traits::{Signed, Zero};

    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    let (lhs, _) = simplifier.simplify(eq.lhs);
    let (rhs, _) = simplifier.simplify(eq.rhs);

    // Match `A · h(g)^(−1)` or `A / h(g)` for h ∈ {sin, cos, tan} (circular)
    // and {sinh, cosh, tanh} (hyperbolic — the parser desugars sech/csch/coth
    // to these reciprocals, so `sech(x) < 1/2` arrives as `1/cosh(x) < 1/2`).
    // The sign case split below is agnostic of WHICH function `s` is; the
    // per-window relations delegate to the full solve pipeline, which owns
    // both families (`cosh > 2` → symmetric rays, `tanh < 1/2` → ray). Note
    // the cot caveat below does NOT apply to coth: tanh has no poles, so
    // `coth = 1/tanh` is exact as functions — unlike `1/tan`, which is
    // undefined at tan's poles where cot itself is 0.
    let recip_trig = |ctx_: &Context, e: ExprId| -> Option<(BigRational, BuiltinFn, ExprId)> {
        let (coeff, core) = peel_rational_coefficient(ctx_, e);
        if coeff.is_zero() {
            return None;
        }
        let trig_of = |ctx_: &Context, e2: ExprId| -> Option<(BuiltinFn, ExprId)> {
            if let Expr::Function(fn_id, args) = ctx_.get(e2) {
                if args.len() == 1 && contains_var(ctx_, args[0], var) {
                    if let Some(f) = ctx_.builtin_of(*fn_id) {
                        if matches!(
                            f,
                            BuiltinFn::Sin
                                | BuiltinFn::Cos
                                | BuiltinFn::Tan
                                | BuiltinFn::Sinh
                                | BuiltinFn::Cosh
                                | BuiltinFn::Tanh
                        ) {
                            return Some((f, args[0]));
                        }
                    }
                }
            }
            None
        };
        if let Expr::Pow(base, exp) = ctx_.get(core) {
            let minus_one = BigRational::from_integer((-1).into());
            if cas_math::numeric_eval::as_rational_const(ctx_, *exp) == Some(minus_one) {
                if let Some((f, g)) = trig_of(ctx_, *base) {
                    return Some((coeff, f, g));
                }
            }
        }
        if let Expr::Div(num, den) = ctx_.get(core) {
            let a = cas_math::numeric_eval::as_rational_const(ctx_, *num)?;
            if a.is_zero() {
                return None;
            }
            if let Some((f, g)) = trig_of(ctx_, *den) {
                return Some((coeff * a, f, g));
            }
        }
        // The simplifier refolds UNIT-numerator reciprocals into the named
        // functions (`1/sin → csc`, `1/cos → sec`, `1/tan → cot`); those ARE
        // the reciprocal shape.
        if let Expr::Function(fn_id, args) = ctx_.get(core) {
            if args.len() == 1 && contains_var(ctx_, args[0], var) {
                if let Some(f) = ctx_.builtin_of(*fn_id) {
                    // NOT Cot: `cot(g)` is DEFINED at tan's poles
                    // (cot(π/2) = 0), so the `1/tan` reduction silently
                    // loses exactly those points from any set that should
                    // contain cot = 0 (final-audit finding: cot(x) >= 0 came
                    // back open at π/2+kπ). cot needs its own window table.
                    let base = match f {
                        BuiltinFn::Csc => Some(BuiltinFn::Sin),
                        BuiltinFn::Sec => Some(BuiltinFn::Cos),
                        _ => None,
                    };
                    if let Some(bf) = base {
                        return Some((coeff, bf, args[0]));
                    }
                }
            }
        }
        None
    };
    let (a_coeff, trig_fn, g, c_expr, op) =
        if let Some((a, f, g)) = recip_trig(&simplifier.context, lhs) {
            if contains_var(&simplifier.context, rhs, var) {
                return None;
            }
            (a, f, g, rhs, eq.op.clone())
        } else if let Some((a, f, g)) = recip_trig(&simplifier.context, rhs) {
            if contains_var(&simplifier.context, lhs, var) {
                return None;
            }
            (a, f, g, lhs, flip_inequality(eq.op.clone()))
        } else {
            return None;
        };
    let c_val = cas_math::numeric_eval::as_rational_const(&simplifier.context, c_expr)?;

    // A/s ⋚ c ⟺ 1/s ⋚' r with r = c/A (dividing by A flips for A < 0).
    let r = c_val / &a_coeff;
    let op = if a_coeff.is_negative() {
        flip_inequality(op)
    } else {
        op
    };

    // Sign case split for `1/s ⋚ r` (s ≠ 0 wherever 1/s is defined; the
    // strict `s > 0` / `s < 0` windows exclude the pole by construction).
    let zero = BigRational::zero();
    let inv_r = if r.is_zero() {
        zero.clone()
    } else {
        BigRational::from_integer(1.into()) / &r
    };
    // (conjunction?, parts): conjunction=true → intersect, false → union.
    let (conj, parts): (bool, Vec<(RelOp, BigRational)>) = if r.is_zero() {
        match op {
            RelOp::Gt | RelOp::Geq => (true, vec![(RelOp::Gt, zero)]),
            RelOp::Lt | RelOp::Leq => (true, vec![(RelOp::Lt, zero)]),
            _ => return None,
        }
    } else if r.is_positive() {
        match op {
            RelOp::Gt => (true, vec![(RelOp::Gt, zero), (RelOp::Lt, inv_r)]),
            RelOp::Geq => (true, vec![(RelOp::Gt, zero), (RelOp::Leq, inv_r)]),
            RelOp::Lt => (false, vec![(RelOp::Lt, zero), (RelOp::Gt, inv_r)]),
            RelOp::Leq => (false, vec![(RelOp::Lt, zero), (RelOp::Geq, inv_r)]),
            _ => return None,
        }
    } else {
        match op {
            RelOp::Lt => (true, vec![(RelOp::Gt, inv_r), (RelOp::Lt, zero)]),
            RelOp::Leq => (true, vec![(RelOp::Geq, inv_r), (RelOp::Lt, zero)]),
            RelOp::Gt => (false, vec![(RelOp::Gt, zero), (RelOp::Lt, inv_r)]),
            RelOp::Geq => (false, vec![(RelOp::Gt, zero), (RelOp::Leq, inv_r)]),
            _ => return None,
        }
    };

    // Sub-solve each `trig(g) ⋚ bound` through the full pipeline and combine.
    let mut acc: Option<SolutionSet> = None;
    for (sub_op, bound) in parts {
        let bound_expr = rational_to_expr(&mut simplifier.context, &bound);
        let trig_call = simplifier.context.call_builtin(trig_fn, vec![g]);
        let sub_eq = Equation {
            lhs: trig_call,
            rhs: bound_expr,
            op: sub_op,
        };
        let (set, _) = crate::solver_entrypoints_solve::solve(&sub_eq, var, simplifier).ok()?;
        if matches!(set, SolutionSet::Residual(_) | SolutionSet::Conditional(_)) {
            return None; // unresolved piece: decline the whole relation
        }
        acc = Some(match acc {
            None => set,
            Some(prev) => combine_piu_sets(simplifier, prev, set, conj)?,
        });
    }
    acc
}

/// `c / g(x) {op} k` with nonzero RATIONAL `c`, RATIONAL `k`, and `g` AFFINE in `var` with a
/// NON-RATIONAL constant intercept (`1/(x+√2) > 0`, `3/(x+√5) ≤ 0`, `1/(x+2^(1/3)) > 0`,
/// `1/(x+√2) > 1`, `1/(x+√2) = 1`), detected on the RAW tree. The simplifier RATIONALIZES
/// such denominators through the conjugate (`1/(x+√2) → (√2−x)/(2−x²)`), fabricating a
/// spurious removable pole at the CONJUGATE that the rational path then punches out of the
/// answer, collapses to a false "No solution", or leaks as a malformed residual. Reduce
/// exactly BEFORE it runs:
/// - `k = 0`: `c/g {op} 0 ⟺ g {op'} 0` (the value is never zero; only the true pole is out).
/// - `k ≠ 0` equation: `c/g = k ⟺ g = c/k ⟺ x = (c/k − b)/a` (a single exact point).
/// - `k ≠ 0` inequality: solve `c/u {op} k` in `u = g(x)` space (all-RATIONAL breakpoints
///   `0` and `c/k`, the already-robust path) and map the set back through the monotonic
///   `x = (u − b)/a` (orientation flips for `a < 0`).
///
/// Rational and SYMBOLIC intercepts decline (their owners already solve them).
pub(super) fn try_solve_const_over_surd_affine_inequality(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<(SolutionSet, Vec<crate::SolveStep>)> {
    use cas_ast::RelOp;
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;
    use num_traits::{Signed, Zero};
    if !matches!(
        eq.op,
        RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq | RelOp::Eq
    ) {
        return None;
    }
    // The threshold must be a rational constant.
    let k = as_rational_const(&simplifier.context, eq.rhs)?;
    // Peel negations into the constant's sign; expect `Div(const, g)` underneath.
    let mut neg = false;
    let mut node = eq.lhs;
    while let Expr::Neg(inner) = simplifier.context.get(node) {
        node = *inner;
        neg = !neg;
    }
    let (num, den) = match simplifier.context.get(node) {
        Expr::Div(n, d) => (*n, *d),
        _ => return None,
    };
    if contains_var(&simplifier.context, num, var) {
        return None;
    }
    let mut c = as_rational_const(&simplifier.context, num)?;
    if c.is_zero() {
        return None;
    }
    if neg {
        c = -c;
    }
    // `g` affine with a NON-RATIONAL *constant* intercept: exactly the forms the
    // rationalizer mangles. A rational intercept has a working owner; a symbolic
    // (free-variable) intercept belongs to the generic isolation path.
    let (a, b) = affine_coefficients(simplifier, den, var)?;
    if as_rational_const(&simplifier.context, b).is_some()
        || !cas_ast::collect_variables(&simplifier.context, b).is_empty()
    {
        return None;
    }
    if k.is_zero() {
        if eq.op == RelOp::Eq {
            return None; // `c/g = 0` is the nonzero-const-over-anything guard's job
        }
        let op_is_upper = matches!(eq.op, RelOp::Gt | RelOp::Geq);
        let den_op = if op_is_upper == c.is_positive() {
            RelOp::Gt
        } else {
            RelOp::Lt
        };
        let zero = simplifier.context.num(0);
        // Narration: the sign of `c/g` versus zero IS the (oriented) sign of
        // `g` — one line with the reduced denominator relation.
        let steps = vec![crate::SolveStep::new(
            "Sign of a reciprocal: c/g compares to zero as its denominator does".to_string(),
            Equation {
                lhs: den,
                rhs: zero,
                op: den_op.clone(),
            },
            crate::ImportanceLevel::Medium,
        )];
        return solve_relation_set(simplifier, var, den, zero, den_op).map(|set| (set, steps));
    }
    if eq.op == RelOp::Eq {
        // `g = c/k` exactly: one root `x = (c/k − b)/a`.
        let target = simplifier.context.add(Expr::Number(c / k));
        let root = map_bound_through_inverse_affine(simplifier, target, &a, b);
        return Some((SolutionSet::Discrete(vec![root]), Vec::new()));
    }
    // Inequality vs a nonzero threshold: solve in `u = g(x)` space, where every
    // breakpoint (`u = 0` pole, `u = c/k` boundary) is RATIONAL, then map back.
    let u_name = format!("__{var}_g");
    let u_var = simplifier.context.var(&u_name);
    let c_expr = simplifier.context.add(Expr::Number(c));
    let u_lhs = simplifier.context.add(Expr::Div(c_expr, u_var));
    let u_set = solve_relation_set(simplifier, &u_name, u_lhs, eq.rhs, eq.op.clone())?;
    map_set_through_inverse_affine(simplifier, u_set, &a, b).map(|set| (set, Vec::new()))
}

/// U2 (scout backlog #4): `c / (a·x+b)^(1/q) ⋚ k` — the reciprocal of a
/// ROOT of a rational-affine argument (`1/sqrt(x) > 2`, `1/sqrt(x-1) > 2`,
/// `1/x^(1/3) > 2`). Solved by the two-stage monotone substitution
/// `w = (a·x+b)^(1/q)`: the w-space relation `c/w ⋚ k` has RATIONAL
/// breakpoints (pole 0, boundary c/k) and an existing exact owner; the
/// w-set then maps through the INCREASING power `u = w^q` (clamped to
/// `w > 0` first for even q — the root's domain), and finally through the
/// inverse affine to x. Declines (`None`) on anything outside the shape.
pub(super) fn try_solve_const_over_root_affine_inequality(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;
    use num_traits::{One, Zero};
    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    let k = as_rational_const(&simplifier.context, eq.rhs)?;
    // Peel negations into the numerator's sign; expect `Div(const, root)`.
    let mut neg = false;
    let mut node = eq.lhs;
    while let Expr::Neg(inner) = simplifier.context.get(node) {
        node = *inner;
        neg = !neg;
    }
    let (num, den) = match simplifier.context.get(node) {
        Expr::Div(n, d) => (*n, *d),
        _ => return None,
    };
    if contains_var(&simplifier.context, num, var) {
        return None;
    }
    let mut c = as_rational_const(&simplifier.context, num)?;
    if c.is_zero() {
        return None;
    }
    if neg {
        c = -c;
    }
    // The denominator must be a UNIT root of the affine argument: sqrt(g) or
    // g^(1/q) with an integer q ≥ 2.
    let (g, q): (ExprId, i64) = match simplifier.context.get(den).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let name = simplifier.context.sym_name(fn_id).to_string();
            if name == "sqrt" {
                (args[0], 2)
            } else {
                return None;
            }
        }
        Expr::Pow(base, exp) => {
            let e = as_rational_const(&simplifier.context, exp)?;
            if !e.numer().is_one() {
                return None; // p ≠ 1: valleys/general powers keep their owners
            }
            let q = i64::try_from(e.denom()).ok()?;
            if q < 2 {
                return None;
            }
            (base, q)
        }
        _ => return None,
    };
    if !contains_var(&simplifier.context, g, var) {
        return None;
    }
    let (a, b) = affine_coefficients(simplifier, g, var)?;

    // Stage 1 — w-space: solve `c/w ⋚ k` exactly (rational breakpoints).
    let w_name = format!("__{var}_w");
    let w_lhs = {
        let w_var = simplifier.context.var(&w_name);
        let c_expr = simplifier.context.add(Expr::Number(c));
        simplifier.context.add(Expr::Div(c_expr, w_var))
    };
    let k_expr = simplifier.context.add(Expr::Number(k));
    let w_set = solve_relation_set(simplifier, &w_name, w_lhs, k_expr, eq.op.clone())?;

    // Even q: the root ranges over (0, ∞) as a denominator (g > 0, w > 0) —
    // clamp the w-set before the power map. Odd q: w ranges over ℝ ∖ {0},
    // already excluded by the w-space pole.
    let w_set = if q % 2 == 0 {
        let zero = simplifier.context.num(0);
        let inf = cas_solver_core::solution_set::pos_inf(&mut simplifier.context);
        let positive = SolutionSet::Continuous(cas_ast::Interval {
            min: zero,
            min_type: cas_ast::BoundType::Open,
            max: inf,
            max_type: cas_ast::BoundType::Open,
        });
        cas_solver_core::solution_set::intersect_solution_sets(&simplifier.context, w_set, positive)
    } else {
        w_set
    };

    // Stage 2 — the increasing power map `u = w^q` (monotone on the clamped
    // domain), endpoint by endpoint; ±∞ passes through (q odd keeps −∞).
    let u_set = map_set_through_increasing_power(simplifier, w_set, q)?;

    // Stage 3 — inverse affine back to x.
    map_set_through_inverse_affine(simplifier, u_set, &a, b)
}

/// A linear inequality with the variable on BOTH sides and a SYMBOLIC-CONSTANT
/// coefficient (`x < x·ln2`, from log-linearizing `e^x < 2^x`): the equation-only
/// linear-collect returns the boundary root `{0}` with the operator DROPPED
/// (family F3 of docs/AUDITORIA_FRONTERA_2026-07-13b.md), and
/// `try_parametric_monotone_guard` never fires because its orientation check
/// needs one var-free side. Collect the difference into `c1·x + c0` (both
/// var-free), decide `sign(c1)` with the same exact tri-state oracle, and
/// recurse on `x {op'} −c0/c1` (op flipped when `c1 < 0`). Rational `c1` keeps
/// its existing owners (this handler requires a symbolic constant); an
/// undecidable sign declines honestly.
pub(crate) fn try_symbolic_linear_coeff_inequality(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    use cas_ast::RelOp;
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;

    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    // At least one side must carry the variable; the shape gate below (linear diff with a
    // var-free NON-rational coefficient) is what scopes the handler. One-sided forms with a
    // symbolic-constant coefficient (`x + x·ln2 < 3`) are the same dropped-operator family.
    if !contains_var(&simplifier.context, eq.lhs, var)
        && !contains_var(&simplifier.context, eq.rhs, var)
    {
        return None;
    }
    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (diff, _) = simplifier.simplify(diff);

    // Coefficient of a single additive term as a linear monomial in `var`:
    // `x` → 1, `x·k`/`k·x` → k, `x/k` → 1/k, `Neg(t)` → −coeff(t); the var-bearing
    // factor must be the bare variable. Returns None for any other shape.
    fn linear_term_coeff(ctx: &mut Context, term: ExprId, var: &str) -> Option<Result<ExprId, ()>> {
        use cas_solver_core::isolation_utils::contains_var;
        match ctx.get(term).clone() {
            Expr::Neg(inner) => match linear_term_coeff(ctx, inner, var)? {
                Ok(c) => Some(Ok(ctx.add(Expr::Neg(c)))),
                Err(()) => Some(Err(())),
            },
            Expr::Variable(sym) if ctx.sym_name(sym) == var => Some(Ok(ctx.num(1))),
            Expr::Div(num, den) if !contains_var(ctx, den, var) => {
                match linear_term_coeff(ctx, num, var)? {
                    Ok(c) => Some(Ok(ctx.add(Expr::Div(c, den)))),
                    Err(()) => Some(Err(())),
                }
            }
            Expr::Mul(..) => {
                let leaves = cas_math::expr_nary::mul_leaves(ctx, term);
                let mut var_leaf: Option<ExprId> = None;
                let mut coeff_factors: Vec<ExprId> = Vec::new();
                for &leaf in leaves.iter() {
                    if contains_var(ctx, leaf, var) {
                        if var_leaf.is_some()
                            || !matches!(ctx.get(leaf), Expr::Variable(s) if ctx.sym_name(*s) == var)
                        {
                            return Some(Err(()));
                        }
                        var_leaf = Some(leaf);
                    } else {
                        coeff_factors.push(leaf);
                    }
                }
                var_leaf?;
                let mut c = ctx.num(1);
                for f in coeff_factors {
                    c = ctx.add(Expr::Mul(c, f));
                }
                Some(Ok(c))
            }
            _ => {
                if contains_var(ctx, term, var) {
                    Some(Err(())) // var in a non-linear position: not our shape
                } else {
                    None // constant term
                }
            }
        }
    }

    let mut coeff_terms: Vec<ExprId> = Vec::new();
    let mut const_terms: Vec<ExprId> = Vec::new();
    for term in cas_math::expr_nary::add_leaves(&simplifier.context, diff) {
        match linear_term_coeff(&mut simplifier.context, term, var) {
            Some(Ok(c)) => coeff_terms.push(c),
            Some(Err(())) => return None, // non-linear in var: not our family
            None => const_terms.push(term),
        }
    }
    if coeff_terms.is_empty() {
        return None;
    }
    let mut c1 = simplifier.context.num(0);
    for c in coeff_terms {
        c1 = simplifier.context.add(Expr::Add(c1, c));
    }
    let (c1, _) = simplifier.simplify(c1);
    // A plain rational coefficient has working owners (and pinned fixtures);
    // this handler exists for the SYMBOLIC-constant coefficient family.
    if as_rational_const(&simplifier.context, c1).is_some() {
        return None;
    }
    let mut c0 = simplifier.context.num(0);
    for t in const_terms {
        c0 = simplifier.context.add(Expr::Add(c0, t));
    }
    // Exact tri-state sign, mirroring try_parametric_monotone_guard.
    let sign_of = |simplifier: &Simplifier, e: ExprId| -> Option<std::cmp::Ordering> {
        cas_math::root_forms::provable_sign_vs_zero(&simplifier.context, e)
            .or_else(|| {
                use cas_math::const_sign::{provable_const_sign, ConstSign};
                Some(match provable_const_sign(&simplifier.context, e)? {
                    ConstSign::Negative => std::cmp::Ordering::Less,
                    ConstSign::Zero => std::cmp::Ordering::Equal,
                    ConstSign::Positive => std::cmp::Ordering::Greater,
                })
            })
            .or_else(|| {
                let (lo, hi) = cas_math::const_sign::const_value_bounds(&simplifier.context, e)?;
                use num_traits::Zero;
                let zero = num_rational::BigRational::zero();
                if hi < zero {
                    Some(std::cmp::Ordering::Less)
                } else if lo > zero {
                    Some(std::cmp::Ordering::Greater)
                } else {
                    None
                }
            })
    };
    match sign_of(simplifier, c1) {
        Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Less) => {
            let negative = matches!(sign_of(simplifier, c1), Some(std::cmp::Ordering::Less));
            let neg_c0 = simplifier.context.add(Expr::Neg(c0));
            let ratio = simplifier.context.add(Expr::Div(neg_c0, c1));
            let new_rhs = simplifier.simplify(ratio).0;
            let new_op = if negative {
                cas_solver_core::isolation_utils::flip_inequality(eq.op.clone())
            } else {
                eq.op.clone()
            };
            // Build the ray DIRECTLY: the reduced relation is `x {op'} const`, fully
            // decided here. Recursing into the full solve entrypoint from inside
            // `solve_inner` would RESET the runtime cycle guards and re-enter the
            // strategy pipeline (observed as an infinite strategy loop on
            // `pi^x > 5`); there is nothing left to solve anyway.
            let set = cas_solver_core::solution_set::isolated_var_solution(
                &mut simplifier.context,
                new_rhs,
                new_op,
            );
            Some(Ok((set, Vec::new())))
        }
        // Provably-zero coefficient: the difference is the constant `c0`; let the
        // var-eliminated classifier own it.
        Some(std::cmp::Ordering::Equal) => None,
        None => Some(Err(CasError::SolverError(
            "Inequalities with symbolic coefficients not yet supported".to_string(),
        ))),
    }
}

pub(super) fn try_polynomial_inequality_sign_analysis(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
    roots: &[ExprId],
) -> Option<SolutionSet> {
    use cas_ast::{BoundType, Interval, RelOp};
    use cas_math::polynomial::Polynomial;
    use cas_solver_core::solution_set::{neg_inf, pos_inf};
    use num_rational::BigRational;
    use num_traits::{FromPrimitive, Zero};
    use std::cmp::Ordering;
    use std::collections::HashMap;

    if roots.is_empty() {
        return None;
    }
    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (diff, _) = simplifier.simplify(diff);
    let poly = Polynomial::from_expr(&simplifier.context, diff, var).ok()?;
    let degree = poly.degree();
    if degree < 1 {
        return None;
    }
    let leading = poly.coeffs.last()?;
    if leading.is_zero() {
        return None;
    }
    let sign_lead = if *leading > BigRational::zero() {
        1
    } else {
        -1
    };

    // Numerically order the roots (for placement only; signs are evaluated exactly).
    let mut ordered: Vec<(ExprId, f64)> = Vec::with_capacity(roots.len());
    for &r in roots {
        let v = cas_math::evaluator_f64::eval_f64(&simplifier.context, r, &HashMap::new())?;
        if !v.is_finite() {
            return None;
        }
        ordered.push((r, v));
    }
    ordered.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    // Distinct roots only (the sign chart below assumes simple roots).
    for w in ordered.windows(2) {
        if (w[0].1 - w[1].1).abs() < 1e-9 {
            return None;
        }
    }
    let k = ordered.len();

    // Exact sign of `poly` at a rational test point strictly inside each interval.
    let exact_sign = |q: &BigRational| -> i32 {
        let v = poly.eval(q);
        match v.cmp(&BigRational::zero()) {
            Ordering::Greater => 1,
            Ordering::Less => -1,
            Ordering::Equal => 0,
        }
    };
    let rat = |x: f64| BigRational::from_f64(x);
    let mut signs: Vec<i32> = Vec::with_capacity(k + 1);
    signs.push(exact_sign(&rat(ordered[0].1 - 1.0)?));
    for i in 1..k {
        let mid = (ordered[i - 1].1 + ordered[i].1) / 2.0;
        signs.push(exact_sign(&rat(mid)?));
    }
    signs.push(exact_sign(&rat(ordered[k - 1].1 + 1.0)?));

    // Consistency guards: no test point landed on a root, signs alternate across every simple root,
    // and the unbounded ends match the leading-coefficient end behaviour. Any failure ⇒ the root set
    // is incomplete/mis-ordered ⇒ bail to the raw set rather than emit an unsound interval union.
    if signs.contains(&0) {
        return None;
    }
    if signs.windows(2).any(|w| w[0] == w[1]) {
        return None;
    }
    let end_right = sign_lead;
    let end_left = if degree % 2 == 0 {
        sign_lead
    } else {
        -sign_lead
    };
    if signs[k] != end_right || signs[0] != end_left {
        return None;
    }

    // Build the satisfying interval union.
    let want_positive = matches!(eq.op, RelOp::Gt | RelOp::Geq);
    let strict = matches!(eq.op, RelOp::Lt | RelOp::Gt);
    let ctx = &mut simplifier.context;
    let mut intervals: Vec<Interval> = Vec::new();
    for (j, &sign) in signs.iter().enumerate() {
        if (sign > 0) != want_positive {
            continue;
        }
        let min = if j == 0 {
            neg_inf(ctx)
        } else {
            ordered[j - 1].0
        };
        let max = if j == k { pos_inf(ctx) } else { ordered[j].0 };
        let min_type = if j == 0 || strict {
            BoundType::Open
        } else {
            BoundType::Closed
        };
        let max_type = if j == k || strict {
            BoundType::Open
        } else {
            BoundType::Closed
        };
        intervals.push(Interval {
            min,
            min_type,
            max,
            max_type,
        });
    }
    Some(match intervals.len() {
        0 => SolutionSet::Empty,
        1 => SolutionSet::Continuous(intervals.into_iter().next()?),
        _ => SolutionSet::Union(intervals),
    })
}

/// BIQUADRATIC INEQUALITY `a·x⁴ + b·x² + c {op} 0` solved EXACTLY through the
/// `z = x²` substitution (2026-07-31, cubic-abs cycle): the z-quadratic's sign
/// chart is decided by RATIONAL arithmetic only (discriminant sign, root signs
/// via `√d ⋚ t ⟺ d ⋚ t²`), and every x-shape (bands, punctured bands, outside
/// unions, point sets) is built DIRECTLY with `√z` endpoint expressions — no
/// symbolic comparator anywhere. Registered as a RECOVERY for lossy incumbents
/// only: the generic isolation's unconditional 4th root asserted «No solution»
/// for the tautology `x⁴ − x² + 1 > 0`.
pub(super) fn try_solve_biquadratic_inequality(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::{BoundType, Interval, RelOp};
    use num_rational::BigRational;
    use num_traits::{Signed, Zero};

    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (diff, _) = simplifier.simplify(diff);
    let poly = cas_math::polynomial::Polynomial::from_expr(&simplifier.context, diff, var).ok()?;
    if poly.degree() != 4 || poly.coeffs.len() != 5 {
        return None;
    }
    let zero = BigRational::zero();
    if !poly.coeffs[1].is_zero() || !poly.coeffs[3].is_zero() {
        return None;
    }
    let (mut a, mut b, mut c0) = (
        poly.coeffs[4].clone(),
        poly.coeffs[2].clone(),
        poly.coeffs[0].clone(),
    );
    if a.is_zero() {
        return None;
    }
    // Normalize to a > 0 (flip the relation for a < 0).
    let mut op = eq.op.clone();
    if a.is_negative() {
        a = -a;
        b = -b;
        c0 = -c0;
        op = flip_inequality(op);
    }
    let disc = &b * &b - BigRational::from_integer(4.into()) * &a * &c0;

    // Shape builders with KNOWN endpoint order (√z ≥ 0 by construction).
    let sqrt_of_z = |simplifier: &mut Simplifier, num: BigRational| -> ExprId {
        // z as an exact expression: rational, or (−b ± √d)/(2a) built below.
        let z = simplifier.context.add(Expr::Number(num));
        let call = simplifier.context.call("sqrt", vec![z]);
        simplifier.simplify(call).0
    };
    let sqrt_of_expr = |simplifier: &mut Simplifier, z: ExprId| -> ExprId {
        let call = simplifier.context.call("sqrt", vec![z]);
        simplifier.simplify(call).0
    };
    let neg_of = |simplifier: &mut Simplifier, e: ExprId| -> ExprId {
        let n = simplifier.context.add(Expr::Neg(e));
        simplifier.simplify(n).0
    };
    let band = |simplifier: &mut Simplifier, r: ExprId, closed: bool| -> SolutionSet {
        let bt = if closed {
            BoundType::Closed
        } else {
            BoundType::Open
        };
        let lo = neg_of(simplifier, r);
        SolutionSet::Continuous(Interval {
            min: lo,
            min_type: bt.clone(),
            max: r,
            max_type: bt,
        })
    };
    let outside = |simplifier: &mut Simplifier, r: ExprId, closed: bool| -> Vec<Interval> {
        let bt = if closed {
            BoundType::Closed
        } else {
            BoundType::Open
        };
        let lo = neg_of(simplifier, r);
        let ninf = cas_solver_core::solution_set::neg_inf(&mut simplifier.context);
        let pinf = cas_solver_core::solution_set::pos_inf(&mut simplifier.context);
        vec![
            Interval {
                min: ninf,
                min_type: BoundType::Open,
                max: lo,
                max_type: bt.clone(),
            },
            Interval {
                min: r,
                min_type: bt,
                max: pinf,
                max_type: BoundType::Open,
            },
        ]
    };
    let zero_point = |simplifier: &mut Simplifier| -> ExprId { simplifier.context.num(0) };

    if disc < zero {
        // Constant sign: q(z) > 0 for every z (a > 0).
        return match op {
            RelOp::Gt | RelOp::Geq => Some(SolutionSet::AllReals),
            RelOp::Lt | RelOp::Leq => Some(SolutionSet::Empty),
            _ => None,
        };
    }
    if disc.is_zero() {
        // q(z) = a·(z − z0)², z0 = −b/(2a) rational.
        let z0 = -&b / (BigRational::from_integer(2.into()) * &a);
        return Some(match op {
            RelOp::Lt => SolutionSet::Empty,
            RelOp::Geq => SolutionSet::AllReals,
            RelOp::Leq => {
                // x² = z0.
                if z0.is_negative() {
                    SolutionSet::Empty
                } else if z0.is_zero() {
                    let z = zero_point(simplifier);
                    SolutionSet::Discrete(vec![z])
                } else {
                    let r = sqrt_of_z(simplifier, z0);
                    let nr = neg_of(simplifier, r);
                    SolutionSet::Discrete(vec![nr, r])
                }
            }
            RelOp::Gt => {
                // x² ≠ z0.
                if z0.is_negative() {
                    SolutionSet::AllReals
                } else if z0.is_zero() {
                    let z = zero_point(simplifier);
                    let ninf = cas_solver_core::solution_set::neg_inf(&mut simplifier.context);
                    let pinf = cas_solver_core::solution_set::pos_inf(&mut simplifier.context);
                    SolutionSet::Union(vec![
                        Interval {
                            min: ninf,
                            min_type: BoundType::Open,
                            max: z,
                            max_type: BoundType::Open,
                        },
                        Interval {
                            min: z,
                            min_type: BoundType::Open,
                            max: pinf,
                            max_type: BoundType::Open,
                        },
                    ])
                } else {
                    let r = sqrt_of_z(simplifier, z0.clone());
                    let nr = neg_of(simplifier, r);
                    let ninf = cas_solver_core::solution_set::neg_inf(&mut simplifier.context);
                    let pinf = cas_solver_core::solution_set::pos_inf(&mut simplifier.context);
                    SolutionSet::Union(vec![
                        Interval {
                            min: ninf,
                            min_type: BoundType::Open,
                            max: nr,
                            max_type: BoundType::Open,
                        },
                        Interval {
                            min: nr,
                            min_type: BoundType::Open,
                            max: r,
                            max_type: BoundType::Open,
                        },
                        Interval {
                            min: r,
                            min_type: BoundType::Open,
                            max: pinf,
                            max_type: BoundType::Open,
                        },
                    ])
                }
            }
            _ => return None,
        });
    }

    // disc > 0: distinct roots z1 < z2 = (−b ∓ √d)/(2a). Root SIGNS decided by
    // rational arithmetic: `√d ⋚ t ⟺ d ⋚ t²` for t ≥ 0 (and √d > t for t < 0).
    let cmp_sqrt_d = |t: &BigRational| -> std::cmp::Ordering {
        if t.is_negative() {
            return std::cmp::Ordering::Greater;
        }
        disc.cmp(&(t * t))
    };
    // sign(z2) = sign(−b + √d) = cmp(√d, b) ; sign(z1) = sign(−b − √d) = cmp(−b, √d).
    let z2_sign = cmp_sqrt_d(&b); // Greater ⟹ z2 > 0, Equal ⟹ z2 = 0, Less ⟹ z2 < 0
    let z1_sign = cmp_sqrt_d(&-&b).reverse();
    // Exact z-root expressions (rational when disc is a perfect square, else surds).
    let build_z = |simplifier: &mut Simplifier, plus: bool| -> ExprId {
        let d_expr = simplifier.context.add(Expr::Number(disc.clone()));
        let sq = simplifier.context.call("sqrt", vec![d_expr]);
        let neg_b = simplifier.context.add(Expr::Number(-&b));
        let num = if plus {
            simplifier.context.add(Expr::Add(neg_b, sq))
        } else {
            simplifier.context.add(Expr::Sub(neg_b, sq))
        };
        let den = simplifier
            .context
            .add(Expr::Number(BigRational::from_integer(2.into()) * &a));
        let q = simplifier.context.add(Expr::Div(num, den));
        simplifier.simplify(q).0
    };
    use std::cmp::Ordering as Ord2;
    Some(match op {
        // q(z) < 0 ⟺ z ∈ (z1, z2); with z = x²: z1 < x² < z2.
        RelOp::Lt | RelOp::Leq => {
            let closed = matches!(op, RelOp::Leq);
            match z2_sign {
                Ord2::Less => SolutionSet::Empty,
                Ord2::Equal => {
                    // x² < 0 impossible; x² ≤ z2 = 0 ⟹ x = 0 (z1 < 0 ✓).
                    if closed {
                        let z = zero_point(simplifier);
                        SolutionSet::Discrete(vec![z])
                    } else {
                        SolutionSet::Empty
                    }
                }
                Ord2::Greater => {
                    let z2e = build_z(simplifier, true);
                    let r2 = sqrt_of_expr(simplifier, z2e);
                    match z1_sign {
                        Ord2::Less => band(simplifier, r2, closed),
                        Ord2::Equal => {
                            // x² > 0: puncture the band at 0 (strict only).
                            if closed {
                                band(simplifier, r2, true)
                            } else {
                                let z = zero_point(simplifier);
                                let lo = neg_of(simplifier, r2);
                                SolutionSet::Union(vec![
                                    Interval {
                                        min: lo,
                                        min_type: BoundType::Open,
                                        max: z,
                                        max_type: BoundType::Open,
                                    },
                                    Interval {
                                        min: z,
                                        min_type: BoundType::Open,
                                        max: r2,
                                        max_type: BoundType::Open,
                                    },
                                ])
                            }
                        }
                        Ord2::Greater => {
                            let z1e = build_z(simplifier, false);
                            let r1 = sqrt_of_expr(simplifier, z1e);
                            let bt = if closed {
                                BoundType::Closed
                            } else {
                                BoundType::Open
                            };
                            let n2 = neg_of(simplifier, r2);
                            let n1 = neg_of(simplifier, r1);
                            SolutionSet::Union(vec![
                                Interval {
                                    min: n2,
                                    min_type: bt.clone(),
                                    max: n1,
                                    max_type: bt.clone(),
                                },
                                Interval {
                                    min: r1,
                                    min_type: bt.clone(),
                                    max: r2,
                                    max_type: bt,
                                },
                            ])
                        }
                    }
                }
            }
        }
        // q(z) > 0 ⟺ z < z1 ∨ z > z2: x² < z1 OR x² > z2.
        RelOp::Gt | RelOp::Geq => {
            let closed = matches!(op, RelOp::Geq);
            match z2_sign {
                Ord2::Less => SolutionSet::AllReals,
                Ord2::Equal => {
                    // x² ≥ 0 = z2 always (Geq ⟹ ℝ); strict punctures 0.
                    if closed {
                        SolutionSet::AllReals
                    } else {
                        let z = zero_point(simplifier);
                        let ninf = cas_solver_core::solution_set::neg_inf(&mut simplifier.context);
                        let pinf = cas_solver_core::solution_set::pos_inf(&mut simplifier.context);
                        SolutionSet::Union(vec![
                            Interval {
                                min: ninf,
                                min_type: BoundType::Open,
                                max: z,
                                max_type: BoundType::Open,
                            },
                            Interval {
                                min: z,
                                min_type: BoundType::Open,
                                max: pinf,
                                max_type: BoundType::Open,
                            },
                        ])
                    }
                }
                Ord2::Greater => {
                    let z2e = build_z(simplifier, true);
                    let r2 = sqrt_of_expr(simplifier, z2e);
                    let mut ivs = outside(simplifier, r2, closed);
                    match z1_sign {
                        Ord2::Less => {}
                        Ord2::Equal => {
                            if closed {
                                let z = zero_point(simplifier);
                                ivs.insert(
                                    1,
                                    Interval {
                                        min: z,
                                        min_type: BoundType::Closed,
                                        max: z,
                                        max_type: BoundType::Closed,
                                    },
                                );
                            }
                        }
                        Ord2::Greater => {
                            let z1e = build_z(simplifier, false);
                            let r1 = sqrt_of_expr(simplifier, z1e);
                            let bt = if closed {
                                BoundType::Closed
                            } else {
                                BoundType::Open
                            };
                            let n1 = neg_of(simplifier, r1);
                            ivs.insert(
                                1,
                                Interval {
                                    min: n1,
                                    min_type: bt.clone(),
                                    max: r1,
                                    max_type: bt,
                                },
                            );
                        }
                    }
                    SolutionSet::Union(ivs)
                }
            }
        }
        _ => return None,
    })
}

/// Intersect an inequality interval result with the implicit REAL domain of the LHS expression.
///
/// [`intersect_inequality_with_function_domain`] only fires for a BARE monotonic LHS
/// (`√(x)`/`ln(x)`/`log(b,x)`); when such a function is a FACTOR or subterm
/// (`ln(x)·(x−2)²`, `√x·(x−4)`), its argument-domain (`x > 0`, `x ≥ 0`) was dropped, so the result
/// wrongly kept the region where the expression is UNDEFINED. This intersects the result with each
/// `Positive`/`NonNegative`/`LowerBound` condition of `infer_implicit_domain(lhs)` (`NonZero` poles
/// are already excluded elsewhere). EXACT and EQ-safe: inequality ops only, interval results only,
/// and it falls back to the unchanged set whenever a domain condition cannot be reduced to a clean
/// interval (an honest no-worse-than-before).
pub(super) fn intersect_inequality_with_expression_domain(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
    set: SolutionSet,
) -> SolutionSet {
    use cas_ast::RelOp;
    use cas_solver_core::solution_set::intersect_solution_sets;

    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return set;
    }
    if !matches!(set, SolutionSet::Continuous(_) | SolutionSet::Union(_)) {
        return set;
    }
    let domain =
        cas_solver_core::domain_inference::infer_implicit_domain(&simplifier.context, eq.lhs, true);
    let mut conds: Vec<ImplicitCondition> = domain.conditions().iter().cloned().collect();
    // The RHS carries implicit domain too: `ln(x) > ln(3−x)` requires `3−x > 0`, but the
    // monotonic isolation only records it as a `Requires` side-condition that never reaches
    // the emitted set. A point where EITHER side is undefined cannot satisfy the relation,
    // and intersecting can only SHRINK the set — same exactness/fallback discipline as LHS.
    let rhs_domain =
        cas_solver_core::domain_inference::infer_implicit_domain(&simplifier.context, eq.rhs, true);
    conds.extend(rhs_domain.conditions().iter().cloned());
    let mut result = set;
    for cond in conds {
        let (arg, threshold, op) = match cond {
            ImplicitCondition::Positive(arg) => (arg, None, RelOp::Gt),
            ImplicitCondition::NonNegative(arg) => (arg, None, RelOp::Geq),
            ImplicitCondition::LowerBound(arg, c) => (arg, Some(c), RelOp::Geq),
            // `arg ≠ 0` (pole) is excluded by the rational-inequality path, not a half-line.
            ImplicitCondition::NonZero(_) => continue,
            // A branch annotation is not a real-domain constraint.
            ImplicitCondition::PrincipalBranch { .. } => continue,
        };
        let rhs = match threshold {
            Some(c) => simplifier.context.add(Expr::Number(c)),
            None => simplifier.context.num(0),
        };
        let domain_eq = Equation { lhs: arg, rhs, op };
        if let Ok((
            d @ (SolutionSet::Continuous(_)
            | SolutionSet::Union(_)
            | SolutionSet::Empty
            | SolutionSet::AllReals),
            _,
        )) = crate::solver_entrypoints_solve::solve(&domain_eq, var, simplifier)
        {
            result = intersect_solution_sets(&simplifier.context, result, d);
        }
    }
    result
}

/// Union the dropped boundary roots of a NON-STRICT inequality back into its interval solution.
///
/// For `f ≤ 0` / `f ≥ 0` every real, in-domain root of `f = lhs − rhs` is a solution (the value `0`
/// satisfies both), but the interval sign-analysis only emits the sign-CHANGE regions and silently
/// drops the isolated roots of even-multiplicity factors that fall outside them. We re-solve the
/// EQUATION `lhs = rhs` (which already excludes poles and filters extraneous/non-finite roots via the
/// same `filter_real_solutions` pass) and union its discrete roots, as degenerate `[p, p]` intervals,
/// into the result. Strict inequalities (`<` / `>`) are left untouched — `0` does NOT satisfy them.
pub(super) fn union_non_strict_inequality_roots(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: CoreSolverOptions,
    ctx: &SolveCtx,
    set: SolutionSet,
) -> SolutionSet {
    use cas_ast::RelOp;
    use cas_solver_core::solution_set::union_solution_sets;

    // Non-strict only: a root of `f` satisfies `f ≤ 0` and `f ≥ 0`, but NOT `f < 0` / `f > 0`.
    if !matches!(eq.op, RelOp::Leq | RelOp::Geq) {
        return set;
    }
    // `AllReals` already contains every root; `Residual`/`Conditional` cannot be cleanly augmented.
    if !matches!(
        set,
        SolutionSet::Continuous(_)
            | SolutionSet::Union(_)
            | SolutionSet::Discrete(_)
            | SolutionSet::Empty
    ) {
        return set;
    }

    let eq_roots = Equation {
        lhs: eq.lhs,
        rhs: eq.rhs,
        op: RelOp::Eq,
    };
    let Ok((roots, _)) = solve_local_core(&eq_roots, var, simplifier, opts, ctx) else {
        return set;
    };
    // Only genuine discrete roots can be unioned in; anything else means no isolated point to add.
    if !matches!(roots, SolutionSet::Discrete(_)) {
        return set;
    }
    // A root that is ALREADY a closed endpoint of `set` is provably in the solution (e.g. the
    // boundaries `1/e`, `e` of `ln(x)^2 ≤ 1` -> `[1/e, e]`), so unioning it is a mathematical
    // no-op. Skip those by EXACT endpoint identity — `union_solution_sets`/`merge_intervals`
    // order endpoints through the rational-only `compare_values`, which cannot order bounds
    // containing `E` (`e^√t`) and would otherwise CORRUPT the band into its two endpoints. The
    // genuinely-dropped roots this function exists for are isolated INTERIOR points (not
    // endpoints), so they survive the filter and are still unioned in.
    let SolutionSet::Discrete(points) = roots else {
        return set;
    };
    let missing: Vec<ExprId> = points
        .into_iter()
        .filter(|&p| !point_is_closed_endpoint(&set, p))
        .collect();
    if missing.is_empty() {
        return set;
    }
    let roots_intervals = discrete_to_intervals(SolutionSet::Discrete(missing));
    union_solution_sets(&simplifier.context, set, roots_intervals)
}
