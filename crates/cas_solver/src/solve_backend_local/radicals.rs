//! `solve_backend_local`: familia `radicals`.
//!
//! Ver la cabecera de `solve_backend_local.rs` para el contexto.

use super::*;

/// Back-substitute `root` for `var` in the original equation and check, over the
/// reals, whether the two sides agree. Used to reject extraneous roots that the
/// case-split solver returns without verification (e.g. `solve(|x| = x-1)`
/// returns `1/2`, but `|1/2| = 1/2 ≠ 1/2 - 1 = -1/2`).
///
/// EXACT (R2, auditoría 2026-07-30 fichas S2b-001/S3-002/Q3c-001): the old
/// f64 back-substitution dropped EXACT rational roots because its tolerance
/// scaled with the RESULT while the rounding error scales with the
/// COEFFICIENTS (`solve((585738843·x − 85131377)·(x−2) = 0)` published the
/// PARTIAL set `{2}` with empty warnings; the `(x+2^k)²`-family flipped to
/// "No solution" exactly at 54 bits), and its confirming direction published
/// spurious roots the exact oracle refutes (`sqrt(x) + 10^(-10) = 0` →
/// `{10^(-20)}`). Now: substitute the rational candidate and decide the sign
/// of `lhs − rhs` with the exact constant oracle (interval arithmetic with
/// directed rounding; rationals, surds, abs, π/e, Taylor-bounded trig).
/// `Zero` ⟹ Verified, `Positive`/`Negative` ⟹ Extraneous, undecidable ⟹
/// Unknown (kept — a probe never confirms, and here not even refutes,
/// beyond what is PROVEN). No magnitude cap: exact arithmetic does not
/// cancel catastrophically.
pub(super) fn check_root(ctx: &mut Context, eq: &Equation, var: &str, root: ExprId) -> RootCheck {
    use cas_math::const_sign::{provable_const_sign, ConstSign};
    use cas_solver_core::substitution::substitute_named_var;

    // Contract unchanged: only rational candidates are back-checked;
    // symbolic/irrational roots stay Unknown (kept).
    if cas_math::numeric_eval::as_rational_const(ctx, root).is_none() {
        return RootCheck::Unknown;
    }
    let lhs_at = substitute_named_var(ctx, eq.lhs, var, root);
    let rhs_at = substitute_named_var(ctx, eq.rhs, var, root);
    let diff = ctx.add(Expr::Sub(lhs_at, rhs_at));
    match provable_const_sign(ctx, diff) {
        Some(ConstSign::Zero) => RootCheck::Verified,
        Some(ConstSign::Positive) | Some(ConstSign::Negative) => RootCheck::Extraneous,
        // Includes undefined at the candidate (e.g. a vanishing denominator):
        // the domain-condition filters own that classification.
        None => RootCheck::Unknown,
    }
}

/// Sign range of the AFFINE-over-radicals normal form `c₀ + Σ cᵢ·rᵢ`, where
/// every `rᵢ` is a nonnegative atom (√u, u^(p/(2q)), |u|, u^(2m)) and every
/// coefficient is rational — the shape of quadratic roots and their affine
/// shifts even UNSIMPLIFIED (`(1−√(4a−3))/2 − 1` never distributes before the
/// filter sees it, so the recursive range walk alone reads `1−√u` as Unknown).
/// Neg iff c₀ < 0 ∧ ∀cᵢ ≤ 0 (each rᵢ ≥ 0 can only push further negative);
/// Pos symmetric; `None` when the tree has any other shape.
fn radical_affine_sign_range(ctx: &Context, e: ExprId) -> Option<RadicalSignRange> {
    use num_rational::BigRational;
    use num_traits::{One, Zero};

    /// `Some(radicand)` for `√u` / `u^(1/2)` (the pairable shapes for the
    /// dominance rule), `None` marker for the other nonnegative atoms.
    fn nonneg_radical_atom(ctx: &Context, e: ExprId) -> Option<Option<ExprId>> {
        use num_integer::Integer as _;
        match ctx.get(e) {
            Expr::Function(fn_id, args) if args.len() == 1 => {
                if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Sqrt) {
                    Some(Some(args[0]))
                } else if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Abs) {
                    Some(None)
                } else {
                    None
                }
            }
            Expr::Pow(b, x) => match cas_math::numeric_eval::as_rational_const(ctx, *x) {
                Some(q) if q == num_rational::BigRational::new(1.into(), 2.into()) => {
                    Some(Some(*b))
                }
                Some(q) if !q.is_integer() && q.denom().is_even() => Some(None),
                Some(q) if q.is_integer() && q.to_integer().is_even() => Some(None),
                _ => None,
            },
            _ => None,
        }
    }
    struct Collect {
        c0: BigRational,
        radicals: Vec<(BigRational, Option<ExprId>)>,
        sym: Option<(ExprId, BigRational)>,
    }
    fn walk(ctx: &Context, e: ExprId, scale: BigRational, acc: &mut Collect) -> bool {
        if let Some(q) = cas_math::numeric_eval::as_rational_const(ctx, e) {
            acc.c0 += scale * q;
            return true;
        }
        if let Some(radicand) = nonneg_radical_atom(ctx, e) {
            acc.radicals.push((scale, radicand));
            return true;
        }
        match ctx.get(e) {
            Expr::Neg(u) => walk(ctx, *u, -scale, acc),
            Expr::Add(l, r) => {
                let (l, r) = (*l, *r);
                walk(ctx, l, scale.clone(), acc) && walk(ctx, r, scale, acc)
            }
            Expr::Sub(l, r) => {
                let (l, r) = (*l, *r);
                walk(ctx, l, scale.clone(), acc) && walk(ctx, r, -scale, acc)
            }
            Expr::Mul(l, r) => {
                let (l, r) = (*l, *r);
                if let Some(q) = cas_math::numeric_eval::as_rational_const(ctx, l) {
                    walk(ctx, r, scale * q, acc)
                } else if let Some(q) = cas_math::numeric_eval::as_rational_const(ctx, r) {
                    walk(ctx, l, scale * q, acc)
                } else {
                    false
                }
            }
            Expr::Div(n, d) => {
                let (n, d) = (*n, *d);
                match cas_math::numeric_eval::as_rational_const(ctx, d) {
                    Some(q) if !q.is_zero() => walk(ctx, n, scale / q, acc),
                    _ => false,
                }
            }
            // A lone non-radical symbolic atom is admitted ONCE (the `v` of
            // the dominance rule below); a second distinct one bails.
            _ => match &mut acc.sym {
                Some((v, s)) if *v == e => {
                    *s += scale;
                    true
                }
                Some(_) => false,
                None => {
                    acc.sym = Some((e, scale));
                    true
                }
            },
        }
    }

    let mut acc = Collect {
        c0: BigRational::zero(),
        radicals: Vec::new(),
        sym: None,
    };
    if !walk(ctx, e, BigRational::one(), &mut acc) {
        return None;
    }
    let zero = BigRational::zero();
    let Collect { c0, radicals, sym } = acc;
    if let Some((v, s)) = sym {
        // DOMINANCE rule: `c₀ + s·v + t·√(q·v² + d)` with q > 0, d > 0 (no
        // linear term) satisfies `√(q·v²+d) > √q·|v|`, so for t < 0 and
        // q·t² ≥ s² the radical term outweighs the symbol term STRICTLY:
        // s·v + t·√ < (|s| − |t|√q)·|v| ≤ 0, hence Neg when c₀ ≤ 0 (the
        // always-extraneous root `(b − √(b²+4))/2` of `√(bx+1) = x`).
        // Everything else with a symbol atom stays Unknown.
        if s.is_zero() {
            return None; // symbol cancelled; conservative (rare)
        }
        let [(t, Some(u))] = radicals.as_slice() else {
            return None;
        };
        let Expr::Variable(vsym) = ctx.get(v) else {
            return None;
        };
        let vname = ctx.sym_name(*vsym).to_string();
        let u_poly = cas_math::polynomial::Polynomial::from_expr(ctx, *u, &vname).ok()?;
        let coeffs = &u_poly.coeffs;
        // Radicand must be exactly q·v² + d, q > 0, d > 0.
        if coeffs.len() != 3 || !coeffs[1].is_zero() {
            return None;
        }
        let (d, q) = (&coeffs[0], &coeffs[2]);
        if *d <= zero || *q <= zero {
            return None;
        }
        let dominates = q * t * t >= &s * &s;
        return Some(if *t < zero && dominates && c0 <= zero {
            RadicalSignRange::Neg
        } else if *t > zero && dominates && c0 >= zero {
            RadicalSignRange::Pos
        } else {
            RadicalSignRange::Unknown
        });
    }
    Some(if c0 < zero && radicals.iter().all(|(c, _)| *c <= zero) {
        RadicalSignRange::Neg
    } else if c0 > zero && radicals.iter().all(|(c, _)| *c >= zero) {
        RadicalSignRange::Pos
    } else if c0.is_zero() && radicals.iter().all(|(c, _)| c.is_zero()) {
        RadicalSignRange::Zero
    } else if c0 <= zero && radicals.iter().all(|(c, _)| *c <= zero) {
        RadicalSignRange::NonPos
    } else if c0 >= zero && radicals.iter().all(|(c, _)| *c >= zero) {
        RadicalSignRange::NonNeg
    } else {
        RadicalSignRange::Unknown
    })
}

fn radical_real_sign_range(ctx: &Context, e: ExprId) -> RadicalSignRange {
    use num_integer::Integer as _;
    use num_traits::{Signed, Zero};
    use RadicalSignRange::*;
    let add = |a: RadicalSignRange, b: RadicalSignRange| -> RadicalSignRange {
        match (a, b) {
            (Zero, s) | (s, Zero) => s,
            (Pos, Pos) | (Pos, NonNeg) | (NonNeg, Pos) => Pos,
            (NonNeg, NonNeg) => NonNeg,
            (Neg, Neg) | (Neg, NonPos) | (NonPos, Neg) => Neg,
            (NonPos, NonPos) => NonPos,
            _ => Unknown,
        }
    };
    let neg = |s: RadicalSignRange| -> RadicalSignRange {
        match s {
            Pos => Neg,
            NonNeg => NonPos,
            Zero => Zero,
            NonPos => NonNeg,
            Neg => Pos,
            Unknown => Unknown,
        }
    };
    let mul = |a: RadicalSignRange, b: RadicalSignRange| -> RadicalSignRange {
        match (a, b) {
            // The whole expression is a real VALUE under the filter premise,
            // so a zero factor annihilates even an undecided cofactor.
            (Zero, _) | (_, Zero) => Zero,
            (Unknown, _) | (_, Unknown) => Unknown,
            (Pos, s) | (s, Pos) => s,
            (Neg, s) | (s, Neg) => neg(s),
            (NonNeg, NonNeg) | (NonPos, NonPos) => NonNeg,
            (NonNeg, NonPos) | (NonPos, NonNeg) => NonPos,
        }
    };
    match ctx.get(e) {
        Expr::Number(q) => {
            if q.is_zero() {
                Zero
            } else if q.is_positive() {
                Pos
            } else {
                Neg
            }
        }
        Expr::Constant(cas_ast::Constant::Pi | cas_ast::Constant::E) => Pos,
        Expr::Neg(u) => neg(radical_real_sign_range(ctx, *u)),
        Expr::Add(l, r) => add(
            radical_real_sign_range(ctx, *l),
            radical_real_sign_range(ctx, *r),
        ),
        Expr::Sub(l, r) => add(
            radical_real_sign_range(ctx, *l),
            neg(radical_real_sign_range(ctx, *r)),
        ),
        Expr::Mul(l, r) => mul(
            radical_real_sign_range(ctx, *l),
            radical_real_sign_range(ctx, *r),
        ),
        Expr::Div(n, d) => {
            let ds = radical_real_sign_range(ctx, *d);
            // Only a denominator of PROVEN strict sign divides safely (a
            // NonNeg denominator could be 0 — the quotient would not be the
            // real value the premise promises).
            match ds {
                Pos => radical_real_sign_range(ctx, *n),
                Neg => neg(radical_real_sign_range(ctx, *n)),
                _ => Unknown,
            }
        }
        Expr::Pow(b, x) => {
            let (b, x) = (*b, *x);
            match cas_math::numeric_eval::as_rational_const(ctx, x) {
                // Principal even root of a real value: ≥ 0 (premise).
                Some(q) if !q.is_integer() && q.denom().is_even() => NonNeg,
                Some(q) if q.is_integer() => {
                    let bs = radical_real_sign_range(ctx, b);
                    if q.to_integer().is_even() {
                        match bs {
                            Zero => Zero,
                            Pos | Neg => Pos,
                            _ => NonNeg,
                        }
                    } else {
                        bs
                    }
                }
                _ => Unknown,
            }
        }
        Expr::Function(fn_id, args) if args.len() == 1 => {
            if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Sqrt)
                || ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Abs)
            {
                NonNeg
            } else {
                Unknown
            }
        }
        _ => Unknown,
    }
}

/// True when `root` PROVABLY violates one of the equation's recorded real-domain
/// conditions (`required_conditions`), making it an extraneous root the solver
/// emitted without enforcing the domain it itself derived — e.g.
/// `solve(ln(x)+ln(x+5)=0)` returns the negative root `½(-√29-5)` which violates
/// `x > 0`. The check is EXACT: it substitutes the root into the condition target
/// and decides the sign with [`provable_sign_vs_zero`] (which handles a single
/// quadratic surd `A + B·√n`, the shape of every quadratic root). A `None` (sign
/// not provable) or any non-matching condition KEEPS the root — we only ever drop
/// on a proof, never on a float estimate, so a valid root can never be lost.
pub(super) fn root_violates_required_condition(
    ctx: &mut Context,
    var: &str,
    root: ExprId,
    conds: &[ImplicitCondition],
) -> bool {
    use cas_math::root_forms::{provable_sign_vs_zero, provable_sign_vs_zero_const_radicand};
    use num_traits::Zero;
    use std::cmp::Ordering;

    if conds.is_empty() {
        return false;
    }
    // Exact sign of a root vs 0: the rational-radicand prover first, then the transcendental-radicand
    // one (radicand `9 + 4e` etc.). Both are proofs, never float estimates, so a valid root is never
    // dropped — a `None` simply keeps the root.
    let sign_vs_zero = |ctx: &Context, at: ExprId| -> Option<Ordering> {
        provable_sign_vs_zero(ctx, at)
            .or_else(|| provable_sign_vs_zero_const_radicand(ctx, at))
            .or_else(|| {
                // Exact interval bounds for the named constants `phi`, `e`, `π` and their arithmetic
                // (`const_value_bounds` uses arbitrary-precision sqrt/interval arithmetic, never an f64
                // estimate). This decides the sign of a root the surd parser cannot read — a radical
                // equation whose squared quadratic is `x²-x-1` returns the golden-ratio constant `phi`,
                // and `-phi < 0` is exactly what rejects the extraneous root of `√(x+1) = -x`.
                let (lo, hi) = cas_math::const_sign::const_value_bounds(ctx, at)?;
                let zero = num_rational::BigRational::zero();
                if hi < zero {
                    Some(Ordering::Less)
                } else if lo > zero {
                    Some(Ordering::Greater)
                } else if lo.is_zero() && hi.is_zero() {
                    Some(Ordering::Equal)
                } else {
                    None // bounds straddle 0 — undecided, keep the root
                }
            })
            .or_else(|| {
                // F10: SYMBOLIC-parameter roots (`(−√(4a+1)−1)/2`) are beyond
                // every constant oracle above. The affine-over-radicals
                // collector (reads unsimplified shifts like `(1−√u)/2 − 1`)
                // and the structural range walk decide them under the
                // root-filter premise (radicals of a candidate real solution
                // are real, principal roots ≥ 0) — still a proof, never an
                // estimate; ranges that include 0 or stay undecided keep the
                // root.
                let range = radical_affine_sign_range(ctx, at)
                    .filter(|r| !matches!(r, RadicalSignRange::Unknown))
                    .unwrap_or_else(|| radical_real_sign_range(ctx, at));
                match range {
                    RadicalSignRange::Pos => Some(Ordering::Greater),
                    RadicalSignRange::Neg => Some(Ordering::Less),
                    RadicalSignRange::Zero => Some(Ordering::Equal),
                    _ => None,
                }
            })
    };
    let var_id = ctx.var(var);
    for cond in conds {
        let violates = match cond {
            // A branch annotation is informational, never a root filter.
            ImplicitCondition::PrincipalBranch { .. } => false,
            // ln(e)/log(e) require e > 0; e ≤ 0 at the root is a violation
            // (e = 0 makes the log undefined, so it is extraneous too).
            ImplicitCondition::Positive(e) => {
                let at = substitute_expr_by_id(ctx, *e, var_id, root);
                matches!(
                    sign_vs_zero(ctx, at),
                    Some(Ordering::Less | Ordering::Equal)
                )
            }
            // sqrt(e) requires e ≥ 0; only e < 0 violates (boundary e = 0 is fine).
            ImplicitCondition::NonNegative(e) => {
                let at = substitute_expr_by_id(ctx, *e, var_id, root);
                matches!(sign_vs_zero(ctx, at), Some(Ordering::Less))
            }
            // 1/e requires e ≠ 0; only a PROVABLE exact zero violates.
            ImplicitCondition::NonZero(e) => {
                let at = substitute_expr_by_id(ctx, *e, var_id, root);
                matches!(sign_vs_zero(ctx, at), Some(Ordering::Equal))
            }
            // acosh(e) etc. require e ≥ lower; only e − lower < 0 violates.
            ImplicitCondition::LowerBound(e, lower) => {
                let at = substitute_expr_by_id(ctx, *e, var_id, root);
                let lb = ctx.add(Expr::Number(lower.clone()));
                let shifted = ctx.add(Expr::Sub(at, lb));
                matches!(sign_vs_zero(ctx, shifted), Some(Ordering::Less))
            }
        };
        if violates {
            return true;
        }
    }
    false
}

/// True if `expr` contains any square-root term `Pow(_, 1/2)`.
pub(super) fn expr_contains_sqrt(ctx: &Context, expr: ExprId) -> bool {
    if as_sqrt_radicand(ctx, expr).is_some() {
        return true;
    }
    match ctx.get(expr).clone() {
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            expr_contains_sqrt(ctx, l) || expr_contains_sqrt(ctx, r)
        }
        Expr::Neg(i) | Expr::Hold(i) => expr_contains_sqrt(ctx, i),
        Expr::Function(_, args) => args.iter().any(|&a| expr_contains_sqrt(ctx, a)),
        _ => false,
    }
}

pub(super) fn collect_radical_split(ctx: &Context, d: ExprId, var: &str) -> Option<RadicalSplit> {
    fn walk(
        ctx: &Context,
        e: ExprId,
        sign: i8,
        var: &str,
        rad: &mut Option<(i8, ExprId)>,
        rest: &mut Vec<(i8, ExprId)>,
    ) -> bool {
        match ctx.get(e) {
            Expr::Add(l, r) => {
                let (l, r) = (*l, *r);
                walk(ctx, l, sign, var, rad, rest) && walk(ctx, r, sign, var, rad, rest)
            }
            Expr::Sub(l, r) => {
                let (l, r) = (*l, *r);
                walk(ctx, l, sign, var, rad, rest) && walk(ctx, r, -sign, var, rad, rest)
            }
            Expr::Neg(inner) => {
                let inner = *inner;
                walk(ctx, inner, -sign, var, rad, rest)
            }
            _ => {
                if let Some(radicand) = as_sqrt_radicand(ctx, e) {
                    if expr_contains_named_var(ctx, radicand, var) {
                        if rad.is_some() {
                            return false; // a second radical
                        }
                        *rad = Some((sign, radicand));
                        return true;
                    }
                }
                rest.push((sign, e));
                true
            }
        }
    }
    let mut rad = None;
    let mut rest = Vec::new();
    if !walk(ctx, d, 1, var, &mut rad, &mut rest) {
        return None;
    }
    let (s, f) = rad?;
    Some((s, f, rest))
}

/// Keep the roots `r` of `f = g²` for which `g(r) ≥ 0` — the genuine boundary `√f = g` points
/// (`√f = |g| = g` requires `g ≥ 0`). `g` is affine and each root a quadratic surd, so `g(r)` is a
/// quadratic surd whose sign `compare_values` decides exactly. Non-`Discrete` root sets (no isolated
/// roots, or the degenerate `f ≡ g²` case which only arises for perfect-square radicands the hook
/// never reaches) contribute no boundary points.
pub(super) fn keep_roots_with_g_nonneg(
    simplifier: &mut Simplifier,
    var: &str,
    roots: SolutionSet,
    g: ExprId,
) -> SolutionSet {
    use cas_solver_core::solution_set::compare_values;
    let points = match roots {
        SolutionSet::Discrete(p) => p,
        _ => return SolutionSet::Empty,
    };
    let var_expr = simplifier.context.var(var);
    let zero = simplifier.context.num(0);
    let kept: Vec<ExprId> = points
        .into_iter()
        .filter(|&r| {
            let g_at_r = substitute_expr_by_id(&mut simplifier.context, g, var_expr, r);
            let (g_at_r, _) = simplifier.simplify(g_at_r);
            compare_values(&simplifier.context, g_at_r, zero) != std::cmp::Ordering::Less
        })
        .collect();
    if kept.is_empty() {
        SolutionSet::Empty
    } else {
        SolutionSet::Discrete(kept)
    }
}

/// If `expr` is `Pow(radicand, 1/2)` (a square root), return the radicand.
fn as_sqrt_radicand(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    use cas_math::numeric_eval::as_rational_const;
    use num_rational::BigRational;
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        let (base, exp) = (*base, *exp);
        if as_rational_const(ctx, exp)? == BigRational::new(1.into(), 2.into()) {
            return Some(base);
        }
    }
    None
}

/// Flatten `expr` into exactly two unit-coefficient square-root radicands (each
/// containing `var`) plus a rational constant remainder: `√f + √g + d`. Returns
/// `(f, g, d)` or None for any other shape (a radical with a coefficient or a
/// minus sign, a third radical, a bare `x` outside a radical, a non-rational
/// constant).
fn collect_two_sqrt_and_const(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, i8, ExprId, i8, num_rational::BigRational)> {
    use cas_math::numeric_eval::as_rational_const;
    use num_rational::BigRational;
    use num_traits::Zero;

    fn walk(
        ctx: &Context,
        expr: ExprId,
        sign: i8,
        var: &str,
        rads: &mut Vec<(ExprId, i8)>,
        constant: &mut BigRational,
    ) -> bool {
        match ctx.get(expr) {
            Expr::Add(l, r) => {
                let (l, r) = (*l, *r);
                walk(ctx, l, sign, var, rads, constant) && walk(ctx, r, sign, var, rads, constant)
            }
            Expr::Sub(l, r) => {
                let (l, r) = (*l, *r);
                walk(ctx, l, sign, var, rads, constant) && walk(ctx, r, -sign, var, rads, constant)
            }
            Expr::Neg(inner) => {
                let inner = *inner;
                walk(ctx, inner, -sign, var, rads, constant)
            }
            _ => {
                if let Some(radicand) = as_sqrt_radicand(ctx, expr) {
                    // A radical `±√(radicand)` with the variable inside — keep its running sign so a
                    // DIFFERENCE `√f − √g` is handled, not only a sum.
                    if expr_contains_named_var(ctx, radicand, var) {
                        rads.push((radicand, sign));
                        return true;
                    }
                    return false;
                }
                if expr_contains_named_var(ctx, expr, var) {
                    return false; // a bare `x` (or other x-term) outside a radical
                }
                match as_rational_const(ctx, expr) {
                    Some(q) => {
                        if sign >= 0 {
                            *constant += q;
                        } else {
                            *constant -= q;
                        }
                        true
                    }
                    None => false, // non-rational constant (π, e, …)
                }
            }
        }
    }

    let mut rads: Vec<(ExprId, i8)> = Vec::new();
    let mut constant = BigRational::zero();
    if !walk(ctx, expr, 1, var, &mut rads, &mut constant) || rads.len() != 2 {
        return None;
    }
    Some((rads[0].0, rads[0].1, rads[1].0, rads[1].1, constant))
}

/// Exact rational square root: returns `√q` when `q ≥ 0` and both numerator and
/// denominator are perfect squares, else None (so `√q` is irrational).
fn perfect_rational_sqrt(q: &num_rational::BigRational) -> Option<num_rational::BigRational> {
    use num_rational::BigRational;
    use num_traits::Signed;
    if q.is_negative() {
        return None;
    }
    let (n, d) = (q.numer(), q.denom());
    let sn = n.sqrt();
    let sd = d.sqrt();
    if &(sn.clone() * &sn) == n && &(sd.clone() * &sd) == d {
        Some(BigRational::new(sn, sd))
    } else {
        None
    }
}

/// Solve an EQUATION that is a sum of two square roots equal to a constant,
/// `√f + √g = c` (e.g. `√(x+3) + √x = 3`). Reduce by squaring once to the single
/// radical `√(f·g) = (c² − f − g)/2`, solve that recursively, then keep only the
/// candidates that EXACTLY satisfy the original — `f(r) ≥ 0`, `g(r) ≥ 0`, and
/// `√f(r) + √g(r) = c` (both radicands perfect rational squares summing to `c`) —
/// which drops the extraneous roots that squaring and the spurious `f,g < 0`
/// branch of the reduced equation introduce. Without this, the isolation path
/// leaks `Solve: solve(x − (c − √g)^(1/(1/2)) = 0, x) = 0` and drops the root.
///
/// Scoped to RATIONAL candidates: a non-rational candidate (surd root) declines
/// (falls back to the existing path) rather than risk an unverified extraneous
/// root — surd-root sums of radicals remain a follow-up.
pub(super) fn try_solve_sum_of_two_radicals_equation(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_math::numeric_eval::as_rational_const;
    use cas_math::polynomial::Polynomial;
    use num_traits::{Signed, Zero};

    if eq.op != cas_ast::RelOp::Eq {
        return None;
    }
    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (expr, _) = simplifier.simplify(diff);

    let (f, sign_f, g, sign_g, constant) =
        collect_two_sqrt_and_const(&simplifier.context, expr, var)?;
    // `s_f·√f + s_g·√g + constant = 0`  ⇒  `s_f·√f + s_g·√g = c` with `c = −constant`.
    let c = -constant;
    // A SUM of square roots (both signs +) is never negative — the difference has no such bound.
    if sign_f == 1 && sign_g == 1 && c.is_negative() {
        return Some(SolutionSet::Empty);
    }

    // Radicands must be polynomials (to evaluate the verification exactly).
    let f_poly = Polynomial::from_expr(&simplifier.context, f, var).ok()?;
    let g_poly = Polynomial::from_expr(&simplifier.context, g, var).ok()?;

    // Squaring `s_f·√f + s_g·√g = c` gives `f + g + 2·s_f·s_g·√(fg) = c²`, i.e. the reduced single
    // radical `√(f·g) = s_f·s_g·(c² − f − g)/2` (the difference flips the sign of the RHS).
    let c2 = simplifier.context.add(Expr::Number(c.clone() * &c));
    let c2_minus_f = simplifier.context.add(Expr::Sub(c2, f));
    let c2_minus_f_minus_g = simplifier.context.add(Expr::Sub(c2_minus_f, g));
    // `s_f·s_g = −1` (a difference) negates the numerator.
    let numerator = if sign_f * sign_g == 1 {
        c2_minus_f_minus_g
    } else {
        simplifier.context.add(Expr::Neg(c2_minus_f_minus_g))
    };
    let two = simplifier.context.num(2);
    let reduced_rhs_raw = simplifier.context.add(Expr::Div(numerator, two));
    let reduced_rhs_poly = Polynomial::from_expr(&simplifier.context, reduced_rhs_raw, var).ok()?;

    // Square `√(fg) = reduced_rhs` to the POLYNOMIAL `fg − reduced_rhs² = 0` and solve
    // THAT, rather than delegating `√(fg) = reduced_rhs` to the single-radical solver:
    // that solver drops or empties roots for several `√(quad) = c·x` monomial-RHS forms
    // (`√(5x²+9x−2) = 3x` → wrong "No solution", `√(5x²+9x) = 3x` → drops `9/4`), which
    // turned every difference `√f − √g = c` whose reduced RHS is a bare monomial into a
    // wrong "No solution". The exact verification below (perfect-square radicands summing
    // to `c`) re-imposes `reduced_rhs ≥ 0` and every domain condition, so squaring here
    // introduces no unfiltered extraneous root.
    let reduced_poly = f_poly
        .mul(&g_poly)
        .sub(&reduced_rhs_poly.mul(&reduced_rhs_poly));
    // A CONSTANT `fg − reduced_rhs²` needs no variable solve (and `solve(const = 0, x)`
    // with no `x` would leak): a nonzero constant means no `x` satisfies the squared
    // equation → the original sum/difference has NO solution (`√(x+1) + √x = 0`, which
    // squares to `−1/4 = 0`); the zero polynomial is the `fg ≡ reduced_rhs²` identity,
    // a continuum we cannot enumerate → decline.
    if reduced_poly.degree() == 0 {
        return if reduced_poly.is_zero() {
            None
        } else {
            Some(SolutionSet::Empty)
        };
    }
    let poly_expr = reduced_poly.to_expr(&mut simplifier.context);
    let zero = simplifier.context.num(0);
    let poly_eq = Equation {
        lhs: poly_expr,
        rhs: zero,
        op: cas_ast::RelOp::Eq,
    };
    let (reduced_sol, _) =
        crate::solver_entrypoints_solve::solve(&poly_eq, var, simplifier).ok()?;
    let candidates = match reduced_sol {
        SolutionSet::Discrete(roots) => roots,
        SolutionSet::Empty => return Some(SolutionSet::Empty),
        _ => return None,
    };

    // Keep candidates that exactly satisfy the ORIGINAL equation.
    let mut kept: Vec<ExprId> = Vec::new();
    for r in candidates {
        let rr = as_rational_const(&simplifier.context, r)?; // non-rational ⇒ decline (scope)
        let fr = f_poly.eval(&rr);
        let gr = g_poly.eval(&rr);
        // Domain: both radicands must be nonnegative for the real square roots to exist.
        if fr.is_negative() || gr.is_negative() {
            continue;
        }
        // Exact check of `s_f·√fr + s_g·√gr == c` with rational `c` and rational radicands.
        let holds = match (perfect_rational_sqrt(&fr), perfect_rational_sqrt(&gr)) {
            // Both radicands are perfect rational squares: compare the signed rational roots.
            (Some(sf), Some(sg)) => {
                let signed_f = if sign_f == 1 { sf } else { -sf };
                let signed_g = if sign_g == 1 { sg } else { -sg };
                signed_f + signed_g == c
            }
            // At least one radicand is an irrational surd: with a rational `c` the only way
            // `s_f·√fr + s_g·√gr = c` can hold is if the two surds CANCEL — a difference
            // (`s_f·s_g = −1`) of equal radicands with `c = 0` (e.g. `√(2x+3) − √(x+5) = 0`
            // at `x = 2`, where both sides equal `√7`). A rational `c ≠ 0` would force the
            // remaining surd to be rational, contradicting non-perfect-square.
            _ => sign_f * sign_g == -1 && c.is_zero() && fr == gr,
        };
        if holds {
            kept.push(r);
        }
    }
    if kept.is_empty() {
        Some(SolutionSet::Empty)
    } else {
        Some(SolutionSet::Discrete(kept))
    }
}

/// Solve a SINGLE radical equal to a polynomial, `√f = g` with `f` a polynomial of
/// degree ≥ 2 and `g` a polynomial in the variable (`√(5x²+9x−2) = 3x`). The
/// isolation core squares and then MIS-FILTERS several `√(quadratic) = c·x`
/// monomial-RHS forms: `√(5x²+9x−2) = 3x` returns a wrong "No solution" (true
/// `{1/4, 2}`) and `√(5x²+9x) = 3x` drops `9/4`. Square exactly to the polynomial
/// `f − g² = 0`, solve it, and keep each root `r` with `g(r) ≥ 0` (the only
/// extraneous-root filter: at a root `f(r) = g(r)² ≥ 0` already, and `√f(r) =
/// |g(r)| = g(r)` iff `g(r) ≥ 0`).
///
/// Gated to a degree-≥2 radicand so the common `√(linear) = …` forms keep their
/// existing (correct) isolation path (no huella churn), and to RATIONAL candidate
/// roots (a surd candidate declines — the exact-sign verification of a surd root is
/// a follow-up, matching the sum-of-radicals scope). `√f = √g` declines (the `g`
/// side is not a polynomial), leaving it to its own owner.
pub(super) fn try_solve_single_radical_equals_polynomial(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    use cas_math::numeric_eval::as_rational_const;
    use cas_math::polynomial::Polynomial;
    use cas_solver_core::isolation_utils::contains_var;
    use num_traits::Signed;

    if eq.op != RelOp::Eq {
        return None;
    }
    let (lhs, _) = simplifier.simplify(eq.lhs);
    let (rhs, _) = simplifier.simplify(eq.rhs);
    // Identify the `√f` side (a bare `Pow(radicand, 1/2)`) and the polynomial `g` side.
    // ALSO accept the MOVED form `c·√f + R(x) = 0` (rational c ≠ 0): the recursive
    // isolation emits the subtracted shape (`x − √(1−x²) = 0`, from `x/√(1−x²) = 1`
    // after the reciprocal power folds) which has no bare-√ side — reconstitute
    // `√f = −R/c` so the same square-and-verify machinery applies.
    let (radicand, g_expr) = if let Some(rad) = as_sqrt_radicand(&simplifier.context, lhs) {
        (rad, rhs)
    } else if let Some(rad) = as_sqrt_radicand(&simplifier.context, rhs) {
        (rad, lhs)
    } else if let Some((rad, g)) = reconstitute_moved_single_radical(simplifier, lhs, rhs, var) {
        (rad, g)
    } else {
        return None;
    };
    if !contains_var(&simplifier.context, radicand, var) {
        return None;
    }
    // `√f = √g` (both radicals) belongs to its own owner, not here.
    if as_sqrt_radicand(&simplifier.context, g_expr).is_some() {
        return None;
    }
    let f_poly = Polynomial::from_expr(&simplifier.context, radicand, var).ok()?;
    let g_poly = Polynomial::from_expr(&simplifier.context, g_expr, var).ok()?;
    // Degree-1 radicands (`√(x+1) = 2`) are handled correctly by the isolation path —
    // stay off them to avoid huella churn. This handler owns the degree-≥2 radicands
    // the isolation core mis-filters.
    if f_poly.degree() < 2 {
        return None;
    }

    // Square: `√f = g ⟹ f = g²`. Solve the polynomial `f − g² = 0`.
    let diff_poly = f_poly.sub(&g_poly.mul(&g_poly));
    if diff_poly.degree() == 0 {
        // A nonzero constant `f − g²` has no root (`√f = g` has no solution); the zero
        // polynomial is the `f ≡ g²` identity — a continuum we cannot enumerate.
        return if diff_poly.is_zero() {
            None
        } else {
            Some(SolutionSet::Empty)
        };
    }
    let poly_expr = diff_poly.to_expr(&mut simplifier.context);
    let zero = simplifier.context.num(0);
    let poly_eq = Equation {
        lhs: poly_expr,
        rhs: zero,
        op: RelOp::Eq,
    };
    let (sol, _) = crate::solver_entrypoints_solve::solve(&poly_eq, var, simplifier).ok()?;
    let candidates = match sol {
        SolutionSet::Discrete(roots) => roots,
        SolutionSet::Empty => return Some(SolutionSet::Empty),
        _ => return None,
    };

    // Keep each root where `g(r) ≥ 0` (the extraneous-root filter): a RATIONAL root
    // evaluates exactly; a SURD/transcendental root builds `g(r)` symbolically
    // (Horner over ExprIds) and decides through the exact const-sign cascade — an
    // UNDECIDABLE sign declines the whole relation (never guess). `f(r) ≥ 0` is
    // automatic at a root of `f − g²`.
    let mut kept: Vec<ExprId> = Vec::new();
    for r in candidates {
        let keep = if let Some(rr) = as_rational_const(&simplifier.context, r) {
            !g_poly.eval(&rr).is_negative()
        } else {
            let mut acc = simplifier.context.num(0);
            for c in g_poly.coeffs.iter().rev() {
                let c_expr = simplifier.context.add(Expr::Number(c.clone()));
                let mul = simplifier.context.add(Expr::Mul(acc, r));
                acc = simplifier.context.add(Expr::Add(mul, c_expr));
            }
            let g_at_r = simplifier.simplify(acc).0;
            let sign = as_rational_const(&simplifier.context, g_at_r)
                .map(|q| q.cmp(&num_rational::BigRational::from_integer(0.into())))
                .or_else(|| {
                    cas_math::root_forms::provable_sign_vs_zero(&simplifier.context, g_at_r)
                })
                .or_else(|| {
                    use cas_math::const_sign::{provable_const_sign, ConstSign};
                    Some(match provable_const_sign(&simplifier.context, g_at_r)? {
                        ConstSign::Negative => std::cmp::Ordering::Less,
                        ConstSign::Zero => std::cmp::Ordering::Equal,
                        ConstSign::Positive => std::cmp::Ordering::Greater,
                    })
                });
            match sign {
                Some(std::cmp::Ordering::Less) => false,
                Some(_) => true,
                None => return None, // undecidable sign: decline honestly
            }
        };
        if keep {
            kept.push(r);
        }
    }
    if kept.is_empty() {
        Some(SolutionSet::Empty)
    } else {
        Some(SolutionSet::Discrete(kept))
    }
}

/// `U(x)/√f {=} k` (or the canonical `U·f^(−1/2)` product) with rational `k ≠ 0`:
/// normalize to the bare-radical equation `√f = U/k` and delegate to the
/// square-and-verify owner. The Mul isolation otherwise moves the VAR-CARRYING
/// reciprocal power and emits the un-refolded `solve(x − 1/(1−x²)^(−1/2) = 0)`
/// self-referential echo (`x/√(1−x²) = 1`, `tan(arcsin(x)) = 1` after its fold).
pub(super) fn try_solve_poly_over_sqrt_equation(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;
    use num_traits::Zero;

    if eq.op != RelOp::Eq {
        return None;
    }
    // Normalize var side / const side.
    let (var_side, k) = if contains_var(&simplifier.context, eq.lhs, var)
        && !contains_var(&simplifier.context, eq.rhs, var)
    {
        (eq.lhs, eq.rhs)
    } else if contains_var(&simplifier.context, eq.rhs, var)
        && !contains_var(&simplifier.context, eq.lhs, var)
    {
        (eq.rhs, eq.lhs)
    } else {
        return None;
    };
    let k_val = as_rational_const(&simplifier.context, k)?;
    if k_val.is_zero() {
        return None; // U/√f = 0 ⟺ U = 0 within the domain — ordinary isolation
    }
    let (var_side, _) = simplifier.simplify(var_side);
    // Match `Div(U, √f)` or a flattened product with exactly one `f^(−1/2)` factor.
    let neg_half = |ctx: &Context, e: ExprId| -> Option<ExprId> {
        if let Expr::Pow(base, exp) = ctx.get(e) {
            if let Some(q) = as_rational_const(ctx, *exp) {
                if q == num_rational::BigRational::new((-1).into(), 2.into()) {
                    return Some(*base);
                }
            }
        }
        None
    };
    let (u_expr, radicand) = match simplifier.context.get(var_side).clone() {
        Expr::Div(num, den) => {
            let rad = as_sqrt_radicand(&simplifier.context, den)
                .or_else(|| neg_half(&simplifier.context, num).map(|_| den))?;
            // (only the bare `√f` denominator shape; anything else declines)
            let rad_ok = as_sqrt_radicand(&simplifier.context, den).is_some();
            if !rad_ok {
                return None;
            }
            (num, rad)
        }
        Expr::Mul(_, _) => {
            // Flatten factors; exactly one must be `f^(−1/2)`.
            fn flatten(ctx: &Context, e: ExprId, out: &mut Vec<ExprId>) {
                if let Expr::Mul(l, r) = ctx.get(e).clone() {
                    flatten(ctx, l, out);
                    flatten(ctx, r, out);
                } else {
                    out.push(e);
                }
            }
            let mut factors = Vec::new();
            flatten(&simplifier.context, var_side, &mut factors);
            let mut rad: Option<ExprId> = None;
            let mut rest: Vec<ExprId> = Vec::new();
            for f in factors {
                if let Some(base) = neg_half(&simplifier.context, f) {
                    if rad.is_some() {
                        return None; // two reciprocal radicals: out of scope
                    }
                    rad = Some(base);
                } else {
                    rest.push(f);
                }
            }
            let rad = rad?;
            let mut u: Option<ExprId> = None;
            for f in rest {
                u = Some(match u {
                    None => f,
                    Some(acc) => simplifier.context.add(Expr::Mul(acc, f)),
                });
            }
            (u?, rad)
        }
        _ => return None,
    };
    if !contains_var(&simplifier.context, radicand, var)
        || !contains_var(&simplifier.context, u_expr, var)
    {
        return None;
    }
    // `U/√f = k ⟺ √f = U/k` (√f > 0 on the domain, so no orientation flip).
    let k_node = simplifier.context.add(Expr::Number(k_val));
    let g = simplifier.context.add(Expr::Div(u_expr, k_node));
    let g = simplifier.simplify(g).0;
    let sqrt_f = {
        let half = simplifier.context.rational(1, 2);
        simplifier.context.add(Expr::Pow(radicand, half))
    };
    let reduced = Equation {
        lhs: sqrt_f,
        rhs: g,
        op: RelOp::Eq,
    };
    let set = try_solve_single_radical_equals_polynomial(simplifier, &reduced, var)?;
    Some(set)
}

/// Match the MOVED single-radical form `c·√f + R(x) {=} 0` (exactly ONE
/// sqrt-carrying additive term, rational `c ≠ 0`, `rhs = 0`) and reconstitute
/// `(radicand, g)` with `g = −R/c` so the square-and-verify owner applies.
fn reconstitute_moved_single_radical(
    simplifier: &mut Simplifier,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;
    use num_traits::Zero;

    if !as_rational_const(&simplifier.context, rhs)
        .map(|q| q.is_zero())
        .unwrap_or(false)
    {
        return None;
    }
    // Flatten additive terms with their signs.
    fn collect_terms(ctx: &Context, e: ExprId, positive: bool, out: &mut Vec<(ExprId, bool)>) {
        match ctx.get(e).clone() {
            Expr::Add(l, r) => {
                collect_terms(ctx, l, positive, out);
                collect_terms(ctx, r, positive, out);
            }
            Expr::Sub(l, r) => {
                collect_terms(ctx, l, positive, out);
                collect_terms(ctx, r, !positive, out);
            }
            Expr::Neg(inner) => collect_terms(ctx, inner, !positive, out),
            _ => out.push((e, positive)),
        }
    }
    let mut terms: Vec<(ExprId, bool)> = Vec::new();
    collect_terms(&simplifier.context, lhs, true, &mut terms);
    if terms.len() < 2 {
        return None;
    }
    // Exactly one term must carry the sqrt: `√f` or `q·√f` with rational q.
    let sqrt_coeff_of = |ctx: &Context, e: ExprId| -> Option<(ExprId, num_rational::BigRational)> {
        if let Some(rad) = as_sqrt_radicand(ctx, e) {
            return Some((rad, num_traits::One::one()));
        }
        if let Expr::Mul(l, r) = ctx.get(e).clone() {
            if let (Some(q), Some(rad)) = (as_rational_const(ctx, l), as_sqrt_radicand(ctx, r)) {
                return Some((rad, q));
            }
            if let (Some(q), Some(rad)) = (as_rational_const(ctx, r), as_sqrt_radicand(ctx, l)) {
                return Some((rad, q));
            }
        }
        None
    };
    let mut sqrt_term: Option<(usize, ExprId, num_rational::BigRational)> = None;
    for (i, (term, positive)) in terms.iter().enumerate() {
        if let Some((rad, q)) = sqrt_coeff_of(&simplifier.context, *term) {
            if sqrt_term.is_some() || q.is_zero() || !contains_var(&simplifier.context, rad, var) {
                return None; // two radicals (sum-of-radicals owner) / degenerate
            }
            let signed = if *positive { q } else { -q };
            sqrt_term = Some((i, rad, signed));
        }
    }
    let (idx, radicand, c) = sqrt_term?;
    // Rebuild R (everything else, with signs), then g = −R/c.
    let mut rest: Option<ExprId> = None;
    for (i, (term, positive)) in terms.iter().enumerate() {
        if i == idx {
            continue;
        }
        let t = if *positive {
            *term
        } else {
            simplifier.context.add(Expr::Neg(*term))
        };
        rest = Some(match rest {
            None => t,
            Some(acc) => simplifier.context.add(Expr::Add(acc, t)),
        });
    }
    let rest = rest?;
    let neg_c = simplifier.context.add(Expr::Number(-c));
    let g = simplifier.context.add(Expr::Div(rest, neg_c));
    let g = simplifier.simplify(g).0;
    Some((radicand, g))
}

/// Match a leaf `coeff · x^e` (any rational `e`, incl. negative), a `var`-free
/// constant `(0, value)`, or a reciprocal `c / x^e` `(−e, c)`. Returns
/// `(exponent, coeff)`; `None` for any other shape. Used by the reciprocal-root
/// solver to build the Laurent map `x^(p/q) → u^p`.
pub(super) fn x_root_laurent_leaf(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(num_rational::BigRational, num_rational::BigRational)> {
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;
    use num_rational::BigRational;
    use num_traits::{One, Zero};
    if !contains_var(ctx, expr, var) {
        return Some((BigRational::zero(), as_rational_const(ctx, expr)?));
    }
    match ctx.get(expr) {
        Expr::Variable(s) if ctx.sym_name(*s) == var => {
            Some((BigRational::one(), BigRational::one()))
        }
        Expr::Pow(base, exp) => {
            let (base, exp) = (*base, *exp);
            if !matches!(ctx.get(base), Expr::Variable(s) if ctx.sym_name(*s) == var) {
                return None;
            }
            Some((as_rational_const(ctx, exp)?, BigRational::one()))
        }
        Expr::Div(n, d) => {
            // `x^a / x^b = x^(a−b)` (`1/x^(1/3)` renders as `x^(2/3)/x`). Recurse on
            // BOTH sides so a monomial numerator is handled, not only a constant.
            let (n, d) = (*n, *d);
            let (ne, nc) = x_root_laurent_leaf(ctx, n, var)?;
            let (de, dc) = x_root_laurent_leaf(ctx, d, var)?;
            if dc.is_zero() {
                return None;
            }
            Some((ne - de, nc / dc))
        }
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            if !contains_var(ctx, l, var) {
                let c = as_rational_const(ctx, l)?;
                let (k, cc) = x_root_laurent_leaf(ctx, r, var)?;
                Some((k, c * cc))
            } else if !contains_var(ctx, r, var) {
                let c = as_rational_const(ctx, r)?;
                let (k, cc) = x_root_laurent_leaf(ctx, l, var)?;
                Some((k, c * cc))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Collect the `(exponent, coeff)` pairs of a sum/difference of `x`-power leaves
/// into `out`, tracking sign through `Add`/`Sub`/`Neg`. Returns `false` if any
/// leaf is not an `x`-power (so the caller declines).
pub(super) fn collect_x_root_laurent_pairs(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    positive: bool,
    out: &mut Vec<(num_rational::BigRational, num_rational::BigRational)>,
) -> bool {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            collect_x_root_laurent_pairs(ctx, l, var, positive, out)
                && collect_x_root_laurent_pairs(ctx, r, var, positive, out)
        }
        Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            collect_x_root_laurent_pairs(ctx, l, var, positive, out)
                && collect_x_root_laurent_pairs(ctx, r, var, !positive, out)
        }
        Expr::Neg(inner) => collect_x_root_laurent_pairs(ctx, *inner, var, !positive, out),
        _ => match x_root_laurent_leaf(ctx, expr, var) {
            Some((e, c)) => {
                out.push((e, if positive { c } else { -c }));
                true
            }
            None => false,
        },
    }
}

/// `trig(u) = trig(v)` (same `sin`/`cos` on both sides, or their sum/difference
/// vs 0) solved by the SUM-TO-PRODUCT identities: `sin u − sin v =
/// 2·cos((u+v)/2)·sin((u−v)/2)`, `cos u − cos v = −2·sin((u+v)/2)·sin((u−v)/2)`,
/// and the `+` variants. The product-zero equation then delegates each factor to
/// the periodic trig solver, whose union over a common period is exact. Without
/// this, a degree-≥3 multiple-angle expansion (`sin(3x) = sin(x)` →
/// `3·sin − 4·sin³`) is not quadratic-in-u and the generic isolation leaks the
/// self-referential `solve(x − arcsin(2·sin(x)³) = 0)`. Dispatched AFTER the
/// existing trig owners, so already-working shapes keep their presentation.
/// Exact m-th root of a non-negative rational, `None` unless both numerator
/// and denominator are perfect m-th powers (the f64 seed only guesses; the
/// EXACT `pow` round-trip decides — R2).
pub(super) fn exact_rational_mth_root(
    value: &num_rational::BigRational,
    m: u32,
) -> Option<num_rational::BigRational> {
    use num_traits::{Signed as _, ToPrimitive as _};
    if value.is_negative() {
        return None;
    }
    let root_of = |n: &num_bigint::BigInt| -> Option<num_bigint::BigInt> {
        let seed = n.to_f64()?.powf(1.0 / f64::from(m)).round() as i64;
        for candidate in seed.saturating_sub(1)..=seed.saturating_add(1) {
            if candidate >= 0 {
                let big = num_bigint::BigInt::from(candidate);
                if big.pow(m) == *n {
                    return Some(big);
                }
            }
        }
        None
    };
    let numer = root_of(value.numer())?;
    let denom = root_of(value.denom())?;
    Some(num_rational::BigRational::new(numer, denom))
}

/// True when `expr` is (or its top-level `Mul` contains) at least two square-root factors —
/// `√A·√B`, whether written `sqrt(_)` or `Pow(_, 1/2)`. This is the shape the simplifier merges to
/// `√(A·B)`, widening the real domain from `{A≥0 ∧ B≥0}` to `{A·B≥0}` and admitting extraneous roots
/// after squaring. Single radicals (handled by the existing range-condition machinery) return false.
pub(super) fn has_radical_product(ctx: &Context, expr: ExprId) -> bool {
    let is_even_root = |e: ExprId| -> bool {
        if cas_math::expr_extract::extract_sqrt_argument_view(ctx, e).is_some() {
            return true;
        }
        if let Expr::Pow(_, exp) = ctx.get(e) {
            if let Some(n) = cas_math::numeric_eval::as_rational_const(ctx, *exp) {
                use num_traits::Zero;
                return !n.is_integer() && (n.denom() % 2i32).is_zero();
            }
        }
        false
    };
    cas_math::expr_nary::mul_leaves(ctx, expr)
        .iter()
        .filter(|&&f| is_even_root(f))
        .count()
        >= 2
}

/// Solve an irreducible-polynomial INEQUALITY `p(x) {<,≤,>,≥} 0` by sign analysis over its already
/// computed real roots. The roots (closed-form, e.g. Cardano radicals or trig forms) are sorted
/// numerically; the polynomial's EXACT sign is sampled at a rational test point strictly inside each
/// interval they cut the real line into; and the satisfying intervals are unioned (open endpoints for
/// strict ops, closed for non-strict — the roots themselves satisfy `≤`/`≥`).
///
/// Returns `None` (falling back to the raw root set) unless the sign chart is fully consistent — the
/// signs alternate across every (simple) root and the unbounded ends match the leading coefficient's
/// end behaviour — so an incomplete or mis-ordered root set can never yield an unsound interval set.
/// Solve the EQUATION form `lhs = rhs` of an inequality and return its discrete real roots, or
/// `None` if it does not reduce to a finite real root set. Lets the inequality sign analysis run over
/// the equation's roots when the inequality path itself declined to `Empty`/`Residual`.
pub(super) fn polynomial_equation_real_roots(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<Vec<ExprId>> {
    let eq_form = Equation {
        lhs: eq.lhs,
        rhs: eq.rhs,
        op: cas_ast::RelOp::Eq,
    };
    let (set, _) = crate::solver_entrypoints_solve::solve(&eq_form, var, simplifier).ok()?;
    match set {
        SolutionSet::Discrete(roots) if !roots.is_empty() => Some(roots),
        _ => None,
    }
}

/// Factor a MONIC integer quartic `x⁴ + b·x³ + c·x² + d·x + e` into two monic integer quadratics
/// `(x² + p·x + q)(x² + r·x + s)`, if it factors over ℚ. By Gauss's lemma a monic integer polynomial
/// that factors over ℚ factors over ℤ, so the constant terms are an integer divisor pair `q·s = e`;
/// for each, `p = (d − q·b)/(s − q)` and `r = b − p` are forced, and the factorization is accepted
/// only when `p, r` are integers and the `x²`/`x³` coefficients match. Returns `None` for an
/// irreducible quartic (e.g. `x⁴ − x − 1`) or coefficients outside `i64`.
/// Exact integer square root of `n`: `Some(r)` with `r ≥ 0` iff `n = r²`, else `None` (negative or
/// non-perfect-square). No float in the keep/reject decision — the `f64` seed is only a starting point
/// that is corrected to the exact integer value before the `r*r == n` test.
pub(super) fn exact_i64_sqrt(n: i64) -> Option<i64> {
    if n < 0 {
        return None;
    }
    let mut r = (n as f64).sqrt() as i64;
    while r > 0 && r * r > n {
        r -= 1;
    }
    while (r + 1) * (r + 1) <= n {
        r += 1;
    }
    if r * r == n {
        Some(r)
    } else {
        None
    }
}

/// Build the REAL roots of `a·x³ + b·x² + c·x + d` (`a ≠ 0`), exactly, by Cardano's method. Normalize
/// to monic `x³ + Bx² + Cx + D`, depress via `x = t − B/3` to `t³ + p·t + q` (`p = C − B²/3`,
/// `q = 2B³/27 − BC/3 + D`), and branch on the depressed-cubic discriminant `Δ = (q/2)² + (p/3)³`:
///
/// * `Δ > 0` — ONE real root `x = ∛(−q/2 + √Δ) + ∛(−q/2 − √Δ) − B/3`. The cube root of the (negative)
///   second radicand is the engine's REAL odd-root.
/// * `Δ < 0` — the *casus irreducibilis*: THREE distinct real roots that cannot be written with real
///   radicals, so use the trigonometric form `x_k = 2√(−p/3)·cos(φ/3 − 2πk/3) − B/3` for `k = 0,1,2`,
///   where `φ = arccos( (3q)/(2p)·√(−3/p) )`. `Δ < 0 ⇒ p < 0`, so `−p/3` and `−3/p` are positive and
///   both square roots are real.
///
/// Returns `None` if `a = 0` or `Δ = 0` (a repeated root of an integer cubic is rational, hence already
/// peeled by the caller's rational-root deflation, so this branch is unreachable in practice).
pub(super) fn build_cubic_real_roots(
    simplifier: &mut Simplifier,
    a: &num_rational::BigRational,
    b: &num_rational::BigRational,
    c: &num_rational::BigRational,
    d: &num_rational::BigRational,
) -> Option<Vec<ExprId>> {
    use cas_solver_core::quadratic_formula::sqrt_expr;
    use num_rational::BigRational;
    use num_traits::Zero;
    let r = |n: i64| BigRational::from_integer(n.into());
    if a.is_zero() {
        return None;
    }
    let big_b = b / a;
    let big_c = c / a;
    let big_d = d / a;
    let b2 = &big_b * &big_b;
    let b3 = &b2 * &big_b;
    let p = &big_c - &b2 / r(3);
    let q = &b3 * r(2) / r(27) - &big_b * &big_c / r(3) + &big_d;
    let q_half = &q / r(2);
    let p_third = &p / r(3);
    let delta = &q_half * &q_half + &p_third * &p_third * &p_third;
    let b_over_3_val = &big_b / r(3);

    let num = |ctx: &mut cas_ast::Context, v: BigRational| ctx.add(Expr::Number(v));

    if delta > BigRational::zero() {
        // Single real root by radicals: ∛(−q/2 + √Δ) + ∛(−q/2 − √Δ) − B/3.
        let ctx = &mut simplifier.context;
        let delta_node = num(ctx, delta);
        let sqrt_delta = sqrt_expr(ctx, delta_node);
        let neg_q_half = num(ctx, -&q / r(2));
        let radicand_plus = ctx.add(Expr::Add(neg_q_half, sqrt_delta));
        let radicand_minus = ctx.add(Expr::Sub(neg_q_half, sqrt_delta));
        let one_third = num(ctx, BigRational::new(1.into(), 3.into()));
        let cbrt_plus = ctx.add(Expr::Pow(radicand_plus, one_third));
        let cbrt_minus = ctx.add(Expr::Pow(radicand_minus, one_third));
        let t = ctx.add(Expr::Add(cbrt_plus, cbrt_minus));
        let b_over_3 = num(ctx, b_over_3_val);
        let root = ctx.add(Expr::Sub(t, b_over_3));
        let (root, _) = simplifier.simplify(root);
        return Some(vec![root]);
    }
    if delta.is_zero() {
        return None; // repeated root ⇒ rational ⇒ already peeled by the caller.
    }

    // Casus irreducibilis (Δ < 0 ⇒ p < 0): three real roots in trigonometric form.
    // φ = arccos( (3q)/(2p) · √(−3/p) ),  x_k = 2√(−p/3)·cos(φ/3 − 2πk/3) − B/3.
    // Build all three (unsimplified) inside one `ctx` borrow, then simplify after it ends.
    let raw_roots: Vec<ExprId> = {
        let ctx = &mut simplifier.context;
        // m = 2·√(−p/3)
        let neg_p_third = num(ctx, -&p / r(3));
        let sqrt_neg_p_third = sqrt_expr(ctx, neg_p_third);
        let two = num(ctx, r(2));
        let m = ctx.add(Expr::Mul(two, sqrt_neg_p_third));
        // φ = arccos( coeff · √(−3/p) ),  coeff = (3q)/(2p)
        let coeff = num(ctx, &q * r(3) / (&p * r(2)));
        let neg_three_over_p = num(ctx, -r(3) / &p);
        let sqrt_neg_three_over_p = sqrt_expr(ctx, neg_three_over_p);
        let arccos_arg = ctx.add(Expr::Mul(coeff, sqrt_neg_three_over_p));
        let phi = ctx.call("arccos", vec![arccos_arg]);
        let one_third = num(ctx, BigRational::new(1.into(), 3.into()));
        let phi_third = ctx.add(Expr::Mul(one_third, phi));
        let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
        let b_over_3 = num(ctx, b_over_3_val);
        let mut rs = Vec::with_capacity(3);
        for k in 0..3i64 {
            // angle = φ/3 − (2k/3)·π   (k = 0 collapses to φ/3 in the simplifier)
            let shift_coeff = num(ctx, r(2 * k) / r(3));
            let shift = ctx.add(Expr::Mul(shift_coeff, pi));
            let angle = ctx.add(Expr::Sub(phi_third, shift));
            let cos_k = ctx.call("cos", vec![angle]);
            let scaled = ctx.add(Expr::Mul(m, cos_k));
            rs.push(ctx.add(Expr::Sub(scaled, b_over_3)));
        }
        rs
    };
    Some(
        raw_roots
            .into_iter()
            .map(|root| simplifier.simplify(root).0)
            .collect(),
    )
}
