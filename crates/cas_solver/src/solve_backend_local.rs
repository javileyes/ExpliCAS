//! Local active solve backend boundary.
//!
//! This backend is solver-owned and executes the solver-native runtime pipeline.
//! Keeping this indirection local lets us switch implementations without
//! changing call sites.

use crate::{CasError, Simplifier, SolveCtx, SolveStep};
use cas_ast::{substitute_expr_by_id, Constant, Context, Equation, Expr, ExprId, SolutionSet};
use cas_solver_core::domain_condition::ImplicitCondition;
use std::collections::HashMap;

use crate::solve_backend_contract::{CoreSolverOptions, SolveBackend};

/// True when the argument `c` of `arcsin`/`arccos` is PROVABLY outside `[-1, 1]`,
/// so the inverse-trig value is non-real and any candidate root containing it must
/// be dropped. EXACT: decides `|c| > 1` for any single quadratic surd `A + B·√n`
/// — covering rationals (`2`, `5/4`) AND surds (`√2`, `√3`, `√2/2`) — via the same
/// exact surd-sign logic as [`cas_math::root_forms::provable_sign_vs_zero`]
/// (`|c| > 1 ⟺ c − 1 > 0 ∨ c + 1 < 0`). A transcendental argument (`π`, `e`) or
/// anything `as_linear_surd` cannot reduce yields `false`, so a valid root is NEVER
/// dropped on an unproven bound (the boundary `|c| = 1`, `arcsin(±1) = ±π/2`, is
/// kept). Never uses f64 — a float gate could drop a root at `c = √2`.
/// Exact sign of the quadratic surd `a + b·√n` (`n ≥ 0`) versus zero. Never uses f64.
fn linear_surd_sign(
    a: &num_rational::BigRational,
    b: &num_rational::BigRational,
    n: &num_rational::BigRational,
) -> std::cmp::Ordering {
    use num_rational::BigRational;
    use num_traits::{Signed, Zero};
    let zero = BigRational::zero();
    if b.is_zero() || n.is_zero() {
        return a.cmp(&zero);
    }
    if a.is_zero() {
        return b.cmp(&zero); // b·√n, with √n > 0
    }
    let (sa, sb) = (a.cmp(&zero), b.cmp(&zero));
    if sa == sb {
        return sa; // same sign -> that sign
    }
    // Opposite signs: sign(a + b·√n) = sign(b) · sign(b²·n − a²).
    let inner = (b * b * n).cmp(&(a * a));
    if b.is_negative() {
        inner.reverse()
    } else {
        inner
    }
}

fn inv_trig_arg_provably_out_of_range(ctx: &Context, c: ExprId) -> bool {
    use cas_math::root_forms::as_linear_surd;
    use num_rational::BigRational;
    use std::cmp::Ordering;

    let Some((a, b, n)) = as_linear_surd(ctx, c) else {
        return false;
    };
    let one = BigRational::from_integer(1.into());
    linear_surd_sign(&(a.clone() - one.clone()), &b, &n) == Ordering::Greater
        || linear_surd_sign(&(a + one), &b, &n) == Ordering::Less
}

/// Position of a constant threshold `c` relative to the closed range `[-1, 1]` of `sin`/`cos`,
/// decided EXACTLY over a single quadratic surd (`A + B·√n`, covering rationals and surds). `None`
/// when `c ∈ (-1, 1)` or its position cannot be proven (transcendental / multi-surd) — those are
/// periodic and left to the residual path.
enum TrigThresholdRegion {
    AboveRange,
    BelowRange,
    AtUpperBound,
    AtLowerBound,
}

fn classify_trig_threshold(ctx: &Context, c: ExprId) -> Option<TrigThresholdRegion> {
    use cas_math::root_forms::as_linear_surd;
    use num_rational::BigRational;
    use std::cmp::Ordering;
    let (a, b, n) = as_linear_surd(ctx, c)?;
    let one = BigRational::from_integer(1.into());
    match linear_surd_sign(&(a.clone() - one.clone()), &b, &n) {
        Ordering::Greater => return Some(TrigThresholdRegion::AboveRange),
        Ordering::Equal => return Some(TrigThresholdRegion::AtUpperBound),
        Ordering::Less => {}
    }
    match linear_surd_sign(&(a + one), &b, &n) {
        Ordering::Less => Some(TrigThresholdRegion::BelowRange),
        Ordering::Equal => Some(TrigThresholdRegion::AtLowerBound),
        Ordering::Greater => None, // strictly inside (-1, 1): periodic, owned by the residual path
    }
}

/// True when `lhs` is a bare `sin(var)` or `cos(var)` (a single builtin call over exactly the solve
/// variable). `sin(2x)`, `2·sin(x)`, `tan(x)`, and compound arguments are rejected — they are not
/// range-bounded by `[-1, 1]` over a bare variable and stay with the periodic residual path.
fn bare_sin_or_cos_of_var(ctx: &Context, lhs: ExprId, var: &str) -> bool {
    use cas_ast::BuiltinFn;
    let Expr::Function(fn_id, args) = ctx.get(lhs) else {
        return false;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(BuiltinFn::Sin | BuiltinFn::Cos)
        )
    {
        return false;
    }
    matches!(ctx.get(args[0]), Expr::Variable(s) if ctx.sym_name(*s) == var)
}

/// Replace the result of a `sin(x)`/`cos(x)` inequality whose threshold is PROVABLY out of the
/// `[-1, 1]` range with the exact `ℝ` / `∅` answer (the generic monotonic inversion otherwise emits a
/// finite ray, sometimes with a non-real `arcsin(c)` endpoint). Only the unambiguous cases are
/// decided: a strictly out-of-range `c`, or the closed boundary (`c = 1` with `≤`/`>`, `c = -1` with
/// `≥`/`<`). The "touch" boundaries (`cos(x) < 1`, `cos(x) ≥ 1`, …) and `c ∈ (-1, 1)` exclude/include
/// only the periodic extremal points, which `ℝ`/`∅` cannot express, so they are left unchanged for
/// the residual path. Equations and non-bare-trig LHS are untouched.
fn intersect_inequality_with_trig_range(
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

/// True when `expr` is not a real value: it contains a non-finite / undefined
/// constant (∞ or undefined) anywhere, or an out-of-range inverse-trig term
/// (`arcsin(c)` / `arccos(c)` with `|c| > 1`, whose real domain is `[-1, 1]`).
/// Such a value is never a real solution of an equation over ℝ — e.g.
/// `solve(cos(x)=2, x)` must not report `{ arccos(2) }`, and `solve(sin(x)=√2, x)`
/// must not report `{ arcsin(√2) }`.
/// Drop discrete solutions that are PROVABLY non-real (the imaginary unit `i`, `√(negative)`, or an
/// even root of a negative — `(-1)^(1/2)`), used only in the RealOnly domain. An odd root of a
/// negative (`(-8)^(1/3) = -2`) is real and is kept. Non-discrete sets are real by construction.
fn drop_non_real_discrete_solutions(ctx: &Context, set: SolutionSet) -> SolutionSet {
    match set {
        SolutionSet::Discrete(xs) => {
            let kept: Vec<ExprId> = xs
                .into_iter()
                .filter(|&x| !cas_math::numeric_eval::expr_contains_imaginary(ctx, x))
                .collect();
            if kept.is_empty() {
                SolutionSet::Empty
            } else {
                SolutionSet::Discrete(kept)
            }
        }
        other => other,
    }
}

fn solution_contains_nonfinite(ctx: &Context, expr: ExprId) -> bool {
    use cas_ast::BuiltinFn;
    match ctx.get(expr) {
        Expr::Constant(Constant::Infinity | Constant::Undefined) => true,
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            solution_contains_nonfinite(ctx, *a) || solution_contains_nonfinite(ctx, *b)
        }
        Expr::Neg(a) | Expr::Hold(a) => solution_contains_nonfinite(ctx, *a),
        Expr::Function(fn_id, args) => {
            // arcsin/arccos of a constant PROVABLY outside [-1, 1] is non-real over ℝ.
            if args.len() == 1
                && (ctx.is_builtin(*fn_id, BuiltinFn::Arcsin)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Arccos)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Asin)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Acos))
                && inv_trig_arg_provably_out_of_range(ctx, args[0])
            {
                return true;
            }
            args.iter().any(|&c| solution_contains_nonfinite(ctx, c))
        }
        Expr::Matrix { data, .. } => data.iter().any(|&c| solution_contains_nonfinite(ctx, c)),
        _ => false,
    }
}

/// Classification of a candidate root by back-substitution into the original
/// equation, evaluated numerically over the reals.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RootCheck {
    /// Both sides evaluate to the same finite real — a genuine solution.
    Verified,
    /// Both sides evaluate to different finite reals — extraneous (drop it).
    Extraneous,
    /// Cannot decide numerically (symbolic/parametric/irrational/undefined) — keep.
    Unknown,
}

/// Back-substitute `root` for `var` in the original equation and check, over the
/// reals, whether the two sides agree. Used to reject extraneous roots that the
/// case-split solver returns without verification (e.g. `solve(|x| = x-1)`
/// returns `1/2`, but `|1/2| = 1/2 ≠ 1/2 - 1 = -1/2`).
///
/// CONSERVATIVE: only ever reports `Extraneous` for a small RATIONAL root with a
/// well-scaled numeric residual. Irrational roots (e.g. `500000 - 127·sqrt(...)`,
/// the small root of `x^2 - 1000000·x + 1`) evaluate via `f64` with catastrophic
/// cancellation, so back-substitution there is unreliable — those stay `Unknown`
/// (kept). Likewise large-magnitude values, where `f64` loses precision.
fn check_root(ctx: &Context, eq: &Equation, var: &str, root: ExprId) -> RootCheck {
    use cas_math::evaluator_f64::eval_f64;
    use num_traits::ToPrimitive;

    // Only a rational root is reliably checkable by float back-substitution.
    let Some(root_q) = cas_math::numeric_eval::as_rational_const(ctx, root) else {
        return RootCheck::Unknown;
    };
    let Some(rv) = root_q.to_f64() else {
        return RootCheck::Unknown;
    };
    // Keep large-magnitude roots conservatively: `f64` back-substitution there
    // (e.g. `(10^15)^2`) cancels catastrophically and could drop a valid root.
    if !rv.is_finite() || rv.abs() > 1e6 {
        return RootCheck::Unknown;
    }
    let mut map = HashMap::new();
    map.insert(var.to_string(), rv);
    match (eval_f64(ctx, eq.lhs, &map), eval_f64(ctx, eq.rhs, &map)) {
        (Some(l), Some(r)) if l.is_finite() && r.is_finite() && l.abs() < 1e9 && r.abs() < 1e9 => {
            let tol = 1e-9 * (1.0 + l.abs() + r.abs());
            if (l - r).abs() <= tol {
                RootCheck::Verified
            } else {
                RootCheck::Extraneous
            }
        }
        _ => RootCheck::Unknown,
    }
}

/// Filter the final real solution set: drop non-finite (∞ / undefined) entries
/// (`solve(3/x=0)` is not `{∞}`) and provably-EXTRANEOUS roots returned by an
/// unverified case-split (`solve(|x|=x-1)` is not `{1/2}`). A discrete set that
/// empties collapses to `Empty`. For a conditional whose every case is discrete
/// and fully classifiable, the verified roots are returned unconditionally
/// (back-substitution subsumes the branch guards); otherwise extraneous roots are
/// dropped in place and the structure is preserved.
fn filter_real_solutions(
    ctx: &mut Context,
    eq: &Equation,
    var: &str,
    set: SolutionSet,
    conds: &[ImplicitCondition],
) -> SolutionSet {
    match set {
        SolutionSet::Discrete(sols) => {
            let mut kept: Vec<ExprId> = Vec::new();
            for s in sols {
                if !solution_contains_nonfinite(ctx, s)
                    && check_root(ctx, eq, var, s) != RootCheck::Extraneous
                    && !root_violates_required_condition(ctx, var, s, conds)
                {
                    kept.push(s);
                }
            }
            if kept.is_empty() {
                SolutionSet::Empty
            } else {
                SolutionSet::Discrete(kept)
            }
        }
        SolutionSet::Conditional(cases) => {
            let fully_classifiable = cases.iter().all(|c| {
                if let SolutionSet::Discrete(roots) = &c.then.solutions {
                    roots.iter().all(|&r| {
                        !solution_contains_nonfinite(ctx, r)
                            && check_root(ctx, eq, var, r) != RootCheck::Unknown
                    })
                } else {
                    false
                }
            });
            if fully_classifiable {
                let mut verified: Vec<ExprId> = Vec::new();
                for c in &cases {
                    if let SolutionSet::Discrete(roots) = &c.then.solutions {
                        for &r in roots {
                            if check_root(ctx, eq, var, r) == RootCheck::Verified
                                && !verified.contains(&r)
                            {
                                verified.push(r);
                            }
                        }
                    }
                }
                if verified.is_empty() {
                    SolutionSet::Empty
                } else {
                    SolutionSet::Discrete(verified)
                }
            } else {
                let mut kept: Vec<_> = Vec::new();
                for mut case in cases {
                    case.then.solutions =
                        filter_real_solutions(ctx, eq, var, case.then.solutions.clone(), conds);
                    if !matches!(case.then.solutions, SolutionSet::Empty) {
                        kept.push(case);
                    }
                }
                if kept.is_empty() {
                    SolutionSet::Empty
                } else {
                    SolutionSet::Conditional(kept)
                }
            }
        }
        // `AllReals` means "every real satisfying the required conditions". When
        // those conditions are mutually contradictory the real domain is EMPTY,
        // so it is "No solution", not "All real numbers" — e.g.
        // `solve(ln(x)=ln(-x), x)` collapses to an identity but requires both
        // `x > 0` (from `ln(x)`) and `x < 0` (from `ln(-x)`).
        SolutionSet::AllReals if required_conditions_are_contradictory(ctx, conds) => {
            SolutionSet::Empty
        }
        other => other,
    }
}

/// True when the conjunction of `conds` is unsatisfiable, so an `AllReals`
/// result actually has an empty real domain. Detects the strict-sign
/// contradiction `e > 0 ∧ -e > 0` (the `ln(x)=ln(-x)` collapse): two `Positive`
/// conditions whose targets are negations of each other (`a == -b`). Conditions
/// are a conjunction, so any contradictory pair empties the domain.
fn required_conditions_are_contradictory(ctx: &Context, conds: &[ImplicitCondition]) -> bool {
    use cas_math::poly_compare::poly_negatively_proportional;

    for (i, c1) in conds.iter().enumerate() {
        let ImplicitCondition::Positive(a) = c1 else {
            continue;
        };
        for c2 in conds.iter().skip(i + 1) {
            let ImplicitCondition::Positive(b) = c2 else {
                continue;
            };
            // `a > 0` and `b > 0` cannot both hold when `a = λ·b` with `λ < 0`
            // (opposite signs everywhere). Covers exact negation `a == -b`
            // (`ln(-x)=ln(x)`) and any negative multiple such as `-8·x` vs `x`
            // (`log(2,-8x)=log(2,x)+k`).
            if poly_negatively_proportional(ctx, *a, *b) {
                return true;
            }
        }
    }
    false
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
fn root_violates_required_condition(
    ctx: &mut Context,
    var: &str,
    root: ExprId,
    conds: &[ImplicitCondition],
) -> bool {
    use cas_math::root_forms::{provable_sign_vs_zero, provable_sign_vs_zero_const_radicand};
    use std::cmp::Ordering;

    if conds.is_empty() {
        return false;
    }
    // Exact sign of a root vs 0: the rational-radicand prover first, then the transcendental-radicand
    // one (radicand `9 + 4e` etc.). Both are proofs, never float estimates, so a valid root is never
    // dropped — a `None` simply keeps the root.
    let sign_vs_zero = |ctx: &Context, at: ExprId| -> Option<Ordering> {
        provable_sign_vs_zero(ctx, at).or_else(|| provable_sign_vs_zero_const_radicand(ctx, at))
    };
    let var_id = ctx.var(var);
    for cond in conds {
        let violates = match cond {
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

/// True when the equation is `c/poly = 0` for a nonzero constant `c` — which has
/// NO real solution (a nonzero constant over anything is never zero; the points
/// where the denominator vanishes make it undefined, not zero). Detected by
/// simplifying `lhs - rhs` to a single fraction with a nonzero-constant numerator.
///
/// Short-circuiting this BEFORE the isolation logic avoids the solver dividing by
/// zero (`poly = c/0 = ∞`), which otherwise fabricates `{∞}` or a malformed nested
/// `solve(x = ∞ - x^2, x)` for denominators like `x^2 + x + 1`.
fn equation_is_nonzero_const_over_polynomial(simplifier: &mut Simplifier, eq: &Equation) -> bool {
    use num_traits::Zero;
    if eq.op != cas_ast::RelOp::Eq {
        return false;
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
fn equation_has_identically_zero_denominator(simplifier: &mut Simplifier, eq: &Equation) -> bool {
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

/// A monotonic function whose inequality result must be intersected with its real
/// argument-domain (which the inversion — square / exponentiate — drops).
#[derive(Clone, Copy)]
enum MonotonicFn {
    /// Even root `√(arg)` / `arg^(1/2k)`: range `[0, ∞)`, domain `{arg ≥ 0}`.
    EvenRoot,
    /// `ln(arg)` / `log(b, arg)`: range `ℝ`, domain `{arg > 0}`.
    Log,
}

/// Detect a monotonic `f(arg)` on the LHS, returning `(kind, arg)`. Covers the
/// `sqrt` builtin, an even-root `Pow` (`x^(1/2)`, `x^(1/4)`, …), `ln`, and the
/// two-argument `log(b, arg)` — and sees THROUGH a POSITIVE rational
/// multiplicative coefficient or divisor (`2·√x`, `√x/2`), which preserves both
/// the argument-domain and the `[0,∞)` even-root range, so the range correction
/// (keyed on the threshold sign) is unaffected. A NEGATIVE coefficient (flips the
/// range) and an ADDITIVE shift (`√x + 1`, shifts the range) are NOT matched and
/// stay honest residuals.
fn detect_monotonic_lhs(ctx: &Context, lhs: ExprId) -> Option<(MonotonicFn, ExprId)> {
    use cas_math::expr_extract::{
        extract_log_base_argument_view, extract_sqrt_argument_view, extract_unary_log_argument_view,
    };
    use num_traits::Signed;
    if let Some(arg) = extract_sqrt_argument_view(ctx, lhs) {
        return Some((MonotonicFn::EvenRoot, arg));
    }
    if let Expr::Pow(base, exp) = ctx.get(lhs) {
        let (base, exp) = (*base, *exp);
        if let Some(n) = cas_math::numeric_eval::as_rational_const(ctx, exp) {
            if cas_math::expr_predicates::is_even_root_exponent(&n) && n.is_positive() {
                return Some((MonotonicFn::EvenRoot, base));
            }
        }
    }
    if let Some(arg) = extract_unary_log_argument_view(ctx, lhs) {
        return Some((MonotonicFn::Log, arg));
    }
    if let Some((_base, arg)) = extract_log_base_argument_view(ctx, lhs) {
        return Some((MonotonicFn::Log, arg));
    }
    let is_pos = |e: ExprId| {
        cas_math::numeric_eval::as_rational_const(ctx, e).is_some_and(|c| c.is_positive())
    };
    match ctx.get(lhs) {
        // `(positive const)·f(arg)` or `f(arg)·(positive const)`.
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            if is_pos(l) {
                detect_monotonic_lhs(ctx, r)
            } else if is_pos(r) {
                detect_monotonic_lhs(ctx, l)
            } else {
                None
            }
        }
        // `f(arg) / (positive const)` (NOT `const / f(arg)`, a reciprocal).
        Expr::Div(num, den) => {
            let (num, den) = (*num, *den);
            if is_pos(den) {
                detect_monotonic_lhs(ctx, num)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Simplify the bound expressions of an interval solution set so a downstream
/// interval-validity comparison uses an EXACT numeric path rather than falling
/// back to structural ordering on unsimplified `Pow` bounds (e.g. `2^2`).
fn simplify_solution_bounds(simplifier: &mut Simplifier, set: SolutionSet) -> SolutionSet {
    fn simp_interval(simplifier: &mut Simplifier, i: cas_ast::Interval) -> cas_ast::Interval {
        let (min, _) = simplifier.simplify(i.min);
        let (max, _) = simplifier.simplify(i.max);
        cas_ast::Interval {
            min,
            min_type: i.min_type,
            max,
            max_type: i.max_type,
        }
    }
    match set {
        SolutionSet::Continuous(i) => SolutionSet::Continuous(simp_interval(simplifier, i)),
        SolutionSet::Union(v) => SolutionSet::Union(
            v.into_iter()
                .map(|i| simp_interval(simplifier, i))
                .collect(),
        ),
        other => other,
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
fn intersect_inequality_with_function_domain(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
    set: SolutionSet,
) -> SolutionSet {
    use cas_ast::{BoundType, Interval, RelOp};
    use cas_solver_core::solution_set::{intersect_solution_sets, pos_inf};
    use num_traits::Signed;

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
        if let Some(c) = cas_math::numeric_eval::as_rational_const(&simplifier.context, eq.rhs) {
            let (neg, pos) = (c.is_negative(), c.is_positive());
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

/// True if `expr` contains the variable named `var`.
fn expr_contains_named_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    cas_ast::collect_variables(ctx, expr)
        .iter()
        .any(|s| s == var)
}

/// Collect the rational exponents of every `x`-power in `expr` (bare `x` is
/// exponent 1), returning `false` if `x` ever appears in a DISALLOWED position:
/// inside a function, as the base of a non-rational/non-positive power, in a
/// denominator, mixed with another variable, or as a compound base. Constants and
/// `x`-free coefficients are fine. The collected exponents are only used to derive
/// the common denominator `q`; the rebuild handles the actual algebra.
fn collect_x_power_exponents(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    out: &mut Vec<num_rational::BigRational>,
) -> bool {
    use cas_math::numeric_eval::as_rational_const;
    use num_traits::{One, Signed};
    match ctx.get(expr) {
        Expr::Number(_) | Expr::Constant(_) => true,
        Expr::Variable(s) => {
            if ctx.sym_name(*s) == var {
                out.push(num_rational::BigRational::one());
                true
            } else {
                false // a different variable — not a univariate x-power polynomial
            }
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            collect_x_power_exponents(ctx, l, var, out)
                && collect_x_power_exponents(ctx, r, var, out)
        }
        Expr::Neg(inner) => {
            let inner = *inner;
            collect_x_power_exponents(ctx, inner, var, out)
        }
        Expr::Div(l, r) => {
            let (l, r) = (*l, *r);
            // `x` in a denominator would be a negative power (Laurent); out of scope.
            if expr_contains_named_var(ctx, r, var) {
                return false;
            }
            collect_x_power_exponents(ctx, l, var, out)
        }
        Expr::Pow(base, exp) => {
            let (base, exp) = (*base, *exp);
            let base_is_x = matches!(ctx.get(base), Expr::Variable(s) if ctx.sym_name(*s) == var);
            if base_is_x {
                let Some(e) = as_rational_const(ctx, exp) else {
                    return false; // x^(non-constant) e.g. x^x
                };
                if !e.is_positive() {
                    return false; // require a positive rational power
                }
                out.push(e);
                return true;
            }
            // Any other power: allowed only if entirely free of `x`.
            !expr_contains_named_var(ctx, base, var) && !expr_contains_named_var(ctx, exp, var)
        }
        // Functions (ln(x), sin(x), …), matrices, etc.: allowed only if `x`-free.
        _ => !expr_contains_named_var(ctx, expr, var),
    }
}

/// Rebuild `expr` with each `x`-power `x^e` replaced by `u^(q·e)` (bare `x` by
/// `u^q`) in the fresh variable `u_var`. Precondition (validated by
/// [`collect_x_power_exponents`]): every `q·e` is a positive integer, so the
/// result is a polynomial in `u`. `x`-free subtrees are returned unchanged.
fn rebuild_x_powers_as_u(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
    u_var: &str,
    q: &num_bigint::BigInt,
) -> ExprId {
    use cas_math::numeric_eval::as_rational_const;
    use num_rational::BigRational;
    if !expr_contains_named_var(ctx, expr, var) {
        return expr;
    }
    match ctx.get(expr).clone() {
        Expr::Variable(_) => {
            // Contains x and is a bare variable ⇒ it is x. x → u^q.
            let u = ctx.var(u_var);
            let qn = ctx.add(Expr::Number(BigRational::from(q.clone())));
            ctx.add(Expr::Pow(u, qn))
        }
        Expr::Pow(base, exp) => {
            let base_is_x = matches!(ctx.get(base), Expr::Variable(s) if ctx.sym_name(*s) == var);
            if base_is_x {
                let e = as_rational_const(ctx, exp).expect("validated rational x-exponent");
                let qe = BigRational::from(q.clone()) * e; // positive integer value
                let u = ctx.var(u_var);
                let en = ctx.add(Expr::Number(qe));
                return ctx.add(Expr::Pow(u, en));
            }
            let nb = rebuild_x_powers_as_u(ctx, base, var, u_var, q);
            let ne = rebuild_x_powers_as_u(ctx, exp, var, u_var, q);
            ctx.add(Expr::Pow(nb, ne))
        }
        Expr::Add(l, r) => {
            let nl = rebuild_x_powers_as_u(ctx, l, var, u_var, q);
            let nr = rebuild_x_powers_as_u(ctx, r, var, u_var, q);
            ctx.add(Expr::Add(nl, nr))
        }
        Expr::Sub(l, r) => {
            let nl = rebuild_x_powers_as_u(ctx, l, var, u_var, q);
            let nr = rebuild_x_powers_as_u(ctx, r, var, u_var, q);
            ctx.add(Expr::Sub(nl, nr))
        }
        Expr::Mul(l, r) => {
            let nl = rebuild_x_powers_as_u(ctx, l, var, u_var, q);
            let nr = rebuild_x_powers_as_u(ctx, r, var, u_var, q);
            ctx.add(Expr::Mul(nl, nr))
        }
        Expr::Div(l, r) => {
            let nl = rebuild_x_powers_as_u(ctx, l, var, u_var, q);
            let nr = rebuild_x_powers_as_u(ctx, r, var, u_var, q);
            ctx.add(Expr::Div(nl, nr))
        }
        Expr::Neg(inner) => {
            let ni = rebuild_x_powers_as_u(ctx, inner, var, u_var, q);
            ctx.add(Expr::Neg(ni))
        }
        _ => expr,
    }
}

/// True if `expr` contains any square-root term `Pow(_, 1/2)`.
fn expr_contains_sqrt(ctx: &Context, expr: ExprId) -> bool {
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

/// Flip a strict/non-strict inequality operator (for multiplying by −1).
fn flip_inequality(op: cas_ast::RelOp) -> cas_ast::RelOp {
    use cas_ast::RelOp;
    match op {
        RelOp::Lt => RelOp::Gt,
        RelOp::Gt => RelOp::Lt,
        RelOp::Leq => RelOp::Geq,
        RelOp::Geq => RelOp::Leq,
        other => other,
    }
}

/// Split `d` into a single square-root term `±√f` (radicand containing `var`) and
/// the remaining signed terms. Returns `(sign, f, rest_terms)` or None when there
/// is not exactly one such radical.
/// A signed additive term `(sign, expr)` in a decomposition.
type SignedTerm = (i8, ExprId);
/// `(radical_sign, radicand, remaining_signed_terms)` from [`collect_radical_split`].
type RadicalSplit = (i8, ExprId, Vec<SignedTerm>);

fn collect_radical_split(ctx: &Context, d: ExprId, var: &str) -> Option<RadicalSplit> {
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

/// Keep the roots `r` of `f = g²` for which `g(r) ≥ 0` — the genuine boundary `√f = g` points
/// (`√f = |g| = g` requires `g ≥ 0`). `g` is affine and each root a quadratic surd, so `g(r)` is a
/// quadratic surd whose sign `compare_values` decides exactly. Non-`Discrete` root sets (no isolated
/// roots, or the degenerate `f ≡ g²` case which only arises for perfect-square radicands the hook
/// never reaches) contribute no boundary points.
fn keep_roots_with_g_nonneg(
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

/// Solve a SIGN condition on `g` (`g > 0`, `g ≥ 0`, or `g < 0`, per `op` ∈ {Gt, Geq, Lt}).
/// When `g` is a rational CONSTANT the recursive solver errors (`solve(-4 < 0, x)` →
/// "variable not found"), so resolve it directly from the constant's sign: `AllReals` when the
/// relation holds, `Empty` otherwise. Non-constant `g` delegates to the recursive solver.
fn solve_g_sign_condition(
    simplifier: &mut Simplifier,
    var: &str,
    g: ExprId,
    op: cas_ast::RelOp,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    use cas_math::numeric_eval::as_rational_const;
    use num_rational::BigRational;
    use num_traits::Zero;

    if let Some(c) = as_rational_const(&simplifier.context, g) {
        let zero = BigRational::zero();
        let holds = match op {
            RelOp::Gt => c > zero,
            RelOp::Geq => c >= zero,
            RelOp::Lt => c < zero,
            _ => return None,
        };
        return Some(if holds {
            SolutionSet::AllReals
        } else {
            SolutionSet::Empty
        });
    }
    let zero = simplifier.context.num(0);
    solve_relation_set(simplifier, var, g, zero, op)
}

/// Solve `lhs {op} rhs` recursively and return just the solution set.
fn solve_relation_set(
    simplifier: &mut Simplifier,
    var: &str,
    lhs: ExprId,
    rhs: ExprId,
    op: cas_ast::RelOp,
) -> Option<SolutionSet> {
    let eq = Equation { lhs, rhs, op };
    crate::solver_entrypoints_solve::solve(&eq, var, simplifier)
        .ok()
        .map(|(s, _)| s)
}

/// Solve a radical INEQUALITY `√f {op} g` (a single square root vs a sqrt-free
/// side) by the correct case split — NOT by squaring blindly, which loses the
/// RHS-sign branches and gives wrong answers (`√x < x-2 → [0,1) ∪ (4,∞)` instead
/// of `(4,∞)`; `√(x-2) > 4-x → (3,6)` instead of `(3,∞)`):
///   √f < g   ⟺  f ≥ 0 ∧ g > 0 ∧ f < g²
///   √f ≤ g   ⟺  f ≥ 0 ∧ g ≥ 0 ∧ f ≤ g²
///   √f > g   ⟺  f ≥ 0 ∧ (g < 0 ∨ f > g²)
///   √f ≥ g   ⟺  f ≥ 0 ∧ (g < 0 ∨ f ≥ g²)
/// Each branch is a polynomial inequality the existing solver handles. Subsumes
/// the radicand-domain handling of `intersect_inequality_with_function_domain`.
/// Solve the polynomial sign relation `poly {op} 0`, handling a CONSTANT polynomial directly (the
/// recursive solver mishandles a constant relation in `x`).
fn solve_poly_sign(
    simplifier: &mut Simplifier,
    var: &str,
    poly: &cas_math::polynomial::Polynomial,
    op: cas_ast::RelOp,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    use num_rational::BigRational;
    if poly.is_zero() || poly.degree() == 0 {
        let k = poly
            .coeffs
            .first()
            .cloned()
            .unwrap_or_else(|| BigRational::from_integer(0.into()));
        let zero = BigRational::from_integer(0.into());
        let holds = match op {
            RelOp::Lt => k < zero,
            RelOp::Leq => k <= zero,
            RelOp::Gt => k > zero,
            RelOp::Geq => k >= zero,
            _ => return None,
        };
        return Some(if holds {
            SolutionSet::AllReals
        } else {
            SolutionSet::Empty
        });
    }
    let expr = poly.to_expr(&mut simplifier.context);
    let zero = simplifier.context.num(0);
    solve_relation_set(simplifier, var, expr, zero, op)
}

/// Express `e` as a ratio of polynomials `(num, den)` in `var`, combining sums/differences/products/
/// quotients/integer-powers over a common denominator WITHOUT cancelling shared factors (so every
/// genuine pole stays in `den`, which the caller's numeric verification relies on). The denominator is
/// the PRODUCT of the sub-denominators, not their lcm — this only raises the MULTIPLICITY of existing
/// poles (never introduces a new pole location, since each factor is a real denominator of `e`), and the
/// caller's `P/D {op} 0` sign analysis is invariant under multiplying both `P` and `D` by the same
/// factor, so the candidate stays exact. Returns `None` if any leaf is not a polynomial in `var` (a
/// fractional power `x^(1/2)`, a transcendental, …) so such inputs decline cleanly.
fn rational_function_of(
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
                    if magnitude > 8 {
                        return None; // bound degree growth
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

/// Split a rational `lhs` into numerator/denominator POLYNOMIALS in `var` WITHOUT cancelling shared
/// factors (see [`rational_function_of`]). Handles sums (`x + 1/x`), products, quotients, negations and
/// integer powers; declines (returns `None`) for any non-polynomial leaf.
fn split_rational_inequality_lhs(
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
fn try_solve_rational_constant_inequality(
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
    // The inequality solver is unreliable past degree 4 (it mis-solves `x⁵ > 1/32`); decline
    // higher-degree pieces rather than risk a wrong answer.
    if p_poly.degree() > 4 || den_poly.degree() > 4 {
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

/// Exact ordering of a rational `r` against the quadratic surd `a + b·√n` (`n ≥ 0`). Squares the
/// comparison `r − a {?} b·√n` with sign tracking so no float ever enters a keep/drop decision.
fn cmp_rational_to_quadratic_surd(
    r: &num_rational::BigRational,
    a: &num_rational::BigRational,
    b: &num_rational::BigRational,
    n: &num_rational::BigRational,
) -> std::cmp::Ordering {
    use num_traits::Zero;
    use std::cmp::Ordering;
    let zero = num_rational::BigRational::zero();
    let diff = r - a; // compare diff {?} b·√n
    if b.is_zero() || n.is_zero() {
        return diff.cmp(&zero);
    }
    // Reduce to a positive coefficient: for b < 0, `diff {?} b·√n  ⟺  reverse(−diff {?} −b·√n)`.
    let (d, bb, reversed) = if *b < zero {
        (-&diff, -b, true)
    } else {
        (diff.clone(), b.clone(), false)
    };
    // bb·√n ≥ 0: if d < 0 it is strictly smaller; otherwise compare the squares exactly.
    let ord = if d < zero {
        Ordering::Less
    } else {
        (&d * &d).cmp(&(&bb * &bb * n))
    };
    if reversed {
        ord.reverse()
    } else {
        ord
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
    // `r {?} bound`, exact. `None` if the bound is a higher surd we cannot order.
    fn cmp_to_bound(ctx: &Context, r: &BigRational, e: ExprId) -> Option<Ordering> {
        let (a, b, n) = bound_surd(ctx, e)?;
        Some(cmp_rational_to_quadratic_surd(r, &a, &b, &n))
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

fn try_solve_radical_inequality(
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

    let (s, f, rest) = collect_radical_split(&simplifier.context, d, var)?;
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
) -> Option<(ExprId, ExprId, num_rational::BigRational)> {
    use cas_math::numeric_eval::as_rational_const;
    use num_rational::BigRational;
    use num_traits::Zero;

    fn walk(
        ctx: &Context,
        expr: ExprId,
        sign: i8,
        var: &str,
        rads: &mut Vec<ExprId>,
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
                    // A radical must be +1·√(radicand) with the variable inside.
                    if sign == 1 && expr_contains_named_var(ctx, radicand, var) {
                        rads.push(radicand);
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

    let mut rads: Vec<ExprId> = Vec::new();
    let mut constant = BigRational::zero();
    if !walk(ctx, expr, 1, var, &mut rads, &mut constant) || rads.len() != 2 {
        return None;
    }
    Some((rads[0], rads[1], constant))
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
fn try_solve_sum_of_two_radicals_equation(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_math::numeric_eval::as_rational_const;
    use cas_math::polynomial::Polynomial;
    use num_rational::BigRational;
    use num_traits::Signed;

    if eq.op != cas_ast::RelOp::Eq {
        return None;
    }
    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (expr, _) = simplifier.simplify(diff);

    let (f, g, constant) = collect_two_sqrt_and_const(&simplifier.context, expr, var)?;
    // `√f + √g + constant = 0`  ⇒  `√f + √g = c` with `c = −constant`.
    let c = -constant;
    if c.is_negative() {
        return Some(SolutionSet::Empty); // a sum of square roots is never negative
    }

    // Radicands must be polynomials (to evaluate the verification exactly).
    let f_poly = Polynomial::from_expr(&simplifier.context, f, var).ok()?;
    let g_poly = Polynomial::from_expr(&simplifier.context, g, var).ok()?;

    // Reduced single-radical equation: √(f·g) = (c² − f − g)/2. Build the
    // radicand as the EXPANDED polynomial product (the single-radical solver
    // declines an un-expanded `√((x+1)(x-1))` but handles `√(x²-1)`).
    let fg = f_poly.mul(&g_poly).to_expr(&mut simplifier.context);
    let half = simplifier
        .context
        .add(Expr::Number(BigRational::new(1.into(), 2.into())));
    let sqrt_fg = simplifier.context.add(Expr::Pow(fg, half));
    let c2 = simplifier.context.add(Expr::Number(c.clone() * &c));
    let c2_minus_f = simplifier.context.add(Expr::Sub(c2, f));
    let c2_minus_f_minus_g = simplifier.context.add(Expr::Sub(c2_minus_f, g));
    let two = simplifier.context.num(2);
    let reduced_rhs_raw = simplifier.context.add(Expr::Div(c2_minus_f_minus_g, two));
    // Distribute the `/2` to the canonical polynomial form (`(9 - 2·x)/2 → 9/2 - x`)
    // via a Polynomial round-trip; the single-radical solver declines the
    // un-distributed `Div(poly, 2)` / `½·(…)` form but handles the affine form.
    let reduced_rhs = match Polynomial::from_expr(&simplifier.context, reduced_rhs_raw, var) {
        Ok(p) => p.to_expr(&mut simplifier.context),
        Err(_) => reduced_rhs_raw,
    };
    let reduced_eq = Equation {
        lhs: sqrt_fg,
        rhs: reduced_rhs,
        op: cas_ast::RelOp::Eq,
    };
    let (reduced_sol, _) =
        crate::solver_entrypoints_solve::solve(&reduced_eq, var, simplifier).ok()?;
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
        if let (Some(sf), Some(sg)) = (perfect_rational_sqrt(&fr), perfect_rational_sqrt(&gr)) {
            if sf + sg == c {
                kept.push(r);
            }
        }
    }
    if kept.is_empty() {
        Some(SolutionSet::Empty)
    } else {
        Some(SolutionSet::Discrete(kept))
    }
}

/// Shared core for "equation is a polynomial in an invertible atom `g(x)`": given
/// the equation already rewritten as `u_expr = 0` in the fresh variable `u_var`
/// (the atom replaced by `u`), require degree ≥ 2 in `u`, solve for `u`, then
/// back-substitute `g(x) = u_root` recursively for each root, letting the existing
/// solver apply the atom's own domain (even root drops negatives; `ln` stays
/// positive; etc.). Returns `None` if `u_expr` is not a degree-≥2 polynomial in
/// `u` or the `u`-equation is not discretely solvable.
///
/// The degree-≥2 gate is both correctness (a degree-1 `u`-equation is a single
/// `g(x) = c`, solved directly) and a recursion guard: the back-substitution is
/// itself a single `g(x) = u_root`, which must NOT re-enter this path.
fn solve_polynomial_in_atom(
    simplifier: &mut Simplifier,
    u_expr: ExprId,
    u_var: &str,
    var: &str,
    back_sub_atom: ExprId,
) -> Option<SolutionSet> {
    use cas_math::polynomial::Polynomial;
    let u_poly = Polynomial::from_expr(&simplifier.context, u_expr, u_var).ok()?;
    if u_poly.degree() < 2 {
        return None;
    }
    let zero = simplifier.context.num(0);
    let u_eq = Equation {
        lhs: u_expr,
        rhs: zero,
        op: cas_ast::RelOp::Eq,
    };
    let (u_solution, _) = crate::solver_entrypoints_solve::solve(&u_eq, u_var, simplifier).ok()?;
    let u_roots = match u_solution {
        SolutionSet::Discrete(roots) => roots,
        SolutionSet::Empty => return Some(SolutionSet::Empty),
        _ => return None, // non-discrete / unsolved u-polynomial: leave to the existing path
    };
    let mut solution = SolutionSet::Empty;
    for u_root in u_roots {
        let back_eq = Equation {
            lhs: back_sub_atom,
            rhs: u_root,
            op: cas_ast::RelOp::Eq,
        };
        let (xs, _) = crate::solver_entrypoints_solve::solve(&back_eq, var, simplifier).ok()?;
        solution =
            cas_solver_core::solution_set::union_solution_sets(&simplifier.context, solution, xs);
    }
    Some(solution)
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
fn try_solve_rational_power_polynomial(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
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
    solve_polynomial_in_atom(simplifier, u_expr, u_var, var, atom)
}

/// Solve an EQUATION that is a polynomial of degree ≥ 2 in `ln(g)` for a single
/// log atom `ln(g)` whose argument contains the variable (e.g.
/// `ln(x)^2 - ln(x) - 2 = 0`, a quadratic in `ln(x)`). Substitute `u = ln(g)`,
/// solve the polynomial in `u`, then back-substitute `ln(g) = u_root` — the
/// recursive solver finishes each as `g = e^(u_root)` with the `ln` domain
/// (`g > 0`). Without this, the isolation path reorients to `x = e^(√(…))` and
/// leaks a malformed `solve(...)` residual while dropping every root.
fn try_solve_polynomial_in_log(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    if eq.op != cas_ast::RelOp::Eq {
        return None;
    }
    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (expr, _) = simplifier.simplify(diff);

    // Find a `ln(arg)` subexpression whose argument contains the variable. If the
    // single substitution does not remove every `x`, the post-check below declines.
    let atom = find_log_atom_containing_var(&simplifier.context, expr, var)?;
    let u_var = "__lns_u";
    let u = simplifier.context.var(u_var);
    let u_expr = substitute_expr_by_id(&mut simplifier.context, expr, atom, u);
    if expr_contains_named_var(&simplifier.context, u_expr, var) {
        return None; // a second, distinct log atom (or x elsewhere) remains
    }
    solve_polynomial_in_atom(simplifier, u_expr, u_var, var, atom)
}

/// Return a `ln(arg)` subexpression of `expr` whose argument contains `var`
/// (the substitution atom for [`try_solve_polynomial_in_log`]), or None.
fn find_log_atom_containing_var(ctx: &Context, expr: ExprId, var: &str) -> Option<ExprId> {
    use cas_ast::BuiltinFn;
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if args.len() == 1
            && ctx.is_builtin(*fn_id, BuiltinFn::Ln)
            && expr_contains_named_var(ctx, args[0], var)
        {
            return Some(expr);
        }
    }
    match ctx.get(expr).clone() {
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            find_log_atom_containing_var(ctx, l, var)
                .or_else(|| find_log_atom_containing_var(ctx, r, var))
        }
        Expr::Neg(inner) | Expr::Hold(inner) => find_log_atom_containing_var(ctx, inner, var),
        Expr::Function(_, args) => args
            .iter()
            .find_map(|&a| find_log_atom_containing_var(ctx, a, var)),
        _ => None,
    }
}

/// Local backend facade selected as the active backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct LocalSolveBackend;

impl SolveBackend for LocalSolveBackend {
    fn solve_with_ctx_and_options(
        eq: &Equation,
        var: &str,
        simplifier: &mut Simplifier,
        opts: CoreSolverOptions,
        ctx: &SolveCtx,
    ) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
        let (set, steps) = solve_local_core(eq, var, simplifier, opts, ctx)?;
        // For a NON-STRICT inequality (`f ≤ 0` / `f ≥ 0`) EVERY real root of `f = lhs − rhs` is a
        // solution (`0` satisfies `≤ 0` and `≥ 0`), but the interval sign-analysis drops isolated
        // roots of even-multiplicity factors (`(x−2)²(x+1) ≤ 0` keeps `(−∞,−1]` but loses `{2}`;
        // `x²/(x−1) ≥ 0` keeps `(1,∞)` but loses `{0}`). Union those roots back in — they exclude
        // poles by construction (a pole is not a root of `f = 0`) and are domain-filtered.
        let set = union_non_strict_inequality_roots(eq, var, simplifier, opts, ctx, set);
        Ok((set, steps))
    }
}

/// True when `lhs` is `log(base, arg)` with the solve VARIABLE in the BASE — i.e. `logₓ(c)`, which is
/// non-monotonic in `x`. A constant-base `log(c, x)` (monotonic, solvable) returns false.
fn is_variable_base_log(ctx: &Context, lhs: ExprId, var: &str) -> bool {
    use cas_ast::BuiltinFn;
    let Expr::Function(fn_id, args) = ctx.get(lhs) else {
        return false;
    };
    args.len() == 2
        && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Log))
        && cas_ast::collect_variables(ctx, args[0]).contains(var)
}

/// Decline a `log(x, c) {op} k` inequality (the variable is the BASE) to an honest residual: `logₓ(c)
/// = ln(c)/ln(x)` is non-monotonic (decreasing on `x > 1`, sign change at `x = 1`), so the engine's
/// monotonic log isolation emits a WRONG ray (and a `1/0 → undefined` bound when `k = 0`). With no
/// exact split representation, the sound outcome is a residual, not a fabricated interval. Equations
/// and constant-base `log(c, x)` (monotonic, solvable) are untouched.
fn try_decline_variable_base_log_inequality(
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
        var,
    ))
}

/// True when `e` contains a `sin`/`cos`/`tan` whose ARGUMENT involves `var` (anywhere in the tree).
/// `sin(2)·x` (constant trig) is false; `sin(2x)`, `x − cos(x)` are true.
fn contains_trig_of_var(ctx: &Context, e: ExprId, var: &str) -> bool {
    use cas_ast::BuiltinFn;
    match ctx.get(e) {
        Expr::Function(fn_id, args) => {
            let (fn_id, args) = (*fn_id, args.clone());
            if matches!(
                ctx.builtin_of(fn_id),
                Some(BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Tan)
            ) && args
                .iter()
                .any(|&a| cas_ast::collect_variables(ctx, a).contains(var))
            {
                return true;
            }
            args.iter().any(|&a| contains_trig_of_var(ctx, a, var))
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            contains_trig_of_var(ctx, *l, var) || contains_trig_of_var(ctx, *r, var)
        }
        Expr::Neg(x) => contains_trig_of_var(ctx, *x, var),
        _ => false,
    }
}

/// Decline a PERIODIC trig inequality (`sin`/`cos`/`tan` of `var`) to an honest residual: its true
/// solution is an infinite PERIODIC UNION which the `SolutionSet` enum cannot represent, so the
/// monotonic inversion otherwise emits a single wrong ray. The bare `sin(x)`/`cos(x)` cases with a
/// threshold PROVABLY outside `[-1, 1]` are EXCLUDED — they are answered exactly (`ℝ`/`∅`) by the
/// trig-range guard after `solve_inner`, so they must not be pre-empted here. Equations are untouched
/// (op gate).
fn try_decline_periodic_trig_inequality(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    let ctx = &simplifier.context;
    if !contains_trig_of_var(ctx, eq.lhs, var) {
        return None;
    }
    // A bare sin/cos with an out-of-range / boundary threshold is solved exactly downstream — leave it.
    if bare_sin_or_cos_of_var(ctx, eq.lhs, var) && classify_trig_threshold(ctx, eq.rhs).is_some() {
        return None;
    }
    Some(cas_solver_core::solve_outcome::residual_solution_set(
        &mut simplifier.context,
        eq.lhs,
        eq.rhs,
        var,
    ))
}

/// Flatten `expr` into a linear combination of the exponential atom `base^x`:
/// accumulate the rational coefficient of every `base^x` (or `c*base^x`) term
/// into `atom_coeff`, collect the signed constant terms (no `var`) into
/// `const_terms`, and return `None` if any term is neither (a leftover
/// `base^(-x)`/higher power, or other `var` structure) — i.e. the expression is
/// not clean `A*base^x + B`.
fn collect_linear_exponential_atom_terms(
    ctx: &Context,
    expr: ExprId,
    atom: ExprId,
    var: &str,
    positive: bool,
    atom_coeff: &mut num_rational::BigRational,
    const_terms: &mut Vec<(bool, ExprId)>,
) -> Option<()> {
    use cas_ast::ordering::compare_expr;
    use cas_solver_core::isolation_utils::contains_var;
    use num_traits::One;
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            collect_linear_exponential_atom_terms(
                ctx,
                l,
                atom,
                var,
                positive,
                atom_coeff,
                const_terms,
            )?;
            collect_linear_exponential_atom_terms(
                ctx,
                r,
                atom,
                var,
                positive,
                atom_coeff,
                const_terms,
            )
        }
        Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            collect_linear_exponential_atom_terms(
                ctx,
                l,
                atom,
                var,
                positive,
                atom_coeff,
                const_terms,
            )?;
            collect_linear_exponential_atom_terms(
                ctx,
                r,
                atom,
                var,
                !positive,
                atom_coeff,
                const_terms,
            )
        }
        Expr::Neg(inner) => {
            let inner = *inner;
            collect_linear_exponential_atom_terms(
                ctx,
                inner,
                atom,
                var,
                !positive,
                atom_coeff,
                const_terms,
            )
        }
        _ => {
            if compare_expr(ctx, expr, atom) == std::cmp::Ordering::Equal {
                if positive {
                    *atom_coeff += num_rational::BigRational::one();
                } else {
                    *atom_coeff -= num_rational::BigRational::one();
                }
                return Some(());
            }
            let mul = if let Expr::Mul(l, r) = ctx.get(expr) {
                Some((*l, *r))
            } else {
                None
            };
            if let Some((l, r)) = mul {
                let coeff = if compare_expr(ctx, l, atom) == std::cmp::Ordering::Equal {
                    cas_math::numeric_eval::as_rational_const(ctx, r)
                } else if compare_expr(ctx, r, atom) == std::cmp::Ordering::Equal {
                    cas_math::numeric_eval::as_rational_const(ctx, l)
                } else {
                    None
                };
                if let Some(coeff) = coeff {
                    if positive {
                        *atom_coeff += coeff;
                    } else {
                        *atom_coeff -= coeff;
                    }
                    return Some(());
                }
            }
            if contains_var(ctx, expr, var) {
                None
            } else {
                const_terms.push((positive, expr));
                Some(())
            }
        }
    }
}

/// A degree-2 exponential inequality collapsed onto one side,
/// `A*base^(2x) + B*base^x {op} c` with NO `base^0` constant term (so `base^x`
/// factors out cleanly), is — since `base^x > 0` — equivalent to the single
/// exponential `base^x {op'} (-B/A)` (`op'` flips when `A < 0`). The single-side
/// terminal answers that even for a SYMBOLIC threshold (`e`, `pi`), where the
/// polynomial-in-u inequality solver rejects the symbolic coefficient. Fixes the
/// silent `e^(2x) - e*e^x < 0 -> {1}` (truth `(-inf,1)`) and the loud
/// `e^(2x) - pi*e^x < 0` "symbolic coefficient" error.
fn try_solve_factorable_exponential_inequality(
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

/// The coefficient of `var` in an AFFINE exponent: `Some(0)` for a constant,
/// `Some(k)` for `k*var + b`; `None` if the exponent is not affine in `var`.
fn exponent_linear_rate(
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

/// True if `expr` contains an exponential `base^(exponent)` (constant base, `var`
/// in the exponent) whose exponent has a NEGATIVE `var`-rate (a `base^(-x)` term)
/// or a non-affine exponent — i.e. not a clean non-negative-power polynomial in
/// `base^x`.
fn exponential_has_negative_rate(ctx: &Context, expr: ExprId, var: &str) -> bool {
    use cas_solver_core::isolation_utils::contains_var;
    use num_traits::Signed;
    match ctx.get(expr).clone() {
        Expr::Pow(base, exponent) => {
            if !contains_var(ctx, base, var) && contains_var(ctx, exponent, var) {
                match exponent_linear_rate(ctx, exponent, var) {
                    Some(rate) if rate.is_negative() => return true,
                    None => return true,
                    _ => {}
                }
            }
            exponential_has_negative_rate(ctx, base, var)
                || exponential_has_negative_rate(ctx, exponent, var)
        }
        Expr::Div(l, r) => {
            // A var-bearing DENOMINATOR is a negative power of an exponential
            // (`5/e^x`, which `expand(diff/base^x)` produces when the original had
            // a `base^0` constant). That is NOT a clean polynomial in base^x.
            contains_var(ctx, r, var)
                || exponential_has_negative_rate(ctx, l, var)
                || exponential_has_negative_rate(ctx, r, var)
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            exponential_has_negative_rate(ctx, l, var) || exponential_has_negative_rate(ctx, r, var)
        }
        Expr::Neg(inner) => exponential_has_negative_rate(ctx, inner, var),
        _ => false,
    }
}

/// The first exponential leaf `base^(exponent)` (constant base, `var` in the
/// exponent) found in `expr`.
fn find_first_exponential(ctx: &Context, expr: ExprId, var: &str) -> Option<ExprId> {
    use cas_solver_core::isolation_utils::contains_var;
    match ctx.get(expr).clone() {
        Expr::Pow(base, exponent) => {
            if !contains_var(ctx, base, var) && contains_var(ctx, exponent, var) {
                return Some(expr);
            }
            find_first_exponential(ctx, base, var)
                .or_else(|| find_first_exponential(ctx, exponent, var))
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            find_first_exponential(ctx, l, var).or_else(|| find_first_exponential(ctx, r, var))
        }
        Expr::Neg(inner) => find_first_exponential(ctx, inner, var),
        _ => None,
    }
}

/// Conservative EXACT proof that `expr` is strictly positive: `e`/`pi`, a
/// positive number, any `positive^anything` (a real power of a positive base),
/// and products/quotients of provably-positive parts. Used only to settle a
/// threshold's sign, never f64.
fn is_provably_positive(ctx: &Context, expr: ExprId) -> bool {
    use num_traits::Signed;
    match ctx.get(expr).clone() {
        Expr::Constant(cas_ast::Constant::E | cas_ast::Constant::Pi) => true,
        Expr::Pow(base, _) => is_provably_positive(ctx, base),
        Expr::Mul(l, r) | Expr::Div(l, r) => {
            is_provably_positive(ctx, l) && is_provably_positive(ctx, r)
        }
        _ => cas_math::numeric_eval::as_rational_const(ctx, expr).is_some_and(|v| v.is_positive()),
    }
}

/// EXACT proof that a threshold is `<= 0`: a non-positive rational, or `-p` with
/// `p` provably positive (e.g. `-e`, `-pi`).
fn threshold_provably_nonpositive(ctx: &Context, threshold: ExprId) -> bool {
    use num_traits::Signed;
    if cas_math::numeric_eval::as_rational_const(ctx, threshold).is_some_and(|v| !v.is_positive()) {
        return true;
    }
    match ctx.get(threshold).clone() {
        Expr::Neg(inner) => is_provably_positive(ctx, inner),
        _ => false,
    }
}

/// A single exponential with a NON-UNIT integer exponent, `a*base^(k*x) + c {op} m`
/// (k >= 2, base > 1, RHS constant). The unit-exponent terminal cannot isolate
/// `base^(k*x)`, so isolate it to `base^(k*x) {op'} threshold` and, since
/// `base^(k*x)` is strictly increasing, recover the monotone ray from the
/// boundary EQUATION `base^(k*x) = threshold` (which the equation solver handles:
/// `solve(e^(2x)=e) -> {1/2}`). This is what lets the degree-3 factor-out cofactor
/// `e^(2x) - e < 0` resolve to `(-inf, 1/2)` — and never rewrites `base^(k*x)` to
/// `(base^k)^x` (the simplifier renormalizes that back, re-entering forever).
fn try_solve_nonunit_exponential_inequality(
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
fn try_isolate_single_exponential_inequality(
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

/// The argument `g` of a bare `abs(g)`, else `None`.
fn match_abs_argument(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if args.len() == 1 && ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Abs) {
            return Some(args[0]);
        }
    }
    None
}

/// A concrete set the abs reduction can intersect/union; a `Residual`/`Conditional`
/// (e.g. a transcendental `g`) is not, so the guard declines on it.
fn is_concrete_solution_set(set: &SolutionSet) -> bool {
    matches!(
        set,
        SolutionSet::Continuous(_)
            | SolutionSet::Union(_)
            | SolutionSet::Empty
            | SolutionSet::AllReals
            | SolutionSet::Discrete(_)
    )
}

/// Solve `g {op} bound` and return the set only if it is concrete (so the abs
/// reduction never combines a residual).
fn solve_concrete_side(
    g: ExprId,
    bound: ExprId,
    op: cas_ast::RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: CoreSolverOptions,
    ctx: &SolveCtx,
) -> Option<SolutionSet> {
    let side = Equation {
        lhs: g,
        rhs: bound,
        op,
    };
    let (set, _) = solve_local_core(&side, var, simplifier, opts, ctx).ok()?;
    if is_concrete_solution_set(&set) {
        Some(set)
    } else {
        None
    }
}

/// `|g(x)| {op} c` with a CONSTANT `c` is reduced to the polynomial inequalities on
/// the two sides of the abs, which the engine already solves correctly — the abs
/// *split* otherwise drops the operator and returns the boundary equation
/// (`|x^2-2x| < 1` -> "No solution"; `<=` -> the boundary points only). For `c > 0`:
///   `|g| < c`  <=>  `g < c` AND `g > -c`      `|g| > c`  <=>  `g > c` OR `g < -c`
/// and the `c <= 0` degenerate cases resolve by sign (`|g| >= 0` always). Declines
/// (-> the existing abs/isolation paths) for a sum of abs, a non-constant RHS, a
/// symbolic `c`, or a `g` whose polynomial-inequality solve is not concrete.
fn try_solve_abs_threshold_inequality(
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

    // c > 0: reduce to the two-sided inequality / the outside union.
    let neg_c = simplifier.context.add(Expr::Number(-c_value));
    let result = match op {
        RelOp::Lt => {
            let upper = solve_concrete_side(g, c_expr, RelOp::Lt, var, simplifier, opts, ctx)?;
            let lower = solve_concrete_side(g, neg_c, RelOp::Gt, var, simplifier, opts, ctx)?;
            cas_solver_core::solution_set::intersect_solution_sets(
                &simplifier.context,
                upper,
                lower,
            )
        }
        RelOp::Leq => {
            let upper = solve_concrete_side(g, c_expr, RelOp::Leq, var, simplifier, opts, ctx)?;
            let lower = solve_concrete_side(g, neg_c, RelOp::Geq, var, simplifier, opts, ctx)?;
            cas_solver_core::solution_set::intersect_solution_sets(
                &simplifier.context,
                upper,
                lower,
            )
        }
        RelOp::Gt => {
            let upper = solve_concrete_side(g, c_expr, RelOp::Gt, var, simplifier, opts, ctx)?;
            let lower = solve_concrete_side(g, neg_c, RelOp::Lt, var, simplifier, opts, ctx)?;
            cas_solver_core::solution_set::union_solution_sets(&simplifier.context, upper, lower)
        }
        RelOp::Geq => {
            let upper = solve_concrete_side(g, c_expr, RelOp::Geq, var, simplifier, opts, ctx)?;
            let lower = solve_concrete_side(g, neg_c, RelOp::Leq, var, simplifier, opts, ctx)?;
            cas_solver_core::solution_set::union_solution_sets(&simplifier.context, upper, lower)
        }
        _ => return None,
    };
    Some(result)
}

/// `expr == ln(var)` (natural log of the bare solve variable)?
fn is_ln_of_var(ctx: &Context, expr: ExprId, var_id: ExprId) -> bool {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        return args.len() == 1
            && ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Ln)
            && args[0] == var_id;
    }
    false
}

/// `expr == ln(var)^2` -> the `ln(var)` node, else `None`.
fn as_ln_var_squared(ctx: &Context, expr: ExprId, var_id: ExprId) -> Option<ExprId> {
    use num_rational::BigRational;
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        let (base, exp) = (*base, *exp);
        let two = BigRational::from_integer(2.into());
        if cas_math::numeric_eval::as_rational_const(ctx, exp) == Some(two)
            && is_ln_of_var(ctx, base, var_id)
        {
            return Some(base);
        }
    }
    None
}

/// `expr == coeff · ln(var)^2` -> `(coeff, ln(var))`, else `None`.
fn match_ln_var_squared_with_coeff(
    ctx: &Context,
    expr: ExprId,
    var_id: ExprId,
) -> Option<(num_rational::BigRational, ExprId)> {
    use num_traits::One;
    if let Some(ln_expr) = as_ln_var_squared(ctx, expr, var_id) {
        return Some((num_rational::BigRational::one(), ln_expr));
    }
    if let Expr::Mul(a, b) = ctx.get(expr) {
        let (a, b) = (*a, *b);
        if let Some(ln_expr) = as_ln_var_squared(ctx, b, var_id) {
            if let Some(r) = cas_math::numeric_eval::as_rational_const(ctx, a) {
                return Some((r, ln_expr));
            }
        }
        if let Some(ln_expr) = as_ln_var_squared(ctx, a, var_id) {
            if let Some(r) = cas_math::numeric_eval::as_rational_const(ctx, b) {
                return Some((r, ln_expr));
            }
        }
    }
    None
}

/// `coeff · ln(x)^2 {op} c` (constant `c`) is non-monotonic in `x`, so the log-isolation
/// path collapses it to the boundary equation and reports "All real numbers if x > 0"
/// (`ln(x)^2 > 1` -> wrong; truth `(0, 1/e) U (e, ∞)`). Reduce to the two SINGLE-`ln`
/// inequalities, which the engine solves exactly: with `u = ln(x)`,
///   `u^2 > t` (t>0) <=> `u > √t` OR `u < -√t`      `u^2 < t` <=> `-√t < u < √t`,
/// and the single-`ln` solver carries the `x > 0` domain through `x = e^u`. `t <= 0`
/// resolves by sign on the domain `(0, ∞)`. Only fires for a bare `ln(x)` (natural log
/// of the solve variable); `log_b(x)^2` already routes correctly or honestly residuals.
fn try_solve_ln_square_inequality(
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
fn positive_linear_coeff_of_var(
    ctx: &Context,
    arg: ExprId,
    var_id: ExprId,
) -> Option<num_rational::BigRational> {
    use num_traits::{One, Signed};
    if arg == var_id {
        return Some(num_rational::BigRational::one());
    }
    if let Expr::Mul(l, r) = ctx.get(arg) {
        let (l, r) = (*l, *r);
        let factor = if r == var_id {
            cas_math::numeric_eval::as_rational_const(ctx, l)
        } else if l == var_id {
            cas_math::numeric_eval::as_rational_const(ctx, r)
        } else {
            None
        };
        if let Some(a) = factor {
            if a.is_positive() {
                return Some(a);
            }
        }
    }
    None
}

/// Decompose `expr == A·trig(arg) + B` where `trig` is a SINGLE bare `Sin`/`Cos`/`Tan` call containing
/// the variable, `A` (≠ 0) and `B` are rational constants, and every other additive part is var-free.
/// Returns `(trig_call, A, B)`, or `None` when `expr` is ALREADY the bare trig call (nothing to peel —
/// `detect` handles that directly) or when the side is not affine in exactly one trig term.
fn peel_affine_trig(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, num_rational::BigRational, num_rational::BigRational)> {
    use num_traits::Zero;
    let mut trig: Option<(ExprId, num_rational::BigRational)> = None;
    let mut offset = num_rational::BigRational::zero();
    accumulate_affine_trig(
        ctx,
        expr,
        &num_traits::One::one(),
        var,
        &mut trig,
        &mut offset,
    )?;
    let (call, a_coeff) = trig?;
    // Bare trig call (A = 1, no wrapper) or a vanishing coefficient: nothing for this rule to do.
    if a_coeff.is_zero() || call == expr {
        return None;
    }
    Some((call, a_coeff, offset))
}

/// Accumulate `expr` (scaled by `scale`) as `A·trig(arg) + B`: a constant leaf adds to `offset`, a
/// rational `Mul`/`Div` scales, `Add`/`Sub`/`Neg` recurse, and a single bare trig call of the variable
/// is recorded with its accumulated coefficient. A second trig term, a trig×trig product, or any other
/// var-bearing shape declines (`None`).
fn accumulate_affine_trig(
    ctx: &Context,
    expr: ExprId,
    scale: &num_rational::BigRational,
    var: &str,
    trig: &mut Option<(ExprId, num_rational::BigRational)>,
    offset: &mut num_rational::BigRational,
) -> Option<()> {
    use cas_ast::BuiltinFn;
    use cas_solver_core::isolation_utils::contains_var;
    use num_traits::Zero;
    if !contains_var(ctx, expr, var) {
        let c = cas_math::numeric_eval::as_rational_const(ctx, expr)?;
        *offset += scale * &c;
        return Some(());
    }
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            accumulate_affine_trig(ctx, inner, &(-scale.clone()), var, trig, offset)
        }
        Expr::Add(a, b) => {
            accumulate_affine_trig(ctx, a, scale, var, trig, offset)?;
            accumulate_affine_trig(ctx, b, scale, var, trig, offset)
        }
        Expr::Sub(a, b) => {
            accumulate_affine_trig(ctx, a, scale, var, trig, offset)?;
            accumulate_affine_trig(ctx, b, &(-scale.clone()), var, trig, offset)
        }
        Expr::Mul(a, b) => {
            if !contains_var(ctx, a, var) {
                let c = cas_math::numeric_eval::as_rational_const(ctx, a)?;
                accumulate_affine_trig(ctx, b, &(scale * &c), var, trig, offset)
            } else if !contains_var(ctx, b, var) {
                let c = cas_math::numeric_eval::as_rational_const(ctx, b)?;
                accumulate_affine_trig(ctx, a, &(scale * &c), var, trig, offset)
            } else {
                None
            }
        }
        Expr::Div(a, b) => {
            if contains_var(ctx, b, var) {
                return None;
            }
            let c = cas_math::numeric_eval::as_rational_const(ctx, b)?;
            if c.is_zero() {
                return None;
            }
            accumulate_affine_trig(ctx, a, &(scale / &c), var, trig, offset)
        }
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let is_trig = ctx
                .builtin_of(fn_id)
                .is_some_and(|b| matches!(b, BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Tan));
            if is_trig && contains_var(ctx, args[0], var) {
                if trig.is_some() {
                    return None; // more than one trig term: not affine in a single trig
                }
                *trig = Some((expr, scale.clone()));
                Some(())
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(crate) fn try_solve_periodic_trig_equation(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<SolutionSet> {
    use cas_ast::{BuiltinFn, RelOp};
    use cas_solver_core::isolation_utils::contains_var;
    use num_rational::BigRational;
    use num_traits::{One, Zero};

    if !matches!(eq.op, RelOp::Eq) {
        return None;
    }
    let (lhs, _) = simplifier.simplify(eq.lhs);
    let (rhs, _) = simplifier.simplify(eq.rhs);
    let var_id = simplifier.context.var(var);

    // `sin(arg)^2 = c`  <=>  `cos(2·arg) = 1 - 2c` ;  `cos(arg)^2 = c`  <=>  `cos(2·arg) = 2c - 1`.
    // Reduce a squared bare trig to the double-angle cosine equation and recurse: the cos branch's
    // c ∈ {0, ±1} gate then maps EXACTLY the single-family cases (`sin(x)^2=1 → cos(2x)=-1 →
    // {π/2 + kπ}`) and declines the two-family ones (`sin(x)^2=1/4 → cos(2x)=1/2`, not in {0,±1}).
    // Peels an optional leading rational coefficient `A` so `A·trig(arg)^2` is recognised (not just a
    // bare `trig(arg)^2`); the coefficient is folded into the constant side as `c/A` below. Without it
    // `4·cos(x)^2 = 1` skipped the reduction and emitted only the two base roots — no `+kπ` family.
    let squared = |ctx: &Context, e: ExprId| -> Option<(BuiltinFn, ExprId, BigRational)> {
        let (coeff, core) = match ctx.get(e) {
            Expr::Mul(l, r) => {
                let (l, r) = (*l, *r);
                if let Some(a) = cas_math::numeric_eval::as_rational_const(ctx, l) {
                    (a, r)
                } else if let Some(a) = cas_math::numeric_eval::as_rational_const(ctx, r) {
                    (a, l)
                } else {
                    (BigRational::one(), e)
                }
            }
            _ => (BigRational::one(), e),
        };
        if coeff.is_zero() {
            return None;
        }
        if let Expr::Pow(base, exp) = ctx.get(core) {
            let (base, exp) = (*base, *exp);
            let two = BigRational::from_integer(2.into());
            if cas_math::numeric_eval::as_rational_const(ctx, exp) == Some(two) {
                if let Expr::Function(fn_id, args) = ctx.get(base) {
                    if args.len() == 1 && contains_var(ctx, args[0], var) {
                        if let Some(b) = ctx.builtin_of(*fn_id) {
                            if matches!(b, BuiltinFn::Sin | BuiltinFn::Cos) {
                                return Some((b, args[0], coeff));
                            }
                        }
                    }
                }
            }
        }
        None
    };
    let squared_hit = if let Some((f, arg, a)) = squared(&simplifier.context, lhs) {
        (!contains_var(&simplifier.context, rhs, var)).then_some((f, arg, rhs, a))
    } else if let Some((f, arg, a)) = squared(&simplifier.context, rhs) {
        (!contains_var(&simplifier.context, lhs, var)).then_some((f, arg, lhs, a))
    } else {
        None
    };
    if let Some((sq_func, arg, c, a_coeff)) = squared_hit {
        let cv = cas_math::numeric_eval::as_rational_const(&simplifier.context, c)? / a_coeff;
        let two_c = &cv + &cv;
        let target = if matches!(sq_func, BuiltinFn::Sin) {
            BigRational::one() - two_c
        } else {
            two_c - BigRational::one()
        };
        let two = simplifier.context.num(2);
        let two_arg = simplifier.context.add(Expr::Mul(two, arg));
        let (two_arg, _) = simplifier.simplify(two_arg);
        let cos_call = simplifier.context.call("cos", vec![two_arg]);
        let target_expr = simplifier.context.add(Expr::Number(target));
        let reduced = Equation {
            lhs: cos_call,
            rhs: target_expr,
            op: RelOp::Eq,
        };
        return try_solve_periodic_trig_equation(&reduced, var, simplifier);
    }

    // `A·trig(a·x) + B = C` (A ≠ 0, B and C constant) -> `trig(a·x) = (C − B)/A`, then recurse. Without
    // this the outside coefficient/offset leaves the trig side a `Mul`/`Add` that `detect` cannot see,
    // so the bare fall-through emitted only the principal value — an INCOMPLETE solution set presented
    // as complete (e.g. `solve(2·sin x = 1)` -> `{π/6}` instead of `{π/6 + 2kπ, 5π/6 + 2kπ}`), unsound.
    {
        let lhs_has = contains_var(&simplifier.context, lhs, var);
        let rhs_has = contains_var(&simplifier.context, rhs, var);
        if lhs_has != rhs_has {
            let (var_side, const_side) = if lhs_has { (lhs, rhs) } else { (rhs, lhs) };
            if let Some((call, a_coeff, b_offset)) =
                peel_affine_trig(&simplifier.context, var_side, var)
            {
                let b_expr = simplifier.context.add(Expr::Number(b_offset));
                let diff = simplifier.context.add(Expr::Sub(const_side, b_expr));
                let a_expr = simplifier.context.add(Expr::Number(a_coeff));
                let reduced_rhs = simplifier.context.add(Expr::Div(diff, a_expr));
                let (reduced_rhs, _) = simplifier.simplify(reduced_rhs);
                let reduced = Equation {
                    lhs: call,
                    rhs: reduced_rhs,
                    op: RelOp::Eq,
                };
                return try_solve_periodic_trig_equation(&reduced, var, simplifier);
            }
        }
    }

    // `trig(a·x)` with a positive rational `a` -> `(func, a)`.
    let detect = |ctx: &Context, e: ExprId| -> Option<(BuiltinFn, BigRational)> {
        if let Expr::Function(fn_id, args) = ctx.get(e) {
            if args.len() == 1 {
                if let Some(b) = ctx.builtin_of(*fn_id) {
                    if matches!(b, BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Tan) {
                        let a = positive_linear_coeff_of_var(ctx, args[0], var_id)?;
                        return Some((b, a));
                    }
                }
            }
        }
        None
    };
    // `trig(a·x) = c` or `c = trig(a·x)`, with `c` constant.
    let (func, coeff, c) = if let Some((f, a)) = detect(&simplifier.context, lhs) {
        if contains_var(&simplifier.context, rhs, var) {
            return None;
        }
        (f, a, rhs)
    } else if let Some((f, a)) = detect(&simplifier.context, rhs) {
        if contains_var(&simplifier.context, lhs, var) {
            return None;
        }
        (f, a, lhs)
    } else {
        return None;
    };

    let pi = simplifier.context.add(Expr::Constant(Constant::Pi));
    let two = simplifier.context.num(2);
    let two_pi = simplifier.context.add(Expr::Mul(two, pi));

    // Representative root(s) for the bare argument `u = a·x`, and the shared period.
    let one = BigRational::one();
    let (bases_u, period_u): (Vec<ExprId>, ExprId) = match func {
        // tan(u)=c is a single family {arctan(c) + kπ} for EVERY constant c.
        BuiltinFn::Tan => {
            let at = simplifier.context.call("arctan", vec![c]);
            (vec![simplifier.simplify(at).0], pi)
        }
        BuiltinFn::Sin | BuiltinFn::Cos => {
            let cv = cas_math::numeric_eval::as_rational_const(&simplifier.context, c)?;
            let is_sin = matches!(func, BuiltinFn::Sin);
            let arc = if is_sin { "arcsin" } else { "arccos" };
            let arc_call = simplifier.context.call(arc, vec![c]);
            let (r1, _) = simplifier.simplify(arc_call);
            if cv.is_zero() {
                // sin(u)=0 → {kπ}; cos(u)=0 → {π/2 + kπ}. The two roots are π apart → ONE family, period π.
                (vec![r1], pi)
            } else if cv == one || cv == -one.clone() {
                // c = ±1: the two roots of the period coincide → ONE family, period 2π.
                (vec![r1], two_pi)
            } else if cv > -one.clone() && cv < one {
                // 0 < |c| < 1: TWO families in [0, 2π), shared period 2π.
                //   sin(u)=c → {arcsin(c) + 2kπ, π - arcsin(c) + 2kπ}
                //   cos(u)=c → {arccos(c) + 2kπ, 2π - arccos(c) + 2kπ}
                let second = if is_sin {
                    simplifier.context.add(Expr::Sub(pi, r1))
                } else {
                    simplifier.context.add(Expr::Sub(two_pi, r1))
                };
                (vec![r1, simplifier.simplify(second).0], two_pi)
            } else {
                // |c| > 1: no real solution; let the existing path report it.
                return None;
            }
        }
        _ => return None,
    };

    // `u = a·x` ⇒ `x = u/a`: divide every base and the period by `a` (a > 1 SHRINKS the period:
    // `cos(2x)=1 → {kπ}`). For `a = 1` this folds back to the bare family.
    let a_expr = simplifier.context.add(Expr::Number(coeff));
    let bases: Vec<ExprId> = bases_u
        .into_iter()
        .map(|b| {
            let d = simplifier.context.add(Expr::Div(b, a_expr));
            simplifier.simplify(d).0
        })
        .collect();
    let period_div = simplifier.context.add(Expr::Div(period_u, a_expr));
    let (period, _) = simplifier.simplify(period_div);
    Some(SolutionSet::Periodic { bases, period })
}

/// Solve a residual product equation `f1·f2·… = 0` whose factors are each a periodic trig equation
/// (`sin(x)·cos(x)=0`, or the `-2·sin(x/2)·sin(3x/2)` sum-to-product form of `cos(2x)-cos(x)=0`) by
/// solving every factor and UNIONING the periodic families over a common period. Returns `None` —
/// leaving the honest residual untouched — if any variable-bearing factor is not a bare periodic
/// trig equation (so non-trig products like `(x-1)·sin(x)=0` stay residual rather than half-solved).
fn try_union_periodic_trig_product(
    simplifier: &mut Simplifier,
    var: &str,
    product_expr: ExprId,
) -> Option<SolutionSet> {
    use cas_solver_core::isolation_utils::contains_var;

    let mut factors = Vec::new();
    collect_product_var_factors(&simplifier.context, product_expr, var, &mut factors);
    if factors.len() < 2 {
        return None;
    }

    let zero = simplifier.context.num(0);
    let mut families: Vec<(Vec<ExprId>, ExprId)> = Vec::with_capacity(factors.len());
    for f in factors {
        if !contains_var(&simplifier.context, f, var) {
            continue;
        }
        let eq = Equation {
            lhs: f,
            rhs: zero,
            op: cas_ast::RelOp::Eq,
        };
        match try_solve_periodic_trig_equation(&eq, var, simplifier) {
            Some(SolutionSet::Periodic { bases, period }) => families.push((bases, period)),
            _ => return None,
        }
    }
    if families.len() < 2 {
        return None;
    }
    union_periodic_families_over_common_period(simplifier, families)
}

/// Flatten a product into its variable-bearing factors, unwrapping `Neg`/`Mul` and dropping constant
/// factors. Each leaf factor that contains `var` is pushed onto `out`.
fn collect_product_var_factors(ctx: &Context, e: ExprId, var: &str, out: &mut Vec<ExprId>) {
    use cas_solver_core::isolation_utils::contains_var;
    match ctx.get(e) {
        Expr::Mul(a, b) => {
            collect_product_var_factors(ctx, *a, var, out);
            collect_product_var_factors(ctx, *b, var, out);
        }
        Expr::Neg(x) => collect_product_var_factors(ctx, *x, var, out),
        _ => {
            if contains_var(ctx, e, var) {
                out.push(e);
            }
        }
    }
}

/// Union periodic families `{baseᵢⱼ + k·periodᵢ}` over a COMMON period. Every period must be a
/// rational multiple of π; the common period is `lcm` of those rationals × π. Each family with
/// period `p` and common period `m·p` expands to `m` shifted copies (`base + t·p`, `t = 0..m`) of
/// each base; the merged bases are then deduplicated modulo the common period. Returns `None` if any
/// period is not a rational multiple of π.
fn union_periodic_families_over_common_period(
    simplifier: &mut Simplifier,
    families: Vec<(Vec<ExprId>, ExprId)>,
) -> Option<SolutionSet> {
    use num_bigint::BigInt;
    use num_integer::Integer;
    use num_rational::BigRational;
    use num_traits::{Signed, Zero};

    let mut qs: Vec<BigRational> = Vec::with_capacity(families.len());
    for (_, period) in &families {
        let q = period_as_rational_multiple_of_pi(simplifier, *period)?;
        if !q.is_positive() {
            return None;
        }
        qs.push(q);
    }
    let common = qs
        .iter()
        .cloned()
        .reduce(|a, b| BigRational::new(a.numer().lcm(b.numer()), a.denom().gcd(b.denom())))?;

    let pi = simplifier.context.add(Expr::Constant(Constant::Pi));
    let mut bases_out: Vec<ExprId> = Vec::new();
    for ((bases, period), q) in families.into_iter().zip(qs.into_iter()) {
        let ratio = &common / &q;
        if !ratio.is_integer() {
            return None;
        }
        let m = ratio.to_integer();
        let mut t = BigInt::zero();
        while t < m {
            let shift = if t.is_zero() {
                None
            } else {
                let tn = simplifier
                    .context
                    .add(Expr::Number(BigRational::from(t.clone())));
                let prod = simplifier.context.add(Expr::Mul(tn, period));
                Some(simplifier.simplify(prod).0)
            };
            for &b in &bases {
                let nb = match shift {
                    None => b,
                    Some(s) => {
                        let sum = simplifier.context.add(Expr::Add(b, s));
                        simplifier.simplify(sum).0
                    }
                };
                bases_out.push(nb);
            }
            t += 1;
        }
    }

    let cn = simplifier.context.add(Expr::Number(common));
    let period_expr = simplifier.context.add(Expr::Mul(cn, pi));
    let period_expr = simplifier.simplify(period_expr).0;

    dedup_bases_modulo_period(simplifier, &mut bases_out, period_expr);
    Some(SolutionSet::Periodic {
        bases: bases_out,
        period: period_expr,
    })
}

/// `period / π` as a positive rational, or `None` if `period` is not a rational multiple of π.
fn period_as_rational_multiple_of_pi(
    simplifier: &mut Simplifier,
    period: ExprId,
) -> Option<num_rational::BigRational> {
    let pi = simplifier.context.add(Expr::Constant(Constant::Pi));
    let ratio = simplifier.context.add(Expr::Div(period, pi));
    let (ratio, _) = simplifier.simplify(ratio);
    cas_math::numeric_eval::as_rational_const(&simplifier.context, ratio)
}

/// Deduplicate bases that are equal modulo `period` (i.e. `(b - b') / period` is an integer).
fn dedup_bases_modulo_period(simplifier: &mut Simplifier, bases: &mut Vec<ExprId>, period: ExprId) {
    let mut kept: Vec<ExprId> = Vec::new();
    for b in std::mem::take(bases) {
        let is_dup = kept.iter().any(|&k| {
            let diff = simplifier.context.add(Expr::Sub(b, k));
            let ratio = simplifier.context.add(Expr::Div(diff, period));
            let (ratio, _) = simplifier.simplify(ratio);
            cas_math::numeric_eval::as_rational_const(&simplifier.context, ratio)
                .is_some_and(|r| r.is_integer())
        });
        if !is_dup {
            kept.push(b);
        }
    }
    *bases = kept;
}

fn solve_local_core(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: CoreSolverOptions,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    // Bare trig equation `sin/cos/tan(x)=c` -> the full periodic family (before the unary-inverse
    // path, which would return only the principal root).
    if let Some(set) = try_solve_periodic_trig_equation(eq, var, simplifier) {
        return Ok((set, Vec::new()));
    }
    if equation_is_nonzero_const_over_polynomial(simplifier, eq)
        || equation_has_identically_zero_denominator(simplifier, eq)
    {
        return Ok((SolutionSet::Empty, Vec::new()));
    }
    // Absolute-value relations (`|x| + |x-1| < 5`, `|x| > x+1`, etc.) are
    // piecewise-linear: the isolate-one-abs strategy below loses terms or returns
    // the boundary point. Solve them exactly here, before any isolation routing.
    // Simplify the two sides first so a `√(perfect square)` collapses to its `|·|`
    // form (`√(x²-6x+9) → |x-3|`) and is recognized as an abs relation. Returns None
    // for anything that is not an abs relation, so other shapes fall through.
    let (abs_lhs, _) = simplifier.simplify(eq.lhs);
    let (abs_rhs, _) = simplifier.simplify(eq.rhs);
    // SOUNDNESS: a relation with an `undefined` side has NO real solution — nothing equals or compares
    // to `undefined`. In RealOnly, `ln(-2)`, `ln(-1)` simplify to `undefined`, so `ln(x) = ln(-2)` and
    // `x = ln(-1)` are unsatisfiable. Without this the isolation path emits a degenerate
    // `AllReals if undefined = 0` conditional (the guard `undefined = 0` is never true).
    if matches!(
        simplifier.context.get(abs_lhs),
        Expr::Constant(Constant::Undefined)
    ) || matches!(
        simplifier.context.get(abs_rhs),
        Expr::Constant(Constant::Undefined)
    ) {
        return Ok((SolutionSet::Empty, Vec::new()));
    }
    // `|g(x)| {op} c` (constant `c`) reduces to the polynomial inequalities on the
    // two sides of the abs; the isolation/split path below drops the operator and
    // returns the boundary equation (`|x^2-2x| < 1` -> "No solution"). Handle it
    // before the sum-of-abs and isolation routing.
    if let Some(set) = try_solve_abs_threshold_inequality(eq, var, simplifier, opts, ctx) {
        return Ok((set, Vec::new()));
    }
    // `ln(x)^2 {op} c` is non-monotonic; the log-isolation path reports "All reals if
    // x>0". Reduce to the two single-`ln` inequalities before that path runs.
    if let Some(set) = try_solve_ln_square_inequality(eq, var, simplifier, opts, ctx) {
        return Ok((set, Vec::new()));
    }
    if let Some(set) = cas_solver_core::solve_outcome::try_solve_sum_of_abs_relation(
        &mut simplifier.context,
        abs_lhs,
        abs_rhs,
        eq.op.clone(),
        var,
    ) {
        return Ok((set, Vec::new()));
    }
    // Equations that are a polynomial of degree ≥ 2 in `x^(1/q)` (`x - 3·√x + 2`,
    // `x^(2/3) - x^(1/3) - 2`, …) are quadratics-in-disguise: the isolation path
    // reorients to `x = f(x)` and leaks a malformed `solve(...)` residual while
    // dropping every root. Solve them by `u = x^(1/q)` substitution here first.
    if let Some(set) = try_solve_rational_power_polynomial(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // Equations that are a polynomial of degree ≥ 2 in `ln(x)`
    // (`ln(x)^2 - ln(x) - 2 = 0`, …) leak the same way; solve them by the
    // `u = ln(x)` substitution.
    if let Some(set) = try_solve_polynomial_in_log(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // A sum of two square roots equal to a constant (`√(x+3) + √x = 3`) leaks
    // the same isolation residual; reduce by squaring and verify exactly.
    if let Some(set) = try_solve_sum_of_two_radicals_equation(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // Radical INEQUALITIES `√f {<,≤,>,≥} g`: solve by the correct case split,
    // not by squaring blindly (which loses the RHS-sign branches and gives
    // wrong answers like `√x < x-2 → [0,1) ∪ (4,∞)`).
    if let Some(set) = try_solve_radical_inequality(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // `N / D {op} c` with a polynomial denominator (e.g. `1/(x²+1) < 1/2`, `1/x³ < 8`,
    // `5/x² > 1/4`): with `P = N − c·D`, solve `P {op} 0` where `D > 0` and `P {flip op} 0`
    // where `D < 0`, then NUMERICALLY verify the candidate before returning it (the general
    // division-sign-split path otherwise reciprocates without flipping, e.g. `1/x³ < 8 →
    // (-∞,1/2)`, wrong).
    if let Some(set) = try_solve_rational_constant_inequality(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // `log(x, c) {op} k` (the variable is the BASE) is non-monotonic; decline to an honest residual
    // rather than letting the generic monotonic isolation emit a wrong ray.
    if let Some(set) = try_decline_variable_base_log_inequality(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // A periodic `sin`/`cos`/`tan` inequality has a periodic-union solution the engine cannot
    // represent; decline to an honest residual instead of a wrong ray (out-of-range bare sin/cos are
    // excluded — they are answered ℝ/∅ by the trig-range guard after solve_inner).
    if let Some(set) = try_decline_periodic_trig_inequality(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // A single-exponential inequality `a*base^x + c {op} k` is isolated to the
    // pure `base^x {op'} (k-c)/a`, which the terminal answers for every base and
    // threshold (the strategy substitution would decline a fractional base).
    if let Some(set) = try_isolate_single_exponential_inequality(eq, var, simplifier, opts, ctx) {
        return Ok((set, Vec::new()));
    }
    // A single exponential with a non-unit integer exponent (`e^(2x) < e`, the
    // factor-out cofactor of a degree-3 inequality) is isolated to
    // `base^(k*x) {op} threshold` and answered from the boundary equation +
    // monotone ray (no `(base^k)^x` rewrite — the simplifier renormalizes it).
    if let Some(set) = try_solve_nonunit_exponential_inequality(eq, var, simplifier, opts, ctx) {
        return Ok((set, Vec::new()));
    }
    // A degree-2 exponential inequality collapsed to one side with no constant
    // term (`e^(2x) - e*e^x < 0`) factors out `base^x > 0` to a single
    // exponential, which the terminal solves even for a symbolic threshold —
    // unlike the polynomial-in-u solver, which rejects the symbolic coefficient.
    if let Some(set) = try_solve_factorable_exponential_inequality(eq, var, simplifier, opts, ctx) {
        return Ok((set, Vec::new()));
    }
    let (set, steps) = crate::solve_core_runtime::solve_inner(eq, var, simplifier, opts, ctx)?;
    // A product of periodic trig factors (`sin(x)·cos(x)=0`, or `cos(2x)-cos(x)=0` after
    // sum-to-product) comes back as a residual product: the zero-product path declines because a
    // factor solves to an infinite `Periodic` family it cannot merge with an immutable context.
    // Union the per-factor periodic families over a common period here (mutable context available),
    // so all branches and their periodicity are emitted instead of a wrong finite set.
    if let SolutionSet::Residual(product) = &set {
        if let Some(unioned) = try_union_periodic_trig_product(simplifier, var, *product) {
            return Ok((unioned, steps));
        }
    }
    let conds = ctx.required_conditions();
    let set = filter_real_solutions(&mut simplifier.context, eq, var, set, &conds);
    // SOUNDNESS (RealOnly): drop a discrete solution that is provably NON-REAL — it carries the
    // imaginary unit `i`, `√(negative)`, or an EVEN root of a negative (`(-1)^(1/2)`). The inversion
    // of `ln`/`exp` does not re-check reality, so `solve(ln(x)=√(-1)) → {e^((-1)^(1/2))}` (= e^i) and
    // `solve(x=i) → {i}` slipped through; in the reals they have no solution. ODD roots of negatives
    // (`(-8)^(1/3) = -2`) stay REAL and are NOT dropped.
    let set = if opts.value_domain.is_real_only() {
        drop_non_real_discrete_solutions(&simplifier.context, set)
    } else {
        set
    };
    // Fold the monotonic-function argument-domain into an inequality result
    // (`sqrt(x)<2 → [0,4)`), which the inversion drops; no-op for equations.
    let set = intersect_inequality_with_function_domain(simplifier, eq, var, set);
    // A `sin(x)`/`cos(x)` inequality with a threshold provably outside [-1, 1] is ℝ or ∅, not the
    // finite ray (possibly with a non-real `arcsin(c)` endpoint) the generic inversion emits. In-range
    // / touch-boundary cases are periodic and left to the residual path; no-op for equations.
    let set = intersect_inequality_with_trig_range(&simplifier.context, eq, var, set);
    // Intersect with the implicit real domain of the WHOLE LHS, so a domain-restricted function
    // appearing as a FACTOR (not the bare LHS) still excludes its undefined region
    // (`ln(x)·(x−2)² ≤ 0` must be `(0,1]∪{2}`, NOT `(−∞,1]∪{2}` — `ln` is undefined for `x ≤ 0`).
    let set = intersect_inequality_with_expression_domain(simplifier, eq, var, set);
    // An irreducible cubic factor with a SINGLE real root (Cardano discriminant Δ > 0) is otherwise
    // either leaked as an honest `Residual`/`Conditional` (standalone `x³+x²+3 = 0`) or silently
    // dropped after its sibling rational roots are peeled (`x⁴+x³+3x → {0}` loses the root of
    // `x³+x²+3`). `try_solve_polynomial_with_cubic_factor` returns the FULL real set — the peeled
    // rational roots PLUS the cubic's radical root — which subsumes whatever the normal solve produced
    // for such a `(rational linear factors)·(irreducible Δ>0 cubic)` polynomial. So REPLACE rather
    // than union: unioning re-introduces the rational roots `complete` already carries (`{0, 0, …}`),
    // and a cubic the normal path already solved cleanly (`x³-2 → {2^(1/3)}`) is reproduced identically
    // by Cardano. Δ ≤ 0 cubics and non-cubic quotients decline, leaving any other result untouched.
    let set = match try_solve_polynomial_with_cubic_factor(simplifier, eq, var) {
        Some(complete)
            if matches!(
                set,
                SolutionSet::Residual(_) | SolutionSet::Conditional(_) | SolutionSet::Discrete(_)
            ) =>
        {
            complete
        }
        _ => set,
    };

    // A BIQUADRATIC `a·x⁴ + b·x² + c` whose `x`-roots are surds (`x⁴-8x²+15 → {±√3, ±√5}`) otherwise
    // leaks a circular residual `solve(x − (8x²−15)^(1/4)=0)`. Solve it by the `z = x²` substitution.
    let set = if matches!(set, SolutionSet::Residual(_) | SolutionSet::Conditional(_)) {
        try_solve_biquadratic(simplifier, eq, var).unwrap_or(set)
    } else {
        set
    };

    // A polynomial whose deflated quartic factor splits into two rational quadratics
    // (`x⁵-5x³+x²-5 = (x+1)(x²-5)(x²-x+1)` drops the `±√5` roots): peel the rational roots and solve
    // the quadratic factors. Replaces a `Residual`/`Conditional`; augments a `Discrete` the normal
    // path left incomplete (only the rational roots) when the quartic factor adds genuinely new roots.
    let set = match try_solve_polynomial_with_quartic_factor(simplifier, eq, var) {
        Some(complete) => match (&set, &complete) {
            (SolutionSet::Residual(_) | SolutionSet::Conditional(_), _) => complete,
            (SolutionSet::Discrete(current), SolutionSet::Discrete(c))
                if c.len() > current.len() =>
            {
                complete
            }
            _ => set,
        },
        None => set,
    };

    // An absolute-value equation `|arg| = c` with a quadratic argument carrying a linear term
    // (`|x²-2x| = 3`) leaks a circular residual from the recursive isolation. Split `arg = ±c` and
    // solve each as a full equation instead.
    let set = if matches!(set, SolutionSet::Residual(_) | SolutionSet::Conditional(_)) {
        try_solve_abs_equality(simplifier, eq, var).unwrap_or(set)
    } else {
        set
    };

    // An IRREDUCIBLE polynomial inequality (`x³+x+1 > 0`, `x³-3x+1 > 0`) is rewritten to `Equal(p,0)`
    // by the normal path, dropping the operator and returning the equation's root SET (so `> 0` and
    // `< 0` give identical output). When the operator is an inequality and the result is a `Discrete`
    // root set, recover the interval solution by sign analysis over those (now closed-form) real roots.
    let set = if matches!(
        eq.op,
        cas_ast::RelOp::Lt | cas_ast::RelOp::Leq | cas_ast::RelOp::Gt | cas_ast::RelOp::Geq
    ) {
        if let SolutionSet::Discrete(roots) = &set {
            let roots = roots.clone();
            try_polynomial_inequality_sign_analysis(simplifier, eq, var, &roots).unwrap_or(set)
        } else {
            set
        }
    } else {
        set
    };

    // A PARAMETRIC linear equation whose coefficient cancelled (`a·x = a → {1}`) dropped the `a ≠ 0`
    // guard and the `a = 0 ⇒ ℝ` branch. Recover them when the result is a single numeric root.
    let set = if let SolutionSet::Discrete(roots) = &set {
        if roots.len() == 1 {
            let root = roots[0];
            try_parametric_linear_degenerate_branch(simplifier, eq, var, root).unwrap_or(set)
        } else {
            set
        }
    } else {
        set
    };
    Ok((set, steps))
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
fn try_polynomial_inequality_sign_analysis(
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

/// Solve a BIQUADRATIC equation `a·x⁴ + b·x² + c = 0` (no odd-degree terms) by the substitution
/// `z = x²`: solve the quadratic `a·z² + b·z + c = 0`, then for each NON-NEGATIVE real `z` root take
/// `x = ±√z`. The normal path only handles biquadratics whose `x`-roots are rational; when they are
/// surds (`x⁴-8x²+15 → {±√3, ±√5}`, `x⁴-2x²-3 → {±√3}`) it leaks a circular residual. Every produced
/// root is verified by numeric back-substitution, so a wrong `z ≥ 0` decision (or a missed root) can
/// never emit an unsound value. Returns `None` for a non-biquadratic quartic or `Empty` when no real
/// root exists (`z` roots both negative or complex).
fn try_solve_biquadratic(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_math::polynomial::Polynomial;
    use cas_solver_core::quadratic_formula::sqrt_expr;
    use num_rational::BigRational;
    use num_traits::{ToPrimitive, Zero};
    use std::collections::HashMap;

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

    // Quadratic `a·z² + b·z + c` in `z = x²`: discriminant and its exact √.
    let r = |n: i64| BigRational::from_integer(n.into());
    let disc = &b * &b - &a * &c * r(4);
    let (af, bf, cf) = (a.to_f64()?, b.to_f64()?, c.to_f64()?);
    let disc_f = bf * bf - 4.0 * af * cf;
    if disc_f < 0.0 {
        // Complex z roots ⇒ no real x.
        return Some(SolutionSet::Empty);
    }

    // Build the exact `z = (−b ± √disc)/(2a)`, then `x = ±√z` for each non-negative z root.
    let ctx = &mut simplifier.context;
    let num = |ctx: &mut cas_ast::Context, v: BigRational| ctx.add(Expr::Number(v));
    let disc_node = num(ctx, disc);
    let sqrt_disc = sqrt_expr(ctx, disc_node);
    let neg_b = num(ctx, -&b);
    let two_a = num(ctx, &a * r(2));
    let mut raw_roots: Vec<ExprId> = Vec::new();
    for s in [1.0f64, -1.0f64] {
        let z_f = (-bf + s * disc_f.sqrt()) / (2.0 * af);
        if z_f < -1e-12 {
            continue; // z < 0 ⇒ x² = z has no real solution
        }
        let signed = if s > 0.0 {
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

    // Simplify, verify each candidate by numeric back-substitution, and dedup by numeric value.
    let mut roots: Vec<ExprId> = Vec::new();
    let mut seen: Vec<f64> = Vec::new();
    for raw in raw_roots {
        let (root, _) = simplifier.simplify(raw);
        let xv = match cas_math::evaluator_f64::eval_f64(&simplifier.context, root, &HashMap::new())
        {
            Some(v) if v.is_finite() => v,
            _ => continue,
        };
        // p(xv) ≈ 0 (scaled by the coefficient magnitude).
        let residual = af * xv.powi(4) + bf * xv * xv + cf;
        let scale = 1.0 + af.abs() * xv.powi(4).abs() + bf.abs() * xv * xv + cf.abs();
        if residual.abs() > 1e-9 * scale {
            continue;
        }
        if seen.iter().any(|&v| (v - xv).abs() < 1e-9) {
            continue;
        }
        seen.push(xv);
        roots.push(root);
    }
    if roots.is_empty() {
        return Some(SolutionSet::Empty);
    }
    Some(SolutionSet::Discrete(roots))
}

/// Factor a MONIC integer quartic `x⁴ + b·x³ + c·x² + d·x + e` into two monic integer quadratics
/// `(x² + p·x + q)(x² + r·x + s)`, if it factors over ℚ. By Gauss's lemma a monic integer polynomial
/// that factors over ℚ factors over ℤ, so the constant terms are an integer divisor pair `q·s = e`;
/// for each, `p = (d − q·b)/(s − q)` and `r = b − p` are forced, and the factorization is accepted
/// only when `p, r` are integers and the `x²`/`x³` coefficients match. Returns `None` for an
/// irreducible quartic (e.g. `x⁴ − x − 1`) or coefficients outside `i64`.
fn factor_monic_quartic_into_rational_quadratics(
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
                continue; // degenerate q = s handled by neither branch; skip
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

/// Solve a polynomial whose deflated quotient (after peeling rational roots) is a degree-4 factor that
/// splits into two rational quadratics — `x⁵-5x³+x²-5 = (x+1)(x²-5)(x²-x+1)` loses the `±√5` roots of
/// `x²-5` because the higher-degree path drops the quartic factor. This peels the rational roots,
/// factors the monic quartic quotient into `(x²+px+q)(x²+rx+s)`, solves each quadratic for its REAL
/// roots `(−p ± √(p²−4q))/2`, and returns the complete real set `rational_roots ∪ {quadratic roots}`.
/// Every root is verified by numeric back-substitution. An irreducible quartic quotient declines.
fn try_solve_polynomial_with_quartic_factor(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_math::polynomial::Polynomial;
    use cas_solver_core::quadratic_formula::sqrt_expr;
    use cas_solver_core::rational_roots::{find_rational_roots, rational_to_expr};
    use num_rational::BigRational;
    use num_traits::ToPrimitive;
    use std::collections::HashMap;
    const MAX_CANDIDATES: usize = 256;

    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (diff, _) = simplifier.simplify(diff);
    let poly = Polynomial::from_expr(&simplifier.context, diff, var).ok()?;
    if poly.degree() < 4 {
        return None;
    }
    let (rational_roots, quotient) = find_rational_roots(poly.coeffs.clone(), MAX_CANDIDATES);
    // The deflated quotient must be a MONIC quartic with integer coefficients.
    if quotient.len() != 5 {
        return None;
    }
    let int_of = |r: &BigRational| -> Option<i64> {
        if r.is_integer() {
            r.to_i64()
        } else {
            None
        }
    };
    if int_of(&quotient[4])? != 1 {
        return None;
    }
    let e = int_of(&quotient[0])?;
    let d = int_of(&quotient[1])?;
    let c = int_of(&quotient[2])?;
    let b = int_of(&quotient[3])?;
    let ((p1, q1), (p2, q2)) = factor_monic_quartic_into_rational_quadratics(b, c, d, e)?;

    // Solve each monic quadratic `x² + p·x + q` for its real roots `(−p ± √(p²−4q))/2`.
    let mut raw_roots: Vec<ExprId> = Vec::new();
    for (p, q) in [(p1, q1), (p2, q2)] {
        let disc = p * p - 4 * q;
        if disc < 0 {
            continue; // complex roots ⇒ no real solution from this factor
        }
        let ctx = &mut simplifier.context;
        let disc_node = ctx.add(Expr::Number(BigRational::from_integer(disc.into())));
        let sqrt_disc = sqrt_expr(ctx, disc_node);
        let neg_p = ctx.add(Expr::Number(BigRational::from_integer((-p).into())));
        let two = ctx.num(2);
        let plus = ctx.add(Expr::Add(neg_p, sqrt_disc));
        let minus = ctx.add(Expr::Sub(neg_p, sqrt_disc));
        raw_roots.push(ctx.add(Expr::Div(plus, two)));
        raw_roots.push(ctx.add(Expr::Div(minus, two)));
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

    // Simplify and verify each quadratic root by numeric back-substitution; dedup by value.
    let mut seen: Vec<f64> = roots
        .iter()
        .filter_map(|&r| cas_math::evaluator_f64::eval_f64(&simplifier.context, r, &HashMap::new()))
        .collect();
    for raw in raw_roots {
        let (root, _) = simplifier.simplify(raw);
        let xv = match cas_math::evaluator_f64::eval_f64(&simplifier.context, root, &HashMap::new())
        {
            Some(v) if v.is_finite() => v,
            _ => continue,
        };
        // Verify against the ORIGINAL polynomial p(xv) ≈ 0.
        let mut residual = 0.0f64;
        let mut scale = 0.0f64;
        for (i, coeff) in poly.coeffs.iter().enumerate() {
            if let Some(cf) = coeff.to_f64() {
                let term = cf * xv.powi(i as i32);
                residual += term;
                scale += term.abs();
            }
        }
        if residual.abs() > 1e-9 * (1.0 + scale) {
            continue;
        }
        if seen.iter().any(|&v| (v - xv).abs() < 1e-9) {
            continue;
        }
        seen.push(xv);
        roots.push(root);
    }
    if roots.is_empty() {
        return Some(SolutionSet::Empty);
    }
    Some(SolutionSet::Discrete(roots))
}

/// Solve an absolute-value equation `|arg(x)| = c` for a NON-NEGATIVE constant `c` by the textbook
/// split `arg = c  ∨  arg = -c`, solving each as a full equation and unioning the roots. The recursive
/// isolation otherwise mishandles a quadratic argument with a linear term — `|x²-2x| = 3` isolates
/// `x² = 2x+3` and emits the circular residual `solve(x − (2x+3)^(1/2) = 0)` instead of `{-1, 3}`,
/// even though `solve(x²-2x = 3)` on its own returns `{-1, 3}`. Scoped to a constant RHS (`c < 0` ⇒
/// no solution; `c = 0` ⇒ the single branch `arg = 0`); a non-constant RHS needs a `g ≥ 0` domain
/// split and is left to the normal path. Roots are deduped by value.
fn try_solve_abs_equality(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_math::numeric_eval::as_rational_const;
    use num_rational::BigRational;
    use num_traits::Zero;
    use std::collections::HashMap;

    if !matches!(eq.op, cas_ast::RelOp::Eq) {
        return None;
    }
    // The left-hand side must be a unary `abs(arg)`.
    let arg = match simplifier.context.get(eq.lhs) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && simplifier.context.sym_name(*fn_id) == "abs" =>
        {
            args[0]
        }
        _ => return None,
    };
    // The right-hand side must be a constant.
    let c = as_rational_const(&simplifier.context, eq.rhs)?;
    if c < BigRational::zero() {
        return Some(SolutionSet::Empty); // |arg| = negative ⇒ no real solution
    }
    let pos = solve_relation_set(simplifier, var, arg, eq.rhs, cas_ast::RelOp::Eq)?;
    let branches = if c.is_zero() {
        vec![pos]
    } else {
        let neg_c = simplifier.context.add(Expr::Neg(eq.rhs));
        let neg = solve_relation_set(simplifier, var, arg, neg_c, cas_ast::RelOp::Eq)?;
        vec![pos, neg]
    };

    // Collect the discrete roots from both branches; bail on any non-discrete sub-result.
    let mut roots: Vec<ExprId> = Vec::new();
    for branch in branches {
        match branch {
            SolutionSet::Discrete(rs) => roots.extend(rs),
            SolutionSet::Empty => {}
            _ => return None,
        }
    }
    // Dedup by numeric value (the `arg = c` / `arg = -c` branches overlap only when `c = 0`).
    let mut seen: Vec<f64> = Vec::new();
    let mut unique: Vec<ExprId> = Vec::new();
    for root in roots {
        match cas_math::evaluator_f64::eval_f64(&simplifier.context, root, &HashMap::new()) {
            Some(v) if v.is_finite() => {
                if seen.iter().any(|&u| (u - v).abs() < 1e-9) {
                    continue;
                }
                seen.push(v);
            }
            _ => {} // non-numeric root: keep it (cannot dedup, but do not drop)
        }
        unique.push(root);
    }
    if unique.is_empty() {
        return Some(SolutionSet::Empty);
    }
    Some(SolutionSet::Discrete(unique))
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
fn try_parametric_linear_degenerate_branch(
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
fn build_cubic_real_roots(
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

/// For a polynomial equation `p(x) = 0`, peel its rational roots and — if the deflated quotient is an
/// irreducible cubic — solve that cubic exactly (radical form for Δ > 0, trigonometric form for the
/// Δ < 0 *casus irreducibilis*). Returns the complete real set `rational_roots ∪ {cubic real roots}`,
/// or `None` when no degree-3 quotient remains (or it has Δ = 0). This closes BOTH the standalone
/// irreducible cubic (`x³+x²+3 = 0`, `x³-3x+1 = 0`) and the higher-degree case where the cubic factor
/// was dropped (`x⁴+x³+3x = x·(x³+x²+3)`).
fn try_solve_polynomial_with_cubic_factor(
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
fn intersect_inequality_with_expression_domain(
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
    let conds: Vec<ImplicitCondition> = domain.conditions().iter().cloned().collect();
    let mut result = set;
    for cond in conds {
        let (arg, threshold, op) = match cond {
            ImplicitCondition::Positive(arg) => (arg, None, RelOp::Gt),
            ImplicitCondition::NonNegative(arg) => (arg, None, RelOp::Geq),
            ImplicitCondition::LowerBound(arg, c) => (arg, Some(c), RelOp::Geq),
            // `arg ≠ 0` (pole) is excluded by the rational-inequality path, not a half-line.
            ImplicitCondition::NonZero(_) => continue,
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
fn union_non_strict_inequality_roots(
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

/// True when `p` is a CLOSED endpoint of `set` (so `p ∈ set` by exact endpoint identity, with no
/// value comparison). Used to drop roots already present from the non-strict root re-union.
fn point_is_closed_endpoint(set: &SolutionSet, p: ExprId) -> bool {
    use cas_ast::BoundType;
    let on_interval = |iv: &cas_ast::Interval| {
        (iv.min == p && iv.min_type == BoundType::Closed)
            || (iv.max == p && iv.max_type == BoundType::Closed)
    };
    match set {
        SolutionSet::Continuous(iv) => on_interval(iv),
        SolutionSet::Union(ivs) => ivs.iter().any(on_interval),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        intersect_inequality_with_function_domain, required_conditions_are_contradictory,
        root_violates_required_condition, solution_contains_nonfinite,
    };
    use cas_ast::{Context, Expr};
    use cas_solver_core::domain_condition::ImplicitCondition;

    #[test]
    fn trig_threshold_classification_and_bare_detection() {
        use super::{bare_sin_or_cos_of_var, classify_trig_threshold, TrigThresholdRegion};
        use num_rational::BigRational;
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sin_x = ctx.call("sin", vec![x]);
        let cos_x = ctx.call("cos", vec![x]);
        assert!(bare_sin_or_cos_of_var(&ctx, sin_x, "x"));
        assert!(bare_sin_or_cos_of_var(&ctx, cos_x, "x"));
        // Compound argument and non-sin/cos are rejected (owned by the periodic residual path).
        let two = ctx.num(2);
        let two_x = ctx.add(Expr::Mul(two, x));
        let sin_2x = ctx.call("sin", vec![two_x]);
        assert!(!bare_sin_or_cos_of_var(&ctx, sin_2x, "x"));
        let tan_x = ctx.call("tan", vec![x]);
        assert!(!bare_sin_or_cos_of_var(&ctx, tan_x, "x"));
        // Threshold regions vs the [-1, 1] range, decided exactly.
        let c2 = ctx.num(2);
        let c1 = ctx.num(1);
        let cm1 = ctx.num(-1);
        let chalf = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
        assert!(matches!(
            classify_trig_threshold(&ctx, c2),
            Some(TrigThresholdRegion::AboveRange)
        ));
        assert!(matches!(
            classify_trig_threshold(&ctx, c1),
            Some(TrigThresholdRegion::AtUpperBound)
        ));
        assert!(matches!(
            classify_trig_threshold(&ctx, cm1),
            Some(TrigThresholdRegion::AtLowerBound)
        ));
        assert!(classify_trig_threshold(&ctx, chalf).is_none()); // in range
    }

    #[test]
    fn contains_trig_of_var_detects_periodic_argument() {
        use super::contains_trig_of_var;
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sin_x = ctx.call("sin", vec![x]);
        assert!(contains_trig_of_var(&ctx, sin_x, "x"));
        // Nested in a sum / product still detected.
        let one = ctx.num(1);
        let x_plus_sin = ctx.add(Expr::Add(x, sin_x));
        assert!(contains_trig_of_var(&ctx, x_plus_sin, "x"));
        let two = ctx.num(2);
        let two_x = ctx.add(Expr::Mul(two, x));
        let sin_2x = ctx.call("sin", vec![two_x]);
        assert!(contains_trig_of_var(&ctx, sin_2x, "x")); // compound argument
                                                          // Constant trig (`sin(2)·x`) does NOT contain the variable inside the trig argument.
        let sin_2 = ctx.call("sin", vec![two]);
        let sin2_times_x = ctx.add(Expr::Mul(sin_2, x));
        assert!(!contains_trig_of_var(&ctx, sin2_times_x, "x"));
        // A non-trig expression is false.
        let lin = ctx.add(Expr::Add(x, one));
        assert!(!contains_trig_of_var(&ctx, lin, "x"));
    }

    #[test]
    fn cmp_rational_to_quadratic_surd_is_exact() {
        use num_rational::BigRational;
        use std::cmp::Ordering;
        let r = |n: i64, d: i64| BigRational::new(n.into(), d.into());
        // φ = ½ + ½·√5 ≈ 1.618 — the worst case for a float gate near the boundary.
        let (a, b, n) = (r(1, 2), r(1, 2), BigRational::from_integer(5.into()));
        assert_eq!(
            super::cmp_rational_to_quadratic_surd(&r(8, 5), &a, &b, &n),
            Ordering::Less // 1.6 < φ
        );
        assert_eq!(
            super::cmp_rational_to_quadratic_surd(&r(13, 8), &a, &b, &n),
            Ordering::Greater // 1.625 > φ
        );
        // 2√5 ≈ 4.472 (a = 0, b = 2, n = 5): exact ordering of nearby rationals.
        let (a0, b2) = (
            BigRational::from_integer(0.into()),
            BigRational::from_integer(2.into()),
        );
        assert_eq!(
            super::cmp_rational_to_quadratic_surd(&r(9, 2), &a0, &b2, &n),
            Ordering::Greater // 4.5 > 2√5
        );
        assert_eq!(
            super::cmp_rational_to_quadratic_surd(&r(4, 1), &a0, &b2, &n),
            Ordering::Less // 4 < 2√5
        );
        // Negative coefficient `1 − √2` ≈ −0.414 (a = 1, b = −1, n = 2): sign handled.
        let (a1, bn1, n2) = (
            BigRational::from_integer(1.into()),
            BigRational::from_integer((-1).into()),
            BigRational::from_integer(2.into()),
        );
        assert_eq!(
            super::cmp_rational_to_quadratic_surd(&r(-1, 2), &a1, &bn1, &n2),
            Ordering::Less // −0.5 < 1 − √2
        );
        assert_eq!(
            super::cmp_rational_to_quadratic_surd(&r(-2, 5), &a1, &bn1, &n2),
            Ordering::Greater // −0.4 > 1 − √2
        );
        // Degenerate (b = 0): a plain rational comparison.
        let z = BigRational::from_integer(0.into());
        assert_eq!(
            super::cmp_rational_to_quadratic_surd(&r(3, 1), &r(3, 1), &z, &z),
            Ordering::Equal
        );
    }

    #[test]
    fn monotonic_inequality_intersects_argument_domain() {
        use cas_ast::{BoundType, Equation, ExprId, RelOp, SolutionSet};
        use cas_solver_core::solution_set::{neg_inf, pos_inf};
        use num_traits::Zero;

        let mut simp = crate::Simplifier::with_default_rules();
        let p = |simp: &mut crate::Simplifier, s: &str| -> ExprId {
            cas_parser::parse(s, &mut simp.context).expect("parse")
        };
        // Build the engine's naive half-line: (-inf, bound) for `<`, (bound, inf) for `>`.
        let half_line = |simp: &mut crate::Simplifier, bound: ExprId, lt: bool| -> SolutionSet {
            let (min, min_t, max, max_t) = if lt {
                (
                    neg_inf(&mut simp.context),
                    BoundType::Open,
                    bound,
                    BoundType::Open,
                )
            } else {
                (
                    bound,
                    BoundType::Open,
                    pos_inf(&mut simp.context),
                    BoundType::Open,
                )
            };
            SolutionSet::Continuous(cas_ast::Interval {
                min,
                min_type: min_t,
                max,
                max_type: max_t,
            })
        };
        let eqn = |lhs: ExprId, rhs: ExprId, op: RelOp| Equation { lhs, rhs, op };

        // sqrt(x) < 2 : naive (-inf, 2^2) intersected with [0, inf) => [0, 4).
        let (sx, two, b4) = (
            p(&mut simp, "sqrt(x)"),
            p(&mut simp, "2"),
            p(&mut simp, "2^2"),
        );
        let set = half_line(&mut simp, b4, true);
        match intersect_inequality_with_function_domain(
            &mut simp,
            &eqn(sx, two, RelOp::Lt),
            "x",
            set,
        ) {
            SolutionSet::Continuous(i) => {
                assert_eq!(i.min_type, BoundType::Closed, "lower bound closed at 0");
                assert!(
                    matches!(simp.context.get(i.min), Expr::Number(n) if n.is_zero()),
                    "lower bound is 0"
                );
            }
            other => panic!("expected [0, 4), got {other:?}"),
        }

        // sqrt(x) < -1 : even-root range disjoint => No solution.
        let (sx2, neg1, b1) = (
            p(&mut simp, "sqrt(x)"),
            p(&mut simp, "-1"),
            p(&mut simp, "(-1)^2"),
        );
        let set = half_line(&mut simp, b1, true);
        assert!(matches!(
            intersect_inequality_with_function_domain(
                &mut simp,
                &eqn(sx2, neg1, RelOp::Lt),
                "x",
                set
            ),
            SolutionSet::Empty
        ));

        // sqrt(x) > -1 : always true on the domain => [0, inf).
        let (sx3, neg1b, b1b) = (
            p(&mut simp, "sqrt(x)"),
            p(&mut simp, "-1"),
            p(&mut simp, "(-1)^2"),
        );
        let set = half_line(&mut simp, b1b, false);
        match intersect_inequality_with_function_domain(
            &mut simp,
            &eqn(sx3, neg1b, RelOp::Gt),
            "x",
            set,
        ) {
            SolutionSet::Continuous(i) => {
                assert_eq!(i.min_type, BoundType::Closed);
                assert!(matches!(simp.context.get(i.min), Expr::Number(n) if n.is_zero()));
            }
            other => panic!("expected [0, inf), got {other:?}"),
        }

        // POSITIVE multiplicative coefficient is seen through: 2·sqrt(x) < 6 has
        // the same [0, ∞) range, so the naive (-inf, 9) folds to [0, 9).
        let (csx, six, b9) = (
            p(&mut simp, "2*sqrt(x)"),
            p(&mut simp, "6"),
            p(&mut simp, "9"),
        );
        let set = half_line(&mut simp, b9, true);
        match intersect_inequality_with_function_domain(
            &mut simp,
            &eqn(csx, six, RelOp::Lt),
            "x",
            set,
        ) {
            SolutionSet::Continuous(i) => {
                assert_eq!(
                    i.min_type,
                    BoundType::Closed,
                    "2·sqrt(x)<6 lower bound closed at 0"
                );
                assert!(matches!(simp.context.get(i.min), Expr::Number(n) if n.is_zero()));
            }
            other => panic!("expected [0, 9), got {other:?}"),
        }

        // EQUATION path is untouched: sqrt(x) = 2 keeps its Discrete result.
        let (sx4, two4) = (p(&mut simp, "sqrt(x)"), p(&mut simp, "2"));
        let disc = SolutionSet::Discrete(vec![p(&mut simp, "4")]);
        assert!(matches!(
            intersect_inequality_with_function_domain(
                &mut simp,
                &eqn(sx4, two4, RelOp::Eq),
                "x",
                disc
            ),
            SolutionSet::Discrete(_)
        ));
    }

    #[test]
    fn extraneous_root_violates_recorded_domain_condition() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        // ln(x)+ln(x+5)=0 records x > 0. The extraneous root ½(-√29-5) ≈ -5.19
        // violates it; the valid root ½(√29-5) ≈ 0.19 does not.
        let positive_x = vec![ImplicitCondition::Positive(x)];
        let ext = cas_parser::parse("1/2*(-sqrt(29) - 5)", &mut ctx).expect("ext");
        let valid = cas_parser::parse("1/2*(sqrt(29) - 5)", &mut ctx).expect("valid");
        assert!(root_violates_required_condition(
            &mut ctx,
            "x",
            ext,
            &positive_x
        ));
        assert!(!root_violates_required_condition(
            &mut ctx,
            "x",
            valid,
            &positive_x
        ));
        // No recorded conditions => never a violation (byte-identical to before).
        assert!(!root_violates_required_condition(&mut ctx, "x", ext, &[]));

        // sqrt(x)=0 => x=0 sits on the NonNegative boundary and must be KEPT.
        let nonneg_x = vec![ImplicitCondition::NonNegative(x)];
        let zero = ctx.num(0);
        assert!(!root_violates_required_condition(
            &mut ctx, "x", zero, &nonneg_x
        ));

        // Reciprocal-surd negative branch -13·13^(-1/2) = -√13 violates x-2 ≥ 0.
        let xm2 = cas_parser::parse("x - 2", &mut ctx).expect("xm2");
        let nonneg_xm2 = vec![ImplicitCondition::NonNegative(xm2)];
        let neg_sqrt13 = cas_parser::parse("-13*13^(-1/2)", &mut ctx).expect("neg13");
        assert!(root_violates_required_condition(
            &mut ctx,
            "x",
            neg_sqrt13,
            &nonneg_xm2
        ));

        // The adversarial convergent: NonZero(93222358·x - 131836323) at √2 must
        // NOT fire (the value is irrational, provably nonzero) — exact arithmetic
        // keeps the valid root where a float gate would drop it.
        let denom = cas_parser::parse("93222358*x - 131836323", &mut ctx).expect("denom");
        let nonzero = vec![ImplicitCondition::NonZero(denom)];
        let root2 = cas_parser::parse("sqrt(2)", &mut ctx).expect("sqrt2");
        assert!(!root_violates_required_condition(
            &mut ctx, "x", root2, &nonzero
        ));
    }

    #[test]
    fn contradictory_positive_conditions_empty_the_real_domain() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let neg_x = cas_parser::parse("-x", &mut ctx).expect("neg x");
        let xp5 = cas_parser::parse("x + 5", &mut ctx).expect("x+5");

        // `x > 0` AND `-x > 0` is impossible — this is the `ln(x)=ln(-x)`
        // collapse (an `AllReals` carrying both must become Empty).
        assert!(required_conditions_are_contradictory(
            &ctx,
            &[
                ImplicitCondition::Positive(x),
                ImplicitCondition::Positive(neg_x),
            ]
        ));
        // `x > 0` AND `x+5 > 0` is satisfiable.
        assert!(!required_conditions_are_contradictory(
            &ctx,
            &[
                ImplicitCondition::Positive(x),
                ImplicitCondition::Positive(xp5),
            ]
        ));
        // NonNegative pair (`x >= 0` AND `-x >= 0`) meets at 0 — NOT contradictory
        // (the check must be strict `> 0`, not `>= 0`).
        assert!(!required_conditions_are_contradictory(
            &ctx,
            &[
                ImplicitCondition::NonNegative(x),
                ImplicitCondition::NonNegative(neg_x),
            ]
        ));
        // `-8x > 0` AND `x > 0` is impossible — the generalized negative-multiple
        // case (`log(2,-8x)=log(2,x)+k`), not just exact negation.
        let neg_8x = cas_parser::parse("-8*x", &mut ctx).expect("-8x");
        assert!(required_conditions_are_contradictory(
            &ctx,
            &[
                ImplicitCondition::Positive(neg_8x),
                ImplicitCondition::Positive(x),
            ]
        ));
        // `2x > 0` AND `x > 0` is a POSITIVE multiple — satisfiable, NOT collapsed.
        let two_x = cas_parser::parse("2*x", &mut ctx).expect("2x");
        assert!(!required_conditions_are_contradictory(
            &ctx,
            &[
                ImplicitCondition::Positive(two_x),
                ImplicitCondition::Positive(x),
            ]
        ));
        // A single condition (or none) is never contradictory.
        assert!(!required_conditions_are_contradictory(
            &ctx,
            &[ImplicitCondition::Positive(x)]
        ));
        assert!(!required_conditions_are_contradictory(&ctx, &[]));
    }

    #[test]
    fn out_of_range_inverse_trig_root_is_not_real() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let neg_three = ctx.num(-3);
        let half = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let one = ctx.num(1);
        let neg_one = ctx.num(-1);

        // arcsin/arccos of |c| > 1 are undefined over ℝ — not real solutions.
        let arcsin_two = ctx.call("arcsin", vec![two]);
        let arccos_two = ctx.call("arccos", vec![two]);
        let arcsin_neg_three = ctx.call("arcsin", vec![neg_three]);
        assert!(solution_contains_nonfinite(&ctx, arcsin_two));
        assert!(solution_contains_nonfinite(&ctx, arccos_two));
        assert!(solution_contains_nonfinite(&ctx, arcsin_neg_three));

        // A root that merely CONTAINS an out-of-range inverse-trig term is also
        // non-real (e.g. `arcsin(2) + 1`).
        let arcsin_two2 = ctx.call("arcsin", vec![two]);
        let shifted = ctx.add(Expr::Add(arcsin_two2, one));
        assert!(solution_contains_nonfinite(&ctx, shifted));

        // In-range / boundary arguments are genuine real values — kept.
        let arcsin_half = ctx.call("arcsin", vec![half]);
        let arccos_one = ctx.call("arccos", vec![one]);
        let arcsin_neg_one = ctx.call("arcsin", vec![neg_one]);
        assert!(!solution_contains_nonfinite(&ctx, arcsin_half));
        assert!(!solution_contains_nonfinite(&ctx, arccos_one));
        assert!(!solution_contains_nonfinite(&ctx, arcsin_neg_one));

        // Round-4 Cluster C: a SURD argument outside [-1, 1] (`√2 ≈ 1.41`, `√3`,
        // `2√2`) is non-real and must be dropped — EXACTLY, via the quadratic-surd
        // sign logic (a float gate could drop a valid root). In-range surds
        // (`√2/2 ≈ 0.71`, `√3/2`) are kept.
        for src in ["sqrt(2)", "sqrt(3)", "2*sqrt(2)", "sqrt(8)", "-sqrt(2)"] {
            let arg = cas_parser::parse(src, &mut ctx).expect("surd");
            let call = ctx.call("arcsin", vec![arg]);
            assert!(
                solution_contains_nonfinite(&ctx, call),
                "arcsin({src}) is non-real (|arg| > 1)"
            );
        }
        for src in ["sqrt(2)/2", "sqrt(3)/2", "1/2", "-sqrt(2)/2"] {
            let arg = cas_parser::parse(src, &mut ctx).expect("surd");
            let call = ctx.call("arccos", vec![arg]);
            assert!(
                !solution_contains_nonfinite(&ctx, call),
                "arccos({src}) is in range (|arg| <= 1), must be KEPT"
            );
        }
    }

    #[test]
    fn identically_zero_denominator_makes_equation_unsolvable() {
        use super::equation_has_identically_zero_denominator;
        use cas_ast::{Equation, RelOp};

        let mut simplifier = crate::Simplifier::with_default_rules();
        let build = |simp: &mut crate::Simplifier, lhs: &str, rhs: &str| -> Equation {
            let l = cas_parser::parse(lhs, &mut simp.context).expect("lhs");
            let r = cas_parser::parse(rhs, &mut simp.context).expect("rhs");
            Equation {
                lhs: l,
                rhs: r,
                op: RelOp::Eq,
            }
        };

        // Provably-zero denominators (literal 0, x-x, 0*x) => identically
        // undefined over ℝ => caught, so the solver returns "No solution".
        for (l, r) in [
            ("x/0", "5"),
            ("x", "1/0"),
            ("x/(x-x)", "0"),
            ("1/(0*x)", "2"),
        ] {
            let eq = build(&mut simplifier, l, r);
            assert!(
                equation_has_identically_zero_denominator(&mut simplifier, &eq),
                "{l} = {r} has an identically-zero denominator"
            );
        }

        // Denominators that are nonzero or merely vanish at isolated points are
        // legitimate (excluded points, not an undefined equation) => NOT caught.
        for (l, r) in [
            ("3/x", "0"),
            ("1/x", "2"),
            ("x/(x-1)", "2"),
            ("1/(x-x+1)", "1"),
        ] {
            let eq = build(&mut simplifier, l, r);
            assert!(
                !equation_has_identically_zero_denominator(&mut simplifier, &eq),
                "{l} = {r} denominator is nonzero or only vanishes at a point"
            );
        }
    }
}
