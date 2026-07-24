//! Local active solve backend boundary.
//!
//! This backend is solver-owned and executes the solver-native runtime pipeline.
//! Keeping this indirection local lets us switch implementations without
//! changing call sites.

use crate::solve_exponential_terms::{
    collect_exp_integer_bases, collect_exp_laurent_terms, collect_exponential_base_terms,
    integer_prime_power, rewrite_exp_bases_to_prime,
};
use crate::{CasError, Simplifier, SolveCtx, SolveStep};
use cas_ast::{substitute_expr_by_id, Constant, Context, Equation, Expr, ExprId, SolutionSet};
// Canonical surd-sign kernel (was a byte-identical private `linear_surd_sign` copy).
use cas_math::root_forms::sign_of_linear_surd as linear_surd_sign;
// Canonical exact surd/nth-root comparators (were private copies; chokepoint A part 2).
use cas_math::root_forms::{cmp_rational_to_nth_root, cmp_rational_to_quadratic_surd};
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
            // A calculus binder's bounds carry ∞ as notation, not as a value:
            // `y = limit(1/x, x, infinity)` has the finite solution 0 — do not
            // drop it (which asserted "No solution", 2026-07-19).
            let name = ctx.sym_name(*fn_id);
            if cas_solver_core::solve_outcome::CALCULUS_BINDER_FN_NAMES.contains(&name) {
                return false;
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
fn try_solve_sum_of_two_radicals_equation(
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
fn try_solve_single_radical_equals_polynomial(
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
fn try_solve_poly_over_sqrt_equation(
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
    // Back-substitute each root `atom = u_root`, then union the branch solutions (periodic-aware).
    let mut branch_sets = Vec::with_capacity(u_roots.len());
    for u_root in u_roots {
        let back_eq = Equation {
            lhs: back_sub_atom,
            rhs: u_root,
            op: cas_ast::RelOp::Eq,
        };
        let (xs, _) = crate::solver_entrypoints_solve::solve(&back_eq, var, simplifier).ok()?;
        branch_sets.push(xs);
    }
    union_branch_solutions(simplifier, branch_sets)
}

/// Union a set of branch/root solution sets. PERIODIC families are collected and combined over a common
/// period (which `union_solution_sets` cannot do for DIFFERENT periods — it would drop one); `Empty`
/// branches are skipped; non-periodic branches (points/intervals) union normally. Shared by the
/// polynomial-in-atom equation solver and the `|A| = c` absolute-value solver.
fn union_branch_solutions(
    simplifier: &mut Simplifier,
    branch_sets: Vec<SolutionSet>,
) -> Option<SolutionSet> {
    let mut solution = SolutionSet::Empty;
    let mut periodic_families: Vec<(Vec<ExprId>, ExprId)> = Vec::new();
    for s in branch_sets {
        match s {
            SolutionSet::Periodic { bases, period } => periodic_families.push((bases, period)),
            SolutionSet::Empty => {} // no real pre-image for this branch (e.g. sin(x) = 2)
            other => {
                solution = cas_solver_core::solution_set::union_solution_sets(
                    &simplifier.context,
                    solution,
                    other,
                );
            }
        }
    }
    if !periodic_families.is_empty() {
        let combined = if periodic_families.len() == 1 {
            let (bases, period) = periodic_families.pop().unwrap();
            SolutionSet::Periodic { bases, period }
        } else {
            union_periodic_families_over_common_period(simplifier, periodic_families)?
        };
        solution = if matches!(solution, SolutionSet::Empty) {
            combined
        } else {
            // Discrete/interval branches mixed with a periodic family are unrepresentable by
            // `union_solution_sets` (its catch-all fires a debug_assert and silently DROPS one
            // operand in release — a latent wrong answer). Honor its documented contract:
            // pre-check combinability at the solver layer and decline the whole relation,
            // leaving an honest residual instead of an incomplete set.
            return None;
        };
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

/// Match a leaf `coeff · x^e` (any rational `e`, incl. negative), a `var`-free
/// constant `(0, value)`, or a reciprocal `c / x^e` `(−e, c)`. Returns
/// `(exponent, coeff)`; `None` for any other shape. Used by the reciprocal-root
/// solver to build the Laurent map `x^(p/q) → u^p`.
fn x_root_laurent_leaf(
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
fn collect_x_root_laurent_pairs(
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
fn try_solve_rational_power_laurent(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
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
    solve_polynomial_in_atom(simplifier, u_expr, "__rpl_u", var, atom)
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

/// Solve an EQUATION that is a polynomial of degree ≥ 2 in `|x|`, e.g.
/// `|x|² − 3·|x| + 2 = 0`. The simplifier folds `|x|² → x²`, so the equation
/// reaches here as `x² − 3·|x| + 2 = 0`; because `x² = |x|²`, it is a quadratic
/// in `u = |x|`. Substitute `u = |x|`, solve `u² − 3u + 2 = 0`, then
/// back-substitute `|x| = u_root` — the recursive `|A| = c` solver drops a
/// negative root and splits each `u_root ≥ 0` into `x = ±u_root`. Without this,
/// the isolation path reorients to `x = √(3·|x| − 2)` and leaks a malformed
/// `solve(...)` residual, dropping the negative branch and every root.
///
/// Gated to abs of the BARE variable (`|x|`, not `|x − 1|`): only then does
/// `x^(2k) = |x|^(2k)` unify. Validated by requiring the difference to be EVEN
/// in `x` — an odd term (`x + |x|`) is not a polynomial in `|x|` and declines to
/// its own handler.
fn try_solve_polynomial_in_abs(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::{BuiltinFn, RelOp};
    use cas_solver_core::isolation_utils::contains_var;

    if eq.op != RelOp::Eq {
        return None;
    }
    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (diff, _) = simplifier.simplify(diff);

    // Atom is `|x|` — abs of the bare variable, so `x^(2k) = |x|^(2k)` unifies.
    let x = simplifier.context.var(var);
    let abs_x = simplifier.context.call_builtin(BuiltinFn::Abs, vec![x]);
    let u_var = "__abs_u";
    let u = simplifier.context.var(u_var);
    let e1 = substitute_expr_by_id(&mut simplifier.context, diff, abs_x, u);
    if e1 == diff {
        return None; // no bare `|x|` present
    }

    // The difference must be EVEN in x for `x^(2k) = |x|^(2k)` (|x| is itself
    // even): a surviving odd component (`x + |x|`) is not a polynomial in |x|.
    let neg_x = simplifier.context.add(Expr::Neg(x));
    let diff_negx = substitute_expr_by_id(&mut simplifier.context, diff, x, neg_x);
    let (diff_negx, _) = simplifier.simplify(diff_negx);
    if diff_negx != diff {
        return None;
    }

    // Unify the even x-powers into the same atom: `x → u` turns `x² − 3u + 2`
    // into `u² − 3u + 2`. Any leftover `x` (or a non-bare `|g|`, which
    // `Polynomial::from_expr` inside the shared core rejects) declines.
    let u_expr = substitute_expr_by_id(&mut simplifier.context, e1, x, u);
    if contains_var(&simplifier.context, u_expr, var) {
        return None;
    }
    solve_polynomial_in_atom(simplifier, u_expr, u_var, var, abs_x)
}

/// Collect every distinct `|f|` sub-term of `expr` whose argument contains
/// `var`, without descending into an abs argument (a nested abs makes the caller
/// decline). Used to require a SINGLE absolute-value term.
fn collect_abs_of_var(ctx: &Context, expr: ExprId, var: &str, out: &mut Vec<ExprId>) {
    use cas_ast::BuiltinFn;
    use cas_solver_core::isolation_utils::contains_var;
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Abs) =>
        {
            if contains_var(ctx, args[0], var) {
                out.push(expr);
            }
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            let (l, r) = (*l, *r);
            collect_abs_of_var(ctx, l, var, out);
            collect_abs_of_var(ctx, r, var, out);
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            let inner = *inner;
            collect_abs_of_var(ctx, inner, var, out);
        }
        Expr::Function(_, args) => {
            let args = args.clone();
            for a in args {
                collect_abs_of_var(ctx, a, var, out);
            }
        }
        _ => {}
    }
}

/// Solve an equation carrying a SINGLE absolute-value term `|f(x)|` linearly,
/// with a NON-CONSTANT polynomial remainder of degree ≥ 2 — `x² + |x−1| − 3 = 0`,
/// i.e. `|f| = g` where `g` is a polynomial. Isolating the abs and recursing is
/// UNSOUND here: the generic path solves only the `f = g` branch (dropping
/// `f = −g`) and skips the `g ≥ 0` domain, so it returns a spurious root and
/// misses a real one (`x²+|x−1|−3=0 → {−2.56, 1.56}` instead of the true
/// `{−1, (−1+√17)/2}`), or leaks a malformed `solve(x−√(3|x−1|−2))` residual
/// (`x²−3|x−1|+2=0`).
///
/// Solve BOTH branches `f = g` and `f = −g`, then keep each root `r` iff
/// `g(r) ≥ 0` — the exact verification, since `|f(r)| = |±g(r)| = g(r)` requires
/// `g(r) ≥ 0` — decided by the constant-sign layer so surd roots are handled
/// (an undecidable sign declines the whole handler, never emitting an
/// unverified set). Gated to `deg(g) ≥ 2`: a linear `g` (`|x−2| = x`) is solved
/// correctly by the isolation path and stays there.
fn try_solve_single_abs_equals_polynomial(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    use cas_math::const_sign::{provable_const_sign, ConstSign};
    use cas_math::numeric_eval::as_rational_const;
    use cas_math::polynomial::Polynomial;
    use cas_solver_core::isolation_utils::contains_var;
    use num_rational::BigRational;
    use num_traits::Zero;

    if eq.op != RelOp::Eq {
        return None;
    }
    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (diff, _) = simplifier.simplify(diff);

    // Exactly one distinct `|f|` sub-term whose argument contains the variable.
    let mut abs_terms: Vec<ExprId> = Vec::new();
    collect_abs_of_var(&simplifier.context, diff, var, &mut abs_terms);
    let mut distinct: Vec<ExprId> = Vec::new();
    for t in abs_terms {
        if !distinct.contains(&t) {
            distinct.push(t);
        }
    }
    if distinct.len() != 1 {
        return None;
    }
    let abs_f = distinct[0];
    let f = match simplifier.context.get(abs_f) {
        Expr::Function(_, args) if args.len() == 1 => args[0],
        _ => return None,
    };

    // `diff` must be linear in `|f|`: `diff = c·|f| + rest`, `c` a nonzero rational.
    let u_var = "__absg_u";
    let u = simplifier.context.var(u_var);
    let diff_u = substitute_expr_by_id(&mut simplifier.context, diff, abs_f, u);
    let zero = simplifier.context.num(0);
    let one = simplifier.context.num(1);
    let two = simplifier.context.num(2);
    let rest = substitute_expr_by_id(&mut simplifier.context, diff_u, u, zero);
    let (rest, _) = simplifier.simplify(rest);
    let at_one = substitute_expr_by_id(&mut simplifier.context, diff_u, u, one);
    let c_diff = simplifier.context.add(Expr::Sub(at_one, rest));
    let (c_diff, _) = simplifier.simplify(c_diff);
    let c = as_rational_const(&simplifier.context, c_diff)?;
    if c.is_zero() {
        return None;
    }
    // Linearity: `diff_u[u→2] − (rest + 2c)` must be the zero polynomial in x.
    let at_two = substitute_expr_by_id(&mut simplifier.context, diff_u, u, two);
    let two_c = simplifier
        .context
        .add(Expr::Number(&c * BigRational::from_integer(2.into())));
    let predicted = simplifier.context.add(Expr::Add(rest, two_c));
    let lin_check = simplifier.context.add(Expr::Sub(at_two, predicted));
    let (lin_check, _) = simplifier.simplify(lin_check);
    if !Polynomial::from_expr(&simplifier.context, lin_check, var)
        .map(|p| p.is_zero())
        .unwrap_or(false)
    {
        return None;
    }

    // `g = −rest / c`. Require it non-constant of degree ≥ 2 (linear `g` and a
    // constant `g` are handled correctly elsewhere).
    let neg_rest = simplifier.context.add(Expr::Neg(rest));
    let c_num = simplifier.context.add(Expr::Number(c));
    let g = simplifier.context.add(Expr::Div(neg_rest, c_num));
    let (g, _) = simplifier.simplify(g);
    if !contains_var(&simplifier.context, g, var) {
        return None;
    }
    match Polynomial::from_expr(&simplifier.context, g, var) {
        Ok(p) if p.degree() >= 2 => {}
        _ => return None,
    }

    // Solve both branches `f = g` and `f = −g`.
    let neg_g = simplifier.context.add(Expr::Neg(g));
    let (neg_g, _) = simplifier.simplify(neg_g);
    let mut candidates: Vec<ExprId> = Vec::new();
    for rhs in [g, neg_g] {
        let branch = Equation {
            lhs: f,
            rhs,
            op: RelOp::Eq,
        };
        let (sol, _) = crate::solver_entrypoints_solve::solve(&branch, var, simplifier).ok()?;
        match sol {
            SolutionSet::Discrete(roots) => candidates.extend(roots),
            SolutionSet::Empty => {}
            _ => return None, // a non-discrete branch ⇒ out of scope
        }
    }

    // Keep `r` iff `g(r) ≥ 0`, decided exactly (surd-aware). Dedup by value.
    let x = simplifier.context.var(var);
    let mut kept: Vec<ExprId> = Vec::new();
    for r in candidates {
        let g_at_r = substitute_expr_by_id(&mut simplifier.context, g, x, r);
        let (g_at_r, _) = simplifier.simplify(g_at_r);
        let sign = if let Some(q) = as_rational_const(&simplifier.context, g_at_r) {
            if q.is_zero() {
                ConstSign::Zero
            } else if q > BigRational::zero() {
                ConstSign::Positive
            } else {
                ConstSign::Negative
            }
        } else {
            provable_const_sign(&simplifier.context, g_at_r)?
        };
        if matches!(sign, ConstSign::Negative) {
            continue;
        }
        if !kept.iter().any(|&k| {
            cas_ast::ordering::compare_expr(&simplifier.context, k, r) == std::cmp::Ordering::Equal
        }) {
            kept.push(r);
        }
    }
    if kept.is_empty() {
        Some(SolutionSet::Empty)
    } else {
        Some(SolutionSet::Discrete(kept))
    }
}

/// Solve a RELATION (inequality or equation) carrying a SINGLE `|f(x)|` term
/// inside a polynomial-in-x context — `x² − 3|x| + 2 < 0`, `x·|x| = 4` — by the
/// textbook sign split at `f = 0`. The generic path treats the abs opaquely: for
/// the inequality it returns a WRONG "No solution" (the true set is
/// `(−2,−1) ∪ (1,2)`); for a MULTIPLICATIVELY entangled equation like `x·|x| = 4`
/// the isolation path reorients to `x = 4/|x|` and leaks a malformed
/// `solve(x − 4/|x| = 0)` residual (true answer `{2}`). On `f ≥ 0`, `|f| = f`; on
/// `f < 0`, `|f| = −f`; solve each polynomial branch, intersect with its domain,
/// and union. For an equation the branch solve yields discrete roots and the
/// intersection keeps only the ones in that branch's half-line.
///
/// Gated to a single abs whose removal leaves a genuine polynomial-in-x
/// remainder — bare `|f| {op} c` (constant remainder), reciprocal/sign,
/// isolated-abs (`|f| = g`), poly-in-|x|, and multi-abs relations keep their
/// own, already-correct handlers (this dispatches strictly after them).
fn try_solve_single_abs_polynomial_relation(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    use cas_math::polynomial::Polynomial;
    use cas_solver_core::isolation_utils::contains_var;
    use cas_solver_core::solution_set::{intersect_solution_sets, union_solution_sets};

    if !matches!(
        eq.op,
        RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq | RelOp::Eq
    ) {
        return None;
    }
    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (diff, _) = simplifier.simplify(diff);

    // Exactly one distinct `|f|` sub-term whose argument contains the variable.
    let mut abs_terms: Vec<ExprId> = Vec::new();
    collect_abs_of_var(&simplifier.context, diff, var, &mut abs_terms);
    let mut distinct: Vec<ExprId> = Vec::new();
    for t in abs_terms {
        if !distinct.contains(&t) {
            distinct.push(t);
        }
    }
    if distinct.len() != 1 {
        return None;
    }
    let abs_f = distinct[0];
    let f = match simplifier.context.get(abs_f) {
        Expr::Function(_, args) if args.len() == 1 => args[0],
        _ => return None,
    };

    // Branch substitutions: `|f| = f` (on f ≥ 0), `|f| = −f` (on f < 0). Both
    // branches must be polynomials in x, else out of scope.
    let zero = simplifier.context.num(0);
    let neg_f = simplifier.context.add(Expr::Neg(f));
    let pos_expr = substitute_expr_by_id(&mut simplifier.context, diff, abs_f, f);
    let (pos_expr, _) = simplifier.simplify(pos_expr);
    let neg_expr = substitute_expr_by_id(&mut simplifier.context, diff, abs_f, neg_f);
    let (neg_expr, _) = simplifier.simplify(neg_expr);
    let (Ok(pos_poly), Ok(neg_poly)) = (
        Polynomial::from_expr(&simplifier.context, pos_expr, var),
        Polynomial::from_expr(&simplifier.context, neg_expr, var),
    ) else {
        return None;
    };

    // The generic path only fails when the abs is entangled with genuine
    // polynomial-in-x structure: either a non-constant remainder after removing
    // the abs (`x² − 3|x| + 2`, the abs added to a polynomial) OR a branch whose
    // degree rises ABOVE the abs argument's own degree (multiplicative `x·|x|`,
    // factor `|x|³ − |x| = |x|(x²−1)` — both raise a degree-1 argument to 2/3).
    //
    // The floor is op-aware. For an EQUATION with an ISOLATED abs of a
    // higher-degree argument and a constant remainder (`|x²−4| = 3`), the split
    // yields the right roots but an ugly form (`−7·7^(−1/2)`); the dedicated
    // isolated-abs equation handler downstream emits the canonical `−√7`, so we
    // decline (floor = the argument's degree keeps `deg == arg_deg` out).
    // Inequalities keep the established `deg ≥ 2` gate (floor 1): their
    // `|quadratic| {op} c` form has always been owned by THIS handler, and the
    // committed contract pins that representation.
    let rest = substitute_expr_by_id(&mut simplifier.context, diff, abs_f, zero);
    let (rest, _) = simplifier.simplify(rest);
    let entangle_floor = if matches!(eq.op, RelOp::Eq) {
        Polynomial::from_expr(&simplifier.context, f, var)
            .map(|p| p.degree())
            .unwrap_or(1)
            .max(1)
    } else {
        1
    };
    let entangled = contains_var(&simplifier.context, rest, var)
        || pos_poly.degree() > entangle_floor
        || neg_poly.degree() > entangle_floor;
    if !entangled {
        return None;
    }

    // A branch whose RAW tree keeps an unexpanded Mul shape can defeat the
    // recursive solver (`−x·(x−1) − 2 < 0` leaks a mangled residual), and the set
    // algebra below would silently swallow the non-concrete operand, dropping a
    // whole region (`|x|·|x−1| < 2` lost the between-the-zeros interval (0, 1)).
    // Fall back to the ALREADY-PARSED branch polynomial, whose `to_expr`
    // canonicalizes to the expanded form the recursive solver does handle.
    let solve_branch = |simplifier: &mut Simplifier,
                        branch_expr: ExprId,
                        branch_poly: &cas_math::polynomial::Polynomial|
     -> Option<SolutionSet> {
        let set = solve_relation_set(simplifier, var, branch_expr, zero, eq.op.clone())?;
        if is_concrete_solution_set(&set) {
            return Some(set);
        }
        let set = solve_poly_sign(simplifier, var, branch_poly, eq.op.clone())?;
        is_concrete_solution_set(&set).then_some(set)
    };
    let pos_branch = solve_branch(simplifier, pos_expr, &pos_poly)?;
    let neg_branch = solve_branch(simplifier, neg_expr, &neg_poly)?;
    let pos_domain = solve_relation_set(simplifier, var, f, zero, RelOp::Geq)?;
    let neg_domain = solve_relation_set(simplifier, var, f, zero, RelOp::Lt)?;

    let final_pos = intersect_solution_sets(&simplifier.context, pos_branch, pos_domain);
    let final_neg = intersect_solution_sets(&simplifier.context, neg_branch, neg_domain);
    Some(union_solution_sets(
        &simplifier.context,
        final_pos,
        final_neg,
    ))
}

/// Solve a relation with TWO OR MORE affine `|f|` terms AND a degree-≥2
/// polynomial remainder — `x² + |x−1| + |x+1| < 5` — by the exact
/// piecewise/breakpoint method. The linear sum-of-abs handler carries only a
/// LINEAR remainder, so a quadratic term makes it decline and the generic path
/// returns a wrong "No solution" (the true set is `(1−√6, √6−1)`).
///
/// Partition ℝ at the sorted breakpoints (`−bᵢ/aᵢ` of each affine argument). On
/// each segment every `|f|` has a fixed sign, so substitute `|f| = ±f` and solve
/// the resulting POLYNOMIAL relation on the whole line, then clip to the closed
/// segment and union. Gated to ≥2 abs and a degree-≥2 remainder (single abs is
/// the sign-split handler's job; a linear remainder the existing sum handler's).
fn try_solve_multi_abs_polynomial_relation(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    use cas_math::polynomial::Polynomial;
    use cas_solver_core::solution_set::{intersect_solution_sets, union_solution_sets};
    use num_rational::BigRational;
    use num_traits::{One, Signed, Zero};

    if !matches!(
        eq.op,
        RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq | RelOp::Eq
    ) {
        return None;
    }
    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (diff, _) = simplifier.simplify(diff);

    // Distinct abs-of-var terms; require ≥ 2 (single abs handled elsewhere).
    let mut raw: Vec<ExprId> = Vec::new();
    collect_abs_of_var(&simplifier.context, diff, var, &mut raw);
    let mut abs_exprs: Vec<ExprId> = Vec::new();
    for t in raw {
        if !abs_exprs.contains(&t) {
            abs_exprs.push(t);
        }
    }
    if abs_exprs.len() < 2 {
        return None;
    }

    // Each argument must be AFFINE, giving a rational breakpoint `−b/a`.
    let mut breakpoints: Vec<BigRational> = Vec::new();
    let mut arg_polys: Vec<Polynomial> = Vec::new();
    let mut args: Vec<ExprId> = Vec::new();
    for &abs_e in &abs_exprs {
        let arg = match simplifier.context.get(abs_e) {
            Expr::Function(_, a) if a.len() == 1 => a[0],
            _ => return None,
        };
        let poly = Polynomial::from_expr(&simplifier.context, arg, var).ok()?;
        if poly.degree() != 1 {
            return None;
        }
        let a = poly
            .coeffs
            .get(1)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let b = poly
            .coeffs
            .first()
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if a.is_zero() {
            return None;
        }
        breakpoints.push(-b / a);
        arg_polys.push(poly);
        args.push(arg);
    }

    // Gate: the abs-free remainder must be non-linear (deg ≥ 2) — the linear
    // sum-of-abs handler already owns a linear remainder (`|x−1| + |x+1| < 3`).
    let zero = simplifier.context.num(0);
    let mut rem = diff;
    for &abs_e in &abs_exprs {
        rem = substitute_expr_by_id(&mut simplifier.context, rem, abs_e, zero);
    }
    let (rem, _) = simplifier.simplify(rem);
    match Polynomial::from_expr(&simplifier.context, rem, var) {
        Ok(p) if p.degree() >= 2 => {}
        _ => return None,
    }

    breakpoints.sort();
    breakpoints.dedup();
    let n = breakpoints.len();
    let two = BigRational::from_integer(2.into());

    // Closed segment `[lo, hi]` as a solution set, via half-line solves; an open
    // end (`None`) contributes `AllReals` (no constraint on that side).
    let segment_set = |simplifier: &mut Simplifier,
                       lo: Option<&BigRational>,
                       hi: Option<&BigRational>|
     -> Option<SolutionSet> {
        let x = simplifier.context.var(var);
        let lo_set = match lo {
            Some(l) => {
                let ln = simplifier.context.add(Expr::Number(l.clone()));
                solve_relation_set(simplifier, var, x, ln, RelOp::Geq)?
            }
            None => SolutionSet::AllReals,
        };
        let hi_set = match hi {
            Some(h) => {
                let hn = simplifier.context.add(Expr::Number(h.clone()));
                solve_relation_set(simplifier, var, x, hn, RelOp::Leq)?
            }
            None => SolutionSet::AllReals,
        };
        Some(intersect_solution_sets(&simplifier.context, lo_set, hi_set))
    };

    let mut solution = SolutionSet::Empty;
    for seg_idx in 0..=n {
        let (lo, hi, test): (Option<BigRational>, Option<BigRational>, BigRational) =
            if seg_idx == 0 {
                let a0 = breakpoints[0].clone();
                let t = &a0 - BigRational::one();
                (None, Some(a0), t)
            } else if seg_idx == n {
                let an = breakpoints[n - 1].clone();
                let t = &an + BigRational::one();
                (Some(an), None, t)
            } else {
                let al = breakpoints[seg_idx - 1].clone();
                let ar = breakpoints[seg_idx].clone();
                let t = (&al + &ar) / &two;
                (Some(al), Some(ar), t)
            };

        // Resolve each `|f| → sign·f` using the sign at the interior test point.
        let mut seg_expr = diff;
        for (i, &abs_e) in abs_exprs.iter().enumerate() {
            let val = arg_polys[i].eval(&test);
            let replacement = if val.is_positive() {
                args[i]
            } else {
                simplifier.context.add(Expr::Neg(args[i]))
            };
            seg_expr = substitute_expr_by_id(&mut simplifier.context, seg_expr, abs_e, replacement);
        }
        let (seg_expr, _) = simplifier.simplify(seg_expr);

        let branch = solve_relation_set(simplifier, var, seg_expr, zero, eq.op.clone())?;
        let seg_set = segment_set(simplifier, lo.as_ref(), hi.as_ref())?;
        let clipped = intersect_solution_sets(&simplifier.context, branch, seg_set);
        solution = union_solution_sets(&simplifier.context, solution, clipped);
    }
    Some(solution)
}

/// Core of [`try_solve_polynomial_in_trig`]: treat `diff` as a polynomial of degree ≥ 2 in a single
/// trig atom `sin(g)`/`cos(g)`/`tan(g)`, substitute `u = trig(g)`, solve `P(u) = 0`, and back-substitute
/// each root through the periodic solver (range guard drops `|u| > 1`). Returns `None` if `diff` is not
/// a polynomial in ONE such atom (a second, distinct atom or `x` remains after substitution).
fn solve_polynomial_in_trig_from_diff(
    simplifier: &mut Simplifier,
    diff: ExprId,
    var: &str,
) -> Option<SolutionSet> {
    let atom = find_trig_atom_containing_var(&simplifier.context, diff, var)?;
    let u_var = "__trig_u";
    let u = simplifier.context.var(u_var);
    let u_expr = substitute_expr_by_id(&mut simplifier.context, diff, atom, u);
    if expr_contains_named_var(&simplifier.context, u_expr, var) {
        return None; // a second, distinct trig atom (or x elsewhere) remains
    }
    solve_polynomial_in_atom(simplifier, u_expr, u_var, var, atom)
}

/// Rewrite every EVEN power of `atom` in `expr` via `atom² = sq_repl` (`atom^(2k) → sq_repl^k`),
/// returning the rewritten tree — or `None` if `atom` occurs to any ODD power (a bare `atom` or
/// `atom^(2k+1)`), which the even-power Pythagorean substitution cannot eliminate. Used to turn a
/// mixed `sin(g)`/`cos(g)` polynomial into a single-atom one via `cos² = 1 − sin²` (or `sin² = 1 − cos²`).
fn rewrite_even_power_of_atom(
    ctx: &mut Context,
    expr: ExprId,
    atom: ExprId,
    sq_repl: ExprId,
) -> Option<ExprId> {
    use cas_ast::ordering::compare_expr;
    // A bare `atom` is an odd (first) power — not eliminable by an even-power substitution.
    if compare_expr(ctx, expr, atom) == std::cmp::Ordering::Equal {
        return None;
    }
    match ctx.get(expr).clone() {
        Expr::Pow(base, exp) => {
            if compare_expr(ctx, base, atom) == std::cmp::Ordering::Equal {
                // `atom^n`: even `n` ⇒ `sq_repl^(n/2)`; odd `n` ⇒ not eliminable.
                let n = cas_math::numeric_eval::as_rational_const(ctx, exp)?;
                if !n.is_integer() {
                    return None;
                }
                let n = num_traits::ToPrimitive::to_i64(&n.to_integer())?;
                if n <= 0 || n % 2 != 0 {
                    return None;
                }
                let half = ctx.num(n / 2);
                return Some(ctx.add(Expr::Pow(sq_repl, half)));
            }
            let base = rewrite_even_power_of_atom(ctx, base, atom, sq_repl)?;
            let exp = rewrite_even_power_of_atom(ctx, exp, atom, sq_repl)?;
            Some(ctx.add(Expr::Pow(base, exp)))
        }
        Expr::Add(l, r) => {
            let l = rewrite_even_power_of_atom(ctx, l, atom, sq_repl)?;
            let r = rewrite_even_power_of_atom(ctx, r, atom, sq_repl)?;
            Some(ctx.add(Expr::Add(l, r)))
        }
        Expr::Sub(l, r) => {
            let l = rewrite_even_power_of_atom(ctx, l, atom, sq_repl)?;
            let r = rewrite_even_power_of_atom(ctx, r, atom, sq_repl)?;
            Some(ctx.add(Expr::Sub(l, r)))
        }
        Expr::Mul(l, r) => {
            let l = rewrite_even_power_of_atom(ctx, l, atom, sq_repl)?;
            let r = rewrite_even_power_of_atom(ctx, r, atom, sq_repl)?;
            Some(ctx.add(Expr::Mul(l, r)))
        }
        Expr::Div(l, r) => {
            let l = rewrite_even_power_of_atom(ctx, l, atom, sq_repl)?;
            let r = rewrite_even_power_of_atom(ctx, r, atom, sq_repl)?;
            Some(ctx.add(Expr::Div(l, r)))
        }
        Expr::Neg(i) => {
            let i = rewrite_even_power_of_atom(ctx, i, atom, sq_repl)?;
            Some(ctx.add(Expr::Neg(i)))
        }
        // A leaf that is not `atom` (and not a Pow of it): keep as-is. A different function carrying the
        // argument would leave the poly-in-single-atom check to fail downstream.
        _ => Some(expr),
    }
}

/// If `diff` is a polynomial in BOTH `sin(g)` and `cos(g)` where one of them occurs only to EVEN powers,
/// eliminate it via the Pythagorean identity (`cos² = 1 − sin²` or `sin² = 1 − cos²`) to obtain a
/// single-atom polynomial, then solve. Handles `2·cos(x)² − sin(x) − 1 = 0` (and its double-angle twin
/// `cos(2x) = sin(x)`, whose simplified form is `2·cos(x)² − sin(x) − 1`). Returns `None` when neither
/// atom is purely even (e.g. a genuine `sin·cos` product).
fn try_solve_mixed_trig_via_pythagorean(
    simplifier: &mut Simplifier,
    diff: ExprId,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::BuiltinFn;
    let probe = find_trig_atom_containing_var(&simplifier.context, diff, var)?;
    let g = match simplifier.context.get(probe) {
        Expr::Function(_, args) if args.len() == 1 => args[0],
        _ => return None,
    };
    let sin_id = simplifier.context.builtin_id(BuiltinFn::Sin);
    let cos_id = simplifier.context.builtin_id(BuiltinFn::Cos);
    let sin_g = simplifier.context.add(Expr::Function(sin_id, vec![g]));
    let cos_g = simplifier.context.add(Expr::Function(cos_id, vec![g]));
    let two = simplifier.context.num(2);
    let one = simplifier.context.num(1);
    // `cos(g)² = 1 − sin(g)²` (eliminate an all-even `cos`).
    let sin_sq = simplifier.context.add(Expr::Pow(sin_g, two));
    let cos_repl = simplifier.context.add(Expr::Sub(one, sin_sq));
    if let Some(reduced) =
        rewrite_even_power_of_atom(&mut simplifier.context, diff, cos_g, cos_repl)
    {
        if let Some(set) = solve_polynomial_in_trig_from_diff(simplifier, reduced, var) {
            return Some(set);
        }
    }
    // `sin(g)² = 1 − cos(g)²` (eliminate an all-even `sin`).
    let one = simplifier.context.num(1);
    let two = simplifier.context.num(2);
    let cos_sq = simplifier.context.add(Expr::Pow(cos_g, two));
    let sin_repl = simplifier.context.add(Expr::Sub(one, cos_sq));
    if let Some(reduced) =
        rewrite_even_power_of_atom(&mut simplifier.context, diff, sin_g, sin_repl)
    {
        return solve_polynomial_in_trig_from_diff(simplifier, reduced, var);
    }
    None
}

/// Solve an EQUATION that is a polynomial of degree ≥ 2 in a single trig atom `sin(g)` / `cos(g)` /
/// `tan(g)` whose argument contains the variable (`2·sin(x)² − 3·sin(x) + 1 = 0`, a quadratic in
/// `sin(x)`). Substitute `u = trig(g)`, solve the polynomial in `u`, then back-substitute
/// `trig(g) = u_root` — each root finishes as the recursive solver's PERIODIC family (with the range
/// guard, so `sin(x) = 2` drops). Without this, the isolation path rewrites `sin²(x)` via the
/// double-angle identity (`cos(2x)`) and leaks an `arcsin(… − cos(2x) …)` residual.
fn try_solve_polynomial_in_trig(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    if eq.op != cas_ast::RelOp::Eq {
        return None;
    }
    // Try the RAW difference FIRST: simplifying would fold `sin²(x)` into `cos(2x)`, destroying the
    // polynomial-in-`sin(x)` structure (the reason this handler exists).
    let raw_diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    if let Some(set) = solve_polynomial_in_trig_from_diff(simplifier, raw_diff, var) {
        return Some(set);
    }
    // Fallback: the SIMPLIFIED difference. This is the DUAL case — a `cos(2x)` (double-angle) term folds
    // to `2·cos(x)² − 1`, turning `cos(2x) + 3·cos(x) + 2 = 0` (two atoms in the raw form) into the
    // single-atom polynomial `2·cos(x)² + 3·cos(x) + 1`. (When the raw form already succeeded — the
    // `sin²` case — we never reach here, so its structure is untouched.) The two-term `cos(2x) ± cos(x)`
    // instead simplifies to a PRODUCT and is solved by the product-equation path, not here.
    let (simplified_diff, _) = simplifier.simplify(raw_diff);
    if let Some(set) = solve_polynomial_in_trig_from_diff(simplifier, simplified_diff, var) {
        return Some(set);
    }
    // Last fallback: the simplified form mixes `sin(g)` and `cos(g)` (e.g. `cos(2x) − sin(x)` folds to
    // the MIXED `2·cos(x)² − sin(x) − 1`). If one atom is purely even, the Pythagorean identity reduces
    // it to a single-atom polynomial.
    try_solve_mixed_trig_via_pythagorean(simplifier, simplified_diff, var)
}

/// Classify a leaf as a (possibly coefficiented) BARE `sin(g)`/`cos(g)` whose argument carries `var`:
/// returns `(is_sin, coeff, g)` where `coeff` is the multiplicative coefficient (`1` if bare). Matches
/// `sin(g)`, `cos(g)`, `c·sin(g)`, `sin(g)·c` (with `c` free of `var`). `None` for anything else.
fn classify_linear_trig_leaf(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(bool, ExprId, ExprId)> {
    use cas_ast::BuiltinFn;
    use cas_solver_core::isolation_utils::contains_var;
    // Bare `sin(g)` / `cos(g)`.
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        let (fn_id, args) = (*fn_id, args.clone());
        if args.len() == 1 && contains_var(ctx, args[0], var) {
            if ctx.is_builtin(fn_id, BuiltinFn::Sin) {
                let one = ctx.num(1);
                return Some((true, one, args[0]));
            }
            if ctx.is_builtin(fn_id, BuiltinFn::Cos) {
                let one = ctx.num(1);
                return Some((false, one, args[0]));
            }
        }
        return None;
    }
    // `c · (coefficiented sin/cos)` with `c` free of `var`. The recursive call may itself carry an inner
    // coefficient (`2 · (√3 · cos(g))`), so MULTIPLY the outer factor by it — do not discard it (a bare
    // sin/cos returns inner coefficient `1`, so simple `c · sin(g)` is unaffected).
    if let Expr::Mul(l, r) = ctx.get(expr) {
        let (l, r) = (*l, *r);
        if !contains_var(ctx, l, var) {
            if let Some((is_sin, inner, g)) = classify_linear_trig_leaf(ctx, r, var) {
                let coeff = ctx.add(Expr::Mul(l, inner));
                return Some((is_sin, coeff, g));
            }
        }
        if !contains_var(ctx, r, var) {
            if let Some((is_sin, inner, g)) = classify_linear_trig_leaf(ctx, l, var) {
                let coeff = ctx.add(Expr::Mul(r, inner));
                return Some((is_sin, coeff, g));
            }
        }
    }
    None
}

/// Accumulate `expr` as a homogeneous linear combination `a·sin(g) + b·cos(g)`: fold each leaf's
/// coefficient into `a`/`b` (with the running `positive` sign) and enforce a single shared argument `g`.
/// `None` on any non-`{sin,cos}(g)` term (a constant, a different argument, or other var structure).
#[allow(clippy::too_many_arguments)]
fn accumulate_linear_sin_cos(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
    positive: bool,
    a: &mut ExprId,
    b: &mut ExprId,
    arg: &mut Option<ExprId>,
    found_sin: &mut bool,
    found_cos: &mut bool,
) -> Option<()> {
    use cas_ast::ordering::compare_expr;
    match ctx.get(expr).clone() {
        Expr::Add(l, r) => {
            accumulate_linear_sin_cos(ctx, l, var, positive, a, b, arg, found_sin, found_cos)?;
            accumulate_linear_sin_cos(ctx, r, var, positive, a, b, arg, found_sin, found_cos)
        }
        Expr::Sub(l, r) => {
            accumulate_linear_sin_cos(ctx, l, var, positive, a, b, arg, found_sin, found_cos)?;
            accumulate_linear_sin_cos(ctx, r, var, !positive, a, b, arg, found_sin, found_cos)
        }
        Expr::Neg(inner) => {
            accumulate_linear_sin_cos(ctx, inner, var, !positive, a, b, arg, found_sin, found_cos)
        }
        _ => {
            let Some((is_sin, coeff, g)) = classify_linear_trig_leaf(ctx, expr, var) else {
                // A var-free ZERO constant (the moved-over RHS `… − 0`) contributes nothing; a NONZERO
                // constant makes the equation inhomogeneous (`a·sin + b·cos = c`) — out of scope.
                if let Some(c) = cas_math::numeric_eval::as_rational_const(ctx, expr) {
                    if num_traits::Zero::is_zero(&c) {
                        return Some(());
                    }
                }
                return None;
            };
            match arg {
                Some(g0) => {
                    if compare_expr(ctx, *g0, g) != std::cmp::Ordering::Equal {
                        return None;
                    }
                }
                None => *arg = Some(g),
            }
            let signed = if positive {
                coeff
            } else {
                ctx.add(Expr::Neg(coeff))
            };
            if is_sin {
                *found_sin = true;
                *a = ctx.add(Expr::Add(*a, signed));
            } else {
                *found_cos = true;
                *b = ctx.add(Expr::Add(*b, signed));
            }
            Some(())
        }
    }
}

/// Solve a HOMOGENEOUS linear trig equation `a·sin(g) + b·cos(g) = 0` (a single shared argument `g`,
/// `a ≠ 0`) by dividing through by `cos(g)`: `tan(g) = −b/a`, handed to the periodic tan solver. When
/// `a ≠ 0` the points `cos(g) = 0` are never solutions (there `a·sin(g) = ±a ≠ 0`), so the division
/// loses nothing. Without this the isolation path leaks an `arcsin(cos(x)·…)` residual. Handles common
/// textbook forms `sin(x) = cos(x)` (→ `tan(x) = 1`), `√3·sin(x) − cos(x) = 0` (→ `tan(x) = 1/√3`),
/// and affine arguments `sin(2x) = cos(2x)`. The INHOMOGENEOUS `a·sin + b·cos = c` (`c ≠ 0`) is a
/// different (auxiliary-angle) reduction and declines here (a leftover constant term fails the collect).
fn try_solve_homogeneous_linear_trig(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::BuiltinFn;
    if eq.op != cas_ast::RelOp::Eq {
        return None;
    }
    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let mut a = simplifier.context.num(0);
    let mut b = simplifier.context.num(0);
    let mut arg = None;
    let (mut found_sin, mut found_cos) = (false, false);
    accumulate_linear_sin_cos(
        &mut simplifier.context,
        diff,
        var,
        true,
        &mut a,
        &mut b,
        &mut arg,
        &mut found_sin,
        &mut found_cos,
    )?;
    // Require a genuine `sin`+`cos` combination (a bare `sin(x) = 0` is owned by the periodic handler).
    if !found_sin || !found_cos {
        return None;
    }
    let g = arg?;
    // `a` must be nonzero: a RATIONAL-zero `a` (the sin terms cancelled) declines; an irrational — hence
    // nonzero — `a` proceeds. Dividing by a nonzero `a` is exactly the divide-by-`cos(g)` step.
    if let Some(av) = cas_math::numeric_eval::as_rational_const(&simplifier.context, a) {
        if num_traits::Zero::is_zero(&av) {
            return None;
        }
    }
    // `tan(g) = −b/a`.
    let neg_b = simplifier.context.add(Expr::Neg(b));
    let rhs = simplifier.context.add(Expr::Div(neg_b, a));
    let (rhs, _) = simplifier.simplify(rhs);
    let tan_id = simplifier.context.builtin_id(BuiltinFn::Tan);
    let tan_g = simplifier.context.add(Expr::Function(tan_id, vec![g]));
    let tan_eq = Equation {
        lhs: tan_g,
        rhs,
        op: cas_ast::RelOp::Eq,
    };
    let (sol, _) = crate::solver_entrypoints_solve::solve(&tan_eq, var, simplifier).ok()?;
    // Trust only a fully resolved periodic/discrete/empty answer (guard against a residual echo).
    match sol {
        SolutionSet::Periodic { .. } | SolutionSet::Discrete(_) | SolutionSet::Empty => Some(sol),
        _ => None,
    }
}

/// Accumulate `expr` as a linear combination `a·sin(g) + b·cos(g) + konst` (coefficients kept as
/// EXPRESSIONS, single shared argument `g`): like [`accumulate_linear_sin_cos`] but ALSO collecting a
/// (possibly nonzero) constant term into `konst`. Keeping the coefficients symbolic admits IRRATIONAL
/// ones (`√3·sin(x)`). `None` on any non-`{sin,cos}(g)`, non-constant term (a different argument or
/// other var structure).
#[allow(clippy::too_many_arguments)]
fn accumulate_linear_sin_cos_const(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
    positive: bool,
    a: &mut ExprId,
    b: &mut ExprId,
    konst: &mut ExprId,
    arg: &mut Option<ExprId>,
    found_sin: &mut bool,
    found_cos: &mut bool,
) -> Option<()> {
    use cas_ast::ordering::compare_expr;
    use cas_solver_core::isolation_utils::contains_var;
    match ctx.get(expr).clone() {
        Expr::Add(l, r) => {
            accumulate_linear_sin_cos_const(
                ctx, l, var, positive, a, b, konst, arg, found_sin, found_cos,
            )?;
            accumulate_linear_sin_cos_const(
                ctx, r, var, positive, a, b, konst, arg, found_sin, found_cos,
            )
        }
        Expr::Sub(l, r) => {
            accumulate_linear_sin_cos_const(
                ctx, l, var, positive, a, b, konst, arg, found_sin, found_cos,
            )?;
            accumulate_linear_sin_cos_const(
                ctx, r, var, !positive, a, b, konst, arg, found_sin, found_cos,
            )
        }
        Expr::Neg(inner) => accumulate_linear_sin_cos_const(
            ctx, inner, var, !positive, a, b, konst, arg, found_sin, found_cos,
        ),
        _ => {
            if let Some((is_sin, coeff, g)) = classify_linear_trig_leaf(ctx, expr, var) {
                match arg {
                    Some(g0) => {
                        if compare_expr(ctx, *g0, g) != std::cmp::Ordering::Equal {
                            return None;
                        }
                    }
                    None => *arg = Some(g),
                }
                let signed = if positive {
                    coeff
                } else {
                    ctx.add(Expr::Neg(coeff))
                };
                if is_sin {
                    *found_sin = true;
                    *a = ctx.add(Expr::Add(*a, signed));
                } else {
                    *found_cos = true;
                    *b = ctx.add(Expr::Add(*b, signed));
                }
                return Some(());
            }
            // Not a trig leaf: must be a `var`-free constant (else out of scope).
            if contains_var(ctx, expr, var) {
                return None;
            }
            *konst = if positive {
                ctx.add(Expr::Add(*konst, expr))
            } else {
                ctx.add(Expr::Sub(*konst, expr))
            };
            Some(())
        }
    }
}

/// Solve an INHOMOGENEOUS linear trig equation `a·sin(g) + b·cos(g) = c` (`c ≠ 0`, `a ≠ 0`) by the
/// auxiliary-angle method: `a·sin(g) + b·cos(g) = R·sin(g + φ)` with `R = √(a²+b²)` and `φ = arctan(b/a)`
/// (normalizing `a > 0`, so `cos φ = a/R > 0` fixes the quadrant), giving `sin(g + φ) = c/R` — dispatched
/// to the shifted-argument trig solver (full periodic family; `|c/R| > 1 ⇒ No solution` via the surd
/// range guard). Coefficients may be rational OR provable-sign surds (`√3·sin(x) + cos(x) = 1`). Without
/// this the isolation leaks an `arcsin(… − cos(g) …)` residual. Homogeneous `c = 0` is the tangent
/// reduction; a pure `sin`/`cos` is owned by the bare/shifted handlers.
fn try_solve_inhomogeneous_linear_trig(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::BuiltinFn;
    use cas_math::numeric_eval::as_rational_const;
    use num_rational::BigRational;
    use num_traits::{Signed, Zero};

    if eq.op != cas_ast::RelOp::Eq {
        return None;
    }
    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let mut a = simplifier.context.num(0);
    let mut b = simplifier.context.num(0);
    let mut konst = simplifier.context.num(0);
    let mut arg = None;
    let (mut found_sin, mut found_cos) = (false, false);
    accumulate_linear_sin_cos_const(
        &mut simplifier.context,
        diff,
        var,
        true,
        &mut a,
        &mut b,
        &mut konst,
        &mut arg,
        &mut found_sin,
        &mut found_cos,
    )?;
    if !found_sin || !found_cos {
        return None; // need a genuine sin+cos combination
    }
    let g = arg?;
    let (a, _) = simplifier.simplify(a);
    let (b, _) = simplifier.simplify(b);
    // `a·sin + b·cos + konst = 0` ⇒ `a·sin + b·cos = −konst = c`.
    let neg_konst = simplifier.context.add(Expr::Neg(konst));
    let (c, _) = simplifier.simplify(neg_konst);
    // `a ≠ 0` and `c ≠ 0` (the homogeneous `c = 0` is the tangent reduction's job).
    let is_zero = |ctx: &Context, e: ExprId| as_rational_const(ctx, e).is_some_and(|v| v.is_zero());
    if is_zero(&simplifier.context, a) || is_zero(&simplifier.context, c) {
        return None;
    }
    // Sign of `a`: rational directly, else an exact surd sign; unprovable ⇒ decline.
    let a_positive = match as_rational_const(&simplifier.context, a) {
        Some(v) => v.is_positive(),
        None => match cas_math::root_forms::provable_sign_vs_zero(&simplifier.context, a)? {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Less => false,
            std::cmp::Ordering::Equal => return None,
        },
    };
    // Normalize `a > 0` by flipping all three signs.
    let (a, b, c) = if a_positive {
        (a, b, c)
    } else {
        let na = simplifier.context.add(Expr::Neg(a));
        let nb = simplifier.context.add(Expr::Neg(b));
        let nc = simplifier.context.add(Expr::Neg(c));
        (
            simplifier.simplify(na).0,
            simplifier.simplify(nb).0,
            simplifier.simplify(nc).0,
        )
    };
    // `R = √(a²+b²)`, `φ = arctan(b/a)`; dispatch `sin(g + φ) = c/R`.
    let a2 = simplifier.context.add(Expr::Mul(a, a));
    let b2 = simplifier.context.add(Expr::Mul(b, b));
    let r2 = simplifier.context.add(Expr::Add(a2, b2));
    let (r2, _) = simplifier.simplify(r2);
    let half = simplifier
        .context
        .add(Expr::Number(BigRational::new(1.into(), 2.into())));
    let r_expr = simplifier.context.add(Expr::Pow(r2, half));
    let c_over_r = simplifier.context.add(Expr::Div(c, r_expr));
    // Simplify `c/R` so a perfect-square `R²` collapses (`2/√9 → 2/3`); otherwise the range guard
    // mis-reads the `√(perfect square)` as an irrational surd and can wrongly report No solution.
    let (c_over_r, _) = simplifier.simplify(c_over_r);
    let ba = simplifier.context.add(Expr::Div(b, a));
    let (ba, _) = simplifier.simplify(ba);
    let arctan_id = simplifier.context.builtin_id(BuiltinFn::Arctan);
    let phi = simplifier.context.add(Expr::Function(arctan_id, vec![ba]));
    let g_plus_phi = simplifier.context.add(Expr::Add(g, phi));
    let sin_id = simplifier.context.builtin_id(BuiltinFn::Sin);
    let sin_call = simplifier
        .context
        .add(Expr::Function(sin_id, vec![g_plus_phi]));
    let new_eq = Equation {
        lhs: sin_call,
        rhs: c_over_r,
        op: cas_ast::RelOp::Eq,
    };
    let (sol, _) = crate::solver_entrypoints_solve::solve(&new_eq, var, simplifier).ok()?;
    match sol {
        SolutionSet::Periodic { .. } | SolutionSet::Discrete(_) | SolutionSet::Empty => Some(sol),
        _ => None,
    }
}

/// Solve an equation that is a *Laurent* polynomial in an exponential atom `base^x` — one that mixes
/// `base^x` with its reciprocal `base^(−x)` (canonicalized to `1/base^x`), e.g. `e^x + e^(−x) = 2`,
/// `3^x + 3^(−x) = 2`, `2^x − 3 + 2^(1−x) = 0`. Substitute `u = base^x` (the existing detector +
/// pattern substitution maps `base^(k·x) → u^k` and `1/base^x → 1/u`), giving a RATIONAL function in
/// `u`; clear the `1/u^k` denominators by multiplying by `u^K` (minimal `K`) to get a polynomial, then
/// hand it to `solve_polynomial_in_atom`, which solves for `u` and back-substitutes `base^x = u_root`
/// (the exp domain drops `u ≤ 0`, so the spurious `u = 0` introduced by the clearing is discarded).
///
/// Without this the isolation path rewrites `e^x + e^(−x)` via the hyperbolic identity and then bails
/// with `función [cosh] no definida` (and the general-base forms bail with `Cannot isolate 'x'`). The
/// pure-positive-power case (`e^(2x) − 3·e^x + 2`, no reciprocal) is left to its existing owner: this
/// handler declines when the substitution is already a polynomial in `u` (no `1/u^k` to clear).
fn try_solve_exponential_reciprocal_polynomial(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use num_rational::BigRational;
    use std::collections::BTreeMap;

    if eq.op != cas_ast::RelOp::Eq {
        return None;
    }
    // Detect the single exponential atom `base^x`. Rejects mixed bases (owned by the base-normalization
    // handler) and any equation where `var` appears OUTSIDE an exponential (`x·e^x = 1` — Lambert-W).
    let atom = cas_solver_core::substitution::detect_exponential_substitution(
        &mut simplifier.context,
        eq.lhs,
        eq.rhs,
        var,
        true,
    )?;
    let base = match simplifier.context.get(atom) {
        Expr::Pow(b, _) => *b,
        _ => return None,
    };
    // Collect the Laurent map `k → coeff` of `Σ coeff·base^(k·x)` from the RAW difference. We must NOT
    // simplify: `simplify` folds `e^x + e^(−x)` into `2·cosh(x)`, destroying the structure (and the
    // isolation path that inherits the cosh then bails `función [cosh] no definida`). Working on the raw
    // tree also means the reciprocal appears as `Pow(base, −x)`, which the map handles directly.
    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let mut map: BTreeMap<i64, BigRational> = BTreeMap::new();
    collect_exp_laurent_terms(&simplifier.context, diff, base, var, true, &mut map)?;
    map.retain(|_, c| !num_traits::Zero::is_zero(c));
    let min_k = *map.keys().next()?;
    let max_k = *map.keys().next_back()?;
    // Require a genuine reciprocal (`min_k < 0`): pure positive-power forms (`e^(2x) − 3·e^x + 2`) are
    // owned by the existing substitution path. Require span ≥ 2 so the shifted `u`-polynomial is at
    // least quadratic (a single exponential is owned by the simpler unwrap path).
    if min_k >= 0 || max_k - min_k < 2 {
        return None;
    }
    // Shift every exponent up by `−min_k` (multiply through by `base^(−min_k·x) > 0`, which loses no
    // real root) to get a polynomial in `u = base^x`: `Σ coeff·u^(k − min_k)`. Build it directly (no
    // `simplify`) and hand it to `solve_polynomial_in_atom`, which solves for `u` and back-substitutes
    // `base^x = u_root` (the exp domain drops `u ≤ 0`, discarding the spurious `u = 0` from the shift).
    let u_var = "__exp_u";
    let u = simplifier.context.var(u_var);
    let mut u_expr = simplifier.context.num(0);
    for (k, c) in &map {
        let coeff = simplifier.context.add(Expr::Number(c.clone()));
        let shift = simplifier.context.num(k - min_k);
        let power = simplifier.context.add(Expr::Pow(u, shift));
        let term = simplifier.context.add(Expr::Mul(coeff, power));
        u_expr = simplifier.context.add(Expr::Add(u_expr, term));
    }
    solve_polynomial_in_atom(simplifier, u_expr, u_var, var, atom)
}

/// Solve `|A(x)| = c` where `A` contains a trig atom and `c` is a nonnegative rational constant, by the
/// textbook split `A = c ∨ A = −c` — solving EACH branch with the full solver so a trig argument yields
/// its PERIODIC family, then unioning. The generic absolute-value isolation solves the branches to
/// PRINCIPAL roots only (`|2·sin(x) − 1| = 1 → {π/2, 0}` instead of `{π/2+2kπ} ∪ {kπ}`). Scoped to a
/// trig-bearing argument so the (correct) non-trig abs path is untouched; bare `|trig| = c` is already
/// handled earlier by the periodic-trig reduction.
fn try_solve_abs_of_trig_equation(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::{BuiltinFn, RelOp};
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;
    use num_rational::BigRational;
    use num_traits::Zero;

    if eq.op != RelOp::Eq {
        return None;
    }
    // LHS must be a unary `|A|` whose argument carries a trig atom in the variable.
    let arg = match simplifier.context.get(eq.lhs) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && simplifier.context.is_builtin(*fn_id, BuiltinFn::Abs) =>
        {
            args[0]
        }
        _ => return None,
    };
    // A non-trig `|A|` is already solved correctly by the existing path.
    find_trig_atom_containing_var(&simplifier.context, arg, var)?;
    if contains_var(&simplifier.context, eq.rhs, var) {
        return None; // a variable RHS needs the `rhs ≥ 0` guard machinery — leave it to the abs path
    }
    let c = as_rational_const(&simplifier.context, eq.rhs)?;
    if c < BigRational::zero() {
        return Some(SolutionSet::Empty); // |A| = negative ⇒ no solution
    }
    // Branches `A = c` and (for c > 0) `A = -c`, each solved fully so trig gives a periodic family.
    let mut branch_rhs = vec![eq.rhs];
    if !c.is_zero() {
        let neg = simplifier.context.add(Expr::Neg(eq.rhs));
        branch_rhs.push(simplifier.simplify(neg).0);
    }
    let mut branch_sets = Vec::with_capacity(branch_rhs.len());
    for rhs in branch_rhs {
        let branch_eq = Equation {
            lhs: arg,
            rhs,
            op: RelOp::Eq,
        };
        let (s, _) = crate::solver_entrypoints_solve::solve(&branch_eq, var, simplifier).ok()?;
        // A trig branch must resolve to a PERIODIC family (or `Empty` via the range guard). A `Discrete`
        // (or other) result means the branch solver returned PRINCIPAL roots — dropping periodicity
        // (e.g. `2·tan(x) − 1 = 1 → {π/4}`). Emitting a principal union would turn the existing honest
        // residual into a wrong answer, so decline and let the residual path own it.
        match s {
            SolutionSet::Periodic { .. } | SolutionSet::Empty => branch_sets.push(s),
            _ => return None,
        }
    }
    union_branch_solutions(simplifier, branch_sets)
}

/// Solve a polynomial-in-`ln(arg)` INEQUALITY `P(ln(x)) {op} 0` (`ln(x)^2 - 3·ln(x) + 2 < 0`, the
/// pure-square `ln(x)^2 - 4 < 0`, …) which the isolation path mis-reported as "No solution". Substitute
/// `u = ln(arg)`, solve the polynomial inequality `P(u) {op} 0` for the u-set, then map each u-interval
/// back through `ln` (a strictly increasing bijection `(0,∞) → ℝ`): `a < ln(x) < b ⟺ e^a < x < e^b`,
/// done by solving the single-`ln` bound relations and intersecting/uniting.
fn try_solve_polynomial_in_log_inequality(
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
fn try_solve_rational_power_polynomial_inequality(
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

/// Return a `sin(arg)` / `cos(arg)` / `tan(arg)` subexpression whose argument contains `var` (the
/// substitution atom for [`try_solve_polynomial_in_trig`]), searching the whole tree, or None.
fn find_trig_atom_containing_var(ctx: &Context, expr: ExprId, var: &str) -> Option<ExprId> {
    use cas_ast::BuiltinFn;
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if args.len() == 1
            && (ctx.is_builtin(*fn_id, BuiltinFn::Sin)
                || ctx.is_builtin(*fn_id, BuiltinFn::Cos)
                || ctx.is_builtin(*fn_id, BuiltinFn::Tan))
            && expr_contains_named_var(ctx, args[0], var)
        {
            return Some(expr);
        }
    }
    match ctx.get(expr).clone() {
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            find_trig_atom_containing_var(ctx, l, var)
                .or_else(|| find_trig_atom_containing_var(ctx, r, var))
        }
        Expr::Neg(inner) | Expr::Hold(inner) => find_trig_atom_containing_var(ctx, inner, var),
        Expr::Function(_, args) => args
            .iter()
            .find_map(|&a| find_trig_atom_containing_var(ctx, a, var)),
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
        // Thread the solve value domain into the simplifier's sticky state so
        // solver-internal plain `simplify()` calls fold complex forms
        // (`√(-3) → i·√3`) under ComplexEnabled. Save/restore: the sticky
        // domain must not leak past this solve invocation.
        let previous_domain = simplifier.set_sticky_value_domain(opts.value_domain);
        let result = solve_local_core(eq, var, simplifier, opts, ctx).map(|(set, steps)| {
            // For a NON-STRICT inequality (`f ≤ 0` / `f ≥ 0`) EVERY real root of `f = lhs − rhs` is
            // a solution (`0` satisfies `≤ 0` and `≥ 0`), but the interval sign-analysis drops
            // isolated roots of even-multiplicity factors (`(x−2)²(x+1) ≤ 0` keeps `(−∞,−1]` but
            // loses `{2}`; `x²/(x−1) ≥ 0` keeps `(1,∞)` but loses `{0}`). Union those roots back in
            // — they exclude poles by construction (a pole is not a root of `f = 0`) and are
            // domain-filtered.
            let set = union_non_strict_inequality_roots(eq, var, simplifier, opts, ctx, set);
            (set, steps)
        });
        simplifier.set_sticky_value_domain(previous_domain);
        result
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
        eq.op.clone(),
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
fn try_solve_boundary_trig_inequality(
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

/// True when `e` is an AFFINE function of `var` — a degree-1 polynomial `a·x + b` (`x`, `x-1`,
/// `2x+3`). The non-monotonicity of a fractional power is invariant under such a shift/scale, so
/// `(x-1)^(2/3)` is a symmetric valley exactly like `x^(2/3)`.
fn is_affine_degree_one(ctx: &Context, e: ExprId, var: &str) -> bool {
    cas_math::polynomial::Polynomial::from_expr(ctx, e, var)
        .map(|p| p.degree() == 1)
        .unwrap_or(false)
}

/// Net exponent of `var` when `e` is a single power term `c·(α)^k` of an AFFINE argument `α = a·x + b`
/// (`x`, `x-1`, `2x+3`), possibly with a constant coefficient, an additive constant (`x^(2/3) + 1`), a
/// quotient form (the simplifier rewrites `1/x^(1/3)` to `x^(2/3)/x`, net `−1/3`), or a `sqrt`
/// (`= ^(1/2)`). Returns `None` for anything that is not a single power of one affine argument (sums of
/// two powers, two distinct radicals, a non-affine base). The coefficient and the additive constant are
/// irrelevant — only the exponent decides monotonicity — so they are not returned.
fn pure_power_monomial_exponent(
    ctx: &Context,
    e: ExprId,
    var: &str,
) -> Option<num_rational::BigRational> {
    use cas_ast::BuiltinFn;
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;
    use num_rational::BigRational;
    use num_traits::{One, Zero};
    match ctx.get(e) {
        Expr::Variable(s) if ctx.sym_name(*s) == var => Some(BigRational::one()),
        Expr::Neg(inner) => pure_power_monomial_exponent(ctx, *inner, var),
        // Peel an additive constant: only one side carries the variable (`x^(2/3) + 1`, `5 - x^(2/3)`).
        Expr::Add(l, r) | Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            match (contains_var(ctx, l, var), contains_var(ctx, r, var)) {
                (true, false) => pure_power_monomial_exponent(ctx, l, var),
                (false, true) => pure_power_monomial_exponent(ctx, r, var),
                _ => None,
            }
        }
        Expr::Pow(base, exp) => {
            let (base, exp) = (*base, *exp);
            let k = as_rational_const(ctx, exp)?;
            // The base is a power-monomial in `var` (recurse) OR an affine argument `a·x + b`, which
            // contributes exponent 1 — so `(x-1)^(2/3)` is a valley exactly like `x^(2/3)`.
            let base_exp = pure_power_monomial_exponent(ctx, base, var)
                .or_else(|| is_affine_degree_one(ctx, base, var).then(BigRational::one))?;
            Some(base_exp * k)
        }
        // `sqrt(α)` of an affine argument is `α^(1/2)` (the simplifier keeps it as a `Sqrt` call, not a
        // `Pow(·, 1/2)`, so `1/sqrt(x)` is `Div(1, Sqrt(x))`).
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Sqrt))
                && is_affine_degree_one(ctx, args[0], var) =>
        {
            Some(BigRational::new(1.into(), 2.into()))
        }
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            match (contains_var(ctx, l, var), contains_var(ctx, r, var)) {
                (true, false) => pure_power_monomial_exponent(ctx, l, var),
                (false, true) => pure_power_monomial_exponent(ctx, r, var),
                (true, true) => {
                    let a = pure_power_monomial_exponent(ctx, l, var)?;
                    let b = pure_power_monomial_exponent(ctx, r, var)?;
                    Some(a + b)
                }
                (false, false) => None,
            }
        }
        Expr::Div(num, den) => {
            let (num, den) = (*num, *den);
            let n = if contains_var(ctx, num, var) {
                pure_power_monomial_exponent(ctx, num, var)?
            } else {
                BigRational::zero()
            };
            let d = if contains_var(ctx, den, var) {
                pure_power_monomial_exponent(ctx, den, var)?
            } else {
                BigRational::zero()
            };
            Some(n - d)
        }
        // A bare affine argument `x - 1` (exponent 1, an integer — never declined, but lets a `Pow`
        // base / `Div` operand recurse uniformly).
        _ if is_affine_degree_one(ctx, e, var) => Some(BigRational::one()),
        _ => None,
    }
}

/// Decompose `e` into `coeff·(α)^exp + addconst` where `α` is an AFFINE function of `var` (`a·x + b`),
/// `coeff`/`addconst` are rational constants, and `exp` is a rational constant. Returns
/// `(coeff, α, exp, addconst)`. Handles a leading coefficient, an additive constant on either side, and
/// `Neg`. Returns `None` for anything else (a sum of two powers, a non-affine base, …).
fn extract_affine_power_term(
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

/// Solve an even-numerator VALLEY power inequality `c·(a·x+b)^(p/q) + d {op} k` exactly (p EVEN,
/// e = p/q > 0). Since `(α)^(p/q) = |α|^(p/q)` and that is increasing in `|α|`, the relation reduces to
/// `|α| {op'} ((k−d)/c)^(q/p)` (op' flips when c < 0), which splits into two linear pieces of the affine
/// argument. The reciprocal valleys (`e < 0`) are left to the decline. Returns `None` for the
/// non-valley shapes so the surrounding dispatch keeps its other behaviour.
fn try_solve_even_power_valley_inequality(
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
fn try_decline_unsound_power_monomial_inequality(
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

/// Solve a two-term exponential equation with DIFFERENT effective bases: `A·M^x + B·N^x = 0` (`M ≠ N`,
/// both positive rationals, no constant term) ⟺ `(M/N)^x = −B/A`, i.e. `x = ln(−B/A) / ln(M/N)`. Covers
/// `4^x − 9^x = 0` (→ `x = 0`), `5·2^x = 3^x`, `2·4^x = 3·9^x` — which otherwise error with "Cannot
/// isolate 'x'" once moved to one side / coefficiented (the A=B forms happen to isolate, but the
/// one-sided forms do not). `(M/N)^x > 0`, so `−B/A ≤ 0` ⇒ NO solution. Same-base forms (`M = N`) are
/// the single-base polynomial path's job and decline here.
fn try_solve_two_different_base_exponential_equation(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use num_traits::Signed;
    let is_inequality = matches!(
        eq.op,
        cas_ast::RelOp::Lt | cas_ast::RelOp::Leq | cas_ast::RelOp::Gt | cas_ast::RelOp::Geq
    );
    if eq.op != cas_ast::RelOp::Eq && !is_inequality {
        return None;
    }
    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let mut terms = Vec::new();
    collect_exponential_base_terms(&simplifier.context, diff, var, true, &mut terms)?;
    if terms.len() != 2 {
        return None;
    }
    let (m, a) = terms[0].clone();
    let (n, b) = terms[1].clone();
    if m == n || num_traits::Zero::is_zero(&a) || num_traits::Zero::is_zero(&b) {
        return None;
    }
    if is_inequality {
        // Scout family B: `A·M^x + B·N^x ⋚ 0` used to fall through to the
        // boundary-equation path, which asserted the ROOT as the solution set
        // (`2^x > 3^x → {0}`, where `>` is false). Divide by `N^x > 0` — the
        // operator is preserved — and hand the single-atom relation
        // `A·(M/N)^x + B ⋚ 0` to the single-exponential path, which handles
        // every base (including 0 < M/N < 1 flips) and threshold correctly.
        let t = m / &n;
        let t_expr = simplifier.context.add(Expr::Number(t));
        let x_expr = simplifier.context.var(var);
        let atom = simplifier.context.add(Expr::Pow(t_expr, x_expr));
        let a_expr = simplifier.context.add(Expr::Number(a));
        let b_expr = simplifier.context.add(Expr::Number(b));
        let scaled = simplifier.context.add(Expr::Mul(a_expr, atom));
        let lhs = simplifier.context.add(Expr::Add(scaled, b_expr));
        let zero = simplifier.context.num(0);
        let reduced = Equation {
            lhs,
            rhs: zero,
            op: eq.op.clone(),
        };
        let (set, _) = crate::solver_entrypoints_solve::solve(&reduced, var, simplifier).ok()?;
        return Some(set);
    }
    // `(M/N)^x = −B/A`; the LHS is strictly positive, so a non-positive ratio has no real solution.
    let ratio = -b / &a;
    if !ratio.is_positive() {
        return Some(SolutionSet::Empty);
    }
    let mn = m / &n;
    // `x = ln(ratio) / ln(M/N)` (well-defined: `M/N > 0`, `M/N ≠ 1` since `M ≠ N`).
    let ratio_expr = simplifier.context.add(Expr::Number(ratio));
    let mn_expr = simplifier.context.add(Expr::Number(mn));
    let ln_ratio = simplifier.context.call("ln", vec![ratio_expr]);
    let ln_mn = simplifier.context.call("ln", vec![mn_expr]);
    let x = simplifier.context.add(Expr::Div(ln_ratio, ln_mn));
    let (x, _) = simplifier.simplify(x);
    Some(SolutionSet::Discrete(vec![x]))
}

/// Solve an exponential equation/inequality whose terms use DIFFERENT integer bases that are powers of
/// a common prime (`4^x − 3·2^x + 2 = 0`): rewrite every `m^g` to `p^(k·g)` (`4^x → 2^(2x)`) so the
/// whole thing is a polynomial in the single atom `p^x`, then solve the normalized relation. Without
/// this, the isolation reports "Cannot isolate: variable appears on both sides" (two distinct bases).
fn try_solve_via_exp_base_normalization(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    let mut bases = Vec::new();
    collect_exp_integer_bases(&simplifier.context, eq.lhs, var, &mut bases);
    collect_exp_integer_bases(&simplifier.context, eq.rhs, var, &mut bases);
    if bases.len() < 2 {
        return None; // one base (or none): already the normal path's job — avoids a rewrite loop
    }
    // Every base must be a power of a SINGLE common prime `p`.
    let (p, _) = integer_prime_power(&bases[0])?;
    for b in &bases[1..] {
        let (q, _) = integer_prime_power(b)?;
        if q != p {
            return None; // e.g. {4, 9}: 2 vs 3 — no common base
        }
    }
    let lhs = rewrite_exp_bases_to_prime(&mut simplifier.context, eq.lhs, var, &p);
    let rhs = rewrite_exp_bases_to_prime(&mut simplifier.context, eq.rhs, var, &p);
    let (lhs, _) = simplifier.simplify(lhs);
    let (rhs, _) = simplifier.simplify(rhs);
    let new_eq = Equation {
        lhs,
        rhs,
        op: eq.op.clone(),
    };
    // The rewritten relation has a single base `p`, so this handler declines on re-entry (no loop).
    let (set, _) = crate::solver_entrypoints_solve::solve(&new_eq, var, simplifier).ok()?;
    Some(set)
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
/// `A/|g(x)| ⋚ c` (A a nonzero rational constant, c constant): rewrite to the
/// exact twin `|A/g| ⋚ c` (A > 0; the A < 0 case negates and flips), which the
/// abs-threshold handler solves with correct pole puncturing — `1/|x| > 2` →
/// `(−1/2, 0) ∪ (0, 1/2)`, `1/|x| > 0` → `ℝ \ {0}`.
fn try_solve_reciprocal_abs_inequality(
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

/// `c / g(x) {op} 0` with a nonzero RATIONAL constant `c`, `0` on the RHS, and a
/// denominator `g` that CONTAINS an absolute value (`1/(|x|−1) < 0`,
/// `5/(|x−3|−1) > 0`, `1/(|x|+1) > 0`). The value `c/g` is never zero and shares
/// `g`'s sign, so `c/g {op} 0 ⟺ g {op'} 0` with a STRICT `op'` (the pole `g = 0`
/// is excluded even for `≤/≥`, since the value is undefined there rather than 0).
///
/// The bare `A/|g|` handler above matches only a lone `abs(g)` denominator, and
/// the affine `c/g {op} 0` reducer (`try_solve_const_over_surd_affine_inequality`)
/// requires `g` AFFINE, so `|x|−1` (abs minus a constant) falls to the generic
/// rational-inequality path, which cannot find `g`'s zeros through the abs and
/// returns garbage (`1/(|x|−1) < 0 → ℝ`; `> 0 → (−∞,−∞)∪(∞,∞)`). Reduce and
/// delegate to the abs solver, which handles `|x|−1 {op} 0` correctly.
///
/// Gated to a denominator that actually contains an abs: affine and rational
/// denominators keep their existing, already-correct owners (no huella change).
fn try_solve_const_over_abs_denominator_vs_zero(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    use cas_math::const_sign::{provable_const_sign, ConstSign};
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;
    use num_traits::Zero;
    use std::cmp::Ordering;

    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    // The RHS must be exactly 0 (the `k ≠ 0` reciprocal forms are owned elsewhere).
    let k = as_rational_const(&simplifier.context, eq.rhs)?;
    if !k.is_zero() {
        return None;
    }
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
    // Only the numerator's SIGN matters for the reduction. Decide it EXACTLY via the
    // shared const-sign chokepoint: a rational directly, else a linear surd
    // (`√2`, `−√2`) via `provable_sign_vs_zero`, else a transcendental constant
    // (`e−3`, `π`) via `provable_const_sign`. A zero or undecidable numerator declines.
    let mut num_sign = as_rational_const(&simplifier.context, num)
        .map(|c| c.cmp(&num_rational::BigRational::from_integer(0.into())))
        .or_else(|| cas_math::root_forms::provable_sign_vs_zero(&simplifier.context, num))
        .or_else(|| {
            Some(match provable_const_sign(&simplifier.context, num)? {
                ConstSign::Negative => Ordering::Less,
                ConstSign::Zero => Ordering::Equal,
                ConstSign::Positive => Ordering::Greater,
            })
        })?;
    if num_sign == Ordering::Equal {
        return None;
    }
    if neg {
        num_sign = num_sign.reverse();
    }
    if !contains_var(&simplifier.context, den, var) {
        return None;
    }
    // Restrict to denominators that CONTAIN an abs of the variable — the broken
    // family. Affine/rational denominators already reduce correctly elsewhere.
    let mut abs_terms: Vec<ExprId> = Vec::new();
    collect_abs_of_var(&simplifier.context, den, var, &mut abs_terms);
    if abs_terms.is_empty() {
        return None;
    }

    // `c/g {op} 0 ⟺ g {op'} 0`, `op'` STRICT: the value is never 0, so `≤/≥`
    // collapse to `</>`, and the pole `g = 0` (undefined value) stays excluded.
    let op_is_upper = matches!(eq.op, RelOp::Gt | RelOp::Geq);
    let num_is_positive = num_sign == Ordering::Greater;
    let den_op = if op_is_upper == num_is_positive {
        RelOp::Gt
    } else {
        RelOp::Lt
    };
    let zero = simplifier.context.num(0);
    solve_relation_set(simplifier, var, den, zero, den_op)
}

/// `f(x)/g(x) {op} k` with a NONZERO rational `k`, where the quotient is NOT purely
/// rational (an abs/ln/log leaf on either side: `1/(|x|−1) > 1`, `1/ln(x) > 2`,
/// `|x|/(x−2) < 1`): split on the denominator sign.
/// `f/g {op} k ⟺ (f − k·g)/g {op} 0`, so under `g > 0` the relation is `p {op} 0`
/// (`p = f − k·g`) and under `g < 0` it flips; the pole `g = 0` stays excluded by the
/// strict sign cases, while non-strict boundaries (`p = 0`, where `f/g = k` exactly)
/// survive inside each case. A quotient of two POLYNOMIALS is owned by the rational
/// inequality path (correct, runs later) and stays declined here; the naive legacy
/// isolation this replaces multiplied by `g` without casing and returned the single
/// naive interval between boundary roots (or collapsed the whole relation to its
/// boundary equation: `|x|/(x−2) < 1` → "No solution").
fn try_solve_division_vs_const_sign_split(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    use cas_math::numeric_eval::as_rational_const;
    use cas_math::polynomial::Polynomial;
    use cas_solver_core::isolation_utils::{contains_var, flip_inequality};
    use cas_solver_core::solution_set::{intersect_solution_sets, union_solution_sets};
    use num_traits::Zero;

    if !matches!(eq.op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq) {
        return None;
    }
    // Nonzero rational threshold only: the vs-zero form has its own strict reduction.
    let k = as_rational_const(&simplifier.context, eq.rhs)?;
    if k.is_zero() {
        return None;
    }
    // Peel negations into the numerator; expect `Div(num, den)` underneath.
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
    // The sign-split needs a variable-carrying denominator (a constant denominator is
    // ordinary isolation), and a quotient of two polynomials stays with the rational
    // owner — claim only the forms it cannot parse (an abs/ln leaf on either side).
    if !contains_var(&simplifier.context, den, var) {
        return None;
    }
    if Polynomial::from_expr(&simplifier.context, num, var).is_ok()
        && Polynomial::from_expr(&simplifier.context, den, var).is_ok()
    {
        return None;
    }

    let num_eff = if neg {
        simplifier.context.add(Expr::Neg(num))
    } else {
        num
    };
    let k_expr = simplifier.context.add(Expr::Number(k));
    let k_den = simplifier.context.add(Expr::Mul(k_expr, den));
    let p = simplifier.context.add(Expr::Sub(num_eff, k_den));
    let zero = simplifier.context.num(0);

    // Every sub-solve must land on an interval/point set the exact set algebra can
    // combine; anything else (residual, conditional, periodic) declines honestly.
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
    let den_pos = solve_relation_set(simplifier, var, den, zero, RelOp::Gt)?;
    let den_neg = solve_relation_set(simplifier, var, den, zero, RelOp::Lt)?;
    let p_same = solve_relation_set(simplifier, var, p, zero, eq.op.clone())?;
    let p_flip = solve_relation_set(simplifier, var, p, zero, flip_inequality(eq.op.clone()))?;
    if !(interval_like(&den_pos)
        && interval_like(&den_neg)
        && interval_like(&p_same)
        && interval_like(&p_flip))
    {
        return None;
    }
    let case_pos = intersect_solution_sets(&simplifier.context, p_same, den_pos);
    let case_neg = intersect_solution_sets(&simplifier.context, p_flip, den_neg);
    Some(union_solution_sets(&simplifier.context, case_pos, case_neg))
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
fn try_solve_trig_sum_to_product_equation(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::BuiltinFn;
    use cas_ast::RelOp;
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;
    use num_traits::Zero;

    if eq.op != RelOp::Eq {
        return None;
    }
    let as_sin_cos = |ctx: &Context, e: ExprId| -> Option<(BuiltinFn, ExprId)> {
        match ctx.get(e) {
            Expr::Function(fn_id, args) if args.len() == 1 => {
                let b = ctx.builtin_of(*fn_id)?;
                if matches!(b, BuiltinFn::Sin | BuiltinFn::Cos) {
                    Some((b, args[0]))
                } else {
                    None
                }
            }
            _ => None,
        }
    };
    // Normalize to a DIFFERENCE `trig(u) − trig(v)`: either one call on each side,
    // or `call ± call = 0` (the `+` flips through `sin(−v) = −sin v`; for cos the
    // sum needs its own identity, handled by the `is_sum` flag).
    let (builtin, u, v, is_sum) = if let (Some((bl, u)), Some((br, v))) = (
        as_sin_cos(&simplifier.context, eq.lhs),
        as_sin_cos(&simplifier.context, eq.rhs),
    ) {
        if bl != br {
            return None;
        }
        (bl, u, v, false)
    } else {
        let rhs_zero = as_rational_const(&simplifier.context, eq.rhs)
            .map(|q| q.is_zero())
            .unwrap_or(false);
        if !rhs_zero {
            return None;
        }
        match simplifier.context.get(eq.lhs).clone() {
            Expr::Sub(l, r) => {
                let (bl, u) = as_sin_cos(&simplifier.context, l)?;
                let (br, v) = as_sin_cos(&simplifier.context, r)?;
                if bl != br {
                    return None;
                }
                (bl, u, v, false)
            }
            Expr::Add(l, r) => {
                let (bl, u) = as_sin_cos(&simplifier.context, l)?;
                let (br, v) = as_sin_cos(&simplifier.context, r)?;
                if bl != br {
                    return None;
                }
                (bl, u, v, true)
            }
            _ => return None,
        }
    };
    if !contains_var(&simplifier.context, u, var) || !contains_var(&simplifier.context, v, var) {
        return None;
    }
    if cas_ast::ordering::compare_expr(&simplifier.context, u, v) == std::cmp::Ordering::Equal {
        return None; // identity (0 = 0): owned by the var-eliminated pipeline
    }
    // Half-sum and half-difference, folded.
    let two = simplifier.context.num(2);
    let sum = simplifier.context.add(Expr::Add(u, v));
    let half_sum = simplifier.context.add(Expr::Div(sum, two));
    let half_sum = simplifier.simplify(half_sum).0;
    let diff = simplifier.context.add(Expr::Sub(u, v));
    let half_diff = simplifier.context.add(Expr::Div(diff, two));
    let half_diff = simplifier.simplify(half_diff).0;
    // Product factors (constants dropped: only the zero set matters):
    //   sin u − sin v = 2·cos(hs)·sin(hd)      sin u + sin v = 2·sin(hs)·cos(hd)
    //   cos u − cos v = −2·sin(hs)·sin(hd)     cos u + cos v = 2·cos(hs)·cos(hd)
    let (f1, f2) = match (builtin, is_sum) {
        (BuiltinFn::Sin, false) => (
            simplifier.context.call("cos", vec![half_sum]),
            simplifier.context.call("sin", vec![half_diff]),
        ),
        (BuiltinFn::Sin, true) => (
            simplifier.context.call("sin", vec![half_sum]),
            simplifier.context.call("cos", vec![half_diff]),
        ),
        (BuiltinFn::Cos, false) => (
            simplifier.context.call("sin", vec![half_sum]),
            simplifier.context.call("sin", vec![half_diff]),
        ),
        (BuiltinFn::Cos, true) => (
            simplifier.context.call("cos", vec![half_sum]),
            simplifier.context.call("cos", vec![half_diff]),
        ),
        _ => return None,
    };
    // Solve each factor's zero set DIRECTLY at top level (each factor is a single
    // bare trig call, so the periodic solver returns its full family) and union the
    // two families. Solving the whole `Mul(f1, f2)` instead lets `simplify` fold a
    // negative half-difference (`sin(−x/2)`) into a top-level `Neg(f1·f2)`, whose
    // recursive re-solve lacks the periodic-product recovery and collapses to a
    // single factor's `{0}` — the orientation-specific defect. Factor-wise solve
    // sidesteps the fold entirely.
    let zero = simplifier.context.num(0);
    let s1 = solve_relation_set(simplifier, var, f1, zero, RelOp::Eq)?;
    let zero2 = simplifier.context.num(0);
    let s2 = solve_relation_set(simplifier, var, f2, zero2, RelOp::Eq)?;
    let resolved = |s: &SolutionSet| {
        matches!(
            s,
            SolutionSet::Discrete(_)
                | SolutionSet::Empty
                | SolutionSet::AllReals
                | SolutionSet::Continuous(_)
                | SolutionSet::Union(_)
                | SolutionSet::Periodic { .. }
        )
    };
    if !resolved(&s1) || !resolved(&s2) {
        return None; // a residual/conditional factor declines to the honest residual
    }
    union_branch_solutions(simplifier, vec![s1, s2])
}

/// NESTED abs relation — an `abs` whose argument contains another `abs` with an
/// AFFINE argument (`||x|−2| {op} x`): partition ℝ at the zeros of the INNER abs
/// arguments, substitute `|u| → ±u` per region (the regional relation reduces to
/// a plain abs relation the existing owners solve), clip each result to its
/// region and union. Every dedicated abs handler declines the nested shape
/// (their `Polynomial::from_expr` gates fail on the interior abs), so it fell to
/// the generic isolation, whose inner sub-solves came back as UNRESOLVED
/// `Conditional` sets that the outer union swallowed: `||x|−2| > x` reported
/// "No solution" for a truth of `(−∞, 1)` — for every relation direction.
/// Deeper nesting recurses naturally: the regional solve re-enters the full
/// solver, which fires this handler again on the next level.
fn try_solve_nested_abs_relation(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    use cas_math::polynomial::Polynomial;
    use cas_solver_core::isolation_utils::contains_var;
    use cas_solver_core::solution_set::{intersect_solution_sets, union_solution_sets};
    use num_rational::BigRational;
    use num_traits::{One, Signed, Zero};

    if !matches!(
        eq.op,
        RelOp::Eq | RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq
    ) {
        return None;
    }
    // Collect abs nodes that sit INSIDE another abs (the inner layer of nesting).
    fn collect_inner_abs(ctx: &Context, expr: ExprId, inside_abs: bool, out: &mut Vec<ExprId>) {
        match ctx.get(expr).clone() {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                collect_inner_abs(ctx, l, inside_abs, out);
                collect_inner_abs(ctx, r, inside_abs, out);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => collect_inner_abs(ctx, inner, inside_abs, out),
            Expr::Function(fn_id, args) => {
                let is_abs = args.len() == 1 && ctx.is_builtin(fn_id, cas_ast::BuiltinFn::Abs);
                if is_abs && inside_abs && !out.contains(&expr) {
                    out.push(expr);
                }
                for arg in args {
                    collect_inner_abs(ctx, arg, inside_abs || is_abs, out);
                }
            }
            _ => {}
        }
    }
    let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    let (diff, _) = simplifier.simplify(diff);
    let mut inner_abs: Vec<ExprId> = Vec::new();
    collect_inner_abs(&simplifier.context, diff, false, &mut inner_abs);
    if inner_abs.is_empty() {
        return None;
    }
    // Claim only the VARIABLE-remainder family (`||x|-2| {op} x`): with a constant
    // remainder (`||x|-2| = 1`) the existing nested-vs-constant owner is already
    // correct, and keeps its pinned root ordering. Discriminate by zeroing every
    // OUTERMOST abs in the difference and checking the leftover for the variable.
    fn collect_outer_abs(ctx: &Context, expr: ExprId, out: &mut Vec<ExprId>) {
        match ctx.get(expr).clone() {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                collect_outer_abs(ctx, l, out);
                collect_outer_abs(ctx, r, out);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => collect_outer_abs(ctx, inner, out),
            Expr::Function(fn_id, args) => {
                if args.len() == 1 && ctx.is_builtin(fn_id, cas_ast::BuiltinFn::Abs) {
                    if !out.contains(&expr) {
                        out.push(expr);
                    }
                } else {
                    for arg in args {
                        collect_outer_abs(ctx, arg, out);
                    }
                }
            }
            _ => {}
        }
    }
    let mut outer_abs: Vec<ExprId> = Vec::new();
    collect_outer_abs(&simplifier.context, diff, &mut outer_abs);
    let zero_probe = simplifier.context.num(0);
    let mut remainder = diff;
    for &abs_e in &outer_abs {
        remainder = substitute_expr_by_id(&mut simplifier.context, remainder, abs_e, zero_probe);
    }
    let (remainder, _) = simplifier.simplify(remainder);
    // Claim the VARIABLE-remainder family (`||x|-2| {op} x`: remainder has the var) AND the
    // abs-vs-abs / nested-abs-sum family (`||x|-5| = |x|`: two outermost abs carry the var, so
    // zeroing them leaves a var-free `0` remainder — but both sides genuinely depend on the var).
    // Decline ONLY when the remainder is var-free AND fewer than two outermost abs carry the var —
    // that is the nested-vs-CONSTANT case (`||x|-2| = 1`), whose existing owner keeps its pinned
    // root ordering.
    let var_outer_abs = outer_abs
        .iter()
        .filter(|&&a| contains_var(&simplifier.context, a, var))
        .count();
    if var_outer_abs < 2 && !contains_var(&simplifier.context, remainder, var) {
        return None;
    }
    // Every inner abs argument must be AFFINE in the variable (rational breakpoint).
    let mut breakpoints: Vec<BigRational> = Vec::new();
    let mut arg_polys: Vec<Polynomial> = Vec::new();
    let mut args: Vec<ExprId> = Vec::new();
    for &abs_e in &inner_abs {
        let arg = match simplifier.context.get(abs_e) {
            Expr::Function(_, a) if a.len() == 1 => a[0],
            _ => return None,
        };
        if !contains_var(&simplifier.context, arg, var) {
            return None;
        }
        let poly = Polynomial::from_expr(&simplifier.context, arg, var).ok()?;
        if poly.degree() != 1 {
            return None;
        }
        let a = poly
            .coeffs
            .get(1)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let b = poly
            .coeffs
            .first()
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if a.is_zero() {
            return None;
        }
        breakpoints.push(-b / a);
        arg_polys.push(poly);
        args.push(arg);
    }
    breakpoints.sort();
    breakpoints.dedup();
    let n = breakpoints.len();
    let two = BigRational::from_integer(2.into());
    let zero = simplifier.context.num(0);

    // Closed segment `[lo, hi]` as a solution set (open end = no constraint).
    let segment_set = |simplifier: &mut Simplifier,
                       lo: Option<&BigRational>,
                       hi: Option<&BigRational>|
     -> Option<SolutionSet> {
        let x = simplifier.context.var(var);
        let lo_set = match lo {
            Some(l) => {
                let ln = simplifier.context.add(Expr::Number(l.clone()));
                solve_relation_set(simplifier, var, x, ln, RelOp::Geq)?
            }
            None => SolutionSet::AllReals,
        };
        let hi_set = match hi {
            Some(h) => {
                let hn = simplifier.context.add(Expr::Number(h.clone()));
                solve_relation_set(simplifier, var, x, hn, RelOp::Leq)?
            }
            None => SolutionSet::AllReals,
        };
        Some(intersect_solution_sets(&simplifier.context, lo_set, hi_set))
    };

    let mut solution = SolutionSet::Empty;
    for seg_idx in 0..=n {
        let (lo, hi, test): (Option<BigRational>, Option<BigRational>, BigRational) =
            if seg_idx == 0 {
                let a0 = breakpoints[0].clone();
                let t = &a0 - BigRational::one();
                (None, Some(a0), t)
            } else if seg_idx == n {
                let an = breakpoints[n - 1].clone();
                let t = &an + BigRational::one();
                (Some(an), None, t)
            } else {
                let al = breakpoints[seg_idx - 1].clone();
                let ar = breakpoints[seg_idx].clone();
                let t = (&al + &ar) / &two;
                (Some(al), Some(ar), t)
            };
        // Resolve each inner `|u| → sign·u` by the argument's sign at the test point.
        let mut seg_expr = diff;
        for (i, &abs_e) in inner_abs.iter().enumerate() {
            let val = arg_polys[i].eval(&test);
            let replacement = if val.is_positive() {
                args[i]
            } else {
                simplifier.context.add(Expr::Neg(args[i]))
            };
            seg_expr = substitute_expr_by_id(&mut simplifier.context, seg_expr, abs_e, replacement);
        }
        let (seg_expr, _) = simplifier.simplify(seg_expr);
        let branch = solve_relation_set(simplifier, var, seg_expr, zero, eq.op.clone())?;
        // An unresolved sub-solve must DECLINE the whole relation — the set algebra
        // silently swallows non-concrete operands (the swallowed-Conditional was
        // this family's root cause).
        if !is_concrete_solution_set(&branch) {
            return None;
        }
        let seg_set = segment_set(simplifier, lo.as_ref(), hi.as_ref())?;
        let clipped = intersect_solution_sets(&simplifier.context, branch, seg_set);
        solution = union_solution_sets(&simplifier.context, solution, clipped);
    }
    Some(solution)
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
fn try_solve_abs_vs_abs_polynomial_inequality(
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
fn try_solve_product_inequality_sign_split(
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

/// Rational coefficient of an additive term that is a RATIONAL multiple of the bare variable
/// (`x` -> 1, `k·x`/`x·k` -> k, `x/k` -> 1/k, `Neg(t)` -> -coeff). Returns `None` if the term is not
/// a rational multiple of `x` alone (e.g. `x^2`, `x·y`, `a·x` with symbolic `a`).
fn rational_coeff_of_bare_var(
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

/// Accumulate `expr` into an affine form `k·x + Σ const_terms` in `var`, walking `Add`/`Sub`/`Neg`
/// so signed constant parts and a `Sub`-form variable term (`a - x`) are captured. `sign_positive`
/// carries the running sign of this subtree. Returns `false` (caller bails) if the variable appears
/// in a non-rational-linear position. Const terms are recorded with their sign for exact rebuild.
fn collect_affine_terms_in_var(
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
fn symbolic_positive_affine_arg_of_var(
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

fn positive_affine_arg_of_var(
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

/// Decompose `expr == A·trig(arg) + B` where `trig` is a SINGLE bare `Sin`/`Cos`/`Tan` call containing
/// the variable, `A` (≠ 0) and `B` are rational constants, and every other additive part is var-free.
/// Returns `(trig_call, A, B)`, or `None` when `expr` is ALREADY the bare trig call (nothing to peel —
/// `detect` handles that directly) or when the side is not affine in exactly one trig term.
fn peel_affine_trig(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, num_rational::BigRational, ExprId)> {
    use num_traits::Zero;
    let mut trig: Option<(ExprId, num_rational::BigRational)> = None;
    let mut offset = ctx.num(0);
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

/// Accumulate `expr` (scaled by `scale`) as `A·trig(arg) + B`: a constant leaf adds `scale·leaf` to the
/// SYMBOLIC `offset` (so a SURD offset like `−√3` is kept, not just a rational — the reason
/// `2·cos(x) − √3 = 0` used to fall through to the principal-root isolation), a rational `Mul`/`Div`
/// scales, `Add`/`Sub`/`Neg` recurse, and a single bare trig call of the variable is recorded with its
/// accumulated (rational) coefficient. A second trig term, a trig×trig product, or any other
/// var-bearing shape declines (`None`).
fn accumulate_affine_trig(
    ctx: &mut Context,
    expr: ExprId,
    scale: &num_rational::BigRational,
    var: &str,
    trig: &mut Option<(ExprId, num_rational::BigRational)>,
    offset: &mut ExprId,
) -> Option<()> {
    use cas_ast::BuiltinFn;
    use cas_solver_core::isolation_utils::contains_var;
    use num_traits::Zero;
    if !contains_var(ctx, expr, var) {
        // A var-free leaf is a constant offset term — keep it symbolically (surd/π allowed).
        let scale_node = ctx.add(Expr::Number(scale.clone()));
        let scaled = ctx.add(Expr::Mul(scale_node, expr));
        *offset = ctx.add(Expr::Add(*offset, scaled));
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

/// Peel a leading rational coefficient (including `Neg` as `-1` and nested `Mul`s of constants) off
/// `e`, returning `(coefficient, core)` with `e = coefficient · core`. `cos(x)^2 - 1` simplifies to
/// `-(sin(x)^2)`, so a `Neg` wrapper must be peeled for the squared-trig detector to see the trig.
fn peel_rational_coefficient(ctx: &Context, e: ExprId) -> (num_rational::BigRational, ExprId) {
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

/// `(builtin, arg)` if `e` is a single `sin`/`cos`/`tan` of an expression containing `var`.
fn trig_call_arg(ctx: &Context, e: ExprId, var: &str) -> Option<(cas_ast::BuiltinFn, ExprId)> {
    use cas_ast::BuiltinFn;
    use cas_solver_core::isolation_utils::contains_var;
    if let Expr::Function(fn_id, args) = ctx.get(e) {
        if args.len() == 1 && contains_var(ctx, args[0], var) {
            if let Some(b) = ctx.builtin_of(*fn_id) {
                if matches!(b, BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Tan) {
                    return Some((b, args[0]));
                }
            }
        }
    }
    None
}

/// When the equation `e = 0` reduces to `trig(arg) = 0`, return `(trig_builtin, arg)`. A power
/// `c·trig(arg)^n` (n ≥ 2) is zero iff the trig is zero. A quotient `c·trig(arg)^n / d` is zero where
/// its NUMERATOR is — and a numerator zero is a genuine solution only where the denominator does not
/// also vanish, so the quotient form fires ONLY when the denominator is a power of the COMPLEMENTARY
/// trig of the same argument (`sin`/`cos` zeros are disjoint), e.g. `sin·tan = sin²/cos`.
fn reduces_to_trig_zero(
    ctx: &Context,
    e: ExprId,
    var: &str,
) -> Option<(cas_ast::BuiltinFn, ExprId)> {
    use cas_ast::BuiltinFn;
    use cas_math::numeric_eval::as_rational_const;
    use num_rational::BigRational;
    use num_traits::Zero;
    let (coeff, core) = peel_rational_coefficient(ctx, e);
    if coeff.is_zero() {
        return None;
    }
    match ctx.get(core) {
        Expr::Pow(base, exp) => {
            let n = as_rational_const(ctx, *exp)?;
            if !n.is_integer() || n < BigRational::from_integer(2.into()) {
                return None;
            }
            trig_call_arg(ctx, *base, var)
        }
        Expr::Div(num, den) => {
            let (num_coeff, num_core) = peel_rational_coefficient(ctx, *num);
            if num_coeff.is_zero() {
                return None;
            }
            let (f, arg) = match ctx.get(num_core) {
                Expr::Pow(base, _) => trig_call_arg(ctx, *base, var)?,
                _ => trig_call_arg(ctx, num_core, var)?,
            };
            let (_, den_core) = peel_rational_coefficient(ctx, *den);
            let (g, arg2) = match ctx.get(den_core) {
                Expr::Pow(base, _) => trig_call_arg(ctx, *base, var)?,
                _ => trig_call_arg(ctx, den_core, var)?,
            };
            let complement = match f {
                BuiltinFn::Sin => BuiltinFn::Cos,
                BuiltinFn::Cos => BuiltinFn::Sin,
                _ => return None,
            };
            (g == complement
                && cas_ast::ordering::compare_expr(ctx, arg, arg2) == std::cmp::Ordering::Equal)
                .then_some((f, arg))
        }
        _ => None,
    }
}

/// Position of a `sin`/`cos` RHS `c` relative to the unit interval, decided EXACTLY.
enum TrigUnitClass {
    Zero,       // c = 0
    Unit,       // |c| = 1
    InOpen,     // 0 < |c| < 1
    OutOfRange, // |c| > 1 (no real solution)
}

/// `c = ±q^e` with `q` a NON-NEGATIVE rational and `e` a POSITIVE rational. Returns `(q, neg)`. Since
/// `q^e` is increasing in `q` for `e > 0`, `q^e {<,=,>} 1 ⟺ q {<,=,>} 1` and `q^e = 0 ⟺ q = 0` — so
/// the magnitude class only needs `q` vs `{0, 1}`. Covers the `n`-th roots `(1/4)^(1/4)`, `4^(1/4)`
/// the even-power reduction produces (which `as_linear_surd` — quadratic surds only — does not).
fn as_nonneg_power_magnitude(
    ctx: &Context,
    c: ExprId,
) -> Option<(num_rational::BigRational, bool)> {
    use cas_math::numeric_eval::as_rational_const;
    use num_traits::Signed;
    match ctx.get(c) {
        Expr::Neg(inner) => {
            let (q, neg) = as_nonneg_power_magnitude(ctx, *inner)?;
            Some((q, !neg))
        }
        Expr::Pow(base, exp) => {
            let q = as_rational_const(ctx, *base)?;
            let e = as_rational_const(ctx, *exp)?;
            if q.is_negative() || !e.is_positive() {
                return None;
            }
            Some((q, false))
        }
        _ => None,
    }
}

/// Classify a constant `sin`/`cos` RHS `c` EXACTLY (never f64): a quadratic surd `a + b·√n` via
/// `linear_surd_sign`, or an `n`-th root `±q^e` via `as_nonneg_power_magnitude`. `None` for a
/// transcendental / unrecognised constant (declines).
fn classify_trig_unit_rhs(ctx: &Context, c: ExprId) -> Option<TrigUnitClass> {
    use cas_math::root_forms::as_linear_surd;
    use num_rational::BigRational;
    use num_traits::{One, Zero};
    use std::cmp::Ordering;
    let one = BigRational::one();
    if let Some((a, b, n)) = as_linear_surd(ctx, c) {
        let sign_c = linear_surd_sign(&a, &b, &n);
        let vs_upper = linear_surd_sign(&(&a - &one), &b, &n);
        let vs_lower = linear_surd_sign(&(&a + &one), &b, &n);
        return Some(
            if vs_upper == Ordering::Greater || vs_lower == Ordering::Less {
                TrigUnitClass::OutOfRange
            } else if vs_upper == Ordering::Equal || vs_lower == Ordering::Equal {
                TrigUnitClass::Unit
            } else if sign_c == Ordering::Equal {
                TrigUnitClass::Zero
            } else {
                TrigUnitClass::InOpen
            },
        );
    }
    if let Some((q, _neg)) = as_nonneg_power_magnitude(ctx, c) {
        return Some(if q.is_zero() {
            TrigUnitClass::Zero
        } else {
            match q.cmp(&one) {
                Ordering::Greater => TrigUnitClass::OutOfRange,
                Ordering::Equal => TrigUnitClass::Unit,
                Ordering::Less => TrigUnitClass::InOpen,
            }
        });
    }
    // Fallback: EXACT rational value-bounds decide the named/transcendental constants the two
    // recognizers above miss — `phi` (the simplifier folds `(1+√5)/2` into the named constant,
    // which `as_linear_surd` cannot see), `e`, `π/4`, `1/e`, `e − 2`. An open interval can only
    // prove STRICT classifications: strictly outside `[−1, 1]` ⇒ OutOfRange, strictly inside
    // `(−1, 1)` with a proven sign ⇒ InOpen. Equality with `{−1, 0, 1}` is unprovable from
    // bounds, so any interval touching those points stays `None` (honest decline, as before).
    if let Some((lo, hi)) = cas_math::const_sign::const_value_bounds(ctx, c) {
        let neg_one = -one.clone();
        if lo > one || hi < neg_one {
            return Some(TrigUnitClass::OutOfRange);
        }
        let zero = BigRational::zero();
        if lo > neg_one && hi < one && (lo > zero || hi < zero) {
            return Some(TrigUnitClass::InOpen);
        }
    }
    None
}

/// Exact sign of a constant `c` versus 0 (`Less`/`Equal`/`Greater`) when `c` is a rational or quadratic
/// surd; `None` if not so reducible. Used to branch `trig^n = c` / `|trig| = c` on the sign of `c`
/// while ALSO accepting a SURD `c` (e.g. `|cos(x)| = √2/2`), which `as_rational_const` rejects.
fn const_sign_vs_zero(ctx: &Context, c: ExprId) -> Option<std::cmp::Ordering> {
    let (a, b, n) = cas_math::root_forms::as_linear_surd(ctx, c)?;
    Some(linear_surd_sign(&a, &b, &n))
}

/// Solve `trig(arg) = value  ∨  trig(arg) = −value` and UNION the periodic families — the reduction
/// target of `trig(arg)^n = c` (n even) and `|trig(arg)| = c`. An out-of-range side solves to `Empty`
/// and is dropped; both empty ⇒ `Empty`; one family ⇒ that family; two ⇒ the merged periodic union.
/// Returns `None` if either side does not solve to a clean `Periodic`/`Empty` (so the caller declines).
fn solve_trig_equals_plus_minus(
    simplifier: &mut Simplifier,
    trig_call: ExprId,
    value: ExprId,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    let neg_value = simplifier.context.add(Expr::Neg(value));
    let (neg_value, _) = simplifier.simplify(neg_value);
    let mut families: Vec<(Vec<ExprId>, ExprId)> = Vec::new();
    for rhs in [value, neg_value] {
        let eq = Equation {
            lhs: trig_call,
            rhs,
            op: RelOp::Eq,
        };
        match try_solve_periodic_trig_equation(&eq, var, simplifier) {
            Some(SolutionSet::Periodic { bases, period }) => families.push((bases, period)),
            Some(SolutionSet::Empty) => {} // out of range — contributes nothing
            _ => return None,
        }
    }
    match families.len() {
        0 => Some(SolutionSet::Empty),
        1 => {
            let (bases, period) = families.pop().unwrap();
            Some(SolutionSet::Periodic { bases, period })
        }
        _ => union_periodic_families_over_common_period(simplifier, families),
    }
}

/// `A·sin/cos(g(x)) ⋚ c` where `|c/A| ≥ 1`: the bounded range of sin/cos
/// settles the relation exactly — attained boundaries reduce to the periodic
/// EQUATION (owned by `try_solve_periodic_trig_equation`, multiple-angle
/// capable), unattainable ones to ∅/ℝ. `|c/A| < 1` returns None (honest
/// decline: the answer is a periodic union of intervals the SolutionSet
/// cannot yet represent).
fn try_solve_trig_weak_boundary_inequality(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<SolutionSet> {
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
) -> Option<SolutionSet> {
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
                boundary_equation(simplifier, 1) // t ≥ 1 ⇔ t = 1
            } else if r > one {
                Some(SolutionSet::Empty)
            } else {
                Some(SolutionSet::AllReals) // r ≤ −1: always true
            }
        }
        RelOp::Gt => {
            if r >= one {
                Some(SolutionSet::Empty) // t > 1 (or more) is unattainable
            } else if r < -&one {
                Some(SolutionSet::AllReals)
            } else {
                // r = −1: the complement ℝ ∖ {touch points} — the table with
                // r = −1 yields exactly the punctured line (len == period,
                // both ends open): sin u > −1 → (−π/2, 3π/2).
                try_emit_trig_interior_interval_union(simplifier, trig_fn, arg, &r, &op, var)
            }
        }
        RelOp::Leq => {
            if r == -&one {
                boundary_equation(simplifier, -1) // t ≤ −1 ⇔ t = −1
            } else if r < -&one {
                Some(SolutionSet::Empty)
            } else {
                Some(SolutionSet::AllReals) // r ≥ 1: always true
            }
        }
        RelOp::Lt => {
            if r <= -&one {
                Some(SolutionSet::Empty)
            } else if r > one {
                Some(SolutionSet::AllReals)
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
) -> Option<SolutionSet> {
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

    Some(SolutionSet::PeriodicIntervalUnion {
        windows: vec![window],
        period,
    })
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
fn try_solve_even_power_or_abs_trig_inequality(
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
                    if matches!(f, BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Tan) {
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
            return None;
        };
    let c_val = as_rational_const(&simplifier.context, c_expr)?;

    // Divide by A: `trig² ⋚ t` (square) or `|trig| ⋚ t` (abs), flipping for A<0.
    let t = c_val / &a_coeff;
    let op = if a_coeff.is_negative() {
        flip_inequality(op)
    } else {
        op
    };

    // Reduce to `|trig| ⋚ r` with r ≥ 0. For the square, r = √t; the
    // non-positive-threshold edges settle by the sign of a square / abs.
    let zero = BigRational::zero();
    if t < zero {
        // trig² (or |trig|) ≥ 0 > t everywhere it is defined.
        return match op {
            RelOp::Lt | RelOp::Leq => Some(SolutionSet::Empty),
            // `> t` / `≥ t` for t < 0 is always true — but tan is undefined
            // at its poles, so only the bounded sin/cos are unconditionally ℝ.
            RelOp::Gt | RelOp::Geq if matches!(trig_fn, BuiltinFn::Sin | BuiltinFn::Cos) => {
                Some(SolutionSet::AllReals)
            }
            _ => None,
        };
    }
    if t == zero {
        match op {
            RelOp::Lt => return Some(SolutionSet::Empty), // trig² < 0 impossible
            RelOp::Leq => return None, // trig² ≤ 0 ⟺ trig = 0, a point set — decline
            // trig² ≥ 0 is always true for the bounded sin/cos; tan is
            // punctured at its poles, so decline there.
            RelOp::Geq if matches!(trig_fn, BuiltinFn::Sin | BuiltinFn::Cos) => {
                return Some(SolutionSet::AllReals)
            }
            RelOp::Geq => return None,
            // trig² > 0 ⟺ trig ≠ 0: fall through to the r = 0 reduction
            // (`trig > 0 ∪ trig < 0` → the punctured line), NOT AllReals.
            RelOp::Gt => {}
            _ => return None,
        }
    }

    // r = √t for the square, r = t for the abs.
    let r_expr = if is_square {
        let t_expr = rational_to_expr(&mut simplifier.context, &t);
        let sqrt_call = simplifier.context.call("sqrt", vec![t_expr]);
        simplifier.simplify(sqrt_call).0
    } else {
        rational_to_expr(&mut simplifier.context, &t)
    };
    let neg_r_expr = {
        let neg = simplifier.context.add(Expr::Neg(r_expr));
        simplifier.simplify(neg).0
    };

    // `|trig| < r` ⟺ `trig > −r ∩ trig < r`; `> r` ⟺ `trig > r ∪ trig < −r`.
    let (conj, parts): (bool, [(RelOp, ExprId); 2]) = match op {
        RelOp::Lt => (true, [(RelOp::Gt, neg_r_expr), (RelOp::Lt, r_expr)]),
        RelOp::Leq => (true, [(RelOp::Geq, neg_r_expr), (RelOp::Leq, r_expr)]),
        RelOp::Gt => (false, [(RelOp::Gt, r_expr), (RelOp::Lt, neg_r_expr)]),
        RelOp::Geq => (false, [(RelOp::Geq, r_expr), (RelOp::Leq, neg_r_expr)]),
        _ => return None,
    };

    let mut acc: Option<SolutionSet> = None;
    for (sub_op, bound_expr) in parts {
        let trig_call = simplifier.context.call_builtin(trig_fn, vec![g]);
        let sub_eq = Equation {
            lhs: trig_call,
            rhs: bound_expr,
            op: sub_op,
        };
        let (set, _) = crate::solver_entrypoints_solve::solve(&sub_eq, var, simplifier).ok()?;
        if matches!(set, SolutionSet::Residual(_) | SolutionSet::Conditional(_)) {
            return None;
        }
        acc = Some(match acc {
            None => set,
            Some(prev) => combine_piu_sets(simplifier, prev, set, conj)?,
        });
    }
    acc
}

fn try_solve_reciprocal_trig_inequality(
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

    // Match `A · trig(g)^(−1)` or `A / trig(g)` (sin/cos/tan only).
    let recip_trig = |ctx_: &Context, e: ExprId| -> Option<(BigRational, BuiltinFn, ExprId)> {
        let (coeff, core) = peel_rational_coefficient(ctx_, e);
        if coeff.is_zero() {
            return None;
        }
        let trig_of = |ctx_: &Context, e2: ExprId| -> Option<(BuiltinFn, ExprId)> {
            if let Expr::Function(fn_id, args) = ctx_.get(e2) {
                if args.len() == 1 && contains_var(ctx_, args[0], var) {
                    if let Some(f) = ctx_.builtin_of(*fn_id) {
                        if matches!(f, BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Tan) {
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

/// Combine two sub-results where each is Empty/AllReals/PeriodicIntervalUnion;
/// PIU pairs go through the circular same-period algebra. Anything else
/// (mixed Periodic points, intervals) declines conservatively.
fn combine_piu_sets(
    simplifier: &mut Simplifier,
    s1: SolutionSet,
    s2: SolutionSet,
    intersect: bool,
) -> Option<SolutionSet> {
    use SolutionSet::{AllReals, Empty, PeriodicIntervalUnion};
    match (s1, s2) {
        (AllReals, s) | (s, AllReals) if intersect => Some(s),
        (AllReals, _) | (_, AllReals) => Some(AllReals),
        (Empty, _) | (_, Empty) if intersect => Some(Empty),
        (Empty, s) | (s, Empty) => Some(s),
        (
            PeriodicIntervalUnion {
                windows: w1,
                period: p1,
            },
            PeriodicIntervalUnion {
                windows: w2,
                period: p2,
            },
        ) => {
            if intersect {
                crate::periodic_interval_union::intersect_periodic_interval_unions_over_common_period(
                    simplifier, &w1, p1, &w2, p2,
                )
            } else {
                crate::periodic_interval_union::union_periodic_interval_unions_over_common_period(
                    simplifier, &w1, p1, &w2, p2,
                )
            }
        }
        _ => None,
    }
}

/// Build an exact rational constant expression.
fn rational_to_expr(ctx: &mut Context, r: &num_rational::BigRational) -> ExprId {
    ctx.add(Expr::Number(r.clone()))
}

/// Airbag: f64-sample `trig(g(x)) op r` at u-space fractions {1/8, 1/2, 7/8}
/// inside the window (must satisfy) and at ±width/8 outside (must not).
/// A sample is a CONTRADICTION only when the sign of `trig − r` disagrees by
/// more than `τ = 1e-9·max(1, |r|)`; unevaluable or |·| ≤ τ ⇒ inconclusive
/// (skipped). Returns false only on a genuine contradiction.
#[allow(clippy::too_many_arguments)]
fn interior_window_samples_consistent(
    simplifier: &mut Simplifier,
    trig_fn: cas_ast::BuiltinFn,
    arg: ExprId,
    r: &num_rational::BigRational,
    op: &cas_ast::RelOp,
    u_lo: ExprId,
    u_hi: ExprId,
    a: &num_rational::BigRational,
    var: &str,
) -> bool {
    use cas_ast::RelOp;
    use num_traits::ToPrimitive;
    let ctx = &simplifier.context;
    let empty: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let (Some(lo), Some(hi)) = (
        cas_math::evaluator_f64::eval_f64(ctx, u_lo, &empty),
        cas_math::evaluator_f64::eval_f64(ctx, u_hi, &empty),
    ) else {
        return true; // endpoints not numerically evaluable: inconclusive
    };
    let width = hi - lo;
    if !(width.is_finite() && width > 0.0) {
        return true;
    }
    let (Some(r_f), Some(a_f)) = (r.to_f64(), a.to_f64()) else {
        return true;
    };
    let b_f = {
        // g(0) = b numerically: evaluate arg at var = 0.
        let mut map = std::collections::HashMap::new();
        map.insert(var.to_string(), 0.0_f64);
        match cas_math::evaluator_f64::eval_f64(ctx, arg, &map) {
            Some(v) if v.is_finite() => v,
            _ => return true,
        }
    };
    let tau = 1e-9 * r_f.abs().max(1.0);
    let trig_eval = |u: f64| -> f64 {
        match trig_fn {
            cas_ast::BuiltinFn::Sin => u.sin(),
            cas_ast::BuiltinFn::Cos => u.cos(),
            cas_ast::BuiltinFn::Tan => u.tan(),
            _ => f64::NAN,
        }
    };
    let satisfies = |u: f64| -> Option<bool> {
        // Evaluate through x = (u−b)/a and back through the ORIGINAL arg to
        // exercise the same composition the solution set claims.
        let x = (u - b_f) / a_f;
        let mut map = std::collections::HashMap::new();
        map.insert(var.to_string(), x);
        let g_x = cas_math::evaluator_f64::eval_f64(&simplifier.context, arg, &map)?;
        let v = trig_eval(g_x);
        if !v.is_finite() {
            return None;
        }
        let d = v - r_f;
        if d.abs() <= tau {
            return None; // too close to the boundary: inconclusive
        }
        Some(match op {
            RelOp::Gt | RelOp::Geq => d > 0.0,
            RelOp::Lt | RelOp::Leq => d < 0.0,
            _ => return None,
        })
    };
    for frac in [0.125_f64, 0.5, 0.875] {
        if satisfies(lo + width * frac) == Some(false) {
            return false; // inside sample refutes the window
        }
    }
    // Punctured-line windows (len == period, |r| = 1) have a measure-zero
    // complement: any "outside" sample wraps into the set one period over
    // and would falsely refute. Skip the outside probes for them.
    let period_u = if matches!(trig_fn, cas_ast::BuiltinFn::Tan) {
        std::f64::consts::PI
    } else {
        2.0 * std::f64::consts::PI
    };
    if (width - period_u).abs() > 1e-9 {
        for outside in [lo - width * 0.125, hi + width * 0.125] {
            if satisfies(outside) == Some(true) {
                return false; // outside sample lands in the claimed complement
            }
        }
    }
    true
}

/// Multiple-angle EXPANSION rewrites (`sin(3x) → 3·sin(x) − 4·sin(x)³`, the
/// quintuple analogue, and the recursive expander) destroy the `trig(n·x) = c`
/// shape inside this handler's own simplifies BEFORE the periodic matcher can
/// see it — the polynomial fallback then asserts a finite arcsin set as the
/// complete answer (family-A soundness bug: `sin(3x)=1/2` lost `2kπ/3`
/// periodicity). Contractions (`Double/Triple Angle Contraction`) stay live —
/// they REBUILD the matchable shape (`sin·cos → sin(2x)/2`).
const MULTIPLE_ANGLE_EXPANSION_RULES: &[&str] = &[
    "Double Angle Identity",
    "Double Angle Expansion",
    "Triple Angle Identity",
    "Quintuple Angle Identity",
    "Recursive Trig Expansion",
];

/// Gated entry: disables the multiple-angle expansions for the duration of
/// the periodic-trig handler (re-entrant: recursive reductions re-enter here
/// and add nothing; only the outermost call restores).
/// `f(g(x)) = c` where `f` is an inverse-trig or hyperbolic function solved
/// by applying its (single, monotone) inverse — `arcsin(x)=c → x=sin(c)`,
/// `sinh(x)=c → x=asinh(c)`, `cosh(x)=c → x=±acosh(c)`. Each bounded-range
/// function is GATED by the exact const-decision layer: a threshold provably
/// outside the range is `No solution`, provably inside reduces and recurses,
/// undecidable declines. (`solve`'s root verification does NOT catch these
/// transcendental range violations, so the gate is mandatory — without it
/// `arcsin(x)=5` would leak the spurious `sin(5)`.)
fn try_solve_inverse_trig_hyperbolic_equation(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    use cas_math::const_sign::{provable_const_sign, ConstSign};
    use cas_solver_core::isolation_utils::contains_var;
    if eq.op != RelOp::Eq {
        return None;
    }
    // Normalize to `f(g) = c` with the call on the LHS and `c` var-free.
    let (call, c) = if contains_var(&simplifier.context, eq.lhs, var)
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
    // Peel a var-free RATIONAL multiplicative wrapper (`A·f(g)`, `f(g)/A`, `−f(g)`)
    // into the constant: `2·arcsin(x) = π/3` reduces to `arcsin(x) = π/6`. The
    // bare-Function match below is the historic coefficient≠1 blind spot — the
    // generic isolation peels the coefficient but builds an UNFOLDED `π/3/2` and
    // leaks the reduced equation un-dispatched (`UnaryInverseKind` has no arc/hyp
    // inverses). Each simplify folds the constant, so the range gates downstream
    // see a canonical threshold.
    let mut call = call;
    let mut c = c;
    loop {
        match simplifier.context.get(call).clone() {
            Expr::Neg(inner) => {
                call = inner;
                let neg = simplifier.context.add(Expr::Neg(c));
                c = simplifier.simplify(neg).0;
            }
            Expr::Mul(l, r) => {
                let (coef, inner) = if contains_var(&simplifier.context, r, var) {
                    (l, r)
                } else {
                    (r, l)
                };
                if contains_var(&simplifier.context, coef, var) {
                    return None;
                }
                match cas_math::numeric_eval::as_rational_const(&simplifier.context, coef) {
                    Some(q) if !num_traits::Zero::is_zero(&q) => {}
                    _ => return None, // non-rational/zero coefficient: not this family
                }
                let div = simplifier.context.add(Expr::Div(c, coef));
                c = simplifier.simplify(div).0;
                call = inner;
            }
            Expr::Div(num, den) => {
                if contains_var(&simplifier.context, den, var) {
                    return None;
                }
                match cas_math::numeric_eval::as_rational_const(&simplifier.context, den) {
                    Some(q) if !num_traits::Zero::is_zero(&q) => {}
                    _ => return None,
                }
                let mul = simplifier.context.add(Expr::Mul(c, den));
                c = simplifier.simplify(mul).0;
                call = num;
            }
            _ => break,
        }
    }
    let (fn_name, g) = match simplifier.context.get(call) {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            (simplifier.context.sym_name(*fn_id).to_string(), args[0])
        }
        _ => return None,
    };
    if !contains_var(&simplifier.context, g, var) {
        return None;
    }

    // Sign of `expr` (a var-free constant) via the exact decision layer.
    let const_sign = |simplifier: &Simplifier, expr: ExprId| -> Option<ConstSign> {
        if let Some(q) = cas_math::numeric_eval::as_rational_const(&simplifier.context, expr) {
            use num_traits::Zero;
            return Some(if q.is_zero() {
                ConstSign::Zero
            } else if q > num_rational::BigRational::zero() {
                ConstSign::Positive
            } else {
                ConstSign::Negative
            });
        }
        provable_const_sign(&simplifier.context, expr)
    };
    // Is `c` provably within `[lo, hi]` (each bound an ExprId)? Verdict:
    // Some(true) provably in, Some(false) provably out, None undecidable.
    let in_closed_range = |simplifier: &mut Simplifier, lo: ExprId, hi: ExprId| -> Option<bool> {
        let hi_minus_c = simplifier.context.add(Expr::Sub(hi, c));
        let hi_minus_c = simplifier.simplify(hi_minus_c).0;
        let c_minus_lo = simplifier.context.add(Expr::Sub(c, lo));
        let c_minus_lo = simplifier.simplify(c_minus_lo).0;
        let s_hi = const_sign(simplifier, hi_minus_c)?;
        let s_lo = const_sign(simplifier, c_minus_lo)?;
        if matches!(s_hi, ConstSign::Negative) || matches!(s_lo, ConstSign::Negative) {
            return Some(false);
        }
        Some(true)
    };

    let pi = simplifier
        .context
        .add(Expr::Constant(cas_ast::Constant::Pi));
    let two = simplifier.context.num(2);
    let half_pi = {
        let e = simplifier.context.add(Expr::Div(pi, two));
        simplifier.simplify(e).0
    };
    let neg_half_pi = {
        let e = simplifier.context.add(Expr::Neg(half_pi));
        simplifier.simplify(e).0
    };
    let zero = simplifier.context.num(0);
    let one = simplifier.context.num(1);
    let neg_one = simplifier.context.num(-1);

    // Reduce `g = forward(c)` and solve for x through the full pipeline.
    let reduce_and_solve = |simplifier: &mut Simplifier, forward: &str| -> Option<SolutionSet> {
        let target = simplifier.context.call(forward, vec![c]);
        let target = simplifier.simplify(target).0;
        let reduced = Equation {
            lhs: g,
            rhs: target,
            op: RelOp::Eq,
        };
        let (set, _) = crate::solver_entrypoints_solve::solve(&reduced, var, simplifier).ok()?;
        Some(set)
    };

    match fn_name.as_str() {
        // Bounded ranges: gate `c`, then apply the forward function.
        "arcsin" | "asin" => match in_closed_range(simplifier, neg_half_pi, half_pi)? {
            true => reduce_and_solve(simplifier, "sin"),
            false => Some(SolutionSet::Empty),
        },
        "arccos" | "acos" => match in_closed_range(simplifier, zero, pi)? {
            true => reduce_and_solve(simplifier, "cos"),
            false => Some(SolutionSet::Empty),
        },
        "arctan" | "atan" => match in_closed_range(simplifier, neg_half_pi, half_pi)? {
            // tan is undefined at ±π/2, but a rational/const c never equals
            // the transcendental π/2, so the closed check is exact here.
            true => reduce_and_solve(simplifier, "tan"),
            false => Some(SolutionSet::Empty),
        },
        // tanh's range is the OPEN (−1, 1): |c| = 1 has no real solution.
        "tanh" => {
            let hi_minus_c = simplifier.context.add(Expr::Sub(one, c));
            let hi_minus_c = simplifier.simplify(hi_minus_c).0;
            let c_minus_lo = simplifier.context.add(Expr::Sub(c, neg_one));
            let c_minus_lo = simplifier.simplify(c_minus_lo).0;
            match (
                const_sign(simplifier, hi_minus_c)?,
                const_sign(simplifier, c_minus_lo)?,
            ) {
                (ConstSign::Positive, ConstSign::Positive) => reduce_and_solve(simplifier, "atanh"),
                (ConstSign::Negative | ConstSign::Zero, _)
                | (_, ConstSign::Negative | ConstSign::Zero) => Some(SolutionSet::Empty),
            }
        }
        // sinh is a bijection ℝ→ℝ: unconditional.
        "sinh" => reduce_and_solve(simplifier, "asinh"),
        // Inverse hyperbolics as the OUTER function — the mirror of the
        // inverse-trig arms above. asinh: ℝ→ℝ and atanh: (−1,1)→ℝ are
        // bijections, so the forward function applies unconditionally
        // (`tanh(c) ∈ (−1,1)` always lands back in atanh's domain). acosh's
        // range is [0, ∞), so `acosh(x) = c` needs `c ≥ 0`; then the preimage
        // `x = cosh(c) ≥ 1` is single (acosh is the non-negative branch, not
        // even like the forward cosh).
        "asinh" | "arcsinh" => reduce_and_solve(simplifier, "sinh"),
        "atanh" | "arctanh" => reduce_and_solve(simplifier, "tanh"),
        "acosh" | "arccosh" => match const_sign(simplifier, c)? {
            ConstSign::Negative => Some(SolutionSet::Empty),
            _ => reduce_and_solve(simplifier, "cosh"),
        },
        // cosh(x)=c is even: c ≥ 1 → g = ±acosh(c) (two branches); c < 1 → ∅.
        "cosh" => {
            let c_minus_one = simplifier.context.add(Expr::Sub(c, one));
            let c_minus_one = simplifier.simplify(c_minus_one).0;
            match const_sign(simplifier, c_minus_one)? {
                ConstSign::Negative => Some(SolutionSet::Empty),
                _ => {
                    let acosh = simplifier.context.call("acosh", vec![c]);
                    let pos = simplifier.simplify(acosh).0;
                    let neg = {
                        let n = simplifier.context.add(Expr::Neg(pos));
                        simplifier.simplify(n).0
                    };
                    // The two branches coincide when acosh(c) = 0 (c = 1):
                    // solve one to avoid a duplicate `{0, 0}` root.
                    let targets: &[ExprId] =
                        if cas_ast::ordering::compare_expr(&simplifier.context, pos, neg)
                            == std::cmp::Ordering::Equal
                        {
                            &[pos]
                        } else {
                            &[pos, neg]
                        };
                    let mut acc: Option<SolutionSet> = None;
                    for &target in targets {
                        let reduced = Equation {
                            lhs: g,
                            rhs: target,
                            op: RelOp::Eq,
                        };
                        let (set, _) =
                            crate::solver_entrypoints_solve::solve(&reduced, var, simplifier)
                                .ok()?;
                        acc = Some(match acc {
                            None => set,
                            Some(prev) => cas_solver_core::solution_set::union_solution_sets(
                                &simplifier.context,
                                prev,
                                set,
                            ),
                        });
                    }
                    acc
                }
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
    try_solve_periodic_trig_equation_with_steps(eq, var, simplifier).map(|(set, _)| set)
}

/// Same as [`try_solve_periodic_trig_equation`], also returning the didactic
/// narration (`solve_steps`): the per-period roots and the periodic families.
/// Sub-uses that reduce OTHER problems to this solver (boundary inequalities,
/// trig products) call the plain wrapper and discard the narration — it only
/// surfaces when this solver answers the user's equation directly.
pub(crate) fn try_solve_periodic_trig_equation_with_steps(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Option<(SolutionSet, Vec<crate::SolveStep>)> {
    let mut added: Vec<&'static str> = Vec::new();
    for rule in MULTIPLE_ANGLE_EXPANSION_RULES {
        if !simplifier.is_rule_disabled(rule) {
            simplifier.disable_rule(rule);
            added.push(rule);
        }
    }
    let mut steps = Vec::new();
    let out = try_solve_periodic_trig_equation_ungated(eq, var, simplifier, &mut steps);
    for rule in added {
        simplifier.enable_rule(rule);
    }
    out.map(|set| (set, steps))
}

fn try_solve_periodic_trig_equation_ungated(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    steps_out: &mut Vec<crate::SolveStep>,
) -> Option<SolutionSet> {
    use cas_ast::{BuiltinFn, RelOp};
    use cas_solver_core::isolation_utils::contains_var;
    use num_rational::BigRational;
    use num_traits::{One, Zero};

    if !matches!(eq.op, RelOp::Eq) {
        return None;
    }
    // COT-SQUARE (re-cycle D, 2026-07-14 — F8 Layer-2): `cot(g)² = c` folds to a `cos = |sin|·√c`
    // shape whose principal-branch inversion fabricated a self-referential arccos tree (Layer-1
    // now declines it honestly, but the CORRECT answer is a periodic family). Reduce on the RAW
    // tree (simplify destroys the cot before this handler runs):
    //   cot(g)² = c  ⟺  cos²(g) = c·sin²(g), sin(g) ≠ 0  ⟺  sin²(g) = 1/(1+c)
    // — unconditionally sound for rational c ≥ 0 (1/(1+c) > 0 forces sin ≠ 0 at every solution,
    // so the cot-pole exclusion is automatic; c < 0 is Empty, a square cannot be negative). The
    // existing sin²-reducer below then emits the full family (`cot²=1 → sin²=1/2 → {π/4+kπ/2}`).
    {
        let bare_cot_square = |ctx: &Context, e: ExprId| -> Option<ExprId> {
            let Expr::Pow(base, exp) = ctx.get(e) else {
                return None;
            };
            let (base, exp) = (*base, *exp);
            if cas_math::numeric_eval::as_rational_const(ctx, exp)
                != Some(BigRational::from_integer(2.into()))
            {
                return None;
            }
            let Expr::Function(fn_id, args) = ctx.get(base) else {
                return None;
            };
            if args.len() == 1
                && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cot))
                && contains_var(ctx, args[0], var)
            {
                Some(args[0])
            } else {
                None
            }
        };
        // Direct `cot(g)² = c` or the shifted diff form `cot(g)² − c = 0`.
        let hit: Option<(ExprId, BigRational)> = if let Some(g) =
            bare_cot_square(&simplifier.context, eq.lhs)
        {
            cas_math::numeric_eval::as_rational_const(&simplifier.context, eq.rhs).map(|c| (g, c))
        } else if cas_math::expr_predicates::is_zero_expr(&simplifier.context, eq.rhs) {
            match simplifier.context.get(eq.lhs).clone() {
                Expr::Sub(l, r) => bare_cot_square(&simplifier.context, l).and_then(|g| {
                    cas_math::numeric_eval::as_rational_const(&simplifier.context, r)
                        .map(|c| (g, c))
                }),
                Expr::Add(l, r) => bare_cot_square(&simplifier.context, l).and_then(|g| {
                    cas_math::numeric_eval::as_rational_const(&simplifier.context, r)
                        .map(|c| (g, -c))
                }),
                _ => None,
            }
        } else {
            None
        };
        if let Some((g, c)) = hit {
            use num_traits::Signed as _;
            if c.is_negative() {
                return Some(SolutionSet::Empty);
            }
            let target = BigRational::one() / (BigRational::one() + &c);
            let sin_g = simplifier.context.call("sin", vec![g]);
            let two = simplifier.context.num(2);
            let sin_sq = simplifier.context.add(Expr::Pow(sin_g, two));
            let target_expr = simplifier.context.add(Expr::Number(target));
            let reduced = Equation {
                lhs: sin_sq,
                rhs: target_expr,
                op: RelOp::Eq,
            };
            return try_solve_periodic_trig_equation(&reduced, var, simplifier);
        }
    }
    let (lhs, _) = simplifier.simplify(eq.lhs);
    let (rhs, _) = simplifier.simplify(eq.rhs);

    // RECIPROCAL-SQUARE (F5 2026-07-13b): `A/trig(g)^2 = c` — sec(x)^2, csc(x)^2, 1/cos(x)^2,
    // 1/sin(x)^2, optionally shifted by a constant (`sec(x)^2 - 2 = 0`) — all canonicalize to
    // `Div(A, Pow(cos|sin(g), 2))`. The bare-squared reducer below never matches the `Div`, so the
    // generic isolation emits only the finite principal-value roots and DROPS the periodic family.
    // Invert to the equivalent `trig(g)^2 = -A/k` (from `A/trig^2 + k = 0`, k = the folded constant
    // remainder, k != 0) and recurse, so the existing double-angle reducer yields the full family.
    {
        use num_traits::Zero as _;
        // Match a term `±A/trig(g)^2`; returns `(Pow(trig(g),2), signed A)`.
        let recip_square = |ctx: &Context, term: ExprId| -> Option<(ExprId, BigRational)> {
            let (inner, sign) = match ctx.get(term) {
                Expr::Neg(i) => (*i, -BigRational::one()),
                _ => (term, BigRational::one()),
            };
            let Expr::Div(num, den) = ctx.get(inner) else {
                return None;
            };
            let (num, den) = (*num, *den);
            let a = cas_math::numeric_eval::as_rational_const(ctx, num)?;
            let Expr::Pow(base, exp) = ctx.get(den) else {
                return None;
            };
            let (base, exp) = (*base, *exp);
            if cas_math::numeric_eval::as_rational_const(ctx, exp)
                != Some(BigRational::from_integer(2.into()))
            {
                return None;
            }
            let Expr::Function(fn_id, args) = ctx.get(base) else {
                return None;
            };
            if args.len() != 1 || !contains_var(ctx, args[0], var) {
                return None;
            }
            match ctx.builtin_of(*fn_id) {
                Some(BuiltinFn::Sin | BuiltinFn::Cos) => Some((den, a * sign)),
                _ => None,
            }
        };
        let diff = simplifier.context.add(Expr::Sub(lhs, rhs));
        let (diff, _) = simplifier.simplify(diff);
        let mut sq: Option<(ExprId, BigRational)> = None;
        let mut k = BigRational::zero();
        let mut shape_ok = true;
        for term in cas_math::expr_nary::add_leaves(&simplifier.context, diff) {
            if let Some((pow, a)) = recip_square(&simplifier.context, term) {
                if sq.is_some() {
                    shape_ok = false;
                    break;
                }
                sq = Some((pow, a));
            } else if let Some(c) =
                cas_math::numeric_eval::as_rational_const(&simplifier.context, term)
            {
                k += c;
            } else {
                shape_ok = false;
                break;
            }
        }
        if shape_ok {
            if let Some((pow, a)) = sq {
                if !k.is_zero() {
                    let target = -a / k; // trig(g)^2 = -A/k
                    let target_expr = simplifier.context.add(Expr::Number(target));
                    let reduced = Equation {
                        lhs: pow,
                        rhs: target_expr,
                        op: RelOp::Eq,
                    };
                    return try_solve_periodic_trig_equation(&reduced, var, simplifier);
                }
            }
        }
    }

    // `sin(arg)^2 = c`  <=>  `cos(2·arg) = 1 - 2c` ;  `cos(arg)^2 = c`  <=>  `cos(2·arg) = 2c - 1`.
    // Reduce a squared bare trig to the double-angle cosine equation and recurse: the cos branch's
    // c ∈ {0, ±1} gate then maps EXACTLY the single-family cases (`sin(x)^2=1 → cos(2x)=-1 →
    // {π/2 + kπ}`) and declines the two-family ones (`sin(x)^2=1/4 → cos(2x)=1/2`, not in {0,±1}).
    // Peels an optional leading rational coefficient `A` so `A·trig(arg)^2` is recognised (not just a
    // bare `trig(arg)^2`); the coefficient is folded into the constant side as `c/A` below. Without it
    // `4·cos(x)^2 = 1` skipped the reduction and emitted only the two base roots — no `+kπ` family.
    let squared = |ctx: &Context, e: ExprId| -> Option<(BuiltinFn, ExprId, BigRational)> {
        let (coeff, core) = peel_rational_coefficient(ctx, e);
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
    // `A·sin(arg)·cos(arg) = c` ⇔ `sin(2·arg) = 2c/A` — the double-angle
    // contraction the default rewriter does not perform. Without it the
    // fixpoint isolation echoed `solve(x - arcsin(c/cos(x)) = 0)` as an
    // ok:true pseudo-result (scout family-A garbage case).
    {
        let sin_cos_product = |ctx: &Context, e: ExprId| -> Option<(ExprId, BigRational)> {
            let (coeff, core) = peel_rational_coefficient(ctx, e);
            if coeff.is_zero() {
                return None;
            }
            if let Expr::Mul(a, b) = ctx.get(core) {
                let (a, b) = (*a, *b);
                let as_trig = |x: ExprId| -> Option<(BuiltinFn, ExprId)> {
                    if let Expr::Function(fn_id, args) = ctx.get(x) {
                        if args.len() == 1 && contains_var(ctx, args[0], var) {
                            if let Some(f) = ctx.builtin_of(*fn_id) {
                                if matches!(f, BuiltinFn::Sin | BuiltinFn::Cos) {
                                    return Some((f, args[0]));
                                }
                            }
                        }
                    }
                    None
                };
                if let (Some((fa, ua)), Some((fb, ub))) = (as_trig(a), as_trig(b)) {
                    if ua == ub && fa != fb {
                        return Some((ua, coeff));
                    }
                }
            }
            None
        };
        let hit = if let Some((arg, a)) = sin_cos_product(&simplifier.context, lhs) {
            (!contains_var(&simplifier.context, rhs, var)).then_some((arg, a, rhs))
        } else if let Some((arg, a)) = sin_cos_product(&simplifier.context, rhs) {
            (!contains_var(&simplifier.context, lhs, var)).then_some((arg, a, lhs))
        } else {
            None
        };
        if let Some((arg, a_coeff, c)) = hit {
            let cv = cas_math::numeric_eval::as_rational_const(&simplifier.context, c)?;
            let target = (&cv + &cv) / a_coeff; // 2c/A
            let two = simplifier.context.num(2);
            let two_arg = simplifier.context.add(Expr::Mul(two, arg));
            let sin_call = simplifier.context.call("sin", vec![two_arg]);
            let target_expr = simplifier.context.add(Expr::Number(target));
            let reduced = Equation {
                lhs: sin_call,
                rhs: target_expr,
                op: RelOp::Eq,
            };
            return try_solve_periodic_trig_equation(&reduced, var, simplifier);
        }
    }

    // TAN(u) = TAN(v) (re-cycle C, 2026-07-14): `tan(u) = tan(v) ⟺ u ≡ v (mod π)`, where both
    // sides are defined. The generic paths mangled these (`tan(2x)=tan(x)` → a garbage
    // `sin(x) − 0·2·cos(x)` residual; `tan(2x)=tan(3x)` HUNG ~216s branching through the sin/cos
    // rewrite). For AFFINE rational-coefficient arguments the equivalence is a pure arithmetic
    // progression: `w·x = kπ − Δb` (w = a₁−a₂), minus the tan POLE progressions — and every pole
    // test is EXACTLY decidable: the solution's rational part must match the pole's rational part
    // (π is irrational, so `π·q₁ = q₂` forces both zero), and the π-parts intersect iff the offset
    // divides by the rational gcd of the steps. Emits the full `Periodic` family.
    {
        use num_integer::Integer as _;
        use num_traits::{Signed as _, ToPrimitive as _};
        let bare_tan_affine = |ctx: &Context, e: ExprId| -> Option<(BigRational, BigRational)> {
            let Expr::Function(fn_id, args) = ctx.get(e) else {
                return None;
            };
            if args.len() != 1
                || !matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Tan))
                || !contains_var(ctx, args[0], var)
            {
                return None;
            }
            let poly = cas_math::polynomial::Polynomial::from_expr(ctx, args[0], var).ok()?;
            if poly.degree() != 1 {
                return None;
            }
            let a = poly
                .coeffs
                .get(1)
                .cloned()
                .unwrap_or_else(BigRational::zero);
            let b = poly
                .coeffs
                .first()
                .cloned()
                .unwrap_or_else(BigRational::zero);
            if a.is_zero() {
                return None;
            }
            Some((a, b))
        };
        // Direct `tan(u) = tan(v)`, or the DIFF form `tan(u) − tan(v) = 0` matched on the RAW
        // tree (simplify collapses the difference into a sin/cos quotient before this handler
        // runs — the raw-tree rule).
        let mut pair = match (
            bare_tan_affine(&simplifier.context, lhs),
            bare_tan_affine(&simplifier.context, rhs),
        ) {
            (Some(p1), Some(p2)) => Some((p1, p2)),
            _ => None,
        };
        if pair.is_none() && cas_math::expr_predicates::is_zero_expr(&simplifier.context, eq.rhs) {
            let raw = eq.lhs;
            let (t1, t2) = match simplifier.context.get(raw).clone() {
                Expr::Sub(l, r) => (Some(l), Some(r)),
                Expr::Add(l, r) => match simplifier.context.get(r) {
                    Expr::Neg(inner) => (Some(l), Some(*inner)),
                    _ => (None, None),
                },
                _ => (None, None),
            };
            if let (Some(t1), Some(t2)) = (t1, t2) {
                if let (Some(p1), Some(p2)) = (
                    bare_tan_affine(&simplifier.context, t1),
                    bare_tan_affine(&simplifier.context, t2),
                ) {
                    pair = Some((p1, p2));
                }
            }
        }
        if let Some(((a1, b1), (a2, b2))) = pair {
            let w = &a1 - &a2;
            if w.is_zero() && !(&b1 - &b2).is_zero() {
                // Same slope, distinct RATIONAL offsets: `tan(t + Δb) = tan(t)` needs
                // `Δb ≡ 0 (mod π)`, impossible for rational Δb ≠ 0 (π is irrational).
                return Some(SolutionSet::Empty);
            }
            if !w.is_zero() {
                // Normalize the solution progression `x = (kπ − Δb)/w` to positive step:
                // x = k·s·π + r0 with s = 1/|w| and rational offset r0.
                let db = &b1 - &b2;
                let (wp, dbp) = if w.is_positive() {
                    (w.clone(), db.clone())
                } else {
                    (-w.clone(), -db)
                };
                let s = BigRational::one() / &wp; // π-part step
                let r0 = -&dbp / &wp; // rational offset of every solution
                let gcd_q = |a: &BigRational, b: &BigRational| -> BigRational {
                    BigRational::new(a.numer().gcd(b.numer()), a.denom().lcm(b.denom()))
                };
                let lcm_z = |a: i64, b: i64| -> i64 { a / a.gcd(&b) * b };
                // Banned residue classes of k, one per pole progression that the
                // solution family can actually reach.
                let mut banned: Vec<(i64, i64)> = Vec::new(); // (k0, modulus L)
                let mut modulus: i64 = 1;
                let mut decidable = true;
                for (ai, bi) in [(&a1, &b1), (&a2, &b2)] {
                    // Pole set of tan(aᵢ·x + bᵢ): x = (1/2 + m)π/|aᵢ| − bᵢ/aᵢ.
                    let rat_pole = -bi / ai;
                    if rat_pole != r0 {
                        continue; // rational parts differ: never hits (π irrational)
                    }
                    let t = (BigRational::one() / ai).abs(); // pole π-step
                    let o = &t / BigRational::from_integer(2.into()); // pole π-offset 1/(2|aᵢ|)
                    let g = gcd_q(&s, &t);
                    if !(&o / &g).is_integer() {
                        continue; // offset not reachable: no hits
                    }
                    // Solve k·s ≡ o (mod t): period of residues L = t/g (integer).
                    let l_big = (&t / &g).to_integer();
                    let Some(l) = l_big.to_i64() else {
                        decidable = false;
                        break;
                    };
                    let mut found = None;
                    for k in 0..l {
                        let lhs_val = BigRational::from_integer(k.into()) * &s - &o;
                        if (lhs_val / &t).is_integer() {
                            found = Some(k);
                            break;
                        }
                    }
                    if let Some(k0) = found {
                        banned.push((k0, l));
                        modulus = lcm_z(modulus, l);
                    }
                }
                if decidable {
                    let mut bases: Vec<ExprId> = Vec::new();
                    for k in 0..modulus {
                        if banned.iter().any(|&(k0, l)| (k - k0).rem_euclid(l) == 0) {
                            continue;
                        }
                        let pi_coeff = BigRational::from_integer(k.into()) * &s;
                        let pi = simplifier
                            .context
                            .add(Expr::Constant(cas_ast::Constant::Pi));
                        let mut base = if pi_coeff.is_zero() {
                            simplifier.context.num(0)
                        } else {
                            let c = simplifier.context.add(Expr::Number(pi_coeff));
                            simplifier.context.add(Expr::Mul(c, pi))
                        };
                        if !r0.is_zero() {
                            let r = simplifier.context.add(Expr::Number(r0.clone()));
                            base = simplifier.context.add(Expr::Add(base, r));
                        }
                        let (base, _) = simplifier.simplify(base);
                        bases.push(base);
                    }
                    if bases.is_empty() {
                        return Some(SolutionSet::Empty);
                    }
                    let period_coeff = BigRational::from_integer(modulus.into()) * &s;
                    let pi = simplifier
                        .context
                        .add(Expr::Constant(cas_ast::Constant::Pi));
                    let pc = simplifier.context.add(Expr::Number(period_coeff));
                    let period = simplifier.context.add(Expr::Mul(pc, pi));
                    let (period, _) = simplifier.simplify(period);
                    return Some(SolutionSet::Periodic { bases, period });
                }
            }
        }
    }

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

    // `c·trig(arg)^n = 0` (n ≥ 2) and the complementary quotient `c·trig(arg)^n / comp(arg)^m = 0`
    // are zero exactly where `trig(arg) = 0`. Covers the odd-power and `Neg` forms the n=2 reduction
    // misses (`-sin(x)^3 = 0` from `(cos+1)(cos-1)·sin`; `sin(x)·tan(x) = sin²/cos = 0`), which else
    // collapsed to the principal root only / a residual.
    {
        let is_zero = |ctx: &Context, e: ExprId| -> bool {
            !contains_var(ctx, e, var)
                && cas_math::numeric_eval::as_rational_const(ctx, e).is_some_and(|c| c.is_zero())
        };
        let hit = if is_zero(&simplifier.context, rhs) {
            reduces_to_trig_zero(&simplifier.context, lhs, var)
        } else if is_zero(&simplifier.context, lhs) {
            reduces_to_trig_zero(&simplifier.context, rhs, var)
        } else {
            None
        };
        if let Some((f, arg)) = hit {
            let zero = simplifier.context.num(0);
            let trig_call = simplifier.context.call_builtin(f, vec![arg]);
            let reduced = Equation {
                lhs: trig_call,
                rhs: zero,
                op: RelOp::Eq,
            };
            return try_solve_periodic_trig_equation(&reduced, var, simplifier);
        }
    }

    // `trig(arg)^n = c` for an ODD integer n ≥ 3 and a constant c  ⇔  `trig(arg) = c^(1/n)` — the map
    // t ↦ tⁿ is a bijection on ℝ for odd n, so this is exact. Reduces `cos(x)^3 = 1 → cos(x) = 1 →
    // {2kπ}`; without it the bare fall-through isolated the principal root only (`{0}`). The n = 2
    // square is handled by the double-angle reduction above; the n = 0 case (RHS already 0) by the
    // zero reduction; even n ≥ 4 is left to the residual path.
    {
        // Restricted to sin/cos: tan(x)^n is rewritten by the simplifier (tan = sin/cos) into a form
        // this Pow-matcher does not see, and the reduced tan(x) = c^(1/n) recursion mangled into a
        // residual — leave tan powers to the existing path.
        let odd_power_trig =
            |ctx: &Context, e: ExprId| -> Option<(ExprId, BigRational, BuiltinFn)> {
                if let Expr::Pow(base, exp) = ctx.get(e) {
                    let (base, exp) = (*base, *exp);
                    let n = cas_math::numeric_eval::as_rational_const(ctx, exp)?;
                    if !n.is_integer() {
                        return None;
                    }
                    let ni = n.to_integer();
                    if ni < num_bigint::BigInt::from(3) || num_integer::Integer::is_even(&ni) {
                        return None;
                    }
                    if let Expr::Function(fn_id, args) = ctx.get(base) {
                        if args.len() == 1 && contains_var(ctx, args[0], var) {
                            if let Some(f) = ctx.builtin_of(*fn_id) {
                                if matches!(f, BuiltinFn::Sin | BuiltinFn::Cos) {
                                    return Some((base, n, f));
                                }
                            }
                        }
                    }
                }
                None
            };
        let hit = if !contains_var(&simplifier.context, rhs, var) {
            odd_power_trig(&simplifier.context, lhs).map(|(call, n, f)| (call, n, f, rhs))
        } else if !contains_var(&simplifier.context, lhs, var) {
            odd_power_trig(&simplifier.context, rhs).map(|(call, n, f)| (call, n, f, lhs))
        } else {
            None
        };
        if let Some((trig_call, n, _f, c)) = hit {
            // SOUNDNESS: sin/cos ∈ [−1, 1], so sin(x)ⁿ ∈ [−1, 1]; if the RHS is PROVABLY |c| > 1 the
            // equation has NO real solution. Without this the reduced `sin(x) = c^(1/n)` (e.g.
            // `sin(x)^3 = 2 → sin(x) = 2^(1/3)`) leaks a spurious non-real `arcsin(2^(1/3))` because the
            // cube root is not a quadratic surd the range guard recognises.
            if let Some((a, b, nn)) = cas_math::root_forms::as_linear_surd(&simplifier.context, c) {
                let one = BigRational::one();
                let vs_upper = linear_surd_sign(&(&a - &one), &b, &nn);
                let vs_lower = linear_surd_sign(&(&a + &one), &b, &nn);
                if vs_upper == std::cmp::Ordering::Greater || vs_lower == std::cmp::Ordering::Less {
                    return Some(SolutionSet::Empty);
                }
            }
            let inv_n = simplifier.context.add(Expr::Number(n.recip())); // 1/n
            let root = simplifier.context.add(Expr::Pow(c, inv_n));
            let (root, _) = simplifier.simplify(root);
            let reduced = Equation {
                lhs: trig_call,
                rhs: root,
                op: RelOp::Eq,
            };
            return try_solve_periodic_trig_equation(&reduced, var, simplifier);
        }
    }

    // `trig(arg)^n = c` for an EVEN integer n ≥ 4 (sin/cos): `sin(x)ⁿ ∈ [0, 1]` for even n, so c < 0 or
    // c > 1 ⇒ NO solution; c = 0 ⇒ `trig(arg) = 0`; 0 < c ≤ 1 ⇒ `trig(arg) = ±c^(1/n)`, union the two
    // families. (n = 2 is the double-angle reduction above; odd n is the bijective reduction above.)
    // Without this, `sin(x)^4 = 1` collapsed to a finite `{π/2, -π/2}` and `sin(x)^4 = 4` leaked a
    // spurious `arcsin(4^(1/4))`.
    {
        let even_power_trig =
            |ctx: &Context, e: ExprId| -> Option<(ExprId, BigRational, BuiltinFn)> {
                if let Expr::Pow(base, exp) = ctx.get(e) {
                    let (base, exp) = (*base, *exp);
                    let n = cas_math::numeric_eval::as_rational_const(ctx, exp)?;
                    if !n.is_integer() {
                        return None;
                    }
                    let ni = n.to_integer();
                    if ni < num_bigint::BigInt::from(4) || !num_integer::Integer::is_even(&ni) {
                        return None;
                    }
                    if let Expr::Function(fn_id, args) = ctx.get(base) {
                        if args.len() == 1 && contains_var(ctx, args[0], var) {
                            if let Some(f) = ctx.builtin_of(*fn_id) {
                                if matches!(f, BuiltinFn::Sin | BuiltinFn::Cos) {
                                    return Some((base, n, f));
                                }
                            }
                        }
                    }
                }
                None
            };
        let hit = if !contains_var(&simplifier.context, rhs, var) {
            even_power_trig(&simplifier.context, lhs).map(|(call, n, f)| (call, n, f, rhs))
        } else if !contains_var(&simplifier.context, lhs, var) {
            even_power_trig(&simplifier.context, rhs).map(|(call, n, f)| (call, n, f, lhs))
        } else {
            None
        };
        if let Some((trig_call, n, _f, c)) = hit {
            match const_sign_vs_zero(&simplifier.context, c)? {
                std::cmp::Ordering::Less => return Some(SolutionSet::Empty), // sin/cos^(even) ≥ 0
                std::cmp::Ordering::Equal => {
                    let zero = simplifier.context.num(0);
                    let reduced = Equation {
                        lhs: trig_call,
                        rhs: zero,
                        op: RelOp::Eq,
                    };
                    return try_solve_periodic_trig_equation(&reduced, var, simplifier);
                }
                std::cmp::Ordering::Greater => {
                    let inv_n = simplifier.context.add(Expr::Number(n.recip())); // 1/n
                    let value = simplifier.context.add(Expr::Pow(c, inv_n)); // c^(1/n) ≥ 0
                    let (value, _) = simplifier.simplify(value);
                    return solve_trig_equals_plus_minus(simplifier, trig_call, value, var);
                }
            }
        }
    }

    // `|trig(arg)| = c` (sin/cos): `|sin/cos| ∈ [0, 1]`, so c < 0 ⇒ NO solution; c = 0 ⇒ `trig = 0`;
    // 0 < c ≤ 1 ⇒ `trig = ±c`, union the families (c > 1 declines via both `±c` solving to Empty).
    // `abs(sin(x)) = 1` collapsed to a finite `{π/2, -π/2}` instead of `{π/2 + kπ}`.
    {
        let abs_trig = |ctx: &Context, e: ExprId| -> Option<(ExprId, BuiltinFn)> {
            if let Expr::Function(fn_id, args) = ctx.get(e) {
                if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Abs) {
                    if let Expr::Function(inner_id, inner_args) = ctx.get(args[0]) {
                        if inner_args.len() == 1 && contains_var(ctx, inner_args[0], var) {
                            if let Some(f) = ctx.builtin_of(*inner_id) {
                                if matches!(f, BuiltinFn::Sin | BuiltinFn::Cos) {
                                    return Some((args[0], f));
                                }
                            }
                        }
                    }
                }
            }
            None
        };
        let hit = if !contains_var(&simplifier.context, rhs, var) {
            abs_trig(&simplifier.context, lhs).map(|(call, f)| (call, f, rhs))
        } else if !contains_var(&simplifier.context, lhs, var) {
            abs_trig(&simplifier.context, rhs).map(|(call, f)| (call, f, lhs))
        } else {
            None
        };
        if let Some((trig_call, _f, c)) = hit {
            // Accept a SURD RHS (`|cos(x)| = √2/2`) too, not just a rational — branch on the exact sign.
            match const_sign_vs_zero(&simplifier.context, c)? {
                std::cmp::Ordering::Less => return Some(SolutionSet::Empty), // |trig| ≥ 0
                std::cmp::Ordering::Equal => {
                    let zero = simplifier.context.num(0);
                    let reduced = Equation {
                        lhs: trig_call,
                        rhs: zero,
                        op: RelOp::Eq,
                    };
                    return try_solve_periodic_trig_equation(&reduced, var, simplifier);
                }
                std::cmp::Ordering::Greater => {
                    return solve_trig_equals_plus_minus(simplifier, trig_call, c, var)
                }
            }
        }
    }

    // `A·trig(a·x) + B = C` (A ≠ 0, B and C constant) -> `trig(a·x) = (C − B)/A`, then recurse. Without
    // this the outside coefficient/offset leaves the trig side a `Mul`/`Add` that `detect` cannot see,
    // so the bare fall-through emitted only the principal value — an INCOMPLETE solution set presented
    // as complete (e.g. `solve(2·sin x = 1)` -> `{π/6}` instead of `{π/6 + 2kπ, 5π/6 + 2kπ}`), unsound.
    // The peel probes the SIMPLIFIED sides first (historical behavior), then the RAW
    // sides: the entry simplify can DESTROY the affine-in-tan structure (`tan(x) + 1`
    // folds to `(sin(x) + cos(x)) / cos(x)`), which dropped the whole periodic family
    // (`solve(tan(x) + 1 = 2)` → `{π/4}`, no `+kπ`) via the principal-only isolation.
    for (side_l, side_r) in [(lhs, rhs), (eq.lhs, eq.rhs)] {
        let lhs_has = contains_var(&simplifier.context, side_l, var);
        let rhs_has = contains_var(&simplifier.context, side_r, var);
        if lhs_has != rhs_has {
            let (var_side, const_side) = if lhs_has {
                (side_l, side_r)
            } else {
                (side_r, side_l)
            };
            if let Some((call, a_coeff, b_offset)) =
                peel_affine_trig(&mut simplifier.context, var_side, var)
            {
                let diff = simplifier.context.add(Expr::Sub(const_side, b_offset));
                let (diff, _) = simplifier.simplify(diff);
                // Skip the `/1` when the coefficient is unit: `Div(diff, 1)` sends `simplify` down a
                // different normal form (`√3/2 → 9/2·3^(-3/2)`) that the notable-angle classifier no
                // longer recognizes, leaving `arcsin(√3/2)` unfolded to `π/3`.
                let reduced_rhs = if num_traits::One::is_one(&a_coeff) {
                    diff
                } else {
                    let a_expr = simplifier.context.add(Expr::Number(a_coeff));
                    let d = simplifier.context.add(Expr::Div(diff, a_expr));
                    simplifier.simplify(d).0
                };
                let reduced = Equation {
                    lhs: call,
                    rhs: reduced_rhs,
                    op: RelOp::Eq,
                };
                return try_solve_periodic_trig_equation(&reduced, var, simplifier);
            }
        }
    }

    // `trig(a·x + b)`: a positive RATIONAL slope keeps the historical exact
    // path; a var-free SYMBOLIC slope with provably positive sign (π·x,
    // √2·x, e·x — the final-audit periodicity-drop family) generalizes it.
    // Both return the slope/offset as EXPRESSION nodes: the map-back below
    // divides bases and period symbolically either way (2π/π → 2).
    let detect = |simplifier: &mut Simplifier, e: ExprId| -> Option<(BuiltinFn, ExprId, ExprId)> {
        let (fn_builtin, arg) = {
            let ctx = &simplifier.context;
            if let Expr::Function(fn_id, args) = ctx.get(e) {
                if args.len() == 1 {
                    if let Some(f) = ctx.builtin_of(*fn_id) {
                        if matches!(f, BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Tan) {
                            (f, args[0])
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            } else {
                return None;
            }
        };
        if let Some((a, b)) = positive_affine_arg_of_var(&simplifier.context, arg, var) {
            let a_expr = simplifier.context.add(Expr::Number(a));
            let b_expr = simplifier.context.add(Expr::Number(b));
            return Some((fn_builtin, a_expr, b_expr));
        }
        let (a_expr, b_expr) = symbolic_positive_affine_arg_of_var(simplifier, arg, var)?;
        Some((fn_builtin, a_expr, b_expr))
    };
    // `trig(a·x + b) = c` or `c = trig(a·x + b)`, with `c` constant.
    let (func, a_expr, b_expr, c) = if let Some((f, a, b)) = detect(simplifier, lhs) {
        if contains_var(&simplifier.context, rhs, var) {
            return None;
        }
        (f, a, b, rhs)
    } else if let Some((f, a, b)) = detect(simplifier, rhs) {
        if contains_var(&simplifier.context, lhs, var) {
            return None;
        }
        (f, a, b, lhs)
    } else {
        return None;
    };

    let pi = simplifier.context.add(Expr::Constant(Constant::Pi));
    let two = simplifier.context.num(2);
    let two_pi = simplifier.context.add(Expr::Mul(two, pi));
    // Set when the Sin/Cos RHS is a PARAMETER: the emitted periodic family is
    // wrapped in the closed-range guard at the return.
    let mut parametric_range_guard: Option<ExprId> = None;

    // Representative root(s) for the bare argument `u = a·x`, and the shared period.
    let (bases_u, period_u): (Vec<ExprId>, ExprId) = match func {
        // tan(u)=c is a single family {arctan(c) + kπ} for EVERY constant c.
        BuiltinFn::Tan => {
            let at = simplifier.context.call("arctan", vec![c]);
            (vec![simplifier.simplify(at).0], pi)
        }
        BuiltinFn::Sin | BuiltinFn::Cos => {
            // Classify the RHS `c` relative to {−1, 0, 1} EXACTLY (never f64): a quadratic surd
            // (`√2/2`, `√3/2`) OR an n-th root (`(1/4)^(1/4)` from the even-power reduction). An
            // out-of-range `c` (|c| > 1) is NO real solution — returning `Empty` here also kills the
            // spurious `arcsin(c)` (= nan) the generic inversion would otherwise leak (`sin(x)^4 = 4`).
            let is_sin = matches!(func, BuiltinFn::Sin);
            // PARAMETRIC RHS (`sin(x) = a`): the classifier cannot place a free
            // symbol against {−1, 0, 1}, and declining here fell through to the
            // principal-only inversion (`{arcsin(a)}` — no supplementary branch, no
            // +2kπ family, no range gate). The two-family InOpen form is correct as
            // a SET for EVERY −1 ≤ c ≤ 1 (at the endpoints the families coincide or
            // interlace; at 0 they alias), so emit it GUARDED by the closed range
            // (`c + 1 ≥ 0` ∧ `1 − c ≥ 0`) via the flag consumed at the map-back tail.
            let classified = classify_trig_unit_rhs(&simplifier.context, c);
            let class = match classified {
                Some(cl) => cl,
                None => {
                    if !cas_ast::collect_variables(&simplifier.context, c).is_empty() {
                        parametric_range_guard = Some(c);
                        TrigUnitClass::InOpen
                    } else {
                        return None;
                    }
                }
            };
            match class {
                TrigUnitClass::OutOfRange => return Some(SolutionSet::Empty),
                TrigUnitClass::Unit => {
                    // c = ±1: the two roots of the period coincide → ONE family, period 2π.
                    let arc = if is_sin { "arcsin" } else { "arccos" };
                    let arc_call = simplifier.context.call(arc, vec![c]);
                    (vec![simplifier.simplify(arc_call).0], two_pi)
                }
                TrigUnitClass::Zero => {
                    // c = 0: sin(u)=0 → {kπ}; cos(u)=0 → {π/2 + kπ}. Two roots π apart → ONE family, period π.
                    let arc = if is_sin { "arcsin" } else { "arccos" };
                    let arc_call = simplifier.context.call(arc, vec![c]);
                    (vec![simplifier.simplify(arc_call).0], pi)
                }
                TrigUnitClass::InOpen => {
                    // 0 < |c| < 1: TWO families in [0, 2π), shared period 2π.
                    //   sin(u)=c → {arcsin(c) + 2kπ, π - arcsin(c) + 2kπ}
                    //   cos(u)=c → {arccos(c) + 2kπ, 2π - arccos(c) + 2kπ}
                    let arc = if is_sin { "arcsin" } else { "arccos" };
                    let arc_call = simplifier.context.call(arc, vec![c]);
                    let (r1, _) = simplifier.simplify(arc_call);
                    let second = if is_sin {
                        simplifier.context.add(Expr::Sub(pi, r1))
                    } else {
                        simplifier.context.add(Expr::Sub(two_pi, r1))
                    };
                    (vec![r1, simplifier.simplify(second).0], two_pi)
                }
            }
        }
        _ => return None,
    };

    // Didactic narration, u-space half: only when the argument IS the bare
    // variable (a=1, b=0) — printing roots of a synthetic `u` for `sin(2x+1)`
    // would name a symbol the student never wrote.
    {
        use num_traits::{One, Zero};
        let arg_is_bare_var =
            cas_math::numeric_eval::as_rational_const(&simplifier.context, a_expr)
                .is_some_and(|q| q.is_one())
                && cas_math::numeric_eval::as_rational_const(&simplifier.context, b_expr)
                    .is_some_and(|q| q.is_zero());
        if arg_is_bare_var && !bases_u.is_empty() {
            let func_name = match func {
                BuiltinFn::Sin => "sin",
                BuiltinFn::Cos => "cos",
                _ => "tan",
            };
            let x = simplifier.context.var(var);
            steps_out.push(crate::SolveStep::new(
                format!("Invert {} over one period", func_name),
                Equation {
                    lhs: x,
                    rhs: bases_u[0],
                    op: RelOp::Eq,
                },
                crate::ImportanceLevel::Medium,
            ));
            if bases_u.len() > 1 {
                steps_out.push(crate::SolveStep::new(
                    "Second solution within the period".to_string(),
                    Equation {
                        lhs: x,
                        rhs: bases_u[1],
                        op: RelOp::Eq,
                    },
                    crate::ImportanceLevel::Medium,
                ));
            }
        }
    }

    // `u = a·x + b` ⇒ `x = (u − b)/a`: shift every base by `−b` then divide it and the period by `a`
    // (a > 1 SHRINKS the period: `cos(2x)=1 → {kπ}`; a = π gives a RATIONAL
    // x-period: `sin(πx)=1 → {1/2 + 2k}`, period 2π/π = 2).
    // Fold to a canonical rational Number when the division collapses (a
    // symbolic slope π/2 leaves `2 / 1/2` unfolded otherwise).
    let fold_rational = |simplifier: &mut Simplifier, e: ExprId| -> ExprId {
        match cas_math::numeric_eval::as_rational_const(&simplifier.context, e) {
            Some(q) => simplifier.context.add(Expr::Number(q)),
            None => e,
        }
    };
    let bases: Vec<ExprId> = bases_u
        .into_iter()
        .map(|u| {
            let shifted = simplifier.context.add(Expr::Sub(u, b_expr));
            let d = simplifier.context.add(Expr::Div(shifted, a_expr));
            let d = simplifier.simplify(d).0;
            fold_rational(simplifier, d)
        })
        .collect();
    let period_div = simplifier.context.add(Expr::Div(period_u, a_expr));
    let (period, _) = simplifier.simplify(period_div);
    let period = fold_rational(simplifier, period);
    // Didactic narration, families half: one step per periodic family, in the
    // exact shape the result set displays (`x = base + k·T`).
    {
        let x = simplifier.context.var(var);
        let k_var = simplifier.context.var("k");
        for base in &bases {
            let k_t = simplifier.context.add(Expr::Mul(k_var, period));
            // A zero base narrates as `x = k·T`, not `x = 0 + k·T`; anything
            // else stays in the exact `base + k·T` shape the result set shows
            // (a general simplify here FACTORS the sum into unreadable forms).
            let base_is_zero =
                cas_math::numeric_eval::as_rational_const(&simplifier.context, *base)
                    .is_some_and(|q| num_traits::Zero::is_zero(&q));
            let family = if base_is_zero {
                k_t
            } else {
                simplifier.context.add(Expr::Add(*base, k_t))
            };
            steps_out.push(crate::SolveStep::new(
                "Periodic family of solutions (k any integer)".to_string(),
                Equation {
                    lhs: x,
                    rhs: family,
                    op: RelOp::Eq,
                },
                crate::ImportanceLevel::Medium,
            ));
        }
    }
    let set = SolutionSet::Periodic { bases, period };
    Some(match parametric_range_guard {
        Some(c) => {
            use cas_ast::{Case, ConditionPredicate, ConditionSet};
            let one = simplifier.context.num(1);
            let c_plus_one = simplifier.context.add(Expr::Add(c, one));
            let c_plus_one = simplifier.simplify(c_plus_one).0;
            let one_b = simplifier.context.num(1);
            let one_minus_c = simplifier.context.add(Expr::Sub(one_b, c));
            let one_minus_c = simplifier.simplify(one_minus_c).0;
            let mut guard = ConditionSet::single(ConditionPredicate::NonNegative(c_plus_one));
            guard.push(ConditionPredicate::NonNegative(one_minus_c));
            SolutionSet::Conditional(vec![Case::new(guard, set)])
        }
        None => set,
    })
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

/// Detect a bare `trig(g) = c` (or `c = trig(g)`) equation: a single `sin`/`cos`/`tan` whose argument
/// `g` carries `var`, against a `var`-free side `c`. Returns `(f, g, c)`.
fn detect_bare_trig_equation(
    ctx: &Context,
    eq: &Equation,
    var: &str,
) -> Option<(cas_ast::BuiltinFn, ExprId, ExprId)> {
    use cas_solver_core::isolation_utils::contains_var;
    let side = |call: ExprId, other: ExprId| -> Option<(cas_ast::BuiltinFn, ExprId, ExprId)> {
        if let Expr::Function(fn_id, args) = ctx.get(call) {
            if args.len() == 1 && contains_var(ctx, args[0], var) && !contains_var(ctx, other, var)
            {
                if let Some(f) = ctx.builtin_of(*fn_id) {
                    if matches!(
                        f,
                        cas_ast::BuiltinFn::Sin | cas_ast::BuiltinFn::Cos | cas_ast::BuiltinFn::Tan
                    ) {
                        return Some((f, args[0], other));
                    }
                }
            }
        }
        None
    };
    side(eq.lhs, eq.rhs).or_else(|| side(eq.rhs, eq.lhs))
}

/// Extract the affine coefficients of `g = a·x + b` (slope `a` a nonzero rational, intercept `b` a
/// `var`-free expression) by sampling `g` at `x ∈ {0, 1, 2}`. Returns `None` if `g` is not affine in
/// `var` (the second difference is nonzero) or the slope is not a nonzero rational.
fn affine_coefficients(
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
fn map_solution_through_affine(
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

/// Solve a bare `trig(a·x + b) = c` whose additive shift `b` is a SYMBOLIC constant (a π-multiple like
/// `π/4`, an `arctan`, a surd — anything that is not a plain rational number). For such a shift the
/// simplifier's angle-addition expansion (`sin(x + π/4) → (√2/2)·(sin x + cos x)`) / the isolation
/// returns only the PRINCIPAL root, dropping the periodic family and the second branch. Solving
/// `trig(u) = c` for `u = a·x + b` (bare, so the existing periodic solver gives the full family) and
/// mapping back through `x = (u − b)/a` restores the periodicity. A PLAIN-rational shift (`sin(x + 1)`)
/// and bare/coefficient forms (`sin(2x)`) are handled correctly by the existing periodic path, so this
/// declines on them (keeping their behaviour and the huella untouched).
fn try_solve_shifted_argument_trig(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    if eq.op != RelOp::Eq {
        return None;
    }
    let (f, g, c) = detect_bare_trig_equation(&simplifier.context, eq, var)?;
    let (a, b) = affine_coefficients(simplifier, g, var)?;
    // Gate to a SYMBOLIC (non-plain-rational) shift — the forms the expansion/isolation mishandles to a
    // principal root. A plain-rational shift (including `b = 0`, the bare form) is already correct via
    // the existing periodic path, so decline it and leave its behaviour (and the huella) untouched.
    if cas_math::numeric_eval::as_rational_const(&simplifier.context, b).is_some() {
        return None;
    }
    // Solve the BARE `trig(u) = c` (full periodic family), then map `u = a·x + b` back to `x`.
    let u_var = "__shift_u";
    let u = simplifier.context.var(u_var);
    let trig_u = simplifier.context.call_builtin(f, vec![u]);
    let u_eq = Equation {
        lhs: trig_u,
        rhs: c,
        op: RelOp::Eq,
    };
    let (u_sol, _) = crate::solver_entrypoints_solve::solve(&u_eq, u_var, simplifier).ok()?;
    map_solution_through_affine(simplifier, u_sol, &a, b)
}

/// Solve `|f(x)| = g(x)` where `f` is a polynomial of degree ≥ 2 and `g` (or, for `|f| = |h|`, its
/// inner `h`) is a polynomial. The textbook split is `|f| = g ⟺ (f = g ∨ f = −g)` with each candidate
/// verified against the ORIGINAL equation `|f(r)| = g(r)` (which enforces the `g ≥ 0` requirement
/// exactly). The linear-`f` case is owned by the piecewise absolute-value handler; the constant-RHS
/// quadratic (`|x²−4| = 3`) by the isolation path — this catches the mixed `|quadratic| = variable`
/// forms (`|x²−1| = x+1`) that otherwise leak an `arcsin`/`sqrt` residual. Declines (residual) if any
/// candidate root is non-rational, so completeness is never overclaimed with unverifiable surds.
/// Solve `|E| = 0` ⟺ `E = 0` by dispatching the argument's zero-set to the full solver. The generic
/// abs isolation mis-handles a FACTORED argument (`|x·(x−2)| = 0 → {0}`, dropping the other factor's
/// root), whereas the direct `x·(x−2) = 0` path returns the complete `{0, 2}`. Scoped to the RHS-zero
/// case so `|E| = c` (c ≠ 0) keeps its own handlers.
fn try_solve_abs_equals_zero(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::{BuiltinFn, RelOp};
    use cas_math::numeric_eval::as_rational_const;
    use num_traits::Zero;
    if eq.op != RelOp::Eq {
        return None;
    }
    let as_abs = |ctx: &Context, e: ExprId| -> Option<ExprId> {
        if let Expr::Function(fn_id, args) = ctx.get(e) {
            if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Abs) {
                return Some(args[0]);
            }
        }
        None
    };
    let is_zero = |ctx: &Context, e: ExprId| as_rational_const(ctx, e).is_some_and(|v| v.is_zero());
    let arg = if is_zero(&simplifier.context, eq.rhs) {
        as_abs(&simplifier.context, eq.lhs)?
    } else if is_zero(&simplifier.context, eq.lhs) {
        as_abs(&simplifier.context, eq.rhs)?
    } else {
        return None;
    };
    let zero = simplifier.context.num(0);
    let inner_eq = Equation {
        lhs: arg,
        rhs: zero,
        op: RelOp::Eq,
    };
    let (sol, _) = crate::solver_entrypoints_solve::solve(&inner_eq, var, simplifier).ok()?;
    Some(sol)
}

fn try_solve_abs_polynomial_equation(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::{BuiltinFn, RelOp};
    use cas_math::numeric_eval::as_rational_const;
    use cas_math::polynomial::Polynomial;
    use cas_solver_core::isolation_utils::contains_var;

    if eq.op != RelOp::Eq {
        return None;
    }
    // Identify `|f|` on one side; the other side is `g`.
    let as_abs = |ctx: &Context, e: ExprId| -> Option<ExprId> {
        if let Expr::Function(fn_id, args) = ctx.get(e) {
            if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Abs) {
                return Some(args[0]);
            }
        }
        None
    };
    let (f, g) = if let Some(f) = as_abs(&simplifier.context, eq.lhs) {
        (f, eq.rhs)
    } else if let Some(f) = as_abs(&simplifier.context, eq.rhs) {
        (f, eq.lhs)
    } else {
        return None;
    };
    // `f` must be a polynomial of degree ≥ 2 (linear `|f|` is the piecewise handler's job).
    let f_poly = Polynomial::from_expr(&simplifier.context, f, var).ok()?;
    if f_poly.degree() < 2 {
        return None;
    }
    // `g` may itself be `|h|`; unwrap for the branch RHS, remembering to take the absolute value in the
    // verification. `g_core` must be a polynomial too (to evaluate exactly at each candidate).
    let (g_core, g_is_abs) = match as_abs(&simplifier.context, g) {
        Some(h) => (h, true),
        None => (g, false),
    };
    if !contains_var(&simplifier.context, g_core, var) {
        return None; // constant RHS is owned by the isolation path (keeps its surd rendering)
    }
    let g_poly = Polynomial::from_expr(&simplifier.context, g_core, var).ok()?;

    // Branches `f = g_core` and `f = −g_core`.
    let neg_g = simplifier.context.add(Expr::Neg(g_core));
    let mut candidates: Vec<ExprId> = Vec::new();
    for rhs in [g_core, neg_g] {
        let branch = Equation {
            lhs: f,
            rhs,
            op: RelOp::Eq,
        };
        let (sol, _) = crate::solver_entrypoints_solve::solve(&branch, var, simplifier).ok()?;
        match sol {
            SolutionSet::Discrete(roots) => candidates.extend(roots),
            SolutionSet::Empty => {}
            _ => return None, // a non-discrete branch ⇒ out of scope
        }
    }

    // Verify each candidate against the ORIGINAL `|f(r)| = g(r)` exactly (this enforces `g(r) ≥ 0`).
    // All candidates must be rational so the check — and completeness — is exact.
    let mut kept: Vec<ExprId> = Vec::new();
    let mut seen: Vec<num_rational::BigRational> = Vec::new();
    for r in candidates {
        let rv = as_rational_const(&simplifier.context, r)?; // non-rational ⇒ decline (scope)
        let fr = f_poly.eval(&rv);
        let gr = g_poly.eval(&rv);
        let abs_fr = num_traits::Signed::abs(&fr);
        let target = if g_is_abs {
            num_traits::Signed::abs(&gr)
        } else {
            gr
        };
        if abs_fr == target && !seen.contains(&rv) {
            seen.push(rv);
            kept.push(r);
        }
    }
    if kept.is_empty() {
        Some(SolutionSet::Empty)
    } else {
        Some(SolutionSet::Discrete(kept))
    }
}

/// Map an interval bound through the inverse affine `x = (u − b)/a` (`a` rational ≠ 0,
/// `b` a constant ExprId). Infinities map to the ±∞ matching the sign of `a`.
fn map_bound_through_inverse_affine(
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
fn map_set_through_inverse_affine(
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
fn try_solve_const_over_surd_affine_inequality(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
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
        return solve_relation_set(simplifier, var, den, zero, den_op);
    }
    if eq.op == RelOp::Eq {
        // `g = c/k` exactly: one root `x = (c/k − b)/a`.
        let target = simplifier.context.add(Expr::Number(c / k));
        let root = map_bound_through_inverse_affine(simplifier, target, &a, b);
        return Some(SolutionSet::Discrete(vec![root]));
    }
    // Inequality vs a nonzero threshold: solve in `u = g(x)` space, where every
    // breakpoint (`u = 0` pole, `u = c/k` boundary) is RATIONAL, then map back.
    let u_name = format!("__{var}_g");
    let u_var = simplifier.context.var(&u_name);
    let c_expr = simplifier.context.add(Expr::Number(c));
    let u_lhs = simplifier.context.add(Expr::Div(c_expr, u_var));
    let u_set = solve_relation_set(simplifier, &u_name, u_lhs, eq.rhs, eq.op.clone())?;
    map_set_through_inverse_affine(simplifier, u_set, &a, b)
}

/// U2 (scout backlog #4): `c / (a·x+b)^(1/q) ⋚ k` — the reciprocal of a
/// ROOT of a rational-affine argument (`1/sqrt(x) > 2`, `1/sqrt(x-1) > 2`,
/// `1/x^(1/3) > 2`). Solved by the two-stage monotone substitution
/// `w = (a·x+b)^(1/q)`: the w-space relation `c/w ⋚ k` has RATIONAL
/// breakpoints (pole 0, boundary c/k) and an existing exact owner; the
/// w-set then maps through the INCREASING power `u = w^q` (clamped to
/// `w > 0` first for even q — the root's domain), and finally through the
/// inverse affine to x. Declines (`None`) on anything outside the shape.
fn try_solve_const_over_root_affine_inequality(
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

/// Map a solution set through the INCREASING power `u = w^q` (valid only
/// when the set lies in a region where the power is monotone increasing —
/// the caller clamps even q to `w > 0`). Bound types are preserved.
fn map_set_through_increasing_power(
    simplifier: &mut Simplifier,
    set: SolutionSet,
    q: i64,
) -> Option<SolutionSet> {
    let map_bound = |simplifier: &mut Simplifier, e: ExprId| -> ExprId {
        let ctx = &simplifier.context;
        if cas_solver_core::solution_set::is_infinity(ctx, e)
            || cas_solver_core::solution_set::is_neg_infinity(ctx, e)
        {
            return e; // (±∞)^q keeps its sign (even q never sees −∞ post-clamp)
        }
        let q_expr = simplifier.context.num(q);
        let p = simplifier.context.add(Expr::Pow(e, q_expr));
        simplifier.simplify(p).0
    };
    let map_interval = |simplifier: &mut Simplifier, iv: cas_ast::Interval| -> cas_ast::Interval {
        cas_ast::Interval {
            min: map_bound(simplifier, iv.min),
            min_type: iv.min_type.clone(),
            max: map_bound(simplifier, iv.max),
            max_type: iv.max_type.clone(),
        }
    };
    Some(match set {
        SolutionSet::Empty => SolutionSet::Empty,
        SolutionSet::Continuous(iv) => SolutionSet::Continuous(map_interval(simplifier, iv)),
        SolutionSet::Union(ivs) => SolutionSet::Union(
            ivs.into_iter()
                .map(|iv| map_interval(simplifier, iv))
                .collect(),
        ),
        SolutionSet::Discrete(pts) => {
            SolutionSet::Discrete(pts.into_iter().map(|p| map_bound(simplifier, p)).collect())
        }
        // AllReals cannot arise from `c/w ⋚ k` (the pole is always excluded);
        // anything else is out of contract.
        _ => return None,
    })
}

/// Rewrite solver-opaque function ALIASES to their canonical invertible forms, recursively:
/// `log2(u) → log(2, u)`, `log10(u) → log(10, u)`, `cbrt(u) → u^(1/3)`. These evaluate,
/// differentiate and integrate fine, but the isolation dispatch has no inverse for them and
/// errored `función [log2] no definida`. The reciprocal-trig aliases (`csc`/`sec`/`cot`) are
/// NOT rewritten here: the simplifier re-folds `1/sin → csc` downstream, so they are handled
/// at the EQUATION level by [`try_solve_reciprocal_trig_equation`]. Returns `None` when
/// nothing changed.
fn normalize_solver_function_aliases(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    use cas_ast::BuiltinFn;
    let node = ctx.get(expr).clone();
    match node {
        Expr::Function(fn_id, args) => {
            let new_args: Vec<ExprId> = args
                .iter()
                .map(|a| normalize_solver_function_aliases(ctx, *a).unwrap_or(*a))
                .collect();
            let changed_args = new_args != args;
            let builtin = ctx.builtin_of(fn_id);
            if new_args.len() == 1 {
                let u = new_args[0];
                let rewritten = match builtin {
                    Some(BuiltinFn::Log2) => {
                        let two = ctx.num(2);
                        Some(ctx.call("log", vec![two, u]))
                    }
                    Some(BuiltinFn::Log10) => {
                        let ten = ctx.num(10);
                        Some(ctx.call("log", vec![ten, u]))
                    }
                    Some(BuiltinFn::Cbrt) => {
                        let third = ctx.add(Expr::Number(num_rational::BigRational::new(
                            1.into(),
                            3.into(),
                        )));
                        Some(ctx.add(Expr::Pow(u, third)))
                    }
                    _ => None,
                };
                if let Some(r) = rewritten {
                    return Some(r);
                }
            }
            if changed_args {
                Some(ctx.add(Expr::Function(fn_id, new_args)))
            } else {
                None
            }
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            let nl = normalize_solver_function_aliases(ctx, l);
            let nr = normalize_solver_function_aliases(ctx, r);
            if nl.is_none() && nr.is_none() {
                return None;
            }
            let (nl, nr) = (nl.unwrap_or(l), nr.unwrap_or(r));
            Some(match ctx.get(expr) {
                Expr::Add(_, _) => ctx.add(Expr::Add(nl, nr)),
                Expr::Sub(_, _) => ctx.add(Expr::Sub(nl, nr)),
                Expr::Mul(_, _) => ctx.add(Expr::Mul(nl, nr)),
                Expr::Div(_, _) => ctx.add(Expr::Div(nl, nr)),
                _ => ctx.add(Expr::Pow(nl, nr)),
            })
        }
        Expr::Neg(inner) => {
            let ni = normalize_solver_function_aliases(ctx, inner)?;
            Some(ctx.add(Expr::Neg(ni)))
        }
        _ => None,
    }
}

/// Match `c / trig(g)` or `c · trig(g)^(−1)` (nonzero rational `c`, `trig ∈
/// {sin, cos, tan}`) and return `(c, trig_builtin, g)`, so `c/trig(g) = k` can
/// reduce to the bare `trig(g) = c/k`.
fn match_reciprocal_trig_call(
    ctx: &Context,
    e: ExprId,
) -> Option<(num_rational::BigRational, cas_ast::BuiltinFn, ExprId)> {
    use cas_ast::BuiltinFn;
    use cas_math::numeric_eval::as_rational_const;
    use num_traits::Zero;
    // Peel the reciprocal shape to `(constant coefficient, trig-call node)`.
    let (c, fn_node) = if let Expr::Div(num, den) = ctx.get(e) {
        let (num, den) = (*num, *den);
        (as_rational_const(ctx, num)?, den)
    } else {
        let (coeff, core) = peel_rational_coefficient(ctx, e);
        if let Expr::Pow(base, exp) = ctx.get(core) {
            let (base, exp) = (*base, *exp);
            let minus_one = num_rational::BigRational::from_integer((-1).into());
            if as_rational_const(ctx, exp) != Some(minus_one) {
                return None;
            }
            (coeff, base)
        } else {
            return None;
        }
    };
    if c.is_zero() {
        return None;
    }
    if let Expr::Function(f, a) = ctx.get(fn_node) {
        if a.len() == 1 {
            if let Some(b @ (BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Tan)) = ctx.builtin_of(*f)
            {
                return Some((c, b, a[0]));
            }
        }
    }
    None
}

/// `csc(g) = c` / `sec(g) = c` / `cot(g) = c` at the EQUATION level (raw tree, constant
/// `c`): reduce to the owning trig solver — `csc ⟺ sin(g) = 1/c` (`c = 0` ⇒ Empty, `1/sin`
/// is never 0), `sec ⟺ cos(g) = 1/c`, and `cot ⟺ cos(g) − c·sin(g) = 0` (the homogeneous
/// linear-trig handler, which keeps `cot(g) = 0 → g = π/2 + kπ` — a `1/tan` rewrite would
/// lose those roots). A subtree rewrite (`csc → 1/sin`) does NOT survive: the simplifier
/// re-folds the reciprocal back to `csc` and the isolation errors `función [csc] no
/// definida`. Inequalities and symbolic RHS decline (honest residuals).
fn try_solve_reciprocal_trig_equation(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::{BuiltinFn, RelOp};
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;
    use num_traits::Zero;
    if eq.op != RelOp::Eq {
        return None;
    }
    // Normalize to `fn(g) = rhs` with the call on the LHS.
    let (call, rhs) = if contains_var(&simplifier.context, eq.lhs, var) {
        (eq.lhs, eq.rhs)
    } else {
        (eq.rhs, eq.lhs)
    };
    if contains_var(&simplifier.context, rhs, var) {
        return None;
    }

    // `c / trig(g) = k` (nonzero constants `c`, `k`; trig ∈ {sin, cos, tan}) —
    // `Div(c, trig(g))` or `c · trig(g)^(−1)`. Reduce to `trig(g) = c/k` and route to
    // the bare-trig solver, which returns the FULL periodic family. Without this the
    // reciprocal form isolates to the boundary and returns only the principal value
    // (`2/sin(x)=4 → {π/6}`, dropping `5π/6` and every `+2kπ`), or the coefficient-1
    // form folds `1/sin → csc` mid-isolation and leaks `solve(csc(x)=2)`.
    if let Some((c, trig_builtin, g)) = match_reciprocal_trig_call(&simplifier.context, call) {
        if !contains_var(&simplifier.context, g, var) {
            return None;
        }
        let k = as_rational_const(&simplifier.context, rhs)?;
        if k.is_zero() {
            // `c/trig(g) = 0` with `c ≠ 0`: sin/cos are bounded so the value is never
            // 0 → Empty; tan can be ±∞ at its poles (`c/tan = 0 ⇒ g = π/2 + kπ`), a
            // distinct shape left to the isolation path.
            return match trig_builtin {
                BuiltinFn::Sin | BuiltinFn::Cos => Some(SolutionSet::Empty),
                _ => None,
            };
        }
        let target = simplifier.context.add(Expr::Number(c / k));
        let trig_name = match trig_builtin {
            BuiltinFn::Sin => "sin",
            BuiltinFn::Cos => "cos",
            _ => "tan",
        };
        let trig = simplifier.context.call(trig_name, vec![g]);
        return solve_relation_set(simplifier, var, trig, target, RelOp::Eq);
    }

    let (fn_id, args) = match simplifier.context.get(call) {
        Expr::Function(f, a) => (*f, a.clone()),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }
    let g = args[0];
    if !contains_var(&simplifier.context, g, var) {
        return None;
    }
    let builtin = simplifier.context.builtin_of(fn_id);
    match builtin {
        Some(BuiltinFn::Csc) | Some(BuiltinFn::Sec) => {
            // `1/trig(g) = c`: `c = 0` is impossible; otherwise `trig(g) = 1/c` (the
            // range check |1/c| ≤ 1 comes free from the sin/cos solver).
            if as_rational_const(&simplifier.context, rhs).is_some_and(|r| r.is_zero()) {
                return Some(SolutionSet::Empty);
            }
            let one = simplifier.context.num(1);
            let recip = simplifier.context.add(Expr::Div(one, rhs));
            let trig_name = if builtin == Some(BuiltinFn::Csc) {
                "sin"
            } else {
                "cos"
            };
            let trig = simplifier.context.call(trig_name, vec![g]);
            solve_relation_set(simplifier, var, trig, recip, RelOp::Eq)
        }
        Some(BuiltinFn::Cot) => {
            // `cos(g)/sin(g) = c ⟺ cos(g) − c·sin(g) = 0` (where sin(g) ≠ 0; the roots of
            // cos − c·sin never coincide with sin = 0, since cos and sin have no common zero).
            let cos = simplifier.context.call("cos", vec![g]);
            let sin = simplifier.context.call("sin", vec![g]);
            let c_sin = simplifier.context.add(Expr::Mul(rhs, sin));
            let lhs = simplifier.context.add(Expr::Sub(cos, c_sin));
            let zero = simplifier.context.num(0);
            solve_relation_set(simplifier, var, lhs, zero, RelOp::Eq)
        }
        _ => None,
    }
}

/// True when `expr` is (or its top-level `Mul` contains) at least two square-root factors —
/// `√A·√B`, whether written `sqrt(_)` or `Pow(_, 1/2)`. This is the shape the simplifier merges to
/// `√(A·B)`, widening the real domain from `{A≥0 ∧ B≥0}` to `{A·B≥0}` and admitting extraneous roots
/// after squaring. Single radicals (handled by the existing range-condition machinery) return false.
fn has_radical_product(ctx: &Context, expr: ExprId) -> bool {
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

fn solve_local_core(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: CoreSolverOptions,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    // P16: the handler chain below re-simplifies the same interned diff/lhs/rhs
    // 4-8x per solve (measured up to 85% redundant `simplify` calls). Scope the
    // engine's solve memo over the whole handler chain; re-entrant across the
    // recursive sub-solves.
    simplifier.begin_solve_simplify_memo();
    let result = solve_local_core_inner(eq, var, simplifier, opts, ctx);
    simplifier.end_solve_simplify_memo();
    // RADICAL-PRODUCT DOMAIN FILTER (F1 2026-07-13b): the simplifier merges `√A·√B → √(A·B)` and the
    // radical handler squares to a polynomial, WIDENING the real domain from `{A≥0 ∧ B≥0}` to
    // `{A·B≥0}` — so an extraneous root (`x=-1` of `√x·√(x-3)=2`, where `(-1)·(-4)=4≥0`) survives
    // verification against the squared/merged form. THIS wrapper still holds the ORIGINAL radical
    // equation for the current solve level, so re-verify the discrete candidates against it:
    // `check_root` sees `√(-1)` (non-real → extraneous), and the per-radicand domain conditions let
    // the exact surd-sign prover drop surd roots (`(3-√17)/2 < 2` violates `x≥2`). Sound and narrow:
    // scoped to real-only radical equations; a genuine root satisfies its own domain and is never
    // dropped. The inner squared sub-solve (no `√`) and every non-radical solve are untouched.
    if opts.value_domain.is_real_only() {
        if let Ok((set, steps)) = result {
            if matches!(set, SolutionSet::Discrete(_))
                && (has_radical_product(&simplifier.context, eq.lhs)
                    || has_radical_product(&simplifier.context, eq.rhs))
            {
                let mut conds = ctx.required_conditions();
                for side in [eq.lhs, eq.rhs] {
                    let dom = cas_solver_core::domain_inference::infer_implicit_domain(
                        &simplifier.context,
                        side,
                        true,
                    );
                    for cond in dom.conditions() {
                        if !conds.contains(cond) {
                            conds.push(cond.clone());
                        }
                    }
                }
                let filtered = filter_real_solutions(&mut simplifier.context, eq, var, set, &conds);
                return Ok((filtered, steps));
            }
            return Ok((set, steps));
        }
    }
    result
}

/// `|affine(x)| {op} c` with a VAR-FREE parameter `c` of UNDECIDABLE sign
/// (`abs(x) = a`, `abs(x) > a`): the unconditional split assumed `c ≥ 0` for the
/// equation (`{a, −a}` — spurious for `a < 0`) while the inequality paths assumed
/// `c < 0` (`> a` → AllReals, `< a` → No solution, `≤ a` → the degenerate
/// `[a,a] ∪ [−a,−a]`). Emit the parameter-space-correct forms instead, built
/// DIRECTLY (never through the set algebra, whose merge cannot order symbolic
/// endpoints like `c` vs `−c`):
///   `>` / `≥`: the two-ray union — universally correct for EVERY real `c`
///   (for `c < 0` the rays overlap and cover ℝ).
///   `=` / `<` / `≤`: guarded by `c ≥ 0` / `c > 0` / `c ≥ 0` (the established
///   single-case Conditional convention, as in `e^x > a`).
/// A parameter with a PROVEN sign keeps its existing (correct) owners.
fn try_solve_abs_vs_symbolic_param(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<Result<SolutionSet, CasError>> {
    use cas_ast::RelOp;
    use cas_ast::{BoundType, Case, ConditionPredicate, ConditionSet, Constant, Interval};
    use cas_math::numeric_eval::as_rational_const;
    use cas_math::polynomial::Polynomial;
    use cas_solver_core::isolation_utils::contains_var;
    use num_traits::{Signed, Zero};

    if !matches!(
        eq.op,
        RelOp::Eq | RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq
    ) {
        return None;
    }
    // Var side / const side; peel a rational coefficient off the abs (`2·|x| = a`).
    let (var_side, c) = if contains_var(&simplifier.context, eq.lhs, var)
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
    // Peel an ADDITIVE var-free constant off the var side (`|f|+k op a` → `|f| op a−k`):
    // move every var-free term to the threshold. The var side must reduce to EXACTLY
    // ONE var-carrying term (the abs, bare or coef-scaled) — `|x|+|x−1| op a` (two abs
    // terms) and `|x|+x op a` (abs plus a bare polynomial) decline, preserving their
    // honest residual. Adding a constant across a relation never flips the operator.
    let (var_side, c) = match simplifier.context.get(var_side).clone() {
        Expr::Add(_, _) | Expr::Sub(_, _) => {
            fn collect(ctx: &Context, e: ExprId, positive: bool, out: &mut Vec<(ExprId, bool)>) {
                match ctx.get(e).clone() {
                    Expr::Add(l, r) => {
                        collect(ctx, l, positive, out);
                        collect(ctx, r, positive, out);
                    }
                    Expr::Sub(l, r) => {
                        collect(ctx, l, positive, out);
                        collect(ctx, r, !positive, out);
                    }
                    _ => out.push((e, positive)),
                }
            }
            let mut terms = Vec::new();
            collect(&simplifier.context, var_side, true, &mut terms);
            let mut var_term: Option<ExprId> = None;
            let mut rest: Option<ExprId> = None; // accumulated var-free terms (with sign)
            for (t, positive) in terms {
                if contains_var(&simplifier.context, t, var) {
                    if var_term.is_some() || !positive {
                        return None; // >1 var term, or a negated abs term: decline honestly
                    }
                    var_term = Some(t);
                } else {
                    let signed = if positive {
                        t
                    } else {
                        simplifier.context.add(Expr::Neg(t))
                    };
                    rest = Some(match rest {
                        None => signed,
                        Some(acc) => simplifier.context.add(Expr::Add(acc, signed)),
                    });
                }
            }
            let vt = var_term?;
            let new_c = match rest {
                None => c,
                Some(k) => {
                    let d = simplifier.context.add(Expr::Sub(c, k));
                    simplifier.simplify(d).0
                }
            };
            (vt, new_c)
        }
        _ => (var_side, c),
    };
    let (abs_call, mut op, c) = match simplifier.context.get(var_side).clone() {
        Expr::Mul(l, r) => {
            let (coef, inner) = if contains_var(&simplifier.context, r, var) {
                (l, r)
            } else {
                (r, l)
            };
            let q = as_rational_const(&simplifier.context, coef)?;
            if q.is_zero() {
                return None;
            }
            let q_node = simplifier.context.add(Expr::Number(q.clone()));
            let scaled = simplifier.context.add(Expr::Div(c, q_node));
            let scaled = simplifier.simplify(scaled).0;
            let op = if q.is_negative() {
                cas_solver_core::isolation_utils::flip_inequality(eq.op.clone())
            } else {
                eq.op.clone()
            };
            (inner, op, scaled)
        }
        _ => (var_side, eq.op.clone(), c),
    };
    let _ = &mut op;
    let f = match_abs_argument(&simplifier.context, abs_call)?;
    if !contains_var(&simplifier.context, f, var) {
        return None;
    }
    // The parameter must be genuinely UNDECIDABLE: not a plain number, no exact
    // oracle verdict, and no structural positivity proof either way.
    if as_rational_const(&simplifier.context, c).is_some() {
        return None;
    }
    let sign_known = cas_math::root_forms::provable_sign_vs_zero(&simplifier.context, c).is_some()
        || cas_math::const_sign::provable_const_sign(&simplifier.context, c).is_some()
        || matches!(
            crate::solver_entrypoints_proof_verify::prove_positive(
                &simplifier.context,
                c,
                crate::runtime::ValueDomain::RealOnly,
            ),
            cas_solver_core::domain_proof::Proof::Proven
        )
        || {
            let neg_c = simplifier.context.add(Expr::Neg(c));
            matches!(
                crate::solver_entrypoints_proof_verify::prove_positive(
                    &simplifier.context,
                    neg_c,
                    crate::runtime::ValueDomain::RealOnly,
                ),
                cas_solver_core::domain_proof::Proof::Proven
            )
        };
    if sign_known {
        return None;
    }
    // AFFINE argument with rational slope: endpoints invert in closed form and
    // their ORDER is decided by the (rational) slope, never by symbolic compare.
    // A NON-AFFINE argument (`abs(x²−1) < a`, `abs(ln(x)) < a`) cannot: the
    // generic path fabricated garbage intervals with symbolic surd endpoints
    // (`(−√(a+1), −√(1−a))`), unguarded four-root equation sets, or a false
    // "No solution" — DECLINE honestly instead.
    let non_affine_decline = || {
        Some(Err(CasError::SolverError(
            "Inequalities with symbolic coefficients not yet supported".to_string(),
        )))
    };
    let Ok(f_poly) = Polynomial::from_expr(&simplifier.context, f, var) else {
        return non_affine_decline();
    };
    if f_poly.degree() != 1 {
        return non_affine_decline();
    }
    let q = f_poly.coeffs.get(1).cloned()?;
    let r = f_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(num_rational::BigRational::zero);
    if q.is_zero() {
        return None;
    }
    // x = (t − r)/q for t ∈ {c, −c}.
    let invert = |simplifier: &mut Simplifier, t: ExprId| -> ExprId {
        let r_node = simplifier.context.add(Expr::Number(r.clone()));
        let shifted = simplifier.context.add(Expr::Sub(t, r_node));
        let q_node = simplifier.context.add(Expr::Number(q.clone()));
        let out = simplifier.context.add(Expr::Div(shifted, q_node));
        simplifier.simplify(out).0
    };
    let neg_c = {
        let n = simplifier.context.add(Expr::Neg(c));
        simplifier.simplify(n).0
    };
    let from_c = invert(simplifier, c);
    let from_neg_c = invert(simplifier, neg_c);
    // Interval orientation by the RATIONAL slope: q > 0 keeps c on the upper side.
    let (lo, hi) = if q.is_positive() {
        (from_neg_c, from_c)
    } else {
        (from_c, from_neg_c)
    };
    let inf = simplifier.context.add(Expr::Constant(Constant::Infinity));
    let neg_inf = {
        let i = simplifier.context.add(Expr::Constant(Constant::Infinity));
        simplifier.context.add(Expr::Neg(i))
    };
    let set = match op {
        RelOp::Eq => SolutionSet::Conditional(vec![Case::new(
            ConditionSet::single(ConditionPredicate::NonNegative(c)),
            SolutionSet::Discrete(vec![from_c, from_neg_c]),
        )]),
        RelOp::Gt | RelOp::Geq => {
            let (outer, inner_bound) = if matches!(op, RelOp::Gt) {
                (BoundType::Open, BoundType::Open)
            } else {
                (BoundType::Open, BoundType::Closed)
            };
            SolutionSet::Union(vec![
                Interval {
                    min: neg_inf,
                    min_type: outer.clone(),
                    max: lo,
                    max_type: inner_bound.clone(),
                },
                Interval {
                    min: hi,
                    min_type: inner_bound.clone(),
                    max: inf,
                    max_type: outer.clone(),
                },
            ])
        }
        RelOp::Lt => SolutionSet::Conditional(vec![Case::new(
            ConditionSet::single(ConditionPredicate::Positive(c)),
            SolutionSet::Continuous(Interval::open(lo, hi)),
        )]),
        RelOp::Leq => SolutionSet::Conditional(vec![Case::new(
            ConditionSet::single(ConditionPredicate::NonNegative(c)),
            SolutionSet::Continuous(Interval::closed(lo, hi)),
        )]),
        _ => return None,
    };
    Some(Ok(set))
}

/// Top-level twin of the isolation-dispatch parametric guard, for the routes that
/// BYPASS it: the pre-strategy factored linear-collect (`(a²+1)·x > b` → the
/// equation-only kernel dropped the operator to a DISCRETE boundary), the
/// even-root threshold correction (`√x < a` → assumed `a ≥ 0` and squared to
/// `[0, a²)`), and the constant-numerator division (`a/x > 1` → `(0, a)`).
/// An ORDER relation whose next monotone step runs through a VAR-FREE NON-NUMERIC
/// constant either transforms EXACTLY (proven sign: positive keeps direction,
/// negative flips) or declines honestly with the canonical symbolic-inequality
/// message. Numeric coefficients never match (zero churn on historical routes).
fn try_parametric_monotone_guard(
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
    // Var side / const side orientation.
    let (lhs, rhs) = if contains_var(&simplifier.context, eq.lhs, var)
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
    let var_free_non_numeric = |ctx: &Context, e: ExprId| -> bool {
        !contains_var(ctx, e, var) && as_rational_const(ctx, e).is_none()
    };
    // Exact tri-state sign of a var-free constant-ish expression: surd/transcendental
    // oracles first, then the structural positivity prover (`a² + 1`).
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
                matches!(
                    crate::solver_entrypoints_proof_verify::prove_positive(
                        &simplifier.context,
                        e,
                        crate::runtime::ValueDomain::RealOnly,
                    ),
                    cas_solver_core::domain_proof::Proof::Proven
                )
                .then_some(std::cmp::Ordering::Greater)
            })
    };
    let decline = || {
        Some(Err(CasError::SolverError(
            "Inequalities with symbolic coefficients not yet supported".to_string(),
        )))
    };
    enum Action {
        Scale {
            kept: ExprId,
            factor: ExprId,
            rhs_multiplies: bool,
        },
        DeclineIfUndecidable {
            probe: ExprId,
        },
    }
    let action = match simplifier.context.get(lhs).clone() {
        Expr::Mul(l, r) => {
            let (kept, factor) = if contains_var(&simplifier.context, l, var)
                && var_free_non_numeric(&simplifier.context, r)
            {
                (l, r)
            } else if contains_var(&simplifier.context, r, var)
                && var_free_non_numeric(&simplifier.context, l)
            {
                (r, l)
            } else {
                return None;
            };
            Action::Scale {
                kept,
                factor,
                rhs_multiplies: false,
            }
        }
        Expr::Div(num, den) => {
            if contains_var(&simplifier.context, num, var)
                && var_free_non_numeric(&simplifier.context, den)
            {
                Action::Scale {
                    kept: num,
                    factor: den,
                    rhs_multiplies: true,
                }
            } else if contains_var(&simplifier.context, den, var)
                && var_free_non_numeric(&simplifier.context, num)
            {
                Action::DeclineIfUndecidable { probe: num }
            } else {
                return None;
            }
        }
        Expr::Pow(base, exp) => {
            let is_sqrt = as_rational_const(&simplifier.context, exp)
                .map(|q| q == num_rational::BigRational::new(1.into(), 2.into()))
                .unwrap_or(false);
            if is_sqrt
                && contains_var(&simplifier.context, base, var)
                && var_free_non_numeric(&simplifier.context, rhs)
            {
                Action::DeclineIfUndecidable { probe: rhs }
            } else {
                return None;
            }
        }
        Expr::Function(fn_id, args) => {
            if args.len() == 1
                && simplifier
                    .context
                    .is_builtin(fn_id, cas_ast::BuiltinFn::Sqrt)
                && contains_var(&simplifier.context, args[0], var)
                && var_free_non_numeric(&simplifier.context, rhs)
            {
                Action::DeclineIfUndecidable { probe: rhs }
            } else {
                return None;
            }
        }
        _ => return None,
    };
    match action {
        Action::Scale {
            kept,
            factor,
            rhs_multiplies,
        } => match sign_of(simplifier, factor) {
            Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Less) => {
                let negative =
                    matches!(sign_of(simplifier, factor), Some(std::cmp::Ordering::Less));
                let combined = if rhs_multiplies {
                    simplifier.context.add(Expr::Mul(rhs, factor))
                } else {
                    simplifier.context.add(Expr::Div(rhs, factor))
                };
                let new_rhs = simplifier.simplify(combined).0;
                let new_op = if negative {
                    cas_solver_core::isolation_utils::flip_inequality(eq.op.clone())
                } else {
                    eq.op.clone()
                };
                let reduced = Equation {
                    lhs: kept,
                    rhs: new_rhs,
                    op: new_op,
                };
                Some(crate::solver_entrypoints_solve::solve(
                    &reduced, var, simplifier,
                ))
            }
            Some(std::cmp::Ordering::Equal) => None,
            None => decline(),
        },
        Action::DeclineIfUndecidable { probe } => match sign_of(simplifier, probe) {
            Some(_) => None,
            None => decline(),
        },
    }
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

fn solve_local_core_inner(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: CoreSolverOptions,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    // Solver-opaque function aliases (`log2`, `log10`, `csc`, `sec`, `cot`, `cbrt`) rewrite to
    // their canonical invertible forms up front, so every handler below sees solvable atoms
    // instead of erroring `función [...] no definida`.
    // `|f(x)| {op} c` with an undecidable-sign parameter: affine arguments emit the
    // parameter-correct guarded/universal forms; non-affine arguments decline
    // honestly (the generic path fabricated symbolic-endpoint garbage).
    if let Some(result) = try_solve_abs_vs_symbolic_param(simplifier, eq, var) {
        return result.map(|set| (set, Vec::new()));
    }
    // Parametric monotone guard: transform exactly on a proven sign or decline
    // honestly BEFORE any strategy (the factored linear-collect would otherwise
    // drop the operator for symbolic coefficients).
    if let Some(result) = try_parametric_monotone_guard(simplifier, eq, var) {
        return result;
    }
    // Var-on-BOTH-sides linear inequality with a symbolic-constant coefficient
    // (`x < x·ln2`): collect to `c1·x + c0`, decide sign(c1) exactly, recurse.
    if let Some(result) = try_symbolic_linear_coeff_inequality(simplifier, eq, var) {
        return result;
    }
    let nl = normalize_solver_function_aliases(&mut simplifier.context, eq.lhs);
    let nr = normalize_solver_function_aliases(&mut simplifier.context, eq.rhs);
    if nl.is_some() || nr.is_some() {
        let normalized = Equation {
            lhs: nl.unwrap_or(eq.lhs),
            rhs: nr.unwrap_or(eq.rhs),
            op: eq.op.clone(),
        };
        return solve_local_core(&normalized, var, simplifier, opts, ctx);
    }
    // `csc/sec/cot(g) = c`: reduce to the owning sin/cos solver (full periodic family).
    if let Some(set) = try_solve_reciprocal_trig_equation(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // `arcsin/arccos/arctan/sinh/cosh/tanh(g) = c`: apply the (range-gated)
    // inverse — the isolation dispatch has no inverse for these and errors.
    if let Some(set) = try_solve_inverse_trig_hyperbolic_equation(eq, var, simplifier) {
        return Ok((set, Vec::new()));
    }
    // `trig(a·x + b) = c` with `b` a SYMBOLIC shift (π-multiple, arctan, surd): the angle-addition
    // expansion / isolation would return only the principal root. Solve `trig(u) = c` for `u = a·x + b`
    // (full periodic family) and map back — BEFORE the bare handler simplifies (expands).
    if let Some(set) = try_solve_shifted_argument_trig(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // Bare trig equation `sin/cos/tan(x)=c` -> the full periodic family (before the unary-inverse
    // path, which would return only the principal root).
    if let Some((set, steps)) = try_solve_periodic_trig_equation_with_steps(eq, var, simplifier) {
        return Ok((set, steps));
    }
    // WEAK-BOUNDARY trig inequality `A·sin/cos(g) ⋚ c` with |c/A| ≥ 1: the range
    // [−1, 1] settles it without interval machinery — `2·sin(x) ≥ 2 ⇔ sin(x) = 1`
    // (full periodic family via the equation handler), `sin(3x) > 1 → ∅`,
    // `cos(2x) ≥ −2 → ℝ`. |c/A| < 1 declines honestly (needs the periodic
    // interval-union representation). Scout cycle-3 backlog #3: the bare form
    // worked; the coefficient/argument wrappers fell to the mutated-echo residual.
    if let Some(set) = try_solve_trig_weak_boundary_inequality(eq, var, simplifier) {
        return Ok((set, Vec::new()));
    }
    if equation_is_nonzero_const_over_polynomial(simplifier, eq)
        || equation_has_identically_zero_denominator(simplifier, eq)
    {
        return Ok((SolutionSet::Empty, Vec::new()));
    }
    // `c/(x+√2) {op} 0` on the RAW tree, before the simplifier rationalizes the surd
    // denominator through its conjugate and fabricates a spurious removable pole.
    if let Some(set) = try_solve_const_over_surd_affine_inequality(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // `c/(a·x+b)^(1/q) {op} k` on the RAW tree (scout #4): before the
    // simplifier rewrites `1/x^(1/3)` into the valley form `x^(2/3)/x`.
    if let Some(set) = try_solve_const_over_root_affine_inequality(simplifier, eq, var) {
        return Ok((set, Vec::new()));
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
    // A SUM of ≥2 sign forms `Σ cᵢ·sign(gᵢ) {op} k` (`(x+1)/|x+1| + (x-1)/|x-1| > 0`) is a step function;
    // partition ℝ at the `gᵢ = 0` poles and test each region. The single-sign handler below keeps the
    // `n = 1` case (this requires ≥ 2 sign terms).
    // PIU: `A·trig(g)² ⋚ c` and `A·|trig(g)| ⋚ c` — reduce the even power /
    // absolute value to a sign case split on `trig(g)` and combine windows
    // (`sin(x)² < 1/4` ⟺ `|sin(x)| < 1/2` ⟺ `sin > −1/2 ∩ sin < 1/2`).
    if let Some(set) = try_solve_even_power_or_abs_trig_inequality(eq, var, simplifier) {
        return Ok((set, Vec::new()));
    }
    // PIU P3b: `A / trig(g) ⋚ c` — reduce by sign cases to window relations
    // on `trig(g)` and combine with the circular same-period algebra
    // (`1/sin(x) > 2` ⟺ 0 < sin(x) < 1/2 → two windows per period).
    if let Some(set) = try_solve_reciprocal_trig_inequality(eq, var, simplifier) {
        return Ok((set, Vec::new()));
    }
    // Scout family C: `A/|g| ⋚ c` — the generic inversion lost the `g = 0`
    // pole (`1/|x| > 2 → (−1/2, 1/2)` including 0) and the c = 0 branch emitted
    // degenerate `(−∞,−∞)` endpoints. `A/|g| = |A/g|` exactly (for A > 0), and
    // the abs-threshold path over a RATIONAL inner argument already punctures
    // poles correctly — rewrite into that twin shape and recurse.
    if let Some(set) = try_solve_reciprocal_abs_inequality(eq, var, simplifier, opts, ctx) {
        return Ok((set, Vec::new()));
    }
    // `c/g {op} 0` with an abs INSIDE the denominator (`1/(|x|−1) < 0`) — the bare
    // `A/|g|` handler above declines (denominator is `|x|−1`, not a lone `abs`),
    // and the generic rational path returns garbage (`ℝ`, `(−∞,−∞)∪(∞,∞)`) because
    // it cannot find `g`'s zeros through the abs. Reduce to `g {op'} 0` (strict)
    // and let the abs solver handle it.
    if let Some(set) = try_solve_const_over_abs_denominator_vs_zero(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // `f/g {op} k` with k ≠ 0 where the quotient is not purely rational
    // (`1/(|x|−1) > 1`, `1/ln(x) > 2`, `|x|/(x−2) < 1`): denominator sign-split.
    // Dispatched after the bare-`A/|g|` and vs-zero owners; a polynomial/polynomial
    // quotient declines inside (owned by the rational path).
    if let Some(set) = try_solve_division_vs_const_sign_split(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // `f·g {op} 0` with a non-polynomial factor (`(x−1)·ln(x) < 0`): factor-sign
    // split on the RAW tree, before the prepass distributes the product away.
    if let Some(set) = try_solve_product_inequality_sign_split(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // `|f| {op} |g|`: polynomial args reduce to the exact polynomial inequality
    // f² − g² {op} 0 (its correct owner); non-polynomial args decline honestly
    // (the generic path fabricated a false "No solution" / mangled leaks).
    if let Some(result) = try_solve_abs_vs_abs_polynomial_inequality(simplifier, eq, var) {
        return result.map(|set| (set, Vec::new()));
    }
    // NESTED abs (`||x|−2| {op} x`): partition at the inner-abs zeros, reduce each
    // region to a plain abs relation, clip and union.
    if let Some(set) = try_solve_nested_abs_relation(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    if let Some(set) = try_solve_sign_sum_relation(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // `g/|g| {op} c` (or `|g|/g {op} c`) is `sign(g) {op} c`, sign ∈ {−1, +1} (undefined at g = 0).
    // Reduce to a sign condition on `g` so the OPEN intervals exclude the `g = 0` pole — the generic
    // path returned a CLOSED ray that wrongly includes the 0/0 point (`x/|x| = 1 → [0, ∞)`) or "No
    // solution" for the inequality forms.
    if let Some(set) = try_solve_sign_via_abs(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // `coeff·sign(g) + offset = h(x)` with a VARIABLE RHS (`x/|x| = x`): the sign
    // form is a step function, so the equation splits on `sign(g) = ±1`. The
    // constant-RHS forms are owned by `try_solve_sign_via_abs` above.
    if let Some(set) = try_solve_sign_form_equals_expr(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // A polynomial inequality with a SINGLE `|f|` term (`x² − 3|x| + 2 < 0`) —
    // the generic path treats the abs opaquely and returns a wrong "No
    // solution". Split at `f = 0` into the `|f| = ±f` branches. Placed after the
    // sign/reciprocal-abs handlers so those keep their forms; the constant-`c`
    // threshold handler below is unaffected (it has no polynomial remainder).
    // Inequality ops ONLY here: the equation form (`x·|x| = 4`) dispatches later,
    // after the isolated-abs and poly-in-|x| equation handlers own their forms.
    if matches!(
        eq.op,
        cas_ast::RelOp::Lt | cas_ast::RelOp::Leq | cas_ast::RelOp::Gt | cas_ast::RelOp::Geq
    ) {
        if let Some(set) = try_solve_single_abs_polynomial_relation(simplifier, eq, var) {
            return Ok((set, Vec::new()));
        }
    }
    // `|g(x)| {op} c` (constant `c`) reduces to the polynomial inequalities on the
    // two sides of the abs; the isolation/split path below drops the operator and
    // returns the boundary equation (`|x^2-2x| < 1` -> "No solution"). Handle it
    // before the sum-of-abs and isolation routing.
    if let Some(set) = try_solve_abs_threshold_inequality(eq, var, simplifier, opts, ctx) {
        return Ok((set, Vec::new()));
    }
    // A polynomial-in-`ln(x)` inequality `P(ln(x)) {op} 0` (`ln(x)^2 - 3·ln(x) + 2 < 0`, also the pure
    // `ln(x)^2 - 4 < 0`) is non-monotonic; the isolation path reports "No solution". Solve `P(u) {op} 0`
    // (u = ln x) and map the u-intervals back through `ln`. Runs before the pure-square handler, which
    // it subsumes (and which only matched a bare `coeff·ln^2` with the constant already on the RHS).
    if let Some(set) = try_solve_polynomial_in_log_inequality(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // `ln(x)^2 {op} c` is non-monotonic; the log-isolation path reports "All reals if
    // x>0". Reduce to the two single-`ln` inequalities before that path runs.
    if let Some(set) = try_solve_ln_square_inequality(eq, var, simplifier, opts, ctx) {
        return Ok((set, Vec::new()));
    }
    // Two-or-more affine `|f|` terms PLUS a degree-≥2 polynomial remainder
    // (`x² + |x−1| + |x+1| < 5`): the linear sum-of-abs handler below carries
    // only a linear remainder, so it declines and the generic path returns a
    // wrong "No solution". Partition at the breakpoints and solve the polynomial
    // relation per segment. Runs before the linear handler (which owns the
    // linear-remainder forms).
    if let Some(set) = try_solve_multi_abs_polynomial_relation(simplifier, eq, var) {
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
    // An even-numerator VALLEY power inequality `c·(a·x+b)^(p/q) + d {op} k` (p even, e = p/q > 0) is
    // `c·|a·x+b|^(p/q) + d {op} k`. SOLVE it exactly by reducing to `|a·x+b| {op'} ((k−d)/c)^(q/p)` —
    // two linear pieces of the affine argument — instead of declining. (`(x-1)^(2/3) > 4` →
    // `|x-1| > 8` → `(−∞,−7)∪(9,∞)`.)
    if let Some(set) = try_solve_even_power_valley_inequality(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // A power-monomial inequality `c·x^e {op} k` whose exponent makes the engine's monotonic
    // isolation UNSOUND — a NEGATIVE non-integer exponent like `1/x^(1/3) > 2` (a reciprocal
    // fractional power the valley reduction above does not cover) — is declined to an honest residual
    // before any handler emits a wrong single ray. Strictly-monotonic powers (`e > 0`, odd numerator:
    // `x^(1/3)`, `x^(3/2)`) and integer-exponent reciprocals (`1/x³`) are NOT declined.
    if let Some(set) = try_decline_unsound_power_monomial_inequality(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // Equations that are a polynomial of degree ≥ 2 in `x^(1/q)` (`x - 3·√x + 2`,
    // `x^(2/3) - x^(1/3) - 2`, …) are quadratics-in-disguise: the isolation path
    // reorients to `x = f(x)` and leaks a malformed `solve(...)` residual while
    // dropping every root. Solve them by `u = x^(1/q)` substitution here first.
    if let Some(set) = try_solve_rational_power_polynomial(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // LAURENT polynomials in `x^(1/q)` — a root mixed with its reciprocal
    // (`√x − 1/√x = 1`, `√x + 1/√x = 5/2`) — leak the same malformed residual
    // (`x = (…)^(1/(1/2))`). Clear the `1/u^k` by shifting and solve in `u = x^(1/q)`.
    if let Some(set) = try_solve_rational_power_laurent(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // Equations that are a polynomial of degree ≥ 2 in `ln(x)`
    // (`ln(x)^2 - ln(x) - 2 = 0`, …) leak the same way; solve them by the
    // `u = ln(x)` substitution.
    if let Some(set) = try_solve_polynomial_in_log(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // Equations that are a polynomial of degree ≥ 2 in `|x|` (`|x|² − 3·|x| + 2 = 0`,
    // stored as `x² − 3·|x| + 2` after `|x|² → x²`) leak the same way — the isolation
    // path reorients to `x = √(3·|x| − 2)`. Solve them by the `u = |x|` substitution
    // (with the `x² = |x|²` even-power unification) here first.
    if let Some(set) = try_solve_polynomial_in_abs(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // A single `|f(x)|` term with a NON-CONSTANT quadratic-or-higher remainder
    // (`x² + |x−1| − 3 = 0`, `|x−1| = 3 − x²`) is `|f| = g(x)`. Isolating the abs
    // is unsound: the generic path solves only `f = g` (dropping `f = −g`) and
    // skips `g ≥ 0`, returning a spurious root while missing a real one — or
    // leaking a malformed residual. Solve both branches and keep `g(r) ≥ 0`.
    if let Some(set) = try_solve_single_abs_equals_polynomial(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // A single `|f|` entangled MULTIPLICATIVELY with a polynomial (`x·|x| = 4`,
    // `x·|x| − x = 0`) is neither `|f| = g` (isolated) nor a pure polynomial-in-|x|
    // (the odd `x` factor is not a function of `|x|`); the isolation path reorients
    // to `x = 4/|x|` and leaks a malformed `solve(x − 4/|x| = 0)` residual. Split at
    // `f = 0` into the `|f| = ±f` polynomial branches, solve each, and keep the roots
    // in that branch's half-line. Placed after the isolated-abs and poly-in-|x|
    // equation handlers so they keep their forms (0 huella delta); equation ops only
    // (the inequality dispatch above owns those).
    if matches!(eq.op, cas_ast::RelOp::Eq) {
        if let Some(set) = try_solve_single_abs_polynomial_relation(simplifier, eq, var) {
            return Ok((set, Vec::new()));
        }
    }
    // Equations that mix an exponential with its RECIPROCAL (`e^x + e^(−x) = 2`, `2^x − 3 + 2^(1−x) = 0`)
    // are Laurent polynomials in `base^x`; the isolation path rewrites them via the cosh identity and
    // bails (`función [cosh] no definida` / `Cannot isolate`). Substitute `u = base^x`, clear the
    // reciprocal, and solve the polynomial in `u`. Pure positive-power forms decline (owned elsewhere).
    if let Some(set) = try_solve_exponential_reciprocal_polynomial(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // Equations that are a polynomial of degree ≥ 2 in a trig atom (`2·sin(x)² − 3·sin(x) + 1 = 0`)
    // leak an `arcsin(… − cos(2x) …)` residual once the double-angle identity fires; substitute
    // `u = sin(x)` (cos/tan) and back-substitute each root through the periodic solver.
    if let Some(set) = try_solve_polynomial_in_trig(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // A HOMOGENEOUS linear trig equation `a·sin(g) + b·cos(g) = 0` (`sin(x) = cos(x)`,
    // `√3·sin(x) − cos(x) = 0`) reduces to `tan(g) = −b/a`; the isolation path otherwise leaks an
    // `arcsin(cos(x)·…)` residual. The inhomogeneous `… = c` (c ≠ 0) declines.
    if let Some(set) = try_solve_homogeneous_linear_trig(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // `U(x)/√f = k`: normalize to the bare radical `√f = U/k` (square-and-verify owner).
    if let Some(set) = try_solve_poly_over_sqrt_equation(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // `trig(u) = trig(v)` same-function, degree-≥3 multiple angles: sum-to-product
    // → periodic product-zero. LAST-RESORT among the trig owners (after the
    // polynomial-in-trig expansion, which keeps its more-folded presentations for
    // the shapes it already solves); without this the isolation leaks the
    // self-referential arcsin echo.
    if let Some(set) = try_solve_trig_sum_to_product_equation(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // An INHOMOGENEOUS linear trig equation `a·sin(g) + b·cos(g) = c` (`3·sin(x) + 4·cos(x) = 5`,
    // `sin(x) + cos(x) = 1`) reduces by the auxiliary angle to `sin(g + arctan(b/a)) = c/√(a²+b²)`; the
    // isolation path otherwise leaks an `arcsin(… − cos(x) …)` residual.
    if let Some(set) = try_solve_inhomogeneous_linear_trig(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // `|A(x)| = c` with a trig-bearing argument: the generic abs isolation solves the two branches to
    // PRINCIPAL roots (`|2·sin(x)−1| = 1 → {π/2, 0}`); solve each branch fully so trig stays periodic.
    if let Some(set) = try_solve_abs_of_trig_equation(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // `|E| = 0 ⟺ E = 0`: dispatch the argument's full zero-set (the generic abs isolation drops all but
    // the first factor of a product, `|x·(x−2)| = 0 → {0}` instead of `{0, 2}`).
    if let Some(set) = try_solve_abs_equals_zero(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // `|f(x)| = g(x)` with a degree-≥2 polynomial `f` and a variable RHS (`|x²−1| = x+1`): split into
    // `f = ±g` and verify each root against the original (enforcing `g ≥ 0`). Linear `|f|` and
    // constant-RHS forms keep their existing handlers.
    if let Some(set) = try_solve_abs_polynomial_equation(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // A sum of two square roots equal to a constant (`√(x+3) + √x = 3`) leaks
    // the same isolation residual; reduce by squaring and verify exactly.
    if let Some(set) = try_solve_sum_of_two_radicals_equation(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // A single radical `√(quadratic+) = polynomial` (`√(5x²+9x−2) = 3x`): the
    // isolation core mis-filters after squaring (wrong "No solution", or a dropped
    // root). Square to `f − g² = 0`, solve, and keep roots with `g(r) ≥ 0`.
    if let Some(set) = try_solve_single_radical_equals_polynomial(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // Radical INEQUALITIES `√f {<,≤,>,≥} g`: solve by the correct case split,
    // not by squaring blindly (which loses the RHS-sign branches and gives
    // wrong answers like `√x < x-2 → [0,1) ∪ (4,∞)`).
    if let Some(set) = try_solve_radical_inequality(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // A polynomial-in-`x^(1/q)` inequality (`x − 3·√x + 2 < 0`, a quadratic in `√x`) is non-monotonic;
    // the isolation path emits an honest-but-incomplete residual. Solve `P(u) {op} 0` (u = x^(1/q)) and
    // map the u-intervals back through `x = u^q` (`u ≥ 0` domain for even q). Runs AFTER the valley /
    // monomial-decline / single-radical handlers so a bare monomial (`x^(2/3) > 2`) or a radical-vs-linear
    // (`√x < x/2 − 3`) keeps their cleaner rendering; this only catches the genuine mixed quadratics.
    if let Some(set) = try_solve_rational_power_polynomial_inequality(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // `A(x) {op} B(x)` with BOTH sides carrying the variable and a RATIONAL difference: move everything
    // to one side so the RHS is the constant 0 and the verified `N/D {op} 0` path below applies. The
    // two-sided form `1/(x-1) > 1/(x+1)` otherwise reached a path that emitted a garbage `inf^(1/2)`
    // bound when the difference numerator is a nonzero constant (`→ 2/(x²-1) > 0`), even though the
    // explicit-difference form `1/(x-1) - 1/(x+1) > 0` solved correctly. Gated to a rational difference,
    // so radical / exponential / trig two-sided inequalities (handled above) are not preempted, and to
    // a denominator of degree ≥ 1 (a polynomial difference declines and falls through to its own path).
    if matches!(
        eq.op,
        cas_ast::RelOp::Lt | cas_ast::RelOp::Leq | cas_ast::RelOp::Gt | cas_ast::RelOp::Geq
    ) && cas_solver_core::isolation_utils::contains_var(&simplifier.context, eq.lhs, var)
        && cas_solver_core::isolation_utils::contains_var(&simplifier.context, eq.rhs, var)
    {
        let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
        if split_rational_inequality_lhs(&mut simplifier.context, diff, var)
            .is_some_and(|(_, den)| den.degree() >= 1)
        {
            let zero = simplifier.context.num(0);
            let reduced = Equation {
                lhs: diff,
                rhs: zero,
                op: eq.op.clone(),
            };
            if let Some(set) = try_solve_rational_constant_inequality(simplifier, &reduced, var) {
                return Ok((set, Vec::new()));
            }
        }
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
    // A bare `sin(x)`/`cos(x)` inequality at the EXACT range boundary `±1`: the touch side
    // (`sin(x) ≥ 1`) is the periodic point set `{π/2 + 2kπ}` (reduce to the boundary equation); the
    // complement side (`sin(x) < 1`) is `ℝ` minus those points → honest residual. Otherwise the generic
    // inversion emits a wrong ray (`[π/2, ∞)`). Runs before the decline below so these are not lumped in.
    if let Some(set) = try_solve_boundary_trig_inequality(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // A periodic `sin`/`cos`/`tan` inequality has a periodic-union solution the engine cannot
    // represent; decline to an honest residual instead of a wrong ray (out-of-range bare sin/cos are
    // excluded — they are answered ℝ/∅ by the trig-range guard after solve_inner).
    if let Some(set) = try_decline_periodic_trig_inequality(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // Mixed integer bases that share a common prime (`4^x − 3·2^x + 2`, `9^x − 4·3^x + 3`): rewrite each
    // `m^g` to `p^(k·g)` (`4^x → 2^(2x)`) so it is a polynomial in the single atom `p^x`, then re-solve.
    // Otherwise the isolation reports "Cannot isolate: variable on both sides" (two distinct bases).
    if let Some(set) = try_solve_via_exp_base_normalization(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // Two exponentials with DIFFERENT (incompatible-prime) bases (`4^x − 9^x = 0`, `5·2^x = 3^x`):
    // `A·M^x + B·N^x = 0 ⟺ (M/N)^x = −B/A`, i.e. `x = ln(−B/A)/ln(M/N)`. The A=B forms happen to
    // isolate; the one-sided / both-coefficiented forms otherwise error with "Cannot isolate".
    if let Some(set) = try_solve_two_different_base_exponential_equation(simplifier, eq, var) {
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
    let mut conds = ctx.required_conditions();
    // RADICAL-EQUATION RANGE CONDITION: an equation reducible to a single isolated radical
    // `s·√f + rest = 0` ⟺ `√f = g` (g = −rest/s) carries the range constraint `g ≥ 0` (√ is
    // nonnegative). Squaring loses it, so the solver returns BOTH quadratic roots — e.g.
    // `√(x+1) = −x` yields `{φ, ½(1−√5)}` but `φ > 0` makes `−x < 0`, an extraneous root. Recording
    // `NonNegative(g)` lets the EXACT surd-sign prover in `root_violates_required_condition` drop it.
    // This is always sound: a genuine root has `g = √f ≥ 0`, so it can never violate; the prover only
    // ever drops on a proof (a `None` keeps the root). The radicand's own `f ≥ 0` is already recorded.
    if eq.op == cas_ast::RelOp::Eq {
        let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
        let (d, _) = simplifier.simplify(diff);
        if let Some((s, _f, rest)) = collect_radical_split(&simplifier.context, d, var) {
            if !rest.is_empty()
                && !rest
                    .iter()
                    .any(|&(_, t)| expr_contains_sqrt(&simplifier.context, t))
            {
                let mut r = simplifier.context.num(0);
                for (sg, term) in rest {
                    r = if sg >= 0 {
                        simplifier.context.add(Expr::Add(r, term))
                    } else {
                        simplifier.context.add(Expr::Sub(r, term))
                    };
                }
                // `s·√f = −rest` ⟹ `√f = −rest/s`; with `s = ±1`, `g = −r` (s>0) or `g = r` (s<0).
                let g = if s >= 0 {
                    simplifier.context.add(Expr::Neg(r))
                } else {
                    r
                };
                let (g, _) = simplifier.simplify(g);
                let cond = ImplicitCondition::NonNegative(g);
                if !conds.contains(&cond) {
                    conds.push(cond);
                }
            }
        }
    }
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
        try_solve_biquadratic(simplifier, eq, var, opts.value_domain.is_real_only()).unwrap_or(set)
    } else {
        set
    };

    // A polynomial whose deflated quartic factor splits into two rational quadratics
    // (`x⁵-5x³+x²-5 = (x+1)(x²-5)(x²-x+1)` drops the `±√5` roots): peel the rational roots and solve
    // the quadratic factors. Replaces a `Residual`/`Conditional`; augments a `Discrete` the normal
    // path left incomplete (only the rational roots) when the quartic factor adds genuinely new roots.
    let set = match try_solve_polynomial_with_quartic_factor(
        simplifier,
        eq,
        var,
        opts.value_domain.is_real_only(),
    ) {
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

    // A degree>=3 polynomial equation with SYMBOLIC coefficients (`x³+p·x+q = 0`) has no
    // closed-form path here (`Polynomial` stores rational coeffs; Cardano is rational-only), so
    // base-side power isolation takes the n-th root of both sides UNCONDITIONALLY -- unlike the
    // exponent-side path, it has no "rhs still has the variable" progress guard -- and leaks a
    // self-referential `solve(x − (−p·x − q)^(1/3) = 0, x)`. That mangled operator is neither the
    // symbolic Cardano roots nor an honest decline. When the ORIGINAL `lhs − rhs` is a genuine
    // polynomial in `var` (non-negative integer powers, coefficients possibly symbolic) of degree
    // >= 3, replace the leak with the honest one-sided echo of the ORIGINAL equation. Gated on
    // Residual/Conditional so every productive path already ran first: a numeric cubic (Cardano ->
    // Discrete), a biquadratic (surd substitution -> Discrete), `x²=√x` (-> Discrete), and
    // `x²=2^x` (not a polynomial in `x` -> degree walker returns None) are all untouched.
    let set = if eq.op == cas_ast::RelOp::Eq
        && matches!(set, SolutionSet::Residual(_) | SolutionSet::Conditional(_))
    {
        let diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
        let (diff, _) = simplifier.simplify(diff);
        match symbolic_poly_degree_in_var(&simplifier.context, diff, var) {
            Some(degree) if degree >= 3 => cas_solver_core::solve_outcome::residual_solution_set(
                &mut simplifier.context,
                eq.lhs,
                eq.rhs,
                eq.op.clone(),
                var,
            ),
            _ => set,
        }
    } else {
        set
    };

    // An IRREDUCIBLE polynomial inequality (`x³+x+1 > 0`, `x³-3x+1 > 0`) is rewritten to `Equal(p,0)`
    // by the normal path, dropping the operator and returning the equation's root SET (so `> 0` and
    // `< 0` give identical output). When the operator is an inequality and the result is a `Discrete`
    // root set, recover the interval solution by sign analysis over those (now closed-form) real roots.
    // An ODD-degree poly with a rational root and a positive-definite even residual (`x⁵-1 =
    // (x-1)(x⁴+x³+x²+x+1)`) DECLINES the inequality to `Empty`/`Residual` even though the EQUATION
    // path finds the real roots ({1}); re-solve `p = 0` for the roots and run the same sign analysis
    // (its alternation + end-behaviour guards keep it sound on an incomplete root set).
    let set = if matches!(
        eq.op,
        cas_ast::RelOp::Lt | cas_ast::RelOp::Leq | cas_ast::RelOp::Gt | cas_ast::RelOp::Geq
    ) {
        match &set {
            SolutionSet::Discrete(roots) => {
                let roots = roots.clone();
                try_polynomial_inequality_sign_analysis(simplifier, eq, var, &roots).unwrap_or(set)
            }
            SolutionSet::Empty | SolutionSet::Residual(_) | SolutionSet::Conditional(_) => {
                match polynomial_equation_real_roots(simplifier, eq, var) {
                    Some(roots) => {
                        try_polynomial_inequality_sign_analysis(simplifier, eq, var, &roots)
                            .unwrap_or(set)
                    }
                    None => set,
                }
            }
            _ => set,
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
/// Solve the EQUATION form `lhs = rhs` of an inequality and return its discrete real roots, or
/// `None` if it does not reduce to a finite real root set. Lets the inequality sign analysis run over
/// the equation's roots when the inequality path itself declined to `Empty`/`Residual`.
fn polynomial_equation_real_roots(
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
fn symbolic_poly_degree_in_var(ctx: &cas_ast::Context, expr: ExprId, var: &str) -> Option<u32> {
    let mut max_degree = 0u32;
    for term in cas_math::expr_nary::add_leaves(ctx, expr) {
        max_degree = max_degree.max(term_degree_in_var(ctx, term, var)?);
    }
    Some(max_degree)
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
    is_real_only: bool,
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
        if is_real_only {
            // Complex z roots ⇒ no real x.
            return Some(SolutionSet::Empty);
        }
        // ComplexEnabled: `x = ±√(a+bi)` is block-B machinery — decline to an
        // honest residual instead of a wrong Empty.
        return None;
    }

    // Build the exact `z = (−b ± √disc)/(2a)`, then `x = ±√z` for each non-negative z root.
    let ctx = &mut simplifier.context;
    let num = |ctx: &mut cas_ast::Context, v: BigRational| ctx.add(Expr::Number(v));
    let disc_node = num(ctx, disc);
    let sqrt_disc = sqrt_expr(ctx, disc_node);
    let neg_b = num(ctx, -&b);
    let two_a = num(ctx, &a * r(2));
    let mut raw_roots: Vec<ExprId> = Vec::new();
    let mut exact_complex_roots: Vec<ExprId> = Vec::new();
    for s in [1.0f64, -1.0f64] {
        let z_f = (-bf + s * disc_f.sqrt()) / (2.0 * af);
        let negative_z = z_f < -1e-12;
        if negative_z && is_real_only {
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
        if negative_z {
            // ComplexEnabled: `±√z` with z < 0 folds to the pure-imaginary
            // pair downstream — exact, bypasses the f64 verification.
            exact_complex_roots.push(sqrt_z);
            exact_complex_roots.push(neg_sqrt_z);
        } else {
            raw_roots.push(sqrt_z);
            raw_roots.push(neg_sqrt_z);
        }
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
    // ComplexEnabled: append the z<0 pure-imaginary pairs (exact, no f64 verify).
    for raw in exact_complex_roots {
        let (root, _) = simplifier.simplify(raw);
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
/// Exact integer square root of `n`: `Some(r)` with `r ≥ 0` iff `n = r²`, else `None` (negative or
/// non-perfect-square). No float in the keep/reject decision — the `f64` seed is only a starting point
/// that is corrected to the exact integer value before the `r*r == n` test.
fn exact_i64_sqrt(n: i64) -> Option<i64> {
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
    is_real_only: bool,
) -> Option<SolutionSet> {
    use cas_math::polynomial::Polynomial;
    use cas_solver_core::quadratic_formula::sqrt_expr;
    use cas_solver_core::rational_roots::{find_rational_roots, rational_to_expr};
    use num_rational::BigRational;
    use num_traits::{ToPrimitive, Zero};
    use std::collections::HashMap;
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
    // ComplexEnabled: append the disc<0 conjugate pairs (exact, no f64 verify).
    for raw in exact_complex_roots {
        let (root, _) = simplifier.simplify(raw);
        roots.push(root);
    }
    if roots.is_empty() {
        return Some(SolutionSet::Empty);
    }
    Some(SolutionSet::Discrete(roots))
}

/// Solve `coeff·sign(g) + offset = h(x)` with a VARIABLE RHS `h` (EQUATION only) by
/// the step-function split: `sign(g) ∈ {−1, +1}`, so the equation holds where
/// `h = coeff + offset` on `g > 0`, OR where `h = −coeff + offset` on `g < 0`.
/// `x/|x| = x` (`sign(x) = x`) → `{1} ∪ {−1} = {−1, 1}` (the pole `x = 0` is excluded
/// by the STRICT `g`-branch); `x/|x| = −x` → `∅ ∪ ∅ = No solution`. The generic
/// isolation instead clears the denominator to `x = x·|x|` and leaks a malformed
/// residual. Constant-RHS forms stay with `try_solve_sign_via_abs`; a sign form on
/// BOTH sides is left to the sign-sum handler.
fn try_solve_sign_form_equals_expr(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    use cas_solver_core::isolation_utils::contains_var;
    use cas_solver_core::solution_set::{intersect_solution_sets, union_solution_sets};

    if eq.op != RelOp::Eq {
        return None;
    }
    // One side is a pure sign form `coeff·sign(g) + offset`; the other side `h`
    // contains the variable but is NOT itself a sign form.
    let (g, coeff, offset, h) =
        if let Some((g, c, o)) = sign_form_coeff_offset(simplifier, eq.lhs, var) {
            if !contains_var(&simplifier.context, eq.rhs, var) {
                return None; // constant RHS: `try_solve_sign_via_abs` owns it
            }
            (g, c, o, eq.rhs)
        } else if let Some((g, c, o)) = sign_form_coeff_offset(simplifier, eq.rhs, var) {
            if !contains_var(&simplifier.context, eq.lhs, var) {
                return None;
            }
            (g, c, o, eq.lhs)
        } else {
            return None;
        };
    if sign_form_coeff_offset(simplifier, h, var).is_some() {
        return None; // sign(g) = sign(h): the sign-sum handler's job
    }

    let zero = simplifier.context.num(0);
    // `sign(g) = +1` branch: `h = coeff + offset` restricted to `g > 0`.
    let pos_target = simplifier.context.add(Expr::Number(&coeff + &offset));
    let pos_roots = solve_relation_set(simplifier, var, h, pos_target, RelOp::Eq)?;
    let pos_domain = solve_relation_set(simplifier, var, g, zero, RelOp::Gt)?;
    let pos = intersect_solution_sets(&simplifier.context, pos_roots, pos_domain);
    // `sign(g) = −1` branch: `h = −coeff + offset` restricted to `g < 0`.
    let neg_target = simplifier.context.add(Expr::Number(-&coeff + &offset));
    let neg_roots = solve_relation_set(simplifier, var, h, neg_target, RelOp::Eq)?;
    let neg_domain = solve_relation_set(simplifier, var, g, zero, RelOp::Lt)?;
    let neg = intersect_solution_sets(&simplifier.context, neg_roots, neg_domain);
    Some(union_solution_sets(&simplifier.context, pos, neg))
}

/// Return the `abs(arg)` argument of `x` (a unary `|·|` call), or None.
fn abs_call_arg(ctx: &Context, x: ExprId) -> Option<ExprId> {
    use cas_ast::BuiltinFn;
    if let Expr::Function(fn_id, args) = ctx.get(x) {
        if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Abs) {
            return Some(args[0]);
        }
    }
    None
}

/// Detect `e = c · sign(g)` where the sign is written as `g/|g|` or `|g|/g`, and `c` is a NONZERO
/// rational coefficient peeled from a leading `Neg`, an outer constant `Mul`, or a coefficiented
/// numerator/denominator. Returns `(g, c)` with `g` carrying the variable. This generalizes the bare
/// `g/|g|` (`c = 1`) form so `-x/|x|` (`c = -1`), `3x/|x|` (`c = 3`), `|x|/(-x)` (`c = -1`) all reduce
/// to a sign condition — the coefficient just rescales the constant RHS (`c·sign(g) {op} k`
/// ⟺ `sign(g) {op} k/c`, flipping a strict op when `c < 0`).
fn sign_form_coeff(
    simplifier: &mut Simplifier,
    e: ExprId,
    var: &str,
) -> Option<(ExprId, num_rational::BigRational)> {
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;
    use num_traits::Zero;

    match simplifier.context.get(e).clone() {
        Expr::Neg(inner) => {
            let (g, c) = sign_form_coeff(simplifier, inner, var)?;
            Some((g, -c))
        }
        Expr::Mul(a, b) => {
            // Peel a constant factor on either side; the other factor must be the sign form.
            if let Some(k) = as_rational_const(&simplifier.context, a) {
                if k.is_zero() {
                    return None;
                }
                let (g, c) = sign_form_coeff(simplifier, b, var)?;
                Some((g, k * c))
            } else if let Some(k) = as_rational_const(&simplifier.context, b) {
                if k.is_zero() {
                    return None;
                }
                let (g, c) = sign_form_coeff(simplifier, a, var)?;
                Some((g, k * c))
            } else {
                None
            }
        }
        Expr::Div(num, den) => {
            // Peel a rational coefficient from BOTH sides first: a scaled sign form
            // `2·|x|/x` or `−|x|/x` simplifies to `Div(Mul(2, |x|), x)` /
            // `Div(Neg(|x|), x)`, so the bare-abs `abs_call_arg` on the raw numerator
            // fails and the whole sign form is missed (the coefficient/negation sibling
            // of the working `|x|/x`). Fold the peeled `nc/dc` into the returned coeff.
            let (nc, num_core) = peel_rational_coefficient(&simplifier.context, num);
            let (dc, den_core) = peel_rational_coefficient(&simplifier.context, den);
            if dc.is_zero() {
                return None;
            }
            let scale = nc / dc;
            if scale.is_zero() {
                return None;
            }
            let den_abs = abs_call_arg(&simplifier.context, den_core);
            let num_abs = abs_call_arg(&simplifier.context, num_core);
            if let Some(a) = den_abs {
                // `num_core/|a| = (num_core/a)·sign(a)`; `num_core/a` must fold to a nonzero rational.
                if !contains_var(&simplifier.context, a, var) {
                    return None;
                }
                let ratio = simplifier.context.add(Expr::Div(num_core, a));
                let (ratio, _) = simplifier.simplify(ratio);
                let c = as_rational_const(&simplifier.context, ratio)?;
                if c.is_zero() {
                    return None;
                }
                Some((a, scale * c))
            } else if let Some(a) = num_abs {
                // `|a|/den_core = (a/den_core)·sign(a)`; `a/den_core` must fold to a nonzero rational.
                if !contains_var(&simplifier.context, a, var) {
                    return None;
                }
                let ratio = simplifier.context.add(Expr::Div(a, den_core));
                let (ratio, _) = simplifier.simplify(ratio);
                let c = as_rational_const(&simplifier.context, ratio)?;
                if c.is_zero() {
                    return None;
                }
                Some((a, scale * c))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Detect `e = coeff·sign(g) + offset` — the sign form [`sign_form_coeff`] plus an additive rational
/// constant peeled from an enclosing `Add`/`Sub` (`x/|x| + 1`, `2 - x/|x|`). Returns `(g, coeff, offset)`.
fn sign_form_coeff_offset(
    simplifier: &mut Simplifier,
    e: ExprId,
    var: &str,
) -> Option<(ExprId, num_rational::BigRational, num_rational::BigRational)> {
    use cas_math::numeric_eval::as_rational_const;
    use num_rational::BigRational;
    use num_traits::Zero;

    // The bare sign form carries no offset.
    if let Some((g, c)) = sign_form_coeff(simplifier, e, var) {
        return Some((g, c, BigRational::zero()));
    }
    match simplifier.context.get(e).clone() {
        Expr::Add(l, r) => {
            if let Some(d) = as_rational_const(&simplifier.context, l) {
                let (g, c, o) = sign_form_coeff_offset(simplifier, r, var)?;
                Some((g, c, o + d))
            } else if let Some(d) = as_rational_const(&simplifier.context, r) {
                let (g, c, o) = sign_form_coeff_offset(simplifier, l, var)?;
                Some((g, c, o + d))
            } else {
                None
            }
        }
        Expr::Sub(l, r) => {
            if let Some(d) = as_rational_const(&simplifier.context, r) {
                // `l − d`: shift the offset down.
                let (g, c, o) = sign_form_coeff_offset(simplifier, l, var)?;
                Some((g, c, o - d))
            } else if let Some(d) = as_rational_const(&simplifier.context, l) {
                // `d − (coeff·sign(g) + o) = −coeff·sign(g) + (d − o)`.
                let (g, c, o) = sign_form_coeff_offset(simplifier, r, var)?;
                Some((g, -c, d - o))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Decompose `expr` into a sum of sign forms plus a rational offset: `expr = Σ cᵢ·sign(gᵢ) + offset`.
/// Walks `Add`/`Sub` (tracking the running sign) and reads each leaf as either a sign form
/// ([`sign_form_coeff`]) or a rational constant. Returns false if any leaf is neither.
fn collect_sign_sum_terms(
    simplifier: &mut Simplifier,
    expr: ExprId,
    var: &str,
    sign: &num_rational::BigRational,
    terms: &mut Vec<(ExprId, num_rational::BigRational)>,
    offset: &mut num_rational::BigRational,
) -> bool {
    use cas_math::numeric_eval::as_rational_const;
    match simplifier.context.get(expr).clone() {
        Expr::Add(l, r) => {
            collect_sign_sum_terms(simplifier, l, var, sign, terms, offset)
                && collect_sign_sum_terms(simplifier, r, var, sign, terms, offset)
        }
        Expr::Sub(l, r) => {
            let neg = -sign.clone();
            collect_sign_sum_terms(simplifier, l, var, sign, terms, offset)
                && collect_sign_sum_terms(simplifier, r, var, &neg, terms, offset)
        }
        _ => {
            if let Some((g, c)) = sign_form_coeff(simplifier, expr, var) {
                terms.push((g, sign.clone() * c));
                true
            } else if let Some(d) = as_rational_const(&simplifier.context, expr) {
                *offset += sign.clone() * d;
                true
            } else {
                false // a variable term that is not a sign form
            }
        }
    }
}

/// Solve a SUM of at least two sign forms `Σ cᵢ·sign(gᵢ) {op} k` (each `gᵢ` affine in the variable) — a
/// step function with jumps at the `gᵢ = 0` poles. `(x+1)/|x+1| + (x-1)/|x-1| > 0` was reported "No
/// solution" (truth `(1, ∞)`). Partition ℝ at the sorted breakpoints `−bᵢ/aᵢ`, evaluate the (constant)
/// sum on each open region with a rational test point, and keep the regions satisfying the relation; the
/// breakpoints themselves are excluded (each is a `0/0` pole of its term).
fn try_solve_sign_sum_relation(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::{BoundType, Interval, RelOp};
    use cas_math::polynomial::Polynomial;
    use cas_solver_core::solution_set::{neg_inf, pos_inf};
    use num_rational::BigRational;
    use num_traits::Zero;

    // Decompose the RAW sides (`Σ cᵢ·sign(gᵢ) + offset`) rather than `simplify(lhs − rhs)`: the
    // simplifier combines a same-sign sum over a common denominator (`(x+1)/|x+1| + (x-1)/|x-1| →`
    // a single fraction), which is no longer a readable sum of sign forms.
    let mut terms: Vec<(ExprId, BigRational)> = Vec::new();
    let mut offset = BigRational::zero();
    let one = BigRational::from_integer(1.into());
    let neg_one = -one.clone();
    if !collect_sign_sum_terms(simplifier, eq.lhs, var, &one, &mut terms, &mut offset)
        || !collect_sign_sum_terms(simplifier, eq.rhs, var, &neg_one, &mut terms, &mut offset)
    {
        return None;
    }
    if terms.len() < 2 {
        return None; // a single sign form: the dedicated handler renders it cleanly
    }
    // Each `gᵢ` must be AFFINE (`aᵢ·x + bᵢ`); its breakpoint is the root `−bᵢ/aᵢ`.
    let mut affine: Vec<(BigRational, BigRational, BigRational)> = Vec::new(); // (a, b, coeff)
    for (g, c) in &terms {
        let poly = Polynomial::from_expr(&simplifier.context, *g, var).ok()?;
        if poly.degree() != 1 {
            return None;
        }
        let a = poly.coeffs.get(1).cloned()?;
        let b = poly
            .coeffs
            .first()
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if a.is_zero() {
            return None;
        }
        affine.push((a, b, c.clone()));
    }
    let mut breaks: Vec<BigRational> = affine.iter().map(|(a, b, _)| -b / a).collect();
    breaks.sort();
    breaks.dedup();

    // `Σ cᵢ·sign(aᵢ·t + bᵢ) + offset {op} 0` at a rational `t` (never a breakpoint, so every `aᵢ·t + bᵢ`
    // is nonzero).
    let satisfies = |t: &BigRational| -> bool {
        let mut s = offset.clone();
        for (a, b, c) in &affine {
            let v = a * t + b;
            if v > BigRational::zero() {
                s += c.clone();
            } else {
                s -= c.clone();
            }
        }
        match eq.op {
            RelOp::Lt => s < BigRational::zero(),
            RelOp::Leq => s <= BigRational::zero(),
            RelOp::Gt => s > BigRational::zero(),
            RelOp::Geq => s >= BigRational::zero(),
            RelOp::Eq => s.is_zero(),
            RelOp::Neq => !s.is_zero(),
        }
    };

    // Regions: `(−∞, r₁)`, `(rⱼ, rⱼ₊₁)`, `(r_k, ∞)`; a satisfying region becomes an OPEN interval.
    let n = breaks.len();
    let one_r = BigRational::from_integer(1.into());
    let two = BigRational::from_integer(2.into());
    let mut intervals: Vec<Interval> = Vec::new();
    for idx in 0..=n {
        let t = if idx == 0 {
            &breaks[0] - &one_r
        } else if idx == n {
            &breaks[n - 1] + &one_r
        } else {
            (&breaks[idx - 1] + &breaks[idx]) / &two
        };
        if !satisfies(&t) {
            continue;
        }
        let min = if idx == 0 {
            neg_inf(&mut simplifier.context)
        } else {
            simplifier
                .context
                .add(Expr::Number(breaks[idx - 1].clone()))
        };
        let max = if idx == n {
            pos_inf(&mut simplifier.context)
        } else {
            simplifier.context.add(Expr::Number(breaks[idx].clone()))
        };
        intervals.push(Interval {
            min,
            min_type: BoundType::Open,
            max,
            max_type: BoundType::Open,
        });
    }
    Some(match intervals.len() {
        0 => SolutionSet::Empty,
        1 => SolutionSet::Continuous(intervals.pop().unwrap()),
        _ => SolutionSet::Union(intervals),
    })
}

/// Solve `sign(g(x)) {op} c` written as `g/|g| {op} c` (or `|g|/g {op} c`), `c` constant. Because
/// `sign(g) ∈ {−1, +1}` (undefined at `g = 0`), the relation reduces to which of those two values
/// satisfy `s {op} c`: only `+1` ⇒ `g > 0`; only `−1` ⇒ `g < 0`; both ⇒ `g ≠ 0`; neither ⇒ ∅. Solving
/// the strict sign condition on `g` yields OPEN intervals that EXCLUDE the `g = 0` pole — the generic
/// path returned a closed ray including the `0/0` point, or "No solution" for the inequality forms.
fn try_solve_sign_via_abs(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;
    use cas_solver_core::solution_set::union_solution_sets;
    use num_rational::BigRational;
    use num_traits::Zero;

    // `(g, coeff, offset)` such that the var side equals `coeff·sign(g) + offset`; `k` is the constant
    // other side; `op` is oriented so the relation reads `coeff·sign(g) + offset {op} k`.
    let (g, coeff, offset, k, op) =
        if let Some((g, coeff, offset)) = sign_form_coeff_offset(simplifier, eq.lhs, var) {
            if contains_var(&simplifier.context, eq.rhs, var) {
                return None;
            }
            let k = as_rational_const(&simplifier.context, eq.rhs)?;
            (g, coeff, offset, k, eq.op.clone())
        } else if let Some((g, coeff, offset)) = sign_form_coeff_offset(simplifier, eq.rhs, var) {
            if contains_var(&simplifier.context, eq.lhs, var) {
                return None;
            }
            // `k {op} coeff·sign(g)+offset` ⟺ `coeff·sign(g)+offset {flip op} k` (Eq/Neq symmetric).
            let op = if matches!(eq.op, RelOp::Eq | RelOp::Neq) {
                eq.op.clone()
            } else {
                flip_inequality(eq.op.clone())
            };
            let k = as_rational_const(&simplifier.context, eq.lhs)?;
            (g, coeff, offset, k, op)
        } else {
            return None;
        };

    // Reduce `coeff·sign(g) + offset {op} k` to `sign(g) {op} (k−offset)/coeff`, flipping a strict op
    // when `coeff < 0` (dividing an inequality by a negative). `Eq`/`Neq` are sign-independent.
    let c = (k - offset) / &coeff;
    let op = if coeff < BigRational::zero() && !matches!(op, RelOp::Eq | RelOp::Neq) {
        flip_inequality(op)
    } else {
        op
    };

    let satisfies = |s: i64| -> bool {
        let sv = BigRational::from_integer(s.into());
        match op {
            RelOp::Eq => sv == c,
            RelOp::Lt => sv < c,
            RelOp::Leq => sv <= c,
            RelOp::Gt => sv > c,
            RelOp::Geq => sv >= c,
            RelOp::Neq => sv != c,
        }
    };
    let neg = satisfies(-1);
    let pos = satisfies(1);
    let zero = simplifier.context.num(0);
    match (neg, pos) {
        (false, false) => Some(SolutionSet::Empty),
        (false, true) => solve_relation_set(simplifier, var, g, zero, RelOp::Gt), // g > 0
        (true, false) => solve_relation_set(simplifier, var, g, zero, RelOp::Lt), // g < 0
        (true, true) => {
            // g ≠ 0: everything except the pole.
            let lo = solve_relation_set(simplifier, var, g, zero, RelOp::Lt)?;
            let hi = solve_relation_set(simplifier, var, g, zero, RelOp::Gt)?;
            Some(union_solution_sets(&simplifier.context, lo, hi))
        }
    }
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
    fn symbolic_poly_degree_in_var_measures_and_rejects_nonpoly() {
        use super::symbolic_poly_degree_in_var;
        use crate::Simplifier;
        use cas_parser::parse;
        // Measure on the SIMPLIFIED tree, exactly as the leak-recovery hook does (so
        // subtraction is already the canonical `Add(Neg(...))` the degree walker expects).
        let mut s = Simplifier::with_default_rules();
        let deg = |s: &mut Simplifier, src: &str| {
            let e = parse(src, &mut s.context).expect("parse");
            let (e, _) = s.simplify(e);
            symbolic_poly_degree_in_var(&s.context, e, "x")
        };
        // Symbolic-coefficient polynomials: degree is the max power of x.
        assert_eq!(deg(&mut s, "x^3 + p*x + q"), Some(3));
        assert_eq!(deg(&mut s, "x^3 + a*x^2 + b*x + c"), Some(3));
        assert_eq!(deg(&mut s, "x^5 + p*x + q"), Some(5));
        assert_eq!(deg(&mut s, "x^3 - p*x^2"), Some(3));
        assert_eq!(deg(&mut s, "x^2 + p*x + q"), Some(2));
        assert_eq!(deg(&mut s, "p*x + q"), Some(1));
        // Non-polynomial occurrences of x -> None, so the guard never fires on these.
        assert_eq!(deg(&mut s, "x^2 - sqrt(x)"), None); // fractional power
        assert_eq!(deg(&mut s, "x^2 - 2^x"), None); // x in the exponent
        assert_eq!(deg(&mut s, "x + 1/x"), None); // x in a denominator
        assert_eq!(deg(&mut s, "x + sin(x)"), None); // x inside a function
    }

    #[test]
    fn trig_unit_rhs_classifies_named_transcendental_constants_via_bounds() {
        use super::{classify_trig_unit_rhs, TrigUnitClass};
        use cas_ast::Constant;
        let mut ctx = Context::new();
        let phi = ctx.add(Expr::Constant(Constant::Phi));
        let e = ctx.add(Expr::Constant(Constant::E));
        let pi = ctx.add(Expr::Constant(Constant::Pi));
        let one = ctx.num(1);
        let two = ctx.num(2);
        let four = ctx.num(4);
        let neg_phi = ctx.add(Expr::Neg(phi));
        let inv_e = ctx.add(Expr::Div(one, e));
        let pi_4 = ctx.add(Expr::Div(pi, four));
        let e_minus_2 = ctx.add(Expr::Sub(e, two));
        // Strictly outside [-1, 1]: phi (the (1+sqrt(5))/2 fold), e, -phi.
        for c in [phi, e, neg_phi] {
            assert!(
                matches!(
                    classify_trig_unit_rhs(&ctx, c),
                    Some(TrigUnitClass::OutOfRange)
                ),
                "expected OutOfRange"
            );
        }
        // Strictly inside (-1, 1) with proven sign: 1/e, pi/4, e - 2.
        for c in [inv_e, pi_4, e_minus_2] {
            assert!(
                matches!(classify_trig_unit_rhs(&ctx, c), Some(TrigUnitClass::InOpen)),
                "expected InOpen"
            );
        }
        // Symbolic RHS stays undecided (owned by the honest decline path).
        let y = ctx.var("y");
        assert!(classify_trig_unit_rhs(&ctx, y).is_none());
    }

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
            cas_math::root_forms::cmp_rational_to_quadratic_surd(&r(8, 5), &a, &b, &n),
            Ordering::Less // 1.6 < φ
        );
        assert_eq!(
            cas_math::root_forms::cmp_rational_to_quadratic_surd(&r(13, 8), &a, &b, &n),
            Ordering::Greater // 1.625 > φ
        );
        // 2√5 ≈ 4.472 (a = 0, b = 2, n = 5): exact ordering of nearby rationals.
        let (a0, b2) = (
            BigRational::from_integer(0.into()),
            BigRational::from_integer(2.into()),
        );
        assert_eq!(
            cas_math::root_forms::cmp_rational_to_quadratic_surd(&r(9, 2), &a0, &b2, &n),
            Ordering::Greater // 4.5 > 2√5
        );
        assert_eq!(
            cas_math::root_forms::cmp_rational_to_quadratic_surd(&r(4, 1), &a0, &b2, &n),
            Ordering::Less // 4 < 2√5
        );
        // Negative coefficient `1 − √2` ≈ −0.414 (a = 1, b = −1, n = 2): sign handled.
        let (a1, bn1, n2) = (
            BigRational::from_integer(1.into()),
            BigRational::from_integer((-1).into()),
            BigRational::from_integer(2.into()),
        );
        assert_eq!(
            cas_math::root_forms::cmp_rational_to_quadratic_surd(&r(-1, 2), &a1, &bn1, &n2),
            Ordering::Less // −0.5 < 1 − √2
        );
        assert_eq!(
            cas_math::root_forms::cmp_rational_to_quadratic_surd(&r(-2, 5), &a1, &bn1, &n2),
            Ordering::Greater // −0.4 > 1 − √2
        );
        // Degenerate (b = 0): a plain rational comparison.
        let z = BigRational::from_integer(0.into());
        assert_eq!(
            cas_math::root_forms::cmp_rational_to_quadratic_surd(&r(3, 1), &r(3, 1), &z, &z),
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
