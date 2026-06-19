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
fn inv_trig_arg_provably_out_of_range(ctx: &Context, c: ExprId) -> bool {
    use cas_math::root_forms::as_linear_surd;
    use num_rational::BigRational;
    use num_traits::{Signed, Zero};
    use std::cmp::Ordering;

    let Some((a, b, n)) = as_linear_surd(ctx, c) else {
        return false;
    };
    // Exact sign of `p + B·√n` versus zero (`n ≥ 0` from `as_linear_surd`).
    let surd_sign = |p: BigRational| -> Ordering {
        let zero = BigRational::zero();
        if b.is_zero() || n.is_zero() {
            return p.cmp(&zero);
        }
        if p.is_zero() {
            return b.cmp(&zero); // B·√n, with √n > 0
        }
        let (sp, sb) = (p.cmp(&zero), b.cmp(&zero));
        if sp == sb {
            return sp; // same sign -> that sign
        }
        // Opposite signs: sign(p + B·√n) = sign(B) · sign(B²·n − p²).
        let inner = (b.clone() * b.clone() * n.clone()).cmp(&(p.clone() * p.clone()));
        if b.is_negative() {
            inner.reverse()
        } else {
            inner
        }
    };
    let one = BigRational::from_integer(1.into());
    surd_sign(a.clone() - one.clone()) == Ordering::Greater || surd_sign(a + one) == Ordering::Less
}

/// True when `expr` is not a real value: it contains a non-finite / undefined
/// constant (∞ or undefined) anywhere, or an out-of-range inverse-trig term
/// (`arcsin(c)` / `arccos(c)` with `|c| > 1`, whose real domain is `[-1, 1]`).
/// Such a value is never a real solution of an equation over ℝ — e.g.
/// `solve(cos(x)=2, x)` must not report `{ arccos(2) }`, and `solve(sin(x)=√2, x)`
/// must not report `{ arcsin(√2) }`.
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
    use cas_math::root_forms::provable_sign_vs_zero;
    use std::cmp::Ordering;

    if conds.is_empty() {
        return false;
    }
    let var_id = ctx.var(var);
    for cond in conds {
        let violates = match cond {
            // ln(e)/log(e) require e > 0; e ≤ 0 at the root is a violation
            // (e = 0 makes the log undefined, so it is extraneous too).
            ImplicitCondition::Positive(e) => {
                let at = substitute_expr_by_id(ctx, *e, var_id, root);
                matches!(
                    provable_sign_vs_zero(ctx, at),
                    Some(Ordering::Less | Ordering::Equal)
                )
            }
            // sqrt(e) requires e ≥ 0; only e < 0 violates (boundary e = 0 is fine).
            ImplicitCondition::NonNegative(e) => {
                let at = substitute_expr_by_id(ctx, *e, var_id, root);
                matches!(provable_sign_vs_zero(ctx, at), Some(Ordering::Less))
            }
            // 1/e requires e ≠ 0; only a PROVABLE exact zero violates.
            ImplicitCondition::NonZero(e) => {
                let at = substitute_expr_by_id(ctx, *e, var_id, root);
                matches!(provable_sign_vs_zero(ctx, at), Some(Ordering::Equal))
            }
            // acosh(e) etc. require e ≥ lower; only e − lower < 0 violates.
            ImplicitCondition::LowerBound(e, lower) => {
                let at = substitute_expr_by_id(ctx, *e, var_id, root);
                let lb = ctx.add(Expr::Number(lower.clone()));
                let shifted = ctx.add(Expr::Sub(at, lb));
                matches!(provable_sign_vs_zero(ctx, shifted), Some(Ordering::Less))
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
/// two-argument `log(b, arg)`.
fn detect_monotonic_lhs(ctx: &Context, lhs: ExprId) -> Option<(MonotonicFn, ExprId)> {
    use cas_math::expr_extract::{
        extract_log_base_argument_view, extract_sqrt_argument_view, extract_unary_log_argument_view,
    };
    if let Some(arg) = extract_sqrt_argument_view(ctx, lhs) {
        return Some((MonotonicFn::EvenRoot, arg));
    }
    if let Expr::Pow(base, exp) = ctx.get(lhs) {
        let (base, exp) = (*base, *exp);
        if let Some(n) = cas_math::numeric_eval::as_rational_const(ctx, exp) {
            use num_traits::Signed;
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
    None
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
    // Only the BARE solve variable as argument; a compound argument stays residual.
    let arg_is_var = matches!(simplifier.context.get(arg), Expr::Variable(s)
        if simplifier.context.sym_name(*s) == var);
    if !arg_is_var {
        return set;
    }

    // Argument-domain over ℝ: even root → [0, ∞) (closed at 0); ln/log → (0, ∞).
    let domain_min_type = match kind {
        MonotonicFn::EvenRoot => BoundType::Closed,
        MonotonicFn::Log => BoundType::Open,
    };
    let domain = {
        let ctx = &mut simplifier.context;
        let zero = ctx.num(0);
        let inf = pos_inf(ctx);
        SolutionSet::Continuous(Interval {
            min: zero,
            min_type: domain_min_type,
            max: inf,
            max_type: BoundType::Open,
        })
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
        if equation_is_nonzero_const_over_polynomial(simplifier, eq)
            || equation_has_identically_zero_denominator(simplifier, eq)
        {
            return Ok((SolutionSet::Empty, Vec::new()));
        }
        let (set, steps) = crate::solve_core_runtime::solve_inner(eq, var, simplifier, opts, ctx)?;
        let conds = ctx.required_conditions();
        let set = filter_real_solutions(&mut simplifier.context, eq, var, set, &conds);
        // Fold the monotonic-function argument-domain into an inequality result
        // (`sqrt(x)<2 → [0,4)`), which the inversion drops; no-op for equations.
        let set = intersect_inequality_with_function_domain(simplifier, eq, var, set);
        Ok((set, steps))
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
