//! `solve_backend_local`: familia `general`.
//!
//! Ver la cabecera de `solve_backend_local.rs` para el contexto.

use super::*;

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

pub(super) fn solution_contains_nonfinite(ctx: &Context, expr: ExprId) -> bool {
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
pub(super) fn required_conditions_are_contradictory(
    ctx: &Context,
    conds: &[ImplicitCondition],
) -> bool {
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

/// Detect a monotonic `f(arg)` on the LHS, returning `(kind, arg)`. Covers the
/// `sqrt` builtin, an even-root `Pow` (`x^(1/2)`, `x^(1/4)`, …), `ln`, and the
/// two-argument `log(b, arg)` — and sees THROUGH a POSITIVE rational
/// multiplicative coefficient or divisor (`2·√x`, `√x/2`), which preserves both
/// the argument-domain and the `[0,∞)` even-root range, so the range correction
/// (keyed on the threshold sign) is unaffected. A NEGATIVE coefficient (flips the
/// range) and an ADDITIVE shift (`√x + 1`, shifts the range) are NOT matched and
/// stay honest residuals.
pub(super) fn detect_monotonic_lhs(ctx: &Context, lhs: ExprId) -> Option<(MonotonicFn, ExprId)> {
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
pub(super) fn simplify_solution_bounds(
    simplifier: &mut Simplifier,
    set: SolutionSet,
) -> SolutionSet {
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

/// Collect the rational exponents of every `x`-power in `expr` (bare `x` is
/// exponent 1), returning `false` if `x` ever appears in a DISALLOWED position:
/// inside a function, as the base of a non-rational/non-positive power, in a
/// denominator, mixed with another variable, or as a compound base. Constants and
/// `x`-free coefficients are fine. The collected exponents are only used to derive
/// the common denominator `q`; the rebuild handles the actual algebra.
pub(super) fn collect_x_power_exponents(
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
pub(super) fn rebuild_x_powers_as_u(
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
/// Republish the u-polynomial's own steps under the DISPLAY name `u`.
///
/// The substitution variable is a collision-safe synthetic (`__trig_u`,
/// `__rps_u`, `__rpl_u`, …) that the reader never typed, and this repo's rule
/// is that a narration line is predicated of the user's equation, never of an
/// internal form carrying a synthetic symbol. Both the rendered description
/// and the step's own equation carry it, so both are rewritten — and so are
/// the sub-steps, which have the same two fields.
pub(super) fn rewrite_substitution_steps_for_display(
    simplifier: &mut Simplifier,
    steps: Vec<crate::SolveStep>,
    u_var: &str,
) -> Vec<crate::SolveStep> {
    let internal = simplifier.context.var(u_var);
    let display = simplifier.context.var("u");
    let rewrite = |ctx: &mut cas_ast::Context, eq: &mut Equation| {
        eq.lhs = substitute_expr_by_id(ctx, eq.lhs, internal, display);
        eq.rhs = substitute_expr_by_id(ctx, eq.rhs, internal, display);
    };
    let mut out = Vec::with_capacity(steps.len());
    for mut step in steps {
        step.description = step.description.replace(u_var, "u");
        rewrite(&mut simplifier.context, &mut step.equation_after);
        for sub in &mut step.substeps {
            sub.description = sub.description.replace(u_var, "u");
            rewrite(&mut simplifier.context, &mut sub.equation_after);
        }
        out.push(step);
    }
    out
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
/// Collect the FINITE interval endpoints and discrete points of `set` (the
/// values the core set algebra will have to ORDER during intersect/union).
pub(super) fn collect_finite_set_endpoints(
    ctx: &Context,
    set: &SolutionSet,
    out: &mut Vec<ExprId>,
) {
    use cas_solver_core::solution_set::{is_infinity, is_neg_infinity};
    let mut push = |e: ExprId| {
        if !is_infinity(ctx, e) && !is_neg_infinity(ctx, e) && !out.contains(&e) {
            out.push(e);
        }
    };
    match set {
        SolutionSet::Continuous(iv) => {
            push(iv.min);
            push(iv.max);
        }
        SolutionSet::Union(ivs) => {
            for iv in ivs {
                push(iv.min);
                push(iv.max);
            }
        }
        SolutionSet::Discrete(pts) => {
            for &p in pts {
                push(p);
            }
        }
        _ => {}
    }
}

/// Rewrite every EVEN power of `atom` in `expr` via `atom² = sq_repl` (`atom^(2k) → sq_repl^k`),
/// returning the rewritten tree — or `None` if `atom` occurs to any ODD power (a bare `atom` or
/// `atom^(2k+1)`), which the even-power Pythagorean substitution cannot eliminate. Used to turn a
/// mixed `sin(g)`/`cos(g)` polynomial into a single-atom one via `cos² = 1 − sin²` (or `sin² = 1 − cos²`).
pub(super) fn rewrite_even_power_of_atom(
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

/// Net exponent of `var` when `e` is a single power term `c·(α)^k` of an AFFINE argument `α = a·x + b`
/// (`x`, `x-1`, `2x+3`), possibly with a constant coefficient, an additive constant (`x^(2/3) + 1`), a
/// quotient form (the simplifier rewrites `1/x^(1/3)` to `x^(2/3)/x`, net `−1/3`), or a `sqrt`
/// (`= ^(1/2)`). Returns `None` for anything that is not a single power of one affine argument (sums of
/// two powers, two distinct radicals, a non-affine base). The coefficient and the additive constant are
/// irrelevant — only the exponent decides monotonicity — so they are not returned.
pub(super) fn pure_power_monomial_exponent(
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

/// A concrete set the abs reduction can intersect/union; a `Residual`/`Conditional`
/// (e.g. a transcendental `g`) is not, so the guard declines on it.
pub(super) fn is_concrete_solution_set(set: &SolutionSet) -> bool {
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
pub(super) fn solve_concrete_side(
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

/// `c = ±q^e` with `q` a NON-NEGATIVE rational and `e` a POSITIVE rational. Returns `(q, neg)`. Since
/// `q^e` is increasing in `q` for `e > 0`, `q^e {<,=,>} 1 ⟺ q {<,=,>} 1` and `q^e = 0 ⟺ q = 0` — so
/// the magnitude class only needs `q` vs `{0, 1}`. Covers the `n`-th roots `(1/4)^(1/4)`, `4^(1/4)`
/// the even-power reduction produces (which `as_linear_surd` — quadratic surds only — does not).
pub(super) fn as_nonneg_power_magnitude(
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

/// Combine two sub-results where each is Empty/AllReals/PeriodicIntervalUnion;
/// PIU pairs go through the circular same-period algebra. Anything else
/// (mixed Periodic points, intervals) declines conservatively.
pub(super) fn combine_piu_sets(
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

/// Airbag: f64-sample `trig(g(x)) op r` at u-space fractions {1/8, 1/2, 7/8}
/// inside the window (must satisfy) and at ±width/8 outside (must not).
/// A sample is a CONTRADICTION only when the sign of `trig − r` disagrees by
/// more than `τ = 1e-9·max(1, |r|)`; unevaluable or |·| ≤ τ ⇒ inconclusive
/// (skipped). Returns false only on a genuine contradiction.
#[allow(clippy::too_many_arguments)]
pub(super) fn interior_window_samples_consistent(
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

/// Push one "Periodic family of solutions" narration line per periodic base,
/// in the exact `x = base + k·T` shape the result set displays. A zero base
/// narrates as `x = k·T`; no general simplify (it FACTORS the sum into
/// unreadable forms). Shared by the periodic solver's map-back tail and the
/// shifted-argument handler (whose u-space solve runs in a synthetic variable
/// the student never wrote — only the mapped families are honest narration).
pub(super) fn push_periodic_family_steps(
    simplifier: &mut Simplifier,
    var: &str,
    bases: &[ExprId],
    period: ExprId,
    steps_out: &mut Vec<crate::SolveStep>,
) {
    let x = simplifier.context.var(var);
    let k_var = simplifier.context.var("k");
    for base in bases {
        let k_t = simplifier.context.add(Expr::Mul(k_var, period));
        let base_is_zero = cas_math::numeric_eval::as_rational_const(&simplifier.context, *base)
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
                op: cas_ast::RelOp::Eq,
            },
            crate::ImportanceLevel::Medium,
        ));
    }
}

/// Flatten a product into its variable-bearing factors, unwrapping `Neg`/`Mul` and dropping constant
/// factors. Each leaf factor that contains `var` is pushed onto `out`.
pub(super) fn collect_product_var_factors(
    ctx: &Context,
    e: ExprId,
    var: &str,
    out: &mut Vec<ExprId>,
) {
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
pub(super) fn union_periodic_families_over_common_period(
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

/// Deduplicate bases that are equal modulo `period` (i.e. `(b - b') / period` is an integer).
pub(super) fn dedup_bases_modulo_period(
    simplifier: &mut Simplifier,
    bases: &mut Vec<ExprId>,
    period: ExprId,
) {
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

/// Map a solution set through the INCREASING power `u = w^q` (valid only
/// when the set lies in a region where the power is monotone increasing —
/// the caller clamps even q to `w > 0`). Bound types are preserved.
pub(super) fn map_set_through_increasing_power(
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

pub(super) fn solve_local_core(
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
    if let Some((set, steps)) = try_solve_shifted_argument_trig(simplifier, eq, var) {
        return Ok((set, steps));
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
    if let Some((set, steps)) = try_solve_trig_weak_boundary_inequality(eq, var, simplifier) {
        return Ok((set, steps));
    }
    if equation_is_nonzero_const_over_polynomial(simplifier, eq)
        || equation_has_identically_zero_denominator(simplifier, eq)
    {
        return Ok((SolutionSet::Empty, Vec::new()));
    }
    // `c/(x+√2) {op} 0` on the RAW tree, before the simplifier rationalizes the surd
    // denominator through its conjugate and fabricates a spurious removable pole.
    if let Some((set, steps)) = try_solve_const_over_surd_affine_inequality(simplifier, eq, var) {
        return Ok((set, steps));
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
    // Hyperbolic RANGE edges: `tanh(g) ⋚ c` with |c| ≥ 1 and `cosh(g) ⋚ c`
    // with c ≤ 1 settle exactly from range(tanh) = (−1, 1) and
    // range(cosh) = [1, ∞) — no inversion needed (F4 hyperbolic member:
    // `tanh(x)² < 1` splits into `tanh < 1 ∧ tanh > −1`, both edges).
    if let Some(set) = try_solve_hyperbolic_range_edge_inequality(eq, var, simplifier) {
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
        if let Some((set, steps)) = try_solve_single_abs_polynomial_relation(simplifier, eq, var) {
            return Ok((set, steps));
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
    {
        let mut atom_steps = Vec::new();
        if let Some(set) = try_solve_rational_power_polynomial(simplifier, eq, var, &mut atom_steps)
        {
            return Ok((set, atom_steps));
        }
    }
    // LAURENT polynomials in `x^(1/q)` — a root mixed with its reciprocal
    // (`√x − 1/√x = 1`, `√x + 1/√x = 5/2`) — leak the same malformed residual
    // (`x = (…)^(1/(1/2))`). Clear the `1/u^k` by shifting and solve in `u = x^(1/q)`.
    {
        let mut atom_steps = Vec::new();
        if let Some(set) = try_solve_rational_power_laurent(simplifier, eq, var, &mut atom_steps) {
            return Ok((set, atom_steps));
        }
    }
    // Equations that are a polynomial of degree ≥ 2 in `ln(x)`
    // (`ln(x)^2 - ln(x) - 2 = 0`, …) leak the same way; solve them by the
    // `u = ln(x)` substitution.
    {
        let mut atom_steps = Vec::new();
        if let Some(set) = try_solve_polynomial_in_log(simplifier, eq, var, &mut atom_steps) {
            return Ok((set, atom_steps));
        }
    }
    // Equations that are a polynomial of degree ≥ 2 in `|x|` (`|x|² − 3·|x| + 2 = 0`,
    // stored as `x² − 3·|x| + 2` after `|x|² → x²`) leak the same way — the isolation
    // path reorients to `x = √(3·|x| − 2)`. Solve them by the `u = |x|` substitution
    // (with the `x² = |x|²` even-power unification) here first.
    {
        let mut atom_steps = Vec::new();
        if let Some(set) = try_solve_polynomial_in_abs(simplifier, eq, var, &mut atom_steps) {
            return Ok((set, atom_steps));
        }
    }
    // A single `|f(x)|` term with a NON-CONSTANT quadratic-or-higher remainder
    // (`x² + |x−1| − 3 = 0`, `|x−1| = 3 − x²`) is `|f| = g(x)`. Isolating the abs
    // is unsound: the generic path solves only `f = g` (dropping `f = −g`) and
    // skips `g ≥ 0`, returning a spurious root while missing a real one — or
    // leaking a malformed residual. Solve both branches and keep `g(r) ≥ 0`.
    {
        let mut abs_steps = Vec::new();
        if let Some(set) =
            try_solve_single_abs_equals_polynomial(simplifier, eq, var, &mut abs_steps)
        {
            return Ok((set, abs_steps));
        }
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
        if let Some((set, steps)) = try_solve_single_abs_polynomial_relation(simplifier, eq, var) {
            return Ok((set, steps));
        }
    }
    // Equations that mix an exponential with its RECIPROCAL (`e^x + e^(−x) = 2`, `2^x − 3 + 2^(1−x) = 0`)
    // are Laurent polynomials in `base^x`; the isolation path rewrites them via the cosh identity and
    // bails (`función [cosh] no definida` / `Cannot isolate`). Substitute `u = base^x`, clear the
    // reciprocal, and solve the polynomial in `u`. Pure positive-power forms decline (owned elsewhere).
    {
        let mut atom_steps = Vec::new();
        if let Some(set) =
            try_solve_exponential_reciprocal_polynomial(simplifier, eq, var, &mut atom_steps)
        {
            return Ok((set, atom_steps));
        }
    }
    // Equations that are a polynomial of degree ≥ 2 in a trig atom (`2·sin(x)² − 3·sin(x) + 1 = 0`)
    // leak an `arcsin(… − cos(2x) …)` residual once the double-angle identity fires; substitute
    // `u = sin(x)` (cos/tan) and back-substitute each root through the periodic solver.
    {
        let mut atom_steps = Vec::new();
        if let Some(set) = try_solve_polynomial_in_trig(simplifier, eq, var, &mut atom_steps) {
            return Ok((set, atom_steps));
        }
    }
    // A HOMOGENEOUS linear trig equation `a·sin(g) + b·cos(g) = 0` (`sin(x) = cos(x)`,
    // `√3·sin(x) − cos(x) = 0`) reduces to `tan(g) = −b/a`; the isolation path otherwise leaks an
    // `arcsin(cos(x)·…)` residual. The inhomogeneous `… = c` (c ≠ 0) declines.
    if let Some((set, steps)) = try_solve_homogeneous_linear_trig(simplifier, eq, var) {
        return Ok((set, steps));
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
    if let Some((set, steps)) = try_solve_inhomogeneous_linear_trig(simplifier, eq, var) {
        return Ok((set, steps));
    }
    // `|A(x)| = c` with a trig-bearing argument: the generic abs isolation solves the two branches to
    // PRINCIPAL roots (`|2·sin(x)−1| = 1 → {π/2, 0}`); solve each branch fully so trig stays periodic.
    if let Some(set) = try_solve_abs_of_trig_equation(simplifier, eq, var) {
        return Ok((set, Vec::new()));
    }
    // `|E| = 0 ⟺ E = 0`: dispatch the argument's full zero-set (the generic abs isolation drops all but
    // the first factor of a product, `|x·(x−2)| = 0 → {0}` instead of `{0, 2}`).
    if let Some((set, steps)) = try_solve_abs_equals_zero(simplifier, eq, var) {
        return Ok((set, steps));
    }
    // `|f(x)| = g(x)` with a degree-≥2 polynomial `f` and a variable RHS (`|x²−1| = x+1`): split into
    // `f = ±g` and verify each root against the original (enforcing `g ≥ 0`). Linear `|f|` and
    // constant-RHS forms keep their existing handlers.
    {
        let mut abs_steps = Vec::new();
        if let Some(set) = try_solve_abs_polynomial_equation(simplifier, eq, var, &mut abs_steps) {
            return Ok((set, abs_steps));
        }
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
    // BOUNDED-DOMAIN INVERSE-FUNCTION CONDITION (F10 m4, frontier-audit
    // 2026-07-14): `asin(g)`/`acos(g)` require `−1 ≤ g ≤ 1` and `artanh(g)`
    // requires `−1 < g < 1`, but the `&Context`-only implicit-domain inference
    // cannot BUILD the condition node `1 − g²` (LowerBound carries a detached
    // rational; there is no upper-bound variant), so identities like
    // `asin(x) + acos(x) = π/2` returned a bare «All real numbers» — an
    // over-claim (the true set is [−1, 1]). This solver-layer site has the
    // mutable context: record `NonNegative(1 − g²)` (resp. `Positive`), whose
    // display already renders as `−1 ≤ x ≤ 1`, into BOTH the published
    // conditions (parity with how `√x = √x` shows «ℝ if x ≥ 0») and the local
    // filter list (an exact-rational root outside the domain now drops).
    if eq.op == cas_ast::RelOp::Eq {
        let mut bounded_args: Vec<(ExprId, bool)> = Vec::new(); // (arg, strict)
        for side in [eq.lhs, eq.rhs] {
            collect_bounded_domain_inverse_args(&simplifier.context, side, var, &mut bounded_args);
        }
        for (g, strict) in bounded_args {
            let two = simplifier.context.num(2);
            let g_sq = simplifier.context.add(Expr::Pow(g, two));
            let one = simplifier.context.num(1);
            let radicand = simplifier.context.add(Expr::Sub(one, g_sq));
            let (radicand, _) = simplifier.simplify(radicand);
            let cond = if strict {
                ImplicitCondition::Positive(radicand)
            } else {
                ImplicitCondition::NonNegative(radicand)
            };
            if !conds.contains(&cond) {
                ctx.note_required_condition(cond.clone());
                conds.push(cond);
            }
        }
    }
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
        // The simplifier's factored normal form can wrap the whole difference in
        // a POSITIVE rational factor (`√x/2 + 1 − y → (1/2)·(√x + 2 − 2y)`) that
        // hides the radical from the term walk; `d = 0` is invariant under it,
        // so peel before splitting. Local to this Eq-only publisher on purpose:
        // peeling inside the shared collector would silently widen the
        // inequality consumer, which would then need an op-flip audit.
        let d = {
            let mut d = d;
            while let Some(inner) = ctx_top_positive_rational_factor(&simplifier.context, d) {
                d = inner;
            }
            d
        };
        if let Some((s, coeff, _f, rest)) = collect_radical_split(&simplifier.context, d, var) {
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
                // `s·c·√f = −rest` ⟹ `√f = −rest/(s·c)`; `c` is a POSITIVE rational
                // magnitude (sign folded into `s`), so dividing by it never flips the
                // `≥ 0` range condition — `2√x − 1 = y` carries `(y+1)/2 ≥ 0` exactly
                // as the unit spelling carries `y + 1 ≥ 0`.
                let g = if s >= 0 {
                    simplifier.context.add(Expr::Neg(r))
                } else {
                    r
                };
                let g = match &coeff {
                    RadicalCoeff::Rational(c) if num_traits::One::is_one(c) => g,
                    RadicalCoeff::Rational(c) => {
                        let inv = simplifier.context.add(Expr::Number(c.recip()));
                        simplifier.context.add(Expr::Mul(inv, g))
                    }
                    // `c_sym·√f = −rest` ⟹ `√f = g/c_sym`: the range condition
                    // `g/c_sym ≥ 0` holds WHATEVER the sign of `c_sym` is (a
                    // genuine root has `g/c_sym = √f ≥ 0`), so the division is
                    // usable here even though the coefficient's sign is
                    // unknown — `y·√x = 2` carries `2/y ≥ 0`, which excludes
                    // the spurious `x = 4/y²` for every `y < 0`.
                    RadicalCoeff::Symbolic(c_sym) => simplifier.context.add(Expr::Div(g, *c_sym)),
                };
                let (g, _) = simplifier.simplify(g);
                // Noise gate for the symbolic branch: a coefficient like `a²`
                // makes `g/c` provably nonnegative (`1/a² ≥ 0`) — a condition
                // that can never exclude anything, so recording it is noise.
                let provably_vacuous = matches!(coeff, RadicalCoeff::Symbolic(_))
                    && cas_math::prove_sign::prove_nonnegative_depth_with(
                        &simplifier.context,
                        g,
                        6,
                        true,
                        |_, _, _| cas_math::tri_proof::TriProof::Unknown,
                    )
                    .is_proven();
                let cond = ImplicitCondition::NonNegative(g);
                if !provably_vacuous && !conds.contains(&cond) {
                    // F10 m3 (frontier-audit 2026-07-14): a PARAMETER-only range
                    // condition (`√x + 3 = y` ⟹ `y − 3 ≥ 0`) can never act as a
                    // root filter — substituting the root leaves it unchanged —
                    // so it must be PUBLISHED. The isolated spelling
                    // `√x = y − 3` already publishes it through the isolation
                    // path; this gives the shifted spelling parity. Solve-var
                    // conditions keep their existing owners (the filter below),
                    // and constant targets are decided by the answer itself,
                    // never displayed.
                    if !cas_solver_core::isolation_utils::contains_var(&simplifier.context, g, var)
                        && cas_math::numeric_eval::as_rational_const(&simplifier.context, g)
                            .is_none()
                        && !cas_ast::collect_variables(&simplifier.context, g).is_empty()
                    {
                        ctx.note_required_condition(cond.clone());
                    }
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

    // BIQUADRATIC INEQUALITY recovery (2026-07-31, cubic-abs cycle): the
    // generic isolation takes the 4th root of both sides UNCONDITIONALLY
    // (`x⁴ − x² + 1 > 0` → `|x| > (x²−1)^(1/4)`, a self-referential branch
    // with a possibly NEGATIVE radicand) and asserted «No solution» for a
    // tautology (the z-quadratic has disc < 0 ⟹ constant sign ⟹ ℝ). Re-derive
    // through the EXACT z = x² sign analysis when the incumbent looks lossy;
    // correct incumbents (`x⁴−x²<1` → (−√φ, √φ)) are never touched.
    let set = if matches!(
        set,
        SolutionSet::Empty | SolutionSet::Residual(_) | SolutionSet::Conditional(_)
    ) {
        try_solve_biquadratic_inequality(simplifier, eq, var).unwrap_or(set)
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
    } else if matches!(&set, SolutionSet::Discrete(_) | SolutionSet::Empty)
        && count_abs_nodes(&simplifier.context, eq.lhs) >= 3
    {
        // F5 (frontier-audit 2026-07-14): NESTED multi-abs `|E| = c` with E a
        // combination of ≥ 2 inner abs terms. The generic isolation recurses
        // per-branch through the NARROW single-abs isolation, which silently
        // drops one branch's roots (`||x|−|x−2|| = 1` returned `{3/2}`,
        // losing `1/2`) or every root at once (`|2|x|−|x−1|| = 2` → wrong
        // Empty, true roots {−3, 1} — the adversarial sweep's find) — a
        // clean-looking but INCOMPLETE result the Residual-gated recovery
        // above never saw. Re-solve both branches through the full solver
        // and take the recovery only when it strictly completes the answer
        // (more roots, or a ray/interval the narrow path could never
        // produce). Single-inner-abs (`||x|−3| = 1`) and plain
        // `|linear| = c` keep their correct isolation path (the ≥ 3 count
        // includes the outer abs).
        let current_len = match &set {
            SolutionSet::Discrete(roots) => roots.len(),
            _ => 0,
        };
        match try_solve_abs_equality(simplifier, eq, var) {
            Some(SolutionSet::Discrete(recovered)) if recovered.len() > current_len => {
                SolutionSet::Discrete(recovered)
            }
            Some(
                full @ (SolutionSet::Continuous(_) | SolutionSet::Union(_) | SolutionSet::AllReals),
            ) => full,
            _ => set,
        }
    } else {
        set
    };

    // F5 members 5-6: `|N| / |D| = c` with a NESTED-abs numerator
    // (`||x|−2| / |x| = 1` → `{−1}`, losing the twin `1`). The plain forms
    // (`|x+1|/|x−1| = 2`, 2 abs nodes) keep their working owner; the nested
    // shape (≥ 3 abs) re-derives through the CLEARED equation
    // `|N| = c·|D|` — whose nested-abs machinery the arm above just fixed —
    // and enforces the ratio's own `D ≠ 0` exactly per root. Same
    // strictly-more-complete replacement contract as the nested-abs arm.
    let set = if matches!(
        &set,
        SolutionSet::Discrete(_)
            | SolutionSet::Empty
            | SolutionSet::Residual(_)
            | SolutionSet::Conditional(_)
    ) && count_abs_nodes(&simplifier.context, eq.lhs) >= 3
    {
        let current_len = match &set {
            SolutionSet::Discrete(roots) => roots.len(),
            SolutionSet::Residual(_) | SolutionSet::Conditional(_) => usize::MAX,
            _ => 0,
        };
        match try_solve_abs_ratio_equality(simplifier, eq, var) {
            Some(SolutionSet::Discrete(recovered))
                if current_len == usize::MAX || recovered.len() > current_len =>
            {
                SolutionSet::Discrete(recovered)
            }
            Some(SolutionSet::Empty) if current_len == usize::MAX => SolutionSet::Empty,
            _ => set,
        }
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

/// True when `p` is a CLOSED endpoint of `set` (so `p ∈ set` by exact endpoint identity, with no
/// value comparison). Used to drop roots already present from the non-strict root re-union.
pub(super) fn point_is_closed_endpoint(set: &SolutionSet, p: ExprId) -> bool {
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
