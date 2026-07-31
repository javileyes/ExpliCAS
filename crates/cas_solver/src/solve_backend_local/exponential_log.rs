//! `solve_backend_local`: familia `exponential_log`.
//!
//! Ver la cabecera de `solve_backend_local.rs` para el contexto.

use super::*;

/// Solve an EQUATION that is a polynomial of degree ≥ 2 in `ln(g)` for a single
/// log atom `ln(g)` whose argument contains the variable (e.g.
/// `ln(x)^2 - ln(x) - 2 = 0`, a quadratic in `ln(x)`). Substitute `u = ln(g)`,
/// solve the polynomial in `u`, then back-substitute `ln(g) = u_root` — the
/// recursive solver finishes each as `g = e^(u_root)` with the `ln` domain
/// (`g > 0`). Without this, the isolation path reorients to `x = e^(√(…))` and
/// leaks a malformed `solve(...)` residual while dropping every root.
pub(super) fn try_solve_polynomial_in_log(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
    steps_out: &mut Vec<crate::SolveStep>,
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
    solve_polynomial_in_atom(simplifier, u_expr, u_var, var, atom, steps_out)
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
pub(super) fn try_solve_exponential_reciprocal_polynomial(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
    steps_out: &mut Vec<crate::SolveStep>,
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
    solve_polynomial_in_atom(simplifier, u_expr, u_var, var, atom, steps_out)
}

/// Return a `ln(arg)` subexpression of `expr` whose argument contains `var`
/// (the substitution atom for [`try_solve_polynomial_in_log`]), or None.
pub(super) fn find_log_atom_containing_var(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
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

/// True when `lhs` is `log(base, arg)` with the solve VARIABLE in the BASE — i.e. `logₓ(c)`, which is
/// non-monotonic in `x`. A constant-base `log(c, x)` (monotonic, solvable) returns false.
pub(super) fn is_variable_base_log(ctx: &Context, lhs: ExprId, var: &str) -> bool {
    use cas_ast::BuiltinFn;
    let Expr::Function(fn_id, args) = ctx.get(lhs) else {
        return false;
    };
    args.len() == 2
        && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Log))
        && cas_ast::collect_variables(ctx, args[0]).contains(var)
}

/// Solve a two-term exponential equation with DIFFERENT effective bases: `A·M^x + B·N^x = 0` (`M ≠ N`,
/// both positive rationals, no constant term) ⟺ `(M/N)^x = −B/A`, i.e. `x = ln(−B/A) / ln(M/N)`. Covers
/// `4^x − 9^x = 0` (→ `x = 0`), `5·2^x = 3^x`, `2·4^x = 3·9^x` — which otherwise error with "Cannot
/// isolate 'x'" once moved to one side / coefficiented (the A=B forms happen to isolate, but the
/// one-sided forms do not). `(M/N)^x > 0`, so `−B/A ≤ 0` ⇒ NO solution. Same-base forms (`M = N`) are
/// the single-base polynomial path's job and decline here.
pub(super) fn try_solve_two_different_base_exponential_equation(
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
pub(super) fn try_solve_via_exp_base_normalization(
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
pub(super) fn collect_linear_exponential_atom_terms(
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

/// True if `expr` contains an exponential `base^(exponent)` (constant base, `var`
/// in the exponent) whose exponent has a NEGATIVE `var`-rate (a `base^(-x)` term)
/// or a non-affine exponent — i.e. not a clean non-negative-power polynomial in
/// `base^x`.
pub(super) fn exponential_has_negative_rate(ctx: &Context, expr: ExprId, var: &str) -> bool {
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
pub(super) fn find_first_exponential(ctx: &Context, expr: ExprId, var: &str) -> Option<ExprId> {
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
pub(super) fn match_ln_var_squared_with_coeff(
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
