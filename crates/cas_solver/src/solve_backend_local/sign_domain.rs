//! `solve_backend_local`: familia `sign_domain`.
//!
//! Ver la cabecera de `solve_backend_local.rs` para el contexto.

use super::*;

/// Solve a SIGN condition on `g` (`g > 0`, `g ≥ 0`, or `g < 0`, per `op` ∈ {Gt, Geq, Lt}).
/// When `g` is a rational CONSTANT the recursive solver errors (`solve(-4 < 0, x)` →
/// "variable not found"), so resolve it directly from the constant's sign: `AllReals` when the
/// relation holds, `Empty` otherwise. Non-constant `g` delegates to the recursive solver.
pub(super) fn solve_g_sign_condition(
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
pub(super) fn solve_poly_sign(
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

/// Union a set of branch/root solution sets. PERIODIC families are collected and combined over a common
/// period (which `union_solution_sets` cannot do for DIFFERENT periods — it would drop one); `Empty`
/// branches are skipped; non-periodic branches (points/intervals) union normally. Shared by the
/// polynomial-in-atom equation solver and the `|A| = c` absolute-value solver.
pub(super) fn union_branch_solutions(
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

/// Collect the arguments of BOUNDED-DOMAIN inverse functions of `var` anywhere
/// in `e`: `asin`/`acos` (closed domain [−1, 1] → strict = false) and
/// `atanh` (open (−1, 1) → strict = true). `acosh`'s lower bound is already
/// recorded by the implicit-domain inference (its `LowerBound` variant carries
/// a detached rational and needs no new node).
pub(super) fn collect_bounded_domain_inverse_args(
    ctx: &Context,
    e: ExprId,
    var: &str,
    out: &mut Vec<(ExprId, bool)>,
) {
    use cas_solver_core::isolation_utils::contains_var;
    match ctx.get(e) {
        Expr::Function(fn_id, args) => {
            let (fn_id, args) = (*fn_id, args.clone());
            if args.len() == 1 && contains_var(ctx, args[0], var) {
                let strict = match ctx.builtin_of(fn_id) {
                    Some(cas_ast::BuiltinFn::Asin | cas_ast::BuiltinFn::Acos) => Some(false),
                    Some(cas_ast::BuiltinFn::Atanh) => Some(true),
                    _ => None,
                };
                if let Some(strict) = strict {
                    if !out.iter().any(|&(g, s)| g == args[0] && s == strict) {
                        out.push((args[0], strict));
                    }
                }
            }
            for a in args {
                collect_bounded_domain_inverse_args(ctx, a, var, out);
            }
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            let (l, r) = (*l, *r);
            collect_bounded_domain_inverse_args(ctx, l, var, out);
            collect_bounded_domain_inverse_args(ctx, r, var, out);
        }
        Expr::Neg(u) => collect_bounded_domain_inverse_args(ctx, *u, var, out),
        _ => {}
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
pub(super) fn threshold_provably_nonpositive(ctx: &Context, threshold: ExprId) -> bool {
    use num_traits::Signed;
    if cas_math::numeric_eval::as_rational_const(ctx, threshold).is_some_and(|v| !v.is_positive()) {
        return true;
    }
    match ctx.get(threshold).clone() {
        Expr::Neg(inner) => is_provably_positive(ctx, inner),
        _ => false,
    }
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
pub(super) fn try_solve_division_vs_const_sign_split(
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

/// Exact sign of a constant `c` versus 0 (`Less`/`Equal`/`Greater`) when `c` is a rational or quadratic
/// surd; `None` if not so reducible. Used to branch `trig^n = c` / `|trig| = c` on the sign of `c`
/// while ALSO accepting a SURD `c` (e.g. `|cos(x)| = √2/2`), which `as_rational_const` rejects.
pub(super) fn const_sign_vs_zero(ctx: &Context, c: ExprId) -> Option<std::cmp::Ordering> {
    let (a, b, n) = cas_math::root_forms::as_linear_surd(ctx, c)?;
    Some(linear_surd_sign(&a, &b, &n))
}

/// Solve `coeff·sign(g) + offset = h(x)` with a VARIABLE RHS `h` (EQUATION only) by
/// the step-function split: `sign(g) ∈ {−1, +1}`, so the equation holds where
/// `h = coeff + offset` on `g > 0`, OR where `h = −coeff + offset` on `g < 0`.
/// `x/|x| = x` (`sign(x) = x`) → `{1} ∪ {−1} = {−1, 1}` (the pole `x = 0` is excluded
/// by the STRICT `g`-branch); `x/|x| = −x` → `∅ ∪ ∅ = No solution`. The generic
/// isolation instead clears the denominator to `x = x·|x|` and leaks a malformed
/// residual. Constant-RHS forms stay with `try_solve_sign_via_abs`; a sign form on
/// BOTH sides is left to the sign-sum handler.
pub(super) fn try_solve_sign_form_equals_expr(
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
pub(super) fn sign_form_coeff_offset(
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
pub(super) fn try_solve_sign_sum_relation(
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
