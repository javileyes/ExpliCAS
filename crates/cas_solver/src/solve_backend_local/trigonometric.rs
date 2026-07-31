//! `solve_backend_local`: familia `trigonometric`.
//!
//! Ver la cabecera de `solve_backend_local.rs` para el contexto.

use super::*;

/// True when the argument `c` of `arcsin`/`arccos` is PROVABLY outside `[-1, 1]`,
/// so the inverse-trig value is non-real and any candidate root containing it must
/// be dropped. EXACT: decides `|c| > 1` for any single quadratic surd `A + B·√n`
/// — covering rationals (`2`, `5/4`) AND surds (`√2`, `√3`, `√2/2`) — via the same
/// exact surd-sign logic as [`cas_math::root_forms::provable_sign_vs_zero`]
/// (`|c| > 1 ⟺ c − 1 > 0 ∨ c + 1 < 0`). A transcendental argument (`π`, `e`) or
/// anything `as_linear_surd` cannot reduce yields `false`, so a valid root is NEVER
/// dropped on an unproven bound (the boundary `|c| = 1`, `arcsin(±1) = ±π/2`, is
/// kept). Never uses f64 — a float gate could drop a root at `c = √2`.
pub(super) fn inv_trig_arg_provably_out_of_range(ctx: &Context, c: ExprId) -> bool {
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

pub(super) fn classify_trig_threshold(ctx: &Context, c: ExprId) -> Option<TrigThresholdRegion> {
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
pub(super) fn bare_sin_or_cos_of_var(ctx: &Context, lhs: ExprId, var: &str) -> bool {
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

/// Core of [`try_solve_polynomial_in_trig`]: treat `diff` as a polynomial of degree ≥ 2 in a single
/// trig atom `sin(g)`/`cos(g)`/`tan(g)`, substitute `u = trig(g)`, solve `P(u) = 0`, and back-substitute
/// each root through the periodic solver (range guard drops `|u| > 1`). Returns `None` if `diff` is not
/// a polynomial in ONE such atom (a second, distinct atom or `x` remains after substitution).
fn solve_polynomial_in_trig_from_diff(
    simplifier: &mut Simplifier,
    diff: ExprId,
    var: &str,
    steps_out: &mut Vec<crate::SolveStep>,
) -> Option<SolutionSet> {
    let atom = find_trig_atom_containing_var(&simplifier.context, diff, var)?;
    let u_var = "__trig_u";
    let u = simplifier.context.var(u_var);
    let u_expr = substitute_expr_by_id(&mut simplifier.context, diff, atom, u);
    if expr_contains_named_var(&simplifier.context, u_expr, var) {
        return None; // a second, distinct trig atom (or x elsewhere) remains
    }
    solve_polynomial_in_atom(simplifier, u_expr, u_var, var, atom, steps_out)
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
    steps_out: &mut Vec<crate::SolveStep>,
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
        if let Some(set) = solve_polynomial_in_trig_from_diff(simplifier, reduced, var, steps_out) {
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
        return solve_polynomial_in_trig_from_diff(simplifier, reduced, var, steps_out);
    }
    None
}

/// Solve an EQUATION that is a polynomial of degree ≥ 2 in a single trig atom `sin(g)` / `cos(g)` /
/// `tan(g)` whose argument contains the variable (`2·sin(x)² − 3·sin(x) + 1 = 0`, a quadratic in
/// `sin(x)`). Substitute `u = trig(g)`, solve the polynomial in `u`, then back-substitute
/// `trig(g) = u_root` — each root finishes as the recursive solver's PERIODIC family (with the range
/// guard, so `sin(x) = 2` drops). Without this, the isolation path rewrites `sin²(x)` via the
/// double-angle identity (`cos(2x)`) and leaks an `arcsin(… − cos(2x) …)` residual.
pub(super) fn try_solve_polynomial_in_trig(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
    steps_out: &mut Vec<crate::SolveStep>,
) -> Option<SolutionSet> {
    if eq.op != cas_ast::RelOp::Eq {
        return None;
    }
    // Try the RAW difference FIRST: simplifying would fold `sin²(x)` into `cos(2x)`, destroying the
    // polynomial-in-`sin(x)` structure (the reason this handler exists).
    let raw_diff = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
    if let Some(set) = solve_polynomial_in_trig_from_diff(simplifier, raw_diff, var, steps_out) {
        return Some(set);
    }
    // Fallback: the SIMPLIFIED difference. This is the DUAL case — a `cos(2x)` (double-angle) term folds
    // to `2·cos(x)² − 1`, turning `cos(2x) + 3·cos(x) + 2 = 0` (two atoms in the raw form) into the
    // single-atom polynomial `2·cos(x)² + 3·cos(x) + 1`. (When the raw form already succeeded — the
    // `sin²` case — we never reach here, so its structure is untouched.) The two-term `cos(2x) ± cos(x)`
    // instead simplifies to a PRODUCT and is solved by the product-equation path, not here.
    let (simplified_diff, _) = simplifier.simplify(raw_diff);
    if let Some(set) =
        solve_polynomial_in_trig_from_diff(simplifier, simplified_diff, var, steps_out)
    {
        return Some(set);
    }
    // Last fallback: the simplified form mixes `sin(g)` and `cos(g)` (e.g. `cos(2x) − sin(x)` folds to
    // the MIXED `2·cos(x)² − sin(x) − 1`). If one atom is purely even, the Pythagorean identity reduces
    // it to a single-atom polynomial.
    try_solve_mixed_trig_via_pythagorean(simplifier, simplified_diff, var, steps_out)
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
pub(super) fn try_solve_homogeneous_linear_trig(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<(SolutionSet, Vec<crate::SolveStep>)> {
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
    let (sol, inner_steps) =
        crate::solver_entrypoints_solve::solve(&tan_eq, var, simplifier).ok()?;
    // Trust only a fully resolved periodic/discrete/empty answer (guard against a residual echo).
    match sol {
        SolutionSet::Periodic { .. } | SolutionSet::Discrete(_) | SolutionSet::Empty => {
            let mut steps = vec![crate::SolveStep::new(
                "Reduce to the tangent (divide both sides by cos)".to_string(),
                tan_eq,
                crate::ImportanceLevel::Medium,
            )];
            steps.extend(inner_steps);
            Some((sol, steps))
        }
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
pub(super) fn try_solve_inhomogeneous_linear_trig(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<(SolutionSet, Vec<crate::SolveStep>)> {
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
    let (sol, inner_steps) =
        crate::solver_entrypoints_solve::solve(&new_eq, var, simplifier).ok()?;
    match sol {
        SolutionSet::Periodic { .. } | SolutionSet::Discrete(_) | SolutionSet::Empty => {
            let mut steps = vec![crate::SolveStep::new(
                "Auxiliary angle: rewrite as R*sin(g + phi) = c/R".to_string(),
                new_eq,
                crate::ImportanceLevel::Medium,
            )];
            steps.extend(inner_steps);
            Some((sol, steps))
        }
        _ => None,
    }
}

/// Return a `sin(arg)` / `cos(arg)` / `tan(arg)` subexpression whose argument contains `var` (the
/// substitution atom for [`try_solve_polynomial_in_trig`]), searching the whole tree, or None.
pub(super) fn find_trig_atom_containing_var(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
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

/// True when `e` contains a `sin`/`cos`/`tan` — or a reciprocal `sec`/`csc`/`cot` — whose ARGUMENT
/// involves `var` (anywhere in the tree). `sin(2)·x` (constant trig) is false; `sin(2x)`,
/// `x − cos(x)`, `sec(x)²` are true. The reciprocals are periodic too: without them, forms no
/// handler matched (`sec(x)³ > 2`, `cot(x)² < 3`) fell past the decline gate into the monotonic
/// inversion, which asserted grossly wrong finite sets (F4: `sec(x)² > 2` → «No solution»).
pub(super) fn contains_trig_of_var(ctx: &Context, e: ExprId, var: &str) -> bool {
    use cas_ast::BuiltinFn;
    match ctx.get(e) {
        Expr::Function(fn_id, args) => {
            let (fn_id, args) = (*fn_id, args.clone());
            if matches!(
                ctx.builtin_of(fn_id),
                Some(
                    BuiltinFn::Sin
                        | BuiltinFn::Cos
                        | BuiltinFn::Tan
                        | BuiltinFn::Sec
                        | BuiltinFn::Csc
                        | BuiltinFn::Cot
                )
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

pub(super) fn try_solve_trig_sum_to_product_equation(
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

/// Classify a constant `sin`/`cos` RHS `c` EXACTLY (never f64): a quadratic surd `a + b·√n` via
/// `linear_surd_sign`, or an `n`-th root `±q^e` via `as_nonneg_power_magnitude`. `None` for a
/// transcendental / unrecognised constant (declines).
pub(super) fn classify_trig_unit_rhs(ctx: &Context, c: ExprId) -> Option<TrigUnitClass> {
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
pub(super) fn try_solve_inverse_trig_hyperbolic_equation(
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
            return try_solve_periodic_trig_equation_ungated(&reduced, var, simplifier, steps_out);
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
        // Match a term `±c·A/trig(g)^(2m)` in its measured spellings (audit
        // 2026-07-30, ficha S1c-001): the bare `Div(A, Pow(trig, 2))`, the
        // coefficient product `Mul(c, Div(…))` (`2·sec(x)² = 8` simplifies to
        // `Mul(2, Div(1, cos²))`, which the Div-only matcher never saw — the
        // finite arccos fallback then DROPPED the periodic family), the
        // divided form `Div(A, Mul(c, Pow))` (`sec(x)²/2`), and the even
        // power `Pow(trig, 2m)` (`sec(x)⁴`). Returns the trig BASE call, `m`
        // and the effective `A` of `A/trig(g)^(2m)`.
        let recip_square = |ctx: &Context, term: ExprId| -> Option<(ExprId, u32, BigRational)> {
            let (inner, mut a_scale) = match ctx.get(term) {
                Expr::Neg(i) => (*i, -BigRational::one()),
                _ => (term, BigRational::one()),
            };
            let inner = match ctx.get(inner) {
                Expr::Mul(l, r) => {
                    if let Some(c) = cas_math::numeric_eval::as_rational_const(ctx, *l) {
                        a_scale *= c;
                        *r
                    } else if let Some(c) = cas_math::numeric_eval::as_rational_const(ctx, *r) {
                        a_scale *= c;
                        *l
                    } else {
                        inner
                    }
                }
                _ => inner,
            };
            let Expr::Div(num, den) = ctx.get(inner) else {
                return None;
            };
            let (num, den) = (*num, *den);
            let a = cas_math::numeric_eval::as_rational_const(ctx, num)?;
            let (den_pow, den_scale) = match ctx.get(den) {
                Expr::Mul(l, r) => {
                    if let Some(c) = cas_math::numeric_eval::as_rational_const(ctx, *l) {
                        (*r, c)
                    } else if let Some(c) = cas_math::numeric_eval::as_rational_const(ctx, *r) {
                        (*l, c)
                    } else {
                        (den, BigRational::one())
                    }
                }
                _ => (den, BigRational::one()),
            };
            if den_scale.is_zero() {
                return None;
            }
            let Expr::Pow(base, exp) = ctx.get(den_pow) else {
                return None;
            };
            let (base, exp) = (*base, *exp);
            let exp_value = cas_math::numeric_eval::as_rational_const(ctx, exp)?;
            if !exp_value.is_integer() {
                return None;
            }
            let exp_int = exp_value.to_integer();
            // F2 2026-07-31: the parity gate is lifted — the FULL exponent
            // `n ≥ 2` is returned and the reducer below branches on parity
            // (even `2m` → square reduction; odd → bijective n-th root).
            if exp_int < 2.into() {
                return None;
            }
            let n: u32 = exp_int.try_into().ok()?;
            let Expr::Function(fn_id, args) = ctx.get(base) else {
                return None;
            };
            if args.len() != 1 || !contains_var(ctx, args[0], var) {
                return None;
            }
            match ctx.builtin_of(*fn_id) {
                Some(BuiltinFn::Sin | BuiltinFn::Cos) => Some((base, n, a_scale * a / den_scale)),
                _ => None,
            }
        };
        let diff = simplifier.context.add(Expr::Sub(lhs, rhs));
        let (diff, _) = simplifier.simplify(diff);
        let mut sq: Option<(ExprId, u32, BigRational)> = None;
        let mut k = BigRational::zero();
        let mut shape_ok = true;
        for term in cas_math::expr_nary::add_leaves(&simplifier.context, diff) {
            if let Some(matched) = recip_square(&simplifier.context, term) {
                if sq.is_some() {
                    shape_ok = false;
                    break;
                }
                sq = Some(matched);
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
            if let Some((base, n, a)) = sq {
                if !k.is_zero() {
                    let target_n = -a / k; // trig(g)^n = target_n
                    if n % 2 == 0 {
                        let m = n / 2;
                        // Reduce the even power to the SQUARE the double-angle
                        // reducer owns: for m > 1, `trig^(2m) = t` ⟺
                        // `trig² = t^(1/m)` (the even power is non-negative, so
                        // only the positive real root exists). A negative `t` has
                        // the SAME (empty) solution set as `trig² = −1`, which
                        // the reducer already decides; a non-perfect m-th root
                        // declines honestly (irrational target: out of this
                        // reducer's exact scope).
                        let target = if m == 1 {
                            Some(target_n)
                        } else if target_n < BigRational::zero() {
                            Some(-BigRational::one())
                        } else {
                            exact_rational_mth_root(&target_n, m)
                        };
                        if let Some(target) = target {
                            let two = simplifier.context.num(2);
                            let pow = simplifier.context.add(Expr::Pow(base, two));
                            let target_expr = simplifier.context.add(Expr::Number(target));
                            let reduced = Equation {
                                lhs: pow,
                                rhs: target_expr,
                                op: RelOp::Eq,
                            };
                            return try_solve_periodic_trig_equation_ungated(
                                &reduced, var, simplifier, steps_out,
                            );
                        }
                    } else {
                        // F2 (frontier-audit 2026-07-14): ODD power. The n-th
                        // root is a BIJECTION on ℝ — sign-preserving, no
                        // extraneous branch — so `trig^n = t ⟺ trig = t^(1/n)`
                        // unconditionally. A perfect rational root reduces
                        // exactly (`sec³ = 8` → `cos = 1/2`); otherwise the
                        // surd root delegates as an expression (the power-1
                        // periodic solver emits the symbolic-arccos family:
                        // `cos = cbrt(1/7)` verified live). Without this the
                        // finite-inverse fallback asserted `{π/3}` as the
                        // complete answer, dropping the whole family.
                        let negative = target_n < BigRational::zero();
                        let abs_target = if negative {
                            -target_n.clone()
                        } else {
                            target_n.clone()
                        };
                        let r_expr = match exact_rational_mth_root(&abs_target, n) {
                            Some(root) => {
                                let root = if negative { -root } else { root };
                                simplifier.context.add(Expr::Number(root))
                            }
                            None => {
                                let t_expr = simplifier.context.add(Expr::Number(abs_target));
                                let one = simplifier.context.num(1);
                                let n_expr = simplifier.context.num(i64::from(n));
                                let inv_n = simplifier.context.add(Expr::Div(one, n_expr));
                                let root = simplifier.context.add(Expr::Pow(t_expr, inv_n));
                                let root = if negative {
                                    simplifier.context.add(Expr::Neg(root))
                                } else {
                                    root
                                };
                                simplifier.simplify(root).0
                            }
                        };
                        let reduced = Equation {
                            lhs: base,
                            rhs: r_expr,
                            op: RelOp::Eq,
                        };
                        return try_solve_periodic_trig_equation_ungated(
                            &reduced, var, simplifier, steps_out,
                        );
                    }
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
            return try_solve_periodic_trig_equation_ungated(&reduced, var, simplifier, steps_out);
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
                    // Narration: the defining identity (with the user's equation
                    // as its line), then one line per surviving family — the
                    // pole-filtered bases, in the exact shape the set displays.
                    steps_out.push(crate::SolveStep::new(
                        "Equal tangents: the arguments differ by a multiple of pi".to_string(),
                        eq.clone(),
                        crate::ImportanceLevel::Medium,
                    ));
                    push_periodic_family_steps(simplifier, var, &bases, period, steps_out);
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
        return try_solve_periodic_trig_equation_ungated(&reduced, var, simplifier, steps_out);
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
            return try_solve_periodic_trig_equation_ungated(&reduced, var, simplifier, steps_out);
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
            return try_solve_periodic_trig_equation_ungated(&reduced, var, simplifier, steps_out);
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
                    return try_solve_periodic_trig_equation_ungated(
                        &reduced, var, simplifier, steps_out,
                    );
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
                    return try_solve_periodic_trig_equation_ungated(
                        &reduced, var, simplifier, steps_out,
                    );
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
                    let a_expr = simplifier.context.add(Expr::Number(a_coeff.clone()));
                    let d = simplifier.context.add(Expr::Div(diff, a_expr));
                    simplifier.simplify(d).0
                };
                let reduced = Equation {
                    lhs: call,
                    rhs: reduced_rhs,
                    op: RelOp::Eq,
                };
                // Narrate the normalization (`2·cos(x) − √3 = 0` → `cos(x) = √3/2`)
                // before delegating: the inner ungated call narrates the periodic
                // inversion itself onto the same steps_out. The narration line
                // shows the RAW `diff / a` quotient (`√3/2`): the solver-side
                // `reduced_rhs` went through simplify for the notable-angle
                // classifier, whose normal form (`3/2·3^(-1/2)`) is exactly the
                // des-simplified shape the educational audit flagged.
                let display_rhs = if num_traits::One::is_one(&a_coeff) {
                    diff
                } else {
                    let a_expr = simplifier.context.add(Expr::Number(a_coeff.clone()));
                    simplifier.context.add(Expr::Div(diff, a_expr))
                };
                steps_out.push(crate::SolveStep::new(
                    "Isolate the trigonometric term".to_string(),
                    Equation {
                        lhs: call,
                        rhs: display_rhs,
                        op: RelOp::Eq,
                    },
                    crate::ImportanceLevel::Medium,
                ));
                return try_solve_periodic_trig_equation_ungated(
                    &reduced, var, simplifier, steps_out,
                );
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
    push_periodic_family_steps(simplifier, var, &bases, period, steps_out);
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
pub(super) fn try_union_periodic_trig_product(
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

/// Solve a bare `trig(a·x + b) = c` whose additive shift `b` is a SYMBOLIC constant (a π-multiple like
/// `π/4`, an `arctan`, a surd — anything that is not a plain rational number). For such a shift the
/// simplifier's angle-addition expansion (`sin(x + π/4) → (√2/2)·(sin x + cos x)`) / the isolation
/// returns only the PRINCIPAL root, dropping the periodic family and the second branch. Solving
/// `trig(u) = c` for `u = a·x + b` (bare, so the existing periodic solver gives the full family) and
/// mapping back through `x = (u − b)/a` restores the periodicity. A PLAIN-rational shift (`sin(x + 1)`)
/// and bare/coefficient forms (`sin(2x)`) are handled correctly by the existing periodic path, so this
/// declines on them (keeping their behaviour and the huella untouched).
pub(super) fn try_solve_shifted_argument_trig(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<(SolutionSet, Vec<crate::SolveStep>)> {
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
    let mapped = map_solution_through_affine(simplifier, u_sol, &a, b)?;
    // Narrate only the MAPPED families: the u-space sub-solve ran in a
    // synthetic variable the student never wrote.
    let mut steps = Vec::new();
    if let SolutionSet::Periodic { bases, period } = &mapped {
        let (bases, period) = (bases.clone(), *period);
        push_periodic_family_steps(simplifier, var, &bases, period, &mut steps);
    }
    Some((mapped, steps))
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
pub(super) fn try_solve_reciprocal_trig_equation(
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
