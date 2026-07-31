//! `solve_backend_local`: familia `absolute_value`.
//!
//! Ver la cabecera de `solve_backend_local.rs` para el contexto.

use super::*;

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
pub(super) fn try_solve_polynomial_in_abs(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
    steps_out: &mut Vec<crate::SolveStep>,
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
    solve_polynomial_in_atom(simplifier, u_expr, u_var, var, abs_x, steps_out)
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
pub(super) fn try_solve_single_abs_equals_polynomial(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
    steps_out: &mut Vec<crate::SolveStep>,
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
    for (case_idx, rhs) in [g, neg_g].into_iter().enumerate() {
        let branch = Equation {
            lhs: f,
            rhs,
            op: RelOp::Eq,
        };
        let branch_display = format!(
            "{} = {}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: f
            },
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: rhs
            }
        );
        steps_out.push(crate::SolveStep::new(
            format!(
                "Split absolute value (Case {}): {}",
                case_idx + 1,
                branch_display
            ),
            branch.clone(),
            crate::ImportanceLevel::Medium,
        ));
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

pub(super) fn try_solve_single_abs_polynomial_relation(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<(SolutionSet, Vec<crate::SolveStep>)> {
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

    // ORDER GUARD (2026-07-31, cubic abs — the F9 playbook): the region ∩
    // branch assembly below runs through the core set algebra, whose endpoint
    // order falls back to a VALUE-BLIND structural compare. Single-real-root
    // cubic endpoints (Cardano `cbrt` sums) are decidable through the
    // const-bounds oracle (odd-root negative-base bounds, this cycle); the
    // casus-irreducibilis TRIG endpoints (`4·cos(arccos(…)/3 − 4π/3)·√⅓`)
    // are still beyond it — committing would silently drop whole regions
    // (`|x³−4x| < 2` lost `(0, 1)…`), so any UNDECIDABLE endpoint pair
    // declines the whole relation to an honest residual instead.
    let mut endpoints: Vec<ExprId> = Vec::new();
    for set in [&pos_branch, &neg_branch, &pos_domain, &neg_domain] {
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
                // Echo the ORIGINAL relation: falling through (None) lets the
                // generic isolation reorient into a MANGLED self-referential
                // residual (`solve(x − cbrt(4x+2) = 0)`).
                return Some((
                    cas_solver_core::solve_outcome::residual_solution_set(
                        &mut simplifier.context,
                        eq.lhs,
                        eq.rhs,
                        eq.op.clone(),
                        var,
                    ),
                    Vec::new(),
                ));
            }
        }
    }

    // Narration: the textbook sign split, one line per case with the
    // SUBSTITUTED relation as its equation. Branch roots are candidates the
    // case-domain intersection can clip, so only the STRUCTURE narrates
    // (candidatos ≠ respuesta — the same rule the verified abs splits follow).
    let steps = vec![
        crate::SolveStep::new(
            "Absolute value by sign: case argument nonnegative (|f| = f)".to_string(),
            Equation {
                lhs: pos_expr,
                rhs: zero,
                op: eq.op.clone(),
            },
            crate::ImportanceLevel::Medium,
        ),
        crate::SolveStep::new(
            "Absolute value by sign: case argument negative (|f| = -f)".to_string(),
            Equation {
                lhs: neg_expr,
                rhs: zero,
                op: eq.op.clone(),
            },
            crate::ImportanceLevel::Medium,
        ),
    ];

    let final_pos = intersect_solution_sets(&simplifier.context, pos_branch, pos_domain);
    let final_neg = intersect_solution_sets(&simplifier.context, neg_branch, neg_domain);
    Some((
        union_solution_sets(&simplifier.context, final_pos, final_neg),
        steps,
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
pub(super) fn try_solve_multi_abs_polynomial_relation(
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

/// Solve `|A(x)| = c` where `A` contains a trig atom and `c` is a nonnegative rational constant, by the
/// textbook split `A = c ∨ A = −c` — solving EACH branch with the full solver so a trig argument yields
/// its PERIODIC family, then unioning. The generic absolute-value isolation solves the branches to
/// PRINCIPAL roots only (`|2·sin(x) − 1| = 1 → {π/2, 0}` instead of `{π/2+2kπ} ∪ {kπ}`). Scoped to a
/// trig-bearing argument so the (correct) non-trig abs path is untouched; bare `|trig| = c` is already
/// handled earlier by the periodic-trig reduction.
pub(super) fn try_solve_abs_of_trig_equation(
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

/// The argument `g` of a bare `abs(g)`, else `None`.
pub(super) fn match_abs_argument(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if args.len() == 1 && ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Abs) {
            return Some(args[0]);
        }
    }
    None
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
pub(super) fn try_solve_const_over_abs_denominator_vs_zero(
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
pub(super) fn try_solve_nested_abs_relation(
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

/// Reduce `trig(g)² ⋚ t` (`is_square`) or `|trig(g)| ⋚ t` to window relations
/// on `trig(g)` and combine with the circular same-period algebra. The square
/// first takes the `|trig| ⋚ √t` route (exact when √t lands rational); when a
/// surd bound makes the sub-solves decline, sin/cos squares fall back to the
/// double-angle reduction — `sin² ⋚ t ⟺ cos(2g) ⋛ 1−2t` (flipped),
/// `cos² ⋚ t ⟺ cos(2g) ⋚ 2t−1` (same) — whose threshold is rational again
/// (the inequality mirror of the equation-path reciprocal-square reducer;
/// before it, `sin(x)² < 1/3` and every `sec²`-derived part declined).
pub(super) fn solve_trig_square_or_abs_rel(
    simplifier: &mut Simplifier,
    trig_fn: cas_ast::BuiltinFn,
    g: ExprId,
    is_square: bool,
    op: cas_ast::RelOp,
    t: num_rational::BigRational,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::{BuiltinFn, RelOp};
    use num_rational::BigRational;
    use num_traits::{One, Zero};

    // The non-positive-threshold edges settle by the sign of a square / abs.
    // Tanh/Cosh join the sin/cos arms: total domain, no poles (inner-argument
    // domain conditions attach through the shared required-conditions
    // machinery, same as `sin(ln(x))² ≥ 0` → «ℝ if x > 0»).
    let pole_free = matches!(
        trig_fn,
        BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Tanh | BuiltinFn::Cosh | BuiltinFn::Sinh
    );
    let zero = BigRational::zero();
    if t < zero {
        // trig² (or |trig|) ≥ 0 > t everywhere it is defined.
        return match op {
            RelOp::Lt | RelOp::Leq => Some(SolutionSet::Empty),
            // `> t` / `≥ t` for t < 0 is always true — but tan is undefined
            // at its poles, so only the pole-free functions are
            // unconditionally ℝ.
            RelOp::Gt | RelOp::Geq if pole_free => Some(SolutionSet::AllReals),
            _ => None,
        };
    }
    if t == zero {
        match op {
            RelOp::Lt => return Some(SolutionSet::Empty), // trig² < 0 impossible
            RelOp::Leq => return None, // trig² ≤ 0 ⟺ trig = 0, a point set — decline
            // trig² ≥ 0 is always true for the pole-free functions; tan is
            // punctured at its poles, so decline there.
            RelOp::Geq if pole_free => return Some(SolutionSet::AllReals),
            RelOp::Geq => return None,
            // trig² > 0 ⟺ trig ≠ 0: fall through to the r = 0 reduction
            // (`trig > 0 ∪ trig < 0` → the punctured line), NOT AllReals.
            RelOp::Gt => {}
            _ => return None,
        }
    }

    if let Some(set) =
        trig_abs_threshold_window_split(simplifier, trig_fn, g, is_square, op.clone(), &t, var)
    {
        return Some(set);
    }

    // Double-angle fallback (squares of sin/cos only).
    if !is_square {
        return None;
    }
    let two = BigRational::from_integer(2.into());
    let (sub_op, bound) = match trig_fn {
        BuiltinFn::Sin => (flip_inequality(op), BigRational::one() - &two * &t),
        BuiltinFn::Cos => (op, &two * &t - BigRational::one()),
        _ => return None,
    };
    let bound_expr = rational_to_expr(&mut simplifier.context, &bound);
    let two_expr = rational_to_expr(&mut simplifier.context, &two);
    let doubled_raw = simplifier.context.add(Expr::Mul(two_expr, g));
    let (doubled, _) = simplifier.simplify(doubled_raw);
    let cos_call = simplifier
        .context
        .call_builtin(BuiltinFn::Cos, vec![doubled]);
    let sub_eq = Equation {
        lhs: cos_call,
        rhs: bound_expr,
        op: sub_op,
    };
    let (set, _) = crate::solver_entrypoints_solve::solve(&sub_eq, var, simplifier).ok()?;
    if matches!(set, SolutionSet::Residual(_) | SolutionSet::Conditional(_)) {
        return None;
    }
    Some(set)
}

/// The `|trig(g)| ⋚ r` window split (r = √t for the square, r = t for the
/// abs): sub-solve `trig ⋚ ±r` through the full pipeline and combine.
fn trig_abs_threshold_window_split(
    simplifier: &mut Simplifier,
    trig_fn: cas_ast::BuiltinFn,
    g: ExprId,
    is_square: bool,
    op: cas_ast::RelOp,
    t: &num_rational::BigRational,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::RelOp;

    // r = √t for the square, r = t for the abs.
    let r_expr = if is_square {
        let t_expr = rational_to_expr(&mut simplifier.context, t);
        let sqrt_call = simplifier.context.call("sqrt", vec![t_expr]);
        simplifier.simplify(sqrt_call).0
    } else {
        rational_to_expr(&mut simplifier.context, t)
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

/// Reduce `hyper(g)² {op} t` (`is_square`) or `|hyper(g)| {op} t` to exact
/// sets: range edges settle to ℝ/∅/{g = 0}/g ≠ 0, and interior thresholds
/// build the symmetric ar*-band DIRECTLY for affine `g` (see
/// `build_affine_symmetric_band_or_complement` — the even-power split's
/// generic route produces rays with symbolic ar*-endpoints the set algebra
/// cannot order). `t` is the already-normalized rational threshold.
pub(super) fn solve_hyperbolic_square_or_abs_rel(
    simplifier: &mut Simplifier,
    hyper_fn: cas_ast::BuiltinFn,
    g: ExprId,
    is_square: bool,
    op: cas_ast::RelOp,
    t: num_rational::BigRational,
    var: &str,
) -> Option<SolutionSet> {
    use cas_ast::{BuiltinFn, RelOp};
    use num_rational::BigRational;
    use num_traits::Zero;

    let g_poly = cas_math::polynomial::Polynomial::from_expr(&simplifier.context, g, var).ok()?;
    if g_poly.degree() < 1 {
        return None;
    }
    let zero = BigRational::zero();
    let one = BigRational::from_integer(1.into());

    // `g = 0` / `g ≠ 0` via the full solver on the POLYNOMIAL relation
    // (rational endpoints — the set algebra is safe there).
    macro_rules! g_eq_zero {
        () => {{
            let z = simplifier.context.num(0);
            solve_relation_set(simplifier, var, g, z, RelOp::Eq)
        }};
    }
    macro_rules! g_ne_zero {
        () => {{
            let z = simplifier.context.num(0);
            let lo = solve_relation_set(simplifier, var, g, z, RelOp::Lt)?;
            let hi = solve_relation_set(simplifier, var, g, z, RelOp::Gt)?;
            Some(cas_solver_core::solution_set::union_solution_sets(
                &simplifier.context,
                lo,
                hi,
            ))
        }};
    }

    if matches!(hyper_fn, BuiltinFn::Cosh) {
        // T = cosh² ∈ [1, ∞) (or |cosh| = cosh ≥ 1): squaring is monotone on
        // the positive side, so `cosh² {op} t ⟺ cosh {op} √t` for t > 0 —
        // and both spellings compare against the minimum via `t vs 1`
        // (√t ⋚ 1 ⟺ t ⋚ 1 exactly).
        if t <= zero {
            return match op {
                RelOp::Lt | RelOp::Leq => Some(SolutionSet::Empty),
                RelOp::Gt | RelOp::Geq => Some(SolutionSet::AllReals),
                _ => None,
            };
        }
        return if t < one {
            match op {
                RelOp::Lt | RelOp::Leq => Some(SolutionSet::Empty),
                RelOp::Gt | RelOp::Geq => Some(SolutionSet::AllReals),
                _ => None,
            }
        } else if t == one {
            match op {
                RelOp::Lt => Some(SolutionSet::Empty),
                RelOp::Leq => g_eq_zero!(),
                RelOp::Geq => Some(SolutionSet::AllReals),
                RelOp::Gt => g_ne_zero!(),
                _ => None,
            }
        } else {
            // Threshold above the minimum: band at acosh(√t) (or acosh(t)).
            let t_expr = simplifier.context.add(Expr::Number(t));
            let c_new = if is_square {
                let sq = simplifier.context.call("sqrt", vec![t_expr]);
                simplifier.simplify(sq).0
            } else {
                t_expr
            };
            let a_pos = simplifier
                .context
                .call_builtin(BuiltinFn::Acosh, vec![c_new]);
            let (a_pos, _) = simplifier.simplify(a_pos);
            build_affine_symmetric_band_or_complement(simplifier, &g_poly, a_pos, op)
        };
    }

    // Odd increasing hyper (sinh / tanh): T = hyper² (or |hyper|) with
    // |hyper(g)| {op} r ⟺ the symmetric band ar*(±r) — plus tanh's range
    // ceiling |tanh| < 1.
    if t < zero {
        return match op {
            RelOp::Lt | RelOp::Leq => Some(SolutionSet::Empty),
            RelOp::Gt | RelOp::Geq => Some(SolutionSet::AllReals),
            _ => None,
        };
    }
    if t == zero {
        return match op {
            RelOp::Lt => Some(SolutionSet::Empty),
            RelOp::Leq => g_eq_zero!(),
            RelOp::Geq => Some(SolutionSet::AllReals),
            RelOp::Gt => g_ne_zero!(),
            _ => None,
        };
    }
    // r = √t for the square, r = t for the abs; tanh saturates at r ≥ 1
    // (both spellings: r ≥ 1 ⟺ t ≥ 1 exactly).
    if matches!(hyper_fn, BuiltinFn::Tanh) && t >= one {
        return match op {
            RelOp::Lt | RelOp::Leq => Some(SolutionSet::AllReals),
            // |tanh| never reaches 1: ≥ r and > r are both empty for r ≥ 1.
            RelOp::Gt | RelOp::Geq => Some(SolutionSet::Empty),
            _ => None,
        };
    }
    let t_expr = simplifier.context.add(Expr::Number(t));
    let r_expr = if is_square {
        let sq = simplifier.context.call("sqrt", vec![t_expr]);
        simplifier.simplify(sq).0
    } else {
        t_expr
    };
    let inv_fn = match hyper_fn {
        BuiltinFn::Sinh => BuiltinFn::Asinh,
        BuiltinFn::Tanh => BuiltinFn::Atanh,
        _ => return None,
    };
    let hi_u = simplifier.context.call_builtin(inv_fn, vec![r_expr]);
    let (hi_u, _) = simplifier.simplify(hi_u);
    build_affine_symmetric_band_or_complement(simplifier, &g_poly, hi_u, op)
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
pub(super) fn try_solve_abs_equals_zero(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<(SolutionSet, Vec<crate::SolveStep>)> {
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
    let mut steps = vec![crate::SolveStep::new(
        "Absolute value is zero exactly when its argument is zero".to_string(),
        inner_eq.clone(),
        crate::ImportanceLevel::Medium,
    )];
    let (sol, inner_steps) =
        crate::solver_entrypoints_solve::solve(&inner_eq, var, simplifier).ok()?;
    steps.extend(inner_steps);
    Some((sol, steps))
}

pub(super) fn try_solve_abs_polynomial_equation(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
    steps_out: &mut Vec<crate::SolveStep>,
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
    for (case_idx, rhs) in [g_core, neg_g].into_iter().enumerate() {
        let branch = Equation {
            lhs: f,
            rhs,
            op: RelOp::Eq,
        };
        let branch_display = format!(
            "{} = {}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: f
            },
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: rhs
            }
        );
        steps_out.push(crate::SolveStep::new(
            format!(
                "Split absolute value (Case {}): {}",
                case_idx + 1,
                branch_display
            ),
            branch.clone(),
            crate::ImportanceLevel::Medium,
        ));
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
pub(super) fn try_solve_abs_vs_symbolic_param(
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

/// Return the `abs(arg)` argument of `x` (a unary `|·|` call), or None.
pub(super) fn abs_call_arg(ctx: &Context, x: ExprId) -> Option<ExprId> {
    use cas_ast::BuiltinFn;
    if let Expr::Function(fn_id, args) = ctx.get(x) {
        if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Abs) {
            return Some(args[0]);
        }
    }
    None
}

/// Solve `sign(g(x)) {op} c` written as `g/|g| {op} c` (or `|g|/g {op} c`), `c` constant. Because
/// `sign(g) ∈ {−1, +1}` (undefined at `g = 0`), the relation reduces to which of those two values
/// satisfy `s {op} c`: only `+1` ⇒ `g > 0`; only `−1` ⇒ `g < 0`; both ⇒ `g ≠ 0`; neither ⇒ ∅. Solving
/// the strict sign condition on `g` yields OPEN intervals that EXCLUDE the `g = 0` pole — the generic
/// path returned a closed ray including the `0/0` point, or "No solution" for the inequality forms.
pub(super) fn try_solve_sign_via_abs(
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
/// Total number of `abs(...)` nodes anywhere in `e` (the F5 nested-multi-abs
/// gate: `|E| = c` with E combining ≥ 2 inner abs counts ≥ 3 with its outer).
pub(super) fn count_abs_nodes(ctx: &Context, e: ExprId) -> usize {
    let own = match ctx.get(e) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Abs) =>
        {
            1
        }
        _ => 0,
    };
    let children: Vec<ExprId> = match ctx.get(e) {
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            vec![*l, *r]
        }
        Expr::Neg(u) => vec![*u],
        Expr::Function(_, args) => args.clone(),
        _ => vec![],
    };
    own + children
        .into_iter()
        .map(|c| count_abs_nodes(ctx, c))
        .sum::<usize>()
}

pub(super) fn try_solve_abs_equality(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::solution_set::union_solution_sets;
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

    // Collect the branch results. All-Discrete branches keep the original
    // value-dedup path below (byte-identical for the pinned `|x²-2x| = 3`
    // family); a ray/interval branch (F5: `|x|−|x−2| = 2` → `[2, ∞)` and
    // `= −2` → `(−∞, 0]` — a FLAT region of the piecewise argument) unions
    // the FULL solution sets instead. Unsolved fragments still bail.
    let mut roots: Vec<ExprId> = Vec::new();
    let mut full_sets: Vec<SolutionSet> = Vec::new();
    let mut all_discrete = true;
    for branch in branches {
        match branch {
            SolutionSet::Discrete(ref rs) => {
                roots.extend(rs.iter().copied());
                full_sets.push(branch);
            }
            SolutionSet::Empty => {}
            SolutionSet::Continuous(_) | SolutionSet::Union(_) | SolutionSet::AllReals => {
                all_discrete = false;
                full_sets.push(branch);
            }
            _ => return None,
        }
    }
    if !all_discrete {
        let mut acc = SolutionSet::Empty;
        for s in full_sets {
            acc = union_solution_sets(&simplifier.context, acc, s);
        }
        return Some(acc);
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

/// F5 members 5-6 (frontier-audit 2026-07-14): `|N| / |D| = c` (c a rational
/// constant) for the NESTED-abs numerator family (`||x|−2| / |x| = 1` →
/// `{−1}`, losing the twin `1`). Clears to `|N| = c·|D|` — the textbook
/// multiplication by the positive `|D|` — re-solves through the FULL
/// pipeline (whose nested-abs machinery the sibling recovery owns), then
/// enforces the ratio's own definedness `D ≠ 0` on each root EXACTLY: a
/// cleared root with `D = 0` is a `0/0` point of the original quotient and
/// must not be published (this is also what makes the `c = 0` reduction
/// `|N| = 0` sound). Declines on any non-Discrete cleared result (a
/// flat-region ray would need point-puncture set algebra) and on any root
/// whose `D` value cannot be decided exactly — never a float drop.
pub(super) fn try_solve_abs_ratio_equality(
    simplifier: &mut Simplifier,
    eq: &Equation,
    var: &str,
) -> Option<SolutionSet> {
    use cas_math::numeric_eval::as_rational_const;
    use num_rational::BigRational;
    use num_traits::Zero;

    if !matches!(eq.op, cas_ast::RelOp::Eq) {
        return None;
    }
    let Expr::Div(num, den) = simplifier.context.get(eq.lhs) else {
        return None;
    };
    let (num, den) = (*num, *den);
    // Both sides of the quotient must be abs calls (`|N| / |D|`).
    match_abs_argument(&simplifier.context, num)?;
    let d_arg = match_abs_argument(&simplifier.context, den)?;
    if !cas_solver_core::isolation_utils::contains_var(&simplifier.context, eq.lhs, var) {
        return None;
    }
    let c = as_rational_const(&simplifier.context, eq.rhs)?;
    if c < BigRational::zero() {
        return Some(SolutionSet::Empty); // |N|/|D| ≥ 0 wherever it is defined
    }
    // Cleared equation `|N| = c·|D|` through the full solver.
    let rhs2_raw = simplifier.context.add(Expr::Mul(eq.rhs, den));
    let (rhs2, _) = simplifier.simplify(rhs2_raw);
    let cleared = solve_relation_set(simplifier, var, num, rhs2, cas_ast::RelOp::Eq)?;
    let roots = match cleared {
        SolutionSet::Discrete(rs) => rs,
        SolutionSet::Empty => return Some(SolutionSet::Empty),
        _ => return None,
    };
    // Exact `D ≠ 0` filter per root (`|d_arg|` vanishes iff `d_arg` does).
    let var_id = simplifier.context.var(var);
    let mut kept = Vec::new();
    for root in roots {
        let d_at = substitute_expr_by_id(&mut simplifier.context, d_arg, var_id, root);
        let (d_at, _) = simplifier.simplify(d_at);
        match as_rational_const(&simplifier.context, d_at) {
            Some(v) if v.is_zero() => {} // 0/0 point of the original: drop
            Some(_) => kept.push(root),
            None => return None, // undecidable definedness: decline whole recovery
        }
    }
    if kept.is_empty() {
        return Some(SolutionSet::Empty);
    }
    Some(SolutionSet::Discrete(kept))
}
