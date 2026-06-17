//! Local active solve backend boundary.
//!
//! This backend is solver-owned and executes the solver-native runtime pipeline.
//! Keeping this indirection local lets us switch implementations without
//! changing call sites.

use crate::{CasError, Simplifier, SolveCtx, SolveStep};
use cas_ast::{Constant, Context, Equation, Expr, ExprId, SolutionSet};
use std::collections::HashMap;

use crate::solve_backend_contract::{CoreSolverOptions, SolveBackend};

/// True when `expr` is not a real value: it contains a non-finite / undefined
/// constant (∞ or undefined) anywhere, or an out-of-range inverse-trig term
/// (`arcsin(c)` / `arccos(c)` with `|c| > 1`, whose real domain is `[-1, 1]`).
/// Such a value is never a real solution of an equation over ℝ — e.g.
/// `solve(cos(x)=2, x)` must not report `{ arccos(2) }`.
fn solution_contains_nonfinite(ctx: &Context, expr: ExprId) -> bool {
    use cas_ast::BuiltinFn;
    use num_traits::Signed;
    match ctx.get(expr) {
        Expr::Constant(Constant::Infinity | Constant::Undefined) => true,
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            solution_contains_nonfinite(ctx, *a) || solution_contains_nonfinite(ctx, *b)
        }
        Expr::Neg(a) | Expr::Hold(a) => solution_contains_nonfinite(ctx, *a),
        Expr::Function(fn_id, args) => {
            // arcsin/arccos of a constant outside [-1, 1] is undefined over ℝ.
            if args.len() == 1
                && (ctx.is_builtin(*fn_id, BuiltinFn::Arcsin)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Arccos)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Asin)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Acos))
                && cas_math::numeric_eval::as_rational_const(ctx, args[0])
                    .is_some_and(|c| c.abs() > num_rational::BigRational::from_integer(1.into()))
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
fn filter_real_solutions(ctx: &Context, eq: &Equation, var: &str, set: SolutionSet) -> SolutionSet {
    match set {
        SolutionSet::Discrete(sols) => {
            let kept: Vec<ExprId> = sols
                .into_iter()
                .filter(|&s| {
                    !solution_contains_nonfinite(ctx, s)
                        && check_root(ctx, eq, var, s) != RootCheck::Extraneous
                })
                .collect();
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
                let kept: Vec<_> = cases
                    .into_iter()
                    .map(|mut case| {
                        case.then.solutions =
                            filter_real_solutions(ctx, eq, var, case.then.solutions.clone());
                        case
                    })
                    .filter(|case| !matches!(case.then.solutions, SolutionSet::Empty))
                    .collect();
                if kept.is_empty() {
                    SolutionSet::Empty
                } else {
                    SolutionSet::Conditional(kept)
                }
            }
        }
        other => other,
    }
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
        if equation_is_nonzero_const_over_polynomial(simplifier, eq) {
            return Ok((SolutionSet::Empty, Vec::new()));
        }
        let (set, steps) = crate::solve_core_runtime::solve_inner(eq, var, simplifier, opts, ctx)?;
        let set = filter_real_solutions(&simplifier.context, eq, var, set);
        Ok((set, steps))
    }
}

#[cfg(test)]
mod tests {
    use super::solution_contains_nonfinite;
    use cas_ast::{Context, Expr};

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
    }
}
