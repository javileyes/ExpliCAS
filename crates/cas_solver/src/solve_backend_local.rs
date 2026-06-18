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
    use cas_math::poly_compare::{poly_relation, SignRelation};

    for (i, c1) in conds.iter().enumerate() {
        let ImplicitCondition::Positive(a) = c1 else {
            continue;
        };
        for c2 in conds.iter().skip(i + 1) {
            let ImplicitCondition::Positive(b) = c2 else {
                continue;
            };
            // `a > 0` and `b > 0` with `a == -b` cannot both hold.
            if matches!(poly_relation(ctx, *a, *b), Some(SignRelation::Negated)) {
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
        let conds = ctx.required_conditions();
        let set = filter_real_solutions(&mut simplifier.context, eq, var, set, &conds);
        Ok((set, steps))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        required_conditions_are_contradictory, root_violates_required_condition,
        solution_contains_nonfinite,
    };
    use cas_ast::{Context, Expr};
    use cas_solver_core::domain_condition::ImplicitCondition;

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
    }
}
