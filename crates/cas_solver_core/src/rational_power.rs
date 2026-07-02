use crate::isolation_utils::{
    contains_var, flip_inequality, is_numeric_one, is_positive_integer_expr,
    match_exponential_var_in_base, match_exponential_var_in_exponent,
};
use crate::log_domain::{
    classify_log_linear_rewrite_route, LogAssumption, LogLinearRewriteRoute, LogSolveDecision,
};
use cas_ast::{Context, Equation, Expr, ExprId, RelOp};

/// Match `Pow(base, p/q)` where `base` contains `var` and `p/q` is non-integer rational.
pub fn match_rational_power(ctx: &Context, expr: ExprId, var: &str) -> Option<(ExprId, i64, i64)> {
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if !contains_var(ctx, *base, var) {
            return None;
        }

        match ctx.get(*exp) {
            Expr::Number(n) => {
                let denom = n.denom();
                let numer = n.numer();
                if *denom == 1.into() {
                    return None;
                }
                let p: i64 = numer.try_into().ok()?;
                let q: i64 = denom.try_into().ok()?;
                if q <= 0 {
                    return None;
                }
                Some((*base, p, q))
            }
            Expr::Div(num_id, den_id) => {
                if let (Expr::Number(p_rat), Expr::Number(q_rat)) =
                    (ctx.get(*num_id), ctx.get(*den_id))
                {
                    if !p_rat.is_integer() || !q_rat.is_integer() {
                        return None;
                    }
                    let p: i64 = p_rat.numer().try_into().ok()?;
                    let q: i64 = q_rat.numer().try_into().ok()?;
                    if q <= 1 {
                        return None;
                    }
                    Some((*base, p, q))
                } else {
                    None
                }
            }
            _ => None,
        }
    } else {
        None
    }
}

/// Rewrite `base^(p/q) = rhs` into `base^p = rhs^q` for rational exponent elimination.
///
/// Returns the transformed equation and exponent pair `(p, q)`.
pub fn rewrite_rational_power_equation(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
) -> Option<(Equation, i64, i64)> {
    let (base, p, q) = match_rational_power(ctx, lhs, var)?;
    let p_expr = ctx.num(p);
    let q_expr = ctx.num(q);
    let new_lhs = ctx.add(Expr::Pow(base, p_expr));
    let new_rhs = ctx.add(Expr::Pow(rhs, q_expr));
    Some((
        Equation {
            lhs: new_lhs,
            rhs: new_rhs,
            op: RelOp::Eq,
        },
        p,
        q,
    ))
}

/// Rewrite helper for the common isolated pattern used by the solver:
/// only rewrites when the equation is an equality with the variable
/// appearing on the left side and not on the right side.
pub fn rewrite_isolated_rational_power_equation(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
    op: RelOp,
    lhs_has_var: bool,
    rhs_has_var: bool,
) -> Option<(Equation, i64, i64)> {
    if op != RelOp::Eq || !lhs_has_var || rhs_has_var {
        return None;
    }
    rewrite_rational_power_equation(ctx, lhs, rhs, var)
}

/// Rewrite an exponential equation with variable in the base:
/// `A^n op B  ->  A op B^(1/n)`, but only when `n` is not a positive integer.
///
/// Returns the transformed equation and the original exponent expression `n`.
pub fn rewrite_variable_base_power_equation(
    ctx: &mut Context,
    target: ExprId,
    other: ExprId,
    var: &str,
    op: RelOp,
    is_lhs: bool,
) -> Option<(Equation, ExprId)> {
    let pattern = match_exponential_var_in_base(ctx, target, var)?;
    let base = pattern.base;
    let exponent = pattern.exponent;

    // Keep integer powers in polynomial/quadratic strategies.
    if is_positive_integer_expr(ctx, exponent) {
        return None;
    }

    let one = ctx.num(1);
    let inv_exp = ctx.add(Expr::Div(one, exponent));
    let transformed_other = ctx.add(Expr::Pow(other, inv_exp));
    let equation = if is_lhs {
        Equation {
            lhs: base,
            rhs: transformed_other,
            op,
        }
    } else {
        Equation {
            lhs: transformed_other,
            rhs: base,
            op,
        }
    };
    Some((equation, exponent))
}

/// True iff `base` is PROVABLY a constant in `(0, 1)`, i.e. `ln(base) < 0`, so an inequality
/// isolated through `log(base, ·)` must flip direction. Exact only (no f64): a rational folds
/// directly; a constant irrational base (`sin(1)`, `√2/2`, `e^(-1)`) is decided by the exact
/// value-bounds oracle. Undecidable/symbolic bases return `false` (no flip without proof —
/// `sin(1)^x > 2` used to return the reversed ray because this only knew rationals).
fn base_is_provably_fraction_below_one(ctx: &Context, base: ExprId) -> bool {
    use num_traits::{One, Signed, Zero};
    if let Some(b) = cas_math::numeric_eval::as_rational_const(ctx, base) {
        return b > num_rational::BigRational::zero() && b < num_rational::BigRational::one();
    }
    matches!(
        cas_math::const_sign::const_value_bounds(ctx, base),
        Some((lo, hi)) if lo.is_positive() && hi < num_rational::BigRational::one()
    )
}

/// Build the log-linear rewrite of `base^exponent = other`:
/// `exponent * ln(base) = ln(other)`.
///
/// `is_lhs` controls which side contained the original exponential target.
pub fn build_log_linear_equation(
    ctx: &mut Context,
    base: ExprId,
    exponent: ExprId,
    other: ExprId,
    op: RelOp,
    is_lhs: bool,
) -> Equation {
    // The rewrite `exponent·ln(base) op ln(other)` is solved downstream by dividing through by the
    // coefficient `ln(base)`, and the mul-isolation pipeline decides that coefficient's sign with the
    // exact `is_known_negative` oracle (which resolves `ln(b)` for any provably `0 < b < 1` base,
    // rational or not) and flips the relation itself. Do NOT pre-flip here: a second flip at this
    // layer double-compensates and reverses the ray (`(1/2)^x>4` regressed exactly that way when the
    // oracle learned to decide `ln(1/2) < 0`). Eq/Neq are unaffected either way.
    let ln_base = ctx.call("ln", vec![base]);
    let lhs_linear = ctx.add(Expr::Mul(exponent, ln_base));
    let ln_other = ctx.call("ln", vec![other]);

    if is_lhs {
        Equation {
            lhs: lhs_linear,
            rhs: ln_other,
            op,
        }
    } else {
        Equation {
            lhs: ln_other,
            rhs: lhs_linear,
            op,
        }
    }
}

/// Route for rewriting `base^exponent op other` into a log-linear equation
/// used by unwrap/isolation strategies.
#[derive(Debug, Clone, PartialEq)]
pub enum LogLinearUnwrapPlan<'a> {
    BaseOneShortcut,
    Proceed {
        equation: Equation,
        assumptions: &'a [LogAssumption],
    },
    Blocked,
}

/// Build log-linear unwrap equation when domain decision allows it.
pub fn plan_log_linear_unwrap_equation<'a>(
    ctx: &mut Context,
    base: ExprId,
    exponent: ExprId,
    other: ExprId,
    op: RelOp,
    is_lhs: bool,
    decision: &'a LogSolveDecision,
) -> LogLinearUnwrapPlan<'a> {
    match classify_log_linear_rewrite_route(is_numeric_one(ctx, base), decision) {
        LogLinearRewriteRoute::BaseOneShortcut => LogLinearUnwrapPlan::BaseOneShortcut,
        LogLinearRewriteRoute::Proceed { assumptions } => LogLinearUnwrapPlan::Proceed {
            equation: build_log_linear_equation(ctx, base, exponent, other, op, is_lhs),
            assumptions,
        },
        LogLinearRewriteRoute::Blocked => LogLinearUnwrapPlan::Blocked,
    }
}

/// Unified unwrap planning for power targets used by solver strategies.
#[derive(Debug, Clone, PartialEq)]
pub enum PowUnwrapPlan {
    VariableBase {
        equation: Equation,
        exponent: ExprId,
    },
    LogLinear {
        equation: Equation,
        base: ExprId,
        assumptions: Vec<LogAssumption>,
    },
}

/// Plan a power unwrap rewrite (`Pow`) using either base-isolation or log-linear route.
pub fn plan_pow_unwrap_rewrite<F>(
    ctx: &mut Context,
    target: ExprId,
    other: ExprId,
    var: &str,
    op: RelOp,
    is_lhs: bool,
    mut classify_log_solve: F,
) -> Option<PowUnwrapPlan>
where
    F: FnMut(&Context, ExprId, ExprId) -> LogSolveDecision,
{
    if let Some((equation, exponent)) =
        rewrite_variable_base_power_equation(ctx, target, other, var, op.clone(), is_lhs)
    {
        return Some(PowUnwrapPlan::VariableBase { equation, exponent });
    }

    let pattern = match_exponential_var_in_exponent(ctx, target, var)?;
    let decision = classify_log_solve(ctx, pattern.base, other);
    match plan_log_linear_unwrap_equation(
        ctx,
        pattern.base,
        pattern.exponent,
        other,
        op,
        is_lhs,
        &decision,
    ) {
        LogLinearUnwrapPlan::Proceed {
            equation,
            assumptions,
        } => Some(PowUnwrapPlan::LogLinear {
            equation,
            base: pattern.base,
            assumptions: assumptions.to_vec(),
        }),
        LogLinearUnwrapPlan::BaseOneShortcut | LogLinearUnwrapPlan::Blocked => None,
    }
}

/// Build the root-isolation rewrite of `base^exponent = rhs`:
/// `base = rhs^(1/exponent)`.
///
/// When `use_abs_base` is true, the equation is built as
/// `abs(base) = rhs^(1/exponent)` (used for even roots).
pub fn build_root_isolation_equation(
    ctx: &mut Context,
    base: ExprId,
    exponent: ExprId,
    rhs: ExprId,
    op: RelOp,
    use_abs_base: bool,
) -> Equation {
    let one = ctx.num(1);
    let inv_exp = ctx.add(Expr::Div(one, exponent));
    let transformed_rhs = ctx.add(Expr::Pow(rhs, inv_exp));
    let lhs = if use_abs_base {
        ctx.call("abs", vec![base])
    } else {
        base
    };
    Equation {
        lhs,
        rhs: transformed_rhs,
        op,
    }
}

/// Build the logarithmic isolation rewrite for `base^exponent op rhs`:
/// `exponent op log(base, rhs)`.
pub fn build_exponent_log_isolation_equation(
    ctx: &mut Context,
    exponent: ExprId,
    base: ExprId,
    rhs: ExprId,
    op: RelOp,
) -> Equation {
    // `log(base, ·)` is DECREASING when `0 < base < 1` (`ln(base) < 0`), so isolating the exponent
    // out of an INEQUALITY must FLIP the relation's direction. Keyed on the EXACT rational base (no
    // f64): only a provable `0 < base < 1` flips; `base > 1`, `base = 1`, and symbolic / irrational
    // bases keep `op` unchanged (the sound default — without a proven sign of `ln(base)` we must not
    // flip). Equality / inequation pass through untouched.
    let op = if matches!(op, RelOp::Lt | RelOp::Leq | RelOp::Gt | RelOp::Geq)
        && base_is_provably_fraction_below_one(ctx, base)
    {
        flip_inequality(op)
    } else {
        op
    };
    let rhs_log = ctx.call("log", vec![base, rhs]);
    Equation {
        lhs: exponent,
        rhs: rhs_log,
        op,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_number_rational_exponent() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let three_over_two = ctx.add(Expr::Number(num_rational::BigRational::new(
            3.into(),
            2.into(),
        )));
        let expr = ctx.add(Expr::Pow(x, three_over_two));
        let m = match_rational_power(&ctx, expr, "x").expect("must match x^(3/2)");
        assert_eq!(m.1, 3);
        assert_eq!(m.2, 2);
    }

    #[test]
    fn reject_integer_exponent() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Pow(x, two));
        assert!(match_rational_power(&ctx, expr, "x").is_none());
    }

    #[test]
    fn reject_when_base_does_not_contain_var() {
        let mut ctx = Context::new();
        let y = ctx.var("y");
        let three_over_two = ctx.add(Expr::Number(num_rational::BigRational::new(
            3.into(),
            2.into(),
        )));
        let expr = ctx.add(Expr::Pow(y, three_over_two));
        assert!(match_rational_power(&ctx, expr, "x").is_none());
    }

    #[test]
    fn rewrite_rational_power_equation_builds_expected_shape() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let three_over_two = ctx.add(Expr::Number(num_rational::BigRational::new(
            3.into(),
            2.into(),
        )));
        let lhs = ctx.add(Expr::Pow(x, three_over_two));
        let (eq, p, q) = rewrite_rational_power_equation(&mut ctx, lhs, y, "x")
            .expect("must rewrite rational exponent equation");

        assert_eq!(p, 3);
        assert_eq!(q, 2);
        assert!(matches!(ctx.get(eq.lhs), Expr::Pow(_, _)));
        assert!(matches!(ctx.get(eq.rhs), Expr::Pow(_, _)));
        assert_eq!(eq.op, RelOp::Eq);
    }

    #[test]
    fn rewrite_isolated_rational_power_equation_rejects_non_equality() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let three_over_two = ctx.rational(3, 2);
        let lhs = ctx.add(Expr::Pow(x, three_over_two));
        assert!(rewrite_isolated_rational_power_equation(
            &mut ctx,
            lhs,
            y,
            "x",
            RelOp::Lt,
            true,
            false
        )
        .is_none());
    }

    #[test]
    fn rewrite_isolated_rational_power_equation_rejects_rhs_with_var() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let three_over_two = ctx.rational(3, 2);
        let lhs = ctx.add(Expr::Pow(x, three_over_two));
        assert!(rewrite_isolated_rational_power_equation(
            &mut ctx,
            lhs,
            x,
            "x",
            RelOp::Eq,
            true,
            true
        )
        .is_none());
    }

    #[test]
    fn rewrite_isolated_rational_power_equation_accepts_isolated_case() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let three_over_two = ctx.rational(3, 2);
        let lhs = ctx.add(Expr::Pow(x, three_over_two));
        let out =
            rewrite_isolated_rational_power_equation(&mut ctx, lhs, y, "x", RelOp::Eq, true, false)
                .expect("isolated rational power should rewrite");
        assert_eq!(out.2, 2);
        assert_eq!(out.0.op, RelOp::Eq);
    }

    #[test]
    fn rewrite_variable_base_power_equation_rewrites_non_integer_exponent() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let half = ctx.rational(1, 2);
        let target = ctx.add(Expr::Pow(x, half));

        let (eq, exp) =
            rewrite_variable_base_power_equation(&mut ctx, target, y, "x", RelOp::Eq, true)
                .expect("non-integer exponent should rewrite");
        assert_eq!(exp, half);
        assert_eq!(eq.lhs, x);
        assert!(matches!(ctx.get(eq.rhs), Expr::Pow(base, _) if *base == y));
    }

    #[test]
    fn rewrite_variable_base_power_equation_rejects_positive_integer_exponent() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let two = ctx.num(2);
        let target = ctx.add(Expr::Pow(x, two));
        assert!(
            rewrite_variable_base_power_equation(&mut ctx, target, y, "x", RelOp::Eq, true)
                .is_none()
        );
    }

    #[test]
    fn rewrite_variable_base_power_equation_handles_rhs_target_orientation() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let half = ctx.rational(1, 2);
        let target = ctx.add(Expr::Pow(x, half));

        let (eq, _) =
            rewrite_variable_base_power_equation(&mut ctx, target, y, "x", RelOp::Eq, false)
                .expect("rewrite should support RHS target");
        assert!(matches!(ctx.get(eq.lhs), Expr::Pow(base, _) if *base == y));
        assert_eq!(eq.rhs, x);
    }

    #[test]
    fn build_log_linear_equation_lhs_orientation() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let x = ctx.var("x");
        let b = ctx.var("b");
        let eq = build_log_linear_equation(&mut ctx, a, x, b, RelOp::Eq, true);

        assert_eq!(eq.op, RelOp::Eq);
        assert!(matches!(ctx.get(eq.lhs), Expr::Mul(_, _)));
        assert!(matches!(ctx.get(eq.rhs), Expr::Function(_, _)));
    }

    #[test]
    fn build_log_linear_equation_rhs_orientation() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let x = ctx.var("x");
        let b = ctx.var("b");
        let eq = build_log_linear_equation(&mut ctx, a, x, b, RelOp::Eq, false);

        assert_eq!(eq.op, RelOp::Eq);
        assert!(matches!(ctx.get(eq.lhs), Expr::Function(_, _)));
        assert!(matches!(ctx.get(eq.rhs), Expr::Mul(_, _)));
    }

    #[test]
    fn plan_log_linear_unwrap_equation_blocks_base_one_shortcut() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let x = ctx.var("x");
        let y = ctx.var("y");
        let decision = crate::log_domain::LogSolveDecision::Ok;
        let plan = plan_log_linear_unwrap_equation(&mut ctx, one, x, y, RelOp::Eq, true, &decision);
        assert!(matches!(plan, LogLinearUnwrapPlan::BaseOneShortcut));
    }

    #[test]
    fn plan_log_linear_unwrap_equation_returns_blocked_for_needs_complex() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let x = ctx.var("x");
        let y = ctx.var("y");
        let decision = crate::log_domain::LogSolveDecision::NeedsComplex("needs complex");
        let plan = plan_log_linear_unwrap_equation(&mut ctx, a, x, y, RelOp::Eq, true, &decision);
        assert!(matches!(plan, LogLinearUnwrapPlan::Blocked));
    }

    #[test]
    fn plan_log_linear_unwrap_equation_builds_equation_with_assumptions() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let x = ctx.var("x");
        let y = ctx.var("y");
        let decision = crate::log_domain::LogSolveDecision::OkWithAssumptions(vec![
            crate::log_domain::LogAssumption::PositiveBase,
        ]);
        let plan = plan_log_linear_unwrap_equation(&mut ctx, a, x, y, RelOp::Eq, true, &decision);
        match plan {
            LogLinearUnwrapPlan::Proceed {
                equation,
                assumptions,
            } => {
                assert_eq!(equation.op, RelOp::Eq);
                assert_eq!(
                    assumptions,
                    &[crate::log_domain::LogAssumption::PositiveBase]
                );
            }
            _ => panic!("expected proceed plan"),
        }
    }

    #[test]
    fn plan_pow_unwrap_rewrite_prefers_variable_base_route() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let half = ctx.rational(1, 2);
        let target = ctx.add(Expr::Pow(x, half));
        let mut classifier_calls = 0usize;

        let plan = plan_pow_unwrap_rewrite(&mut ctx, target, y, "x", RelOp::Eq, true, |_, _, _| {
            classifier_calls += 1;
            crate::log_domain::LogSolveDecision::Ok
        })
        .expect("unwrap plan should be available");

        assert_eq!(classifier_calls, 0);
        assert!(matches!(plan, PowUnwrapPlan::VariableBase { .. }));
    }

    #[test]
    fn plan_pow_unwrap_rewrite_builds_log_linear_plan() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let x = ctx.var("x");
        let y = ctx.var("y");
        let target = ctx.add(Expr::Pow(a, x));

        let plan = plan_pow_unwrap_rewrite(&mut ctx, target, y, "x", RelOp::Eq, true, |_, _, _| {
            crate::log_domain::LogSolveDecision::OkWithAssumptions(vec![
                crate::log_domain::LogAssumption::PositiveBase,
                crate::log_domain::LogAssumption::PositiveRhs,
            ])
        })
        .expect("unwrap plan should be available");

        match plan {
            PowUnwrapPlan::LogLinear {
                equation,
                base,
                assumptions,
            } => {
                assert_eq!(base, a);
                assert_eq!(equation.op, RelOp::Eq);
                assert_eq!(assumptions.len(), 2);
            }
            _ => panic!("expected log-linear plan"),
        }
    }

    #[test]
    fn plan_pow_unwrap_rewrite_returns_none_for_blocked_log_route() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let x = ctx.var("x");
        let y = ctx.var("y");
        let target = ctx.add(Expr::Pow(a, x));

        let plan = plan_pow_unwrap_rewrite(&mut ctx, target, y, "x", RelOp::Eq, true, |_, _, _| {
            crate::log_domain::LogSolveDecision::NeedsComplex("needs complex")
        });

        assert!(plan.is_none());
    }

    #[test]
    fn build_root_isolation_equation_regular_base() {
        let mut ctx = Context::new();
        let b = ctx.var("b");
        let e = ctx.var("e");
        let r = ctx.var("r");
        let eq = build_root_isolation_equation(&mut ctx, b, e, r, RelOp::Eq, false);
        assert_eq!(eq.lhs, b);
        assert_eq!(eq.op, RelOp::Eq);
        assert!(matches!(ctx.get(eq.rhs), Expr::Pow(base, _) if *base == r));
    }

    #[test]
    fn build_root_isolation_equation_abs_base() {
        let mut ctx = Context::new();
        let b = ctx.var("b");
        let e = ctx.var("e");
        let r = ctx.var("r");
        let eq = build_root_isolation_equation(&mut ctx, b, e, r, RelOp::Eq, true);
        assert!(
            matches!(ctx.get(eq.lhs), Expr::Function(_, args) if args.len() == 1 && args[0] == b)
        );
        assert!(matches!(ctx.get(eq.rhs), Expr::Pow(base, _) if *base == r));
    }

    #[test]
    fn build_exponent_log_isolation_equation_shape() {
        let mut ctx = Context::new();
        let e = ctx.var("e");
        let b = ctx.var("b");
        let r = ctx.var("r");
        let eq = build_exponent_log_isolation_equation(&mut ctx, e, b, r, RelOp::Leq);
        assert_eq!(eq.lhs, e);
        assert_eq!(eq.op, RelOp::Leq);
        assert!(
            matches!(ctx.get(eq.rhs), Expr::Function(_, args) if args.len() == 2 && args[0] == b && args[1] == r)
        );
    }

    #[test]
    fn exponent_log_isolation_flips_for_provably_fractional_base() {
        // The `log(base, ·)` isolation IS the semantic flip site (no downstream
        // division): a PROVABLY `0 < base < 1` constant flips the relation —
        // rational (1/2) via the fold, irrational (sin(1), √2/2) via the exact
        // value-bounds oracle. Symbolic and >1 bases pass through unflipped.
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let parse = |ctx: &mut Context, src: &str| cas_parser::parse(src, ctx).expect("parse");
        let sin1 = parse(&mut ctx, "sin(1)");
        let eq = build_exponent_log_isolation_equation(&mut ctx, x, sin1, two, RelOp::Gt);
        assert_eq!(eq.op, RelOp::Lt, "sin(1)^x > 2 must flip (sin(1) ∈ (0,1))");
        let half_sqrt2 = parse(&mut ctx, "sqrt(2)/2");
        let eq2 = build_exponent_log_isolation_equation(&mut ctx, x, half_sqrt2, two, RelOp::Leq);
        assert_eq!(eq2.op, RelOp::Geq, "(√2/2)^x ≤ 2 must flip");
        let pi = parse(&mut ctx, "pi");
        let eq3 = build_exponent_log_isolation_equation(&mut ctx, x, pi, two, RelOp::Gt);
        assert_eq!(eq3.op, RelOp::Gt, "pi > 1 keeps the direction");
        let sym = ctx.var("a");
        let eq4 = build_exponent_log_isolation_equation(&mut ctx, x, sym, two, RelOp::Gt);
        assert_eq!(eq4.op, RelOp::Gt, "symbolic base: no flip without proof");
        // Equality never flips.
        let eq5 = build_exponent_log_isolation_equation(&mut ctx, x, sin1, two, RelOp::Eq);
        assert_eq!(eq5.op, RelOp::Eq);
    }

    #[test]
    fn log_linear_equation_never_pre_flips_inequality() {
        // Direction ownership: the downstream mul-isolation decides the sign of the
        // coefficient `ln(base)` with the exact oracle and flips there. The builder
        // must pass `op` through UNCHANGED for every base, or the two layers
        // double-flip (`(1/2)^x > 4` regressed to the reversed ray). The end-to-end
        // direction contract lives in the CLI test
        // `test_eval_fractional_base_exponential_inequality_direction`.
        let mut ctx = Context::new();
        let exponent = ctx.var("x");
        let other = ctx.num(4);
        let half = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let two = ctx.num(2);
        let eq = build_log_linear_equation(&mut ctx, half, exponent, other, RelOp::Gt, true);
        assert_eq!(eq.op, RelOp::Gt, "(1/2)^x > 4: flip is owned downstream");
        let eq2 = build_log_linear_equation(&mut ctx, two, exponent, other, RelOp::Gt, true);
        assert_eq!(eq2.op, RelOp::Gt, "2^x > 4 keeps Gt");
        // Equality is never flipped anywhere.
        let eq3 = build_log_linear_equation(&mut ctx, half, exponent, other, RelOp::Eq, true);
        assert_eq!(eq3.op, RelOp::Eq);
        // The fractional-base predicate stays exact (used by the log-isolation builder).
        assert!(base_is_provably_fraction_below_one(&ctx, half));
        assert!(!base_is_provably_fraction_below_one(&ctx, two));
    }
}
