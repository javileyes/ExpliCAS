use crate::isolation_utils::{
    contains_var, is_positive_integer_expr, match_exponential_var_in_base,
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
}
