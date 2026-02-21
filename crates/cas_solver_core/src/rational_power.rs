use crate::isolation_utils::contains_var;
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
}
