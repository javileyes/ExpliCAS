use crate::isolation_utils::contains_var;
use cas_ast::{Context, Expr, ExprId};

/// Represents a linear expression in one variable: `coef * var + constant`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearForm {
    pub coef: ExprId,
    pub constant: ExprId,
}

/// Extract the linear form of an expression with respect to `var`.
///
/// Returns `None` when the expression is non-linear in `var`
/// (for example `var^2`, `sin(var)`, or `var` in denominator).
pub fn linear_form(ctx: &mut Context, expr: ExprId, var: &str) -> Option<LinearForm> {
    // Base case: var-free expression is a constant.
    if !contains_var(ctx, expr, var) {
        let zero = ctx.num(0);
        return Some(LinearForm {
            coef: zero,
            constant: expr,
        });
    }

    match ctx.get(expr).clone() {
        Expr::Variable(sym_id) if ctx.sym_name(sym_id) == var => {
            let one = ctx.num(1);
            let zero = ctx.num(0);
            Some(LinearForm {
                coef: one,
                constant: zero,
            })
        }
        Expr::Variable(_) => {
            let zero = ctx.num(0);
            Some(LinearForm {
                coef: zero,
                constant: expr,
            })
        }
        Expr::Add(u, v) => {
            let lf_u = linear_form(ctx, u, var)?;
            let lf_v = linear_form(ctx, v, var)?;
            let coef = ctx.add(Expr::Add(lf_u.coef, lf_v.coef));
            let constant = ctx.add(Expr::Add(lf_u.constant, lf_v.constant));
            Some(LinearForm { coef, constant })
        }
        Expr::Sub(u, v) => {
            let lf_u = linear_form(ctx, u, var)?;
            let lf_v = linear_form(ctx, v, var)?;
            let coef = ctx.add(Expr::Sub(lf_u.coef, lf_v.coef));
            let constant = ctx.add(Expr::Sub(lf_u.constant, lf_v.constant));
            Some(LinearForm { coef, constant })
        }
        Expr::Neg(u) => {
            let lf_u = linear_form(ctx, u, var)?;
            let coef = ctx.add(Expr::Neg(lf_u.coef));
            let constant = ctx.add(Expr::Neg(lf_u.constant));
            Some(LinearForm { coef, constant })
        }
        Expr::Mul(u, v) => {
            let u_has = contains_var(ctx, u, var);
            let v_has = contains_var(ctx, v, var);

            match (u_has, v_has) {
                (true, true) => None,
                (false, true) => {
                    let lf_v = linear_form(ctx, v, var)?;
                    let coef = ctx.add(Expr::Mul(u, lf_v.coef));
                    let constant = ctx.add(Expr::Mul(u, lf_v.constant));
                    Some(LinearForm { coef, constant })
                }
                (true, false) => {
                    let lf_u = linear_form(ctx, u, var)?;
                    let coef = ctx.add(Expr::Mul(lf_u.coef, v));
                    let constant = ctx.add(Expr::Mul(lf_u.constant, v));
                    Some(LinearForm { coef, constant })
                }
                (false, false) => {
                    let zero = ctx.num(0);
                    Some(LinearForm {
                        coef: zero,
                        constant: expr,
                    })
                }
            }
        }
        Expr::Div(u, v) => {
            if contains_var(ctx, v, var) {
                return None;
            }
            let lf_u = linear_form(ctx, u, var)?;
            let coef = ctx.add(Expr::Div(lf_u.coef, v));
            let constant = ctx.add(Expr::Div(lf_u.constant, v));
            Some(LinearForm { coef, constant })
        }
        Expr::Pow(base, exp) => {
            if contains_var(ctx, exp, var) {
                return None;
            }
            if contains_var(ctx, base, var) {
                if let Expr::Number(n) = ctx.get(exp) {
                    if *n == num_rational::BigRational::from_integer(1.into()) {
                        return linear_form(ctx, base, var);
                    }
                }
                return None;
            }

            let zero = ctx.num(0);
            Some(LinearForm {
                coef: zero,
                constant: expr,
            })
        }
        Expr::Function(_, args) => {
            if args.iter().any(|&a| contains_var(ctx, a, var)) {
                None
            } else {
                let zero = ctx.num(0);
                Some(LinearForm {
                    coef: zero,
                    constant: expr,
                })
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_var_as_unit_linear_form() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let lf = linear_form(&mut ctx, x, "x").expect("x must be linear in x");
        assert!(
            matches!(ctx.get(lf.coef), Expr::Number(n) if *n == num_rational::BigRational::from_integer(1.into()))
        );
        assert!(
            matches!(ctx.get(lf.constant), Expr::Number(n) if *n == num_rational::BigRational::from_integer(0.into()))
        );
    }

    #[test]
    fn rejects_square_as_nonlinear() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let x2 = ctx.add(Expr::Pow(x, two));
        assert!(linear_form(&mut ctx, x2, "x").is_none());
    }

    #[test]
    fn keeps_coeff_and_constant_var_free() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let one = ctx.num(1);
        let x_plus_one = ctx.add(Expr::Add(x, one));
        let expr = ctx.add(Expr::Mul(y, x_plus_one)); // y*(x+1) = y*x + y
        let lf = linear_form(&mut ctx, expr, "x").expect("must be linear in x");
        assert!(!contains_var(&ctx, lf.coef, "x"));
        assert!(!contains_var(&ctx, lf.constant, "x"));
    }
}
