use cas_ast::{Context, Expr, ExprId};

/// Substitute a named variable with a value in an expression tree.
pub fn substitute_named_var(ctx: &mut Context, expr: ExprId, var: &str, value: ExprId) -> ExprId {
    let expr_data = ctx.get(expr).clone();

    match expr_data {
        Expr::Variable(sym_id) if ctx.sym_name(sym_id) == var => value,
        Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) => expr,

        Expr::Add(a, b) => {
            let a_sub = substitute_named_var(ctx, a, var, value);
            let b_sub = substitute_named_var(ctx, b, var, value);
            ctx.add(Expr::Add(a_sub, b_sub))
        }
        Expr::Sub(a, b) => {
            let a_sub = substitute_named_var(ctx, a, var, value);
            let b_sub = substitute_named_var(ctx, b, var, value);
            ctx.add(Expr::Sub(a_sub, b_sub))
        }
        Expr::Mul(a, b) => {
            let a_sub = substitute_named_var(ctx, a, var, value);
            let b_sub = substitute_named_var(ctx, b, var, value);
            ctx.add(Expr::Mul(a_sub, b_sub))
        }
        Expr::Div(a, b) => {
            let a_sub = substitute_named_var(ctx, a, var, value);
            let b_sub = substitute_named_var(ctx, b, var, value);
            ctx.add(Expr::Div(a_sub, b_sub))
        }
        Expr::Pow(a, b) => {
            let a_sub = substitute_named_var(ctx, a, var, value);
            let b_sub = substitute_named_var(ctx, b, var, value);
            ctx.add(Expr::Pow(a_sub, b_sub))
        }
        Expr::Neg(a) => {
            let a_sub = substitute_named_var(ctx, a, var, value);
            ctx.add(Expr::Neg(a_sub))
        }
        Expr::Function(name, args) => {
            let args_sub: Vec<_> = args
                .iter()
                .map(|&arg| substitute_named_var(ctx, arg, var, value))
                .collect();
            ctx.add(Expr::Function(name, args_sub))
        }
        Expr::Matrix { rows, cols, data } => {
            let data_sub: Vec<_> = data
                .iter()
                .map(|&elem| substitute_named_var(ctx, elem, var, value))
                .collect();
            ctx.add(Expr::Matrix {
                rows,
                cols,
                data: data_sub,
            })
        }
        Expr::Hold(inner) => {
            let inner_sub = substitute_named_var(ctx, inner, var, value);
            ctx.add(Expr::Hold(inner_sub))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitute_simple_named_var() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let one = ctx.num(1);
        let expr = ctx.add(Expr::Add(x, one));

        let replaced = substitute_named_var(&mut ctx, expr, "x", y);
        match ctx.get(replaced) {
            Expr::Add(lhs, rhs) => {
                let lhs_is_y =
                    matches!(ctx.get(*lhs), Expr::Variable(sym) if ctx.sym_name(*sym) == "y");
                let rhs_is_y =
                    matches!(ctx.get(*rhs), Expr::Variable(sym) if ctx.sym_name(*sym) == "y");
                let lhs_is_one = matches!(
                    ctx.get(*lhs),
                    Expr::Number(n) if *n == num_rational::BigRational::from_integer(1.into())
                );
                let rhs_is_one = matches!(
                    ctx.get(*rhs),
                    Expr::Number(n) if *n == num_rational::BigRational::from_integer(1.into())
                );
                assert!(
                    (lhs_is_y && rhs_is_one) || (rhs_is_y && lhs_is_one),
                    "expected Add(y, 1) in canonical order, got lhs={:?}, rhs={:?}",
                    ctx.get(*lhs),
                    ctx.get(*rhs)
                );
            }
            other => panic!("expected Add after substitution, got {other:?}"),
        }
    }
}
