//! Solver entry helpers shared by runtime crates.
//!
//! Keeps equation-shape extraction (`expr` vs `lhs op rhs`) outside runtime
//! orchestrators so the solve boundary can stay thin.

use cas_ast::{BuiltinFn, Context, Equation, Expr, ExprId, RelOp};

/// Build the equation to solve from an input expression.
///
/// - If `expr` is a relation builtin (`Equal`, `Less`, `Greater`), use it.
/// - Otherwise fallback to `expr = 0`.
pub fn equation_from_expr_or_zero(ctx: &mut Context, expr: ExprId) -> Equation {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Equal) && args.len() == 2 =>
        {
            Equation {
                lhs: args[0],
                rhs: args[1],
                op: RelOp::Eq,
            }
        }
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Less) && args.len() == 2 =>
        {
            Equation {
                lhs: args[0],
                rhs: args[1],
                op: RelOp::Lt,
            }
        }
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Greater) && args.len() == 2 =>
        {
            Equation {
                lhs: args[0],
                rhs: args[1],
                op: RelOp::Gt,
            }
        }
        _ => Equation {
            lhs: expr,
            rhs: ctx.num(0),
            op: RelOp::Eq,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::equation_from_expr_or_zero;
    use cas_ast::{BuiltinFn, Equation, RelOp};
    use num_traits::Zero;

    #[test]
    fn equation_from_expr_or_zero_uses_relation_builtin_when_present() {
        let mut ctx = cas_ast::Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.num(2);
        let expr = ctx.call_builtin(BuiltinFn::Less, vec![lhs, rhs]);
        let eq = equation_from_expr_or_zero(&mut ctx, expr);
        let expected = Equation {
            lhs,
            rhs,
            op: RelOp::Lt,
        };
        assert_eq!(eq, expected);
    }

    #[test]
    fn equation_from_expr_or_zero_falls_back_to_expr_equals_zero() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let expr = ctx.add_raw(cas_ast::Expr::Add(x, one));
        let eq = equation_from_expr_or_zero(&mut ctx, expr);
        assert_eq!(eq.lhs, expr);
        assert_eq!(eq.op, RelOp::Eq);
        assert!(
            matches!(ctx.get(eq.rhs), cas_ast::Expr::Number(n) if n.is_zero()),
            "fallback rhs should be numeric zero"
        );
    }
}
