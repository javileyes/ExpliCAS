//! Source-side direct/reciprocal trig-affine integrand detection for calculus shortcuts.

use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;

pub(super) fn expr_contains_direct_trig_with_affine_arg(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(cas_ast::hold::unwrap_internal_hold(ctx, current)) {
            Expr::Function(fn_id, args)
                if args.len() == 1
                    && matches!(
                        ctx.builtin_of(*fn_id),
                        Some(BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Sec | BuiltinFn::Csc)
                    )
                    && Polynomial::from_expr(ctx, args[0], var_name)
                        .is_ok_and(|poly| poly.degree() == 1) =>
            {
                return true;
            }
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::Pow(left, right) => {
                stack.push(*left);
                stack.push(*right);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Number(_)
            | Expr::Constant(_)
            | Expr::Variable(_)
            | Expr::SessionRef(_)
            | Expr::Matrix { .. } => {}
        }
    }
    false
}
