use cas_ast::{Context, Expr, ExprId, RelOp};
use std::cmp::Ordering;

/// Create a residual solve expression: solve(__eq__(lhs, rhs), var)
/// Used when solver can't justify a step but wants graceful degradation.
pub fn mk_residual_solve(ctx: &mut Context, lhs: ExprId, rhs: ExprId, var: &str) -> ExprId {
    let eq_expr = cas_ast::eq::wrap_eq(ctx, lhs, rhs);
    let var_expr = ctx.var(var);
    ctx.call("solve", vec![eq_expr, var_expr])
}

/// Check whether an expression contains a specific named variable.
pub fn contains_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    match ctx.get(expr) {
        Expr::Variable(sym_id) => ctx.sym_name(*sym_id) == var,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            contains_var(ctx, *l, var) || contains_var(ctx, *r, var)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => contains_var(ctx, *inner, var),
        Expr::Function(_, args) => args.iter().any(|&arg| contains_var(ctx, arg, var)),
        Expr::Matrix { data, .. } => data.iter().any(|&elem| contains_var(ctx, elem, var)),
        Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) => false,
    }
}

/// Check if an expression is known to be negative.
///
/// Recursively analyzes Mul products using XOR logic:
/// `(-a) * b` is negative, `(-a) * (-b)` is positive.
pub fn is_known_negative(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => *n < num_rational::BigRational::from_integer(0.into()),
        Expr::Neg(_) => true,
        Expr::Mul(l, r) => is_known_negative(ctx, *l) ^ is_known_negative(ctx, *r),
        _ => false,
    }
}

/// Attempt to recompose a^e / b^e -> (a/b)^e when both powers have the same exponent.
pub fn try_recompose_pow_quotient(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let expr_data = ctx.get(expr).clone();
    if let Expr::Div(num, den) = expr_data {
        let num_data = ctx.get(num).clone();
        let den_data = ctx.get(den).clone();
        if let (Expr::Pow(a, e1), Expr::Pow(b, e2)) = (num_data, den_data) {
            if cas_ast::ordering::compare_expr(ctx, e1, e2) == Ordering::Equal {
                let new_base = ctx.add(Expr::Div(a, b));
                return Some(ctx.add(Expr::Pow(new_base, e1)));
            }
        }
    }
    None
}

/// Flip inequality direction under multiplication/division by a negative value.
pub fn flip_inequality(op: RelOp) -> RelOp {
    match op {
        RelOp::Eq => RelOp::Eq,
        RelOp::Neq => RelOp::Neq,
        RelOp::Lt => RelOp::Gt,
        RelOp::Gt => RelOp::Lt,
        RelOp::Leq => RelOp::Geq,
        RelOp::Geq => RelOp::Leq,
    }
}

/// Check if expr is `1/var` pattern (simple reciprocal of target variable).
pub fn is_simple_reciprocal(ctx: &Context, expr: ExprId, var: &str) -> bool {
    if let Expr::Div(num, denom) = ctx.get(expr) {
        let is_one = matches!(
            ctx.get(*num),
            Expr::Number(n) if *n == num_rational::BigRational::from_integer(1.into())
        );
        let is_var =
            matches!(ctx.get(*denom), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var);
        is_one && is_var
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_simple_reciprocal() {
        let mut ctx = Context::new();
        let r = ctx.var("R");
        let one = ctx.num(1);
        let reciprocal = ctx.add(Expr::Div(one, r));

        assert!(is_simple_reciprocal(&ctx, reciprocal, "R"));
        assert!(!is_simple_reciprocal(&ctx, reciprocal, "X"));
        assert!(!is_simple_reciprocal(&ctx, r, "R"));
    }
}
