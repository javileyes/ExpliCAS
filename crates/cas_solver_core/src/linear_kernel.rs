use crate::isolation_utils::contains_var;
use crate::linear_form::linear_form;
use cas_ast::{Context, Expr, ExprId};

/// Pure linear solve kernel extracted from solver orchestration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearSolveKernel {
    /// Coefficient in `coef * var + constant = 0`
    pub coef: ExprId,
    /// Constant term in `coef * var + constant = 0`
    pub constant: ExprId,
    /// Raw symbolic solution `-constant / coef`
    pub solution: ExprId,
}

/// Derive linear solving primitives for `lhs = rhs`.
///
/// Returns `None` when the equation is non-linear in `var`.
pub fn derive_linear_solve_kernel(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
    var: &str,
) -> Option<LinearSolveKernel> {
    let expr = ctx.add(Expr::Sub(lhs, rhs));
    let lf = linear_form(ctx, expr, var)?;

    // Safety check: coefficient should be var-free.
    if contains_var(ctx, lf.coef, var) {
        return None;
    }

    let neg_constant = ctx.add(Expr::Neg(lf.constant));
    let solution = ctx.add(Expr::Div(neg_constant, lf.coef));
    Some(LinearSolveKernel {
        coef: lf.coef,
        constant: lf.constant,
        solution,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_kernel_for_linear_equation() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_plus_one = ctx.add(Expr::Add(x, one));
        let rhs = ctx.num(0);

        let kernel =
            derive_linear_solve_kernel(&mut ctx, x_plus_one, rhs, "x").expect("must be linear");
        assert!(!contains_var(&ctx, kernel.coef, "x"));
    }

    #[test]
    fn rejects_nonlinear_equation() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let lhs = ctx.add(Expr::Pow(x, two));
        let rhs = ctx.num(0);
        assert!(derive_linear_solve_kernel(&mut ctx, lhs, rhs, "x").is_none());
    }
}
