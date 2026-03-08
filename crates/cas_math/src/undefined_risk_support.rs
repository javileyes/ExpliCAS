//! Structural detection for undefined-risk subexpressions.
//!
//! This module scans expressions for divisions whose denominator is not proven
//! non-zero by a caller-provided oracle.

use cas_ast::{Context, Expr, ExprId};

/// Returns true when expression contains a division `a / b` where `b` is not
/// proven non-zero by `is_nonzero_proven`.
pub fn has_undefined_risk_with<F>(ctx: &Context, expr: ExprId, mut is_nonzero_proven: F) -> bool
where
    F: FnMut(&Context, ExprId) -> bool,
{
    let mut stack = vec![expr];
    while let Some(e) = stack.pop() {
        match ctx.get(e) {
            Expr::Div(num, den) => {
                if !is_nonzero_proven(ctx, *den) {
                    return true;
                }
                stack.push(*num);
                stack.push(*den);
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => {
                stack.push(*inner);
            }
            Expr::Function(_, args) => {
                for arg in args {
                    stack.push(*arg);
                }
            }
            Expr::Matrix { data, .. } => {
                for elem in data {
                    stack.push(*elem);
                }
            }
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::has_undefined_risk_with;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn detects_risk_when_denominator_not_proven() {
        let mut ctx = Context::new();
        let expr = parse("x/(y+1)", &mut ctx).expect("parse");
        let risk = has_undefined_risk_with(&ctx, expr, |_ctx, _den| false);
        assert!(risk);
    }

    #[test]
    fn no_risk_when_denominator_is_proven_nonzero() {
        let mut ctx = Context::new();
        let expr = parse("x/2", &mut ctx).expect("parse");
        let risk = has_undefined_risk_with(&ctx, expr, |_ctx, _den| true);
        assert!(!risk);
    }
}
