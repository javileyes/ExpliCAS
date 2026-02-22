use crate::isolation_utils::flip_inequality;
use cas_ast::{Equation, Expr};

/// Rewrite `lhs op rhs` by subtracting `rhs` on both sides:
/// `lhs + (-rhs) op rhs + (-rhs)`.
pub fn subtract_rhs_from_both_sides(
    ctx: &mut cas_ast::Context,
    equation: &Equation,
) -> Equation {
    let neg_rhs = ctx.add(Expr::Neg(equation.rhs));
    let lhs = ctx.add(Expr::Add(equation.lhs, neg_rhs));
    let rhs = ctx.add(Expr::Add(equation.rhs, neg_rhs));
    Equation {
        lhs,
        rhs,
        op: equation.op.clone(),
    }
}

/// Swap equation sides and flip inequality direction when needed.
pub fn swap_sides_with_inequality_flip(equation: &Equation) -> Equation {
    Equation {
        lhs: equation.rhs,
        rhs: equation.lhs,
        op: flip_inequality(equation.op.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::{Context, Expr, RelOp};

    #[test]
    fn subtract_rhs_from_both_sides_preserves_operator() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let eq = Equation {
            lhs: x,
            rhs: two,
            op: RelOp::Leq,
        };
        let rewritten = subtract_rhs_from_both_sides(&mut ctx, &eq);
        assert_eq!(rewritten.op, RelOp::Leq);
        assert!(matches!(ctx.get(rewritten.lhs), Expr::Add(_, _)));
        assert!(matches!(ctx.get(rewritten.rhs), Expr::Add(_, _)));
    }

    #[test]
    fn subtract_rhs_from_both_sides_builds_negated_rhs_once() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let eq = Equation {
            lhs: x,
            rhs: y,
            op: RelOp::Eq,
        };
        let rewritten = subtract_rhs_from_both_sides(&mut ctx, &eq);

        let lhs_neg = match ctx.get(rewritten.lhs) {
            Expr::Add(_, neg) => *neg,
            other => panic!("expected Add on lhs, got {:?}", other),
        };
        let rhs_neg = match ctx.get(rewritten.rhs) {
            Expr::Add(_, neg) => *neg,
            other => panic!("expected Add on rhs, got {:?}", other),
        };
        assert_eq!(lhs_neg, rhs_neg, "negated rhs should be shared node");
    }

    #[test]
    fn swap_sides_flips_strict_inequality_direction() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let eq = Equation {
            lhs: x,
            rhs: y,
            op: RelOp::Lt,
        };
        let swapped = swap_sides_with_inequality_flip(&eq);
        assert_eq!(swapped.lhs, y);
        assert_eq!(swapped.rhs, x);
        assert_eq!(swapped.op, RelOp::Gt);
    }

    #[test]
    fn swap_sides_keeps_equality_operator() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let eq = Equation {
            lhs: x,
            rhs: y,
            op: RelOp::Eq,
        };
        let swapped = swap_sides_with_inequality_flip(&eq);
        assert_eq!(swapped.op, RelOp::Eq);
    }
}
