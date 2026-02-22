use crate::isolation_utils::{apply_sign_flip, flip_inequality};
use cas_ast::{Equation, Expr, ExprId, RelOp};

/// Rewrite `lhs op rhs` by subtracting `rhs` on both sides:
/// `lhs + (-rhs) op rhs + (-rhs)`.
pub fn subtract_rhs_from_both_sides(ctx: &mut cas_ast::Context, equation: &Equation) -> Equation {
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

/// Rewrite `(-inner) op rhs` as `inner flip(op) -rhs`.
pub fn isolate_negated_lhs(
    ctx: &mut cas_ast::Context,
    inner: ExprId,
    rhs: ExprId,
    op: RelOp,
) -> Equation {
    let neg_rhs = ctx.add(Expr::Neg(rhs));
    Equation {
        lhs: inner,
        rhs: neg_rhs,
        op: flip_inequality(op),
    }
}

/// Rewrite `(kept + moved) op rhs` as `kept op (rhs - moved)`.
pub fn isolate_add_operand(
    ctx: &mut cas_ast::Context,
    kept: ExprId,
    moved: ExprId,
    rhs: ExprId,
    op: RelOp,
) -> Equation {
    let new_rhs = ctx.add(Expr::Sub(rhs, moved));
    Equation {
        lhs: kept,
        rhs: new_rhs,
        op,
    }
}

/// Rewrite `(minuend - subtrahend) op rhs` as `minuend op (rhs + subtrahend)`.
pub fn isolate_sub_minuend(
    ctx: &mut cas_ast::Context,
    minuend: ExprId,
    subtrahend: ExprId,
    rhs: ExprId,
    op: RelOp,
) -> Equation {
    let new_rhs = ctx.add(Expr::Add(rhs, subtrahend));
    Equation {
        lhs: minuend,
        rhs: new_rhs,
        op,
    }
}

/// Rewrite `(minuend - subtrahend) op rhs` as
/// `subtrahend flip(op) (minuend - rhs)`.
pub fn isolate_sub_subtrahend(
    ctx: &mut cas_ast::Context,
    minuend: ExprId,
    subtrahend: ExprId,
    rhs: ExprId,
    op: RelOp,
) -> Equation {
    let new_rhs = ctx.add(Expr::Sub(minuend, rhs));
    Equation {
        lhs: subtrahend,
        rhs: new_rhs,
        op: flip_inequality(op),
    }
}

/// Rewrite `(kept * moved) op rhs` as `kept op (rhs / moved)`,
/// flipping inequality when `moved` is known negative.
pub fn isolate_mul_factor(
    ctx: &mut cas_ast::Context,
    kept: ExprId,
    moved: ExprId,
    rhs: ExprId,
    op: RelOp,
    moved_is_negative: bool,
) -> Equation {
    let new_rhs = ctx.add(Expr::Div(rhs, moved));
    Equation {
        lhs: kept,
        rhs: new_rhs,
        op: apply_sign_flip(op, moved_is_negative),
    }
}

/// Rewrite `(numerator / denominator) op rhs` as
/// `numerator op (rhs * denominator)`, flipping inequality when
/// `denominator` is known negative.
pub fn isolate_div_numerator(
    ctx: &mut cas_ast::Context,
    numerator: ExprId,
    denominator: ExprId,
    rhs: ExprId,
    op: RelOp,
    denominator_is_negative: bool,
) -> Equation {
    let new_rhs = ctx.add(Expr::Mul(rhs, denominator));
    Equation {
        lhs: numerator,
        rhs: new_rhs,
        op: apply_sign_flip(op, denominator_is_negative),
    }
}

/// Rewrite `(numerator / denominator) op rhs` as
/// `denominator op (numerator / rhs)`.
pub fn isolate_div_denominator(
    ctx: &mut cas_ast::Context,
    denominator: ExprId,
    numerator: ExprId,
    rhs: ExprId,
    op: RelOp,
) -> Equation {
    let new_rhs = ctx.add(Expr::Div(numerator, rhs));
    Equation {
        lhs: denominator,
        rhs: new_rhs,
        op,
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

    #[test]
    fn isolate_negated_lhs_flips_inequality_and_negates_rhs() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let eq = isolate_negated_lhs(&mut ctx, x, y, RelOp::Lt);
        assert_eq!(eq.lhs, x);
        assert_eq!(eq.op, RelOp::Gt);
        assert!(matches!(ctx.get(eq.rhs), Expr::Neg(v) if *v == y));
    }

    #[test]
    fn isolate_negated_lhs_preserves_equality() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let eq = isolate_negated_lhs(&mut ctx, x, y, RelOp::Eq);
        assert_eq!(eq.op, RelOp::Eq);
    }

    #[test]
    fn isolate_add_operand_moves_term_to_rhs_as_subtraction() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let eq = isolate_add_operand(&mut ctx, a, b, c, RelOp::Eq);
        assert_eq!(eq.lhs, a);
        assert_eq!(eq.op, RelOp::Eq);
        assert!(matches!(ctx.get(eq.rhs), Expr::Sub(l, r) if *l == c && *r == b));
    }

    #[test]
    fn isolate_sub_minuend_moves_subtrahend_to_rhs_as_addition() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let eq = isolate_sub_minuend(&mut ctx, a, b, c, RelOp::Leq);
        assert_eq!(eq.lhs, a);
        assert_eq!(eq.op, RelOp::Leq);
        assert!(
            matches!(ctx.get(eq.rhs), Expr::Add(l, r) if (*l == c && *r == b) || (*l == b && *r == c))
        );
    }

    #[test]
    fn isolate_sub_subtrahend_flips_inequality() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let eq = isolate_sub_subtrahend(&mut ctx, a, b, c, RelOp::Lt);
        assert_eq!(eq.lhs, b);
        assert_eq!(eq.op, RelOp::Gt);
        assert!(matches!(ctx.get(eq.rhs), Expr::Sub(l, r) if *l == a && *r == c));
    }

    #[test]
    fn isolate_mul_factor_flips_inequality_for_negative_divisor() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let eq = isolate_mul_factor(&mut ctx, a, b, c, RelOp::Lt, true);
        assert_eq!(eq.lhs, a);
        assert_eq!(eq.op, RelOp::Gt);
        assert!(matches!(ctx.get(eq.rhs), Expr::Div(l, r) if *l == c && *r == b));
    }

    #[test]
    fn isolate_div_numerator_flips_inequality_for_negative_denominator() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let eq = isolate_div_numerator(&mut ctx, a, b, c, RelOp::Leq, true);
        assert_eq!(eq.lhs, a);
        assert_eq!(eq.op, RelOp::Geq);
        assert!(
            matches!(ctx.get(eq.rhs), Expr::Mul(l, r) if (*l == c && *r == b) || (*l == b && *r == c))
        );
    }

    #[test]
    fn isolate_div_denominator_builds_division_rhs() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let eq = isolate_div_denominator(&mut ctx, a, b, c, RelOp::Eq);
        assert_eq!(eq.lhs, a);
        assert_eq!(eq.op, RelOp::Eq);
        assert!(matches!(ctx.get(eq.rhs), Expr::Div(l, r) if *l == b && *r == c));
    }
}
