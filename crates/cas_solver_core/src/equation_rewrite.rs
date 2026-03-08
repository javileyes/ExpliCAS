use crate::isolation_utils::{
    apply_sign_flip, denominator_sign_case_ops, flip_inequality, is_numeric_zero,
    isolated_denominator_variable_case_ops, SignCaseOps,
};
use crate::solution_set::pos_inf;
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

/// Classification for denominator-isolation rewrite choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivDenominatorIsolationKind {
    RhsZeroToInfinity,
    DivisionRewrite,
}

/// Rewrite denominator isolation with a safety guard for `rhs = 0`:
/// - if `rhs == 0`: `denominator op +inf`
/// - otherwise:     `denominator op numerator/rhs`
pub fn isolate_div_denominator_with_zero_rhs_guard(
    ctx: &mut cas_ast::Context,
    denominator: ExprId,
    numerator: ExprId,
    rhs: ExprId,
    op: RelOp,
) -> (Equation, DivDenominatorIsolationKind) {
    if is_numeric_zero(ctx, rhs) {
        (
            Equation {
                lhs: denominator,
                rhs: pos_inf(ctx),
                op,
            },
            DivDenominatorIsolationKind::RhsZeroToInfinity,
        )
    } else {
        (
            isolate_div_denominator(ctx, denominator, numerator, rhs, op),
            DivDenominatorIsolationKind::DivisionRewrite,
        )
    }
}

/// Build both branch equations for absolute-value isolation:
/// `|arg| op rhs`  ->  `(arg op rhs)` and `(arg flip(op) -rhs)`.
pub fn isolate_abs_branches(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
    rhs: ExprId,
    op: RelOp,
) -> (Equation, Equation) {
    let positive = Equation {
        lhs: arg,
        rhs,
        op: op.clone(),
    };
    let neg_rhs = ctx.add(Expr::Neg(rhs));
    let negative = Equation {
        lhs: arg,
        rhs: neg_rhs,
        op: flip_inequality(op),
    };
    (positive, negative)
}

/// Pair of equations for positive/negative sign branches.
#[derive(Debug, Clone, PartialEq)]
pub struct SignSplitEquations {
    pub positive: Equation,
    pub negative: Equation,
}

/// Clone split operators/LHS and replace both branch RHS with a shared RHS.
pub fn with_shared_rhs(split: &SignSplitEquations, rhs: ExprId) -> SignSplitEquations {
    SignSplitEquations {
        positive: Equation {
            lhs: split.positive.lhs,
            rhs,
            op: split.positive.op.clone(),
        },
        negative: Equation {
            lhs: split.negative.lhs,
            rhs,
            op: split.negative.op.clone(),
        },
    }
}

/// Build inequality split for `numerator / denominator op rhs`:
/// - positive branch: `numerator op (rhs * denominator)`
/// - negative branch: `numerator flip(op) (rhs * denominator)`
///
/// and sign-domain equations `denominator > 0`, `denominator < 0`.
pub fn build_division_denominator_sign_split(
    ctx: &mut cas_ast::Context,
    numerator: ExprId,
    denominator: ExprId,
    rhs: ExprId,
    op: RelOp,
) -> Option<(SignSplitEquations, Equation, Equation)> {
    let (op_pos, op_neg) = denominator_sign_case_ops(op)?;
    let transformed_rhs = ctx.add(Expr::Mul(rhs, denominator));
    let branches = SignSplitEquations {
        positive: Equation {
            lhs: numerator,
            rhs: transformed_rhs,
            op: op_pos,
        },
        negative: Equation {
            lhs: numerator,
            rhs: transformed_rhs,
            op: op_neg,
        },
    };
    let domain_pos = build_sign_domain_equation(ctx, denominator, true);
    let domain_neg = build_sign_domain_equation(ctx, denominator, false);
    Some((branches, domain_pos, domain_neg))
}

/// Build inequality split for an already isolated denominator equation:
/// `lhs op rhs` with `lhs` being the denominator variable.
///
/// Returns:
/// - positive branch with op for `lhs > 0`
/// - negative branch with op for `lhs < 0`
pub fn build_isolated_denominator_sign_split(
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
) -> Option<SignSplitEquations> {
    let (op_pos, op_neg) = isolated_denominator_variable_case_ops(op)?;
    Some(SignSplitEquations {
        positive: Equation {
            lhs,
            rhs,
            op: op_pos,
        },
        negative: Equation {
            lhs,
            rhs,
            op: op_neg,
        },
    })
}

/// Build one zero-product sign case:
/// `left case.left 0` and `right case.right 0`.
pub fn build_product_zero_sign_case(
    ctx: &mut cas_ast::Context,
    left: ExprId,
    right: ExprId,
    case: &SignCaseOps,
) -> (Equation, Equation) {
    let zero = ctx.num(0);
    (
        Equation {
            lhs: left,
            rhs: zero,
            op: case.left.clone(),
        },
        Equation {
            lhs: right,
            rhs: zero,
            op: case.right.clone(),
        },
    )
}

/// Build a sign-domain equation: `expr > 0` when `positive` else `expr < 0`.
pub fn build_sign_domain_equation(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    positive: bool,
) -> Equation {
    let zero = ctx.num(0);
    Equation {
        lhs: expr,
        rhs: zero,
        op: if positive { RelOp::Gt } else { RelOp::Lt },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::{Constant, Context, Expr, RelOp};

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

    #[test]
    fn isolate_div_denominator_with_zero_rhs_guard_returns_infinity() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let zero = ctx.num(0);

        let (eq, kind) =
            isolate_div_denominator_with_zero_rhs_guard(&mut ctx, a, b, zero, RelOp::Eq);
        assert_eq!(kind, DivDenominatorIsolationKind::RhsZeroToInfinity);
        assert_eq!(eq.lhs, a);
        assert_eq!(eq.op, RelOp::Eq);
        assert!(matches!(
            ctx.get(eq.rhs),
            Expr::Constant(c) if matches!(c, Constant::Infinity)
        ));
    }

    #[test]
    fn isolate_div_denominator_with_zero_rhs_guard_rewrites_nonzero_rhs() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");

        let (eq, kind) = isolate_div_denominator_with_zero_rhs_guard(&mut ctx, a, b, c, RelOp::Eq);
        assert_eq!(kind, DivDenominatorIsolationKind::DivisionRewrite);
        assert_eq!(eq.lhs, a);
        assert_eq!(eq.op, RelOp::Eq);
        assert!(matches!(ctx.get(eq.rhs), Expr::Div(l, r) if *l == b && *r == c));
    }

    #[test]
    fn isolate_abs_branches_builds_positive_and_flipped_negative_cases() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let (pos, neg) = isolate_abs_branches(&mut ctx, a, b, RelOp::Lt);

        assert_eq!(pos.lhs, a);
        assert_eq!(pos.rhs, b);
        assert_eq!(pos.op, RelOp::Lt);

        assert_eq!(neg.lhs, a);
        assert_eq!(neg.op, RelOp::Gt);
        assert!(matches!(ctx.get(neg.rhs), Expr::Neg(v) if *v == b));
    }

    #[test]
    fn build_product_zero_sign_case_uses_case_ops() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let case = SignCaseOps {
            left: RelOp::Geq,
            right: RelOp::Leq,
        };
        let (eq_a, eq_b) = build_product_zero_sign_case(&mut ctx, a, b, &case);
        assert_eq!(eq_a.lhs, a);
        assert_eq!(eq_b.lhs, b);
        assert_eq!(eq_a.op, RelOp::Geq);
        assert_eq!(eq_b.op, RelOp::Leq);
        assert!(matches!(ctx.get(eq_a.rhs), Expr::Number(_)));
        assert!(matches!(ctx.get(eq_b.rhs), Expr::Number(_)));
    }

    #[test]
    fn build_sign_domain_equation_positive_and_negative() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let pos = build_sign_domain_equation(&mut ctx, x, true);
        let neg = build_sign_domain_equation(&mut ctx, x, false);
        assert_eq!(pos.op, RelOp::Gt);
        assert_eq!(neg.op, RelOp::Lt);
        assert!(matches!(ctx.get(pos.rhs), Expr::Number(_)));
        assert!(matches!(ctx.get(neg.rhs), Expr::Number(_)));
    }

    #[test]
    fn build_division_denominator_sign_split_builds_two_ops_and_domains() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");

        let (branches, dpos, dneg) =
            build_division_denominator_sign_split(&mut ctx, a, b, c, RelOp::Lt)
                .expect("inequality split should be available");

        assert_eq!(branches.positive.lhs, a);
        assert_eq!(branches.negative.lhs, a);
        assert_eq!(branches.positive.op, RelOp::Lt);
        assert_eq!(branches.negative.op, RelOp::Gt);
        assert_eq!(branches.positive.rhs, branches.negative.rhs);
        assert_eq!(dpos.lhs, b);
        assert_eq!(dneg.lhs, b);
        assert_eq!(dpos.op, RelOp::Gt);
        assert_eq!(dneg.op, RelOp::Lt);
    }

    #[test]
    fn build_isolated_denominator_sign_split_uses_isolated_op_pair() {
        let mut ctx = Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.var("r");
        let split = build_isolated_denominator_sign_split(lhs, rhs, RelOp::Lt)
            .expect("inequality split should be available");
        assert_eq!(split.positive.lhs, lhs);
        assert_eq!(split.negative.lhs, lhs);
        assert_eq!(split.positive.rhs, rhs);
        assert_eq!(split.negative.rhs, rhs);
        // For isolated denominator case, pair is intentionally swapped.
        assert_eq!(split.positive.op, RelOp::Gt);
        assert_eq!(split.negative.op, RelOp::Lt);
    }

    #[test]
    fn with_shared_rhs_replaces_branch_rhs_preserving_lhs_and_ops() {
        let mut ctx = Context::new();
        let lhs = ctx.var("x");
        let rhs_a = ctx.var("a");
        let rhs_b = ctx.var("b");
        let split = build_isolated_denominator_sign_split(lhs, rhs_a, RelOp::Leq)
            .expect("inequality split should be available");
        let rewritten = with_shared_rhs(&split, rhs_b);

        assert_eq!(rewritten.positive.lhs, split.positive.lhs);
        assert_eq!(rewritten.negative.lhs, split.negative.lhs);
        assert_eq!(rewritten.positive.op, split.positive.op);
        assert_eq!(rewritten.negative.op, split.negative.op);
        assert_eq!(rewritten.positive.rhs, rhs_b);
        assert_eq!(rewritten.negative.rhs, rhs_b);
    }
}
