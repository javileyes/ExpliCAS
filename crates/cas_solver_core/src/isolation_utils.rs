use crate::solution_set::{intersect_solution_sets, union_solution_sets};
use cas_ast::{Context, Expr, ExprId, RelOp, SolutionSet};
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

/// Combine the two branch solution sets generated from `|A| op B`.
///
/// For equalities/greater-than forms both branches are alternatives (union).
/// For less-than forms both constraints must hold simultaneously (intersection).
pub fn combine_abs_branch_sets(
    ctx: &Context,
    op: RelOp,
    positive_branch: SolutionSet,
    negative_branch: SolutionSet,
) -> SolutionSet {
    match op {
        RelOp::Eq | RelOp::Neq | RelOp::Gt | RelOp::Geq => {
            union_solution_sets(ctx, positive_branch, negative_branch)
        }
        RelOp::Lt | RelOp::Leq => intersect_solution_sets(ctx, positive_branch, negative_branch),
    }
}

/// Relational operators to apply to each factor in a product-sign split case.
#[derive(Debug, Clone, PartialEq)]
pub struct SignCaseOps {
    pub left: RelOp,
    pub right: RelOp,
}

/// Build the two sign cases for product inequalities with zero RHS:
/// `A*B op 0`.
///
/// Returns `None` for non-inequality operators.
pub fn product_zero_inequality_cases(op: RelOp) -> Option<(SignCaseOps, SignCaseOps)> {
    match op {
        RelOp::Gt => Some((
            SignCaseOps {
                left: RelOp::Gt,
                right: RelOp::Gt,
            },
            SignCaseOps {
                left: RelOp::Lt,
                right: RelOp::Lt,
            },
        )),
        RelOp::Geq => Some((
            SignCaseOps {
                left: RelOp::Geq,
                right: RelOp::Geq,
            },
            SignCaseOps {
                left: RelOp::Leq,
                right: RelOp::Leq,
            },
        )),
        RelOp::Lt => Some((
            SignCaseOps {
                left: RelOp::Gt,
                right: RelOp::Lt,
            },
            SignCaseOps {
                left: RelOp::Lt,
                right: RelOp::Gt,
            },
        )),
        RelOp::Leq => Some((
            SignCaseOps {
                left: RelOp::Geq,
                right: RelOp::Leq,
            },
            SignCaseOps {
                left: RelOp::Leq,
                right: RelOp::Geq,
            },
        )),
        RelOp::Eq | RelOp::Neq => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::{BoundType, Interval};

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

    #[test]
    fn test_combine_abs_branch_sets_union_for_eq() {
        let mut ctx = Context::new();
        let i1 = Interval {
            min: ctx.num(0),
            min_type: BoundType::Closed,
            max: ctx.num(1),
            max_type: BoundType::Closed,
        };
        let i2 = Interval {
            min: ctx.num(2),
            min_type: BoundType::Closed,
            max: ctx.num(3),
            max_type: BoundType::Closed,
        };

        let set = combine_abs_branch_sets(
            &ctx,
            RelOp::Eq,
            SolutionSet::Continuous(i1),
            SolutionSet::Continuous(i2),
        );
        assert!(matches!(set, SolutionSet::Union(v) if v.len() == 2));
    }

    #[test]
    fn test_combine_abs_branch_sets_intersection_for_lt() {
        let mut ctx = Context::new();
        let i1 = Interval {
            min: ctx.num(0),
            min_type: BoundType::Closed,
            max: ctx.num(2),
            max_type: BoundType::Closed,
        };
        let i2 = Interval {
            min: ctx.num(1),
            min_type: BoundType::Closed,
            max: ctx.num(3),
            max_type: BoundType::Closed,
        };

        let set = combine_abs_branch_sets(
            &ctx,
            RelOp::Lt,
            SolutionSet::Continuous(i1),
            SolutionSet::Continuous(i2),
        );
        match set {
            SolutionSet::Continuous(i) => {
                assert_eq!(i.min, ctx.num(1));
                assert_eq!(i.max, ctx.num(2));
            }
            other => panic!("Expected Continuous intersection, got {:?}", other),
        }
    }

    #[test]
    fn test_product_zero_inequality_cases_gt() {
        let (c1, c2) = product_zero_inequality_cases(RelOp::Gt).expect("expected cases");
        assert_eq!(
            c1,
            SignCaseOps {
                left: RelOp::Gt,
                right: RelOp::Gt
            }
        );
        assert_eq!(
            c2,
            SignCaseOps {
                left: RelOp::Lt,
                right: RelOp::Lt
            }
        );
    }

    #[test]
    fn test_product_zero_inequality_cases_leq() {
        let (c1, c2) = product_zero_inequality_cases(RelOp::Leq).expect("expected cases");
        assert_eq!(
            c1,
            SignCaseOps {
                left: RelOp::Geq,
                right: RelOp::Leq
            }
        );
        assert_eq!(
            c2,
            SignCaseOps {
                left: RelOp::Leq,
                right: RelOp::Geq
            }
        );
    }

    #[test]
    fn test_product_zero_inequality_cases_eq_none() {
        assert!(product_zero_inequality_cases(RelOp::Eq).is_none());
        assert!(product_zero_inequality_cases(RelOp::Neq).is_none());
    }
}
