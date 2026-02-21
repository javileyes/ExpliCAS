use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_predicates::is_zero_expr as is_zero;

/// Clean up solver step descriptions for readability.
pub fn cleanup_step_description(desc: &str) -> String {
    if desc.starts_with("Subtract -(") || desc.starts_with("Subtract -") {
        return "Move terms to one side".to_string();
    }
    if desc.starts_with("Add -(") || desc.starts_with("Add -") {
        return "Move terms to one side".to_string();
    }
    desc.to_string()
}

/// Normalize signs in an expression for didactic display.
pub fn normalize_expr_signs(ctx: &mut Context, expr: ExprId) -> ExprId {
    normalize_signs_recursive(ctx, expr)
}

fn normalize_signs_recursive(ctx: &mut Context, expr: ExprId) -> ExprId {
    let expr_data = ctx.get(expr).clone();

    match expr_data {
        Expr::Sub(lhs, rhs) => {
            let norm_lhs = normalize_signs_recursive(ctx, lhs);
            let norm_rhs = normalize_signs_recursive(ctx, rhs);

            if is_zero(ctx, norm_lhs) {
                if let Expr::Neg(inner) = ctx.get(norm_rhs).clone() {
                    return normalize_signs_recursive(ctx, inner);
                }
                return ctx.add(Expr::Neg(norm_rhs));
            }

            if let Expr::Neg(inner) = ctx.get(norm_rhs).clone() {
                return ctx.add(Expr::Add(norm_lhs, inner));
            }

            if norm_lhs != lhs || norm_rhs != rhs {
                ctx.add(Expr::Sub(norm_lhs, norm_rhs))
            } else {
                expr
            }
        }
        Expr::Neg(inner) => {
            let norm_inner = normalize_signs_recursive(ctx, inner);
            if let Expr::Neg(inner_inner) = ctx.get(norm_inner).clone() {
                return normalize_signs_recursive(ctx, inner_inner);
            }
            if norm_inner != inner {
                ctx.add(Expr::Neg(norm_inner))
            } else {
                expr
            }
        }
        Expr::Add(l, r) => {
            let nl = normalize_signs_recursive(ctx, l);
            let nr = normalize_signs_recursive(ctx, r);
            if nl != l || nr != r {
                ctx.add(Expr::Add(nl, nr))
            } else {
                expr
            }
        }
        Expr::Mul(l, r) => {
            let nl = normalize_signs_recursive(ctx, l);
            let nr = normalize_signs_recursive(ctx, r);
            if nl != l || nr != r {
                ctx.add(Expr::Mul(nl, nr))
            } else {
                expr
            }
        }
        Expr::Div(l, r) => {
            let nl = normalize_signs_recursive(ctx, l);
            let nr = normalize_signs_recursive(ctx, r);
            if nl != l || nr != r {
                ctx.add(Expr::Div(nl, nr))
            } else {
                expr
            }
        }
        Expr::Pow(base, exp) => {
            let nb = normalize_signs_recursive(ctx, base);
            let ne = normalize_signs_recursive(ctx, exp);
            if nb != base || ne != exp {
                ctx.add(Expr::Pow(nb, ne))
            } else {
                expr
            }
        }
        Expr::Function(name, args) => {
            let new_args: Vec<ExprId> = args
                .iter()
                .map(|&a| normalize_signs_recursive(ctx, a))
                .collect();
            if new_args != args {
                ctx.add(Expr::Function(name, new_args))
            } else {
                expr
            }
        }
        Expr::Hold(inner) => {
            let norm_inner = normalize_signs_recursive(ctx, inner);
            if norm_inner != inner {
                ctx.add(Expr::Hold(norm_inner))
            } else {
                expr
            }
        }
        Expr::Matrix { rows, cols, data } => {
            let new_data: Vec<ExprId> = data
                .iter()
                .map(|&d| normalize_signs_recursive(ctx, d))
                .collect();
            if new_data != data {
                ctx.add(Expr::Matrix {
                    rows,
                    cols,
                    data: new_data,
                })
            } else {
                expr
            }
        }
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => expr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_zero_minus_neg() {
        let mut ctx = Context::new();
        let t = ctx.var("t");
        let neg_t = ctx.add(Expr::Neg(t));
        let zero = ctx.num(0);
        let expr = ctx.add(Expr::Sub(zero, neg_t));

        let result = normalize_expr_signs(&mut ctx, expr);
        assert!(matches!(ctx.get(result), Expr::Variable(v) if ctx.sym_name(*v) == "t"));
    }

    #[test]
    fn normalize_sub_neg() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let neg_b = ctx.add(Expr::Neg(b));
        let expr = ctx.add(Expr::Sub(a, neg_b));

        let result = normalize_expr_signs(&mut ctx, expr);
        assert!(matches!(ctx.get(result), Expr::Add(_, _)));
    }

    #[test]
    fn normalize_double_neg() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let neg_x = ctx.add(Expr::Neg(x));
        let double_neg = ctx.add(Expr::Neg(neg_x));

        let result = normalize_expr_signs(&mut ctx, double_neg);
        assert!(matches!(ctx.get(result), Expr::Variable(v) if ctx.sym_name(*v) == "x"));
    }

    #[test]
    fn cleanup_subtract_negative() {
        assert_eq!(
            cleanup_step_description("Subtract -(x) from both sides"),
            "Move terms to one side"
        );
    }
}
