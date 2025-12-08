use crate::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use std::cmp::Ordering;

/// Canonicalize an expression by recursively ordering commutative operations.
/// This ensures that semantically equivalent expressions have identical structure.
///
/// Examples:
/// - `b*a` → `a*b` (if `a` < `b`)
/// - `y+x` → `x+y` (if `x` < `y`)
pub fn canonicalize_expression(ctx: &mut Context, expr: ExprId) -> ExprId {
    let expr_data = ctx.get(expr).clone();

    match expr_data {
        // Commutative binary operations: ensure left < right
        Expr::Add(l, r) | Expr::Mul(l, r) => {
            // Recursively canonicalize operands first
            let l_canon = canonicalize_expression(ctx, l);
            let r_canon = canonicalize_expression(ctx, r);

            // Order them canonically
            let cmp = compare_expr(ctx, l_canon, r_canon);

            let (left, right) = if cmp == Ordering::Greater {
                // Import DisplayExpr for logging
                #[cfg(test)] // Only enable logging in test builds or if explicitly enabled
                use cas_ast::DisplayExpr;

                #[cfg(test)]
                eprintln!("CANON: Swapping operands in {:?}", expr_data);
                #[cfg(test)]
                eprintln!(
                    "  Left:  {} (ExprId {:?})",
                    DisplayExpr {
                        context: ctx,
                        id: l_canon
                    },
                    l_canon.0
                );
                #[cfg(test)]
                eprintln!(
                    "  Right: {} (ExprId {:?})",
                    DisplayExpr {
                        context: ctx,
                        id: r_canon
                    },
                    r_canon.0
                );
                #[cfg(test)]
                eprintln!("  compare_expr returned: {:?}", cmp);
                (r_canon, l_canon) // Swap to ensure left < right
            } else {
                (l_canon, r_canon)
            };

            // Reconstruct with canonical order
            match expr_data {
                Expr::Add(_, _) => ctx.add(Expr::Add(left, right)),
                Expr::Mul(_, _) => ctx.add(Expr::Mul(left, right)),
                _ => unreachable!(),
            }
        }

        // Non-commutative binary operations: just recurse
        Expr::Sub(l, r) => {
            let l_canon = canonicalize_expression(ctx, l);
            let r_canon = canonicalize_expression(ctx, r);
            ctx.add(Expr::Sub(l_canon, r_canon))
        }

        Expr::Div(l, r) => {
            let l_canon = canonicalize_expression(ctx, l);
            let r_canon = canonicalize_expression(ctx, r);
            ctx.add(Expr::Div(l_canon, r_canon))
        }

        Expr::Pow(b, e) => {
            let b_canon = canonicalize_expression(ctx, b);
            let e_canon = canonicalize_expression(ctx, e);
            ctx.add(Expr::Pow(b_canon, e_canon))
        }

        // Unary operations: recurse
        Expr::Neg(e) => {
            let e_canon = canonicalize_expression(ctx, e);
            ctx.add(Expr::Neg(e_canon))
        }

        // Functions: recurse on arguments
        Expr::Function(name, args) => {
            let args_canon: Vec<ExprId> = args
                .into_iter()
                .map(|arg| canonicalize_expression(ctx, arg))
                .collect();
            ctx.add(Expr::Function(name, args_canon))
        }

        // Matrix: recurse on data
        Expr::Matrix { rows, cols, data } => {
            let data_canon: Vec<ExprId> = data
                .into_iter()
                .map(|elem| canonicalize_expression(ctx, elem))
                .collect();
            ctx.add(Expr::Matrix {
                rows,
                cols,
                data: data_canon,
            })
        }

        // Atoms: already canonical
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) => expr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::{Constant, DisplayExpr};
    use num_bigint::BigInt;
    use num_rational::BigRational;

    #[test]
    fn test_commutative_mul() {
        let mut ctx = Context::new();
        let x = ctx.add(Expr::Variable("x".to_string()));
        let y = ctx.add(Expr::Variable("y".to_string()));

        // y*x should canonicalize to x*y (since "x" < "y" alphabetically)
        let yx = ctx.add(Expr::Mul(y, x));
        let canonical = canonicalize_expression(&mut ctx, yx);

        let display = DisplayExpr {
            context: &ctx,
            id: canonical,
        };
        assert_eq!(display.to_string(), "x * y");
    }

    #[test]
    fn test_commutative_add() {
        let mut ctx = Context::new();
        let two = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(2))));
        let one = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(1))));

        // 2+1 should canonicalize to 1+2 (since 1 < 2)
        let add = ctx.add(Expr::Add(two, one));
        let canonical = canonicalize_expression(&mut ctx, add);

        let display = DisplayExpr {
            context: &ctx,
            id: canonical,
        };
        assert_eq!(display.to_string(), "1 + 2");
    }

    #[test]
    fn test_nested_canonicalization() {
        let mut ctx = Context::new();
        let a = ctx.add(Expr::Variable("a".to_string()));
        let b = ctx.add(Expr::Variable("b".to_string()));
        let c = ctx.add(Expr::Variable("c".to_string()));
        let d = ctx.add(Expr::Variable("d".to_string()));

        // (d+c) * (b+a) → (c+d) * (a+b)
        let dc = ctx.add(Expr::Add(d, c));
        let ba = ctx.add(Expr::Add(b, a));
        let expr = ctx.add(Expr::Mul(dc, ba));

        let canonical = canonicalize_expression(&mut ctx, expr);
        let display = DisplayExpr {
            context: &ctx,
            id: canonical,
        };
        assert_eq!(display.to_string(), "(a + b) * (c + d)");
    }

    #[test]
    fn test_non_commutative_preserved() {
        let mut ctx = Context::new();
        let a = ctx.add(Expr::Variable("a".to_string()));
        let b = ctx.add(Expr::Variable("b".to_string()));

        // b-a should NOT become a-b (subtraction is not commutative)
        let sub = ctx.add(Expr::Sub(b, a));
        let canonical = canonicalize_expression(&mut ctx, sub);

        let display = DisplayExpr {
            context: &ctx,
            id: canonical,
        };
        assert_eq!(display.to_string(), "b - a");
    }
}
