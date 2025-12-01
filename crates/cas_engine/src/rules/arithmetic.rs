use crate::rule::Rewrite;
use crate::define_rule;
use cas_ast::Expr;
use num_traits::{Zero, One};

define_rule!(
    AddZeroRule,
    "Identity Property of Addition",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Add(lhs, rhs) = expr_data {
            if let Expr::Number(n) = ctx.get(rhs) {
                if n.is_zero() {
                    return Some(Rewrite {
                        new_expr: lhs,
                        description: "x + 0 = x".to_string(),
                    });
                }
            }
            if let Expr::Number(n) = ctx.get(lhs) {
                if n.is_zero() {
                    return Some(Rewrite {
                        new_expr: rhs,
                        description: "0 + x = x".to_string(),
                    });
                }
            }
        }
        None
    }
);

define_rule!(
    MulOneRule,
    "Identity Property of Multiplication",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Mul(lhs, rhs) = expr_data {
            if let Expr::Number(n) = ctx.get(rhs) {
                if n.is_one() {
                    return Some(Rewrite {
                        new_expr: lhs,
                        description: "x * 1 = x".to_string(),
                    });
                }
            }
            if let Expr::Number(n) = ctx.get(lhs) {
                if n.is_one() {
                    return Some(Rewrite {
                        new_expr: rhs,
                        description: "1 * x = x".to_string(),
                    });
                }
            }
        }
        None
    }
);

define_rule!(
    MulZeroRule,
    "Zero Property of Multiplication",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Mul(lhs, rhs) = expr_data {
            if let Expr::Number(n) = ctx.get(rhs) {
                if n.is_zero() {
                    let zero = ctx.num(0);
                    return Some(Rewrite {
                        new_expr: zero,
                        description: "x * 0 = 0".to_string(),
                    });
                }
            }
            if let Expr::Number(n) = ctx.get(lhs) {
                if n.is_zero() {
                    let zero = ctx.num(0);
                    return Some(Rewrite {
                        new_expr: zero,
                        description: "0 * x = 0".to_string(),
                    });
                }
            }
        }
        None
    }
);

define_rule!(
    DivZeroRule,
    "Zero Property of Division",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Div(lhs, rhs) = expr_data {
            if let Expr::Number(n) = ctx.get(lhs) {
                if n.is_zero() {
                    // Check if denominator is zero?
                    // Ideally yes, but for symbolic simplification we often assume non-zero.
                    // If denominator is explicitly zero, CombineConstantsRule handles it (or we can check).
                    if let Expr::Number(d) = ctx.get(rhs) {
                        if d.is_zero() {
                            return None; // Undefined
                        }
                    }
                    
                    let zero = ctx.num(0);
                    return Some(Rewrite {
                        new_expr: zero,
                        description: "0 / x = 0".to_string(),
                    });
                }
            }
        }
        None
    }
);

define_rule!(
    CombineConstantsRule,
    "Combine Constants",
    |ctx, expr| {
        // We need to clone data to avoid borrowing ctx while mutating it later
        let expr_data = ctx.get(expr).clone();
        match expr_data {
            Expr::Add(lhs, rhs) => {
                let lhs_data = ctx.get(lhs).clone();
                let rhs_data = ctx.get(rhs).clone();
                if let (Expr::Number(n1), Expr::Number(n2)) = (&lhs_data, &rhs_data) {
                    let sum = n1 + n2;
                    let new_expr = ctx.add(Expr::Number(sum.clone()));
                    return Some(Rewrite {
                        new_expr,
                        description: format!("{} + {} = {}", n1, n2, sum),
                    });
                }
                // Handle nested: c1 + (c2 + x) -> (c1+c2) + x
                if let Expr::Number(n1) = lhs_data {
                    if let Expr::Add(rl, rr) = rhs_data {
                        let rl_data = ctx.get(rl).clone();
                        if let Expr::Number(n2) = rl_data {
                            let sum = &n1 + &n2;
                            let sum_expr = ctx.add(Expr::Number(sum.clone()));
                            let new_expr = ctx.add(Expr::Add(sum_expr, rr));
                            return Some(Rewrite {
                                new_expr,
                                description: format!("Combine nested constants: {} + {}", n1, n2),
                            });
                        }
                    }
                }
            }
            Expr::Mul(lhs, rhs) => {
                let lhs_data = ctx.get(lhs).clone();
                let rhs_data = ctx.get(rhs).clone();
                if let (Expr::Number(n1), Expr::Number(n2)) = (&lhs_data, &rhs_data) {
                    let prod = n1 * n2;
                    let new_expr = ctx.add(Expr::Number(prod.clone()));
                    return Some(Rewrite {
                        new_expr,
                        description: format!("{} * {} = {}", n1, n2, prod),
                    });
                }
                // Handle nested: c1 * (c2 * x) -> (c1*c2) * x
                if let Expr::Number(ref n1) = lhs_data {
                    if let Expr::Mul(rl, rr) = rhs_data {
                        let rl_data = ctx.get(rl).clone();
                        if let Expr::Number(n2) = rl_data {
                            let prod = n1 * &n2;
                            let prod_expr = ctx.add(Expr::Number(prod.clone()));
                            let new_expr = ctx.add(Expr::Mul(prod_expr, rr));
                            return Some(Rewrite {
                                new_expr,
                                description: format!("Combine nested constants: {} * {}", n1, n2),
                            });
                        }
                    }
                }

                // Handle c1 * (x / c2) -> (c1/c2) * x
                if let Expr::Number(ref n1) = lhs_data {
                    if let Expr::Div(num, den) = rhs_data {
                        let den_data = ctx.get(den).clone();
                        if let Expr::Number(n2) = den_data {
                            if !n2.is_zero() {
                                let ratio = n1 / &n2;
                                let ratio_expr = ctx.add(Expr::Number(ratio));
                                let new_expr = ctx.add(Expr::Mul(ratio_expr, num));
                                return Some(Rewrite {
                                    new_expr,
                                    description: format!("{} * (x / {}) -> ({} / {}) * x", n1, n2, n1, n2),
                                });
                            }
                        }
                    }
                }
            }
            Expr::Sub(lhs, rhs) => {
                let lhs_data = ctx.get(lhs).clone();
                let rhs_data = ctx.get(rhs).clone();
                if let (Expr::Number(n1), Expr::Number(n2)) = (&lhs_data, &rhs_data) {
                    let diff = n1 - n2;
                    let new_expr = ctx.add(Expr::Number(diff.clone()));
                    return Some(Rewrite {
                        new_expr,
                        description: format!("{} - {} = {}", n1, n2, diff),
                    });
                }
            }
            Expr::Div(lhs, rhs) => {
                let lhs_data = ctx.get(lhs).clone();
                let rhs_data = ctx.get(rhs).clone();
                if let (Expr::Number(n1), Expr::Number(n2)) = (&lhs_data, &rhs_data) {
                    if !n2.is_zero() {
                        let quot = n1 / n2;
                        let new_expr = ctx.add(Expr::Number(quot.clone()));
                        return Some(Rewrite {
                            new_expr,
                            description: format!("{} / {} = {}", n1, n2, quot),
                        });
                    } else {
                        let undef = ctx.add(Expr::Constant(cas_ast::Constant::Undefined));
                        return Some(Rewrite {
                            new_expr: undef,
                            description: "Division by zero".to_string(),
                        });
                    }
                }
            }
            _ => {}
        }
        None
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::{DisplayExpr, Context};

    #[test]
    fn test_add_zero() {
        let mut ctx = Context::new();
        let rule = AddZeroRule;
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let expr = ctx.add(Expr::Add(x, zero));
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "x");
    }

    #[test]
    fn test_mul_one() {
        let mut ctx = Context::new();
        let rule = MulOneRule;
        let one = ctx.num(1);
        let y = ctx.var("y");
        let expr = ctx.add(Expr::Mul(one, y));
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "y");
    }

    #[test]
    fn test_combine_constants() {
        let mut ctx = Context::new();
        let rule = CombineConstantsRule;
        let two = ctx.num(2);
        let three = ctx.num(3);
        let expr = ctx.add(Expr::Add(two, three));
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "5");
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));
    simplifier.add_rule(Box::new(MulZeroRule));
    simplifier.add_rule(Box::new(DivZeroRule));
    simplifier.add_rule(Box::new(CombineConstantsRule));
}
