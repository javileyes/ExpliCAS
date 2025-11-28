use crate::rule::Rewrite;
use crate::define_rule;
use cas_ast::Expr;
use std::cmp::Ordering;
use crate::ordering::compare_expr;

define_rule!(
    CanonicalizeNegationRule,
    "Canonicalize Negation",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        
        // 1. Subtraction: a - b -> a + (-b)
        if let Expr::Sub(lhs, rhs) = expr_data {
            let neg_rhs = ctx.add(Expr::Neg(rhs));
            let new_expr = ctx.add(Expr::Add(lhs, neg_rhs));
            return Some(Rewrite {
                new_expr,
                description: "a - b = a + (-b)".to_string(),
            });
        }

        // 2. Negation: -x -> -1 * x
        if let Expr::Neg(inner) = expr_data {
            let inner_data = ctx.get(inner).clone();
            if let Expr::Number(n) = inner_data {
                // -(-5) -> 5 (Handled by parser usually, but good to have)
                // Actually parser produces Neg(Number(5)).
                // If we have Neg(Number(5)), we want Number(-5).
                let neg_n = -n.clone();
                let new_expr = ctx.add(Expr::Number(neg_n.clone()));
                return Some(Rewrite {
                    new_expr,
                    description: format!("-({}) = {}", n, neg_n),
                });
            } else {
                // -x -> -1 * x
                let minus_one = ctx.num(-1);
                let new_expr = ctx.add(Expr::Mul(minus_one, inner));
                return Some(Rewrite {
                    new_expr,
                    description: "-x = -1 * x".to_string(),
                });
            }
        }
        None
    }
);

define_rule!(
    CanonicalizeAddRule,
    "Canonicalize Addition",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Add(lhs, rhs) = expr_data {
            // 1. Basic Swap: b + a -> a + b if b < a
            if compare_expr(ctx, rhs, lhs) == Ordering::Less {
                let new_expr = ctx.add(Expr::Add(rhs, lhs));
                return Some(Rewrite {
                    new_expr,
                    description: "Reorder addition terms".to_string(),
                });
            }
            
            // 2. Rotation: a + (b + c) -> b + (a + c) if b < a
            // This allows sorting nested terms.
            let rhs_data = ctx.get(rhs).clone();
            if let Expr::Add(rl, rr) = rhs_data {
                if compare_expr(ctx, rl, lhs) == Ordering::Less {
                    let inner_add = ctx.add(Expr::Add(lhs, rr));
                    let new_expr = ctx.add(Expr::Add(rl, inner_add));
                    return Some(Rewrite {
                        new_expr,
                        description: "Rotate addition terms".to_string(),
                    });
                }
            }
        }
        None
    }
);

define_rule!(
    CanonicalizeMulRule,
    "Canonicalize Multiplication",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Mul(lhs, rhs) = expr_data {
            // 1. Basic Swap: b * a -> a * b if b < a
            if compare_expr(ctx, rhs, lhs) == Ordering::Less {
                let new_expr = ctx.add(Expr::Mul(rhs, lhs));
                return Some(Rewrite {
                    new_expr,
                    description: "Reorder multiplication factors".to_string(),
                });
            }

            // 2. Rotation: a * (b * c) -> b * (a * c) if b < a
            let rhs_data = ctx.get(rhs).clone();
            if let Expr::Mul(rl, rr) = rhs_data {
                if compare_expr(ctx, rl, lhs) == Ordering::Less {
                    let inner_mul = ctx.add(Expr::Mul(lhs, rr));
                    let new_expr = ctx.add(Expr::Mul(rl, inner_mul));
                    return Some(Rewrite {
                        new_expr,
                        description: "Rotate multiplication factors".to_string(),
                    });
                }
            }
        }
        None
    }
);

define_rule!(
    CanonicalizeRootRule,
    "Canonicalize Roots",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Function(name, args) = expr_data {
            if name == "sqrt" {
                if args.len() == 1 {
                    let arg = args[0];
                    
                    // Try to factor argument to find perfect squares
                    // e.g. sqrt(x^2 + 2x + 1) -> sqrt((x+1)^2) -> |x+1|
                    use crate::polynomial::Polynomial;
                    use crate::rules::algebra::collect_variables;
                    
                    let vars = collect_variables(ctx, arg);
                    if vars.len() == 1 {
                        let var = vars.iter().next().unwrap();
                        if let Ok(poly) = Polynomial::from_expr(ctx, arg, var) {
                             let factors = poly.factor_rational_roots();
                             if !factors.is_empty() {
                                 let first = &factors[0];
                                 if factors.iter().all(|f| f == first) {
                                     let count = factors.len() as u32;
                                     if count >= 2 {
                                         // arg = base^count
                                         // sqrt(base^count) = |base|^(count/2) if count is even?
                                         // sqrt(x^2) = |x|.
                                         // sqrt(x^4) = x^2 = |x|^2.
                                         // sqrt(x^3) = |x| * sqrt(x).
                                         
                                         let base = first.to_expr(ctx);
                                         let k = count / 2;
                                         let rem = count % 2;
                                         
                                         let abs_base = ctx.add(Expr::Function("abs".to_string(), vec![base]));
                                         
                                         let term1 = if k == 1 {
                                             abs_base
                                         } else {
                                             let k_expr = ctx.num(k as i64);
                                             ctx.add(Expr::Pow(abs_base, k_expr))
                                         };
                                         
                                         if rem == 0 {
                                             return Some(Rewrite {
                                                 new_expr: term1,
                                                 description: "Simplify perfect square root".to_string(),
                                             });
                                         } else {
                                             // Remainder 1: term1 * sqrt(base)
                                             let sqrt_base = ctx.add(Expr::Function("sqrt".to_string(), vec![base]));
                                             let new_expr = ctx.add(Expr::Mul(term1, sqrt_base));
                                             return Some(Rewrite {
                                                 new_expr,
                                                 description: "Simplify square root factors".to_string(),
                                             });
                                         }
                                     }
                                 }
                             }
                        }
                    }

                    // sqrt(x) -> x^(1/2)
                    let half = ctx.rational(1, 2);
                    let new_expr = ctx.add(Expr::Pow(args[0], half));
                    return Some(Rewrite {
                        new_expr,
                        description: "sqrt(x) = x^(1/2)".to_string(),
                    });
                } else if args.len() == 2 {
                    // sqrt(x, n) -> x^(1/n)
                    let one = ctx.num(1);
                    let exp = ctx.add(Expr::Div(one, args[1]));
                    let new_expr = ctx.add(Expr::Pow(args[0], exp));
                    return Some(Rewrite {
                        new_expr,
                        description: format!("sqrt(x, n) = x^(1/n)"),
                    });
                }
            } else if name == "root" && args.len() == 2 {
                 // root(x, n) -> x^(1/n)
                 let one = ctx.num(1);
                 let exp = ctx.add(Expr::Div(one, args[1]));
                 let new_expr = ctx.add(Expr::Pow(args[0], exp));
                 return Some(Rewrite {
                    new_expr,
                    description: format!("root(x, n) = x^(1/n)"),
                });
            }
        }
        None
    }
);

define_rule!(
    AssociativityRule,
    "Associativity (Flattening)",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        match expr_data {
            // (a + b) + c -> a + (b + c)
            Expr::Add(lhs, rhs) => {
                let lhs_data = ctx.get(lhs).clone();
                if let Expr::Add(ll, lr) = lhs_data {
                    let inner_add = ctx.add(Expr::Add(lr, rhs));
                    let new_expr = ctx.add(Expr::Add(ll, inner_add));
                    return Some(Rewrite {
                        new_expr,
                        description: "Associativity: (a + b) + c -> a + (b + c)".to_string(),
                    });
                }
            }
            // (a * b) * c -> a * (b * c)
            Expr::Mul(lhs, rhs) => {
                let lhs_data = ctx.get(lhs).clone();
                if let Expr::Mul(ll, lr) = lhs_data {
                    let inner_mul = ctx.add(Expr::Mul(lr, rhs));
                    let new_expr = ctx.add(Expr::Mul(ll, inner_mul));
                    return Some(Rewrite {
                        new_expr,
                        description: "Associativity: (a * b) * c -> a * (b * c)".to_string(),
                    });
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
    use cas_parser::parse;
    use cas_ast::{DisplayExpr, Context};

    #[test]
    fn test_canonicalize_negation() {
        let mut ctx = Context::new();
        let rule = CanonicalizeNegationRule;
        // -5 -> -5 (Number)
        let expr = parse("-5", &mut ctx).unwrap(); // Neg(Number(5))
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        // The display might look the same "-5", but the structure is different.
        // Let's check if it's a Number.
        if let Expr::Number(n) = ctx.get(rewrite.new_expr) {
            assert_eq!(format!("{}", n), "-5");
        } else {
            panic!("Expected Number, got {:?}", ctx.get(rewrite.new_expr));
        }
    }

    #[test]
    fn test_canonicalize_sqrt() {
        let mut ctx = Context::new();
        let rule = CanonicalizeRootRule;
        // sqrt(x)
        let x = ctx.var("x");
        let expr = ctx.add(Expr::Function("sqrt".to_string(), vec![x]));
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        // Should be x^(1/2)
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "x^(1/2)");
    }

    #[test]
    fn test_canonicalize_nth_root() {
        let mut ctx = Context::new();
        let rule = CanonicalizeRootRule;
        
        // sqrt(x, 3) -> x^(1/3)
        let x = ctx.var("x");
        let three = ctx.num(3);
        let expr = ctx.add(Expr::Function("sqrt".to_string(), vec![x, three]));
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "x^(1 / 3)");

        // root(x, 4) -> x^(1/4)
        let four = ctx.num(4);
        let expr2 = ctx.add(Expr::Function("root".to_string(), vec![x, four]));
        let rewrite2 = rule.apply(&mut ctx, expr2).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite2.new_expr }), "x^(1 / 4)");
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(CanonicalizeNegationRule));
    simplifier.add_rule(Box::new(CanonicalizeAddRule));
    simplifier.add_rule(Box::new(CanonicalizeMulRule));
    simplifier.add_rule(Box::new(CanonicalizeRootRule));
    simplifier.add_rule(Box::new(AssociativityRule));
}
