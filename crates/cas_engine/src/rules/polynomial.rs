use crate::rule::Rewrite;
use crate::define_rule;
use cas_ast::{Expr, ExprId, Context};
use num_traits::{ToPrimitive, Signed};
use num_rational::BigRational;
use num_traits::One;
use crate::ordering::compare_expr;
use std::cmp::Ordering;

define_rule!(
    DistributeRule,
    "Distributive Property",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Mul(l, r) = expr_data {
            // a * (b + c) -> a*b + a*c
            let r_data = ctx.get(r).clone();
            if let Expr::Add(b, c) = r_data {
                let ab = ctx.add(Expr::Mul(l, b));
                let ac = ctx.add(Expr::Mul(l, c));
                let new_expr = ctx.add(Expr::Add(ab, ac));
                return Some(Rewrite {
                    new_expr,
                    description: "Distribute".to_string(),
                });
            }
            // (b + c) * a -> b*a + c*a
            let l_data = ctx.get(l).clone();
            if let Expr::Add(b, c) = l_data {
                let ba = ctx.add(Expr::Mul(b, r));
                let ca = ctx.add(Expr::Mul(c, r));
                let new_expr = ctx.add(Expr::Add(ba, ca));
                return Some(Rewrite {
                    new_expr,
                    description: "Distribute".to_string(),
                });
            }
        }
        None
    }
);

define_rule!(
    AnnihilationRule,
    "Annihilation",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Sub(l, r) = expr_data {
            if compare_expr(ctx, l, r) == Ordering::Equal {
                let zero = ctx.num(0);
                return Some(Rewrite {
                    new_expr: zero,
                    description: "x - x = 0".to_string(),
                });
            }
        }
        None
    }
);

define_rule!(
    CombineLikeTermsRule,
    "Combine Like Terms",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Add(l, r) = expr_data {
            // Helper to extract (coeff, var_part)
            // 2x -> (2, x)
            // x -> (1, x)
            let get_parts = |context: &Context, e: ExprId| -> Option<(BigRational, ExprId)> {
                match context.get(e) {
                    Expr::Mul(a, b) => {
                        if let Expr::Number(n) = context.get(*a) {
                            Some((n.clone(), *b))
                        } else if let Expr::Number(n) = context.get(*b) {
                            Some((n.clone(), *a))
                        } else {
                            None
                        }
                    }
                    Expr::Number(_) => None, // Handled by CombineConstantsRule
                    _ => Some((BigRational::one(), e)),
                }
            };

            if let (Some((c1, v1)), Some((c2, v2))) = (get_parts(ctx, l), get_parts(ctx, r)) {
                if compare_expr(ctx, v1, v2) == Ordering::Equal {
                    let new_coeff = &c1 + &c2;
                    let new_term = if new_coeff.is_one() {
                        v1
                    } else {
                        let coeff_expr = ctx.add(Expr::Number(new_coeff.clone()));
                        ctx.add(Expr::Mul(coeff_expr, v1))
                    };
                    return Some(Rewrite {
                        new_expr: new_term,
                        description: format!("Combine like terms: {}{} + {}{}", c1, v1, c2, v2), // Note: Display might be tricky here without context, but it's just a string description
                    });
                }
            }
        }
        None
    }
);

define_rule!(
    BinomialExpansionRule,
    "Binomial Expansion",
    |ctx, expr| {
        // (a + b)^n
        let expr_data = ctx.get(expr).clone();
        if let Expr::Pow(base, exp) = expr_data {
            let base_data = ctx.get(base).clone();
            if let Expr::Add(a, b) = base_data {
                let exp_data = ctx.get(exp).clone();
                if let Expr::Number(n) = exp_data {
                    if n.is_integer() && !n.is_negative() {
                        let n_val = n.to_integer().to_u32()?;
                        // Limit expansion to reasonable size to prevent explosion
                        if n_val >= 2 && n_val <= 10 {
                            // Expand: sum(k=0 to n) (n choose k) * a^(n-k) * b^k
                            let mut terms = Vec::new();
                            for k in 0..=n_val {
                                let coeff = binomial_coeff(n_val, k);
                                let exp_a = n_val - k;
                                let exp_b = k;
                                
                                let term_a = if exp_a == 0 { ctx.num(1) } else if exp_a == 1 { a } else { 
                                    let e = ctx.num(exp_a as i64);
                                    ctx.add(Expr::Pow(a, e)) 
                                };
                                let term_b = if exp_b == 0 { ctx.num(1) } else if exp_b == 1 { b } else { 
                                    let e = ctx.num(exp_b as i64);
                                    ctx.add(Expr::Pow(b, e)) 
                                };
                                
                                let mut term = ctx.add(Expr::Mul(term_a, term_b));
                                if coeff > 1 {
                                    let c = ctx.num(coeff as i64);
                                    term = ctx.add(Expr::Mul(c, term));
                                }
                                terms.push(term);
                            }
                            
                            // Sum up terms
                            let mut expanded = terms[0];
                            for i in 1..terms.len() {
                                expanded = ctx.add(Expr::Add(expanded, terms[i]));
                            }
                            
                            return Some(Rewrite {
                                new_expr: expanded,
                                description: format!("Expand binomial power ^{}", n_val),
                            });
                        }
                    }
                }
            }
        }
        None
    }
);

fn binomial_coeff(n: u32, k: u32) -> u32 {
    if k == 0 || k == n {
        return 1;
    }
    if k > n {
        return 0;
    }
    let mut res = 1;
    for i in 0..k {
        res = res * (n - i) / (i + 1);
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::{DisplayExpr, Context};

    #[test]
    fn test_distribute() {
        let mut ctx = Context::new();
        let rule = DistributeRule;
        // 2 * (x + 3)
        let two = ctx.num(2);
        let x = ctx.var("x");
        let three = ctx.num(3);
        let add = ctx.add(Expr::Add(x, three));
        let expr = ctx.add(Expr::Mul(two, add));

        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        // Should be (2 * x) + (2 * 3)
        // Note: Simplification of 2*3 happens in a later pass by CombineConstantsRule
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "2 * x + 2 * 3");
    }

    #[test]
    fn test_annihilation() {
        let mut ctx = Context::new();
        let rule = AnnihilationRule;
        let x = ctx.var("x");
        let expr = ctx.add(Expr::Sub(x, x));
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "0");
    }

    #[test]
    fn test_combine_like_terms() {
        let mut ctx = Context::new();
        let rule = CombineLikeTermsRule;
        
        // 2x + 3x -> 5x
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let term1 = ctx.add(Expr::Mul(two, x));
        let term2 = ctx.add(Expr::Mul(three, x));
        let expr = ctx.add(Expr::Add(term1, term2));

        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "5 * x");

        // x + 2x -> 3x
        let term1 = x;
        let term2 = ctx.add(Expr::Mul(two, x));
        let expr2 = ctx.add(Expr::Add(term1, term2));
        let rewrite2 = rule.apply(&mut ctx, expr2).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite2.new_expr }), "3 * x");

        // ln(x) + ln(x) -> 2 * ln(x)
        let ln_x = ctx.add(Expr::Function("ln".to_string(), vec![x]));
        let expr3 = ctx.add(Expr::Add(ln_x, ln_x));
        let rewrite3 = rule.apply(&mut ctx, expr3).unwrap();
        // ln(x) is log(e, x), prints as ln(x)
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite3.new_expr }), "2 * ln(x)");
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(DistributeRule));
    simplifier.add_rule(Box::new(AnnihilationRule));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(BinomialExpansionRule));
}
