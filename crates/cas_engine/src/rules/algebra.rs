use crate::rule::Rewrite;
use crate::define_rule;
use cas_ast::{Expr, ExprId, Context};
use crate::polynomial::Polynomial;
use std::collections::HashSet;
use num_traits::{One, Signed, ToPrimitive};


define_rule!(
    SimplifyFractionRule,
    "Simplify Algebraic Fraction",
    |ctx, expr| {
        let (num, den) = if let Expr::Div(n, d) = ctx.get(expr) {
            (*n, *d)
        } else {
            return None;
        };

        // 1. Identify variable
        let vars = collect_variables(ctx, expr);
        if vars.len() != 1 {
            return None; // Only univariate for now
        }
        let var = vars.iter().next().unwrap();

        // 2. Convert to Polynomials
        let p_num = Polynomial::from_expr(ctx, num, var).ok()?;
        let p_den = Polynomial::from_expr(ctx, den, var).ok()?;

        // 3. Compute GCD
        let gcd = p_num.gcd(&p_den);

        // 4. Check if GCD is non-trivial (degree > 0 or constant != 1)
        // Actually, even constant GCD is useful for reducing 2x/2 -> x
        if gcd.degree() == 0 && gcd.leading_coeff().is_one() {
            return None;
        }

        // 5. Divide
        let (new_num_poly, rem_num) = p_num.div_rem(&gcd);
        let (new_den_poly, rem_den) = p_den.div_rem(&gcd);

        if !rem_num.is_zero() || !rem_den.is_zero() {
            // Should not happen if GCD is correct
            return None;
        }

        let new_num = new_num_poly.to_expr(ctx);
        let new_den = new_den_poly.to_expr(ctx);
        let gcd_expr = gcd.to_expr(ctx);

        // If denominator is 1, return numerator
        if let Expr::Number(n) = ctx.get(new_den) {
            if n.is_one() {
                return Some(Rewrite {
                    new_expr: new_num,
                    description: format!("Simplified fraction by GCD: {:?}", gcd_expr), // Debug format
                });
            }
        }

        return Some(Rewrite {
            new_expr: ctx.add(Expr::Div(new_num, new_den)),
            description: format!("Simplified fraction by GCD: {:?}", gcd_expr),
        });
    }
);

define_rule!(
    NestedFractionRule,
    "Simplify Nested Fraction",
    |ctx, expr| {
        let (num, den) = if let Expr::Div(n, d) = ctx.get(expr) {
            (*n, *d)
        } else {
            return None;
        };

        let num_denoms = collect_denominators(ctx, num);
        let den_denoms = collect_denominators(ctx, den);
        
        if num_denoms.is_empty() && den_denoms.is_empty() {
            return None;
        }
        
        // Collect all unique denominators
        let mut all_denoms = Vec::new();
        all_denoms.extend(num_denoms);
        all_denoms.extend(den_denoms);
        
        if all_denoms.is_empty() {
            return None;
        }
        
        // Construct the common multiplier (product of all unique denominators)
        // Ideally LCM, but product is safer for now.
        // We need to deduplicate.
        let mut unique_denoms: Vec<ExprId> = Vec::new();
        for d in all_denoms {
            if !unique_denoms.contains(&d) {
                unique_denoms.push(d);
            }
        }
        
        if unique_denoms.is_empty() {
            return None;
        }

        let mut multiplier = unique_denoms[0];
        for i in 1..unique_denoms.len() {
            multiplier = ctx.add(Expr::Mul(multiplier, unique_denoms[i]));
        }
        
        // Multiply num and den by multiplier
        let new_num = distribute(ctx, num, multiplier);
        let new_den = distribute(ctx, den, multiplier);
        
        let new_expr = ctx.add(Expr::Div(new_num, new_den));
        if new_expr == expr {
            return None;
        }

        return Some(Rewrite {
            new_expr,
            description: format!("Multiply by common denominator {:?}", multiplier),
        });
    }
);

fn distribute(ctx: &mut Context, target: ExprId, multiplier: ExprId) -> ExprId {
    let target_data = ctx.get(target).clone();
    match target_data {
        Expr::Add(l, r) => {
            let dl = distribute(ctx, l, multiplier);
            let dr = distribute(ctx, r, multiplier);
            ctx.add(Expr::Add(dl, dr))
        },
        Expr::Sub(l, r) => {
            let dl = distribute(ctx, l, multiplier);
            let dr = distribute(ctx, r, multiplier);
            ctx.add(Expr::Sub(dl, dr))
        },
        Expr::Mul(l, r) => {
            // Try to distribute into the side that has denominators
            let l_denoms = collect_denominators(ctx, l);
            if !l_denoms.is_empty() {
                 let dl = distribute(ctx, l, multiplier);
                 return ctx.add(Expr::Mul(dl, r));
            }
            let r_denoms = collect_denominators(ctx, r);
            if !r_denoms.is_empty() {
                 let dr = distribute(ctx, r, multiplier);
                 return ctx.add(Expr::Mul(l, dr));
            }
            // If neither has explicit denominators, just multiply
            ctx.add(Expr::Mul(target, multiplier))
        },
        Expr::Div(l, r) => {
            // (l / r) * m. 
            // Check if m is a multiple of r.
            if let Some(quotient) = get_quotient(ctx, multiplier, r) {
                // m = q * r.
                // (l / r) * (q * r) = l * q
                return ctx.add(Expr::Mul(l, quotient));
            }
            // If not, we are stuck with (l/r)*m.
            let div_expr = ctx.add(Expr::Div(l, r));
            ctx.add(Expr::Mul(div_expr, multiplier))
        },
        _ => ctx.add(Expr::Mul(target, multiplier))
    }
}

fn get_quotient(ctx: &mut Context, dividend: ExprId, divisor: ExprId) -> Option<ExprId> {
    if dividend == divisor {
        return Some(ctx.num(1));
    }
    
    let dividend_data = ctx.get(dividend).clone();
    
    match dividend_data {
        Expr::Mul(l, r) => {
            if let Some(q) = get_quotient(ctx, l, divisor) {
                return Some(ctx.add(Expr::Mul(q, r)));
            }
            if let Some(q) = get_quotient(ctx, r, divisor) {
                return Some(ctx.add(Expr::Mul(l, q)));
            }
            None
        }
        _ => None
    }
}

fn collect_denominators(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut denoms = Vec::new();
    match ctx.get(expr) {
        Expr::Div(_, den) => {
            denoms.push(*den);
            // Recurse? Maybe not needed for simple cases.
        },
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            denoms.extend(collect_denominators(ctx, *l));
            denoms.extend(collect_denominators(ctx, *r));
        },
        Expr::Pow(b, e) => {
            // Check for negative exponent?
            if let Expr::Number(n) = ctx.get(*e) {
                if n.is_negative() {
                    // b^-k = 1/b^k. Denominator is b^k (or b if k=-1)
                    // For simplicity, let's just handle 1/x style Divs first.
                }
            }
            denoms.extend(collect_denominators(ctx, *b));
        },
        _ => {}
    }
    denoms
}

define_rule!(
    ExpandRule,
    "Expand Polynomial",
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if name == "expand" && args.len() == 1 {
                let arg = args[0];
                // Try to convert to polynomial
                let vars = collect_variables(ctx, arg);
                if vars.is_empty() {
                    // Constant expression. Already expanded.
                    // Just return the arg (simplification removes the expand wrapper).
                    return Some(Rewrite {
                        new_expr: arg,
                        description: "expand(constant) -> constant".to_string(),
                    });
                }
                if vars.len() != 1 {
                    return None;
                }
                let var = vars.iter().next().unwrap();
                
                if let Ok(poly) = Polynomial::from_expr(ctx, arg, var) {
                    return Some(Rewrite {
                        new_expr: poly.to_expr(ctx),
                        description: "expand(poly)".to_string(),
                    });
                }
            }
        }
        None
    }
);

define_rule!(
    FactorRule,
    "Factor Polynomial",
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if name == "factor" && args.len() == 1 {
                let arg = args[0];
                let vars = collect_variables(ctx, arg);
                if vars.len() != 1 {
                    return None;
                }
                let var = vars.iter().next().unwrap();
                
                if let Ok(poly) = Polynomial::from_expr(ctx, arg, var) {
                    if poly.is_zero() { return None; }

                    // 1. Extract content (common constant factor)
                    let factors = poly.factor_rational_roots();
                    
                    if factors.len() == 1 {
                        // Irreducible (over rationals) or just trivial
                        let content = poly.content();
                        let min_deg = poly.min_degree();
                        if content.is_one() && min_deg == 0 {
                            return Some(Rewrite {
                                new_expr: arg,
                                description: "Irreducible".to_string(),
                            });
                        }
                    }

                    // Construct the expression from factors
                    // factors[0] * factors[1] * ...
                    let mut res = factors[0].to_expr(ctx);
                    for factor in factors.iter().skip(1) {
                        let f_expr = factor.to_expr(ctx);
                        res = ctx.add(Expr::Mul(res, f_expr));
                    }

                    return Some(Rewrite {
                        new_expr: res,
                        description: "Factorization".to_string(),
                    });
                }
            }
        }
        None
    }
);

fn collect_variables(ctx: &Context, expr: ExprId) -> HashSet<String> {
    use crate::visitors::VariableCollector;
    use cas_ast::Visitor;
    
    let mut collector = VariableCollector::new();
    collector.visit_expr(ctx, expr);
    collector.vars
}


define_rule!(
    FactorDifferenceSquaresRule,
    "Factor Difference of Squares",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        let (l, r) = match expr_data {
            Expr::Sub(l, r) => (l, r),
            Expr::Add(a, b) => {
                // Check if one is negative
                if is_negative_term(ctx, b) {
                    (a, negate_term(ctx, b))
                } else if is_negative_term(ctx, a) {
                    (b, negate_term(ctx, a))
                } else {
                    return None;
                }
            },
            _ => return None,
        };

        if let (Some(root_l), Some(root_r)) = (get_square_root(ctx, l), get_square_root(ctx, r)) {
            // a^2 - b^2 = (a - b)(a + b)
            let term1 = ctx.add(Expr::Sub(root_l, root_r));
            
            // Check for Pythagorean identity in term2 (a + b)
            // sin^2 + cos^2 = 1
            let mut term2 = ctx.add(Expr::Add(root_l, root_r));
            
            if is_sin_cos_pair(ctx, root_l, root_r) {
                 term2 = ctx.num(1);
            }

            let new_expr = ctx.add(Expr::Mul(term1, term2));
            
            return Some(Rewrite {
                new_expr,
                description: "Factor difference of squares".to_string(),
            });
        }
        None
    }
);

use crate::helpers::{get_square_root, get_trig_arg, is_trig_pow};

fn is_sin_cos_pair(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    let arg_a = get_trig_arg(ctx, a);
    let arg_b = get_trig_arg(ctx, b);
    
    // Check if args match and are Some
    if arg_a.is_none() || arg_b.is_none() {
        return false;
    }
    if arg_a != arg_b {
        return false;
    }

    (is_trig_pow(ctx, a, "sin", 2) && is_trig_pow(ctx, b, "cos", 2)) ||
    (is_trig_pow(ctx, a, "cos", 2) && is_trig_pow(ctx, b, "sin", 2))
}

fn is_negative_term(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Neg(_) => true,
        Expr::Mul(l, _) => {
            if let Expr::Number(n) = ctx.get(*l) {
                n.is_negative()
            } else {
                false
            }
        },
        Expr::Number(n) => n.is_negative(),
        _ => false
    }
}

fn negate_term(ctx: &mut Context, expr: ExprId) -> ExprId {
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Neg(inner) => inner,
        Expr::Mul(l, r) => {
            if let Expr::Number(n) = ctx.get(l) {
                if n.is_negative() {
                    let new_n = (-n).to_i64().unwrap();
                    if new_n == 1 {
                        return r;
                    }
                    let num_expr = ctx.num(new_n);
                    return ctx.add(Expr::Mul(num_expr, r));
                }
            }
            ctx.add(Expr::Neg(expr))
        },
        Expr::Number(n) => ctx.num((-n).to_i64().unwrap()),
        _ => ctx.add(Expr::Neg(expr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_parser::parse;
    use cas_ast::DisplayExpr;

    #[test]
    fn test_simplify_fraction() {
        let mut ctx = Context::new();
        let rule = SimplifyFractionRule;

        // (x^2 - 1) / (x + 1) -> x - 1
        let expr = parse("(x^2 - 1) / (x + 1)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        // Result might be -1 + x or x - 1 depending on polynomial to_expr order
        // Polynomial to_expr outputs lowest degree first? 
        // My implementation: "1 + x" for x+1.
        // x^2 - 1 = (x-1)(x+1). 
        // (x-1) -> -1 + x
        let s = format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr });
        assert!(s.contains("x"));
        assert!(s.contains("-1"));
    }
    
    #[test]
    fn test_simplify_fraction_2() {
        let mut ctx = Context::new();
        let rule = SimplifyFractionRule;
        // (x^2 + 2*x + 1) / (x + 1) -> x + 1
        let expr = parse("(x^2 + 2*x + 1) / (x + 1)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        let s = format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr });
        assert!(s.contains("1"));
        assert!(s.contains("x"));
    }

    #[test]
    fn test_factor_difference_squares() {
        let mut ctx = Context::new();
        let rule = FactorRule;
        // factor(x^2 - 1) -> (x - 1)(x + 1)
        // Note: My implementation produces (x-1) and (x+1) (or similar).
        // Order depends on root finding.
        // Roots are 1, -1.
        // Factors: (x-1), (x+1).
        let expr = parse("factor(x^2 - 1)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        let res = format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr });
        assert!(res.contains("x - 1") || res.contains("-1 + x") || res.contains("x + -1"));
        assert!(res.contains("x + 1") || res.contains("1 + x"));
    }

    #[test]
    fn test_factor_perfect_square() {
        let mut ctx = Context::new();
        let rule = FactorRule;
        // factor(x^2 + 2x + 1) -> (x + 1)(x + 1)
        let expr = parse("factor(x^2 + 2*x + 1)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        let res = format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr });
        // Should be (x+1) * (x+1)
        assert!(res.contains("x + 1") || res.contains("1 + x"));
        assert!(res.contains("*"));
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(SimplifyFractionRule));
    simplifier.add_rule(Box::new(NestedFractionRule));
    simplifier.add_rule(Box::new(ExpandRule));
    simplifier.add_rule(Box::new(FactorRule));
    simplifier.add_rule(Box::new(FactorDifferenceSquaresRule));
}
