use crate::rule::Rewrite;
use crate::define_rule;
use cas_ast::{Expr, ExprId, Context};
use crate::polynomial::Polynomial;
use std::collections::HashSet;
use crate::ordering::compare_expr;
use std::cmp::Ordering;
use num_traits::{One, Signed, Zero};
use num_rational::BigRational;


define_rule!(
    SimplifyFractionRule,
    "Simplify Nested Fraction",
    |ctx, expr| {
        // eprintln!("SimplifyFractionRule visiting: {:?}", ctx.get(expr));
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
        
        if p_den.is_zero() {
            return None;
        }

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
        
        // println!("NestedFractionRule: {} -> {}", cas_ast::DisplayExpr { context: ctx, id: expr }, cas_ast::DisplayExpr { context: ctx, id: new_expr });

        if new_expr == expr {
            return None;
        }

        // Complexity Check: Ensure we actually reduced the number of divisions or total nodes
        // Counting Div nodes is a good heuristic for "nested fraction simplified"
        let count_divs = |id| count_nodes_of_type(ctx, id, "Div");
        let old_divs = count_divs(expr);
        let new_divs = count_divs(new_expr);

        if new_divs >= old_divs {
             // If we didn't reduce Div count, maybe we reduced total size?
             // But usually we want to reduce nesting.
             // If we went from (1+1/x)/1 -> (x+1)/x, Div count is same (1 -> 1).
             // But nesting is gone.
             // (1+1/x) has 1 Div. Total expr has 2 Divs.
             // (x+1)/x has 1 Div.
             // So Div count SHOULD decrease.
             // Wait, (1+1/x)/1:
             //   Div(Add(1, Div(1, x)), 1) -> 2 Divs.
             // (x+1)/x:
             //   Div(Add(x, 1), x) -> 1 Div.
             // So Div count should decrease.
             
             // What if we have (1/x)/(1/y) -> y/x?
             // Div(Div(1,x), Div(1,y)) -> 3 Divs.
             // Div(y, x) -> 1 Div.
             // Decrease.
             
             // So requiring decrease in Div count is a good check.
             return None;
        }

        // eprintln!("NestedFractionRule rewriting: {:?} -> {:?}", expr, new_expr);
        return Some(Rewrite {
            new_expr,
            description: format!("Multiply by common denominator {:?}", multiplier),
        });
    }
);

fn count_nodes_of_type(ctx: &Context, expr: ExprId, variant: &str) -> usize {
    let mut count = 0;
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        let node = ctx.get(id);
        if get_variant_name(node) == variant {
            count += 1;
        }
        match node {
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            },
            Expr::Neg(e) => stack.push(*e),
            Expr::Function(_, args) => stack.extend(args),
            _ => {}
        }
    }
    count
}

// Helper to get variant name (duplicated from engine.rs, should be shared but for now local is fine or use public if available)
fn get_variant_name(expr: &Expr) -> &'static str {
    match expr {
        Expr::Number(_) => "Number",
        Expr::Constant(_) => "Constant",
        Expr::Variable(_) => "Variable",
        Expr::Add(_, _) => "Add",
        Expr::Sub(_, _) => "Sub",
        Expr::Mul(_, _) => "Mul",
        Expr::Div(_, _) => "Div",
        Expr::Pow(_, _) => "Pow",
        Expr::Neg(_) => "Neg",
        Expr::Function(_, _) => "Function",
    }
}

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
                 // Chain distribution: result is dl * r = (l*m) * r.
                 // We want to distribute dl into r to clear r's denominators if any.
                 return distribute(ctx, r, dl);
            }
            let r_denoms = collect_denominators(ctx, r);
            if !r_denoms.is_empty() {
                 let dr = distribute(ctx, r, multiplier);
                 // Chain distribution: result is l * dr = l * (r*m).
                 // Distribute dr into l.
                 return distribute(ctx, l, dr);
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
            // eprintln!("distribute failed to divide: {:?} / {:?} by {:?}", multiplier, r, multiplier);
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
                let new_expr = crate::expand::expand(ctx, arg);
                if new_expr != expr {
                    return Some(Rewrite {
                        new_expr,
                        description: "expand()".to_string(),
                    });
                } else {
                    // If expand didn't change anything, maybe we should just unwrap?
                    // "expand(x)" -> "x"
                    return Some(Rewrite {
                        new_expr: arg,
                        description: "expand(atom)".to_string(),
                    });
                }
            }
        }
        None
    }
);

define_rule!(
    ConservativeExpandRule,
    "Conservative Expand",
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            // If explicit expand() call, always expand
            if name == "expand" && args.len() == 1 {
                let arg = args[0];
                let new_expr = crate::expand::expand(ctx, arg);
                if new_expr != expr {
                    return Some(Rewrite {
                        new_expr,
                        description: "expand()".to_string(),
                    });
                } else {
                    return Some(Rewrite {
                        new_expr: arg,
                        description: "expand(atom)".to_string(),
                    });
                }
            }
        }
        
        // Implicit expansion (e.g. (x+1)^2)
        // Only expand if complexity does not increase
        let new_expr = crate::expand::expand(ctx, expr);
        if new_expr != expr {
            let old_count = count_nodes(ctx, expr);
            let new_count = count_nodes(ctx, new_expr);
            
            if new_count <= old_count {
                 // Check for structural equality to avoid loops with ID regeneration
                 if crate::ordering::compare_expr(ctx, new_expr, expr) == std::cmp::Ordering::Equal {
                     return None;
                 }
                 return Some(Rewrite {
                     new_expr,
                     description: "Conservative Expansion".to_string(),
                 });
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
                // Use the general factor entry point which tries polynomial then diff squares
                let new_expr = crate::factor::factor(ctx, arg);
                if new_expr != arg {
                    return Some(Rewrite {
                        new_expr,
                        description: "Factorization".to_string(),
                    });
                }
            }
        }
        None
    }
);

pub fn collect_variables(ctx: &Context, expr: ExprId) -> HashSet<String> {
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
        // 1. Check if it's an Add/Sub chain
        match ctx.get(expr) {
            Expr::Add(_, _) | Expr::Sub(_, _) => {},
            _ => return None,
        }

        // 2. Flatten the chain
        let mut terms = Vec::new();
        crate::helpers::flatten_add(ctx, expr, &mut terms);

        // 3. Separate into potential squares and negative squares
        // We look for pairs (A, B) where A is a square and B is a negative square (B = -C^2)
        // i.e. A - C^2.
        
        // Optimization: We only need to find ONE pair that works.
        // We can iterate O(N^2) or sort? O(N^2) is fine for small N.
        
        for i in 0..terms.len() {
            for j in 0..terms.len() {
                if i == j { continue; }
                
                let t1 = terms[i];
                let t2 = terms[j];
                
                // Check if t1 + t2 forms a difference of squares
                // We construct a temporary Add(t1, t2) and call factor_difference_squares
                // This reuses the existing logic (including get_square_root and is_pythagorean)
                let pair = ctx.add(Expr::Add(t1, t2));
                
                if let Some(factored) = crate::factor::factor_difference_squares(ctx, pair) {
                     // Complexity check
                     let old_count = count_nodes(ctx, pair);
                     let new_count = count_nodes(ctx, factored);
                     
                     // Smart Check:
                     // If the result is a Mul, it means we factored into (A-B)(A+B).
                     // This usually increases complexity and blocks cancellation (due to DistributeRule guard).
                     // So we require STRICT reduction (<).
                     //
                     // If the result is NOT a Mul, it means we used a Pythagorean identity (sin^2+cos^2=1).
                     // The result is just (A-B). This is a simplification we want, even if size is same.
                     // So we allow SAME size (<=).
                     
                     let is_mul = matches!(ctx.get(factored), Expr::Mul(_, _));
                     let allowed = if is_mul {
                         new_count < old_count
                     } else {
                         new_count <= old_count
                     };

                     if allowed {
                         // Found a pair!
                         // Construct the new expression: Factored + (Terms - {t1, t2})
                         let mut new_terms = Vec::new();
                         new_terms.push(factored);
                         for k in 0..terms.len() {
                             if k != i && k != j {
                                 new_terms.push(terms[k]);
                             }
                         }
                         
                         // Rebuild Add chain
                         if new_terms.is_empty() {
                             return Some(Rewrite {
                                 new_expr: ctx.num(0),
                                 description: "Factor difference of squares (Empty)".to_string(),
                             });
                         }
                         
                         let mut new_expr = new_terms[0];
                         for t in new_terms.iter().skip(1) {
                             new_expr = ctx.add(Expr::Add(new_expr, *t));
                         }
                         
                         return Some(Rewrite {
                             new_expr,
                             description: "Factor difference of squares (N-ary)".to_string(),
                         });
                     }
                }
            }
        }
        None
    }
);

define_rule!(
    AutomaticFactorRule,
    "Automatic Factorization",
    |ctx, expr| {
        // Only try to factor if it's an Add or Sub (polynomial-like)
        match ctx.get(expr) {
            Expr::Add(_, _) | Expr::Sub(_, _) => {},
            _ => return None,
        }

        // Try factor_polynomial first
        if let Some(new_expr) = crate::factor::factor_polynomial(ctx, expr) {
             if new_expr != expr {
                 // Complexity check: Only accept if it strictly reduces size
                 // This prevents loops with ExpandRule which usually increases size (or keeps it same)
                 let old_count = count_nodes(ctx, expr);
                 let new_count = count_nodes(ctx, new_expr);
                 
                 if new_count < old_count {
                     // Check for structural equality (though unlikely if count reduced)
                     if crate::ordering::compare_expr(ctx, new_expr, expr) == std::cmp::Ordering::Equal {
                         return None;
                     }
                     return Some(Rewrite {
                         new_expr,
                         description: "Automatic Factorization (Reduced Size)".to_string(),
                     });
                 }
             }
        }
        
        // Try difference of squares
        // Note: Diff squares usually increases size: a^2 - b^2 (5) -> (a-b)(a+b) (7)
        // So this will rarely trigger with strict size check unless terms simplify further.
        // e.g. x^4 - 1 -> (x^2-1)(x^2+1) -> (x-1)(x+1)(x^2+1).
        // x^4 - 1 (5 nodes). (x-1)(x+1)(x^2+1) (many nodes).
        // So auto-factoring diff squares is risky for loops.
        // We'll skip it for now in AutomaticFactorRule, or only if it reduces size.
        if let Some(new_expr) = crate::factor::factor_difference_squares(ctx, expr) {
             if new_expr != expr {
                 let old_count = count_nodes(ctx, expr);
                 let new_count = count_nodes(ctx, new_expr);
                 if new_count < old_count {
                     return Some(Rewrite {
                         new_expr,
                         description: "Automatic Factorization (Diff Squares)".to_string(),
                     });
                 }
             }
        }

        None
    }
);

fn count_nodes(ctx: &Context, expr: ExprId) -> usize {
    match ctx.get(expr) {
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            1 + count_nodes(ctx, *l) + count_nodes(ctx, *r)
        },
        Expr::Neg(e) => 1 + count_nodes(ctx, *e),
        Expr::Function(_, args) => 1 + args.iter().map(|a| count_nodes(ctx, *a)).sum::<usize>(),
        _ => 1
    }
}



// Removed local is_sin_cos_pair, is_negative_term, negate_term as they are now in factor module (or internal to it)
// If other rules need them, I should expose them from factor module or helpers.
// Checking file... is_negative_term was used by FactorDifferenceSquaresRule only.
// negate_term was used by FactorDifferenceSquaresRule only.
// is_sin_cos_pair was used by FactorDifferenceSquaresRule only.
// So safe to remove.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_parser::parse;
    use cas_ast::{DisplayExpr, Context};

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
        // Should be (x+1)^2
        assert!(res.contains("x + 1") || res.contains("1 + x"));
        assert!(res.contains("^2") || res.contains("^ 2"));
    }
}


define_rule!(
    AddFractionsRule,
    "Add Algebraic Fractions",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Add(l, r) = expr_data {
            // eprintln!("AddFractionsRule checking: {:?}", expr);
            // Check if either is a fraction or Neg(fraction)
            let is_frac = |e: ExprId| {
                match ctx.get(e) {
                    Expr::Div(_, _) => true,
                    Expr::Neg(inner) => matches!(ctx.get(*inner), Expr::Div(_, _)),
                    _ => false
                }
            };
            if !is_frac(l) && !is_frac(r) {
                return None;
            }

            // Helper to get num/den
            let mut get_nd = |e: ExprId| -> (ExprId, ExprId) {
                let expr_data = ctx.get(e).clone();
                match expr_data {
                    Expr::Div(n, d) => (n, d),
                    Expr::Neg(inner) => {
                        let inner_data = ctx.get(inner).clone();
                        if let Expr::Div(n, d) = inner_data {
                            let neg_n = ctx.add(Expr::Neg(n));
                            (neg_n, d)
                        } else {
                            (e, ctx.num(1))
                        }
                    },
                    Expr::Mul(l, r) => {
                        // Check for c * (n/d) or (n/d) * c
                        let r_data = ctx.get(r).clone();
                        if let Expr::Div(n, d) = r_data {
                            let l_data = ctx.get(l).clone();
                            if let Expr::Number(_) = l_data {
                                return (ctx.add(Expr::Mul(l, n)), d);
                            }
                        }
                        let l_data = ctx.get(l).clone();
                        if let Expr::Div(n, d) = l_data {
                            let r_data = ctx.get(r).clone();
                            if let Expr::Number(_) = r_data {
                                return (ctx.add(Expr::Mul(n, r)), d);
                            }
                        }
                        (e, ctx.num(1))
                    },
                    _ => (e, ctx.num(1))
                }
            };

            let (n1, d1) = get_nd(l);
            let (n2, d2) = get_nd(r);

            // If denominators are same, simple add
            if crate::ordering::compare_expr(ctx, d1, d2) == std::cmp::Ordering::Equal {
                let new_num = ctx.add(Expr::Add(n1, n2));
                let new_expr = ctx.add(Expr::Div(new_num, d1));
                
                // Complexity check
                let old_complexity = count_nodes(ctx, expr);
                let new_complexity = count_nodes(ctx, new_expr);
                
                if new_complexity <= old_complexity {
                    return Some(Rewrite {
                        new_expr,
                        description: "Add fractions with same denominator".to_string(),
                    });
                }
            }

            // Different denominators. Try to find LCM.
            // We need variables to use Polynomials.
            let vars = collect_variables(ctx, expr);
            if vars.len() != 1 {
                // Multivariate LCM is hard. Just use product?
                // For now, only support univariate LCM for "Rational Crusher".
                // Or simple product if simple expressions.
                return None; 
            }
            let var = vars.iter().next().unwrap();

            let p_d1_res = Polynomial::from_expr(ctx, d1, var);
            let p_d2_res = Polynomial::from_expr(ctx, d2, var);
            let p_n1_res = Polynomial::from_expr(ctx, n1, var);
            let p_n2_res = Polynomial::from_expr(ctx, n2, var);

            // Check if denominators are negations of each other (d1 = -d2)
            // This allows combining A/d + B/(-d) -> (A-B)/d even if A, B are not polynomials
            if let (Ok(p_d1), Ok(p_d2)) = (&p_d1_res, &p_d2_res) {
                if p_d1.add(p_d2).is_zero() {
                     let new_num = ctx.add(Expr::Sub(n1, n2));
                     let new_expr = ctx.add(Expr::Div(new_num, d1));
                     
                     // Complexity check
                     let old_complexity = count_nodes(ctx, expr);
                     let new_complexity = count_nodes(ctx, new_expr);
                     
                     if new_complexity < old_complexity {
                        return Some(Rewrite {
                            new_expr,
                            description: "Add fractions with negated denominator".to_string(),
                        });
                     }
                }
            }

            let new_expr = if let (Ok(p_d1), Ok(p_d2), Ok(p_n1), Ok(p_n2)) = (p_d1_res, p_d2_res, p_n1_res, p_n2_res) {
                // Polynomial path (with simplification)
                let gcd_den = p_d1.gcd(&p_d2);
                let (lcm_poly, _) = p_d1.mul(&p_d2).div_rem(&gcd_den); 
                
                let (m1_poly, _) = p_d2.div_rem(&gcd_den);
                let (m2_poly, _) = p_d1.div_rem(&gcd_den);
                
                let term1 = p_n1.mul(&m1_poly);
                let term2 = p_n2.mul(&m2_poly);
                let new_num_poly = term1.add(&term2);
                
                let common = new_num_poly.gcd(&lcm_poly);
                let (final_num_poly, _) = new_num_poly.div_rem(&common);
                let (final_den_poly, _) = lcm_poly.div_rem(&common);
                
                let new_num = final_num_poly.to_expr(ctx);
                let new_den = final_den_poly.to_expr(ctx);
                ctx.add(Expr::Div(new_num, new_den))
            } else {
                // Fallback path disabled to prevent infinite loops with NestedFractionRule
                return None;
            };
            
            // Complexity check
            let old_complexity = count_nodes(ctx, expr);
            let new_complexity = count_nodes(ctx, new_expr);
            
            // println!("AddFractions: {} -> {}", cas_ast::DisplayExpr { context: ctx, id: expr }, cas_ast::DisplayExpr { context: ctx, id: new_expr });
            // println!("Complexity: {} -> {}", old_complexity, new_complexity);

            if new_complexity < old_complexity {
                // eprintln!("AddFractionsRule rewriting (fallback): {:?} -> {:?}", expr, new_expr);
                return Some(Rewrite {
                    new_expr,
                    description: "Add fractions (fallback/simplified)".to_string(),
                });
            } else {
                 // println!("Rejected due to complexity increase");
            }
        }
        None
    }
);


define_rule!(
    SimplifyMulDivRule,
    "Simplify Multiplication with Division",
    |ctx, expr| {
        // eprintln!("SimplifyMulDivRule checking: {:?}", expr);
        let expr_data = ctx.get(expr).clone();
        if let Expr::Mul(l, r) = expr_data {
            let one = ctx.num(1); // Pre-calculate to avoid mutable borrow in closure

            // Helper to check if expr is Div or Pow(..., -1) or Mul(..., Pow(..., -1))
            // Now only captures ctx immutably
            let get_num_den = |e: ExprId| -> Option<(ExprId, ExprId)> {
                match ctx.get(e) {
                    Expr::Div(n, d) => Some((*n, *d)),
                    Expr::Pow(b, exp) => {
                        if let Expr::Number(n) = ctx.get(*exp) {
                            if n.is_integer() && *n == num_rational::BigRational::from_integer((-1).into()) {
                                return Some((one, *b));
                            }
                        }
                        None
                    },
                    Expr::Mul(ml, mr) => {
                         // Check ml * mr^-1
                        if let Expr::Pow(b, e) = ctx.get(*mr) {
                             if let Expr::Number(n) = ctx.get(*e) {
                                if n.is_integer() && *n == num_rational::BigRational::from_integer((-1).into()) {
                                    return Some((*ml, *b));
                                }
                            }
                        }
                        // Check mr * ml^-1
                        if let Expr::Pow(b, e) = ctx.get(*ml) {
                             if let Expr::Number(n) = ctx.get(*e) {
                                if n.is_integer() && *n == num_rational::BigRational::from_integer((-1).into()) {
                                    return Some((*mr, *b));
                                }
                            }
                        }
                        None
                    },
                    _ => None
                }
            };

            // Check for (a/b) * (c/d)
            let gd_l = get_num_den(l);
            let gd_r = get_num_den(r);
            // eprintln!("  Mul operands: l={:?}, r={:?}", l, r);
            // eprintln!("  get_num_den(l)={:?}, get_num_den(r)={:?}", gd_l, gd_r);

            if let (Some((n1, d1)), Some((n2, d2))) = (gd_l, gd_r) {
                // eprintln!("SimplifyMulDivRule MATCHED: {:?} * {:?}", l, r);
                let new_num = ctx.add(Expr::Mul(n1, n2));
                let new_den = ctx.add(Expr::Mul(d1, d2));
                let new_expr = ctx.add(Expr::Div(new_num, new_den));
                // eprintln!("SimplifyFractionRule rewriting: {:?} -> {:?}", expr, new_expr);
                return Some(Rewrite {
                    new_expr,
                    description: "Simplify fraction (GCD)".to_string(),
                });
            }

            // Check for a * (b/c) or (b/c) * a
            // We need to be careful about borrowing ctx mutably for add/Mul/Div
            // So we collect info first, then mutate.
            
            // Check for a * (b/c) or (b/c) * a
            // We need to be careful about borrowing ctx mutably for add/Mul/Div
            // So we collect info first, then mutate.
            
            if let Some((num, den)) = gd_l {
                // (num / den) * r
                if crate::ordering::compare_expr(ctx, den, r) == std::cmp::Ordering::Equal {
                    return Some(Rewrite {
                        new_expr: num,
                        description: "Cancel division: (a/b)*b -> a".to_string(),
                    });
                }
                let new_num = ctx.add(Expr::Mul(num, r));
                let new_expr = ctx.add(Expr::Div(new_num, den));
                
                // Avoid combining if r is a number or constant (prefer c * (a/b) for CombineLikeTerms)
                if matches!(ctx.get(r), Expr::Number(_) | Expr::Constant(_)) {
                    return None;
                }

                return Some(Rewrite {
                    new_expr,
                    description: "Combine Mul and Div".to_string(),
                });
            }

            if let Some((num, den)) = gd_r {
                // l * (num / den)
                if crate::ordering::compare_expr(ctx, den, l) == std::cmp::Ordering::Equal {
                    return Some(Rewrite {
                        new_expr: num,
                        description: "Cancel division: a*(b/a) -> b".to_string(),
                    });
                }
                
                // Avoid combining if l is a number or constant
                if matches!(ctx.get(l), Expr::Number(_) | Expr::Constant(_)) {
                    return None;
                }

                let new_num = ctx.add(Expr::Mul(l, num));
                let new_expr = ctx.add(Expr::Div(new_num, den));
                return Some(Rewrite {
                    new_expr,
                    description: "Combine Mul and Div".to_string(),
                });
            }
        }
        None
    }
);

define_rule!(
    RationalizeDenominatorRule,
    "Rationalize Denominator",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        
        // Helper to extract num/den from Div, Pow(x, -1), or Mul(x, Pow(y, -1))
        let (num, den) = match expr_data {
            Expr::Div(n, d) => (n, d),
            Expr::Pow(b, e) => {
                if let Expr::Number(n) = ctx.get(e) {
                    if n.is_integer() && *n == num_rational::BigRational::from_integer((-1).into()) {
                        (ctx.num(1), b)
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            },
            Expr::Mul(l, r) => {
                // Check l * r^-1
                if let Expr::Pow(b, e) = ctx.get(r) {
                     if let Expr::Number(n) = ctx.get(*e) {
                        if n.is_integer() && *n == num_rational::BigRational::from_integer((-1).into()) {
                            (l, *b)
                        } else {
                            // Check r * l^-1
                            if let Expr::Pow(b_l, e_l) = ctx.get(l) {
                                if let Expr::Number(n_l) = ctx.get(*e_l) {
                                    if n_l.is_integer() && *n_l == num_rational::BigRational::from_integer((-1).into()) {
                                        (r, *b_l)
                                    } else {
                                        return None;
                                    }
                                } else {
                                    return None;
                                }
                            } else {
                                return None;
                            }
                        }
                    } else {
                         // Check r * l^-1
                        if let Expr::Pow(b_l, e_l) = ctx.get(l) {
                            if let Expr::Number(n_l) = ctx.get(*e_l) {
                                if n_l.is_integer() && *n_l == num_rational::BigRational::from_integer((-1).into()) {
                                    (r, *b_l)
                                } else {
                                    return None;
                                }
                            } else {
                                return None;
                            }
                        } else {
                            return None;
                        }
                    }
                } else {
                    // Check r * l^-1
                    if let Expr::Pow(b_l, e_l) = ctx.get(l) {
                        if let Expr::Number(n_l) = ctx.get(*e_l) {
                            if n_l.is_integer() && *n_l == num_rational::BigRational::from_integer((-1).into()) {
                                (r, *b_l)
                            } else {
                                return None;
                            }
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                }
            },
            _ => return None,
        };
        
        // Check if denominator has roots
        // We look for Add(a, b) or Sub(a, b) where one or both involve roots.
        // Simple case: sqrt(x) + 1, sqrt(x) - 1, sqrt(x) + sqrt(y)
        
        let den_data = ctx.get(den).clone();
        let (l, r, is_add) = match den_data {
            Expr::Add(l, r) => (l, r, true),
            Expr::Sub(l, r) => (l, r, false),
            _ => return None,
        };

        // Check for roots
        let has_root = |e: ExprId| -> bool {
            match ctx.get(e) {
                Expr::Pow(_, exp) => {
                    if let Expr::Number(n) = ctx.get(*exp) {
                        !n.is_integer()
                    } else {
                        false
                    }
                },
                Expr::Function(name, _) => name == "sqrt",
                Expr::Mul(_, _) => false, // Simplified
                _ => false
            }
        };

        let l_root = has_root(l);
        let r_root = has_root(r);
        
        // eprintln!("Rationalize Check: {:?} has_root(l)={} has_root(r)={}", den_data, l_root, r_root);
        // eprintln!("Rationalize Check: {:?} has_root(l)={} has_root(r)={} l={:?} r={:?}", den_data, l_root, r_root, ctx.get(l), ctx.get(r));

        if !l_root && !r_root {
            return None;
        }

        // Construct conjugate
        let conjugate = if is_add {
            ctx.add(Expr::Sub(l, r))
        } else {
            ctx.add(Expr::Add(l, r))
        };

        // Multiply num by conjugate
        let new_num = ctx.add(Expr::Mul(num, conjugate));
        
        // Compute new den = l^2 - r^2
        // (l+r)(l-r) = l^2 - r^2
        // (l-r)(l+r) = l^2 - r^2
        let two = ctx.num(2);
        let l_sq = ctx.add(Expr::Pow(l, two));
        let r_sq = ctx.add(Expr::Pow(r, two));
        let new_den = ctx.add(Expr::Sub(l_sq, r_sq));
        
        let new_expr = ctx.add(Expr::Div(new_num, new_den));
        return Some(Rewrite {
            new_expr,
            description: "Rationalize denominator (diff squares)".to_string(),
        });
    }
);

define_rule!(
    CancelCommonFactorsRule,
    "Cancel Common Factors",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        
        // Helper to extract num/den from Div, Pow(x, -1), or Mul(x, Pow(y, -1))
        let (num, den) = match expr_data {
            Expr::Div(n, d) => (n, d),
            Expr::Pow(b, e) => {
                if let Expr::Number(n) = ctx.get(e) {
                    if n.is_integer() && *n == num_rational::BigRational::from_integer((-1).into()) {
                        (ctx.num(1), b)
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            },
            Expr::Mul(l, r) => {
                // Check l * r^-1
                if let Expr::Pow(b, e) = ctx.get(r) {
                     if let Expr::Number(n) = ctx.get(*e) {
                        if n.is_integer() && *n == num_rational::BigRational::from_integer((-1).into()) {
                            (l, *b)
                        } else {
                            // Check r * l^-1
                            if let Expr::Pow(b_l, e_l) = ctx.get(l) {
                                if let Expr::Number(n_l) = ctx.get(*e_l) {
                                    if n_l.is_integer() && *n_l == num_rational::BigRational::from_integer((-1).into()) {
                                        (r, *b_l)
                                    } else {
                                        return None;
                                    }
                                } else {
                                    return None;
                                }
                            } else {
                                return None;
                            }
                        }
                    } else {
                         // Check r * l^-1
                        if let Expr::Pow(b_l, e_l) = ctx.get(l) {
                            if let Expr::Number(n_l) = ctx.get(*e_l) {
                                if n_l.is_integer() && *n_l == num_rational::BigRational::from_integer((-1).into()) {
                                    (r, *b_l)
                                } else {
                                    return None;
                                }
                            } else {
                                return None;
                            }
                        } else {
                            return None;
                        }
                    }
                } else {
                    // Check r * l^-1
                    if let Expr::Pow(b_l, e_l) = ctx.get(l) {
                        if let Expr::Number(n_l) = ctx.get(*e_l) {
                            if n_l.is_integer() && *n_l == num_rational::BigRational::from_integer((-1).into()) {
                                (r, *b_l)
                            } else {
                                return None;
                            }
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                }
            },
            _ => return None,
        };

        // Helper to collect factors
        fn collect_factors(ctx: &Context, e: ExprId) -> Vec<ExprId> {
            let mut factors = Vec::new();
            let mut stack = vec![e];
            while let Some(curr) = stack.pop() {
                if let Expr::Mul(l, r) = ctx.get(curr) {
                    stack.push(*r);
                    stack.push(*l);
                } else {
                    factors.push(curr);
                }
            }
            factors
        }

        let mut num_factors = collect_factors(ctx, num);
        let mut den_factors = collect_factors(ctx, den);
        
        // println!("Cancel Check: NumFactors={:?} DenFactors={:?}", num_factors, den_factors);

        let mut changed = false;
        let mut i = 0;
        while i < num_factors.len() {
            let nf = num_factors[i];
            let mut found = false;
            for j in 0..den_factors.len() {
                let df = den_factors[j];
                
                // Check exact match
                if crate::ordering::compare_expr(ctx, nf, df) == std::cmp::Ordering::Equal {
                    // Found common factor
                    den_factors.remove(j);
                    found = true;
                    changed = true;
                    break;
                }
                
                // Check power cancellation: nf = x^n, df = x^m
                // Case 1: nf = base^n, df = base. (n > 1)
                let nf_pow = if let Expr::Pow(b, e) = ctx.get(nf) { Some((*b, *e)) } else { None };
                if let Some((b, e)) = nf_pow {
                    if crate::ordering::compare_expr(ctx, b, df) == std::cmp::Ordering::Equal {
                        // nf = df^e. Cancel df.
                        // New nf = df^(e-1).
                        // We need to construct new term.
                        if let Expr::Number(n) = ctx.get(e) {
                             let new_exp = n - num_rational::BigRational::one();
                             let new_term = if new_exp.is_one() {
                                 b
                             } else {
                                 let exp_node = ctx.add(Expr::Number(new_exp));
                                 ctx.add(Expr::Pow(b, exp_node))
                             };
                             num_factors[i] = new_term; // Update factor in place
                             den_factors.remove(j); // Remove denominator factor
                             found = false; // We didn't remove num factor entirely, just modified it.
                             changed = true;
                             break;
                        }
                    }
                }
                
                // Case 2: nf = base, df = base^m. (m > 1)
                // Cancel nf from df. df becomes df^(m-1).
                let df_pow = if let Expr::Pow(b, e) = ctx.get(df) { Some((*b, *e)) } else { None };
                if let Some((b, e)) = df_pow {
                    if crate::ordering::compare_expr(ctx, nf, b) == std::cmp::Ordering::Equal {
                        if let Expr::Number(n) = ctx.get(e) {
                             let new_exp = n - num_rational::BigRational::one();
                             let new_term = if new_exp.is_one() {
                                 b
                             } else {
                                 let exp_node = ctx.add(Expr::Number(new_exp));
                                 ctx.add(Expr::Pow(b, exp_node))
                             };
                             den_factors[j] = new_term; // Update den factor
                             found = true; // Remove num factor
                             changed = true;
                             break;
                        }
                    }
                }
                
                // Case 3: nf = base^n, df = base^m.
                if let Some((b_n, e_n)) = nf_pow {
                    if let Some((b_d, e_d)) = df_pow {
                        if crate::ordering::compare_expr(ctx, b_n, b_d) == std::cmp::Ordering::Equal {
                            // Found common base. Compare exponents.
                            if let (Expr::Number(n), Expr::Number(m)) = (ctx.get(e_n), ctx.get(e_d)) {
                                if n > m {
                                    // nf = x^n, df = x^m, n > m.
                                    // Cancel df. nf becomes x^(n-m).
                                    let new_exp = n - m;
                                    let new_term = if new_exp.is_one() {
                                        b_n
                                    } else {
                                        let exp_node = ctx.add(Expr::Number(new_exp));
                                        ctx.add(Expr::Pow(b_n, exp_node))
                                    };
                                    num_factors[i] = new_term;
                                    den_factors.remove(j);
                                    found = false; // Modified num factor, keep checking it against other den factors?
                                    // Actually, we reduced it. We should probably continue checking THIS factor against others?
                                    // But for simplicity, let's say we made progress.
                                    changed = true;
                                    break;
                                } else if m > n {
                                    // nf = x^n, df = x^m, m > n.
                                    // Cancel nf. df becomes x^(m-n).
                                    let new_exp = m - n;
                                    let new_term = if new_exp.is_one() {
                                        b_d
                                    } else {
                                        let exp_node = ctx.add(Expr::Number(new_exp));
                                        ctx.add(Expr::Pow(b_d, exp_node))
                                    };
                                    den_factors[j] = new_term;
                                    found = true; // Removed num factor
                                    changed = true;
                                    break;
                                } else {
                                    // n == m. Exact match handled above?
                                    // Exact match might not catch it if they are different ExprIds but structurally equal.
                                    // But compare_expr should handle it.
                                    // If we are here, just remove both.
                                    den_factors.remove(j);
                                    found = true;
                                    changed = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            if found {
                num_factors.remove(i);
            } else {
                i += 1;
            }
        }
        
        if changed {
            // Reconstruct
            let new_num = if num_factors.is_empty() {
                ctx.num(1)
            } else {
                let mut res = num_factors[0];
                for f in num_factors.iter().skip(1) {
                    res = ctx.add(Expr::Mul(res, *f));
                }
                res
            };
            
            let new_den = if den_factors.is_empty() {
                ctx.num(1)
            } else {
                let mut res = den_factors[0];
                for f in den_factors.iter().skip(1) {
                    res = ctx.add(Expr::Mul(res, *f));
                }
                res
            };
            
            // If den is 1, return num
            if let Expr::Number(n) = ctx.get(new_den) {
                if n.is_one() {
                    return Some(Rewrite {
                        new_expr: new_num,
                        description: "Cancel common factors (to scalar)".to_string(),
                    });
                }
            }
            
            let new_expr = ctx.add(Expr::Div(new_num, new_den));
            return Some(Rewrite {
                new_expr,
                description: "Cancel common factors".to_string(),
            });
        }
        None
    }
);



define_rule!(
    RootDenestingRule,
    "Root Denesting",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();

        // We look for sqrt(A + B) or sqrt(A - B)
        // Also handle Pow(inner, 1/2)
        let inner = if let Expr::Function(name, args) = &expr_data {
            if name == "sqrt" && args.len() == 1 {
                Some(args[0])
            } else {
                None
            }
        } else if let Expr::Pow(b, e) = &expr_data {
            if let Expr::Number(n) = ctx.get(*e) {
                 if *n.numer() == 1.into() && *n.denom() == 2.into() {
                     Some(*b)
                 } else {
                     None
                 }
            } else {
                None
            }
        } else {
            None
        };

        if inner.is_none() { return None; }
        let inner = inner.unwrap();
        let inner_data = ctx.get(inner).clone();
        println!("RootDenesting checking inner: {:?}", inner_data);

        let (a, b, is_add) = match inner_data {
            Expr::Add(l, r) => (l, r, true),
            Expr::Sub(l, r) => (l, r, false),
            _ => return None,
        };
        
        // Helper to identify if a term is C*sqrt(D) or sqrt(D)
        // Returns (Option<C>, D). If C is None, it means 1.
        fn analyze_sqrt_term(ctx: &Context, e: cas_ast::ExprId) -> Option<(Option<cas_ast::ExprId>, cas_ast::ExprId)> {
            match ctx.get(e) {
                Expr::Function(fname, fargs) if fname == "sqrt" && fargs.len() == 1 => {
                    Some((None, fargs[0]))
                },
                Expr::Pow(b, e) => {
                    // Check for b^(3/2) -> b * sqrt(b)
                    if let Expr::Number(n) = ctx.get(*e) {
                        println!("Checking Pow: base={:?}, exp={}", b, n);
                        if *n.numer() == 3.into() && *n.denom() == 2.into() {
                            return Some((Some(*b), *b));
                        }
                    }
                    None
                },
                Expr::Mul(l, r) => {
                    // Helper to check for sqrt/pow(1/2)
                    let is_sqrt = |e: cas_ast::ExprId| -> Option<cas_ast::ExprId> {
                        match ctx.get(e) {
                            Expr::Function(fname, fargs) if fname == "sqrt" && fargs.len() == 1 => Some(fargs[0]),
                            Expr::Pow(b, e) => {
                                if let Expr::Number(n) = ctx.get(*e) {
                                    if *n.numer() == 1.into() && *n.denom() == 2.into() {
                                        return Some(*b);
                                    }
                                }
                                None
                            },
                            _ => None
                        }
                    };

                    // Check l * sqrt(r)
                    if let Some(inner) = is_sqrt(*r) {
                        return Some((Some(*l), inner));
                    }
                    // Check sqrt(l) * r
                    if let Some(inner) = is_sqrt(*l) {
                        return Some((Some(*r), inner));
                    }
                    None
                },
                _ => None
            }
        }

        // We need to determine which is the "rational" part A and which is the "surd" part sqrt(B).
        // Try both permutations.
        
        // We can't use a closure that captures ctx mutably and calls methods on it easily.
        // So we inline the logic or use a macro/helper that takes ctx.
        
        let check_permutation = |ctx: &mut Context, term_a: cas_ast::ExprId, term_b: cas_ast::ExprId| -> Option<crate::rule::Rewrite> {
            // Assume term_a is A, term_b is C*sqrt(D)
            // We need to call analyze_sqrt_term which takes &Context.
            // But we need to mutate ctx later.
            // So we analyze first.
            
            let sqrt_parts = analyze_sqrt_term(ctx, term_b);
            
            if let Some((c_opt, d)) = sqrt_parts {
                // We have sqrt(A +/- C*sqrt(D))
                // Effective B_eff = C^2 * D
                let c = c_opt.unwrap_or_else(|| ctx.num(1));
                
                // We need numerical values to check the condition
                if let (Expr::Number(val_a), Expr::Number(val_c), Expr::Number(val_d)) = (ctx.get(term_a), ctx.get(c), ctx.get(d)) {
                     let val_c2 = val_c * val_c;
                     let val_beff = val_c2 * val_d;
                     let val_a2 = val_a * val_a;
                     let val_delta = val_a2 - val_beff.clone();
                     
                     if val_delta >= num_rational::BigRational::zero() {
                         if val_delta.is_integer() {
                             let int_delta = val_delta.to_integer();
                             let sqrt_delta = int_delta.sqrt();
                             
                             if sqrt_delta.clone() * sqrt_delta.clone() == int_delta {
                                 // Perfect square!
                                 let z_val = ctx.add(Expr::Number(num_rational::BigRational::from_integer(sqrt_delta)));
                                 
                                 // Found Z!
                                 // Result = sqrt((A+Z)/2) +/- sqrt((A-Z)/2)
                                 let two = ctx.num(2);
                                 
                                 let term1_num = ctx.add(Expr::Add(term_a, z_val));
                                 let term1_frac = ctx.add(Expr::Div(term1_num, two));
                                 let term1 = ctx.add(Expr::Function("sqrt".to_string(), vec![term1_frac]));
                                 
                                 let term2_num = ctx.add(Expr::Sub(term_a, z_val));
                                 let term2_frac = ctx.add(Expr::Div(term2_num, two));
                                 let term2 = ctx.add(Expr::Function("sqrt".to_string(), vec![term2_frac]));
                                 
                                 // Check sign of C
                let c_is_negative = if let Expr::Number(n) = ctx.get(c) {
                    n < &num_rational::BigRational::zero()
                } else {
                    false
                };

                // If is_add is true, we have A + C*sqrt(D).
                // If C is negative, effective operation is subtraction.
                // If is_add is false, we have A - C*sqrt(D).
                // If C is negative, effective operation is addition.
                
                let effective_sub = if is_add {
                    c_is_negative
                } else {
                    !c_is_negative
                };

                let new_expr = if effective_sub {
                     ctx.add(Expr::Sub(term1, term2))
                } else {
                     ctx.add(Expr::Add(term1, term2))
                };
                                 
                                 return Some(crate::rule::Rewrite {
                                     new_expr,
                                     description: "Denest square root".to_string(),
                                 });
                             }
                         }
                     }
                }

            }
            None
        };
        
        if let Some(rw) = check_permutation(ctx, a, b) { return Some(rw); }
        if let Some(rw) = check_permutation(ctx, b, a) { return Some(rw); }
        None
    }
);

define_rule!(
    FractionAddRule,
    "Add Fractions",
    |ctx, expr| {
        // a/c + b/c -> (a+b)/c
        let expr_data = ctx.get(expr).clone();
        if let Expr::Add(l, r) = expr_data {
            let l_data = ctx.get(l).clone();
            let r_data = ctx.get(r).clone();
            
            // Case 1: Both are Divs
            if let (Expr::Div(ln, ld), Expr::Div(rn, rd)) = (&l_data, &r_data) {
                if compare_expr(ctx, *ld, *rd) == Ordering::Equal {
                    let new_num = ctx.add(Expr::Add(*ln, *rn));
                    let new_expr = ctx.add(Expr::Div(new_num, *ld));
                    return Some(Rewrite {
                        new_expr,
                        description: "Combine fractions".to_string(),
                    });
                }
            }
            
            // Case 2: One is Div, other is Mul(-1, Div) (Subtraction)
            // A/C - B/C -> (A-B)/C
            if let Expr::Div(ln, ld) = &l_data {
                if let Expr::Mul(m_l, m_r) = &r_data {
                    // Check for -1 * (B/C)
                    let mut neg_div = None;
                    if let Expr::Number(n) = ctx.get(*m_l) {
                        if *n == -BigRational::one() { neg_div = Some(*m_r); }
                    } else if let Expr::Number(n) = ctx.get(*m_r) {
                        if *n == -BigRational::one() { neg_div = Some(*m_l); }
                    }
                    
                    if let Some(nd) = neg_div {
                        if let Expr::Div(rn, rd) = ctx.get(nd) {
                            if compare_expr(ctx, *ld, *rd) == Ordering::Equal {
                                let new_num = ctx.add(Expr::Sub(*ln, *rn));
                                let new_expr = ctx.add(Expr::Div(new_num, *ld));
                                return Some(Rewrite {
                                    new_expr,
                                    description: "Combine fraction subtraction".to_string(),
                                });
                            }
                        }
                    }
                }
            }
            
            // Symmetric case for -B/C + A/C
            if let Expr::Div(rn, rd) = &r_data {
                if let Expr::Mul(m_l, m_r) = &l_data {
                     // Check for -1 * (A/C)
                    let mut neg_div = None;
                    if let Expr::Number(n) = ctx.get(*m_l) {
                        if *n == -BigRational::one() { neg_div = Some(*m_r); }
                    } else if let Expr::Number(n) = ctx.get(*m_r) {
                        if *n == -BigRational::one() { neg_div = Some(*m_l); }
                    }
                    
                    if let Some(nd) = neg_div {
                        if let Expr::Div(ln, ld) = ctx.get(nd) {
                            if compare_expr(ctx, *ld, *rd) == Ordering::Equal {
                                // -A/C + B/C -> (B-A)/C
                                let new_num = ctx.add(Expr::Sub(*rn, *ln));
                                let new_expr = ctx.add(Expr::Div(new_num, *rd));
                                return Some(Rewrite {
                                    new_expr,
                                    description: "Combine fraction subtraction".to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }
        None
    }
);

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(SimplifyFractionRule));
    simplifier.add_rule(Box::new(NestedFractionRule));
    simplifier.add_rule(Box::new(AddFractionsRule));
    simplifier.add_rule(Box::new(SimplifyMulDivRule));
    simplifier.add_rule(Box::new(ExpandRule));
    simplifier.add_rule(Box::new(RationalizeDenominatorRule));
    // simplifier.add_rule(Box::new(FractionAddRule)); // Disabled: Conflicts with DistributeRule (Div)
    simplifier.add_rule(Box::new(FactorRule));
    simplifier.add_rule(Box::new(CancelCommonFactorsRule));
    simplifier.add_rule(Box::new(RootDenestingRule));
    simplifier.add_rule(Box::new(SimplifySquareRootRule));
    simplifier.add_rule(Box::new(PullConstantFromFractionRule));
    // simplifier.add_rule(Box::new(FactorDifferenceSquaresRule)); // Too aggressive for default, causes loops with DistributeRule
}





define_rule!(
    SimplifySquareRootRule,
    "Simplify Square Root",
    |ctx, expr| {
        let arg = if let Expr::Function(name, args) = ctx.get(expr) {
            if name == "sqrt" && args.len() == 1 {
                Some(args[0])
            } else {
                None
            }
        } else if let Expr::Pow(b, e) = ctx.get(expr) {
            if let Expr::Number(n) = ctx.get(*e) {
                if *n.numer() == 1.into() && *n.denom() == 2.into() {
                    Some(*b)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        if let Some(arg) = arg {
            // Only try to factor if argument is Add/Sub (polynomial)
            match ctx.get(arg) {
                Expr::Add(_, _) | Expr::Sub(_, _) => {},
                _ => return None,
            }

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
            }

        None
    }
);

define_rule!(
    PullConstantFromFractionRule,
    "Pull Constant From Fraction",
    |ctx, expr| {
        let (n, d) = if let Expr::Div(n, d) = ctx.get(expr) {
            (*n, *d)
        } else {
            return None;
        };

        let num_data = ctx.get(n).clone();
        if let Expr::Mul(l, r) = num_data {
            // Check if l or r is a number/constant
            let l_is_const = matches!(ctx.get(l), Expr::Number(_) | Expr::Constant(_));
            let r_is_const = matches!(ctx.get(r), Expr::Number(_) | Expr::Constant(_));
            
            if l_is_const {
                // (c * x) / y -> c * (x / y)
                let div = ctx.add(Expr::Div(r, d));
                let new_expr = ctx.add(Expr::Mul(l, div));
                return Some(Rewrite {
                    new_expr,
                    description: "Pull constant from numerator".to_string(),
                });
            } else if r_is_const {
                // (x * c) / y -> c * (x / y)
                let div = ctx.add(Expr::Div(l, d));
                let new_expr = ctx.add(Expr::Mul(r, div));
                return Some(Rewrite {
                    new_expr,
                    description: "Pull constant from numerator".to_string(),
                });
            }
        }
        // Also handle Neg: (-x) / y -> -1 * (x / y)
        if let Expr::Neg(inner) = num_data {
                let minus_one = ctx.num(-1);
                let div = ctx.add(Expr::Div(inner, d));
                let new_expr = ctx.add(Expr::Mul(minus_one, div));
                return Some(Rewrite {
                    new_expr,
                    description: "Pull negation from numerator".to_string(),
                });
        }
        None
    }
);
