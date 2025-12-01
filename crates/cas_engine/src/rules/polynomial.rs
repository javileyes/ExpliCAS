use crate::rule::Rewrite;
use crate::define_rule;
use cas_ast::{Expr, ExprId, Context};
use num_traits::{ToPrimitive, Signed};
use num_rational::BigRational;
use num_traits::{One, Zero};
use crate::ordering::compare_expr;
use std::cmp::Ordering;
use crate::polynomial::Polynomial;

define_rule!(
    DistributeRule,
    "Distributive Property",
    |ctx, expr| {
        println!("DistributeRule visiting: {:?}", ctx.get(expr));
        let expr_data = ctx.get(expr).clone();
        if let Expr::Mul(l, r) = expr_data {
            // a * (b + c) -> a*b + a*c
            let r_data = ctx.get(r).clone();
            if let Expr::Add(b, c) = r_data {
                // Distribute if 'l' is a Number, Function, Add/Sub, Pow, Mul, or Div.
                // We exclude Var to keep x(x+1) factored, but allow x^2(x+1) to expand.
                let l_expr = ctx.get(l);
                let should_distribute = matches!(l_expr, Expr::Number(_)) 
                    || matches!(l_expr, Expr::Function(_, _))
                    || matches!(l_expr, Expr::Add(_, _))
                    || matches!(l_expr, Expr::Sub(_, _))
                    || matches!(l_expr, Expr::Pow(_, _))
                    || matches!(l_expr, Expr::Mul(_, _))
                    || matches!(l_expr, Expr::Div(_, _));
                
                if !should_distribute {
                    return None;
                }

                // CRITICAL: Avoid undoing FactorDifferenceSquaresRule
                // If we have (A+B)(A-B), do NOT distribute.
                if is_conjugate(ctx, l, r) {
                    return None;
                }
                
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
                // Same logic for 'r'
                let r_expr = ctx.get(r);
                let should_distribute = matches!(r_expr, Expr::Number(_)) 
                    || matches!(r_expr, Expr::Function(_, _))
                    || matches!(r_expr, Expr::Add(_, _))
                    || matches!(r_expr, Expr::Sub(_, _))
                    || matches!(r_expr, Expr::Pow(_, _))
                    || matches!(r_expr, Expr::Mul(_, _))
                    || matches!(r_expr, Expr::Div(_, _));

                if !should_distribute {
                    return None;
                }

                // CRITICAL: Avoid undoing FactorDifferenceSquaresRule
                if is_conjugate(ctx, l, r) {
                    return None;
                }

                let ba = ctx.add(Expr::Mul(b, r));
                let ca = ctx.add(Expr::Mul(c, r));
                let new_expr = ctx.add(Expr::Add(ba, ca));
                return Some(Rewrite {
                    new_expr,
                    description: "Distribute".to_string(),
                });
            }
        }

        // Handle Division Distribution: (a + b) / c -> a/c + b/c
        if let Expr::Div(l, r) = expr_data {
            println!("DistributeRule Div: l={:?} r={:?}", ctx.get(l), ctx.get(r));
            let l_data = ctx.get(l).clone();
            
            // Helper to check if division simplifies (shares factors)
            let simplifies = |ctx: &Context, num: ExprId, den: ExprId| -> bool {
                if num == den { return true; }
                
                // Structural factor check
                let get_factors = |e: ExprId| -> Vec<ExprId> {
                    let mut factors = Vec::new();
                    let mut stack = vec![e];
                    while let Some(curr) = stack.pop() {
                        if let Expr::Mul(a, b) = ctx.get(curr) {
                            stack.push(*a);
                            stack.push(*b);
                        } else {
                            factors.push(curr);
                        }
                    }
                    factors
                };
                
                let num_factors = get_factors(num);
                let den_factors = get_factors(den);
                
                // println!("shares_factor: num={:?} den={:?}", num_factors, den_factors);
                // for f in &num_factors { println!("  num factor: {:?}", ctx.get(*f)); }
                // for f in &den_factors { println!("  den factor: {:?}", ctx.get(*f)); }
                
                println!("shares_factor: num={:?} den={:?}", num_factors, den_factors);
                for f in &num_factors { println!("  num factor: {:?} args: {:?}", ctx.get(*f), if let Expr::Function(_, args) = ctx.get(*f) { args.iter().map(|a| ctx.get(*a)).collect::<Vec<_>>() } else { vec![] }); }
                for f in &den_factors { println!("  den factor: {:?} args: {:?}", ctx.get(*f), if let Expr::Function(_, args) = ctx.get(*f) { args.iter().map(|a| ctx.get(*a)).collect::<Vec<_>>() } else { vec![] }); }

                for df in den_factors {
                    // Check for structural equality using compare_expr
                    let found = num_factors.iter().any(|nf| {
                        compare_expr(ctx, *nf, df) == Ordering::Equal
                    });
                    
                    if found {
                        // println!("  Found common factor: {:?}", ctx.get(df));
                        return true;
                    }
                }
                
                // Fallback to Polynomial GCD for polynomial cases (like (x^2+x)/x)
                // Only if structural check failed
                let vars = crate::rules::algebra::collect_variables(ctx, num);
                if vars.is_empty() { return false; } 
                
                for var in vars {
                    if let (Ok(p_num), Ok(p_den)) = (Polynomial::from_expr(ctx, num, &var), Polynomial::from_expr(ctx, den, &var)) {
                        if p_den.is_zero() { continue; }
                        let gcd = p_num.gcd(&p_den);
                        if gcd.degree() > 0 || !gcd.leading_coeff().is_one() {
                            return true;
                        }
                    }
                }
                false
            };

            if let Expr::Add(a, b) = l_data {
                // Only distribute if at least one term simplifies
                if simplifies(ctx, a, r) || simplifies(ctx, b, r) {
                    let ac = ctx.add(Expr::Div(a, r));
                    let bc = ctx.add(Expr::Div(b, r));
                    let new_expr = ctx.add(Expr::Add(ac, bc));
                    return Some(Rewrite {
                        new_expr,
                        description: "Distribute division (simplifying)".to_string(),
                    });
                }
            }
            if let Expr::Sub(a, b) = l_data {
                if simplifies(ctx, a, r) || simplifies(ctx, b, r) {
                    let ac = ctx.add(Expr::Div(a, r));
                    let bc = ctx.add(Expr::Div(b, r));
                    let new_expr = ctx.add(Expr::Sub(ac, bc));
                    return Some(Rewrite {
                        new_expr,
                        description: "Distribute division (simplifying)".to_string(),
                    });
                }
            }
        }
        None
    }
);

fn is_conjugate(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    // Check for (A+B) and (A-B) or (A-B) and (A+B)
    let a_expr = ctx.get(a);
    let b_expr = ctx.get(b);

    match (a_expr, b_expr) {
        (Expr::Add(a1, a2), Expr::Sub(b1, b2)) | (Expr::Sub(b1, b2), Expr::Add(a1, a2)) => {
            // (A+B) vs (A-B)
            // Check if A=A and B=B
            // Or A=B and B=A (commutative)
            let a1 = *a1; let a2 = *a2;
            let b1 = *b1; let b2 = *b2;
            
            // Direct match: A+B vs A-B
            if compare_expr(ctx, a1, b1) == Ordering::Equal && compare_expr(ctx, a2, b2) == Ordering::Equal {
                return true;
            }
            // Commutative A: B+A vs A-B (A matches A, B matches B)
            if compare_expr(ctx, a2, b1) == Ordering::Equal && compare_expr(ctx, a1, b2) == Ordering::Equal {
                return true;
            }
            
            // What about -B+A? Canonicalization usually handles this to Sub(A,B) or Add(A, Neg(B)).
            // If we have Add(A, Neg(B)), it's not Sub.
            false
        },
        (Expr::Add(a1, a2), Expr::Add(b1, b2)) => {
            // (A+B) vs (A+(-B)) or ((-B)+A)
            // Check if one term is negation of another
            let a1 = *a1; let a2 = *a2;
            let b1 = *b1; let b2 = *b2;
            
            // Case 1: b2 is neg(a2) -> (A+B)(A-B)
            if is_negation(ctx, a2, b2) && compare_expr(ctx, a1, b1) == Ordering::Equal {
                return true;
            }
            // Case 2: b1 is neg(a2) -> (A+B)(-B+A)
            if is_negation(ctx, a2, b1) && compare_expr(ctx, a1, b2) == Ordering::Equal {
                return true;
            }
            // Case 3: b2 is neg(a1) -> (A+B)(B-A) -> No, that's -(A-B)(A+B)? No.
            // (A+B)(B-A) = B^2 - A^2. This IS a conjugate pair.
            if is_negation(ctx, a1, b2) && compare_expr(ctx, a2, b1) == Ordering::Equal {
                return true;
            }
            // Case 4: b1 is neg(a1) -> (A+B)(-A+B)
            if is_negation(ctx, a1, b1) && compare_expr(ctx, a2, b2) == Ordering::Equal {
                return true;
            }
            false
        },
        _ => false
    }
}

fn is_negation(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    // Check if b is Neg(a) or Mul(-1, a)
    if check_negation_structure(ctx, b, a) {
        return true;
    }
    // Check if a is Neg(b) or Mul(-1, b)
    if check_negation_structure(ctx, a, b) {
        return true;
    }
    false
}

fn check_negation_structure(ctx: &Context, potential_neg: ExprId, original: ExprId) -> bool {
    match ctx.get(potential_neg) {
        Expr::Neg(n) => compare_expr(ctx, original, *n) == Ordering::Equal,
        Expr::Mul(l, r) => {
            // Check for -1 * original
            if let Expr::Number(n) = ctx.get(*l) {
                if *n == -BigRational::one() && compare_expr(ctx, *r, original) == Ordering::Equal {
                    return true;
                }
            }
            // Check for original * -1
            if let Expr::Number(n) = ctx.get(*r) {
                if *n == -BigRational::one() && compare_expr(ctx, *l, original) == Ordering::Equal {
                    return true;
                }
            }
            false
        },
        _ => false
    }
}

define_rule!(
    DistributeConstantRule,
    "Distribute Constant",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Mul(l, r) = expr_data {
            // c * (a + b) -> c*a + c*b where c is a number
            let l_is_num = matches!(ctx.get(l), Expr::Number(_));
            let r_is_num = matches!(ctx.get(r), Expr::Number(_));

            if l_is_num {
                let r_data = ctx.get(r).clone();
                if let Expr::Add(a, b) = r_data {
                    let ca = ctx.add(Expr::Mul(l, a));
                    let cb = ctx.add(Expr::Mul(l, b));
                    let new_expr = ctx.add(Expr::Add(ca, cb));
                    return Some(Rewrite {
                        new_expr,
                        description: "Distribute Constant".to_string(),
                    });
                }
            }
            
            if r_is_num {
                let l_data = ctx.get(l).clone();
                if let Expr::Add(a, b) = l_data {
                    let ac = ctx.add(Expr::Mul(a, r));
                    let bc = ctx.add(Expr::Mul(b, r));
                    let new_expr = ctx.add(Expr::Add(ac, bc));
                    return Some(Rewrite {
                        new_expr,
                        description: "Distribute Constant".to_string(),
                    });
                }
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
        if let Expr::Add(_, _) = ctx.get(expr) {
            // Flatten
            let mut terms = Vec::new();
            flatten_add(ctx, expr, &mut terms);
            
            if terms.len() < 2 { return None; }

            // Extract (coeff, var_part)
            let mut parsed_terms = Vec::new();
            for t in terms {
                let (c, v) = get_parts(ctx, t);
                parsed_terms.push((c, v));
            }
            
            println!("CombineLikeTerms: {:?}", parsed_terms);
            for (c, v) in &parsed_terms {
                println!("  Term: coeff={:?}, var={:?} -> {:?}", c, v, ctx.get(*v));
            }

            // Sort by var_part to bring like terms together
            parsed_terms.sort_by(|a, b| compare_expr(ctx, a.1, b.1));
            
            // Combine
            let mut new_terms = Vec::new();
            if parsed_terms.is_empty() { return None; } // Should not happen

            let mut current_coeff = parsed_terms[0].0.clone();
            let mut current_var = parsed_terms[0].1;
            let mut changed = false;
            
            for i in 1..parsed_terms.len() {
                let (c, v) = &parsed_terms[i];
                if compare_expr(ctx, current_var, *v) == Ordering::Equal {
                    current_coeff += c;
                    changed = true;
                } else {
                    if !current_coeff.is_zero() {
                        new_terms.push(make_term(ctx, current_coeff, current_var));
                    } else {
                        changed = true; // Zero removed
                    }
                    current_coeff = c.clone();
                    current_var = *v;
                }
            }
            if !current_coeff.is_zero() {
                new_terms.push(make_term(ctx, current_coeff, current_var));
            } else {
                changed = true;
            }
            
            if !changed {
                return None;
            }
            
            if new_terms.is_empty() {
                return Some(Rewrite {
                    new_expr: ctx.num(0),
                    description: "Combine like terms (all cancelled)".to_string(),
                });
            }
            
            // Rebuild Add chain (left-associative)
            let mut res = new_terms[0];
            for t in new_terms.iter().skip(1) {
                res = ctx.add(Expr::Add(res, *t));
            }
            
            return Some(Rewrite {
                new_expr: res,
                description: "Global Combine Like Terms".to_string(),
            });
        }
        None
    }
);

fn flatten_add(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            flatten_add(ctx, *l, terms);
            flatten_add(ctx, *r, terms);
        }
        _ => terms.push(expr),
    }
}

fn get_parts(context: &mut Context, e: ExprId) -> (BigRational, ExprId) {
    match context.get(e) {
        Expr::Mul(a, b) => {
            if let Expr::Number(n) = context.get(*a) {
                (n.clone(), *b)
            } else if let Expr::Number(n) = context.get(*b) {
                (n.clone(), *a)
            } else {
                (BigRational::one(), e)
            }
        }
        Expr::Number(n) => (n.clone(), context.num(1)), // Treat constant as c * 1
        _ => (BigRational::one(), e),
    }
}

fn make_term(ctx: &mut Context, coeff: BigRational, var: ExprId) -> ExprId {
    if let Expr::Number(n) = ctx.get(var) {
        if n.is_one() {
            return ctx.add(Expr::Number(coeff));
        }
    }
    
    if coeff.is_one() {
        var
    } else {
        let c = ctx.add(Expr::Number(coeff));
        ctx.add(Expr::Mul(c, var))
    }
}


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
