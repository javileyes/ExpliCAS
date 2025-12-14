use crate::build::mul2_raw;
use crate::define_rule;
use crate::ordering::compare_expr;
use crate::polynomial::Polynomial;
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{Context, Expr, ExprId};
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Zero};
use num_traits::{Signed, ToPrimitive};
use std::cmp::Ordering;

/// Helper: Build a simple 2-factor product (no normalization).
#[inline]

/// Check if an expression is a binomial (sum or difference of exactly 2 terms)
/// Examples: (a + b), (a - b), (x + (-y))
fn is_binomial(ctx: &Context, e: ExprId) -> bool {
    match ctx.get(e) {
        Expr::Add(_, _) | Expr::Sub(_, _) => true,
        _ => false,
    }
}

define_rule!(DistributeRule, "Distributive Property", |ctx, expr| {
    // Don't distribute if expression is in canonical form (e.g., inside abs() or sqrt())
    // This protects patterns like abs((x-2)(x+2)) from expanding
    if crate::canonical_forms::is_canonical_form(ctx, expr) {
        return None;
    }

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
                || matches!(l_expr, Expr::Div(_, _))
                || (matches!(l_expr, Expr::Variable(_))
                    && crate::rules::algebra::collect_variables(ctx, expr).len() > 1);

            if !should_distribute {
                return None;
            }

            // CRITICAL: Avoid undoing FactorDifferenceSquaresRule
            // If we have (A+B)(A-B), do NOT distribute.
            if is_conjugate(ctx, l, r) {
                return None;
            }

            // CRITICAL: Don't expand binomial*binomial products like (a-b)*(a-c)
            // This preserves factored form for opposite denominator detection
            if is_binomial(ctx, l) && is_binomial(ctx, r) {
                return None;
            }

            // EDUCATIONAL: Don't distribute fractional coefficient over binomial
            // Preserves clean form like 1/2*(√2-1) instead of √2/2 - 1/2
            if let Expr::Number(n) = ctx.get(l) {
                if !n.is_integer() && is_binomial(ctx, r) {
                    return None;
                }
            }

            let ab = smart_mul(ctx, l, b);
            let ac = smart_mul(ctx, l, c);
            let new_expr = ctx.add(Expr::Add(ab, ac));
            return Some(Rewrite {
                new_expr,
                description: "Distribute".to_string(),
                before_local: None,
                after_local: None, domain_assumption: None,
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
                || matches!(r_expr, Expr::Div(_, _))
                || (matches!(r_expr, Expr::Variable(_))
                    && crate::rules::algebra::collect_variables(ctx, expr).len() > 1);

            if !should_distribute {
                return None;
            }

            // CRITICAL: Avoid undoing FactorDifferenceSquaresRule
            if is_conjugate(ctx, l, r) {
                return None;
            }

            // CRITICAL: Don't expand binomial*binomial products (Policy A+)
            // This preserves factored form like (a+b)*(c+d)
            if is_binomial(ctx, l) && is_binomial(ctx, r) {
                return None;
            }

            // EDUCATIONAL: Don't distribute fractional coefficient over binomial
            // Preserves clean form like (√2-1)/2 instead of √2/2 - 1/2
            if let Expr::Number(n) = ctx.get(r) {
                if !n.is_integer() && is_binomial(ctx, l) {
                    return None;
                }
            }

            let ba = smart_mul(ctx, b, r);
            let ca = smart_mul(ctx, c, r);
            let new_expr = ctx.add(Expr::Add(ba, ca));
            return Some(Rewrite {
                new_expr,
                description: "Distribute".to_string(),
                before_local: None,
                after_local: None, domain_assumption: None,
            });
        }
    }

    // Handle Division Distribution: (a + b) / c -> a/c + b/c
    if let Expr::Div(l, r) = expr_data {
        let l_data = ctx.get(l).clone();

        // Helper to check if division simplifies (shares factors) and return factor size
        let get_simplification_reduction = |ctx: &Context, num: ExprId, den: ExprId| -> usize {
            if num == den {
                return cas_ast::count_nodes(ctx, num);
            }

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

            for df in den_factors {
                // Check for structural equality using compare_expr
                let found = num_factors
                    .iter()
                    .any(|nf| compare_expr(ctx, *nf, df) == Ordering::Equal);

                if found {
                    let factor_size = cas_ast::count_nodes(ctx, df);
                    // Factor removed from num and den -> 2 * size
                    let mut reduction = factor_size * 2;
                    // If factor is entire denominator, Div is removed -> +1
                    if df == den {
                        reduction += 1;
                    }
                    return reduction;
                }

                // Check for numeric GCD
                if let Expr::Number(n_den) = ctx.get(df) {
                    let found_numeric = num_factors.iter().any(|nf| {
                        if let Expr::Number(n_num) = ctx.get(*nf) {
                            if n_num.is_integer() && n_den.is_integer() {
                                let num_int = n_num.to_integer();
                                let den_int = n_den.to_integer();
                                if !num_int.is_zero() && !den_int.is_zero() {
                                    let gcd = num_int.gcd(&den_int);
                                    return gcd > One::one();
                                }
                            }
                        }
                        false
                    });
                    if found_numeric {
                        return 1; // Conservative estimate for number simplification
                    }
                }
            }

            // Fallback to Polynomial GCD
            let vars = crate::rules::algebra::collect_variables(ctx, num);
            if vars.is_empty() {
                return 0;
            }

            for var in vars {
                if let (Ok(p_num), Ok(p_den)) = (
                    Polynomial::from_expr(ctx, num, &var),
                    Polynomial::from_expr(ctx, den, &var),
                ) {
                    if p_den.is_zero() {
                        continue;
                    }
                    let gcd = p_num.gcd(&p_den);
                    // println!("DistributeRule Poly GCD check: num={:?} den={:?} var={} gcd={:?}", ctx.get(num), ctx.get(den), var, gcd);
                    if gcd.degree() > 0 || !gcd.leading_coeff().is_one() {
                        // Estimate complexity of GCD
                        // If GCD cancels denominator (degree match), reduction is high
                        if gcd.degree() == p_den.degree() {
                            // Assume denominator is removed (size(den) + 1)
                            return cas_ast::count_nodes(ctx, den) + 1;
                        }
                        // Otherwise, just return 1
                        return 1;
                    }
                }
            }
            0
        };

        if let Expr::Add(a, b) = l_data {
            let red_a = get_simplification_reduction(ctx, a, r);
            let red_b = get_simplification_reduction(ctx, b, r);

            // Only distribute if EITHER term simplifies
            if red_a > 0 || red_b > 0 {
                let ac = ctx.add(Expr::Div(a, r));
                let bc = ctx.add(Expr::Div(b, r));
                let new_expr = ctx.add(Expr::Add(ac, bc));

                // Check complexity to prevent cycles with AddFractionsRule
                let old_complexity = cas_ast::count_nodes(ctx, expr);
                let new_complexity = cas_ast::count_nodes(ctx, new_expr);

                // Allow if predicted complexity (after simplification) is not worse
                if new_complexity <= old_complexity + red_a + red_b {
                    return Some(Rewrite {
                        new_expr,
                        description: "Distribute division (simplifying)".to_string(),
                        before_local: None,
                        after_local: None, domain_assumption: None,
                    });
                }
            }
        }
        if let Expr::Sub(a, b) = l_data {
            let red_a = get_simplification_reduction(ctx, a, r);
            let red_b = get_simplification_reduction(ctx, b, r);

            if red_a > 0 || red_b > 0 {
                let ac = ctx.add(Expr::Div(a, r));
                let bc = ctx.add(Expr::Div(b, r));
                let new_expr = ctx.add(Expr::Sub(ac, bc));

                // Check complexity to prevent cycles with AddFractionsRule
                let old_complexity = cas_ast::count_nodes(ctx, expr);
                let new_complexity = cas_ast::count_nodes(ctx, new_expr);

                // Allow if predicted complexity (after simplification) is not worse
                if new_complexity <= old_complexity + red_a + red_b {
                    return Some(Rewrite {
                        new_expr,
                        description: "Distribute division (simplifying)".to_string(),
                        before_local: None,
                        after_local: None, domain_assumption: None,
                    });
                }
            }
        }
    }
    None
});

fn is_conjugate(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    // Check for (A+B) and (A-B) or (A-B) and (A+B)
    let a_expr = ctx.get(a);
    let b_expr = ctx.get(b);

    match (a_expr, b_expr) {
        (Expr::Add(a1, a2), Expr::Sub(b1, b2)) | (Expr::Sub(b1, b2), Expr::Add(a1, a2)) => {
            // (A+B) vs (A-B)
            // Check if A=A and B=B
            // Or A=B and B=A (commutative)
            let a1 = *a1;
            let a2 = *a2;
            let b1 = *b1;
            let b2 = *b2;

            // Direct match: A+B vs A-B
            if compare_expr(ctx, a1, b1) == Ordering::Equal
                && compare_expr(ctx, a2, b2) == Ordering::Equal
            {
                return true;
            }
            // Commutative A: B+A vs A-B (A matches A, B matches B)
            if compare_expr(ctx, a2, b1) == Ordering::Equal
                && compare_expr(ctx, a1, b2) == Ordering::Equal
            {
                return true;
            }

            // What about -B+A? Canonicalization usually handles this to Sub(A,B) or Add(A, Neg(B)).
            // If we have Add(A, Neg(B)), it's not Sub.
            false
        }
        (Expr::Add(a1, a2), Expr::Add(b1, b2)) => {
            // (A+B) vs (A+(-B)) or ((-B)+A)
            // Check if one term is negation of another
            let a1 = *a1;
            let a2 = *a2;
            let b1 = *b1;
            let b2 = *b2;

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
        }
        _ => false,
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
        }
        _ => false,
    }
}

define_rule!(AnnihilationRule, "Annihilation", |ctx, expr| {
    let expr_data = ctx.get(expr).clone();
    if let Expr::Sub(l, r) = expr_data {
        if compare_expr(ctx, l, r) == Ordering::Equal {
            let zero = ctx.num(0);
            return Some(Rewrite {
                new_expr: zero,
                description: "x - x = 0".to_string(),
                before_local: None,
                after_local: None, domain_assumption: None,
            });
        }
    }
    None
});

define_rule!(CombineLikeTermsRule, "Combine Like Terms", |ctx, expr| {
    // Only try to collect if it's an Add or Mul, as those are the main things collect handles
    // (and Pow for constant folding, but that's handled elsewhere usually)
    let is_add_or_mul = match ctx.get(expr) {
        Expr::Add(_, _) | Expr::Mul(_, _) => true,
        _ => false,
    };

    if is_add_or_mul {
        // CRITICAL: Do NOT apply to expressions containing matrices
        // Matrix addition/subtraction has dedicated rules (MatrixAddRule, MatrixSubRule)
        // and collect() incorrectly simplifies M + M to 2*M
        if contains_matrix(ctx, expr) {
            return None;
        }

        let new_expr = crate::collect::collect(ctx, expr);
        if new_expr != expr {
            // Check if structurally different to avoid infinite loops with ID regeneration
            if crate::ordering::compare_expr(ctx, new_expr, expr) == Ordering::Equal {
                return None;
            }

            return Some(Rewrite {
                new_expr,
                description: "Global Combine Like Terms".to_string(),
                before_local: None,
                after_local: None, domain_assumption: None,
            });
        }
    }
    None
});

/// Helper function to check if an expression contains a matrix
fn contains_matrix(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Matrix { .. } => true,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            contains_matrix(ctx, *l) || contains_matrix(ctx, *r)
        }
        Expr::Neg(e) => contains_matrix(ctx, *e),
        Expr::Function(_, args) => args.iter().any(|arg| contains_matrix(ctx, *arg)),
        _ => false,
    }
}

define_rule!(BinomialExpansionRule, "Binomial Expansion", |ctx, expr| {
    // Skip if expression is in canonical (elegant) form
    // e.g., ((x+1)*(x-1))^2 should stay as is, not expand
    if crate::canonical_forms::is_canonical_form(ctx, expr) {
        return None;
    }

    // (a + b)^n
    let expr_data = ctx.get(expr).clone();
    if let Expr::Pow(base, exp) = expr_data {
        let base_data = ctx.get(base).clone();
        let (a, b) = match base_data {
            Expr::Add(a, b) => (a, b),
            Expr::Sub(a, b) => {
                let neg_b = ctx.add(Expr::Neg(b));
                (a, neg_b)
            }
            _ => return None,
        };

        let exp_data = ctx.get(exp).clone();
        if let Expr::Number(n) = exp_data {
            if n.is_integer() && !n.is_negative() {
                if let Some(n_val) = n.to_integer().to_u32() {
                    // Limit expansion to reasonable size to prevent explosion
                    if (2..=10).contains(&n_val) {
                        // Expand: sum(k=0 to n) (n choose k) * a^(n-k) * b^k
                        let mut terms = Vec::new();
                        for k in 0..=n_val {
                            let coeff = binomial_coeff(n_val, k);
                            let exp_a = n_val - k;
                            let exp_b = k;

                            let term_a = if exp_a == 0 {
                                ctx.num(1)
                            } else if exp_a == 1 {
                                a
                            } else {
                                let e = ctx.num(exp_a as i64);
                                ctx.add(Expr::Pow(a, e))
                            };
                            let term_b = if exp_b == 0 {
                                ctx.num(1)
                            } else if exp_b == 1 {
                                b
                            } else {
                                let e = ctx.num(exp_b as i64);
                                ctx.add(Expr::Pow(b, e))
                            };

                            let mut term = mul2_raw(ctx, term_a, term_b);
                            if coeff > 1 {
                                let c = ctx.num(coeff as i64);
                                term = mul2_raw(ctx, c, term);
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
                            before_local: None,
                            after_local: None, domain_assumption: None,
                        });
                    }
                }
            }
        }
    }
    None
});

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
    use cas_ast::{Context, DisplayExpr};

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

        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        // Should be (2 * x) + (2 * 3)
        // Note: Simplification of 2*3 happens in a later pass by CombineConstantsRule
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "2 * 3 + 2 * x" // Canonical: numbers before variables
        );
    }

    #[test]
    fn test_annihilation() {
        let mut ctx = Context::new();
        let rule = AnnihilationRule;
        let x = ctx.var("x");
        let expr = ctx.add(Expr::Sub(x, x));
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
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

        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "5 * x"
        );

        // x + 2x -> 3x
        let term1 = x;
        let term2 = ctx.add(Expr::Mul(two, x));
        let expr2 = ctx.add(Expr::Add(term1, term2));
        let rewrite2 = rule
            .apply(
                &mut ctx,
                expr2,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite2.new_expr
                }
            ),
            "3 * x"
        );

        // ln(x) + ln(x) -> 2 * ln(x)
        let ln_x = ctx.add(Expr::Function("ln".to_string(), vec![x]));
        let expr3 = ctx.add(Expr::Add(ln_x, ln_x));
        let rewrite3 = rule
            .apply(
                &mut ctx,
                expr3,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        // ln(x) is log(e, x), prints as ln(x)
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite3.new_expr
                }
            ),
            "2 * ln(x)"
        );
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(DistributeRule));
    simplifier.add_rule(Box::new(AnnihilationRule));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(BinomialExpansionRule));
}
