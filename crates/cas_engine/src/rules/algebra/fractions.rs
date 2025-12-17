use crate::build::mul2_raw;
use crate::define_rule;
use crate::multipoly::{multipoly_from_expr, multipoly_to_expr, MultiPoly, PolyBudget};
use crate::phase::PhaseMask;
use crate::polynomial::Polynomial;
use crate::rule::Rewrite;
use cas_ast::count_nodes;
use cas_ast::{Context, DisplayExpr, Expr, ExprId};
use num_traits::{One, Signed, Zero};
use std::cmp::Ordering;

use super::helpers::*;

// =============================================================================
// Multivariate GCD (Layer 1: monomial + content)
// =============================================================================

/// Try to compute GCD of two expressions using multivariate polynomial representation.
/// Returns None if expressions can't be converted to polynomials or if GCD is trivial (1).
/// Returns Some((quotient_num, quotient_den, gcd_expr)) if non-trivial GCD found.
fn try_multivar_gcd(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
) -> Option<(ExprId, ExprId, ExprId)> {
    let budget = PolyBudget::default();

    // Try to convert both to MultiPoly
    let p_num = multipoly_from_expr(ctx, num, &budget).ok()?;
    let p_den = multipoly_from_expr(ctx, den, &budget).ok()?;

    // Skip if not multivariate (let univariate path handle it)
    if p_num.vars.len() <= 1 {
        return None;
    }

    // Align variables if needed
    if p_num.vars != p_den.vars {
        return None; // For now, require same vars
    }

    // Layer 1: Monomial GCD
    let mono_gcd = p_num.monomial_gcd_with(&p_den).ok()?;
    let has_mono_gcd = mono_gcd.iter().any(|&e| e > 0);

    // Layer 1: Content GCD
    let content_num = p_num.content();
    let content_den = p_den.content();
    let content_gcd = gcd_rational(content_num.clone(), content_den.clone());
    let has_content_gcd = !content_gcd.is_one();

    // If no GCD found at Layer 1, skip
    if !has_mono_gcd && !has_content_gcd {
        return None;
    }

    // Divide by monomial GCD
    let (p_num, p_den) = if has_mono_gcd {
        (
            p_num.div_monomial_exact(&mono_gcd)?,
            p_den.div_monomial_exact(&mono_gcd)?,
        )
    } else {
        (p_num, p_den)
    };

    // Divide by content GCD
    let (p_num, p_den) = if has_content_gcd {
        (
            p_num.div_scalar_exact(&content_gcd)?,
            p_den.div_scalar_exact(&content_gcd)?,
        )
    } else {
        (p_num, p_den)
    };

    // Build GCD expression (monomial * content)
    let mut gcd_poly = MultiPoly::one(p_num.vars.clone());

    if has_mono_gcd {
        gcd_poly = gcd_poly.mul_monomial(&mono_gcd).ok()?;
    }

    if has_content_gcd {
        gcd_poly = gcd_poly.mul_scalar(&content_gcd);
    }

    // Convert back to expressions
    let new_num = multipoly_to_expr(&p_num, ctx);
    let new_den = multipoly_to_expr(&p_den, ctx);
    let gcd_expr = multipoly_to_expr(&gcd_poly, ctx);

    Some((new_num, new_den, gcd_expr))
}

// ========== Helper to extract fraction parts from both Div and Mul(1/n,x) ==========
// This is needed because canonicalization may convert Div(x,n) to Mul(1/n,x)

/// Extract (numerator, denominator, is_fraction) from an expression.
/// Recognizes:
/// - Div(num, den) → (num, den, true)
/// - Mul(Number(1/n), x) or Mul(x, Number(1/n)) → (x, n, true) where numerator of coeff is ±1
/// - Mul(Div(1,den), x) or Mul(x, Div(1,den)) → (x, den, true) for symbolic denominators
/// - anything else → (expr, 1, false)
fn extract_as_fraction(ctx: &mut Context, expr: ExprId) -> (ExprId, ExprId, bool) {
    use num_bigint::BigInt;
    use num_rational::BigRational;
    use num_traits::Signed;

    let expr_data = ctx.get(expr).clone();

    // Case 1: Direct Div
    if let Expr::Div(num, den) = expr_data {
        return (num, den, true);
    }

    // Case 2 & 3: Mul with fractional coefficient
    if let Expr::Mul(l, r) = expr_data {
        // Helper to check if a Number is ±1/n and extract denominator
        let check_unit_fraction = |n: &BigRational| -> Option<(BigInt, bool)> {
            if n.is_integer() {
                return None;
            }
            let numer = n.numer();
            let abs_numer: BigInt = if numer < &BigInt::from(0) {
                -numer.clone()
            } else {
                numer.clone()
            };
            if abs_numer == BigInt::from(1) {
                let is_negative = numer.is_negative();
                return Some((n.denom().clone(), is_negative));
            }
            None
        };

        // Helper to check if expression is Div(1, den) or Div(-1, den)
        let check_unit_div = |factor: ExprId| -> Option<(ExprId, bool)> {
            if let Expr::Div(num, den) = ctx.get(factor).clone() {
                if let Expr::Number(n) = ctx.get(num) {
                    if n.is_integer() {
                        let n_val = n.numer();
                        if *n_val == BigInt::from(1) {
                            return Some((den, false));
                        } else if *n_val == BigInt::from(-1) {
                            return Some((den, true));
                        }
                    }
                }
            }
            None
        };

        // Case 2: Check for Number(±1/n)
        if let Expr::Number(n) = ctx.get(l).clone() {
            if let Some((denom, is_neg)) = check_unit_fraction(&n) {
                let denom_expr = ctx.add(Expr::Number(BigRational::from_integer(denom)));
                if is_neg {
                    let neg_r = ctx.add(Expr::Neg(r));
                    return (neg_r, denom_expr, true);
                }
                return (r, denom_expr, true);
            }
        }
        if let Expr::Number(n) = ctx.get(r).clone() {
            if let Some((denom, is_neg)) = check_unit_fraction(&n) {
                let denom_expr = ctx.add(Expr::Number(BigRational::from_integer(denom)));
                if is_neg {
                    let neg_l = ctx.add(Expr::Neg(l));
                    return (neg_l, denom_expr, true);
                }
                return (l, denom_expr, true);
            }
        }

        // Case 3: Check for Div(1, den) or Div(-1, den) - symbolic denominator
        if let Some((den, is_neg)) = check_unit_div(l) {
            if is_neg {
                let neg_r = ctx.add(Expr::Neg(r));
                return (neg_r, den, true);
            }
            return (r, den, true);
        }
        if let Some((den, is_neg)) = check_unit_div(r) {
            if is_neg {
                let neg_l = ctx.add(Expr::Neg(l));
                return (neg_l, den, true);
            }
            return (l, den, true);
        }
    }

    // Not a recognized fraction form
    let one = ctx.num(1);
    (expr, one, false)
}

// ========== Micro-API for safe Mul construction ==========
// Use this instead of ctx.add(Expr::Mul(...)) in this file.

define_rule!(
    SimplifyFractionRule,
    "Simplify Nested Fraction",
    |ctx, expr| {
        use cas_ast::views::RationalFnView;

        // Use RationalFnView to detect any fraction form while preserving structure
        let view = RationalFnView::from(ctx, expr)?;
        let (num, den) = (view.num, view.den);

        // 0. Try multivariate GCD first (Layer 1: monomial + content)
        let vars = collect_variables(ctx, expr);
        if vars.len() > 1 {
            if let Some((new_num, new_den, gcd_expr)) = try_multivar_gcd(ctx, num, den) {
                // Build factored form for display
                let factored_num = mul2_raw(ctx, new_num, gcd_expr);
                let factored_den = if let Expr::Number(n) = ctx.get(new_den) {
                    if n.is_one() {
                        gcd_expr
                    } else {
                        mul2_raw(ctx, new_den, gcd_expr)
                    }
                } else {
                    mul2_raw(ctx, new_den, gcd_expr)
                };
                let factored_form = ctx.add(Expr::Div(factored_num, factored_den));

                // If denominator is 1, return just numerator
                if let Expr::Number(n) = ctx.get(new_den) {
                    if n.is_one() {
                        return Some(Rewrite {
                            new_expr: new_num,
                            description: format!(
                                "Simplified fraction by GCD: {}",
                                DisplayExpr {
                                    context: ctx,
                                    id: gcd_expr
                                }
                            ),
                            before_local: Some(factored_form),
                            after_local: Some(new_num),
                            domain_assumption: None,
                        });
                    }
                }

                let result = ctx.add(Expr::Div(new_num, new_den));
                return Some(Rewrite {
                    new_expr: result,
                    description: format!(
                        "Simplified fraction by GCD: {}",
                        DisplayExpr {
                            context: ctx,
                            id: gcd_expr
                        }
                    ),
                    before_local: Some(factored_form),
                    after_local: Some(result),
                    domain_assumption: None,
                });
            }
        }

        // 1. Univariate path: require single variable
        if vars.len() != 1 {
            return None;
        }
        let var = vars.iter().next().unwrap();

        // 2. Convert to Polynomials
        let p_num = Polynomial::from_expr(ctx, num, var).ok()?;
        let p_den = Polynomial::from_expr(ctx, den, var).ok()?;

        if p_den.is_zero() {
            return None;
        }

        // 3. Compute Polynomial GCD (monic)
        let poly_gcd = p_num.gcd(&p_den);

        // 4. Compute Numeric Content GCD
        // Polynomial GCD is monic, so it misses numeric factors like 27x^3 / 9 -> gcd=9
        let content_num = p_num.content();
        let content_den = p_den.content();

        // Helper to compute GCD of two rationals (assuming integers for now)
        let numeric_gcd = gcd_rational(content_num, content_den);

        // 5. Combine
        // full_gcd = poly_gcd * numeric_gcd
        let scalar = Polynomial::new(vec![numeric_gcd], var.to_string());
        let full_gcd = poly_gcd.mul(&scalar);

        // 6. Check if GCD is non-trivial
        // If degree is 0 and constant is 1, it's trivial.
        if full_gcd.degree() == 0 && full_gcd.leading_coeff().is_one() {
            return None;
        }

        // 7. Divide
        let (new_num_poly, rem_num) = p_num.div_rem(&full_gcd);
        let (new_den_poly, rem_den) = p_den.div_rem(&full_gcd);

        if !rem_num.is_zero() || !rem_den.is_zero() {
            return None;
        }

        let new_num = new_num_poly.to_expr(ctx);
        let new_den = new_den_poly.to_expr(ctx);
        let gcd_expr = full_gcd.to_expr(ctx);

        // Build factored form for "Rule:" display: (new_num * gcd) / (new_den * gcd)
        // This shows the factorization step more clearly
        let factored_num = mul2_raw(ctx, new_num, gcd_expr);
        let factored_den = if let Expr::Number(n) = ctx.get(new_den) {
            if n.is_one() {
                gcd_expr // denominator is just the GCD
            } else {
                mul2_raw(ctx, new_den, gcd_expr)
            }
        } else {
            mul2_raw(ctx, new_den, gcd_expr)
        };
        let factored_form = ctx.add(Expr::Div(factored_num, factored_den));

        // If denominator is 1, return numerator
        if let Expr::Number(n) = ctx.get(new_den) {
            if n.is_one() {
                return Some(Rewrite {
                    new_expr: new_num,
                    description: format!(
                        "Simplified fraction by GCD: {}",
                        DisplayExpr {
                            context: ctx,
                            id: gcd_expr
                        }
                    ),
                    before_local: Some(factored_form),
                    after_local: Some(new_num),
                    domain_assumption: None,
                });
            }
        }

        let result = ctx.add(Expr::Div(new_num, new_den));
        return Some(Rewrite {
            new_expr: result,
            description: format!(
                "Simplified fraction by GCD: {}",
                DisplayExpr {
                    context: ctx,
                    id: gcd_expr
                }
            ),
            before_local: Some(factored_form),
            after_local: Some(result),
            domain_assumption: None,
        });
    }
);

define_rule!(
    NestedFractionRule,
    "Simplify Complex Fraction",
    |ctx, expr| {
        use cas_ast::views::RationalFnView;

        // Use RationalFnView to detect any fraction form
        let view = RationalFnView::from(ctx, expr)?;
        let (num, den) = (view.num, view.den);

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
            multiplier = mul2_raw(ctx, multiplier, unique_denoms[i]);
        }

        // Multiply num and den by multiplier
        let new_num = distribute(ctx, num, multiplier);
        let new_den = distribute(ctx, den, multiplier);

        let new_expr = ctx.add(Expr::Div(new_num, new_den));

        if new_expr == expr {
            return None;
        }

        // Complexity Check: Ensure we actually reduced the number of divisions or total nodes
        // Counting Div nodes is a good heuristic for "nested fraction simplified"
        let count_divs = |id| count_nodes_of_type(ctx, id, "Div");
        let old_divs = count_divs(expr);
        let new_divs = count_divs(new_expr);

        if new_divs >= old_divs {
            return None;
        }

        return Some(Rewrite {
            new_expr,
            description: "Simplify nested fraction".to_string(),
            before_local: None,
            after_local: None,
            domain_assumption: None,
        });
    }
);

define_rule!(
    SimplifyMulDivRule,
    "Simplify Multiplication with Division",
    |ctx, expr| {
        use cas_ast::views::FractionParts;

        let expr_data = ctx.get(expr).clone();
        if let Expr::Mul(l, r) = expr_data {
            // Use FractionParts to detect any fraction-like structure
            let fp_l = FractionParts::from(&*ctx, l);
            let fp_r = FractionParts::from(&*ctx, r);

            // If neither side has denominators, nothing to do
            if !fp_l.is_fraction() && !fp_r.is_fraction() {
                return None;
            }

            // Check for simple cancellation: (a/b) * b -> a
            // Only for simple cases to avoid over-simplification
            if fp_l.is_fraction() && fp_l.den.len() == 1 && fp_l.den[0].exp == 1 {
                let den_base = fp_l.den[0].base;
                // Check if r equals the denominator
                if crate::ordering::compare_expr(ctx, den_base, r) == std::cmp::Ordering::Equal {
                    // Cancel: (a/b) * b -> a
                    let result = if fp_l.num.is_empty() {
                        ctx.num(fp_l.sign as i64)
                    } else {
                        let num_prod = FractionParts::build_product_static(ctx, &fp_l.num);
                        if fp_l.sign < 0 {
                            ctx.add(Expr::Neg(num_prod))
                        } else {
                            num_prod
                        }
                    };
                    return Some(Rewrite {
                        new_expr: result,
                        description: "Cancel division: (a/b)*b -> a".to_string(),
                        before_local: None,
                        after_local: None,
                        domain_assumption: None,
                    });
                }
            }

            // Check for simple cancellation: a * (b/a) -> b
            if fp_r.is_fraction() && fp_r.den.len() == 1 && fp_r.den[0].exp == 1 {
                let den_base = fp_r.den[0].base;
                if crate::ordering::compare_expr(ctx, den_base, l) == std::cmp::Ordering::Equal {
                    let result = if fp_r.num.is_empty() {
                        ctx.num(fp_r.sign as i64)
                    } else {
                        let num_prod = FractionParts::build_product_static(ctx, &fp_r.num);
                        if fp_r.sign < 0 {
                            ctx.add(Expr::Neg(num_prod))
                        } else {
                            num_prod
                        }
                    };
                    return Some(Rewrite {
                        new_expr: result,
                        description: "Cancel division: a*(b/a) -> b".to_string(),
                        before_local: None,
                        after_local: None,
                        domain_assumption: None,
                    });
                }
            }

            // Avoid combining if either side is just a constant (prefer k * (a/b) for CombineLikeTerms)
            if matches!(ctx.get(l), Expr::Number(_) | Expr::Constant(_))
                || matches!(ctx.get(r), Expr::Number(_) | Expr::Constant(_))
            {
                return None;
            }

            // Combine into single fraction: (n1/d1) * (n2/d2) -> (n1*n2)/(d1*d2)
            // Only do this if at least one side is an actual fraction
            if fp_l.is_fraction() || fp_r.is_fraction() {
                // Build combined numerator: products of all num factors
                let mut combined_num = Vec::new();
                combined_num.extend(fp_l.num.iter().cloned());
                combined_num.extend(fp_r.num.iter().cloned());

                // Build combined denominator
                let mut combined_den = Vec::new();
                combined_den.extend(fp_l.den.iter().cloned());
                combined_den.extend(fp_r.den.iter().cloned());

                let combined_sign = (fp_l.sign as i16 * fp_r.sign as i16) as i8;

                let result_fp = FractionParts {
                    sign: combined_sign,
                    num: combined_num,
                    den: combined_den,
                };

                // Build as division for didactic output
                let new_expr = result_fp.build_as_div(ctx);

                // Avoid no-op rewrites
                if new_expr == expr {
                    return None;
                }

                return Some(Rewrite {
                    new_expr,
                    description: "Combine fractions in multiplication".to_string(),
                    before_local: None,
                    after_local: None,
                    domain_assumption: None,
                });
            }
        }
        None
    }
);

/// Check if one denominator divides the other
/// Returns (new_n1, new_n2, common_den, is_divisible)
///
/// For example:
/// - d1=2, d2=2n → d2 = n·d1, so multiply n1 by n: (n1·n, n2, 2n, true)
/// - d1=2n, d2=2 → d1 = n·d2, so multiply n2 by n: (n1, n2·n, 2n, true)
fn check_divisible_denominators(
    ctx: &mut Context,
    n1: ExprId,
    n2: ExprId,
    d1: ExprId,
    d2: ExprId,
) -> (ExprId, ExprId, ExprId, bool) {
    // Try to find if d2 = k * d1 (d1 divides d2)
    if let Some(k) = try_extract_factor(ctx, d2, d1) {
        // d2 = k * d1, so use d2 as common denominator
        // n1/d1 = n1*k/d2
        let new_n1 = mul2_raw(ctx, n1, k);
        return (new_n1, n2, d2, true);
    }

    // Try to find if d1 = k * d2 (d2 divides d1)
    if let Some(k) = try_extract_factor(ctx, d1, d2) {
        // d1 = k * d2, so use d1 as common denominator
        // n2/d2 = n2*k/d1
        let new_n2 = mul2_raw(ctx, n2, k);
        return (n1, new_n2, d1, true);
    }

    // Not divisible
    (n1, n2, d1, false)
}

/// Returns Some(k) if expr = k * factor, None otherwise
fn try_extract_factor(ctx: &Context, expr: ExprId, factor: ExprId) -> Option<ExprId> {
    // Direct equality check (same ExprId)
    if expr == factor {
        return None; // k would be 1, not useful
    }

    // Check if expr is a Mul containing factor
    if let Expr::Mul(l, r) = ctx.get(expr) {
        // Check l * r where one equals factor (using ExprId equality or structural)
        if *l == factor || exprs_equal(ctx, *l, factor) {
            return Some(*r); // expr = factor * r, so k = r
        }
        if *r == factor || exprs_equal(ctx, *r, factor) {
            return Some(*l); // expr = l * factor, so k = l
        }

        // For nested Mul, we'd need a more sophisticated approach
        // For now, only handle simple a*b case where one of them is the factor
    }

    None
}

/// Check if two expressions are structurally equal
fn exprs_equal(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    if a == b {
        return true;
    }
    match (ctx.get(a), ctx.get(b)) {
        (Expr::Number(n1), Expr::Number(n2)) => n1 == n2,
        (Expr::Variable(v1), Expr::Variable(v2)) => v1 == v2,
        (Expr::Constant(c1), Expr::Constant(c2)) => c1 == c2,
        (Expr::Add(l1, r1), Expr::Add(l2, r2)) => {
            exprs_equal(ctx, *l1, *l2) && exprs_equal(ctx, *r1, *r2)
        }
        (Expr::Sub(l1, r1), Expr::Sub(l2, r2)) => {
            exprs_equal(ctx, *l1, *l2) && exprs_equal(ctx, *r1, *r2)
        }
        (Expr::Mul(l1, r1), Expr::Mul(l2, r2)) => {
            exprs_equal(ctx, *l1, *l2) && exprs_equal(ctx, *r1, *r2)
        }
        (Expr::Div(l1, r1), Expr::Div(l2, r2)) => {
            exprs_equal(ctx, *l1, *l2) && exprs_equal(ctx, *r1, *r2)
        }
        (Expr::Pow(l1, r1), Expr::Pow(l2, r2)) => {
            exprs_equal(ctx, *l1, *l2) && exprs_equal(ctx, *r1, *r2)
        }
        (Expr::Neg(e1), Expr::Neg(e2)) => exprs_equal(ctx, *e1, *e2),
        _ => false,
    }
}

define_rule!(AddFractionsRule, "Add Fractions", |ctx, expr| {
    use cas_ast::views::FractionParts;

    let expr_data = ctx.get(expr).clone();
    if let Expr::Add(l, r) = expr_data {
        // First try FractionParts (handles direct Div and complex multiplicative patterns)
        let fp_l = FractionParts::from(&*ctx, l);
        let fp_r = FractionParts::from(&*ctx, r);

        let (n1, d1, is_frac1) = fp_l.to_num_den(ctx);
        let (n2, d2, is_frac2) = fp_r.to_num_den(ctx);

        // If FractionParts didn't detect fractions, try extract_as_fraction as fallback
        // This handles Mul(1/n, x) pattern that FractionParts misses
        let (n1, d1, is_frac1) = if !is_frac1 {
            extract_as_fraction(ctx, l)
        } else {
            (n1, d1, is_frac1)
        };
        let (n2, d2, is_frac2) = if !is_frac2 {
            extract_as_fraction(ctx, r)
        } else {
            (n2, d2, is_frac2)
        };

        if !is_frac1 && !is_frac2 {
            return None;
        }

        // Check if d2 = -d1 or d2 == d1 (semantic comparison for cross-tree equality)
        let (n2, d2, opposite_denom, same_denom) = {
            // Use semantic comparison: denominators from different subexpressions may have same value but different ExprIds
            let cmp = crate::ordering::compare_expr(ctx, d1, d2);
            if d1 == d2 || cmp == Ordering::Equal {
                (n2, d2, false, true)
            } else if are_denominators_opposite(ctx, d1, d2) {
                // Convert d2 -> d1, n2 -> -n2
                let minus_n2 = ctx.add(Expr::Neg(n2));
                (minus_n2, d1, true, false)
            } else {
                (n2, d2, false, false)
            }
        };

        // Check if one denominator divides the other (d2 = k * d1 or d1 = k * d2)
        // This allows combining 1/2 + 1/(2n) = n/(2n) + 1/(2n) = (n+1)/(2n)
        let (n1, n2, common_den, divisible_denom) =
            check_divisible_denominators(ctx, n1, n2, d1, d2);
        let same_denom = same_denom || divisible_denom;

        // Complexity heuristic
        let old_complexity = count_nodes(ctx, expr);

        // a/b + c/d = (ad + bc) / bd
        let ad = mul2_raw(ctx, n1, d2);
        let bc = mul2_raw(ctx, n2, d1);

        let new_num = if opposite_denom || same_denom {
            ctx.add(Expr::Add(n1, n2))
        } else {
            ctx.add(Expr::Add(ad, bc))
        };

        let new_den = if opposite_denom || same_denom {
            common_den
        } else {
            mul2_raw(ctx, d1, d2)
        };

        // Try to simplify common den
        let common_den = if same_denom || opposite_denom {
            common_den
        } else {
            new_den
        };

        let new_expr = ctx.add(Expr::Div(new_num, common_den));
        let new_complexity = count_nodes(ctx, new_expr);

        // If complexity explodes, avoid adding fractions unless denominators are related
        // Exception: if denominators are numbers, always combine: 1/2 + 1/3 = 5/6
        let is_numeric = |e: ExprId| matches!(ctx.get(e), Expr::Number(_));
        if is_numeric(d1) && is_numeric(d2) {
            return Some(Rewrite {
                new_expr,
                description: "Add numeric fractions".to_string(),
                before_local: None,
                after_local: None,
                domain_assumption: None,
            });
        }

        let simplifies = |ctx: &Context, num: ExprId, den: ExprId| -> (bool, bool) {
            // Heuristics to see if new fraction simplifies
            // e.g. cancellation of factors
            // or algebraic simplification
            // Just checking if we reduced node count isn't enough,
            // because un-added fractions might be smaller locally but harder to work with.
            // But we don't want to create massive expressions.

            // Factor cancellation check would be good.
            // Is there a factor F in num and den?

            // Check if num is 0
            if let Expr::Number(n) = ctx.get(num) {
                if n.is_zero() {
                    return (true, true);
                }
            }

            // Check negation
            let is_negation = |ctx: &Context, a: ExprId, b: ExprId| -> bool {
                if let Expr::Neg(n) = ctx.get(a) {
                    *n == b
                } else if let Expr::Neg(n) = ctx.get(b) {
                    *n == a
                } else if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(a), ctx.get(b)) {
                    n1 == &-n2
                } else {
                    false
                }
            };

            if is_negation(ctx, num, den) {
                return (true, false);
            }

            // Try Polynomial GCD Check
            let vars = collect_variables(ctx, new_num);
            if vars.len() == 1 {
                if let Some(var) = vars.iter().next() {
                    if let Ok(p_num) = Polynomial::from_expr(ctx, new_num, var) {
                        if let Ok(p_den) = Polynomial::from_expr(ctx, common_den, var) {
                            if !p_den.is_zero() {
                                let gcd = p_num.gcd(&p_den);
                                if gcd.degree() > 0 || !gcd.leading_coeff().is_one() {
                                    // println!(
                                    //     "  -> Simplifies via GCD! deg={} lc={}",
                                    //     gcd.degree(),
                                    //     gcd.leading_coeff()
                                    // );
                                    let is_proper = p_num.degree() < p_den.degree();
                                    return (true, is_proper);
                                }
                            }
                        }
                    }
                }
            }

            (false, false)
        };

        let (does_simplify, is_proper) = simplifies(ctx, new_num, common_den);

        // println!(
        //     "AddFractions check: old={} new={} simplify={} limit={}",
        //     old_complexity,
        //     new_complexity,
        //     does_simplify,
        //     (old_complexity * 3) / 2
        // );

        // Allow complexity growth if we found a simplification (GCD)
        // BUT strict check against improper fractions to prevent loops with polynomial division
        // (DividePolynomialsRule splits improper fractions, AddFractions combines them -> loop)
        if opposite_denom
            || same_denom
            || new_complexity <= old_complexity
            || (does_simplify && is_proper && new_complexity < (old_complexity * 2))
        {
            // println!("AddFractions APPLIED: old={} new={} simplify={}", old_complexity, new_complexity, does_simplify);
            return Some(Rewrite {
                new_expr,
                description: "Add fractions: a/b + c/d -> (ad+bc)/bd".to_string(),
                before_local: None,
                after_local: None,
                domain_assumption: None,
            });
        }
    }
    None
});

define_rule!(
    RationalizeDenominatorRule,
    "Rationalize Denominator",
    None,
    PhaseMask::RATIONALIZE,
    |ctx, expr| {
        use cas_ast::views::FractionParts;

        // Use FractionParts to detect any fraction structure
        let fp = FractionParts::from(&*ctx, expr);
        if !fp.is_fraction() {
            return None;
        }

        let (num, den, _) = fp.to_num_den(ctx);

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
                }
                Expr::Function(name, _) => name == "sqrt",
                _ => false,
            }
        };

        let l_root = has_root(l);
        let r_root = has_root(r);

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
        let new_num = mul2_raw(ctx, num, conjugate);

        // Compute new den = l^2 - r^2
        let two = ctx.num(2);
        let l_sq = ctx.add(Expr::Pow(l, two));
        let r_sq = ctx.add(Expr::Pow(r, two));
        let new_den = ctx.add(Expr::Sub(l_sq, r_sq));

        let new_expr = ctx.add(Expr::Div(new_num, new_den));
        return Some(Rewrite {
            new_expr,
            description: "Rationalize denominator (diff squares)".to_string(),
            before_local: None,
            after_local: None,
            domain_assumption: None,
        });
    }
);

/// Collect all additive terms from an expression
/// For a + b + c, returns vec![a, b, c]
fn collect_additive_terms(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut terms = Vec::new();
    collect_terms_recursive(ctx, expr, &mut terms);
    terms
}

fn collect_terms_recursive(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            collect_terms_recursive(ctx, *l, terms);
            collect_terms_recursive(ctx, *r, terms);
        }
        _ => {
            // It's a leaf term (including Sub which we treat as single term)
            terms.push(expr);
        }
    }
}

/// Check if an expression contains an irrational (root)
fn contains_irrational(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Pow(_, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                !n.is_integer() // Fractional exponent = root
            } else {
                false
            }
        }
        Expr::Function(name, _) => name == "sqrt",
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            contains_irrational(ctx, *l) || contains_irrational(ctx, *r)
        }
        Expr::Neg(e) => contains_irrational(ctx, *e),
        _ => false,
    }
}

/// Build a sum from a list of terms
fn build_sum(ctx: &mut Context, terms: &[ExprId]) -> ExprId {
    if terms.is_empty() {
        return ctx.num(0);
    }
    let mut result = terms[0];
    for &term in terms.iter().skip(1) {
        result = ctx.add(Expr::Add(result, term));
    }
    result
}

define_rule!(
    GeneralizedRationalizationRule,
    "Generalized Rationalization",
    |ctx, expr| {
        use cas_ast::views::FractionParts;

        // Use FractionParts to detect any fraction structure
        let fp = FractionParts::from(&*ctx, expr);
        if !fp.is_fraction() {
            return None;
        }

        let (num, den, _) = fp.to_num_den(ctx);

        let terms = collect_additive_terms(ctx, den);

        // Only apply to 3+ terms (binary case handled by RationalizeDenominatorRule)
        if terms.len() < 3 {
            return None;
        }

        // Check if any term contains a root
        let has_roots = terms.iter().any(|&t| contains_irrational(ctx, t));
        if !has_roots {
            return None;
        }

        // Strategy: Group as (first n-1 terms) + last_term
        // Then apply conjugate: multiply by (group - last) / (group - last)
        let last_term = terms[terms.len() - 1];
        let group_terms = &terms[..terms.len() - 1];
        let group = build_sum(ctx, group_terms);

        // Conjugate: (group - last_term)
        let conjugate = ctx.add(Expr::Sub(group, last_term));

        // New numerator: num * conjugate
        let new_num = mul2_raw(ctx, num, conjugate);

        // New denominator: group^2 - last_term^2 (difference of squares)
        let two = ctx.num(2);
        let group_sq = ctx.add(Expr::Pow(group, two));
        let last_sq = ctx.add(Expr::Pow(last_term, two));
        let new_den = ctx.add(Expr::Sub(group_sq, last_sq));

        let new_expr = ctx.add(Expr::Div(new_num, new_den));

        Some(Rewrite {
            new_expr,
            description: format!(
                "Rationalize: group {} terms and multiply by conjugate",
                terms.len()
            ),
            before_local: None,
            after_local: None,
            domain_assumption: None,
        })
    }
);

/// Collect all multiplicative factors from an expression
fn collect_mul_factors(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut factors = Vec::new();
    collect_factors_recursive(ctx, expr, &mut factors);
    factors
}

fn collect_factors_recursive(ctx: &Context, expr: ExprId, factors: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Mul(l, r) => {
            collect_factors_recursive(ctx, *l, factors);
            collect_factors_recursive(ctx, *r, factors);
        }
        _ => {
            factors.push(expr);
        }
    }
}

/// Extract root from expression: sqrt(n) or n^(1/k)
/// Returns (radicand, index) where expr = radicand^(1/index)
fn extract_root_base(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr).clone() {
        Expr::Function(name, args) if name == "sqrt" && args.len() >= 1 => {
            // sqrt(n) = n^(1/2), return (n, 2)
            let two = ctx.num(2);
            Some((args[0], two))
        }
        Expr::Pow(base, exp) => {
            if let Expr::Number(n) = ctx.get(exp).clone() {
                if !n.is_integer() && n.numer().is_one() {
                    // n^(1/k) - return (n, k)
                    let k_expr = ctx.add(Expr::Number(num_rational::BigRational::from_integer(
                        n.denom().clone(),
                    )));
                    return Some((base, k_expr));
                }
            }
            if let Expr::Div(num_exp, den_exp) = ctx.get(exp).clone() {
                if let Expr::Number(n) = ctx.get(num_exp).clone() {
                    if n.is_one() {
                        return Some((base, den_exp));
                    }
                }
            }
            None
        }
        _ => None,
    }
}

define_rule!(
    RationalizeProductDenominatorRule,
    "Rationalize Product Denominator",
    None,
    PhaseMask::RATIONALIZE,
    |ctx, expr| {
        use cas_ast::views::FractionParts;

        // Handle fractions with product denominators containing roots
        let fp = FractionParts::from(&*ctx, expr);
        if !fp.is_fraction() {
            return None;
        }

        let (num, den, _) = fp.to_num_den(ctx);

        let factors = collect_mul_factors(ctx, den);

        // Find a root factor
        let mut root_factor = None;
        let mut non_root_factors = Vec::new();

        for &factor in &factors {
            if extract_root_base(ctx, factor).is_some() && root_factor.is_none() {
                root_factor = Some(factor);
            } else {
                non_root_factors.push(factor);
            }
        }

        let root = root_factor?;

        // Don't apply if denominator is ONLY a root (handled elsewhere or simpler)
        if non_root_factors.is_empty() {
            // Just sqrt(n) in denominator - still rationalize
            if let Some((radicand, _index)) = extract_root_base(ctx, root) {
                // Check if radicand is a binomial (Add or Sub) - these can cause infinite loops
                // when both numerator and denominator have binomial radicals like sqrt(x+y)/sqrt(x-y)
                let is_binomial_radical =
                    matches!(ctx.get(radicand), Expr::Add(_, _) | Expr::Sub(_, _));
                if is_binomial_radical && contains_irrational(ctx, num) {
                    return None;
                }

                // Don't rationalize if radicand is a simple number - power rules handle these better
                // e.g., sqrt(2) / 2^(1/3) should simplify via power combination to 2^(1/6)
                if matches!(ctx.get(radicand), Expr::Number(_)) {
                    return None;
                }

                // 1/sqrt(n) -> sqrt(n)/n
                let new_num = mul2_raw(ctx, num, root);
                let new_den = radicand;
                let new_expr = ctx.add(Expr::Div(new_num, new_den));
                return Some(Rewrite {
                    new_expr,
                    description: "Rationalize: multiply by √n/√n".to_string(),
                    before_local: None,
                    after_local: None,
                    domain_assumption: None,
                });
            }
            return None;
        }

        // Don't apply if radicand is a simple number - power rules can handle these better
        // e.g., 2*sqrt(2) / (2*2^(1/3)) should simplify via power combination, not rationalization
        if let Some((radicand, _index)) = extract_root_base(ctx, root) {
            if matches!(ctx.get(radicand), Expr::Number(_)) {
                return None;
            }
        }

        // We have: num / (other_factors * root) where root = radicand^(1/index)
        // To rationalize, we need to multiply by radicand^((index-1)/index) / radicand^((index-1)/index)
        // This gives: root * radicand^((index-1)/index) = radicand^(1/index + (index-1)/index) = radicand^1 = radicand
        //
        // For sqrt (index=2): multiply by radicand^(1/2) to get radicand^(1/2 + 1/2) = radicand
        // For cbrt (index=3): multiply by radicand^(2/3) to get radicand^(1/3 + 2/3) = radicand

        if let Some((radicand, index)) = extract_root_base(ctx, root) {
            // Compute the conjugate exponent: (index - 1) / index
            // For square root (index=2): conjugate = 1/2, so conjugate_power = radicand^(1/2) = sqrt(radicand)
            // For cube root (index=3): conjugate = 2/3, so conjugate_power = radicand^(2/3)

            // Get index as integer if possible
            let index_val = if let Expr::Number(n) = ctx.get(index) {
                if n.is_integer() {
                    Some(n.to_integer())
                } else {
                    None
                }
            } else {
                None
            };

            // Only handle integer indices for now
            let index_int = index_val?;
            if index_int <= num_bigint::BigInt::from(1) {
                return None; // Not a valid root index
            }

            // Build conjugate exponent (index - 1) / index
            let one = num_bigint::BigInt::from(1);
            let conjugate_num = &index_int - &one;
            let conjugate_exp = num_rational::BigRational::new(conjugate_num, index_int);
            let conjugate_exp_id = ctx.add(Expr::Number(conjugate_exp));

            // conjugate_power = radicand^((index-1)/index)
            let conjugate_power = ctx.add(Expr::Pow(radicand, conjugate_exp_id));

            // New numerator: num * conjugate_power
            let new_num = mul2_raw(ctx, num, conjugate_power);

            // Build new denominator: other_factors * radicand (since root * conjugate_power = radicand)
            let mut new_den = radicand;
            for &factor in &non_root_factors {
                new_den = mul2_raw(ctx, new_den, factor);
            }

            let new_expr = ctx.add(Expr::Div(new_num, new_den));
            return Some(Rewrite {
                new_expr,
                description: "Rationalize product denominator".to_string(),
                before_local: None,
                after_local: None,
                domain_assumption: None,
            });
        }

        None
    }
);

define_rule!(
    CancelCommonFactorsRule,
    "Cancel Common Factors",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();

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

        let (mut num_factors, mut den_factors) = match expr_data {
            Expr::Div(n, d) => (collect_factors(ctx, n), collect_factors(ctx, d)),
            Expr::Pow(b, e) => {
                if let Expr::Number(n) = ctx.get(e) {
                    if n.is_integer() && *n == num_rational::BigRational::from_integer((-1).into())
                    {
                        (vec![ctx.num(1)], collect_factors(ctx, b))
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }
            Expr::Mul(_, _) => {
                let factors = collect_factors(ctx, expr);
                let mut nf = Vec::new();
                let mut df = Vec::new();
                for f in factors {
                    if let Expr::Pow(b, e) = ctx.get(f) {
                        if let Expr::Number(n) = ctx.get(*e) {
                            if n.is_integer()
                                && *n == num_rational::BigRational::from_integer((-1).into())
                            {
                                df.extend(collect_factors(ctx, *b));
                                continue;
                            }
                        }
                    }
                    nf.push(f);
                }
                if df.is_empty() {
                    return None;
                }
                (nf, df)
            }
            _ => return None,
        };
        // NOTE: Pythagorean identity simplification (k - k*sin² → k*cos²) has been
        // extracted to TrigPythagoreanSimplifyRule for pedagogical clarity.
        // CancelCommonFactorsRule now does pure factor cancellation.

        let mut changed = false;
        let mut i = 0;
        while i < num_factors.len() {
            let nf = num_factors[i];
            // println!("Processing num factor: {:?}", ctx.get(nf));
            let mut found = false;
            for j in 0..den_factors.len() {
                let df = den_factors[j];

                // Check exact match
                if crate::ordering::compare_expr(ctx, nf, df) == std::cmp::Ordering::Equal {
                    den_factors.remove(j);
                    found = true;
                    changed = true;
                    break;
                }

                // Check power cancellation: nf = x^n, df = x^m
                // Case 1: nf = base^n, df = base. (n > 1)
                let nf_pow = if let Expr::Pow(b, e) = ctx.get(nf) {
                    Some((*b, *e))
                } else {
                    None
                };
                if let Some((b, e)) = nf_pow {
                    // println!("Checking power cancellation: nf={:?} df={:?}", ctx.get(nf), ctx.get(df));
                    if crate::ordering::compare_expr(ctx, b, df) == std::cmp::Ordering::Equal {
                        // println!("  Base matches!");
                        if let Expr::Number(n) = ctx.get(e) {
                            let new_exp = n - num_rational::BigRational::one();
                            let new_term = if new_exp.is_one() {
                                b
                            } else {
                                let exp_node = ctx.add(Expr::Number(new_exp));
                                ctx.add(Expr::Pow(b, exp_node))
                            };
                            num_factors[i] = new_term;
                            den_factors.remove(j);
                            found = false; // Modified num factor
                            changed = true;
                            break;
                        }
                    }
                }

                // Case 2: nf = base, df = base^m. (m > 1)
                let df_pow = if let Expr::Pow(b, e) = ctx.get(df) {
                    Some((*b, *e))
                } else {
                    None
                };
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
                            den_factors[j] = new_term;
                            found = true; // Remove num factor
                            changed = true;
                            break;
                        }
                    }
                }

                // Case 3: nf = base^n, df = base^m (integer exponents only)
                // Fractional exponents are handled atomically by QuotientOfPowersRule
                if let Some((b_n, e_n)) = nf_pow {
                    if let Some((b_d, e_d)) = df_pow {
                        if crate::ordering::compare_expr(ctx, b_n, b_d) == std::cmp::Ordering::Equal
                        {
                            if let (Expr::Number(n), Expr::Number(m)) = (ctx.get(e_n), ctx.get(e_d))
                            {
                                // Skip fractional exponents - QuotientOfPowersRule handles them
                                if !n.is_integer() || !m.is_integer() {
                                    // Continue to next factor, don't process this pair
                                } else {
                                    if n > m {
                                        let new_exp = n - m;
                                        let new_term = if new_exp.is_one() {
                                            b_n
                                        } else {
                                            let exp_node = ctx.add(Expr::Number(new_exp));
                                            ctx.add(Expr::Pow(b_n, exp_node))
                                        };
                                        num_factors[i] = new_term;
                                        den_factors.remove(j);
                                        found = false;
                                        changed = true;
                                        break;
                                    } else if m > n {
                                        let new_exp = m - n;
                                        let new_term = if new_exp.is_one() {
                                            b_d
                                        } else {
                                            let exp_node = ctx.add(Expr::Number(new_exp));
                                            ctx.add(Expr::Pow(b_d, exp_node))
                                        };
                                        den_factors[j] = new_term;
                                        found = true;
                                        changed = true;
                                        break;
                                    } else {
                                        den_factors.remove(j);
                                        found = true;
                                        changed = true;
                                        break;
                                    }
                                } // end else for integer exponents
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
                let mut n = num_factors[0];
                for &f in num_factors.iter().skip(1) {
                    n = mul2_raw(ctx, n, f);
                }
                n
            };
            let new_den = if den_factors.is_empty() {
                ctx.num(1)
            } else {
                let mut d = den_factors[0];
                for &f in den_factors.iter().skip(1) {
                    d = mul2_raw(ctx, d, f);
                }
                d
            };

            let new_expr = ctx.add(Expr::Div(new_num, new_den));
            return Some(Rewrite {
                new_expr,
                description: "Cancel common factors".to_string(),
                before_local: None,
                after_local: None,
                domain_assumption: None,
            });
        }

        None
    }
);

// Atomized rule for quotient of powers: a^n / a^m = a^(n-m)
// This is separated from CancelCommonFactorsRule for pedagogical clarity
define_rule!(QuotientOfPowersRule, "Quotient of Powers", |ctx, expr| {
    use cas_ast::views::FractionParts;

    let fp = FractionParts::from(&*ctx, expr);
    if !fp.is_fraction() {
        return None;
    }

    let (num, den, _) = fp.to_num_den(ctx);
    let num_data = ctx.get(num).clone();
    let den_data = ctx.get(den).clone();

    // Case 1: a^n / a^m where both are Pow
    if let (Expr::Pow(b_n, e_n), Expr::Pow(b_d, e_d)) = (&num_data, &den_data) {
        // Check same base
        if crate::ordering::compare_expr(ctx, *b_n, *b_d) == std::cmp::Ordering::Equal {
            // Check if exponents are numeric (so we can subtract)
            if let (Expr::Number(n), Expr::Number(m)) = (ctx.get(*e_n), ctx.get(*e_d)) {
                // Only handle fractional exponents here - integer case is in CancelCommonFactors
                if n.is_integer() && m.is_integer() {
                    return None;
                }

                let diff = n - m;
                if diff.is_zero() {
                    // a^n / a^n = 1
                    return Some(Rewrite {
                        new_expr: ctx.num(1),
                        description: "a^n / a^n = 1".to_string(),
                        before_local: None,
                        after_local: None,
                        domain_assumption: None,
                    });
                } else if diff.is_one() {
                    // Result is just the base
                    return Some(Rewrite {
                        new_expr: *b_n,
                        description: "a^n / a^m = a^(n-m)".to_string(),
                        before_local: None,
                        after_local: None,
                        domain_assumption: None,
                    });
                } else {
                    let new_exp = ctx.add(Expr::Number(diff));
                    let new_expr = ctx.add(Expr::Pow(*b_n, new_exp));
                    return Some(Rewrite {
                        new_expr,
                        description: "a^n / a^m = a^(n-m)".to_string(),
                        before_local: None,
                        after_local: None,
                        domain_assumption: None,
                    });
                }
            }
        }
    }

    // Case 2: a^n / a (denominator has implicit exponent 1)
    if let Expr::Pow(b_n, e_n) = &num_data {
        if crate::ordering::compare_expr(ctx, *b_n, den) == std::cmp::Ordering::Equal {
            if let Expr::Number(n) = ctx.get(*e_n) {
                if !n.is_integer() {
                    let new_exp_val = n - num_rational::BigRational::one();
                    if new_exp_val.is_one() {
                        return Some(Rewrite {
                            new_expr: *b_n,
                            description: "a^n / a = a^(n-1)".to_string(),
                            before_local: None,
                            after_local: None,
                            domain_assumption: None,
                        });
                    } else {
                        let new_exp = ctx.add(Expr::Number(new_exp_val));
                        let new_expr = ctx.add(Expr::Pow(*b_n, new_exp));
                        return Some(Rewrite {
                            new_expr,
                            description: "a^n / a = a^(n-1)".to_string(),
                            before_local: None,
                            after_local: None,
                            domain_assumption: None,
                        });
                    }
                }
            }
        }
    }

    // Case 3: a / a^m (numerator has implicit exponent 1)
    if let Expr::Pow(b_d, e_d) = &den_data {
        if crate::ordering::compare_expr(ctx, num, *b_d) == std::cmp::Ordering::Equal {
            if let Expr::Number(m) = ctx.get(*e_d) {
                if !m.is_integer() {
                    let new_exp_val = num_rational::BigRational::one() - m;
                    let new_exp = ctx.add(Expr::Number(new_exp_val));
                    let new_expr = ctx.add(Expr::Pow(num, new_exp));
                    return Some(Rewrite {
                        new_expr,
                        description: "a / a^m = a^(1-m)".to_string(),
                        before_local: None,
                        after_local: None,
                        domain_assumption: None,
                    });
                }
            }
        }
    }

    None
});

define_rule!(
    PullConstantFromFractionRule,
    "Pull Constant From Fraction",
    |ctx, expr| {
        // NOTE: Keep simple Div detection to avoid infinite loop with Combine Like Terms
        // when detecting Neg(Div(...)) as a fraction
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
                let new_expr = mul2_raw(ctx, l, div);
                return Some(Rewrite {
                    new_expr,
                    description: "Pull constant from numerator".to_string(),
                    before_local: None,
                    after_local: None,
                    domain_assumption: None,
                });
            } else if r_is_const {
                // (x * c) / y -> c * (x / y)
                let div = ctx.add(Expr::Div(l, d));
                let new_expr = mul2_raw(ctx, r, div);
                return Some(Rewrite {
                    new_expr,
                    description: "Pull constant from numerator".to_string(),
                    before_local: None,
                    after_local: None,
                    domain_assumption: None,
                });
            }
        }
        // Also handle Neg: (-x) / y -> -1 * (x / y)
        if let Expr::Neg(inner) = num_data {
            let minus_one = ctx.num(-1);
            let div = ctx.add(Expr::Div(inner, d));
            let new_expr = mul2_raw(ctx, minus_one, div);
            return Some(Rewrite {
                new_expr,
                description: "Pull negation from numerator".to_string(),
                before_local: None,
                after_local: None,
                domain_assumption: None,
            });
        }
        None
    }
);

define_rule!(
    FactorBasedLCDRule,
    "Factor-Based LCD",
    Some(vec!["Add"]),
    |ctx, expr| {
        use crate::ordering::compare_expr;
        use std::cmp::Ordering;

        // Normalize a binomial to canonical form: (a-b) where a < b alphabetically
        // Returns (canonical_expr, sign_flip) where sign_flip is true if we negated
        let normalize_binomial = |ctx: &mut Context, e: ExprId| -> (ExprId, bool) {
            match ctx.get(e).clone() {
                Expr::Add(l, r) => {
                    if let Expr::Neg(inner) = ctx.get(r).clone() {
                        // Form: l + (-inner) = l - inner
                        if compare_expr(ctx, l, inner) == Ordering::Less {
                            (e, false) // Already canonical
                        } else {
                            // Create: -(inner - l) = (l - inner) negated
                            let neg_l = ctx.add(Expr::Neg(l));
                            let canonical = ctx.add(Expr::Add(inner, neg_l));
                            (canonical, true)
                        }
                    } else {
                        (e, false) // Not a subtraction pattern
                    }
                }
                Expr::Sub(l, r) => {
                    if compare_expr(ctx, l, r) == Ordering::Less {
                        (e, false)
                    } else {
                        let canonical = ctx.add(Expr::Sub(r, l));
                        (canonical, true)
                    }
                }
                _ => (e, false),
            }
        };

        // Extract factors from a product expression
        let get_factors = |ctx: &Context, e: ExprId| -> Vec<ExprId> {
            let mut factors = Vec::new();
            let mut stack = vec![e];
            while let Some(curr) = stack.pop() {
                match ctx.get(curr) {
                    Expr::Mul(l, r) => {
                        stack.push(*l);
                        stack.push(*r);
                    }
                    _ => factors.push(curr),
                }
            }
            factors
        };

        // Check if expression is a binomial (Add with Neg or Sub)
        let is_binomial = |ctx: &Context, e: ExprId| -> bool {
            match ctx.get(e) {
                Expr::Add(_, r) => matches!(ctx.get(*r), Expr::Neg(_)),
                Expr::Sub(_, _) => true,
                _ => false,
            }
        };

        // Check if two expressions are equal (by compare_expr)
        let expr_eq = |ctx: &Context, a: ExprId, b: ExprId| -> bool {
            compare_expr(ctx, a, b) == Ordering::Equal
        };

        // ===== Main Logic =====

        // Collect all terms from the Add tree
        let mut terms = Vec::new();
        let mut stack = vec![expr];
        while let Some(curr) = stack.pop() {
            match ctx.get(curr) {
                Expr::Add(l, r) => {
                    stack.push(*l);
                    stack.push(*r);
                }
                _ => terms.push(curr),
            }
        }

        // Need at least 3 fractions - AddFractionsRule handles 2-fraction cases
        if terms.len() < 3 {
            return None;
        }

        // Extract (numerator, denominator) from each fraction
        let mut fractions: Vec<(ExprId, ExprId)> = Vec::new();
        for term in &terms {
            match ctx.get(*term) {
                Expr::Div(num, den) => fractions.push((*num, *den)),
                _ => return None, // Not all terms are fractions
            }
        }

        // For each denominator, extract and normalize binomial factors
        // Store: Vec<(canonical_factor, sign_flip)> for each fraction
        let mut all_factor_sets: Vec<Vec<(ExprId, bool)>> = Vec::new();

        for (_, den) in &fractions {
            let raw_factors = get_factors(ctx, *den);

            // All factors must be binomials for this rule to apply
            let mut normalized = Vec::new();
            for f in raw_factors {
                if !is_binomial(ctx, f) {
                    return None;
                }
                let (canonical, flipped) = normalize_binomial(ctx, f);
                normalized.push((canonical, flipped));
            }

            if normalized.is_empty() {
                return None;
            }

            all_factor_sets.push(normalized);
        }

        // Collect all unique canonical factors (the LCD factors)
        let mut unique_factors: Vec<ExprId> = Vec::new();
        for factor_set in &all_factor_sets {
            for (canonical, _) in factor_set {
                let exists = unique_factors.iter().any(|u| expr_eq(ctx, *u, *canonical));
                if !exists {
                    unique_factors.push(*canonical);
                }
            }
        }

        // Skip if all fractions already have the same denominator
        let all_same = all_factor_sets.iter().all(|fs| {
            fs.len() == unique_factors.len()
                && unique_factors
                    .iter()
                    .all(|uf| fs.iter().any(|(cf, _)| expr_eq(ctx, *cf, *uf)))
        });
        if all_same && fractions.len() == terms.len() {
            // If fractions share all factors (same LCD), AddFractionsRule handles it
            return None;
        }

        // Build LCD as product of all unique factors
        let lcd = if unique_factors.len() == 1 {
            unique_factors[0]
        } else {
            let mut product = unique_factors[0];
            for i in 1..unique_factors.len() {
                product = mul2_raw(ctx, product, unique_factors[i]);
            }
            product
        };

        // For each fraction, compute numerator contribution
        let mut numerator_terms: Vec<ExprId> = Vec::new();

        for (i, (num, _den)) in fractions.iter().enumerate() {
            let factor_set = &all_factor_sets[i];

            // Compute overall sign from normalization
            let sign_flips: usize = factor_set.iter().filter(|(_, f)| *f).count();
            let is_negative = sign_flips % 2 == 1;

            // Find missing factors (in unique but not in this denominator)
            let mut missing: Vec<ExprId> = Vec::new();
            for uf in &unique_factors {
                let present = factor_set.iter().any(|(cf, _)| expr_eq(ctx, *cf, *uf));
                if !present {
                    missing.push(*uf);
                }
            }

            // Multiply numerator by all missing factors
            let mut contribution = *num;
            for mf in missing {
                contribution = mul2_raw(ctx, contribution, mf);
            }

            // Apply sign
            if is_negative {
                contribution = ctx.add(Expr::Neg(contribution));
            }

            numerator_terms.push(contribution);
        }

        // Sum all numerator contributions
        let total_num = if numerator_terms.len() == 1 {
            numerator_terms[0]
        } else {
            let mut sum = numerator_terms[0];
            for i in 1..numerator_terms.len() {
                sum = ctx.add(Expr::Add(sum, numerator_terms[i]));
            }
            sum
        };

        // Create the combined fraction
        let new_expr = ctx.add(Expr::Div(total_num, lcd));

        Some(Rewrite {
            new_expr,
            description: "Combine fractions with factor-based LCD".to_string(),
            before_local: None,
            after_local: None,
            domain_assumption: None,
        })
    }
);

// ========== Light Rationalization for Single Numeric Surd Denominators ==========
// Transforms: num / (k * √n) → (num * √n) / (k * n)
// Only applies when:
// - denominator contains exactly one numeric square root
// - base of the root is a positive integer
// - no variables inside the radical

define_rule!(
    RationalizeSingleSurdRule,
    "Rationalize Single Surd",
    None,
    PhaseMask::RATIONALIZE,
    |ctx, expr| {
        use cas_ast::views::as_rational_const;
        use num_rational::BigRational;
        use num_traits::ToPrimitive;

        // Only match Div expressions
        let (num, den) = match ctx.get(expr).clone() {
            Expr::Div(n, d) => (n, d),
            _ => return None,
        };

        // Check denominator for Pow(Number(n), 1/2) patterns
        // We need to find exactly one surd in the denominator factors

        // Helper to check if an expression is a numeric square root
        fn is_numeric_sqrt(ctx: &Context, id: ExprId) -> Option<i64> {
            if let Expr::Pow(base, exp) = ctx.get(id) {
                // Check exponent is 1/2 (using robust detection)
                let exp_val = as_rational_const(ctx, *exp, 8)?;
                let half = BigRational::new(1.into(), 2.into());
                if exp_val != half {
                    return None;
                }
                // Check base is a positive integer
                if let Expr::Number(n) = ctx.get(*base) {
                    if n.is_integer() {
                        return n.numer().to_i64().filter(|&x| x > 0);
                    }
                }
            }
            None
        }

        // Try different denominator patterns
        let (sqrt_n_value, other_den_factors): (i64, Vec<ExprId>) = match ctx.get(den).clone() {
            // Case 1: Denominator is just √n
            Expr::Pow(_, _) => {
                if let Some(n) = is_numeric_sqrt(ctx, den) {
                    (n, vec![])
                } else {
                    return None;
                }
            }

            // Case 2: Denominator is k * √n or √n * k (one level of Mul)
            Expr::Mul(l, r) => {
                if let Some(n) = is_numeric_sqrt(ctx, l) {
                    // √n * k form
                    (n, vec![r])
                } else if let Some(n) = is_numeric_sqrt(ctx, r) {
                    // k * √n form
                    (n, vec![l])
                } else {
                    // Check if either side is a Mul containing √n (two levels)
                    // For simplicity, we only handle shallow cases
                    return None;
                }
            }

            // Case 3: Function("sqrt", [n])
            Expr::Function(name, ref args) if name == "sqrt" && args.len() == 1 => {
                if let Expr::Number(n) = ctx.get(args[0]) {
                    if n.is_integer() {
                        if let Some(n_int) = n.numer().to_i64().filter(|&x| x > 0) {
                            (n_int, vec![])
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                } else {
                    return None; // Variable inside sqrt
                }
            }

            _ => return None,
        };

        // Build the rationalized form: (num * √n) / (other_den * n)
        let n_expr = ctx.num(sqrt_n_value);
        let half = ctx.rational(1, 2);
        let sqrt_n = ctx.add(Expr::Pow(n_expr, half));

        // New numerator: num * √n
        let new_num = mul2_raw(ctx, num, sqrt_n);

        // New denominator: other_den_factors * n
        let n_in_den = ctx.num(sqrt_n_value);
        let new_den = if other_den_factors.is_empty() {
            n_in_den
        } else {
            let mut den_product = other_den_factors[0];
            for &f in &other_den_factors[1..] {
                den_product = mul2_raw(ctx, den_product, f);
            }
            mul2_raw(ctx, den_product, n_in_den)
        };

        let new_expr = ctx.add(Expr::Div(new_num, new_den));

        // Optional: Check node count didn't explode (shouldn't for this simple transform)
        if count_nodes(ctx, new_expr) > count_nodes(ctx, expr) + 10 {
            return None;
        }

        Some(Rewrite {
            new_expr,
            description: format!(
                "{} / {} -> {} / {}",
                DisplayExpr {
                    context: ctx,
                    id: num
                },
                DisplayExpr {
                    context: ctx,
                    id: den
                },
                DisplayExpr {
                    context: ctx,
                    id: new_num
                },
                DisplayExpr {
                    context: ctx,
                    id: new_den
                }
            ),
            before_local: None,
            after_local: None,
            domain_assumption: None,
        })
    }
);

// ========== Binomial Conjugate Rationalization (Level 1) ==========
// Transforms: num / (A + B√n) → num * (A - B√n) / (A² - B²·n)
// Only applies when:
// - denominator is a binomial with exactly one numeric surd term
// - A, B are rational, n is a positive integer
// - uses closed-form arithmetic (no calls to general simplifier)

define_rule!(
    RationalizeBinomialSurdRule,
    "Rationalize Binomial Denominator",
    None,
    PhaseMask::RATIONALIZE,
    |ctx, expr| {
        use crate::rationalize_policy::RationalizeReason;
        use cas_ast::views::{as_rational_const, count_distinct_numeric_surds, is_surd_free};
        use num_rational::BigRational;
        use num_traits::ToPrimitive;

        // Only match Div expressions
        let (num, den) = match ctx.get(expr).clone() {
            Expr::Div(n, d) => (n, d),
            _ => {
                tracing::trace!(target: "rationalize", "skipped: not a division");
                return None;
            }
        };

        // Budget guard: denominator shouldn't be too complex
        let den_nodes = count_nodes(ctx, den);
        if den_nodes > 30 {
            tracing::debug!(target: "rationalize", reason = ?RationalizeReason::BudgetExceeded, 
                            nodes = den_nodes, max = 30, "auto rationalize rejected");
            return None;
        }

        // Multi-surd guard: only rationalize if denominator has exactly 1 distinct surd
        // Level 1.5 blocks multi-surd expressions (reserved for `rationalize` command)
        let distinct_surds = count_distinct_numeric_surds(ctx, den, 50);
        if distinct_surds == 0 {
            tracing::trace!(target: "rationalize", "skipped: no surds found");
            return None;
        }
        if distinct_surds > 1 {
            tracing::debug!(target: "rationalize", reason = ?RationalizeReason::MultiSurdBlocked,
                            surds = distinct_surds, "auto rationalize rejected");
            return None;
        }

        // Try to parse denominator as A ± B√n (binomial surd)
        // Patterns: Add(A, Mul(B, √n)), Add(A, √n), Sub(A, Mul(B, √n)), etc.

        struct BinomialSurd {
            a: BigRational, // Rational constant term
            b: BigRational, // Coefficient of surd
            n: i64,         // Radicand (square-free positive integer)
            is_sub: bool,   // true if A - B√n, false if A + B√n
        }

        fn parse_binomial_surd(ctx: &Context, den: ExprId) -> Option<BinomialSurd> {
            // Helper to check if expression is a numeric √n
            fn is_numeric_sqrt(ctx: &Context, id: ExprId) -> Option<i64> {
                match ctx.get(id) {
                    Expr::Pow(base, exp) => {
                        let exp_val = as_rational_const(ctx, *exp, 8)?;
                        let half = BigRational::new(1.into(), 2.into());
                        if exp_val != half {
                            return None;
                        }
                        if let Expr::Number(n) = ctx.get(*base) {
                            if n.is_integer() {
                                return n.numer().to_i64().filter(|&x| x > 0);
                            }
                        }
                        None
                    }
                    Expr::Function(name, args) if name == "sqrt" && args.len() == 1 => {
                        if let Expr::Number(n) = ctx.get(args[0]) {
                            if n.is_integer() {
                                return n.numer().to_i64().filter(|&x| x > 0);
                            }
                        }
                        None
                    }
                    _ => None,
                }
            }

            // Helper to parse B*√n or √n (B=1)
            fn parse_surd_term(ctx: &Context, id: ExprId) -> Option<(BigRational, i64)> {
                // Try √n directly (B=1)
                if let Some(n) = is_numeric_sqrt(ctx, id) {
                    return Some((BigRational::from_integer(1.into()), n));
                }
                // Try B * √n
                if let Expr::Mul(l, r) = ctx.get(id) {
                    if let Some(n) = is_numeric_sqrt(ctx, *r) {
                        if let Some(b) = as_rational_const(ctx, *l, 8) {
                            return Some((b, n));
                        }
                    }
                    if let Some(n) = is_numeric_sqrt(ctx, *l) {
                        if let Some(b) = as_rational_const(ctx, *r, 8) {
                            return Some((b, n));
                        }
                    }
                }
                None
            }

            match ctx.get(den) {
                // A + surd_term
                Expr::Add(l, r) => {
                    // Try l=A (rational), r=B√n
                    if let Some(a) = as_rational_const(ctx, *l, 8) {
                        if let Some((b, n)) = parse_surd_term(ctx, *r) {
                            return Some(BinomialSurd {
                                a,
                                b,
                                n,
                                is_sub: false,
                            });
                        }
                    }
                    // Try l=B√n, r=A
                    if let Some(a) = as_rational_const(ctx, *r, 8) {
                        if let Some((b, n)) = parse_surd_term(ctx, *l) {
                            return Some(BinomialSurd {
                                a,
                                b,
                                n,
                                is_sub: false,
                            });
                        }
                    }
                    None
                }
                // A - surd_term
                Expr::Sub(l, r) => {
                    if let Some(a) = as_rational_const(ctx, *l, 8) {
                        if let Some((b, n)) = parse_surd_term(ctx, *r) {
                            return Some(BinomialSurd {
                                a,
                                b,
                                n,
                                is_sub: true,
                            });
                        }
                    }
                    None
                }
                _ => None,
            }
        }

        // Helper to extract binomial factor from a Mul chain (Level 1.5)
        // Returns (k_factors, binomial) where k_factors are surd-free factors
        fn extract_binomial_from_product(
            ctx: &Context,
            den: ExprId,
        ) -> Option<(Vec<ExprId>, BinomialSurd)> {
            // First try direct binomial (Level 1)
            if let Some(surd) = parse_binomial_surd(ctx, den) {
                return Some((vec![], surd));
            }

            // Try Mul chain (Level 1.5)
            match ctx.get(den) {
                Expr::Mul(_, _) => {
                    // Flatten the Mul chain preserving order
                    fn collect_factors(ctx: &Context, id: ExprId, factors: &mut Vec<ExprId>) {
                        match ctx.get(id) {
                            Expr::Mul(l, r) => {
                                collect_factors(ctx, *l, factors);
                                collect_factors(ctx, *r, factors);
                            }
                            _ => factors.push(id),
                        }
                    }

                    let mut factors = Vec::new();
                    collect_factors(ctx, den, &mut factors);

                    // Find exactly one binomial factor; others must be surd-free
                    let mut binomial_idx = None;
                    for (i, &factor) in factors.iter().enumerate() {
                        if let Some(_) = parse_binomial_surd(ctx, factor) {
                            if binomial_idx.is_some() {
                                // Multiple binomials → not Level 1.5
                                return None;
                            }
                            binomial_idx = Some(i);
                        } else if !is_surd_free(ctx, factor, 20) {
                            // Factor is neither binomial nor surd-free → skip
                            return None;
                        }
                    }

                    let binomial_idx = binomial_idx?;
                    let binomial = parse_binomial_surd(ctx, factors[binomial_idx])?;

                    // Collect K factors (those not the binomial)
                    let k_factors: Vec<_> = factors
                        .into_iter()
                        .enumerate()
                        .filter(|(i, _)| *i != binomial_idx)
                        .map(|(_, f)| f)
                        .collect();

                    Some((k_factors, binomial))
                }
                _ => None,
            }
        }

        let (k_factors, surd) = extract_binomial_from_product(ctx, den)?;

        // Build k_factor product (outside the helper to avoid borrow issues)
        let k_factor: Option<ExprId> = if k_factors.is_empty() {
            None
        } else if k_factors.len() == 1 {
            Some(k_factors[0])
        } else {
            let mut k = k_factors[0];
            for &f in &k_factors[1..] {
                k = ctx.add(Expr::Mul(k, f));
            }
            Some(k)
        };

        // Compute conjugate: if A + B√n, conjugate is A - B√n (and vice versa)
        // New denominator = A² - B²·n (always the same)
        let a_sq = &surd.a * &surd.a;
        let b_sq = &surd.b * &surd.b;
        let b_sq_n = &b_sq * BigRational::from_integer(surd.n.into());
        let new_den_val = &a_sq - &b_sq_n;

        // Check denominator is non-zero
        if new_den_val == BigRational::from_integer(0.into()) {
            return None;
        }

        // Build conjugate expression: A ∓ B√n
        let a_expr = ctx.add(Expr::Number(surd.a.clone()));
        let n_expr = ctx.num(surd.n);
        let half = ctx.rational(1, 2);
        let sqrt_n = ctx.add(Expr::Pow(n_expr, half));

        let b_sqrt_n = if surd.b == BigRational::from_integer(1.into()) {
            sqrt_n
        } else if surd.b == BigRational::from_integer((-1).into()) {
            ctx.add(Expr::Neg(sqrt_n))
        } else {
            let b_expr = ctx.add(Expr::Number(surd.b.clone()));
            mul2_raw(ctx, b_expr, sqrt_n)
        };

        // conjugate = A - B√n if original was A + B√n (is_sub=false)
        // conjugate = A + B√n if original was A - B√n (is_sub=true)
        let conjugate = if surd.is_sub {
            ctx.add(Expr::Add(a_expr, b_sqrt_n))
        } else {
            ctx.add(Expr::Sub(a_expr, b_sqrt_n))
        };

        // Build new numerator: num * conjugate
        // But first, handle negative denominator by absorbing sign into conjugate
        let (final_conjugate, final_den_val) = if new_den_val < BigRational::from_integer(0.into())
        {
            // Negative denominator: negate the entire conjugate
            // This produces -(A + B√n) instead of -A - B√n, which is cleaner for display
            let negated_conjugate = ctx.add(Expr::Neg(conjugate));
            (negated_conjugate, -new_den_val.clone())
        } else {
            (conjugate, new_den_val.clone())
        };

        let new_num = mul2_raw(ctx, num, final_conjugate);

        // Build new denominator as Number (now always positive or handled)
        let new_den = ctx.add(Expr::Number(final_den_val.clone()));

        // If denominator is 1, just return numerator (possibly divided by K)
        let new_expr = if final_den_val == BigRational::from_integer(1.into()) {
            // new_den = K (if present) or 1
            match k_factor {
                Some(k) => ctx.add(Expr::Div(new_num, k)),
                None => new_num,
            }
        } else {
            // new_den = K * (A² - B²n) or just (A² - B²n)
            let rationalized_den = ctx.add(Expr::Number(final_den_val.clone()));
            let full_den = match k_factor {
                Some(k) => mul2_raw(ctx, k, rationalized_den),
                None => rationalized_den,
            };
            ctx.add(Expr::Div(new_num, full_den))
        };

        // Verify we actually made progress (denominator is now rational)
        if count_nodes(ctx, new_expr) > count_nodes(ctx, expr) + 20 {
            return None;
        }

        Some(Rewrite {
            new_expr,
            description: format!(
                "{} / {} -> {} / {}",
                DisplayExpr {
                    context: ctx,
                    id: num
                },
                DisplayExpr {
                    context: ctx,
                    id: den
                },
                DisplayExpr {
                    context: ctx,
                    id: new_num
                },
                DisplayExpr {
                    context: ctx,
                    id: new_den
                }
            ),
            before_local: None,
            after_local: None,
            domain_assumption: None,
        })
    }
);

// ============================================================================
// R1: Absorb Negation Into Difference Factor
// ============================================================================
// -1/((x-y)*...) → 1/((y-x)*...)
// Absorbs the negative sign by flipping one difference in the denominator.
// Differences can be Sub(x,y) or Add(x, Neg(y)) or Add(x, Mul(-1,y)).

/// Check if expression is a difference (x - y) in any canonical form
/// Returns Some((x, y)) if it's a difference
fn extract_difference(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr) {
        Expr::Sub(l, r) => Some((*l, *r)),
        Expr::Add(l, r) => {
            // Check if right is Neg(y)
            if let Expr::Neg(inner) = ctx.get(*r) {
                return Some((*l, *inner));
            }
            // Check if right is Mul(-1, y) or Mul(y, -1) with negative number
            if let Expr::Mul(a, b) = ctx.get(*r) {
                if let Expr::Number(n) = ctx.get(*a) {
                    if n.is_negative() && *n == num_rational::BigRational::from_integer((-1).into())
                    {
                        return Some((*l, *b));
                    }
                }
                if let Expr::Number(n) = ctx.get(*b) {
                    if n.is_negative() && *n == num_rational::BigRational::from_integer((-1).into())
                    {
                        return Some((*l, *a));
                    }
                }
            }
            // Check if left is Neg(x)
            if let Expr::Neg(inner) = ctx.get(*l) {
                return Some((*r, *inner));
            }
            None
        }
        _ => None,
    }
}

/// Build a difference expression: always use Sub form now that canonicalization
/// works properly with our fixes
fn build_difference(ctx: &mut Context, x: ExprId, y: ExprId) -> ExprId {
    ctx.add(Expr::Sub(x, y))
}

define_rule!(
    AbsorbNegationIntoDifferenceRule,
    "Absorb Negation Into Difference",
    |ctx, expr| {
        // Check for Neg(Div(...)) or Div with negative numerator
        let (is_neg_wrapped, div_num, div_den) = match ctx.get(expr) {
            Expr::Neg(inner) => {
                if let Expr::Div(n, d) = ctx.get(*inner) {
                    (true, *n, *d)
                } else {
                    return None;
                }
            }
            Expr::Div(n, d) => {
                if let Expr::Number(num_val) = ctx.get(*n) {
                    if num_val.is_negative() {
                        (false, *n, *d)
                    } else {
                        return None;
                    }
                } else if let Expr::Neg(_) = ctx.get(*n) {
                    (false, *n, *d)
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        // Collect all factors from denominator
        let mut factors: Vec<ExprId> = collect_mul_factors(ctx, div_den);

        // Find a difference factor to flip
        let mut flip_index: Option<usize> = None;
        let mut diff_pair: Option<(ExprId, ExprId)> = None;
        for (i, &f) in factors.iter().enumerate() {
            if let Some((x, y)) = extract_difference(ctx, f) {
                flip_index = Some(i);
                diff_pair = Some((x, y));
                break;
            }
        }

        let idx = flip_index?;
        let (x, y) = diff_pair?;

        // Flip the difference: (x - y) → (y - x)
        let flipped = build_difference(ctx, y, x);
        factors[idx] = flipped;

        // Rebuild denominator
        let new_den = factors.iter().copied().fold(None, |acc, f| {
            Some(match acc {
                Some(a) => mul2_raw(ctx, a, f),
                None => f,
            })
        })?;

        // Handle numerator: remove the negation
        let new_num = if is_neg_wrapped {
            div_num
        } else if let Expr::Number(n) = ctx.get(div_num) {
            ctx.add(Expr::Number(-n.clone()))
        } else if let Expr::Neg(inner) = ctx.get(div_num) {
            *inner
        } else {
            return None;
        };

        let new_expr = ctx.add(Expr::Div(new_num, new_den));

        Some(Rewrite {
            new_expr,
            description: "Absorb negation into difference factor".to_string(),
            before_local: None,
            after_local: None,
            domain_assumption: None,
        })
    }
);

// ============================================================================
// R2: Canonicalize Products of Same-Tail Differences
// ============================================================================
// 1/((p-t)*(q-t)) → 1/((t-p)*(t-q))
// When two difference factors share the same "tail" (right operand),
// flip both to have that common element first.
// Double-flip preserves the overall sign.

define_rule!(
    CanonicalDifferenceProductRule,
    "Canonicalize Difference Product",
    |ctx, expr| {
        let (num, den) = if let Expr::Div(n, d) = ctx.get(expr) {
            (*n, *d)
        } else {
            return None;
        };

        // Check if denominator is Mul of exactly two Sub expressions
        let (factor1, factor2) = if let Expr::Mul(l, r) = ctx.get(den) {
            (*l, *r)
        } else {
            return None;
        };

        // Both factors must be Sub
        let (p, t1) = if let Expr::Sub(a, b) = ctx.get(factor1) {
            (*a, *b)
        } else {
            return None;
        };
        let (q, t2) = if let Expr::Sub(a, b) = ctx.get(factor2) {
            (*a, *b)
        } else {
            return None;
        };

        // Check if they share the same tail
        if crate::ordering::compare_expr(ctx, t1, t2) != Ordering::Equal {
            return None;
        }

        let t = t1;

        // Only flip if the current form is NOT canonical
        // Canonical: (t - p) * (t - q) where t comes first in both
        // Current is (p - t) * (q - t) - needs flipping
        // Guard: if t already comes first in both, don't flip (avoid loops)
        let t_already_first_1 = if let Expr::Sub(a, _) = ctx.get(factor1) {
            crate::ordering::compare_expr(ctx, *a, t) == Ordering::Equal
        } else {
            false
        };
        let t_already_first_2 = if let Expr::Sub(a, _) = ctx.get(factor2) {
            crate::ordering::compare_expr(ctx, *a, t) == Ordering::Equal
        } else {
            false
        };

        if t_already_first_1 && t_already_first_2 {
            return None; // Already canonical
        }

        // Flip both: (p-t) → (t-p), (q-t) → (t-q)
        let new_factor1 = ctx.add(Expr::Sub(t, p));
        let new_factor2 = ctx.add(Expr::Sub(t, q));
        let new_den = mul2_raw(ctx, new_factor1, new_factor2);

        let new_expr = ctx.add(Expr::Div(num, new_den));

        Some(Rewrite {
            new_expr,
            description: "Canonicalize same-tail difference product".to_string(),
            before_local: None,
            after_local: None,
            domain_assumption: None,
        })
    }
);
