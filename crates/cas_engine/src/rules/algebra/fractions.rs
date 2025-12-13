use crate::define_rule;
use crate::polynomial::Polynomial;
use crate::rule::Rewrite;
use cas_ast::expression::count_nodes;
use cas_ast::{Context, DisplayExpr, Expr, ExprId};
use num_traits::{One, Zero};

use super::helpers::*;

// ========== Micro-API for safe Mul construction ==========
// Use this instead of ctx.add(Expr::Mul(...)) in this file.

/// Build a simple 2-factor product (right-associative, no normalization).
#[inline]
fn mul2_raw(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    ctx.add(Expr::Mul(a, b))
}

define_rule!(
    SimplifyFractionRule,
    "Simplify Nested Fraction",
    |ctx, expr| {
        use cas_ast::views::RationalFnView;

        // Use RationalFnView to detect any fraction form while preserving structure
        let view = RationalFnView::from(ctx, expr)?;
        let (num, den) = (view.num, view.den);

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
        // Use FractionParts to detect fractions uniformly
        let fp_l = FractionParts::from(&*ctx, l);
        let fp_r = FractionParts::from(&*ctx, r);

        let (n1, d1, is_frac1) = fp_l.to_num_den(ctx);
        let (n2, d2, is_frac2) = fp_r.to_num_den(ctx);

        if !is_frac1 && !is_frac2 {
            // println!("  Not fractions: {:?} {:?}", is_frac1, is_frac2);
            return None;
        }
        // println!(
        //     "  Got fractions: l={:?} r={:?} (frac: {} {})",
        //     l, r, is_frac1, is_frac2
        // );
        // println!("  n1={:?} d1={:?}", ctx.get(n1), ctx.get(d1));
        // println!("  n2={:?} d2={:?}", ctx.get(n2), ctx.get(d2));

        // Check if d2 = -d1 or d2 == d1
        let (n2, d2, opposite_denom, same_denom) = {
            if d1 == d2 {
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
            });
        }
    }
    None
});

define_rule!(
    RationalizeDenominatorRule,
    "Rationalize Denominator",
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
                    });
                } else if diff.is_one() {
                    // Result is just the base
                    return Some(Rewrite {
                        new_expr: *b_n,
                        description: "a^n / a^m = a^(n-m)".to_string(),
                        before_local: None,
                        after_local: None,
                    });
                } else {
                    let new_exp = ctx.add(Expr::Number(diff));
                    let new_expr = ctx.add(Expr::Pow(*b_n, new_exp));
                    return Some(Rewrite {
                        new_expr,
                        description: "a^n / a^m = a^(n-m)".to_string(),
                        before_local: None,
                        after_local: None,
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
                        });
                    } else {
                        let new_exp = ctx.add(Expr::Number(new_exp_val));
                        let new_expr = ctx.add(Expr::Pow(*b_n, new_exp));
                        return Some(Rewrite {
                            new_expr,
                            description: "a^n / a = a^(n-1)".to_string(),
                            before_local: None,
                            after_local: None,
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
        })
    }
);
