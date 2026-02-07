//! Polynomial manipulation rules: distribution, annihilation, combining like terms,
//! expansion, and factoring.
//!
//! This module is split into submodules:
//! - `expansion`: Binomial/multinomial expansion, auto-expand, polynomial identity detection
//! - `factoring`: Heuristic common factor extraction

mod expansion;
mod factoring;

pub use expansion::{
    AutoExpandPowSumRule, AutoExpandSubCancelRule, BinomialExpansionRule,
    ExpandSmallBinomialPowRule, HeuristicPolyNormalizeAddRule, PolynomialIdentityZeroRule,
};
pub use factoring::HeuristicExtractCommonFactorAddRule;

use crate::define_rule;
use crate::nary::{build_balanced_add, AddView, Sign};
use crate::ordering::compare_expr;
use crate::phase::PhaseMask;
use crate::polynomial::Polynomial;
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{Context, Expr, ExprId};
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::Signed;
use num_traits::{One, Zero};
use std::cmp::Ordering;

/// Check if an expression is a binomial (sum or difference of exactly 2 terms)
/// Examples: (a + b), (a - b), (x + (-y))
fn is_binomial(ctx: &Context, e: ExprId) -> bool {
    matches!(ctx.get(e), Expr::Add(_, _) | Expr::Sub(_, _))
}

// DistributeRule: Runs in CORE, TRANSFORM, RATIONALIZE but NOT in POST
// This prevents Factor↔Distribute infinite loops (FactorCommonIntegerFromAdd runs in POST)
define_rule!(
    DistributeRule,
    "Distributive Property",
    None,
    // NO POST: evita ciclo con FactorCommonIntegerFromAdd (ver test_factor_distribute_no_loop)
    PhaseMask::CORE | PhaseMask::TRANSFORM | PhaseMask::RATIONALIZE,
    |ctx, expr, parent_ctx| {
        use crate::semantics::NormalFormGoal;

        // GATE: Don't distribute when goal is Collected or Factored
        // This prevents undoing the effect of collect() or factor() commands
        match parent_ctx.goal() {
            NormalFormGoal::Collected | NormalFormGoal::Factored => return None,
            _ => {}
        }

        // Don't distribute if expression is in canonical form (e.g., inside abs() or sqrt())
        // This protects patterns like abs((x-2)(x+2)) from expanding
        if crate::canonical_forms::is_canonical_form(ctx, expr) {
            return None;
        }

        // GUARD: Block distribution when sin(4x) identity pattern is detected
        // This allows Sin4xIdentityZeroRule to see 4*sin(t)*cos(t)*(cos²-sin²) as a single product
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.has_sin4x_identity_pattern {
                return None;
            }
        }
        // Use zero-clone destructuring pattern
        let (l, r) = crate::helpers::as_mul(ctx, expr)?;

        // GUARD: Skip distribution when a factor is 1.
        // 1*(a+b) -> 1*a + 1*b is a visual no-op (MulOne is applied in rendering),
        // and produces confusing "Before/After identical" steps.
        if crate::helpers::is_one(ctx, l) || crate::helpers::is_one(ctx, r) {
            return None;
        }

        // a * (b + c) -> a*b + a*c
        if let Some((b, c)) = crate::helpers::as_add(ctx, r) {
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
            return Some(
                Rewrite::new(new_expr)
                    .desc("Distribute")
                    .local(expr, new_expr),
            );
        }

        // (b + c) * a -> b*a + c*a
        if let Some((b, c)) = crate::helpers::as_add(ctx, l) {
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
            return Some(
                Rewrite::new(new_expr)
                    .desc("Distribute")
                    .local(expr, new_expr),
            );
        }

        // Handle Division Distribution: (a + b) / c -> a/c + b/c
        // Using AddView for shape-independent n-ary handling
        if let Some((numer, denom)) = crate::helpers::as_div(ctx, expr) {
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

            // N-ARY: Use AddView for shape-independent handling of sums
            // This correctly handles ((a+b)+c), (a+(b+c)), and balanced trees
            let num_view = AddView::from_expr(ctx, numer);

            // Check if it's actually a sum (more than 1 term)
            if num_view.terms.len() > 1 {
                // Calculate total reduction potential
                let mut total_reduction: usize = 0;
                let mut any_simplifies = false;

                for &(term, _sign) in &num_view.terms {
                    let red = get_simplification_reduction(ctx, term, denom);
                    if red > 0 {
                        any_simplifies = true;
                        total_reduction += red;
                    }
                }

                // Only distribute if at least one term simplifies
                if any_simplifies {
                    // Build new terms: each term divided by denominator
                    let new_terms: Vec<ExprId> = num_view
                        .terms
                        .iter()
                        .map(|&(term, sign)| {
                            let div_term = ctx.add(Expr::Div(term, denom));
                            match sign {
                                Sign::Pos => div_term,
                                Sign::Neg => ctx.add(Expr::Neg(div_term)),
                            }
                        })
                        .collect();

                    // Rebuild as balanced sum
                    let new_expr = build_balanced_add(ctx, &new_terms);

                    // Check complexity to prevent cycles with AddFractionsRule
                    let old_complexity = cas_ast::count_nodes(ctx, expr);
                    let new_complexity = cas_ast::count_nodes(ctx, new_expr);

                    // Allow if predicted complexity (after simplification) is not worse
                    if new_complexity <= old_complexity + total_reduction {
                        return Some(
                            Rewrite::new(new_expr)
                                .desc("Distribute division (simplifying)")
                                .local(expr, new_expr),
                        );
                    }
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

/// Unwrap __hold(X) to X, otherwise return the expression unchanged
/// Delegates to canonical implementation in cas_ast::hold
fn unwrap_hold(ctx: &Context, expr: ExprId) -> ExprId {
    cas_ast::hold::unwrap_hold(ctx, expr)
}

/// Normalize a term by extracting negation from leading coefficient
/// For example: (-15)*z with flag false → 15*z with flag true
/// Returns (normalized_expr, effective_negation_flag)
fn normalize_term_sign(ctx: &Context, term: ExprId, neg: bool) -> (ExprId, bool) {
    // Check if it's a Mul with a negative number as first or second operand
    if let Expr::Mul(l, r) = ctx.get(term) {
        // Check left operand for negative number
        if let Expr::Number(n) = ctx.get(*l) {
            if n.is_negative() {
                // Flip the sign and negate the coefficient
                // We can't easily create a new expression here, so we'll compare differently
                return (term, !neg);
            }
        }
        // Check right operand for negative number
        if let Expr::Number(n) = ctx.get(*r) {
            if n.is_negative() {
                return (term, !neg);
            }
        }
    }

    // Check if it's a negative number itself
    if let Expr::Number(n) = ctx.get(term) {
        if n.is_negative() {
            return (term, !neg);
        }
    }

    (term, neg)
}

/// Check if two expressions are polynomially equal (same after expansion)
fn poly_equal(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    // Identical IDs
    if a == b {
        return true;
    }

    // First try structural comparison
    if compare_expr(ctx, a, b) == Ordering::Equal {
        return true;
    }

    let expr_a = ctx.get(a);
    let expr_b = ctx.get(b);

    // Try deep comparison for Pow expressions
    if let (Expr::Pow(base_a, exp_a), Expr::Pow(base_b, exp_b)) = (expr_a, expr_b) {
        if poly_equal(ctx, *exp_a, *exp_b) {
            return poly_equal(ctx, *base_a, *base_b);
        }
    }

    // Try deep comparison for Mul expressions (commutative)
    if let (Expr::Mul(l_a, r_a), Expr::Mul(l_b, r_b)) = (expr_a, expr_b) {
        // Try both orderings
        if (poly_equal(ctx, *l_a, *l_b) && poly_equal(ctx, *r_a, *r_b))
            || (poly_equal(ctx, *l_a, *r_b) && poly_equal(ctx, *r_a, *l_b))
        {
            return true;
        }

        // Also check for opposite coefficients: 15*z vs -15*z
        // Check if one left operand is the negation of the other
        if let (Expr::Number(n_a), Expr::Number(n_b)) = (ctx.get(*l_a), ctx.get(*l_b)) {
            if n_a == &-n_b.clone() && poly_equal(ctx, *r_a, *r_b) {
                return true; // Same up to sign
            }
        }
        if let (Expr::Number(n_a), Expr::Number(n_b)) = (ctx.get(*r_a), ctx.get(*r_b)) {
            if n_a == &-n_b.clone() && poly_equal(ctx, *l_a, *l_b) {
                return true; // Same up to sign
            }
        }
    }

    // Try deep comparison for Neg expressions
    if let (Expr::Neg(inner_a), Expr::Neg(inner_b)) = (expr_a, expr_b) {
        return poly_equal(ctx, *inner_a, *inner_b);
    }

    // For any additive expressions, flatten and compare term sets
    // This handles Add, Sub, and mixed cases
    let is_additive_a = matches!(expr_a, Expr::Add(_, _) | Expr::Sub(_, _));
    let is_additive_b = matches!(expr_b, Expr::Add(_, _) | Expr::Sub(_, _));

    if is_additive_a && is_additive_b {
        let mut terms_a: Vec<(ExprId, bool)> = Vec::new();
        let mut terms_b: Vec<(ExprId, bool)> = Vec::new();
        flatten_additive_terms(ctx, a, false, &mut terms_a);
        flatten_additive_terms(ctx, b, false, &mut terms_b);

        if terms_a.len() == terms_b.len() {
            let mut matched = vec![false; terms_b.len()];
            for (term_a, neg_a) in &terms_a {
                let mut found = false;

                // Normalize term_a: extract negation from leading coefficient if any
                let (norm_a, eff_neg_a) = normalize_term_sign(ctx, *term_a, *neg_a);

                for (j, (term_b, neg_b)) in terms_b.iter().enumerate() {
                    if matched[j] {
                        continue;
                    }

                    // Normalize term_b: extract negation from leading coefficient if any
                    let (norm_b, eff_neg_b) = normalize_term_sign(ctx, *term_b, *neg_b);

                    // Now compare with effective negation
                    if eff_neg_a != eff_neg_b {
                        continue;
                    }

                    // Use poly_equal recursively for term comparison
                    if poly_equal(ctx, norm_a, norm_b) {
                        matched[j] = true;
                        found = true;
                        break;
                    }
                }
                if !found {
                    return false;
                }
            }
            if matched.iter().all(|&m| m) {
                return true;
            }
        }
    }

    // Fallback: try polynomial comparison for univariate case
    let vars_a: Vec<_> = crate::rules::algebra::collect_variables(ctx, a)
        .into_iter()
        .collect();
    let vars_b: Vec<_> = crate::rules::algebra::collect_variables(ctx, b)
        .into_iter()
        .collect();

    // Only compare if same single variable
    if vars_a.len() == 1 && vars_b.len() == 1 && vars_a[0] == vars_b[0] {
        let var = &vars_a[0];
        if let (Ok(poly_a), Ok(poly_b)) = (
            Polynomial::from_expr(ctx, a, var),
            Polynomial::from_expr(ctx, b, var),
        ) {
            return poly_a == poly_b;
        }
    }

    false
}

/// Flatten an additive expression into a list of (term, is_negated) pairs
///
/// Uses canonical AddView from nary.rs for shape-independence and __hold transparency.
/// (See ARCHITECTURE.md "Canonical Utilities Registry")
fn flatten_additive_terms(
    ctx: &Context,
    expr: ExprId,
    negated: bool,
    terms: &mut Vec<(ExprId, bool)>,
) {
    use crate::nary::{add_terms_signed, Sign};

    // Use canonical AddView
    let signed_terms = add_terms_signed(ctx, expr);

    for (term, sign) in signed_terms {
        // XOR the incoming negation with the sign from AddView
        let is_negated = match sign {
            Sign::Pos => negated,
            Sign::Neg => !negated,
        };
        terms.push((term, is_negated));
    }
}

// AnnihilationRule: Detects and cancels terms like x - x or __hold(sum) - sum
// Domain Mode Policy: Like AddInverseRule, we must respect domain_mode
// because if `x` can be undefined (e.g., a/(a-1) when a=1), then x - x
// is undefined, not 0.
// - Strict: only if no term contains potentially-undefined subexpressions
// - Assume: always apply (educational mode assumption: all expressions are defined)
// - Generic: same as Assume
define_rule!(AnnihilationRule, "Annihilation", |ctx, expr, parent_ctx| {
    // Helper: check if expression contains any Div with non-literal denominator
    // Delegates to canonical implementation that handles Hold/Matrix
    fn has_undefined_risk(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
        crate::collect::has_undefined_risk(ctx, expr)
    }

    // Only process Add/Sub expressions
    // CLONE_OK: Multi-branch match on Mul/Add/Sub/Pow requires owned Expr
    let expr_data = ctx.get(expr).clone();
    if !matches!(expr_data, Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    // Flatten all terms
    let mut terms: Vec<(ExprId, bool)> = Vec::new();
    flatten_additive_terms(ctx, expr, false, &mut terms);

    if terms.len() < 2 {
        return None;
    }

    // CASE 1: Look for simple pairs that cancel (term and its negation)
    for i in 0..terms.len() {
        for j in (i + 1)..terms.len() {
            let (term_i, neg_i) = &terms[i];
            let (term_j, neg_j) = &terms[j];

            // Only if opposite signs
            if neg_i == neg_j {
                continue;
            }

            // Unwrap __hold for comparison
            let unwrapped_i = unwrap_hold(ctx, *term_i);
            let unwrapped_j = unwrap_hold(ctx, *term_j);

            // Check structural or polynomial equality
            if poly_equal(ctx, unwrapped_i, unwrapped_j) {
                // These terms cancel. If they're the only 2 terms, result is 0
                if terms.len() == 2 {
                    // DOMAIN MODE GATE: Check for undefined risk
                    let domain_mode = parent_ctx.domain_mode();
                    let either_has_risk =
                        has_undefined_risk(ctx, *term_i) || has_undefined_risk(ctx, *term_j);

                    if domain_mode == crate::DomainMode::Strict && either_has_risk {
                        return None;
                    }

                    // Note: domain assumption would be emitted here if Assume mode and either_has_risk
                    // but assumption_events are not emitted for this case yet

                    let zero = ctx.num(0);
                    return Some(Rewrite::new(zero).desc("x - x = 0"));
                }
            }
        }
    }

    // CASE 2: Handle __hold(A+B+...) with distributed -(A) -(B) -(...)
    // Find __hold terms and check if remaining negated terms sum to their content
    for (idx, (term, is_neg)) in terms.iter().enumerate() {
        if *is_neg {
            continue; // Only check positive __hold terms
        }

        // Check if this is a __hold using canonical helper
        if cas_ast::hold::is_hold(ctx, *term) {
            // Unwrap the held content
            let held_content = cas_ast::hold::unwrap_hold(ctx, *term);

            // Flatten the held content to get its terms
            let mut held_terms: Vec<(ExprId, bool)> = Vec::new();
            flatten_additive_terms(ctx, held_content, false, &mut held_terms);

            // Get all other terms (excluding this __hold)
            let other_terms: Vec<(ExprId, bool)> = terms
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, t)| *t)
                .collect();

            // Check if held_terms and other_terms cancel out
            // They cancel if for each held term there's an opposite signed other term
            if held_terms.len() == other_terms.len() {
                let mut all_cancel = true;
                let mut used = vec![false; other_terms.len()];

                for (held_term, held_neg) in &held_terms {
                    let mut found = false;

                    for (j, (other_term, other_neg)) in other_terms.iter().enumerate() {
                        if used[j] {
                            continue;
                        }

                        // Check if terms cancel (one positive, one negative equivalently)
                        // Case 1: Same term with opposite flags
                        if *other_neg != *held_neg {
                            // Use poly_equal for more robust comparison
                            // This handles cases where expressions are semantically equal
                            // but structurally different (e.g., Mul(15,x) vs Mul(x,15))
                            if poly_equal(ctx, *held_term, *other_term) {
                                used[j] = true;
                                found = true;
                                break;
                            }
                        }

                        // Case 2: Number with same flag but opposite value (e.g., 1 vs -1)
                        if *other_neg == *held_neg {
                            if let (Expr::Number(n1), Expr::Number(n2)) =
                                (ctx.get(*held_term), ctx.get(*other_term))
                            {
                                if n1 == &-n2.clone() {
                                    used[j] = true;
                                    found = true;
                                    break;
                                }
                            }
                        }
                    }

                    if !found {
                        all_cancel = false;
                        break;
                    }
                }

                if all_cancel && used.iter().all(|&u| u) {
                    let zero = ctx.num(0);
                    return Some(Rewrite::new(zero).desc("__hold(sum) - sum = 0"));
                }
            }
        }
    }

    None
});

// CombineLikeTermsRule: Collects like terms in Add/Mul expressions
// Now uses collect_with_semantics for domain_mode awareness:
// - Strict: refuses to cancel terms with undefined risk (e.g., x/(x+1) - x/(x+1))
// - Assume: cancels with domain_assumption warning
// - Generic: cancels unconditionally
define_rule!(
    CombineLikeTermsRule,
    "Combine Like Terms",
    |ctx, expr, parent_ctx| {
        // Only try to collect if it's an Add or Mul
        let is_add_or_mul = matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Mul(_, _));

        if is_add_or_mul {
            // CRITICAL: Do NOT apply to non-commutative expressions (e.g., matrices)
            if !ctx.is_mul_commutative(expr) {
                return None;
            }

            // Use semantics-aware collect that respects domain_mode
            let result = crate::collect::collect_with_semantics(ctx, expr, parent_ctx)?;

            // Check if structurally different to avoid infinite loops with ID regeneration
            if crate::ordering::compare_expr(ctx, result.new_expr, expr) == Ordering::Equal {
                return None;
            }

            // V2.14.26: Skip trivial changes that only normalize -1 coefficients
            // without actually combining or cancelling any terms.
            // This avoids noisy steps like "-1·x → -x" that don't add didactic value.
            if result.cancelled.is_empty() && result.combined.is_empty() {
                return None;
            }

            // V2.9.18: Restore granular focus using CollectResult's cancelled/combined groups
            // This provides specific focus like "5 - 5 → 0" for didactic clarity
            // Timeline highlighting uses step.path separately for broader context
            let (before_local, after_local, description) = select_best_focus(ctx, &result);

            let mut rewrite = Rewrite::new(result.new_expr).desc(description);
            if let (Some(before), Some(after)) = (before_local, after_local) {
                rewrite = rewrite.local(before, after);
            }
            return Some(rewrite);
        }
        None
    }
);

/// Build an additive expression from a list of terms (for focus display)
fn build_additive_expr(ctx: &mut Context, terms: &[ExprId]) -> ExprId {
    if terms.is_empty() {
        return ctx.num(0);
    }
    let mut result = terms[0];
    for &term in &terms[1..] {
        result = ctx.add(Expr::Add(result, term));
    }
    result
}

/// Select focus for didactic display from a CollectResult
/// Shows ALL combined and cancelled groups together for complete picture
fn select_best_focus(
    ctx: &mut Context,
    result: &crate::collect::CollectResult,
) -> (Option<ExprId>, Option<ExprId>, String) {
    // Collect all original terms and all result terms from all groups
    let mut all_before_terms: Vec<ExprId> = Vec::new();
    let mut all_after_terms: Vec<ExprId> = Vec::new();
    let mut has_cancellation = false;
    let mut has_combination = false;

    // Add cancelled groups (result is 0, but we skip adding 0 since it doesn't change sum)
    for cancelled in &result.cancelled {
        all_before_terms.extend(&cancelled.original_terms);
        has_cancellation = true;
        // Don't add 0 to after terms - it's implicit
    }

    // Add combined groups
    for combined in &result.combined {
        all_before_terms.extend(&combined.original_terms);
        all_after_terms.push(combined.combined_term);
        has_combination = true;
    }

    // If we have no groups, fallback
    if all_before_terms.is_empty() {
        return (None, None, "Combine like terms".to_string());
    }

    // Build the before expression from all original terms
    let focus_before = build_additive_expr(ctx, &all_before_terms);

    // Build the after expression
    let focus_after = if all_after_terms.is_empty() {
        // Only cancellations, result is 0
        ctx.num(0)
    } else {
        build_additive_expr(ctx, &all_after_terms)
    };

    // Choose appropriate description
    let description = if has_cancellation && has_combination {
        "Cancel and combine like terms".to_string()
    } else if has_cancellation {
        "Cancel opposite terms".to_string()
    } else {
        "Combine like terms".to_string()
    };

    (Some(focus_before), Some(focus_after), description)
}

/// Count the number of terms in a sum/difference expression
/// Returns the count of additive terms (flattening nested Add/Sub)
fn count_additive_terms(ctx: &Context, expr: ExprId) -> usize {
    match ctx.get(expr) {
        Expr::Add(l, r) => count_additive_terms(ctx, *l) + count_additive_terms(ctx, *r),
        Expr::Sub(l, r) => count_additive_terms(ctx, *l) + count_additive_terms(ctx, *r),
        Expr::Neg(inner) => count_additive_terms(ctx, *inner),
        _ => 1, // A single term (Variable, Number, Mul, Pow, etc.)
    }
}

/// BinomialExpansionRule: (a + b)^n → expanded polynomial
/// ONLY expands true binomials (exactly 2 terms).
/// Multinomial expansion (>2 terms) is NOT done by default to avoid explosion.
/// Use explicit expand() mode for multinomial expansion.
/// Implements Rule directly to access ParentContext
pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(DistributeRule));
    simplifier.add_rule(Box::new(AnnihilationRule));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(BinomialExpansionRule));
    // V2.15.8: ExpandSmallBinomialPowRule - controlled by autoexpand_binomials flag
    // Enable via REPL: set autoexpand_binomials on
    simplifier.add_rule(Box::new(ExpandSmallBinomialPowRule));
    simplifier.add_rule(Box::new(AutoExpandPowSumRule));
    simplifier.add_rule(Box::new(AutoExpandSubCancelRule));
    simplifier.add_rule(Box::new(PolynomialIdentityZeroRule));
    // V2.15.8: HeuristicPolyNormalizeAddRule - poly-normalize Add/Sub in Heuristic mode
    // V2.15.9: HeuristicExtractCommonFactorAddRule - extract common factors first (priority 110)
    simplifier.add_rule(Box::new(HeuristicExtractCommonFactorAddRule));
    simplifier.add_rule(Box::new(HeuristicPolyNormalizeAddRule));
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
        // x^2 * (x + 3) - use x^2 (not an integer) so guard doesn't block
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let x_sq = ctx.add(Expr::Pow(x, two));
        let add = ctx.add(Expr::Add(x, three));
        let expr = ctx.add(Expr::Mul(x_sq, add));

        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        // Should be (x^2 * x) + (x^2 * 3) before further simplification
        // Note: x^2*x -> x^3 happens in a later pass, not in DistributeRule
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "x^2 * x + x^2 * 3" // Canonical: polynomial order (x terms before constants)
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
        let ln_x = ctx.call("ln", vec![x]);
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

    #[test]
    fn test_polynomial_identity_zero_rule() {
        // Test: (a+b)^2 - (a^2 + 2ab + b^2) = 0
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");

        // (a+b)^2
        let a_plus_b = ctx.add(Expr::Add(a, b));
        let two = ctx.num(2);
        let a_plus_b_sq = ctx.add(Expr::Pow(a_plus_b, two));

        // a^2 + 2ab + b^2
        let a_sq = ctx.add(Expr::Pow(a, two));
        let b_sq = ctx.add(Expr::Pow(b, two));
        let ab = ctx.add(Expr::Mul(a, b));
        let two_ab = ctx.add(Expr::Mul(two, ab));
        let sum1 = ctx.add(Expr::Add(a_sq, two_ab));
        let rhs = ctx.add(Expr::Add(sum1, b_sq));

        // (a+b)^2 - (a^2 + 2ab + b^2)
        let expr = ctx.add(Expr::Sub(a_plus_b_sq, rhs));

        let rule = PolynomialIdentityZeroRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        // Should simplify to 0
        assert!(rewrite.is_some(), "Polynomial identity should be detected");
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.unwrap().new_expr
                }
            ),
            "0"
        );
    }

    #[test]
    fn test_polynomial_identity_zero_rule_non_identity() {
        // Test: (a+b)^2 - a^2 ≠ 0 (not an identity)
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");

        // (a+b)^2
        let a_plus_b = ctx.add(Expr::Add(a, b));
        let two = ctx.num(2);
        let a_plus_b_sq = ctx.add(Expr::Pow(a_plus_b, two));

        // a^2
        let a_sq = ctx.add(Expr::Pow(a, two));

        // (a+b)^2 - a^2
        let expr = ctx.add(Expr::Sub(a_plus_b_sq, a_sq));

        let rule = PolynomialIdentityZeroRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        // Should NOT return a rewrite (not an identity to 0)
        assert!(rewrite.is_none(), "Non-identity should not trigger rule");
    }
}
