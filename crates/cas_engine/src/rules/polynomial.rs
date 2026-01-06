use crate::build::mul2_raw;
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
use num_traits::{One, Zero};
use num_traits::{Signed, ToPrimitive};
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
                    after_local: None,
                    assumption_events: Default::default(),
            required_conditions: vec![],
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
                    after_local: None,
                    assumption_events: Default::default(),
            required_conditions: vec![],
                });
            }
        }

        // Handle Division Distribution: (a + b) / c -> a/c + b/c
        // Using AddView for shape-independent n-ary handling
        if let Expr::Div(l, r) = expr_data {
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
            let num_view = AddView::from_expr(ctx, l);

            // Check if it's actually a sum (more than 1 term)
            if num_view.terms.len() > 1 {
                // Calculate total reduction potential
                let mut total_reduction: usize = 0;
                let mut any_simplifies = false;

                for &(term, _sign) in &num_view.terms {
                    let red = get_simplification_reduction(ctx, term, r);
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
                            let div_term = ctx.add(Expr::Div(term, r));
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
                        return Some(Rewrite {
                            new_expr,
                            description: "Distribute division (simplifying)".to_string(),
                            before_local: None,
                            after_local: None,
                            assumption_events: Default::default(),
            required_conditions: vec![],
                        });
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
fn unwrap_hold(ctx: &Context, expr: ExprId) -> ExprId {
    if let Expr::Function(name, args) = ctx.get(expr) {
        if name == "__hold" && args.len() == 1 {
            return args[0];
        }
    }
    expr
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
    use crate::domain::Proof;
    use crate::helpers::prove_nonzero;

    // Helper: check if expression contains any Div with non-literal denominator
    fn has_undefined_risk(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
        let mut stack = vec![expr];
        while let Some(e) = stack.pop() {
            match ctx.get(e) {
                Expr::Div(_, den) => {
                    if prove_nonzero(ctx, *den) != Proof::Proven {
                        return true;
                    }
                    stack.push(*den);
                }
                Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Pow(l, r) => {
                    stack.push(*l);
                    stack.push(*r);
                }
                Expr::Neg(inner) => {
                    stack.push(*inner);
                }
                Expr::Function(_, args) => {
                    for arg in args {
                        stack.push(*arg);
                    }
                }
                _ => {}
            }
        }
        false
    }

    // Only process Add/Sub expressions
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
                    return Some(Rewrite {
                        new_expr: zero,
                        description: "x - x = 0".to_string(),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(),
            required_conditions: vec![],
                    });
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

        // Check if this is a __hold
        if let Expr::Function(name, args) = ctx.get(*term) {
            if name == "__hold" && args.len() == 1 {
                let held_content = args[0];

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
                        return Some(Rewrite {
                            new_expr: zero,
                            description: "__hold(sum) - sum = 0".to_string(),
                            before_local: None,
                            after_local: None,
                            assumption_events: Default::default(),
            required_conditions: vec![],
                        });
                    }
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

            // Note: result.assumption contains warning about denominators
            // but assumption_events are not emitted for this case yet

            return Some(Rewrite {
                new_expr: result.new_expr,
                description: "Combine like terms".to_string(),
                before_local: None,
                after_local: None,
                assumption_events: Default::default(),
            required_conditions: vec![],
            });
        }
        None
    }
);

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
pub struct BinomialExpansionRule;

impl crate::rule::Rule for BinomialExpansionRule {
    fn name(&self) -> &str {
        "Binomial Expansion"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Skip if expression is in canonical (elegant) form
        if crate::canonical_forms::is_canonical_form(ctx, expr) {
            return None;
        }

        // GUARD: Don't expand if this expression is protected as a sqrt-square base
        // This is set by pre-scan when this expr is inside sqrt(u²) or sqrt(u*u)
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.is_sqrt_square_protected(expr) {
                // Protected from expansion - let sqrt(u²) → |u| shortcut fire instead
                return None;
            }
        }

        // (a + b)^n - ONLY true binomials (exactly 2 terms)
        let expr_data = ctx.get(expr).clone();
        if let Expr::Pow(base, exp) = expr_data {
            let base_data = ctx.get(base).clone();

            // CRITICAL GUARD: Only expand if base has exactly 2 terms
            // This prevents multinomial expansion like (1 + x1 + x2 + ... + x7)^7
            // which would produce thousands of terms
            let term_count = count_additive_terms(ctx, base);
            if term_count != 2 {
                return None; // Not a binomial, skip expansion
            }

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
                        // Only expand binomials in explicit expand mode
                        // In Standard mode, preserve structure like (x+1)^3
                        // This prevents unwanted expansion when doing poly_gcd(a*g, b*g) - g
                        if !parent_ctx.is_expand_mode() {
                            return None;
                        }

                        // Limit expansion to reasonable exponents even in expand mode
                        if (2..=20).contains(&n_val) {
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
                            for &term in terms.iter().skip(1) {
                                expanded = ctx.add(Expr::Add(expanded, term));
                            }

                            return Some(Rewrite {
                                new_expr: expanded,
                                description: format!("Expand binomial power ^{}", n_val),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
            required_conditions: vec![],
                            });
                        }
                    }
                }
            }
        }
        None
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Pow"])
    }
}

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

// =============================================================================
// AutoExpandPowSumRule: Auto-expand cheap binomials/polynomials within budget
// =============================================================================

/// Rule for auto-expanding cheap power-of-sum expressions within budget limits.
/// Unlike BinomialExpansionRule, this is opt-in and checks budget constraints.
///
/// Only triggers when `parent_ctx.is_auto_expand()` is true.
/// Respects budget limits: max_pow_exp, max_base_terms, max_generated_terms, max_vars.
pub struct AutoExpandPowSumRule;

impl AutoExpandPowSumRule {
    /// Count additive terms in an expression
    fn count_add_terms(ctx: &Context, expr: ExprId) -> u32 {
        match ctx.get(expr) {
            Expr::Add(l, r) => Self::count_add_terms(ctx, *l) + Self::count_add_terms(ctx, *r),
            _ => 1,
        }
    }

    /// Count unique variables in an expression
    fn count_variables(
        ctx: &Context,
        expr: ExprId,
        visited: &mut std::collections::HashSet<String>,
    ) {
        match ctx.get(expr) {
            Expr::Variable(name) => {
                visited.insert(name.clone());
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
                Self::count_variables(ctx, *l, visited);
                Self::count_variables(ctx, *r, visited);
            }
            Expr::Pow(b, e) => {
                Self::count_variables(ctx, *b, visited);
                Self::count_variables(ctx, *e, visited);
            }
            Expr::Neg(e) => {
                Self::count_variables(ctx, *e, visited);
            }
            Expr::Function(_, args) => {
                for arg in args {
                    Self::count_variables(ctx, *arg, visited);
                }
            }
            _ => {}
        }
    }

    /// Estimate number of terms generated by multinomial expansion: C(n+k-1, k-1)
    /// For binomial (k=2): C(n+1, 1) = n+1
    fn estimate_terms(k: u32, n: u32) -> u32 {
        // Multinomial: number of terms = C(n+k-1, k-1)
        // For binomial: C(n+1, 1) = n+1
        // For trinomial: C(n+2, 2) = (n+1)(n+2)/2
        // etc.
        if k <= 1 {
            return 1;
        }
        // Compute C(n+k-1, k-1) = C(n+k-1, n)
        let top = n + k - 1;
        let bottom = k - 1;
        binomial_coeff(top, bottom)
    }
}

impl crate::rule::Rule for AutoExpandPowSumRule {
    fn name(&self) -> &str {
        "Auto Expand Power Sum"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Expand if: global auto-expand mode OR inside a marked cancellation context
        // (e.g., difference quotient like ((x+h)^n - x^n)/h)
        let in_expand_context = parent_ctx.in_auto_expand_context();
        if !(parent_ctx.is_auto_expand() || in_expand_context) {
            return None;
        }

        // Get budget
        let budget = parent_ctx.auto_expand_budget()?;

        // Skip if expression is in canonical form
        if crate::canonical_forms::is_canonical_form(ctx, expr) {
            return None;
        }

        // Pattern: Pow(Add(...), n)
        let expr_data = ctx.get(expr).clone();
        if let Expr::Pow(base, exp) = expr_data {
            // Check exponent is a small positive integer
            let exp_data = ctx.get(exp).clone();
            if let Expr::Number(n) = exp_data {
                if !n.is_integer() || n.is_negative() {
                    return None;
                }
                let n_val = n.to_integer().to_u32()?;

                // Budget check 1: max_pow_exp
                if n_val > budget.max_pow_exp {
                    return None;
                }
                // At least square to be useful
                if n_val < 2 {
                    return None;
                }

                // Check base is an Add
                let base_data = ctx.get(base).clone();
                if !matches!(base_data, Expr::Add(_, _)) {
                    return None;
                }

                // Budget check 2: max_base_terms
                let num_terms = Self::count_add_terms(ctx, base);
                if num_terms > budget.max_base_terms {
                    return None;
                }

                // Budget check 3: max_generated_terms
                let estimated_result_terms = Self::estimate_terms(num_terms, n_val);
                if estimated_result_terms > budget.max_generated_terms {
                    return None;
                }

                // Budget check 4: max_vars
                let mut vars = std::collections::HashSet::new();
                Self::count_variables(ctx, base, &mut vars);
                if vars.len() as u32 > budget.max_vars {
                    return None;
                }

                // All budget checks passed!
                // For binomials (2 terms), use binomial expansion
                if num_terms == 2 {
                    // Extract a and b from Add(a, b)
                    if let Expr::Add(a, b) = base_data {
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
                                let exp_a_id = ctx.num(exp_a as i64);
                                ctx.add(Expr::Pow(a, exp_a_id))
                            };

                            let term_b = if exp_b == 0 {
                                ctx.num(1)
                            } else if exp_b == 1 {
                                b
                            } else {
                                let exp_b_id = ctx.num(exp_b as i64);
                                ctx.add(Expr::Pow(b, exp_b_id))
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
                        for &term in terms.iter().skip(1) {
                            expanded = ctx.add(Expr::Add(expanded, term));
                        }

                        return Some(Rewrite {
                            new_expr: expanded,
                            description: format!("Auto-expand (a+b)^{}", n_val),
                            before_local: None,
                            after_local: None,
                            assumption_events: Default::default(),
            required_conditions: vec![],
                        });
                    }
                }

                // For trinomials and higher, use general multinomial expansion
                // (more complex, skip for now - only binomials are auto-expanded)
                // Users can use explicit expand() for higher-order polynomials
            }
        }

        None
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Pow"])
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        // Include RATIONALIZE so auto-expand can clean up after rationalization
        // e.g., 1/(1+√2+√3) → ... → (1+√2)² - 3 → needs auto-expand to become 2√2
        crate::phase::PhaseMask::CORE
            | crate::phase::PhaseMask::TRANSFORM
            | crate::phase::PhaseMask::RATIONALIZE
    }
}

// =============================================================================
// AutoExpandSubCancelRule: Zero-shortcut for Sub(Pow(Add..), polynomial)
// =============================================================================
//
// Detects patterns like `(x+1)^2 - (x^2 + 2*x + 1)` and proves cancellation
// to 0 via polynomial comparison WITHOUT expanding the AST.
//
// Strategy:
// 1. Detect Sub(lhs, rhs) where one side is Pow(Add.., n)
// 2. Convert both sides to MultiPoly representation
// 3. If P - Q = 0, return Rewrite to 0

use crate::multipoly::{MultiPoly, PolyBudget};

/// AutoExpandSubCancelRule: Zero-shortcut for Sub(Pow(Add..), polynomial)
/// Priority 95 (higher than AutoExpandPowSumRule at 50)
pub struct AutoExpandSubCancelRule;

impl AutoExpandSubCancelRule {
    /// Convert expression to MultiPoly (returns None if not polynomial-representable)
    fn expr_to_multipoly(
        ctx: &Context,
        id: ExprId,
        vars: &mut Vec<String>,
        budget: &PolyBudget,
    ) -> Option<MultiPoly> {
        Self::expr_to_multipoly_inner(ctx, id, vars, budget, 0)
    }

    fn expr_to_multipoly_inner(
        ctx: &Context,
        id: ExprId,
        vars: &mut Vec<String>,
        budget: &PolyBudget,
        depth: usize,
    ) -> Option<MultiPoly> {
        // Depth limit to prevent stack overflow
        if depth > 50 {
            return None;
        }

        match ctx.get(id) {
            Expr::Number(n) => {
                // Constant polynomial
                Some(MultiPoly::from_const(n.clone()))
            }
            Expr::Variable(name) => {
                // Variable: ensure it's in our vars list
                if !vars.contains(name) {
                    if vars.len() >= 4 {
                        return None; // Too many variables
                    }
                    vars.push(name.clone());
                }
                // Create polynomial for this variable
                let idx = vars.iter().position(|v| v == name)?;
                let mut mono = vec![0u32; vars.len()];
                mono[idx] = 1;
                let terms = vec![(BigRational::one(), mono)];
                Some(MultiPoly {
                    vars: vars.clone(),
                    terms,
                })
            }
            Expr::Add(l, r) => {
                let p = Self::expr_to_multipoly_inner(ctx, *l, vars, budget, depth + 1)?;
                let q = Self::expr_to_multipoly_inner(ctx, *r, vars, budget, depth + 1)?;
                // Align variables
                let (p, q) = Self::align_vars(p, q, vars);
                p.add(&q).ok()
            }
            Expr::Sub(l, r) => {
                let p = Self::expr_to_multipoly_inner(ctx, *l, vars, budget, depth + 1)?;
                let q = Self::expr_to_multipoly_inner(ctx, *r, vars, budget, depth + 1)?;
                let (p, q) = Self::align_vars(p, q, vars);
                p.sub(&q).ok()
            }
            Expr::Mul(l, r) => {
                let p = Self::expr_to_multipoly_inner(ctx, *l, vars, budget, depth + 1)?;
                let q = Self::expr_to_multipoly_inner(ctx, *r, vars, budget, depth + 1)?;
                let (p, q) = Self::align_vars(p, q, vars);
                p.mul(&q, budget).ok()
            }
            Expr::Neg(inner) => {
                let p = Self::expr_to_multipoly_inner(ctx, *inner, vars, budget, depth + 1)?;
                Some(p.neg())
            }
            Expr::Pow(base, exp) => {
                // Only handle integer exponents >= 0
                if let Expr::Number(n) = ctx.get(*exp) {
                    if n.is_integer() && !n.is_negative() {
                        let exp_val = n.to_integer().to_u32()?;
                        if exp_val > budget.max_total_degree {
                            return None;
                        }
                        // Recursively convert base - this may grow vars
                        let base_poly =
                            Self::expr_to_multipoly_inner(ctx, *base, vars, budget, depth + 1)?;

                        // Handle exp == 0 case
                        if exp_val == 0 {
                            return Some(MultiPoly::one(vars.clone()));
                        }

                        // Compute base^exp via repeated multiplication
                        // Start with base aligned to current vars
                        let base_aligned = Self::align_to_vars(&base_poly, vars);
                        let mut result = base_aligned.clone();
                        for _ in 1..exp_val {
                            result = result.mul(&base_aligned, budget).ok()?;
                            if result.num_terms() > budget.max_terms {
                                return None;
                            }
                        }
                        return Some(result);
                    }
                }
                None
            }
            _ => None, // Not polynomial
        }
    }

    /// Align two polynomials to have the same variable set
    fn align_vars(p: MultiPoly, q: MultiPoly, target_vars: &[String]) -> (MultiPoly, MultiPoly) {
        (
            Self::align_to_vars(&p, target_vars),
            Self::align_to_vars(&q, target_vars),
        )
    }

    /// Align a polynomial to the target variable set
    fn align_to_vars(p: &MultiPoly, target_vars: &[String]) -> MultiPoly {
        if p.vars == target_vars {
            return p.clone();
        }
        // Reindex monomials to target_vars
        let mut new_terms = Vec::new();
        for (coeff, mono) in &p.terms {
            let mut new_mono = vec![0u32; target_vars.len()];
            for (i, var) in p.vars.iter().enumerate() {
                if let Some(target_idx) = target_vars.iter().position(|v| v == var) {
                    new_mono[target_idx] = mono[i];
                }
            }
            new_terms.push((coeff.clone(), new_mono));
        }
        MultiPoly {
            vars: target_vars.to_vec(),
            terms: new_terms,
        }
    }
}

impl crate::rule::Rule for AutoExpandSubCancelRule {
    fn name(&self) -> &str {
        "AutoExpandSubCancelRule"
    }

    fn priority(&self) -> i32 {
        95
    }

    fn allowed_phases(&self) -> PhaseMask {
        PhaseMask::TRANSFORM
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Only trigger if in auto-expand context (marked by scanner)
        // Use _for_expr to also check if current node is marked (not just ancestors)
        if !parent_ctx.in_auto_expand_context_for_expr(expr) {
            return None;
        }

        // Must be Sub or Add to be a cancellation candidate
        let is_sub_or_add = matches!(ctx.get(expr), Expr::Sub(_, _) | Expr::Add(_, _));
        if !is_sub_or_add {
            return None;
        }

        // Budget for polynomial conversion
        let budget = PolyBudget {
            max_terms: 100,
            max_total_degree: 8,
            max_pow_exp: 4, // Small limit for cancellation checks
        };

        // Convert entire expression to MultiPoly
        // For Sub(a,b) this computes a-b
        // For Add(a, Neg(b), Neg(c), ...) this computes a + (-b) + (-c) + ...
        // If the result is 0, we have cancellation
        let mut vars = Vec::new();
        let poly = Self::expr_to_multipoly(ctx, expr, &mut vars, &budget)?;

        // If the result is zero, we have proved cancellation!
        if poly.is_zero() {
            let zero = ctx.num(0);
            return Some(Rewrite {
                new_expr: zero,
                description: "Polynomial equality: expressions cancel to 0".to_string(),
                before_local: None,
                after_local: None,
                assumption_events: Default::default(),
            required_conditions: vec![],
            });
        }

        None
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(DistributeRule));
    simplifier.add_rule(Box::new(AnnihilationRule));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(BinomialExpansionRule));
    simplifier.add_rule(Box::new(AutoExpandPowSumRule));
    simplifier.add_rule(Box::new(AutoExpandSubCancelRule));
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
            "x^2 * 3 + x^2 * x" // Exact Distribute output before simplification
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
