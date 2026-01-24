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
                        return Some(Rewrite::new(zero).desc("__hold(sum) - sum = 0"));
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
        // CLONE_OK: Multi-branch match on Pow followed by nested Number checks
        let expr_data = ctx.get(expr).clone();
        if let Expr::Pow(base, exp) = expr_data {
            // CLONE_OK: Nested Add inspection after Pow destructuring
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

            // CLONE_OK: Exponent inspection for Neg/Number patterns
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

                            return Some(
                                Rewrite::new(expanded)
                                    .desc(format!("Expand binomial power ^{}", n_val)),
                            );
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
            Expr::Variable(sym_id) => {
                let name = ctx.sym_name(*sym_id).to_string();
                visited.insert(name);
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

        // Get budget - use default if in context but no explicit budget set
        let default_budget = crate::phase::ExpandBudget::default();
        let budget = parent_ctx.auto_expand_budget().unwrap_or(&default_budget);

        // Skip if expression is in canonical form
        if crate::canonical_forms::is_canonical_form(ctx, expr) {
            return None;
        }

        // Pattern: Pow(Add(...), n) - use zero-clone destructuring
        let (base, exp) = crate::helpers::as_pow(ctx, expr)?;

        // Check exponent is a small positive integer
        let n_val = {
            let exp_expr = ctx.get(exp);
            match exp_expr {
                Expr::Number(n) if n.is_integer() && !n.is_negative() => n.to_integer().to_u32()?,
                _ => return None,
            }
        };

        // Budget check 1: max_pow_exp
        if n_val > budget.max_pow_exp {
            return None;
        }
        // At least square to be useful
        if n_val < 2 {
            return None;
        }

        // Check base is an Add and extract terms
        let (a, b) = match crate::helpers::as_add(ctx, base) {
            Some((a, b)) => (a, b),
            None => return None,
        };

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
            // Use a and b extracted above
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

            return Some(Rewrite::new(expanded).desc(format!("Auto-expand (a+b)^{}", n_val)));
        }

        // For trinomials and higher, use general multinomial expansion
        // (more complex, skip for now - only binomials are auto-expanded)
        // Users can use explicit expand() for higher-order polynomials

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

    fn importance(&self) -> crate::step::ImportanceLevel {
        // Auto-expand steps are didactically important: users should see the expansion
        crate::step::ImportanceLevel::Medium
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
            Expr::Variable(sym_id) => {
                // Variable: ensure it's in our vars list
                let name = ctx.sym_name(*sym_id).to_string();
                if !vars.contains(&name) {
                    if vars.len() >= 4 {
                        return None; // Too many variables
                    }
                    vars.push(name.clone());
                }
                // Create polynomial for this variable
                let idx = vars.iter().position(|v| v == &name)?;
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
            return Some(Rewrite::new(zero).desc("Polynomial equality: expressions cancel to 0"));
        }

        None
    }
}

// =============================================================================
// PolynomialIdentityZeroRule: Detect polynomial identities that equal 0
// =============================================================================
//
// This rule normalizes any Add/Sub expression to polynomial form and checks
// if it equals 0. Unlike AutoExpandSubCancelRule, it runs ALWAYS (not just
// in marked contexts), but with stricter budgets to avoid explosions.
//
// Examples that now simplify to 0:
// - (a+b+c)^3 - (a^3+b^3+c^3 + 3(a+b)(b+c)(c+a))
// - Sophie-Germain identity
// - Any polynomial identity with small degree/terms

/// PolynomialIdentityZeroRule: Always-on polynomial identity detector
/// Converts expressions to MultiPoly form and checks if result is 0.
/// Priority 90 (lower than AutoExpandSubCancelRule at 95 to avoid duplicate work)
pub struct PolynomialIdentityZeroRule;

impl PolynomialIdentityZeroRule {
    /// Budget limits for polynomial conversion
    /// V2.15.8: Increased max_pow_exp to 6 for binomial identities like (x+1)^5 - expansion = 0
    fn poly_budget() -> PolyBudget {
        PolyBudget {
            max_terms: 50,       // Max monomials in result
            max_total_degree: 6, // Max total degree (covers up to n=6)
            max_pow_exp: 6,      // Max exponent in Pow nodes
        }
    }

    /// Quick check: does expression look polynomial-like and worth checking?
    /// Avoids expensive conversion for obviously non-polynomial expressions.
    fn is_polynomial_candidate(ctx: &Context, expr: ExprId) -> bool {
        Self::is_polynomial_candidate_inner(ctx, expr, 0)
    }

    fn is_polynomial_candidate_inner(ctx: &Context, expr: ExprId, depth: usize) -> bool {
        if depth > 30 {
            return false; // Too deep
        }

        match ctx.get(expr) {
            Expr::Number(_) | Expr::Variable(_) => true,
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
                Self::is_polynomial_candidate_inner(ctx, *l, depth + 1)
                    && Self::is_polynomial_candidate_inner(ctx, *r, depth + 1)
            }
            Expr::Neg(inner) => Self::is_polynomial_candidate_inner(ctx, *inner, depth + 1),
            Expr::Pow(base, exp) => {
                // Only integer exponents, and check base
                if let Expr::Number(n) = ctx.get(*exp) {
                    if n.is_integer() && !n.is_negative() {
                        use num_traits::ToPrimitive;
                        if let Some(e) = n.to_integer().to_u32() {
                            if e <= 6 {
                                // V2.15.8: Extended budget for binomial identities
                                return Self::is_polynomial_candidate_inner(ctx, *base, depth + 1);
                            }
                        }
                    }
                }
                false
            }
            _ => false, // Functions, Division, etc. are not polynomial
        }
    }

    /// Convert expression to MultiPoly (reusing AutoExpandSubCancelRule's method)
    fn expr_to_multipoly(
        ctx: &Context,
        id: ExprId,
        vars: &mut Vec<String>,
        budget: &PolyBudget,
    ) -> Option<MultiPoly> {
        AutoExpandSubCancelRule::expr_to_multipoly(ctx, id, vars, budget)
    }
}

impl crate::rule::Rule for PolynomialIdentityZeroRule {
    fn name(&self) -> &str {
        "Polynomial Identity"
    }

    fn priority(&self) -> i32 {
        90 // Lower than AutoExpandSubCancelRule (95) to avoid duplicate work
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
        // Skip in Solve mode - preserve structure for equation solving
        if parent_ctx.is_solve_context() {
            return None;
        }

        // Must be Add or Sub to be a cancellation candidate
        let is_sub_or_add = matches!(ctx.get(expr), Expr::Sub(_, _) | Expr::Add(_, _));
        if !is_sub_or_add {
            return None;
        }

        // Quick node count check (avoid expensive conversion for huge expressions)
        let node_count = cas_ast::count_nodes(ctx, expr);
        if node_count > 100 {
            return None; // Too big, skip
        }

        // Quick polynomial-like check
        if !Self::is_polynomial_candidate(ctx, expr) {
            return None;
        }

        // Try to convert to MultiPoly
        let budget = Self::poly_budget();
        let mut vars = Vec::new();
        let poly = Self::expr_to_multipoly(ctx, expr, &mut vars, &budget)?;

        // Check variable count
        if vars.len() > 4 {
            return None; // Too many variables
        }

        // If the result is zero, we have a polynomial identity!
        if poly.is_zero() {
            let zero = ctx.num(0);

            // Split terms into positive and negative to show LHS/RHS normal forms
            // For an expression like A + B - C - D, we show:
            //   LHS (positive): A + B expanded
            //   RHS (negative): C + D expanded
            let (positive_terms, negative_terms) = {
                let mut pos = Vec::new();
                let mut neg = Vec::new();

                // Collect all additive terms
                fn collect_terms(
                    ctx: &Context,
                    e: ExprId,
                    pos: &mut Vec<ExprId>,
                    neg: &mut Vec<ExprId>,
                ) {
                    match ctx.get(e) {
                        Expr::Add(a, b) => {
                            collect_terms(ctx, *a, pos, neg);
                            collect_terms(ctx, *b, pos, neg);
                        }
                        Expr::Sub(a, b) => {
                            collect_terms(ctx, *a, pos, neg);
                            // b is subtracted, so it goes to negative
                            neg.push(*b);
                        }
                        Expr::Neg(inner) => {
                            neg.push(*inner);
                        }
                        _ => {
                            pos.push(e);
                        }
                    }
                }
                collect_terms(ctx, expr, &mut pos, &mut neg);
                (pos, neg)
            };

            // Build proof data with LHS/RHS if we have both positive and negative terms
            let proof_data = if !positive_terms.is_empty() && !negative_terms.is_empty() {
                // Build polys for positive sum (LHS) and negative sum (RHS)
                let mut lhs_poly = crate::multipoly::MultiPoly::zero(vars.clone());
                let mut rhs_poly = crate::multipoly::MultiPoly::zero(vars.clone());

                // Sum positive terms - use the same vars we already collected
                for &term in &positive_terms {
                    let mut _term_vars = vars.clone();
                    if let Some(term_poly) =
                        Self::expr_to_multipoly(ctx, term, &mut _term_vars, &budget)
                    {
                        // If same vars, can add directly
                        if term_poly.vars == lhs_poly.vars {
                            if let Ok(sum) = lhs_poly.add(&term_poly) {
                                lhs_poly = sum;
                            }
                        }
                    }
                }

                // Sum negative terms (these are the RHS that was subtracted)
                for &term in &negative_terms {
                    let mut _term_vars = vars.clone();
                    if let Some(term_poly) =
                        Self::expr_to_multipoly(ctx, term, &mut _term_vars, &budget)
                    {
                        if term_poly.vars == rhs_poly.vars {
                            if let Ok(sum) = rhs_poly.add(&term_poly) {
                                rhs_poly = sum;
                            }
                        }
                    }
                }

                crate::multipoly_display::PolynomialProofData::from_identity(
                    ctx,
                    &lhs_poly,
                    &rhs_poly,
                    vars.clone(),
                )
            } else {
                // No clear LHS/RHS split
                crate::multipoly_display::PolynomialProofData {
                    monomials: 0,
                    degree: 0,
                    vars: vars.clone(),
                    normal_form_expr: Some(zero),
                    lhs_stats: None,
                    rhs_stats: None,
                }
            };

            return Some(
                Rewrite::new(zero)
                    .desc("Polynomial identity: normalize and cancel to 0")
                    .poly_proof(proof_data),
            );
        }

        None
    }
}

// =============================================================================
// HeuristicExtractCommonFactorAddRule: Extract common base factors from Add/Sub
// =============================================================================
//
// V2.15.9: In Heuristic mode, extracts common polynomial base factors from sums:
// - (x+1)^4 + 4*(x+1)^3 → (x+1)³ * (x+5)
// - (x+1)^3 + (x+1)^4 → (x+1)³ * (x+2)
//
// SAFE MODE: Only handles exactly 2 terms with same base Pow(base, int_exp).
// Priority 110 (higher than HeuristicPolyNormalizeAddRule at 100).

/// Parsed representation of a term in an Add/Sub expression
/// Represents: sign * coeff * base^exp
#[derive(Debug, Clone)]
struct ParsedTerm {
    sign: i8,     // +1 or -1
    coeff: i64,   // Integer coefficient (1 if implicit)
    base: ExprId, // The base expression (e.g., x+1)
    exp: u32,     // The exponent (>= 1)
}

/// HeuristicExtractCommonFactorAddRule: Extract common base factors from sums
pub struct HeuristicExtractCommonFactorAddRule;

impl HeuristicExtractCommonFactorAddRule {
    /// Parse a term into (sign, coeff, base, exp) form
    /// Returns None if term doesn't match expected pattern
    fn parse_term(ctx: &Context, term: ExprId, positive: bool) -> Option<ParsedTerm> {
        let sign: i8 = if positive { 1 } else { -1 };

        // Try to match: coeff * Pow(base, exp) or Pow(base, exp) or coeff * base
        match ctx.get(term) {
            // Direct power: base^exp
            Expr::Pow(base, exp_id) => {
                let exp = Self::extract_int_exp(ctx, *exp_id)?;
                if exp >= 1 {
                    Some(ParsedTerm {
                        sign,
                        coeff: 1,
                        base: *base,
                        exp,
                    })
                } else {
                    None
                }
            }
            // Multiplication: coeff * something
            Expr::Mul(l, r) => {
                // Try: coeff * base^exp
                if let Some(c) = Self::extract_int_coeff(ctx, *l) {
                    if let Expr::Pow(base, exp_id) = ctx.get(*r) {
                        let exp = Self::extract_int_exp(ctx, *exp_id)?;
                        if exp >= 1 {
                            return Some(ParsedTerm {
                                sign,
                                coeff: c,
                                base: *base,
                                exp,
                            });
                        }
                    }
                    // Try: coeff * base (implicit exp=1)
                    // Only if base is Add (polynomial-like)
                    if matches!(ctx.get(*r), Expr::Add(_, _)) {
                        return Some(ParsedTerm {
                            sign,
                            coeff: c,
                            base: *r,
                            exp: 1,
                        });
                    }
                }
                // Try: base^exp * coeff (reversed order)
                if let Some(c) = Self::extract_int_coeff(ctx, *r) {
                    if let Expr::Pow(base, exp_id) = ctx.get(*l) {
                        let exp = Self::extract_int_exp(ctx, *exp_id)?;
                        if exp >= 1 {
                            return Some(ParsedTerm {
                                sign,
                                coeff: c,
                                base: *base,
                                exp,
                            });
                        }
                    }
                    // Try: base * coeff (implicit exp=1)
                    if matches!(ctx.get(*l), Expr::Add(_, _)) {
                        return Some(ParsedTerm {
                            sign,
                            coeff: c,
                            base: *l,
                            exp: 1,
                        });
                    }
                }
                None
            }
            // Negation: -something
            Expr::Neg(inner) => Self::parse_term(ctx, *inner, !positive),
            _ => None,
        }
    }

    /// Extract integer exponent from an expression
    fn extract_int_exp(ctx: &Context, exp_id: ExprId) -> Option<u32> {
        if let Expr::Number(n) = ctx.get(exp_id) {
            if n.is_integer() && !n.is_negative() {
                use num_traits::ToPrimitive;
                return n.to_integer().to_u32();
            }
        }
        None
    }

    /// Extract integer coefficient from an expression
    fn extract_int_coeff(ctx: &Context, expr: ExprId) -> Option<i64> {
        if let Expr::Number(n) = ctx.get(expr) {
            if n.is_integer() {
                use num_traits::ToPrimitive;
                return n.to_integer().to_i64();
            }
        }
        None
    }

    /// Check structural equality of two expression bases
    fn bases_equal(ctx: &Context, a: ExprId, b: ExprId) -> bool {
        if a == b {
            return true;
        }
        Self::exprs_equal_recursive(ctx, a, b)
    }

    /// Recursive structural equality check for expressions
    fn exprs_equal_recursive(ctx: &Context, a: ExprId, b: ExprId) -> bool {
        if a == b {
            return true;
        }
        match (ctx.get(a), ctx.get(b)) {
            (Expr::Number(n1), Expr::Number(n2)) => n1 == n2,
            (Expr::Variable(v1), Expr::Variable(v2)) => v1 == v2,
            (Expr::Constant(c1), Expr::Constant(c2)) => c1 == c2,
            (Expr::Add(l1, r1), Expr::Add(l2, r2)) => {
                Self::exprs_equal_recursive(ctx, *l1, *l2)
                    && Self::exprs_equal_recursive(ctx, *r1, *r2)
            }
            (Expr::Sub(l1, r1), Expr::Sub(l2, r2)) => {
                Self::exprs_equal_recursive(ctx, *l1, *l2)
                    && Self::exprs_equal_recursive(ctx, *r1, *r2)
            }
            (Expr::Mul(l1, r1), Expr::Mul(l2, r2)) => {
                Self::exprs_equal_recursive(ctx, *l1, *l2)
                    && Self::exprs_equal_recursive(ctx, *r1, *r2)
            }
            (Expr::Div(l1, r1), Expr::Div(l2, r2)) => {
                Self::exprs_equal_recursive(ctx, *l1, *l2)
                    && Self::exprs_equal_recursive(ctx, *r1, *r2)
            }
            (Expr::Pow(l1, r1), Expr::Pow(l2, r2)) => {
                Self::exprs_equal_recursive(ctx, *l1, *l2)
                    && Self::exprs_equal_recursive(ctx, *r1, *r2)
            }
            (Expr::Neg(e1), Expr::Neg(e2)) => Self::exprs_equal_recursive(ctx, *e1, *e2),
            (Expr::Function(n1, args1), Expr::Function(n2, args2)) => {
                n1 == n2
                    && args1.len() == args2.len()
                    && args1
                        .iter()
                        .zip(args2.iter())
                        .all(|(a1, a2)| Self::exprs_equal_recursive(ctx, *a1, *a2))
            }
            _ => false,
        }
    }

    /// Build the quotient term: term / base^g_exp
    /// term = sign * coeff * base^exp
    /// result = sign * coeff * base^(exp - g_exp)
    fn build_quotient_term(ctx: &mut Context, term: &ParsedTerm, g_exp: u32) -> ExprId {
        let remaining_exp = term.exp.saturating_sub(g_exp);

        // Build: coeff * base^remaining_exp (or just coeff or just base^rem)
        let coeff_part = if term.coeff == 1 {
            None
        } else {
            Some(ctx.num(term.coeff.abs()))
        };

        let power_part = if remaining_exp == 0 {
            None
        } else if remaining_exp == 1 {
            Some(term.base)
        } else {
            let exp_id = ctx.num(remaining_exp as i64);
            Some(ctx.add(Expr::Pow(term.base, exp_id)))
        };

        // Combine coeff and power parts
        let unsigned_result = match (coeff_part, power_part) {
            (None, None) => ctx.num(1),
            (Some(c), None) => c,
            (None, Some(p)) => p,
            (Some(c), Some(p)) => ctx.add(Expr::Mul(c, p)),
        };

        // Apply sign
        if term.sign < 0 || term.coeff < 0 {
            // XOR of signs: if exactly one is negative, result is negative
            let total_negative = (term.sign < 0) ^ (term.coeff < 0);
            if total_negative {
                ctx.add(Expr::Neg(unsigned_result))
            } else {
                unsigned_result
            }
        } else {
            unsigned_result
        }
    }

    /// Simplify an Add expression by combining numeric constants
    /// e.g., (x + 1) + 4 → x + 5, or 1 + 4 → 5
    fn simplify_add_constants(ctx: &mut Context, expr: ExprId) -> ExprId {
        // Collect all additive terms and sum numeric ones
        let mut numeric_sum: i64 = 0;
        let mut non_numeric: Vec<ExprId> = Vec::new();

        Self::collect_add_terms_for_const_fold(ctx, expr, true, &mut numeric_sum, &mut non_numeric);

        // Rebuild expression
        if non_numeric.is_empty() {
            // All numeric
            ctx.num(numeric_sum)
        } else {
            // Start with first non-numeric term
            let mut result = non_numeric[0];
            for term in &non_numeric[1..] {
                result = ctx.add(Expr::Add(result, *term));
            }
            // Add numeric sum if non-zero
            if numeric_sum != 0 {
                let num_expr = ctx.num(numeric_sum);
                result = ctx.add(Expr::Add(result, num_expr));
            }
            result
        }
    }

    fn collect_add_terms_for_const_fold(
        ctx: &Context,
        expr: ExprId,
        positive: bool,
        numeric_sum: &mut i64,
        non_numeric: &mut Vec<ExprId>,
    ) {
        match ctx.get(expr) {
            Expr::Add(l, r) => {
                Self::collect_add_terms_for_const_fold(ctx, *l, positive, numeric_sum, non_numeric);
                Self::collect_add_terms_for_const_fold(ctx, *r, positive, numeric_sum, non_numeric);
            }
            Expr::Sub(l, r) => {
                Self::collect_add_terms_for_const_fold(ctx, *l, positive, numeric_sum, non_numeric);
                Self::collect_add_terms_for_const_fold(
                    ctx,
                    *r,
                    !positive,
                    numeric_sum,
                    non_numeric,
                );
            }
            Expr::Neg(inner) => {
                Self::collect_add_terms_for_const_fold(
                    ctx,
                    *inner,
                    !positive,
                    numeric_sum,
                    non_numeric,
                );
            }
            Expr::Number(n) => {
                if n.is_integer() {
                    use num_traits::ToPrimitive;
                    if let Some(v) = n.to_integer().to_i64() {
                        if positive {
                            *numeric_sum += v;
                        } else {
                            *numeric_sum -= v;
                        }
                        return;
                    }
                }
                // Non-integer or overflow: treat as non-numeric
                if positive {
                    non_numeric.push(expr);
                } else {
                    // Would need to wrap in Neg but we're avoiding that
                    non_numeric.push(expr);
                }
            }
            _ => {
                non_numeric.push(expr);
            }
        }
    }
}

impl crate::rule::Rule for HeuristicExtractCommonFactorAddRule {
    fn name(&self) -> &str {
        "Heuristic Extract Common Factor"
    }

    fn priority(&self) -> i32 {
        110 // Higher than HeuristicPolyNormalizeAddRule (100) to try factorization first
    }

    fn allowed_phases(&self) -> PhaseMask {
        PhaseMask::TRANSFORM
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Add", "Sub"])
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Only trigger when heuristic_poly is On
        use crate::options::HeuristicPoly;
        if parent_ctx.heuristic_poly() != HeuristicPoly::On {
            return None;
        }

        // Skip in Solve mode
        if parent_ctx.is_solve_context() {
            return None;
        }

        // === SAFE MODE: Only handle Add(A, B) or Sub(A, B) with exactly 2 terms ===
        let (term1_id, term2_id, term2_positive) = match ctx.get(expr).clone() {
            Expr::Add(l, r) => (l, r, true),
            Expr::Sub(l, r) => (l, r, false),
            _ => return None,
        };

        // Parse both terms
        let term1 = Self::parse_term(ctx, term1_id, true)?;
        let term2 = Self::parse_term(ctx, term2_id, term2_positive)?;

        // Both bases must be equal (structural comparison)
        if !Self::bases_equal(ctx, term1.base, term2.base) {
            return None;
        }

        // Base must be compound (Add) to be interesting for polynomial factorization
        if !matches!(ctx.get(term1.base), Expr::Add(_, _)) {
            return None;
        }

        // GCD exponent = min(exp1, exp2)
        let g_exp = term1.exp.min(term2.exp);
        if g_exp == 0 {
            return None;
        }

        // Build quotient terms
        let q1 = Self::build_quotient_term(ctx, &term1, g_exp);
        let q2 = Self::build_quotient_term(ctx, &term2, g_exp);

        // Build inner sum: q1 + q2
        let inner_sum_raw = ctx.add(Expr::Add(q1, q2));

        // Simplify constants in inner_sum (e.g., x + 1 + 4 → x + 5)
        let inner_sum = Self::simplify_add_constants(ctx, inner_sum_raw);

        // Build factor: base^g_exp
        let factor = if g_exp == 1 {
            term1.base
        } else {
            let exp_id = ctx.num(g_exp as i64);
            ctx.add(Expr::Pow(term1.base, exp_id))
        };

        // Build result: factor * inner_sum
        // Wrap in __hold to prevent DistributeRule from expanding it back
        let product = ctx.add(Expr::Mul(factor, inner_sum));
        let new_expr = ctx.add(Expr::Function("__hold".to_string(), vec![product]));

        // Complexity check: result should be simpler
        let old_nodes = cas_ast::count_nodes(ctx, expr);
        let new_nodes = cas_ast::count_nodes(ctx, new_expr);
        if new_nodes > old_nodes + 5 {
            return None; // Don't make things worse
        }

        Some(
            Rewrite::new(new_expr)
                .desc("Extract common polynomial factor")
                .local(expr, new_expr),
        )
    }
}

// =============================================================================
// HeuristicPolyNormalizeAddRule: Poly-normalize Add/Sub with binomial powers
// =============================================================================
//
// V2.15.8: In Heuristic mode, normalizes Add/Sub expressions containing Pow(Add, n)
// to polynomial form using MultiPoly arithmetic, producing flattened results with
// combined like terms.
//
// Example: (x+1)^3 + x^3 → 2x³ + 3x² + 3x + 1 (not x³ + ... + x³)

/// HeuristicPolyNormalizeAddRule: Poly-normalize sums with binomial powers
/// Priority 42 (after ExpandSmallBinomialPowRule at 40, before others)
pub struct HeuristicPolyNormalizeAddRule;

impl HeuristicPolyNormalizeAddRule {
    /// Check if expression contains Pow(Add, n) with 2 ≤ n ≤ 6
    fn contains_pow_add(ctx: &Context, expr: ExprId) -> bool {
        Self::contains_pow_add_inner(ctx, expr, 0)
    }

    fn contains_pow_add_inner(ctx: &Context, expr: ExprId, depth: usize) -> bool {
        if depth > 20 {
            return false;
        }
        match ctx.get(expr) {
            Expr::Pow(base, exp) => {
                // Check if this is Pow(Add, n) with 2 ≤ n ≤ 6 AND base is polynomial-like
                if matches!(ctx.get(*base), Expr::Add(_, _)) {
                    if let Expr::Number(n) = ctx.get(*exp) {
                        if n.is_integer() && !n.is_negative() {
                            use num_traits::ToPrimitive;
                            if let Some(e) = n.to_integer().to_u32() {
                                // Must be polynomial-like base (no functions like sqrt, sin)
                                if (2..=6).contains(&e)
                                    && crate::auto_expand_scan::looks_polynomial_like(ctx, *base)
                                {
                                    return true;
                                }
                            }
                        }
                    }
                }
                Self::contains_pow_add_inner(ctx, *base, depth + 1)
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
                Self::contains_pow_add_inner(ctx, *l, depth + 1)
                    || Self::contains_pow_add_inner(ctx, *r, depth + 1)
            }
            Expr::Neg(inner) => Self::contains_pow_add_inner(ctx, *inner, depth + 1),
            _ => false,
        }
    }
}

impl crate::rule::Rule for HeuristicPolyNormalizeAddRule {
    fn name(&self) -> &str {
        "Heuristic Poly Normalize"
    }

    fn priority(&self) -> i32 {
        100 // Very high priority - must process Add BEFORE children Pow are expanded
    }

    fn allowed_phases(&self) -> PhaseMask {
        PhaseMask::TRANSFORM
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Add", "Sub"])
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Only trigger when heuristic_poly is On
        use crate::options::HeuristicPoly;
        if parent_ctx.heuristic_poly() != HeuristicPoly::On {
            return None;
        }

        // Skip in Solve mode
        if parent_ctx.is_solve_context() {
            return None;
        }

        // Must be Add or Sub
        if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
            return None;
        }

        // Must contain at least one Pow(Add, n) with 2 ≤ n ≤ 6
        // We process the ORIGINAL Add before children are expanded by ExpandSmallBinomialPowRule
        if !Self::contains_pow_add(ctx, expr) {
            return None;
        }

        // Quick size check
        let node_count = cas_ast::count_nodes(ctx, expr);
        if node_count > 80 {
            return None;
        }

        // Try to convert to MultiPoly (this expands and combines terms)
        let budget = PolyBudget {
            max_terms: 40,
            max_total_degree: 6,
            max_pow_exp: 6,
        };

        let mut vars = Vec::new();
        let poly = AutoExpandSubCancelRule::expr_to_multipoly(ctx, expr, &mut vars, &budget)?;

        // Check if result is reasonable
        if poly.terms.len() > 30 || vars.len() > 3 {
            return None;
        }

        // If polynomial is zero, let PolynomialIdentityZeroRule handle it
        if poly.is_zero() {
            return None;
        }

        // Convert back to expression using multipoly_to_expr (produces flattened Add)
        let new_expr = crate::multipoly::multipoly_to_expr(&poly, ctx);

        // Don't rewrite to same expression
        if new_expr == expr {
            return None;
        }

        Some(
            Rewrite::new(new_expr)
                .desc("Expand and combine polynomial terms (heuristic)")
                .local(expr, new_expr),
        )
    }
}

//
// V2.15.8: Expands (a+b)^n automatically when:
// - n is a small positive integer (2 ≤ n ≤ 6)
// - Base is a simple polynomial (≤3 linear terms: constants, variables, c*x)
// - Estimated output terms ≤ 20
//
// This enables `simplify((x+1)^5 - polynomial)` to work without explicit expand().
// The result is wrapped in __hold() to prevent factorization rules from undoing it.

use crate::multinomial_expand::{try_expand_multinomial_direct, MultinomialExpandBudget};

/// ExpandSmallBinomialPowRule: Always-on expansion for small binomial/trinomial powers
/// Priority 40 (before AutoExpandPowSumRule at 50, after most algebraic rules)
pub struct ExpandSmallBinomialPowRule;

impl crate::rule::Rule for ExpandSmallBinomialPowRule {
    fn name(&self) -> &str {
        "Expand Small Power"
    }

    fn priority(&self) -> i32 {
        40 // Before AutoExpandPowSumRule (50), after basic algebra
    }

    fn allowed_phases(&self) -> PhaseMask {
        // Only in TRANSFORM phase to avoid interfering with early simplification
        PhaseMask::TRANSFORM
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Pow"])
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // V2.15.9: Check autoexpand_binomials mode (Off/On)
        use crate::options::AutoExpandBinomials;
        match parent_ctx.autoexpand_binomials() {
            AutoExpandBinomials::Off => return None, // Never expand standalone
            AutoExpandBinomials::On => {
                // Always expand (subject to budget checks below)
            }
        }

        // Skip in Solve mode - preserve structure for equation solving
        if parent_ctx.is_solve_context() {
            return None;
        }

        // Skip if already in auto-expand context (let AutoExpandPowSumRule handle)
        if parent_ctx.in_auto_expand_context()
            && parent_ctx.autoexpand_binomials() != AutoExpandBinomials::On
        {
            return None;
        }

        // Pattern: Pow(base, exp)
        let (base, exp) = crate::helpers::as_pow(ctx, expr)?;

        // Very restrictive budget for automatic expansion in generic mode
        // - max_exp: 6 (binomial (x+1)^6 = 7 terms, trinomial (a+b+c)^4 = 15 terms)
        // - max_base_terms: 3 (binomial or trinomial only)
        // - max_vars: 2 (keeps output manageable)
        // - max_output_terms: 20 (strict limit to prevent bloat)
        let budget = MultinomialExpandBudget {
            max_exp: 6,
            max_base_terms: 3,
            max_vars: 2,
            max_output_terms: 20,
        };

        // try_expand_multinomial_direct already:
        // 1. Checks exponent is small positive integer
        // 2. Extracts linear terms (fails if base has functions/div)
        // 3. Estimates output terms and checks budget
        // 4. Wraps result in __hold() for anti-cycle protection
        let expanded = try_expand_multinomial_direct(ctx, base, exp, &budget)?;

        Some(
            Rewrite::new(expanded)
                .desc("Expand binomial/trinomial power")
                .local(expr, expanded),
        )
    }
}

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
