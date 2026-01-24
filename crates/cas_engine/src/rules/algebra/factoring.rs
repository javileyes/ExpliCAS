use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::count_nodes;
use cas_ast::Expr;
use std::cmp::Ordering;

/// Check if two expressions form a conjugate pair: (A+B) and (A-B) or vice versa
/// Returns Some((a, b)) if they are conjugates, None otherwise
fn is_conjugate_pair(
    ctx: &cas_ast::Context,
    l: cas_ast::ExprId,
    r: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    use crate::ordering::compare_expr;

    let l_expr = ctx.get(l);
    let r_expr = ctx.get(r);

    match (l_expr, r_expr) {
        (Expr::Add(a1, a2), Expr::Sub(b1, b2)) | (Expr::Sub(b1, b2), Expr::Add(a1, a2)) => {
            // Copy the ExprIds (they're Copy types from pattern match)
            let (a1, a2, b1, b2) = (*a1, *a2, *b1, *b2);

            // Direct match: (A+B) vs (A-B)
            if compare_expr(ctx, a1, b1) == Ordering::Equal
                && compare_expr(ctx, a2, b2) == Ordering::Equal
            {
                return Some((a1, a2));
            }
            // Commutative: (B+A) vs (A-B) → A=b1, B=a1
            if compare_expr(ctx, a2, b1) == Ordering::Equal
                && compare_expr(ctx, a1, b2) == Ordering::Equal
            {
                return Some((b1, b2));
            }
            None
        }
        // Handle canonicalized form: Sub(a, b) becomes Add(-b, a) or Add(a, -b)
        (Expr::Add(a1, a2), Expr::Add(b1, b2)) => {
            let (a1, a2, b1, b2) = (*a1, *a2, *b1, *b2);

            // Case 1: (A+B) vs (A+(-B)) where b2 = -a2
            if is_negation(ctx, a2, b2) && compare_expr(ctx, a1, b1) == Ordering::Equal {
                return Some((a1, a2));
            }
            // Case 2: (A+B) vs ((-B)+A) where b1 = -a2
            if is_negation(ctx, a2, b1) && compare_expr(ctx, a1, b2) == Ordering::Equal {
                return Some((a1, a2));
            }
            // Case 3: (A+B) vs (B+(-A)) where b2 = -a1
            if is_negation(ctx, a1, b2) && compare_expr(ctx, a2, b1) == Ordering::Equal {
                return Some((a2, a1)); // b^2 - a^2
            }
            // Case 4: (A+B) vs ((-A)+B) where b1 = -a1
            if is_negation(ctx, a1, b1) && compare_expr(ctx, a2, b2) == Ordering::Equal {
                return Some((a2, a1));
            }
            None
        }
        _ => None,
    }
}

/// Check if `b` is the negation of `a` (Neg(a) or Mul(-1, a) or Number(-n) vs Number(n))
fn is_negation(ctx: &cas_ast::Context, a: cas_ast::ExprId, b: cas_ast::ExprId) -> bool {
    // Check numeric negation: Number(n) vs Number(-n)
    if let (Expr::Number(n_a), Expr::Number(n_b)) = (ctx.get(a), ctx.get(b)) {
        if n_a == &(-n_b.clone()) {
            return true;
        }
    }

    match ctx.get(b) {
        Expr::Neg(inner) if *inner == a => true,
        Expr::Mul(l, r) => {
            // Check for -1 * a or a * -1
            let l_id = *l;
            let r_id = *r;
            (is_minus_one(ctx, l_id) && r_id == a) || (is_minus_one(ctx, r_id) && l_id == a)
        }
        _ => false,
    }
}

/// Check if expression is -1
fn is_minus_one(ctx: &cas_ast::Context, e: cas_ast::ExprId) -> bool {
    use num_bigint::BigInt;
    use num_rational::BigRational;
    if let Expr::Number(n) = ctx.get(e) {
        n == &BigRational::from_integer(BigInt::from(-1))
    } else if let Expr::Neg(inner) = ctx.get(e) {
        if let Expr::Number(n) = ctx.get(*inner) {
            n == &BigRational::from_integer(BigInt::from(1))
        } else {
            false
        }
    } else {
        false
    }
}

/// Check if two N-ary sums form a conjugate pair: (U+V) and (U-V)
/// where U can be any sum of terms and V is the single differing term.
///
/// Returns Some((U, V)) if they are conjugates, None otherwise.
/// U is returned as an ExprId representing the common sum.
/// V is the term that differs by sign between the two expressions.
///
/// Algorithm:
/// 1. Flatten both expressions to signed term lists
/// 2. Normalize each term by extracting embedded signs (Neg, Mul(-1,...), Number(-n))
/// 3. Build multisets keyed by structural equality of normalized core
/// 4. Valid conjugate iff exactly one term has diff = +2 or -2 (sign flip)
fn is_nary_conjugate_pair(
    ctx: &mut cas_ast::Context,
    l: cas_ast::ExprId,
    r: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    use crate::nary::{add_terms_signed, build_balanced_add, Sign};
    use crate::ordering::compare_expr;
    use num_traits::Signed;

    /// Extract the "unsigned core" of a term and its effective sign.
    ///
    /// This function handles:
    /// - Neg(x) → (x, -1)
    /// - Number(-n) → (Number(n), -1)
    /// - Products with negative coefficients: flattens, extracts sign, sorts, rebuilds
    ///
    /// For products, we flatten to factors, extract any negative sign from numeric
    /// coefficients, sort the factors canonically, and rebuild. This ensures that
    /// `2*a*b` compares equal regardless of tree structure (left vs right associative).
    fn normalize_term(ctx: &mut cas_ast::Context, term: cas_ast::ExprId) -> (cas_ast::ExprId, i32) {
        use crate::helpers::flatten_mul;

        match ctx.get(term).clone() {
            Expr::Neg(inner) => {
                // Recursively normalize the inner term
                let (core, sign) = normalize_term(ctx, inner);
                (core, -sign)
            }
            Expr::Number(n) => {
                if n.is_negative() {
                    let pos_n = -n.clone();
                    let pos_term = ctx.add(Expr::Number(pos_n));
                    (pos_term, -1)
                } else {
                    (term, 1)
                }
            }
            Expr::Mul(_, _) => {
                // Flatten the product to get all factors
                let mut factors: Vec<cas_ast::ExprId> = Vec::new();
                flatten_mul(ctx, term, &mut factors);

                // Extract sign from any negative numeric coefficient
                let mut overall_sign: i32 = 1;
                let mut unsigned_factors: Vec<cas_ast::ExprId> = Vec::new();

                for factor in factors {
                    match ctx.get(factor).clone() {
                        Expr::Neg(inner) => {
                            overall_sign *= -1;
                            // Check if inner is also negative number
                            if let Expr::Number(n) = ctx.get(inner).clone() {
                                if n.is_negative() {
                                    // Neg(Number(-x)) = x
                                    unsigned_factors.push(ctx.add(Expr::Number(-n)));
                                } else {
                                    unsigned_factors.push(inner);
                                }
                            } else {
                                unsigned_factors.push(inner);
                            }
                        }
                        Expr::Number(n) if n.is_negative() => {
                            overall_sign *= -1;
                            unsigned_factors.push(ctx.add(Expr::Number(-n)));
                        }
                        _ => {
                            unsigned_factors.push(factor);
                        }
                    }
                }

                // Sort factors canonically using compare_expr
                unsigned_factors.sort_by(|a, b| compare_expr(ctx, *a, *b));

                // Rebuild the product (right-associatively for consistency)
                let canonical_core = if unsigned_factors.is_empty() {
                    ctx.add(Expr::Number(num_rational::BigRational::from_integer(
                        num_bigint::BigInt::from(1),
                    )))
                } else if unsigned_factors.len() == 1 {
                    unsigned_factors[0]
                } else {
                    // Build right-associative: a * (b * (c * ...))
                    let mut result = match unsigned_factors.last() {
                        Some(r) => *r,
                        None => ctx.num(1),
                    };
                    for factor in unsigned_factors.iter().rev().skip(1) {
                        result = ctx.add(Expr::Mul(*factor, result));
                    }
                    result
                };

                (canonical_core, overall_sign)
            }
            _ => (term, 1),
        }
    }

    // Flatten both sides
    let l_terms = add_terms_signed(ctx, l);
    let r_terms = add_terms_signed(ctx, r);

    // Must have same number of terms
    if l_terms.len() != r_terms.len() || l_terms.is_empty() {
        return None;
    }

    // Budget guard: don't process huge expressions
    const MAX_TERMS: usize = 16;
    if l_terms.len() > MAX_TERMS {
        return None;
    }

    // Normalize all terms and compute effective signs
    // Each entry: (normalized_core, effective_sign)
    let l_normalized: Vec<(cas_ast::ExprId, i32)> = l_terms
        .iter()
        .map(|&(term, sign)| {
            let sign_val = match sign {
                Sign::Pos => 1,
                Sign::Neg => -1,
            };
            let (core, term_sign) = normalize_term(ctx, term);
            (core, sign_val * term_sign)
        })
        .collect();

    let r_normalized: Vec<(cas_ast::ExprId, i32)> = r_terms
        .iter()
        .map(|&(term, sign)| {
            let sign_val = match sign {
                Sign::Pos => 1,
                Sign::Neg => -1,
            };
            let (core, term_sign) = normalize_term(ctx, term);
            (core, sign_val * term_sign)
        })
        .collect();

    // Build multiset: group by normalized core
    // Each group: (core, net_sign_in_L, net_sign_in_R)
    let mut groups: Vec<(cas_ast::ExprId, i32, i32)> = Vec::new();

    // Process L terms
    for &(core, sign) in &l_normalized {
        let mut found = false;
        for (rep, l_count, _) in groups.iter_mut() {
            if compare_expr(ctx, *rep, core) == Ordering::Equal {
                *l_count += sign;
                found = true;
                break;
            }
        }
        if !found {
            groups.push((core, sign, 0));
        }
    }

    // Process R terms
    for &(core, sign) in &r_normalized {
        let mut found = false;
        for (rep, _, r_count) in groups.iter_mut() {
            if compare_expr(ctx, *rep, core) == Ordering::Equal {
                *r_count += sign;
                found = true;
                break;
            }
        }
        if !found {
            groups.push((core, 0, sign));
        }
    }

    // Compute diffs and identify the differing term
    // Conjugate pair: exactly one term with diff=+2 or diff=-2 (sign flip)
    // All other terms must have diff=0
    let mut v_term: Option<cas_ast::ExprId> = None;
    let mut common_terms: Vec<(cas_ast::ExprId, i32)> = Vec::new();

    for (core, l_count, r_count) in &groups {
        let diff = l_count - r_count;

        if diff == 0 {
            // Common term - add to U with the sign from L
            if *l_count != 0 {
                common_terms.push((*core, *l_count));
            }
        } else if diff == 2 || diff == -2 {
            // This term has opposite signs in L and R
            // In L: +v, in R: -v → diff = 1 - (-1) = 2
            // In L: -v, in R: +v → diff = -1 - 1 = -2
            if v_term.is_some() {
                // More than one differing term - not a simple conjugate
                return None;
            }
            v_term = Some(*core);
        } else {
            // Invalid diff (not 0 or ±2) - not a conjugate pair
            return None;
        }
    }

    // Must have exactly one differing term
    let v = v_term?;

    // Need at least one common term for a meaningful conjugate
    if common_terms.is_empty() {
        return None;
    }

    // Build U as the sum of common terms with their signs
    let u_terms: Vec<cas_ast::ExprId> = common_terms
        .iter()
        .map(|(term, count)| {
            if *count > 0 {
                *term
            } else {
                ctx.add(Expr::Neg(*term))
            }
        })
        .collect();

    let u = if u_terms.len() == 1 {
        u_terms[0]
    } else {
        build_balanced_add(ctx, &u_terms)
    };

    Some((u, v))
}

// DifferenceOfSquaresRule: Expands conjugate products
// (a - b) * (a + b) → a² - b²
// Now supports N-ary sums: (U + V)(U - V) → U² - V²
// Phase: CORE | POST (structural simplification, not expansion)
define_rule!(
    DifferenceOfSquaresRule,
    "Difference of Squares (Product to Difference)",
    None,
    PhaseMask::CORE | PhaseMask::POST,
    |ctx, expr| {
        // Match Mul(l, r) where l and r are conjugate binomials
        if let Expr::Mul(l, r) = ctx.get(expr) {
            let l = *l;
            let r = *r;

            // Try fast binary matcher first
            if let Some((a, b)) = is_conjugate_pair(ctx, l, r) {
                // Create a² - b²
                let two = ctx.num(2);
                let a_squared = ctx.add(Expr::Pow(a, two));
                let b_squared = ctx.add(Expr::Pow(b, two));
                let new_expr = ctx.add(Expr::Sub(a_squared, b_squared));

                return Some(Rewrite::new(new_expr).desc("(a-b)(a+b) = a² - b²"));
            }

            // Try N-ary matcher for sums with 3+ terms
            if let Some((u, v)) = is_nary_conjugate_pair(ctx, l, r) {
                // Create U² - V²
                let two = ctx.num(2);
                let u_squared = ctx.add(Expr::Pow(u, two));
                let v_squared = ctx.add(Expr::Pow(v, two));
                let new_expr = ctx.add(Expr::Sub(u_squared, v_squared));

                return Some(
                    Rewrite::new(new_expr).desc("(U+V)(U-V) = U² - V² (conjugate product)"),
                );
            }
        }
        None
    }
);

define_rule!(
    FactorRule,
    "Factor Polynomial",
    Some(vec!["Function"]), // Target Function expressions specifically
    PhaseMask::CORE | PhaseMask::POST,
    |ctx, expr| {
        if let Expr::Function(fn_id, args) = ctx.get(expr) { let name = ctx.sym_name(*fn_id);
            if name == "factor" && args.len() == 1 {
                let arg = args[0];
                // Use the general factor entry point which tries polynomial then diff squares
                let new_expr = crate::factor::factor(ctx, arg);
                if new_expr != arg {
                    // Wrap in __hold() to prevent other rules from undoing the factorization
                    // (e.g., DifferenceOfSquaresRule converts (a-b)(a+b) back to a²-b²)
                    let held = ctx.call("__hold", vec![new_expr]);
                    return Some(Rewrite::new(held).desc("Factorization"));
                }
            }
        }
        None
    }
);

define_rule!(
    FactorDifferenceSquaresRule,
    "Factor Difference of Squares",
    |ctx, expr| {
        // match Expr::Add(l, r)
        let expr_data = ctx.get(expr).clone();
        if let Expr::Add(_, _) | Expr::Sub(_, _) = expr_data {
            // Check
        } else {
            return None;
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
                if i == j {
                    continue;
                }

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
                        for (k, &term) in terms.iter().enumerate() {
                            if k != i && k != j {
                                new_terms.push(term);
                            }
                        }

                        // Rebuild Add chain
                        if new_terms.is_empty() {
                            return Some(
                                Rewrite::new(ctx.num(0))
                                    .desc("Factor difference of squares (Empty)"),
                            );
                        }

                        let mut new_expr = new_terms[0];
                        for t in new_terms.iter().skip(1) {
                            new_expr = ctx.add(Expr::Add(new_expr, *t));
                        }

                        return Some(
                            Rewrite::new(new_expr).desc("Factor difference of squares (N-ary)"),
                        );
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
            Expr::Add(_, _) | Expr::Sub(_, _) => {}
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
                    if crate::ordering::compare_expr(ctx, new_expr, expr)
                        == std::cmp::Ordering::Equal
                    {
                        return None;
                    }
                    return Some(
                        Rewrite::new(new_expr).desc("Automatic Factorization (Reduced Size)"),
                    );
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
                    return Some(
                        Rewrite::new(new_expr).desc("Automatic Factorization (Diff Squares)"),
                    );
                }
            }
        }

        None
    }
);

// FactorCommonIntegerFromAdd: Factor out common integer GCD from sum terms
// Example: 2*√2 - 2 → 2*(√2 - 1)
// Phase: POST (runs after rationalization to clean up results)
define_rule!(
    FactorCommonIntegerFromAdd,
    "Factor Common Integer",
    None,
    PhaseMask::POST,
    |ctx, expr| {
        use crate::rules::algebra::helpers::gcd_rational;
        use num_rational::BigRational;
        use num_traits::{One, Signed};

        // Only match simple binary Add(a, b)
        let (l, r) = match ctx.get(expr) {
            Expr::Add(l, r) => (*l, *r),
            _ => return None,
        };

        // Extract integer coefficient from a term
        fn get_int_coef(ctx: &cas_ast::Context, term: cas_ast::ExprId) -> Option<BigRational> {
            match ctx.get(term) {
                Expr::Number(n) if n.is_integer() => Some(n.clone()),
                Expr::Mul(a, b) => {
                    if let Expr::Number(n) = ctx.get(*a) {
                        if n.is_integer() {
                            return Some(n.clone());
                        }
                    }
                    if let Expr::Number(n) = ctx.get(*b) {
                        if n.is_integer() {
                            return Some(n.clone());
                        }
                    }
                    None
                }
                Expr::Neg(inner) => get_int_coef(ctx, *inner).map(|c| -c),
                _ => None,
            }
        }

        // Get coefficients
        let coef_l = get_int_coef(ctx, l);
        let coef_r = get_int_coef(ctx, r);

        let coef_l = coef_l?;
        let coef_r = coef_r?;

        // CRITICAL: Skip if either term contains a Variable
        // We only want to factor pure numeric expressions like 2*√2 - 2, NOT algebraic like 2*x - 6
        fn contains_variable(ctx: &cas_ast::Context, e: cas_ast::ExprId) -> bool {
            match ctx.get(e) {
                Expr::Variable(_) => true,
                Expr::Add(l, r)
                | Expr::Sub(l, r)
                | Expr::Mul(l, r)
                | Expr::Div(l, r)
                | Expr::Pow(l, r) => contains_variable(ctx, *l) || contains_variable(ctx, *r),
                Expr::Neg(inner) => contains_variable(ctx, *inner),
                Expr::Function(_, args) => args.iter().any(|a| contains_variable(ctx, *a)),
                _ => false,
            }
        }

        if contains_variable(ctx, l) || contains_variable(ctx, r) {
            return None;
        }

        // Compute GCD of absolute values
        let gcd = gcd_rational(coef_l.abs(), coef_r.abs());
        if gcd <= BigRational::one() {
            return None;
        }

        // Check GCD is at least 2
        let gcd_int = gcd.to_integer();
        if gcd_int <= num_bigint::BigInt::from(1) {
            return None;
        }

        // Divide coefficients by GCD
        fn divide_term(
            ctx: &mut cas_ast::Context,
            term: cas_ast::ExprId,
            gcd: &BigRational,
        ) -> cas_ast::ExprId {
            match ctx.get(term).clone() {
                Expr::Number(n) => {
                    let new_n = &n / gcd;
                    ctx.add(Expr::Number(new_n))
                }
                Expr::Mul(a, b) => {
                    if let Expr::Number(n) = ctx.get(a).clone() {
                        let new_n = &n / gcd;
                        if new_n.is_one() {
                            return b;
                        }
                        let num = ctx.add(Expr::Number(new_n));
                        return ctx.add_raw(Expr::Mul(num, b));
                    }
                    if let Expr::Number(n) = ctx.get(b).clone() {
                        let new_n = &n / gcd;
                        if new_n.is_one() {
                            return a;
                        }
                        let num = ctx.add(Expr::Number(new_n));
                        return ctx.add_raw(Expr::Mul(a, num));
                    }
                    term
                }
                Expr::Neg(inner) => {
                    let divided = divide_term(ctx, inner, gcd);
                    ctx.add(Expr::Neg(divided))
                }
                _ => term,
            }
        }

        let new_l = divide_term(ctx, l, &gcd);
        let new_r = divide_term(ctx, r, &gcd);
        let inner = ctx.add_raw(Expr::Add(new_l, new_r));
        let gcd_expr = ctx.add(Expr::Number(gcd.clone()));
        let new_expr = ctx.add_raw(Expr::Mul(gcd_expr, inner));

        // Note: We intentionally allow node count to increase for GCD factoring
        // because 2*(3 + 2*√5) is mathematically cleaner than 6 + 4*√5

        Some(
            Rewrite::new(new_expr)
                .desc(format!("Factor out {}", gcd_int))
                .local(expr, new_expr),
        )
    }
);

// SumThreeCubesZeroRule: Simplifies x³ + y³ + z³ → 3xyz when x + y + z = 0
// Classic identity: x³ + y³ + z³ - 3xyz = (x+y+z)(x²+y²+z²-xy-yz-zx)
// When x+y+z = 0, we get x³ + y³ + z³ = 3xyz
//
// This handles cyclic differences: (a-b)³ + (b-c)³ + (c-a)³ = 3(a-b)(b-c)(c-a)
// because (a-b) + (b-c) + (c-a) = 0 always
define_rule!(
    SumThreeCubesZeroRule,
    "Sum of Three Cubes (Zero Sum Identity)",
    |ctx, expr| {
        use crate::helpers::flatten_add;

        // Match Add expressions only
        match ctx.get(expr) {
            Expr::Add(_, _) => {}
            _ => return None,
        }

        // Flatten the sum
        let mut terms = Vec::new();
        flatten_add(ctx, expr, &mut terms);

        // We need at least 3 cube terms (and no more for the pure identity)
        // For safety, only match exactly 3 cubes with no other terms
        if terms.len() != 3 {
            return None;
        }

        // Extract bases from cubes: term must be Pow(base, 3)
        let mut bases: Vec<cas_ast::ExprId> = Vec::new();
        for &term in &terms {
            let (base, is_neg) = match ctx.get(term).clone() {
                Expr::Pow(b, e) => {
                    if let Expr::Number(n) = ctx.get(e).clone() {
                        if n.is_integer() && n.to_integer() == num_bigint::BigInt::from(3) {
                            (b, false)
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                }
                Expr::Neg(inner) => {
                    // Handle -(x^3) form
                    if let Expr::Pow(b, e) = ctx.get(inner).clone() {
                        if let Expr::Number(n) = ctx.get(e).clone() {
                            if n.is_integer() && n.to_integer() == num_bigint::BigInt::from(3) {
                                (b, true)
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
                _ => return None,
            };

            // For negative cubes (-x³), we need to negate the base
            if is_neg {
                let neg_base = ctx.add(Expr::Neg(base));
                bases.push(neg_base);
            } else {
                bases.push(base);
            }
        }

        // Check if bases sum to zero structurally
        // Build the sum x + y + z and try to simplify it
        let sum_bases = {
            let xy = ctx.add(Expr::Add(bases[0], bases[1]));
            ctx.add(Expr::Add(xy, bases[2]))
        };

        // Try to prove the sum is zero by simplifying
        if !is_structurally_zero(ctx, sum_bases) {
            return None;
        }

        // Sum of bases is zero! Apply the identity: x³ + y³ + z³ = 3xyz
        let three = ctx.num(3);
        // Build 3 * x * y * z
        let xy = crate::rules::algebra::helpers::smart_mul(ctx, bases[0], bases[1]);
        let xyz = crate::rules::algebra::helpers::smart_mul(ctx, xy, bases[2]);
        let inner_result = crate::rules::algebra::helpers::smart_mul(ctx, three, xyz);
        // Wrap in __hold to prevent DistributeRule from expanding
        let result = ctx.call("__hold", vec![inner_result]);

        Some(
            Rewrite::new(result)
                .desc("x³ + y³ + z³ = 3xyz (when x + y + z = 0)")
                .local(expr, result),
        )
    }
);

/// Check if an expression is structurally zero after simplification
/// This handles cases like (a-b) + (b-c) + (c-a) = 0
fn is_structurally_zero(ctx: &mut cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    use num_traits::Zero;

    // First, simple check: is it literally 0?
    if let Expr::Number(n) = ctx.get(expr) {
        return n.is_zero();
    }

    // Flatten the sum and collect terms with signs
    // For (a-b) + (b-c) + (c-a), we expect:
    // +a, -b, +b, -c, +c, -a → all cancel
    let mut atomic_terms: std::collections::HashMap<String, i32> = std::collections::HashMap::new();

    fn collect_atoms(
        ctx: &cas_ast::Context,
        expr: cas_ast::ExprId,
        sign: i32,
        atoms: &mut std::collections::HashMap<String, i32>,
    ) {
        match ctx.get(expr).clone() {
            Expr::Add(l, r) => {
                collect_atoms(ctx, l, sign, atoms);
                collect_atoms(ctx, r, sign, atoms);
            }
            Expr::Sub(l, r) => {
                collect_atoms(ctx, l, sign, atoms);
                collect_atoms(ctx, r, -sign, atoms);
            }
            Expr::Neg(inner) => {
                collect_atoms(ctx, inner, -sign, atoms);
            }
            _ => {
                // Use display string as key for structural comparison
                let key = format!(
                    "{}",
                    cas_ast::DisplayExpr {
                        context: ctx,
                        id: expr
                    }
                );
                *atoms.entry(key).or_insert(0) += sign;
            }
        }
    }

    collect_atoms(ctx, expr, 1, &mut atomic_terms);

    // Check if all coefficients are zero
    atomic_terms.values().all(|&coef| coef == 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn test_is_nary_conjugate_pair_sophie_germain() {
        let mut ctx = cas_ast::Context::new();

        // Parse both sides of the product
        let l = parse("a^2 + 2*b^2 + 2*a*b", &mut ctx).expect("parse L");
        let r = parse("a^2 + 2*b^2 - 2*a*b", &mut ctx).expect("parse R");

        println!(
            "L = {}",
            cas_ast::display::DisplayExpr {
                context: &ctx,
                id: l
            }
        );
        println!(
            "R = {}",
            cas_ast::display::DisplayExpr {
                context: &ctx,
                id: r
            }
        );

        let result = is_nary_conjugate_pair(&mut ctx, l, r);

        println!("Result: {:?}", result);

        assert!(result.is_some(), "Should detect conjugate pair");

        if let Some((u, v)) = result {
            println!(
                "U = {}",
                cas_ast::display::DisplayExpr {
                    context: &ctx,
                    id: u
                }
            );
            println!(
                "V = {}",
                cas_ast::display::DisplayExpr {
                    context: &ctx,
                    id: v
                }
            );
        }
    }

    #[test]
    fn test_is_conjugate_pair_simple() {
        let mut ctx = cas_ast::Context::new();

        let l = parse("x + 1", &mut ctx).expect("parse L");
        let r = parse("x - 1", &mut ctx).expect("parse R");

        let result = is_conjugate_pair(&ctx, l, r);

        assert!(result.is_some(), "Should detect simple conjugate pair");
    }

    #[test]
    fn test_difference_of_squares_rule_on_product() {
        use crate::parent_context::ParentContext;
        use crate::rule::Rule;

        let mut ctx = cas_ast::Context::new();

        // Parse the full product
        let expr =
            parse("(a^2 + 2*b^2 + 2*a*b)*(a^2 + 2*b^2 - 2*a*b)", &mut ctx).expect("parse product");

        println!(
            "Product = {}",
            cas_ast::display::DisplayExpr {
                context: &ctx,
                id: expr
            }
        );

        // Apply the rule directly
        let rule = DifferenceOfSquaresRule;
        let parent_ctx = ParentContext::root();
        let result = rule.apply(&mut ctx, expr, &parent_ctx);

        println!("Rule result: {:?}", result.is_some());

        assert!(
            result.is_some(),
            "DifferenceOfSquaresRule should match the product"
        );

        if let Some(rewrite) = result {
            println!(
                "Rewrite: {} -> {}",
                cas_ast::display::DisplayExpr {
                    context: &ctx,
                    id: expr
                },
                cas_ast::display::DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                },
            );
        }
    }

    #[test]
    fn test_difference_of_squares_reordered_terms() {
        use crate::parent_context::ParentContext;
        use crate::rule::Rule;

        let mut ctx = cas_ast::Context::new();

        // Parse with the order as appears after canonicalization
        // Note: The REPL shows (a² + 2·b² - 2*a·b)·(a² + 2·b² + 2·a·b)
        let expr =
            parse("(a^2 + 2*b^2 - 2*a*b)*(a^2 + 2*b^2 + 2*a*b)", &mut ctx).expect("parse product");

        println!(
            "Product (reordered) = {}",
            cas_ast::display::DisplayExpr {
                context: &ctx,
                id: expr
            }
        );

        // Apply the rule directly
        let rule = DifferenceOfSquaresRule;
        let parent_ctx = ParentContext::root();
        let result = rule.apply(&mut ctx, expr, &parent_ctx);

        println!("Rule result: {:?}", result.is_some());

        // This should also match because (U-V)(U+V) is the same as (U+V)(U-V)
        assert!(
            result.is_some(),
            "DifferenceOfSquaresRule should match the reordered product"
        );

        if let Some(rewrite) = result {
            println!(
                "Rewrite: {} -> {}",
                cas_ast::display::DisplayExpr {
                    context: &ctx,
                    id: expr
                },
                cas_ast::display::DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                },
            );
        }
    }

    #[test]
    fn test_simplifier_applies_difference_of_squares() {
        use crate::parent_context::ParentContext;
        use crate::rule::Rule;
        use crate::Simplifier;

        // Create simplifier with default rules (includes DifferenceOfSquaresRule)
        let mut simplifier = Simplifier::with_default_rules();

        // Parse the product
        let expr = parse(
            "(a^2 + 2*b^2 + 2*a*b)*(a^2 + 2*b^2 - 2*a*b)",
            &mut simplifier.context,
        )
        .expect("parse");

        println!(
            "Input: {}",
            cas_ast::display::DisplayExpr {
                context: &simplifier.context,
                id: expr
            }
        );

        // Run simplifier
        let (result, steps) = simplifier.simplify(expr);

        println!(
            "Output: {}",
            cas_ast::display::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        println!("Number of steps: {}", steps.len());
        for step in &steps {
            println!("  Step: {}", step.rule_name);
        }

        // Now try to apply DifferenceOfSquaresRule directly to the OUTPUT
        // to see if it would have matched if given a chance
        let rule = DifferenceOfSquaresRule;
        let parent_ctx = ParentContext::root();
        let rule_result = rule.apply(&mut simplifier.context, result, &parent_ctx);

        println!(
            "DifferenceOfSquaresRule direct application to OUTPUT: {:?}",
            rule_result.is_some()
        );

        // Inspect the structure of result
        // Extract ExprIds first to avoid borrow conflicts
        let factors = {
            match simplifier.context.get(result) {
                cas_ast::Expr::Mul(l, r) => Some((*l, *r)),
                _ => None,
            }
        };

        if let Some((l, r)) = factors {
            // Verify that we can identify this as a conjugate pair and the rule applies
            let conjugate = is_nary_conjugate_pair(&mut simplifier.context, l, r);
            // After the fix, the conjugate pair should be recognized
            assert!(
                conjugate.is_some(),
                "is_nary_conjugate_pair should recognize the canonicalized conjugate pair"
            );
        }

        // Verify that DifferenceOfSquaresRule was applied (indicated by step name)
        let dos_applied = steps
            .iter()
            .any(|s| s.rule_name.contains("Difference of Squares"));
        assert!(
            dos_applied,
            "DifferenceOfSquaresRule should be applied during simplification"
        );
    }
}
