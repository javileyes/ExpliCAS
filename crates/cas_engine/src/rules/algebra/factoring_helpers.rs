//! Helper functions for factoring rules.
//!
//! Contains conjugate pair detection, negation checking, and structural zero
//! verification utilities used by the factoring rules.

use cas_ast::Expr;
use std::cmp::Ordering;

/// Check if two expressions form a conjugate pair: (A+B) and (A-B) or vice versa
/// Returns Some((a, b)) if they are conjugates, None otherwise
pub(super) fn is_conjugate_pair(
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
pub(super) fn is_negation(ctx: &cas_ast::Context, a: cas_ast::ExprId, b: cas_ast::ExprId) -> bool {
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
pub(super) fn is_minus_one(ctx: &cas_ast::Context, e: cas_ast::ExprId) -> bool {
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
pub(super) fn is_nary_conjugate_pair(
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
        // Pre-extract variant info from borrow to avoid cloning full Expr
        enum TermKind {
            Neg(cas_ast::ExprId),
            Num(num_rational::BigRational),
            Mul,
            Other,
        }
        let kind = match ctx.get(term) {
            Expr::Neg(inner) => TermKind::Neg(*inner),
            Expr::Number(n) => TermKind::Num(n.clone()),
            Expr::Mul(_, _) => TermKind::Mul,
            _ => TermKind::Other,
        };
        match kind {
            TermKind::Neg(inner) => {
                let (core, sign) = normalize_term(ctx, inner);
                (core, -sign)
            }
            TermKind::Num(n) => {
                if n.is_negative() {
                    let pos_n = -n;
                    let pos_term = ctx.add(Expr::Number(pos_n));
                    (pos_term, -1)
                } else {
                    (term, 1)
                }
            }
            TermKind::Mul => {
                // Flatten the product to get all factors
                let factors = crate::nary::mul_leaves(ctx, term);

                // Extract sign from any negative numeric coefficient
                let mut overall_sign: i32 = 1;
                let mut unsigned_factors: Vec<cas_ast::ExprId> = Vec::new();

                for factor in factors {
                    // Pre-extract variant info from borrow before any ctx.add calls
                    let factor_info = match ctx.get(factor) {
                        Expr::Neg(inner) => Some(("neg", *inner)),
                        Expr::Number(n) if n.is_negative() => {
                            let neg_n = -n.clone();
                            // We need to defer the ctx.add, so store the value
                            let pos_term = ctx.add(Expr::Number(neg_n));
                            overall_sign *= -1;
                            unsigned_factors.push(pos_term);
                            continue;
                        }
                        _ => None,
                    };
                    if let Some(("neg", inner)) = factor_info {
                        overall_sign *= -1;
                        // Check if inner is a negative number
                        if let Expr::Number(n) = ctx.get(inner) {
                            if n.is_negative() {
                                let pos_n = -n.clone();
                                unsigned_factors.push(ctx.add(Expr::Number(pos_n)));
                            } else {
                                unsigned_factors.push(inner);
                            }
                        } else {
                            unsigned_factors.push(inner);
                        }
                    } else {
                        unsigned_factors.push(factor);
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
            TermKind::Other => (term, 1),
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

/// Check if an expression is structurally zero after simplification
/// This handles cases like (a-b) + (b-c) + (c-a) = 0
pub(super) fn is_structurally_zero(ctx: &mut cas_ast::Context, expr: cas_ast::ExprId) -> bool {
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
        match ctx.get(expr) {
            Expr::Add(l, r) => {
                let (l, r) = (*l, *r);
                collect_atoms(ctx, l, sign, atoms);
                collect_atoms(ctx, r, sign, atoms);
            }
            Expr::Sub(l, r) => {
                let (l, r) = (*l, *r);
                collect_atoms(ctx, l, sign, atoms);
                collect_atoms(ctx, r, -sign, atoms);
            }
            Expr::Neg(inner) => {
                collect_atoms(ctx, *inner, -sign, atoms);
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
