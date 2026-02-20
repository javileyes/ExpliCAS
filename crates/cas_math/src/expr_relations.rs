//! Structural relation helpers between expressions.
//!
//! These predicates are used by polynomial/factoring rules to detect
//! negation and conjugate relationships without binding to engine modules.

use crate::polynomial::Polynomial;
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed};
use std::cmp::Ordering;

/// Check if two expressions are structural negations of each other.
///
/// Supports:
/// - `Neg(a)`
/// - `Mul(-1, a)` and `Mul(a, -1)`
pub fn is_negation(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    if let (Expr::Number(n_a), Expr::Number(n_b)) = (ctx.get(a), ctx.get(b)) {
        if n_a == &(-n_b.clone()) {
            return true;
        }
    }

    check_negation_structure(ctx, b, a)
        || check_negation_structure(ctx, a, b)
        || check_negated_mul_coeff(ctx, a, b)
}

/// Extract the inner expression from a structural negation:
/// - `Neg(x)` -> `x`
/// - `Mul(-1, x)` / `Mul(x, -1)` -> `x`
pub fn extract_negated_inner(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Neg(inner) => Some(*inner),
        Expr::Mul(l, r) => {
            if let Expr::Number(n) = ctx.get(*l) {
                if *n == -BigRational::one() {
                    return Some(*r);
                }
            }
            if let Expr::Number(n) = ctx.get(*r) {
                if *n == -BigRational::one() {
                    return Some(*l);
                }
            }
            None
        }
        _ => None,
    }
}

fn check_negation_structure(ctx: &Context, potential_neg: ExprId, original: ExprId) -> bool {
    if let Some(inner) = extract_negated_inner(ctx, potential_neg) {
        return compare_expr(ctx, original, inner) == Ordering::Equal;
    }
    false
}

fn check_negated_mul_coeff(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    let Some((coeff_a, term_a)) = extract_numeric_factor(ctx, a) else {
        return false;
    };
    let Some((coeff_b, term_b)) = extract_numeric_factor(ctx, b) else {
        return false;
    };

    coeff_a == -coeff_b && compare_expr(ctx, term_a, term_b) == Ordering::Equal
}

fn extract_numeric_factor(ctx: &Context, expr: ExprId) -> Option<(BigRational, ExprId)> {
    let Expr::Mul(l, r) = ctx.get(expr) else {
        return None;
    };

    if let Expr::Number(n) = ctx.get(*l) {
        return Some((n.clone(), *r));
    }
    if let Expr::Number(n) = ctx.get(*r) {
        return Some((n.clone(), *l));
    }
    None
}

/// Check whether `a` and `b` are a conjugate additive pair.
///
/// Recognizes:
/// - `(A + B)` with `(A - B)` (order-insensitive on additive terms)
/// - canonicalized additive variants like `(A + B)` with `(A + (-B))`
pub fn is_conjugate_add_sub(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    let a_expr = ctx.get(a);
    let b_expr = ctx.get(b);

    match (a_expr, b_expr) {
        (Expr::Add(a1, a2), Expr::Sub(b1, b2)) | (Expr::Sub(b1, b2), Expr::Add(a1, a2)) => {
            let a1 = *a1;
            let a2 = *a2;
            let b1 = *b1;
            let b2 = *b2;

            if compare_expr(ctx, a1, b1) == Ordering::Equal
                && compare_expr(ctx, a2, b2) == Ordering::Equal
            {
                return true;
            }
            if compare_expr(ctx, a2, b1) == Ordering::Equal
                && compare_expr(ctx, a1, b2) == Ordering::Equal
            {
                return true;
            }
            false
        }
        (Expr::Add(a1, a2), Expr::Add(b1, b2)) => {
            let a1 = *a1;
            let a2 = *a2;
            let b1 = *b1;
            let b2 = *b2;

            if is_negation(ctx, a2, b2) && compare_expr(ctx, a1, b1) == Ordering::Equal {
                return true;
            }
            if is_negation(ctx, a2, b1) && compare_expr(ctx, a1, b2) == Ordering::Equal {
                return true;
            }
            if is_negation(ctx, a1, b2) && compare_expr(ctx, a2, b1) == Ordering::Equal {
                return true;
            }
            if is_negation(ctx, a1, b1) && compare_expr(ctx, a2, b2) == Ordering::Equal {
                return true;
            }
            false
        }
        _ => false,
    }
}

/// Extract `(A, B)` when two expressions are an additive conjugate pair.
///
/// Recognizes:
/// - `(A + B)` with `(A - B)` (including swapped term order)
/// - canonicalized additive form `(A + B)` with `(A + (-B))`
///
/// Returns `Some((A, B))` preserving the base orientation for difference-of-squares usage.
pub fn conjugate_add_sub_pair(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, ExprId)> {
    let left_expr = ctx.get(left);
    let right_expr = ctx.get(right);

    match (left_expr, right_expr) {
        (Expr::Add(a1, a2), Expr::Sub(b1, b2)) | (Expr::Sub(b1, b2), Expr::Add(a1, a2)) => {
            let a1 = *a1;
            let a2 = *a2;
            let b1 = *b1;
            let b2 = *b2;

            if compare_expr(ctx, a1, b1) == Ordering::Equal
                && compare_expr(ctx, a2, b2) == Ordering::Equal
            {
                return Some((a1, a2));
            }
            if compare_expr(ctx, a2, b1) == Ordering::Equal
                && compare_expr(ctx, a1, b2) == Ordering::Equal
            {
                return Some((b1, b2));
            }
            None
        }
        (Expr::Add(a1, a2), Expr::Add(b1, b2)) => {
            let a1 = *a1;
            let a2 = *a2;
            let b1 = *b1;
            let b2 = *b2;

            if is_negation(ctx, a2, b2) && compare_expr(ctx, a1, b1) == Ordering::Equal {
                return Some((a1, a2));
            }
            if is_negation(ctx, a2, b1) && compare_expr(ctx, a1, b2) == Ordering::Equal {
                return Some((a1, a2));
            }
            if is_negation(ctx, a1, b2) && compare_expr(ctx, a2, b1) == Ordering::Equal {
                return Some((a2, a1));
            }
            if is_negation(ctx, a1, b1) && compare_expr(ctx, a2, b2) == Ordering::Equal {
                return Some((a2, a1));
            }
            None
        }
        _ => None,
    }
}

/// Check whether two binomials are conjugates with sign normalization.
///
/// Recognizes cases like:
/// - `(x + 1)` vs `(x - 1)`
/// - `(x - 1)` vs `(1 + x)`
/// - `(-1 + x)` vs `(1 + x)` (numeric sign normalization)
pub fn is_conjugate_binomial(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    let (a_terms, a_base_signs) = match ctx.get(a) {
        Expr::Add(x, y) => (vec![*x, *y], vec![true, true]),
        Expr::Sub(x, y) => (vec![*x, *y], vec![true, false]),
        _ => return false,
    };

    let (b_terms, b_base_signs) = match ctx.get(b) {
        Expr::Add(x, y) => (vec![*x, *y], vec![true, true]),
        Expr::Sub(x, y) => (vec![*x, *y], vec![true, false]),
        _ => return false,
    };

    let mut a_norm = Vec::new();
    let mut a_signs = Vec::new();
    for (&term, &base_sign) in a_terms.iter().zip(a_base_signs.iter()) {
        let (norm_term, is_pos) = normalize_term_sign(ctx, term);
        a_norm.push(norm_term);
        a_signs.push(base_sign == is_pos);
    }

    let mut b_norm = Vec::new();
    let mut b_signs = Vec::new();
    for (&term, &base_sign) in b_terms.iter().zip(b_base_signs.iter()) {
        let (norm_term, is_pos) = normalize_term_sign(ctx, term);
        b_norm.push(norm_term);
        b_signs.push(base_sign == is_pos);
    }

    let same_order = a_norm.len() == b_norm.len()
        && a_norm
            .iter()
            .zip(b_norm.iter())
            .all(|(&x, &y)| terms_equal_normalized(ctx, x, y));

    let swapped_order = a_norm.len() == 2
        && b_norm.len() == 2
        && terms_equal_normalized(ctx, a_norm[0], b_norm[1])
        && terms_equal_normalized(ctx, a_norm[1], b_norm[0]);

    if !same_order && !swapped_order {
        return false;
    }

    let b_signs_to_check = if same_order {
        b_signs
    } else {
        vec![b_signs[1], b_signs[0]]
    };

    let diff_count = a_signs
        .iter()
        .zip(b_signs_to_check.iter())
        .filter(|(a_sign, b_sign)| a_sign != b_sign)
        .count();

    diff_count == 1
}

fn terms_equal_normalized(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    use num_traits::Signed;

    if let (Expr::Number(na), Expr::Number(nb)) = (ctx.get(a), ctx.get(b)) {
        return na.abs() == nb.abs();
    }

    compare_expr(ctx, a, b) == Ordering::Equal
}

fn normalize_term_sign(ctx: &Context, expr: ExprId) -> (ExprId, bool) {
    match ctx.get(expr) {
        Expr::Neg(inner) => (*inner, false),
        Expr::Number(n) => (expr, !n.is_negative()),
        _ => (expr, true),
    }
}

#[derive(Clone, Copy)]
enum AddSign {
    Pos,
    Neg,
}

impl AddSign {
    fn to_i32(self) -> i32 {
        match self {
            AddSign::Pos => 1,
            AddSign::Neg => -1,
        }
    }

    fn negate(self) -> Self {
        match self {
            AddSign::Pos => AddSign::Neg,
            AddSign::Neg => AddSign::Pos,
        }
    }
}

fn add_terms_signed(ctx: &Context, root: ExprId) -> Vec<(ExprId, AddSign)> {
    let mut out = Vec::new();
    let mut stack = vec![(root, AddSign::Pos)];

    while let Some((id, sign)) = stack.pop() {
        if crate::poly_result::is_poly_ref_or_result(ctx, id) {
            out.push((id, sign));
            continue;
        }

        let id = cas_ast::hold::unwrap_hold(ctx, id);
        match ctx.get(id) {
            Expr::Add(l, r) => {
                stack.push((*r, sign));
                stack.push((*l, sign));
            }
            Expr::Sub(l, r) => {
                stack.push((*r, sign.negate()));
                stack.push((*l, sign));
            }
            Expr::Neg(inner) => {
                stack.push((*inner, sign.negate()));
            }
            _ => out.push((id, sign)),
        }
    }

    out
}

fn mul_leaves(ctx: &Context, root: ExprId) -> Vec<ExprId> {
    let mut out = Vec::new();
    let mut stack = vec![root];

    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Mul(l, r) => {
                stack.push(*r);
                stack.push(*l);
            }
            _ => out.push(id),
        }
    }

    out
}

fn build_balanced_add(ctx: &mut Context, terms: &[ExprId]) -> ExprId {
    match terms.len() {
        0 => ctx.num(0),
        1 => terms[0],
        2 => ctx.add(Expr::Add(terms[0], terms[1])),
        n => {
            let mid = n / 2;
            let left = build_balanced_add(ctx, &terms[..mid]);
            let right = build_balanced_add(ctx, &terms[mid..]);
            ctx.add(Expr::Add(left, right))
        }
    }
}

/// Detect conjugate pair in n-ary additive forms: `(U+V)` and `(U-V)`.
///
/// Returns `(U, V)` where `U` can be a sum of multiple terms and `V` is the
/// single sign-flipped term.
pub fn conjugate_nary_add_sub_pair(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, ExprId)> {
    fn normalize_term(ctx: &mut Context, term: ExprId) -> (ExprId, i32) {
        enum TermKind {
            Neg(ExprId),
            Num(BigRational),
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
                let factors = mul_leaves(ctx, term);
                let mut overall_sign: i32 = 1;
                let mut unsigned_factors: Vec<ExprId> = Vec::new();

                for factor in factors {
                    let factor_info = match ctx.get(factor) {
                        Expr::Neg(inner) => Some(("neg", *inner)),
                        Expr::Number(n) if n.is_negative() => {
                            let neg_n = -n.clone();
                            let pos_term = ctx.add(Expr::Number(neg_n));
                            overall_sign *= -1;
                            unsigned_factors.push(pos_term);
                            continue;
                        }
                        _ => None,
                    };
                    if let Some(("neg", inner)) = factor_info {
                        overall_sign *= -1;
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

                unsigned_factors.sort_by(|a, b| compare_expr(ctx, *a, *b));

                let canonical_core = if unsigned_factors.is_empty() {
                    ctx.num(1)
                } else if unsigned_factors.len() == 1 {
                    unsigned_factors[0]
                } else {
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

    let left_terms = add_terms_signed(ctx, left);
    let right_terms = add_terms_signed(ctx, right);

    if left_terms.len() != right_terms.len() || left_terms.is_empty() {
        return None;
    }
    const MAX_TERMS: usize = 16;
    if left_terms.len() > MAX_TERMS {
        return None;
    }

    let left_normalized: Vec<(ExprId, i32)> = left_terms
        .iter()
        .map(|&(term, sign)| {
            let (core, term_sign) = normalize_term(ctx, term);
            (core, sign.to_i32() * term_sign)
        })
        .collect();
    let right_normalized: Vec<(ExprId, i32)> = right_terms
        .iter()
        .map(|&(term, sign)| {
            let (core, term_sign) = normalize_term(ctx, term);
            (core, sign.to_i32() * term_sign)
        })
        .collect();

    let mut groups: Vec<(ExprId, i32, i32)> = Vec::new();
    for &(core, sign) in &left_normalized {
        let mut found = false;
        for (rep, l_count, _) in &mut groups {
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
    for &(core, sign) in &right_normalized {
        let mut found = false;
        for (rep, _, r_count) in &mut groups {
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

    let mut v_term: Option<ExprId> = None;
    let mut common_terms: Vec<(ExprId, i32)> = Vec::new();

    for (core, l_count, r_count) in &groups {
        let diff = l_count - r_count;
        if diff == 0 {
            if *l_count != 0 {
                common_terms.push((*core, *l_count));
            }
        } else if diff == 2 || diff == -2 {
            if v_term.is_some() {
                return None;
            }
            v_term = Some(*core);
        } else {
            return None;
        }
    }

    let v = v_term?;
    if common_terms.is_empty() {
        return None;
    }

    let u_terms: Vec<ExprId> = common_terms
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

/// Count additive terms by flattening `Add/Sub` recursively.
///
/// `Neg` preserves term count of its inner expression.
pub fn count_additive_terms(ctx: &Context, expr: ExprId) -> usize {
    match ctx.get(expr) {
        Expr::Add(l, r) => count_additive_terms(ctx, *l) + count_additive_terms(ctx, *r),
        Expr::Sub(l, r) => count_additive_terms(ctx, *l) + count_additive_terms(ctx, *r),
        Expr::Neg(inner) => count_additive_terms(ctx, *inner),
        _ => 1,
    }
}

/// Collect additive terms as `(term, is_negated)` pairs.
///
/// Uses canonical additive decomposition and keeps `__hold(...)` transparent.
fn collect_additive_terms_signed(
    ctx: &Context,
    expr: ExprId,
    negated: bool,
    terms: &mut Vec<(ExprId, bool)>,
) {
    let signed_terms = add_terms_signed(ctx, expr);

    for (term, sign) in signed_terms {
        let is_negated = match sign {
            AddSign::Pos => negated,
            AddSign::Neg => !negated,
        };
        terms.push((term, is_negated));
    }
}

fn normalize_term_sign_for_poly_eq(ctx: &Context, term: ExprId, neg: bool) -> (ExprId, bool) {
    // Check if it's a Mul with a negative numeric factor.
    if let Expr::Mul(l, r) = ctx.get(term) {
        if let Expr::Number(n) = ctx.get(*l) {
            if n.is_negative() {
                return (term, !neg);
            }
        }
        if let Expr::Number(n) = ctx.get(*r) {
            if n.is_negative() {
                return (term, !neg);
            }
        }
    }

    // Or a negative numeric literal directly.
    if let Expr::Number(n) = ctx.get(term) {
        if n.is_negative() {
            return (term, !neg);
        }
    }

    (term, neg)
}

/// Check if two expressions are polynomially equal (same after expansion).
pub fn poly_equal(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    if a == b {
        return true;
    }

    if compare_expr(ctx, a, b) == Ordering::Equal {
        return true;
    }

    let expr_a = ctx.get(a);
    let expr_b = ctx.get(b);

    if let (Expr::Pow(base_a, exp_a), Expr::Pow(base_b, exp_b)) = (expr_a, expr_b) {
        if poly_equal(ctx, *exp_a, *exp_b) {
            return poly_equal(ctx, *base_a, *base_b);
        }
    }

    if let (Expr::Mul(l_a, r_a), Expr::Mul(l_b, r_b)) = (expr_a, expr_b) {
        if (poly_equal(ctx, *l_a, *l_b) && poly_equal(ctx, *r_a, *r_b))
            || (poly_equal(ctx, *l_a, *r_b) && poly_equal(ctx, *r_a, *l_b))
        {
            return true;
        }

        // Same up to opposite numeric coefficient sign.
        if let (Expr::Number(n_a), Expr::Number(n_b)) = (ctx.get(*l_a), ctx.get(*l_b)) {
            if n_a == &-n_b.clone() && poly_equal(ctx, *r_a, *r_b) {
                return true;
            }
        }
        if let (Expr::Number(n_a), Expr::Number(n_b)) = (ctx.get(*r_a), ctx.get(*r_b)) {
            if n_a == &-n_b.clone() && poly_equal(ctx, *l_a, *l_b) {
                return true;
            }
        }
    }

    if let (Expr::Neg(inner_a), Expr::Neg(inner_b)) = (expr_a, expr_b) {
        return poly_equal(ctx, *inner_a, *inner_b);
    }

    let is_additive_a = matches!(expr_a, Expr::Add(_, _) | Expr::Sub(_, _));
    let is_additive_b = matches!(expr_b, Expr::Add(_, _) | Expr::Sub(_, _));

    if is_additive_a && is_additive_b {
        let mut terms_a: Vec<(ExprId, bool)> = Vec::new();
        let mut terms_b: Vec<(ExprId, bool)> = Vec::new();
        collect_additive_terms_signed(ctx, a, false, &mut terms_a);
        collect_additive_terms_signed(ctx, b, false, &mut terms_b);

        if terms_a.len() == terms_b.len() {
            let mut matched = vec![false; terms_b.len()];
            for (term_a, neg_a) in &terms_a {
                let mut found = false;
                let (norm_a, eff_neg_a) = normalize_term_sign_for_poly_eq(ctx, *term_a, *neg_a);

                for (j, (term_b, neg_b)) in terms_b.iter().enumerate() {
                    if matched[j] {
                        continue;
                    }
                    let (norm_b, eff_neg_b) = normalize_term_sign_for_poly_eq(ctx, *term_b, *neg_b);
                    if eff_neg_a != eff_neg_b {
                        continue;
                    }
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

    // Fallback: canonical univariate polynomial comparison.
    let vars_a: Vec<_> = cas_ast::collect_variables(ctx, a).into_iter().collect();
    let vars_b: Vec<_> = cas_ast::collect_variables(ctx, b).into_iter().collect();
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

/// Check if an expression is structurally zero after additive cancellation.
///
/// This handles cyclic additive telescoping shapes like:
/// `(a-b) + (b-c) + (c-a)`.
pub fn is_structurally_zero(ctx: &Context, expr: ExprId) -> bool {
    use num_traits::Zero;

    if let Expr::Number(n) = ctx.get(expr) {
        return n.is_zero();
    }

    fn add_atom(ctx: &Context, atoms: &mut Vec<(ExprId, i32)>, atom: ExprId, sign: i32) {
        for (existing, coeff) in atoms.iter_mut() {
            if compare_expr(ctx, *existing, atom) == Ordering::Equal {
                *coeff += sign;
                return;
            }
        }
        atoms.push((atom, sign));
    }

    fn collect_atoms(ctx: &Context, expr: ExprId, sign: i32, atoms: &mut Vec<(ExprId, i32)>) {
        match ctx.get(expr) {
            Expr::Add(l, r) => {
                collect_atoms(ctx, *l, sign, atoms);
                collect_atoms(ctx, *r, sign, atoms);
            }
            Expr::Sub(l, r) => {
                collect_atoms(ctx, *l, sign, atoms);
                collect_atoms(ctx, *r, -sign, atoms);
            }
            Expr::Neg(inner) => {
                collect_atoms(ctx, *inner, -sign, atoms);
            }
            _ => add_atom(ctx, atoms, expr, sign),
        }
    }

    let mut atoms = Vec::new();
    collect_atoms(ctx, expr, 1, &mut atoms);
    atoms.into_iter().all(|(_, coeff)| coeff == 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn negation_detection_covers_neg_and_mul_minus_one() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("x");
        let neg_x = parse("-x", &mut ctx).expect("-x");
        let mul_neg_x = parse("(-1)*x", &mut ctx).expect("(-1)*x");
        let three = parse("3", &mut ctx).expect("3");
        let neg_three = parse("-3", &mut ctx).expect("-3");
        assert!(is_negation(&ctx, x, neg_x));
        assert!(is_negation(&ctx, x, mul_neg_x));
        assert!(is_negation(&ctx, three, neg_three));
    }

    #[test]
    fn negation_detection_covers_opposite_mul_coefficients() {
        let mut ctx = Context::new();
        let pos = parse("2*x", &mut ctx).expect("2*x");
        let neg = parse("-2*x", &mut ctx).expect("-2*x");
        let non_neg = parse("3*x", &mut ctx).expect("3*x");
        assert!(is_negation(&ctx, pos, neg));
        assert!(!is_negation(&ctx, pos, non_neg));
    }

    #[test]
    fn extract_negated_inner_handles_neg_and_mul_minus_one() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("x");
        let neg_x = parse("-x", &mut ctx).expect("-x");
        let mul_left = parse("(-1)*x", &mut ctx).expect("(-1)*x");
        let mul_right = parse("x*(-1)", &mut ctx).expect("x*(-1)");
        let pos_x = parse("x", &mut ctx).expect("x-again");

        assert_eq!(extract_negated_inner(&ctx, neg_x), Some(x));
        assert_eq!(extract_negated_inner(&ctx, mul_left), Some(x));
        assert_eq!(extract_negated_inner(&ctx, mul_right), Some(x));
        assert_eq!(extract_negated_inner(&ctx, pos_x), None);
    }

    #[test]
    fn conjugate_detection_for_add_sub_and_add_neg() {
        let mut ctx = Context::new();
        let add = parse("a+b", &mut ctx).expect("a+b");
        let sub = parse("a-b", &mut ctx).expect("a-b");
        let add_neg = parse("a+(-b)", &mut ctx).expect("a+(-b)");
        let same = parse("a+b", &mut ctx).expect("a+b");
        assert!(is_conjugate_add_sub(&ctx, add, sub));
        assert!(is_conjugate_add_sub(&ctx, add, add_neg));
        assert!(!is_conjugate_add_sub(&ctx, add, same));
    }

    #[test]
    fn conjugate_pair_extraction_returns_terms() {
        let mut ctx = Context::new();
        let left = parse("x+3", &mut ctx).expect("left");
        let right = parse("x-3", &mut ctx).expect("right");
        let pair = conjugate_add_sub_pair(&ctx, left, right).expect("pair");
        let x = parse("x", &mut ctx).expect("x");
        let three = parse("3", &mut ctx).expect("3");
        assert_eq!(compare_expr(&ctx, pair.0, x), Ordering::Equal);
        assert_eq!(compare_expr(&ctx, pair.1, three), Ordering::Equal);
    }

    #[test]
    fn conjugate_binomial_detection_handles_sign_normalization() {
        let mut ctx = Context::new();
        let left = parse("x-1", &mut ctx).expect("x-1");
        let right = parse("x+1", &mut ctx).expect("x+1");
        let same = parse("x+1", &mut ctx).expect("x+1");
        assert!(is_conjugate_binomial(&ctx, left, right));
        assert!(!is_conjugate_binomial(&ctx, right, same));
    }

    #[test]
    fn additive_term_count_handles_nested_sub_and_neg() {
        let mut ctx = Context::new();
        let nested = parse("a-(b-c)", &mut ctx).expect("nested");
        let neg = parse("-(a+b)", &mut ctx).expect("neg");
        assert_eq!(count_additive_terms(&ctx, nested), 3);
        assert_eq!(count_additive_terms(&ctx, neg), 2);
    }

    #[test]
    fn collect_additive_terms_signed_tracks_signs_across_subtractions() {
        let mut ctx = Context::new();
        let expr = parse("a-(b-c)", &mut ctx).expect("expr");
        let a = parse("a", &mut ctx).expect("a");
        let b = parse("b", &mut ctx).expect("b");
        let c = parse("c", &mut ctx).expect("c");

        let mut terms = Vec::new();
        collect_additive_terms_signed(&ctx, expr, false, &mut terms);

        assert_eq!(terms.len(), 3);
        assert!(terms.iter().any(|(id, neg)| *id == a && !*neg));
        assert!(terms.iter().any(|(id, neg)| *id == b && *neg));
        assert!(terms.iter().any(|(id, neg)| *id == c && !*neg));
    }

    #[test]
    fn poly_equal_matches_equivalent_univariate_forms() {
        let mut ctx = Context::new();
        let a = parse("(x+1)^2", &mut ctx).expect("a");
        let b = parse("x^2 + 2*x + 1", &mut ctx).expect("b");
        let c = parse("x^2 + 1", &mut ctx).expect("c");

        assert!(poly_equal(&ctx, a, b));
        assert!(!poly_equal(&ctx, a, c));
    }

    #[test]
    fn conjugate_nary_pair_detects_sophie_germain_shape() {
        let mut ctx = Context::new();
        let left = parse("a^2 + 2*b^2 + 2*a*b", &mut ctx).expect("left");
        let right = parse("a^2 + 2*b^2 - 2*a*b", &mut ctx).expect("right");
        assert!(conjugate_nary_add_sub_pair(&mut ctx, left, right).is_some());
    }

    #[test]
    fn structural_zero_detects_telescoping_sum() {
        let mut ctx = Context::new();
        let expr = parse("(a-b)+(b-c)+(c-a)", &mut ctx).expect("expr");
        assert!(is_structurally_zero(&ctx, expr));
    }

    #[test]
    fn structural_zero_rejects_nonzero_sum() {
        let mut ctx = Context::new();
        let expr = parse("(a-b)+(b-c)", &mut ctx).expect("expr");
        assert!(!is_structurally_zero(&ctx, expr));
    }
}
