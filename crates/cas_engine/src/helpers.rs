//! # Helpers Module
//!
//! This module provides shared utility functions for expression manipulation
//! and pattern matching. These functions are used across multiple rule files
//! to avoid code duplication.
//!
//! ## Categories
//!
//! - **Expression Predicates**: `is_one`, `is_zero`, `is_negative`, `is_half`
//! - **Value Extraction**: `get_integer`, `get_parts`, `get_variant_name`
//! - **Flattening**: `flatten_add`, `flatten_add_sub_chain`, `flatten_mul`, `flatten_mul_chain`
//! - **Trigonometry**: `is_trig_pow`, `get_trig_arg`, `extract_double_angle_arg`
//! - **Pi Helpers**: `is_pi`, `is_pi_over_n`, `build_pi_over_n`
//! - **Roots**: `get_square_root`

use cas_ast::{Context, Expr, ExprId};
use num_traits::{One, Signed, ToPrimitive};

/// Check if expression is a trigonometric function raised to a specific power.
///
/// # Example
/// ```ignore
/// // Matches sin(x)^2
/// is_trig_pow(ctx, expr, "sin", 2)
/// ```
pub fn is_trig_pow(context: &Context, expr: ExprId, name: &str, power: i64) -> bool {
    if let Expr::Pow(base, exp) = context.get(expr) {
        if let Expr::Number(n) = context.get(*exp) {
            if n.is_integer() && n.to_integer() == power.into() {
                if let Expr::Function(func_name, args) = context.get(*base) {
                    return func_name == name && args.len() == 1;
                }
            }
        }
    }
    false
}

/// Extract the argument from a trigonometric power expression.
///
/// For an expression like `sin(x)^2`, returns `Some(x)`.
pub fn get_trig_arg(context: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Pow(base, _) = context.get(expr) {
        if let Expr::Function(_, args) = context.get(*base) {
            if args.len() == 1 {
                return Some(args[0]);
            }
        }
    }
    None
}

pub fn get_square_root(context: &mut Context, expr: ExprId) -> Option<ExprId> {
    // We need to clone the expression to avoid borrowing issues if we need to inspect it deeply
    // But context.get returns reference.
    // We can't hold reference to context while mutating it.
    // So we should extract necessary data first.

    let expr_data = context.get(expr).clone();

    match expr_data {
        Expr::Pow(b, e) => {
            if let Expr::Number(n) = context.get(e) {
                if n.is_integer() {
                    let val = n.to_integer();
                    if &val % 2 == 0.into() {
                        let two = num_bigint::BigInt::from(2);
                        let new_exp_val = (val / two).to_i64()?;
                        let new_exp = context.num(new_exp_val);

                        // If new_exp is 1, simplify to b
                        if let Expr::Number(ne) = context.get(new_exp) {
                            if ne.is_one() {
                                return Some(b);
                            }
                        }
                        return Some(context.add(Expr::Pow(b, new_exp)));
                    }
                }
            }
            None
        }
        Expr::Number(n) => {
            if n.is_integer() && !n.is_negative() {
                let val = n.to_integer();
                let sqrt = val.sqrt();
                if &sqrt * &sqrt == val {
                    return Some(context.num(sqrt.to_i64()?));
                }
            }
            None
        }
        _ => None,
    }
}

pub fn extract_double_angle_arg(context: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Mul(lhs, rhs) = context.get(expr) {
        if let Expr::Number(n) = context.get(*lhs) {
            if n.is_integer() && n.to_integer() == 2.into() {
                return Some(*rhs);
            }
        }
        if let Expr::Number(n) = context.get(*rhs) {
            if n.is_integer() && n.to_integer() == 2.into() {
                return Some(*lhs);
            }
        }
    }
    None
}

/// Flatten an Add chain into a list of terms (simple version).
///
/// This only handles `Add` nodes. For handling `Sub` and `Neg`, use
/// `flatten_add_sub_chain` instead.
pub fn flatten_add(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            flatten_add(ctx, *l, terms);
            flatten_add(ctx, *r, terms);
        }
        _ => terms.push(expr),
    }
}

/// Flatten an Add/Sub chain into a list of terms, converting subtractions to Neg.
/// This is used by collect and grouping modules for like-term collection.
///
/// Unlike `flatten_add`, this handles:
/// - `Add(a, b)` → [a, b]
/// - `Sub(a, b)` → [a, Neg(b)]
/// - `Neg(Neg(x))` → [x]
pub fn flatten_add_sub_chain(ctx: &mut Context, expr: ExprId) -> Vec<ExprId> {
    let mut terms = Vec::new();
    flatten_add_sub_recursive(ctx, expr, &mut terms, false);
    terms
}

fn flatten_add_sub_recursive(
    ctx: &mut Context,
    expr: ExprId,
    terms: &mut Vec<ExprId>,
    negate: bool,
) {
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Add(l, r) => {
            flatten_add_sub_recursive(ctx, l, terms, negate);
            flatten_add_sub_recursive(ctx, r, terms, negate);
        }
        Expr::Sub(l, r) => {
            flatten_add_sub_recursive(ctx, l, terms, negate);
            flatten_add_sub_recursive(ctx, r, terms, !negate);
        }
        Expr::Neg(inner) => {
            // Handle double negation: Neg(Neg(x)) -> x
            flatten_add_sub_recursive(ctx, inner, terms, !negate);
        }
        _ => {
            if negate {
                terms.push(ctx.add(Expr::Neg(expr)));
            } else {
                terms.push(expr);
            }
        }
    }
}

/// Flatten a Mul chain into a list of factors (simple version).
///
/// This only handles `Mul` nodes. For handling `Neg` as `-1 * expr`,
/// use `flatten_mul_chain` instead.
pub fn flatten_mul(ctx: &Context, expr: ExprId, factors: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Mul(l, r) => {
            flatten_mul(ctx, *l, factors);
            flatten_mul(ctx, *r, factors);
        }
        _ => factors.push(expr),
    }
}

/// Flatten a Mul chain into a list of factors, handling Neg as -1 multiplication.
/// Returns Vec<ExprId> where Neg(e) is converted to [num(-1), ...factors of e...]
pub fn flatten_mul_chain(ctx: &mut Context, expr: ExprId) -> Vec<ExprId> {
    let mut factors = Vec::new();
    flatten_mul_recursive(ctx, expr, &mut factors);
    factors
}

fn flatten_mul_recursive(ctx: &mut Context, expr: ExprId, factors: &mut Vec<ExprId>) {
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Mul(l, r) => {
            flatten_mul_recursive(ctx, l, factors);
            flatten_mul_recursive(ctx, r, factors);
        }
        Expr::Neg(e) => {
            // Treat Neg(e) as -1 * e
            let neg_one = ctx.num(-1);
            factors.push(neg_one);
            flatten_mul_recursive(ctx, e, factors);
        }
        _ => {
            factors.push(expr);
        }
    }
}

pub fn get_parts(context: &mut Context, e: ExprId) -> (num_rational::BigRational, ExprId) {
    match context.get(e) {
        Expr::Mul(a, b) => {
            if let Expr::Number(n) = context.get(*a) {
                (n.clone(), *b)
            } else if let Expr::Number(n) = context.get(*b) {
                (n.clone(), *a)
            } else {
                (num_rational::BigRational::one(), e)
            }
        }
        Expr::Number(n) => (n.clone(), context.num(1)), // Treat constant as c * 1
        _ => (num_rational::BigRational::one(), e),
    }
}

// ========== Pi Helpers ==========

/// Check if expression equals π/n for a given denominator (handles both Div and Mul forms)
pub fn is_pi_over_n(ctx: &Context, expr: ExprId, denom: i32) -> bool {
    // Handle Div form: pi/n
    if let Expr::Div(num, den) = ctx.get(expr) {
        if let Expr::Constant(c) = ctx.get(*num) {
            if matches!(c, cas_ast::Constant::Pi) {
                if let Expr::Number(n) = ctx.get(*den) {
                    return *n == num_rational::BigRational::from_integer(denom.into());
                }
            }
        }
    }

    // Handle Mul form: (1/n) * pi
    if let Expr::Mul(l, r) = ctx.get(expr) {
        let (num_part, const_part) = if let Expr::Constant(_) = ctx.get(*l) {
            (*r, *l)
        } else if let Expr::Constant(_) = ctx.get(*r) {
            (*l, *r)
        } else {
            return false;
        };

        if let Expr::Constant(c) = ctx.get(const_part) {
            if matches!(c, cas_ast::Constant::Pi) {
                if let Expr::Number(n) = ctx.get(num_part) {
                    return *n == num_rational::BigRational::new(1.into(), denom.into());
                }
            }
        }
    }

    false
}

/// Check if expression is exactly π
pub fn is_pi(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Constant(cas_ast::Constant::Pi))
}

/// Check if expression equals a specific numeric value
pub fn is_numeric_value(ctx: &Context, expr: ExprId, numer: i32, denom: i32) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        *n == num_rational::BigRational::new(numer.into(), denom.into())
    } else {
        false
    }
}

/// Build π/n expression
pub fn build_pi_over_n(ctx: &mut Context, denom: i64) -> ExprId {
    let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
    let d = ctx.num(denom);
    ctx.add(Expr::Div(pi, d))
}

/// Check if expression equals 1/2
pub fn is_half(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        *n.numer() == 1.into() && *n.denom() == 2.into()
    } else {
        false
    }
}

// ========== Common Expression Predicates ==========
// These functions were previously duplicated across multiple files.
// Now consolidated here for consistency and maintainability.

/// Check if expression is the number 1
pub fn is_one(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        n.is_one()
    } else {
        false
    }
}

/// Check if expression is the number 0
pub fn is_zero(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        num_traits::Zero::is_zero(n)
    } else {
        false
    }
}

/// Check if expression is a negative number
pub fn is_negative(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        n.is_negative()
    } else if let Expr::Neg(_) = ctx.get(expr) {
        true
    } else {
        false
    }
}

/// Try to extract an integer value from an expression
pub fn get_integer(ctx: &Context, expr: ExprId) -> Option<i64> {
    if let Expr::Number(n) = ctx.get(expr) {
        if n.is_integer() {
            n.to_integer().to_i64()
        } else {
            None
        }
    } else {
        None
    }
}

/// Get the variant name of an expression (for debugging/display)
pub fn get_variant_name(expr: &Expr) -> &'static str {
    match expr {
        Expr::Number(_) => "Number",
        Expr::Variable(_) => "Variable",
        Expr::Constant(_) => "Constant",
        Expr::Add(_, _) => "Add",
        Expr::Sub(_, _) => "Sub",
        Expr::Mul(_, _) => "Mul",
        Expr::Div(_, _) => "Div",
        Expr::Pow(_, _) => "Pow",
        Expr::Neg(_) => "Neg",
        Expr::Function(_, _) => "Function",
        Expr::Matrix { .. } => "Matrix",
    }
}

// ========== Normal Form Scoring ==========

/// Count total nodes in an expression tree
pub fn count_all_nodes(ctx: &Context, expr: ExprId) -> usize {
    let mut count = 0;
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        count += 1;
        match ctx.get(id) {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(e) => stack.push(*e),
            Expr::Function(_, args) => stack.extend(args),
            Expr::Matrix { data, .. } => stack.extend(data),
            _ => {}
        }
    }
    count
}

/// Count nodes matching a predicate
pub fn count_nodes_matching<F>(ctx: &Context, expr: ExprId, pred: F) -> usize
where
    F: Fn(&Expr) -> bool,
{
    let mut count = 0;
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        let node = ctx.get(id);
        if pred(node) {
            count += 1;
        }
        match node {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(e) => stack.push(*e),
            Expr::Function(_, args) => stack.extend(args),
            Expr::Matrix { data, .. } => stack.extend(data),
            _ => {}
        }
    }
    count
}

/// Score expression for normal form quality (lower is better).
/// Returns (divs_subs, total_nodes, mul_inversions) for lexicographic comparison.
///
/// Expressions with fewer Div/Sub nodes are preferred (C2 canonical form).
/// Ties are broken by total node count (simpler is better).
/// Final tie-breaker: fewer out-of-order adjacent pairs in Mul chains.
///
/// For performance-critical comparisons, use `compare_nf_score_lazy` instead.
pub fn nf_score(ctx: &Context, id: ExprId) -> (usize, usize, usize) {
    let divs_subs = count_nodes_matching(ctx, id, |e| matches!(e, Expr::Div(..) | Expr::Sub(..)));
    let total = count_all_nodes(ctx, id);
    let inversions = mul_unsorted_adjacent(ctx, id);
    (divs_subs, total, inversions)
}

/// First two components of nf_score: (divs_subs, total_nodes)
/// Uses single traversal for efficiency (counts both in one pass).
fn nf_score_base(ctx: &Context, id: ExprId) -> (usize, usize) {
    let mut divs_subs = 0;
    let mut total = 0;
    let mut stack = vec![id];

    while let Some(node_id) = stack.pop() {
        total += 1;

        match ctx.get(node_id) {
            Expr::Div(..) | Expr::Sub(..) => divs_subs += 1,
            _ => {}
        }

        // Push children
        match ctx.get(node_id) {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args),
            Expr::Matrix { data, .. } => stack.extend(data),
            _ => {}
        }
    }

    (divs_subs, total)
}

/// Compare nf_score lazily: only computes mul_unsorted_adjacent if first two components tie.
/// Returns true if `after` is strictly better (lower) than `before`.
pub fn nf_score_after_is_better(ctx: &Context, before: ExprId, after: ExprId) -> bool {
    let before_base = nf_score_base(ctx, before);
    let after_base = nf_score_base(ctx, after);

    // Compare first two components
    if after_base < before_base {
        return true; // Clear improvement
    }
    if after_base > before_base {
        return false; // Worse
    }

    // Tie on (divs_subs, total) - need to compare mul_inversions
    let before_inv = mul_unsorted_adjacent(ctx, before);
    let after_inv = mul_unsorted_adjacent(ctx, after);
    after_inv < before_inv
}

/// Count out-of-order adjacent pairs in Mul chains (right-associative).
///
/// For a chain `a * (b * (c * d))` with factors `[a, b, c, d]`:
/// - Counts how many pairs (f[i], f[i+1]) have compare_expr(f[i], f[i+1]) == Greater
///
/// This metric allows canonicalizing rewrites that only reorder Mul factors.
pub fn mul_unsorted_adjacent(ctx: &Context, root: ExprId) -> usize {
    use crate::ordering::compare_expr;
    use std::cmp::Ordering;
    use std::collections::HashSet;

    // Collect all Mul nodes and identify which are right-children of other Muls
    let mut mul_nodes: HashSet<ExprId> = HashSet::new();
    let mut mul_right_children: HashSet<ExprId> = HashSet::new();

    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Mul(l, r) => {
                mul_nodes.insert(id);
                if matches!(ctx.get(*r), Expr::Mul(..)) {
                    mul_right_children.insert(*r);
                }
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args),
            Expr::Matrix { data, .. } => stack.extend(data),
            _ => {}
        }
    }

    // Heads are Mul nodes that are NOT the right child of another Mul
    let heads: Vec<_> = mul_nodes.difference(&mul_right_children).copied().collect();

    let mut inversions = 0;

    for head in heads {
        // Linearize factors by following right-assoc pattern: a*(b*(c*d)) -> [a,b,c,d]
        let mut factors = Vec::new();
        let mut current = head;

        loop {
            if let Expr::Mul(l, r) = ctx.get(current).clone() {
                factors.push(l);
                if matches!(ctx.get(r), Expr::Mul(..)) {
                    current = r;
                } else {
                    factors.push(r);
                    break;
                }
            } else {
                factors.push(current);
                break;
            }
        }

        // Count adjacent inversions
        for i in 0..factors.len().saturating_sub(1) {
            if compare_expr(ctx, factors[i], factors[i + 1]) == Ordering::Greater {
                inversions += 1;
            }
        }
    }

    inversions
}

// ========== Numeric Evaluation ==========

/// Extract a rational constant from an expression, handling multiple representations.
///
/// Supports (all must be purely numeric - returns None if any variable/function present):
/// - `Number(n)` - direct rational
/// - `Div(a, b)` - fraction (recursive)
/// - `Neg(a)` - negation (recursive)
/// - `Mul(a, b)` - product (recursive)
/// - `Add(a, b)` - sum (recursive)
/// - `Sub(a, b)` - difference (recursive)
///
/// This is the canonical helper for numeric evaluation. Used by:
/// - `SemanticEqualityChecker::try_evaluate_numeric`
/// - `EvaluatePowerRule` for exponent matching
pub fn as_rational_const(ctx: &Context, expr: ExprId) -> Option<num_rational::BigRational> {
    use num_traits::Zero;

    match ctx.get(expr) {
        Expr::Number(n) => Some(n.clone()),

        Expr::Div(num, den) => {
            let n = as_rational_const(ctx, *num)?;
            let d = as_rational_const(ctx, *den)?;
            if !d.is_zero() {
                Some(n / d)
            } else {
                None
            }
        }

        Expr::Neg(inner) => {
            let val = as_rational_const(ctx, *inner)?;
            Some(-val)
        }

        Expr::Mul(l, r) => {
            let lv = as_rational_const(ctx, *l)?;
            let rv = as_rational_const(ctx, *r)?;
            Some(lv * rv)
        }

        Expr::Add(l, r) => {
            let lv = as_rational_const(ctx, *l)?;
            let rv = as_rational_const(ctx, *r)?;
            Some(lv + rv)
        }

        Expr::Sub(l, r) => {
            let lv = as_rational_const(ctx, *l)?;
            let rv = as_rational_const(ctx, *r)?;
            Some(lv - rv)
        }

        // Variables, Constants, Functions, Pow, Matrix -> not purely numeric
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn test_is_one() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let zero = ctx.num(0);
        let x = ctx.var("x");

        assert!(is_one(&ctx, one));
        assert!(!is_one(&ctx, two));
        assert!(!is_one(&ctx, zero));
        assert!(!is_one(&ctx, x));
    }

    #[test]
    fn test_is_zero() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let one = ctx.num(1);
        let x = ctx.var("x");

        assert!(is_zero(&ctx, zero));
        assert!(!is_zero(&ctx, one));
        assert!(!is_zero(&ctx, x));
    }

    #[test]
    fn test_is_negative() {
        let mut ctx = Context::new();
        let neg_one = ctx.num(-1);
        let one = ctx.num(1);
        let x = ctx.var("x");
        let neg_x = ctx.add(Expr::Neg(x));

        assert!(is_negative(&ctx, neg_one));
        assert!(!is_negative(&ctx, one));
        assert!(is_negative(&ctx, neg_x));
        assert!(!is_negative(&ctx, x));
    }

    #[test]
    fn test_get_integer() {
        let mut ctx = Context::new();
        let five = ctx.num(5);
        let half = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let x = ctx.var("x");

        assert_eq!(get_integer(&ctx, five), Some(5));
        assert_eq!(get_integer(&ctx, half), None); // Not an integer
        assert_eq!(get_integer(&ctx, x), None);
    }

    #[test]
    fn test_flatten_add() {
        let mut ctx = Context::new();
        let expr = parse("a + b + c", &mut ctx).unwrap();
        let mut terms = Vec::new();
        flatten_add(&ctx, expr, &mut terms);
        assert_eq!(terms.len(), 3);
    }

    #[test]
    fn test_flatten_add_sub_chain() {
        let mut ctx = Context::new();
        let expr = parse("a + b - c", &mut ctx).unwrap();
        let terms = flatten_add_sub_chain(&mut ctx, expr);
        // Should have 3 terms: a, b, Neg(c)
        assert_eq!(terms.len(), 3);
    }

    #[test]
    fn test_flatten_mul() {
        let mut ctx = Context::new();
        let expr = parse("a * b * c", &mut ctx).unwrap();
        let mut factors = Vec::new();
        flatten_mul(&ctx, expr, &mut factors);
        assert_eq!(factors.len(), 3);
    }

    #[test]
    fn test_flatten_mul_chain_with_neg() {
        let mut ctx = Context::new();
        let expr = parse("-a * b", &mut ctx).unwrap();
        let factors = flatten_mul_chain(&mut ctx, expr);
        // Should have factors including -1
        assert!(factors.len() >= 2);
    }

    #[test]
    fn test_get_variant_name() {
        let mut ctx = Context::new();
        let a = ctx.num(1);
        let b = ctx.num(2);

        // Test with actual expressions from context
        assert_eq!(get_variant_name(ctx.get(a)), "Number");

        let x = ctx.var("x");
        assert_eq!(get_variant_name(ctx.get(x)), "Variable");

        let sum = ctx.add(Expr::Add(a, b));
        assert_eq!(get_variant_name(ctx.get(sum)), "Add");
    }

    #[test]
    fn test_is_pi_over_n() {
        let mut ctx = Context::new();
        let pi_over_2 = build_pi_over_n(&mut ctx, 2);
        let pi_over_4 = build_pi_over_n(&mut ctx, 4);

        assert!(is_pi_over_n(&ctx, pi_over_2, 2));
        assert!(!is_pi_over_n(&ctx, pi_over_2, 4));
        assert!(is_pi_over_n(&ctx, pi_over_4, 4));
    }

    #[test]
    fn test_is_half() {
        let mut ctx = Context::new();
        let half = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let one = ctx.num(1);

        assert!(is_half(&ctx, half));
        assert!(!is_half(&ctx, one));
    }

    #[test]
    fn test_is_pi() {
        let mut ctx = Context::new();
        let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let x = ctx.var("x");

        assert!(is_pi(&ctx, pi));
        assert!(!is_pi(&ctx, e));
        assert!(!is_pi(&ctx, x));
    }

    #[test]
    fn test_as_rational_const_number() {
        let mut ctx = Context::new();
        let half = ctx.rational(1, 2);
        let result = as_rational_const(&ctx, half);
        assert_eq!(
            result,
            Some(num_rational::BigRational::new(1.into(), 2.into()))
        );
    }

    #[test]
    fn test_as_rational_const_div() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let div = ctx.add(Expr::Div(one, two));
        let result = as_rational_const(&ctx, div);
        assert_eq!(
            result,
            Some(num_rational::BigRational::new(1.into(), 2.into()))
        );
    }

    #[test]
    fn test_as_rational_const_neg_div() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let div = ctx.add(Expr::Div(one, two));
        let neg = ctx.add(Expr::Neg(div));
        let result = as_rational_const(&ctx, neg);
        assert_eq!(
            result,
            Some(num_rational::BigRational::new((-1).into(), 2.into()))
        );
    }

    #[test]
    fn test_as_rational_const_variable_returns_none() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        assert!(as_rational_const(&ctx, x).is_none());
    }

    #[test]
    fn test_as_rational_const_mul_with_variable_returns_none() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let x = ctx.var("x");
        let mul = ctx.add(Expr::Mul(two, x));
        assert!(as_rational_const(&ctx, mul).is_none());
    }
}
