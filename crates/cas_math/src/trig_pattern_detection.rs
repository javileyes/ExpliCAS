#![allow(dead_code)]
use crate::expr_destructure::{as_mul, as_neg};
use crate::expr_predicates::is_two_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use std::cmp::Ordering;

/// Check if two expressions are semantically equal
fn args_equal(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    cas_ast::ordering::compare_expr(ctx, a, b) == Ordering::Equal
}

/// Collect top-level additive terms from an Add chain.
/// Only flattens Add nodes; Sub, Neg, etc. are treated as leaf terms.
pub fn collect_add_chain(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Add(a, b) => {
            collect_add_chain(ctx, *a, terms);
            collect_add_chain(ctx, *b, terms);
        }
        _ => terms.push(expr),
    }
}

/// Check if all top-level additive terms of `expr` are negative, and if so,
/// return the fully negated expression.
///
/// Examples:
/// - `-2*u - 3` -> `Some(2*u + 3)`
/// - `-u + 3` -> `None`
pub fn try_extract_all_negative_sum(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Option<BigRational>)> {
    let mut terms: Vec<ExprId> = Vec::new();
    collect_add_chain(ctx, expr, &mut terms);

    if terms.len() < 2 {
        return None;
    }

    let zero = BigRational::from_integer(0.into());
    let one = BigRational::from_integer(1.into());
    let mut negated_terms: Vec<ExprId> = Vec::new();

    for &term in &terms {
        if let Some(inner) = as_neg(ctx, term) {
            negated_terms.push(inner);
            continue;
        }

        if let Expr::Number(n) = ctx.get(term) {
            if *n < zero {
                negated_terms.push(ctx.add(Expr::Number(-n.clone())));
                continue;
            }
            return None;
        }

        if let Some((a, b)) = as_mul(ctx, term) {
            if let Expr::Number(n) = ctx.get(a) {
                if *n < zero {
                    let abs_n = -n.clone();
                    if abs_n == one {
                        negated_terms.push(b);
                    } else {
                        let abs_expr = ctx.add(Expr::Number(abs_n));
                        negated_terms.push(ctx.add(Expr::Mul(abs_expr, b)));
                    }
                    continue;
                }
                return None;
            }

            if let Expr::Number(n) = ctx.get(b) {
                if *n < zero {
                    let abs_n = -n.clone();
                    if abs_n == one {
                        negated_terms.push(a);
                    } else {
                        let abs_expr = ctx.add(Expr::Number(abs_n));
                        negated_terms.push(ctx.add(Expr::Mul(a, abs_expr)));
                    }
                    continue;
                }
                return None;
            }
        }

        return None;
    }

    let mut result = negated_terms[0];
    for &term in &negated_terms[1..] {
        result = ctx.add(Expr::Add(result, term));
    }
    Some((result, None))
}

/// Check if expression is sec²(x), returns Some(x) if true
pub fn is_sec_squared(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            if is_two_expr(ctx, *exp) {
                match ctx.get(*base) {
                    Expr::Function(fn_id, args)
                        if ctx.is_builtin(*fn_id, BuiltinFn::Sec) && args.len() == 1 =>
                    {
                        Some(args[0])
                    }
                    _ => None,
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Check if expression is tan²(x), returns Some(x) if true
pub fn is_tan_squared(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            if is_two_expr(ctx, *exp) {
                match ctx.get(*base) {
                    Expr::Function(fn_id, args)
                        if ctx.is_builtin(*fn_id, BuiltinFn::Tan) && args.len() == 1 =>
                    {
                        Some(args[0])
                    }
                    _ => None,
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Check if expression is csc²(x), returns Some(x) if true
pub fn is_csc_squared(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            if is_two_expr(ctx, *exp) {
                match ctx.get(*base) {
                    Expr::Function(fn_id, args)
                        if ctx.is_builtin(*fn_id, BuiltinFn::Csc) && args.len() == 1 =>
                    {
                        Some(args[0])
                    }
                    _ => None,
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Check if expression is cot²(x), returns Some(x) if true
pub fn is_cot_squared(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            if is_two_expr(ctx, *exp) {
                match ctx.get(*base) {
                    Expr::Function(fn_id, args)
                        if ctx.is_builtin(*fn_id, BuiltinFn::Cot) && args.len() == 1 =>
                    {
                        Some(args[0])
                    }
                    _ => None,
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Detect sec²(x) - tan²(x) pattern
pub fn is_sec_tan_pattern(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Sub(left, right) = ctx.get(expr) {
        if let (Some(sec_arg), Some(tan_arg)) =
            (is_sec_squared(ctx, *left), is_tan_squared(ctx, *right))
        {
            return args_equal(ctx, sec_arg, tan_arg);
        }
    }
    false
}

/// Detect csc²(x) - cot²(x) pattern
pub fn is_csc_cot_pattern(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Sub(left, right) = ctx.get(expr) {
        if let (Some(csc_arg), Some(cot_arg)) =
            (is_csc_squared(ctx, *left), is_cot_squared(ctx, *right))
        {
            return args_equal(ctx, csc_arg, cot_arg);
        }
    }
    false
}

/// Detect (sec²-tan²) ± C or (csc²-cot²) ± C
pub fn is_pythagorean_with_constant(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Sub(left, _) | Expr::Add(left, _) => {
            is_sec_tan_pattern(ctx, *left) || is_csc_cot_pattern(ctx, *left)
        }
        _ => false,
    }
}

/// Check if given expr_id appears within a larger expression
fn expr_contains(ctx: &Context, haystack: ExprId, needle: ExprId) -> bool {
    if haystack == needle {
        return true;
    }

    match ctx.get(haystack) {
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            expr_contains(ctx, *l, needle) || expr_contains(ctx, *r, needle)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => expr_contains(ctx, *inner, needle),
        Expr::Function(_, args) => args.iter().any(|&arg| expr_contains(ctx, arg, needle)),
        Expr::Matrix { data, .. } => data.iter().any(|&elem| expr_contains(ctx, elem, needle)),
        // Leaves
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => false,
    }
}

/// Check if pattern_root is a Pythagorean pattern and contains target
fn matches_pattern_containing(ctx: &Context, pattern_root: ExprId, target: ExprId) -> bool {
    // Check if pattern_root is Pythagorean and contains target
    if is_sec_tan_pattern(ctx, pattern_root) || is_csc_cot_pattern(ctx, pattern_root) {
        return expr_contains(ctx, pattern_root, target);
    }

    if is_pythagorean_with_constant(ctx, pattern_root) {
        return expr_contains(ctx, pattern_root, target);
    }

    false
}

/// Check if given expr_id is part of a Pythagorean pattern in its ancestor chain
pub fn is_part_of_pythagorean_pattern(
    ctx: &Context,
    expr_id: ExprId,
    ancestors: &[ExprId],
) -> bool {
    ancestors
        .iter()
        .any(|&ancestor| matches_pattern_containing(ctx, ancestor, expr_id))
}

/// Smart guard for trig functions: checks if this function will be squared and used in Pythagorean pattern
/// This is specifically for protecting tan/sec/cot/csc from conversion when they're part of identities
pub fn should_preserve_trig_function(
    ctx: &Context,
    func_expr: ExprId,
    func_name: &str,
    ancestors: &[ExprId],
) -> bool {
    // Strategy:
    // 1. Find if this function appears in a Pow(this_func, 2) in ancestors
    // 2. If yes, check if THAT squared term appears in a Pythagorean pattern with its pair

    if ancestors.is_empty() {
        return false;
    }

    // println!(
    //     "  [DEBUG] should_preserve checking {} with {} ancestors",
    //     func_name,
    //     ancestors.len()
    // );

    // Step 1: Find Pow(this_func, 2) in ancestors
    let mut squared_id: Option<ExprId> = None;
    for &ancestor in ancestors {
        // println!(
        //     "    Checking ancestor {:?}: {:?}",
        //     ancestor,
        //     ctx.get(ancestor)
        // );
        if let Expr::Pow(base, exp) = ctx.get(ancestor) {
            if *base == func_expr && is_two_expr(ctx, *exp) {
                // println!("    -> Found squared! {:?}", ancestor);
                squared_id = Some(ancestor);
                break;
            }
        }
    }

    let squared_id = match squared_id {
        Some(id) => id,
        None => {
            // println!("    -> Not squared, allowing conversion");
            return false; // Not being squared
        }
    };

    // println!(
    //     "    Squared ID: {:?}, now checking for Pythagorean pattern...",
    //     squared_id
    // );

    // Step 2: Check if this squared term is in a Pythagorean pattern
    // Look through ALL ancestors to find Sub expressions that might contain our squared term
    for &ancestor in ancestors {
        if let Expr::Sub(left, right) = ctx.get(ancestor) {
            // println!(
            //     "      Found Sub ancestor: left={:?} right={:?}",
            //     left, right
            // );
            // println!("        left expr: {:?}", ctx.get(*left));
            // println!("        right expr: {:?}", ctx.get(*right));

            // Check if this Sub is a Pythagorean pattern and contains our squared term
            let left_id = *left;
            let right_id = *right;

            // Pattern 1: sec² - tan²
            if func_name == "tan" && right_id == squared_id {
                // println!("        Checking if left is sec²...");
                if is_sec_squared(ctx, left_id).is_some() {
                    // println!("        -> MATCH! Preserving tan");
                    return true;
                }
            }
            if func_name == "sec"
                && left_id == squared_id
                && is_tan_squared(ctx, right_id).is_some()
            {
                return true;
            }

            // Pattern 2: csc² - cot²
            if func_name == "cot"
                && right_id == squared_id
                && is_csc_squared(ctx, left_id).is_some()
            {
                return true;
            }
            if func_name == "csc"
                && left_id == squared_id
                && is_cot_squared(ctx, right_id).is_some()
            {
                return true;
            }
        }
    }

    // println!("    -> No Pythagorean pattern found");
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    fn build_context_with_sec_tan_minus_one() -> (Context, ExprId, ExprId, ExprId) {
        let mut ctx = Context::new();

        // Build: sec²(x) - tan²(x) - 1
        let x = ctx.var("x");
        let sec_x = ctx.call_builtin(cas_ast::BuiltinFn::Sec, vec![x]);
        let tan_x = ctx.call_builtin(cas_ast::BuiltinFn::Tan, vec![x]);
        let two = ctx.num(2);
        let sec_sq = ctx.add(Expr::Pow(sec_x, two));
        let tan_sq = ctx.add(Expr::Pow(tan_x, two));
        let sec_minus_tan = ctx.add(Expr::Sub(sec_sq, tan_sq));
        let one = ctx.num(1);
        let final_expr = ctx.add(Expr::Sub(sec_minus_tan, one));

        (ctx, final_expr, sec_sq, tan_sq)
    }

    #[test]
    fn test_is_sec_squared() {
        let (ctx, _, sec_sq, _) = build_context_with_sec_tan_minus_one();
        assert!(is_sec_squared(&ctx, sec_sq).is_some());
    }

    #[test]
    fn test_try_extract_all_negative_sum_extracts_expected_expression() {
        let mut ctx = Context::new();
        let u = ctx.var("u");
        let neg_two = ctx.num(-2);
        let neg_three = ctx.num(-3);
        let neg_two_u = ctx.add(Expr::Mul(neg_two, u));
        let expr = ctx.add(Expr::Add(neg_two_u, neg_three));
        let (extracted, _) =
            try_extract_all_negative_sum(&mut ctx, expr).expect("should extract negative sum");
        let expected = parse("2*u + 3", &mut ctx).expect("expected");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, extracted, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn test_try_extract_all_negative_sum_rejects_mixed_sign_sum() {
        let mut ctx = Context::new();
        let expr = parse("-u + 3", &mut ctx).expect("expr");
        assert!(try_extract_all_negative_sum(&mut ctx, expr).is_none());
    }

    #[test]
    fn test_is_tan_squared() {
        let (ctx, _, _, tan_sq) = build_context_with_sec_tan_minus_one();
        assert!(is_tan_squared(&ctx, tan_sq).is_some());
    }

    #[test]
    fn test_is_sec_tan_pattern() {
        let mut ctx = Context::new();

        // Build: sec²(x) - tan²(x)
        let x = ctx.var("x");
        let sec_x = ctx.call_builtin(cas_ast::BuiltinFn::Sec, vec![x]);
        let tan_x = ctx.call_builtin(cas_ast::BuiltinFn::Tan, vec![x]);
        let two = ctx.num(2);
        let sec_sq = ctx.add(Expr::Pow(sec_x, two));
        let tan_sq = ctx.add(Expr::Pow(tan_x, two));
        let pattern = ctx.add(Expr::Sub(sec_sq, tan_sq));

        assert!(is_sec_tan_pattern(&ctx, pattern));
    }

    #[test]
    fn test_is_pythagorean_with_constant() {
        let (ctx, final_expr, _, _) = build_context_with_sec_tan_minus_one();
        assert!(is_pythagorean_with_constant(&ctx, final_expr));
    }

    #[test]
    fn test_is_part_of_pythagorean_pattern() {
        let (ctx, final_expr, _, tan_sq) = build_context_with_sec_tan_minus_one();

        // tan_sq should be detected as part of Pythagorean pattern
        // when final_expr is in its ancestors
        let ancestors = vec![final_expr];
        assert!(is_part_of_pythagorean_pattern(&ctx, tan_sq, &ancestors));
    }
}
