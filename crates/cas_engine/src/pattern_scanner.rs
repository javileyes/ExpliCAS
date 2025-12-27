use crate::pattern_detection::{is_cot_squared, is_csc_squared, is_sec_squared, is_tan_squared};
use crate::pattern_marks::PatternMarks;
use cas_ast::{Context, Expr, ExprId};

/// Scan the entire expression tree and mark ExprIds that are part of Pythagorean patterns.
/// This pre-analysis pass allows guards to make correct decisions even with bottom-up processing.
pub fn scan_and_mark_patterns(ctx: &Context, root: ExprId, marks: &mut PatternMarks) {
    scan_recursive(ctx, root, marks);
}

fn scan_recursive(ctx: &Context, expr_id: ExprId, marks: &mut PatternMarks) {
    // First, recursively scan all children
    match ctx.get(expr_id) {
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            scan_recursive(ctx, *l, marks);
            scan_recursive(ctx, *r, marks);
        }
        Expr::Neg(inner) => {
            scan_recursive(ctx, *inner, marks);
        }
        Expr::Function(_, args) => {
            for &arg in args {
                scan_recursive(ctx, arg, marks);
            }
        }
        Expr::Matrix { data, .. } => {
            for &elem in data {
                scan_recursive(ctx, elem, marks);
            }
        }
        _ => {}
    }

    // After scanning children, check if THIS node is a special pattern
    check_and_mark_pythagorean_pattern(ctx, expr_id, marks);
    check_and_mark_sqrt_square_pattern(ctx, expr_id, marks);
    check_and_mark_trig_square_pattern(ctx, expr_id, marks);
    check_and_mark_inverse_trig_pattern(ctx, expr_id, marks);
}

fn check_and_mark_pythagorean_pattern(ctx: &Context, expr_id: ExprId, marks: &mut PatternMarks) {
    // Check if this is a Sub expression with Pythagorean pattern
    if let Expr::Sub(left, right) = ctx.get(expr_id) {
        let left_id = *left;
        let right_id = *right;

        // Pattern 1: sec²(x) - tan²(x)
        if let (Some(sec_arg), Some(tan_arg)) =
            (is_sec_squared(ctx, left_id), is_tan_squared(ctx, right_id))
        {
            // Check that arguments match
            if crate::ordering::compare_expr(ctx, sec_arg, tan_arg) == std::cmp::Ordering::Equal {
                // Mark the tan function inside tan²
                if let Expr::Pow(tan_base, _) = ctx.get(right_id) {
                    marks.mark_pythagorean(*tan_base);
                }
                // Mark the sec function inside sec²
                if let Expr::Pow(sec_base, _) = ctx.get(left_id) {
                    marks.mark_pythagorean(*sec_base);
                }
            }
        }

        // Pattern 2: csc²(x) - cot²(x)
        if let (Some(csc_arg), Some(cot_arg)) =
            (is_csc_squared(ctx, left_id), is_cot_squared(ctx, right_id))
        {
            // Check that arguments match
            if crate::ordering::compare_expr(ctx, csc_arg, cot_arg) == std::cmp::Ordering::Equal {
                // Mark the cot function inside cot²
                if let Expr::Pow(cot_base, _) = ctx.get(right_id) {
                    marks.mark_pythagorean(*cot_base);
                }
                // Mark the csc function inside csc²
                if let Expr::Pow(csc_base, _) = ctx.get(left_id) {
                    marks.mark_pythagorean(*csc_base);
                }
            }
        }
    }

    // Also check Add patterns like (sec²-tan²) + C or (sec²-tan²) - C
    if let Expr::Add(left, _) | Expr::Sub(left, _) = ctx.get(expr_id) {
        // Recursively check if left is a Sub with Pythagorean pattern
        check_and_mark_pythagorean_pattern(ctx, *left, marks);
    }
}

/// Detect sqrt(u²) or sqrt(u*u) patterns and mark the base for protection.
/// This prevents BinomialExpansionRule from expanding the squared expression,
/// allowing the sqrt(u²) → |u| shortcut to fire instead.
fn check_and_mark_sqrt_square_pattern(ctx: &Context, expr_id: ExprId, marks: &mut PatternMarks) {
    // Check for Pow(base, 1/2) where base is u² or u*u
    if let Expr::Pow(base, exp) = ctx.get(expr_id) {
        // Check if exponent is 1/2
        if crate::helpers::is_half(ctx, *exp) {
            let base_id = *base;
            let base_expr = ctx.get(base_id);

            // Case 1: base is Pow(u, 2) → mark the whole base (the u² expression)
            if let Expr::Pow(_, inner_exp) = base_expr {
                if let Expr::Number(n) = ctx.get(*inner_exp) {
                    if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into()) {
                        marks.mark_sqrt_square(base_id);
                        return;
                    }
                }
            }

            // Case 2: base is Mul(u, u) → mark the whole base (the u*u expression)
            if let Expr::Mul(left, right) = base_expr {
                if crate::ordering::compare_expr(ctx, *left, *right) == std::cmp::Ordering::Equal {
                    marks.mark_sqrt_square(base_id);
                    return;
                }
            }
        }
    }

    // Also check for sqrt(base) function form
    if let Expr::Function(name, args) = ctx.get(expr_id) {
        if name == "sqrt" && args.len() == 1 {
            let base_id = args[0];
            let base_expr = ctx.get(base_id);

            // Case 1: base is Pow(u, 2)
            if let Expr::Pow(_, inner_exp) = base_expr {
                if let Expr::Number(n) = ctx.get(*inner_exp) {
                    if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into()) {
                        marks.mark_sqrt_square(base_id);
                        return;
                    }
                }
            }

            // Case 2: base is Mul(u, u)
            if let Expr::Mul(left, right) = base_expr {
                if crate::ordering::compare_expr(ctx, *left, *right) == std::cmp::Ordering::Equal {
                    marks.mark_sqrt_square(base_id);
                }
            }
        }
    }
}

/// Detect sin²(u) + cos²(v) patterns where u ≡ v (using commutative compare_expr)
/// Mark the sin/cos Function nodes so AngleIdentityRule won't expand them
fn check_and_mark_trig_square_pattern(ctx: &Context, expr_id: ExprId, marks: &mut PatternMarks) {
    // Only check Add patterns (sin² + cos²)
    if !matches!(ctx.get(expr_id), Expr::Add(_, _)) {
        return;
    }

    // Flatten the Add to get all terms
    let mut terms = Vec::new();
    crate::helpers::flatten_add(ctx, expr_id, &mut terms);

    // Extract (func_name, func_id, arg_id) from terms that are sin²/cos²
    // Returns: ("sin"|"cos", function ExprId, argument ExprId)
    let extract_trig_square =
        |ctx: &Context, term: ExprId| -> Option<(&'static str, ExprId, ExprId)> {
            // Match Pow(Function("sin"|"cos", [arg]), 2)
            if let Expr::Pow(base, exp) = ctx.get(term) {
                // Check exponent is 2
                if let Expr::Number(n) = ctx.get(*exp) {
                    if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into()) {
                        // Check base is sin or cos
                        if let Expr::Function(name, args) = ctx.get(*base) {
                            if args.len() == 1 {
                                match name.as_str() {
                                    "sin" => return Some(("sin", *base, args[0])),
                                    "cos" => return Some(("cos", *base, args[0])),
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
            None
        };

    // Collect all sin² and cos² terms with their info
    let mut sin_terms: Vec<(ExprId, ExprId)> = Vec::new(); // (func_id, arg_id)
    let mut cos_terms: Vec<(ExprId, ExprId)> = Vec::new();

    for &term in &terms {
        if let Some((name, func_id, arg_id)) = extract_trig_square(ctx, term) {
            match name {
                "sin" => sin_terms.push((func_id, arg_id)),
                "cos" => cos_terms.push((func_id, arg_id)),
                _ => {}
            }
        }
    }

    // Find pairs with matching arguments (using commutative compare_expr)
    for (sin_func_id, sin_arg) in &sin_terms {
        for (cos_func_id, cos_arg) in &cos_terms {
            if crate::ordering::compare_expr(ctx, *sin_arg, *cos_arg) == std::cmp::Ordering::Equal {
                // Found a match! Mark both functions for protection
                marks.mark_trig_square(*sin_func_id);
                marks.mark_trig_square(*cos_func_id);
            }
        }
    }
}

/// Detect arctan(tan(x)), arcsin(sin(x)), arccos(cos(x)) patterns.
/// Mark the inner trig function (tan, sin, cos) for protection so it won't
/// be converted to sin/cos before the inverse-trig simplification can fire.
fn check_and_mark_inverse_trig_pattern(ctx: &Context, expr_id: ExprId, marks: &mut PatternMarks) {
    // Only check Function nodes
    if let Expr::Function(outer_name, outer_args) = ctx.get(expr_id) {
        if outer_args.len() != 1 {
            return;
        }
        let inner_id = outer_args[0];

        // Check if outer is an inverse trig function
        let inverse_inner_pair = match outer_name.as_str() {
            "arctan" | "atan" => Some("tan"),
            "arcsin" | "asin" => Some("sin"),
            "arccos" | "acos" => Some("cos"),
            _ => None,
        };

        if let Some(expected_inner_name) = inverse_inner_pair {
            // Check if inner is the matching trig function
            if let Expr::Function(inner_name, inner_args) = ctx.get(inner_id) {
                if inner_name == expected_inner_name && inner_args.len() == 1 {
                    // Found arctan(tan(u)) or similar - mark the inner function
                    marks.mark_inverse_trig(inner_id);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_marks_sec_tan_pattern() {
        let mut ctx = Context::new();
        let mut marks = PatternMarks::new();

        // Build: sec²(x) - tan²(x)
        let x = ctx.var("x");
        let sec_x = ctx.add(Expr::Function("sec".into(), vec![x]));
        let tan_x = ctx.add(Expr::Function("tan".into(), vec![x]));
        let two = ctx.num(2);
        let sec_sq = ctx.add(Expr::Pow(sec_x, two));
        let tan_sq = ctx.add(Expr::Pow(tan_x, two));
        let pattern = ctx.add(Expr::Sub(sec_sq, tan_sq));

        scan_and_mark_patterns(&ctx, pattern, &mut marks);

        // Both sec(x) and tan(x) should be marked
        assert!(marks.is_pythagorean_protected(sec_x));
        assert!(marks.is_pythagorean_protected(tan_x));
    }

    #[test]
    fn test_scan_marks_with_constant() {
        let mut ctx = Context::new();
        let mut marks = PatternMarks::new();

        // Build: sec²(x) - tan²(x) - 1
        let x = ctx.var("x");
        let sec_x = ctx.add(Expr::Function("sec".into(), vec![x]));
        let tan_x = ctx.add(Expr::Function("tan".into(), vec![x]));
        let two = ctx.num(2);
        let sec_sq = ctx.add(Expr::Pow(sec_x, two));
        let tan_sq = ctx.add(Expr::Pow(tan_x, two));
        let sec_minus_tan = ctx.add(Expr::Sub(sec_sq, tan_sq));
        let one = ctx.num(1);
        let final_expr = ctx.add(Expr::Sub(sec_minus_tan, one));

        scan_and_mark_patterns(&ctx, final_expr, &mut marks);

        // Both sec(x) and tan(x) should be marked
        assert!(marks.is_pythagorean_protected(sec_x));
        assert!(marks.is_pythagorean_protected(tan_x));
    }

    #[test]
    fn test_scan_marks_trig_square_simple() {
        let mut ctx = Context::new();
        let mut marks = PatternMarks::new();

        // Build: sin(x)^2 + cos(x)^2
        let x = ctx.var("x");
        let sin_x = ctx.add(Expr::Function("sin".into(), vec![x]));
        let cos_x = ctx.add(Expr::Function("cos".into(), vec![x]));
        let two = ctx.num(2);
        let sin_sq = ctx.add(Expr::Pow(sin_x, two));
        let cos_sq = ctx.add(Expr::Pow(cos_x, two));
        let pattern = ctx.add(Expr::Add(sin_sq, cos_sq));

        scan_and_mark_patterns(&ctx, pattern, &mut marks);

        // Both sin(x) and cos(x) should be marked
        assert!(
            marks.is_trig_square_protected(sin_x),
            "sin(x) should be protected"
        );
        assert!(
            marks.is_trig_square_protected(cos_x),
            "cos(x) should be protected"
        );
    }

    #[test]
    fn test_scan_marks_trig_square_commuted_args() {
        let mut ctx = Context::new();
        let mut marks = PatternMarks::new();

        // Build: sin(2*x + 1)^2 + cos(1 + 2*x)^2
        // Args are commuted: 2*x + 1 vs 1 + 2*x
        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);

        // sin arg: 2*x + 1  => Add(Mul(2, x), 1)
        let two_x = ctx.add(Expr::Mul(two, x));
        let sin_arg = ctx.add(Expr::Add(two_x, one));
        let sin_func = ctx.add(Expr::Function("sin".into(), vec![sin_arg]));
        let sin_sq = ctx.add(Expr::Pow(sin_func, two));

        // cos arg: 1 + 2*x  => Add(1, Mul(2, x))
        let two_2 = ctx.num(2);
        let one_2 = ctx.num(1);
        let two_x_2 = ctx.add(Expr::Mul(two_2, x));
        let cos_arg = ctx.add(Expr::Add(one_2, two_x_2));
        let cos_func = ctx.add(Expr::Function("cos".into(), vec![cos_arg]));
        let two_3 = ctx.num(2);
        let cos_sq = ctx.add(Expr::Pow(cos_func, two_3));

        let pattern = ctx.add(Expr::Add(sin_sq, cos_sq));

        scan_and_mark_patterns(&ctx, pattern, &mut marks);

        // Both should be marked because compare_expr handles Add commutativity
        assert!(
            marks.is_trig_square_protected(sin_func),
            "sin(2*x+1) should be protected"
        );
        assert!(
            marks.is_trig_square_protected(cos_func),
            "cos(1+2*x) should be protected"
        );
    }

    #[test]
    fn test_scan_marks_inverse_trig_pattern() {
        let mut ctx = Context::new();
        let mut marks = PatternMarks::new();

        // Build: arctan(tan(x))
        let x = ctx.var("x");
        let tan_x = ctx.add(Expr::Function("tan".into(), vec![x]));
        let arctan_tan_x = ctx.add(Expr::Function("arctan".into(), vec![tan_x]));

        scan_and_mark_patterns(&ctx, arctan_tan_x, &mut marks);

        // tan(x) should be marked for protection
        assert!(
            marks.is_inverse_trig_protected(tan_x),
            "tan(x) in arctan(tan(x)) should be protected"
        );
    }

    #[test]
    fn test_scan_marks_arcsin_sin_pattern() {
        let mut ctx = Context::new();
        let mut marks = PatternMarks::new();

        // Build: arcsin(sin(x))
        let x = ctx.var("x");
        let sin_x = ctx.add(Expr::Function("sin".into(), vec![x]));
        let arcsin_sin_x = ctx.add(Expr::Function("arcsin".into(), vec![sin_x]));

        scan_and_mark_patterns(&ctx, arcsin_sin_x, &mut marks);

        // sin(x) should be marked for protection
        assert!(
            marks.is_inverse_trig_protected(sin_x),
            "sin(x) in arcsin(sin(x)) should be protected"
        );
    }
}
