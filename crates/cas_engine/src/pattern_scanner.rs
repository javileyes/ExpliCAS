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
}
