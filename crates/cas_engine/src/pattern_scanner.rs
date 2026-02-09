use crate::pattern_detection::{is_cot_squared, is_csc_squared, is_sec_squared, is_tan_squared};
use crate::pattern_marks::PatternMarks;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};

/// Scan the entire expression tree and mark ExprIds that are part of Pythagorean patterns.
/// This pre-analysis pass allows guards to make correct decisions even with bottom-up processing.
pub fn scan_and_mark_patterns(ctx: &Context, root: ExprId, marks: &mut PatternMarks) {
    scan_recursive(ctx, root, marks);
}

/// Iterative post-order traversal using explicit stack.
/// Visits all children before parent, then marks parent patterns.
fn scan_recursive(ctx: &Context, root: ExprId, marks: &mut PatternMarks) {
    // Two stacks: work for traversal, process for post-order marking
    let mut work_stack = vec![root];
    let mut process_stack = Vec::new();

    // First pass: collect all nodes for post-order processing
    while let Some(expr_id) = work_stack.pop() {
        process_stack.push(expr_id);

        match ctx.get(expr_id) {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                work_stack.push(*l);
                work_stack.push(*r);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => {
                work_stack.push(*inner);
            }
            Expr::Function(_, args) => {
                for &arg in args {
                    work_stack.push(arg);
                }
            }
            Expr::Matrix { data, .. } => {
                for &elem in data {
                    work_stack.push(elem);
                }
            }
            _ => {}
        }
    }

    // Second pass: process in reverse (post-order) to check patterns
    while let Some(expr_id) = process_stack.pop() {
        check_and_mark_pythagorean_pattern(ctx, expr_id, marks);
        check_and_mark_sqrt_square_pattern(ctx, expr_id, marks);
        check_and_mark_trig_square_pattern(ctx, expr_id, marks);
        check_and_mark_inverse_trig_pattern(ctx, expr_id, marks);
        check_and_mark_sum_quotient_pattern(ctx, expr_id, marks);
        check_and_mark_tan_triple_product_pattern(ctx, expr_id, marks);
        check_and_mark_tan_difference_identity_pattern(ctx, expr_id, marks);
        check_and_mark_sin4x_identity_pattern(ctx, expr_id, marks);
    }
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
    if let Expr::Function(fn_id, args) = ctx.get(expr_id) {
        if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 {
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
                        if let Expr::Function(fn_id, args) = ctx.get(*base) {
                            if args.len() == 1 {
                                match ctx.sym_name(*fn_id) {
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
        let inverse_inner_pair = match ctx.sym_name(*outer_name) {
            "arctan" | "atan" => Some("tan"),
            "arcsin" | "asin" => Some("sin"),
            "arccos" | "acos" => Some("cos"),
            _ => None,
        };

        if let Some(expected_inner_name) = inverse_inner_pair {
            // Check if inner is the matching trig function
            if let Expr::Function(inner_name, inner_args) = ctx.get(inner_id) {
                if ctx.sym_name(*inner_name) == expected_inner_name && inner_args.len() == 1 {
                    // Found arctan(tan(u)) or similar - mark the inner function
                    marks.mark_inverse_trig(inner_id);
                }
            }
        }
    }
}

/// Detect (sin(A)+sin(B))/(cos(C)+cos(D)) patterns where {A,B} == {C,D}.
/// Mark all sin/cos function nodes for protection so TripleAngleRule won't expand them.
/// This allows SinCosSumQuotientRule to fire and apply sum-to-product identities.
fn check_and_mark_sum_quotient_pattern(ctx: &Context, expr_id: ExprId, marks: &mut PatternMarks) {
    // Only check Div nodes
    let Expr::Div(num_id, den_id) = ctx.get(expr_id) else {
        return;
    };
    let num_id = *num_id;
    let den_id = *den_id;

    // Helper: extract sin/cos function nodes from a 2-term sum
    fn extract_trig_sum_nodes(
        ctx: &Context,
        expr: ExprId,
        fn_name: &str,
    ) -> Option<(ExprId, ExprId, ExprId, ExprId)> {
        let mut terms = Vec::new();
        crate::helpers::flatten_add(ctx, expr, &mut terms);

        if terms.len() != 2 {
            return None;
        }

        // Extract function nodes and their args
        let (func_id1, arg1) = extract_trig_func_and_arg(ctx, terms[0], fn_name)?;
        let (func_id2, arg2) = extract_trig_func_and_arg(ctx, terms[1], fn_name)?;

        Some((func_id1, arg1, func_id2, arg2))
    }

    fn extract_trig_func_and_arg(
        ctx: &Context,
        id: ExprId,
        fn_name: &str,
    ) -> Option<(ExprId, ExprId)> {
        if let Expr::Function(fn_id, args) = ctx.get(id) {
            let name = ctx.sym_name(*fn_id);
            if name == fn_name && args.len() == 1 {
                return Some((id, args[0]));
            }
        }
        None
    }

    // Try to extract sin(A) + sin(B) from numerator
    let Some((sin_func1, sin_a, sin_func2, sin_b)) = extract_trig_sum_nodes(ctx, num_id, "sin")
    else {
        return;
    };

    // Try to extract cos(C) + cos(D) from denominator
    let Some((cos_func1, cos_c, cos_func2, cos_d)) = extract_trig_sum_nodes(ctx, den_id, "cos")
    else {
        return;
    };

    // Check if {A,B} == {C,D} as multisets
    use std::cmp::Ordering;
    let direct = crate::ordering::compare_expr(ctx, sin_a, cos_c) == Ordering::Equal
        && crate::ordering::compare_expr(ctx, sin_b, cos_d) == Ordering::Equal;
    let crossed = crate::ordering::compare_expr(ctx, sin_a, cos_d) == Ordering::Equal
        && crate::ordering::compare_expr(ctx, sin_b, cos_c) == Ordering::Equal;

    if direct || crossed {
        // Mark all four function nodes for protection
        marks.mark_sum_quotient(sin_func1);
        marks.mark_sum_quotient(sin_func2);
        marks.mark_sum_quotient(cos_func1);
        marks.mark_sum_quotient(cos_func2);
    }
}

/// Detect tan(u)·tan(π/3+u)·tan(π/3-u) patterns.
/// Mark all tan() function nodes for protection so TanToSinCosRule won't expand them,
/// allowing TanTripleProductRule to fire.
fn check_and_mark_tan_triple_product_pattern(
    ctx: &Context,
    expr_id: ExprId,
    marks: &mut PatternMarks,
) {
    // Only check Mul nodes
    if !matches!(ctx.get(expr_id), Expr::Mul(_, _)) {
        return;
    }

    // Flatten multiplication to get factors
    // Note: This needs mutable ctx, but we only have immutable here.
    // We'll use a simpler check instead.
    let mut factors = Vec::new();
    let mut stack = vec![expr_id];
    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Mul(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            _ => factors.push(id),
        }
    }

    // Extract all tan(arg) functions
    let mut tan_nodes: Vec<(ExprId, ExprId)> = Vec::new(); // (func_id, arg)
    for &factor in &factors {
        if let Expr::Function(fn_id, args) = ctx.get(factor) {
            let name = ctx.sym_name(*fn_id);
            if name == "tan" && args.len() == 1 {
                tan_nodes.push((factor, args[0]));
            }
        }
    }

    // Need exactly 3 tan factors
    if tan_nodes.len() != 3 {
        return;
    }

    // Try each argument as the potential "u"
    for i in 0..3 {
        let u = tan_nodes[i].1;
        let (j, k) = match i {
            0 => (1, 2),
            1 => (0, 2),
            2 => (0, 1),
            _ => unreachable!(),
        };

        let arg_j = tan_nodes[j].1;
        let arg_k = tan_nodes[k].1;

        // Check if the set forms {u, u+π/3, π/3-u}
        let match1 =
            is_u_plus_pi_over_3_check(ctx, arg_j, u) && is_pi_over_3_minus_u_check(ctx, arg_k, u);
        let match2 =
            is_pi_over_3_minus_u_check(ctx, arg_j, u) && is_u_plus_pi_over_3_check(ctx, arg_k, u);

        if match1 || match2 {
            // Mark all 3 tan nodes for protection
            for (tan_node, _) in &tan_nodes {
                marks.mark_tan_triple_product(*tan_node);
            }
            return;
        }
    }
}

/// Helper: Check if expr equals u + π/3 (or π/3 + u)
fn is_u_plus_pi_over_3_check(ctx: &Context, expr: ExprId, u: ExprId) -> bool {
    if let Expr::Add(l, r) = ctx.get(expr).clone() {
        // Case: u + π/3
        if crate::ordering::compare_expr(ctx, l, u) == std::cmp::Ordering::Equal {
            return is_pi_over_3_check(ctx, r);
        }
        // Case: π/3 + u
        if crate::ordering::compare_expr(ctx, r, u) == std::cmp::Ordering::Equal {
            return is_pi_over_3_check(ctx, l);
        }
    }
    false
}

/// Helper: Check if expr equals π/3 - u (or -u + π/3 in canonicalized form)
fn is_pi_over_3_minus_u_check(ctx: &Context, expr: ExprId, u: ExprId) -> bool {
    // Pattern 1: Sub(π/3, u)
    if let Expr::Sub(l, r) = ctx.get(expr).clone() {
        if is_pi_over_3_check(ctx, l)
            && crate::ordering::compare_expr(ctx, r, u) == std::cmp::Ordering::Equal
        {
            return true;
        }
    }
    // Pattern 2: Add(π/3, Neg(u)) or Add(Neg(u), π/3) - canonicalized subtraction
    if let Expr::Add(l, r) = ctx.get(expr).clone() {
        // Add(π/3, Neg(u))
        if is_pi_over_3_check(ctx, l) {
            if let Expr::Neg(inner) = ctx.get(r).clone() {
                if crate::ordering::compare_expr(ctx, inner, u) == std::cmp::Ordering::Equal {
                    return true;
                }
            }
        }
        // Add(Neg(u), π/3)
        if is_pi_over_3_check(ctx, r) {
            if let Expr::Neg(inner) = ctx.get(l).clone() {
                if crate::ordering::compare_expr(ctx, inner, u) == std::cmp::Ordering::Equal {
                    return true;
                }
            }
        }
    }
    false
}

/// Helper: Check if an expression is π/3
fn is_pi_over_3_check(ctx: &Context, expr: ExprId) -> bool {
    // Pattern 1: Div(π, 3)
    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        if matches!(ctx.get(num), Expr::Constant(cas_ast::Constant::Pi)) {
            if let Expr::Number(n) = ctx.get(den) {
                if n.is_integer() && *n.numer() == 3.into() {
                    return true;
                }
            }
        }
    }

    // Pattern 2: Mul(Number(1/3), π) - canonicalized form from CanonicalizeDivRule
    if let Expr::Mul(l, r) = ctx.get(expr).clone() {
        // Check Mul(1/3, π)
        if let Expr::Number(n) = ctx.get(l) {
            if *n == num_rational::BigRational::new(1.into(), 3.into())
                && matches!(ctx.get(r), Expr::Constant(cas_ast::Constant::Pi))
            {
                return true;
            }
        }
        // Check Mul(π, 1/3)
        if let Expr::Number(n) = ctx.get(r) {
            if *n == num_rational::BigRational::new(1.into(), 3.into())
                && matches!(ctx.get(l), Expr::Constant(cas_ast::Constant::Pi))
            {
                return true;
            }
        }
    }

    false
}

/// Detect tan(a-b) - (tan(a)-tan(b))/(1+tan(a)*tan(b)) patterns.
/// Mark all tan() function nodes for protection so TanToSinCosRule won't expand them,
/// allowing TanDifferenceIdentityZeroRule to fire.
fn check_and_mark_tan_difference_identity_pattern(
    ctx: &Context,
    expr_id: ExprId,
    marks: &mut PatternMarks,
) {
    // Only check Sub nodes or Add with negation (a + (-b) form)
    let (left, right) = match ctx.get(expr_id) {
        Expr::Sub(l, r) => (*l, *r),
        Expr::Add(l, r) => {
            // Check for Add(a, Neg(b)) form
            if let Expr::Neg(inner) = ctx.get(*r) {
                (*l, *inner)
            } else if let Expr::Neg(inner) = ctx.get(*l) {
                (*r, *inner)
            } else {
                return;
            }
        }
        _ => return,
    };

    // Try both orderings: tan(a-b) - RHS or RHS - tan(a-b)
    if try_match_tan_diff_identity(ctx, left, right, marks) {
        return;
    }
    try_match_tan_diff_identity(ctx, right, left, marks);
}

/// Try to match tan(a-b) on lhs and (tan(a)-tan(b))/(1+tan(a)*tan(b)) on rhs
fn try_match_tan_diff_identity(
    ctx: &Context,
    lhs: ExprId,
    rhs: ExprId,
    marks: &mut PatternMarks,
) -> bool {
    // LHS should be tan(a - b)
    let Expr::Function(fn_id, args) = ctx.get(lhs) else {
        return false;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Tan) || args.len() != 1 {
        return false;
    }
    let tan_arg = args[0];

    // Extract a and b from (a - b) or (a + (-b))
    let (a, b) = match ctx.get(tan_arg) {
        Expr::Sub(l, r) => (*l, *r),
        Expr::Add(l, r) => {
            if let Expr::Neg(inner) = ctx.get(*r) {
                (*l, *inner)
            } else if let Expr::Neg(inner) = ctx.get(*l) {
                (*r, *inner)
            } else {
                return false;
            }
        }
        _ => return false,
    };

    // RHS should be (tan(a) - tan(b)) / (1 + tan(a)*tan(b))
    let Expr::Div(num, den) = ctx.get(rhs) else {
        return false;
    };
    let num = *num;
    let den = *den;

    // Check numerator: tan(a) - tan(b)
    let (tan_a_num, tan_b_num) = match ctx.get(num) {
        Expr::Sub(l, r) => (*l, *r),
        Expr::Add(l, r) => {
            if let Expr::Neg(inner) = ctx.get(*r) {
                (*l, *inner)
            } else {
                return false;
            }
        }
        _ => return false,
    };

    // Verify tan_a_num is tan(a)
    if let Expr::Function(name_a, args_a) = ctx.get(tan_a_num) {
        if !ctx.is_builtin(*name_a, BuiltinFn::Tan) || args_a.len() != 1 {
            return false;
        }
        if crate::ordering::compare_expr(ctx, args_a[0], a) != std::cmp::Ordering::Equal {
            return false;
        }
    } else {
        return false;
    }

    // Verify tan_b_num is tan(b)
    if let Expr::Function(name_b, args_b) = ctx.get(tan_b_num) {
        if !ctx.is_builtin(*name_b, BuiltinFn::Tan) || args_b.len() != 1 {
            return false;
        }
        if crate::ordering::compare_expr(ctx, args_b[0], b) != std::cmp::Ordering::Equal {
            return false;
        }
    } else {
        return false;
    }

    // Check denominator: 1 + tan(a)*tan(b)
    if !matches_one_plus_tan_product(ctx, den, a, b) {
        return false;
    }

    // All matched! Set global flag to block ALL tan expansion
    marks.has_tan_identity_pattern = true;

    // Also mark individual nodes (for completeness)
    marks.mark_identity_cancellation(lhs); // tan(a-b)
    marks.mark_identity_cancellation(tan_a_num); // tan(a) in numerator
    marks.mark_identity_cancellation(tan_b_num); // tan(b) in numerator

    // Mark tan nodes in denominator product
    if let Expr::Add(add_l, add_r) = ctx.get(den) {
        let product_part = if let Expr::Number(n) = ctx.get(*add_l) {
            if *n == num_rational::BigRational::from_integer(1.into()) {
                *add_r
            } else {
                *add_l
            }
        } else if let Expr::Number(n) = ctx.get(*add_r) {
            if *n == num_rational::BigRational::from_integer(1.into()) {
                *add_l
            } else {
                *add_r
            }
        } else {
            return true; // Pattern matched, even if we can't mark all tan nodes
        };

        if let Expr::Mul(ml, mr) = ctx.get(product_part) {
            marks.mark_identity_cancellation(*ml);
            marks.mark_identity_cancellation(*mr);
        }
    }

    true
}

/// Helper to match 1 + tan(a)*tan(b)
fn matches_one_plus_tan_product(ctx: &Context, expr: ExprId, a: ExprId, b: ExprId) -> bool {
    let Expr::Add(l, r) = ctx.get(expr) else {
        return false;
    };
    let l = *l;
    let r = *r;

    let (one_part, product_part) = if let Expr::Number(n) = ctx.get(l) {
        if *n == num_rational::BigRational::from_integer(1.into()) {
            (l, r)
        } else {
            return false;
        }
    } else if let Expr::Number(n) = ctx.get(r) {
        if *n == num_rational::BigRational::from_integer(1.into()) {
            (r, l)
        } else {
            return false;
        }
    } else {
        return false;
    };
    let _ = one_part;

    // product_part should be tan(a)*tan(b)
    let Expr::Mul(ml, mr) = ctx.get(product_part) else {
        return false;
    };
    let ml = *ml;
    let mr = *mr;

    // Check if both are tan functions
    let (Expr::Function(n1, args1), Expr::Function(n2, args2)) = (ctx.get(ml), ctx.get(mr)) else {
        return false;
    };

    if !ctx.is_builtin(*n1, BuiltinFn::Tan)
        || !ctx.is_builtin(*n2, BuiltinFn::Tan)
        || args1.len() != 1
        || args2.len() != 1
    {
        return false;
    }

    let arg1 = args1[0];
    let arg2 = args2[0];

    // Check (arg1==a && arg2==b) or (arg1==b && arg2==a)
    let match1 = crate::ordering::compare_expr(ctx, arg1, a) == std::cmp::Ordering::Equal
        && crate::ordering::compare_expr(ctx, arg2, b) == std::cmp::Ordering::Equal;
    let match2 = crate::ordering::compare_expr(ctx, arg1, b) == std::cmp::Ordering::Equal
        && crate::ordering::compare_expr(ctx, arg2, a) == std::cmp::Ordering::Equal;

    match1 || match2
}

/// Detect sin(4t) - 4*sin(t)*cos(t)*(cos(t)^2-sin(t)^2) patterns.
/// Sets has_sin4x_identity_pattern to true so DistributeRule won't distribute,
/// allowing Sin4xIdentityZeroRule to fire.
fn check_and_mark_sin4x_identity_pattern(ctx: &Context, expr_id: ExprId, marks: &mut PatternMarks) {
    // Only check Sub nodes or Add with negation (a + (-b) form)
    let (left, right) = match ctx.get(expr_id) {
        Expr::Sub(l, r) => (*l, *r),
        Expr::Add(l, r) => {
            if let Expr::Neg(inner) = ctx.get(*r) {
                (*l, *inner)
            } else if let Expr::Neg(inner) = ctx.get(*l) {
                (*r, *inner)
            } else {
                return;
            }
        }
        _ => return,
    };

    // Try both orderings: sin(4t) - RHS or RHS - sin(4t)
    if try_match_sin4x_identity(ctx, left, right, marks) {
        return;
    }
    try_match_sin4x_identity(ctx, right, left, marks);
}

/// Try to match sin(4t) on lhs and 4*sin(t)*cos(t)*(cos²-sin²) on rhs
fn try_match_sin4x_identity(
    ctx: &Context,
    lhs: ExprId,
    rhs: ExprId,
    marks: &mut PatternMarks,
) -> bool {
    // LHS should be sin(4*t)
    let Expr::Function(fn_id, args) = ctx.get(lhs) else {
        return false;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Sin) || args.len() != 1 {
        return false;
    }
    let sin_arg = args[0];

    // Check if arg is 4*t
    let t = match ctx.get(sin_arg) {
        Expr::Mul(l, r) => {
            let l = *l;
            let r = *r;
            if let Expr::Number(n) = ctx.get(l) {
                if *n == num_rational::BigRational::from_integer(4.into()) {
                    r
                } else {
                    return false;
                }
            } else if let Expr::Number(n) = ctx.get(r) {
                if *n == num_rational::BigRational::from_integer(4.into()) {
                    l
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }
        _ => return false,
    };

    // RHS should be 4*sin(t)*cos(t)*(cos(t)^2 - sin(t)^2)
    // Flatten multiplication to get factors
    let mut factors = Vec::new();
    let mut stack = vec![rhs];
    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Mul(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            _ => factors.push(id),
        }
    }

    // Look for: 4, sin(t), cos(t), (cos(t)^2 - sin(t)^2) or cos(2t)
    let mut has_four = false;
    let mut has_sin_t = false;
    let mut has_cos_t = false;
    let mut has_diff_squares = false;

    for &factor in &factors {
        // Check for 4
        if let Expr::Number(n) = ctx.get(factor) {
            if *n == num_rational::BigRational::from_integer(4.into()) {
                has_four = true;
                continue;
            }
        }
        // Check for sin(t) or cos(t)
        if let Expr::Function(fn_name, fn_args) = ctx.get(factor) {
            if ctx.is_builtin(*fn_name, BuiltinFn::Sin)
                && fn_args.len() == 1
                && crate::ordering::compare_expr(ctx, fn_args[0], t) == std::cmp::Ordering::Equal
            {
                has_sin_t = true;
                continue;
            }
            if ctx.is_builtin(*fn_name, BuiltinFn::Cos) && fn_args.len() == 1 {
                let arg = fn_args[0];
                if crate::ordering::compare_expr(ctx, arg, t) == std::cmp::Ordering::Equal {
                    has_cos_t = true;
                    continue;
                }
                // Also check for cos(2t) which equals cos²-sin²
                if let Expr::Mul(cl, cr) = ctx.get(arg) {
                    let cl = *cl;
                    let cr = *cr;
                    let is_2t = if let Expr::Number(n) = ctx.get(cl) {
                        *n == num_rational::BigRational::from_integer(2.into())
                            && crate::ordering::compare_expr(ctx, cr, t)
                                == std::cmp::Ordering::Equal
                    } else if let Expr::Number(n) = ctx.get(cr) {
                        *n == num_rational::BigRational::from_integer(2.into())
                            && crate::ordering::compare_expr(ctx, cl, t)
                                == std::cmp::Ordering::Equal
                    } else {
                        false
                    };
                    if is_2t {
                        has_diff_squares = true;
                        continue;
                    }
                }
            }
        }
        // Check for cos²(t) - sin²(t) or equivalent forms
        if let Expr::Sub(sl, sr) = ctx.get(factor) {
            let sl = *sl;
            let sr = *sr;
            // Standard: cos²(t) - sin²(t)
            if is_cos_squared_t(ctx, sl, t) && is_sin_squared_t(ctx, sr, t) {
                has_diff_squares = true;
                continue;
            }
            // Alternate: 2·cos²(t) - 1 (also equals cos(2t))
            if is_2cos_sq_minus_1(ctx, sl, sr, t) {
                has_diff_squares = true;
                continue;
            }
            // Alternate: 1 - 2·sin²(t) (also equals cos(2t))
            if is_1_minus_2sin_sq(ctx, sl, sr, t) {
                has_diff_squares = true;
                continue;
            }
        }
    }

    if has_four && has_sin_t && has_cos_t && has_diff_squares {
        marks.has_sin4x_identity_pattern = true;
        return true;
    }
    false
}

fn is_sin_squared_t(ctx: &Context, expr: ExprId, t: ExprId) -> bool {
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*exp) {
            if *n == num_rational::BigRational::from_integer(2.into()) {
                if let Expr::Function(fn_id, args) = ctx.get(*base) {
                    let name = ctx.sym_name(*fn_id);
                    if name == "sin" && args.len() == 1 {
                        return crate::ordering::compare_expr(ctx, args[0], t)
                            == std::cmp::Ordering::Equal;
                    }
                }
            }
        }
    }
    false
}

fn is_cos_squared_t(ctx: &Context, expr: ExprId, t: ExprId) -> bool {
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*exp) {
            if *n == num_rational::BigRational::from_integer(2.into()) {
                if let Expr::Function(fn_id, args) = ctx.get(*base) {
                    let name = ctx.sym_name(*fn_id);
                    if name == "cos" && args.len() == 1 {
                        return crate::ordering::compare_expr(ctx, args[0], t)
                            == std::cmp::Ordering::Equal;
                    }
                }
            }
        }
    }
    false
}

/// Check if lhs - rhs matches 2·cos²(t) - 1
fn is_2cos_sq_minus_1(ctx: &Context, lhs: ExprId, rhs: ExprId, t: ExprId) -> bool {
    // rhs should be 1
    if let Expr::Number(n) = ctx.get(rhs) {
        if *n != num_rational::BigRational::from_integer(1.into()) {
            return false;
        }
    } else {
        return false;
    }

    // lhs should be 2*cos²(t)
    if let Expr::Mul(ml, mr) = ctx.get(lhs) {
        let ml = *ml;
        let mr = *mr;
        // Check for 2 * cos²(t)
        if let Expr::Number(n) = ctx.get(ml) {
            if *n == num_rational::BigRational::from_integer(2.into()) {
                return is_cos_squared_t(ctx, mr, t);
            }
        }
        if let Expr::Number(n) = ctx.get(mr) {
            if *n == num_rational::BigRational::from_integer(2.into()) {
                return is_cos_squared_t(ctx, ml, t);
            }
        }
    }
    false
}

/// Check if lhs - rhs matches 1 - 2·sin²(t)
fn is_1_minus_2sin_sq(ctx: &Context, lhs: ExprId, rhs: ExprId, t: ExprId) -> bool {
    // lhs should be 1
    if let Expr::Number(n) = ctx.get(lhs) {
        if *n != num_rational::BigRational::from_integer(1.into()) {
            return false;
        }
    } else {
        return false;
    }

    // rhs should be 2*sin²(t)
    if let Expr::Mul(ml, mr) = ctx.get(rhs) {
        let ml = *ml;
        let mr = *mr;
        // Check for 2 * sin²(t)
        if let Expr::Number(n) = ctx.get(ml) {
            if *n == num_rational::BigRational::from_integer(2.into()) {
                return is_sin_squared_t(ctx, mr, t);
            }
        }
        if let Expr::Number(n) = ctx.get(mr) {
            if *n == num_rational::BigRational::from_integer(2.into()) {
                return is_sin_squared_t(ctx, ml, t);
            }
        }
    }
    false
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
        let sec_x = ctx.call_builtin(cas_ast::BuiltinFn::Sec, vec![x]);
        let tan_x = ctx.call_builtin(cas_ast::BuiltinFn::Tan, vec![x]);
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
        let sec_x = ctx.call_builtin(cas_ast::BuiltinFn::Sec, vec![x]);
        let tan_x = ctx.call_builtin(cas_ast::BuiltinFn::Tan, vec![x]);
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
        let sin_x = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![x]);
        let cos_x = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![x]);
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
        let sin_func = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![sin_arg]);
        let sin_sq = ctx.add(Expr::Pow(sin_func, two));

        // cos arg: 1 + 2*x  => Add(1, Mul(2, x))
        let two_2 = ctx.num(2);
        let one_2 = ctx.num(1);
        let two_x_2 = ctx.add(Expr::Mul(two_2, x));
        let cos_arg = ctx.add(Expr::Add(one_2, two_x_2));
        let cos_func = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![cos_arg]);
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
        let tan_x = ctx.call_builtin(cas_ast::BuiltinFn::Tan, vec![x]);
        let arctan_tan_x = ctx.call_builtin(cas_ast::BuiltinFn::Arctan, vec![tan_x]);

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
        let sin_x = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![x]);
        let arcsin_sin_x = ctx.call_builtin(cas_ast::BuiltinFn::Arcsin, vec![sin_x]);

        scan_and_mark_patterns(&ctx, arcsin_sin_x, &mut marks);

        // sin(x) should be marked for protection
        assert!(
            marks.is_inverse_trig_protected(sin_x),
            "sin(x) in arcsin(sin(x)) should be protected"
        );
    }
}
