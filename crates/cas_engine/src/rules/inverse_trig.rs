use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use num_traits::One;
use std::cmp::Ordering;

// ==================== Helper Functions for Pattern Matching ====================

/// Check if expression equals 1
fn is_one(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        n.is_one()
    } else {
        false
    }
}

/// Check if two expressions are reciprocals: a = 1/b or b = 1/a
fn are_reciprocals(ctx: &Context, expr1: ExprId, expr2: ExprId) -> bool {
    // Get clones to avoid borrow issues
    let data1 = ctx.get(expr1).clone();
    let data2 = ctx.get(expr2).clone();

    // Case 1: expr2 = 1 / expr1
    if let Expr::Div(num, den) = &data2 {
        if is_one(ctx, *num) {
            // Compare semantically, not just ExprId
            if crate::ordering::compare_expr(ctx, *den, expr1) == Ordering::Equal {
                return true;
            }
        }
    }

    // Case 2: expr1 = 1 / expr2
    if let Expr::Div(num, den) = &data1 {
        if is_one(ctx, *num) {
            // Compare semantically, not just ExprId
            if crate::ordering::compare_expr(ctx, *den, expr2) == Ordering::Equal {
                return true;
            }
        }
    }

    // Case 3: Both are numeric (Number or Div of Numbers) and their product is 1
    // This handles cases like:
    //   - Number(2) and Div(Number(1), Number(2))
    //   - Div(Number(1), Number(3)) and Number(3)
    //   - Number(2) and Number(1/2) (though latter is rare)

    // Try to extract numeric values
    let num1_opt = extract_numeric_value(ctx, &data1);
    let num2_opt = extract_numeric_value(ctx, &data2);

    if let (Some(n1), Some(n2)) = (num1_opt, num2_opt) {
        let product = n1 * n2;
        if product.is_one() {
            return true;
        }
    }

    false
}

/// Extract numeric value from an expression if it's a pure number or fraction of numbers
fn extract_numeric_value(ctx: &Context, expr: &Expr) -> Option<num_rational::BigRational> {
    match expr {
        Expr::Number(n) => Some(n.clone()),
        Expr::Div(num_id, den_id) => {
            // Check if both numerator and denominator are numbers
            if let (Expr::Number(num), Expr::Number(den)) = (ctx.get(*num_id), ctx.get(*den_id)) {
                Some(num / den)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Check if function name is atan/arctan
fn is_atan(name: &str) -> bool {
    name == "atan" || name == "arctan"
}

/// Build sum of all terms except indices i and j
/// Returns None if no terms remain, Some(expr) otherwise
fn build_sum_without(
    ctx: &mut Context,
    terms: &[ExprId],
    skip_i: usize,
    skip_j: usize,
) -> Option<ExprId> {
    let mut remaining: Vec<ExprId> = Vec::new();

    for (idx, &term) in terms.iter().enumerate() {
        if idx != skip_i && idx != skip_j {
            remaining.push(term);
        }
    }

    match remaining.len() {
        0 => None,               // No remaining terms
        1 => Some(remaining[0]), // Single term
        _ => {
            // Build Add tree from remaining terms
            let mut result = remaining[0];
            for &term in &remaining[1..] {
                result = ctx.add(Expr::Add(result, term));
            }
            Some(result)
        }
    }
}

/// Collect all additive terms from an expression (flattens Add tree)
/// For example: ((a + b) + c) → [a, b, c]
fn collect_add_terms_flat(ctx: &Context, expr_id: ExprId) -> Vec<ExprId> {
    let mut terms = Vec::new();
    collect_add_terms_recursive(ctx, expr_id, &mut terms);
    terms
}

/// Recursively collect additive terms
fn collect_add_terms_recursive(ctx: &Context, expr_id: ExprId, terms: &mut Vec<ExprId>) {
    match ctx.get(expr_id) {
        Expr::Add(l, r) => {
            collect_add_terms_recursive(ctx, *l, terms);
            collect_add_terms_recursive(ctx, *r, terms);
        }
        _ => {
            terms.push(expr_id);
        }
    }
}

/// Combine optional base with new term
fn combine_with_term(ctx: &mut Context, base: Option<ExprId>, new_term: ExprId) -> ExprId {
    match base {
        None => new_term,
        Some(b) => ctx.add(Expr::Add(b, new_term)),
    }
}

/// Check if both expressions match a pattern, handling negation automatically
///
/// This is a generalized helper for pair rules. If the check_fn succeeds:
/// - For positive pairs: returns result directly
/// - For negated pairs: returns negated result
///
/// Pattern: If f(x) + g(x) = V, then -f(x) - g(x) = -V
fn check_pair_with_negation<F>(
    ctx: &mut Context,
    term_i_data: Expr,
    term_j_data: Expr,
    terms: &[ExprId],
    i: usize,
    j: usize,
    check_fn: F,
) -> Option<Rewrite>
where
    F: Fn(&mut Context, &Expr, &Expr) -> Option<(ExprId, String)>,
{
    // Case 1: Try direct match (positive pair)
    if let Some((result, desc)) = check_fn(ctx, &term_i_data, &term_j_data) {
        let remaining = build_sum_without(ctx, terms, i, j);
        let final_result = combine_with_term(ctx, remaining, result);
        return Some(Rewrite {
            new_expr: final_result,
            description: desc,
        });
    }

    // Case 2: Try negated match (-f(x) - g(x) = -result)
    if let (Expr::Neg(inner_i), Expr::Neg(inner_j)) = (&term_i_data, &term_j_data) {
        let inner_i_data = ctx.get(*inner_i).clone();
        let inner_j_data = ctx.get(*inner_j).clone();

        if let Some((result, desc)) = check_fn(ctx, &inner_i_data, &inner_j_data) {
            // Negate the result
            let neg_result = ctx.add(Expr::Neg(result));
            let remaining = build_sum_without(ctx, terms, i, j);
            let final_result = combine_with_term(ctx, remaining, neg_result);

            return Some(Rewrite {
                new_expr: final_result,
                description: format!("-[{}]", desc),
            });
        }
    }

    None
}

/// Check if expression is sin(x)/cos(x) pattern (expanded tan)
/// Returns Some(x) if pattern matches, None otherwise
fn is_expanded_tan(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Div(num, den) = ctx.get(expr) {
        if let (Expr::Function(num_fn, num_args), Expr::Function(den_fn, den_args)) =
            (ctx.get(*num), ctx.get(*den))
        {
            if num_fn == "sin"
                && den_fn == "cos"
                && num_args.len() == 1
                && den_args.len() == 1
                && num_args[0] == den_args[0]
            {
                return Some(num_args[0]); // Return the argument x
            }
        }
    }
    None
}

// ==================== Inverse Trig Identity Rules ====================

// Rule 1: Composition Identities - sin(arcsin(x)) = x, etc.
define_rule!(
    InverseTrigCompositionRule,
    "Inverse Trig Composition",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(outer_name, outer_args) = ctx.get(expr) {
            if outer_args.len() == 1 {
                let inner_expr = outer_args[0];

                // Check for literal function composition first
                if let Expr::Function(inner_name, inner_args) = ctx.get(inner_expr) {
                    if inner_args.len() == 1 {
                        let x = inner_args[0];

                        // sin(arcsin(x)) = x (also handles asin variant)
                        if outer_name == "sin" && (inner_name == "arcsin" || inner_name == "asin") {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "sin(arcsin(x)) = x".to_string(),
                            });
                        }

                        // cos(arccos(x)) = x (also handles acos variant)
                        if outer_name == "cos" && (inner_name == "arccos" || inner_name == "acos") {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "cos(arccos(x)) = x".to_string(),
                            });
                        }

                        // tan(arctan(x)) = x (also handles atan variant)
                        if outer_name == "tan" && (inner_name == "arctan" || inner_name == "atan") {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "tan(arctan(x)) = x".to_string(),
                            });
                        }

                        // arcsin(sin(x)) = x (with domain restrictions, but we simplify anyway)
                        if outer_name == "arcsin" && inner_name == "sin" {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "arcsin(sin(x)) = x".to_string(),
                            });
                        }

                        // arccos(cos(x)) = x
                        if outer_name == "arccos" && inner_name == "cos" {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "arccos(cos(x)) = x".to_string(),
                            });
                        }

                        // arctan(tan(x)) = x
                        if outer_name == "arctan" && inner_name == "tan" {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "arctan(tan(x)) = x".to_string(),
                            });
                        }
                    }
                }

                // ✨ NEW: Check for expanded tan in arctan: arctan(sin(x)/cos(x)) = x
                if outer_name == "arctan" {
                    if let Some(arg) = is_expanded_tan(ctx, inner_expr) {
                        return Some(Rewrite {
                            new_expr: arg,
                            description: "arctan(sin(x)/cos(x)) = arctan(tan(x)) = x".to_string(),
                        });
                    }
                }
            }
        }
        None
    }
);

// Rule 2: arcsin(x) + arccos(x) = π/2
// Enhanced to search across all additive terms (n-ary matching)
// and handle negated pairs: -arcsin(x) - arccos(x) = -π/2
define_rule!(
    InverseTrigSumRule,
    "Inverse Trig Sum Identity",
    Some(vec!["Add"]),
    |ctx, expr| {
        // Flatten Add tree to get all terms
        let terms = collect_add_terms_flat(ctx, expr);

        // Search for asin/acos pairs among all terms
        for i in 0..terms.len() {
            for j in (i + 1)..terms.len() {
                let term_i_data = ctx.get(terms[i]).clone();
                let term_j_data = ctx.get(terms[j]).clone();

                // Use generalized helper to check both positive and negated pairs
                if let Some(rewrite) = check_pair_with_negation(
                    ctx,
                    term_i_data,
                    term_j_data,
                    &terms,
                    i,
                    j,
                    |ctx, expr_i, expr_j| {
                        // Check if both are asin/acos functions with equal arguments
                        if let (Expr::Function(name_i, args_i), Expr::Function(name_j, args_j)) =
                            (expr_i, expr_j)
                        {
                            if args_i.len() == 1 && args_j.len() == 1 {
                                let arg_i = args_i[0];
                                let arg_j = args_j[0];

                                // Check if arguments are equal
                                let args_equal = arg_i == arg_j
                                    || crate::ordering::compare_expr(ctx, arg_i, arg_j)
                                        == std::cmp::Ordering::Equal;

                                if args_equal {
                                    // Check for both "asin"/"arcsin" and "acos"/"arccos" variants
                                    let is_arcsin = |name: &str| name == "arcsin" || name == "asin";
                                    let is_arccos = |name: &str| name == "arccos" || name == "acos";

                                    if (is_arcsin(name_i) && is_arccos(name_j))
                                        || (is_arccos(name_i) && is_arcsin(name_j))
                                    {
                                        // Found asin(x) + acos(x)! Build π/2
                                        let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                                        let two = ctx.num(2);
                                        let pi_half = ctx.add(Expr::Div(pi, two));

                                        return Some((
                                            pi_half,
                                            "arcsin(x) + arccos(x) = π/2".to_string(),
                                        ));
                                    }
                                }
                            }
                        }
                        None
                    },
                ) {
                    return Some(rewrite);
                }
            }
        }
        None
    }
);

// Rule 3: arctan(x) + arctan(1/x) = π/2 (for x > 0)
// Enhanced to search across all additive terms (n-ary matching)
define_rule!(
    InverseTrigAtanRule,
    "Inverse Tan Relations",
    Some(vec!["Add"]),
    |ctx, expr| {
        // Collect all additive terms (flattens nested Add nodes)
        let terms = collect_add_terms_flat(ctx, expr);

        // Need at least 2 terms to find a pair
        if terms.len() < 2 {
            return None;
        }

        // Search for atan(x) + atan(1/x) among all pairs of terms
        for i in 0..terms.len() {
            for j in (i + 1)..terms.len() {
                let term_i_data = ctx.get(terms[i]).clone();
                let term_j_data = ctx.get(terms[j]).clone();

                // Use generalized helper to check both positive and negated pairs
                if let Some(rewrite) = check_pair_with_negation(
                    ctx,
                    term_i_data,
                    term_j_data,
                    &terms,
                    i,
                    j,
                    |ctx, expr_i, expr_j| {
                        // Check if both are atan functions with reciprocal arguments
                        if let (Expr::Function(name_i, args_i), Expr::Function(name_j, args_j)) =
                            (expr_i, expr_j)
                        {
                            if is_atan(name_i)
                                && is_atan(name_j)
                                && args_i.len() == 1
                                && args_j.len() == 1
                            {
                                if are_reciprocals(ctx, args_i[0], args_j[0]) {
                                    // Found atan(x) + atan(1/x)! Build π/2
                                    let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                                    let two = ctx.num(2);
                                    let pi_half = ctx.add(Expr::Div(pi, two));

                                    return Some((
                                        pi_half,
                                        "arctan(x) + arctan(1/x) = π/2".to_string(),
                                    ));
                                }
                            }
                        }
                        None
                    },
                ) {
                    return Some(rewrite);
                }
            }
        }

        None
    }
);

// Rule 4: Negative argument handling for inverse trig
define_rule!(
    InverseTrigNegativeRule,
    "Inverse Trig Negative Argument",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if args.len() == 1 {
                let arg = args[0];

                // Check for negative argument: Neg(x) or Mul(-1, x)
                let inner_opt = match ctx.get(arg) {
                    Expr::Neg(inner) => Some(*inner),
                    Expr::Mul(l, r) => {
                        if let Expr::Number(n) = ctx.get(*l) {
                            if *n == num_rational::BigRational::from_integer((-1).into()) {
                                Some(*r)
                            } else {
                                None
                            }
                        } else if let Expr::Number(n) = ctx.get(*r) {
                            if *n == num_rational::BigRational::from_integer((-1).into()) {
                                Some(*l)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                if let Some(inner) = inner_opt {
                    match name.as_str() {
                        "arcsin" => {
                            // arcsin(-x) = -arcsin(x)
                            let arcsin_inner =
                                ctx.add(Expr::Function("arcsin".to_string(), vec![inner]));
                            let new_expr = ctx.add(Expr::Neg(arcsin_inner));
                            return Some(Rewrite {
                                new_expr,
                                description: "arcsin(-x) = -arcsin(x)".to_string(),
                            });
                        }
                        "arctan" => {
                            // arctan(-x) = -arctan(x)
                            let arctan_inner =
                                ctx.add(Expr::Function("arctan".to_string(), vec![inner]));
                            let new_expr = ctx.add(Expr::Neg(arctan_inner));
                            return Some(Rewrite {
                                new_expr,
                                description: "arctan(-x) = -arctan(x)".to_string(),
                            });
                        }
                        "arccos" => {
                            // arccos(-x) = π - arccos(x)
                            let arccos_inner =
                                ctx.add(Expr::Function("arccos".to_string(), vec![inner]));
                            let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                            let new_expr = ctx.add(Expr::Sub(pi, arccos_inner));
                            return Some(Rewrite {
                                new_expr,
                                description: "arccos(-x) = π - arccos(x)".to_string(),
                            });
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }
);

// ==================== Phase 5: Inverse Function Relations ====================
// Unify inverse trig functions by converting arcsec/arccsc/arccot to arccos/arcsin/arctan

/// arcsec(x) → arccos(1/x)
define_rule!(
    ArcsecToArccosRule,
    "arcsec(x) → arccos(1/x)",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if (name == "arcsec" || name == "asec") && args.len() == 1 {
                let arg = args[0];

                // Build 1/arg
                let one = ctx.num(1);
                let reciprocal = ctx.add(Expr::Div(one, arg));

                // Build arccos(1/arg)
                let result = ctx.add(Expr::Function("arccos".to_string(), vec![reciprocal]));

                return Some(Rewrite {
                    new_expr: result,
                    description: "arcsec(x) → arccos(1/x)".to_string(),
                });
            }
        }
        None
    }
);

/// arccsc(x) → arcsin(1/x)
define_rule!(
    ArccscToArcsinRule,
    "arccsc(x) → arcsin(1/x)",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if (name == "arccsc" || name == "acsc") && args.len() == 1 {
                let arg = args[0];

                // Build 1/arg
                let one = ctx.num(1);
                let reciprocal = ctx.add(Expr::Div(one, arg));

                // Build arcsin(1/arg)
                let result = ctx.add(Expr::Function("arcsin".to_string(), vec![reciprocal]));

                return Some(Rewrite {
                    new_expr: result,
                    description: "arccsc(x) → arcsin(1/x)".to_string(),
                });
            }
        }
        None
    }
);

/// arccot(x) → arctan(1/x)
/// Simplified version - works for all x ≠ 0 on principal branch
define_rule!(
    ArccotToArctanRule,
    "arccot(x) → arctan(1/x)",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if (name == "arccot" || name == "acot") && args.len() == 1 {
                let arg = args[0];

                // Build 1/arg
                let one = ctx.num(1);
                let reciprocal = ctx.add(Expr::Div(one, arg));

                // Build arctan(1/arg)
                let result = ctx.add(Expr::Function("arctan".to_string(), vec![reciprocal]));

                return Some(Rewrite {
                    new_expr: result,
                    description: "arccot(x) → arctan(1/x)".to_string(),
                });
            }
        }
        None
    }
);

// ==================== Registration ====================

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(InverseTrigCompositionRule));
    simplifier.add_rule(Box::new(InverseTrigSumRule));
    simplifier.add_rule(Box::new(InverseTrigAtanRule));
    simplifier.add_rule(Box::new(InverseTrigNegativeRule));
    simplifier.add_rule(Box::new(ArcsecToArccosRule));
    simplifier.add_rule(Box::new(ArccscToArcsinRule));
    simplifier.add_rule(Box::new(ArccotToArctanRule));
}
