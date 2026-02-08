use crate::define_rule;
use crate::helpers::is_one;
use crate::nary::build_balanced_add;
use crate::rule::Rewrite;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_traits::One;
use std::cmp::Ordering;

// ==================== Helper Functions for Pattern Matching ====================

// is_one is now imported from crate::helpers

/// Check if two expressions are reciprocals: a = 1/b or b = 1/a
fn are_reciprocals(ctx: &Context, expr1: ExprId, expr2: ExprId) -> bool {
    // Case 1: expr2 = 1 / expr1
    if let Expr::Div(num, den) = ctx.get(expr2) {
        if is_one(ctx, *num) && crate::ordering::compare_expr(ctx, *den, expr1) == Ordering::Equal {
            return true;
        }
    }

    // Case 2: expr1 = 1 / expr2
    if let Expr::Div(num, den) = ctx.get(expr1) {
        if is_one(ctx, *num) && crate::ordering::compare_expr(ctx, *den, expr2) == Ordering::Equal {
            return true;
        }
    }

    // Case 3: Both are numeric and their product is 1
    let num1_opt = extract_numeric_value(ctx, expr1);
    let num2_opt = extract_numeric_value(ctx, expr2);

    if let (Some(n1), Some(n2)) = (num1_opt, num2_opt) {
        if (n1 * n2).is_one() {
            return true;
        }
    }

    false
}

/// Extract numeric value from an expression if it's a pure number or fraction of numbers
fn extract_numeric_value(ctx: &Context, id: ExprId) -> Option<num_rational::BigRational> {
    match ctx.get(id) {
        Expr::Number(n) => Some(n.clone()),
        Expr::Div(num_id, den_id) => {
            if let (Expr::Number(num), Expr::Number(den)) = (ctx.get(*num_id), ctx.get(*den_id)) {
                Some(num / den)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Check if any reciprocal atan pairs exist in the list of terms
/// This helps Machin rule avoid combining terms when reciprocal pairs should be matched first
fn has_reciprocal_atan_pair(ctx: &Context, terms: &[ExprId]) -> bool {
    // Collect all atan arguments
    let mut atan_args: Vec<ExprId> = Vec::new();
    for &term in terms {
        if let Expr::Function(fn_id, args) = ctx.get(term) {
            if let Some(b) = ctx.builtin_of(*fn_id) {
                if matches!(b, BuiltinFn::Atan | BuiltinFn::Arctan) && args.len() == 1 {
                    atan_args.push(args[0]);
                }
            }
        }
    }

    // Check if any pair are reciprocals
    for i in 0..atan_args.len() {
        for j in (i + 1)..atan_args.len() {
            if are_reciprocals(ctx, atan_args[i], atan_args[j]) {
                return true;
            }
        }
    }
    false
}

/// Build sum of all terms except indices i and j
/// Returns None if no terms remain, Some(expr) otherwise
/// Uses build_balanced_add for consistent balanced tree construction.
fn build_sum_without(
    ctx: &mut Context,
    terms: &[ExprId],
    skip_i: usize,
    skip_j: usize,
) -> Option<ExprId> {
    let remaining: Vec<ExprId> = terms
        .iter()
        .enumerate()
        .filter(|(idx, _)| *idx != skip_i && *idx != skip_j)
        .map(|(_, &term)| term)
        .collect();

    match remaining.len() {
        0 => None,
        _ => Some(build_balanced_add(ctx, &remaining)),
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
#[allow(clippy::too_many_arguments)] // All parameters are semantically distinct
fn check_pair_with_negation<F>(
    ctx: &mut Context,
    term_i: ExprId,
    term_j: ExprId,
    terms: &[ExprId],
    i: usize,
    j: usize,
    check_fn: F,
) -> Option<Rewrite>
where
    F: Fn(&mut Context, &Expr, &Expr) -> Option<(ExprId, String)>,
{
    // Clone once inside to break borrow aliasing (ctx.get vs ctx.add)
    let term_i_data = ctx.get(term_i).clone();
    let term_j_data = ctx.get(term_j).clone();
    // Case 1: Try direct match (positive pair)
    if let Some((result, desc)) = check_fn(ctx, &term_i_data, &term_j_data) {
        let remaining = build_sum_without(ctx, terms, i, j);
        let final_result = combine_with_term(ctx, remaining, result);

        // Build local before: Add(term_i, term_j) to show exactly what matched
        let local_before = ctx.add(Expr::Add(term_i, term_j));

        return Some(
            Rewrite::new(final_result)
                .desc(desc)
                .local(local_before, result),
        );
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

            // Build local before: Add(term_i, term_j) to show exactly what matched
            let local_before = ctx.add(Expr::Add(term_i, term_j));

            return Some(
                Rewrite::new(final_result)
                    .desc_lazy(|| format!("-[{}]", desc))
                    .local(local_before, neg_result),
            );
        }
    }

    // Case 3: Try coefficient match (k*f(x) + k*g(x) = k*V)
    // Extract coefficients from Mul expressions
    if let (Expr::Mul(coef_i, inner_i), Expr::Mul(coef_j, inner_j)) = (&term_i_data, &term_j_data) {
        // Check if coefficients are equal
        if crate::ordering::compare_expr(ctx, *coef_i, *coef_j) == std::cmp::Ordering::Equal {
            let inner_i_data = ctx.get(*inner_i).clone();
            let inner_j_data = ctx.get(*inner_j).clone();

            if let Some((result, desc)) = check_fn(ctx, &inner_i_data, &inner_j_data) {
                // Multiply result by coefficient: k * V
                let scaled_result = ctx.add(Expr::Mul(*coef_i, result));
                let remaining = build_sum_without(ctx, terms, i, j);
                let final_result = combine_with_term(ctx, remaining, scaled_result);

                let local_before = ctx.add(Expr::Add(term_i, term_j));

                return Some(
                    Rewrite::new(final_result)
                        .desc_lazy(|| format!("k·[{}]", desc))
                        .local(local_before, scaled_result),
                );
            }
        }
    }

    // Case 4: Try negated coefficient match (k*(-f(x)) + k*(-g(x)) = -k*V)
    if let (Expr::Mul(coef_i, inner_i), Expr::Mul(coef_j, inner_j)) = (&term_i_data, &term_j_data) {
        if crate::ordering::compare_expr(ctx, *coef_i, *coef_j) == std::cmp::Ordering::Equal {
            if let (Expr::Neg(neg_i), Expr::Neg(neg_j)) = (ctx.get(*inner_i), ctx.get(*inner_j)) {
                let inner_i_data = ctx.get(*neg_i).clone();
                let inner_j_data = ctx.get(*neg_j).clone();

                if let Some((result, desc)) = check_fn(ctx, &inner_i_data, &inner_j_data) {
                    // Multiply result by -coefficient: -k * V
                    let scaled_result = ctx.add(Expr::Mul(*coef_i, result));
                    let neg_scaled = ctx.add(Expr::Neg(scaled_result));
                    let remaining = build_sum_without(ctx, terms, i, j);
                    let final_result = combine_with_term(ctx, remaining, neg_scaled);

                    let local_before = ctx.add(Expr::Add(term_i, term_j));

                    return Some(
                        Rewrite::new(final_result)
                            .desc_lazy(|| format!("-k·[{}]", desc))
                            .local(local_before, neg_scaled),
                    );
                }
            }
        }
    }

    None
}

// ==================== Inverse Trig Identity Rules ====================

// Rule 1: Composition Identities - sin(arcsin(x)) = x, etc.
// sin(arcsin(x)) and cos(arccos(x)) require x ∈ [-1, 1] (domain of inverse)
// tan(arctan(x)) is always valid (arctan has domain R)
pub struct InverseTrigCompositionRule;

impl crate::rule::Rule for InverseTrigCompositionRule {
    fn name(&self) -> &str {
        "Inverse Trig Composition"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        if let Expr::Function(outer_name, outer_args) = ctx.get(expr) {
            if outer_args.len() == 1 {
                let inner_expr = outer_args[0];

                // Check for literal function composition first
                if let Expr::Function(inner_name, inner_args) = ctx.get(inner_expr) {
                    if inner_args.len() == 1 {
                        let x = inner_args[0];

                        // sin(arcsin(x)) = x (requires x ∈ [-1, 1])
                        if ctx.is_builtin(*outer_name, BuiltinFn::Sin)
                            && (ctx.is_builtin(*inner_name, BuiltinFn::Arcsin)
                                || ctx.is_builtin(*inner_name, BuiltinFn::Asin))
                        {
                            // Check domain_mode: sin(arcsin(x)) requires x ∈ [-1,1]
                            let mode = parent_ctx.domain_mode();
                            match mode {
                                crate::domain::DomainMode::Strict => {
                                    // Check if x is a literal number in [-1, 1]
                                    if let Expr::Number(n) = ctx.get(x) {
                                        let one = num_rational::BigRational::one();
                                        let neg_one = -one.clone();
                                        if *n >= neg_one && *n <= one {
                                            return Some(
                                                Rewrite::new(x).desc("sin(arcsin(x)) = x"),
                                            );
                                        }
                                    }
                                    // Variable: don't simplify in strict mode
                                    return None;
                                }
                                crate::domain::DomainMode::Generic => {
                                    return Some(Rewrite::new(x).desc("sin(arcsin(x)) = x"));
                                }
                                crate::domain::DomainMode::Assume => {
                                    return Some(
                                        Rewrite::new(x)
                                            .desc("sin(arcsin(x)) = x (assuming x ∈ [-1, 1])")
                                            .assume(crate::assumptions::AssumptionEvent::defined(
                                                ctx, x,
                                            )),
                                    );
                                }
                            }
                        }

                        // cos(arccos(x)) = x (requires x ∈ [-1, 1])
                        if ctx.is_builtin(*outer_name, BuiltinFn::Cos)
                            && (ctx.is_builtin(*inner_name, BuiltinFn::Arccos)
                                || ctx.is_builtin(*inner_name, BuiltinFn::Acos))
                        {
                            let mode = parent_ctx.domain_mode();
                            match mode {
                                crate::domain::DomainMode::Strict => {
                                    if let Expr::Number(n) = ctx.get(x) {
                                        let one = num_rational::BigRational::one();
                                        let neg_one = -one.clone();
                                        if *n >= neg_one && *n <= one {
                                            return Some(
                                                Rewrite::new(x).desc("cos(arccos(x)) = x"),
                                            );
                                        }
                                    }
                                    return None;
                                }
                                crate::domain::DomainMode::Generic => {
                                    return Some(Rewrite::new(x).desc("cos(arccos(x)) = x"));
                                }
                                crate::domain::DomainMode::Assume => {
                                    return Some(
                                        Rewrite::new(x)
                                            .desc("cos(arccos(x)) = x (assuming x ∈ [-1, 1])")
                                            .assume(crate::assumptions::AssumptionEvent::defined(
                                                ctx, x,
                                            )),
                                    );
                                }
                            }
                        }

                        // tan(arctan(x)) = x (always safe - arctan has domain R)
                        if ctx.is_builtin(*outer_name, BuiltinFn::Tan)
                            && (ctx.is_builtin(*inner_name, BuiltinFn::Arctan)
                                || ctx.is_builtin(*inner_name, BuiltinFn::Atan))
                        {
                            return Some(Rewrite::new(x).desc("tan(arctan(x)) = x"));
                        }

                        // SAFE NESTED CASE: arctan(tan(arctan(u))) = arctan(u)
                        // Because arctan(u) is always in (-π/2, π/2), it's safe to cancel
                        let is_outer_arctan = ctx.is_builtin(*outer_name, BuiltinFn::Arctan)
                            || ctx.is_builtin(*outer_name, BuiltinFn::Atan);
                        let is_inner_tan = ctx.is_builtin(*inner_name, BuiltinFn::Tan);
                        if is_outer_arctan && is_inner_tan {
                            // Check if the argument of tan is arctan(something)
                            if let Expr::Function(innermost_name, innermost_args) = ctx.get(x) {
                                let is_innermost_arctan = ctx
                                    .is_builtin(*innermost_name, BuiltinFn::Arctan)
                                    || ctx.is_builtin(*innermost_name, BuiltinFn::Atan);
                                if is_innermost_arctan && innermost_args.len() == 1 {
                                    // atan(tan(atan(u))) → atan(u)
                                    let atan_u = x; // x is already atan(u)
                                    return Some(Rewrite::new(atan_u).desc(
                                        "arctan(tan(arctan(u))) = arctan(u) (principal branch)",
                                    ));
                                }
                            }
                        }

                        // NOTE: We intentionally DO NOT simplify:
                        // - arcsin(sin(x)) → x  (unsafe: fails outside [-π/2, π/2])
                        // - arccos(cos(x)) → x  (unsafe: fails outside [0, π])
                        // - arctan(tan(x)) → x  (unsafe: fails outside (-π/2, π/2))
                        // These are handled by PrincipalBranchInverseTrigRule when inv_trig=principal
                    }
                }
            }
        }
        None
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }
}

// Rule 2: arcsin(x) + arccos(x) = π/2
// Enhanced to search across all additive terms (n-ary matching)
// and handle negated pairs: -arcsin(x) - arccos(x) = -π/2
define_rule!(
    InverseTrigSumRule,
    "Inverse Trig Sum Identity",
    Some(crate::target_kind::TargetKindSet::ADD),
    |ctx, expr| {
        // Flatten Add tree to get all terms
        let terms = crate::helpers::flatten_add_sub_chain(ctx, expr);

        // Search for asin/acos pairs among all terms
        for i in 0..terms.len() {
            for j in (i + 1)..terms.len() {
                // Use generalized helper to check both positive and negated pairs
                if let Some(rewrite) = check_pair_with_negation(
                    ctx,
                    terms[i],
                    terms[j],
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
                                    let is_i_arcsin = matches!(
                                        ctx.builtin_of(*name_i),
                                        Some(BuiltinFn::Arcsin | BuiltinFn::Asin)
                                    );
                                    let is_j_arcsin = matches!(
                                        ctx.builtin_of(*name_j),
                                        Some(BuiltinFn::Arcsin | BuiltinFn::Asin)
                                    );
                                    let is_i_arccos = matches!(
                                        ctx.builtin_of(*name_i),
                                        Some(BuiltinFn::Arccos | BuiltinFn::Acos)
                                    );
                                    let is_j_arccos = matches!(
                                        ctx.builtin_of(*name_j),
                                        Some(BuiltinFn::Arccos | BuiltinFn::Acos)
                                    );

                                    if (is_i_arcsin && is_j_arccos) || (is_i_arccos && is_j_arcsin)
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
// Only applies at root Add level (when parent is NOT Add) to ensure all pairs are visible
pub struct InverseTrigAtanRule;

impl crate::rule::Rule for InverseTrigAtanRule {
    fn name(&self) -> &str {
        "Inverse Tan Relations"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        // Only match on Add expressions
        if !matches!(ctx.get(expr), Expr::Add(_, _)) {
            return None;
        }

        // GUARD: Skip if this Add is inside another Add (sub-sum)
        // This ensures we only process at the root Add level where we can see ALL terms
        if let Some(parent_id) = parent_ctx.immediate_parent() {
            if matches!(ctx.get(parent_id), Expr::Add(_, _)) {
                return None;
            }
        }

        // Collect all additive terms (flattens nested Add nodes)
        let terms = crate::helpers::flatten_add_sub_chain(ctx, expr);

        // Need at least 2 terms to find a pair
        if terms.len() < 2 {
            return None;
        }

        // Search for atan(x) + atan(1/x) among all pairs of terms
        for i in 0..terms.len() {
            for j in (i + 1)..terms.len() {
                // Use generalized helper to check both positive and negated pairs
                if let Some(rewrite) = check_pair_with_negation(
                    ctx,
                    terms[i],
                    terms[j],
                    &terms,
                    i,
                    j,
                    |ctx, expr_i, expr_j| {
                        // Check if both are atan functions with reciprocal arguments
                        if let (Expr::Function(name_i, args_i), Expr::Function(name_j, args_j)) =
                            (expr_i, expr_j)
                        {
                            let is_i_atan = matches!(
                                ctx.builtin_of(*name_i),
                                Some(BuiltinFn::Atan | BuiltinFn::Arctan)
                            );
                            let is_j_atan = matches!(
                                ctx.builtin_of(*name_j),
                                Some(BuiltinFn::Atan | BuiltinFn::Arctan)
                            );
                            if is_i_atan
                                && is_j_atan
                                && args_i.len() == 1
                                && args_j.len() == 1
                                && are_reciprocals(ctx, args_i[0], args_j[0])
                            {
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
                        None
                    },
                ) {
                    return Some(rewrite);
                }
            }
        }

        None
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::ADD)
    }

    fn priority(&self) -> i32 {
        10 // Higher priority than Machin rule (-10) to run first
    }
}

// Rule: arctan(a) + arctan(b) = arctan((a+b)/(1-a*b)) when a,b are rational and 1-a*b > 0
// This is Machin's identity (simplified form) - enables atan(1/2)+atan(1/3) = π/4
//
// CRITICAL: Uses manual Rule impl to access parent_ctx for sub-sum guard.
// Only applies at "root sum" level - skips when parent is also Add to avoid
// interfering with reciprocal pair detection in larger sums.
pub struct AtanAddRationalRule;

impl crate::rule::Rule for AtanAddRationalRule {
    fn name(&self) -> &str {
        "Arctan Addition (Machin)"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        // GUARD: Skip if this Add is inside another Add (sub-sum)
        // This prevents combining atan(1/2)+atan(1/5) inside atan(2)+atan(1/2)+atan(5)+atan(1/5)
        // before the reciprocal rule can find atan(2)+atan(1/2) → π/2
        if let Some(parent_id) = parent_ctx.immediate_parent() {
            if matches!(ctx.get(parent_id), Expr::Add(_, _)) {
                return None; // Don't combine in sub-sums
            }
        }

        // Collect all additive terms
        let terms = crate::helpers::flatten_add_sub_chain(ctx, expr);

        // Need at least 2 terms
        if terms.len() < 2 {
            return None;
        }

        // GUARD: Skip if ANY reciprocal pairs exist in the sum
        // Let InverseTrigAtanRule handle those first
        if has_reciprocal_atan_pair(ctx, &terms) {
            return None;
        }

        // Search for atan(a) + atan(b) pairs among all terms
        for i in 0..terms.len() {
            for j in (i + 1)..terms.len() {
                // Check if both are atan functions with rational arguments
                if let (Expr::Function(name_i, args_i), Expr::Function(name_j, args_j)) =
                    (ctx.get(terms[i]), ctx.get(terms[j]))
                {
                    let (name_i, args_i) = (*name_i, args_i.clone());
                    let (name_j, args_j) = (*name_j, args_j.clone());
                    let is_i_atan = matches!(
                        ctx.builtin_of(name_i),
                        Some(BuiltinFn::Atan | BuiltinFn::Arctan)
                    );
                    let is_j_atan = matches!(
                        ctx.builtin_of(name_j),
                        Some(BuiltinFn::Atan | BuiltinFn::Arctan)
                    );
                    if is_i_atan && is_j_atan && args_i.len() == 1 && args_j.len() == 1 {
                        let arg_i = args_i[0];
                        let arg_j = args_j[0];

                        // Extract rational values from both arguments
                        let val_i = extract_numeric_value(ctx, arg_i);
                        let val_j = extract_numeric_value(ctx, arg_j);

                        if let (Some(a), Some(b)) = (val_i, val_j) {
                            // Guard: 1 - a*b > 0 (ensures no branch crossing)
                            let ab = &a * &b;
                            let one = num_rational::BigRational::from_integer(1.into());
                            // Guard: Skip if a*b = 1 (reciprocals) - let InverseTrigAtanRule handle those
                            if ab == one {
                                continue;
                            }

                            let one_minus_ab = &one - &ab;

                            // Skip if 1-ab <= 0 (would need +kπ correction)
                            if one_minus_ab <= num_rational::BigRational::from_integer(0.into()) {
                                continue;
                            }

                            // Compute (a+b)/(1-ab)
                            let a_plus_b = &a + &b;
                            let result_val = &a_plus_b / &one_minus_ab;

                            // Build the result expression: arctan((a+b)/(1-ab))
                            let result_num = ctx.add(Expr::Number(result_val));
                            let result_atan = ctx.call("arctan", vec![result_num]);

                            // Build remaining terms (if any)
                            let remaining = build_sum_without(ctx, &terms, i, j);
                            let final_result = combine_with_term(ctx, remaining, result_atan);

                            // Build local before: Add(term_i, term_j)
                            let local_before = ctx.add(Expr::Add(terms[i], terms[j]));

                            return Some(
                                crate::rule::Rewrite::new(final_result)
                                    .desc_lazy(|| {
                                        format!(
                                            "arctan({}) + arctan({}) = arctan((a+b)/(1-ab))",
                                            a, b
                                        )
                                    })
                                    .local(local_before, result_atan),
                            );
                        }
                    }
                }
            }
        }

        None
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::ADD)
    }

    fn priority(&self) -> i32 {
        -10 // Lower priority than reciprocal pair detection (InverseTrigAtanRule)
    }
}

// Rule 4: Negative argument handling for inverse trig
define_rule!(
    InverseTrigNegativeRule,
    "Inverse Trig Negative Argument",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        if let Expr::Function(fn_id, args) = ctx.get(expr) {
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
                    match ctx.builtin_of(*fn_id) {
                        Some(BuiltinFn::Arcsin) => {
                            // arcsin(-x) = -arcsin(x)
                            let arcsin_inner = ctx.call("arcsin", vec![inner]);
                            let new_expr = ctx.add(Expr::Neg(arcsin_inner));
                            return Some(Rewrite::new(new_expr).desc("arcsin(-x) = -arcsin(x)"));
                        }
                        Some(BuiltinFn::Arctan) => {
                            // arctan(-x) = -arctan(x)
                            let arctan_inner = ctx.call("arctan", vec![inner]);
                            let new_expr = ctx.add(Expr::Neg(arctan_inner));
                            return Some(Rewrite::new(new_expr).desc("arctan(-x) = -arctan(x)"));
                        }
                        Some(BuiltinFn::Arccos) => {
                            // arccos(-x) = π - arccos(x)
                            let arccos_inner = ctx.call("arccos", vec![inner]);
                            let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                            let new_expr = ctx.add(Expr::Sub(pi, arccos_inner));
                            return Some(Rewrite::new(new_expr).desc("arccos(-x) = π - arccos(x)"));
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

// arcsec(x) → arccos(1/x)
define_rule!(
    ArcsecToArccosRule,
    "arcsec(x) → arccos(1/x)",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            if matches!(
                ctx.builtin_of(*fn_id),
                Some(BuiltinFn::Asec | BuiltinFn::Arcsec)
            ) && args.len() == 1
            {
                let arg = args[0];

                // Build 1/arg
                let one = ctx.num(1);
                let reciprocal = ctx.add(Expr::Div(one, arg));

                // Build arccos(1/arg)
                let result = ctx.call("arccos", vec![reciprocal]);

                return Some(Rewrite::new(result).desc("arcsec(x) → arccos(1/x)"));
            }
        }
        None
    }
);

// arccsc(x) → arcsin(1/x)
define_rule!(
    ArccscToArcsinRule,
    "arccsc(x) → arcsin(1/x)",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            if matches!(
                ctx.builtin_of(*fn_id),
                Some(BuiltinFn::Acsc | BuiltinFn::Arccsc)
            ) && args.len() == 1
            {
                let arg = args[0];

                // Build 1/arg
                let one = ctx.num(1);
                let reciprocal = ctx.add(Expr::Div(one, arg));

                // Build arcsin(1/arg)
                let result = ctx.call("arcsin", vec![reciprocal]);

                return Some(Rewrite::new(result).desc("arccsc(x) → arcsin(1/x)"));
            }
        }
        None
    }
);

// arccot(x) → arctan(1/x)
// Simplified version - works for all x ≠ 0 on principal branch
define_rule!(
    ArccotToArctanRule,
    "arccot(x) → arctan(1/x)",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            if matches!(
                ctx.builtin_of(*fn_id),
                Some(BuiltinFn::Acot | BuiltinFn::Arccot)
            ) && args.len() == 1
            {
                let arg = args[0];

                // Build 1/arg
                let one = ctx.num(1);
                let reciprocal = ctx.add(Expr::Div(one, arg));

                // Build arctan(1/arg)
                let result = ctx.call("arctan", vec![reciprocal]);

                return Some(Rewrite::new(result).desc("arccot(x) → arctan(1/x)"));
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
    // AtanAddRationalRule: Uses sub-sum guard (skips when parent is Add) to avoid
    // interfering with reciprocal pairs in larger sums.
    simplifier.add_rule(Box::new(AtanAddRationalRule));
    simplifier.add_rule(Box::new(InverseTrigNegativeRule));
    simplifier.add_rule(Box::new(ArcsecToArccosRule));
    simplifier.add_rule(Box::new(ArccscToArcsinRule));
    simplifier.add_rule(Box::new(ArccotToArctanRule));
    // PrincipalBranchInverseTrigRule: Self-gated by parent_ctx.inv_trig_policy().
    // Always registered; only applies when inv_trig == PrincipalValue.
    simplifier.add_rule(Box::new(PrincipalBranchInverseTrigRule));
}

// ==================== Principal Branch Rules (Educational) ====================
//
// These rules simplify inverse∘function compositions like arctan(tan(u)) → u.
// They are ONLY valid when u is in the principal domain, so they emit warnings.
//
// GATED BY: parent_ctx.inv_trig_policy() == InverseTrigPolicy::PrincipalValue
// This ensures these rules only fire when explicitly enabled via --inv-trig=principal

pub struct PrincipalBranchInverseTrigRule;

impl crate::rule::Rule for PrincipalBranchInverseTrigRule {
    fn name(&self) -> &str {
        "Principal Branch Inverse Trig"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        // GATE: Only apply when inv_trig policy is PrincipalValue
        if parent_ctx.inv_trig_policy() != crate::semantics::InverseTrigPolicy::PrincipalValue {
            return None;
        }

        if let Expr::Function(outer_name, outer_args) = ctx.get(expr) {
            if outer_args.len() != 1 {
                return None;
            }
            let inner = outer_args[0];

            // Extract function info from inner without cloning
            let inner_fn_info = if let Expr::Function(fn_id, args) = ctx.get(inner) {
                Some((*fn_id, args.clone()))
            } else {
                None
            };
            // Check for Div pattern
            let inner_div = if let Expr::Div(n, d) = ctx.get(inner) {
                Some((*n, *d))
            } else {
                None
            };

            // Pattern: arcsin(sin(u)) → u (assuming u ∈ [-π/2, π/2])
            if ctx.is_builtin(*outer_name, BuiltinFn::Arcsin) {
                if let Some((ref inner_name, ref inner_args)) = inner_fn_info {
                    if ctx.is_builtin(*inner_name, BuiltinFn::Sin) && inner_args.len() == 1 {
                        let u = inner_args[0];
                        return Some(
                            Rewrite::new(u)
                                .desc("arcsin(sin(u)) → u (principal branch)")
                                .local(expr, u)
                                .assume(
                                    crate::assumptions::AssumptionEvent::inv_trig_principal_range(
                                        ctx, "arcsin", u,
                                    ),
                                ),
                        );
                    }
                }
            }

            // Pattern: arccos(cos(u)) → u (assuming u ∈ [0, π])
            if ctx.is_builtin(*outer_name, BuiltinFn::Arccos) {
                if let Some((ref inner_name, ref inner_args)) = inner_fn_info {
                    if ctx.is_builtin(*inner_name, BuiltinFn::Cos) && inner_args.len() == 1 {
                        let u = inner_args[0];
                        return Some(
                            Rewrite::new(u)
                                .desc("arccos(cos(u)) → u (principal branch)")
                                .local(expr, u)
                                .assume(
                                    crate::assumptions::AssumptionEvent::inv_trig_principal_range(
                                        ctx, "arccos", u,
                                    ),
                                ),
                        );
                    }
                }
            }

            // Pattern: arctan(tan(u)) → u (assuming u ∈ (-π/2, π/2))
            if ctx.is_builtin(*outer_name, BuiltinFn::Arctan) {
                if let Some((ref inner_name, ref inner_args)) = inner_fn_info {
                    if ctx.is_builtin(*inner_name, BuiltinFn::Tan) && inner_args.len() == 1 {
                        let u = inner_args[0];
                        return Some(
                            Rewrite::new(u)
                                .desc("arctan(tan(u)) → u (principal branch)")
                                .local(expr, u)
                                .assume(
                                    crate::assumptions::AssumptionEvent::inv_trig_principal_range(
                                        ctx, "arctan", u,
                                    ),
                                ),
                        );
                    }
                }
            }

            // Pattern: arctan(sin(u)/cos(u)) → u (tan(u) in disguise)
            if ctx.is_builtin(*outer_name, BuiltinFn::Arctan) {
                if let Some((num, den)) = inner_div {
                    let num_fn_info = if let Expr::Function(n, a) = ctx.get(num) {
                        Some((*n, a.clone()))
                    } else {
                        None
                    };
                    let den_fn_info = if let Expr::Function(n, a) = ctx.get(den) {
                        Some((*n, a.clone()))
                    } else {
                        None
                    };
                    if let (Some((n_name, ref n_args)), Some((d_name, ref d_args))) =
                        (num_fn_info, den_fn_info)
                    {
                        if ctx.is_builtin(n_name, BuiltinFn::Sin)
                            && ctx.is_builtin(d_name, BuiltinFn::Cos)
                            && n_args.len() == 1
                            && d_args.len() == 1
                            && (n_args[0] == d_args[0]
                                || crate::ordering::compare_expr(ctx, n_args[0], d_args[0])
                                    == std::cmp::Ordering::Equal)
                        {
                            let u = n_args[0];
                            return Some(Rewrite::new(u)
                                .desc("arctan(sin(u)/cos(u)) → u (principal branch)")
                                .local(expr, u)
                                .assume(crate::assumptions::AssumptionEvent::inv_trig_principal_range(ctx, "arctan", u)));
                        }
                    }
                }
            }
        }
        None
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }

    fn priority(&self) -> i32 {
        0 // Default priority
    }
}
