//! Trig values and specialized identity rules.

use crate::define_rule;
use crate::helpers::{as_add, as_div, as_mul, as_sub};
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use crate::rules::trigonometry::values::detect_special_angle;
use cas_ast::{BuiltinFn, Expr, ExprId};

// =============================================================================
// TRIPLE TANGENT PRODUCT IDENTITY
// tan(u) · tan(π/3 - u) · tan(π/3 + u) = tan(3u)
// =============================================================================

/// Matches tan(u)·tan(π/3+u)·tan(π/3-u) and simplifies to tan(3u).
/// Must run BEFORE TanToSinCosRule to prevent expansion.
pub struct TanTripleProductRule;

impl crate::rule::Rule for TanTripleProductRule {
    fn name(&self) -> &str {
        "Triple Tangent Product (π/3)"
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::MUL)
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::TRANSFORM
    }

    fn soundness(&self) -> crate::rule::SoundnessLabel {
        // This rule introduces requires (cos ≠ 0) for the tangent definitions
        crate::rule::SoundnessLabel::EquivalenceUnderIntroducedRequires
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        use crate::helpers::{as_fn1, flatten_mul_chain};

        // Flatten multiplication to get factors
        let factors = flatten_mul_chain(ctx, expr);

        // We need at least 3 factors
        if factors.len() < 3 {
            return None;
        }

        // Extract all tan(arg) functions
        let mut tan_args: Vec<(ExprId, ExprId)> = Vec::new(); // (factor_id, arg)
        for &factor in &factors {
            if let Some(arg) = as_fn1(ctx, factor, "tan") {
                tan_args.push((factor, arg));
            }
        }

        // We need exactly 3 tan factors
        if tan_args.len() != 3 {
            return None;
        }

        // Try each argument as the potential "u"
        for i in 0..3 {
            let u = tan_args[i].1;
            let (j, k) = match i {
                0 => (1, 2),
                1 => (0, 2),
                2 => (0, 1),
                _ => unreachable!(),
            };

            let arg_j = tan_args[j].1;
            let arg_k = tan_args[k].1;

            // Check both orderings: (u+π/3, π/3-u) or (π/3-u, u+π/3)
            let match1 = is_u_plus_pi_over_3(ctx, arg_j, u) && is_pi_over_3_minus_u(ctx, arg_k, u);
            let match2 = is_pi_over_3_minus_u(ctx, arg_j, u) && is_u_plus_pi_over_3(ctx, arg_k, u);

            if match1 || match2 {
                // Build tan(3u)
                let three = ctx.num(3);
                let three_u = smart_mul(ctx, three, u);
                let tan_3u = ctx.call("tan", vec![three_u]);

                // If there are other factors beyond the 3 tans, multiply them
                let other_factors: Vec<ExprId> = factors
                    .iter()
                    .copied()
                    .filter(|&f| f != tan_args[0].0 && f != tan_args[1].0 && f != tan_args[2].0)
                    .collect();

                let result = if other_factors.is_empty() {
                    // Wrap in __hold to prevent expansion
                    cas_ast::hold::wrap_hold(ctx, tan_3u)
                } else {
                    // Multiply tan(3u) with other factors
                    let held_tan = cas_ast::hold::wrap_hold(ctx, tan_3u);
                    let mut product = held_tan;
                    for &f in &other_factors {
                        product = smart_mul(ctx, product, f);
                    }
                    product
                };

                // Build domain conditions: cos(u), cos(u+π/3), cos(π/3−u) ≠ 0
                // These are required for the tangent functions to be defined
                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                let three = ctx.num(3);
                let pi_over_3 = ctx.add(Expr::Div(pi, three));
                let u_plus_pi3 = ctx.add(Expr::Add(u, pi_over_3));
                let pi3_minus_u = ctx.add(Expr::Sub(pi_over_3, u));
                let cos_u = ctx.call("cos", vec![u]);
                let cos_u_plus = ctx.call("cos", vec![u_plus_pi3]);
                let cos_pi3_minus = ctx.call("cos", vec![pi3_minus_u]);

                // Format u for display in substeps
                let u_str = cas_ast::DisplayExpr {
                    context: ctx,
                    id: u,
                }
                .to_string();

                return Some(
                    Rewrite::new(result)
                        .desc("tan(u)·tan(π/3+u)·tan(π/3−u) = tan(3u)")
                        .substep(
                            "Normalizar argumentos",
                            vec![format!(
                                "π/3 − u se representa como −u + π/3 para comparar como u + const"
                            )],
                        )
                        .substep(
                            "Reconocer patrón",
                            vec![
                                format!("Sea u = {}", u_str),
                                format!("Factores: tan(u), tan(u + π/3), tan(π/3 − u)"),
                            ],
                        )
                        .substep(
                            "Aplicar identidad",
                            vec![format!("tan(u)·tan(u + π/3)·tan(π/3 − u) = tan(3u)")],
                        )
                        .requires(crate::implicit_domain::ImplicitCondition::NonZero(cos_u))
                        .requires(crate::implicit_domain::ImplicitCondition::NonZero(
                            cos_u_plus,
                        ))
                        .requires(crate::implicit_domain::ImplicitCondition::NonZero(
                            cos_pi3_minus,
                        )),
                );
            }
        }

        None
    }
}

/// Check if expr equals u + π/3 (or π/3 + u)
fn is_u_plus_pi_over_3(ctx: &cas_ast::Context, expr: ExprId, u: ExprId) -> bool {
    if let Some((l, r)) = as_add(ctx, expr) {
        // Case: u + π/3
        if crate::ordering::compare_expr(ctx, l, u) == std::cmp::Ordering::Equal {
            return is_pi_over_3(ctx, r);
        }
        // Case: π/3 + u
        if crate::ordering::compare_expr(ctx, r, u) == std::cmp::Ordering::Equal {
            return is_pi_over_3(ctx, l);
        }
    }
    false
}

/// Check if expr equals π/3 - u (or -u + π/3 in canonicalized form)
fn is_pi_over_3_minus_u(ctx: &cas_ast::Context, expr: ExprId, u: ExprId) -> bool {
    // Pattern 1: Sub(π/3, u)
    if let Some((l, r)) = as_sub(ctx, expr) {
        if is_pi_over_3(ctx, l)
            && crate::ordering::compare_expr(ctx, r, u) == std::cmp::Ordering::Equal
        {
            return true;
        }
    }
    // Pattern 2: Add(π/3, Neg(u)) or Add(Neg(u), π/3) - canonicalized subtraction
    if let Some((l, r)) = as_add(ctx, expr) {
        // Add(π/3, Neg(u))
        if is_pi_over_3(ctx, l) {
            if let Expr::Neg(inner) = ctx.get(r) {
                if crate::ordering::compare_expr(ctx, *inner, u) == std::cmp::Ordering::Equal {
                    return true;
                }
            }
        }
        // Add(Neg(u), π/3)
        if is_pi_over_3(ctx, r) {
            if let Expr::Neg(inner) = ctx.get(l) {
                if crate::ordering::compare_expr(ctx, *inner, u) == std::cmp::Ordering::Equal {
                    return true;
                }
            }
        }
    }
    false
}

/// Check if an expression is π/3 (i.e., Div(π, 3) or canonicalized Mul(1/3, π))
fn is_pi_over_3(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    // Pattern 1: Div(π, 3)
    if let Some((num, den)) = as_div(ctx, expr) {
        if matches!(ctx.get(num), Expr::Constant(cas_ast::Constant::Pi)) {
            if let Expr::Number(n) = ctx.get(den) {
                if n.is_integer() && *n.numer() == 3.into() {
                    return true;
                }
            }
        }
    }

    // Pattern 2: Mul(Number(1/3), π) - canonicalized form from CanonicalizeDivRule
    if let Some((l, r)) = as_mul(ctx, expr) {
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

/// Runtime check: is this tan() part of a tan(u)·tan(π/3+u)·tan(π/3-u) triple product?
/// This is called during rule application to prevent TanToSinCosRule from expanding
/// tan() nodes that will be handled by TanTripleProductRule.
fn is_part_of_tan_triple_product(
    ctx: &cas_ast::Context,
    tan_expr: ExprId,
    parent_ctx: &crate::parent_context::ParentContext,
) -> bool {
    // Verify this is actually a tan() function
    if !matches!(ctx.get(tan_expr), Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, BuiltinFn::Tan) && args.len() == 1)
    {
        return false;
    }

    // Find the highest Mul ancestor in the chain
    // Ancestors are stored from furthest to closest: [great-grandparent, grandparent, parent]
    // We want to find the outermost Mul that contains this tan()
    let ancestors = parent_ctx.all_ancestors();

    // Find the first (earliest in list = highest in tree) Mul ancestor
    let mut mul_root: Option<ExprId> = None;
    for &ancestor in ancestors {
        if matches!(ctx.get(ancestor), Expr::Mul(_, _)) {
            mul_root = Some(ancestor);
            break; // Take the highest Mul (first in ancestor list)
        }
    }

    let Some(mul_root) = mul_root else {
        return false;
    };

    // Flatten the Mul to get all factors
    let mut factors = Vec::new();
    let mut stack = vec![mul_root];
    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Mul(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            _ => factors.push(id),
        }
    }

    // Collect tan() arguments
    let mut tan_args: Vec<ExprId> = Vec::new();
    for &factor in &factors {
        if let Expr::Function(fn_id, args) = ctx.get(factor) {
            if ctx.is_builtin(*fn_id, BuiltinFn::Tan) && args.len() == 1 {
                tan_args.push(args[0]);
            }
        }
    }

    // Need exactly 3 tan() factors for triple product
    if tan_args.len() != 3 {
        return false;
    }

    // Check if they form the triple product pattern {u, u+π/3, π/3-u}
    for i in 0..3 {
        let u = tan_args[i];
        let others: Vec<_> = tan_args
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, &arg)| arg)
            .collect();

        let arg_j = others[0];
        let arg_k = others[1];

        // Check both orderings
        let match1 = is_u_plus_pi_over_3(ctx, arg_j, u) && is_pi_over_3_minus_u(ctx, arg_k, u);
        let match2 = is_pi_over_3_minus_u(ctx, arg_j, u) && is_u_plus_pi_over_3(ctx, arg_k, u);

        if match1 || match2 {
            return true;
        }
    }

    false
}

/// Check if an expression is a "multiple angle" pattern: n*x where n is integer > 1.
/// This is used to gate tan(n*x) → sin/cos expansion, which leads to complexity explosion
/// via triple-angle formulas.
pub fn is_multiple_angle(ctx: &cas_ast::Context, arg: ExprId) -> bool {
    use cas_ast::Expr;

    // Pattern: Mul(Number(n), x) or Mul(x, Number(n)) where n is integer > 1
    if let Expr::Mul(l, r) = ctx.get(arg) {
        // Check left side for integer > 1
        if let Expr::Number(n) = ctx.get(*l) {
            if n.is_integer() {
                let val = n.numer().clone();
                if val > 1.into() || val < (-1).into() {
                    return true;
                }
            }
        }
        // Check right side for integer > 1
        if let Expr::Number(n) = ctx.get(*r) {
            if n.is_integer() {
                let val = n.numer().clone();
                if val > 1.into() || val < (-1).into() {
                    return true;
                }
            }
        }
    }

    false
}

/// Check if an expression has a "large coefficient" pattern: n*x where |n| > 2.
/// This guards against exponential explosion in trig expansions.
/// sin(16*x) would trigger this, blocking sin(a+b) decomposition.
pub fn has_large_coefficient(ctx: &cas_ast::Context, arg: ExprId) -> bool {
    use cas_ast::Expr;

    // Pattern: Mul(Number(n), x) or Mul(x, Number(n)) where |n| > 2
    if let Expr::Mul(l, r) = ctx.get(arg) {
        let check_large = |id: ExprId| -> bool {
            if let Expr::Number(n) = ctx.get(id) {
                if n.is_integer() {
                    let val = n.numer().clone();
                    val > num_bigint::BigInt::from(2) || val < num_bigint::BigInt::from(-2)
                } else {
                    false
                }
            } else {
                false
            }
        };
        // Check both sides
        if check_large(*l) || check_large(*r) {
            return true;
        }
    }

    // Also check for Add/Sub patterns that contain multiples
    // This catches sin(13x + 3x) patterns
    if let Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) = ctx.get(arg) {
        if is_multiple_angle(ctx, *lhs) || is_multiple_angle(ctx, *rhs) {
            return true;
        }
    }

    false
}

/// Convert tan(x) to sin(x)/cos(x) UNLESS it's part of a Pythagorean pattern
pub struct TanToSinCosRule;

impl crate::rule::Rule for TanToSinCosRule {
    fn name(&self) -> &str {
        "Tan to Sin/Cos"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use cas_ast::Expr;

        // GUARD: Check pattern_marks - don't convert if protected
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.is_pythagorean_protected(expr) {
                return None; // Skip conversion - part of Pythagorean identity
            }
            // Inverse trig pattern protection is UNCONDITIONAL.
            // We always preserve arctan(tan(x)), arcsin(sin(x)), etc.
            // The policy only controls whether it SIMPLIFIES to x, not whether we expand it.
            if marks.is_inverse_trig_protected(expr) {
                return None; // Preserve pattern: arctan(tan(x)) stays as-is
            }
            // Tan triple product protection: tan(u)·tan(π/3+u)·tan(π/3-u) = tan(3u)
            // Don't expand tan() if it's part of this pattern - let TanTripleProductRule handle it.
            if marks.is_tan_triple_product_protected(expr) {
                return None;
            }
            // Identity cancellation protection: tan(a-b) - (tan(a)-tan(b))/(1+tan(a)*tan(b))
            // Don't expand tan() if part of this pattern - let TanDifferenceIdentityZeroRule handle it.
            if marks.is_identity_cancellation_protected(expr) {
                return None;
            }
            // Global flag: if ANY tan identity pattern was detected, block ALL tan→sin/cos expansion
            // This is needed because ExprIds change during bottom-up simplification
            if marks.has_tan_identity_pattern {
                return None;
            }
        }

        // GUARD: Also check immediate parent for inverse trig composition.
        // This is a fallback in case pattern_marks wasn't pre-scanned.
        if let Some(parent_id) = parent_ctx.immediate_parent() {
            if let Expr::Function(fn_id, _) = ctx.get(parent_id) {
                if matches!(
                    ctx.builtin_of(*fn_id),
                    Some(BuiltinFn::Arctan | BuiltinFn::Arcsin | BuiltinFn::Arccos)
                ) {
                    return None; // Preserve arctan(tan(x)) pattern
                }
            }
        }

        // GUARD: Runtime check for triple product pattern.
        // If this tan() is inside a Mul that forms tan(u)·tan(π/3+u)·tan(π/3-u), don't expand.
        // This works even after ExprIds change from canonicalization because we check the
        // current structure, not pre-scanned marks.
        if is_part_of_tan_triple_product(ctx, expr, parent_ctx) {
            return None; // Let TanTripleProductRule handle it
        }

        // GUARD: Anti-worsen for multiple angles.
        // Don't expand tan(n*x) for integer n > 1, as it leads to explosive
        // triple-angle formulas: tan(3x) → (3sin(x) - 4sin³(x))/(4cos³(x) - 3cos(x))
        // This is almost never useful for simplification.
        let (fn_id, args) = match ctx.get(expr) {
            Expr::Function(fn_id, args) => (*fn_id, args.clone()),
            _ => return None,
        };
        if matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Tan)) && args.len() == 1 {
            // GUARD: Don't expand tan(n*x) - causes complexity explosion
            if is_multiple_angle(ctx, args[0]) {
                return None;
            }
            // GUARD: Don't expand tan at special angles that have known values
            // Let EvaluateTrigTableRule handle these instead
            if detect_special_angle(ctx, args[0]).is_some() {
                return None;
            }
        }

        // Original conversion logic
        if matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Tan)) && args.len() == 1 {
            // tan(x) -> sin(x) / cos(x)
            let sin_x = ctx.call("sin", vec![args[0]]);
            let cos_x = ctx.call("cos", vec![args[0]]);
            let new_expr = ctx.add(Expr::Div(sin_x, cos_x));
            return Some(crate::rule::Rewrite::new(new_expr).desc("tan(x) -> sin(x)/cos(x)"));
        }
        None
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        // Exclude PostCleanup to avoid cycle with TrigQuotientRule
        // TanToSinCos expands for algebra, TrigQuotient reconverts to canonical form
        // NOTE: CORE is included because some tests (e.g., test_tangent_sum) need tan→sin/cos expansion
        // TanTripleProductRule is registered BEFORE this rule and will handle triple product patterns
        crate::phase::PhaseMask::CORE
            | crate::phase::PhaseMask::TRANSFORM
            | crate::phase::PhaseMask::RATIONALIZE
    }
}

/// Convert trig quotients to their canonical function forms:
/// - sin(x)/cos(x) → tan(x)
/// - cos(x)/sin(x) → cot(x)
/// - 1/sin(x) → csc(x)
/// - 1/cos(x) → sec(x)
/// - 1/tan(x) → cot(x)
pub struct TrigQuotientRule;

impl crate::rule::Rule for TrigQuotientRule {
    fn name(&self) -> &str {
        "Trig Quotient"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use cas_ast::Expr;

        let (num, den) = as_div(ctx, expr)?;

        // Extract function info from num and den without cloning
        // We need fn_id and args for both, so extract them in scoped borrows
        let num_fn_info = if let Expr::Function(fn_id, args) = ctx.get(num) {
            Some((*fn_id, args.clone()))
        } else {
            None
        };
        let den_fn_info = if let Expr::Function(fn_id, args) = ctx.get(den) {
            Some((*fn_id, args.clone()))
        } else {
            None
        };

        // Pattern: sin(x)/cos(x) → tan(x)
        if let (Some((num_fn_id, ref num_args)), Some((den_fn_id, ref den_args))) =
            (&num_fn_info, &den_fn_info)
        {
            let num_builtin = ctx.builtin_of(*num_fn_id);
            let den_builtin = ctx.builtin_of(*den_fn_id);
            if matches!(num_builtin, Some(BuiltinFn::Sin))
                && matches!(den_builtin, Some(BuiltinFn::Cos))
                && num_args.len() == 1
                && den_args.len() == 1
                && crate::ordering::compare_expr(ctx, num_args[0], den_args[0])
                    == std::cmp::Ordering::Equal
            {
                let tan_x = ctx.call("tan", vec![num_args[0]]);
                return Some(crate::rule::Rewrite::new(tan_x).desc("sin(x)/cos(x) → tan(x)"));
            }

            // Pattern: cos(x)/sin(x) → cot(x)
            if matches!(num_builtin, Some(BuiltinFn::Cos))
                && matches!(den_builtin, Some(BuiltinFn::Sin))
                && num_args.len() == 1
                && den_args.len() == 1
                && crate::ordering::compare_expr(ctx, num_args[0], den_args[0])
                    == std::cmp::Ordering::Equal
            {
                let cot_x = ctx.call("cot", vec![num_args[0]]);
                return Some(crate::rule::Rewrite::new(cot_x).desc("cos(x)/sin(x) → cot(x)"));
            }
        }

        // Pattern: 1/sin(x) → csc(x)
        if crate::helpers::is_one(ctx, num) {
            if let Some((den_fn_id, ref den_args)) = den_fn_info {
                let den_builtin = ctx.builtin_of(den_fn_id);
                if matches!(den_builtin, Some(BuiltinFn::Sin)) && den_args.len() == 1 {
                    let csc_x = ctx.call("csc", vec![den_args[0]]);
                    return Some(crate::rule::Rewrite::new(csc_x).desc("1/sin(x) → csc(x)"));
                }
                if matches!(den_builtin, Some(BuiltinFn::Cos)) && den_args.len() == 1 {
                    let sec_x = ctx.call("sec", vec![den_args[0]]);
                    return Some(crate::rule::Rewrite::new(sec_x).desc("1/cos(x) → sec(x)"));
                }
                if matches!(den_builtin, Some(BuiltinFn::Tan)) && den_args.len() == 1 {
                    let cot_x = ctx.call("cot", vec![den_args[0]]);
                    return Some(crate::rule::Rewrite::new(cot_x).desc("1/tan(x) → cot(x)"));
                }
            }
        }

        None
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::DIV)
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        // Only run in PostCleanup to avoid cycle with TanToSinCosRule
        crate::phase::PhaseMask::POST
    }
}

// Secant-Tangent Pythagorean Identity: sec²(x) - tan²(x) = 1
// Also recognizes factored form: (sec(x) + tan(x)) * (sec(x) - tan(x)) = 1
define_rule!(
    SecTanPythagoreanRule,
    "Secant-Tangent Pythagorean Identity",
    |ctx, expr| {
        use crate::pattern_detection::{is_sec_squared, is_tan_squared};

        let (left, right) = as_add(ctx, expr)?;
        // Try both orderings: Add(sec², Neg(tan²)) or Add(Neg(tan²), sec²)
        for (pos, neg) in [(left, right), (right, left)] {
            if let Expr::Neg(neg_inner) = ctx.get(neg) {
                // Check if pos=sec²  and neg_inner=tan²
                if let (Some(sec_arg), Some(tan_arg)) =
                    (is_sec_squared(ctx, pos), is_tan_squared(ctx, *neg_inner))
                {
                    if crate::ordering::compare_expr(ctx, sec_arg, tan_arg)
                        == std::cmp::Ordering::Equal
                    {
                        return Some(Rewrite::new(ctx.num(1)).desc("sec²(x) - tan²(x) = 1"));
                    }
                }
            }
        }

        None
    }
);

// Cosecant-Cotangent Pythagorean Identity: csc²(x) - cot²(x) = 1
// NOTE: Subtraction is normalized to Add(a, Neg(b))
define_rule!(
    CscCotPythagoreanRule,
    "Cosecant-Cotangent Pythagorean Identity",
    |ctx, expr| {
        use crate::pattern_detection::{is_cot_squared, is_csc_squared};

        let (left, right) = as_add(ctx, expr)?;
        for (pos, neg) in [(left, right), (right, left)] {
            if let Expr::Neg(neg_inner) = ctx.get(neg) {
                // Check if pos=csc² and neg_inner=cot²
                if let (Some(csc_arg), Some(cot_arg)) =
                    (is_csc_squared(ctx, pos), is_cot_squared(ctx, *neg_inner))
                {
                    if crate::ordering::compare_expr(ctx, csc_arg, cot_arg)
                        == std::cmp::Ordering::Equal
                    {
                        return Some(Rewrite::new(ctx.num(1)).desc("csc²(x) - cot²(x) = 1"));
                    }
                }
            }
        }

        None
    }
);
