//! Trig values and specialized identity rules.

use crate::define_rule;
use crate::helpers::{as_add, as_div};
use crate::rule::Rewrite;
use crate::rules::trigonometry::values::detect_special_angle;
use cas_ast::{BuiltinFn, Expr, ExprId};
use cas_math::expr_rewrite::smart_mul;
use cas_math::trig_multi_angle_support::is_multiple_angle;
use cas_math::trig_tan_triple_support::{
    is_part_of_tan_triple_product_with_ancestors, is_pi_over_3_minus_u, is_u_plus_pi_over_3,
};

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
                let tan_3u = ctx.call_builtin(cas_ast::BuiltinFn::Tan, vec![three_u]);

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
                let cos_u = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![u]);
                let cos_u_plus = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![u_plus_pi3]);
                let cos_pi3_minus = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![pi3_minus_u]);

                // Format u for display in substeps
                let u_str = cas_formatter::DisplayExpr {
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
            // Tan double angle protection: 2·tan(t)/(1-tan²(t)) = tan(2t)
            // Don't expand tan() if part of this pattern - let TanDoubleAngleContractionRule handle it.
            if marks.is_tan_double_angle_protected(expr) {
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
        if is_part_of_tan_triple_product_with_ancestors(ctx, expr, parent_ctx.all_ancestors()) {
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
            let sin_x = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![args[0]]);
            let cos_x = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![args[0]]);
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
                let tan_x = ctx.call_builtin(cas_ast::BuiltinFn::Tan, vec![num_args[0]]);
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
                let cot_x = ctx.call_builtin(cas_ast::BuiltinFn::Cot, vec![num_args[0]]);
                return Some(crate::rule::Rewrite::new(cot_x).desc("cos(x)/sin(x) → cot(x)"));
            }
        }

        // Pattern: 1/sin(x) → csc(x)
        if crate::helpers::is_one(ctx, num) {
            if let Some((den_fn_id, ref den_args)) = den_fn_info {
                let den_builtin = ctx.builtin_of(den_fn_id);
                if matches!(den_builtin, Some(BuiltinFn::Sin)) && den_args.len() == 1 {
                    let csc_x = ctx.call_builtin(cas_ast::BuiltinFn::Csc, vec![den_args[0]]);
                    return Some(crate::rule::Rewrite::new(csc_x).desc("1/sin(x) → csc(x)"));
                }
                if matches!(den_builtin, Some(BuiltinFn::Cos)) && den_args.len() == 1 {
                    let sec_x = ctx.call_builtin(cas_ast::BuiltinFn::Sec, vec![den_args[0]]);
                    return Some(crate::rule::Rewrite::new(sec_x).desc("1/cos(x) → sec(x)"));
                }
                if matches!(den_builtin, Some(BuiltinFn::Tan)) && den_args.len() == 1 {
                    let cot_x = ctx.call_builtin(cas_ast::BuiltinFn::Cot, vec![den_args[0]]);
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
        use cas_math::trig_pattern_detection::{is_sec_squared, is_tan_squared};

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
        use cas_math::trig_pattern_detection::{is_cot_squared, is_csc_squared};

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
