//! Expansion and contraction rules for trigonometric expressions.

use crate::define_rule;
use crate::helpers::{as_div, as_mul, as_sub, extract_double_angle_arg};
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{BuiltinFn, Expr, ExprId};

// Import helpers from sibling modules (via re-exports in parent)
use super::{build_avg, build_half_diff, is_multiple_angle, normalize_for_even_fn};

// =============================================================================
// STANDALONE SUM-TO-PRODUCT RULE
// sin(A)+sin(B) → 2·sin((A+B)/2)·cos((A-B)/2)
// sin(A)-sin(B) → 2·cos((A+B)/2)·sin((A-B)/2)
// cos(A)+cos(B) → 2·cos((A+B)/2)·cos((A-B)/2)
// cos(A)-cos(B) → -2·sin((A+B)/2)·sin((A-B)/2)
// =============================================================================
// This rule applies sum-to-product identities to standalone sums/differences
// of trig functions (not inside quotients handled by SinCosSumQuotientRule).
//
// GATING: Only apply when both arguments are rational multiples of π, ensuring
// the transformed expression can be evaluated via trig table lookup (π/4, π/6, etc.)
// This prevents unnecessary expansion of symbolic expressions like sin(a)+sin(b).
//
// MATCHERS: Uses semantic TrigSumMatch (unordered) and TrigDiffMatch (ordered)
// to ensure correct sign handling for difference identities.
define_rule!(
    TrigSumToProductRule,
    "Sum-to-Product Identity",
    |ctx, expr| {
        use crate::helpers::{extract_rational_pi_multiple, match_trig_diff, match_trig_sum};

        // Try all four patterns
        enum Pattern {
            SinSum { arg1: ExprId, arg2: ExprId },
            SinDiff { a: ExprId, b: ExprId }, // ordered!
            CosSum { arg1: ExprId, arg2: ExprId },
            CosDiff { a: ExprId, b: ExprId }, // ordered!
        }

        let pattern = if let Some(m) = match_trig_sum(ctx, expr, "sin") {
            Pattern::SinSum {
                arg1: m.arg1,
                arg2: m.arg2,
            }
        } else if let Some(m) = match_trig_diff(ctx, expr, "sin") {
            Pattern::SinDiff { a: m.a, b: m.b }
        } else if let Some(m) = match_trig_sum(ctx, expr, "cos") {
            Pattern::CosSum {
                arg1: m.arg1,
                arg2: m.arg2,
            }
        } else if let Some(m) = match_trig_diff(ctx, expr, "cos") {
            Pattern::CosDiff { a: m.a, b: m.b }
        } else {
            return None;
        };

        // Extract (A, B) and the function name
        let (arg_a, arg_b, is_diff, fn_name) = match pattern {
            Pattern::SinSum { arg1, arg2 } => (arg1, arg2, false, "sin"),
            Pattern::SinDiff { a, b } => (a, b, true, "sin"),
            Pattern::CosSum { arg1, arg2 } => (arg1, arg2, false, "cos"),
            Pattern::CosDiff { a, b } => (a, b, true, "cos"),
        };

        // GATING: Only apply when BOTH arguments are rational multiples of π
        // This ensures the result can be simplified via trig table lookup
        let pi_a = extract_rational_pi_multiple(ctx, arg_a);
        let pi_b = extract_rational_pi_multiple(ctx, arg_b);
        if pi_a.is_none() || pi_b.is_none() {
            return None; // Don't expand symbolic sums
        }

        // Build avg = (A+B)/2 and half_diff = (A-B)/2
        let avg = build_avg(ctx, arg_a, arg_b);
        let half_diff = build_half_diff(ctx, arg_a, arg_b);
        let two = ctx.num(2);

        let (result, desc) = match (fn_name, is_diff) {
            // sin(A) + sin(B) → 2·sin(avg)·cos(half_diff)
            ("sin", false) => {
                let sin_avg = ctx.call("sin", vec![avg]);
                let cos_half = ctx.call("cos", vec![half_diff]);
                let product = smart_mul(ctx, sin_avg, cos_half);
                let result = smart_mul(ctx, two, product);
                (result, "sin(A)+sin(B) = 2·sin((A+B)/2)·cos((A-B)/2)")
            }
            // sin(A) - sin(B) → 2·cos(avg)·sin(half_diff)
            // Note: half_diff preserves order (A-B)/2 for correct sign
            ("sin", true) => {
                let cos_avg = ctx.call("cos", vec![avg]);
                let sin_half = ctx.call("sin", vec![half_diff]);
                let product = smart_mul(ctx, cos_avg, sin_half);
                let result = smart_mul(ctx, two, product);
                (result, "sin(A)-sin(B) = 2·cos((A+B)/2)·sin((A-B)/2)")
            }
            // cos(A) + cos(B) → 2·cos(avg)·cos(half_diff)
            ("cos", false) => {
                // For cos, half_diff sign doesn't matter (even function)
                let half_diff_normalized = normalize_for_even_fn(ctx, half_diff);
                let cos_avg = ctx.call("cos", vec![avg]);
                let cos_half = ctx.call("cos", vec![half_diff_normalized]);
                let product = smart_mul(ctx, cos_avg, cos_half);
                let result = smart_mul(ctx, two, product);
                (result, "cos(A)+cos(B) = 2·cos((A+B)/2)·cos((A-B)/2)")
            }
            // cos(A) - cos(B) → -2·sin(avg)·sin(half_diff)
            ("cos", true) => {
                let sin_avg = ctx.call("sin", vec![avg]);
                let sin_half = ctx.call("sin", vec![half_diff]);
                let product = smart_mul(ctx, sin_avg, sin_half);
                let two_product = smart_mul(ctx, two, product);
                let result = ctx.add(Expr::Neg(two_product));
                (result, "cos(A)-cos(B) = -2·sin((A+B)/2)·sin((A-B)/2)")
            }
            _ => return None,
        };

        Some(Rewrite::new(result).desc(desc))
    }
);

// =============================================================================
// HALF-ANGLE TANGENT RULE
// (1 - cos(2x)) / sin(2x) → tan(x)
// sin(2x) / (1 + cos(2x)) → tan(x)
// =============================================================================
// These are half-angle tangent identities derived from:
//   1 - cos(2x) = 2·sin²(x)
//   1 + cos(2x) = 2·cos²(x)
//   sin(2x) = 2·sin(x)·cos(x)
//
// DOMAIN WARNING: This transformation can extend the domain:
// - Pattern 1: Original requires sin(2x) ≠ 0, but tan(x) only requires cos(x) ≠ 0
// - Pattern 2: Original requires 1+cos(2x) ≠ 0, but tan(x) only requires cos(x) ≠ 0
//
// To preserve soundness, we introduce requires for cos(x) ≠ 0 (for tan(x) to be defined)
// and inherit the original denominator ≠ 0 condition.
//
// Uses SoundnessLabel::EquivalenceUnderIntroducedRequires
pub struct HalfAngleTangentRule;

impl crate::rule::Rule for HalfAngleTangentRule {
    fn name(&self) -> &str {
        "Half-Angle Tangent Identity"
    }

    fn priority(&self) -> i32 {
        50 // Normal priority
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::helpers::extract_double_angle_arg;
        use crate::implicit_domain::ImplicitCondition;

        // Only match Div nodes
        let (num_id, den_id) = as_div(ctx, expr)?;

        // Pattern 1: (1 - cos(2x)) / sin(2x) → tan(x)
        // Pattern 2: sin(2x) / (1 + cos(2x)) → tan(x)

        enum Pattern {
            OneMinusCosOverSin { x: ExprId, sin_2x: ExprId },
            SinOverOnePlusCos { x: ExprId, one_plus_cos_2x: ExprId },
        }

        let pattern = 'pattern: {
            // Try Pattern 1: (1 - cos(2x)) / sin(2x)
            // Numerator can be: Sub(1, cos(2x)) OR Add(1, Neg(cos(2x))) (canonicalized)

            // Helper to extract cos(2x) from either cos(2x) or Neg(cos(2x))
            let try_extract_cos_2x = |ctx: &cas_ast::Context,
                                      id: ExprId|
             -> Option<(ExprId, bool)> {
                if let Expr::Function(fn_id, args) = ctx.get(id) {
                    if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cos)) && args.len() == 1 {
                        return extract_double_angle_arg(ctx, args[0]).map(|x| (x, false));
                    }
                }
                // Check for Neg(cos(2x))
                if let Expr::Neg(inner) = ctx.get(id) {
                    if let Expr::Function(fn_id, args) = ctx.get(*inner) {
                        if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cos)) && args.len() == 1
                        {
                            return extract_double_angle_arg(ctx, args[0]).map(|x| (x, true));
                            // negated
                        }
                    }
                }
                None
            };

            // Check Sub(1, cos(2x))
            if let Expr::Sub(one_id, cos_id) = ctx.get(num_id) {
                if let Expr::Number(n) = ctx.get(*one_id) {
                    if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()) {
                        if let Some((x, false)) = try_extract_cos_2x(ctx, *cos_id) {
                            // Check if denominator is sin(2x) with same argument
                            if let Expr::Function(den_fn_id, den_args) = ctx.get(den_id) {
                                if matches!(ctx.builtin_of(*den_fn_id), Some(BuiltinFn::Sin))
                                    && den_args.len() == 1
                                {
                                    if let Some(x2) = extract_double_angle_arg(ctx, den_args[0]) {
                                        if crate::ordering::compare_expr(ctx, x, x2)
                                            == std::cmp::Ordering::Equal
                                        {
                                            break 'pattern Some(Pattern::OneMinusCosOverSin {
                                                x,
                                                sin_2x: den_id,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Check Add(1, Neg(cos(2x))) or Add(Neg(cos(2x)), 1) - canonicalized form
            if let Expr::Add(left, right) = ctx.get(num_id) {
                // Try left=1, right=Neg(cos)
                let try_order = |one: ExprId, neg_cos: ExprId| -> Option<ExprId> {
                    if let Expr::Number(n) = ctx.get(one) {
                        if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into())
                        {
                            if let Some((x, true)) = try_extract_cos_2x(ctx, neg_cos) {
                                return Some(x);
                            }
                        }
                    }
                    None
                };

                // Try both orders
                let x_opt = try_order(*left, *right).or_else(|| try_order(*right, *left));

                if let Some(x) = x_opt {
                    // Check if denominator is sin(2x) with same argument
                    if let Expr::Function(den_fn_id, den_args) = ctx.get(den_id) {
                        if matches!(ctx.builtin_of(*den_fn_id), Some(BuiltinFn::Sin))
                            && den_args.len() == 1
                        {
                            if let Some(x2) = extract_double_angle_arg(ctx, den_args[0]) {
                                if crate::ordering::compare_expr(ctx, x, x2)
                                    == std::cmp::Ordering::Equal
                                {
                                    break 'pattern Some(Pattern::OneMinusCosOverSin {
                                        x,
                                        sin_2x: den_id,
                                    });
                                }
                            }
                        }
                    }
                }
            }

            // Try Pattern 2: sin(2x) / (1 + cos(2x))
            // Numerator: sin(2x)
            if let Expr::Function(fn_id, args) = ctx.get(num_id) {
                if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Sin)) && args.len() == 1 {
                    if let Some(x) = extract_double_angle_arg(ctx, args[0]) {
                        // Denominator: 1 + cos(2x) or Add(1, cos(2x))
                        if let Expr::Add(left, right) = ctx.get(den_id) {
                            // Check both orders: 1 + cos(2x) or cos(2x) + 1
                            let (one_id, cos_id) = if matches!(ctx.get(*left), Expr::Number(n) if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()))
                            {
                                (*left, *right)
                            } else if matches!(ctx.get(*right), Expr::Number(n) if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()))
                            {
                                (*right, *left)
                            } else {
                                break 'pattern None;
                            };

                            // Verify one_id is 1
                            if let Expr::Number(n) = ctx.get(one_id) {
                                if n.is_integer()
                                    && *n == num_rational::BigRational::from_integer(1.into())
                                {
                                    // Check if cos_id is cos(2x) with same x
                                    if let Expr::Function(cos_fn_id, cos_args) = ctx.get(cos_id) {
                                        if matches!(
                                            ctx.builtin_of(*cos_fn_id),
                                            Some(BuiltinFn::Cos)
                                        ) && cos_args.len() == 1
                                        {
                                            if let Some(x2) =
                                                extract_double_angle_arg(ctx, cos_args[0])
                                            {
                                                if crate::ordering::compare_expr(ctx, x, x2)
                                                    == std::cmp::Ordering::Equal
                                                {
                                                    break 'pattern Some(
                                                        Pattern::SinOverOnePlusCos {
                                                            x,
                                                            one_plus_cos_2x: den_id,
                                                        },
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            None
        }?;

        // Build tan(x)
        let (x, denom_expr, desc) = match pattern {
            Pattern::OneMinusCosOverSin { x, sin_2x } => {
                (x, sin_2x, "(1 - cos(2x))/sin(2x) = tan(x)")
            }
            Pattern::SinOverOnePlusCos { x, one_plus_cos_2x } => {
                (x, one_plus_cos_2x, "sin(2x)/(1 + cos(2x)) = tan(x)")
            }
        };

        let tan_x = ctx.call("tan", vec![x]);

        // Build cos(x) for the NonZero require
        let cos_x = ctx.call("cos", vec![x]);

        // Create rewrite with requires:
        // 1. Original denominator ≠ 0 (inherited from the division)
        // 2. cos(x) ≠ 0 (for tan(x) to be defined)
        let rewrite = Rewrite::new(tan_x)
            .desc(desc)
            .requires(ImplicitCondition::NonZero(denom_expr))
            .requires(ImplicitCondition::NonZero(cos_x));

        Some(rewrite)
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Div"])
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn soundness(&self) -> crate::rule::SoundnessLabel {
        crate::rule::SoundnessLabel::EquivalenceUnderIntroducedRequires
    }
}

define_rule!(
    DoubleAngleRule,
    "Double Angle Identity",
    |ctx, expr, parent_ctx| {
        // GUARD: Don't expand double angle inside a Div context
        // This prevents sin(2x)/cos(2x) from being "polinomized" to a worse form.
        // Expansion should only happen when it helps simplification, not in canonical quotients.
        if parent_ctx
            .has_ancestor_matching(ctx, |c, id| matches!(c.get(id), cas_ast::Expr::Div(_, _)))
        {
            return None;
        }

        // GUARD: Don't expand when sin(4x) identity pattern is detected
        // This allows Sin4xIdentityZeroRule to see 4*sin*cos*cos(2t) intact
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.has_sin4x_identity_pattern {
                return None;
            }
        }

        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            if args.len() == 1 {
                // Check if arg is 2*x or x*2
                // We need to match "2 * x"
                if let Some(inner_var) = extract_double_angle_arg(ctx, args[0]) {
                    // GUARD: Anti-worsen for multiple angles.
                    // Don't expand sin(2*(8x)) = sin(16x) because the inner argument
                    // is already a multiple (8x). This would cause exponential recursion:
                    // sin(16x) → 2sin(8x)cos(8x) → 2·2sin(4x)cos(4x)·... = explosion
                    if is_multiple_angle(ctx, inner_var) {
                        return None;
                    }

                    match ctx.builtin_of(*fn_id) {
                        Some(BuiltinFn::Sin) => {
                            // sin(2x) -> 2sin(x)cos(x)
                            let two = ctx.num(2);
                            let sin_x = ctx.call("sin", vec![inner_var]);
                            let cos_x = ctx.call("cos", vec![inner_var]);
                            let sin_cos = smart_mul(ctx, sin_x, cos_x);
                            let new_expr = smart_mul(ctx, two, sin_cos);
                            return Some(Rewrite::new(new_expr).desc("sin(2x) -> 2sin(x)cos(x)"));
                        }
                        Some(BuiltinFn::Cos) => {
                            // cos(2x) -> cos^2(x) - sin^2(x)
                            let two = ctx.num(2);
                            let cos_x = ctx.call("cos", vec![inner_var]);
                            let cos2 = ctx.add(Expr::Pow(cos_x, two));

                            let sin_x = ctx.call("sin", vec![inner_var]);
                            let sin2 = ctx.add(Expr::Pow(sin_x, two));

                            let new_expr = ctx.add(Expr::Sub(cos2, sin2));
                            return Some(
                                Rewrite::new(new_expr).desc("cos(2x) -> cos^2(x) - sin^2(x)"),
                            );
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }
);

// Double Angle Contraction Rule: 2·sin(t)·cos(t) → sin(2t), cos²(t) - sin²(t) → cos(2t)
// This is the INVERSE of DoubleAngleRule - contracts expanded forms back to double angle.
// Essential for recognizing Weierstrass substitution identities.
pub struct DoubleAngleContractionRule;

impl crate::rule::Rule for DoubleAngleContractionRule {
    fn name(&self) -> &str {
        "Double Angle Contraction"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // GUARD: Don't contract when sin(4x) identity pattern is detected
        // This preserves 4*sin*cos*(cos²-sin²) for Sin4xIdentityZeroRule
        if let Some(marks) = _parent_ctx.pattern_marks() {
            if marks.has_sin4x_identity_pattern {
                return None;
            }
        }

        // Pattern 1: 2·sin(t)·cos(t) → sin(2t)
        // Matches Mul(2, Mul(sin(t), cos(t))) or Mul(Mul(2, sin(t)), cos(t)) etc.
        if let Some((l, r)) = as_mul(ctx, expr) {
            if let Some((sin_arg, cos_arg)) = self.extract_two_sin_cos(ctx, l, r) {
                // Check if sin and cos have the same argument
                if crate::ordering::compare_expr(ctx, sin_arg, cos_arg) == std::cmp::Ordering::Equal
                {
                    // Build sin(2*t)
                    let two = ctx.num(2);
                    let double_arg = ctx.add(Expr::Mul(two, sin_arg));
                    let sin_2t = ctx.call("sin", vec![double_arg]);
                    return Some(Rewrite::new(sin_2t).desc("2·sin(t)·cos(t) = sin(2t)"));
                }
            }
        }

        // Pattern 2: cos²(t) - sin²(t) → cos(2t)
        if let Some((l, r)) = as_sub(ctx, expr) {
            if let Some((cos_arg, sin_arg)) = self.extract_cos2_minus_sin2(ctx, l, r) {
                // Check if cos² and sin² have the same argument
                if crate::ordering::compare_expr(ctx, cos_arg, sin_arg) == std::cmp::Ordering::Equal
                {
                    // Build cos(2*t)
                    let two = ctx.num(2);
                    let double_arg = ctx.add(Expr::Mul(two, cos_arg));
                    let cos_2t = ctx.call("cos", vec![double_arg]);
                    return Some(Rewrite::new(cos_2t).desc("cos²(t) - sin²(t) = cos(2t)"));
                }
            }
        }

        None
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Mul", "Sub"])
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn priority(&self) -> i32 {
        200 // Run before expansion rules to prevent ping-pong
    }
}

impl DoubleAngleContractionRule {
    /// Extract (sin_arg, cos_arg) from 2·sin(t)·cos(t) pattern
    fn extract_two_sin_cos(
        &self,
        ctx: &cas_ast::Context,
        l: ExprId,
        r: ExprId,
    ) -> Option<(ExprId, ExprId)> {
        // Check all possible arrangements of 2, sin(t), cos(t)
        let two_rat = num_rational::BigRational::from_integer(2.into());

        // Case: Mul(2, Mul(sin, cos))
        if let Expr::Number(n) = ctx.get(l) {
            if *n == two_rat {
                if let Expr::Mul(a, b) = ctx.get(r) {
                    return self.extract_sin_cos_pair(ctx, *a, *b);
                }
            }
        }

        // Case: Mul(Mul(...), 2)
        if let Expr::Number(n) = ctx.get(r) {
            if *n == two_rat {
                if let Expr::Mul(a, b) = ctx.get(l) {
                    return self.extract_sin_cos_pair(ctx, *a, *b);
                }
            }
        }

        // Case: Mul(Mul(2, sin), cos) or Mul(Mul(2, cos), sin)
        if let Expr::Mul(inner_l, inner_r) = ctx.get(l) {
            if let Expr::Number(n) = ctx.get(*inner_l) {
                if *n == two_rat {
                    // inner_r is either sin or cos
                    return self.extract_trig_and_match(ctx, *inner_r, r);
                }
            }
            if let Expr::Number(n) = ctx.get(*inner_r) {
                if *n == two_rat {
                    return self.extract_trig_and_match(ctx, *inner_l, r);
                }
            }
        }

        // Case: Mul(sin, Mul(2, cos)) or similar
        if let Expr::Mul(inner_l, inner_r) = ctx.get(r) {
            if let Expr::Number(n) = ctx.get(*inner_l) {
                if *n == two_rat {
                    return self.extract_trig_and_match(ctx, *inner_r, l);
                }
            }
            if let Expr::Number(n) = ctx.get(*inner_r) {
                if *n == two_rat {
                    return self.extract_trig_and_match(ctx, *inner_l, l);
                }
            }
        }

        None
    }

    fn extract_sin_cos_pair(
        &self,
        ctx: &cas_ast::Context,
        a: ExprId,
        b: ExprId,
    ) -> Option<(ExprId, ExprId)> {
        // Check if a is sin and b is cos, or vice versa
        if let Expr::Function(fn_id_a, args_a) = ctx.get(a) {
            if let Expr::Function(fn_id_b, args_b) = ctx.get(b) {
                if args_a.len() == 1 && args_b.len() == 1 {
                    let builtin_a = ctx.builtin_of(*fn_id_a);
                    let builtin_b = ctx.builtin_of(*fn_id_b);
                    if matches!(builtin_a, Some(BuiltinFn::Sin))
                        && matches!(builtin_b, Some(BuiltinFn::Cos))
                    {
                        return Some((args_a[0], args_b[0]));
                    }
                    if matches!(builtin_a, Some(BuiltinFn::Cos))
                        && matches!(builtin_b, Some(BuiltinFn::Sin))
                    {
                        return Some((args_b[0], args_a[0]));
                    }
                }
            }
        }
        None
    }

    fn extract_trig_and_match(
        &self,
        ctx: &cas_ast::Context,
        trig1: ExprId,
        trig2: ExprId,
    ) -> Option<(ExprId, ExprId)> {
        if let Expr::Function(fn_id1, args1) = ctx.get(trig1) {
            if let Expr::Function(fn_id2, args2) = ctx.get(trig2) {
                if args1.len() == 1 && args2.len() == 1 {
                    let builtin1 = ctx.builtin_of(*fn_id1);
                    let builtin2 = ctx.builtin_of(*fn_id2);
                    if matches!(builtin1, Some(BuiltinFn::Sin))
                        && matches!(builtin2, Some(BuiltinFn::Cos))
                    {
                        return Some((args1[0], args2[0]));
                    }
                    if matches!(builtin1, Some(BuiltinFn::Cos))
                        && matches!(builtin2, Some(BuiltinFn::Sin))
                    {
                        return Some((args2[0], args1[0]));
                    }
                }
            }
        }
        None
    }

    /// Extract (cos_arg, sin_arg) from cos²(t) - sin²(t) pattern
    fn extract_cos2_minus_sin2(
        &self,
        ctx: &cas_ast::Context,
        l: ExprId,
        r: ExprId,
    ) -> Option<(ExprId, ExprId)> {
        // l should be cos²(t), r should be sin²(t)
        let two_rat = num_rational::BigRational::from_integer(2.into());

        if let Expr::Pow(base_l, exp_l) = ctx.get(l) {
            if let Expr::Number(n) = ctx.get(*exp_l) {
                if *n == two_rat {
                    if let Expr::Function(fn_id_l, args_l) = ctx.get(*base_l) {
                        if matches!(ctx.builtin_of(*fn_id_l), Some(BuiltinFn::Cos))
                            && args_l.len() == 1
                        {
                            // Check r is sin²
                            if let Expr::Pow(base_r, exp_r) = ctx.get(r) {
                                if let Expr::Number(m) = ctx.get(*exp_r) {
                                    if *m == two_rat {
                                        if let Expr::Function(fn_id_r, args_r) = ctx.get(*base_r) {
                                            if matches!(
                                                ctx.builtin_of(*fn_id_r),
                                                Some(BuiltinFn::Sin)
                                            ) && args_r.len() == 1
                                            {
                                                return Some((args_l[0], args_r[0]));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use crate::rules::trigonometry::identities::{
        AngleIdentityRule, EvaluateTrigRule, TanToSinCosRule,
    };
    use cas_ast::{Context, DisplayExpr};
    use cas_parser::parse;

    #[test]
    fn test_evaluate_trig_zero() {
        let mut ctx = Context::new();
        let rule = EvaluateTrigRule;

        // sin(0) -> 0
        let expr = parse("sin(0)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );

        // cos(0) -> 1
        let expr = parse("cos(0)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "1"
        );

        // tan(0) -> 0
        let expr = parse("tan(0)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
    }

    #[test]
    fn test_evaluate_trig_identities() {
        let mut ctx = Context::new();
        let rule = EvaluateTrigRule;

        // sin(-x) -> -sin(x)
        let expr = parse("sin(-x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "-sin(x)"
        );

        // cos(-x) -> cos(x)
        let expr = parse("cos(-x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "cos(x)"
        );

        // tan(-x) -> -tan(x)
        let expr = parse("tan(-x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "-tan(x)"
        );
    }

    #[test]
    fn test_trig_identities() {
        let mut ctx = Context::new();
        let rule = AngleIdentityRule;

        // sin(x + y)
        let expr = parse("sin(x + y)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert!(format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        )
        .contains("sin(x)"));

        // cos(x + y) -> cos(x)cos(y) - sin(x)sin(y)
        let expr = parse("cos(x + y)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        let res = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );
        assert!(res.contains("cos(x)"));
        assert!(res.contains("-"));

        // sin(x - y)
        let expr = parse("sin(x - y)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert!(format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        )
        .contains("-"));
    }

    #[test]
    fn test_tan_to_sin_cos() {
        let mut ctx = Context::new();
        let rule = TanToSinCosRule;
        let expr = parse("tan(x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "sin(x) / cos(x)"
        );
    }

    #[test]
    fn test_double_angle() {
        let mut ctx = Context::new();
        let rule = DoubleAngleRule;

        // sin(2x)
        let expr = parse("sin(2 * x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );
        // Check that result contains the key components, regardless of order
        assert!(
            result_str.contains("sin(x)"),
            "Result should contain sin(x), got: {}",
            result_str
        );
        assert!(
            result_str.contains("cos(x)"),
            "Result should contain cos(x), got: {}",
            result_str
        );
        assert!(
            result_str.contains("2") || result_str.contains("* 2") || result_str.contains("2 *"),
            "Result should contain 2, got: {}",
            result_str
        );

        // cos(2x)
        let expr = parse("cos(2 * x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert!(format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        )
        .contains("cos(x)^2 - sin(x)^2"));
    }

    #[test]
    fn test_evaluate_inverse_trig() {
        let mut ctx = Context::new();
        let rule = EvaluateTrigRule;

        // arcsin(0) -> 0
        let expr = parse("arcsin(0)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );

        // arccos(1) -> 0
        let expr = parse("arccos(1)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );

        // arcsin(1) -> pi/2
        // Note: pi/2 might be formatted as "pi / 2" or similar depending on Display impl
        let expr = parse("arcsin(1)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert!(format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        )
        .contains("pi"));
        assert!(format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        )
        .contains("2"));

        // arccos(0) -> pi/2
        let expr = parse("arccos(0)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert!(format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        )
        .contains("pi"));
        assert!(format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        )
        .contains("2"));
    }
}
