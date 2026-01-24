//! Expansion and contraction rules for trigonometric expressions.

use crate::define_rule;
use crate::helpers::{extract_double_angle_arg, extract_triple_angle_arg};
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{Expr, ExprId};
use num_traits::{One, Zero};

// Import helpers from sibling modules (via re-exports in parent)
use super::{
    build_avg, build_half_diff, extract_trig_arg, is_multiple_angle, normalize_for_even_fn,
};

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
        let Expr::Div(num_id, den_id) = ctx.get(expr).clone() else {
            return None;
        };

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
            let try_extract_cos_2x =
                |ctx: &cas_ast::Context, id: ExprId| -> Option<(ExprId, bool)> {
                    if let Expr::Function(fn_id, args) = ctx.get(id) {
                        let name = ctx.sym_name(*fn_id);
                        if name == "cos" && args.len() == 1 {
                            return extract_double_angle_arg(ctx, args[0]).map(|x| (x, false));
                        }
                    }
                    // Check for Neg(cos(2x))
                    if let Expr::Neg(inner) = ctx.get(id) {
                        if let Expr::Function(fn_id, args) = ctx.get(*inner) {
                            let name = ctx.sym_name(*fn_id);
                            if name == "cos" && args.len() == 1 {
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
                                let den_name = ctx.sym_name(*den_fn_id);
                                if den_name == "sin" && den_args.len() == 1 {
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
                        let den_name = ctx.sym_name(*den_fn_id);
                        if den_name == "sin" && den_args.len() == 1 {
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
                let name = ctx.sym_name(*fn_id);
                if name == "sin" && args.len() == 1 {
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
                                        let cos_name_str = ctx.sym_name(*cos_fn_id);
                                        if cos_name_str == "cos" && cos_args.len() == 1 {
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
            let name = ctx.sym_name(*fn_id);
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

                    match ctx.sym_name(*fn_id) {
                        "sin" => {
                            // sin(2x) -> 2sin(x)cos(x)
                            let two = ctx.num(2);
                            let sin_x = ctx.call("sin", vec![inner_var]);
                            let cos_x = ctx.call("cos", vec![inner_var]);
                            let sin_cos = smart_mul(ctx, sin_x, cos_x);
                            let new_expr = smart_mul(ctx, two, sin_cos);
                            return Some(Rewrite::new(new_expr).desc("sin(2x) -> 2sin(x)cos(x)"));
                        }
                        "cos" => {
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
        if let Expr::Mul(l, r) = ctx.get(expr).clone() {
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
        if let Expr::Sub(l, r) = ctx.get(expr).clone() {
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
                    let name_a = ctx.sym_name(*fn_id_a);
                    let name_b = ctx.sym_name(*fn_id_b);
                    if name_a == "sin" && name_b == "cos" {
                        return Some((args_a[0], args_b[0]));
                    }
                    if name_a == "cos" && name_b == "sin" {
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
                    let name1 = ctx.sym_name(*fn_id1);
                    let name2 = ctx.sym_name(*fn_id2);
                    if name1 == "sin" && name2 == "cos" {
                        return Some((args1[0], args2[0]));
                    }
                    if name1 == "cos" && name2 == "sin" {
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
                        let name_l = ctx.sym_name(*fn_id_l);
                        if name_l == "cos" && args_l.len() == 1 {
                            // Check r is sin²
                            if let Expr::Pow(base_r, exp_r) = ctx.get(r) {
                                if let Expr::Number(m) = ctx.get(*exp_r) {
                                    if *m == two_rat {
                                        if let Expr::Function(fn_id_r, args_r) = ctx.get(*base_r) {
                                            let name_r = ctx.sym_name(*fn_id_r);
                                            if name_r == "sin" && args_r.len() == 1 {
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

// Triple Angle Shortcut Rule: sin(3x) → 3sin(x) - 4sin³(x), cos(3x) → 4cos³(x) - 3cos(x)
// This is a performance optimization to avoid recursive expansion via double-angle rules.
// Reduces ~23 rewrites to ~3-5 for triple angle expressions.
define_rule!(
    TripleAngleRule,
    "Triple Angle Identity",
    |ctx, expr, parent_ctx| {
        // GUARD 1: Skip if marked for protection
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.is_sum_quotient_protected(expr) {
                return None;
            }
        }

        // GUARD 2: Skip if inside sum-quotient pattern (defer to SinCosSumQuotientRule)
        if is_inside_trig_quotient_pattern(ctx, expr, parent_ctx) {
            return None;
        }

        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            let name = ctx.sym_name(*fn_id);
            if args.len() == 1 {
                // Check if arg is 3*x or x*3
                if let Some(inner_var) = extract_triple_angle_arg(ctx, args[0]) {
                    match ctx.sym_name(*fn_id) {
                        "sin" => {
                            // sin(3x) → 3sin(x) - 4sin³(x)
                            let three = ctx.num(3);
                            let four = ctx.num(4);
                            let exp_three = ctx.num(3); // Separate for Pow exponent
                            let sin_x = ctx.call("sin", vec![inner_var]);

                            // 3*sin(x)
                            let term1 = smart_mul(ctx, three, sin_x);

                            // sin³(x) = sin(x)^3
                            let sin_cubed = ctx.add(Expr::Pow(sin_x, exp_three));
                            // 4*sin³(x)
                            let term2 = smart_mul(ctx, four, sin_cubed);

                            // 3sin(x) - 4sin³(x)
                            let new_expr = ctx.add(Expr::Sub(term1, term2));
                            return Some(
                                Rewrite::new(new_expr).desc("sin(3x) → 3sin(x) - 4sin³(x)"),
                            );
                        }
                        "cos" => {
                            // cos(3x) → 4cos³(x) - 3cos(x)
                            let three = ctx.num(3);
                            let four = ctx.num(4);
                            let exp_three = ctx.num(3); // Separate for Pow exponent
                            let cos_x = ctx.call("cos", vec![inner_var]);

                            // cos³(x) = cos(x)^3
                            let cos_cubed = ctx.add(Expr::Pow(cos_x, exp_three));
                            // 4*cos³(x)
                            let term1 = smart_mul(ctx, four, cos_cubed);

                            // 3*cos(x)
                            let term2 = smart_mul(ctx, three, cos_x);

                            // 4cos³(x) - 3cos(x)
                            let new_expr = ctx.add(Expr::Sub(term1, term2));
                            return Some(
                                Rewrite::new(new_expr).desc("cos(3x) → 4cos³(x) - 3cos(x)"),
                            );
                        }
                        "tan" => {
                            // tan(3x) → (3tan(x) - tan³(x)) / (1 - 3tan²(x))
                            let one = ctx.num(1);
                            let three = ctx.num(3);
                            let exp_two = ctx.num(2);
                            let exp_three = ctx.num(3);
                            let tan_x = ctx.call("tan", vec![inner_var]);

                            // Numerator: 3tan(x) - tan³(x)
                            let three_tan = smart_mul(ctx, three, tan_x);
                            let tan_cubed = ctx.add(Expr::Pow(tan_x, exp_three));
                            let numer = ctx.add(Expr::Sub(three_tan, tan_cubed));

                            // Denominator: 1 - 3tan²(x)
                            let tan_squared = ctx.add(Expr::Pow(tan_x, exp_two));
                            let three_tan_squared = smart_mul(ctx, three, tan_squared);
                            let denom = ctx.add(Expr::Sub(one, three_tan_squared));

                            let new_expr = ctx.add(Expr::Div(numer, denom));
                            return Some(
                                Rewrite::new(new_expr)
                                    .desc("tan(3x) → (3tan(x) - tan³(x))/(1 - 3tan²(x))"),
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

// Quintuple Angle Rule: sin(5x) → 16sin⁵(x) - 20sin³(x) + 5sin(x)
// This is a direct expansion to avoid recursive explosion via double/triple angle.
define_rule!(
    QuintupleAngleRule,
    "Quintuple Angle Identity",
    |ctx, expr, parent_ctx| {
        // GUARD 1: Skip if marked for protection
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.is_sum_quotient_protected(expr) {
                return None;
            }
        }

        // GUARD 2: Skip if inside sum-quotient pattern
        if is_inside_trig_quotient_pattern(ctx, expr, parent_ctx) {
            return None;
        }

        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            let name = ctx.sym_name(*fn_id);
            if args.len() == 1 {
                // Check if arg is 5*x or x*5
                if let Some(inner_var) = crate::helpers::extract_quintuple_angle_arg(ctx, args[0]) {
                    match ctx.sym_name(*fn_id) {
                        "sin" => {
                            // sin(5x) → 16sin⁵(x) - 20sin³(x) + 5sin(x)
                            let five = ctx.num(5);
                            let sixteen = ctx.num(16);
                            let twenty = ctx.num(20);
                            let exp_three = ctx.num(3);
                            let exp_five = ctx.num(5);
                            let sin_x = ctx.call("sin", vec![inner_var]);

                            // 16sin⁵(x)
                            let sin_5 = ctx.add(Expr::Pow(sin_x, exp_five));
                            let term1 = smart_mul(ctx, sixteen, sin_5);

                            // 20sin³(x)
                            let sin_3 = ctx.add(Expr::Pow(sin_x, exp_three));
                            let term2 = smart_mul(ctx, twenty, sin_3);

                            // 5sin(x)
                            let term3 = smart_mul(ctx, five, sin_x);

                            // 16sin⁵(x) - 20sin³(x) + 5sin(x)
                            let sub1 = ctx.add(Expr::Sub(term1, term2));
                            let new_expr = ctx.add(Expr::Add(sub1, term3));
                            return Some(
                                Rewrite::new(new_expr)
                                    .desc("sin(5x) → 16sin⁵(x) - 20sin³(x) + 5sin(x)"),
                            );
                        }
                        "cos" => {
                            // cos(5x) → 16cos⁵(x) - 20cos³(x) + 5cos(x)
                            let five = ctx.num(5);
                            let sixteen = ctx.num(16);
                            let twenty = ctx.num(20);
                            let exp_three = ctx.num(3);
                            let exp_five = ctx.num(5);
                            let cos_x = ctx.call("cos", vec![inner_var]);

                            // 16cos⁵(x)
                            let cos_5 = ctx.add(Expr::Pow(cos_x, exp_five));
                            let term1 = smart_mul(ctx, sixteen, cos_5);

                            // 20cos³(x)
                            let cos_3 = ctx.add(Expr::Pow(cos_x, exp_three));
                            let term2 = smart_mul(ctx, twenty, cos_3);

                            // 5cos(x)
                            let term3 = smart_mul(ctx, five, cos_x);

                            // 16cos⁵(x) - 20cos³(x) + 5cos(x)
                            let sub1 = ctx.add(Expr::Sub(term1, term2));
                            let new_expr = ctx.add(Expr::Add(sub1, term3));
                            return Some(
                                Rewrite::new(new_expr)
                                    .desc("cos(5x) → 16cos⁵(x) - 20cos³(x) + 5cos(x)"),
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

/// Check if a trig function is inside a potential sum-quotient pattern
/// (sin(A)±sin(B)) / (cos(A)±cos(B))
/// Returns true if expansion should be deferred to SinCosSumQuotientRule
fn is_inside_trig_quotient_pattern(
    ctx: &cas_ast::Context,
    _expr: ExprId,
    parent_ctx: &crate::parent_context::ParentContext,
) -> bool {
    // Check if any ancestor is a Div with the sum-quotient pattern
    parent_ctx.has_ancestor_matching(ctx, |c, id| {
        if let Expr::Div(num, den) = c.get(id) {
            // Check if numerator is Add or Sub of sin functions
            let num_is_sin_sum_or_diff = is_binary_trig_op(c, *num, "sin");
            // Check if denominator is Add of cos functions
            let den_is_cos_sum = is_trig_sum(c, *den, "cos");
            num_is_sin_sum_or_diff && den_is_cos_sum
        } else {
            false
        }
    })
}

/// Check if expr is Add(trig(A), trig(B)) or Sub(trig(A), trig(B)) or Add(trig(A), Neg(trig(B)))
fn is_binary_trig_op(ctx: &cas_ast::Context, expr: ExprId, fn_name: &str) -> bool {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            // Check for Add(sin(A), sin(B))
            if extract_trig_arg(ctx, *l, fn_name).is_some()
                && extract_trig_arg(ctx, *r, fn_name).is_some()
            {
                return true;
            }
            // Check for Add(sin(A), Neg(sin(B)))
            if let Expr::Neg(inner) = ctx.get(*r) {
                if extract_trig_arg(ctx, *l, fn_name).is_some()
                    && extract_trig_arg(ctx, *inner, fn_name).is_some()
                {
                    return true;
                }
            }
            if let Expr::Neg(inner) = ctx.get(*l) {
                if extract_trig_arg(ctx, *r, fn_name).is_some()
                    && extract_trig_arg(ctx, *inner, fn_name).is_some()
                {
                    return true;
                }
            }
            false
        }
        Expr::Sub(l, r) => {
            extract_trig_arg(ctx, *l, fn_name).is_some()
                && extract_trig_arg(ctx, *r, fn_name).is_some()
        }
        _ => false,
    }
}

/// Check if expr is Add(trig(A), trig(B))
fn is_trig_sum(ctx: &cas_ast::Context, expr: ExprId, fn_name: &str) -> bool {
    if let Expr::Add(l, r) = ctx.get(expr) {
        return extract_trig_arg(ctx, *l, fn_name).is_some()
            && extract_trig_arg(ctx, *r, fn_name).is_some();
    }
    false
}

define_rule!(
    RecursiveTrigExpansionRule,
    "Recursive Trig Expansion",
    |ctx, expr, parent_ctx| {
        // GUARD 1: Skip if this trig function is marked for protection
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.is_sum_quotient_protected(expr) {
                return None;
            }
            // GUARD 1b: Skip if sin(4x) identity pattern detected
            // This prevents sin(4*t) from expanding before Sin4xIdentityZeroRule can fire
            if marks.has_sin4x_identity_pattern {
                return None;
            }
        }

        // GUARD 2: Skip if we're inside a potential sum-quotient pattern
        // This heuristic checks: if the trig function is inside a Div, and both
        // numerator and denominator are Add/Sub of trig functions, defer to
        // SinCosSumQuotientRule instead of expanding.
        if is_inside_trig_quotient_pattern(ctx, expr, parent_ctx) {
            return None;
        }

        let expr_data = ctx.get(expr).clone();
        if let Expr::Function(fn_id, args) = expr_data {
            let name = ctx.sym_name(*fn_id);
            if args.len() == 1 && (name == "sin" || name == "cos") {
                // Check for n * x where n is integer > 2
                let inner = args[0];
                let inner_data = ctx.get(inner).clone();

                let (n_val, x_val) = if let Expr::Mul(l, r) = inner_data {
                    if let Expr::Number(n) = ctx.get(l) {
                        if n.is_integer() {
                            (n.to_integer(), r)
                        } else {
                            return None;
                        }
                    } else if let Expr::Number(n) = ctx.get(r) {
                        if n.is_integer() {
                            (n.to_integer(), l)
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                } else {
                    return None;
                };

                if n_val > num_bigint::BigInt::from(2) && n_val <= num_bigint::BigInt::from(6) {
                    // GUARD: Only expand sin(n*x) for small n (3-6).
                    // For n > 6, the expansion grows exponentially without benefit.
                    // This prevents catastrophic expansion like sin(671*x) → 670 recursive steps.

                    // Rewrite sin(nx) -> sin((n-1)x + x)

                    let n_minus_1 = n_val.clone() - 1;
                    let n_minus_1_expr = ctx.add(Expr::Number(
                        num_rational::BigRational::from_integer(n_minus_1),
                    ));
                    let term_nm1 = smart_mul(ctx, n_minus_1_expr, x_val);

                    // sin(nx) = sin((n-1)x)cos(x) + cos((n-1)x)sin(x)
                    // cos(nx) = cos((n-1)x)cos(x) - sin((n-1)x)sin(x)

                    let sin_nm1 = ctx.call("sin", vec![term_nm1]);
                    let cos_nm1 = ctx.call("cos", vec![term_nm1]);
                    let sin_x = ctx.call("sin", vec![x_val]);
                    let cos_x = ctx.call("cos", vec![x_val]);

                    if name == "sin" {
                        let t1 = smart_mul(ctx, sin_nm1, cos_x);
                        let t2 = smart_mul(ctx, cos_nm1, sin_x);
                        let new_expr = ctx.add(Expr::Add(t1, t2));
                        return Some(
                            Rewrite::new(new_expr).desc(format!("sin({}x) expansion", n_val)),
                        );
                    } else {
                        // cos
                        let t1 = smart_mul(ctx, cos_nm1, cos_x);
                        let t2 = smart_mul(ctx, sin_nm1, sin_x);
                        let new_expr = ctx.add(Expr::Sub(t1, t2));
                        return Some(
                            Rewrite::new(new_expr).desc(format!("cos({}x) expansion", n_val)),
                        );
                    }
                }
            }
        }
        None
    }
);

define_rule!(
    CanonicalizeTrigSquareRule,
    "Canonicalize Trig Square",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        // cos^n(x) -> (1 - sin^2(x))^(n/2) for even n
        let expr_data = ctx.get(expr).clone();
        if let Expr::Pow(base, exp) = expr_data {
            let n_opt = if let Expr::Number(n) = ctx.get(exp) {
                Some(n.clone())
            } else {
                None
            };

            if let Some(n) = n_opt {
                if n.is_integer()
                    && n.to_integer() % 2 == 0.into()
                    && n > num_rational::BigRational::zero()
                {
                    // Limit power to avoid explosion? Let's say <= 4 for now.
                    if n <= num_rational::BigRational::from_integer(4.into()) {
                        if let Expr::Function(fn_id, args) = ctx.get(base) { let name = ctx.sym_name(*fn_id);
                            if name == "cos" && args.len() == 1 {
                                let arg = args[0];
                                // (1 - sin^2(x))^(n/2)
                                let one = ctx.num(1);
                                let sin_x = ctx.call("sin", vec![arg]);
                                let two = ctx.num(2);
                                let sin_sq = ctx.add(Expr::Pow(sin_x, two));
                                let base_term = ctx.add(Expr::Sub(one, sin_sq));

                                let half_n = n / num_rational::BigRational::from_integer(2.into());

                                if half_n.is_one() {
                                    return Some(Rewrite::new(base_term).desc("cos^2(x) -> 1 - sin^2(x)"));
                                } else {
                                    let half_n_expr = ctx.add(Expr::Number(half_n));
                                    let new_expr = ctx.add(Expr::Pow(base_term, half_n_expr));
                                    return Some(Rewrite::new(new_expr).desc("cos^2k(x) -> (1 - sin^2(x))^k"));
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }
);
