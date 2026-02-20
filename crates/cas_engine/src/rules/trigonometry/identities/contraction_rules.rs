//! Contraction rules for trigonometric expressions.
//!
//! These are the INVERSE of expansion rules — they contract expanded forms back
//! to compact representations (half-angle tangent, double angle contraction).

use crate::helpers::{as_div, as_mul, as_sub};
use crate::nary::Sign;
use crate::rule::Rewrite;
use cas_ast::{BuiltinFn, Expr, ExprId};
use cas_math::trig_contraction_support::{
    extract_coeff_trig_squared, extract_cos2_minus_sin2, extract_two_sin_cos,
    match_angle_diff_fraction, match_angle_sum_fraction,
};

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

        let tan_x = ctx.call_builtin(cas_ast::BuiltinFn::Tan, vec![x]);

        // Build cos(x) for the NonZero require
        let cos_x = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![x]);

        // Create rewrite with requires:
        // 1. Original denominator ≠ 0 (inherited from the division)
        // 2. cos(x) ≠ 0 (for tan(x) to be defined)
        let rewrite = Rewrite::new(tan_x)
            .desc(desc)
            .requires(ImplicitCondition::NonZero(denom_expr))
            .requires(ImplicitCondition::NonZero(cos_x));

        Some(rewrite)
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::DIV)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn soundness(&self) -> crate::rule::SoundnessLabel {
        crate::rule::SoundnessLabel::EquivalenceUnderIntroducedRequires
    }
}

// =============================================================================
// DOUBLE ANGLE CONTRACTION RULE
// 2·sin(t)·cos(t) → sin(2t), cos²(t) - sin²(t) → cos(2t)
// =============================================================================
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
            if let Some((sin_arg, cos_arg)) = extract_two_sin_cos(ctx, l, r) {
                // Check if sin and cos have the same argument
                if crate::ordering::compare_expr(ctx, sin_arg, cos_arg) == std::cmp::Ordering::Equal
                {
                    // Build sin(2*t)
                    let two = ctx.num(2);
                    let double_arg = ctx.add(Expr::Mul(two, sin_arg));
                    let sin_2t = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![double_arg]);
                    return Some(Rewrite::new(sin_2t).desc("2·sin(t)·cos(t) = sin(2t)"));
                }
            }
        }

        // Pattern 2: cos²(t) - sin²(t) → cos(2t)
        if let Some((l, r)) = as_sub(ctx, expr) {
            if let Some((cos_arg, sin_arg)) = extract_cos2_minus_sin2(ctx, l, r) {
                // Check if cos² and sin² have the same argument
                if crate::ordering::compare_expr(ctx, cos_arg, sin_arg) == std::cmp::Ordering::Equal
                {
                    // Build cos(2*t)
                    let two = ctx.num(2);
                    let double_arg = ctx.add(Expr::Mul(two, cos_arg));
                    let cos_2t = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![double_arg]);
                    return Some(Rewrite::new(cos_2t).desc("cos²(t) - sin²(t) = cos(2t)"));
                }
            }
        }

        None
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::MUL.union(crate::target_kind::TargetKindSet::SUB))
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn priority(&self) -> i32 {
        200 // Run before expansion rules to prevent ping-pong
    }
}

// =============================================================================
// Cos2xAdditiveContractionRule: 1 - 2·sin²(t) → cos(2t), 2·cos²(t) - 1 → cos(2t)
// =============================================================================
// These are alternate forms of the double-angle cosine identity that the
// existing DoubleAngleContractionRule does not handle (it only handles
// cos²(t) - sin²(t) → cos(2t)).
//
// Mathematical identities:
//   cos(2t) = 1 - 2·sin²(t)
//   cos(2t) = 2·cos²(t) - 1
//
// We scan additive leaves for a pair: constant ±1 and ∓2·trig²(t).

pub struct Cos2xAdditiveContractionRule;

impl crate::rule::Rule for Cos2xAdditiveContractionRule {
    fn name(&self) -> &str {
        "Cos 2x Additive Contraction"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Only apply to Add/Sub expressions
        if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
            return None;
        }

        // Use add_terms_signed to properly decompose both Add and Sub nodes.
        // add_leaves only flattens Add, so Sub(2cos²t, 1) would be a single
        // term and never match. add_terms_signed decomposes it into
        // [(2cos²t, +), (1, -)].
        let signed_terms = crate::nary::add_terms_signed(ctx, expr);

        // Only apply to exactly 2-term additive expressions to avoid
        // disrupting larger Pythagorean chains where the contraction would
        // prevent downstream symbolic cancellation.
        if signed_terms.len() != 2 {
            return None;
        }

        let one_rat = num_rational::BigRational::from_integer(1.into());
        let two_rat = num_rational::BigRational::from_integer(2.into());
        let neg_two_rat = num_rational::BigRational::from_integer((-2).into());

        // Find a constant ±1 term (sign is tracked by the signed_terms flag)
        for (i, &(term_i, sign_i)) in signed_terms.iter().enumerate() {
            let term_val = match ctx.get(term_i) {
                Expr::Number(n) => {
                    if sign_i == Sign::Pos {
                        n.clone()
                    } else {
                        -n.clone()
                    }
                }
                _ => continue,
            };

            // Check if this is +1 or -1 (after accounting for sign)
            let is_pos_one = term_val == one_rat;
            let is_neg_one = term_val == -one_rat.clone();
            if !is_pos_one && !is_neg_one {
                continue;
            }

            // Look for a matching ±2·trig²(t) in the remaining terms
            for (j, &(term_j, sign_j)) in signed_terms.iter().enumerate() {
                if j == i {
                    continue;
                }

                if let Some((trig_arg, trig_is_sin, mut coeff)) =
                    extract_coeff_trig_squared(ctx, term_j)
                {
                    // Account for the additive sign from the signed decomposition
                    if sign_j == Sign::Neg {
                        coeff = -coeff;
                    }

                    // Pattern A: 1 - 2·sin²(t) → cos(2t)
                    //   Requires: is_pos_one=true, trig_is_sin=true, coeff=-2
                    // Pattern B: 2·cos²(t) - 1 → cos(2t)
                    //   Requires: is_neg_one=true, trig_is_sin=false, coeff=+2
                    // Pattern C: -1 + 2·cos²(t) → cos(2t)
                    //   Requires: is_neg_one=true, trig_is_sin=false, coeff=+2
                    // Pattern D: -2·sin²(t) + 1 → cos(2t)
                    //   Same as A with different ordering

                    let matches = (is_pos_one && trig_is_sin && coeff == neg_two_rat)
                        || (is_neg_one && !trig_is_sin && coeff == two_rat);

                    if !matches {
                        continue;
                    }

                    // Build cos(2t)
                    let two = ctx.num(2);
                    let double_arg = ctx.add(Expr::Mul(two, trig_arg));
                    let cos_2t = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![double_arg]);

                    let desc = if trig_is_sin {
                        "1 - 2·sin²(t) = cos(2t)"
                    } else {
                        "2·cos²(t) - 1 = cos(2t)"
                    };

                    return Some(Rewrite::new(cos_2t).desc(desc));
                }
            }
        }

        None
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        // POST only: prevents oscillation with HalfAngleExpansion and other trig
        // rules during TRANSFORM. In POST no expansion rules fire, so contraction
        // is stable and normalises 2cos²(t)-1 / 1-2sin²(t) to cos(2t).
        crate::phase::PhaseMask::POST
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::ADD.union(crate::target_kind::TargetKindSet::SUB))
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn priority(&self) -> i32 {
        200
    }
}

// =============================================================================
// AngleSumFractionToTanRule
// (sin(a)cos(b) + cos(a)sin(b)) / (cos(a)cos(b) - sin(a)sin(b)) → tan(a+b)
// (sin(a)cos(b) - cos(a)sin(b)) / (cos(a)cos(b) + sin(a)sin(b)) → tan(a-b)
// =============================================================================
// This rule contracts expanded angle-addition fractions back to tan.
// It targets Div nodes only, so it cannot loop with AngleIdentityRule
// (which targets Function nodes with sin/cos of Add/Sub arguments).
//
// Also handles the case where the numerator/denominator has extra common
// factors (e.g., multiplied through by cos²), by first trying the
// bare 2-term pattern.

pub struct AngleSumFractionToTanRule;

impl crate::rule::Rule for AngleSumFractionToTanRule {
    fn name(&self) -> &str {
        "Angle Sum Fraction to Tan"
    }

    fn priority(&self) -> i32 {
        190 // Below contraction rules at 200, above normal
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Only match Div nodes
        let (num, den) = as_div(ctx, expr)?;

        // Try angle-sum pattern: numerator = sin(a)cos(b) + cos(a)sin(b)
        //                        denominator = cos(a)cos(b) - sin(a)sin(b)
        if let Some((a, b)) = match_angle_sum_fraction(ctx, num, den) {
            let sum_arg = ctx.add(Expr::Add(a, b));
            let tan_result = ctx.call_builtin(BuiltinFn::Tan, vec![sum_arg]);
            return Some(
                Rewrite::new(tan_result)
                    .desc("(sin(a)cos(b)+cos(a)sin(b))/(cos(a)cos(b)-sin(a)sin(b)) = tan(a+b)"),
            );
        }

        // Try angle-difference pattern: numerator = sin(a)cos(b) - cos(a)sin(b)
        //                               denominator = cos(a)cos(b) + sin(a)sin(b)
        if let Some((a, b)) = match_angle_diff_fraction(ctx, num, den) {
            let diff_arg = ctx.add(Expr::Sub(a, b));
            let tan_result = ctx.call_builtin(BuiltinFn::Tan, vec![diff_arg]);
            return Some(
                Rewrite::new(tan_result)
                    .desc("(sin(a)cos(b)-cos(a)sin(b))/(cos(a)cos(b)+sin(a)sin(b)) = tan(a-b)"),
            );
        }

        None
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::DIV)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}
